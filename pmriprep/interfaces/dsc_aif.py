"""AIF (Arterial Input Function) detection interfaces for DSC-MRI.

This module implements Nipype interfaces for AIF detection in DSC perfusion
processing, with multi-level detection strategies (atlas-based, relaxed
criteria, synthetic fallback) and comprehensive quality metrics.

Implements search-mask preparation, adaptive-percentile candidate pre-filtering,
SSS/edge spatial exclusions, tissue-TTP cross-validation, cost-function
scoring, 3x3x3 ROI averaging, and Savitzky-Golay smoothing.
"""

import math
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_erosion, generate_binary_structure  # type: ignore[import-untyped]
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    isdefined,
    traits,
)

from pmriprep.utils.dsc_constants import (
    AIF_PEAK_CEILING_3T,
    AIF_PEAK_FLOOR_3T,
    AIF_PEAK_SPREAD_MAX_SECONDS,
    AIF_TAIL_FRACTION_MAX,
    AIF_TAIL_WINDOW_S,
    AIF_TTP_NEAR_TISSUE_S,
    AIF_TTP_REJECT_AFTER_TISSUE_S,
    EPSILON_CONCENTRATION,
    EPSILON_SIGNAL,
    GD_DOTA_STANDARD_DOSE,
    LESION_EXCLUSION_DISTANCE_MM,
    LESION_EXCLUSION_FLOOR_MM,
    SYNTH_AIF_BASE_PEAK_3T,
    SYNTH_AIF_DOSE_CAP,
    SYNTH_AIF_MIN_PEAK,
    SYNTH_AIF_RECIRC_ALPHA,
    SYNTH_AIF_RECIRC_DELAY_S,
    SYNTH_AIF_RECIRC_DISPERSION_S,
    SYNTH_AIF_RECIRC_FRACTION,
)

# Seconds after baseline end before first-pass window begins.
# 3 s is short enough to capture early peaks in fast-TR acquisitions while
# candidates with spurious noise peaks are already rejected by the TTP
# minimum check in ``_score_candidate_cost`` (strict ≥ 5 s, relaxed ≥ 3 s).
_FP_GAP_SECONDS: float = 3.0

# Minimum number of AIF candidates before accepting a cascade level.
# If a level finds fewer than this, the cascade continues to more relaxed
# criteria to build a larger candidate pool (quality gating is downstream).
MIN_AIF_CANDIDATES_ACCEPT: int = 3

#: Gamma peak ratio band for cascade quality gate.
#: If averaged AIF peak ratio falls outside this range, continue cascading
#: to the next tier (keeping the current result as fallback).
#: Physics: >12% AIF peak error exceeds the ~10% CBF CoV expected in
#: normal brain.
GAMMA_PEAK_RATIO_CASCADE_MIN: float = 0.88
GAMMA_PEAK_RATIO_CASCADE_MAX: float = 1.12

#: Gamma R² floor for synthetic fallback.
#: R² < 0 means gamma fit is worse than mean — curve has no bolus shape.
#: A true arterial bolus will never fit worse than a horizontal line.
GAMMA_R2_SYNTHETIC_FALLBACK: float = 0.0

#: Minimum baseline-return ratio (%) for an accepted averaged AIF.
#: BRR = mean(tail) / peak × 100. True arterial washout produces +5 to +30%;
#: slightly negative (≥-2%) is within T1-recirculation noise.  Below -3%
#: indicates T1-leakage dominance or motion breaking the mono-exponential
#: ΔR2* assumption. Tighter than the per-candidate permissive floor
#: (-10%) because averaging should reduce noise, not let bad voxels
#: dominate.
AIF_AVG_BRR_MIN: float = -3.0

#: S0 ratio floor: reject candidates with baseline signal below this
#: fraction of median brain S0.  Below 0.10, baseline SNR < 5 and the
#: logarithmic ΔR2* conversion is numerically unstable.
S0_RATIO_FLOOR: float = 0.10

# Fraction of user-configured n_candidates to keep at each criteria level.
# As criteria relax, candidate quality drops — fewer candidates concentrates
# on the best-scoring voxels from a noisier pool.
_CASCADE_N_CANDIDATES_FRACTION: dict[str, float] = {
    "strict": 1.0,
    "relaxed": 0.6,
    "permissive": 0.4,
}


class AifQualityResult(NamedTuple):
    """Return type for ``_aif_passes_quality``."""

    passes: bool
    spr: float
    peak_ratio: Optional[float]
    brr: float


class AIFPoolAcceptanceResult(NamedTuple):
    """Accepted inverse-cost-weighted AIF candidate pool."""

    curve: npt.NDArray[Any]
    mask: npt.NDArray[Any]
    roi_size: int
    spr: float
    peak_ratio: Optional[float]
    brr: float


_AIFCandidate = tuple[float, tuple[int, int, int], dict[str, Any]]
_AcceptedAIF = tuple[npt.NDArray[Any], npt.NDArray[Any], int]


class _TrimmedPoolAcceptanceResult(NamedTuple):
    """Accepted averaged-AIF result and the candidate pool that produced it."""

    result: AIFPoolAcceptanceResult
    candidates: list[_AIFCandidate]


class _SingleBestAIFResult(NamedTuple):
    """Accepted single-candidate fallback curve, mask, and ROI size."""

    curve: npt.NDArray[Any]
    mask: npt.NDArray[Any]
    roi_size: int
    candidate: _AIFCandidate


class _TierCandidateResult(NamedTuple):
    """Candidate survivors and diagnostics produced by one cascade tier."""

    candidates: list[_AIFCandidate]
    n_available: int
    filter_metadata: dict[str, Any]
    postfilter_diagnostics: Optional[dict[str, Any]]


class _CascadeUnionState(NamedTuple):
    """Mutable collections used to accumulate candidates across tiers."""

    candidates: list[_AIFCandidate]
    coordinates: set[tuple[int, int, int]]
    tier_provenance: dict[str, int]
    coordinate_origins: dict[tuple[int, int, int], str]
    filter_metadata: dict[str, Any]


class _UnionAcceptanceResult(NamedTuple):
    """Accepted union pool with its ordered provenance."""

    accepted_aif: _AcceptedAIF
    candidates: list[_AIFCandidate]
    detection_level: str
    tier_provenance: dict[str, int]


class _CascadeDetectionResult(NamedTuple):
    """Candidate cascade outcome consumed by final AIF construction."""

    accepted_aif: Optional[_AcceptedAIF]
    candidates: list[_AIFCandidate]
    n_available: int
    detection_level: str
    filter_metadata: dict[str, Any]
    initial_level: Optional[str]
    tiers_attempted: list[str]
    tier_provenance: dict[str, int]
    postfilter_diagnostics: dict[str, dict[str, Any]]
    spatial_filter_diagnostics: dict[str, dict[str, Any]]


class _FinalizedDetectedAIF(NamedTuple):
    """Detected AIF arrays, QC state, and updated baseline."""

    curve: npt.NDArray[Any]
    mask: npt.NDArray[Any]
    quality_score: float
    qc_metrics: dict[str, Any]
    baseline_frames: int


class _FinalAIFState(NamedTuple):
    """Final DSC AIF arrays and metadata ready for serialization."""

    curve: npt.NDArray[Any]
    mask: npt.NDArray[Any]
    measured_curve: npt.NDArray[Any]
    raw_curve: Optional[npt.NDArray[Any]]
    gamma_curve: Optional[npt.NDArray[Any]]
    affine: npt.NDArray[Any]
    detection_level: str
    quality_score: float
    gamma_fit_applied: bool
    baseline_frames: int
    candidate_count: int
    qc_metrics: dict[str, Any]


class _DSCAIFDetectionContext(NamedTuple):
    """Validated image data, acquisition state, and detection parameters."""

    concentration: npt.NDArray[Any]
    affine: npt.NDArray[Any]
    brain_mask: npt.NDArray[Any]
    atlas_mask: Optional[npt.NDArray[Any]]
    atlas_is_mra: bool
    full_atlas_mask: Optional[npt.NDArray[Any]]
    lesion_mask: Optional[npt.NDArray[Any]]
    enhancing_mask: Optional[npt.NDArray[Any]]
    signal: Optional[npt.NDArray[Any]]
    aif_method: str
    initial_criteria: str
    n_candidates: int
    baseline_frames: int
    gamma_fit_mode: str
    disable_fallback: bool
    lesion_distance: float
    n_timepoints: int
    field_strength: float
    synthetic_effective_dose: float
    voxel_sizes: tuple[float, float, float]
    tr: float


_CascadeAttempt = tuple[str, Optional[npt.NDArray[Any]], str, bool]


# ---------------------------------------------------------------------------
# Top-level helper functions (backward-compatible with utils/dsc_aif.py)
# ---------------------------------------------------------------------------


#: QC fields populated from a real-AIF candidate that become stale when
#: the cascade falls back to synthetic (either at no-candidates or
#: gamma-R² rejection). Cleared by :func:`_clear_stale_real_aif_keys`
#: so ``detection_method=='synthetic'`` cannot coexist with stale
#: spatial data (e.g. ``n_aif_voxels_selected=0`` alongside a real
#: ``aif_voxel_coord`` from a since-rejected candidate).
_STALE_REAL_AIF_KEYS: Final[tuple[str, ...]] = (
    "aif_voxel_coord", "aif_centroid_mm", "aif_hemisphere",
    "candidate_coords", "candidate_scores",
    "n_left_candidates", "n_right_candidates",
    "hemisphere_coverage", "is_bilateral",
    "aif_slice", "aif_edge_slice_warning",
    "best_cost", "mean_cost",
    "cascade_accepted_roi_size",
    "aif_metrics",
    "aif_tier_provenance",
)


def _clear_stale_real_aif_keys(qc_metrics: dict[str, Any]) -> None:
    """Pop real-AIF QC fields that would be stale under synthetic fallback."""
    for k in _STALE_REAL_AIF_KEYS:
        qc_metrics.pop(k, None)


def _stamp_synthetic_fallback_common(
    qc_metrics: dict[str, Any],
    *,
    reason: str,
    tiers_attempted: Sequence[str],
) -> None:
    """Set core synthetic-fallback QC fields consistently across all paths.

    Parameters
    ----------
    qc_metrics : dict
        Mutable QC dict to stamp (fresh or prepopulated).
    reason : str
        Fallback reason (``"no_candidates"`` or ``"gamma_r2_floor"``).
    tiers_attempted : Sequence[str]
        Cascade tiers tried before fallback.  Only written when the
        key is absent (Path B inherits from the real-AIF run).

    Notes
    -----
    ``detection_method`` is intentionally NOT set here — callers set it
    via the shared ``detection_level`` variable.  Fields handled by the
    common block downstream (``synthetic_aif_warnings``,
    ``cbv_auc_denominator_is_synthetic``) are also omitted; they key on
    ``detection_level == "synthetic"`` and fire for any synthetic path.
    """
    qc_metrics["synthetic"] = True
    qc_metrics["n_aif_voxels_selected"] = 0
    qc_metrics["n_aif_voxels_available"] = 0
    qc_metrics["synthetic_fallback_reason"] = reason
    qc_metrics["aif_quality"] = "degraded"
    if "cascade_tiers_attempted" not in qc_metrics:
        qc_metrics["cascade_tiers_attempted"] = list(tiers_attempted)


def _flag_retained_below_floor(
    qc_metrics: dict[str, Any],
    *,
    disable_fallback: bool,
    detection_level: str,
    gamma_fit_applied: bool,
    floor: float = GAMMA_R2_SYNTHETIC_FALLBACK,
) -> None:
    """Flag a real AIF retained below the gamma-fit floor when fallback is disabled.

    With synthetic fallback enabled, a sub-floor gamma fit (R² < ``floor`` = fits
    worse than a flat line = no bolus shape) triggers a synthetic swap. With
    ``disable_aif_fallback`` the detected AIF is retained instead; this annotates
    that case so it is not silent — the AIF itself is NOT modified, only flagged.

    No-ops unless fallback is disabled, a real (non-synthetic) AIF was detected,
    the gamma fit converged but was not applied, and ``gamma_fit_r_squared`` is
    present and below the floor. Optimizer failures are reported separately and
    do not establish that the measured AIF is inadequate.
    """
    if gamma_fit_applied or not disable_fallback or detection_level == "synthetic":
        return
    if qc_metrics.get("gamma_fit_converged") is not True:
        return
    gamma_r2 = qc_metrics.get("gamma_fit_r_squared")
    if gamma_r2 is None or gamma_r2 >= floor:
        return
    import logging

    qc_metrics["aif_gamma_fit_below_floor_retained"] = True
    qc_metrics["gamma_r2_retained"] = gamma_r2
    logging.getLogger(__name__).warning(
        "Gamma fit R²=%.3f < %.1f but synthetic AIF fallback is disabled; retaining "
        "the detected AIF. Deconvolution maps (CBF, MTT, Tmax) may be unreliable; "
        "AUC-nCBV and PSR remain interpretable.",
        gamma_r2, floor,
    )


def _costs_to_scores(costs: list[float]) -> list[float]:
    """Convert DSC AIF candidate costs (lower=better) to scores (higher=better)."""
    if not costs:
        return []
    min_cost = min(costs)
    max_cost = max(costs)
    if max_cost == min_cost:
        return [1.0] * len(costs)
    return [
        float(np.clip(1.0 - (c - min_cost) / (max_cost - min_cost), 0.0, 1.0))
        for c in costs
    ]


def compute_aif_cnr(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute contrast-to-noise ratio for an AIF curve.

    CNR = (peak - baseline_mean) / baseline_std

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).

    Returns
    -------
    float
        Contrast-to-noise ratio. Higher is better (>4.0 typically good).
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0.0

    baseline = aif_curve[:baseline_frames]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))

    if baseline_std < EPSILON_CONCENTRATION:
        return 100.0

    peak = float(np.max(aif_curve))
    cnr = (peak - baseline_mean) / baseline_std

    return max(0.0, cnr)


def compute_aif_washout_fraction(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
    tail_frames: int = 5,
) -> float:
    """Compute washout fraction for an AIF curve.

    Washout fraction = (peak - tail_mean) / (peak - baseline_mean)
    A value of 1.0 indicates complete washout; 0.0 indicates no washout.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).
    tail_frames : int
        Number of tail frames to average (default: 5).

    Returns
    -------
    float
        Washout fraction (0-1). Arterial curves typically show >0.2 washout.
    """
    if len(aif_curve) < baseline_frames + tail_frames + 1:
        return 0.0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    tail_mean = float(np.mean(aif_curve[-tail_frames:]))
    peak = float(np.max(aif_curve))

    denom = peak - baseline_mean
    if denom < EPSILON_CONCENTRATION:
        return 0.0

    washout = (peak - tail_mean) / denom

    return max(0.0, min(1.0, washout))


def _compute_aif_peak_ratio(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int,
    tr: float,
) -> Optional[float]:
    """Compute gamma peak ratio for an averaged AIF curve.

    Fits a gamma-variate model and returns peak_fit / peak_orig.
    Used by the cascade quality gate to assess AIF shape fidelity.

    Parameters
    ----------
    aif_curve : np.ndarray
        Averaged AIF concentration-time curve.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    tr : float
        Repetition time in seconds.

    Returns
    -------
    float or None
        Peak ratio (fitted/measured), or None if gamma fit fails.
    """
    from pmriprep.utils.dsc_aif import fit_gamma_variate

    result = fit_gamma_variate(aif_curve, baseline_frames=baseline_frames, tr=tr)
    if not result.fit_converged or result.r_squared < 0.5:
        return None
    peak_orig = float(np.max(aif_curve[baseline_frames:]))
    peak_fit = float(np.max(result.fitted_curve[baseline_frames:]))
    if peak_orig < 1e-6:
        return None
    return peak_fit / peak_orig


def compute_baseline_stability(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute baseline stability as coefficient of variation.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).

    Returns
    -------
    float
        Coefficient of variation (std/mean) of baseline. Lower is better (<0.1 good).
    """
    if len(aif_curve) < baseline_frames:
        return 1.0

    baseline = aif_curve[:baseline_frames]
    baseline_mean = np.abs(np.mean(baseline))
    baseline_std = float(np.std(baseline))

    if baseline_mean < EPSILON_CONCENTRATION:
        return baseline_std / EPSILON_CONCENTRATION

    return baseline_std / float(baseline_mean)


# ---------------------------------------------------------------------------
# Nipype interface specs
# ---------------------------------------------------------------------------

class AIFDetectionInputSpec(BaseInterfaceInputSpec):
    """Input specification for AIFDetection interface."""

    concentration_file = File(
        exists=True,
        mandatory=True,
        desc="4D NIfTI concentration curves (output from signal_to_concentration)",
    )
    brain_mask_file = File(
        exists=True,
        mandatory=True,
        desc="3D brain mask NIfTI",
    )
    atlas_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 3D arterial atlas mask in DSC space (MCA/ACA)",
    )
    full_atlas_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 3D full arterial atlas mask in DSC space (incl. posterior circulation)",
    )
    aif_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional user-supplied AIF search mask (overrides atlas mask)",
    )
    lesion_mask_file = traits.Either(
        File(exists=True),
        traits.Enum(""),
        mandatory=False,
        desc="Optional 3D lesion mask for exclusion",
    )
    enhancing_mask_file = traits.Either(
        File(exists=True),
        traits.Enum(""),
        mandatory=False,
        desc="Optional 3D enhancing tumor mask for laterality determination",
    )

    signal_file = File(
        exists=True,
        mandatory=False,
        desc="4D raw signal for S0-based partial volume rejection",
    )
    external_aif_file = File(
        exists=True,
        mandatory=False,
        desc="External AIF curve file (.txt, .npz, or .npy). "
             "Values must be ΔR2* concentration in s⁻¹. "
             "Bypasses all detection when provided.",
    )

    # Detection parameters
    aif_method = traits.Enum(
        "auto",
        "synthetic",
        usedefault=True,
        desc="AIF detection method: 'auto' (multi-level) or 'synthetic' (gamma-variate)",
    )
    criteria = traits.Enum(
        "strict",
        "relaxed",
        usedefault=True,
        desc="AIF quality criteria: 'strict' or 'relaxed'",
    )
    n_candidates = traits.Int(
        10,
        usedefault=True,
        desc="Number of AIF candidate voxels to average",
    )
    lesion_distance = traits.Float(
        LESION_EXCLUSION_DISTANCE_MM,
        usedefault=True,
        desc="Minimum distance (mm) from lesion for AIF candidates",
    )
    disable_fallback = traits.Bool(
        False,
        usedefault=True,
        desc="Disable synthetic AIF fallback when detection fails",
    )
    strict_roi_constraint = traits.Bool(
        False,
        usedefault=True,
        desc="If True, do NOT fall back to whole brain when atlas/ROI mask is provided",
    )
    baseline_frames = traits.Int(
        5,
        usedefault=True,
        desc="Number of pre-bolus baseline frames",
    )
    tr = traits.Float(
        0.0,
        usedefault=True,
        desc="Repetition time in seconds (0 = auto-detect from NIfTI header)",
    )
    dose = traits.Float(
        1.0,
        usedefault=True,
        desc="Contrast agent dose multiplier (0.5=half, 1.0=full, 2.0=double)",
    )
    agent_standard_dose = traits.Float(
        0.1,
        usedefault=True,
        desc="Agent's recommended molecular dose in mmol/kg "
             "(0.1=standard, 0.05=gadopiclenol, 0.04=gadoquatrane)",
    )
    gd_multiplicity = traits.Int(
        1,
        usedefault=True,
        desc="Gd3+ ions per molecule (1=monomeric, 4=gadoquatrane). "
             "Affects DSC susceptibility scaling for AIF peak thresholds.",
    )
    field_strength = traits.Float(
        3.0,
        usedefault=True,
        desc="Magnetic field strength in Tesla (from BIDS MagneticFieldStrength). "
             "Used for B0 scaling of synthetic AIF peak amplitude.",
    )

    # Output file
    out_aif_curve = File(
        genfile=True,
        desc="Output AIF curve as 1D text file",
    )
    out_aif_mask = File(
        genfile=True,
        desc="Output AIF voxel mask NIfTI",
    )


class AIFDetectionOutputSpec(TraitedSpec):
    """Output specification for AIFDetection interface."""

    out_aif_curve = File(exists=True, desc="AIF curve as 1D text file")
    out_aif_mask = File(exists=True, desc="AIF voxel mask NIfTI")
    out_raw_aif_curve = traits.Str(
        desc="Pre-gamma-fit raw AIF curve text file (empty if no gamma fit)",
    )
    out_gamma_aif_curve = File(
        desc="Gamma-fitted AIF curve text file (always produced)",
    )
    out_measured_aif_curve = File(
        desc="Measured AIF curve text file (pre-gamma; always produced). "
             "Consumed by AUC CBV denominator. For external/synthetic "
             "AIF paths, contains the external/synthetic curve itself.",
    )
    out_aif_coord = traits.List(traits.Int, desc="Best AIF voxel coordinate [x, y, z]")
    aif_curve = traits.List(traits.Float, desc="AIF concentration curve values")
    detection_level = traits.Str(
        desc="Detection level: 'atlas_strict', 'atlas_relaxed', 'atlas_permissive', "
             "'brain_relaxed', 'brain_permissive', 'synthetic', or 'external'"
    )
    quality_score = traits.Float(desc="Overall AIF quality score (0-1)")
    candidate_count = traits.Int(desc="Number of candidate voxels evaluated")
    gamma_fit_applied = traits.Bool(desc="Whether gamma-variate fitting was applied")
    aif_baseline_frames = traits.Int(
        desc="Detected pre-bolus baseline frames for downstream use",
    )
    qc_metrics = traits.Dict(desc="Detailed QC metrics dictionary")


class AIFDetection(BaseInterface):
    """Arterial Input Function (AIF) detection for DSC-MRI.

    Implements the AIF detection pipeline with:
    - Brain mask erosion (4 iterations) to exclude cortical veins
    - Adaptive percentile candidate pre-filtering (99.5% -> 95%)
    - SSS / edge-slice spatial exclusions
    - Tissue TTP cross-validation
    - Cost-function scoring (lower = better) with S0-dominant weighting
    - 3x3x3 ROI averaging with Savitzky-Golay smoothing
    - Multi-level fallback: atlas strict/relaxed/permissive -> brain relaxed/permissive -> synthetic
    """

    input_spec = AIFDetectionInputSpec
    output_spec = AIFDetectionOutputSpec

    # -----------------------------------------------------------------
    # Filename helpers
    # -----------------------------------------------------------------

    def _gen_outfilename(self, suffix: str = "_aif_curve.txt") -> str:
        """Generate output filename based on input."""
        in_file = Path(self.inputs.concentration_file)
        if in_file.name.endswith(".nii.gz"):
            base = in_file.name[:-7]
        elif in_file.name.endswith(".nii"):
            base = in_file.stem
        else:
            base = in_file.stem
        return str(Path.cwd() / f"{base}{suffix}")

    # -----------------------------------------------------------------
    # Step 2: Search mask preparation
    # -----------------------------------------------------------------

    @staticmethod
    def _prepare_search_mask(
        brain_mask: npt.NDArray[Any],
        roi_mask: Optional[npt.NDArray[Any]],
        lesion_mask: Optional[npt.NDArray[Any]],
        erosion_iterations: int = 4,
    ) -> npt.NDArray[Any]:
        """Prepare binary search mask with erosion and exclusion.

        Erosion for cortical vein exclusion, optional ROI intersection,
        and binary lesion exclusion.  Distance-based lesion filtering
        happens *after* scoring (see ``_apply_post_scoring_filters``).

        Parameters
        ----------
        brain_mask : ndarray
            3D binary brain mask.
        roi_mask : ndarray or None
            Optional 3D ROI (atlas) mask to intersect.
        lesion_mask : ndarray or None
            Optional 3D lesion mask to exclude.
        erosion_iterations : int
            Number of binary erosion iterations (default 4).

        Returns
        -------
        ndarray
            3D boolean search mask.
        """
        # In-plane-only erosion: preserves all slices while excluding
        # cortical surface voxels. 3D erosion with 4 iterations removes
        # ~4 voxels from all faces including z, which wipes out skull-base
        # slices where the best AIF candidates (ICA, MCA M1) reside.
        # Input orientation is RAS, so z is the slice direction (axis 2).
        struct_inplane = generate_binary_structure(3, 1)  # 6-connected
        struct_inplane[:, :, 0] = 0  # Remove z-minus connectivity
        struct_inplane[:, :, 2] = 0  # Remove z-plus connectivity
        eroded = binary_erosion(
            brain_mask > 0.5,
            structure=struct_inplane,
            iterations=erosion_iterations,
        ).astype(bool)

        if roi_mask is not None:
            search = eroded & (roi_mask > 0.5)
            if not np.any(search):
                # Fallback: uneroded intersection
                search = (brain_mask > 0.5) & (roi_mask > 0.5)
        else:
            search = eroded

        # Preserve a non-empty search domain when erosion removes all voxels.
        if not np.any(search):
            search = (brain_mask > 0.5).copy()
            if roi_mask is not None:
                search &= roi_mask > 0.5

        if lesion_mask is not None and np.any(lesion_mask > 0.5):
            lesion_bool = lesion_mask > 0.5
            pre_lesion = search.copy()
            search = search & ~lesion_bool
            if not np.any(search):
                search = pre_lesion

        return search  # type: ignore[no-any-return]

    # -----------------------------------------------------------------
    # Step 2b: World-coordinate hemisphere methods
    # -----------------------------------------------------------------

    @staticmethod
    def _determine_tumor_laterality(
        lesion_mask: npt.NDArray[Any],
        affine: npt.NDArray[Any],
        midline_tolerance_mm: float = 10.0,
    ) -> Optional[str]:
        """Determine tumor hemisphere using world (RAS) coordinates.

        Parameters
        ----------
        lesion_mask : ndarray
            3D lesion mask (values > 0.5 = lesion).
        affine : ndarray
            4x4 affine transformation matrix (voxel -> RAS world).
        midline_tolerance_mm : float
            Distance from midline (x=0) for bilateral classification.

        Returns
        -------
        str or None
            ``'left'``, ``'right'``, ``'bilateral'``, or ``None`` if
            the mask is empty.
        """
        coords = np.argwhere(lesion_mask > 0.5)  # (N, 3) voxel coords
        if len(coords) == 0:
            return None

        # Center of mass in voxel space
        com_voxel = coords.mean(axis=0)

        # Convert to world coordinates using affine
        com_world = affine @ np.append(com_voxel, 1.0)
        x_mm = float(com_world[0])

        # Convert all lesion voxels to world for bilateral check
        coords_homog = np.column_stack([coords, np.ones(len(coords))])
        coords_world = (affine @ coords_homog.T).T
        x_coords_mm = coords_world[:, 0]

        x_min, x_max = float(x_coords_mm.min()), float(x_coords_mm.max())

        # Bilateral if tumor significantly spans midline on both sides
        if x_min < -midline_tolerance_mm and x_max > midline_tolerance_mm:
            return "bilateral"

        # RAS convention: x < 0 = left hemisphere
        return "left" if x_mm < 0 else "right"

    @staticmethod
    def _count_candidates_by_hemisphere(
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        affine: npt.NDArray[Any],
        tumor_laterality: str,
    ) -> tuple[int, int]:
        """Count candidate laterality without changing selection scores.

        Parameters
        ----------
        top_candidates : list
            List of ``(cost, (x,y,z), metrics)`` tuples.
        affine : ndarray
            4x4 affine transformation matrix (voxel -> RAS world).
        tumor_laterality : str
            ``'left'``, ``'right'``, or ``'bilateral'``.

        Returns
        -------
        tuple of (n_contra, n_ipsi)
            n_contra : count of contralateral candidates.
            n_ipsi : count of ipsilateral candidates.
        """
        if tumor_laterality == "bilateral":
            return 0, 0

        contra_side = "right" if tumor_laterality == "left" else "left"

        n_contra = 0
        n_ipsi = 0

        for _, coord, _ in top_candidates:
            coord_homog = np.array(
                [coord[0], coord[1], coord[2], 1.0]
            )
            world = affine @ coord_homog
            cand_side = "left" if world[0] < 0 else "right"

            if cand_side == contra_side:
                n_contra += 1
            else:
                n_ipsi += 1

        return n_contra, n_ipsi

    # -----------------------------------------------------------------
    # Step 3: Spatial exclusions
    # -----------------------------------------------------------------

    @staticmethod
    def _apply_spatial_exclusions(
        candidate_indices: npt.NDArray[Any],
        brain_shape: tuple[int, ...],
        *,
        skip_sss: bool = False,
    ) -> npt.NDArray[Any]:
        """Exclude SSS (midline + superior/posterior) and edge-slice voxels.

        Parameters
        ----------
        candidate_indices : ndarray
            Nx3 array of voxel coordinates.
        brain_shape : tuple
            (nx, ny, nz) spatial dimensions.
        skip_sss : bool
            Skip SSS exclusion (atlas-constrained search already excludes
            venous structures). Edge-slice exclusion still applies.

        Returns
        -------
        ndarray
            Filtered Nx3 array.
        """
        if len(candidate_indices) == 0:
            return candidate_indices

        nx, ny, nz = brain_shape[:3]
        kept: list[npt.NDArray[Any]] = list(candidate_indices)

        if not skip_sss:
            x_center = nx // 2
            x_exclude = int(nx * 0.1)  # Central 20% (+/-10%)
            z_superior = int(nz * 0.65)  # Superior 35%
            y_posterior = int(ny * 0.45)  # Posterior 45%

            filtered = []
            for idx in kept:
                x, y, z = int(idx[0]), int(idx[1]), int(idx[2])
                is_midline = (x_center - x_exclude) <= x <= (x_center + x_exclude)
                if is_midline and (z > z_superior or y < y_posterior):
                    continue
                filtered.append(idx)

            if len(filtered) > 0:
                kept = filtered

        # Edge slice exclusion (z == 0 or z == nz-1)
        non_edge = [idx for idx in kept if 0 < int(idx[2]) < nz - 1]
        if len(non_edge) > 0:
            kept = non_edge

        return np.array(kept)

    # -----------------------------------------------------------------
    # Step 4: Percentile candidate finder
    # -----------------------------------------------------------------

    def _find_aif_candidate_voxels(
        self,
        conc_data: npt.NDArray[Any],
        search_mask: npt.NDArray[Any],
        baseline_frames: int,
        *,
        skip_sss: bool = False,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Find initial candidate voxels via percentile thresholding.

        Collects all voxels in the top 10% of peak concentration within
        *search_mask*.  Downstream scoring handles ranking; a broad pool
        prevents valid arterial candidates (e.g. pericallosal artery at
        P95) from being excluded when larger basal vessels dominate the
        top percentile.

        Parameters
        ----------
        conc_data : ndarray
            4D concentration (x, y, z, t).
        search_mask : ndarray
            3D boolean mask.
        baseline_frames : int
            Number of pre-bolus frames.
        skip_sss : bool
            Skip SSS spatial exclusion (for atlas-constrained search).

        Returns
        -------
        candidate_indices : ndarray
            Nx3 array of voxel coordinates.
        peak_conc_map : ndarray
            3D peak concentration map.
        """
        masked = conc_data * search_mask[:, :, :, np.newaxis]
        masked[masked < 0] = 0
        peak_conc_map = np.max(masked[..., baseline_frames:], axis=-1)
        peak_conc_map[~search_mask] = -np.inf

        if not np.any(np.isfinite(peak_conc_map)):
            return np.empty((0, 3), dtype=int), peak_conc_map

        valid_peaks = peak_conc_map[search_mask]
        valid_peaks = valid_peaks[np.isfinite(valid_peaks)]
        if len(valid_peaks) == 0:
            return np.empty((0, 3), dtype=int), peak_conc_map

        thresh = float(np.nanpercentile(valid_peaks, 90.0))
        candidates = np.argwhere(peak_conc_map >= thresh)

        if candidates.shape[0] == 0:
            return np.empty((0, 3), dtype=int), peak_conc_map

        # Apply spatial exclusions
        candidates = self._apply_spatial_exclusions(
            candidates, conc_data.shape[:3], skip_sss=skip_sss,
        )

        return candidates, peak_conc_map

    # -----------------------------------------------------------------
    # Step 5: Tissue TTP reference
    # -----------------------------------------------------------------

    @staticmethod
    def _calculate_tissue_ttp_reference(
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        baseline_frames: int,
    ) -> float:
        """Compute median tissue TTP (in frames) for cross-validation.

        Uses moderate-concentration voxels (25th-75th percentile) to
        represent brain parenchyma. Arterial blood should arrive BEFORE
        tissue; venous structures peak AFTER tissue.

        Returns
        -------
        float
            Median tissue TTP in frame units.
        """
        peak_time_map = (
            np.argmax(conc_data[..., baseline_frames:], axis=-1)
            + baseline_frames
        )
        peak_conc_all = np.max(conc_data[..., baseline_frames:], axis=-1)

        brain_peaks = peak_conc_all[brain_mask > 0.5]
        positive_peaks = brain_peaks[brain_peaks > 0]
        if len(positive_peaks) < 100:
            return float(np.median(peak_time_map[brain_mask > 0.5]))

        p25 = float(np.percentile(positive_peaks, 25))
        p75 = float(np.percentile(positive_peaks, 75))
        parenchyma = (
            (brain_mask > 0.5)
            & (peak_conc_all > p25)
            & (peak_conc_all < p75)
        )

        if np.sum(parenchyma) > 100:
            return float(np.median(peak_time_map[parenchyma]))
        return float(np.median(peak_time_map[brain_mask > 0.5]))

    # -----------------------------------------------------------------
    # Step 6: Candidate metric calculation
    # -----------------------------------------------------------------

    @staticmethod
    def _calculate_candidate_metrics(
        conc_curve: npt.NDArray[Any],
        signal_curve: Optional[npt.NDArray[Any]],
        tr: float,
        baseline_frames: int,
    ) -> Optional[dict[str, float]]:
        """Compute comprehensive AIF quality metrics for one candidate.

        Returns None if the curve is too short or degenerate.
        """
        n_time = len(conc_curve)
        if n_time < baseline_frames + 10:
            return None

        baseline_conc = float(np.mean(conc_curve[:baseline_frames]))

        # First-pass window: _FP_GAP_SECONDS-40 s after baseline
        fp_start = baseline_frames + max(1, int(_FP_GAP_SECONDS / tr))
        fp_end = min(baseline_frames + int(40.0 / tr), n_time)
        if fp_start >= n_time or fp_end <= fp_start:
            return None

        fp_window = conc_curve[fp_start:fp_end]
        fp_peak_rel = int(np.argmax(fp_window))
        fp_peak_abs = fp_start + fp_peak_rel
        fp_peak_val = float(fp_window[fp_peak_rel])
        fp_peak_time = fp_peak_abs * tr

        # BRR — baseline return ratio (tail_mean / peak * 100)
        exclude_end = int(0.2 * n_time)
        tail_start = max(fp_end, n_time - exclude_end - 10)
        tail_end = max(tail_start + 1, n_time - exclude_end)
        tail_window = conc_curve[tail_start:tail_end]
        if len(tail_window) > 0 and fp_peak_val > EPSILON_SIGNAL:
            brr = float(np.mean(tail_window)) / fp_peak_val * 100.0
        else:
            brr = float("nan")

        # tail_fraction_late — late-tail mean / first-pass peak (fraction,
        # not %). Distinct from BRR: BRR uses an interior window
        # (n_time - 0.2*n_time - 10 .. n_time - 0.2*n_time), tail_fraction_late
        # uses the absolute last AIF_TAIL_WINDOW_S of the acquisition. Used
        # by the post-score kinetic filter to reject tissue-PVE / BBB-leakage
        # voxels with sustained late plateau.
        tail_late_n = max(1, int(round(AIF_TAIL_WINDOW_S / tr)))
        tail_late_window = conc_curve[-tail_late_n:]
        if len(tail_late_window) > 0 and fp_peak_val > EPSILON_SIGNAL:
            tail_fraction_late = float(np.mean(tail_late_window)) / fp_peak_val
        else:
            tail_fraction_late = float("nan")

        # PBR — peak-to-baseline ratio
        if baseline_conc > EPSILON_SIGNAL:
            pbr = fp_peak_val / baseline_conc
        else:
            pbr = float("nan")

        # SPR — secondary peak ratio
        post_fp_start = fp_end
        post_fp_end = n_time - exclude_end
        if post_fp_end > post_fp_start:
            post_curve = conc_curve[post_fp_start:post_fp_end]
            if len(post_curve) > 0 and fp_peak_val > EPSILON_SIGNAL:
                spr = float(np.max(post_curve)) / fp_peak_val * 100.0
            else:
                spr = 0.0
        else:
            spr = 0.0

        # PMI — post-peak monotonicity (% decreasing in 30 s after peak)
        decay_frames = int(30.0 / tr)
        decay_end = min(fp_peak_abs + decay_frames, n_time)
        decay_win = conc_curve[fp_peak_abs:decay_end]
        if len(decay_win) > 1:
            diffs = np.diff(decay_win)
            pmi = float(np.sum(diffs < 0)) / len(diffs) * 100.0
        else:
            pmi = float("nan")

        # FWHM in seconds (absolute half-max, from concentration curve)
        half_max = fp_peak_val / 2.0
        # Contiguous first-pass width: walk the rising/falling edges outward from
        # the first-pass peak so a recirculation/leakage lobe above half-max does
        # not inflate the width (which would trip the strict fwhm_max reject).
        if fp_peak_val > EPSILON_SIGNAL and conc_curve[fp_peak_abs] > half_max:
            lo = fp_peak_abs
            while lo > 0 and conc_curve[lo - 1] > half_max:
                lo -= 1
            hi = fp_peak_abs
            while hi < n_time - 1 and conc_curve[hi + 1] > half_max:
                hi += 1
            fwhm = float(hi - lo) * tr
        else:
            fwhm = float("nan")

        # PH and PSR from signal curve (not concentration)
        ph = float("nan")
        psr = float("nan")
        s0 = float("nan")
        if signal_curve is not None and len(signal_curve) == n_time:
            s0 = float(np.mean(signal_curve[:baseline_frames])) if baseline_frames > 0 else float("nan")
            post_bl_sig = signal_curve[baseline_frames:]
            if len(post_bl_sig) > 0 and abs(s0) > EPSILON_SIGNAL:
                s_min = float(np.min(post_bl_sig))
                ph = 100.0 * (s0 - s_min) / s0
                ph = max(0.0, ph)
                # PSR
                tail_n = min(5, len(signal_curve) // 10)
                if tail_n > 0 and abs(s0 - s_min) > EPSILON_SIGNAL:
                    s1 = float(np.mean(signal_curve[-tail_n:]))
                    psr = 100.0 * (s1 - s_min) / (s0 - s_min)

        scan_duration = (n_time - 1) * tr
        max_allowed_ttp = 0.8 * scan_duration

        return {
            "ttp": fp_peak_time,
            "ph": ph,
            "psr": psr,
            "s0": s0,
            "peak_conc": fp_peak_val,
            "baseline_return_ratio": brr,
            "peak_to_baseline_ratio": pbr,
            "secondary_peak_ratio": spr,
            "post_peak_monotonicity": pmi,
            "fwhm": fwhm,
            "baseline_conc": baseline_conc,
            "dt": tr,
            "pre_bolus_frames": baseline_frames,
            "scan_duration": scan_duration,
            "max_allowed_ttp": max_allowed_ttp,
            "peak_time_frame": float(fp_peak_abs),
            "tail_fraction_late": tail_fraction_late,
        }

    # -----------------------------------------------------------------
    # Effective dose helper (shared by candidate scoring + synthetic AIF)
    # -----------------------------------------------------------------

    def _effective_dose(
        self,
        dose: Optional[float] = None,
        agent_standard_dose: Optional[float] = None,
        gd_multiplicity: Optional[int] = None,
    ) -> float:
        """Effective Gd dose relative to 0.1 mmol/kg reference (unclamped).

        Parameters
        ----------
        dose : float, optional
            Override; falls back to self.inputs.dose.
        agent_standard_dose : float, optional
            Override; falls back to self.inputs.agent_standard_dose.
        gd_multiplicity : int, optional
            Override; falls back to self.inputs.gd_multiplicity.

        Returns
        -------
        float
            Raw multiplier ``(agent_std * dose * gd_mult) / GD_DOTA_STANDARD_DOSE``.
            Consumers apply their own clamping.
        """
        d = dose if dose is not None else (
            float(self.inputs.dose) if isdefined(self.inputs.dose) else 1.0
        )
        a = agent_standard_dose if agent_standard_dose is not None else (
            float(self.inputs.agent_standard_dose)
            if isdefined(self.inputs.agent_standard_dose) else 0.1
        )
        g = gd_multiplicity if gd_multiplicity is not None else (
            int(self.inputs.gd_multiplicity)
            if isdefined(self.inputs.gd_multiplicity) else 1
        )
        return (a * d * g) / GD_DOTA_STANDARD_DOSE

    # -----------------------------------------------------------------
    # Step 7: Cost-function scoring
    # -----------------------------------------------------------------

    def _score_candidate_cost(
        self,
        metrics: dict[str, float],
        criteria: str = "strict",
        median_s0: Optional[float] = None,
        *,
        dose: Optional[float] = None,
        agent_standard_dose: Optional[float] = None,
        gd_multiplicity: Optional[int] = None,
    ) -> float:
        """Score a candidate with the AIF cost function.

        Lower is better.  Returns ``inf`` for physiologically implausible
        candidates (hard rejection).

        Parameters
        ----------
        metrics : dict
            Output of ``_calculate_candidate_metrics``.
        criteria : str
            One of 'strict', 'relaxed', 'permissive'.
        median_s0 : float or None
            Median baseline signal for S0 partial-volume rejection.
        dose : float or None
            Contrast agent dose multiplier (default 1.0).
        agent_standard_dose : float or None
            Standard molecular dose in mmol/kg (default 0.1).
        gd_multiplicity : int or None
            Gd3+ ions per molecule (1=monomeric, 4=gadoquatrane).
            Falls back to self.inputs.gd_multiplicity if None.

        Returns
        -------
        float
            Cost (lower = better), or inf if rejected.
        """
        baseline_end_time = (
            (metrics["pre_bolus_frames"] - 1) * metrics["dt"]
            if metrics["pre_bolus_frames"] > 0
            else 0.0
        )
        ttp_after_baseline = metrics["ttp"] - baseline_end_time

        # --- Hard rejections ---

        # 1. End-of-scan exclusion
        if metrics["ttp"] > metrics.get("max_allowed_ttp", float("inf")):
            return float("inf")

        # 2. BRR range check — valid arterial BRR should be 5-35%
        # Negative BRR indicates T1 leakage contamination (tail > baseline)
        brr = metrics.get("baseline_return_ratio", float("nan"))
        brr_range: dict[str, tuple[float, float]] = {
            "strict": (5.0, 30.0),      # strict: clear arterial recirculation
            "relaxed": (0.0, 40.0),     # relaxed: reject obvious T1 contamination
            "permissive": (-10.0, 50.0),  # permissive: allow some noise margin
        }
        brr_lo, brr_hi = brr_range.get(criteria, (-10.0, 50.0))
        if not np.isnan(brr) and not (brr_lo < brr < brr_hi):
            return float("inf")

        # 3. PBR
        pbr = metrics.get("peak_to_baseline_ratio", float("nan"))
        pbr_thresh = {"strict": 3.0, "relaxed": 2.0, "permissive": 2.0}
        if not np.isnan(pbr) and pbr < pbr_thresh.get(criteria, 2.0):
            return float("inf")

        # 4. SPR
        spr = metrics.get("secondary_peak_ratio", 0.0)
        spr_thresh = {"strict": 70, "relaxed": 85, "permissive": 100}
        if spr > spr_thresh.get(criteria, 100):
            return float("inf")

        # 5. PMI
        pmi = metrics.get("post_peak_monotonicity", float("nan"))
        pmi_thresh = {"strict": 60, "relaxed": 55, "permissive": 40}
        if not np.isnan(pmi) and pmi < pmi_thresh.get(criteria, 40):
            return float("inf")

        # 5b. FWHM max
        fwhm = metrics.get("fwhm", float("nan"))
        fwhm_max = {"strict": 10.0, "relaxed": 12.0, "permissive": 18.0}
        if not np.isnan(fwhm) and fwhm > fwhm_max.get(criteria, 18.0):
            return float("inf")

        # 5c. FWHM min — reject sub-TR spikes (not physiological).
        # Arterial first-pass FWHM is typically 4-8 s; anything below
        # ~2 s is noise or intravoxel dephasing artifact.
        fwhm_min = {"strict": 3.0, "relaxed": 2.5, "permissive": 2.0}
        if not np.isnan(fwhm) and fwhm < fwhm_min.get(criteria, 2.0):
            return float("inf")

        # 6. S0 partial volume rejection
        if median_s0 is not None and not np.isnan(metrics.get("s0", float("nan"))):
            s0_ratio = metrics["s0"] / (median_s0 + EPSILON_SIGNAL)

            if s0_ratio < S0_RATIO_FLOOR:
                return float("inf")

            s0_factor = {"strict": 0.40, "relaxed": 0.60, "permissive": 0.80}
            if s0_ratio > s0_factor.get(criteria, 0.80):
                return float("inf")

        # 7. TTP window
        ttp_ranges: dict[str, tuple[float, float]] = {
            "strict": (5.0, 40.0),
            "relaxed": (3.0, 40.0),
            "permissive": (2.0, 45.0),
        }
        min_ttp, max_ttp = ttp_ranges.get(criteria, (2.0, 45.0))
        if not (min_ttp <= ttp_after_baseline <= max_ttp):
            return float("inf")

        # 8. PH min
        ph = metrics.get("ph", float("nan"))
        ph_min = {"strict": 25, "relaxed": 15, "permissive": 12}
        if not np.isnan(ph) and ph < ph_min.get(criteria, 12):
            return float("inf")

        # 8b. Peak concentration bounds — B0- and dose-scaled artifact rejection.
        # Arterial peak ΔR2* scales ∝ B0·dose, so both floor and ceiling ride the
        # same law as the synthetic AIF peak (a fixed floor would only be valid at
        # its 3T/full-dose calibration point). Floor is clamped to the ΔR2* noise
        # guard; ceiling scales up for super-doses but never below its 3T value.
        effective_dose = 1.0  # default, overwritten below if peak_conc available
        peak_conc = metrics.get("peak_conc", float("nan"))
        if not np.isnan(peak_conc):
            # Cap 3.0 (vs 2.5 for synthetic AIF): artifact rejection tolerates
            # slight nonlinearity; synthetic AIF amplitude must stay linear.
            effective_dose = float(np.clip(
                self._effective_dose(dose, agent_standard_dose, gd_multiplicity),
                0.2, 3.0,
            ))
            field_strength = (
                float(self.inputs.field_strength)
                if isdefined(self.inputs.field_strength) else 3.0
            )
            b0_scale = field_strength / 3.0

            peak_floor = max(
                AIF_PEAK_FLOOR_3T * b0_scale * effective_dose,
                SYNTH_AIF_MIN_PEAK,
            )
            peak_ceiling = AIF_PEAK_CEILING_3T * b0_scale * max(effective_dose, 1.0)

            if peak_conc < peak_floor or peak_conc > peak_ceiling:
                return float("inf")

        # 9. PSR max
        psr = metrics.get("psr", float("nan"))
        psr_max = {"strict": 110, "relaxed": 120, "permissive": 130}
        if not np.isnan(psr) and psr > psr_max.get(criteria, 130):
            return float("inf")

        # 9b. PSR min (signal never recovers → venous/leakage contamination)
        psr_min = {"strict": 40, "relaxed": 30, "permissive": 20}
        if not np.isnan(psr) and psr < psr_min.get(criteria, 20):
            return float("inf")

        # --- Continuous scoring ---
        ttp_score = float(np.exp(ttp_after_baseline / 20.0))
        ph_score = 100.0 / (ph + EPSILON_SIGNAL) if not np.isnan(ph) else 5.0
        brr_score = brr / 10.0 if not np.isnan(brr) else 5.0
        psr_val = psr if not np.isnan(psr) else 100.0
        psr_score = abs(psr_val - 100.0) / 10.0
        mono_bonus = (100 - pmi) / 20.0 if not np.isnan(pmi) else 2.5

        # Peak concentration bonus — higher peak ΔR2* = better deconvolution SNR.
        # Log-scale centered at ~10 s⁻¹, capped to prevent susceptibility
        # artifact over-selection.  Cap scales with effective_dose for
        # high-Gd agents (gadoquatrane).
        peak_conc_val = metrics.get("peak_conc", float("nan"))
        if not np.isnan(peak_conc_val) and peak_conc_val > 0:
            eff = effective_dose
            cap = 30.0 * max(eff, 1.0)
            peak_conc_score = -np.log(min(peak_conc_val, cap) / 10.0)
        else:
            peak_conc_score = 5.0

        score = (
            1.0 * ttp_score
            + 0.5 * ph_score
            + 2.0 * brr_score
            + 0.3 * psr_score
            + 0.5 * mono_bonus
            + 3.0 * peak_conc_score
        )

        return float(score)

    # -----------------------------------------------------------------
    # Step 8: Score + select candidates
    # -----------------------------------------------------------------

    def _score_and_select_candidates(
        self,
        candidate_indices: npt.NDArray[Any],
        conc_data: npt.NDArray[Any],
        signal_data: Optional[npt.NDArray[Any]],
        peak_conc_map: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
        criteria: str,
        tissue_ttp_frame: float,
        atlas_mask: Optional[npt.NDArray[Any]] = None,
    ) -> tuple[list[tuple[float, tuple[int, int, int], dict[str, Any]]], int]:
        """Score pre-filtered candidates and return ALL that pass.

        Post-scoring distance, laterality, and kinetic filters run after this
        method, so all passing candidates are returned for downstream
        selection.

        Returns
        -------
        tuple of (scored_candidates, n_available)
            scored_candidates : list of (cost, (x,y,z), metrics) sorted
                ascending by cost.
            n_available : total number of candidates that passed scoring.
        """
        # Median S0 for partial volume detection
        median_s0: Optional[float] = None
        if signal_data is not None:
            bl = max(1, baseline_frames)
            s0_map = np.mean(signal_data[..., :bl], axis=-1)
            brain_s0 = s0_map[brain_mask > 0.5]
            positive_s0 = brain_s0[brain_s0 > 0]
            if len(positive_s0) > 0:
                median_s0 = float(np.median(positive_s0))

        scored: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []

        for idx_arr in candidate_indices:
            ix = (int(idx_arr[0]), int(idx_arr[1]), int(idx_arr[2]))
            conc_curve = conc_data[ix[0], ix[1], ix[2], :]

            sig_curve: Optional[npt.NDArray[Any]] = None
            if signal_data is not None:
                sig_curve = signal_data[ix[0], ix[1], ix[2], :]

            m = self._calculate_candidate_metrics(
                conc_curve, sig_curve, tr, baseline_frames
            )
            if m is None:
                continue

            # Skip if critical metrics are NaN
            if np.isnan(m["ttp"]):
                continue

            m["peak_conc"] = float(peak_conc_map[ix])

            cost = self._score_candidate_cost(m, criteria=criteria, median_s0=median_s0)

            # Tissue TTP cross-validation. Compared in seconds (not frames) so
            # the causality tolerance is TR-invariant: a feeding artery peaks
            # before parenchyma, so a candidate peaking after tissue is venous.
            candidate_ttp_frame = m["ttp"] / tr
            seconds_before_tissue = (tissue_ttp_frame - candidate_ttp_frame) * tr

            if np.isfinite(cost):
                if seconds_before_tissue < -AIF_TTP_REJECT_AFTER_TISSUE_S:
                    cost = float("inf")  # Peaks well after tissue → venous
                elif seconds_before_tissue < 0.0:
                    cost += 10.0  # Peaks just after tissue
                elif seconds_before_tissue < AIF_TTP_NEAR_TISSUE_S:
                    cost += 2.0  # Mild penalty: close to tissue

            # Atlas-preference penalty — prefer known arterial territories
            if (
                np.isfinite(cost)
                and atlas_mask is not None
                and not atlas_mask[ix[0], ix[1], ix[2]]
            ):
                cost += 10.0

            if np.isfinite(cost):
                scored.append((cost, ix, m))

        # Sort by cost ascending (lower = better)
        scored.sort(key=lambda x: x[0])
        n_available = len(scored)
        return scored, n_available

    # -----------------------------------------------------------------
    # Step 8b: Post-scoring distance filter and laterality accounting
    # -----------------------------------------------------------------

    def _apply_post_scoring_filters(
        self,
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        lesion_mask: Optional[npt.NDArray[Any]],
        affine: npt.NDArray[Any],
        voxel_sizes: tuple[float, ...],
        lesion_min_distance_mm: float = LESION_EXCLUSION_DISTANCE_MM,
        enhancing_mask: Optional[npt.NDArray[Any]] = None,
    ) -> tuple[
        list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        dict[str, Any],
    ]:
        """Apply distance filtering and record candidate laterality.

        Step 4.5: Distance-based lesion exclusion.
        Step 4.6: Descriptive laterality accounting when a unilateral
        lesion mask is present. Candidate costs and ordering are unchanged.

        Parameters
        ----------
        top_candidates : list
            Scored ``(cost, (x,y,z), metrics)`` tuples.
        lesion_mask : ndarray or None
            3D lesion mask (dilated whole-tumor) for distance exclusion.
        affine : ndarray
            4x4 affine transformation matrix (voxel -> RAS world).
        voxel_sizes : tuple of float
            Voxel dimensions in mm ``(dx, dy, dz)``.
        lesion_min_distance_mm : float
            Minimum Euclidean distance from lesion in mm.
        enhancing_mask : ndarray or None
            3D enhancing tumor mask for laterality determination.
            Uses this instead of ``lesion_mask`` for laterality to
            avoid bilateral misclassification from dilated whole-tumor
            masks that cross the midline.

        Returns
        -------
        tuple of (filtered_candidates, filter_metadata)
        """
        meta: dict[str, Any] = {}

        # --- Step 4.5: Distance-based lesion filtering ---
        if (
            lesion_mask is not None
            and lesion_min_distance_mm > 0
            and np.any(lesion_mask > 0.5)
        ):
            lesion_coords = np.argwhere(lesion_mask > 0.5)  # (N, 3)
            # Scale to mm
            lesion_mm = lesion_coords * np.array(voxel_sizes)

            filtered = []
            excluded = []
            for entry in top_candidates:
                _, coord, metrics = entry
                coord_mm = np.array(coord) * np.array(voxel_sizes)
                dists = np.linalg.norm(lesion_mm - coord_mm, axis=1)
                min_dist = float(np.min(dists))
                metrics["lesion_distance_mm"] = min_dist

                if min_dist >= lesion_min_distance_mm:
                    filtered.append(entry)
                else:
                    excluded.append(entry)

            meta["n_distance_excluded"] = len(excluded)
            if filtered:
                top_candidates = filtered
            else:
                # A fully excluded tier is normal cascade behavior. Return an
                # empty pool so the next tier is attempted; terminal fallback
                # or failure is reported explicitly downstream.
                meta["all_candidates_excluded_by_distance"] = True
                top_candidates = []

        # --- Step 4.6: Descriptive candidate laterality accounting ---
        # Uses enhancing_mask for laterality if provided (avoids bilateral
        # misclassification from dilated whole-tumor masks crossing midline).
        tumor_laterality: Optional[str] = None
        n_contra = 0
        n_ipsi = 0

        laterality_mask = enhancing_mask if (
            enhancing_mask is not None and np.any(enhancing_mask > 0.5)
        ) else lesion_mask

        if (
            laterality_mask is not None
            and np.any(laterality_mask > 0.5)
        ):
            tumor_laterality = self._determine_tumor_laterality(
                laterality_mask, affine
            )
            if tumor_laterality is not None and tumor_laterality != "bilateral":
                n_contra, n_ipsi = self._count_candidates_by_hemisphere(
                    top_candidates, affine, tumor_laterality,
                )

        meta["tumor_laterality"] = tumor_laterality
        meta["n_contralateral"] = n_contra
        meta["n_ipsilateral"] = n_ipsi
        meta["laterality_mask_source"] = (
            "enhancing" if (enhancing_mask is not None and np.any(enhancing_mask > 0.5))
            else "lesion"
        )

        return top_candidates, meta

    # -----------------------------------------------------------------
    # Step 8.5: Per-voxel kinetic+anchor filter (post-score, pre-average)
    # -----------------------------------------------------------------

    @staticmethod
    def _apply_postscore_aif_filters(
        scored: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        tr: float,
    ) -> tuple[
        list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        dict[str, Any],
    ]:
        """Apply per-voxel kinetic-coherence filters.

        Two filters in order:

        1. **TR-scaled peak-spread filter**: keep voxels whose
           ``peak_time_frame`` is within
           ``ceil(AIF_PEAK_SPREAD_MAX_SECONDS / tr)`` frames of the
           input pool's 20th-percentile peak time. The 20th-percentile
           reference (rather than ``min``) is robust to a single
           early-peaking outlier (e.g. a noise voxel that would
           otherwise drag the gate down by 1-2 frames and reject
           legitimate co-arterial voxels). The TR-scaled spread limit
           accommodates frame quantization on argmax peak detection
           plus per-voxel argmax jitter at SNR 20-40 (~±1 frame), so
           true co-arterial voxels at TR=1.5 s spanning 2 measured
           frames are retained.
        2. **Tail/peak filter**: keep voxels with
           ``tail_fraction_late <= AIF_TAIL_FRACTION_MAX`` (default
           0.15 — see ``dsc_constants.py`` for the PVF rationale).

        TTP causality (peak before tissue) is **not** enforced here;
        it is the responsibility of ``_score_candidate_cost`` upstream
        (``_score_and_select_candidates`` rejects ``frames_before_tissue
        < -2`` outright and applies graduated penalties for borderline
        cases).

        Parameters
        ----------
        scored : list
            List of ``(cost, coord, metrics)`` tuples from
            ``_score_and_select_candidates``. Each ``metrics`` dict
            must contain ``peak_time_frame`` (float, frame index
            of first-pass peak) and ``tail_fraction_late`` (float,
            last 30 s mean ΔR2* / first-pass peak).
        tr : float
            Repetition time in seconds. Used to convert the
            ``AIF_PEAK_SPREAD_MAX_SECONDS`` physiologic spread
            ceiling to a frame-count gate at runtime.

        Returns
        -------
        tuple
            ``(filtered, diagnostics)`` where ``filtered`` is the
            survivor list (preserving input order) and ``diagnostics``
            is a dict with per-filter rejection counts and pool stats:
            ``aif_n_after_filters``, ``aif_n_filtered_peakspread``,
            ``aif_n_filtered_tail``, ``aif_peak_time_spread_frames``,
            ``aif_tail_max_fraction``.
        """
        zero_diag: dict[str, Any] = {
            "aif_n_after_filters": 0,
            "aif_n_filtered_peakspread": 0,
            "aif_n_filtered_tail": 0,
            "aif_peak_time_spread_frames": 0,
            "aif_tail_max_fraction": float("nan"),
        }
        if not scored:
            return [], zero_diag

        # Filter 1: TR-scaled peak-spread with 20th-percentile reference.
        # Compute reference from finite peak_time_frames only (NaN/inf
        # would poison np.percentile); if no finite values exist, reject
        # the whole pool.
        peak_times_all = [
            float(m.get("peak_time_frame", float("inf")))
            for _, _, m in scored
        ]
        finite_peak_times = [pt for pt in peak_times_all if np.isfinite(pt)]
        if not finite_peak_times:
            return [], zero_diag
        earliest_pt = float(np.percentile(finite_peak_times, 20))
        max_frame_spread = max(
            1, int(np.ceil(AIF_PEAK_SPREAD_MAX_SECONDS / max(tr, 1e-3)))
        )
        spread_max = earliest_pt + max_frame_spread
        spread_pass: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        n_filtered_spread = 0
        for cost, coord, metrics in scored:
            pt = float(metrics.get("peak_time_frame", float("inf")))
            if pt <= spread_max:
                spread_pass.append((cost, coord, metrics))
            else:
                n_filtered_spread += 1

        # Filter 2: tail/peak threshold
        tail_pass: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        n_filtered_tail = 0
        for cost, coord, metrics in spread_pass:
            tail = metrics.get("tail_fraction_late")
            if tail is None or not np.isfinite(tail):
                n_filtered_tail += 1
                continue
            if float(tail) <= float(AIF_TAIL_FRACTION_MAX):
                tail_pass.append((cost, coord, metrics))
            else:
                n_filtered_tail += 1

        # Diagnostics over the final survivor pool
        if tail_pass:
            peak_times = [
                float(m["peak_time_frame"]) for _, _, m in tail_pass
            ]
            tails = [
                float(m["tail_fraction_late"]) for _, _, m in tail_pass
            ]
            spread = max(peak_times) - min(peak_times)
            tail_max = max(tails)
        else:
            spread = 0
            tail_max = float("nan")

        diag = {
            "aif_n_after_filters": len(tail_pass),
            "aif_n_filtered_peakspread": n_filtered_spread,
            "aif_n_filtered_tail": n_filtered_tail,
            "aif_peak_time_spread_frames": spread,
            "aif_tail_max_fraction": tail_max,
        }
        return tail_pass, diag

    # -----------------------------------------------------------------
    # Step 9: ROI averaging + smoothing
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_roi_averaged_aif(
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        tr: float,
        roi_size: int = 3,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Compute inverse-cost weighted AIF with 3x3x3 ROI averaging.

        Parameters
        ----------
        top_candidates : list
            List of (cost, (x,y,z), metrics) tuples.
        conc_data : ndarray
            4D concentration data.
        brain_mask : ndarray
            3D brain mask.
        tr : float
            Frame period in seconds, used to keep temporal smoothing
            consistent across acquisition rates.
        roi_size : int
            Side length of cubic ROI around each candidate.

        Returns
        -------
        aif_curve : ndarray
            1D AIF time course.
        aif_mask : ndarray
            3D boolean mask of voxels used.
        """
        nt = conc_data.shape[-1]
        aif_mask = np.zeros(conc_data.shape[:3], dtype=bool)

        if len(top_candidates) == 1:
            _, coord, _ = top_candidates[0]
            slices = []
            for dim in range(3):
                s = max(0, coord[dim] - roi_size // 2)
                e = min(conc_data.shape[dim], coord[dim] + roi_size // 2 + 1)
                slices.append(slice(s, e))
            roi = np.zeros(conc_data.shape[:3], dtype=bool)
            roi[tuple(slices)] = True
            valid = roi & (brain_mask > 0.5)
            aif_mask |= valid
            roi_data = conc_data[valid]
            if roi_data.shape[0] > 0:
                aif_curve = np.mean(roi_data, axis=0)
            else:
                aif_mask[coord] = True
                aif_curve = conc_data[coord[0], coord[1], coord[2], :]
        else:
            costs = np.array([c for c, _, _ in top_candidates])
            weights = 1.0 / (costs - costs.min() + 0.1)
            weights /= np.sum(weights)

            weighted_aif = np.zeros(nt, dtype=np.float64)
            for i, (cost, coord, _) in enumerate(top_candidates):
                slices = []
                for dim in range(3):
                    s = max(0, coord[dim] - roi_size // 2)
                    e = min(conc_data.shape[dim], coord[dim] + roi_size // 2 + 1)
                    slices.append(slice(s, e))
                roi = np.zeros(conc_data.shape[:3], dtype=bool)
                roi[tuple(slices)] = True
                valid = roi & (brain_mask > 0.5)
                aif_mask |= valid

                roi_data = conc_data[valid]
                if roi_data.shape[0] > 0:
                    roi_aif = np.mean(roi_data, axis=0)
                else:
                    roi_aif = conc_data[coord[0], coord[1], coord[2], :]
                weighted_aif += weights[i] * roi_aif

            aif_curve = weighted_aif

        # Smooth over a fixed duration rather than a fixed frame count so the
        # arterial peak receives comparable filtering across acquisition TRs.
        if len(aif_curve) >= 5:
            from pmriprep.utils.dsc_aif import get_aif_smoothing_window

            window_length = get_aif_smoothing_window(len(aif_curve), tr)
            aif_curve = savgol_filter(
                aif_curve, window_length=window_length, polyorder=2
            )

        # Negative ΔR2* is physically valid (symmetric noise, T1 recirculation).
        # Do NOT clip to zero — it biases the AIF integral, corrupts the SVD
        # Toeplitz matrix, and disables the gamma-fit defect detector.

        return aif_curve, aif_mask

    # -----------------------------------------------------------------
    # AIF quality checks (extracted for use in cascade quality gate)
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_aif_spr(
        curve: npt.NDArray[Any],
        baseline_frames: int,
        tr: float,
    ) -> tuple[float, int, float]:
        """Compute secondary peak ratio for an AIF curve.

        Returns
        -------
        spr : float
            Secondary peak ratio (%).
        fp_peak_frame : int
            First-pass peak frame index.
        fp_peak_val : float
            First-pass peak value.
        """
        n_time = len(curve)
        fp_start = baseline_frames + max(1, int(_FP_GAP_SECONDS / tr))
        fp_end = min(baseline_frames + int(40.0 / tr), n_time)
        if fp_start >= n_time or fp_end <= fp_start:
            return 0.0, -1, 0.0

        fp_window = curve[fp_start:fp_end]
        fp_peak_rel = int(np.argmax(fp_window))
        fp_peak_abs = fp_start + fp_peak_rel
        fp_peak_val = float(fp_window[fp_peak_rel])

        if baseline_frames < 1:
            return 0.0, fp_peak_abs, fp_peak_val
        baseline = float(np.mean(curve[:baseline_frames]))
        fp_height = fp_peak_val - baseline
        if fp_height < EPSILON_SIGNAL:
            return 0.0, fp_peak_abs, fp_peak_val

        recirc_start = fp_end + max(1, int(15.0 / tr))
        if recirc_start >= n_time:
            return 0.0, fp_peak_abs, fp_peak_val

        post_recirc = curve[recirc_start:]
        spr = (float(np.max(post_recirc)) - baseline) / fp_height * 100.0
        return spr, fp_peak_abs, fp_peak_val

    @staticmethod
    def _aif_global_max_in_fp(
        curve: npt.NDArray[Any],
        baseline_frames: int,
        tr: float,
    ) -> bool:
        """Return True if global maximum falls within the first-pass window."""
        n_time = len(curve)
        fp_start = baseline_frames + max(1, int(_FP_GAP_SECONDS / tr))
        fp_end = min(baseline_frames + int(40.0 / tr), n_time)
        global_max_frame = int(np.argmax(curve))
        return fp_start <= global_max_frame < fp_end

    @staticmethod
    def _compute_aif_brr(
        curve: npt.NDArray[Any],
        baseline_frames: int,
        tr: float,
    ) -> float:
        """Compute baseline-return ratio (%) for an averaged AIF curve.

        Matches the per-candidate BRR computation in
        ``_calculate_candidate_metrics``: mean of the late-tail window
        divided by the first-pass peak, times 100.
        """
        n_time = len(curve)
        fp_start = baseline_frames + max(1, int(_FP_GAP_SECONDS / tr))
        fp_end = min(baseline_frames + int(40.0 / tr), n_time)
        if fp_start >= n_time or fp_end <= fp_start:
            return float("nan")

        fp_window = curve[fp_start:fp_end]
        fp_peak_val = float(np.max(fp_window))
        if fp_peak_val <= EPSILON_SIGNAL:
            return float("nan")

        exclude_end = int(0.2 * n_time)
        tail_start = max(fp_end, n_time - exclude_end - 10)
        tail_end = max(tail_start + 1, n_time - exclude_end)
        tail_window = curve[tail_start:tail_end]
        if len(tail_window) == 0:
            return float("nan")
        return float(np.mean(tail_window)) / fp_peak_val * 100.0

    @staticmethod
    def _aif_passes_quality(
        curve: npt.NDArray[Any],
        baseline_frames: int,
        tr: float,
    ) -> AifQualityResult:
        """Check if an AIF curve passes quality gates.

        Returns
        -------
        AifQualityResult
            Named tuple with passes, spr, peak_ratio, brr.
        """
        spr, _, _ = AIFDetection._compute_aif_spr(curve, baseline_frames, tr)
        gmax_ok = AIFDetection._aif_global_max_in_fp(curve, baseline_frames, tr)
        pr = _compute_aif_peak_ratio(curve, baseline_frames, tr)
        pr_ok = (
            pr is not None
            and GAMMA_PEAK_RATIO_CASCADE_MIN <= pr <= GAMMA_PEAK_RATIO_CASCADE_MAX
        )
        brr = AIFDetection._compute_aif_brr(curve, baseline_frames, tr)
        brr_ok = np.isnan(brr) or brr >= AIF_AVG_BRR_MIN

        passes = spr <= 100.0 and gmax_ok and pr_ok and brr_ok
        return AifQualityResult(passes, spr, pr, brr)

    def _try_aif_pool_acceptance(
        self,
        candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
    ) -> Optional[AIFPoolAcceptanceResult]:
        """Compute the inverse-cost-weighted averaged AIF and gate it.

        Each candidate contributes its own voxel (``roi_size=1``). Tissue
        contamination from 3x3x3 neighbourhoods had no SNR benefit over the
        weighted mean across multiple candidate voxels.

        Returns
        -------
        AIFPoolAcceptanceResult or None
            Named tuple with (curve, mask, roi_size, spr, peak_ratio,
            brr) when the averaged curve passes quality gates;
            ``None`` otherwise.
        """
        curve, mask = self._compute_roi_averaged_aif(
            candidates, conc_data, brain_mask, tr, roi_size=1,
        )
        qr = self._aif_passes_quality(curve, baseline_frames, tr)
        if qr.passes:
            return AIFPoolAcceptanceResult(
                curve, mask, 1,
                qr.spr, qr.peak_ratio, qr.brr,
            )
        return None

    # -----------------------------------------------------------------
    # Synthetic AIF (calibrated gamma-variate)
    # -----------------------------------------------------------------

    def _compute_tissue_max_for_synthetic(
        self,
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
    ) -> Optional[float]:
        """Compute tissue max ΔR2* for synthetic AIF amplitude scaling.

        Parameters
        ----------
        conc_data : ndarray
            4D concentration data (x, y, z, t).
        brain_mask : ndarray
            3D brain mask.

        Returns
        -------
        float or None
            95th percentile of max ΔR2* across brain voxels, or None if
            no brain voxels available.
        """
        brain_conc = conc_data[brain_mask > 0.5]
        if brain_conc.size == 0:
            return None
        max_per_voxel = (
            brain_conc.max(axis=-1)
            if brain_conc.ndim > 1
            else float(brain_conc.max())
        )
        return float(np.percentile(max_per_voxel, 95))

    def _generate_synthetic_aif(
        self,
        n_timepoints: int,
        tr: float = 1.5,
        baseline_frames: int = 5,
        tissue_max_delta_r2star: Optional[float] = None,
        field_strength: float = 3.0,
        effective_dose: float = 1.0,
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Generate synthetic AIF using physiologically calibrated gamma-variate.

        Peak amplitude is dose- and B0-scaled from population-median arterial
        ΔR2* at 3T.

        Parameters
        ----------
        n_timepoints : int
            Number of time frames.
        tr : float, optional
            Repetition time in seconds.
        baseline_frames : int, optional
            Number of pre-bolus baseline frames.
        tissue_max_delta_r2star : float or None, optional
            95th percentile of max ΔR2* across brain
            voxels. QC diagnostic only — does not drive peak amplitude.
        field_strength : float, optional
            Magnetic field strength in Tesla.
        effective_dose : float, optional
            ``(agent_std * dose * gd_mult) / 0.1``, clamped.

        Returns
        -------
        tuple
            Tuple of (aif_curve, qc_metrics_dict).
        """
        # --- First-pass gamma-variate parameters ---
        alpha = 4.0
        time_to_peak = 4.5  # seconds after arrival
        arrival_frame = baseline_frames + 2
        t0 = arrival_frame * tr  # bolus arrival time (seconds)

        # Peak amplitude: population-median arterial ΔR2* scaled by B0 and dose
        peak = max(
            SYNTH_AIF_BASE_PEAK_3T * (field_strength / 3.0) * effective_dose,
            SYNTH_AIF_MIN_PEAK,
        )
        amplitude_source = "population_calibrated"

        # Recirculation (second-pass) parameters — dispersed, so a smaller,
        # flatter lobe than the first pass (its own alpha, not the first-pass 4.0)
        recirc_fraction = SYNTH_AIF_RECIRC_FRACTION
        recirc_alpha = SYNTH_AIF_RECIRC_ALPHA
        recirc_delay = SYNTH_AIF_RECIRC_DELAY_S  # seconds after first-pass peak
        recirc_dispersion = SYNTH_AIF_RECIRC_DISPERSION_S  # seconds (broadening)

        # --- Build time axis ---
        t = np.arange(n_timepoints) * tr
        aif: npt.NDArray[np.float64] = np.zeros(
            n_timepoints, dtype=np.float64,
        )

        # --- First-pass component ---
        post_arrival = t > t0
        if np.any(post_arrival):
            t_shifted = t[post_arrival] - t0
            ratio = t_shifted / time_to_peak
            # Gamma-variate: C(t) = C_peak * (ratio)^alpha * exp(alpha*(1-ratio))
            aif[post_arrival] = peak * (ratio ** alpha) * np.exp(
                alpha * (1.0 - ratio)
            )

        # --- Recirculation component ---
        recirc_t0 = t0 + time_to_peak + recirc_delay
        post_recirc = t > recirc_t0
        if np.any(post_recirc):
            t_recirc = t[post_recirc] - recirc_t0
            ratio_r = t_recirc / recirc_dispersion
            recirc_peak = peak * recirc_fraction
            aif[post_recirc] += recirc_peak * (ratio_r ** recirc_alpha) * np.exp(
                recirc_alpha * (1.0 - ratio_r)
            )

        # Clip negative values (numerical safety)
        aif = np.clip(aif, 0.0, None)

        # --- QC metrics ---
        metrics: dict[str, Any] = {
            "synthetic_aif_peak_delta_r2star": float(np.max(aif)),
            "synthetic_aif_target_peak": peak,
            "synthetic_aif_amplitude_source": amplitude_source,
            "synthetic_aif_peak_formula": "population_median_b0_scaled",
            "synthetic_aif_effective_dose": effective_dose,
            "synthetic_aif_field_strength": field_strength,
            "synthetic_aif_alpha": alpha,
            "synthetic_aif_time_to_peak_s": time_to_peak,
            "synthetic_aif_arrival_frame": arrival_frame,
            "synthetic_aif_recirc_fraction": recirc_fraction,
            "synthetic_aif_recirc_alpha": recirc_alpha,
            "synthetic_aif_recirc_delay_s": recirc_delay,
        }
        if tissue_max_delta_r2star is not None:
            metrics["synthetic_aif_tissue_p95_delta_r2star"] = float(
                tissue_max_delta_r2star
            )
            if peak > 0 and tissue_max_delta_r2star > 0:
                metrics["synthetic_aif_peak_to_tissue_ratio"] = round(
                    peak / tissue_max_delta_r2star, 2
                )

        return aif, metrics

    # -----------------------------------------------------------------
    # Step 9b: Finalize AIF metrics
    # -----------------------------------------------------------------

    @staticmethod
    def _finalize_aif_metrics(
        aif_curve: npt.NDArray[Any],
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        affine: npt.NDArray[Any],
        tr: float,
        n_candidates_requested: int,
        n_available: int,
        baseline_frames: int = 12,
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Finalize AIF: detect bolus arrival and compute comprehensive QC.

        Parameters
        ----------
        aif_curve : ndarray
            1D smoothed AIF concentration curve.
        top_candidates : list
            ``(cost, (x,y,z), metrics)`` tuples used.
        conc_data : ndarray
            4D concentration data.
        brain_mask : ndarray
            3D brain mask.
        affine : ndarray
            4x4 affine matrix.
        tr : float
            Repetition time in seconds.
        n_candidates_requested : int
            How many candidates were requested.
        n_available : int
            How many candidates passed scoring.
        baseline_frames : int
            Configured number of pre-bolus baseline frames.

        Returns
        -------
        tuple of (aif_curve, qc_dict)
        """
        from pmriprep.utils.dsc_signal import detect_bolus_arrival

        # Bolus arrival on the final AIF curve
        baseline_window = min(
            baseline_frames, max(2, len(aif_curve) // 3)
        )
        arrival_idx, n_baseline_detected = detect_bolus_arrival(
            aif_curve,
            baseline_frames=baseline_window,
            n_consecutive=3,
        )
        arrival_time_s = arrival_idx * tr
        total_frames = len(aif_curve)
        baseline_fraction = n_baseline_detected / total_frames

        # Best candidate info
        best_cost, best_coord, best_m = top_candidates[0]
        best_coord_int = tuple(int(c) for c in best_coord)

        # Convert best coord to world mm
        coord_homog = np.array(
            [best_coord[0], best_coord[1], best_coord[2], 1.0]
        )
        aif_centroid_mm = (affine @ coord_homog)[:3].tolist()

        # Hemisphere distribution (voxel midline for counting)
        midline_x = brain_mask.shape[0] // 2
        n_left = sum(
            1 for _, coord, _ in top_candidates if coord[0] < midline_x
        )
        n_right = len(top_candidates) - n_left
        is_bilateral = n_left > 0 and n_right > 0

        # Determine AIF hemisphere via world coordinates
        aif_hemisphere = "left" if aif_centroid_mm[0] < 0 else "right"

        # Edge slice warning
        n_slices = conc_data.shape[2]
        aif_z = best_coord_int[2]
        is_edge = aif_z == 0 or aif_z == n_slices - 1

        n_used = len(top_candidates)

        # Hemisphere coverage (harmonized with DCE)
        if n_left > 0 and n_right > 0:
            hemisphere_coverage = "bilateral"
        elif n_left > 0:
            hemisphere_coverage = "unilateral_left"
        else:
            hemisphere_coverage = "unilateral_right"

        qc: dict[str, Any] = {
            # Candidate counts
            "n_aif_voxels_selected": n_used,
            "n_aif_voxels_requested": n_candidates_requested,
            "n_aif_voxels_available": n_available,
            "aif_insufficient_candidates": n_available < 3,
            # Baseline / arrival
            "n_baseline_frames": n_baseline_detected,
            "baseline_fraction": float(baseline_fraction),
            "arrival_time_seconds": float(arrival_time_s),
            # Spatial
            "aif_voxel_coord": list(best_coord_int),
            "aif_centroid_mm": [float(v) for v in aif_centroid_mm],
            "aif_hemisphere": aif_hemisphere,
            "candidate_coords": [
                [int(c) for c in coord]
                for _, coord, _ in top_candidates
            ],
            "candidate_scores": _costs_to_scores(
                [float(cost) for cost, _, _ in top_candidates]
            ),
            # Hemisphere distribution
            "n_left_candidates": n_left,
            "n_right_candidates": n_right,
            "hemisphere_coverage": hemisphere_coverage,
            "is_bilateral": is_bilateral,
            # Edge slice
            "aif_slice": aif_z,
            "n_slices": n_slices,
            "aif_edge_slice_warning": is_edge,
            # Best candidate quality metrics
            "best_cost": float(best_cost),
            "mean_cost": float(
                np.mean([c for c, _, _ in top_candidates])
            ),
            # Detection-time AIF metrics under a single canonical
            # nested key. ``peak_conc`` is ΔR2* peak in 1/s (DSC) /
            # plasma-Gd peak in mM (DCE callers); other fields come
            # from the best candidate's metric record.
            "aif_metrics": {
                "peak_conc": float(np.max(aif_curve)),
                "ttp": float(best_m.get("ttp", 0.0)),
                "fwhm": float(best_m.get("fwhm", 0.0)),
                "ph": float(best_m.get("ph", 0.0)),
                "psr": float(best_m.get("psr", 0.0)),
                "brr": float(best_m.get("baseline_return_ratio", 0.0)),
                "arrival": float(arrival_time_s),
            },
        }

        return aif_curve, qc

    # -----------------------------------------------------------------
    # External AIF Handling
    # -----------------------------------------------------------------

    def _handle_external_aif(
        self,
        runtime: Any,
    ) -> Any:
        """Handle externally supplied AIF curve, bypassing all detection.

        Loads an external AIF from .txt, .npz, or .npy file, validates its
        length against DSC timepoints, and writes outputs. External AIFs are
        assumed to already be in ΔR2* concentration (s⁻¹).

        Parameters
        ----------
        runtime : Any
            Nipype runtime object.

        Returns
        -------
        runtime
            Updated Nipype runtime object.
        """
        import logging
        import nibabel as nib  # type: ignore[import-untyped]

        _logger = logging.getLogger(__name__)

        ext_path = Path(self.inputs.external_aif_file)

        # Load concentration image for shape and affine
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]
        n_timepoints = conc_data.shape[3]

        # Load external AIF
        suffix = ext_path.suffix.lower()
        if suffix == ".npz":
            npz = np.load(str(ext_path))
            if "aif" in npz:
                aif_curve = np.asarray(npz["aif"], dtype=np.float64)
            elif "delta_r2_star" in npz:
                aif_curve = np.asarray(npz["delta_r2_star"], dtype=np.float64)
            else:
                raise ValueError(
                    f"External AIF .npz must contain 'aif' or 'delta_r2_star' key, "
                    f"found: {list(npz.keys())}"
                )
        elif suffix == ".npy":
            aif_curve = np.load(str(ext_path)).astype(np.float64)
        elif suffix == ".tsv":
            aif_curve = np.loadtxt(str(ext_path), dtype=np.float64, skiprows=1)
        else:
            try:
                aif_curve = np.loadtxt(str(ext_path), dtype=np.float64)
            except ValueError:
                aif_curve = np.loadtxt(str(ext_path), dtype=np.float64, delimiter=",")

        # Squeeze (N,1) -> (N,)
        aif_curve = np.squeeze(aif_curve)
        if aif_curve.ndim != 1:
            raise ValueError(
                f"External AIF must be 1-D, got shape {aif_curve.shape}. "
                f"Multi-column files are not supported."
            )

        # Validate finite values
        if not np.all(np.isfinite(aif_curve)):
            raise ValueError(
                "External AIF contains NaN or Inf values. "
                "All values must be finite."
            )

        # Warn about negative values (expected for ΔR2* noise)
        if np.any(aif_curve < 0):
            n_neg = int(np.sum(aif_curve < 0))
            neg_frac = n_neg / len(aif_curve)
            if neg_frac > 0.3:
                _logger.warning(
                    "External AIF has %d negative values (%.0f%%) — check "
                    "baseline subtraction.", n_neg, neg_frac * 100,
                )
            else:
                _logger.info(
                    "External AIF has %d negative values (%.0f%%, normal "
                    "post-bolus noise).", n_neg, neg_frac * 100,
                )

        # Validate length
        if len(aif_curve) != n_timepoints:
            raise ValueError(
                f"External AIF has {len(aif_curve)} timepoints but DSC data "
                f"has {n_timepoints} volumes. They must match."
            )

        # Validate non-zero peak
        peak = float(np.max(aif_curve))
        if peak < 1e-6:
            raise ValueError(
                f"External AIF peak ({peak:.2e}) is near zero. "
                "Check that values are ΔR2* in s⁻¹ (not raw signal)."
            )

        # Warn if values look like they might be in mM rather than s⁻¹
        if peak < 0.5:
            _logger.warning(
                "External DSC AIF peak (%.3f) is very low. DSC AIF values "
                "should be ΔR2* in s⁻¹ (typically 5-50 s⁻¹ at peak). "
                "Values < 0.5 suggest wrong units (mM instead of s⁻¹).",
                peak,
            )

        from pmriprep.utils.dsc_aif import compute_aif_fwhm_frames

        # Refine the bootstrap baseline from the supplied arterial curve so
        # external and detected AIFs expose the same single downstream state.
        bootstrap_baseline = int(self.inputs.baseline_frames)
        tr = float(self.inputs.tr)
        if tr <= 0:
            zooms = conc_img.header.get_zooms()  # type: ignore[attr-defined,union-attr]
            tr = float(zooms[3]) if len(zooms) >= 4 and zooms[3] > 0 else 1.5
            if tr > 100.0:
                tr = 1.5
        baseline_frames, baseline_diagnostics = (
            self._refine_selected_aif_baseline(
                aif_curve,
                preliminary_baseline=bootstrap_baseline,
                bootstrap_baseline=bootstrap_baseline,
                tr=tr,
            )
        )
        cnr = compute_aif_cnr(aif_curve, baseline_frames)
        fwhm_frames = compute_aif_fwhm_frames(aif_curve, baseline_frames)
        washout = compute_aif_washout_fraction(aif_curve, baseline_frames)
        baseline_cv = compute_baseline_stability(aif_curve, baseline_frames)

        # Compute quality score from measured metrics (not hardcoded 1.0)
        quality_score = 0.0
        # CNR contribution (0-0.4): >10 is good
        quality_score += min(cnr / 25.0, 1.0) * 0.4
        # FWHM contribution (0-0.2): 3-15 frames is good
        if 3 <= fwhm_frames <= 15:
            quality_score += 0.2
        elif fwhm_frames > 0:
            quality_score += 0.1
        # Washout contribution (0-0.2): >0.2 is arterial
        quality_score += min(washout / 0.5, 1.0) * 0.2
        # Baseline stability contribution (0-0.2): CV < 0.1 is good
        if baseline_cv < 0.1:
            quality_score += 0.2
        elif baseline_cv < 0.3:
            quality_score += 0.1
        quality_score = min(quality_score, 1.0)

        qc_metrics: dict[str, Any] = {
            "external_aif": True,
            "external_aif_file": str(ext_path),
            "detection_method": "external",
            "quality_score": quality_score,
            "peak_delta_r2_star": peak,
            "cnr": cnr,
            "fwhm_frames": fwhm_frames,
            "washout_fraction": washout,
            "baseline_cv": baseline_cv,
            "n_candidates": 0,
            "baseline_frames": baseline_frames,
            "bootstrap_baseline_frames": bootstrap_baseline,
            "aif_baseline_frames": baseline_frames,
            "baseline_frames_used": baseline_frames,
            "n_baseline_frames": baseline_frames,
            "baseline_detection_method": "selected_aif_refined",
            # Detection-time AIF metrics under the canonical nested
            # key. The external path computes a subset of fields —
            # ttp / ph / psr / arrival are not derivable here, so they
            # are omitted (consumers must use .get(...) for them).
            "aif_metrics": {
                "peak_conc": float(peak),
                "fwhm": float(fwhm_frames),
                "brr": float(washout),
            },
        }
        qc_metrics.update(baseline_diagnostics)

        # Write AIF curve
        out_aif_curve = self._gen_outfilename("_aif_curve.txt")
        np.savetxt(out_aif_curve, aif_curve)

        # Write zero AIF mask (no spatial origin for external AIF)
        aif_mask_data = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        out_aif_mask = self._gen_outfilename("_aif_mask.nii.gz")
        mask_img = nib.Nifti1Image(aif_mask_data, affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_aif_mask)  # type: ignore[attr-defined]

        _logger.info(
            "Using external DSC AIF from %s (peak=%.2f s⁻¹, CNR=%.1f, quality=%.2f)",
            ext_path.name, peak, cnr, quality_score,
        )

        # No gamma fit for external AIF. Save external curve as both gamma
        # and measured outputs so downstream consumers always find a file.
        out_gamma_aif = self._gen_outfilename("_gamma_aif_curve.txt")
        np.savetxt(out_gamma_aif, aif_curve)
        out_measured_aif = self._gen_outfilename("_measured_aif_curve.txt")
        np.savetxt(out_measured_aif, aif_curve)
        qc_metrics["gamma_aif_fallback_to_raw"] = True
        qc_metrics["cbv_auc_denominator_is_synthetic"] = True  # external is user-supplied

        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "out_gamma_aif_curve": out_gamma_aif,
            "out_measured_aif_curve": out_measured_aif,
            "out_raw_aif_curve": "",
            "out_aif_coord": [0, 0, 0],
            "aif_curve": [float(x) for x in aif_curve],
            "detection_level": "external",
            "quality_score": quality_score,
            "candidate_count": 0,
            "gamma_fit_applied": False,
            "aif_baseline_frames": baseline_frames,
            "qc_metrics": qc_metrics,
        }

        return runtime

    def _load_detection_context(self, logger: Any) -> _DSCAIFDetectionContext:
        """Load and validate images, acquisition state, and detection options."""
        import nibabel as nib  # type: ignore[import-untyped]

        concentration_image = nib.load(self.inputs.concentration_file)
        concentration = concentration_image.get_fdata()  # type: ignore[attr-defined]
        affine = concentration_image.affine  # type: ignore[attr-defined]

        brain_image = nib.load(self.inputs.brain_mask_file)
        brain_mask = brain_image.get_fdata()  # type: ignore[attr-defined]
        if brain_mask.shape[:3] != concentration.shape[:3]:
            raise RuntimeError(
                f"Brain mask shape {brain_mask.shape[:3]} does not match "
                f"concentration data shape {concentration.shape[:3]}. "
                "The brain mask must be in native DSC space. "
                "Ensure transform_brain_mask node is connected upstream."
            )

        atlas_mask: Optional[npt.NDArray[Any]] = None
        atlas_is_mra = False
        if isdefined(self.inputs.aif_mask_file) and self.inputs.aif_mask_file:
            atlas_image = nib.load(self.inputs.aif_mask_file)
            atlas_mask = atlas_image.get_fdata()  # type: ignore[attr-defined]
        elif self.inputs.atlas_mask_file:
            atlas_is_mra = True
            atlas_image = nib.load(self.inputs.atlas_mask_file)
            atlas_mask = atlas_image.get_fdata()  # type: ignore[attr-defined]

        full_atlas_mask: Optional[npt.NDArray[Any]] = None
        if (
            isdefined(self.inputs.full_atlas_mask_file)
            and self.inputs.full_atlas_mask_file
        ):
            full_atlas_image = nib.load(self.inputs.full_atlas_mask_file)
            full_atlas_mask = full_atlas_image.get_fdata()  # type: ignore[attr-defined]

        lesion_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.lesion_mask_file:
            lesion_image = nib.load(self.inputs.lesion_mask_file)
            lesion_data = lesion_image.get_fdata()  # type: ignore[attr-defined]
            needs_resample = (
                lesion_data.shape[:3] != concentration.shape[:3]
                or not np.allclose(
                    lesion_image.affine,  # type: ignore[attr-defined]
                    concentration_image.affine,  # type: ignore[attr-defined]
                    atol=1e-3,
                )
            )
            if needs_resample:
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]

                resampled = resample_to_img(
                    lesion_image, concentration_image, interpolation="nearest",
                )
                lesion_mask = (resampled.get_fdata() > 0.5).astype(np.float64)  # type: ignore[attr-defined]
            else:
                lesion_mask = lesion_data

        enhancing_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.enhancing_mask_file:
            enhancing_image = nib.load(self.inputs.enhancing_mask_file)
            enhancing_data = enhancing_image.get_fdata()  # type: ignore[attr-defined]
            needs_resample = (
                enhancing_data.shape[:3] != concentration.shape[:3]
                or not np.allclose(
                    enhancing_image.affine,  # type: ignore[attr-defined]
                    concentration_image.affine,  # type: ignore[attr-defined]
                    atol=1e-3,
                )
            )
            if needs_resample:
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]

                resampled = resample_to_img(
                    enhancing_image, concentration_image, interpolation="nearest",
                )
                enhancing_mask = (resampled.get_fdata() > 0.5).astype(np.float64)  # type: ignore[attr-defined]
            else:
                enhancing_mask = enhancing_data

        signal: Optional[npt.NDArray[Any]] = None
        if self.inputs.signal_file:
            signal_image = nib.load(self.inputs.signal_file)
            signal = signal_image.get_fdata()  # type: ignore[attr-defined]

        lesion_distance = float(self.inputs.lesion_distance)
        if lesion_distance < LESION_EXCLUSION_FLOOR_MM:
            logger.warning(
                "lesion_distance=%.1f mm is below hard minimum (%.1f mm); "
                "clamping to %.1f mm.",
                lesion_distance,
                LESION_EXCLUSION_FLOOR_MM,
                LESION_EXCLUSION_FLOOR_MM,
            )
            lesion_distance = LESION_EXCLUSION_FLOOR_MM
        elif lesion_distance < LESION_EXCLUSION_DISTANCE_MM:
            logger.warning(
                "lesion_distance=%.1f mm is below recommended minimum "
                "(%.1f mm); peritumoral vascular abnormalities may "
                "contaminate AIF candidates.",
                lesion_distance,
                LESION_EXCLUSION_DISTANCE_MM,
            )

        zooms = concentration_image.header.get_zooms()  # type: ignore[attr-defined,union-attr]
        tr = float(self.inputs.tr)
        if tr <= 0:
            if len(zooms) >= 4 and zooms[3] > 0:
                tr = float(zooms[3])
                if tr > 100.0:
                    logger.warning(
                        "Auto-detected TR=%.1fs from NIfTI header seems too "
                        "large (possible millisecond units). Falling back to 1.5s.",
                        tr,
                    )
                    tr = 1.5
            else:
                tr = 1.5

        field_strength = (
            float(self.inputs.field_strength)
            if isdefined(self.inputs.field_strength)
            else 3.0
        )
        synthetic_effective_dose = float(
            np.clip(self._effective_dose(), 0.2, SYNTH_AIF_DOSE_CAP)
        )

        return _DSCAIFDetectionContext(
            concentration=concentration,
            affine=affine,
            brain_mask=brain_mask,
            atlas_mask=atlas_mask,
            atlas_is_mra=atlas_is_mra,
            full_atlas_mask=full_atlas_mask,
            lesion_mask=lesion_mask,
            enhancing_mask=enhancing_mask,
            signal=signal,
            aif_method=str(self.inputs.aif_method),
            initial_criteria=str(self.inputs.criteria),
            n_candidates=int(self.inputs.n_candidates),
            baseline_frames=int(self.inputs.baseline_frames),
            gamma_fit_mode="auto",
            disable_fallback=bool(self.inputs.disable_fallback),
            lesion_distance=lesion_distance,
            n_timepoints=concentration.shape[3],
            field_strength=field_strength,
            synthetic_effective_dose=synthetic_effective_dose,
            voxel_sizes=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
            tr=tr,
        )

    @staticmethod
    def _build_cascade_attempts(
        initial_criteria: str,
        atlas_mask: Optional[npt.NDArray[Any]],
        atlas_is_mra: bool,
        full_atlas_mask: Optional[npt.NDArray[Any]],
        strict_roi_constraint: bool,
    ) -> list[_CascadeAttempt]:
        """Build the ordered DSC AIF search tiers permitted by the inputs."""
        criteria_order = ["strict", "relaxed", "permissive"]
        try:
            start_index = criteria_order.index(initial_criteria)
        except ValueError:
            start_index = 0

        attempts: list[_CascadeAttempt] = []
        if atlas_mask is not None:
            for criteria in criteria_order:
                attempts.append(
                    (criteria, atlas_mask, f"atlas_{criteria}", atlas_is_mra)
                )

        if full_atlas_mask is not None:
            attempts.append(
                ("permissive", full_atlas_mask, "full_atlas_permissive", True)
            )

        if atlas_mask is None or not strict_roi_constraint:
            for criteria in ("relaxed", "permissive"):
                attempts.append((criteria, None, f"brain_{criteria}", False))

        return [
            attempt
            for attempt in attempts
            if criteria_order.index(attempt[0]) >= start_index
        ]

    def _try_trimmed_pool_acceptance(
        self,
        candidates: Sequence[_AIFCandidate],
        concentration: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
    ) -> Optional[_TrimmedPoolAcceptanceResult]:
        """Gate the averaged pool while progressively removing poor tails."""
        pool = list(candidates)
        while len(pool) >= MIN_AIF_CANDIDATES_ACCEPT:
            result = self._try_aif_pool_acceptance(
                pool,
                concentration,
                brain_mask,
                tr,
                baseline_frames,
            )
            if result is not None:
                return _TrimmedPoolAcceptanceResult(result, pool)

            def trim_key(index: int) -> float:
                candidate = pool[index]
                brr = candidate[2].get("baseline_return_ratio", float("nan"))
                if not np.isnan(brr):
                    return float(brr)
                return -float(candidate[0])

            worst_index = min(range(len(pool)), key=trim_key)
            pool.pop(worst_index)

        return None

    def _prepare_tier_candidates(
        self,
        context: _DSCAIFDetectionContext,
        criteria: str,
        roi: Optional[npt.NDArray[Any]],
        is_atlas: bool,
        n_candidates: int,
        baseline_frames: int,
        tissue_ttp: float,
    ) -> _TierCandidateResult:
        """Run candidate preparation and filtering for one cascade tier."""
        search_mask = self._prepare_search_mask(
            context.brain_mask,
            roi,
            context.lesion_mask,
        )
        candidate_voxels, peak_map = self._find_aif_candidate_voxels(
            context.concentration,
            search_mask,
            baseline_frames,
            skip_sss=is_atlas,
        )
        if candidate_voxels.shape[0] == 0:
            return _TierCandidateResult([], 0, {}, None)

        candidates, n_available = self._score_and_select_candidates(
            candidate_voxels,
            context.concentration,
            context.signal,
            peak_map,
            context.brain_mask,
            context.tr,
            baseline_frames,
            criteria,
            tissue_ttp,
            atlas_mask=context.atlas_mask if roi is None else None,
        )
        if not candidates:
            return _TierCandidateResult([], n_available, {}, None)

        candidates, filter_metadata = self._apply_post_scoring_filters(
            candidates,
            context.lesion_mask,
            context.affine,
            context.voxel_sizes,
            lesion_min_distance_mm=context.lesion_distance,
            enhancing_mask=context.enhancing_mask,
        )
        if not candidates:
            return _TierCandidateResult([], n_available, filter_metadata, None)

        candidates, postfilter_diagnostics = self._apply_postscore_aif_filters(
            candidates,
            tr=context.tr,
        )
        if not candidates:
            return _TierCandidateResult(
                [],
                n_available,
                filter_metadata,
                postfilter_diagnostics,
            )

        candidates = sorted(candidates, key=lambda candidate: candidate[0])[
            :n_candidates
        ]
        return _TierCandidateResult(
            candidates,
            n_available,
            filter_metadata,
            postfilter_diagnostics,
        )

    def _try_single_best_fallback(
        self,
        candidates: Sequence[_AIFCandidate],
        concentration: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
        logger: Any,
    ) -> Optional[_SingleBestAIFResult]:
        """Try the lowest-cost candidate when multi-candidate pools fail."""
        if not candidates:
            return None

        best_candidate = min(candidates, key=lambda candidate: candidate[0])
        for roi_size in (1, 3):
            curve, mask = self._compute_roi_averaged_aif(
                [best_candidate],
                concentration,
                brain_mask,
                tr,
                roi_size=roi_size,
            )
            if not self._aif_global_max_in_fp(curve, baseline_frames, tr):
                continue

            brr = self._compute_aif_brr(curve, baseline_frames, tr)
            if not np.isnan(brr) and brr < AIF_AVG_BRR_MIN:
                logger.warning(
                    "AIF cascade: single-best candidate rejected "
                    "(BRR=%.1f%% < %.1f%%); falling through to synthetic AIF",
                    brr,
                    AIF_AVG_BRR_MIN,
                )
                continue

            return _SingleBestAIFResult(curve, mask, roi_size, best_candidate)

        return None

    def _prepare_cached_synthetic_fallback(
        self,
        concentration: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Copy the cached synthetic AIF and add tissue-relative QC metrics."""
        curve = self._cached_synth_aif.copy()
        metrics = dict(self._cached_synth_metrics)
        tissue_max = self._compute_tissue_max_for_synthetic(
            concentration,
            brain_mask,
        )
        if tissue_max is None:
            return curve, metrics

        metrics["synthetic_aif_tissue_p95_delta_r2star"] = round(
            float(tissue_max),
            2,
        )
        peak = metrics.get("synthetic_aif_target_peak", 0.0)
        if tissue_max > 0 and peak > 0:
            metrics["synthetic_aif_peak_to_tissue_ratio"] = round(
                peak / tissue_max,
                2,
            )
        return curve, metrics

    @staticmethod
    def _add_candidates_to_union(
        state: _CascadeUnionState,
        candidates: Sequence[_AIFCandidate],
        level_name: str,
        limit: int,
        filter_metadata: dict[str, Any],
    ) -> None:
        """Add unique tier candidates to the capped post-cascade union pool."""
        added = 0
        for candidate in candidates:
            coordinate = candidate[1]
            if coordinate in state.coordinates:
                continue
            if len(state.candidates) >= limit:
                break
            state.candidates.append(candidate)
            state.coordinates.add(coordinate)
            state.coordinate_origins[coordinate] = level_name
            added += 1

        state.tier_provenance[level_name] = added
        if added > 0 and not state.filter_metadata:
            state.filter_metadata.update(filter_metadata)

    def _try_union_fallback(
        self,
        state: _CascadeUnionState,
        attempts: Sequence[_CascadeAttempt],
        context: _DSCAIFDetectionContext,
        baseline_frames: int,
    ) -> Optional[_UnionAcceptanceResult]:
        """Try the accumulated union and rebuild provenance after trimming."""
        if len(state.candidates) < MIN_AIF_CANDIDATES_ACCEPT:
            return None

        acceptance = self._try_trimmed_pool_acceptance(
            state.candidates,
            context.concentration,
            context.brain_mask,
            context.tr,
            baseline_frames,
        )
        if acceptance is None:
            return None

        candidates = acceptance.candidates
        tier_provenance: dict[str, int] = {}
        for _, coordinate, _ in candidates:
            level_name = state.coordinate_origins.get(coordinate)
            if level_name is not None:
                tier_provenance[level_name] = tier_provenance.get(level_name, 0) + 1

        contributing_levels = [
            level_name
            for _, _, level_name, _ in attempts
            if tier_provenance.get(level_name, 0) > 0
        ]
        detection_level = (
            "+".join(contributing_levels) if contributing_levels else "synthetic"
        )
        result = acceptance.result
        return _UnionAcceptanceResult(
            (result.curve, result.mask, result.roi_size),
            candidates,
            detection_level,
            tier_provenance,
        )

    def _finalize_detected_aif(
        self,
        cascade: _CascadeDetectionResult,
        context: _DSCAIFDetectionContext,
        aif_baseline_frames: int,
    ) -> _FinalizedDetectedAIF:
        """Construct the detected AIF mask and its cascade-aware QC metrics."""
        if cascade.accepted_aif is None or not cascade.candidates:
            raise RuntimeError("Cannot finalize an empty DSC AIF cascade result")

        curve, accepted_mask, accepted_roi_size = cascade.accepted_aif
        mask = np.zeros(context.concentration.shape[:3], dtype=np.uint8)
        mask[accepted_mask] = 1
        for _, coordinate, _ in cascade.candidates:
            mask[coordinate[0], coordinate[1], coordinate[2]] = 1

        baseline_frames, baseline_diagnostics = self._refine_selected_aif_baseline(
            curve,
            preliminary_baseline=aif_baseline_frames,
            bootstrap_baseline=context.baseline_frames,
            tr=context.tr,
        )

        mean_cost = float(np.mean([cost for cost, _, _ in cascade.candidates]))
        quality_score = min(1.0, 1.0 / (1.0 + mean_cost))
        curve, qc_metrics = self._finalize_aif_metrics(
            curve,
            cascade.candidates,
            context.concentration,
            context.brain_mask,
            context.affine,
            context.tr,
            context.n_candidates,
            cascade.n_available,
            baseline_frames=baseline_frames,
        )
        qc_metrics.update(baseline_diagnostics)
        qc_metrics.update(cascade.filter_metadata)
        if cascade.initial_level is not None:
            qc_metrics["cascade_downgraded"] = (
                cascade.initial_level != cascade.detection_level
            )
            qc_metrics["cascade_initial_level"] = cascade.initial_level
            qc_metrics["cascade_final_level"] = cascade.detection_level
        qc_metrics["cascade_tiers_attempted"] = cascade.tiers_attempted
        qc_metrics["cascade_accepted_roi_size"] = accepted_roi_size
        qc_metrics["aif_quality"] = "detected"
        qc_metrics["aif_tier_provenance"] = dict(cascade.tier_provenance)
        qc_metrics["aif_n_after_filters"] = sum(
            diagnostics.get("aif_n_after_filters", 0)
            for tier, diagnostics in cascade.postfilter_diagnostics.items()
            if cascade.tier_provenance.get(tier, 0) > 0
        )
        qc_metrics["aif_n_filtered_peakspread"] = sum(
            diagnostics.get("aif_n_filtered_peakspread", 0)
            for diagnostics in cascade.postfilter_diagnostics.values()
        )
        qc_metrics["aif_n_filtered_tail"] = sum(
            diagnostics.get("aif_n_filtered_tail", 0)
            for diagnostics in cascade.postfilter_diagnostics.values()
        )

        peak_frames = [
            float(metrics.get("peak_time_frame", float("nan")))
            for _, _, metrics in cascade.candidates
        ]
        peak_frames = [value for value in peak_frames if np.isfinite(value)]
        tail_fractions = [
            float(metrics.get("tail_fraction_late", float("nan")))
            for _, _, metrics in cascade.candidates
        ]
        tail_fractions = [
            value for value in tail_fractions if np.isfinite(value)
        ]
        if peak_frames:
            qc_metrics["aif_peak_time_spread_frames"] = (
                max(peak_frames) - min(peak_frames)
            )
        if tail_fractions:
            qc_metrics["aif_tail_max_fraction"] = max(tail_fractions)

        spr, first_pass_frame, _ = self._compute_aif_spr(
            curve,
            aif_baseline_frames,
            context.tr,
        )
        qc_metrics["averaged_aif_spr"] = round(spr, 1)
        qc_metrics["averaged_aif_fp_peak_frame"] = first_pass_frame
        qc_metrics["averaged_aif_final_roi_size"] = accepted_roi_size
        return _FinalizedDetectedAIF(
            curve,
            mask,
            quality_score,
            qc_metrics,
            baseline_frames,
        )

    @staticmethod
    def _refine_selected_aif_baseline(
        curve: npt.NDArray[Any],
        *,
        preliminary_baseline: int,
        bootstrap_baseline: int,
        tr: float,
    ) -> tuple[int, dict[str, Any]]:
        """Refine the preliminary arrival estimate from the selected AIF.

        The whole-brain percentile estimate is available before AIF selection
        and therefore remains the preliminary authority. Once an arterial
        curve has been selected, its cleaner bolus signature may replace that
        estimate when the preliminary detector only returned the bootstrap
        floor or when the arterial ensemble has high-confidence disagreement.

        Parameters
        ----------
        curve : numpy.ndarray
            Selected measured or externally supplied AIF curve.
        preliminary_baseline : int
            Whole-brain percentile arrival estimate.
        bootstrap_baseline : int
            Configured baseline used to bootstrap motion correction and signal
            conversion.
        tr : float
            Repetition time in seconds.

        Returns
        -------
        tuple of (int, dict)
            Final baseline frame count and refinement diagnostics.
        """
        import logging

        from pmriprep.utils.baseline_detection import detect_baseline_ensemble

        logger = logging.getLogger(__name__)
        diagnostics: dict[str, Any] = {
            "baseline_preliminary_frames": int(preliminary_baseline),
            "baseline_refinement_source": "whole_brain_percentile",
        }
        if len(curve) < 3:
            diagnostics["baseline_refinement_method"] = "short_aif"
            return preliminary_baseline, diagnostics

        try:
            refined, ensemble = detect_baseline_ensemble(
                np.maximum(curve, 0.0),
                tr=tr,
                min_baseline=2,
            )
        except Exception:
            diagnostics["baseline_refinement_method"] = "ensemble_unavailable"
            diagnostics["baseline_frames_used"] = int(preliminary_baseline)
            return preliminary_baseline, diagnostics

        confidence = str(ensemble.get("confidence", "low"))
        diagnostics.update(
            baseline_refined_candidate_frames=int(refined),
            baseline_refinement_confidence=confidence,
            baseline_refinement_method=ensemble.get("method", "ensemble"),
        )
        if refined < 2 or refined > len(curve) // 2:
            diagnostics["baseline_refinement_method"] = "ensemble_out_of_range"
            diagnostics["baseline_frames_used"] = int(preliminary_baseline)
            return preliminary_baseline, diagnostics

        final = preliminary_baseline
        if preliminary_baseline <= bootstrap_baseline and refined > bootstrap_baseline:
            final = refined
            diagnostics["baseline_refinement_source"] = "selected_aif_fallback_recovery"
        elif confidence == "high" and abs(refined - preliminary_baseline) > 5:
            final = refined
            diagnostics["baseline_refinement_source"] = "selected_aif_high_confidence"
            logger.info(
                "Selected AIF refines DSC baseline: AIF=%d (confidence=%s), "
                "whole-brain percentile=%d",
                refined,
                confidence,
                preliminary_baseline,
            )
        elif refined > preliminary_baseline + 3:
            diagnostics["baseline_refinement_disagrees"] = True
            logger.warning(
                "Selected-AIF baseline estimate differs: AIF=%d, "
                "whole-brain percentile=%d; retaining the preliminary estimate.",
                refined,
                preliminary_baseline,
            )

        diagnostics["baseline_frames_used"] = int(final)
        return int(final), diagnostics

    def _prepare_detection_baseline(
        self,
        context: _DSCAIFDetectionContext,
        logger: Any,
    ) -> int:
        """Detect arterial baseline timing and initialize synthetic fallback."""
        brain_concentration = context.concentration[context.brain_mask > 0.5]
        if brain_concentration.ndim > 1 and brain_concentration.shape[0] > 100:
            from pmriprep.utils.baseline_detection import detect_baseline_ensemble

            p90_curve, p99_curve = np.percentile(
                brain_concentration,
                [90, 99],
                axis=0,
            )
            mean_curve = np.mean(brain_concentration, axis=0)
            baseline_frames, diagnostics = detect_baseline_ensemble(
                p99_curve,
                tr=context.tr,
                min_baseline=max(5, int(10.0 / max(context.tr, 0.1))),
                additional_signals=[p90_curve, mean_curve],
            )
            logger.debug(
                "Ensemble baseline detection: arrival=%d, conf=%s, method=%s",
                baseline_frames,
                diagnostics.get("confidence", "unknown"),
                diagnostics.get("method", "unknown"),
            )
        else:
            baseline_frames = context.baseline_frames

        self._cached_synth_aif, self._cached_synth_metrics = (
            self._generate_synthetic_aif(
                context.n_timepoints,
                context.tr,
                baseline_frames,
                field_strength=context.field_strength,
                effective_dose=context.synthetic_effective_dose,
            )
        )
        return baseline_frames

    def _run_candidate_cascade(
        self,
        context: _DSCAIFDetectionContext,
        aif_baseline_frames: int,
        logger: Any,
    ) -> _CascadeDetectionResult:
        """Run ordered candidate tiers and return the accepted cascade state."""
        conc_data = context.concentration
        brain_mask = context.brain_mask
        atlas_mask = context.atlas_mask
        atlas_is_mra = context.atlas_is_mra
        full_atlas_mask = context.full_atlas_mask
        aif_method = context.aif_method
        initial_criteria = context.initial_criteria
        n_candidates = context.n_candidates
        tr = context.tr
        # Initialize
        detection_level = "synthetic"
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        n_available = 0
        filter_meta: dict[str, Any] = {}
        cascade_initial_level: Optional[str] = None
        accepted_aif: Optional[tuple[npt.NDArray[Any], npt.NDArray[Any], int]] = None
        tiers_attempted: list[str] = []
        # Post-cascade fallback pool. Each tier's kinetic-filter
        # survivors are appended (capped at n_candidates total). The
        # union does NOT gate the cascade — it is only consulted as a
        # fallback if every tier's averaged-pool gate fails.
        union_state = _CascadeUnionState([], set(), {}, {}, {})
        # Per-tier post-filter diagnostics (n_filtered_* counts, pool
        # spread, max tail). Recorded even on tiers the union does not
        # accept from, so the synthetic-fallback branch can surface why
        # each tier failed.
        postfilter_diags_per_tier: dict[str, dict[str, Any]] = {}
        spatial_filter_diags_per_tier: dict[str, dict[str, Any]] = {}
        if aif_method == "synthetic":
            pass  # Go directly to synthetic
        else:
            attempts = self._build_cascade_attempts(
                initial_criteria,
                atlas_mask,
                atlas_is_mra,
                full_atlas_mask,
                bool(self.inputs.strict_roi_constraint),
            )

            # Compute tissue-TTP reference once — independent of tier
            # because it operates on the whole brain mask. Used by the
            # existing per-tier scorer AND by the new post-score
            # kinetic+anchor filter.
            tissue_ttp = self._calculate_tissue_ttp_reference(
                conc_data, brain_mask, aif_baseline_frames
            )

            for crit, roi, level_name, is_atlas in attempts:
                tiers_attempted.append(level_name)
                frac = _CASCADE_N_CANDIDATES_FRACTION.get(crit, 0.6)
                level_n_candidates = max(
                    MIN_AIF_CANDIDATES_ACCEPT,
                    math.ceil(frac * n_candidates),
                )

                tier_candidates = self._prepare_tier_candidates(
                    context,
                    crit,
                    roi,
                    is_atlas,
                    level_n_candidates,
                    aif_baseline_frames,
                    tissue_ttp,
                )
                scored = tier_candidates.candidates
                f_meta = tier_candidates.filter_metadata
                if "n_distance_excluded" in f_meta:
                    spatial_filter_diags_per_tier[level_name] = dict(f_meta)
                if tier_candidates.postfilter_diagnostics is not None:
                    postfilter_diags_per_tier[level_name] = (
                        tier_candidates.postfilter_diagnostics
                    )
                if not scored:
                    continue

                if cascade_initial_level is None:
                    cascade_initial_level = level_name
                # Sort survivors by cost (lowest first).
                scored.sort(key=lambda x: x[0])

                # Accumulate this tier's survivors into the global
                # union (capped at n_candidates total across all tiers
                # — broader than the per-tier ``level_n_candidates``
                # used by the per-tier gate below). The union is the
                # POST-CASCADE fallback pool only and does NOT gate
                # the cascade. Looser tiers always run if the current
                # tier's averaged-pool gate fails, so the cascade
                # is fully exhausted before any fallback.  Synthetic
                # AIF is the LAST RESORT — reachable only after every
                # tier's per-tier gate AND the union/single-best
                # fallbacks have failed.
                self._add_candidates_to_union(
                    union_state,
                    scored,
                    level_name,
                    n_candidates,
                    f_meta,
                )

                # Per-tier averaged-pool gate with progressive worst-BRR
                # trim.  Asks "do this tier's voxels behave as a
                # coherent arterial pool together?" — if not, drop the
                # candidate with the most-negative BRR (worst tail
                # contaminator) and retry, down to
                # MIN_AIF_CANDIDATES_ACCEPT.  If the tier can't reach
                # an acceptable pool, the cascade falls through to the
                # next (looser) tier.
                tier_acceptance = self._try_trimmed_pool_acceptance(
                    scored[:level_n_candidates],
                    conc_data,
                    brain_mask,
                    tr,
                    aif_baseline_frames,
                )
                if tier_acceptance is not None:
                    tier_result = tier_acceptance.result
                    tier_pool = tier_acceptance.candidates
                    accepted_aif = (
                        tier_result.curve,
                        tier_result.mask,
                        tier_result.roi_size,
                    )
                    top_candidates = list(tier_pool)
                    n_available = len(tier_pool)
                    detection_level = level_name
                    filter_meta = dict(f_meta)
                    union_state.filter_metadata.clear()
                    union_state.filter_metadata.update(f_meta)
                    union_state.tier_provenance.clear()
                    union_state.tier_provenance[level_name] = len(tier_pool)

                if accepted_aif is not None:
                    break  # cascade complete — tier accepted

            # Post-cascade fallback 1: union pool with progressive
            # worst-BRR trim.
            # Only runs if EVERY tier's averaged-pool gate failed
            # above. Pools kinetic-filter survivors across tiers, then
            # mirrors the per-tier trim semantics: drop the most-
            # negative-BRR voxel one at a time until the averaged pool
            # passes or the pool reaches MIN_AIF_CANDIDATES_ACCEPT.
            if accepted_aif is None:
                union_acceptance = self._try_union_fallback(
                    union_state,
                    attempts,
                    context,
                    aif_baseline_frames,
                )
                if union_acceptance is not None:
                    accepted_aif = union_acceptance.accepted_aif
                    top_candidates = list(union_acceptance.candidates)
                    n_available = len(union_acceptance.candidates)
                    detection_level = union_acceptance.detection_level
                    union_state.tier_provenance.clear()
                    union_state.tier_provenance.update(
                        union_acceptance.tier_provenance
                    )
                    filter_meta = dict(union_state.filter_metadata)

            # Single-best fallback: union has voxels but per-tier and
            # union averaged-pool gates all rejected. Try the lowest-cost union
            # voxel; the relaxed gate uses global_max_in_fp and
            # AIF_AVG_BRR_MIN.
            if accepted_aif is None and len(union_state.candidates) > 0:
                single_best = self._try_single_best_fallback(
                    union_state.candidates,
                    conc_data,
                    brain_mask,
                    tr,
                    aif_baseline_frames,
                    logger,
                )
                if single_best is not None:
                    accepted_aif = (
                        single_best.curve,
                        single_best.mask,
                        single_best.roi_size,
                    )
                    best_only = [single_best.candidate]
                    top_candidates = best_only
                    n_available = len(union_state.candidates)
                    # Use the chosen voxel's source tier (looked up
                    # via the per-coord origin map) rather than the
                    # last attempted tier, because the single-best
                    # voxel typically came from a stricter tier than
                    # the last one we tried.
                    best_coord = best_only[0][1]
                    detection_level = union_state.coordinate_origins.get(
                        best_coord,
                        tiers_attempted[-1] if tiers_attempted else "single_best",
                    )
                    # Provenance must describe the single emitted voxel, not the
                    # full pre-fallback union pool the trim loop left in
                    # union_tier_provenance.
                    union_state.tier_provenance.clear()
                    union_state.tier_provenance[detection_level] = 1
                    filter_meta = dict(union_state.filter_metadata)
                    logger.warning(
                        "AIF cascade: using single-best candidate "
                        "(union failed averaged-pool validation)"
                    )

        cascade = _CascadeDetectionResult(
            accepted_aif,
            top_candidates,
            n_available,
            detection_level,
            filter_meta,
            cascade_initial_level,
            tiers_attempted,
            dict(union_state.tier_provenance),
            postfilter_diags_per_tier,
            spatial_filter_diags_per_tier,
        )

        return cascade

    def _serialize_detection_outputs(self, state: _FinalAIFState) -> None:
        """Write final DSC AIF artifacts and populate Nipype results."""
        import nibabel as nib  # type: ignore[import-untyped]

        out_curve = self._gen_outfilename("_aif_curve.txt")
        np.savetxt(out_curve, state.curve)
        out_raw = ""
        if state.raw_curve is not None:
            out_raw = self._gen_outfilename("_raw_aif_curve.txt")
            np.savetxt(out_raw, state.raw_curve)

        out_mask = self._gen_outfilename("_aif_mask.nii.gz")
        nib.save(nib.Nifti1Image(state.mask, state.affine), out_mask)  # type: ignore[attr-defined]
        out_measured = self._gen_outfilename("_measured_aif_curve.txt")
        np.savetxt(out_measured, state.measured_curve)
        out_gamma = self._gen_outfilename("_gamma_aif_curve.txt")
        np.savetxt(
            out_gamma,
            state.gamma_curve if state.gamma_curve is not None else state.curve,
        )
        if state.gamma_curve is None:
            state.qc_metrics["gamma_aif_fallback_to_raw"] = True

        self._results = {
            "out_aif_curve": out_curve,
            "out_aif_mask": out_mask,
            "out_gamma_aif_curve": out_gamma,
            "out_measured_aif_curve": out_measured,
            "out_raw_aif_curve": out_raw,
            "out_aif_coord": state.qc_metrics.get("aif_voxel_coord", []),
            "aif_curve": [float(value) for value in state.curve],
            "detection_level": state.detection_level,
            "quality_score": state.quality_score,
            "candidate_count": state.candidate_count,
            "gamma_fit_applied": state.gamma_fit_applied,
            "aif_baseline_frames": state.baseline_frames,
            "qc_metrics": state.qc_metrics,
        }

    def _finalize_post_cascade(
        self,
        context: _DSCAIFDetectionContext,
        cascade: _CascadeDetectionResult,
        aif_baseline_frames: int,
        logger: Any,
    ) -> _FinalAIFState:
        """Finalize gamma fitting, fallback, and QC after the candidate cascade."""
        conc_data = context.concentration
        bootstrap_baseline_frames = context.baseline_frames
        baseline_frames = context.baseline_frames
        candidate_baseline_frames: Optional[int] = None
        detection_level = cascade.detection_level
        top_candidates = cascade.candidates
        aif_mask_arr = np.zeros(conc_data.shape[:3], dtype=np.uint8)

        if cascade.accepted_aif is not None and top_candidates:
            finalized = self._finalize_detected_aif(
                cascade,
                context,
                aif_baseline_frames,
            )
            aif_curve = finalized.curve
            aif_mask_arr = finalized.mask
            quality_score = finalized.quality_score
            qc_metrics = finalized.qc_metrics
            baseline_frames = finalized.baseline_frames
            candidate_baseline_frames = finalized.baseline_frames
        elif not context.disable_fallback:
            if context.aif_method != "synthetic":
                distance_excluded = sum(
                    int(metrics.get("n_distance_excluded", 0))
                    for metrics in cascade.spatial_filter_diagnostics.values()
                )
                distance_detail = (
                    f" ({distance_excluded} candidates excluded by lesion distance)"
                    if distance_excluded
                    else ""
                )
                logger.warning(
                    "DSC AIF detection exhausted all candidate tiers; "
                    "using the synthetic AIF fallback%s",
                    distance_detail,
                )
            aif_curve, synth_metrics = self._prepare_cached_synthetic_fallback(
                conc_data,
                context.brain_mask,
            )
            detection_level = "synthetic"
            quality_score = 0.3
            qc_metrics = {}
            _stamp_synthetic_fallback_common(
                qc_metrics,
                reason="no_candidates",
                tiers_attempted=cascade.tiers_attempted,
            )
            qc_metrics.update(synth_metrics)
            _clear_stale_real_aif_keys(qc_metrics)
            baseline_frames = aif_baseline_frames
        else:
            logger.error(
                "DSC AIF detection exhausted all candidate tiers and "
                "synthetic fallback is disabled"
            )
            raise RuntimeError(
                "AIF detection failed at all levels and "
                "synthetic fallback is disabled"
            )

        qc_metrics["detection_method"] = detection_level
        if cascade.spatial_filter_diagnostics:
            qc_metrics["cascade_spatial_filter_diags"] = dict(
                cascade.spatial_filter_diagnostics
            )
        qc_metrics["quality_score"] = quality_score
        if detection_level != "synthetic":
            qc_metrics["n_aif_voxels_selected"] = len(top_candidates)
        qc_metrics["s0_rejection_available"] = context.signal is not None
        qc_metrics["tr_used"] = context.tr
        qc_metrics["baseline_frames_initial"] = baseline_frames
        qc_metrics["aif_baseline_frames"] = aif_baseline_frames
        qc_metrics["baseline_frames_used"] = baseline_frames
        qc_metrics["baseline_detection_method"] = "auto_p99"

        if detection_level == "synthetic":
            qc_metrics["synthetic_aif_warnings"] = [
                "Synthetic AIF used - automatic detection failed",
                "Relative quantification only (rCBV, rCBF normalized to contralateral)",
                "TTP and Tmax depend on the synthetic AIF timing and are qualitative only",
                "Manual AIF selection recommended for clinical use",
            ]

        measured_aif = aif_curve.copy()
        if detection_level == "synthetic":
            qc_metrics["cbv_auc_denominator_is_synthetic"] = True

        gamma_fit_applied = False
        raw_aif_curve = None
        gamma_fitted_curve = None
        if context.gamma_fit_mode != "off" and detection_level != "synthetic":
            gamma_fit_applied, raw_aif_curve, aif_curve, gamma_fitted_curve = (
                self._apply_gamma_fit_if_appropriate(
                    aif_curve,
                    context.gamma_fit_mode,
                    baseline_frames,
                    context.tr,
                    qc_metrics,
                )
            )
        elif context.gamma_fit_mode != "off" and detection_level == "synthetic":
            qc_metrics["gamma_fit_applied"] = False
            qc_metrics["gamma_fit_rejected_reason"] = "synthetic_aif"

        if (
            not gamma_fit_applied
            and not context.disable_fallback
            and detection_level != "synthetic"
        ):
            gamma_r2 = qc_metrics.get("gamma_fit_r_squared")
            gamma_fit_converged = qc_metrics.get("gamma_fit_converged")
            if (
                gamma_fit_converged is True
                and gamma_r2 is not None
                and gamma_r2 < GAMMA_R2_SYNTHETIC_FALLBACK
            ):
                logger.warning(
                    "Gamma fit R²=%.3f < %.1f; falling back to synthetic AIF",
                    gamma_r2,
                    GAMMA_R2_SYNTHETIC_FALLBACK,
                )
                rejected_measured_aif = {
                    "detection_level": detection_level,
                    "baseline_frames": candidate_baseline_frames,
                    "arrival_time_seconds": qc_metrics.get("arrival_time_seconds"),
                    "peak_frame": qc_metrics.get("averaged_aif_fp_peak_frame"),
                    "gamma_fit_r_squared": gamma_r2,
                    "gamma_fit_auc_ratio": qc_metrics.get("gamma_fit_auc_ratio"),
                    "gamma_fit_rejected_reason": qc_metrics.get(
                        "gamma_fit_rejected_reason"
                    ),
                }
                aif_curve, synth_metrics = self._prepare_cached_synthetic_fallback(
                    conc_data,
                    context.brain_mask,
                )
                baseline_frames = aif_baseline_frames
                detection_level = "synthetic"
                quality_score = 0.3
                aif_mask_arr[:] = 0
                _clear_stale_real_aif_keys(qc_metrics)
                _stamp_synthetic_fallback_common(
                    qc_metrics,
                    reason="gamma_r2_floor",
                    tiers_attempted=cascade.tiers_attempted,
                )
                qc_metrics["detection_method"] = detection_level
                qc_metrics["quality_score"] = quality_score
                qc_metrics["gamma_r2_at_fallback"] = gamma_r2
                qc_metrics.update(synth_metrics)
                qc_metrics["rejected_measured_aif"] = {
                    key: value
                    for key, value in rejected_measured_aif.items()
                    if value is not None
                }
                qc_metrics["synthetic_aif_warnings"] = [
                    "Synthetic AIF used - gamma fit R2 below threshold",
                    "Relative quantification only (rCBV, rCBF normalized to contralateral)",
                    "TTP and Tmax depend on the synthetic AIF timing and are qualitative only",
                    "Manual AIF selection recommended for clinical use",
                ]
                gamma_fitted_curve = None
                raw_aif_curve = None
                measured_aif = aif_curve.copy()
                qc_metrics["cbv_auc_denominator_is_synthetic"] = True

        _flag_retained_below_floor(
            qc_metrics,
            disable_fallback=context.disable_fallback,
            detection_level=detection_level,
            gamma_fit_applied=gamma_fit_applied,
        )

        if (
            (
                qc_metrics.get("cascade_downgraded", False)
                or detection_level == "synthetic"
            )
            and cascade.postfilter_diagnostics
        ):
            qc_metrics["cascade_postfilter_diags"] = dict(
                cascade.postfilter_diagnostics
            )

        # Final analysis provenance must describe the curve that leaves this
        # interface, not a measured candidate discarded during fallback.
        qc_metrics["bootstrap_baseline_frames"] = bootstrap_baseline_frames
        qc_metrics["baseline_frames_initial"] = bootstrap_baseline_frames
        qc_metrics["aif_baseline_frames"] = baseline_frames
        qc_metrics["baseline_frames_used"] = baseline_frames
        qc_metrics["n_baseline_frames"] = baseline_frames
        qc_metrics["baseline_detection_method"] = (
            "selected_aif_refined"
            if detection_level != "synthetic"
            else "whole_brain_ensemble"
        )
        if candidate_baseline_frames is not None:
            qc_metrics["candidate_baseline_frames"] = candidate_baseline_frames
        if detection_level == "synthetic":
            synthetic_arrival = qc_metrics.get("synthetic_aif_arrival_frame")
            if synthetic_arrival is not None:
                arrival_seconds = float(synthetic_arrival) * context.tr
                qc_metrics["arrival_time_seconds"] = arrival_seconds
                aif_metrics = dict(qc_metrics.get("aif_metrics") or {})
                aif_metrics["arrival"] = arrival_seconds
                qc_metrics["aif_metrics"] = aif_metrics

        # Canonical AIF metrics describe the curve emitted by this interface.
        # Gamma fitting can alter its peak after candidate-level metrics were
        # assembled, so refresh the final amplitude at the state boundary.
        aif_metrics = dict(qc_metrics.get("aif_metrics") or {})
        aif_metrics["peak_conc"] = float(np.max(aif_curve))
        qc_metrics["aif_metrics"] = aif_metrics

        return _FinalAIFState(
            aif_curve,
            aif_mask_arr,
            measured_aif,
            raw_aif_curve,
            gamma_fitted_curve,
            context.affine,
            detection_level,
            quality_score,
            gamma_fit_applied,
            baseline_frames,
            len(top_candidates),
            qc_metrics,
        )

    # -----------------------------------------------------------------
    # Step 10: Main pipeline  (_run_interface)
    # -----------------------------------------------------------------

    def _run_interface(self, runtime: Any) -> Any:
        """Execute the AIF detection pipeline.

        1. Build attempts list: atlas (strict/relaxed/permissive) then
           brain (relaxed/permissive).
        2. Each attempt: prepare mask -> find candidates -> score ->
           post-scoring distance filtering and laterality accounting.
        3. On success, finalize metrics and break.
        4. Synthetic fallback if all attempts fail.
        """
        import logging

        _logger = logging.getLogger(__name__)

        # ===== External AIF: bypass all detection if provided =====
        if isdefined(self.inputs.external_aif_file) and self.inputs.external_aif_file:
            return self._handle_external_aif(runtime)

        context = self._load_detection_context(_logger)
        aif_baseline_frames = self._prepare_detection_baseline(context, _logger)
        cascade = self._run_candidate_cascade(
            context, aif_baseline_frames, _logger,
        )
        self._serialize_detection_outputs(
            self._finalize_post_cascade(
                context,
                cascade,
                aif_baseline_frames,
                _logger,
            )
        )

        return runtime

    @staticmethod
    def _apply_gamma_fit_if_appropriate(
        aif_curve: npt.NDArray[Any],
        gamma_fit_mode: str,
        baseline_frames: int,
        tr: float,
        qc_metrics: dict[str, Any],
    ) -> tuple[bool, Optional[npt.NDArray[Any]], npt.NDArray[Any], Optional[npt.NDArray[Any]]]:
        """Attempt gamma-variate fitting and decide whether to apply it.

        Parameters
        ----------
        aif_curve : np.ndarray
            Current AIF curve (may be modified if fit is applied).
        gamma_fit_mode : str
            ``"on"`` applies when the fit converges with R² > 0.5;
            ``"auto"`` uses the rule-based policy.
        baseline_frames : int
            Number of pre-bolus baseline frames.
        tr : float
            Repetition time in seconds.
        qc_metrics : dict
            QC metrics dict (modified in place).

        Returns
        -------
        tuple[bool, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]
            (gamma_fit_applied, raw_aif_curve, aif_curve, gamma_fitted_curve)
            raw_aif_curve is the original AIF when fit is applied,
            or the fitted curve when fit is rejected (for QC overlay).
            gamma_fitted_curve is always the fitted curve when fit converges
            (regardless of accept/reject decision), for AUC CBV computation.
        """
        from pmriprep.utils.dsc_aif import (
            fit_gamma_variate,
            should_apply_gamma_fit,
        )

        try:
            original_aif = aif_curve.copy()
            fit_result = fit_gamma_variate(
                aif_curve=aif_curve,
                baseline_frames=baseline_frames,
                remove_recirculation=True,
                preserve_auc=True,
                tr=tr,
            )
            qc_metrics["gamma_fit_converged"] = fit_result.fit_converged
            if fit_result.failure_reason is not None:
                qc_metrics["gamma_fit_failure_reason"] = fit_result.failure_reason
            if fit_result.fit_converged:
                qc_metrics["gamma_fit_r_squared"] = fit_result.r_squared
                qc_metrics["gamma_fit_auc_ratio"] = fit_result.auc_ratio

            if gamma_fit_mode == "on":
                # ``on`` mode uses the convergence and R² gate.
                if fit_result.fit_converged and fit_result.r_squared > 0.5:
                    qc_metrics["gamma_fit_applied"] = True
                    qc_metrics["gamma_fit_mode"] = "on"
                    qc_metrics["gamma_recirculation_removed"] = (
                        fit_result.recirculation_removed
                    )
                    return True, original_aif, fit_result.fitted_curve, fit_result.fitted_curve
                else:
                    qc_metrics["gamma_fit_applied"] = False
                    qc_metrics["gamma_fit_mode"] = "on"
                    qc_metrics["gamma_fit_rejected_reason"] = (
                        fit_result.failure_reason or "optimizer_not_converged"
                        if not fit_result.fit_converged
                        else f"low_r_squared_{fit_result.r_squared:.3f}"
                    )
                    gamma_curve = fit_result.fitted_curve if fit_result.fit_converged else None
                    return False, None, aif_curve, gamma_curve

            # Auto mode
            decision, detail_metrics = should_apply_gamma_fit(
                original_aif=original_aif,
                fit_result=fit_result,
                baseline_frames=baseline_frames,
            )
            qc_metrics["gamma_fit_mode"] = "auto"
            qc_metrics["gamma_fit_auto_metrics"] = detail_metrics

            if decision.apply:
                qc_metrics["gamma_fit_applied"] = True
                qc_metrics["gamma_fit_auto_reason"] = decision.reason
                qc_metrics["gamma_fit_auto_defect"] = decision.defect_detected
                qc_metrics["gamma_recirculation_removed"] = (
                    fit_result.recirculation_removed
                )
                return True, original_aif, fit_result.fitted_curve, fit_result.fitted_curve
            else:
                qc_metrics["gamma_fit_applied"] = False
                qc_metrics["gamma_fit_rejected_reason"] = decision.reason
                if not fit_result.fit_converged:
                    return False, None, aif_curve, None
                # Save fitted curve for QC overlay (rejected fit shown dashed)
                qc_metrics["gamma_fit_rejection_curve_available"] = True
                return False, fit_result.fitted_curve, aif_curve, fit_result.fitted_curve

        except Exception:
            qc_metrics["gamma_fit_applied"] = False
            qc_metrics["gamma_fit_converged"] = False
            qc_metrics["gamma_fit_failure_reason"] = "fit_exception"
            qc_metrics["gamma_fit_rejected_reason"] = "exception"
            return False, None, aif_curve, None

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
