"""AIF (Arterial Input Function) detection interfaces for DSC-MRI.

This module implements Nipype interfaces for AIF detection in DSC perfusion
processing, with multi-level detection strategies (atlas-based, relaxed
criteria) and comprehensive quality metrics.

Implements search-mask preparation, adaptive-percentile candidate pre-filtering,
physical brain-edge and posterior-midline exclusions, tissue-TTP
cross-validation, cost-function scoring, candidate-pool averaging, and
Savitzky-Golay smoothing.
"""

import math
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.ndimage import (  # type: ignore[import-untyped]
    binary_fill_holes,
    distance_transform_edt,
)
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    isdefined,
    traits,
)

from pmriprep.utils.aif_laterality import describe_tumor_relative_aif_candidates
from pmriprep.utils.dsc_constants import (
    AIF_PEAK_CEILING_3T,
    AIF_PEAK_FLOOR_3T,
    AIF_MIN_PEAK,
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
)

# Seconds after baseline end before first-pass window begins.
# 3 s is short enough to capture early peaks in fast-TR acquisitions while
# candidates with spurious noise peaks are already rejected by the TTP
# minimum check in ``_score_candidate_cost`` (strict ≥ 5 s, relaxed ≥ 3 s).
_FP_GAP_SECONDS: float = 3.0

#: Minimum physical distance from the outer in-plane brain boundary for DSC
#: AIF candidates. The original brain mask is intersected afterward,
#: so internal holes remain excluded without creating additional exclusion rims.
_BRAIN_EDGE_EXCLUSION_MM: Final[float] = 8.0
_WB_POSTERIOR_FRACTION: Final[float] = 0.60
_WB_MIDLINE_HALF_WIDTH_FRACTION: Final[float] = 0.15
_WB_SYMMETRY_MAX_ANGLE_DEGREES: Final[float] = 25.0
_WB_SYMMETRY_MAX_SHIFT_MM: Final[float] = 10.0

# Minimum number of AIF candidates before accepting a cascade level.
# If a level finds fewer than this, the cascade continues to more relaxed
# criteria to build a larger candidate pool (quality gating is downstream).
MIN_AIF_CANDIDATES_ACCEPT: int = 3

#: Gamma peak ratio band for cascade quality gate.
#: If averaged AIF peak ratio falls outside this range, continue cascading
#: to the next tier (keeping the current result as fallback).
GAMMA_PEAK_RATIO_CASCADE_MIN: float = 0.88
GAMMA_PEAK_RATIO_CASCADE_MAX: float = 1.12

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
    rejected: Optional["_RejectedAIF"] = None


class _FinalizedDetectedAIF(NamedTuple):
    """Detected AIF arrays, QC state, and updated baseline."""

    curve: npt.NDArray[Any]
    mask: npt.NDArray[Any]
    quality_score: float
    qc_metrics: dict[str, Any]
    baseline_frames: int


class _RejectedAIF(NamedTuple):
    """Measured curve and matching spatial support retained for failure QC."""

    curve: npt.NDArray[Any]
    mask: npt.NDArray[Any]
    qc: dict[str, Any]


class _TissueBaseline(NamedTuple):
    """Whole-brain timing estimate and its own detector diagnostics."""

    frames: int
    diagnostics: dict[str, Any]


class _FinalAIFState(NamedTuple):
    """Available analysis input and separate diagnostic curves for serialization."""

    curve: Optional[npt.NDArray[Any]]
    mask: Optional[npt.NDArray[Any]]
    gamma_curve: Optional[npt.NDArray[Any]]
    rejected: Optional[_RejectedAIF]
    affine: npt.NDArray[Any]
    detection_level: str
    quality_score: Optional[float]
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
    initial_criteria: str
    n_candidates: int
    baseline_frames: int
    lesion_distance: float
    n_timepoints: int
    voxel_sizes: tuple[float, float, float]
    tr: float


_CascadeAttempt = tuple[str, Optional[npt.NDArray[Any]], str, bool]


# ---------------------------------------------------------------------------
# Top-level AIF quality helpers
# ---------------------------------------------------------------------------


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
        Peak ratio (fitted/measured), or None if fitting fails, R² is below
        0.5, or the measured post-baseline peak is below 1e-6.
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
        desc="Optional AIF search mask; measured candidates must lie inside it",
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
        desc="Agent's standard dose in mmol Gd/kg "
             "(0.1=standard, 0.05=gadopiclenol, 0.04=gadoquatrane)",
    )
    gd_multiplicity = traits.Int(
        1,
        usedefault=True,
        desc="Gd3+ ions per molecule (1=monomeric, 4=gadoquatrane). "
             "Descriptive metadata; dose is already expressed per Gd.",
    )
    field_strength = traits.Float(
        3.0,
        usedefault=True,
        desc="Magnetic field strength in Tesla (from BIDS MagneticFieldStrength). "
             "Used for field-dependent arterial candidate thresholds.",
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
    out_gamma_aif_curve = File(exists=True, desc="Converged gamma model for QC")
    out_rejected_aif_curve = File(exists=True, desc="Rejected measured curve for QC")
    out_rejected_aif_mask = File(exists=True, desc="Voxels contributing to the rejected QC curve")
    out_aif_coord = traits.List(traits.Int, desc="Best AIF voxel coordinate [x, y, z]")
    aif_curve = traits.List(traits.Float, desc="AIF concentration curve values")
    detection_level = traits.Str(
        desc="Accepted atlas/brain tier or '+'-joined contributing tiers; "
             "'unavailable' or 'external' for absent or supplied curves"
    )
    quality_score = traits.Float(desc="Overall AIF quality score (0-1)")
    candidate_count = traits.Int(desc="Number of retained candidate records at finalization")
    aif_baseline_frames = traits.Int(
        desc="Detected pre-bolus baseline frames for downstream use",
    )
    qc_metrics = traits.Dict(desc="Detailed QC metrics dictionary")


class AIFDetection(BaseInterface):
    """Arterial Input Function (AIF) detection for DSC-MRI.

    Implements the AIF detection pipeline with:
    - Physical in-plane brain-edge exclusion for every search tier
    - Adaptive percentile candidate pre-filtering (99.5% -> 95%)
    - Symmetry-aligned posterior midline and edge-slice exclusions
    - Tissue TTP cross-validation
    - Cost-function scoring (lower = better) with S0-dominant weighting
    - Inverse-cost weighted candidate-voxel averaging with fixed-duration temporal smoothing
    - Multi-level fallback: atlas strict/relaxed/permissive -> brain relaxed/permissive
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
        voxel_sizes: tuple[float, float, float],
        *,
        apply_posterior_midline_exclusion: bool = False,
        posterior_midline_zone: Optional[npt.NDArray[Any]] = None,
    ) -> npt.NDArray[Any]:
        """Prepare a binary search mask with edge and lesion exclusion.

        Every tier uses a physical in-plane distance from the filled outer
        brain boundary before optional arterial-atlas intersection. Whole-
        brain tiers additionally exclude a posterior midline venous zone and
        every tier excludes the first and last acquired slices.
        Distance-based lesion filtering happens *after* scoring (see
        ``_apply_post_scoring_filters``).

        Parameters
        ----------
        brain_mask : ndarray
            3D binary brain mask.
        roi_mask : ndarray or None
            Optional 3D ROI (atlas) mask to intersect.
        lesion_mask : ndarray or None
            Optional 3D lesion mask to exclude.
        voxel_sizes : tuple of float
            Voxel sizes in millimetres.
        apply_posterior_midline_exclusion : bool
            Apply the whole-brain posterior midline venous exclusion.
        posterior_midline_zone : ndarray or None
            Symmetry-aligned exclusion zone. A scanner-axis posterior band is
            used if symmetry estimation was unavailable.

        Returns
        -------
        ndarray
            3D boolean search mask.

        Raises
        ------
        ValueError
            If voxel sizes are not three finite positive values.
        """
        brain_bool = brain_mask > 0.5
        if len(voxel_sizes) != 3 or any(
            not np.isfinite(size) or size <= 0 for size in voxel_sizes
        ):
            raise ValueError(
                "voxel_sizes must contain three finite positive values"
            )

        envelope = np.stack(
            [
                binary_fill_holes(brain_bool[:, :, z])
                for z in range(brain_bool.shape[2])
            ],
            axis=2,
        )
        distance_from_edge = np.stack(
            [
                distance_transform_edt(
                    envelope[:, :, z],
                    sampling=voxel_sizes[:2],
                )
                for z in range(brain_bool.shape[2])
            ],
            axis=2,
        )
        search = brain_bool & (
            distance_from_edge >= _BRAIN_EDGE_EXCLUSION_MM
        )
        if roi_mask is not None:
            search &= roi_mask > 0.5

        # Restore only the tier domain if the physical edge rule empties it;
        # the remaining spatial exclusions below stay active.
        if not np.any(search):
            search = brain_bool.copy()
            if roi_mask is not None:
                search &= roi_mask > 0.5

        if apply_posterior_midline_exclusion:
            zone = posterior_midline_zone
            if zone is None:
                zone = AIFDetection._scanner_axis_posterior_midline_zone(
                    brain_bool,
                )
            pre_midline = search.copy()
            search[np.asarray(zone) > 0] = False
            if not np.any(search):
                search = pre_midline

        if search.shape[2] > 2:
            pre_edge_slices = search.copy()
            search[:, :, 0] = False
            search[:, :, -1] = False
            if not np.any(search):
                search = pre_edge_slices

        if lesion_mask is not None and np.any(lesion_mask > 0.5):
            lesion_bool = lesion_mask > 0.5
            pre_lesion = search.copy()
            search = search & ~lesion_bool
            if not np.any(search):
                search = pre_lesion

        return search  # type: ignore[no-any-return]

    @staticmethod
    def _scanner_axis_posterior_midline_zone(
        brain_mask: npt.NDArray[Any],
    ) -> npt.NDArray[np.bool_]:
        """Construct a posterior midline fallback from the brain bounding box."""
        brain = np.asarray(brain_mask) > 0.5
        projection = np.any(brain, axis=2)
        points = np.argwhere(projection)
        zone_2d = np.zeros(projection.shape, dtype=bool)
        if points.size == 0:
            return np.broadcast_to(zone_2d[:, :, None], brain.shape).copy()

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        x_center = 0.5 * (float(x_min) + float(x_max))
        half_width = _WB_MIDLINE_HALF_WIDTH_FRACTION * float(x_max - x_min)
        posterior_cutoff = float(y_min) + (
            _WB_POSTERIOR_FRACTION * float(y_max - y_min)
        )
        x_grid, y_grid = np.indices(projection.shape, dtype=float)
        zone_2d = (
            (np.abs(x_grid - x_center) <= half_width)
            & (y_grid <= posterior_cutoff)
        )
        return np.broadcast_to(zone_2d[:, :, None], brain.shape).copy()

    @staticmethod
    def _brain_symmetry_midline_zone(
        brain_mask: npt.NDArray[Any],
        affine: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[np.bool_] | None, float, float]:
        """Estimate a posterior midline exclusion zone from brain symmetry.

        The axial brain projection is reflected across candidate in-plane
        axes. The best-aligned axis defines a central band over the posterior
        portion of the brain while preserving anterior midline territory.

        Parameters
        ----------
        brain_mask : ndarray
            Three-dimensional brain mask.
        affine : ndarray
            Voxel-to-world affine for ``brain_mask``.

        Returns
        -------
        zone : ndarray of bool or None
            Three-dimensional exclusion zone, or ``None`` if unavailable.
        angle_degrees : float
            Estimated midline-normal angle relative to the first in-plane
            voxel axis.
        overlap : float
            Dice overlap between the projection and its reflection.
        """
        from scipy.ndimage import map_coordinates

        brain = np.asarray(brain_mask) > 0.5
        projection = np.any(brain, axis=2)
        points = np.argwhere(projection)
        if points.shape[0] < 10:
            return None, float("nan"), float("nan")

        affine_array = np.asarray(affine, dtype=float)
        first_axis = affine_array[:3, 0]
        second_axis = affine_array[:3, 1]
        first_norm = float(np.linalg.norm(first_axis))
        if first_norm <= 0:
            return None, float("nan"), float("nan")
        basis_first = first_axis / first_norm
        second_orthogonal = second_axis - np.dot(
            second_axis,
            basis_first,
        ) * basis_first
        second_norm = float(np.linalg.norm(second_orthogonal))
        if second_norm <= 0:
            return None, float("nan"), float("nan")
        basis_second = second_orthogonal / second_norm
        basis = np.column_stack((basis_first, basis_second))
        plane_matrix = basis.T @ affine_array[:3, :2]
        plane_offset = basis.T @ affine_array[:3, 3]
        try:
            inverse_plane_matrix = np.linalg.inv(plane_matrix)
        except np.linalg.LinAlgError:
            return None, float("nan"), float("nan")

        grid = np.indices(projection.shape, dtype=float).reshape(2, -1).T
        grid_plane = grid @ plane_matrix.T + plane_offset
        brain_plane = points @ plane_matrix.T + plane_offset
        center = np.mean(brain_plane, axis=0)

        def _search(
            angles: npt.NDArray[np.floating[Any]],
            shifts: npt.NDArray[np.floating[Any]],
        ) -> tuple[float, float, float, npt.NDArray[np.float64], float]:
            best: tuple[
                float,
                float,
                float,
                npt.NDArray[np.float64],
                float,
            ] | None = None
            projection_float = projection.astype(float)
            for angle_degrees in angles:
                angle_radians = np.deg2rad(angle_degrees)
                normal = np.asarray(
                    [np.cos(angle_radians), np.sin(angle_radians)],
                    dtype=np.float64,
                )
                center_projection = float(center @ normal)
                for shift_mm in shifts:
                    position = center_projection + float(shift_mm)
                    signed = grid_plane @ normal - position
                    reflected_plane = grid_plane - 2.0 * signed[:, None] * normal
                    reflected_voxels = (
                        reflected_plane - plane_offset
                    ) @ inverse_plane_matrix.T
                    reflected = map_coordinates(
                        projection_float,
                        [reflected_voxels[:, 0], reflected_voxels[:, 1]],
                        order=1,
                        mode="constant",
                        cval=0.0,
                    ).reshape(projection.shape)
                    denominator = float(np.sum(projection) + np.sum(reflected))
                    overlap = (
                        2.0 * float(np.sum(projection * reflected)) / denominator
                        if denominator > 0 else 0.0
                    )
                    candidate = (
                        overlap, float(angle_degrees), position, normal,
                        float(shift_mm),
                    )
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is None:
                raise RuntimeError("DSC brain-symmetry search produced no result")
            return best

        coarse = _search(
            np.arange(
                -_WB_SYMMETRY_MAX_ANGLE_DEGREES,
                _WB_SYMMETRY_MAX_ANGLE_DEGREES + 0.01,
                2.0,
            ),
            np.arange(
                -_WB_SYMMETRY_MAX_SHIFT_MM,
                _WB_SYMMETRY_MAX_SHIFT_MM + 0.01,
                1.0,
            ),
        )
        best = _search(
            np.arange(coarse[1] - 2.0, coarse[1] + 2.01, 0.25),
            np.arange(coarse[4] - 1.0, coarse[4] + 1.01, 0.25),
        )
        overlap, angle_degrees, position, normal, _ = best
        signed = (grid_plane @ normal - position).reshape(projection.shape)
        brain_width = float(np.ptp(brain_plane @ normal))

        anterior_axis = np.asarray([-normal[1], normal[0]], dtype=float)
        world_anterior = np.asarray([0.0, 1.0, 0.0]) @ basis
        if float(anterior_axis @ world_anterior) < 0:
            anterior_axis *= -1.0
        brain_ap = brain_plane @ anterior_axis
        posterior_cutoff = float(np.min(brain_ap)) + (
            _WB_POSTERIOR_FRACTION * float(np.ptp(brain_ap))
        )
        grid_ap = (grid_plane @ anterior_axis).reshape(projection.shape)
        zone_2d = (
            (grid_ap <= posterior_cutoff)
            & (np.abs(signed) <= _WB_MIDLINE_HALF_WIDTH_FRACTION * brain_width)
        )
        zone = np.broadcast_to(zone_2d[:, :, None], brain.shape).copy()
        return zone, angle_degrees, overlap

    # -----------------------------------------------------------------
    # Step 3: Percentile candidate finder
    # -----------------------------------------------------------------

    def _find_aif_candidate_voxels(
        self,
        conc_data: npt.NDArray[Any],
        search_mask: npt.NDArray[Any],
        baseline_frames: int,
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
        """Compute concentration and signal quality metrics for an AIF curve.

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
    # Effective dose helper (used by candidate scoring)
    # -----------------------------------------------------------------

    def _effective_dose(
        self,
        dose: Optional[float] = None,
        agent_standard_dose: Optional[float] = None,
    ) -> float:
        """Effective Gd dose relative to 0.1 mmol Gd/kg (unclamped).

        Parameters
        ----------
        dose : float, optional
            Override; falls back to self.inputs.dose.
        agent_standard_dose : float, optional
            Override; falls back to self.inputs.agent_standard_dose.

        Returns
        -------
        float
            Raw multiplier ``(agent_std * dose) / GD_DOTA_STANDARD_DOSE``.
            Consumers apply their own clamping.
        """
        d = dose if dose is not None else (
            float(self.inputs.dose) if isdefined(self.inputs.dose) else 1.0
        )
        a = agent_standard_dose if agent_standard_dose is not None else (
            float(self.inputs.agent_standard_dose)
            if isdefined(self.inputs.agent_standard_dose) else 0.1
        )
        return (a * d) / GD_DOTA_STANDARD_DOSE

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
    ) -> float:
        """Score a candidate with the AIF cost function.

        Lower is better.  Returns ``inf`` for candidates that fail a
        configured acceptance gate.

        Parameters
        ----------
        metrics : dict
            Output of ``_calculate_candidate_metrics``.
        criteria : str
            One of 'strict', 'relaxed', 'permissive'.
        median_s0 : float or None
            Median baseline signal for S0 partial-volume rejection.
        dose : float or None
            Contrast agent dose multiplier. None uses the interface setting,
            or 1.0 if that setting is undefined.
        agent_standard_dose : float or None
            Standard dose in mmol Gd/kg. None uses the interface
            setting, or 0.1 if that setting is undefined.

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

        # 2. Tier-specific baseline-return-ratio acceptance window
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

        # 5c. Reject FWHM below the criterion-specific minimum in seconds.
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

        # Peak bounds scale with field strength and effective dose, with an
        # absolute noise floor and a ceiling for susceptibility artifacts.
        effective_dose = 1.0  # default, overwritten below if peak_conc available
        peak_conc = metrics.get("peak_conc", float("nan"))
        if not np.isnan(peak_conc):
            effective_dose = float(np.clip(
                self._effective_dose(dose, agent_standard_dose),
                0.2, 3.0,
            ))
            field_strength = (
                float(self.inputs.field_strength)
                if isdefined(self.inputs.field_strength) else 3.0
            )
            b0_scale = field_strength / 3.0

            peak_floor = max(
                AIF_PEAK_FLOOR_3T * b0_scale * effective_dose,
                AIF_MIN_PEAK,
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
    # Step 8b: Post-scoring lesion-distance filter
    # -----------------------------------------------------------------

    def _apply_post_scoring_filters(
        self,
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        lesion_mask: Optional[npt.NDArray[Any]],
        voxel_sizes: tuple[float, ...],
        lesion_min_distance_mm: float = LESION_EXCLUSION_DISTANCE_MM,
    ) -> tuple[
        list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        dict[str, Any],
    ]:
        """Apply distance-based lesion exclusion.

        Step 4.5: Distance-based lesion exclusion.

        Parameters
        ----------
        top_candidates : list
            Scored ``(cost, (x,y,z), metrics)`` tuples.
        lesion_mask : ndarray or None
            3D lesion mask (dilated whole-tumor) for distance exclusion.
        voxel_sizes : tuple of float
            Voxel dimensions in mm ``(dx, dy, dz)``.
        lesion_min_distance_mm : float
            Minimum Euclidean distance from lesion in mm.

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
           input pool's 20th-percentile peak time, with a one-frame minimum.
           The reference and spread are computed before the tail filter.
        2. **Tail/peak filter**: keep voxels with
           ``tail_fraction_late <= AIF_TAIL_FRACTION_MAX`` (default
           0.15).

        TTP causality (peak before tissue) is **not** enforced here;
        it is the responsibility of ``_score_candidate_cost`` upstream
        using the peak-time difference in seconds and the configured
        ``AIF_TTP_REJECT_AFTER_TISSUE_S`` and ``AIF_TTP_NEAR_TISSUE_S`` limits.

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
            ``AIF_PEAK_SPREAD_MAX_SECONDS`` configured spread
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
        roi_size: int = 1,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Average candidate ROIs with inverse-cost weights and temporal smoothing.

        Parameters
        ----------
        top_candidates : list
            List of (cost, (x,y,z), metrics) tuples.
        conc_data : ndarray
            4D concentration or signal data for the selected candidates.
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
            Averaged time course in the input data's units.
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

        Each candidate contributes its own voxel (``roi_size=1``), with
        inverse-cost weighting across candidates.

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
    # Step 9b: Finalize AIF metrics
    # -----------------------------------------------------------------

    @staticmethod
    def _finalize_aif_metrics(
        aif_curve: npt.NDArray[Any],
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        signal_data: Optional[npt.NDArray[Any]],
        brain_mask: npt.NDArray[Any],
        affine: npt.NDArray[Any],
        tr: float,
        n_candidates_requested: int,
        n_available: int,
        baseline_frames: int = 12,
        roi_size: int = 1,
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Finalize AIF: detect bolus arrival and compute comprehensive QC.

        Parameters
        ----------
        aif_curve : ndarray
            Accepted smoothed measured AIF concentration curve before gamma fitting.
        top_candidates : list
            ``(cost, (x,y,z), metrics)`` tuples used.
        signal_data : ndarray or None
            Original 4D signal used to measure the accepted pool's fractional
            signal drop. An unavailable signal leaves PH unavailable.
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
            Finalized number of pre-bolus baseline frames.
        roi_size : int
            Side length of each candidate ROI used for the measured AIF.

        Returns
        -------
        tuple of (aif_curve, qc_dict)
        """
        from pmriprep.utils.dsc_signal import detect_bolus_arrival

        pooled_signal = None
        if signal_data is not None:
            pooled_signal, _ = AIFDetection._compute_roi_averaged_aif(
                top_candidates, signal_data, brain_mask, tr, roi_size=roi_size,
            )
            if not np.all(np.isfinite(pooled_signal)):
                pooled_signal = None
        pooled_metrics = AIFDetection._calculate_candidate_metrics(
            aif_curve, pooled_signal, tr, baseline_frames,
        ) or {}

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

        # Edge slice warning
        n_slices = brain_mask.shape[2]
        aif_z = best_coord_int[2]
        is_edge = aif_z == 0 or aif_z == n_slices - 1

        n_used = len(top_candidates)

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
            "candidate_coords": [
                [int(c) for c in coord]
                for _, coord, _ in top_candidates
            ],
            "candidate_scores": _costs_to_scores(
                [float(cost) for cost, _, _ in top_candidates]
            ),
            # Edge slice
            "aif_slice": aif_z,
            "n_slices": n_slices,
            "aif_edge_slice_warning": is_edge,
            # Best candidate quality metrics
            "best_cost": float(best_cost),
            "mean_cost": float(
                np.mean([c for c, _, _ in top_candidates])
            ),
            # Peak amplitude is measured on the combined curve. TTP, FWHM
            # and PSR describe the best candidate; prior PH/BRR describe
            # the combined measured pool and are added when available.
            "aif_metrics": {
                "peak_conc": float(np.max(aif_curve)),
                "ttp": float(best_m.get("ttp", 0.0)),
                "fwhm": float(best_m.get("fwhm", 0.0)),
                "signal_recovery_percent": float(best_m.get("psr", 0.0)),
            },
        }

        for source, target in (
            ("ph", "peak_height_percent"),
            ("baseline_return_ratio", "baseline_return_ratio_percent"),
        ):
            value = pooled_metrics.get(source)
            qc["aif_metrics"][target] = float(value) if value is not None else float("nan")

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

        Raises
        ------
        ValueError
            If an NPZ file lacks the required AIF key, the curve is not a finite
            1D vector matching the scan length, or its peak is below 1e-6.
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
            "n_baseline_frames": baseline_frames,
            "baseline_detection_method": "selected_aif_refined",
            # Detection-time AIF metrics under the canonical nested
            # key. The external path records peak amplitude and FWHM in
            # frames; its concentration washout fraction is recorded above.
            "aif_metrics": {
                "peak_conc": float(peak),
                "fwhm": float(fwhm_frames),
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

        qc_metrics["detection_status"] = "not_attempted"
        qc_metrics["aif_source"] = "external"
        qc_metrics["cbv_auc_denominator_available"] = True

        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "out_aif_coord": [0, 0, 0],
            "aif_curve": [float(x) for x in aif_curve],
            "detection_level": "external",
            "quality_score": quality_score,
            "candidate_count": 0,
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
            initial_criteria=str(self.inputs.criteria),
            n_candidates=int(self.inputs.n_candidates),
            baseline_frames=int(self.inputs.baseline_frames),
            lesion_distance=lesion_distance,
            n_timepoints=concentration.shape[3],
            voxel_sizes=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
            tr=tr,
        )

    @staticmethod
    def _build_cascade_attempts(
        initial_criteria: str,
        atlas_mask: Optional[npt.NDArray[Any]],
        atlas_is_mra: bool,
        full_atlas_mask: Optional[npt.NDArray[Any]],
        user_mask_supplied: bool,
    ) -> list[_CascadeAttempt]:
        """Build the ordered DSC AIF search tiers permitted by the inputs."""
        criteria_order = ["strict", "relaxed", "permissive"]
        try:
            start_index = criteria_order.index(initial_criteria)
        except ValueError:
            start_index = 0

        attempts: list[_CascadeAttempt] = []
        if atlas_mask is not None:
            region = "user_mask" if user_mask_supplied else "atlas"
            for criteria in criteria_order:
                attempts.append(
                    (criteria, atlas_mask, f"{region}_{criteria}",
                     atlas_is_mra or user_mask_supplied)
                )

        if full_atlas_mask is not None and not user_mask_supplied:
            attempts.append(
                ("permissive", full_atlas_mask, "full_atlas_permissive", True)
            )

        if not user_mask_supplied:
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
        roi_constrained: bool,
        posterior_midline_zone: Optional[npt.NDArray[Any]],
        symmetry_angle: float,
        symmetry_overlap: float,
        n_candidates: int,
        baseline_frames: int,
        tissue_ttp: float,
    ) -> _TierCandidateResult:
        """Run candidate preparation and filtering for one cascade tier."""
        search_mask = self._prepare_search_mask(
            context.brain_mask,
            roi,
            context.lesion_mask,
            voxel_sizes=context.voxel_sizes,
            apply_posterior_midline_exclusion=not roi_constrained,
            posterior_midline_zone=posterior_midline_zone,
        )
        candidate_voxels, peak_map = self._find_aif_candidate_voxels(
            context.concentration,
            search_mask,
            baseline_frames,
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
            context.voxel_sizes,
            lesion_min_distance_mm=context.lesion_distance,
        )
        if not roi_constrained:
            filter_metadata["whole_brain_symmetry_method"] = (
                "brain_projection_reflection"
                if posterior_midline_zone is not None
                else "scanner_axis_fallback"
            )
            if np.isfinite(symmetry_angle):
                filter_metadata["whole_brain_symmetry_angle_degrees"] = float(
                    symmetry_angle,
                )
            if np.isfinite(symmetry_overlap):
                filter_metadata["whole_brain_symmetry_overlap"] = float(
                    symmetry_overlap,
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
        self, candidates: Sequence[_AIFCandidate],
        concentration: npt.NDArray[Any], brain_mask: npt.NDArray[Any],
        tr: float, baseline_frames: int, logger: Any,
    ) -> tuple[Optional[_SingleBestAIFResult], Optional[_RejectedAIF]]:
        """Apply pooled curve checks to the best voxel and retain rejection QC."""
        if not candidates:
            return None, None
        best_candidate = min(candidates, key=lambda candidate: candidate[0])
        curve, mask = self._compute_roi_averaged_aif(
            [best_candidate], concentration, brain_mask, tr, roi_size=1,
        )
        quality = self._aif_passes_quality(curve, baseline_frames, tr)
        if quality.passes:
            return _SingleBestAIFResult(curve, mask, 1, best_candidate), None
        if not self._aif_global_max_in_fp(curve, baseline_frames, tr):
            reason = "peak_outside_first_pass"
        elif quality.spr > 100.0:
            reason = "secondary_peak"
        elif quality.peak_ratio is None or not (
            GAMMA_PEAK_RATIO_CASCADE_MIN <= quality.peak_ratio <= GAMMA_PEAK_RATIO_CASCADE_MAX
        ):
            reason = "peak_ratio"
        else:
            reason = "negative_tail"
        rejected = _RejectedAIF(curve, mask, {
            "reason": reason,
            "roi_size": 1,
            "baseline_frames_used": baseline_frames,
            "aif_voxel_coord": list(best_candidate[1]),
            "candidate_coords": [list(best_candidate[1])],
            "candidate_scores": _costs_to_scores([best_candidate[0]]),
            "baseline_return_ratio": float(quality.brr) if np.isfinite(quality.brr) else None,
        })
        logger.debug("Single-best AIF rejected: %s", reason)
        return None, rejected




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
        if added > 0:
            if not state.filter_metadata:
                state.filter_metadata.update(filter_metadata)
            else:
                for key, value in filter_metadata.items():
                    if key.startswith("whole_brain_symmetry_"):
                        state.filter_metadata[key] = value

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
            "+".join(contributing_levels) if contributing_levels else "measured"
        )
        result = acceptance.result
        return _UnionAcceptanceResult(
            (result.curve, result.mask, result.roi_size),
            candidates,
            detection_level,
            tier_provenance,
        )

    def _aif_averaging_mask(self, context: _DSCAIFDetectionContext) -> npt.NDArray[Any]:
        """Limit arterial ROI averaging to the supplied search region, if any."""
        if isdefined(self.inputs.aif_mask_file) and self.inputs.aif_mask_file:
            assert context.atlas_mask is not None
            return (context.brain_mask > 0.5) & (context.atlas_mask > 0.5)
        return context.brain_mask

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
            context.signal,
            self._aif_averaging_mask(context),
            context.affine,
            context.tr,
            context.n_candidates,
            cascade.n_available,
            baseline_frames=baseline_frames,
            roi_size=accepted_roi_size,
        )
        qc_metrics.update(baseline_diagnostics)
        qc_metrics.update(cascade.filter_metadata)
        qc_metrics.update(
            describe_tumor_relative_aif_candidates(
                [coordinate for _, coordinate, _ in cascade.candidates],
                context.affine,
                context.lesion_mask,
                enhancing_mask=context.enhancing_mask,
                brain_mask=context.brain_mask,
            )
        )
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
            "preliminary_baseline_frames": int(preliminary_baseline),
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
            diagnostics["n_baseline_frames"] = int(preliminary_baseline)
            return preliminary_baseline, diagnostics

        confidence = str(ensemble.get("confidence", "low"))
        diagnostics.update(
            baseline_refined_candidate_frames=int(refined),
            baseline_refinement_confidence=confidence,
            baseline_refinement_method=ensemble.get("method", "ensemble"),
        )
        if refined < 2 or refined > len(curve) // 2:
            diagnostics["baseline_refinement_method"] = "ensemble_out_of_range"
            diagnostics["n_baseline_frames"] = int(preliminary_baseline)
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

        diagnostics["n_baseline_frames"] = int(final)
        return int(final), diagnostics

    def _prepare_detection_baseline(
        self, context: _DSCAIFDetectionContext, logger: Any,
    ) -> _TissueBaseline:
        """Estimate whole-brain baseline timing and retain its confidence."""
        brain_concentration = context.concentration[context.brain_mask > 0.5]
        diagnostics: dict[str, Any] = {"confidence": "unknown", "method": "configured"}
        if brain_concentration.ndim > 1 and brain_concentration.shape[0] > 100:
            from pmriprep.utils.baseline_detection import detect_baseline_ensemble

            p90_curve, p99_curve = np.percentile(brain_concentration, [90, 99], axis=0)
            mean_curve = np.mean(brain_concentration, axis=0)
            baseline_frames, diagnostics = detect_baseline_ensemble(
                p99_curve, tr=context.tr,
                min_baseline=max(5, int(10.0 / max(context.tr, 0.1))),
                additional_signals=[p90_curve, mean_curve],
            )
            logger.debug("Tissue baseline: arrival=%d, confidence=%s",
                         baseline_frames, diagnostics.get("confidence", "unknown"))
        else:
            baseline_frames = context.baseline_frames
        return _TissueBaseline(baseline_frames, diagnostics)


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
        initial_criteria = context.initial_criteria
        n_candidates = context.n_candidates
        tr = context.tr
        # Initialize
        detection_level = "unavailable"
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
        # accept from, so the unavailable-AIF result can report why
        # each tier failed.
        postfilter_diags_per_tier: dict[str, dict[str, Any]] = {}
        spatial_filter_diags_per_tier: dict[str, dict[str, Any]] = {}
        rejected: Optional[_RejectedAIF] = None
        attempts = self._build_cascade_attempts(
            initial_criteria,
            atlas_mask,
            atlas_is_mra,
            full_atlas_mask,
            isdefined(self.inputs.aif_mask_file) and bool(self.inputs.aif_mask_file),
        )

        # Compute tissue-TTP reference once — independent of tier
        # because it operates on the whole brain mask. Used by the
        # per-tier scorer and post-score kinetic/anchor filter.
        tissue_ttp = self._calculate_tissue_ttp_reference(
            conc_data, brain_mask, aif_baseline_frames
        )
        posterior_midline_zone: Optional[npt.NDArray[Any]] = None
        symmetry_angle = float("nan")
        symmetry_overlap = float("nan")
        symmetry_computed = False

        for crit, roi, level_name, roi_constrained in attempts:
            if not roi_constrained and not symmetry_computed:
                (
                    posterior_midline_zone,
                    symmetry_angle,
                    symmetry_overlap,
                ) = self._brain_symmetry_midline_zone(
                    brain_mask,
                    context.affine,
                )
                symmetry_computed = True
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
                roi_constrained,
                posterior_midline_zone,
                symmetry_angle,
                symmetry_overlap,
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

            # Accumulate survivors across tiers, capped at n_candidates.
            # Test the union and single-best candidate after all per-tier
            # pools fail; if those also fail, report the AIF as unavailable.
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
        # voxel with the same curve-quality gate as the averaged pools.
        if accepted_aif is None and len(union_state.candidates) > 0:
            single_best, rejected = self._try_single_best_fallback(
                union_state.candidates,
                conc_data,
                self._aif_averaging_mask(context),
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
            rejected,
        )

        return cascade

    def _serialize_detection_outputs(self, state: _FinalAIFState) -> None:
        """Write available analysis and diagnostic artifacts independently."""
        import nibabel as nib

        results: dict[str, Any] = {
            "out_aif_coord": state.qc_metrics.get("aif_voxel_coord", []),
            "aif_curve": state.curve.tolist() if state.curve is not None else [],
            "detection_level": state.detection_level,
            "candidate_count": state.candidate_count,
            "aif_baseline_frames": state.baseline_frames,
            "qc_metrics": state.qc_metrics,
        }
        if state.quality_score is not None:
            results["quality_score"] = state.quality_score
        curves = {
            "out_aif_curve": (state.curve, "_aif_curve.txt"),
            "out_gamma_aif_curve": (state.gamma_curve, "_gamma_aif_curve.txt"),
            "out_rejected_aif_curve": (
                state.rejected.curve if state.rejected is not None else None,
                "_rejected_aif_curve.txt",
            ),
        }
        for name, (curve, suffix) in curves.items():
            if curve is not None:
                path = self._gen_outfilename(suffix)
                np.savetxt(path, curve)
                results[name] = path
        masks = {
            "out_aif_mask": (state.mask, "_aif_mask.nii.gz"),
            "out_rejected_aif_mask": (
                state.rejected.mask if state.rejected is not None else None,
                "_rejected_aif_mask.nii.gz",
            ),
        }
        for name, (mask, suffix) in masks.items():
            if mask is not None:
                path = self._gen_outfilename(suffix)
                nib.save(nib.Nifti1Image(mask.astype(np.uint8), state.affine), path)
                results[name] = path
        self._results = results


    def _finalize_post_cascade(
        self, context: _DSCAIFDetectionContext, cascade: _CascadeDetectionResult,
        baseline: _TissueBaseline, logger: Any,
    ) -> _FinalAIFState:
        """Retain an accepted measured AIF or report unavailable arterial input."""
        curve = None
        mask = None
        quality_score = None
        baseline_frames = baseline.frames
        qc: dict[str, Any] = {}
        if cascade.accepted_aif is not None and cascade.candidates:
            selected = self._finalize_detected_aif(cascade, context, baseline.frames)
            curve, mask = selected.curve, selected.mask
            baseline_frames, quality_score = selected.baseline_frames, selected.quality_score
            qc.update(selected.qc_metrics)
            qc.update(detection_status="accepted", aif_source="measured",
                      quality_score=quality_score, cbv_auc_denominator_available=True)
        else:
            qc.update(
                detection_status="failed", aif_source="unavailable",
                failure_reason="quality_checks_failed" if cascade.rejected is not None else "no_candidates",
                n_aif_voxels_selected=0, cbv_auc_denominator_available=False,
            )
            logger.warning("DSC AIF detection failed; retaining independent tissue maps")
        if isdefined(self.inputs.aif_mask_file) and self.inputs.aif_mask_file:
            qc["search_region"] = "user_mask"
            if curve is None:
                assert context.atlas_mask is not None
                qc["failure_detail"] = (
                    "empty_user_mask"
                    if not np.any(context.atlas_mask > 0.5)
                    else qc["failure_reason"]
                )
                qc["failure_reason"] = "user_mask_exhausted"
        qc.update(
            detection_method=cascade.detection_level,
            n_aif_voxels_selected=len(cascade.candidates) if curve is not None else 0,
            cascade_tiers_attempted=list(cascade.tiers_attempted),
            cascade_postfilter_diags=cascade.postfilter_diagnostics,
            cascade_spatial_filter_diags=cascade.spatial_filter_diagnostics,
            s0_rejection_available=context.signal is not None,
            tr_used=context.tr,
            bootstrap_baseline_frames=context.baseline_frames,
            preliminary_baseline_frames=baseline.frames,
            tissue_baseline_confidence=baseline.diagnostics.get("confidence", "unknown"),
            tissue_baseline_diagnostics=baseline.diagnostics,
            n_baseline_frames=baseline_frames,
            baseline_detection_method="selected_aif_refined" if curve is not None else "whole_brain_ensemble",
            brain_edge_exclusion_target_mm=_BRAIN_EDGE_EXCLUSION_MM,
        )
        gamma_curve = None
        if curve is not None:
            gamma_curve = self._fit_gamma_for_qc(curve, baseline_frames, context.tr, qc)
            metrics = dict(qc.get("aif_metrics") or {})
            metrics["peak_conc"] = float(np.max(curve))
            qc["aif_metrics"] = metrics
        elif cascade.rejected is not None:
            diagnostic_qc = dict(cascade.rejected.qc)
            gamma_curve = self._fit_gamma_for_qc(
                cascade.rejected.curve, baseline.frames, context.tr, diagnostic_qc,
            )
            qc["rejected_candidate"] = diagnostic_qc
        return _FinalAIFState(
            curve, mask, gamma_curve, cascade.rejected, context.affine,
            cascade.detection_level, quality_score, baseline_frames,
            len(cascade.candidates) if curve is not None else 0, qc,
        )


    # -----------------------------------------------------------------
    # Step 10: Main pipeline  (_run_interface)
    # -----------------------------------------------------------------

    def _run_interface(self, runtime: Any) -> Any:
        """Execute the AIF detection pipeline.

        Prepare the baseline, run the candidate cascade, then finalize
        diagnostic gamma fitting before serializing curves and QC.
        External input bypasses detection.
        """
        import logging

        _logger = logging.getLogger(__name__)

        # ===== External AIF: bypass all detection if provided =====
        if isdefined(self.inputs.external_aif_file) and self.inputs.external_aif_file:
            return self._handle_external_aif(runtime)

        context = self._load_detection_context(_logger)
        baseline = self._prepare_detection_baseline(context, _logger)
        cascade = self._run_candidate_cascade(
            context, baseline.frames, _logger,
        )
        self._serialize_detection_outputs(
            self._finalize_post_cascade(
                context,
                cascade,
                baseline,
                _logger,
            )
        )

        return runtime

    @staticmethod
    def _fit_gamma_for_qc(
        aif_curve: npt.NDArray[Any], baseline_frames: int,
        tr: float, qc_metrics: dict[str, Any],
    ) -> Optional[npt.NDArray[Any]]:
        """Fit a diagnostic gamma model while preserving the measured input."""
        from pmriprep.utils.dsc_aif import fit_gamma_variate, compute_gamma_diagnostics

        fit = fit_gamma_variate(aif_curve, baseline_frames=baseline_frames,
                               remove_recirculation=True, preserve_auc=True, tr=tr)
        qc_metrics["gamma_fit_converged"] = fit.fit_converged
        if fit.failure_reason is not None:
            qc_metrics["gamma_fit_failure_reason"] = fit.failure_reason
        if not fit.fit_converged:
            return None
        qc_metrics["gamma_fit_r_squared"] = float(fit.r_squared)
        qc_metrics["gamma_fit_evaluation_start_frame"] = baseline_frames
        qc_metrics["gamma_fit_evaluation_end_frame"] = baseline_frames + fit.first_pass_n
        qc_metrics["gamma_fit_diagnostics"] = compute_gamma_diagnostics(
            aif_curve, fit, baseline_frames,
        )
        return fit.fitted_curve


    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """Return only artifacts produced by this run."""
        return dict(getattr(self, "_results", {}))
