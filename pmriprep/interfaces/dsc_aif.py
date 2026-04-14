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
from typing import Any, Optional

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
    EPSILON_CONCENTRATION,
    EPSILON_SIGNAL,
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
#: Physics: >12% AIF peak error exceeds normal CBF CoV (~10%,
#: Mutsaerts et al., JCBFM 2015).
GAMMA_PEAK_RATIO_CASCADE_MIN: float = 0.88
GAMMA_PEAK_RATIO_CASCADE_MAX: float = 1.12

#: Gamma R² floor for synthetic fallback.
#: R² < 0 means gamma fit is worse than mean — curve has no bolus shape.
#: A true arterial bolus will never fit worse than a horizontal line.
GAMMA_R2_SYNTHETIC_FALLBACK: float = 0.0

#: S0 ratio floor: reject candidates with baseline signal below this
#: fraction of median brain S0.  Below 0.10, baseline SNR < 5 and the
#: logarithmic ΔR2* conversion is numerically unstable.
S0_RATIO_FLOOR: float = 0.10

#: S0 ratio score cap: candidates between S0_RATIO_FLOOR and this value
#: are admitted but capped in the S0 scoring term so marginal-SNR voxels
#: do not overwhelm shape-quality metrics.
S0_RATIO_SCORE_CAP: float = 0.15

# Fraction of user-configured n_candidates to keep at each criteria level.
# As criteria relax, candidate quality drops — fewer candidates concentrates
# on the best-scoring voxels from a noisier pool.
_CASCADE_N_CANDIDATES_FRACTION: dict[str, float] = {
    "strict": 1.0,
    "relaxed": 0.6,
    "permissive": 0.4,
    # Note: "degraded" removed in v0.19.16
}


# ---------------------------------------------------------------------------
# Top-level helper functions (backward-compatible with utils/dsc_aif.py)
# ---------------------------------------------------------------------------

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


def compute_aif_fwhm_frames(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> int:
    """Compute full width at half maximum (FWHM) of AIF peak in frames.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).

    Returns
    -------
    int
        FWHM in number of frames. Arterial curves typically 5-15 frames.
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    peak = float(np.max(aif_curve))
    peak_idx = int(np.argmax(aif_curve))

    half_max = baseline_mean + (peak - baseline_mean) / 2.0

    rising_idx = 0
    for i in range(baseline_frames, peak_idx):
        if aif_curve[i] >= half_max:
            rising_idx = i
            break

    falling_idx = len(aif_curve) - 1
    for i in range(peak_idx + 1, len(aif_curve)):
        if aif_curve[i] <= half_max:
            falling_idx = i
            break

    fwhm = falling_idx - rising_idx
    return max(0, fwhm)


def _compute_aif_peak_ratio(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int,
    tr: float,
) -> Optional[float]:
    """Compute gamma peak ratio for an averaged AIF curve.

    Fits a gamma-variate model and returns peak_fit / peak_orig.
    Used by the cascade quality gate to assess AIF shape fidelity.

    Args:
        aif_curve: Averaged AIF concentration-time curve.
        baseline_frames: Number of pre-bolus baseline frames.
        tr: Repetition time in seconds.

    Returns:
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
        desc="Optional 3D arterial atlas mask in DSC space",
    )
    aif_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional user-supplied AIF search mask (overrides atlas mask)",
    )
    lesion_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 3D lesion mask for exclusion",
    )
    enhancing_mask_file = File(
        exists=True,
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
        15.0,
        usedefault=True,
        desc="Minimum distance (mm) from lesion for AIF candidates",
    )
    gamma_fit = traits.Enum(
        "auto",
        "off",
        "on",
        usedefault=True,
        desc="Gamma-variate fitting mode: 'auto', 'on', or 'off'",
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
        # After RAS enforcement (Fix 7), z is consistently the slice
        # direction (third axis, index 2).
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

        # Safety: if erosion removed everything, fall back to uneroded
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
    def _filter_candidates_by_hemisphere(
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        affine: npt.NDArray[Any],
        tumor_laterality: str,
    ) -> tuple[
        list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        str,
        int,
        int,
    ]:
        """Apply soft cost penalty to ipsilateral candidates.

        For non-bilateral tumors, adds a cost penalty to ipsilateral
        candidates rather than hard-filtering them. This preserves
        bilateral representation while preferring contralateral voxels.

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
        tuple of (filtered, hemisphere_used, n_contra, n_ipsi)
            filtered : all candidates with ipsilateral cost penalty,
                sorted by adjusted cost.
            hemisphere_used : ``'bilateral'`` or ``'both'``.
            n_contra : count of contralateral candidates.
            n_ipsi : count of ipsilateral candidates.
        """
        if tumor_laterality == "bilateral":
            return top_candidates, "both", 0, 0

        contra_side = "right" if tumor_laterality == "left" else "left"

        # Soft penalty for ipsilateral candidates (cost range ~2-15)
        IPSILATERAL_PENALTY = 3.0

        contra: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        adjusted_ipsi: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []

        for entry in top_candidates:
            cost, coord, metrics = entry
            coord_homog = np.array(
                [coord[0], coord[1], coord[2], 1.0]
            )
            world = affine @ coord_homog
            cand_side = "left" if world[0] < 0 else "right"

            if cand_side == contra_side:
                contra.append(entry)
            else:
                adjusted_ipsi.append(
                    (cost + IPSILATERAL_PENALTY, coord, metrics)
                )

        n_contra = len(contra)
        n_ipsi = len(adjusted_ipsi)

        combined = contra + adjusted_ipsi
        combined.sort(key=lambda x: x[0])
        return combined, "bilateral", n_contra, n_ipsi

    # -----------------------------------------------------------------
    # Step 2c: Bilateral balance (ensures both hemispheres represented)
    # -----------------------------------------------------------------

    @staticmethod
    def _ensure_bilateral_balance(
        scored: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        affine: npt.NDArray[Any],
        n_select: int,
    ) -> list[tuple[float, tuple[int, int, int], dict[str, Any]]]:
        """Ensure bilateral AIF representation.

        Guarantees at least ``min_per_hemisphere`` candidates from each
        hemisphere (world-space x=0 as midline). Remaining slots filled
        by cost ranking.

        ``min_per_hemisphere`` is adaptive: ``max(1, min(2, n_select // 4))``
          - n_select=10 -> 2 per hemisphere
          - n_select=5 (GBM) -> 1 per hemisphere

        Parameters
        ----------
        scored : list
            List of ``(cost, (x,y,z), metrics)`` tuples, sorted by cost.
        affine : ndarray
            4x4 affine transformation matrix (voxel -> RAS world).
        n_select : int
            Number of candidates to select.

        Returns
        -------
        list
            Selected candidates with bilateral guarantee, sorted by cost.
        """
        min_per_hemisphere = max(1, min(2, n_select // 4))

        if len(scored) <= n_select or min_per_hemisphere <= 0:
            return scored[:n_select]

        # Split by world-space x (RAS: x<0 = left)
        left: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        right: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        for entry in scored:
            _, coord, _ = entry
            world_x = float(
                affine[0, :3] @ np.array(coord, dtype=float) + affine[0, 3]
            )
            (left if world_x < 0 else right).append(entry)

        # Graceful fallthrough: if one hemisphere is empty, just return by cost
        if not left or not right:
            return scored[:n_select]

        # Reserve minimum from each, fill rest by cost
        min_l = min(min_per_hemisphere, len(left))
        min_r = min(min_per_hemisphere, len(right))
        selected = left[:min_l] + right[:min_r]
        used = {id(e) for e in selected}
        remaining = [e for e in scored if id(e) not in used]
        selected.extend(remaining[: n_select - len(selected)])
        selected.sort(key=lambda x: x[0])
        return selected

    # -----------------------------------------------------------------
    # Step 3: Spatial exclusions
    # -----------------------------------------------------------------

    @staticmethod
    def _apply_spatial_exclusions(
        candidate_indices: npt.NDArray[Any],
        brain_shape: tuple[int, ...],
    ) -> npt.NDArray[Any]:
        """Exclude SSS (midline + superior/posterior) and edge-slice voxels.

        Parameters
        ----------
        candidate_indices : ndarray
            Nx3 array of voxel coordinates.
        brain_shape : tuple
            (nx, ny, nz) spatial dimensions.

        Returns
        -------
        ndarray
            Filtered Nx3 array.
        """
        if len(candidate_indices) == 0:
            return candidate_indices

        nx, ny, nz = brain_shape[:3]
        x_center = nx // 2
        x_exclude = int(nx * 0.1)  # Central 20% (+/-10%)
        z_superior = int(nz * 0.65)  # Superior 35%
        y_posterior = int(ny * 0.45)  # Posterior 45%

        # SSS exclusion
        kept = []
        for idx in candidate_indices:
            x, y, z = int(idx[0]), int(idx[1]), int(idx[2])
            is_midline = (x_center - x_exclude) <= x <= (x_center + x_exclude)
            if is_midline and (z > z_superior or y < y_posterior):
                continue  # SSS region
            kept.append(idx)

        if len(kept) == 0:
            # Would remove all — keep originals
            kept = list(candidate_indices)

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

        # Apply spatial exclusions
        candidates = self._apply_spatial_exclusions(
            candidates, conc_data.shape[:3]
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

        baseline_end_time = (baseline_frames - 1) * tr if baseline_frames > 0 else 0.0
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
        above = conc_curve > half_max
        if np.any(above):
            idxs = np.where(above)[0]
            fwhm = float(idxs[-1] - idxs[0]) * tr if len(idxs) >= 2 else tr
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
        }

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
        dt = metrics.get("dt", 1.5)
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

        # 8b. Peak concentration bounds — dose-scaled artifact rejection
        # Calibrated from 20+ subjects: valid AIFs have peak_conc 5-50 s^-1 at
        # effective_dose=1.0 (standard GBCA full). Artifact example: 486 s^-1.
        peak_conc = metrics.get("peak_conc", float("nan"))
        if not np.isnan(peak_conc):
            BASE_PEAK_FLOOR = 5.0  # s^-1 — tissue contamination floor (dose-independent)
            BASE_PEAK_CEILING = 120.0  # s^-1 — susceptibility artifact ceiling

            # Effective dose only raises the ceiling for high-Gd agents (gadoquatrane);
            # peak_floor is fixed regardless of dose.
            # DSC ΔR2* scales with total Gd3+ ions, so multiply by gd_multiplicity
            # for tetrameric agents (gadoquatrane: 4 Gd per molecule).
            if dose is not None:
                dose_val = dose
            else:
                dose_val = float(self.inputs.dose) if isdefined(self.inputs.dose) else 1.0
            if agent_standard_dose is not None:
                agent_std = agent_standard_dose
            else:
                agent_std = (
                    float(self.inputs.agent_standard_dose)
                    if isdefined(self.inputs.agent_standard_dose)
                    else 0.1
                )
            if gd_multiplicity is not None:
                gd_mult = gd_multiplicity
            else:
                gd_mult = (
                    int(self.inputs.gd_multiplicity)
                    if isdefined(self.inputs.gd_multiplicity)
                    else 1
                )
            effective_dose = (agent_std * dose_val * gd_mult) / 0.1

            # Clamp to prevent pathological thresholds
            effective_dose = np.clip(effective_dose, 0.2, 3.0)

            peak_floor = BASE_PEAK_FLOOR
            peak_ceiling = max(BASE_PEAK_CEILING * effective_dose, BASE_PEAK_CEILING)

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

        # S0 scoring (dominant factor — 5x weight)
        s0_score = 0.0
        if median_s0 is not None and not np.isnan(metrics.get("s0", float("nan"))):
            s0_ratio = metrics["s0"] / (median_s0 + EPSILON_SIGNAL)
            s0_for_scoring = max(s0_ratio, S0_RATIO_SCORE_CAP)
            s0_score = (s0_for_scoring - 0.35) * 20.0

        score = (
            1.0 * ttp_score
            + 0.5 * ph_score
            + 2.0 * brr_score
            + 0.3 * psr_score
            + 0.5 * mono_bonus
            + 5.0 * s0_score
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

        Post-scoring filters (distance, contralateral penalty, bilateral
        balance) run after this method, so all passing candidates are
        returned for downstream selection.

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

            # Tissue TTP cross-validation
            candidate_ttp_frame = m["ttp"] / tr
            frames_before_tissue = tissue_ttp_frame - candidate_ttp_frame

            if np.isfinite(cost):
                if frames_before_tissue < -2:
                    cost = float("inf")  # Peaks >2 frames after tissue → venous
                elif frames_before_tissue < 0:
                    cost += 10.0  # Peaks 0-2 frames after tissue
                elif frames_before_tissue < 2:
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
    # Step 8b: Post-scoring filters (distance + contralateral)
    # -----------------------------------------------------------------

    def _apply_post_scoring_filters(
        self,
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        lesion_mask: Optional[npt.NDArray[Any]],
        affine: npt.NDArray[Any],
        voxel_sizes: tuple[float, ...],
        lesion_min_distance_mm: float = 15.0,
        enhancing_mask: Optional[npt.NDArray[Any]] = None,
    ) -> tuple[
        list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        dict[str, Any],
    ]:
        """Apply distance and soft contralateral filters after scoring.

        Step 4.5: Distance-based lesion exclusion.
        Step 4.6: Soft contralateral penalty (always applied when a
        lesion mask is present; no hard filter).

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
        import logging

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
                # All excluded — keep originals with warning
                logging.getLogger(__name__).warning(
                    "All %d AIF candidates were <%.0fmm from lesion. "
                    "Keeping closest candidate(s).",
                    len(top_candidates), lesion_min_distance_mm,
                )
                for _, _, m in top_candidates:
                    m["lesion_distance_warning"] = True

        # --- Step 4.6: Soft contralateral hemisphere penalty ---
        # Always applied when a lesion mask is present. Uses enhancing_mask
        # for laterality if provided (avoids bilateral misclassification
        # from dilated whole-tumor masks crossing midline).
        tumor_laterality: Optional[str] = None
        hemisphere_used = "both"
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
                (
                    top_candidates,
                    hemisphere_used,
                    n_contra,
                    n_ipsi,
                ) = self._filter_candidates_by_hemisphere(
                    top_candidates, affine, tumor_laterality,
                )
            elif tumor_laterality == "bilateral":
                hemisphere_used = "both"

        meta["tumor_laterality"] = tumor_laterality
        meta["hemisphere_used"] = hemisphere_used
        meta["n_contralateral"] = n_contra
        meta["n_ipsilateral"] = n_ipsi
        meta["laterality_mask_source"] = (
            "enhancing" if (enhancing_mask is not None and np.any(enhancing_mask > 0.5))
            else "lesion"
        )

        return top_candidates, meta

    # -----------------------------------------------------------------
    # Step 9: ROI averaging + smoothing
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_roi_averaged_aif(
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
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
            weights = 1.0 / (costs + 0.1)
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

        # Savitzky-Golay smoothing (window=5, polyorder=2)
        if len(aif_curve) >= 5:
            aif_curve = savgol_filter(aif_curve, window_length=5, polyorder=2)

        # Negative ΔR2* is physically valid (symmetric noise, T1 recirculation).
        # Do NOT clip to zero — it biases the AIF integral, corrupts the SVD
        # Toeplitz matrix, and disables the gamma-fit defect detector.
        # (Calamante MRM 1999, Willats NMR Biomed 2006)

        return aif_curve, aif_mask

    # -----------------------------------------------------------------
    # Post-averaging AIF validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_averaged_aif(
        aif_curve: npt.NDArray[Any],
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        dict[str, Any],
    ]:
        """Validate averaged AIF and retry with smaller ROI if needed.

        Checks whether recirculation exceeds the first-pass peak in the
        ROI-averaged AIF.  If so, re-averages with ``roi_size=1`` and,
        as a last resort, uses the single best candidate voxel.

        Parameters
        ----------
        aif_curve : ndarray
            1D AIF time course from ``_compute_roi_averaged_aif``.
        top_candidates : list
            List of ``(cost, (x,y,z), metrics)`` tuples.
        conc_data : ndarray
            4D concentration data.
        brain_mask : ndarray
            3D brain mask.
        tr : float
            Repetition time in seconds.
        baseline_frames : int
            Number of pre-bolus baseline frames.

        Returns
        -------
        aif_curve : ndarray
            Validated (possibly re-computed) AIF time course.
        aif_mask : ndarray
            3D boolean mask of voxels used.
        top_candidates : list
            Candidates actually used (may be truncated to 1).
        validation_qc : dict
            Validation metadata for QC reporting.
        """
        n_time = len(aif_curve)

        def _compute_spr(curve: npt.NDArray[Any]) -> tuple[float, int, float]:
            """Return (SPR%, first-pass peak frame, first-pass peak value)."""
            fp_start = baseline_frames + max(1, int(_FP_GAP_SECONDS / tr))
            fp_end = min(baseline_frames + int(40.0 / tr), n_time)
            if fp_start >= n_time or fp_end <= fp_start:
                return 0.0, -1, 0.0

            fp_window = curve[fp_start:fp_end]
            fp_peak_rel = int(np.argmax(fp_window))
            fp_peak_abs = fp_start + fp_peak_rel
            fp_peak_val = float(fp_window[fp_peak_rel])

            baseline = float(np.mean(curve[:baseline_frames]))
            fp_height = fp_peak_val - baseline
            if fp_height < EPSILON_SIGNAL:
                return 0.0, fp_peak_abs, fp_peak_val

            # Recirculation window: 15 s buffer after first-pass window
            recirc_start = fp_end + max(1, int(15.0 / tr))
            if recirc_start >= n_time:
                return 0.0, fp_peak_abs, fp_peak_val

            post_recirc = curve[recirc_start:]
            spr = (float(np.max(post_recirc)) - baseline) / fp_height * 100.0
            return spr, fp_peak_abs, fp_peak_val

        def _global_max_in_fp(curve: npt.NDArray[Any]) -> bool:
            """True if global maximum falls within the first-pass window."""
            fp_start = baseline_frames + max(1, int(_FP_GAP_SECONDS / tr))
            fp_end = min(baseline_frames + int(40.0 / tr), n_time)
            global_max_frame = int(np.argmax(curve))
            return fp_start <= global_max_frame < fp_end

        validation_qc: dict[str, Any] = {}
        spr, fp_peak_frame, fp_peak_val = _compute_spr(aif_curve)
        validation_qc["averaged_aif_spr"] = round(spr, 1)
        validation_qc["averaged_aif_fp_peak_frame"] = fp_peak_frame

        spr_bad = spr > 100.0
        gmax_bad = not _global_max_in_fp(aif_curve)
        needs_retry = spr_bad or gmax_bad

        if needs_retry:
            if spr_bad and gmax_bad:
                validation_qc["averaged_aif_retry_reason"] = "both"
            elif spr_bad:
                validation_qc["averaged_aif_retry_reason"] = "spr"
            else:
                validation_qc["averaged_aif_retry_reason"] = "global_max_outside_fp"

        if not needs_retry:
            # Original curve is acceptable
            aif_mask = np.zeros(conc_data.shape[:3], dtype=bool)
            # Reconstruct mask from candidates with default roi_size=3
            for _, coord, _ in top_candidates:
                slices = tuple(
                    slice(
                        max(0, coord[d] - 1),
                        min(conc_data.shape[d], coord[d] + 2),
                    )
                    for d in range(3)
                )
                roi = np.zeros(conc_data.shape[:3], dtype=bool)
                roi[slices] = True
                aif_mask |= roi & (brain_mask > 0.5)

            validation_qc["averaged_aif_validation_retries"] = 0
            validation_qc["averaged_aif_final_roi_size"] = 3
            return aif_curve, aif_mask, top_candidates, validation_qc

        # Retry 1: re-average with roi_size=1 (no neighborhood)
        retry_curve, retry_mask = AIFDetection._compute_roi_averaged_aif(
            top_candidates, conc_data, brain_mask, roi_size=1,
        )
        spr_r1, _, _ = _compute_spr(retry_curve)
        validation_qc["averaged_aif_spr_retry1"] = round(spr_r1, 1)

        if spr_r1 <= 100.0 and _global_max_in_fp(retry_curve):
            validation_qc["averaged_aif_validation_retries"] = 1
            validation_qc["averaged_aif_final_roi_size"] = 1
            return retry_curve, retry_mask, top_candidates, validation_qc

        # Retry 2: single best candidate only, roi_size=1
        best_only = [top_candidates[0]]
        retry2_curve, retry2_mask = AIFDetection._compute_roi_averaged_aif(
            best_only, conc_data, brain_mask, roi_size=1,
        )
        spr_r2, _, _ = _compute_spr(retry2_curve)
        validation_qc["averaged_aif_spr_retry2"] = round(spr_r2, 1)

        if spr_r2 <= 100.0 and _global_max_in_fp(retry2_curve):
            validation_qc["averaged_aif_validation_retries"] = 2
            validation_qc["averaged_aif_final_roi_size"] = 1
            return retry2_curve, retry2_mask, best_only, validation_qc

        # Final: accept best single-voxel result with warning
        validation_qc["averaged_aif_validation_retries"] = 2
        validation_qc["averaged_aif_final_roi_size"] = 1
        validation_qc["aif_recirculation_warning"] = True
        validation_qc["aif_confidence"] = "reduced"
        return retry2_curve, retry2_mask, best_only, validation_qc

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
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Generate synthetic AIF using physiologically calibrated gamma-variate.

        Uses ΔR2*-scaled peak amplitude with recirculation
        component for more realistic bolus shape.

        Parameters
        ----------
        n_timepoints : int
            Number of time frames.
        tr : float
            Repetition time in seconds.
        baseline_frames : int
            Number of pre-bolus baseline frames.
        tissue_max_delta_r2star : float or None
            95th percentile of max ΔR2* across brain voxels. Used to scale
            AIF peak amplitude. If None, uses literature default (250 s⁻¹).

        Returns
        -------
        aif : ndarray
            Synthetic AIF curve (ΔR2* units, s⁻¹).
        metrics : dict
            QC metadata describing the synthetic AIF parameters.
        """
        # --- First-pass gamma-variate parameters ---
        alpha = 4.0
        time_to_peak = 4.5  # seconds after arrival
        arrival_frame = baseline_frames + 2
        t0 = arrival_frame * tr  # bolus arrival time (seconds)

        # Peak amplitude scaling
        if tissue_max_delta_r2star is not None and tissue_max_delta_r2star > 0:
            peak = float(np.clip(tissue_max_delta_r2star * 6.0, 20.0, 400.0))
            amplitude_source = "tissue_scaled"
        else:
            peak = 250.0  # literature default (Bleeker 2011)
            amplitude_source = "literature_default"

        # Recirculation parameters
        recirc_fraction = 0.35
        recirc_delay = 20.0  # seconds after first-pass peak
        recirc_dispersion = 8.0  # seconds (broadening)

        # --- Build time axis ---
        t = np.arange(n_timepoints) * tr
        aif = np.zeros(n_timepoints, dtype=np.float64)

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
            aif[post_recirc] += recirc_peak * (ratio_r ** alpha) * np.exp(
                alpha * (1.0 - ratio_r)
            )

        # Clip negative values (numerical safety)
        aif = np.clip(aif, 0.0, None)

        # --- QC metrics ---
        metrics: dict[str, Any] = {
            "synthetic_aif_peak_delta_r2star": float(np.max(aif)),
            "synthetic_aif_target_peak": peak,
            "synthetic_aif_amplitude_source": amplitude_source,
            "synthetic_aif_alpha": alpha,
            "synthetic_aif_time_to_peak_s": time_to_peak,
            "synthetic_aif_arrival_frame": arrival_frame,
            "synthetic_aif_recirc_fraction": recirc_fraction,
            "synthetic_aif_recirc_delay_s": recirc_delay,
        }
        if tissue_max_delta_r2star is not None:
            metrics["tissue_max_delta_r2star"] = float(
                tissue_max_delta_r2star
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

        qc: dict[str, Any] = {
            # Candidate counts
            "n_aif_voxels_used": n_used,
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
            "all_candidate_coords": [
                [int(c) for c in coord]
                for _, coord, _ in top_candidates
            ],
            # Hemisphere distribution
            "n_left_hemisphere": n_left,
            "n_right_hemisphere": n_right,
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
            "best_ttp": best_m.get("ttp", 0.0),
            "best_fwhm": best_m.get("fwhm", 0.0),
            "best_ph": best_m.get("ph", 0.0),
            "best_psr": best_m.get("psr", 0.0),
            "best_brr": best_m.get("baseline_return_ratio", 0.0),
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
        else:
            # Plain text: try whitespace-delimited first, then comma
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

        # Compute QC metrics
        baseline_frames = int(self.inputs.baseline_frames)
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
            "detection_level": "external",
            "quality_score": quality_score,
            "peak_delta_r2_star": peak,
            "cnr": cnr,
            "fwhm_frames": fwhm_frames,
            "washout_fraction": washout,
            "baseline_cv": baseline_cv,
            "n_candidates": 0,
            "baseline_frames": baseline_frames,
            # Keys accessed by AIFQCPlot via .get() with NaN defaults
            "best_fwhm": float(fwhm_frames),
            "best_brr": float(washout),
            "best_psr": peak,
        }

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

        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "out_raw_aif_curve": "",
            "out_aif_coord": [0, 0, 0],
            "aif_curve": [float(x) for x in aif_curve],
            "detection_level": "external",
            "quality_score": quality_score,
            "candidate_count": 0,
            "gamma_fit_applied": False,
            "aif_baseline_frames": int(self.inputs.baseline_frames),
            "qc_metrics": qc_metrics,
        }

        return runtime

    # -----------------------------------------------------------------
    # Step 10: Main pipeline  (_run_interface)
    # -----------------------------------------------------------------

    def _run_interface(self, runtime: Any) -> Any:
        """Execute the AIF detection pipeline.

        1. Build attempts list: atlas (strict/relaxed/permissive) then
           brain (relaxed/permissive).
        2. Each attempt: prepare mask -> find candidates -> score ->
           post-scoring filters (distance + contralateral).
        3. On success, finalize metrics and break.
        4. Synthetic fallback if all attempts fail.
        """
        import logging
        import nibabel as nib  # type: ignore[import-untyped]

        _logger = logging.getLogger(__name__)

        # ===== External AIF: bypass all detection if provided =====
        if isdefined(self.inputs.external_aif_file) and self.inputs.external_aif_file:
            return self._handle_external_aif(runtime)

        from pmriprep.utils.dsc_signal import detect_bolus_arrival

        # Load data
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]

        brain_img = nib.load(self.inputs.brain_mask_file)
        brain_mask: npt.NDArray[Any] = brain_img.get_fdata()  # type: ignore[attr-defined]

        if brain_mask.shape[:3] != conc_data.shape[:3]:
            raise RuntimeError(
                f"Brain mask shape {brain_mask.shape[:3]} does not match "
                f"concentration data shape {conc_data.shape[:3]}. "
                "The brain mask must be in native DSC space. "
                "Ensure transform_brain_mask node is connected upstream."
            )

        # Optional atlas mask (user-supplied aif_mask_file takes priority)
        atlas_mask: Optional[npt.NDArray[Any]] = None
        if isdefined(self.inputs.aif_mask_file) and self.inputs.aif_mask_file:
            atlas_img = nib.load(self.inputs.aif_mask_file)
            atlas_mask = atlas_img.get_fdata()  # type: ignore[attr-defined]
        elif self.inputs.atlas_mask_file:
            atlas_img = nib.load(self.inputs.atlas_mask_file)
            atlas_mask = atlas_img.get_fdata()  # type: ignore[attr-defined]

        # Optional lesion mask
        lesion_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.lesion_mask_file:
            lesion_img = nib.load(self.inputs.lesion_mask_file)
            lesion_data = lesion_img.get_fdata()  # type: ignore[attr-defined]
            needs_resample = (
                lesion_data.shape[:3] != conc_data.shape[:3]
                or not np.allclose(lesion_img.affine, conc_img.affine, atol=1e-3)  # type: ignore[attr-defined]
            )
            if needs_resample:
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]
                resampled = resample_to_img(
                    lesion_img, conc_img, interpolation="nearest"
                )
                lesion_mask = (resampled.get_fdata() > 0.5).astype(np.float64)  # type: ignore[attr-defined]
            else:
                lesion_mask = lesion_data

        # Optional enhancing mask for laterality determination
        enhancing_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.enhancing_mask_file:
            enh_img = nib.load(self.inputs.enhancing_mask_file)
            enh_data = enh_img.get_fdata()  # type: ignore[attr-defined]
            needs_resample = (
                enh_data.shape[:3] != conc_data.shape[:3]
                or not np.allclose(enh_img.affine, conc_img.affine, atol=1e-3)  # type: ignore[attr-defined]
            )
            if needs_resample:
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]
                resampled = resample_to_img(
                    enh_img, conc_img, interpolation="nearest"
                )
                enhancing_mask = (resampled.get_fdata() > 0.5).astype(np.float64)  # type: ignore[attr-defined]
            else:
                enhancing_mask = enh_data

        # Load raw signal for S0 and signal-based metrics
        signal_data: Optional[npt.NDArray[Any]] = None
        if self.inputs.signal_file:
            signal_img = nib.load(self.inputs.signal_file)
            signal_data = signal_img.get_fdata()  # type: ignore[attr-defined]

        # Parameters
        aif_method = str(self.inputs.aif_method)
        initial_criteria = str(self.inputs.criteria)
        n_candidates = int(self.inputs.n_candidates)
        baseline_frames = int(self.inputs.baseline_frames)
        gamma_fit_mode = str(self.inputs.gamma_fit)
        disable_fallback = bool(self.inputs.disable_fallback)
        lesion_distance = float(self.inputs.lesion_distance)
        n_timepoints = conc_data.shape[3]

        # Voxel sizes for distance-based lesion exclusion and TR detection
        zooms = conc_img.header.get_zooms()  # type: ignore[attr-defined,union-attr]
        voxel_sizes = tuple(float(z) for z in zooms[:3])

        # TR: from input, or auto-detect from NIfTI header, fallback 1.5 s
        tr = float(self.inputs.tr)
        if tr <= 0:
            if len(zooms) >= 4 and zooms[3] > 0:
                tr = float(zooms[3])
                # Sanity: DSC TR is typically 1-3 s; detect ms/us units
                if tr > 100.0:
                    _logger.warning(
                        "Auto-detected TR=%.1fs from NIfTI header seems too "
                        "large (possible millisecond units). Falling back to 1.5s.",
                        tr,
                    )
                    tr = 1.5
            else:
                tr = 1.5

        # --- Auto-detect actual bolus arrival for AIF first-pass window ---
        # Use 99th percentile across brain voxels — naturally selects arterial
        # timing without transit delay (avoids GAP 6 problem where mean brain
        # concentration detects tissue arrival 4-12 s later than arteries).
        # Derivative-anchored detection operates on tissue signals and needs
        # no AIF voxels — runs even for synthetic AIF.  Skipping it would
        # default to pre_bolus (often 5), catastrophic for long baselines
        # and misaligning synthetic AIF bolus timing with the actual data.
        brain_conc_all = conc_data[brain_mask > 0.5]
        if (
            brain_conc_all.ndim > 1
            and brain_conc_all.shape[0] > 100
        ):
            from pmriprep.utils.baseline_detection import detect_baseline_ensemble

            # Compute multiple percentiles for multi-signal consensus
            p90_curve, p99_curve = np.percentile(
                brain_conc_all, [90, 99], axis=0,
            )
            mean_curve = np.mean(brain_conc_all, axis=0)

            detected_arrival, bl_diag = detect_baseline_ensemble(
                p99_curve, tr=tr,
                min_baseline=max(5, int(10.0 / max(tr, 0.1))),
                additional_signals=[p90_curve, mean_curve],
            )
            aif_baseline_frames = detected_arrival
            _logger.debug(
                "Ensemble baseline detection: arrival=%d, conf=%s, method=%s",
                detected_arrival,
                bl_diag.get("confidence", "unknown"),
                bl_diag.get("method", "unknown"),
            )
        else:
            aif_baseline_frames = baseline_frames  # fallback to configured

        # Initialize
        detection_level = "synthetic"
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        n_available = 0
        filter_meta: dict[str, Any] = {}
        cascade_initial_level: Optional[str] = None
        # Quality gate stash: if an accepted tier has bad peak ratio,
        # stash it and continue cascading once to find a better AIF.
        quality_stash: Optional[
            tuple[list, int, str, dict[str, Any], Optional[float]]
        ] = None  # (scored, n_avail, level_name, f_meta, peak_ratio)

        if aif_method == "synthetic":
            pass  # Go directly to synthetic
        else:
            # --- GAP 7: Unified attempts loop ---
            # Build full attempts list, then filter by initial_criteria
            criteria_order = ["strict", "relaxed", "permissive"]
            try:
                start_idx = criteria_order.index(initial_criteria)
            except ValueError:
                start_idx = 0

            all_attempts: list[tuple[str, Optional[npt.NDArray[Any]], str]] = []
            if atlas_mask is not None:
                for crit in ["strict", "relaxed", "permissive"]:
                    all_attempts.append((crit, atlas_mask, f"atlas_{crit}"))

            has_roi_constraint = atlas_mask is not None
            strict_roi = bool(self.inputs.strict_roi_constraint)
            if not (has_roi_constraint and strict_roi):
                # Note: brain_degraded removed (v0.19.16) - thresholds too loose,
                # admitted non-arterial curves. Synthetic fallback is more reliable.
                for crit in ["relaxed", "permissive"]:
                    all_attempts.append((crit, None, f"brain_{crit}"))

            # Skip attempts with criteria stricter than initial_criteria
            attempts = [
                (crit, roi, name)
                for crit, roi, name in all_attempts
                if criteria_order.index(crit) >= start_idx
            ]

            for crit, roi, level_name in attempts:
                frac = _CASCADE_N_CANDIDATES_FRACTION.get(crit, 0.6)
                level_n_candidates = max(
                    MIN_AIF_CANDIDATES_ACCEPT,
                    math.ceil(frac * n_candidates),
                )

                search = self._prepare_search_mask(
                    brain_mask, roi, lesion_mask
                )
                cands, peak_map = self._find_aif_candidate_voxels(
                    conc_data, search, aif_baseline_frames
                )
                if cands.shape[0] == 0:
                    continue
                tissue_ttp = self._calculate_tissue_ttp_reference(
                    conc_data, brain_mask, aif_baseline_frames
                )
                scored, n_avail = self._score_and_select_candidates(
                    cands, conc_data, signal_data, peak_map,
                    brain_mask, tr, aif_baseline_frames, crit,
                    tissue_ttp,
                    atlas_mask=atlas_mask if roi is None else None,
                )
                if not scored:
                    continue

                # Post-scoring filters (distance + soft contralateral)
                scored, f_meta = self._apply_post_scoring_filters(
                    scored, lesion_mask, affine, voxel_sizes,
                    lesion_min_distance_mm=lesion_distance,
                    enhancing_mask=enhancing_mask,
                )
                if not scored:
                    continue

                # Bilateral balance (always applied)
                scored = self._ensure_bilateral_balance(
                    scored, affine, level_n_candidates,
                )

                if cascade_initial_level is None:
                    cascade_initial_level = level_name

                if len(scored) >= MIN_AIF_CANDIDATES_ACCEPT:
                    # Quality gate: check gamma peak ratio before accepting
                    if quality_stash is None:
                        temp_aif, _ = self._compute_roi_averaged_aif(
                            scored, conc_data, brain_mask
                        )
                        temp_aif, _, _, _ = self._validate_averaged_aif(
                            temp_aif, scored, conc_data, brain_mask,
                            tr, aif_baseline_frames,
                        )
                        pr = _compute_aif_peak_ratio(
                            temp_aif, aif_baseline_frames, tr
                        )
                        if (pr is None
                                or pr < GAMMA_PEAK_RATIO_CASCADE_MIN
                                or pr > GAMMA_PEAK_RATIO_CASCADE_MAX):
                            # Bad quality (gamma fit failed or peak ratio out of band)
                            # — stash and continue to next tier
                            quality_stash = (
                                scored, n_avail, level_name, f_meta, pr,
                            )
                            continue

                    if quality_stash is not None:
                        # Second tier with >= MIN — compare with stashed
                        temp_aif2, _ = self._compute_roi_averaged_aif(
                            scored, conc_data, brain_mask
                        )
                        temp_aif2, _, _, _ = self._validate_averaged_aif(
                            temp_aif2, scored, conc_data, brain_mask,
                            tr, aif_baseline_frames,
                        )
                        pr2 = _compute_aif_peak_ratio(
                            temp_aif2, aif_baseline_frames, tr
                        )
                        stashed_pr = quality_stash[4]
                        # Pick whichever has peak ratio closer to 1.0
                        use_current = (
                            pr2 is not None
                            and (stashed_pr is None
                                 or abs(pr2 - 1.0) < abs(stashed_pr - 1.0))
                        )
                        if use_current:
                            top_candidates = scored
                            n_available = n_avail
                            detection_level = level_name
                            filter_meta = f_meta
                            break
                        elif pr2 is None and stashed_pr is None:
                            if len(scored) > len(quality_stash[0]):
                                quality_stash = (
                                    scored, n_avail, level_name, f_meta, pr2,
                                )
                            continue
                        else:
                            (top_candidates, n_available,
                             detection_level, filter_meta, _) = quality_stash
                            break

                    # Good quality — accept normally
                    top_candidates = scored
                    n_available = n_avail
                    detection_level = level_name
                    filter_meta = f_meta
                    break
                elif len(scored) > len(top_candidates):
                    # Below MIN — stash best-so-far but keep cascading.
                    # With variable level_n_candidates, equal counts across levels
                    # favor the stricter (earlier) level by design.
                    top_candidates = scored
                    n_available = n_avail
                    detection_level = level_name
                    filter_meta = f_meta

            # If quality gate stashed a result but loop exhausted without
            # finding a better tier, use the stashed (>= MIN) result.
            if quality_stash is not None and (
                not top_candidates
                or len(top_candidates) < MIN_AIF_CANDIDATES_ACCEPT
            ):
                (top_candidates, n_available,
                 detection_level, filter_meta, _) = quality_stash

            if top_candidates and len(top_candidates) < MIN_AIF_CANDIDATES_ACCEPT:
                import logging
                logging.getLogger(__name__).warning(
                    "AIF cascade found only %d candidates (minimum %d preferred) "
                    "at level '%s'. Quantification may be less stable.",
                    len(top_candidates), MIN_AIF_CANDIDATES_ACCEPT,
                    detection_level,
                )

        # ---- Build AIF curve ----
        aif_curve: npt.NDArray[Any]
        aif_mask_arr = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        quality_score = 0.0
        qc_metrics: dict[str, Any] = {}

        # Track initial baseline for QC reporting
        baseline_frames_initial = baseline_frames

        if top_candidates:
            aif_curve, aif_bool_mask = self._compute_roi_averaged_aif(
                top_candidates, conc_data, brain_mask
            )

            # Validate averaged AIF (retry with smaller ROI if needed)
            aif_curve, aif_bool_mask, top_candidates, validation_qc = (
                self._validate_averaged_aif(
                    aif_curve, top_candidates, conc_data, brain_mask,
                    tr, aif_baseline_frames,
                )
            )
            aif_mask_arr[aif_bool_mask] = 1
            for _, coord, _ in top_candidates:
                aif_mask_arr[coord[0], coord[1], coord[2]] = 1

            # Stage 2: re-detect baseline on final smoothed AIF curve
            stage2_window = min(
                aif_baseline_frames, max(2, len(aif_curve) // 3)
            )
            _, aif_detected_baseline = detect_bolus_arrival(
                aif_curve,
                baseline_frames=stage2_window,
                n_consecutive=3,
            )
            if aif_detected_baseline > baseline_frames:
                baseline_frames = aif_detected_baseline

            # Quality: inverse of mean cost, clamped to [0, 1]
            mean_cost = float(np.mean([c for c, _, _ in top_candidates]))
            quality_score = min(1.0, 1.0 / (1.0 + mean_cost))

            # --- GAP 8: Finalize AIF metrics ---
            aif_curve, finalized_qc = self._finalize_aif_metrics(
                aif_curve, top_candidates, conc_data, brain_mask,
                affine, tr, n_candidates, n_available,
                baseline_frames=baseline_frames,
            )
            qc_metrics.update(finalized_qc)
            qc_metrics.update(filter_meta)
            if cascade_initial_level is not None:
                qc_metrics["cascade_downgraded"] = (
                    cascade_initial_level != detection_level
                )
                qc_metrics["cascade_initial_level"] = cascade_initial_level
                qc_metrics["cascade_final_level"] = detection_level
            if quality_stash is not None:
                qc_metrics["cascade_quality_gate_triggered"] = True
                qc_metrics["cascade_stashed_level"] = quality_stash[2]
                qc_metrics["cascade_stashed_peak_ratio"] = quality_stash[4]
            else:
                qc_metrics["cascade_quality_gate_triggered"] = False
            qc_metrics.update(validation_qc)

        elif not disable_fallback:
            # Synthetic fallback: no candidates found at any cascade level
            tissue_max = self._compute_tissue_max_for_synthetic(conc_data, brain_mask)
            aif_curve, synth_metrics = self._generate_synthetic_aif(
                n_timepoints, tr, aif_baseline_frames,
                tissue_max_delta_r2star=tissue_max,
            )
            detection_level = "synthetic"
            quality_score = 0.3
            qc_metrics = {
                "synthetic": True,
                "n_aif_voxels_used": 0,
                "n_aif_voxels_available": 0,
                "synthetic_fallback_reason": "no_candidates",
            }
            qc_metrics.update(synth_metrics)
        else:
            raise RuntimeError(
                "AIF detection failed at all levels and "
                "synthetic fallback is disabled"
            )

        qc_metrics["detection_level"] = detection_level
        qc_metrics["n_candidates_effective"] = len(top_candidates)
        qc_metrics["s0_rejection_available"] = signal_data is not None
        qc_metrics["tr_used"] = tr
        qc_metrics["baseline_frames_initial"] = baseline_frames  # configured S0 baseline
        qc_metrics["aif_baseline_frames"] = aif_baseline_frames  # auto-detected for AIF
        qc_metrics["baseline_frames_used"] = baseline_frames
        qc_metrics["baseline_detection_method"] = "auto_p99"

        if detection_level == "synthetic":
            qc_metrics["synthetic_aif_warnings"] = [
                "Synthetic AIF used - automatic detection failed",
                "Relative quantification only (rCBV, rCBF normalized to contralateral)",
                "TTP and Tmax are valid (timing independent of AIF amplitude)",
                "Manual AIF selection recommended for clinical use",
            ]
        # Note: brain_degraded tier removed in v0.19.16 - cascade now goes
        # directly from brain_permissive to synthetic AIF.

        # Gamma-variate fitting
        gamma_fit_applied = False
        raw_aif_curve = None
        if gamma_fit_mode != "off" and detection_level != "synthetic":
            gamma_fit_applied, raw_aif_curve, aif_curve = (
                self._apply_gamma_fit_if_appropriate(
                    aif_curve, gamma_fit_mode, baseline_frames, tr, qc_metrics,
                )
            )
        elif gamma_fit_mode != "off" and detection_level == "synthetic":
            qc_metrics["gamma_fit_applied"] = False
            qc_metrics["gamma_fit_rejected_reason"] = "synthetic_aif"

        # Post-gamma-fit quality gate: reject AIF if gamma fit is catastrophically bad
        # R² < 0 means the gamma model fits worse than a horizontal line — no bolus shape.
        if (
            not gamma_fit_applied
            and not disable_fallback
            and detection_level != "synthetic"
        ):
            gamma_r2 = qc_metrics.get("gamma_fit_r_squared")
            if gamma_r2 is not None and gamma_r2 < GAMMA_R2_SYNTHETIC_FALLBACK:
                _logger.warning(
                    "Gamma fit R²=%.3f < %.1f; falling back to synthetic AIF",
                    gamma_r2,
                    GAMMA_R2_SYNTHETIC_FALLBACK,
                )
                # Synthetic fallback: gamma fit catastrophically bad
                tissue_max = self._compute_tissue_max_for_synthetic(conc_data, brain_mask)
                aif_curve, synth_metrics = self._generate_synthetic_aif(
                    n_timepoints, tr, aif_baseline_frames,
                    tissue_max_delta_r2star=tissue_max,
                )
                detection_level = "synthetic"
                quality_score = 0.3
                aif_mask_arr[:] = 0  # Clear AIF mask (no voxels used)
                # Update QC metrics
                qc_metrics["detection_level"] = detection_level
                qc_metrics["synthetic"] = True
                qc_metrics["n_aif_voxels_used"] = 0
                qc_metrics["n_aif_voxels_available"] = 0
                qc_metrics["synthetic_fallback_reason"] = "gamma_r2_floor"
                qc_metrics["gamma_r2_at_fallback"] = gamma_r2
                qc_metrics.update(synth_metrics)
                qc_metrics["synthetic_aif_warnings"] = [
                    "Synthetic AIF used - gamma fit R² below threshold",
                    "Relative quantification only (rCBV, rCBF normalized to contralateral)",
                    "TTP and Tmax are valid (timing independent of AIF amplitude)",
                    "Manual AIF selection recommended for clinical use",
                ]

        # Save outputs
        out_aif_curve = self._gen_outfilename("_aif_curve.txt")
        np.savetxt(out_aif_curve, aif_curve)

        out_raw_aif = ""
        if raw_aif_curve is not None:
            out_raw_aif = self._gen_outfilename("_raw_aif_curve.txt")
            np.savetxt(out_raw_aif, raw_aif_curve)

        out_aif_mask = self._gen_outfilename("_aif_mask.nii.gz")
        mask_img = nib.Nifti1Image(aif_mask_arr, affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_aif_mask)  # type: ignore[attr-defined]

        n_cand = len(top_candidates) if top_candidates else 0
        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "out_raw_aif_curve": out_raw_aif,
            "out_aif_coord": qc_metrics.get("aif_voxel_coord", []),
            "aif_curve": [float(x) for x in aif_curve],
            "detection_level": detection_level,
            "quality_score": quality_score,
            "candidate_count": n_cand,
            "gamma_fit_applied": gamma_fit_applied,
            "aif_baseline_frames": aif_baseline_frames,
            "qc_metrics": qc_metrics,
        }

        return runtime

    @staticmethod
    def _apply_gamma_fit_if_appropriate(
        aif_curve: npt.NDArray[Any],
        gamma_fit_mode: str,
        baseline_frames: int,
        tr: float,
        qc_metrics: dict[str, Any],
    ) -> tuple[bool, Optional[npt.NDArray[Any]], npt.NDArray[Any]]:
        """Attempt gamma-variate fitting and decide whether to apply it.

        Parameters
        ----------
        aif_curve : np.ndarray
            Current AIF curve (may be modified if fit is applied).
        gamma_fit_mode : str
            "on" (legacy: apply if R²>0.5) or "auto" (rule-based).
        baseline_frames : int
            Number of pre-bolus baseline frames.
        tr : float
            Repetition time in seconds.
        qc_metrics : dict
            QC metrics dict (modified in place).

        Returns
        -------
        tuple[bool, Optional[np.ndarray], np.ndarray]
            (gamma_fit_applied, raw_aif_curve, aif_curve)
            raw_aif_curve is the original AIF when fit is applied,
            or the fitted curve when fit is rejected (for QC overlay).
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

            if gamma_fit_mode == "on":
                # Legacy mode: accept if converged and R² > 0.5
                if fit_result.fit_converged and fit_result.r_squared > 0.5:
                    qc_metrics["gamma_fit_applied"] = True
                    qc_metrics["gamma_fit_mode"] = "on"
                    qc_metrics["gamma_fit_r_squared"] = fit_result.r_squared
                    qc_metrics["gamma_fit_auc_ratio"] = fit_result.auc_ratio
                    qc_metrics["gamma_recirculation_removed"] = (
                        fit_result.recirculation_removed
                    )
                    return True, original_aif, fit_result.fitted_curve
                else:
                    qc_metrics["gamma_fit_applied"] = False
                    qc_metrics["gamma_fit_mode"] = "on"
                    qc_metrics["gamma_fit_rejected_reason"] = (
                        "no_convergence"
                        if not fit_result.fit_converged
                        else f"low_r_squared_{fit_result.r_squared:.3f}"
                    )
                    return False, None, aif_curve

            # Auto mode
            decision, detail_metrics = should_apply_gamma_fit(
                original_aif=original_aif,
                fit_result=fit_result,
                baseline_frames=baseline_frames,
            )
            qc_metrics["gamma_fit_mode"] = "auto"
            qc_metrics["gamma_fit_auto_metrics"] = detail_metrics
            qc_metrics["gamma_fit_r_squared"] = fit_result.r_squared
            qc_metrics["gamma_fit_auc_ratio"] = fit_result.auc_ratio

            if decision.apply:
                qc_metrics["gamma_fit_applied"] = True
                qc_metrics["gamma_fit_auto_reason"] = decision.reason
                qc_metrics["gamma_fit_auto_defect"] = decision.defect_detected
                qc_metrics["gamma_recirculation_removed"] = (
                    fit_result.recirculation_removed
                )
                return True, original_aif, fit_result.fitted_curve
            else:
                qc_metrics["gamma_fit_applied"] = False
                qc_metrics["gamma_fit_rejected_reason"] = decision.reason
                # Save fitted curve for QC overlay (rejected fit shown dashed)
                qc_metrics["gamma_fit_rejection_curve_available"] = True
                return False, fit_result.fitted_curve, aif_curve

        except Exception:
            qc_metrics["gamma_fit_applied"] = False
            qc_metrics["gamma_fit_rejected_reason"] = "exception"
            return False, None, aif_curve

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
