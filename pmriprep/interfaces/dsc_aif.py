"""AIF (Arterial Input Function) detection interfaces for DSC-MRI.

This module implements Nipype interfaces for AIF detection in DSC perfusion
processing, with multi-level detection strategies (atlas-based, relaxed
criteria, synthetic fallback) and comprehensive quality metrics.

Ported from perf-plus ``dsc_compute_maps.py`` AIF pipeline to match its
search-mask preparation, adaptive-percentile candidate pre-filtering,
SSS/edge spatial exclusions, tissue-TTP cross-validation, cost-function
scoring, 3x3x3 ROI averaging, and Savitzky-Golay smoothing.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation, binary_erosion  # type: ignore[import-untyped]
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)

from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
    EPSILON_SIGNAL,
)


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
    lesion_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 3D lesion mask for exclusion",
    )

    signal_file = File(
        exists=True,
        mandatory=False,
        desc="4D raw signal for S0-based partial volume rejection",
    )

    # Detection parameters
    aif_method = traits.Enum(
        "auto",
        "population",
        usedefault=True,
        desc="AIF detection method: 'auto' (multi-level) or 'population' (synthetic)",
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
    prefer_contralateral = traits.Bool(
        True,
        usedefault=True,
        desc="Prefer contralateral AIF candidates when lesion present",
    )
    gamma_fit = traits.Bool(
        False,
        usedefault=True,
        desc="Apply gamma-variate fitting to AIF",
    )
    disable_fallback = traits.Bool(
        False,
        usedefault=True,
        desc="Disable synthetic AIF fallback when detection fails",
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
    aif_curve = traits.List(traits.Float, desc="AIF concentration curve values")
    detection_level = traits.Str(desc="Detection level: 'atlas', 'relaxed', 'synthetic'")
    quality_score = traits.Float(desc="Overall AIF quality score (0-1)")
    candidate_count = traits.Int(desc="Number of candidate voxels evaluated")
    gamma_fit_applied = traits.Bool(desc="Whether gamma-variate fitting was applied")
    qc_metrics = traits.Dict(desc="Detailed QC metrics dictionary")


class AIFDetection(BaseInterface):
    """Arterial Input Function (AIF) detection for DSC-MRI.

    Implements the perf-plus AIF detection pipeline with:
    - Brain mask erosion (4 iterations) to exclude cortical veins
    - Adaptive percentile candidate pre-filtering (99.5% -> 95%)
    - SSS / edge-slice spatial exclusions
    - Tissue TTP cross-validation
    - Cost-function scoring (lower = better) with S0-dominant weighting
    - 3x3x3 ROI averaging with Savitzky-Golay smoothing
    - Multi-level fallback: atlas strict/relaxed -> brain strict->degraded -> synthetic
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
    # Step 2: Search mask preparation  (perf-plus _prepare_aif_search_mask)
    # -----------------------------------------------------------------

    @staticmethod
    def _prepare_search_mask(
        brain_mask: npt.NDArray[Any],
        roi_mask: Optional[npt.NDArray[Any]],
        lesion_mask: Optional[npt.NDArray[Any]],
        erosion_iterations: int = 4,
        lesion_dilation: int = 0,
    ) -> npt.NDArray[Any]:
        """Prepare binary search mask with erosion and exclusion.

        Matches perf-plus ``_prepare_aif_search_mask``: erosion for
        cortical vein exclusion, optional ROI intersection, and binary
        lesion exclusion with optional dilation buffer.  Distance-based
        lesion filtering happens *after* scoring (see
        ``_apply_post_scoring_filters``).

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
        lesion_dilation : int
            Number of binary dilation iterations for lesion buffer
            (default 0, use mask as-is).

        Returns
        -------
        ndarray
            3D boolean search mask.
        """
        eroded = binary_erosion(
            brain_mask > 0.5, iterations=erosion_iterations
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
            if lesion_dilation > 0:
                lesion_bool = binary_dilation(
                    lesion_bool, iterations=lesion_dilation
                )
            pre_lesion = search.copy()
            search = search & ~lesion_bool
            if not np.any(search):
                search = pre_lesion

        return search

    # -----------------------------------------------------------------
    # Step 2b: World-coordinate hemisphere methods
    # (perf-plus determine_tumor_laterality + filter_aif_candidates_by_hemisphere)
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
        prefer_contralateral: bool = True,
    ) -> tuple[
        list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        str,
        int,
        int,
    ]:
        """Filter scored candidates by hemisphere preference.

        For non-bilateral tumors, prefers contralateral hemisphere
        candidates.  Falls back to ipsilateral if no contralateral
        candidates are available.

        Parameters
        ----------
        top_candidates : list
            List of ``(cost, (x,y,z), metrics)`` tuples.
        affine : ndarray
            4x4 affine transformation matrix (voxel -> RAS world).
        tumor_laterality : str
            ``'left'``, ``'right'``, or ``'bilateral'``.
        prefer_contralateral : bool
            Whether to prefer contralateral candidates.

        Returns
        -------
        tuple of (filtered, hemisphere_used, n_contra, n_ipsi)
            filtered : candidates restricted to preferred hemisphere.
            hemisphere_used : ``'contralateral'``, ``'ipsilateral'``,
                or ``'both'``.
            n_contra : count of contralateral candidates.
            n_ipsi : count of ipsilateral candidates.
        """
        if (
            tumor_laterality == "bilateral"
            or not prefer_contralateral
        ):
            return top_candidates, "both", 0, 0

        contra_side = "right" if tumor_laterality == "left" else "left"

        contra: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        ipsi: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []

        for entry in top_candidates:
            _, coord, _ = entry
            coord_homog = np.array(
                [coord[0], coord[1], coord[2], 1.0]
            )
            world = affine @ coord_homog
            cand_side = "left" if world[0] < 0 else "right"

            if cand_side == contra_side:
                contra.append(entry)
            else:
                ipsi.append(entry)

        n_contra = len(contra)
        n_ipsi = len(ipsi)

        if n_contra >= 1:
            return contra, "contralateral", n_contra, n_ipsi

        # No contralateral candidates — fall back to ipsilateral
        filtered = ipsi if n_ipsi > 0 else top_candidates
        return filtered, "ipsilateral", n_contra, n_ipsi

    # -----------------------------------------------------------------
    # Step 3: Spatial exclusions  (perf-plus _apply_spatial_exclusions)
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
    # Step 4: Adaptive percentile candidate finder
    # (perf-plus _find_aif_candidate_voxels)
    # -----------------------------------------------------------------

    def _find_aif_candidate_voxels(
        self,
        conc_data: npt.NDArray[Any],
        search_mask: npt.NDArray[Any],
        baseline_frames: int,
        min_candidates: int = 10,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Find initial candidate voxels via adaptive percentile thresholding.

        Parameters
        ----------
        conc_data : ndarray
            4D concentration (x, y, z, t).
        search_mask : ndarray
            3D boolean mask.
        baseline_frames : int
            Number of pre-bolus frames.
        min_candidates : int
            Desired minimum number of candidates.

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

        percentiles = [99.5, 99.0, 98.5, 98.0, 97.0, 95.0]
        candidates: Optional[npt.NDArray[Any]] = None

        for pct in percentiles:
            thresh = float(np.nanpercentile(valid_peaks, pct))
            candidates = np.argwhere(peak_conc_map >= thresh)
            if candidates.shape[0] >= min_candidates:
                break

        if candidates is None or candidates.shape[0] == 0:
            return np.empty((0, 3), dtype=int), peak_conc_map

        # Apply spatial exclusions
        candidates = self._apply_spatial_exclusions(
            candidates, conc_data.shape[:3]
        )

        return candidates, peak_conc_map

    # -----------------------------------------------------------------
    # Step 5: Tissue TTP reference
    # (perf-plus _calculate_tissue_ttp_reference)
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
    # Step 6: perf-plus metric calculation
    # (perf-plus calculate_aif_quality_metrics + calculate_signal_metrics)
    # -----------------------------------------------------------------

    @staticmethod
    def _calculate_candidate_metrics(
        conc_curve: npt.NDArray[Any],
        signal_curve: Optional[npt.NDArray[Any]],
        tr: float,
        baseline_frames: int,
    ) -> Optional[dict[str, float]]:
        """Compute comprehensive AIF quality metrics for one candidate.

        Matches perf-plus ``calculate_aif_quality_metrics`` output dict.

        Returns None if the curve is too short or degenerate.
        """
        n_time = len(conc_curve)
        if n_time < baseline_frames + 10:
            return None

        baseline_end_time = (baseline_frames - 1) * tr if baseline_frames > 0 else 0.0
        baseline_conc = float(np.mean(conc_curve[:baseline_frames]))

        # First-pass window: 10-40 s after baseline
        fp_start = baseline_frames + max(1, int(10.0 / tr))
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
    # Step 7: Cost-function scoring  (perf-plus score_aif_candidate)
    # -----------------------------------------------------------------

    @staticmethod
    def _score_candidate_cost(
        metrics: dict[str, float],
        criteria: str = "strict",
        median_s0: Optional[float] = None,
    ) -> float:
        """Score a candidate with perf-plus cost function.

        Lower is better.  Returns ``inf`` for physiologically implausible
        candidates (hard rejection).

        Parameters
        ----------
        metrics : dict
            Output of ``_calculate_candidate_metrics``.
        criteria : str
            One of 'strict', 'relaxed', 'permissive', 'degraded'.
        median_s0 : float or None
            Median baseline signal for S0 partial-volume rejection.

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

        # 2. BRR
        brr = metrics.get("baseline_return_ratio", float("nan"))
        brr_thresh = {"strict": 30, "relaxed": 40, "permissive": 50, "degraded": 70}
        if not np.isnan(brr) and brr > brr_thresh.get(criteria, 70):
            return float("inf")

        # 3. PBR
        pbr = metrics.get("peak_to_baseline_ratio", float("nan"))
        pbr_thresh = {"strict": 3.0, "relaxed": 2.0, "permissive": 1.5, "degraded": 1.2}
        if not np.isnan(pbr) and pbr < pbr_thresh.get(criteria, 1.2):
            return float("inf")

        # 4. SPR
        spr = metrics.get("secondary_peak_ratio", 0.0)
        spr_thresh = {"strict": 70, "relaxed": 85, "permissive": 100, "degraded": 150}
        if spr > spr_thresh.get(criteria, 150):
            return float("inf")

        # 5. PMI
        pmi = metrics.get("post_peak_monotonicity", float("nan"))
        pmi_thresh = {"strict": 60, "relaxed": 50, "permissive": 40, "degraded": 30}
        if not np.isnan(pmi) and pmi < pmi_thresh.get(criteria, 30):
            return float("inf")

        # 5b. FWHM max
        fwhm = metrics.get("fwhm", float("nan"))
        fwhm_max = {"strict": 10.0, "relaxed": 15.0, "permissive": 25.0, "degraded": 40.0}
        if not np.isnan(fwhm) and fwhm > fwhm_max.get(criteria, 40.0):
            return float("inf")

        # 6. S0 partial volume rejection
        if median_s0 is not None and not np.isnan(metrics.get("s0", float("nan"))):
            s0_factor = {"strict": 0.40, "relaxed": 0.60, "permissive": 0.80, "degraded": 0.95}
            s0_threshold = median_s0 * s0_factor.get(criteria, 0.95)
            if metrics["s0"] > s0_threshold:
                return float("inf")

        # 7. TTP window
        ttp_ranges: dict[str, tuple[float, float]] = {
            "strict": (5.0, 40.0),
            "relaxed": (3.0, 50.0),
            "permissive": (2.0, 60.0),
            "degraded": (1.0, 70.0),
        }
        min_ttp, max_ttp = ttp_ranges.get(criteria, (1.0, 70.0))
        if not (min_ttp <= ttp_after_baseline <= max_ttp):
            return float("inf")

        # 8. PH min
        ph = metrics.get("ph", float("nan"))
        ph_min = {"strict": 25, "relaxed": 15, "permissive": 10, "degraded": 5}
        if not np.isnan(ph) and ph < ph_min.get(criteria, 5):
            return float("inf")

        # 9. PSR max
        psr = metrics.get("psr", float("nan"))
        psr_max = {"strict": 110, "relaxed": 120, "permissive": 130, "degraded": 150}
        if not np.isnan(psr) and psr > psr_max.get(criteria, 150):
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
            s0_score = (s0_ratio - 0.35) * 20.0

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
    # (perf-plus _score_and_select_candidates)
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
        n_top: int = 10,
    ) -> tuple[list[tuple[float, tuple[int, int, int], dict[str, Any]]], int]:
        """Score pre-filtered candidates and return top N.

        Returns
        -------
        tuple of (top_candidates, n_available)
            top_candidates : list of (cost, (x,y,z), metrics) sorted ascending.
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

            if np.isfinite(cost):
                scored.append((cost, ix, m))

        # Sort by cost ascending (lower = better)
        scored.sort(key=lambda x: x[0])
        n_available = len(scored)
        return scored[:n_top], n_available

    # -----------------------------------------------------------------
    # Step 8b: Post-scoring filters (distance + contralateral)
    # (perf-plus steps 4.5 + 4.6 in _find_aif_single_attempt)
    # -----------------------------------------------------------------

    def _apply_post_scoring_filters(
        self,
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        lesion_mask: Optional[npt.NDArray[Any]],
        affine: npt.NDArray[Any],
        voxel_sizes: tuple[float, ...],
        lesion_min_distance_mm: float = 15.0,
        prefer_contralateral: bool = True,
    ) -> tuple[
        list[tuple[float, tuple[int, int, int], dict[str, Any]]],
        dict[str, Any],
    ]:
        """Apply distance and hemisphere filters after scoring.

        Matches perf-plus steps 4.5 (distance) and 4.6 (contralateral).

        Parameters
        ----------
        top_candidates : list
            Scored ``(cost, (x,y,z), metrics)`` tuples.
        lesion_mask : ndarray or None
            3D lesion mask.
        affine : ndarray
            4x4 affine transformation matrix (voxel -> RAS world).
        voxel_sizes : tuple of float
            Voxel dimensions in mm ``(dx, dy, dz)``.
        lesion_min_distance_mm : float
            Minimum Euclidean distance from lesion in mm.
        prefer_contralateral : bool
            Whether to prefer contralateral hemisphere candidates.

        Returns
        -------
        tuple of (filtered_candidates, filter_metadata)
        """
        import warnings

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
                warnings.warn(
                    f"All {len(top_candidates)} AIF candidates were "
                    f"<{lesion_min_distance_mm}mm from lesion. "
                    "Keeping closest candidate(s).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                for _, _, m in top_candidates:
                    m["lesion_distance_warning"] = True

        # --- Step 4.6: Contralateral hemisphere preference ---
        tumor_laterality: Optional[str] = None
        hemisphere_used = "both"
        n_contra = 0
        n_ipsi = 0

        if (
            lesion_mask is not None
            and prefer_contralateral
            and np.any(lesion_mask > 0.5)
        ):
            tumor_laterality = self._determine_tumor_laterality(
                lesion_mask, affine
            )
            if tumor_laterality is not None and tumor_laterality != "bilateral":
                (
                    top_candidates,
                    hemisphere_used,
                    n_contra,
                    n_ipsi,
                ) = self._filter_candidates_by_hemisphere(
                    top_candidates, affine, tumor_laterality,
                    prefer_contralateral=prefer_contralateral,
                )
            elif tumor_laterality == "bilateral":
                hemisphere_used = "both"

        meta["tumor_laterality"] = tumor_laterality
        meta["hemisphere_used"] = hemisphere_used
        meta["n_contralateral"] = n_contra
        meta["n_ipsilateral"] = n_ipsi

        return top_candidates, meta

    # -----------------------------------------------------------------
    # Step 9: ROI averaging + smoothing
    # (perf-plus _compute_weighted_aif + _finalize_aif_metrics)
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

        # Clip negative to zero
        aif_curve = np.clip(aif_curve, 0.0, None)

        return aif_curve, aif_mask

    # -----------------------------------------------------------------
    # Synthetic AIF (unchanged)
    # -----------------------------------------------------------------

    def _generate_synthetic_aif(
        self,
        n_timepoints: int,
        tr: float = 1.5,
        baseline_frames: int = 5,
    ) -> npt.NDArray[Any]:
        """Generate synthetic (population) AIF using gamma-variate model."""
        t0 = baseline_frames * tr
        alpha = 3.5
        beta = 1.5
        amplitude = 10.0

        t = np.arange(n_timepoints) * tr
        aif = np.zeros(n_timepoints, dtype=np.float64)
        post_arrival = t > t0

        if np.any(post_arrival):
            t_shifted = t[post_arrival] - t0
            t_max = alpha * beta
            norm_factor = (t_max ** alpha) * np.exp(-alpha)
            aif[post_arrival] = (
                amplitude
                * (t_shifted ** alpha)
                * np.exp(-t_shifted / beta)
                / norm_factor
            )

        return aif

    # -----------------------------------------------------------------
    # Step 9b: Finalize AIF metrics
    # (perf-plus _finalize_aif_metrics)
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

        Returns
        -------
        tuple of (aif_curve, qc_dict)
        """
        from pmriprep.utils.dsc_signal import detect_bolus_arrival

        # Bolus arrival on the final AIF curve
        arrival_idx, n_baseline_detected = detect_bolus_arrival(
            aif_curve, baseline_frames=3
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
            "n_aif_candidates_used": n_used,
            "n_aif_candidates_requested": n_candidates_requested,
            "n_aif_candidates_available": n_available,
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
    # Step 10: Main pipeline  (_run_interface)
    # -----------------------------------------------------------------

    def _run_interface(self, runtime: Any) -> Any:
        """Execute AIF detection with perf-plus pipeline.

        Pipeline mirrors perf-plus ``find_aif`` / ``_find_aif_single_attempt``:
        1. Build attempts list: atlas (strict/relaxed/permissive) then
           brain (relaxed/permissive/degraded).
        2. Each attempt: prepare mask -> find candidates -> score ->
           post-scoring filters (distance + contralateral).
        3. On success, finalize metrics and break.
        4. Synthetic fallback if all attempts fail.
        """
        import nibabel as nib  # type: ignore[import-untyped]

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

        # Optional atlas mask
        atlas_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.atlas_mask_file:
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
        gamma_fit = bool(self.inputs.gamma_fit)
        disable_fallback = bool(self.inputs.disable_fallback)
        lesion_distance = float(self.inputs.lesion_distance)
        prefer_contralateral = bool(self.inputs.prefer_contralateral)
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
                    import warnings
                    warnings.warn(
                        f"Auto-detected TR={tr:.1f}s from NIfTI header seems too "
                        "large (possible millisecond units). Falling back to 1.5s.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    tr = 1.5
            else:
                tr = 1.5

        # --- GAP 6: Auto-detect baseline frames ---
        mean_brain_conc = np.mean(
            conc_data[brain_mask > 0.5], axis=0
        )
        _, detected_baseline = detect_bolus_arrival(
            mean_brain_conc, baseline_frames=3
        )
        if detected_baseline > baseline_frames:
            baseline_frames = detected_baseline

        # Initialize
        detection_level = "synthetic"
        top_candidates: list[tuple[float, tuple[int, int, int], dict[str, Any]]] = []
        n_available = 0
        filter_meta: dict[str, Any] = {}

        if aif_method == "population":
            pass  # Go directly to synthetic
        else:
            # --- GAP 7: Unified attempts loop ---
            # Build full attempts list, then filter by initial_criteria
            criteria_order = ["strict", "relaxed", "permissive", "degraded"]
            try:
                start_idx = criteria_order.index(initial_criteria)
            except ValueError:
                start_idx = 0

            all_attempts: list[tuple[str, Optional[npt.NDArray[Any]], str]] = []
            if atlas_mask is not None:
                for crit in ["strict", "relaxed", "permissive"]:
                    all_attempts.append((crit, atlas_mask, f"atlas_{crit}"))
            for crit in ["relaxed", "permissive", "degraded"]:
                all_attempts.append((crit, None, f"brain_{crit}"))

            # Skip attempts with criteria stricter than initial_criteria
            attempts = [
                (crit, roi, name)
                for crit, roi, name in all_attempts
                if criteria_order.index(crit) >= start_idx
            ]

            for crit, roi, level_name in attempts:
                search = self._prepare_search_mask(
                    brain_mask, roi, lesion_mask
                )
                cands, peak_map = self._find_aif_candidate_voxels(
                    conc_data, search, baseline_frames
                )
                if cands.shape[0] == 0:
                    continue
                tissue_ttp = self._calculate_tissue_ttp_reference(
                    conc_data, brain_mask, baseline_frames
                )
                scored, n_avail = self._score_and_select_candidates(
                    cands, conc_data, signal_data, peak_map,
                    brain_mask, tr, baseline_frames, crit,
                    tissue_ttp, n_top=n_candidates,
                )
                if not scored:
                    continue

                # Post-scoring filters (distance + contralateral)
                scored, f_meta = self._apply_post_scoring_filters(
                    scored, lesion_mask, affine, voxel_sizes,
                    lesion_min_distance_mm=lesion_distance,
                    prefer_contralateral=prefer_contralateral,
                )
                if scored:
                    top_candidates = scored
                    n_available = n_avail
                    detection_level = level_name
                    filter_meta = f_meta
                    break

        # ---- Build AIF curve ----
        aif_curve: npt.NDArray[Any]
        aif_mask_arr = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        quality_score = 0.0
        qc_metrics: dict[str, Any] = {}

        if top_candidates:
            aif_curve, aif_bool_mask = self._compute_roi_averaged_aif(
                top_candidates, conc_data, brain_mask
            )
            aif_mask_arr[aif_bool_mask] = 1
            for _, coord, _ in top_candidates:
                aif_mask_arr[coord[0], coord[1], coord[2]] = 1

            # Quality: inverse of mean cost, clamped to [0, 1]
            mean_cost = float(np.mean([c for c, _, _ in top_candidates]))
            quality_score = min(1.0, 1.0 / (1.0 + mean_cost))

            # --- GAP 8: Finalize AIF metrics ---
            aif_curve, finalized_qc = self._finalize_aif_metrics(
                aif_curve, top_candidates, conc_data, brain_mask,
                affine, tr, n_candidates, n_available,
            )
            qc_metrics.update(finalized_qc)
            qc_metrics.update(filter_meta)

        elif not disable_fallback:
            aif_curve = self._generate_synthetic_aif(
                n_timepoints, tr, baseline_frames
            )
            detection_level = "synthetic"
            quality_score = 0.3
            qc_metrics = {
                "synthetic": True,
                "n_aif_candidates_used": 0,
                "n_aif_candidates_available": 0,
            }
        else:
            raise RuntimeError(
                "AIF detection failed at all levels and "
                "synthetic fallback is disabled"
            )

        qc_metrics["detection_level"] = detection_level
        qc_metrics["s0_rejection_available"] = signal_data is not None
        qc_metrics["tr_used"] = tr
        qc_metrics["baseline_frames_used"] = baseline_frames

        # --- GAP 9: Quantification validity flags ---
        if "degraded" in detection_level:
            qc_metrics["quantification_validity"] = {
                "absolute_cbv": False,
                "absolute_cbf": False,
                "relative_cbv": "CAUTION",
                "mtt": "CAUTION",
                "ttp": True,
                "tmax": True,
            }
            qc_metrics["degraded_aif_warnings"] = [
                "AIF quality is DEGRADED - quantification unreliable",
                "Absolute CBV/CBF values are NOT VALID",
                "Relative CBV may have 30-50% error vs optimal AIF",
                "Use TTP/Tmax for timing analysis (unaffected by AIF quality)",
                "Consider manual AIF selection for publication-quality results",
            ]
        elif detection_level != "synthetic":
            is_high_quality = any(
                lvl in detection_level
                for lvl in ("strict", "relaxed")
            )
            qc_metrics["quantification_validity"] = {
                "absolute_cbv": True if is_high_quality else "ACCEPTABLE",
                "absolute_cbf": True if is_high_quality else "ACCEPTABLE",
                "relative_cbv": True,
                "mtt": True,
                "ttp": True,
                "tmax": True,
            }

        # Gamma-variate fitting
        gamma_fit_applied = False
        if gamma_fit and detection_level != "synthetic":
            try:
                from pmriprep.utils.dsc_aif import fit_gamma_variate

                fit_result = fit_gamma_variate(
                    aif_curve=aif_curve,
                    baseline_frames=baseline_frames,
                    remove_recirculation=True,
                    preserve_auc=True,
                    tr=tr,
                )
                if fit_result.fit_converged and fit_result.r_squared > 0.5:
                    aif_curve = fit_result.fitted_curve
                    gamma_fit_applied = True
                    qc_metrics["gamma_fit_r_squared"] = fit_result.r_squared
                    qc_metrics["gamma_fit_auc_ratio"] = fit_result.auc_ratio
                    qc_metrics["gamma_recirculation_removed"] = (
                        fit_result.recirculation_removed
                    )
            except Exception:
                pass

        # Save outputs
        out_aif_curve = self._gen_outfilename("_aif_curve.txt")
        np.savetxt(out_aif_curve, aif_curve)

        out_aif_mask = self._gen_outfilename("_aif_mask.nii.gz")
        mask_img = nib.Nifti1Image(aif_mask_arr, affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_aif_mask)  # type: ignore[attr-defined]

        n_cand = len(top_candidates) if top_candidates else 0
        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "aif_curve": [float(x) for x in aif_curve],
            "detection_level": detection_level,
            "quality_score": quality_score,
            "candidate_count": n_cand,
            "gamma_fit_applied": gamma_fit_applied,
            "qc_metrics": qc_metrics,
        }

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
