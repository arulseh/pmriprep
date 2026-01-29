"""AIF (Arterial Input Function) detection interfaces for DSC-MRI.

This module implements Nipype interfaces for AIF detection in DSC perfusion
processing, with multi-level detection strategies (atlas-based, relaxed
criteria, synthetic fallback) and comprehensive quality metrics.
"""

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)

from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
    QC_CNR_THRESHOLD,
)


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
        # Perfect baseline (zero noise), return high CNR
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

    # Denominator: signal drop from baseline to peak
    denom = peak - baseline_mean
    if denom < EPSILON_CONCENTRATION:
        return 0.0

    # Numerator: recovery from peak to tail
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

    # Find first crossing above half max (rising edge)
    rising_idx = 0
    for i in range(baseline_frames, peak_idx):
        if aif_curve[i] >= half_max:
            rising_idx = i
            break

    # Find first crossing below half max after peak (falling edge)
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
        20.0,
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

    This interface implements multi-level AIF detection with atlas-based
    search, relaxed criteria fallback, and synthetic AIF generation.

    The detection follows this hierarchy:
    1. Atlas-based search (strict criteria)
    2. Atlas-based search (relaxed criteria)
    3. Whole-brain search (relaxed criteria)
    4. Synthetic (population) AIF as fallback

    Parameters
    ----------
    concentration_file : str
        4D NIfTI with concentration time courses (voxels × time).
    brain_mask_file : str
        3D brain mask NIfTI.
    atlas_mask_file : str, optional
        3D arterial atlas mask in DSC space for constrained search.
    lesion_mask_file : str, optional
        3D lesion mask for excluding potentially abnormal regions.
    aif_method : str
        'auto' for multi-level detection, 'population' for synthetic.
    criteria : str
        Initial criteria level: 'strict' or 'relaxed'.
    n_candidates : int
        Number of top candidate voxels to average for final AIF.
    lesion_distance : float
        Minimum distance (mm) from lesion for candidate voxels.
    prefer_contralateral : bool
        Prefer contralateral hemisphere when lesion present.
    gamma_fit : bool
        Apply gamma-variate fitting to isolate first-pass bolus.
    disable_fallback : bool
        Disable synthetic AIF fallback (error if detection fails).
    baseline_frames : int
        Number of pre-bolus frames for baseline estimation.

    Returns
    -------
    out_aif_curve : str
        Path to AIF curve text file.
    out_aif_mask : str
        Path to AIF voxel mask NIfTI.
    aif_curve : list
        AIF concentration values.
    detection_level : str
        Method used: 'atlas', 'relaxed', or 'synthetic'.
    quality_score : float
        Overall quality score (0-1).
    candidate_count : int
        Number of candidate voxels used.
    gamma_fit_applied : bool
        Whether gamma fitting was applied.
    qc_metrics : dict
        Detailed quality metrics.

    Examples
    --------
    >>> aif = AIFDetection()
    >>> aif.inputs.concentration_file = 'concentration_4d.nii.gz'
    >>> aif.inputs.brain_mask_file = 'brain_mask.nii.gz'
    >>> aif.inputs.atlas_mask_file = 'arterial_atlas.nii.gz'
    >>> aif.inputs.n_candidates = 10
    >>> result = aif.run()  # doctest: +SKIP

    Notes
    -----
    AIF quality criteria:

    Strict thresholds:
    - Washout fraction ≥ 0.20
    - FWHM: 5-15 frames
    - CNR ≥ 4.0
    - Baseline CV < 0.10

    Relaxed thresholds:
    - Washout fraction ≥ 0.10
    - FWHM: 3-20 frames
    - CNR ≥ 2.0
    - Baseline CV < 0.20

    The final AIF is computed as the average of the top N candidate
    voxels ranked by a composite quality score.

    See Also
    --------
    pmriprep.utils.dsc_aif : Core AIF detection utilities (US-035, US-036)
    """

    input_spec = AIFDetectionInputSpec
    output_spec = AIFDetectionOutputSpec

    # AIF quality thresholds — 4 levels from strict to degraded.
    # Each level includes the original 4 checks plus 5 new checks:
    #   baseline_return, peak_to_baseline, secondary_peak, monotonicity, s0_fraction
    _STRICT_THRESHOLDS: dict[str, Any] = {
        "washout_min": 0.20,
        "fwhm_min": 5,
        "fwhm_max": 15,
        "cnr_min": 4.0,
        "baseline_cv_max": 0.10,
        "baseline_return_max": 0.30,
        "peak_to_baseline_min": 3.0,
        "secondary_peak_max": 0.70,
        "monotonicity_min": 0.60,
        "s0_fraction_max": 0.40,
    }
    _RELAXED_THRESHOLDS: dict[str, Any] = {
        "washout_min": 0.10,
        "fwhm_min": 3,
        "fwhm_max": 20,
        "cnr_min": 2.0,
        "baseline_cv_max": 0.20,
        "baseline_return_max": 0.40,
        "peak_to_baseline_min": 2.0,
        "secondary_peak_max": 0.85,
        "monotonicity_min": 0.50,
        "s0_fraction_max": 0.60,
    }
    _PERMISSIVE_THRESHOLDS: dict[str, Any] = {
        "washout_min": 0.05,
        "fwhm_min": 2,
        "fwhm_max": 25,
        "cnr_min": 1.5,
        "baseline_cv_max": 0.30,
        "baseline_return_max": 0.50,
        "peak_to_baseline_min": 1.5,
        "secondary_peak_max": 1.00,
        "monotonicity_min": 0.40,
        "s0_fraction_max": 0.80,
    }
    _DEGRADED_THRESHOLDS: dict[str, Any] = {
        "washout_min": 0.02,
        "fwhm_min": 1,
        "fwhm_max": 40,
        "cnr_min": 1.0,
        "baseline_cv_max": 0.40,
        "baseline_return_max": 0.70,
        "peak_to_baseline_min": 1.2,
        "secondary_peak_max": 1.50,
        "monotonicity_min": 0.30,
        "s0_fraction_max": 0.95,
    }

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

    def _get_thresholds(self, criteria: str) -> dict[str, Any]:
        """Get AIF quality thresholds for criteria level."""
        if criteria == "strict":
            return self._STRICT_THRESHOLDS.copy()
        elif criteria == "permissive":
            return self._PERMISSIVE_THRESHOLDS.copy()
        elif criteria == "degraded":
            return self._DEGRADED_THRESHOLDS.copy()
        return self._RELAXED_THRESHOLDS.copy()

    def _score_candidate(
        self,
        curve: npt.NDArray[Any],
        baseline_frames: int,
        thresholds: dict[str, Any],
        s0_value: Optional[float] = None,
        median_s0: Optional[float] = None,
    ) -> tuple[float, dict[str, float]]:
        """Score a candidate AIF curve with 8 quality checks.

        Returns a composite quality score and individual metrics.
        Score is 0 if thresholds are not met, otherwise 0-1.

        Quality checks (matching perf-plus source):
        1. CNR (contrast-to-noise ratio)
        2. Washout fraction
        3. FWHM (full width at half maximum)
        4. Baseline CV (coefficient of variation)
        5. S0-based partial volume rejection (weight 5x — dominant factor)
        6. Baseline return ratio
        7. Peak-to-baseline ratio
        8. Post-peak monotonicity
        """
        metrics: dict[str, float] = {}

        # --- Original 4 metrics ---
        cnr = compute_aif_cnr(curve, baseline_frames)
        washout = compute_aif_washout_fraction(curve, baseline_frames)
        fwhm = compute_aif_fwhm_frames(curve, baseline_frames)
        baseline_cv = compute_baseline_stability(curve, baseline_frames)

        metrics["cnr"] = cnr
        metrics["washout"] = washout
        metrics["fwhm"] = float(fwhm)
        metrics["baseline_cv"] = baseline_cv

        # Check original thresholds
        if cnr < thresholds["cnr_min"]:
            return 0.0, metrics
        if washout < thresholds["washout_min"]:
            return 0.0, metrics
        if fwhm < thresholds["fwhm_min"] or fwhm > thresholds["fwhm_max"]:
            return 0.0, metrics
        if baseline_cv > thresholds["baseline_cv_max"]:
            return 0.0, metrics

        # --- New metric 5: S0-based partial volume rejection ---
        # Physics: blood T2* ~15ms at 3T, tissue T2* ~60ms.
        # Pure blood voxels have LOW S0 (fast T2* decay).
        # s0_fraction = s0_voxel / median_s0_brain
        # Reject if s0_fraction > threshold (too much tissue signal)
        s0_fraction = 0.0
        if (
            s0_value is not None
            and median_s0 is not None
            and median_s0 > EPSILON_CONCENTRATION
        ):
            s0_fraction = s0_value / median_s0
            metrics["s0_fraction"] = s0_fraction
            if "s0_fraction_max" in thresholds and s0_fraction > thresholds["s0_fraction_max"]:
                return 0.0, metrics

        # --- New metric 6: Baseline return ratio ---
        # (tail_mean - baseline_mean) / (peak - baseline_mean)
        # High return ratio means signal didn't recover → not arterial
        n_frames = len(curve)
        tail_frames = min(5, max(1, n_frames // 10))
        baseline_mean = float(np.mean(curve[:baseline_frames]))
        tail_mean = float(np.mean(curve[-tail_frames:]))
        peak = float(np.max(curve))
        peak_minus_baseline = peak - baseline_mean

        baseline_return = 0.0
        if peak_minus_baseline > EPSILON_CONCENTRATION:
            baseline_return = abs(tail_mean - baseline_mean) / peak_minus_baseline
        metrics["baseline_return"] = baseline_return
        if "baseline_return_max" in thresholds and baseline_return > thresholds["baseline_return_max"]:
            return 0.0, metrics

        # --- New metric 7: Peak-to-baseline ratio ---
        # peak_conc / baseline_std — should be high for arterial
        baseline_std = float(np.std(curve[:baseline_frames]))
        peak_to_baseline = 0.0
        if baseline_std > EPSILON_CONCENTRATION:
            peak_to_baseline = peak_minus_baseline / baseline_std
        else:
            peak_to_baseline = 100.0  # Perfect baseline
        metrics["peak_to_baseline"] = peak_to_baseline
        if "peak_to_baseline_min" in thresholds and peak_to_baseline < thresholds["peak_to_baseline_min"]:
            return 0.0, metrics

        # --- New metric 8: Secondary peak detection ---
        # Detect secondary peak after initial washout; reject if too large
        # relative to the primary peak (suggests venous contamination)
        peak_idx = int(np.argmax(curve))
        secondary_peak_ratio = 0.0
        # Look for second peak starting well after the primary
        search_start = min(peak_idx + max(5, n_frames // 10), n_frames - 2)
        if search_start < n_frames - 1:
            post_washout = curve[search_start:]
            if len(post_washout) > 2:
                local_max_idx = int(np.argmax(post_washout))
                local_max_val = float(post_washout[local_max_idx])
                if peak_minus_baseline > EPSILON_CONCENTRATION:
                    secondary_peak_ratio = (local_max_val - baseline_mean) / peak_minus_baseline
        metrics["secondary_peak"] = secondary_peak_ratio
        if "secondary_peak_max" in thresholds and secondary_peak_ratio > thresholds["secondary_peak_max"]:
            return 0.0, metrics

        # --- New metric 9: Post-peak monotonicity ---
        # Fraction of post-peak frames with decreasing values.
        # Arterial curves should show monotonic decay after peak.
        post_peak = curve[peak_idx + 1:]
        if len(post_peak) > 1:
            diffs = np.diff(post_peak)
            monotonicity = float(np.sum(diffs <= 0)) / len(diffs)
        else:
            monotonicity = 1.0
        metrics["monotonicity"] = monotonicity
        if "monotonicity_min" in thresholds and monotonicity < thresholds["monotonicity_min"]:
            return 0.0, metrics

        # --- Composite score (all thresholds met) ---
        # Weight factors match perf-plus source:
        # S0 fraction = 50% (5x weight — dominant factor)
        # CNR = 20% (2x)
        # Washout, FWHM, baseline_cv, new metrics split remainder
        score = 0.0

        # Weights with S0 data:    S0=0.50, CNR=0.20, washout=0.10, FWHM=0.05,
        #                           baseline_cv=0.05, monotonicity=0.05, return=0.05
        # Weights without S0 data: CNR=0.40, washout=0.20, FWHM=0.10,
        #                           baseline_cv=0.10, monotonicity=0.10, return=0.10
        has_s0 = (
            s0_value is not None
            and median_s0 is not None
            and "s0_fraction_max" in thresholds
        )

        if has_s0:
            w_s0, w_cnr, w_wash, w_fwhm, w_cv, w_mono, w_ret = (
                0.50, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05
            )
        else:
            w_s0, w_cnr, w_wash, w_fwhm, w_cv, w_mono, w_ret = (
                0.00, 0.40, 0.20, 0.10, 0.10, 0.10, 0.10
            )

        # S0 contribution: lower s0_fraction is better (more arterial)
        if has_s0:
            max_s0 = thresholds["s0_fraction_max"]
            s0_score = max(0.0, 1.0 - s0_fraction / max_s0) if max_s0 > 0 else 0.5
            score += w_s0 * s0_score

        # CNR contribution
        cnr_score = min(1.0, (cnr - thresholds["cnr_min"]) / 16.0)
        score += w_cnr * cnr_score

        # Washout contribution: closer to 0.5 is ideal
        ideal_washout = 0.5
        washout_score = 1.0 - abs(washout - ideal_washout) / 0.5
        score += w_wash * max(0, washout_score)

        # FWHM contribution
        ideal_fwhm = 9
        fwhm_range = thresholds["fwhm_max"] - thresholds["fwhm_min"]
        fwhm_score = 1.0 - abs(fwhm - ideal_fwhm) / max(1, fwhm_range / 2)
        score += w_fwhm * max(0, fwhm_score)

        # Baseline CV contribution: lower is better
        baseline_score = 1.0 - (baseline_cv / thresholds["baseline_cv_max"])
        score += w_cv * max(0, baseline_score)

        # Monotonicity contribution
        score += w_mono * monotonicity

        # Baseline return contribution: lower is better
        if "baseline_return_max" in thresholds and thresholds["baseline_return_max"] > 0:
            return_score = 1.0 - baseline_return / thresholds["baseline_return_max"]
            score += w_ret * max(0, return_score)

        return min(1.0, score), metrics

    def _find_candidates_in_mask(
        self,
        concentration_data: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        n_candidates: int,
        baseline_frames: int,
        thresholds: dict[str, Any],
        s0_map: Optional[npt.NDArray[Any]] = None,
        median_s0: Optional[float] = None,
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Find candidate AIF voxels within a mask.

        Returns voxel coordinates, scores, and metrics for candidates
        that meet the threshold criteria, sorted by score descending.

        Parameters
        ----------
        s0_map : np.ndarray, optional
            3D map of baseline signal intensity (S0) for partial volume
            rejection. Computed from raw signal baseline frames.
        median_s0 : float, optional
            Median S0 within the brain mask for normalization.
        """
        candidates: list[tuple[int, int, int]] = []
        scores: list[float] = []
        all_metrics: list[dict[str, float]] = []

        # Get mask coordinates
        mask_coords = np.where(mask > 0)
        n_voxels = len(mask_coords[0])

        for i in range(n_voxels):
            x, y, z = (
                int(mask_coords[0][i]),
                int(mask_coords[1][i]),
                int(mask_coords[2][i]),
            )
            curve = concentration_data[x, y, z, :]

            # Skip if curve is empty or all zeros
            if np.max(np.abs(curve)) < EPSILON_CONCENTRATION:
                continue

            # Get S0 value for this voxel if available
            s0_value: Optional[float] = None
            if s0_map is not None:
                s0_value = float(s0_map[x, y, z])

            score, metrics = self._score_candidate(
                curve, baseline_frames, thresholds,
                s0_value=s0_value, median_s0=median_s0,
            )
            if score > 0:
                candidates.append((x, y, z))
                scores.append(score)
                all_metrics.append(metrics)

        # Sort by score descending
        if candidates:
            sorted_indices = np.argsort(scores)[::-1]
            candidates = [candidates[i] for i in sorted_indices[:n_candidates]]
            scores = [scores[i] for i in sorted_indices[:n_candidates]]
            all_metrics = [all_metrics[i] for i in sorted_indices[:n_candidates]]

        return candidates, scores, all_metrics

    def _generate_synthetic_aif(
        self,
        n_timepoints: int,
        tr: float = 1.5,
        baseline_frames: int = 5,
    ) -> npt.NDArray[Any]:
        """Generate synthetic (population) AIF using gamma-variate model.

        Uses population parameters from literature for a typical AIF.
        This provides relative quantification only.

        Parameters
        ----------
        n_timepoints : int
            Number of time points in the curve.
        tr : float
            Repetition time in seconds (default: 1.5s).
        baseline_frames : int
            Number of baseline frames before bolus arrival.

        Returns
        -------
        np.ndarray
            Synthetic AIF concentration curve.
        """
        # Population parameters for gamma-variate AIF
        # Based on typical DSC-MRI values
        t0 = baseline_frames * tr  # Bolus arrival time (seconds)
        alpha = 3.5  # Shape parameter
        beta = 1.5  # Scale parameter (seconds)
        amplitude = 10.0  # Peak concentration (arbitrary units for relative)

        # Time vector
        t = np.arange(n_timepoints) * tr

        # Gamma-variate: C(t) = A * (t-t0)^alpha * exp(-(t-t0)/beta)
        aif = np.zeros(n_timepoints, dtype=np.float64)
        post_arrival = t > t0

        if np.any(post_arrival):
            t_shifted = t[post_arrival] - t0
            # Normalize so max occurs at t = alpha * beta
            t_max = alpha * beta
            norm_factor = (t_max ** alpha) * np.exp(-alpha)
            aif[post_arrival] = (
                amplitude
                * (t_shifted ** alpha)
                * np.exp(-t_shifted / beta)
                / norm_factor
            )

        return aif

    def _run_interface(self, runtime: Any) -> Any:
        """Execute AIF detection with 7-level fallback hierarchy.

        Detection levels (matching perf-plus source):
        1. Atlas-based, strict criteria
        2. Atlas-based, relaxed criteria
        3. Whole-brain, user-specified criteria
        4. Whole-brain, relaxed criteria
        5. Whole-brain, permissive criteria
        6. Whole-brain, degraded criteria
        7. Synthetic (population) AIF fallback
        """
        import nibabel as nib  # type: ignore[import-untyped]

        # Load data
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]

        brain_img = nib.load(self.inputs.brain_mask_file)
        brain_mask: npt.NDArray[Any] = brain_img.get_fdata()  # type: ignore[attr-defined]

        # Verify brain mask is in DSC space (guaranteed by extract_dsc_brain node)
        if brain_mask.shape[:3] != conc_data.shape[:3]:
            raise RuntimeError(
                f"Brain mask shape {brain_mask.shape[:3]} does not match "
                f"concentration data shape {conc_data.shape[:3]}. "
                "The brain mask must be in native DSC space. "
                "Ensure extract_dsc_brain node is connected upstream."
            )

        # Optional atlas mask
        atlas_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.atlas_mask_file:
            atlas_img = nib.load(self.inputs.atlas_mask_file)
            atlas_mask = atlas_img.get_fdata()  # type: ignore[attr-defined]

        # Optional lesion mask — may be in T1w space, resample to DSC if needed.
        # Use affine comparison (not just shape) to detect space mismatch.
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

        # --- Load raw signal for S0-based partial volume rejection ---
        s0_map: Optional[npt.NDArray[Any]] = None
        median_s0: Optional[float] = None
        if self.inputs.signal_file:
            signal_img = nib.load(self.inputs.signal_file)
            signal_data: npt.NDArray[Any] = signal_img.get_fdata()  # type: ignore[attr-defined]
            baseline_frames_s0 = int(self.inputs.baseline_frames)

            # Compute S0 map: mean of baseline (pre-bolus) frames
            # S0 reflects tissue T2* — blood has low T2* (~15ms at 3T)
            # so pure arterial voxels have LOW S0
            if signal_data.shape[3] > baseline_frames_s0:
                s0_map = np.mean(signal_data[:, :, :, :baseline_frames_s0], axis=3)
            else:
                s0_map = np.mean(signal_data, axis=3)

            # Compute median S0 within brain mask for normalization
            brain_voxels = s0_map[brain_mask > 0.5]
            if len(brain_voxels) > 0:
                median_s0 = float(np.median(brain_voxels[brain_voxels > 0]))

        # Parameters
        aif_method = str(self.inputs.aif_method)
        criteria = str(self.inputs.criteria)
        n_candidates = int(self.inputs.n_candidates)
        baseline_frames = int(self.inputs.baseline_frames)
        gamma_fit = bool(self.inputs.gamma_fit)
        disable_fallback = bool(self.inputs.disable_fallback)

        n_timepoints = conc_data.shape[3]
        tr = 1.5  # Default TR, would ideally come from header

        # Initialize results
        detection_level = "synthetic"
        candidates: list[tuple[int, int, int]] = []
        scores: list[float] = []
        metrics_list: list[dict[str, float]] = []

        # Skip auto detection if population method requested
        if aif_method == "population":
            # Go directly to synthetic AIF
            pass
        else:
            # Multi-level AIF detection (7 levels)

            # Create search mask (brain minus lesion with distance buffer)
            search_mask = brain_mask.copy()
            if lesion_mask is not None:
                # Simple exclusion: remove lesion voxels
                # Full distance-based exclusion would be in utils/dsc_aif.py (US-035)
                search_mask[lesion_mask > 0] = 0

            # Level 1: Atlas-based with strict criteria
            if atlas_mask is not None:
                combined_mask = search_mask * atlas_mask
                thresholds = self._get_thresholds("strict")
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, combined_mask, n_candidates, baseline_frames,
                    thresholds, s0_map=s0_map, median_s0=median_s0,
                )
                if candidates:
                    detection_level = "atlas_strict"

            # Level 2: Atlas-based with relaxed criteria
            if not candidates and atlas_mask is not None:
                combined_mask = search_mask * atlas_mask
                thresholds = self._get_thresholds("relaxed")
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, combined_mask, n_candidates, baseline_frames,
                    thresholds, s0_map=s0_map, median_s0=median_s0,
                )
                if candidates:
                    detection_level = "atlas_relaxed"

            # Level 3: Whole-brain with user-specified criteria
            if not candidates:
                thresholds = self._get_thresholds(criteria)
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, search_mask, n_candidates, baseline_frames,
                    thresholds, s0_map=s0_map, median_s0=median_s0,
                )
                if candidates:
                    detection_level = f"brain_{criteria}"

            # Level 4: Whole-brain with relaxed criteria (if not already relaxed)
            if not candidates and criteria == "strict":
                thresholds = self._get_thresholds("relaxed")
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, search_mask, n_candidates, baseline_frames,
                    thresholds, s0_map=s0_map, median_s0=median_s0,
                )
                if candidates:
                    detection_level = "brain_relaxed"

            # Level 5: Whole-brain with permissive criteria
            if not candidates:
                thresholds = self._get_thresholds("permissive")
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, search_mask, n_candidates, baseline_frames,
                    thresholds, s0_map=s0_map, median_s0=median_s0,
                )
                if candidates:
                    detection_level = "brain_permissive"

            # Level 6: Whole-brain with degraded criteria
            if not candidates:
                thresholds = self._get_thresholds("degraded")
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, search_mask, n_candidates, baseline_frames,
                    thresholds, s0_map=s0_map, median_s0=median_s0,
                )
                if candidates:
                    detection_level = "brain_degraded"

        # Generate AIF curve
        aif_curve: npt.NDArray[Any]
        aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        quality_score = 0.0
        qc_metrics: dict[str, Any] = {}

        if candidates:
            # Average curves from candidate voxels
            curves = []
            for x, y, z in candidates:
                curves.append(conc_data[x, y, z, :])
                aif_mask[x, y, z] = 1

            aif_curve = np.mean(curves, axis=0)
            quality_score = float(np.mean(scores))

            # Aggregate QC metrics
            if metrics_list:
                qc_metrics = {
                    "cnr_mean": float(np.mean([m["cnr"] for m in metrics_list])),
                    "washout_mean": float(np.mean([m["washout"] for m in metrics_list])),
                    "fwhm_mean": float(np.mean([m["fwhm"] for m in metrics_list])),
                    "baseline_cv_mean": float(
                        np.mean([m["baseline_cv"] for m in metrics_list])
                    ),
                    "n_candidates": len(candidates),
                }
                # Add new metrics if present
                if "s0_fraction" in metrics_list[0]:
                    qc_metrics["s0_fraction_mean"] = float(
                        np.mean([m.get("s0_fraction", 0.0) for m in metrics_list])
                    )
                if "monotonicity" in metrics_list[0]:
                    qc_metrics["monotonicity_mean"] = float(
                        np.mean([m.get("monotonicity", 0.0) for m in metrics_list])
                    )
                if "baseline_return" in metrics_list[0]:
                    qc_metrics["baseline_return_mean"] = float(
                        np.mean([m.get("baseline_return", 0.0) for m in metrics_list])
                    )

        elif not disable_fallback:
            # Level 7: Generate synthetic AIF as final fallback
            aif_curve = self._generate_synthetic_aif(n_timepoints, tr, baseline_frames)
            detection_level = "synthetic"
            quality_score = 0.3  # Low score for synthetic
            qc_metrics = {
                "synthetic": True,
                "cnr_mean": 0.0,
                "washout_mean": 0.0,
                "fwhm_mean": 0.0,
                "baseline_cv_mean": 0.0,
                "n_candidates": 0,
            }
        else:
            # No fallback, raise error
            raise RuntimeError(
                "AIF detection failed at all 6 levels and synthetic fallback is disabled"
            )

        qc_metrics["detection_level"] = detection_level
        qc_metrics["s0_rejection_available"] = s0_map is not None

        # Gamma-variate fitting (wire existing utils/dsc_aif.py implementation)
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
                    qc_metrics["gamma_recirculation_removed"] = fit_result.recirculation_removed
            except Exception:
                # Gamma fitting failed, use unmodified AIF
                pass

        # Save outputs
        out_aif_curve = self._gen_outfilename("_aif_curve.txt")
        np.savetxt(out_aif_curve, aif_curve)

        out_aif_mask = self._gen_outfilename("_aif_mask.nii.gz")
        mask_img = nib.Nifti1Image(aif_mask, affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_aif_mask)  # type: ignore[attr-defined]

        # Store results for _list_outputs
        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "aif_curve": [float(x) for x in aif_curve],
            "detection_level": detection_level,
            "quality_score": quality_score,
            "candidate_count": len(candidates),
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
