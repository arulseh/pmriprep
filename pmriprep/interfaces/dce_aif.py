"""AIF (Arterial Input Function) detection interfaces for DCE-MRI.

This module implements Nipype interfaces for AIF detection in DCE perfusion
processing, with multi-level detection strategies (atlas-based, automatic,
population) and comprehensive quality metrics for DCE-specific requirements.

The DCE AIF differs from DSC AIF in several important ways:
- Concentration is in mM (not relative signal change)
- Expected peak values are in physiological range (5-7.6 mM for full dose)
- FWHM is in seconds (8-25s) rather than frames
- Blood T1 depends on field strength (1.5T→1.35s, 3T→1.65s)
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

from pmriprep.utils.dce_constants import (
    AIF_THRESHOLDS_RELAXED,
    AIF_THRESHOLDS_STRICT,
    DCE_AIF_EXPECTED_FULL_DOSE,
    EPSILON_T1,
    HEMATOCRIT,
    get_aif_thresholds,
    get_blood_t1,
)


# =============================================================================
# Parker Population AIF Parameters
# =============================================================================
# Based on Parker et al. (2006) "Experimentally-derived functional form
# for a population-averaged high-temporal-resolution arterial input
# function for dynamic contrast-enhanced MRI"


class ParkerAIFParams:
    """Parameters for Parker population AIF model.

    The Parker model describes AIF as sum of two Gaussians plus exponential decay:
        C_p(t) = sum_i(A_i * exp(-(t - T_i)^2 / (2*sigma_i^2))) + alpha * exp(-beta*t) / (1 + exp(-s*(t-tau)))

    Where:
        - A_1, A_2: Amplitudes of the two Gaussian components (mM)
        - T_1, T_2: Center times (minutes)
        - sigma_1, sigma_2: Widths of Gaussians (minutes)
        - alpha, beta, s, tau: Parameters for exponential tail

    Reference values are for 0.1 mmol/kg dose of Gd contrast at 3T.
    """

    # First Gaussian (first pass)
    A1: float = 0.809  # mM
    T1: float = 0.17046  # minutes
    sigma1: float = 0.0563  # minutes

    # Second Gaussian (recirculation)
    A2: float = 0.330  # mM
    T2: float = 0.365  # minutes
    sigma2: float = 0.132  # minutes

    # Exponential tail
    alpha: float = 1.050  # mM
    beta: float = 0.1685  # min^-1
    s: float = 38.078  # steepness
    tau: float = 0.483  # minutes


def parker_population_aif(
    t: npt.NDArray[Any],
    dose_factor: float = 1.0,
) -> npt.NDArray[Any]:
    """Generate Parker population AIF.

    Args:
        t: Time array in seconds.
        dose_factor: Scaling factor for dose (1.0 = full dose 0.1 mmol/kg,
            0.5 = half dose, 2.0 = double dose).

    Returns:
        Concentration time course in mM.

    References:
        Parker GJM et al. "Experimentally-derived functional form for a
        population-averaged high-temporal-resolution arterial input function
        for dynamic contrast-enhanced MRI." Magn Reson Med. 2006;56:993-1000.
    """
    # Convert time to minutes
    t_min = t / 60.0

    # Get parameters
    p = ParkerAIFParams()

    # First Gaussian component
    g1 = p.A1 * np.exp(-((t_min - p.T1) ** 2) / (2 * p.sigma1**2))

    # Second Gaussian component
    g2 = p.A2 * np.exp(-((t_min - p.T2) ** 2) / (2 * p.sigma2**2))

    # Exponential tail with sigmoid onset
    # exp_tail = alpha * exp(-beta * t) / (1 + exp(-s * (t - tau)))
    sigmoid = 1.0 / (1.0 + np.exp(-p.s * (t_min - p.tau)))
    exp_tail = p.alpha * np.exp(-p.beta * t_min) * sigmoid

    # Total AIF
    aif = (g1 + g2 + exp_tail) * dose_factor

    # Ensure non-negative
    aif = np.maximum(aif, 0.0)

    return aif


# =============================================================================
# DCE-specific AIF Quality Metrics
# =============================================================================


def compute_dce_aif_peak_concentration(
    aif_curve: npt.NDArray[Any],
) -> float:
    """Compute peak concentration of DCE AIF in mM.

    Args:
        aif_curve: AIF concentration time course in mM.

    Returns:
        Peak concentration in mM.
    """
    return float(np.max(aif_curve))


def compute_dce_aif_fwhm_seconds(
    aif_curve: npt.NDArray[Any],
    tr: float,
    baseline_frames: int = 5,
) -> float:
    """Compute FWHM of DCE AIF peak in seconds.

    Args:
        aif_curve: AIF concentration time course in mM.
        tr: Temporal resolution in seconds.
        baseline_frames: Number of pre-contrast baseline frames.

    Returns:
        FWHM in seconds.
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0.0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    peak = float(np.max(aif_curve))
    peak_idx = int(np.argmax(aif_curve))

    half_max = baseline_mean + (peak - baseline_mean) / 2.0

    # Find first crossing above half max (rising edge)
    rising_idx = baseline_frames
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

    fwhm_frames = falling_idx - rising_idx
    fwhm_seconds = fwhm_frames * tr

    return max(0.0, fwhm_seconds)


def compute_dce_aif_washout(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
    tail_frames: int = 5,
) -> float:
    """Compute washout fraction for DCE AIF.

    Washout fraction = (peak - tail_mean) / (peak - baseline_mean)

    Args:
        aif_curve: AIF concentration time course in mM.
        baseline_frames: Number of pre-contrast baseline frames.
        tail_frames: Number of tail frames to average.

    Returns:
        Washout fraction (0-1).
    """
    if len(aif_curve) < baseline_frames + tail_frames + 1:
        return 0.0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    tail_mean = float(np.mean(aif_curve[-tail_frames:]))
    peak = float(np.max(aif_curve))

    denom = peak - baseline_mean
    if denom < EPSILON_T1:
        return 0.0

    washout = (peak - tail_mean) / denom
    return max(0.0, min(1.0, washout))


def compute_dce_aif_baseline_stability(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute baseline stability as standard deviation for DCE AIF.

    For DCE, baseline should be approximately zero (pre-contrast).

    Args:
        aif_curve: AIF concentration time course in mM.
        baseline_frames: Number of pre-contrast baseline frames.

    Returns:
        Standard deviation of baseline in mM.
    """
    if len(aif_curve) < baseline_frames:
        return float("inf")

    baseline = aif_curve[:baseline_frames]
    return float(np.std(baseline))


def compute_dce_aif_cnr(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute contrast-to-noise ratio for DCE AIF.

    CNR = peak / baseline_std

    For DCE, baseline should be near zero (pre-contrast).

    Args:
        aif_curve: AIF concentration time course in mM.
        baseline_frames: Number of pre-contrast baseline frames.

    Returns:
        Contrast-to-noise ratio.
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0.0

    baseline = aif_curve[:baseline_frames]
    baseline_std = float(np.std(baseline))

    if baseline_std < EPSILON_T1:
        # Perfect baseline, return high CNR
        return 100.0

    peak = float(np.max(aif_curve))
    cnr = peak / baseline_std

    return max(0.0, cnr)


def compute_dce_aif_auc(
    aif_curve: npt.NDArray[Any],
    tr: float,
) -> float:
    """Compute area under the curve for DCE AIF.

    Args:
        aif_curve: AIF concentration time course in mM.
        tr: Temporal resolution in seconds.

    Returns:
        AUC in mM·s.
    """
    # Use trapezoidal integration
    # Try numpy 2.0+ trapezoid, fall back to trapz
    trapezoid_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc = float(trapezoid_fn(aif_curve) * tr)

    return auc


# =============================================================================
# DCE AIF Detection Interface
# =============================================================================


class DCEAIFDetectionInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCE AIFDetection interface."""

    concentration_file = File(
        exists=True,
        mandatory=True,
        desc="4D NIfTI concentration curves in mM (output from signal_to_concentration)",
    )
    brain_mask_file = File(
        exists=True,
        mandatory=True,
        desc="3D brain mask NIfTI",
    )
    atlas_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 3D arterial atlas mask in DCE space",
    )
    lesion_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 3D lesion mask for exclusion",
    )

    # Detection parameters
    aif_method = traits.Enum(
        "auto",
        "population",
        usedefault=True,
        desc="AIF detection method: 'auto' (multi-level) or 'population' (Parker)",
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
    scale_aif = traits.Bool(
        False,
        usedefault=True,
        desc="Scale detected AIF to population reference peak",
    )
    target_peak = traits.Float(
        0.0,
        usedefault=True,
        desc="Target peak concentration for scaling (mM). 0 = use field-strength default",
    )
    baseline_frames = traits.Int(
        5,
        usedefault=True,
        desc="Number of pre-contrast baseline frames",
    )
    field_strength = traits.Float(
        3.0,
        usedefault=True,
        desc="Magnetic field strength in Tesla (for blood T1 lookup)",
    )
    tr = traits.Float(
        mandatory=True,
        desc="Temporal resolution (TR) in seconds",
    )
    dose = traits.Enum(
        "full",
        "half",
        "double",
        usedefault=True,
        desc="Contrast agent dose level (affects expected peak)",
    )

    # Output files
    out_aif_curve = File(
        genfile=True,
        desc="Output AIF curve as 1D text file",
    )
    out_aif_mask = File(
        genfile=True,
        desc="Output AIF voxel mask NIfTI",
    )


class DCEAIFDetectionOutputSpec(TraitedSpec):
    """Output specification for DCE AIFDetection interface."""

    out_aif_curve = File(exists=True, desc="AIF curve as 1D text file")
    out_aif_mask = File(exists=True, desc="AIF voxel mask NIfTI")
    aif_curve = traits.List(traits.Float, desc="AIF concentration curve values (mM)")
    detection_method = traits.Str(
        desc="Detection method used: 'atlas', 'automatic', 'population'"
    )
    quality_score = traits.Float(desc="Overall AIF quality score (0-1)")
    peak_concentration = traits.Float(desc="Peak AIF concentration in mM")
    fwhm_seconds = traits.Float(desc="FWHM of AIF peak in seconds")
    auc = traits.Float(desc="Area under the AIF curve (mM·s)")
    blood_t1 = traits.Float(desc="Blood T1 value used for this field strength (s)")
    scaling_factor = traits.Float(desc="Scaling factor applied to AIF (1.0 if unscaled)")
    candidate_count = traits.Int(desc="Number of candidate voxels used")
    qc_metrics = traits.Dict(desc="Detailed QC metrics dictionary")


class AIFDetection(BaseInterface):
    """Arterial Input Function (AIF) detection for DCE-MRI.

    This interface implements multi-level AIF detection with atlas-based
    search, automatic detection, and Parker population AIF fallback.
    It is specifically designed for DCE-MRI where concentration is in mM.

    The detection follows this hierarchy:
    1. Atlas-based search (strict criteria)
    2. Atlas-based search (relaxed criteria)
    3. Automatic whole-brain search (relaxed criteria)
    4. Parker population AIF as fallback

    DCE-specific considerations:
    - Blood T1 depends on field strength (1.5T→1.35s, 3T→1.65s)
    - Expected peak concentration for full dose: 5.0-7.6 mM
    - FWHM typically 8-25 seconds
    - Population AIF from Parker et al. (2006)

    Parameters
    ----------
    concentration_file : str
        4D NIfTI with concentration time courses in mM.
    brain_mask_file : str
        3D brain mask NIfTI.
    atlas_mask_file : str, optional
        3D arterial atlas mask for constrained search.
    lesion_mask_file : str, optional
        3D lesion mask for excluding potentially abnormal regions.
    aif_method : str
        'auto' for multi-level detection, 'population' for Parker AIF.
    criteria : str
        Initial criteria level: 'strict' or 'relaxed'.
    n_candidates : int
        Number of top candidate voxels to average for final AIF.
    scale_aif : bool
        If True, scale AIF to population reference peak.
    target_peak : float
        Target peak for scaling (0 = use default for field strength).
    baseline_frames : int
        Number of pre-contrast frames for baseline estimation.
    field_strength : float
        Magnetic field strength in Tesla.
    tr : float
        Temporal resolution in seconds.
    dose : str
        Contrast dose level: 'half', 'full', or 'double'.

    Returns
    -------
    out_aif_curve : str
        Path to AIF curve text file.
    out_aif_mask : str
        Path to AIF voxel mask NIfTI.
    aif_curve : list
        AIF concentration values in mM.
    detection_method : str
        Method used: 'atlas', 'automatic', or 'population'.
    quality_score : float
        Overall quality score (0-1).
    peak_concentration : float
        Peak AIF concentration in mM.
    fwhm_seconds : float
        FWHM of AIF peak in seconds.
    auc : float
        Area under the AIF curve (mM·s).
    blood_t1 : float
        Blood T1 value used (seconds).
    scaling_factor : float
        Scaling factor applied to AIF.
    candidate_count : int
        Number of candidate voxels used.
    qc_metrics : dict
        Detailed quality metrics.

    Notes
    -----
    Expected AIF characteristics for full dose (0.1 mmol/kg) at 3T:

    - Peak concentration: 5.0-7.6 mM
    - FWHM: 8-25 seconds
    - Baseline should be ~0 (pre-contrast)
    - Washout fraction: ≥0.2 (strict) or ≥0.1 (relaxed)

    The Parker population AIF (2006) is used as fallback and for scaling
    reference. It represents a well-characterized arterial input for
    standard contrast dose.

    References
    ----------
    Parker GJM et al. "Experimentally-derived functional form for a
    population-averaged high-temporal-resolution arterial input function
    for dynamic contrast-enhanced MRI." Magn Reson Med. 2006;56:993-1000.

    See Also
    --------
    pmriprep.utils.dce_aif : Core DCE AIF utilities (US-061, US-062)
    """

    input_spec = DCEAIFDetectionInputSpec
    output_spec = DCEAIFDetectionOutputSpec

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

    def _get_dose_factor(self) -> float:
        """Get dose factor based on dose setting."""
        dose = str(self.inputs.dose)
        if dose == "half":
            return 0.5
        elif dose == "double":
            return 2.0
        return 1.0

    def _get_expected_peak_range(self) -> tuple[float, float]:
        """Get expected peak concentration range based on dose."""
        dose_factor = self._get_dose_factor()
        base_min = DCE_AIF_EXPECTED_FULL_DOSE.peak_min_mm
        base_max = DCE_AIF_EXPECTED_FULL_DOSE.peak_max_mm

        return (base_min * dose_factor, base_max * dose_factor)

    def _score_dce_candidate(
        self,
        curve: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
        criteria: str,
    ) -> tuple[float, dict[str, float]]:
        """Score a candidate DCE AIF curve.

        Returns a composite quality score and individual metrics.
        Score is 0 if thresholds are not met, otherwise 0-1.
        """
        metrics: dict[str, float] = {}

        # Compute individual metrics
        peak = compute_dce_aif_peak_concentration(curve)
        fwhm = compute_dce_aif_fwhm_seconds(curve, tr, baseline_frames)
        washout = compute_dce_aif_washout(curve, baseline_frames)
        baseline_std = compute_dce_aif_baseline_stability(curve, baseline_frames)
        cnr = compute_dce_aif_cnr(curve, baseline_frames)
        auc = compute_dce_aif_auc(curve, tr)

        metrics["peak_mm"] = peak
        metrics["fwhm_seconds"] = fwhm
        metrics["washout"] = washout
        metrics["baseline_std"] = baseline_std
        metrics["cnr"] = cnr
        metrics["auc"] = auc

        # Get thresholds based on criteria
        thresholds = get_aif_thresholds(criteria)

        # Get expected peak range for dose
        peak_min, peak_max = self._get_expected_peak_range()

        # Check thresholds
        # 1. Washout
        if washout < thresholds.washout_min:
            return 0.0, metrics

        # 2. FWHM in seconds
        if fwhm < thresholds.fwhm_min_seconds or fwhm > thresholds.fwhm_max_seconds:
            return 0.0, metrics

        # 3. Peak concentration (must be physiological)
        # Allow some tolerance for partial volume effects
        if peak < peak_min * 0.3 or peak > peak_max * 3.0:
            return 0.0, metrics

        # 4. CNR (should be reasonably high)
        min_cnr = 5.0 if criteria == "strict" else 3.0
        if cnr < min_cnr:
            return 0.0, metrics

        # Compute composite score (all thresholds met)
        score = 0.0

        # Peak contribution (0-0.3): closer to expected range is better
        expected_peak = (peak_min + peak_max) / 2.0
        peak_deviation = abs(peak - expected_peak) / expected_peak
        peak_score = max(0.0, 1.0 - peak_deviation)
        score += 0.3 * peak_score

        # CNR contribution (0-0.25): higher is better
        cnr_score = min(1.0, cnr / 50.0)  # Normalize to 0-1
        score += 0.25 * cnr_score

        # Washout contribution (0-0.25): closer to 0.5 is ideal
        washout_score = 1.0 - abs(washout - 0.5) / 0.5
        score += 0.25 * max(0.0, washout_score)

        # FWHM contribution (0-0.2): closer to ~15s is ideal for DCE
        ideal_fwhm = 15.0
        fwhm_range = thresholds.fwhm_max_seconds - thresholds.fwhm_min_seconds
        fwhm_score = 1.0 - abs(fwhm - ideal_fwhm) / (fwhm_range / 2)
        score += 0.2 * max(0.0, fwhm_score)

        return min(1.0, score), metrics

    @staticmethod
    def _apply_spatial_exclusions(
        candidate_indices: list[tuple[int, int, int]],
        brain_shape: tuple[int, ...],
    ) -> list[tuple[int, int, int]]:
        """Exclude SSS (midline + superior/posterior) and edge-slice voxels.

        Ported from DSC AIF detector. Excludes voxels that are in the
        sagittal sinus region (midline AND superior/posterior).

        Args:
            candidate_indices: List of (x, y, z) voxel coordinates.
            brain_shape: (nx, ny, nz) spatial dimensions.

        Returns:
            Filtered list of coordinates.
        """
        if not candidate_indices:
            return candidate_indices

        nx, ny, nz = brain_shape[:3]
        x_center = nx // 2
        x_exclude = int(nx * 0.1)  # Central 20% (+/-10%)
        z_superior = int(nz * 0.65)  # Superior 35%
        y_posterior = int(ny * 0.45)  # Posterior 45%

        # SSS exclusion
        kept: list[tuple[int, int, int]] = []
        for coord in candidate_indices:
            x, y, z = coord
            is_midline = (x_center - x_exclude) <= x <= (x_center + x_exclude)
            if is_midline and (z > z_superior or y < y_posterior):
                continue  # SSS region
            kept.append(coord)

        if not kept:
            # Would remove all — keep originals
            kept = list(candidate_indices)

        # Edge slice exclusion (z == 0 or z == nz-1)
        non_edge = [c for c in kept if 0 < c[2] < nz - 1]
        if non_edge:
            kept = non_edge

        return kept

    @staticmethod
    def _validate_peak_timing(
        candidates: list[tuple[int, int, int]],
        concentration_data: npt.NDArray[Any],
        n_timepoints: int,
    ) -> list[tuple[int, int, int]]:
        """Reject candidates peaking after 40% of timepoints.

        Args:
            candidates: Candidate voxel coordinates.
            concentration_data: 4D concentration data.
            n_timepoints: Total number of timepoints.

        Returns:
            Filtered candidates with valid peak timing.
        """
        if not candidates:
            return candidates

        max_peak_idx = int(n_timepoints * 0.4)
        kept: list[tuple[int, int, int]] = []
        for x, y, z in candidates:
            peak_idx = int(np.argmax(concentration_data[x, y, z, :]))
            if peak_idx <= max_peak_idx:
                kept.append((x, y, z))

        if not kept:
            # Would remove all — keep originals
            return candidates
        return kept

    def _find_candidates_in_mask(
        self,
        concentration_data: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        n_candidates: int,
        tr: float,
        baseline_frames: int,
        criteria: str,
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Find candidate AIF voxels within a mask.

        Applies spatial exclusion (SSS/edge), peak timing validation, and
        quality scoring. Returns top candidates sorted by score descending.
        """
        candidates: list[tuple[int, int, int]] = []
        scores: list[float] = []
        all_metrics: list[dict[str, float]] = []

        # Get mask coordinates
        mask_coords = np.where(mask > 0)
        n_voxels = len(mask_coords[0])
        n_timepoints = concentration_data.shape[3]

        for i in range(n_voxels):
            x, y, z = (
                int(mask_coords[0][i]),
                int(mask_coords[1][i]),
                int(mask_coords[2][i]),
            )
            curve = concentration_data[x, y, z, :]

            # Skip if curve is empty or all zeros
            peak = float(np.max(curve))
            if peak < EPSILON_T1:
                continue

            score, metrics = self._score_dce_candidate(
                curve, tr, baseline_frames, criteria
            )
            if score > 0:
                candidates.append((x, y, z))
                scores.append(score)
                all_metrics.append(metrics)

        # Apply spatial exclusions (SSS + edge slices)
        if candidates:
            brain_shape = concentration_data.shape[:3]
            filtered = self._apply_spatial_exclusions(candidates, brain_shape)
            # Keep only filtered candidates with their scores/metrics
            filtered_set = set(filtered)
            keep_idx = [i for i, c in enumerate(candidates) if c in filtered_set]
            candidates = [candidates[i] for i in keep_idx]
            scores = [scores[i] for i in keep_idx]
            all_metrics = [all_metrics[i] for i in keep_idx]

        # Peak timing validation
        if candidates:
            valid_timing = self._validate_peak_timing(
                candidates, concentration_data, n_timepoints
            )
            valid_set = set(valid_timing)
            keep_idx = [i for i, c in enumerate(candidates) if c in valid_set]
            candidates = [candidates[i] for i in keep_idx]
            scores = [scores[i] for i in keep_idx]
            all_metrics = [all_metrics[i] for i in keep_idx]

        # Sort by score descending and take top N
        if candidates:
            sorted_indices = np.argsort(scores)[::-1]
            candidates = [candidates[int(i)] for i in sorted_indices[:n_candidates]]
            scores = [scores[int(i)] for i in sorted_indices[:n_candidates]]
            all_metrics = [all_metrics[int(i)] for i in sorted_indices[:n_candidates]]

        return candidates, scores, all_metrics

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE AIF detection."""
        import nibabel as nib  # type: ignore[import-untyped]

        # Load data
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]

        brain_img = nib.load(self.inputs.brain_mask_file)
        brain_mask: npt.NDArray[Any] = brain_img.get_fdata()  # type: ignore[attr-defined]

        # Optional atlas mask
        atlas_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.atlas_mask_file:
            atlas_img = nib.load(self.inputs.atlas_mask_file)
            atlas_mask = atlas_img.get_fdata()  # type: ignore[attr-defined]

        # Optional lesion mask
        lesion_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.lesion_mask_file:
            lesion_img = nib.load(self.inputs.lesion_mask_file)
            lesion_mask = lesion_img.get_fdata()  # type: ignore[attr-defined]

        # Parameters
        aif_method = str(self.inputs.aif_method)
        criteria = str(self.inputs.criteria)
        n_candidates = int(self.inputs.n_candidates)
        baseline_frames = int(self.inputs.baseline_frames)
        field_strength = float(self.inputs.field_strength)
        tr = float(self.inputs.tr)
        scale_aif = bool(self.inputs.scale_aif)
        target_peak = float(self.inputs.target_peak)

        n_timepoints = conc_data.shape[3]

        # Get blood T1 for this field strength
        blood_t1 = get_blood_t1(field_strength)

        # Initialize results
        detection_method: Literal["atlas", "automatic", "population"] = "population"
        candidates: list[tuple[int, int, int]] = []
        scores: list[float] = []
        metrics_list: list[dict[str, float]] = []

        # Skip auto detection if population method requested
        if aif_method == "population":
            # Go directly to population AIF
            pass
        else:
            # Multi-level AIF detection

            # Create search mask (brain minus lesion)
            search_mask = brain_mask.copy()
            if lesion_mask is not None:
                search_mask[lesion_mask > 0] = 0

            # Level 1: Atlas-based with strict criteria
            if atlas_mask is not None:
                combined_mask = search_mask * atlas_mask
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, combined_mask, n_candidates, tr, baseline_frames, "strict"
                )
                if candidates:
                    detection_method = "atlas"

            # Level 2: Atlas-based with relaxed criteria
            if not candidates and atlas_mask is not None:
                combined_mask = search_mask * atlas_mask
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data,
                    combined_mask,
                    n_candidates,
                    tr,
                    baseline_frames,
                    "relaxed",
                )
                if candidates:
                    detection_method = "atlas"

            # Level 3: Whole-brain with user-specified criteria
            if not candidates:
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, search_mask, n_candidates, tr, baseline_frames, criteria
                )
                if candidates:
                    detection_method = "automatic"

            # Level 4: Whole-brain with relaxed criteria (if not already relaxed)
            if not candidates and criteria == "strict":
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data,
                    search_mask,
                    n_candidates,
                    tr,
                    baseline_frames,
                    "relaxed",
                )
                if candidates:
                    detection_method = "automatic"

        # Generate AIF curve
        aif_curve: npt.NDArray[Any]
        aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        quality_score = 0.0
        scaling_factor = 1.0
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
                    "peak_mm_mean": float(np.mean([m["peak_mm"] for m in metrics_list])),
                    "fwhm_seconds_mean": float(
                        np.mean([m["fwhm_seconds"] for m in metrics_list])
                    ),
                    "washout_mean": float(
                        np.mean([m["washout"] for m in metrics_list])
                    ),
                    "cnr_mean": float(np.mean([m["cnr"] for m in metrics_list])),
                    "auc_mean": float(np.mean([m["auc"] for m in metrics_list])),
                    "n_candidates": len(candidates),
                }

        else:
            # Generate BAT-aligned Parker population AIF
            from pmriprep.utils.dce_bat import detect_bolus_arrival_time

            dose_factor = self._get_dose_factor()
            t = np.arange(n_timepoints) * tr

            # Detect BAT from global mean concentration curve
            mask_bool = brain_mask > 0
            global_curve = np.array([
                float(np.mean(conc_data[..., tidx][mask_bool]))
                for tidx in range(n_timepoints)
            ])
            # time_points in minutes for BAT detection
            t_min = t / 60.0
            bat_result = detect_bolus_arrival_time(
                global_curve, t_min, baseline_frames
            )

            # Time-shift Parker AIF to align with detected BAT
            bat_shift = bat_result.bat_seconds
            t_shifted = t - bat_shift
            t_shifted = np.maximum(t_shifted, 0.0)
            aif_curve = parker_population_aif(t_shifted, dose_factor)
            detection_method = "population"
            quality_score = 0.5  # Moderate score for population AIF

            qc_metrics = {
                "population_aif": True,
                "dose_factor": dose_factor,
                "n_candidates": 0,
                "bat_seconds": bat_result.bat_seconds,
                "bat_method": bat_result.method,
                "bat_confidence": bat_result.confidence,
            }

        # Apply AIF scaling if requested
        if scale_aif and detection_method != "population":
            current_peak = float(np.max(aif_curve))
            if current_peak > EPSILON_T1:
                # Determine target peak
                if target_peak > 0:
                    target = target_peak
                else:
                    # Use field-strength appropriate value
                    peak_min, peak_max = self._get_expected_peak_range()
                    target = (peak_min + peak_max) / 2.0

                scaling_factor = target / current_peak
                aif_curve = aif_curve * scaling_factor

        # Compute final AIF metrics
        final_peak = compute_dce_aif_peak_concentration(aif_curve)
        final_fwhm = compute_dce_aif_fwhm_seconds(aif_curve, tr, baseline_frames)
        final_auc = compute_dce_aif_auc(aif_curve, tr)

        # Add final metrics to QC dict
        qc_metrics.update(
            {
                "final_peak_mm": final_peak,
                "final_fwhm_seconds": final_fwhm,
                "final_auc": final_auc,
                "blood_t1_seconds": blood_t1,
                "field_strength_t": field_strength,
                "scaling_factor": scaling_factor,
                "detection_method": detection_method,
                "quality_score": quality_score,
            }
        )

        # Save outputs
        out_aif_curve = self._gen_outfilename("_dce_aif_curve.txt")
        np.savetxt(out_aif_curve, aif_curve)

        out_aif_mask = self._gen_outfilename("_dce_aif_mask.nii.gz")
        mask_img = nib.Nifti1Image(aif_mask, affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_aif_mask)  # type: ignore[attr-defined]

        # Store results for _list_outputs
        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "aif_curve": [float(x) for x in aif_curve],
            "detection_method": detection_method,
            "quality_score": quality_score,
            "peak_concentration": final_peak,
            "fwhm_seconds": final_fwhm,
            "auc": final_auc,
            "blood_t1": blood_t1,
            "scaling_factor": scaling_factor,
            "candidate_count": len(candidates),
            "qc_metrics": qc_metrics,
        }

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
