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

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    isdefined,
    traits,
)

from pmriprep.utils.dce_constants import (
    AIF_FWHM_MAX_RELAXED_FRAMES,
    AIF_FWHM_MAX_STRICT_FRAMES,
    AIF_FWHM_NOISE_SCORE,
    AIF_FWHM_SCORING_WIDTH,
    AIF_FWHM_VENOUS_SCORE,
    AIF_FWHM_VENOUS_THRESHOLD_FRAMES,
    AIF_IDEAL_FWHM_FRAMES,
    AIF_FWHM_MAX_RELAXED_SECONDS,
    AIF_IDEAL_FWHM_SECONDS,
    AIF_PEAK_TIMING_FRACTION,
    AIF_THRESHOLDS_RELAXED,
    AIF_THRESHOLDS_STRICT,
    AIF_TIMING_BEFORE_TISSUE_FRAMES,
    AIF_WASHOUT_THRESHOLD_RELAXED,
    AIF_WASHOUT_THRESHOLD_STRICT,
    BRAIN_EROSION_ITERATIONS,
    BRAIN_EROSION_MIN_VOXELS,
    DCE_AIF_EXPECTED_FULL_DOSE,
    EPSILON_CONCENTRATION,
    EPSILON_T1,
    HEMATOCRIT_ARTERIAL,
    LESION_EXCLUSION_DISTANCE_MM,
    MIDLINE_EXCLUSION_FRACTION,
    PARKER_R1_REFERENCE,
    SLAB_EDGE_EXCLUSION_MM,
    SLAB_EDGE_EXCLUSION_MM_ATLAS,
    SUPERIOR_EXCLUSION_FRACTION,
    compute_fwhm_threshold_frames,
    compute_timing_margin_frames,
    get_aif_thresholds,
    get_blood_t1,
    is_low_temporal_resolution,
)

logger = logging.getLogger(__name__)

# Per-candidate quality filtering for detected AIF
# Candidates below this score are partial-volume contaminated (tiers 6-7)
_MIN_CANDIDATE_SCORE = 0.5
# Minimum survivors for a reliable detected AIF (SNR >= 21, basic outlier check)
_MIN_CANDIDATES_FOR_DETECTED = 2

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
    r1_actual: float | None = None,
) -> npt.NDArray[Any]:
    """Generate Parker population AIF.

    Uses normalized Gaussian components where A1/A2 are areas (mM·min) under
    the Gaussian curves. Dividing by (sigma * sqrt(2*pi)) converts these to
    peak amplitudes, producing ~6 mM peak for full dose (0.1 mmol/kg), which
    matches Parker et al. 2006 reported values (6.3 ± 1.5 mM).

    Args:
        t: Time array in seconds.
        dose_factor: Scaling factor for dose (1.0 = full dose 0.1 mmol/kg,
            0.5 = half dose, 2.0 = double dose).
        r1_actual: Actual r1 relaxivity (s^-1 mM^-1) of the contrast agent
            used.  If provided, a relaxivity correction is applied:
            ``effective_factor = dose_factor * (r1_actual / PARKER_R1_REFERENCE)``
            where ``PARKER_R1_REFERENCE = 4.1`` (Gd-DTPA at 1.5T).
            Use this for non-Gd-DTPA agents (e.g. Vueway r1=11.6).

    Returns:
        Concentration time course in mM.

    References:
        Parker GJM et al. "Experimentally-derived functional form for a
        population-averaged high-temporal-resolution arterial input function
        for dynamic contrast-enhanced MRI." Magn Reson Med. 2006;56:993-1000.
    """
    # Convert time to minutes
    t_min = t / 60.0

    # Apply relaxivity correction if non-standard agent
    effective_factor = dose_factor
    if r1_actual is not None and r1_actual > 0:
        effective_factor = dose_factor * (r1_actual / PARKER_R1_REFERENCE)

    # Get parameters
    p = ParkerAIFParams()

    # Normalized Gaussians: A1/A2 are areas (mM·min), not peak amplitudes
    # Dividing by (sigma * sqrt(2*pi)) converts area to peak amplitude in mM
    # This produces ~6 mM peak for full dose, matching Parker et al. reported values
    norm1 = p.sigma1 * np.sqrt(2.0 * np.pi)
    g1 = (p.A1 / norm1) * np.exp(-((t_min - p.T1) ** 2) / (2 * p.sigma1**2))

    norm2 = p.sigma2 * np.sqrt(2.0 * np.pi)
    g2 = (p.A2 / norm2) * np.exp(-((t_min - p.T2) ** 2) / (2 * p.sigma2**2))

    # Exponential tail with sigmoid onset
    # exp_tail = alpha * exp(-beta * t) / (1 + exp(-s * (t - tau)))
    # Use maximum(t_min, 0) to avoid issues with negative time
    sigmoid = 1.0 / (1.0 + np.exp(-p.s * (t_min - p.tau)))
    exp_tail = p.alpha * np.exp(-p.beta * np.maximum(t_min, 0.0)) * sigmoid

    # Total AIF
    aif = (g1 + g2 + exp_tail) * effective_factor

    # Ensure non-negative
    aif = np.maximum(aif, 0.0)

    return np.asarray(aif)


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
    """Compute FWHM of DCE AIF peak in seconds with sub-frame interpolation.

    Uses linear interpolation between frames for sub-frame precision at
    half-maximum crossings.

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

    # Find rising edge with sub-frame interpolation
    rising_time = float(baseline_frames) * tr
    for i in range(baseline_frames, peak_idx):
        if aif_curve[i] >= half_max:
            if i > baseline_frames and aif_curve[i - 1] < half_max:
                denom = aif_curve[i] - aif_curve[i - 1]
                if abs(denom) > EPSILON_T1:
                    frac = (half_max - aif_curve[i - 1]) / denom
                    rising_time = (i - 1 + frac) * tr
                else:
                    rising_time = float(i) * tr
            else:
                rising_time = float(i) * tr
            break

    # Find falling edge with sub-frame interpolation
    falling_time = float(len(aif_curve) - 1) * tr
    for i in range(peak_idx + 1, len(aif_curve)):
        if aif_curve[i] <= half_max:
            if aif_curve[i - 1] > half_max:
                denom = aif_curve[i - 1] - aif_curve[i]
                if abs(denom) > EPSILON_T1:
                    frac = (aif_curve[i - 1] - half_max) / denom
                    falling_time = (i - 1 + frac) * tr
                else:
                    falling_time = float(i) * tr
            else:
                falling_time = float(i) * tr
            break

    fwhm_seconds = falling_time - rising_time
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

    if baseline_std < EPSILON_CONCENTRATION:
        # Near-zero baseline noise, return high CNR
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
    atlas_mask_file = traits.Either(
        traits.Str,
        None,
        desc="Optional 3D arterial atlas mask in DCE space (broadest)",
    )
    atlas_mask_files = traits.Either(
        traits.List(traits.Str),
        None,
        desc=(
            "Optional list of atlas masks in DCE space, ordered tightest to "
            "broadest. Enables tiered atlas cascade: MCA_ACA → CoW → full."
        ),
    )
    aif_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional user-supplied AIF search mask (overrides atlas mask)",
    )
    external_aif_file = File(
        exists=True,
        mandatory=False,
        desc="External AIF curve file (.txt or .npz with 'Cp'/'aif' key). "
             "Values must be plasma concentration in mM, temporally aligned "
             "with DCE volumes. Bypasses all AIF detection.",
    )
    lesion_mask_file = traits.Either(
        File(exists=True),
        None,
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
    dose = traits.Float(
        1.0,
        usedefault=True,
        desc="Contrast agent dose multiplier (0.5=half, 1.0=full, 2.0=double)",
    )
    r1_relaxivity = traits.Float(
        0.0,
        usedefault=True,
        desc=(
            "r1 relaxivity (s^-1 mM^-1) of the contrast agent.  "
            "0 = assume Gd-DTPA (no correction).  Set to actual "
            "relaxivity for non-standard agents (e.g. Vueway r1=11.6)."
        ),
    )
    agent_r1_known = traits.Bool(
        False, usedefault=True,
        desc="True when r1_relaxivity is from a confirmed agent lookup "
             "(not generic field-strength default). Controls Parker AIF correction.",
    )
    agent_standard_dose = traits.Float(
        0.1,
        usedefault=True,
        desc="Standard clinical dose for this agent in mmol/kg "
             "(0.1 for most agents, 0.05 for Vueway/gadopiclenol).",
    )
    hematocrit = traits.Float(
        HEMATOCRIT_ARTERIAL,
        usedefault=True,
        desc="Arterial hematocrit for blood-to-plasma AIF correction",
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
    out_aif_mask = File(desc="AIF voxel mask NIfTI (None for population AIF)")
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

    The detection follows a multi-level cascade:
    1. Atlas-based search (strict → relaxed → permissive)
    2. Automatic whole-brain search (relaxed only)
    3. Parker population AIF as fallback

    Enhanced search includes:
    - Brain mask erosion to exclude cortical surface veins
    - Slab edge exclusion to avoid inflow artifacts
    - Adaptive percentile cascade for candidate pre-filtering
    - Tissue TTP cross-validation to reject venous voxels
    - Upstroke slope metric for arterial/venous discrimination
    - Cluster-based spatial selection (26-connectivity)
    - Inverse-score weighted averaging (single-voxel default, optional ROI expansion)
    - Savitzky-Golay smoothing
    - Contralateral hemisphere preference and distance-based lesion filtering

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
    dose : float
        Contrast dose multiplier (0.5=half, 1.0=full, 2.0=double).
    r1_relaxivity : float
        Contrast agent relaxivity for Parker AIF correction.

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
        """Get dose factor based on dose setting and agent standard dose.

        dose_factor = (agent_standard_dose * multiplier) / 0.1

        Examples:
          - Standard agent, full dose:  (0.1 * 1.0) / 0.1 = 1.0
          - Standard agent, half dose:  (0.1 * 0.5) / 0.1 = 0.5
          - Vueway, half dose:          (0.05 * 0.5) / 0.1 = 0.25
          - Vueway, full dose:          (0.05 * 1.0) / 0.1 = 0.5
        """
        std_dose = float(self.inputs.agent_standard_dose)
        multiplier = float(self.inputs.dose)
        return (std_dose * multiplier) / 0.1

    def _get_expected_peak_range(self) -> tuple[float, float]:
        """Get expected peak concentration range based on dose."""
        dose_factor = self._get_dose_factor()
        base_min = DCE_AIF_EXPECTED_FULL_DOSE.peak_min_mm
        base_max = DCE_AIF_EXPECTED_FULL_DOSE.peak_max_mm

        return (base_min * dose_factor, base_max * dose_factor)

    # -----------------------------------------------------------------
    # Signal Characteristics (Phase 2)
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_signal_characteristics(
        curves: npt.NDArray[Any],
        baseline_frames: int = 5,
        tr: float = 1.0,
    ) -> dict[str, npt.NDArray[Any]]:
        """Compute AIF candidate signal characteristics (vectorized).

        Computes peak values, timing, enhancement, washout, upstroke slope,
        and sub-frame FWHM for each candidate voxel curve.

        Args:
            curves: 2D array (n_voxels, n_timepoints) of concentration curves.
            baseline_frames: Number of pre-contrast baseline frames.
            tr: Temporal resolution in seconds.

        Returns:
            Dictionary with arrays (one value per voxel):
            - peak_values: Peak concentration after baseline.
            - peak_times: Time-to-peak in seconds.
            - peak_indices: Frame index of peak.
            - baseline: Mean baseline concentration.
            - enhancement: Peak minus baseline.
            - washout_fraction: (peak - tail) / (peak - baseline).
            - fwhm_seconds: Sub-frame interpolated FWHM in seconds.
            - upstroke_slope: Maximum dC/dt during wash-in.
        """
        n_voxels, n_timepoints = curves.shape

        # Baseline
        baseline = np.mean(curves[:, :baseline_frames], axis=1)

        # Post-baseline portion
        post = curves[:, baseline_frames:]
        peak_values = np.max(post, axis=1)
        peak_rel_indices = np.argmax(post, axis=1)
        peak_indices = peak_rel_indices + baseline_frames
        peak_times = peak_indices.astype(np.float64) * tr

        # Enhancement
        enhancement = peak_values - baseline

        # Washout fraction: (peak - tail_mean) / (peak - baseline)
        tail_frames = min(5, n_timepoints - baseline_frames - 1)
        if tail_frames > 0:
            tail_mean = np.mean(curves[:, -tail_frames:], axis=1)
        else:
            tail_mean = curves[:, -1]
        denom = peak_values - baseline
        safe_denom = np.where(np.abs(denom) > EPSILON_T1, denom, 1.0)
        washout_fraction = np.clip(
            (peak_values - tail_mean) / safe_denom, 0.0, 1.0
        )
        washout_fraction = np.where(np.abs(denom) > EPSILON_T1, washout_fraction, 0.0)

        # Upstroke slope: max dC/dt during wash-in (baseline to peak)
        upstroke_slope = np.zeros(n_voxels, dtype=np.float64)
        for v in range(n_voxels):
            pi = int(peak_indices[v])
            if pi > baseline_frames:
                segment = curves[v, baseline_frames : pi + 1]
                if len(segment) > 1:
                    diffs = np.diff(segment) / tr
                    upstroke_slope[v] = float(np.max(diffs))

        # Sub-frame FWHM
        fwhm_seconds = np.zeros(n_voxels, dtype=np.float64)
        for v in range(n_voxels):
            fwhm_seconds[v] = compute_dce_aif_fwhm_seconds(
                curves[v, :], tr, baseline_frames
            )

        # FWHM in frames (for arterial criteria cascade)
        fwhm_frames = fwhm_seconds / tr if tr > 0 else fwhm_seconds

        return {
            "peak_values": peak_values,
            "peak_times": peak_times,
            "peak_indices": peak_indices,
            "baseline": baseline,
            "enhancement": enhancement,
            "washout_fraction": washout_fraction,
            "fwhm_seconds": fwhm_seconds,
            "fwhm_frames": fwhm_frames,
            "upstroke_slope": upstroke_slope,
        }

    # -----------------------------------------------------------------
    # Arterial Criteria Cascade
    # -----------------------------------------------------------------

    @staticmethod
    def _apply_arterial_criteria(
        chars: dict[str, npt.NDArray[Any]],
        spatial_exclude: npt.NDArray[Any],
        nt: int,
        percentile_threshold: float = 98.0,
        n_candidates: int = 100,
        tr: float = 2.0,
    ) -> tuple[npt.NDArray[Any], float, int]:
        """Apply arterial criteria with 7-tier fallback cascade.

        TR-adaptive: at low temporal resolution (TR > 5s), timing-based
        criteria are replaced with early_arrival (40% fraction) which is
        TR-independent. FWHM thresholds are adapted to seconds-based limits.

        Args:
            chars: Signal characteristics dictionary from _compute_signal_characteristics.
            spatial_exclude: Boolean array of voxels to exclude (midline, superior, etc).
            nt: Number of timepoints.
            percentile_threshold: Percentile for peak threshold (default 98).
            n_candidates: Target number of candidates.
            tr: Temporal resolution in seconds (for TR-adaptive thresholds).

        Returns:
            (candidate_indices, tissue_ttp, tier_used) where:
            - candidate_indices: Array of indices passing criteria
            - tissue_ttp: Median tissue TTP frame
            - tier_used: Which tier (1-7) produced the result
        """
        n_voxels = len(chars["peak_values"])
        low_tr = is_low_temporal_resolution(tr)
        timing_margin = compute_timing_margin_frames(tr)

        # 1. High peak enhancement (exclude spatially constrained voxels)
        peak_threshold = float(np.percentile(chars["peak_values"], percentile_threshold))
        high_peak = (chars["peak_values"] > peak_threshold) & (~spatial_exclude)

        # 1b. Relaxed peak gate for lower tiers (P95 admits smaller-caliber
        # arterial candidates like pericallosal artery that fall below P98)
        peak_threshold_relaxed = float(np.percentile(chars["peak_values"], 95.0))
        high_peak_relaxed = (chars["peak_values"] > peak_threshold_relaxed) & (~spatial_exclude)

        # 2. Tissue reference timing (median TTP of 25th-75th percentile enhancement voxels)
        enh_values = chars["enhancement"]
        p25 = float(np.percentile(enh_values, 25))
        p75 = float(np.percentile(enh_values, 75))
        moderate_enh = (enh_values > p25) & (enh_values < p75)

        if np.sum(moderate_enh) > 100:
            tissue_ttp = float(np.median(chars["peak_indices"][moderate_enh]))
        else:
            tissue_ttp = float(np.median(chars["peak_indices"]))

        # 3. Arterial timing: AIF should peak BEFORE tissue (by timing margin)
        arterial_timing = chars["peak_indices"] < (tissue_ttp - timing_margin)

        # 3b. Relaxed arterial timing: AIF must peak BEFORE tissue (no margin)
        arterial_timing_relaxed = chars["peak_indices"] < tissue_ttp

        # 4. Early arrival: peak in first 40% of acquisition (TR-independent)
        early_arrival = chars["peak_indices"] < (nt * AIF_PEAK_TIMING_FRACTION)

        # 5. Rapid uptake (90th percentile enhancement)
        enhancement_threshold = float(np.percentile(enh_values, 90))
        rapid_uptake = enh_values > enhancement_threshold

        # 6. Washout criterion (20% strict)
        has_washout = chars["washout_fraction"] > AIF_WASHOUT_THRESHOLD_STRICT

        # 7. FWHM criterion (TR-adaptive: min of frame-based and seconds-based)
        fwhm_strict = compute_fwhm_threshold_frames(tr, strict=True)
        narrow_bolus = chars["fwhm_frames"] < fwhm_strict

        # 8. Upstroke criterion (top 25% of high-peak voxels)
        upstroke_high_peak = chars["upstroke_slope"][high_peak]
        if len(upstroke_high_peak) > 10:
            upstroke_threshold = float(np.percentile(upstroke_high_peak, 75))
            steep_upstroke = chars["upstroke_slope"] > upstroke_threshold
        else:
            steep_upstroke = np.ones(n_voxels, dtype=bool)

        min_candidates = n_candidates // 2

        # At low TR, use early_arrival instead of arterial_timing in Tiers 1-5
        # because the timing margin (1 frame = 8s) exceeds physiological transit time
        timing_gate = early_arrival if low_tr else arterial_timing

        # TIER 1: Strict criteria
        candidates_mask = (
            high_peak & timing_gate & rapid_uptake &
            has_washout & narrow_bolus & steep_upstroke
        )
        candidate_indices = np.where(candidates_mask)[0]
        tier_used = 1

        # TIER 2: Relax timing to early arrival
        if len(candidate_indices) < min_candidates:
            candidates_mask = (
                high_peak & early_arrival & rapid_uptake &
                has_washout & narrow_bolus
            )
            candidate_indices = np.where(candidates_mask)[0]
            tier_used = 2

        # TIER 3: Relax enhancement
        if len(candidate_indices) < min_candidates:
            candidates_mask = (
                high_peak & early_arrival & has_washout & narrow_bolus
            )
            candidate_indices = np.where(candidates_mask)[0]
            tier_used = 3

        # TIER 4: Relax washout to 10%
        if len(candidate_indices) < min_candidates:
            has_washout_relaxed = chars["washout_fraction"] > AIF_WASHOUT_THRESHOLD_RELAXED
            candidates_mask = (
                high_peak & early_arrival & has_washout_relaxed & narrow_bolus
            )
            candidate_indices = np.where(candidates_mask)[0]
            tier_used = 4

        # TIER 5: Relax FWHM + peak gate (P95 admits smaller arterial branches)
        if len(candidate_indices) < min_candidates:
            fwhm_relaxed = compute_fwhm_threshold_frames(tr, strict=False)
            narrow_bolus_relaxed = chars["fwhm_frames"] < fwhm_relaxed
            has_washout_relaxed = chars["washout_fraction"] > AIF_WASHOUT_THRESHOLD_RELAXED
            candidates_mask = (
                high_peak_relaxed & early_arrival & has_washout_relaxed & narrow_bolus_relaxed
            )
            candidate_indices = np.where(candidates_mask)[0]
            tier_used = 5

        # TIER 6: At high TR, drop timing (use peak + FWHM only);
        # at normal TR, use relaxed timing + FWHM
        if len(candidate_indices) < 10:
            if low_tr:
                candidates_mask = high_peak_relaxed & narrow_bolus
            else:
                candidates_mask = high_peak_relaxed & arterial_timing_relaxed & narrow_bolus
            candidate_indices = np.where(candidates_mask)[0]
            tier_used = 6

        # TIER 7: Last resort — at high TR, peak only (always produces candidates);
        # at normal TR, peak + relaxed timing
        if len(candidate_indices) < 10:
            if low_tr:
                candidates_mask = high_peak_relaxed
            else:
                candidates_mask = high_peak_relaxed & arterial_timing_relaxed
            candidate_indices = np.where(candidates_mask)[0]
            tier_used = 7

        return candidate_indices, tissue_ttp, tier_used

    # -----------------------------------------------------------------
    # Tissue TTP Reference (Phase 3)
    # -----------------------------------------------------------------

    @staticmethod
    def _calculate_tissue_ttp_reference(
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        baseline_frames: int,
    ) -> float:
        """Compute median tissue TTP from 25th-75th percentile brain voxels.

        Uses the interquartile range of peak concentrations to select
        parenchymal tissue voxels (avoiding vessels and CSF), then
        returns the median time-to-peak as a reference for cross-validation.

        Args:
            conc_data: 4D concentration data (x, y, z, t).
            brain_mask: 3D brain mask.
            baseline_frames: Number of pre-contrast baseline frames.

        Returns:
            Median tissue TTP as frame index (float).
        """
        # Peak time and concentration maps (post-baseline)
        post_data = conc_data[..., baseline_frames:]
        peak_time_map = np.argmax(post_data, axis=-1) + baseline_frames
        peak_conc_all = np.max(post_data, axis=-1)

        # Get parenchymal voxels (25th-75th percentile of peak concentration)
        brain_peaks = peak_conc_all[brain_mask > 0.5]
        positive_peaks = brain_peaks[brain_peaks > 0]

        if len(positive_peaks) < 10:
            # Not enough voxels; return midpoint as fallback
            return float(conc_data.shape[-1]) * 0.3

        p25 = float(np.percentile(positive_peaks, 25))
        p75 = float(np.percentile(positive_peaks, 75))

        parenchyma = (
            (brain_mask > 0.5)
            & (peak_conc_all > p25)
            & (peak_conc_all < p75)
        )

        if np.sum(parenchyma) < 10:
            return float(conc_data.shape[-1]) * 0.3

        return float(np.median(peak_time_map[parenchyma]))

    # -----------------------------------------------------------------
    # Search Mask Preparation (Phase 1)
    # -----------------------------------------------------------------

    @staticmethod
    def _prepare_search_mask(
        brain_mask: npt.NDArray[Any],
        voxel_sizes: tuple[float, ...],
        lesion_mask: Optional[npt.NDArray[Any]] = None,
        lesion_distance_mm: float = LESION_EXCLUSION_DISTANCE_MM,
        skip_midline: bool = False,
        slab_edge_mm: float = SLAB_EDGE_EXCLUSION_MM,
    ) -> npt.NDArray[Any]:
        """Prepare the search mask with erosion, slab exclusion, and lesion distance.

        Steps:
        1. Erode brain mask (8 iterations) to exclude cortical surface veins.
        2. Exclude inferior/superior slab edges (``slab_edge_mm``) to avoid
           slab profile roll-off artifacts.
        3. Midline exclusion (skipped when ``skip_midline=True`` for
           atlas-constrained search where the atlas already excludes sinuses).
        4. Superior slice exclusion (top 25%).
        5. Lesion exclusion (binary + distance-based).

        Args:
            brain_mask: 3D brain mask.
            voxel_sizes: Voxel dimensions (dx, dy, dz) in mm.
            lesion_mask: Optional 3D lesion mask.
            lesion_distance_mm: Minimum distance (mm) from lesion.
            skip_midline: Skip midline exclusion for atlas-constrained search.
            slab_edge_mm: Distance (mm) from inferior/superior slab edges
                to exclude. Use ``SLAB_EDGE_EXCLUSION_MM_ATLAS`` for
                atlas-constrained search.

        Returns:
            Prepared 3D search mask.
        """
        from scipy.ndimage import binary_erosion

        # Step 1: Erode brain mask
        search_mask = binary_erosion(
            brain_mask > 0.5, iterations=BRAIN_EROSION_ITERATIONS
        ).astype(brain_mask.dtype)
        if np.sum(search_mask) < BRAIN_EROSION_MIN_VOXELS:
            search_mask = (brain_mask > 0.5).astype(brain_mask.dtype)

        # Step 2: Slab edge exclusion (slab_edge_mm from inferior/superior)
        nz = search_mask.shape[2]
        slice_thickness = float(voxel_sizes[2]) if len(voxel_sizes) > 2 else 1.0
        if slice_thickness > 0:
            edge_slices = max(1, int(np.ceil(slab_edge_mm / slice_thickness)))
            if edge_slices * 2 < nz:
                search_mask[:, :, :edge_slices] = 0
                search_mask[:, :, -edge_slices:] = 0

        # Step 3: Midline exclusion — skip for atlas-constrained search
        # (MRA-derived atlas already excludes venous structures; midline
        # exclusion removes 47-68% of atlas voxels including ICA and ACA)
        if not skip_midline:
            AIFDetection._apply_midline_exclusion(search_mask)

        # Step 4: Superior slice exclusion (top 25%) - avoid cortical vein drainage
        z_superior = int(nz * SUPERIOR_EXCLUSION_FRACTION)
        if z_superior < nz:
            search_mask[:, :, z_superior:] = 0

        # Step 5: Lesion exclusion (binary + distance-based)
        if lesion_mask is not None and np.any(lesion_mask > 0):
            # Binary exclusion
            search_mask[lesion_mask > 0] = 0

            # Distance-based exclusion
            if lesion_distance_mm > 0:
                from scipy.ndimage import distance_transform_edt

                lesion_distance_map = distance_transform_edt(
                    lesion_mask < 0.5, sampling=voxel_sizes[:3]
                )
                search_mask[lesion_distance_map < lesion_distance_mm] = 0

        return np.asarray(search_mask)

    @staticmethod
    def _apply_midline_exclusion(mask: npt.NDArray[Any]) -> None:
        """Zero out the central midline strip of a search mask (in-place).

        Used for whole-brain AIF search to exclude sagittal sinus.
        Not needed for atlas-constrained search where the atlas already
        excludes venous structures.
        """
        nx = mask.shape[0]
        x_center = nx // 2
        x_exclude = int(nx * MIDLINE_EXCLUSION_FRACTION)
        x_lo = max(0, x_center - x_exclude)
        x_hi = min(nx, x_center + x_exclude + 1)
        mask[x_lo:x_hi, :, :] = 0

    # -----------------------------------------------------------------
    # Contralateral Hemisphere Preference (Phase 6)
    # -----------------------------------------------------------------

    @staticmethod
    def _determine_tumor_laterality(
        lesion_mask: npt.NDArray[Any],
        affine: npt.NDArray[Any],
    ) -> str | None:
        """Determine tumor laterality from lesion mask center-of-mass.

        Args:
            lesion_mask: 3D lesion mask.
            affine: 4x4 NIfTI affine matrix (voxel→RAS world).

        Returns:
            'left', 'right', 'bilateral', or None if no lesion.
        """
        lesion_coords = np.argwhere(lesion_mask > 0.5)
        if len(lesion_coords) == 0:
            return None

        # Center of mass in voxel space
        com_voxel = np.mean(lesion_coords, axis=0)

        # Convert to RAS world coordinates
        com_world = affine[:3, :3] @ com_voxel + affine[:3, 3]
        x_ras = com_world[0]

        # Check extent: bilateral if lesion spans >10mm past midline on both sides
        all_world = (affine[:3, :3] @ lesion_coords.T).T + affine[:3, 3]
        x_min, x_max = float(np.min(all_world[:, 0])), float(np.max(all_world[:, 0]))

        if x_min < -10.0 and x_max > 10.0:
            return "bilateral"
        elif x_ras < 0:
            return "left"
        else:
            return "right"

    @staticmethod
    def _filter_candidates_by_hemisphere(
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        metrics_list: list[dict[str, float]],
        brain_shape: tuple[int, ...],
        affine: npt.NDArray[Any],
        tumor_laterality: str | None,
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Apply soft score penalty to ipsilateral candidates.

        For non-bilateral tumors, subtracts a score penalty from
        ipsilateral candidates rather than hard-filtering them. This
        preserves bilateral representation while preferring contralateral
        voxels.

        Args:
            candidates: Candidate voxel coordinates.
            scores: Candidate scores (higher is better).
            metrics_list: Candidate metrics.
            brain_shape: Spatial dimensions.
            affine: Affine matrix.
            tumor_laterality: 'left', 'right', 'bilateral', or None.

        Returns:
            All (candidates, scores, metrics) with ipsilateral scores
            penalized. Not sorted (caller handles sorting).
        """
        if not candidates or tumor_laterality is None or tumor_laterality == "bilateral":
            return candidates, scores, metrics_list

        # Soft penalty for ipsilateral candidates
        # DCE scores are 0-1, proportional to DSC 3.0/~15.0
        IPSILATERAL_PENALTY_SCORE = 0.12

        adjusted_scores = list(scores)
        for i, (x, y, z) in enumerate(candidates):
            voxel = np.array([x, y, z], dtype=float)
            world = affine[:3, :3] @ voxel + affine[:3, 3]
            x_ras = world[0]

            is_ipsilateral = (
                (tumor_laterality == "left" and x_ras < 0)
                or (tumor_laterality == "right" and x_ras > 0)
            )
            if is_ipsilateral:
                adjusted_scores[i] = max(0.01, adjusted_scores[i] - IPSILATERAL_PENALTY_SCORE)

        return candidates, adjusted_scores, metrics_list

    @staticmethod
    def _ensure_bilateral_balance(
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        metrics_list: list[dict[str, float]],
        affine: npt.NDArray[Any],
        n_select: int,
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Ensure bilateral AIF representation.

        Guarantees at least ``min_per_hemisphere`` candidates from each
        hemisphere (world-space x=0 as midline). Remaining slots filled
        by score ranking (higher is better).

        ``min_per_hemisphere`` is adaptive: ``max(1, min(2, n_select // 4))``

        Args:
            candidates: Candidate voxel coordinates.
            scores: Candidate scores (higher is better).
            metrics_list: Candidate metrics.
            affine: 4x4 affine matrix (voxel -> RAS world).
            n_select: Number of candidates to select.

        Returns:
            Selected (candidates, scores, metrics) with bilateral guarantee,
            sorted by score descending.
        """
        if not candidates or len(candidates) <= n_select:
            return candidates, scores, metrics_list

        min_per_hemisphere = max(1, min(2, n_select // 4))

        # Sort by score descending first
        sorted_idx = sorted(
            range(len(candidates)), key=lambda i: scores[i], reverse=True
        )

        # Split by world-space x (RAS: x<0 = left)
        left_idx: list[int] = []
        right_idx: list[int] = []
        for i in sorted_idx:
            x, y, z = candidates[i]
            voxel = np.array([x, y, z], dtype=float)
            world_x = float(affine[0, :3] @ voxel + affine[0, 3])
            (left_idx if world_x < 0 else right_idx).append(i)

        # Graceful fallthrough: if one hemisphere is empty, just return top N
        if not left_idx or not right_idx:
            top = sorted_idx[:n_select]
            return (
                [candidates[i] for i in top],
                [scores[i] for i in top],
                [metrics_list[i] for i in top],
            )

        # Reserve minimum from each, fill rest by score
        min_l = min(min_per_hemisphere, len(left_idx))
        min_r = min(min_per_hemisphere, len(right_idx))
        selected = set(left_idx[:min_l] + right_idx[:min_r])

        # Fill remaining slots from sorted order
        for i in sorted_idx:
            if len(selected) >= n_select:
                break
            selected.add(i)

        # Sort final selection by score descending
        final = sorted(selected, key=lambda i: scores[i], reverse=True)
        return (
            [candidates[i] for i in final],
            [scores[i] for i in final],
            [metrics_list[i] for i in final],
        )

    @staticmethod
    def _filter_venous_candidates(
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        metrics_list: list[dict[str, float]],
        min_non_venous: int = 3,
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Filter out candidates flagged as likely venous.

        This method enforces the venous exclusion that is flagged but not
        enforced elsewhere in the pipeline. It should be called unconditionally
        after candidate selection to remove venous contamination.

        Args:
            candidates: Candidate voxel coordinates.
            scores: Candidate scores.
            metrics_list: Candidate metrics (must contain is_likely_venous flag).
            min_non_venous: Minimum non-venous candidates required to apply filter.

        Returns:
            Filtered (candidates, scores, metrics) with venous candidates removed.
            If filtering would leave fewer than min_non_venous candidates,
            returns original input unchanged (with warning logged).
        """
        if not candidates:
            return candidates, scores, metrics_list

        # Identify non-venous candidates
        non_venous_indices = [
            i for i, m in enumerate(metrics_list)
            if not m.get("is_likely_venous", False)
        ]

        n_venous = len(candidates) - len(non_venous_indices)

        if n_venous == 0:
            # No venous candidates, return unchanged
            return candidates, scores, metrics_list

        if len(non_venous_indices) >= min_non_venous:
            # Enough non-venous candidates, filter out venous
            logger.info(
                "Filtering %d venous candidates (keeping %d arterial)",
                n_venous, len(non_venous_indices),
            )
            return (
                [candidates[i] for i in non_venous_indices],
                [scores[i] for i in non_venous_indices],
                [metrics_list[i] for i in non_venous_indices],
            )

        # Not enough non-venous candidates — keep all candidates rather than
        # returning empty. A detected AIF with marginal timing is better than
        # population AIF which introduces 25-40% patient-specific variability
        # (Parker 2006). The venous flag is preserved in metrics for QC.
        logger.warning(
            "VENOUS CONTAMINATION WARNING: %d of %d candidates flagged as likely venous. "
            "Insufficient non-venous candidates (%d < %d). "
            "Keeping all candidates (detected AIF preferred over population fallback).",
            n_venous, len(candidates), len(non_venous_indices), min_non_venous,
        )
        return candidates, scores, metrics_list

    # -----------------------------------------------------------------
    # Candidate Scoring (replaces old _score_dce_candidate)
    # -----------------------------------------------------------------

    def _score_dce_candidate(
        self,
        curve: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
        criteria: str,
        tissue_ttp_frame: float | None = None,
        atlas_constrained: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """Score a candidate DCE AIF curve.

        Returns a composite quality score and individual metrics.
        Score is 0 if thresholds are not met, otherwise 0-1.

        Args:
            curve: Candidate concentration time course.
            tr: Temporal resolution in seconds.
            baseline_frames: Number of pre-contrast frames.
            criteria: Quality criteria level.
            tissue_ttp_frame: Tissue TTP for timing validation (optional).

        Returns:
            (score, metrics) tuple.
        """
        metrics: dict[str, float] = {}

        # Compute individual metrics
        peak = compute_dce_aif_peak_concentration(curve)
        fwhm = compute_dce_aif_fwhm_seconds(curve, tr, baseline_frames)
        washout = compute_dce_aif_washout(curve, baseline_frames)
        baseline_std = compute_dce_aif_baseline_stability(curve, baseline_frames)
        cnr = compute_dce_aif_cnr(curve, baseline_frames)
        auc = compute_dce_aif_auc(curve, tr)

        # Upstroke slope
        peak_idx = int(np.argmax(curve))
        upstroke = 0.0
        if peak_idx > baseline_frames:
            segment = curve[baseline_frames : peak_idx + 1]
            if len(segment) > 1:
                upstroke = float(np.max(np.diff(segment) / tr))

        # Peak time in frames
        peak_time_frame = float(peak_idx)

        metrics["peak_mm"] = peak
        metrics["fwhm_seconds"] = fwhm
        metrics["washout"] = washout
        metrics["baseline_std"] = baseline_std
        metrics["cnr"] = cnr
        metrics["auc"] = auc
        metrics["upstroke_slope"] = upstroke
        metrics["peak_time_frame"] = peak_time_frame

        # Get thresholds based on criteria (TR-adaptive)
        thresholds = get_aif_thresholds(criteria, tr=tr)

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
        # Whole-brain: stricter peak gate (no spatial prior to compensate).
        # 0.5 × 5.0 mM = 2.5 mM floor at full dose. At TR=4.76s, a true
        # artery sampled at ~60% of peak ≈ 3.6 mM still passes.
        # Atlas-constrained: uniform 0.15 — low peaks expected from temporal
        # undersampling + partial volume in known arterial territory.
        # Atlas spatial prior compensates; quarter-dose at long TR produces
        # peaks ~0.2-0.3 mM that need the lower floor.
        if atlas_constrained:
            peak_tol = 0.15
        else:
            peak_tol = 0.5 if criteria in ("strict", "relaxed") else 0.3
        peak_ceiling = 3.0 if criteria in ("strict", "relaxed") else 5.0
        if peak < peak_min * peak_tol or peak > peak_max * peak_ceiling:
            return 0.0, metrics

        # 4. CNR
        cnr_map = {"strict": 5.0, "relaxed": 3.0, "permissive": 2.0, "degraded": 1.0}
        min_cnr = cnr_map.get(criteria, 3.0)
        if cnr < min_cnr:
            return 0.0, metrics

        # Compute composite score (all thresholds met)
        # At low TR, timing weight reduced from 0.15 to 0.05; redistribute
        # 0.05 each to FWHM and washout for better discrimination
        low_tr = is_low_temporal_resolution(tr)
        w_fwhm = 0.25 if low_tr else 0.15
        w_washout = 0.25 if low_tr else 0.20
        score = 0.0

        # Peak contribution (0-0.20): closer to expected range is better
        expected_peak = (peak_min + peak_max) / 2.0
        peak_deviation = abs(peak - expected_peak) / max(expected_peak, EPSILON_T1)
        peak_score = max(0.0, 1.0 - peak_deviation)
        score += 0.20 * peak_score

        # CNR contribution (0-0.15): higher is better
        cnr_score = min(1.0, cnr / 50.0)
        score += 0.15 * cnr_score

        # Washout contribution: closer to 0.5 is ideal
        washout_score = 1.0 - abs(washout - 0.5) / 0.5
        score += w_washout * max(0.0, washout_score)

        # FWHM contribution: closer to ~15s is ideal for DCE
        ideal_fwhm = 15.0
        fwhm_range = thresholds.fwhm_max_seconds - thresholds.fwhm_min_seconds
        fwhm_score = 1.0 - abs(fwhm - ideal_fwhm) / max(fwhm_range / 2, 1.0)
        score += w_fwhm * max(0.0, fwhm_score)

        # Upstroke slope contribution (0-0.15): steeper is better for arteries
        if upstroke > 0:
            upstroke_score = min(1.0, upstroke / (peak / max(tr * 3, EPSILON_T1)))
            score += 0.15 * upstroke_score

        # Timing contribution: AIF should peak before tissue
        # At high TR, timing is less discriminative so reduce weight
        if tissue_ttp_frame is not None:
            timing_margin = compute_timing_margin_frames(tr)
            frames_before_tissue = tissue_ttp_frame - peak_time_frame
            low_tr = is_low_temporal_resolution(tr)

            if low_tr:
                # At high TR: full credit if same frame or before tissue
                if frames_before_tissue >= 0:
                    timing_score = 1.0
                else:
                    timing_score = 0.0
                # Reduced timing weight (0.05), redistribute to FWHM and washout
                timing_weight = 0.05
            else:
                if frames_before_tissue >= timing_margin:
                    timing_score = 1.0
                elif frames_before_tissue >= 0:
                    timing_score = frames_before_tissue / timing_margin
                else:
                    timing_score = 0.0
                timing_weight = 0.15

            score += timing_weight * timing_score
            metrics["timing_score"] = timing_score

        return min(1.0, score), metrics

    # -----------------------------------------------------------------
    # Rescue Scoring
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_rescue_score(
        candidate_peak: float,
        peak_min_expected: float,
        peak_floor: float,
        tier_used: int,
        is_likely_venous: bool,
        atlas_constrained: bool,
    ) -> float:
        """Compute rescue score for a candidate that scored 0 from the main scorer.

        When the arterial cascade accepts a candidate but the scorer rejects it
        (e.g., due to FWHM/CNR), this determines whether to rescue it with a
        tier-based fallback score, scaled by peak adequacy for whole-brain mode.

        Atlas-constrained detection skips peak adequacy scaling because low
        measured peaks (0.8-1.3 mM at full dose) are expected from temporal
        undersampling + partial volume in known arterial territory.

        Args:
            candidate_peak: Candidate's peak concentration in mM.
            peak_min_expected: Expected minimum arterial peak for this dose.
            peak_floor: Minimum peak to allow rescue (criteria/mode-dependent).
            tier_used: Arterial cascade tier (1=strict, 7=last resort).
            is_likely_venous: Whether timing-based venous flag was set.
            atlas_constrained: Whether detection used atlas spatial constraint.

        Returns:
            Rescue score (0.0 if rejected, else tier-based score x peak adequacy).
        """
        if candidate_peak < peak_floor:
            return 0.0
        base_score = max(0.1, round(1.0 - (tier_used - 1) * 0.1, 2))
        # For whole-brain detection, scale by peak adequacy to penalize
        # sub-arterial peaks (venous contamination risk without spatial prior).
        # Atlas-constrained detection skips this — low peaks are expected
        # from temporal undersampling + partial volume in known arterial territory.
        if atlas_constrained:
            peak_adequacy = 1.0
        else:
            # Guard against zero peak_min_expected (e.g., missing dose metadata);
            # min(1.0, ...) caps adequacy so the guard only prevents division by zero.
            peak_adequacy = min(1.0, candidate_peak / max(peak_min_expected, 1e-6))
        if is_likely_venous:
            return base_score * 0.3 * peak_adequacy
        return base_score * peak_adequacy

    # -----------------------------------------------------------------
    # Spatial Exclusions (unchanged from original)
    # -----------------------------------------------------------------

    @staticmethod
    def _apply_spatial_exclusions(
        candidate_indices: list[tuple[int, int, int]],
        brain_shape: tuple[int, ...],
    ) -> list[tuple[int, int, int]]:
        """Exclude edge-slice voxels (z == 0 or z == nz-1).

        Note: Midline and superior exclusions are now handled in
        _prepare_search_mask() for earlier, more comprehensive filtering.
        This method now only handles edge slice exclusion as a final check.

        Args:
            candidate_indices: List of (x, y, z) voxel coordinates.
            brain_shape: (nx, ny, nz) spatial dimensions.

        Returns:
            Filtered list of coordinates.
        """
        if not candidate_indices:
            return candidate_indices

        nz = brain_shape[2]

        # Edge slice exclusion (z == 0 or z == nz-1)
        non_edge = [c for c in candidate_indices if 0 < c[2] < nz - 1]

        # Return filtered list, or original if filtering would remove all
        return non_edge if non_edge else list(candidate_indices)

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

    # -----------------------------------------------------------------
    # Cluster-Based Selection (Phase 4)
    # -----------------------------------------------------------------

    @staticmethod
    def _score_and_select_cluster(
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        metrics_list: list[dict[str, float]],
        brain_shape: tuple[int, ...],
        tr: float = 2.0,
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Score spatial clusters and select the best arterial cluster.

        Uses 26-connectivity labeling and composite scoring:
        0.3 * FWHM_score + 0.3 * washout_score + 0.15 * upstroke_score + 0.25 * size_score

        Args:
            candidates: Candidate voxel coordinates.
            scores: Per-candidate quality scores.
            metrics_list: Per-candidate metric dictionaries.
            brain_shape: Spatial dimensions (nx, ny, nz).
            tr: Temporal resolution in seconds (for FWHM frame conversion).

        Returns:
            (candidates, scores, metrics) for the best cluster.
            If only 1 cluster exists or clustering fails, returns input unchanged.
        """
        if len(candidates) < 3:
            return candidates, scores, metrics_list

        from scipy.ndimage import label

        # Build binary volume of candidate voxels
        vol = np.zeros(brain_shape[:3], dtype=np.int32)
        for x, y, z in candidates:
            vol[x, y, z] = 1

        # 26-connectivity structuring element
        struct = np.ones((3, 3, 3), dtype=np.int32)
        labeled, n_labels = label(vol, structure=struct)

        if n_labels <= 1:
            return candidates, scores, metrics_list

        # Map each candidate to its cluster label
        candidate_labels = [labeled[x, y, z] for x, y, z in candidates]

        # Compute cluster scores
        best_cluster_label = 0
        best_cluster_score = -1.0

        for cluster_id in range(1, n_labels + 1):
            cluster_indices = [
                i for i, lbl in enumerate(candidate_labels) if lbl == cluster_id
            ]
            if not cluster_indices:
                continue

            cluster_size = len(cluster_indices)

            # FWHM score: Gaussian centered on ideal arterial FWHM
            # Convert fwhm_seconds to frames using actual TR
            fwhm_vals = [
                metrics_list[i].get("fwhm_seconds", 30.0) for i in cluster_indices
            ]
            mean_fwhm = float(np.mean(fwhm_vals))
            effective_tr = max(tr, 0.1)  # Guard against zero TR
            mean_fwhm_frames = mean_fwhm / effective_tr

            # TR-adaptive ideal FWHM and venous threshold in frames
            ideal_fwhm_frames = AIF_IDEAL_FWHM_SECONDS / effective_tr
            venous_threshold_frames = AIF_FWHM_MAX_RELAXED_SECONDS / effective_tr

            if mean_fwhm_frames < 0.5:
                # Noise spike - too narrow
                fwhm_score = AIF_FWHM_NOISE_SCORE
            elif mean_fwhm_frames > venous_threshold_frames:
                # Venous - too broad
                fwhm_score = AIF_FWHM_VENOUS_SCORE
            else:
                # Gaussian scoring centered on ideal FWHM
                deviation = (
                    (mean_fwhm_frames - ideal_fwhm_frames)
                    / AIF_FWHM_SCORING_WIDTH
                )
                fwhm_score = float(np.exp(-deviation**2))

            # Washout score: prefer high washout (arterial)
            washout_vals = [
                metrics_list[i].get("washout", 0.0) for i in cluster_indices
            ]
            washout_score = float(np.mean(washout_vals))

            # Upstroke score: prefer steep upstroke (arterial)
            upstroke_vals = [
                metrics_list[i].get("upstroke_slope", 0.0) for i in cluster_indices
            ]
            max_upstroke = max(upstroke_vals) if upstroke_vals else 0.0
            upstroke_ref = max_upstroke if max_upstroke > 0 else 1.0
            upstroke_score = float(np.mean(upstroke_vals)) / upstroke_ref

            # Size score: prefer larger clusters (more spatial coherence)
            max_cluster = max(
                sum(1 for lbl in candidate_labels if lbl == cid)
                for cid in range(1, n_labels + 1)
            )
            size_score = cluster_size / max(max_cluster, 1)

            composite = (
                0.30 * fwhm_score
                + 0.30 * washout_score
                + 0.15 * upstroke_score
                + 0.25 * size_score
            )

            if composite > best_cluster_score:
                best_cluster_score = composite
                best_cluster_label = cluster_id

        # Extract best cluster
        keep_idx = [
            i for i, lbl in enumerate(candidate_labels) if lbl == best_cluster_label
        ]

        if not keep_idx:
            return candidates, scores, metrics_list

        return (
            [candidates[i] for i in keep_idx],
            [scores[i] for i in keep_idx],
            [metrics_list[i] for i in keep_idx],
        )

    # -----------------------------------------------------------------
    # Vectorized Candidate Search with Arterial Cascade
    # -----------------------------------------------------------------

    def _find_candidates_in_mask(
        self,
        concentration_data: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        n_candidates: int,
        tr: float,
        baseline_frames: int,
        criteria: str,
        tissue_ttp_frame: float | None = None,
        atlas_constrained: bool = False,
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Find candidate AIF voxels using 7-tier arterial cascade.

        Uses vectorized boolean operations with 7-tier fallback cascade:
        1. Strict: high_peak & arterial_timing & rapid_uptake & has_washout & narrow_bolus & steep_upstroke
        2. Relax timing to early_arrival
        3. Relax enhancement (drop rapid_uptake)
        4. Relax washout to 10%
        5. Relax FWHM to 8 frames
        6. Peak + FWHM only
        7. Peak only (last resort)

        Args:
            concentration_data: 4D concentration data.
            mask: 3D search mask.
            n_candidates: Number of candidates to return.
            tr: Temporal resolution in seconds.
            baseline_frames: Number of pre-contrast frames.
            criteria: Quality criteria level (used for final scoring only).
            tissue_ttp_frame: Ignored (computed internally from data).

        Returns:
            (candidates, scores, metrics_list) — top N voxels.
        """
        brain_shape = concentration_data.shape[:3]
        n_timepoints = concentration_data.shape[3]

        # Get all masked voxel coordinates
        mask_coords = np.argwhere(mask > 0.5)
        if mask_coords.shape[0] == 0:
            return [], [], []

        # Extract curves for all masked voxels (vectorized)
        curves = np.array([
            concentration_data[x, y, z, :]
            for x, y, z in mask_coords
        ])

        # Cap at physiological maximum to reject resampling artifacts
        # (max arterial peak ~15 mM for double dose; 20 mM provides margin)
        MAX_PHYSIOLOGICAL_CONC_MM = 20.0
        curves = np.minimum(curves, MAX_PHYSIOLOGICAL_CONC_MM)

        n_voxels = curves.shape[0]

        if n_voxels == 0:
            return [], [], []

        # Compute signal characteristics for ALL masked voxels (vectorized)
        chars = self._compute_signal_characteristics(curves, baseline_frames, tr)

        # Build spatial exclusion array matching mask_coords order
        # Edge slices only (midline/superior already in mask)
        nz = brain_shape[2]
        spatial_exclude = np.array([
            (z == 0 or z == nz - 1) for x, y, z in mask_coords
        ], dtype=bool)

        # Apply 7-tier arterial criteria cascade (TR-adaptive)
        candidate_idx_in_masked, tissue_ttp_computed, tier_used = self._apply_arterial_criteria(
            chars, spatial_exclude, n_timepoints,
            percentile_threshold=98.0, n_candidates=n_candidates * 2,
            tr=tr,
        )

        logger.debug(
            "Arterial cascade: tier %d, %d candidates (of %d masked voxels)",
            tier_used, len(candidate_idx_in_masked), n_voxels,
        )

        if len(candidate_idx_in_masked) == 0:
            return [], [], []

        # Convert indices back to voxel coordinates
        candidates: list[tuple[int, int, int]] = []
        all_metrics: list[dict[str, float]] = []

        for idx in candidate_idx_in_masked:
            x, y, z = int(mask_coords[idx, 0]), int(mask_coords[idx, 1]), int(mask_coords[idx, 2])
            candidates.append((x, y, z))

            # Build metrics dict from vectorized characteristics
            metrics: dict[str, float] = {
                "peak_mm": float(chars["peak_values"][idx]),
                "fwhm_seconds": float(chars["fwhm_seconds"][idx]),
                "fwhm_frames": float(chars["fwhm_frames"][idx]),
                "washout": float(chars["washout_fraction"][idx]),
                "baseline_std": 0.0,  # Not computed in vectorized path
                "cnr": 0.0,  # Will compute below if needed
                "auc": 0.0,  # Will compute below if needed
                "upstroke_slope": float(chars["upstroke_slope"][idx]),
                "peak_time_frame": float(chars["peak_indices"][idx]),
                "enhancement": float(chars["enhancement"][idx]),
                "tier_used": float(tier_used),
            }

            # Compute CNR for QC
            curve = curves[idx]
            if baseline_frames > 0:
                baseline_std = float(np.std(curve[:baseline_frames]))
                if baseline_std > EPSILON_CONCENTRATION:
                    metrics["cnr"] = metrics["peak_mm"] / baseline_std
                    metrics["baseline_std"] = baseline_std
                else:
                    metrics["cnr"] = 100.0

            # Compute AUC
            trapezoid_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            metrics["auc"] = float(trapezoid_fn(curve) * tr)

            # Check timing against tissue TTP (TR-adaptive)
            frames_before_tissue = tissue_ttp_computed - metrics["peak_time_frame"]
            metrics["frames_before_tissue"] = frames_before_tissue
            timing_margin = compute_timing_margin_frames(tr)
            if is_low_temporal_resolution(tr):
                # At high TR, only flag venous if peak is >1 frame AFTER tissue
                if frames_before_tissue < -1:
                    metrics["is_likely_venous"] = True
            else:
                if frames_before_tissue < timing_margin:
                    metrics["is_likely_venous"] = True

            all_metrics.append(metrics)

        # Score candidates for ranking (using individual _score_dce_candidate for ranking)
        # Pass tissue TTP so timing is factored into scoring
        # Precompute physiological peak floor (loop-invariant)
        peak_min_expected, _ = self._get_expected_peak_range()
        # Whole-brain: higher floor (match scorer's peak_tol) to reject venous
        # candidates that lack a spatial prior.
        # Atlas-constrained: keep permissive floor — low peaks (0.8-1.3 mM at
        # full dose) are expected from temporal undersampling + partial volume
        # in known arterial territory (ICA/MCA).
        if atlas_constrained:
            peak_floor = peak_min_expected * 0.15  # 0.75 mM at full dose
        else:
            peak_tol_for_rescue = 0.5 if criteria in ("strict", "relaxed") else 0.3
            peak_floor = peak_min_expected * peak_tol_for_rescue  # 2.5 mM at full dose strict

        scores: list[float] = []
        for idx, (x, y, z) in enumerate(candidates):
            curve = concentration_data[x, y, z, :]
            score, scored_metrics = self._score_dce_candidate(
                curve, tr, baseline_frames, criteria,
                tissue_ttp_frame=tissue_ttp_computed,
                atlas_constrained=atlas_constrained,
            )
            # Update metrics with timing score if computed
            if "timing_score" in scored_metrics:
                all_metrics[idx]["timing_score"] = scored_metrics["timing_score"]
            # If score is 0, check whether peak is physiologically plausible
            # before applying tier-based fallback
            if score <= 0:
                candidate_peak = all_metrics[idx].get("peak_mm", 0.0)
                score = self._compute_rescue_score(
                    candidate_peak, peak_min_expected, peak_floor,
                    tier_used, all_metrics[idx].get("is_likely_venous", False),
                    atlas_constrained,
                )
                if score <= 0:
                    logger.debug(
                        "Rescue rejected voxel %d: peak %.2f mM < floor %.2f mM "
                        "(criteria=%s, atlas=%s)",
                        idx, candidate_peak, peak_floor, criteria, atlas_constrained,
                    )
            scores.append(score)

        # ----- Peak timing validation (40% cutoff) -----
        if candidates:
            valid_timing = self._validate_peak_timing(
                candidates, concentration_data, n_timepoints
            )
            valid_set = set(valid_timing)
            keep_idx_pt = [i for i, c in enumerate(candidates) if c in valid_set]
            if keep_idx_pt:
                candidates = [candidates[i] for i in keep_idx_pt]
                scores = [scores[i] for i in keep_idx_pt]
                all_metrics = [all_metrics[i] for i in keep_idx_pt]

        # ----- Cluster-based selection -----
        # Skip clustering for atlas-constrained search: atlas + arterial
        # cascade + venous filtering already eliminate false positives, and
        # forcing candidates into one cluster discards valid bilateral ICA
        # voxels.  Keep clustering for whole-brain search where spatial
        # coherence discriminates true arterial clusters from scattered FPs.
        if len(candidates) > n_candidates and not atlas_constrained:
            candidates, scores, all_metrics = self._score_and_select_cluster(
                candidates, scores, all_metrics, brain_shape, tr=tr
            )

        # ----- Filter venous candidates (Fix 1: Enforce venous exclusion) -----
        # This is the critical fix: venous candidates were flagged but never filtered
        if candidates:
            candidates, scores, all_metrics = self._filter_venous_candidates(
                candidates, scores, all_metrics, min_non_venous=3
            )

        # ----- Sort by score descending (return ALL for downstream filtering) -----
        if candidates:
            sorted_indices = np.argsort(scores)[::-1]
            candidates = [candidates[int(i)] for i in sorted_indices]
            scores = [scores[int(i)] for i in sorted_indices]
            all_metrics = [all_metrics[int(i)] for i in sorted_indices]

        return candidates, scores, all_metrics

    # -----------------------------------------------------------------
    # Weighted ROI Averaging + Smoothing (Phase 5)
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_weighted_roi_aif(
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        roi_size: int = 1,
        tr: float = 1.0,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Compute inverse-score weighted AIF with optional ROI averaging and smoothing.

        When ``roi_size=1`` (default), each candidate contributes only its own
        voxel curve — no spatial expansion into neighboring tissue.  This
        avoids tissue contamination that
        broadens the AIF at high spatial resolution.

        Args:
            candidates: Candidate voxel coordinates.
            scores: Quality scores (higher is better).
            conc_data: 4D concentration data.
            brain_mask: 3D brain mask.
            roi_size: Side length of cubic ROI (1 = single voxel).
            tr: Temporal resolution in seconds (for adaptive smoothing).

        Returns:
            (aif_curve, aif_mask) — smoothed 1D AIF and 3D boolean mask.
        """
        from scipy.signal import savgol_filter

        nt = conc_data.shape[-1]
        aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)

        if len(candidates) == 1:
            x, y, z = candidates[0]
            # Cubic ROI of roi_size^3 around single candidate
            slices = []
            for dim, coord_val in enumerate((x, y, z)):
                s = max(0, coord_val - roi_size // 2)
                e = min(conc_data.shape[dim], coord_val + roi_size // 2 + 1)
                slices.append(slice(s, e))
            roi = np.zeros(conc_data.shape[:3], dtype=bool)
            roi[tuple(slices)] = True
            valid = roi & (brain_mask > 0.5)
            aif_mask[valid] = 1
            roi_data = conc_data[valid]
            if roi_data.shape[0] > 0:
                aif_curve = np.mean(roi_data, axis=0)
            else:
                aif_mask[x, y, z] = 1
                aif_curve = conc_data[x, y, z, :]
        else:
            # Inverse-score weighting (lower cost = higher weight)
            score_arr = np.array(scores)
            weights = score_arr / np.sum(score_arr)  # higher score → higher weight

            weighted_aif = np.zeros(nt, dtype=np.float64)
            for i, (x, y, z) in enumerate(candidates):
                slices = []
                for dim, coord_val in enumerate((x, y, z)):
                    s = max(0, coord_val - roi_size // 2)
                    e = min(conc_data.shape[dim], coord_val + roi_size // 2 + 1)
                    slices.append(slice(s, e))
                roi = np.zeros(conc_data.shape[:3], dtype=bool)
                roi[tuple(slices)] = True
                valid = roi & (brain_mask > 0.5)
                aif_mask[valid] = 1

                roi_data = conc_data[valid]
                if roi_data.shape[0] > 0:
                    roi_aif = np.mean(roi_data, axis=0)
                else:
                    roi_aif = conc_data[x, y, z, :]
                weighted_aif += weights[i] * roi_aif

            aif_curve = weighted_aif

        # Savitzky-Golay smoothing for noise reduction (standard TR only).
        # At TR > 5s, the acquisition itself low-pass filters the signal
        # and the first-pass peak spans only 1-3 frames. SG would
        # destructively broaden the peak (e.g., 44% attenuation at TR=7.6s).
        # Do not smooth DCE AIF by default at long TR.
        if len(aif_curve) >= 5 and tr <= 5.0:
            sg_window = min(5, len(aif_curve))
            sg_poly = 2
            if sg_window >= 3:
                aif_curve = savgol_filter(
                    aif_curve, window_length=sg_window, polyorder=sg_poly,
                )

        # Clip negative to zero
        aif_curve = np.maximum(aif_curve, 0.0)

        return aif_curve, aif_mask

    # -----------------------------------------------------------------
    # Post-Averaging FWHM Validation with Retry Cascade
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_dce_averaged_aif(
        aif_curve: npt.NDArray[Any],
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        metrics_list: list[dict[str, float]],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], dict[str, Any]]:
        """Validate averaged AIF FWHM and retry with fewer candidates if degraded.

        Tissue contamination from ROI expansion or heterogeneous candidates can
        broaden the averaged AIF well beyond the individual candidate FWHM.
        This method detects such broadening and retries with progressively
        fewer candidates until the FWHM is acceptable.

        Ported from DSC ``_validate_averaged_aif`` (dsc_aif.py:1333) with
        DCE-specific criteria (FWHM ratio instead of SPR).

        Args:
            aif_curve: 1D averaged AIF from ``_compute_weighted_roi_aif``.
            candidates: Candidate voxel coordinates.
            scores: Candidate quality scores.
            metrics_list: Per-candidate metric dicts (must contain ``fwhm_seconds``).
            conc_data: 4D concentration data.
            brain_mask: 3D brain mask.
            tr: Temporal resolution in seconds.
            baseline_frames: Number of pre-bolus baseline frames.

        Returns:
            (aif_curve, aif_mask, validation_qc) — validated AIF, mask, and QC dict.
        """
        validation_qc: dict[str, Any] = {}

        # Compute final FWHM of the averaged curve
        final_fwhm = compute_dce_aif_fwhm_seconds(aif_curve, tr, baseline_frames)
        final_peak = float(np.max(aif_curve))

        # Compute mean candidate FWHM and peak
        candidate_fwhms = [
            m.get("fwhm_seconds", 0.0) for m in metrics_list if m.get("fwhm_seconds", 0.0) > 0
        ]
        candidate_peaks = [m.get("peak_mm", 0.0) for m in metrics_list if m.get("peak_mm", 0.0) > 0]

        if not candidate_fwhms or not candidate_peaks:
            # No valid candidate metrics — can't validate, accept as-is
            aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
            for x, y, z in candidates:
                aif_mask[x, y, z] = 1
            validation_qc["aif_validation_skipped"] = True
            return aif_curve, aif_mask, validation_qc

        mean_cand_fwhm = float(np.mean(candidate_fwhms))
        mean_cand_peak = float(np.mean(candidate_peaks))

        validation_qc["averaged_fwhm_seconds"] = round(final_fwhm, 1)
        validation_qc["candidate_fwhm_mean_seconds"] = round(mean_cand_fwhm, 1)
        validation_qc["averaged_peak_mm"] = round(final_peak, 3)
        validation_qc["candidate_peak_mean_mm"] = round(mean_cand_peak, 3)

        # Check for broadening / attenuation
        fwhm_ratio = final_fwhm / max(mean_cand_fwhm, 0.1)
        peak_ratio = final_peak / max(mean_cand_peak, EPSILON_CONCENTRATION)

        needs_retry = fwhm_ratio > 2.0 or peak_ratio < 0.3
        validation_qc["fwhm_ratio"] = round(fwhm_ratio, 2)
        validation_qc["peak_ratio"] = round(peak_ratio, 2)

        if not needs_retry:
            validation_qc["aif_validation_retries"] = 0
            aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
            for x, y, z in candidates:
                aif_mask[x, y, z] = 1
            return aif_curve, aif_mask, validation_qc

        # --- Retry 1: roi_size=1 with top 3 candidates ---
        retry_reason = []
        if fwhm_ratio > 2.0:
            retry_reason.append(f"fwhm_ratio={fwhm_ratio:.1f}")
        if peak_ratio < 0.3:
            retry_reason.append(f"peak_ratio={peak_ratio:.2f}")
        validation_qc["aif_validation_retry_reason"] = ", ".join(retry_reason)

        logger.warning(
            "AIF FWHM validation: averaged FWHM %.1fs vs candidate mean %.1fs "
            "(ratio %.1f). Retrying with top 3 candidates, roi_size=1.",
            final_fwhm, mean_cand_fwhm, fwhm_ratio,
        )

        top_n = min(3, len(candidates))
        retry1_cands = candidates[:top_n]
        retry1_scores = scores[:top_n]
        retry1_curve, retry1_mask = AIFDetection._compute_weighted_roi_aif(
            retry1_cands, retry1_scores, conc_data, brain_mask,
            roi_size=1, tr=tr,
        )
        retry1_fwhm = compute_dce_aif_fwhm_seconds(retry1_curve, tr, baseline_frames)
        retry1_peak = float(np.max(retry1_curve))
        validation_qc["retry1_fwhm_seconds"] = round(retry1_fwhm, 1)
        validation_qc["retry1_peak_mm"] = round(retry1_peak, 3)

        retry1_fwhm_ratio = retry1_fwhm / max(mean_cand_fwhm, 0.1)
        if retry1_fwhm_ratio <= 2.0 and retry1_peak / max(mean_cand_peak, EPSILON_CONCENTRATION) >= 0.3:
            validation_qc["aif_validation_retries"] = 1
            validation_qc["aif_validation_final_roi_size"] = 1
            return retry1_curve, retry1_mask, validation_qc

        # --- Retry 2: single best candidate, no averaging ---
        logger.warning(
            "AIF FWHM validation retry 1 still degraded (FWHM %.1fs). "
            "Using single best candidate.",
            retry1_fwhm,
        )

        best_cand = [candidates[0]]
        best_score = [scores[0]]
        retry2_curve, retry2_mask = AIFDetection._compute_weighted_roi_aif(
            best_cand, best_score, conc_data, brain_mask,
            roi_size=1, tr=tr,
        )
        retry2_fwhm = compute_dce_aif_fwhm_seconds(retry2_curve, tr, baseline_frames)
        validation_qc["retry2_fwhm_seconds"] = round(retry2_fwhm, 1)
        validation_qc["retry2_peak_mm"] = round(float(np.max(retry2_curve)), 3)

        # Pick the best result across all attempts
        attempts = [
            (aif_curve, fwhm_ratio, 0),
            (retry1_curve, retry1_fwhm_ratio, 1),
            (retry2_curve, retry2_fwhm / max(mean_cand_fwhm, 0.1), 2),
        ]
        # Prefer the attempt closest to candidate FWHM (ratio closest to 1.0)
        best_attempt = min(attempts, key=lambda a: abs(a[1] - 1.0))
        best_curve = best_attempt[0]
        best_retry_idx = best_attempt[2]

        if best_retry_idx == 0:
            best_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
            for x, y, z in candidates:
                best_mask[x, y, z] = 1
        elif best_retry_idx == 1:
            best_mask = retry1_mask
        else:
            best_mask = retry2_mask

        validation_qc["aif_validation_retries"] = 2
        validation_qc["aif_validation_best_attempt"] = best_retry_idx
        validation_qc["aif_validation_final_roi_size"] = 1
        if best_attempt[1] > 2.0:
            validation_qc["aif_fwhm_warning"] = True
            validation_qc["aif_confidence"] = "reduced"

        return best_curve, best_mask, validation_qc

    # -----------------------------------------------------------------
    # S0-based Partial Volume Rejection (Phase 5)
    # -----------------------------------------------------------------

    @staticmethod
    def _reject_partial_volume_candidates(
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        metrics_list: list[dict[str, float]],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        baseline_frames: int,
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Reject candidates with high baseline signal (partial volume).

        Candidates with baseline signal (S0) above median brain S0 indicate
        tissue contamination (large tissue fraction = partial volume AIF).

        Args:
            candidates: Candidate coordinates.
            scores: Candidate scores.
            metrics_list: Candidate metrics.
            conc_data: 4D concentration data.
            brain_mask: 3D brain mask.
            baseline_frames: Number of baseline frames.

        Returns:
            Filtered (candidates, scores, metrics).
        """
        if not candidates or baseline_frames < 1:
            return candidates, scores, metrics_list

        # Compute S0 (mean baseline) for each candidate
        s0_candidates = np.array([
            float(np.mean(conc_data[x, y, z, :baseline_frames]))
            for x, y, z in candidates
        ])

        # Compute median S0 across brain
        baseline_mean_vol = np.mean(conc_data[..., :baseline_frames], axis=-1)
        brain_s0 = baseline_mean_vol[brain_mask > 0.5]
        if len(brain_s0) == 0:
            return candidates, scores, metrics_list
        median_brain_s0 = float(np.median(brain_s0))

        if median_brain_s0 < EPSILON_T1:
            return candidates, scores, metrics_list

        # Reject candidates with S0 > 2× median (high tissue content)
        keep_idx = [
            i for i, s0 in enumerate(s0_candidates)
            if s0 <= median_brain_s0 * 2.0
        ]

        if not keep_idx or len(keep_idx) < 3:
            # Don't reject if it would leave too few candidates
            return candidates, scores, metrics_list

        return (
            [candidates[i] for i in keep_idx],
            [scores[i] for i in keep_idx],
            [metrics_list[i] for i in keep_idx],
        )

    # -----------------------------------------------------------------
    # External AIF Handling
    # -----------------------------------------------------------------

    def _handle_external_aif(
        self,
        runtime: Any,
        conc_data: npt.NDArray[Any],
        affine: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
    ) -> Any:
        """Handle externally supplied AIF curve, bypassing all detection.

        Loads an external AIF from .txt, .npz, or .npy file, validates shape,
        values, and length against DCE timepoints, and writes outputs.
        External AIFs are assumed to already be in plasma concentration (mM)
        — no hematocrit correction or AIF scaling is applied.

        Parameters
        ----------
        runtime : Any
            Nipype runtime object.
        conc_data : ndarray
            4D concentration data (x, y, z, t).
        affine : ndarray
            NIfTI affine matrix from concentration image.
        brain_mask : ndarray
            3D brain mask.

        Returns
        -------
        runtime
            Updated Nipype runtime object.
        """
        import nibabel as nib  # type: ignore[import-untyped]

        ext_path = Path(self.inputs.external_aif_file)
        n_timepoints = conc_data.shape[3]
        tr = float(self.inputs.tr)
        baseline_frames = int(self.inputs.baseline_frames)
        field_strength = float(self.inputs.field_strength)
        blood_t1 = get_blood_t1(field_strength)

        # Load external AIF
        suffix = ext_path.suffix.lower()
        if suffix == ".npz":
            npz = np.load(str(ext_path))
            if "Cp" in npz:
                aif_curve = np.asarray(npz["Cp"], dtype=np.float64)
            elif "aif" in npz:
                aif_curve = np.asarray(npz["aif"], dtype=np.float64)
            else:
                raise ValueError(
                    f"External AIF .npz must contain 'Cp' or 'aif' key, "
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

        # Shape validation: squeeze (N,1) -> (N,), reject if not 1-D
        aif_curve = np.squeeze(aif_curve)
        if aif_curve.ndim != 1:
            raise ValueError(
                f"External AIF must be 1-D, got shape {aif_curve.shape}. "
                f"Multi-column files are not supported — provide one value per line."
            )

        # Value validation: NaN/Inf
        if not np.all(np.isfinite(aif_curve)):
            raise ValueError(
                "External AIF contains NaN or Inf values. "
                "All values must be finite."
            )

        # Value validation: negative values
        if np.any(aif_curve < 0):
            n_neg = int(np.sum(aif_curve < 0))
            logger.warning(
                "External AIF has %d negative values; clipping to zero.", n_neg
            )
            aif_curve = np.clip(aif_curve, 0.0, None)

        # Validate length
        if len(aif_curve) != n_timepoints:
            raise ValueError(
                f"External AIF has {len(aif_curve)} timepoints but DCE data "
                f"has {n_timepoints} volumes. They must match."
            )

        # Validate non-zero peak
        peak = float(np.max(aif_curve))
        if peak < 1e-6:
            raise ValueError(
                f"External AIF peak ({peak:.2e}) is near zero. "
                "Check that values are concentration in mM (not raw signal)."
            )

        # Physiological range warnings
        if peak < 1.0:
            logger.warning(
                "External AIF peak (%.3f mM) is below expected range "
                "(typical full-dose: 5-7.6 mM). Check units and dose.",
                peak,
            )
        elif peak > 20.0:
            logger.warning(
                "External AIF peak (%.1f mM) exceeds expected range "
                "(typical full-dose: 5-7.6 mM). Check units.",
                peak,
            )
        baseline_mean = float(np.mean(aif_curve[:baseline_frames])) if baseline_frames > 0 else 0.0
        if baseline_mean > 0.5:
            logger.warning(
                "External AIF baseline mean (%.3f mM) is elevated. "
                "Expected near-zero before contrast arrival.",
                baseline_mean,
            )

        # Log hematocrit notice
        logger.info(
            "External AIF assumed to be in PLASMA concentration (mM). "
            "No hematocrit correction applied."
        )

        # Compute final AIF metrics
        final_peak = compute_dce_aif_peak_concentration(aif_curve)
        final_fwhm = compute_dce_aif_fwhm_seconds(aif_curve, tr, baseline_frames)
        final_auc = compute_dce_aif_auc(aif_curve, tr)
        final_washout = compute_dce_aif_washout(aif_curve, baseline_frames)

        # Compute quality score from measured metrics (not hardcoded)
        quality_score = 0.0
        # Peak in physiological range (0-0.3)
        if 2.0 <= final_peak <= 15.0:
            quality_score += 0.3
        elif 1.0 <= final_peak <= 20.0:
            quality_score += 0.15
        # FWHM in expected range (0-0.25): 8-25s typical
        if 5.0 <= final_fwhm <= 40.0:
            quality_score += 0.25
        elif final_fwhm > 0:
            quality_score += 0.1
        # Washout > 0.2 is arterial (0-0.25)
        quality_score += min(final_washout / 0.6, 1.0) * 0.25
        # Baseline stability (0-0.2)
        if baseline_frames > 1 and baseline_mean < 0.1:
            quality_score += 0.2
        elif baseline_mean < 0.5:
            quality_score += 0.1
        quality_score = min(quality_score, 1.0)

        # QC metrics
        from pmriprep.utils.qc_standards import QCStatus

        qc_metrics: dict[str, Any] = {
            "external_aif": True,
            "external_aif_file": str(ext_path),
            "aif_source": "external",  # For boilerplate text generation
            "final_peak_mm": final_peak,
            "final_fwhm_seconds": final_fwhm,
            "final_auc": final_auc,
            "final_washout": final_washout,
            "blood_t1_seconds": blood_t1,
            "field_strength_t": field_strength,
            "scaling_factor": 1.0,
            "detection_method": "external",
            "quality_score": quality_score,
            "baseline_frames": baseline_frames,
            "n_aif_voxels": 0,
            "aif_status": QCStatus.PASS,
        }

        # Write AIF curve
        out_aif_curve = self._gen_outfilename("_dce_aif_curve.txt")
        np.savetxt(out_aif_curve, aif_curve)

        # Write zero AIF mask (no spatial origin for external AIF)
        aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        out_aif_mask = self._gen_outfilename("_dce_aif_mask.nii.gz")
        mask_img = nib.Nifti1Image(aif_mask, affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_aif_mask)  # type: ignore[attr-defined]

        logger.info(
            "Using external AIF from %s (peak=%.3f mM, AUC=%.1f mM·s, quality=%.2f)",
            ext_path.name, final_peak, final_auc, quality_score,
        )

        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "aif_curve": [float(x) for x in aif_curve],
            "detection_method": "external",
            "quality_score": quality_score,
            "peak_concentration": final_peak,
            "fwhm_seconds": final_fwhm,
            "auc": final_auc,
            "blood_t1": blood_t1,
            "scaling_factor": 1.0,
            "candidate_count": 0,
            "qc_metrics": qc_metrics,
        }

        return runtime

    # -----------------------------------------------------------------
    # Main Interface Execution
    # -----------------------------------------------------------------

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE AIF detection."""
        import nibabel as nib  # type: ignore[import-untyped]

        # Load data
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]
        header = conc_img.header  # type: ignore[attr-defined]

        brain_img = nib.load(self.inputs.brain_mask_file)
        brain_mask: npt.NDArray[Any] = brain_img.get_fdata()  # type: ignore[attr-defined]

        # ===== External AIF: bypass all detection if provided =====
        if isdefined(self.inputs.external_aif_file) and self.inputs.external_aif_file:
            return self._handle_external_aif(
                runtime, conc_data, affine, brain_mask,
            )

        # Optional atlas masks — build ordered list (tightest → broadest)
        # User-supplied aif_mask_file overrides all atlas masks.
        atlas_masks: list[npt.NDArray[Any]] = []
        atlas_names: list[str] = []
        if isdefined(self.inputs.aif_mask_file) and self.inputs.aif_mask_file:
            atlas_img = nib.load(self.inputs.aif_mask_file)
            atlas_masks = [atlas_img.get_fdata()]  # type: ignore[attr-defined]
            atlas_names = [Path(self.inputs.aif_mask_file).name.split(".")[0].replace("MNI_2mm_MRA_arterial_", "")]
        else:
            # Load tiered atlas cascade (atlas_mask_files, tightest first)
            if (
                isdefined(self.inputs.atlas_mask_files)
                and self.inputs.atlas_mask_files is not None
            ):
                for af in self.inputs.atlas_mask_files:
                    af_path = Path(str(af))
                    if af_path.exists() and str(af) not in ("", "None"):
                        atlas_masks.append(
                            nib.load(str(af_path)).get_fdata()  # type: ignore[attr-defined]
                        )
                        atlas_names.append(
                            af_path.name.split(".")[0].replace("MNI_2mm_MRA_arterial_", "")
                        )
            # Backward compat: single atlas_mask_file appended if not already loaded
            if (
                not atlas_masks
                and isdefined(self.inputs.atlas_mask_file)
                and self.inputs.atlas_mask_file is not None
                and str(self.inputs.atlas_mask_file) != "None"
                and Path(self.inputs.atlas_mask_file).exists()
            ):
                atlas_masks = [
                    nib.load(self.inputs.atlas_mask_file).get_fdata()  # type: ignore[attr-defined]
                ]
                atlas_names = [Path(self.inputs.atlas_mask_file).name.split(".")[0].replace("MNI_2mm_MRA_arterial_", "")]

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
        lesion_distance = float(self.inputs.lesion_distance)
        r1_relaxivity = float(self.inputs.r1_relaxivity)

        n_timepoints = conc_data.shape[3]

        # Get blood T1 for this field strength
        blood_t1 = get_blood_t1(field_strength)

        # Voxel sizes from header
        voxel_sizes = tuple(float(v) for v in header.get_zooms()[:3])  # type: ignore[attr-defined]

        # Initialize results
        detection_method: Literal["atlas", "automatic", "population"] = "population"
        candidates: list[tuple[int, int, int]] = []
        scores: list[float] = []
        metrics_list: list[dict[str, float]] = []
        tissue_ttp_frame: float | None = None
        _detection_attempt: dict[str, Any] | None = None
        atlas_tier_used: str | None = None
        validation_qc: dict[str, Any] = {}

        # Skip auto detection if population method requested
        if aif_method == "population":
            pass
        else:
            # ===== Phase 1: Prepare search masks =====
            _lesion_dist = max(lesion_distance, LESION_EXCLUSION_DISTANCE_MM)
            # Atlas search: skip midline exclusion (MRA-derived atlas
            # already excludes venous structures) and use reduced slab
            # edge exclusion (atlas constrains to known arterial regions)
            atlas_search_mask = self._prepare_search_mask(
                brain_mask,
                voxel_sizes,
                lesion_mask=lesion_mask,
                lesion_distance_mm=_lesion_dist,
                skip_midline=True,
                slab_edge_mm=SLAB_EDGE_EXCLUSION_MM_ATLAS,
            )
            # Whole-brain fallback: full slab edge exclusion + midline
            wb_search_mask = self._prepare_search_mask(
                brain_mask,
                voxel_sizes,
                lesion_mask=lesion_mask,
                lesion_distance_mm=_lesion_dist,
                skip_midline=False,
                slab_edge_mm=SLAB_EDGE_EXCLUSION_MM,
            )

            # ===== Phase 3: Compute tissue TTP reference =====
            tissue_ttp_frame = self._calculate_tissue_ttp_reference(
                conc_data, brain_mask, baseline_frames
            )

            # ===== Multi-level detection cascade =====
            # Atlas cascade: tightest → broadest, each with strict → permissive.
            # Trying the tightest atlas first (MCA_ACA) minimizes tissue
            # contamination; broader atlases serve as fallback.
            if atlas_masks:
                for atlas_idx, atlas_mask in enumerate(atlas_masks):
                    for crit in ("strict", "relaxed", "permissive"):
                        combined_mask = (atlas_search_mask > 0.5) & (atlas_mask > 0.5)
                        combined_mask = combined_mask.astype(atlas_search_mask.dtype)
                        candidates, scores, metrics_list = self._find_candidates_in_mask(
                            conc_data, combined_mask, n_candidates, tr,
                            baseline_frames, crit,
                            tissue_ttp_frame=tissue_ttp_frame,
                            atlas_constrained=True,
                        )
                        if candidates:
                            detection_method = "atlas"
                            atlas_tier_used = f"{atlas_names[atlas_idx]}_{crit}"
                            break
                    if candidates:
                        break

            # Whole-brain fallback: relaxed only.
            # If the full atlas cascade AND whole-brain relaxed all fail,
            # data quality is too poor for voxel-based AIF — go to population
            # rather than scraping with permissive/degraded (which finds tissue).
            if not candidates:
                for crit in ("relaxed",):
                    candidates, scores, metrics_list = self._find_candidates_in_mask(
                        conc_data, wb_search_mask, n_candidates, tr,
                        baseline_frames, crit,
                        tissue_ttp_frame=tissue_ttp_frame,
                        atlas_constrained=False,
                    )
                    if candidates:
                        detection_method = "automatic"
                        break

            # ===== Phase 5: S0-based partial volume rejection =====
            if candidates:
                candidates, scores, metrics_list = self._reject_partial_volume_candidates(
                    candidates, scores, metrics_list,
                    conc_data, brain_mask, baseline_frames,
                )

            # ===== Phase 6: Soft contralateral penalty =====
            if candidates and lesion_mask is not None:
                tumor_lat = self._determine_tumor_laterality(lesion_mask, affine)
                candidates, scores, metrics_list = self._filter_candidates_by_hemisphere(
                    candidates, scores, metrics_list,
                    conc_data.shape[:3], affine, tumor_lat,
                )

            # ===== Bilateral balance (always applied) =====
            if candidates:
                candidates, scores, metrics_list = self._ensure_bilateral_balance(
                    candidates, scores, metrics_list,
                    affine, n_candidates,
                )

            # ===== Take top N =====
            if candidates and len(candidates) > n_candidates:
                sorted_idx = sorted(
                    range(len(candidates)), key=lambda i: scores[i], reverse=True
                )[:n_candidates]
                candidates = [candidates[i] for i in sorted_idx]
                scores = [scores[i] for i in sorted_idx]
                metrics_list = [metrics_list[i] for i in sorted_idx]

            # Venous filtering already applied inside _find_candidates_in_mask
            # (after cluster selection). No need to duplicate here.

            # ===== Per-candidate quality filter =====
            # Low-scoring candidates (< 0.5) are partial-volume contaminated
            # (tiers 6-7). Averaging them in degrades the AIF. Filter by
            # individual score, then decide based on survivor count.
            n_filtered_out = 0
            if candidates and aif_method != "population":
                n_before = len(candidates)
                keep = [i for i, s in enumerate(scores) if s >= _MIN_CANDIDATE_SCORE]
                n_filtered_out = n_before - len(keep)

                if len(keep) >= _MIN_CANDIDATES_FOR_DETECTED:
                    # Enough high-quality candidates survive
                    if n_filtered_out > 0:
                        logger.info(
                            "AIF quality filter: kept %d/%d candidates "
                            "(removed %d with score < %.2f).",
                            len(keep), n_before, n_filtered_out,
                            _MIN_CANDIDATE_SCORE,
                        )
                    candidates = [candidates[i] for i in keep]
                    scores = [scores[i] for i in keep]
                    metrics_list = [metrics_list[i] for i in keep]
                elif len(keep) == 1:
                    # Single high-quality voxel — use it but warn
                    logger.warning(
                        "AIF quality filter: only 1/%d candidate with "
                        "score >= %.2f. Using single voxel AIF "
                        "(no consistency check possible).",
                        n_before, _MIN_CANDIDATE_SCORE,
                    )
                    candidates = [candidates[keep[0]]]
                    scores = [scores[keep[0]]]
                    metrics_list = [metrics_list[keep[0]]]
                else:
                    # No candidates above threshold — fall back to population
                    best_score = max(scores) if scores else 0.0
                    _detection_attempt = {
                        "auto_detection_attempted": True,
                        "auto_best_score": float(best_score),
                        "auto_n_candidates": n_before,
                        "auto_n_above_threshold": 0,
                        "auto_fallback_reason": (
                            f"0/{n_before} candidates scored >= "
                            f"{_MIN_CANDIDATE_SCORE}"
                        ),
                    }
                    logger.warning(
                        "AIF quality filter: 0/%d candidates scored >= %.2f "
                        "(best=%.2f). Falling back to Parker population AIF.",
                        n_before, _MIN_CANDIDATE_SCORE, best_score,
                    )
                    candidates = []
                    scores = []
                    metrics_list = []

        # ===== Generate AIF curve =====
        aif_curve: npt.NDArray[Any]
        aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        quality_score = 0.0
        scaling_factor = 1.0
        qc_metrics: dict[str, Any] = {}

        if candidates:
            # Phase 5: Weighted ROI averaging + smoothing
            aif_curve, aif_mask = self._compute_weighted_roi_aif(
                candidates, scores, conc_data, brain_mask, tr=tr,
            )

            # Phase 5b: Validate averaged AIF FWHM against candidate FWHM
            aif_curve, aif_mask, validation_qc = self._validate_dce_averaged_aif(
                aif_curve, candidates, scores, metrics_list,
                conc_data, brain_mask, tr, baseline_frames,
            )

            # Convert measured AIF from blood to plasma concentration
            # (tissue conc is in blood units; AIF must be in plasma units
            #  for Tofts model: plasma = blood / (1 - Hct))
            aif_curve = aif_curve / (1.0 - float(self.inputs.hematocrit))
            quality_score = float(np.mean(scores))

            # Aggregate QC metrics
            if metrics_list:
                # Get tier used from first candidate (all from same tier)
                tier_used = int(metrics_list[0].get("tier_used", 0))
                fwhm_frames_mean = float(
                    np.mean([m.get("fwhm_frames", 0.0) for m in metrics_list])
                )
                qc_metrics = {
                    "aif_source": "auto",  # For boilerplate text generation
                    "peak_mm_mean": float(np.mean([m["peak_mm"] for m in metrics_list])),
                    "fwhm_seconds_mean": float(
                        np.mean([m["fwhm_seconds"] for m in metrics_list])
                    ),
                    "fwhm_frames_mean": fwhm_frames_mean,
                    "washout_mean": float(
                        np.mean([m["washout"] for m in metrics_list])
                    ),
                    "cnr_mean": float(np.mean([m["cnr"] for m in metrics_list])),
                    "auc_mean": float(np.mean([m["auc"] for m in metrics_list])),
                    "upstroke_slope_mean": float(
                        np.mean([m.get("upstroke_slope", 0.0) for m in metrics_list])
                    ),
                    "n_aif_voxels": len(candidates),
                    "n_aif_voxels_filtered_out": n_filtered_out,
                    "candidate_coords": [list(c) for c in candidates],
                    "candidate_scores": [float(s) for s in scores],
                    "arterial_cascade_tier": tier_used,
                    "low_tr_mode": is_low_temporal_resolution(tr),
                }
                if atlas_tier_used is not None:
                    qc_metrics["atlas_tier_used"] = atlas_tier_used

            # Merge FWHM validation QC
            if validation_qc:
                qc_metrics.update(validation_qc)

            # Hemisphere coverage: classify candidates as left/right via RAS x-coord
            if candidates:
                n_left = 0
                n_right = 0
                for cx, cy, cz in candidates:
                    world = affine[:3, :3] @ np.array([cx, cy, cz]) + affine[:3, 3]
                    if world[0] < 0:
                        n_left += 1
                    else:
                        n_right += 1
                qc_metrics["n_left_candidates"] = n_left
                qc_metrics["n_right_candidates"] = n_right
                if n_left > 0 and n_right > 0:
                    qc_metrics["hemisphere_coverage"] = "bilateral"
                elif n_left > 0:
                    qc_metrics["hemisphere_coverage"] = "unilateral_left"
                else:
                    qc_metrics["hemisphere_coverage"] = "unilateral_right"

            # Venous contamination detection and warning
            # Check if any candidates were flagged as likely venous
            n_venous = sum(
                1 for m in metrics_list if m.get("is_likely_venous", False)
            )
            if n_venous > 0:
                qc_metrics["n_likely_venous_candidates"] = n_venous
                qc_metrics["venous_fraction"] = n_venous / len(metrics_list)

            # Cross-validate final AIF peak timing against tissue TTP
            if tissue_ttp_frame is not None:
                aif_peak_frame = int(np.argmax(aif_curve))
                timing_diff = tissue_ttp_frame - aif_peak_frame
                qc_timing_margin = compute_timing_margin_frames(tr)
                low_tr = is_low_temporal_resolution(tr)
                qc_metrics["aif_ttp_frame"] = aif_peak_frame
                qc_metrics["tissue_ttp_frame"] = tissue_ttp_frame
                qc_metrics["aif_before_tissue_frames"] = timing_diff

                # At high TR, only flag as venous if peak is >1 frame after tissue
                if low_tr:
                    qc_metrics["is_likely_venous"] = timing_diff < -1
                else:
                    qc_metrics["is_likely_venous"] = timing_diff < qc_timing_margin

                if timing_diff < -1:
                    logger.warning(
                        "VENOUS CONTAMINATION: AIF peaks %.0f frames AFTER tissue TTP. "
                        "Likely SSS or venous structure.",
                        abs(timing_diff),
                    )
                elif not low_tr and timing_diff < qc_timing_margin:
                    logger.warning(
                        "MARGINAL AIF TIMING: AIF peaks only %.0f frames before tissue TTP "
                        "(expected >= %d). May have venous contamination.",
                        timing_diff,
                        qc_timing_margin,
                    )

        else:
            # Generate BAT-aligned Parker population AIF
            dose_factor = self._get_dose_factor()
            t = np.arange(n_timepoints) * tr

            # For population AIF, use baseline_frames * TR as BAT. Data-driven BAT
            # detection on mean brain concentration detects tissue transit
            # delay, not arterial arrival.
            bat_shift = baseline_frames * tr
            t_shifted = t - bat_shift

            # Apply r1 relaxivity correction for non-standard agents.
            # Parker AIF was calibrated with Gd-DTPA (r1=4.1). Tissue
            # concentrations use the actual agent's r1 (e.g. 11.6 for Vueway).
            # The correction ensures AIF and tissue are in the same
            # "relaxivity space".
            r1_arg: float | None = None
            if bool(self.inputs.agent_r1_known):
                r1_relaxivity = float(self.inputs.r1_relaxivity)
                if r1_relaxivity > 0:
                    r1_arg = r1_relaxivity

            t_for_parker = np.maximum(t_shifted, 0.0)
            aif_curve = parker_population_aif(
                t_for_parker, dose_factor, r1_actual=r1_arg
            )

            # Zero out pre-bolus values (where t_shifted < 0)
            # Use small positive value (1e-6) to avoid log(0) issues downstream
            aif_curve = np.where(t_shifted < 0, 1e-6, aif_curve)

            # Sanity check: population AIF must have proper peak
            # If BAT shift clipped the AIF too early, regenerate without shift
            pop_peak = float(np.max(aif_curve))
            expected_min_peak = 2.5 * dose_factor  # ~2.5 mM minimum for full dose
            bat_override = False
            original_bat_seconds = bat_shift

            if pop_peak < expected_min_peak:
                # BAT shift is wrong - regenerate without shift
                t_unshifted = np.arange(n_timepoints) * tr
                aif_curve = parker_population_aif(
                    t_unshifted, dose_factor, r1_actual=r1_arg
                )
                bat_override = True

            # Scale population AIF to match pipeline's internal AIF convention.
            # The detected AIF path applies /(1-Hct) to convert blood-voxel
            # ΔR1/r1 to true plasma concentration C_p (line 2612).
            # Parker (2006) derived AIF parameters from ΔR1/r1 in blood voxels
            # WITHOUT Hct correction. Under FXL, ΔR1/r1 = (1-Hct) × C_p,
            # so Parker's reported "C_p" values are effectively in (1-Hct)*C_p
            # units. Dividing by (1-Hct) brings the population AIF into the
            # same true-plasma scale as the Hct-corrected detected AIF.
            aif_curve = aif_curve / (1.0 - float(self.inputs.hematocrit))

            detection_method = "population"
            quality_score = 0.5  # Moderate score for population AIF

            qc_metrics = {
                "population_aif": True,
                "aif_source": "population",  # For boilerplate text generation
                "dose_factor": dose_factor,
                "n_aif_voxels": 0,
                "bat_seconds": bat_shift,
                "bat_method": "baseline_frames",
            }
            if bat_override:
                qc_metrics["bat_override"] = True
                qc_metrics["original_bat_seconds"] = original_bat_seconds
            if _detection_attempt is not None:
                qc_metrics.update(_detection_attempt)

        # Apply AIF scaling if requested
        if scale_aif and detection_method not in ("population", "external"):
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

                if scaling_factor > 3.0:
                    logger.warning(
                        "AIF scaling factor %.2f > 3.0 — detected AIF peak "
                        "(%.2f mM) is far below expected arterial range. "
                        "Results may be unreliable.",
                        scaling_factor, current_peak,
                    )
                elif scaling_factor > 2.0:
                    logger.warning(
                        "AIF scaling factor %.2f > 2.0 — detected AIF peak "
                        "(%.2f mM) is substantially below expected range.",
                        scaling_factor, current_peak,
                    )

        # Compute final AIF metrics
        final_peak = compute_dce_aif_peak_concentration(aif_curve)
        final_fwhm = compute_dce_aif_fwhm_seconds(aif_curve, tr, baseline_frames)
        final_auc = compute_dce_aif_auc(aif_curve, tr)
        final_washout = compute_dce_aif_washout(aif_curve, baseline_frames)

        # Compute AIF status for HTML report
        from pmriprep.utils.qc_standards import QCStatus
        if not np.isnan(quality_score):
            if quality_score >= 0.7:
                aif_status = QCStatus.PASS
            elif quality_score >= 0.4:
                aif_status = QCStatus.WARN
            else:
                aif_status = QCStatus.POOR
        else:
            method_to_status = {
                "auto": QCStatus.PASS,
                "external": QCStatus.PASS,
                "population": QCStatus.WARN,
                "fallback": QCStatus.POOR,
            }
            aif_status = method_to_status.get(detection_method, QCStatus.WARN)

        # Add final metrics to QC dict
        qc_metrics.update(
            {
                "final_peak_mm": final_peak,
                "final_fwhm_seconds": final_fwhm,
                "final_auc": final_auc,
                "final_washout": final_washout,
                "blood_t1_seconds": blood_t1,
                "field_strength_t": field_strength,
                "scaling_factor": scaling_factor,
                "scaling_factor_warning": scaling_factor > 2.0,
                "detection_method": detection_method,
                "quality_score": quality_score,
                "baseline_frames": baseline_frames,
                "aif_status": aif_status,
            }
        )

        # Save outputs
        out_aif_curve = self._gen_outfilename("_dce_aif_curve.txt")
        np.savetxt(out_aif_curve, aif_curve)

        # Save AIF mask (zeros for population AIF, actual voxels otherwise)
        out_aif_mask = self._gen_outfilename("_dce_aif_mask.nii.gz")
        mask_img = nib.Nifti1Image(aif_mask, affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_aif_mask)  # type: ignore[attr-defined]

        # Store results for _list_outputs
        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,  # Zeros for population AIF
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
