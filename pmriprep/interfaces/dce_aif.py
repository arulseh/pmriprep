"""AIF (Arterial Input Function) detection interfaces for DCE-MRI.

This module implements Nipype interfaces for AIF detection in DCE perfusion
processing, with multi-level detection strategies (atlas-based, whole-brain,
population) and comprehensive quality metrics for DCE-specific requirements.

The DCE AIF differs from DSC AIF in several important ways:
- Concentration is in mM (not relative signal change)
- Expected peak values are in physiological range (5-7.6 mM for full dose)
- FWHM is in seconds (8-25s) rather than frames
- Blood T1 depends on field strength (1.5T→1.35s, 3T→1.65s)
"""

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, TypedDict

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
    AIF_CNR_SATURATION,
    AIF_FWHM_BAND_HALFWIDTH_SECONDS,
    AIF_IDEAL_FWHM_SECONDS,
    AIF_PEAK_TIMING_FRACTION,
    AIF_TAIL_PERSISTENCE_FLOOR,
    AIF_TAIL_PERSISTENCE_TARGET,
    AIF_WASHOUT_BAND_HALFWIDTH,
    AIF_WASHOUT_BAND_HALFWIDTH_UPPER,
    AIF_WASHOUT_PHYSIOLOGIC_CENTER,
    AIF_WASHOUT_THRESHOLD_RELAXED,
    AIF_WASHOUT_THRESHOLD_STRICT,
    BRAIN_EROSION_ITERATIONS,
    BRAIN_EROSION_ITERATIONS_ATLAS,
    BRAIN_EROSION_MIN_VOXELS,
    DCE_AIF_EXPECTED_FULL_DOSE,
    DCE_AIF_GATE_FWHM_MAX_SECONDS,
    DCE_AIF_GATE_G6_MIN_SLABS_UNILATERAL,
    DCE_AIF_GATE_G6_MIN_VOXELS_BILATERAL,
    DCE_AIF_GATE_G6_MIN_VOXELS_UNILATERAL,
    DCE_AIF_GATE_LATE_TAIL_RATIO_MAX,
    DCE_AIF_GATE_MULTIBUMP_MAX,
    DCE_AIF_GATE_RATIO30_HARD,
    DCE_AIF_GATE_RATIO30_WARN,
    DCE_AIF_GATE_SCALE_G6_CEILING,
    DCE_AIF_GATE_SCALE_G6_HIGH,
    DCE_AIF_GATE_SCALE_HARD,
    DCE_AIF_MIN_TRIM_POOL,
    DCE_AIF_PEAK_RATIO_WARN_THRESHOLD,
    DCE_AIF_SCALE_WARN_SUBSTANTIAL,
    DCE_AIF_SCALE_WARN_UNRELIABLE,
    DCE_AIF_TAIL_MIN_SEPARATION_SECONDS,
    DCE_AIF_TAIL_PEAK_FRACTION,
    DCE_AIF_TAIL_SHOULDER_SECONDS,
    EPSILON_CONCENTRATION,
    HEMATOCRIT_ARTERIAL,
    INFLOW_EXCLUSION_MAX_SLAB_FRACTION,
    INFLOW_EXCLUSION_MM_INFERIOR_BASE,
    LESION_EXCLUSION_DISTANCE_MM,
    LESION_EXCLUSION_FLOOR_MM,
    MIDLINE_EXCLUSION_FRACTION,
    MIN_MEANINGFUL_ENHANCEMENT_MM,
    SLAB_EDGE_EXCLUSION_MM_ROLLOFF,
    SSS_EXCLUSION_MM,
    SUPERIOR_EXCLUSION_FRACTION,
    WB_FALLBACK_DEPTH_EROSION_MM,
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
# Maximum post-arrival peak time (seconds) for the EARLIEST candidate in a
# tier to be considered "anchored" by an arterial-grade voxel. Derived from
# physiological arterial bolus arrival range (4-12s post-injection)
# plus capillary MTT margin (~4s in GM) for
# adult brain DCE. Tiers whose earliest candidate peaks later than this
# are treated as venous-only clusters and fail Pass-1 acceptance. If
# pediatric or extreme-cardiac-output cohorts are added later, expose as
# a config knob rather than tuning this constant.
_MAX_ANCHOR_POST_ARRIVAL_S: float = 10.0


def _inflow_exclusion_mm(
    slice_thickness_mm: float,
    n_slices: int,
) -> float:
    """Adaptive inferior-side inflow exclusion (mm) for 3D-slab DCE.

    Combines a geometric base (``INFLOW_EXCLUSION_MM_INFERIOR_BASE`` =
    20mm covers ~5 RF pulses' worth of arterial inflow at typical
    pulse-TRs) with a partition-edge term (4×slice_thickness for
    slab-profile wings) and a slab-fraction cap
    (``INFLOW_EXCLUSION_MAX_SLAB_FRACTION`` = 0.35) to prevent
    over-exclusion on thick-slab acquisitions. Caller must gate use of
    this function on ``MRAcquisitionType == "3D"``; 2D multi-slice
    excitation does not produce slab-entry inflow.

    Parameters
    ----------
    slice_thickness_mm : float
        Through-plane voxel size in mm.
    n_slices : int
        Number of slices in the acquisition slab.

    Returns
    -------
    float
        Recommended inferior-side exclusion distance in mm.
    """
    slab_depth_mm = slice_thickness_mm * n_slices
    return min(
        max(INFLOW_EXCLUSION_MM_INFERIOR_BASE, 4.0 * slice_thickness_mm),
        INFLOW_EXCLUSION_MAX_SLAB_FRACTION * slab_depth_mm,
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

    Uses normalized Gaussian components where A1/A2 are areas (mM·min) under
    the Gaussian curves. Dividing by (sigma * sqrt(2*pi)) converts these to
    peak amplitudes, producing ~6 mM peak for full dose (0.1 mmol/kg), which
    matches the population value (6.3 ± 1.5 mM).

    Output is a true-mM concentration curve. Concentration is relaxivity-
    independent (both tissue and detected AIF are ΔR1/r1_true = true mM), so NO
    r1 scaling is applied — only the molar-dose factor. The agent's r1 governs
    signal, which is already divided out upstream.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds.
    dose_factor : float, optional
        Scaling factor for dose (1.0 = full dose 0.1 mmol/kg,
        0.5 = half dose, 2.0 = double dose).

    Returns
    -------
    np.ndarray
        Concentration time course in mM.

    """
    # Convert time to minutes
    t_min = t / 60.0

    effective_factor = dose_factor

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

    Parameters
    ----------
    aif_curve : np.ndarray
        AIF concentration time course in mM.

    Returns
    -------
    float
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

    Parameters
    ----------
    aif_curve : np.ndarray
        AIF concentration time course in mM.
    tr : float
        Temporal resolution in seconds.
    baseline_frames : int, optional
        Number of pre-contrast baseline frames.

    Returns
    -------
    float
        FWHM in seconds.
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0.0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    peak = float(np.max(aif_curve))
    peak_idx = int(np.argmax(aif_curve))

    if peak_idx < baseline_frames:
        # Global max sits inside the baseline window -> no real bolus enhancement;
        # the rising-edge search would be empty and yield a spurious near-full FWHM.
        return 0.0

    half_max = baseline_mean + (peak - baseline_mean) / 2.0

    # Find rising edge with sub-frame interpolation
    rising_time = float(baseline_frames) * tr
    for i in range(baseline_frames, peak_idx):
        if aif_curve[i] >= half_max:
            if i > baseline_frames and aif_curve[i - 1] < half_max:
                denom = aif_curve[i] - aif_curve[i - 1]
                if abs(denom) > MIN_MEANINGFUL_ENHANCEMENT_MM:
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
                if abs(denom) > MIN_MEANINGFUL_ENHANCEMENT_MM:
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

    Parameters
    ----------
    aif_curve : np.ndarray
        AIF concentration time course in mM.
    baseline_frames : int, optional
        Number of pre-contrast baseline frames.
    tail_frames : int, optional
        Number of tail frames to average.

    Returns
    -------
    float
        Washout fraction (0-1).
    """
    if len(aif_curve) < baseline_frames + tail_frames + 1:
        return 0.0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    tail_mean = float(np.mean(aif_curve[-tail_frames:]))
    peak = float(np.max(aif_curve))

    denom = peak - baseline_mean
    if denom < MIN_MEANINGFUL_ENHANCEMENT_MM:
        return 0.0

    washout = (peak - tail_mean) / denom
    return max(0.0, min(1.0, washout))


def compute_dce_aif_peak_amplitude_warning(
    final_peak: float,
    expected_max_mm: float,
    detection_method: str,
    threshold: float = DCE_AIF_PEAK_RATIO_WARN_THRESHOLD,
) -> tuple[float, bool]:
    """Compute the peak-amplitude QC warning ratio + flag.

    Compares the cluster-mean AIF peak (plasma units, post-Hct division)
    against the dose-scaled Parker plasma ``peak_max`` and flags when the
    ratio exceeds ``threshold``. Suppressed for population-AIF fallback
    (population AIF is tautologically within its own expected range).

    Parameters
    ----------
    final_peak : float
        Cluster-mean AIF peak concentration (plasma, mM).
    expected_max_mm : float
        Dose-scaled Parker plasma ``peak_max`` (mM).
    detection_method : str
        Cascade detection method label
        ("atlas" / "whole_brain" / "population" / "external" /
        "fallback"). Warning is suppressed when this equals
        ``"population"``.
    threshold : float, optional
        Multiplier on ``expected_max_mm`` above which the
        warning fires. Defaults to
        ``DCE_AIF_PEAK_RATIO_WARN_THRESHOLD``.

    Returns
    -------
    tuple of (float, bool)
        ``(ratio, warning)`` — ``ratio`` is ``final_peak / expected_max_mm``
        (NaN when suppressed or when ``expected_max_mm <= 0``);
        ``warning`` is ``True`` iff the ratio exceeds ``threshold`` and
        is not suppressed.
    """
    if expected_max_mm <= 0 or detection_method == "population":
        return float("nan"), False
    ratio = float(final_peak) / expected_max_mm
    return ratio, ratio > threshold


def detect_dce_aif_multi_bump_tail(
    aif_curve: npt.NDArray[Any],
    tr: float,
    baseline_frames: int = 5,
    secondary_peak_fraction: float = DCE_AIF_TAIL_PEAK_FRACTION,
    tail_shoulder_seconds: float = DCE_AIF_TAIL_SHOULDER_SECONDS,
    min_separation_seconds: float = DCE_AIF_TAIL_MIN_SEPARATION_SECONDS,
) -> tuple[int, list[float]]:
    """Count secondary maxima in the post-peak AIF tail.

    Detects local maxima in the post-first-pass region whose absolute
    amplitude exceeds ``secondary_peak_fraction`` x first-pass peak.
    QC-only flag for motion / inflow / venous mixing artifacts; does not
    alter AIF detection or candidate selection.

    The 60s post-peak shoulder (``DCE_AIF_TAIL_SHOULDER_SECONDS``) excludes the
    physiologic recirculation hump, so a clean AIF has NO counted secondary
    maxima in-window; ``count > 1`` flags late-tail anomalies (venous mixing /
    motion / inflow) beyond 60s post-peak.

    Window parameters are specified in seconds and converted to frames
    via ``tr`` so the detector behaves consistently across acquisitions
    with TRs from sub-second (compressed-sensing 4D) to ~6s (clinical
    DCE-VIBE).

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration curve.
    tr : float
        Repetition time of the AIF time series (seconds).
    baseline_frames : int, optional
        Number of pre-contrast frames defining baseline.
    secondary_peak_fraction : float, optional
        Fraction of first-pass peak amplitude
        (absolute) above which a tail maximum is counted.
    tail_shoulder_seconds : float, optional
        Time (seconds) to skip immediately after
        the first-pass peak before searching for secondary maxima
        (avoids counting the descending shoulder of the first pass
        as a separate peak; ~half a typical first-pass FWHM).
    min_separation_seconds : float, optional
        Minimum time (seconds) between detected
        secondary peaks (~1/3 of the typical recirculation period;
        ``scipy.signal.find_peaks`` ``distance`` parameter,
        converted to frames using ``tr``).

    Returns
    -------
    tuple of (int, list of float)
        ``(count, heights)`` where ``count`` is the number of secondary
        peaks and ``heights`` is their absolute concentrations.
    """
    from scipy.signal import find_peaks

    if tr <= 0:
        return 0, []
    tail_shoulder_frames = max(1, int(round(tail_shoulder_seconds / tr)))
    min_separation_frames = max(1, int(round(min_separation_seconds / tr)))

    if aif_curve.size <= baseline_frames + tail_shoulder_frames + 2:
        return 0, []

    post = aif_curve[baseline_frames:]
    peak_idx_rel = int(np.argmax(post))
    first_pass_peak = float(post[peak_idx_rel])
    if first_pass_peak <= 0:
        return 0, []

    tail = post[peak_idx_rel + tail_shoulder_frames:]
    if tail.size < 3:
        return 0, []

    height_threshold = secondary_peak_fraction * first_pass_peak
    peak_indices, _ = find_peaks(
        tail, height=height_threshold, distance=min_separation_frames,
    )
    heights = [float(tail[i]) for i in peak_indices]
    return len(peak_indices), heights


def compute_dce_aif_baseline_stability(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute baseline stability as standard deviation for DCE AIF.

    For DCE, baseline should be approximately zero (pre-contrast).

    Parameters
    ----------
    aif_curve : np.ndarray
        AIF concentration time course in mM.
    baseline_frames : int, optional
        Number of pre-contrast baseline frames.

    Returns
    -------
    float
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

    Parameters
    ----------
    aif_curve : np.ndarray
        AIF concentration time course in mM.
    baseline_frames : int, optional
        Number of pre-contrast baseline frames.

    Returns
    -------
    float
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

    Parameters
    ----------
    aif_curve : np.ndarray
        AIF concentration time course in mM.
    tr : float
        Temporal resolution in seconds.

    Returns
    -------
    float
        AUC in mM·s.
    """
    # Use trapezoidal integration
    # Try numpy 2.0+ trapezoid, fall back to trapz
    trapezoid_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc = float(trapezoid_fn(aif_curve) * tr)

    return auc


def compute_dce_aif_ratio_30s(
    aif_curve: npt.NDArray[Any],
    tr: float,
    peak_idx: int | None = None,
) -> float | None:
    """Compute the ratio C[peak + 30s] / C[peak] for the AIF curve.

    The post-peak sample is taken at the larger of 4 frames or
    ``ceil(30 / tr)`` frames after the peak, so the gate behaves
    consistently across temporal resolutions from sub-second compressed-
    sensing 4D up to ~7 s clinical DCE-VIBE. Parker bi-exponential gives
    ratio ~0.32 at peak+30 s; venous AIFs sit at >=0.55.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration curve.
    tr : float
        Temporal resolution in seconds.
    peak_idx : int, optional
        Pre-computed peak index. If None, uses ``argmax(aif_curve)``.

    Returns
    -------
    float or None
        ``C[peak + 30s] / C[peak]``, or ``None`` if the curve is too
        short or the peak is non-positive.
    """
    if tr <= 0 or aif_curve.size < 3:
        return None
    if peak_idx is None:
        peak_idx = int(np.argmax(aif_curve))
    pk = float(aif_curve[peak_idx])
    if pk <= 0:
        return None
    f_post = peak_idx + max(4, int(np.ceil(30.0 / tr)))
    if f_post >= aif_curve.size:
        return None
    return float(aif_curve[f_post] / pk)


def resample_parker_template(
    frame_times_s: npt.NDArray[Any],
    dose_factor: float = 1.0,
) -> npt.NDArray[Any]:
    """Parker population AIF sampled on the acquisition frame grid.

    Wraps :func:`parker_population_aif` so the reference shape is defined at the
    *actual* frame times rather than a fixed grid. This keeps every downstream
    fidelity comparison invariant to temporal resolution and acquisition length.

    Parameters
    ----------
    frame_times_s : np.ndarray
        Frame times in seconds (monotonic, starting at 0).
    dose_factor : float, optional
        Molar-dose scaling passed through to :func:`parker_population_aif`.

    Returns
    -------
    np.ndarray
        Parker template concentration (mM) at ``frame_times_s``.
    """
    return parker_population_aif(np.asarray(frame_times_s, dtype=float), dose_factor)


def _two_sided_band_score(value: float, center: float, half_width: float) -> float:
    """Tent score in [0, 1]: 1 at ``center``, 0 at ``center +/- half_width``."""
    if half_width <= 0.0:
        return 1.0 if value == center else 0.0
    return float(max(0.0, 1.0 - abs(value - center) / half_width))


def _asymmetric_band_score(
    value: float, center: float, lower_half_width: float, upper_half_width: float,
) -> float:
    """Asymmetric tent in [0, 1]: 1 at ``center``, with independent widths.

    Used for washout, where deviation below the physiologic center is a strong
    contamination signal (strict, ``lower_half_width``) but deviation above is
    mostly benign long-acquisition drift (lenient, ``upper_half_width``).
    """
    hw = lower_half_width if value <= center else upper_half_width
    if hw <= 0.0:
        return 1.0 if value == center else 0.0
    return float(max(0.0, 1.0 - abs(value - center) / hw))


def score_averaged_aif_fidelity(
    aif_curve: npt.NDArray[Any],
    tr: float,
    baseline_frames: int = 5,
    dose_factor: float = 1.0,
    hematocrit: float = HEMATOCRIT_ARTERIAL,
) -> tuple[float, dict[str, float]]:
    """Physics-grounded fidelity score (0-1) for a *pooled/averaged* DCE AIF.

    Unlike the per-candidate scorer (whose mean over voxels never inspects the
    pooled curve), this evaluates the emitted, pooled AIF directly. It is a QC /
    regression-sentinel scalar: it rank-orders a faithful arterial AIF above a
    tail-collapsed (over-washed) or partial-volume-diluted one — the two failure
    modes that bias Ktrans. It is NOT a runtime accept/reject gate (real detected
    AIFs cluster low because they systematically under-recover the real-MCA late
    tail, a second-order effect for Ktrans); per-subject contamination is gated on
    the specific signatures (tail-ratio, washout floors, peak band, FWHM, causality).

    Caveat: the washout and tail-persistence terms use a fixed last-frames tail, so
    this QC scalar reads slightly higher on acquisitions longer than ~7 min (the tail
    is sampled deeper into renal clearance). Cosmetic here — this scalar never gates;
    the gating per-candidate scorer uses an asymmetric washout band to absorb the drift.

    Anchored to physiologic arterial targets, not to pmriprep's Parker template:
    the true DCE AIF is a real hematocrit-corrected middle-cerebral-artery curve
    (per the OSIPI DRO ground truth) with a heavier recirculation tail than
    idealized Parker, so a shape residual vs Parker would penalize the true AIF.

    Terms (each in [0, 1], then weighted):

    - **washout**: two-sided deviation from the physiologic arterial washout
      (:data:`AIF_WASHOUT_PHYSIOLOGIC_CENTER`); never "more is better".
    - **tail**: late-tail-mean / peak, two-sided around the physiologic persistence
      plateau — penalizes both a collapsed tail and a tissue-diluted (too-high) one.
    - **scale**: peak in the dose-scaled band + a one-sided AUC floor vs Parker
      (a normalized-only score would hide the amplitude error that biases
      Ktrans ~ 1 / AIF integral; a richer-than-Parker AUC is physiologic).
    - **width**: FWHM near :data:`AIF_IDEAL_FWHM_SECONDS` + a positive upslope (a
      broad venous curve fails the sharp arterial upstroke).

    Parameters
    ----------
    aif_curve : np.ndarray
        Emitted AIF concentration (mM), plasma-corrected as delivered.
    tr : float
        Temporal resolution in seconds.
    baseline_frames : int, optional
        Number of pre-contrast frames.
    dose_factor : float, optional
        Molar-dose factor for the template + peak band (1.0 = full dose).
    hematocrit : float, optional
        Arterial hematocrit used to convert the Parker whole-blood reference
        template and peak band to plasma concentration. The input curve is
        already plasma concentration and is not converted again.

    Returns
    -------
    tuple of (float, dict)
        The composite fidelity score and the per-term sub-scores/metrics.
    """
    if not 0.0 <= hematocrit < 1.0:
        raise ValueError(f"hematocrit must be in [0, 1), got {hematocrit}")

    curve = np.asarray(aif_curve, dtype=float)
    metrics: dict[str, float] = {}
    # Need >= baseline + 6 frames so the last-5-frame washout tail is meaningful,
    # a positive tr for the time-based windows, and a positive peak.
    if (
        tr <= 0.0
        or curve.size < baseline_frames + 6
        or float(np.max(curve)) <= 0.0
    ):
        return 0.0, {"fidelity_valid": 0.0}

    frame_times = np.arange(curve.size, dtype=float) * tr
    # Arrival-aligned Parker template, used only as a one-sided AUC floor (a real
    # MCA AIF — the OSIPI DRO ground truth is a hematocrit-corrected middle-
    # cerebral-artery curve, not a Parker form — carries a heavier recirculation
    # tail than idealized Parker, so a full post-peak *shape* residual vs Parker
    # would penalize the true richer AIF. Discrimination therefore comes from the
    # physiologic scalar terms below, not a shape-vs-Parker residual.)
    plasma_factor = 1.0 / (1.0 - hematocrit)
    native = resample_parker_template(frame_times, dose_factor) * plasma_factor
    t_shift = float(np.argmax(curve)) * tr - float(np.argmax(native)) * tr
    template = (
        resample_parker_template(frame_times - t_shift, dose_factor)
        * plasma_factor
    )

    # --- scale: absolute peak (two-sided band) + AUC (one-sided floor) ---
    peak = compute_dce_aif_peak_concentration(curve)
    peak_min = (
        DCE_AIF_EXPECTED_FULL_DOSE.peak_min_mm * dose_factor * plasma_factor
    )
    peak_max = (
        DCE_AIF_EXPECTED_FULL_DOSE.peak_max_mm * dose_factor * plasma_factor
    )
    peak_center = 0.5 * (peak_min + peak_max)
    peak_score = _two_sided_band_score(
        peak, peak_center, 0.5 * (peak_max - peak_min) + peak_center * 0.15
    )
    auc = compute_dce_aif_auc(curve, tr)
    template_auc = compute_dce_aif_auc(template, tr)
    auc_ratio = auc / template_auc if template_auc > 0 else 0.0
    # One-sided: a richer-than-Parker AUC is physiologic; only penalize a deficient
    # AUC (tail collapse / partial-volume dilution → too little tracer delivered).
    auc_score = 1.0 if auc_ratio >= 1.0 else max(0.0, 1.0 - (1.0 - auc_ratio) / 0.5)
    scale_score = 0.5 * peak_score + 0.5 * auc_score
    metrics["peak_mm"] = peak
    metrics["reference_hematocrit"] = hematocrit
    metrics["auc_ratio"] = auc_ratio
    metrics["scale_score"] = scale_score

    # --- width: FWHM band + positive upslope ---
    fwhm = compute_dce_aif_fwhm_seconds(curve, tr, baseline_frames)
    fwhm_score = _two_sided_band_score(
        fwhm, AIF_IDEAL_FWHM_SECONDS, AIF_FWHM_BAND_HALFWIDTH_SECONDS
    )
    peak_idx = int(np.argmax(curve))
    upslope = 0.0
    if peak_idx > baseline_frames:
        seg = curve[baseline_frames:peak_idx + 1]
        if seg.size > 1:
            upslope = float(np.max(np.diff(seg) / tr))
    upslope_score = 1.0 if upslope > 0.0 else 0.0
    width_score = 0.7 * fwhm_score + 0.3 * upslope_score
    metrics["fwhm_seconds"] = fwhm
    metrics["width_score"] = width_score

    # --- tail persistence: late-tail-mean / peak (two-sided physiologic band) ---
    # Peaked at the physiologic recirculation plateau; penalized both below the
    # collapse floor (tail-collapse / over-washout — V1, over-selected AIFs) AND
    # above it (a tissue-diluted partial-volume AIF does not wash out, so its tail
    # sits too high). Half-width = target - floor puts the lower zero at the floor.
    tail_frames = max(1, int(round(30.0 / tr)))
    tail_ratio = float(np.mean(curve[-tail_frames:])) / peak if peak > 0 else 0.0
    tail_score = _two_sided_band_score(
        tail_ratio,
        AIF_TAIL_PERSISTENCE_TARGET,
        AIF_TAIL_PERSISTENCE_TARGET - AIF_TAIL_PERSISTENCE_FLOOR,
    )
    metrics["tail_ratio"] = tail_ratio
    metrics["tail_score"] = tail_score

    # --- washout: two-sided deviation from the physiologic arterial center ---
    washout = compute_dce_aif_washout(curve, baseline_frames)
    washout_score = _two_sided_band_score(
        washout, AIF_WASHOUT_PHYSIOLOGIC_CENTER, AIF_WASHOUT_BAND_HALFWIDTH
    )
    metrics["washout"] = washout
    metrics["washout_score"] = washout_score

    # Washout and tail-persistence jointly carry the tail-collapse / partial-volume
    # discrimination; scale guards the amplitude (Ktrans ~ 1/AIF integral); width
    # rejects broad venous curves.
    fidelity = (
        0.30 * washout_score
        + 0.30 * tail_score
        + 0.25 * scale_score
        + 0.15 * width_score
    )
    metrics["fidelity"] = fidelity
    return float(fidelity), metrics


# =============================================================================
# DCE AIF acceptance gates (G1-G6 + G6 escape)
# =============================================================================


class GateQC(TypedDict):
    """Per-pool gate metrics computed inside the progressive trim loop."""

    peak_mm: float
    fwhm_seconds: float
    auc_mm_s: float
    washout: float
    multi_bump_count: int
    late_tail_per_peak: float | None
    ratio_30: float | None


GateStatus = Literal[
    "PASS",
    "PASS_WARN",
    "G6_ADMIT",
    "FAIL_G1",
    "FAIL_G2",
    "FAIL_G3",
    "FAIL_G4",
    "FAIL_G5",
    "FAIL_G6_a",
    "FAIL_G6_f",
    "FAIL_G6_g",
    "SKIP_G5",
]


class GateDecision(TypedDict):
    """Per-trim-iteration gate decision record (for QC provenance)."""

    tier: str
    n_pool: int
    scale_factor: float
    qc: GateQC
    status: GateStatus
    failing_gate: str | None
    warns: list[str]


def _apply_aif_scale(
    aif_curve: npt.NDArray[Any],
    hct: float,
    target_peak: float,
    expected_range_fn: Callable[[], tuple[float, float]],
) -> tuple[npt.NDArray[Any], float]:
    """Apply conservative upward AIF amplitude calibration.

    Mirrors the production scaling block but is callable per trim
    iteration. Preserves the peak-amplitude guard: when the detected peak is
    below ``MIN_MEANINGFUL_ENHANCEMENT_MM`` the input curve is returned
    unscaled with a scaling factor of 1.0.

    Parameters
    ----------
    aif_curve : np.ndarray
        Plasma-converted AIF curve (already divided by ``1 - Hct``).
    hct : float
        Arterial hematocrit (used to convert Parker blood-voxel target
        amplitudes to plasma units).
    target_peak : float
        User-supplied target peak (mM blood). When ``> 0``, explicitly
        calibrate to that target. When ``<= 0``, preserve measured peaks at
        or above the dose-adjusted expected lower bound and raise only
        sub-bound peaks to that lower bound.
    expected_range_fn : Callable
        Zero-arg callable returning ``(peak_min, peak_max)`` in blood
        units for the current dose / field strength.

    Returns
    -------
    (np.ndarray, float)
        ``(scaled_curve, scaling_factor)``. Given plasma-converted input the
        factor is a pure blood-amplitude ratio (target / detected peak); the
        ``1 / (1 - Hct)`` on both cancels, so it is Hct-invariant.
    """
    current_peak = float(np.max(aif_curve))
    if current_peak <= MIN_MEANINGFUL_ENHANCEMENT_MM:
        return aif_curve, 1.0
    plasma_conv = 1.0 / (1.0 - hct)
    if target_peak > 0:
        target = target_peak * plasma_conv
    else:
        peak_min, _peak_max = expected_range_fn()
        target = peak_min * plasma_conv
        if current_peak >= target:
            return aif_curve, 1.0
    scaling_factor = target / current_peak
    return aif_curve * scaling_factor, float(scaling_factor)


@dataclass(frozen=True)
class AIFScaleApplication:
    """Resolved detected-AIF amplitude calibration and QC factors."""

    curve: npt.NDArray[Any]
    applied_factor: float
    required_factor: float
    mode: Literal["upward_floor", "none", "explicit_target"]


def _resolve_aif_scale(
    aif_curve: npt.NDArray[Any],
    hct: float,
    target_peak: float,
    expected_range_fn: Callable[[], tuple[float, float]],
    *,
    apply_peak_floor: bool,
) -> AIFScaleApplication:
    """Separate amplitude-adequacy assessment from amplitude mutation."""
    calibrated, required_factor = _apply_aif_scale(
        aif_curve, hct, target_peak, expected_range_fn
    )
    if target_peak > 0:
        return AIFScaleApplication(
            calibrated, required_factor, required_factor, "explicit_target"
        )
    if apply_peak_floor:
        return AIFScaleApplication(
            calibrated, required_factor, required_factor, "upward_floor"
        )
    return AIFScaleApplication(aif_curve, 1.0, required_factor, "none")


def _compute_pool_qc(
    scaled_curve: npt.NDArray[Any],
    tr: float,
    baseline_frames: int,
) -> GateQC:
    """Run all QC functions a single time on a (scaled) AIF curve.

    Wraps ``compute_dce_aif_*`` helpers + ``compute_dce_aif_ratio_30s``
    + ``detect_dce_aif_multi_bump_tail`` (using the new shoulder=60 s
    and fraction=0.40 constants) into a flat ``GateQC`` dict. Pure;
    safe to call inside a progressive trim loop.
    """
    n_volumes = int(scaled_curve.size)
    peak_idx = int(np.argmax(scaled_curve))
    peak_val = float(scaled_curve[peak_idx])

    fwhm_s = compute_dce_aif_fwhm_seconds(scaled_curve, tr, baseline_frames)
    auc = compute_dce_aif_auc(scaled_curve, tr)
    washout = compute_dce_aif_washout(scaled_curve, baseline_frames)
    bump_count, _ = detect_dce_aif_multi_bump_tail(
        scaled_curve, tr=tr, baseline_frames=baseline_frames,
    )
    r30 = compute_dce_aif_ratio_30s(scaled_curve, tr, peak_idx=peak_idx)

    if tr > 0 and peak_val > 0:
        # Window is "2 min POST-PEAK" (the 0.35 threshold's basis), not from scan
        # start — else a delayed bolus counts first-pass as tail.
        idx_120 = peak_idx + int(120.0 / tr)
        if idx_120 < n_volumes:
            late_tail = float(scaled_curve[idx_120:].mean() / peak_val)
        else:
            late_tail = None
    else:
        late_tail = None

    return GateQC(
        peak_mm=peak_val,
        fwhm_seconds=float(fwhm_s),
        auc_mm_s=float(auc),
        washout=float(washout),
        multi_bump_count=int(bump_count),
        late_tail_per_peak=late_tail,
        ratio_30=r30,
    )


def _evaluate_gates(
    scale_factor: float,
    qc: GateQC,
    atlas_tier: str | None,
    n_voxels: int,
    hemisphere: str,
    n_slabs_spanned: int,
) -> tuple[GateStatus, GateDecision]:
    """Evaluate the 7 acceptance gates against a candidate pool.

    G2/G3/G4/G5' HARD failures take precedence over G1; if any of those
    fire, the status is ``FAIL_G{N}`` and G6 is NOT consulted (G6
    escapes amplitude failures only). When G1 is the sole HARD failure,
    G6 sub-criteria (a)/(f)/(g) are checked: if all hold, status is
    ``G6_ADMIT``; otherwise the failing G6 sub-criterion is reported.

    G5 is the only WARN-tier gate currently; the design supports future
    WARN gates (aggregate-warn check reserved for when a second is added).

    Parameters
    ----------
    scale_factor : float
        Output of ``_apply_aif_scale`` for the current pool.
    qc : GateQC
        Output of ``_compute_pool_qc``.
    atlas_tier : str or None
        Tier label (e.g. ``"MCA_ACA_dce_strict"`` / ``"..._relaxed"`` /
        ``"..._permissive"``). Used only by G6 (f).
    n_voxels : int
        Pool size (used by G6 (g)).
    hemisphere : str
        ``"bilateral"`` / ``"unilateral_left"`` / ``"unilateral_right"``.
    n_slabs_spanned : int
        Distinct z-slice count in the pool (used by G6 (g) unilateral).

    Returns
    -------
    (GateStatus, GateDecision)
        Decision record carrying status + accumulated WARNs + failing
        gate (if any).
    """
    warns: list[str] = []

    g2_fail = (
        qc["late_tail_per_peak"] is not None
        and qc["late_tail_per_peak"] > DCE_AIF_GATE_LATE_TAIL_RATIO_MAX
    )
    g3_fail = qc["fwhm_seconds"] > DCE_AIF_GATE_FWHM_MAX_SECONDS
    g4_fail = qc["multi_bump_count"] > DCE_AIF_GATE_MULTIBUMP_MAX

    r30 = qc["ratio_30"]
    if r30 is None:
        g5_state: Literal["PASS", "WARN", "HARD", "SKIP"] = "SKIP"
    elif r30 > DCE_AIF_GATE_RATIO30_HARD:
        g5_state = "HARD"
    elif r30 > DCE_AIF_GATE_RATIO30_WARN:
        g5_state = "WARN"
        warns.append("G5")
    else:
        g5_state = "PASS"

    g1_fail = scale_factor > DCE_AIF_GATE_SCALE_HARD

    def _record(status: GateStatus, failing: str | None) -> tuple[GateStatus, GateDecision]:
        return status, GateDecision(
            tier=atlas_tier or "",
            n_pool=n_voxels,
            scale_factor=float(scale_factor),
            qc=qc,
            status=status,
            failing_gate=failing,
            warns=warns,
        )

    if g2_fail:
        return _record("FAIL_G2", "G2")
    if g3_fail:
        return _record("FAIL_G3", "G3")
    if g4_fail:
        return _record("FAIL_G4", "G4")
    if g5_state == "HARD":
        return _record("FAIL_G5", "G5")

    # C5: a sub-threshold peak is no real bolus — a HARD amplitude reject that the
    # G6 escape (which rescues dilution/over-scale, not a ~0-peak AIF) must NOT
    # rescue. Placed with the other HARD gates, before the G6 clause. This is
    # caught gate-side because _apply_aif_scale returns scale=1.0 for such a peak,
    # which would otherwise pass the scale-based G1.
    if qc["peak_mm"] < MIN_MEANINGFUL_ENHANCEMENT_MM:
        return _record("FAIL_G1", "G1")

    if not g1_fail:
        if warns:
            return _record("PASS_WARN", None)
        return _record("PASS", None)

    # G1 fails — consult G6 escape clause.
    g6_a = scale_factor <= DCE_AIF_GATE_SCALE_G6_CEILING
    g6_f = atlas_tier in (
        "MCA_ACA_dce_strict",
        "MCA_ACA_dce_relaxed",
        "MCA_ACA_dce_permissive",
    )
    if hemisphere == "bilateral":
        g6_g = n_voxels >= DCE_AIF_GATE_G6_MIN_VOXELS_BILATERAL
    else:
        g6_g = (
            n_voxels >= DCE_AIF_GATE_G6_MIN_VOXELS_UNILATERAL
            and n_slabs_spanned >= DCE_AIF_GATE_G6_MIN_SLABS_UNILATERAL
        )

    if g6_a and g6_f and g6_g:
        warns_with_g6 = list(warns) + ["G1_via_G6"]
        return "G6_ADMIT", GateDecision(
            tier=atlas_tier or "",
            n_pool=n_voxels,
            scale_factor=float(scale_factor),
            qc=qc,
            status="G6_ADMIT",
            failing_gate=None,
            warns=warns_with_g6,
        )

    if not g6_a:
        return _record("FAIL_G6_a", "G6_a")
    if not g6_f:
        return _record("FAIL_G6_f", "G6_f")
    return _record("FAIL_G6_g", "G6_g")


def _select_worst_for_trim(
    metrics_list: list[dict[str, float]],
    scores: list[float],
    failing_gate: str,
) -> int:
    """Composite G-aware "worst candidate" selector for progressive trim.

    Returns the index into ``metrics_list`` of the candidate to drop.
    Branches are aligned with the failing gate so that the trim
    direction physically helps the failing check on the next iteration.
    Tie-breaking on argmin / argmax: lowest index (deterministic).

    Branches:

      ``G1`` / ``G6_a``  -> argmin(``peak_mm``)         most diluted
      ``G2`` / ``G5``    -> argmin(``washout``)         most venous-like
      ``G3``             -> argmax(``fwhm_seconds``)    widest bolus
      ``G4``             -> argmin(``washout``)         bumps correlate
                                                          with low washout
      fallback           -> argmin(``scores``)          composite quality
    """
    if not metrics_list:
        return 0
    failing_root = failing_gate or "fallback"
    if failing_root in ("G1", "G6_a"):
        return int(np.argmin([m.get("peak_mm", 0.0) for m in metrics_list]))
    if failing_root in ("G2", "G5"):
        return int(np.argmin([m.get("washout", 0.0) for m in metrics_list]))
    if failing_root == "G3":
        return int(np.argmax([m.get("fwhm_seconds", 0.0) for m in metrics_list]))
    if failing_root == "G4":
        return int(np.argmin([m.get("washout", 0.0) for m in metrics_list]))
    # Fallback — drop lowest composite score.
    return int(np.argmin(scores))


def _count_slabs_spanned(candidates: list[tuple[int, int, int]]) -> int:
    """Count unique z-slice indices in a candidate ``(i, j, k)`` list.

    Used by G6 (g) unilateral path: ``n>=10 AND n_slabs_spanned>=2``.
    """
    if not candidates:
        return 0
    return len({int(k) for (_i, _j, k) in candidates})


def _classify_hemisphere(
    candidates: list[tuple[int, int, int]],
    affine: npt.NDArray[Any],
) -> tuple[str, int, int]:
    """Classify a candidate pool's hemisphere spread via RAS x-coordinate.

    A candidate's world x-coordinate ``< 0`` (RAS convention) is left.

    Parameters
    ----------
    candidates : list of (int, int, int)
        Voxel ``(i, j, k)`` coordinates.
    affine : np.ndarray
        4x4 voxel-to-world affine.

    Returns
    -------
    (str, int, int)
        ``(hemisphere, n_left, n_right)`` where hemisphere is
        ``"bilateral"`` / ``"unilateral_left"`` / ``"unilateral_right"``.
    """
    n_left = 0
    n_right = 0
    for cx, cy, cz in candidates:
        world = affine[:3, :3] @ np.array([cx, cy, cz]) + affine[:3, 3]
        if world[0] < 0:
            n_left += 1
        else:
            n_right += 1
    if n_left > 0 and n_right > 0:
        hemisphere = "bilateral"
    elif n_left > 0:
        hemisphere = "unilateral_left"
    else:
        hemisphere = "unilateral_right"
    return hemisphere, n_left, n_right


@dataclass
class TierResult:
    """Outcome of a single tier's progressive-trim acceptance attempt.

    Returned by ``_evaluate_tier_with_trim`` on accept; ``None`` on
    cascade fall-through.
    """

    candidates: list[tuple[int, int, int]]
    scores: list[float]
    metrics: list[dict[str, float]]
    curve_scaled: npt.NDArray[Any]
    scale_factor: float
    required_scale_factor: float
    qc: GateQC
    atlas_tier: str
    hemisphere: str
    n_slabs: int
    status: GateStatus
    gate_decisions: list[GateDecision] = field(default_factory=list)


@dataclass(frozen=True)
class PoolGateOutcome:
    """Scored + gated result of averaging a single candidate pool.

    Returned by ``_score_and_gate_pool`` (the shared "average ->
    plasma-convert -> rescale -> QC -> hemisphere/slabs -> gate" body used
    by both the progressive-trim loop and the C2 re-gate). ``curve_scaled``
    is the exact curve the shape gates evaluated, so callers emit it directly.
    ``required_scale_factor`` remains the G1/G6 amplitude-adequacy input even
    when automatic peak-floor mutation is disabled.
    """

    curve_scaled: npt.NDArray[Any]
    scale_factor: float
    required_scale_factor: float
    qc: GateQC
    hemisphere: str
    n_slabs: int
    status: GateStatus
    decision: GateDecision


@dataclass(frozen=True)
class TierEvaluation:
    """Outcome of one tier's gate-and-trim evaluation.

    Carries both the accepted ``TierResult`` (when one is found) and the
    full per-iteration ``gate_decisions`` log so that failing tiers
    contribute their diagnostics to the QC sidecar even though their
    pool was discarded.
    """

    result: TierResult | None
    gate_decisions: list[GateDecision] = field(default_factory=list)


@dataclass(frozen=True)
class CascadeResult:
    """Aggregated result of ``_run_two_pass_cascade``.

    Replaces the prior positional-tuple return so adding new fields
    (e.g. per-tier gate-decision logs in this revision) doesn't shift
    consumers' positional unpacking.
    """

    candidates: list[tuple[int, int, int]]
    scores: list[float]
    metrics_list: list[dict[str, float]]
    detection_method: str
    atlas_tier_used: str | None
    cascade_tiers_attempted: int
    anchor_outcomes: list[dict[str, Any]]
    tier_result: TierResult | None
    tier_gate_decisions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LoadedInputs:
    """Loaded 4D concentration + geometry + brain mask for one AIF run."""

    conc_data: npt.NDArray[Any]
    affine: npt.NDArray[Any]
    header: Any
    brain_mask: npt.NDArray[Any]


@dataclass(frozen=True)
class AIFParams:
    """Scalar detection parameters + derived geometry, read once per run."""

    aif_method: str
    n_candidates: int
    baseline_frames: int
    field_strength: float
    tr: float
    target_peak: float
    lesion_distance: float
    n_timepoints: int
    blood_t1: float
    voxel_sizes: tuple[float, ...]


@dataclass(frozen=True)
class DetectionState:
    """Outcome of the detection phase (or its population-fallback defaults).

    Bundles the ~11 locals the detection cascade + post-cascade filters produce
    and that the emit / finalize phases consume. ``population_default()``
    reproduces the pre-detection init block for the population path.
    """

    detection_method: Literal["atlas", "whole_brain", "population"]
    candidates: list[tuple[int, int, int]]
    scores: list[float]
    metrics_list: list[dict[str, float]]
    tissue_ttp_frame: float | None
    detection_attempt: dict[str, Any] | None
    atlas_tier_used: str | None
    anchor_outcomes: list[dict[str, Any]]
    tier_gate_decisions: list[dict[str, Any]]
    tier_result: TierResult | None
    n_filtered_out: int

    @classmethod
    def population_default(cls) -> "DetectionState":
        """Init values for the population path (no detection run)."""
        return cls(
            detection_method="population", candidates=[], scores=[],
            metrics_list=[], tissue_ttp_frame=None, detection_attempt=None,
            atlas_tier_used=None, anchor_outcomes=[], tier_gate_decisions=[],
            tier_result=None, n_filtered_out=0,
        )


@dataclass(frozen=True)
class EmitResult:
    """The emitted AIF curve + mask + provenance, shared by both emit branches.

    ``tier_result``/``detection_method`` ride here (not on ``DetectionState``)
    because the emit branches rebuild/override them: the C2 re-gate rebuilds
    ``tier_result``, and the population emit branch reassigns
    ``detection_method`` to ``"population"`` when the detected pool was emptied.
    """

    aif_curve: npt.NDArray[Any]
    aif_mask: npt.NDArray[Any]
    quality_score: float
    scaling_factor: float
    required_scaling_factor: float
    calibration_mode: str
    qc_metrics: dict[str, Any]
    tier_result: TierResult | None
    detection_method: Literal["atlas", "whole_brain", "population"]


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
    atlas_mask_files = traits.List(
        traits.Str,
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
    # Either(File(exists=True), Enum("")) accepts a real path OR the literal
    # empty string. Empty string is the documented Function-node-returns-empty
    # convention used by the upstream _transform_lesion_mask_to_dce helper
    # when no whole-tumor mask is available for a subject. The runtime guard
    # in _run_interface handles the empty-string case (it's falsy). Analogous
    # to the Either(File, Enum(None)) sentinel idiom in qc_writers (None vs
    # "" sentinel reflects the upstream Function-node return contract).
    lesion_mask_file = traits.Either(
        File(exists=True),
        traits.Enum(""),
        mandatory=False,
        desc=(
            "Optional 3D lesion mask for exclusion. Accepts the literal "
            "empty string when upstream Function nodes have no mask to "
            "provide (per the workflow Function-node-returns-empty-string "
            "convention used by _transform_lesion_mask_to_dce)."
        ),
    )

    # Detection parameters
    aif_method = traits.Enum(
        "auto",
        "population",
        usedefault=True,
        desc="AIF detection method: 'auto' (multi-level) or 'population' (Parker)",
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
    target_peak = traits.Float(
        0.0,
        usedefault=True,
        desc=(
            "Explicit target peak concentration (mM) for AIF calibration, in blood-voxel "
            "units (Parker 2006 convention, i.e. ΔR1/r1 pre-Hct-correction). "
            "The pipeline divides by (1-Hct) internally to compare against "
            "the plasma-scale aif_curve. Values >0 override aif_peak_floor; "
            "0 follows the configured automatic peak-floor policy."
        ),
    )
    aif_peak_floor = traits.Bool(
        True,
        usedefault=True,
        desc=(
            "Raise detected AIF peaks below the dose-adjusted expected lower "
            "bound. False preserves measured amplitude while retaining the "
            "hypothetical correction factor for amplitude QC. Explicit "
            "target_peak calibration takes precedence."
        ),
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
    mr_acquisition_type = traits.Str(
        "",
        usedefault=True,
        desc=(
            "BIDS MRAcquisitionType field ('3D' or '2D'). QC metadata "
            "only — surfaced in the AIF sidecar and informational logs "
            "for downstream review. Inferior inflow exclusion is "
            "applied universally and does not depend on this value "
            "(BIDS metadata cannot reliably distinguish slab-selective "
            "2D-prep sequences from plain 2D-MS)."
        ),
    )
    dose = traits.Float(
        1.0,
        usedefault=True,
        desc="Contrast agent dose multiplier (0.5=half, 1.0=full, 2.0=double)",
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
        desc="Detection method used: 'atlas', 'whole_brain', 'population'"
    )
    quality_score = traits.Float(desc="Overall AIF quality score (0-1)")
    peak_concentration = traits.Float(desc="Peak AIF concentration in mM")
    fwhm_seconds = traits.Float(desc="FWHM of AIF peak in seconds")
    auc = traits.Float(desc="Area under the AIF curve (mM·s)")
    blood_t1 = traits.Float(desc="Blood T1 value used for this field strength (s)")
    scaling_factor = traits.Float(desc="Scaling factor applied to AIF (1.0 if unscaled)")
    candidate_count = traits.Int(desc="Number of candidate voxels used")
    qc_metrics = traits.Dict(desc="Detailed QC metrics dictionary")
    aif_confidence = traits.Str(
        "standard",
        usedefault=True,
        desc="AIF acceptance-gate confidence: 'standard' or 'reduced' "
             "(G6-admitted PV-diluted but anatomically correct).",
    )
    aif_uncertainty_inflation_factor = traits.Float(
        1.0,
        usedefault=True,
        desc="Base multiplier on PK posterior sigma from upstream AIF "
             "confidence (1.0 = standard, 2.0 = G6-admitted low-scale, "
             "3.0 = G6-admitted high-scale or population AIF). Mapped "
             "per-parameter downstream via DCE_AIF_G6_SIGMA_INFLATION "
             "(ve uninflated).",
    )


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
    atlas_mask_files : list of str, optional
        Arterial atlas masks for constrained search, ordered from tightest to
        broadest territory.
    lesion_mask_file : str, optional
        3D lesion mask for excluding potentially abnormal regions.
    aif_method : str
        'auto' for multi-level detection, 'population' for Parker AIF.
    n_candidates : int
        Number of top candidate voxels to average for final AIF.
    aif_peak_floor : bool
        Apply the automatic dose-adjusted lower peak floor to detected AIFs.
        Amplitude adequacy remains active when disabled.
    target_peak : float
        Explicit calibration target; 0 follows ``aif_peak_floor``.
    baseline_frames : int
        Number of pre-contrast frames for baseline estimation.
    field_strength : float
        Magnetic field strength in Tesla.
    tr : float
        Temporal resolution in seconds.
    dose : float
        Contrast dose multiplier (0.5=half, 1.0=full, 2.0=double).

    Returns
    -------
    out_aif_curve : str
        Path to AIF curve text file.
    out_aif_mask : str
        Path to AIF voxel mask NIfTI.
    aif_curve : list
        AIF concentration values in mM.
    detection_method : str
        Method used: 'atlas', 'whole_brain', or 'population'.
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

    The Parker population AIF is used as fallback and for scaling
    reference. It represents a well-characterized arterial input for
    standard contrast dose.

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

        Examples
        --------
        - Standard agent, full dose:  (0.1 * 1.0) / 0.1 = 1.0
        - Standard agent, half dose:  (0.1 * 0.5) / 0.1 = 0.5
        - Vueway, half dose:          (0.05 * 0.5) / 0.1 = 0.25
        - Vueway, full dose:          (0.05 * 1.0) / 0.1 = 0.5
        """
        std_dose = float(self.inputs.agent_standard_dose)
        multiplier = float(self.inputs.dose)
        return (std_dose * multiplier) / 0.1

    def _get_expected_peak_range(self) -> tuple[float, float]:
        """Get expected peak concentration range based on dose.

        Returns values in **blood-voxel units** (raw Parker C_p, pre-Hct
        division) — ``DCE_AIF_EXPECTED_FULL_DOSE`` is on the same ~6 mM scale as
        the raw ``parker_population_aif`` output. Per-voxel candidate gates
        compare per-voxel ``peak`` (also blood, pre-Hct division) against this
        range — blood-vs-blood, consistent. Consumers that compare against a
        plasma (``/(1-Hct)``-divided) curve must convert this range by
        ``1/(1-Hct)`` first, as the scaling sites and the ``final_peak`` QC
        warning do.
        """
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

        Parameters
        ----------
        curves : np.ndarray
            2D array (n_voxels, n_timepoints) of concentration curves.
        baseline_frames : int, optional
            Number of pre-contrast baseline frames.
        tr : float, optional
            Temporal resolution in seconds.

        Returns
        -------
        dict
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
        safe_denom = np.where(np.abs(denom) > MIN_MEANINGFUL_ENHANCEMENT_MM, denom, 1.0)
        washout_fraction = np.clip(
            (peak_values - tail_mean) / safe_denom, 0.0, 1.0
        )
        washout_fraction = np.where(np.abs(denom) > MIN_MEANINGFUL_ENHANCEMENT_MM, washout_fraction, 0.0)

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
        tissue_ttp_frame: float | None = None,
    ) -> tuple[npt.NDArray[Any], float, int]:
        """Apply arterial criteria with 7-tier fallback cascade.

        TR-adaptive: at low temporal resolution (TR > 5s), timing-based
        criteria are replaced with early_arrival (40% fraction) which is
        TR-independent. FWHM thresholds are adapted to seconds-based limits.

        Parameters
        ----------
        chars : dict
            Signal characteristics from ``_compute_signal_characteristics``.
        spatial_exclude : ndarray
            Boolean array of voxels to exclude (midline, superior, etc).
        nt : int
            Number of timepoints.
        percentile_threshold : float
            Percentile for peak threshold (default 98).
        n_candidates : int
            Target number of candidates.
        tr : float
            Temporal resolution in seconds (for TR-adaptive thresholds).
        tissue_ttp_frame : float, optional
            Brain-wide tissue TTP frame from
            ``_calculate_tissue_ttp_reference``.  When provided, overrides
            the local IQR-of-moderate-enhancement reference for the
            ``arterial_timing`` / ``arterial_timing_relaxed`` masks AND
            for the returned ``tissue_ttp`` value.  When ``None``, falls
            back to the local computation (preserves whole-brain
            semantics on callers that don't have a brain-wide reference
            available).

        Returns
        -------
        tuple
            ``(candidate_indices, tissue_ttp, tier_used)``:
            - ``candidate_indices``: array of indices passing criteria
            - ``tissue_ttp``: tissue TTP frame (brain-wide when
              ``tissue_ttp_frame`` was supplied, else local)
            - ``tier_used``: which tier (1-7) produced the result

        Notes
        -----
        Atlas-constrained callers must pass ``tissue_ttp_frame`` because
        the local IQR is computed inside the atlas mask, where the
        moderate-enhancement subset is dominated by arterial structures
        and collapses onto the arterial peak time itself — falsely
        flagging arterial voxels as venous downstream.
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

        # 2. Tissue reference timing.  Prefer the brain-wide value passed
        # by the cascade; otherwise compute the IQR-of-moderate-enhancement
        # median locally.  Atlas-constrained callers must pass the
        # brain-wide value: the local IQR inside an atlas mask is
        # arterial-dominated and collapses onto the arterial peak time.
        enh_values = chars["enhancement"]
        if tissue_ttp_frame is not None:
            tissue_ttp = float(tissue_ttp_frame)
        else:
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

        Parameters
        ----------
        conc_data : np.ndarray
            4D concentration data (x, y, z, t).
        brain_mask : np.ndarray
            3D brain mask.
        baseline_frames : int
            Number of pre-contrast baseline frames.

        Returns
        -------
        float
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
        slab_edge_inferior_mm: float = SLAB_EDGE_EXCLUSION_MM_ROLLOFF,
        slab_edge_superior_mm: float = SSS_EXCLUSION_MM,
        erosion_iterations: int = BRAIN_EROSION_ITERATIONS,
        physical_erosion_mm: float = 0.0,
    ) -> npt.NDArray[Any]:
        """Prepare the search mask with erosion, slab exclusion, and lesion distance.

        Steps (in order):

        1. Binary erosion of brain mask (``erosion_iterations``).
        2. Directional slab-edge exclusion: separate inferior / superior
           margins (inferior usually larger to remove 3D-slab entry-slice
           inflow contamination).
        3. Midline exclusion (skipped when ``skip_midline=True``).
        4. Top-25% superior cut (``SUPERIOR_EXCLUSION_FRACTION``).
        5. Lesion exclusion (binary + distance-based).
        6. Physical-distance erosion in mm (``physical_erosion_mm``),
           applied last so its mm-from-edge measurement uses the
           ORIGINAL ``brain_mask``, not the post-step-1..5 search mask.

        Atlas vs WB asymmetry: atlas callers rely on the MRA-derived
        anatomical specificity of the atlas mask itself, so they use a
        small ``BRAIN_EROSION_ITERATIONS_ATLAS=2`` + ``SLAB_EDGE_
        EXCLUSION_MM_ROLLOFF=4 mm`` rolloff edge and ``physical_erosion_mm
        =0``. Whole-brain fallback callers have no anatomical prior and
        need a depth-protective margin, so they layer ``BRAIN_EROSION_
        ITERATIONS=8`` + ``WB_FALLBACK_DEPTH_EROSION_MM=10 mm`` physical
        depth on top.

        Parameters
        ----------
        brain_mask : ndarray
            3D brain mask.
        voxel_sizes : tuple of float
            Voxel dimensions ``(dx, dy, dz)`` in mm.
        lesion_mask : ndarray, optional
            3D lesion mask.
        lesion_distance_mm : float
            Minimum distance (mm) from lesion.
        skip_midline : bool
            Skip midline exclusion for atlas-constrained search.
        slab_edge_inferior_mm : float
            Distance (mm) from the INFERIOR slab edge (low-z under RAS+;
            pipeline enforces RAS+ reorientation upstream). Callers
            should pass ``_inflow_exclusion_mm(...)`` to remove entry-
            slice inflow contamination. Default
            ``SLAB_EDGE_EXCLUSION_MM_ROLLOFF`` is rolloff-only; production
            call sites override.
        slab_edge_superior_mm : float
            Distance (mm) from the SUPERIOR slab edge. Whole-brain
            callers should pass ``SSS_EXCLUSION_MM`` (default) to keep
            the superior sagittal sinus out of candidates; atlas-
            constrained callers should pass
            ``SLAB_EDGE_EXCLUSION_MM_ROLLOFF`` since the atlas already
            excludes SSS by anatomy. Step 4 below applies an additional
            ``SUPERIOR_EXCLUSION_FRACTION`` (top 25%) cut on top of this.
        erosion_iterations : int
            Number of binary erosion iterations. Use
            ``BRAIN_EROSION_ITERATIONS_ATLAS`` for atlas-constrained
            search.
        physical_erosion_mm : float
            Additional physical-distance erosion in mm, applied as Step 6
            on top of the slab/midline/top-25%/lesion cuts. Used by the
            whole-brain fallback caller to strip the lateral cortical
            surface where cortical-vein drainage produces arterial-shaped
            curves at clinical TR. Atlas callers leave at
            ``0.0`` (preserves full atlas territory). Computed via
            ``distance_transform_edt(brain_mask, sampling=voxel_sizes)``
            so anisotropic voxels are handled in physical mm. If the
            depth cut would leave fewer than ``BRAIN_EROSION_MIN_VOXELS``
            voxels (thin sagittal slab / pediatric / atrophic case), the
            cut is skipped and a warning is logged.

        Returns
        -------
        ndarray
            Prepared 3D search mask.
        """
        from scipy.ndimage import binary_erosion

        # Step 1: Erode brain mask
        search_mask = binary_erosion(
            brain_mask > 0.5, iterations=erosion_iterations
        ).astype(brain_mask.dtype)
        if np.sum(search_mask) < BRAIN_EROSION_MIN_VOXELS:
            search_mask = (brain_mask > 0.5).astype(brain_mask.dtype)

        # Step 2: Directional slab-edge exclusion under RAS+.
        # Inferior side uses the larger inflow exclusion; superior side
        # uses the small rolloff exclusion. The superior cut is typically
        # subsumed by Step 4 (top-25% exclusion); the parameter is exposed
        # for completeness and for thick-slab cases.
        #
        # Axial-slab proxy guard: through-plane (z) thickness must be
        # greater than in-plane voxel size for the inferior-as-inflow
        # assumption to hold. Sagittal/coronal slabs reorient the inflow
        # direction; in those cases we fall back to symmetric exclusion
        # using the smaller of the two values to avoid over-rejecting
        # valid candidates.
        nz = search_mask.shape[2]
        slice_thickness = float(voxel_sizes[2]) if len(voxel_sizes) > 2 else 1.0
        in_plane_max = max(
            float(voxel_sizes[0]) if len(voxel_sizes) > 0 else 1.0,
            float(voxel_sizes[1]) if len(voxel_sizes) > 1 else 1.0,
        )
        # Use >= so isotropic 3D-VIBE (1×1×1mm) is treated as axial under
        # RAS+: only sagittal/coronal acquisitions have in-plane voxels
        # strictly larger than through-plane after RAS+ reorientation.
        is_axial_slab = slice_thickness >= in_plane_max
        effective_inferior_mm = slab_edge_inferior_mm
        effective_superior_mm = slab_edge_superior_mm
        if not is_axial_slab:
            fallback_mm = min(slab_edge_inferior_mm, slab_edge_superior_mm)
            effective_inferior_mm = fallback_mm
            effective_superior_mm = fallback_mm
            logger.info(
                "Non-axial DCE acquisition (slice_thickness=%.2f mm < "
                "in-plane=%.2f mm); falling back to symmetric slab-edge "
                "exclusion using min=%.2f mm.",
                slice_thickness, in_plane_max, fallback_mm,
            )
        if slice_thickness > 0:
            inferior_slices = max(1, int(np.ceil(effective_inferior_mm / slice_thickness)))
            superior_slices = max(1, int(np.ceil(effective_superior_mm / slice_thickness)))
            if inferior_slices + superior_slices < nz:
                search_mask[:, :, :inferior_slices] = 0   # inferior (low z under RAS+)
                search_mask[:, :, -superior_slices:] = 0  # superior (high z under RAS+)

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

        # Step 6: Physical-distance erosion (mm). Removes the lateral
        # cortical surface where cortical-vein drainage produces
        # arterial-shaped curves at clinical TR. Distance is measured
        # against the ORIGINAL brain mask, in physical mm, so anisotropic
        # voxels are handled correctly. Falls back to the pre-erosion
        # mask if the cut would evacuate the search space (thin sagittal
        # slab / pediatric / atrophic acquisitions).
        if physical_erosion_mm > 0:
            from scipy.ndimage import distance_transform_edt

            depth_mm = distance_transform_edt(
                brain_mask > 0.5, sampling=voxel_sizes[:3]
            )
            candidate_mask = (search_mask > 0.5) & (depth_mm >= physical_erosion_mm)
            n_kept = int(np.sum(candidate_mask))
            if n_kept >= BRAIN_EROSION_MIN_VOXELS:
                search_mask = candidate_mask.astype(brain_mask.dtype)
            else:
                logger.warning(
                    "Physical erosion of %.1f mm leaves %d voxels "
                    "(<%d minimum); falling back to pre-erosion mask.",
                    physical_erosion_mm,
                    n_kept,
                    BRAIN_EROSION_MIN_VOXELS,
                )

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

        Parameters
        ----------
        lesion_mask : np.ndarray
            3D lesion mask.
        affine : np.ndarray
            4x4 NIfTI affine matrix (voxel→RAS world).

        Returns
        -------
        str or None
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

        Parameters
        ----------
        candidates : list
            Candidate voxel coordinates.
        scores : list
            Candidate scores (higher is better).
        metrics_list : list
            Candidate metrics.
        brain_shape : tuple
            Spatial dimensions.
        affine : np.ndarray
            Affine matrix.
        tumor_laterality : str or None
            'left', 'right', 'bilateral', or None.

        Returns
        -------
        tuple of (list, list, list)
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

        Parameters
        ----------
        candidates : list
            Candidate voxel coordinates.
        scores : list
            Candidate scores (higher is better).
        metrics_list : list
            Candidate metrics.
        affine : np.ndarray
            4x4 affine matrix (voxel -> RAS world).
        n_select : int
            Number of candidates to select.

        Returns
        -------
        tuple of (list, list, list)
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

        Parameters
        ----------
        candidates : list
            Candidate voxel coordinates.
        scores : list
            Candidate scores.
        metrics_list : list
            Candidate metrics (must contain is_likely_venous flag).
        min_non_venous : int, optional
            Minimum non-venous candidates required to apply filter.

        Returns
        -------
        tuple of (list, list, list)
            Filtered (candidates, scores, metrics) with venous candidates
            removed. If at least one non-venous candidate survives, drops
            only the venous-flagged ones (preserving the survivors). If
            ALL candidates are flagged venous, returns ``([], [], [])`` so
            the cascade treats the tier as failed (venous-derived AIFs
            produce quantitatively biased PK estimates).
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

        if non_venous_indices:
            # At least one non-venous survives — drop venous, keep survivors.
            # Cascade Pass 1/Pass 2 contract handles count from there.
            if len(non_venous_indices) >= min_non_venous:
                logger.info(
                    "Filtering %d venous candidates (keeping %d arterial)",
                    n_venous, len(non_venous_indices),
                )
            else:
                logger.warning(
                    "Filtering %d venous candidates (keeping %d arterial; "
                    "below min_non_venous=%d, cascade Pass 2 will handle).",
                    n_venous, len(non_venous_indices), min_non_venous,
                )
            return (
                [candidates[i] for i in non_venous_indices],
                [scores[i] for i in non_venous_indices],
                [metrics_list[i] for i in non_venous_indices],
            )

        # All flagged venous — fail the tier. The cascade will descend or
        # fall to Parker (venous-derived AIFs produce quantitatively
        # biased PK estimates).
        logger.warning(
            "Tier rejected: %d/%d candidates flagged as likely venous "
            "(no non-venous alternative). Cascade will descend.",
            n_venous, len(candidates),
        )
        return [], [], []

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

        Parameters
        ----------
        curve : np.ndarray
            Candidate concentration time course.
        tr : float
            Temporal resolution in seconds.
        baseline_frames : int
            Number of pre-contrast frames.
        criteria : str
            Quality criteria level.
        tissue_ttp_frame : float or None, optional
            Tissue TTP for timing validation (optional).
        atlas_constrained : bool, optional
            Whether detection used atlas spatial constraint.

        Returns
        -------
        tuple of (float, dict)
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
        peak_deviation = abs(peak - expected_peak) / max(
            expected_peak, EPSILON_CONCENTRATION
        )
        peak_score = max(0.0, 1.0 - peak_deviation)
        score += 0.20 * peak_score

        # CNR contribution (0-0.15): higher is better, but saturated so a
        # multi-voxel average cannot buy back a degraded tail through CNR alone.
        cnr_score = min(1.0, cnr / AIF_CNR_SATURATION)
        score += 0.15 * cnr_score

        # Washout contribution: asymmetric band around the physiologic arterial
        # washout. A monotonic "higher washout = better" reward is wrong — it
        # rewards an under-washed (tissue-diluted) or collapsed tail. Penalize
        # under-washing steeply (contamination) but over-washing softly: on a long
        # acquisition a genuine artery reads high simply because the fixed tail is
        # sampled deeper into renal clearance, and it must still clear the accept gate.
        washout_score = _asymmetric_band_score(
            washout,
            AIF_WASHOUT_PHYSIOLOGIC_CENTER,
            AIF_WASHOUT_BAND_HALFWIDTH,
            AIF_WASHOUT_BAND_HALFWIDTH_UPPER,
        )
        score += w_washout * washout_score

        # FWHM contribution: closer to ideal first-pass width
        ideal_fwhm = AIF_IDEAL_FWHM_SECONDS
        fwhm_range = thresholds.fwhm_max_seconds - thresholds.fwhm_min_seconds
        fwhm_score = 1.0 - abs(fwhm - ideal_fwhm) / max(fwhm_range / 2, 1.0)
        score += w_fwhm * max(0.0, fwhm_score)

        # Upstroke slope contribution (0-0.15): steeper is better for arteries
        if upstroke > 0:
            # tr*3 is a time (s); guard only against a missing/zero TR.
            upstroke_score = min(1.0, upstroke / (peak / max(tr * 3.0, 1e-6)))
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

        weight_sum = 0.20 + 0.15 + w_washout + w_fwhm + 0.15
        if tissue_ttp_frame is not None:
            weight_sum += timing_weight
        return min(1.0, score / max(weight_sum, 0.01)), metrics

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

        Parameters
        ----------
        candidate_peak : float
            Candidate's peak concentration in mM.
        peak_min_expected : float
            Expected minimum arterial peak for this dose.
        peak_floor : float
            Minimum peak to allow rescue (criteria/mode-dependent).
        tier_used : int
            Arterial cascade tier (1=strict, 7=last resort).
        is_likely_venous : bool
            Whether timing-based venous flag was set.
        atlas_constrained : bool
            Whether detection used atlas spatial constraint.

        Returns
        -------
        float
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

    @staticmethod
    def _validate_peak_timing(
        candidates: list[tuple[int, int, int]],
        concentration_data: npt.NDArray[Any],
        n_timepoints: int,
    ) -> list[tuple[int, int, int]]:
        """Reject candidates peaking after 40% of timepoints.

        Parameters
        ----------
        candidates : list
            Candidate voxel coordinates.
        concentration_data : np.ndarray
            4D concentration data.
        n_timepoints : int
            Total number of timepoints.

        Returns
        -------
        list
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
        """Select the best spatial cluster among keeper-eligible candidates.

        Uses 26-connectivity labeling and composite scoring:
        0.3 * FWHM_score + 0.3 * washout_score + 0.15 * upstroke_score + 0.25 * size_score

        Parameters
        ----------
        candidates : list
            Candidate voxel coordinates.
        scores : list
            Per-candidate quality scores.
        metrics_list : list
            Per-candidate metric dictionaries.
        brain_shape : tuple
            Spatial dimensions (nx, ny, nz).
        tr : float, optional
            Temporal resolution in seconds (for FWHM frame conversion).

        Returns
        -------
        tuple of (list, list, list)
            (candidates, scores, metrics) for the best cluster.
            Candidates below the downstream keeper threshold are removed before
            clustering. If only one eligible cluster exists or clustering fails,
            returns the eligible candidates unchanged.
        """
        eligible_idx = [
            i for i, score in enumerate(scores) if score >= _MIN_CANDIDATE_SCORE
        ]
        candidates = [candidates[i] for i in eligible_idx]
        scores = [scores[i] for i in eligible_idx]
        metrics_list = [metrics_list[i] for i in eligible_idx]

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

            # FWHM score in physical units. Candidate FWHM is already measured
            # in seconds, so scoring it in frames would make the same physical
            # curve receive a different cluster score at different TRs.
            fwhm_vals = [
                metrics_list[i].get("fwhm_seconds", 30.0) for i in cluster_indices
            ]
            mean_fwhm = float(np.mean(fwhm_vals))
            fwhm_score = _two_sided_band_score(
                mean_fwhm,
                AIF_IDEAL_FWHM_SECONDS,
                AIF_FWHM_BAND_HALFWIDTH_SECONDS,
            )

            # Match the per-candidate asymmetric physiologic washout band.
            # A monotonic raw washout reward favors tail-collapsed curves.
            washout_vals = [
                metrics_list[i].get("washout", 0.0) for i in cluster_indices
            ]
            washout_score = float(np.mean([
                _asymmetric_band_score(
                    value,
                    AIF_WASHOUT_PHYSIOLOGIC_CENTER,
                    AIF_WASHOUT_BAND_HALFWIDTH,
                    AIF_WASHOUT_BAND_HALFWIDTH_UPPER,
                )
                for value in washout_vals
            ]))

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

    @staticmethod
    def _cascade_exhaustion_provenance(
        cascade_tiers_attempted: int,
        anchor_outcomes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the QC provenance dict for Pass-3 (population) fallback.

        Used when ``_run_two_pass_cascade`` returns ``detection_method=
        "population"`` with empty candidates — i.e. no tier produced any
        anchored keeper anywhere. The downstream per-candidate quality
        filter at ``_run_interface`` only writes provenance when
        ``candidates`` is non-empty, so this helper is the only path that
        captures Pass-3 exhaustion.

        Parameters
        ----------
        cascade_tiers_attempted : int
            Total tier attempts executed by the
            cascade (max 13 = 4 atlases × 3 criteria + 1 whole-brain).
        anchor_outcomes : list or None, optional
            Per-tier anchor diagnostics from the cascade.
            When provided, embedded under ``cascade_anchor_outcomes``
            so QC reviewers can see why each tier failed (quorum,
            anchor, or both).

        Returns
        -------
        dict
            Provenance dict consumed by ``qc_metrics.update(...)`` in
            ``_run_interface``.
        """
        prov: dict[str, Any] = {
            "auto_detection_attempted": True,
            "auto_best_score": 0.0,
            "auto_n_candidates": 0,
            "auto_n_above_threshold": 0,
            "auto_fallback_reason": (
                f"all {cascade_tiers_attempted} cascade tiers "
                f"exhausted; no tier produced any anchored keeper"
            ),
            "cascade_tiers_attempted": cascade_tiers_attempted,
            "cascade_best_score": 0.0,
        }
        if anchor_outcomes is not None:
            prov["cascade_anchor_outcomes"] = anchor_outcomes
        return prov

    @staticmethod
    def _quorum_count(scores: list[float]) -> int:
        """Count candidates that would survive the per-candidate quality filter.

        Mirrors the threshold used by the downstream quality filter
        (``score >= _MIN_CANDIDATE_SCORE``). Used by the two-pass cascade
        as the predicate for "this tier produced a usable result"; the
        cascade only short-circuits when ``_quorum_count(scores) >=
        _MIN_CANDIDATES_FOR_DETECTED``.

        Parameters
        ----------
        scores : list
            Per-candidate composite scores.

        Returns
        -------
        int
            Number of candidates with score at or above the keeper threshold.
        """
        return sum(1 for s in scores if s >= _MIN_CANDIDATE_SCORE)

    @staticmethod
    def _has_arterial_anchor(
        metrics_list: list[dict[str, float]],
        baseline_frames: int,
        tr: float,
    ) -> tuple[bool, float]:
        """Mouridsen-style cluster-anchor predicate.

        A tier is "anchored" if at least one candidate has post-arrival
        peak time within the physiological arterial range. This catches
        clusters with low per-voxel peaks but at least one voxel with
        arterial-grade timing — while rejecting all-venous clusters
        where every candidate lags arterial timing by more than
        ``_MAX_ANCHOR_POST_ARRIVAL_S``.

        Operates on the post-filter survivor list returned by
        ``_find_candidates_in_mask`` (NOT the raw mask candidates) — by
        the time it is called, cluster selection and venous filtering
        have already run.

        Parameters
        ----------
        metrics_list : list
            Per-candidate metrics dicts; each must contain
            ``peak_time_frame``.
        baseline_frames : int
            Bolus arrival frame in the AIF (from
            baseline detection — same value used by the cascade).
        tr : float
            Temporal resolution in seconds.

        Returns
        -------
        tuple of (bool, float)
            ``(has_anchor, earliest_post_arr_s)``. ``has_anchor`` is True
            iff the earliest candidate's post-arrival peak time is at or
            below ``_MAX_ANCHOR_POST_ARRIVAL_S``. ``earliest_post_arr_s``
            is returned for QC logging regardless of pass/fail (``inf``
            when ``metrics_list`` is empty).
        """
        if not metrics_list:
            return False, float("inf")
        # Default to ``inf`` for missing key so the gate fails-closed
        # rather than falsely anchoring (peak_time_frame=0 → very negative
        # post-arrival → would always anchor).
        earliest_post_arr_s = min(
            (float(m.get("peak_time_frame", float("inf"))) - baseline_frames) * tr
            for m in metrics_list
        )
        return (
            earliest_post_arr_s <= _MAX_ANCHOR_POST_ARRIVAL_S,
            earliest_post_arr_s,
        )

    def _score_and_gate_pool(
        self,
        cands: list[tuple[int, int, int]],
        scores: list[float],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        affine: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
        hct: float,
        target_peak: float,
        atlas_tier_label: str,
    ) -> PoolGateOutcome:
        """Average a candidate pool and score it against the G1-G6 gates.

        Runs the shared sequence: weighted-ROI average -> plasma-convert
        (``/ (1 - Hct)``) -> amplitude rescale -> QC -> hemisphere / slab
        span -> gate evaluation. Used by both the progressive-trim loop and
        the post-cascade C2 re-gate so the two paths cannot drift.

        Parameters
        ----------
        cands : list of (int, int, int)
            Candidate voxel coordinates for this pool.
        scores : list of float
            Per-candidate composite scores (averaging weights).
        conc_data, brain_mask, affine : np.ndarray
            Concentration series, brain mask, and voxel-to-world affine.
        tr : float
            Repetition time (s).
        baseline_frames : int
            Pre-contrast baseline frame count.
        hct : float
            Arterial hematocrit (``0 <= hct < 1``).
        target_peak : float
            Scaling target-peak override (``<= 0`` -> field default).
        atlas_tier_label : str
            Tier label passed through to the gate evaluation (G6 (f)).

        Returns
        -------
        PoolGateOutcome
            The scaled curve, scale factor, QC, hemisphere, slab span,
            gate status, and gate decision for this pool.
        """
        curve_blood, _ = self._compute_weighted_roi_aif(
            cands, scores, conc_data, brain_mask, tr=tr,
        )
        curve_plasma = curve_blood / (1.0 - hct) if hct < 1.0 else curve_blood
        scale = _resolve_aif_scale(
            curve_plasma,
            hct,
            target_peak,
            self._get_expected_peak_range,
            apply_peak_floor=bool(getattr(self.inputs, "aif_peak_floor", True)),
        )
        qc = _compute_pool_qc(scale.curve, tr, baseline_frames)
        hemisphere, _n_left, _n_right = _classify_hemisphere(cands, affine)
        n_slabs = _count_slabs_spanned(cands)
        status, decision = _evaluate_gates(
            scale.required_factor, qc, atlas_tier_label,
            len(cands), hemisphere, n_slabs,
        )
        return PoolGateOutcome(
            curve_scaled=scale.curve,
            scale_factor=scale.applied_factor,
            required_scale_factor=scale.required_factor,
            qc=qc,
            hemisphere=hemisphere,
            n_slabs=n_slabs,
            status=status,
            decision=decision,
        )

    def _evaluate_pool_with_gates_and_trim(
        self,
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        metrics_list: list[dict[str, float]],
        conc_data: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        affine: npt.NDArray[Any],
        tr: float,
        baseline_frames: int,
        hct: float,
        target_peak: float,
        atlas_tier_label: str,
    ) -> TierEvaluation:
        """Run progressive trim with G1-G6 gates on a candidate pool.

        Per iteration: average pool -> Hct-correct -> rescale -> compute
        QC -> evaluate gates. On acceptance (PASS / PASS_WARN /
        G6_ADMIT) returns a ``TierEvaluation`` with the accepted
        ``TierResult`` and the gate-decision log up to that iteration.
        On rejection, drops the worst candidate by composite G-aware
        metric and retries. When the pool falls below
        ``DCE_AIF_MIN_TRIM_POOL`` without acceptance, returns a
        ``TierEvaluation`` with ``result=None`` and the full per-
        iteration ``gate_decisions`` log preserved (caller writes them
        to the per-tier QC trail before continuing the cascade).
        """
        pool_idx = list(range(len(candidates)))
        gate_decisions: list[GateDecision] = []

        while len(pool_idx) >= DCE_AIF_MIN_TRIM_POOL:
            sub_cands = [candidates[i] for i in pool_idx]
            sub_scores = [scores[i] for i in pool_idx]
            sub_metrics = [metrics_list[i] for i in pool_idx]

            # Average + plasma-convert + rescale + QC + hemisphere/slabs + gate
            # (recomputed each iter because dropping a candidate may change them).
            outcome = self._score_and_gate_pool(
                sub_cands, sub_scores, conc_data, brain_mask, affine, tr,
                baseline_frames, hct, target_peak, atlas_tier_label,
            )
            gate_decisions.append(outcome.decision)

            if outcome.status in ("PASS", "PASS_WARN", "G6_ADMIT"):
                return TierEvaluation(
                    result=TierResult(
                        candidates=sub_cands,
                        scores=sub_scores,
                        metrics=sub_metrics,
                        curve_scaled=outcome.curve_scaled,
                        scale_factor=outcome.scale_factor,
                        required_scale_factor=outcome.required_scale_factor,
                        qc=outcome.qc,
                        atlas_tier=atlas_tier_label,
                        hemisphere=outcome.hemisphere,
                        n_slabs=outcome.n_slabs,
                        status=outcome.status,
                        gate_decisions=gate_decisions,
                    ),
                    gate_decisions=gate_decisions,
                )

            failing_gate = outcome.decision.get("failing_gate") or ""

            if failing_gate in ("G6_f", "G6_g"):
                return TierEvaluation(result=None, gate_decisions=gate_decisions)

            worst_local = _select_worst_for_trim(
                sub_metrics, sub_scores, failing_gate,
            )
            pool_idx.pop(worst_local)

        return TierEvaluation(result=None, gate_decisions=gate_decisions)

    def _run_two_pass_cascade(
        self,
        conc_data: npt.NDArray[Any],
        atlas_search_mask: npt.NDArray[Any],
        wb_search_mask: npt.NDArray[Any],
        atlas_masks: list[npt.NDArray[Any]],
        atlas_names: list[str],
        n_candidates: int,
        tr: float,
        baseline_frames: int,
        tissue_ttp_frame: float | None,
        brain_mask: npt.NDArray[Any] | None = None,
        affine: npt.NDArray[Any] | None = None,
        hct: float | None = None,
        target_peak: float | None = None,
    ) -> CascadeResult:
        """Run the two-pass multi-level AIF detection cascade.

        Pass 1: descend through every tier (4 atlases × {strict, relaxed,
        permissive} + whole-brain relaxed = up to 13 tiers), short-circuiting
        only when a tier produces ≥ 2 keepers AND has at least one
        arterial-grade timing anchor (Mouridsen-style cluster anchor).

        Pass 2: if no tier had ≥ 2 keepers, fall back to the best
        anchored single-keeper tier observed during Pass 1 (highest
        max-score among tiers whose lone keeper passes the anchor gate).

        Population AIF (Pass 3, handled by the caller) only triggers when
        no tier produced any anchored keeper anywhere.

        Whole-brain stays at "relaxed" (not permissive) because the atlas's
        spatial constraint is what makes permissive-tier candidates safe;
        without it, permissive admits tissue voxels.

        Parameters
        ----------
        conc_data : ndarray
            4D DCE concentration data ``(x, y, z, t)``.
        atlas_search_mask : ndarray
            Search mask for atlas-constrained tiers.
        wb_search_mask : ndarray
            Whole-brain search mask for the WB-relaxed tier.
        atlas_masks : list of ndarray
            Atlas masks ordered tightest → broadest. Empty list skips
            the atlas cascade.
        atlas_names : list of str
            Names parallel to ``atlas_masks`` (used for the
            ``atlas_tier_used`` provenance string).
        n_candidates : int
            Target candidate count per tier.
        tr : float
            Temporal resolution in seconds.
        baseline_frames : int
            Number of pre-bolus baseline frames.
        tissue_ttp_frame : float, optional
            Reference tissue time-to-peak frame (brain-wide).

        Returns
        -------
        CascadeResult
            Aggregated cascade outcome. Notable fields:
            - ``detection_method``: ``"atlas"`` / ``"whole_brain"`` /
              ``"population"``; ``"population"`` indicates no anchored
              keeper anywhere (caller handles Parker fallback).
            - ``atlas_tier_used``: ``f"{atlas_name}_{crit}"`` for the
              accepted tier (or ``"wb_relaxed"`` for whole-brain),
              ``None`` for population.
            - ``anchor_outcomes``: per-tier dicts with keys ``tier``,
              ``n_candidates``, ``quorum_count``, ``earliest_post_arr_s``,
              ``has_anchor``.
            - ``tier_gate_decisions``: per-tier dicts with keys
              ``tier``, ``outcome`` (``"PASS"`` / ``"trim_exhausted"`` /
              ``"skipped_quorum"`` / ``"skipped_anchor"``), and
              ``decisions`` (the per-iteration ``GateDecision`` log;
              empty when the tier was skipped before gates ran).
              Order matches ``anchor_outcomes`` (atlas tightest →
              broadest × strict → permissive, then ``wb_relaxed``).
        """
        candidates: list[tuple[int, int, int]] = []
        scores: list[float] = []
        metrics_list: list[dict[str, float]] = []
        detection_method: str = "population"
        atlas_tier_used: str | None = None
        cascade_tiers_attempted = 0
        anchor_outcomes: list[dict[str, Any]] = []
        tier_gate_decisions: list[dict[str, Any]] = []
        tier_result: TierResult | None = None

        # Per-tier gate evaluation is enabled when the caller supplies
        # the inputs needed to average + rescale + compute QC. Older
        # call sites that omit them get the legacy quorum + anchor
        # acceptance contract.
        gates_enabled = (
            brain_mask is not None
            and affine is not None
            and hct is not None
            and target_peak is not None
        )

        # Pass 2 tracking: (cands, scores, metrics, method, label, max_score)
        best_single_keeper: tuple[
            list[tuple[int, int, int]], list[float],
            list[dict[str, float]], str, str, float,
        ] | None = None

        def _record_tier(
            cands: list[tuple[int, int, int]],
            scs: list[float],
            label: str,
            has_anchor: bool,
            earliest_post_arr_s: float,
        ) -> None:
            anchor_outcomes.append({
                "tier": label,
                "n_candidates": len(cands),
                "quorum_count": self._quorum_count(scs),
                "earliest_post_arr_s": earliest_post_arr_s,
                "has_anchor": has_anchor,
            })

        def _record_tier_gate_outcome(
            label: str,
            outcome: str,
            decisions: list[GateDecision] | None = None,
        ) -> None:
            """Append a per-tier gate-evaluation entry for QC.

            Mirrors ``anchor_outcomes`` ordering. Captures the gate
            decisions even from rejected tiers so the QC consumer can
            see which gate killed each pool. ``decisions`` is the full
            per-iteration log from ``_evaluate_pool_with_gates_and_trim``;
            ``None`` (rendered as an empty list) when the tier was
            skipped before gates ran.
            """
            tier_gate_decisions.append({
                "tier": label,
                "outcome": outcome,
                "decisions": list(decisions) if decisions else [],
            })

        def _record_single_keeper(
            cands: list[tuple[int, int, int]],
            scs: list[float],
            mets: list[dict[str, float]],
            method: str,
            label: str,
        ) -> None:
            """Record an anchored single keeper for Pass 2 fallback.

            Pass-2 acceptance requires the keeper itself to satisfy the
            anchor gate (a single keeper is the only metric we have, so
            the cluster anchor degenerates to per-keeper timing). An
            unanchored single keeper is rejected — likely venous, not a
            useful Pass-2 fallback.
            """
            nonlocal best_single_keeper
            if not scs:
                return
            n_keepers = self._quorum_count(scs)
            if n_keepers != 1:
                return
            max_s = max(scs)
            if max_s < _MIN_CANDIDATE_SCORE:
                return
            keeper_idx = next(
                i for i, s in enumerate(scs) if s >= _MIN_CANDIDATE_SCORE
            )
            keeper_anchor, _ = self._has_arterial_anchor(
                [mets[keeper_idx]], baseline_frames, tr,
            )
            if not keeper_anchor:
                return
            if best_single_keeper is None or max_s > best_single_keeper[5]:
                best_single_keeper = (
                    list(cands), list(scs), list(mets),
                    method, label, max_s,
                )

        # Pass 1: atlas cascade (tightest → broadest × strict → permissive)
        pass1_succeeded = False
        if atlas_masks:
            for atlas_idx, atlas_mask in enumerate(atlas_masks):
                for crit in ("strict", "relaxed", "permissive"):
                    combined_mask = (atlas_search_mask > 0.5) & (atlas_mask > 0.5)
                    combined_mask = combined_mask.astype(atlas_search_mask.dtype)
                    cands, scs, mets = self._find_candidates_in_mask(
                        conc_data, combined_mask, n_candidates, tr,
                        baseline_frames, crit,
                        tissue_ttp_frame=tissue_ttp_frame,
                        atlas_constrained=True,
                    )
                    cascade_tiers_attempted += 1
                    label = f"{atlas_names[atlas_idx]}_{crit}"
                    has_anchor, earliest = self._has_arterial_anchor(
                        mets, baseline_frames, tr,
                    )
                    _record_tier(cands, scs, label, has_anchor, earliest)
                    if (
                        self._quorum_count(scs) >= _MIN_CANDIDATES_FOR_DETECTED
                        and has_anchor
                    ):
                        if gates_enabled:
                            evaluation = self._evaluate_pool_with_gates_and_trim(
                                cands, scs, mets,
                                conc_data, brain_mask, affine,  # type: ignore[arg-type]
                                tr, baseline_frames,
                                hct,  # type: ignore[arg-type]
                                target_peak,  # type: ignore[arg-type]
                                label,
                            )
                            tier_result = evaluation.result
                            if tier_result is None:
                                # Trim exhausted this tier — record the
                                # full gate-decision log and continue
                                # cascade.
                                _record_tier_gate_outcome(
                                    label, "trim_exhausted",
                                    evaluation.gate_decisions,
                                )
                                _record_single_keeper(cands, scs, mets, "atlas", label)
                                continue
                            candidates = tier_result.candidates
                            scores = tier_result.scores
                            metrics_list = tier_result.metrics
                            _record_tier_gate_outcome(
                                label, "PASS", evaluation.gate_decisions,
                            )
                        else:
                            candidates, scores, metrics_list = cands, scs, mets
                            _record_tier_gate_outcome(label, "PASS", None)
                        detection_method = "atlas"
                        atlas_tier_used = label
                        pass1_succeeded = True
                        break
                    # Skipped: failed quorum or anchor before gates ran.
                    _record_tier_gate_outcome(
                        label,
                        "skipped_anchor" if not has_anchor else "skipped_quorum",
                        None,
                    )
                    _record_single_keeper(cands, scs, mets, "atlas", label)
                if pass1_succeeded:
                    break

        # Pass 1 (cont.): whole-brain relaxed
        if not pass1_succeeded:
            cands, scs, mets = self._find_candidates_in_mask(
                conc_data, wb_search_mask, n_candidates, tr,
                baseline_frames, "relaxed",
                tissue_ttp_frame=tissue_ttp_frame,
                atlas_constrained=False,
            )
            cascade_tiers_attempted += 1
            has_anchor, earliest = self._has_arterial_anchor(
                mets, baseline_frames, tr,
            )
            _record_tier(cands, scs, "wb_relaxed", has_anchor, earliest)
            if (
                self._quorum_count(scs) >= _MIN_CANDIDATES_FOR_DETECTED
                and has_anchor
            ):
                if gates_enabled:
                    evaluation = self._evaluate_pool_with_gates_and_trim(
                        cands, scs, mets,
                        conc_data, brain_mask, affine,  # type: ignore[arg-type]
                        tr, baseline_frames,
                        hct,  # type: ignore[arg-type]
                        target_peak,  # type: ignore[arg-type]
                        "wb_relaxed",
                    )
                    tier_result = evaluation.result
                    if tier_result is None:
                        _record_tier_gate_outcome(
                            "wb_relaxed", "trim_exhausted",
                            evaluation.gate_decisions,
                        )
                        _record_single_keeper(
                            cands, scs, mets, "whole_brain", "wb_relaxed",
                        )
                    else:
                        candidates = tier_result.candidates
                        scores = tier_result.scores
                        metrics_list = tier_result.metrics
                        _record_tier_gate_outcome(
                            "wb_relaxed", "PASS", evaluation.gate_decisions,
                        )
                        detection_method = "whole_brain"
                        pass1_succeeded = True
                else:
                    candidates, scores, metrics_list = cands, scs, mets
                    _record_tier_gate_outcome("wb_relaxed", "PASS", None)
                    detection_method = "whole_brain"
                    pass1_succeeded = True
            else:
                _record_tier_gate_outcome(
                    "wb_relaxed",
                    "skipped_anchor" if not has_anchor else "skipped_quorum",
                    None,
                )
                _record_single_keeper(
                    cands, scs, mets, "whole_brain", "wb_relaxed",
                )

        # Pass 2: best anchored single-keeper tier
        if not pass1_succeeded and best_single_keeper is not None:
            (
                candidates, scores, metrics_list,
                detection_method, atlas_tier_used, _,
            ) = best_single_keeper

        return CascadeResult(
            candidates=candidates,
            scores=scores,
            metrics_list=metrics_list,
            detection_method=detection_method,
            atlas_tier_used=atlas_tier_used,
            cascade_tiers_attempted=cascade_tiers_attempted,
            anchor_outcomes=anchor_outcomes,
            tier_result=tier_result,
            tier_gate_decisions=tier_gate_decisions,
        )

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

        Parameters
        ----------
        concentration_data : ndarray
            4D concentration data.
        mask : ndarray
            3D search mask.
        n_candidates : int
            Number of candidates to return.
        tr : float
            Temporal resolution in seconds.
        baseline_frames : int
            Number of pre-contrast frames.
        criteria : str
            Quality criteria level (used for final scoring only).
        tissue_ttp_frame : float, optional
            Brain-wide tissue TTP frame (from
            ``_calculate_tissue_ttp_reference``).  Threaded into
            ``_apply_arterial_criteria`` and used downstream as
            ``tissue_ttp_computed`` for the per-voxel
            ``is_likely_venous`` flag and the scorer's timing
            contribution.  Atlas-constrained callers must supply this:
            the local IQR-of-moderate-enhancement reference inside an
            atlas mask is arterial-dominated and collapses onto the
            arterial peak time, falsely flagging arterial voxels as
            venous.  When ``None``, falls back to the local computation
            (preserves whole-brain semantics).
        atlas_constrained : bool
            Whether the mask is an atlas-restricted search region.

        Returns
        -------
        tuple
            ``(candidates, scores, metrics_list)`` — top N voxels.
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

        # Apply 7-tier arterial criteria cascade (TR-adaptive).  Pass the
        # caller-supplied brain-wide tissue_ttp_frame through so the
        # downstream venous-flag test (line below) and the scorer's
        # timing contribution use the parenchymal reference rather than
        # an atlas-local IQR collapsing onto arterial timing.
        candidate_idx_in_masked, tissue_ttp_computed, tier_used = self._apply_arterial_criteria(
            chars, spatial_exclude, n_timepoints,
            percentile_threshold=98.0, n_candidates=n_candidates * 2,
            tr=tr,
            tissue_ttp_frame=tissue_ttp_frame,
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
                    tier_used,
                    bool(all_metrics[idx].get("is_likely_venous", False)),
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

        Parameters
        ----------
        candidates : list
            Candidate voxel coordinates.
        scores : list
            Quality scores (higher is better).
        conc_data : np.ndarray
            4D concentration data.
        brain_mask : np.ndarray
            3D brain mask.
        roi_size : int, optional
            Side length of cubic ROI (1 = single voxel).
        tr : float, optional
            Temporal resolution in seconds (for adaptive smoothing).

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
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
            # Inverse-score weighting (higher score = higher weight).
            # Guard against an all-zero score array (e.g., trim pool that
            # outlived its keepers); fall back to uniform weights so the
            # averaged curve stays defined and downstream gates / fallback
            # logic still see a real curve.
            score_arr = np.array(scores, dtype=np.float64)
            score_sum = float(np.sum(score_arr))
            if score_sum <= 0.0:
                weights = np.full(score_arr.shape, 1.0 / max(score_arr.size, 1))
            else:
                weights = score_arr / score_sum

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

        Parameters
        ----------
        aif_curve : np.ndarray
            1D averaged AIF from ``_compute_weighted_roi_aif``.
        candidates : list
            Candidate voxel coordinates.
        scores : list
            Candidate quality scores.
        metrics_list : list
            Per-candidate metric dicts (must contain ``fwhm_seconds``).
        conc_data : np.ndarray
            4D concentration data.
        brain_mask : np.ndarray
            3D brain mask.
        tr : float
            Temporal resolution in seconds.
        baseline_frames : int
            Number of pre-bolus baseline frames.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, dict)
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
            validation_qc["aif_fwhm_confidence"] = "reduced"

        return best_curve, best_mask, validation_qc

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
        elif suffix == ".tsv":
            aif_curve = np.loadtxt(str(ext_path), dtype=np.float64, skiprows=1)
        else:
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

        # Quality score: the same consolidated averaged-AIF fidelity metric used
        # for detected AIFs (external AIF is assumed pre-scaled full-dose plasma).
        quality_score, _ = score_averaged_aif_fidelity(
            aif_curve,
            tr,
            baseline_frames,
            dose_factor=1.0,
            hematocrit=float(self.inputs.hematocrit),
        )

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
            "required_scaling_factor": 1.0,
            "aif_amplitude_calibration": "external",
            "detection_method": "external",
            "quality_score": quality_score,
            "baseline_frames": baseline_frames,
            "n_aif_voxels_selected": 0,
            "aif_status": QCStatus.PASS,
            # User-supplied AIF is trusted (already PASS); mark confidence
            # "standard" so the provenance flag and QC recommendation agree
            # rather than falling through to the report-only quality_score.
            "aif_confidence": "standard",
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

    def _load_and_validate_inputs(self) -> LoadedInputs:
        """Validate hematocrit + load the concentration series and brain mask (P0)."""
        import nibabel as nib  # type: ignore[import-untyped]

        # Hematocrit gates the blood->plasma conversion (/(1-Hct)) at every AIF
        # scaling site; it must be in [0, 1) or the AIF goes inf/negative. Fail
        # fast on a bad config/sidecar value rather than writing a corrupt AIF.
        _hct_in = float(self.inputs.hematocrit)
        if not (0.0 <= _hct_in < 1.0):
            raise ValueError(
                "hematocrit must be in [0, 1) for blood->plasma AIF conversion, "
                f"got {_hct_in!r}."
            )

        # Load data
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]
        header = conc_img.header  # type: ignore[attr-defined]

        brain_img = nib.load(self.inputs.brain_mask_file)
        brain_mask: npt.NDArray[Any] = brain_img.get_fdata()  # type: ignore[attr-defined]

        return LoadedInputs(
            conc_data=conc_data, affine=affine, header=header, brain_mask=brain_mask,
        )

    def _load_atlas_and_lesion_masks(
        self,
    ) -> tuple[list[npt.NDArray[Any]], list[str], Optional[npt.NDArray[Any]]]:
        """Load the optional atlas cascade masks + lesion mask (P2)."""
        import nibabel as nib  # type: ignore[import-untyped]

        # Optional atlas masks — build ordered list (tightest → broadest)
        # User-supplied aif_mask_file overrides all atlas masks.
        atlas_masks: list[npt.NDArray[Any]] = []
        atlas_names: list[str] = []
        if isdefined(self.inputs.aif_mask_file) and self.inputs.aif_mask_file:
            atlas_img = nib.load(self.inputs.aif_mask_file)
            atlas_masks = [atlas_img.get_fdata()]  # type: ignore[attr-defined]
            atlas_names = [re.sub(r"^MNI_\d+mm_MRA_arterial_", "", Path(self.inputs.aif_mask_file).name.split(".")[0])]
        else:
            # Load tiered atlas cascade (atlas_mask_files, tightest first)
            if isdefined(self.inputs.atlas_mask_files):
                for af in self.inputs.atlas_mask_files:
                    af_path = Path(str(af))
                    if af_path.exists() and str(af) not in ("", "None"):
                        atlas_masks.append(
                            nib.load(str(af_path)).get_fdata()  # type: ignore[attr-defined]
                        )
                        atlas_names.append(
                            re.sub(r"^MNI_\d+mm_MRA_arterial_", "", af_path.name.split(".")[0])
                        )
        # Optional lesion mask
        lesion_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.lesion_mask_file:
            lesion_img = nib.load(self.inputs.lesion_mask_file)
            lesion_mask = lesion_img.get_fdata()  # type: ignore[attr-defined]

        return atlas_masks, atlas_names, lesion_mask

    def _read_params(
        self, conc_data: npt.NDArray[Any], header: Any,
    ) -> AIFParams:
        """Read scalar detection parameters + derive geometry (P3)."""
        aif_method = str(self.inputs.aif_method)
        n_candidates = int(self.inputs.n_candidates)
        baseline_frames = int(self.inputs.baseline_frames)
        field_strength = float(self.inputs.field_strength)
        tr = float(self.inputs.tr)
        target_peak = float(self.inputs.target_peak)
        lesion_distance = float(self.inputs.lesion_distance)

        n_timepoints = conc_data.shape[3]

        # Get blood T1 for this field strength
        blood_t1 = get_blood_t1(field_strength)

        # Voxel sizes from header
        voxel_sizes = tuple(float(v) for v in header.get_zooms()[:3])  # type: ignore[attr-defined]

        return AIFParams(
            aif_method=aif_method, n_candidates=n_candidates,
            baseline_frames=baseline_frames, field_strength=field_strength, tr=tr,
            target_peak=target_peak, lesion_distance=lesion_distance,
            n_timepoints=n_timepoints, blood_t1=blood_t1, voxel_sizes=voxel_sizes,
        )

    def _run_detection(
        self,
        loaded: LoadedInputs,
        atlas_masks: list[npt.NDArray[Any]],
        atlas_names: list[str],
        lesion_mask: Optional[npt.NDArray[Any]],
        params: AIFParams,
    ) -> DetectionState:
        """Run the detection cascade + post-cascade filters (P5)."""
        conc_data = loaded.conc_data
        brain_mask = loaded.brain_mask
        affine = loaded.affine
        aif_method = params.aif_method
        n_candidates = params.n_candidates
        baseline_frames = params.baseline_frames
        tr = params.tr
        target_peak = params.target_peak
        lesion_distance = params.lesion_distance
        voxel_sizes = params.voxel_sizes
        # _detection_attempt is assigned only conditionally in the body
        # (cascade exhaustion / 0-survivor); re-init so the success path binds.
        _detection_attempt: dict[str, Any] | None = None
        # Restore the Literal type the narrowing (below) relies on.
        detection_method: Literal["atlas", "whole_brain", "population"]
        # ===== Phase 1: Prepare search masks =====
        if lesion_distance < LESION_EXCLUSION_FLOOR_MM:
            logger.warning(
                "lesion_distance=%.1f mm is below hard minimum (%.1f mm); "
                "clamping to %.1f mm.",
                lesion_distance, LESION_EXCLUSION_FLOOR_MM,
                LESION_EXCLUSION_FLOOR_MM,
            )
            _lesion_dist = LESION_EXCLUSION_FLOOR_MM
        elif lesion_distance < LESION_EXCLUSION_DISTANCE_MM:
            logger.warning(
                "lesion_distance=%.1f mm is below recommended minimum "
                "(%.1f mm); peritumoral vascular abnormalities may "
                "contaminate AIF candidates.",
                lesion_distance, LESION_EXCLUSION_DISTANCE_MM,
            )
            _lesion_dist = lesion_distance
        else:
            _lesion_dist = lesion_distance
        # Inferior inflow exclusion is applied universally — BIDS
        # MRAcquisitionType does not reliably distinguish slab-
        # selective 2D-prep sequences (Look-Locker, FAIR, MOTSA
        # inferior partitions, slab-encoded TWIST) from plain 2D-MS,
        # and the latter are the only case where slab-entry inflow
        # is genuinely absent. Cost on plain 2D-MS DCE is ~5
        # inferior slices in WB search; benefit is correct behavior
        # for all slab-selective acquisitions.
        inferior_inflow_mm = _inflow_exclusion_mm(
            voxel_sizes[2], conc_data.shape[2],
        )
        if self.inputs.mr_acquisition_type:
            logger.info(
                "DCE MRAcquisitionType=%s; inflow exclusion applied "
                "universally.",
                (self.inputs.mr_acquisition_type or "").upper(),
            )

        # Atlas search: skip midline exclusion (MRA-derived atlas
        # already excludes venous structures, including SSS) and use
        # rolloff-only superior margin.
        atlas_search_mask = self._prepare_search_mask(
            brain_mask,
            voxel_sizes,
            lesion_mask=lesion_mask,
            lesion_distance_mm=_lesion_dist,
            skip_midline=True,
            slab_edge_inferior_mm=inferior_inflow_mm,
            slab_edge_superior_mm=SLAB_EDGE_EXCLUSION_MM_ROLLOFF,
            erosion_iterations=BRAIN_EROSION_ITERATIONS_ATLAS,
        )
        # Whole-brain fallback: full erosion, midline exclusion,
        # SSS-sized superior margin, and a 10 mm physical-depth cut
        # to strip the lateral cortical surface where cortical-vein
        # drainage produces arterial-shaped curves at clinical TR.
        # Atlas search skips this cut so it
        # retains 75-94 % of MCA_ACA territory.
        wb_search_mask = self._prepare_search_mask(
            brain_mask,
            voxel_sizes,
            lesion_mask=lesion_mask,
            lesion_distance_mm=_lesion_dist,
            skip_midline=False,
            slab_edge_inferior_mm=inferior_inflow_mm,
            slab_edge_superior_mm=SSS_EXCLUSION_MM,
            erosion_iterations=BRAIN_EROSION_ITERATIONS,
            physical_erosion_mm=WB_FALLBACK_DEPTH_EROSION_MM,
        )

        # ===== Phase 3: Compute tissue TTP reference =====
        tissue_ttp_frame = self._calculate_tissue_ttp_reference(
            conc_data, brain_mask, baseline_frames
        )

        # ===== Two-pass multi-level detection cascade =====
        # See ``_run_two_pass_cascade`` docstring for contract details.
        # Brain mask, affine, hct, and target_peak enable per-tier
        # G1-G6 acceptance gates with progressive trim; without them
        # the cascade falls back to its legacy quorum + anchor
        # acceptance criterion.
        cascade_result = self._run_two_pass_cascade(
            conc_data,
            atlas_search_mask=atlas_search_mask,
            wb_search_mask=wb_search_mask,
            atlas_masks=atlas_masks,
            atlas_names=atlas_names,
            n_candidates=n_candidates,
            tr=tr,
            baseline_frames=baseline_frames,
            tissue_ttp_frame=tissue_ttp_frame,
            brain_mask=brain_mask,
            affine=affine,
            hct=float(self.inputs.hematocrit),
            target_peak=target_peak,
        )
        candidates = cascade_result.candidates
        scores = cascade_result.scores
        metrics_list = cascade_result.metrics_list
        detection_method_str = cascade_result.detection_method
        atlas_tier_used = cascade_result.atlas_tier_used
        cascade_tiers_attempted = cascade_result.cascade_tiers_attempted
        anchor_outcomes = cascade_result.anchor_outcomes
        tier_result = cascade_result.tier_result
        tier_gate_decisions = cascade_result.tier_gate_decisions
        # Narrow back to Literal for the typed local.
        detection_method = (  # type: ignore[assignment]
            "atlas" if detection_method_str == "atlas"
            else "whole_brain" if detection_method_str == "whole_brain"
            else "population"
        )

        # Cascade-exhaustion provenance: when the two-pass cascade
        # returns ``detection_method="population"`` (no tier produced
        # any anchored keeper anywhere), record the fact + tier-attempt
        # count + per-tier anchor outcomes so QC reviewers can see why
        # each tier failed. The downstream per-candidate quality
        # filter only writes provenance when ``candidates`` is
        # non-empty, so this branch is the only path that captures
        # Pass-3 exhaustion.
        if not candidates and detection_method_str == "population":
            _detection_attempt = self._cascade_exhaustion_provenance(
                cascade_tiers_attempted,
                anchor_outcomes=anchor_outcomes,
            )
            logger.warning(
                "AIF cascade: all %d tiers exhausted without producing "
                "any anchored keeper. Falling back to Parker population AIF.",
                cascade_tiers_attempted,
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
                # No candidates above threshold — fall back to population.
                # With the two-pass cascade above, this branch is now
                # only reachable when no tier produced any keeper across
                # the entire cascade (population is Pass 3).
                best_score = max(scores) if scores else 0.0
                _detection_attempt = {
                    "auto_detection_attempted": True,
                    "auto_best_score": float(best_score),
                    "auto_n_candidates": n_before,
                    "auto_n_above_threshold": 0,
                    "auto_fallback_reason": (
                        f"all {cascade_tiers_attempted} cascade tiers "
                        f"exhausted; final tier: 0/{n_before} candidates "
                        f"scored >= {_MIN_CANDIDATE_SCORE}"
                    ),
                    "cascade_tiers_attempted": cascade_tiers_attempted,
                    "cascade_best_score": float(best_score),
                }
                logger.warning(
                    "AIF quality filter: 0/%d candidates scored >= %.2f "
                    "(best=%.2f) after %d cascade tiers. "
                    "Falling back to Parker population AIF.",
                    n_before, _MIN_CANDIDATE_SCORE, best_score,
                    cascade_tiers_attempted,
                )
                candidates = []
                scores = []
                metrics_list = []

        return DetectionState(
            detection_method=detection_method,
            candidates=candidates,
            scores=scores,
            metrics_list=metrics_list,
            tissue_ttp_frame=tissue_ttp_frame,
            detection_attempt=_detection_attempt,
            atlas_tier_used=atlas_tier_used,
            anchor_outcomes=anchor_outcomes,
            tier_gate_decisions=tier_gate_decisions,
            tier_result=tier_result,
            n_filtered_out=n_filtered_out,
        )

    def _emit_detected_aif(
        self, state: DetectionState, loaded: LoadedInputs, params: AIFParams,
    ) -> EmitResult:
        """Average + emit the detected AIF curve (P6, detected branch)."""
        conc_data = loaded.conc_data
        brain_mask = loaded.brain_mask
        affine = loaded.affine
        tr = params.tr
        baseline_frames = params.baseline_frames
        target_peak = params.target_peak
        candidates = state.candidates
        scores = state.scores
        metrics_list = state.metrics_list
        detection_method = state.detection_method
        atlas_tier_used = state.atlas_tier_used
        tissue_ttp_frame = state.tissue_ttp_frame
        tier_result = state.tier_result
        n_filtered_out = state.n_filtered_out
        # validation_qc is emit-local (relocated from the pre-detection init);
        # the C2/reuse sub-branches never assign it before its read.
        validation_qc: dict[str, Any] = {}
        aif_curve: npt.NDArray[Any]
        aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        quality_score = 0.0
        scaling_factor = 1.0
        required_scaling_factor = 1.0
        calibration_mode = (
            "explicit_target" if target_peak > 0
            else "upward_floor"
            if bool(getattr(self.inputs, "aif_peak_floor", True))
            else "none"
        )
        qc_metrics: dict[str, Any] = {}

        # Phase 5: Weighted ROI averaging + smoothing
        # When the cascade ran G1-G6 acceptance gates with progressive
        # trim (gates_enabled in _run_two_pass_cascade), the accepted
        # tier's averaged + Hct-corrected + rescaled curve is already
        # in ``tier_result``; reuse it to avoid double-scaling and to
        # propagate the per-iteration gate decisions downstream. The
        # legacy averaging path runs only when the cascade fell back to
        # its pre-gate accept-on-quorum-and-anchor contract (no inputs
        # were supplied to enable gates).
        if tier_result is not None:
            tier_cand_set = set(tuple(c) for c in tier_result.candidates)
            cur_cand_set = set(tuple(c) for c in candidates)
            if cur_cand_set != tier_cand_set:
                # C2: post-cascade filters shrank the accepted pool. Re-gate
                # the EMITTED subset (single evaluation, no trim) and rebuild
                # tier_result so downstream provenance (confidence /
                # uncertainty / gate decisions) describes the emitted AIF, not
                # the pre-filter pool. Provenance-only: a re-gate FAIL is
                # recorded (downgrades confidence below) but does NOT trigger a
                # new population fallback here.
                outcome = self._score_and_gate_pool(
                    candidates, scores, conc_data, brain_mask, affine, tr,
                    baseline_frames, float(self.inputs.hematocrit),
                    target_peak, tier_result.atlas_tier,
                )
                aif_curve = outcome.curve_scaled
                scaling_factor = outcome.scale_factor
                required_scaling_factor = outcome.required_scale_factor
                tier_result = TierResult(
                    candidates=list(candidates),
                    scores=list(scores),
                    metrics=list(metrics_list),
                    curve_scaled=aif_curve,
                    scale_factor=scaling_factor,
                    required_scale_factor=required_scaling_factor,
                    qc=outcome.qc,
                    atlas_tier=tier_result.atlas_tier,
                    hemisphere=outcome.hemisphere,
                    n_slabs=outcome.n_slabs,
                    status=outcome.status,
                    gate_decisions=[outcome.decision],
                )
            else:
                aif_curve = tier_result.curve_scaled
                scaling_factor = tier_result.scale_factor
                required_scaling_factor = tier_result.required_scale_factor
            aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
            for cx, cy, cz in candidates:
                aif_mask[cx, cy, cz] = 1
        else:
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
        # Emitted quality score = fidelity of the POOLED/averaged AIF (report /
        # regression sentinel), not the mean of per-voxel scores — the latter
        # never inspected the averaged curve, so a tail-diluted pool could score
        # high. Reported for QC; contamination is gated on the specific gates.
        quality_score, _ = score_averaged_aif_fidelity(
            np.asarray(aif_curve, dtype=float), tr, baseline_frames,
            dose_factor=self._get_dose_factor(),
            hematocrit=float(self.inputs.hematocrit),
        )

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
                "n_aif_voxels_selected": len(candidates),
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
            hemi_cov, n_left, n_right = _classify_hemisphere(candidates, affine)
            qc_metrics["n_left_candidates"] = n_left
            qc_metrics["n_right_candidates"] = n_right
            qc_metrics["hemisphere_coverage"] = hemi_cov

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

        return EmitResult(
            aif_curve=aif_curve, aif_mask=aif_mask, quality_score=quality_score,
            scaling_factor=scaling_factor,
            required_scaling_factor=required_scaling_factor,
            calibration_mode=calibration_mode,
            qc_metrics=qc_metrics,
            tier_result=tier_result, detection_method=detection_method,
        )

    def _emit_population_aif(
        self, state: DetectionState, loaded: LoadedInputs, params: AIFParams,
    ) -> EmitResult:
        """Emit the BAT-aligned Parker population AIF (P6, population branch)."""
        conc_data = loaded.conc_data
        tr = params.tr
        n_timepoints = params.n_timepoints
        baseline_frames = params.baseline_frames
        _detection_attempt = state.detection_attempt
        detection_method: Literal["atlas", "whole_brain", "population"]
        # Prologue defaults the population branch relies on (it assigns neither).
        aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        scaling_factor = 1.0
        # Generate BAT-aligned Parker population AIF
        dose_factor = self._get_dose_factor()
        t = np.arange(n_timepoints) * tr

        # For population AIF, use baseline_frames * TR as BAT. Data-driven BAT
        # detection on mean brain concentration detects tissue transit
        # delay, not arterial arrival.
        bat_shift = baseline_frames * tr
        t_shifted = t - bat_shift

        # Population AIF is a true-mM concentration curve; NO relaxivity
        # scaling (concentration is r1-independent — see parker_population_aif).
        t_for_parker = np.maximum(t_shifted, 0.0)
        aif_curve = parker_population_aif(t_for_parker, dose_factor)

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
            aif_curve = parker_population_aif(t_unshifted, dose_factor)
            bat_override = True

        # Scale population AIF to match pipeline's plasma-Tofts
        # convention.
        # Parker's published AIF parameters were derived from blood-voxel
        # ΔR1/r1 in a cohort with Hct = 0.42, so under FXL the published
        # "C_p" values are effectively (1-Hct) × C_p_true. Dividing by
        # (1-Hct) brings the population AIF into the same plasma scale
        # as the Hct-corrected detected AIF (line 2744) — required for
        # longitudinal consistency when a subject flips between detected
        # and population across sessions. OSIPI/QIBA DROs distribute
        # Parker as C_p directly (test-harness convention); pmriprep's
        # (1-Hct) ≈ 0.58× offset against OSIPI GT is a known harness-vs-
        # clinical convention mismatch documented in the manuscript.
        aif_curve = aif_curve / (1.0 - float(self.inputs.hematocrit))

        detection_method = "population"
        quality_score = 0.5  # Moderate score for population AIF

        qc_metrics = {
            "population_aif": True,
            "aif_source": "population",  # For boilerplate text generation
            "dose_factor": dose_factor,
            "n_aif_voxels_selected": 0,
            "bat_seconds": bat_shift,
            "bat_method": "baseline_frames",
        }
        if bat_override:
            qc_metrics["bat_override"] = True
            qc_metrics["original_bat_seconds"] = original_bat_seconds
        if _detection_attempt is not None:
            qc_metrics.update(_detection_attempt)

        return EmitResult(
            aif_curve=aif_curve, aif_mask=aif_mask, quality_score=quality_score,
            scaling_factor=scaling_factor,
            required_scaling_factor=1.0,
            calibration_mode="population",
            qc_metrics=qc_metrics,
            tier_result=state.tier_result, detection_method=detection_method,
        )

    def _finalize_and_save(
        self, runtime: Any, emit: EmitResult, state: DetectionState,
        loaded: LoadedInputs, params: AIFParams,
    ) -> Any:
        """Provenance + confidence + Pass-2 anchor + final metrics + save (P7-P11)."""
        import nibabel as nib  # type: ignore[import-untyped]
        aif_curve = emit.aif_curve
        aif_mask = emit.aif_mask
        quality_score = emit.quality_score
        scaling_factor = emit.scaling_factor
        required_scaling_factor = emit.required_scaling_factor
        calibration_mode = emit.calibration_mode
        qc_metrics = emit.qc_metrics
        tier_result = emit.tier_result
        detection_method = emit.detection_method
        anchor_outcomes = state.anchor_outcomes
        tier_gate_decisions = state.tier_gate_decisions
        candidates = state.candidates
        affine = loaded.affine
        tr = params.tr
        baseline_frames = params.baseline_frames
        target_peak = params.target_peak
        blood_t1 = params.blood_t1
        field_strength = params.field_strength

        # ===== Cascade provenance (fires on both detected and population) =====
        if anchor_outcomes:
            qc_metrics["cascade_anchor_outcomes"] = anchor_outcomes
        if tier_gate_decisions:
            qc_metrics["cascade_gate_decisions"] = [
                {
                    "tier": entry["tier"],
                    "outcome": entry["outcome"],
                    "decisions": [
                        {
                            "tier": d["tier"],
                            "n_pool": d["n_pool"],
                            "scale_factor": d["scale_factor"],
                            "status": d["status"],
                            "failing_gate": d["failing_gate"],
                            "warns": list(d["warns"]),
                            "qc": dict(d["qc"]),
                        }
                        for d in entry["decisions"]
                    ],
                }
                for entry in tier_gate_decisions
            ]
        if tier_result is not None:
            qc_metrics["aif_gate_decisions"] = [
                {
                    "tier": d["tier"],
                    "n_pool": d["n_pool"],
                    "scale_factor": d["scale_factor"],
                    "status": d["status"],
                    "failing_gate": d["failing_gate"],
                    "warns": list(d["warns"]),
                    "qc": dict(d["qc"]),
                }
                for d in tier_result.gate_decisions
            ]
        # Confidence describes the EMITTED curve, not the (possibly discarded)
        # detected pool. A population emit is always "low" even when a detected
        # tier_result is forwarded for provenance — the post-cascade quality
        # filter can empty the detected pool after its tier passed the aggregate
        # gates, so tier_result being non-None does not mean a detected AIF was
        # emitted. Check detection_method first.
        if detection_method == "population":
            qc_metrics["aif_confidence"] = "low"
            qc_metrics["aif_g6_admitted"] = False
            qc_metrics["aif_uncertainty_inflation_factor"] = 3.0
        elif tier_result is not None:
            if tier_result.status == "G6_ADMIT":
                qc_metrics["aif_confidence"] = "reduced"
                qc_metrics["aif_g6_admitted"] = True
                sf = tier_result.required_scale_factor
                if sf > DCE_AIF_GATE_SCALE_G6_HIGH:
                    qc_metrics["aif_uncertainty_inflation_factor"] = 3.0
                else:
                    qc_metrics["aif_uncertainty_inflation_factor"] = 2.0
            elif tier_result.status in ("PASS", "PASS_WARN"):
                qc_metrics["aif_confidence"] = "standard"
                qc_metrics["aif_g6_admitted"] = False
                qc_metrics["aif_uncertainty_inflation_factor"] = 1.0
            else:
                # C2 re-gate of the emitted subset FAILED a gate: emit the
                # detected AIF (no new fallback) but downgrade confidence and
                # inflate uncertainty honestly. Only reachable via the divergence
                # re-gate; the cascade itself only ever sets PASS/PASS_WARN/G6.
                qc_metrics["aif_confidence"] = "reduced"
                qc_metrics["aif_g6_admitted"] = False
                qc_metrics["aif_uncertainty_inflation_factor"] = 2.0
        else:
            qc_metrics["aif_confidence"] = "standard"
            qc_metrics["aif_g6_admitted"] = False
            qc_metrics["aif_uncertainty_inflation_factor"] = 1.0

        # Amplitude-anchor the Pass-2 fallback AIF to the dose-scaled population
        # target via the shared _apply_aif_scale helper (identical math to the
        # per-tier trim loop and the C2 re-gate). External AIFs bypass (supplied
        # pre-scaled in plasma mM, Hct-corrected — see README); the gated cascade
        # path bypasses (already scaled inside the trim loop, scale_factor recorded).
        if detection_method not in ("population", "external") and tier_result is None:
            current_peak = float(np.max(aif_curve))
            scale = _resolve_aif_scale(
                aif_curve,
                float(self.inputs.hematocrit),
                target_peak,
                self._get_expected_peak_range,
                apply_peak_floor=bool(getattr(self.inputs, "aif_peak_floor", True)),
            )
            aif_curve = scale.curve
            scaling_factor = scale.applied_factor
            required_scaling_factor = scale.required_factor
            calibration_mode = scale.mode
            # required_scaling_factor is a pure blood-amplitude ratio (population
            # floor / detected peak): the 1/(1-Hct) plasma conversion cancels, so
            # these WARN thresholds remain fixed and Hct-invariant even when the
            # configured policy preserves the measured amplitude.
            if required_scaling_factor > DCE_AIF_SCALE_WARN_UNRELIABLE:
                logger.warning(
                    "AIF amplitude-calibration factor required %.2f > %.1f — detected AIF peak "
                    "(%.2f mM plasma) is far below expected arterial range. "
                    "Results may be unreliable.",
                    required_scaling_factor,
                    DCE_AIF_SCALE_WARN_UNRELIABLE,
                    current_peak,
                )
            elif required_scaling_factor > DCE_AIF_SCALE_WARN_SUBSTANTIAL:
                logger.warning(
                    "AIF amplitude-calibration factor required %.2f > %.1f — detected AIF peak "
                    "(%.2f mM plasma) is substantially below expected range.",
                    required_scaling_factor,
                    DCE_AIF_SCALE_WARN_SUBSTANTIAL,
                    current_peak,
                )

        # Compute final AIF metrics
        final_peak = compute_dce_aif_peak_concentration(aif_curve)
        final_fwhm = compute_dce_aif_fwhm_seconds(aif_curve, tr, baseline_frames)
        final_auc = compute_dce_aif_auc(aif_curve, tr)
        final_washout = compute_dce_aif_washout(aif_curve, baseline_frames)

        # Peak-amplitude QC warning. final_peak is PLASMA (post-Hct division at
        # the scaling site), but _get_expected_peak_range returns BLOOD-scale
        # (raw Parker), so convert it to plasma (/(1-Hct)) before comparing —
        # mirroring the scaling sites. Suppressed for population fallback
        # (population AIF is tautologically within its own expected range).
        _, expected_max_blood = self._get_expected_peak_range()
        _hct = float(self.inputs.hematocrit)
        expected_max_mm = (
            expected_max_blood / (1.0 - _hct) if _hct < 1.0 else expected_max_blood
        )
        peak_amplitude_ratio, peak_amplitude_warning = (
            compute_dce_aif_peak_amplitude_warning(
                final_peak, expected_max_mm, detection_method,
            )
        )
        if peak_amplitude_warning:
            logger.warning(
                "DCE AIF peak %.2f mM exceeds %.1fx dose-scaled expected "
                "max (%.2f mM); possible inflow contamination or motion "
                "artifact.",
                final_peak,
                DCE_AIF_PEAK_RATIO_WARN_THRESHOLD,
                expected_max_mm,
            )

        # Multi-bump tail detector. Physiologic recirculation produces
        # exactly one secondary maximum (~30s post-peak); count > 1
        # indicates motion / inflow / venous mixing. Suppressed for
        # population fallback.
        if detection_method != "population":
            multi_bump_count, multi_bump_heights = (
                detect_dce_aif_multi_bump_tail(
                    aif_curve, tr=tr, baseline_frames=baseline_frames,
                )
            )
        else:
            multi_bump_count, multi_bump_heights = 0, []
        multi_bump_warning = multi_bump_count > 1
        if multi_bump_warning:
            logger.warning(
                "DCE AIF post-peak tail has %d secondary maxima > %.0f%% "
                "of first-pass peak (>1 expected from physiologic "
                "recirculation); possible motion/inflow artifact.",
                multi_bump_count,
                DCE_AIF_TAIL_PEAK_FRACTION * 100,
            )

        # Compute AIF status for HTML report from the gate-based confidence,
        # NOT the report-only fidelity score (quality_score). The fidelity
        # metric is systematically compressed (~0.26 for essentially every
        # detected AIF, from the known MCA-tail under-recovery) and is
        # physics-scoped as report-only, so thresholding it mislabels a normal
        # detected AIF as POOR. aif_confidence already encodes the G1-G6 gate
        # outcome: "standard" (gates passed) / "reduced" (G6-admitted or C2
        # re-gate fail) / "low" (population fallback).
        from pmriprep.utils.qc_standards import QCStatus
        _aif_confidence = qc_metrics.get("aif_confidence", "reduced")
        if _aif_confidence == "standard":
            aif_status = QCStatus.PASS
        else:  # "reduced" (marginal / G6-admitted / C2 re-gate fail) or
            # "low" (population fallback) — flag for review, not certify
            aif_status = QCStatus.WARN

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
                "scaling_factor_warning": scaling_factor > DCE_AIF_SCALE_WARN_SUBSTANTIAL,
                "required_scaling_factor": required_scaling_factor,
                "required_scaling_factor_warning": (
                    required_scaling_factor > DCE_AIF_SCALE_WARN_SUBSTANTIAL
                ),
                "aif_amplitude_calibration": calibration_mode,
                "detection_method": detection_method,
                "quality_score": quality_score,
                "baseline_frames": baseline_frames,
                "aif_status": aif_status,
                "hematocrit": float(self.inputs.hematocrit),
                "expected_peak_max_mm": float(expected_max_mm),
                "peak_amplitude_ratio": (
                    None if not np.isfinite(peak_amplitude_ratio)
                    else float(peak_amplitude_ratio)
                ),
                "peak_amplitude_warning": bool(peak_amplitude_warning),
                "multi_bump_tail_count": int(multi_bump_count),
                "multi_bump_tail_heights_mm": multi_bump_heights,
                "multi_bump_tail_warning": bool(multi_bump_warning),
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
            "aif_confidence": str(qc_metrics.get("aif_confidence", "standard")),
            "aif_uncertainty_inflation_factor": float(
                qc_metrics.get("aif_uncertainty_inflation_factor", 1.0)
            ),
        }

        return runtime

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE AIF detection.

        Notes
        -----
        Two distinct gate-decision QC fields are emitted to the
        ``Metrics.aif_detection`` block of the per-session sidecar:

        - ``aif_gate_decisions``: per-iteration ``GateDecision`` log of
          the **winning** tier's progressive trim loop. Empty when no
          tier wins (population fallback). Consumers reading this for
          QC pass/fail summaries see only the accepted pool's history.
        - ``cascade_gate_decisions``: per-tier outcome list, one entry
          per attempted tier in cascade order (atlas tightest →
          broadest × strict → permissive, then ``wb_relaxed``). Each
          entry has ``tier`` (label), ``outcome`` (``"PASS"`` /
          ``"trim_exhausted"`` / ``"skipped_quorum"`` /
          ``"skipped_anchor"``), and ``decisions`` (the full
          per-iteration ``GateDecision`` log; empty when the tier was
          skipped before gates ran). This field captures the
          diagnostic trail for tiers whose pool was discarded — needed
          for diagnosing cascades that exhaust to population.

        Order of ``cascade_gate_decisions`` matches
        ``cascade_anchor_outcomes`` and is asserted in
        ``tests/unit/test_dce_aif.py``.
        """
        loaded = self._load_and_validate_inputs()
        if isdefined(self.inputs.external_aif_file) and self.inputs.external_aif_file:
            return self._handle_external_aif(
                runtime, loaded.conc_data, loaded.affine, loaded.brain_mask,
            )
        atlas_masks, atlas_names, lesion_mask = self._load_atlas_and_lesion_masks()
        params = self._read_params(loaded.conc_data, loaded.header)

        if params.aif_method == "population":
            state = DetectionState.population_default()
        else:
            state = self._run_detection(
                loaded, atlas_masks, atlas_names, lesion_mask, params,
            )

        if state.candidates:
            emit = self._emit_detected_aif(state, loaded, params)
        else:
            emit = self._emit_population_aif(state, loaded, params)

        return self._finalize_and_save(runtime, emit, state, loaded, params)

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
