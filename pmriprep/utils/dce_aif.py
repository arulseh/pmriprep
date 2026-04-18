"""DCE AIF (Arterial Input Function) quality assessment utilities.

This module provides utilities for assessing the quality of DCE AIF curves,
including comparison to Parker population AIF parameters and comprehensive
quality metrics for DCE-specific requirements.

The DCE AIF differs from DSC AIF in several important ways:
- Concentration is in mM (millimolar) not relative signal change
- Expected peak values are in physiological range (5-7.6 mM for full dose)
- FWHM is measured in seconds (typically 8-25s) rather than frames
- Quality assessment compares to Parker et al. (2006) population parameters
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_constants import (
    AIF_THRESHOLDS_DEGRADED,
    AIF_THRESHOLDS_PERMISSIVE,
    AIF_THRESHOLDS_RELAXED,
    AIF_THRESHOLDS_STRICT,
    DCE_AIF_EXPECTED_FULL_DOSE,
    EPSILON_CONCENTRATION,
    EPSILON_T1,
    get_aif_thresholds,
)
from pmriprep.utils.qc_standards import QCStatus


# =============================================================================
# Parker Population AIF Reference Parameters
# =============================================================================
# Based on Parker et al. (2006) "Experimentally-derived functional form
# for a population-averaged high-temporal-resolution arterial input
# function for dynamic contrast-enhanced MRI"


class ParkerAIFExpected:
    """Expected AIF parameters from Parker population AIF (2006).

    These values are used as reference for quality assessment.
    Reference is for 0.1 mmol/kg dose of Gd contrast at 3T.
    """

    # Peak concentration (mM) for full dose (0.1 mmol/kg)
    PEAK_MIN_MM: float = 5.0
    PEAK_MAX_MM: float = 7.6
    PEAK_TYPICAL_MM: float = 6.3

    # FWHM (seconds) - typical bolus width
    FWHM_MIN_SEC: float = 8.0
    FWHM_MAX_SEC: float = 25.0
    FWHM_TYPICAL_SEC: float = 15.0

    # Washout fraction (fraction of peak that washes out)
    WASHOUT_MIN_STRICT: float = 0.20
    WASHOUT_MIN_RELAXED: float = 0.10
    WASHOUT_TYPICAL: float = 0.35

    # Baseline stability (standard deviation should be small relative to peak)
    BASELINE_STD_MAX_FRACTION: float = 0.05  # Max 5% of peak

    # AUC (mM·s) - area under curve for full dose
    AUC_MIN: float = 30.0
    AUC_MAX: float = 120.0
    AUC_TYPICAL: float = 70.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DCEAIFQualityMetrics:
    """Individual quality metrics for a DCE AIF curve.

    Attributes
    ----------
    peak_concentration : float
        Peak concentration in mM.
    fwhm_seconds : float
        Full width at half maximum in seconds.
    washout_fraction : float
        Washout fraction (0-1), higher indicates better arterial washout.
    baseline_std : float
        Standard deviation of baseline in mM.
    baseline_mean : float
        Mean baseline value in mM (should be ~0 for pre-contrast).
    cnr : float
        Contrast-to-noise ratio (peak / baseline_std).
    auc : float
        Area under the curve in mM·s.
    time_to_peak : float
        Time to peak in seconds from start of acquisition.
    """

    peak_concentration: float
    fwhm_seconds: float
    washout_fraction: float
    baseline_std: float
    baseline_mean: float
    cnr: float
    auc: float
    time_to_peak: float


@dataclass
class DCEAIFQualityResult:
    """Result of DCE AIF quality assessment.

    Attributes
    ----------
    status : str
        Overall QC status: "PASS", "WARN", or "POOR".
    quality_score : float
        Overall quality score (0-1).
    metrics : DCEAIFQualityMetrics
        Individual computed metrics.
    meets_strict : bool
        Whether the AIF meets strict quality criteria.
    meets_relaxed : bool
        Whether the AIF meets relaxed quality criteria.
    criteria_details : dict[str, str]
        Per-metric pass/warn/fail status details.
    comparison_to_parker : dict[str, Any]
        Comparison of metrics to Parker population AIF.
    validation_notes : list[str]
        List of validation notes or warnings.
    """

    status: str
    quality_score: float
    metrics: DCEAIFQualityMetrics
    meets_strict: bool
    meets_relaxed: bool
    criteria_details: dict[str, str] = field(default_factory=dict)
    comparison_to_parker: dict[str, Any] = field(default_factory=dict)
    validation_notes: list[str] = field(default_factory=list)


# =============================================================================
# Core Metric Functions
# =============================================================================


def compute_dce_peak_concentration(aif_curve: npt.NDArray[Any]) -> float:
    """Compute peak concentration of DCE AIF in mM.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.

    Returns
    -------
    float
        Peak concentration in mM.
    """
    return float(np.max(aif_curve))


def compute_dce_fwhm_seconds(
    aif_curve: npt.NDArray[Any],
    tr: float,
    baseline_frames: int = 5,
) -> float:
    """Compute FWHM of DCE AIF peak in seconds.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.
    tr : float
        Temporal resolution in seconds.
    baseline_frames : int
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


def compute_dce_washout_fraction(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
    tail_frames: int = 5,
) -> float:
    """Compute washout fraction for DCE AIF.

    Washout fraction = (peak - tail_mean) / (peak - baseline_mean)
    A value of 1.0 indicates complete washout to baseline.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.
    baseline_frames : int
        Number of pre-contrast baseline frames.
    tail_frames : int
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
    if denom < EPSILON_T1:
        return 0.0

    washout = (peak - tail_mean) / denom
    return max(0.0, min(1.0, washout))


def compute_dce_baseline_std(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute baseline standard deviation for DCE AIF.

    For DCE, baseline should be approximately zero (pre-contrast).

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.
    baseline_frames : int
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


def compute_dce_baseline_mean(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute baseline mean for DCE AIF.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.
    baseline_frames : int
        Number of pre-contrast baseline frames.

    Returns
    -------
    float
        Mean baseline value in mM.
    """
    if len(aif_curve) < baseline_frames:
        return 0.0

    baseline = aif_curve[:baseline_frames]
    return float(np.mean(baseline))


def compute_dce_cnr(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute contrast-to-noise ratio for DCE AIF.

    CNR = peak / baseline_std

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.
    baseline_frames : int
        Number of pre-contrast baseline frames.

    Returns
    -------
    float
        Contrast-to-noise ratio.
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0.0

    baseline_std = compute_dce_baseline_std(aif_curve, baseline_frames)

    if baseline_std < EPSILON_CONCENTRATION:
        # Near-zero baseline noise, return high CNR
        return 100.0

    peak = float(np.max(aif_curve))
    cnr = peak / baseline_std

    return max(0.0, cnr)


def compute_dce_auc(
    aif_curve: npt.NDArray[Any],
    tr: float,
) -> float:
    """Compute area under the curve for DCE AIF.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.
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


def compute_dce_time_to_peak(
    aif_curve: npt.NDArray[Any],
    tr: float,
) -> float:
    """Compute time to peak for DCE AIF.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.
    tr : float
        Temporal resolution in seconds.

    Returns
    -------
    float
        Time to peak in seconds from start.
    """
    peak_idx = int(np.argmax(aif_curve))
    return peak_idx * tr


# =============================================================================
# Quality Assessment
# =============================================================================


def assess_dce_aif_quality(
    aif_curve: npt.NDArray[Any],
    tr: float,
    baseline_frames: int = 5,
    criteria: Literal["strict", "relaxed", "permissive", "degraded"] = "strict",
    dose_level: float = 1.0,
    field_strength: float = 3.0,
) -> DCEAIFQualityResult:
    """Assess the quality of a DCE AIF curve and return pass/warn/fail status.

    This function evaluates a DCE AIF curve against four criteria levels
    and provides a comprehensive quality assessment. Metrics are compared to
    the Parker et al. (2006) population AIF as a reference.

    Quality is assessed based on these key metrics:
    - Peak concentration: Should match expected range for dose (5-7.6 mM for full)
    - FWHM: Should be in physiological range (8-25 seconds)
    - Washout: Should show adequate washout (≥0.2 strict, ≥0.1 relaxed)
    - Baseline stability: Pre-contrast baseline should have low noise
    - CNR: Sufficient contrast-to-noise for reliable curve fitting

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course in mM.
    tr : float
        Temporal resolution (TR) in seconds.
    baseline_frames : int
        Number of pre-contrast baseline frames (default: 5).
    criteria : Literal["strict", "relaxed", "permissive", "degraded"]
        Primary criteria level for overall status determination.
        All four levels are evaluated regardless.
    dose_level : float
        Contrast dose level, affects expected peak range.
    field_strength : float
        Magnetic field strength in Tesla (default: 3.0).

    Returns
    -------
    DCEAIFQualityResult
        Comprehensive quality assessment with status, score, and per-metric details.

    Notes
    -----
    Overall status is determined as follows:
    - PASS: All metrics meet the specified criteria level thresholds
    - WARN: Some metrics fail specified criteria but all meet relaxed thresholds
    - POOR: One or more metrics fail even relaxed thresholds

    Expected AIF characteristics for full dose (0.1 mmol/kg) at 3T:
    - Peak concentration: 5.0-7.6 mM
    - FWHM: 8-25 seconds
    - Baseline should be ~0 mM (pre-contrast)
    - Washout fraction: ≥0.2 (strict) or ≥0.1 (relaxed)

    Comparison to Parker population AIF:
    - The Parker model is the reference standard for DCE AIF
    - Deviations from Parker parameters may indicate:
      - Partial volume effects (lower peak)
      - Timing issues (shifted peak)
      - Venous contamination (slow washout)
      - Technical problems (high baseline noise)

    References
    ----------
    Parker GJM et al. "Experimentally-derived functional form for a
    population-averaged high-temporal-resolution arterial input function
    for dynamic contrast-enhanced MRI." Magn Reson Med. 2006;56:993-1000.

    Examples
    --------
    >>> import numpy as np
    >>> # Assess a DCE AIF curve with TR=5s
    >>> result = assess_dce_aif_quality(aif_curve, tr=5.0, baseline_frames=5)
    >>> print(f"Status: {result.status}, Score: {result.quality_score:.2f}")
    >>> print(f"Peak: {result.metrics.peak_concentration:.2f} mM")
    >>> print(f"FWHM: {result.metrics.fwhm_seconds:.1f} s")
    """
    # Compute all metrics
    peak = compute_dce_peak_concentration(aif_curve)
    fwhm = compute_dce_fwhm_seconds(aif_curve, tr, baseline_frames)
    washout = compute_dce_washout_fraction(aif_curve, baseline_frames)
    baseline_std = compute_dce_baseline_std(aif_curve, baseline_frames)
    baseline_mean = compute_dce_baseline_mean(aif_curve, baseline_frames)
    cnr = compute_dce_cnr(aif_curve, baseline_frames)
    auc = compute_dce_auc(aif_curve, tr)
    time_to_peak = compute_dce_time_to_peak(aif_curve, tr)

    # Store metrics
    metrics = DCEAIFQualityMetrics(
        peak_concentration=peak,
        fwhm_seconds=fwhm,
        washout_fraction=washout,
        baseline_std=baseline_std,
        baseline_mean=baseline_mean,
        cnr=cnr,
        auc=auc,
        time_to_peak=time_to_peak,
    )

    # Dose level IS the multiplier
    dose_factor = float(dose_level)

    # Get expected peak range for dose
    expected_peak_min = ParkerAIFExpected.PEAK_MIN_MM * dose_factor
    expected_peak_max = ParkerAIFExpected.PEAK_MAX_MM * dose_factor
    expected_peak_typical = ParkerAIFExpected.PEAK_TYPICAL_MM * dose_factor

    # Get thresholds for all 4 levels (TR-adaptive)
    strict_thresholds = get_aif_thresholds("strict", tr=tr)
    relaxed_thresholds = get_aif_thresholds("relaxed", tr=tr)
    permissive_thresholds = get_aif_thresholds("permissive", tr=tr)
    degraded_thresholds = get_aif_thresholds("degraded", tr=tr)

    # Evaluate against strict thresholds
    strict_pass = _evaluate_dce_thresholds(
        metrics=metrics,
        thresholds=strict_thresholds,
        expected_peak_min=expected_peak_min,
        expected_peak_max=expected_peak_max,
    )

    # Evaluate against relaxed thresholds
    relaxed_pass = _evaluate_dce_thresholds(
        metrics=metrics,
        thresholds=relaxed_thresholds,
        expected_peak_min=expected_peak_min * 0.3,  # Allow lower due to PV effects
        expected_peak_max=expected_peak_max * 3.0,  # Allow higher for some tolerance
    )

    # Evaluate against permissive thresholds
    permissive_pass = _evaluate_dce_thresholds(
        metrics=metrics,
        thresholds=permissive_thresholds,
        expected_peak_min=expected_peak_min * 0.15,
        expected_peak_max=expected_peak_max * 5.0,
    )

    # Evaluate against degraded thresholds
    degraded_pass = _evaluate_dce_thresholds(
        metrics=metrics,
        thresholds=degraded_thresholds,
        expected_peak_min=expected_peak_min * 0.1,
        expected_peak_max=expected_peak_max * 10.0,
    )

    # Determine per-metric status for detailed reporting
    criteria_details = _get_dce_criteria_details(
        metrics=metrics,
        strict_thresholds=strict_thresholds,
        relaxed_thresholds=relaxed_thresholds,
        expected_peak_min=expected_peak_min,
        expected_peak_max=expected_peak_max,
    )

    # Compute comparison to Parker AIF
    comparison_to_parker = _compare_to_parker(
        metrics=metrics,
        dose_factor=dose_factor,
    )

    # Build validation notes
    validation_notes = _generate_validation_notes(
        metrics=metrics,
        comparison_to_parker=comparison_to_parker,
        dose_factor=dose_factor,
    )

    # Compute quality score (0-1)
    quality_score = _compute_dce_quality_score(
        metrics=metrics,
        expected_peak_typical=expected_peak_typical,
    )

    # Determine overall status based on requested criteria level
    # Cascade: if criteria=strict, PASS requires strict, WARN allows relaxed
    # If criteria=permissive, PASS requires permissive, WARN allows degraded
    pass_map = {
        "strict": strict_pass,
        "relaxed": relaxed_pass,
        "permissive": permissive_pass,
        "degraded": degraded_pass,
    }
    # The "next easier" level for WARN fallback
    fallback_map = {
        "strict": relaxed_pass,
        "relaxed": permissive_pass,
        "permissive": degraded_pass,
        "degraded": degraded_pass,
    }

    if pass_map[criteria]:
        status = QCStatus.PASS
    elif fallback_map[criteria]:
        status = QCStatus.WARN
    else:
        status = QCStatus.POOR

    return DCEAIFQualityResult(
        status=status,
        quality_score=quality_score,
        metrics=metrics,
        meets_strict=strict_pass,
        meets_relaxed=relaxed_pass,
        criteria_details=criteria_details,
        comparison_to_parker=comparison_to_parker,
        validation_notes=validation_notes,
    )


def _evaluate_dce_thresholds(
    metrics: DCEAIFQualityMetrics,
    thresholds: Any,
    expected_peak_min: float,
    expected_peak_max: float,
) -> bool:
    """Check if all metrics meet the specified thresholds.

    Parameters
    ----------
    metrics : DCEAIFQualityMetrics
        Computed metrics.
    thresholds : AIFThresholds
        Threshold values.
    expected_peak_min : float
        Minimum expected peak concentration.
    expected_peak_max : float
        Maximum expected peak concentration.

    Returns
    -------
    bool
        True if all metrics meet thresholds, False otherwise.
    """
    # Check washout
    if metrics.washout_fraction < thresholds.washout_min:
        return False

    # Check FWHM bounds (in seconds)
    if metrics.fwhm_seconds < thresholds.fwhm_min_seconds:
        return False
    if metrics.fwhm_seconds > thresholds.fwhm_max_seconds:
        return False

    # Check peak concentration is in physiological range
    if metrics.peak_concentration < expected_peak_min:
        return False
    if metrics.peak_concentration > expected_peak_max:
        return False

    # Check CNR (should be reasonably high for reliable curve fitting)
    min_cnr = 5.0 if thresholds.washout_min >= 0.20 else 3.0  # strict vs relaxed
    if metrics.cnr < min_cnr:
        return False

    return True


def _get_dce_criteria_details(
    metrics: DCEAIFQualityMetrics,
    strict_thresholds: Any,
    relaxed_thresholds: Any,
    expected_peak_min: float,
    expected_peak_max: float,
) -> dict[str, str]:
    """Get per-metric pass/warn/fail status details.

    Parameters
    ----------
    metrics : DCEAIFQualityMetrics
        Computed metrics.
    strict_thresholds : AIFThresholds
        Strict threshold values.
    relaxed_thresholds : AIFThresholds
        Relaxed threshold values.
    expected_peak_min : float
        Minimum expected peak concentration.
    expected_peak_max : float
        Maximum expected peak concentration.

    Returns
    -------
    dict[str, str]
        Dictionary mapping metric name to status string.
    """
    details: dict[str, str] = {}

    # Peak concentration evaluation
    if expected_peak_min <= metrics.peak_concentration <= expected_peak_max:
        details["peak_concentration"] = QCStatus.PASS
    elif expected_peak_min * 0.3 <= metrics.peak_concentration <= expected_peak_max * 3.0:
        details["peak_concentration"] = QCStatus.WARN
    else:
        details["peak_concentration"] = QCStatus.POOR

    # Washout evaluation (higher is better)
    if metrics.washout_fraction >= strict_thresholds.washout_min:
        details["washout"] = QCStatus.PASS
    elif metrics.washout_fraction >= relaxed_thresholds.washout_min:
        details["washout"] = QCStatus.WARN
    else:
        details["washout"] = QCStatus.POOR

    # FWHM evaluation (within range is better)
    if (
        strict_thresholds.fwhm_min_seconds
        <= metrics.fwhm_seconds
        <= strict_thresholds.fwhm_max_seconds
    ):
        details["fwhm"] = QCStatus.PASS
    elif (
        relaxed_thresholds.fwhm_min_seconds
        <= metrics.fwhm_seconds
        <= relaxed_thresholds.fwhm_max_seconds
    ):
        details["fwhm"] = QCStatus.WARN
    else:
        details["fwhm"] = QCStatus.POOR

    # CNR evaluation (higher is better)
    if metrics.cnr >= 5.0:
        details["cnr"] = QCStatus.PASS
    elif metrics.cnr >= 3.0:
        details["cnr"] = QCStatus.WARN
    else:
        details["cnr"] = QCStatus.POOR

    # Baseline evaluation (lower std is better)
    baseline_noise_fraction = metrics.baseline_std / max(
        metrics.peak_concentration, EPSILON_T1
    )
    if baseline_noise_fraction <= ParkerAIFExpected.BASELINE_STD_MAX_FRACTION:
        details["baseline"] = QCStatus.PASS
    elif baseline_noise_fraction <= ParkerAIFExpected.BASELINE_STD_MAX_FRACTION * 2:
        details["baseline"] = QCStatus.WARN
    else:
        details["baseline"] = QCStatus.POOR

    # AUC evaluation
    if ParkerAIFExpected.AUC_MIN <= metrics.auc <= ParkerAIFExpected.AUC_MAX:
        details["auc"] = QCStatus.PASS
    elif ParkerAIFExpected.AUC_MIN * 0.3 <= metrics.auc <= ParkerAIFExpected.AUC_MAX * 3.0:
        details["auc"] = QCStatus.WARN
    else:
        details["auc"] = QCStatus.POOR

    return details


def _compare_to_parker(
    metrics: DCEAIFQualityMetrics,
    dose_factor: float,
) -> dict[str, Any]:
    """Compare metrics to Parker population AIF reference.

    Parameters
    ----------
    metrics : DCEAIFQualityMetrics
        Computed metrics.
    dose_factor : float
        Dose scaling factor (0.5 for half, 1.0 for full, 2.0 for double).

    Returns
    -------
    dict[str, Any]
        Comparison results.
    """
    expected_peak = ParkerAIFExpected.PEAK_TYPICAL_MM * dose_factor
    expected_fwhm = ParkerAIFExpected.FWHM_TYPICAL_SEC
    expected_auc = ParkerAIFExpected.AUC_TYPICAL * dose_factor

    # Compute deviations from expected
    peak_deviation = (metrics.peak_concentration - expected_peak) / expected_peak
    fwhm_deviation = (metrics.fwhm_seconds - expected_fwhm) / expected_fwhm
    auc_deviation = (metrics.auc - expected_auc) / expected_auc if expected_auc > 0 else 0.0

    return {
        "expected_peak_mm": expected_peak,
        "expected_fwhm_sec": expected_fwhm,
        "expected_auc": expected_auc,
        "peak_deviation_fraction": peak_deviation,
        "fwhm_deviation_fraction": fwhm_deviation,
        "auc_deviation_fraction": auc_deviation,
        "dose_factor": dose_factor,
        "peak_within_expected": (
            ParkerAIFExpected.PEAK_MIN_MM * dose_factor
            <= metrics.peak_concentration
            <= ParkerAIFExpected.PEAK_MAX_MM * dose_factor
        ),
        "fwhm_within_expected": (
            ParkerAIFExpected.FWHM_MIN_SEC
            <= metrics.fwhm_seconds
            <= ParkerAIFExpected.FWHM_MAX_SEC
        ),
    }


def _generate_validation_notes(
    metrics: DCEAIFQualityMetrics,
    comparison_to_parker: dict[str, Any],
    dose_factor: float,
) -> list[str]:
    """Generate validation notes and warnings.

    Parameters
    ----------
    metrics : DCEAIFQualityMetrics
        Computed metrics.
    comparison_to_parker : dict[str, Any]
        Parker comparison results.
    dose_factor : float
        Dose scaling factor.

    Returns
    -------
    list[str]
        List of validation notes.
    """
    notes: list[str] = []

    # Peak concentration notes
    peak_dev = comparison_to_parker["peak_deviation_fraction"]
    if peak_dev < -0.3:
        notes.append(
            f"Peak concentration ({metrics.peak_concentration:.2f} mM) is "
            f"significantly lower than expected. Possible partial volume effects."
        )
    elif peak_dev > 0.5:
        notes.append(
            f"Peak concentration ({metrics.peak_concentration:.2f} mM) is "
            f"higher than expected. Verify AIF voxel selection."
        )

    # FWHM notes
    if metrics.fwhm_seconds < ParkerAIFExpected.FWHM_MIN_SEC:
        notes.append(
            f"FWHM ({metrics.fwhm_seconds:.1f}s) is narrower than expected. "
            f"May indicate fast acquisition or truncated bolus."
        )
    elif metrics.fwhm_seconds > ParkerAIFExpected.FWHM_MAX_SEC:
        notes.append(
            f"FWHM ({metrics.fwhm_seconds:.1f}s) is wider than expected. "
            f"May indicate venous contamination or dispersion."
        )

    # Washout notes
    if metrics.washout_fraction < ParkerAIFExpected.WASHOUT_MIN_STRICT:
        notes.append(
            f"Washout fraction ({metrics.washout_fraction:.2f}) is low. "
            f"May indicate venous contamination or recirculation."
        )

    # Baseline notes
    baseline_noise_frac = metrics.baseline_std / max(
        metrics.peak_concentration, EPSILON_T1
    )
    if baseline_noise_frac > ParkerAIFExpected.BASELINE_STD_MAX_FRACTION * 2:
        notes.append(
            f"High baseline noise ({metrics.baseline_std:.3f} mM). "
            f"May affect concentration accuracy."
        )

    # Baseline mean notes (should be near zero)
    if abs(metrics.baseline_mean) > 0.5:
        notes.append(
            f"Baseline mean ({metrics.baseline_mean:.3f} mM) is not near zero. "
            f"Verify baseline subtraction or pre-contrast timing."
        )

    # CNR notes
    if metrics.cnr < 5.0:
        notes.append(
            f"CNR ({metrics.cnr:.1f}) is marginal. "
            f"Curve fitting may have reduced reliability."
        )

    # AUC notes (proportional to dose)
    expected_auc = ParkerAIFExpected.AUC_TYPICAL * dose_factor
    if metrics.auc < expected_auc * 0.3:
        notes.append(
            f"AUC ({metrics.auc:.1f} mM·s) is much lower than expected. "
            f"May indicate incomplete bolus capture or signal loss."
        )
    elif metrics.auc > expected_auc * 3.0:
        notes.append(
            f"AUC ({metrics.auc:.1f} mM·s) is much higher than expected. "
            f"May indicate dose error or signal contamination."
        )

    return notes


def _compute_dce_quality_score(
    metrics: DCEAIFQualityMetrics,
    expected_peak_typical: float,
) -> float:
    """Compute overall quality score (0-1) based on metrics.

    The score is a weighted combination of individual metric scores:
    - Peak concentration: 30% weight (key for PK modeling)
    - CNR: 25% weight (important for reliability)
    - Washout: 25% weight (indicates arterial origin)
    - FWHM: 15% weight (indicates bolus quality)
    - Baseline: 5% weight (noise assessment)

    Parameters
    ----------
    metrics : DCEAIFQualityMetrics
        Computed metrics.
    expected_peak_typical : float
        Expected typical peak concentration for the dose.

    Returns
    -------
    float
        Quality score between 0 and 1.
    """
    score = 0.0

    # Peak contribution (0-0.3): closer to expected is better
    peak_deviation = abs(metrics.peak_concentration - expected_peak_typical)
    peak_deviation_norm = peak_deviation / expected_peak_typical
    peak_score = max(0.0, 1.0 - peak_deviation_norm)
    score += 0.3 * peak_score

    # CNR contribution (0-0.25): higher is better
    # Normalize: 3 = marginal, 10+ = excellent
    cnr_score = min(1.0, max(0.0, (metrics.cnr - 3.0) / 7.0))
    score += 0.25 * cnr_score

    # Washout contribution (0-0.25): closer to 0.35-0.5 is ideal
    ideal_washout = 0.4
    washout_deviation = abs(metrics.washout_fraction - ideal_washout)
    washout_score = max(0.0, 1.0 - washout_deviation / 0.4)
    score += 0.25 * washout_score

    # FWHM contribution (0-0.15): closer to ~15s is ideal
    ideal_fwhm = ParkerAIFExpected.FWHM_TYPICAL_SEC
    fwhm_range = ParkerAIFExpected.FWHM_MAX_SEC - ParkerAIFExpected.FWHM_MIN_SEC
    fwhm_deviation = abs(metrics.fwhm_seconds - ideal_fwhm)
    fwhm_score = max(0.0, 1.0 - fwhm_deviation / (fwhm_range / 2.0))
    score += 0.15 * fwhm_score

    # Baseline contribution (0-0.05): lower noise is better
    baseline_noise_frac = metrics.baseline_std / max(
        metrics.peak_concentration, EPSILON_T1
    )
    baseline_score = max(
        0.0, 1.0 - baseline_noise_frac / ParkerAIFExpected.BASELINE_STD_MAX_FRACTION
    )
    score += 0.05 * baseline_score

    return min(1.0, score)


# =============================================================================
# File-Based Wrapper
# =============================================================================


def assess_dce_aif_quality_from_file(
    aif_file: str,
    tr: float,
    baseline_frames: int = 5,
    criteria: Literal["strict", "relaxed", "permissive", "degraded"] = "strict",
    dose_level: float = 1.0,
    field_strength: float = 3.0,
) -> DCEAIFQualityResult:
    """Assess DCE AIF quality from a text file.

    Convenience wrapper that loads AIF curve from file.

    Parameters
    ----------
    aif_file : str
        Path to text file containing 1D AIF curve in mM.
    tr : float
        Temporal resolution (TR) in seconds.
    baseline_frames : int
        Number of pre-contrast baseline frames.
    criteria : Literal["strict", "relaxed"]
        Primary criteria level for overall status.
    dose_level : float
        Contrast dose level.
    field_strength : float
        Magnetic field strength in Tesla.

    Returns
    -------
    DCEAIFQualityResult
        Quality assessment result.
    """
    aif_curve = np.loadtxt(aif_file)

    return assess_dce_aif_quality(
        aif_curve=aif_curve,
        tr=tr,
        baseline_frames=baseline_frames,
        criteria=criteria,
        dose_level=dose_level,
        field_strength=field_strength,
    )


