"""AIF (Arterial Input Function) detection utilities for DSC-MRI.

This module provides utilities for atlas-based AIF detection, candidate
scoring, and quality assessment for DSC perfusion processing.

The atlas-based search constrains AIF detection to anatomically plausible
arterial regions (e.g., MCA territory) and scores candidates based on
arterial curve characteristics: high peak, sharp washout, narrow FWHM.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import optimize

from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
    LESION_EXCLUSION_DISTANCE_MM,
)
from pmriprep.utils.qc_standards import QCStatus


# =============================================================================
# Constants for AIF Detection
# =============================================================================

# AIF quality thresholds by criteria level
AIF_STRICT_THRESHOLDS: dict[str, float] = {
    "washout_min": 0.20,
    "fwhm_min": 5.0,
    "fwhm_max": 15.0,
    "cnr_min": 4.0,
    "baseline_cv_max": 0.10,
    "peak_time_min": 3.0,  # frames after baseline
    "peak_time_max": 20.0,  # frames after baseline
}

AIF_RELAXED_THRESHOLDS: dict[str, float] = {
    "washout_min": 0.10,
    "fwhm_min": 3.0,
    "fwhm_max": 20.0,
    "cnr_min": 2.0,
    "baseline_cv_max": 0.20,
    "peak_time_min": 2.0,
    "peak_time_max": 25.0,
}


def get_aif_thresholds(
    criteria: Literal["strict", "relaxed"] = "strict",
) -> dict[str, float]:
    """Get AIF quality thresholds for specified criteria level.

    Parameters
    ----------
    criteria : Literal["strict", "relaxed"]
        Criteria level: 'strict' for high-quality AIF detection,
        'relaxed' for lower-quality but more robust detection.

    Returns
    -------
    dict[str, float]
        Dictionary of threshold values.
    """
    if criteria == "strict":
        return AIF_STRICT_THRESHOLDS.copy()
    return AIF_RELAXED_THRESHOLDS.copy()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AIFCandidateMetrics:
    """Metrics for an AIF candidate voxel.

    Attributes
    ----------
    cnr : float
        Contrast-to-noise ratio (higher is better).
    washout : float
        Washout fraction (0-1, closer to 0.5 ideal).
    fwhm : float
        Full width at half maximum in frames.
    baseline_cv : float
        Coefficient of variation of baseline (lower is better).
    peak_time : int
        Time to peak in frames from start.
    peak_value : float
        Peak concentration value.
    """

    cnr: float
    washout: float
    fwhm: float
    baseline_cv: float
    peak_time: int
    peak_value: float


@dataclass
class AIFCandidate:
    """An AIF candidate voxel with location, score, and metrics.

    Attributes
    ----------
    coordinates : Tuple[int, int, int]
        (x, y, z) voxel coordinates.
    score : float
        Composite quality score (0-1).
    metrics : AIFCandidateMetrics
        Individual quality metrics.
    curve : npt.NDArray[Any]
        The concentration time course for this voxel.
    """

    coordinates: Tuple[int, int, int]
    score: float
    metrics: AIFCandidateMetrics
    curve: npt.NDArray[Any]


@dataclass
class AIFSearchResult:
    """Result of atlas-based AIF search.

    Attributes
    ----------
    candidates : Sequence[AIFCandidate]
        List of candidate voxels sorted by score (descending).
    aif_curve : npt.NDArray[Any]
        Final AIF curve (average of top candidates).
    aif_mask : npt.NDArray[Any]
        Binary mask indicating AIF voxel locations.
    detection_level : str
        Detection method used: 'atlas_strict', 'atlas_relaxed',
        'whole_brain_strict', 'whole_brain_relaxed', or 'synthetic'.
    quality_score : float
        Overall quality score (0-1).
    n_candidates_evaluated : int
        Total number of voxels evaluated.
    n_candidates_passed : int
        Number of voxels passing threshold criteria.
    qc_metrics : dict[str, Any]
        Aggregate QC metrics from candidates.
    """

    candidates: Sequence[AIFCandidate]
    aif_curve: npt.NDArray[Any]
    aif_mask: npt.NDArray[Any]
    detection_level: str
    quality_score: float
    n_candidates_evaluated: int
    n_candidates_passed: int
    qc_metrics: dict[str, Any]


@dataclass
class AIFQualityResult:
    """Result of AIF quality assessment.

    Attributes
    ----------
    status : str
        Overall QC status: "PASS", "WARN", or "POOR".
    quality_score : float
        Overall quality score (0-1).
    washout_fraction : float
        Signal recovery after bolus passage (0-1).
    fwhm_frames : int
        Full width at half maximum of the bolus in frames.
    baseline_cv : float
        Coefficient of variation of baseline (lower is better).
    cnr : float
        Contrast-to-noise ratio (higher is better).
    peak_value : float
        Peak concentration value.
    peak_time : int
        Time to peak in frames from start.
    meets_strict : bool
        Whether the AIF meets strict criteria.
    meets_relaxed : bool
        Whether the AIF meets relaxed criteria.
    criteria_details : dict[str, str]
        Per-metric pass/warn/fail status details.
    """

    status: str
    quality_score: float
    washout_fraction: float
    fwhm_frames: int
    baseline_cv: float
    cnr: float
    peak_value: float
    peak_time: int
    meets_strict: bool
    meets_relaxed: bool
    criteria_details: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Core Metric Functions
# =============================================================================


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


def compute_peak_time(
    aif_curve: npt.NDArray[Any],
) -> int:
    """Compute time to peak in frames.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.

    Returns
    -------
    int
        Frame index of peak concentration.
    """
    return int(np.argmax(aif_curve))


# =============================================================================
# AIF Quality Assessment
# =============================================================================


def assess_aif_quality(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
    criteria: Literal["strict", "relaxed"] = "strict",
) -> AIFQualityResult:
    """Assess the quality of an AIF curve and return pass/warn/fail status.

    This function evaluates an AIF curve against strict and relaxed criteria
    and provides a comprehensive quality assessment including individual
    metric evaluations.

    Quality is assessed based on four key metrics:
    - Washout fraction: Signal recovery after bolus (arterial shows >20% strict, >10% relaxed)
    - FWHM: Bolus width in frames (5-15 strict, 3-20 relaxed)
    - Baseline stability: CV of pre-bolus signal (<0.10 strict, <0.20 relaxed)
    - CNR: Contrast-to-noise ratio (>4.0 strict, >2.0 relaxed)

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).
    criteria : Literal["strict", "relaxed"]
        Primary criteria level for overall status determination.
        Both strict and relaxed thresholds are evaluated regardless.

    Returns
    -------
    AIFQualityResult
        Comprehensive quality assessment with status, score, and per-metric details.

    Notes
    -----
    Overall status is determined as follows:
    - PASS: All metrics meet the specified criteria level thresholds
    - WARN: Some metrics fail specified criteria but all meet relaxed thresholds
    - POOR: One or more metrics fail even relaxed thresholds

    Clinical significance of AIF quality:
    - PSR-like interpretation: washout >50% is ideal, <20% suggests venous contamination
    - Narrow FWHM (5-10 frames) indicates arterial input without dispersion
    - Low baseline CV (<0.1) ensures accurate bolus peak detection
    - High CNR (>4) provides reliable curve shape for deconvolution

    Examples
    --------
    >>> import numpy as np
    >>> # Create a synthetic AIF curve
    >>> t = np.arange(60)
    >>> aif = np.zeros(60)
    >>> aif[5:] = 10 * ((t[5:] - 5) ** 2) * np.exp(-(t[5:] - 5) / 3)  # gamma-variate
    >>> result = assess_aif_quality(aif, baseline_frames=5)
    >>> print(result.status)
    PASS
    """
    # Compute all metrics
    cnr = compute_aif_cnr(aif_curve, baseline_frames)
    washout = compute_aif_washout_fraction(aif_curve, baseline_frames)
    fwhm = compute_aif_fwhm_frames(aif_curve, baseline_frames)
    baseline_cv = compute_baseline_stability(aif_curve, baseline_frames)
    peak_time = compute_peak_time(aif_curve)
    peak_value = float(np.max(aif_curve))

    # Get both threshold sets for evaluation
    strict_thresholds = get_aif_thresholds("strict")
    relaxed_thresholds = get_aif_thresholds("relaxed")

    # Evaluate against strict thresholds
    strict_pass = _evaluate_against_thresholds(
        cnr, washout, fwhm, baseline_cv, peak_time, baseline_frames, strict_thresholds
    )

    # Evaluate against relaxed thresholds
    relaxed_pass = _evaluate_against_thresholds(
        cnr, washout, fwhm, baseline_cv, peak_time, baseline_frames, relaxed_thresholds
    )

    # Determine per-metric status for detailed reporting
    criteria_details = _get_criteria_details(
        cnr, washout, fwhm, baseline_cv, strict_thresholds, relaxed_thresholds
    )

    # Compute quality score (0-1) based on how well metrics compare to ideal
    quality_score = _compute_quality_score(
        cnr, washout, fwhm, baseline_cv, strict_thresholds
    )

    # Determine overall status
    if criteria == "strict":
        if strict_pass:
            status = QCStatus.PASS
        elif relaxed_pass:
            status = QCStatus.WARN
        else:
            status = QCStatus.POOR
    else:  # relaxed criteria
        if relaxed_pass:
            status = QCStatus.PASS
        else:
            status = QCStatus.POOR

    return AIFQualityResult(
        status=status,
        quality_score=quality_score,
        washout_fraction=washout,
        fwhm_frames=fwhm,
        baseline_cv=baseline_cv,
        cnr=cnr,
        peak_value=peak_value,
        peak_time=peak_time,
        meets_strict=strict_pass,
        meets_relaxed=relaxed_pass,
        criteria_details=criteria_details,
    )


def _evaluate_against_thresholds(
    cnr: float,
    washout: float,
    fwhm: int,
    baseline_cv: float,
    peak_time: int,
    baseline_frames: int,
    thresholds: dict[str, float],
) -> bool:
    """Check if all metrics meet the specified thresholds.

    Parameters
    ----------
    cnr : float
        Contrast-to-noise ratio.
    washout : float
        Washout fraction (0-1).
    fwhm : int
        Full width at half maximum in frames.
    baseline_cv : float
        Coefficient of variation of baseline.
    peak_time : int
        Time to peak in frames.
    baseline_frames : int
        Number of baseline frames.
    thresholds : dict[str, float]
        Threshold values from get_aif_thresholds().

    Returns
    -------
    bool
        True if all metrics meet thresholds, False otherwise.
    """
    # Check CNR
    if cnr < thresholds["cnr_min"]:
        return False

    # Check washout
    if washout < thresholds["washout_min"]:
        return False

    # Check FWHM bounds
    if fwhm < thresholds["fwhm_min"] or fwhm > thresholds["fwhm_max"]:
        return False

    # Check baseline stability
    if baseline_cv > thresholds["baseline_cv_max"]:
        return False

    # Check peak time is reasonable
    peak_from_baseline = peak_time - baseline_frames
    if peak_from_baseline < thresholds["peak_time_min"]:
        return False
    if peak_from_baseline > thresholds["peak_time_max"]:
        return False

    return True


def _get_criteria_details(
    cnr: float,
    washout: float,
    fwhm: int,
    baseline_cv: float,
    strict_thresholds: dict[str, float],
    relaxed_thresholds: dict[str, float],
) -> dict[str, str]:
    """Get per-metric pass/warn/fail status details.

    Parameters
    ----------
    cnr : float
        Contrast-to-noise ratio.
    washout : float
        Washout fraction (0-1).
    fwhm : int
        Full width at half maximum in frames.
    baseline_cv : float
        Coefficient of variation of baseline.
    strict_thresholds : dict[str, float]
        Strict threshold values.
    relaxed_thresholds : dict[str, float]
        Relaxed threshold values.

    Returns
    -------
    dict[str, str]
        Dictionary mapping metric name to status string.
    """
    details: dict[str, str] = {}

    # CNR evaluation (higher is better)
    if cnr >= strict_thresholds["cnr_min"]:
        details["cnr"] = QCStatus.PASS
    elif cnr >= relaxed_thresholds["cnr_min"]:
        details["cnr"] = QCStatus.WARN
    else:
        details["cnr"] = QCStatus.POOR

    # Washout evaluation (higher is better, up to ~0.5)
    if washout >= strict_thresholds["washout_min"]:
        details["washout"] = QCStatus.PASS
    elif washout >= relaxed_thresholds["washout_min"]:
        details["washout"] = QCStatus.WARN
    else:
        details["washout"] = QCStatus.POOR

    # FWHM evaluation (within range is better)
    fwhm_float = float(fwhm)
    if (
        strict_thresholds["fwhm_min"] <= fwhm_float <= strict_thresholds["fwhm_max"]
    ):
        details["fwhm"] = QCStatus.PASS
    elif (
        relaxed_thresholds["fwhm_min"] <= fwhm_float <= relaxed_thresholds["fwhm_max"]
    ):
        details["fwhm"] = QCStatus.WARN
    else:
        details["fwhm"] = QCStatus.POOR

    # Baseline CV evaluation (lower is better)
    if baseline_cv <= strict_thresholds["baseline_cv_max"]:
        details["baseline_cv"] = QCStatus.PASS
    elif baseline_cv <= relaxed_thresholds["baseline_cv_max"]:
        details["baseline_cv"] = QCStatus.WARN
    else:
        details["baseline_cv"] = QCStatus.POOR

    return details


def _compute_quality_score(
    cnr: float,
    washout: float,
    fwhm: int,
    baseline_cv: float,
    thresholds: dict[str, float],
) -> float:
    """Compute overall quality score (0-1) based on metrics.

    The score is a weighted combination of individual metric scores:
    - CNR: 40% weight (most important for deconvolution reliability)
    - Washout: 30% weight (indicates arterial vs venous)
    - FWHM: 20% weight (indicates dispersion)
    - Baseline CV: 10% weight (noise assessment)

    Parameters
    ----------
    cnr : float
        Contrast-to-noise ratio.
    washout : float
        Washout fraction (0-1).
    fwhm : int
        Full width at half maximum in frames.
    baseline_cv : float
        Coefficient of variation of baseline.
    thresholds : dict[str, float]
        Threshold values for normalization.

    Returns
    -------
    float
        Quality score between 0 and 1.
    """
    score = 0.0

    # CNR contribution (0-0.4): normalized to expected range
    # Typical good CNR is 4-20, exceptional is >20
    cnr_normalized = min(1.0, max(0.0, (cnr - thresholds["cnr_min"]) / 16.0))
    score += 0.4 * cnr_normalized

    # Washout contribution (0-0.3): closer to 0.4-0.6 is ideal for arterial
    # Peak score at 0.5, decays towards 0 and 1
    ideal_washout = 0.5
    washout_deviation = abs(washout - ideal_washout)
    washout_normalized = max(0.0, 1.0 - washout_deviation / 0.5)
    score += 0.3 * washout_normalized

    # FWHM contribution (0-0.2): closer to 8-10 frames is ideal
    ideal_fwhm = 9.0
    fwhm_range = thresholds["fwhm_max"] - thresholds["fwhm_min"]
    fwhm_deviation = abs(float(fwhm) - ideal_fwhm)
    fwhm_normalized = max(0.0, 1.0 - fwhm_deviation / (fwhm_range / 2.0))
    score += 0.2 * fwhm_normalized

    # Baseline stability contribution (0-0.1): lower CV is better
    baseline_normalized = max(0.0, 1.0 - baseline_cv / thresholds["baseline_cv_max"])
    score += 0.1 * baseline_normalized

    return min(1.0, score)


# =============================================================================
# Candidate Scoring
# =============================================================================


def compute_candidate_metrics(
    curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> AIFCandidateMetrics:
    """Compute all quality metrics for a candidate AIF curve.

    Parameters
    ----------
    curve : np.ndarray
        1D concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames.

    Returns
    -------
    AIFCandidateMetrics
        Computed metrics for the candidate.
    """
    cnr = compute_aif_cnr(curve, baseline_frames)
    washout = compute_aif_washout_fraction(curve, baseline_frames)
    fwhm = float(compute_aif_fwhm_frames(curve, baseline_frames))
    baseline_cv = compute_baseline_stability(curve, baseline_frames)
    peak_time = compute_peak_time(curve)
    peak_value = float(np.max(curve))

    return AIFCandidateMetrics(
        cnr=cnr,
        washout=washout,
        fwhm=fwhm,
        baseline_cv=baseline_cv,
        peak_time=peak_time,
        peak_value=peak_value,
    )


def score_aif_candidate(
    metrics: AIFCandidateMetrics,
    thresholds: dict[str, float],
    baseline_frames: int = 5,
) -> float:
    """Score an AIF candidate based on arterial characteristics.

    Returns 0 if any threshold is not met, otherwise returns a composite
    score (0-1) based on how well the curve matches ideal arterial
    characteristics.

    Parameters
    ----------
    metrics : AIFCandidateMetrics
        Computed metrics for the candidate.
    thresholds : dict[str, float]
        Quality thresholds (from get_aif_thresholds).
    baseline_frames : int
        Number of baseline frames (for peak time validation).

    Returns
    -------
    float
        Composite quality score (0-1), or 0 if thresholds not met.
    """
    # Check hard thresholds - return 0 if any fail
    if metrics.cnr < thresholds["cnr_min"]:
        return 0.0
    if metrics.washout < thresholds["washout_min"]:
        return 0.0
    if metrics.fwhm < thresholds["fwhm_min"] or metrics.fwhm > thresholds["fwhm_max"]:
        return 0.0
    if metrics.baseline_cv > thresholds["baseline_cv_max"]:
        return 0.0

    # Check peak time is reasonable
    peak_from_baseline = metrics.peak_time - baseline_frames
    if peak_from_baseline < thresholds["peak_time_min"]:
        return 0.0
    if peak_from_baseline > thresholds["peak_time_max"]:
        return 0.0

    # Compute composite score (all thresholds met)
    # Weight factors: CNR most important, then washout, then FWHM, then baseline
    score = 0.0

    # CNR contribution (0-0.4): normalized to expected range
    # Typical good CNR is 4-20, exceptional is >20
    cnr_score = min(1.0, (metrics.cnr - thresholds["cnr_min"]) / 16.0)
    score += 0.4 * cnr_score

    # Washout contribution (0-0.3): closer to 0.4-0.6 is ideal for arterial
    ideal_washout = 0.5
    washout_score = 1.0 - abs(metrics.washout - ideal_washout) / 0.5
    score += 0.3 * max(0.0, washout_score)

    # FWHM contribution (0-0.2): closer to 8-10 frames is ideal
    ideal_fwhm = 9.0
    fwhm_range = thresholds["fwhm_max"] - thresholds["fwhm_min"]
    fwhm_score = 1.0 - abs(metrics.fwhm - ideal_fwhm) / (fwhm_range / 2.0)
    score += 0.2 * max(0.0, fwhm_score)

    # Baseline stability contribution (0-0.1): lower CV is better
    baseline_score = 1.0 - (metrics.baseline_cv / thresholds["baseline_cv_max"])
    score += 0.1 * max(0.0, baseline_score)

    return min(1.0, score)


# =============================================================================
# Distance and Exclusion Utilities
# =============================================================================


def compute_distance_map(
    mask: npt.NDArray[Any],
    voxel_sizes: Tuple[float, float, float],
) -> npt.NDArray[Any]:
    """Compute distance transform from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (nonzero indicates region of interest).
    voxel_sizes : Tuple[float, float, float]
        Voxel dimensions in mm (x, y, z).

    Returns
    -------
    np.ndarray
        3D array of Euclidean distances (mm) from nearest mask voxel.
    """
    from scipy import ndimage

    # Invert mask for distance transform (distance FROM mask)
    inverted = (mask <= 0).astype(np.uint8)

    # Compute distance with voxel spacing
    distance = ndimage.distance_transform_edt(inverted, sampling=voxel_sizes)

    return distance  # type: ignore[no-any-return]


def exclude_near_lesion(
    search_mask: npt.NDArray[Any],
    lesion_mask: npt.NDArray[Any],
    voxel_sizes: Tuple[float, float, float],
    min_distance_mm: float = LESION_EXCLUSION_DISTANCE_MM,
) -> npt.NDArray[Any]:
    """Exclude voxels within specified distance of lesion.

    Parameters
    ----------
    search_mask : np.ndarray
        3D binary mask of candidate search region.
    lesion_mask : np.ndarray
        3D binary mask of lesion(s) to avoid.
    voxel_sizes : Tuple[float, float, float]
        Voxel dimensions in mm.
    min_distance_mm : float
        Minimum distance (mm) from lesion for valid candidates.

    Returns
    -------
    np.ndarray
        Updated search mask with near-lesion voxels excluded.
    """
    # Compute distance from lesion
    distance = compute_distance_map(lesion_mask, voxel_sizes)

    # Create exclusion mask
    far_from_lesion = distance >= min_distance_mm

    # Combine with search mask
    return (search_mask > 0) & far_from_lesion


def get_contralateral_mask(
    brain_mask: npt.NDArray[Any],
    lesion_mask: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Get mask for contralateral hemisphere relative to lesion.

    Assumes axial orientation with left-right along first axis.

    Parameters
    ----------
    brain_mask : np.ndarray
        3D brain mask.
    lesion_mask : np.ndarray
        3D lesion mask.

    Returns
    -------
    np.ndarray
        3D mask for contralateral hemisphere within brain.
    """
    # Find lesion center of mass
    lesion_coords = np.where(lesion_mask > 0)
    if len(lesion_coords[0]) == 0:
        return brain_mask.copy()  # type: ignore[no-any-return]

    lesion_center_x = np.mean(lesion_coords[0])
    brain_center_x = brain_mask.shape[0] / 2.0

    # Determine which side lesion is on
    lesion_on_left = lesion_center_x < brain_center_x

    # Create contralateral mask
    contralateral = np.zeros_like(brain_mask, dtype=np.uint8)
    midline = int(brain_mask.shape[0] / 2)

    if lesion_on_left:
        # Lesion on left, use right hemisphere
        contralateral[midline:, :, :] = 1
    else:
        # Lesion on right, use left hemisphere
        contralateral[:midline, :, :] = 1

    return (brain_mask > 0) & (contralateral > 0)


# =============================================================================
# Atlas-Based AIF Search
# =============================================================================


def atlas_based_aif_search(
    concentration_data: npt.NDArray[Any],
    brain_mask: npt.NDArray[Any],
    atlas_mask: Optional[npt.NDArray[Any]] = None,
    lesion_mask: Optional[npt.NDArray[Any]] = None,
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    criteria: Literal["strict", "relaxed"] = "strict",
    n_candidates: int = 10,
    baseline_frames: int = 5,
    lesion_distance_mm: float = LESION_EXCLUSION_DISTANCE_MM,
) -> AIFSearchResult:
    """Perform atlas-constrained AIF search.

    This function implements multi-level AIF detection:
    1. Atlas-based with strict criteria
    2. Atlas-based with relaxed criteria
    3. Whole-brain with strict criteria
    4. Whole-brain with relaxed criteria

    Parameters
    ----------
    concentration_data : np.ndarray
        4D concentration time courses (x, y, z, time).
    brain_mask : np.ndarray
        3D brain mask.
    atlas_mask : np.ndarray, optional
        3D arterial atlas mask in DSC space. If None, uses whole brain.
    lesion_mask : np.ndarray, optional
        3D lesion mask for exclusion.
    voxel_sizes : Tuple[float, float, float]
        Voxel dimensions in mm.
    criteria : Literal["strict", "relaxed"]
        Initial criteria level for thresholds.
    n_candidates : int
        Number of top candidate voxels to return.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    lesion_distance_mm : float
        Minimum distance (mm) from lesion for candidates.

    Returns
    -------
    AIFSearchResult
        Complete search result with candidates, AIF curve, and QC metrics.
    """
    # Build base search mask
    search_mask = brain_mask.copy().astype(np.uint8)

    # Apply lesion exclusion
    if lesion_mask is not None and np.any(lesion_mask > 0):
        search_mask = exclude_near_lesion(
            search_mask, lesion_mask, voxel_sizes, lesion_distance_mm
        ).astype(np.uint8)

        # Note: contralateral preference is now handled via soft penalty
        # in the main AIF detection interface (_filter_candidates_by_hemisphere)
        # rather than hard-filtering the search mask here.

    # Define search levels
    # Each level is (name, mask, criteria_level)
    search_levels: list[
        Tuple[str, npt.NDArray[Any], Literal["strict", "relaxed"]]
    ] = []

    if atlas_mask is not None:
        # Level 1: Atlas with strict
        search_levels.append(
            ("atlas_strict", search_mask * (atlas_mask > 0).astype(np.uint8), "strict")
        )
        # Level 2: Atlas with relaxed
        search_levels.append(
            ("atlas_relaxed", search_mask * (atlas_mask > 0).astype(np.uint8), "relaxed")
        )

    # Level 3: Whole brain with user criteria
    search_levels.append((f"whole_brain_{criteria}", search_mask, criteria))

    # Level 4: Whole brain with relaxed (if not already relaxed)
    if criteria == "strict":
        search_levels.append(("whole_brain_relaxed", search_mask, "relaxed"))

    # Try each level until we find candidates
    all_candidates: list[AIFCandidate] = []
    detection_level = "none"
    n_evaluated = 0

    for level_name, level_mask, level_criteria in search_levels:
        if np.sum(level_mask) == 0:
            continue

        thresholds = get_aif_thresholds(level_criteria)
        level_candidates, level_evaluated = _search_candidates_in_mask(
            concentration_data,
            level_mask,
            thresholds,
            baseline_frames,
            n_candidates * 2,  # Get extra candidates for ranking
        )
        n_evaluated += level_evaluated

        if level_candidates:
            all_candidates = level_candidates
            detection_level = level_name
            break

    # Build result
    if not all_candidates:
        # No candidates found - return empty result
        empty_curve = np.zeros(concentration_data.shape[3], dtype=np.float64)
        empty_mask = np.zeros(concentration_data.shape[:3], dtype=np.uint8)
        return AIFSearchResult(
            candidates=[],
            aif_curve=empty_curve,
            aif_mask=empty_mask,
            detection_level="none",
            quality_score=0.0,
            n_candidates_evaluated=n_evaluated,
            n_candidates_passed=0,
            qc_metrics={
                "cnr_mean": 0.0,
                "washout_mean": 0.0,
                "fwhm_mean": 0.0,
                "baseline_cv_mean": 0.0,
            },
        )

    # Sort by score and take top N
    all_candidates.sort(key=lambda c: c.score, reverse=True)
    top_candidates = all_candidates[:n_candidates]

    # Build AIF mask
    aif_mask = np.zeros(concentration_data.shape[:3], dtype=np.uint8)
    for cand in top_candidates:
        x, y, z = cand.coordinates
        aif_mask[x, y, z] = 1

    # Average curves for final AIF
    curves = [c.curve for c in top_candidates]
    aif_curve = np.mean(np.stack(curves, axis=0), axis=0)

    # Aggregate QC metrics
    quality_score = float(np.mean([c.score for c in top_candidates]))
    qc_metrics = {
        "cnr_mean": float(np.mean([c.metrics.cnr for c in top_candidates])),
        "cnr_std": float(np.std([c.metrics.cnr for c in top_candidates])),
        "washout_mean": float(np.mean([c.metrics.washout for c in top_candidates])),
        "fwhm_mean": float(np.mean([c.metrics.fwhm for c in top_candidates])),
        "baseline_cv_mean": float(
            np.mean([c.metrics.baseline_cv for c in top_candidates])
        ),
        "peak_value_mean": float(
            np.mean([c.metrics.peak_value for c in top_candidates])
        ),
        "n_aif_voxels_selected": len(top_candidates),
    }

    return AIFSearchResult(
        candidates=top_candidates,
        aif_curve=aif_curve,
        aif_mask=aif_mask,
        detection_level=detection_level,
        quality_score=quality_score,
        n_candidates_evaluated=n_evaluated,
        n_candidates_passed=len(all_candidates),
        qc_metrics=qc_metrics,
    )


def _search_candidates_in_mask(
    concentration_data: npt.NDArray[Any],
    mask: npt.NDArray[Any],
    thresholds: dict[str, float],
    baseline_frames: int,
    max_candidates: int,
) -> Tuple[list[AIFCandidate], int]:
    """Search for AIF candidates within a mask.

    Parameters
    ----------
    concentration_data : np.ndarray
        4D concentration data.
    mask : np.ndarray
        3D search mask.
    thresholds : dict[str, float]
        Quality thresholds.
    baseline_frames : int
        Number of baseline frames.
    max_candidates : int
        Maximum candidates to return.

    Returns
    -------
    Tuple[list[AIFCandidate], int]
        List of passing candidates and total voxels evaluated.
    """
    candidates: list[AIFCandidate] = []

    # Get mask coordinates
    mask_coords = np.where(mask > 0)
    n_voxels = len(mask_coords[0])

    for i in range(n_voxels):
        x, y, z = (
            int(mask_coords[0][i]),
            int(mask_coords[1][i]),
            int(mask_coords[2][i]),
        )
        curve = concentration_data[x, y, z, :].astype(np.float64)

        # Skip empty curves
        if np.max(np.abs(curve)) < EPSILON_CONCENTRATION:
            continue

        # Compute metrics
        metrics = compute_candidate_metrics(curve, baseline_frames)

        # Score candidate
        score = score_aif_candidate(metrics, thresholds, baseline_frames)

        if score > 0:
            candidates.append(
                AIFCandidate(
                    coordinates=(x, y, z),
                    score=score,
                    metrics=metrics,
                    curve=curve,
                )
            )

    # Sort by score and limit
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates], n_voxels


# =============================================================================
# File-Based Wrapper
# =============================================================================


def atlas_based_aif_search_from_files(
    concentration_file: Union[str, Path],
    brain_mask_file: Union[str, Path],
    atlas_mask_file: Optional[Union[str, Path]] = None,
    lesion_mask_file: Optional[Union[str, Path]] = None,
    criteria: Literal["strict", "relaxed"] = "strict",
    n_candidates: int = 10,
    baseline_frames: int = 5,
    lesion_distance_mm: float = LESION_EXCLUSION_DISTANCE_MM,
) -> AIFSearchResult:
    """Perform atlas-based AIF search from NIfTI files.

    Convenience wrapper that loads NIfTI files and calls atlas_based_aif_search.

    Parameters
    ----------
    concentration_file : str or Path
        Path to 4D concentration NIfTI file.
    brain_mask_file : str or Path
        Path to 3D brain mask NIfTI file.
    atlas_mask_file : str or Path, optional
        Path to 3D arterial atlas mask NIfTI file.
    lesion_mask_file : str or Path, optional
        Path to 3D lesion mask NIfTI file.
    criteria : Literal["strict", "relaxed"]
        Quality criteria level.
    n_candidates : int
        Number of top candidates to return.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    lesion_distance_mm : float
        Minimum distance (mm) from lesion.

    Returns
    -------
    AIFSearchResult
        Complete search result.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    # Load concentration data
    conc_img = nib.load(concentration_file)
    concentration_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]

    # Get voxel sizes
    header = conc_img.header  # type: ignore[attr-defined]
    zooms = header.get_zooms()[:3]  # type: ignore[attr-defined]
    voxel_sizes: Tuple[float, float, float] = (
        float(zooms[0]),
        float(zooms[1]),
        float(zooms[2]),
    )

    # Load brain mask
    brain_img = nib.load(brain_mask_file)
    brain_mask: npt.NDArray[Any] = brain_img.get_fdata()  # type: ignore[attr-defined]

    # Load optional atlas mask
    atlas_mask: Optional[npt.NDArray[Any]] = None
    if atlas_mask_file is not None:
        atlas_img = nib.load(atlas_mask_file)
        atlas_mask = atlas_img.get_fdata()  # type: ignore[attr-defined]

    # Load optional lesion mask
    lesion_mask: Optional[npt.NDArray[Any]] = None
    if lesion_mask_file is not None:
        lesion_img = nib.load(lesion_mask_file)
        lesion_mask = lesion_img.get_fdata()  # type: ignore[attr-defined]

    return atlas_based_aif_search(
        concentration_data=concentration_data,
        brain_mask=brain_mask,
        atlas_mask=atlas_mask,
        lesion_mask=lesion_mask,
        voxel_sizes=voxel_sizes,
        criteria=criteria,
        n_candidates=n_candidates,
        baseline_frames=baseline_frames,
        lesion_distance_mm=lesion_distance_mm,
    )


# =============================================================================
# Gamma-Variate AIF Fitting
# =============================================================================


@dataclass
class GammaVariateFitResult:
    """Result of gamma-variate fitting to an AIF curve.

    The gamma-variate model is:
        C(t) = A × (t - t0)^α × exp(-(t - t0) / β)  for t > t0
        C(t) = 0                                     for t <= t0

    Where:
        A: Amplitude scaling factor
        t0: Bolus arrival time
        α (alpha): Shape parameter controlling the rise steepness
        β (beta): Scale parameter controlling the decay rate

    Derived parameters:
        Time to peak (TTP) = t0 + α × β
        AUC = A × β^(α+1) × Γ(α+1)  (total dose)

    Attributes
    ----------
    fitted_curve : npt.NDArray[Any]
        The fitted gamma-variate curve (same length as input).
    amplitude : float
        Amplitude parameter A.
    t0 : float
        Bolus arrival time in frames.
    alpha : float
        Shape parameter α (controls rise steepness).
    beta : float
        Scale parameter β (controls decay rate, in frames).
    auc_original : float
        First-pass-window AUC of the input curve when ``remove_recirculation=True``;
        full-curve AUC otherwise. Window matches ``auc_fitted`` so ``auc_ratio``
        is apples-to-apples.
    auc_fitted : float
        Area under curve of fitted result, integrated over the same window
        as ``auc_original``. Note: when ``preserve_auc=True`` (production
        default), this is post-scaling and equals ``auc_original`` by
        construction. ``auc_ratio`` is computed BEFORE scaling, so it
        retains shape-fit-quality semantics.
    auc_ratio : float
        Ratio of fitted/original AUC computed BEFORE preservation scaling
        — measures gamma-fit shape mismatch against the empirical
        first-pass area independent of dose-preservation. Should be ~1.0
        for a good fit. Reported in QC for provenance; not a hard gate
        (see ``should_apply_gamma_fit``).
    residual_rms : float
        RMS residual between original and fitted.
    r_squared : float
        Coefficient of determination (R²).
    fit_converged : bool
        Whether the optimization converged successfully.
    recirculation_removed : bool
        Whether recirculation was removed from the fit.
    first_pass_n : int
        Number of samples in the first-pass window
        ``[baseline_frames : peak_idx + max(10, 2*FWHM)]`` when
        ``remove_recirculation=True``; equal to ``n_frames - baseline_frames``
        otherwise. Gamma fitting is ill-posed below ~8 samples (Welker AJNR
        2015, Calamante 2013).
    """

    fitted_curve: npt.NDArray[Any]
    amplitude: float
    t0: float
    alpha: float
    beta: float
    auc_original: float
    auc_fitted: float
    auc_ratio: float
    residual_rms: float
    r_squared: float
    fit_converged: bool
    recirculation_removed: bool = True
    first_pass_n: int = 0


def _gamma_variate_model(
    t: npt.NDArray[Any],
    amplitude: float,
    t0: float,
    alpha: float,
    beta: float,
) -> npt.NDArray[Any]:
    """Compute gamma-variate function.

    C(t) = A × (t - t0)^α × exp(-(t - t0) / β)  for t > t0
    C(t) = 0                                     for t <= t0

    Parameters
    ----------
    t : np.ndarray
        Time points (frame indices).
    amplitude : float
        Amplitude scaling factor A.
    t0 : float
        Bolus arrival time.
    alpha : float
        Shape parameter (>0).
    beta : float
        Scale parameter (>0).

    Returns
    -------
    np.ndarray
        Gamma-variate function values.
    """
    result = np.zeros_like(t, dtype=np.float64)
    mask = t > t0

    if np.any(mask):
        dt = t[mask] - t0
        # Avoid numerical issues with very small alpha
        alpha_safe = max(alpha, 0.01)
        beta_safe = max(beta, 0.01)

        # Compute gamma-variate: A × (t-t0)^α × exp(-(t-t0)/β)
        result[mask] = amplitude * np.power(dt, alpha_safe) * np.exp(-dt / beta_safe)

    return result


def _gamma_variate_residual(
    params: Tuple[float, float, float, float],
    t: npt.NDArray[Any],
    y: npt.NDArray[Any],
    weights: Optional[npt.NDArray[Any]] = None,
) -> npt.NDArray[Any]:
    """Compute weighted residuals for gamma-variate fitting.

    Parameters
    ----------
    params : Tuple[float, float, float, float]
        (amplitude, t0, alpha, beta)
    t : np.ndarray
        Time points.
    y : np.ndarray
        Observed values.
    weights : np.ndarray, optional
        Fitting weights.

    Returns
    -------
    np.ndarray
        Weighted residuals.
    """
    amplitude, t0, alpha, beta = params
    y_pred = _gamma_variate_model(t, amplitude, t0, alpha, beta)
    residuals = y - y_pred

    if weights is not None:
        residuals = residuals * weights

    return residuals  # type: ignore[no-any-return]


def _estimate_initial_params(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> Tuple[float, float, float, float]:
    """Estimate initial parameters for gamma-variate fitting.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames.

    Returns
    -------
    Tuple[float, float, float, float]
        Initial estimates for (amplitude, t0, alpha, beta).
    """
    n_frames = len(aif_curve)
    t = np.arange(n_frames, dtype=np.float64)

    # Find peak
    baseline = np.mean(aif_curve[:baseline_frames])
    curve_shifted = aif_curve - baseline
    peak_idx = int(np.argmax(curve_shifted))
    peak_value = float(np.max(curve_shifted))

    if peak_value < EPSILON_CONCENTRATION:
        # Flat curve, return defaults
        return (1.0, float(baseline_frames), 2.0, 3.0)

    # Estimate t0: first frame where signal exceeds 10% of peak
    threshold = baseline + 0.1 * peak_value
    rising_indices = np.where(aif_curve > threshold)[0]
    if len(rising_indices) > 0:
        t0_estimate = float(rising_indices[0]) - 0.5
    else:
        t0_estimate = float(baseline_frames)

    # Ensure t0 is not negative
    t0_estimate = max(0.0, t0_estimate)

    # Time to peak from t0
    ttp_from_t0 = max(1.0, peak_idx - t0_estimate)

    # For gamma-variate: TTP = alpha × beta
    # Estimate alpha from rise steepness (typical range 1-5)
    # Steeper rise → smaller alpha
    # Estimate based on FWHM
    fwhm = compute_aif_fwhm_frames(aif_curve, baseline_frames)
    if fwhm > 0:
        # Approximate relationship: FWHM ≈ 2.35 × beta for Gaussian-like shapes
        # For gamma-variate, FWHM/TTP gives info about alpha
        fwhm_ttp_ratio = fwhm / ttp_from_t0 if ttp_from_t0 > 0 else 2.0
        # Lower ratio → sharper peak → lower alpha
        alpha_estimate = max(0.5, min(5.0, 2.0 * fwhm_ttp_ratio))
    else:
        alpha_estimate = 2.0

    # Compute beta from TTP and alpha: TTP = alpha × beta → beta = TTP/alpha
    beta_estimate = ttp_from_t0 / alpha_estimate if alpha_estimate > 0 else 3.0
    beta_estimate = max(0.5, beta_estimate)

    # Estimate amplitude: at peak, A × (TTP/beta)^alpha × exp(-alpha)
    # Solving for A
    peak_term = np.power(ttp_from_t0, alpha_estimate) * np.exp(-alpha_estimate)
    if peak_term > EPSILON_CONCENTRATION:
        amplitude_estimate = peak_value / peak_term
    else:
        amplitude_estimate = peak_value

    return (amplitude_estimate, t0_estimate, alpha_estimate, beta_estimate)


def _compute_auc(
    curve: npt.NDArray[Any],
    tr: float = 1.0,
) -> float:
    """Compute area under curve using trapezoidal integration.

    Parameters
    ----------
    curve : np.ndarray
        1D time course.
    tr : float
        Time step (frame duration) for scaling.

    Returns
    -------
    float
        Area under curve.
    """
    # Use scipy's trapezoid for compatibility
    from scipy import integrate

    auc = float(integrate.trapezoid(curve)) * tr
    return auc


def _find_recirculation_start(
    aif_curve: npt.NDArray[Any],
    fitted_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> int:
    """Find the frame where recirculation begins.

    Recirculation is detected where the measured curve starts to
    deviate above the fitted gamma-variate (first-pass only).

    Parameters
    ----------
    aif_curve : np.ndarray
        Original AIF curve.
    fitted_curve : np.ndarray
        Fitted gamma-variate curve.
    baseline_frames : int
        Number of baseline frames.

    Returns
    -------
    int
        Frame index where recirculation starts, or -1 if not detected.
    """
    n_frames = len(aif_curve)

    # Find peak of fitted curve
    peak_idx = int(np.argmax(fitted_curve))

    # Look for recirculation after the washout phase begins
    # Check for sustained deviation above fitted curve
    min_frames_after_peak = 5  # Wait for initial washout
    start_search = min(peak_idx + min_frames_after_peak, n_frames - 5)

    if start_search >= n_frames - 2:
        return -1

    # Compute residual
    residual = aif_curve - fitted_curve

    # Find where residual becomes consistently positive
    # (measured > fitted, indicating recirculation contribution)
    window = 3  # Require sustained deviation
    for i in range(start_search, n_frames - window):
        # Check if next few frames are all above fitted
        if np.all(residual[i : i + window] > 0):
            return i

    return -1


def fit_gamma_variate(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
    remove_recirculation: bool = True,
    preserve_auc: bool = True,
    tr: float = 1.0,
) -> GammaVariateFitResult:
    """Fit a gamma-variate model to an AIF curve.

    This function fits the model:
        C(t) = A × (t - t0)^α × exp(-(t - t0) / β)

    The gamma-variate model represents the first-pass bolus only,
    allowing removal of recirculation effects for preload correction.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames for baseline subtraction.
    remove_recirculation : bool
        If True, fit only the first-pass portion and remove recirculation.
        If False, fit the entire curve.
    preserve_auc : bool
        If True, scale the fitted curve to match the first-pass AUC (when
        remove_recirculation=True) or the full AUC (when False).
    tr : float
        Repetition time in seconds (for AUC calculation).

    Returns
    -------
    GammaVariateFitResult
        Fitted parameters and curve with quality metrics.

    Notes
    -----
    The gamma-variate model is widely used for AIF characterization because:
    - It naturally models the first-pass bolus shape
    - It has analytical AUC: AUC = A × β^(α+1) × Γ(α+1)
    - Recirculation appears as positive residual after washout

    For preload correction:
    - Fit gamma-variate to measured AIF
    - The fitted curve represents first-pass only
    - Residual (measured - fitted) contains recirculation
    - Using fitted AIF for deconvolution removes recirculation effects

    Clinical interpretation:
    - Alpha (α): Rise steepness; typical 1.5-3.5
    - Beta (β): Decay rate; typical 1.5-4.0 frames
    - TTP = α × β: Time to peak from arrival
    - MTT_AIF = (α + 1) × β: Mean transit time of bolus

    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic AIF with recirculation
    >>> t = np.arange(60)
    >>> aif = np.zeros(60)
    >>> # First pass: gamma-variate
    >>> first_pass = 10 * np.power(np.maximum(0, t - 5), 2.0) * np.exp(-(t - 5) / 3)
    >>> # Recirculation: delayed, smaller peak
    >>> recirc = 3 * np.power(np.maximum(0, t - 20), 1.5) * np.exp(-(t - 20) / 5)
    >>> aif = first_pass + recirc
    >>> result = fit_gamma_variate(aif, baseline_frames=5, remove_recirculation=True)
    >>> print(f"Fit R²: {result.r_squared:.3f}, AUC ratio: {result.auc_ratio:.3f}")
    Fit R²: 0.8..., AUC ratio: 0.9...
    """
    n_frames = len(aif_curve)
    t = np.arange(n_frames, dtype=np.float64)

    # Subtract baseline
    baseline = float(np.mean(aif_curve[:baseline_frames]))
    curve_shifted = aif_curve - baseline

    # Get initial parameter estimates
    p0 = _estimate_initial_params(aif_curve, baseline_frames)

    # Define bounds for parameters
    # (amplitude, t0, alpha, beta)
    bounds_lower = (0.0, 0.0, 0.1, 0.1)
    bounds_upper = (np.inf, float(n_frames) / 2.0, 10.0, 20.0)

    # Create fitting weights
    # Higher weight around peak, lower in baseline and tail
    peak_idx = int(np.argmax(curve_shifted))
    weights = np.ones(n_frames, dtype=np.float64)

    # Reduce weight of baseline
    weights[:baseline_frames] = 0.1

    if remove_recirculation:
        # For recirculation removal, focus fitting on first pass
        # Use data up to ~2× FWHM after peak
        fwhm = compute_aif_fwhm_frames(aif_curve, baseline_frames)
        cutoff = min(peak_idx + max(10, 2 * fwhm), n_frames)
        weights[cutoff:] = 0.01  # Minimal weight on tail/recirculation

    # Perform nonlinear least squares fit
    try:
        result = optimize.least_squares(
            _gamma_variate_residual,
            p0,
            args=(t, curve_shifted, weights),
            bounds=(bounds_lower, bounds_upper),
            method="trf",
            max_nfev=1000,
        )
        fit_converged = result.success
        amplitude, t0, alpha, beta = result.x
    except Exception:
        # Fit failed, return initial estimates
        fit_converged = False
        amplitude, t0, alpha, beta = p0

    # Generate fitted curve
    fitted_curve = _gamma_variate_model(t, amplitude, t0, alpha, beta)

    # Compute fit quality metrics on UNSCALED curve (before AUC scaling)
    # R² and residual_rms measure shape fit quality and must not be
    # corrupted by dose-preservation scaling
    if remove_recirculation:
        eval_end = min(
            peak_idx + max(10, 2 * compute_aif_fwhm_frames(aif_curve, baseline_frames)),
            n_frames,
        )
        eval_start = baseline_frames
    else:
        eval_start = baseline_frames
        eval_end = n_frames

    ss_res = float(
        np.sum((curve_shifted[eval_start:eval_end] - fitted_curve[eval_start:eval_end]) ** 2)
    )
    ss_tot = float(
        np.sum(
            (curve_shifted[eval_start:eval_end] - np.mean(curve_shifted[eval_start:eval_end])) ** 2
        )
    )

    if ss_tot > EPSILON_CONCENTRATION:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = 0.0

    # Clip R² to valid range (can be negative for poor fits)
    r_squared = max(-1.0, min(1.0, r_squared))

    # Compute RMS residual on unscaled curve
    residuals = curve_shifted - fitted_curve
    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    # Detect recirculation if requested
    recirculation_removed = False
    if remove_recirculation:
        recirc_start = _find_recirculation_start(
            curve_shifted, fitted_curve, baseline_frames
        )
        if recirc_start > 0:
            recirculation_removed = True
            # The fitted gamma-variate naturally excludes recirculation
            # since it only models first-pass bolus

    # Symmetric AUC window: when removing recirculation, integrate BOTH the
    # original and the fitted curve over the same first-pass window so the
    # ratio is apples-to-apples (else the asymmetric ratio misfires on any
    # AIF with non-trivial post-bolus tail — preloaded protocols always,
    # and any clean recirculation pattern). See Welker 2015 AJNR 36:E41
    # and Dickie 2024 MRM 91:1761 (OSIPI lexicon).
    if remove_recirculation:
        cutoff_for_auc = min(
            peak_idx + max(10, 2 * compute_aif_fwhm_frames(aif_curve, baseline_frames)),
            n_frames,
        )
        auc_original = _compute_auc(
            curve_shifted[baseline_frames:cutoff_for_auc], tr,
        )
        auc_target = auc_original
        auc_fitted = _compute_auc(
            fitted_curve[baseline_frames:cutoff_for_auc], tr,
        )
        first_pass_n = max(0, cutoff_for_auc - baseline_frames)
    else:
        auc_original = _compute_auc(curve_shifted, tr)
        auc_target = auc_original
        auc_fitted = _compute_auc(fitted_curve, tr)
        first_pass_n = max(0, n_frames - baseline_frames)

    # Compute AUC ratio BEFORE preservation scaling so it reflects the
    # actual shape mismatch between the gamma fit and the empirical
    # first-pass area. With preserve_auc=True (production default), scaling
    # forces auc_fitted == auc_target by construction, which would make a
    # post-scaling ratio mechanically 1.0 and useless as a QC metric.
    if auc_original > EPSILON_CONCENTRATION:
        auc_ratio = auc_fitted / auc_original
    else:
        auc_ratio = 1.0

    # Optionally scale to preserve AUC (first-pass or full, depending on mode)
    if preserve_auc and auc_fitted > EPSILON_CONCENTRATION:
        scale_factor = auc_target / auc_fitted
        fitted_curve = fitted_curve * scale_factor
        amplitude = amplitude * scale_factor
        auc_fitted = auc_target  # by construction

    return GammaVariateFitResult(
        fitted_curve=fitted_curve,
        amplitude=amplitude,
        t0=t0,
        alpha=alpha,
        beta=beta,
        auc_original=auc_original,
        auc_fitted=auc_fitted,
        auc_ratio=auc_ratio,
        residual_rms=residual_rms,
        r_squared=r_squared,
        fit_converged=fit_converged,
        recirculation_removed=recirculation_removed,
        first_pass_n=first_pass_n,
    )


# =============================================================================
# Auto Gamma-Variate Decision Logic
# =============================================================================

# Hard gate thresholds — fit must satisfy ALL to be eligible
GAMMA_R2_THRESHOLD: float = 0.85
GAMMA_PEAK_RATIO_MIN: float = 0.90
GAMMA_PEAK_RATIO_MAX: float = 1.10
# AUC ratio bounds retained for QC provenance only — no longer a hard gate.
# Symmetric-window auc_ratio (see ``fit_gamma_variate``) makes this metric
# meaningful, but R² + peak_ratio + defect detection already cover the
# pathology this band would catch (Welker 2015 AJNR 36:E41, Dickie 2024 MRM 91:1761).
GAMMA_AUC_RATIO_MIN: float = 0.85
GAMMA_AUC_RATIO_MAX: float = 1.15
# Soft warning band: auc_ratio outside this is logged in QC but does not
# trigger fallback (Welker 2015 AJNR 36:E41).
GAMMA_AUC_RATIO_SOFT_MIN: float = 0.70
GAMMA_AUC_RATIO_SOFT_MAX: float = 1.30
# Minimum first-pass samples for gamma fitting to be well-posed (Welker
# 2015 AJNR 36:E41, Calamante 2013 Prog Nucl Magn Reson Spectrosc 74:1).
GAMMA_MIN_FIRST_PASS_SAMPLES: int = 8

# Defect detection thresholds
GAMMA_SPR_THRESHOLD: float = 0.25
GAMMA_NOISE_RATIO_THRESHOLD: float = 2.5
GAMMA_NEG_TAIL_MIN_FRAMES: int = 3
GAMMA_NEG_TAIL_FRACTION: float = 0.05


@dataclass(frozen=True)
class GammaAutoDecision:
    """Result of the auto gamma-variate decision.

    Attributes
    ----------
    apply : bool
        Whether to apply the gamma-variate fit.
    reason : str
        Human-readable reason for the decision.
    defect_detected : str or None
        Type of defect detected: "recirculation",
        "noise_and_negative_tail", or None.
    """

    apply: bool
    reason: str
    defect_detected: Optional[str] = None


def should_apply_gamma_fit(
    original_aif: npt.NDArray[Any],
    fit_result: GammaVariateFitResult,
    baseline_frames: int = 5,
) -> tuple["GammaAutoDecision", dict[str, Any]]:
    """Decide whether to apply gamma-variate fitting to the AIF.

    Uses hard gates (fit quality) and defect triggers (recirculation,
    noise + negative tail) to decide. The gamma-variate is applied only
    when the fit is good AND a defect is detected that gamma fitting
    would correct.

    Hard gates (must all pass for eligibility):
        - ``fit_converged``
        - ``first_pass_n >= GAMMA_MIN_FIRST_PASS_SAMPLES`` (8 samples).
          Below this, gamma fitting is ill-posed regardless of metrics.
        - ``r_squared >= GAMMA_R2_THRESHOLD``
        - ``peak_ratio in [GAMMA_PEAK_RATIO_MIN, GAMMA_PEAK_RATIO_MAX]``

    AUC ratio is NOT a hard gate; it is reported in ``metrics["auc_ratio"]``
    for QC provenance, with ``metrics["auc_ratio_warning"]`` set when the
    symmetric-window ratio falls outside ``[GAMMA_AUC_RATIO_SOFT_MIN,
    GAMMA_AUC_RATIO_SOFT_MAX]``. R² + peak_ratio + defect detection already
    cover the pathology a hard AUC gate would catch (physics review:
    Welker AJNR 2015, Dickie et al. MRM 2024 OSIPI lexicon).

    Parameters
    ----------
    original_aif : np.ndarray
        1D original AIF concentration curve (before gamma fitting).
    fit_result : GammaVariateFitResult
        Result from ``fit_gamma_variate()``.
    baseline_frames : int
        Number of pre-bolus baseline frames.

    Returns
    -------
    tuple[GammaAutoDecision, dict[str, Any]]
        Decision and detailed metrics for QC logging.
    """
    metrics: dict[str, Any] = {}

    # --- Hard gates ---
    if not fit_result.fit_converged:
        return (
            GammaAutoDecision(apply=False, reason="Fit did not converge"),
            metrics,
        )

    # Eligibility prerequisite: gamma fitting is ill-posed below ~8 first-pass
    # samples (Welker 2015 AJNR 36:E41, Calamante 2013 Prog Nucl Magn Reson Spectrosc 74:1).
    metrics["first_pass_n"] = fit_result.first_pass_n
    if fit_result.first_pass_n < GAMMA_MIN_FIRST_PASS_SAMPLES:
        return (
            GammaAutoDecision(
                apply=False,
                reason=(
                    f"insufficient first-pass samples for gamma fit "
                    f"({fit_result.first_pass_n} < {GAMMA_MIN_FIRST_PASS_SAMPLES})"
                ),
            ),
            metrics,
        )

    if fit_result.r_squared < GAMMA_R2_THRESHOLD:
        metrics["r_squared"] = fit_result.r_squared
        return (
            GammaAutoDecision(
                apply=False,
                reason=f"R2={fit_result.r_squared:.3f} < {GAMMA_R2_THRESHOLD}",
            ),
            metrics,
        )

    peak_orig = float(np.max(original_aif))
    peak_fit = float(np.max(fit_result.fitted_curve))
    if peak_orig > EPSILON_CONCENTRATION:
        peak_ratio = peak_fit / peak_orig
    else:
        peak_ratio = 1.0
    metrics["peak_ratio"] = peak_ratio

    if not (GAMMA_PEAK_RATIO_MIN <= peak_ratio <= GAMMA_PEAK_RATIO_MAX):
        return (
            GammaAutoDecision(
                apply=False,
                reason=f"Peak ratio {peak_ratio:.3f} outside [{GAMMA_PEAK_RATIO_MIN}, {GAMMA_PEAK_RATIO_MAX}]",
            ),
            metrics,
        )

    # AUC ratio: reported for QC provenance with a soft warning band, but
    # NOT a hard gate. R² + peak_ratio + defect detection already cover the
    # pathology this gate would catch (physics review: Welker 2015 AJNR 36:E41,
    # Dickie 2024 MRM 91:1761).
    auc_ratio = fit_result.auc_ratio
    metrics["auc_ratio"] = auc_ratio
    metrics["auc_ratio_warning"] = not (
        GAMMA_AUC_RATIO_SOFT_MIN <= auc_ratio <= GAMMA_AUC_RATIO_SOFT_MAX
    )

    # --- Defect detection ---
    n = len(original_aif)
    peak_idx = int(np.argmax(original_aif))
    baseline_mean = float(np.mean(original_aif[:baseline_frames])) if baseline_frames > 0 else 0.0
    peak_val = peak_orig - baseline_mean  # baseline-corrected peak

    # 1) Recirculation defect: secondary peak ratio > threshold AND is local max
    # Look in post-first-pass window
    fwhm = compute_aif_fwhm_frames(original_aif, baseline_frames)
    recirc_start = peak_idx + max(fwhm, 5)
    spr = 0.0
    spr_is_local_max = False

    if recirc_start < n - 1 and peak_val > EPSILON_CONCENTRATION:
        tail = original_aif[recirc_start:] - baseline_mean
        if len(tail) > 2:
            secondary_peak_idx_local = int(np.argmax(tail))
            secondary_peak_val = float(tail[secondary_peak_idx_local])
            spr = secondary_peak_val / peak_val if peak_val > EPSILON_CONCENTRATION else 0.0

            # Check if it's a true local maximum (derivative sign change)
            abs_idx = recirc_start + secondary_peak_idx_local
            if 0 < secondary_peak_idx_local < len(tail) - 1:
                left_slope = tail[secondary_peak_idx_local] - tail[secondary_peak_idx_local - 1]
                right_slope = tail[secondary_peak_idx_local + 1] - tail[secondary_peak_idx_local]
                spr_is_local_max = left_slope > 0 and right_slope < 0

    metrics["spr"] = spr
    metrics["spr_is_local_max"] = spr_is_local_max
    recirculation_defect = spr > GAMMA_SPR_THRESHOLD and spr_is_local_max

    # 2) Noise defect: 2nd derivative L2 norm ratio (fitted vs original) > threshold
    if n > 4:
        d2_orig = np.diff(original_aif, n=2)
        d2_fit = np.diff(fit_result.fitted_curve, n=2)
        l2_orig = float(np.sqrt(np.sum(d2_orig**2)))
        l2_fit = float(np.sqrt(np.sum(d2_fit**2)))
        noise_ratio = l2_orig / l2_fit if l2_fit > EPSILON_CONCENTRATION else 0.0
    else:
        noise_ratio = 0.0
    metrics["noise_ratio"] = noise_ratio
    noise_defect = noise_ratio > GAMMA_NOISE_RATIO_THRESHOLD

    # 3) Negative tail defect: >N frames below zero AND min < -fraction of peak
    if peak_idx < n - 1 and peak_val > EPSILON_CONCENTRATION:
        tail_full = original_aif[peak_idx:] - baseline_mean
        neg_frames = int(np.sum(tail_full < 0))
        neg_min = float(np.min(tail_full))
        neg_min_fraction = abs(neg_min) / peak_val
    else:
        neg_frames = 0
        neg_min_fraction = 0.0
    metrics["neg_tail_frames"] = neg_frames
    metrics["neg_tail_min_fraction"] = neg_min_fraction
    negative_tail_defect = (
        neg_frames >= GAMMA_NEG_TAIL_MIN_FRAMES
        and neg_min_fraction > GAMMA_NEG_TAIL_FRACTION
    )

    # --- Decision ---
    if recirculation_defect:
        return (
            GammaAutoDecision(
                apply=True,
                reason=f"Recirculation detected (SPR={spr:.3f}, local max)",
                defect_detected="recirculation",
            ),
            metrics,
        )

    if noise_defect and negative_tail_defect:
        return (
            GammaAutoDecision(
                apply=True,
                reason=(
                    f"Noise (ratio={noise_ratio:.2f}) and negative tail "
                    f"({neg_frames} frames, min={neg_min_fraction:.3f}*peak)"
                ),
                defect_detected="noise_and_negative_tail",
            ),
            metrics,
        )

    return (
        GammaAutoDecision(
            apply=False,
            reason="No defect detected -- AIF is clean",
        ),
        metrics,
    )


def fit_gamma_variate_from_file(
    aif_file: Union[str, Path],
    baseline_frames: int = 5,
    remove_recirculation: bool = True,
    preserve_auc: bool = True,
    tr: Optional[float] = None,
) -> GammaVariateFitResult:
    """Fit gamma-variate to an AIF from a text file.

    Convenience wrapper that loads AIF from file.

    Parameters
    ----------
    aif_file : str or Path
        Path to text file containing 1D AIF curve.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    remove_recirculation : bool
        If True, remove recirculation from fit.
    preserve_auc : bool
        If True, scale fitted curve to match original AUC.
    tr : float, optional
        Repetition time in seconds. If None, defaults to 1.0.

    Returns
    -------
    GammaVariateFitResult
        Fitted parameters and curve.
    """
    aif_curve = np.loadtxt(aif_file)

    if tr is None:
        tr = 1.0

    return fit_gamma_variate(
        aif_curve=aif_curve,
        baseline_frames=baseline_frames,
        remove_recirculation=remove_recirculation,
        preserve_auc=preserve_auc,
        tr=tr,
    )


# =============================================================================
# Synthetic AIF Generation
# =============================================================================


# Population parameters for synthetic gamma-variate AIF
# Based on literature values: Parker et al. (2006), Mouridsen et al. (2006)
SYNTHETIC_AIF_PARAMS: dict[str, float] = {
    # Gamma-variate parameters (typical values for healthy subjects at 3T)
    "alpha": 2.5,  # Shape parameter (typical 2-3)
    "beta": 3.0,  # Scale parameter in frames (typical 2.5-4)
    # Timing parameters
    "arrival_frame": 5.0,  # Typical bolus arrival (frames after start)
    # Amplitude scaling factors by dose
    "peak_deltaR2_full_dose": 40.0,  # s^-1, typical peak ΔR2* for full dose
    "peak_deltaR2_half_dose": 20.0,  # s^-1, typical peak ΔR2* for half dose
    "peak_deltaR2_double_dose": 80.0,  # s^-1, typical peak ΔR2* for double dose
}


@dataclass
class SyntheticAIFResult:
    """Result of synthetic AIF generation.

    Synthetic AIFs are used as a fallback when automatic AIF detection fails.
    They enable relative perfusion quantification but are NOT suitable for
    absolute CBV/CBF measurements.

    Attributes
    ----------
    aif_curve : npt.NDArray[Any]
        Generated synthetic AIF curve.
    amplitude : float
        Amplitude scaling factor used.
    t0 : float
        Bolus arrival time (frames).
    alpha : float
        Gamma-variate shape parameter.
    beta : float
        Gamma-variate scale parameter.
    dose_level : float
        Estimated dose multiplier (e.g. 0.5, 1.0, 2.0).
    scaling_factor : float
        Factor applied to match estimated global signal.
    is_synthetic : bool
        Always True - indicates synthetic AIF.
    absolute_quantification_valid : bool
        Always False - pipeline produces relative quantification only.
    warning_message : str
        Warning about limitations of synthetic AIF.
    qc_metrics : dict[str, Any]
        Quality metrics including source information.
    """

    aif_curve: npt.NDArray[Any]
    amplitude: float
    t0: float
    alpha: float
    beta: float
    dose_level: float
    scaling_factor: float
    is_synthetic: bool = True
    absolute_quantification_valid: bool = False
    warning_message: str = ""
    qc_metrics: dict[str, Any] = field(default_factory=dict)


def _estimate_dose_from_global_signal(
    concentration_data: npt.NDArray[Any],
    brain_mask: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> Tuple[float, float, float]:
    """Estimate contrast dose from global signal characteristics.

    Uses the peak ΔR2* of the global average signal to estimate the
    contrast dose level as a continuous float multiplier.

    Parameters
    ----------
    concentration_data : np.ndarray
        4D concentration/ΔR2* data (x, y, z, time).
    brain_mask : np.ndarray
        3D brain mask.
    baseline_frames : int
        Number of pre-bolus baseline frames.

    Returns
    -------
    Tuple[float, float, float]
        (dose_level, peak_deltaR2, scaling_factor)
        dose_level: Continuous dose multiplier (e.g. 0.5, 1.0, 2.0)
        peak_deltaR2: Estimated peak ΔR2* value
        scaling_factor: Factor to apply to population AIF
    """
    # Compute global average signal within brain
    mask_indices = np.where(brain_mask > 0)
    if len(mask_indices[0]) == 0:
        # Empty mask, return defaults
        return (1.0, 40.0, 1.0)

    # Extract time courses and compute mean
    time_courses = concentration_data[mask_indices]  # (n_voxels, n_frames)
    global_signal = np.mean(time_courses, axis=0)

    # Compute peak ΔR2* (relative to baseline)
    baseline_mean: float = float(np.mean(global_signal[:baseline_frames]))
    peak_value: float = float(np.max(global_signal))
    peak_deltaR2 = peak_value - baseline_mean

    # Estimate dose as continuous multiplier relative to full-dose reference.
    # Full dose peak ΔR2* ≈ 40 s^-1 at 3T; ΔR2* scales linearly with dose.
    full_dose_peak = SYNTHETIC_AIF_PARAMS["peak_deltaR2_full_dose"]
    if full_dose_peak > EPSILON_CONCENTRATION:
        dose_level = max(peak_deltaR2, EPSILON_CONCENTRATION) / full_dose_peak
    else:
        dose_level = 1.0

    # With continuous dose estimation, dose_level already captures the
    # observed-vs-expected ratio — scaling_factor is unity by definition.
    # Callers that previously relied on scaling_factor should use dose_level.
    scaling_factor = 1.0

    return (dose_level, peak_deltaR2, scaling_factor)


def generate_synthetic_aif(
    n_frames: int,
    baseline_frames: int = 5,
    dose_level: float = 1.0,
    field_strength: float = 3.0,
    global_scaling: Optional[float] = None,
    concentration_data: Optional[npt.NDArray[Any]] = None,
    brain_mask: Optional[npt.NDArray[Any]] = None,
) -> SyntheticAIFResult:
    """Generate a synthetic AIF using population-derived gamma-variate parameters.

    .. deprecated::
        Legacy utility with different parameters than the production
        ``AIFDetection._generate_synthetic_aif``. Retained for test
        compatibility only; production uses the interface method.

    This function creates a synthetic arterial input function when automatic
    AIF detection fails. The synthetic AIF enables relative perfusion
    quantification but absolute CBV/CBF measurements are NOT valid.

    The synthetic AIF is generated using a gamma-variate model with
    population-derived parameters from the literature (Parker et al. 2006,
    Mouridsen et al. 2006).

    Parameters
    ----------
    n_frames : int
        Number of time frames for the AIF curve.
    baseline_frames : int
        Number of pre-bolus baseline frames (AIF = 0 for these frames).
    dose_level : float
        Contrast dose multiplier (0.5=half, 1.0=full, 2.0=double).
        Used to set the AIF amplitude via linear scaling.
    field_strength : float
        Magnetic field strength in Tesla. Affects expected relaxivity.
    global_scaling : float, optional
        Custom scaling factor for AIF amplitude. If None, uses dose_level defaults.
    concentration_data : np.ndarray, optional
        4D concentration data for automatic dose estimation.
        If provided along with brain_mask, dose will be estimated from global signal.
    brain_mask : np.ndarray, optional
        3D brain mask for global signal calculation.

    Returns
    -------
    SyntheticAIFResult
        Synthetic AIF with parameters and warning flags.

    Notes
    -----
    **WARNING**: Synthetic AIFs have significant limitations:

    1. **Absolute quantification is NOT valid**: CBF, CBV in absolute units
       (mL/100g/min, mL/100g) require a measured AIF. Synthetic AIFs provide
       only relative values.

    2. **Inter-subject comparisons are limited**: Without measured AIF,
       differences between subjects may reflect AIF variations rather than
       true perfusion differences.

    3. **Within-subject comparisons are acceptable**: Relative CBF/CBV
       differences within the same scan (e.g., lesion vs normal tissue)
       remain valid.

    4. **Normalization recommended**: Use relative CBV (rCBV = CBV / CBV_NAWM)
       to partially compensate for AIF uncertainty.

    The synthetic AIF is flagged with:
    - is_synthetic = True
    - absolute_quantification_valid = False
    - warning_message explaining limitations

    Downstream processing should check these flags and:
    - Mark outputs as relative-only
    - Warn users about quantification limitations
    - Recommend rCBV normalization

    Clinical context:
    - Synthetic AIFs may be necessary in tumor patients where lesions
      affect vessel selection
    - Multi-site studies benefit from synthetic AIFs for consistency
    - Research protocols should document synthetic AIF usage

    Examples
    --------
    >>> # Generate synthetic AIF for 60-frame scan, full dose
    >>> result = generate_synthetic_aif(n_frames=60, baseline_frames=5, dose_level=1.0)  # doctest: +SKIP
    >>> print(f"Peak ΔR2*: {result.aif_curve.max():.1f} s^-1")  # doctest: +SKIP
    >>> print(f"Warning: {result.warning_message}")  # doctest: +SKIP

    >>> # Auto-estimate dose from global signal
    >>> result = generate_synthetic_aif(  # doctest: +SKIP
    ...     n_frames=60,
    ...     baseline_frames=5,
    ...     concentration_data=conc_4d,
    ...     brain_mask=mask,
    ... )
    >>> print(f"Estimated dose: {result.dose_level}")  # doctest: +SKIP

    See Also
    --------
    atlas_based_aif_search : Preferred AIF detection method
    fit_gamma_variate : Gamma-variate fitting for measured AIFs
    """
    import warnings
    warnings.warn(
        "generate_synthetic_aif() uses legacy parameters. "
        "Production uses AIFDetection._generate_synthetic_aif().",
        DeprecationWarning,
        stacklevel=2,
    )
    # Get population parameters
    alpha = SYNTHETIC_AIF_PARAMS["alpha"]
    beta = SYNTHETIC_AIF_PARAMS["beta"]
    t0 = SYNTHETIC_AIF_PARAMS["arrival_frame"]

    # Ensure baseline is respected
    if t0 < baseline_frames:
        t0 = float(baseline_frames)

    # Estimate dose from global signal if data provided
    estimated_dose = dose_level
    estimated_peak = 0.0
    auto_scaling = 1.0

    if concentration_data is not None and brain_mask is not None:
        estimated_dose, estimated_peak, auto_scaling = _estimate_dose_from_global_signal(
            concentration_data, brain_mask, baseline_frames
        )
        # Use estimated dose if not explicitly specified
        if global_scaling is None:
            global_scaling = auto_scaling
            dose_level = estimated_dose

    # Determine amplitude from dose level — linear scaling from full-dose reference
    base_amplitude = SYNTHETIC_AIF_PARAMS["peak_deltaR2_full_dose"] * float(dose_level)

    # Apply field strength correction (approximate)
    # r2* scales roughly linearly with field strength
    field_factor = field_strength / 3.0  # Normalize to 3T
    base_amplitude = base_amplitude * field_factor

    # Apply custom scaling if provided
    if global_scaling is not None:
        amplitude = base_amplitude * global_scaling
    else:
        amplitude = base_amplitude

    # Generate time array
    t = np.arange(n_frames, dtype=np.float64)

    # Generate gamma-variate AIF
    # The peak of gamma-variate occurs at t0 + alpha*beta
    # We need to scale amplitude so that max(curve) = target peak
    # For gamma-variate: max = A * (alpha)^alpha * exp(-alpha) / (beta^alpha)
    # So: A = target_peak * (beta^alpha) / (alpha^alpha * exp(-alpha))

    peak_term = np.power(alpha, alpha) * np.exp(-alpha) / np.power(beta, alpha)
    if peak_term > EPSILON_CONCENTRATION:
        amplitude_scaled = amplitude / peak_term
    else:
        amplitude_scaled = amplitude

    # Generate the curve
    aif_curve = _gamma_variate_model(t, amplitude_scaled, t0, alpha, beta)

    # Ensure baseline is exactly zero
    aif_curve[:baseline_frames] = 0.0

    # Build warning message
    warning_parts = [
        "SYNTHETIC AIF WARNING:",
        "This AIF was generated using population parameters because automatic detection failed.",
        "LIMITATIONS:",
        "  - Absolute CBF/CBV values are NOT valid",
        "  - Inter-subject comparisons are unreliable",
        "  - Use relative values (rCBV = CBV/CBV_NAWM) for clinical interpretation",
        "RECOMMENDATIONS:",
        "  - Report rCBV instead of absolute CBV",
        "  - Note synthetic AIF usage in analysis records",
        "  - Consider re-acquisition if absolute values are needed",
    ]
    warning_message = "\n".join(warning_parts)

    # Build QC metrics
    qc_metrics: dict[str, Any] = {
        "source": "synthetic",
        "population_params": {
            "alpha": alpha,
            "beta": beta,
            "t0": t0,
        },
        "dose_level": dose_level,
        "field_strength": field_strength,
        "scaling_factor": global_scaling if global_scaling is not None else 1.0,
        "peak_value": float(np.max(aif_curve)),
        "auc": float(_compute_auc(aif_curve)),
        "fwhm_frames": compute_aif_fwhm_frames(aif_curve, baseline_frames),
        "auto_estimated": concentration_data is not None,
        "estimated_peak_from_signal": estimated_peak if estimated_peak > 0 else None,
    }

    return SyntheticAIFResult(
        aif_curve=aif_curve,
        amplitude=amplitude_scaled,
        t0=t0,
        alpha=alpha,
        beta=beta,
        dose_level=dose_level,
        scaling_factor=global_scaling if global_scaling is not None else 1.0,
        is_synthetic=True,
        absolute_quantification_valid=False,
        warning_message=warning_message,
        qc_metrics=qc_metrics,
    )


def generate_synthetic_aif_from_files(
    n_frames: int,
    brain_mask_file: Optional[Union[str, Path]] = None,
    concentration_file: Optional[Union[str, Path]] = None,
    baseline_frames: int = 5,
    dose_level: float = 1.0,
    field_strength: float = 3.0,
) -> SyntheticAIFResult:
    """Generate synthetic AIF with optional automatic dose estimation from files.

    Convenience wrapper that loads NIfTI files for automatic dose estimation.

    Parameters
    ----------
    n_frames : int
        Number of time frames.
    brain_mask_file : str or Path, optional
        Path to brain mask NIfTI file.
    concentration_file : str or Path, optional
        Path to 4D concentration NIfTI file.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    dose_level : float
        Contrast dose multiplier if not auto-estimated (default 1.0).
    field_strength : float
        Magnetic field strength in Tesla.

    Returns
    -------
    SyntheticAIFResult
        Synthetic AIF with parameters and warning flags.
    """
    concentration_data: Optional[npt.NDArray[Any]] = None
    brain_mask: Optional[npt.NDArray[Any]] = None

    if concentration_file is not None and brain_mask_file is not None:
        import nibabel as nib  # type: ignore[import-untyped]

        conc_img = nib.load(concentration_file)
        concentration_data = conc_img.get_fdata()  # type: ignore[attr-defined]

        mask_img = nib.load(brain_mask_file)
        brain_mask = mask_img.get_fdata()  # type: ignore[attr-defined]

    return generate_synthetic_aif(
        n_frames=n_frames,
        baseline_frames=baseline_frames,
        dose_level=dose_level,
        field_strength=field_strength,
        concentration_data=concentration_data,
        brain_mask=brain_mask,
    )
