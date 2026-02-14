"""T1 map validation and correction utilities for VFA T1 mapping.

This module provides field-strength-dependent tissue T1 reference values,
T1 map correction strategies (auto/offset/fallback), and cross-validation
of field strength from measured T1 distributions.

Ported from perf-plus dce_vfa_t1_mapping.py with adaptations for
NiPreps-style dataclasses and logging.

References
----------
Wansapura JP et al. "NMR relaxation times in the human brain at 3.0
Tesla." JMRI. 1999;9(4):531-538.

Stanisz GJ et al. "T1, T2 relaxation and magnetization transfer in
tissue at 3T." MRM. 2005;54(3):507-512.

Rooney WD et al. "Magnetic field and tissue dependencies of human
brain longitudinal 1H2O relaxation in vivo." MRM. 2007;57(2):308-318.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("pmriprep.utils.dce_t1_validation")

# Tissue-bias thresholds for auto-correction override (ms).
# If |median - expected| exceeds this even within warn range, apply offset.
TISSUE_BIAS_THRESHOLD_MS: float = 150.0
# Lowered threshold when B1 QC has failed (residual B1 inhomogeneity likely).
B1_FAIL_BIAS_THRESHOLD_MS: float = 100.0

# ============================================================================
# T1 Tissue Reference Values by Field Strength
# ============================================================================

T1_TISSUE_REFERENCE: Dict[float, Dict[str, Any]] = {
    1.5: {
        "wm": (550, 750),
        "gm": (900, 1100),
        "brain_median": (700, 950),
        "brain_mean": 825,
        "brain_sd": 150,
        "qc_min": 525,
        "qc_max_strict": 1125,
        "qc_max_lenient": 1275,
        "csf": 3000,
        "default": 900,
        "t1_upper_bound": 5000,
        "reject_min": 450,
        "reject_max": 1500,
        "warn_low": 650,
        "warn_high": 1000,
        "expected_median": 825,
    },
    3.0: {
        "wm": (800, 1000),
        "gm": (1200, 1400),
        "brain_median": (900, 1200),
        "brain_mean": 1050,
        "brain_sd": 175,
        "qc_min": 700,
        "qc_max_strict": 1400,
        "qc_max_lenient": 1575,
        "csf": 4000,
        "default": 1200,
        "t1_upper_bound": 6000,
        "reject_min": 600,
        "reject_max": 1800,
        "warn_low": 850,
        "warn_high": 1250,
        "expected_median": 1050,
    },
}


# ============================================================================
# Dataclasses for results
# ============================================================================


@dataclass
class FieldStrengthValidation:
    """Result of cross-validating field strength from measured T1."""

    field_strength_valid: bool = True
    confidence: str = "high"
    warning: Optional[str] = None
    suggested_field_strength: Optional[float] = None
    distance_specified: float = 0.0
    distance_opposite: float = 0.0
    specified_field_t: float = 3.0
    opposite_field_t: float = 1.5
    expected_range_specified: Tuple[float, float] = (900, 1200)
    expected_range_opposite: Tuple[float, float] = (700, 950)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "field_strength_valid": self.field_strength_valid,
            "confidence": self.confidence,
            "warning": self.warning,
            "suggested_field_strength": self.suggested_field_strength,
            "distance_specified": self.distance_specified,
            "distance_opposite": self.distance_opposite,
            "specified_field_t": self.specified_field_t,
            "opposite_field_t": self.opposite_field_t,
            "expected_range_specified": list(self.expected_range_specified),
            "expected_range_opposite": list(self.expected_range_opposite),
        }


@dataclass
class T1CorrectionResult:
    """Result of T1 map correction."""

    correction_applied: str = "none"
    reason: str = ""
    median_t1_original: float = 0.0
    median_t1_corrected: float = 0.0
    expected_median: float = 0.0
    field_strength_t: float = 3.0
    has_lesion: bool = False
    correction_mode_requested: str = "auto"
    offset_ms: Optional[float] = None
    fallback_value_ms: Optional[float] = None
    deviation_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "correction_applied": self.correction_applied,
            "reason": self.reason,
            "median_t1_original": self.median_t1_original,
            "median_t1_corrected": self.median_t1_corrected,
            "expected_median": self.expected_median,
            "field_strength_t": self.field_strength_t,
            "has_lesion": self.has_lesion,
            "correction_mode_requested": self.correction_mode_requested,
        }
        if self.offset_ms is not None:
            d["offset_ms"] = self.offset_ms
        if self.fallback_value_ms is not None:
            d["fallback_value_ms"] = self.fallback_value_ms
        if self.deviation_type is not None:
            d["deviation_type"] = self.deviation_type
        return d


# ============================================================================
# Helper functions
# ============================================================================


def get_field_strength_key(field_strength: float) -> float:
    """Round field strength to nearest standard value (1.5 or 3.0).

    Parameters
    ----------
    field_strength : float
        Scanner field strength in Tesla.

    Returns
    -------
    float
        Nearest standard field strength (1.5 or 3.0).

    Raises
    ------
    ValueError
        If field_strength is outside plausible range (0.5-7.0 T).
    """
    if not np.isfinite(field_strength):
        raise ValueError(f"Field strength must be finite, got {field_strength}")
    if field_strength <= 0:
        raise ValueError(f"Field strength must be positive, got {field_strength} T")
    if field_strength < 0.5:
        raise ValueError(
            f"Field strength {field_strength} T is suspiciously low. "
            f"Did you specify in mT instead of T? (Expected 1.5 or 3.0 T)"
        )
    if field_strength > 7.0:
        raise ValueError(
            f"Field strength {field_strength} T exceeds maximum supported (7.0 T). "
            f"Check units - value should be in Tesla."
        )
    return 1.5 if field_strength < 2.25 else 3.0


def _distance_from_range(value: float, range_tuple: Tuple[float, float]) -> float:
    """Calculate distance from expected range (0 if within range)."""
    low, high = range_tuple
    if low <= value <= high:
        return 0.0
    elif value < low:
        return low - value
    else:
        return value - high


# ============================================================================
# Cross-validation
# ============================================================================


def cross_validate_field_strength(
    median_t1: float,
    specified_field_strength: float,
) -> FieldStrengthValidation:
    """Cross-validate field strength by checking if measured T1 matches expectations.

    T1 relaxation times are field-strength-dependent:
    - At 1.5T: WM ~600-700ms, GM ~900-1100ms, brain median ~700-950ms
    - At 3.0T: WM ~800-1000ms, GM ~1200-1400ms, brain median ~900-1200ms

    If the measured T1 is significantly closer to the opposite field strength's
    expected range, warn the user that field strength may be mis-specified.

    Parameters
    ----------
    median_t1 : float
        Measured median T1 in milliseconds.
    specified_field_strength : float
        User-specified or auto-detected field strength in Tesla.

    Returns
    -------
    FieldStrengthValidation
        Validation result with confidence and optional warning.
    """
    field_key = get_field_strength_key(specified_field_strength)
    opposite_key = 3.0 if field_key == 1.5 else 1.5

    expected_specified = T1_TISSUE_REFERENCE[field_key]["brain_median"]
    expected_opposite = T1_TISSUE_REFERENCE[opposite_key]["brain_median"]

    dist_specified = _distance_from_range(median_t1, expected_specified)
    dist_opposite = _distance_from_range(median_t1, expected_opposite)

    result = FieldStrengthValidation(
        field_strength_valid=True,
        confidence="high",
        specified_field_t=field_key,
        opposite_field_t=opposite_key,
        distance_specified=dist_specified,
        distance_opposite=dist_opposite,
        expected_range_specified=expected_specified,
        expected_range_opposite=expected_opposite,
    )

    MISMATCH_THRESHOLD_MS = 100
    SIGNIFICANT_ERROR_MS = 150

    if (
        dist_specified > SIGNIFICANT_ERROR_MS
        and dist_opposite < dist_specified - MISMATCH_THRESHOLD_MS
    ):
        result.field_strength_valid = False
        result.confidence = "low"
        result.suggested_field_strength = opposite_key
        result.warning = (
            f"FIELD STRENGTH MISMATCH: Measured median T1 ({median_t1:.0f}ms) is closer to "
            f"{opposite_key}T expected range ({expected_opposite[0]}-{expected_opposite[1]}ms) "
            f"than {field_key}T expected range "
            f"({expected_specified[0]}-{expected_specified[1]}ms). "
            f"Distance from {field_key}T: {dist_specified:.0f}ms, "
            f"distance from {opposite_key}T: {dist_opposite:.0f}ms."
        )
        logger.warning(result.warning)
    elif dist_specified > 0 and dist_specified <= SIGNIFICANT_ERROR_MS:
        result.confidence = "medium"

    return result


# ============================================================================
# T1 correction
# ============================================================================


def apply_t1_correction(
    t1_map: npt.NDArray[np.floating],
    field_strength: float,
    correction_mode: str = "auto",
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    has_lesion: bool = False,
    b1_qc_pass: bool = True,
) -> Tuple[npt.NDArray[np.floating], T1CorrectionResult]:
    """Apply T1 map correction based on median T1 validation.

    Correction modes:
    - ``none``: No correction, use VFA T1 as-is.
    - ``offset``: Apply uniform offset to match expected median
      (preserves spatial variation).
    - ``fallback``: Replace with uniform literature T1 value.
    - ``auto``: Automatically choose based on median T1 deviation.

    Parameters
    ----------
    t1_map : ndarray
        T1 map in milliseconds.
    field_strength : float
        Scanner field strength in Tesla (1.5 or 3.0).
    correction_mode : str
        Correction strategy: ``none``, ``offset``, ``fallback``, or ``auto``.
    brain_mask : ndarray, optional
        Brain mask for calculating median T1.
    has_lesion : bool
        Whether a lesion mask is present (affects auto-mode thresholds).
    b1_qc_pass : bool
        B1 QC pass status. When False, lowers tissue-bias threshold
        for auto correction (residual B1 inhomogeneity likely).

    Returns
    -------
    t1_corrected : ndarray
        Corrected T1 map in milliseconds.
    correction_result : T1CorrectionResult
        Information about the correction applied.
    """
    field_key = get_field_strength_key(field_strength)
    t1_ref = T1_TISSUE_REFERENCE[field_key]

    # Calculate median T1 from valid voxels
    if brain_mask is not None:
        valid_mask = (brain_mask > 0.5) & ~np.isnan(t1_map) & (t1_map > 0)
    else:
        valid_mask = ~np.isnan(t1_map) & (t1_map > 0)

    valid_t1 = t1_map[valid_mask]

    if len(valid_t1) == 0:
        return t1_map, T1CorrectionResult(
            correction_applied="none",
            reason="No valid T1 values",
            correction_mode_requested=correction_mode,
            field_strength_t=field_key,
        )

    median_t1 = float(np.median(valid_t1))
    expected_median = t1_ref["expected_median"]

    result = T1CorrectionResult(
        median_t1_original=median_t1,
        expected_median=expected_median,
        field_strength_t=field_key,
        has_lesion=has_lesion,
        correction_mode_requested=correction_mode,
    )

    if correction_mode == "none":
        result.correction_applied = "none"
        result.reason = "User requested no correction"
        result.median_t1_corrected = median_t1
        return t1_map, result

    elif correction_mode == "fallback":
        t1_corrected = np.where(valid_mask, t1_ref["default"], t1_map)
        result.correction_applied = "fallback"
        result.reason = "User requested fallback to literature value"
        result.fallback_value_ms = t1_ref["default"]
        result.median_t1_corrected = t1_ref["default"]
        return t1_corrected, result

    elif correction_mode == "offset":
        offset = expected_median - median_t1
        t1_corrected = t1_map + offset
        t1_corrected = np.clip(t1_corrected, t1_ref["reject_min"], t1_ref["reject_max"])
        t1_corrected = np.where(np.isnan(t1_map), np.nan, t1_corrected)
        result.correction_applied = "offset"
        result.reason = "User requested offset correction"
        result.offset_ms = offset
        result.median_t1_corrected = float(np.nanmedian(t1_corrected[valid_mask]))
        return t1_corrected, result

    elif correction_mode == "auto":
        return _auto_correct_t1(
            t1_map, t1_ref, valid_mask, median_t1, result,
            b1_qc_pass=b1_qc_pass,
        )

    else:
        raise ValueError(
            f"Unknown correction mode: {correction_mode}. "
            f"Valid options: 'none', 'offset', 'fallback', 'auto'"
        )


def _auto_correct_t1(
    t1_map: npt.NDArray[np.floating],
    t1_ref: Dict[str, Any],
    valid_mask: npt.NDArray[np.bool_],
    median_t1: float,
    result: T1CorrectionResult,
    b1_qc_pass: bool = True,
) -> Tuple[npt.NDArray[np.floating], T1CorrectionResult]:
    """Auto-mode T1 correction logic.

    Decides between no correction, offset, or fallback based on
    how far the median T1 deviates from expected values.
    """
    expected_median = t1_ref["expected_median"]

    # Case 1: Hard reject - impossible values
    if median_t1 < t1_ref["reject_min"]:
        logger.warning(
            "Median T1 (%.0fms) below reject threshold (%.0fms) - "
            "falling back to literature value (%.0fms)",
            median_t1, t1_ref["reject_min"], t1_ref["default"],
        )
        t1_corrected = np.where(valid_mask, t1_ref["default"], t1_map)
        result.correction_applied = "fallback"
        result.reason = (
            f"Median T1 below reject threshold ({t1_ref['reject_min']}ms)"
        )
        result.fallback_value_ms = t1_ref["default"]
        result.median_t1_corrected = t1_ref["default"]
        return t1_corrected, result

    elif median_t1 > t1_ref["reject_max"]:
        logger.warning(
            "Median T1 (%.0fms) above reject threshold (%.0fms) - "
            "falling back to literature value (%.0fms)",
            median_t1, t1_ref["reject_max"], t1_ref["default"],
        )
        t1_corrected = np.where(valid_mask, t1_ref["default"], t1_map)
        result.correction_applied = "fallback"
        result.reason = (
            f"Median T1 above reject threshold ({t1_ref['reject_max']}ms)"
        )
        result.fallback_value_ms = t1_ref["default"]
        result.median_t1_corrected = t1_ref["default"]
        return t1_corrected, result

    # Case 2: Within acceptable range
    elif t1_ref["warn_low"] <= median_t1 <= t1_ref["warn_high"]:
        # Secondary check: systematic bias despite acceptable median
        bias_threshold = (
            TISSUE_BIAS_THRESHOLD_MS if b1_qc_pass
            else B1_FAIL_BIAS_THRESHOLD_MS
        )
        bias_from_expected = abs(median_t1 - expected_median)
        if bias_from_expected > bias_threshold:
            # Fall through to offset correction (Case 3)
            logger.info(
                "Median T1 (%.0fms) within warn range but biased "
                "%.0fms from expected (%.0fms), threshold %.0fms%s "
                "- applying offset correction",
                median_t1, bias_from_expected, expected_median,
                bias_threshold,
                " (B1 QC failed)" if not b1_qc_pass else "",
            )
        else:
            logger.info(
                "Median T1 (%.0fms) within acceptable range (%.0f-%.0fms)",
                median_t1, t1_ref["warn_low"], t1_ref["warn_high"],
            )
            result.correction_applied = "none"
            result.reason = "Median T1 within acceptable range"
            result.median_t1_corrected = median_t1
            return t1_map, result

    # Case 3: Soft warn - apply offset correction
    # (also reached via tissue-bias override from Case 2)
    if median_t1 < expected_median:
        deviation_type = "low"
    else:
        deviation_type = "high"

    offset = expected_median - median_t1
    logger.info(
        "Median T1 (%.0fms) %s deviation - applying offset correction: %+.0fms",
        median_t1, deviation_type, offset,
    )

    t1_corrected = t1_map + offset
    clamp_min = t1_ref["reject_min"] * 0.9
    clamp_max = t1_ref["reject_max"] * 1.1
    t1_corrected = np.clip(t1_corrected, clamp_min, clamp_max)
    t1_corrected = np.where(np.isnan(t1_map), np.nan, t1_corrected)

    result.correction_applied = "offset"
    result.reason = f"Median T1 {deviation_type} - offset correction applied"
    result.deviation_type = deviation_type
    result.offset_ms = offset
    result.median_t1_corrected = float(np.nanmedian(t1_corrected[valid_mask]))
    return t1_corrected, result
