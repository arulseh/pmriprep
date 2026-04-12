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

Wright PJ et al. "Water proton T1 measurements in brain tissue at 7,
3, and 1.5 T using IR-EPI, IR-TSE, and MPRAGE." MRM. 2008;26(10):1362-1368.

Marques JP et al. "MP2RAGE, a self bias-field corrected sequence for
improved segmentation and T1-mapping at high field." NeuroImage.
2010;49(2):1271-1281.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("pmriprep.utils.dce_t1_validation")

# Tissue-bias thresholds for auto-correction override (ms).
# If |median - expected| exceeds this even within warn range, apply offset.
TISSUE_BIAS_THRESHOLD_MS: float = 150.0
# Lowered threshold when B1 QC has failed (residual B1 inhomogeneity likely).
B1_FAIL_BIAS_THRESHOLD_MS: float = 100.0

# VFA NaN fraction thresholds for tiered auto-fallback.
# Below this: VFA T1 is used as-is (no concern).
VFA_NAN_FRACTION_GOOD: float = 0.20
# Above this (for well-conditioned VFA): abandon VFA entirely.
VFA_NAN_FRACTION_TISSUE_FALLBACK: float = 0.40

# VFA conditioning thresholds (Deoni SCL et al. MRM 2003).
# VFA condition number = E1/(1-E1); higher means 1% B1 error
# propagates to larger T1 error.
# E1 > 0.997 → cond > 332 → fallback unconditionally
_E1_UNCONDITIONAL_FALLBACK: Final[float] = 0.997
# E1 > 0.993 → cond > 141 → fallback unless B1-corrected & B1 QC passed
_E1_CONDITIONAL_FALLBACK: Final[float] = 0.993


# ============================================================================
# T1 Tissue Reference Values by Field Strength
# ============================================================================

# brain_mean = 0.60 * wm_midpoint + 0.40 * gm_midpoint
# (approximate whole-brain WM:GM volume ratio, Ge et al. AJNR 2002)
# Derived values: qc_max_strict = gm[1], qc_max_lenient = qc_max_strict + brain_sd,
# expected_median = brain_mean, warn_high = expected_median + brain_sd
T1_TISSUE_REFERENCE: Dict[float, Dict[str, Any]] = {
    1.5: {
        "wm": (550, 750),
        "gm": (900, 1200),
        "brain_median": (700, 950),
        "brain_mean": 810,
        "brain_sd": 150,
        "qc_min": 525,
        "qc_max_strict": 1200,
        "qc_max_lenient": 1350,
        "csf": 3000,
        "default": 900,
        "t1_upper_bound": 5000,
        "t1_parenchyma_upper": 2000,
        "reject_min": 450,
        "reject_max": 1500,
        "warn_low": 650,
        "warn_high": 960,
        "expected_median": 810,
    },
    3.0: {
        "wm": (800, 1000),
        "gm": (1250, 1500),
        "brain_median": (900, 1250),
        "brain_mean": 1090,
        "brain_sd": 200,
        "qc_min": 700,
        "qc_max_strict": 1500,
        "qc_max_lenient": 1700,
        "csf": 4000,
        "default": 1200,
        "t1_upper_bound": 6000,
        "t1_parenchyma_upper": 2500,
        "reject_min": 600,
        "reject_max": 1800,
        "warn_low": 850,
        "warn_high": 1290,
        "expected_median": 1090,
    },
    # 7.0T: Rooney 2007 (WM/GM T1), Wright 2008, Marques 2010
    7.0: {
        "wm": (1000, 1250),
        "gm": (1800, 2100),
        "brain_median": (1300, 1700),
        "brain_mean": 1455,
        "brain_sd": 250,
        "qc_min": 1000,
        "qc_max_strict": 2100,
        "qc_max_lenient": 2350,
        "csf": 4500,
        "default": 1700,
        "t1_upper_bound": 7000,
        "t1_parenchyma_upper": 3500,
        "reject_min": 800,
        "reject_max": 2400,
        "warn_low": 1200,
        "warn_high": 1705,
        "expected_median": 1455,
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
    nan_fraction: Optional[float] = None
    tissue_t1_used: bool = False
    vfa_fit_nrmse: Optional[float] = None

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
        if self.nan_fraction is not None:
            d["nan_fraction"] = self.nan_fraction
        d["tissue_t1_used"] = self.tissue_t1_used
        if self.vfa_fit_nrmse is not None:
            d["vfa_fit_nrmse"] = self.vfa_fit_nrmse
        return d


# ============================================================================
# Helper functions
# ============================================================================


def get_field_strength_key(field_strength: float) -> float:
    """Round field strength to nearest standard value (1.5, 3.0, or 7.0).

    Parameters
    ----------
    field_strength : float
        Scanner field strength in Tesla.

    Returns
    -------
    float
        Nearest standard field strength (1.5, 3.0, or 7.0).

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
    if field_strength < 2.25:
        return 1.5
    elif field_strength < 5.0:
        return 3.0
    else:
        return 7.0


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
# VFA quality and tissue-segmented T1 helpers
# ============================================================================


def get_nan_abandon_threshold(e1_median: float) -> float:
    """Compute NaN fraction above which VFA T1 should be abandoned.

    For well-conditioned VFA (E1 < 0.99), use the standard threshold (0.40).
    For ill-conditioned VFA (E1 >= 0.99), linearly decrease the threshold
    down to 0.20 at E1 = 0.999+, because even surviving voxels are unreliable.

    Parameters
    ----------
    e1_median : float
        Median E1 = exp(-TR/T1) value from VFA data.

    Returns
    -------
    float
        NaN fraction threshold above which VFA should be abandoned.
    """
    if e1_median < 0.99:
        return VFA_NAN_FRACTION_TISSUE_FALLBACK
    # Linear ramp: 0.40 at E1=0.99 → 0.20 at E1=0.999
    return max(
        VFA_NAN_FRACTION_GOOD,
        VFA_NAN_FRACTION_TISSUE_FALLBACK - (e1_median - 0.99) * 20.0,
    )


def build_tissue_t1_map(
    wm_probseg: npt.NDArray[np.floating],
    gm_probseg: npt.NDArray[np.floating],
    field_strength: float,
    smooth_fwhm_vox: float = 1.5,
) -> npt.NDArray[np.floating]:
    """Build a tissue-segmented T1 map from WM/GM probability maps.

    Uses literature T1 values weighted by tissue probability. Regions with
    low total probability (neither WM nor GM) receive the field-strength
    expected median. The map is smoothed with a Gaussian kernel to reduce
    sharp tissue boundaries.

    Parameters
    ----------
    wm_probseg : ndarray
        White matter probability map (0-1).
    gm_probseg : ndarray
        Gray matter probability map (0-1).
    field_strength : float
        Scanner field strength in Tesla.
    smooth_fwhm_vox : float
        FWHM of Gaussian smoothing kernel in voxels (default: 1.5).
        ``gaussian_filter`` operates in voxel units; the default of 1.5
        gives ~2-3mm smoothing for typical DCE voxel sizes (1.5-2mm).

    Returns
    -------
    ndarray
        Tissue-segmented T1 map in milliseconds.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

    field_key = get_field_strength_key(field_strength)
    t1_ref = T1_TISSUE_REFERENCE[field_key]

    # Midpoints from literature ranges (e.g., WM 800-1000 → 900ms at 3T)
    wm_range = t1_ref["wm"]
    gm_range = t1_ref["gm"]
    t1_wm = float(wm_range[0] + wm_range[1]) / 2.0
    t1_gm = float(gm_range[0] + gm_range[1]) / 2.0
    expected_median = t1_ref["expected_median"]

    # Probability-weighted T1
    p_total = wm_probseg + gm_probseg
    # Use expected_median where tissue probability is too low
    low_prob_mask = p_total < 0.1
    p_total_safe = np.where(low_prob_mask, 1.0, p_total)
    t1_map = (wm_probseg * t1_wm + gm_probseg * t1_gm) / p_total_safe
    t1_map[low_prob_mask] = expected_median

    # Smooth AFTER filling defaults (no NaN propagation)
    if smooth_fwhm_vox > 0:
        # FWHM to sigma: sigma = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.355
        sigma = smooth_fwhm_vox / 2.3548
        t1_map = gaussian_filter(t1_map, sigma=sigma)

    return np.asarray(t1_map, dtype=np.float32)


# ============================================================================
# Cross-validation
# ============================================================================


def cross_validate_field_strength(
    median_t1: float,
    specified_field_strength: float,
) -> FieldStrengthValidation:
    """Cross-validate field strength by checking if measured T1 matches expectations.

    T1 relaxation times are field-strength-dependent:
    - At 1.5T: WM ~600-700ms, GM ~900-1200ms, brain median ~700-950ms
    - At 3.0T: WM ~800-1000ms, GM ~1250-1500ms, brain median ~900-1250ms
    - At 7.0T: WM ~1000-1250ms, GM ~1800-2100ms, brain median ~1300-1700ms

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

    # Find the alternative field strength closest to measured median
    other_keys = [k for k in T1_TISSUE_REFERENCE if k != field_key]
    distances = {
        k: _distance_from_range(median_t1, T1_TISSUE_REFERENCE[k]["brain_median"])
        for k in other_keys
    }
    opposite_key = min(distances, key=lambda k: distances[k])

    expected_specified = T1_TISSUE_REFERENCE[field_key]["brain_median"]
    expected_opposite = T1_TISSUE_REFERENCE[opposite_key]["brain_median"]

    dist_specified = _distance_from_range(median_t1, expected_specified)
    dist_opposite = distances[opposite_key]

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
    nan_fraction: Optional[float] = None,
    wm_probseg: Optional[npt.NDArray[np.floating]] = None,
    gm_probseg: Optional[npt.NDArray[np.floating]] = None,
    e1_median: Optional[float] = None,
    b1_corrected: bool = False,
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
    nan_fraction : float, optional
        Fraction of brain voxels with NaN T1 (0-1). Used by auto-mode
        to decide between offset correction and tissue-segmented fallback.
    wm_probseg : ndarray, optional
        White matter probability map (resampled to VFA space).
        Used for tissue-segmented T1 fallback.
    gm_probseg : ndarray, optional
        Gray matter probability map (resampled to VFA space).
        Used for tissue-segmented T1 fallback.
    e1_median : float, optional
        Median E1 = exp(-TR/T1) from VFA data. Used by auto-mode to
        adjust the NaN abandon threshold for ill-conditioned VFA.
    b1_corrected : bool
        Whether B1 field mapping was applied. At severe VFA conditioning
        (unconditional zone, E1>0.997), absence of B1 correction triggers
        fallback. In the conditional zone (E1 0.993-0.997), VFA T1 is
        preserved for offset correction if NaN fraction is manageable.

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
        fill = (brain_mask > 0.5) if brain_mask is not None else valid_mask
        t1_corrected = np.where(fill, t1_ref["default"], np.nan)
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
            nan_fraction=nan_fraction,
            wm_probseg=wm_probseg,
            gm_probseg=gm_probseg,
            field_strength=field_strength,
            e1_median=e1_median,
            b1_corrected=b1_corrected,
            brain_mask=brain_mask,
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
    nan_fraction: Optional[float] = None,
    wm_probseg: Optional[npt.NDArray[np.floating]] = None,
    gm_probseg: Optional[npt.NDArray[np.floating]] = None,
    field_strength: float = 3.0,
    e1_median: Optional[float] = None,
    b1_corrected: bool = False,
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[npt.NDArray[np.floating], T1CorrectionResult]:
    """Auto-mode T1 correction logic.

    Decides between no correction, offset, tissue-segmented fallback,
    or uniform fallback based on median T1 deviation and VFA conditioning.

    Decision hierarchy:

    1. **Reject threshold** (median T1 outside physiological range):
       When median T1 < reject_min or > reject_max, VFA T1 is definitively
       wrong regardless of NaN fraction. Tissue-segmented T1 is used if
       WM/GM probsegs are available; otherwise uniform fallback is applied.

    2. **Conditioning gate** (VFA E1 approaching 1):
       - *Unconditional zone* (E1 > 0.997, cond > 332): VFA signal surface
         is flat in the T1 direction; VFA T1 is abandoned for tissue-segmented
         or uniform fallback regardless of B1 status.
       - *Conditional zone* (E1 0.993-0.997, cond 141-332): VFA T1 is biased
         but retains spatial information. Without B1 correction, offset
         correction is preferred over fallback when NaN fraction is manageable,
         as it preserves spatial variation while fixing the median bias
         (Deoni SCL et al. MRM 2003). When NaN fraction exceeds the E1-adaptive
         abandon threshold, fallback is used instead.

    3. **Offset correction**: Applied when median T1 deviates from expected
       but is within acceptable range.
    """
    expected_median = t1_ref["expected_median"]
    result.nan_fraction = nan_fraction

    # Fallback fill mask: entire brain, not just valid VFA voxels.
    # valid_mask excludes NaN/non-positive VFA voxels, but fallbacks
    # should cover all brain tissue for downstream PK fitting.
    fill_mask = (brain_mask > 0.5) if brain_mask is not None else valid_mask

    # Compute E1-dependent NaN abandon threshold
    nan_abandon = get_nan_abandon_threshold(e1_median) if e1_median is not None else VFA_NAN_FRACTION_TISSUE_FALLBACK

    # Conditioning gate: VFA T1 is unreliable when E1 approaches 1
    if e1_median is not None and e1_median > _E1_CONDITIONAL_FALLBACK:
        unconditional = e1_median > _E1_UNCONDITIONAL_FALLBACK
        b1_insufficient = not b1_corrected or not b1_qc_pass
        cond_number = e1_median / (1.0 - e1_median) if e1_median < 1.0 else float("inf")

        if unconditional:
            # Unconditional zone (E1>0.997, cond>332): signal surface is
            # flat in T1 direction, VFA T1 is genuinely useless.
            logger.warning(
                "VFA conditioning gate (unconditional): E1=%.4f (cond=%.0f) "
                "— falling back to tissue-segmented T1",
                e1_median, cond_number,
            )
            if wm_probseg is not None and gm_probseg is not None:
                t1_corrected = build_tissue_t1_map(wm_probseg, gm_probseg, field_strength)
                t1_corrected = np.where(fill_mask, t1_corrected, np.nan)
                result.correction_applied = "tissue_segmented"
                result.tissue_t1_used = True
                result.reason = f"VFA conditioning gate: E1={e1_median:.4f} above unconditional threshold"
                result.median_t1_corrected = expected_median
            else:
                t1_corrected = np.where(fill_mask, t1_ref["default"], np.nan)
                result.correction_applied = "fallback"
                result.reason = f"VFA conditioning gate (no tissue maps): E1={e1_median:.4f} above unconditional threshold"
                result.fallback_value_ms = t1_ref["default"]
                result.median_t1_corrected = t1_ref["default"]
            return t1_corrected, result

        elif b1_insufficient:
            # Conditional zone (E1 0.993-0.997, cond 141-332): VFA T1 is
            # biased but retains spatial information. Offset correction
            # preserves spatial variation while fixing median bias.
            # Safety net: if NaN fraction exceeds E1-adaptive threshold,
            # too many voxels failed for offset to be meaningful.
            if nan_fraction is not None and nan_fraction > nan_abandon:
                logger.warning(
                    "VFA conditioning gate: E1=%.4f (cond=%.0f), "
                    "NaN fraction %.1f%% > abandon threshold %.1f%% "
                    "— falling back to tissue-segmented T1",
                    e1_median, cond_number,
                    nan_fraction * 100, nan_abandon * 100,
                )
                if wm_probseg is not None and gm_probseg is not None:
                    t1_corrected = build_tissue_t1_map(wm_probseg, gm_probseg, field_strength)
                    t1_corrected = np.where(fill_mask, t1_corrected, np.nan)
                    result.correction_applied = "tissue_segmented"
                    result.tissue_t1_used = True
                    result.reason = (
                        f"VFA conditioning gate: B1 absent, "
                        f"NaN {nan_fraction:.1%} > {nan_abandon:.1%} threshold"
                    )
                    result.median_t1_corrected = expected_median
                else:
                    t1_corrected = np.where(fill_mask, t1_ref["default"], np.nan)
                    result.correction_applied = "fallback"
                    result.reason = (
                        f"VFA conditioning gate (no tissue maps): B1 absent, "
                        f"NaN {nan_fraction:.1%} > {nan_abandon:.1%} threshold"
                    )
                    result.fallback_value_ms = t1_ref["default"]
                    result.median_t1_corrected = t1_ref["default"]
                return t1_corrected, result

            # Manageable NaN fraction: fall through to Cases 1/2/3
            # for offset correction, preserving VFA spatial variation.
            logger.info(
                "VFA conditioning gate: E1=%.4f (cond=%.0f), B1 absent "
                "but NaN fraction %.1f%% manageable — falling through "
                "to offset correction",
                e1_median, cond_number,
                (nan_fraction or 0) * 100,
            )

    # Case 1: Hard reject - impossible values
    # When median T1 exceeds reject threshold, VFA is fundamentally wrong.
    # Prefer tissue-segmented T1 (preserves WM/GM contrast) over uniform fallback.
    if median_t1 < t1_ref["reject_min"] or median_t1 > t1_ref["reject_max"]:
        direction = "below" if median_t1 < t1_ref["reject_min"] else "above"
        threshold = t1_ref["reject_min"] if direction == "below" else t1_ref["reject_max"]

        if wm_probseg is not None and gm_probseg is not None:
            logger.warning(
                "Median T1 (%.0fms) %s reject threshold (%.0fms) - "
                "using tissue-segmented T1 map",
                median_t1, direction, threshold,
            )
            t1_corrected = build_tissue_t1_map(
                wm_probseg, gm_probseg, field_strength,
            )
            # Preserve NaN outside brain (downstream code uses NaN for masking)
            t1_corrected = np.where(fill_mask, t1_corrected, np.nan)
            result.correction_applied = "tissue_segmented"
            result.tissue_t1_used = True
            result.reason = (
                f"Median T1 {direction} reject threshold ({threshold}ms) - "
                "tissue-segmented fallback"
            )
            result.median_t1_corrected = expected_median
            return t1_corrected, result

        logger.warning(
            "Median T1 (%.0fms) %s reject threshold (%.0fms) - "
            "falling back to literature value (%.0fms) (no tissue maps)",
            median_t1, direction, threshold, t1_ref["default"],
        )
        t1_corrected = np.where(fill_mask, t1_ref["default"], np.nan)
        result.correction_applied = "fallback"
        result.reason = (
            f"Median T1 {direction} reject threshold ({threshold}ms), "
            "no tissue maps available"
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


# ============================================================================
# Interior NaN gap-fill
# ============================================================================

# Gap-fill constants (in mm, converted to voxels via affine)
GAPFILL_IDW_RADIUS_MM: float = 4.5  # ~3 voxels at 1.5mm resolution
GAPFILL_MIN_NEIGHBOR_FRACTION: float = 0.30  # 30% of ball voxels must be valid
GAPFILL_MIN_NEIGHBOR_FLOOR: int = 6  # absolute floor for very coarse voxels
GAPFILL_BOUNDARY_FWHM_MM: float = 1.5  # smoothing at fill boundary


@dataclass
class T1GapFillResult:
    """Result of interior NaN gap-fill for T1 maps."""

    n_nan_before: int = 0
    n_filled_interp: int = 0
    n_filled_tissue: int = 0
    n_remaining: int = 0
    method: str = "none"  # "none", "interp_only"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "n_nan_before": self.n_nan_before,
            "n_filled_interp": self.n_filled_interp,
            "n_filled_tissue": self.n_filled_tissue,
            "n_remaining": self.n_remaining,
            "method": self.method,
        }


def fill_t1_gaps(
    t1_map: npt.NDArray[np.floating],
    brain_mask: npt.NDArray[np.bool_],
    field_strength: float,
    voxel_sizes_mm: Optional[npt.NDArray[np.floating]] = None,
) -> Tuple[npt.NDArray[np.floating], T1GapFillResult]:
    """Fill interior NaN gaps in T1 map within brain mask.

    Uses IDW (inverse-distance-weighted) interpolation for gaps with
    sufficient valid neighbors — preserves local T1 variation. Gaps
    without enough neighbors (large holes, necrotic cores) remain NaN
    so that downstream PK fitting produces NaN Ktrans rather than
    fabricated values.

    The minimum neighbor threshold scales with resolution: 30% of ball
    kernel voxels must be valid, with an absolute floor of 6.

    Boundary smoothing blends filled voxels with original neighbors to
    avoid sharp discontinuities.

    Parameters
    ----------
    t1_map : ndarray
        T1 map in milliseconds (float32). Not modified in place.
    brain_mask : ndarray of bool
        Brain mask (True = brain voxel).
    field_strength : float
        Scanner field strength in Tesla.
    voxel_sizes_mm : ndarray, optional
        Voxel dimensions in mm (3-element). If None, assumes isotropic 1.5mm.

    Returns
    -------
    t1_filled : ndarray
        T1 map with NaN gaps filled (float32).
    result : T1GapFillResult
        Metrics describing the gap-fill operation.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter, generate_binary_structure  # type: ignore[import-untyped]
    from scipy.signal import fftconvolve  # type: ignore[import-untyped]

    result = T1GapFillResult()
    gap_mask = np.isnan(t1_map) & brain_mask
    result.n_nan_before = int(np.sum(gap_mask))

    if result.n_nan_before == 0:
        return t1_map.copy().astype(np.float32), result

    t1_filled = t1_map.copy()

    # Convert radius from mm to voxels
    if voxel_sizes_mm is not None:
        vox = np.asarray(voxel_sizes_mm, dtype=np.float64)
    else:
        vox = np.array([1.5, 1.5, 1.5], dtype=np.float64)

    radius_vox = GAPFILL_IDW_RADIUS_MM / vox  # per-axis radius in voxels

    # Stage 1: IDW interpolation
    # Build ball kernel in voxel units (handles anisotropic voxels)
    half_size = np.ceil(radius_vox).astype(int)
    grid = np.mgrid[
        -half_size[0]:half_size[0] + 1,
        -half_size[1]:half_size[1] + 1,
        -half_size[2]:half_size[2] + 1,
    ]
    # Normalized distance (ellipsoidal for anisotropic voxels)
    dist = np.sqrt(
        (grid[0] * vox[0]) ** 2 + (grid[1] * vox[1]) ** 2 + (grid[2] * vox[2]) ** 2
    )
    ball = dist <= GAPFILL_IDW_RADIUS_MM
    ball_center = tuple(half_size)
    ball[ball_center] = False  # exclude center voxel

    # IDW weights: 1/distance within ball, 0 outside
    kernel = np.zeros_like(dist, dtype=np.float64)
    kernel[ball] = 1.0 / np.maximum(dist[ball], 1e-6)

    # Count valid neighbors per gap voxel
    valid_map = np.isfinite(t1_filled) & brain_mask
    valid_float = valid_map.astype(np.float64)
    neighbor_count = fftconvolve(valid_float, np.ones_like(kernel) * ball, mode="same")
    neighbor_count = np.round(neighbor_count).astype(int)

    # IDW interpolation: weighted sum / weight total
    t1_for_conv = np.where(valid_map, t1_filled, 0.0).astype(np.float64)
    weighted_sum = fftconvolve(t1_for_conv, kernel, mode="same")
    weight_total = fftconvolve(valid_float, kernel, mode="same")

    # Dynamic threshold: 30% of ball voxels, floor of 6
    n_ball_voxels = int(np.sum(ball))
    min_neighbors = max(
        GAPFILL_MIN_NEIGHBOR_FLOOR,
        int(n_ball_voxels * GAPFILL_MIN_NEIGHBOR_FRACTION),
    )

    interp_mask = gap_mask & (neighbor_count >= min_neighbors)
    safe_weights = np.where(weight_total > 0, weight_total, 1.0)
    interp_values = weighted_sum / safe_weights

    t1_filled[interp_mask] = interp_values[interp_mask].astype(np.float32)
    result.n_filled_interp = int(np.sum(interp_mask))

    # Remaining gaps stay NaN — downstream PK handles NaN→NaN Ktrans
    # (no tissue fallback to avoid fabricating Ktrans in necrotic cores)

    # Boundary smoothing: Gaussian on 1-voxel dilated fill boundary only
    all_filled = interp_mask
    if np.any(all_filled):
        struct = generate_binary_structure(3, 1)
        boundary = binary_dilation(all_filled, structure=struct) & ~all_filled & valid_map
        smooth_region = all_filled | boundary

        if np.any(smooth_region):
            # FWHM to sigma in voxels (per-axis for anisotropic)
            sigma_vox = (GAPFILL_BOUNDARY_FWHM_MM / vox) / 2.3548
            # Smooth only the fill+boundary region
            t1_smooth_input = np.where(smooth_region, t1_filled, 0.0).astype(np.float64)
            weight_input = smooth_region.astype(np.float64)
            t1_smoothed = gaussian_filter(t1_smooth_input, sigma=sigma_vox)
            weight_smoothed = gaussian_filter(weight_input, sigma=sigma_vox)
            # Safe divide
            safe_w = np.where(weight_smoothed > 0, weight_smoothed, 1.0)
            t1_boundary_smoothed = t1_smoothed / safe_w
            # Only apply smoothing to filled voxels (not original valid voxels)
            t1_filled[all_filled] = t1_boundary_smoothed[all_filled].astype(np.float32)

    # Ensure voxels outside brain mask stay NaN
    t1_filled[~brain_mask] = np.nan

    result.n_remaining = int(np.sum(np.isnan(t1_filled) & brain_mask))

    # Determine method string
    if result.n_filled_interp > 0:
        result.method = "interp_only"

    logger.info(
        "T1 gap-fill: %d NaN voxels → %d interp, %d tissue, %d remaining",
        result.n_nan_before, result.n_filled_interp,
        result.n_filled_tissue, result.n_remaining,
    )

    return t1_filled.astype(np.float32), result
