"""QC standards and colorblind-safe visualization palette.

Standardized quality control thresholds and colorblind-safe color palette
for consistent QC assessment and visualization across the pipeline.

References:
    - Paul Tol's color schemes: https://personal.sron.nl/~pault/
    - Motion QC thresholds based on Power et al. (2012)
    - Physiological T1 values from Stanisz et al. (2005)
"""

from typing import Final, NamedTuple


# =============================================================================
# Paul Tol Colorblind-Safe Palette
# =============================================================================
# Based on Paul Tol's accessible color schemes for scientific visualization
# These colors are distinguishable by most people with color vision deficiencies


class ColorPalette(NamedTuple):
    """Colorblind-safe color palette for visualizations.

    Colors are hex values suitable for matplotlib and other plotting libraries.
    """
    primary: str        # Main/default color
    secondary: str      # Secondary/alternate color
    success: str        # Pass/good status
    warning: str        # Warning/marginal status
    error: str          # Fail/bad status


#: Paul Tol's bright qualitative scheme with semantic mapping
#: - Primary: blue (accessible, professional)
#: - Secondary: cyan (contrast with primary)
#: - Success: green (universal "good" association)
#: - Warning: yellow/orange (caution)
#: - Error: red (universal "problem" association)
COLORBLIND_PALETTE: Final[ColorPalette] = ColorPalette(
    primary="#4477AA",      # Tol bright blue
    secondary="#66CCEE",    # Tol bright cyan
    success="#228833",      # Tol bright green
    warning="#CCBB44",      # Tol bright yellow
    error="#EE6677",        # Tol bright red (muted for accessibility)
)


#: Extended palette for multi-class visualizations (up to 7 colors)
#: All colors from Paul Tol's bright qualitative scheme
EXTENDED_PALETTE: Final[tuple[str, ...]] = (
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Purple
    "#BBBBBB",  # Grey
)


#: Sequential colormap endpoints for heatmaps (blue to red via white)
#: Safe for most color vision deficiencies
SEQUENTIAL_COLORMAP: Final[tuple[str, str, str]] = (
    "#364B9A",  # Dark blue (low values)
    "#FFFFFF",  # White (middle)
    "#A50026",  # Dark red (high values)
)


#: Diverging colormap for centered data (e.g., correlation, difference maps)
DIVERGING_COLORMAP: Final[tuple[str, str, str]] = (
    "#4575B4",  # Blue (negative)
    "#FFFFBF",  # Light yellow (zero)
    "#D73027",  # Red (positive)
)


# =============================================================================
# Motion QC Thresholds
# =============================================================================
# Based on Power et al. (2012) framewise displacement thresholds
# FD = sum(|d_i - d_{i-1}|) for 6 motion parameters, with rotation scaled by 50mm


class QCThresholds(NamedTuple):
    """QC thresholds for pass/warn/fail classification."""
    pass_threshold: float   # Values below this: PASS
    warn_threshold: float   # Values between pass and warn: WARN; above warn: FAIL


#: Framewise displacement (FD) thresholds in mm
#: Pass: FD < 0.3mm (minimal motion)
#: Warn: 0.3mm <= FD < 0.5mm (moderate motion)
#: Fail: FD >= 0.5mm (excessive motion)
MOTION_FD_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.3,
    warn_threshold=0.5,
)

#: RMS displacement thresholds in mm
#: Slightly more lenient than FD as RMS is less sensitive to transients
MOTION_RMS_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.5,
    warn_threshold=1.0,
)

#: Maximum fraction of outlier frames (FD > 0.5mm) allowed
MOTION_OUTLIER_FRACTION_MAX: Final[float] = 0.20


# =============================================================================
# DICE Coefficient Thresholds
# =============================================================================
# For mask overlap QC (registration, segmentation)


#: DICE coefficient thresholds for mask comparison
#: Pass: DICE > 0.85 (excellent overlap)
#: Warn: 0.75 < DICE <= 0.85 (acceptable overlap)
#: Fail: DICE <= 0.75 (poor overlap)
DICE_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.85,
    warn_threshold=0.75,
)

#: Stricter DICE thresholds for critical masks (e.g., brain mask)
DICE_THRESHOLDS_STRICT: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.90,
    warn_threshold=0.85,
)


# =============================================================================
# Correlation Thresholds
# =============================================================================
# For image similarity assessment (registration QC, reproducibility)


#: Correlation coefficient thresholds for image comparison
#: Pass: r > 0.7 (strong correlation)
#: Warn: 0.5 < r <= 0.7 (moderate correlation)
#: Fail: r <= 0.5 (weak correlation)
CORRELATION_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.7,
    warn_threshold=0.5,
)

#: Normalized mutual information (NMI) thresholds for multi-modal registration
NMI_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.3,     # NMI typically lower than correlation
    warn_threshold=0.2,
)


# =============================================================================
# Physiological Ranges at 3T
# =============================================================================
# Expected T1 values for QC validation at 3.0 Tesla
# Based on Stanisz et al. (2005), Wright et al. (2008)


class T1Range(NamedTuple):
    """Expected T1 range in milliseconds."""
    min: float
    max: float


#: White matter T1 at 3T (ms)
WM_T1_RANGE_3T: Final[T1Range] = T1Range(800.0, 900.0)

#: Gray matter T1 at 3T (ms)
GM_T1_RANGE_3T: Final[T1Range] = T1Range(1200.0, 1400.0)

#: CSF T1 at 3T (ms)
CSF_T1_RANGE_3T: Final[T1Range] = T1Range(3500.0, 5000.0)


class PhysiologicalRanges3T(NamedTuple):
    """Physiological parameter ranges at 3T for QC validation."""
    wm_t1: T1Range      # White matter T1 in ms
    gm_t1: T1Range      # Gray matter T1 in ms
    csf_t1: T1Range     # CSF T1 in ms


#: Combined physiological ranges at 3T
PHYSIOLOGICAL_RANGES_3T: Final[PhysiologicalRanges3T] = PhysiologicalRanges3T(
    wm_t1=WM_T1_RANGE_3T,
    gm_t1=GM_T1_RANGE_3T,
    csf_t1=CSF_T1_RANGE_3T,
)


# =============================================================================
# QC Status Enumeration Helper
# =============================================================================


class QCStatus:
    """QC status strings for consistent reporting."""
    PASS: Final[str] = "PASS"
    WARN: Final[str] = "WARN"
    FAIL: Final[str] = "FAIL"
    UNKNOWN: Final[str] = "UNKNOWN"


def get_qc_status(value: float, thresholds: QCThresholds, higher_is_better: bool = True) -> str:
    """Determine QC status based on value and thresholds.

    Args:
        value: The metric value to evaluate
        thresholds: QCThresholds with pass/warn thresholds
        higher_is_better: If True, values above pass_threshold are PASS.
                         If False, values below pass_threshold are PASS.

    Returns:
        QC status string: "PASS", "WARN", or "FAIL"
    """
    if higher_is_better:
        # For metrics like DICE, correlation where higher is better
        if value > thresholds.pass_threshold:
            return QCStatus.PASS
        elif value > thresholds.warn_threshold:
            return QCStatus.WARN
        else:
            return QCStatus.FAIL
    else:
        # For metrics like motion FD where lower is better
        if value < thresholds.pass_threshold:
            return QCStatus.PASS
        elif value < thresholds.warn_threshold:
            return QCStatus.WARN
        else:
            return QCStatus.FAIL


def get_status_color(status: str) -> str:
    """Get the appropriate color for a QC status.

    Args:
        status: QC status string ("PASS", "WARN", "FAIL", or "UNKNOWN")

    Returns:
        Hex color code from the colorblind-safe palette
    """
    status_colors = {
        QCStatus.PASS: COLORBLIND_PALETTE.success,
        QCStatus.WARN: COLORBLIND_PALETTE.warning,
        QCStatus.FAIL: COLORBLIND_PALETTE.error,
        QCStatus.UNKNOWN: COLORBLIND_PALETTE.secondary,
    }
    return status_colors.get(status, COLORBLIND_PALETTE.secondary)
