"""DCE perfusion physics constants.

Centralized constants for DCE-MRI processing including signal conversion,
pharmacokinetic modeling, T1 mapping, and AIF quality assessment.
"""

from typing import Final, NamedTuple


# =============================================================================
# Epsilon Guards
# =============================================================================
# Small values to prevent numerical instabilities (divide by zero, log of zero)

#: Minimum ve (extravascular extracellular volume fraction) value
EPSILON_VE: Final[float] = 0.001

#: Minimum T1 value in seconds
EPSILON_T1: Final[float] = 0.01

#: Minimum Ktrans value in min^-1
EPSILON_KTRANS: Final[float] = 0.001

#: Minimum vp (plasma volume fraction) value
EPSILON_VP: Final[float] = 0.001

#: Minimum B1+ correction factor
EPSILON_B1: Final[float] = 1e-6


# =============================================================================
# Field-Strength Dependent R1 Relaxivity
# =============================================================================

#: r1 relaxivity by field strength (T) in s^-1 per mM (Gd-DTPA, plasma)
#: Values from literature for typical gadolinium-based contrast agents
R1_RELAXIVITY_BY_FIELD: Final[dict[float, float]] = {
    1.5: 4.1,   # Gd-DTPA at 1.5T
    3.0: 3.7,   # Gd-DTPA at 3.0T (slight decrease at higher field)
    7.0: 3.5,   # Gd-DTPA at 7.0T
}


def get_r1_relaxivity(field_strength: float) -> float:
    """Get r1 relaxivity for a given field strength.

    Uses linear interpolation between known values.

    Args:
        field_strength: Magnetic field strength in Tesla

    Returns:
        r1 relaxivity in s^-1 per mM
    """
    # Direct lookup for known field strengths
    if field_strength in R1_RELAXIVITY_BY_FIELD:
        return R1_RELAXIVITY_BY_FIELD[field_strength]

    # Linear interpolation for intermediate field strengths
    fields = sorted(R1_RELAXIVITY_BY_FIELD.keys())

    # Clamp to valid range
    if field_strength <= fields[0]:
        return R1_RELAXIVITY_BY_FIELD[fields[0]]
    if field_strength >= fields[-1]:
        return R1_RELAXIVITY_BY_FIELD[fields[-1]]

    # Find surrounding known field strengths
    for i, f in enumerate(fields[:-1]):
        if f <= field_strength < fields[i + 1]:
            f_low, f_high = f, fields[i + 1]
            r1_low = R1_RELAXIVITY_BY_FIELD[f_low]
            r1_high = R1_RELAXIVITY_BY_FIELD[f_high]

            # Linear interpolation
            fraction = (field_strength - f_low) / (f_high - f_low)
            return r1_low + fraction * (r1_high - r1_low)

    # Fallback (should not reach here)
    return R1_RELAXIVITY_BY_FIELD[3.0]


# =============================================================================
# Blood T1 Values by Field Strength
# =============================================================================

#: Blood T1 values (seconds) by field strength at 37°C
#: Values from Lu et al. (2004), Stanisz et al. (2005)
BLOOD_T1_BY_FIELD: Final[dict[float, float]] = {
    1.5: 1.35,  # Blood T1 at 1.5T
    3.0: 1.65,  # Blood T1 at 3.0T
    7.0: 2.10,  # Blood T1 at 7.0T (extrapolated)
}


def get_blood_t1(field_strength: float) -> float:
    """Get blood T1 value for a given field strength.

    Uses linear interpolation between known values.

    Args:
        field_strength: Magnetic field strength in Tesla

    Returns:
        Blood T1 in seconds
    """
    # Direct lookup for known field strengths
    if field_strength in BLOOD_T1_BY_FIELD:
        return BLOOD_T1_BY_FIELD[field_strength]

    # Linear interpolation for intermediate field strengths
    fields = sorted(BLOOD_T1_BY_FIELD.keys())

    # Clamp to valid range
    if field_strength <= fields[0]:
        return BLOOD_T1_BY_FIELD[fields[0]]
    if field_strength >= fields[-1]:
        return BLOOD_T1_BY_FIELD[fields[-1]]

    # Find surrounding known field strengths
    for i, f in enumerate(fields[:-1]):
        if f <= field_strength < fields[i + 1]:
            f_low, f_high = f, fields[i + 1]
            t1_low = BLOOD_T1_BY_FIELD[f_low]
            t1_high = BLOOD_T1_BY_FIELD[f_high]

            # Linear interpolation
            fraction = (field_strength - f_low) / (f_high - f_low)
            return t1_low + fraction * (t1_high - t1_low)

    # Fallback (should not reach here)
    return BLOOD_T1_BY_FIELD[3.0]


# =============================================================================
# Tissue T1 QC Ranges by Field Strength
# =============================================================================

class T1Range(NamedTuple):
    """Expected T1 range for a tissue type in milliseconds."""
    min: float
    max: float


class TissueT1Ranges(NamedTuple):
    """Expected T1 ranges for different tissue types."""
    white_matter: T1Range
    gray_matter: T1Range
    csf: T1Range


#: Tissue T1 QC ranges by field strength (in milliseconds)
#: Values from Stanisz et al. (2005), Wright et al. (2008)
TISSUE_T1_RANGES_BY_FIELD: Final[dict[float, TissueT1Ranges]] = {
    1.5: TissueT1Ranges(
        white_matter=T1Range(550, 750),     # WM T1 at 1.5T: ~620ms
        gray_matter=T1Range(900, 1200),     # GM T1 at 1.5T: ~1000ms
        csf=T1Range(3000, 5000),            # CSF T1 at 1.5T: ~4000ms
    ),
    3.0: TissueT1Ranges(
        white_matter=T1Range(800, 1000),    # WM T1 at 3.0T: ~850ms
        gray_matter=T1Range(1200, 1600),    # GM T1 at 3.0T: ~1400ms
        csf=T1Range(3500, 5500),            # CSF T1 at 3.0T: ~4500ms
    ),
    7.0: TissueT1Ranges(
        white_matter=T1Range(1000, 1300),   # WM T1 at 7.0T: ~1100ms
        gray_matter=T1Range(1800, 2200),    # GM T1 at 7.0T: ~2000ms
        csf=T1Range(4000, 6000),            # CSF T1 at 7.0T: ~5000ms
    ),
}


def get_tissue_t1_ranges(field_strength: float) -> TissueT1Ranges:
    """Get expected tissue T1 ranges for QC validation.

    Returns the nearest field strength's ranges (no interpolation for QC).

    Args:
        field_strength: Magnetic field strength in Tesla

    Returns:
        TissueT1Ranges with WM, GM, and CSF expected ranges in ms
    """
    # Find nearest known field strength
    fields = sorted(TISSUE_T1_RANGES_BY_FIELD.keys())

    if field_strength <= fields[0]:
        return TISSUE_T1_RANGES_BY_FIELD[fields[0]]
    if field_strength >= fields[-1]:
        return TISSUE_T1_RANGES_BY_FIELD[fields[-1]]

    # Find nearest
    for i, f in enumerate(fields[:-1]):
        if f <= field_strength < fields[i + 1]:
            # Return nearest
            if (field_strength - f) < (fields[i + 1] - field_strength):
                return TISSUE_T1_RANGES_BY_FIELD[f]
            else:
                return TISSUE_T1_RANGES_BY_FIELD[fields[i + 1]]

    return TISSUE_T1_RANGES_BY_FIELD[3.0]


# =============================================================================
# Hematocrit Constant
# =============================================================================

#: Default hematocrit value (whole blood Gd correction)
#: Used for: C_plasma = C_blood / (1 - hematocrit)
HEMATOCRIT: Final[float] = 0.42


# =============================================================================
# AIF Quality Thresholds
# =============================================================================

class AIFThresholds(NamedTuple):
    """Thresholds for AIF quality assessment."""
    washout_min: float          # Minimum washout fraction
    fwhm_min_seconds: float     # Minimum FWHM in seconds
    fwhm_max_seconds: float     # Maximum FWHM in seconds
    baseline_frames: int        # Number of frames for baseline


#: Strict AIF quality thresholds (standard clinical quality)
AIF_THRESHOLDS_STRICT: Final[AIFThresholds] = AIFThresholds(
    washout_min=0.20,           # At least 20% washout from peak
    fwhm_min_seconds=8.0,       # FWHM at least 8 seconds
    fwhm_max_seconds=25.0,      # FWHM at most 25 seconds
    baseline_frames=5,          # 5 frames for stable baseline
)

#: Relaxed AIF quality thresholds (for difficult cases, e.g., GBM with preload)
AIF_THRESHOLDS_RELAXED: Final[AIFThresholds] = AIFThresholds(
    washout_min=0.10,           # At least 10% washout (reduced for preload)
    fwhm_min_seconds=5.0,       # FWHM at least 5 seconds
    fwhm_max_seconds=35.0,      # FWHM at most 35 seconds (wider for preload)
    baseline_frames=3,          # 3 frames for baseline (shorter for preload)
)


def get_aif_thresholds(criteria: str = "strict") -> AIFThresholds:
    """Get AIF quality thresholds based on criteria level.

    Args:
        criteria: Either "strict" or "relaxed"

    Returns:
        AIFThresholds with appropriate values

    Raises:
        ValueError: If criteria is not "strict" or "relaxed"
    """
    if criteria == "strict":
        return AIF_THRESHOLDS_STRICT
    elif criteria == "relaxed":
        return AIF_THRESHOLDS_RELAXED
    else:
        raise ValueError(f"Unknown AIF criteria: {criteria}. Use 'strict' or 'relaxed'.")


# =============================================================================
# DCE AIF Expected Values (Full Dose)
# =============================================================================

class AIFExpectedValues(NamedTuple):
    """Expected AIF peak concentration ranges for dose validation."""
    peak_min_mm: float      # Minimum expected peak in mM
    peak_max_mm: float      # Maximum expected peak in mM
    auc_min: float          # Minimum expected AUC
    auc_max: float          # Maximum expected AUC


#: Expected AIF values for full-dose contrast (0.1 mmol/kg)
#: Based on Parker et al. (2006) population AIF parameters
DCE_AIF_EXPECTED_FULL_DOSE: Final[AIFExpectedValues] = AIFExpectedValues(
    peak_min_mm=5.0,        # Minimum expected peak concentration
    peak_max_mm=7.6,        # Maximum expected peak concentration
    auc_min=1.0,            # Minimum AUC (normalized)
    auc_max=3.0,            # Maximum AUC (normalized)
)


# =============================================================================
# Pharmacokinetic Parameter Bounds
# =============================================================================

class ParameterBounds(NamedTuple):
    """Min/max bounds for a physiological parameter."""
    min: float
    max: float


#: Ktrans bounds in min^-1
KTRANS_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 3.0)

#: ve (extravascular extracellular volume) bounds (fraction)
VE_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.01, 1.0)

#: vp (plasma volume fraction) bounds
VP_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.001, 0.3)

#: kep (efflux rate) bounds in min^-1
KEP_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 10.0)

#: Fp (plasma flow) bounds in mL/min/100mL for 2CXM
FP_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 200.0)

#: PS (permeability-surface area product) bounds in mL/min/100mL for 2CXM
PS_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 100.0)


# =============================================================================
# B1+ Mapping QC Thresholds
# =============================================================================

#: Expected mean B1+ correction factor (ideally 1.0)
B1_EXPECTED_MEAN: Final[float] = 1.0

#: B1+ mean acceptable range (low, high)
B1_MEAN_RANGE: Final[tuple[float, float]] = (0.85, 1.15)

#: Maximum acceptable B1+ spatial standard deviation
B1_MAX_STD: Final[float] = 0.3


# =============================================================================
# T1 Mapping Defaults
# =============================================================================

#: Default T1 value (ms) when T1 mapping is skipped
DEFAULT_T1_MS: Final[float] = 1200.0

#: Lenient T1 QC range (ms) for clinical variability
LENIENT_T1_RANGE: Final[tuple[float, float]] = (500.0, 2500.0)


# =============================================================================
# DCE Processing Defaults
# =============================================================================

#: Default number of baseline frames for pre-contrast calculation
DEFAULT_BASELINE_FRAMES: Final[int] = 5

#: Default tSNR threshold for voxel inclusion
DEFAULT_TSNR_THRESHOLD: Final[float] = 3.0

#: Tissue density in g/mL (for unit conversions)
TISSUE_DENSITY: Final[float] = 1.04
