"""DSC perfusion physics constants.

Centralized constants for DSC-MRI processing including signal conversion,
deconvolution, and physiological parameter bounds.
"""

from typing import Final, NamedTuple

# =============================================================================
# Epsilon Guards
# =============================================================================
# Small values to prevent numerical instabilities (divide by zero, log of zero)

#: Minimum signal value before log transformation
EPSILON_SIGNAL: Final[float] = 1e-9

#: Minimum delta R2* value for concentration conversion
EPSILON_DELTA_R2: Final[float] = 1e-10

#: Minimum concentration value
EPSILON_CONCENTRATION: Final[float] = 1e-10


# =============================================================================
# SVD Deconvolution Thresholds
# =============================================================================

#: Default singular value truncation threshold (fraction of max) for sSVD
DEFAULT_SVD_THRESHOLD: Final[float] = 0.2

#: Default oscillation index threshold for oSVD
DEFAULT_OI_THRESHOLD: Final[float] = 0.065


# =============================================================================
# Tikhonov Regularization
# =============================================================================

class TikhonovRange(NamedTuple):
    """Range for Tikhonov regularization parameter lambda."""
    min: float
    max: float


#: Default lambda search range for Tikhonov regularization with GCV
DEFAULT_LAMBDA_RANGE: Final[TikhonovRange] = TikhonovRange(1e-4, 1.0)


# =============================================================================
# Perfusion Parameter Bounds
# =============================================================================

class ParameterBounds(NamedTuple):
    """Min/max bounds for a physiological parameter."""
    min: float
    max: float


#: CBF bounds in mL/100g/min
CBF_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 200.0)

#: CBV bounds in mL/100g
CBV_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 20.0)

#: MTT bounds in seconds
MTT_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 30.0)


# =============================================================================
# Vascular (Gamma-Variate) Model Parameters
# =============================================================================

#: Alpha (shape) parameter bounds for gamma-variate model
VASCULAR_ALPHA_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.5, 10.0)

#: Beta (scale) parameter bounds in seconds for gamma-variate model
VASCULAR_BETA_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.1, 10.0)

#: Delay parameter bounds in seconds (can be negative for early arrival)
VASCULAR_DELAY_BOUNDS: Final[ParameterBounds] = ParameterBounds(-3.0, 10.0)


# =============================================================================
# Bayesian Inference Defaults
# =============================================================================

#: Default number of MCMC samples for posterior estimation
MCMC_SAMPLES: Final[int] = 500

#: Maximum iterations for MAP estimation with L-BFGS-B
MAP_MAX_ITER: Final[int] = 1000


# =============================================================================
# QC Thresholds
# =============================================================================

#: Maximum Gelman-Rubin R-hat for MCMC convergence
QC_RHAT_MAX: Final[float] = 1.1

#: Minimum effective sample size for MCMC convergence
QC_ESS_MIN: Final[int] = 100

#: Minimum CNR for reliable perfusion quantification
QC_CNR_THRESHOLD: Final[float] = 4.0


# =============================================================================
# Field-Strength Dependent Constants
# =============================================================================

#: r2* relaxivity by field strength (T) in s^-1 per mM
R2_STAR_BY_FIELD: Final[dict[float, float]] = {
    1.5: 5.5,
    3.0: 12.0,
    7.0: 30.0,
}


def get_r2_star(field_strength: float) -> float:
    """Get r2* relaxivity for a given field strength.

    Uses linear interpolation between known values.

    Args:
        field_strength: Magnetic field strength in Tesla

    Returns:
        r2* relaxivity in s^-1 per mM
    """
    # Direct lookup for known field strengths
    if field_strength in R2_STAR_BY_FIELD:
        return R2_STAR_BY_FIELD[field_strength]

    # Linear interpolation for intermediate field strengths
    fields = sorted(R2_STAR_BY_FIELD.keys())

    # Clamp to valid range
    if field_strength <= fields[0]:
        return R2_STAR_BY_FIELD[fields[0]]
    if field_strength >= fields[-1]:
        return R2_STAR_BY_FIELD[fields[-1]]

    # Find surrounding known field strengths
    for i, f in enumerate(fields[:-1]):
        if f <= field_strength < fields[i + 1]:
            f_low, f_high = f, fields[i + 1]
            r2_low = R2_STAR_BY_FIELD[f_low]
            r2_high = R2_STAR_BY_FIELD[f_high]

            # Linear interpolation
            fraction = (field_strength - f_low) / (f_high - f_low)
            return r2_low + fraction * (r2_high - r2_low)

    # Fallback (should not reach here)
    return R2_STAR_BY_FIELD[3.0]


# =============================================================================
# Absolute Quantification Calibration
# =============================================================================

#: Large vessel hematocrit (population average)
HEMATOCRIT_LV: Final[float] = 0.40

#: Brain tissue density (g/mL)
RHO_BRAIN: Final[float] = 1.04

#: Blood density (g/mL)
RHO_BLOOD: Final[float] = 1.05

#: GRE vessel calibration factor at 3T (Boxerman et al. 2006)
VESSEL_CALIBRATION_GRE_3T: Final[float] = 0.14


def calculate_k_h(
    hematocrit: float = HEMATOCRIT_LV,
    rho_brain: float = RHO_BRAIN,
    rho_blood: float = RHO_BLOOD,
) -> float:
    """K_H calibration factor for absolute quantification (Østergaard 1996).

    K_H = (rho_brain / rho_blood) * (1 / hematocrit) * 100

    Args:
        hematocrit: Large vessel hematocrit fraction.
        rho_brain: Brain tissue density in g/mL.
        rho_blood: Blood density in g/mL.

    Returns:
        K_H calibration factor (dimensionless scaling).
    """
    return (rho_brain / rho_blood) * (1.0 / hematocrit) * 100.0
