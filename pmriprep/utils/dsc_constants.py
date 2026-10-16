"""DSC perfusion physics constants.

Centralized constants for DSC-MRI processing including signal conversion,
deconvolution, and physiological parameter bounds.
"""

from typing import Final, Literal, NamedTuple, get_args

# =============================================================================
# AIF Detection Cascade Levels
# =============================================================================
# Canonical strings for the DSC AIF detection cascade. Referenced by the
# sidecar writer, HTML report, and QC code to prevent string drift.

DSCDetectionLevel = Literal[
    "atlas_strict",
    "atlas_relaxed",
    "atlas_permissive",
    "full_atlas_permissive",
    "brain_relaxed",
    "brain_permissive",
    "single_best",
    "synthetic",
    "external",
]

#: Runtime tuple of valid detection-level strings.
DSC_DETECTION_LEVELS: Final[tuple[str, ...]] = get_args(DSCDetectionLevel)


def split_detection_level(detection_level: str) -> list[str]:
    """Split a possibly compound ``+``-joined detection_level into parts.

    Parameters
    ----------
    detection_level : str
        Single tier name or ``+``-joined compound (e.g.
        ``"atlas_relaxed+atlas_permissive"``).  Non-string or empty
        inputs return ``[]``.

    Returns
    -------
    list[str]
        Individual tier name strings.
    """
    if not isinstance(detection_level, str) or not detection_level:
        return []
    return detection_level.split("+")


def is_valid_detection_level(detection_level: str) -> bool:
    """Check whether every part of a detection_level is canonical.

    Parameters
    ----------
    detection_level : str
        Single or ``+``-joined compound detection level string.

    Returns
    -------
    bool
        ``True`` if non-empty and every part is in
        :data:`DSC_DETECTION_LEVELS`.
    """
    parts = split_detection_level(detection_level)
    return bool(parts) and all(p in DSC_DETECTION_LEVELS for p in parts)


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

#: Floor for gamma-variate shape parameter alpha when computing
#: beta = MTT / alpha. Prevents division-by-zero on degenerate fits;
#: physiological alpha is O(1-10) so 1e-10 is safely below the floor.
EPSILON_SHAPE: Final[float] = 1e-10


# =============================================================================
# Contrast agent dose reference
# =============================================================================

#: Reference Gd administered dose (mmol Gd/kg) against which other agents
#: and doses are scaled. Gd-DOTA class single-dose standard = 0.1 mmol/kg.
#: Preload ΔR2* thresholds, CNR coverage thresholds, and residual-
#: equivalent fraction calculations all normalize to this value.
GD_DOTA_STANDARD_DOSE: Final[float] = 0.1


# =============================================================================
# Synthetic AIF Calibration
# =============================================================================

#: Population-median arterial peak ΔR2* (s⁻¹) at 3T for standard single-dose
#: Gd dose (0.1 mmol Gd/kg).
SYNTH_AIF_BASE_PEAK_3T: Final[float] = 35.0

#: Upper cap on the effective-dose multiplier.
SYNTH_AIF_DOSE_CAP: Final[float] = 2.5

#: Minimum synthetic AIF peak amplitude (s⁻¹). At 3T with SNR≈20, TE≈30 ms,
#: per-voxel ΔR2* noise ≈ 1/(TE×SNR) ≈ 1.7 s⁻¹; 3.0 is ~2 sigma guard.
SYNTH_AIF_MIN_PEAK: Final[float] = 3.0

#: Synthetic-AIF recirculation (second-pass) lobe. A dispersed ΔR2* second pass
#: is small relative to the first pass, and flatter/broader than it.
#: Second-pass peak amplitude as a fraction of the first-pass peak.
SYNTH_AIF_RECIRC_FRACTION: Final[float] = 0.12
#: Recirculation gamma-variate shape parameter; flatter than the first pass.
SYNTH_AIF_RECIRC_ALPHA: Final[float] = 3.0
#: Seconds from the first-pass peak to the recirculation peak.
SYNTH_AIF_RECIRC_DELAY_S: Final[float] = 20.0
#: Recirculation gamma-variate time-to-peak (s); broader than the first pass.
SYNTH_AIF_RECIRC_DISPERSION_S: Final[float] = 8.0

#: Peak-ΔR2* candidate rejection bounds at 3T / full dose (s⁻¹). Both bounds
#: are B0·dose-scaled at the use site, tracking the same law as the synthetic AIF
#: peak. Floor separates artery (~35 s⁻¹ median) from tissue first-pass and is
#: clamped to ``SYNTH_AIF_MIN_PEAK``; ceiling rejects susceptibility artifact.
AIF_PEAK_FLOOR_3T: Final[float] = 5.0
AIF_PEAK_CEILING_3T: Final[float] = 120.0


# =============================================================================
# SVD Deconvolution Thresholds
# =============================================================================

#: Default singular value truncation threshold (fraction of max) for sSVD
DEFAULT_SVD_THRESHOLD: Final[float] = 0.2

#: Target oscillation index for oSVD. The search selects the first truncation
#: threshold meeting this target, or the upper threshold if none does.
DEFAULT_OI_THRESHOLD: Final[float] = 0.095

#: Number of evenly spaced truncation thresholds in the oSVD search.
DEFAULT_OSVD_GRID_STEPS: Final[int] = 50


# =============================================================================
# Tikhonov Regularization
# =============================================================================

class TikhonovRange(NamedTuple):
    """Range for Tikhonov regularization parameter lambda."""

    min: float
    max: float


#: Default lambda search range for Tikhonov regularization, expressed
#: as fractions of σ_max(A) (largest singular value of the AIF convolution
#: matrix). Absolute lambda values therefore scale with the AIF and sampling.
DEFAULT_LAMBDA_RANGE: Final[TikhonovRange] = TikhonovRange(1e-3, 10.0)

#: Number of log-spaced lambda values in the discrete L-curve sweep.
LCURVE_N_LAMBDAS: Final[int] = 100

#: Number of leading residue samples excluded from the first-difference
#: penalty in the modified Tikhonov operator. For these samples the side
#: constraint penalizes the residue magnitude (an identity block) instead
#: of the difference from the preceding sample.
DEFAULT_TIKHONOV_PENALTY_P: Final[int] = 2


# =============================================================================
# Perfusion Parameter Bounds
# =============================================================================

class ParameterBounds(NamedTuple):
    """Min/max bounds for a physiological parameter."""

    min: float
    max: float


#: CBF bounds in relative s⁻¹ (ΔR2* deconvolution units).
#: Field strength, TE, dose, and contrast agent all cancel in the
#: deconvolution (r2* scales both tissue and AIF equally).
CBF_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 2.0)

#: Vascular fitting-space MTT bounds in seconds; not an SVD/CPI readout clamp.
MTT_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.5, 60.0)


# =============================================================================
# Vascular (Gamma-Variate) Model Parameters
# =============================================================================

#: Gamma shape fitting support, including narrow transit-time distributions.
#: The ceiling is a finite model-domain limit, not a measured shape value.
VASCULAR_ALPHA_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.5, 100.0)

#: Marginal beta envelope in seconds, derived from beta = MTT / alpha.
#: Joint support is defined in MTT/alpha space, not an independent beta box.
VASCULAR_BETA_BOUNDS: Final[ParameterBounds] = ParameterBounds(
    MTT_BOUNDS.min / VASCULAR_ALPHA_BOUNDS.max,
    MTT_BOUNDS.max / VASCULAR_ALPHA_BOUNDS.min,
)

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

#: CNR cutoff used by the perfusion QC policy.
QC_CNR_THRESHOLD: Final[float] = 4.0

#: Mean-CNR cutoff for the gross-quality exclusion gate.
QC_MEAN_CNR_MINIMUM: Final[float] = 3.0

#: Time target used to choose the odd Savitzky-Golay AIF smoothing window.
AIF_SMOOTHING_TARGET_SECONDS: Final[float] = 7.5


# =============================================================================
# AIF Per-Voxel Kinetic Filters (post-score, pre-averaging)
# =============================================================================

#: Maximum peak-time offset from the pool's 20th percentile, in seconds.
#: Converted to frames with ``max(1, ceil(SECONDS / tr))``.
AIF_PEAK_SPREAD_MAX_SECONDS: Final[float] = 3.0

#: Candidate timing limits relative to tissue TTP, in seconds.
#: The first sets the late-peak rejection limit; the second sets the
#: near-tissue penalty band. Neither establishes vessel identity by itself.
AIF_TTP_REJECT_AFTER_TISSUE_S: Final[float] = 2.0
AIF_TTP_NEAR_TISSUE_S: Final[float] = 2.0

#: Maximum tail/peak fraction for an AIF voxel (mean ΔR2* over the last
#: ``AIF_TAIL_WINDOW_S`` divided by first-pass peak).
AIF_TAIL_FRACTION_MAX: Final[float] = 0.15

#: Late-tail measurement window in seconds, measured from acquisition end.
AIF_TAIL_WINDOW_S: Final[float] = 30.0

#: Minimum distance (mm) from a lesion for AIF candidates.
LESION_EXCLUSION_FLOOR_MM: Final[float] = 10.0

#: Recommended minimum distance (mm) from lesion for AIF candidate voxels.
LESION_EXCLUSION_DISTANCE_MM: Final[float] = 15.0


# =============================================================================
# Field-Strength Dependent Constants
# =============================================================================

#: In-vivo tissue transverse relaxivity r2* by field strength (T) in s^-1 per mM
#: tissue Gd, for gradient-echo DSC with Gd-DTPA in cortical tissue.
#:
#: Caveats:
#:   - Linear B0 scaling holds 1.5T-3T (static-dephasing regime). The 7T
#:     value is extrapolated; r2* at 7T depends non-linearly on vessel
#:     architecture (diffusion narrowing, non-Lorentzian lineshape).
#:   - Cortical-GM number; WM is ~10-20% lower.
#:   - Macrocyclic agents within ~10% of Gd-DTPA at standard dose.
#:   - Used only for residual-fraction QC (residual ΔR2* → equivalent [Gd]);
#:     does NOT enter deconvolution math (r2* cancels in CBF/CBV ratios).
R2_STAR_TISSUE_LINEAR_BY_FIELD: Final[dict[float, float]] = {
    1.5: 87.0,   # GE-EPI cortical tissue
    3.0: 174.0,  # Linear B0 scaling
    7.0: 400.0,  # Extrapolated; non-linearity at 7T (see caveat above)
}

#: Alias of the tissue-relaxivity lookup.
R2_STAR_BY_FIELD = R2_STAR_TISSUE_LINEAR_BY_FIELD


def get_r2_star_tissue(field_strength: float) -> float:
    """Get in-vivo tissue transverse relaxivity r2* for a given field strength.

    Uses linear interpolation across :data:`R2_STAR_TISSUE_LINEAR_BY_FIELD`.
    Values assume Gd-DTPA and cortical tissue; the 7T value is extrapolated.

    Parameters
    ----------
    field_strength : float
        Magnetic field strength in Tesla.

    Returns
    -------
    float
        Tissue r2* relaxivity in s^-1 per mM tissue Gd.
    """
    # Direct lookup for known field strengths
    if field_strength in R2_STAR_TISSUE_LINEAR_BY_FIELD:
        return R2_STAR_TISSUE_LINEAR_BY_FIELD[field_strength]

    # Linear interpolation for intermediate field strengths
    fields = sorted(R2_STAR_TISSUE_LINEAR_BY_FIELD.keys())

    # Clamp to valid range
    if field_strength <= fields[0]:
        return R2_STAR_TISSUE_LINEAR_BY_FIELD[fields[0]]
    if field_strength >= fields[-1]:
        return R2_STAR_TISSUE_LINEAR_BY_FIELD[fields[-1]]

    # Find surrounding known field strengths
    for i, f in enumerate(fields[:-1]):
        if f <= field_strength < fields[i + 1]:
            f_low, f_high = f, fields[i + 1]
            r2_low = R2_STAR_TISSUE_LINEAR_BY_FIELD[f_low]
            r2_high = R2_STAR_TISSUE_LINEAR_BY_FIELD[f_high]

            # Linear interpolation
            fraction = (field_strength - f_low) / (f_high - f_low)
            return r2_low + fraction * (r2_high - r2_low)

    # Fallback (should not reach here)
    return R2_STAR_TISSUE_LINEAR_BY_FIELD[3.0]


def get_r2_star(field_strength: float) -> float:
    """Get in-vivo tissue r2* relaxivity.

    See get_r2_star_tissue() for details.

    Parameters
    ----------
    field_strength : float
        Magnetic field strength in Tesla.

    Returns
    -------
    float
        Tissue r2* relaxivity in s^-1 per mM.
    """
    return get_r2_star_tissue(field_strength)


# =============================================================================
# Absolute Quantification Calibration
# =============================================================================

#: Large-vessel hematocrit, matched to dce_constants.HEMATOCRIT_ARTERIAL
#: so the large-vessel Hct is consistent
#: across the DSC and DCE modules.
HEMATOCRIT_LV: Final[float] = 0.42

#: Brain tissue density (g/mL)
RHO_BRAIN: Final[float] = 1.04

#: Blood density (g/mL)
RHO_BLOOD: Final[float] = 1.05

#: GRE vessel calibration factor at 3T.
VESSEL_CALIBRATION_GRE_3T: Final[float] = 0.14


def calculate_k_h(
    hematocrit: float = HEMATOCRIT_LV,
    rho_brain: float = RHO_BRAIN,
    rho_blood: float = RHO_BLOOD,
) -> float:
    """K_H calibration factor for absolute quantification.

    K_H = (rho_brain / rho_blood) * (1 / (1 - hematocrit)) * 100

    The (1 - hematocrit) term accounts for Gd distributing in plasma
    (fraction = 1 - Hct), not whole blood.

    Parameters
    ----------
    hematocrit : float, optional
        Large vessel hematocrit fraction.
    rho_brain : float, optional
        Brain tissue density in g/mL.
    rho_blood : float, optional
        Blood density in g/mL.

    Returns
    -------
    float
        K_H calibration factor (dimensionless scaling).
    """
    return (rho_brain / rho_blood) * (1.0 / (1.0 - hematocrit)) * 100.0
