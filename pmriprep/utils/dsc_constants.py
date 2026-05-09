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
    "accumulated",
    "single_best",
    "synthetic",
    "external",
]

#: Runtime tuple of valid detection-level strings.
DSC_DETECTION_LEVELS: Final[tuple[str, ...]] = get_args(DSCDetectionLevel)

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
# Leakage Correction (K2)
# =============================================================================

#: Upper bound for K2 leakage rate (s⁻¹). Literature maximum is ~0.04 s⁻¹;
#: values above 0.5 are numerically degenerate OLS fits. Existing QC already
#: treats K2 > 0.5 as "severe leakage."
K2_UPPER_BOUND: Final[float] = 0.5

# Bidirectional K2 (Leu et al. AJNR 2016, PMID 27079371)
# kep = EES efflux rate in s^-1 (NOT min^-1 like DCE literature).
# Using s^-1 for consistency with K2 units and Leu 2016 convention.
# To convert from DCE (min^-1): kep_s = kep_min / 60.0
#
# Leu 2016 measured GBM values:
#   DCE kep: 0.0050 ± 0.0018 s^-1 (= 0.30 ± 0.11 min^-1)
#   DSC kep: 0.0057 ± 0.0042 s^-1 (= 0.34 ± 0.25 min^-1)
# Figure 3D scatter: DCE 0-0.03, DSC 0-0.05 s^-1

#: Default kep for bidirectional K2 correction (s⁻¹), GBM mean from Leu 2016
KEP_DSC_DEFAULT: Final[float] = 0.005

#: Lower bound for kep (s⁻¹), below this ≈ standard BSW
KEP_DSC_LOWER: Final[float] = 0.001

#: Upper bound for kep (s⁻¹), Leu Fig 3D upper range
KEP_DSC_UPPER: Final[float] = 0.05


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
#: monomeric Gd (0.1 mmol/kg). Vonken MRM 41:524 (1999) reports arterial peak
#: ΔR2* of 30–50 s⁻¹ at 3T; Willats PMB 51:5007 (2006) AIF characterisation.
#: 35.0 is the population median. For tetrameric agents at standard molecular
#: dose, linear gd_multiplicity scaling may overestimate by up to 30% due to
#: partial saturation of the static dephasing regime above ~15 mM blood [Gd].
SYNTH_AIF_BASE_PEAK_3T: Final[float] = 35.0

#: Upper cap on effective_dose multiplier. Kjølby MRM 56:187 (2006): ΔR2*
#: linearity holds to ~15 mM blood [Gd]; 2.5× reference dose stays within.
SYNTH_AIF_DOSE_CAP: Final[float] = 2.5

#: Minimum synthetic AIF peak amplitude (s⁻¹). At 3T with SNR≈20, TE≈30 ms,
#: per-voxel ΔR2* noise ≈ 1/(TE×SNR) ≈ 1.7 s⁻¹; 3.0 is ~2 sigma guard.
SYNTH_AIF_MIN_PEAK: Final[float] = 3.0


# =============================================================================
# K2 residual thresholds (two tiers — different decision commitment levels)
# =============================================================================

#: Sigma-selection tier (coarser, tolerates looseness). Used by
#: _classify_preload_k2 in dsc_cbv_constraint to tune the SOFT prior
#: sigma based on leakage residuals; the decisions here only modulate
#: uncertainty, never commit to a preload class label.
K2_SIGMA_SUPPRESSED: Final[float] = 0.01      # s⁻¹
LEAKY_SIGMA_LOW: Final[float] = 5.0           # percent
LEAKY_SIGMA_HIGH: Final[float] = 15.0         # percent
K2_SIGMA_CORRECTION: Final[float] = 0.005     # s⁻¹

#: Primary-classification tier (stricter, commits to a discrete class).
#: Used by classify_preload_adequacy_from_k2 when the auto path must
#: decide adequate/partial/inadequate absent user input. Relaxed from
#: literature-derived 0.003 to 0.005 per physics consult 2026-04-18:
#: observed well-suppressed half-dose GBM cohort sits at k2_median
#: 0.0018–0.0048, so 0.005 captures it without reclassifying as partial.
#: References: Boxerman AJNR 2006, Leu JMRI 2016, Welker AJNR 2015.
K2_PRIMARY_ADEQUATE: Final[float] = 0.005     # s⁻¹
K2_PRIMARY_INADEQUATE: Final[float] = 0.01    # s⁻¹, == K2_SIGMA_SUPPRESSED
LEAKY_PRIMARY_ADEQUATE: Final[float] = 1.0    # percent
LEAKY_PRIMARY_INADEQUATE: Final[float] = 5.0  # percent, == LEAKY_SIGMA_LOW


# =============================================================================
# SVD Deconvolution Thresholds
# =============================================================================

#: Default singular value truncation threshold (fraction of max) for sSVD
DEFAULT_SVD_THRESHOLD: Final[float] = 0.2

#: Default oscillation index threshold for oSVD.
#: Adapted for 3T clinical SNR (~20-40); literature range 0.04-0.10
#: (Liu 2010 NMR Biomed 23:286; Wu 2003 used 0.095 at SNR~100).
#: OI(λ) is decreasing in λ, so tightening the OI target forces the
#: linspace search upward to a higher truncation threshold, suppressing
#: noise-driven MTT inflation. Loosening (e.g. to Wu's 0.095) lets the
#: algorithm pick smaller thresholds where residue-tail noise inflates
#: AUC and hence MTT. See Calamante 2013 NMR Biomed 26:792 §4.2 on the
#: intrinsic MTT noise sensitivity of oSVD at clinical SNR.
DEFAULT_OI_THRESHOLD: Final[float] = 0.065


# =============================================================================
# Tikhonov Regularization
# =============================================================================

class TikhonovRange(NamedTuple):
    """Range for Tikhonov regularization parameter lambda."""
    min: float
    max: float


#: Default lambda search range for Tikhonov regularization, expressed
#: as fractions of σ_max(A) (largest singular value of the AIF Toeplitz
#: matrix). The natural scale for first-order Tikhonov is the operator
#: norm; absolute λ values are AIF-dependent and not portable. The 4-decade
#: span (1e-3 to 10) covers the well-conditioned regime where the L-curve
#: actually has a corner — narrower ranges miss corners, wider ranges (e.g.
#: starting at 1e-6·σ_max) descend into the rank-deficient AtA noise floor
#: where the L-curve becomes non-monotone.
DEFAULT_LAMBDA_RANGE: Final[TikhonovRange] = TikhonovRange(1e-3, 10.0)

#: Number of log-spaced lambda values in the L-curve sweep. 50 is a
#: defensible compromise between corner-localization quality (Hansen
#: recommends ~30 minimum) and per-voxel runtime cost.
LCURVE_N_LAMBDAS: Final[int] = 50

#: Curvature threshold (log-log space) for L-curve corner detection.
#: Below this, the curve is treated as cornerless and the caller falls
#: back to ``λ = 0.1 · σ_max(A)`` (Hansen 1998 default). Empirically on
#: the gamma-variate DSC DRO: noiseless data peaks at curvature ≈1.0
#: (no real corner; the bend is the over-regularization elbow), while
#: any real noise (≥0.5%) gives peak curvature ≥10. A threshold of 2.0
#: cleanly separates the two regimes across all four manuscript tissues.
LCURVE_CURVATURE_FALLBACK: Final[float] = 2.0


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

#: Minimum CNR for reliable perfusion quantification (Rose criterion: CNR > 3-5)
QC_CNR_THRESHOLD: Final[float] = 4.0

#: Minimum mean CNR for excluding total failures (failed injection, extravasation)
#: Below this threshold, data is unusable regardless of coverage
QC_MEAN_CNR_MINIMUM: Final[float] = 3.0

#: Target temporal smoothing window for AIF re-extraction (seconds).
#: Bolus FWHM is typically 3-6 s; ~7.5 s gives single-pole filtering of the
#: peak without distorting it. Used by Savitzky-Golay on the re-extracted
#: AIF; the integer window length is computed at runtime as an odd number
#: of samples per ``max(5, int(round(target / dt)) | 1)``.
AIF_SMOOTHING_TARGET_SECONDS: Final[float] = 7.5


# =============================================================================
# AIF Per-Voxel Kinetic Filters (post-score, pre-averaging)
# =============================================================================

#: Maximum peak-time spread (seconds) within an averaged AIF voxel pool.
#: True intra-arterial dispersion across the MCA tree (M1 → M3) is sub-
#: second (Calamante NMR Biomed 2013; Bleeker MRM 2011), but the MEASURED
#: spread among co-arterial voxels at typical DSC TR (1-2 s) exceeds this
#: due to (a) frame quantization on argmax peak detection and (b) per-voxel
#: argmax jitter at SNR 20-40 (~±1 frame). Converted at runtime to a frame
#: count via ``max(1, ceil(SECONDS / tr))``. At TR=1.5 s → 2 frames; at
#: TR=2.0 s → 2 frames; at TR=0.5 s (high-temp DSC) → 6 frames.
AIF_PEAK_SPREAD_MAX_SECONDS: Final[float] = 3.0

#: Maximum tail/peak fraction for an AIF voxel (mean ΔR2* over the last
#: ``AIF_TAIL_WINDOW_S`` divided by first-pass peak). Catches BBB-leakage
#: and tissue-PVE-driven late plateau. Threshold of 0.15 (not the
#: physiologically tighter 0.10) is set per Bleeker MRM 2009 PVF analysis
#: at ~1.875 mm in-plane resolution: M1 lumen (~2.5 mm) fills <30% of a
#: voxel, so PVE-inflated tails of 10-15% are physically expected for
#: arterial voxels at this resolution.
AIF_TAIL_FRACTION_MAX: Final[float] = 0.15

#: Late-tail measurement window (seconds, from end of acquisition). At
#: typical bolus arrival ~30 s and recirculation decay <5% peak by t > 4×
#: MTT (Willats MRM 2006, Vonken MRM 1999), 30 s places the window well
#: past first-pass + recirculation in normal tissue, so a 15% tail
#: indicates extravasation/PVE rather than residual recirculation.
AIF_TAIL_WINDOW_S: Final[float] = 30.0


# =============================================================================
# Field-Strength Dependent Constants
# =============================================================================

#: In-vivo tissue transverse relaxivity r2* by field strength (T) in s^-1 per mM
#: tissue Gd, for gradient-echo DSC with Gd-DTPA in cortical tissue.
#:
#: Sources:
#:   Kjølby BF, Østergaard L, Kiselev VG. "Theoretical model of intravascular
#:     paramagnetic tracers effect on tissue relaxation." Magn Reson Med
#:     2006;56:187-197.
#:   Calamante F, Vonken EJPA, van Osch MJP. "Contrast agent concentration
#:     measurements affecting quantification of bolus-tracking perfusion MRI."
#:     Magn Reson Med 2007;58:544-553 (blood vs tissue r2* distinction).
#:
#: The earlier value 28.9 s^-1·T^-1·mM^-1 from van Osch 2003 is whole-blood
#: Gd-DTPA r2* (used for AIF quantification), not tissue r2*. Applying it
#: to tissue ΔR2* under-estimates inferred tissue [Gd] by ~2× because the
#: mesoscopic field perturbation around microvasculature amplifies tissue
#: ΔR2* per unit tissue Gd relative to per unit blood Gd. The values below
#: are tissue-calibrated.
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
    1.5: 87.0,   # Kjølby 2006, GE-EPI cortical tissue
    3.0: 174.0,  # Linear B0 scaling
    7.0: 400.0,  # Extrapolated; non-linearity at 7T (see caveat above)
}

#: Legacy alias for backwards compatibility
R2_STAR_BY_FIELD = R2_STAR_TISSUE_LINEAR_BY_FIELD


def get_r2_star_tissue(field_strength: float) -> float:
    """Get in-vivo tissue transverse relaxivity r2* for a given field strength.

    Uses linear interpolation across :data:`R2_STAR_TISSUE_LINEAR_BY_FIELD`,
    which holds Kjølby 2006 / Calamante 2007 tissue-calibrated values for
    Gd-DTPA in cortical tissue. See the constant docstring for caveats
    (Gd-DTPA assumed; cortical-GM numbers; 7T extrapolated).

    Args:
        field_strength: Magnetic field strength in Tesla.

    Returns:
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
    """Get in-vivo tissue r2* relaxivity (legacy alias).

    See get_r2_star_tissue() for details.

    Args:
        field_strength: Magnetic field strength in Tesla.

    Returns:
        Tissue r2* relaxivity in s^-1 per mM.
    """
    return get_r2_star_tissue(field_strength)


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

    K_H = (rho_brain / rho_blood) * (1 / (1 - hematocrit)) * 100

    The (1 - hematocrit) term accounts for Gd distributing in plasma
    (fraction = 1 - Hct), not whole blood.

    Args:
        hematocrit: Large vessel hematocrit fraction.
        rho_brain: Brain tissue density in g/mL.
        rho_blood: Blood density in g/mL.

    Returns:
        K_H calibration factor (dimensionless scaling).

    References:
        Østergaard L, et al. Magn Reson Med. 1996;36(5):715-725, Eq. 1.
        Rempp KA, et al. Radiology. 1994;193(3):637-641.
        Calamante F, et al. Magn Reson Med. 1999;41(1):95-103.
    """
    return (rho_brain / rho_blood) * (1.0 / (1.0 - hematocrit)) * 100.0
