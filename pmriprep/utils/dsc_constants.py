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
# Leakage Correction (K2)
# =============================================================================

#: Numerical safety bound for K2 leakage fits (s⁻¹), not a physiological
#: maximum. Larger unconstrained OLS estimates are treated as degenerate.
K2_UPPER_BOUND: Final[float] = 0.2

# Bidirectional K2
# kep = EES efflux rate in s^-1 (NOT min^-1 like DCE literature).
# Using s^-1 for consistency with K2 units.
# To convert from DCE (min^-1): kep_s = kep_min / 60.0
#
# Measured GBM values:
#   DCE kep: 0.0050 ± 0.0018 s^-1 (= 0.30 ± 0.11 min^-1)
#   DSC kep: 0.0057 ± 0.0042 s^-1 (= 0.34 ± 0.25 min^-1)
# Figure 3D scatter: DCE 0-0.03, DSC 0-0.05 s^-1

#: Default kep for bidirectional K2 correction (s⁻¹), GBM mean
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
#: monomeric Gd (0.1 mmol/kg). For tetrameric agents at standard molecular
#: dose, linear gd_multiplicity scaling may overestimate by up to 30% due to
#: partial saturation of the static dephasing regime above ~15 mM blood [Gd].
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
#: literature-derived 0.003 to 0.005: observed well-suppressed half-dose
#: GBM cohort sits at k2_median
#: 0.0018–0.0048, so 0.005 captures it without reclassifying as partial.
K2_PRIMARY_ADEQUATE: Final[float] = 0.005     # s⁻¹
K2_PRIMARY_INADEQUATE: Final[float] = 0.01    # s⁻¹, == K2_SIGMA_SUPPRESSED
LEAKY_PRIMARY_ADEQUATE: Final[float] = 1.0    # percent
LEAKY_PRIMARY_INADEQUATE: Final[float] = 5.0  # percent, == LEAKY_SIGMA_LOW

#: Positive-tail (BSW-blind overestimate) detector. With preload, residual
#: extravasation flips to a T2*-dominant persistent POSITIVE ΔR2* tail that
#: the non-negative BSW correction cannot remove (it would need K2 < 0), so
#: K2 pins at 0 and the full-integral CBV over-estimates. A voxel is flagged
#: when its late-window tail persists AND its K2 is zero:
#:
#:   late_tail_ratio = mean(conc[final TAIL_LATE_FRAC of frames]) / peak(conc)
#:
#: The ratio is TE-invariant (the −1/TE factor cancels), so it is scale- and
#: protocol-portable. Thresholds calibrated on the Semmineh GBM DSC DRO
#: (fire on the +40% preloaded overestimate configs, silent on the T1-dip
#: underestimate configs) and validated silent on 16 real clinical scans.
TAIL_LATE_FRAC: Final[float] = 0.25            # final fraction of frames = "late window"
TAIL_RATIO_TAU: Final[float] = 0.08            # per-voxel late/peak ratio to count as positive-tail
POSITIVE_TAIL_PRESENT_FRAC: Final[float] = 0.60  # tumor-voxel fraction → widen vp anchor (σ 0.60)
POSITIVE_TAIL_SEVERE_FRAC: Final[float] = 0.80   # tumor-voxel fraction → skip vp anchor


# =============================================================================
# SVD Deconvolution Thresholds
# =============================================================================

#: Default singular value truncation threshold (fraction of max) for sSVD
DEFAULT_SVD_THRESHOLD: Final[float] = 0.2

#: Default oscillation index threshold for oSVD. OI(λ) decreases with the
#: truncation threshold λ, so a tighter (smaller) OI target forces the search
#: to a higher threshold, suppressing noise-driven residue-tail oscillation and
#: the MTT inflation it causes; a looser target permits smaller thresholds. The
#: value is the SNR-robust in-basin choice across the clinical TR range, where
#: the oSVD CBF optimum rises modestly with TR and falls with SNR.
DEFAULT_OI_THRESHOLD: Final[float] = 0.095

#: Default Linear search-grid resolution (number of evenly-spaced truncation
#: thresholds) for the oSVD oscillation-index search. Grid-resolution
#: sensitivity is modest and saturates by ~50 steps at the clinical operating
#: point; adaptive bisection is avoided as it converges to the smallest
#: qualifying threshold at the noise edge, inflating the residue-tail integral
#: and the MTT estimate.
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
#: matrix). The natural scale for first-order Tikhonov is the operator
#: norm; absolute λ values are AIF-dependent and not portable. The 4-decade
#: span (1e-3 to 10) covers the well-conditioned regime where the L-curve
#: actually has a corner — narrower ranges miss corners, wider ranges (e.g.
#: starting at 1e-6·σ_max) descend into the rank-deficient AtA noise floor
#: where the L-curve becomes non-monotone.
DEFAULT_LAMBDA_RANGE: Final[TikhonovRange] = TikhonovRange(1e-3, 10.0)

#: Number of log-spaced lambda values in the L-curve sweep. 50 is a
#: defensible compromise between corner-localization quality (which
#: degrades below ~30 samples) and per-voxel runtime cost.
LCURVE_N_LAMBDAS: Final[int] = 50

#: Morozov discrepancy factor for Tikhonov λ-selection. The residue is
#: regularized until the fit residual ‖A·r − C‖ reaches the expected noise
#: magnitude τ·σ_n·√n and no further — fitting below the noise floor is fitting
#: noise. τ=1.0 is Morozov's principle stated exactly, appropriate when the
#: noise estimator (robust MAD of successive concentration differences) reads
#: the true σ_n — which holds when the bolus excursion is sparse relative to the
#: acquisition window (a majority-flat baseline + tail), the regime of clinical
#: DSC. Where that assumption weakens the estimator reads slightly high, biasing
#: τ=1 toward mild over-regularization (the safe direction, a CBF under-read).
TIKHONOV_MOROZOV_TAU: Final[float] = 1.0

#: Degenerate-noise threshold for Tikhonov λ-selection, as a fraction of the
#: tissue-curve peak. When σ_n < this·max(tissue) the noise floor is
#: unmeasurable (essentially noise-free data, e.g. inverse-crime DROs), the
#: discrepancy target is meaningless, and the selector falls through to the
#: fixed TIKHONOV_FALLBACK_LAMBDA_FRACTION. 0.005 sits between synthetic-clean
#: curves (σ_n/peak ≈ 0.0001–0.001) and typical real acquisitions (≥0.01), so on
#: majority-flat clinical DSC curves the guard fires only on near-noise-free
#: data. (σ_n is the MAD-of-diffs estimate; on very short or broad curves signal
#: slope can lift it, so this separation is regime-dependent, not absolute.)
TIKHONOV_MOROZOV_DEGENERATE_FRAC: Final[float] = 0.005

#: Fallback λ (as a fraction of σ_max(A)) used by the Tikhonov DEGENERATE path
#: only — when the noise floor is unmeasurable (σ_n below the degenerate
#: fraction), the Morozov discrepancy target is meaningless, so a fixed small
#: regularization is applied. 0.015 is an EMPIRICAL fraction measured on the
#: near-noise-free DRO AIF (the CBF-optimal noiseless fraction is ≈0.013; 0.015
#: adds a small margin toward more regularization). It governs only data with no
#: measurable noise floor — curves with measurable noise use the discrepancy
#: selector — so its exact value does not affect recovery on such curves.
TIKHONOV_FALLBACK_LAMBDA_FRACTION: Final[float] = 0.015

#: Number of leading residue samples excluded from the first-difference
#: penalty in the modified Tikhonov operator. For these samples the side
#: constraint penalizes the residue magnitude (an identity block) instead
#: of its derivative, so the regularizer does not oversmooth the sharp
#: initial rise of a dispersed residue (which would depress CBF = max R).
#: p = 2 is the smallest count that protects the rise without leaving the
#: initial samples unregularized; it is stable across SNR, MTT, and clinical
#: DSC repetition times (coarser TR under-protects the rise).
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

#: MTT bounds in seconds
MTT_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 30.0)


# =============================================================================
# Vascular (Gamma-Variate) Model Parameters
# =============================================================================

#: Alpha (shape) parameter bounds for gamma-variate model. The upper bound is
#: capped at 10: above ~20-30 the shape α is effectively unidentifiable (CBV
#: = CBF·MTT is invariant in α at fixed MTT), so the cap bounds an
#: ill-conditioned shape parameter at no cost to CBF/CBV/MTT.
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
#: peak without distorting it. Used by Savitzky-Golay on detected and
#: re-extracted AIFs; the integer window length is computed at runtime as an
#: odd number of samples with a three-sample floor for the quadratic filter.
AIF_SMOOTHING_TARGET_SECONDS: Final[float] = 7.5


# =============================================================================
# AIF Per-Voxel Kinetic Filters (post-score, pre-averaging)
# =============================================================================

#: Maximum peak-time spread (seconds) within an averaged AIF voxel pool.
#: True intra-arterial dispersion across the MCA tree (M1 → M3) is sub-
#: second, but the MEASURED
#: spread among co-arterial voxels at typical DSC TR (1-2 s) exceeds this
#: due to (a) frame quantization on argmax peak detection and (b) per-voxel
#: argmax jitter at SNR 20-40 (~±1 frame). Converted at runtime to a frame
#: count via ``max(1, ceil(SECONDS / tr))``. At TR=1.5 s → 2 frames; at
#: TR=2.0 s → 2 frames; at TR=0.5 s (high-temp DSC) → 6 frames.
AIF_PEAK_SPREAD_MAX_SECONDS: Final[float] = 3.0

#: Tissue-TTP causality tolerance (s) for arterial-vs-venous candidate gating.
#: A feeding artery peaks before parenchyma, so a candidate peaking more than
#: ``AIF_TTP_REJECT_AFTER_TISSUE_S`` after tissue is venous and hard-rejected;
#: ``AIF_TTP_NEAR_TISSUE_S`` bounds the mild near-tissue penalty band. Applied
#: in seconds (frame difference × tr) so the gate is TR-invariant.
AIF_TTP_REJECT_AFTER_TISSUE_S: Final[float] = 2.0
AIF_TTP_NEAR_TISSUE_S: Final[float] = 2.0

#: Maximum tail/peak fraction for an AIF voxel (mean ΔR2* over the last
#: ``AIF_TAIL_WINDOW_S`` divided by first-pass peak). Catches BBB-leakage
#: and tissue-PVE-driven late plateau while allowing modest partial-volume
#: tail elevation.
AIF_TAIL_FRACTION_MAX: Final[float] = 0.15

#: Late-tail measurement window (seconds, from end of acquisition). At
#: typical bolus arrival ~30 s and recirculation decay <5% peak by t > 4×
#: MTT, 30 s places the window well
#: past first-pass + recirculation in normal tissue, so a 15% tail
#: indicates extravasation/PVE rather than residual recirculation.
AIF_TAIL_WINDOW_S: Final[float] = 30.0

#: Hard minimum distance (mm) from lesion for AIF candidates. Below this,
#: partial volume with tumor feeding vessels corrupts bolus shape.
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
