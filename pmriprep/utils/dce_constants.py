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

#: E1 upper bound offset (E1 clipped to [EPSILON_T1, 1.0 - EPSILON_E1_UPPER])
#: Upper bound offset 1e-4 gives E1 <= 0.9999, preserving long-T1 voxels
EPSILON_E1_UPPER: Final[float] = 1e-4

#: Minimum value for sin/tan of corrected flip angles to prevent division by zero
EPSILON_TRIG: Final[float] = 1e-6

#: Signal epsilon for baseline guard
EPSILON_SIGNAL: Final[float] = 1e-9

#: Epsilon for signal ratio denominators
EPSILON_RATIO: Final[float] = 1e-10

#: Minimum baseline concentration noise for CNR computation (mM)
#: Below this, baseline is effectively noiseless — return max CNR
EPSILON_CONCENTRATION: Final[float] = 1e-6

#: Minimum signal standard deviation for voxel to be considered valid
#: Value is in mM (concentration units), not raw signal units
#: Normal tissue concentration std is 0.01-0.5 mM; 0.01 filters noise-only voxels
MIN_SIGNAL_STD: Final[float] = 0.01

#: Minimum enhancement in mM for voxel to be considered valid
MIN_ENHANCEMENT_MM: Final[float] = 1e-8


#: Maximum number of slices for 2D per-slice erosion.
#: 3D VIBE thin slabs (≤30 slices) have well-defined slice profile —
#: through-plane partial volume is minimal, so only in-plane erosion
#: is physically motivated. Whole-brain volumes (>30 slices) use 3D erosion.
SLAB_EROSION_MAX_SLICES: Final[int] = 30

#: Minimum flow rate to prevent division by zero in 2CXM eigenvalue calculations
EPSILON_FLOW: Final[float] = 1e-9

#: Minimum eigenvalue separation for 2CXM partial fraction decomposition
EPSILON_EIGENVALUE: Final[float] = 1e-10


# =============================================================================
# Field-Strength Dependent R1 Relaxivity
# =============================================================================

#: r1 relaxivity by field strength (T) in s^-1 per mM (Gd-DTPA, plasma)
#: OSIPI standard values (https://osipi.org/)
R1_RELAXIVITY_BY_FIELD: Final[dict[float, float]] = {
    1.5: 3.9,   # Gd-DTPA at 1.5T (OSIPI standard)
    3.0: 3.8,   # Gd-DTPA at 3.0T (OSIPI standard)
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

#: Default arterial hematocrit for AIF blood-to-plasma conversion.
#: QIBA DCE-MRI v2.0 recommendation. Configurable via config YAML.
#: Used for: C_plasma = C_blood / (1 - hematocrit)
HEMATOCRIT_ARTERIAL: Final[float] = 0.42

#: Backwards-compat alias
HEMATOCRIT = HEMATOCRIT_ARTERIAL

#: Fåhræus ratio: capillary Hct / arterial Hct.
#: Red blood cells travel faster than plasma in small vessels,
#: reducing capillary hematocrit relative to arterial.
#: MRI CBV includes arterioles + capillaries + venules.
#: Refs: Lipowsky, Microcirculation 2005; Levin et al., MRM 2004;
#: Boxerman et al., MRM 1995. Literature range: 0.80-0.90.
FAHRAEUS_RATIO: Final[float] = 0.85


def capillary_hematocrit(hct_arterial: float = HEMATOCRIT_ARTERIAL) -> float:
    """Derive capillary hematocrit from arterial via Fåhræus effect.

    Args:
        hct_arterial: Arterial hematocrit (default 0.42).

    Returns:
        Capillary hematocrit (default 0.42 × 0.85 = 0.357).
    """
    return hct_arterial * FAHRAEUS_RATIO


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

#: Permissive AIF quality thresholds (for very difficult cases)
AIF_THRESHOLDS_PERMISSIVE: Final[AIFThresholds] = AIFThresholds(
    washout_min=0.05,           # At least 5% washout
    fwhm_min_seconds=3.0,       # FWHM at least 3 seconds
    fwhm_max_seconds=45.0,      # FWHM at most 45 seconds
    baseline_frames=3,          # 3 frames for baseline
)

#: Degraded AIF quality thresholds (last resort before population fallback)
AIF_THRESHOLDS_DEGRADED: Final[AIFThresholds] = AIFThresholds(
    washout_min=0.02,           # Minimal washout required
    fwhm_min_seconds=2.0,       # FWHM at least 2 seconds
    fwhm_max_seconds=60.0,      # FWHM at most 60 seconds
    baseline_frames=2,          # 2 frames for baseline
)


def get_aif_thresholds(criteria: str = "strict", tr: float = 0.0) -> AIFThresholds:
    """Get AIF quality thresholds based on criteria level.

    Supports TR-adaptive FWHM minimum for low temporal resolution DCE
    acquisitions (TR > 5s), where the bolus may span only 1-2 samples.

    Args:
        criteria: One of "strict", "relaxed", "permissive", or "degraded".
        tr: Temporal resolution in seconds. If > 5.0, FWHM minimum is
            adapted to ``max(tr * 0.8, base_fwhm_min)``.

    Returns:
        AIFThresholds with appropriate values.

    Raises:
        ValueError: If criteria is not a recognized level.
    """
    thresholds_map: dict[str, AIFThresholds] = {
        "strict": AIF_THRESHOLDS_STRICT,
        "relaxed": AIF_THRESHOLDS_RELAXED,
        "permissive": AIF_THRESHOLDS_PERMISSIVE,
        "degraded": AIF_THRESHOLDS_DEGRADED,
    }
    if criteria not in thresholds_map:
        valid = ", ".join(sorted(thresholds_map.keys()))
        raise ValueError(f"Unknown AIF criteria: {criteria}. Use one of: {valid}.")

    base = thresholds_map[criteria]

    if tr > 5.0:
        adaptive_min = max(tr * 0.8, base.fwhm_min_seconds)
        return base._replace(fwhm_min_seconds=adaptive_min)

    return base


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
KTRANS_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 5.0)

#: ve (extravascular extracellular volume) bounds (fraction)
VE_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 0.9)

#: vp (plasma volume fraction) bounds
VP_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.001, 0.3)

#: kep (efflux rate) bounds in min^-1
KEP_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 10.0)

#: Maximum kep (efflux rate) for the forward model. Must be high enough to
#: avoid biasing the optimizer — capping in evaluate() flattens the objective
#: surface for Ktrans/ve combinations where true kep > cap. 1000.0 is
#: effectively uncapped for physiological parameters
#: while still guarding against degenerate numerics.
MAX_KEP: Final[float] = 1000.0

#: Maximum eigenvalue for 2CXM slow compartment (similar to kep)
MAX_EIGENVALUE_SLOW: Final[float] = 20.0

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
# B1+ Field-Strength-Dependent Thresholds
# =============================================================================


class B1Thresholds(NamedTuple):
    """Field-strength-dependent B1+ thresholds.

    AFI B1+ inhomogeneity increases with field strength due to shorter RF
    wavelength relative to head size.  At 7 T dielectric focusing can push
    B1+ to 1.8 or higher in the brain centre while temporal lobes may drop
    to 0.3 (Teeuwisse MRM 2012).  Thresholds must widen accordingly.

    Attributes:
        clip_low: AFI computation clipping lower bound.
        clip_high: AFI computation clipping upper bound.
        vfa_clip_low: VFA T1 mapping clipping lower bound.
        vfa_clip_high: VFA T1 mapping clipping upper bound.
        dce_clip_low: DCE signal conversion clipping lower bound.
        dce_clip_high: DCE signal conversion clipping upper bound.
        outlier_deviation: Local deviation threshold for outlier rejection.
        max_std: QC max whole-brain B1+ standard deviation.
        extreme_low: QC flagging threshold (low).
        extreme_high: QC flagging threshold (high).
        mean_range: QC acceptable mean range.
    """

    clip_low: float
    clip_high: float
    vfa_clip_low: float
    vfa_clip_high: float
    dce_clip_low: float
    dce_clip_high: float
    outlier_deviation: float
    max_std: float
    extreme_low: float
    extreme_high: float
    mean_range: tuple[float, float]


#: B1+ threshold lookup by field strength (Tesla).
#: References: Yarnykh MRM 2007, Teeuwisse MRM 2012, Nehrke & Bornert MRM 2014.
B1_THRESHOLDS_BY_FIELD: Final[dict[float, B1Thresholds]] = {
    1.5: B1Thresholds(
        clip_low=0.4,
        clip_high=1.6,
        vfa_clip_low=0.6,
        vfa_clip_high=1.4,
        dce_clip_low=0.6,
        dce_clip_high=1.4,
        outlier_deviation=0.20,
        max_std=0.15,
        extreme_low=0.70,
        extreme_high=1.30,
        mean_range=(0.90, 1.10),
    ),
    3.0: B1Thresholds(
        clip_low=0.2,
        clip_high=2.0,
        vfa_clip_low=0.5,
        vfa_clip_high=1.7,
        dce_clip_low=0.5,
        dce_clip_high=1.5,
        outlier_deviation=0.30,
        max_std=0.25,
        extreme_low=0.50,
        extreme_high=1.50,
        mean_range=(0.85, 1.15),
    ),
    7.0: B1Thresholds(
        clip_low=0.2,
        clip_high=2.5,
        vfa_clip_low=0.3,
        vfa_clip_high=2.0,
        dce_clip_low=0.3,
        dce_clip_high=2.0,
        outlier_deviation=0.50,
        max_std=0.45,
        extreme_low=0.30,
        extreme_high=1.80,
        mean_range=(0.75, 1.25),
    ),
}


def get_b1_thresholds(field_strength: float) -> B1Thresholds:
    """Get B1+ thresholds for the nearest known field strength.

    Uses the same nearest-field-strength lookup pattern as
    :func:`get_tissue_t1_ranges`.  No interpolation — thresholds are
    inherently discrete (literature-validated per field strength).

    Args:
        field_strength: Magnetic field strength in Tesla.

    Returns:
        B1Thresholds for the closest catalogued field strength.
    """
    fields = sorted(B1_THRESHOLDS_BY_FIELD.keys())

    if field_strength <= fields[0]:
        return B1_THRESHOLDS_BY_FIELD[fields[0]]
    if field_strength >= fields[-1]:
        return B1_THRESHOLDS_BY_FIELD[fields[-1]]

    for i, f in enumerate(fields[:-1]):
        if f <= field_strength < fields[i + 1]:
            if (field_strength - f) < (fields[i + 1] - field_strength):
                return B1_THRESHOLDS_BY_FIELD[f]
            else:
                return B1_THRESHOLDS_BY_FIELD[fields[i + 1]]

    return B1_THRESHOLDS_BY_FIELD[3.0]


# =============================================================================
# B1+ Math-Derived Constants (not field-dependent)
# =============================================================================

#: Signal ratio validity range for AFI: r = S1/S2 in (low, high) for stable arccos.
B1_RATIO_VALID_RANGE: Final[tuple[float, float]] = (0.05, 1.5)

#: Minimum |n - r| for denominator stability in AFI equation.
B1_DENOMINATOR_THRESHOLD: Final[float] = 0.1

#: Minimum normalized convolution coverage for reliable local B1+ estimate.
B1_MIN_COVERAGE: Final[float] = 0.10

#: Minimum smoothing sigma in voxels.
B1_MIN_SIGMA_VOXELS: Final[float] = 1.5


# =============================================================================
# NLLS T1 Fitting
# =============================================================================

#: NLLS refinement threshold for hybrid T1 fitting.
#: Voxels with E1 > this threshold have T1 >> TR, making DESPOT1
#: linearization unreliable. At TR=4ms, E1=0.995 corresponds to T1~800ms.
#: Most brain tissue at clinical field strengths falls near or above this.
#: For short-TR VFA (<10ms), this captures CSF and edematous tissue.
NLLS_REFINEMENT_E1_THRESHOLD: Final[float] = 0.995


# =============================================================================
# T1 Mapping Defaults
# =============================================================================

#: Default T1 value (ms) when T1 mapping is skipped
DEFAULT_T1_MS: Final[float] = 1200.0


# =============================================================================
# DCE Processing Defaults
# =============================================================================

#: Default number of baseline frames for pre-contrast calculation
DEFAULT_BASELINE_FRAMES: Final[int] = 5

#: Default tSNR threshold for voxel inclusion (0 = fit all brain voxels)
DEFAULT_TSNR_THRESHOLD: Final[float] = 0.0

#: Tissue density in g/mL (for unit conversions)
TISSUE_DENSITY: Final[float] = 1.04


# =============================================================================
# AIF Search Mask Preparation
# =============================================================================

#: Number of binary erosion iterations for brain mask before AIF search.
#: Excludes cortical surface veins that would contaminate the AIF.
BRAIN_EROSION_ITERATIONS: Final[int] = 8

#: Minimum number of voxels remaining after erosion; fall back to original
#: mask if erosion leaves fewer than this.
BRAIN_EROSION_MIN_VOXELS: Final[int] = 100

#: Distance (mm) from inferior/superior slab edges to exclude from AIF search.
#: Avoids 3D VIBE inflow artifacts at slab boundaries.
SLAB_EDGE_EXCLUSION_MM: Final[float] = 15.0

#: Minimum distance (mm) from lesion for AIF candidate voxels.
LESION_EXCLUSION_DISTANCE_MM: Final[float] = 15.0

#: Parker AIF reference relaxivity (s^-1 mM^-1) for Gd-DTPA at 1.5T.
#: Used for relaxivity correction when contrast agent differs from original model.
PARKER_R1_REFERENCE: Final[float] = 4.1


# =============================================================================
# AIF Spatial Exclusion Filters
# =============================================================================
#: Exclusions applied during search mask preparation to avoid venous structures.
#: SSS (Superior Sagittal Sinus) runs midline and drains superiorly.

#: Midline exclusion: fraction from center on each side (total = 2x this value).
#: E.g., 0.10 excludes central 20% of FOV, avoiding SSS, ISS, sagittal sinus.
MIDLINE_EXCLUSION_FRACTION: Final[float] = 0.10

#: Superior exclusion: fraction of slices from top to exclude.
#: E.g., 0.75 means exclude top 25% (z >= 0.75 * nz).
#: Avoids cortical vein drainage zone.
SUPERIOR_EXCLUSION_FRACTION: Final[float] = 0.75

# NOTE: No posterior exclusion applied.
# Only midline and superior exclusions are applied. The ISS is adequately
# excluded by the midline exclusion since it runs along the falx cerebri.


# =============================================================================
# AIF Timing Criterion
# =============================================================================
#: Arterial blood arrives BEFORE tissue enhancement.
#: This defines minimum frames the AIF peak must precede tissue TTP.

#: Number of frames AIF must peak BEFORE tissue TTP to be considered arterial.
#: Candidates peaking at or after tissue are likely venous (SSS, etc.).
AIF_TIMING_BEFORE_TISSUE_FRAMES: Final[int] = 2

#: Fraction of acquisition for early arrival criterion (AIF should peak in first 40%).
AIF_PEAK_TIMING_FRACTION: Final[float] = 0.4


# =============================================================================
# AIF FWHM Thresholds (Frame-Based)
# =============================================================================
#: These are used in the 7-tier arterial criteria cascade.

#: Strict FWHM threshold in frames (Tier 1-3).
AIF_FWHM_MAX_STRICT_FRAMES: Final[int] = 5

#: Relaxed FWHM threshold in frames (Tier 4-5).
AIF_FWHM_MAX_RELAXED_FRAMES: Final[int] = 8

#: Washout threshold for strict arterial criterion.
AIF_WASHOUT_THRESHOLD_STRICT: Final[float] = 0.20

#: Washout threshold for relaxed arterial criterion.
AIF_WASHOUT_THRESHOLD_RELAXED: Final[float] = 0.10


# =============================================================================
# AIF FWHM Scoring (Gaussian-Based)
# =============================================================================
#: Gaussian scoring centered on ideal arterial FWHM (~3 frames for typical DCE).
#: Venous structures have broader FWHM due to slower washout.

#: Ideal FWHM in frames for arterial first-pass bolus.
AIF_IDEAL_FWHM_FRAMES: Final[float] = 3.0

#: Width parameter for Gaussian FWHM scoring.
AIF_FWHM_SCORING_WIDTH: Final[float] = 3.0

#: FWHM threshold (frames) above which structure is likely venous.
AIF_FWHM_VENOUS_THRESHOLD_FRAMES: Final[int] = 8

#: Score assigned to candidates with venous-like FWHM (> threshold).
AIF_FWHM_VENOUS_SCORE: Final[float] = 0.2

#: Score assigned to candidates with noise-like FWHM (< 0.5 frames).
AIF_FWHM_NOISE_SCORE: Final[float] = 0.1


# =============================================================================
# TR-Adaptive AIF Detection
# =============================================================================
#: Constants and helpers for TR-adaptive AIF detection.
#: At low temporal resolution (TR > 5s), frame-based thresholds become
#: unphysiological (e.g., 2 frames at TR=8s = 16s, far beyond arterial-tissue
#: transit time of 3-5s). These helpers adapt thresholds to work at any TR.

#: Target arterial-tissue transit time in seconds (physiology-based).
AIF_TIMING_TARGET_SECONDS: Final[float] = 4.0

#: TR threshold above which timing-based discrimination is unreliable.
AIF_LOW_TR_THRESHOLD: Final[float] = 5.0


# =============================================================================
# DCE Signal Quality QC Thresholds
# =============================================================================

#: Minimum peak enhancement for reliable DCE quantification (%)
#: 5% ≈ 2σ above noise floor; failed injection shows <3%
#: Heye et al., NMR Biomed 2016: WM enhancement can be 15% in elderly
QC_PEAK_ENHANCEMENT_MIN: Final[float] = 5.0

#: Peak enhancement below this triggers REVIEW (%)
#: Half-dose standard Gd-DTPA shows 10-20% in WM
QC_PEAK_ENHANCEMENT_WARN: Final[float] = 20.0

#: Minimum baseline tSNR for reliable PK fitting
#: Tofts JMRI 2010: Ktrans CV >50% at tSNR<3
QC_DCE_BASELINE_TSNR_MIN: Final[float] = 3.0

#: Baseline tSNR below this triggers REVIEW
#: Tofts JMRI 2010: Ktrans CV ~20% at tSNR=8
QC_DCE_BASELINE_TSNR_WARN: Final[float] = 8.0

#: B1+ mean acceptable range for QC gating
#: Values outside this range trigger REVIEW (not EXCLUDE — B1 bias is correctable)
#: Uses 1.5T extreme thresholds as conservative bound across field strengths
#: Ref: B1_THRESHOLDS_BY_FIELD extreme_low/extreme_high at 1.5T
QC_B1_MEAN_LOW: Final[float] = 0.7
QC_B1_MEAN_HIGH: Final[float] = 1.3

#: Strict arterial FWHM upper bound in seconds.
AIF_FWHM_MAX_STRICT_SECONDS: Final[float] = 25.0

#: Relaxed arterial FWHM upper bound in seconds.
AIF_FWHM_MAX_RELAXED_SECONDS: Final[float] = 45.0

#: Ideal arterial FWHM in seconds for scoring.
AIF_IDEAL_FWHM_SECONDS: Final[float] = 12.0


def compute_timing_margin_frames(tr: float) -> int:
    """Compute TR-adaptive timing margin in frames.

    At TR=2s, returns 2 frames (4s / 2s). At TR>=4s, returns 1 frame.
    Ensures the timing margin is physiologically meaningful at any TR.

    Args:
        tr: Temporal resolution in seconds.

    Returns:
        Timing margin in frames (minimum 1).
    """
    if tr <= 0:
        return 1
    return max(1, round(AIF_TIMING_TARGET_SECONDS / tr))


def compute_fwhm_threshold_frames(tr: float, strict: bool = True) -> float:
    """Compute TR-adaptive FWHM threshold in frames.

    Takes the minimum of the frame-based threshold and
    a seconds-based threshold converted to frames. This ensures the
    threshold is meaningful at any TR:
    - At TR=2s: min(5, 25/2=12.5) = 5 frames (frame-based dominates)
    - At TR=8s: min(5, 25/8=3.1) = 3.1 frames (seconds-based dominates)

    Args:
        tr: Temporal resolution in seconds.
        strict: If True, use strict thresholds; otherwise relaxed.

    Returns:
        FWHM threshold in frames.
    """
    if tr <= 0:
        return 5.0 if strict else 8.0
    frame_based = 5.0 if strict else 8.0
    sec_based = (AIF_FWHM_MAX_STRICT_SECONDS if strict
                 else AIF_FWHM_MAX_RELAXED_SECONDS) / tr
    return min(frame_based, sec_based)


def is_low_temporal_resolution(tr: float) -> bool:
    """Check if TR is too high for reliable timing-based AIF discrimination.

    At TR > 5s, the 2-frame timing margin exceeds physiological
    arterial-tissue transit time, making timing criteria unreliable.

    Args:
        tr: Temporal resolution in seconds.

    Returns:
        True if TR is above the low-TR threshold.
    """
    return tr > AIF_LOW_TR_THRESHOLD
