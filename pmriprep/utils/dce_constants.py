"""DCE perfusion physics constants.

Centralized constants for DCE-MRI processing including signal conversion,
pharmacokinetic modeling, T1 mapping, and AIF quality assessment.
"""

from typing import Final, Literal, NamedTuple, get_args


# =============================================================================
# AIF Detection Levels
# =============================================================================

DCEDetectionLevel = Literal[
    "atlas",
    "whole_brain",
    "population",
    "external",
]

DCE_DETECTION_LEVELS: Final[tuple[str, ...]] = get_args(DCEDetectionLevel)


# =============================================================================
# Epsilon Guards
# =============================================================================
# Small values to prevent numerical instabilities (divide by zero, log of zero)

#: Minimum ve (extravascular extracellular volume fraction) value
EPSILON_VE: Final[float] = 0.001

#: Numerical EES-volume floor for positive-volume forward evaluation
EPSILON_VE_FORWARD: Final[float] = 1e-6

#: Maximum combined plasma and extravascular extracellular volume fraction
VP_VE_TOTAL_MAX: Final[float] = 1.0

#: Minimum T1 value in seconds
EPSILON_T1: Final[float] = 0.01

#: Dimensionless E1 lower clip (E1 = exp(-TR/T1)). Distinct role from EPSILON_T1
#: (which is a T1-in-seconds floor); at TR 4 ms this implies T1 >= ~0.87 ms, well
#: below any tissue, so it only catches noise-driven near-zero/negative slopes.
EPSILON_E1_LOWER: Final[float] = 0.01

#: Dimensionless floor on the DESPOT1 weight denominator ``1 - E1*cos(alpha)``.
#: With E1 clipped to <= 1 - EPSILON_E1_UPPER the denominator is >= ~1e-4 for real
#: flip angles, so this is a pure divide-by-zero guard that never binds; it must be
#: far below that so the correct large weights for long-T1/small-flip points survive.
EPSILON_WEIGHT_DENOM: Final[float] = 1e-6

#: Relative tolerance for degeneracy of the DESPOT1 normal-equation determinants
#: (``n_fa*sum_x2 - sum_x^2`` and its weighted analogue). Amplitude-invariant, so it
#: keys on collinearity of the flip-angle signals rather than signal magnitude.
EPSILON_DENOM_RTOL: Final[float] = 1e-9

#: Absolute strictly-positive backstop for the determinant guard. Keeps the floored
#: denominator non-zero when the signal scale is ~0 (all-zero-signal voxel), so E1
#: resolves to a finite 0 rather than 0/0 = NaN.
EPSILON_DENOM_ABS: Final[float] = 1e-12

#: Minimum Ktrans value in min^-1
EPSILON_KTRANS: Final[float] = 0.001

#: Numerical plasma-volume floor for division in the 2CXM forward model
EPSILON_VP: Final[float] = 1e-6

#: Minimum B1+ correction factor
EPSILON_B1: Final[float] = 1e-6

#: E1 upper bound offset (E1 clipped to [EPSILON_E1_LOWER, 1.0 - EPSILON_E1_UPPER])
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

#: Concentration standard-deviation cutoff (mM) for the voxel variation check.
MIN_SIGNAL_STD: Final[float] = 0.01

#: Minimum enhancement in mM for voxel to be considered valid
MIN_ENHANCEMENT_MM: Final[float] = 1e-8

#: Bolus-amplitude floor (mM) for AIF scaling and shape calculations.
#: Guards weak peaks and small washout denominators.
MIN_MEANINGFUL_ENHANCEMENT_MM: Final[float] = 0.01


#: Maximum slice count for per-slice T1 edge erosion when CSF QC support is
#: unavailable. Volumes above this count use 3D erosion.
SLAB_EROSION_MAX_SLICES: Final[int] = 30

#: Minimum flow rate to prevent division by zero in 2CXM eigenvalue calculations
EPSILON_FLOW: Final[float] = 1e-9

#: Minimum eigenvalue separation for 2CXM partial fraction decomposition
EPSILON_EIGENVALUE: Final[float] = 1e-10


# =============================================================================
# Blood T1 Values by Field Strength
# =============================================================================

#: Blood T1 values (seconds) by field strength at 37°C.
BLOOD_T1_BY_FIELD: Final[dict[float, float]] = {
    1.5: 1.35,  # Blood T1 at 1.5T
    3.0: 1.65,  # Blood T1 at 3.0T
    7.0: 2.10,  # Blood T1 at 7.0T (extrapolated)
}


def get_blood_t1(field_strength: float) -> float:
    """Get blood T1 value for a given field strength.

    Uses linear interpolation between known values.

    Parameters
    ----------
    field_strength : float
        Magnetic field strength in Tesla

    Returns
    -------
    float
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


#: Tissue T1 QC ranges by field strength (in milliseconds).
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

    Parameters
    ----------
    field_strength : float
        Magnetic field strength in Tesla

    Returns
    -------
    TissueT1Ranges
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
#: Configurable via YAML. Used for C_plasma = C_blood / (1 - hematocrit).
HEMATOCRIT_ARTERIAL: Final[float] = 0.42

#: Alias of the arterial hematocrit default.
HEMATOCRIT = HEMATOCRIT_ARTERIAL

#: Fåhræus ratio: capillary Hct / arterial Hct.
#: Red blood cells travel faster than plasma in small vessels,
#: reducing capillary hematocrit relative to arterial.
#: MRI CBV includes arterioles + capillaries + venules.
#: Expected range: 0.80-0.90.
FAHRAEUS_RATIO: Final[float] = 0.85


def capillary_hematocrit(hct_arterial: float = HEMATOCRIT_ARTERIAL) -> float:
    """Derive capillary hematocrit from arterial via Fåhræus effect.

    Parameters
    ----------
    hct_arterial : float, optional
        Arterial hematocrit (default 0.42).

    Returns
    -------
    float
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


#: Strict AIF acceptance thresholds.
AIF_THRESHOLDS_STRICT: Final[AIFThresholds] = AIFThresholds(
    washout_min=0.20,           # At least 20% washout from peak
    fwhm_min_seconds=8.0,       # FWHM at least 8 seconds
    fwhm_max_seconds=25.0,      # FWHM at most 25 seconds
    baseline_frames=5,          # 5 baseline frames
)

#: Relaxed AIF acceptance thresholds.
AIF_THRESHOLDS_RELAXED: Final[AIFThresholds] = AIFThresholds(
    washout_min=0.10,           # At least 10% washout
    fwhm_min_seconds=5.0,       # FWHM at least 5 seconds
    fwhm_max_seconds=35.0,      # FWHM at most 35 seconds
    baseline_frames=3,          # 3 baseline frames
)

#: Permissive AIF acceptance thresholds: the relaxed washout floor with
#: a wider FWHM interval.
AIF_THRESHOLDS_PERMISSIVE: Final[AIFThresholds] = AIFThresholds(
    washout_min=0.10,           # ≥10% washout (matches relaxed; tissue rejection)
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


#: Multiplier on dose-scaled plasma ``peak_max_mm`` above which the AIF
#: QC warning fires. Suppressed for population-AIF fallback by construction.
DCE_AIF_PEAK_RATIO_WARN_THRESHOLD: Final[float] = 1.5

#: Multi-bump tail detection. Fraction of first-pass peak above which a
#: post-peak local maximum is counted as a "bump."
DCE_AIF_TAIL_PEAK_FRACTION: Final[float] = 0.40

#: Time (seconds) to skip after the first-pass peak before counting tail maxima.
DCE_AIF_TAIL_SHOULDER_SECONDS: Final[float] = 60.0

#: Minimum secondary-peak separation (seconds), converted to frames for
#: the ``scipy.signal.find_peaks`` distance parameter.
DCE_AIF_TAIL_MIN_SEPARATION_SECONDS: Final[float] = 10.0

# =============================================================================
# DCE AIF acceptance gates (G1-G6 + G6 escape)
# =============================================================================

#: G1 — maximum amplitude-floor scale factor before rejection, unless the
#: G6 exception's shape and spatial-consensus conditions are met.
DCE_AIF_GATE_SCALE_HARD: Final[float] = 3.0

#: G6 (a) — upper scale-factor limit for the amplitude-floor exception.
DCE_AIF_GATE_SCALE_G6_CEILING: Final[float] = 8.0

#: Scale-factor threshold selecting the larger uncertainty-inflation setting
#: for a G6-admitted AIF.
DCE_AIF_GATE_SCALE_G6_HIGH: Final[float] = 6.0

#: Pass-2 fallback amplitude-scaling WARN thresholds (report-only; unlike the
#: DCE_AIF_GATE_SCALE_* gates above, these never reject). They act on the
#: Hct-invariant blood-amplitude scale ratio (population target / detected
#: peak): the (1-Hct) plasma conversion cancels between numerator and
#: denominator, so a fixed bar is correct and must not be re-derived per Hct.
#: SUBSTANTIAL: detected peak below ~1/3.4 (~29%) of the population target.
DCE_AIF_SCALE_WARN_SUBSTANTIAL: Final[float] = 3.4
#: UNRELIABLE: detected peak below ~1/5.2 (~19%) of the population target.
DCE_AIF_SCALE_WARN_UNRELIABLE: Final[float] = 5.2

#: G2 — late-tail/peak ratio rejection threshold.
DCE_AIF_GATE_LATE_TAIL_RATIO_MAX: Final[float] = 0.35

#: G3 — FWHM rejection threshold (seconds).
DCE_AIF_GATE_FWHM_MAX_SECONDS: Final[float] = 22.0

#: G4 — maximum accepted count of late-tail secondary peaks.
DCE_AIF_GATE_MULTIBUMP_MAX: Final[int] = 2

#: G5' — rejection and warning thresholds for the post-peak ratio_30 measure.
DCE_AIF_GATE_RATIO30_HARD: Final[float] = 0.50
DCE_AIF_GATE_RATIO30_WARN: Final[float] = 0.40

#: G6 (g) — minimum candidate count for the amplitude-floor exception.
#: Bilateral support uses the first count; unilateral support requires the
#: second count and the minimum number of distinct z-slabs.
DCE_AIF_GATE_G6_MIN_VOXELS_BILATERAL: Final[int] = 5
DCE_AIF_GATE_G6_MIN_VOXELS_UNILATERAL: Final[int] = 10
DCE_AIF_GATE_G6_MIN_SLABS_UNILATERAL: Final[int] = 2

#: Per-parameter uncertainty multipliers for base inflation=2.0. For larger
#: base inflation, the excess above 1.0 is scaled proportionally; base<=1.0
#: leaves all uncertainties unchanged. This policy leaves ve uninflated.
DCE_AIF_G6_SIGMA_INFLATION: Final[dict[str, float]] = {
    "Ktrans": 2.0,
    "vp": 1.5,
    "ve": 1.0,
    "Fp": 2.0,
    "PS": 2.0,
}

#: Minimum candidate pool size for progressive trim acceptance.
#: Matches DSC's MIN_AIF_CANDIDATES_ACCEPT and the existing DCE
#: _MIN_CANDIDATES_FOR_DETECTED for cross-modality consistency.
DCE_AIF_MIN_TRIM_POOL: Final[int] = 2


def get_aif_thresholds(criteria: str = "strict", tr: float = 0.0) -> AIFThresholds:
    """Get AIF quality thresholds based on criteria level.

    Supports TR-adaptive FWHM minimum for low temporal resolution DCE
    acquisitions (TR > 5s), where the bolus may span only 1-2 samples.

    Parameters
    ----------
    criteria : str, optional
        One of "strict", "relaxed", "permissive", or "degraded".
    tr : float, optional
        Temporal resolution in seconds. If > 5.0, FWHM minimum is
        adapted to ``max(tr * 0.8, base_fwhm_min)``.

    Returns
    -------
    AIFThresholds
        AIFThresholds with appropriate values.

    Raises
    ------
    ValueError
        If criteria is not a recognized level.
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


#: Expected AIF values for full-dose contrast (0.1 mmol/kg).
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
VE_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 1.0)

#: vp (plasma volume fraction) bounds
VP_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 1.0)

#: kep (efflux rate) bounds in min^-1
KEP_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 10.0)

#: Fp (plasma flow) bounds in mL/min/100mL for 2CXM
FP_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 500.0)

#: PS (permeability-surface area product) bounds in mL/min/100mL for 2CXM
PS_BOUNDS: Final[ParameterBounds] = ParameterBounds(0.0, 500.0)


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

    Processing bounds for AFI, VFA T1 mapping and DCE conversion are stored
    separately from the QC thresholds for each nominal field strength.

    Attributes
    ----------
    clip_low : float
        AFI computation clipping lower bound.
    clip_high : float
        AFI computation clipping upper bound.
    vfa_clip_low : float
        VFA T1 mapping clipping lower bound.
    vfa_clip_high : float
        VFA T1 mapping clipping upper bound.
    dce_clip_low : float
        DCE signal conversion clipping lower bound.
    dce_clip_high : float
        DCE signal conversion clipping upper bound.
    outlier_deviation : float
        Local deviation threshold for outlier rejection.
    max_std : float
        QC max whole-brain B1+ standard deviation.
    extreme_low : float
        QC flagging threshold (low).
    extreme_high : float
        QC flagging threshold (high).
    mean_range : tuple[float, float]
        QC acceptable mean range.
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
        dce_clip_high=1.7,
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
    :func:`get_tissue_t1_ranges`. Selects a catalogued threshold set without
    interpolation; values outside the table use the nearest endpoint.

    Parameters
    ----------
    field_strength : float
        Magnetic field strength in Tesla.

    Returns
    -------
    B1Thresholds
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

#: Signal ratio validity range for AFI: r = S2/S1 (long-TR over short-TR) in
#: (low, high) for stable arccos.
B1_RATIO_VALID_RANGE: Final[tuple[float, float]] = (0.05, 1.5)

#: Minimum ``|n - r|`` for denominator stability in AFI equation.
B1_DENOMINATOR_THRESHOLD: Final[float] = 0.1

#: Minimum normalized convolution coverage for reliable local B1+ estimate.
B1_MIN_COVERAGE: Final[float] = 0.10

#: Minimum smoothing sigma in voxels.
B1_MIN_SIGMA_VOXELS: Final[float] = 1.5


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

#: Reduced erosion for atlas-constrained search. The MRA-derived atlas
#: already excludes cortical veins; minimal erosion suffices to avoid
#: partial-volume at the brain boundary.
BRAIN_EROSION_ITERATIONS_ATLAS: Final[int] = 2

#: Minimum number of voxels remaining after erosion; fall back to original
#: mask if erosion leaves fewer than this.
BRAIN_EROSION_MIN_VOXELS: Final[int] = 100

#: Distance (mm) from slab edges to exclude as slab-profile rolloff
#: (B1/RF rolloff at the outermost 1-2 partitions). Applied at any
#: edge where neither SSS exclusion nor 3D-slab inflow exclusion
#: dominates: atlas-tier inferior + superior, and as the inferior
#: rolloff baseline before ``_inflow_exclusion_mm`` may extend it.
#: WB superior always uses ``SSS_EXCLUSION_MM`` (larger; SSS is
#: dorsal midline). Inferior side at both atlas + WB tiers uses
#: ``_inflow_exclusion_mm`` (applied universally — see
#: ``pmriprep/interfaces/dce_aif.py`` cascade call sites).
SLAB_EDGE_EXCLUSION_MM_ROLLOFF: Final[float] = 4.0

#: Distance (mm) from the SUPERIOR slab edge to exclude in the
#: whole-brain AIF search to keep the superior sagittal sinus out of
#: the unconstrained candidate pool. SSS is a dorsal-midline venous
#: structure that enhances heavily and early — a classic venous
#: false positive when no atlas constraint is in play. The atlas tier
#: does not need this cut because the MRA-derived atlas already
#: filters SSS by anatomy. Applied to WB superior under both 2D and
#: 3D acquisitions; not applied to atlas tier or to WB inferior.
SSS_EXCLUSION_MM: Final[float] = 15.0

#: Base distance (mm) from the INFERIOR slab edge to exclude for
#: 3D-slab entry-slice inflow. Inflow contamination has two components:
#: a geometric refresh depth (unsaturated spins flowing into the slab;
#: 8-15mm for arterial velocities at typical 3-6ms pulse-TR) and a
#: partition-count component (slab-profile wings; ~3-5 outermost
#: partitions). The base 20mm covers the geometric depth for thin
#: isotropic acquisitions; the helper ``_inflow_exclusion_mm`` adapts
#: upward for thicker partitions and caps at 35% of slab depth so very
#: thick / thin-slab acquisitions don't over-exclude. Pipeline enforces
#: RAS+ reorientation upstream, so z=0 is always the inferior partition.
INFLOW_EXCLUSION_MM_INFERIOR_BASE: Final[float] = 20.0

#: Cap on inferior inflow exclusion as a fraction of total slab depth.
#: Other search-mask exclusions are applied separately.
INFLOW_EXCLUSION_MAX_SLAB_FRACTION: Final[float] = 0.35

#: Hard minimum distance (mm) from lesion for AIF candidates. Below this,
#: partial volume with tumor feeding vessels corrupts bolus shape.
LESION_EXCLUSION_FLOOR_MM: Final[float] = 10.0

#: Recommended minimum distance (mm) from lesion for AIF candidate voxels.
LESION_EXCLUSION_DISTANCE_MM: Final[float] = 15.0

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
#: Constants and helpers for adapting AIF timing thresholds to the interval
#: between dynamic volumes.

#: Target timing margin in seconds, rounded to at least one frame.
AIF_TIMING_TARGET_SECONDS: Final[float] = 4.0

#: Temporal-resolution cutoff (seconds) for the low-resolution policy.
AIF_LOW_TR_THRESHOLD: Final[float] = 5.0


# =============================================================================
# DCE Signal Quality QC Thresholds
# =============================================================================

#: Lower peak-enhancement cutoff (%) for signal-quality QC.
QC_PEAK_ENHANCEMENT_MIN: Final[float] = 5.0

#: Peak-enhancement warning threshold (%).
QC_PEAK_ENHANCEMENT_WARN: Final[float] = 20.0

#: Lower baseline-tSNR cutoff for signal-quality QC.
QC_DCE_BASELINE_TSNR_MIN: Final[float] = 3.0

#: Baseline-tSNR warning threshold.
QC_DCE_BASELINE_TSNR_WARN: Final[float] = 8.0

#: Mean B1+ warning bounds for report QC, shared across field strengths.
QC_B1_MEAN_LOW: Final[float] = 0.7
QC_B1_MEAN_HIGH: Final[float] = 1.3

#: Strict arterial FWHM upper bound in seconds.
AIF_FWHM_MAX_STRICT_SECONDS: Final[float] = 25.0

#: Relaxed arterial FWHM upper bound in seconds.
AIF_FWHM_MAX_RELAXED_SECONDS: Final[float] = 45.0

#: Ideal arterial FWHM in seconds for scoring.
AIF_IDEAL_FWHM_SECONDS: Final[float] = 12.0

#: Half-width (seconds) of the FWHM acceptance band around the ideal, used by the
#: averaged-AIF fidelity scorer.
AIF_FWHM_BAND_HALFWIDTH_SECONDS: Final[float] = 8.0

#: Center of the averaged-AIF washout scoring band (peak versus late-tail mean).
AIF_WASHOUT_PHYSIOLOGIC_CENTER: Final[float] = 0.78

#: Lower half-width of the washout scoring band.
AIF_WASHOUT_BAND_HALFWIDTH: Final[float] = 0.13

#: Upper half-width of the washout scoring band.
AIF_WASHOUT_BAND_HALFWIDTH_UPPER: Final[float] = 0.20

#: Target and lower bound for the late-tail-mean/peak persistence score.
AIF_TAIL_PERSISTENCE_TARGET: Final[float] = 0.25
AIF_TAIL_PERSISTENCE_FLOOR: Final[float] = 0.15

#: CNR value at which the averaged-AIF scorer's CNR contribution saturates.
AIF_CNR_SATURATION: Final[float] = 17.5

# The averaged-AIF fidelity composite is diagnostic. Runtime acceptance uses
# the individual shape, amplitude and timing gates.


def compute_timing_margin_frames(tr: float) -> int:
    """Compute TR-adaptive timing margin in frames.

    At TR=2s, returns 2 frames (4s / 2s). At TR>=4s, returns 1 frame.
    Nonpositive intervals return the one-frame minimum.

    Parameters
    ----------
    tr : float
        Temporal resolution in seconds.

    Returns
    -------
    int
        Timing margin in frames (minimum 1).
    """
    if tr <= 0:
        return 1
    return max(1, round(AIF_TIMING_TARGET_SECONDS / tr))


def compute_fwhm_threshold_frames(tr: float, strict: bool = True) -> float:
    """Compute the max-FWHM ("narrow bolus") threshold in frames.

    Converts the configured seconds ceiling to frames, with a two-frame
    minimum. Nonpositive intervals return 5 strict or 8 relaxed frames.
    - At TR=2s: 25/2 = 12.5 frames (= 25 s)
    - At TR=0.5s: 25/0.5 = 50 frames (= 25 s)
    - At TR=15s: max(2, 25/15=1.7) = 2 frames (resolution floor)

    Parameters
    ----------
    tr : float
        Temporal resolution in seconds.
    strict : bool, optional
        If True, use strict thresholds; otherwise relaxed.

    Returns
    -------
    float
        Max-FWHM threshold in frames.
    """
    if tr <= 0:
        return 5.0 if strict else 8.0
    sec_based = (AIF_FWHM_MAX_STRICT_SECONDS if strict
                 else AIF_FWHM_MAX_RELAXED_SECONDS) / tr
    return max(2.0, sec_based)


def is_low_temporal_resolution(tr: float) -> bool:
    """Check whether the volume interval exceeds the low-resolution cutoff.

    The low-resolution policy applies above ``AIF_LOW_TR_THRESHOLD``.

    Parameters
    ----------
    tr : float
        Temporal resolution in seconds.

    Returns
    -------
    bool
        True if TR is above the low-TR threshold.
    """
    return tr > AIF_LOW_TR_THRESHOLD


# =============================================================================
# DSC→DCE Cross-Modality vp Constraint Constants
# =============================================================================
# Governs how DSC-derived CBV maps into DCE vp priors via NAWM
# normalization instead of absolute-CBV conversion.

#: Upper supported vp for DSC-derived prior inputs. Larger scaled values
#: are withheld by ``ncbv_to_vp_prior``.
VP_MAX_PHYSIOLOGICAL: Final[float] = 0.2

#: Assumed NAWM plasma-volume fraction used to scale nCBV into the DCE vp prior.
NAWM_VP_REFERENCE: Final[float] = 0.010

#: Accepted median range for the DSC-derived vp prior. A median outside this
#: range disables the prior through the cross-modality quality gate.
VP_MEDIAN_MIN: Final[float] = 0.005
VP_MEDIAN_MAX: Final[float] = 0.03

#: Fixed relative SD for the quality-qualified DSC-informed DCE vp prior.
#: This is the widest qualified release-policy width, not an uncertainty
#: estimated from the subject's K2 distribution.
SIGMA_SOFT_DSC_PRIOR: Final[float] = 0.60
