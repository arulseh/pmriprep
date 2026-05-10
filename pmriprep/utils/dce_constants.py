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

#: Maximum allowed sum of ve + vp when the volume constraint is enforced
#: post-fit. 0.999 leaves a small slack to keep LOG_LOGIT inverse-transform
#: numerically well-behaved at the boundary; using exactly 1.0 occasionally
#: produces NaN gradients in the boundary cell of the optimizer.
VP_VE_TOTAL_MAX: Final[float] = 0.999

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
#
# Single source of truth: CONTRAST_AGENTS in contrast_agent.py.
# This module's get_r1_relaxivity(field_strength) delegates to that table
# under the Gd-DTPA fallback agent. Per-agent values, per-field citations,
# and 7T support live there.


def get_r1_relaxivity(field_strength: float) -> float:
    """Get fallback r1 relaxivity (Gd-DTPA) for a given field strength.

    Used when the BIDS sidecar does not identify a specific contrast agent.
    Delegates to :func:`pmriprep.utils.contrast_agent.lookup_r1` with
    ``"gd-dtpa"``, so all per-agent values come from one table.

    Args:
        field_strength: Magnetic field strength in Tesla.

    Returns:
        r1 relaxivity in s^-1 per mM.

    Raises:
        ValueError: If field_strength > 5.0 T and no peer-reviewed 7T value
            exists for the fallback agent. Gd-DTPA has a Noebauer-Huhmann
            2010 7T value, so this should not raise in practice.
    """
    from pmriprep.utils.contrast_agent import lookup_r1

    return lookup_r1("gd-dtpa", field_strength)


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
#: Note: ``washout_min`` matches the relaxed value (0.10) — physics review
#: (Calamante NMR Biomed 2013) showed the CNR relaxation (5.0→2.0) is
#: sufficient for the sensitivity gain on low-dose/preloaded protocols, and
#: lowering washout below 0.10 admits tissue voxels without proportional
#: arterial-detection benefit.
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
#: QC warning fires. Derived from two independent bounds, not tuned to
#: any specific dataset:
#:
#: 1. Encoded Parker 2006 population spread is ``peak_max / peak_min
#:    = 7.6 / 5.0 = 1.52`` (P95/P5-equivalent across cardiac output,
#:    injection rate, weight). Literature observed-peak ceilings
#:    (Parker 2006 max ~8.5 mM plasma, Schabel 2008 max ~9 mM plasma)
#:    scale to ~1.18x expected_max — both views of the same biology
#:    place clean-AIF ceiling well below 1.5x.
#: 2. Engineering tolerance: the largest plausible single-source
#:    operator/equipment error (50% over-dose, mis-recorded weight,
#:    50% injection-rate error) inflates peak by 1.5x. Threshold sits
#:    at this boundary — peaks above 1.5x cannot be explained by
#:    realistic dose error and require artifact diagnosis (inflow,
#:    motion, partial volume).
#:
#: 2.0x would be too lax — past the literature-observed clean-AIF
#: ceiling, so escapes from the slab-edge filter would not fire.
#: Suppressed for population-AIF fallback by construction.
DCE_AIF_PEAK_RATIO_WARN_THRESHOLD: Final[float] = 1.5

#: Multi-bump tail detection. Fraction of first-pass peak above which a
#: post-peak local maximum is counted as a "bump." Parker plasma late
#: tail (post-60 s) sits at 25-30% of peak; threshold above that level
#: requires a true anomaly above Parker's predicted decay envelope.
DCE_AIF_TAIL_PEAK_FRACTION: Final[float] = 0.40

#: Time (seconds) to skip immediately after the first-pass peak before
#: searching for secondary maxima. Skips the physiologic recirculation
#: hump (~25-35 s post-peak per Parker 2006 second Gaussian) plus its
#: descending shoulder (~45-60 s). True late-tail anomalies (motion,
#: inflow, venous mixing) live beyond 60 s post-peak.
DCE_AIF_TAIL_SHOULDER_SECONDS: Final[float] = 60.0

#: Minimum time (seconds) between detected secondary peaks
#: (~1/3 of the typical recirculation period; ``scipy.signal.find_peaks``
#: ``distance`` parameter, converted to frames using AIF tr).
DCE_AIF_TAIL_MIN_SEPARATION_SECONDS: Final[float] = 10.0

# =============================================================================
# DCE AIF acceptance gates (G1-G7 + G6 escape)
# =============================================================================

#: G1 — scale_aif amplitude floor. Scaling factor above this threshold
#: triggers HARD reject; G6 escape allows up to G6_SCALE_CEILING.
#: Parker plasma peak ~10 mM at 1.5T full-dose Gd-DTPA (Parker 2006 MRM
#: 56:993); scale_factor is dimensionless against the dose-corrected
#: target so the threshold is field-strength and dose independent.
DCE_AIF_GATE_SCALE_HARD: Final[float] = 3.0

#: G6 (a) — escape ceiling for amplitude floor when shape gates pass.
#: 6x corresponds to partial-volume fill fraction of about 0.17 (1 mm
#: caliber artery in 3.5 mm slab, off-axis); below this, surrounding
#: tissue uptake bleed distorts AIF shape during first pass for
#: tumor-adjacent K^trans of about 0.05 min^-1.
DCE_AIF_GATE_SCALE_G6_CEILING: Final[float] = 6.0

#: G2 — late-tail/peak ratio HARD reject threshold.
#: Parker bi-exponential arterial plasma decays to <=0.20 by 2 min
#: post-peak; venous mixing produces persistent late-tail above 0.40
#: (Calamante 2003 NMR Biomed 16:467).
DCE_AIF_GATE_LATE_TAIL_RATIO_MAX: Final[float] = 0.35

#: G3 — FWHM HARD reject threshold (seconds).
#: Assumes power injection >=2 mL/s. For hand-injection cohorts,
#: override to ~28 s (Calamante 2003 NMR Biomed 16:467). Distal-ACA
#: broadens FWHM by 2-4 s relative to M1 (Calamante 2010 JCBFM 30:1393).
DCE_AIF_GATE_FWHM_MAX_SECONDS: Final[float] = 22.0

#: G4 — multi-bump tail HARD reject threshold (count).
#: Smooth Reynolds-regime arterial flow has exactly one secondary
#: maximum (recirculation hump at ~30 s post-peak); more than 2 bumps
#: in the late tail indicates motion, inflow, or venous mixing.
DCE_AIF_GATE_MULTIBUMP_MAX: Final[int] = 2

#: G5' — ratio_30 thresholds (post-peak frame at max(4, ceil(30/TR))).
#: Parker bi-exponential gives ratio ~0.32 at peak+30 s; venous AIFs
#: sit at >=0.55 (Bleeker 2009 MRM 62:96). HARD threshold leaves
#: headroom for distal-ACA broadening, Hct correction, and
#: recirculation pickup. WARN tier flags borderline cases for G7
#: aggregation.
DCE_AIF_GATE_RATIO30_HARD: Final[float] = 0.50
DCE_AIF_GATE_RATIO30_WARN: Final[float] = 0.40

#: G6 (g) — minimum voxel count and spatial consensus for amplitude-
#: floor escape. Bilateral n>=5 OR unilateral n>=10 with >=2 distinct
#: z-slabs provides independent-spatial-consensus rule-out of
#: localized artifact. n>=10 is the empirical reliability floor for
#: individual AIFs (Calamante 2010 JCBFM 30:1393).
DCE_AIF_GATE_G6_MIN_VOXELS_BILATERAL: Final[int] = 5
DCE_AIF_GATE_G6_MIN_VOXELS_UNILATERAL: Final[int] = 10
DCE_AIF_GATE_G6_MIN_SLABS_UNILATERAL: Final[int] = 2

#: G6-admitted reduced-confidence AIF: per-parameter PK posterior sigma
#: multiplier reflecting Tofts/2CXM AIF-peak sensitivity hierarchy
#: (Buckley 2002 MRM 47:601). ve is uninflated because ve fits late-
#: phase washout slope, decoupled from AIF peak amplitude. Workflow
#: base inflation=2.0 propagates per-parameter via this map; base=1.0
#: (standard AIF) yields all 1.0x regardless of map.
#:
#: kep is omitted because it is not a directly-fitted parameter; it is
#: derived downstream as Ktrans/ve and inherits Ktrans's inflation
#: through that derivation. Add explicit kep_std propagation here only
#: if the pipeline begins emitting a separate kep uncertainty map.
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

#: Cap on inflow exclusion as a fraction of total slab depth.
#: Guarantees ≥65% of slab survives bilaterally for AIF search even on
#: thick-slab acquisitions (e.g. 8 slices × 8mm = 64mm slab where a
#: naive 4×slice = 32mm exclusion would zero half the slab).
INFLOW_EXCLUSION_MAX_SLAB_FRACTION: Final[float] = 0.35

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


# =============================================================================
# DSC→DCE Cross-Modality vp Constraint Constants
# =============================================================================
# Governs how DSC-derived CBV maps into DCE vp priors via NAWM
# normalization instead of absolute-CBV conversion.

#: Maximum physiological vp value. Absolute upper bound on plasma volume
#: fraction in any brain voxel; a vp > 0.2 indicates pure-vessel or
#: catastrophic fit failure, not tissue. Used by ``ncbv_to_vp_prior`` and
#: ``cbv_to_vp`` to NaN-out out-of-range entries before they enter DCE
#: fitting. Centralized here alongside the other vp gates so all vp
#: thresholds live in one module.
VP_MAX_PHYSIOLOGICAL: Final[float] = 0.2

#: Physiological plasma volume fraction assumed for contralateral NAWM.
#: Used as the anchor for converting nCBV → vp prior in the DSC→DCE
#: cross-modality constraint. Source: Larsson et al. MRM 2009;62:1270
#: (direct DCE measurement, NAWM vp = 0.0089 ± 0.0038). Corroborated by
#: Sourbron & Buckley NMR Biomed 2013;26:1004 (WM vp 0.005–0.02 literature
#: range).
#:
#: Hct closure constraint: this anchor uses the same large-vessel Hct =
#: 0.42 convention as the AIF-path /(1-Hct) conversion at ``dce_aif.py``.
#: Mixing capillary-Hct and large-vessel-Hct across the AIF→tissue chain
#: violates plasma-fraction closure.
NAWM_VP_REFERENCE: Final[float] = 0.010

#: Physiological plausibility range for DCE vp prior (plasma volume fraction).
#: With vp = NAWM_VP_REFERENCE × nCBV and physiological whole-brain nCBV
#: medians ~0.4–3.0, vp_median is algebraically pinned near 0.005–0.03.
#: A scale error in nCBV or a corrupted NAWM mask breaches this range.
#: WM vp ≈ 0.005–0.015 (Larsson 2009), GM ≈ 0.015–0.025, tumor ≈ 0.02–0.05.
#: Factor-of-3 band around the NAWM anchor preserves the
#: gross-miscalibration semantics of the gate.
VP_MEDIAN_MIN: Final[float] = 0.005
VP_MEDIAN_MAX: Final[float] = 0.03

#: Acceptable tolerance on the post-DCE-fit vp / nCBV ratio, used by
#: cross-validation. Target is NAWM_VP_REFERENCE (0.010). Tolerance is a
#: factor-of-2 in either direction (log-symmetric) — the Hct ±20%, NAWM
#: CBV ±15%, DCE NLLS noise ±25%, σ=0.40 posterior spread ±15% per-component
#: RSS is ~40%, so factor-of-2 (±100%) is conservative for a
#: gross-miscalibration gate rather than a biological-scatter flag.
#: Values outside [0.005, 0.02] indicate the DCE posterior vp and DSC nCBV
#: don't agree within expected tolerance — flag as validation mismatch.
VP_NCBV_RATIO_MIN: Final[float] = 0.005
VP_NCBV_RATIO_MAX: Final[float] = 0.02

#: nCBV input range for cross-validation (per-voxel values to retain).
#: Upper bound 5.0 covers hypervascular HGG tumor rim (Law 2008, Hu 2009
#: report rCBV up to 6 in clinical GBM). Voxels beyond 5.0 are likely
#: vessels or partial-volume artifacts and are excluded from ratio stats.
MIN_NCBV_FOR_VALIDATION: Final[float] = 0.3
MAX_NCBV_FOR_VALIDATION: Final[float] = 5.0

#: Soft-prior sigma for the DCE vp constraint under NAWM-normalized nCBV.
#: Unified at 0.40 — the ~40–50% residual scale uncertainty is dominated
#: by Hct heterogeneity (~15% CoV) and capillary-vs-total-vessel volume
#: partitioning (~20%), which are biological and do not diminish with
#: better DSC SNR. Stratifying σ by DSC confidence confuses orthogonal
#: uncertainties; a single σ = 0.40 is the physical floor. The three
#: constants below are retained (vs collapsed) as hooks for a future
#: re-stratification review; all currently equal 0.40.
SIGMA_SOFT_DEFAULT: Final[float] = 0.40
SIGMA_SOFT_PRELOAD_SUPPRESSED: Final[float] = 0.40
SIGMA_SOFT_HIGH_K2: Final[float] = 0.40

#: Circular-detector std threshold when cross-validation uses nCBV.
#: Ratio vp/nCBV is centered at NAWM_VP_REFERENCE;
#: threshold ≈10% of that center separates "tight-SOFT prior-dominated"
#: from "informative DCE posterior" (wide SOFT).
#: Legacy threshold 0.01 stays for the non-nCBV path (vp/rel_cbv ratio scale).
CIRCULAR_STD_THRESHOLD_NCBV: Final[float] = 0.002
CIRCULAR_STD_THRESHOLD_ABS_CBV: Final[float] = 0.01

#: Overshoot fraction above which vp is prior-dominated and xval tolerance
#: is widened. Jensen-bias signature of first-pass temporal undersampling
#: (Buckley MRM 2002). Shared with the constraint-block detector in
#: ``dce_compute.py`` — single source of truth.
ACQUISITION_LIMITED_OVERSHOOT_THRESHOLD: Final[float] = 0.7

#: Widened lower edge of the vp/nCBV ratio band, applied only when the
#: ``acquisition_limited`` flag fires. Pinned at the brain-capillary
#: physiological floor (Pardridge 2011, vp ≥ 0.005) so it does not
#: undercut the magnitude-plausibility gate. Acquisition-limited CRLB band
#: from Buckley MRM 2002 first-pass temporal-undersampling analysis —
#: independent of ``NAWM_VP_REFERENCE`` position (physics-expert 2026-04-19).
#: Excursions below 0.005 remain a real data issue even in the
#: prior-dominated regime.
VP_NCBV_RATIO_MIN_LIMITED: Final[float] = 0.005

#: Widened upper edge of the vp/nCBV ratio band. Acquisition-limited CRLB
#: ceiling from Buckley MRM 2002 — independent of ``NAWM_VP_REFERENCE``.
#: The widened band brackets the prior-dominated NAWM posterior envelope
#: and intentionally spans up to the legacy magnitude-gate ceiling. Scope
#: is NAWM ratios — tumor-vp excursions are outside the xval domain.
VP_NCBV_RATIO_MAX_LIMITED: Final[float] = 0.06
