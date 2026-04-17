"""DSC preload detection and contrast agent residual estimation.

Automatic detection of contrast preload from prior DCE injection
and classification of residual contrast effects on DSC signal.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
    EPSILON_DELTA_R2,
    EPSILON_SIGNAL,
    K2_UPPER_BOUND,
    KEP_DSC_DEFAULT,
    KEP_DSC_LOWER,
    KEP_DSC_UPPER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Preload Classification
# =============================================================================


class PreloadClass(str, Enum):
    """Preload classification levels based on baseline ΔR2*.

    Classification based on residual contrast effects at 3T.
    """

    NONE = "none"
    """No detectable preload: baseline ΔR2* < 0.5 s⁻¹"""

    MINIMAL = "minimal"
    """Minimal preload: 0.5 ≤ baseline ΔR2* < 2.0 s⁻¹"""

    HALF_DOSE = "half_dose"
    """Half-dose preload: 2.0 ≤ baseline ΔR2* < 5.0 s⁻¹"""

    SIGNIFICANT = "significant"
    """Significant preload: 5.0 ≤ baseline ΔR2* < 10.0 s⁻¹"""

    SEVERE = "severe"
    """Severe preload: baseline ΔR2* ≥ 10.0 s⁻¹"""


class PreloadThresholds(NamedTuple):
    """Thresholds for preload classification by baseline ΔR2* (s⁻¹)."""

    none_max: float
    minimal_max: float
    half_dose_max: float
    significant_max: float


#: Default preload thresholds at 3T (s⁻¹)
PRELOAD_THRESHOLDS_3T: Final[PreloadThresholds] = PreloadThresholds(
    none_max=0.5,
    minimal_max=2.0,
    half_dose_max=5.0,
    significant_max=10.0,
)

#: Preload thresholds by field strength
PRELOAD_THRESHOLDS_BY_FIELD: Final[dict[float, PreloadThresholds]] = {
    1.5: PreloadThresholds(
        none_max=0.25,
        minimal_max=1.0,
        half_dose_max=2.5,
        significant_max=5.0,
    ),
    3.0: PRELOAD_THRESHOLDS_3T,
    7.0: PreloadThresholds(
        none_max=1.25,
        minimal_max=5.0,
        half_dose_max=12.5,
        significant_max=25.0,
    ),
}


#: Baseline agent standard dose (mmol Gd/kg) for which
#: ``PRELOAD_THRESHOLDS_BY_FIELD`` is calibrated. Used to linearly scale
#: thresholds for agents with different standard doses (e.g., Vueway 0.05).
_PRELOAD_THRESHOLDS_BASELINE_DOSE: Final[float] = 0.1


def get_preload_thresholds(
    field_strength: float = 3.0,
    agent_standard_dose: float = _PRELOAD_THRESHOLDS_BASELINE_DOSE,
) -> PreloadThresholds:
    """Get preload classification thresholds for field strength and agent.

    Thresholds are in ΔR2* units (s⁻¹) and were calibrated for standard
    Gd-DOTA-class agents (``agent_standard_dose = 0.1 mmol Gd/kg``). Agents
    with different standard doses (e.g., gadopiclenol/Vueway at 0.05,
    gadoquatrane at 0.04) produce systematically different ΔR2* per
    "fraction of standard dose." Tissue r2* itself is agent-independent
    (Gd³⁺ susceptibility), so linear dose-normalization is the correct
    correction: ``effective_threshold = base_threshold × (agent_standard_dose / 0.1)``.

    Args:
        field_strength: Magnetic field strength in Tesla. Uses nearest
            available field strength if exact match not in table.
        agent_standard_dose: Agent's recommended standard single dose in
            mmol Gd/kg. Default 0.1 (Gd-DOTA class). Thresholds scale
            linearly; values outside [0.01, 0.2] produce a warning.

    Returns:
        PreloadThresholds (in ΔR2* s⁻¹) scaled for the specified agent.
    """
    if agent_standard_dose <= 0:
        raise ValueError(
            f"agent_standard_dose must be > 0, got {agent_standard_dose}"
        )
    if not (0.01 <= agent_standard_dose <= 0.2):
        logger.warning(
            "agent_standard_dose=%.3f outside typical range [0.01, 0.2]; "
            "preload thresholds may be miscalibrated.",
            agent_standard_dose,
        )

    if field_strength in PRELOAD_THRESHOLDS_BY_FIELD:
        base = PRELOAD_THRESHOLDS_BY_FIELD[field_strength]
    else:
        fields = sorted(PRELOAD_THRESHOLDS_BY_FIELD.keys())
        nearest = min(fields, key=lambda x: abs(x - field_strength))
        base = PRELOAD_THRESHOLDS_BY_FIELD[nearest]

    scale = agent_standard_dose / _PRELOAD_THRESHOLDS_BASELINE_DOSE
    return PreloadThresholds(
        none_max=base.none_max * scale,
        minimal_max=base.minimal_max * scale,
        half_dose_max=base.half_dose_max * scale,
        significant_max=base.significant_max * scale,
    )


#: Administered-dose bin edges for user-specified ``preload`` float.
#: Clinically calibrated (ASFNR Welker 2015, Boxerman 2020):
#:   quarter-dose preload (0.25) → HALF_DOSE class
#:   half-dose preload (0.5)      → HALF_DOSE class
#:   full-dose preload (1.0)      → SIGNIFICANT class
#:   double-dose (>1.15)          → SEVERE class
#: Edges are heuristic for K2 uncertainty bucketing; the MINIMAL→HALF_DOSE
#: edge (0.20) and SIGNIFICANT→SEVERE edge (1.15) in particular are
#: conventions rather than literature-derived boundaries.
#:
#: Note: these edges describe ADMINISTERED preload dose fractions (what the
#: technician injected). Auto-detection from signal uses a separate set of
#: ΔR2*-based thresholds (PRELOAD_THRESHOLDS_BY_FIELD) which measure the
#: RESIDUAL-EQUIVALENT concentration at DSC start (what survived clearance).
#: The two edge tables are physically distinct and are not expected to
#: numerically align; they share the same class names for downstream
#: K2-uncertainty bucketing only.
_PRELOAD_DOSE_EDGES: Final[dict[PreloadClass, tuple[float, float]]] = {
    PreloadClass.NONE:         (0.00, 0.01),
    PreloadClass.MINIMAL:      (0.01, 0.20),
    PreloadClass.HALF_DOSE:    (0.20, 0.60),
    PreloadClass.SIGNIFICANT:  (0.60, 1.15),
    PreloadClass.SEVERE:       (1.15, float("inf")),
}


def preload_class_from_dose(preload_dose: float) -> PreloadClass:
    """Bin a user-specified administered preload dose fraction to a class.

    Bin edges are right-exclusive half-open intervals ``[lo, hi)``, matching
    :func:`classify_preload` and :func:`get_preload_thresholds` semantics.

    Args:
        preload_dose: Administered preload as a fraction of the agent's
            standard single dose (0.0 = no preload, 0.5 = half-dose,
            1.0 = full dose). Must be finite and non-negative.

    Returns:
        PreloadClass enum value for K2 uncertainty tuning.

    Raises:
        ValueError: If ``preload_dose`` is negative or non-finite.
    """
    if not np.isfinite(preload_dose) or preload_dose < 0:
        raise ValueError(
            f"preload_dose must be finite and non-negative, got {preload_dose}"
        )
    for cls, (lo, hi) in _PRELOAD_DOSE_EDGES.items():
        if lo <= preload_dose < hi:
            return cls
    # Should be unreachable since SEVERE has hi=inf, but guard anyway
    return PreloadClass.SEVERE


def residual_equivalent_fraction_from_delta_r2(
    delta_r2: float,
    field_strength: float = 3.0,
    agent_standard_dose: float = _PRELOAD_THRESHOLDS_BASELINE_DOSE,
) -> float:
    """Convert baseline ΔR2* to a residual-equivalent dose fraction.

    This is NOT the administered preload dose. It is the residual tissue
    Gd concentration at DSC start, expressed as a fraction of what the
    agent's standard dose would produce if fully retained. Actual
    administered preload is typically 1.5×–3× larger than this value
    because of renal clearance between preload and DSC bolus (5–10 min for
    ASFNR-style preload protocols; clearance depends on GFR).

    Used for QC transparency so auto-detection output can be compared to
    user-specified administered dose (different physical quantities, same
    class system for bucketing).

    Args:
        delta_r2: Baseline ΔR2* in s⁻¹ (from detect_preload_from_signal).
        field_strength: Magnetic field in Tesla.
        agent_standard_dose: Agent's standard dose in mmol Gd/kg.

    Returns:
        Residual-equivalent fraction. 0.0 if ``delta_r2 <= 0`` (no detectable
        preload signal — valid physical edge case).

    Raises:
        ValueError: If ``agent_standard_dose <= 0`` (consistent with
            :func:`get_preload_thresholds`).
    """
    if agent_standard_dose <= 0:
        raise ValueError(
            f"agent_standard_dose must be > 0, got {agent_standard_dose}"
        )
    if delta_r2 <= 0:
        return 0.0
    from pmriprep.utils.dsc_constants import get_r2_star

    r2_star = get_r2_star(field_strength)
    # mM residual = delta_r2 / r2* ; fraction = mM / agent_standard_dose
    # (with the convention that agent_standard_dose's mmol/kg ≈ mmol/L in
    # whole-brain tissue at unity partition coefficient — same
    # approximation as get_r2_star calibration).
    return float(delta_r2 / r2_star / agent_standard_dose)


# =============================================================================
# Preload Detection Result
# =============================================================================


@dataclass
class PreloadResult:
    """Result of preload detection analysis.

    Attributes:
        classification: Preload severity classification.
        baseline_delta_r2: Measured baseline ΔR2* in s⁻¹.
        estimated_residual: Estimated residual concentration in mM (if available).
        detection_method: Method used for detection.
        confidence: P(correct classification) in [0, 1]. Computed from the
            SEM of mean baseline signal propagated to ΔR2* units, compared
            to the distance to the nearest classification edge. NOT a
            measure of tissue heterogeneity. Assumes per-voxel noise is
            Gaussian and independent so that SEM_mean ≈ std(S) / √N
            (central limit theorem) — spatial correlation violates this
            and can inflate the reported confidence (minor; ±5% typical).
            0 when SEM is ill-defined (empty mask, zero baseline).
        details: Additional details about the detection. For signal-based
            detection includes `tissue_heterogeneity_cv`, `sem_delta_r2`,
            `distance_to_nearest_edge`, `effective_thresholds`,
            `residual_equivalent_fraction`, among others.
    """

    classification: PreloadClass
    baseline_delta_r2: float
    estimated_residual: Optional[float] = None
    detection_method: str = "baseline_delta_r2"
    confidence: float = 1.0
    details: Optional[dict[str, Any]] = None

    @property
    def is_preloaded(self) -> bool:
        """Check if any preload is detected."""
        return self.classification != PreloadClass.NONE

    @property
    def requires_correction(self) -> bool:
        """Check if preload is significant enough to require correction."""
        return self.classification in (
            PreloadClass.HALF_DOSE,
            PreloadClass.SIGNIFICANT,
            PreloadClass.SEVERE,
        )


# =============================================================================
# Preload Detection Functions
# =============================================================================


def classify_preload(
    baseline_delta_r2: float,
    field_strength: float = 3.0,
    agent_standard_dose: float = _PRELOAD_THRESHOLDS_BASELINE_DOSE,
) -> PreloadClass:
    """Classify preload severity from baseline ΔR2* (agent-aware).

    Thresholds scale linearly with ``agent_standard_dose`` — see
    :func:`get_preload_thresholds` for rationale.

    Args:
        baseline_delta_r2: Baseline ΔR2* value in s⁻¹.
        field_strength: Magnetic field strength in Tesla.
        agent_standard_dose: Agent's standard dose in mmol Gd/kg
            (default 0.1 for Gd-DOTA class).

    Returns:
        PreloadClass indicating severity level.
    """
    thresholds = get_preload_thresholds(field_strength, agent_standard_dose)

    if baseline_delta_r2 < thresholds.none_max:
        return PreloadClass.NONE
    elif baseline_delta_r2 < thresholds.minimal_max:
        return PreloadClass.MINIMAL
    elif baseline_delta_r2 < thresholds.half_dose_max:
        return PreloadClass.HALF_DOSE
    elif baseline_delta_r2 < thresholds.significant_max:
        return PreloadClass.SIGNIFICANT
    else:
        return PreloadClass.SEVERE


def detect_preload_from_signal(
    signal: npt.NDArray[np.floating],
    te: float,
    baseline_frames: int = 10,
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    field_strength: float = 3.0,
    reference_signal: Optional[float] = None,
    agent_standard_dose: float = _PRELOAD_THRESHOLDS_BASELINE_DOSE,
) -> PreloadResult:
    """Detect preload from DSC signal baseline.

    Analyzes the baseline signal before bolus arrival to estimate
    residual contrast effects from prior DCE injection.

    Method:
        1. Compute mean signal over baseline frames
        2. Compare to expected non-preloaded signal (or use reference)
        3. Estimate baseline ΔR2* = -ln(S_baseline/S_reference) / TE
        4. Classify based on ΔR2* thresholds

    Args:
        signal: 4D DSC signal array (x, y, z, time).
        te: Echo time in seconds.
        baseline_frames: Number of pre-bolus frames to analyze.
        brain_mask: Optional 3D binary mask for brain voxels.
        field_strength: Magnetic field strength in Tesla.
        reference_signal: Expected signal without preload. If None,
            uses the 95th percentile of baseline signal as reference
            (assuming some voxels have minimal preload effect).

    Returns:
        PreloadResult with classification and metrics.

    Raises:
        ValueError: If signal is not 4D.
        ValueError: If TE is not positive.
    """
    # Validate inputs
    if signal.ndim != 4:
        raise ValueError(f"Signal must be 4D, got {signal.ndim}D")

    if te <= 0:
        raise ValueError(f"TE must be positive, got {te}")

    n_timepoints = signal.shape[3]
    if baseline_frames >= n_timepoints:
        raise ValueError(
            f"baseline_frames ({baseline_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    # Compute baseline signal
    baseline_signal = np.mean(signal[..., :baseline_frames], axis=-1)

    # Apply brain mask if provided
    if brain_mask is not None:
        baseline_values = baseline_signal[brain_mask]
    else:
        # Use non-zero voxels as proxy for brain
        nonzero_mask = baseline_signal > EPSILON_SIGNAL
        baseline_values = baseline_signal[nonzero_mask]

    if baseline_values.size == 0:
        # No valid voxels, assume no preload
        return PreloadResult(
            classification=PreloadClass.NONE,
            baseline_delta_r2=0.0,
            detection_method="baseline_delta_r2",
            confidence=0.0,
            details={"error": "No valid brain voxels found"},
        )

    # Determine reference signal
    if reference_signal is None:
        # Use 95th percentile as reference (least affected by preload)
        reference_signal = float(np.percentile(baseline_values, 95))

    # Compute mean baseline signal in brain
    mean_baseline = float(np.mean(baseline_values))

    # Compute baseline ΔR2*
    # ΔR2* = -ln(S/S0) / TE
    # Preload causes signal drop, so S < S0
    signal_ratio = mean_baseline / max(reference_signal, EPSILON_SIGNAL)
    signal_ratio_clipped = np.clip(signal_ratio, EPSILON_SIGNAL, 1.0)

    baseline_delta_r2 = -float(np.log(signal_ratio_clipped)) / te
    baseline_delta_r2 = max(baseline_delta_r2, 0.0)  # Can't be negative

    # Classify preload (agent-aware thresholds)
    classification = classify_preload(
        baseline_delta_r2, field_strength, agent_standard_dose,
    )
    effective_thresholds = get_preload_thresholds(
        field_strength, agent_standard_dose,
    )

    # Estimate residual concentration if r2* available
    from pmriprep.utils.dsc_constants import get_r2_star

    r2_star = get_r2_star(field_strength)
    estimated_residual = baseline_delta_r2 / r2_star if baseline_delta_r2 > 0 else None

    # Residual-equivalent fraction of standard dose (for QC symmetry with
    # user-specified administered dose — different physical quantities, see
    # docstring of residual_equivalent_fraction_from_delta_r2).
    residual_equivalent_fraction = residual_equivalent_fraction_from_delta_r2(
        baseline_delta_r2, field_strength, agent_standard_dose,
    )

    # Legacy CV retained as a separate QC detail — measures anatomical
    # tissue heterogeneity (WM/GM/CSF T2* spread), NOT detection uncertainty.
    # Renamed ("signal_cv" → "tissue_heterogeneity_cv") so consumers don't
    # confuse it with measurement confidence.
    tissue_heterogeneity_cv = (
        float(np.std(baseline_values, ddof=1) / mean_baseline)
        if mean_baseline > 0 and baseline_values.size >= 2
        else float("inf")
    )

    # Confidence = P(ΔR2* truly lies in the reported classification bin).
    # Error propagation: ΔR2* = -ln(S/S_ref) / TE, so
    #   d(ΔR2*)/dS = -1 / (TE × S)  → SEM_ΔR2* ≈ SEM_S / (TE × mean_baseline)
    # S_ref (p95 of same baseline) treated as fixed — simplification with
    # <10% confidence bias for typical brain masks. Assumes voxels are
    # independent Gaussian samples; spatial correlation inflates reported
    # confidence by a small amount at typical SNR (DSC baseline >30).
    n_vox = int(baseline_values.size)
    if mean_baseline <= 0 or n_vox < 2:
        confidence = 0.0
        sem_delta_r2 = float("inf")
        distance_to_nearest_edge = 0.0
    else:
        sem_s = float(np.std(baseline_values, ddof=1)) / np.sqrt(n_vox)
        sem_delta_r2 = sem_s / (te * mean_baseline)
        edges = (
            effective_thresholds.none_max,
            effective_thresholds.minimal_max,
            effective_thresholds.half_dose_max,
            effective_thresholds.significant_max,
        )
        distance_to_nearest_edge = float(
            min(abs(baseline_delta_r2 - e) for e in edges)
        )
        if not np.isfinite(sem_delta_r2) or sem_delta_r2 <= 0.0:
            confidence = 0.0
        else:
            z = distance_to_nearest_edge / sem_delta_r2
            # Two-tailed: P(|true - measured| < distance | SEM) under normal.
            confidence = float(
                max(0.0, min(1.0, 1.0 - 2.0 * norm.cdf(-z)))
            )

    logger.debug(
        "preload confidence: z-score components — "
        "sem_delta_r2=%.4f, distance_to_nearest_edge=%.4f, "
        "baseline_delta_r2=%.4f, classification=%s, confidence=%.4f",
        sem_delta_r2, distance_to_nearest_edge, baseline_delta_r2,
        classification.value, confidence,
    )

    return PreloadResult(
        classification=classification,
        baseline_delta_r2=baseline_delta_r2,
        estimated_residual=estimated_residual,
        detection_method="baseline_delta_r2",
        confidence=confidence,
        details={
            "mean_baseline_signal": mean_baseline,
            "reference_signal": reference_signal,
            "signal_ratio": float(signal_ratio),
            "tissue_heterogeneity_cv": tissue_heterogeneity_cv,
            "sem_delta_r2": sem_delta_r2,
            "distance_to_nearest_edge": distance_to_nearest_edge,
            "field_strength": field_strength,
            "n_brain_voxels": int(baseline_values.size),
            "agent_standard_dose": agent_standard_dose,
            "effective_thresholds": {
                "none_max": effective_thresholds.none_max,
                "minimal_max": effective_thresholds.minimal_max,
                "half_dose_max": effective_thresholds.half_dose_max,
                "significant_max": effective_thresholds.significant_max,
            },
            "residual_equivalent_fraction": residual_equivalent_fraction,
        },
    )


def detect_preload_from_dce_protocol(
    dce_ran: bool,
    dce_end_time: Optional[float] = None,
    dsc_start_time: Optional[float] = None,
    dce_dose: float = 1.0,
) -> PreloadResult:
    """Detect preload from DCE protocol information.

    Uses knowledge of prior DCE acquisition to estimate preload.

    Args:
        dce_ran: Whether DCE was acquired before DSC.
        dce_end_time: Time when DCE acquisition ended (seconds from session start).
        dsc_start_time: Time when DSC acquisition started.
        dce_dose: Dose multiplier (0.5=half, 1.0=full, 2.0=double).

    Returns:
        PreloadResult with estimated preload from protocol.
    """
    if not dce_ran:
        return PreloadResult(
            classification=PreloadClass.NONE,
            baseline_delta_r2=0.0,
            detection_method="dce_protocol",
            confidence=1.0,
            details={"dce_ran": False},
        )

    # Estimate preload based on dose and timing
    # Reference: typical contrast clearance kinetics
    dose_factor = float(dce_dose)

    # Calculate time elapsed since DCE
    elapsed_time: Optional[float] = None
    if dce_end_time is not None and dsc_start_time is not None:
        elapsed_time = dsc_start_time - dce_end_time

    # Estimate baseline ΔR2* based on dose and clearance
    # Using approximate clearance model: C(t) = C0 * exp(-t/tau)
    # tau ~ 20 min for healthy subjects, shorter for tumors
    clearance_tau = 1200.0  # 20 minutes in seconds

    # Base ΔR2* for full dose immediately after DCE (3T)
    base_delta_r2 = 5.0  # s⁻¹, half-dose equivalent

    if elapsed_time is not None and elapsed_time > 0:
        # Apply clearance decay
        decay_factor = float(np.exp(-elapsed_time / clearance_tau))
    else:
        # Assume minimal clearance (just finished DCE)
        decay_factor = 0.9

    estimated_delta_r2 = base_delta_r2 * dose_factor * decay_factor

    classification = classify_preload(estimated_delta_r2, 3.0)

    return PreloadResult(
        classification=classification,
        baseline_delta_r2=estimated_delta_r2,
        detection_method="dce_protocol",
        confidence=0.7,  # Lower confidence than direct measurement
        details={
            "dce_ran": True,
            "dce_dose": dce_dose,
            "elapsed_time": elapsed_time,
            "dose_factor": dose_factor,
            "decay_factor": decay_factor,
        },
    )


def detect_preload_from_dicom(
    dicom_metadata: dict[str, Any],
) -> PreloadResult:
    """Detect preload from DICOM metadata.

    Looks for contrast agent administration information in DICOM tags.

    Args:
        dicom_metadata: Dictionary of DICOM metadata fields.

    Returns:
        PreloadResult based on DICOM information.
    """
    # Look for contrast administration tags
    contrast_bolus_agent = dicom_metadata.get("ContrastBolusAgent", "")
    contrast_bolus_volume = dicom_metadata.get("ContrastBolusVolume", None)
    contrast_bolus_start_time = dicom_metadata.get("ContrastBolusStartTime", None)

    # Check if this is a contrast-enhanced sequence
    is_contrast = bool(contrast_bolus_agent) or dicom_metadata.get(
        "ContrastBolusAgentAdministered", False
    )

    if not is_contrast:
        return PreloadResult(
            classification=PreloadClass.NONE,
            baseline_delta_r2=0.0,
            detection_method="dicom_metadata",
            confidence=0.5,  # Uncertain - absence of tag doesn't guarantee no contrast
            details={"contrast_info_found": False},
        )

    # Try to estimate dose from volume
    # Standard: 0.1 mmol/kg, ~7-10 mL for typical adult
    if contrast_bolus_volume is not None:
        try:
            volume_ml = float(contrast_bolus_volume)
            # Estimate dose category
            if volume_ml < 5:
                dose = "half"
                estimated_delta_r2 = 2.5
            elif volume_ml < 15:
                dose = "full"
                estimated_delta_r2 = 5.0
            else:
                dose = "double"
                estimated_delta_r2 = 10.0
        except (ValueError, TypeError):
            dose = "unknown"
            estimated_delta_r2 = 5.0  # Assume full dose
    else:
        dose = "unknown"
        estimated_delta_r2 = 5.0

    classification = classify_preload(estimated_delta_r2, 3.0)

    return PreloadResult(
        classification=classification,
        baseline_delta_r2=estimated_delta_r2,
        detection_method="dicom_metadata",
        confidence=0.6,
        details={
            "contrast_agent": contrast_bolus_agent,
            "volume_ml": contrast_bolus_volume,
            "estimated_dose": dose,
        },
    )


def detect_preload(
    signal: Optional[npt.NDArray[np.floating]] = None,
    te: Optional[float] = None,
    baseline_frames: int = 10,
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    field_strength: float = 3.0,
    dce_ran: Optional[bool] = None,
    dce_metadata: Optional[dict[str, Any]] = None,
    dicom_metadata: Optional[dict[str, Any]] = None,
    agent_standard_dose: float = _PRELOAD_THRESHOLDS_BASELINE_DOSE,
) -> PreloadResult:
    """Detect preload using all available information.

    Combines multiple detection methods and returns the most confident result:
    1. Direct signal measurement (highest confidence)
    2. DCE protocol information
    3. DICOM metadata

    Args:
        signal: 4D DSC signal array (x, y, z, time). Optional.
        te: Echo time in seconds. Required if signal provided.
        baseline_frames: Number of pre-bolus frames for signal analysis.
        brain_mask: Optional 3D binary mask for brain voxels.
        field_strength: Magnetic field strength in Tesla.
        dce_ran: Whether DCE was acquired before DSC. Optional.
        dce_metadata: DCE protocol information (dose, timing). Optional.
        dicom_metadata: DICOM metadata dictionary. Optional.
        agent_standard_dose: Agent's standard dose in mmol Gd/kg
            (default 0.1 for Gd-DOTA class). Forwarded to the signal-based
            detector so classification thresholds scale linearly with the
            administered agent's standard dose.

    Returns:
        PreloadResult from the most reliable detection method.
    """
    results: list[PreloadResult] = []

    # Method 1: Direct signal measurement (most reliable)
    if signal is not None and te is not None:
        result = detect_preload_from_signal(
            signal=signal,
            te=te,
            baseline_frames=baseline_frames,
            brain_mask=brain_mask,
            field_strength=field_strength,
            agent_standard_dose=agent_standard_dose,
        )
        results.append(result)

    # Method 2: DCE protocol
    if dce_ran is not None:
        dce_info = dce_metadata or {}
        result = detect_preload_from_dce_protocol(
            dce_ran=dce_ran,
            dce_end_time=dce_info.get("end_time"),
            dsc_start_time=dce_info.get("dsc_start_time"),
            dce_dose=float(dce_info.get("dose", 1.0)),
        )
        results.append(result)

    # Method 3: DICOM metadata
    if dicom_metadata is not None:
        result = detect_preload_from_dicom(dicom_metadata)
        results.append(result)

    # Return most confident result
    if not results:
        # No detection methods available, assume no preload
        return PreloadResult(
            classification=PreloadClass.NONE,
            baseline_delta_r2=0.0,
            detection_method="none",
            confidence=0.0,
            details={"error": "No detection methods available"},
        )

    # Sort by confidence, return highest.
    # Note: confidence scales differ across methods — signal-based is now
    # probabilistic P(correct classification) via SEM+z-score (this function
    # post-v0.25.0), while DCE-protocol and DICOM-metadata methods return
    # heuristic values (~0.5–0.7). A low-confidence signal detection
    # (near-edge ΔR2* or tiny mask) may defer to DCE timing, which is
    # usually acceptable — but confidence values are not directly comparable
    # across methods. Consider sorting by detection_method priority first
    # if this causes issues in practice.
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results[0]


# =============================================================================
# K2 Leakage Correction (BSW Method)
# =============================================================================


@dataclass
class K2CorrectionResult:
    """Result of K2 leakage correction.

    The K2 parameter quantifies contrast agent leakage through
    the blood-brain barrier (BBB). Based on the Boxerman-Schmainda-Weisskoff
    (BSW) model for leakage correction in DSC-MRI.

    Attributes:
        concentration_corrected: Leakage-corrected concentration curves (4D).
        k2_map: K2 parameter map (3D). Units: min⁻¹ or dimensionless rate.
        auc_tissue: Area under tissue curve before correction (3D).
        correction_applied: Boolean indicating if correction was applied.
        qc_metrics: QC metrics from correction.
        method: Leakage correction method ("bsw" or "bidirectional").
        kep_used: kep value used for bidirectional correction (s⁻¹), or None.
    """

    concentration_corrected: npt.NDArray[np.floating]
    k2_map: npt.NDArray[np.floating]
    auc_tissue: npt.NDArray[np.floating]
    correction_applied: bool = True
    qc_metrics: Optional[dict[str, Any]] = None
    method: str = "bsw"
    kep_used: Optional[float] = None


def _compute_mean_brain_reference(
    concentration: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]],
) -> Optional[npt.NDArray[np.floating]]:
    """Compute mean brain reference curve C̄(t) per Boxerman 2006.

    Averages concentration across non-enhancing brain voxels.
    Excludes voxels with negative mean post-bolus concentration
    (T1 leakage dominance) if enough non-enhancing voxels remain.

    Args:
        concentration: 4D concentration array (x, y, z, time).
        brain_mask: Optional 3D brain mask. If None, uses voxels
            with max concentration above noise floor.

    Returns:
        1D array (n_timepoints,) or None if insufficient voxels.
    """
    _MIN_VOXELS = 10
    _WARN_VOXELS = 100

    # Select brain voxels
    if brain_mask is not None:
        valid_indices = np.where(brain_mask)
    else:
        max_conc = np.max(concentration, axis=-1)
        valid_indices = np.where(max_conc > EPSILON_CONCENTRATION)

    n_brain = len(valid_indices[0])
    if n_brain < _MIN_VOXELS:
        logger.warning(
            "Insufficient brain voxels for mean reference: %d (need ≥%d)",
            n_brain, _MIN_VOXELS,
        )
        return None

    # Extract brain voxel time courses: (n_voxels, n_timepoints)
    brain_curves = concentration[valid_indices[0], valid_indices[1], valid_indices[2], :]

    # Exclude voxels with negative mean in last 50% of timepoints
    # (T1 leakage dominance — these would bias C̄ downward)
    n_tp = brain_curves.shape[1]
    post_half = brain_curves[:, n_tp // 2:]
    post_mean = np.mean(post_half, axis=1)
    non_leaky = post_mean >= 0

    n_excluded = int(np.sum(~non_leaky))
    n_remaining = int(np.sum(non_leaky))

    if n_remaining >= _MIN_VOXELS:
        brain_curves = brain_curves[non_leaky]
    else:
        # Not enough non-leaky voxels — fall back to ALL brain voxels
        logger.info(
            "Only %d non-leaky voxels (need ≥%d), using all %d brain voxels",
            n_remaining, _MIN_VOXELS, n_brain,
        )
        n_excluded = 0

    c_bar = np.mean(brain_curves, axis=0)

    n_used = len(brain_curves)
    logger.info(
        "Mean brain reference: %d voxels used, %d excluded (T1 leakage), "
        "peak C̄ = %.3f",
        n_used, n_excluded, float(np.max(c_bar)),
    )
    if n_used < _WARN_VOXELS:
        logger.warning(
            "Mean brain reference from only %d voxels (<100 recommended "
            "per Bjornerud 2011); consider checking brain mask quality",
            n_used,
        )

    return np.asarray(c_bar, dtype=np.float64)


def estimate_k2_per_voxel(
    concentration: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    tr: float = 1.5,
) -> npt.NDArray[np.floating]:
    """Estimate K2 leakage parameter per voxel using BSW model.

    K2 represents the rate of contrast agent leakage through the BBB.
    Estimated using the Boxerman-Schmainda-Weisskoff model with a mean
    brain reference curve C̄(t) (Boxerman et al., AJNR 2006):

        C_voxel(t) = K1 × C̄(t) + K2_reg × ∫₀ᵗ C̄(τ)dτ

    K1 captures the instantaneous susceptibility scaling relative to
    the mean brain, and K2_reg captures T1-shortening leakage.

    Sign convention:
        - T1-dominant leakage *reduces* observed ΔR2* → K2_reg < 0
          → K2_physical = -K2_reg > 0
        - T2*-dominant leakage *increases* observed ΔR2* → K2_reg > 0
          → clipped to K2_physical = 0 (conservative: no correction
          rather than wrong-sign correction)

    Args:
        concentration: 4D concentration array (x, y, z, time).
        brain_mask: Optional 3D brain mask.
        tr: Repetition time in seconds for integration.

    Returns:
        3D K2 map with leakage rate per voxel (s⁻¹).

    Notes:
        - K2 ≈ 0: Intact blood-brain barrier
        - K2 > 0: BBB disruption (T1 leakage)
        - Uses vectorized OLS for efficiency

    References:
        Boxerman JL, Schmainda KM, Weisskoff RM. AJNR 2006;27:859-867.
    """
    if concentration.ndim != 4:
        raise ValueError(f"Concentration must be 4D, got {concentration.ndim}D")

    n_timepoints = concentration.shape[3]
    k2_map = np.zeros(concentration.shape[:3], dtype=np.float64)

    # Compute mean brain reference curve C̄(t)
    c_bar = _compute_mean_brain_reference(concentration, brain_mask)
    if c_bar is None:
        return k2_map

    # Derive regression start from C̄ peak (2 frames after peak)
    regression_start = int(np.argmax(c_bar)) + 2
    if regression_start >= n_timepoints - 2:
        logger.warning(
            "Mean reference peak at frame %d of %d — too late for regression",
            regression_start - 2, n_timepoints,
        )
        return k2_map

    # Build design matrix ONCE (same for all voxels)
    # c_bar_post: instantaneous reference (sliced from regression_start)
    c_bar_post = c_bar[regression_start:]
    # cumsum_c_bar: integral from t=0, preserving first-pass leakage contribution
    cumsum_c_bar = np.cumsum(c_bar) * tr
    # Then sliced — different time origin from c_bar_post by design
    cumsum_c_bar_post = cumsum_c_bar[regression_start:]

    X = np.column_stack([c_bar_post, cumsum_c_bar_post])  # (n_post, 2)

    # Condition check on shared design matrix
    if np.linalg.cond(X) > 1e8:
        logger.warning("Mean brain reference design matrix ill-conditioned")
        return k2_map

    # Determine valid voxels
    if brain_mask is not None:
        valid_mask = brain_mask
    else:
        max_conc = np.max(concentration, axis=-1)
        valid_mask = max_conc > EPSILON_CONCENTRATION

    valid_indices = np.where(valid_mask)
    n_voxels = len(valid_indices[0])

    if n_voxels == 0:
        return k2_map

    if n_voxels > 1_000_000:
        logger.warning(
            "K2 estimation on %d voxels (~%.0f MB); consider tighter brain mask",
            n_voxels, n_voxels * (n_timepoints - regression_start) * 8 / 1e6,
        )

    # Vectorized solve: extract all valid voxel time courses
    Y = concentration[
        valid_indices[0], valid_indices[1], valid_indices[2], regression_start:
    ]  # (n_voxels, n_post)

    # lstsq: X @ coeffs = Y.T → coeffs shape (2, n_voxels)
    coeffs, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)

    # NEGATE: K2_physical = -K2_regression (T1 leakage gives negative K2_reg)
    # Clip: only correct T1-dominant leakage (positive K2_phys)
    k2_values = np.clip(-coeffs[1, :], 0.0, K2_UPPER_BOUND)

    k2_map[valid_indices] = k2_values

    return k2_map


def apply_k2_correction(
    concentration: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    tr: float = 1.5,
    k2_map: Optional[npt.NDArray[np.floating]] = None,
    baseline_frames: int = 5,
) -> K2CorrectionResult:
    """Apply K2 leakage correction using the BSW method with mean brain reference.

    Implements the Boxerman-Schmainda-Weisskoff (BSW) leakage correction
    for DSC-MRI using a mean brain reference curve C̄(t) rather than AIF
    (Boxerman et al., AJNR 2006):

        C_corrected(t) = C_tissue(t) + K2 × ∫₀ᵗ C̄(τ)dτ

    K2 is estimated from the reference scale, so correction must use the
    same reference. Using AIF would produce wrong magnitudes (AIF >> C̄).

    Baseline subtraction is applied before K2 estimation to ensure the
    BSW model assumption C(t)=0 at baseline is satisfied. This is critical
    for preloaded acquisitions where residual Gd creates a DC offset.

    Args:
        concentration: 4D concentration array (x, y, z, time).
        brain_mask: Optional 3D brain mask for voxel selection.
        tr: Repetition time in seconds for integration.
        k2_map: Pre-computed K2 map. If None, K2 is estimated per voxel.
        baseline_frames: Number of pre-bolus baseline frames for
            baseline subtraction. Default: 5.

    Returns:
        K2CorrectionResult with corrected concentration and K2 map.

    References:
        Boxerman JL, Schmainda KM, Weisskoff RM. AJNR 2006;27(4):859-867.
    """
    if concentration.ndim != 4:
        raise ValueError(f"Concentration must be 4D, got {concentration.ndim}D")

    # Baseline-subtract before K2 estimation. The BSW model assumes
    # C(t)=0 at baseline; non-zero offset biases the cumulative integral
    # in K2 regression. For non-preloaded data, baseline ≈ 0 (no-op).
    # Ref: Ostergaard MRM 1996, Boxerman AJNR 2006.
    n_bl = max(baseline_frames, 4)
    conc_bl = np.mean(concentration[..., :n_bl], axis=-1, keepdims=True)
    concentration = concentration - conc_bl

    # Compute AUC before any early returns (needed for QC regardless)
    cumsum_tissue = np.cumsum(concentration, axis=-1) * tr
    auc_tissue = cumsum_tissue[..., -1].copy()

    # Estimate K2 if not provided
    if k2_map is None:
        k2_map = estimate_k2_per_voxel(
            concentration=concentration,
            brain_mask=brain_mask,
            tr=tr,
        )

    # Gate K2: voxels with negligible leakage (intact BBB) are left uncorrected
    K2_THRESHOLD = 1e-4  # s⁻¹
    k2_map = np.where(k2_map > K2_THRESHOLD, k2_map, 0.0)

    # Compute mean brain reference for correction
    c_bar = _compute_mean_brain_reference(concentration, brain_mask)
    if c_bar is None:
        # No valid brain reference — skip correction entirely.
        # Using AIF as fallback would be wrong: AIF >> C̄, so K2 × ∫AIF
        # would overcorrect by 5-10x. No correction is safer.
        logger.warning(
            "K2 correction skipped: insufficient brain voxels for mean reference"
        )
        return K2CorrectionResult(
            concentration_corrected=concentration.copy(),
            k2_map=k2_map,
            auc_tissue=auc_tissue,
            correction_applied=False,
            method="bsw",
        )

    # Compute cumulative integral of mean brain reference
    cumsum_ref = np.cumsum(c_bar) * tr
    cumsum_ref_4d = cumsum_ref[np.newaxis, np.newaxis, np.newaxis, :]

    # Expand K2 map to match time dimension for broadcasting
    k2_expanded = k2_map[..., np.newaxis]

    # Apply correction: C_corrected(t) = C_tissue(t) + K2 × ∫₀ᵗ C̄(τ)dτ
    # T1 leakage reduces apparent ΔR2*, so we ADD back the leaked component.
    concentration_corrected = concentration + k2_expanded * cumsum_ref_4d

    # Allow negative values after baseline subtraction — symmetric noise
    # around zero preserves the mean and avoids positive bias from clipping.

    # Compute QC metrics
    qc_metrics = _compute_k2_qc_metrics(k2_map, brain_mask, auc_tissue)
    qc_metrics["k2_method"] = "bsw"

    return K2CorrectionResult(
        concentration_corrected=concentration_corrected,
        k2_map=k2_map,
        auc_tissue=auc_tissue,
        correction_applied=True,
        qc_metrics=qc_metrics,
        method="bsw",
    )


# K2 QC thresholds (s⁻¹)
# Boxerman et al. AJNR 2006;27:859-867; Schmainda et al. Neuro Oncol 2015
_K2_CLINICAL_THRESHOLD: Final[float] = 0.04  # clinically meaningful BBB disruption
_K2_LEAKY_THRESHOLD: Final[float] = 0.1  # used by dsc_cbv_constraint.py
_K2_SEVERE_THRESHOLD: Final[float] = 0.10  # physiological maximum (Boxerman 2006)


def _compute_k2_qc_metrics(
    k2_map: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]],
    auc_tissue: npt.NDArray[np.floating],
) -> dict[str, Any]:
    """Compute QC metrics for K2 leakage correction.

    Args:
        k2_map: 3D K2 parameter map.
        brain_mask: Optional 3D brain mask.
        auc_tissue: 3D area under tissue curve.

    Returns:
        Dictionary of QC metrics.
    """
    # Select values within brain
    if brain_mask is not None:
        k2_values = k2_map[brain_mask]
        auc_values = auc_tissue[brain_mask]
    else:
        k2_values = k2_map[k2_map > 0]
        auc_values = auc_tissue[auc_tissue > 0]

    if k2_values.size == 0:
        return {
            "k2_mean": 0.0,
            "k2_std": 0.0,
            "k2_max": 0.0,
            "k2_median": 0.0,
            "percent_leaky_voxels": 0.0,
            "percent_severe_leakage": 0.0,
            "percent_above_clinical_max": 0.0,
        }

    # K2 statistics
    k2_mean = float(np.mean(k2_values))
    k2_std = float(np.std(k2_values))
    k2_max = float(np.max(k2_values))
    k2_median = float(np.median(k2_values))

    # Fraction of voxels with BBB disruption (consumed by dsc_cbv_constraint.py)
    n_leaky = int(np.sum(k2_values > _K2_LEAKY_THRESHOLD))
    percent_leaky = 100.0 * n_leaky / k2_values.size

    # Fraction with severe leakage (numerically suspect fits)
    n_severe = int(np.sum(k2_values > _K2_SEVERE_THRESHOLD))
    percent_severe = 100.0 * n_severe / k2_values.size

    # Fraction above clinical maximum (Boxerman 2006)
    n_clinical = int(np.sum(k2_values > _K2_CLINICAL_THRESHOLD))
    percent_clinical = 100.0 * n_clinical / k2_values.size

    return {
        "k2_mean": k2_mean,
        "k2_std": k2_std,
        "k2_max": k2_max,
        "k2_median": k2_median,
        "percent_leaky_voxels": percent_leaky,
        "percent_severe_leakage": percent_severe,
        "percent_above_clinical_max": percent_clinical,
        "n_brain_voxels": int(k2_values.size),
        "auc_mean": float(np.mean(auc_values)) if auc_values.size > 0 else 0.0,
    }


# =============================================================================
# Bidirectional K2 Leakage Correction (Leu et al. 2016)
# =============================================================================


def _compute_ees_concentration(
    c_bar: npt.NDArray[np.floating],
    kep: float,
    tr: float,
) -> npt.NDArray[np.floating]:
    """Compute instantaneous EES concentration with washout.

    Convolves the mean brain reference C̄(t) with an exponential washout
    kernel to model contrast agent efflux from the extravascular
    extracellular space (EES).

    C_e(t) = C̄(t) ⊗ exp(-kep × t)

    where ⊗ denotes convolution and kep is the EES efflux rate constant.

    When kep → 0, exp(-kep × t) → 1, and C_e(t) → cumsum(C̄) × TR
    (approaches standard BSW cumulative integral).

    Args:
        c_bar: Mean brain reference curve, shape (n_timepoints,).
        kep: EES efflux rate constant in s⁻¹.
        tr: Repetition time in seconds.

    Returns:
        Instantaneous EES concentration C_e(t), shape (n_timepoints,).

    References:
        Leu K et al. AJNR 2016;37(8):1440-1446. doi:10.3174/ajnr.A4759
    """
    n_tp = len(c_bar)
    time = np.arange(n_tp) * tr

    # Exponential washout kernel
    exp_kernel = np.exp(-kep * time)

    # Convolve and scale by TR (rectangular integration)
    c_e = np.convolve(c_bar, exp_kernel, mode="full")[:n_tp] * tr

    return np.asarray(c_e, dtype=np.float64)


def estimate_k2_bidirectional(
    concentration: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    tr: float = 1.5,
    kep: float = KEP_DSC_DEFAULT,
) -> npt.NDArray[np.floating]:
    """Estimate K2 leakage parameter using bidirectional model.

    Extends the BSW model to account for contrast agent washout from
    the EES using a two-regressor design matrix:

        C_voxel(t) = K1 × C̄(t) + K2_reg × C_e(t)

    where C_e(t) = C̄(t) ⊗ exp(-kep × t) is the instantaneous EES
    concentration accounting for washout.

    Args:
        concentration: 4D concentration array (x, y, z, time).
        brain_mask: Optional 3D brain mask.
        tr: Repetition time in seconds for integration.
        kep: EES efflux rate constant in s⁻¹.

    Returns:
        3D K2 map with leakage rate per voxel (s⁻¹).

    References:
        Leu K et al. AJNR 2016;37(8):1440-1446. doi:10.3174/ajnr.A4759
    """
    if concentration.ndim != 4:
        raise ValueError(f"Concentration must be 4D, got {concentration.ndim}D")

    n_timepoints = concentration.shape[3]
    k2_map = np.zeros(concentration.shape[:3], dtype=np.float64)

    # Compute mean brain reference curve C̄(t)
    c_bar = _compute_mean_brain_reference(concentration, brain_mask)
    if c_bar is None:
        return k2_map

    # Compute instantaneous EES concentration with washout
    c_e = _compute_ees_concentration(c_bar, kep, tr)

    # Derive regression start from C̄ peak (2 frames after peak)
    regression_start = int(np.argmax(c_bar)) + 2
    if regression_start >= n_timepoints - 2:
        logger.warning(
            "Mean reference peak at frame %d of %d — too late for regression",
            regression_start - 2, n_timepoints,
        )
        return k2_map

    # Build design matrix: [C̄(t), C_e(t)]
    c_bar_post = c_bar[regression_start:]
    c_e_post = c_e[regression_start:]

    X = np.column_stack([c_bar_post, c_e_post])  # (n_post, 2)

    # Condition check on shared design matrix
    if np.linalg.cond(X) > 1e8:
        logger.warning("Bidirectional K2 design matrix ill-conditioned")
        return k2_map

    # Determine valid voxels
    if brain_mask is not None:
        valid_mask = brain_mask
    else:
        max_conc = np.max(concentration, axis=-1)
        valid_mask = max_conc > EPSILON_CONCENTRATION

    valid_indices = np.where(valid_mask)
    n_voxels = len(valid_indices[0])

    if n_voxels == 0:
        return k2_map

    # Vectorized solve: extract all valid voxel time courses
    Y = concentration[
        valid_indices[0], valid_indices[1], valid_indices[2], regression_start:
    ]  # (n_voxels, n_post)

    # lstsq: X @ coeffs = Y.T → coeffs shape (2, n_voxels)
    coeffs, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)

    # NEGATE: K2_physical = -K2_regression (T1 leakage gives negative K2_reg)
    # Clip: only correct T1-dominant leakage (positive K2_phys)
    k2_values = np.clip(-coeffs[1, :], 0.0, K2_UPPER_BOUND)

    k2_map[valid_indices] = k2_values

    return k2_map


def apply_k2_correction_bidirectional(
    concentration: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    tr: float = 1.5,
    kep: Optional[float] = None,
    k2_map: Optional[npt.NDArray[np.floating]] = None,
    baseline_frames: int = 5,
) -> K2CorrectionResult:
    """Apply bidirectional K2 leakage correction (Leu et al. 2016).

    Extends the standard BSW model to account for contrast agent washout
    from the EES:

        C_corrected(t) = C_tissue(t) + K2 × C_e(t)

    where C_e(t) = C̄(t) ⊗ exp(-kep × t) is the instantaneous EES
    concentration with exponential washout.

    Key difference from standard BSW:
    - BSW uses cumulative integral ∫C̄(τ)dτ (no washout, always increases)
    - Bidirectional uses C_e(t) (accounts for washout, can decrease)

    This produces ~16.6% lower rCBV in high-grade gliomas because standard
    BSW overcorrects late timepoints (Leu 2016).

    Args:
        concentration: 4D concentration array (x, y, z, time).
        brain_mask: Optional 3D brain mask for voxel selection.
        tr: Repetition time in seconds.
        kep: EES efflux rate in s⁻¹. If None, uses KEP_DSC_DEFAULT (0.005).
        k2_map: Pre-computed K2 map. If None, K2 is estimated per voxel.
        baseline_frames: Number of pre-bolus baseline frames.

    Returns:
        K2CorrectionResult with corrected concentration and K2 map.

    References:
        Leu K et al. AJNR 2016;37(8):1440-1446. doi:10.3174/ajnr.A4759
    """
    if concentration.ndim != 4:
        raise ValueError(f"Concentration must be 4D, got {concentration.ndim}D")

    if kep is None:
        kep = KEP_DSC_DEFAULT

    # Validate kep range
    if not (KEP_DSC_LOWER <= kep <= KEP_DSC_UPPER):
        logger.warning(
            "kep=%.4f s^-1 outside typical range [%.3f, %.3f] s^-1 (Leu 2016)",
            kep, KEP_DSC_LOWER, KEP_DSC_UPPER,
        )

    # Baseline-subtract before K2 estimation
    n_bl = max(baseline_frames, 4)
    conc_bl = np.mean(concentration[..., :n_bl], axis=-1, keepdims=True)
    concentration = concentration - conc_bl

    # Compute AUC before any early returns (needed for QC)
    cumsum_tissue = np.cumsum(concentration, axis=-1) * tr
    auc_tissue = cumsum_tissue[..., -1].copy()

    # Estimate K2 if not provided
    if k2_map is None:
        k2_map = estimate_k2_bidirectional(
            concentration=concentration,
            brain_mask=brain_mask,
            tr=tr,
            kep=kep,
        )

    # Gate K2: voxels with negligible leakage are left uncorrected
    K2_THRESHOLD = 1e-4  # s⁻¹
    k2_map = np.where(k2_map > K2_THRESHOLD, k2_map, 0.0)

    # Compute mean brain reference for correction
    c_bar = _compute_mean_brain_reference(concentration, brain_mask)
    if c_bar is None:
        logger.warning(
            "Bidirectional K2 correction skipped: insufficient brain voxels"
        )
        return K2CorrectionResult(
            concentration_corrected=concentration.copy(),
            k2_map=k2_map,
            auc_tissue=auc_tissue,
            correction_applied=False,
            method="bidirectional",
            kep_used=kep,
        )

    # Compute instantaneous EES concentration with washout
    c_e = _compute_ees_concentration(c_bar, kep, tr)
    c_e_4d = c_e[np.newaxis, np.newaxis, np.newaxis, :]

    # Expand K2 map for broadcasting
    k2_expanded = k2_map[..., np.newaxis]

    # Apply correction: C_corrected(t) = C_tissue(t) + K2 × C_e(t)
    concentration_corrected = concentration + k2_expanded * c_e_4d

    # Compute QC metrics
    qc_metrics = _compute_k2_qc_metrics(k2_map, brain_mask, auc_tissue)
    qc_metrics["k2_method"] = "bidirectional"
    qc_metrics["kep_used"] = kep

    return K2CorrectionResult(
        concentration_corrected=concentration_corrected,
        k2_map=k2_map,
        auc_tissue=auc_tissue,
        correction_applied=True,
        qc_metrics=qc_metrics,
        method="bidirectional",
        kep_used=kep,
    )


def estimate_preload_correction_factor(
    preload_result: PreloadResult,
    field_strength: float = 3.0,
) -> float:
    """Estimate correction factor for preload effects.

    The correction factor can be used to adjust quantitative
    perfusion values for preload-induced signal loss.

    Args:
        preload_result: Result from preload detection.
        field_strength: Magnetic field strength in Tesla.

    Returns:
        Correction factor (>= 1.0, multiply CBV/CBF by this).
    """
    if not preload_result.is_preloaded:
        return 1.0

    # Signal loss due to preload reduces apparent contrast effect
    # Correction factor based on baseline T2* shortening
    # Reference: Boxerman et al. approach

    delta_r2 = preload_result.baseline_delta_r2

    # For small preload, correction is approximately linear
    # For larger preload, effect is more complex
    if delta_r2 < 2.0:
        # Minimal correction for small preload
        correction = 1.0 + 0.05 * delta_r2
    elif delta_r2 < 5.0:
        # Moderate correction
        correction = 1.1 + 0.03 * (delta_r2 - 2.0)
    else:
        # Significant correction, but cap to avoid over-correction
        correction = min(1.25 + 0.02 * (delta_r2 - 5.0), 1.5)

    return correction
