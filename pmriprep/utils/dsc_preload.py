"""DSC preload detection and contrast agent residual estimation.

Automatic detection of contrast preload from prior DCE injection
and classification of residual contrast effects on DSC signal.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
    EPSILON_SIGNAL,
    GD_DOTA_STANDARD_DOSE,
)

logger = logging.getLogger(__name__)

_K2_REFERENCE_TAIL_FRAMES: Final[int] = 10


# =============================================================================
# Preload Classification
# =============================================================================


class PreloadClass(str, Enum):
    """Administered-preload / residual-ΔR2* severity levels.

    Internal advisory bins for residual contrast effects estimated from
    baseline ΔR2*. Production outputs retain the continuous measurement rather
    than publishing these categories.
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

    UNKNOWN = "unknown"
    """Preload could not be determined."""


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


def get_preload_thresholds(
    field_strength: float = 3.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> PreloadThresholds:
    """Get preload classification thresholds for field strength and agent.

    Thresholds are in ΔR2* units (s⁻¹) and were calibrated for standard
    Gd-DOTA-class agents (``agent_standard_dose = 0.1 mmol Gd/kg``). Agents
    with different standard doses (e.g., gadopiclenol/Vueway at 0.05,
    gadoquatrane at 0.04) produce systematically different ΔR2* per
    "fraction of standard dose." Tissue r2* itself is agent-independent
    (Gd³⁺ susceptibility), so linear dose-normalization is the correct
    correction: ``effective_threshold = base_threshold × (agent_standard_dose / 0.1)``.

    Parameters
    ----------
    field_strength : float, optional
        Magnetic field strength in Tesla. Uses nearest
        available field strength if exact match not in table.
    agent_standard_dose : float, optional
        Agent's recommended standard single dose in
        mmol Gd/kg. Default 0.1 (Gd-DOTA class). Thresholds scale
        linearly; values outside [0.01, 0.2] produce a warning.

    Returns
    -------
    PreloadThresholds
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

    scale = agent_standard_dose / GD_DOTA_STANDARD_DOSE
    return PreloadThresholds(
        none_max=base.none_max * scale,
        minimal_max=base.minimal_max * scale,
        half_dose_max=base.half_dose_max * scale,
        significant_max=base.significant_max * scale,
    )


#: Administered-dose bin edges for user-specified ``preload`` float.
#: Clinically calibrated:
#:   quarter-dose preload (0.25) → HALF_DOSE class
#:   half-dose preload (0.5)      → HALF_DOSE class
#:   full-dose preload (1.0)      → SIGNIFICANT class
#:   double-dose (>1.15)          → SEVERE class
#: Edges are heuristic for K2 uncertainty bucketing; the MINIMAL→HALF_DOSE
#: edge (0.20) and SIGNIFICANT→SEVERE edge (1.15) in particular are
#: conventions rather than literature-derived boundaries.
def residual_equivalent_fraction_from_delta_r2(
    delta_r2: float,
    field_strength: float = 3.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> float:
    """Convert baseline ΔR2* to a residual-equivalent dose fraction.

    This is NOT the administered preload dose. It is the residual tissue
    Gd concentration at DSC start, expressed as a fraction of what the
    agent's standard dose would produce if fully retained. Actual
    administered preload is typically 1.5×–3× larger than this value
    because of renal clearance between preload and DSC bolus (5–10 min for
    ASFNR-style preload protocols; clearance depends on GFR).

    Used for QC transparency so the advisory estimate can be compared to
    user-specified administered dose (different physical quantities, same
    class system for bucketing).

    Parameters
    ----------
    delta_r2 : float
        Baseline ΔR2* in s⁻¹ (from detect_preload_from_signal).
    field_strength : float, optional
        Magnetic field in Tesla.
    agent_standard_dose : float, optional
        Agent's standard dose in mmol Gd/kg.

    Returns
    -------
    float
        Residual-equivalent fraction. 0.0 if ``delta_r2 <= 0`` (no detectable
        preload signal — valid physical edge case).

    Raises
    ------
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

    Attributes
    ----------
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
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> PreloadClass:
    """Classify preload severity from baseline ΔR2* (agent-aware).

    Thresholds scale linearly with ``agent_standard_dose`` — see
    :func:`get_preload_thresholds` for rationale.

    Parameters
    ----------
    baseline_delta_r2 : float
        Baseline ΔR2* value in s⁻¹.
    field_strength : float, optional
        Magnetic field strength in Tesla.
    agent_standard_dose : float, optional
        Agent's standard dose in mmol Gd/kg
        (default 0.1 for Gd-DOTA class).

    Returns
    -------
    PreloadClass
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
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> PreloadResult:
    """Detect preload from DSC signal baseline.

    Analyzes the baseline signal before bolus arrival to estimate
    residual contrast effects from prior DCE injection.

    Method:
        1. Compute mean signal over baseline frames
        2. Compare to expected non-preloaded signal (or use reference)
        3. Estimate baseline ΔR2* = -ln(S_baseline/S_reference) / TE
        4. Classify based on ΔR2* thresholds

    Parameters
    ----------
    signal : npt.NDArray[np.floating]
        4D DSC signal array (x, y, z, time).
    te : float
        Echo time in seconds.
    baseline_frames : int, optional
        Number of pre-bolus frames to analyze.
    brain_mask : npt.NDArray[np.bool_], optional
        Optional 3D binary mask for brain voxels.
    field_strength : float, optional
        Magnetic field strength in Tesla.
    reference_signal : float, optional
        Expected signal without preload. If None,
        uses the 95th percentile of baseline signal as reference
        (assuming some voxels have minimal preload effect).

    Returns
    -------
    PreloadResult
        PreloadResult with classification and metrics.

    Raises
    ------
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

    # Anatomical CV is retained as a separate QC detail and measures
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
    field_strength: float = 3.0,
) -> PreloadResult:
    """Detect preload from DCE protocol information.

    Uses knowledge of prior DCE acquisition to estimate preload.

    Parameters
    ----------
    dce_ran : bool
        Whether DCE was acquired before DSC.
    dce_end_time : float, optional
        Time when DCE acquisition ended (seconds from session start).
    dsc_start_time : float, optional
        Time when DSC acquisition started.
    dce_dose : float, optional
        Dose multiplier (0.5=half, 1.0=full, 2.0=double).
    field_strength : float, optional
        Magnetic field strength in Tesla. Used to scale the
        base ΔR2* (which scales linearly with B0 in tissue) and to
        select field-appropriate preload thresholds.

    Returns
    -------
    PreloadResult
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

    # Estimate baseline ΔR2* based on dose and clearance.
    # Clearance model: C(t) = C0 * exp(-t/tau); tau ~ 20 min healthy.
    clearance_tau = 1200.0  # 20 minutes in seconds

    # Base ΔR2* for full dose immediately after DCE, calibrated at 3T,
    # scaled linearly with B0 (Yablonskiy regime, susceptibility-driven).
    base_delta_r2_3t = 5.0  # s⁻¹, half-dose equivalent at 3T
    base_delta_r2 = base_delta_r2_3t * (field_strength / 3.0)

    if elapsed_time is not None and elapsed_time > 0:
        # Apply clearance decay
        decay_factor = float(np.exp(-elapsed_time / clearance_tau))
    else:
        # Assume minimal clearance (just finished DCE)
        decay_factor = 0.9

    estimated_delta_r2 = base_delta_r2 * dose_factor * decay_factor

    classification = classify_preload(estimated_delta_r2, field_strength)

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
            "field_strength": field_strength,
        },
    )


def detect_preload_from_dicom(
    dicom_metadata: dict[str, Any],
    field_strength: float = 3.0,
) -> PreloadResult:
    """Detect preload from DICOM metadata.

    Looks for contrast agent administration information in DICOM tags.

    Parameters
    ----------
    dicom_metadata : dict[str, Any]
        Dictionary of DICOM metadata fields.
    field_strength : float, optional
        Magnetic field strength in Tesla. ΔR2* per
        unit Gd scales linearly with B0; the dose-bin estimates
        below are calibrated at 3T and scaled accordingly.

    Returns
    -------
    PreloadResult
        PreloadResult based on DICOM information.
    """
    # Look for contrast administration tags
    contrast_bolus_agent = dicom_metadata.get("ContrastBolusAgent", "")
    contrast_bolus_volume = dicom_metadata.get("ContrastBolusVolume", None)

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

    # Try to estimate dose from volume.
    # Standard: 0.1 mmol/kg, ~7-10 mL for typical adult.
    # ΔR2* values are 3T half/full/double-dose equivalents, scaled to
    # field_strength linearly (Yablonskiy regime).
    b0_scale = field_strength / 3.0
    if contrast_bolus_volume is not None:
        try:
            volume_ml = float(contrast_bolus_volume)
            if volume_ml < 5:
                dose = "half"
                estimated_delta_r2 = 2.5 * b0_scale
            elif volume_ml < 15:
                dose = "full"
                estimated_delta_r2 = 5.0 * b0_scale
            else:
                dose = "double"
                estimated_delta_r2 = 10.0 * b0_scale
        except (ValueError, TypeError):
            dose = "unknown"
            estimated_delta_r2 = 5.0 * b0_scale  # Assume full dose
    else:
        dose = "unknown"
        estimated_delta_r2 = 5.0 * b0_scale

    classification = classify_preload(estimated_delta_r2, field_strength)

    return PreloadResult(
        classification=classification,
        baseline_delta_r2=estimated_delta_r2,
        detection_method="dicom_metadata",
        confidence=0.6,
        details={
            "contrast_agent": contrast_bolus_agent,
            "volume_ml": contrast_bolus_volume,
            "estimated_dose": dose,
            "field_strength": field_strength,
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
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> PreloadResult:
    """Detect preload using all available information.

    Combines multiple detection methods and returns the most confident result:
    1. Direct signal measurement (highest confidence)
    2. DCE protocol information
    3. DICOM metadata

    Parameters
    ----------
    signal : npt.NDArray[np.floating], optional
        4D DSC signal array (x, y, z, time). Optional.
    te : float, optional
        Echo time in seconds. Required if signal provided.
    baseline_frames : int, optional
        Number of pre-bolus frames for signal analysis.
    brain_mask : npt.NDArray[np.bool_], optional
        Optional 3D binary mask for brain voxels.
    field_strength : float, optional
        Magnetic field strength in Tesla.
    dce_ran : bool, optional
        Whether DCE was acquired before DSC. Optional.
    dce_metadata : dict[str, Any], optional
        DCE protocol information (dose, timing). Optional.
    dicom_metadata : dict[str, Any], optional
        DICOM metadata dictionary. Optional.
    agent_standard_dose : float, optional
        Agent's standard dose in mmol Gd/kg
        (default 0.1 for Gd-DOTA class). Forwarded to the signal-based
        detector so classification thresholds scale linearly with the
        administered agent's standard dose.

    Returns
    -------
    PreloadResult
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
            field_strength=field_strength,
        )
        results.append(result)

    # Method 3: DICOM metadata
    if dicom_metadata is not None:
        result = detect_preload_from_dicom(
            dicom_metadata, field_strength=field_strength,
        )
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
    # Note: confidence scales differ across methods — signal-based is
    # probabilistic P(correct classification) via SEM+z-score, while
    # DCE-protocol and DICOM-metadata methods return heuristic values
    # (~0.5–0.7). A low-confidence signal detection (near-edge ΔR2* or
    # tiny mask) may defer to DCE timing, which is usually acceptable —
    # but confidence values are not directly comparable across methods.
    # Consider sorting by detection_method priority first if this causes
    # issues in practice.
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

    Attributes
    ----------
        concentration_corrected: Leakage-corrected concentration curves (4D).
        k2_map: Signed K2 parameter map (3D), in s⁻¹.
        auc_tissue: Area under tissue curve before correction (3D).
        correction_applied: Boolean indicating if correction was applied.
        qc_metrics: QC metrics from correction.
        method: Leakage correction method ("bsw").
    """

    concentration_corrected: npt.NDArray[np.floating]
    k2_map: npt.NDArray[np.floating]
    auc_tissue: npt.NDArray[np.floating]
    correction_applied: bool = True
    qc_metrics: Optional[dict[str, Any]] = None
    method: str = "bsw"


def _compute_bsw_reference(
    concentration: npt.NDArray[np.floating],
    signal: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]],
    baseline_frames: int = 5,
    reference_qc: Optional[dict[str, Any]] = None,
) -> Optional[npt.NDArray[np.floating]]:
    """Compute the conventional BSW non-enhancing reference curve C̄(t).

    Reference membership follows a conventional signal-domain profile: the
    mean of the final ten raw-signal samples must not exceed the voxelwise
    baseline mean by more than one baseline sample SD. The returned reference
    remains the mean Delta-R2* curve over the selected support.

    Parameters
    ----------
    concentration : npt.NDArray[np.floating]
        4D concentration array (x, y, z, time).
    signal : npt.NDArray[np.floating]
        Motion-corrected raw DSC signal on the same grid and time axis.
    brain_mask : Optional[npt.NDArray[np.bool_]]
        Optional 3D brain mask. If None, uses voxels
        with max concentration above noise floor.
    baseline_frames : int, optional
        Number of pre-bolus frames used to estimate temporal noise.
    reference_qc : Optional[dict[str, Any]], optional
        Mutable dictionary populated with reference-selection accounting.

    Returns
    -------
    Optional[npt.NDArray[np.floating]]
        1D array (n_timepoints,) or None if insufficient voxels.
    """
    # Reference support floors target a p95 whole-brain median-K2 perturbation
    # below 0.001 s^-1. The warning level additionally targets a p95 relative
    # reference-curve error of approximately 5-6%. These are stability
    # safeguards, not physiological thresholds.
    _MIN_VOXELS = 250
    _WARN_VOXELS = 1000

    if signal.shape != concentration.shape:
        raise ValueError(
            "Raw signal and concentration must have identical shapes for "
            f"BSW reference selection, got {signal.shape} and {concentration.shape}"
        )

    # Select brain voxels
    if brain_mask is not None:
        valid_indices = np.where(brain_mask)
    else:
        max_conc = np.max(concentration, axis=-1)
        valid_indices = np.where(max_conc > EPSILON_CONCENTRATION)

    n_brain = len(valid_indices[0])
    if reference_qc is not None:
        reference_qc["k2_reference_total_voxels"] = int(n_brain)
    if n_brain < _MIN_VOXELS:
        if reference_qc is not None:
            reference_qc["k2_reference_selected_voxels"] = 0
            reference_qc["k2_reference_available"] = False
            reference_qc["k2_reference_failure_reason"] = (
                "insufficient_brain_voxels"
            )
        logger.warning(
            "Insufficient brain voxels for mean reference: %d (need ≥%d)",
            n_brain, _MIN_VOXELS,
        )
        return None

    # Extract brain voxel time courses: (n_voxels, n_timepoints)
    brain_curves = concentration[valid_indices[0], valid_indices[1], valid_indices[2], :]
    brain_signal = signal[valid_indices[0], valid_indices[1], valid_indices[2], :]

    n_tp = brain_curves.shape[1]
    if n_tp < 4:
        if reference_qc is not None:
            reference_qc["k2_reference_selected_voxels"] = 0
            reference_qc["k2_reference_available"] = False
            reference_qc["k2_reference_failure_reason"] = "insufficient_timepoints"
        logger.warning("Insufficient timepoints for BSW reference selection: %d", n_tp)
        return None
    n_bl = min(max(int(baseline_frames), 2), n_tp - 2)
    n_tail = _K2_REFERENCE_TAIL_FRAMES
    if n_tp - n_bl < n_tail:
        if reference_qc is not None:
            reference_qc["k2_reference_selected_voxels"] = 0
            reference_qc["k2_reference_available"] = False
            reference_qc["k2_reference_failure_reason"] = "insufficient_late_tail"
        logger.warning(
            "Insufficient post-baseline frames for BSW reference selection: "
            "%d available, %d required",
            n_tp - n_bl,
            n_tail,
        )
        return None
    if reference_qc is not None:
        reference_qc["k2_reference_baseline_frames"] = int(n_bl)
        reference_qc["k2_reference_tail_frames"] = int(n_tail)
        reference_qc["k2_reference_selection"] = (
            "raw_signal_final10_mean_1_baseline_sample_sd"
        )
        reference_qc["k2_reference_selection_domain"] = "raw_signal"
    baseline_mean = np.mean(brain_signal[:, :n_bl], axis=1)
    baseline_std = np.std(brain_signal[:, :n_bl], axis=1, ddof=1)
    late_mean = np.mean(brain_signal[:, -n_tail:], axis=1)
    non_leaky = (
        np.isfinite(late_mean)
        & np.isfinite(baseline_mean)
        & np.isfinite(baseline_std)
        & (baseline_mean > EPSILON_SIGNAL)
        & (late_mean <= baseline_mean + baseline_std)
    )

    n_excluded = int(np.sum(~non_leaky))
    n_remaining = int(np.sum(non_leaky))
    if reference_qc is not None:
        reference_qc["k2_reference_selected_voxels"] = n_remaining
        reference_qc["k2_reference_selected_fraction"] = float(n_remaining / n_brain)

    if n_remaining < _MIN_VOXELS:
        if reference_qc is not None:
            reference_qc["k2_reference_available"] = False
            reference_qc["k2_reference_failure_reason"] = (
                "insufficient_selected_support"
            )
        logger.warning(
            "Insufficient BSW reference voxels after late-tail selection: "
            "%d of %d (need ≥%d)",
            n_remaining,
            n_brain,
            _MIN_VOXELS,
        )
        return None
    brain_curves = brain_curves[non_leaky]

    c_bar = np.mean(brain_curves, axis=0)
    if reference_qc is not None:
        reference_qc["k2_reference_available"] = True
        reference_qc["k2_reference_peak"] = float(np.max(c_bar))

    n_used = len(brain_curves)
    logger.info(
        "Mean brain reference: %d voxels used, %d excluded by late-tail criterion, "
        "peak C̄ = %.3f",
        n_used, n_excluded, float(np.max(c_bar)),
    )
    if n_used < _WARN_VOXELS:
        logger.warning(
            "Mean brain reference from only %d voxels (<1000 recommended); "
            "consider checking brain mask quality",
            n_used,
        )

    return np.asarray(c_bar, dtype=np.float64)


def estimate_k2_per_voxel(
    concentration: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    tr: float = 1.5,
    baseline_frames: int = 5,
    signal: Optional[npt.NDArray[np.floating]] = None,
    reference_curve: Optional[npt.NDArray[np.floating]] = None,
) -> npt.NDArray[np.floating]:
    """Estimate K2 leakage parameter per voxel using BSW model.

    K2 represents the rate of contrast agent leakage through the BBB.
    Estimated using the Boxerman-Schmainda-Weisskoff model with a mean
    brain reference curve C̄(t):

        C_voxel(t) = K1 × C̄(t) - K2 × ∫₀ᵗ C̄(τ)dτ

    K1 captures instantaneous susceptibility scaling relative to the mean
    brain. K2 remains signed so both fitted leakage directions are retained.

    Parameters
    ----------
    concentration : npt.NDArray[np.floating]
        4D concentration array (x, y, z, time).
    brain_mask : Optional[npt.NDArray[np.bool_]]
        Optional 3D brain mask.
    tr : float, optional
        Repetition time in seconds for integration.
    baseline_frames : int, optional
        Number of pre-bolus frames used for reference selection.
    signal : Optional[npt.NDArray[np.floating]], optional
        Motion-corrected raw signal used by the reference selector.
        Required unless ``reference_curve`` is supplied.
    reference_curve : Optional[npt.NDArray[np.floating]], optional
        Precomputed BSW reference. When omitted, it is selected from the data.

    Returns
    -------
    npt.NDArray[np.floating]
        3D K2 map with leakage rate per voxel (s⁻¹).

    Notes
    -----
        - K2 ≈ 0: Intact blood-brain barrier
        - K2 > 0: BBB disruption (T1 leakage)
        - K2 < 0: Opposite-sign leakage contribution retained by the fit
        - Uses vectorized OLS for efficiency

    """
    if concentration.ndim != 4:
        raise ValueError(f"Concentration must be 4D, got {concentration.ndim}D")
    if concentration.shape[-1] < 4:
        raise ValueError("K2 correction requires at least 4 timepoints")

    n_timepoints = concentration.shape[3]
    k2_map = np.full(concentration.shape[:3], np.nan, dtype=np.float64)

    # Compute mean brain reference curve C̄(t)
    reference_qc: dict[str, Any] = {}
    c_bar = reference_curve
    if c_bar is None:
        if signal is None:
            raise ValueError(
                "Raw signal is required for BSW reference selection"
            )
        c_bar = _compute_bsw_reference(
            concentration,
            signal,
            brain_mask,
            baseline_frames=baseline_frames,
            reference_qc=reference_qc,
        )
    else:
        reference_qc["k2_reference_available"] = True
        reference_qc["k2_reference_selection"] = "provided"
    if c_bar is None:
        return k2_map
    c_bar = np.asarray(c_bar, dtype=np.float64)
    if c_bar.shape != (n_timepoints,) or not np.all(np.isfinite(c_bar)):
        raise ValueError(
            f"Reference curve must be finite with shape ({n_timepoints},), "
            f"got {c_bar.shape}"
        )

    cumsum_c_bar = cumulative_trapezoid(c_bar, dx=tr, initial=0.0)
    X = np.column_stack([c_bar, -cumsum_c_bar])

    # A rank-deficient design cannot identify K1 and K2 independently.
    if np.linalg.matrix_rank(X) < X.shape[1]:
        logger.warning("Mean brain reference design matrix is rank deficient")
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
            n_voxels, n_voxels * n_timepoints * 8 / 1e6,
        )

    # Vectorized solve: extract all valid voxel time courses
    Y = concentration[valid_indices]

    # lstsq: X @ coeffs = Y.T → coeffs shape (2, n_voxels)
    coeffs, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)

    k2_map[valid_indices] = coeffs[1, :]

    return k2_map


def apply_k2_correction(
    concentration: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    tr: float = 1.5,
    k2_map: Optional[npt.NDArray[np.floating]] = None,
    baseline_frames: int = 5,
    signal: Optional[npt.NDArray[np.floating]] = None,
    reference_curve: Optional[npt.NDArray[np.floating]] = None,
) -> K2CorrectionResult:
    """Apply K2 leakage correction using the BSW method with mean brain reference.

    Implements the Boxerman-Schmainda-Weisskoff (BSW) leakage correction
    for DSC-MRI using a mean brain reference curve C̄(t) rather than AIF:

        C_corrected(t) = C_tissue(t) + K2 × ∫₀ᵗ C̄(τ)dτ

    K2 is estimated from the reference scale, so correction must use the
    same reference. Using AIF would produce wrong magnitudes (AIF >> C̄).

    Baseline subtraction is applied before K2 estimation to ensure the
    BSW model assumption C(t)=0 at baseline is satisfied. This is critical
    for preloaded acquisitions where residual Gd creates a DC offset.

    Parameters
    ----------
    concentration : npt.NDArray[np.floating]
        4D concentration array (x, y, z, time).
    brain_mask : Optional[npt.NDArray[np.bool_]]
        Optional 3D brain mask for voxel selection.
    tr : float, optional
        Repetition time in seconds for integration.
    k2_map : Optional[npt.NDArray[np.floating]]
        Pre-computed K2 map. If None, K2 is estimated per voxel.
    baseline_frames : int, optional
        Number of pre-bolus baseline frames for
        baseline subtraction. Default: 5.
    signal : Optional[npt.NDArray[np.floating]], optional
        Motion-corrected raw signal used by the reference selector.
        Required unless ``reference_curve`` is supplied.
    reference_curve : Optional[npt.NDArray[np.floating]], optional
        Precomputed BSW reference. Primarily useful for validation.

    Returns
    -------
    K2CorrectionResult
        K2CorrectionResult with corrected concentration and K2 map.

    """
    if concentration.ndim != 4:
        raise ValueError(f"Concentration must be 4D, got {concentration.ndim}D")
    if concentration.shape[-1] < 4:
        raise ValueError("K2 correction requires at least 4 timepoints")

    # Baseline-subtract before K2 estimation. The BSW model assumes
    # C(t)=0 at baseline; non-zero offset biases the cumulative integral
    # in K2 regression. For non-preloaded data, baseline ≈ 0 (no-op).
    n_bl = min(max(int(baseline_frames), 4), concentration.shape[-1] - 2)
    conc_bl = np.mean(concentration[..., :n_bl], axis=-1, keepdims=True)
    concentration = concentration - conc_bl

    # Compute AUC before any early returns (needed for QC regardless).
    # Trapezoidal integration over the concentration curve.
    auc_tissue = np.trapezoid(concentration, dx=tr, axis=-1)

    reference_qc: dict[str, Any] = {}
    c_bar = reference_curve
    if c_bar is None:
        if signal is None:
            raise ValueError(
                "Raw signal is required for BSW reference selection"
            )
        c_bar = _compute_bsw_reference(
            concentration,
            signal,
            brain_mask,
            baseline_frames=n_bl,
            reference_qc=reference_qc,
        )
    else:
        reference_qc["k2_reference_available"] = True
        reference_qc["k2_reference_selection"] = "provided"
    if c_bar is None:
        logger.warning(
            "K2 correction skipped: insufficient late-tail reference support"
        )
        return K2CorrectionResult(
            concentration_corrected=concentration.copy(),
            k2_map=np.full(concentration.shape[:3], np.nan, dtype=np.float64),
            auc_tissue=auc_tissue,
            correction_applied=False,
            qc_metrics={"k2_method": "bsw", "k2_signed": True, **reference_qc},
            method="bsw",
        )

    # Estimate K2 if not provided
    if k2_map is None:
        k2_map = estimate_k2_per_voxel(
            concentration=concentration,
            brain_mask=brain_mask,
            tr=tr,
            baseline_frames=n_bl,
            reference_curve=c_bar,
        )

    # Apply the conventional cumulative reference integral over the complete
    # baseline-subtracted dynamic series.
    cumsum_ref = cumulative_trapezoid(c_bar, dx=tr, initial=0.0)

    # Expand K2 map to match time dimension for broadcasting
    finite_k2 = np.where(np.isfinite(k2_map), k2_map, 0.0)
    # Apply correction: C_corrected(t) = C_tissue(t) + K2 × ∫ C̄(τ)dτ.
    concentration_corrected = concentration.copy()
    concentration_corrected += (
        finite_k2[..., np.newaxis] * cumsum_ref[np.newaxis, np.newaxis, np.newaxis, :]
    )

    # Allow negative values after baseline subtraction — symmetric noise
    # around zero preserves the mean and avoids positive bias from clipping.

    # Compute QC metrics
    qc_metrics = _compute_k2_qc_metrics(k2_map, brain_mask, auc_tissue)
    qc_metrics["k2_method"] = "bsw"
    qc_metrics["k2_signed"] = True
    qc_metrics.update(reference_qc)

    return K2CorrectionResult(
        concentration_corrected=concentration_corrected,
        k2_map=k2_map,
        auc_tissue=auc_tissue,
        correction_applied=True,
        qc_metrics=qc_metrics,
        method="bsw",
    )


def _compute_k2_qc_metrics(
    k2_map: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]],
    auc_tissue: npt.NDArray[np.floating],
) -> dict[str, Any]:
    """Compute QC metrics for K2 leakage correction.

    Parameters
    ----------
    k2_map : npt.NDArray[np.floating]
        3D K2 parameter map.
    brain_mask : Optional[npt.NDArray[np.bool_]]
        Optional 3D brain mask.
    auc_tissue : npt.NDArray[np.floating]
        3D area under tissue curve.

    Returns
    -------
    dict[str, Any]
        Dictionary of QC metrics.
    """
    # Select one common support for K2 and AUC summaries.
    if brain_mask is not None:
        candidate_mask = np.asarray(brain_mask, dtype=bool)
    else:
        candidate_mask = np.ones(k2_map.shape, dtype=bool)

    n_total = int(np.sum(candidate_mask))
    valid_mask = (
        candidate_mask & np.isfinite(k2_map) & np.isfinite(auc_tissue)
    )
    k2_values = k2_map[valid_mask]
    auc_values = auc_tissue[valid_mask]

    if k2_values.size == 0:
        return {
            "k2_mean": np.nan,
            "k2_std": np.nan,
            "k2_min": np.nan,
            "k2_max": np.nan,
            "k2_median": np.nan,
            "k2_p05": np.nan,
            "k2_p95": np.nan,
            "percent_negative_k2": np.nan,
            "n_brain_voxels": n_total,
            "n_k2_valid": 0,
            "percent_k2_undefined": 100.0 if n_total else np.nan,
            "auc_mean": float(np.mean(auc_values)) if auc_values.size else np.nan,
        }

    # K2 statistics
    k2_mean = float(np.mean(k2_values))
    k2_std = float(np.std(k2_values))
    k2_min = float(np.min(k2_values))
    k2_max = float(np.max(k2_values))
    k2_median = float(np.median(k2_values))
    k2_p05, k2_p95 = np.percentile(k2_values, [5, 95])

    return {
        "k2_mean": k2_mean,
        "k2_std": k2_std,
        "k2_min": k2_min,
        "k2_max": k2_max,
        "k2_median": k2_median,
        "k2_p05": float(k2_p05),
        "k2_p95": float(k2_p95),
        "percent_negative_k2": float(100.0 * np.mean(k2_values < 0.0)),
        "n_brain_voxels": n_total,
        "n_k2_valid": int(k2_values.size),
        "percent_k2_undefined": float(
            100.0 * (n_total - k2_values.size) / n_total
        ) if n_total else np.nan,
        "auc_mean": float(np.mean(auc_values)) if auc_values.size > 0 else 0.0,
    }

def estimate_preload_correction_factor(
    preload_result: PreloadResult,
    field_strength: float = 3.0,
) -> float:
    """Estimate correction factor for preload effects.

    The correction factor can be used to adjust quantitative
    perfusion values for preload-induced signal loss.

    Parameters
    ----------
    preload_result : PreloadResult
        Result from preload detection.
    field_strength : float, optional
        Magnetic field strength in Tesla.

    Returns
    -------
    float
        Correction factor (>= 1.0, multiply CBV/CBF by this).
    """
    if not preload_result.is_preloaded:
        return 1.0

    # Signal loss due to preload reduces apparent contrast effect
    # Correction factor based on baseline T2* shortening
    # Boxerman-Schmainda-Weisskoff approach

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
