"""DSC preload detection and contrast agent residual estimation.

Automatic detection of contrast preload from prior DCE injection
and classification of residual contrast effects on DSC signal.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dsc_constants import EPSILON_DELTA_R2, EPSILON_SIGNAL


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


def get_preload_thresholds(field_strength: float = 3.0) -> PreloadThresholds:
    """Get preload classification thresholds for field strength.

    Uses nearest field strength if exact match not available.

    Args:
        field_strength: Magnetic field strength in Tesla.

    Returns:
        PreloadThresholds for the specified field strength.
    """
    if field_strength in PRELOAD_THRESHOLDS_BY_FIELD:
        return PRELOAD_THRESHOLDS_BY_FIELD[field_strength]

    # Use nearest field strength
    fields = sorted(PRELOAD_THRESHOLDS_BY_FIELD.keys())
    nearest = min(fields, key=lambda x: abs(x - field_strength))
    return PRELOAD_THRESHOLDS_BY_FIELD[nearest]


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
        confidence: Confidence score (0-1) in the detection.
        details: Additional details about the detection.
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
) -> PreloadClass:
    """Classify preload severity from baseline ΔR2*.

    Args:
        baseline_delta_r2: Baseline ΔR2* value in s⁻¹.
        field_strength: Magnetic field strength in Tesla.

    Returns:
        PreloadClass indicating severity level.
    """
    thresholds = get_preload_thresholds(field_strength)

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

    # Classify preload
    classification = classify_preload(baseline_delta_r2, field_strength)

    # Estimate residual concentration if r2* available
    from pmriprep.utils.dsc_constants import get_r2_star

    r2_star = get_r2_star(field_strength)
    estimated_residual = baseline_delta_r2 / r2_star if baseline_delta_r2 > 0 else None

    # Compute confidence based on signal statistics
    # Higher confidence when signal distribution is consistent
    signal_cv = float(np.std(baseline_values) / mean_baseline) if mean_baseline > 0 else 1.0
    confidence = max(0.0, min(1.0, 1.0 - signal_cv))

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
            "signal_cv": signal_cv,
            "field_strength": field_strength,
            "n_brain_voxels": int(baseline_values.size),
        },
    )


def detect_preload_from_dce_protocol(
    dce_ran: bool,
    dce_end_time: Optional[float] = None,
    dsc_start_time: Optional[float] = None,
    dce_dose: str = "full",
) -> PreloadResult:
    """Detect preload from DCE protocol information.

    Uses knowledge of prior DCE acquisition to estimate preload.

    Args:
        dce_ran: Whether DCE was acquired before DSC.
        dce_end_time: Time when DCE acquisition ended (seconds from session start).
        dsc_start_time: Time when DSC acquisition started.
        dce_dose: Dose used for DCE ("half", "full", "double").

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
    dose_factor = {"half": 0.5, "full": 1.0, "double": 2.0}.get(dce_dose.lower(), 1.0)

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
        )
        results.append(result)

    # Method 2: DCE protocol
    if dce_ran is not None:
        dce_info = dce_metadata or {}
        result = detect_preload_from_dce_protocol(
            dce_ran=dce_ran,
            dce_end_time=dce_info.get("end_time"),
            dsc_start_time=dce_info.get("dsc_start_time"),
            dce_dose=dce_info.get("dose", "full"),
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

    # Sort by confidence, return highest
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results[0]


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
