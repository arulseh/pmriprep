"""Bolus Arrival Time (BAT) detection for DCE-MRI.

Detects the time of contrast agent arrival using threshold and gradient
methods, with a default fallback.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

#: Largest plausible AIF-to-tissue bolus transit delay (seconds). Beyond this,
#: the estimate is treated as a detection failure and no shift is applied.
MAX_AIF_TISSUE_DELAY_S: float = 12.0
#: A delay below this fraction of the volume TR is below temporal resolution
#: (sub-frame) and is not applied — at coarse TR, sub-frame "delays" are
#: quantization noise that would harm test-retest reproducibility.
AIF_DELAY_SUBFRAME_FRACTION: float = 0.25


@dataclass
class BATResult:
    """Result of bolus arrival time detection.

    Attributes
    ----------
    bat_seconds : float
        Bolus arrival time in seconds.
    peak_seconds : float
        Time of peak enhancement in seconds.
    confidence : str
        Detection confidence ("high" or "low").
    method : str
        Method used ("threshold", "gradient", or "default").
    """

    bat_seconds: float
    peak_seconds: float
    confidence: str
    method: str


def detect_bolus_arrival_time(
    curve: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    baseline_frames: int = 5,
) -> BATResult:
    """Detect bolus arrival time using threshold then gradient fallback.

    Methods attempted in order:
    1. Threshold: first time concentration exceeds baseline + 3*std
    2. Gradient: first time the derivative exceeds 3*std(derivative_baseline)
    3. Default: 10% of total acquisition time

    Uses adaptive baseline frame count and MAD-based outlier
    detection to handle frame 0 artifacts.

    Parameters
    ----------
    curve : numpy.ndarray
        1D concentration time course (mM).
    time_points : numpy.ndarray
        1D time array in minutes.
    baseline_frames : int, optional
        Number of pre-contrast baseline frames.

    Returns
    -------
    BATResult
        BATResult with detected BAT and metadata.
    """
    n = len(curve)
    if n < baseline_frames + 2:
        return BATResult(
            bat_seconds=0.0,
            peak_seconds=0.0,
            confidence="low",
            method="default",
        )

    # Convert time to seconds
    time_sec = time_points * 60.0
    peak_idx = int(np.argmax(curve))
    peak_seconds = float(time_sec[peak_idx])

    # Adaptive baseline frame count
    n_baseline = min(baseline_frames, n // 4)
    n_baseline = max(2, n_baseline)

    baseline = curve[:n_baseline]

    # Outlier detection using MAD (robust to artifacts like frame 0 spikes)
    # MAD = median absolute deviation, converted to std-equivalent via 1.4826
    median_bl = np.median(baseline)
    mad = np.median(np.abs(baseline - median_bl))
    if mad > 1e-9:
        z_scores = np.abs(baseline - median_bl) / (1.4826 * mad)
        valid_mask = z_scores < 3.0
        if np.sum(valid_mask) >= 2:
            baseline = baseline[valid_mask]

    bl_mean = float(np.mean(baseline))
    bl_std = float(np.std(baseline))

    # Method 1: Threshold detection
    # Start search after baseline frames (use n_baseline for consistency)
    threshold = bl_mean + 3.0 * max(bl_std, 1e-9)
    for i in range(n_baseline, n):
        if curve[i] > threshold:
            return BATResult(
                bat_seconds=float(time_sec[i]),
                peak_seconds=peak_seconds,
                confidence="high",
                method="threshold",
            )

    # Method 2: Gradient detection
    if n > n_baseline + 1:
        dt = float(time_sec[1] - time_sec[0])
        if dt > 0:
            gradient = np.diff(curve) / dt
            grad_baseline = gradient[:max(n_baseline - 1, 1)]
            grad_std = float(np.std(grad_baseline))
            grad_threshold = 3.0 * max(grad_std, 1e-9)

            for i in range(n_baseline, len(gradient)):
                if gradient[i] > grad_threshold:
                    return BATResult(
                        bat_seconds=float(time_sec[i]),
                        peak_seconds=peak_seconds,
                        confidence="high",
                        method="gradient",
                    )

    # Method 3: Default fallback (10% of total time)
    total_time = float(time_sec[-1] - time_sec[0])
    default_bat = float(time_sec[0]) + total_time * 0.1

    return BATResult(
        bat_seconds=default_bat,
        peak_seconds=peak_seconds,
        confidence="low",
        method="default",
    )


def shift_aif(
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    delay_seconds: float,
) -> npt.NDArray[np.float64]:
    """Time-shift an AIF curve by a delay via linear interpolation.

    Aligns the AIF to delayed tissue by resampling it at ``t - delay``:
    ``aif_aligned(t) = aif(t - delay)``. A positive ``delay_seconds`` shifts
    the AIF later in time, so its rise meets a tissue curve that enhances
    after the sampling artery (the expected sign for a contralateral AIF).

    Pre-acquisition query times (``t - delay`` below the first sample) are
    filled with the AIF baseline value ``aif[0]`` (not zero), so no spurious
    negative ramp or acausal signal is introduced; query times past the last
    sample are filled with ``aif[-1]``.

    Parameters
    ----------
    aif : numpy.ndarray
        1D AIF plasma concentration curve (mM).
    time_points : numpy.ndarray
        1D time array in minutes (the axis the PK models convolve on). May be
        non-uniform.
    delay_seconds : float
        Delay in seconds. Zero returns the AIF unchanged.

    Returns
    -------
    numpy.ndarray
        The time-shifted AIF, same shape as ``aif``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(10) * 0.1  # minutes
    >>> aif = np.array([0, 0, 0, 1.0, 2, 1, 0.5, 0.4, 0.3, 0.2])
    >>> shifted = shift_aif(aif, t, delay_seconds=6.0)  # +1 frame (0.1 min)
    >>> bool(shifted[4] == aif[3])
    True
    """
    arr: npt.NDArray[np.float64] = np.asarray(aif, dtype=np.float64)
    if delay_seconds == 0.0 or arr.shape[0] < 2:
        unchanged: npt.NDArray[np.float64] = arr.copy()
        return unchanged
    delay_minutes = delay_seconds / 60.0
    query = np.asarray(time_points, dtype=np.float64) - delay_minutes
    shifted: npt.NDArray[np.float64] = np.interp(
        query, time_points, arr, left=float(arr[0]), right=float(arr[-1]),
    )
    return shifted


def compute_aif_tissue_delay(
    aif_curve: npt.NDArray[np.floating],
    tissue_roi_curve: npt.NDArray[np.floating],
    time_points_min: npt.NDArray[np.floating],
    tr_seconds: float,
    baseline_frames: int = 5,
) -> tuple[float, dict[str, Any]]:
    """Compute the guarded AIF-to-tissue bolus-arrival delay, in seconds.

    The delay is ``tissue_BAT - AIF_BAT``, where both arrival times come from
    :func:`detect_bolus_arrival_time` applied to the arterial AIF curve and an
    aggregate tissue ROI curve respectively, on the same acquisition time grid
    (so the difference is a true common-clock transit interval). The arterial
    AIF curve must be the measured AIF — never a brain-mean surrogate, which
    measures tissue transit rather than arterial arrival.

    A series of physics guards return ``0.0`` (no shift) whenever the delay is
    unreliable or below what the temporal resolution can resolve, so applying
    the result is always safe.

    Parameters
    ----------
    aif_curve : numpy.ndarray
        1D measured arterial input function curve.
    tissue_roi_curve : numpy.ndarray
        1D aggregate (e.g. median) tissue concentration curve over the ROI.
    time_points_min : numpy.ndarray
        1D time array in minutes (passed to ``detect_bolus_arrival_time``,
        which returns seconds).
    tr_seconds : float
        Volume TR (temporal resolution between dynamics) in seconds. Sets the
        sub-frame no-op threshold; this is the volume TR, not the pulse TR.
    baseline_frames : int, optional
        Number of pre-contrast baseline frames.

    Returns
    -------
    tuple of (float, dict)
        ``(delay_seconds, provenance)``. ``delay_seconds`` is ``0.0`` when no
        shift should be applied; ``provenance`` records both arrival times,
        the raw delay, whether it was applied, and the reason.
    """
    aif_bat = detect_bolus_arrival_time(aif_curve, time_points_min, baseline_frames)
    tissue_bat = detect_bolus_arrival_time(
        tissue_roi_curve, time_points_min, baseline_frames,
    )
    delay = float(tissue_bat.bat_seconds - aif_bat.bat_seconds)
    tr_s = float(tr_seconds)
    subframe = AIF_DELAY_SUBFRAME_FRACTION * tr_s
    # Delay expressed in frames makes the frame-quantization auditable: a
    # 1-frame shift reads as ~1.0, which is what drives coarse-TR per-visit
    # inconsistency.
    delay_frames = round(delay / tr_s, 2) if tr_s > 0 else 0.0

    prov: dict[str, Any] = {
        "bat_aif_seconds": round(aif_bat.bat_seconds, 4),
        "bat_tissue_seconds": round(tissue_bat.bat_seconds, 4),
        "bat_aif_method": aif_bat.method,
        "bat_tissue_method": tissue_bat.method,
        "raw_delay_seconds": round(delay, 4),
        "raw_delay_frames": delay_frames,
    }

    def _noop(reason: str) -> tuple[float, dict[str, Any]]:
        return 0.0, {**prov, "applied": False, "reason": reason}

    # A guessed (default-method) or low-confidence arrival on either curve
    # makes the delay sign meaningless — never shift on it.
    if (
        aif_bat.confidence == "low"
        or aif_bat.method == "default"
        or tissue_bat.confidence == "low"
        or tissue_bat.method == "default"
    ):
        return _noop("low_bat_confidence")
    # A contralateral AIF should not arrive after the tissue; a clearly
    # negative delay signals a detection failure.
    if delay < -subframe:
        return _noop("negative_delay_rejected")
    # Beyond the physiological ceiling, treat as a detection failure.
    if abs(delay) > MAX_AIF_TISSUE_DELAY_S:
        return _noop("delay_clamped")
    # Sub-frame delay is below temporal resolution — applying it would inject
    # quantization noise (the documented coarse-TR repeatability hazard).
    if abs(delay) < subframe:
        return _noop("delay_below_resolution")

    return delay, {
        **prov, "applied": True, "reason": "applied",
        "delay_seconds": round(delay, 4),
        "delay_frames": delay_frames,
    }
