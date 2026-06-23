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
#: Leading-edge level, as a fraction of peak enhancement above baseline, at
#: which the sub-frame arrival time is read. Anchoring at a fixed fraction of
#: each curve's own peak samples the AIF and tissue at corresponding points on
#: the bolus, so a shared timing bias cancels in the tissue-minus-AIF
#: difference. The 0.10 fraction is a pipeline design choice (low enough to
#: catch the leading edge, above the 3-sigma noise the gate already enforces).
BAT_ARRIVAL_PEAK_FRACTION: float = 0.10
#: Physical lower bound (seconds) on a resolvable arterial-to-tissue transit,
#: combined with AIF_DELAY_SUBFRAME_FRACTION * TR into a hybrid floor. This is a
#: soft-ramp floor on the applied shift, not a hard reject threshold.
MIN_AIF_TISSUE_DELAY_S: float = 1.5
#: Top of the soft-ramp band as a multiple of the floor: the applied shift ramps
#: linearly from zero at the floor to full magnitude at this multiple of it.
BAT_RAMP_BAND_FACTOR: float = 2.0


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


def _interp_crossing(
    time_sec: npt.NDArray[np.floating],
    series: npt.NDArray[np.floating],
    i: int,
    level: float,
) -> float:
    """Linearly interpolate the sub-frame time at which ``series`` crosses ``level``.

    The crossing is located between frames ``i-1`` and ``i`` (where ``series[i]``
    is the first sample at or above ``level``) by linear interpolation of the
    rising segment, resolving the arrival to a fraction of a frame rather than
    snapping it to the frame grid.

    Parameters
    ----------
    time_sec : numpy.ndarray
        1D time array in seconds.
    series : numpy.ndarray
        1D series whose ``level`` crossing is being located (concentration for
        the threshold method, derivative for the gradient method).
    i : int
        Index of the first frame at or above ``level``.
    level : float
        Crossing level.

    Returns
    -------
    float
        Sub-frame crossing time in seconds. Falls back to ``time_sec[i]`` (the
        frame-quantized crossing) when no prior frame exists (``i <= 0``) or the
        segment is non-rising/degenerate (``series[i] - series[i-1] <= 0``).

    Notes
    -----
    The interpolation fraction is clamped to ``[0, 1]`` so the result never
    precedes ``time_sec[i-1]`` nor follows ``time_sec[i]``.
    """
    if i <= 0:
        return float(time_sec[i])
    denom = float(series[i] - series[i - 1])
    if denom <= 0.0:
        return float(time_sec[i])
    frac = (float(level) - float(series[i - 1])) / denom
    frac = min(1.0, max(0.0, frac))
    return float(time_sec[i - 1]) + frac * float(time_sec[i] - time_sec[i - 1])


def detect_bolus_arrival_time(
    curve: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    baseline_frames: int = 5,
) -> BATResult:
    """Detect bolus arrival time using threshold then gradient fallback.

    Methods attempted in order:
    1. Threshold: the peak must clear baseline + 3*std (genuine enhancement);
       the arrival is then read at ``BAT_ARRIVAL_PEAK_FRACTION`` of the peak on
       the leading edge, interpolated to sub-frame precision.
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

    # Method 1: Threshold detection with sub-frame, peak-relative timing.
    # The peak must clear the noise floor (genuine enhancement) and fall after
    # the baseline window; the arrival is then read at a fixed fraction of the
    # peak on the leading edge, interpolated to sub-frame precision.
    noise = 3.0 * max(bl_std, 1e-9)
    peak_enh = float(curve[peak_idx] - bl_mean)
    if peak_enh > noise and peak_idx >= n_baseline:
        arrival_level = bl_mean + BAT_ARRIVAL_PEAK_FRACTION * peak_enh
        for j in range(n_baseline, peak_idx + 1):
            if curve[j] >= arrival_level:
                return BATResult(
                    bat_seconds=_interp_crossing(time_sec, curve, j, arrival_level),
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
        ``(applied_seconds, provenance)``. ``applied_seconds`` is the shift to
        apply — the soft-ramped delay, or ``0.0`` when no shift should be
        applied. ``provenance`` records both arrival times, whether their
        detection methods match, the raw delay (seconds and frames), the
        resolution floor, whether the applied shift was attenuated by the ramp,
        the applied shift, and the reason.
    """
    aif_bat = detect_bolus_arrival_time(aif_curve, time_points_min, baseline_frames)
    tissue_bat = detect_bolus_arrival_time(
        tissue_roi_curve, time_points_min, baseline_frames,
    )
    delay = float(tissue_bat.bat_seconds - aif_bat.bat_seconds)
    tr_s = float(tr_seconds)
    # Hybrid floor: the larger of a fraction of the volume TR and a physical
    # lower bound on a resolvable transit. The physical term holds the floor at
    # fast TR, where a fraction of TR alone would fall below what a sub-frame
    # leading-edge estimate can resolve.
    floor = max(AIF_DELAY_SUBFRAME_FRACTION * tr_s, MIN_AIF_TISSUE_DELAY_S)
    # Delay in frames makes the quantization auditable: a ~1-frame raw delay
    # reads as ~1.0, which is what drove the coarse-TR per-visit inconsistency.
    raw_delay_frames = round(delay / tr_s, 2) if tr_s > 0 else 0.0

    prov: dict[str, Any] = {
        "bat_aif_seconds": round(aif_bat.bat_seconds, 4),
        "bat_tissue_seconds": round(tissue_bat.bat_seconds, 4),
        "bat_aif_method": aif_bat.method,
        "bat_tissue_method": tissue_bat.method,
        "methods_match": aif_bat.method == tissue_bat.method,
        "raw_delay_seconds": round(delay, 4),
        "raw_delay_frames": raw_delay_frames,
        "floor_seconds": round(floor, 4),
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
    # The comparators below tile the no-op, ramp, and full bands without gap or
    # overlap. A contralateral AIF should not arrive after the tissue; a clearly
    # negative delay signals a detection failure.
    if delay < -floor:
        return _noop("negative_delay_rejected")
    # Beyond the physiological ceiling, treat as a detection failure.
    if abs(delay) > MAX_AIF_TISSUE_DELAY_S:
        return _noop("delay_clamped")
    # At or below the floor the delay is below what the sub-frame estimate can
    # resolve — applying it would inject noise (the coarse-TR repeatability
    # hazard).
    if abs(delay) <= floor:
        return _noop("delay_below_resolution")

    # Soft ramp between the floor and twice the floor: attenuate the applied
    # shift linearly from zero to full magnitude so the correction is continuous
    # in the estimated delay. A hard cutoff would reintroduce a per-visit
    # zero-or-full discontinuity at the boundary. The formula is sign-preserving,
    # though negative in-band delays are already rejected above.
    band_top = BAT_RAMP_BAND_FACTOR * floor
    if abs(delay) < band_top:
        applied = delay * (abs(delay) - floor) / floor
        attenuated = True
    else:
        applied = delay
        attenuated = False
    applied_frames = round(applied / tr_s, 2) if tr_s > 0 else 0.0

    # delay_seconds/delay_frames are the back-compat alias of the applied shift;
    # applied_delay_* is the self-documenting name (keep both — existing
    # consumers and tests read delay_frames).
    return applied, {
        **prov,
        "applied": True,
        "reason": "applied",
        "attenuated": attenuated,
        "delay_seconds": round(applied, 4),
        "delay_frames": applied_frames,
        "applied_delay_seconds": round(applied, 4),
        "applied_delay_frames": applied_frames,
    }
