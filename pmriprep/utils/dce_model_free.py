"""Model-free DCE maps.

Computes model-assumption-free quantitative maps from DCE signal and concentration data:
IAUC, peak enhancement, time-to-peak, signal enhancement ratio, wash-in slope,
wash-out slope, bolus arrival time, and curve type classification.
"""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# Require sampled enhancement to exceed precontrast temporal noise.
_SER_MIN_ENHANCEMENT_NOISE_SD = 3.0

# TTP selects a maximum over the post-reference series, so use a conservative
# noise gate rather than the fixed-time SER threshold.
_TTP_MIN_PEAK_NOISE_SD = 5.0

def compute_iauc(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    cutoff_s: float,
    bolus_arrival_frame: int = 0,
) -> npt.NDArray[np.floating]:
    """Compute Initial Area Under Curve via trapezoidal integration.

    Integrates concentration from the selected global IAUC reference frame to
    exactly ``cutoff_s`` seconds later. The concentration at a cutoff between
    acquired volumes is linearly interpolated.

    Parameters
    ----------
    concentration : numpy.ndarray
        4D concentration data (x, y, z, time) in mM.
    time_points : numpy.ndarray
        1D time array in minutes.
    cutoff_s : float
        Integration cutoff in seconds.
    bolus_arrival_frame : int, optional
        Frame defining the start of the integration window.

    Returns
    -------
    numpy.ndarray
        3D IAUC map (mM·min). Returns NaN when the requested interval is not
        fully acquired or its samples are non-finite.
    """
    if cutoff_s <= 0:
        raise ValueError("cutoff_s must be positive")
    if time_points.ndim != 1 or concentration.shape[-1] != time_points.size:
        raise ValueError("time_points must be 1D and match the concentration time axis")
    if not 0 <= bolus_arrival_frame < time_points.size:
        raise ValueError("bolus_arrival_frame is outside the time axis")
    if not np.all(np.isfinite(time_points)) or np.any(np.diff(time_points) <= 0):
        raise ValueError("time_points must be finite and strictly increasing")

    start_min = float(time_points[bolus_arrival_frame])
    cutoff_min = start_min + cutoff_s / 60.0
    final_min = float(time_points[-1])
    time_tolerance_min = 1e-9
    if cutoff_min > final_min + time_tolerance_min:
        return np.full(concentration.shape[:3], np.nan, dtype=np.float64)
    cutoff_min = min(cutoff_min, final_min)

    end_index = int(np.searchsorted(time_points, cutoff_min, side="left"))
    t_valid = time_points[bolus_arrival_frame : end_index + 1].copy()
    c_valid = concentration[..., bolus_arrival_frame : end_index + 1].copy()
    if time_points[end_index] != cutoff_min:
        lower = end_index - 1
        weight = (cutoff_min - time_points[lower]) / (
            time_points[end_index] - time_points[lower]
        )
        c_valid[..., -1] = (
            concentration[..., lower]
            + weight * (concentration[..., end_index] - concentration[..., lower])
        )
        t_valid[-1] = cutoff_min

    # Trapezoidal integration along last axis
    iauc = np.trapezoid(c_valid, x=t_valid, axis=-1)
    valid = np.all(np.isfinite(c_valid), axis=-1)
    return np.where(valid, iauc, np.nan)  # type: ignore[no-any-return]


def compute_peak_enhancement(
    signal_data: npt.NDArray[np.floating],
    baseline_frames: int = 5,
) -> npt.NDArray[np.floating]:
    """Compute peak enhancement percentage from raw signal data.

    (S_max,post - S_baseline) / S_baseline × 100

    Must be computed on **raw MRI signal**, not concentration.
    Concentration has baseline ~0 by definition, causing division by
    near-zero and overflow.

    Parameters
    ----------
    signal_data : numpy.ndarray
        4D raw signal data (x, y, z, time).
    baseline_frames : int, optional
        Number of pre-contrast baseline frames.

    Returns
    -------
    numpy.ndarray
        3D peak enhancement percentage map. Voxels with a non-positive
        baseline or any non-finite signal samples are NaN.
    """
    if baseline_frames < 1 or baseline_frames >= signal_data.shape[-1]:
        raise ValueError("baseline_frames must leave at least one post-baseline frame")
    baseline_mean = np.mean(signal_data[..., :baseline_frames], axis=-1)
    peak = np.max(signal_data[..., baseline_frames:], axis=-1)
    valid = np.all(np.isfinite(signal_data), axis=-1) & (baseline_mean > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        enhancement = (peak - baseline_mean) / baseline_mean * 100.0

    return np.asarray(np.where(valid, enhancement, np.nan), dtype=np.float64)


def compute_time_to_peak(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    reference_frame: int = 0,
    baseline_frames: int = 5,
    brain_mask: Optional[npt.NDArray[np.floating]] = None,
) -> npt.NDArray[np.floating]:
    """Compute tissue time to peak from a selected global reference.

    Parameters
    ----------
    concentration : numpy.ndarray
        4D concentration data (x, y, z, time) in mM.
    time_points : numpy.ndarray
        1D time array in minutes.
    reference_frame : int, optional
        Selected global contrast-reference frame. Peak searching begins here.
    baseline_frames : int, optional
        Number of precontrast frames used to estimate concentration noise.
    brain_mask : numpy.ndarray, optional
        Brain mask used to pool a robust spatial noise floor. When omitted,
        all spatial voxels contribute.

    Returns
    -------
    numpy.ndarray
        3D reference-relative time-to-peak map in seconds. Voxels with
        non-finite samples or peak enhancement not exceeding five baseline
        noise standard deviations are NaN.
    """
    if time_points.ndim != 1 or concentration.shape[-1] != time_points.size:
        raise ValueError("time_points must be 1D and match the concentration time axis")
    if not np.all(np.isfinite(time_points)) or np.any(np.diff(time_points) <= 0):
        raise ValueError("time_points must be finite and strictly increasing")
    if not 0 <= reference_frame < time_points.size:
        raise ValueError("reference_frame is outside the time axis")
    if baseline_frames < 1 or baseline_frames >= time_points.size:
        raise ValueError("baseline_frames must leave at least one post-baseline frame")
    if brain_mask is not None and brain_mask.shape != concentration.shape[:3]:
        raise ValueError("brain_mask must match the concentration spatial shape")

    undefined = np.full(concentration.shape[:3], np.nan, dtype=np.float64)
    if baseline_frames < 2:
        return undefined

    baseline = concentration[..., :baseline_frames]
    baseline_mean = np.mean(baseline, axis=-1)
    baseline_noise = np.std(baseline, axis=-1, ddof=1)
    noise_support = np.isfinite(baseline_noise)
    if brain_mask is not None:
        noise_support &= brain_mask > 0
    pooled_values = baseline_noise[noise_support]
    pooled_noise = (
        float(np.median(pooled_values)) if pooled_values.size else 0.0
    )
    effective_noise = np.maximum(baseline_noise, pooled_noise)
    post_reference = concentration[..., reference_frame:]
    peak_indices = np.argmax(post_reference, axis=-1) + reference_frame
    peak = np.take_along_axis(
        concentration, peak_indices[..., np.newaxis], axis=-1,
    )[..., 0]
    scale = np.maximum(np.abs(baseline_mean), np.abs(peak))
    threshold = np.maximum(
        _TTP_MIN_PEAK_NOISE_SD * effective_noise,
        16.0 * np.finfo(np.float64).eps * scale,
    )
    valid = (
        np.all(np.isfinite(concentration), axis=-1)
        & np.isfinite(baseline_noise)
        & ((peak - baseline_mean) > threshold)
    )
    ttp_seconds = (time_points[peak_indices] - time_points[reference_frame]) * 60.0
    return np.where(valid, ttp_seconds, np.nan)  # type: ignore[no-any-return]


def compute_signal_enhancement_ratio(
    signal_data: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    bolus_arrival_frame: int = 0,
    baseline_frames: int = 5,
    early_time_s: float = 120.0,
    late_time_s: float = 320.0,
) -> npt.NDArray[np.floating]:
    """Compute signal enhancement ratio at fixed post-reference times.

    ``SER = (S_early - S0) / (S_late - S0)``

    Signal values at the predefined early and late times are linearly
    interpolated. The stored ratio is not clipped. Early enhancement and the
    absolute late enhancement must both exceed three precontrast noise standard
    deviations. A negative late enhancement is retained when it clears that
    validity threshold.

    Parameters
    ----------
    signal_data : numpy.ndarray
        4D motion-corrected DCE signal data (x, y, z, time).
    time_points : numpy.ndarray
        1D time array in minutes.
    bolus_arrival_frame : int, optional
        Frame index of bolus arrival.
    baseline_frames : int, optional
        Number of pre-contrast frames used to estimate S0.
    early_time_s : float, optional
        Early sample time in seconds after the selected global reference.
    late_time_s : float, optional
        Late sample time in seconds after the selected global reference.

    Returns
    -------
    numpy.ndarray
        3D SER map. Undefined voxels are NaN.
    """
    if time_points.ndim != 1 or signal_data.shape[-1] != time_points.size:
        raise ValueError("time_points must be 1D and match the signal time axis")
    undefined = np.full(signal_data.shape[:3], np.nan, dtype=np.float64)
    if baseline_frames < 1 or baseline_frames >= signal_data.shape[-1]:
        raise ValueError("baseline_frames must leave at least one post-baseline frame")
    if baseline_frames < 2:
        return undefined
    if not 0 <= bolus_arrival_frame < time_points.size:
        raise ValueError("bolus_arrival_frame is outside the time axis")
    if not np.all(np.isfinite(time_points)) or np.any(np.diff(time_points) <= 0):
        raise ValueError("time_points must be finite and strictly increasing")
    if not np.isfinite(early_time_s) or early_time_s <= 0:
        raise ValueError("early_time_s must be finite and positive")
    if not np.isfinite(late_time_s) or late_time_s <= early_time_s:
        raise ValueError("late_time_s must be finite and greater than early_time_s")

    reference_min = float(time_points[bolus_arrival_frame])
    sample_times = reference_min + np.asarray([early_time_s, late_time_s]) / 60.0
    if sample_times[-1] > float(time_points[-1]) + 1e-9:
        return undefined
    sample_times = np.minimum(sample_times, float(time_points[-1]))

    def _sample(sample_time: float) -> npt.NDArray[np.floating]:
        upper = int(np.searchsorted(time_points, sample_time, side="left"))
        if upper == 0 or np.isclose(time_points[upper], sample_time, rtol=0.0, atol=1e-9):
            return np.asarray(signal_data[..., upper], dtype=np.float64)
        lower = upper - 1
        weight = (sample_time - time_points[lower]) / (
            time_points[upper] - time_points[lower]
        )
        return np.asarray(
            signal_data[..., lower] + weight * (
                signal_data[..., upper] - signal_data[..., lower]
            ),
            dtype=np.float64,
        )

    baseline_samples = signal_data[..., :baseline_frames]
    baseline = np.mean(baseline_samples, axis=-1)
    baseline_noise = np.std(baseline_samples, axis=-1, ddof=1)
    early_signal = _sample(float(sample_times[0]))
    late_signal = _sample(float(sample_times[1]))
    early_enhancement = early_signal - baseline
    late_enhancement = late_signal - baseline
    scale = np.maximum.reduce((np.abs(baseline), np.abs(early_signal), np.abs(late_signal)))
    numerical_tolerance = 16.0 * np.finfo(np.float64).eps * scale
    enhancement_threshold = np.maximum(
        _SER_MIN_ENHANCEMENT_NOISE_SD * baseline_noise,
        numerical_tolerance,
    )
    valid = (
        np.isfinite(baseline)
        & np.isfinite(baseline_noise)
        & np.isfinite(early_enhancement)
        & np.isfinite(late_enhancement)
        & (early_enhancement > enhancement_threshold)
        & (np.abs(late_enhancement) > enhancement_threshold)
    )
    ser = undefined.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ser[valid] = early_enhancement[valid] / late_enhancement[valid]

    return ser  # type: ignore[no-any-return]


def compute_wash_in_slope(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    bolus_arrival_frame: int = 0,
    volume_tr_s: float | None = None,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> npt.NDArray[np.floating]:
    """Compute maximum upslope during wash-in phase.

    Uses Savitzky-Golay smoothing followed by finite differences to estimate
    the maximum rate of concentration increase between bolus arrival and peak.

    Parameters
    ----------
    concentration : numpy.ndarray
        4D concentration data (x, y, z, time) in mM.
    time_points : numpy.ndarray
        1D time array in minutes.
    bolus_arrival_frame : int, optional
        Frame index of bolus arrival.
    volume_tr_s : float or None, optional
        Volume TR in seconds (for temporal resolution gating).
    savgol_window : int, optional
        Savitzky-Golay filter window length.
    savgol_polyorder : int, optional
        Savitzky-Golay polynomial order.

    Returns
    -------
    numpy.ndarray
        3D wash-in slope map (mM/min). NaN where slope is unreliable.
    """
    spatial_shape = concentration.shape[:3]
    n_timepoints = concentration.shape[-1]

    # TR gating
    if volume_tr_s is not None:
        if volume_tr_s > 15.0:
            logger.error(
                "Volume TR %.1fs exceeds 15s hard gate — wash-in slope "
                "unreliable, returning NaN map",
                volume_tr_s,
            )
            return np.full(spatial_shape, np.nan, dtype=np.float64)
        if volume_tr_s > 5.0:
            logger.warning(
                "Volume TR %.1fs exceeds 5s — wash-in slope may be unreliable",
                volume_tr_s,
            )

    # Savitzky-Golay smoothing (operate on copy)
    smoothed = concentration.copy()
    window = min(savgol_window, n_timepoints // 2 * 2 + 1)
    if window < 3 or window <= savgol_polyorder:
        logger.warning(
            "Too few timepoints (%d) for Savitzky-Golay smoothing — "
            "computing wash-in slope from unsmoothed data",
            n_timepoints,
        )
    if window >= 3 and window > savgol_polyorder:
        # Ensure window is odd
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            smoothed = savgol_filter(
                smoothed, window_length=window, polyorder=savgol_polyorder, axis=-1,
            )
            # Clip negative overshoots from SG filtering
            np.maximum(smoothed, 0.0, out=smoothed)

    # Finite differences: derivative = dC/dt (mM/min)
    dt = np.diff(time_points)  # (n_timepoints-1,)
    dt_safe = np.where(np.abs(dt) > 1e-9, dt, np.nan)
    derivative = np.diff(smoothed, axis=-1) / dt_safe  # (..., n_timepoints-1)

    # Vectorized pre-peak masking
    peak_indices = np.argmax(concentration, axis=-1)  # (x, y, z)
    n_deriv = derivative.shape[-1]
    frame_idx = np.arange(n_deriv)

    # True where frame >= bolus_arrival AND frame < peak
    pre_peak_mask = (frame_idx >= bolus_arrival_frame) & (
        frame_idx < peak_indices[..., np.newaxis]
    )

    # Max derivative in pre-peak window
    derivative_masked = np.where(pre_peak_mask, derivative, -np.inf)
    slope = np.max(derivative_masked, axis=-1)

    # NaN where fewer than 2 valid pre-peak frames
    n_pre_peak = np.sum(pre_peak_mask, axis=-1)
    slope = np.where(n_pre_peak >= 2, slope, np.nan)

    return slope  # type: ignore[no-any-return]


def compute_wash_out_slope(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    volume_tr_s: float | None = None,
    post_peak_start_s: float = 30.0,
    post_peak_end_s: float = 120.0,
    min_points: int = 4,
) -> npt.NDArray[np.floating]:
    """Compute linear washout slope via least-squares regression.

    Uses a TTP-relative window (per-voxel) for regression. Fully vectorized
    via 4D boolean mask broadcasting.

    Parameters
    ----------
    concentration : numpy.ndarray
        4D concentration data (x, y, z, time) in mM.
    time_points : numpy.ndarray
        1D time array in minutes.
    volume_tr_s : float or None, optional
        Volume TR in seconds. If None, estimated from time_points.
    post_peak_start_s : float, optional
        Start of washout window relative to TTP (seconds).
    post_peak_end_s : float, optional
        End of washout window relative to TTP (seconds).
    min_points : int, optional
        Minimum points required for regression.

    Returns
    -------
    numpy.ndarray
        3D wash-out slope map (mM/min, signed: negative=washout).
        NaN where insufficient data points.
    """
    n_time = concentration.shape[-1]

    ttp_idx = np.argmax(concentration, axis=-1)  # (x, y, z)

    # Determine volume TR
    if volume_tr_s is not None:
        vol_tr = volume_tr_s
    else:
        vol_tr = float(time_points[1] - time_points[0]) * 60.0

    # Build per-voxel boolean mask via broadcasting
    time_idx = np.arange(n_time)
    frames_post_peak = time_idx[np.newaxis, np.newaxis, np.newaxis, :] - ttp_idx[..., np.newaxis]
    seconds_post_peak = frames_post_peak * vol_tr
    washout_mask = (seconds_post_peak >= post_peak_start_s) & (
        seconds_post_peak <= post_peak_end_s
    )

    n_valid = washout_mask.sum(axis=-1)

    # Batched linear regression via normal equations
    t_broadcast = np.broadcast_to(time_points, concentration.shape)
    t_masked = np.where(washout_mask, t_broadcast, 0.0)
    c_masked = np.where(washout_mask, concentration, 0.0)

    n_safe = np.maximum(n_valid, 1).astype(np.float64)
    t_mean = t_masked.sum(axis=-1) / n_safe
    c_mean = c_masked.sum(axis=-1) / n_safe

    t_dev = np.where(washout_mask, t_broadcast - t_mean[..., np.newaxis], 0.0)
    c_dev = np.where(washout_mask, concentration - c_mean[..., np.newaxis], 0.0)
    numerator = (t_dev * c_dev).sum(axis=-1)
    denominator = (t_dev**2).sum(axis=-1)

    denom_safe = np.where(np.abs(denominator) > 1e-12, denominator, np.nan)
    slope = numerator / denom_safe
    slope = np.where(n_valid >= min_points, slope, np.nan)

    return slope  # type: ignore[no-any-return]


def compute_curve_type(
    ser: npt.NDArray[np.floating],
    persistent_threshold: float = 0.9,
    washout_threshold: float = 1.1,
) -> npt.NDArray[np.integer]:
    """Classify DCE enhancement curves based on SER values.

    Three descriptive classes are emitted for meaningfully enhancing voxels:
    persistent-like, plateau-like, and washout-like. These labels do not imply
    an organ-specific diagnostic standard.

    Parameters
    ----------
    ser : numpy.ndarray
        3D Signal Enhancement Ratio map.
    persistent_threshold : float, optional
        SER below this is persistent (default 0.9).
    washout_threshold : float, optional
        SER above this is washout (default 1.1).

    Returns
    -------
    numpy.ndarray
        3D int8 map: 0=unclassified (NaN SER), 1=persistent,
        2=plateau, 3=washout.
    """
    curve_type = np.zeros(ser.shape, dtype=np.int8)
    valid = np.isfinite(ser)
    curve_type[valid & (ser < persistent_threshold)] = 1
    curve_type[valid & (ser >= persistent_threshold) & (ser <= washout_threshold)] = 2
    curve_type[valid & (ser > washout_threshold)] = 3
    return curve_type
