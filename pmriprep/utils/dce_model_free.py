"""Model-free DCE maps (QIBA-recommended).

Computes model-assumption-free quantitative maps from DCE concentration data:
IAUC, peak enhancement, time-to-peak, signal enhancement ratio, wash-in slope,
wash-out slope, bolus arrival time, and curve type classification.
"""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# Signal below this is dead/background (air, outside FOV).
# Real tissue signal is hundreds-thousands.
_PEAK_ENH_SIGNAL_FLOOR = 1.0

# No tissue produces >5000% signal enhancement; anything above is
# numerical artifact from near-zero baseline voxels.
_PEAK_ENH_MAX_CAP = 5000.0


def compute_iauc(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    cutoff_s: float,
) -> npt.NDArray[np.floating]:
    """Compute Initial Area Under Curve via trapezoidal integration.

    Integrates concentration from t=0 up to cutoff_s seconds.

    Args:
        concentration: 4D concentration data (x, y, z, time) in mM.
        time_points: 1D time array in minutes.
        cutoff_s: Integration cutoff in seconds.

    Returns:
        3D IAUC map (mM·min).
    """
    cutoff_min = cutoff_s / 60.0
    valid_mask = time_points <= cutoff_min

    if np.sum(valid_mask) < 2:
        return np.zeros(concentration.shape[:3], dtype=np.float64)

    t_valid = time_points[valid_mask]
    c_valid = concentration[..., valid_mask]

    # Trapezoidal integration along last axis
    iauc = np.trapezoid(c_valid, x=t_valid, axis=-1)
    return np.maximum(iauc, 0.0)  # type: ignore[no-any-return]


def compute_peak_enhancement(
    signal_data: npt.NDArray[np.floating],
    baseline_frames: int = 5,
    max_cap: float = _PEAK_ENH_MAX_CAP,
) -> npt.NDArray[np.floating]:
    """Compute peak enhancement percentage from raw signal data.

    (S_max - S_baseline) / max(S_baseline, epsilon) × 100

    Must be computed on **raw MRI signal**, not concentration.
    Concentration has baseline ~0 by definition, causing division by
    near-zero and overflow.

    Args:
        signal_data: 4D raw signal data (x, y, z, time).
        baseline_frames: Number of pre-contrast baseline frames.
        max_cap: Maximum enhancement percentage (default 5000%).
            No tissue produces >5000% signal enhancement.

    Returns:
        3D peak enhancement percentage map.
    """
    baseline_mean = np.mean(signal_data[..., :baseline_frames], axis=-1)
    peak = np.max(signal_data, axis=-1)

    # Signal < 1.0 is dead/background — avoid division by near-zero
    baseline_safe = np.maximum(np.abs(baseline_mean), _PEAK_ENH_SIGNAL_FLOOR)
    enhancement = (peak - baseline_mean) / baseline_safe * 100.0

    enhancement = np.clip(enhancement, 0.0, max_cap)
    return np.asarray(enhancement, dtype=np.float64)  # type: ignore[return-value]


def compute_time_to_peak(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute time of maximum enhancement in seconds.

    Args:
        concentration: 4D concentration data (x, y, z, time) in mM.
        time_points: 1D time array in minutes.

    Returns:
        3D time-to-peak map in seconds.
    """
    peak_indices = np.argmax(concentration, axis=-1)
    ttp_minutes = time_points[peak_indices]
    return ttp_minutes * 60.0  # type: ignore[no-any-return]


def compute_signal_enhancement_ratio(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    bolus_arrival_frame: int = 0,
    early_offset: tuple[float, float] = (0.5, 2.5),
    late_fraction: float = 0.2,
) -> npt.NDArray[np.floating]:
    """Compute Signal Enhancement Ratio (SER) for washout characterization.

    SER = mean(C_early) / mean(C_late)

    High SER (>1) indicates washout; low SER (<1) indicates persistent enhancement.

    Time windows are relative to bolus arrival and adaptive to acquisition
    length, so the metric is valid for both short (<6 min) and long (>10 min)
    acquisitions.

    Args:
        concentration: 4D concentration data (x, y, z, time) in mM.
        time_points: 1D time array in minutes.
        bolus_arrival_frame: Frame index of bolus arrival.
        early_offset: (start, end) in minutes post-bolus for early phase.
        late_fraction: Fraction of post-bolus duration for late window
            (capped at 5 min for very long acquisitions).

    Returns:
        3D SER map. Returns ones where windows have insufficient frames.
    """
    ones = np.ones(concentration.shape[:3], dtype=np.float64)

    bolus_time = time_points[bolus_arrival_frame]

    # Early window: fixed offset from bolus arrival
    early_start = bolus_time + early_offset[0]
    early_end = bolus_time + early_offset[1]
    early_mask = (time_points >= early_start) & (time_points <= early_end)

    # Late window: last fraction of post-bolus acquisition, capped at 5 min
    post_bolus_duration = time_points[-1] - bolus_time
    late_duration = min(late_fraction * post_bolus_duration, 5.0)
    late_start = time_points[-1] - late_duration
    late_mask = (time_points >= late_start) & (time_points <= time_points[-1])

    if np.sum(early_mask) < 2 or np.sum(late_mask) < 2:
        return ones

    early_mean = np.mean(concentration[..., early_mask], axis=-1)
    late_mean = np.mean(concentration[..., late_mask], axis=-1)

    # Scale floor to data magnitude to avoid million-fold amplification
    early_median = float(np.nanmedian(early_mean))
    floor = max(0.05 * max(early_median, 0.0), 1e-6)
    late_safe = np.maximum(np.abs(late_mean), floor)
    ser = early_mean / late_safe

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

    Args:
        concentration: 4D concentration data (x, y, z, time) in mM.
        time_points: 1D time array in minutes.
        bolus_arrival_frame: Frame index of bolus arrival.
        volume_tr_s: Volume TR in seconds (for temporal resolution gating).
        savgol_window: Savitzky-Golay filter window length.
        savgol_polyorder: Savitzky-Golay polynomial order.

    Returns:
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

    Args:
        concentration: 4D concentration data (x, y, z, time) in mM.
        time_points: 1D time array in minutes.
        volume_tr_s: Volume TR in seconds. If None, estimated from time_points.
        post_peak_start_s: Start of washout window relative to TTP (seconds).
        post_peak_end_s: End of washout window relative to TTP (seconds).
        min_points: Minimum points required for regression.

    Returns:
        3D wash-out slope map (mM/min, signed: negative=washout).
        NaN where insufficient data points.
    """
    spatial_shape = concentration.shape[:3]
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


def compute_bolus_arrival_time(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    baseline_frames: int = 5,
    brain_mask: Optional[npt.NDArray[np.floating]] = None,
) -> npt.NDArray[np.floating]:
    """Compute per-voxel bolus arrival time map.

    Wraps ``detect_bolus_arrival_time`` from ``pmriprep.utils.dce_bat``,
    iterating over all brain voxels. Low-confidence or default-method
    results are set to NaN.

    Args:
        concentration: 4D concentration data (x, y, z, time) in mM.
        time_points: 1D time array in minutes.
        baseline_frames: Number of pre-contrast baseline frames.
        brain_mask: Optional 3D brain mask. When provided, voxels where
            mask <= 0 are skipped (remain NaN), avoiding ~50-70% of
            the per-voxel loop iterations.

    Returns:
        3D BAT map in seconds. NaN for low-confidence detections.
    """
    from pmriprep.utils.dce_bat import detect_bolus_arrival_time

    spatial_shape = concentration.shape[:3]
    bat_map = np.full(spatial_shape, np.nan, dtype=np.float64)

    if brain_mask is not None and brain_mask.shape != spatial_shape:
        raise ValueError(
            f"brain_mask shape {brain_mask.shape} does not match "
            f"concentration spatial shape {spatial_shape}"
        )

    flat_conc = concentration.reshape(-1, concentration.shape[-1])
    flat_bat = bat_map.ravel()
    flat_mask = brain_mask.ravel().astype(bool) if brain_mask is not None else None

    has_signal = np.max(np.abs(flat_conc), axis=1) >= 1e-9
    if flat_mask is not None:
        n_active = int(np.sum(has_signal & flat_mask))
        n_in_mask = int(np.sum(flat_mask))
        logger.info(
            "Computing BAT for %d active voxels (%d in mask, %d total)",
            n_active, n_in_mask, flat_conc.shape[0],
        )
    else:
        n_active = int(np.sum(has_signal))
        logger.info(
            "Computing BAT for %d active voxels (of %d total)",
            n_active, flat_conc.shape[0],
        )

    for i in range(flat_conc.shape[0]):
        if flat_mask is not None and not flat_mask[i]:
            continue
        curve = flat_conc[i]
        # Skip voxels with no enhancement
        if np.max(np.abs(curve)) < 1e-9:
            continue
        result = detect_bolus_arrival_time(curve, time_points, baseline_frames)
        if result.confidence != "low" and result.method != "default":
            flat_bat[i] = result.bat_seconds

    return bat_map


def compute_curve_type(
    ser: npt.NDArray[np.floating],
    persistent_threshold: float = 0.9,
    washout_threshold: float = 1.1,
) -> npt.NDArray[np.integer]:
    """Classify DCE enhancement curves based on SER values.

    Three-class classification following BI-RADS convention:
    - Type I (persistent): SER < persistent_threshold
    - Type II (plateau): persistent_threshold <= SER <= washout_threshold
    - Type III (washout): SER > washout_threshold

    Args:
        ser: 3D Signal Enhancement Ratio map.
        persistent_threshold: SER below this is persistent (default 0.9).
        washout_threshold: SER above this is washout (default 1.1).

    Returns:
        3D int8 map: 0=unclassified (NaN SER), 1=persistent,
        2=plateau, 3=washout.
    """
    curve_type = np.zeros(ser.shape, dtype=np.int8)
    valid = np.isfinite(ser)
    curve_type[valid & (ser < persistent_threshold)] = 1
    curve_type[valid & (ser >= persistent_threshold) & (ser <= washout_threshold)] = 2
    curve_type[valid & (ser > washout_threshold)] = 3
    return curve_type
