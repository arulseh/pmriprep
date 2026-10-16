"""Model-free DCE maps.

Computes maps directly from DCE signal and concentration time courses:
IAUC, peak enhancement, time-to-peak, signal enhancement ratio, and bolus
arrival time.
"""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

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

    Raises
    ------
    ValueError
        If ``cutoff_s <= 0``, the time axis is not a finite, strictly increasing
        1D vector matching the concentration length, or the reference frame is
        out of range.
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
        Number of precontrast baseline frames, at least 1 and less than the
        number of timepoints. Default is 5.

    Returns
    -------
    numpy.ndarray
        3D peak enhancement percentage map. Voxels with a non-positive
        baseline or any non-finite signal samples are NaN.

    Raises
    ------
    ValueError
        If the baseline count is below 1 or leaves no post-baseline frame.
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
        Must be at least 1 and less than the number of timepoints; a count
        of 1 produces an all-NaN map. Default is 5.
    brain_mask : numpy.ndarray, optional
        Brain mask used to pool a robust spatial noise floor. When omitted,
        all spatial voxels contribute.

    Returns
    -------
    numpy.ndarray
        3D reference-relative time-to-peak map in seconds. Voxels with
        non-finite samples or insufficient peak enhancement are NaN. The
        threshold is the larger of five effective baseline noise standard
        deviations and a floating-point tolerance. Effective noise is the
        larger of voxel noise and the median noise within ``brain_mask``.

    Raises
    ------
    ValueError
        If the time axis is not a finite, strictly increasing 1D vector
        matching the concentration length, the reference frame is out of range,
        the baseline count is below 1 or leaves no post-baseline frame, or the
        brain mask has a different spatial shape.
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
    early_time_s: float = 60.0,
    late_time_s: float = 300.0,
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
        Number of precontrast frames used to estimate S0 and its sample noise.
        Must be at least 1 and less than the number of timepoints; a count
        of 1 produces an all-NaN map before checking timing values. Default 5.
    early_time_s : float, optional
        Early sample time in seconds after the selected global reference.
    late_time_s : float, optional
        Late sample time in seconds after the selected global reference.

    Returns
    -------
    numpy.ndarray
        3D dimensionless SER map. Undefined voxels are NaN; inadequate
        acquisition coverage or a one-frame baseline gives an all-NaN map.

    Raises
    ------
    ValueError
        If time and signal lengths differ, the time axis is not 1D, or the
        baseline count is below 1 or leaves no post-baseline frame. With at
        least two baseline frames, also rejects an invalid reference frame,
        nonfinite or nonincreasing times, and early/late offsets that are not
        finite, positive and strictly ordered.
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
