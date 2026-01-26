"""DSC signal processing utilities.

Functions for converting DSC-MRI signal to concentration (ΔR2*)
and computing derived metrics like PSR and peak height.
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dsc_constants import (
    EPSILON_SIGNAL,
    R2_STAR_BY_FIELD,
    get_r2_star,
)


def signal_to_concentration(
    signal: npt.NDArray[np.floating],
    te: float,
    baseline_frames: int = 10,
    r2_star: Optional[float] = None,
    field_strength: Optional[float] = None,
) -> npt.NDArray[np.floating]:
    """Convert DSC signal to concentration via ΔR2*.

    Computes concentration from the signal decay using the relationship:
        ΔR2* = -ln(S(t)/S0) / TE
        C(t) = ΔR2* / r2*  (if absolute quantification)

    In relative mode (no r2* provided), returns ΔR2* directly.
    In absolute mode (r2* provided), returns concentration in mM.

    Args:
        signal: 4D array of DSC signal (x, y, z, time) or 1D time series.
            Signal values should be positive.
        te: Echo time in seconds. Typical values: 0.025-0.040s.
        baseline_frames: Number of pre-bolus frames for S0 estimation.
            Default is 10 frames.
        r2_star: Transverse relaxivity in s^-1 per mM for absolute
            quantification. If None, returns relative ΔR2*.
        field_strength: Magnetic field strength in Tesla. Used to look up
            r2* if r2_star is not provided but absolute quantification
            is desired. Common values: 1.5, 3.0, 7.0.

    Returns:
        Concentration array with same shape as input signal.
        Units are s^-1 (ΔR2*) for relative mode or mM for absolute mode.

    Raises:
        ValueError: If signal array is empty or has invalid dimensions.
        ValueError: If te is not positive.
        ValueError: If baseline_frames exceeds signal length.

    Notes:
        - Signal ratio is clipped to (EPSILON_SIGNAL, 1.0) before log
          to prevent numerical issues with zero or negative values.
        - Baseline S0 is computed as mean of first baseline_frames.
        - For 4D input, baseline is computed per-voxel.
    """
    # Validate inputs
    if signal.size == 0:
        raise ValueError("Signal array is empty")

    if te <= 0:
        raise ValueError(f"TE must be positive, got {te}")

    # Handle 1D input
    if signal.ndim == 1:
        n_timepoints = signal.shape[0]
    elif signal.ndim == 4:
        n_timepoints = signal.shape[3]
    else:
        raise ValueError(
            f"Signal must be 1D (time) or 4D (x,y,z,time), got {signal.ndim}D"
        )

    if baseline_frames >= n_timepoints:
        raise ValueError(
            f"baseline_frames ({baseline_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    if baseline_frames < 1:
        raise ValueError(f"baseline_frames must be at least 1, got {baseline_frames}")

    # Compute baseline S0
    if signal.ndim == 1:
        s0 = np.mean(signal[:baseline_frames])
    else:
        # 4D: mean over first baseline_frames along time axis
        s0 = np.mean(signal[..., :baseline_frames], axis=-1, keepdims=True)

    # Compute signal ratio S(t)/S0, clip to valid range
    # Clip s0 to prevent division by zero
    s0_safe = np.maximum(s0, EPSILON_SIGNAL)
    signal_ratio = signal / s0_safe

    # Clip signal ratio to (EPSILON_SIGNAL, 1.0) before log
    # Upper bound of 1.0 prevents log of values > 1 (which would give negative ΔR2*)
    # Lower bound prevents log(0) or log(negative)
    signal_ratio_clipped = np.clip(signal_ratio, EPSILON_SIGNAL, 1.0)

    # Compute ΔR2* = -ln(S/S0) / TE
    # Note: negative of log because signal decreases with contrast
    delta_r2_star = -np.log(signal_ratio_clipped) / te

    # Determine r2* for conversion
    effective_r2_star: Optional[float] = None

    if r2_star is not None:
        effective_r2_star = r2_star
    elif field_strength is not None:
        effective_r2_star = get_r2_star(field_strength)

    # Return relative ΔR2* or absolute concentration
    if effective_r2_star is not None:
        # Absolute quantification: C = ΔR2* / r2*
        concentration = delta_r2_star / effective_r2_star
    else:
        # Relative mode: return ΔR2* directly (units: s^-1)
        concentration = delta_r2_star

    return concentration


def signal_to_delta_r2_star(
    signal: npt.NDArray[np.floating],
    te: float,
    baseline_frames: int = 10,
) -> npt.NDArray[np.floating]:
    """Convert DSC signal to ΔR2* (relative concentration).

    Convenience wrapper for signal_to_concentration with relative mode.

    Args:
        signal: 4D array of DSC signal (x, y, z, time) or 1D time series.
        te: Echo time in seconds.
        baseline_frames: Number of pre-bolus frames for S0 estimation.

    Returns:
        ΔR2* array with same shape as input signal, units s^-1.
    """
    return signal_to_concentration(
        signal=signal,
        te=te,
        baseline_frames=baseline_frames,
        r2_star=None,
        field_strength=None,
    )


def get_baseline_signal(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 10,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute baseline signal S0 from pre-bolus frames.

    Args:
        signal: DSC signal array (1D, 3D, or 4D).
        baseline_frames: Number of pre-bolus frames to average.

    Returns:
        Baseline signal value (scalar for 1D) or array (for 3D/4D).
    """
    if signal.ndim == 1:
        return float(np.mean(signal[:baseline_frames]))
    elif signal.ndim == 4:
        return np.mean(signal[..., :baseline_frames], axis=-1)
    else:
        raise ValueError(f"Signal must be 1D or 4D, got {signal.ndim}D")


def find_bolus_arrival(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 10,
    threshold_factor: float = 3.0,
) -> int:
    """Find the bolus arrival time point.

    Detects when signal drops significantly below baseline,
    indicating contrast agent arrival.

    Args:
        signal: 1D DSC signal time course.
        baseline_frames: Number of pre-bolus frames.
        threshold_factor: Number of baseline standard deviations
            for threshold (default 3.0).

    Returns:
        Index of bolus arrival frame.

    Raises:
        ValueError: If signal is not 1D.
        ValueError: If no bolus arrival is detected.
    """
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got {signal.ndim}D")

    baseline = signal[:baseline_frames]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)

    # Threshold: signal drops below baseline - threshold_factor * std
    threshold = baseline_mean - threshold_factor * baseline_std

    # Find first point below threshold after baseline
    for i in range(baseline_frames, len(signal)):
        if signal[i] < threshold:
            return i

    raise ValueError("No bolus arrival detected in signal")


def compute_psr(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 10,
    tail_frames: int = 10,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute percent signal recovery (PSR).

    PSR quantifies how much signal recovers after the bolus passage,
    indicating blood-brain barrier integrity.

    Formula:
        PSR = 100 × (S_tail - S_min) / (S_baseline - S_min)

    Clinical interpretation:
        - PSR < 70%: Blood-brain barrier disruption (leakage)
        - PSR 70-90%: Partial BBB disruption
        - PSR > 90%: Intact blood-brain barrier

    Args:
        signal: DSC signal array. Can be:
            - 1D (time,): Single voxel time course
            - 4D (x, y, z, time): Full 4D volume
        baseline_frames: Number of pre-bolus frames for S_baseline.
            Default is 10.
        tail_frames: Number of post-bolus frames for S_tail.
            Default is 10 (last 10 frames).

    Returns:
        PSR value(s) as percentage (0-100+ range).
        - Scalar float for 1D input
        - 3D array (x, y, z) for 4D input

    Raises:
        ValueError: If signal has invalid dimensions.
        ValueError: If baseline_frames or tail_frames exceeds signal length.

    Notes:
        - PSR can exceed 100% in cases where the signal overshoots baseline
          after bolus passage (rare, usually indicates motion artifact).
        - PSR can be negative if signal never recovers (very high leakage
          or signal artifact).
        - Epsilon is added to denominator to prevent division by zero.
    """
    # Validate input dimensions
    if signal.ndim not in (1, 4):
        raise ValueError(f"Signal must be 1D or 4D, got {signal.ndim}D")

    # Get number of timepoints
    n_timepoints = signal.shape[-1] if signal.ndim == 4 else signal.shape[0]

    if baseline_frames >= n_timepoints:
        raise ValueError(
            f"baseline_frames ({baseline_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    if tail_frames >= n_timepoints:
        raise ValueError(
            f"tail_frames ({tail_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    if baseline_frames < 1 or tail_frames < 1:
        raise ValueError("baseline_frames and tail_frames must be at least 1")

    # Compute signal components
    if signal.ndim == 1:
        # 1D time series
        s_baseline = float(np.mean(signal[:baseline_frames]))
        s_min = float(np.min(signal))
        s_tail = float(np.mean(signal[-tail_frames:]))
    else:
        # 4D volume (x, y, z, time)
        s_baseline = np.mean(signal[..., :baseline_frames], axis=-1)
        s_min = np.min(signal, axis=-1)
        s_tail = np.mean(signal[..., -tail_frames:], axis=-1)

    # Compute PSR with epsilon guard in denominator
    denominator = s_baseline - s_min
    # Add small epsilon to prevent division by zero
    denominator_safe = np.maximum(np.abs(denominator), EPSILON_SIGNAL)
    # Preserve sign of denominator
    denominator_safe = np.where(
        denominator >= 0, denominator_safe, -denominator_safe
    )

    psr = 100.0 * (s_tail - s_min) / denominator_safe

    # Return appropriate type
    if signal.ndim == 1:
        return float(psr)
    else:
        return psr


def compute_peak_height(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 10,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute peak height (PH) of DSC signal.

    PH quantifies the maximum signal drop during bolus passage,
    reflecting contrast agent concentration and vascular density.

    Formula:
        PH = 100 × (S_baseline - S_min) / S_baseline

    Clinical interpretation:
        - Higher PH indicates greater signal drop (more contrast effect)
        - PH correlates with cerebral blood volume (CBV)
        - Typical values: 20-60% in normal tissue, may be higher in tumors

    Args:
        signal: DSC signal array. Can be:
            - 1D (time,): Single voxel time course
            - 4D (x, y, z, time): Full 4D volume
        baseline_frames: Number of pre-bolus frames for S_baseline.
            Default is 10.

    Returns:
        Peak height value(s) as percentage (0-100 range typically).
        - Scalar float for 1D input
        - 3D array (x, y, z) for 4D input

    Raises:
        ValueError: If signal has invalid dimensions.
        ValueError: If baseline_frames exceeds signal length.

    Notes:
        - PH is always positive when computed from valid DSC signal
          (signal drops during bolus passage).
        - PH can exceed 100% if signal drops below zero (artifact).
        - Epsilon is added to baseline to prevent division by zero.
    """
    # Validate input dimensions
    if signal.ndim not in (1, 4):
        raise ValueError(f"Signal must be 1D or 4D, got {signal.ndim}D")

    # Get number of timepoints
    n_timepoints = signal.shape[-1] if signal.ndim == 4 else signal.shape[0]

    if baseline_frames >= n_timepoints:
        raise ValueError(
            f"baseline_frames ({baseline_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    if baseline_frames < 1:
        raise ValueError("baseline_frames must be at least 1")

    # Compute signal components
    if signal.ndim == 1:
        # 1D time series
        s_baseline = float(np.mean(signal[:baseline_frames]))
        s_min = float(np.min(signal))
    else:
        # 4D volume (x, y, z, time)
        s_baseline = np.mean(signal[..., :baseline_frames], axis=-1)
        s_min = np.min(signal, axis=-1)

    # Compute PH with epsilon guard in denominator
    s_baseline_safe = np.maximum(np.abs(s_baseline), EPSILON_SIGNAL)
    # Preserve sign of baseline
    s_baseline_safe = np.where(
        s_baseline >= 0, s_baseline_safe, -s_baseline_safe
    )

    ph = 100.0 * (s_baseline - s_min) / s_baseline_safe

    # Return appropriate type
    if signal.ndim == 1:
        return float(ph)
    else:
        return ph
