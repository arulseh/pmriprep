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
