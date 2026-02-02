"""Model-free DCE maps (QIBA-recommended).

Computes model-assumption-free quantitative maps from DCE concentration data:
IAUC, peak enhancement, time-to-peak, and signal enhancement ratio.
"""

import numpy as np
import numpy.typing as npt


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
    return np.maximum(iauc, 0.0)


def compute_peak_enhancement(
    concentration: npt.NDArray[np.floating],
    baseline_frames: int = 5,
) -> npt.NDArray[np.floating]:
    """Compute peak enhancement percentage.

    (max_concentration - baseline_mean) / max(baseline_mean, epsilon) × 100

    Args:
        concentration: 4D concentration data (x, y, z, time) in mM.
        baseline_frames: Number of pre-contrast baseline frames.

    Returns:
        3D peak enhancement percentage map.
    """
    baseline_mean = np.mean(concentration[..., :baseline_frames], axis=-1)
    peak = np.max(concentration, axis=-1)

    baseline_safe = np.maximum(np.abs(baseline_mean), 1e-9)
    enhancement = (peak - baseline_mean) / baseline_safe * 100.0

    return np.maximum(enhancement, 0.0)


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
    return ttp_minutes * 60.0


def compute_signal_enhancement_ratio(
    concentration: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    early_window: tuple[float, float] = (0.5, 2.0),
    late_window: tuple[float, float] = (5.0, 10.0),
) -> npt.NDArray[np.floating]:
    """Compute Signal Enhancement Ratio (SER) for washout characterization.

    SER = mean(C_early) / mean(C_late)

    High SER (>1) indicates washout; low SER (<1) indicates persistent enhancement.

    Args:
        concentration: 4D concentration data (x, y, z, time) in mM.
        time_points: 1D time array in minutes.
        early_window: (start, end) in minutes for early phase.
        late_window: (start, end) in minutes for late phase.

    Returns:
        3D SER map.
    """
    early_mask = (time_points >= early_window[0]) & (time_points <= early_window[1])
    late_mask = (time_points >= late_window[0]) & (time_points <= late_window[1])

    if np.sum(early_mask) < 1 or np.sum(late_mask) < 1:
        return np.ones(concentration.shape[:3], dtype=np.float64)

    early_mean = np.mean(concentration[..., early_mask], axis=-1)
    late_mean = np.mean(concentration[..., late_mask], axis=-1)

    late_safe = np.maximum(np.abs(late_mean), 1e-9)
    ser = early_mean / late_safe

    return ser
