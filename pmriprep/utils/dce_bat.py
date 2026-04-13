"""Bolus Arrival Time (BAT) detection for DCE-MRI.

Detects the time of contrast agent arrival using threshold and gradient
methods, with a default fallback.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BATResult:
    """Result of bolus arrival time detection.

    Attributes:
        bat_seconds: Bolus arrival time in seconds.
        peak_seconds: Time of peak enhancement in seconds.
        confidence: Detection confidence ("high" or "low").
        method: Method used ("threshold", "gradient", or "default").
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

    Args:
        curve: 1D concentration time course (mM).
        time_points: 1D time array in minutes.
        baseline_frames: Number of pre-contrast baseline frames.

    Returns:
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
