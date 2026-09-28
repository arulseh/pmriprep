"""Automatic CPI count from the mean tissue tail, before geometry limits."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION


def select_n_control_points(
    mean_tissue: npt.NDArray[np.floating],
    dt: float,
    min_n: int = 4,
    max_n: int = 24,
) -> int:
    """Recommend a count from the tissue tail before acquisition support checks.

    Parameters
    ----------
    mean_tissue : ndarray
        Mean tissue concentration across in-mask voxels, shape (T,).
    dt : float
        Frame period in seconds.
    min_n, max_n : int, optional
        Limits on the heuristic recommendation. The fitting policy separately
        restricts the result to supported counts and feasible knot geometry.

    Returns
    -------
    int
        Ceiling of tail-decay seconds divided by three, clipped to the limits.
    """
    decay_sec = estimate_tail_decay_time_seconds(mean_tissue, float(dt))
    return max(min_n, min(max_n, int(np.ceil(decay_sec / 3.0))))


def estimate_tail_decay_time_seconds(
    mean_tissue: npt.NDArray[np.floating],
    dt: float,
    threshold_fraction: float = 0.1,
    fallback_sec: float = 30.0,
    max_decay_fraction: float = 0.5,
) -> float:
    """Time from post-peak frame to first frame at <``threshold_fraction`` of peak.

    Returns a property of the mean tissue concentration curve. For
    exponential residues this scales roughly with mean MTT, but it
    conflates AIF tail and tissue MTT, so callers treat it as an opaque
    proxy when selecting the control-point count.

    Parameters
    ----------
    mean_tissue : np.ndarray
        Mean tissue concentration across in-mask voxels,
        shape (T,). Pre-computed by the caller.
    dt : float
        Frame period in seconds.
    threshold_fraction : float, optional
        Fraction of peak defining the tail end.
        Defaults to 0.1.
    fallback_sec : float, optional
        Returned when peak detection fails (zero curve,
        peak at last frame, etc.).
    max_decay_fraction : float, optional
        Cap returned decay time at
        ``max_decay_fraction × T·dt``. Prevents runaway N on
        preloaded studies where the tail never drops below threshold.

    Returns
    -------
    float
        Capped decay time in seconds, or ``fallback_sec`` if unavailable.
    """
    n_frames = len(mean_tissue)
    if n_frames < 2:
        return fallback_sec
    if not np.all(np.isfinite(mean_tissue)):
        return fallback_sec
    peak_frame = int(np.argmax(mean_tissue))
    peak_val = float(mean_tissue[peak_frame])
    if peak_val < EPSILON_CONCENTRATION:
        return fallback_sec
    if peak_frame >= n_frames - 1:
        return fallback_sec
    threshold = peak_val * threshold_fraction
    post_peak = mean_tissue[peak_frame:]
    below = np.where(post_peak < threshold)[0]
    if len(below) == 0:
        decay_frames = len(post_peak)
    else:
        decay_frames = int(below[0])
    decay_sec = float(decay_frames * dt)
    cap_sec = float(max_decay_fraction * n_frames * dt)
    return min(decay_sec, cap_sec)
