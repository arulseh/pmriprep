"""Finite-window delay handling for parametric DSC convolution."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def _apply_post_hoc_delay_shift(
    conv: npt.NDArray[np.floating],
    delay: npt.NDArray[np.floating],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Linearly shift sampled rows, zero-filling outside their supplied support.

    Parameters
    ----------
    conv : ndarray
        Sampled curves, shape (N, T).
    delay : ndarray
        Per-row delay in seconds, shape (N,).
    dt : float
        Frame period in seconds.

    Returns
    -------
    ndarray
        Shifted rows with the same shape as the input.
    """
    n_voxels, n_times = conv.shape
    delay_frac = delay / dt
    n_shift = np.floor(delay_frac).astype(int)
    f_shift = delay_frac - n_shift
    src = np.arange(n_times)[None, :] - n_shift[:, None]
    row_idx = np.arange(n_voxels)[:, None]
    valid_lo = (src >= 0) & (src < n_times)
    shifted = np.where(
        valid_lo,
        (1.0 - f_shift[:, None]) * conv[row_idx, np.clip(src, 0, n_times - 1)],
        0.0,
    )
    src_hi = src - 1
    valid_hi = valid_lo & (src_hi >= 0) & (src_hi < n_times)
    shifted += np.where(
        valid_hi,
        f_shift[:, None] * conv[row_idx, np.clip(src_hi, 0, n_times - 1)],
        0.0,
    )
    return shifted


def _volterra_tail_matrix(
    aif: npt.NDArray[np.floating], dt: float, extra: int
) -> npt.NDArray[np.float64]:
    """Build only the additional Volterra rows under zero future AIF input.

    Parameters
    ----------
    aif : ndarray
        Observed arterial input, shape (T,).
    dt : float
        Frame period in seconds.
    extra : int
        Number of response samples required after the observed window.

    Returns
    -------
    ndarray
        Operator rows with shape (extra, T + extra).
    """
    n_times = len(aif)
    rows = np.arange(n_times, n_times + extra)[:, None]
    lag = rows - np.arange(n_times + extra)[None, :]
    # Pad both sides for the centered stencil; the AIF is zero after its end.
    padded = np.pad(np.asarray(aif, dtype=np.float64), (1, extra + 1))
    index = np.clip(lag + 1, 1, len(padded) - 2)
    matrix: npt.NDArray[np.float64] = (
        padded[index - 1] + 4.0 * padded[index] + padded[index + 1]
    ) / 6.0
    matrix[lag < 0] = 0.0
    matrix[lag == 0] = (2.0 * aif[0] + aif[1]) / 6.0
    matrix[:, 0] = (2.0 * padded[rows[:, 0] + 1] + padded[rows[:, 0]]) / 6.0
    return matrix * dt


def apply_convolution_delay(
    conv: npt.NDArray[np.floating],
    residue: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    delay: npt.NDArray[np.floating],
    residue_tail: Callable[
        [npt.NDArray[np.intp], npt.NDArray[np.float64]], npt.NDArray[np.floating]
    ],
    residue_scale: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Apply delay after completing the causal response needed for advances.

    Negative delays require response samples after the acquisition window.
    These retain contributions from the observed AIF, assuming zero additional
    input after its end. Only additional convolution rows are evaluated;
    the cached native convolution and residue definition are preserved.

    Parameters
    ----------
    conv : ndarray
        Native convolved curves, shape (N, T).
    residue : ndarray
        Native residue samples, shape (N, T).
    aif : ndarray
        Observed arterial input, shape (T,).
    time_points : ndarray
        Uniform acquisition times in seconds, shape (T,).
    delay : ndarray
        Per-voxel delay in seconds, shape (N,).
    residue_tail : callable
        Evaluate residue at additional times for the supplied voxel indices.
        Must return shape (number of indices, number of additional times).
    residue_scale : ndarray or None, optional
        Per-voxel amplitude when residue samples are normalized (CPI).

    Returns
    -------
    ndarray
        Delayed tissue curves on the original acquisition grid, shape (N, T).
    """
    dt = float(time_points[1] - time_points[0])
    result = _apply_post_hoc_delay_shift(conv, delay, dt)
    negative = np.flatnonzero(delay < 0)
    if negative.size == 0:
        return result
    extra = int(np.ceil(-float(np.min(delay[negative])) / dt))
    additional_times = time_points[-1] + dt * np.arange(1, extra + 1, dtype=np.float64)
    extended_residue = np.concatenate(
        [residue[negative], residue_tail(negative, additional_times)], axis=1
    )
    tail = extended_residue @ _volterra_tail_matrix(aif, dt, extra).T
    if residue_scale is not None:
        tail *= residue_scale[negative, None]
    extended_conv = np.concatenate([conv[negative], tail], axis=1)
    result[negative] = _apply_post_hoc_delay_shift(
        extended_conv, delay[negative], dt
    )[:, :len(time_points)]
    return result
