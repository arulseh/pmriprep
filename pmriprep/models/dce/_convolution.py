"""Exact analytic convolution of a piecewise-linear AIF with an exponential.

Implements the exact-convolution (``expconv``) used by reference DCE software:
the integral ``I(t) = integral_0^t Cp(tau) exp(-rate (t-tau)) dtau`` for a
piecewise-linear Cp, evaluated exactly on the sample grid. Being exact, it
introduces no quadrature bias for fast-decaying (high-rate) kernels and so does
not under-read the concentration of high-kep tissue.

The recursion increment over each segment [t_i, t_{i+1}] is
``J_i = Cp_i * dt * g0(x) + (Cp_{i+1}-Cp_i) * dt * g1(x)`` with ``x = rate*dt``,
``g0(x) = (1 - e^-x)/x`` and ``g1(x) = (x - (1 - e^-x))/x^2``. Both g0, g1 are
bounded, dimensionless functions of x with clean small-x series (g0 -> 1,
g1 -> 1/2), so the scheme needs no division by ``rate`` and degenerates to the
trapezoidal integral at rate -> 0.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

#: Below this x = rate*dt the ``x - (1 - e^-x)`` subtraction in g1 loses
#: precision; use the series expansion instead.
_SMALL_X_THRESHOLD = 1e-4


def analytic_exp_convolution(
    aif: npt.NDArray[np.floating],
    rate: Union[float, npt.NDArray[np.floating]],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Exact convolution ``I = Cp (*) exp(-rate*t)`` for piecewise-linear Cp.

    Parameters
    ----------
    aif : np.ndarray
        AIF plasma concentration Cp on a uniform grid of spacing ``dt``, shape
        ``(T,)``. Assumed causal (zero pre-arrival), so ``I(0) = 0``.
    rate : float or np.ndarray
        Exponential rate(s) in min^-1. A scalar for a single curve, or shape
        ``(V,)`` for V voxels convolved against the shared AIF.
    dt : float
        Sample spacing in minutes.

    Returns
    -------
    np.ndarray
        The convolution integral ``I``. Shape ``(T,)`` for scalar ``rate``,
        otherwise ``(V, T)``.
    """
    aif = np.asarray(aif, dtype=np.float64)
    is_scalar = np.ndim(rate) == 0
    r = np.atleast_1d(np.asarray(rate, dtype=np.float64))  # (V,)
    n = aif.shape[0]

    x = r * dt  # (V,)
    e = np.exp(-x)
    small = x < _SMALL_X_THRESHOLD
    x_safe = np.where(small, 1.0, x)
    e0 = -np.expm1(-x)  # 1 - e^-x, stable

    # g0 = (1 - e^-x)/x -> 1 ; g1 = (x - (1 - e^-x))/x^2 -> 1/2  as x -> 0.
    g0 = np.where(small, 1.0 - x / 2.0 + x * x / 6.0, e0 / x_safe)
    g1 = np.where(
        small,
        0.5 - x / 6.0 + x * x / 24.0,
        (x - e0) / (x_safe * x_safe),
    )
    c0 = dt * g0  # (V,) multiplies Cp_i
    c1 = dt * g1  # (V,) multiplies (Cp_{i+1} - Cp_i)

    dcp = np.diff(aif)  # (T-1,) shared across voxels
    if r.shape[0] == 1:
        # Single rate: the recursion conv[i+1] = e*conv[i] + add[i] is a
        # first-order IIR filter; run it in C via lfilter (matches the loop
        # exactly and avoids the per-step Python overhead of a scalar fit).
        add = aif[:-1] * c0[0] + dcp * c1[0]  # (T-1,)
        driver = np.concatenate(([0.0], add))  # (T,); conv[0] = 0
        conv1d: npt.NDArray[np.float64] = np.asarray(
            lfilter([1.0], [1.0, -float(e[0])], driver), dtype=np.float64
        )
        return conv1d if is_scalar else conv1d[np.newaxis, :]

    # Multiple per-voxel rates: sequential over time, vectorized over voxels.
    conv = np.zeros((r.shape[0], n), dtype=np.float64)
    for i in range(n - 1):
        conv[:, i + 1] = e * conv[:, i] + (aif[i] * c0 + dcp[i] * c1)

    return conv
