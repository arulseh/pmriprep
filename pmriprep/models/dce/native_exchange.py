"""Native bounded 2CXM fitting with owned held-endpoint PCHIP contexts."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from scipy.special import hyp1f1

from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce._native_backend import D, RECURRENCE, Evaluator, solve
from pmriprep.utils.dce_constants import (
    EPSILON_EIGENVALUE,
    EPSILON_FLOW,
    EPSILON_VE_FORWARD,
    EPSILON_VP,
)

# Generic physical starts in Fp, PS, ve, vp.
STARTS = np.array(
    [[20.0, 1.0, 0.1, 0.01], [80.0, 20.0, 0.3, 0.03], [200.0, 100.0, 0.6, 0.08]]
)


class NativeInput:
    """Prepare held-endpoint PCHIP input for nonnegative-offset native fits.

    Parameters
    ----------
    aif, times : ndarray
        Finite plasma AIF in mM and uniform acquisition times in minutes.
    delay_seconds : float
        Nonnegative effective AIF offset in seconds.
    observed : ndarray, optional
        Original retained frame indices. Full input history is preserved.

    Notes
    -----
    Convolution returns independent arrays; cached workspace access is locked.
    Signed diagnostic offsets require a separately validated fitting path.
    """

    def __init__(
        self,
        aif: np.ndarray,
        times: np.ndarray,
        delay_seconds: float = 0.0,
        observed: np.ndarray | None = None,
    ) -> None:
        if not np.isfinite(delay_seconds) or delay_seconds < 0:
            raise ValueError("Native input requires a finite nonnegative delay.")
        prepared = DelayedPchip(aif, times, delay_seconds, observed)
        self.aif, self.t, self.h = prepared.aif, prepared.t, prepared.h
        self.delay, self.observed = prepared.delay, prepared.observed
        self.output_t, self.q, self.cp = prepared.output_t, prepared.q, prepared.cp
        self.poly = prepared.poly
        self.coefficients = self.poly.c[::-1]
        clipped = np.clip(self.q, 0.0, self.t[-1])
        self.k = np.clip(
            np.searchsorted(self.t, clipped, side="right") - 1, 0, len(aif) - 2
        )
        self.s = clipped - self.t[self.k]
        self.unique_s, self.inverse_s = np.unique(self.s, return_inverse=True)
        self.full_coeff = self.coefficients * self.h ** np.arange(1, 5)[:, None]
        self.part_coeff = (
            self.coefficients[:, self.k] * self.s ** np.arange(1, 5)[:, None]
        )
        self.held = np.minimum(self.output_t, self.delay)
        self.tail = np.maximum(self.output_t - self.delay, 0.0)
        self.widths = np.r_[self.h, self.unique_s]
        self.before = self.q < 0
        self.buffers: dict[int, tuple[Any, ...]] = {}
        self._lock = Lock()

    def convolution(self, rate: float | np.ndarray) -> np.ndarray:
        """Evaluate the same PCHIP exponential integrals, returning fresh storage.

        Parameters
        ----------
        rate : float or ndarray
            Nonnegative scalar or one-dimensional rates.

        Returns
        -------
        ndarray
            Independent output, unaffected by later calls on this context.
        """
        with self._lock:
            return self._convolution(rate)

    def _convolution(self, rate: float | np.ndarray) -> np.ndarray:
        """Evaluate with exclusive access to the context buffers."""
        scalar = np.ndim(rate) == 0
        rates = np.atleast_1d(np.asarray(rate, float))
        if rates.ndim != 1 or not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("Finite nonnegative rates required.")
        count = len(rates)
        if count not in self.buffers:
            increments = np.empty((count, self.full_coeff.shape[1]), dtype=np.float64)
            decay = np.empty(count, dtype=np.float64)
            states = np.empty((count, len(self.t)), dtype=np.float64)
            ptrs = tuple(a.ctypes.data_as(D) for a in [increments, decay, states])
            self.buffers[count] = (increments, decay, states, *ptrs)
        increments, decay, states, ip, dp, sp = self.buffers[count]
        x = rates[:, None] * self.widths
        negative_x = -x
        moment = np.stack(
            [hyp1f1(1.0, m + 2.0, negative_x) / (m + 1.0) for m in range(4)]
        )
        np.matmul(moment[:, :, 0].T, self.full_coeff, out=increments)
        np.exp(-rates * self.h, out=decay)
        RECURRENCE(ip, dp, sp, ct.c_size_t(count), ct.c_size_t(increments.shape[1]))
        out = np.exp(-rates[:, None] * self.s) * states[:, self.k]
        out += np.sum(
            moment[:, :, 1:][:, :, self.inverse_s] * self.part_coeff[:, None, :], axis=0
        )
        out[:, self.before] = 0.0
        if self.aif[0] != 0:
            held_moment = hyp1f1(1.0, 2.0, -rates[:, None] * self.held)
            out += (
                self.aif[0]
                * np.exp(-rates[:, None] * self.tail)
                * self.held
                * held_moment
            )
        return np.asarray(out[0] if scalar else out)


def predict(context: NativeInput, params: np.ndarray) -> np.ndarray:
    """Evaluate 2CXM at fixed delay; params are Fp, PS, ve and vp.

    Parameters
    ----------
    context : NativeInput
        Exact integration context, including the fixed delay.
    params : ndarray
        One parameter vector or a batch. Fp/PS are mL/min/100 mL.

    Returns
    -------
    ndarray
        Predicted tissue concentrations, with the input batch convention.
    """
    scalar = np.ndim(params) == 1
    p = np.atleast_2d(params)
    flow = np.maximum(p[:, 0], EPSILON_FLOW) / 100.0
    ps = np.maximum(p[:, 1], 0.0) / 100.0
    ve, vp = np.maximum(p[:, 2], EPSILON_VE_FORWARD), np.maximum(p[:, 3], EPSILON_VP)
    pe, ep, out = ps / vp, ps / ve, flow / vp
    trace = out + pe + ep
    high = (trace + np.sqrt(np.maximum(trace**2 - 4 * out * ep, 0.0))) / 2
    low = np.divide(out * ep, high, out=np.zeros_like(high), where=high > 0)
    high, low = (
        np.maximum(high, EPSILON_EIGENVALUE),
        np.maximum(low, EPSILON_EIGENVALUE),
    )
    degenerate = high - low < EPSILON_EIGENVALUE
    low = np.where(degenerate, high, low)
    difference = np.maximum(high - low, EPSILON_EIGENVALUE)
    a1 = np.where(degenerate, flow, flow * (high - ep - pe) / difference)
    a2 = np.where(degenerate, 0.0, flow * (ep + pe - low) / difference)
    convolution = context.convolution(np.r_[high, low])
    curves = a1[:, None] * convolution[: len(p)] + a2[:, None] * convolution[len(p) :]
    zero_ve = p[:, 2] <= 0
    if zero_ve.any():
        curves[zero_ve] = flow[zero_ve, None] * context.convolution(out[zero_ve])
    return np.asarray(curves[0] if scalar else curves)


def encode(params: np.ndarray) -> np.ndarray:
    """Map Fp/PS/ve/vp into scaled flow and joint-volume box coordinates.

    Parameters
    ----------
    params : ndarray
        Physical Fp/PS in mL/min/100 mL and dimensionless ve/vp.

    Returns
    -------
    ndarray
        Four optimizer coordinates with flows scaled by 100.
    """
    p = np.asarray(params)
    return np.array([p[0] / 100.0, p[1] / 100.0, p[2], p[3] / max(1 - p[2], 1e-12)])


def decode(x: np.ndarray) -> np.ndarray:
    """Map box coordinates back to physical Fp/PS/ve/vp.

    Parameters
    ----------
    x : ndarray
        Single or batched four-coordinate optimizer vectors.

    Returns
    -------
    ndarray
        Physical parameters with ve + vp <= 1 for bounded coordinates.
    """
    x = np.asarray(x)
    return np.stack(
        [100 * x[..., 0], 100 * x[..., 1], x[..., 2], (1 - x[..., 2]) * x[..., 3]],
        axis=-1,
    )


@dataclass
class ExchangeFit:
    """Physical parameters and termination evidence of the lowest-SSE start.

    Attributes
    ----------
    params : ndarray
        Fp/PS in mL/min/100 mL, followed by dimensionless ve/vp.
    sse : float
        Direct unscaled residual sum of squares.
    status : int
        Solver termination status: zero is budget exhaustion; positive values
        indicate gradient, objective, step or combined objective/step stopping.
    nfev : int
        Total residual evaluations across all starts.
    optimality : float
        Scaled gradient norm at the selected start's termination.
    """

    params: np.ndarray
    sse: float
    status: int
    nfev: int
    optimality: float


def fit_native(
    context: NativeInput,
    curve: np.ndarray,
    previous: np.ndarray | None = None,
    extra: np.ndarray | None = None,
    max_nfev: int = 180,
) -> ExchangeFit:
    """Fit 2CXM from full generic starts and optional continuation.

    Parameters
    ----------
    context : NativeInput
        Prepared nonnegative-offset input and observation support.
    curve : ndarray
        Finite tissue concentrations at retained frames.
    previous : ndarray, optional
        Previous-delay Fp/PS/ve/vp solution, tried before generic starts.
    extra : ndarray, optional
        Additional physical starts, appended after the generic starts.
    max_nfev : int
        Residual evaluation budget per start, including retry budgets.

    Returns
    -------
    ExchangeFit
        Lowest direct-SSE physical solution, its status and total evaluation
        count. A lower-SSE unconverged start retains its own failure status.
    """
    curve = np.asarray(curve, dtype=float)
    for name, value in (("previous", previous), ("extra", extra)):
        if value is not None:
            value = np.asarray(value, dtype=float)
            valid_shape = (
                value.shape == (4,)
                if name == "previous"
                else value.ndim == 2 and value.shape[1] == 4
            )
            if not valid_shape or not np.isfinite(value).all():
                raise ValueError(f"Finite four-parameter {name} starts required.")
    lower, upper = np.zeros(4), np.array([5.0, 5.0, 1.0, 1.0])
    starts = STARTS if extra is None else np.concatenate([STARTS, extra])
    if previous is not None:
        starts = np.vstack([previous, starts])
    evaluator = Evaluator(context, curve)
    best = None
    total = 0
    for start in starts:
        x0 = np.clip(encode(start), lower + 1e-10, upper - 1e-10)
        result = solve(evaluator, x0, max_nfev=max_nfev)
        total += result.nfev
        params = decode(result.x)
        sse = float(np.sum((predict(context, params) - curve) ** 2))
        candidate = ExchangeFit(params, sse, result.status, total, result.optimality)
        if best is None or sse < best.sse:
            best = candidate
    assert best is not None
    best.nfev = total
    return best
