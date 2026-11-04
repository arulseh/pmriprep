"""Native bounded 2CXM fitting with owned held-endpoint PCHIP contexts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce._native_backend import Evaluator, solve
from pmriprep.models.dce.cached_input import CachedPchip as NativeInput
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


def predict(context: NativeInput | DelayedPchip, params: np.ndarray) -> np.ndarray:
    """Evaluate 2CXM at fixed delay; params are Fp, PS, ve and vp.

    Parameters
    ----------
    context : NativeInput or DelayedPchip
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
    context: NativeInput | DelayedPchip,
    curve: np.ndarray,
    previous: np.ndarray | None = None,
    extra: np.ndarray | None = None,
    max_nfev: int = 180,
) -> ExchangeFit:
    """Fit 2CXM from full generic starts and optional continuation.

    Parameters
    ----------
    context : NativeInput or DelayedPchip
        Prepared input and observation support. A general DelayedPchip permits
        signed diagnostic offsets without changing the delay-selection grid.
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
