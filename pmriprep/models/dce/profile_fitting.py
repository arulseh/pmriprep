"""Fixed-delay kinetic fitting by profiling rate and solving linear coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.utils.dce_constants import EPSILON_VE_FORWARD, KTRANS_BOUNDS, VE_BOUNDS


@dataclass
class ToftsFit:
    """Physical Standard Tofts parameters and exact residual sums.

    Attributes
    ----------
    params : ndarray
        Member-by-parameter array: Ktrans in min^-1 followed by ve.
    sse : ndarray
        Residual sum of squares for each member at the supplied fixed delay.
    """

    params: np.ndarray
    sse: np.ndarray


def tofts_predict(context: DelayedPchip, params: np.ndarray) -> np.ndarray:
    """Evaluate Standard Tofts with no vascular term.

    Parameters
    ----------
    context : DelayedPchip
        Fixed plasma input, observation indices and delay.
    params : ndarray
        Ktrans in min^-1 and dimensionless ve.

    Returns
    -------
    ndarray
        Tissue concentration in mM at retained observations.
    """
    params = np.asarray(params, dtype=float)
    if params.shape != (2,) or not np.isfinite(params).all():
        raise ValueError("Finite Ktrans and ve parameters required.")
    ktrans, ve = params
    if not (KTRANS_BOUNDS.min <= ktrans <= KTRANS_BOUNDS.max
            and VE_BOUNDS.min <= ve <= VE_BOUNDS.max):
        raise ValueError("Tofts parameters are outside their physical bounds.")
    if ve <= 0 or ktrans == 0:
        return np.zeros(len(context.output_t))
    return np.asarray(
        ktrans * context.convolution(ktrans / max(ve, EPSILON_VE_FORWARD))
    )


def fit_tofts(context: DelayedPchip, curves: np.ndarray) -> ToftsFit:
    """Fit Standard Tofts by bounded rate profiling and scalar refinement.

    Parameters
    ----------
    context : DelayedPchip
        Fixed input and delay.
    curves : ndarray
        Finite member curves, members by retained frames.

    Returns
    -------
    ToftsFit
        Ktrans in [0,5], ve in [0,1], and final residual SSE.
    """
    y = np.asarray(curves, float)
    if y.ndim != 2 or y.shape[1] != len(context.output_t) or not np.isfinite(y).all():
        raise ValueError("Finite member-by-frame curves required.")
    if y.shape[1] == 0:
        raise ValueError("At least one retained observation is required.")
    if len(y) == 0:
        return ToftsFit(np.empty((0, 2)), np.empty(0))
    rates = np.r_[0.0, np.geomspace(1e-10, KTRANS_BOUNDS.max / EPSILON_VE_FORWARD, 256)]
    basis = context.convolution(rates)
    norms = np.maximum(np.sum(basis**2, axis=1), 1e-300)
    kt = np.clip(
        (basis @ y.T) / norms[:, None],
        rates[:, None] * EPSILON_VE_FORWARD,
        np.minimum(rates[:, None] * VE_BOUNDS.max, KTRANS_BOUNDS.max),
    )
    errors = np.sum((kt[:, :, None] * basis[:, None] - y[None]) ** 2, axis=2)
    indices = np.argmin(errors, axis=0)
    best = errors[indices, np.arange(len(y))].copy()
    best_k = kt[indices, np.arange(len(y))].copy()
    best_rate = rates[indices].copy()
    for v in range(len(y)):
        minima = (
            np.flatnonzero(
                (errors[2:-1, v] < errors[1:-2, v]) & (errors[2:-1, v] <= errors[3:, v])
            )
            + 2
        )
        if 1 < indices[v] < len(rates) - 1:
            minima = np.unique(np.r_[minima, indices[v]])

        def candidate(log_rate: float) -> tuple[float, float]:
            r = float(np.exp(log_rate))
            b = context.convolution(r)
            k = float(
                np.clip(
                    (b @ y[v]) / max(float(b @ b), 1e-300),
                    r * EPSILON_VE_FORWARD,
                    min(r * VE_BOUNDS.max, KTRANS_BOUNDS.max),
                )
            )
            return float(np.sum((k * b - y[v]) ** 2)), k

        for index in minima:
            result = minimize_scalar(
                lambda z: candidate(z)[0],
                bounds=(np.log(rates[index - 1]), np.log(rates[index + 1])),
                method="bounded",
                options=dict(xatol=1e-12, maxiter=100),
            )
            err, k = candidate(float(result.x))
            if result.success and err < best[v]:
                best[v], best_k[v], best_rate[v] = err, k, np.exp(result.x)
    ve = np.divide(best_k, best_rate, out=np.zeros_like(best_k), where=best_rate > 0)
    params = np.column_stack((best_k, ve))
    predicted = np.stack([tofts_predict(context, p) for p in params])
    return ToftsFit(params, np.sum((predicted - y) ** 2, axis=1))
