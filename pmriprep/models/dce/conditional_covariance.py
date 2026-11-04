"""Domain-aware conditional covariance at fixed kinetic family and delay."""

from typing import Callable

import numpy as np


def feasible(p: np.ndarray, upper: np.ndarray, volumes: tuple[int, ...]) -> bool:
    """Check the parameter box and, where present, joint volume constraint.

    Parameters
    ----------
    p, upper : ndarray
        Physical coordinates and their upper bounds; lower bounds are zero.
    volumes : tuple of int
        Coordinates constrained to sum to at most one, or empty.

    Returns
    -------
    bool
        Whether every coordinate lies in the permitted domain.
    """
    return bool(np.isfinite(p).all() and (p >= 0).all() and (p <= upper).all()
                and (not volumes or p[list(volumes)].sum() <= 1))


def jacobian(p: np.ndarray, upper: np.ndarray, volumes: tuple[int, ...],
             predict: Callable[[np.ndarray], np.ndarray], factor: float = 1.) -> tuple[np.ndarray, list[str]]:
    """Differentiate within the domain using fixed nominal steps.

    Parameters
    ----------
    p, upper : ndarray
        Physical coordinates and their upper bounds; lower bounds are zero.
    volumes : tuple
        Coordinates constrained to sum to at most one, or empty.
    predict : callable
        Original forward operator at fixed delay.
    factor : float
        One or one half for the declared stability comparison.

    Returns
    -------
    jac, schemes
        Derivatives and per-coordinate stencil; unavailable columns are NaN.
    """
    base = np.asarray(predict(p)).reshape(-1)
    jac = np.full((len(base), len(p)), np.nan)
    schemes = []
    if not feasible(p, upper, volumes) or not np.isfinite(base).all():
        return jac, ['unavailable']*len(p)
    for k in range(len(p)):
        h = factor*1e-4*max(abs(p[k]), 1e-3*upper[k])
        d = np.zeros_like(p)
        d[k] = h
        if feasible(p+d, upper, volumes) and feasible(p-d, upper, volumes):
            jac[:, k] = (predict(p+d)-predict(p-d))/(2*h)
            schemes.append('central')
        elif feasible(p+d, upper, volumes) and feasible(p+2*d, upper, volumes):
            jac[:, k] = (-3*base+4*predict(p+d)-predict(p+2*d))/(2*h)
            schemes.append('forward')
        elif feasible(p-d, upper, volumes) and feasible(p-2*d, upper, volumes):
            jac[:, k] = (3*base-4*predict(p-d)+predict(p-2*d))/(2*h)
            schemes.append('backward')
        else:
            schemes.append('unavailable')
    return jac, schemes


def covariance(jac: np.ndarray, sse: float) -> tuple[np.ndarray, float, str]:
    """Compute full covariance or an explicit unavailable reason.

    Parameters
    ----------
    jac : ndarray
        Observation-by-parameter derivative in physical coordinates.
    sse : float
        Residual sum of squares at the conditional estimate.

    Returns
    -------
    covariance : ndarray
        Full covariance scaled by SSE/(n-k), or NaNs when unavailable.
    rcond : float
        Squared smallest-to-largest singular-value ratio after column scaling.
    reason : str
        ``ok``, ``rank_deficient`` or ``covariance_unavailable``.
    """
    n, k = jac.shape
    missing = np.full((k, k), np.nan)
    if not np.isfinite(jac).all() or n <= k or not np.isfinite(sse) or sse < 0:
        return missing, 0., 'covariance_unavailable'
    norms = np.linalg.norm(jac, axis=0)
    if (norms == 0).any():
        return missing, 0., 'rank_deficient'
    _, s, vt = np.linalg.svd(jac/norms, full_matrices=False)
    rcond = float((s[-1]/s[0])**2)
    if rcond < 1e-10 or s[-1] <= np.finfo(float).eps*max(n, k)*s[0]:
        return missing, rcond, 'rank_deficient'
    return sse/(n-k)*(vt.T/s**2)@vt/norms[:, None]/norms[None, :], rcond, 'ok'
