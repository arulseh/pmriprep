"""Extended-Tofts evidence required for automatic exchange-model admission."""

from dataclasses import dataclass

import numpy as np

from pmriprep.models.dce.native_exchange import NativeInput
from pmriprep.utils.dce_constants import KTRANS_BOUNDS


@dataclass
class ExchangeAdmission:
    """Detection and residual evidence at the exchange model's selected delay.

    Attributes
    ----------
    g1, g2 : bool
        ET two-term detection and the one-parameter BIC improvement tests.
    reason : str
        Detection failure, insufficient improvement, or successful admission.
    se : ndarray
        Conditional physical ET standard errors for Ktrans, ve and vp.
    rcond, derivative_change : float
        Scaled information reciprocal condition and half-step derivative change.
        These are diagnostics, not additional admission thresholds.
    improvement : float
        n*log(SSE_ET/SSE_CX) - log(n); G2 requires strictly positive evidence.
    """

    g1: bool
    g2: bool
    reason: str
    se: np.ndarray
    rcond: float
    derivative_change: float
    improvement: float

    @property
    def passed(self) -> bool:
        """Return whether both admission conditions hold."""
        return self.g1 and self.g2


def exchange_admission(
    context: NativeInput, et_params: np.ndarray, et_sse: float, exchange_sse: float
) -> ExchangeAdmission:
    """Test ET term detection and exchange residual gain at one fixed delay.

    Parameters
    ----------
    context : NativeInput
        Common plasma input and retained observations at CX-selected delay.
    et_params : ndarray
        Conditional ET Ktrans (min^-1), ve and vp.
    et_sse, exchange_sse : float
        Conditional residual sums on the identical observation set.

    Returns
    -------
    ExchangeAdmission
        Frozen automatic-admission evidence, separate from public fit QC.

    Notes
    -----
    Covariance uses n-3 even if a boundary coordinate is inactive. The active
    derivative convention, rank check and two-SE margins belong to admission;
    the primary-parameter identification mask has a separate specification.
    """
    p = np.asarray(et_params, dtype=float)
    n = len(context.output_t)
    upper = np.array([KTRANS_BOUNDS.max, 1.0, 1.0])
    se = np.full(3, np.nan)
    rcond, maximum = 0.0, np.nan
    improvement = np.nan
    if (
        np.isfinite(et_sse)
        and np.isfinite(exchange_sse)
        and min(et_sse, exchange_sse) >= 0
        and n > 0
    ):
        improvement = float(
            n * np.log(max(et_sse, 1e-30) / max(exchange_sse, 1e-30)) - np.log(n)
        )

    def result(g1: bool, reason: str) -> ExchangeAdmission:
        """Retain diagnostics independently of the final admission result."""
        g2 = bool(np.isfinite(improvement) and improvement > 0)
        return ExchangeAdmission(
            g1,
            g2,
            reason if not g1 else "admitted" if g2 else "insufficient_exchange_gain",
            se,
            rcond,
            maximum,
            improvement,
        )

    if (
        p.shape != (3,)
        or not np.isfinite(p).all()
        or not np.isfinite(et_sse)
        or et_sse < 0
        or n <= 3
        or np.any(p < 0)
        or np.any(p > upper)
    ):
        return result(False, "reduced_model_unavailable")
    active = (p > 1e-6) & (upper - p > 1e-6)
    ids = np.flatnonzero(active)
    gap = 1 - p[[1, 2]].sum()
    if gap <= 1e-6 or not len(ids):
        return result(False, "ET_bound_or_uncertainty_unavailable")
    jac = np.full((n, 3), np.nan)
    maximum = 0.0

    def predict(point: np.ndarray) -> np.ndarray:
        """Evaluate ET in physical coordinates without a fit or reparameterization."""
        kt, ve, vp = point
        rate = kt / ve if ve > 0 else 0.0
        return np.asarray(kt * context.convolution(rate) + vp * context.cp)

    for i in ids:
        step = min(1e-4 * abs(p[i]), 0.25 * p[i], 0.25 * (upper[i] - p[i]))
        if i in (1, 2):
            step = min(step, 0.25 * gap)
        if step <= 0 or p[i] + step == p[i]:
            return result(False, "ET_bound_or_uncertainty_unavailable")
        perturbation = np.zeros(3)
        perturbation[i] = step
        column = (predict(p + perturbation) - predict(p - perturbation)) / (2 * step)
        half = (predict(p + perturbation / 2) - predict(p - perturbation / 2)) / step
        jac[:, i] = column
        maximum = max(
            maximum,
            float(np.linalg.norm(column - half))
            / max(float(np.linalg.norm(half)), 1e-30),
        )
    matrix = jac[:, ids]
    norms = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(matrix).all() or np.any(norms == 0):
        return result(False, "ET_bound_or_uncertainty_unavailable")
    _, singular, vt = np.linalg.svd(matrix / norms, full_matrices=False)
    rcond = float((singular[-1] / singular[0]) ** 2)
    if singular[-1] <= np.finfo(float).eps * max(matrix.shape) * singular[0]:
        return result(False, "ET_bound_or_uncertainty_unavailable")
    inverse = (vt.T / singular**2) @ vt
    local = et_sse / (n - 3) * inverse / norms[:, None] / norms[None, :]
    se[ids] = np.sqrt(np.maximum(np.diag(local), 0.0))
    if not np.isfinite(se[[0, 2]]).all():
        return result(False, "ET_bound_or_uncertainty_unavailable")
    margins = np.maximum(2 * se, 1e-6 * upper)
    for coordinate, label in ((0, "Ktrans"), (2, "vp")):
        if (
            p[coordinate] < 2 * se[coordinate]
            or p[coordinate] <= margins[coordinate]
            or upper[coordinate] - p[coordinate] <= margins[coordinate]
        ):
            return result(False, f"ET_{label}_not_detected")
    return result(True, "two_terms_detected")
