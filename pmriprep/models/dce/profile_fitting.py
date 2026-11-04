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


_ET_RATE_GRID = np.geomspace(1e-10, KTRANS_BOUNDS.max / EPSILON_VE_FORWARD, 256)


@dataclass
class ProfileFit:
    """Fixed-delay ET or Patlak fit with direct residual sums.

    Attributes
    ----------
    params : ndarray
        Member-by-parameter array containing Ktrans (min^-1), ve and vp.
        Patlak ve is NaN because the model has no backflux compartment.
    sse : ndarray
        Residual sum of squares for each member. A finite fit alone does not
        establish parameter identification or physiological validity.
    """

    params: np.ndarray
    sse: np.ndarray


def _solve_box(
    aa: np.ndarray,
    ab: np.ndarray,
    bb: float,
    ya: np.ndarray,
    yb: np.ndarray,
    yy: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the two-column least-squares box exactly using interior and edges.

    Parameters
    ----------
    aa, ab, bb, ya, yb, yy : arrays or float
        Gram matrix entries, data cross-products and squared data norms.
    lower, upper : ndarray
        Bounds on the first coefficient; the second is bounded by [0, 1].

    Returns
    -------
    tuple of ndarray
        SSE and the two bounded coefficients, in broadcast-compatible shape.
    """
    safe_aa = np.maximum(aa, 1e-300)
    safe_bb = max(bb, 1e-300)
    determinant = aa * bb - ab**2
    regular = determinant > 1e-14 * np.maximum(aa * bb, 1e-300)
    safe_det = np.where(regular, determinant, 1.0)
    k = (ya * bb - yb * ab) / safe_det
    v = (yb * aa - ya * ab) / safe_det
    admissible = regular & (k >= lower) & (k <= upper) & (v >= 0) & (v <= 1)

    def error(kt: np.ndarray, vp: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.maximum(
                yy
                - 2 * kt * ya
                - 2 * vp * yb
                + kt**2 * aa
                + 2 * kt * vp * ab
                + vp**2 * bb,
                0.0,
            )
        )

    best = np.where(admissible, error(k, v), np.inf)
    for fixed_v in (0.0, 1.0):
        candidate_k = np.clip((ya - fixed_v * ab) / safe_aa, lower, upper)
        candidate_v = np.full_like(candidate_k, fixed_v)
        candidate_error = error(candidate_k, candidate_v)
        use = candidate_error < best
        best, k, v = (
            np.where(use, candidate_error, best),
            np.where(use, candidate_k, k),
            np.where(use, candidate_v, v),
        )
    for fixed_k in (lower, upper):
        candidate_v = np.clip((yb - fixed_k * ab) / safe_bb, 0.0, 1.0)
        candidate_k = np.broadcast_to(fixed_k, candidate_v.shape)
        candidate_error = error(candidate_k, candidate_v)
        use = candidate_error < best
        best, k, v = (
            np.where(use, candidate_error, best),
            np.where(use, candidate_k, k),
            np.where(use, candidate_v, v),
        )
    return best, k, v


def _solve_et_polygon(
    aa: np.ndarray,
    ab: np.ndarray,
    bb: float,
    ya: np.ndarray,
    yb: np.ndarray,
    yy: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the kinetic polygon including Ktrans/kep + vp <= 1.

    Parameters
    ----------
    aa, ab, bb, ya, yb, yy : arrays or float
        Design cross-products for convolution and vascular columns.
    lower, upper : ndarray
        Ktrans bounds; the lower bound equals kep times the numerical ve floor.

    Returns
    -------
    tuple of ndarray
        SSE, Ktrans and vp for the best interior or boundary solution.
    """
    rate = lower / EPSILON_VE_FORWARD
    safe_rate = np.where(rate > 0, rate, 1.0)
    safe_aa, safe_bb = np.maximum(aa, 1e-300), max(bb, 1e-300)
    determinant = aa * bb - ab**2
    regular = determinant > 1e-14 * np.maximum(aa * bb, 1e-300)
    safe_det = np.where(regular, determinant, 1.0)
    k, v = (ya * bb - yb * ab) / safe_det, (yb * aa - ya * ab) / safe_det
    admissible = (
        regular & (k >= lower) & (k <= upper) & (v >= 0) & (v + k / safe_rate <= 1)
    )

    def error(kt: np.ndarray, vp: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.maximum(
                yy
                - 2 * kt * ya
                - 2 * vp * yb
                + kt**2 * aa
                + 2 * kt * vp * ab
                + vp**2 * bb,
                0.0,
            )
        )

    best = np.where(admissible, error(k, v), np.inf)
    candidates = [(np.clip(ya / safe_aa, lower, upper), np.zeros_like(ya))]
    for fixed_k in (lower, upper):
        candidate_v = np.clip(
            (yb - fixed_k * ab) / safe_bb, 0.0, np.maximum(1 - fixed_k / safe_rate, 0.0)
        )
        candidates.append((np.broadcast_to(fixed_k, candidate_v.shape), candidate_v))
    slope_norm = aa - 2 * ab / safe_rate + bb / safe_rate**2
    slope_valid = slope_norm > 1e-14 * np.maximum(aa + bb / safe_rate**2, 1e-300)
    slope_k = np.clip(
        (ya - yb / safe_rate - ab + bb / safe_rate)
        / np.where(slope_valid, slope_norm, 1.0),
        lower,
        upper,
    )
    candidates.append((slope_k, np.maximum(1 - slope_k / safe_rate, 0.0)))
    for candidate_k, candidate_v in candidates:
        candidate_error = error(candidate_k, candidate_v)
        use = candidate_error < best
        best, k, v = (
            np.where(use, candidate_error, best),
            np.where(use, candidate_k, k),
            np.where(use, candidate_v, v),
        )
    return best, k, v


def fit_extended_tofts(context: DelayedPchip, y: np.ndarray) -> ProfileFit:
    """Profile detected rate minima with constrained linear Ktrans/vp solutions.

    Parameters
    ----------
    context : DelayedPchip
        Fixed plasma input, delay and observation indices.
    y : ndarray
        Finite tissue concentration curves, members by retained observations.

    Returns
    -------
    ProfileFit
        Ktrans in [0, 5] min^-1, ve/vp in [0, 1], with ve + vp <= 1.

    Notes
    -----
    At fixed kep, the positive-ve bound becomes kep*EPSILON_VE_FORWARD <= Ktrans <= kep.
    Detected local minima on a 256-point logarithmic rate grid are bracketed
    and refined by 38 golden-section iterations. Rate zero is tested
    explicitly. Zero-Ktrans fits report ve zero, which is then unidentifiable.
    Final SSE is recomputed from residuals, not Gram sums.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(context.cp) or not np.isfinite(y).all():
        raise ValueError("Finite member-by-frame curves required.")
    if y.shape[1] == 0:
        raise ValueError("At least one retained observation is required.")
    if len(y) == 0:
        return ProfileFit(np.empty((0, 3)), np.empty(0))
    cp = context.cp
    rates = np.r_[0.0, _ET_RATE_GRID]
    basis = context.convolution(rates)
    bb = float(cp @ cp)
    yb = y @ cp
    yy = np.sum(y * y, axis=1)
    aa = np.sum(basis * basis, axis=1)
    ab = basis @ cp
    ya = basis @ y.T
    costs, kt, vp = _solve_et_polygon(
        aa[:, None],
        ab[:, None],
        bb,
        ya,
        yb[None, :],
        yy[None, :],
        rates[:, None] * EPSILON_VE_FORWARD,
        np.minimum(rates[:, None], KTRANS_BOUNDS.max),
    )
    best_index = np.argmin(costs, axis=0)
    voxel = np.arange(len(y))
    best_rates = rates[best_index].copy()
    best_kt, best_vp = kt[best_index, voxel].copy(), vp[best_index, voxel].copy()
    best_sse = costs[best_index, voxel].copy()
    candidate = (costs[2:-1] < costs[1:-2]) & (costs[2:-1] <= costs[3:])
    candidate &= costs[2:-1] < np.minimum(costs[1:-2], costs[3:]) - 1e-13 * np.maximum(
        yy, 1e-10
    )
    grid_indices, voxel_indices = np.where(candidate)
    grid_indices += 2
    interior = (best_index > 1) & (best_index < len(rates) - 1)
    pairs = np.unique(
        np.column_stack(
            (
                np.r_[grid_indices, best_index[interior]],
                np.r_[voxel_indices, voxel[interior]],
            )
        ),
        axis=0,
    )
    if len(pairs):
        ix, vox = pairs[:, 0], pairs[:, 1]
        left, right = np.log(rates[ix - 1]), np.log(rates[ix + 1])

        def paired(log_rate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            rate = np.exp(log_rate)
            column = context.convolution(rate)
            return _solve_et_polygon(
                np.sum(column**2, axis=1),
                column @ cp,
                bb,
                np.sum(column * y[vox], axis=1),
                yb[vox],
                yy[vox],
                rate * EPSILON_VE_FORWARD,
                np.minimum(rate, KTRANS_BOUNDS.max),
            )

        ratio = (np.sqrt(5.0) - 1) / 2
        x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
        f1, _, _ = paired(x1)
        f2, _, _ = paired(x2)
        for _ in range(38):
            choose_left = f1 < f2
            right = np.where(choose_left, x2, right)
            left = np.where(choose_left, left, x1)
            old_x1, old_x2, old_f1, old_f2 = x1, x2, f1, f2
            fresh = np.where(
                choose_left,
                right - ratio * (right - left),
                left + ratio * (right - left),
            )
            value, _, _ = paired(fresh)
            x1 = np.where(choose_left, fresh, old_x2)
            x2 = np.where(choose_left, old_x1, fresh)
            f1 = np.where(choose_left, value, old_f2)
            f2 = np.where(choose_left, old_f1, value)
        final = (left + right) / 2
        error, k, v = paired(final)
        for j, target in enumerate(vox):
            if error[j] < best_sse[target]:
                best_sse[target], best_kt[target], best_vp[target] = (
                    error[j],
                    k[j],
                    v[j],
                )
                best_rates[target] = np.exp(final[j])
    ve = np.divide(
        best_kt, best_rates, out=np.zeros_like(best_kt), where=best_rates > 0
    )
    ve = np.where(best_kt > 0, np.clip(ve, EPSILON_VE_FORWARD, 1.0), 0.0)
    predicted = (
        best_kt[:, None] * context.convolution(best_rates) + best_vp[:, None] * cp
    )
    return ProfileFit(
        np.column_stack((best_kt, ve, best_vp)), np.sum((predicted - y) ** 2, axis=1)
    )


def fit_patlak(context: DelayedPchip, y: np.ndarray) -> ProfileFit:
    """Solve bounded Patlak kinetics without changing observation support.

    Parameters
    ----------
    context : DelayedPchip
        Fixed plasma input, delay and observation indices.
    y : ndarray
        Finite tissue concentration curves, members by retained observations.

    Returns
    -------
    ProfileFit
        Ktrans in [0, 5] min^-1, undefined ve (NaN), vp in [0, 1], and
        residual SSE for each member. No backflux parameter is fitted.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(context.cp) or not np.isfinite(y).all():
        raise ValueError("Finite member-by-frame curves required.")
    if y.shape[1] == 0:
        raise ValueError("At least one retained observation is required.")
    if len(y) == 0:
        return ProfileFit(np.empty((0, 3)), np.empty(0))
    cp, area = context.cp, context.convolution(0.0)
    _, kt, vp = _solve_box(
        np.array(area @ area),
        np.array(area @ cp),
        float(cp @ cp),
        y @ area,
        y @ cp,
        np.sum(y * y, axis=1),
        np.array(KTRANS_BOUNDS.min),
        np.array(KTRANS_BOUNDS.max),
    )
    predicted = kt[:, None] * area + vp[:, None] * cp
    return ProfileFit(
        np.column_stack((kt, np.full_like(kt, np.nan), vp)),
        np.sum((predicted - y) ** 2, axis=1),
    )
