"""Memory-bounded ensemble evaluation of signal CPI."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import gammaincinv, gammaln, roots_jacobi

from .derivatives import plan_stencil
from .continuous import ContinuousCPIConvolution, polynomial_overlaps
from .dispersion import DispersionIntegrationError
from .prepared import PreparedCPIConvolution
from .probability import Array
from . import native

if TYPE_CHECKING:
    from .signal import FixedLayoutSignalCPI


@dataclass(frozen=True)
class PlainGeometry:
    """Immutable overlap arrays for one exact knot layout and elapsed-time row."""

    frames: np.ndarray
    residue: np.ndarray
    half: Array
    local: Array
    aif: Array

    @property
    def nbytes(self) -> int:
        """Return owned array storage for the bounded geometry cache."""
        return sum(array.nbytes for array in
                   (self.frames, self.residue, self.half, self.local, self.aif))


class EnsembleConvolution:
    """Integrate multiple parameter vectors sharing an acquired AIF/time grid.

    Parameters
    ----------
    base : ContinuousCPIConvolution
        Verified AIF reconstruction, support and polynomial coefficients.
    batch_size : int, optional
        Maximum parameter vectors evaluated in one integration block. This is
        a memory setting, not a model parameter or an accuracy setting.

    Notes
    -----
    Plain convolution batches exact polynomial overlap integrals. Gamma
    convolution batches adaptive quadrature across curves and time points,
    with independent per-curve error targets and refinement budgets. No curve
    may borrow another curve's larger signal scale to pass its error check.
    """

    def __init__(
        self,
        base: ContinuousCPIConvolution,
        batch_size: int = 8,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch size must be a positive integer")
        self.base, self.batch_size = base, batch_size
        self.rules = {n: np.polynomial.legendre.leggauss(n) for n in (16, 32)}
        self._geometry_cache: OrderedDict[tuple[str, bytes, str, bytes], PlainGeometry] = OrderedDict()
        self._geometry_bytes = 0

    def clear_transient_state(self) -> None:
        """Discard history-dependent workspaces between independent fits."""
        self._geometry_cache.clear()
        self._geometry_bytes = 0

    def evaluate(
        self,
        knots: Array,
        values: Array,
        flow: Array,
        delay: Array,
        gamma: Array | None = None,
    ) -> Array:
        """Return one concentration curve per parameter row, in input order."""
        if (
            knots.ndim != 2
            or knots.shape != values.shape
            or knots.shape[1] < 2
            or flow.shape != (knots.shape[0],)
            or delay.shape != flow.shape
            or not all(np.all(np.isfinite(x)) for x in (knots, values, flow, delay))
            or np.any(knots[:, 0] != 0)
            or np.any(np.diff(knots, axis=1) <= 0)
        ):
            raise ValueError(
                "finite ordered residue rows and matching flow/delay required"
            )
        if gamma is not None and (
            gamma.shape != (knots.shape[0], 2)
            or not np.all(np.isfinite(gamma))
            or np.any(gamma[:, 0] < 0)
            or np.any(gamma[:, 1] <= 0)
            or not np.all(np.isfinite(1 + gamma[:, 0] * gamma[:, 1]))
        ):
            raise ValueError(
                "finite nonnegative gamma modes and positive rates required"
            )
        result = np.zeros((knots.shape[0], self.base.time.size))
        for first in range(0, knots.shape[0], self.batch_size):
            block = slice(first, first + self.batch_size)
            active = np.flatnonzero(flow[block] != 0) + first
            if not active.size:
                continue
            target = self.base.time[None, :] - delay[active, None]
            unit = (
                self._plain(knots[active], values[active], target)
                if gamma is None
                else self._gamma(knots[active], values[active], target, gamma[active])
            )
            result[active] = flow[active, None] * unit
        return result

    def _plain(self, knots: Array, values: Array, target: Array) -> Array:
        """Evaluate unchanged polynomial overlaps using the available backend."""
        if (native.available() and len(knots)
                and all(a.dtype == np.float64 for a in (knots, values, target))):
            coefficients = np.stack([
                np.column_stack((PchipInterpolator(x, y).c, [0., 0., 0., y[-1]]))
                for x, y in zip(knots, values, strict=True)
            ])
            arterial = (np.vstack((self.base.slope, self.base.amplitude))
                        if self.base.aif_coefficients is None else self.base.aif_coefficients)
            result = native.convolve(knots, coefficients, target, self.base.left,
                self.base.right, arterial, self.base.nodes, self.base.weights)
            if result is not None:
                return result
        return self._plain_python(knots, values, target)

    def _plain_python(self, knots: Array, values: Array, target: Array) -> Array:
        """Reuse identical overlap geometry while recomputing each residue."""
        result = np.zeros(target.shape, dtype=float)
        for row in range(len(knots)):
            geometry = self._plain_geometry(knots[row], target[row])
            coefficients = np.column_stack((PchipInterpolator(knots[row], values[row]).c,
                                             [0., 0., 0., values[row, -1]]))
            c = coefficients[:, geometry.residue].T
            local = geometry.local
            residue = ((c[:, 0, None]*local+c[:, 1, None])*local+c[:, 2, None])*local+c[:, 3, None]
            integral = geometry.half*((geometry.aif*residue) @ self.base.weights)
            result[row] = np.bincount(geometry.frames, weights=integral, minlength=target.shape[1])
        return result

    def _plain_geometry(self, knots: Array, target: Array) -> PlainGeometry:
        """Cache exact, owned geometry within sixteen entries and eight MB."""
        key = (knots.dtype.str, knots.tobytes(), target.dtype.str, target.tobytes())
        if key in self._geometry_cache:
            self._geometry_cache.move_to_end(key)
            return self._geometry_cache[key]
        base = self.base
        _, frames, ai, ri, lo, hi = polynomial_overlaps(base.left, base.right,
                                                        knots[None], target[None])
        half = (hi - lo) / 2
        points = (hi + lo)[:, None] / 2 + half[:, None] * base.nodes
        local = points - knots[ri, None]
        aif_local = target[frames, None] - points - base.left[ai, None]
        if base.aif_coefficients is None:
            aif = base.amplitude[ai, None] + base.slope[ai, None] * aif_local
        else:
            ac = base.aif_coefficients[:, ai]
            aif = (
                (ac[0, :, None] * aif_local + ac[1, :, None]) * aif_local
                + ac[2, :, None]
            ) * aif_local + ac[3, :, None]
        arrays = (frames.copy(), ri.copy(), half, local, aif)
        for array in arrays:
            array.flags.writeable = False
        geometry = PlainGeometry(*arrays)
        size = geometry.nbytes + sum(map(len, key))
        if size <= 8_000_000:
            while (self._geometry_cache and
                   (len(self._geometry_cache) >= 16 or self._geometry_bytes+size > 8_000_000)):
                old_key, old = self._geometry_cache.popitem(last=False)
                self._geometry_bytes -= old.nbytes + sum(map(len, old_key))
            self._geometry_cache[key] = geometry
            self._geometry_bytes += size
        return geometry

    def _gamma(self, knots: Array, values: Array, target: Array, gamma: Array) -> Array:
        """Batch checked gamma integrals on exact response-polynomial intervals."""
        if not self.base.left.size:
            return np.zeros_like(target)
        # Prepare each response once. All quadrature-node work below is batched.
        prepared = PreparedCPIConvolution.from_operator(self.base)
        shape = 1 + gamma[:, 0] * gamma[:, 1]
        rate = gamma[:, 1]
        probabilities = np.array(
            [1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6, 1 - 1e-12]
        )
        transport = gammaincinv(shape[:, None], probabilities) / rate[:, None]
        parts = []
        for row in range(target.shape[0]):
            if np.max(target[row]) <= self.base.left[0]:
                continue
            prepared._prepare(knots[row], values[row], float(np.max(target[row])))
            edges = prepared._breaks
            cuts = np.sort(
                np.clip(
                    np.column_stack(
                        (
                            target[row, :, None] - edges[None, :],
                            np.broadcast_to(
                                transport[row], (target.shape[1], probabilities.size)
                            ),
                        )
                    ),
                    0,
                    np.maximum(target[row] - self.base.left[0], 0)[:, None],
                ),
                axis=1,
            )
            lower, upper = cuts[:, :-1], cuts[:, 1:]
            frames, columns = np.nonzero(upper > lower)
            lo, hi = lower[frames, columns], upper[frames, columns]
            midpoint = target[row, frames] - (lo + hi) / 2
            indices = np.clip(
                np.searchsorted(edges, midpoint, side="right") - 1, 0, edges.size - 2
            )
            parts.append(
                (
                    np.full(frames.size, row),
                    frames,
                    lo,
                    hi,
                    edges[indices],
                    edges[indices + 1],
                    prepared._coefficients[:, indices].T,
                )
            )
        if not parts:
            return np.zeros_like(target)
        rows, frames, lo, hi, poly_lo, poly_hi, coefficients = (
            np.concatenate([part[i] for part in parts]) for i in range(7)
        )
        jacobi = {}
        unique_shapes, shape_index = np.unique(shape, return_inverse=True)
        for order in self.rules:
            nodes, weights = [], []
            for value in unique_shapes:
                x, w = roots_jacobi(order, 0, value - 1)
                nodes.append(x)
                weights.append(w / np.sum(w))
            jacobi[order] = (np.array(nodes), np.array(weights))

        def integrate(order: int) -> Array:
            nodes, weights = self.rules[order]
            width = hi - lo
            lag = lo[:, None] + width[:, None] * (nodes + 1) / 2
            sh, ra = shape[rows, None], rate[rows, None]
            log_weights = (
                np.log(width[:, None] * weights / 2)
                + sh * np.log(ra)
                - gammaln(sh)
                + (sh - 1) * np.log(lag)
                - ra * lag
            )
            origin = lo == 0
            if np.any(origin):
                jnodes, jweights = jacobi[order]
                ids = shape_index[rows[origin]]
                lag[origin] = hi[origin, None] * (jnodes[ids] + 1) / 2
                log_weights[origin] = (
                    np.log(jweights[ids])
                    + sh[origin] * np.log(ra[origin] * hi[origin, None])
                    - gammaln(sh[origin] + 1)
                    - ra[origin] * lag[origin]
                )
            local = (
                2
                * (target[rows, frames, None] - lag - poly_lo[:, None])
                / (poly_hi - poly_lo)[:, None]
                - 1
            )
            response = np.polynomial.chebyshev.chebval(
                local, coefficients.T[:, :, None], tensor=False
            )
            return np.asarray(np.sum(response * np.exp(log_weights), axis=1))

        for _ in range(30):
            coarse, fine = integrate(16), integrate(32)
            error = 4 * np.abs(fine - coarse)
            ids = rows * target.shape[1] + frames
            total = np.bincount(ids, weights=fine, minlength=target.size).reshape(
                target.shape
            )
            errors = np.bincount(ids, weights=error, minlength=target.size).reshape(
                target.shape
            )
            tolerance = np.maximum(1e-200, 2e-12 * np.max(np.abs(total), axis=1) / 8)
            if np.all(np.isfinite(total)) and np.all(errors <= tolerance[:, None]):
                return np.asarray(total, dtype=float)
            count = np.bincount(ids, minlength=target.size)
            refine = error > tolerance[rows] / np.maximum(1, count[ids])
            sizes = np.bincount(rows, minlength=target.shape[0]) + np.bincount(
                rows[refine], minlength=target.shape[0]
            )
            if not np.any(refine) or np.any(sizes > 100000):
                break
            middle = (lo[refine] + hi[refine]) / 2
            lo, hi = (
                np.r_[lo[~refine], lo[refine], middle],
                np.r_[hi[~refine], middle, hi[refine]],
            )
            rows, frames, poly_lo, poly_hi, coefficients = (
                np.concatenate((x[~refine], x[refine], x[refine]))
                for x in (rows, frames, poly_lo, poly_hi, coefficients)
            )
        raise DispersionIntegrationError(
            "ensemble gamma quadrature did not meet integration accuracy"
        )


class SignalBatchEvaluator:
    """Batch predictions and both levels of the VI finite-difference work.

    Exact duplicate unit-concentration requests share one evaluation; flow
    and baseline signal remain distinct analytic factors in each signal row.
    No Jacobian column or covariance-trace contribution is frozen or omitted.
    """

    def __init__(
        self,
        model: FixedLayoutSignalCPI,
        batch_size: int = 8,
    ) -> None:
        if model.continuous is None:
            raise ValueError("batch evaluation requires continuous integration")
        self.model = model
        self.convolution = EnsembleConvolution(model.continuous, batch_size)
        self.domain = getattr(model, "derivative_domain", model.valid_mean)

    def clear_transient_state(self) -> None:
        """Discard convolution workspaces before an independent fit."""
        self.convolution.clear_transient_state()

    def concentration(self, means: Array) -> Array:
        """Evaluate ordered parameter rows without sharing their fit state."""
        if (
            means.ndim != 2
            or means.shape[1:] != self.model.prior.mean.shape
            or not np.all(np.isfinite(means))
        ):
            raise ValueError("finite parameter matrix required")
        if not means.shape[0]:
            return np.empty((0, self.model.time.size))
        return self.convolution.evaluate(
            np.array([self.model.knot_times(m) for m in means]),
            np.array([self.model.knot_values(m) for m in means]),
            means[:, 0],
            means[:, self.model.delay_index],
            np.exp(means[:, -2:]) if self.model.model == "cpi_vtf" else None,
        )

    def predictions(self, means: Array) -> Array:
        """Return measured-signal predictions for a parameter ensemble."""
        return means[:, 1, None] * np.exp(-self.model.te * self.concentration(means))

    def predict(self, mean: Array) -> Array:
        """Provide the scalar engine adapter to the batched forward evaluator."""
        return np.asarray(self.predictions(mean[None, :])[0])

    def jacobians(self, means: Array) -> Array:
        """Evaluate chain-rule derivatives with checked gamma-kernel reuse."""
        gamma = self._transport_coordinates()
        try:
            return self._chain_rule_jacobians(means, gamma)
        except DispersionIntegrationError:
            if gamma is None:
                raise
            # Failure to certify an auxiliary kernel does not invalidate the
            # original parameter-domain finite-difference calculation.
            return self._chain_rule_jacobians(means, None)

    def _transport_coordinates(self) -> tuple[int, int] | None:
        """Return log-mode/log-rate indices when adjacent-shape reuse is supported."""
        return None

    def _chain_rule_jacobians(
        self, means: Array, gamma: tuple[int, int] | None,
    ) -> Array:
        """Combine fourth-order shape derivatives with analytic signal factors.

        Parameters
        ----------
        means : ndarray
            Parameter rows in the model's inferred coordinates.
        gamma : tuple of int or None
            Log-mode/log-rate indices, or None for independent stencils. For
            shape a=1+mode*rate, the rate partial is the mode partial plus
            a*(C_a-C_(a+1)), where the auxiliary response keeps rate fixed.
            Auxiliary kernels are integration terms, not fitted parameter rows.

        Returns
        -------
        ndarray
            Signal Jacobians, with shape (rows, frames, parameters).
        """
        if (
            means.ndim != 2
            or means.shape[1:] != self.model.prior.mean.shape
            or not np.all(np.isfinite(means))
        ):
            raise ValueError("finite parameter matrix required")
        columns = [i for i in range(2, means.shape[1]) if gamma is None or i != gamma[1]]
        sensitivity = np.ones(len(means))
        auxiliary: list[Array] = []
        if gamma is not None:
            mode, rate = gamma
            shape = 1 + np.exp(means[:, mode]) * np.exp(means[:, rate])
            # Scale mode perturbations by their shift relative to kernel SD;
            # the rate identity subtracts two potentially large partials.
            sensitivity = np.maximum(1., (shape - 1) / np.sqrt(shape))
            adjacent = means.copy()
            adjacent[:, mode] = np.logaddexp(means[:, mode], -means[:, rate])
            auxiliary.append(adjacent)
        plans = [
            [
                plan_stencil(mean, i, max(1.0, abs(mean[i])) /
                             (sensitivity[row] if gamma is not None and i == gamma[0] else 1.),
                             self.domain)
                for i in columns
            ]
            for row, mean in enumerate(means)
        ]
        if not plans:
            return np.empty((0, self.model.time.size, self.model.prior.mean.size))
        requests = np.concatenate((means, *auxiliary, *[p.points for row in plans for p in row]))
        requests[:, :2] = 1.0
        unique, inverse = np.unique(requests, axis=0, return_inverse=True)
        curves = self.concentration(unique)[inverse]
        units = curves[: means.shape[0]]
        attenuation = np.exp(-self.model.te * means[:, 0, None] * units)
        prediction = means[:, 1, None] * attenuation
        result = np.empty((means.shape[0], self.model.time.size, means.shape[1]))
        result[:, :, 0] = -self.model.te * units * prediction
        result[:, :, 1] = attenuation
        index = means.shape[0] * (1 + len(auxiliary))
        for row, row_plans in enumerate(plans):
            for column, plan in zip(columns, row_plans, strict=True):
                derivative = plan.apply(curves[index : index + 4], units[row])
                result[row, :, column] = (
                    -self.model.te * means[row, 0] * prediction[row] * derivative
                )
                index += 4
        if gamma is not None:
            adjacent_units = curves[len(means):2*len(means)]
            result[:, :, rate] = (result[:, :, mode] - self.model.te * means[:, 0, None]
                                   * prediction * shape[:, None] * (units - adjacent_units))
        return result

    def jacobian(self, mean: Array) -> Array:
        """Return one Jacobian through the ensemble implementation."""
        return np.asarray(self.jacobians(mean[None, :])[0])
