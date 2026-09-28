"""Checked continuous gamma transport of the acquired arterial input."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import gammainc, gammaincc, gammainccinv, gammaincinv, gammaln, roots_jacobi

from pmriprep.models.dsc.signal_cpi.batch import EnsembleConvolution
from pmriprep.models.dsc.signal_cpi.continuous import (
    ContinuousCPIConvolution, IntegralGeometryCache, PolynomialIntegral,
)
from pmriprep.models.dsc.signal_cpi.dispersion import DispersionIntegrationError
from . import native
from .precision import TRANSPORT_INTERPOLATION_RELATIVE_ERROR, TRANSPORT_RAW_RELATIVE_ERROR
from .quadrature import IntervalReuseConvolution


_INTERVAL_BLOCK_SIZE = 4096


def _gamma_interval_cdfs(shape: float, lower: np.ndarray, upper: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate four adjacent gamma shapes at unique, unrounded endpoints."""
    endpoints, inverse = np.unique(np.concatenate((lower, upper)), return_inverse=True)
    a = shape + np.arange(4)[:, None]
    p, q = gammainc(a, endpoints[None]), gammaincc(a, endpoints[None])
    left, right = inverse[:lower.size], inverse[lower.size:]
    return p[:, left], p[:, right], q[:, left], q[:, right]


@dataclass(frozen=True)
class EndpointLayout:
    """Unscaled endpoint ordering, independent of gamma shape and rate."""

    points: np.ndarray
    inverse: np.ndarray


class GammaEndpointCache:
    """Bound exact endpoint layouts across kernels without caching CDF values."""

    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[bytes, ...], EndpointLayout] = OrderedDict()
        self.nbytes = 0

    def evaluate(self, shape: float, rate: float, lower: np.ndarray, upper: np.ndarray,
                 limit: float, *, layout_key: tuple[bytes, ...] | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute fresh CDFs after applying this kernel's exact upper cutoff.

        Parameters
        ----------
        shape, rate : float
            Positive gamma parameters; four adjacent shapes are evaluated.
        lower, upper : ndarray
            Nonnegative float64 lag endpoints before upper truncation. Active
            lower endpoints must lie below the current kernel cutoff.
        limit : float
            Current kernel's upper integration cutoff in unscaled lag units.
        layout_key : tuple of bytes, optional
            Exact generating coordinates and active-interval mask. These must
            uniquely determine the unscaled endpoints, without gamma parameters.

        Returns
        -------
        tuple of ndarray
            Lower-tail and upper-tail CDFs at both endpoint arrays.
        """
        if lower.dtype != np.float64 or upper.dtype != np.float64:
            # Preserve dtype-dependent scalar promotion before deduplication.
            return _gamma_interval_cdfs(shape, rate*lower, rate*np.minimum(upper, limit))
        key = ((b'endpoints', lower.tobytes(), upper.tobytes()) if layout_key is None
               else (b'intervals', *layout_key))
        layout = self.entries.get(key)
        if layout is None:
            points, inverse = np.unique(np.r_[lower, upper], return_inverse=True)
            if points.size <= np.iinfo(np.int32).max:
                inverse = inverse.astype(np.int32)
            points.flags.writeable = inverse.flags.writeable = False
            layout = EndpointLayout(points, inverse)
            size = points.nbytes + inverse.nbytes + sum(map(len, key))
            if size <= 8_000_000:
                while self.entries and (len(self.entries) >= 16 or self.nbytes+size > 8_000_000):
                    old_key, old = self.entries.popitem(last=False)
                    self.nbytes -= old.points.nbytes + old.inverse.nbytes + sum(map(len, old_key))
                self.entries[key] = layout
                self.nbytes += size
        else:
            self.entries.move_to_end(key)
        stop = int(np.searchsorted(layout.points, limit, side='left'))
        points = (np.r_[layout.points[:stop], limit]
                  if stop < layout.points.size else layout.points)
        inverse = np.minimum(layout.inverse, stop)
        a = shape + np.arange(4)[:, None]
        scaled = rate*points[None]
        p, q = gammainc(a, scaled), gammaincc(a, scaled)
        left, right = inverse[:lower.size], inverse[lower.size:]
        return p[:, left], p[:, right], q[:, left], q[:, right]

    def clear(self) -> None:
        """Discard cached endpoint layouts between independent fits."""
        self.entries.clear()
        self.nbytes = 0


def gamma_log_density(lag: np.ndarray, shape: float, rate: float) -> np.ndarray:
    """Evaluate the density without subtracting large normalization terms."""
    mode = shape-1
    x = rate*lag
    if mode == 0:
        return np.asarray(np.log(rate)-x)
    if mode < 8:
        with np.errstate(divide='ignore'):
            return np.asarray(np.log(rate)-gammaln(shape) + mode*np.log(x)-x)
    inv = 1/mode
    coefficients = [1/12, -1/360, 1/1260, -1/1680, 1/1188, -691/360360,
                    1/156, -3617/122400, 43867/244188, -174611/125400]
    correction = sum(c*inv**(2*i+1) for i, c in enumerate(coefficients))
    delta = (x-mode)/mode
    small = abs(delta) < .25
    with np.errstate(divide='ignore'):
        log_minus = np.log1p(delta)-delta
    z = delta[small]
    log_minus[small] = sum((-1)**(n+1)*z**n/n for n in range(2, 33))
    return np.asarray(np.log(rate)-.5*np.log(2*np.pi*mode)-correction+mode*log_minus)


class DispersedAIF:
    """Prepare a checked continuous gamma convolution of a fixed measured AIF.

    Parameters
    ----------
    base : ContinuousCPIConvolution
        Original acquired-AIF reconstruction, including its finite support.
    mode, rate : float
        Gamma mode in seconds and positive rate in inverse seconds.
    horizon : float
        Covering positive elapsed time. Gamma transport is not cut off at the
        final acquisition frame.
    expansion_nodes : int, optional
        Polynomial order used by the checked interval representation.
    refinement_edges : ndarray, optional
        Previously refined interval boundaries. These are geometry only: all
        coefficients and independent accuracy checks are recomputed. The
        requested horizon and measured-AIF boundaries are always included.
    geometry_cache : IntegralGeometryCache, optional
        Shared bounded storage for exact residue-overlap geometry.
    endpoint_cache : GammaEndpointCache, optional
        Shared bounded storage for exact unscaled gamma endpoint orderings.
    """

    def __init__(self, base: ContinuousCPIConvolution, mode: float, rate: float,
                 horizon: float, *, expansion_nodes: int = 16,
                 refinement_edges: np.ndarray | None = None,
                 geometry_cache: IntegralGeometryCache | None = None,
                 endpoint_cache: GammaEndpointCache | None = None) -> None:
        if (not np.all(np.isfinite([mode, rate, horizon, 1+mode*rate]))
                or mode < 0 or rate <= 0 or horizon <= 0):
            raise ValueError('finite nonnegative mode, positive rate and horizon required')
        if expansion_nodes not in (8, 12, 16, 24, 32):
            raise ValueError('unsupported polynomial quadrature order')
        if refinement_edges is not None:
            refinement_edges = np.asarray(refinement_edges, dtype=float)
            if (refinement_edges.ndim != 1 or not np.all(np.isfinite(refinement_edges))
                    or np.any(refinement_edges < 0) or np.any(np.diff(refinement_edges) <= 0)):
                raise ValueError('finite increasing nonnegative refinement edges required')
            refinement_edges = refinement_edges.copy()
        self.refinement_edges = refinement_edges
        self.geometry_cache = geometry_cache
        self.endpoint_cache = endpoint_cache
        self.base, self.shape, self.rate = base, 1 + mode * rate, rate
        self.horizon = horizon
        self.expansion_nodes = expansion_nodes
        self.rules = {n: np.polynomial.legendre.leggauss(n) for n in (16, 32)}
        self.jacobi = {}
        for n in self.rules:
            x, w = roots_jacobi(n, 0, self.shape - 1)
            self.jacobi[n] = x, w / w.sum()
        if base.aif_coefficients is None:
            self.input_coefficients = np.vstack((np.zeros((2, len(base.left))),
                                                base.slope, base.amplitude))
        else:
            self.input_coefficients = base.aif_coefficients
        widths = base.right - base.left
        self.scale = float(np.max(np.sum(abs(self.input_coefficients)
                            * widths[None] ** np.arange(3, -1, -1)[:, None], axis=0), initial=0.))
        self.raw_tolerance = TRANSPORT_RAW_RELATIVE_ERROR * self.scale
        self.interpolation_tolerance = TRANSPORT_INTERPOLATION_RELATIVE_ERROR * self.scale
        self.tail_probability = 1e-18
        self.kernel_lower = gammaincinv(self.shape, self.tail_probability)/rate
        self.kernel_upper = gammainccinv(self.shape, self.tail_probability)/rate
        self.raw_points = 0
        self.analytic_intervals = 0
        self.quadrature_intervals = 0
        self._prepare()

    def _raw(self, target: np.ndarray) -> np.ndarray:
        """Integrate the piecewise cubic AIF against the gamma density."""
        base = self.base
        self.raw_points += len(target)
        layout = native.interval_layout(target, base.left, base.right,
                                         self.kernel_lower, self.kernel_upper)
        if layout is None:
            lo = np.maximum(0., target[:, None] - base.right[None])
            hi = np.minimum(self.kernel_upper, np.maximum(0., target[:, None] - base.left[None]))
            active = (hi > lo) & (hi > self.kernel_lower)
            rows, segments = np.nonzero(active)
            lo, hi = lo[rows, segments], hi[rows, segments]
            packed = np.packbits(active).tobytes()
        else:
            rows, segments, lo, hi, packed = layout
        layout_key = (target.dtype.str.encode(), target.tobytes(),
                      base.left.dtype.str.encode(), base.left.tobytes(),
                      base.right.dtype.str.encode(), base.right.tobytes(),
                      np.asarray((len(target), len(base.left)), dtype=np.intp).tobytes(), packed)
        value, roundoff = self._moment_blocks(target, rows, segments, lo, hi, layout_key)
        counts = np.bincount(rows, minlength=len(target))
        budget = self.raw_tolerance - 2*self.tail_probability*self.scale
        tolerance = budget / np.maximum(1, counts[rows])
        direct = np.isfinite(value) & (roundoff <= tolerance)
        self.analytic_intervals += int(np.count_nonzero(direct))
        pending = ~direct
        self.quadrature_intervals += int(np.count_nonzero(pending))
        result = np.bincount(rows[direct], weights=value[direct], minlength=len(target))
        if np.any(pending):
            remaining = budget - np.bincount(
                rows[direct], weights=roundoff[direct], minlength=len(target))
            result += self._quadrature(target, rows[pending], segments[pending],
                                       lo[pending], hi[pending], remaining)
        return np.asarray(result)

    def _moment_blocks(
        self, target: np.ndarray, rows: np.ndarray, segments: np.ndarray,
        lo: np.ndarray, hi: np.ndarray, layout_key: tuple[bytes, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound moment temporaries without changing interval values or reductions."""
        value, roundoff = np.empty(len(rows)), np.empty(len(rows))
        factors = np.r_[1., np.cumprod((self.shape+np.arange(3))/self.rate)][:, None]
        for first in range(0, len(rows), _INTERVAL_BLOCK_SIZE):
            stop = min(first + _INTERVAL_BLOCK_SIZE, len(rows))
            block = slice(first, stop)
            t = target[rows[block]] - self.base.left[segments[block]]
            if self.endpoint_cache is None:
                p0, p1, q0, q1 = _gamma_interval_cdfs(
                    self.shape, self.rate*lo[block], self.rate*hi[block])
            else:
                key = (*layout_key, np.asarray([first, stop], dtype=np.intp).tobytes())
                p0, p1, q0, q1 = self.endpoint_cache.evaluate(
                    self.shape, self.rate, lo[block], t, self.kernel_upper, layout_key=key)
            contracted = native.contract_moments(t, segments[block], self.input_coefficients,
                                                  factors[:, 0], p0, p1, q0, q1)
            if contracted is not None:
                value[block], roundoff[block] = contracted
                continue
            c = self.input_coefficients[:, segments[block]]
            power, absolute_power = np.empty_like(c), np.empty_like(c)
            for k in range(4):
                pieces = np.array([c[3-j] * comb(j, k) * t**(j-k) * (-1)**k
                                   for j in range(k, 4)])
                power[k] = pieces.sum(axis=0)
                absolute_power[k] = abs(pieces).sum(axis=0)
            use_upper = p0 > .5
            mass = np.where(use_upper, q0-q1, p1-p0)
            masses_sum = np.where(use_upper, q0+q1, p1+p0)
            terms = power * factors * mass
            value[block] = np.sum(terms, axis=0)
            roundoff[block] = 16 * np.finfo(float).eps * np.sum(
                absolute_power * factors * masses_sum + abs(terms), axis=0)
        return value, roundoff

    def _quadrature(self, target: np.ndarray, rows: np.ndarray, segments: np.ndarray,
                    lo: np.ndarray, hi: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
        """Resolve unsafe moment cancellation with independent interval refinement."""
        def integral(order: int, first: int) -> np.ndarray:
            result = np.empty(len(rows)-first)
            for start in range(first, len(rows), _INTERVAL_BLOCK_SIZE):
                stop = min(start + _INTERVAL_BLOCK_SIZE, len(rows))
                block = slice(start, stop)
                result[start-first:stop-first] = self._quadrature_values(
                    order, target, rows[block], segments[block], lo[block], hi[block])
            return result

        first = 0
        coarse, fine = np.empty(len(rows)), np.empty(len(rows))
        for _ in range(30):
            coarse[first:], fine[first:] = integral(16, first), integral(32, first)
            error = 4*abs(fine-coarse)
            totals = np.bincount(rows, weights=fine, minlength=len(target))
            errors = np.bincount(rows, weights=error, minlength=len(target))
            if np.all(np.isfinite(totals)) and np.all(errors <= tolerance):
                return np.asarray(totals)
            counts = np.bincount(rows, minlength=len(target))
            split = (errors[rows] > tolerance[rows]) & (error > tolerance[rows]/counts[rows])
            if not np.any(split):
                break
            keep = ~split
            first = int(np.count_nonzero(keep))
            coarse, fine = (np.r_[arr[keep], np.empty(2*np.count_nonzero(split))]
                            for arr in (coarse, fine))
            middle = (lo[split]+hi[split])/2
            lo, hi = np.r_[lo[keep], lo[split], middle], np.r_[hi[keep], middle, hi[split]]
            rows, segments = (np.r_[arr[keep], arr[split], arr[split]] for arr in (rows, segments))
            if len(rows) > 2000000:
                break
        raise DispersionIntegrationError('dispersed AIF quadrature did not meet accuracy')

    def _quadrature_values(
        self, order: int, target: np.ndarray, rows: np.ndarray, segments: np.ndarray,
        left: np.ndarray, right: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one bounded block; global error checks remain in the caller."""
        x, w = self.rules[order]
        width = right-left
        lag = left[:, None] + width[:, None] * (x+1)/2
        logw = np.log(width[:, None]*w/2) + gamma_log_density(lag, self.shape, self.rate)
        origin = left <= right/4
        correction = np.zeros(left.size)
        if np.any(origin):
            jx, jw = self.jacobi[order]
            lag[origin] = right[origin, None]*(jx+1)/2
            logw[origin] = (np.log(jw) + self.shape*np.log(self.rate*right[origin, None])
                           - gammaln(self.shape+1) - self.rate*lag[origin])
            # Subtract two origin integrals only where the power weight is needed.
            shifted = origin & (left > 0)
            if np.any(shifted):
                low_lag = left[shifted, None]*(jx+1)/2
                low_logw = (np.log(jw) + self.shape*np.log(self.rate*left[shifted, None])
                            - gammaln(self.shape+1) - self.rate*low_lag)
                low_local = (target[rows[shifted], None] - low_lag
                             - self.base.left[segments[shifted], None])
                low_c = self.input_coefficients[:, segments[shifted], None]
                low_signal = ((low_c[0]*low_local+low_c[1])*low_local+low_c[2])*low_local+low_c[3]
                correction[shifted] = np.sum(low_signal*np.exp(low_logw), axis=1)
        local = target[rows, None] - lag - self.base.left[segments, None]
        c = self.input_coefficients[:, segments, None]
        signal = ((c[0]*local+c[1])*local+c[2])*local+c[3]
        return np.asarray(np.sum(signal*np.exp(logw), axis=1)-correction)

    def _prepare(self) -> None:
        """Refine local Chebyshev segments using independent off-grid checks."""
        edges = np.unique(np.r_[0., self.base.left, self.base.right, self.horizon])
        if self.refinement_edges is not None:
            proposed = np.union1d(edges, self.refinement_edges)
            # A geometry hint must not consume the independent refinement budget.
            if np.count_nonzero((proposed >= 0) & (proposed <= self.horizon)) <= 20001:
                edges = proposed
        edges = edges[(edges >= 0) & (edges <= self.horizon)]
        lo, hi = edges[:-1], edges[1:]
        n = self.expansion_nodes
        nodes = np.cos(np.pi*(np.arange(n)+.5)/n)
        check = np.cos(np.pi*(np.arange(n+1)+.5)/(n+1))
        transform = np.polynomial.chebyshev.chebvander(nodes, n-1).T/(n/2)
        transform[0] /= 2
        kept: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(36):
            mid, half = (lo+hi)/2, (hi-lo)/2
            points = (mid[:, None]+half[:, None]*nodes).ravel()
            samples = self._raw(points).reshape(-1, n)
            coefficients = samples @ transform.T
            expected = self._raw((mid[:, None]+half[:, None]*check).ravel()).reshape(-1, n+1)
            predicted = coefficients @ np.polynomial.chebyshev.chebvander(check, n-1).T
            error = 4*np.max(abs(expected-predicted), axis=1)
            accepted = error <= self.interpolation_tolerance
            kept.append((lo[accepted], hi[accepted], coefficients[accepted], error[accepted]))
            if np.all(accepted):
                break
            todo = ~accepted
            lo, hi = np.r_[lo[todo], mid[todo]], np.r_[mid[todo], hi[todo]]
            if sum(len(part[0]) for part in kept)+len(lo) > 20000:
                raise DispersionIntegrationError('dispersed AIF interpolation exceeds segment budget')
        else:
            raise DispersionIntegrationError(
                f'dispersed AIF interpolation did not meet accuracy: shape={self.shape}, '
                f'rate={self.rate}, horizon={self.horizon}, '
                f'error_ratio={float(np.max(error))/self.interpolation_tolerance}, '
                f'intervals={list(zip(lo.tolist(), hi.tolist()))[:8]}')
        self.left, self.right, self.coefficients, self.error = (
            np.concatenate([part[index] for part in kept]) for index in range(4))
        order = np.argsort(self.left)
        self.left, self.right = self.left[order], self.right[order]
        self.coefficients, self.error = self.coefficients[order], self.error[order]
        self.nodes, self.weights = np.polynomial.legendre.leggauss((n+4)//2)
        self.integral = PolynomialIntegral(self.left, self.right, self.coefficients.T,
                                          self.nodes, self.weights, chebyshev=True,
                                          geometry_cache=self.geometry_cache)

    def convolve(self, knots: np.ndarray, values: np.ndarray,
                 target: np.ndarray) -> np.ndarray:
        """Integrate the prepared AIF and cubic residue on exact overlap segments."""
        if np.max(target) > self.horizon:
            raise ValueError('dispersed AIF does not cover elapsed time')
        coefficients = np.column_stack((PchipInterpolator(knots, values).c,
                                        [0., 0., 0., values[-1]]))
        return np.asarray(self.integral.evaluate(knots[None], coefficients[None], target[None])[0])


class AssociatedConvolution(EnsembleConvolution):
    """Reuse dispersed AIFs only across exactly matching transport parameters."""

    def __init__(self, base: ContinuousCPIConvolution, batch_size: int = 8) -> None:
        super().__init__(base, batch_size)
        self.cache: OrderedDict[tuple[float, float], DispersedAIF] = OrderedDict()
        self.failed: OrderedDict[tuple[float, float], bool] = OrderedDict()
        self.reference = IntervalReuseConvolution(base, batch_size)
        self.integral_geometry = IntegralGeometryCache()
        self.endpoint_layouts = GammaEndpointCache()
        self.stats: dict[str, Any] = {}
        self.fallback_rows = 0

    def clear_transient_state(self) -> None:
        """Discard all fitted-row history while retaining immutable AIF data."""
        super().clear_transient_state()
        self.cache.clear()
        self.failed.clear()
        self.reference.clear_transient_state()
        self.integral_geometry.clear()
        self.endpoint_layouts.clear()
        self.stats.clear()
        self.fallback_rows = 0

    def _refinement_edges(self, key: tuple[float, float]) -> np.ndarray | None:
        """Reuse the smallest nearby stencil partition, never its fitted values."""
        candidates = []
        for pair, prepared in self.cache.items():
            changed = [index for index in range(2) if pair[index] != key[index]]
            if len(changed) != 1:
                continue
            index = changed[0]
            if min(key[index], pair[index]) <= 0:
                continue
            previous, current = np.log(pair[index]), np.log(key[index])
            distance = abs(current-previous)
            # Match the existing derivative stencil's maximum four-step reach.
            reach = 4*np.finfo(float).eps**.2*max(1., abs(previous))
            if distance <= reach:
                candidates.append((len(prepared.left), distance, prepared))
        if not candidates:
            return None
        prepared = min(candidates, key=lambda item: item[:2])[2]
        return np.asarray(np.r_[prepared.left, prepared.right[-1]])

    def _gamma(self, knots: np.ndarray, values: np.ndarray, target: np.ndarray,
               gamma: np.ndarray) -> np.ndarray:
        """Preserve every row while sharing immutable dispersed input functions."""
        result = np.zeros_like(target)
        if not self.base.left.size:
            return result
        horizon = max(0., float(np.max(target)))
        if horizon == 0:
            return result
        fallback = []
        for row in range(len(knots)):
            key = (float(gamma[row, 0]), float(gamma[row, 1]))
            if key in self.failed:
                self.failed.move_to_end(key)
                fallback.append(row)
                continue
            prepared = self.cache.get(key)
            if prepared is None or prepared.horizon < horizon:
                edges = self._refinement_edges(key)
                try:
                    try:
                        prepared = DispersedAIF(self.base, *key, horizon*1.1,
                                               refinement_edges=edges,
                                               geometry_cache=self.integral_geometry,
                                               endpoint_cache=self.endpoint_layouts)
                    except DispersionIntegrationError:
                        if edges is None:
                            raise
                        self.stats['seed_retries'] = self.stats.get('seed_retries', 0)+1
                        prepared = DispersedAIF(self.base, *key, horizon*1.1,
                                               geometry_cache=self.integral_geometry,
                                               endpoint_cache=self.endpoint_layouts)
                except DispersionIntegrationError as error:
                    self.failed[key] = True
                    while len(self.failed) > 128:
                        self.failed.popitem(last=False)
                    self.stats['failed_preparations'] = self.stats.get('failed_preparations', 0)+1
                    examples = self.stats.setdefault('failure_examples', [])
                    if len(examples) < 8:
                        examples.append(dict(mode=key[0], rate=key[1], horizon=horizon*1.1,
                                             reason=str(error)))
                    fallback.append(row)
                    continue
                self.cache[key] = prepared
                self.stats['maps'] = self.stats.get('maps', 0)+1
                self.stats['seeded_maps'] = (self.stats.get('seeded_maps', 0)
                                            + int(prepared.refinement_edges is not None))
                self.stats['max_segments'] = max(self.stats.get('max_segments', 0), len(prepared.left))
                while len(self.cache) > 128:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            result[row] = prepared.convolve(knots[row], values[row], target[row])
        if fallback:
            self.fallback_rows += len(fallback)
            result[fallback] = self.reference._gamma(knots[fallback], values[fallback],
                                                     target[fallback], gamma[fallback])
        return result
