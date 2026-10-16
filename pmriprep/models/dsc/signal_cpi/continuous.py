"""Continuous causal convolution of measured AIF and cubic CPI residue."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from .probability import Array
from . import native


def polynomial_overlaps(
    left: Array, right: Array, knots: Array, target: Array,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], Array, Array]:
    """Enumerate nonempty AIF/residue overlaps in row/frame/AIF/residue order.

    Parameters
    ----------
    left, right : ndarray
        Ordered arterial polynomial interval endpoints, with possible gaps.
    knots : ndarray
        Ordered residue knots, one row per parameter vector, starting at zero.
        The last residue interval extends to infinity.
    target : ndarray
        Elapsed output times for each parameter row, with either delay sign.

    Returns
    -------
    rows, frames, arterial, residue, lower, upper : ndarray
        Sparse interval indices and integration limits. No dense array with a
        residue-segment axis is allocated. Touching intervals have zero area.
    """
    lower = np.maximum(target[:, :, None] - right[None, None], 0.)
    upper = target[:, :, None] - left[None, None]
    rows, frames, arterial = np.nonzero(upper > lower)
    lo, hi = lower[rows, frames, arterial], upper[rows, frames, arterial]
    first = np.empty(rows.size, dtype=int)
    last = np.empty(rows.size, dtype=int)
    edges = np.searchsorted(rows, np.arange(len(knots) + 1))
    for row, times in enumerate(knots):
        block = slice(edges[row], edges[row + 1])
        first[block] = np.searchsorted(times, lo[block], side="right") - 1
        last[block] = np.searchsorted(times, hi[block], side="left") - 1
    counts = last - first + 1
    repeated = np.repeat(np.arange(rows.size), counts)
    residue = (first[repeated] + np.arange(repeated.size)
               - np.repeat(np.cumsum(counts) - counts, counts))
    rows, frames, arterial = rows[repeated], frames[repeated], arterial[repeated]
    ends = np.column_stack((knots[:, 1:], np.full(len(knots), np.inf)))
    return (rows, frames, arterial, residue,
            np.maximum(lo[repeated], knots[rows, residue]),
            np.minimum(hi[repeated], ends[rows, residue]))


@dataclass(frozen=True)
class IntegralGeometry:
    """Owned immutable overlap data for a fixed arterial polynomial and layout."""

    rows: NDArray[np.intp]
    frames: NDArray[np.intp]
    residue: NDArray[np.intp]
    full: NDArray[np.bool_]
    local: Array
    half: Array
    moments: Array
    partial_half: Array
    partial_local: Array
    partial_aif: Array

    @property
    def nbytes(self) -> int:
        """Return the owned array payload size."""
        return sum(getattr(self, field.name).nbytes for field in fields(self))


GeometryKey = tuple[object, tuple[int, ...], str, bytes, tuple[int, ...], str, bytes]


class IntegralGeometryCache:
    """Share an eight-MB, sixteen-entry cache across arterial representations.

    Notes
    -----
    Keys retain a unique token, not the arterial object or a recyclable object
    ID. Evicted arterial models cannot collide with new models or be kept alive
    by their cached geometry. The bound covers keys and array payloads, not RSS.
    """

    def __init__(self) -> None:
        self.entries: OrderedDict[GeometryKey, IntegralGeometry] = OrderedDict()
        self.nbytes = 0

    def get(self, key: GeometryKey) -> IntegralGeometry | None:
        """Retrieve an exact layout and mark it recently used."""
        found = self.entries.get(key)
        if found is not None:
            self.entries.move_to_end(key)
        return found

    def put(self, key: GeometryKey, geometry: IntegralGeometry) -> None:
        """Keep only layouts that fit within the aggregate payload budget."""
        size = geometry.nbytes + len(key[3]) + len(key[6])
        existing = self.entries.pop(key, None)
        if existing is not None:
            self.nbytes -= existing.nbytes + len(key[3]) + len(key[6])
        if size > 8_000_000:
            return
        while self.entries and (len(self.entries) >= 16 or self.nbytes + size > 8_000_000):
            old_key, old = self.entries.popitem(last=False)
            self.nbytes -= old.nbytes + len(old_key[3]) + len(old_key[6])
        self.entries[key] = geometry
        self.nbytes += size

    def clear(self) -> None:
        """Discard cached geometry while retaining the reusable cache object."""
        self.entries.clear()
        self.nbytes = 0


class PolynomialIntegral:
    """Reuse exact arterial moments on whole segments of a cubic convolution.

    Parameters
    ----------
    left, right : ndarray
        Ordered arterial interval endpoints.
    coefficients : ndarray
        Coefficient by interval matrix, descending powers or ascending Chebyshev.
    nodes, weights : ndarray
        Gaussian rule exact for the arterial polynomial times a cubic residue.
    chebyshev : bool, optional
        Use normalized Chebyshev coordinates instead of powers from the left edge.
    geometry_cache : IntegralGeometryCache, optional
        Shared bounded cache for exactly equal knot/time layouts. No caching
        is performed when omitted.

    Notes
    -----
    A whole arterial interval integrates a cubic residue from four centered
    moments. Intervals split by a residue knot retain the original Gaussian
    evaluation. This changes neither the quadrature order nor temporal sampling.
    """

    def __init__(self, left: Array, right: Array, coefficients: Array,
                 nodes: Array, weights: Array, *, chebyshev: bool = False,
                 geometry_cache: IntegralGeometryCache | None = None) -> None:
        self.left, self.right, self.coefficients = left, right, coefficients
        self.nodes, self.weights, self.chebyshev = nodes, weights, chebyshev
        self.geometry_cache = geometry_cache
        self._geometry_token = object()
        self.half = (right-left)/2
        local = self.half[:, None]*(1-nodes)
        aif = self._values(np.arange(left.size), local)
        self.moments = np.stack([
            self.half*np.sum(aif*(nodes**j*weights)[None], axis=1) for j in range(4)
        ])

    def _values(self, intervals: NDArray[np.intp], local: Array) -> Array:
        """Evaluate the arterial polynomial in its original coordinate system."""
        coefficients = self.coefficients[:, intervals, None]
        if self.chebyshev:
            return np.asarray(np.polynomial.chebyshev.chebval(
                local/self.half[intervals, None]-1, coefficients, tensor=False))
        result = np.zeros_like(local)
        for coefficient in coefficients:
            result = result*local+coefficient
        return np.asarray(result)

    def evaluate(self, knots: Array, coefficients: Array, target: Array) -> Array:
        """Convolve batches of cubic residues, including their constant tails."""
        result = native.convolve(knots, coefficients, target, self.left, self.right,
            self.coefficients, self.nodes, self.weights, chebyshev=self.chebyshev,
            moments=self.moments)
        if result is not None:
            return result
        return self._evaluate_python(knots, coefficients, target)

    def _evaluate_python(self, knots: Array, coefficients: Array, target: Array) -> Array:
        """Evaluate the independent NumPy moment and split-quadrature reference."""
        geometry = self._geometry(knots, target)
        rows, frames, ri, full = geometry.rows, geometry.frames, geometry.residue, geometry.full
        integral = np.empty(rows.size)
        local, h, m = geometry.local, geometry.half, geometry.moments
        c = coefficients[rows[full], :, ri[full]].T
        integral[full] = (
            (((c[0]*local+c[1])*local+c[2])*local+c[3])*m[0]
            + ((3*c[0]*local+2*c[1])*local+c[2])*h*m[1]
            + (3*c[0]*local+c[1])*h**2*m[2] + c[0]*h**3*m[3])
        partial = ~full
        c = coefficients[rows[partial], :, ri[partial]].T[:, :, None]
        local = geometry.partial_local
        residue = ((c[0]*local+c[1])*local+c[2])*local+c[3]
        integral[partial] = geometry.partial_half*((geometry.partial_aif*residue) @ self.weights)
        return np.asarray(np.bincount(rows*target.shape[1]+frames, weights=integral,
                                     minlength=target.size).reshape(target.shape))

    def _geometry(self, knots: Array, target: Array) -> IntegralGeometry:
        """Prepare the unchanged arithmetic geometry, without residue amplitudes."""
        key = (self._geometry_token, knots.shape, knots.dtype.str, knots.tobytes(),
               target.shape, target.dtype.str, target.tobytes())
        if self.geometry_cache is not None:
            cached = self.geometry_cache.get(key)
            if cached is not None:
                return cached
        rows, frames, ai, ri, lo, hi = polynomial_overlaps(self.left, self.right, knots, target)
        full = ((lo == target[rows, frames]-self.right[ai])
                & (hi == target[rows, frames]-self.left[ai]))
        selected = ai[full]
        local = (hi[full]+lo[full])/2-knots[rows[full], ri[full]]
        h = self.half[selected]
        m = self.moments[:, selected]
        partial = ~full
        half = (hi[partial]-lo[partial])/2
        points = (hi[partial]+lo[partial])[:, None]/2+half[:, None]*self.nodes
        partial_local = points-knots[rows[partial], ri[partial], None]
        alocal = target[rows[partial], frames[partial], None]-points-self.left[ai[partial], None]
        aif = self._values(ai[partial], alocal)
        arrays = (rows, frames, ri, full, local, h, m, half, partial_local, aif)
        for array in arrays:
            array.flags.writeable = False
        geometry = IntegralGeometry(*arrays)
        if self.geometry_cache is not None:
            self.geometry_cache.put(key, geometry)
        return geometry


class ContinuousCPIConvolution:
    """Integrate the acquired AIF and residue at arbitrary signed delays.

    Parameters
    ----------
    aif, time : ndarray
        Finite AIF samples and strictly increasing relative times starting at
        zero. The AIF is zero outside this window.
    aif_interpolation : {"linear", "pchip"}, optional
        Linear or shape-preserving cubic reconstruction between AIF samples.

    Notes
    -----
    Each overlap has a degree-four integrand for a linear AIF, or degree six
    for a cubic AIF. Three- or four-point Gaussian quadrature is exact for the
    corresponding polynomial.
    No tissue-curve resampling or temporal integration grid is introduced.
    The residue holds its final amplitude after the last knot. Integration
    starts at zero and has zero area at zero elapsed time, including when the
    first AIF sample is nonzero.
    """

    def __init__(
        self,
        aif: Array,
        time: Array,
        *,
        aif_interpolation: Literal["linear", "pchip"] = "linear",
    ) -> None:
        if (
            aif.ndim != 1
            or time.ndim != 1
            or aif.shape != time.shape
            or time.size < 2
            or not np.all(np.isfinite(aif))
            or not np.all(np.isfinite(time))
            or time[0] != 0
            or np.any(np.diff(time) <= 0)
        ):
            raise ValueError("finite AIF and increasing relative times required")
        if aif_interpolation not in ("linear", "pchip"):
            raise ValueError("unknown AIF interpolation")
        self.aif_interpolation = aif_interpolation
        self.time = time.copy()
        active = (aif[:-1] != 0) | (aif[1:] != 0)
        self.left = time[:-1][active]
        self.right = time[1:][active]
        self.amplitude = aif[:-1][active]
        self.slope = (np.diff(aif) / np.diff(time))[active]
        self.nodes = np.array([-np.sqrt(3.0 / 5), 0.0, np.sqrt(3.0 / 5)])
        self.weights = np.array([5.0 / 9, 8.0 / 9, 5.0 / 9])
        self.aif_coefficients: Array | None = None
        if aif_interpolation == "pchip":
            self.aif_coefficients = PchipInterpolator(time, aif).c[:, active]
            self.nodes, self.weights = np.polynomial.legendre.leggauss(4)

    def evaluate(self, knots: Array, values: Array, flow: float, delay: float) -> Array:
        """Integrate at the acquired times without clipping either delay sign.

        Parameters
        ----------
        knots, values : ndarray
            Ordered residue times starting at zero and their amplitudes.
        flow, delay : float
            Flow multiplier and signed tissue delay in seconds.

        Returns
        -------
        ndarray
            Tissue concentration in the units implied by AIF and flow.

        Raises
        ------
        ValueError
            If the residue does not start at zero or flow/delay is nonfinite.
            Invalid residue geometry also raises during interpolation.
        """
        if knots[0] != 0 or not np.isfinite(flow) or not np.isfinite(delay):
            raise ValueError("residue must start at zero; flow/delay must be finite")
        return self.evaluate_at(knots, values, flow, self.time - delay)

    def evaluate_at(
        self, knots: Array, values: Array, flow: float, target: Array
    ) -> Array:
        """Integrate at arbitrary elapsed times, including outside the acquisition.

        Parameters
        ----------
        knots, values : ndarray
            Residue times beginning at zero and their interpolated amplitudes.
        flow : float
            Finite flow multiplier.
        target : ndarray
            Finite elapsed times relative to tissue arrival. Nonpositive times
            have exactly zero area. The AIF remains zero outside its measured
            support and the residue holds its final value beyond its last knot.

        Returns
        -------
        ndarray
            Concentration at each supplied elapsed time, in the supplied order.

        Raises
        ------
        ValueError
            If the residue does not start at zero, flow is nonfinite, or target
            times are not a finite vector. Invalid residue geometry also raises.
        """
        if (
            knots[0] != 0
            or not np.isfinite(flow)
            or target.ndim != 1
            or not np.all(np.isfinite(target))
        ):
            raise ValueError(
                "residue origin, finite flow and finite target vector required"
            )
        residue = PchipInterpolator(knots, values)
        coefficients = np.column_stack((residue.c, [0.0, 0.0, 0.0, values[-1]]))
        residue_right = np.r_[knots[1:], np.inf]
        lower = np.maximum(
            target[:, None, None] - self.right[None, :, None], knots[None, None, :]
        )
        upper = np.minimum(
            target[:, None, None] - self.left[None, :, None],
            residue_right[None, None, :],
        )
        rows, aif_segments, residue_segments = np.nonzero(upper > lower)
        lo = lower[rows, aif_segments, residue_segments]
        hi = upper[rows, aif_segments, residue_segments]
        half = (hi - lo) / 2
        points = (hi + lo)[:, None] / 2 + half[:, None] * self.nodes
        local = points - knots[residue_segments, None]
        c = coefficients[:, residue_segments, None]
        r = ((c[0] * local + c[1]) * local + c[2]) * local + c[3]
        aif_local = target[rows, None] - points - self.left[aif_segments, None]
        if self.aif_coefficients is None:
            a = (
                self.amplitude[aif_segments, None]
                + self.slope[aif_segments, None] * aif_local
            )
        else:
            ac = self.aif_coefficients[:, aif_segments, None]
            a = ((ac[0] * aif_local + ac[1]) * aif_local + ac[2]) * aif_local + ac[3]
        integral = half * ((a * r) @ self.weights)
        return np.asarray(
            flow * np.bincount(rows, weights=integral, minlength=target.size)
        )
