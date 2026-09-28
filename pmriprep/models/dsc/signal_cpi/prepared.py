"""Prepared polynomial evaluation of continuous CPI convolution."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .continuous import ContinuousCPIConvolution
from .probability import Array


class PreparedCPIConvolution(ContinuousCPIConvolution):
    """Represent the exact AIF/residue convolution on its polynomial segments.

    Notes
    -----
    Two piecewise cubic inputs convolve to a piecewise polynomial of degree at
    most seven. Breakpoints are the sums of their input breakpoints. Eight
    Chebyshev nodes therefore determine each segment without a temporal grid
    approximation. Values at these nodes use the original overlap integral.
    Cache validity requires identical residue knots/amplitudes and a covering
    horizon; a changed parameter cannot reuse a stale response.
    """

    def __init__(
        self,
        aif: Array,
        time: Array,
        *,
        aif_interpolation: Literal["linear", "pchip"] = "pchip",
    ) -> None:
        super().__init__(aif, time, aif_interpolation=aif_interpolation)
        self._initialize_cache()

    @classmethod
    def from_operator(cls, base: ContinuousCPIConvolution) -> PreparedCPIConvolution:
        """Share immutable AIF coefficients with a fresh residue-response cache."""
        prepared = cls.__new__(cls)
        prepared.__dict__.update(base.__dict__)
        prepared._initialize_cache()
        return prepared

    def _initialize_cache(self) -> None:
        """Initialize residue-dependent state independently of the AIF."""
        self._knots: Array | None = None
        self._values: Array | None = None
        self._breaks = np.array([0.0])
        self._coefficients = np.empty((8, 0))
        self._nodes = np.cos(np.pi * (np.arange(8) + 0.5) / 8)
        self._transform = np.polynomial.chebyshev.chebvander(self._nodes, 7).T / 4
        self._transform[0] /= 2

    def _prepare(self, knots: Array, values: Array, horizon: float) -> None:
        """Build interval polynomials using exact overlap integration."""
        aif_breaks = np.unique(np.r_[self.left, self.right])
        sums = (aif_breaks[:, None] + knots[None, :]).ravel()
        self._breaks = np.unique(
            np.r_[0.0, sums[(sums > 0) & (sums < horizon)], horizon]
        )
        lo, hi = self._breaks[:-1], self._breaks[1:]
        nodes = (lo + hi)[:, None] / 2 + (hi - lo)[:, None] / 2 * self._nodes
        target = nodes.ravel()
        samples = np.empty_like(target)
        for first in range(0, target.size, 256):
            block = slice(first, first+256)
            samples[block] = super().evaluate_at(knots, values, 1.0, target[block])
        samples = samples.reshape(-1, 8)
        self._coefficients = self._transform @ samples.T
        self._knots, self._values = knots.copy(), values.copy()

    def evaluate_at(
        self, knots: Array, values: Array, flow: float, target: Array
    ) -> Array:
        """Evaluate the prepared exact segments at arbitrary requested times."""
        if (
            knots[0] != 0
            or not np.isfinite(flow)
            or target.ndim != 1
            or not np.all(np.isfinite(target))
        ):
            raise ValueError(
                "residue origin, finite flow and finite target vector required"
            )
        if not target.size or np.max(target) <= 0 or not self.left.size:
            return np.zeros_like(target)
        horizon = float(np.max(target))
        if (
            self._knots is None
            or self._values is None
            or not np.array_equal(knots, self._knots)
            or not np.array_equal(values, self._values)
            or horizon > self._breaks[-1]
        ):
            self._prepare(knots, values, horizon)
        result = np.zeros_like(target)
        active = target > 0
        times = target[active]
        indices = np.minimum(
            np.searchsorted(self._breaks, times, side="right") - 1,
            self._breaks.size - 2,
        )
        lo, hi = self._breaks[indices], self._breaks[indices + 1]
        local = 2 * (times - lo) / (hi - lo) - 1
        result[active] = flow * np.polynomial.chebyshev.chebval(
            local, self._coefficients[:, indices], tensor=False
        )
        return result
