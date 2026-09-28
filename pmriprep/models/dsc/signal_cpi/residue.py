"""Monotone intrinsic CPI residue with an inferred initial slope."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly

from .batch import EnsembleConvolution, SignalBatchEvaluator
from .continuous import ContinuousCPIConvolution
from .geometry import Model
from .probability import GaussianPrior
from .signal import FixedLayoutSignalCPI, MovingLayoutSignalCPI
from .transport import AssociatedConvolution


def slope_ratio(knots: np.ndarray, values: np.ndarray) -> float:
    """Return the shape-preserving origin slope divided by its first secant."""
    secant = (values[1] - values[0]) / (knots[1] - knots[0])
    if secant == 0:
        return 1.0
    ratio = float(PchipInterpolator(knots, values).derivative()(0) / secant)
    if not -1e-12 <= ratio <= 3.0 + 1e-12:
        raise ValueError("PCHIP endpoint slope outside shape-preserving range")
    return float(np.clip(ratio, 0.0, 3.0))


def initial_slopes(knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Compute the PCHIP origin derivative for each row without building splines.

    Parameters
    ----------
    knots, values : ndarray
        Matching matrices of ordered knot times and residue amplitudes.

    Returns
    -------
    ndarray
        Shape-preserving one-sided derivatives, including zero/flat intervals.
    """
    widths = np.diff(knots[:, :3], axis=1)
    secants = np.diff(values[:, :3], axis=1) / widths
    if knots.shape[1] == 2:
        return np.asarray(secants[:, 0])
    h0, h1 = widths.T
    m0, m1 = secants.T
    derivative = ((2*h0 + h1)*m0 - h0*m1) / (h0+h1)
    wrong_sign = np.sign(derivative) != np.sign(m0)
    limit = (~wrong_sign) & (np.sign(m0) != np.sign(m1)) & (abs(derivative) > 3*abs(m0))
    derivative[wrong_sign] = 0.
    derivative[limit] = 3*m0[limit]
    return np.asarray(derivative)


def changed_residue(knots: np.ndarray, values: np.ndarray, alpha: float) -> PPoly:
    """Change only the origin derivative, preserving all knot values and joins.

    Parameters
    ----------
    knots, values : ndarray
        Ordered times from zero and positive residue amplitudes.
    alpha : float
        Origin derivative divided by the first secant, between zero and three.

    Returns
    -------
    PPoly
        First-segment Hermite modification of the original PCHIP polynomial.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 3:
        raise ValueError("slope ratio must lie in [0,3]")
    original = PchipInterpolator(knots, values)
    width = knots[1] - knots[0]
    secant = (values[1] - values[0]) / width
    delta = alpha * secant - original.c[2, 0]
    coefficients = original.c.copy()
    coefficients[:, 0] += [delta / width**2, -2 * delta / width, delta, 0.0]
    return PPoly(coefficients, knots)


def slope_basis_integral(operator: ContinuousCPIConvolution, widths: np.ndarray,
                         delays: np.ndarray) -> np.ndarray:
    """Batch exact convolution with t*(1-t/h)^2 on the first segment only."""
    if (widths.ndim != 1 or delays.shape != widths.shape
            or not np.all(np.isfinite(widths)) or np.any(widths <= 0)
            or not np.all(np.isfinite(delays))):
        raise ValueError("positive finite widths and finite delays required")
    result = np.zeros((len(widths), operator.time.size))
    for first in range(0, len(widths), 8):
        block = slice(first, first + 8)
        target = operator.time[None] - delays[block, None]
        lower = np.maximum(target[:, :, None] - operator.right[None, None], 0)
        upper = np.minimum(target[:, :, None] - operator.left[None, None],
                           widths[block, None, None])
        rows, frames, segments = np.nonzero(upper > lower)
        lo, hi = lower[rows, frames, segments], upper[rows, frames, segments]
        half = (hi - lo) / 2
        t = (hi + lo)[:, None] / 2 + half[:, None] * operator.nodes
        basis = t * (1 - t / widths[first + rows, None]) ** 2
        local = target[rows, frames, None] - t - operator.left[segments, None]
        if operator.aif_coefficients is None:
            aif = operator.amplitude[segments, None] + operator.slope[segments, None] * local
        else:
            c = operator.aif_coefficients[:, segments, None]
            aif = ((c[0] * local + c[1]) * local + c[2]) * local + c[3]
        integral = half * ((aif * basis) @ operator.weights)
        result[block] = np.bincount(rows * target.shape[1] + frames,
            weights=integral, minlength=target.size).reshape(target.shape)
    return result


def basis_knots(widths: np.ndarray, endpoint: float) -> np.ndarray:
    """Build a compact zero-tail PCHIP basis with its second zero past h."""
    if widths.ndim != 1 or np.any(widths <= 0) or not np.all(np.isfinite(widths)):
        raise ValueError("positive finite widths required")
    if endpoint not in (2., 3.):
        raise ValueError("basis endpoint must be two or three widths")
    return np.asarray(widths[:, None] * np.array([0., 1., endpoint]))


def dispersed_basis(convolution: EnsembleConvolution, widths: np.ndarray,
                    delays: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Disperse the exact cubic basis without changing the gamma quadrature.

    Parameters
    ----------
    convolution : EnsembleConvolution
        Existing checked acquired-AIF gamma integrator.
    widths, delays : ndarray
        Positive first-segment lengths and signed arrival delays.
    gamma : ndarray
        Gamma mode/rate pairs, one per parameter row.

    Returns
    -------
    ndarray
        AIF convolved with gamma transport and t*(1-t/h)^2 on [0,h].
    """
    n = len(widths)
    if delays.shape != widths.shape or gamma.shape != (n, 2):
        raise ValueError("matching widths, delays and gamma rows required")
    knots = np.concatenate((basis_knots(widths, 3.), basis_knots(widths, 2.)))
    values = np.tile([1., 0., 0.], (2*n, 1))
    result = convolution.evaluate(knots, values, np.ones(2*n),
                                  np.tile(delays, 2), np.tile(gamma, (2, 1)))
    return np.asarray(6 * widths[:, None] * (result[:n] - result[n:]))


class PhysicalSignalCPI(MovingLayoutSignalCPI):
    """Moving monotone residue with a bounded first-segment Hermite slope.

    Parameters
    ----------
    observed, aif, time : ndarray
        Measured signal, acquired delta-R2-star AIF and relative seconds.
    dt, te : float
        Sampling interval and echo time, in seconds.
    n_control_points : int
        Number of ordered knots, including zero and the acquisition endpoint.
    model : {"cpi", "cpi_vtf"}
        Intrinsic residue alone or with gamma arterial transport.

    Notes
    -----
    The slope multiplier has a N(1,1) prior and support [0,3]. It multiplies
    the nonpositive first secant; every amplitude ratio is at most one.
    The endpoint amplitude is inferred, not forced to zero. Gamma transport
    can yield a rising effective response without a rising intrinsic residue.
    """

    def __init__(
        self, observed: np.ndarray, aif: np.ndarray, time: np.ndarray,
        dt: float, te: float, n_control_points: int, model: Model,
    ) -> None:
        super().__init__(observed, aif, time, dt, te, n_control_points, model,
                         integration="continuous", aif_interpolation="pchip")
        original_starts = super().starts()
        self.slope_index = self.prior.mean.size
        covariance = np.zeros((self.slope_index + 1, self.slope_index + 1))
        covariance[:-1, :-1] = self.prior.covariance
        covariance[-1, -1] = 1.0
        self.prior = GaussianPrior(np.r_[self.prior.mean, 1.0], covariance)
        self.lower = np.r_[self.lower, 0.0]
        self.upper = np.r_[self.upper, 3.0]
        self.upper[2:self.delay_index] = 0.0
        self._starts = tuple(
            np.r_[m, slope_ratio(self.knot_times(m), self.knot_values(m))]
            for m in original_starts
        )
        if not all(self.valid_mean(m) for m in self._starts):
            raise ValueError("data-derived start violates intrinsic residue support")

    def starts(self) -> tuple[np.ndarray, ...]:
        """Return owned starts reproducing the data-derived PCHIP curves."""
        return tuple(m.copy() for m in self._starts)

    def slope_change(self, mean: np.ndarray) -> tuple[float, float]:
        """Return first-segment width and derivative change in inverse seconds."""
        knots, values = self.knot_times(mean), self.knot_values(mean)
        secant = (values[1] - values[0]) / knots[1]
        old_slope = PchipInterpolator(knots, values).c[2, 0]
        return float(knots[1]), float(mean[self.slope_index] * secant - old_slope)

    def residue(self, mean: np.ndarray) -> PPoly:
        """Return the actual intrinsic polynomial used for fitting and readout."""
        return changed_residue(self.knot_times(mean), self.knot_values(mean),
                               float(mean[self.slope_index]))

    def concentration(self, mean: np.ndarray) -> np.ndarray:
        """Evaluate concentration with the same exact first-segment correction."""
        original = mean[:self.slope_index]
        baseline = super().concentration(original)
        width, delta = self.slope_change(mean)
        if self.dispersion is not None:
            mode, rate = np.exp(original[-2:])
            terms = [
                self.dispersion.evaluate(
                    basis_knots(np.array([width]), end)[0], np.array([1., 0., 0.]),
                    1., float(mean[self.delay_index]), mode, rate,
                )
                for end in (3., 2.)
            ]
            correction = 6 * width * (terms[0] - terms[1])
        else:
            assert self.continuous is not None
            correction = slope_basis_integral(
                self.continuous, np.array([width]), np.array([mean[self.delay_index]])
            )[0]
        return np.asarray(baseline + mean[0] * delta * correction)


class PhysicalSignalEvaluator(SignalBatchEvaluator):
    """Evaluate physical CPI, reusing only exactly identical unit responses.

    Parameters
    ----------
    model : PhysicalSignalCPI
        One acquired input and its intrinsic residue/transport coordinates.
    cache_size : int, optional
        Maximum cached curves. Zero disables persistent curve reuse.

    Notes
    -----
    Flow, baseline and slope remain distinct factors. A checked associated
    gamma evaluation reuses arterial transport without approximating nearby
    parameters. Integration failures use checked quadrature, not another model.
    """

    def __init__(self, model: PhysicalSignalCPI, cache_size: int = 512) -> None:
        super().__init__(model)
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache size must be a nonnegative integer")
        self.cache_size = cache_size
        self.cache: OrderedDict[tuple[str, bytes], np.ndarray] = OrderedDict()
        if model.model == "cpi_vtf":
            self.convolution = AssociatedConvolution(self.convolution.base)

    def clear_transient_state(self) -> None:
        """Discard exact response and convolution caches between fitted rows."""
        self.cache.clear()
        self.convolution.clear_transient_state()

    def _reuse(
        self, role: str, keys: np.ndarray,
        evaluate: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Scatter exact unique responses with bounded persistent cache storage."""
        unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        identities = [(role, row.tobytes()) for row in unique]
        curves = np.empty((len(unique), len(self.model.time)))
        missing = []
        for index, key in enumerate(identities):
            if key in self.cache:
                curves[index] = self.cache[key]
                self.cache.move_to_end(key)
            else:
                missing.append(index)
        if missing:
            indices = np.asarray(missing, dtype=int)
            values = evaluate(first[indices])
            curves[indices] = values
            if self.cache_size:
                for index, value in zip(indices, values, strict=True):
                    stored = value.copy()
                    stored.flags.writeable = False
                    self.cache[identities[index]] = stored
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
        return np.asarray(curves[inverse])

    def _transport_coordinates(self) -> tuple[int, int] | None:
        """Locate gamma coordinates before the physical initial-slope parameter."""
        model = cast(PhysicalSignalCPI, self.model)
        return ((model.slope_index - 2, model.slope_index - 1)
                if model.model == "cpi_vtf" else None)

    def concentration(self, means: np.ndarray) -> np.ndarray:
        """Keep the baseline-plus-slope arithmetic and explicit gamma indices."""
        model = cast(PhysicalSignalCPI, self.model)
        if (means.ndim != 2 or means.shape[1:] != model.prior.mean.shape
                or not np.all(np.isfinite(means))):
            raise ValueError("finite matching parameter matrix required")
        if not len(means):
            return np.empty((0, len(model.time)))
        delays = means[:, model.delay_index]
        knots = np.array([model.knot_times(m) for m in means])
        values = np.array([model.knot_values(m) for m in means])
        secants = (values[:, 1] - values[:, 0]) / knots[:, 1]
        geometry = np.column_stack((knots[:, 1],
            means[:, model.slope_index] * secants - initial_slopes(knots, values)))
        if model.model == "cpi":
            assert model.continuous is not None
            gamma = None
        else:
            gamma = np.exp(means[:, model.slope_index-2:model.slope_index])

        def baseline_at(indices: np.ndarray) -> np.ndarray:
            return self.convolution.evaluate(
                knots[indices], values[indices],
                np.ones(len(indices)), delays[indices],
                None if gamma is None else gamma[indices])

        def basis_at(indices: np.ndarray) -> np.ndarray:
            if gamma is None:
                assert model.continuous is not None
                return slope_basis_integral(model.continuous, geometry[indices, 0], delays[indices])
            return dispersed_basis(self.convolution, geometry[indices, 0],
                                   delays[indices], gamma[indices])

        baseline = self._reuse("baseline", means[:, 2:model.slope_index], baseline_at)
        basis_keys = (np.column_stack((geometry[:, 0], delays)) if gamma is None
                      else np.column_stack((geometry[:, 0], delays, gamma)))
        basis = self._reuse("basis", basis_keys, basis_at)
        return np.asarray(means[:, 0, None] * baseline
                          + means[:, 0, None] * geometry[:, 1, None] * basis)


def signal_evaluator(model: FixedLayoutSignalCPI) -> SignalBatchEvaluator:
    """Use the evaluator matching the model's complete intrinsic polynomial."""
    from .conditioned import FixedDelayEvaluator, FixedDelaySignalCPI

    if isinstance(model, FixedDelaySignalCPI):
        return FixedDelayEvaluator(model)
    return (PhysicalSignalEvaluator(model) if isinstance(model, PhysicalSignalCPI)
            else SignalBatchEvaluator(model))
