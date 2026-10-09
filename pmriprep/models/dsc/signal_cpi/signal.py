"""Signal CPI forward models with joint flow, baseline, shape and delay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly

from .continuous import ContinuousCPIConvolution
from .derivatives import fourth_order_partial
from .dispersion import ContinuousDispersedCPI
from .geometry import Model, _gamma_vtf, _initial_coordinates, _spectral_seed, _times_from_coordinates, _volterra_apply
from .probability import Array, GaussianPrior


SIGNAL_BASELINE_PRIOR_MEAN = 100.0


@dataclass(frozen=True)
class LogSpacingDomain:
    """Free log intervals whose sum leaves room for a fixed final knot.

    Parameters
    ----------
    indices : slice
        Inferred coordinates containing the independent log intervals.
    available : float
        Maximum sum of free intervals, in seconds. The reserved final interval
        is excluded from this duration.

    Notes
    -----
    This is a convex constraint on the mean, not a normalized truncated
    posterior. Box bounds supply the individual minimum interval lengths.
    """

    indices: slice
    available: float

    def valid_mean(self, mean: Array) -> bool:
        """Require the total free duration to remain inside the endpoint."""
        return bool(np.sum(np.exp(mean[self.indices])) <= self.available)


class FixedLayoutSignalCPI:
    """Signal forward model with joint natural-unit flow/S0 and log-ratio inference.

    Parameters
    ----------
    observed, aif, time : ndarray
        Validated signal, delta-R2-star AIF, and relative times in seconds.
    dt, te : float
        Frame interval and echo time in seconds.
    n_control_points : int
        Fixed diagnostic order, including the origin and free-amplitude endpoint.
    model : {"cpi", "cpi_vtf"}
        Include a gamma dispersion kernel when requested.
    integration : {"native", "continuous"}, optional
        Native-grid comparator or continuous CPI/CPI+VTF convolution.
    aif_interpolation : {"linear", "pchip"}, optional
        Reconstruct the measured AIF between samples. Cubic reconstruction
        requires continuous integration and does not alter the measured values.

    Notes
    -----
    The final knot is at the last acquired time; its amplitude is inferred.
    Priors are Gaussian in the inferred coordinates, with no softmax-spacing
    penalty masquerading as a Gaussian prior. Knot-motion inference is absent
    in this diagnostic. Posterior support near a guard remains approximate.
    """

    def __init__(
        self,
        observed: Array,
        aif: Array,
        time: Array,
        dt: float,
        te: float,
        n_control_points: int,
        model: Model,
        *,
        integration: Literal["native", "continuous"] = "native",
        aif_interpolation: Literal["linear", "pchip"] = "linear",
    ) -> None:
        if integration not in ("native", "continuous"):
            raise ValueError("unknown CPI integration method")
        if aif_interpolation not in ("linear", "pchip"):
            raise ValueError("unknown AIF interpolation")
        if integration != "continuous" and aif_interpolation != "linear":
            raise ValueError("cubic AIF reconstruction requires continuous integration")
        self.integration = integration
        self.aif_interpolation = aif_interpolation
        self.continuous = (
            ContinuousCPIConvolution(aif, time, aif_interpolation=aif_interpolation)
            if integration == "continuous"
            else None
        )
        self.dispersion = (
            ContinuousDispersedCPI(aif, time, aif_interpolation=aif_interpolation)
            if integration == "continuous" and model == "cpi_vtf"
            else None
        )
        self.observed, self.time, self.dt, self.te = observed, time, dt, te
        self.n_points, self.model = n_control_points, model
        self.delay_index = n_control_points + 1
        self.duration = float(time[-1])
        coordinates = _initial_coordinates(n_control_points, self.duration, dt)
        self.knots = _times_from_coordinates(
            coordinates, n_control_points, self.duration, dt
        )
        self.grid = np.arange(2 * time.size - 1, dtype=float) * dt
        self.convolve = _volterra_apply(np.interp(self.grid, time, aif, right=0.0), dt)
        self.seed_flow, self.seed_mtt, self.seed_delay = _spectral_seed(
            observed, aif, time, dt, te
        )
        self.seed_baseline = float(np.max(observed))
        mean = np.r_[
            0.0,
            SIGNAL_BASELINE_PRIOR_MEAN,
            np.full(n_control_points - 1, np.log(0.5)),
            0.0,
        ]
        variance = np.r_[1e6, 1e6, np.full(n_control_points - 1, 0.67), 10.0]
        if model == "cpi_vtf":
            mean, variance = np.r_[mean, np.log([2.0, 2.0])], np.r_[variance, 4.0, 4.0]
        self.prior = GaussianPrior(mean, np.diag(variance))
        self.lower = np.r_[
            0.0, 0.0, np.full(n_control_points - 1, -24.0), -self.duration + dt
        ]
        self.upper = np.r_[
            np.inf,
            np.inf,
            np.full(n_control_points - 1, np.log(1.5)),
            self.duration - dt,
        ]
        if model == "cpi_vtf":
            self.lower = np.r_[self.lower, np.log([0.05, 0.05])]
            self.upper = np.r_[self.upper, np.log([30.0, 20.0])]
        self.lower[1] = np.nextafter(0.0, 1.0)
        self.lower[self.delay_index] = np.nextafter(
            self.lower[self.delay_index], np.inf
        )
        self.upper[self.delay_index] = np.nextafter(
            self.upper[self.delay_index], -np.inf
        )

    def valid_mean(self, mean: Array) -> bool:
        """Check numerical/physical mean support without clipping proposals."""
        if mean.shape != self.prior.mean.shape or not np.all(np.isfinite(mean)):
            return False
        return bool(np.all(mean >= self.lower) and np.all(mean <= self.upper))

    def knot_values(self, mean: Array) -> Array:
        """Convert inferred log amplitude ratios to residue control points."""
        return np.asarray(np.r_[1.0, np.exp(np.cumsum(mean[2 : self.delay_index]))])

    def knot_times(self, mean: Array) -> Array:
        """Return fixed diagnostic times, independent of the inferred mean."""
        return self.knots.copy()

    def residue(self, mean: Array) -> PPoly:
        """Return the intrinsic polynomial for consistent perfusion readout."""
        return PchipInterpolator(self.knot_times(mean), self.knot_values(mean))

    def arrival_delay(self, mean: Array) -> float:
        """Return inferred arrival in seconds for forward/readout consistency."""
        return float(mean[self.delay_index])

    def concentration(self, mean: Array) -> Array:
        """Predict vascular concentration using the verified complete operator."""
        if self.dispersion is not None:
            return self.dispersion.evaluate(
                self.knot_times(mean),
                self.knot_values(mean),
                float(mean[0]),
                float(mean[self.delay_index]),
                float(np.exp(mean[-2])),
                float(np.exp(mean[-1])),
            )
        if self.continuous is not None:
            return self.continuous.evaluate(
                self.knot_times(mean),
                self.knot_values(mean),
                float(mean[0]),
                float(mean[self.delay_index]),
            )
        interpolation = PchipInterpolator(self.knot_times(mean), self.knot_values(mean))
        residue = interpolation(np.minimum(self.grid, self.duration))
        undelayed = mean[0] * self.convolve(residue)
        if self.model == "cpi_vtf":
            kernel = _gamma_vtf(
                self.grid, self.dt, float(np.exp(mean[-2])), float(np.exp(mean[-1]))
            )
            undelayed = np.convolve(undelayed, kernel)[: self.grid.size] * self.dt
        return np.asarray(
            np.interp(
                self.time - mean[self.delay_index], self.grid, undelayed, left=0.0
            )
        )

    def predict(self, mean: Array) -> Array:
        """Evaluate measured signal with S0 as an inferred parameter."""
        return np.asarray(mean[1] * np.exp(-self.te * self.concentration(mean)))

    def signal_jacobian(self, mean: Array) -> Array:
        """Differentiate continuous signal without differencing its baseline.

        Flow and baseline columns are analytic. Shape/timing columns use a
        fourth-order concentration derivative followed by the exact signal
        chain rule. This derivative is restricted to the continuous operator;
        the native-grid interpolation has a different differentiability contract.
        """
        if self.continuous is None:
            raise ValueError("signal Jacobian requires continuous integration")
        unit_mean = mean.copy()
        unit_mean[0] = 1.0
        unit_concentration = self.concentration(unit_mean)
        attenuation = np.exp(-self.te * mean[0] * unit_concentration)
        prediction = mean[1] * attenuation
        result = np.empty((self.observed.size, mean.size))
        result[:, 0] = -self.te * unit_concentration * prediction
        result[:, 1] = attenuation
        domain = (
            self.derivative_domain
            if isinstance(self, MovingLayoutSignalCPI)
            else self.valid_mean
        )

        def unit_curve(value: Array) -> Array:
            parameters = value.copy()
            parameters[0] = 1.0
            return self.concentration(parameters)

        for index in range(2, mean.size):
            derivative = fourth_order_partial(
                unit_curve,
                mean,
                unit_concentration,
                index,
                max(1.0, abs(mean[index])),
                domain,
            )
            result[:, index] = -self.te * mean[0] * prediction * derivative
        return result

    def starts(self) -> tuple[Array, ...]:
        """Construct data-derived starts, without fitting a MAP initializer."""
        starts = []
        for delay in dict.fromkeys(
            (self.seed_delay, 0.0, self.seed_delay - 2 * self.dt)
        ):
            mean = np.r_[
                self.seed_flow,
                self.seed_baseline,
                np.maximum(-np.diff(self.knots) / self.seed_mtt, -23.0),
                delay,
            ]
            if self.model == "cpi_vtf":
                mean = np.r_[mean, np.log([2.0, 2.0])]
            mean = self.prepare_start(mean)
            if self.valid_mean(mean):
                starts.append(mean)
        if not starts:
            raise ValueError("no valid data-derived initialization")
        return tuple(starts)

    def prepare_start(self, mean: Array) -> Array:
        """Return fixed-layout coordinates unchanged."""
        return mean


class MovingLayoutSignalCPI(FixedLayoutSignalCPI):
    """Infer actual log intervals, with a fixed-time free-amplitude final knot.

    The first n-2 intervals are independent log coordinates with Gaussian
    priors. The final interval is the time left before the acquisition endpoint.
    All intervals must be at least one third of a frame. The shared forward
    operator and amplitude/flow/delay priors are unchanged by knot motion.
    """

    def __init__(
        self,
        observed: Array,
        aif: Array,
        time: Array,
        dt: float,
        te: float,
        n_control_points: int,
        model: Model,
        *,
        integration: Literal["native", "continuous"] = "native",
        aif_interpolation: Literal["linear", "pchip"] = "linear",
    ) -> None:
        super().__init__(
            observed,
            aif,
            time,
            dt,
            te,
            n_control_points,
            model,
            integration=integration,
            aif_interpolation=aif_interpolation,
        )
        self.spacing_start = self.delay_index + 1
        self.spacing_indices = slice(self.spacing_start, 2 * n_control_points)
        count = n_control_points - 2
        self.spacing_domain = LogSpacingDomain(
            self.spacing_indices, self.duration - dt / 3
        )

        def insert(base: Array, values: Array) -> Array:
            return np.asarray(
                np.r_[base[: self.spacing_start], values, base[self.spacing_start :]]
            )

        self.prior = GaussianPrior(
            insert(self.prior.mean, np.full(count, np.log(8.0))),
            np.diag(insert(np.diag(self.prior.covariance), np.full(count, 0.40))),
        )
        self.lower = insert(self.lower, np.full(count, np.log(dt / 3)))
        self.upper = insert(
            self.upper, np.full(count, np.log(self.duration - count * dt / 3))
        )

    def valid_mean(self, mean: Array) -> bool:
        """Respect both individual spacing bounds and the fixed endpoint."""
        return super().valid_mean(mean) and self.spacing_domain.valid_mean(mean)

    def knot_times(self, mean: Array) -> Array:
        """Construct ordered knots from inferred, unnormalized log intervals."""
        return np.asarray(
            np.r_[0.0, np.cumsum(np.exp(mean[self.spacing_indices])), self.duration]
        )

    def prepare_start(self, mean: Array) -> Array:
        """Initialize free intervals from the data-independent logarithmic layout."""
        return np.asarray(
            np.r_[
                mean[: self.spacing_start],
                np.log(np.diff(self.knots)[:-1]),
                mean[self.spacing_start :],
            ]
        )

    def derivative_domain(self, mean: Array) -> bool:
        """Differentiate ordered positive gaps beyond the minimum-spacing guard.

        The interpolant is well-defined when gaps are positive, even below the
        minimum permitted for a fitted mean. This extension avoids undefined
        coordinate derivatives where both minimum and total-gap limits bind.
        """
        if mean.shape != self.prior.mean.shape or not np.all(np.isfinite(mean)):
            return False
        lower, upper = self.lower.copy(), self.upper.copy()
        lower[self.spacing_indices] = np.log(np.finfo(float).tiny)
        upper[self.spacing_indices] = np.log(self.duration)
        return bool(
            np.all(mean >= lower)
            and np.all(mean <= upper)
            and np.sum(np.exp(mean[self.spacing_indices])) < self.duration
        )
