"""Continuous gamma transport forward operator."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gammainc, gammaincinv

from .prepared import PreparedCPIConvolution
from .probability import Array


class DispersionIntegrationError(RuntimeError):
    """Requested dispersion quadrature accuracy was not established."""


class ContinuousDispersedCPI:
    """Convolve the acquired AIF, intrinsic residue and unit-area gamma kernel.

    Parameters
    ----------
    aif, time : ndarray
        AIF and strictly increasing relative acquisition times starting at zero.
    aif_interpolation : {"linear", "pchip"}, optional
        Reconstruction of the acquired AIF between its samples.

    Notes
    -----
    The kernel has shape 1 + sharpness * time_to_peak and rate sharpness.
    Integration uses its probability coordinate, where the density cancels
    against the change-of-variable Jacobian. This resolves subframe kernels
    without sampling their density on the acquisition grid. Probability mass
    after the integration horizon is omitted, never redistributed or normalized
    into the acquired window. Residue and AIF support follow the plain operator.
    """

    def __init__(
        self,
        aif: Array,
        time: Array,
        *,
        aif_interpolation: Literal["linear", "pchip"] = "pchip",
    ) -> None:
        self.base = PreparedCPIConvolution(
            aif, time, aif_interpolation=aif_interpolation
        )

    def evaluate(
        self,
        knots: Array,
        values: Array,
        flow: float,
        delay: float,
        time_to_peak: float,
        sharpness: float,
    ) -> Array:
        """Evaluate concentration before the measured-signal exponential.

        Parameters
        ----------
        knots, values : ndarray
            Intrinsic residue, with the same meaning as in plain CPI.
        flow, delay : float
            Flow multiplier and signed tissue delay in the acquisition units.
        time_to_peak : float
            Nonnegative kernel mode in the same time units as the acquisition.
        sharpness : float
            Positive gamma rate, in inverse acquisition-time units.

        Returns
        -------
        ndarray
            Flow times the causal triple convolution, at acquisition times.

        Raises
        ------
        ValueError
            If flow, delay, mode or rate is nonfinite, mode is negative, rate is
            not positive, or the resulting gamma shape is nonfinite.
        DispersionIntegrationError
            Adaptive integration fails to meet its relative error criterion.
        """
        if (
            not np.all(np.isfinite([flow, delay, time_to_peak, sharpness]))
            or time_to_peak < 0
            or sharpness <= 0
        ):
            raise ValueError(
                "finite flow/delay and nonnegative mode/positive rate required"
            )
        shape = 1.0 + sharpness * time_to_peak
        if not np.isfinite(shape):
            raise ValueError("gamma shape must be finite")
        target = self.base.time - delay
        plain = self.base.evaluate_at(knots, values, 1.0, target)
        if flow == 0 or self.base.left.size == 0:
            return np.zeros_like(target)
        elapsed = np.maximum(target - self.base.left[0], 0.0)
        probabilities = gammainc(shape, sharpness * elapsed)
        upper = float(np.max(probabilities))
        if upper == 0:
            return np.zeros_like(target)

        def integrand(probability: float) -> Array:
            if probability <= 0:
                return plain
            if probability >= 1:
                return np.zeros_like(target)
            lag = gammaincinv(shape, probability) / sharpness
            return self.base.evaluate_at(knots, values, 1.0, target - lag)

        points = np.unique(probabilities[(probabilities > 0) & (probabilities < upper)])
        integral, error, info = quad_vec(
            integrand,
            0.0,
            upper,
            epsabs=1e-200,
            epsrel=2e-12,
            norm="max",
            points=points,
            limit=10000,
            full_output=True,
        )
        if (
            not info.success
            or not np.isfinite(error)
            or not np.all(np.isfinite(integral))
        ):
            raise DispersionIntegrationError(
                f"gamma convolution did not meet integration accuracy: {info.message}"
            )
        return np.asarray(flow * integral)
