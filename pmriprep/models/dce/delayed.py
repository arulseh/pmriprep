"""Signed delayed PCHIP input with analytic cubic-exponential integration.

Times and exponential rates use minutes; delay uses seconds. Input endpoints
are held, and convolution starts at zero at the first acquisition time.
Selecting output frames never removes intervening AIF history.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def _cubic_moments(x: np.ndarray) -> list[np.ndarray]:
    """Return integrals of u**m exp(-x*(1-u)), u=0..1, m=0..3.

    Parameters
    ----------
    x : ndarray
        Nonnegative rate times interval length.

    Returns
    -------
    list of ndarray
        Four bounded moments, stable at zero and large rates.
    """
    small = x < 1.
    xs = np.where(small, x, 0.)
    safe = np.where(small, 1., x)
    out: list[np.ndarray] = []
    for order in range(4):
        term = np.full_like(x, 1. / (order + 1))
        series = term.copy()
        for j in range(1, 20):
            term = term * (-xs) / (order + j + 1)
            series += term
        recurrence = (-np.expm1(-x) / safe if order == 0
                      else (1. - order * out[-1]) / safe)
        out.append(np.where(small, series, recurrence))
    return out


class DelayedPchip:
    """Integrate a signed shifted input, with held endpoints and zero initial state.

    Parameters
    ----------
    aif, times : ndarray
        Finite plasma AIF in mM and uniform increasing acquisition times in
        minutes. The first acquisition time defines the integration origin.
    delay_seconds : float
        Effective offset; positive means tissue later than the AIF.
    observed : ndarray, optional
        Integer output indices, retained in the supplied order. Omitting
        observations never removes input history.

    Attributes
    ----------
    cp : ndarray
        Shifted plasma input at retained observations, in mM.
    observed : ndarray
        Original acquisition indices of the retained observations.
    output_t : ndarray
        Retained times in minutes relative to acquisition origin.
    held_before, held_after : ndarray
        Flags for observations using a held input endpoint.

    Notes
    -----
    Negative and beyond-grid offsets are allowed for timing diagnostics. The
    operator does not impose a delay-selection policy or discard baseline input.
    """

    def __init__(
        self,
        aif: np.ndarray,
        times: np.ndarray,
        delay_seconds: float = 0.0,
        observed: np.ndarray | None = None,
    ) -> None:
        aif, times = np.array(aif, dtype=float, copy=True), np.asarray(times, float)
        if (
            aif.ndim != 1
            or aif.shape != times.shape
            or len(aif) < 2
            or not np.isfinite(aif).all()
            or not np.isfinite(times).all()
            or not np.isfinite(delay_seconds)
        ):
            raise ValueError("Finite matching vectors and finite delay required.")
        self.h = float(times[1] - times[0])
        if self.h <= 0 or not np.allclose(
            np.diff(times), self.h, rtol=1e-12, atol=1e-14
        ):
            raise ValueError("Uniform increasing input times required.")
        self.t = np.arange(len(aif)) * self.h
        self.aif = aif
        self.delay = delay_seconds / 60
        indices = np.arange(len(aif)) if observed is None else np.asarray(observed)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(aif))
        ):
            raise ValueError("Valid integer observation indices required.")
        self.observed = indices.copy()
        self.output_t = self.t[self.observed]
        self.q = self.output_t - self.delay
        self.held_before = self.q < 0
        self.held_after = self.q > self.t[-1]
        self.poly = PchipInterpolator(self.t, aif, extrapolate=False)
        self.cp = np.asarray(self.poly(np.clip(self.q, 0, self.t[-1])))

        shifted = self.t + self.delay
        self.knots = np.unique(
            np.r_[0.0, shifted[(shifted > 0) & (shifted < self.t[-1])], self.t[-1]]
        )
        self.widths = np.diff(self.knots)
        coefficients = np.zeros((4, len(self.widths)))
        mid = (self.knots[:-1] + self.knots[1:]) / 2 - self.delay
        before, after = mid < 0, mid > self.t[-1]
        coefficients[0, before], coefficients[0, after] = aif[0], aif[-1]
        inside = ~(before | after)
        segment = np.clip(
            np.searchsorted(self.t, mid[inside], side="right") - 1, 0, len(aif) - 2
        )
        offset = self.knots[:-1][inside] - self.delay - self.t[segment]
        c = self.poly.c[::-1, segment]
        coefficients[0, inside] = c[0] + offset * (
            c[1] + offset * (c[2] + offset * c[3])
        )
        coefficients[1, inside] = c[1] + offset * (2 * c[2] + 3 * offset * c[3])
        coefficients[2, inside] = c[2] + 3 * offset * c[3]
        coefficients[3, inside] = c[3]
        self.coefficients = coefficients
        self.k = np.clip(
            np.searchsorted(self.knots, self.output_t, side="right") - 1,
            0,
            len(self.widths) - 1,
        )
        self.partial = self.output_t - self.knots[self.k]

    def convolution(self, rate: float | np.ndarray) -> np.ndarray:
        """Return integral_0^t Cp(u-d) exp[-rate*(t-u)] du in mM minutes.

        Parameters
        ----------
        rate : float or ndarray
            Nonnegative scalar or one-dimensional array in min^-1.

        Returns
        -------
        ndarray
            Frames for scalar input, or rates by frames for a vector.
        """
        scalar = np.ndim(rate) == 0
        rates = np.atleast_1d(np.asarray(rate, float))
        if rates.ndim != 1 or not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("Finite nonnegative scalar/vector rates required.")
        x = rates[:, None] * self.widths
        g = _cubic_moments(x)
        increments = sum(
            g[m] * self.widths ** (m + 1) * self.coefficients[m] for m in range(4)
        )
        states = np.zeros((len(rates), len(self.knots)))
        decay = np.exp(-x)
        for j in range(len(self.widths)):
            states[:, j + 1] = decay[:, j] * states[:, j] + increments[:, j]
        partial_x = rates[:, None] * self.partial
        out = np.exp(-partial_x) * states[:, self.k]
        gp = _cubic_moments(partial_x)
        for m in range(4):
            out += gp[m] * self.partial ** (m + 1) * self.coefficients[m, self.k]
        return np.asarray(out[0] if scalar else out)
