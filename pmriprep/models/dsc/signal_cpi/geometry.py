"""Signal acquisition validation and data-derived CPI initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
from scipy.special import gammainc

from pmriprep.utils.dsc_deconvolution import build_volterra_matrix
from .probability import Array

Model = Literal["cpi", "cpi_vtf"]


def _validate(
    signal: npt.ArrayLike,
    aif_concentration: npt.ArrayLike,
    time_seconds: npt.ArrayLike,
    echo_time_seconds: float,
) -> tuple[Array, Array, Array, float]:
    """Validate a uniformly sampled signal/AIF acquisition."""
    observed = np.asarray(signal, dtype=np.float64)
    aif = np.asarray(aif_concentration, dtype=np.float64)
    time = np.asarray(time_seconds, dtype=np.float64)
    if any(value.ndim != 1 for value in (observed, aif, time)):
        raise ValueError("signal, AIF, and time must be one-dimensional")
    if not (observed.size == aif.size == time.size) or observed.size < 20:
        raise ValueError("signal, AIF, and time must have equal length of at least 20")
    if not all(np.all(np.isfinite(value)) for value in (observed, aif, time)):
        raise ValueError("signal, AIF, and time must be finite")
    if np.any(observed < 0) or np.max(observed) <= 0 or np.max(aif) <= 0:
        raise ValueError(
            "signal must be non-negative and signal/AIF must contain information"
        )
    if not np.isfinite(echo_time_seconds) or echo_time_seconds <= 0:
        raise ValueError("echo_time_seconds must be finite and positive")
    steps = np.diff(time)
    if np.any(steps <= 0):
        raise ValueError("time must be strictly increasing")
    dt = float(np.median(steps))
    if not np.allclose(steps, dt, rtol=1e-3, atol=1e-6):
        raise ValueError("the reference currently requires uniform sampling")
    return observed, aif, time - time[0], dt


def _times_from_coordinates(
    coordinates: Array, n_points: int, duration: float, dt: float
) -> Array:
    """Map coordinates to ordered knots with TR/3 minimum separation."""
    intervals = n_points - 1
    floor = dt / 3.0
    available = duration - intervals * floor
    if available <= 0:
        raise ValueError("acquisition too short for requested control-point order")
    logits = np.concatenate((coordinates, [0.0]))
    weights = np.exp(logits - np.max(logits))
    weights /= np.sum(weights)
    knots = np.concatenate(([0.0], np.cumsum(floor + available * weights)))
    knots[-1] = duration
    return np.asarray(knots, dtype=np.float64)


def _initial_coordinates(n_points: int, duration: float, dt: float) -> Array:
    """Initialize ordered knots with greater resolution at early times."""
    scale = max(2.0 * dt, 1.0)
    u = np.linspace(0.0, 1.0, n_points)
    knots = scale * np.expm1(u * np.log1p(duration / scale))
    free = np.maximum(np.diff(knots) - dt / 3.0, np.finfo(float).eps)
    return np.asarray(np.log(free[:-1] / free[-1]), dtype=np.float64)


def _gamma_vtf(time: Array, dt: float, ttp: float, sharpness: float) -> Array:
    """Return gamma mass per lag bin without renormalizing a truncated tail."""
    if ttp <= 0 or sharpness <= 0:
        raise ValueError("gamma peak and sharpness must be positive")
    shape = 1.0 + sharpness * ttp
    upper = gammainc(shape, sharpness * (time + 0.5 * dt))
    lower = gammainc(shape, sharpness * np.maximum(time - 0.5 * dt, 0.0))
    return np.asarray((upper - lower) / dt, dtype=np.float64)


@dataclass(frozen=True)
class _VolterraConvolution:
    """Pickle-safe corrected convolution with a fixed acquired input."""

    kernel: Array
    correction: Array

    def __call__(self, residue: Array) -> Array:
        """Apply the prepared kernel and origin correction without approximation."""
        return np.asarray(
            np.convolve(self.kernel, residue)[:self.correction.size]
            + self.correction * residue[0]
        )


def _volterra_apply(aif: Array, dt: float) -> Callable[[Array], Array]:
    """Prepare convolution exactly equivalent to the corrected Volterra matrix."""
    matrix = np.asarray(build_volterra_matrix(aif, dt), dtype=np.float64)
    kernel = np.concatenate((matrix[1:, 1], [0.0]))
    correction = matrix[:, 0] - kernel

    return _VolterraConvolution(kernel, correction)


def _spectral_seed(
    observed: Array, aif: Array, time: Array, dt: float, te: float
) -> tuple[float, float, float]:
    """Initialize from a truncated circular inverse, used only as a seed."""
    pre = max(4, min(observed.size // 5, int(np.argmax(aif > 0.05 * np.max(aif)))))
    baseline = max(float(np.median(observed[:pre])), np.finfo(float).tiny)
    concentration = -np.log(np.maximum(observed / baseline, 1e-8)) / te
    n = 2 * len(time)
    spectrum = np.fft.rfft(aif, n=n) * dt
    threshold = 0.2 * np.max(np.abs(spectrum))
    inverse = np.zeros_like(spectrum)
    keep = np.abs(spectrum) > threshold
    inverse[keep] = 1.0 / spectrum[keep]
    response = np.fft.irfft(np.fft.rfft(concentration, n=n) * inverse, n=n)
    peak = int(np.argmax(response))
    delay = (peak if peak < n // 2 else peak - n) * dt
    flow = max(float(response[peak]), 1e-6)
    volume = max(
        float(np.trapezoid(concentration, dx=dt) / np.trapezoid(aif, dx=dt)), 0.0
    )
    mtt = float(np.clip(volume / flow, dt, time[-1]))
    return flow, mtt, float(np.clip(delay, -time[-1] / 2, time[-1] / 2))
