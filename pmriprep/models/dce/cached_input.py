"""Reusable convolution workspace for the shared delayed plasma input."""

from __future__ import annotations

import ctypes as ct
from threading import Lock
from typing import Any

import numpy as np
from scipy.special import hyp1f1

from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce._native_backend import D, RECURRENCE


class CachedPchip(DelayedPchip):
    """Cache held-endpoint PCHIP integrals for nonnegative-offset kinetic fits.

    Parameters
    ----------
    aif, times : ndarray
        Finite plasma AIF in mM and uniform acquisition times in minutes.
    delay_seconds : float
        Nonnegative effective AIF offset in seconds.
    observed : ndarray, optional
        Original retained frame indices. Full input history is preserved.

    Notes
    -----
    Convolution returns independent arrays; cached workspace access is locked.
    Cache-specific workspace is separate from the inherited input geometry.
    Negative diagnostic offsets use the general DelayedPchip operator.
    """

    def __init__(
        self,
        aif: np.ndarray,
        times: np.ndarray,
        delay_seconds: float = 0.0,
        observed: np.ndarray | None = None,
    ) -> None:
        if not np.isfinite(delay_seconds) or delay_seconds < 0:
            raise ValueError("Cached input requires a finite nonnegative delay.")
        super().__init__(aif, times, delay_seconds, observed)
        self._interval_coefficients = self.poly.c[::-1]
        clipped = np.clip(self.q, 0.0, self.t[-1])
        self._interval_index = np.clip(
            np.searchsorted(self.t, clipped, side="right") - 1, 0, len(aif) - 2
        )
        self.s = clipped - self.t[self._interval_index]
        self.unique_s, self.inverse_s = np.unique(self.s, return_inverse=True)
        self.full_coeff = self._interval_coefficients * self.h ** np.arange(1, 5)[:, None]
        self.part_coeff = (
            self._interval_coefficients[:, self._interval_index] * self.s ** np.arange(1, 5)[:, None]
        )
        self.held = np.minimum(self.output_t, self.delay)
        self.tail = np.maximum(self.output_t - self.delay, 0.0)
        self._moment_widths = np.r_[self.h, self.unique_s]
        self.before = self.q < 0
        self.buffers: dict[int, tuple[Any, ...]] = {}
        self._lock = Lock()

    def convolution(self, rate: float | np.ndarray) -> np.ndarray:
        """Evaluate the same PCHIP exponential integrals, returning fresh storage.

        Parameters
        ----------
        rate : float or ndarray
            Nonnegative scalar or one-dimensional rates.

        Returns
        -------
        ndarray
            Independent output, unaffected by later calls on this context.
        """
        with self._lock:
            return self._convolution(rate)

    def _convolution(self, rate: float | np.ndarray) -> np.ndarray:
        """Evaluate with exclusive access to the context buffers."""
        scalar = np.ndim(rate) == 0
        rates = np.atleast_1d(np.asarray(rate, float))
        if rates.ndim != 1 or not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("Finite nonnegative rates required.")
        count = len(rates)
        if count not in self.buffers:
            increments = np.empty((count, self.full_coeff.shape[1]), dtype=np.float64)
            decay = np.empty(count, dtype=np.float64)
            states = np.empty((count, len(self.t)), dtype=np.float64)
            ptrs = tuple(a.ctypes.data_as(D) for a in [increments, decay, states])
            self.buffers[count] = (increments, decay, states, *ptrs)
        increments, decay, states, ip, dp, sp = self.buffers[count]
        x = rates[:, None] * self._moment_widths
        negative_x = -x
        moment = np.stack(
            [hyp1f1(1.0, m + 2.0, negative_x) / (m + 1.0) for m in range(4)]
        )
        np.matmul(moment[:, :, 0].T, self.full_coeff, out=increments)
        np.exp(-rates * self.h, out=decay)
        RECURRENCE(ip, dp, sp, ct.c_size_t(count), ct.c_size_t(increments.shape[1]))
        out = np.exp(-rates[:, None] * self.s) * states[:, self._interval_index]
        out += np.sum(
            moment[:, :, 1:][:, :, self.inverse_s] * self.part_coeff[:, None, :], axis=0
        )
        out[:, self.before] = 0.0
        if self.aif[0] != 0:
            held_moment = hyp1f1(1.0, 2.0, -rates[:, None] * self.held)
            out += (
                self.aif[0]
                * np.exp(-rates[:, None] * self.tail)
                * self.held
                * held_moment
            )
        return np.asarray(out[0] if scalar else out)
