"""Optional runtime dispatch to the installed compiled convolution kernel."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

import numpy as np

_kernel: ModuleType | None
try:
    _kernel = importlib.import_module(f"{__package__}._convolution")
except ImportError:
    _kernel = None

_warned = False


def available() -> bool:
    """Report compiled availability and warn once before Python-only fitting."""
    global _warned
    if _kernel is None and not _warned:
        logging.getLogger(__name__).warning(
            "Compiled CPI convolution is unavailable; using the slower Python "
            "calculation. Build/install the package to enable acceleration."
        )
        _warned = True
    return _kernel is not None


def convolve(
    knots: np.ndarray, coefficients: np.ndarray, target: np.ndarray,
    left: np.ndarray, right: np.ndarray, arterial: np.ndarray,
    nodes: np.ndarray, weights: np.ndarray, *, chebyshev: bool = False,
    moments: np.ndarray | None = None,
) -> np.ndarray | None:
    """Evaluate unchanged polynomial integrals using validated native buffers.

    Parameters
    ----------
    knots, coefficients, target : ndarray
        Residue knots (rows, knots), descending cubic coefficients (rows, 4,
        knots), and elapsed observation times (rows, frames).
    left, right, arterial : ndarray
        Arterial interval edges and coefficient-by-interval matrix.
    nodes, weights : ndarray
        Gaussian integration rule for split intervals.
    chebyshev : bool, optional
        Interpret arterial coefficients in normalized Chebyshev coordinates.
    moments : ndarray, optional
        Four centered arterial moments for whole intervals.

    Returns
    -------
    ndarray or None
        Integrated curves, or None when the compiled module or native float64
        inputs are unavailable. Calculation failures propagate to the caller.

    Notes
    -----
    Copies only noncontiguous or unaligned input buffers. This function adds no
    threads, numerical approximation, fitting policy or runtime compilation.
    The extension independently validates buffers, including direct callers.
    """
    arrays = [knots, coefficients, target, left, right, arterial, nodes, weights]
    if moments is not None:
        arrays.append(moments)
    if not available() or any(a.dtype != np.dtype("float64") for a in arrays):
        return None
    buffers = [np.require(a, dtype=np.float64, requirements=["C", "A"]) for a in arrays]
    if target.ndim != 2:
        raise ValueError("elapsed times must be a two-dimensional array")
    output = np.empty(target.shape, dtype=np.float64)
    assert _kernel is not None  # Availability checked above, not native input validation.
    _kernel.convolve(*buffers[:8], buffers[8] if moments is not None else None,
                     output, chebyshev)
    return output


def interval_layout(
    target: np.ndarray, left: np.ndarray, right: np.ndarray, lower: float, upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bytes] | None:
    """Enumerate active arterial intervals without rectangular temporary arrays.

    Parameters
    ----------
    target, left, right : ndarray
        Elapsed target times and arterial interval edges, as native float64.
    lower, upper : float
        Nonnegative lower and strictly greater finite upper kernel cutoffs.

    Returns
    -------
    tuple or None
        Read-only row indices, segment indices, lower/upper lags and the exact
        packed active mask. None selects Python for unavailable native code or
        unsupported dtypes. Returned arrays own their immutable byte backing.
    """
    arrays = (target, left, right)
    if not available() or any(a.dtype != np.float64 for a in arrays):
        return None
    buffers = [np.require(a, requirements=['C', 'A']) for a in arrays]
    assert _kernel is not None
    rows, segments, lo, hi, packed = _kernel.interval_layout(*buffers, lower, upper)
    return (np.frombuffer(rows, dtype=np.intp), np.frombuffer(segments, dtype=np.intp),
            np.frombuffer(lo, dtype=np.float64), np.frombuffer(hi, dtype=np.float64), packed)


def contract_moments(
    local: np.ndarray, segments: np.ndarray, coefficients: np.ndarray,
    factors: np.ndarray, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Contract cubic arterial coefficients with unchanged gamma interval masses.

    Parameters
    ----------
    local, segments : ndarray
        Time offsets and native-intp arterial indices for each active interval.
    coefficients : ndarray
        Descending cubic coefficients, shape (4, arterial intervals).
    factors : ndarray
        Four precomputed gamma moment multipliers.
    p0, p1, q0, q1 : ndarray
        Lower/upper-tail CDFs at interval endpoints, each shape (4, intervals).
        C- and Fortran-contiguous layouts may differ between arrays.

    Returns
    -------
    tuple or None
        Read-only moment values and roundoff estimates backed by owned bytes,
        or None when Python must handle unavailable native code or dtypes.

    Notes
    -----
    Non-finite arithmetic results propagate to the caller's cancellation guard
    and adaptive quadrature. This function does not decide integration accuracy.
    """
    arrays = (local, coefficients, factors, p0, p1, q0, q1)
    if (not available() or segments.dtype != np.intp
            or any(a.dtype != np.float64 for a in arrays)):
        return None
    leading = [np.require(a, requirements=['C', 'A'])
               for a in (local, segments, coefficients, factors)]
    cdfs = [np.require(a, requirements=['A'] if a.flags.f_contiguous else ['C', 'A'])
            for a in (p0, p1, q0, q1)]
    assert _kernel is not None
    value, error = _kernel.contract_moments(*leading, *cdfs)
    return np.frombuffer(value, dtype=np.float64), np.frombuffer(error, dtype=np.float64)
