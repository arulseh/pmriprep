"""Validation of shared frame-wise likelihood weights."""

from __future__ import annotations

import numpy as np

from .probability import Array


def checked_weights(weights: Array | None, frames: int) -> Array | None:
    """Return owned downweighting factors, canonicalizing uniform ones to None.

    Parameters
    ----------
    weights : ndarray or None
        One finite likelihood weight per frame, between zero and one.
        At least one frame must retain positive weight. None is unweighted.
    frames : int
        Number of acquired observations.

    Returns
    -------
    ndarray or None
        Read-only weights, or None for the exact unweighted path. Weights are
        not normalized: their sum determines the effective observation count.

    Raises
    ------
    ValueError
        If weights do not provide one finite value in [0, 1] per frame
        with a positive total.
    """
    if weights is None:
        return None
    result = np.asarray(weights, dtype=float)
    if (result.shape != (frames,) or not np.all(np.isfinite(result))
            or np.any(result < 0) or np.any(result > 1) or result.sum() <= 0):
        raise ValueError("one finite weight in [0, 1] per frame and positive total required")
    if np.all(result == 1):
        return None
    result = result.copy()
    result.flags.writeable = False
    return result
