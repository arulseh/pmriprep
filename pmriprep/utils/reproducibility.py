"""Test-retest reproducibility metrics for perfusion MRI.

This module provides pure-computation functions for quantifying
test-retest reproducibility of perfusion parameter maps. All functions
operate on numpy arrays with no I/O dependencies.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class BlandAltmanResult:
    """Results from a Bland-Altman analysis.

    Attributes
    ----------
        bias: Mean difference (v1 - v2).
        lower_limit: Lower 95% limit of agreement (bias - 1.96*sd).
        upper_limit: Upper 95% limit of agreement (bias + 1.96*sd).
        sd_diff: Standard deviation of differences.
        n_samples: Number of valid paired samples.
    """

    bias: float
    lower_limit: float
    upper_limit: float
    sd_diff: float
    n_samples: int


def compute_percent_difference(
    v1: ArrayLike,
    v2: ArrayLike,
) -> float:
    """Compute percent difference between two scalar or array summaries.

    For scalar inputs (e.g., session medians)::

        %diff = 2 * |v1 - v2| / (v1 + v2) * 100

    For array inputs, computes the median of voxel-wise percent differences,
    retaining only finite pairs whose absolute pair mean exceeds 1e-10.

    Parameters
    ----------
    v1 : array-like
        First measurement (scalar-like or array).
    v2 : array-like
        Second measurement (same shape as v1).

    Returns
    -------
    float
        Percent difference. For single-element inputs, returns 0.0 when the
        sum is zero. For larger arrays, returns 0.0 when no valid pair remains.

    Raises
    ------
    ValueError
        If inputs have different shapes.
    """
    v1 = np.atleast_1d(np.asarray(v1, dtype=np.float64))
    v2 = np.atleast_1d(np.asarray(v2, dtype=np.float64))

    if v1.shape != v2.shape:
        raise ValueError(
            f"Input shapes must match: {v1.shape} vs {v2.shape}"
        )

    if v1.size == 1 and v2.size == 1:
        # Scalar case
        s = float(v1.item()) + float(v2.item())
        if s == 0.0:
            return 0.0
        return 2.0 * abs(float(v1.item()) - float(v2.item())) / s * 100.0

    # Array case: voxel-wise percent difference
    mean_val = (v1 + v2) / 2.0
    valid = np.isfinite(v1) & np.isfinite(v2) & (np.abs(mean_val) > 1e-10)

    if not np.any(valid):
        return 0.0

    diff = np.abs(v1[valid] - v2[valid])
    pct = 2.0 * diff / (v1[valid] + v2[valid]) * 100.0
    return float(np.median(pct))


def compute_within_subject_cv(
    v1: ArrayLike,
    v2: ArrayLike,
) -> float:
    """Compute within-subject coefficient of variation.

    Returns half of ``compute_percent_difference(v1, v2)``, using the
    same pair filtering and zero-result rules.

    Parameters
    ----------
    v1 : array-like
        First measurement (scalar-like or array).
    v2 : array-like
        Second measurement (same shape as v1).

    Returns
    -------
    float
        Within-subject CV as a percentage.

    Raises
    ------
    ValueError
        If inputs have different shapes after conversion to at least 1D.
    """
    return compute_percent_difference(v1, v2) / 2.0


def compute_bland_altman(
    v1: NDArray[np.floating],
    v2: NDArray[np.floating],
) -> BlandAltmanResult:
    """Compute Bland-Altman analysis (bias and 95% limits of agreement).

    Computes Bland-Altman limits as:
        - Bias = mean(v1 - v2)
        - Limits = bias +/- 1.96 * SD(v1 - v2)

    Parameters
    ----------
    v1 : NDArray
        First session values (1D array).
    v2 : NDArray
        Second session values (1D array, same length).

    Returns
    -------
    BlandAltmanResult
        BlandAltmanResult with bias, limits of agreement, and sample size.

    Raises
    ------
    ValueError
        If inputs have different lengths or fewer than 2 elements.
    """
    v1 = np.asarray(v1, dtype=np.float64).ravel()
    v2 = np.asarray(v2, dtype=np.float64).ravel()

    if v1.shape != v2.shape:
        raise ValueError(
            f"Input shapes must match: {v1.shape} vs {v2.shape}"
        )

    # Filter NaN/Inf
    valid = np.isfinite(v1) & np.isfinite(v2)
    v1 = v1[valid]
    v2 = v2[valid]

    n = len(v1)
    if n < 2:
        raise ValueError(f"Need at least 2 valid pairs, got {n}")

    diff = v1 - v2
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))

    return BlandAltmanResult(
        bias=bias,
        lower_limit=bias - 1.96 * sd_diff,
        upper_limit=bias + 1.96 * sd_diff,
        sd_diff=sd_diff,
        n_samples=n,
    )
