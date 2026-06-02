"""Test-retest reproducibility metrics for perfusion MRI.

This module provides pure-computation functions for quantifying
test-retest reproducibility of perfusion parameter maps. All functions
operate on numpy arrays with no I/O dependencies.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


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


@dataclass
class ReproducibilityResult:
    """Container for all reproducibility metrics for a parameter.

    Attributes
    ----------
        parameter_name: Name of the perfusion parameter (e.g., "Ktrans").
        median_ses1: Median value in session 1.
        median_ses2: Median value in session 2.
        percent_difference: Percent difference between session medians.
        within_subject_cv: Within-subject coefficient of variation.
        pearson_r: Pearson correlation coefficient (voxel-wise, if available).
        icc: ICC(2,1) value (voxel-wise, if available).
        bland_altman: Bland-Altman analysis result (if available).
        n_voxels_ses1: Number of valid voxels in session 1.
        n_voxels_ses2: Number of valid voxels in session 2.
    """

    parameter_name: str
    median_ses1: float
    median_ses2: float
    percent_difference: float
    within_subject_cv: float
    pearson_r: Optional[float] = None
    icc: Optional[float] = None
    bland_altman: Optional[BlandAltmanResult] = None
    n_voxels_ses1: int = 0
    n_voxels_ses2: int = 0


def compute_percent_difference(
    v1: NDArray[np.floating],
    v2: NDArray[np.floating],
) -> float:
    """Compute percent difference between two scalar or array summaries.

    For scalar inputs (e.g., session medians):
        %diff = 2 * |v1 - v2| / (v1 + v2) * 100

    For array inputs, computes the median of voxel-wise percent differences,
    excluding voxels where both values are near zero.

    Parameters
    ----------
    v1 : NDArray
        First measurement (scalar-like or 1D array).
    v2 : NDArray
        Second measurement (same shape as v1).

    Returns
    -------
    float
        Percent difference. Returns 0.0 if both inputs are zero.

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
    v1: NDArray[np.floating],
    v2: NDArray[np.floating],
) -> float:
    """Compute within-subject coefficient of variation.

    For two repeated measurements, CV = median_%diff / 2.
    This is the standard approach for test-retest CV with k=2 sessions.

    Parameters
    ----------
    v1 : NDArray
        First measurement (scalar-like or 1D array).
    v2 : NDArray
        Second measurement (same shape as v1).

    Returns
    -------
    float
        Within-subject CV as a percentage.
    """
    return compute_percent_difference(v1, v2) / 2.0


def compute_icc_2_1(
    v1: NDArray[np.floating],
    v2: NDArray[np.floating],
) -> float:
    """Compute ICC(2,1) — two-way random, single measures.

    Uses Shrout & Fleiss (1979) formulation for k=2 raters/sessions:
        ICC(2,1) = (BMS - EMS) / (BMS + EMS + 2*(JMS - EMS)/n)

    where:
        BMS = between-subjects mean square
        JMS = between-judges (sessions) mean square
        EMS = error (residual) mean square
        n = number of subjects (voxels)

    Parameters
    ----------
    v1 : NDArray
        First session values (1D array).
    v2 : NDArray
        Second session values (1D array, same length).

    Returns
    -------
    float
        ICC(2,1) value. Typically in range [-1, 1], but can exceed
        these bounds with small samples.

    Raises
    ------
    ValueError
        If inputs have different lengths or fewer than 3 elements.
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
    if n < 3:
        raise ValueError(f"Need at least 3 valid pairs, got {n}")

    k = 2  # Number of sessions/raters

    # Stack into n x k matrix
    data = np.column_stack([v1, v2])

    # Grand mean
    grand_mean = np.mean(data)

    # Subject means (row means)
    subject_means = np.mean(data, axis=1)

    # Judge/session means (column means)
    judge_means = np.mean(data, axis=0)

    # Sum of squares
    # Between-subjects SS
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)

    # Between-judges SS
    ss_judges = n * np.sum((judge_means - grand_mean) ** 2)

    # Total SS
    ss_total = np.sum((data - grand_mean) ** 2)

    # Error (residual) SS
    ss_error = ss_total - ss_subjects - ss_judges

    # Degrees of freedom
    df_subjects = n - 1
    df_judges = k - 1
    df_error = (n - 1) * (k - 1)

    # Mean squares
    bms = ss_subjects / df_subjects  # Between-subjects MS
    jms = ss_judges / df_judges if df_judges > 0 else 0.0  # Between-judges MS
    ems = ss_error / df_error if df_error > 0 else 0.0  # Error MS

    # ICC(2,1)
    denominator = bms + ems + 2.0 * (jms - ems) / n
    if abs(denominator) < 1e-10:
        return 0.0

    icc = (bms - ems) / denominator
    return float(icc)


def compute_bland_altman(
    v1: NDArray[np.floating],
    v2: NDArray[np.floating],
) -> BlandAltmanResult:
    """Compute Bland-Altman analysis (bias and 95% limits of agreement).

    Follows Bland & Altman (1986):
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
