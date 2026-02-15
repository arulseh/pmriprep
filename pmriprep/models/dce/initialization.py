"""Shared initialization utilities for DCE pharmacokinetic models.

Provides data-driven initial parameter estimation for Tofts-family models
(Standard Tofts, Extended Tofts). These functions estimate starting values
from the observed tissue concentration curve and AIF shape.

Functions:
    estimate_ktrans_patlak: Estimate Ktrans via Patlak regression on post-bolus data.
    estimate_ve_late_phase: Estimate ve from late-phase quasi-equilibrium ratio.
    estimate_vp_peak_ratio: Estimate vp from tissue-to-AIF peak ratio.
"""

__all__ = [
    "estimate_ktrans_patlak",
    "estimate_ve_late_phase",
    "estimate_vp_peak_ratio",
]

import numpy as np
import numpy.typing as npt


def estimate_ktrans_patlak(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    peak_idx: int,
) -> float:
    """Estimate Ktrans via Patlak regression on post-bolus data.

    Performs linear regression on the Patlak plot (C_t/C_p vs integral(C_p)/C_p)
    using post-bolus time points where the AIF is above 5% of the peak.
    Requires at least 4 valid post-bolus points.

    Args:
        tissue_curve: Observed tissue concentration C_t(t).
        aif: Arterial input function Cp(t).
        time_points: Time points in minutes.
        peak_idx: Index of AIF peak.

    Returns:
        Estimated Ktrans value, or 0.1 if regression cannot be performed.
    """
    n = len(time_points)
    dt = float(np.mean(np.diff(time_points)))

    post_bolus = np.arange(peak_idx + 1, n)
    cp_post = aif[post_bolus]
    valid = cp_post > 0.05 * np.max(aif)

    if np.sum(valid) < 4:
        return 0.1

    idx = post_bolus[valid]
    cp = aif[idx]
    ct = tissue_curve[idx]
    cum_aif = np.cumsum(aif[: idx[-1] + 1]) * dt
    X = cum_aif[idx] / cp
    Y = ct / cp
    slope = float(np.polyfit(X, Y, 1)[0])
    if not np.isfinite(slope):
        return 0.1
    return max(slope, 0.001)


def estimate_ve_late_phase(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    peak_idx: int,
) -> float:
    """Estimate ve from late-phase quasi-equilibrium ratio.

    At late time points, the tissue and plasma concentrations approach
    equilibrium: C_t/C_p -> ve. Uses the mean ratio of tissue-to-AIF
    concentration starting 12 frames after peak.

    Args:
        tissue_curve: Observed tissue concentration C_t(t).
        aif: Arterial input function Cp(t).
        peak_idx: Index of AIF peak.

    Returns:
        Estimated ve value (clipped to 0.01-0.8), or 0.2 if estimation
        cannot be performed.
    """
    n = len(tissue_curve)
    late_start = min(peak_idx + 12, n - 5)

    if late_start >= n - 3:
        return 0.2

    ct_late = tissue_curve[late_start:]
    cp_late = aif[late_start:]
    mean_cp = float(np.mean(cp_late))

    if mean_cp <= 0.05 * np.max(aif):
        return 0.2

    return float(np.clip(np.mean(ct_late) / mean_cp, 0.01, 0.8))


def estimate_vp_peak_ratio(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
) -> float:
    """Estimate vp from tissue-to-AIF peak ratio.

    At the bolus peak, the vascular contribution dominates:
    C_t(peak) ~ vp * C_p(peak), so vp ~ C_t(peak) / C_p(peak).

    Args:
        tissue_curve: Observed tissue concentration C_t(t).
        aif: Arterial input function Cp(t).

    Returns:
        Estimated vp value (unclipped — caller should clip to VP_BOUNDS),
        or 0.02 if AIF peak is near zero.
    """
    aif_peak = float(np.max(aif))
    tissue_peak = float(np.max(tissue_curve))

    if aif_peak <= 1e-6:
        return 0.02

    return float(tissue_peak / aif_peak)
