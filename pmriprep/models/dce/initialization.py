"""Shared initialization utilities for DCE pharmacokinetic models.

Provides data-driven initial parameter estimation for Tofts-family models
(Standard Tofts, Extended Tofts). These functions estimate starting values
from the observed tissue concentration curve and AIF shape.

Functions:
    estimate_ktrans_uptake: Estimate Ktrans from the initial-uptake slope.
    estimate_ve_late_phase: Estimate ve from late-phase quasi-equilibrium ratio.
    estimate_vp_peak_ratio: Estimate vp from tissue-to-AIF peak ratio.
"""

__all__ = [
    "estimate_ktrans_uptake",
    "estimate_ve_late_phase",
    "estimate_vp_peak_ratio",
]

import numpy as np
import numpy.typing as npt


def estimate_ktrans_uptake(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    peak_idx: int,
) -> float:
    """Estimate Ktrans from the initial-uptake slope.

    During the uptake phase, before efflux is significant, the tissue
    concentration follows C_t(t) ~ Ktrans * integral(Cp) (+ vp * Cp), so a
    linear least-squares solve of C_t against the cumulative AIF integral over
    the window from the AIF peak to the tissue peak recovers Ktrans
    independently of kep. Unlike a late-time divided-plot slope, this does not
    saturate for reversible (high-kep) tissue, where C_t/Cp plateaus at ve and
    the divided-plot slope collapses toward zero. A plasma-volume term is
    carried as a nuisance regressor so the vascular contribution does not bias
    the recovered slope.

    Parameters
    ----------
    tissue_curve : np.ndarray
        Observed tissue concentration C_t(t).
    aif : np.ndarray
        Arterial input function Cp(t).
    time_points : np.ndarray
        Time points in minutes.
    peak_idx : int
        Index of AIF peak.

    Returns
    -------
    float
        Estimated Ktrans value, or 0.1 if the slope cannot be estimated.
    """
    n = len(time_points)
    dt = float(np.mean(np.diff(time_points)))

    tissue_peak_idx = int(np.argmax(tissue_curve))
    start = peak_idx + 1
    end = min(max(tissue_peak_idx, start + 3), n - 1)
    if end - start < 3:
        return 0.1

    idx = np.arange(start, end + 1)
    cum_aif = np.cumsum(aif) * dt
    if float(np.max(np.abs(cum_aif[idx]))) < 1e-9:
        return 0.1
    design = np.column_stack([cum_aif[idx], aif[idx]])
    try:
        sol, _resid, _rank, _sv = np.linalg.lstsq(
            design, tissue_curve[idx], rcond=None
        )
    except np.linalg.LinAlgError:
        return 0.1
    slope = float(sol[0])
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

    Parameters
    ----------
    tissue_curve : np.ndarray
        Observed tissue concentration C_t(t).
    aif : np.ndarray
        Arterial input function Cp(t).
    peak_idx : int
        Index of AIF peak.

    Returns
    -------
    float
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

    Parameters
    ----------
    tissue_curve : np.ndarray
        Observed tissue concentration C_t(t).
    aif : np.ndarray
        Arterial input function Cp(t).

    Returns
    -------
    float
        Estimated vp value (unclipped — caller should clip to VP_BOUNDS),
        or 0.02 if AIF peak is near zero.
    """
    aif_peak = float(np.max(aif))
    tissue_peak = float(np.max(tissue_curve))

    if aif_peak <= 1e-6:
        return 0.02

    return float(tissue_peak / aif_peak)
