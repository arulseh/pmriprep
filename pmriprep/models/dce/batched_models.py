"""Batched DCE pharmacokinetic model evaluation.

This module provides vectorized pharmacokinetic model evaluation for
DCE-MRI analysis. The key optimization is the exact analytic (piecewise-linear
AIF) convolution vectorized across many voxels.

Supported models:
- Standard Tofts: C_t(t) = Ktrans * [Cp ⊗ exp(-kep*t)]
- Extended Tofts: C_t(t) = vp * Cp(t) + Ktrans * [Cp ⊗ exp(-kep*t)]
- Patlak: C_t(t) = vp * Cp(t) + Ktrans * ∫Cp(τ)dτ
- 2CXM: C_t(t) = [A1*exp(-K1*t) + A2*exp(-K2*t)] ⊗ Ca(t)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from pmriprep.models.dce._convolution import analytic_exp_convolution
from pmriprep.utils.dce_constants import (
    EPSILON_EIGENVALUE,
    EPSILON_FLOW,
    EPSILON_VE,
    MAX_KEP,
    EPSILON_VP,
    MAX_EIGENVALUE_SLOW,
)

logger = logging.getLogger("nipype.interface")

_BATCH_SIZE = 50000


@dataclass
class ModelResult:
    """Result from batched model evaluation.

    Contains tissue concentration curves computed via vectorized
    exact analytic convolution.

    Attributes
    ----------
    tissue_curves : np.ndarray
        Computed tissue concentration curves.
        Shape: (n_voxels, n_timepoints).
    n_voxels_processed : int
        Number of voxels processed.
    model_type : str
        Model used ('tofts', 'extended_tofts', 'patlak', or '2cxm').
    """

    tissue_curves: npt.NDArray[np.floating]
    n_voxels_processed: int
    model_type: str


def _validate_uniform_time_points(
    time_points: npt.NDArray[np.floating],
) -> float:
    """Validate that time points are uniformly spaced and return dt.

    The analytic convolution assumes uniform temporal spacing, so unsupported
    axes fail rather than being evaluated with the first interval.

    Parameters
    ----------
    time_points : np.ndarray
        Time points in minutes, shape (T,).

    Returns
    -------
    float
        Time step dt in minutes.
    """
    time_points = np.asarray(time_points, dtype=np.float64)
    if time_points.ndim != 1:
        raise ValueError(
            f"Time points must be one-dimensional; got shape {time_points.shape}."
        )
    if not np.all(np.isfinite(time_points)):
        raise ValueError("Time points must contain only finite values.")

    n = len(time_points)
    if n <= 1:
        return 1.0
    dts = np.diff(time_points)
    dt = float(dts[0])
    if np.any(dts <= 0.0):
        raise ValueError("Time points must be strictly increasing.")
    if not np.allclose(dts, dt, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "Nonuniform time points are unsupported by the analytic "
            "convolution models."
        )
    return dt


def batched_tofts_model(
    ktrans: npt.NDArray[np.floating],
    ve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    batch_size: Optional[int] = None,
) -> ModelResult:
    """Batched Standard Tofts model evaluation.

    Computes tissue concentration curves for multiple voxels using
    exact analytic convolution.

    Model equation:
        C_t(t) = Ktrans * [Cp ⊗ exp(-kep*t)]

    where kep = Ktrans / ve.

    Parameters
    ----------
    ktrans : np.ndarray
        Volume transfer constants, shape (n_voxels,) in min^-1.
    ve : np.ndarray
        Extravascular extracellular volume fractions, shape (n_voxels,).
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,) in mM.
    time_points : np.ndarray
        Time points in minutes, shape (n_timepoints,).
    batch_size : int, optional
        Number of voxels per batch. If None, uses default.

    Returns
    -------
    ModelResult
        ModelResult containing tissue concentration curves and metadata.

    Raises
    ------
    ValueError
        If ktrans and ve have different shapes.
    ValueError
        If aif and time_points have different lengths.
    """
    if ktrans.shape != ve.shape:
        raise ValueError(
            f"ktrans shape ({ktrans.shape}) must match ve shape ({ve.shape})"
        )

    if len(aif) != len(time_points):
        raise ValueError(
            f"AIF length ({len(aif)}) must match time_points length ({len(time_points)})"
        )

    if len(aif) == 0:
        raise ValueError("AIF cannot be empty")

    n_voxels = len(ktrans)
    n_timepoints = len(aif)

    if n_voxels == 0:
        return ModelResult(
            tissue_curves=np.zeros((0, n_timepoints), dtype=np.float64),
            n_voxels_processed=0,
            model_type="tofts",
        )

    dt = _validate_uniform_time_points(time_points)

    if batch_size is None:
        batch_size = _BATCH_SIZE

    tissue_curves = np.zeros((n_voxels, n_timepoints), dtype=np.float64)

    for batch_start in range(0, n_voxels, batch_size):
        batch_end = min(batch_start + batch_size, n_voxels)

        ktrans_batch = ktrans[batch_start:batch_end]
        ve_batch = ve[batch_start:batch_end]

        ve_safe = np.maximum(ve_batch, EPSILON_VE)
        # Cap kep to match the scalar path (tofts.evaluate also caps at
        # MAX_KEP), so the analytic convolution gives identical scalar/batched
        # output at the ve->0 degenerate limit; also a physical bound.
        kep = np.minimum(ktrans_batch / ve_safe, MAX_KEP)

        conv_result = analytic_exp_convolution(aif, kep, dt)

        tissue_curves[batch_start:batch_end, :] = ktrans_batch[:, None] * conv_result

    return ModelResult(
        tissue_curves=tissue_curves,
        n_voxels_processed=n_voxels,
        model_type="tofts",
    )


def batched_extended_tofts_model(
    ktrans: npt.NDArray[np.floating],
    ve: npt.NDArray[np.floating],
    vp: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    batch_size: Optional[int] = None,
) -> ModelResult:
    """Batched Extended Tofts model evaluation.

    Computes tissue concentration curves for multiple voxels using
    exact analytic convolution. Includes the vascular plasma volume
    term (vp).

    Model equation:
        C_t(t) = vp * Cp(t) + Ktrans * [Cp ⊗ exp(-kep*t)]

    where kep = Ktrans / ve.

    Parameters
    ----------
    ktrans : np.ndarray
        Volume transfer constants, shape (n_voxels,) in min^-1.
    ve : np.ndarray
        Extravascular extracellular volume fractions, shape (n_voxels,).
    vp : np.ndarray
        Plasma volume fractions, shape (n_voxels,).
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,) in mM.
    time_points : np.ndarray
        Time points in minutes, shape (n_timepoints,).
    batch_size : int, optional
        Number of voxels per batch. If None, uses default.

    Returns
    -------
    ModelResult
        ModelResult containing tissue concentration curves and metadata.

    Raises
    ------
    ValueError
        If ktrans, ve, and vp have different shapes.
    ValueError
        If aif and time_points have different lengths.
    """
    if ktrans.shape != ve.shape or ktrans.shape != vp.shape:
        raise ValueError(
            f"ktrans shape ({ktrans.shape}), ve shape ({ve.shape}), and "
            f"vp shape ({vp.shape}) must all match"
        )

    if len(aif) != len(time_points):
        raise ValueError(
            f"AIF length ({len(aif)}) must match time_points length ({len(time_points)})"
        )

    if len(aif) == 0:
        raise ValueError("AIF cannot be empty")

    n_voxels = len(ktrans)
    n_timepoints = len(aif)

    if n_voxels == 0:
        return ModelResult(
            tissue_curves=np.zeros((0, n_timepoints), dtype=np.float64),
            n_voxels_processed=0,
            model_type="extended_tofts",
        )

    dt = _validate_uniform_time_points(time_points)

    if batch_size is None:
        batch_size = _BATCH_SIZE

    tissue_curves = np.zeros((n_voxels, n_timepoints), dtype=np.float64)

    for batch_start in range(0, n_voxels, batch_size):
        batch_end = min(batch_start + batch_size, n_voxels)

        ktrans_batch = ktrans[batch_start:batch_end]
        ve_batch = ve[batch_start:batch_end]
        vp_batch = vp[batch_start:batch_end]

        vascular_term = vp_batch[:, None] * aif[None, :]

        ve_safe = np.maximum(ve_batch, EPSILON_VE)
        # Cap kep — see tofts batch above for rationale.
        kep = np.minimum(ktrans_batch / ve_safe, MAX_KEP)

        conv_result = analytic_exp_convolution(aif, kep, dt)

        extravasation_term = ktrans_batch[:, None] * conv_result

        tissue_curves[batch_start:batch_end, :] = vascular_term + extravasation_term

    return ModelResult(
        tissue_curves=tissue_curves,
        n_voxels_processed=n_voxels,
        model_type="extended_tofts",
    )


def batched_patlak_model(
    ktrans: npt.NDArray[np.floating],
    vp: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    batch_size: Optional[int] = None,
) -> ModelResult:
    """Batched Patlak model evaluation.

    Computes tissue concentration curves for multiple voxels using
    the Patlak model (no backflux assumption).

    Model equation:
        C_t(t) = vp * Cp(t) + Ktrans * ∫₀ᵗ Cp(τ)dτ

    No convolution kernel needed — uses cumulative trapezoidal integration.

    Parameters
    ----------
    ktrans : np.ndarray
        Volume transfer constants, shape (n_voxels,) in min^-1.
    vp : np.ndarray
        Plasma volume fractions, shape (n_voxels,).
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,) in mM.
    time_points : np.ndarray
        Time points in minutes, shape (n_timepoints,).
    batch_size : int, optional
        Unused (kept for API consistency). Patlak is trivially
        vectorizable with minimal memory.

    Returns
    -------
    ModelResult
        ModelResult containing tissue concentration curves and metadata.

    Raises
    ------
    ValueError
        If ktrans and vp have different shapes.
    ValueError
        If aif and time_points have different lengths.
    """
    if ktrans.shape != vp.shape:
        raise ValueError(
            f"ktrans shape ({ktrans.shape}) must match vp shape ({vp.shape})"
        )

    if len(aif) != len(time_points):
        raise ValueError(
            f"AIF length ({len(aif)}) must match time_points length ({len(time_points)})"
        )

    if len(aif) == 0:
        raise ValueError("AIF cannot be empty")

    n_voxels = len(ktrans)
    n_timepoints = len(aif)

    if n_voxels == 0:
        return ModelResult(
            tissue_curves=np.zeros((0, n_timepoints), dtype=np.float64),
            n_voxels_processed=0,
            model_type="patlak",
        )

    dts = np.diff(time_points)
    aif_integral = np.zeros_like(aif)
    aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dts)

    tissue = vp[:, None] * aif[None, :] + ktrans[:, None] * aif_integral[None, :]

    return ModelResult(
        tissue_curves=tissue.astype(np.float64),
        n_voxels_processed=n_voxels,
        model_type="patlak",
    )


def batched_2cxm_model(
    fp: npt.NDArray[np.floating],
    ps: npt.NDArray[np.floating],
    ve: npt.NDArray[np.floating],
    vp: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    batch_size: Optional[int] = None,
) -> ModelResult:
    """Batched Two-Compartment Exchange Model (2CXM) evaluation.

    Computes tissue concentration curves using eigenvalue decomposition
    and exact analytic convolution.

    Model equation:
        C_t(t) = [A1*exp(-K1*t) + A2*exp(-K2*t)] ⊗ Ca(t)

    where K1, K2, A1, A2 are derived from Fp, PS, ve, vp via eigenvalue
    decomposition of the two-compartment system.

    CRITICAL: Fp and PS are in mL/min/100mL and must be converted to
    fractional rates (/100) before eigenvalue computation, matching the
    CPU model at two_compartment_exchange.py:277-278.

    Parameters
    ----------
    fp : np.ndarray
        Plasma flow rates, shape (n_voxels,) in mL/min/100mL.
    ps : np.ndarray
        Permeability-surface area products, shape (n_voxels,) in mL/min/100mL.
    ve : np.ndarray
        Extravascular extracellular volume fractions, shape (n_voxels,).
    vp : np.ndarray
        Plasma volume fractions, shape (n_voxels,).
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,) in mM.
    time_points : np.ndarray
        Time points in minutes, shape (n_timepoints,).
    batch_size : int, optional
        Number of voxels per batch. If None, uses default.

    Returns
    -------
    ModelResult
        ModelResult containing tissue concentration curves and metadata.

    Raises
    ------
    ValueError
        If parameter arrays have different shapes.
    ValueError
        If aif and time_points have different lengths.

    """
    if not (fp.shape == ps.shape == ve.shape == vp.shape):
        raise ValueError(
            f"All parameter shapes must match: fp={fp.shape}, ps={ps.shape}, "
            f"ve={ve.shape}, vp={vp.shape}"
        )

    if len(aif) != len(time_points):
        raise ValueError(
            f"AIF length ({len(aif)}) must match time_points length ({len(time_points)})"
        )

    if len(aif) == 0:
        raise ValueError("AIF cannot be empty")

    n_voxels = len(fp)
    n_timepoints = len(aif)

    if n_voxels == 0:
        return ModelResult(
            tissue_curves=np.zeros((0, n_timepoints), dtype=np.float64),
            n_voxels_processed=0,
            model_type="2cxm",
        )

    dt = _validate_uniform_time_points(time_points)

    if batch_size is None:
        batch_size = 25000

    tissue_curves = np.zeros((n_voxels, n_timepoints), dtype=np.float64)

    for batch_start in range(0, n_voxels, batch_size):
        batch_end = min(batch_start + batch_size, n_voxels)

        fp_d = fp[batch_start:batch_end]
        ps_d = ps[batch_start:batch_end]
        ve_d = ve[batch_start:batch_end]
        vp_d = vp[batch_start:batch_end]

        fp_rate = np.maximum(fp_d, EPSILON_FLOW) / 100.0
        ps_rate = np.maximum(ps_d, 0.0) / 100.0

        vp_safe = np.maximum(vp_d, EPSILON_VP)
        ve_safe = np.maximum(ve_d, EPSILON_VE)

        k_pe = ps_rate / vp_safe
        k_ep = ps_rate / ve_safe
        k_out = fp_rate / vp_safe

        trace = -(k_out + k_pe + k_ep)
        det = k_out * k_ep

        discriminant = np.maximum(trace ** 2 - 4 * det, 0.0)
        sqrt_disc = np.sqrt(discriminant)

        k1 = (-trace + sqrt_disc) / 2
        k2_raw = (-trace - sqrt_disc) / 2

        k1_final = np.maximum(k1, k2_raw)
        k2_final = np.minimum(k1, k2_raw)

        k1_final = np.maximum(k1_final, EPSILON_EIGENVALUE)
        k2_final = np.clip(k2_final, EPSILON_EIGENVALUE, MAX_EIGENVALUE_SLOW)

        denom = k1_final - k2_final
        degenerate = denom < EPSILON_EIGENVALUE
        k2_used = np.where(degenerate, k1_final, k2_final)
        denom_safe = np.maximum(denom, EPSILON_EIGENVALUE)

        a1 = np.where(
            degenerate,
            fp_rate,
            fp_rate * (k1_final - k_ep - k_pe) / denom_safe,
        )
        a2 = np.where(
            degenerate,
            np.zeros_like(fp_rate),
            fp_rate * (k_ep + k_pe - k2_used) / denom_safe,
        )

        conv1 = analytic_exp_convolution(aif, k1_final, dt)
        conv2 = analytic_exp_convolution(aif, k2_used, dt)

        tissue_curves[batch_start:batch_end, :] = a1[:, None] * conv1 + a2[:, None] * conv2

    return ModelResult(
        tissue_curves=tissue_curves,
        n_voxels_processed=n_voxels,
        model_type="2cxm",
    )


def evaluate_model(
    model_name: str,
    params: dict[str, npt.NDArray[np.floating]],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    batch_size: Optional[int] = None,
) -> ModelResult:
    """Evaluate a DCE model using the appropriate batched function.

    This is a convenience wrapper that dispatches to the appropriate
    batched model evaluation function based on the model name.

    Parameters
    ----------
    model_name : str
        Model to evaluate ('tofts', 'extended_tofts', 'patlak',
        or '2cxm').
    params : dict
        Dictionary of parameter arrays:
            - For 'tofts': {'ktrans': ..., 've': ...}
            - For 'extended_tofts': {'ktrans': ..., 've': ..., 'vp': ...}
            - For 'patlak': {'ktrans': ..., 'vp': ...}
            - For '2cxm': {'fp': ..., 'ps': ..., 've': ..., 'vp': ...}
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,) in mM.
    time_points : np.ndarray
        Time points in minutes, shape (n_timepoints,).
    batch_size : int, optional
        Number of voxels per batch. Auto-determined if None.

    Returns
    -------
    ModelResult
        ModelResult containing tissue concentration curves.

    Raises
    ------
    ValueError
        If model_name is not recognized.
    KeyError
        If required parameters are missing from params dict.
    """
    if model_name == "tofts":
        return batched_tofts_model(
            ktrans=params["ktrans"],
            ve=params["ve"],
            aif=aif,
            time_points=time_points,
            batch_size=batch_size,
        )
    elif model_name == "extended_tofts":
        return batched_extended_tofts_model(
            ktrans=params["ktrans"],
            ve=params["ve"],
            vp=params["vp"],
            aif=aif,
            time_points=time_points,
            batch_size=batch_size,
        )
    elif model_name == "patlak":
        return batched_patlak_model(
            ktrans=params["ktrans"],
            vp=params["vp"],
            aif=aif,
            time_points=time_points,
            batch_size=batch_size,
        )
    elif model_name == "2cxm":
        return batched_2cxm_model(
            fp=params["fp"],
            ps=params["ps"],
            ve=params["ve"],
            vp=params["vp"],
            aif=aif,
            time_points=time_points,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: "
            "'tofts', 'extended_tofts', 'patlak', '2cxm'"
        )
