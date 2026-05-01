"""Batched DCE pharmacokinetic model evaluation.

This module provides vectorized pharmacokinetic model evaluation for
DCE-MRI analysis. The key optimization is using FFT-based convolution for
efficient batch processing across many voxels.

Supported models:
- Standard Tofts: C_t(t) = Ktrans * [Cp ⊗ exp(-kep*t)]
- Extended Tofts: C_t(t) = vp * Cp(t) + Ktrans * [Cp ⊗ exp(-kep*t)]
- Patlak: C_t(t) = vp * Cp(t) + Ktrans * ∫Cp(τ)dτ
- 2CXM: C_t(t) = [A1*exp(-K1*t) + A2*exp(-K2*t)] ⊗ Ca(t)

References:
    Tofts PS, Kermode AG. Measurement of the blood-brain barrier
    permeability and leakage space using dynamic MR imaging.
    1. Fundamental concepts. Magn Reson Med. 1991;17(2):357-367.

    Tofts PS, Brix G, Buckley DL, et al. Estimating kinetic parameters
    from dynamic contrast-enhanced T1-weighted MRI of a diffusable
    tracer: standardized quantities and symbols. J Magn Reson Imaging.
    1999;10(3):223-232.

    Sourbron SP, Buckley DL. On the scope and interpretation of the Tofts
    models for DCE-MRI. Magn Reson Med. 2011;66(3):735-745.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("nipype.interface")

from pmriprep.utils.dce_constants import (
    EPSILON_EIGENVALUE,
    EPSILON_FLOW,
    EPSILON_VE,
    MAX_KEP,
    EPSILON_VP,
    MAX_EIGENVALUE_SLOW,
)

_BATCH_SIZE = 50000


@dataclass
class ModelResult:
    """Result from batched model evaluation.

    Contains tissue concentration curves computed via vectorized
    FFT-based convolution.

    Attributes:
        tissue_curves: Computed tissue concentration curves.
            Shape: (n_voxels, n_timepoints).
        n_voxels_processed: Number of voxels processed.
        model_type: Model used ('tofts', 'extended_tofts', 'patlak', or '2cxm').
    """

    tissue_curves: npt.NDArray[np.floating]
    n_voxels_processed: int
    model_type: str


def _fft_convolve(
    aif: npt.NDArray[np.floating],
    exp_kernel: npt.NDArray[np.floating],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Perform FFT-based convolution.

    Computes the convolution: result = aif ⊗ exp_kernel
    using the convolution theorem: conv(a, b) = ifft(fft(a) * fft(b))

    Args:
        aif: Arterial input function, shape (n_timepoints,).
        exp_kernel: Exponential decay kernel, shape (n_voxels, n_timepoints).
        dt: Time step in minutes.

    Returns:
        Convolution result, shape (n_voxels, n_timepoints).
    """
    n_timepoints = aif.shape[0]
    n_voxels = exp_kernel.shape[0]

    pad_length = 2 * n_timepoints

    aif_padded = np.zeros(pad_length, dtype=np.float64)
    aif_padded[:n_timepoints] = aif

    exp_padded = np.zeros((n_voxels, pad_length), dtype=np.float64)
    exp_padded[:, :n_timepoints] = exp_kernel

    aif_fft = np.fft.fft(aif_padded)
    exp_fft = np.fft.fft(exp_padded, axis=1)

    product_fft = aif_fft[None, :] * exp_fft

    conv_result = np.fft.ifft(product_fft, axis=1).real

    return conv_result[:, :n_timepoints] * dt


def _build_exp_kernel(
    kep: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Build exponential decay kernels for all voxels.

    Computes exp(-kep * t) for each voxel.

    Args:
        kep: Efflux rate constants, shape (n_voxels,).
        time_points: Time points in minutes, shape (n_timepoints,).

    Returns:
        Exponential kernels, shape (n_voxels, n_timepoints).
    """
    lag_times = time_points - time_points[0]
    result: npt.NDArray[np.floating] = np.exp(-kep[:, None] * lag_times[None, :])
    return result


def _validate_uniform_time_points(
    time_points: npt.NDArray[np.floating],
) -> float:
    """Validate that time points are uniformly spaced and return dt.

    FFT convolution assumes uniform temporal spacing. Warns if variation
    exceeds 5% but does not error, since DCE data is almost always uniform.

    Args:
        time_points: Time points in minutes, shape (T,).

    Returns:
        Time step dt in minutes.
    """
    n = len(time_points)
    if n <= 1:
        return 1.0
    dts = np.diff(time_points)
    dt = float(dts[0])
    if dt <= 0:
        raise ValueError(
            "Time points must be strictly increasing; got first dt=%.6f min." % dt
        )
    if len(dts) > 1:
        variation = (np.max(dts) - np.min(dts)) / np.mean(dts)
        if variation > 0.05:
            logger.warning(
                "Non-uniform temporal spacing (%.1f%% variation). "
                "FFT convolution assumes uniform dt=%.4f min.",
                variation * 100,
                dt,
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
    FFT-based convolution.

    Model equation:
        C_t(t) = Ktrans * [Cp ⊗ exp(-kep*t)]

    where kep = Ktrans / ve.

    Args:
        ktrans: Volume transfer constants, shape (n_voxels,) in min^-1.
        ve: Extravascular extracellular volume fractions, shape (n_voxels,).
        aif: Arterial input function, shape (n_timepoints,) in mM.
        time_points: Time points in minutes, shape (n_timepoints,).
        batch_size: Number of voxels per batch. If None, uses default.

    Returns:
        ModelResult containing tissue concentration curves and metadata.

    Raises:
        ValueError: If ktrans and ve have different shapes.
        ValueError: If aif and time_points have different lengths.
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
        # Cap kep to match scalar tofts.fit() — degenerate voxels with
        # ve at EPSILON_VE and moderate Ktrans otherwise produce
        # exp(-kep·t) that underflows to 0 immediately and wipes the
        # convolution kernel, breaking scalar/batched parity.
        kep = np.minimum(ktrans_batch / ve_safe, MAX_KEP)

        exp_kernel = _build_exp_kernel(kep, time_points)
        conv_result = _fft_convolve(aif, exp_kernel, dt)

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
    FFT-based convolution. Includes the vascular plasma volume
    term (vp).

    Model equation:
        C_t(t) = vp * Cp(t) + Ktrans * [Cp ⊗ exp(-kep*t)]

    where kep = Ktrans / ve.

    Args:
        ktrans: Volume transfer constants, shape (n_voxels,) in min^-1.
        ve: Extravascular extracellular volume fractions, shape (n_voxels,).
        vp: Plasma volume fractions, shape (n_voxels,).
        aif: Arterial input function, shape (n_timepoints,) in mM.
        time_points: Time points in minutes, shape (n_timepoints,).
        batch_size: Number of voxels per batch. If None, uses default.

    Returns:
        ModelResult containing tissue concentration curves and metadata.

    Raises:
        ValueError: If ktrans, ve, and vp have different shapes.
        ValueError: If aif and time_points have different lengths.
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

        exp_kernel = _build_exp_kernel(kep, time_points)
        conv_result = _fft_convolve(aif, exp_kernel, dt)

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

    Args:
        ktrans: Volume transfer constants, shape (n_voxels,) in min^-1.
        vp: Plasma volume fractions, shape (n_voxels,).
        aif: Arterial input function, shape (n_timepoints,) in mM.
        time_points: Time points in minutes, shape (n_timepoints,).
        batch_size: Unused (kept for API consistency). Patlak is trivially
            vectorizable with minimal memory.

    Returns:
        ModelResult containing tissue concentration curves and metadata.

    Raises:
        ValueError: If ktrans and vp have different shapes.
        ValueError: If aif and time_points have different lengths.
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
    and FFT-based convolution.

    Model equation:
        C_t(t) = [A1*exp(-K1*t) + A2*exp(-K2*t)] ⊗ Ca(t)

    where K1, K2, A1, A2 are derived from Fp, PS, ve, vp via eigenvalue
    decomposition of the two-compartment system.

    CRITICAL: Fp and PS are in mL/min/100mL and must be converted to
    fractional rates (/100) before eigenvalue computation, matching the
    CPU model at two_compartment_exchange.py:277-278.

    Args:
        fp: Plasma flow rates, shape (n_voxels,) in mL/min/100mL.
        ps: Permeability-surface area products, shape (n_voxels,) in mL/min/100mL.
        ve: Extravascular extracellular volume fractions, shape (n_voxels,).
        vp: Plasma volume fractions, shape (n_voxels,).
        aif: Arterial input function, shape (n_timepoints,) in mM.
        time_points: Time points in minutes, shape (n_timepoints,).
        batch_size: Number of voxels per batch. If None, uses default.

    Returns:
        ModelResult containing tissue concentration curves and metadata.

    Raises:
        ValueError: If parameter arrays have different shapes.
        ValueError: If aif and time_points have different lengths.

    References:
        Sourbron SP, Buckley DL. On the scope and interpretation of the
        Tofts models for DCE-MRI. Magn Reson Med. 2011;66(3):735-745.
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
            fp_rate * (k1_final - k_ep) / denom_safe,
        )
        a2 = np.where(
            degenerate,
            np.zeros_like(fp_rate),
            fp_rate * (k_ep - k2_used) / denom_safe,
        )

        exp1 = _build_exp_kernel(k1_final, time_points)
        exp2 = _build_exp_kernel(k2_used, time_points)
        conv1 = _fft_convolve(aif, exp1, dt)
        conv2 = _fft_convolve(aif, exp2, dt)

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

    Args:
        model_name: Model to evaluate ('tofts', 'extended_tofts', 'patlak',
            or '2cxm').
        params: Dictionary of parameter arrays:
            - For 'tofts': {'ktrans': ..., 've': ...}
            - For 'extended_tofts': {'ktrans': ..., 've': ..., 'vp': ...}
            - For 'patlak': {'ktrans': ..., 'vp': ...}
            - For '2cxm': {'fp': ..., 'ps': ..., 've': ..., 'vp': ...}
        aif: Arterial input function, shape (n_timepoints,) in mM.
        time_points: Time points in minutes, shape (n_timepoints,).
        batch_size: Number of voxels per batch. Auto-determined if None.

    Returns:
        ModelResult containing tissue concentration curves.

    Raises:
        ValueError: If model_name is not recognized.
        KeyError: If required parameters are missing from params dict.
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
