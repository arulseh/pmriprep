"""GPU-accelerated batched DCE pharmacokinetic model evaluation.

This module provides GPU-accelerated pharmacokinetic model evaluation for
DCE-MRI analysis. The key optimization is using FFT-based convolution for
efficient batch processing across many voxels.

Performance characteristics:
- 20-100x speedup for >5000 voxels compared to CPU batch processing
- Memory efficient: processes voxels in batches based on available GPU memory
- Graceful fallback: automatically uses CPU if CuPy is unavailable

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
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("nipype.interface")

from pmriprep.utils.dce_constants import (
    EPSILON_EIGENVALUE,
    EPSILON_FLOW,
    EPSILON_VE,
    EPSILON_VP,
    MAX_EIGENVALUE_SLOW,
)
from pmriprep.utils.gpu import (
    ArrayLike,
    clear_gpu_memory,
    get_array_module,
    get_gpu_memory_info,
    is_cupy_available,
    synchronize,
    to_device,
    to_host,
)


@dataclass
class GPUModelResult:
    """Result from GPU batched model evaluation.

    Contains tissue concentration curves computed via GPU-accelerated
    FFT-based convolution.

    Attributes:
        tissue_curves: Computed tissue concentration curves.
            Shape: (n_voxels, n_timepoints).
        n_voxels_processed: Number of voxels processed on GPU.
        used_gpu: Whether GPU was actually used (False if fallback to CPU).
        model_type: Model used ('tofts', 'extended_tofts', 'patlak', or '2cxm').
    """

    tissue_curves: npt.NDArray[np.floating]
    n_voxels_processed: int
    used_gpu: bool
    model_type: str


# Backward compatibility alias
GPUToftsResult = GPUModelResult


def _fft_convolve_gpu(
    aif: ArrayLike,
    exp_kernel: ArrayLike,
    dt: float,
    xp: Any,
) -> ArrayLike:
    """Perform FFT-based convolution on GPU.

    Computes the convolution: result = aif ⊗ exp_kernel
    using the convolution theorem: conv(a, b) = ifft(fft(a) * fft(b))

    Args:
        aif: Arterial input function, shape (n_timepoints,).
        exp_kernel: Exponential decay kernel, shape (n_voxels, n_timepoints).
        dt: Time step in minutes.
        xp: Array module (numpy or cupy).

    Returns:
        Convolution result, shape (n_voxels, n_timepoints).
    """
    n_timepoints = aif.shape[0]
    n_voxels = exp_kernel.shape[0]

    # Zero-pad for linear convolution
    # Full convolution length = n + m - 1, but we want same length output
    # Use 2*n for simplicity and efficiency (power of 2)
    pad_length = 2 * n_timepoints

    # Pad AIF (same for all voxels)
    aif_padded = xp.zeros(pad_length, dtype=xp.float64)
    aif_padded[:n_timepoints] = aif

    # Pad exponential kernels (different for each voxel)
    exp_padded = xp.zeros((n_voxels, pad_length), dtype=xp.float64)
    exp_padded[:, :n_timepoints] = exp_kernel

    # FFT of AIF (computed once, broadcast to all voxels)
    aif_fft = xp.fft.fft(aif_padded)

    # FFT of exponential kernels (batched)
    exp_fft = xp.fft.fft(exp_padded, axis=1)

    # Multiply in frequency domain
    product_fft = aif_fft[None, :] * exp_fft

    # Inverse FFT
    conv_result = xp.fft.ifft(product_fft, axis=1).real

    # Extract the valid portion (first n_timepoints) and scale by dt
    return conv_result[:, :n_timepoints] * dt


def _build_exp_kernel_gpu(
    kep: ArrayLike,
    time_points: ArrayLike,
    xp: Any,
) -> ArrayLike:
    """Build exponential decay kernels for all voxels.

    Computes exp(-kep * t) for each voxel.

    Args:
        kep: Efflux rate constants, shape (n_voxels,).
        time_points: Time points in minutes, shape (n_timepoints,).
        xp: Array module (numpy or cupy).

    Returns:
        Exponential kernels, shape (n_voxels, n_timepoints).
    """
    # Broadcasting: (n_voxels, 1) * (1, n_timepoints) -> (n_voxels, n_timepoints)
    # Use zero-based lag times: h(tau) = exp(-kep * tau) where tau starts at 0
    lag_times = time_points - time_points[0]
    return xp.exp(-kep[:, None] * lag_times[None, :])


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


def batched_tofts_model_gpu(
    ktrans: npt.NDArray[np.floating],
    ve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
) -> GPUModelResult:
    """GPU-accelerated batched Standard Tofts model evaluation.

    Computes tissue concentration curves for multiple voxels using
    FFT-based convolution on GPU.

    Model equation:
        C_t(t) = Ktrans * [Cp ⊗ exp(-kep*t)]

    where kep = Ktrans / ve.

    Args:
        ktrans: Volume transfer constants, shape (n_voxels,) in min^-1.
        ve: Extravascular extracellular volume fractions, shape (n_voxels,).
        aif: Arterial input function, shape (n_timepoints,) in mM.
        time_points: Time points in minutes, shape (n_timepoints,).
        use_gpu: Whether to use GPU if available. Default True.
        batch_size: Number of voxels per GPU batch. If None, auto-determined
            based on available GPU memory. Default None.

    Returns:
        GPUModelResult containing tissue concentration curves and metadata.

    Raises:
        ValueError: If ktrans and ve have different shapes.
        ValueError: If aif and time_points have different lengths.
    """
    # Validate inputs
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

    # Handle edge case
    if n_voxels == 0:
        return GPUModelResult(
            tissue_curves=np.zeros((0, n_timepoints), dtype=np.float64),
            n_voxels_processed=0,
            used_gpu=False,
            model_type="tofts",
        )

    # Determine if we can use GPU
    use_gpu_actual = use_gpu and is_cupy_available()

    # Get array module
    if use_gpu_actual:
        xp = get_array_module(prefer_gpu=True)
    else:
        xp = np

    dt = _validate_uniform_time_points(time_points)

    # Determine batch size based on GPU memory
    if batch_size is None:
        if use_gpu_actual:
            batch_size = _get_optimal_batch_size(n_timepoints)
        else:
            batch_size = 50000  # CPU batch size

    # Initialize output array (always on CPU for final output)
    tissue_curves = np.zeros((n_voxels, n_timepoints), dtype=np.float64)

    # Transfer AIF and time points to device (shared across batches)
    aif_device = to_device(aif, "gpu" if use_gpu_actual else "cpu")
    time_device = to_device(time_points, "gpu" if use_gpu_actual else "cpu")

    # Process voxels in batches
    for batch_start in range(0, n_voxels, batch_size):
        batch_end = min(batch_start + batch_size, n_voxels)

        # Get parameters for this batch
        ktrans_batch = ktrans[batch_start:batch_end]
        ve_batch = ve[batch_start:batch_end]

        # Transfer to device
        ktrans_device = to_device(ktrans_batch, "gpu" if use_gpu_actual else "cpu")
        ve_device = to_device(ve_batch, "gpu" if use_gpu_actual else "cpu")

        # Compute kep = Ktrans / ve (with epsilon guard)
        ve_safe = xp.maximum(ve_device, EPSILON_VE)
        kep = ktrans_device / ve_safe

        # Build exponential kernels
        exp_kernel = _build_exp_kernel_gpu(kep, time_device, xp)

        # FFT-based convolution
        conv_result = _fft_convolve_gpu(aif_device, exp_kernel, dt, xp)

        # Apply Ktrans scaling: C_t = Ktrans * convolution
        tissue_batch = ktrans_device[:, None] * conv_result

        # Transfer result to CPU
        tissue_curves[batch_start:batch_end, :] = to_host(tissue_batch)

        # Clear GPU memory between batches
        if use_gpu_actual:
            synchronize()
            clear_gpu_memory()

    return GPUModelResult(
        tissue_curves=tissue_curves,
        n_voxels_processed=n_voxels,
        used_gpu=use_gpu_actual,
        model_type="tofts",
    )


def batched_extended_tofts_model_gpu(
    ktrans: npt.NDArray[np.floating],
    ve: npt.NDArray[np.floating],
    vp: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
) -> GPUModelResult:
    """GPU-accelerated batched Extended Tofts model evaluation.

    Computes tissue concentration curves for multiple voxels using
    FFT-based convolution on GPU. Includes the vascular plasma volume
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
        use_gpu: Whether to use GPU if available. Default True.
        batch_size: Number of voxels per GPU batch. If None, auto-determined
            based on available GPU memory. Default None.

    Returns:
        GPUModelResult containing tissue concentration curves and metadata.

    Raises:
        ValueError: If ktrans, ve, and vp have different shapes.
        ValueError: If aif and time_points have different lengths.
    """
    # Validate inputs
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

    # Handle edge case
    if n_voxels == 0:
        return GPUModelResult(
            tissue_curves=np.zeros((0, n_timepoints), dtype=np.float64),
            n_voxels_processed=0,
            used_gpu=False,
            model_type="extended_tofts",
        )

    # Determine if we can use GPU
    use_gpu_actual = use_gpu and is_cupy_available()

    # Get array module
    if use_gpu_actual:
        xp = get_array_module(prefer_gpu=True)
    else:
        xp = np

    dt = _validate_uniform_time_points(time_points)

    # Determine batch size based on GPU memory
    if batch_size is None:
        if use_gpu_actual:
            batch_size = _get_optimal_batch_size(n_timepoints)
        else:
            batch_size = 50000  # CPU batch size

    # Initialize output array (always on CPU for final output)
    tissue_curves = np.zeros((n_voxels, n_timepoints), dtype=np.float64)

    # Transfer AIF and time points to device (shared across batches)
    aif_device = to_device(aif, "gpu" if use_gpu_actual else "cpu")
    time_device = to_device(time_points, "gpu" if use_gpu_actual else "cpu")

    # Process voxels in batches
    for batch_start in range(0, n_voxels, batch_size):
        batch_end = min(batch_start + batch_size, n_voxels)

        # Get parameters for this batch
        ktrans_batch = ktrans[batch_start:batch_end]
        ve_batch = ve[batch_start:batch_end]
        vp_batch = vp[batch_start:batch_end]

        # Transfer to device
        ktrans_device = to_device(ktrans_batch, "gpu" if use_gpu_actual else "cpu")
        ve_device = to_device(ve_batch, "gpu" if use_gpu_actual else "cpu")
        vp_device = to_device(vp_batch, "gpu" if use_gpu_actual else "cpu")

        # Compute vascular contribution: vp * Cp(t)
        vascular_term = vp_device[:, None] * aif_device[None, :]

        # Compute kep = Ktrans / ve (with epsilon guard)
        ve_safe = xp.maximum(ve_device, EPSILON_VE)
        kep = ktrans_device / ve_safe

        # Build exponential kernels
        exp_kernel = _build_exp_kernel_gpu(kep, time_device, xp)

        # FFT-based convolution
        conv_result = _fft_convolve_gpu(aif_device, exp_kernel, dt, xp)

        # Apply Ktrans scaling: extravasation = Ktrans * convolution
        extravasation_term = ktrans_device[:, None] * conv_result

        # Sum contributions: C_t = vascular + extravasation
        tissue_batch = vascular_term + extravasation_term

        # Transfer result to CPU
        tissue_curves[batch_start:batch_end, :] = to_host(tissue_batch)

        # Clear GPU memory between batches
        if use_gpu_actual:
            synchronize()
            clear_gpu_memory()

    return GPUModelResult(
        tissue_curves=tissue_curves,
        n_voxels_processed=n_voxels,
        used_gpu=use_gpu_actual,
        model_type="extended_tofts",
    )


def batched_patlak_model_gpu(
    ktrans: npt.NDArray[np.floating],
    vp: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
) -> GPUModelResult:
    """GPU-accelerated batched Patlak model evaluation.

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
        use_gpu: Whether to use GPU if available. Default True.
        batch_size: Unused (kept for API consistency). Patlak is trivially
            vectorizable with minimal memory.

    Returns:
        GPUModelResult containing tissue concentration curves and metadata.

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
        return GPUModelResult(
            tissue_curves=np.zeros((0, n_timepoints), dtype=np.float64),
            n_voxels_processed=0,
            used_gpu=False,
            model_type="patlak",
        )

    use_gpu_actual = use_gpu and is_cupy_available()

    if use_gpu_actual:
        xp = get_array_module(prefer_gpu=True)
    else:
        xp = np

    # Compute AIF cumulative trapezoidal integral (shared across voxels)
    aif_arr = to_device(aif, "gpu" if use_gpu_actual else "cpu")
    tp_arr = to_device(time_points, "gpu" if use_gpu_actual else "cpu")

    dts = xp.diff(tp_arr)
    aif_integral = xp.zeros_like(aif_arr)
    aif_integral[1:] = xp.cumsum(0.5 * (aif_arr[:-1] + aif_arr[1:]) * dts)

    # Vectorized: C_t = vp * Cp + Ktrans * ∫Cp
    ktrans_d = to_device(ktrans, "gpu" if use_gpu_actual else "cpu")
    vp_d = to_device(vp, "gpu" if use_gpu_actual else "cpu")

    tissue = vp_d[:, None] * aif_arr[None, :] + ktrans_d[:, None] * aif_integral[None, :]

    tissue_curves = to_host(tissue).astype(np.float64)

    if use_gpu_actual:
        synchronize()
        clear_gpu_memory()

    return GPUModelResult(
        tissue_curves=tissue_curves,
        n_voxels_processed=n_voxels,
        used_gpu=use_gpu_actual,
        model_type="patlak",
    )


def batched_2cxm_model_gpu(
    fp: npt.NDArray[np.floating],
    ps: npt.NDArray[np.floating],
    ve: npt.NDArray[np.floating],
    vp: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
) -> GPUModelResult:
    """GPU-accelerated batched Two-Compartment Exchange Model (2CXM) evaluation.

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
        use_gpu: Whether to use GPU if available. Default True.
        batch_size: Number of voxels per GPU batch. If None, auto-determined.

    Returns:
        GPUModelResult containing tissue concentration curves and metadata.

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
        return GPUModelResult(
            tissue_curves=np.zeros((0, n_timepoints), dtype=np.float64),
            n_voxels_processed=0,
            used_gpu=False,
            model_type="2cxm",
        )

    use_gpu_actual = use_gpu and is_cupy_available()

    if use_gpu_actual:
        xp = get_array_module(prefer_gpu=True)
    else:
        xp = np

    dt = _validate_uniform_time_points(time_points)

    # 2CXM needs ~2x memory of Tofts (two exponential kernels, two FFTs)
    if batch_size is None:
        if use_gpu_actual:
            batch_size = _get_optimal_batch_size(n_timepoints, memory_multiplier=2.0)
        else:
            batch_size = 25000

    tissue_curves = np.zeros((n_voxels, n_timepoints), dtype=np.float64)

    aif_device = to_device(aif, "gpu" if use_gpu_actual else "cpu")
    time_device = to_device(time_points, "gpu" if use_gpu_actual else "cpu")

    for batch_start in range(0, n_voxels, batch_size):
        batch_end = min(batch_start + batch_size, n_voxels)

        fp_d = to_device(fp[batch_start:batch_end], "gpu" if use_gpu_actual else "cpu")
        ps_d = to_device(ps[batch_start:batch_end], "gpu" if use_gpu_actual else "cpu")
        ve_d = to_device(ve[batch_start:batch_end], "gpu" if use_gpu_actual else "cpu")
        vp_d = to_device(vp[batch_start:batch_end], "gpu" if use_gpu_actual else "cpu")

        # Unit conversion: mL/min/100mL → fractional rate (min⁻¹)
        fp_rate = xp.maximum(fp_d, EPSILON_FLOW) / 100.0
        ps_rate = xp.maximum(ps_d, 0.0) / 100.0

        # Epsilon guards on volume fractions
        vp_safe = xp.maximum(vp_d, EPSILON_VP)
        ve_safe = xp.maximum(ve_d, EPSILON_VE)

        # Rate constants for the system matrix
        k_pe = ps_rate / vp_safe  # plasma → EES transfer rate
        k_ep = ps_rate / ve_safe  # EES → plasma transfer rate
        k_out = fp_rate / vp_safe  # plasma outflow rate

        # Eigenvalues from trace/determinant of system matrix
        trace = -(k_out + k_pe + k_ep)
        det = k_out * k_ep

        discriminant = xp.maximum(trace ** 2 - 4 * det, 0.0)
        sqrt_disc = xp.sqrt(discriminant)

        k1 = (-trace + sqrt_disc) / 2
        k2_raw = (-trace - sqrt_disc) / 2

        # Ensure k1 >= k2
        k1_final = xp.maximum(k1, k2_raw)
        k2_final = xp.minimum(k1, k2_raw)

        # Guard eigenvalues
        k1_final = xp.maximum(k1_final, EPSILON_EIGENVALUE)
        k2_final = xp.clip(k2_final, EPSILON_EIGENVALUE, MAX_EIGENVALUE_SLOW)

        # Degenerate case: k1 ≈ k2 → single exponential (matches CPU line 327-331)
        denom = k1_final - k2_final
        degenerate = denom < EPSILON_EIGENVALUE
        k2_used = xp.where(degenerate, k1_final, k2_final)
        denom_safe = xp.maximum(denom, EPSILON_EIGENVALUE)

        # Coefficients — use fp_rate, NOT fp (units!)
        a1 = xp.where(
            degenerate,
            fp_rate,
            fp_rate * (k1_final - k_ep) / denom_safe,
        )
        a2 = xp.where(
            degenerate,
            xp.zeros_like(fp_rate),
            fp_rate * (k_ep - k2_used) / denom_safe,
        )

        # Two FFT convolutions
        exp1 = _build_exp_kernel_gpu(k1_final, time_device, xp)
        exp2 = _build_exp_kernel_gpu(k2_used, time_device, xp)
        conv1 = _fft_convolve_gpu(aif_device, exp1, dt, xp)
        conv2 = _fft_convolve_gpu(aif_device, exp2, dt, xp)

        tissue_batch = a1[:, None] * conv1 + a2[:, None] * conv2

        tissue_curves[batch_start:batch_end, :] = to_host(tissue_batch)

        if use_gpu_actual:
            synchronize()
            clear_gpu_memory()

    return GPUModelResult(
        tissue_curves=tissue_curves,
        n_voxels_processed=n_voxels,
        used_gpu=use_gpu_actual,
        model_type="2cxm",
    )


def _get_optimal_batch_size(
    n_timepoints: int,
    safety_factor: float = 0.5,
    memory_multiplier: float = 1.0,
) -> int:
    """Estimate optimal batch size based on GPU memory.

    Args:
        n_timepoints: Number of time points per curve.
        safety_factor: Fraction of free GPU memory to use. Default 0.5.
        memory_multiplier: Model-specific memory multiplier. 2CXM needs ~2x
            due to two exponential kernels and two FFTs. Default 1.0.

    Returns:
        Recommended batch size for GPU processing.
        Returns 50000 if GPU unavailable (CPU batch size).
    """
    if not is_cupy_available():
        return 50000

    mem_info = get_gpu_memory_info()
    free_memory = mem_info["free"]

    # Memory per voxel estimate for FFT-based convolution:
    # - Input parameters: 3-5 doubles = ~40 bytes
    # - Exponential kernel: 2 * n_timepoints doubles (padded) = 16 * n bytes
    # - FFT workspace: ~4 * n_timepoints complex doubles = 64 * n bytes
    # - Output: n_timepoints doubles = 8 * n bytes
    # Total per voxel: ~90 * n_timepoints bytes
    bytes_per_voxel = int(90 * n_timepoints * memory_multiplier)

    usable_memory = int(free_memory * safety_factor)
    batch_size = max(100, usable_memory // max(bytes_per_voxel, 1))

    return batch_size


# Keep old name for backward compatibility
_get_optimal_batch_size_tofts = _get_optimal_batch_size


def get_optimal_batch_size(
    n_timepoints: int,
    model: str = "extended_tofts",
    safety_factor: float = 0.5,
) -> int:
    """Estimate optimal batch size based on GPU memory.

    Args:
        n_timepoints: Number of time points per curve.
        model: Model type. '2cxm' uses 2x memory multiplier.
        safety_factor: Fraction of free GPU memory to use. Default 0.5.

    Returns:
        Recommended batch size for GPU processing.
        Returns 50000 if GPU unavailable (CPU batch size).
    """
    multiplier = 2.0 if model == "2cxm" else 1.0
    return _get_optimal_batch_size(n_timepoints, safety_factor, multiplier)


def evaluate_model_gpu(
    model_name: str,
    params: dict[str, npt.NDArray[np.floating]],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
) -> GPUModelResult:
    """Evaluate a DCE model on GPU using the appropriate batched function.

    This is a convenience wrapper that dispatches to the appropriate
    GPU-accelerated model evaluation function based on the model name.

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
        use_gpu: Whether to use GPU if available. Default True.
        batch_size: Number of voxels per GPU batch. Auto-determined if None.

    Returns:
        GPUModelResult containing tissue concentration curves.

    Raises:
        ValueError: If model_name is not recognized.
        KeyError: If required parameters are missing from params dict.
    """
    if model_name == "tofts":
        return batched_tofts_model_gpu(
            ktrans=params["ktrans"],
            ve=params["ve"],
            aif=aif,
            time_points=time_points,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )
    elif model_name == "extended_tofts":
        return batched_extended_tofts_model_gpu(
            ktrans=params["ktrans"],
            ve=params["ve"],
            vp=params["vp"],
            aif=aif,
            time_points=time_points,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )
    elif model_name == "patlak":
        return batched_patlak_model_gpu(
            ktrans=params["ktrans"],
            vp=params["vp"],
            aif=aif,
            time_points=time_points,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )
    elif model_name == "2cxm":
        return batched_2cxm_model_gpu(
            fp=params["fp"],
            ps=params["ps"],
            ve=params["ve"],
            vp=params["vp"],
            aif=aif,
            time_points=time_points,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: "
            "'tofts', 'extended_tofts', 'patlak', '2cxm'"
        )
