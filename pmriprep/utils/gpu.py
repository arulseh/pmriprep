"""GPU array management utilities for PMRIPrep.

This module provides a unified interface for working with arrays on CPU and GPU.
It abstracts the differences between NumPy and CuPy to enable seamless GPU
acceleration when available, with graceful fallback to CPU-only processing.

Features:
- Automatic detection of CuPy availability
- Unified array module interface (numpy/cupy)
- Device transfer utilities (to_device, to_host)
- GPU memory management for batch processing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import numpy as np
import numpy.typing as npt

# Type aliases
ArrayLike = Union[npt.NDArray[Any], Any]  # numpy array or cupy array
T = TypeVar("T", bound=npt.NDArray[Any])

# Global flag for CuPy availability
_CUPY_AVAILABLE: bool = False
_cupy_module: Any = None

# Try to import CuPy
try:
    import cupy as cp  # type: ignore[import-untyped]

    _CUPY_AVAILABLE = True
    _cupy_module = cp
except ImportError:
    pass


def is_cupy_available() -> bool:
    """Check if CuPy is available for GPU acceleration.

    Returns:
        True if CuPy is installed and functional, False otherwise.
    """
    return _CUPY_AVAILABLE


def get_array_module(
    *arrays: ArrayLike, prefer_gpu: bool = False
) -> Any:
    """Get the appropriate array module (numpy or cupy) for the given arrays.

    This function returns the array module that can handle all input arrays.
    If any input is a CuPy array, returns cupy. Otherwise returns numpy.

    Args:
        *arrays: Input arrays (numpy or cupy arrays).
        prefer_gpu: If True and no arrays provided, return cupy if available.
            If False or cupy unavailable, return numpy.

    Returns:
        The array module (numpy or cupy) appropriate for the inputs.

    Examples:
        >>> xp = get_array_module(some_array)
        >>> result = xp.zeros((10, 10))
    """
    # If CuPy is not available, always return numpy
    if not _CUPY_AVAILABLE:
        return np

    # Check if any input is a CuPy array
    for arr in arrays:
        if arr is not None and hasattr(arr, "__module__"):
            if "cupy" in getattr(arr, "__module__", ""):
                return _cupy_module

    # If prefer_gpu and no arrays force numpy, return cupy
    if prefer_gpu and _CUPY_AVAILABLE:
        return _cupy_module

    return np


def to_device(
    arr: npt.NDArray[Any],
    device: Literal["cpu", "cuda", "gpu"] = "gpu",
) -> ArrayLike:
    """Transfer a numpy array to the specified device.

    Args:
        arr: NumPy array to transfer.
        device: Target device. One of:
            - "cpu": Keep on CPU (returns numpy array)
            - "cuda" or "gpu": Transfer to GPU (returns cupy array)

    Returns:
        Array on the target device. Returns numpy array if device is "cpu"
        or if CuPy is unavailable. Returns cupy array if device is "cuda"
        or "gpu" and CuPy is available.

    Notes:
        If CuPy is not available and device is "cuda" or "gpu", the array
        is returned unchanged (stays on CPU) without raising an error.
    """
    if device == "cpu":
        return arr

    # GPU requested
    if not _CUPY_AVAILABLE:
        # Graceful fallback: return CPU array
        return arr

    # Transfer to GPU
    return _cupy_module.asarray(arr)


def to_host(arr: ArrayLike) -> npt.NDArray[Any]:
    """Transfer an array to CPU (host) memory.

    This function handles both numpy and cupy arrays. Numpy arrays are
    returned unchanged, while cupy arrays are transferred to CPU.

    Args:
        arr: Array to transfer (numpy or cupy).

    Returns:
        NumPy array on CPU.

    Notes:
        If the input is already a numpy array, it is returned unchanged
        (no copy is made unless necessary).
    """
    # If already numpy, return as-is
    if isinstance(arr, np.ndarray):
        return arr

    # If CuPy array, transfer to host
    if _CUPY_AVAILABLE and hasattr(arr, "get"):
        return arr.get()  # type: ignore[no-any-return]

    # Fallback: try to convert to numpy array
    return np.asarray(arr)


def get_device_name() -> str:
    """Get the name of the current GPU device.

    Returns:
        Device name string, or "CPU" if no GPU is available.
    """
    if not _CUPY_AVAILABLE:
        return "CPU"

    try:
        device = _cupy_module.cuda.Device()
        return str(device.name.decode() if hasattr(device.name, "decode") else device.name)
    except Exception:
        return "GPU (unknown)"


def get_gpu_memory_info() -> dict[str, int]:
    """Get GPU memory information.

    Returns:
        Dictionary with keys:
            - "free": Free memory in bytes
            - "total": Total memory in bytes
            - "used": Used memory in bytes

        Returns zeros for all values if CuPy is unavailable.
    """
    if not _CUPY_AVAILABLE:
        return {"free": 0, "total": 0, "used": 0}

    try:
        mem_info = _cupy_module.cuda.runtime.memGetInfo()
        free, total = mem_info
        return {
            "free": int(free),
            "total": int(total),
            "used": int(total - free),
        }
    except Exception:
        return {"free": 0, "total": 0, "used": 0}


def synchronize() -> None:
    """Synchronize the GPU device.

    This ensures all GPU operations have completed before continuing.
    Does nothing if CuPy is unavailable.
    """
    if _CUPY_AVAILABLE:
        try:
            _cupy_module.cuda.Device().synchronize()
        except Exception:
            pass


def clear_gpu_memory() -> None:
    """Clear unused GPU memory.

    Frees cached memory blocks that are no longer in use.
    Does nothing if CuPy is unavailable.
    """
    if _CUPY_AVAILABLE:
        try:
            mempool = _cupy_module.get_default_memory_pool()
            mempool.free_all_blocks()
        except Exception:
            pass


def array_on_device(arr: ArrayLike) -> Literal["cpu", "gpu"]:
    """Determine which device an array is on.

    Args:
        arr: Array to check (numpy or cupy).

    Returns:
        "cpu" if numpy array, "gpu" if cupy array.
    """
    if _CUPY_AVAILABLE and hasattr(arr, "__module__"):
        if "cupy" in getattr(arr, "__module__", ""):
            return "gpu"
    return "cpu"


def ensure_contiguous(arr: ArrayLike) -> ArrayLike:
    """Ensure an array is contiguous in memory.

    This is important for efficient GPU operations and data transfer.

    Args:
        arr: Array to make contiguous (numpy or cupy).

    Returns:
        Contiguous array on the same device as input.
    """
    xp = get_array_module(arr)
    return xp.ascontiguousarray(arr)
