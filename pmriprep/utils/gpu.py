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


class GPUVoxelProcessor:
    """GPU memory management for large-scale voxel-wise processing.

    This class provides automatic batch size optimization based on available
    GPU memory, memory pooling for reduced allocation overhead, and efficient
    batch iteration for processing large volumes.

    The processor estimates memory requirements per voxel and determines the
    optimal batch size to maximize GPU utilization while avoiding out-of-memory
    errors. It also provides memory pooling to reduce allocation overhead
    during iterative processing.

    Attributes:
        n_voxels: Total number of voxels to process.
        n_timepoints: Number of timepoints per voxel (for time series data).
        bytes_per_element: Size of each array element in bytes (default: 8 for float64).
        memory_fraction: Fraction of free GPU memory to use (default: 0.5).
        min_batch_size: Minimum batch size to use.
        device: Target device ("gpu", "cuda", or "cpu").

    Example:
        >>> processor = GPUVoxelProcessor(
        ...     n_voxels=100000,
        ...     n_timepoints=60,
        ...     memory_fraction=0.5,
        ... )
        >>> for batch_start, batch_end, batch_data in processor.iter_batches(data):
        ...     # Process batch_data on GPU
        ...     results[batch_start:batch_end] = process(batch_data)
    """

    def __init__(
        self,
        n_voxels: int,
        n_timepoints: int = 1,
        bytes_per_element: int = 8,
        memory_fraction: float = 0.5,
        min_batch_size: int = 100,
        device: Literal["cpu", "cuda", "gpu"] = "gpu",
        output_arrays: int = 5,
        intermediate_factor: float = 2.0,
    ) -> None:
        """Initialize the GPU voxel processor.

        Args:
            n_voxels: Total number of voxels to process.
            n_timepoints: Number of timepoints per voxel (for 4D data).
            bytes_per_element: Bytes per array element (8 for float64, 4 for float32).
            memory_fraction: Fraction of free GPU memory to use (0.0-1.0).
                Higher values maximize utilization but risk OOM errors.
            min_batch_size: Minimum voxels per batch.
            device: Target device ("gpu", "cuda", or "cpu").
            output_arrays: Expected number of output arrays per voxel
                (e.g., CBF, CBV, MTT, Tmax, residue = 5).
            intermediate_factor: Multiplier for intermediate memory usage
                during computation (accounts for temporaries).
        """
        if n_voxels <= 0:
            raise ValueError(f"n_voxels must be positive, got {n_voxels}")
        if n_timepoints <= 0:
            raise ValueError(f"n_timepoints must be positive, got {n_timepoints}")
        if not 0.0 < memory_fraction <= 1.0:
            raise ValueError(f"memory_fraction must be in (0.0, 1.0], got {memory_fraction}")
        if min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be positive, got {min_batch_size}")

        self.n_voxels = n_voxels
        self.n_timepoints = n_timepoints
        self.bytes_per_element = bytes_per_element
        self.memory_fraction = memory_fraction
        self.min_batch_size = min_batch_size
        self.device = device
        self.output_arrays = output_arrays
        self.intermediate_factor = intermediate_factor

        # Compute memory requirements per voxel
        # Input: n_timepoints elements per voxel
        # Output: output_arrays elements per voxel (or n_timepoints if keeping full residue)
        # Intermediate: varies by algorithm, use intermediate_factor
        self._bytes_per_voxel = self._estimate_bytes_per_voxel()

        # Determine if GPU is actually available
        self._use_gpu = (
            device in ("gpu", "cuda") and is_cupy_available()
        )

        # Memory pool for reusable arrays (reduces allocation overhead)
        self._memory_pool: dict[str, ArrayLike] = {}
        self._pool_shapes: dict[str, tuple[int, ...]] = {}

    def _estimate_bytes_per_voxel(self) -> int:
        """Estimate memory requirements per voxel in bytes.

        This accounts for:
        - Input time series data
        - Output arrays (CBF, CBV, MTT, etc.)
        - Intermediate computation arrays

        Returns:
            Estimated bytes needed per voxel.
        """
        # Input: one time series per voxel
        input_bytes = self.n_timepoints * self.bytes_per_element

        # Output: output_arrays scalars OR residue function per voxel
        # Use max(output_arrays, n_timepoints) for conservative estimate
        output_bytes = max(self.output_arrays, self.n_timepoints) * self.bytes_per_element

        # Intermediate: factor for temporary arrays during computation
        intermediate_bytes = int(input_bytes * self.intermediate_factor)

        total = input_bytes + output_bytes + intermediate_bytes
        return total

    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory.

        This method queries the available GPU memory and calculates how many
        voxels can be processed in a single batch while staying within the
        memory_fraction limit.

        Returns:
            Optimal number of voxels per batch. Returns n_voxels if using CPU
            or if all voxels fit in available memory.

        Notes:
            - Always returns at least min_batch_size
            - Returns n_voxels if processing on CPU (no memory limit)
            - Accounts for memory_fraction safety margin
        """
        # CPU processing: no memory limit, process all at once
        if not self._use_gpu:
            return self.n_voxels

        # Get available GPU memory
        mem_info = get_gpu_memory_info()
        free_bytes = mem_info["free"]

        if free_bytes == 0:
            # Unknown memory, use conservative default
            return max(self.min_batch_size, min(1000, self.n_voxels))

        # Calculate available memory with safety margin
        available_bytes = int(free_bytes * self.memory_fraction)

        # Calculate batch size
        batch_size = available_bytes // max(self._bytes_per_voxel, 1)

        # Clamp to valid range
        batch_size = max(self.min_batch_size, batch_size)
        batch_size = min(batch_size, self.n_voxels)

        return batch_size

    def get_memory_estimate(self, batch_size: int | None = None) -> dict[str, int | float]:
        """Estimate memory usage for a given batch size.

        Args:
            batch_size: Batch size to estimate. If None, uses optimal batch size.

        Returns:
            Dictionary with memory estimates:
                - "per_voxel": Bytes per voxel
                - "batch_total": Total bytes for batch
                - "gpu_free": Free GPU memory (0 if CPU)
                - "gpu_total": Total GPU memory (0 if CPU)
                - "utilization": Estimated memory utilization (0-1)
        """
        if batch_size is None:
            batch_size = self.get_optimal_batch_size()

        batch_total = batch_size * self._bytes_per_voxel
        mem_info = get_gpu_memory_info()

        utilization = 0.0
        if mem_info["total"] > 0:
            utilization = batch_total / mem_info["total"]

        return {
            "per_voxel": self._bytes_per_voxel,
            "batch_total": batch_total,
            "gpu_free": mem_info["free"],
            "gpu_total": mem_info["total"],
            "utilization": utilization,
        }

    def allocate_pool(self, name: str, shape: tuple[int, ...], dtype: Any = None) -> ArrayLike:
        """Allocate or retrieve a reusable array from the memory pool.

        This method reduces allocation overhead by reusing arrays across
        batch iterations. Arrays are created on the target device (GPU or CPU).

        Args:
            name: Unique identifier for this array in the pool.
            shape: Shape of the array to allocate.
            dtype: Data type (default: float64).

        Returns:
            Array from the pool (may be reused if shape matches).

        Notes:
            If an array with the same name exists but different shape,
            it is replaced with a new allocation.
        """
        if dtype is None:
            dtype = np.float64

        # Check if we have a compatible array in the pool
        if name in self._memory_pool and self._pool_shapes.get(name) == shape:
            return self._memory_pool[name]

        # Allocate new array
        xp = get_array_module(prefer_gpu=self._use_gpu)
        arr = xp.zeros(shape, dtype=dtype)

        # Store in pool
        self._memory_pool[name] = arr
        self._pool_shapes[name] = shape

        return arr

    def free_pool(self, name: str | None = None) -> None:
        """Free arrays from the memory pool.

        Args:
            name: Specific array to free. If None, frees entire pool.
        """
        if name is None:
            self._memory_pool.clear()
            self._pool_shapes.clear()
        elif name in self._memory_pool:
            del self._memory_pool[name]
            del self._pool_shapes[name]

        # Trigger GPU memory cleanup if on GPU
        if self._use_gpu:
            clear_gpu_memory()

    def iter_batches(
        self,
        data: npt.NDArray[Any],
        batch_size: int | None = None,
    ) -> Any:
        """Iterate over data in optimized batches.

        This generator yields batches of voxel data, automatically handling
        GPU transfer and memory management. Each batch is transferred to the
        target device (GPU or CPU) for processing.

        Args:
            data: Input data array of shape (n_voxels, n_timepoints) or (n_voxels,).
                First dimension should be voxels.
            batch_size: Batch size to use. If None, uses optimal batch size.

        Yields:
            Tuples of (start_idx, end_idx, batch_data) where:
                - start_idx: Starting index in original array
                - end_idx: Ending index in original array
                - batch_data: Batch on target device (GPU or CPU array)

        Example:
            >>> processor = GPUVoxelProcessor(n_voxels=100000, n_timepoints=60)
            >>> results = np.zeros(100000)
            >>> for start, end, batch in processor.iter_batches(data):
            ...     batch_result = some_gpu_function(batch)
            ...     results[start:end] = to_host(batch_result)
        """
        if batch_size is None:
            batch_size = self.get_optimal_batch_size()

        n_total = data.shape[0]

        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)

            # Extract batch
            batch_cpu = data[start_idx:end_idx]

            # Transfer to target device
            if self._use_gpu:
                batch_device = to_device(batch_cpu, device="gpu")
            else:
                batch_device = batch_cpu

            yield start_idx, end_idx, batch_device

            # Synchronize and potentially clear memory between batches
            if self._use_gpu:
                synchronize()

    def iter_batches_multiple(
        self,
        *arrays: npt.NDArray[Any],
        batch_size: int | None = None,
    ) -> Any:
        """Iterate over multiple arrays in synchronized batches.

        Similar to iter_batches, but handles multiple input arrays that
        should be batched together along the first dimension.

        Args:
            *arrays: Input arrays to batch. All must have same first dimension.
            batch_size: Batch size to use. If None, uses optimal batch size.

        Yields:
            Tuples of (start_idx, end_idx, *batch_arrays) where batch_arrays
            are the batched portions of each input array on target device.

        Raises:
            ValueError: If arrays have different first dimensions.

        Example:
            >>> for start, end, conc_batch, mask_batch in processor.iter_batches_multiple(
            ...     concentration_curves, brain_mask
            ... ):
            ...     process(conc_batch, mask_batch)
        """
        if not arrays:
            return

        # Verify all arrays have same first dimension
        n_total = arrays[0].shape[0]
        for i, arr in enumerate(arrays[1:], start=2):
            if arr.shape[0] != n_total:
                raise ValueError(
                    f"Array 1 has shape {arrays[0].shape} but array {i} has shape {arr.shape}. "
                    "First dimensions must match."
                )

        if batch_size is None:
            batch_size = self.get_optimal_batch_size()

        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)

            # Extract and transfer batches
            batches: list[ArrayLike] = []
            for arr in arrays:
                batch_cpu = arr[start_idx:end_idx]
                if self._use_gpu:
                    batch_device = to_device(batch_cpu, device="gpu")
                else:
                    batch_device = batch_cpu
                batches.append(batch_device)

            yield (start_idx, end_idx, *batches)

            # Synchronize between batches
            if self._use_gpu:
                synchronize()

    def process_in_batches(
        self,
        data: npt.NDArray[Any],
        process_func: Any,
        output_shape: tuple[int, ...] | None = None,
        batch_size: int | None = None,
    ) -> npt.NDArray[Any]:
        """Process all voxels using a function with automatic batching.

        This is a convenience method that handles the full batch processing
        loop, including result aggregation.

        Args:
            data: Input data array of shape (n_voxels, ...).
            process_func: Function that takes a batch and returns results.
                Should accept GPU arrays if GPU processing is enabled.
            output_shape: Shape of output array. If None, assumes (n_voxels,).
            batch_size: Batch size to use. If None, uses optimal batch size.

        Returns:
            Complete results array on CPU.

        Example:
            >>> def compute_cbf(batch):
            ...     # batch is on GPU if GPU enabled
            ...     return some_computation(batch)
            >>> results = processor.process_in_batches(data, compute_cbf)
        """
        n_total = data.shape[0]
        if output_shape is None:
            output_shape = (n_total,)

        # Allocate output on CPU
        results = np.zeros(output_shape, dtype=np.float64)

        # Process batches
        for start_idx, end_idx, batch_data in self.iter_batches(data, batch_size):
            batch_result = process_func(batch_data)

            # Transfer result to CPU and store
            batch_result_cpu = to_host(batch_result)
            results[start_idx:end_idx] = batch_result_cpu

        return results

    def __repr__(self) -> str:
        """Return string representation of the processor."""
        return (
            f"GPUVoxelProcessor(n_voxels={self.n_voxels}, "
            f"n_timepoints={self.n_timepoints}, "
            f"device={'GPU' if self._use_gpu else 'CPU'}, "
            f"optimal_batch_size={self.get_optimal_batch_size()})"
        )
