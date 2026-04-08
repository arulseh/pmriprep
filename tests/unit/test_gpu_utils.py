"""Unit tests for GPU array management utilities."""

import numpy as np
import pytest

from pmriprep.utils.gpu import (
    BatchedVoxelProcessor,
    array_on_device,
    ensure_contiguous,
    get_array_module,
    get_device_name,
    get_gpu_memory_info,
    is_cupy_available,
    to_device,
    to_host,
)


# ---------------------------------------------------------------------------
# Availability / module selection
# ---------------------------------------------------------------------------

class TestIsCupyAvailable:
    """Tests for CuPy availability check."""

    def test_returns_bool(self) -> None:
        """is_cupy_available returns a boolean."""
        result = is_cupy_available()
        assert isinstance(result, bool)


class TestGetArrayModule:
    """Tests for array module selection."""

    def test_numpy_array_returns_numpy(self) -> None:
        """Numpy array input returns the numpy module."""
        arr = np.array([1.0, 2.0])
        xp = get_array_module(arr)
        assert xp is np

    def test_prefer_gpu_false_returns_numpy(self) -> None:
        """prefer_gpu=False returns numpy even if cupy is available."""
        xp = get_array_module(prefer_gpu=False)
        assert xp is np


# ---------------------------------------------------------------------------
# Device transfer
# ---------------------------------------------------------------------------

class TestToDevice:
    """Tests for CPU/GPU array transfer."""

    def test_cpu_returns_same_array(self) -> None:
        """to_device with device='cpu' returns the input array unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_device(arr, device="cpu")
        assert result is arr


class TestToHost:
    """Tests for transferring arrays to host (CPU) memory."""

    def test_numpy_returns_same_array(self) -> None:
        """Numpy input is returned without copy."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_host(arr)
        assert result is arr


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

class TestGetDeviceName:
    """Tests for device name query."""

    def test_no_gpu_returns_cpu(self) -> None:
        """Without CuPy, device name is 'CPU'."""
        if is_cupy_available():
            pytest.skip("CuPy is available; cannot test CPU-only path")
        assert get_device_name() == "CPU"


class TestGetGpuMemoryInfo:
    """Tests for GPU memory info query."""

    def test_no_gpu_returns_zeros(self) -> None:
        """Without CuPy, all memory values are zero."""
        if is_cupy_available():
            pytest.skip("CuPy is available; cannot test CPU-only path")
        info = get_gpu_memory_info()
        assert info["free"] == 0
        assert info["total"] == 0
        assert info["used"] == 0


# ---------------------------------------------------------------------------
# Array inspection / contiguity
# ---------------------------------------------------------------------------

class TestArrayOnDevice:
    """Tests for device location query."""

    def test_numpy_is_cpu(self) -> None:
        """Numpy array reports device as 'cpu'."""
        arr = np.array([1.0, 2.0])
        assert array_on_device(arr) == "cpu"


class TestEnsureContiguous:
    """Tests for memory contiguity enforcement."""

    def test_non_contiguous_becomes_contiguous(self) -> None:
        """Non-contiguous slice is made contiguous with same values."""
        arr = np.arange(20).reshape(4, 5)
        col = arr[:, 2]  # non-contiguous view
        assert not col.flags["C_CONTIGUOUS"]

        result = ensure_contiguous(col)
        assert result.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(result, col)


# ---------------------------------------------------------------------------
# BatchedVoxelProcessor
# ---------------------------------------------------------------------------

class TestBatchedVoxelProcessor:
    """Tests for the GPU batch processor."""

    def test_construction_sets_attributes(self) -> None:
        """Valid constructor arguments are stored as attributes."""
        proc = BatchedVoxelProcessor(
            n_voxels=5000,
            n_timepoints=60,
            memory_fraction=0.4,
            min_batch_size=50,
        )
        assert proc.n_voxels == 5000
        assert proc.n_timepoints == 60
        assert proc.memory_fraction == pytest.approx(0.4)
        assert proc.min_batch_size == 50

    def test_zero_voxels_raises(self) -> None:
        """n_voxels=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_voxels must be positive"):
            BatchedVoxelProcessor(n_voxels=0)

    def test_negative_memory_fraction_raises(self) -> None:
        """Negative memory_fraction raises ValueError."""
        with pytest.raises(ValueError, match="memory_fraction"):
            BatchedVoxelProcessor(n_voxels=100, memory_fraction=-0.1)

    def test_cpu_optimal_batch_size_equals_n_voxels(self) -> None:
        """On CPU, optimal batch size equals total voxels (no memory limit)."""
        proc = BatchedVoxelProcessor(n_voxels=10000, device="cpu")
        assert proc.get_optimal_batch_size() == 10000

    def test_iter_batches_correct_splits(self) -> None:
        """100 voxels with batch_size=30 yields 4 batches of 30,30,30,10."""
        proc = BatchedVoxelProcessor(n_voxels=100, device="cpu")
        data = np.arange(100, dtype=np.float64)

        batches = list(proc.iter_batches(data, batch_size=30))
        assert len(batches) == 4

        sizes = [end - start for start, end, _ in batches]
        assert sizes == [30, 30, 30, 10]

        # Verify indices cover full range
        all_indices = []
        for start, end, batch in batches:
            assert len(batch) == end - start
            all_indices.extend(range(start, end))
        assert all_indices == list(range(100))

    def test_process_in_batches_identity(self) -> None:
        """Identity function through process_in_batches reproduces input."""
        proc = BatchedVoxelProcessor(n_voxels=50, device="cpu")
        data = np.linspace(0, 1, 50)

        result = proc.process_in_batches(
            data,
            process_func=lambda x: x,
            batch_size=20,
        )
        np.testing.assert_array_equal(result, data)
