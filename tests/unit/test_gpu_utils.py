"""Unit tests for GPU detection utilities."""

from pmriprep.utils.gpu import (
    detect_gpu_count,
    get_device_name,
)


class TestDetectGpuCount:
    """Tests for GPU device count detection."""

    def test_returns_int(self) -> None:
        """detect_gpu_count returns an integer."""
        result = detect_gpu_count()
        assert isinstance(result, int)

    def test_returns_non_negative(self) -> None:
        """detect_gpu_count returns >= 0."""
        assert detect_gpu_count() >= 0


class TestGetDeviceName:
    """Tests for device name query."""

    def test_returns_cpu(self) -> None:
        """get_device_name returns 'CPU'."""
        assert get_device_name() == "CPU"
