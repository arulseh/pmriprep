"""Unit tests for QC visualization pure utility functions."""

import numpy as np
import pytest

from pmriprep.utils.qc_utils import (
    calculate_framewise_displacement,
    get_radiological_slice,
)


class TestGetRadiologicalSlice:
    """Tests for radiological slice extraction."""

    def test_axial_middle_slice(self) -> None:
        """Axial slice (axis=2) of (10,10,10) returns a 2D array of shape (10,10)."""
        data = np.random.default_rng(42).random((10, 10, 10))
        result = get_radiological_slice(data, axis=2)
        assert result.ndim == 2
        assert result.shape == (10, 10)

    def test_coronal_slice(self) -> None:
        """Coronal slice (axis=1) returns correct 2D extraction."""
        data = np.random.default_rng(42).random((10, 12, 14))
        result = get_radiological_slice(data, axis=1)
        assert result.ndim == 2
        # Coronal: data[:, slice_idx, :] is (10, 14), then rot90 -> (14, 10), then fliplr -> (14, 10)
        assert result.shape == (14, 10)

    def test_sagittal_slice(self) -> None:
        """Sagittal slice (axis=0) returns correct 2D extraction."""
        data = np.random.default_rng(42).random((10, 12, 14))
        result = get_radiological_slice(data, axis=0)
        assert result.ndim == 2
        # Sagittal: data[slice_idx, :, :] is (12, 14), then rot90 -> (14, 12)
        assert result.shape == (14, 12)

    def test_custom_index(self) -> None:
        """Specifying slice_idx=3 extracts that particular slice."""
        data = np.zeros((10, 10, 10))
        # Mark axial slice 3 distinctly
        data[:, :, 3] = 1.0
        result = get_radiological_slice(data, axis=2, slice_idx=3)
        # All values in this slice should be 1.0
        assert np.all(result == 1.0)

    def test_non_3d_raises(self) -> None:
        """Non-3D input raises ValueError."""
        data_2d = np.zeros((10, 10))
        with pytest.raises(ValueError, match="Expected 3D"):
            get_radiological_slice(data_2d)

        data_4d = np.zeros((10, 10, 10, 5))
        with pytest.raises(ValueError, match="Expected 3D"):
            get_radiological_slice(data_4d)


class TestCalculateFramewiseDisplacement:
    """Tests for framewise displacement calculation."""

    def test_known_motion_params(self) -> None:
        """Known translation step of 1mm in x gives FD=1 for that frame."""
        # 3 frames: no motion, then 1mm x-translation, then no further change
        # Format: [rx, ry, rz, tx, ty, tz]
        params = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # +1mm in tx
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # no further change
        ])
        fd = calculate_framewise_displacement(params)
        assert fd[0] == pytest.approx(0.0)
        assert fd[1] == pytest.approx(1.0)
        assert fd[2] == pytest.approx(0.0)

    def test_first_frame_always_zero(self) -> None:
        """First FD value is always 0 regardless of motion parameters."""
        rng = np.random.default_rng(42)
        params = rng.random((10, 6))
        fd = calculate_framewise_displacement(params)
        assert fd[0] == pytest.approx(0.0)
        assert len(fd) == 10

    def test_constant_params_zero_fd(self) -> None:
        """Constant (non-zero) motion parameters give FD=0 for all frames."""
        # All frames have the same translation [0,0,0,1,0,0] - no change between frames
        params = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], (5, 1))
        fd = calculate_framewise_displacement(params)
        np.testing.assert_allclose(fd, 0.0, atol=1e-15)

    def test_rotation_scaled_by_radius(self) -> None:
        """Rotation displacement is scaled by the head radius parameter."""
        # 0.01 rad rotation in rx between frames
        params = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        fd_default = calculate_framewise_displacement(params, radius=50.0)
        fd_large = calculate_framewise_displacement(params, radius=100.0)
        # FD from rotation = radius * |delta_r| = 50 * 0.01 = 0.5 (default)
        assert fd_default[1] == pytest.approx(0.5)
        assert fd_large[1] == pytest.approx(1.0)
