"""Unit tests for QC visualization pure utility functions."""

import numpy as np
import pytest

from pmriprep.utils.qc_utils import (
    calculate_framewise_displacement,
    get_radiological_slice,
)


class TestGetRadiologicalSlice:
    """Tests for radiological slice extraction."""

    @pytest.mark.parametrize("axis,slice_idx,expected", [
        (2, None, [[232, 132, 32], [222, 122, 22], [212, 112, 12], [202, 102, 2]]),
        (1, None, [
            [224, 124, 24], [223, 123, 23], [222, 122, 22],
            [221, 121, 21], [220, 120, 20],
        ]),
        (0, None, [
            [104, 114, 124, 134], [103, 113, 123, 133], [102, 112, 122, 132],
            [101, 111, 121, 131], [100, 110, 120, 130],
        ]),
        (2, 1, [[231, 131, 31], [221, 121, 21], [211, 111, 11], [201, 101, 1]]),
    ], ids=["axial", "coronal", "sagittal", "custom_axial"])
    def test_slice_orientation(
        self, axis: int, slice_idx: int | None, expected: list[list[int]],
    ) -> None:
        """Coordinate-coded voxels retain their radiological display positions."""
        x, y, z = np.indices((3, 4, 5))
        data = 100 * x + 10 * y + z
        result = get_radiological_slice(data, axis=axis, slice_idx=slice_idx)
        np.testing.assert_array_equal(result, expected)

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
        # 3 frames: nonzero initial pose, then a 1 mm step and no further change.
        # Format: [rx, ry, rz, tx, ty, tz]
        params = np.array([
            [0.1, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 3.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 3.0, 0.0, 0.0],
        ])
        fd = calculate_framewise_displacement(params)
        np.testing.assert_array_equal(fd, [0.0, 1.0, 0.0])

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
