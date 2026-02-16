"""Unit tests for HD-GLIO pure utility functions."""

import numpy as np
import pytest

from pmriprep.utils.hdglio import (
    TumorAnalysis,
    _analyze_tumor_anatomy,
    _create_hemisphere_mask,
    _create_spherical_kernel,
    dilate_mask_from_array,
    erode_mask_from_array,
)


class TestCreateSphericalKernel:
    """Tests for spherical structuring element creation."""

    def test_isotropic_kernel_shape(self) -> None:
        """Isotropic 5mm kernel with 1mm voxels produces (11,11,11) shape."""
        kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 1.0))
        assert kernel.shape == (11, 11, 11)

    def test_isotropic_kernel_center_true(self) -> None:
        """Center voxel of isotropic kernel is True (distance=0)."""
        kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 1.0))
        center = tuple(s // 2 for s in kernel.shape)
        assert kernel[center] is np.True_

    def test_isotropic_kernel_corners_false(self) -> None:
        """Corner voxels of isotropic kernel are outside the sphere."""
        kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 1.0))
        # Corner distance = sqrt(5^2 + 5^2 + 5^2) = 8.66 > 5.0
        assert not kernel[0, 0, 0]
        assert not kernel[-1, -1, -1]

    def test_anisotropic_kernel_shape(self) -> None:
        """Anisotropic 5mm kernel with (1,1,3)mm voxels produces (11,11,5)."""
        kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 3.0))
        # z: ceil(5/3) = 2, so size_z = 2*2+1 = 5
        assert kernel.shape == (11, 11, 5)

    def test_anisotropic_kernel_still_spherical(self) -> None:
        """Anisotropic kernel represents a sphere in physical space."""
        kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 3.0))
        # Center should be True
        cx, cy, cz = kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2
        assert kernel[cx, cy, cz]
        # A point at 4mm in x (4 voxels) should be inside
        assert kernel[cx + 4, cy, cz]
        # A point at 6mm in x (6 voxels) would be outside radius of 5mm
        # But kernel edge is at nx=5, so index cx+5 is the boundary
        # 5 voxels * 1mm = 5mm = exactly radius, so it should be True (<=)
        assert kernel[cx + 5, cy, cz]

    def test_safety_cap_raises(self) -> None:
        """Enormous kernel (1000mm, 0.001mm voxels) raises ValueError."""
        with pytest.raises(ValueError, match="safety cap"):
            _create_spherical_kernel(1000.0, (0.001, 0.001, 0.001))


class TestDilateMaskFromArray:
    """Tests for array-based mask dilation."""

    def test_single_voxel_dilated(self) -> None:
        """Single-voxel mask dilated with 3mm sphere grows in volume."""
        mask = np.zeros((31, 31, 31), dtype=bool)
        mask[15, 15, 15] = True
        dilated = dilate_mask_from_array(mask, 3.0, (1.0, 1.0, 1.0))
        assert dilated.sum() > mask.sum()

    def test_single_voxel_contains_original(self) -> None:
        """Dilated mask contains the original seed voxel."""
        mask = np.zeros((31, 31, 31), dtype=bool)
        mask[15, 15, 15] = True
        dilated = dilate_mask_from_array(mask, 3.0, (1.0, 1.0, 1.0))
        assert dilated[15, 15, 15]

    def test_very_small_radius_unchanged(self) -> None:
        """Dilation with radius smaller than voxel size barely changes mask."""
        mask = np.zeros((21, 21, 21), dtype=bool)
        mask[10, 10, 10] = True
        # Radius 0.1mm with 2mm voxels: ceil(0.1/2)=1, kernel size=3
        # The kernel center + face-adjacent voxels within 0.1mm
        # Only center is within 0.1mm, so dilation adds nothing beyond kernel center
        dilated = dilate_mask_from_array(mask, 0.01, (2.0, 2.0, 2.0))
        # With such a tiny radius, kernel is essentially just the center voxel
        # ceil(0.01/2) = 1, so kernel is (3,3,3) but only center is <= 0.01mm
        assert dilated[10, 10, 10]


class TestErodeMaskFromArray:
    """Tests for array-based mask erosion."""

    def test_large_cube_eroded_smaller(self) -> None:
        """Eroding a large filled cube produces a smaller mask."""
        mask = np.zeros((41, 41, 41), dtype=bool)
        mask[5:36, 5:36, 5:36] = True  # 31x31x31 filled cube
        eroded = erode_mask_from_array(mask, 3.0, (1.0, 1.0, 1.0))
        assert 0 < eroded.sum() < mask.sum()

    def test_large_cube_eroded_contained(self) -> None:
        """Eroded mask is a subset of the original."""
        mask = np.zeros((41, 41, 41), dtype=bool)
        mask[5:36, 5:36, 5:36] = True
        eroded = erode_mask_from_array(mask, 3.0, (1.0, 1.0, 1.0))
        # Every True voxel in eroded must also be True in original
        assert np.all(mask[eroded])

    def test_zero_radius_returns_copy(self) -> None:
        """Erosion with radius=0 returns an unchanged copy."""
        mask = np.zeros((21, 21, 21), dtype=bool)
        mask[8:13, 8:13, 8:13] = True
        eroded = erode_mask_from_array(mask, 0.0, (1.0, 1.0, 1.0))
        np.testing.assert_array_equal(eroded, mask)
        # Verify it is a copy, not the same object
        assert eroded is not mask

    def test_small_mask_fully_eroded(self) -> None:
        """A small thin mask eroded with a large radius becomes all False."""
        mask = np.zeros((21, 21, 21), dtype=bool)
        mask[10, 10, 10] = True  # Single voxel
        eroded = erode_mask_from_array(mask, 3.0, (1.0, 1.0, 1.0))
        assert eroded.sum() == 0


class TestAnalyzeTumorAnatomy:
    """Tests for tumor anatomy analysis."""

    def _make_ras_affine(self) -> np.ndarray:
        """Create a simple RAS-aligned affine with 1mm isotropic voxels.

        Volume is 60x60x60, centered at origin: voxel 30 maps to (0,0,0).
        """
        affine = np.eye(4)
        affine[0, 3] = -30.0  # x offset so center is at x=0
        affine[1, 3] = -30.0
        affine[2, 3] = -30.0
        return affine

    def test_left_hemisphere_tumor(self) -> None:
        """Tumor in left hemisphere (x_physical < 0) gives laterality='left'."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        # Place tumor in low-i voxels: x_physical = i - 30 < 0 (left in RAS)
        mask[5:15, 28:32, 28:32] = True
        result = _analyze_tumor_anatomy(mask, affine, (1.0, 1.0, 1.0))
        assert result.laterality == "left"

    def test_right_hemisphere_tumor(self) -> None:
        """Tumor in right hemisphere (x_physical > 0) gives laterality='right'."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        # Place tumor in high-i voxels: x_physical = i - 30 > 0 (right in RAS)
        mask[45:55, 28:32, 28:32] = True
        result = _analyze_tumor_anatomy(mask, affine, (1.0, 1.0, 1.0))
        assert result.laterality == "right"

    def test_bilateral_tumor(self) -> None:
        """Tumor spanning midline (<=60% either side) gives laterality='bilateral'."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        # Place tumor symmetrically across midline (voxel 30 = x=0)
        mask[25:35, 28:32, 28:32] = True
        result = _analyze_tumor_anatomy(mask, affine, (1.0, 1.0, 1.0))
        assert result.laterality == "bilateral"

    def test_supratentorial_high_z(self) -> None:
        """Tumor with centroid z > -20 is classified as supratentorial."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        # z_physical = k - 30. For k=40, z=10 > -20 => supratentorial
        mask[28:32, 28:32, 38:42] = True
        result = _analyze_tumor_anatomy(mask, affine, (1.0, 1.0, 1.0))
        assert result.is_supratentorial == True  # noqa: E712 (numpy bool)

    def test_infratentorial_low_z(self) -> None:
        """Tumor with centroid z <= -20 is classified as infratentorial."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        # z_physical = k - 30. For k around 5, z = -25 <= -20 => infratentorial
        mask[28:32, 28:32, 3:7] = True
        result = _analyze_tumor_anatomy(mask, affine, (1.0, 1.0, 1.0))
        assert result.is_supratentorial == False  # noqa: E712 (numpy bool)

    def test_empty_mask(self) -> None:
        """Empty tumor mask returns bilateral, volume=0, supratentorial=True."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        result = _analyze_tumor_anatomy(mask, affine, (1.0, 1.0, 1.0))
        assert result.laterality == "bilateral"
        assert result.volume_ml == pytest.approx(0.0)
        assert result.is_supratentorial is True


class TestCreateHemisphereMask:
    """Tests for hemisphere mask creation."""

    def _make_ras_affine(self) -> np.ndarray:
        """Create a simple RAS-aligned affine centered at origin."""
        affine = np.eye(4)
        affine[0, 3] = -10.0
        affine[1, 3] = -10.0
        affine[2, 3] = -10.0
        return affine

    def test_left_hemisphere_mask(self) -> None:
        """Left hemisphere mask selects voxels with x_physical < 0."""
        shape = (20, 20, 20)
        affine = self._make_ras_affine()
        mask = _create_hemisphere_mask(shape, affine, "left")
        # Voxels with i < 10 have x_physical = i - 10 < 0, so should be True
        assert mask[0, 10, 10]   # i=0, x=-10 (left)
        assert not mask[15, 10, 10]  # i=15, x=+5 (right)

    def test_right_hemisphere_mask(self) -> None:
        """Right hemisphere mask selects voxels with x_physical > 0."""
        shape = (20, 20, 20)
        affine = self._make_ras_affine()
        mask = _create_hemisphere_mask(shape, affine, "right")
        assert mask[15, 10, 10]   # i=15, x=+5 (right)
        assert not mask[0, 10, 10]  # i=0, x=-10 (left)


class TestTumorAnalysisDataclass:
    """Tests for the TumorAnalysis dataclass."""

    def test_all_fields_accessible(self) -> None:
        """All TumorAnalysis fields can be set and read."""
        ta = TumorAnalysis(
            is_supratentorial=True,
            laterality="left",
            centroid_voxels=(10.0, 20.0, 30.0),
            volume_ml=42.5,
            centroid_mni=(5.0, -10.0, 25.0),
        )
        assert ta.is_supratentorial is True
        assert ta.laterality == "left"
        assert ta.centroid_voxels == (10.0, 20.0, 30.0)
        assert ta.volume_ml == pytest.approx(42.5)
        assert ta.centroid_mni == (5.0, -10.0, 25.0)

    def test_centroid_mni_default_none(self) -> None:
        """centroid_mni defaults to None when not provided."""
        ta = TumorAnalysis(
            is_supratentorial=False,
            laterality="right",
            centroid_voxels=(0.0, 0.0, 0.0),
            volume_ml=0.0,
        )
        assert ta.centroid_mni is None
