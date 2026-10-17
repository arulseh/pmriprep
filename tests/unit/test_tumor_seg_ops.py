"""Unit tests for tumor segmentation pure utility functions."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.utils.tumor_seg import (
    _analyze_tumor_anatomy,
    _create_hemisphere_mask,
    _create_spherical_kernel,
    _estimate_brain_symmetry_midline,
    create_nawm_roi,
    dilate_mask_from_array,
    erode_mask_from_array,
)
from pmriprep.workflows.tumor_seg.base import _extract_nawm_outputs


def test_extract_nawm_outputs_separates_qc_and_quantitative_paths() -> None:
    """An undersized mask remains available for QC but not quantification."""
    result = {
        "output_file": "/tmp/nawm.nii.gz",
        "analysis": {
            "valid": False,
            "quantitative_use_exclusion_reason": (
                "insufficient_voxels_after_erosion_relaxation"
            ),
        },
    }

    raw, quantitative, valid, reason = _extract_nawm_outputs(result)

    assert raw == "/tmp/nawm.nii.gz"
    assert quantitative == ""
    assert valid is False
    assert reason == "insufficient_voxels_after_erosion_relaxation"


def test_extract_nawm_outputs_passes_valid_mask_to_quantification() -> None:
    """A generated mask meeting the size criterion retains both roles."""
    result = {
        "output_file": "/tmp/nawm.nii.gz",
        "analysis": {"valid": True},
    }

    raw, quantitative, valid, reason = _extract_nawm_outputs(result)

    assert raw == quantitative == "/tmp/nawm.nii.gz"
    assert valid is True
    assert reason == ""


class TestCreateSphericalKernel:
    """Tests for spherical structuring element creation."""

    def test_isotropic_kernel_extent(self) -> None:
        """A 5 mm sphere includes its center and axis tips, but excludes corners."""
        kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 1.0))
        assert kernel.shape == (11, 11, 11)
        assert kernel[5, 5, 5]
        assert kernel[10, 5, 5]
        assert kernel[5, 10, 5]
        assert kernel[5, 5, 10]
        assert not kernel[0, 0, 0]
        assert not kernel[-1, -1, -1]

    def test_anisotropic_kernel_extent_in_millimeters(self) -> None:
        """A 5 mm sphere respects the 3 mm through-plane spacing."""
        kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 3.0))
        assert kernel.shape == (11, 11, 5)
        assert kernel[5, 5, 2]
        assert kernel[9, 5, 2]
        assert kernel[10, 5, 2]
        assert kernel[5, 5, 3]  # 3 mm from center
        assert not kernel[5, 5, 4]  # 6 mm from center
        assert kernel[9, 5, 3]  # 4 mm in-plane and 3 mm through-plane: 5 mm
        assert not kernel[10, 5, 3]

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
        assert dilated[15, 15, 15]

    def test_very_small_radius_unchanged(self) -> None:
        """A radius smaller than the nearest voxel spacing leaves the mask intact."""
        mask = np.zeros((21, 21, 21), dtype=bool)
        mask[10, 10, 10] = True
        dilated = dilate_mask_from_array(mask, 0.01, (2.0, 2.0, 2.0))
        np.testing.assert_array_equal(dilated, mask)


class TestErodeMaskFromArray:
    """Tests for array-based mask erosion."""

    def test_large_cube_eroded_smaller(self) -> None:
        """Eroding a large filled cube produces a smaller mask."""
        mask = np.zeros((41, 41, 41), dtype=bool)
        mask[5:36, 5:36, 5:36] = True  # 31x31x31 filled cube
        eroded = erode_mask_from_array(mask, 3.0, (1.0, 1.0, 1.0))
        assert 0 < eroded.sum() < mask.sum()
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
        result = _analyze_tumor_anatomy(mask, affine)
        assert result.laterality == "left"

    def test_right_hemisphere_tumor(self) -> None:
        """Tumor in right hemisphere (x_physical > 0) gives laterality='right'."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        # Place tumor in high-i voxels: x_physical = i - 30 > 0 (right in RAS)
        mask[45:55, 28:32, 28:32] = True
        result = _analyze_tumor_anatomy(mask, affine)
        assert result.laterality == "right"

    def test_bilateral_tumor(self) -> None:
        """Tumor spanning midline (<=60% either side) gives laterality='bilateral'."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        # Place tumor symmetrically across midline (voxel 30 = x=0)
        mask[25:35, 28:32, 28:32] = True
        result = _analyze_tumor_anatomy(mask, affine)
        assert result.laterality == "bilateral"

    def test_supratentorial_atlas_overlap(self) -> None:
        """The largest atlas overlap classifies a supratentorial tumor."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        mask[28:32, 28:32, 38:42] = True
        supra = np.zeros_like(mask)
        supra[:, :, 30:] = True
        cerebellum = np.zeros_like(mask)
        cerebellum[:, :, :20] = True
        result = _analyze_tumor_anatomy(mask, affine, supra, cerebellum)
        assert result.is_supratentorial is True
        assert result.compartment == "supratentorial"

    def test_infratentorial_atlas_overlap(self) -> None:
        """The largest atlas overlap classifies an infratentorial tumor."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        mask[28:32, 28:32, 3:7] = True
        supra = np.zeros_like(mask)
        supra[:, :, 30:] = True
        cerebellum = np.zeros_like(mask)
        cerebellum[:, :, :20] = True
        result = _analyze_tumor_anatomy(mask, affine, supra, cerebellum)
        assert result.is_supratentorial is False
        assert result.compartment == "infratentorial"

    def test_empty_mask(self) -> None:
        """Empty tumor mask returns bilateral, volume=0, supratentorial=True."""
        affine = self._make_ras_affine()
        mask = np.zeros((60, 60, 60), dtype=bool)
        result = _analyze_tumor_anatomy(mask, affine)
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
        assert not mask[10, 10, 10]  # i=10, x=0 (midline)
        assert not mask[15, 10, 10]  # i=15, x=+5 (right)

    def test_right_hemisphere_mask(self) -> None:
        """Right hemisphere mask selects voxels with x_physical > 0."""
        shape = (20, 20, 20)
        affine = self._make_ras_affine()
        mask = _create_hemisphere_mask(shape, affine, "right")
        assert mask[15, 10, 10]   # i=15, x=+5 (right)
        assert not mask[10, 10, 10]  # i=10, x=0 (midline)
        assert not mask[0, 10, 10]  # i=0, x=-10 (left)

    def test_subject_midline_does_not_assume_world_origin(self) -> None:
        """A translated native image is divided using brain symmetry."""
        shape = (20, 20, 20)
        affine = np.eye(4)
        affine[:3, 3] = (100.0, -10.0, -10.0)
        brain = np.ones(shape, dtype=bool)
        midline = _estimate_brain_symmetry_midline(brain, affine)
        assert midline is not None
        left = _create_hemisphere_mask(shape, affine, "left", midline)
        right = _create_hemisphere_mask(shape, affine, "right", midline)
        assert left[2, 10, 10]
        assert right[17, 10, 10]
        assert np.any(left)
        assert np.any(right)


@pytest.mark.parametrize("use_atlas", [False, True])
def test_bilateral_nawm_preserves_lower_tumor_volume_hemisphere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_atlas: bool,
) -> None:
    """Atlas restriction must not undo bilateral hemisphere selection."""
    shape = (20, 8, 8)
    affine = np.eye(4)
    affine[:3, 3] = (-10.0, -4.0, -4.0)

    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[4:6, 3:5, 3:5] = 1  # 8 left-hemisphere voxels
    tumor[14:17, 3:5, 3:5] = 1  # 12 right-hemisphere voxels
    wm = np.ones(shape, dtype=np.float32)
    brain = np.ones(shape, dtype=np.uint8)

    def _save(name: str, data: np.ndarray) -> Path:
        path = tmp_path / name
        nib.save(nib.Nifti1Image(data, affine), path)
        return path

    tumor_file = _save("tumor.nii.gz", tumor)
    wm_file = _save("wm.nii.gz", wm)
    brain_file = _save("brain.nii.gz", brain)

    atlas_kwargs: dict[str, str] = {}
    atlas_constraint = np.ones(shape, dtype=bool)
    if use_atlas:
        region_dir = tmp_path / "regions"
        region_dir.mkdir()
        supra_source = _save("regions/Supra.nii.gz", np.ones(shape, dtype=np.uint8))
        _save("regions/Cerebellum.nii.gz", np.zeros(shape, dtype=np.uint8))
        transform = tmp_path / "t1w_to_mni.mat"
        np.savetxt(transform, np.eye(4))

        transform_calls = 0

        def _mock_transform(atlas_path: str, *_args: str) -> str:
            nonlocal transform_calls
            transform_calls += 1
            data = (
                atlas_constraint.astype(np.uint8)
                if Path(atlas_path) == supra_source
                else np.zeros(shape, dtype=np.uint8)
            )
            return str(_save(f"transformed_{transform_calls}.nii.gz", data))

        monkeypatch.setattr(
            "pmriprep.utils.tumor_seg._transform_atlas_to_t1w",
            _mock_transform,
        )
        atlas_kwargs = {
            "t1w_to_mni_xfm": str(transform),
            "reference_image": str(tumor_file),
            "region_masks_dir": str(region_dir),
        }

    result = create_nawm_roi(
        tumor_mask_file=tumor_file,
        wm_probseg_file=wm_file,
        output_file=tmp_path / f"nawm_{use_atlas}.nii.gz",
        tumor_margin_mm=0.0,
        nawm_erosion=0,
        nawm_min_voxels=1,
        brain_mask_file=brain_file,
        **atlas_kwargs,
    )

    analysis = result["analysis"]
    assert analysis["tumor_laterality"] == "bilateral"
    assert analysis["selected_hemisphere"] == "left"
    assert analysis["atlas_used"] is use_atlas
    assert analysis["valid"] is use_atlas
    assert analysis["nawm_compartment"] == (
        "supratentorial" if use_atlas else "hemisphere_unrestricted"
    )
    assert analysis["quantitative_use_exclusion_reason"] == (
        None if use_atlas else "supratentorial_atlas_unavailable"
    )

    nawm = np.asarray(nib.load(result["output_file"]).dataobj) > 0
    left = _create_hemisphere_mask(shape, affine, "left")
    assert np.any(nawm)
    assert not np.any(nawm & ~left)
    assert np.array_equal(nawm, left & atlas_constraint & ~(tumor > 0))


def test_infratentorial_tumor_uses_bilateral_supratentorial_nawm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cerebellar tumor never selects cerebellar white matter as reference."""
    shape = (20, 8, 10)
    affine = np.eye(4)
    affine[:3, 3] = (-10.0, -4.0, -5.0)
    supra = np.zeros(shape, dtype=np.uint8)
    supra[:, :, 5:] = 1
    cerebellum = np.zeros(shape, dtype=np.uint8)
    cerebellum[:, :, :5] = 1
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[8:12, 3:5, 1:3] = 1

    def _save(name: str, data: np.ndarray) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, affine), path)
        return path

    tumor_file = _save("tumor.nii.gz", tumor)
    wm_file = _save("wm.nii.gz", np.ones(shape, dtype=np.float32))
    brain_file = _save("brain.nii.gz", np.ones(shape, dtype=np.uint8))
    region_dir = tmp_path / "regions"
    supra_source = _save("regions/Supra.nii.gz", supra)
    _save("regions/Cerebellum.nii.gz", cerebellum)
    transform = tmp_path / "t1w_to_mni.mat"
    np.savetxt(transform, np.eye(4))
    calls = 0

    def _mock_transform(atlas_path: str, *_args: str) -> str:
        nonlocal calls
        calls += 1
        data = supra if Path(atlas_path) == supra_source else cerebellum
        return str(_save(f"transformed_{calls}.nii.gz", data))

    monkeypatch.setattr(
        "pmriprep.utils.tumor_seg._transform_atlas_to_t1w", _mock_transform
    )
    result = create_nawm_roi(
        tumor_file,
        wm_file,
        output_file=tmp_path / "nawm.nii.gz",
        tumor_margin_mm=0.0,
        nawm_erosion=0,
        nawm_min_voxels=1,
        t1w_to_mni_xfm=str(transform),
        reference_image=str(tumor_file),
        region_masks_dir=str(region_dir),
        brain_mask_file=brain_file,
    )

    analysis = result["analysis"]
    assert analysis["compartment_label"] == "infratentorial"
    assert analysis["selected_hemisphere"] == "both"
    assert "infratentorial_tumor_uses_bilateral_supratentorial_nawm" in analysis["warnings"]
    nawm = np.asarray(nib.load(result["output_file"]).dataobj) > 0
    assert np.array_equal(nawm, supra.astype(bool))
