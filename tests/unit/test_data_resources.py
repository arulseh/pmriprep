"""Tests for bundled package resources."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.data import DATA_DIR, get_arterial_atlas


_MASK_RESOURCES = {
    "mni_aif/MNI_2mm_MRA_arterial_MCA_ACA.nii.gz": (
        2, 4022, (2.374441, 7.854301, -5.204376),
    ),
    "mni_aif/MNI_2mm_MRA_arterial_MCA_ACA_dil.nii.gz": (
        2, 8894, (2.172701, 6.712615, -4.106364),
    ),
    "mni_aif/MNI_2mm_MRA_arterial_full_mask.nii.gz": (
        2, 6936, (1.481834, 1.475490, -15.814302),
    ),
    "mni_aif/MNI_2mm_MRA_arterial_full_mask_dil.nii.gz": (
        2, 15029, (1.409941, 0.263491, -14.618671),
    ),
    "mni_regions/Cerebellum.nii.gz": (
        1, 299986, (0.772573, -58.532265, -41.522081),
    ),
    "mni_regions/Cerebellum_L.nii.gz": (
        1, 144798, (-23.584359, -58.658814, -41.843423),
    ),
    "mni_regions/Cerebellum_R.nii.gz": (
        1, 154156, (23.655842, -58.324678, -41.160733),
    ),
    "mni_regions/Supra.nii.gz": (
        1, 1976693, (0.778314, -14.624698, 17.017161),
    ),
    "mni_regions/Supra_L.nii.gz": (
        1, 965594, (-30.873115, -15.420921, 16.765615),
    ),
    "mni_regions/Supra_R.nii.gz": (
        1, 1008935, (31.072580, -13.914659, 17.222104),
    ),
}

_TEMPLATEFLOW_GRIDS = {
    1: (
        (182, 218, 182),
        np.array([
            [1.0, 0.0, 0.0, -91.0],
            [0.0, 1.0, 0.0, -126.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
    2: (
        (91, 109, 91),
        np.array([
            [2.0, 0.0, 0.0, -90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
}


def test_default_arterial_atlas_exists() -> None:
    """Resolve the default arterial atlas from the installed resources."""
    atlas = get_arterial_atlas()

    assert atlas.is_file()
    assert atlas.name == "MNI_2mm_MRA_arterial_full_mask_dil.nii.gz"


@pytest.mark.parametrize(
    ("relative_path", "resolution", "expected_voxels", "expected_centroid"),
    [
        (Path(relative_path), resolution, expected_voxels, expected_centroid)
        for relative_path, (
            resolution, expected_voxels, expected_centroid,
        ) in _MASK_RESOURCES.items()
    ],
)
def test_bundled_masks_use_templateflow_grids(
    relative_path: Path,
    resolution: int,
    expected_voxels: int,
    expected_centroid: tuple[float, float, float],
) -> None:
    """Keep bundled masks binary and on the exact MNI152NLin6Asym grids."""
    image = nib.load(DATA_DIR / "atlases" / relative_path)
    data = np.asanyarray(image.dataobj)
    expected_shape, expected_affine = _TEMPLATEFLOW_GRIDS[resolution]

    assert image.shape == expected_shape
    np.testing.assert_array_equal(image.affine, expected_affine)
    np.testing.assert_array_equal(image.get_qform(), expected_affine)
    np.testing.assert_array_equal(image.get_sform(), expected_affine)
    assert int(image.header["qform_code"]) == 4
    assert int(image.header["sform_code"]) == 4
    assert set(np.unique(data)) == {0.0, 1.0}
    assert np.count_nonzero(data) == expected_voxels
    voxel_centroid = np.mean(np.argwhere(data > 0), axis=0)
    world_centroid = nib.affines.apply_affine(image.affine, voxel_centroid)
    np.testing.assert_allclose(world_centroid, expected_centroid, atol=1e-4)
