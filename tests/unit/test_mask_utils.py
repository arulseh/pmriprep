"""Unit tests for pmriprep.utils.mask_utils.

Tests MergeMasks Nipype SimpleInterface class.
"""

import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _create_test_mask(
    shape: tuple = (10, 10, 10),
    affine: np.ndarray | None = None,
    data: np.ndarray | None = None,
    tmpdir: str | None = None,
    fname: str = "mask.nii.gz",
) -> str:
    """Create a test NIfTI mask file and return its path."""
    if affine is None:
        affine = np.eye(4)
    if data is None:
        data = np.zeros(shape, dtype=np.uint8)
    img = nib.Nifti1Image(data, affine)
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, fname)
    nib.save(img, out_path)
    return out_path


class TestMergeMasks:
    """Tests for MergeMasks interface."""

    def test_merge_both_masks(self, tmp_path: Path) -> None:
        """Merging two masks produces voxel-wise OR."""
        from pmriprep.utils.mask_utils import MergeMasks

        data_a = np.zeros((5, 5, 5), dtype=np.uint8)
        data_a[0:3, 0:3, 0:3] = 1  # cube in corner
        data_b = np.zeros((5, 5, 5), dtype=np.uint8)
        data_b[2:5, 2:5, 2:5] = 1  # overlapping cube

        mask_a = _create_test_mask(data=data_a, tmpdir=str(tmp_path), fname="a.nii.gz")
        mask_b = _create_test_mask(data=data_b, tmpdir=str(tmp_path), fname="b.nii.gz")

        merge = MergeMasks()
        merge.inputs.mask_a = mask_a
        merge.inputs.mask_b = mask_b

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            result = merge.run()
        finally:
            os.chdir(old_cwd)

        merged_img = nib.load(result.outputs.merged_mask)
        merged_data = np.asarray(merged_img.dataobj)

        expected = (data_a | data_b).astype(np.uint8)
        np.testing.assert_array_equal(merged_data, expected)

    def test_merge_one_none_a_only(self, tmp_path: Path) -> None:
        """When only mask_a is provided, output is mask_a passthrough."""
        from pmriprep.utils.mask_utils import MergeMasks

        data = np.ones((5, 5, 5), dtype=np.uint8)
        mask_a = _create_test_mask(data=data, tmpdir=str(tmp_path), fname="a.nii.gz")

        merge = MergeMasks()
        merge.inputs.mask_a = mask_a
        # mask_b not set

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            result = merge.run()
        finally:
            os.chdir(old_cwd)

        assert result.outputs.merged_mask == mask_a

    def test_merge_one_none_b_only(self, tmp_path: Path) -> None:
        """When only mask_b is provided, output is mask_b passthrough."""
        from pmriprep.utils.mask_utils import MergeMasks

        data = np.ones((5, 5, 5), dtype=np.uint8)
        mask_b = _create_test_mask(data=data, tmpdir=str(tmp_path), fname="b.nii.gz")

        merge = MergeMasks()
        merge.inputs.mask_b = mask_b

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            result = merge.run()
        finally:
            os.chdir(old_cwd)

        assert result.outputs.merged_mask == mask_b

    def test_merge_both_none(self, tmp_path: Path) -> None:
        """When neither mask is provided, output is None."""
        from pmriprep.utils.mask_utils import MergeMasks

        merge = MergeMasks()

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            result = merge.run()
        finally:
            os.chdir(old_cwd)

        assert result.outputs.merged_mask is None

    def test_merge_geometry_mismatch(self, tmp_path: Path) -> None:
        """When masks have different geometries, mask_b is resampled to mask_a."""
        from pmriprep.utils.mask_utils import MergeMasks

        # mask_a: 5x5x5, 2mm isotropic
        affine_a = np.diag([2.0, 2.0, 2.0, 1.0])
        data_a = np.zeros((5, 5, 5), dtype=np.uint8)
        data_a[0:2, 0:2, 0:2] = 1
        mask_a = _create_test_mask(data=data_a, affine=affine_a, tmpdir=str(tmp_path), fname="a.nii.gz")

        # mask_b: 10x10x10, 1mm isotropic
        affine_b = np.diag([1.0, 1.0, 1.0, 1.0])
        data_b = np.zeros((10, 10, 10), dtype=np.uint8)
        data_b[3:7, 3:7, 3:7] = 1
        mask_b = _create_test_mask(data=data_b, affine=affine_b, tmpdir=str(tmp_path), fname="b.nii.gz")

        merge = MergeMasks()
        merge.inputs.mask_a = mask_a
        merge.inputs.mask_b = mask_b

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            result = merge.run()
        finally:
            os.chdir(old_cwd)

        merged_img = nib.load(result.outputs.merged_mask)
        # Output should have same geometry as mask_a
        assert merged_img.shape == (5, 5, 5)
        np.testing.assert_allclose(merged_img.affine, affine_a)


