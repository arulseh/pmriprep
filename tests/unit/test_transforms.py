"""Tests for pmriprep.utils.transforms — FSL .mat to ITK X5 (.h5) conversion."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _make_nifti(path: Path, shape: tuple = (10, 10, 10)) -> str:
    """Create a minimal NIfTI for testing."""
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine=np.eye(4))
    nib.save(img, str(path))
    return str(path)


def _make_fsl_mat(path: Path) -> str:
    """Create a synthetic FSL FLIRT 4x4 identity matrix."""
    np.savetxt(str(path), np.eye(4))
    return str(path)


class TestFslToX5:
    """Test fsl_to_x5 conversion utility."""

    def test_roundtrip_identity(self, tmp_path: Path) -> None:
        """Converting identity .mat to .h5 and loading preserves the identity."""
        from nitransforms.linear import Affine

        from pmriprep.utils.transforms import fsl_to_x5

        mat = _make_fsl_mat(tmp_path / "identity.mat")
        ref = _make_nifti(tmp_path / "ref.nii.gz")
        mov = _make_nifti(tmp_path / "mov.nii.gz")

        h5_path = fsl_to_x5(mat, ref, mov)

        assert h5_path.endswith(".h5")
        assert Path(h5_path).exists()

        loaded = Affine.from_filename(h5_path)
        np.testing.assert_allclose(loaded.matrix, np.eye(4), atol=1e-6)

    def test_roundtrip_nonidentity(self, tmp_path: Path) -> None:
        """The composite contains the converted affine, not a trailing identity."""
        from nitransforms.linear import Affine

        from pmriprep.utils.transforms import fsl_to_x5

        moving = _make_nifti(tmp_path / "moving.nii.gz")
        reference = _make_nifti(tmp_path / "reference.nii.gz")
        matrix = np.eye(4)
        matrix[:3, 3] = [4.0, -3.0, 2.0]
        mat = tmp_path / "nonidentity.mat"
        Affine(matrix, reference=reference).to_filename(
            mat, fmt="fsl", moving=moving
        )

        loaded = Affine.from_filename(fsl_to_x5(mat, reference, moving))

        np.testing.assert_allclose(loaded.matrix, matrix, atol=1e-6)

    def test_ants_reads_composite(self, tmp_path: Path) -> None:
        """The emitted transform follows the ITK HDF5 composite contract."""
        if shutil.which("antsApplyTransforms") is None:
            pytest.skip("ANTs is not installed")

        from nitransforms.linear import Affine

        from pmriprep.utils.transforms import fsl_to_x5

        moving = _make_nifti(tmp_path / "moving.nii.gz")
        moving_data = np.zeros((10, 10, 10), dtype=np.float32)
        moving_data[5, 5, 5] = 1.0
        nib.save(nib.Nifti1Image(moving_data, np.eye(4)), moving)
        reference = _make_nifti(tmp_path / "reference.nii.gz")
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 0.0, 0.0]
        mat = tmp_path / "translation.mat"
        Affine(matrix, reference=reference).to_filename(
            mat, fmt="fsl", moving=moving
        )
        h5_path = fsl_to_x5(mat, reference, moving)

        result = subprocess.run(
            [
                "antsApplyTransforms",
                "-d", "3",
                "-i", moving,
                "-r", reference,
                "-o", str(tmp_path / "resampled.nii.gz"),
                "-n", "NearestNeighbor",
                "-t", h5_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        resampled = np.asarray(
            nib.load(tmp_path / "resampled.nii.gz").get_fdata()
        )
        assert np.count_nonzero(resampled) == 1
        assert not np.array_equal(resampled, moving_data)

    def test_sentinel_empty_string(self) -> None:
        """Empty string sentinel returns empty string."""
        from pmriprep.utils.transforms import fsl_to_x5

        assert fsl_to_x5("", "ref.nii.gz", "mov.nii.gz") == ""

    def test_sentinel_none(self) -> None:
        """None sentinel returns empty string."""
        from pmriprep.utils.transforms import fsl_to_x5

        assert fsl_to_x5(None, "ref.nii.gz", "mov.nii.gz") == ""

    def test_h5_written_next_to_mat(self, tmp_path: Path) -> None:
        """Output .h5 is in the same directory as the input .mat."""
        from pmriprep.utils.transforms import fsl_to_x5

        subdir = tmp_path / "work" / "node"
        subdir.mkdir(parents=True)
        mat = _make_fsl_mat(subdir / "xfm.mat")
        ref = _make_nifti(tmp_path / "ref.nii.gz")
        mov = _make_nifti(tmp_path / "mov.nii.gz")

        h5_path = fsl_to_x5(mat, ref, mov)

        assert Path(h5_path).parent == subdir
        assert Path(h5_path).name == "xfm.h5"
