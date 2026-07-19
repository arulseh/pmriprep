"""Tests for pmriprep.utils.transforms — FSL .mat to ITK X5 (.h5) conversion."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


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
