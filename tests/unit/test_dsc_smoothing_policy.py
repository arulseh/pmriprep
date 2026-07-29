"""DSC smoothed-derivative policy tests."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from pmriprep.workflows.dsc._helpers import _smooth_maps


def _write_nifti(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray | None = None,
) -> str:
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)
    return str(path)


def test_smoothing_uses_brain_support_and_preserves_invalid_voxels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Smoothing neither dilutes brain edges nor fills failed-fit holes."""
    monkeypatch.chdir(tmp_path)
    data = np.zeros((7, 7, 7), dtype=np.float32)
    brain = np.zeros_like(data)
    brain[2:5, 2:5, 2:5] = 1.0
    data[brain > 0] = 10.0
    data[3, 3, 3] = np.nan
    map_file = _write_nifti(tmp_path / "mtt.nii.gz", data)
    mask_file = _write_nifti(tmp_path / "mask.nii.gz", brain)

    def _fake_fslmaths(command: list[str], **_kwargs) -> SimpleNamespace:
        assert command[0] == "fslmaths"
        source = nib.load(command[1])
        source_data = source.get_fdata()
        sigma = float(command[command.index("-s") + 1])
        output = gaussian_filter(source_data, sigma=sigma)
        nib.save(nib.Nifti1Image(output, source.affine, source.header), command[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_fslmaths)
    output = _smooth_maps({"mtt": map_file}, mask_file, 3.0)["mtt"]
    smoothed = nib.load(output).get_fdata()

    assert np.isnan(smoothed[3, 3, 3])
    assert np.allclose(smoothed[(brain > 0) & np.isfinite(data)], 10.0)
    assert np.all(smoothed[brain == 0] == 0.0)


def test_smoothing_rejects_mismatched_mask_shape(tmp_path: Path) -> None:
    """A mismatched mask fails before invoking the external smoother."""
    map_file = _write_nifti(tmp_path / "mtt.nii.gz", np.ones((3, 3, 3)))
    mask_file = _write_nifti(tmp_path / "mask.nii.gz", np.ones((2, 2, 2)))

    with np.testing.assert_raises_regex(ValueError, "Brain mask shape"):
        _smooth_maps({"mtt": map_file}, mask_file, 3.0)


def test_smoothing_rejects_mismatched_mask_affine(tmp_path: Path) -> None:
    """A same-shaped mask in a different grid cannot define smoothing support."""
    data = np.ones((3, 3, 3))
    map_file = _write_nifti(tmp_path / "mtt.nii.gz", data)
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    mask_file = _write_nifti(tmp_path / "mask.nii.gz", data, shifted)

    with np.testing.assert_raises_regex(ValueError, "Brain mask affine"):
        _smooth_maps({"mtt": map_file}, mask_file, 3.0)
