"""Tests for registered VFA T1 transfer into DCE space."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.workflows.dce._helpers import _transform_t1_to_dce


def _write_image(path: Path, data: np.ndarray, affine: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)
    return str(path)


def test_poor_vfa_registration_rejects_t1_map(tmp_path: Path) -> None:
    """A poor magnitude-image registration must trigger population-T1 fallback."""
    t1_file = _write_image(tmp_path / "t1.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    matrix = tmp_path / "vfa_to_dce.mat"
    matrix.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    assert _transform_t1_to_dce(
        t1_file, ref_file, str(matrix), {"qc_status": "POOR"},
    ) is None


def test_unknown_vfa_registration_qc_fails_loudly(tmp_path: Path) -> None:
    """Unavailable QC must not silently change concentration quantification."""
    t1_file = _write_image(tmp_path / "t1.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    matrix = tmp_path / "vfa_to_dce.mat"
    matrix.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    with pytest.raises(RuntimeError, match="QC could not be evaluated"):
        _transform_t1_to_dce(
            t1_file, ref_file, str(matrix), {"qc_status": "UNKNOWN"},
        )


def test_available_vfa_t1_requires_registration_inputs(tmp_path: Path) -> None:
    """A measured T1 map must not fall back because workflow wiring is absent."""
    t1_file = _write_image(tmp_path / "t1.nii.gz", np.ones((2, 2, 2)), np.eye(4))

    with pytest.raises(RuntimeError, match="registration transform is missing"):
        _transform_t1_to_dce(t1_map_file=t1_file)


def test_registered_t1_preserves_values_over_valid_support(
    tmp_path: Path, monkeypatch,
) -> None:
    """Support-normalized interpolation does not dilute valid T1 values."""
    from nibabel.processing import resample_from_to

    source = np.full((2, 2, 2), np.nan, dtype=np.float32)
    source[0] = 1200.0
    t1_file = _write_image(
        tmp_path / "t1.nii.gz", source, np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((4, 4, 4)), np.eye(4))
    matrix = tmp_path / "vfa_to_dce.mat"
    matrix.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    def _fake_run(command, **_kwargs):
        source_img = nib.load(command[command.index("-in") + 1])
        reference_img = nib.load(command[command.index("-ref") + 1])
        interpolation = command[command.index("-interp") + 1]
        order = 0 if interpolation == "nearestneighbour" else 1
        output = resample_from_to(source_img, reference_img, order=order)
        nib.save(output, command[command.index("-out") + 1])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.chdir(tmp_path)
    result = _transform_t1_to_dce(
        t1_file, ref_file, str(matrix), {"qc_status": "PASS"},
    )

    assert result is not None
    output = nib.load(result).get_fdata()
    assert output.shape == (4, 4, 4)
    assert np.allclose(output[np.isfinite(output)], 1200.0)
    assert np.isnan(output).any()
