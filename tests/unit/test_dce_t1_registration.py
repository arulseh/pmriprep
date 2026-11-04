"""Tests for registered VFA T1 transfer into DCE space."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.workflows.dce._helpers import (
    _apply_registered_b1_ratio,
    _transform_b1_to_dce,
    _transform_b1_to_vfa,
    _transform_t1_to_dce,
)


def _write_image(path: Path, data: np.ndarray, affine: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)
    return str(path)


def test_poor_vfa_registration_rejects_t1_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected registration uses the configured fallback during conversion."""
    t1_file = _write_image(tmp_path / "t1.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    matrix = tmp_path / "vfa_to_dce.mat"
    matrix.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    retained_t1 = _transform_t1_to_dce(
        t1_file, ref_file, str(matrix), {"qc_status": "POOR"},
    )
    assert retained_t1 is None
    from pmriprep.workflows.dce._helpers import _signal_to_concentration_node
    from pmriprep.utils.dce_signal import signal_to_concentration

    monkeypatch.chdir(tmp_path)
    signal = np.full((2, 2, 2, 10), 100.)
    signal[..., 5:] = 130.
    signal_file = _write_image(tmp_path / "dce.nii.gz", signal, np.eye(4))
    output, qc = _signal_to_concentration_node(
        signal_file, ref_file, retained_t1, None, 15., .005, 3., 3.8, 5, False,
        t1_registration_qc={"qc_status": "POOR"}, t1_default_ms=850.,
    )
    expected = signal_to_concentration(signal, 15., .005, t1_default_ms=850.,
                                       r1_relaxivity=3.8, baseline_frames=5,
                                       convert_to_plasma=False)
    np.testing.assert_allclose(nib.load(output).get_fdata(), expected.concentration, rtol=1e-6)
    assert qc["assumed_t1_ms"] == 850.
    assert qc["assumed_t1_policy"] == "uniform_configured"


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
    assert np.count_nonzero(np.isfinite(output)) > 0
    assert np.allclose(output[np.isfinite(output)], 1200.0)
    assert np.isnan(output).any()


def test_registered_b1_preserves_ratio_over_valid_support(
    tmp_path: Path, monkeypatch,
) -> None:
    """Magnitude-derived transfer preserves B1+ ratios at support boundaries."""
    from nibabel.processing import resample_from_to

    source = np.zeros((2, 2, 2), dtype=np.float32)
    source[0] = 0.9
    b1_file = _write_image(
        tmp_path / "b1.nii.gz", source, np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((4, 4, 4)), np.eye(4))
    matrix = tmp_path / "afi_to_target.mat"
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
    result = _apply_registered_b1_ratio(
        b1_file, ref_file, str(matrix), {"qc_status": "PASS"},
        "b1_registered.nii.gz",
    )

    assert result is not None
    output = nib.load(result).get_fdata()
    assert np.count_nonzero(output > 0) > 0
    assert np.allclose(output[output > 0], 0.9)
    assert np.any(output == 0)


def test_precomputed_b1_is_regridded_before_dce_to_vfa_transform(
    tmp_path: Path, monkeypatch,
) -> None:
    """The inverse FLIRT matrix is applied in its DCE moving-image frame."""
    from nibabel.processing import resample_from_to

    b1_file = _write_image(
        tmp_path / "b1_coarse.nii.gz",
        np.full((2, 2, 2), 0.9),
        np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    dce_file = _write_image(
        tmp_path / "dce_reference.nii.gz", np.ones((4, 4, 4)), np.eye(4),
    )
    vfa_file = _write_image(
        tmp_path / "vfa_reference.nii.gz", np.ones((4, 4, 4)), np.eye(4),
    )
    matrix = tmp_path / "dce_to_vfa.mat"
    matrix.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    applied_source_grids: list[tuple[tuple[int, ...], np.ndarray]] = []

    def _fake_run(command, **_kwargs):
        source_img = nib.load(command[command.index("-in") + 1])
        reference_img = nib.load(command[command.index("-ref") + 1])
        applied_source_grids.append((source_img.shape, source_img.affine.copy()))
        interpolation = command[command.index("-interp") + 1]
        order = 0 if interpolation == "nearestneighbour" else 1
        nib.save(
            resample_from_to(source_img, reference_img, order=order),
            command[command.index("-out") + 1],
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.chdir(tmp_path)
    result = _transform_b1_to_vfa(
        b1_file,
        vfa_file,
        str(matrix),
        {"qc_status": "PASS"},
        registration_required=True,
        registration_source_file=dce_file,
    )

    assert result is not None
    assert applied_source_grids
    assert all(shape == (4, 4, 4) for shape, _ in applied_source_grids)
    assert all(np.allclose(affine, np.eye(4)) for _, affine in applied_source_grids)


def test_poor_b1_registration_uses_nominal_flip_angle(tmp_path: Path) -> None:
    """A poor AFI magnitude registration rejects the measured correction."""
    b1_file = _write_image(tmp_path / "b1.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    matrix = tmp_path / "afi_to_target.mat"
    matrix.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    assert _apply_registered_b1_ratio(
        b1_file, ref_file, str(matrix), {"qc_status": "POOR"}, "out.nii.gz",
    ) is None


def test_unknown_b1_registration_qc_fails_loudly(tmp_path: Path) -> None:
    """Unavailable AFI QC cannot silently disable measured B1 correction."""
    b1_file = _write_image(tmp_path / "b1.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    matrix = tmp_path / "afi_to_target.mat"
    matrix.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    with pytest.raises(RuntimeError, match="QC could not be evaluated"):
        _apply_registered_b1_ratio(
            b1_file, ref_file, str(matrix), {"qc_status": "UNKNOWN"},
            "out.nii.gz",
        )


def test_afi_b1_requires_registration_matrix(tmp_path: Path) -> None:
    """AFI must not silently enter the precomputed-map header path."""
    b1_file = _write_image(tmp_path / "b1.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((2, 2, 2)), np.eye(4))

    with pytest.raises(RuntimeError, match="registration transform is missing"):
        _transform_b1_to_dce(
            b1_file, ref_file, registration_required=True,
        )


def test_registered_b1_without_target_overlap_is_rejected(
    tmp_path: Path, monkeypatch,
) -> None:
    """Registered transfer with no support falls back to nominal flip angles."""
    from nibabel.processing import resample_from_to

    source_affine = np.eye(4)
    source_affine[:3, 3] = 1000.0
    b1_file = _write_image(
        tmp_path / "b1.nii.gz", np.ones((2, 2, 2)), source_affine,
    )
    ref_file = _write_image(tmp_path / "ref.nii.gz", np.ones((2, 2, 2)), np.eye(4))
    matrix = tmp_path / "afi_to_target.mat"
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
    assert _apply_registered_b1_ratio(
        b1_file, ref_file, str(matrix), {"qc_status": "PASS"}, "out.nii.gz",
    ) is None
