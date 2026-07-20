"""Characterization tests for DSC compute input preparation."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import ComputeDSCMaps


def _write_image(path: Path, data: np.ndarray) -> str:
    """Write a test NIfTI image and return its path."""
    nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 3.0, 1.0])), path)
    return str(path)


def test_load_compute_inputs_preserves_arrays_and_settings(tmp_path: Path) -> None:
    """Loaded arrays, geometry, optional signal, and traits remain coherent."""
    concentration = np.arange(3 * 4 * 2 * 6, dtype=np.float32).reshape(3, 4, 2, 6)
    mask = np.zeros((3, 4, 2), dtype=np.uint8)
    mask[1:, 1:, :] = 1
    signal = concentration + 100.0
    aif = np.linspace(-0.1, 2.0, 6)

    concentration_file = _write_image(tmp_path / "concentration.nii.gz", concentration)
    mask_file = _write_image(tmp_path / "mask.nii.gz", mask)
    signal_file = _write_image(tmp_path / "signal.nii.gz", signal)
    aif_file = tmp_path / "aif.txt"
    np.savetxt(aif_file, aif)

    interface = ComputeDSCMaps(
        concentration_file=concentration_file,
        brain_mask_file=mask_file,
        aif_file=str(aif_file),
        signal_file=signal_file,
        method="cpi",
        tr=1.2,
        svd_threshold=0.15,
        oi_threshold=0.07,
        cpi_control_points=4,
        cpi_infer_cpt=False,
        cpi_infer_delay=True,
        inference="map",
        mcmc_samples=321,
        baseline_frames=7,
        tail_frames=8,
        te=0.03,
    )

    result = interface._load_compute_inputs()

    np.testing.assert_allclose(result.concentration, concentration)
    np.testing.assert_array_equal(result.mask, mask)
    np.testing.assert_array_equal(result.mask_bool, mask.astype(bool))
    np.testing.assert_allclose(result.signal, signal)
    np.testing.assert_allclose(result.aif, aif)
    np.testing.assert_allclose(result.affine, np.diag([2.0, 2.0, 3.0, 1.0]))
    assert result.header.get_data_shape() == concentration.shape
    assert result.method == "cpi"
    assert result.dt == pytest.approx(1.2)
    assert result.svd_threshold == pytest.approx(0.15)
    assert result.oi_threshold == pytest.approx(0.07)
    assert result.cpi_control_points == 4
    assert result.cpi_infer_cpt is False
    assert result.cpi_infer_delay is True
    assert result.inference == "map"
    assert result.mcmc_samples == 321
    assert result.baseline_frames == 7
    assert result.tail_frames == 8
    assert result.te == pytest.approx(0.03)


def test_load_compute_inputs_rejects_off_grid_mask(tmp_path: Path) -> None:
    """Native-grid validation retains its actionable failure message."""
    concentration_file = _write_image(
        tmp_path / "concentration.nii.gz", np.zeros((3, 4, 2, 6), dtype=np.float32)
    )
    mask_file = _write_image(tmp_path / "mask.nii.gz", np.ones((3, 3, 2), dtype=np.uint8))
    aif_file = tmp_path / "aif.txt"
    np.savetxt(aif_file, np.arange(6, dtype=np.float64))
    interface = ComputeDSCMaps(
        concentration_file=concentration_file,
        brain_mask_file=mask_file,
        aif_file=str(aif_file),
    )

    with pytest.raises(RuntimeError, match="must be in native DSC space"):
        interface._load_compute_inputs()


def test_prepare_motion_weighting_matches_fitting_array_and_qc() -> None:
    """Motion QC describes the exact weights forwarded to parametric fitting."""
    header = SimpleNamespace(get_zooms=lambda: (2.0, 2.5, 3.0, 1.0))
    interface = ComputeDSCMaps(
        fd_values=[0.0, 0.2, 1.0, 0.1, 0.8, 0.0],
        baseline_frames=2,
    )

    result = interface._prepare_motion_weighting("cpi", header, 6, 1.5)

    assert result.weights is not None
    assert result.weights.shape == (6,)
    assert result.qc["applied"] is True
    assert result.qc["n_downweighted_frames"] == int(np.sum(result.weights < 0.95))
    assert result.qc["min_weight"] == float(np.min(result.weights))
    assert result.qc["first_pass_weights"] == result.weights[2:6].tolist()


@pytest.mark.parametrize("method", ["sSVD", "oSVD", "tikhonov", "cpi"])
def test_prepare_motion_weighting_disabled_paths(method: str) -> None:
    """SVD methods and missing FD values retain stable disabled state."""
    header = SimpleNamespace(get_zooms=lambda: (2.0, 2.5, 3.0, 1.0))
    interface = ComputeDSCMaps()
    if method != "cpi":
        interface.inputs.fd_values = [0.0, 0.5, 0.0]

    result = interface._prepare_motion_weighting(method, header, 3, 1.5)

    assert result.weights is None
    assert result.qc == {"applied": False}
