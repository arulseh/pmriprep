"""Characterization tests for DSC compute input preparation."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import ComputeDSCMaps
from pmriprep.utils.dsc_signal import compute_peak_height_legacy, compute_psr_legacy


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


def test_prepare_baseline_preserves_pass_through_k2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged baseline subtracts offsets and retains upstream K2 provenance."""
    concentration = np.broadcast_to(
        np.array([2.0, 2.0, 2.0, 2.0, 4.0, 6.0]), (2, 2, 1, 6)
    ).copy()
    concentration_file = _write_image(tmp_path / "concentration.nii.gz", concentration)
    mask_file = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 1), dtype=np.uint8))
    k2_file = _write_image(tmp_path / "upstream_k2.nii.gz", np.ones((2, 2, 1)))
    aif_file = tmp_path / "aif.txt"
    np.savetxt(aif_file, [1.0, 1.0, 1.0, 1.0, 3.0, 5.0])
    interface = ComputeDSCMaps(
        concentration_file=concentration_file,
        brain_mask_file=mask_file,
        aif_file=str(aif_file),
        baseline_frames=4,
        k2_map_input_file=k2_file,
    )
    monkeypatch.setattr(
        ComputeDSCMaps,
        "_detect_baseline_frames",
        lambda self, *args, **kwargs: (4, {"arrival_index": 4}),
    )

    result = interface._prepare_baseline_and_leakage(interface._load_compute_inputs())

    assert result.detected_frames == 4
    assert result.n_baseline == 4
    assert result.recomputed_raw_concentration is None
    assert result.k2_map_path == k2_file
    np.testing.assert_allclose(result.aif[:4], 0.0)
    np.testing.assert_allclose(result.concentration[..., :4], 0.0)
    assert result.detection_info["aif_baseline_offset"] == 1.0
    assert result.detection_info["conc_median_baseline_offset"] == 2.0


def test_prepare_baseline_recomputes_s0_and_k2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier detected baseline keeps S0, K2, and saved-map state coherent."""
    monkeypatch.chdir(tmp_path)
    concentration_file = _write_image(
        tmp_path / "concentration.nii.gz", np.zeros((2, 2, 1, 6), dtype=np.float32)
    )
    mask_file = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 1), dtype=np.uint8))
    signal = np.broadcast_to(
        np.array([100.0, 100.0, 80.0, 60.0, 50.0, 50.0]), (2, 2, 1, 6)
    ).copy()
    signal_file = _write_image(tmp_path / "signal.nii.gz", signal)
    aif_file = tmp_path / "aif.txt"
    np.savetxt(aif_file, [0.2, 0.2, 1.0, 2.0, 1.0, 0.5])
    interface = ComputeDSCMaps(
        concentration_file=concentration_file,
        brain_mask_file=mask_file,
        aif_file=str(aif_file),
        signal_file=signal_file,
        baseline_frames=4,
        tr=1.5,
        te=0.05,
    )
    monkeypatch.setattr(
        ComputeDSCMaps,
        "_detect_baseline_frames",
        lambda self, *args, **kwargs: (2, {"arrival_index": 2}),
    )

    expected_raw = -np.log(signal / 100.0) / 0.05

    class _K2Result:
        concentration_corrected = expected_raw + 3.0
        k2_map = np.full((2, 2, 1), 0.25)

    def _fake_k2(**kwargs: object) -> _K2Result:
        np.testing.assert_allclose(kwargs["concentration"], expected_raw)
        assert kwargs["baseline_frames"] == 2
        return _K2Result()

    monkeypatch.setattr("pmriprep.utils.dsc_preload.apply_k2_correction", _fake_k2)

    result = interface._prepare_baseline_and_leakage(interface._load_compute_inputs())

    np.testing.assert_allclose(result.recomputed_raw_concentration, expected_raw)
    np.testing.assert_allclose(result.concentration[..., :4].mean(axis=-1), 0.0)
    assert result.detected_frames == 2
    assert result.n_baseline == 4
    assert result.detection_info["s0_recomputed"] is True
    assert result.detection_info["s0_recomputed_frames"] == 2
    assert result.detection_info["k2_recomputed"] is True
    assert result.k2_map_path is not None
    np.testing.assert_allclose(nib.load(result.k2_map_path).get_fdata(), 0.25)


def test_compute_signal_metrics_preserves_legacy_maps() -> None:
    """Raw-signal PSR and peak height are forwarded without transformation."""
    signal = np.array([[[[100.0, 100.0, 70.0, 80.0, 90.0]]]])

    result = ComputeDSCMaps()._compute_signal_metrics(signal, (1, 1, 1), 2, 2)

    expected_psr = compute_psr_legacy(signal, 2, 2)
    expected_peak = compute_peak_height_legacy(signal, 2)
    np.testing.assert_allclose(result.psr, expected_psr)
    np.testing.assert_allclose(result.peak_height, expected_peak)
    assert result.qc == {}


def test_compute_signal_metrics_without_signal_returns_zero_maps() -> None:
    """Unavailable raw signal retains zero maps and explicit QC provenance."""
    result = ComputeDSCMaps()._compute_signal_metrics(None, (2, 3, 1), 4, 5)

    np.testing.assert_array_equal(result.psr, np.zeros((2, 3, 1)))
    np.testing.assert_array_equal(result.peak_height, np.zeros((2, 3, 1)))
    assert result.qc == {"psr_ph_from_signal": False}
