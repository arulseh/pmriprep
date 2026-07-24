"""Characterization tests for DSC compute input preparation."""

import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import ComputeDSCMaps, _DSCDeconvolutionMaps
from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION, EPSILON_SIGNAL
from pmriprep.utils.dsc_signal import (
    compute_peak_height_legacy,
    compute_psr_legacy,
    compute_ttp_legacy,
)


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

    result = interface._prepare_motion_weighting("cpi", header, 6, 1.5, 3)

    assert result.weights is not None
    assert result.weights.shape == (6,)
    assert result.qc["applied"] is True
    assert result.qc["n_downweighted_frames"] == int(np.sum(result.weights < 0.95))
    assert result.qc["min_weight"] == float(np.min(result.weights))
    assert result.qc["first_pass_weights"] == result.weights[3:6].tolist()


@pytest.mark.parametrize("method", ["sSVD", "oSVD", "tikhonov", "cpi"])
def test_prepare_motion_weighting_disabled_paths(method: str) -> None:
    """SVD methods and missing FD values retain stable disabled state."""
    header = SimpleNamespace(get_zooms=lambda: (2.0, 2.5, 3.0, 1.0))
    interface = ComputeDSCMaps()
    if method != "cpi":
        interface.inputs.fd_values = [0.0, 0.5, 0.0]

    result = interface._prepare_motion_weighting(method, header, 3, 1.5, 2)

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


def test_prepare_baseline_replaces_stale_k2_after_later_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later detected baseline saves the K2 map fitted to that baseline."""
    monkeypatch.chdir(tmp_path)
    concentration_file = _write_image(
        tmp_path / "concentration.nii.gz", np.zeros((2, 2, 1, 7), dtype=np.float32)
    )
    mask_file = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 1), dtype=np.uint8))
    signal = np.broadcast_to(
        np.array([100.0, 100.0, 100.0, 100.0, 100.0, 70.0, 60.0]),
        (2, 2, 1, 7),
    ).copy()
    signal_file = _write_image(tmp_path / "signal.nii.gz", signal)
    stale_k2_file = _write_image(
        tmp_path / "upstream_k2.nii.gz", np.full((2, 2, 1), 9.0)
    )
    aif_file = tmp_path / "aif.txt"
    np.savetxt(aif_file, [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0])
    interface = ComputeDSCMaps(
        concentration_file=concentration_file,
        brain_mask_file=mask_file,
        aif_file=str(aif_file),
        signal_file=signal_file,
        baseline_frames=2,
        k2_map_input_file=stale_k2_file,
        tr=1.5,
        te=0.05,
    )
    monkeypatch.setattr(
        ComputeDSCMaps,
        "_detect_baseline_frames",
        lambda self, *args, **kwargs: (5, {"arrival_index": 5}),
    )

    expected_raw = -np.log(signal / 100.0) / 0.05

    class _K2Result:
        concentration_corrected = expected_raw + 3.0
        k2_map = np.full((2, 2, 1), 0.5)

    monkeypatch.setattr(
        "pmriprep.utils.dsc_preload.apply_k2_correction",
        lambda **kwargs: _K2Result(),
    )

    result = interface._prepare_baseline_and_leakage(interface._load_compute_inputs())

    assert result.detected_frames == 5
    assert result.k2_map_path is not None
    assert result.k2_map_path != stale_k2_file
    np.testing.assert_allclose(nib.load(result.k2_map_path).get_fdata(), 0.5)


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


def test_select_raw_concentration_preserves_source_precedence(tmp_path: Path) -> None:
    """Recomputed, configured raw, and corrected sources retain their order."""
    corrected = np.zeros((1, 1, 1, 4), dtype=np.float64)
    configured = np.ones_like(corrected)
    recomputed = np.full_like(corrected, 2.0)
    raw_file = _write_image(tmp_path / "raw.nii.gz", configured)
    interface = ComputeDSCMaps(raw_concentration_file=raw_file)

    assert interface._select_raw_concentration(corrected, recomputed) is recomputed
    np.testing.assert_allclose(
        interface._select_raw_concentration(corrected, None), configured
    )
    assert ComputeDSCMaps()._select_raw_concentration(corrected, None) is corrected


def test_compute_timing_maps_uses_unclipped_ttp_for_nawm_reference(
    tmp_path: Path,
) -> None:
    """NAWM-relative timing derives from raw TTP before output clipping."""
    n_voxels = 102
    concentration = np.zeros((n_voxels, 1, 1, 8), dtype=np.float64)
    concentration[:101, 0, 0, 4] = 2.0
    concentration[101, 0, 0, 6] = 2.0
    mask = np.ones((n_voxels, 1, 1), dtype=bool)
    nawm = np.zeros(mask.shape, dtype=np.uint8)
    nawm[:101] = 1
    nawm_file = _write_image(tmp_path / "nawm.nii.gz", nawm)
    interface = ComputeDSCMaps(nawm_mask_file=nawm_file)

    result = interface._compute_timing_maps(
        concentration,
        mask,
        detected_baseline=2,
        detection_info={"arrival_index": 3},
        dt=2.0,
    )

    raw_ttp = compute_ttp_legacy(concentration, 2, 2.0)
    np.testing.assert_allclose(result.ttp, np.clip(raw_ttp, 0.0, 60.0))
    np.testing.assert_allclose(
        result.ttp_corrected, np.clip(raw_ttp - 6.0, 0.0, 30.0)
    )
    assert result.relative_ttp is not None
    np.testing.assert_allclose(result.relative_ttp, raw_ttp - np.median(raw_ttp[:101]))
    assert result.relative_ttp[101, 0, 0] == pytest.approx(4.0)
    assert result.aif_bat_seconds == pytest.approx(6.0)


def test_compute_timing_maps_requires_more_than_100_nawm_voxels(
    tmp_path: Path,
) -> None:
    """The existing strict NAWM sample-size gate remains unchanged."""
    concentration = np.zeros((100, 1, 1, 6), dtype=np.float64)
    concentration[..., 4] = 1.0
    mask = np.ones((100, 1, 1), dtype=bool)
    nawm_file = _write_image(tmp_path / "nawm.nii.gz", mask.astype(np.uint8))

    result = ComputeDSCMaps(nawm_mask_file=nawm_file)._compute_timing_maps(
        concentration, mask, 2, {"arrival_index": 2}, 1.5
    )

    assert result.relative_ttp is None


def test_compute_auc_maps_preserves_symmetric_measured_aif_ratio(
    tmp_path: Path,
) -> None:
    """Tissue and measured AIF share BAT window, baseline subtraction, and dt."""
    corrected = np.array([[[[0.0, 0.0, 1.0, 2.0, 1.0, 0.0]]]])
    raw = corrected + 5.0
    measured_aif = np.array([1.0, 1.0, 3.0, 5.0, 3.0, 1.0])
    measured_file = tmp_path / "measured_aif.txt"
    np.savetxt(measured_file, measured_aif)
    interface = ComputeDSCMaps(
        measured_aif_file=str(measured_file), preload_classification="none"
    )

    result = interface._compute_auc_maps(
        raw,
        corrected,
        n_baseline=2,
        detected_baseline=2,
        detection_info={"arrival_index": 2},
        dt=1.0,
    )

    expected_raw = np.trapezoid(raw - raw[..., :2].mean(axis=-1, keepdims=True), axis=-1)
    expected_corrected = np.trapezoid(corrected, axis=-1)
    expected_aif_auc = np.trapezoid((measured_aif - 1.0)[2:])
    expected_tissue_auc = np.trapezoid(corrected[..., 2:], axis=-1)
    np.testing.assert_allclose(result.raw, expected_raw)
    np.testing.assert_allclose(result.corrected, expected_corrected)
    np.testing.assert_allclose(result.difference, expected_raw - expected_corrected)
    np.testing.assert_allclose(result.cbv, expected_tissue_auc / expected_aif_auc)
    assert result.qc == {
        "cbv_auc_bat_frame": 2,
        "cbv_auc_end_frame": 6,
        "cbv_auc_aif_integral": round(float(expected_aif_auc), 6),
    }


def test_compute_auc_maps_near_zero_measured_aif_returns_nan(
    tmp_path: Path,
) -> None:
    """A nonpositive measured-AIF integral retains unavailable CBV semantics."""
    measured_file = tmp_path / "measured_aif.txt"
    np.savetxt(measured_file, np.ones(6))
    concentration = np.ones((2, 1, 1, 6), dtype=np.float64)

    result = ComputeDSCMaps(measured_aif_file=str(measured_file))._compute_auc_maps(
        concentration, concentration, 2, 2, {"arrival_index": 2}, 1.0
    )

    assert np.all(np.isnan(result.cbv))
    assert result.qc["cbv_auc_aif_integral"] == 0.0


def test_compute_auc_maps_missing_aif_and_preload_warning() -> None:
    """Missing denominator and administered preload retain separate QC flags."""
    concentration = np.ones((1, 1, 1, 6), dtype=np.float64)

    result = ComputeDSCMaps(preload_classification="half_dose")._compute_auc_maps(
        concentration, concentration, 2, 2, {"arrival_index": 2}, 1.0
    )

    assert np.all(np.isnan(result.cbv))
    assert result.qc["cbv_auc_measured_aif_missing"] is True
    assert "residual leakage" in result.qc["cbv_auc_preload_warning"]


def test_compute_quality_maps_preserves_noise_conventions() -> None:
    """CNR uses ddof=0 while signal SNR and tSNR retain ddof=1."""
    raw = np.array(
        [
            [[[0.0, 1.0, 2.0, 6.0, 3.0]]],
            [[[1.0, 1.0, 1.0, 4.0, 2.0]]],
        ]
    )
    signal = np.array(
        [
            [[[100.0, 102.0, 98.0, 75.0, 85.0]]],
            [[[50.0, 50.0, 50.0, 40.0, 45.0]]],
        ]
    )

    result = ComputeDSCMaps()._compute_quality_maps(raw, signal, 3)

    raw_std = np.std(raw[..., :3], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_cnr = np.where(
            raw_std > EPSILON_CONCENTRATION, np.max(raw, axis=-1) / raw_std, 0.0
        )
    signal_baseline = signal[..., :3]
    signal_std = np.std(signal_baseline, axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_snr = np.where(
            signal_std > EPSILON_SIGNAL,
            np.mean(signal_baseline, axis=-1) / signal_std,
            0.0,
        )
    all_std = np.std(signal, axis=-1, ddof=1)
    expected_tsnr = np.where(
        all_std > EPSILON_SIGNAL, np.mean(signal, axis=-1) / all_std, 0.0
    )
    np.testing.assert_allclose(result.cnr, expected_cnr)
    np.testing.assert_allclose(result.snr, expected_snr)
    np.testing.assert_allclose(result.tsnr, expected_tsnr)


def test_compute_quality_maps_without_signal_returns_zero_snr() -> None:
    """Missing raw signal leaves concentration CNR intact and signal maps zero."""
    raw = np.array([[[[0.0, 1.0, 2.0, 5.0]]]])

    result = ComputeDSCMaps()._compute_quality_maps(raw, None, 3)

    assert result.cnr[0, 0, 0] > 0.0
    np.testing.assert_array_equal(result.snr, np.zeros((1, 1, 1)))
    np.testing.assert_array_equal(result.tsnr, np.zeros((1, 1, 1)))


@pytest.mark.parametrize("method", ["sSVD", "oSVD", "tikhonov"])
def test_finalize_mtt_uses_auc_cbv_for_svd_family(method: str) -> None:
    """SVD-family MTT uses measured-AIF CBV divided by unscaled CBF."""
    fitted = np.full((4,), 99.0)
    cbf = np.array([0.5, 0.0, 0.25, 0.1])
    auc_cbv = np.array([2.0, 3.0, np.nan, 1.5])

    result = ComputeDSCMaps._finalize_mtt(method, fitted, cbf, auc_cbv)

    np.testing.assert_allclose(result, [4.0, 0.0, 0.0, 15.0])


@pytest.mark.parametrize("method", ["vascular", "cpi"])
def test_finalize_mtt_preserves_parametric_result(method: str) -> None:
    """Parametric methods retain the MTT returned by their fitted model."""
    fitted = np.array([2.0, 4.0])

    result = ComputeDSCMaps._finalize_mtt(
        method, fitted, np.array([0.5, 0.5]), np.array([1.0, 1.0])
    )

    assert result is fitted


def test_summarize_dsc_maps_preserves_population_statistics_and_provenance() -> None:
    """Final map QC uses the brain domain and carries baseline provenance."""
    mask = np.array([True, True, False])
    qc = ComputeDSCMaps._summarize_dsc_maps(
        rel_cbf=np.array([10.0, 30.0, 999.0]),
        rel_cbv=np.array([1.0, 3.0, 999.0]),
        rel_cbv_auc=np.array([-2.0, 4.0, 999.0]),
        mtt=np.array([5.0, 9.0, 999.0]),
        tmax=np.array([3.0, 7.0, 999.0]),
        mask_bool=mask,
        aif_level="atlas",
        preload_class="adequate",
        baseline_info={"baseline_frames_detected": 4},
        aif_bat_seconds=6.0,
    )

    assert qc["cbf_mean"] == pytest.approx(20.0)
    assert qc["cbf_std"] == pytest.approx(10.0)
    assert qc["cbv_mean"] == pytest.approx(2.0)
    assert qc["cbv_auc_median"] == pytest.approx(1.0)
    assert qc["cbv_auc_finite_voxels"] == 2
    assert qc["cbv_auc_negative_voxels"] == 1
    assert qc["cbv_auc_negative_fraction"] == pytest.approx(0.5)
    assert qc["mtt_mean"] == pytest.approx(7.0)
    assert qc["tmax_median"] == pytest.approx(5.0)
    assert qc["n_voxels"] == 2
    assert qc["baseline_frames_detected"] == 4
    assert qc["aif_bat_seconds"] == pytest.approx(6.0)
    assert qc["cbv_literature_range_applicable"] is True


def test_summarize_dsc_maps_empty_synthetic_domain() -> None:
    """Empty masks retain zero summaries and synthetic-CBV applicability flag."""
    empty = np.zeros(2)
    qc = ComputeDSCMaps._summarize_dsc_maps(
        empty,
        empty,
        empty,
        empty,
        empty,
        np.zeros(2, dtype=bool),
        "synthetic",
        "unknown",
        {},
        0.0,
    )

    assert qc["n_voxels"] == 0
    assert qc["cbf_mean"] == 0.0
    assert qc["cbv_auc_median"] == 0.0
    assert qc["cbv_auc_finite_voxels"] == 0
    assert qc["cbv_auc_negative_voxels"] == 0
    assert qc["cbv_auc_negative_fraction"] == 0.0
    assert qc["cbv_literature_range_applicable"] is False


def test_scale_cbf_uncertainty_preserves_optionality_and_units() -> None:
    """Available CBF uncertainty maps convert to per-minute units independently."""
    std = np.array([0.1, 0.2])
    upper = np.array([0.3, 0.4])

    scaled_std, scaled_lower, scaled_upper = ComputeDSCMaps._scale_cbf_uncertainty(
        std, None, upper
    )

    np.testing.assert_allclose(scaled_std, [6.0, 12.0])
    assert scaled_lower is None
    np.testing.assert_allclose(scaled_upper, [18.0, 24.0])
    np.testing.assert_allclose(std, [0.1, 0.2])


def test_finalize_dsc_maps_preserves_ranges_units_and_nan_policy() -> None:
    """Final maps retain established clipping, CBF units, and timing NaN policy."""
    values = np.array([-1.0, 2.0, 2000.0])
    mask = np.array([True, True, False])

    result = ComputeDSCMaps._finalize_dsc_maps(
        cbf=np.array([-1.0, 2.0, 200.0]),
        cbv=np.array([-1.0, 12.0, 2.0]),
        cbv_auc=np.array([np.nan, 4.0, 8.0]),
        mtt=np.array([-1.0, 40.0, 5.0]),
        tmax=np.array([-1.0, 70.0, 5.0]),
        ttp=values,
        ttp_corrected=np.array([np.nan, 3.0, 4.0]),
        relative_ttp=np.array([np.nan, -2.0, 4.0]),
        auc_raw=values,
        auc_corrected=values + 1.0,
        auc_difference=values + 2.0,
        cnr=values,
        snr=values,
        tsnr=values,
        psr=np.array([-1.0, 250.0, 5.0]),
        peak_height=np.array([-1.0, 120.0, 5.0]),
        delay=np.array([1.0, 2.0, 3.0]),
        residues=np.arange(9.0).reshape(3, 3),
        cbf_std=np.array([0.1, 0.2, 0.3]),
        cbf_ci_lower=None,
        cbf_ci_upper=np.array([0.2, 0.3, 0.4]),
        mask_bool=mask,
    )

    np.testing.assert_allclose(result.rel_cbf, [0.0, 100.0, 0.0])
    np.testing.assert_allclose(result.rel_cbv, [0.0, 10.0, 0.0])
    assert np.isnan(result.rel_cbv_auc[0])
    np.testing.assert_allclose(result.mtt, [0.0, 30.0, 0.0])
    np.testing.assert_allclose(result.tmax, [0.0, 60.0, 0.0])
    np.testing.assert_allclose(result.psr, [0.0, 200.0, 0.0])
    np.testing.assert_allclose(result.peak_height, [0.0, 100.0, 0.0])
    np.testing.assert_allclose(result.ttp_corrected, [0.0, 3.0, 0.0])
    np.testing.assert_allclose(result.relative_ttp, [0.0, -2.0, 0.0])
    np.testing.assert_allclose(result.delay, [1.0, 2.0, 0.0])
    np.testing.assert_allclose(result.residues[2], 0.0)
    np.testing.assert_allclose(result.cbf_std, [6.0, 12.0, 18.0])
    assert result.cbf_ci_lower is None
    np.testing.assert_allclose(result.cbf_ci_upper, [12.0, 18.0, 24.0])


def test_finalize_dsc_maps_preserves_optional_outputs() -> None:
    """Absent timing, delay, residue, and uncertainty maps remain absent."""
    data = np.ones((1,))
    result = ComputeDSCMaps._finalize_dsc_maps(
        cbf=data,
        cbv=data,
        cbv_auc=data,
        mtt=data,
        tmax=data,
        ttp=data,
        ttp_corrected=data,
        relative_ttp=None,
        auc_raw=data,
        auc_corrected=data,
        auc_difference=data,
        cnr=data,
        snr=data,
        tsnr=data,
        psr=data,
        peak_height=data,
        delay=None,
        residues=None,
        cbf_std=None,
        cbf_ci_lower=None,
        cbf_ci_upper=None,
        mask_bool=np.ones((1,), dtype=bool),
    )

    assert result.relative_ttp is None
    assert result.delay is None
    assert result.residues is None
    assert result.cbf_std is None
    assert result.cbf_ci_lower is None
    assert result.cbf_ci_upper is None


def test_save_map_overrides_upstream_header_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit output dtypes survive an incompatible upstream NIfTI header."""
    monkeypatch.chdir(tmp_path)
    source = _write_image(tmp_path / "concentration.nii.gz", np.ones((2, 1, 1), dtype=np.int16))
    source_img = nib.load(source)
    interface = ComputeDSCMaps(concentration_file=source)

    float_path = interface._save_map(
        np.array([[[0.25]], [[0.75]]]),
        "_float",
        source_img.affine,
        source_img.header,
    )
    mask_path = interface._save_map(
        np.array([[[0]], [[1]]]),
        "_mask",
        source_img.affine,
        source_img.header,
        dtype=np.uint8,
    )

    float_img = nib.load(float_path)
    mask_img = nib.load(mask_path)
    assert float_img.get_data_dtype() == np.dtype(np.float32)
    assert mask_img.get_data_dtype() == np.dtype(np.uint8)
    np.testing.assert_allclose(float_img.get_fdata().ravel(), [0.25, 0.75])
    np.testing.assert_array_equal(mask_img.get_fdata().ravel(), [0.0, 1.0])
    assert source_img.get_data_dtype() == np.dtype(np.int16)


def test_save_base_maps_preserves_result_keys_and_suffixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconditional map serialization retains its public key/suffix registry."""
    data = np.ones((1,))
    maps = ComputeDSCMaps._finalize_dsc_maps(
        cbf=data,
        cbv=data,
        cbv_auc=data,
        mtt=data,
        tmax=data,
        ttp=data,
        ttp_corrected=data,
        relative_ttp=None,
        auc_raw=data,
        auc_corrected=data,
        auc_difference=data,
        cnr=data,
        snr=data,
        tsnr=data,
        psr=data,
        peak_height=data,
        delay=None,
        residues=None,
        cbf_std=None,
        cbf_ci_lower=None,
        cbf_ci_upper=None,
        mask_bool=np.ones((1,), dtype=bool),
    )
    interface = ComputeDSCMaps()
    suffixes: list[str] = []

    def _record_map(*args: object, **kwargs: object) -> str:
        suffix = str(args[1])
        suffixes.append(suffix)
        return f"{suffix}.nii.gz"

    monkeypatch.setattr(interface, "_save_map", _record_map)
    qc = {"n_voxels": 1}
    results = interface._save_base_maps(maps, np.eye(4), object(), qc)

    expected = {
        "out_mtt": "_mtt",
        "out_tmax": "_tmax",
        "out_psr": "_psr",
        "out_ph": "_ph",
        "out_ttp": "_ttp",
        "out_ttp_corrected": "_ttp_corrected",
        "out_cnr": "_cnr",
        "out_auc_raw": "_auc_raw",
        "out_auc_corr": "_auc_corr",
        "out_auc_diff": "_auc_diff",
        "out_rel_cbf": "_rel_cbf",
        "out_rel_cbv": "_rel_cbv",
        "out_rel_cbv_auc": "_rel_cbv_auc",
        "out_rel_auc_raw": "_rel_auc_raw",
        "out_rel_auc_corr": "_rel_auc_corr",
        "out_snr": "_snr",
        "out_tsnr": "_tsnr",
    }
    assert suffixes == list(expected.values())
    assert {key: results[key] for key in expected} == {
        key: f"{suffix}.nii.gz" for key, suffix in expected.items()
    }
    assert results["qc_metrics"] is qc


def test_save_parametric_maps_preserves_masking_cv_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parametric outputs retain CV guards, CI policy, and diagnostic masking."""
    interface = ComputeDSCMaps()
    saved: dict[str, tuple[np.ndarray, object]] = {}

    def _record_map(
        data: np.ndarray,
        suffix: str,
        *args: object,
        dtype: object = np.float32,
    ) -> str:
        saved[suffix] = (data.copy(), dtype)
        return f"{suffix}.nii.gz"

    monkeypatch.setattr(interface, "_save_map", _record_map)
    mask = np.array([True, True, False])
    results = interface._save_parametric_maps(
        rel_cbf=np.array([10.0, 0.0, 20.0]),
        rel_cbv=np.array([2.0, 4.0, 8.0]),
        cbf_std=np.array([2.0, 3.0, 4.0]),
        cbv_std=np.array([1.0, 2.0, 3.0]),
        cbf_ci_lower=np.array([8.0, -1.0, 18.0]),
        cbf_ci_upper=np.array([12.0, 1.0, 22.0]),
        cbv_ci_lower=np.array([1.0, 3.0, 7.0]),
        cbv_ci_upper=np.array([3.0, 5.0, 9.0]),
        parametric_r2=np.array([0.9, 0.5, 0.8]),
        fallback_mask=np.array([0, 1, 1]),
        mask_bool=mask,
        inference="vi",
        affine=np.eye(4),
        header=object(),
    )

    assert set(results) == {
        "out_cbf_std", "out_cbf_cv", "out_cbv_std", "out_cbv_cv",
        "out_cbf_ci95_lower", "out_cbf_ci95_upper",
        "out_cbv_ci95_lower", "out_cbv_ci95_upper",
        "out_parametric_r2", "out_fallback_mask",
    }
    np.testing.assert_allclose(saved["_cbf_std"][0], [2.0, np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(saved["_cbf_cv"][0], [0.2, np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(saved["_cbv_cv"][0], [0.5, np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(
        saved["_cbf_ci95_lower"][0], [8.0, np.nan, 0.0], equal_nan=True
    )
    np.testing.assert_allclose(saved["_dscParametric_r2"][0][:2], [0.9, 0.5])
    assert np.isnan(saved["_dscParametric_r2"][0][2])
    np.testing.assert_array_equal(saved["_dscFallback_mask"][0], [0, 1, 0])
    assert saved["_dscFallback_mask"][1] is np.uint8


def test_save_parametric_maps_preserves_map_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAP intervals are serialized while independently absent maps stay absent."""
    interface = ComputeDSCMaps()
    monkeypatch.setattr(
        interface,
        "_save_map",
        lambda data, suffix, *args, **kwargs: f"{suffix}.nii.gz",
    )
    data = np.ones((1,))

    results = interface._save_parametric_maps(
        rel_cbf=data,
        rel_cbv=data,
        cbf_std=None,
        cbv_std=data,
        cbf_ci_lower=data,
        cbf_ci_upper=data,
        cbv_ci_lower=data,
        cbv_ci_upper=data,
        parametric_r2=None,
        fallback_mask=None,
        mask_bool=np.ones((1,), dtype=bool),
        inference="map",
        affine=np.eye(4),
        header=object(),
    )

    assert set(results) == {
        "out_cbv_std", "out_cbv_cv",
        "out_cbf_ci95_lower", "out_cbf_ci95_upper",
        "out_cbv_ci95_lower", "out_cbv_ci95_upper",
    }


def test_add_fit_qc_metrics_preserves_voxel_fit_and_tier(
    tmp_path: Path,
) -> None:
    """Fit QC assembly retains voxel formatting, rounding, and tier provenance."""
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps({"voxel": [1, 0, 1], "r_squared": 0.87654}))
    interface = ComputeDSCMaps()
    interface._results = {"out_qc_fit_data": str(fit_path)}
    interface._qc_voxel_tier = "enhancing_tumor"
    qc: dict[str, object] = {}
    rel_cbv = np.array([[[1.0, 2.0]], [[3.45678, 4.56789]]])

    interface._add_fit_qc_metrics(rel_cbv, qc)

    assert qc == {
        "qc_voxel": "(1, 0, 1)",
        "qc_voxel_cbv": 4.5679,
        "qc_voxel_rsquared": 0.8765,
        "qc_voxel_tier": "enhancing_tumor",
    }


def test_add_fit_qc_metrics_preserves_absent_and_default_r2(tmp_path: Path) -> None:
    """Missing fit output is inert and a missing serialized R2 defaults to zero."""
    interface = ComputeDSCMaps()
    qc: dict[str, object] = {"existing": True}
    interface._results = {}
    interface._add_fit_qc_metrics(np.ones((1, 1, 1)), qc)
    assert qc == {"existing": True}

    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps({"voxel": [0, 0, 0]}))
    interface._results = {"out_qc_fit_data": str(fit_path)}
    interface._add_fit_qc_metrics(np.full((1, 1, 1), 2.0), qc)
    assert qc["qc_voxel_rsquared"] == 0.0
    assert "qc_voxel_tier" not in qc


def test_save_ancillary_maps_preserves_optional_maps_and_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ancillary outputs retain optionality, model masking, and residue layout."""
    monkeypatch.chdir(tmp_path)
    source = _write_image(tmp_path / "concentration.nii.gz", np.ones((2, 1, 1), dtype=np.int16))
    k2_path = _write_image(tmp_path / "upstream_k2.nii.gz", np.array([[[2.0]], [[9.0]]]))
    source_img = nib.load(source)
    interface = ComputeDSCMaps(concentration_file=source)
    mask = np.array([[[True]], [[False]]])
    residues = np.array([[[[1.0, 2.0]]], [[[0.0, 0.0]]]])

    results = interface._save_ancillary_maps(
        relative_ttp=np.array([[[3.0]], [[0.0]]]),
        k2_map_path=k2_path,
        delay=np.array([[[4.0]], [[0.0]]]),
        alpha=np.array([[[5.0]], [[8.0]]]),
        beta=None,
        cth=np.array([[[6.0]], [[9.0]]]),
        residues=residues,
        save_residue=True,
        mask_bool=mask,
        spatial_shape=mask.shape,
        affine=source_img.affine,
        header=source_img.header,
    )

    assert set(results) == {
        "out_rttp", "out_k2", "out_delay", "out_alpha", "out_cth", "out_residues"
    }
    np.testing.assert_allclose(nib.load(results["out_k2"]).get_fdata().ravel(), [2.0, 0.0])
    np.testing.assert_allclose(nib.load(results["out_alpha"]).get_fdata().ravel(), [5.0, 0.0])
    np.testing.assert_allclose(nib.load(results["out_cth"]).get_fdata().ravel(), [6.0, 0.0])
    np.testing.assert_allclose(nib.load(results["out_residues"]).get_fdata(), residues)


def test_save_ancillary_maps_always_emits_zero_k2_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent optional maps still emit the established all-zero K2 output."""
    monkeypatch.chdir(tmp_path)
    source = _write_image(tmp_path / "concentration.nii.gz", np.ones((2, 1, 1), dtype=np.int16))
    source_img = nib.load(source)
    interface = ComputeDSCMaps(concentration_file=source)

    results = interface._save_ancillary_maps(
        relative_ttp=None,
        k2_map_path=None,
        delay=None,
        alpha=None,
        beta=None,
        cth=None,
        residues=np.ones((2, 1, 1, 2)),
        save_residue=False,
        mask_bool=np.ones((2, 1, 1), dtype=bool),
        spatial_shape=(2, 1, 1),
        affine=source_img.affine,
        header=source_img.header,
    )

    assert set(results) == {"out_k2"}
    np.testing.assert_array_equal(nib.load(results["out_k2"]).get_fdata(), 0.0)


@pytest.mark.parametrize("method", ["vascular", "cpi"])
def test_save_fit_qc_data_dispatches_parametric_methods(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vascular-family methods retain all parametric QC fitting settings."""
    interface = ComputeDSCMaps()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        interface,
        "_save_qc_fit_data",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        interface,
        "_save_qc_fit_data_svd",
        lambda *args, **kwargs: pytest.fail("SVD QC dispatcher called"),
    )
    data = np.ones((1,))
    signal = np.full((1,), 2.0)

    interface._save_fit_qc_data(
        rel_cbf=data,
        concentration=data,
        aif=data,
        mask_bool=data.astype(bool),
        dt=1.5,
        method=method,
        inference="vi",
        mcmc_samples=321,
        cpi_control_points=7,
        cpi_infer_cpt=True,
        cpi_infer_delay=False,
        threshold=0.2,
        oi_threshold=0.1,
        signal=signal,
    )

    assert len(calls) == 1
    assert calls[0][0][4:] == (1.5, method, "vi", 321, 7, True, False)
    assert calls[0][1]["signal_data"] is signal


@pytest.mark.parametrize("method", ["sSVD", "oSVD", "tikhonov"])
def test_save_fit_qc_data_dispatches_svd_methods(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SVD-family methods retain deconvolution threshold QC settings."""
    interface = ComputeDSCMaps()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        interface,
        "_save_qc_fit_data_svd",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        interface,
        "_save_qc_fit_data",
        lambda *args, **kwargs: pytest.fail("parametric QC dispatcher called"),
    )
    data = np.ones((1,))

    interface._save_fit_qc_data(
        rel_cbf=data,
        concentration=data,
        aif=data,
        mask_bool=data.astype(bool),
        dt=2.0,
        method=method,
        inference="map",
        mcmc_samples=100,
        cpi_control_points=5,
        cpi_infer_cpt=False,
        cpi_infer_delay=True,
        threshold=0.25,
        oi_threshold=0.15,
        signal=None,
    )

    assert len(calls) == 1
    assert calls[0][0][4:] == (2.0, method, 0.25, 0.15)
    assert calls[0][1]["signal_data"] is None


@pytest.mark.parametrize("method", ["sSVD", "oSVD", "tikhonov"])
def test_run_deconvolution_normalizes_svd_state(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SVD dispatch retains solver arguments and leaves parametric fields absent."""
    interface = ComputeDSCMaps()
    arrays = [np.full((1,), value) for value in range(1, 7)]
    qc = {"method": method}
    calls: list[tuple[object, ...]] = []

    def _svd(*args: object) -> tuple[object, ...]:
        calls.append(args)
        return (*arrays, qc)

    monkeypatch.setattr(interface, "_deconvolve_svd", _svd)
    monkeypatch.setattr(
        interface,
        "_deconvolve_parametric",
        lambda *args, **kwargs: pytest.fail("parametric solver called"),
    )
    data = np.ones((1,))
    result = interface._run_deconvolution(
        concentration=data,
        aif=data,
        mask=data,
        method=method,
        dt=1.5,
        threshold=0.2,
        oi_threshold=0.1,
        inference="vi",
        mcmc_samples=100,
        cpi_control_points=5,
        cpi_infer_cpt=False,
        cpi_infer_delay=False,
        motion_weighting=SimpleNamespace(weights=None, qc={}),
    )

    assert calls[0][3:] == (method, 1.5, 0.2, 0.1)
    assert result.cbf is arrays[0]
    assert result.delay is arrays[4]
    assert result.residues is arrays[5]
    assert result.qc is qc
    assert result.alpha is None
    assert result.cbf_std is None


@pytest.mark.parametrize("method", ["vascular", "cpi"])
def test_run_deconvolution_normalizes_parametric_state(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parametric dispatch preserves all returned maps and motion provenance."""
    interface = ComputeDSCMaps()
    arrays = [np.full((1,), value) for value in range(1, 16)]
    qc: dict[str, object] = {"method": method}
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _parametric(*args: object, **kwargs: object) -> tuple[object, ...]:
        calls.append((args, kwargs))
        return (*arrays[:4], qc, *arrays[4:])

    monkeypatch.setattr(interface, "_deconvolve_parametric", _parametric)
    monkeypatch.setattr(
        interface,
        "_deconvolve_svd",
        lambda *args, **kwargs: pytest.fail("SVD solver called"),
    )
    data = np.ones((1,))
    weights = np.array([0.5])
    motion_qc = {"enabled": True}
    result = interface._run_deconvolution(
        concentration=data,
        aif=data,
        mask=data,
        method=method,
        dt=2.0,
        threshold=0.2,
        oi_threshold=0.1,
        inference="map",
        mcmc_samples=321,
        cpi_control_points=7,
        cpi_infer_cpt=True,
        cpi_infer_delay=True,
        motion_weighting=SimpleNamespace(weights=weights, qc=motion_qc),
    )

    assert calls[0][0][3:] == (method, 2.0, "map", 321, 7, True, True)
    assert calls[0][1]["motion_weights"] is weights
    assert result.cbf is arrays[0]
    assert result.alpha is arrays[4]
    assert result.fallback_mask is arrays[14]
    assert result.delay is None
    assert result.residues is None
    assert result.qc["motion_weighting"] is motion_qc


def test_prepare_final_outputs_preserves_composition_and_qc_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Postprocessing composes characterized map helpers without changing state."""
    interface = ComputeDSCMaps()
    data = np.ones((1, 1, 1))
    raw = np.full((1, 1, 1, 4), 2.0)
    concentration = np.full((1, 1, 1, 4), 3.0)
    qc: dict[str, object] = {"solver": True}
    deconvolution = _DSCDeconvolutionMaps(
        cbf=data,
        cbv=data * 2.0,
        mtt=data * 3.0,
        tmax=data * 4.0,
        qc=qc,
        delay=data * 5.0,
        residues=np.ones((1, 1, 1, 4)),
        cbf_std=data * 0.1,
        cbf_ci_lower=data * 0.8,
        cbf_ci_upper=data * 1.2,
    )
    final_maps = SimpleNamespace(
        rel_cbf=data,
        rel_cbv=data,
        rel_cbv_auc=data,
        mtt=data,
        tmax=data,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        interface,
        "_compute_signal_metrics",
        lambda *args: calls.append("signal") or SimpleNamespace(
            psr=data * 6.0, peak_height=data * 7.0, qc={"signal": True}
        ),
    )
    monkeypatch.setattr(
        interface,
        "_select_raw_concentration",
        lambda *args: calls.append("raw") or raw,
    )
    monkeypatch.setattr(
        interface,
        "_compute_timing_maps",
        lambda *args: calls.append("timing") or SimpleNamespace(
            ttp=data * 8.0,
            ttp_corrected=data * 9.0,
            relative_ttp=data * 10.0,
            aif_bat_seconds=4.5,
        ),
    )
    monkeypatch.setattr(
        interface,
        "_compute_auc_maps",
        lambda *args: calls.append("auc") or SimpleNamespace(
            raw=data * 11.0,
            corrected=data * 12.0,
            difference=data,
            cbv=data * 13.0,
            qc={"auc": True},
        ),
    )
    monkeypatch.setattr(
        interface,
        "_compute_quality_maps",
        lambda *args: calls.append("quality") or SimpleNamespace(
            cnr=data * 14.0, snr=data * 15.0, tsnr=data * 16.0
        ),
    )
    monkeypatch.setattr(
        interface,
        "_finalize_mtt",
        lambda *args: calls.append("mtt") or data * 17.0,
    )
    monkeypatch.setattr(
        interface,
        "_finalize_dsc_maps",
        lambda **kwargs: calls.append("finalize") or final_maps,
    )
    monkeypatch.setattr(
        interface,
        "_summarize_dsc_maps",
        lambda *args: calls.append("summary") or {"summary": True},
    )

    result = interface._prepare_final_outputs(
        deconvolution=deconvolution,
        concentration=concentration,
        recomputed_raw_concentration=raw,
        signal=None,
        spatial_shape=(1, 1, 1),
        mask_bool=data.astype(bool),
        detected_baseline=2,
        n_baseline=3,
        baseline_info={"arrival_index": 2},
        dt=1.5,
        tail_frames=4,
        method="oSVD",
        aif_level="atlas",
        preload_class="adequate",
    )

    assert calls == ["signal", "raw", "timing", "auc", "quality", "mtt", "finalize", "summary"]
    assert result.raw_concentration is raw
    assert result.maps is final_maps
    assert result.qc is qc
    assert qc == {"solver": True, "signal": True, "auc": True, "summary": True}


def test_assemble_outputs_preserves_serialization_order_and_optional_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final output assembly retains first-pass, fit-QC, and parametric ordering."""
    interface = ComputeDSCMaps(first_pass_cbv=True, save_residue=True)
    data = np.ones((1, 1, 1))
    maps = SimpleNamespace(
        rel_cbf=data,
        rel_cbv=data * 2.0,
        relative_ttp=data * 3.0,
        delay=data * 4.0,
        residues=np.ones((1, 1, 1, 2)),
        cbf_std=data * 0.1,
        cbf_ci_lower=data * 0.8,
        cbf_ci_upper=data * 1.2,
    )
    qc: dict[str, object] = {"existing": True}
    postprocessing = SimpleNamespace(raw_concentration=data, maps=maps, qc=qc)
    inputs = SimpleNamespace(
        affine=np.eye(4),
        header=object(),
        mask_bool=data.astype(bool),
        dt=1.5,
        method="vascular",
        inference="vi",
        mcmc_samples=500,
        cpi_control_points=6,
        cpi_infer_cpt=True,
        cpi_infer_delay=False,
        svd_threshold=0.2,
        oi_threshold=0.1,
        signal=None,
    )
    baseline = SimpleNamespace(
        concentration=np.ones((1, 1, 1, 2)),
        aif=np.ones((2,)),
        n_baseline=2,
        k2_map_path="k2.nii.gz",
    )
    deconvolution = _DSCDeconvolutionMaps(
        cbf=data,
        cbv=data,
        mtt=data,
        tmax=data,
        qc=qc,
        alpha=data * 5.0,
        cbv_std=data * 0.2,
        cbv_ci_lower=data * 0.7,
        cbv_ci_upper=data * 1.3,
        parametric_r2=data * 0.9,
        fallback_mask=data,
    )
    calls: list[str] = []
    base_results: dict[str, object] = {"out_base": "base.nii.gz"}
    monkeypatch.setattr(
        interface,
        "_save_base_maps",
        lambda *args: calls.append("base") or base_results,
    )
    monkeypatch.setattr(
        interface,
        "_compute_first_pass_cbv",
        lambda **kwargs: calls.append("first_pass") or "first_pass.nii.gz",
    )
    monkeypatch.setattr(
        interface,
        "_save_ancillary_maps",
        lambda **kwargs: calls.append("ancillary") or {"out_k2": "k2.nii.gz"},
    )
    monkeypatch.setattr(
        interface,
        "_save_fit_qc_data",
        lambda **kwargs: calls.append("fit"),
    )
    monkeypatch.setattr(
        interface,
        "_add_fit_qc_metrics",
        lambda *args: calls.append("fit_summary"),
    )
    monkeypatch.setattr(
        interface,
        "_save_parametric_maps",
        lambda **kwargs: calls.append("parametric") or {"out_cbf_std": "std.nii.gz"},
    )

    results = interface._assemble_outputs(
        inputs=inputs,
        baseline=baseline,
        deconvolution=deconvolution,
        postprocessing=postprocessing,
    )

    assert calls == ["base", "first_pass", "ancillary", "fit", "fit_summary", "parametric"]
    assert results is base_results
    assert interface._results is results
    assert results["out_rel_cbv_firstpass"] == "first_pass.nii.gz"
    assert results["out_k2"] == "k2.nii.gz"
    assert results["out_cbf_std"] == "std.nii.gz"
