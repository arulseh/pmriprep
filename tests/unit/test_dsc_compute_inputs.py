"""Characterization tests for DSC compute input preparation."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import ComputeDSCMaps
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
        rel_cbv_auc=np.array([2.0, 4.0, 999.0]),
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
    assert qc["cbv_auc_median"] == pytest.approx(3.0)
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
