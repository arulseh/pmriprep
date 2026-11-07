"""Characterization tests for DSC compute input preparation."""

import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import (
    ComputeDSCMaps,
    _DSCDeconvolutionMaps,
)
from pmriprep.utils.dsc_signal import (
    compute_psr,
)


def _write_image(path: Path, data: np.ndarray) -> str:
    """Write a test NIfTI image and return its path."""
    nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 3.0, 1.0])), path)
    return str(path)


@pytest.mark.parametrize("method", ["sSVD", "oSVD", "tikhonov"])
def test_compute_maps_keeps_nonfinite_tissue_local(tmp_path: Path, method: str) -> None:
    """A bad in-brain curve yields missing maps without stopping valid neighbours."""
    from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

    time = np.arange(40) * 1.5
    u = np.maximum(time - 9, 0)
    aif = u ** 3 * np.exp(-u / 2)
    aif /= aif.max()
    tissue = build_volterra_matrix(aif, 1.5) @ (0.01 * np.exp(-time / 5))
    curves = np.tile(tissue, (2, 1, 1, 1))
    mask_file = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 1, 1)))
    aif_file = tmp_path / "aif.txt"
    np.savetxt(aif_file, aif)
    results = []
    for name in ("clean", "nonfinite"):
        if name == "nonfinite":
            curves[1, 0, 0, 15] = np.nan
        concentration_file = _write_image(tmp_path / f"{name}.nii.gz", curves)
        destination = tmp_path / name
        destination.mkdir()
        result = ComputeDSCMaps(
            concentration_file=concentration_file,
            raw_concentration_file=concentration_file,
            brain_mask_file=mask_file, aif_file=str(aif_file),
            method=method, tr=1.5, te=0.04,
            baseline_frames=6, leakage_method="none", aif_detection_level="external",
        ).run(cwd=str(destination))
        results.append(result.outputs)
    for output in ("out_rel_cbf", "out_rel_cbv", "out_rel_cbv_auc", "out_mtt", "out_tmax"):
        clean = nib.load(getattr(results[0], output)).get_fdata()
        bad = nib.load(getattr(results[1], output)).get_fdata()
        assert np.isfinite(bad[0]).all()
        np.testing.assert_allclose(bad[0], clean[0])
        assert np.isnan(bad[1]).all()


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
        cpi_control_points=5,
        cpi_infer_delay=True,
        inference="vi",
        baseline_frames=7,
        tail_frames=8,
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
    assert result.cpi_control_points == 5
    assert result.cpi_infer_delay is True
    assert result.inference == "vi"
    assert result.baseline_frames == 7
    assert result.tail_frames == 8


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


def test_prepare_baseline_preserves_pass_through_k2(
    tmp_path: Path,
) -> None:
    """Final AIF baseline centers arrays and retains upstream K2 provenance."""
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
    result = interface._prepare_baseline_and_leakage(interface._load_compute_inputs())

    assert result.detected_frames == 4
    assert result.n_baseline == 4
    assert result.k2_map_path == k2_file
    np.testing.assert_allclose(result.aif[:4], 0.0)
    np.testing.assert_allclose(result.concentration[..., :4], 0.0)
    assert result.detection_info["aif_baseline_offset"] == 1.0
    assert result.detection_info["conc_median_baseline_offset"] == 2.0


def test_compute_signal_metrics_computes_noise_gated_psr() -> None:
    """Raw-signal PSR uses the configured windows and significance gate."""
    signal = np.array([[[[100.0, 101.0, 99.0, 100.0, 70.0, 75.0, 80.0, 85.0]]]])
    mask = np.ones((1, 1, 1), dtype=bool)

    result = ComputeDSCMaps()._compute_signal_metrics(signal, (1, 1, 1), 4, 3, mask)

    expected_psr = compute_psr(signal, 4, 3, brain_mask=mask, min_drop_noise_sd=5.0)
    np.testing.assert_allclose(result.psr, expected_psr)
    assert result.qc["psr_from_signal"] is True
    assert result.qc["psr_valid_fraction"] == 1.0


def test_compute_signal_metrics_without_signal_returns_nan_map() -> None:
    """Unavailable raw signal is undefined with explicit QC provenance."""
    result = ComputeDSCMaps()._compute_signal_metrics(
        None, (2, 3, 1), 4, 5, np.ones((2, 3, 1), dtype=bool)
    )

    assert np.isnan(result.psr).all()
    assert result.qc == {"psr_from_signal": False, "psr_valid_fraction": 0.0}


def test_select_raw_concentration_preserves_source_precedence(tmp_path: Path) -> None:
    """Configured raw and corrected sources retain their order."""
    corrected = np.zeros((1, 1, 1, 4), dtype=np.float64)
    configured = np.ones_like(corrected)
    raw_file = _write_image(tmp_path / "raw.nii.gz", configured)
    interface = ComputeDSCMaps(raw_concentration_file=raw_file)

    np.testing.assert_allclose(interface._select_raw_concentration(corrected), configured)
    assert ComputeDSCMaps()._select_raw_concentration(corrected) is corrected


def test_compute_timing_maps_uses_aif_arrival_reference() -> None:
    """The sole tissue TTP map is referenced to detected AIF arrival."""
    concentration = np.zeros((2, 1, 1, 8), dtype=np.float64)
    concentration[0, 0, 0, 4] = 2.0
    concentration[1, 0, 0, 6] = 2.0
    mask = np.ones((2, 1, 1), dtype=bool)

    result = ComputeDSCMaps()._compute_timing_maps(
        concentration,
        mask,
        detected_baseline=2,
        detection_info={"arrival_index": 3},
        dt=2.0,
    )

    np.testing.assert_allclose(result.ttp[:, 0, 0], [2.0, 6.0])
    assert result.baseline_reference_seconds == pytest.approx(6.0)


def test_compute_timing_maps_keeps_prearrival_peaks_undefined() -> None:
    """A tissue peak before the AIF reference is invalid, not a zero-second TTP."""
    concentration = np.zeros((1, 1, 1, 6), dtype=np.float64)
    concentration[..., 2] = 1.0
    result = ComputeDSCMaps()._compute_timing_maps(
        concentration, np.ones((1, 1, 1), dtype=bool), 2, {"arrival_index": 3}, 1.5
    )

    assert np.isnan(result.ttp[0, 0, 0])


def test_compute_timing_maps_preserves_delays_above_30_seconds() -> None:
    """Severe physiological delay must not be clipped to a display boundary."""
    concentration = np.zeros((1, 1, 1, 30), dtype=np.float64)
    concentration[..., 25] = 1.0
    result = ComputeDSCMaps()._compute_timing_maps(
        concentration, np.ones((1, 1, 1), dtype=bool), 2, {"arrival_index": 3}, 2.0
    )

    assert result.ttp[0, 0, 0] == pytest.approx(44.0)


def test_compute_auc_maps_preserves_symmetric_measured_aif_ratio(
    tmp_path: Path,
) -> None:
    """Signed tissue ratios include the full bolus and recirculation window."""
    pulse = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.25])
    corrected = np.stack((pulse, -pulse))[:, None, None, :]
    raw = corrected + 5.0
    measured_aif = 2.0 * pulse + 1.0
    dt = 1.5
    measured_file = tmp_path / "measured_aif.txt"
    np.savetxt(measured_file, measured_aif)
    interface = ComputeDSCMaps(aif_file=str(measured_file))

    result = interface._compute_auc_maps(
        raw,
        corrected,
        n_baseline=2,
        detected_baseline=2,
        detection_info={"arrival_index": 2},
        dt=dt,
    )

    expected_raw = np.trapezoid(raw - raw[..., :2].mean(axis=-1, keepdims=True), dx=dt, axis=-1)
    expected_corrected = np.trapezoid(corrected, dx=dt, axis=-1)
    expected_aif_auc = np.trapezoid((measured_aif - 1.0)[2:], dx=dt)
    np.testing.assert_allclose(result.raw, expected_raw)
    np.testing.assert_allclose(result.corrected, expected_corrected)
    np.testing.assert_allclose(result.difference, expected_raw - expected_corrected)
    np.testing.assert_allclose(result.cbv[:, 0, 0], [0.5, -0.5])
    assert {key: result.qc[key] for key in ("cbv_auc_bat_frame", "cbv_auc_end_frame", "cbv_auc_aif_integral")} == {
        "cbv_auc_bat_frame": 2,
        "cbv_auc_end_frame": len(pulse),
        "cbv_auc_aif_integral": round(float(expected_aif_auc), 6),
    }


def test_compute_auc_maps_near_zero_measured_aif_returns_nan(
    tmp_path: Path,
) -> None:
    """A nonpositive measured-AIF integral retains unavailable CBV semantics."""
    measured_file = tmp_path / "measured_aif.txt"
    np.savetxt(measured_file, np.ones(6))
    concentration = np.ones((2, 1, 1, 6), dtype=np.float64)

    result = ComputeDSCMaps(aif_file=str(measured_file))._compute_auc_maps(
        concentration, concentration, 2, 2, {"arrival_index": 2}, 1.0
    )

    assert np.all(np.isnan(result.cbv))
    assert result.qc["cbv_auc_aif_integral"] == 0.0
    assert np.isfinite(result.tissue).all()
    qc = ComputeDSCMaps._summarize_dsc_maps(
        result.cbv, result.cbv, result.cbv, result.cbv, result.cbv,
        np.ones(result.cbv.shape, dtype=bool), "external", {}, 2.0,
    )
    from pmriprep.utils.sidecar_builders import dump_sidecar_json, filter_pipeline_qc

    qc["tissue_auc_finite_voxels"] = int(np.isfinite(result.tissue).sum())
    qc["tissue_auc_negative_fraction"] = float(np.mean(result.tissue < 0))
    public = json.loads(dump_sidecar_json(filter_pipeline_qc({"deconvolution": qc})))
    published = public["deconvolution"]
    assert published["cbv_auc_finite_voxels"] == 0
    assert published["tissue_auc_finite_voxels"] == 2
    assert published["cbv_auc_negative_fraction"] is None
    assert published["tissue_auc_negative_fraction"] == 0.0


def test_compute_auc_maps_does_not_clip_large_finite_values(tmp_path: Path) -> None:
    """Large finite AUC and CBV estimates remain visible for downstream QC."""
    measured_file = tmp_path / "measured_aif.txt"
    np.savetxt(measured_file, np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.1]))
    concentration = np.zeros((1, 1, 1, 6), dtype=np.float64)
    concentration[..., 2:] = 5000.0

    result = ComputeDSCMaps(aif_file=str(measured_file))._compute_auc_maps(
        concentration, concentration, 2, 2, {"arrival_index": 2}, 1.0
    )

    assert result.corrected[0, 0, 0] > 10000.0
    assert result.cbv[0, 0, 0] > 10.0


def test_compute_auc_maps_missing_aif_reports_unavailable_cbv() -> None:
    """Missing measured-AIF denominator makes AUC-CBV unavailable."""
    concentration = np.ones((1, 1, 1, 6), dtype=np.float64)

    result = ComputeDSCMaps()._compute_auc_maps(
        concentration, concentration, 2, 2, {"arrival_index": 2}, 1.0
    )

    assert result.cbv is None
    assert result.qc["cbv_auc_unavailable_reason"] == "aif_unavailable"
    np.testing.assert_allclose(result.tissue, 3.0)


def test_compute_quality_maps_uses_signal_drop_and_baseline_tsnr() -> None:
    """CNR and tSNR use the raw signal and stationary baseline window."""
    signal = np.array(
        [
            [[[100.0, 102.0, 98.0, 75.0, 85.0]]],
            [[[50.0, 50.0, 50.0, 40.0, 45.0]]],
        ]
    )

    mask = np.ones((2, 1, 1), dtype=bool)
    result = ComputeDSCMaps()._compute_quality_maps(signal, 3, mask.shape, mask)

    np.testing.assert_allclose(result.cnr.ravel(), [12.5, 5.0])
    assert result.tsnr[0, 0, 0] == pytest.approx(50.0)
    assert np.isnan(result.tsnr[1, 0, 0])


def test_compute_quality_maps_without_signal_returns_nan() -> None:
    """Missing raw signal leaves both signal-quality maps undefined."""
    mask = np.ones((1, 1, 1), dtype=bool)

    result = ComputeDSCMaps()._compute_quality_maps(None, 3, mask.shape, mask)

    assert np.isnan(result.cnr).all()
    assert np.isnan(result.tsnr).all()


def test_compute_quality_maps_rejects_no_drop_and_short_baseline() -> None:
    """CNR requires a positive signal drop and both maps require baseline variance."""
    mask = np.ones((1, 1, 1), dtype=bool)
    no_drop = np.array([[[[100.0, 102.0, 98.0, 105.0, 110.0]]]])

    result = ComputeDSCMaps()._compute_quality_maps(no_drop, 3, mask.shape, mask)
    short = ComputeDSCMaps()._compute_quality_maps(no_drop, 1, mask.shape, mask)

    assert np.isnan(result.cnr).all()
    assert np.isfinite(result.tsnr).all()
    assert np.isnan(short.cnr).all()
    assert np.isnan(short.tsnr).all()


@pytest.mark.parametrize("method", ["sSVD", "oSVD", "tikhonov"])
def test_finalize_mtt_uses_auc_cbv_for_svd_family(method: str) -> None:
    """SVD-family MTT uses measured-AIF CBV divided by unscaled CBF."""
    fitted = np.full((4,), 99.0)
    cbf = np.array([0.5, 0.0, 0.25, 0.1])
    auc_cbv = np.array([2.0, 3.0, np.nan, 1.5])

    result = ComputeDSCMaps._finalize_mtt(method, fitted, cbf, auc_cbv)

    np.testing.assert_allclose(
        result, [4.0, np.nan, np.nan, 15.0], equal_nan=True
    )


@pytest.mark.parametrize("method", ["cpi"])
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
        rel_cbv_auc=np.array([np.nan, 4.0, 999.0]),
        mtt=np.array([5.0, 9.0, 999.0]),
        tmax=np.array([3.0, 7.0, 999.0]),
        mask_bool=mask,
        aif_level="atlas",
        baseline_info={"baseline_frames_detected": 4},
        baseline_reference_seconds=6.0,
        cbv_auc_qc_source=np.array([-2.0, 4.0, 999.0]),
    )

    assert qc["cbf_mean"] == pytest.approx(20.0)
    assert qc["cbf_std"] == pytest.approx(10.0)
    assert qc["cbv_mean"] == pytest.approx(2.0)
    assert qc["cbv_auc_median"] == pytest.approx(4.0)
    assert qc["cbv_auc_finite_voxels"] == 2
    assert qc["cbv_auc_negative_voxels"] == 1
    assert qc["cbv_auc_negative_fraction"] == pytest.approx(0.5)
    assert qc["mtt_mean"] == pytest.approx(7.0)
    assert qc["tmax_median"] == pytest.approx(5.0)
    assert qc["n_voxels"] == 2
    assert qc["baseline_frames_detected"] == 4
    assert qc["baseline_reference_seconds"] == pytest.approx(6.0)
    assert qc["cbv_literature_range_applicable"] is True


@pytest.mark.parametrize("method", ["cpi"])
def test_parametric_timing_summary_is_arrival_delay(method: str) -> None:
    """Fitted parametric timing is not summarized as conventional Tmax."""
    values = np.array([-0.5, 0.5])
    qc = ComputeDSCMaps._summarize_dsc_maps(
        values,
        values,
        values,
        values,
        values,
        np.ones(2, dtype=bool),
        "atlas",
        {},
        0.0,
        method=method,
    )
    assert qc["arrival_delay_mean"] == pytest.approx(0.0)
    assert qc["arrival_delay_median"] == pytest.approx(0.0)
    assert qc["arrival_delay_finite_voxels"] == 2
    assert qc["arrival_delay_finite_fraction"] == pytest.approx(1.0)
    assert "tmax_mean" not in qc
    assert "tmax_median" not in qc


def test_parametric_timing_summary_omits_empty_measurement() -> None:
    """An unestimated arrival-delay map does not produce a NaN median label."""
    values = np.ones(2)
    qc = ComputeDSCMaps._summarize_dsc_maps(
        values,
        values,
        values,
        values,
        np.full(2, np.nan),
        np.ones(2, dtype=bool),
        "atlas",
        {},
        0.0,
        method="cpi",
    )
    assert qc["arrival_delay_finite_voxels"] == 0
    assert qc["arrival_delay_finite_fraction"] == 0.0
    assert "arrival_delay_mean" not in qc
    assert "arrival_delay_median" not in qc


def test_summarize_dsc_maps_ignores_undefined_voxels() -> None:
    """One unavailable estimate does not erase summaries of valid voxels."""
    values = np.array([2.0, np.nan])
    qc = ComputeDSCMaps._summarize_dsc_maps(
        values,
        values,
        values,
        values,
        values,
        np.ones(2, dtype=bool),
        "atlas",
        {},
        0.0,
    )

    assert qc["cbf_mean"] == pytest.approx(2.0)
    assert qc["mtt_median"] == pytest.approx(2.0)


def test_scale_cbf_uncertainty_preserves_optionality_and_units() -> None:
    """Available CBF uncertainty converts to the relative-map per-minute units."""
    std = np.array([0.1, 0.2])
    scaled_std = ComputeDSCMaps._scale_cbf_uncertainty(std)

    np.testing.assert_allclose(scaled_std, [6.0, 12.0])
    assert ComputeDSCMaps._scale_cbf_uncertainty(None) is None
    np.testing.assert_allclose(std, [0.1, 0.2])


def test_finalize_dsc_maps_preserves_ranges_units_and_nan_policy() -> None:
    """Final maps retain established ranges, CBF units, and timing NaN policy."""
    values = np.array([-1.0, 2.0, 2000.0])
    mask = np.array([True, True, False])

    result = ComputeDSCMaps._finalize_dsc_maps(
        cbf=np.array([-1.0, 2.0, 200.0]),
        cbv=np.array([-1.0, 12.0, 2.0]),
        cbv_auc=np.array([np.nan, 4.0, 8.0]),
        mtt=np.array([-1.0, 40.0, 5.0]),
        tmax=np.array([-1.0, 70.0, 5.0]),
        ttp=np.array([np.nan, 3.0, 4.0]),
        tissue_auc=values,
        auc_raw=values,
        auc_corrected=values + 1.0,
        auc_difference=values + 2.0,
        cnr=np.array([-1.0, 2000.0, 5.0]),
        tsnr=values,
        psr=np.array([-1.0, 250.0, 5.0]),
        cbf_std=np.array([0.1, 0.2, 0.3]),
        mask_bool=mask,
    )

    assert np.isnan(result.rel_cbf[0])
    np.testing.assert_allclose(result.rel_cbf[1:], [120.0, 0.0])
    assert np.isnan(result.rel_cbv[0])
    np.testing.assert_allclose(result.rel_cbv[1:], [12.0, 0.0])
    assert np.isnan(result.rel_cbv_auc[0])
    assert np.isnan(result.mtt[0])
    np.testing.assert_allclose(result.mtt[1:], [40.0, 0.0])
    np.testing.assert_allclose(result.tmax, [-1.0, 70.0, 0.0])
    np.testing.assert_allclose(result.cnr, [-1.0, 2000.0, 0.0])
    np.testing.assert_allclose(result.psr, [-1.0, 250.0, 0.0])
    np.testing.assert_allclose(result.ttp[1:], [3.0, 0.0])
    assert np.isnan(result.ttp[0])
    np.testing.assert_allclose(result.cbf_std, [6.0, 12.0, 18.0])


def test_finalize_dsc_maps_preserves_optional_uncertainty() -> None:
    """Absent uncertainty remains absent."""
    data = np.ones((1,))
    result = ComputeDSCMaps._finalize_dsc_maps(
        cbf=data,
        cbv=data,
        cbv_auc=data,
        mtt=data,
        tmax=data,
        ttp=data,
        tissue_auc=data,
        auc_raw=data,
        auc_corrected=data,
        auc_difference=data,
        cnr=data,
        tsnr=data,
        psr=data,
        cbf_std=None,
        mask_bool=np.ones((1,), dtype=bool),
    )

    assert result.cbf_std is None


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
        tissue_auc=data,
        auc_raw=data,
        auc_corrected=data,
        auc_difference=data,
        cnr=data,
        tsnr=data,
        psr=data,
        cbf_std=None,
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
        "out_tissue_auc": "_tissue_auc",
        "out_mtt": "_mtt",
        "out_tmax": "_tmax",
        "out_psr": "_psr",
        "out_ttp": "_ttp",
        "out_cnr": "_cnr",
        "out_auc_raw": "_auc_raw",
        "out_auc_corr": "_auc_corr",
        "out_auc_diff": "_auc_diff",
        "out_rel_cbf": "_rel_cbf",
        "out_rel_cbv": "_rel_cbv",
        "out_rel_cbv_auc": "_rel_cbv_auc",
        "out_tsnr": "_tsnr",
    }
    assert set(suffixes) == set(expected.values())
    assert {key: results[key] for key in expected} == {
        key: f"{suffix}.nii.gz" for key, suffix in expected.items()
    }
    assert results["qc_metrics"] is qc


def test_save_parametric_maps_preserves_masking_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parametric outputs retain uncertainty and diagnostic masking."""
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
        cbf_std=np.array([2.0, 3.0, 4.0]),
        cbv_std=np.array([1.0, 2.0, 3.0]),
        parametric_r2=np.array([0.9, 0.5, 0.8]),
        fallback_mask=np.array([0, 1, 1]),
        mask_bool=mask,
        affine=np.eye(4),
        header=object(),
    )

    assert set(results) == {
        "out_cbf_std", "out_cbv_std",
        "out_parametric_r2", "out_fallback_mask",
    }
    np.testing.assert_allclose(saved["_cbf_std"][0], [2.0, np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(saved["_dscParametric_r2"][0][:2], [0.9, 0.5])
    assert np.isnan(saved["_dscParametric_r2"][0][2])
    np.testing.assert_array_equal(saved["_dscFallback_mask"][0], [0, 1, 0])
    assert saved["_dscFallback_mask"][1] is np.uint8


def test_save_parametric_maps_preserves_optional_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independently unavailable uncertainty maps remain absent."""
    interface = ComputeDSCMaps()
    monkeypatch.setattr(
        interface,
        "_save_map",
        lambda data, suffix, *args, **kwargs: f"{suffix}.nii.gz",
    )
    data = np.ones((1,))

    results = interface._save_parametric_maps(
        cbf_std=None,
        cbv_std=data,
        parametric_r2=None,
        fallback_mask=None,
        mask_bool=np.ones((1,), dtype=bool),
        affine=np.eye(4),
        header=object(),
    )

    assert set(results) == {"out_cbv_std"}


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
    """Ancillary outputs retain model masking."""
    monkeypatch.chdir(tmp_path)
    source = _write_image(tmp_path / "concentration.nii.gz", np.ones((2, 1, 1), dtype=np.int16))
    k2_path = _write_image(tmp_path / "upstream_k2.nii.gz", np.array([[[2.0]], [[9.0]]]))
    source_img = nib.load(source)
    interface = ComputeDSCMaps(concentration_file=source)
    mask = np.array([[[True]], [[False]]])
    results = interface._save_ancillary_maps(
        k2_map_path=k2_path,
        leakage_method="bsw",
        mask_bool=mask,
        affine=source_img.affine,
        header=source_img.header,
    )

    assert set(results) == {"out_k2"}
    np.testing.assert_allclose(nib.load(results["out_k2"]).get_fdata().ravel(), [2.0, 0.0])


def test_save_ancillary_maps_rejects_missing_k2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing upstream K2 map cannot be represented as measured zero."""
    monkeypatch.chdir(tmp_path)
    source = _write_image(tmp_path / "concentration.nii.gz", np.ones((2, 1, 1), dtype=np.int16))
    source_img = nib.load(source)
    interface = ComputeDSCMaps(concentration_file=source)

    with pytest.raises(RuntimeError, match="produced no coefficient map"):
        interface._save_ancillary_maps(
            k2_map_path=None,
            leakage_method="bsw",
            mask_bool=np.ones((2, 1, 1), dtype=bool),
            affine=source_img.affine,
            header=source_img.header,
        )


def test_save_ancillary_maps_allows_missing_k2_when_leakage_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicitly disabled leakage correction must not synthesize a K2 map."""
    monkeypatch.chdir(tmp_path)
    source = _write_image(tmp_path / "concentration.nii.gz", np.ones((1, 1, 1)))
    source_img = nib.load(source)
    interface = ComputeDSCMaps(concentration_file=source, leakage_method="none")

    results = interface._save_ancillary_maps(
        k2_map_path=None,
        leakage_method="none",
        mask_bool=np.ones((1, 1, 1), dtype=bool),
        affine=source_img.affine,
        header=source_img.header,
    )

    assert "out_k2" not in results


@pytest.mark.parametrize("method", ["cpi"])
def test_save_fit_qc_data_dispatches_parametric_methods(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPI QC uses the selected signal fit without a separate refit."""
    interface = ComputeDSCMaps()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        interface,
        "_save_signal_cpi_qc",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        interface,
        "_save_qc_fit_data_svd",
        lambda *args, **kwargs: pytest.fail("SVD QC dispatcher called"),
    )
    data = np.ones((1,))
    signal = np.full((1,), 2.0)

    selected_fit = object()
    interface._save_fit_qc_data(
        rel_cbf=data,
        concentration=data,
        aif=data,
        mask_bool=data.astype(bool),
        dt=1.5,
        method=method,
        threshold=0.2,
        oi_threshold=0.1,
        signal=signal,
        cpi_fit=selected_fit,
    )

    assert len(calls) == 1
    assert calls[0][0][4:] == (1.5, signal, selected_fit)


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
        "_save_signal_cpi_qc",
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
    arrays = [np.full((1,), value) for value in range(1, 5)]
    qc = {"method": method}
    calls: list[tuple[object, ...]] = []

    def _svd(*args: object) -> tuple[object, ...]:
        calls.append(args)
        return (*arrays, qc)

    monkeypatch.setattr(interface, "_deconvolve_svd", _svd)
    monkeypatch.setattr(
        interface,
        "_deconvolve_signal_cpi",
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
    )

    assert calls[0][3:] == (method, 1.5, 0.2, 0.1)
    assert result.cbf is arrays[0]
    assert result.qc is qc
    assert result.cbf_std is None


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
        cbf_std=data * 0.1,
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
        lambda *args: calls.append("signal")
        or SimpleNamespace(psr=data * 6.0, qc={"signal": True}),
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
            ttp=data * 9.0,
            baseline_reference_seconds=4.5,
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
            tissue=data * 13.0,
            qc={"auc": True},
        ),
    )
    monkeypatch.setattr(
        interface,
        "_compute_quality_maps",
        lambda *args: calls.append("quality")
        or SimpleNamespace(cnr=data * 14.0, tsnr=data * 16.0),
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
    )

    assert calls == ["signal", "raw", "timing", "auc", "quality", "mtt", "finalize", "summary"]
    assert result.raw_concentration is raw
    assert result.maps is final_maps
    assert result.qc is qc
    assert all(qc[key] is True for key in ("solver", "signal", "auc", "summary"))
    assert qc["processing_status"] == "complete"


def test_assemble_outputs_preserves_serialization_order_and_optional_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final output assembly retains fit-QC and parametric ordering."""
    interface = ComputeDSCMaps()
    data = np.ones((1, 1, 1))
    maps = SimpleNamespace(
        rel_cbf=data,
        rel_cbv=data * 2.0,
        cbf_std=data * 0.1,
    )
    qc: dict[str, object] = {"existing": True}
    postprocessing = SimpleNamespace(raw_concentration=data, maps=maps, qc=qc)
    inputs = SimpleNamespace(
        affine=np.eye(4),
        header=object(),
        mask_bool=data.astype(bool),
        dt=1.5,
        method="cpi",
        inference="vi",
        svd_threshold=0.2,
        oi_threshold=0.1,
        signal=None,
        leakage_method="bsw",
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
        cbv_std=data * 0.2,
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

    assert calls == ["base", "ancillary", "fit", "fit_summary", "parametric"]
    assert results is base_results
    assert interface._results is results
    assert results["out_k2"] == "k2.nii.gz"
    assert results["out_cbf_std"] == "std.nii.gz"


def test_partial_maps_keep_same_window_ncbv_and_optional_sinks(tmp_path, monkeypatch) -> None:
    """A missing AIF preserves tissue nCBV and flows through optional map sinks."""
    from nipype.interfaces.base import isdefined
    from nipype.interfaces.utility import Function
    from pmriprep.utils.dsc_deconvolution import build_volterra_matrix
    from pmriprep.workflows.dsc._helpers import _normalize_cbv, _normalize_cbf
    from pmriprep.workflows.dsc.datasinks import _copy_perf_map

    time = np.arange(40) * 1.5
    u = np.maximum(time - 9, 0)
    aif = u**3 * np.exp(-u / 2)
    tissue = build_volterra_matrix(aif, 1.5) @ (0.01 * np.exp(-time / 5))
    concentration = np.stack([tissue, 2 * tissue]).reshape(2, 1, 1, -1)
    signal = 1000 * np.exp(-0.04 * concentration)
    signal[..., :6] += np.array([-2, 1, 2, -1, 1, -1])
    image = _write_image(tmp_path / "concentration.nii.gz", concentration)
    raw = _write_image(tmp_path / "signal.nii.gz", signal)
    mask = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 1, 1)))
    nawm = _write_image(tmp_path / "nawm.nii.gz", np.array([1., 0.]).reshape(2, 1, 1))
    aif_path = tmp_path / "aif.txt"
    np.savetxt(aif_path, aif)
    ncbv_maps = []
    for name in ("complete", "partial"):
        cwd = tmp_path / name
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        kwargs = {"aif_file": str(aif_path)} if name == "complete" else {}
        outputs = ComputeDSCMaps(
            concentration_file=image, raw_concentration_file=image,
            signal_file=raw, brain_mask_file=mask, baseline_frames=6,
            method="oSVD", tr=1.5, te=0.04, leakage_method="none", **kwargs,
        ).run(cwd=str(cwd)).outputs
        ncbv_path, factor = _normalize_cbv(outputs.out_tissue_auc, "", "01", "", nawm)
        ncbv_maps.append(nib.load(ncbv_path).get_fdata())
        assert factor > 0
        for attr in ("out_psr", "out_ttp", "out_cnr", "out_tsnr", "out_auc_raw", "out_auc_corr"):
            assert np.isfinite(nib.load(getattr(outputs, attr)).get_fdata()).all()
        if name == "partial":
            assert outputs.qc_metrics["processing_status"] == "partial"
            assert not isdefined(outputs.out_rel_cbf)
            json.dumps(outputs.qc_metrics, allow_nan=False)
            normalizer = Function(input_names=["cbf_file"], output_names=["path", "factor"], function=_normalize_cbf)
            assert normalizer.run().outputs.path == ""
            sink = Function(
                input_names=["in_file", "output_dir", "subject", "session", "desc", "suffix", "meta_dict"],
                output_names=["path"], function=_copy_perf_map,
            )
            assert sink.run().outputs.path == ""
            sink.inputs.in_file = ncbv_path
            sink.inputs.output_dir = str(cwd / "derivatives")
            sink.inputs.subject = "01"
            sink.inputs.desc = "dscNawmNorm"
            sink.inputs.suffix = "cbv"
            sink.inputs.meta_dict = {}
            assert Path(sink.run().outputs.path).is_file()
        else:
            auc_cbv = nib.load(outputs.out_rel_cbv_auc).get_fdata()
            np.testing.assert_allclose(ncbv_maps[-1], auc_cbv / auc_cbv[0, 0, 0], rtol=1e-6)
    np.testing.assert_array_equal(*ncbv_maps)
