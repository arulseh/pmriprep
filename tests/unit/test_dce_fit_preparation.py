"""Characterization tests for DCE fit-input and domain preparation."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import ComputeDCEMaps, _DCEFitPreparation
from pmriprep.utils.dce_voxel_prep import prepare_voxel_for_fitting


def test_compute_tsnr_uses_baseline_sample_standard_deviation() -> None:
    """DCE baseline tSNR matches the DSC sample-SD and NaN conventions."""
    data = np.array(
        [
            [[[100.0, 102.0, 98.0, 150.0]]],
            [[[50.0, 50.0, 50.0, 80.0]]],
            [[[100.0, 102.0, 98.0, 150.0]]],
        ]
    )
    mask = np.array([[[1]], [[1]], [[0]]], dtype=np.uint8)

    tsnr = ComputeDCEMaps()._compute_tsnr(data, mask, baseline_frames=3)

    assert tsnr[0, 0, 0] == 50.0
    assert np.isnan(tsnr[1, 0, 0])
    assert tsnr[2, 0, 0] == 0.0


def test_compute_tsnr_short_baseline_is_undefined() -> None:
    """A single baseline frame cannot define temporal SNR."""
    data = np.ones((1, 1, 1, 3), dtype=np.float64)
    mask = np.ones((1, 1, 1), dtype=np.uint8)

    assert np.isnan(ComputeDCEMaps()._compute_tsnr(data, mask, 1)).all()


def test_voxel_preparation_preserves_already_converted_concentration() -> None:
    """Fitting preparation must not baseline-subtract converted data twice."""
    converted = np.array([-0.02, 0.02, 0.0, 0.20, 0.50], dtype=np.float64)

    prepared = prepare_voxel_for_fitting(
        converted,
        baseline_frames=3,
        skip_variation_check=True,
    )

    np.testing.assert_array_equal(prepared.concentration, converted)


def test_prepare_fit_inputs_preserves_model_time_and_domain_gates(
    tmp_path: Path,
) -> None:
    """Raw-signal QC and enhancement gates define the pre-BAT fitting domain."""
    shape = (2, 2, 1)
    concentration = np.zeros((*shape, 4), dtype=np.float32)
    concentration[0, 0, 0] = [0.0, 0.0, 0.2, 0.5]
    concentration[1, 0, 0] = [0.0, 0.0, 0.3, 0.7]
    concentration[1, 1, 0] = [0.0, 0.0, 0.4, 0.8]
    mask = np.ones(shape, dtype=np.uint8)
    mask[1, 0, 0] = 0

    signal = np.full((*shape, 4), 100.0, dtype=np.float32)
    signal[0, 0, 0] = [100.0, 102.0, 110.0, 125.0]
    signal[1, 0, 0] = [100.0, 102.0, 110.0, 125.0]

    concentration_file = tmp_path / "concentration.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    signal_file = tmp_path / "signal.nii.gz"
    aif_file = tmp_path / "aif.txt"
    nib.save(nib.Nifti1Image(concentration, np.eye(4)), concentration_file)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)
    nib.save(nib.Nifti1Image(signal, np.eye(4)), signal_file)
    np.savetxt(aif_file, [0.0, 0.0, 1.0, 0.5])

    interface = ComputeDCEMaps(
        concentration_file=str(concentration_file),
        brain_mask_file=str(mask_file),
        dce_signal_file=str(signal_file),
        aif_file=str(aif_file),
        volume_tr=2.0,
        model="auto",
        baseline_frames=2,
        tsnr_threshold=10.0,
        inference="nlls",
        mcmc_samples=123,
        save_fitted_curves=True,
    )

    prepared = interface._prepare_fit_inputs()

    np.testing.assert_allclose(prepared.time_points, [0.0, 1 / 30, 2 / 30, 0.1])
    assert prepared.selection_tr_seconds == 2.0
    assert prepared.is_auto is True
    assert prepared.candidate_names == ["patlak", "extended_tofts", "2cxm"]
    assert prepared.model_name == "2cxm"
    assert prepared.baseline_frames == 2
    assert prepared.inference == "nlls"
    assert prepared.mcmc_samples == 123
    assert prepared.save_fitted_curves is True
    expected_domain = np.zeros(shape, dtype=bool)
    expected_domain[0, 0, 0] = True
    np.testing.assert_array_equal(prepared.fitting_mask, expected_domain)
    assert prepared.tsnr_map[0, 0, 0] > 10.0
    assert prepared.peak_enhancement is not None
    assert prepared.peak_enhancement[0, 0, 0] > 0.0


def test_fitting_domain_is_invariant_to_raw_signal_scale(tmp_path: Path) -> None:
    """Arbitrary scanner intensity scaling must not change voxel admission."""
    shape = (2, 1, 1)
    concentration = np.zeros((*shape, 5), dtype=np.float32)
    concentration[0, 0, 0] = [0.0, 0.0, 0.02, 0.08, 0.04]
    concentration[1, 0, 0] = [0.0, 0.0, 0.0, 0.0, 0.0]
    mask = np.ones(shape, dtype=np.uint8)
    signal = np.zeros((*shape, 5), dtype=np.float32)
    signal[0, 0, 0] = [100.0, 101.0, 108.0, 125.0, 112.0]
    signal[1, 0, 0] = [100.0, 101.0, 102.0, 103.0, 102.0]

    concentration_file = tmp_path / "concentration.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    aif_file = tmp_path / "aif.txt"
    nib.save(nib.Nifti1Image(concentration, np.eye(4)), concentration_file)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)
    np.savetxt(aif_file, [0.0, 0.0, 1.0, 0.5, 0.2])

    domains = []
    for label, scale in (("native", 1.0), ("scaled", 0.001)):
        signal_file = tmp_path / f"{label}_signal.nii.gz"
        nib.save(
            nib.Nifti1Image(signal * scale, np.eye(4)),
            signal_file,
        )
        interface = ComputeDCEMaps(
            concentration_file=str(concentration_file),
            brain_mask_file=str(mask_file),
            dce_signal_file=str(signal_file),
            aif_file=str(aif_file),
            volume_tr=2.0,
            model="auto",
            baseline_frames=2,
            tsnr_threshold=0.0,
        )
        domains.append(interface._prepare_fit_inputs().fitting_mask)

    np.testing.assert_array_equal(domains[0], domains[1])
    assert domains[0][0, 0, 0]
    assert not domains[0][1, 0, 0]


def test_fitting_domain_rejects_nonfinite_concentration(tmp_path: Path) -> None:
    """A positive peak cannot admit a curve containing nonfinite samples."""
    concentration = np.array([[[[0.0, 0.0, np.nan, 0.2, 0.1]]]])
    mask = np.ones((1, 1, 1), dtype=np.uint8)
    concentration_file = tmp_path / "concentration.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    aif_file = tmp_path / "aif.txt"
    nib.save(nib.Nifti1Image(concentration, np.eye(4)), concentration_file)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)
    np.savetxt(aif_file, [0.0, 0.0, 1.0, 0.5, 0.2])

    interface = ComputeDCEMaps(
        concentration_file=str(concentration_file),
        brain_mask_file=str(mask_file),
        aif_file=str(aif_file),
        volume_tr=2.0,
        model="auto",
        baseline_frames=2,
        tsnr_threshold=0.0,
    )

    assert not interface._prepare_fit_inputs().fitting_mask[0, 0, 0]

    interface.inputs.tsnr_threshold = 5.0
    with pytest.raises(ValueError, match="tSNR threshold.*unavailable"):
        interface._prepare_fit_inputs()


def _empty_preparation() -> _DCEFitPreparation:
    """Build a minimal prepared domain for BAT gating tests."""
    concentration = np.zeros((1, 1, 1, 3), dtype=float)
    return _DCEFitPreparation(
        concentration=concentration,
        affine=np.eye(4),
        header=None,
        mask=np.ones((1, 1, 1), dtype=float),
        aif=np.array([0.0, 1.0, 0.5]),
        time_points=np.array([0.0, 0.1, 0.2]),
        n_timepoints=3,
        selection_tr_seconds=6.0,
        model_name="extended_tofts",
        is_auto=False,
        candidate_names=[],
        baseline_frames=1,
        tsnr_threshold=0.0,
        tsnr_map=np.ones((1, 1, 1), dtype=float),
        peak_enhancement=None,
        fitting_mask=np.zeros((1, 1, 1), dtype=bool),
        inference="nlls",
        mcmc_samples=500,
        save_fitted_curves=False,
    )


def test_align_aif_to_tissue_preserves_non_alignment_reasons() -> None:
    """Configuration, AIF provenance, and empty domains retain distinct QC."""
    prepared = _empty_preparation()
    interface = ComputeDCEMaps(aif_bat_alignment="off", aif_detection_method="atlas")
    result = interface._align_aif_to_tissue(prepared)
    assert result.correction == {"applied": False, "reason": "alignment_off"}
    assert result.aif is prepared.aif

    interface.inputs.aif_bat_alignment = "auto"
    interface.inputs.aif_detection_method = "population"
    result = interface._align_aif_to_tissue(prepared)
    assert result.correction == {
        "applied": False,
        "reason": "population_aif_not_aligned",
    }

    interface.inputs.aif_detection_method = "atlas"
    result = interface._align_aif_to_tissue(prepared)
    assert result.correction == {
        "applied": False,
        "reason": "bat_detection_failed",
    }


def test_compute_model_free_maps_assembles_all_vectorized_outputs() -> None:
    """The model-free bundle preserves raw peak QC and semi-quantitative maps."""
    prepared = _empty_preparation()._replace(
        concentration=np.array([[[[0.0, 0.0, 1.0]]]]),
        baseline_frames=2,
        mask=np.ones((1, 1, 1), dtype=float),
        fitting_mask=np.ones((1, 1, 1), dtype=bool),
        peak_enhancement=np.array([[[25.0]]]),
    )
    interface = ComputeDCEMaps(volume_tr=6.0, baseline_frames=2, bolus_arrival_frame=0)

    maps = interface._compute_model_free_maps(prepared)

    assert maps.peak_enhancement is prepared.peak_enhancement
    assert maps.iauc60.shape == (1, 1, 1)
    assert maps.iauc90.shape == (1, 1, 1)
    assert maps.iauc120.shape == (1, 1, 1)
    assert maps.time_to_peak[0, 0, 0] == 12.0
    assert maps.signal_enhancement_ratio.shape == (1, 1, 1)
    assert maps.iauc_qc["reference_frame"] == 0
    assert maps.iauc_qc["configured_baseline_frames"] == 2
    assert maps.iauc_qc["effective_baseline_frames"] == 2
    assert maps.iauc_qc["detected_bolus_arrival_frame"] == 0


def test_prepare_motion_weighting_returns_weights_and_matching_qc() -> None:
    """Motion-weight QC describes the exact array passed to fitting."""
    header = SimpleNamespace(get_zooms=lambda: (2.0, 2.5, 3.0, 1.0))
    interface = ComputeDCEMaps(
        fd_values=[0.0, 0.2, 1.0, 0.1, 0.8, 0.0],
        bolus_arrival_frame=2,
    )

    result = interface._prepare_motion_weighting(header, 6)

    assert result.weights is not None
    assert result.weights.shape == (6,)
    assert result.qc["applied"] is True
    assert result.qc["n_downweighted_frames"] == int(
        np.sum(result.weights < 0.95)
    )
    assert result.qc["min_weight"] == float(np.min(result.weights))
    assert result.qc["washin_weights"] == result.weights[2:6].tolist()


def test_prepare_motion_weighting_noops_without_fd() -> None:
    """Missing motion inputs produce no weights and stable disabled QC."""
    header = SimpleNamespace(get_zooms=lambda: (2.0, 2.5, 3.0, 1.0))

    result = ComputeDCEMaps()._prepare_motion_weighting(header, 6)

    assert result.weights is None
    assert result.qc == {"applied": False}


def test_initialize_parameter_maps_preserves_auto_and_uncertainty_state() -> None:
    """Auto fitting uses NaN PK sentinels and allocates optional fit state."""
    prepared = _empty_preparation()._replace(
        is_auto=True,
        model_name="2cxm",
        candidate_names=["patlak", "extended_tofts", "2cxm"],
        save_fitted_curves=True,
    )
    interface = ComputeDCEMaps(aif_uncertainty_inflation_factor=2.0)

    maps = interface._initialize_parameter_maps(
        prepared, prepared.model_name, prepared.candidate_names,
    )

    assert np.isnan(maps.ktrans).all()
    assert np.isnan(maps.ve).all()
    assert np.isnan(maps.vp).all()
    assert np.isnan(maps.kep).all()
    assert np.isnan(maps.fp).all()
    assert np.isnan(maps.ps).all()
    assert np.isnan(maps.extraction_fraction).all()
    assert maps.fitted_curves is not None
    assert maps.fitted_curves.shape == prepared.concentration.shape
    assert np.isnan(maps.fitted_curves).all()
    assert maps.save_2cxm is True
    assert maps.retained_nondegenerate.dtype == np.bool_
    assert maps.model_selection.dtype == np.int8
    assert maps.delta_bic.dtype == np.float32
    assert any(factor > 1.0 for factor in maps.sigma_factor.values())


def test_initialize_parameter_maps_fixed_model_uses_nan_pk_sentinels() -> None:
    """Explicit fixed models distinguish unfitted voxels from true zero fits."""
    prepared = _empty_preparation()._replace(model_name="tofts", is_auto=False)

    maps = ComputeDCEMaps()._initialize_parameter_maps(prepared, "tofts", [])

    assert np.isnan(maps.ktrans).all()
    assert np.isnan(maps.ve).all()
    assert np.isnan(maps.vp).all()
    assert np.isnan(maps.kep).all()
    assert maps.fitted_curves is None
    assert maps.save_2cxm is False
    assert set(maps.sigma_factor.values()) == {1.0}
