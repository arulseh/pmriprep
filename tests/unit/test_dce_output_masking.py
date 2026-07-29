"""Integration tests for masking saved DCE quantitative derivatives."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import (
    ComputeDCEMaps,
    _DCEModelFreeMaps,
    _build_dce_qc_mask,
    _evaluate_dce_qc_mask,
    _compute_dce_coefficient_of_variation,
    _write_dce_map,
    _write_dce_fitted_curves,
    _write_dce_model_free_maps,
    _write_dce_parameter_maps,
    _write_dce_terminal_outputs,
    _write_dce_uncertainty_maps,
    _zero_dce_background,
)
from pmriprep.utils.dce_constants import EPSILON_VE
from pmriprep.models.dce.batched_models import batched_tofts_model


def test_zero_dce_background_preserves_in_mask_nan_and_dtype() -> None:
    """Masking retains in-brain sentinels and categorical array dtype."""
    mask = np.array([[[True, False]]])
    values = np.array([[[np.nan, 7.0]]], dtype=np.float32)
    categorical = np.array([[[3, 4]]], dtype=np.int8)
    masked_values = _zero_dce_background(values, mask)
    masked_categorical = _zero_dce_background(categorical, mask)
    assert np.isnan(masked_values[0, 0, 0])
    assert masked_values[0, 0, 1] == 0.0
    assert masked_categorical.tolist() == [[[3, 0]]]
    assert masked_categorical.dtype == np.int8


def test_zero_dce_background_broadcasts_over_time() -> None:
    """The spatial mask applies to every frame of fitted 4D curves."""
    mask = np.array([[[True, False]]])
    curves = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
    masked = _zero_dce_background(curves, mask)
    np.testing.assert_array_equal(masked[0, 0, 0], curves[0, 0, 0])
    np.testing.assert_array_equal(masked[0, 0, 1], 0.0)


def test_build_dce_qc_mask_preserves_strict_thresholds() -> None:
    """QC acceptance retains inclusive signal gates and strict PK gates."""
    shape = (1, 1, 4)
    mask = _build_dce_qc_mask(
        np.ones(shape, dtype=bool),
        np.array([[[10.0, 10.0, 10.0, 10.0]]]),
        np.array([[[10.0, 10.0, 10.0, 10.0]]]),
        np.array([[[0.1, 0.0, 5.0, 0.1]]]),
        np.array([[[0.5, 0.9, 0.9, 0.49]]]),
        np.full(shape, 0.2),
        np.full(shape, 0.1),
        np.full(shape, 0.01),
        np.ones(shape, dtype=np.int8),
        np.zeros(shape, dtype=np.int8),
        {"patlak": 1},
        False,
        "tofts",
        tsnr_threshold=10.0,
    )
    assert mask.tolist() == [[[1, 0, 0, 0]]]
    assert mask.dtype == np.uint8


def test_build_dce_qc_mask_auto_patlak_allows_nan_parameters() -> None:
    """Auto-selected Patlak voxels retain undefined ve and kep semantics."""
    shape = (1, 1, 2)
    mask = _build_dce_qc_mask(
        np.ones(shape, dtype=bool),
        np.full(shape, 20.0),
        None,
        np.full(shape, 0.1),
        np.full(shape, 0.8),
        np.array([[[np.nan, 0.8]]]),
        np.array([[[0.1, 0.2]]]),
        np.array([[[np.nan, 0.001]]]),
        np.ones(shape, dtype=np.int8),
        np.array([[[1, 3]]], dtype=np.int8),
        {"patlak": 1},
        True,
        "auto",
    )
    assert mask.tolist() == [[[1, 0]]]


def test_build_dce_qc_mask_skips_signal_qc_without_raw_signal() -> None:
    """Concentration-only inputs do not apply invalid signal-domain gates."""
    shape = (1, 1, 1)
    evaluation = _evaluate_dce_qc_mask(
        np.ones(shape, dtype=bool),
        np.zeros(shape),
        None,
        np.full(shape, 0.1),
        np.full(shape, 0.8),
        np.full(shape, 0.2),
        np.full(shape, 0.1),
        np.full(shape, 0.5),
        np.ones(shape, dtype=np.int8),
        np.zeros(shape, dtype=np.int8),
        {"patlak": 1},
        False,
        "extended_tofts",
    )

    assert evaluation.mask.tolist() == [[[1]]]
    assert evaluation.acceptance_policy["acquisition"] == {
        "minimum_tsnr": None,
        "minimum_peak_enhancement_percent": None,
    }


def test_build_dce_qc_mask_uses_model_aware_r_squared_thresholds() -> None:
    """Patlak retains non-negative R² while reversible fits require 0.5."""
    shape = (1, 1, 4)
    mask = _build_dce_qc_mask(
        np.ones(shape, dtype=bool),
        np.full(shape, 20.0),
        np.full(shape, 20.0),
        np.full(shape, 0.1),
        np.array([[[0.0, -0.01, 0.49, 0.5]]]),
        np.full(shape, 0.2),
        np.full(shape, 0.1),
        np.full(shape, 0.5),
        np.ones(shape, dtype=np.int8),
        np.array([[[1, 1, 3, 3]]], dtype=np.int8),
        {"patlak": 1},
        True,
        "auto",
    )
    assert mask.tolist() == [[[1, 0, 0, 1]]]


def test_build_dce_qc_mask_applies_patlak_threshold_in_fixed_mode() -> None:
    """Fixed Patlak uses the same non-negative R² acceptance as auto mode."""
    shape = (1, 1, 2)
    common = {
        "fitting_mask": np.ones(shape, dtype=bool),
        "tsnr": np.full(shape, 20.0),
        "peak_enhancement": np.full(shape, 20.0),
        "ktrans": np.full(shape, 0.1),
        "r_squared": np.array([[[0.0, -0.01]]]),
        "ve": np.full(shape, np.nan),
        "vp": np.full(shape, 0.1),
        "kep": np.full(shape, np.nan),
        "quality_flag": np.ones(shape, dtype=np.int8),
        "model_selection": np.zeros(shape, dtype=np.int8),
        "model_selection_codes": {"patlak": 1},
        "is_auto": False,
        "model_name": "patlak",
    }

    assert _build_dce_qc_mask(**common).tolist() == [[[1, 0]]]


def test_evaluate_dce_qc_mask_reports_acceptance_policy() -> None:
    """QC provenance distinguishes acquisition and model-fit acceptance."""
    shape = (1, 1, 1)
    evaluation = _evaluate_dce_qc_mask(
        np.ones(shape, dtype=bool),
        np.full(shape, 20.0),
        np.full(shape, 20.0),
        np.full(shape, 0.1),
        np.full(shape, 0.1),
        np.full(shape, np.nan),
        np.full(shape, 0.1),
        np.full(shape, np.nan),
        np.ones(shape, dtype=np.int8),
        np.ones(shape, dtype=np.int8),
        {"patlak": 1},
        True,
        "auto",
    )
    assert evaluation.acceptance_policy["acquisition"]["minimum_tsnr"] is None
    assert evaluation.acceptance_policy["fit"]["minimum_r_squared"] == {
        "patlak": 0.0,
        "reversible_models": 0.5,
    }


def test_dce_qc_mask_uses_configured_tsnr_threshold() -> None:
    """A non-default configured threshold controls acceptance and provenance."""
    shape = (1, 1, 2)
    evaluation = _evaluate_dce_qc_mask(
        np.ones(shape, dtype=bool),
        np.array([[[11.9, 12.0]]]),
        np.full(shape, 20.0),
        np.full(shape, 0.1),
        np.full(shape, 0.8),
        np.full(shape, 0.2),
        np.full(shape, 0.1),
        np.full(shape, 0.5),
        np.ones(shape, dtype=np.int8),
        np.zeros(shape, dtype=np.int8),
        {"patlak": 1},
        False,
        "extended_tofts",
        tsnr_threshold=12.0,
    )

    assert evaluation.mask.tolist() == [[[0, 1]]]
    assert evaluation.component_failures["tsnr"] == 1
    assert evaluation.acceptance_policy["acquisition"]["minimum_tsnr"] == 12.0


def test_default_zero_tsnr_threshold_is_diagnostic_only() -> None:
    """The default zero threshold never excludes an otherwise accepted fit."""
    shape = (1, 1, 1)
    evaluation = _evaluate_dce_qc_mask(
        np.ones(shape, dtype=bool),
        np.zeros(shape),
        np.full(shape, 20.0),
        np.full(shape, 0.1),
        np.full(shape, 0.8),
        np.full(shape, 0.2),
        np.full(shape, 0.1),
        np.full(shape, 0.5),
        np.ones(shape, dtype=np.int8),
        np.zeros(shape, dtype=np.int8),
        {"patlak": 1},
        False,
        "extended_tofts",
        tsnr_threshold=0.0,
    )

    assert evaluation.mask.tolist() == [[[1]]]
    assert evaluation.component_failures["tsnr"] == 0
    assert evaluation.acceptance_policy["acquisition"]["minimum_tsnr"] is None


def test_build_dce_qc_mask_rejects_quality_and_required_parameter_failures() -> None:
    """Only good fits with all model-required parameters are accepted."""
    shape = (1, 1, 4)
    common = {
        "fitting_mask": np.ones(shape, dtype=bool),
        "tsnr": np.full(shape, 20.0),
        "peak_enhancement": np.full(shape, 20.0),
        "ktrans": np.full(shape, 0.1),
        "r_squared": np.full(shape, 0.8),
        "ve": np.array([[[0.2, 0.2, np.nan, 0.2]]]),
        "vp": np.array([[[0.1, 0.1, 0.1, np.nan]]]),
        "kep": np.full(shape, 0.5),
        "quality_flag": np.array([[[1, 2, 1, 1]]], dtype=np.int8),
        "model_selection": np.zeros(shape, dtype=np.int8),
        "model_selection_codes": {"patlak": 1},
        "is_auto": False,
        "model_name": "extended_tofts",
    }
    mask = _build_dce_qc_mask(**common)
    assert mask.tolist() == [[[1, 0, 0, 0]]]


def test_build_dce_qc_mask_requires_finite_kep_for_reversible_model() -> None:
    """Undefined kep is acceptable for Patlak only, not Tofts-family fits."""
    shape = (1, 1, 1)
    arguments = {
        "fitting_mask": np.ones(shape, dtype=bool),
        "tsnr": np.full(shape, 20.0),
        "peak_enhancement": np.full(shape, 20.0),
        "ktrans": np.full(shape, 0.1),
        "r_squared": np.full(shape, 0.8),
        "ve": np.full(shape, 0.2),
        "vp": np.full(shape, 0.1),
        "kep": np.full(shape, np.nan),
        "quality_flag": np.ones(shape, dtype=np.int8),
        "model_selection": np.zeros(shape, dtype=np.int8),
        "model_selection_codes": {"patlak": 1},
        "is_auto": False,
    }
    reversible = _build_dce_qc_mask(
        **arguments,
        model_name="extended_tofts",
    )
    patlak = _build_dce_qc_mask(
        **arguments,
        model_name="patlak",
    )
    assert not reversible[0, 0, 0]
    assert patlak[0, 0, 0]


def test_evaluate_dce_qc_mask_reports_overlapping_component_failures() -> None:
    """QC provenance counts each failed criterion within the fitting domain."""
    shape = (1, 1, 3)
    evaluation = _evaluate_dce_qc_mask(
        np.ones(shape, dtype=bool),
        np.array([[[9.0, 20.0, 20.0]]]),
        np.array([[[20.0, 9.0, 20.0]]]),
        np.array([[[0.1, 0.1, 0.0]]]),
        np.array([[[0.8, 0.8, 0.4]]]),
        np.full(shape, 0.2),
        np.full(shape, 0.1),
        np.full(shape, 0.5),
        np.array([[[1, 1, 2]]], dtype=np.int8),
        np.zeros(shape, dtype=np.int8),
        {"patlak": 1},
        False,
        "extended_tofts",
        tsnr_threshold=10.0,
    )
    assert evaluation.mask.tolist() == [[[0, 0, 0]]]
    assert evaluation.component_failures == {
        "tsnr": 1,
        "peak_enhancement": 1,
        "ktrans_range": 1,
        "r_squared": 1,
        "quality_flag": 1,
        "model_parameters": 0,
    }


def test_compute_dce_coefficient_of_variation_preserves_auto_sentinels() -> None:
    """CV maps retain the strict epsilon gate, auto NaN, and zero background."""
    values = np.array([[[0.2, EPSILON_VE, 0.4, 0.8]]])
    standard_deviation = np.array([[[0.1, 0.1, 0.2, 0.4]]])
    mask = np.array([[[True, True, True, False]]])
    result = _compute_dce_coefficient_of_variation(
        values,
        standard_deviation,
        mask,
        True,
    )
    assert result.dtype == np.float32
    assert result[0, 0, 0] == pytest.approx(0.5)
    assert np.isnan(result[0, 0, 1])
    assert result[0, 0, 2] == pytest.approx(0.5)
    assert result[0, 0, 3] == 0.0


def test_compute_dce_coefficient_of_variation_marks_undefined_values() -> None:
    """Relative uncertainty is undefined when the estimate is zero."""
    result = _compute_dce_coefficient_of_variation(
        np.zeros((1, 1, 1)),
        np.ones((1, 1, 1)),
        np.ones((1, 1, 1), dtype=bool),
        False,
    )
    assert np.isnan(result[0, 0, 0])


def test_write_dce_map_overrides_inherited_header_dtype(tmp_path) -> None:
    """DCE serialization writes the requested dtype without inherited scaling."""
    header = nib.Nifti1Header()
    header.set_data_dtype(np.int16)
    output = tmp_path / "map.nii.gz"
    _write_dce_map(
        np.array([[[0.25, 1.5]]], dtype=np.float64),
        str(output),
        affine=np.eye(4),
        header=header,
    )
    image = nib.load(output)
    assert image.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_allclose(image.get_fdata(), [[[0.25, 1.5]]])
    assert header.get_data_dtype() == np.dtype(np.int16)


def test_write_dce_fitted_curves_preserves_four_dimensions(tmp_path) -> None:
    """Fitted-curve serialization retains shape, affine, values, and dtype."""
    curves = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    output = tmp_path / "curves.nii.gz"
    _write_dce_fitted_curves(curves, str(output), affine)
    image = nib.load(output)
    assert image.shape == curves.shape
    assert image.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_array_equal(image.get_fdata(), curves)
    np.testing.assert_array_equal(image.affine, affine)


@pytest.mark.parametrize("has_peak", [True, False])
def test_write_dce_model_free_maps_preserves_output_contract(has_peak: bool) -> None:
    """Model-free serialization retains suffixes, dtype, and peak optionality."""
    array = np.ones((1, 1, 1))
    maps = _DCEModelFreeMaps(
        array, array, array, array if has_peak else None, array,
        array, array, array, array, array.astype(np.int8), {}, {},
    )
    writes: list[tuple[str, object]] = []

    def generate_path(suffix: str) -> str:
        return f"output{suffix}.nii.gz"

    def write_map(data: np.ndarray, path: str, dtype: object = np.float32) -> None:
        writes.append((path, dtype))

    outputs = _write_dce_model_free_maps(maps, generate_path, write_map)
    assert outputs["out_iauc60"] == "output_IAUC60.nii.gz"
    assert outputs["out_peak_enhancement"] == "output_peak_enhancement.nii.gz"
    assert outputs["out_curve_type"] == "output_curve_type.nii.gz"
    assert writes[-1] == ("output_curve_type.nii.gz", np.int8)
    peak_writes = [path for path, _ in writes if "peak_enhancement" in path]
    assert bool(peak_writes) is has_peak


@pytest.mark.parametrize(
    ("save_ve", "save_vp", "save_2cxm", "optional_keys"),
    [
        (False, True, False, {"out_vp"}),
        (True, True, True, {"out_ve", "out_kep", "out_vp", "out_fp", "out_ps", "out_e"}),
    ],
)
def test_write_dce_parameter_maps_preserves_applicability(
    save_ve: bool,
    save_vp: bool,
    save_2cxm: bool,
    optional_keys: set[str],
) -> None:
    """PK output assembly retains parameter and model applicability."""
    array = np.ones((1, 1, 2))
    maps = SimpleNamespace(
        ktrans=array, r_squared=array, quality_flag=array.astype(np.int8),
        vp_at_bound=array, ve=array, kep=array, vp=array, fp=array, ps=array,
        extraction_fraction=array, save_2cxm=save_2cxm,
    )
    writes: list[tuple[str, object, np.ndarray]] = []

    def write_map(data: np.ndarray, path: str, dtype: object = np.float32) -> None:
        writes.append((path, dtype, data))

    outputs = _write_dce_parameter_maps(
        maps,
        np.array([[[True, False]]]),
        save_ve,
        save_vp,
        lambda suffix: f"output{suffix}.nii.gz",
        write_map,
    )
    always = {"out_ktrans", "out_r2", "out_quality_flag", "out_vp_at_bound_map"}
    assert set(outputs) == always | optional_keys
    quality_write = next(write for write in writes if "quality_flag" in write[0])
    assert quality_write[1] is np.uint8
    bound_write = next(write for write in writes if "vpAtBound" in write[0])
    assert bound_write[2].tolist() == [[[1.0, 0.0]]]


@pytest.mark.parametrize(
    ("save_ve", "save_vp", "save_2cxm", "expected_count"),
    [(False, False, False, 4), (True, True, True, 16)],
)
def test_write_dce_uncertainty_maps_preserves_applicability(
    save_ve: bool,
    save_vp: bool,
    save_2cxm: bool,
    expected_count: int,
) -> None:
    """Uncertainty output groups retain model-specific applicability."""
    values = np.full((1, 1, 1), 0.2)
    sigma = np.full((1, 1, 1), 0.1)
    maps = SimpleNamespace(
        ktrans=values, ve=values, vp=values, fp=values, ps=values,
        ktrans_std=sigma, ve_std=sigma, vp_std=sigma,
        fp_std=sigma, ps_std=sigma, save_2cxm=save_2cxm,
        ktrans_ci95_lower=values - sigma,
        ktrans_ci95_upper=values + sigma,
        ve_ci95_lower=values - sigma,
        ve_ci95_upper=values + sigma,
        vp_ci95_lower=values - sigma,
        vp_ci95_upper=values + sigma,
        fp_ci95_lower=values - sigma,
        fp_ci95_upper=values + sigma,
        ps_ci95_lower=values - sigma,
        ps_ci95_upper=values + sigma,
    )
    writes: list[str] = []
    outputs = _write_dce_uncertainty_maps(
        maps,
        np.ones((1, 1, 1), dtype=bool),
        False,
        save_ve,
        save_vp,
        lambda suffix: f"output{suffix}.nii.gz",
        lambda data, path: writes.append(path),
    )
    assert len(outputs) == expected_count
    assert writes == list(outputs.values())
    assert ("out_ktrans_cv" in outputs) is not save_2cxm
    assert ("out_fp_cv" in outputs) is save_2cxm
    assert ("out_ktrans_std" in outputs) is not save_2cxm
    assert ("out_ktrans_ci95_lower" in outputs) is not save_2cxm
    assert ("out_fp_std" in outputs) is save_2cxm
    assert ("out_fp_ci95_upper" in outputs) is save_2cxm


def test_auto_uncertainty_retains_ktrans_cv_outside_2cxm_voxels() -> None:
    """Auto mode retains its mixed-model Ktrans diagnostic derivative."""
    values = np.full((1, 1, 1), 0.2)
    sigma = np.full((1, 1, 1), 0.1)
    maps = SimpleNamespace(
        ktrans=values,
        ve=values,
        vp=values,
        fp=values,
        ps=values,
        ktrans_std=sigma,
        ve_std=sigma,
        vp_std=sigma,
        fp_std=sigma,
        ps_std=sigma,
        save_2cxm=True,
        ktrans_ci95_lower=values - sigma,
        ktrans_ci95_upper=values + sigma,
        ve_ci95_lower=values - sigma,
        ve_ci95_upper=values + sigma,
        vp_ci95_lower=values - sigma,
        vp_ci95_upper=values + sigma,
        fp_ci95_lower=values - sigma,
        fp_ci95_upper=values + sigma,
        ps_ci95_lower=values - sigma,
        ps_ci95_upper=values + sigma,
    )

    outputs = _write_dce_uncertainty_maps(
        maps,
        np.ones((1, 1, 1), dtype=bool),
        True,
        True,
        True,
        lambda suffix: f"output{suffix}.nii.gz",
        lambda data, path: None,
    )

    assert "out_ktrans_cv" in outputs
    assert "out_ktrans_std" in outputs
    assert "out_ktrans_ci95_lower" in outputs


@pytest.mark.parametrize("is_auto", [False, True])
def test_write_dce_terminal_outputs_preserves_auto_applicability(is_auto: bool) -> None:
    """Terminal output assembly emits selection maps only in auto mode."""
    shape = (1, 1, 1)
    values = np.full(shape, 0.2)
    maps = SimpleNamespace(
        ktrans=values,
        r_squared=np.full(shape, 0.8),
        ve=values,
        vp=values,
        kep=np.full(shape, 0.01),
        optimizer_converged=np.ones(shape, dtype=bool),
        quality_flag=np.ones(shape, dtype=np.int8),
        model_selection=np.ones(shape, dtype=np.int8),
        model_selection_codes={"patlak": 1},
        delta_bic=values,
        fitted_curves=None,
    )
    writes: list[tuple[str, object]] = []

    def write_map(data: np.ndarray, path: str, dtype: object = np.float32) -> None:
        writes.append((path, dtype))

    qc_metrics: dict[str, object] = {
        "voxel_accounting": {
            "denominators": {"brain": 1, "attempted": 1, "retained": 1},
        }
    }
    outputs = _write_dce_terminal_outputs(
        maps,
        np.ones(shape, dtype=bool),
        np.full(shape, 20.0),
        None,
        is_auto,
        "auto" if is_auto else "tofts",
        False,
        np.eye(4),
        lambda suffix: f"output{suffix}.nii.gz",
        write_map,
        qc_metrics,
    )
    assert {
        "out_tsnr", "out_fitting_mask", "out_optimizer_converged_mask",
        "out_qc_mask",
    }.issubset(outputs)
    assert ("out_model_selection" in outputs) is is_auto
    assert ("out_delta_bic" in outputs) is is_auto
    assert next(dtype for path, dtype in writes if "qc_mask" in path) is np.uint8
    accepted = qc_metrics["voxel_accounting"]["quantitative_qc_accepted"]
    assert accepted["count"] == 1
    assert accepted["fraction_of_retained"] == 1.0
    assert qc_metrics["qc_component_failures"] == {
        "scope": "fitted_voxels",
        "counts_are_overlapping": True,
        "tsnr": 0,
        "peak_enhancement": 0,
        "ktrans_range": 0,
        "r_squared": 0,
        "quality_flag": 0,
        "model_parameters": 0,
    }
    if is_auto:
        assert next(dtype for path, dtype in writes if "modelSelection" in path) is np.int8


def test_saved_dce_categorical_maps_are_brain_masked(tmp_path) -> None:
    """Saved curve, fit-quality, and model-selection labels have zero background."""
    shape = (4, 4, 4)
    n_timepoints = 40
    time_points = np.linspace(0.0, 4.0, n_timepoints)
    aif = (
        5.0 * np.exp(-0.5 * ((time_points - 0.5) / 0.15) ** 2)
        + 1.5 * np.exp(-0.3 * np.maximum(time_points - 0.5, 0.0))
    )
    tissue_curve = batched_tofts_model(
        np.array([0.15]),
        np.array([0.3]),
        aif,
        time_points,
    ).tissue_curves[0]

    concentration = np.broadcast_to(tissue_curve, (*shape, n_timepoints)).copy()
    mask = np.ones(shape, dtype=np.uint8)
    mask[0, :, :] = 0

    concentration_file = tmp_path / "concentration.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    aif_file = tmp_path / "aif.txt"
    time_points_file = tmp_path / "time_points.txt"
    nib.save(nib.Nifti1Image(concentration.astype(np.float32), np.eye(4)), concentration_file)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)
    np.savetxt(aif_file, aif)
    np.savetxt(time_points_file, time_points)

    interface = ComputeDCEMaps(
        concentration_file=str(concentration_file),
        aif_file=str(aif_file),
        brain_mask_file=str(mask_file),
        model="auto",
        inference="nlls",
        time_points_file=str(time_points_file),
        baseline_frames=5,
        constraint_mode="skip",
        n_procs=1,
    )
    workdir = tmp_path / "run"
    workdir.mkdir()
    previous_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        result = interface.run()
    finally:
        os.chdir(previous_cwd)

    background = mask == 0
    for output_name in ("out_curve_type", "out_quality_flag", "out_model_selection"):
        data = nib.load(getattr(result.outputs, output_name)).get_fdata()
        assert np.all(data[background] == 0), output_name
    for output_name in (
        "out_ktrans_std", "out_ve_std", "out_vp_std",
        "out_ktrans_ci95_lower", "out_ktrans_ci95_upper",
        "out_ve_ci95_lower", "out_ve_ci95_upper",
        "out_vp_ci95_lower", "out_vp_ci95_upper",
    ):
        assert Path(getattr(result.outputs, output_name)).exists(), output_name
