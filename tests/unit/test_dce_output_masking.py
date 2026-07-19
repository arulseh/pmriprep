"""Integration tests for masking saved DCE quantitative derivatives."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np

from pmriprep.interfaces.dce_compute import (
    ComputeDCEMaps,
    _build_dce_qc_mask,
    _zero_dce_background,
)
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
        np.zeros(shape, dtype=np.int8),
        {"patlak": 1},
        False,
        "tofts",
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
        np.array([[[1, 3]]], dtype=np.int8),
        {"patlak": 1},
        True,
        "auto",
    )
    assert mask.tolist() == [[[1, 0]]]


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
