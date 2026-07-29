"""Tests for DCE spatial-smoothing policy."""

from pathlib import Path

import nibabel as nib
import numpy as np

from pmriprep.workflows.dce._helpers import _derive_smoothed_kep
from pmriprep.workflows.dce.base import init_dce_smoothing_wf
from pmriprep.workflows.dce.datasinks import (
    _build_dce_smoothed_meta,
    _build_model_free_extra,
)
from pmriprep.workflows.dsc.datasinks import _build_smoothed_meta


def _save_map(path: Path, data: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
    return str(path)


def test_derive_smoothed_kep_uses_smoothed_components(
    tmp_path: Path, monkeypatch,
) -> None:
    """Kep is derived from smoothed Ktrans and ve with validity protection."""
    ktrans = _save_map(
        tmp_path / "ktrans_fwhm3.nii.gz",
        np.array([[[0.5, 0.2, np.nan, 0.4]]]),
    )
    ve = _save_map(
        tmp_path / "ve_fwhm3.nii.gz",
        np.array([[[0.25, 0.0001, 0.4, 0.02]]]),
    )
    native_kep = _save_map(
        tmp_path / "kep.nii.gz", np.array([[[2.0, 1.0, np.nan, 5.0]]])
    )
    mask = _save_map(tmp_path / "mask.nii.gz", np.array([[[1, 1, 1, 0]]]))
    monkeypatch.chdir(tmp_path)

    result = _derive_smoothed_kep(
        {"ktrans": ktrans, "ve": ve}, native_kep, mask,
        epsilon_ve=0.001, max_kep=10.0,
    )

    assert set(result) == {"ktrans", "ve", "kep"}
    kep = nib.load(result["kep"]).get_fdata()
    np.testing.assert_allclose(
        kep,
        np.array([[[2.0, np.nan, np.nan, 0.0]]]),
        equal_nan=True,
    )


def test_dce_smoothing_workflow_excludes_uncertainty_and_native_kep() -> None:
    """Only independent PK inputs are smoothed and uncertainty stays native."""
    workflow = init_dce_smoothing_wf(fwhm=3.0)

    input_fields = set(workflow.get_node("inputnode").interface.inputs.copyable_trait_names())
    assert "kep" not in input_fields
    assert "kep_validity" in input_fields
    assert "ktrans_std" not in input_fields
    assert "ve_std" not in input_fields
    assert "vp_std" not in input_fields
    assert workflow.get_node("derive_smoothed_kep") is not None
    assert "smooth_uncertainty_maps" not in workflow.list_node_names()


def test_smoothed_metadata_separates_smoothing_from_output_masking() -> None:
    """Sidecars describe the smoothing operation and output mask separately."""
    dce_meta = _build_dce_smoothed_meta({}, "Smoothed Ktrans", 3.0)
    dsc_meta = _build_smoothed_meta({}, "Smoothed CBF", "rel_cbf", None, None, 3.0)

    for metadata in (dce_meta, dsc_meta):
        assert metadata["SmoothingType"] == "Gaussian"
        assert metadata["SmoothingFWHM"] == 3.0
        assert metadata["OutputMaskApplied"] is True
        assert "SmoothingMasked" not in metadata


def test_smoothed_metadata_limits_input_compartment_to_pk_maps() -> None:
    """Model-free smoothed maps do not inherit PK-input metadata."""
    base = {"InputFunctionCompartment": "plasma"}
    pk_meta = _build_dce_smoothed_meta(
        base, "Smoothed Ktrans", 3.0, map_key="ktrans",
    )
    model_free_meta = _build_dce_smoothed_meta(
        base, "Smoothed IAUC60", 3.0, map_key="iauc60",
    )

    assert pk_meta["InputFunctionCompartment"] == "plasma"
    assert "InputFunctionCompartment" not in model_free_meta


def test_smoothed_ser_metadata_records_configured_windows() -> None:
    """Smoothed SER provenance retains the exact protocol-defined timing."""
    metadata = _build_dce_smoothed_meta(
        {}, "Smoothed SER", 3.0, map_key="ser",
        config_dict={
            "ser_early_time_s": 90.0,
            "ser_late_time_s": 300.0,
        },
    )

    assert metadata["InputCurve"] == "motion_corrected_signal"
    assert metadata["EarlySampleTimeSecondsAfterReference"] == 90.0
    assert metadata["LateSampleTimeSecondsAfterReference"] == 300.0
    assert metadata["SampleInterpolation"] == "linear"
    assert metadata["MinimumEnhancementNoiseSD"] == 3.0


def test_native_ser_metadata_records_configured_windows() -> None:
    """Native SER provenance uses configured rather than static defaults."""
    metadata = _build_model_free_extra(
        {"InputCurve": "motion_corrected_signal"},
        "ser",
        {
            "ser_early_time_s": 90.0,
            "ser_late_time_s": 300.0,
        },
    )

    assert metadata is not None
    assert metadata["EarlySampleTimeSecondsAfterReference"] == 90.0
    assert metadata["LateSampleTimeSecondsAfterReference"] == 300.0
    assert metadata["SampleInterpolation"] == "linear"
    assert metadata["MinimumEnhancementNoiseSD"] == 3.0
