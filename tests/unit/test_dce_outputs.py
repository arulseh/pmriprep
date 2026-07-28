"""Tests for DCE outputs sidecar generation."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.utils.sidecar_builders import build_dce_base_sidecar
from pmriprep.workflows.dce.datasinks import (
    _build_delta_bic_meta,
    _build_model_selection_meta,
)


def _ref_nifti(tmp_path: Path) -> Path:
    p = tmp_path / "ref.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), p)
    return p


def test_pk_sidecar_records_plasma_input_compartment(tmp_path: Path) -> None:
    sidecar = build_dce_base_sidecar(
        source_paths=[],
        reference_voxel_source=_ref_nifti(tmp_path),
        bids_dir=str(tmp_path),
        subject="01", session="01",
        provenance_dict={"PerfusionProcessing": {"modality": "DCE"}},
        qc_metrics={"aif_detection": {"hematocrit": 0.42}},
    )
    assert sidecar["InputFunctionCompartment"] == "plasma"
    assert "KtransConvention" not in sidecar
    assert "KtransConventionHct" not in sidecar


def test_non_pk_sidecar_omits_input_compartment(tmp_path: Path) -> None:
    sidecar = build_dce_base_sidecar(
        source_paths=[],
        reference_voxel_source=_ref_nifti(tmp_path),
        bids_dir=str(tmp_path),
        subject="01", session="01",
        provenance_dict={"PerfusionProcessing": {"modality": "DCE"}},
        qc_metrics={"aif_detection": {"hematocrit": 0.38}},
        pk_dependent=False,
    )
    assert "InputFunctionCompartment" not in sidecar


def test_minimal_sidecar_omits_input_compartment(tmp_path: Path) -> None:
    sidecar = build_dce_base_sidecar(
        source_paths=[],
        reference_voxel_source=_ref_nifti(tmp_path),
        bids_dir=str(tmp_path),
        subject="01", session="01",
        provenance_dict={"PerfusionProcessing": {"modality": "DCE"}},
        qc_metrics={"aif_detection": {"hematocrit": 0.42}},
        minimal=True,
    )
    assert "InputFunctionCompartment" not in sidecar


def test_sidecar_carries_m3_fields(tmp_path: Path) -> None:
    sidecar = build_dce_base_sidecar(
        source_paths=[tmp_path / "src.nii.gz"],
        reference_voxel_source=_ref_nifti(tmp_path),
        bids_dir=str(tmp_path),
        subject="01", session="01",
    )
    assert sidecar["SpatialReference"].startswith("bids::sub-")
    assert sidecar["Resolution"] == "native"
    assert isinstance(sidecar["VoxelSize"], list) and len(sidecar["VoxelSize"]) == 3
    assert isinstance(sidecar["Sources"], list)


@pytest.mark.parametrize("pk_modeling", [None, "invalid", 3.0, []])
def test_dce_diagnostic_sidecars_ignore_nonmapping_pk_qc(pk_modeling: object) -> None:
    """Malformed optional PK QC does not prevent derivative sidecar creation."""
    qc_metrics = {"pk_modeling": pk_modeling}

    model_meta = _build_model_selection_meta({}, qc_metrics)
    bic_meta = _build_delta_bic_meta({}, qc_metrics)

    assert "ModelDistribution" not in model_meta
    assert "R2Median" not in bic_meta
    assert "ConvergenceRate" not in bic_meta


def test_dce_diagnostic_sidecars_carry_canonical_aif_provenance() -> None:
    """AIF-dependent BIC diagnostics use the shared DCE AIF block."""
    from pmriprep.utils.aif_provenance import build_aif_provenance_block

    qc_metrics = {
        "aif_detection": {
            "detection_method": "atlas",
            "aif_confidence": "reduced",
            "peak_scaling_enabled": True,
            "peak_scaling_applied": True,
            "raw_peak_mm": 2.0,
            "emitted_peak_mm": 6.0,
            "peak_scaling_factor": 3.0,
        },
        "pk_modeling": {},
    }
    expected = build_aif_provenance_block(qc_metrics, modality="DCE")

    for sidecar in (
        _build_model_selection_meta({}, qc_metrics),
        _build_delta_bic_meta({}, qc_metrics),
    ):
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "qualitative_only"
        assert sidecar["PerfusionProcessing"]["AIF"] == expected
