"""Tests for DCE outputs sidecar generation."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.utils.sidecar_builders import build_dce_base_sidecar


def _ref_nifti(tmp_path: Path) -> Path:
    p = tmp_path / "ref.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), p)
    return p


def test_sidecar_ktrans_convention_key_default_hct(tmp_path: Path) -> None:
    sidecar = build_dce_base_sidecar(
        source_paths=[],
        reference_voxel_source=_ref_nifti(tmp_path),
        bids_dir=str(tmp_path),
        subject="01", session="01",
        provenance_dict={"PerfusionProcessing": {"modality": "DCE"}},
        qc_metrics={"aif_detection": {"hematocrit": 0.42}},
    )
    assert sidecar["KtransConvention"] == "plasma_Tofts1999"
    assert sidecar["KtransConventionHct"] == pytest.approx(0.42)


def test_sidecar_ktrans_convention_key_honours_hct_override(tmp_path: Path) -> None:
    sidecar = build_dce_base_sidecar(
        source_paths=[],
        reference_voxel_source=_ref_nifti(tmp_path),
        bids_dir=str(tmp_path),
        subject="01", session="01",
        provenance_dict={"PerfusionProcessing": {"modality": "DCE"}},
        qc_metrics={"aif_detection": {"hematocrit": 0.38}},
    )
    assert sidecar["KtransConvention"] == "plasma_Tofts1999"
    assert sidecar["KtransConventionHct"] == pytest.approx(0.38)


def test_minimal_sidecar_omits_convention_keys(tmp_path: Path) -> None:
    sidecar = build_dce_base_sidecar(
        source_paths=[],
        reference_voxel_source=_ref_nifti(tmp_path),
        bids_dir=str(tmp_path),
        subject="01", session="01",
        provenance_dict={"PerfusionProcessing": {"modality": "DCE"}},
        qc_metrics={"aif_detection": {"hematocrit": 0.42}},
        minimal=True,
    )
    assert "KtransConvention" not in sidecar
    assert "KtransConventionHct" not in sidecar


def test_sidecar_carries_m3_fields(tmp_path: Path) -> None:
    sidecar = build_dce_base_sidecar(
        source_paths=[tmp_path / "src.nii.gz"],
        reference_voxel_source=_ref_nifti(tmp_path),
        bids_dir=str(tmp_path),
        subject="01", session="01",
    )
    assert sidecar["SpatialReference"].startswith("bids::pmriprep/")
    assert sidecar["Resolution"] == "native"
    assert isinstance(sidecar["VoxelSize"], list) and len(sidecar["VoxelSize"]) == 3
    assert isinstance(sidecar["Sources"], list)
