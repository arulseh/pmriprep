"""PR1 (BIDS arc finishing): anat per-image sidecar flat-key enrichment.

Asserts that the rewritten Phase 1 anat enrich helpers produce flat-prefixed
PascalCase top-level keys (no nested blocks) per bids-spec-expert round 3
(DSC) + round 1 (PR1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from pmriprep.utils.sidecar_builders import (
    build_anat_preproc_sidecar,
    enrich_brain_mask_sidecar,
    enrich_preproc_anat_sidecar,
)


@pytest.fixture
def base_sidecar() -> dict[str, Any]:
    """Representative minimal-base sidecar from build_anat_*_sidecar."""
    return {
        "Description": "Brain mask from HD-BET",
        "SkullStripped": True,
        "SpatialReference": "bids::sub-X/anat/sub-X_space-T1w_desc-preproc_T1w.nii.gz",
        "Sources": ["bids:raw:sub-X/anat/sub-X_T1w.nii.gz"],
        "VoxelSize": [1.0, 1.0, 1.0],
    }


class TestBrainMaskFlatKeys:
    def test_all_fields_present(self, base_sidecar):
        qc = {
            "brain_extraction": {
                "tool": "hdbet", "version": "v1.0",
                "brain_volume_ml": 1500.0, "brain_volume_mm3": 1_500_000.0,
            },
        }
        out = enrich_brain_mask_sidecar(dict(base_sidecar), qc)
        assert out["BrainExtractionTool"] == "hdbet"
        # Retired per-file key; tool version now lives in dataset-level GeneratedBy.
        assert "BrainExtractionVersion" not in out
        assert out["BrainExtractionVolume"] == pytest.approx(1500.0)
        assert out["BrainExtractionVolumeMm3"] == pytest.approx(1_500_000.0)
        # No nested block (regression guard)
        assert "BrainExtraction" not in out
        # No unit-suffix key (round-3 anti-suffix rule)
        assert "BrainExtractionVolume_mL" not in out
        # Base fields preserved
        assert out["SkullStripped"] is True
        assert out["SpatialReference"] == base_sidecar["SpatialReference"]

    def test_partial_qc(self, base_sidecar):
        qc = {"brain_extraction": {"tool": "hdbet"}}
        out = enrich_brain_mask_sidecar(dict(base_sidecar), qc)
        assert out["BrainExtractionTool"] == "hdbet"
        # Optional fields omitted when not in source
        assert "BrainExtractionVersion" not in out
        assert "BrainExtractionVolume" not in out


class TestPreprocAnatFlatKeys:
    def test_measured_preprocessing_setting(self, base_sidecar):
        qc = {"preprocessing": {"denoising": True}}
        out = enrich_preproc_anat_sidecar(dict(base_sidecar), qc)
        assert out["PreprocessingDenoising"] is True

    def test_unproduced_fields_are_not_enriched(self, base_sidecar):
        qc = {
            "preprocessing": {"denoising": False},
            "coregistration": {
                "method": "flirt", "cost_function": "corratio",
                "correlation_ratio": 0.94,
            },
        }
        out = enrich_preproc_anat_sidecar(dict(base_sidecar), qc)
        assert all(not key.startswith("Coregistration") for key in out)

    def test_no_nested_blocks(self, base_sidecar):
        qc = {
            "preprocessing": {"denoising": True},
            "coregistration": {"method": "flirt"},
        }
        out = enrich_preproc_anat_sidecar(dict(base_sidecar), qc)
        # No "Preprocessing" or "Coregistration" nested blocks
        assert "Preprocessing" not in out
        assert "Coregistration" not in out


@pytest.mark.parametrize(
    ("processing_kind", "expected"),
    [
        ("reference", False),
        ("coregistered", True),
        ("cropped_coregistered", True),
        ("subtraction", True),
    ],
)
def test_anat_preproc_sidecar_records_coregistration(
    tmp_path: Path, processing_kind: str, expected: bool,
) -> None:
    """Structured registration provenance does not depend on prose."""
    source = tmp_path / "sub-01_T1w.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)),
        source,
    )

    sidecar = build_anat_preproc_sidecar(
        description="test",
        source_paths=[source],
        reference_voxel_source=source,
        bids_dir=tmp_path,
        subject="01",
        session=None,
        processing_kind=processing_kind,
    )

    assert sidecar["CoregistrationApplied"] is expected
