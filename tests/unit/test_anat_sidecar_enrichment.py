"""PR1 (BIDS arc finishing): anat per-image sidecar flat-key enrichment.

Asserts that the rewritten Phase 1 anat enrich helpers produce flat-prefixed
PascalCase top-level keys (no nested blocks) per bids-spec-expert round 3
(DSC) + round 1 (PR1).
"""
from __future__ import annotations

from typing import Any

import pytest

from pmriprep.utils.sidecar_builders import (
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
    def test_t1w_no_coreg_keys(self, base_sidecar):
        qc = {
            "preprocessing": {
                "denoising": True, "bias_correction": "N4", "acpc_aligned": True,
            },
            "coregistration": {"method": "flirt"},  # ignored when coreg_modality=None
        }
        out = enrich_preproc_anat_sidecar(dict(base_sidecar), qc, coreg_modality=None)
        assert out["PreprocessingDenoising"] is True
        assert out["PreprocessingBiasCorrection"] == "N4"
        assert out["PreprocessingACPCAligned"] is True
        assert all(not k.startswith("Coregistration") for k in out)

    def test_t2w_with_coreg_keys(self, base_sidecar):
        qc = {
            "preprocessing": {"denoising": False},
            "coregistration": {
                "method": "flirt", "cost_function": "corratio",
                "correlation_ratio": 0.94,
            },
        }
        out = enrich_preproc_anat_sidecar(dict(base_sidecar), qc, coreg_modality="T2w")
        assert out["CoregistrationT2wMethod"] == "flirt"
        assert out["CoregistrationT2wCostFunction"] == "corratio"
        assert out["CoregistrationT2wCorrelationRatio"] == pytest.approx(0.94)

    def test_no_nested_blocks(self, base_sidecar):
        qc = {
            "preprocessing": {"denoising": True},
            "coregistration": {"method": "flirt"},
        }
        out = enrich_preproc_anat_sidecar(dict(base_sidecar), qc, coreg_modality="FLAIR")
        # No "Preprocessing" or "Coregistration" nested blocks
        assert "Preprocessing" not in out
        assert "Coregistration" not in out
