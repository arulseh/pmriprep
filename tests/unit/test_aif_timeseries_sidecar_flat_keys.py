"""Phase 2a: AIF curve sidecar uses flat ``AIF*``-prefixed top-level keys.

Asserts that ``build_aif_timeseries_sidecar`` (and the qc_writers AIF
sidecar emitter) produce a JSON sidecar with flat top-level keys (NOT
nested under an ``AIFDetection`` block — that nesting was the round-1
shape-parity smell).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import pytest

from pmriprep.utils.sidecar_builders import build_aif_timeseries_sidecar


@pytest.fixture
def aif_qc() -> dict[str, Any]:
    """Producer-shape AIF QC dict (subset matching dsc_aif.py output)."""
    return {
        "detection_method": "atlas_strict",
        "n_aif_voxels_selected": 6,
        "best_psr": 78.0,                # → 0.78 after /100
        "best_brr": 92.0,                # → 0.92 after /100
        "quality_score": 0.81,
        "cascade_final_level": "atlas_strict",
        "hemisphere_coverage": "bilateral",
        "gamma_fit_applied": True,
        "gamma_fit_r_squared": 0.94,
        "gamma_fit_auc_ratio": 0.97,
        "synthetic": False,
        "cbv_auc_denominator_is_synthetic": False,
        "hematocrit": 0.42,
        "aif_metrics": {
            "peak_conc": 12.5,
            "fwhm": 6.2,
            "ttp": 18.0,
            "arrival": 9.5,
        },
    }


@pytest.fixture
def bids_root(tmp_path) -> Path:
    bids = tmp_path / "bids"
    perf = bids / "sub-X" / "ses-Y" / "perf"
    perf.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = perf / "sub-X_ses-Y_dsc.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src)
    return bids


class TestFlatKeyShape:
    def test_dsc_flat_keys_present(self, aif_qc, bids_root):
        sidecar = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[bids_root / "sub-X" / "ses-Y" / "perf" / "sub-X_ses-Y_dsc.nii.gz"],
            bids_dir=bids_root,
            subject="X",
            session="Y",
            aif_qc=aif_qc,
        )
        # Flat top-level keys, NOT nested
        assert sidecar["AIFMethod"] == "atlas_strict"
        assert sidecar["AIFNumberOfVoxelsSelected"] == 6
        assert sidecar["AIFPeakAmplitude"] == pytest.approx(12.5)
        assert sidecar["AIFFWHM"] == pytest.approx(6.2)
        assert sidecar["AIFTimeToPeak"] == pytest.approx(18.0)
        assert sidecar["AIFArrival"] == pytest.approx(9.5)
        assert sidecar["AIFBaselineReturnRatio"] == pytest.approx(0.92)
        assert sidecar["AIFPeakSignalRatio"] == pytest.approx(0.78)
        assert sidecar["AIFQualityScore"] == pytest.approx(0.81)
        assert sidecar["AIFArterialCascadeTier"] == "atlas_strict"
        assert sidecar["AIFHemisphereCoverage"] == "bilateral"
        assert sidecar["AIFGammaFitApplied"] is True
        assert sidecar["AIFGammaFitRSquared"] == pytest.approx(0.94)
        assert sidecar["AIFGammaFitAUCRatio"] == pytest.approx(0.97)
        assert sidecar["AIFSynthetic"] is False
        assert sidecar["AIFUsedAsCBVDenominator"] is False
        assert sidecar["AIFHematocrit"] == pytest.approx(0.42)

    def test_no_legacy_nested_block(self, aif_qc, bids_root):
        """Regression: must not re-emit the round-1 AIFDetection nest."""
        sidecar = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc=aif_qc,
        )
        assert "AIFDetection" not in sidecar
        # Old PascalCase-with-units keys must not appear
        assert "PeakConcentration_mM" not in sidecar
        assert "FWHM_seconds" not in sidecar

    def test_dce_modality_units(self, aif_qc, bids_root):
        sidecar = build_aif_timeseries_sidecar(
            modality="DCE",
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc=aif_qc,
        )
        assert sidecar["Units"] == "mM"
        # Same flat AIF* keys regardless of modality
        assert sidecar["AIFPeakAmplitude"] == pytest.approx(12.5)


class TestOptionalKeysOmitted:
    def test_missing_aif_qc(self, bids_root):
        sidecar = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc=None,
        )
        # No AIF* keys at all
        assert all(not k.startswith("AIF") for k in sidecar)
        # But base sidecar still has Description / Columns / Units
        assert sidecar["Description"].startswith("Arterial input function")
        assert sidecar["Units"] == "1/s"

    def test_partial_aif_qc(self, bids_root):
        sidecar = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc={"detection_method": "synthetic", "synthetic": True},
        )
        assert sidecar["AIFMethod"] == "synthetic"
        assert sidecar["AIFSynthetic"] is True
        # Keys not in source dict are omitted, not null
        assert "AIFFWHM" not in sidecar
        assert "AIFGammaFitApplied" not in sidecar


class TestRejectsBadModality:
    def test_invalid_modality_raises(self, bids_root):
        with pytest.raises(ValueError, match="modality must be"):
            build_aif_timeseries_sidecar(
                modality="ASL",
                source_paths=[],
                bids_dir=bids_root,
                subject="X",
                session=None,
                aif_qc=None,
            )
