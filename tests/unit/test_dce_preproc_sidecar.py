"""PR1 (BIDS arc finishing): DCE preproc-4D sidecar carries flat-key cosmetic QC.

Mirrors v0.45.0 ``test_dsc_preproc_sidecar_carries_flat_qc.py``.
"""
from __future__ import annotations

from typing import Any

import pytest

from pmriprep.utils.sidecar_builders import enrich_dce_preproc_sidecar


@pytest.fixture
def base_sidecar() -> dict[str, Any]:
    return {
        "Description": "Motion-corrected 4D DCE time series",
        "SkullStripped": False,
        "SpatialReference": "bids::pmriprep/sub-X/perf/sub-X_desc-dceMean_stat.nii.gz",
        "Sources": ["bids::sub-X/perf/sub-X_dce.nii.gz"],
        "Resolution": "native",
        "VoxelSize": [1.0, 1.0, 3.0],
        "KtransConvention": "plasma_Tofts1999",
    }


@pytest.fixture
def dce_qc_metrics() -> dict[str, Any]:
    """Representative DCE merged qc_metrics (from workflows/dce/base.py:3286)."""
    return {
        "t1_mapping": {
            "method": "hybrid",
            "b1_correction": "afi",
            "t1_correction": {
                "correction_applied": "offset", "tissue_t1_used": False,
            },
        },
        "b1_mapping": {"mean_b1": 0.96, "pct_correction_gt_20": 8.5},
        "preprocessing": {
            "fd_mean": 0.21, "fd_max": 0.78, "percent_outliers": 2.1,
            "motion_status": "PASS",
            "baseline_tsnr": 32.4, "peak_enh_pct": 65.0,
            "n_volumes": 75, "tr_ms": 4060.0,
        },
        "nawm_normalization": True,
    }


class TestT1Keys:
    def test_t1_mapping_keys(self, base_sidecar, dce_qc_metrics):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=dce_qc_metrics)
        assert out["T1MappingMethod"] == "hybrid"
        assert out["T1B1Correction"] == "afi"
        assert out["T1CorrectionApplied"] == "offset"
        assert out["T1UseTissueLiterature"] is False


class TestB1Keys:
    def test_b1_mapping_keys(self, base_sidecar, dce_qc_metrics):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=dce_qc_metrics)
        assert out["B1MeanWholeBrain"] == pytest.approx(0.96)
        assert out["B1PercentVoxelsCorrectionAbove20"] == pytest.approx(8.5)


class TestMotionAndSignalKeys:
    def test_unprefixed_motion_signal(self, base_sidecar, dce_qc_metrics):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=dce_qc_metrics)
        # Unprefixed (matches DSC convention; future desc-dcePreproc_dce
        # and desc-dscPreproc_dsc share these key names)
        assert out["MotionFDMean"] == pytest.approx(0.21)
        assert out["MotionFDMax"] == pytest.approx(0.78)
        assert out["MotionPercentOutliers"] == pytest.approx(2.1)
        assert out["MotionStatus"] == "PASS"
        assert out["BaselineTSNR"] == pytest.approx(32.4)
        assert out["PeakEnhancementPercent"] == pytest.approx(65.0)
        assert out["NumberOfVolumes"] == 75
        # tr_ms (4060ms) → RepetitionTime in seconds
        assert out["RepetitionTime"] == pytest.approx(4.06)


class TestTopLevelFlags:
    def test_nawm_normalization(self, base_sidecar, dce_qc_metrics):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=dce_qc_metrics)
        assert out["NAWMNormalizationApplied"] is True


class TestBaseFieldsSurvive:
    def test_base_fields_preserved(self, base_sidecar, dce_qc_metrics):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=dce_qc_metrics)
        assert out["Description"] == base_sidecar["Description"]
        assert out["SkullStripped"] is False
        assert out["SpatialReference"] == base_sidecar["SpatialReference"]
        assert out["Sources"] == base_sidecar["Sources"]
        assert out["KtransConvention"] == "plasma_Tofts1999"


class TestEmptyInputs:
    def test_no_qc_metrics(self, base_sidecar):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=None)
        # No flat keys added; base fields untouched
        assert all(not k.startswith(("T1", "B1", "Motion")) for k in out
                   if k not in {"T1w_in_use",})
        assert out["SpatialReference"] == base_sidecar["SpatialReference"]

    def test_partial_qc_metrics(self, base_sidecar):
        qc = {"t1_mapping": {"method": "hybrid"}}
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=qc)
        assert out["T1MappingMethod"] == "hybrid"
        # Missing fields stay absent (no None / empty values)
        assert "T1B1Correction" not in out
        assert "B1MeanWholeBrain" not in out
        assert "MotionFDMean" not in out
