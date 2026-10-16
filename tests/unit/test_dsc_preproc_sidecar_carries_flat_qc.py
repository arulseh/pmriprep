"""``desc-dscPreproc_dsc.json`` carries flat-prefixed cosmetic QC.

Asserts that ``enrich_dsc_preproc_sidecar`` injects ``Preload*`` and
unprefixed motion/signal keys onto a base sidecar dict, and that base
fields (``SkullStripped``, ``SpatialReference``, ``Sources``, and
``VoxelSize``) survive enrichment unchanged.
"""
from __future__ import annotations

from typing import Any

import pytest

from pmriprep.utils.sidecar_builders import enrich_dsc_preproc_sidecar


@pytest.fixture
def base_sidecar() -> dict[str, Any]:
    """Representative minimal-base sidecar shape from ``_build_base``."""
    return {
        "Description": "Motion-corrected 4D DSC time series",
        "SkullStripped": False,
        "SpatialReference": "bids::sub-X/perf/sub-X_desc-dscMean_stat.nii.gz",
        "Sources": ["bids:raw:sub-X/perf/sub-X_dsc.nii.gz"],
        "VoxelSize": [3.0, 3.0, 5.0],
    }


@pytest.fixture
def preload_qc() -> dict[str, Any]:
    return {
        "k2_method": "bsw",
        "preload_source": "unknown",
        "preload_administered_dose_fraction": None,
        "k2_median": -0.002,
        "k2_signed": True,
        "k2_reference_selection": (
            "raw_signal_final10_mean_1_baseline_sample_sd"
        ),
        "k2_reference_selection_domain": "raw_signal",
        "percent_negative_k2": 61.0,
        "percent_k2_undefined": 0.4,
        "n_k2_valid": 1000,
    }


@pytest.fixture
def preprocessing_stats() -> dict[str, Any]:
    return {
        "fd_mean": 0.18,
        "fd_max": 0.62,
        "percent_outliers": 1.5,
        "motion_status": "PASS",
        "mean_cnr": 5.4,
        "fraction_cnr_gt_4": 0.81,
        "mean_tsnr": 38.2,
        "peak_drop_pct": 24.0,
        "n_volumes": 60,
        "tr_ms": 1500.0,
    }


class TestPreloadCosmeticKeys:
    def test_preload_keys_added(self, base_sidecar, preload_qc, preprocessing_stats):
        out = enrich_dsc_preproc_sidecar(
            dict(base_sidecar),
            preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        assert out["PreloadK2Method"] == "bsw"
        assert out["PreloadSource"] == "unknown"
        assert "PreloadAdministeredDoseFraction" not in out
        assert out["PreloadK2Median"] == pytest.approx(-0.002)
        assert out["PreloadK2Signed"] is True
        assert out["PreloadK2ReferenceSelectionDomain"] == "raw_signal"
        assert out["PreloadPercentNegativeK2"] == pytest.approx(61.0)
        assert out["PreloadPercentK2Undefined"] == pytest.approx(0.4)
        assert out["PreloadK2ValidVoxels"] == 1000

class TestPreprocessingCosmeticKeys:
    def test_motion_and_signal_keys_added(
        self, base_sidecar, preload_qc, preprocessing_stats
    ):
        out = enrich_dsc_preproc_sidecar(
            dict(base_sidecar),
            preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        # Motion fields have no modality prefix
        assert out["MotionFDMean"] == pytest.approx(0.18)
        assert out["MotionFDMax"] == pytest.approx(0.62)
        assert out["MotionPercentOutliers"] == pytest.approx(1.5)
        assert out["MotionStatus"] == "PASS"
        # Signal
        assert out["CNRWholeBrain"] == pytest.approx(5.4)
        assert out["FractionCNRGreaterThan4"] == pytest.approx(0.81)
        assert out["MeanTSNR"] == pytest.approx(38.2)
        assert out["PeakSignalDropPercent"] == pytest.approx(24.0)
        assert out["NumberOfVolumes"] == 60

    def test_repetition_time_in_seconds(self, base_sidecar, preprocessing_stats):
        out = enrich_dsc_preproc_sidecar(
            dict(base_sidecar),
            preload_qc=None,
            preprocessing_stats=preprocessing_stats,
        )
        # Producer carries tr_ms; sidecar must convert to canonical BIDS
        # RepetitionTime in seconds.
        assert out["RepetitionTime"] == pytest.approx(1.5)
        assert "PreprocessingTR_ms" not in out


class TestBaseSidecarFieldsSurvive:
    """Risk #2: enrichment must preserve base-sidecar fields verbatim."""

    def test_base_fields_preserved(
        self, base_sidecar, preload_qc, preprocessing_stats
    ):
        out = enrich_dsc_preproc_sidecar(
            dict(base_sidecar),
            preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        assert out["Description"] == base_sidecar["Description"]
        assert out["SkullStripped"] is False
        assert out["SpatialReference"] == base_sidecar["SpatialReference"]
        assert out["Sources"] == base_sidecar["Sources"]
        assert out["VoxelSize"] == [3.0, 3.0, 5.0]


class TestEmptyInputs:
    def test_no_qc_dicts(self, base_sidecar):
        out = enrich_dsc_preproc_sidecar(
            dict(base_sidecar), preload_qc=None, preprocessing_stats=None,
        )
        # Sidecar still emits valid (smaller) form; no Preload* / Motion* keys
        assert all(not k.startswith("Preload") for k in out)
        assert all(not k.startswith("Motion") for k in out)
        # Base fields untouched
        assert out["SpatialReference"] == base_sidecar["SpatialReference"]

    def test_empty_dicts(self, base_sidecar):
        out = enrich_dsc_preproc_sidecar(
            dict(base_sidecar), preload_qc={}, preprocessing_stats={},
        )
        assert all(not k.startswith("Preload") for k in out)
        assert all(not k.startswith("Motion") for k in out)
