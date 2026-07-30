"""DCE preproc-4D sidecar carries flat-key cosmetic QC.

Mirrors ``test_dsc_preproc_sidecar_carries_flat_qc.py``.
"""
from __future__ import annotations

from typing import Any

import pytest

from pmriprep.utils.sidecar_builders import (
    enrich_dce_preproc_sidecar,
    enrich_recommendation_keys,
)


@pytest.fixture
def base_sidecar() -> dict[str, Any]:
    return {
        "Description": "Motion-corrected 4D DCE time series",
        "SkullStripped": False,
        "SpatialReference": "bids::sub-X/perf/sub-X_desc-dceMean_stat.nii.gz",
        "Sources": ["bids:raw:sub-X/perf/sub-X_dce.nii.gz"],
        "Resolution": "native",
        "VoxelSize": [1.0, 1.0, 3.0],
        "InputFunctionCompartment": "plasma",
    }


@pytest.fixture
def dce_qc_metrics() -> dict[str, Any]:
    """Representative DCE merged qc_metrics (from workflows/dce/base.py:3286)."""
    return {
        "t1_mapping": {
            "method": "nlls",
            "b1_correction": "afi",
            "t1_correction": {
                "correction_applied": "fallback",
                "fallback_value_ms": 1200.0,
                "reason": "Measured T1 rejected",
            },
        },
        "b1_mapping": {"mean_b1": 0.96, "pct_correction_gt_20": 8.5},
        "preprocessing": {
            "fd_mean": 0.21, "fd_max": 0.78, "percent_outliers": 2.1,
            "motion_status": "PASS",
            "motion_qc_available": True,
            "baseline_tsnr": 32.4, "peak_enh_pct": 65.0,
            "n_volumes": 75, "tr_ms": 3.8, "volume_tr_s": 4.06,
            "baseline_frames_used": 5,
            "baseline_minimum_frames": 5,
            "detected_bolus_frame": 2,
            "baseline_fallback_applied": True,
            "baseline_fallback_reason": "detected_bolus_before_minimum_baseline",
            "baseline_status": "POOR",
        },
    }


class TestT1Keys:
    def test_t1_mapping_keys(self, base_sidecar, dce_qc_metrics):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=dce_qc_metrics)
        assert out["T1MappingMethod"] == "nlls"
        assert out["T1B1Correction"] == "afi"
        assert out["T1CorrectionApplied"] == "fallback"
        assert out["T1Source"] == "uniform_field_strength_fallback"
        assert out["T1MappingAttempted"] is True
        assert out["T1MappingAccepted"] is False
        assert out["T1AssumedValue"] == pytest.approx(1200.0)
        assert out["T1FallbackReason"] == "Measured T1 rejected"

    def test_assumed_t1_keys(self, base_sidecar):
        qc = {
            "t1_mapping": {
                "method": "assumed",
                "b1_correction": "precomputed",
                "assumed_t1_ms": 1200.0,
                "assumed_t1_field_strength_t": 3.0,
                "assumed_t1_policy": "uniform_field_strength_reference",
            }
        }
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=qc)
        assert out["T1AssumedValue"] == pytest.approx(1200.0)
        assert out["T1AssumedValueUnits"] == "ms"
        assert out["T1AssumptionFieldStrength"] == pytest.approx(3.0)
        assert out["T1AssumptionPolicy"] == "uniform_field_strength_reference"
        assert out["T1B1Correction"] == "precomputed"
        assert out["T1Source"] == "uniform_field_strength_assumption"
        assert out["T1MappingAttempted"] is False


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
        assert out["MotionQCAvailable"] is True
        assert out["BaselineTSNR"] == pytest.approx(32.4)
        assert out["PeakEnhancementPercent"] == pytest.approx(65.0)
        assert out["NumberOfVolumes"] == 75
        assert out["BaselineFramesUsed"] == 5
        assert out["BaselineMinimumFrames"] == 5
        assert out["BolusArrivalFrameDetected"] == 2
        assert out["BaselineFallbackApplied"] is True
        assert out["BaselineFallbackReason"] == (
            "detected_bolus_before_minimum_baseline"
        )
        assert out["BaselineQCStatus"] == "POOR"
        # Volume and excitation TR retain their distinct BIDS meanings.
        assert out["RepetitionTime"] == pytest.approx(4.06)
        assert out["RepetitionTimeExcitation"] == pytest.approx(0.0038)

    def test_baseline_tsnr_nan_dropped(self, base_sidecar, dce_qc_metrics):
        """NaN baseline tSNR is dropped from the flat sidecar keys."""
        dce_qc_metrics["preprocessing"]["baseline_tsnr"] = float("nan")
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=dce_qc_metrics)
        assert "BaselineTSNR" not in out


def test_baseline_and_aif_processing_warnings_coexist() -> None:
    """DCE baseline fallback uses the existing processing-warning channel."""
    sidecar: dict[str, Any] = {}
    qc = {
        "preprocessing": {
            "baseline_warning": "Baseline fallback may bias PK maps.",
            "baseline_fallback_applied": True,
            "detected_bolus_frame": 2,
            "baseline_minimum_frames": 5,
        },
        "aif_detection": {"aif_source": "population"},
    }

    enrich_recommendation_keys(sidecar, modality="DCE", qc_metrics=qc)

    warnings = sidecar["QCProcessingWarnings"]
    assert sidecar["QCRecommendationAction"] == "exclude"
    assert "Inadequate pre-contrast DCE baseline" in (
        sidecar["QCRecommendationReason"]
    )
    assert len(warnings) == 2
    assert "Parker population model" in warnings[0]
    assert warnings[1] == "Baseline fallback may bias PK maps."


def test_motion_qc_warning_uses_processing_warning_channel() -> None:
    """A motion-QC reduction failure reaches the existing HTML banner."""
    sidecar: dict[str, Any] = {}
    qc = {
        "preprocessing": {
            "motion_qc_warning": "Motion quality is unknown.",
        },
    }

    enrich_recommendation_keys(sidecar, modality="DCE", qc_metrics=qc)

    assert sidecar["QCProcessingWarnings"] == ["Motion quality is unknown."]


def test_vfa_alignment_warning_uses_processing_warning_channel() -> None:
    """An inter-VFA alignment warning reaches the report Review section."""
    sidecar: dict[str, Any] = {}
    warning = "Potential inter-VFA misalignment detected."
    qc = {
        "t1_mapping": {
            "vfa_preprocessing": {
                "motion_assessment": {
                    "qc_status": "WARN",
                    "warning": warning,
                },
            },
        },
    }

    enrich_recommendation_keys(sidecar, modality="DCE", qc_metrics=qc)

    assert sidecar["QCProcessingWarnings"] == [warning]


class TestBaseFieldsSurvive:
    def test_base_fields_preserved(self, base_sidecar, dce_qc_metrics):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=dce_qc_metrics)
        assert out["Description"] == base_sidecar["Description"]
        assert out["SkullStripped"] is False
        assert out["SpatialReference"] == base_sidecar["SpatialReference"]
        assert out["Sources"] == base_sidecar["Sources"]
        assert out["InputFunctionCompartment"] == "plasma"


class TestEmptyInputs:
    def test_no_qc_metrics(self, base_sidecar):
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=None)
        # No flat keys added; base fields untouched
        assert all(not k.startswith(("T1", "B1", "Motion")) for k in out
                   if k not in {"T1w_in_use",})
        assert out["SpatialReference"] == base_sidecar["SpatialReference"]

    def test_partial_qc_metrics(self, base_sidecar):
        qc = {"t1_mapping": {"method": "nlls"}}
        out = enrich_dce_preproc_sidecar(dict(base_sidecar), qc_metrics=qc)
        assert out["T1MappingMethod"] == "nlls"
        # Missing fields stay absent (no None / empty values)
        assert "T1B1Correction" not in out
        assert "B1MeanWholeBrain" not in out
        assert "MotionFDMean" not in out


class TestSanitizeForJson:
    """The recursive sanitizer that keeps sidecars valid, number-typed JSON."""

    def test_drops_nan_and_inf_dict_entries(self):
        from pmriprep.utils.sidecar_builders import sanitize_for_json

        out = sanitize_for_json({"a": 1.0, "b": float("nan"), "c": float("inf"), "d": "x"})
        assert out == {"a": 1.0, "d": "x"}

    def test_recurses_into_nested_dicts(self):
        from pmriprep.utils.sidecar_builders import sanitize_for_json

        out = sanitize_for_json({"outer": {"ok": 2, "bad": float("nan")}})
        assert out == {"outer": {"ok": 2}}

    def test_nulls_nonfinite_list_elements(self):
        from pmriprep.utils.sidecar_builders import sanitize_for_json

        out = sanitize_for_json({"vals": [1.0, float("nan"), 3.0]})
        assert out == {"vals": [1.0, None, 3.0]}

    def test_keeps_finite_and_non_float(self):
        from pmriprep.utils.sidecar_builders import sanitize_for_json

        payload = {"i": 5, "f": 1.5, "s": "y", "b": True, "n": None}
        assert sanitize_for_json(payload) == payload

    def test_numpy_scalars_dropped_or_coerced(self):
        """numpy NaN/Inf dropped; finite numpy scalars coerced to native (not strings)."""
        import json

        import numpy as np

        from pmriprep.utils.sidecar_builders import sanitize_for_json

        out = sanitize_for_json(
            {"a": np.float32("nan"), "b": np.float64("inf"),
             "c": np.float32(2.5), "n": np.int32(7)}
        )
        assert "a" not in out and "b" not in out
        # Finite numpy scalars coerced to python-native so they serialize as
        # numbers, not quoted strings under a default=str encoder.
        assert not isinstance(out["c"], np.generic)
        assert not isinstance(out["n"], np.generic)
        assert out["c"] == pytest.approx(2.5)
        assert out["n"] == 7
        assert json.loads(json.dumps(out)) == {"c": pytest.approx(2.5), "n": 7}

    def test_handles_tuples_like_lists(self):
        """A tuple with a non-finite element must not re-emit a bare NaN token."""
        from pmriprep.utils.sidecar_builders import sanitize_for_json

        out = sanitize_for_json({"shape": (4, float("nan"), 3)})
        assert out == {"shape": [4, None, 3]}

    def test_coerces_numpy_arrays(self):
        """np.ndarray is coerced to a list, with non-finite elements nulled."""
        import numpy as np

        from pmriprep.utils.sidecar_builders import sanitize_for_json

        out = sanitize_for_json({"arr": np.array([1.0, np.nan, 2.0], dtype=np.float32)})
        assert out == {"arr": [pytest.approx(1.0), None, pytest.approx(2.0)]}

    def test_dump_sidecar_json_strict_valid_and_numeric(self):
        """dump_sidecar_json emits strict JSON: no bare NaN/Infinity, numpy as numbers."""
        import numpy as np

        from pmriprep.utils.sidecar_builders import dump_sidecar_json

        s = dump_sidecar_json(
            {"nan": float("nan"), "f32": np.float32(1.5),
             "curve": [1.0, float("inf"), 2.0]}
        )
        assert "NaN" not in s and "Infinity" not in s
        # numpy scalar rendered as a bare number, not a quoted string
        assert "1.5" in s and '"1.5"' not in s


class TestPreprocMetaNoNaN:
    """End-to-end: the assembled preproc sidecar never carries a bare NaN token."""

    def test_dce_nan_absent_flat_and_nested_and_valid_json(self):
        import json

        import numpy as np

        from pmriprep.workflows.dce.datasinks import _build_dce_preproc_meta

        base = {"Description": "x", "Sources": ["bids:raw:sub-X/perf/sub-X_dce.nii.gz"]}
        qc = {
            "preprocessing": {
                "baseline_tsnr": float("nan"),
                "numpy_nan": np.float32("nan"),  # numpy scalar path
                "mean_tsnr": 20.0,
                "shape": (4, float("nan"), 3),  # tuple path
            },
        }
        meta = _build_dce_preproc_meta(base, "desc", qc_metrics=qc)
        assert meta["SkullStripped"] is False
        # Flat cosmetic keys dropped
        assert "BaselineTSNR" not in meta
        # Nested PipelineQC block sanitized (finite metric survives)
        nested = meta["PipelineQC"]["preprocessing"]
        assert "baseline_tsnr" not in nested
        assert "numpy_nan" not in nested
        assert nested["mean_tsnr"] == 20.0
        assert nested["shape"] == [4, None, 3]
        # Strict: allow_nan=False raises on ANY NaN/Inf token (bare or numpy-leaked)
        json.dumps(meta, allow_nan=False)

    def test_dce_no_motion_description_is_not_motion_corrected(self):
        from pmriprep.workflows.dce.datasinks import _build_dce_preproc_meta

        meta = _build_dce_preproc_meta(
            {},
            "Motion-corrected 4D DCE time series",
            config_dict={"motion_correct": False},
        )

        assert meta["MotionCorrectionApplied"] is False
        assert "without rigid-body motion correction" in meta["Description"]
        assert meta["SkullStripped"] is False

    def test_dsc_preproc_meta_sanitized(self):
        """The DSC sibling builder is sanitized too (symmetry)."""
        import json

        from pmriprep.workflows.dsc.datasinks import _build_dsc_preproc_meta

        qc = {"preprocessing": {"baseline_tsnr": float("nan"), "mean_cnr": 4.2}}
        meta = _build_dsc_preproc_meta({"Description": "x"}, "desc", qc_metrics=qc)
        assert meta["SkullStripped"] is True
        assert "baseline_tsnr" not in meta["PipelineQC"]["preprocessing"]
        assert meta["PipelineQC"]["preprocessing"]["mean_cnr"] == 4.2
        json.dumps(meta, allow_nan=False)
