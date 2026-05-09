"""Resume-writer wiring for DSC motion + detection-time AIF metrics.

Verifies that ``_build_dsc_resume_meta`` folds motion fields into
``preprocessing_stats`` and nests detection metrics under
``aif_qc.aif_metrics``, so the cosmetic-key builder and the HTML
aggregator both see the expected shape.
"""
from __future__ import annotations

from typing import Any

import pytest

from pmriprep.workflows.dsc.datasinks import _build_dsc_resume_meta


@pytest.fixture
def synthetic_inputs() -> dict[str, Any]:
    return {
        "aif_qc": {
            "detection_method": "atlas_relaxed",
            "n_aif_voxels_selected": 6,
            "best_psr": 78.0,
            "best_brr": 92.0,
            "best_ttp": 22.5,
            "best_fwhm": 6.0,
            "best_ph": 41.2,
        },
        "preload_qc": {
            "k2_method": "bsw",
            "k2_median": 0.012,
            "preload_classification": "none",
        },
        "preprocessing_stats": {
            "peak_drop_pct": 34.7,
            "mean_tsnr": 14.9,
            "mean_cnr": 25.3,
            "fraction_cnr_gt_4": 0.916,
            "tr_ms": 1500.0,
            "te_ms": 35.0,
            "n_volumes": 120,
        },
        "motion_qc": {
            "fd_mean": 0.173,
            "fd_max": 1.372,
            "motion_status": "WARN",
            "n_outliers": 2,
            "outlier_frames": [3, 17],
            "motion_thresh": 0.5,
            "fd_list": [0.0, 0.3, 0.18],
            "n_volumes": 120,
            "percent_outliers": 1.7,
        },
        "aif_metrics": {
            "peak_conc": 44.4,
            "fwhm": 6.0,
            "ttp": 22.5,
            "arrival": 14.5,
        },
    }


class TestBuildDscResumeMeta:
    def test_motion_folded_into_preprocessing_stats(self, synthetic_inputs):
        meta = _build_dsc_resume_meta(**synthetic_inputs)
        pp = meta["preprocessing_stats"]
        assert pp["motion_status"] == "WARN"
        assert pp["fd_mean"] == pytest.approx(0.173)
        assert pp["fd_max"] == pytest.approx(1.372)
        assert pp["n_outliers"] == 2
        assert pp["outlier_frames"] == [3, 17]
        assert pp["percent_outliers"] == pytest.approx(1.7)
        # Acquisition stats from preprocessing_stats are preserved
        assert pp["mean_tsnr"] == pytest.approx(14.9)
        assert pp["tr_ms"] == pytest.approx(1500.0)

    def test_aif_metrics_nested_under_aif_qc(self, synthetic_inputs):
        meta = _build_dsc_resume_meta(**synthetic_inputs)
        aq = meta["aif_qc"]
        # Existing flat keys preserved
        assert aq["detection_method"] == "atlas_relaxed"
        assert aq["best_psr"] == pytest.approx(78.0)
        # Detection-time metrics now nested under the legacy aif_metrics key
        assert aq["aif_metrics"]["peak_conc"] == pytest.approx(44.4)
        assert aq["aif_metrics"]["fwhm"] == pytest.approx(6.0)
        assert aq["aif_metrics"]["ttp"] == pytest.approx(22.5)

    def test_minimal_inputs_match_legacy_shape(self):
        """Without motion_qc / aif_metrics, output is shape-equivalent to
        the writer's behavior before motion + AIF detection metrics
        were threaded in.
        """
        meta = _build_dsc_resume_meta(
            aif_qc={"detection_method": "atlas_strict"},
            preload_qc={"k2_method": "bsw"},
            preprocessing_stats={"mean_tsnr": 14.0},
        )
        assert meta["aif_qc"] == {"detection_method": "atlas_strict"}
        assert "aif_metrics" not in meta["aif_qc"]
        assert meta["preprocessing_stats"] == {"mean_tsnr": 14.0}
        assert "motion_status" not in meta["preprocessing_stats"]

    def test_empty_motion_qc_does_not_inject_keys(self):
        meta = _build_dsc_resume_meta(
            aif_qc={"detection_method": "atlas_strict"},
            preprocessing_stats={"mean_tsnr": 14.0},
            motion_qc={},
        )
        assert "motion_status" not in meta["preprocessing_stats"]

    def test_partial_motion_keys_propagate(self):
        meta = _build_dsc_resume_meta(
            aif_qc={"detection_method": "atlas_strict"},
            preprocessing_stats={"mean_tsnr": 14.0},
            motion_qc={"fd_mean": 0.1, "motion_status": "PASS"},
        )
        pp = meta["preprocessing_stats"]
        assert pp["motion_status"] == "PASS"
        assert pp["fd_mean"] == pytest.approx(0.1)
        assert "fd_max" not in pp


class TestResumeWriterWorkflowConstruction:
    """Graph-level wiring guard for the resume_meta Function node."""

    def test_resume_meta_inputs_include_motion_qc_and_aif_metrics(self):
        """``init_dsc_datasinks_wf`` builds the resume_meta Function node
        with both ``motion_qc`` and ``aif_metrics`` in its input_names.
        Guards against silently dropping the rewiring during a refactor.
        """
        from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

        wf = init_dsc_datasinks_wf(
            output_dir="/tmp/pmriprep_test",
            bids_dir="/tmp/pmriprep_test_bids",
            subject_id="X",
            session_id="Y",
            present_inputs={"dsc_moco"},
            has_smoothing=False,
        )
        resume_meta = wf.get_node("resume_meta")
        assert resume_meta is not None, "resume_meta node missing from datasinks_wf"
        input_names = list(resume_meta.inputs.copyable_trait_names())
        assert "motion_qc" in input_names
        assert "aif_metrics" in input_names
        assert "aif_qc" in input_names
        assert "preprocessing_stats" in input_names

    def test_inputnode_exposes_motion_qc_and_aif_metrics(self):
        """The datasinks_wf inputnode advertises the new fields so the
        parent workflow can connect dsc_wf outputs to them.
        """
        from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

        wf = init_dsc_datasinks_wf(
            output_dir="/tmp/pmriprep_test",
            bids_dir="/tmp/pmriprep_test_bids",
            subject_id="X",
            session_id="Y",
            present_inputs={"dsc_moco"},
            has_smoothing=False,
        )
        inputnode = wf.get_node("inputnode")
        assert inputnode is not None
        fields = list(inputnode.inputs.copyable_trait_names())
        assert "motion_qc" in fields
        assert "aif_metrics" in fields
