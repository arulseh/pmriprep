"""DSC resume checkpoint round-trip via work/ directory.

Asserts:
  * ``_write_dsc_resume_checkpoint`` lands the file at
    ``work/pmriprep_sub_<sub>_ses_<ses>_wf/dsc_resume.json``
  * ``_find_previous_dsc_outputs`` reads it back and recovers the
    in-memory dicts (no key translation)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pmriprep.workflows.base import _find_previous_dsc_outputs
from pmriprep.workflows.dsc.datasinks import _write_dsc_resume_checkpoint


@pytest.fixture
def representative_qc_dicts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    aif_qc = {
        "detection_method": "atlas_strict",
        "n_aif_voxels_selected": 6,
        "cbv_auc_denominator_is_synthetic": False,
        "cnr_whole_brain": {"fraction_cnr_gt_4": 0.81, "mean_cnr": 5.4},
        "aif_metrics": {
            "peak_conc": 12.5,
            "fwhm": 6.2,
            "psr": 78.0,
            "brr": 92.0,
        },
    }
    preload_qc = {
        "k2_method": "bsw",
        "preload_classification": "none",
        "k2_median": 0.012,
        "percent_leaky_voxels": 4.7,
    }
    preprocessing_stats = {
        "fd_mean": 0.18, "fd_max": 0.62,
        "mean_cnr": 5.4, "fraction_cnr_gt_4": 0.81,
        "tr_ms": 1500.0, "n_volumes": 60,
    }
    return aif_qc, preload_qc, preprocessing_stats


class TestCheckpointFileWrite:
    def test_landing_path(self, tmp_path: Path, representative_qc_dicts: Any) -> None:
        aif_qc, preload_qc, preprocessing_stats = representative_qc_dicts
        out = _write_dsc_resume_checkpoint(
            work_dir=str(tmp_path),
            subject_id="X", session_id="Y",
            aif_qc=aif_qc, preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        assert out is not None
        assert Path(out).exists()
        assert Path(out).name == "dsc_resume.json"
        assert "pmriprep_sub_X_ses_Y_wf" in str(out)

    def test_landing_path_no_session(
        self, tmp_path: Path,
    ) -> None:
        out = _write_dsc_resume_checkpoint(
            work_dir=str(tmp_path),
            subject_id="X", session_id="",
        )
        assert out is not None
        assert "pmriprep_sub_X_wf" in str(out)
        assert "ses_" not in str(out)

    def test_undefined_work_dir_returns_empty(self) -> None:
        from nipype.interfaces.base import Undefined
        out = _write_dsc_resume_checkpoint(
            work_dir=Undefined,
            subject_id="X", session_id="Y",
        )
        assert out == ""

    def test_motion_merge(self, tmp_path: Path) -> None:
        out = _write_dsc_resume_checkpoint(
            work_dir=str(tmp_path),
            subject_id="01", session_id="01",
            preprocessing_stats={"tr_ms": 1500},
            motion_qc={"fd_mean": 0.2, "fd_max": 0.5},
        )
        data = json.loads(Path(out).read_text())
        assert data["preprocessing_stats"]["fd_mean"] == 0.2
        assert data["preprocessing_stats"]["tr_ms"] == 1500


class TestRoundTripViaResumeHelper:
    def test_resume_recovers_verbatim_dicts(
        self, tmp_path: Path, representative_qc_dicts: Any,
    ) -> None:
        aif_qc, preload_qc, preprocessing_stats = representative_qc_dicts
        _write_dsc_resume_checkpoint(
            work_dir=str(tmp_path),
            subject_id="X", session_id="Y",
            aif_qc=aif_qc, preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
            work_dir=tmp_path,
        )
        assert recovered["aif_qc"] == aif_qc
        assert recovered["preload_qc"] == preload_qc
        assert recovered["preprocessing_stats"] == preprocessing_stats

    def test_no_work_dir_returns_none(self, tmp_path: Path) -> None:
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
        )
        assert recovered["aif_qc"] is None
        assert recovered["preload_qc"] is None
        assert recovered["preprocessing_stats"] is None

    def test_malformed_json_fails_closed(self, tmp_path: Path) -> None:
        wf_dir = tmp_path / "pmriprep_sub_X_ses_Y_wf"
        wf_dir.mkdir(parents=True)
        (wf_dir / "dsc_resume.json").write_text("this is not json {{{")
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
            work_dir=tmp_path,
        )
        assert recovered["aif_qc"] is None
        assert recovered["preload_qc"] is None
        assert recovered["preprocessing_stats"] is None

    def test_partial_checkpoint(self, tmp_path: Path) -> None:
        wf_dir = tmp_path / "pmriprep_sub_X_ses_Y_wf"
        wf_dir.mkdir(parents=True)
        (wf_dir / "dsc_resume.json").write_text(
            json.dumps({"aif_qc": {"detection_method": "synthetic"}}),
        )
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
            work_dir=tmp_path,
        )
        assert recovered["aif_qc"] == {"detection_method": "synthetic"}
        assert recovered["preload_qc"] is None
        assert recovered["preprocessing_stats"] is None
