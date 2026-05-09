"""Phase 2b/3a: resume sidecar round-trip + write path.

Asserts:
  * ``build_dsc_resume_sidecar`` emits verbatim snake_case top-level dicts
  * ``_write_resume_sidecar`` lands the file at
    ``perf/sub-X[_ses-Y]_desc-dscResume_pmriprep.json``
  * ``_find_previous_dsc_outputs`` reads it back and recovers the
    in-memory dicts byte-for-byte (no key translation)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pmriprep.utils.sidecar_builders import build_dsc_resume_sidecar
from pmriprep.workflows.base import _find_previous_dsc_outputs
from pmriprep.workflows.dsc.datasinks import _write_resume_sidecar


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


class TestBuildResumeSidecar:
    def test_verbatim_snake_case_dicts(self, representative_qc_dicts):
        aif_qc, preload_qc, preprocessing_stats = representative_qc_dicts
        sidecar = build_dsc_resume_sidecar(
            aif_qc=aif_qc,
            preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        assert sidecar["aif_qc"] == aif_qc
        assert sidecar["preload_qc"] == preload_qc
        assert sidecar["preprocessing_stats"] == preprocessing_stats
        assert sidecar["PMRIPrepInternal"] is True
        assert "Description" in sidecar

    def test_omits_missing_dicts(self):
        sidecar = build_dsc_resume_sidecar(aif_qc=None, preload_qc=None)
        assert "aif_qc" not in sidecar
        assert "preload_qc" not in sidecar
        assert "preprocessing_stats" not in sidecar
        # Provenance fields still present
        assert sidecar["PMRIPrepInternal"] is True


class TestSidecarFileWrite:
    def test_landing_path(self, tmp_path, representative_qc_dicts):
        aif_qc, preload_qc, preprocessing_stats = representative_qc_dicts
        meta = build_dsc_resume_sidecar(
            aif_qc=aif_qc, preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        out = _write_resume_sidecar(
            output_dir=str(tmp_path), subject="X", session="Y", meta_dict=meta,
        )
        assert Path(out).exists()
        # perf/ landing per user direction (not session root)
        assert Path(out).parent.name == "perf"
        assert Path(out).name == "sub-X_ses-Y_desc-dscResume_pmriprep.json"

    def test_no_session(self, tmp_path, representative_qc_dicts):
        aif_qc, preload_qc, preprocessing_stats = representative_qc_dicts
        meta = build_dsc_resume_sidecar(
            aif_qc=aif_qc, preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        out = _write_resume_sidecar(
            output_dir=str(tmp_path), subject="X", session="", meta_dict=meta,
        )
        assert Path(out).name == "sub-X_desc-dscResume_pmriprep.json"
        assert "ses-" not in Path(out).parent.parent.name


class TestRoundTripViaResumeHelper:
    def test_resume_recovers_verbatim_dicts(self, tmp_path, representative_qc_dicts):
        aif_qc, preload_qc, preprocessing_stats = representative_qc_dicts
        meta = build_dsc_resume_sidecar(
            aif_qc=aif_qc, preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
        )
        _write_resume_sidecar(
            output_dir=str(tmp_path), subject="X", session="Y", meta_dict=meta,
        )

        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
        )
        assert recovered["aif_qc"] == aif_qc
        assert recovered["preload_qc"] == preload_qc
        assert recovered["preprocessing_stats"] == preprocessing_stats

    def test_no_resume_sidecar_returns_none(self, tmp_path):
        # Empty derivatives tree → resume helper returns None for all QC dicts
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
        )
        assert recovered["aif_qc"] is None
        assert recovered["preload_qc"] is None
        assert recovered["preprocessing_stats"] is None

    def test_malformed_json_fails_closed(self, tmp_path):
        # Create a minimal derivatives tree with bad JSON resume sidecar.
        # Phase 3 fail-closed test (code-reviewer round 2 requirement).
        perf = tmp_path / "pmriprep" / "sub-X" / "ses-Y" / "perf"
        perf.mkdir(parents=True)
        (perf / "sub-X_ses-Y_desc-dscResume_pmriprep.json").write_text(
            "this is not json {{{",
        )
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
        )
        # Should NOT crash; recovers no QC dicts (clean SKIP downstream)
        assert recovered["aif_qc"] is None
        assert recovered["preload_qc"] is None
        assert recovered["preprocessing_stats"] is None

    def test_partial_resume_sidecar(self, tmp_path):
        # Only aif_qc present → others recover as None
        perf = tmp_path / "pmriprep" / "sub-X" / "ses-Y" / "perf"
        perf.mkdir(parents=True)
        (perf / "sub-X_ses-Y_desc-dscResume_pmriprep.json").write_text(
            json.dumps({
                "PMRIPrepInternal": True,
                "aif_qc": {"detection_method": "synthetic"},
            }),
        )
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
        )
        assert recovered["aif_qc"] == {"detection_method": "synthetic"}
        assert recovered["preload_qc"] is None
        assert recovered["preprocessing_stats"] is None
