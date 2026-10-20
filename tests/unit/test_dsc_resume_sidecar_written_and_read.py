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
from types import SimpleNamespace
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from pmriprep.config import DCEConfig, DSCConfig
from pmriprep.interfaces.dce_compute import ComputeDCEMaps
from pmriprep.utils.dsc_cbv_constraint import ConstraintMode
from pmriprep.workflows.base import _connect_master_dsc_to_dce, _find_previous_dsc_outputs
from pmriprep.workflows.dsc.datasinks import _write_dsc_resume_checkpoint


@pytest.fixture
def representative_qc_dicts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    aif_qc = {
        "detection_method": "atlas_strict",
        "n_aif_voxels_selected": 6,
        "gamma_fit_diagnostics": {"peak_ratio": 1.05, "shape_score": 0.9},
        "hematocrit": 0.42,
        "quality_score": 0.81,
        "cnr_whole_brain": {"fraction_cnr_gt_4": 0.81, "mean_cnr": 5.4},
        "aif_metrics": {
            "ph": 41.2,
            "peak_conc": 12.5,
            "fwhm": 6.2,
            "ttp": 18.0,
            "psr": 78.0,
            "brr": 92.0,
        },
    }
    preload_qc = {
        "k2_method": "bsw",
        "preload_source": "unknown",
        "preload_administered_dose_fraction": None,
        "k2_median": -0.002,
        "k2_signed": True,
        "percent_negative_k2": 61.0,
        "k2_mean": 0.011, "k2_std": 0.005,
        "k2_min": -0.021, "k2_max": 0.087,
        "k2_p05": -0.004, "k2_p95": 0.033,
        "n_brain_voxels": 142_000,
        "n_k2_valid": 141_000,
        "auc_mean": 1.23,
        "percent_k2_undefined": 1000 / 1420,
        "k2_reference_available": True,
        "k2_reference_total_voxels": 1000,
        "k2_reference_selected_voxels": 600,
        "k2_reference_selected_fraction": 0.6,
        "k2_reference_baseline_frames": 20,
        "k2_reference_tail_frames": 10,
        "k2_reference_selection": "raw_signal_final10_mean_1_baseline_sample_sd",
        "k2_reference_selection_domain": "raw_signal",
        "k2_reference_peak": 4.0,
        "k2_reference_failure_reason": "",
        "leakage_correction_disabled": False,
    }
    preprocessing_stats = {
        "fd_mean": 0.18, "fd_max": 0.62,
        "mean_cnr": 5.4, "fraction_cnr_gt_4": 0.81,
        "tr_ms": 1500.0, "n_volumes": 60,
        "percent_outliers": 1.5,
        "motion_status": "PASS",
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
    def test_recovers_nested_inverse_transform(self, tmp_path: Path) -> None:
        """The skip-DSC loader follows the nested registration node directory."""
        xfm_dir = (
            tmp_path / "pmriprep_sub_X_ses_Y_wf" / "dsc_wf"
            / "dsc_registration_wf" / "invert_dsc_xfm"
        )
        xfm_dir.mkdir(parents=True)
        xfm = xfm_dir / "t1w_to_dsc_inv.mat"
        xfm.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
            work_dir=tmp_path,
        )

        assert recovered["dsc_to_t1w_xfm"] == str(xfm)

    @pytest.mark.parametrize(
        ("registration_status", "preload_source", "dose", "configured"),
        [
            ("PASS", "user_specified", 0.75, True),
            ("POOR", "user_specified", 0.0, True),
            (None, "unknown", None, False),
            ("PASS", None, None, None),
        ],
    )
    def test_resume_recovers_verbatim_dicts(
        self, tmp_path: Path, representative_qc_dicts: Any,
        registration_status: str | None,
        preload_source: str | None, dose: float | None, configured: bool | None,
    ) -> None:
        """An nCBV-only cache supplies its actual QC to the existing prior gate."""
        aif_qc, preload_qc, preprocessing_stats = representative_qc_dicts
        if preload_source is None:
            preload_qc.pop("preload_source")
            preload_qc.pop("preload_administered_dose_fraction")
        else:
            preload_qc.update(
                preload_source=preload_source, preload_administered_dose_fraction=dose,
            )
        # Qualifying tissue measurements isolate the registration decision.
        preprocessing_stats = dict(
            preprocessing_stats, fraction_cnr_gt_4=0.99, mean_cnr=10.0,
        )
        registration_qc = (
            {"qc_status": registration_status, "dice": 0.91}
            if registration_status is not None else None
        )
        _write_dsc_resume_checkpoint(
            work_dir=str(tmp_path),
            subject_id="X", session_id="Y",
            aif_qc=aif_qc, preload_qc=preload_qc,
            preprocessing_stats=preprocessing_stats,
            registration_qc=registration_qc,
        )
        perf_dir = tmp_path / "pmriprep" / "sub-X" / "ses-Y" / "perf"
        perf_dir.mkdir(parents=True)
        ncbv = perf_dir / "sub-X_ses-Y_desc-dscNawmNormCbv_stat.nii.gz"
        mask = tmp_path / "mask.nii.gz"
        for path in (ncbv, mask):
            nib.save(nib.Nifti1Image(np.ones((6, 6, 6)), np.eye(4)), path)
        xfm_dir = (
            tmp_path / "pmriprep_sub_X_ses_Y_wf" / "dsc_wf"
            / "dsc_registration_wf" / "invert_dsc_xfm"
        )
        xfm_dir.mkdir(parents=True)
        xfm = xfm_dir / "inverse.mat"
        np.savetxt(xfm, np.eye(4))
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
            work_dir=tmp_path,
        )
        assert recovered["aif_qc"] == aif_qc
        assert recovered["preload_qc"] == preload_qc
        assert recovered["preprocessing_stats"] == preprocessing_stats
        assert recovered["registration_qc"] == registration_qc
        assert recovered["rel_cbv_map"] is None

        inputs = SimpleNamespace(
            dsc_cbv_file=None, dsc_ncbv_file=None, dsc_registration_qc={},
        )
        pk = SimpleNamespace()
        validation = SimpleNamespace(dsc_preload_configured=None)
        dce = SimpleNamespace(inputs=SimpleNamespace(
            inputnode=inputs,
            dce_pk_modeling_wf=SimpleNamespace(pk_modeling=pk),
            dce_dsc_validation_wf=SimpleNamespace(inputnode=validation),
        ))
        _connect_master_dsc_to_dce(
            workflow=SimpleNamespace(), dsc_wf=None, dce_wf=dce,
            dsc_config=DSCConfig(preload="unknown" if configured else 0.5),
            dce_config=DCEConfig(model="patlak", inference="vi"),
            output_dir=tmp_path, work_dir=tmp_path, subject_id="X", session_id="Y",
        )
        assert inputs.dsc_ncbv_file == str(ncbv)
        assert inputs.dsc_cbv_file is None
        assert inputs.dsc_to_t1w_xfm == str(xfm)
        assert inputs.dsc_registration_qc == (registration_qc or {})
        assert pk.dsc_aif_qc == aif_qc
        assert pk.dsc_preload_qc == preload_qc
        assert pk.dsc_preprocessing_stats == preprocessing_stats
        assert validation.dsc_preload_configured is configured
        interface = ComputeDCEMaps(
            dsc_ncbv_file=inputs.dsc_ncbv_file, brain_mask_file=str(mask),
            dsc_aif_qc=pk.dsc_aif_qc,
            dsc_preprocessing_stats=pk.dsc_preprocessing_stats,
            dsc_preload_qc=pk.dsc_preload_qc,
            dsc_registration_qc=inputs.dsc_registration_qc,
            dce_registration_qc={"qc_status": "PASS"},
        )
        prior = interface._prepare_vp_constraint(SimpleNamespace(
            model_name="patlak", is_auto=False, candidate_names=[],
            inference="vi", mask=np.ones((6, 6, 6), dtype=bool),
        ))
        if registration_status == "PASS":
            assert prior.mode == ConstraintMode.SOFT
            assert prior.vp_map is not None
            np.testing.assert_allclose(prior.vp_map, 0.01)
        else:
            assert prior.mode == ConstraintMode.SKIP
            assert prior.qc["skip_reason"] == "cross_modal_registration_qc"

    def test_no_work_dir_returns_none(self, tmp_path: Path) -> None:
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
        )
        assert recovered["aif_qc"] is None
        assert recovered["preload_qc"] is None
        assert recovered["preprocessing_stats"] is None
        assert recovered["registration_qc"] is None

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
        assert recovered["registration_qc"] is None

    def test_partial_checkpoint(self, tmp_path: Path) -> None:
        wf_dir = tmp_path / "pmriprep_sub_X_ses_Y_wf"
        wf_dir.mkdir(parents=True)
        (wf_dir / "dsc_resume.json").write_text(
            json.dumps({"aif_qc": {"detection_method": "unavailable"}}),
        )
        recovered = _find_previous_dsc_outputs(
            output_dir=tmp_path, subject_id="X", session_id="Y",
            work_dir=tmp_path,
        )
        assert recovered["aif_qc"] == {"detection_method": "unavailable"}
        assert recovered["preload_qc"] is None
        assert recovered["preprocessing_stats"] is None
        assert recovered["registration_qc"] is None
