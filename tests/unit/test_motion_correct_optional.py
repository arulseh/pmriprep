"""Tests for configurable perfusion motion correction and its provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------
class TestConfigSurface:
    def test_defaults_true_both_configs(self) -> None:
        from pmriprep.config.dce import DCEConfig
        from pmriprep.config.dsc import DSCConfig

        assert DSCConfig().motion_correct is True
        assert DCEConfig().motion_correct is True

    def test_roundtrip_false_dsc(self) -> None:
        from pmriprep.config.loader import build_dsc_config

        cfg = build_dsc_config({"dsc": {"motion_correct": False}})
        assert cfg.motion_correct is False

    def test_roundtrip_false_dce(self) -> None:
        from pmriprep.config.loader import build_dce_config

        cfg = build_dce_config({"dce": {"motion_correct": False}})
        assert cfg.motion_correct is False


# ---------------------------------------------------------------------------
# DCE motion_thresh config parity (mirrors DSC)
# ---------------------------------------------------------------------------
class TestDCEMotionThresh:
    def test_default_is_half_mm(self) -> None:
        from pmriprep.config.dce import DCEConfig

        assert DCEConfig().motion_thresh == 0.5

    def test_post_init_coerces_string(self) -> None:
        from pmriprep.config.dce import DCEConfig

        cfg = DCEConfig(motion_thresh="0.3")
        assert cfg.motion_thresh == 0.3
        assert isinstance(cfg.motion_thresh, float)

    def test_roundtrip_via_loader(self) -> None:
        from pmriprep.config.loader import build_dce_config

        cfg = build_dce_config({"dce": {"motion_thresh": 0.3}})
        assert cfg.motion_thresh == 0.3

    def test_wired_into_dce_preprocess_node(self, tmp_path: Path) -> None:
        from pmriprep.config.dce import DCEConfig
        from pmriprep.workflows.dce.base import init_dce_preproc_wf

        bids = tmp_path / "bids"
        ses = bids / "sub-0001" / "ses-0001"
        (ses / "perf").mkdir(parents=True)
        (ses / "perf" / "sub-0001_ses-0001_dce.nii.gz").touch()
        (bids / "dataset_description.json").write_text(
            '{"Name": "T", "BIDSVersion": "1.9.0"}'
        )

        wf = init_dce_preproc_wf(
            bids_dir=bids, output_dir=bids / "derivatives",
            subject_id="0001", session_id="0001",
            config=DCEConfig(motion_thresh=0.3), has_tissue_seg=True,
        )
        node = next(n for n in wf._get_all_nodes() if n.name == "dce_preprocess")
        # Config value must reach the node, not the interface default (0.5).
        assert node.inputs.motion_thresh == 0.3


# ---------------------------------------------------------------------------
# DSC workflow build-time branch
# ---------------------------------------------------------------------------
def _outputnode_source(wf, field):
    """Return (src_node_name, src_field) feeding ``outputnode.<field>``."""
    for u, v, d in wf._graph.edges(data=True):
        if v.name == "outputnode":
            for out, inp in d.get("connect", []):
                if inp == field:
                    return u.name, out
    return None


class TestDSCMotionBranch:
    def _wf(self, apply_moco):
        from pmriprep.workflows.dsc.base import init_dsc_motion_wf

        return init_dsc_motion_wf(
            motion_thresh=0.5, baseline_frames=5, subject_id="01",
            session_id=None, mem_gb=2.0, apply_moco=apply_moco,
        )

    def test_apply_uses_motion_correct(self) -> None:
        wf = self._wf(True)
        assert _outputnode_source(wf, "dsc_moco") == ("motion_correct", "out_file")
        assert _outputnode_source(wf, "dsc_mean") == ("motion_correct", "mean_img")
        names = {n.name for n in wf._get_all_nodes()}
        assert "reorient_mean" not in names

    def test_off_uses_reoriented_series(self) -> None:
        wf = self._wf(False)
        # Uncorrected series + its mean feed downstream ...
        assert _outputnode_source(wf, "dsc_moco") == ("dsc_reorient", "out_file")
        assert _outputnode_source(wf, "dsc_mean") == ("reorient_mean", "mean_file")
        # ... but residual motion is still estimated by mcflirt.
        assert _outputnode_source(wf, "motion_params") == ("motion_correct", "out_params")
        assert _outputnode_source(wf, "fd_values") == ("motion_correct", "fd")

    def test_off_adds_reorient_mean_node(self) -> None:
        names = {n.name for n in self._wf(False)._get_all_nodes()}
        assert "reorient_mean" in names

    def test_baseline_frames_reaches_motion_interface(self) -> None:
        wf = self._wf(True)
        node = next(n for n in wf._get_all_nodes() if n.name == "motion_correct")
        assert node.inputs.baseline_frames == 5
        assert not hasattr(node.inputs, "ref_vol")


class TestDSCMotionReference:
    """Tests for the DSC motion-reference contract."""

    def test_reference_mean_uses_configured_baseline(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import nibabel as nib

        from pmriprep.interfaces.dsc import MotionCorrection

        data = np.broadcast_to(
            np.arange(1, 7, dtype=np.float32), (2, 2, 2, 6),
        ).copy()
        in_file = tmp_path / "dsc.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), in_file)
        monkeypatch.chdir(tmp_path)

        interface = MotionCorrection(in_file=str(in_file), baseline_frames=3)
        reference = nib.load(interface._compute_pre_bolus_mean()).get_fdata()

        np.testing.assert_allclose(reference, 2.0)


# ---------------------------------------------------------------------------
# Confounds provenance (condition A)
# ---------------------------------------------------------------------------
class TestConfoundsProvenance:
    def test_applied_true_default(self, tmp_path: Path) -> None:
        from pmriprep.utils.outputs import write_confounds_sidecar

        out = tmp_path / "c.json"
        write_confounds_sidecar(out)
        data = json.loads(out.read_text())
        assert data["MotionCorrectionApplied"] is True
        assert "Note" not in data

    def test_applied_false_records_note(self, tmp_path: Path) -> None:
        from pmriprep.utils.outputs import write_confounds_sidecar

        out = tmp_path / "c.json"
        write_confounds_sidecar(out, motion_applied=False)
        data = json.loads(out.read_text())
        assert data["MotionCorrectionApplied"] is False
        assert "residual motion" in data["Note"].lower()


# ---------------------------------------------------------------------------
# Motion QC plot annotation (condition D)
# ---------------------------------------------------------------------------
class TestMotionQCPlotAnnotation:
    def _params(self, tmp_path):
        par = tmp_path / "motion.par"
        np.savetxt(par, np.zeros((10, 6)))
        return str(par)

    def test_applied_no_note(self, tmp_path: Path, monkeypatch) -> None:
        from pmriprep.workflows.dsc._helpers import _generate_motion_qc_plot

        monkeypatch.chdir(tmp_path)
        _, summary = _generate_motion_qc_plot(
            self._params(tmp_path), tr=1.5, motion_applied=True,
        )
        assert summary["motion_applied"] is True
        assert "motion_note" not in summary

    def test_off_carries_residual_note(self, tmp_path: Path, monkeypatch) -> None:
        from pmriprep.workflows.dsc._helpers import _generate_motion_qc_plot

        monkeypatch.chdir(tmp_path)
        _, summary = _generate_motion_qc_plot(
            self._params(tmp_path), tr=1.5, motion_applied=False,
        )
        assert summary["motion_applied"] is False
        assert "residual motion" in summary["motion_note"].lower()


# ---------------------------------------------------------------------------
# Residual-FD recommendation (condition B)
# ---------------------------------------------------------------------------
class TestResidualMotionRecommendation:
    def _dsc_qc(self, *, motion_applied, status):
        # DSC nests motion_applied under preprocessing.motion_summary.
        return {
            "preprocessing": {
                "motion_status": status,
                "fd_mean": 0.8, "fd_max": 1.5, "percent_outliers": 20.0,
                "motion_summary": {"motion_applied": motion_applied},
            },
            "aif_detection": {},
        }

    def test_off_and_marginal_warns_bias(self) -> None:
        from pmriprep.utils.qc_standards import QCStatus, recommend_dsc_qc

        out = recommend_dsc_qc(self._dsc_qc(motion_applied=False, status=QCStatus.WARN))
        text = json.dumps(out["recommendation"]).lower()
        assert "not" in text and "applied" in text and "biased" in text

    def test_on_and_marginal_is_generic(self) -> None:
        from pmriprep.utils.qc_standards import QCStatus, recommend_dsc_qc

        out = recommend_dsc_qc(self._dsc_qc(motion_applied=True, status=QCStatus.WARN))
        text = json.dumps(out["recommendation"]).lower()
        assert "marginal" in text
        assert "biased" not in text

    def test_dce_off_and_marginal_warns_bias(self) -> None:
        from pmriprep.utils.qc_standards import QCStatus, recommend_dce_qc

        # DCE stores motion_applied at the top of the preprocessing dict.
        qc = {
            "preprocessing": {
                "motion_status": QCStatus.WARN,
                "fd_mean": 0.8, "fd_max": 1.5, "percent_outliers": 20.0,
                "motion_applied": False,
            },
        }
        out = recommend_dce_qc(qc)
        text = json.dumps(out["recommendation"]).lower()
        assert "biased" in text


# ---------------------------------------------------------------------------
# DCE interface passthrough (mcflirt mocked): apply=False must NOT re-resample
# ---------------------------------------------------------------------------
class TestDCEInterfacePassthrough:
    def _make_dce(self, tmp_path):
        import nibabel as nib

        rng = np.random.default_rng(0)
        data = np.full((6, 6, 6, 12), 100.0)
        data[..., 4:] += np.linspace(0, 50, 8)  # enhancement after baseline
        data += rng.normal(0, 1.0, data.shape)
        path = tmp_path / "dce.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
        return str(path), data.astype(np.float32)

    def _patch_mcflirt(self, monkeypatch, in_file):
        """Fake mcflirt: write a *distinct* 'corrected' 4D (input + 100) + .par."""
        import subprocess

        import nibabel as nib

        real_run = subprocess.run

        def fake_run(cmd, **kwargs):
            if not (isinstance(cmd, (list, tuple)) and "mcflirt" in cmd):
                return real_run(cmd, **kwargs)
            out_base = cmd[cmd.index("-out") + 1]
            img = nib.load(in_file)
            corrected = img.get_fdata() + 100.0  # clearly != input
            nib.save(
                nib.Nifti1Image(corrected.astype(np.float32), img.affine, img.header),
                f"{out_base}.nii.gz",
            )
            n_t = img.shape[3]
            np.savetxt(f"{out_base}.par", np.zeros((n_t, 6)))

            class _R:
                returncode = 0
                stderr = ""

            return _R()

        monkeypatch.setattr(subprocess, "run", fake_run)

    def _run(self, tmp_path, monkeypatch, apply_moco):
        import nibabel as nib

        from pmriprep.interfaces.dce import DCEPreprocess

        in_file, in_data = self._make_dce(tmp_path)
        self._patch_mcflirt(monkeypatch, in_file)
        monkeypatch.chdir(tmp_path)

        iface = DCEPreprocess(
            in_file=in_file, baseline_frames=5,
            apply_motion_correction=apply_moco,
        )
        res = iface.run()
        out = nib.load(res.outputs.out_file).get_fdata().astype(np.float32)
        return in_data, out, res

    def test_off_passes_input_through(self, tmp_path: Path, monkeypatch) -> None:
        import nibabel as nib

        in_data, out, res = self._run(tmp_path, monkeypatch, apply_moco=False)
        # out_file must equal the input, NOT the +100 'corrected' series.
        assert np.allclose(out, in_data)
        # mean_img / baseline_mean_img must also derive from the input, not the
        # mcflirt-resampled series (condition C: means tied to processed series).
        mean_img = nib.load(res.outputs.mean_img).get_fdata()
        assert np.allclose(mean_img, in_data.mean(axis=3), atol=1e-3)
        base_img = nib.load(res.outputs.baseline_mean_img).get_fdata()
        assert np.allclose(base_img, in_data[..., :5].mean(axis=3), atol=1e-3)
        # mcflirt still ran, so residual motion params are produced.
        assert Path(res.outputs.out_params).exists()
        assert res.outputs.qc_metrics["motion_applied"] is False

    def test_on_uses_corrected(self, tmp_path: Path, monkeypatch) -> None:
        in_data, out, res = self._run(tmp_path, monkeypatch, apply_moco=True)
        assert np.allclose(out, in_data + 100.0)
        assert res.outputs.qc_metrics["motion_applied"] is True


# ---------------------------------------------------------------------------
# Methods boilerplate: motion-correction sentence must be conditional
# ---------------------------------------------------------------------------
class TestBoilerplateConditionalMoco:
    def test_dsc_applied_says_motion_corrected(self) -> None:
        from pmriprep.utils.report_utils import _build_dsc_boilerplate

        text = _build_dsc_boilerplate({"preprocessing": {"motion_applied": True}})
        assert "motion-corrected using MCFLIRT" in text

    def test_dsc_off_does_not_claim_correction(self) -> None:
        from pmriprep.utils.report_utils import _build_dsc_boilerplate

        text = _build_dsc_boilerplate({"preprocessing": {"motion_applied": False}})
        assert "motion-corrected using MCFLIRT" not in text
        assert "DSC motion correction was omitted" in text
        assert "residual motion in the input was assessed for quality control" in text

    def test_dsc_default_absent_flag_claims_correction(self) -> None:
        # Absent flag (older sidecars / default runs) → assume applied.
        from pmriprep.utils.report_utils import _build_dsc_boilerplate

        text = _build_dsc_boilerplate({"preprocessing": {}})
        assert "motion-corrected using MCFLIRT" in text

    def test_dce_applied_says_motion_corrected(self) -> None:
        from pmriprep.utils.report_utils import _build_dce_boilerplate

        text = _build_dce_boilerplate({"preprocessing": {"motion_applied": True}})
        # DCE names MCFLIRT + cites it, mirroring DSC.
        assert "motion-corrected using MCFLIRT [@jenkinson_2002]" in text

    def test_dce_off_does_not_claim_correction(self) -> None:
        from pmriprep.utils.report_utils import _build_dce_boilerplate

        text = _build_dce_boilerplate({"preprocessing": {"motion_applied": False}})
        assert "motion-corrected using MCFLIRT" not in text
        assert "DCE motion correction was omitted" in text
        assert "DCE permeability images" not in text
        assert "residual motion in the input was assessed for quality control" in text


class TestMotionAppliedSurvivesPipelineQCFilter:
    """The DSC flat motion_applied must survive filter_pipeline_qc (motion_summary
    is stripped), so the report boilerplate can read it from the sidecar."""

    def test_flat_key_present_and_survives_strip(self) -> None:
        from pmriprep.utils.sidecar_builders import filter_pipeline_qc
        from pmriprep.workflows.dsc._helpers import _merge_qc_metrics

        merged = _merge_qc_metrics(
            motion_qc={"motion_status": "pass"},
            preload_qc={}, aif_qc={}, deconv_qc={}, registration_qc={},
            brain_extraction_qc={
                "method_used": "hdbet",
                "hdbet_device": "cuda",
                "hdbet_tta_used": True,
            },
            normalization_factor=0.0,
            motion_summary={"motion_applied": False, "fd_mean": 0.05},
        )
        assert merged["preprocessing"]["motion_applied"] is False
        assert merged["brain_extraction"]["hdbet_tta_used"] is True

        filtered = filter_pipeline_qc(merged)
        # motion_summary is stripped, but the flat flag remains.
        assert "motion_summary" not in filtered["preprocessing"]
        assert filtered["preprocessing"]["motion_applied"] is False
