"""Integration tests for DSC sidecar AIF-provenance fields (v0.29.0).

Exercises the full ``_run_interface`` write path of ``DSCDerivativeWriter``
with a controlled ``qc_metrics`` dict and asserts the resulting JSON
sidecars carry the expected ``PerfusionProcessing.AIF`` block plus
per-map ``AIFDependent`` and ``TemplateAIFUsage`` tags.

Also verifies the drift-guard coverage check: every map emitted by the
writer must have a corresponding entry in ``TEMPLATE_AIF_USAGE``.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_outputs import (
    AIF_INDEPENDENT_MAPS,
    DSCDerivativeWriter,
    TEMPLATE_AIF_USAGE,
    apply_aif_sidecar,
    build_aif_provenance_block,
)


def _write_dummy_nifti(path: Path) -> str:
    """Write a 4×4×4 zero-filled NIfTI and return its path as str."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, path)  # type: ignore[attr-defined]
    return str(path)


@pytest.fixture
def minimal_bids_dir(tmp_path):
    """Minimal BIDS dataset root for `_build_base_sidecar` to consume."""
    bids = tmp_path / "bids"
    bids.mkdir()
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "test", "BIDSVersion": "1.8.0"})
    )
    return bids


@pytest.fixture
def minimal_inputs(tmp_path, minimal_bids_dir):
    """Dict of trait values that satisfies DSCDerivativeWriter mandatories."""
    return {
        "output_dir": str(tmp_path / "derivatives"),
        "subject_id": "test01",
        "session_id": "01",
        "bids_dir": str(minimal_bids_dir),
        "mtt_map": _write_dummy_nifti(tmp_path / "mtt.nii.gz"),
        "tmax_map": _write_dummy_nifti(tmp_path / "tmax.nii.gz"),
        "psr_map": _write_dummy_nifti(tmp_path / "psr.nii.gz"),
        "ph_map": _write_dummy_nifti(tmp_path / "ph.nii.gz"),
        "rel_cbf_map": _write_dummy_nifti(tmp_path / "rcbf.nii.gz"),
        "rel_cbv_map": _write_dummy_nifti(tmp_path / "rcbv.nii.gz"),
        "rel_cbv_auc_map": _write_dummy_nifti(tmp_path / "rcbv_auc.nii.gz"),
        "ncbv_map": _write_dummy_nifti(tmp_path / "ncbv.nii.gz"),
        "ncbf_map": _write_dummy_nifti(tmp_path / "ncbf.nii.gz"),
    }


def _load_sidecar(derivatives_root: Path, suffix: str) -> dict:
    """Load the one sidecar matching the given BIDS desc suffix."""
    matches = list(derivatives_root.rglob(f"*{suffix}*.json"))
    assert len(matches) == 1, (
        f"Expected exactly one sidecar matching {suffix!r}; "
        f"found {len(matches)}: {matches}"
    )
    with open(matches[0]) as f:
        return json.load(f)


def _aif_block(sidecar: dict) -> dict:
    return sidecar["PerfusionProcessing"]["AIF"]


def _run_writer(inputs: dict, qc_metrics: dict) -> Path:
    writer = DSCDerivativeWriter()
    for k, v in inputs.items():
        setattr(writer.inputs, k, v)
    writer.inputs.qc_metrics = qc_metrics
    writer.run()
    return Path(inputs["output_dir"])


# =============================================================================
# AIF block derivation (pure function tests)
# =============================================================================


class TestBuildAifProvenanceBlock:
    def test_measured_atlas_strict(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.85}}
        )
        assert block["Source"] == "measured"
        assert block["DetectionLevel"] == "atlas_strict"
        assert block["QualityFlag"] == "pass"
        assert block["GammaFitApplied"] is False
        assert block["CrossSubjectComparable"] is True
        assert "SyntheticFallbackReason" not in block
        assert "ExternalAIFVerified" not in block

    def test_gamma_fit_measured(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "atlas_relaxed",
                "quality_score": 0.75,
                "gamma_fit_applied": True,
            }}
        )
        assert block["Source"] == "gamma_fit_measured"
        assert block["GammaFitApplied"] is True
        assert block["QualityFlag"] == "pass"
        assert block["CrossSubjectComparable"] is True

    def test_template_synthetic(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "synthetic",
                "quality_score": 0.9,  # forced to poor regardless
            },
             "synthetic_fallback_reason": "no_candidates"}
        )
        assert block["Source"] == "template"
        assert block["QualityFlag"] == "poor"
        assert block["CrossSubjectComparable"] is False
        assert block["SyntheticFallbackReason"] == "no_candidates"

    def test_external(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "external",
                "quality_score": 0.65,
            }}
        )
        assert block["Source"] == "external"
        assert block["QualityFlag"] == "warn"
        assert block["CrossSubjectComparable"] is True
        assert block["ExternalAIFVerified"] is False

    def test_quality_flag_thresholds(self) -> None:
        for score, expected in [(0.9, "pass"), (0.7, "pass"),
                                (0.6, "warn"), (0.5, "warn"),
                                (0.4, "poor"), (0.1, "poor")]:
            block = build_aif_provenance_block(
                {"aif_detection": {"detection_method": "brain_permissive",
                         "quality_score": score}}
            )
            assert block["QualityFlag"] == expected, (
                f"score={score} → {block['QualityFlag']}, expected {expected}"
            )

    def test_empty_qc_returns_empty(self) -> None:
        assert build_aif_provenance_block(None) == {}
        assert build_aif_provenance_block({}) == {}
        assert build_aif_provenance_block({"aif_detection": {}}) == {}


# =============================================================================
# apply_aif_sidecar (per-map stamping)
# =============================================================================


class TestApplyAifSidecar:
    def test_k2_is_aif_independent_and_safe(self) -> None:
        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "synthetic"}}
        apply_aif_sidecar(sidecar, "k2", qc)
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"
        assert _aif_block(sidecar)["Source"] == "template"

    def test_mtt_is_dependent_and_qualitative(self) -> None:
        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "synthetic"}}
        apply_aif_sidecar(sidecar, "mtt", qc)
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "qualitative_only"

    def test_ncbv_is_dependent_but_safe(self) -> None:
        """nCBV is AIF-dependent in computation but NAWM ratio cancels
        amplitude → safe under template."""
        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "synthetic"}}
        apply_aif_sidecar(sidecar, "ncbv", qc)
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_ttp_is_independent_and_safe(self) -> None:
        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "synthetic"}}
        apply_aif_sidecar(sidecar, "ttp", qc)
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_parametric_r2_is_independent_and_safe(self) -> None:
        """Parametric-fit R² is AIF-amplitude-independent (model-fit
        diagnostic). Tagged ``safe``.
        """
        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "synthetic"}}
        apply_aif_sidecar(sidecar, "parametric_r2", qc)
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_fallback_mask_is_independent_and_safe(self) -> None:
        """oSVD-fallback mask is a binary gate decision; AIF-independent."""
        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "synthetic"}}
        apply_aif_sidecar(sidecar, "fallback", qc)
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_unknown_map_key_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown DSC map key"):
            apply_aif_sidecar({}, "not_a_real_map", {"aif_detection": {}})

    def test_preserves_existing_perfusion_processing(self) -> None:
        sidecar = {"PerfusionProcessing": {"ContrastAgent": {"GenericName": "gadobutrol"}}}
        qc = {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.9}}
        apply_aif_sidecar(sidecar, "cbv", qc)
        perf = sidecar["PerfusionProcessing"]
        assert perf["ContrastAgent"]["GenericName"] == "gadobutrol"
        assert perf["AIF"]["Source"] == "measured"

    def test_missing_qc_stamps_tags_only(self) -> None:
        sidecar: dict = {}
        apply_aif_sidecar(sidecar, "mtt", None)
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "qualitative_only"
        assert "PerfusionProcessing" not in sidecar


# =============================================================================
# TEMPLATE_AIF_USAGE coverage (drift guard)
# =============================================================================


class TestTemplateAifUsageCoverage:
    """Every map_key the writer might call apply_aif_sidecar() with must
    be registered in TEMPLATE_AIF_USAGE — otherwise the writer will
    raise KeyError at runtime on a real pipeline run.
    """

    def test_all_writer_keys_registered(self) -> None:
        # Mirror the key sets hardcoded in DSCDerivativeWriter._run_interface.
        rel_map_keys = {"cbf", "cbv", "cbv_auc", "auc", "auc_corr"}
        core_map_keys = {"mtt", "tmax", "psr", "ph"}
        optional_map_keys = {"ttp", "ttpCorrected", "rttp", "cnr",
                             "auc_diff", "snr", "tsnr"}
        normalized_keys = {"ncbv", "ncbf"}
        leakage_keys = {"k2"}
        vascular_keys = {"alpha", "beta", "cth"}
        bayesian_keys = {"cbf_std", "cbv_std", "cbfCV", "cbvCV",
                         "cbfCI95lower", "cbfCI95upper",
                         "cbvCI95lower", "cbvCI95upper"}
        # Parametric-fit diagnostics added in v0.35.0.
        parametric_diag_keys = {"parametric_r2", "fallback"}
        smoothed_hemo_keys = {"mtt", "rel_cbf", "rel_cbv", "ncbf", "ncbv"}
        smoothed_auc_keys = {"rel_auc_corr"}

        all_keys = (rel_map_keys | core_map_keys | optional_map_keys
                    | normalized_keys | leakage_keys | vascular_keys
                    | bayesian_keys | parametric_diag_keys
                    | smoothed_hemo_keys | smoothed_auc_keys)
        missing = all_keys - set(TEMPLATE_AIF_USAGE.keys())
        assert not missing, (
            f"Writer keys not registered in TEMPLATE_AIF_USAGE: {missing}. "
            "Add them with the appropriate tag."
        )

    def test_aif_independent_subset_of_registered(self) -> None:
        extra = AIF_INDEPENDENT_MAPS - set(TEMPLATE_AIF_USAGE.keys())
        assert not extra, (
            f"AIF_INDEPENDENT_MAPS contains keys not in TEMPLATE_AIF_USAGE: "
            f"{extra}"
        )


# =============================================================================
# End-to-end writer integration
# =============================================================================


class TestWriterAifBlockPresentOnAllMaps:
    """Running the full writer with a measured-AIF qc_metrics dict stamps
    PerfusionProcessing.AIF + AIFDependent + TemplateAIFUsage on every
    emitted sidecar."""

    def test_all_core_and_rel_maps_stamped_measured(self, minimal_inputs) -> None:
        qc = {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.85}}
        derivatives = _run_writer(minimal_inputs, qc)

        for suffix in ["dscRelAUC_cbv", "dscRel_cbv", "dscRel_cbf",
                       "dscNawmNorm_cbv", "dscNawmNorm_cbf",
                       "dsc_mtt", "dsc_tmax", "dsc_psr", "dsc_ph"]:
            sidecar = _load_sidecar(derivatives, suffix)
            assert _aif_block(sidecar)["Source"] == "measured", suffix
            assert _aif_block(sidecar)["QualityFlag"] == "pass", suffix
            assert "AIFDependent" in sidecar, suffix
            assert "TemplateAIFUsage" in sidecar, suffix

    def test_k2_independent(self, minimal_inputs, tmp_path) -> None:
        minimal_inputs["k2_map"] = _write_dummy_nifti(tmp_path / "k2.nii.gz")
        qc = {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.9}}
        derivatives = _run_writer(minimal_inputs, qc)
        sidecar = _load_sidecar(derivatives, "dsc_k2")
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_synthetic_template_flags(self, minimal_inputs) -> None:
        qc = {
            "aif_detection": {"detection_method": "synthetic", "quality_score": 0.3},
            "synthetic_fallback_reason": "no_candidates",
        }
        derivatives = _run_writer(minimal_inputs, qc)
        cbv = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert _aif_block(cbv)["Source"] == "template"
        assert _aif_block(cbv)["CrossSubjectComparable"] is False
        assert _aif_block(cbv)["SyntheticFallbackReason"] == "no_candidates"
        assert cbv["AIFDependent"] is True
        assert cbv["TemplateAIFUsage"] == "ratio_defensible"

        mtt = _load_sidecar(derivatives, "dsc_mtt")
        assert mtt["TemplateAIFUsage"] == "qualitative_only"

    def test_gamma_fit_measured_source(self, minimal_inputs) -> None:
        qc = {"aif_detection": {
            "detection_method": "atlas_relaxed",
            "quality_score": 0.72,
            "gamma_fit_applied": True,
        }}
        derivatives = _run_writer(minimal_inputs, qc)
        cbv = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert _aif_block(cbv)["Source"] == "gamma_fit_measured"
        assert _aif_block(cbv)["GammaFitApplied"] is True
        assert _aif_block(cbv)["CrossSubjectComparable"] is True


class TestSmoothedSidecars:
    def test_smoothed_cbv_synthetic(self, minimal_inputs, tmp_path) -> None:
        minimal_inputs["smooth_fwhm"] = 3.0
        minimal_inputs["smooth_hemodynamic"] = {
            "rel_cbv": _write_dummy_nifti(tmp_path / "rcbv_sm.nii.gz"),
            "ncbv": _write_dummy_nifti(tmp_path / "ncbv_sm.nii.gz"),
        }
        qc = {"aif_detection": {"detection_method": "synthetic", "quality_score": 0.3}}
        derivatives = _run_writer(minimal_inputs, qc)
        raw = _load_sidecar(derivatives, "dscRelAUCSm3_cbv")
        assert _aif_block(raw)["Source"] == "template"
        assert raw["TemplateAIFUsage"] == "ratio_defensible"
        ncbv_sm = _load_sidecar(derivatives, "dscNawmNormSm3_cbv")
        assert _aif_block(ncbv_sm)["Source"] == "template"
        assert ncbv_sm["TemplateAIFUsage"] == "safe"


class TestMissingAifQc:
    def test_missing_qc_stamps_tags_but_no_block(self, minimal_inputs) -> None:
        """Without qc_metrics the AIF block cannot be assembled, but
        per-map AIFDependent/TemplateAIFUsage tags are still stamped."""
        writer = DSCDerivativeWriter()
        for k, v in minimal_inputs.items():
            setattr(writer.inputs, k, v)
        writer.run()
        derivatives = Path(minimal_inputs["output_dir"])
        sidecar = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "ratio_defensible"
        perf = sidecar.get("PerfusionProcessing", {})
        assert "AIF" not in perf
