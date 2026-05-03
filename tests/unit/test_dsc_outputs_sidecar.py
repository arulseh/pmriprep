"""Integration tests for DSC sidecar AIF-provenance fields.

Exercises the full DSC datasinks factory write path with a controlled
``qc_metrics`` dict and asserts the resulting JSON sidecars carry the
expected ``PerfusionProcessing.AIF`` block plus per-map ``AIFDependent``
and ``TemplateAIFUsage`` tags.

Also verifies the drift-guard coverage check: every map emitted by the
factory must have a corresponding entry in ``TEMPLATE_AIF_USAGE``.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.utils.dsc_aif_provenance import (
    AIF_INDEPENDENT_MAPS,
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
    """Minimal BIDS dataset root with a stub source DSC NIfTI."""
    bids = tmp_path / "bids"
    perf = bids / "sub-test01" / "ses-01" / "perf"
    perf.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "test", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = perf / "sub-test01_ses-01_dsc.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src)
    return bids


@pytest.fixture
def minimal_inputs(tmp_path, minimal_bids_dir):
    """Dict of trait values that satisfies the DSC factory inputnode."""
    src_dsc = minimal_bids_dir / "sub-test01" / "ses-01" / "perf" / "sub-test01_ses-01_dsc.nii.gz"
    return {
        "output_dir": str(tmp_path / "derivatives"),
        "subject_id": "test01",
        "session_id": "01",
        "bids_dir": str(minimal_bids_dir),
        "source_dsc": str(src_dsc),
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


def _run_writer(inputs: dict, qc_metrics: dict | None, tmp_path: Path | None = None) -> Path:
    """Build and run the DSC datasinks factory workflow."""
    from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

    # Forward every per-input key supplied by the test as ``present_inputs``
    # so the factory builds a sink for it.
    present = set(inputs.keys()) - {
        "output_dir", "bids_dir", "subject_id", "session_id", "qc_metrics",
        "smooth_fwhm", "smooth_hemodynamic", "smooth_auc",
    }
    has_smoothing = "smooth_hemodynamic" in inputs or "smooth_auc" in inputs
    wf = init_dsc_datasinks_wf(
        output_dir=inputs["output_dir"],
        bids_dir=inputs["bids_dir"],
        subject_id=inputs["subject_id"],
        session_id=inputs["session_id"],
        present_inputs=present,
        has_smoothing=has_smoothing,
    )
    if tmp_path is not None:
        wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    for k, v in inputs.items():
        if k in {"output_dir", "subject_id", "session_id", "bids_dir"}:
            continue
        setattr(inp, k, v)
    if qc_metrics is not None:
        inp.qc_metrics = qc_metrics
    wf.run()
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

    def test_synthetic_template(self) -> None:
        block = build_aif_provenance_block({
            "aif_detection": {"detection_method": "synthetic", "quality_score": 0.2},
            "synthetic_fallback_reason": "no_candidates",
        })
        assert block["Source"] == "template"
        assert block["QualityFlag"] == "poor"
        assert block["CrossSubjectComparable"] is False
        assert block["SyntheticFallbackReason"] == "no_candidates"

    def test_external_aif(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "external", "quality_score": 0.9}}
        )
        assert block["Source"] == "external"
        assert block["ExternalAIFVerified"] is False

    def test_gamma_fit_applied(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_relaxed",
                                "quality_score": 0.72, "gamma_fit_applied": True}}
        )
        assert block["Source"] == "gamma_fit_measured"
        assert block["GammaFitApplied"] is True

    def test_quality_warn(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.6}}
        )
        assert block["QualityFlag"] == "warn"

    def test_quality_poor(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.3}}
        )
        assert block["QualityFlag"] == "poor"

    def test_missing_qc_returns_empty(self) -> None:
        assert build_aif_provenance_block(None) == {}
        assert build_aif_provenance_block({}) == {}
        assert build_aif_provenance_block({"aif_detection": {}}) == {}


class TestApplyAifSidecar:
    def test_stamps_dependent_and_block(self) -> None:
        sidecar: dict = {}
        apply_aif_sidecar(
            sidecar, "cbf",
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.8}},
        )
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "ratio_defensible"
        assert sidecar["PerfusionProcessing"]["AIF"]["Source"] == "measured"

    def test_independent_map_marked(self) -> None:
        sidecar: dict = {}
        apply_aif_sidecar(
            sidecar, "k2",
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.9}},
        )
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_unknown_map_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown DSC map key"):
            apply_aif_sidecar({}, "not_a_real_map", None)

    def test_missing_qc_stamps_tags_only(self) -> None:
        sidecar: dict = {}
        apply_aif_sidecar(sidecar, "cbf", None)
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "ratio_defensible"
        assert "PerfusionProcessing" not in sidecar


# =============================================================================
# Drift guard
# =============================================================================


class TestTemplateAifUsageCoverage:
    """Every map_key the factory might call apply_aif_sidecar with must be
    registered in TEMPLATE_AIF_USAGE — otherwise the workflow raises
    KeyError at runtime on a real pipeline run.
    """

    def test_factory_keys_subset_of_registered(self) -> None:
        from pmriprep.workflows.dsc.datasinks import _DSC_OUTPUT_KEYS

        missing = _DSC_OUTPUT_KEYS - set(TEMPLATE_AIF_USAGE.keys())
        assert not missing, (
            f"Factory keys not registered in TEMPLATE_AIF_USAGE: {missing}. "
            "Add them with the appropriate tag."
        )

    def test_aif_independent_subset_of_registered(self) -> None:
        extra = AIF_INDEPENDENT_MAPS - set(TEMPLATE_AIF_USAGE.keys())
        assert not extra, (
            f"AIF_INDEPENDENT_MAPS contains keys not in TEMPLATE_AIF_USAGE: "
            f"{extra}"
        )


# =============================================================================
# End-to-end factory integration
# =============================================================================


class TestWriterAifBlockPresentOnAllMaps:
    """Running the full factory with a measured-AIF qc_metrics dict stamps
    PerfusionProcessing.AIF + AIFDependent + TemplateAIFUsage on every
    emitted sidecar."""

    def test_all_core_and_rel_maps_stamped_measured(self, minimal_inputs, tmp_path) -> None:
        qc = {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.85}}
        derivatives = _run_writer(minimal_inputs, qc, tmp_path)

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
        derivatives = _run_writer(minimal_inputs, qc, tmp_path)
        sidecar = _load_sidecar(derivatives, "dsc_k2")
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_synthetic_template_flags(self, minimal_inputs, tmp_path) -> None:
        qc = {
            "aif_detection": {"detection_method": "synthetic", "quality_score": 0.3},
            "synthetic_fallback_reason": "no_candidates",
        }
        derivatives = _run_writer(minimal_inputs, qc, tmp_path)
        cbv = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert _aif_block(cbv)["Source"] == "template"
        assert _aif_block(cbv)["CrossSubjectComparable"] is False
        assert _aif_block(cbv)["SyntheticFallbackReason"] == "no_candidates"
        assert cbv["AIFDependent"] is True
        assert cbv["TemplateAIFUsage"] == "ratio_defensible"

        mtt = _load_sidecar(derivatives, "dsc_mtt")
        assert mtt["TemplateAIFUsage"] == "qualitative_only"

    def test_gamma_fit_measured_source(self, minimal_inputs, tmp_path) -> None:
        qc = {"aif_detection": {
            "detection_method": "atlas_relaxed",
            "quality_score": 0.72,
            "gamma_fit_applied": True,
        }}
        derivatives = _run_writer(minimal_inputs, qc, tmp_path)
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
        derivatives = _run_writer(minimal_inputs, qc, tmp_path)
        raw = _load_sidecar(derivatives, "dscRelAUCSm3_cbv")
        assert _aif_block(raw)["Source"] == "template"
        assert raw["TemplateAIFUsage"] == "ratio_defensible"
        ncbv_sm = _load_sidecar(derivatives, "dscNawmNormSm3_cbv")
        assert _aif_block(ncbv_sm)["Source"] == "template"
        assert ncbv_sm["TemplateAIFUsage"] == "safe"


class TestMissingAifQc:
    def test_missing_qc_stamps_tags_but_no_block(self, minimal_inputs, tmp_path) -> None:
        """Without qc_metrics the AIF block cannot be assembled, but
        per-map AIFDependent/TemplateAIFUsage tags are still stamped."""
        derivatives = _run_writer(minimal_inputs, None, tmp_path)
        sidecar = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "ratio_defensible"


# =============================================================================
# Structural underlays: T1w/FLAIR/T2w in DSC space must land in anat/, not perf/
# =============================================================================


class TestStructuralUnderlayDatatype:
    """Regression: T1w-in-DSC sink must write to anat/ (not perf/).

    niworkflows derives ``datatype`` from ``source_file``; with
    ``source_dsc`` (a perf-datatype file) the default would be ``perf``,
    which makes pybids reject the path because BIDS does not allow ``T1w``
    suffix in ``perf/``. ``_add_anat_sink`` must override datatype="anat".
    """

    def test_t1w_in_dsc_lands_in_anat(self, minimal_inputs, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

        minimal_inputs["t1w_in_dsc"] = _write_dummy_nifti(tmp_path / "t1w_in_dsc.nii.gz")
        present = (set(minimal_inputs.keys()) - {
            "output_dir", "bids_dir", "subject_id", "session_id", "qc_metrics",
        })
        wf = init_dsc_datasinks_wf(
            output_dir=minimal_inputs["output_dir"],
            bids_dir=minimal_inputs["bids_dir"],
            subject_id=minimal_inputs["subject_id"],
            session_id=minimal_inputs["session_id"],
            present_inputs=present,
            has_postcon_t1w=True,  # triggers ce-GAD entity
        )
        wf.base_dir = str(tmp_path / "work")
        inp = wf.inputs.inputnode
        for k, v in minimal_inputs.items():
            if k in {"output_dir", "subject_id", "session_id", "bids_dir"}:
                continue
            setattr(inp, k, v)
        wf.run()  # would raise "Could not build path" before fix

        derivatives = Path(minimal_inputs["output_dir"])
        anat_dir = derivatives / "pmriprep" / "sub-test01" / "ses-01" / "anat"
        perf_dir = derivatives / "pmriprep" / "sub-test01" / "ses-01" / "perf"
        # Underlay must land in anat/ with ce-GAD + space-dsc entities,
        # NOT in perf/. Postcon-mode → ce-GAD must propagate via the
        # niworkflows `ceagent` trait (NOT `ce` — that trait doesn't exist).
        expected_nii = anat_dir / "sub-test01_ses-01_ce-GAD_space-dsc_T1w.nii.gz"
        assert expected_nii.exists(), (
            f"underlay missing at {expected_nii}; "
            f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
        )
        perf_t1w = list(perf_dir.glob("*T1w.nii.gz")) if perf_dir.exists() else []
        assert not perf_t1w, f"T1w must not land in perf/; got {perf_t1w}"
