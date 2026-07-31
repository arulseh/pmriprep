"""Aggregator snapshot: ``_load_pipeline_sidecars`` returns the nested shape.

The HTML aggregator reads canonical per-image sidecars (anchor
``desc-{anat,dsc,dce}Preproc_*`` + the pipeline-internal resume sidecars
for DSC/DCE) and translates flat ``QC*`` keys into the nested
``qc_metrics`` shape that the boilerplate generators
(``_build_dsc_boilerplate`` etc.) expect.

This test pins the translation table by writing realistic per-image
sidecars to a temp session and asserting the loader produces a nested
dict with the right top-level keys.
"""
from __future__ import annotations

import json
from pathlib import Path

from pmriprep.utils.report_utils import _load_pipeline_sidecars


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_load_pipeline_sidecars_dsc(tmp_path: Path) -> None:
    """DSC: anchor PipelineQC + QC* flat keys."""
    ses = tmp_path / "pmriprep" / "sub-01" / "ses-0001"
    prefix = "sub-01_ses-0001"

    _write(
        ses / "perf" / f"{prefix}_desc-dscPreproc_dsc.json",
        {
            "Description": "Motion-corrected 4D DSC time series",
            "QCRecommendationAction": "review",
            "QCRecommendationReason": "AIF marginal",
            "QCWarnings": ["Brain mask in DSC space not available"],
            "QCProvenanceWarnings": ["DSC preload auto-detected"],
            "QCAcquisitionWarnings": ["Low effective dose"],
            "QCProcessingWarnings": ["AIF fell back to template"],
            "PipelineQC": {
                "aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.8},
                "preload": {"k2_method": "bsw", "preload_classification": "none"},
                "preprocessing": {"fd_mean": 0.12, "n_volumes": 60},
            },
        },
    )

    out = _load_pipeline_sidecars(ses, prefix, "dsc")

    assert out["aif_detection"]["detection_method"] == "atlas_strict"
    assert out["preload"]["k2_method"] == "bsw"
    assert out["preprocessing"]["fd_mean"] == 0.12
    assert out["recommendation"]["action"] == "review"
    assert out["recommendation"]["reason"] == "AIF marginal"
    assert out["warnings"] == ["Brain mask in DSC space not available"]
    assert out["provenance_warnings"] == ["DSC preload auto-detected"]
    assert out["acquisition_warnings"] == ["Low effective dose"]
    assert out["processing_warnings"] == ["AIF fell back to template"]


def test_load_pipeline_sidecars_dce(tmp_path: Path) -> None:
    """DCE: anchor PipelineQC + QC* flat keys."""
    ses = tmp_path / "pmriprep" / "sub-02" / "ses-0001"
    prefix = "sub-02_ses-0001"

    _write(
        ses / "perf" / f"{prefix}_desc-dcePreproc_dce.json",
        {
            "QCRecommendationAction": "include",
            "QCRecommendationReason": "AIF good",
            "QCWarnings": ["NAWM mask in DCE space not available"],
            "PipelineQC": {
                "t1_mapping": {"method": "nlls", "b1_correction": "afi"},
                "pk_modeling": {"model": "tofts", "qc_voxel_rsquared": 0.85},
            },
        },
    )

    out = _load_pipeline_sidecars(ses, prefix, "dce")

    assert out["t1_mapping"]["method"] == "nlls"
    assert out["pk_modeling"]["model"] == "tofts"
    assert out["recommendation"]["action"] == "include"
    assert out["warnings"] == ["NAWM mask in DCE space not available"]


def test_load_pipeline_sidecars_anat(tmp_path: Path) -> None:
    """Anat: only QC* warnings flat keys (no recommendation pathway)."""
    ses = tmp_path / "pmriprep" / "sub-03" / "ses-0001"
    prefix = "sub-03_ses-0001"

    _write(
        ses / "anat" / f"{prefix}_space-T1w_desc-preproc_T1w.json",
        {
            "Description": "Preprocessed T1w",
            "QCWarnings": [
                "Tissue segmentation (WM/GM/CSF probability maps) not available",
                "FLAIR not available",
            ],
        },
    )

    out = _load_pipeline_sidecars(ses, prefix, "anat")

    assert "recommendation" not in out
    assert out["warnings"] == [
        "Tissue segmentation (WM/GM/CSF probability maps) not available",
        "FLAIR not available",
    ]


def test_load_pipeline_sidecars_anat_boilerplate_fields(tmp_path: Path) -> None:
    """Anat composer surfaces nested fields ``_build_anat_boilerplate`` reads.

    Anchor anchor + brain-mask sidecar + probseg sidecars carry flat
    keys that translate back to nested ``preprocessing`` /
    ``brain_extraction`` / ``tissue_segmentation`` blocks.
    """
    ses = tmp_path / "pmriprep" / "sub-07" / "ses-0001"
    prefix = "sub-07_ses-0001"

    _write(
        ses / "anat" / f"{prefix}_space-T1w_desc-preproc_T1w.json",
        {
            "Description": "Preprocessed T1w",
            "PreprocessingDenoising": True,
            "PreprocessingBiasCorrection": "N4",
            "PreprocessingACPCAligned": True,
        },
    )
    _write(
        ses / "anat" / f"{prefix}_space-T1w_desc-brain_mask.json",
        {
            "BrainExtractionTool": "hdbet",
            "BrainExtractionVersion": "1.0",
            "BrainExtractionVoxelCount": 12345,
            "BrainExtractionVolume": 1.234,
        },
    )
    for tissue in ("WM", "GM", "CSF"):
        _write(
            ses / "anat" / f"{prefix}_space-T1w_label-{tissue}_probseg.json",
            {"TissueType": tissue},
        )

    out = _load_pipeline_sidecars(ses, prefix, "anat")

    assert out["preprocessing"]["denoising"] is True
    assert out["preprocessing"]["bias_correction"] == "N4"
    assert out["preprocessing"]["acpc_aligned"] is True
    assert out["brain_extraction"]["tool"] == "hdbet"
    # BrainExtractionVersion retired: tool version now lives in dataset-level
    # GeneratedBy, not the per-file sidecar, so it is not surfaced here.
    assert "version" not in out["brain_extraction"]
    assert set(out["tissue_segmentation"].keys()) == {"WM", "GM", "CSF"}
    assert out["tissue_segmentation"]["WM"] == {}

    # Boilerplate generator now picks up the real values
    from pmriprep.utils.report_utils import _build_anat_boilerplate

    text = _build_anat_boilerplate(out)
    assert "with prior denoising" in text
    assert "HD-BET" in text  # _BRAIN_EXTRACTION_TEXT["hdbet"]
    assert "Tissue segmentation" in text


def test_anat_boilerplate_unknown_tool_omits_citation() -> None:
    """Unknown / missing brain-extraction tool emits generic sentence + no ref."""
    from pmriprep.utils.report_utils import _build_anat_boilerplate

    # tool key missing entirely
    text = _build_anat_boilerplate({})
    assert "Brain extraction was performed." in text
    assert "@isensee_2019" not in text
    assert "@avants_2011" not in text

    # tool present but unrecognized (e.g., a future tool not yet in the
    # boilerplate dict) — must not silently miscite HD-BET.
    text2 = _build_anat_boilerplate(
        {"brain_extraction": {"tool": "synthstrip"}},
    )
    assert "HD-BET" not in text2
    assert "@isensee_2019" not in text2

    # Known tool still emits its specific sentence + citation.
    text3 = _build_anat_boilerplate(
        {"brain_extraction": {"tool": "hdbet"}},
    )
    assert "HD-BET" in text3
    assert "@isensee_2019" in text3


def test_anat_boilerplate_coregistration_is_conditional() -> None:
    """The secondary-modality coregistration clause only appears when secondary
    structurals were coregistered; the MNI152 transform is always stated."""
    from pmriprep.utils.report_utils import _build_anat_boilerplate

    # T1w-only run: no secondary coregistration -> MNI152 transform only.
    t1w_only = _build_anat_boilerplate({})
    assert "Secondary structural modalities were coregistered" not in t1w_only
    assert "T1-weighted and MNI152 space" in t1w_only

    # Secondary modalities present -> combined clause, single FLIRT citation.
    multi = _build_anat_boilerplate(
        {"coregistration": {"secondary_modalities": True}},
    )
    assert "Secondary structural modalities were coregistered" in multi
    assert "T1-weighted and MNI152 space" in multi
    # Combined into one sentence -> only one FLIRT citation in the reg clause.
    assert multi.count("using FLIRT [@jenkinson_2002]") == 1


def test_anat_boilerplate_denoising_cites_manjon() -> None:
    """The optional ANTs DenoiseImage step names + cites Manjón 2010."""
    from pmriprep.utils.report_utils import _build_anat_boilerplate

    off = _build_anat_boilerplate({})
    assert "DenoiseImage" not in off

    on = _build_anat_boilerplate({"preprocessing": {"denoising": True}})
    assert "ANTs DenoiseImage [@manjon_2010]" in on


def test_load_pipeline_sidecars_anat_tool_normalization(tmp_path: Path) -> None:
    """Brain-extraction tool token is lowercased (defensive against ``HD-BET``)."""
    ses = tmp_path / "pmriprep" / "sub-08" / "ses-0001"
    prefix = "sub-08_ses-0001"
    _write(
        ses / "anat" / f"{prefix}_space-T1w_desc-preproc_T1w.json",
        {"Description": "x"},
    )
    _write(
        ses / "anat" / f"{prefix}_space-T1w_desc-brain_mask.json",
        {"BrainExtractionTool": "HD-BET"},
    )
    out = _load_pipeline_sidecars(ses, prefix, "anat")
    assert out["brain_extraction"]["tool"] == "hdbet"


def test_load_pipeline_sidecars_missing_returns_empty(tmp_path: Path) -> None:
    """Empty dict when the anchor sidecar does not exist."""
    ses = tmp_path / "pmriprep" / "sub-04" / "ses-0001"
    out = _load_pipeline_sidecars(ses, "sub-04_ses-0001", "dsc")
    assert out == {}


def test_load_pipeline_sidecars_unknown_pipeline_returns_empty(tmp_path: Path) -> None:
    """Unknown pipeline name → empty dict (no exception)."""
    out = _load_pipeline_sidecars(tmp_path, "sub-X", "bogus")
    assert out == {}


def test_load_pipeline_sidecars_anchor_without_qc_keys(tmp_path: Path) -> None:
    """Anchor sidecar present but carrying no ``QC*`` keys → no recommendation/warnings."""
    ses = tmp_path / "pmriprep" / "sub-05" / "ses-0001"
    prefix = "sub-05_ses-0001"

    # Anchor with provenance fields only (no PipelineQC or QC* keys)
    _write(
        ses / "perf" / f"{prefix}_desc-dscPreproc_dsc.json",
        {"Description": "x", "SkullStripped": True},
    )

    out = _load_pipeline_sidecars(ses, prefix, "dsc")
    assert out == {}, (
        "Loader should return empty dict when no PipelineQC or QC* keys "
        f"are present; got {out!r}"
    )


def test_load_pipeline_sidecars_malformed_anchor(tmp_path: Path) -> None:
    """Malformed anchor JSON does not raise; loader degrades to empty."""
    ses = tmp_path / "pmriprep" / "sub-06" / "ses-0001"
    prefix = "sub-06_ses-0001"
    anchor = ses / "perf" / f"{prefix}_desc-dcePreproc_dce.json"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("{ not valid json")

    out = _load_pipeline_sidecars(ses, prefix, "dce")
    assert out == {}
