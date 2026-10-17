"""Smoke + sidecar-correctness tests for ``init_anat_datasinks_wf``."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def minimal_bids(tmp_path: Path) -> Path:
    """Create a minimal BIDS root with one T1w source NIfTI."""
    bids = tmp_path / "bids"
    raw = bids / "sub-01" / "ses-0001" / "anat"
    raw.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
        raw / "sub-01_ses-0001_T1w.nii.gz",
    )
    (raw / "sub-01_ses-0001_T1w.json").write_text(
        json.dumps({"RepetitionTime": 2.0})
    )
    return bids


def _make_nifti(path: Path) -> Path:
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
        path,
    )
    return path


def test_anat_datasinks_wf_minimal(tmp_path: Path, minimal_bids: Path) -> None:
    """T1w-only run produces preproc + brain mask + acpc xfm with sidecars."""
    from pmriprep.workflows.anat.datasinks import init_anat_datasinks_wf

    out_root = tmp_path / "out"
    work = tmp_path / "work"

    src_t1w = minimal_bids / "sub-01" / "ses-0001" / "anat" / "sub-01_ses-0001_T1w.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.zeros((4, 4, 4), dtype=np.float32),
            np.diag([2.0, 2.0, 2.0, 1.0]),
        ),
        src_t1w,
    )
    deriv_t1w = _make_nifti(tmp_path / "t1w_preproc.nii.gz")
    deriv_mask = _make_nifti(tmp_path / "brain_mask.nii.gz")
    fake_xfm = tmp_path / "acpc.h5"
    fake_xfm.write_text("# fake transform\n")

    wf = init_anat_datasinks_wf(
        output_dir=str(out_root),
        bids_dir=str(minimal_bids),
        subject_id="01",
        session_id="0001",
        has_t1w_pre=True,
        has_t1w_postcon=False,
        has_t2w=False,
        has_flair=False,
        has_flair_postcon=False,
        has_tissue_seg=False,
        has_subtraction=False,
    )
    wf.base_dir = str(work)

    inputs = wf.inputs.inputnode
    inputs.source_t1w = str(src_t1w)
    inputs.t1w_preproc = str(deriv_t1w)
    inputs.brain_mask = str(deriv_mask)
    inputs.acpc_xfm = str(fake_xfm)
    inputs.qc_metrics = {
        "resolution_tier": "excellent",
        "brain_extraction": {"tool": "hdbet", "brain_volume_ml": 42.0},
    }

    wf.run()

    pmri = out_root / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    t1w_out = pmri / "sub-01_ses-0001_space-T1w_desc-preproc_T1w.nii.gz"
    t1w_json = t1w_out.with_suffix("").with_suffix(".json")
    mask_out = pmri / "sub-01_ses-0001_space-T1w_desc-brain_mask.nii.gz"
    mask_json = mask_out.with_suffix("").with_suffix(".json")

    assert t1w_out.exists(), f"Missing: {t1w_out}"
    assert t1w_json.exists(), f"Missing sidecar: {t1w_json}"
    assert mask_out.exists()
    assert mask_json.exists()

    # Required sidecar fields
    sidecar = json.loads(t1w_json.read_text())
    assert isinstance(sidecar["Sources"], list) and sidecar["Sources"]
    assert sidecar["Sources"][0].startswith("bids:raw:")
    assert sidecar["SpatialReference"].startswith("bids::sub-")
    assert "Resolution" not in sidecar
    assert isinstance(sidecar["VoxelSize"], list) and len(sidecar["VoxelSize"]) == 3
    assert sidecar["ProcessingSteps"] == [
        "Reorientation to RAS and robust FOV cropping",
        "N4 bias field correction",
        "Conditional conformation of submillimeter input voxels to 1 mm",
        "Brain extraction",
    ]
    assert sidecar["VoxelSize"] == [1.0, 1.0, 1.0]
    assert sidecar["OriginalResolution"]["VoxelDimensions"] == [2.0, 2.0, 2.0]
    warnings = sidecar.get("QCWarnings", [])
    assert any("Tissue segmentation" in w for w in warnings), warnings
    assert not any("FLAIR not available" in w for w in warnings), warnings
    assert not any("T2w not available" in w for w in warnings), warnings

    mask_sidecar = json.loads(mask_json.read_text())
    assert "SkullStripped" not in mask_sidecar
    assert mask_sidecar["ProcessingSteps"][-1] == "Brain-mask estimation"
    assert mask_sidecar["BrainExtractionTool"] == "hdbet"
    assert mask_sidecar["BrainExtractionVolume"] == 42.0

    # dataset_description.json written
    dd = out_root / "pmriprep" / "dataset_description.json"
    assert dd.exists()


def test_postcon_only_secondary_anat_references_postcon_t1w(tmp_path: Path) -> None:
    """Postcontrast-reference outputs and secondary anatomy share the actual reference."""
    from pmriprep.workflows.anat.datasinks import init_anat_datasinks_wf

    bids = tmp_path / "bids"
    raw = bids / "sub-01" / "ses-0001" / "anat"
    raw.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src_postcon = _make_nifti(raw / "sub-01_ses-0001_acq-gre_ce-GAD_T1w.nii.gz")
    src_flair = _make_nifti(raw / "sub-01_ses-0001_acq-3d_ce-GAD_FLAIR.nii.gz")

    out_root = tmp_path / "out"
    fake_xfm = tmp_path / "acpc.h5"
    fake_xfm.write_text("ok\n")
    flair_xfm = tmp_path / "flair_to_t1w.h5"
    flair_xfm.write_text("ok\n")
    wf = init_anat_datasinks_wf(
        output_dir=str(out_root), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        has_t1w_pre=False, has_t1w_postcon=True,
        has_t2w=False, has_flair=True, has_flair_postcon=False,
        has_tissue_seg=False, has_subtraction=False,
        is_postcon_ref=True,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_t1w_postcon = str(src_postcon)
    inp.source_flair = str(src_flair)
    inp.t1w_preproc = str(_make_nifti(tmp_path / "t1w.nii.gz"))
    inp.brain_mask = str(_make_nifti(tmp_path / "mask.nii.gz"))
    inp.flair_coreg = str(_make_nifti(tmp_path / "flair.nii.gz"))
    inp.acpc_xfm = str(fake_xfm)
    inp.flair_to_t1w_xfm = str(flair_xfm)
    wf.run()

    sidecar_path = (
        out_root / "pmriprep" / "sub-01" / "ses-0001" / "anat"
        / "sub-01_ses-0001_acq-3d_ce-GAD_space-T1w_desc-preproc_FLAIR.json"
    )
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["SpatialReference"] == (
        "bids::sub-01/ses-0001/anat/"
        "sub-01_ses-0001_acq-gre_ce-GAD_space-T1w_desc-preproc_T1w.nii.gz"
    )

    anat = sidecar_path.parent
    t1w = anat / "sub-01_ses-0001_acq-gre_ce-GAD_space-T1w_desc-preproc_T1w.nii.gz"
    mask = anat / "sub-01_ses-0001_acq-gre_ce-GAD_space-T1w_desc-brain_mask.nii.gz"
    xfm = anat / "sub-01_ses-0001_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5"
    assert t1w.exists()
    assert mask.exists()
    assert xfm.read_bytes() == fake_xfm.read_bytes()
    t1w_meta = json.loads(t1w.with_suffix("").with_suffix(".json").read_text())
    assert t1w_meta["SpatialReference"] == sidecar["SpatialReference"]
    assert t1w_meta["Sources"] == [
        "bids:raw:sub-01/ses-0001/anat/sub-01_ses-0001_acq-gre_ce-GAD_T1w.nii.gz"
    ]


def test_multi_modality_kitchen_sink(tmp_path: Path, minimal_bids: Path) -> None:
    """All optional modalities present: every writer fires + sidecars correct."""
    from pmriprep.workflows.anat.datasinks import init_anat_datasinks_wf

    raw = minimal_bids / "sub-01" / "ses-0001" / "anat"
    src_t1w = raw / "sub-01_ses-0001_T1w.nii.gz"
    src_postcon = raw / "sub-01_ses-0001_ce-GAD_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src_postcon)
    (raw / "sub-01_ses-0001_ce-GAD_T1w.json").write_text(
        json.dumps({"ContrastBolusAgent": "Gadovist"})
    )
    src_t2w = raw / "sub-01_ses-0001_T2w.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src_t2w)
    (raw / "sub-01_ses-0001_T2w.json").write_text("{}")
    src_flair = raw / "sub-01_ses-0001_FLAIR.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src_flair)
    (raw / "sub-01_ses-0001_FLAIR.json").write_text("{}")

    out_root = tmp_path / "out"
    def deriv(name: str) -> Path:
        return _make_nifti(tmp_path / name)
    def fake_xfm(name: str) -> None:
        (tmp_path / name).write_text("ok\n")

    wf = init_anat_datasinks_wf(
        output_dir=str(out_root), bids_dir=str(minimal_bids),
        subject_id="01", session_id="0001",
        has_t1w_pre=True, has_t1w_postcon=True,
        has_t2w=True, has_flair=True, has_flair_postcon=False,
        has_tissue_seg=True, has_subtraction=True,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_t1w = str(src_t1w)
    inp.source_t1w_postcon = str(src_postcon)
    inp.source_t2w = str(src_t2w)
    inp.source_flair = str(src_flair)
    inp.t1w_preproc = str(deriv("t1w.nii.gz"))
    inp.brain_mask = str(deriv("mask.nii.gz"))
    inp.csf_probseg = str(deriv("csf.nii.gz"))
    inp.gm_probseg = str(deriv("gm.nii.gz"))
    inp.wm_probseg = str(deriv("wm.nii.gz"))
    inp.t1w_postcon_coreg = str(deriv("postcon.nii.gz"))
    inp.t2w_coreg = str(deriv("t2w.nii.gz"))
    inp.flair_coreg = str(deriv("flair.nii.gz"))
    inp.t1w_subtraction = str(deriv("subtract.nii.gz"))
    fake_xfm("acpc.h5")
    fake_xfm("postcon_to_t1w.h5")
    fake_xfm("t2w_to_t1w.h5")
    fake_xfm("flair_to_t1w.h5")
    inp.acpc_xfm = str(tmp_path / "acpc.h5")
    inp.t1w_postcon_to_t1w_xfm = str(tmp_path / "postcon_to_t1w.h5")
    inp.t2w_to_t1w_xfm = str(tmp_path / "t2w_to_t1w.h5")
    inp.flair_to_t1w_xfm = str(tmp_path / "flair_to_t1w.h5")
    inp.qc_metrics = {"ok": True}
    wf.run()

    anat_dir = out_root / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = [
        "sub-01_ses-0001_space-T1w_desc-preproc_T1w.nii.gz",
        "sub-01_ses-0001_space-T1w_desc-brain_mask.nii.gz",
        "sub-01_ses-0001_space-T1w_label-CSF_probseg.nii.gz",
        "sub-01_ses-0001_space-T1w_label-GM_probseg.nii.gz",
        "sub-01_ses-0001_space-T1w_label-WM_probseg.nii.gz",
        "sub-01_ses-0001_ce-GAD_space-T1w_desc-preproc_T1w.nii.gz",
        "sub-01_ses-0001_space-T1w_desc-preproc_T2w.nii.gz",
        "sub-01_ses-0001_space-T1w_desc-preproc_FLAIR.nii.gz",
        "sub-01_ses-0001_space-T1w_desc-subtraction_T1w.nii.gz",
        "sub-01_ses-0001_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
        "sub-01_ses-0001_from-T1wCE_to-T1w_mode-image_xfm.h5",
        "sub-01_ses-0001_from-T2w_to-T1w_mode-image_xfm.h5",
        "sub-01_ses-0001_from-FLAIR_to-T1w_mode-image_xfm.h5",
    ]
    for name in expected:
        assert (anat_dir / name).exists(), f"Missing: {name}"
        if name.endswith(".nii.gz"):
            sidecar = anat_dir / name.replace(".nii.gz", ".json")
            body = json.loads(sidecar.read_text())
            assert isinstance(body["Sources"], list) and body["Sources"], name
            assert all(s.startswith("bids:raw:") for s in body["Sources"]), name
            assert body["SpatialReference"].startswith("bids::"), name
            assert "Resolution" not in body, name
            assert body["VoxelSize"] == [1.0, 1.0, 1.0], name

    wm_sidecar = json.loads(
        (anat_dir / "sub-01_ses-0001_space-T1w_label-WM_probseg.json").read_text()
    )
    assert wm_sidecar["TissueType"] == "WM"
    assert not any(
        key in wm_sidecar
        for key in (
            "TissueSegmentationMeanProbability",
            "TissueSegmentationVoxelCount",
            "TissueSegmentationVolume",
        )
    )

    # Postcon T1w sidecar must carry the contrast-agent block
    postcon_sidecar = json.loads(
        (anat_dir / "sub-01_ses-0001_ce-GAD_space-T1w_desc-preproc_T1w.json").read_text()
    )
    assert "PerfusionProcessing" in postcon_sidecar
    assert "ContrastAgent" in postcon_sidecar["PerfusionProcessing"]
    assert "Rigid registration to the anatomical reference" in postcon_sidecar[
        "ProcessingSteps"
    ]
    assert postcon_sidecar["ProcessingSteps"][0] == (
        "Reorientation to RAS and robust FOV cropping"
    )

    subtraction_sidecar = json.loads(
        (anat_dir / "sub-01_ses-0001_space-T1w_desc-subtraction_T1w.json").read_text()
    )
    assert subtraction_sidecar["ProcessingSteps"] == [
        "Rigid registration of the post-contrast T1w image to the pre-contrast reference",
        "Post-contrast minus pre-contrast signal subtraction within the brain mask",
    ]
    assert subtraction_sidecar["Units"] == "arbitrary"
    assert subtraction_sidecar["Estimator"] == (
        "registered_postcontrast_signal_minus_precontrast_signal"
    )
    assert subtraction_sidecar["IntensityNormalizationApplied"] is False
    assert subtraction_sidecar["Quantitative"] is False
    assert "not suitable for cross-subject" in subtraction_sidecar["Interpretation"]
