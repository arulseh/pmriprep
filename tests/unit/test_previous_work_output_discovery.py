"""Regression tests for prior-run work-directory output discovery."""

from pathlib import Path

from pmriprep.utils.naming import workflow_dirname
from pmriprep.workflows.base import (
    _find_previous_anat_outputs,
    _find_previous_tumor_seg_inputs,
    _find_previous_tumor_seg_outputs,
)


def _write(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    return str(path)


def test_anat_resume_uses_canonical_master_directory(tmp_path: Path) -> None:
    """Anatomical outputs are recovered beneath the current master name."""
    anat = tmp_path / workflow_dirname("X", "Y") / "anat_wf"
    t1w = _write(anat / "bias_correct" / "t1w_biascorr.nii.gz")
    mask = _write(anat / "brain_extract" / "t1w_brain_mask.nii.gz")
    wm = _write(
        anat / "anat_tissue_segmentation_wf" / "tissue_segment" / "t1w_pve_2.nii.gz"
    )
    postcon = _write(
        anat / "anat_secondary_modalities_wf" / "mask_postcon"
        / "t1w_postcon_masked.nii.gz"
    )
    flair = _write(
        anat / "anat_secondary_modalities_wf" / "coreg_flair"
        / "flair_coreg.nii.gz"
    )

    recovered = _find_previous_anat_outputs(tmp_path, "X", "Y")

    assert recovered["t1w_preproc"] == t1w
    assert recovered["brain_mask"] == mask
    assert recovered["wm_probseg"] == wm
    assert recovered["t1w_postcon_coreg"] == postcon
    assert recovered["flair_coreg"] == flair


def test_tumor_inputs_resume_uses_canonical_master_directory(tmp_path: Path) -> None:
    """Tumor-segmentation inputs are recovered from the prior anat workflow."""
    anat = tmp_path / workflow_dirname("X", None) / "anat_wf"
    masked = _write(anat / "tumor_seg_mask_t1w" / "t1w_masked.nii.gz")
    wm = _write(
        anat / "anat_tissue_segmentation_wf" / "tissue_segment" / "t1w_pve_2.nii.gz"
    )

    recovered = _find_previous_tumor_seg_inputs(tmp_path, "X", None)

    assert recovered["tumor_seg_t1w_pre"] == masked
    assert recovered["wm_probseg"] == wm


def test_tumor_outputs_resume_uses_canonical_master_directory(tmp_path: Path) -> None:
    """Tumor masks are recovered beneath the current master workflow name."""
    tumor = tmp_path / workflow_dirname("X", "Y") / "tumor_seg_wf"
    nawm = _write(tumor / "extract_masks" / "subject_nawm.nii.gz")
    whole = _write(tumor / "extract_masks" / "subject_wholetumor_mask.nii.gz")

    recovered = _find_previous_tumor_seg_outputs(tmp_path, "X", "Y")

    assert recovered["nawm_mask"] == nawm
    assert recovered["whole_tumor_mask"] == whole
