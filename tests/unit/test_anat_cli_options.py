"""Focused contracts for public anatomical CLI options."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.config.anat import AnatConfig
from pmriprep.interfaces.anat import BiasCorrection, BrainExtraction
from pmriprep.workflows.anat.base import init_anat_preproc_wf


def _nifti(path: Path) -> Path:
    """Write a small valid NIfTI fixture."""
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), path)
    return path


def test_ants_brain_extraction_uses_templateflow_resources(
    tmp_path: Path, monkeypatch
) -> None:
    """The public ANTs choice builds a complete command without external paths."""
    source = _nifti(tmp_path / "source.nii.gz")
    template = _nifti(tmp_path / "tpl-OASIS30ANTs_res-01_T1w.nii.gz")
    brain_template = _nifti(
        tmp_path / "tpl-OASIS30ANTs_res-01_desc-brain_T1w.nii.gz"
    )
    prior = _nifti(tmp_path / "prior.nii.gz")
    from templateflow import api as tflow

    def _get(*args, **kwargs):
        return prior if kwargs.get("suffix") == "probseg" else [brain_template, template]

    monkeypatch.setattr(tflow, "get", _get)

    interface = BrainExtraction(in_file=str(source), tool="ants")
    assert f"-e {template}" in interface.cmdline
    assert f"-m {prior}" in interface.cmdline
    assert "antsBrainExtraction.sh" in interface.cmdline


def test_brain_extraction_reports_volume_without_uncalibrated_status(
    tmp_path: Path, monkeypatch,
) -> None:
    """Brain-mask size is descriptive and does not imply extraction quality."""
    source = _nifti(tmp_path / "source.nii.gz")
    monkeypatch.chdir(tmp_path)
    interface = BrainExtraction(
        in_file=str(source), tool="hdbet", device="cpu",
    )
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, np.diag([2.0, 2.0, 2.0, 1.0])), interface._gen_maskfilename())

    qc = interface._list_outputs()["qc_metrics"]

    assert qc["brain_volume_mm3"] == pytest.approx(216.0)
    assert qc["brain_volume_ml"] == pytest.approx(0.216)
    assert qc["hdbet_device"] == "cpu"
    assert qc["hdbet_tta_used"] is False
    assert "qc_status" not in qc
    assert "warnings" not in qc


@pytest.mark.parametrize(
    ("device", "sloppy", "expects_disabled"),
    [
        ("cpu", False, True),
        ("cuda", False, False),
        ("mps", False, False),
        ("cuda", True, True),
    ],
)
def test_anatomical_hdbet_tta_policy(
    tmp_path: Path,
    device: str,
    sloppy: bool,
    expects_disabled: bool,
) -> None:
    """The anatomical command follows the shared device and mode policy."""
    source = _nifti(tmp_path / f"source_{device}_{sloppy}.nii.gz")
    command = BrainExtraction(
        in_file=str(source),
        tool="hdbet",
        device=device,
        sloppy=sloppy,
    ).cmdline
    assert ("--disable_tta" in command) is expects_disabled


def test_denoise_adds_ants_denoising_before_n4(tmp_path: Path) -> None:
    """The public denoise choice changes the executed anatomical command."""
    source = _nifti(tmp_path / "source.nii.gz")
    command = BiasCorrection(in_file=str(source), denoise=True).cmdline
    assert "DenoiseImage" in command
    assert command.index("DenoiseImage") < command.index("N4BiasFieldCorrection")


def test_anat_options_reach_workflow_nodes(tmp_path: Path) -> None:
    """Both public anatomical choices propagate into the constructed graph."""
    workflow = init_anat_preproc_wf(
        bids_dir=tmp_path,
        subject_id="01",
        config=AnatConfig(tool="ants", denoise=True),
    )
    nodes = {node.name: node for node in workflow._get_all_nodes()}
    assert nodes["brain_extract"].inputs.tool == "ants"
    assert nodes["bias_correct"].inputs.denoise is True
    assert nodes["merge_qc"].inputs.denoising is True

    sloppy_workflow = init_anat_preproc_wf(
        bids_dir=tmp_path,
        subject_id="01",
        config=AnatConfig(sloppy=True),
    )
    sloppy_nodes = {node.name: node for node in sloppy_workflow._get_all_nodes()}
    assert sloppy_nodes["brain_extract"].inputs.sloppy is True
