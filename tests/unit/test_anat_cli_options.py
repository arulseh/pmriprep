"""Focused contracts for public anatomical CLI options."""

from pathlib import Path

import nibabel as nib
import numpy as np

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
