"""Characterization tests for anatomical parent-workflow decomposition."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.config.anat import AnatConfig
from pmriprep.workflows.anat.base import (
    _create_acpc_registration_filmstrip,
    _guarded_tissue_segment,
    _merge_fast_inputs,
    _route_fast_input,
    _run_hybrid_fast_chain,
    init_anat_qc_filmstrips_wf,
    init_anat_secondary_modalities_wf,
    init_anat_tissue_segmentation_wf,
)


def test_acpc_filmstrip_compares_subject_and_template_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACPC registration QC supplies both brain boundaries to the plotter."""
    subject = np.zeros((7, 7, 7), dtype=np.float32)
    subject[1:6, 1:6, 1:6] = 1.0
    template = np.zeros_like(subject)
    template[2:7, 1:6, 1:6] = 2.0
    affine = np.eye(4)
    subject_file = tmp_path / "subject_acpc.nii.gz"
    template_file = tmp_path / "mni_brain.nii.gz"
    nib.save(nib.Nifti1Image(subject, affine), subject_file)
    nib.save(nib.Nifti1Image(template, affine), template_file)

    captured: dict[str, object] = {}

    def _capture_plot(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(str(kwargs["output_path"]))

    monkeypatch.setattr("pmriprep.data.get_mni_brain_template", lambda: template_file)
    monkeypatch.setattr(
        "pmriprep.utils.qc_utils.plot_registration_qc_filmstrip",
        _capture_plot,
    )

    _create_acpc_registration_filmstrip(str(subject_file))

    np.testing.assert_array_equal(captured["mask"], subject > 0)
    np.testing.assert_array_equal(captured["moving_mask"], template > 0)


def test_acpc_filmstrip_rejects_grid_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACPC QC cannot compare arrays that do not share the MNI voxel grid."""
    subject_file = tmp_path / "subject_acpc.nii.gz"
    template_file = tmp_path / "mni_brain.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)), subject_file)
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), shifted), template_file)
    monkeypatch.setattr("pmriprep.data.get_mni_brain_template", lambda: template_file)

    with pytest.raises(ValueError, match="must share a voxel grid"):
        _create_acpc_registration_filmstrip(str(subject_file))


def test_tissue_segmentation_workflow_contract() -> None:
    """The extracted workflow retains the four named FAST routing nodes."""
    workflow = init_anat_tissue_segmentation_wf(denoise=False)

    assert {node.name for node in workflow._graph.nodes()} == {
        "inputnode",
        "check_fast_mode",
        "hybrid_fast_chain",
        "merge_fast",
        "tissue_segment",
        "outputnode",
    }
    assert workflow.get_node("hybrid_fast_chain").inputs.denoise is False


def test_secondary_modalities_workflow_all_branches() -> None:
    """All optional modality chains retain their established node names."""
    workflow = init_anat_secondary_modalities_wf(
        config=AnatConfig(),
        has_t1w_postcon=True,
        has_t2w=True,
        has_flair=True,
        has_flair_postcon=True,
    )

    assert {node.name for node in workflow._graph.nodes()} == {
        "inputnode",
        "outputnode",
        *{
            f"{stage}_{suffix}"
            for stage in ("reorient", "bias_correct", "coreg", "mask")
            for suffix in ("postcon", "t2w", "flair", "flair_postcon")
        },
    }


def test_secondary_modalities_workflow_empty_contract() -> None:
    """The public boundary remains connectable when no modality is present."""
    workflow = init_anat_secondary_modalities_wf(
        config=AnatConfig(),
        has_t1w_postcon=False,
        has_t2w=False,
        has_flair=False,
        has_flair_postcon=False,
    )

    assert {node.name for node in workflow._graph.nodes()} == {"inputnode", "outputnode"}


def test_qc_filmstrips_workflow_all_branches() -> None:
    """QC extraction retains every conditional filmstrip node and title."""
    workflow = init_anat_qc_filmstrips_wf(
        subject_id="0001",
        session_id="0002",
        has_t1w_postcon=True,
        has_t2w=True,
        has_flair=True,
        has_flair_postcon=True,
    )

    assert {node.name for node in workflow._graph.nodes()} == {
        "inputnode",
        "outputnode",
        "brain_mask_filmstrip",
        "tissue_seg_filmstrip",
        "acpc_filmstrip",
        "coreg_qc_postcon",
        "coreg_qc_t2w",
        "coreg_qc_flair",
        "coreg_qc_flair_postcon",
    }
    assert workflow.get_node("coreg_qc_postcon").inputs.title == "T1w PostCon to T1w"
    assert workflow.get_node("coreg_qc_flair_postcon").inputs.session_id == "0002"


def test_fast_normal_and_empty_paths_do_not_run_commands() -> None:
    """Normal routing and absent-input guards preserve their passthroughs."""
    assert _route_fast_input("brain.nii.gz", is_postcon_ref=False) == "brain.nii.gz"
    assert _merge_fast_inputs("normal.nii.gz", "hybrid.nii.gz") == "normal.nii.gz"
    assert _merge_fast_inputs(None, "hybrid.nii.gz") == "hybrid.nii.gz"
    assert _guarded_tissue_segment(None) == (None, None, None)
    assert _run_hybrid_fast_chain(is_postcon_ref=False) is None


@pytest.mark.parametrize("fast_t1w", [None, "precon.nii.gz"])
def test_fast_postcon_route_is_owned_by_hybrid_chain(
    monkeypatch: pytest.MonkeyPatch,
    fast_t1w: str | None,
) -> None:
    """Postcontrast-reference routing never sends the normal input to FAST."""
    monkeypatch.setattr(
        "pmriprep.utils.structural_reference._determine_processing_mode_v2",
        lambda _pre, _post: (Path("postcon.nii.gz"), True, fast_t1w),
    )

    assert _route_fast_input(
        "brain.nii.gz",
        t1w_precon="precon.nii.gz",
        t1w_postcon="postcon.nii.gz",
        is_postcon_ref=True,
    ) is None
