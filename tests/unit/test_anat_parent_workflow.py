"""Characterization tests for anatomical parent-workflow decomposition."""

from pathlib import Path

import pytest

from pmriprep.config.anat import AnatConfig
from pmriprep.workflows.anat.base import (
    _guarded_tissue_segment,
    _merge_fast_inputs,
    _route_fast_input,
    _run_hybrid_fast_chain,
    init_anat_secondary_modalities_wf,
    init_anat_tissue_segmentation_wf,
)


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
