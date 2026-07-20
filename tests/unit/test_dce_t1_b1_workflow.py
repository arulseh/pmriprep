"""Construction contracts for DCE T1/B1 preparation."""

from pathlib import Path

from pmriprep.config.dce import DCEConfig
from pmriprep.workflows.dce.base import init_dce_t1_b1_wf


def _names(config: DCEConfig) -> set[str]:
    workflow = init_dce_t1_b1_wf(
        config=config, subject_id="0001", session_id="0001",
    )
    return set(workflow.list_node_names())


def test_t1_b1_workflow_vfa_mode() -> None:
    """VFA mode constructs the complete B1-corrected T1 chain."""
    assert _names(DCEConfig()) == {
        "inputnode", "outputnode", "b1_mapping", "vfa_reorient",
        "vfa_preprocess", "transform_b1_to_vfa", "t1_mapping",
    }


def test_t1_b1_workflow_external_t1_mode() -> None:
    """External T1 retains B1 preparation without constructing VFA fitting."""
    config = DCEConfig(t1_map_file=Path("external_t1map.nii.gz"))
    assert _names(config) == {"inputnode", "outputnode", "b1_mapping"}


def test_t1_b1_workflow_no_t1_mode() -> None:
    """No-T1 mode preserves an explicit empty NiPreps-style contract."""
    assert _names(DCEConfig(no_t1=True)) == {"inputnode", "outputnode"}
