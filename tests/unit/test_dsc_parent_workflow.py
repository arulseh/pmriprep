"""Focused contracts for DSC parent-workflow decomposition."""

from pmriprep.workflows.dsc.base import init_dsc_underlays_wf


def test_dsc_underlays_workflow_contract() -> None:
    """Requested underlays retain independent transform nodes and outputs."""
    underlays = [
        "t1w_precon", "t1w_postcon", "t1w_subtraction",
        "flair_precon", "flair_postcon", "t2w",
    ]
    workflow = init_dsc_underlays_wf(underlays=underlays)
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "transform_t1_to_dsc",
        "transform_t1_postcon_to_dsc", "transform_t1_subtraction_to_dsc",
        "transform_flair_to_dsc", "transform_flair_postcon_to_dsc",
        "transform_t2w_to_dsc",
    }
    assert set(workflow.get_node("outputnode").interface._fields) == {
        "t1w_in_dsc", "t1w_postcon_in_dsc", "t1w_subtraction_in_dsc",
        "flair_in_dsc", "flair_postcon_in_dsc", "t2w_in_dsc",
    }


def test_dsc_underlays_workflow_omits_unrequested_transforms() -> None:
    """An empty request constructs no transform nodes."""
    workflow = init_dsc_underlays_wf(underlays=[])
    assert workflow.list_node_names() == []
