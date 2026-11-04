"""CLI configuration contracts for independent DCE T1 and B1 controls."""

from pmriprep.cli.run import _build_parser
from pmriprep.config.dce import DCEConfig
from pmriprep.config.loader import merge_cli_config
from pmriprep.workflows.dce.base import init_dce_t1_b1_wf


def _workflow_names(config: DCEConfig) -> set[str]:
    """Return node names for the DCE T1/B1 preparation workflow."""
    workflow = init_dce_t1_b1_wf(
        config=config,
        subject_id="0001",
        session_id="0001",
    )
    return set(workflow.list_node_names())


def test_cli_no_t1_preserves_default_b1_processing() -> None:
    """``--dce-no-t1`` skips VFA fitting without disabling default B1."""
    args = _build_parser().parse_args([
        "/tmp/bids",
        "/tmp/out",
        "participant",
        "--dce-no-t1",
    ])
    merged = merge_cli_config(vars(args), {})
    config = DCEConfig(**merged["dce"])

    assert config.no_t1 is True
    assert config.b1_method == "auto"
    assert _workflow_names(config) == {
        "inputnode",
        "outputnode",
        "b1_mapping",
    }


def test_cli_no_t1_with_explicit_b1_opt_out() -> None:
    """Only the independent B1 option disables B1 in no-T1 mode."""
    args = _build_parser().parse_args([
        "/tmp/bids",
        "/tmp/out",
        "participant",
        "--dce-no-t1",
        "--dce-b1-method",
        "none",
    ])
    merged = merge_cli_config(vars(args), {})
    config = DCEConfig(**merged["dce"])

    assert _workflow_names(config) == {"inputnode", "outputnode"}


def test_yaml_no_t1_preserves_configured_b1_processing() -> None:
    """YAML-shaped configuration follows the same independent controls."""
    merged = merge_cli_config(
        {},
        {"dce": {"no_t1": True, "b1_method": "precomputed", "t1_default_ms": 850}},
    )
    config = DCEConfig(**merged["dce"])

    assert config.t1_default_ms == 850.0
    assert _workflow_names(config) == {
        "inputnode",
        "outputnode",
        "b1_mapping",
    }
