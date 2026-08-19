"""CLI visibility contracts for supported advanced inference modes."""

from pmriprep.cli.run import _build_parser


def test_internal_inference_modes_remain_backward_compatible() -> None:
    """Existing explicit internal-mode invocations remain accepted."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "/tmp/bids",
            "/tmp/out",
            "participant",
            "--dsc-inference",
            "map",
            "--dce-inference",
            "map",
        ]
    )

    assert args.dsc_inference == "map"
    assert args.dce_inference == "map"

    args = parser.parse_args(
        [
            "/tmp/bids",
            "/tmp/out",
            "participant",
            "--dsc-inference",
            "nlls",
        ]
    )
    assert args.dsc_inference == "nlls"


def test_map_inference_is_hidden_from_normal_help() -> None:
    """Normal help advertises recommended inference modes only."""
    parser = _build_parser()
    help_text = parser.format_help()
    dsc_help = parser._option_string_actions["--dsc-inference"].help

    assert "--dsc-inference {vi,svb}" in help_text
    assert "--dce-inference {nlls,vi}" in help_text
    assert dsc_help is not None
    assert "nonlinear least squares" not in dsc_help
    assert "Laplace MAP" not in help_text
    assert "maximum a posteriori" not in help_text


def test_help_states_external_t1_alignment_contract() -> None:
    """External T1 help exposes the input-space requirement."""
    help_text = _build_parser().format_help()
    normalized = " ".join(help_text.split())

    assert "--dce-t1-map FILE" in help_text
    assert "aligned in world coordinates with DCE" in normalized
    assert "alignment is reported but not corrected" in normalized


def test_help_avoids_stale_or_overstated_method_claims() -> None:
    """Public help remains concise and implementation-neutral."""
    help_text = _build_parser().format_help()

    assert "not yet in the stable BIDS specification" not in help_text
    assert "Gadovist 5.0" not in help_text
    assert "Dynamic Contrast Enhanced" not in help_text
