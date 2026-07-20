"""CLI visibility contracts for supported advanced inference modes."""

from pmriprep.cli.run import _build_parser


def test_map_inference_remains_backward_compatible() -> None:
    """Existing explicit MAP invocations remain accepted."""
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


def test_map_inference_is_hidden_from_normal_help() -> None:
    """Normal help advertises recommended inference modes only."""
    help_text = _build_parser().format_help()

    assert "--dsc-inference {nlls,vi,svb}" in help_text
    assert "--dce-inference {nlls,vi}" in help_text
    assert "Laplace MAP" not in help_text
    assert "maximum a posteriori" not in help_text
