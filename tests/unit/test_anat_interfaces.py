"""Interface-contract tests for anatomical preprocessing."""


def test_bias_correction_outputs_match_command_products() -> None:
    """Bias correction reports only the requested corrected image."""
    from pmriprep.interfaces.anat import BiasCorrectionOutputSpec

    outputs = BiasCorrectionOutputSpec().traits()
    assert "out_file" in outputs
    assert "bias_field" not in outputs
