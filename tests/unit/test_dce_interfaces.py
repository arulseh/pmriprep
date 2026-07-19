"""Interface-contract tests for DCE preprocessing."""


def test_vfa_preprocess_inputs_match_registration_responsibility() -> None:
    """VFA preprocessing accepts only inputs used for registration and QC."""
    from pmriprep.interfaces.dce import VFAPreprocessInputSpec

    traits = VFAPreprocessInputSpec().traits()
    assert "vfa_files" in traits
    assert "flip_angles" in traits
    assert "skip_coregistration" in traits
    assert "tr" not in traits
    assert "field_strength" not in traits
