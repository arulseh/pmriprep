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


def test_dce_preprocess_exposes_ensemble_detection_inputs() -> None:
    """DCE baseline detection exposes timing and mask inputs."""
    from pmriprep.interfaces.dce import DCEPreprocessInputSpec

    traits = DCEPreprocessInputSpec().traits()
    assert "baseline_frames" in traits
    assert "brain_mask" in traits
    assert "volume_tr" in traits
    assert "contrast_threshold" not in traits


def test_dce_aif_inputs_match_cascade_controls() -> None:
    """DCE AIF inputs omit an unsupported global criteria selector."""
    from pmriprep.interfaces.dce_aif import DCEAIFDetectionInputSpec

    traits = DCEAIFDetectionInputSpec().traits()
    assert "aif_method" in traits
    assert "n_candidates" in traits
    assert "atlas_mask_files" in traits
    assert "atlas_mask_file" not in traits
    assert "criteria" not in traits
