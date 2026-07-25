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


def test_vfa_motion_warning_requires_concordant_evidence() -> None:
    """Optimizer motion alone cannot trigger an inter-VFA warning."""
    from pmriprep.interfaces.dce import VFAPreprocess

    common = {
        "translation_mm": 1.0,
        "rotation_deg": 0.4,
        "min_voxel_size_mm": 2.0,
        "gradient_ncc_delta": 0.02,
        "foreground_dice_delta": 0.0,
    }
    assert VFAPreprocess._motion_warning_state(
        **common,
        spgr_nrmse_fractional_change=-0.03,
    ) == "PASS"
    assert VFAPreprocess._motion_warning_state(
        **common,
        spgr_nrmse_fractional_change=0.08,
    ) == "WARN"


def test_vfa_motion_warning_rejects_small_or_anatomically_worse_candidates() -> None:
    """Warnings require a material transform and stable anatomical overlap."""
    from pmriprep.interfaces.dce import VFAPreprocess

    assert VFAPreprocess._motion_warning_state(
        translation_mm=0.2,
        rotation_deg=0.1,
        min_voxel_size_mm=2.0,
        gradient_ncc_delta=0.03,
        foreground_dice_delta=0.0,
        spgr_nrmse_fractional_change=0.10,
    ) == "PASS"
    assert VFAPreprocess._motion_warning_state(
        translation_mm=1.0,
        rotation_deg=0.4,
        min_voxel_size_mm=2.0,
        gradient_ncc_delta=0.03,
        foreground_dice_delta=-0.02,
        spgr_nrmse_fractional_change=0.10,
    ) == "PASS"


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
