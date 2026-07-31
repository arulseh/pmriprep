"""Interface-contract tests for DCE preprocessing."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def test_vfa_preprocess_inputs_match_registration_responsibility() -> None:
    """VFA preprocessing accepts only inputs used for registration and QC."""
    from pmriprep.interfaces.dce import VFAPreprocessInputSpec

    traits = VFAPreprocessInputSpec().traits()
    assert "vfa_files" in traits
    assert "flip_angles" in traits
    assert "target_flip_angle" in traits
    assert "skip_coregistration" in traits
    assert "tr" not in traits
    assert "field_strength" not in traits


def test_dce_config_reports_vfa_t1_usage(tmp_path: Path) -> None:
    """One configuration property controls VFA-branch activation."""
    from pmriprep.config.dce import DCEConfig

    assert DCEConfig().uses_vfa_t1 is True
    assert DCEConfig(no_t1=True).uses_vfa_t1 is False
    assert DCEConfig(t1_map_file=tmp_path / "T1map.nii.gz").uses_vfa_t1 is False


def test_vfa_reference_matches_dce_flip_angle() -> None:
    """Reference selection targets nominal contrast rather than spatial variance."""
    from pmriprep.interfaces.dce import VFAPreprocess

    assert VFAPreprocess._select_reference_index(
        [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        25.0,
    ) == 4


def test_vfa_reference_tie_prefers_higher_angle() -> None:
    """Equidistant candidates use a deterministic higher-angle tie break."""
    from pmriprep.interfaces.dce import VFAPreprocess

    assert VFAPreprocess._select_reference_index([5.0, 15.0, 25.0], 20.0) == 2


def test_vfa_reference_rejects_invalid_angles() -> None:
    """Invalid acquisition metadata cannot silently choose a reference."""
    from pmriprep.interfaces.dce import VFAPreprocess

    with pytest.raises(ValueError, match="finite and positive"):
        VFAPreprocess._select_reference_index([5.0, float("nan")], 25.0)


def test_vfa_reference_requires_multiple_angles() -> None:
    """A single image cannot support VFA T1 fitting."""
    from pmriprep.interfaces.dce import VFAPreprocess

    with pytest.raises(ValueError, match="at least two"):
        VFAPreprocess._select_reference_index([5.0], 25.0)


def test_vfa_reference_requires_distinct_angles() -> None:
    """Repeated acquisitions at one angle cannot identify VFA T1."""
    from pmriprep.interfaces.dce import VFAPreprocess

    with pytest.raises(ValueError, match="distinct flip angles"):
        VFAPreprocess._select_reference_index([15.0, 15.0], 25.0)


def test_vfa_preprocess_rejects_file_angle_count_mismatch(
    tmp_path: Path,
) -> None:
    """Partial VFA sidecar metadata cannot shift angle-to-image association."""
    from pmriprep.interfaces.dce import VFAPreprocess

    files = []
    for index in range(3):
        path = tmp_path / f"vfa_{index}.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), path)
        files.append(str(path))

    interface = VFAPreprocess(
        vfa_files=files,
        flip_angles=[5.0, 15.0],
        target_flip_angle=15.0,
        skip_coregistration=True,
    )
    with pytest.raises(ValueError, match="counts differ"):
        interface.run(cwd=str(tmp_path))


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
