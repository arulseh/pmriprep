"""Tests for compact DSC leakage-correction QC accounting."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.workflows.dsc._helpers import _merge_qc_metrics


def _write_nifti(path: Path, data: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
    return str(path)


def test_merge_qc_records_descriptive_bsw_effect(tmp_path: Path) -> None:
    """NAWM effect, support, and negative-CBV frequency reach preload QC."""
    raw = _write_nifti(tmp_path / "raw.nii.gz", np.array([[[2.0, 4.0, 8.0]]]))
    corrected = _write_nifti(
        tmp_path / "corrected.nii.gz", np.array([[[1.0, 3.0, 8.0]]])
    )
    nawm = _write_nifti(tmp_path / "nawm.nii.gz", np.array([[[1, 1, 0]]]))

    merged = _merge_qc_metrics(
        motion_qc=None,
        preload_qc={
            "k2_method": "bsw",
            "preload_correction_applied": True,
            "k2_reference_available": True,
            "k2_reference_selected_fraction": 0.75,
        },
        aif_qc=None,
        deconv_qc={"cbv_auc_negative_fraction": 0.02},
        registration_qc=None,
        brain_extraction_qc=None,
        normalization_factor=1.0,
        auc_raw_file=raw,
        auc_corr_file=corrected,
        nawm_mask_file=nawm,
    )

    leakage = merged["preload"]
    assert isinstance(leakage, dict)
    assert leakage["k2_reference_available"] is True
    assert leakage["k2_reference_selected_fraction"] == pytest.approx(0.75)
    assert leakage["nawm_auc_raw_median"] == pytest.approx(3.0)
    assert leakage["nawm_auc_corrected_median"] == pytest.approx(2.0)
    assert "negative_corrected_cbv_fraction" not in leakage


def test_merge_qc_omits_nawm_effect_without_mask() -> None:
    """Whole brain is not substituted when an NAWM mask is unavailable."""
    merged = _merge_qc_metrics(
        motion_qc=None,
        preload_qc={"k2_method": "bsw"},
        aif_qc=None,
        deconv_qc=None,
        registration_qc=None,
        brain_extraction_qc=None,
        normalization_factor=1.0,
    )

    leakage = merged["preload"]
    assert isinstance(leakage, dict)
    assert "nawm_auc_raw_median" not in leakage
    assert "nawm_auc_corrected_median" not in leakage


def test_merge_qc_records_generated_nawm_exclusion_reason() -> None:
    """DSC QC explains why auto-NAWM normalization was withheld."""
    merged = _merge_qc_metrics(
        motion_qc=None,
        preload_qc=None,
        aif_qc=None,
        deconv_qc=None,
        registration_qc=None,
        brain_extraction_qc=None,
        normalization_factor=None,
        nawm_exclusion_reason="insufficient_voxels_after_erosion_relaxation",
    )

    assert merged["nawm_reference"] == {
        "eligible_for_quantification": False,
        "exclusion_reason": "insufficient_voxels_after_erosion_relaxation",
    }


def test_merge_qc_rejects_mismatched_nawm_geometry(tmp_path: Path) -> None:
    """A sidecar statistic is never computed across mismatched voxel grids."""
    raw = _write_nifti(tmp_path / "raw.nii.gz", np.ones((2, 2, 2)))
    corrected = _write_nifti(tmp_path / "corrected.nii.gz", np.ones((2, 2, 2)))
    nawm = _write_nifti(tmp_path / "nawm.nii.gz", np.ones((2, 2, 1)))

    with pytest.raises(ValueError, match="AUC.*NAWM mask geometry"):
        _merge_qc_metrics(
            motion_qc=None,
            preload_qc={"k2_method": "bsw"},
            aif_qc=None,
            deconv_qc=None,
            registration_qc=None,
            brain_extraction_qc=None,
            normalization_factor=1.0,
            auc_raw_file=raw,
            auc_corr_file=corrected,
            nawm_mask_file=nawm,
        )
