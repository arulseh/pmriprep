"""Regression tests for DSC registration QC mask handling."""

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from pmriprep.interfaces.dsc import DSCRegistration


def _save_volume(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    """Write a NIfTI test volume."""
    nib.save(nib.Nifti1Image(data, affine), path)


def test_dsc_registration_qc_transforms_anatomical_mask(
    tmp_path: Path,
) -> None:
    """Final DSC QC uses a transformed mask rather than the anatomical grid."""
    dsc_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    anat_affine = np.eye(4)
    rng = np.random.default_rng(42)
    dsc_data = rng.uniform(50.0, 200.0, size=(8, 8, 4)).astype(np.float32)
    anat_mask = np.ones((16, 16, 8), dtype=np.uint8)
    transformed_mask_data = np.ones((8, 8, 4), dtype=np.uint8)
    native_mask_data = np.zeros((8, 8, 4), dtype=np.uint8)
    native_mask_data[2:6, 2:6, 1:3] = 1

    registered = tmp_path / "registered_t1w.nii.gz"
    dsc_mean = tmp_path / "dsc_mean.nii.gz"
    anatomical_mask = tmp_path / "brain_mask_t1w.nii.gz"
    transformed_mask = tmp_path / "brain_mask_dsc.nii.gz"
    native_mask = tmp_path / "native_brain_mask_dsc.nii.gz"
    matrix = tmp_path / "registration.mat"
    _save_volume(registered, dsc_data, dsc_affine)
    _save_volume(dsc_mean, dsc_data, dsc_affine)
    _save_volume(anatomical_mask, anat_mask, anat_affine)
    _save_volume(transformed_mask, transformed_mask_data, dsc_affine)
    _save_volume(native_mask, native_mask_data, dsc_affine)
    np.savetxt(matrix, np.eye(4))

    interface = DSCRegistration()
    interface.inputs.brain_mask = str(anatomical_mask)
    interface.inputs.native_brain_mask = str(native_mask)
    interface.inputs.out_matrix = str(matrix)

    with patch(
        "pmriprep.utils.registration.transform_mask_to_grid",
        return_value=str(transformed_mask),
    ) as transform:
        metrics = interface._compute_qc_metrics(
            str(registered), str(dsc_mean)
        )

    transform.assert_called_once_with(
        str(anatomical_mask),
        str(dsc_mean),
        str(matrix),
        out_dir=tmp_path,
    )
    assert metrics["brain_mask_used"] is True
    assert metrics["dice"] > 0.8
    assert metrics["qc_status"] == "PASS"
    assert np.isfinite(metrics["nmi"])
    assert np.isfinite(metrics["correlation"])
    assert "dice_computation_failed" not in metrics
    assert metrics["native_perfusion_mask_metrics_role"] == "descriptive_not_gated"
    assert metrics["native_perfusion_mask_used"] is True
    assert metrics["native_perfusion_mask_dice"] < 0.5
    assert np.isfinite(metrics["native_perfusion_mask_correlation"])
    assert np.isfinite(metrics["native_perfusion_mask_nmi"])


def test_dsc_registration_qc_degrades_to_unmasked_metrics(
    tmp_path: Path,
) -> None:
    """An unavailable transformed mask does not emit a false Dice failure."""
    affine = np.eye(4)
    rng = np.random.default_rng(7)
    data = rng.uniform(10.0, 100.0, size=(6, 6, 4)).astype(np.float32)
    registered = tmp_path / "registered_t1w.nii.gz"
    dsc_mean = tmp_path / "dsc_mean.nii.gz"
    anatomical_mask = tmp_path / "brain_mask_t1w.nii.gz"
    matrix = tmp_path / "registration.mat"
    _save_volume(registered, data, affine)
    _save_volume(dsc_mean, data, affine)
    _save_volume(anatomical_mask, np.ones((12, 12, 8), dtype=np.uint8), affine)
    np.savetxt(matrix, np.eye(4))

    interface = DSCRegistration()
    interface.inputs.brain_mask = str(anatomical_mask)
    interface.inputs.out_matrix = str(matrix)

    with patch(
        "pmriprep.utils.registration.transform_mask_to_grid",
        return_value="",
    ):
        metrics = interface._compute_qc_metrics(
            str(registered), str(dsc_mean)
        )

    assert metrics["brain_mask_used"] is False
    assert metrics["qc_status"] == "UNKNOWN"
    assert "dice" not in metrics
    assert "dice_computation_failed" not in metrics
    assert np.isfinite(metrics["nmi"])
    assert np.isfinite(metrics["correlation"])


def test_dsc_registration_qc_rejects_inadequate_overlap(
    tmp_path: Path,
) -> None:
    """Quantitative transfer is rejected when final mask overlap is below 0.90."""
    affine = np.eye(4)
    data = np.random.default_rng(11).uniform(
        10.0, 100.0, size=(6, 6, 4)
    ).astype(np.float32)
    mask = np.ones((6, 6, 4), dtype=np.uint8)
    registered = tmp_path / "registered_t1w.nii.gz"
    dsc_mean = tmp_path / "dsc_mean.nii.gz"
    anatomical_mask = tmp_path / "brain_mask_t1w.nii.gz"
    transformed_mask = tmp_path / "brain_mask_dsc.nii.gz"
    matrix = tmp_path / "registration.mat"
    _save_volume(registered, data, affine)
    _save_volume(dsc_mean, data, affine)
    _save_volume(anatomical_mask, mask, affine)
    _save_volume(transformed_mask, mask, affine)
    np.savetxt(matrix, np.eye(4))

    interface = DSCRegistration()
    interface.inputs.brain_mask = str(anatomical_mask)
    interface.inputs.out_matrix = str(matrix)

    with (
        patch(
            "pmriprep.utils.registration.transform_mask_to_grid",
            return_value=str(transformed_mask),
        ),
        patch("pmriprep.utils.registration.calculate_dice", return_value=0.89),
    ):
        metrics = interface._compute_qc_metrics(
            str(registered), str(dsc_mean)
        )

    assert metrics["dice"] == 0.89
    assert metrics["qc_status"] == "POOR"
    assert metrics["qc_status_threshold"] == 0.90


def test_dsc_registration_qc_reports_dice_failure(
    tmp_path: Path,
    caplog,
) -> None:
    """A Dice-computation error is visible and fails quantitative transfer closed."""
    affine = np.eye(4)
    data = np.random.default_rng(23).uniform(
        10.0, 100.0, size=(6, 6, 4)
    ).astype(np.float32)
    registered = tmp_path / "registered_t1w.nii.gz"
    dsc_mean = tmp_path / "dsc_mean.nii.gz"
    anatomical_mask = tmp_path / "brain_mask_t1w.nii.gz"
    transformed_mask = tmp_path / "brain_mask_dsc.nii.gz"
    matrix = tmp_path / "registration.mat"
    _save_volume(registered, data, affine)
    _save_volume(dsc_mean, data, affine)
    _save_volume(anatomical_mask, data, affine)
    _save_volume(transformed_mask, data, affine)
    np.savetxt(matrix, np.eye(4))

    interface = DSCRegistration()
    interface.inputs.brain_mask = str(anatomical_mask)
    interface.inputs.out_matrix = str(matrix)

    with (
        patch(
            "pmriprep.utils.registration.transform_mask_to_grid",
            return_value=str(transformed_mask),
        ),
        patch(
            "pmriprep.utils.registration.calculate_dice",
            side_effect=RuntimeError("synthetic Dice failure"),
        ),
        caplog.at_level("WARNING"),
    ):
        metrics = interface._compute_qc_metrics(
            str(registered), str(dsc_mean)
        )

    assert metrics["dice_computation_failed"] is True
    assert metrics["qc_status"] == "UNKNOWN"
    assert "synthetic Dice failure" in caplog.text
