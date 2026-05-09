"""Regression tests for DCE registration QC mask handling.

These tests guard against the silent shape-mismatch bug where the
input ``brain_mask`` (T1w/anat space) was passed to NMI / Correlation /
DICE helpers that operate on DCE-space images, raising ValueError on
every metric and dropping the corresponding QC fields from
``dce_qc.json``.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest


def _save_volume(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _identity_flirt_mat(path: Path) -> None:
    np.savetxt(str(path), np.eye(4), fmt="%.6f")


@pytest.mark.skipif(
    shutil.which("flirt") is None or shutil.which("fslmaths") is None,
    reason="FSL FLIRT/fslmaths not available",
)
class TestTransformMaskToGrid:
    """Helper-level test for transform_mask_to_grid same-space shortcut."""

    def test_same_space_shortcut_produces_dce_grid_mask(
        self, tmp_path: Path
    ) -> None:
        """When mask and reference already share geometry, the shortcut
        binarises and hole-fills without invoking FLIRT.
        """
        from pmriprep.utils.registration import transform_mask_to_grid

        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        mask_data = np.zeros((8, 8, 4), dtype=np.uint8)
        mask_data[2:6, 2:6, 1:3] = 1
        ref_data = np.ones((8, 8, 4), dtype=np.float32)

        mask_file = tmp_path / "mask.nii.gz"
        ref_file = tmp_path / "ref.nii.gz"
        mat_file = tmp_path / "identity.mat"
        _save_volume(mask_file, mask_data, affine)
        _save_volume(ref_file, ref_data, affine)
        _identity_flirt_mat(mat_file)

        out = transform_mask_to_grid(
            str(mask_file),
            str(ref_file),
            str(mat_file),
            out_dir=tmp_path,
        )
        assert out
        out_img = nib.load(out)
        assert out_img.shape == (8, 8, 4)
        out_data = np.asarray(out_img.dataobj)
        assert set(np.unique(out_data).tolist()) <= {0, 1}
        assert out_data.sum() > 0
        assert not np.isnan(out_data).any()

    def test_returns_empty_when_mask_unset(self, tmp_path: Path) -> None:
        """Unset mask returns "" so downstream traits.Str ports accept it."""
        from pmriprep.utils.registration import transform_mask_to_grid

        ref_file = tmp_path / "ref.nii.gz"
        _save_volume(
            ref_file,
            np.ones((4, 4, 2), dtype=np.float32),
            np.eye(4),
        )
        assert transform_mask_to_grid(None, str(ref_file), None) == ""


class TestDCERegistrationQCMaskRegression:
    """End-to-end regression: input mask in anat space, images in DCE space.

    Today's bug: ``_compute_qc_metrics`` passes the anat-grid mask
    directly to the QC helpers, every metric raises on shape mismatch,
    and the resulting dict is missing nmi/correlation/dice and reports
    ``qc_status='UNKNOWN'``. After the fix, the input mask is resampled
    onto the DCE grid first; metrics populate and the gate reports
    PASS/WARN/POOR.
    """

    def test_qc_metrics_populate_when_input_mask_in_anat_space(
        self, tmp_path: Path
    ) -> None:
        from pmriprep.interfaces.dce import DCERegistration

        # DCE-space images (8,8,4 grid) — identical so metrics are well-defined
        dce_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        rng = np.random.default_rng(0)
        dce_mean_data = rng.uniform(50.0, 200.0, size=(8, 8, 4)).astype(np.float32)
        registered_t1w_data = dce_mean_data.copy()

        # Anat-space mask on a different grid (16,16,8) — the shape that
        # triggers the original bug.
        anat_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        anat_mask_data = np.zeros((16, 16, 8), dtype=np.uint8)
        anat_mask_data[2:14, 2:14, 1:7] = 1

        # DCE-grid mask that the resampler is expected to produce.
        dce_mask_data = np.ones((8, 8, 4), dtype=np.int16)

        t1w_file = tmp_path / "t1w.nii.gz"
        dce_mean_file = tmp_path / "dce_mean.nii.gz"
        registered_t1w_file = tmp_path / "registered_t1w.nii.gz"
        anat_mask_file = tmp_path / "brain_mask_anat.nii.gz"
        dce_mask_file = tmp_path / "brain_mask_dce.nii.gz"
        out_matrix = tmp_path / "registration.mat"

        _save_volume(t1w_file, anat_mask_data.astype(np.float32), anat_affine)
        _save_volume(dce_mean_file, dce_mean_data, dce_affine)
        _save_volume(registered_t1w_file, registered_t1w_data, dce_affine)
        _save_volume(anat_mask_file, anat_mask_data, anat_affine)
        _save_volume(dce_mask_file, dce_mask_data, dce_affine)
        _identity_flirt_mat(out_matrix)

        interface = DCERegistration()
        interface.inputs.t1w_file = str(t1w_file)
        interface.inputs.dce_mean = str(dce_mean_file)
        interface.inputs.brain_mask = str(anat_mask_file)
        interface.inputs.out_file = str(registered_t1w_file)
        interface.inputs.out_matrix = str(out_matrix)

        # Patch transform_mask_to_grid where _compute_qc_metrics imports
        # it (inside the method body) so the test does not depend on
        # FLIRT being installed.
        with patch(
            "pmriprep.utils.registration.transform_mask_to_grid",
            return_value=str(dce_mask_file),
        ):
            qc_metrics = interface._compute_qc_metrics(
                str(registered_t1w_file), str(dce_mean_file)
            )

        assert "nmi" in qc_metrics, qc_metrics
        assert "correlation" in qc_metrics, qc_metrics
        assert "dice" in qc_metrics, qc_metrics
        assert np.isfinite(qc_metrics["nmi"])
        assert np.isfinite(qc_metrics["correlation"])
        assert np.isfinite(qc_metrics["dice"])

        assert qc_metrics["qc_status"] in {"PASS", "WARN", "POOR"}, qc_metrics
        assert qc_metrics["brain_mask_used"] is True

    def test_qc_status_unknown_when_mask_resample_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """If the resampler returns "" (transform unavailable), the DICE
        branch short-circuits but NMI/Correlation still compute unmasked.
        ``brain_mask_used`` reports False, status is PASS/WARN/POOR.
        """
        from pmriprep.interfaces.dce import DCERegistration

        dce_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        anat_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        rng = np.random.default_rng(1)
        dce_mean_data = rng.uniform(50.0, 200.0, size=(8, 8, 4)).astype(np.float32)
        registered_t1w_data = dce_mean_data.copy()

        anat_mask_data = np.ones((16, 16, 8), dtype=np.uint8)

        t1w_file = tmp_path / "t1w.nii.gz"
        dce_mean_file = tmp_path / "dce_mean.nii.gz"
        registered_t1w_file = tmp_path / "registered_t1w.nii.gz"
        anat_mask_file = tmp_path / "brain_mask_anat.nii.gz"
        out_matrix = tmp_path / "registration.mat"

        _save_volume(t1w_file, anat_mask_data.astype(np.float32), anat_affine)
        _save_volume(dce_mean_file, dce_mean_data, dce_affine)
        _save_volume(registered_t1w_file, registered_t1w_data, dce_affine)
        _save_volume(anat_mask_file, anat_mask_data, anat_affine)
        _identity_flirt_mat(out_matrix)

        interface = DCERegistration()
        interface.inputs.t1w_file = str(t1w_file)
        interface.inputs.dce_mean = str(dce_mean_file)
        interface.inputs.brain_mask = str(anat_mask_file)
        interface.inputs.out_file = str(registered_t1w_file)
        interface.inputs.out_matrix = str(out_matrix)

        with patch(
            "pmriprep.utils.registration.transform_mask_to_grid",
            return_value="",
        ):
            qc_metrics = interface._compute_qc_metrics(
                str(registered_t1w_file), str(dce_mean_file)
            )

        # NMI / Correlation still compute (unmasked); DICE absent.
        assert "nmi" in qc_metrics
        assert "correlation" in qc_metrics
        assert "dice" not in qc_metrics
        assert qc_metrics["brain_mask_used"] is False
        assert qc_metrics["qc_status"] in {"PASS", "WARN", "POOR"}


class TestDCERegistrationQCStatusGating:
    """qc_status routing under the recalibrated cross-modal thresholds.

    Stubs the metric helpers so the test exercises only the threshold
    table + decision block — independent of FSL availability.
    """

    @pytest.mark.parametrize(
        "dice,corr,nmi,det,expected",
        [
            (0.96, 0.08, 0.04, 1.00, "PASS"),
            (0.96, 0.02, 0.04, 1.00, "POOR"),
            (0.96, 0.05, 0.04, 1.00, "PASS"),
            (0.70, 0.20, 0.05, 1.00, "POOR"),
            (0.96, 0.20, 0.05, 1.30, "WARN"),
            (0.96, 0.40, 0.50, 1.00, "PASS"),
        ],
        ids=[
            "clean_cross_modal",
            "corr_below_floor",
            "corr_in_advisory_band",
            "dice_below_min",
            "det_out_of_range",
            "high_nmi_no_warning",
        ],
    )
    def test_qc_status_routing(
        self, tmp_path, dice, corr, nmi, det, expected,
    ):
        from pmriprep.interfaces.dce import DCERegistration

        dce_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        rng = np.random.default_rng(0)
        dce_mean_data = rng.uniform(50.0, 200.0, size=(8, 8, 4)).astype(np.float32)
        registered_t1w_data = dce_mean_data.copy()
        anat_mask_data = np.ones((8, 8, 4), dtype=np.uint8)

        t1w_file = tmp_path / "t1w.nii.gz"
        dce_mean_file = tmp_path / "dce_mean.nii.gz"
        registered_t1w_file = tmp_path / "registered_t1w.nii.gz"
        anat_mask_file = tmp_path / "brain_mask.nii.gz"
        out_matrix = tmp_path / "registration.mat"

        _save_volume(t1w_file, anat_mask_data.astype(np.float32), dce_affine)
        _save_volume(dce_mean_file, dce_mean_data, dce_affine)
        _save_volume(registered_t1w_file, registered_t1w_data, dce_affine)
        _save_volume(anat_mask_file, anat_mask_data, dce_affine)
        _identity_flirt_mat(out_matrix)

        interface = DCERegistration()
        interface.inputs.t1w_file = str(t1w_file)
        interface.inputs.dce_mean = str(dce_mean_file)
        interface.inputs.brain_mask = str(anat_mask_file)
        interface.inputs.out_file = str(registered_t1w_file)
        interface.inputs.out_matrix = str(out_matrix)

        with patch(
            "pmriprep.utils.registration.transform_mask_to_grid",
            return_value=str(anat_mask_file),
        ), patch(
            "pmriprep.utils.registration.calculate_nmi_from_files",
            return_value=nmi,
        ), patch(
            "pmriprep.utils.registration.calculate_correlation_from_files",
            return_value=corr,
        ), patch(
            "pmriprep.utils.registration.calculate_dice",
            return_value=dice,
        ), patch.object(
            DCERegistration, "_check_transform_determinant",
            return_value=det,
        ):
            qc_metrics = interface._compute_qc_metrics(
                str(registered_t1w_file), str(dce_mean_file)
            )

        assert qc_metrics["qc_status"] == expected, qc_metrics
