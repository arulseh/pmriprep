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


def test_header_alignment_retained_when_fitted_overlap_is_worse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration optimization cannot replace a clearly better header alignment."""
    from pmriprep.interfaces.dce import DCERegistration
    from pmriprep.utils import registration

    paths = {
        name: tmp_path / name
        for name in (
            "header.nii.gz",
            "header.mat",
            "path_a.nii.gz",
            "path_a.mat",
            "path_b.nii.gz",
            "path_b.mat",
            "dce_mean.nii.gz",
        )
    }
    for path in paths.values():
        path.touch()

    overlap = {
        str(paths["header.nii.gz"]): 0.947,
        str(paths["path_a.nii.gz"]): 0.055,
        str(paths["path_b.nii.gz"]): 0.055,
    }
    monkeypatch.setattr(
        registration,
        "compute_dice_for_selection",
        lambda moving, _fixed: overlap[moving],
    )

    selected = DCERegistration()._select_best_path(
        header_aligned_file=str(paths["header.nii.gz"]),
        header_matrix=str(paths["header.mat"]),
        out_file_a=str(paths["path_a.nii.gz"]),
        out_matrix_a=str(paths["path_a.mat"]),
        out_file_b=str(paths["path_b.nii.gz"]),
        out_matrix_b=str(paths["path_b.mat"]),
        dce_mean=str(paths["dce_mean.nii.gz"]),
    )

    assert selected == (
        str(paths["header.nii.gz"]),
        str(paths["header.mat"]),
        0.947,
        0.055,
        0.055,
        "header",
        0.947,
    )


def test_header_alignment_retained_for_submargin_fitted_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negligible fitted-overlap gain does not displace header alignment."""
    from pmriprep.interfaces.dce import DCERegistration
    from pmriprep.utils import registration

    paths = {
        name: tmp_path / name
        for name in (
            "header.nii.gz",
            "header.mat",
            "path_a.nii.gz",
            "path_a.mat",
            "path_b.nii.gz",
            "path_b.mat",
            "dce_mean.nii.gz",
        )
    }
    for path in paths.values():
        path.touch()
    overlap = {
        str(paths["header.nii.gz"]): 0.925,
        str(paths["path_a.nii.gz"]): 0.930,
        str(paths["path_b.nii.gz"]): 0.900,
    }
    monkeypatch.setattr(
        registration,
        "compute_dice_for_selection",
        lambda moving, _fixed: overlap[moving],
    )

    selected = DCERegistration()._select_best_path(
        header_aligned_file=str(paths["header.nii.gz"]),
        header_matrix=str(paths["header.mat"]),
        out_file_a=str(paths["path_a.nii.gz"]),
        out_matrix_a=str(paths["path_a.mat"]),
        out_file_b=str(paths["path_b.nii.gz"]),
        out_matrix_b=str(paths["path_b.mat"]),
        dce_mean=str(paths["dce_mean.nii.gz"]),
    )

    assert selected[-2] == "header"
    assert selected[-1] == 0.925


def test_fitted_registration_selected_for_material_overlap_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fitted path replaces header alignment when overlap clearly improves."""
    from pmriprep.interfaces.dce import DCERegistration
    from pmriprep.utils import registration

    paths = {
        name: tmp_path / name
        for name in (
            "header.nii.gz",
            "header.mat",
            "path_a.nii.gz",
            "path_a.mat",
            "path_b.nii.gz",
            "path_b.mat",
            "dce_mean.nii.gz",
        )
    }
    for path in paths.values():
        path.touch()
    overlap = {
        str(paths["header.nii.gz"]): 0.925,
        str(paths["path_a.nii.gz"]): 0.950,
        str(paths["path_b.nii.gz"]): 0.900,
    }
    monkeypatch.setattr(
        registration,
        "compute_dice_for_selection",
        lambda moving, _fixed: overlap[moving],
    )

    selected = DCERegistration()._select_best_path(
        header_aligned_file=str(paths["header.nii.gz"]),
        header_matrix=str(paths["header.mat"]),
        out_file_a=str(paths["path_a.nii.gz"]),
        out_matrix_a=str(paths["path_a.mat"]),
        out_file_b=str(paths["path_b.nii.gz"]),
        out_matrix_b=str(paths["path_b.mat"]),
        dce_mean=str(paths["dce_mean.nii.gz"]),
    )

    assert selected[-2] == "A"
    assert selected[-1] == 0.950


def test_transform_offset_from_header_reports_physical_motion(
    tmp_path: Path,
) -> None:
    """Registration offset is reported in physical units without a QC cutoff."""
    from pmriprep.interfaces.dce import DCERegistration
    from pmriprep.utils.registration import write_ras_transform_to_fsl

    moving = tmp_path / "moving.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    header_matrix = tmp_path / "header.mat"
    selected_matrix = tmp_path / "selected.mat"
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    _save_volume(moving, np.ones((8, 8, 6)), affine)
    _save_volume(reference, np.ones((8, 8, 6)), affine)
    write_ras_transform_to_fsl(
        np.eye(4), moving, reference, header_matrix
    )
    selected_ras = np.eye(4)
    selected_ras[:3, 3] = [3.0, 4.0, 0.0]
    write_ras_transform_to_fsl(
        selected_ras, moving, reference, selected_matrix
    )

    metrics = DCERegistration._transform_offset_from_header(
        selected_matrix=str(selected_matrix),
        header_matrix=str(header_matrix),
        moving_file=str(moving),
        reference_file=str(reference),
    )

    assert metrics["selected_vs_header_origin_translation_mm"] == pytest.approx(5.0)
    assert metrics["selected_vs_header_rotation_degrees"] == pytest.approx(0.0)
    assert metrics["selected_vs_header_rms_displacement_mm"] == pytest.approx(5.0)


def test_fsl_header_matrix_is_identity_in_original_moving_frame(
    tmp_path: Path,
) -> None:
    """Header transfer matrix represents world identity for the original source."""
    from nitransforms.linear import Affine

    from pmriprep.utils.registration import write_ras_transform_to_fsl

    moving = tmp_path / "moving.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    matrix = tmp_path / "header.mat"
    moving_affine = np.array([
        [1.0, 0.0, 0.0, -88.0],
        [0.0, 1.0, 0.0, -105.0],
        [0.0, 0.0, 1.0, -70.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    reference_affine = np.array([
        [2.5, 0.0, 0.0, -68.0],
        [0.0, 2.5, 0.0, -90.0],
        [0.0, 0.0, 4.0, -60.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    _save_volume(moving, np.ones((16, 16, 16)), moving_affine)
    _save_volume(reference, np.ones((8, 8, 8)), reference_affine)

    write_ras_transform_to_fsl(np.eye(4), moving, reference, matrix)
    decoded = Affine.from_filename(
        matrix,
        fmt="fsl",
        reference=reference,
        moving=moving,
    )

    np.testing.assert_allclose(decoded.matrix, np.eye(4), atol=1e-7)


def test_reframed_path_a_preserves_world_transform(tmp_path: Path) -> None:
    """Path-A matrix retains its world transform for the original source grid."""
    from nitransforms.linear import Affine

    from pmriprep.utils.registration import (
        reframe_fsl_transform,
        write_ras_transform_to_fsl,
    )

    original = tmp_path / "original.nii.gz"
    header = tmp_path / "header.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    path_a_header = tmp_path / "path_a_header.mat"
    path_a_original = tmp_path / "path_a_original.mat"
    _save_volume(original, np.ones((16, 16, 16)), np.diag([1.0, 1.0, 1.0, 1.0]))
    _save_volume(header, np.ones((8, 8, 8)), np.diag([2.0, 2.0, 2.0, 1.0]))
    _save_volume(reference, np.ones((8, 8, 8)), np.diag([2.0, 2.0, 2.0, 1.0]))
    ras_transform = np.eye(4)
    ras_transform[:3, 3] = [2.0, -3.0, 1.5]
    write_ras_transform_to_fsl(
        ras_transform, header, reference, path_a_header
    )

    reframe_fsl_transform(
        path_a_header,
        header,
        original,
        reference,
        path_a_original,
    )
    decoded = Affine.from_filename(
        path_a_original,
        fmt="fsl",
        reference=reference,
        moving=original,
    )

    np.testing.assert_allclose(decoded.matrix, ras_transform, atol=1e-7)


@pytest.mark.skipif(shutil.which("flirt") is None, reason="FSL FLIRT unavailable")
def test_header_matrix_applies_original_source_to_reference_grid(
    tmp_path: Path,
) -> None:
    """Published header matrix reproduces header transfer from the original grid."""
    import subprocess

    from pmriprep.interfaces.dce import DCERegistration

    moving = tmp_path / "moving.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    matrix = tmp_path / "header.mat"
    reapplied = tmp_path / "reapplied.nii.gz"
    moving_data = np.zeros((16, 16, 16), dtype=np.float32)
    moving_data[4:12, 5:13, 3:11] = 100.0
    moving_affine = np.array([
        [1.0, 0.0, 0.0, -8.0],
        [0.0, 1.0, 0.0, -8.0],
        [0.0, 0.0, 1.0, -8.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    reference_affine = np.array([
        [2.0, 0.0, 0.0, -6.0],
        [0.0, 2.0, 0.0, -7.0],
        [0.0, 0.0, 2.0, -5.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    _save_volume(moving, moving_data, moving_affine)
    _save_volume(reference, np.ones((8, 8, 8)), reference_affine)
    expected = tmp_path / "header_aligned.nii.gz"
    DCERegistration._run_header_alignment(
        str(moving),
        str(reference),
        str(expected),
        str(matrix),
    )

    subprocess.run(
        [
            "flirt",
            "-in",
            str(moving),
            "-ref",
            str(reference),
            "-applyxfm",
            "-init",
            str(matrix),
            "-interp",
            "trilinear",
            "-out",
            str(reapplied),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expected_data = np.asarray(nib.load(expected).dataobj)
    reapplied_data = np.asarray(nib.load(reapplied).dataobj)
    np.testing.assert_allclose(reapplied_data, expected_data, atol=1e-4)


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
