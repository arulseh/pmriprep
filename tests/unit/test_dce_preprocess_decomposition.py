"""Characterization tests for DCE preprocessing phase extraction."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce import DCEPreprocess


def _write_dce(path: Path, shape: tuple[int, int, int, int] = (4, 5, 6, 12)) -> None:
    """Write a deterministic DCE series."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 3.0, 1.0])), path)


def test_prepare_preprocess_inputs_flags_fixed_baseline_below_minimum(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """An inadequate configured baseline is replaced and marked invalid."""
    dce_file = tmp_path / "dce.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    _write_dce(dce_file)
    nib.save(
        nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.uint8), np.eye(4)),
        mask_file,
    )
    captured: dict[str, Any] = {}

    def _detect(self: DCEPreprocess, data: np.ndarray, mask: np.ndarray):
        captured["shape"] = data.shape
        captured["mask"] = mask
        return 7, {"method": "characterized", "confidence": "high"}

    monkeypatch.setattr(DCEPreprocess, "_detect_bolus_arrival", _detect)
    interface = DCEPreprocess(
        in_file=str(dce_file), brain_mask=str(mask_file), baseline_frames=2
    )

    prepared = interface._prepare_preprocess_inputs()

    assert prepared.data.dtype == np.float64
    assert prepared.data.shape == (4, 5, 6, 12)
    assert prepared.n_volumes == 12
    assert prepared.brain_mask is not None
    assert prepared.brain_mask.dtype == np.bool_
    assert captured["shape"] == prepared.data.shape
    assert captured["mask"] is prepared.brain_mask
    assert prepared.baseline_frames == 5
    assert prepared.baseline_indices == tuple(range(5))
    assert prepared.bolus_frame == 7
    assert prepared.baseline_diagnostics["method"] == "characterized"
    assert prepared.baseline_diagnostics["confidence"] == "high"
    assert prepared.baseline_diagnostics["baseline_status"] == "POOR"
    assert prepared.baseline_diagnostics["baseline_fallback_applied"] is True
    assert prepared.baseline_diagnostics["baseline_fallback_reason"] == (
        "configured_baseline_below_minimum"
    )


def test_prepare_preprocess_inputs_preserves_auto_baseline_and_mask_fallback(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Auto detection receives no mask when the optional mask grid mismatches."""
    dce_file = tmp_path / "dce.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    _write_dce(dce_file)
    nib.save(
        nib.Nifti1Image(np.ones((3, 5, 6), dtype=np.uint8), np.eye(4)),
        mask_file,
    )
    captured: dict[str, Any] = {}

    def _detect(self: DCEPreprocess, data: np.ndarray, mask: Any):
        captured["mask"] = mask
        return 7, 7, {"method": "auto", "confidence": "medium"}

    monkeypatch.setattr(DCEPreprocess, "_auto_detect_baseline", _detect)
    interface = DCEPreprocess(
        in_file=str(dce_file), brain_mask=str(mask_file), baseline_frames=0
    )

    prepared = interface._prepare_preprocess_inputs()

    assert captured["mask"] is None
    assert prepared.brain_mask is None
    assert prepared.baseline_frames == 7
    assert prepared.baseline_indices == tuple(range(7))
    assert prepared.bolus_frame == 7
    assert prepared.baseline_diagnostics == {
        "method": "auto",
        "confidence": "medium",
    }


def test_prepare_preprocess_inputs_surfaces_invalid_early_auto_baseline(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """An impossible early auto result falls back nonfatally with provenance."""
    dce_file = tmp_path / "dce.nii.gz"
    _write_dce(dce_file)
    monkeypatch.setattr(
        DCEPreprocess,
        "_auto_detect_baseline",
        lambda self, data, mask: (
            2,
            2,
            {"method": "characterized", "confidence": "low"},
        ),
    )
    interface = DCEPreprocess(in_file=str(dce_file), baseline_frames=0)

    prepared = interface._prepare_preprocess_inputs()

    assert prepared.baseline_frames == 5
    assert prepared.baseline_indices == tuple(range(5))
    assert prepared.bolus_frame == 5
    assert prepared.baseline_diagnostics["detected_bolus_frame"] == 2
    assert prepared.baseline_diagnostics["baseline_frames_used"] == 5
    assert prepared.baseline_diagnostics["baseline_minimum_frames"] == 5
    assert prepared.baseline_diagnostics["baseline_status"] == "POOR"
    assert prepared.baseline_diagnostics["baseline_fallback_applied"] is True
    assert "PK maps are unreliable" in prepared.baseline_diagnostics["baseline_warning"]


def test_prepare_preprocess_inputs_flags_series_without_five_baseline_frames(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """A short acquisition uses available frames but remains quantitatively invalid."""
    dce_file = tmp_path / "short_dce.nii.gz"
    _write_dce(dce_file, shape=(4, 5, 6, 4))
    monkeypatch.setattr(
        DCEPreprocess,
        "_auto_detect_baseline",
        lambda self, data, mask: (
            3,
            3,
            {"method": "characterized", "confidence": "low"},
        ),
    )
    interface = DCEPreprocess(in_file=str(dce_file), baseline_frames=0)

    prepared = interface._prepare_preprocess_inputs()

    assert prepared.baseline_frames == 3
    assert prepared.baseline_indices == (0, 1, 2)
    assert prepared.bolus_frame == 3
    assert prepared.baseline_diagnostics["baseline_minimum_frames"] == 5
    assert prepared.baseline_diagnostics["baseline_fallback_reason"] == (
        "insufficient_volumes_for_minimum_baseline"
    )


@pytest.mark.parametrize("ref_vol", ["baseline_mean", "mean", "first", 4])
def test_prepare_motion_reference(
    tmp_path: Path, monkeypatch: Any, ref_vol: str | int,
) -> None:
    """ANTs receives the requested reference on the acquisition grid."""
    dce_file = tmp_path / "dce.nii.gz"
    _write_dce(dce_file)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DCEPreprocess, "_detect_bolus_arrival",
                        lambda self, data, mask: (5, {"confidence": "high"}))
    interface = DCEPreprocess(in_file=str(dce_file), baseline_frames=5, ref_vol=ref_vol)
    prepared = interface._prepare_preprocess_inputs()
    actual = nib.load(interface._prepare_motion_reference(prepared))
    if ref_vol == "baseline_mean":
        expected = prepared.data[..., :5].mean(axis=3)
    elif ref_vol == "mean":
        expected = prepared.data.mean(axis=3)
    else:
        expected = prepared.data[..., 0 if ref_vol == "first" else ref_vol]
    np.testing.assert_allclose(actual.get_fdata(), expected)
    np.testing.assert_array_equal(actual.affine, prepared.affine)


def test_reference_preserves_fractional_mean_from_integer_input(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """A derived mean must not inherit integer storage from the acquisition."""
    dce_file = tmp_path / "integer_dce.nii.gz"
    spatial = np.arange(120, dtype=np.int16).reshape(4, 5, 6) * 17
    temporal = np.array([0, 1, 2, 3, 5, 7], dtype=np.int16)
    data = spatial[..., None] + temporal
    nib.save(nib.Nifti1Image(data, np.diag([2., 2., 3., 1.])), dce_file)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DCEPreprocess, "_detect_bolus_arrival",
                        lambda self, data, mask: (5, {"confidence": "high"}))
    interface = DCEPreprocess(in_file=str(dce_file), baseline_frames=5)
    prepared = interface._prepare_preprocess_inputs()
    reference = nib.load(interface._prepare_motion_reference(prepared))
    assert reference.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_array_equal(
        reference.get_fdata(), data[..., :5].mean(axis=3).astype(np.float32),
    )
    np.testing.assert_array_equal(reference.affine, prepared.affine)
    original_bytes = dce_file.read_bytes()
    outputs = interface._write_preprocess_outputs(prepared, str(dce_file), "motion.par")
    for path, expected in (
        (outputs.mean_file, data.mean(axis=3)),
        (outputs.baseline_mean_file, data[..., :5].mean(axis=3)),
    ):
        derived = nib.load(path)
        assert derived.get_data_dtype() == np.dtype(np.float32)
        np.testing.assert_array_equal(derived.get_fdata(), expected.astype(np.float32))
    assert dce_file.read_bytes() == original_bytes


def test_summarize_motion_qc_preserves_fd_and_outlier_reduction(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Motion reduction retains FD arithmetic, thresholds, and baseline QC."""
    dce_file = tmp_path / "dce.nii.gz"
    params_file = tmp_path / "motion.par"
    _write_dce(dce_file, shape=(4, 5, 6, 3))
    np.savetxt(
        params_file,
        np.array([
            [0.00, 0, 0, 0.0, 0, 0],
            [0.00, 0, 0, 0.6, 0, 0],
            [0.01, 0, 0, 0.6, 0, 0],
        ]),
    )
    monkeypatch.setattr(
        DCEPreprocess,
        "_detect_bolus_arrival",
        lambda self, data, mask: (
            5,
            {"method": "characterized", "confidence": "medium"},
        ),
    )
    interface = DCEPreprocess(
        in_file=str(dce_file),
        baseline_frames=5,
        motion_thresh=0.55,
        head_radius=50.0,
        apply_motion_correction=False,
        ref_vol="first",
    )
    prepared = interface._prepare_preprocess_inputs()

    qc = interface._summarize_motion_qc(str(params_file), prepared)

    assert qc.fd == pytest.approx((0.0, 0.6, 0.5))
    assert qc.fd_mean == pytest.approx(0.55)
    assert qc.fd_max == pytest.approx(0.6)
    assert qc.n_outliers == 1
    assert qc.outlier_frames == (1,)
    assert qc.metrics["fd_list"] == pytest.approx([0.0, 0.6, 0.5])
    assert qc.metrics["percent_outliers"] == pytest.approx(100.0 / 3.0)
    assert qc.metrics["baseline_frames"] == 2
    assert qc.metrics["bolus_arrival_frame"] == 2
    assert qc.metrics["baseline_detection_method"] == "characterized"
    assert qc.metrics["baseline_detection_confidence"] == "medium"
    assert qc.metrics["reference_type"] == "first"
    assert qc.metrics["motion_applied"] is False


def test_summarize_motion_qc_surfaces_nonfatal_failure(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Missing parameters retain nonfatal outputs with visible QC provenance."""
    dce_file = tmp_path / "dce.nii.gz"
    _write_dce(dce_file)
    monkeypatch.setattr(
        DCEPreprocess,
        "_detect_bolus_arrival",
        lambda self, data, mask: (5, {"confidence": "high"}),
    )
    interface = DCEPreprocess(in_file=str(dce_file), baseline_frames=5)
    prepared = interface._prepare_preprocess_inputs()

    qc = interface._summarize_motion_qc(str(tmp_path / "missing.par"), prepared)

    assert qc.fd == ()
    assert qc.fd_mean == 0.0
    assert qc.fd_max == 0.0
    assert qc.n_outliers == 0
    assert qc.outlier_frames == ()
    assert qc.metrics["motion_status"] == "UNKNOWN"
    assert qc.metrics["motion_qc_available"] is False
    assert qc.metrics["motion_qc_error"] == "FileNotFoundError"
    assert "Motion quality is unknown" in qc.metrics["motion_qc_warning"]
    assert qc.metrics["baseline_frames"] == 5
    assert qc.metrics["bolus_arrival_frame"] == 5
    assert qc.metrics["motion_applied"] is True


def test_motion_qc_failure_retains_baseline_exclusion_state(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Motion reduction failure cannot erase an inadequate baseline."""
    dce_file = tmp_path / "dce.nii.gz"
    _write_dce(dce_file)
    monkeypatch.setattr(
        DCEPreprocess,
        "_auto_detect_baseline",
        lambda self, data, mask: (
            2,
            2,
            {"method": "characterized", "confidence": "low"},
        ),
    )
    interface = DCEPreprocess(in_file=str(dce_file), baseline_frames=0)
    prepared = interface._prepare_preprocess_inputs()

    qc = interface._summarize_motion_qc(str(tmp_path / "missing.par"), prepared)

    assert qc.metrics["motion_qc_available"] is False
    assert qc.metrics["baseline_fallback_applied"] is True
    assert qc.metrics["baseline_status"] == "POOR"
    assert qc.metrics["detected_bolus_frame"] == 2
    assert "PK maps are unreliable" in qc.metrics["baseline_warning"]
