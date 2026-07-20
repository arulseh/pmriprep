"""Characterization tests for DCE preprocessing phase extraction."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from pmriprep.interfaces.dce import DCEPreprocess


def _write_dce(path: Path, shape: tuple[int, int, int, int] = (4, 5, 6, 12)) -> None:
    """Write a deterministic DCE series."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 3.0, 1.0])), path)


def test_prepare_preprocess_inputs_preserves_fixed_baseline_and_mask(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Configured baseline keeps its clamp and still detects arrival for QC."""
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
        return 5, {"method": "characterized", "confidence": "high"}

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
    assert prepared.baseline_frames == 3
    assert prepared.baseline_indices == (0, 1, 2)
    assert prepared.bolus_frame == 5
    assert prepared.baseline_diagnostics == {
        "method": "characterized",
        "confidence": "high",
    }


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
