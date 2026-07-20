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


@pytest.mark.parametrize(
    "ref_vol,interpolation,reference_args,interpolation_arg",
    [
        ("mean", "trilinear", ["-meanvol"], None),
        ("first", "sinc", ["-refvol", "0"], "-sinc_final"),
        (4, "spline", ["-refvol", "4"], "-spline_final"),
    ],
)
def test_prepare_mcflirt_plan_preserves_reference_and_interpolation_flags(
    tmp_path: Path,
    monkeypatch: Any,
    ref_vol: str | int,
    interpolation: str,
    reference_args: list[str],
    interpolation_arg: str | None,
) -> None:
    """Reference modes and final interpolation map to the same CLI flags."""
    dce_file = tmp_path / "dce.nii.gz"
    _write_dce(dce_file)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        DCEPreprocess,
        "_detect_bolus_arrival",
        lambda self, data, mask: (5, {"confidence": "high"}),
    )
    interface = DCEPreprocess(
        in_file=str(dce_file),
        baseline_frames=3,
        ref_vol=ref_vol,
        interpolation=interpolation,
        cost="normmi",
    )

    plan = interface._prepare_mcflirt_plan(interface._prepare_preprocess_inputs())
    command = list(plan.command)

    assert command[:7] == [
        "mcflirt", "-in", str(dce_file), "-out", plan.out_base, "-plots", "-stats"
    ]
    assert "-mats" in command
    assert command[command.index("-cost") + 1] == "normmi"
    start = command.index(reference_args[0])
    assert command[start:start + len(reference_args)] == reference_args
    assert command[-2:] == ["-stages", "4"]
    if interpolation_arg is None:
        assert "-sinc_final" not in command
        assert "-spline_final" not in command
    else:
        assert interpolation_arg in command
    assert plan.reference_file is None


def test_prepare_mcflirt_plan_writes_baseline_mean_reference(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Baseline-mean mode writes the characterized float32 reference volume."""
    dce_file = tmp_path / "dce.nii.gz"
    _write_dce(dce_file)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        DCEPreprocess,
        "_detect_bolus_arrival",
        lambda self, data, mask: (5, {"confidence": "high"}),
    )
    interface = DCEPreprocess(in_file=str(dce_file), baseline_frames=3)
    prepared = interface._prepare_preprocess_inputs()

    plan = interface._prepare_mcflirt_plan(prepared)

    assert plan.reference_file == str(tmp_path / "_tmp_baseline_mean.nii.gz")
    assert list(plan.command)[-2:] == ["-stages", "4"]
    ref_index = plan.command.index("-reffile")
    assert plan.command[ref_index + 1] == plan.reference_file
    reference = nib.load(plan.reference_file).get_fdata()
    assert np.allclose(reference, prepared.data[..., :3].mean(axis=3))
