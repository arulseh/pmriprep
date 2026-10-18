"""Characterization tests for DSC AIF input loading and acquisition context."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection


def _write_nifti(
    path: Path,
    data: np.ndarray,
    *,
    tr: float | None = None,
) -> str:
    image = nib.Nifti1Image(data, np.eye(4))
    if tr is not None:
        image.header.set_zooms((*image.header.get_zooms()[:3], tr))
    nib.save(image, path)
    return str(path)


def test_rejects_brain_mask_outside_dsc_grid(tmp_path: Path) -> None:
    """The public interface rejects a brain mask with different dimensions."""
    concentration = _write_nifti(
        tmp_path / "concentration.nii.gz",
        np.zeros((6, 6, 6, 20), dtype=np.float32),
        tr=1.5,
    )
    brain_mask = _write_nifti(
        tmp_path / "brain_mask.nii.gz",
        np.ones((5, 6, 6), dtype=np.uint8),
    )

    interface = AIFDetection(
        concentration_file=concentration,
        brain_mask_file=brain_mask,
    )

    with pytest.raises(RuntimeError, match="must be in native DSC space"):
        interface.run(cwd=tmp_path)


def test_header_tr_reaches_failed_detection_qc(tmp_path: Path) -> None:
    """A zero input TR uses the fourth NIfTI zoom in downstream QC."""
    concentration = _write_nifti(
        tmp_path / "concentration.nii.gz",
        np.zeros((6, 6, 6, 20), dtype=np.float32),
        tr=2.25,
    )
    brain_mask = _write_nifti(
        tmp_path / "brain_mask.nii.gz",
        np.ones((6, 6, 6), dtype=np.uint8),
    )

    result = AIFDetection(
        concentration_file=concentration,
        brain_mask_file=brain_mask,
        tr=0.0,
    ).run(cwd=tmp_path)

    assert result.outputs.detection_level == "unavailable"
    assert result.outputs.qc_metrics["tr_used"] == pytest.approx(2.25)
