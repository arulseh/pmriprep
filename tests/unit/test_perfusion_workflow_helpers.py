"""Tests for helpers shared by DSC and DCE workflow graphs."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from traits.trait_base import Undefined

from pmriprep.workflows._perfusion import (
    select_aif_exclusion_mask,
    validate_native_mask,
)


def _write_image(path: Path, shape: tuple[int, int, int], affine: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine), path)
    return str(path)


@pytest.mark.parametrize("missing", [None, "", Undefined])
def test_select_aif_exclusion_mask_handles_missing_inputs(missing: object) -> None:
    assert select_aif_exclusion_mask(missing, missing) == ""  # type: ignore[arg-type]


def test_select_aif_exclusion_mask_precedence() -> None:
    assert select_aif_exclusion_mask("user.nii.gz", "tumor.nii.gz") == "user.nii.gz"
    assert select_aif_exclusion_mask("", "tumor.nii.gz") == "tumor.nii.gz"


@pytest.mark.parametrize("missing", [None, "", Undefined])
def test_validate_native_mask_handles_missing_input(missing: object) -> None:
    assert validate_native_mask(missing, "reference.nii.gz") == ""  # type: ignore[arg-type]


def test_validate_native_mask_accepts_matching_grid(tmp_path: Path) -> None:
    reference = _write_image(tmp_path / "reference.nii.gz", (3, 4, 5), np.eye(4))
    mask = _write_image(tmp_path / "mask.nii.gz", (3, 4, 5), np.eye(4))
    assert validate_native_mask(mask, reference, "dce.roi_mask") == mask


@pytest.mark.parametrize("n_volumes", [1, 2])
def test_validate_native_mask_rejects_4d_mask(tmp_path: Path, n_volumes: int) -> None:
    """Matching spatial dimensions must not hide an extra mask dimension."""
    reference = _write_image(tmp_path / "reference.nii.gz", (3, 4, 5), np.eye(4))
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5, n_volumes), dtype=np.uint8), np.eye(4)), mask)
    with pytest.raises(ValueError, match=r"dce\.roi_mask.*expected a 3D mask.*got 4D"):
        validate_native_mask(str(mask), reference, "dce.roi_mask")


def test_validate_native_mask_rejects_shape_mismatch(tmp_path: Path) -> None:
    reference = _write_image(tmp_path / "reference.nii.gz", (3, 4, 5), np.eye(4))
    mask = _write_image(tmp_path / "mask.nii.gz", (3, 4, 6), np.eye(4))
    with pytest.raises(ValueError, match="dsc.roi_mask shape"):
        validate_native_mask(mask, reference, "dsc.roi_mask")


def test_validate_native_mask_rejects_affine_mismatch(tmp_path: Path) -> None:
    reference = _write_image(tmp_path / "reference.nii.gz", (3, 4, 5), np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    mask = _write_image(tmp_path / "mask.nii.gz", (3, 4, 5), shifted)
    with pytest.raises(ValueError, match="affine differs"):
        validate_native_mask(mask, reference, "dce.lesion_mask")
