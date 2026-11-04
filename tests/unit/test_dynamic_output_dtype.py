"""Derived dynamic images preserve precision independently of input storage."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("product", ["mean", "baseline", "early", "dsc", "dce"])
def test_integer_input_derived_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, product: str, scaled: bool,
) -> None:
    """Integer and equivalent float inputs produce identical float32 derivatives."""
    from pmriprep.workflows.dce._helpers import (
        _compute_early_mean,
        _signal_to_concentration_node,
    )
    from pmriprep.workflows.dsc._helpers import (
        _compute_mean_4d,
        _compute_prebolus_mean,
        _signal_to_concentration,
    )

    monkeypatch.chdir(tmp_path)
    spatial = np.arange(24, dtype=np.int16).reshape(2, 3, 4) * 17 + 1000
    temporal = np.array([0, 1, 2, 3, 5, 20, 35, 40], dtype=np.int16)
    data = spatial[..., None] + temporal
    affine = np.diag([2., 2., 3., 1.])
    source = tmp_path / "integer.nii.gz"
    img = nib.Nifti1Image(data, affine)
    if scaled:
        img.header.set_slope_inter(0.3, 19.1)
    nib.save(img, source)
    original_bytes = source.read_bytes()
    values = nib.load(source).get_fdata()
    equivalent = tmp_path / "float.nii.gz"
    nib.save(nib.Nifti1Image(values, affine), equivalent)
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(data.shape[:3], dtype=np.uint8), affine), mask)

    def derive(path: Path) -> nib.Nifti1Image:
        """Run the production serializer and load its result before reuse."""
        if product == "mean":
            output = _compute_mean_4d(str(path))
        elif product == "baseline":
            output = _compute_prebolus_mean(str(path), 5)
        elif product == "early":
            output = _compute_early_mean(str(path))
        elif product == "dsc":
            output, _ = _signal_to_concentration(str(path), 0.03, 5)
        else:
            output, _ = _signal_to_concentration_node(
                str(path), str(mask), None, None, 15., 0.005, 3., 3.8, 5, True,
            )
        result = nib.load(output)
        result.get_fdata()
        return result

    actual = derive(source)
    expected = derive(equivalent)
    assert actual.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_array_equal(actual.get_fdata(), expected.get_fdata())
    np.testing.assert_array_equal(actual.affine, affine)
    if product in {"mean", "baseline", "early"}:
        n = {"mean": values.shape[3], "baseline": 5, "early": 3}[product]
        mean = values[..., :n].mean(axis=3)
        np.testing.assert_array_equal(actual.get_fdata(), mean.astype(np.float32))
    assert source.read_bytes() == original_bytes
