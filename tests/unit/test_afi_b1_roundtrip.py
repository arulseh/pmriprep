"""Can-fail test: the AFI B1 interface must recover a known flip angle from
physically-generated AFI signals.

Guards the AFI signal-ratio orientation. The ratio must be r = S2/S1
(long-TR over short-TR) for cos(alpha) = (r*n - 1)/(n - r); the reciprocal
r = S1/S2 drives cos(alpha) out of [-1, 1] for normal-flip voxels, collapsing
B1 toward zero. Prior detection-only tests never fed physical signals through
the interface, so that orientation error was latent.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce import B1Mapping


def _afi_signals(
    t1_ms: float, tr1_ms: float, tr2_ms: float, alpha_deg: float, m0: float = 1000.0
) -> tuple[float, float]:
    """Yarnykh AFI steady-state signals (ideal spoiling) for both TR periods."""
    a = np.radians(alpha_deg)
    e1 = np.exp(-tr1_ms / t1_ms)
    e2 = np.exp(-tr2_ms / t1_ms)
    den = 1.0 - e1 * e2 * np.cos(a) ** 2
    s1 = m0 * np.sin(a) * (1.0 - e2 + (1.0 - e1) * e2 * np.cos(a)) / den
    s2 = m0 * np.sin(a) * (1.0 - e1 + (1.0 - e2) * e1 * np.cos(a)) / den
    return s1, s2


def test_afi_b1_recovers_known_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Run in tmp_path so the interface's cwd-relative outputs (QC figure,
    # reoriented AFI) don't pollute the repo root.
    monkeypatch.chdir(tmp_path)
    t1, tr1, tr2 = 1000.0, 20.0, 100.0
    nominal_fa, actual_fa = 60.0, 54.0  # true B1 = 0.90
    s1, s2 = _afi_signals(t1, tr1, tr2, actual_fa)

    shape = (8, 8, 8)
    afi = np.zeros((*shape, 2), dtype=np.float32)
    afi[..., 0] = s1  # vol0 = short-TR signal (S1)
    afi[..., 1] = s2  # vol1 = long-TR signal (S2)

    mask = np.ones(shape, dtype=np.uint8)

    affine = np.eye(4)
    afi_file = tmp_path / "sub-01_TB1AFI.nii.gz"
    nib.save(nib.Nifti1Image(afi, affine), afi_file)
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), mask_file)

    b1 = B1Mapping(
        afi_4d_file=str(afi_file),
        brain_mask=str(mask_file),
        tr1=tr1,
        tr2=tr2,
        nominal_flip_angle=nominal_fa,
        smoothing_fwhm=0.0,
        out_file=str(tmp_path / "b1.nii.gz"),
    )
    result = b1.run()

    assert result.outputs.mean_b1 == pytest.approx(actual_fa / nominal_fa, abs=0.02)
    assert result.outputs.qc_metrics["n_valid_voxels"] == np.prod(shape)
    assert result.outputs.qc_pass is True


def test_afi_smoothing_fills_supported_internal_gap(tmp_path: Path) -> None:
    """Normalized convolution fills a small internal AFI support gap."""
    t1, tr1, tr2 = 1000.0, 20.0, 100.0
    nominal_fa, actual_fa = 60.0, 54.0
    s1, s2 = _afi_signals(t1, tr1, tr2, actual_fa)
    shape = (10, 10, 10)
    afi = np.empty((*shape, 2), dtype=np.float32)
    afi[..., 0] = s1
    afi[..., 1] = s2
    afi[5, 5, 5, :] = 0.0
    mask = np.ones(shape, dtype=np.uint8)
    afi_file = tmp_path / "sub-01_TB1AFI.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(afi, np.eye(4)), afi_file)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)

    result = B1Mapping(
        afi_4d_file=str(afi_file),
        brain_mask=str(mask_file),
        tr1=tr1,
        tr2=tr2,
        nominal_flip_angle=nominal_fa,
        smoothing_fwhm=4.0,
    ).run(cwd=str(tmp_path))
    mapped = nib.load(result.outputs.out_file).get_fdata()
    assert mapped[5, 5, 5] == pytest.approx(actual_fa / nominal_fa, abs=0.02)
    assert result.outputs.qc_metrics["n_valid_voxels"] == np.prod(shape) - 1
    assert result.outputs.qc_metrics["n_b1_interpolated_voxels"] == 1


def test_afi_b1_reversed_volume_order_uses_per_volume_tr_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sorted TR overrides must not detach TR identity from reversed volumes."""
    import json

    monkeypatch.chdir(tmp_path)
    t1, tr1, tr2 = 1000.0, 20.0, 100.0
    nominal_fa, actual_fa = 60.0, 48.0
    s1, s2 = _afi_signals(t1, tr1, tr2, actual_fa)
    shape = (8, 8, 8)
    afi = np.empty((*shape, 2), dtype=np.float32)
    afi[..., 0] = s2
    afi[..., 1] = s1
    afi_file = tmp_path / "sub-01_TB1AFI.nii.gz"
    nib.save(nib.Nifti1Image(afi, np.eye(4)), afi_file)
    (tmp_path / "sub-01_TB1AFI.json").write_text(json.dumps({
        "RepetitionTimeExcitation": [tr2 / 1000.0, tr1 / 1000.0],
        "FlipAngle": nominal_fa,
    }))

    result = B1Mapping(
        afi_4d_file=str(afi_file),
        tr1=tr1,
        tr2=tr2,
        nominal_flip_angle=nominal_fa,
        smoothing_fwhm=0.0,
        out_file=str(tmp_path / "b1_reversed.nii.gz"),
    ).run()
    assert result.outputs.mean_b1 == pytest.approx(actual_fa / nominal_fa, abs=0.02)


def test_afi_b1_requires_nominal_flip_angle(tmp_path: Path) -> None:
    """AFI cannot silently assume a vendor-specific nominal flip angle."""
    afi_file = tmp_path / "sub-01_TB1AFI.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 2), np.float32), np.eye(4)), afi_file)
    with pytest.raises(ValueError, match="nominal_flip_angle not set"):
        B1Mapping(
            afi_4d_file=str(afi_file), tr1=20.0, tr2=100.0,
            smoothing_fwhm=0.0,
        ).run(cwd=str(tmp_path))


@pytest.mark.parametrize("tr1,tr2", [(0.0, 100.0), (20.0, 20.0), (100.0, 20.0)])
def test_afi_b1_rejects_invalid_tr_pair(
    tmp_path: Path, tr1: float, tr2: float,
) -> None:
    """AFI requires a finite positive short/long TR ordering."""
    afi_file = tmp_path / f"afi_{tr1}_{tr2}.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 2), np.float32), np.eye(4)), afi_file)
    with pytest.raises(ValueError, match="0 < TR1 < TR2"):
        B1Mapping(
            afi_4d_file=str(afi_file), tr1=tr1, tr2=tr2,
            nominal_flip_angle=60.0, smoothing_fwhm=0.0,
        ).run(cwd=str(tmp_path))


def test_afi_b1_empty_physical_support_cannot_pass_qc(tmp_path: Path) -> None:
    """Ratios implying cos(alpha)>1 are unavailable, not clipped valid data."""
    afi = np.empty((4, 4, 4, 2), dtype=np.float32)
    afi[..., 0] = 100.0
    afi[..., 1] = 140.0
    afi_file = tmp_path / "afi_impossible.nii.gz"
    nib.save(nib.Nifti1Image(afi, np.eye(4)), afi_file)
    result = B1Mapping(
        afi_4d_file=str(afi_file), tr1=20.0, tr2=100.0,
        nominal_flip_angle=60.0, smoothing_fwhm=0.0,
    ).run(cwd=str(tmp_path))
    assert result.outputs.qc_metrics["n_valid_voxels"] == 0
    assert result.outputs.qc_metrics["n_cosine_invalid"] == np.prod(afi.shape[:3])
    assert result.outputs.qc_pass is False
    assert not np.any(nib.load(result.outputs.out_file).get_fdata())


def test_b1_resampling_preserves_measured_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A uniform measured region must not be extrapolated across the target FOV."""
    from pmriprep.workflows.dce._helpers import _transform_b1_to_dce

    monkeypatch.chdir(tmp_path)
    source = np.zeros((3, 3, 3), dtype=np.float32)
    source[1:, 1:, 1:] = 0.9
    source_file = tmp_path / "b1_source.nii.gz"
    reference_file = tmp_path / "dce_reference.nii.gz"
    nib.save(nib.Nifti1Image(source, np.eye(4)), source_file)
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), np.float32), np.eye(4)), reference_file)

    out_file = _transform_b1_to_dce(str(source_file), str(reference_file))
    assert out_file is not None
    transformed = nib.load(out_file).get_fdata()
    assert transformed.shape == (5, 5, 5)
    assert transformed[1, 1, 1] == pytest.approx(0.9)
    assert transformed[4, 4, 4] == 0.0
