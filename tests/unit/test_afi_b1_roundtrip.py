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

    # A low-signal slab (excluded from the brain mask) pulls the interface's
    # 5th-percentile signal threshold below the real signals, so the tissue
    # voxels are not masked out by the strict-greater-than noise gate.
    afi[:, :, 0, :] = 2.0
    mask = np.ones(shape, dtype=np.uint8)
    mask[:, :, 0] = 0

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
