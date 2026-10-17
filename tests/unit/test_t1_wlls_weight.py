"""Test WLLS T1 recovery and residuals computed at B1-corrected flip angles."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce import T1Mapping

# --- SPGR / DESPOT1 helpers ------------------------------------------------

def _spgr(s0: float, fa_rad: np.ndarray, tr_s: float, t1_s: float) -> np.ndarray:
    """Spoiled gradient-echo steady-state signal."""
    e1 = np.exp(-tr_s / t1_s)
    return s0 * np.sin(fa_rad) * (1.0 - e1) / (1.0 - e1 * np.cos(fa_rad))


def _linearize(signal: np.ndarray, fa_rad: np.ndarray):
    """Return (x=S/tan a, y=S/sin a, sin a, cos a) broadcast to (n_fa, nvox,1,1)."""
    sin_a = np.sin(fa_rad)[:, None, None, None]
    cos_a = np.cos(fa_rad)[:, None, None, None]
    tan_a = sin_a / cos_a
    x = signal / tan_a
    y = signal / sin_a
    sin_b = np.broadcast_to(sin_a, signal.shape).copy()
    cos_b = np.broadcast_to(cos_a, signal.shape).copy()
    return x, y, sin_b, cos_b


def _fit(signal, fa_rad, tr_s):
    """Call the production WLLS fitter, returning T1 (seconds), shape (nvox,)."""
    x, y, sin_b, cos_b = _linearize(signal, fa_rad)
    shape = signal.shape[1:]
    t1 = T1Mapping()._fit_t1_wlls(x, y, sin_b, cos_b, tr_s, shape)
    return t1.reshape(-1)


_TR_S = 0.004  # 4 ms short pulse TR
_FA_EXPOSED = np.deg2rad([2.0, 5.0, 12.0])  # >=3 angles, small-flip anchor
_T1_EXPOSED = 0.600  # 600 ms -> E1 ~ 0.9934, in the exposed band (0.99, 0.995)


class TestNoiselessRecovery:
    """Noiseless SPGR lies exactly on the DESPOT1 line -> exact E1 for any weights."""

    @pytest.mark.parametrize("t1_s", [0.4, 0.6, 0.8, 1.2])
    def test_three_angle_recovers_t1(self, t1_s: float) -> None:
        """>=3-angle noiseless fit recovers the ground-truth T1 within 0.1%."""
        sig = _spgr(500.0, _FA_EXPOSED, _TR_S, t1_s)[:, None, None, None]
        t1 = _fit(sig, _FA_EXPOSED, _TR_S)
        assert t1[0] == pytest.approx(t1_s, rel=1e-3)

    def test_four_angle_stable(self) -> None:
        """A 4th anchor does not destabilize the dominant-weight fit (noiseless)."""
        fa = np.deg2rad([2.0, 5.0, 12.0, 20.0])
        sig = _spgr(500.0, fa, _TR_S, _T1_EXPOSED)[:, None, None, None]
        t1 = _fit(sig, fa, _TR_S)
        assert t1[0] == pytest.approx(_T1_EXPOSED, rel=1e-3)


class TestDegenerateGuard:
    """Zero-scale voxels stay finite and two flip angles recover noiseless T1."""

    def test_zero_signal_voxel_is_finite(self) -> None:
        """All-zero signal -> determinant 0, but the abs backstop avoids 0/0 NaN."""
        sig = np.zeros((3, 1, 1, 1))
        t1 = _fit(sig, _FA_EXPOSED, _TR_S)
        assert np.all(np.isfinite(t1))

    def test_two_angle_recovers_t1(self) -> None:
        """Two noiseless flip angles recover T1 with the configured weighting."""
        fa2 = np.deg2rad([3.0, 15.0])
        sig = _spgr(500.0, fa2, _TR_S, _T1_EXPOSED)[:, None, None, None]

        t1 = _fit(sig, fa2, _TR_S)
        assert t1[0] == pytest.approx(_T1_EXPOSED, rel=1e-3)


class TestVFAFitNRMSECorrectedAngles:
    """B1-corrected fitting, finite support and QC through T1Mapping."""

    @pytest.mark.parametrize(
        ("n_slices", "upper_t1_s", "erosion_mode", "expected_asymmetry", "warn", "fail"),
        [
            (30, 1.0, "2d_per_slice", 0.0, False, False),
            (31, 1.2, "3d", 0.1818, True, False),
            (8, 1.5, "2d_per_slice", 0.4, True, True),
        ],
    )
    def test_corrected_fit_support_and_asymmetry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        n_slices: int, upper_t1_s: float,
        erosion_mode: str, expected_asymmetry: float, warn: bool, fail: bool,
    ) -> None:
        """Known SPGR signals retain the expected slab interior and T1 asymmetry."""
        # Keep the interface's cwd-relative outputs (T1 map, QC PNG) in tmp.
        monkeypatch.chdir(tmp_path)
        nominal_deg = np.array([3.0, 7.0, 15.0])
        b1 = 1.2
        tr_ms = 5.0
        t1_s = 1.000
        s0 = 500.0
        shape = (8, 8, n_slices)

        corrected_rad = np.deg2rad(nominal_deg * b1)
        affine = np.eye(4)
        affine[0, 3] = -3.5
        vfa_files = []
        for i, fa_rad in enumerate(corrected_rad):
            vol = np.full(shape, _spgr(s0, np.array([fa_rad]), tr_ms / 1000.0, t1_s)[0])
            vol[4:] = _spgr(s0, np.array([fa_rad]), tr_ms / 1000.0, upper_t1_s)[0]
            p = tmp_path / f"vfa_{i}.nii.gz"
            nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), str(p))
            vfa_files.append(str(p))

        b1_p = tmp_path / "b1.nii.gz"
        nib.save(nib.Nifti1Image(np.full(shape, b1, np.float32), affine), str(b1_p))
        mask_p = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones(shape, np.uint8), affine), str(mask_p))

        out = T1Mapping(
            vfa_files=vfa_files,
            flip_angles=list(nominal_deg),
            tr=tr_ms,
            b1_map=str(b1_p),
            brain_mask=str(mask_p),
            t1_method="wlls",
            t1_correction_mode="none",
        ).run().outputs

        nrmse = out.qc_metrics["vfa_conditioning"]["vfa_fit_nrmse"]
        assert nrmse is not None
        assert nrmse < 0.02  # perfect fit at the correct angles -> essentially zero

        expected_support = np.zeros(shape, dtype=bool)
        expected_support[1:-1, 1:-1, :] = True
        if erosion_mode == "3d":
            expected_support[:, :, [0, -1]] = False
        measured = nib.load(out.out_file).get_fdata()
        np.testing.assert_array_equal(np.isfinite(measured), expected_support)
        truth = np.full(shape, 1000.0)
        truth[4:] = upper_t1_s * 1000.0
        np.testing.assert_allclose(measured[expected_support], truth[expected_support], rtol=1e-3)
        qc = out.qc_metrics
        assert qc["edge_erosion_mode"] == erosion_mode
        assert qc["edge_voxels_eroded"] == np.count_nonzero(~expected_support)
        assert qc["t1_support"]["directly_measured_voxels"] == np.count_nonzero(expected_support)
        asymmetry = qc["hemispheric_asymmetry"]
        assert asymmetry["asymmetry_index"] == pytest.approx(expected_asymmetry, abs=1e-4)
        assert asymmetry["warn"] is warn
        assert asymmetry["fail"] is fail
