"""Tests for the DESPOT1 WLLS weight de-bias and the VFA-fit NRMSE QC fix.

Locks the fix that removed the dimensional misuse of ``EPSILON_T1`` (a T1-in-seconds
floor) as a magnitude floor on the heteroscedastic weight denominator
``1 - E1*cos(alpha)`` and on the normal-equation determinants. The old 0.01 floor
de-weighted the informative long-T1 / small-flip points (Deoni 2003 Eq 2/3, weighted
per Chang 2008), biasing T1 upward at >=3 flip angles. Also covers the report-only
VFA-fit NRMSE now being evaluated at the B1-corrected flip angles.
"""
from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce import T1Mapping

# --- SPGR / DESPOT1 helpers ------------------------------------------------

def _spgr(s0: float, fa_rad: np.ndarray, tr_s: float, t1_s: float) -> np.ndarray:
    """Spoiled gradient-echo steady-state signal (Deoni 2003 Eq 1)."""
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


class TestExposedRegime:
    """The chosen config lands where the OLD 0.01 floor actually fired."""

    def test_smallflip_denominator_between_old_and_new_floor(self) -> None:
        """denom_w for the small-flip anchor is in (1e-6, 0.01): old floor bit it."""
        e1 = np.exp(-_TR_S / _T1_EXPOSED)
        denom_w = 1.0 - e1 * np.cos(_FA_EXPOSED[0])  # smallest flip
        assert 1e-6 < denom_w < 1e-2  # new floor 1e-6 spares it; old 0.01 clamped it


class TestNoiselessRecovery:
    """Noiseless SPGR lies exactly on the DESPOT1 line -> exact E1 for any weights."""

    @pytest.mark.parametrize("t1_s", [0.4, 0.6, 0.8, 1.2])
    def test_three_angle_recovers_t1(self, t1_s: float) -> None:
        """>=3-angle noiseless fit recovers the ground-truth T1 to fp precision."""
        sig = _spgr(500.0, _FA_EXPOSED, _TR_S, t1_s)[:, None, None, None]
        t1 = _fit(sig, _FA_EXPOSED, _TR_S)
        assert t1[0] == pytest.approx(t1_s, rel=1e-3)

    def test_four_angle_stable(self) -> None:
        """A 4th anchor does not destabilize the dominant-weight fit (noiseless)."""
        fa = np.deg2rad([2.0, 5.0, 12.0, 20.0])
        sig = _spgr(500.0, fa, _TR_S, _T1_EXPOSED)[:, None, None, None]
        t1 = _fit(sig, fa, _TR_S)
        assert t1[0] == pytest.approx(_T1_EXPOSED, rel=1e-3)


class TestWeightDebiasVsOldFloor:
    """Monte-Carlo: correct weight lowers DESPOT1 bias without inflating variance.

    Simulates the OLD behaviour by monkeypatching ``EPSILON_WEIGHT_DENOM`` back to
    0.01 (the fitter re-imports the constant on every call), on the SAME noise
    realisations, so old vs new is a paired comparison. The de-bias is a mean
    effect (unweighted GLLS overestimates T1; restoring the weight pulls it toward
    truth), so it is measured on mean bias / RMSE, not the median.
    """

    def _mc(self, monkeypatch, floor: float, seed: int = 20240712, n: int = 8000):
        monkeypatch.setattr(
            "pmriprep.utils.dce_constants.EPSILON_WEIGHT_DENOM", floor
        )
        rng = np.random.default_rng(seed)
        sig_true = _spgr(1.0, _FA_EXPOSED, _TR_S, _T1_EXPOSED)
        sigma = sig_true.max() / 30.0  # SNR ~30 on the peak flip
        noisy = sig_true[:, None, None, None] + rng.normal(
            0.0, sigma, size=(len(_FA_EXPOSED), n, 1, 1)
        )
        return _fit(noisy, _FA_EXPOSED, _TR_S)

    def test_new_floor_reduces_bias_and_rmse(self, monkeypatch) -> None:
        """New 1e-6 floor: smaller mean bias, smaller RMSE, no variance blow-up."""
        # Same seed -> paired on identical noise.
        t1_old = self._mc(monkeypatch, 0.01)
        t1_new = self._mc(monkeypatch, 1e-6)
        ok = np.isfinite(t1_old) & np.isfinite(t1_new)
        t1_old, t1_new = t1_old[ok], t1_new[ok]

        bias_old = abs(t1_old.mean() - _T1_EXPOSED)
        bias_new = abs(t1_new.mean() - _T1_EXPOSED)
        rmse_old = np.sqrt(np.mean((t1_old - _T1_EXPOSED) ** 2))
        rmse_new = np.sqrt(np.mean((t1_new - _T1_EXPOSED) ** 2))

        # Old GLLS-like weight overestimates T1 (positive bias); the correct
        # weight is systematically closer to truth on the same noise.
        assert (t1_old - t1_new).mean() > 0  # paired: old runs higher
        assert bias_new < bias_old  # mean de-bias (Deoni/Chang weight restored)
        assert rmse_new < rmse_old  # accuracy improves; variance not inflated


class TestDegenerateGuard:
    """Zero-scale voxels stay finite; 2-angle case is weight-invariant."""

    def test_zero_signal_voxel_is_finite(self) -> None:
        """All-zero signal -> determinant 0, but the abs backstop avoids 0/0 NaN."""
        sig = np.zeros((3, 1, 1, 1))
        t1 = _fit(sig, _FA_EXPOSED, _TR_S)
        assert np.all(np.isfinite(t1))

    def test_nfa2_weight_invariant(self, monkeypatch) -> None:
        """Two angles determine the line exactly: old vs new floor agree (allclose)."""
        fa2 = np.deg2rad([3.0, 15.0])
        sig = _spgr(500.0, fa2, _TR_S, _T1_EXPOSED)[:, None, None, None]

        monkeypatch.setattr("pmriprep.utils.dce_constants.EPSILON_WEIGHT_DENOM", 0.01)
        t1_old = _fit(sig, fa2, _TR_S)
        monkeypatch.setattr("pmriprep.utils.dce_constants.EPSILON_WEIGHT_DENOM", 1e-6)
        t1_new = _fit(sig, fa2, _TR_S)

        assert np.allclose(t1_old, t1_new, rtol=1e-6)
        assert t1_new[0] == pytest.approx(_T1_EXPOSED, rel=1e-3)


class TestVFAFitNRMSECorrectedAngles:
    """Fix 2: the report-only VFA-fit NRMSE is evaluated at B1-corrected angles."""

    def test_nrmse_near_zero_for_perfect_corrected_data(
        self, tmp_path, monkeypatch
    ) -> None:
        """Perfect SPGR at corrected angles (B1=1.2) -> NRMSE ~ 0, not inflated.

        If the forward model used nominal angles (the old bug), s_pred would be
        evaluated at the wrong flip and NRMSE would be materially non-zero.
        """
        # Keep the interface's cwd-relative outputs (T1 map, QC PNG) in tmp.
        monkeypatch.chdir(tmp_path)
        nominal_deg = np.array([3.0, 7.0, 15.0])
        b1 = 1.2
        tr_ms = 5.0
        t1_s = 1.000
        s0 = 500.0
        shape = (8, 8, 8)

        corrected_rad = np.deg2rad(nominal_deg * b1)
        affine = np.eye(4)
        vfa_files = []
        for i, fa_rad in enumerate(corrected_rad):
            vol = np.full(shape, _spgr(s0, np.array([fa_rad]), tr_ms / 1000.0, t1_s)[0])
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
