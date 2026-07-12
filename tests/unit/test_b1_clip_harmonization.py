"""Tests for the 3T B1+ clip harmonization (Finding 3).

The VFA T1 fit and the signal->concentration inversion consume the same B1+ field
but historically clipped it to different ceilings at 3T (VFA 1.7, concentration
1.5). For a voxel with true B1 in (1.5, 1.7] the concentration path then inverted
C(t) at the wrong (clipped) flip while the T1 map used the true flip, biasing
Ktrans. Raising 3T ``dce_clip_high`` to 1.7 harmonizes the two paths.
"""
from __future__ import annotations

import numpy as np
import pytest

import pmriprep.utils.dce_constants as dce_constants
from pmriprep.utils.dce_constants import get_r1_relaxivity
from pmriprep.utils.dce_signal import signal_to_concentration

_NOMINAL_FA_DEG = 20.0
_TR_S = 0.005
_T1_PRE_MS = 1000.0
_R1 = get_r1_relaxivity(3.0)  # s^-1 mM^-1, matches the inversion's lookup
_N_BASELINE = 5


def _spgr(s0: float, fa_rad: float, tr_s: float, t1_s: float) -> float:
    e1 = np.exp(-tr_s / t1_s)
    return s0 * np.sin(fa_rad) * (1.0 - e1) / (1.0 - e1 * np.cos(fa_rad))


def _forward_signal(true_b1: float, c_peak: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a (1,1,1,T) SPGR series at the TRUE flip and its concentration curve.

    5 pre-bolus frames (C=0) then a 5-frame ramp to ``c_peak`` mM.
    """
    true_fa = np.deg2rad(_NOMINAL_FA_DEG * true_b1)
    r1_0 = 1.0 / (_T1_PRE_MS / 1000.0)
    c_curve = np.concatenate([np.zeros(_N_BASELINE), np.linspace(0.0, c_peak, 5)])
    s = np.array([
        _spgr(1000.0, true_fa, _TR_S, 1.0 / (r1_0 + _R1 * c)) for c in c_curve
    ])
    return s.reshape(1, 1, 1, -1), c_curve


def _invert(signal: np.ndarray, b1: float) -> np.ndarray:
    t1_map = np.full((1, 1, 1), _T1_PRE_MS)
    b1_map = np.full((1, 1, 1), b1)
    res = signal_to_concentration(
        signal, _NOMINAL_FA_DEG, _TR_S, t1_map=t1_map, b1_map=b1_map,
        r1_relaxivity=_R1, field_strength=3.0, baseline_frames=_N_BASELINE,
        convert_to_plasma=False,
    )
    return res.concentration.reshape(-1)


class TestHighB1Harmonization:
    """True B1 in the newly-included (1.5, 1.7] band."""

    def test_new_clip_recovers_peak_and_beats_old(self, monkeypatch) -> None:
        """NEW (1.7) recovers the known peak C; OLD (1.5) is more biased."""
        c_peak = 0.5
        signal, c_curve = _forward_signal(true_b1=1.6, c_peak=c_peak)

        # NEW: production constants (dce_clip_high == 1.7 == vfa_clip_high).
        c_new = _invert(signal, b1=1.6)
        err_new = abs(c_new[-1] - c_peak)

        # OLD: restore the split ceiling (dce_clip_high == 1.5).
        old_entry = dce_constants.B1_THRESHOLDS_BY_FIELD[3.0]._replace(
            dce_clip_high=1.5
        )
        monkeypatch.setitem(dce_constants.B1_THRESHOLDS_BY_FIELD, 3.0, old_entry)
        c_old = _invert(signal, b1=1.6)
        err_old = abs(c_old[-1] - c_peak)

        assert err_new == pytest.approx(0.0, abs=5e-3)  # flip now matches the T1 map
        assert err_new < err_old  # harmonization removes the high-B1 bias


class TestLowB1Unchanged:
    """True B1 <= 1.5 is below both ceilings: no behaviour change."""

    def test_b1_below_old_ceiling_is_identical(self, monkeypatch) -> None:
        """B1 = 1.4 gives byte-identical C under the old and new clip."""
        signal, _ = _forward_signal(true_b1=1.4, c_peak=0.5)
        c_new = _invert(signal, b1=1.4)

        old_entry = dce_constants.B1_THRESHOLDS_BY_FIELD[3.0]._replace(
            dce_clip_high=1.5
        )
        monkeypatch.setitem(dce_constants.B1_THRESHOLDS_BY_FIELD, 3.0, old_entry)
        c_old = _invert(signal, b1=1.4)

        assert np.allclose(c_new, c_old, rtol=0, atol=0)
