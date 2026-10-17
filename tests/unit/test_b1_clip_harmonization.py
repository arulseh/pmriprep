"""Concentration recovery with B1-corrected SPGR flip angles."""
from __future__ import annotations

import numpy as np
import pytest

from pmriprep.utils.dce_signal import signal_to_concentration

_NOMINAL_FA_DEG = 20.0
_TR_S = 0.005
_T1_PRE_MS = 1000.0
_R1 = 3.7  # s^-1 mM^-1
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


@pytest.mark.parametrize("b1", [1.4, 1.6])
def test_concentration_recovery_with_b1(b1: float) -> None:
    """SPGR inversion recovers the known curve at the effective flip angle."""
    signal, expected = _forward_signal(true_b1=b1, c_peak=0.5)
    np.testing.assert_allclose(_invert(signal, b1), expected, rtol=0, atol=5e-3)
