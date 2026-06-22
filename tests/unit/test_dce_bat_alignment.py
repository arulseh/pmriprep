"""Unit tests for DCE AIF->tissue BAT pre-alignment (``pmriprep.utils.dce_bat``).

Covers the ``shift_aif`` time-shift helper, every guard branch of
``compute_aif_tissue_delay``, and the minute-vs-second units contract of the
underlying BAT detector (guards the 60x conversion trap).
"""
from __future__ import annotations

import numpy as np

from pmriprep.utils.dce_bat import (
    AIF_DELAY_SUBFRAME_FRACTION,
    MAX_AIF_TISSUE_DELAY_S,
    compute_aif_tissue_delay,
    detect_bolus_arrival_time,
    shift_aif,
)


def _step(n: int, arrival: int, val: float = 5.0) -> np.ndarray:
    """Step curve: 0 before ``arrival``, ``val`` from ``arrival`` onward."""
    c = np.zeros(n, dtype=np.float64)
    c[arrival:] = val
    return c


def _minutes(n: int, tr_seconds: float) -> np.ndarray:
    """Uniform time axis in minutes for ``n`` frames spaced ``tr_seconds``."""
    return np.arange(n, dtype=np.float64) * (tr_seconds / 60.0)


class TestShiftAif:
    def test_zero_delay_is_identity(self):
        t = _minutes(20, 6.0)
        aif = np.linspace(0.0, 5.0, 20)
        np.testing.assert_array_equal(shift_aif(aif, t, 0.0), aif)

    def test_one_frame_shift_aligns(self):
        # 6 s == 0.1 min == one frame at this spacing; aif(t - 1 frame).
        t = _minutes(20, 6.0)
        aif = np.arange(20, dtype=np.float64)
        s = shift_aif(aif, t, 6.0)
        assert s[5] == aif[4]          # shifted later by one frame
        assert s[0] == aif[0]          # baseline left-fill, not zero/negative

    def test_baseline_left_fill_no_spurious_negative(self):
        t = _minutes(20, 6.0)
        aif = np.concatenate([np.zeros(3), np.linspace(0.0, 5.0, 17)])
        s = shift_aif(aif, t, 12.0)    # +2 frames
        assert s.min() >= aif.min()    # no acausal negative ramp introduced
        assert s[0] == aif[0]

    def test_non_uniform_axis(self):
        t = np.array([0.0, 0.05, 0.12, 0.2, 0.31, 0.45, 0.6, 0.8])
        aif = np.array([0.0, 1, 3, 5, 4, 3, 2, 1])
        s = shift_aif(aif, t, 3.0)
        assert s.shape == aif.shape
        assert np.all(np.isfinite(s))


class TestComputeAifTissueDelay:
    def test_applied_fine_tr(self):
        n, tr = 30, 2.0
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 9), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert prov["applied"] is True
        assert prov["reason"] == "applied"
        assert abs(d - 6.0) < 1e-6      # (9-6) frames * 2 s
        assert "granularity" not in prov  # added by the caller, not the helper

    def test_subresolution_noop(self):
        n, tr = 30, 8.0                 # coarse TR; same arrival frame -> 0 delay
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 6), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert d == 0.0
        assert prov["reason"] == "delay_below_resolution"

    def test_ceiling_clamp(self):
        n, tr = 40, 2.0                 # (25-6)*2 = 38 s > 12 s ceiling
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 25), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert d == 0.0
        assert prov["reason"] == "delay_clamped"
        assert prov["raw_delay_seconds"] > MAX_AIF_TISSUE_DELAY_S

    def test_negative_delay_rejected(self):
        n, tr = 30, 2.0                 # tissue arrives before the AIF
        d, prov = compute_aif_tissue_delay(
            _step(n, 9), _step(n, 6), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert d == 0.0
        assert prov["reason"] == "negative_delay_rejected"

    def test_low_confidence_gate(self):
        n, tr = 30, 2.0                 # flat tissue -> default-method BAT
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), np.zeros(n), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert d == 0.0
        assert prov["reason"] == "low_bat_confidence"


class TestBatUnitsContract:
    def test_bat_seconds_equals_frame_index_times_tr(self):
        # An AIF arriving at frame 6 with TR=2 s must report BAT = 12.0 s.
        # If the minute->second conversion were dropped this would be 0.2.
        n, tr = 30, 2.0
        r = detect_bolus_arrival_time(_step(n, 6), _minutes(n, tr), baseline_frames=3)
        assert r.method == "threshold"
        assert r.bat_seconds == 6 * tr


def test_subframe_fraction_constant_is_quarter():
    assert AIF_DELAY_SUBFRAME_FRACTION == 0.25
