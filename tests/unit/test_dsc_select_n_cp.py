"""Tests for adaptive control-point selection under cpi_infer_cpt=True.

Calibration reference: tests/benchmarks/run_cpi_n_sweep_infer_cpt.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from pmriprep.models.dsc.cpi import (
    CPI_N_MAX,
    CPI_N_MIN,
    estimate_tail_decay_time_seconds,
    select_n_control_points,
)


def _synthetic_aif(n: int = 60, dt: float = 1.5, baseline: int = 10) -> np.ndarray:
    """Parker-like AIF: gamma-variate first pass + recirculation tail."""
    t = np.arange(n) * dt
    aif = np.zeros(n, dtype=np.float64)
    bolus_frames = max(0, n - baseline)
    if bolus_frames > 0:
        t_bolus = np.arange(bolus_frames) * dt
        first = (t_bolus / 3.0) ** 3.0 * np.exp(-t_bolus / 1.5) * 0.3
        recirc = 0.05 * np.exp(-((t_bolus - 12.0) ** 2) / (2.0 * 8.0**2))
        aif[baseline:] = first + recirc
    return aif


def _convolve_exp(aif: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Mean tissue from convolving AIF with single-exp residue."""
    n = len(aif)
    t = np.arange(n) * dt
    residue = np.exp(-t / tau)
    # Trapezoidal Toeplitz convolution (matches CPI forward model)
    a = aif.copy()
    tissue = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(i + 1):
            w = 0.5 if (j == 0 or j == i) else 1.0
            s += w * a[j] * residue[i - j]
        tissue[i] = s * dt
    return tissue


class TestEstimateTailDecayTime:
    """Direct tests of the tail-decay helper."""

    def test_short_tail(self) -> None:
        """Short-MTT tissue: decay-to-10% lands within a few frames of 2.3·τ
        plus AIF tail (empirically ≈ 19s for τ=4 with this AIF)."""
        aif = _synthetic_aif()
        tissue = _convolve_exp(aif, dt=1.5, tau=4.0)
        decay = estimate_tail_decay_time_seconds(tissue, dt=1.5)
        # Loose bound: ~3-5x τ in seconds for short MTT
        assert 9.0 < decay < 30.0

    def test_long_tail(self) -> None:
        """Long-MTT tissue (τ=12s) decays ~30s with this AIF."""
        aif = _synthetic_aif()
        tissue = _convolve_exp(aif, dt=1.5, tau=12.0)
        decay = estimate_tail_decay_time_seconds(tissue, dt=1.5)
        assert 25.0 < decay < 45.0

    def test_long_tail_above_short(self) -> None:
        """Decay must monotonically grow with MTT."""
        aif = _synthetic_aif()
        d_short = estimate_tail_decay_time_seconds(
            _convolve_exp(aif, 1.5, tau=4.0), dt=1.5,
        )
        d_long = estimate_tail_decay_time_seconds(
            _convolve_exp(aif, 1.5, tau=12.0), dt=1.5,
        )
        assert d_long > d_short

    def test_zero_curve_fallback(self) -> None:
        zeros = np.zeros(60, dtype=np.float64)
        decay = estimate_tail_decay_time_seconds(zeros, dt=1.5, fallback_sec=42.0)
        assert decay == 42.0

    def test_all_nan_curve_fallback(self) -> None:
        """All-NaN tissue (e.g. from a corrupted brain mask) returns fallback,
        not the cap. Without the explicit isfinite guard, NaN comparisons
        would silently pass through to the cap path."""
        nans = np.full(60, np.nan, dtype=np.float64)
        decay = estimate_tail_decay_time_seconds(nans, dt=1.5, fallback_sec=42.0)
        assert decay == 42.0

    def test_peak_at_last_frame_fallback(self) -> None:
        """Monotone-rising curve (peak at last frame) → fallback."""
        rising = np.linspace(0.0, 1.0, 30, dtype=np.float64)
        decay = estimate_tail_decay_time_seconds(rising, dt=1.5, fallback_sec=42.0)
        assert decay == 42.0

    def test_capped_when_no_threshold_crossing(self) -> None:
        """Tail that never drops below 10% (preloaded study) is capped at
        ``max_decay_fraction × T·dt``."""
        n = 40
        dt = 1.5
        # Curve: peak at frame 5, then plateau at 50% of peak (never crosses 10%)
        tissue = np.zeros(n, dtype=np.float64)
        tissue[:5] = np.linspace(0.0, 1.0, 5)
        tissue[5:] = 0.5
        decay = estimate_tail_decay_time_seconds(
            tissue, dt=dt, max_decay_fraction=0.5,
        )
        cap = 0.5 * n * dt
        assert decay <= cap + 1e-9


class TestSelectNAdaptive:
    """Tests of the adaptive floor in select_n_control_points."""

    @pytest.fixture
    def aif_dt(self) -> tuple[np.ndarray, float]:
        return _synthetic_aif(), 1.5

    def test_no_change_when_infer_cpt_false(self, aif_dt) -> None:
        """Non-adaptive (positional, infer_cpt=False) path ignores mean_tissue."""
        # Bolus duration 15 frames → n_bolus = 9, n_spacing = 4 → N = 9
        n_legacy = select_n_control_points(15)
        assert n_legacy == 9
        # Even with mean_tissue provided, infer_cpt=False ignores it
        aif, dt = aif_dt
        mt = _convolve_exp(aif, dt, tau=20.0)  # would force N high if active
        n_no_change = select_n_control_points(
            15, mean_tissue=mt, dt=dt, infer_cpt=False,
        )
        assert n_no_change == 9

    def test_adaptive_ignores_bolus_duration(self, aif_dt) -> None:
        """Adaptive path discards ``bolus_duration_frames``.

        Under ``infer_cpt=True`` n_bolus is dropped (CPs migrate to where
        they're needed). The selector should depend only on the tissue
        tail, not on the bolus passage length passed in.
        """
        aif, dt = aif_dt
        mt = _convolve_exp(aif, dt, tau=4.0)
        # Same tissue, very different bolus_duration_frames inputs:
        n_short_bolus = select_n_control_points(
            5, infer_cpt=True, mean_tissue=mt, dt=dt,
        )
        n_long_bolus = select_n_control_points(
            40, infer_cpt=True, mean_tissue=mt, dt=dt,
        )
        # Non-adaptive would give n_bolus=4 vs 22 — wildly different. Adaptive
        # ignores bolus_duration_frames entirely.
        assert n_short_bolus == n_long_bolus
        # And the non-adaptive path responds to bolus_duration_frames:
        n_legacy_short = select_n_control_points(5)   # infer_cpt=False
        n_legacy_long = select_n_control_points(40)
        assert n_legacy_short < n_legacy_long

    def test_medium_mtt_clears_cliff(self, aif_dt) -> None:
        """τ=12s tissue must yield N ≥ 8 (empirical cliff)."""
        aif, dt = aif_dt
        mt = _convolve_exp(aif, dt, tau=12.0)
        n = select_n_control_points(15, infer_cpt=True, mean_tissue=mt, dt=dt)
        assert n >= 8

    def test_stroke_mtt_elevates_n(self, aif_dt) -> None:
        """τ=20s stroke-like tissue must elevate N above the non-adaptive baseline of 9."""
        aif, dt = aif_dt
        mt = _convolve_exp(aif, dt, tau=20.0)
        n = select_n_control_points(15, infer_cpt=True, mean_tissue=mt, dt=dt)
        # Stroke needs N≥14 per the cliff analysis; clamp at max_n=24
        assert n >= 14
        assert n <= CPI_N_MAX

    def test_pathological_mtt_clamped_to_max(self, aif_dt) -> None:
        """Extreme MTT clamps to CPI_N_MAX."""
        # Synthesize a curve that never decays below 10% in a long acquisition;
        # cap will set decay = 0.5 * 90s = 45s → N = ceil(45/3) = 15.
        # To actually hit CPI_N_MAX we need an even longer apparent decay.
        # Use the cap directly: compose a 200-frame curve with persistent tail.
        n_frames = 200
        dt = 1.5
        tissue = np.zeros(n_frames, dtype=np.float64)
        tissue[:5] = np.linspace(0.0, 1.0, 5)
        tissue[5:] = 0.5  # plateau forever — cap kicks in at 0.5 * 300s = 150s
        n = select_n_control_points(
            15, infer_cpt=True, mean_tissue=tissue, dt=dt,
        )
        assert n == CPI_N_MAX  # 24

    def test_min_n_floor_holds(self, aif_dt) -> None:
        """If both non-adaptive and adaptive floors are below min_n, return min_n."""
        # Tiny bolus_duration → non-adaptive n_bolus=2, n_spacing=4. Zero tissue
        # gives fallback 30s decay → N=10 (overrides). So use a curve where
        # decay is small enough that min_n=4 dominates.
        n_frames = 30
        dt = 1.5
        # Single short spike then immediate decay → fast tail
        tissue = np.zeros(n_frames, dtype=np.float64)
        tissue[5] = 1.0  # peak
        tissue[6] = 0.05  # below 10% next frame
        n = select_n_control_points(
            2, infer_cpt=True, mean_tissue=tissue, dt=dt,
        )
        assert n == CPI_N_MIN  # 4
