"""Tests for derivative-anchored baseline detection.

Tests the Phase 1 (derivative anchor) + Phase 2 (z-score refinement) +
Phase 3 (multi-signal consensus) + Phase 4 (confidence) architecture.
"""

import numpy as np
import pytest

from pmriprep.utils.baseline_detection import (
    _detect_amplitude_zscore,
    _detect_cusum,
    _detect_gradient_zscore,
    _detect_max_slope_backtrack,
    _detect_second_derivative_peak,
    _detect_single_signal,
    detect_baseline_ensemble,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_bolus_curve(
    n: int = 80,
    baseline_frames: int = 15,
    peak: float = 10.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Synthetic bolus curve with sharp logistic rise."""
    signal = np.zeros(n, dtype=np.float64)
    for i in range(baseline_frames, n):
        t = i - baseline_frames
        rise = 1.0 / (1.0 + np.exp(-(t - 2.0) / 0.8))
        decay = np.exp(-max(0, t - 4) / 8.0)
        signal[i] = peak * rise * decay
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        signal += rng.normal(0, noise_std, n)
    return signal, baseline_frames


def _make_preloaded_curve(
    n: int = 120,
    baseline_frames: int = 42,
    preload_amp: float = 2.0,
    washout_tau: float = 30.0,
    peak: float = 15.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Preloaded curve: exponential washout + sharp bolus."""
    t = np.arange(n, dtype=np.float64) * 1.5
    signal = preload_amp * np.exp(-t / washout_tau)
    for i in range(baseline_frames, n):
        t_rel = i - baseline_frames
        rise = 1.0 / (1.0 + np.exp(-(t_rel - 2.0) / 0.8))
        decay = np.exp(-max(0, t_rel - 4) / 8.0)
        signal[i] += peak * rise * decay
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        signal += rng.normal(0, noise_std, n)
    return signal, baseline_frames


# =============================================================================
# Phase 1: Individual detectors
# =============================================================================


class TestPhase1Detectors:
    """Phase 1 derivative-based detectors."""

    def test_max_slope_clean(self) -> None:
        signal, true = _make_bolus_curve(baseline_frames=15)
        result = _detect_max_slope_backtrack(signal, min_baseline=3)
        assert result is not None
        assert abs(result - true) <= 4

    def test_max_slope_preloaded(self) -> None:
        signal, true = _make_preloaded_curve(baseline_frames=42)
        result = _detect_max_slope_backtrack(signal, min_baseline=5)
        assert result is not None
        assert abs(result - true) <= 3

    def test_max_slope_constant_returns_none(self) -> None:
        signal = np.ones(60) * 5.0
        assert _detect_max_slope_backtrack(signal) is None

    def test_second_deriv_clean(self) -> None:
        signal, true = _make_bolus_curve(baseline_frames=15)
        result = _detect_second_derivative_peak(signal, min_baseline=3)
        assert result is not None
        assert abs(result - true) <= 4

    def test_second_deriv_constant_returns_none(self) -> None:
        signal = np.ones(60) * 5.0
        assert _detect_second_derivative_peak(signal) is None


# =============================================================================
# Phase 2: Z-score detectors with derived baseline_window
# =============================================================================


class TestPhase2Detectors:
    """Phase 2 z-score detectors with anchor-derived baseline_window."""

    def test_amplitude_zscore(self) -> None:
        signal, true = _make_bolus_curve(baseline_frames=15)
        result = _detect_amplitude_zscore(signal, baseline_window=12, n_consecutive=2)
        assert result is not None
        assert abs(result - true) <= 4

    def test_gradient_zscore(self) -> None:
        signal, true = _make_bolus_curve(baseline_frames=15)
        result = _detect_gradient_zscore(signal, baseline_window=12, n_consecutive=2)
        assert result is not None
        assert abs(result - true) <= 4

    def test_cusum(self) -> None:
        signal, true = _make_bolus_curve(baseline_frames=15)
        result = _detect_cusum(signal, baseline_window=12)
        assert result is not None
        assert abs(result - true) <= 4


# =============================================================================
# Single-signal Phase 1+2
# =============================================================================


class TestSingleSignal:
    """_detect_single_signal combines Phase 1 + Phase 2."""

    def test_clean_bolus(self) -> None:
        signal, true = _make_bolus_curve(baseline_frames=15)
        arrival, conf, diag = _detect_single_signal(signal, tr=1.5, min_baseline=3)
        assert abs(arrival - true) <= 2
        assert conf in ("high", "medium")

    def test_preloaded(self) -> None:
        signal, true = _make_preloaded_curve(baseline_frames=42)
        arrival, conf, diag = _detect_single_signal(signal, tr=1.5, min_baseline=5)
        assert abs(arrival - true) <= 3
        assert conf in ("high", "medium")

    def test_tr_adaptive_skips_d2_at_long_tr(self) -> None:
        signal, _ = _make_bolus_curve(n=20, baseline_frames=8)
        _, _, diag = _detect_single_signal(signal, tr=7.0, min_baseline=3)
        assert diag["phase1_tr_rule"] == "slope_only"

    def test_tr_adaptive_uses_d2_at_short_tr(self) -> None:
        signal, _ = _make_bolus_curve(baseline_frames=15)
        _, _, diag = _detect_single_signal(signal, tr=1.5, min_baseline=3)
        assert diag["phase1_tr_rule"] == "min"


# =============================================================================
# Ensemble: full pipeline
# =============================================================================


class TestEnsemble:
    """Full detect_baseline_ensemble tests."""

    def test_clean_data(self) -> None:
        signal, true = _make_bolus_curve(baseline_frames=15)
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)
        assert abs(arrival - true) <= 2
        assert diag["confidence"] in ("high", "medium")

    def test_preloaded_washout(self) -> None:
        signal, true = _make_preloaded_curve(baseline_frames=42)
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=5)
        assert abs(arrival - true) <= 3

    def test_preloaded_noisy(self) -> None:
        signal, true = _make_preloaded_curve(baseline_frames=42, noise_std=0.3)
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=5)
        assert abs(arrival - true) <= 4

    @pytest.mark.parametrize(
        "tr,bl",
        [
            pytest.param(1.0, 20, id="TR1.0"),
            pytest.param(1.5, 15, id="TR1.5"),
            pytest.param(2.0, 12, id="TR2.0"),
            pytest.param(2.5, 10, id="TR2.5"),
            pytest.param(7.0, 8, id="TR7.0"),
            pytest.param(7.5, 7, id="TR7.5"),
        ],
    )
    def test_tr_sweep(self, tr: float, bl: int) -> None:
        n = int(120 / tr)
        signal, true = _make_bolus_curve(n=n, baseline_frames=bl, peak=10.0)
        arrival, diag = detect_baseline_ensemble(signal, tr=tr, min_baseline=3)
        tol = max(2, int(3.0 / tr))
        assert abs(arrival - true) <= tol, (
            f"TR={tr}: arrival={arrival}, true={true}, diag={diag}"
        )

    def test_multi_signal_consensus(self) -> None:
        """Phase 3: multiple signals should converge."""
        signal1, true = _make_bolus_curve(baseline_frames=15)
        signal2, _ = _make_bolus_curve(baseline_frames=15, peak=8.0, seed=99)
        signal3, _ = _make_bolus_curve(baseline_frames=15, peak=12.0, seed=123)

        arrival, diag = detect_baseline_ensemble(
            signal1, tr=1.5, min_baseline=3,
            additional_signals=[signal2, signal3],
        )
        assert abs(arrival - true) <= 2
        assert "phase3_consensus" in diag

    def test_confidence_high(self) -> None:
        signal, _ = _make_bolus_curve(baseline_frames=15)
        _, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)
        assert diag["confidence"] == "high"

    def test_confidence_in_diagnostics(self) -> None:
        signal, _ = _make_bolus_curve(baseline_frames=10)
        _, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)
        assert "confidence" in diag
        assert diag["confidence"] in ("high", "medium", "low")
        assert "primary" in diag
        assert "arrival_frame" in diag


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:

    def test_all_nan(self) -> None:
        signal = np.full(60, np.nan)
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=5)
        assert arrival == 5
        assert diag["confidence"] == "low"

    def test_all_constant(self) -> None:
        signal = np.ones(60) * 100.0
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=5)
        assert arrival == 5
        assert diag["confidence"] == "low"

    def test_short_signal(self) -> None:
        signal = np.array([1.0, 2.0, 3.0])
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=5)
        assert arrival == 5

    def test_long_baseline_preloaded(self) -> None:
        signal, true = _make_preloaded_curve(n=120, baseline_frames=45)
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=5)
        assert abs(arrival - true) <= 3

    def test_motion_spike_rejected(self) -> None:
        """Single-frame motion spike should be smoothed out."""
        signal = np.zeros(80)
        signal[10] = 5.0  # moderate motion spike (single frame)
        # Real bolus at frame 30 (sharp, sustained)
        for i in range(30, 80):
            t = i - 30
            rise = 1.0 / (1.0 + np.exp(-(t - 2.0) / 0.8))
            decay = np.exp(-max(0, t - 4) / 8.0)
            signal[i] = 15.0 * rise * decay
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)
        assert arrival >= 25, (
            f"Motion spike not rejected: arrival={arrival}, diag={diag}"
        )


# =============================================================================
# Regression: matches existing methods on clean data
# =============================================================================


class TestRegression:

    @pytest.mark.parametrize("bl", [5, 10, 15, 20])
    def test_matches_amplitude_zscore(self, bl: int) -> None:
        from pmriprep.utils.dsc_signal import detect_bolus_arrival

        signal, _ = _make_bolus_curve(baseline_frames=bl)
        arr_amp, _ = detect_bolus_arrival(signal, baseline_frames=3)
        arr_ens, _ = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)
        assert abs(arr_amp - arr_ens) <= 3, (
            f"Amplitude={arr_amp}, Ensemble={arr_ens}"
        )


# =============================================================================
# Short Baseline Magnitude Guard (sub-0092 regression fix)
# =============================================================================


class TestShortBaselineMagnitudeGuard:
    """Tests for magnitude guard on Phase 2 earlier refinement.

    Prevents false early detection when z-score std is unreliable
    (short baselines) and signal drift is small (<1.5%).
    """

    def test_short_baseline_with_drift_no_false_early(self) -> None:
        """OSIPI sub-0092 scenario: 8 baseline frames, 0.7% drift, bolus at frame 8.

        Should detect frame 7-8, NOT frame 5-6.
        """
        n = 20
        signal = np.zeros(n, dtype=np.float64)
        # Baseline with small upward drift (0.5-1% over 8 frames)
        baseline_level = 100.0
        for i in range(8):
            signal[i] = baseline_level + 0.08 * i  # ~0.08% per frame
        # Bolus at frame 8+
        for i in range(8, n):
            t = i - 8
            rise = 1.0 / (1.0 + np.exp(-(t - 1.0) / 0.5))
            decay = np.exp(-max(0, t - 3) / 6.0)
            signal[i] = baseline_level + 0.64 + 30.0 * rise * decay

        arrival, diag = detect_baseline_ensemble(signal, tr=7.5, min_baseline=3)
        # Should detect at frame 7 or 8 (derivative anchor), NOT 5-6
        assert arrival >= 7, (
            f"arrival={arrival}, expected >=7. "
            f"action={diag.get('primary', {}).get('phase2_action')}"
        )

    def test_magnitude_guard_rejects_low_change(self) -> None:
        """Verify the magnitude guard rejects <1.5% earlier detection."""
        n = 30
        signal = np.zeros(n, dtype=np.float64)
        baseline_level = 50.0
        # Baseline with tiny drift at frame 8 (~0.5%)
        for i in range(12):
            signal[i] = baseline_level + 0.03 * i
        # Sharp bolus at frame 12
        for i in range(12, n):
            t = i - 12
            rise = 1.0 / (1.0 + np.exp(-(t - 1.0) / 0.5))
            decay = np.exp(-max(0, t - 3) / 6.0)
            signal[i] = baseline_level + 0.36 + 20.0 * rise * decay

        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)
        action = diag.get("primary", {}).get("phase2_action", "")
        # Should NOT have refined_earlier (magnitude too low or skipped)
        assert "refined_earlier_to" not in action, (
            f"Should not refine earlier, got action={action}"
        )

    def test_genuine_onset_passes_magnitude_guard(self) -> None:
        """Gradual onset with >1.5% enhancement should still be detected early."""
        n = 30
        signal = np.zeros(n, dtype=np.float64)
        baseline_level = 50.0
        for i in range(10):
            signal[i] = baseline_level
        # Gradual onset starting frame 10: 3% enhancement (above 1.5% guard)
        signal[10] = baseline_level * 1.03
        signal[11] = baseline_level * 1.06
        # Then sharp bolus
        for i in range(12, n):
            t = i - 12
            rise = 1.0 / (1.0 + np.exp(-(t - 0.5) / 0.3))
            signal[i] = baseline_level * 1.06 + 25.0 * rise

        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)
        # Should detect early onset (frame 10-11), not wait for frame 12
        assert arrival <= 12, (
            f"arrival={arrival}, expected <=12 (gradual onset should be caught)"
        )

    def test_long_baseline_unchanged(self) -> None:
        """Long baseline (40+ frames) should work identically to before."""
        signal, expected_bl = _make_bolus_curve(
            n=120, baseline_frames=42, peak=15.0
        )
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)
        assert abs(arrival - expected_bl) <= 2, (
            f"arrival={arrival}, expected ~{expected_bl}"
        )
