"""Tests for ensemble baseline detection.

Exercises the 5 individual detectors and the ensemble consensus logic
on synthetic bolus curves with known ground truth arrival times.
"""

import numpy as np
import pytest

from pmriprep.utils.baseline_detection import (
    _detect_amplitude_zscore,
    _detect_cusum,
    _detect_gradient_zscore,
    _detect_max_slope_backtrack,
    _detect_variance_ratio,
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
    """Generate a synthetic bolus curve with known arrival.

    Uses a sharp logistic rise to simulate realistic bolus onset
    (steeper than a gamma-variate's gradual tail).

    Returns (signal, true_arrival_frame).
    """
    signal = np.zeros(n, dtype=np.float64)

    # Sharp logistic rise at baseline_frames, then exponential decay
    for i in range(baseline_frames, n):
        t_rel = i - baseline_frames
        rise = 1.0 / (1.0 + np.exp(-(t_rel - 2.0) / 0.8))
        decay = np.exp(-max(0, t_rel - 4) / 8.0)
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
    """Generate a preloaded concentration curve (exponential washout + bolus)."""
    t = np.arange(n, dtype=np.float64) * 1.5  # TR=1.5s
    signal = preload_amp * np.exp(-t / washout_tau)

    # Sharp logistic bolus at baseline_frames
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
# Individual detector tests
# =============================================================================


class TestIndividualDetectors:
    """Each detector should find the correct arrival on clean data."""

    @pytest.mark.parametrize(
        "detector,name",
        [
            pytest.param(
                lambda s, bl, nc: _detect_amplitude_zscore(s, bl, nc),
                "amplitude",
                id="A",
            ),
            pytest.param(
                lambda s, bl, nc: _detect_gradient_zscore(s, bl, nc),
                "gradient",
                id="B",
            ),
            pytest.param(
                lambda s, bl, _: _detect_cusum(s, bl),
                "cusum",
                id="C",
            ),
            pytest.param(
                lambda s, bl, _: _detect_variance_ratio(s, bl),
                "variance",
                id="D",
            ),
            pytest.param(
                lambda s, _, __: _detect_max_slope_backtrack(s),
                "maxslope",
                id="E",
            ),
        ],
    )
    def test_clean_bolus(self, detector, name) -> None:
        """Detector finds arrival within ±4 frames on clean data.

        Individual detectors may be ±4 frames off — the ensemble
        consensus corrects for this via weighted median.
        """
        signal, true_arrival = _make_bolus_curve(baseline_frames=15)
        result = detector(signal, 10, 2)
        assert result is not None, f"{name} returned None"
        assert abs(result - true_arrival) <= 4, (
            f"{name}: detected {result}, true={true_arrival}"
        )

    @pytest.mark.parametrize(
        "detector",
        [
            _detect_amplitude_zscore,
            _detect_gradient_zscore,
            _detect_cusum,
            _detect_variance_ratio,
            _detect_max_slope_backtrack,
        ],
    )
    def test_constant_returns_none(self, detector) -> None:
        """Constant signal should return None."""
        signal = np.ones(60, dtype=np.float64) * 5.0
        if detector in (_detect_amplitude_zscore, _detect_gradient_zscore):
            result = detector(signal, 10, 2)
        elif detector == _detect_max_slope_backtrack:
            result = detector(signal)
        else:
            result = detector(signal, 10)
        # Either None or a fallback index — should NOT be < 10 (false trigger)
        if result is not None:
            assert result >= 10 or result >= len(signal) - 1


# =============================================================================
# Ensemble consensus tests
# =============================================================================


class TestEnsembleConsensus:
    """Tests for the ensemble consensus logic."""

    def test_clean_data_all_agree(self) -> None:
        """On clean data, all methods agree and consensus matches."""
        signal, true_arrival = _make_bolus_curve(baseline_frames=15)
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)

        assert abs(arrival - true_arrival) <= 2, (
            f"Ensemble: {arrival}, true={true_arrival}, diag={diag}"
        )
        assert diag["n_valid_methods"] >= 4

    def test_preloaded_washout(self) -> None:
        """Preloaded data: amplitude z-score may fail, ensemble correct."""
        signal, true_arrival = _make_preloaded_curve(baseline_frames=42)
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)

        assert abs(arrival - true_arrival) <= 3, (
            f"Preloaded: {arrival}, true={true_arrival}, diag={diag}"
        )

    def test_preloaded_noisy(self) -> None:
        """Noisy preloaded data still detects correctly."""
        signal, true_arrival = _make_preloaded_curve(
            baseline_frames=42, noise_std=0.3,
        )
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)

        assert abs(arrival - true_arrival) <= 4, (
            f"Noisy preloaded: {arrival}, true={true_arrival}, diag={diag}"
        )

    def test_conservative_bias(self) -> None:
        """Low agreement should take the latest (most conservative) estimate."""
        # Use a signal where the bolus onset is ambiguous
        signal = np.zeros(80, dtype=np.float64)
        # Very gradual onset
        signal[20:] = np.linspace(0, 0.5, 60)
        signal[40:] += np.linspace(0, 10, 40)

        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)

        # With ambiguous onset, should err on the side of longer baseline
        assert arrival >= 15, (
            f"Conservative bias failed: {arrival}, diag={diag}"
        )

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
        """Works across TR range."""
        n = int(120 / tr)
        signal, true_arrival = _make_bolus_curve(
            n=n, baseline_frames=bl, peak=10.0,
        )
        arrival, diag = detect_baseline_ensemble(
            signal, tr=tr, min_baseline=3,
        )

        tol = max(2, int(3.0 / tr))
        assert abs(arrival - true_arrival) <= tol, (
            f"TR={tr}: {arrival}, true={true_arrival}, diag={diag}"
        )

    def test_short_baseline(self) -> None:
        """Very short signal returns min_baseline fallback."""
        signal = np.array([0.0, 0.0, 5.0, 10.0, 8.0], dtype=np.float64)
        arrival, diag = detect_baseline_ensemble(
            signal, tr=1.5, min_baseline=3,
        )
        assert arrival == 3
        assert "fallback" in diag.get("method", "")

    def test_long_baseline_preloaded(self) -> None:
        """40+ frame baseline with preload correctly detected."""
        signal, true_arrival = _make_preloaded_curve(
            n=120, baseline_frames=45,
        )
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)

        assert abs(arrival - true_arrival) <= 3, (
            f"Long baseline: {arrival}, true={true_arrival}, diag={diag}"
        )


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Edge cases that should not crash."""

    def test_all_nan(self) -> None:
        """All-NaN signal returns fallback."""
        signal = np.full(60, np.nan)
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=5)
        assert arrival == 5
        assert "fallback" in diag.get("method", "")

    def test_all_constant(self) -> None:
        """Constant signal returns fallback."""
        signal = np.ones(60, dtype=np.float64) * 100.0
        arrival, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=5)
        assert arrival == 5
        assert "fallback" in diag.get("method", "")

    def test_signal_shorter_than_min_baseline(self) -> None:
        """Signal shorter than min_baseline returns fallback."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        arrival, diag = detect_baseline_ensemble(
            signal, tr=1.5, min_baseline=5,
        )
        assert arrival == 5

    def test_diagnostics_keys(self) -> None:
        """Diagnostics dict has expected keys."""
        signal, _ = _make_bolus_curve(baseline_frames=10)
        _, diag = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)

        assert "per_method" in diag
        assert "n_valid_methods" in diag
        assert "method" in diag
        assert "arrival_frame" in diag


# =============================================================================
# Regression: ensemble matches existing detectors on clean data
# =============================================================================


class TestRegressionAgainstExisting:
    """Ensemble should agree with existing methods on standard data."""

    @pytest.mark.parametrize("bl", [5, 10, 15, 20])
    def test_matches_amplitude_zscore(self, bl: int) -> None:
        """On clean non-preloaded data, ensemble ≈ amplitude z-score."""
        from pmriprep.utils.dsc_signal import detect_bolus_arrival

        signal, _ = _make_bolus_curve(baseline_frames=bl)

        arr_amp, _ = detect_bolus_arrival(signal, baseline_frames=3)
        arr_ens, _ = detect_baseline_ensemble(signal, tr=1.5, min_baseline=3)

        assert abs(arr_amp - arr_ens) <= 2, (
            f"Amplitude={arr_amp}, Ensemble={arr_ens}"
        )
