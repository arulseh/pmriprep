"""Phase 3: Bolus arrival detection tests.

Tests detect_bolus_arrival() accuracy against known ground truth arrival
times. This function feeds n_baseline_frames into signal_to_concentration,
which feeds everything else.

Exercises:
    dsc_signal.py: detect_bolus_arrival()
"""

import numpy as np
import pytest

from pmriprep.utils.dsc_signal import (
    detect_bolus_arrival,
    detect_bolus_arrival_gradient,
)

from .conftest import (
    add_gaussian_noise,
    generate_dsc_aif_dro,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_concentration_with_arrival(
    baseline_frames: int = 10,
    n_timepoints: int = 60,
    dt_sec: float = 1.5,
    peak_delta_r2: float = 5.0,
    noise_frac: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Generate a concentration curve with known bolus arrival.

    Returns (concentration_curve, true_arrival_idx).
    """
    # Generate gamma-variate with arrival at baseline_frames
    aif, _ = generate_dsc_aif_dro(
        n_timepoints=n_timepoints - baseline_frames,
        dt_sec=dt_sec,
        baseline_frames=0,
        peak_delta_r2=peak_delta_r2,
    )

    # Prepend zero baseline
    curve = np.zeros(n_timepoints, dtype=np.float64)
    curve[baseline_frames:] = aif

    # Add noise
    if noise_frac > 0:
        peak = float(np.max(np.abs(curve)))
        if peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=seed)

    return curve, baseline_frames


# =============================================================================
# Test classes
# =============================================================================


class TestBolusArrivalDetection:
    """Test bolus arrival detection accuracy."""

    BASELINE_LENGTHS = [
        pytest.param(3, id="bl3"),
        pytest.param(5, id="bl5"),
        pytest.param(10, id="bl10"),
        pytest.param(15, id="bl15"),
        pytest.param(20, id="bl20"),
    ]

    SNRS_AS_NOISE_FRAC = [
        pytest.param(0.0, id="noiseless"),
        pytest.param(0.02, id="noise2pct"),
        pytest.param(0.05, id="noise5pct"),
        pytest.param(0.10, id="noise10pct"),
        pytest.param(0.20, id="noise20pct"),
    ]

    @pytest.mark.parametrize("bl_true", BASELINE_LENGTHS)
    @pytest.mark.parametrize("noise_frac", SNRS_AS_NOISE_FRAC)
    def test_arrival_detection_accuracy(
        self, bl_true: int, noise_frac: float,
    ) -> None:
        """Arrival index should be within +/-2 frames of ground truth."""
        curve, true_arrival = _make_concentration_with_arrival(
            baseline_frames=bl_true,
            noise_frac=noise_frac,
        )

        arrival_idx, n_baseline = detect_bolus_arrival(
            curve, baseline_frames=3,
        )

        # At moderate noise, allow +/-3 frames tolerance
        tol = 2 if noise_frac <= 0.05 else 3
        assert abs(arrival_idx - true_arrival) <= tol, (
            f"Arrival detected at {arrival_idx}, true={true_arrival}, "
            f"tolerance={tol} (noise_frac={noise_frac})"
        )

    @pytest.mark.parametrize("bl_true", BASELINE_LENGTHS)
    def test_n_baseline_consistency(self, bl_true: int) -> None:
        """n_baseline_frames should be consistent with arrival_idx."""
        curve, _ = _make_concentration_with_arrival(
            baseline_frames=bl_true, noise_frac=0.0,
        )

        arrival_idx, n_baseline = detect_bolus_arrival(curve, baseline_frames=3)

        # n_baseline should be >= 2 (function guarantees this)
        assert n_baseline >= 2
        # n_baseline should be <= arrival_idx
        assert n_baseline <= arrival_idx + 1

    @pytest.mark.parametrize("n_consecutive", [1, 2, 3])
    def test_n_consecutive_parameter(self, n_consecutive: int) -> None:
        """n_consecutive should not change result much for clean signals."""
        curve, true_arrival = _make_concentration_with_arrival(
            baseline_frames=10, noise_frac=0.0,
        )

        arrival_idx, _ = detect_bolus_arrival(
            curve, baseline_frames=3, n_consecutive=n_consecutive,
        )

        # Clean signal: n_consecutive shouldn't change result much
        assert abs(arrival_idx - true_arrival) <= 3

    def test_n_consecutive_rejects_spike(self) -> None:
        """n_consecutive >= 2 should reject isolated noise spikes."""
        # Create curve with a single-frame spike before real bolus
        curve = np.zeros(60, dtype=np.float64)
        # Spike at frame 5
        curve[5] = 10.0
        # Real bolus at frame 15
        curve[15:] = np.linspace(0, 20, 45)

        _, _ = detect_bolus_arrival(curve, baseline_frames=3, n_consecutive=1)
        arrival_2, _ = detect_bolus_arrival(
            curve, baseline_frames=3, n_consecutive=2,
        )

        # With n_consecutive=2, should skip the spike at frame 5
        # and find the real bolus around frame 15
        assert arrival_2 >= 10, (
            f"n_consecutive=2 should skip spike, got arrival={arrival_2}"
        )


class TestBolusArrivalEdgeCases:
    """Edge cases for bolus arrival detection."""

    def test_flat_line_no_bolus(self) -> None:
        """Flat line (no bolus) should return reasonable fallback."""
        curve = np.ones(60, dtype=np.float64) * 5.0
        arrival_idx, n_baseline = detect_bolus_arrival(curve, baseline_frames=3)

        # Should return fallback (n_baseline = baseline_frames)
        assert arrival_idx >= 2
        assert n_baseline >= 2

    def test_immediate_bolus(self) -> None:
        """Bolus starting at frame 1-2 should be detected."""
        curve = np.zeros(60, dtype=np.float64)
        # Bolus starts immediately after 2 frames
        curve[2:] = np.linspace(0, 30, 58)

        arrival_idx, n_baseline = detect_bolus_arrival(
            curve, baseline_frames=2,
        )

        assert arrival_idx <= 5
        assert n_baseline >= 2

    def test_gradual_onset(self) -> None:
        """Slow-rising bolus should still be detected."""
        curve = np.zeros(60, dtype=np.float64)
        # Very gradual onset starting at frame 10
        for i in range(10, 60):
            curve[i] = 0.2 * (i - 10) ** 1.5  # Gentle power law

        arrival_idx, _ = detect_bolus_arrival(curve, baseline_frames=3)

        # Should detect near the actual onset
        assert arrival_idx <= 25, (
            f"Gradual onset not detected early enough: {arrival_idx}"
        )

    def test_preloaded_elevated_baseline(self) -> None:
        """Preloaded baseline (elevated before bolus) should still work."""
        curve = np.zeros(60, dtype=np.float64)
        # Elevated baseline (preload)
        curve[:15] = 2.0
        # Bolus on top of preload at frame 15
        curve[15:] = 2.0 + np.linspace(0, 20, 45)

        arrival_idx, _ = detect_bolus_arrival(curve, baseline_frames=3)

        # Should detect near frame 15 despite elevated baseline
        assert arrival_idx <= 20

    def test_baseline_clipping(self) -> None:
        """baseline_frames=20 with n_timepoints=40 -> clipped to 10."""
        curve = np.zeros(40, dtype=np.float64)
        curve[10:] = np.linspace(0, 15, 30)

        arrival_idx, n_baseline = detect_bolus_arrival(
            curve, baseline_frames=20,
        )

        # n_baseline should be clipped (len//4 = 10)
        assert n_baseline <= 10

    def test_very_short_curve(self) -> None:
        """Very short curve should not crash."""
        curve = np.array([0.0, 0.0, 5.0, 10.0, 8.0], dtype=np.float64)

        arrival_idx, n_baseline = detect_bolus_arrival(
            curve, baseline_frames=2,
        )

        assert arrival_idx >= 0
        assert n_baseline >= 2

    def test_all_zero_curve(self) -> None:
        """All-zero curve should return reasonable fallback."""
        curve = np.zeros(60, dtype=np.float64)

        arrival_idx, n_baseline = detect_bolus_arrival(
            curve, baseline_frames=5,
        )

        assert arrival_idx >= 2
        assert n_baseline >= 2

    def test_negative_values_handled(self) -> None:
        """Concentration with some negative values should not crash."""
        curve = np.zeros(60, dtype=np.float64)
        curve[:10] = -0.5  # Slightly negative baseline
        curve[15:] = np.linspace(0, 20, 45)

        arrival_idx, n_baseline = detect_bolus_arrival(
            curve, baseline_frames=3,
        )

        assert arrival_idx >= 0
        assert n_baseline >= 2


# =============================================================================
# Gradient-based detection tests
# =============================================================================


def _make_preloaded_curve(
    n_timepoints: int = 120,
    dt_sec: float = 1.5,
    preload_amplitude: float = 2.0,
    washout_tau: float = 30.0,
    bolus_arrival: int = 42,
    peak_delta_r2: float = 15.0,
    noise_frac: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Generate a preloaded DSC concentration curve.

    Exponentially decaying preload baseline with a gamma-variate bolus
    superimposed at ``bolus_arrival``.

    Returns (concentration_curve, true_arrival_idx).
    """
    t = np.arange(n_timepoints) * dt_sec
    # Exponential washout from preload
    curve = preload_amplitude * np.exp(-t / washout_tau)

    # Add bolus at arrival
    bolus_part, _ = generate_dsc_aif_dro(
        n_timepoints=n_timepoints - bolus_arrival,
        dt_sec=dt_sec,
        baseline_frames=0,
        peak_delta_r2=peak_delta_r2,
    )
    curve[bolus_arrival:] += bolus_part

    if noise_frac > 0:
        peak = float(np.max(np.abs(curve)))
        if peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=seed)

    return curve, bolus_arrival


class TestBolusArrivalGradient:
    """Test gradient-based bolus arrival detection."""

    def test_preloaded_decaying_baseline(self) -> None:
        """Preloaded washout + bolus at frame 42 should be detected."""
        curve, true_arrival = _make_preloaded_curve(
            bolus_arrival=42, noise_frac=0.0,
        )

        arrival_idx, n_baseline = detect_bolus_arrival_gradient(
            curve, baseline_window=20, n_consecutive=2,
        )

        assert abs(arrival_idx - true_arrival) <= 2, (
            f"Arrival detected at {arrival_idx}, true={true_arrival}"
        )
        assert n_baseline >= 2

    def test_preloaded_noisy_washout(self) -> None:
        """Noisy preloaded washout should still detect real bolus."""
        curve, true_arrival = _make_preloaded_curve(
            bolus_arrival=42, noise_frac=0.05,
        )

        arrival_idx, n_baseline = detect_bolus_arrival_gradient(
            curve, baseline_window=20, n_consecutive=2,
        )

        assert abs(arrival_idx - true_arrival) <= 3, (
            f"Arrival detected at {arrival_idx}, true={true_arrival}"
        )

    def test_nonpreloaded_zero_baseline(self) -> None:
        """Flat-zero baseline with bolus — EPSILON floor handles sigma=0."""
        curve, true_arrival = _make_concentration_with_arrival(
            baseline_frames=10, noise_frac=0.0,
        )

        arrival_idx, n_baseline = detect_bolus_arrival_gradient(
            curve, baseline_window=10, n_consecutive=2,
        )

        assert abs(arrival_idx - true_arrival) <= 2
        assert n_baseline >= 2

    def test_nonpreloaded_matches_original(self) -> None:
        """Zero baseline: gradient method agrees with amplitude method."""
        curve, true_arrival = _make_concentration_with_arrival(
            baseline_frames=15, noise_frac=0.0,
        )

        arr_amp, _ = detect_bolus_arrival(curve, baseline_frames=3)
        arr_grad, _ = detect_bolus_arrival_gradient(
            curve, baseline_window=10, n_consecutive=2,
        )

        assert abs(arr_amp - arr_grad) <= 2, (
            f"Amplitude={arr_amp}, Gradient={arr_grad}"
        )

    def test_gradient_rejects_slow_drift(self) -> None:
        """Linear drift + bolus at frame 30 should detect ~30, not drift."""
        curve = np.zeros(80, dtype=np.float64)
        # Slow linear drift
        curve += np.linspace(0, 0.5, 80)
        # Real bolus at frame 30
        bolus, _ = generate_dsc_aif_dro(
            n_timepoints=50, dt_sec=1.5, baseline_frames=0, peak_delta_r2=10.0,
        )
        curve[30:] += bolus

        arrival_idx, _ = detect_bolus_arrival_gradient(
            curve, baseline_window=20, n_consecutive=2,
        )

        assert abs(arrival_idx - 30) <= 3, (
            f"Drift test: arrival={arrival_idx}, expected ~30"
        )

    def test_gradient_flat_line_fallback(self) -> None:
        """Flat input returns fallback without crash."""
        curve = np.ones(60, dtype=np.float64) * 5.0

        arrival_idx, n_baseline = detect_bolus_arrival_gradient(
            curve, baseline_window=15,
        )

        # Should return fallback index
        assert arrival_idx >= 0
        assert n_baseline >= 2

    def test_short_baseline_minimum_guard(self) -> None:
        """Very short curve triggers minimum sample guard."""
        curve = np.array([0.0, 1.0, 5.0, 10.0, 8.0, 3.0], dtype=np.float64)

        arrival_idx, n_baseline = detect_bolus_arrival_gradient(
            curve, baseline_window=10, n_consecutive=2,
        )

        # With 6 values, len//4 = 1 < 4 → minimum guard triggers fallback
        assert arrival_idx >= 0
        assert arrival_idx < len(curve)
        assert n_baseline >= 2

    @pytest.mark.parametrize(
        "tr,n_consec",
        [
            pytest.param(1.0, 3, id="TR1.0"),
            pytest.param(1.5, 2, id="TR1.5"),
            pytest.param(2.0, 2, id="TR2.0"),
            pytest.param(2.5, 1, id="TR2.5"),
        ],
    )
    def test_tr_sweep(self, tr: float, n_consec: int) -> None:
        """Detection works across TR range with adaptive n_consecutive."""
        n_tp = int(120 / tr)  # ~120 s acquisition
        bolus_frame = int(42 * 1.5 / tr)  # Same temporal position

        curve, true_arrival = _make_preloaded_curve(
            n_timepoints=n_tp, dt_sec=tr, bolus_arrival=bolus_frame,
        )

        arrival_idx, _ = detect_bolus_arrival_gradient(
            curve, baseline_window=min(20, n_tp // 4),
            n_consecutive=n_consec,
        )

        tol = max(2, int(3.0 / tr))  # Frame tolerance scales with TR
        assert abs(arrival_idx - true_arrival) <= tol, (
            f"TR={tr}: arrival={arrival_idx}, true={true_arrival}, tol={tol}"
        )

    def test_amplitude_validation_rejects_spurious(self) -> None:
        """Gradient spike without sustained concentration rise is rejected."""
        # Flat baseline with a single-frame gradient blip (no real bolus)
        curve = np.ones(60, dtype=np.float64) * 2.0
        # Tiny blip: +0.001 for 1 frame then back to baseline
        curve[20] = 2.001
        curve[21] = 2.001

        arrival_idx, n_baseline = detect_bolus_arrival_gradient(
            curve, baseline_window=15, n_consecutive=1,
        )

        # Should NOT detect frame 20 as bolus (amplitude check fails)
        # Should return fallback instead
        assert arrival_idx != 20, (
            "Spurious gradient blip should be rejected by amplitude check"
        )


class TestFWHMMinimumGuard:
    """Test that the FWHM minimum guard rejects sub-TR noise spikes."""

    def test_fwhm_below_minimum_rejected(self) -> None:
        """FWHM = 1.5s (1 TR at TR=1.5s) should be rejected at all levels."""
        from pmriprep.interfaces.dsc_aif import AIFDetection

        detector = AIFDetection.__new__(AIFDetection)

        metrics = {
            "ttp": 10.0,
            "ph": 50.0,
            "psr": 90.0,
            "s0": 100.0,
            "peak_conc": 20.0,
            "baseline_return_ratio": 10.0,
            "peak_to_baseline_ratio": 5.0,
            "secondary_peak_ratio": 30.0,
            "post_peak_monotonicity": 80.0,
            "fwhm": 1.5,  # 1 TR at TR=1.5s — too narrow
            "baseline_conc": 0.1,
            "dt": 1.5,
            "pre_bolus_frames": 5,
            "scan_duration": 180.0,
            "max_allowed_ttp": 144.0,
        }

        for criteria in ["strict", "relaxed", "permissive", "degraded"]:
            cost = detector._score_candidate_cost(
                metrics, criteria, median_s0=None,
            )
            assert cost == float("inf"), (
                f"FWHM=1.5s should be rejected at '{criteria}' level, "
                f"got cost={cost}"
            )
