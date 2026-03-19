"""Phase 3: Bolus arrival detection tests.

Tests detect_bolus_arrival() accuracy against known ground truth arrival
times. This function feeds n_baseline_frames into signal_to_concentration,
which feeds everything else.

Exercises:
    dsc_signal.py: detect_bolus_arrival()
"""

import numpy as np
import pytest

from pmriprep.utils.dsc_signal import detect_bolus_arrival

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
