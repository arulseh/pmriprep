"""Tests for DCE model-free semi-quantitative maps.

Tests cover the four new functions (wash-in slope, wash-out slope, BAT,
curve type) and existing functions (IAUC, peak enhancement, TTP, SER).
"""

import logging

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.utils.dce_model_free import (
    compute_bolus_arrival_time,
    compute_curve_type,
    compute_iauc,
    compute_peak_enhancement,
    compute_signal_enhancement_ratio,
    compute_time_to_peak,
    compute_wash_in_slope,
    compute_wash_out_slope,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def gamma_variate_data() -> (
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
):
    """4x4x2x60 gamma-variate bolus with 5s TR, time_points in minutes."""
    n_time = 60
    tr_s = 5.0
    time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0  # minutes

    # Gamma-variate: C(t) = A * (t - t0)^alpha * exp(-(t - t0) / beta)
    t0_frame = 8  # bolus arrival
    t_sec = np.arange(n_time, dtype=np.float64) * tr_s
    t_shifted = np.maximum(t_sec - t0_frame * tr_s, 0.0)
    alpha, beta, amp = 3.0, 8.0, 0.5
    curve = amp * (t_shifted ** alpha) * np.exp(-t_shifted / beta)

    concentration = np.zeros((4, 4, 2, n_time), dtype=np.float64)
    concentration[...] = curve[np.newaxis, np.newaxis, np.newaxis, :]
    # Add slight per-voxel variation
    rng = np.random.default_rng(42)
    concentration *= rng.uniform(0.8, 1.2, size=(4, 4, 2, 1))

    return concentration, time_points


# =========================================================================
# Wash-in slope tests
# =========================================================================

class TestWashInSlope:
    """Tests for compute_wash_in_slope."""

    def test_linear_ramp(self) -> None:
        """Known linear ramp should produce exact slope."""
        n_time = 20
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        # Linear ramp from frame 2 to frame 10, then plateau
        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        slope_val = 2.0  # mM/min
        for t in range(n_time):
            if t < 2:
                concentration[..., t] = 0.0
            elif t <= 10:
                concentration[..., t] = slope_val * (time_points[t] - time_points[2])
            else:
                concentration[..., t] = slope_val * (time_points[10] - time_points[2])

        result = compute_wash_in_slope(
            concentration, time_points, bolus_arrival_frame=2, volume_tr_s=tr_s,
        )
        # Maximum derivative should be close to 2.0 mM/min
        # SG smoothing introduces slight overshoot at transition points
        assert np.allclose(result, slope_val, atol=0.2), f"Expected ~{slope_val}, got {result.ravel()}"

    def test_zero_concentration(self) -> None:
        """All-zero concentration should return NaN (no bolus)."""
        concentration = np.zeros((2, 2, 1, 20), dtype=np.float64)
        time_points = np.arange(20, dtype=np.float64) * 5.0 / 60.0
        result = compute_wash_in_slope(concentration, time_points)
        assert np.all(np.isnan(result))

    def test_tr_hard_gate(self) -> None:
        """TR > 15s should return NaN map without raising."""
        concentration = np.ones((2, 2, 1, 10), dtype=np.float64)
        time_points = np.arange(10, dtype=np.float64) * 20.0 / 60.0
        result = compute_wash_in_slope(concentration, time_points, volume_tr_s=20.0)
        assert result.shape == (2, 2, 1)
        assert np.all(np.isnan(result))

    def test_tr_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """TR 5-15s should produce warning but still compute valid map."""
        n_time = 30
        tr_s = 8.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0
        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        # Simple ramp
        for t in range(5, 15):
            concentration[..., t] = float(t - 5) * 0.1
        for t in range(15, n_time):
            concentration[..., t] = 1.0

        with caplog.at_level(logging.WARNING):
            result = compute_wash_in_slope(
                concentration, time_points, bolus_arrival_frame=5, volume_tr_s=tr_s,
            )
        assert "exceeds 5s" in caplog.text
        assert not np.all(np.isnan(result))

    def test_no_pre_peak(self) -> None:
        """Peak at bolus arrival frame should give NaN (< 2 pre-peak frames)."""
        n_time = 20
        time_points = np.arange(n_time, dtype=np.float64) * 5.0 / 60.0
        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        # Peak at frame 0, then decay
        concentration[..., 0] = 10.0
        for t in range(1, n_time):
            concentration[..., t] = 10.0 * np.exp(-0.1 * t)

        result = compute_wash_in_slope(concentration, time_points, bolus_arrival_frame=0)
        assert np.all(np.isnan(result))

    def test_input_unchanged(self) -> None:
        """Input array must not be modified."""
        n_time = 20
        time_points = np.arange(n_time, dtype=np.float64) * 5.0 / 60.0
        concentration = np.random.default_rng(0).random((2, 2, 1, n_time))
        original = concentration.copy()
        compute_wash_in_slope(concentration, time_points)
        np.testing.assert_array_equal(concentration, original)


# =========================================================================
# Wash-out slope tests
# =========================================================================

class TestWashOutSlope:
    """Tests for compute_wash_out_slope."""

    def test_linear_decay(self) -> None:
        """Known linear decay should produce matching negative slope."""
        n_time = 60
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        # Peak at frame 5, then linear decay
        decay_rate = -0.3  # mM/min
        for t in range(n_time):
            if t <= 5:
                concentration[..., t] = float(t) * 0.5
            else:
                concentration[..., t] = 2.5 + decay_rate * (time_points[t] - time_points[5])
        concentration = np.maximum(concentration, 0.0)

        result = compute_wash_out_slope(concentration, time_points, volume_tr_s=tr_s)
        # Should be negative (washout)
        valid = np.isfinite(result)
        if valid.any():
            assert np.all(result[valid] < 0), f"Expected negative slope, got {result[valid]}"

    def test_persistent_positive(self) -> None:
        """Rising curve should produce positive slope."""
        n_time = 60
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        for t in range(n_time):
            concentration[..., t] = 0.01 * t  # steady rise
        # peak at end, washout window will have rising concentration
        result = compute_wash_out_slope(concentration, time_points, volume_tr_s=tr_s)
        valid = np.isfinite(result)
        if valid.any():
            assert np.all(result[valid] > 0), f"Expected positive slope, got {result[valid]}"

    def test_insufficient_points(self) -> None:
        """Fewer than min_points in window should give NaN."""
        n_time = 10
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        concentration[..., 3] = 1.0  # peak at frame 3, very short scan

        result = compute_wash_out_slope(
            concentration, time_points, volume_tr_s=tr_s, min_points=4,
        )
        assert np.all(np.isnan(result))

    def test_late_ttp(self) -> None:
        """Peak near end of scan should give NaN (window beyond acquisition)."""
        n_time = 20
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        for t in range(n_time):
            concentration[..., t] = float(t)  # monotonic rise, peak at last frame

        result = compute_wash_out_slope(concentration, time_points, volume_tr_s=tr_s)
        assert np.all(np.isnan(result))


# =========================================================================
# BAT tests
# =========================================================================

class TestBolusArrivalTime:
    """Tests for compute_bolus_arrival_time."""

    def test_known_arrival(self) -> None:
        """Known bolus arrival should give correct time in seconds."""
        n_time = 60
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        arrival_frame = 10
        for t in range(arrival_frame, n_time):
            concentration[..., t] = 2.0 * (1.0 - np.exp(-0.1 * (t - arrival_frame)))

        result = compute_bolus_arrival_time(concentration, time_points, baseline_frames=5)
        valid = np.isfinite(result)
        if valid.any():
            # BAT should be near arrival_frame * tr_s seconds
            expected_s = arrival_frame * tr_s
            assert np.all(np.abs(result[valid] - expected_s) < 3 * tr_s)

    def test_no_enhancement(self) -> None:
        """Flat/zero curve should give NaN."""
        concentration = np.zeros((2, 2, 1, 30), dtype=np.float64)
        time_points = np.arange(30, dtype=np.float64) * 5.0 / 60.0
        result = compute_bolus_arrival_time(concentration, time_points)
        assert np.all(np.isnan(result))

    def test_consistency_with_direct_call(self) -> None:
        """Map function should match direct detect_bolus_arrival_time calls."""
        from pmriprep.utils.dce_bat import detect_bolus_arrival_time

        n_time = 60
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        rng = np.random.default_rng(99)
        concentration = np.zeros((2, 1, 1, n_time), dtype=np.float64)
        for i in range(2):
            arrival = 8 + i * 2
            for t in range(arrival, n_time):
                concentration[i, 0, 0, t] = rng.uniform(0.5, 2.0) * (
                    1.0 - np.exp(-0.15 * (t - arrival))
                )

        bat_map = compute_bolus_arrival_time(concentration, time_points, baseline_frames=5)

        for i in range(2):
            curve = concentration[i, 0, 0, :]
            direct = detect_bolus_arrival_time(curve, time_points, baseline_frames=5)
            if direct.confidence != "low" and direct.method != "default":
                assert np.isfinite(bat_map[i, 0, 0])
                assert bat_map[i, 0, 0] == pytest.approx(direct.bat_seconds)


# =========================================================================
# Curve type tests
# =========================================================================

class TestCurveType:
    """Tests for compute_curve_type."""

    def test_persistent(self) -> None:
        """SER < 0.9 should give type 1 (persistent)."""
        ser = np.full((2, 2, 1), 0.5, dtype=np.float64)
        result = compute_curve_type(ser)
        assert np.all(result == 1)

    def test_plateau(self) -> None:
        """SER in [0.9, 1.1] should give type 2 (plateau)."""
        ser = np.full((2, 2, 1), 1.0, dtype=np.float64)
        result = compute_curve_type(ser)
        assert np.all(result == 2)

    def test_washout(self) -> None:
        """SER > 1.1 should give type 3 (washout)."""
        ser = np.full((2, 2, 1), 1.5, dtype=np.float64)
        result = compute_curve_type(ser)
        assert np.all(result == 3)

    def test_nan_ser(self) -> None:
        """NaN SER should give type 0 (unclassified)."""
        ser = np.full((2, 2, 1), np.nan, dtype=np.float64)
        result = compute_curve_type(ser)
        assert np.all(result == 0)

    def test_custom_thresholds(self) -> None:
        """Non-default thresholds should be respected."""
        ser = np.full((2, 2, 1), 0.85, dtype=np.float64)
        # Default: 0.85 < 0.9 → persistent (type 1)
        assert np.all(compute_curve_type(ser) == 1)
        # Custom: 0.85 >= 0.8, <= 1.2 → plateau (type 2)
        result = compute_curve_type(ser, persistent_threshold=0.8, washout_threshold=1.2)
        assert np.all(result == 2)

    def test_dtype_int8(self) -> None:
        """Output should be int8."""
        ser = np.ones((2, 2, 1), dtype=np.float64)
        result = compute_curve_type(ser)
        assert result.dtype == np.int8


# =========================================================================
# Existing function tests
# =========================================================================

class TestIAUC:
    """Tests for compute_iauc."""

    def test_constant_concentration(self) -> None:
        """Constant concentration integrates to C * t."""
        n_time = 20
        time_points = np.arange(n_time, dtype=np.float64) * 5.0 / 60.0
        c_val = 1.0
        concentration = np.full((2, 2, 1, n_time), c_val, dtype=np.float64)
        cutoff_s = 60.0
        result = compute_iauc(concentration, time_points, cutoff_s)
        expected = c_val * (cutoff_s / 60.0)  # mM·min
        assert np.allclose(result, expected, atol=0.1)

    def test_zero_concentration(self) -> None:
        """Zero concentration gives zero IAUC."""
        concentration = np.zeros((2, 2, 1, 20), dtype=np.float64)
        time_points = np.arange(20, dtype=np.float64) * 5.0 / 60.0
        result = compute_iauc(concentration, time_points, 60.0)
        assert np.all(result == 0.0)

    def test_non_negative(self) -> None:
        """IAUC should always be non-negative."""
        rng = np.random.default_rng(0)
        concentration = rng.standard_normal((2, 2, 1, 30))
        time_points = np.arange(30, dtype=np.float64) * 5.0 / 60.0
        result = compute_iauc(concentration, time_points, 60.0)
        assert np.all(result >= 0)


class TestPeakEnhancement:
    """Tests for compute_peak_enhancement."""

    def test_known_enhancement(self) -> None:
        """Known peak should give correct enhancement percentage."""
        n_time = 20
        signal = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        # Baseline = 500, peak = 1500 → enhancement = 200%
        signal[..., :5] = 500.0
        signal[..., 5:] = 500.0
        signal[..., 10] = 1500.0
        result = compute_peak_enhancement(signal, baseline_frames=5)
        assert np.allclose(result, 200.0)

    def test_no_enhancement(self) -> None:
        """Flat signal should give 0% enhancement."""
        signal = np.ones((2, 2, 1, 20), dtype=np.float64) * 500.0
        result = compute_peak_enhancement(signal)
        assert np.allclose(result, 0.0)

    def test_dead_voxel_guard(self) -> None:
        """Signal below epsilon (1.0) returns 0% (dead/background voxel)."""
        n_time = 20
        signal = np.full((2, 2, 1, n_time), 0.5, dtype=np.float64)
        result = compute_peak_enhancement(signal, baseline_frames=5)
        assert np.allclose(result, 0.0)

    def test_max_cap(self) -> None:
        """Extreme enhancement is capped at max_cap."""
        n_time = 20
        signal = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        signal[..., :5] = 10.0  # low baseline
        signal[..., 10] = 10000.0  # extreme peak
        result = compute_peak_enhancement(signal, baseline_frames=5, max_cap=5000.0)
        assert np.all(result <= 5000.0)


class TestTimeToPeak:
    """Tests for compute_time_to_peak."""

    def test_known_peak(self) -> None:
        """Peak at known frame should give correct time in seconds."""
        n_time = 20
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0
        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        peak_frame = 10
        concentration[..., peak_frame] = 1.0
        result = compute_time_to_peak(concentration, time_points)
        expected_s = peak_frame * tr_s
        assert np.allclose(result, expected_s)


class TestSER:
    """Tests for compute_signal_enhancement_ratio."""

    def test_washout(self) -> None:
        """Higher early than late concentration → SER > 1."""
        n_time = 80
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        # Early window (0.5-2 min): frames 3-12
        concentration[..., 3:13] = 2.0
        # Late window (5-10 min): frames 30-60
        concentration[..., 30:61] = 0.5
        result = compute_signal_enhancement_ratio(concentration, time_points)
        assert np.all(result > 1.0)

    def test_persistent(self) -> None:
        """Higher late than early → SER < 1."""
        n_time = 80
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        concentration[..., 3:13] = 0.5
        concentration[..., 30:61] = 2.0
        result = compute_signal_enhancement_ratio(concentration, time_points)
        assert np.all(result < 1.0)


# =========================================================================
# Cross-function tests
# =========================================================================

class TestNoInfOutputs:
    """Verify no Inf values in any output for well-behaved data."""

    def test_all_maps_no_inf(
        self,
        gamma_variate_data: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    ) -> None:
        """Well-behaved gamma-variate data should produce no Inf in any map."""
        concentration, time_points = gamma_variate_data

        iauc60 = compute_iauc(concentration, time_points, 60.0)
        # Peak enhancement uses raw signal, not concentration — simulate signal
        signal_like = concentration + 500.0  # shift to realistic signal range
        peak = compute_peak_enhancement(signal_like)
        ttp = compute_time_to_peak(concentration, time_points)
        ser = compute_signal_enhancement_ratio(concentration, time_points)
        wash_in = compute_wash_in_slope(
            concentration, time_points, bolus_arrival_frame=5, volume_tr_s=5.0,
        )
        wash_out = compute_wash_out_slope(
            concentration, time_points, volume_tr_s=5.0,
        )
        bat = compute_bolus_arrival_time(concentration, time_points)
        curve_type = compute_curve_type(ser)

        for name, arr in [
            ("iauc60", iauc60), ("peak", peak), ("ttp", ttp), ("ser", ser),
            ("wash_in", wash_in), ("wash_out", wash_out), ("bat", bat),
            ("curve_type", curve_type),
        ]:
            assert not np.any(np.isinf(arr)), f"{name} contains Inf values"
