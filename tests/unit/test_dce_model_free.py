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
        """A rising delayed tail after an early global peak has positive slope."""
        n_time = 60
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        concentration[..., :6] = np.linspace(0.0, 2.5, 6)
        concentration[..., 6:] = np.linspace(1.0, 2.0, n_time - 6)
        result = compute_wash_out_slope(concentration, time_points, volume_tr_s=tr_s)
        valid = np.isfinite(result)
        assert valid.all()
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

    def test_bat_with_brain_mask(self) -> None:
        """Masked-out voxels should remain NaN; unmasked should get valid BAT."""
        n_time = 60
        tr_s = 5.0
        time_points = np.arange(n_time, dtype=np.float64) * tr_s / 60.0

        # 2x2x1 volume, all voxels have enhancement
        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        arrival_frame = 10
        for t in range(arrival_frame, n_time):
            concentration[..., t] = 2.0 * (1.0 - np.exp(-0.1 * (t - arrival_frame)))

        # Mask: only top-left voxel is in-brain
        brain_mask = np.zeros((2, 2, 1), dtype=np.float64)
        brain_mask[0, 0, 0] = 1.0

        result = compute_bolus_arrival_time(
            concentration, time_points, baseline_frames=5, brain_mask=brain_mask,
        )

        # Masked-out voxels must be NaN
        assert np.isnan(result[0, 1, 0])
        assert np.isnan(result[1, 0, 0])
        assert np.isnan(result[1, 1, 0])
        # Unmasked voxel should have a finite BAT
        assert np.isfinite(result[0, 0, 0])


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

    def test_window_starts_at_bolus_arrival(self) -> None:
        """Pre-contrast acquisition time must not shorten the IAUC window."""
        time_points = np.arange(10, dtype=np.float64) * 15.0 / 60.0
        concentration = np.ones((1, 1, 1, 10), dtype=np.float64)
        result = compute_iauc(
            concentration, time_points, 60.0, bolus_arrival_frame=2,
        )
        assert result[0, 0, 0] == pytest.approx(1.0)

    def test_interpolates_exact_cutoff(self) -> None:
        """A cutoff between frames uses a linearly interpolated endpoint."""
        time_points = np.array([0.0, 0.4, 0.8, 1.2], dtype=np.float64)
        concentration = (2.0 * time_points).reshape(1, 1, 1, -1)
        result = compute_iauc(concentration, time_points, 60.0)
        assert result[0, 0, 0] == pytest.approx(1.0)

    def test_incomplete_window_returns_nan(self) -> None:
        """A partial acquisition must not be mislabeled as full-window IAUC."""
        time_points = np.arange(6, dtype=np.float64) * 10.0 / 60.0
        concentration = np.ones((1, 1, 1, 6), dtype=np.float64)
        result = compute_iauc(concentration, time_points, 60.0)
        assert np.isnan(result[0, 0, 0])

    def test_exact_window_tolerates_ulp_shortfall(self) -> None:
        """Equivalent endpoint expressions do not reject a complete window."""
        time_points = np.linspace(0.0, 1.0, 31, dtype=np.float64)
        time_points[-1] = np.nextafter(1.0, -np.inf)
        concentration = np.ones((1, 1, 1, time_points.size), dtype=np.float64)
        result = compute_iauc(concentration, time_points, 60.0)
        assert result[0, 0, 0] == pytest.approx(1.0)

    def test_nonfinite_interval_returns_nan(self) -> None:
        """Non-finite concentration samples make that voxel undefined."""
        time_points = np.arange(8, dtype=np.float64) * 10.0 / 60.0
        concentration = np.ones((1, 1, 1, 8), dtype=np.float64)
        concentration[..., 3] = np.nan
        result = compute_iauc(concentration, time_points, 60.0)
        assert np.isnan(result[0, 0, 0])

    @pytest.mark.parametrize(
        "time_points",
        [
            np.array([0.0, 0.5, 0.4, 1.0]),
            np.array([0.0, 0.5, np.nan, 1.0]),
        ],
        ids=["nonmonotonic", "nonfinite"],
    )
    def test_invalid_time_axis_raises(self, time_points: np.ndarray) -> None:
        """Programming/input-contract violations are not physiological NaNs."""
        concentration = np.ones((1, 1, 1, time_points.size), dtype=np.float64)
        with pytest.raises(ValueError, match="finite and strictly increasing"):
            compute_iauc(concentration, time_points, 60.0)


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
        """Arbitrary scanner scaling below one does not change the result."""
        n_time = 20
        signal = np.full((2, 2, 1, n_time), 0.5, dtype=np.float64)
        signal[..., 10] = 1.5
        result = compute_peak_enhancement(signal, baseline_frames=5)
        assert np.allclose(result, 200.0)

    def test_baseline_peak_is_excluded(self) -> None:
        """A pre-contrast maximum is not reported as enhancement."""
        signal = np.full((1, 1, 1, 10), 500.0, dtype=np.float64)
        signal[..., 0] = 600.0
        signal[..., 5:] = 550.0
        result = compute_peak_enhancement(signal, baseline_frames=5)
        assert result[0, 0, 0] == pytest.approx((550.0 - 520.0) / 520.0 * 100.0)

    @pytest.mark.parametrize("invalid_baseline", [0.0, -10.0])
    def test_nonpositive_baseline_returns_nan(self, invalid_baseline: float) -> None:
        """Undefined baseline ratios remain visibly undefined."""
        signal = np.full((1, 1, 1, 10), invalid_baseline, dtype=np.float64)
        signal[..., 5:] = 100.0
        result = compute_peak_enhancement(signal, baseline_frames=5)
        assert np.isnan(result[0, 0, 0])

    def test_nonfinite_signal_returns_nan(self) -> None:
        """A partially non-finite signal curve is not assigned a peak."""
        signal = np.full((1, 1, 1, 10), 100.0, dtype=np.float64)
        signal[..., 7] = np.nan
        result = compute_peak_enhancement(signal, baseline_frames=5)
        assert np.isnan(result[0, 0, 0])

    @pytest.mark.parametrize("baseline_frames", [0, 10])
    def test_invalid_baseline_window_raises(self, baseline_frames: int) -> None:
        """Invalid processing configuration raises instead of making a map."""
        signal = np.ones((1, 1, 1, 10), dtype=np.float64)
        with pytest.raises(ValueError, match="leave at least one"):
            compute_peak_enhancement(signal, baseline_frames=baseline_frames)

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

    @pytest.mark.parametrize(
        "curve",
        [
            np.zeros(8, dtype=np.float64),
            -np.arange(8, dtype=np.float64),
            np.array([0.0, 0.1, np.nan, 0.4, 0.2, 0.1, 0.0, 0.0]),
        ],
        ids=["flat", "nonpositive", "nonfinite"],
    )
    def test_undefined_curve_returns_nan(self, curve: np.ndarray) -> None:
        """Invalid or non-enhancing curves must not produce plausible TTP."""
        time_points = np.arange(curve.size, dtype=np.float64) * 5.0 / 60.0
        result = compute_time_to_peak(curve.reshape(1, 1, 1, -1), time_points)
        assert np.isnan(result[0, 0, 0])

    def test_mixed_voxel_validity(self) -> None:
        """An invalid voxel does not prevent a valid voxel from being mapped."""
        time_points = np.arange(6, dtype=np.float64) * 4.0 / 60.0
        concentration = np.zeros((2, 1, 1, 6), dtype=np.float64)
        concentration[0, 0, 0, 3] = 0.5
        concentration[1, 0, 0, 2] = np.nan
        result = compute_time_to_peak(concentration, time_points)
        assert result[0, 0, 0] == pytest.approx(12.0)
        assert np.isnan(result[1, 0, 0])


class TestSER:
    """Tests for compute_signal_enhancement_ratio."""

    def test_washout(self) -> None:
        """Higher early than late concentration → SER > 1."""
        # 80 frames, 10s each → ~13.3 min total; bolus at frame 2
        n_time = 80
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        bolus_frame = 2
        concentration = np.full((2, 2, 1, n_time), 100.0, dtype=np.float64)
        # Early window: bolus + 0.5 to bolus + 2.5 min → ~frames 5-17
        concentration[..., 5:18] = 120.0
        # Late window: last 20% of post-bolus → ~last 16 frames
        concentration[..., 64:] = 110.0
        result = compute_signal_enhancement_ratio(
            concentration, time_points, bolus_arrival_frame=bolus_frame,
        )
        assert np.all(result > 1.0)

    def test_persistent(self) -> None:
        """Higher late than early → SER < 1."""
        n_time = 80
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        bolus_frame = 2
        concentration = np.full((2, 2, 1, n_time), 100.0, dtype=np.float64)
        concentration[..., 5:18] = 110.0
        concentration[..., 64:] = 120.0
        result = compute_signal_enhancement_ratio(
            concentration, time_points, bolus_arrival_frame=bolus_frame,
        )
        assert np.all(result < 1.0)

    def test_plateau_near_one(self) -> None:
        """Constant concentration post-bolus → SER ≈ 1.0."""
        n_time = 60
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        bolus_frame = 3
        concentration = np.full((2, 2, 1, n_time), 100.0, dtype=np.float64)
        concentration[..., bolus_frame:] = 115.0  # constant post-bolus
        result = compute_signal_enhancement_ratio(
            concentration, time_points, bolus_arrival_frame=bolus_frame,
            baseline_frames=bolus_frame,
        )
        assert np.allclose(result, 1.0, atol=0.05)

    def test_short_acquisition(self) -> None:
        """Short acquisition (5 min) uses adaptive late window, not absolute."""
        # 40 frames, 7.6s each → ~5.1 min; bolus at frame 7
        n_time = 40
        time_points = np.arange(n_time, dtype=np.float64) * 7.6 / 60.0
        bolus_frame = 7
        concentration = np.full((3, 3, 1, n_time), 100.0, dtype=np.float64)
        # Early: frames ~10-25 (0.5-2.5 min post-bolus)
        concentration[..., 10:26] = 120.0
        # Late: last 20% of post-bolus → ~last 6-7 frames
        concentration[..., 33:] = 110.0
        result = compute_signal_enhancement_ratio(
            concentration, time_points, bolus_arrival_frame=bolus_frame,
        )
        # Should be ~2.0, definitely not millions
        assert np.all(result > 1.0)
        assert np.all(result < 10.0)

    def test_non_enhancing_signal_is_undefined(self) -> None:
        """A non-enhancing signal must not be classified as plateau-like."""
        n_time = 60
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        concentration = np.zeros((2, 2, 1, n_time), dtype=np.float64)
        # Near-zero everywhere
        concentration[..., :] = 1e-12
        result = compute_signal_enhancement_ratio(
            concentration, time_points, bolus_arrival_frame=3,
        )
        assert np.all(np.isnan(result))

    def test_uses_baseline_subtracted_signal_enhancement(self) -> None:
        """SER is the ratio of early and late enhancement above S0."""
        n_time = 80
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        signal = np.full((1, 1, 1, n_time), 100.0, dtype=np.float64)
        signal[..., 5:18] = 140.0
        signal[..., 64:] = 120.0

        result = compute_signal_enhancement_ratio(
            signal, time_points, bolus_arrival_frame=2, baseline_frames=2,
        )

        assert result[0, 0, 0] == pytest.approx(2.0)

    def test_extreme_ser_clamped(self) -> None:
        """Near-zero late concentration → raw SER >> 10, clamped to max_cap."""
        n_time = 80
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        bolus_frame = 2
        concentration = np.full((2, 2, 1, n_time), 100.0, dtype=np.float64)
        # High early concentration
        concentration[..., 5:18] = 150.0
        # Near-zero late concentration (triggers large raw SER)
        concentration[..., 64:] = 100.0001
        result = compute_signal_enhancement_ratio(
            concentration, time_points, bolus_arrival_frame=bolus_frame,
        )
        assert np.all(result <= 10.0)
        assert np.all(result == 10.0)

    def test_long_acquisition_late_capped(self) -> None:
        """Long acquisition (>10 min): late window capped at 5 min."""
        # 120 frames, 10s each → 20 min; bolus at frame 5
        n_time = 120
        time_points = np.arange(n_time, dtype=np.float64) * 10.0 / 60.0
        bolus_frame = 5
        concentration = np.full((2, 2, 1, n_time), 100.0, dtype=np.float64)
        # Early: strong signal
        concentration[..., 8:20] = 130.0
        # Fill late portion: high value only in last 5 min (frames 90-120)
        concentration[..., 90:] = 110.0
        # Put low value in the preceding region (frames 60-90)
        concentration[..., 60:90] = 101.0
        result = compute_signal_enhancement_ratio(
            concentration, time_points, bolus_arrival_frame=bolus_frame,
        )
        # Late window should capture frames ~90-120 (last 5 min), not 60-120
        # So SER ≈ 3.0/1.0 = 3.0, not 3.0/0.55 ≈ 5.5
        assert np.all(result > 1.0)
        assert np.all(result < 5.0)


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
