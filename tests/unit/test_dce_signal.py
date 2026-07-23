"""Unit tests for DCE signal processing utilities."""

import numpy as np
import pytest

from pmriprep.utils.dce_signal import (
    concentration_to_plasma,
    e1_to_r1,
    ernst_signal_model,
    plasma_to_blood_concentration,
    signal_to_concentration,
    signal_to_concentration_simple,
    solve_e1_from_signal_ratio,
)
from pmriprep.utils.dce_constants import EPSILON_E1_UPPER, EPSILON_T1, HEMATOCRIT
from pmriprep.utils.dce_t1_validation import T1_TISSUE_REFERENCE


class TestErnstSignalModel:
    """Tests for the Ernst equation signal model."""

    def test_ernst_signal_model_known_values(self) -> None:
        """Verify Ernst equation against hand-computed values."""
        s0 = 1000.0
        tr = 0.005  # 5 ms
        t1 = 1.0  # 1 s
        flip_deg = 15.0
        flip_rad = np.deg2rad(flip_deg)

        # Hand-compute: E1 = exp(-0.005/1.0) = exp(-0.005)
        e1 = np.exp(-tr / t1)
        sin_a = np.sin(flip_rad)
        cos_a = np.cos(flip_rad)
        expected = s0 * sin_a * (1.0 - e1) / (1.0 - cos_a * e1)

        result = ernst_signal_model(s0, flip_rad, tr, t1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ernst_signal_model_array_input(self) -> None:
        """Ernst equation works with numpy arrays for T1 parameter."""
        s0 = 1000.0
        tr = 0.005
        flip_rad = np.deg2rad(15.0)
        t1_array = np.array([0.5, 1.0, 1.5, 2.0])

        result = ernst_signal_model(s0, flip_rad, tr, t1_array)
        assert result.shape == t1_array.shape

        # Each element should match the scalar version
        for i, t1_val in enumerate(t1_array):
            expected = ernst_signal_model(s0, flip_rad, tr, float(t1_val))
            np.testing.assert_allclose(result[i], expected, rtol=1e-10)


class TestSolveE1FromSignalRatio:
    """Tests for solving E1 from signal ratio."""

    def test_solve_e1_round_trip(self) -> None:
        """Signal ratio = 1.0 should recover baseline E1."""
        tr = 0.005
        t1 = 1.0
        flip_rad = np.deg2rad(15.0)

        e1_baseline = np.exp(-tr / t1)
        signal_ratio = 1.0

        e1_recovered = solve_e1_from_signal_ratio(signal_ratio, flip_rad, e1_baseline)
        np.testing.assert_allclose(e1_recovered, e1_baseline, rtol=1e-6)

    def test_solve_e1_enhanced_signal(self) -> None:
        """Signal ratio > 1 (contrast enhancement) should decrease E1."""
        tr = 0.005
        t1 = 1.0
        flip_rad = np.deg2rad(15.0)

        e1_baseline = np.exp(-tr / t1)
        # Enhancement: signal ratio > 1 means T1 shortened, so E1 decreases
        e1_enhanced = solve_e1_from_signal_ratio(1.5, flip_rad, e1_baseline)
        assert e1_enhanced < e1_baseline

    def test_solve_e1_clipping(self) -> None:
        """Output E1 is clipped to [EPSILON_T1, 1.0 - EPSILON_E1_UPPER]."""
        flip_rad = np.deg2rad(15.0)
        e1_baseline = 0.995

        # Very large signal ratio should push E1 toward lower bound
        e1_low = solve_e1_from_signal_ratio(100.0, flip_rad, e1_baseline)
        assert e1_low >= EPSILON_T1

        # Very small signal ratio should push E1 toward upper bound
        e1_high = solve_e1_from_signal_ratio(0.001, flip_rad, e1_baseline)
        assert e1_high <= 1.0 - EPSILON_E1_UPPER


class TestE1ToR1:
    """Tests for E1 to R1 conversion."""

    def test_e1_to_r1(self) -> None:
        """E1 = exp(-TR/T1) -> R1 = 1/T1 = -ln(E1)/TR."""
        tr = 0.005
        t1 = 1.0
        e1 = np.exp(-tr / t1)

        r1 = e1_to_r1(e1, tr)
        expected_r1 = 1.0 / t1  # = 1.0 s^-1
        np.testing.assert_allclose(r1, expected_r1, rtol=1e-6)

    def test_e1_to_r1_clipping(self) -> None:
        """Very small and very large E1 values are clipped before log."""
        tr = 0.005

        # Very small E1 (should be clipped to EPSILON_T1)
        r1_low = e1_to_r1(1e-10, tr)
        expected_low = -np.log(EPSILON_T1) / tr
        np.testing.assert_allclose(r1_low, expected_low, rtol=1e-10)

        # Very large E1 near 1.0 (should be clipped to 1.0 - EPSILON_E1_UPPER)
        r1_high = e1_to_r1(0.99999, tr)
        expected_high = -np.log(1.0 - EPSILON_E1_UPPER) / tr
        np.testing.assert_allclose(r1_high, expected_high, rtol=1e-10)


class TestConcentrationConversions:
    """Tests for blood/plasma concentration conversions."""

    def test_concentration_plasma_round_trip(self) -> None:
        """plasma_to_blood(concentration_to_plasma(c)) recovers c."""
        c_blood = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        c_plasma = concentration_to_plasma(c_blood)
        c_recovered = plasma_to_blood_concentration(c_plasma)
        np.testing.assert_allclose(c_recovered, c_blood, rtol=1e-10)

    def test_concentration_to_plasma_factor(self) -> None:
        """Default factor is 1/(1 - 0.42) = 1/0.58."""
        c_blood = np.array([1.0])
        c_plasma = concentration_to_plasma(c_blood)
        expected_factor = 1.0 / (1.0 - HEMATOCRIT)
        np.testing.assert_allclose(c_plasma[0], expected_factor, rtol=1e-10)

    def test_concentration_to_plasma_custom_hematocrit(self) -> None:
        """Custom hematocrit = 0.45 gives factor 1/0.55."""
        c_blood = np.array([1.0])
        hct = 0.45
        c_plasma = concentration_to_plasma(c_blood, hematocrit=hct)
        expected = 1.0 / (1.0 - hct)
        np.testing.assert_allclose(c_plasma[0], expected, rtol=1e-10)


class TestFieldStrengthDefaultT1:
    """Tests for field-strength-aware default T1 when no T1 map provided."""

    @pytest.fixture
    def simple_signal(self) -> np.ndarray:
        """1D DCE signal with 20 timepoints, baseline + enhancement."""
        rng = np.random.default_rng(0)
        signal = np.ones(20) * 500.0
        signal[10:] = 600.0  # simple enhancement
        signal += rng.normal(0, 5, 20)
        return signal

    def test_default_t1_at_1_5t(self, simple_signal: np.ndarray) -> None:
        """At 1.5T with no T1 map, default T1 should be 900ms."""
        result = signal_to_concentration(
            simple_signal, flip_angle_deg=15.0, tr=0.005,
            t1_map=None, t1_default_ms=None, field_strength=1.5,
        )
        # R1_baseline = 1/T1 = 1/0.9 ≈ 1.111 s^-1
        expected_r1 = 1.0 / (T1_TISSUE_REFERENCE[1.5]["default"] / 1000.0)
        np.testing.assert_allclose(result.r1_baseline, expected_r1, rtol=1e-6)

    def test_default_t1_at_3t(self, simple_signal: np.ndarray) -> None:
        """At 3T with no T1 map, default T1 should be 1200ms."""
        result = signal_to_concentration(
            simple_signal, flip_angle_deg=15.0, tr=0.005,
            t1_map=None, t1_default_ms=None, field_strength=3.0,
        )
        expected_r1 = 1.0 / (T1_TISSUE_REFERENCE[3.0]["default"] / 1000.0)
        np.testing.assert_allclose(result.r1_baseline, expected_r1, rtol=1e-6)

    def test_default_t1_at_7t(self, simple_signal: np.ndarray) -> None:
        """At 7T with no T1 map, default T1 should be 1700ms."""
        result = signal_to_concentration(
            simple_signal, flip_angle_deg=15.0, tr=0.005,
            t1_map=None, t1_default_ms=None, field_strength=7.0,
        )
        expected_r1 = 1.0 / (T1_TISSUE_REFERENCE[7.0]["default"] / 1000.0)
        np.testing.assert_allclose(result.r1_baseline, expected_r1, rtol=1e-6)

    def test_explicit_t1_overrides_field_strength(
        self, simple_signal: np.ndarray,
    ) -> None:
        """Explicit t1_default_ms takes precedence over field-strength lookup."""
        result = signal_to_concentration(
            simple_signal, flip_angle_deg=15.0, tr=0.005,
            t1_map=None, t1_default_ms=1000.0, field_strength=1.5,
        )
        expected_r1 = 1.0 / (1000.0 / 1000.0)
        np.testing.assert_allclose(result.r1_baseline, expected_r1, rtol=1e-6)

    def test_simple_wrapper_field_strength_default(
        self, simple_signal: np.ndarray,
    ) -> None:
        """signal_to_concentration_simple uses field-strength default when t1_pre_ms=None."""
        conc_15t = signal_to_concentration_simple(
            simple_signal, flip_angle_deg=15.0, tr=0.005,
            t1_pre_ms=None, field_strength=1.5,
        )
        conc_3t = signal_to_concentration_simple(
            simple_signal, flip_angle_deg=15.0, tr=0.005,
            t1_pre_ms=None, field_strength=3.0,
        )
        # Different field strengths → different concentrations
        assert not np.allclose(conc_15t, conc_3t)


class TestBaselineSubtraction:
    """Tests for pre-bolus baseline subtraction in concentration conversion."""

    def test_pre_bolus_concentration_unbiased(self) -> None:
        """Pre-bolus concentration mean should be ~0, not biased positive."""
        rng = np.random.default_rng(42)
        n_time = 30
        baseline_frames = 10
        # Simulate a 1D signal: constant baseline + enhancement after frame 15
        s0 = 500.0
        signal = np.full(n_time, s0)
        signal[15:] = 600.0  # 20% enhancement
        signal = signal + rng.normal(0, 10, n_time)  # noise

        result = signal_to_concentration(
            signal, flip_angle_deg=15.0, tr=0.005,
            t1_default_ms=1000.0, baseline_frames=baseline_frames,
            convert_to_plasma=False,
        )
        # Signed pre-bolus noise remains centered exactly at zero.
        pre_bolus = result.concentration[:baseline_frames]
        assert np.mean(pre_bolus) == pytest.approx(0.0, abs=1e-12)
        assert np.any(pre_bolus < 0)

    def test_enhancement_preserved_after_baseline_subtraction(self) -> None:
        """Post-bolus enhancement should still be positive and non-trivial."""
        n_time = 30
        baseline_frames = 10
        signal = np.full(n_time, 500.0)
        signal[15:] = 700.0  # 40% enhancement
        result = signal_to_concentration(
            signal, flip_angle_deg=15.0, tr=0.005,
            t1_default_ms=1000.0, baseline_frames=baseline_frames,
            convert_to_plasma=False,
        )
        # Post-enhancement concentration should be clearly positive
        post_bolus = result.concentration[15:]
        assert np.all(post_bolus > 0), "Enhancement lost after baseline subtraction"
        assert np.mean(post_bolus) > np.mean(result.concentration[:baseline_frames]), (
            "Enhancement not larger than baseline"
        )
