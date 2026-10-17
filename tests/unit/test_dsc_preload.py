"""Unit tests for DSC preload detection and classification."""

import numpy as np
import pytest

from pmriprep.utils.dsc_preload import (
    PreloadClass,
    classify_preload,
    get_preload_thresholds,
)


class TestClassifyPreload:
    """Tests for classify_preload."""

    def test_classify_preload_none(self) -> None:
        """Low baseline delta_r2 at 3T classifies as NONE."""
        assert classify_preload(0.1, field_strength=3.0) == PreloadClass.NONE

    def test_classify_preload_minimal(self) -> None:
        """delta_r2=1.0 at 3T classifies as MINIMAL (0.5 <= 1.0 < 2.0)."""
        assert classify_preload(1.0, field_strength=3.0) == PreloadClass.MINIMAL

    def test_classify_preload_half_dose(self) -> None:
        """delta_r2=3.0 at 3T classifies as HALF_DOSE (2.0 <= 3.0 < 5.0)."""
        assert classify_preload(3.0, field_strength=3.0) == PreloadClass.HALF_DOSE

    def test_classify_preload_significant(self) -> None:
        """delta_r2=7.0 at 3T classifies as SIGNIFICANT (5.0 <= 7.0 < 10.0)."""
        assert classify_preload(7.0, field_strength=3.0) == PreloadClass.SIGNIFICANT

    def test_classify_preload_severe(self) -> None:
        """delta_r2=15.0 at 3T classifies as SEVERE (>= 10.0)."""
        assert classify_preload(15.0, field_strength=3.0) == PreloadClass.SEVERE

    def test_classify_preload_boundary(self) -> None:
        """At exact threshold (0.5 at 3T), classify as MINIMAL not NONE."""
        assert classify_preload(0.5, field_strength=3.0) == PreloadClass.MINIMAL

    def test_classify_preload_negative(self) -> None:
        """Negative delta_r2 classifies as NONE (below all thresholds)."""
        assert classify_preload(-0.1, field_strength=3.0) == PreloadClass.NONE


class TestGetPreloadThresholds:
    """Tests for get_preload_thresholds."""

    def test_get_preload_thresholds_3t(self) -> None:
        """3T thresholds match expected values."""
        t = get_preload_thresholds(3.0)
        assert t.none_max == 0.5
        assert t.minimal_max == 2.0
        assert t.half_dose_max == 5.0
        assert t.significant_max == 10.0

    def test_get_preload_thresholds_field_scaling(self) -> None:
        """Thresholds increase monotonically with field strength."""
        t_15 = get_preload_thresholds(1.5)
        t_30 = get_preload_thresholds(3.0)
        t_70 = get_preload_thresholds(7.0)

        assert t_15.none_max < t_30.none_max < t_70.none_max
        assert t_15.minimal_max < t_30.minimal_max < t_70.minimal_max
        assert t_15.half_dose_max < t_30.half_dose_max < t_70.half_dose_max
        assert t_15.significant_max < t_30.significant_max < t_70.significant_max


class TestPreloadClassEnum:
    """Tests for PreloadClass enum values."""

    def test_preload_class_enum_values(self) -> None:
        """PreloadClass enum members have expected string values."""
        assert PreloadClass.NONE.value == "none"
        assert PreloadClass.MINIMAL.value == "minimal"
        assert PreloadClass.HALF_DOSE.value == "half_dose"
        assert PreloadClass.SIGNIFICANT.value == "significant"
        assert PreloadClass.SEVERE.value == "severe"


# =============================================================================
# Agent-aware thresholds + float preload support
# =============================================================================


class TestAgentNormalizedThresholds:
    """Thresholds scale linearly with agent_standard_dose."""

    def test_default_agent_matches_baseline_3t(self) -> None:
        from pmriprep.utils.dsc_preload import PRELOAD_THRESHOLDS_3T
        baseline = PRELOAD_THRESHOLDS_3T
        scaled = get_preload_thresholds(3.0, agent_standard_dose=0.1)
        assert scaled.none_max == pytest.approx(baseline.none_max)
        assert scaled.minimal_max == pytest.approx(baseline.minimal_max)
        assert scaled.half_dose_max == pytest.approx(baseline.half_dose_max)
        assert scaled.significant_max == pytest.approx(baseline.significant_max)

    def test_vueway_thresholds_halved(self) -> None:
        """Vueway agent_standard_dose=0.05 halves every threshold."""
        from pmriprep.utils.dsc_preload import PRELOAD_THRESHOLDS_3T
        baseline = PRELOAD_THRESHOLDS_3T
        scaled = get_preload_thresholds(3.0, agent_standard_dose=0.05)
        assert scaled.none_max == pytest.approx(baseline.none_max * 0.5)
        assert scaled.half_dose_max == pytest.approx(baseline.half_dose_max * 0.5)
        assert scaled.significant_max == pytest.approx(baseline.significant_max * 0.5)

    def test_gadoquatrane_thresholds_scaled(self) -> None:
        """Gadoquatrane (0.04 mmol/kg) gives 0.4x thresholds."""
        from pmriprep.utils.dsc_preload import PRELOAD_THRESHOLDS_3T
        scaled = get_preload_thresholds(3.0, agent_standard_dose=0.04)
        assert scaled.half_dose_max == pytest.approx(
            PRELOAD_THRESHOLDS_3T.half_dose_max * 0.4
        )

    def test_classify_agent_aware(self) -> None:
        """Same ΔR2* classifies differently across agent_standard_dose."""
        delta_r2 = 1.0  # s^-1
        # At 0.1 (Dotarem): 1.0 falls in MINIMAL [0.5, 2.0)
        cls_dotarem = classify_preload(delta_r2, 3.0, agent_standard_dose=0.1)
        assert cls_dotarem == PreloadClass.MINIMAL
        # At 0.05 (Vueway): thresholds halved → 1.0 falls in HALF_DOSE [1.0, 2.5)
        cls_vueway = classify_preload(delta_r2, 3.0, agent_standard_dose=0.05)
        assert cls_vueway == PreloadClass.HALF_DOSE

    def test_invalid_agent_dose_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            get_preload_thresholds(3.0, agent_standard_dose=0.0)
        with pytest.raises(ValueError, match="must be > 0"):
            get_preload_thresholds(3.0, agent_standard_dose=-0.1)


class TestResidualEquivalentFraction:
    """Physics of ΔR2* → residual-equivalent fraction inversion."""

    def test_zero_delta_r2_is_zero(self) -> None:
        from pmriprep.utils.dsc_preload import (
            residual_equivalent_fraction_from_delta_r2,
        )
        assert residual_equivalent_fraction_from_delta_r2(0.0, 3.0, 0.1) == 0.0

    def test_roundtrip_via_r2_star(self) -> None:
        """Given mM = dose_fraction × agent_std × r2_star_per_mM, invert."""
        from pmriprep.utils.dsc_preload import (
            residual_equivalent_fraction_from_delta_r2,
        )
        from pmriprep.utils.dsc_constants import get_r2_star
        field = 3.0
        agent_std = 0.1
        r2 = get_r2_star(field)
        # Pick a dose fraction, compute implied delta_r2, invert
        for fraction in [0.05, 0.2, 0.5, 1.0]:
            delta_r2 = fraction * agent_std * r2
            inferred = residual_equivalent_fraction_from_delta_r2(
                delta_r2, field, agent_std,
            )
            assert inferred == pytest.approx(fraction, rel=1e-6)

    def test_agent_dose_scales_inversion(self) -> None:
        """Halving agent_standard_dose doubles the inferred fraction."""
        from pmriprep.utils.dsc_preload import (
            residual_equivalent_fraction_from_delta_r2,
        )
        delta_r2 = 2.0
        f_std = residual_equivalent_fraction_from_delta_r2(delta_r2, 3.0, 0.1)
        f_vueway = residual_equivalent_fraction_from_delta_r2(delta_r2, 3.0, 0.05)
        assert f_vueway == pytest.approx(2 * f_std)


class TestPreloadBinningIndependence:
    """User-float and auto-detection use SEPARATE edge tables by design.

    They measure different physical quantities (administered vs residual)
    and are not expected to numerically align. Both map to the same class
    names for downstream K2 bucketing.
    """

    def test_auto_detection_midpoints_classify_correctly(self) -> None:
        """ΔR2* at known bin midpoints returns the expected PreloadClass."""
        from pmriprep.utils.dsc_preload import PRELOAD_THRESHOLDS_3T as t
        # Midpoint of MINIMAL bin: (0.5 + 2.0) / 2 = 1.25 s^-1
        assert classify_preload(
            (t.none_max + t.minimal_max) / 2, 3.0, 0.1
        ) == PreloadClass.MINIMAL
        # Midpoint of HALF_DOSE bin
        assert classify_preload(
            (t.minimal_max + t.half_dose_max) / 2, 3.0, 0.1
        ) == PreloadClass.HALF_DOSE
        # Midpoint of SIGNIFICANT bin
        assert classify_preload(
            (t.half_dose_max + t.significant_max) / 2, 3.0, 0.1
        ) == PreloadClass.SIGNIFICANT


class TestAgentDoseFlowsThroughDetectPreload:
    """detect_preload_from_signal records effective thresholds and agent."""

    def test_signal_detection_logs_effective_thresholds(self) -> None:
        from pmriprep.utils.dsc_preload import detect_preload_from_signal
        # Small synthetic 4D volume with near-zero baseline
        signal = np.full((4, 4, 1, 30), 1000.0, dtype=np.float32)
        # Add mild signal variation
        rng = np.random.default_rng(0)
        signal += rng.normal(0, 10, signal.shape).astype(np.float32)
        result = detect_preload_from_signal(
            signal=signal,
            te=0.03,
            baseline_frames=5,
            field_strength=3.0,
            agent_standard_dose=0.05,
        )
        assert result.details is not None
        assert result.details["agent_standard_dose"] == pytest.approx(0.05)
        assert "effective_thresholds" in result.details
        assert "residual_equivalent_fraction" in result.details


# =============================================================================
# Z-score-to-threshold confidence score
# =============================================================================


class TestConfidenceScore:
    """Confidence is P(correct classification), from SEM vs nearest bin edge."""

    @staticmethod
    def _signal_with_delta_r2(
        target_delta_r2: float,
        reference: float,
        te: float,
        std_signal: float,
        shape: tuple = (8, 8, 2, 20),
        seed: int = 0,
    ) -> np.ndarray:
        """Build a 4D signal whose baseline yields approximately the target ΔR2*.

        Uses S = S_ref * exp(-ΔR2* * TE).
        """
        mean_s = reference * np.exp(-target_delta_r2 * te)
        rng = np.random.default_rng(seed)
        return rng.normal(mean_s, std_signal, shape).astype(np.float32)

    def test_deep_in_none_bin_high_confidence(self) -> None:
        from pmriprep.utils.dsc_preload import detect_preload_from_signal
        # ΔR2* ≈ 0, far from none_max=0.5, low noise → confidence ≈ 1
        sig = self._signal_with_delta_r2(
            target_delta_r2=0.0, reference=1000.0, te=0.03,
            std_signal=1.0,
        )
        r = detect_preload_from_signal(
            sig, te=0.03, baseline_frames=5, reference_signal=1000.0,
        )
        assert r.confidence > 0.95

    def test_near_edge_low_confidence(self) -> None:
        from pmriprep.utils.dsc_preload import detect_preload_from_signal
        # ΔR2* ≈ 0.5 (right at none_max at 3T) + moderate per-voxel noise
        sig = self._signal_with_delta_r2(
            target_delta_r2=0.5, reference=1000.0, te=0.03,
            std_signal=150.0, shape=(4, 4, 1, 20),
        )
        r = detect_preload_from_signal(
            sig, te=0.03, baseline_frames=5, reference_signal=1000.0,
        )
        assert 0.0 <= r.confidence < 0.6

    def test_confidence_scales_with_n_voxels(self) -> None:
        from pmriprep.utils.dsc_preload import detect_preload_from_signal
        # Target ΔR2* = 1.25 (middle of MINIMAL bin [0.5, 2.0]; distance to
        # nearest edge ~0.75). Picked mid-bin so small-mask sampling noise
        # on the mean doesn't dominate the distance calculation.
        kwargs = dict(
            target_delta_r2=1.25, reference=1000.0, te=0.03, std_signal=50.0,
        )
        small = self._signal_with_delta_r2(shape=(3, 3, 1, 20), **kwargs)
        large = self._signal_with_delta_r2(shape=(16, 16, 4, 20), **kwargs)
        rs = detect_preload_from_signal(
            small, te=0.03, baseline_frames=5, reference_signal=1000.0,
        )
        rl = detect_preload_from_signal(
            large, te=0.03, baseline_frames=5, reference_signal=1000.0,
        )
        # Primary invariant: larger N → smaller SEM (mathematical identity).
        assert rl.details["sem_delta_r2"] < rs.details["sem_delta_r2"]
        # Given comparable mid-bin distance-to-edge, larger N → ≥ confidence.
        assert rl.confidence >= rs.confidence

    def test_zero_mean_baseline_zero_confidence(self) -> None:
        from pmriprep.utils.dsc_preload import detect_preload_from_signal
        sig = np.zeros((4, 4, 1, 20), dtype=np.float32)
        mask = np.ones((4, 4, 1), dtype=bool)
        r = detect_preload_from_signal(
            sig, te=0.03, baseline_frames=5, brain_mask=mask,
            reference_signal=1000.0,
        )
        assert r.confidence == 0.0

    def test_details_includes_new_keys(self) -> None:
        from pmriprep.utils.dsc_preload import detect_preload_from_signal
        sig = self._signal_with_delta_r2(
            target_delta_r2=0.0, reference=1000.0, te=0.03, std_signal=5.0,
        )
        r = detect_preload_from_signal(
            sig, te=0.03, baseline_frames=5, reference_signal=1000.0,
        )
        assert r.details is not None
        assert "tissue_heterogeneity_cv" in r.details
        assert "sem_delta_r2" in r.details
        assert "distance_to_nearest_edge" in r.details
        assert "signal_cv" not in r.details

    def test_confidence_in_unit_interval(self) -> None:
        from pmriprep.utils.dsc_preload import detect_preload_from_signal
        # Random inputs across a range of parameters; confidence must stay in [0, 1].
        rng = np.random.default_rng(42)
        for _ in range(10):
            target = float(rng.uniform(0.0, 8.0))
            std = float(rng.uniform(5.0, 200.0))
            sig = self._signal_with_delta_r2(
                target_delta_r2=target, reference=1000.0, te=0.03,
                std_signal=std,
            )
            r = detect_preload_from_signal(
                sig, te=0.03, baseline_frames=5, reference_signal=1000.0,
            )
            assert 0.0 <= r.confidence <= 1.0


class TestPreloadClassHasUnknown:
    """PreloadClass.UNKNOWN is a defined member."""

    def test_unknown_exists(self) -> None:
        assert PreloadClass.UNKNOWN.value == "unknown"


class TestPreloadConfig:
    """Verify DSCConfig.preload `'unknown' | float` semantics."""

    def test_default_is_unknown(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        config = DSCConfig()
        assert config.preload == "unknown"

    def test_valid_float_dose(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        for frac in [0.0, 0.25, 0.5, 1.0, 1.5]:
            config = DSCConfig(preload=frac)
            assert config.preload == frac

    def test_invalid_string_raises(self) -> None:
        """Named dose-class strings are rejected; preload requires a numeric value."""
        from pmriprep.config.dsc import DSCConfig
        for enum_val in [
            "auto", "half_dose", "significant", "severe", "minimal", "none", "yes",
        ]:
            with pytest.raises(ValueError, match="Invalid preload"):
                DSCConfig(preload=enum_val)  # type: ignore[arg-type]

    def test_auto_value_points_to_unknown(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        with pytest.raises(ValueError, match=r"'auto'.*Use 'unknown'"):
            DSCConfig(preload="auto")  # type: ignore[arg-type]

    def test_out_of_range_raises(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        with pytest.raises(ValueError, match="outside valid range"):
            DSCConfig(preload=2.5)
        with pytest.raises(ValueError, match="outside valid range"):
            DSCConfig(preload=-0.1)


class TestCLIPreloadType:
    """Parsing of --dsc-preload as 'unknown' | float via _preload_type()."""

    def test_unknown_passes_through(self) -> None:
        from pmriprep.cli.run import _preload_type
        assert _preload_type("unknown") == "unknown"

    def test_zero_is_user_specified(self) -> None:
        """0.0 must be preserved (not falsy-coerced to None/unknown)."""
        from pmriprep.cli.run import _preload_type
        val = _preload_type("0.0")
        assert val == 0.0
        assert val is not None
        # Downstream config-merge path uses `is not None`, so this is a
        # regression guard against accidental truthiness checks.

    def test_valid_floats(self) -> None:
        from pmriprep.cli.run import _preload_type
        for s, expected in [("0.25", 0.25), ("0.5", 0.5), ("1.0", 1.0), ("2.0", 2.0)]:
            assert _preload_type(s) == expected

    def test_invalid_string_raises(self) -> None:
        import argparse
        from pmriprep.cli.run import _preload_type
        with pytest.raises(argparse.ArgumentTypeError, match="must be 'unknown'"):
            _preload_type("half_dose")
        with pytest.raises(argparse.ArgumentTypeError, match="must be 'unknown'"):
            _preload_type("yes")
        with pytest.raises(argparse.ArgumentTypeError, match="must be 'unknown'"):
            _preload_type("auto")

    def test_out_of_range_raises(self) -> None:
        import argparse
        from pmriprep.cli.run import _preload_type
        with pytest.raises(argparse.ArgumentTypeError, match=r"\[0\.0, 2\.0\]"):
            _preload_type("2.5")
        with pytest.raises(argparse.ArgumentTypeError, match=r"\[0\.0, 2\.0\]"):
            _preload_type("-0.1")

    def test_non_finite_raises(self) -> None:
        import argparse
        from pmriprep.cli.run import _preload_type
        with pytest.raises(argparse.ArgumentTypeError):
            _preload_type("inf")
        with pytest.raises(argparse.ArgumentTypeError):
            _preload_type("nan")
