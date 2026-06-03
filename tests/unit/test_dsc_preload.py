"""Unit tests for DSC preload detection and classification."""

import numpy as np
import pytest

from pmriprep.utils.dsc_preload import (
    PreloadClass,
    classify_preload,
    classify_preload_adequacy_from_k2,
    get_preload_thresholds,
    _compute_ees_concentration,
)
from pmriprep.utils.dsc_constants import (
    KEP_DSC_DEFAULT,
    KEP_DSC_LOWER,
    KEP_DSC_UPPER,
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


class TestKepConstants:
    """Tests for KEP DSC constants."""

    def test_kep_default_in_range(self) -> None:
        """KEP_DSC_DEFAULT should be within valid range."""
        assert KEP_DSC_LOWER <= KEP_DSC_DEFAULT <= KEP_DSC_UPPER

    def test_kep_default_matches_leu_2016(self) -> None:
        """KEP_DSC_DEFAULT should match Leu 2016 GBM mean (~0.005 s⁻¹)."""
        assert KEP_DSC_DEFAULT == pytest.approx(0.005, rel=0.01)

    def test_kep_bounds_order(self) -> None:
        """Lower bound should be less than upper bound."""
        assert KEP_DSC_LOWER < KEP_DSC_UPPER


class TestComputeEESConcentration:
    """Tests for _compute_ees_concentration() helper."""

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        c_bar = np.sin(np.linspace(0, np.pi, 60))
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert c_e.shape == c_bar.shape

    def test_output_dtype(self) -> None:
        """Output should be float64."""
        c_bar = np.sin(np.linspace(0, np.pi, 60))
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert c_e.dtype == np.float64

    def test_zero_input_gives_zero_output(self) -> None:
        """Zero input should give zero output."""
        c_bar = np.zeros(60)
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert np.allclose(c_e, 0)

    def test_zero_kep_approaches_cumsum(self) -> None:
        """When kep=0, exp(-kep*t)=1, so convolution gives cumsum."""
        c_bar = np.ones(10)  # Step function
        tr = 1.0

        # kep=0 gives exp_kernel=1 everywhere, so convolution = cumsum * tr
        c_e = _compute_ees_concentration(c_bar, kep=0.0, tr=tr)
        expected = np.cumsum(c_bar) * tr

        np.testing.assert_allclose(c_e, expected, rtol=1e-10)

    def test_positive_input_gives_positive_output(self) -> None:
        """Positive concentration should give positive C_e."""
        c_bar = np.abs(np.sin(np.linspace(0, np.pi, 60)))
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert np.all(c_e >= 0)

    def test_high_kep_causes_faster_decay(self) -> None:
        """Higher kep should cause faster decay after bolus."""
        # Single pulse input
        c_bar = np.exp(-((np.arange(60) - 20) ** 2) / 50)
        tr = 1.5

        c_e_low = _compute_ees_concentration(c_bar, kep=0.001, tr=tr)
        c_e_high = _compute_ees_concentration(c_bar, kep=0.05, tr=tr)

        # At late timepoints, high kep should decay more
        assert c_e_high[-1] < c_e_low[-1], (
            "High kep should give lower C_e at late timepoints"
        )


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


class TestPreloadClassFromDose:
    """Bin user-specified administered preload fraction to class."""

    def test_zero_is_none(self) -> None:
        from pmriprep.utils.dsc_preload import preload_class_from_dose
        assert preload_class_from_dose(0.0) == PreloadClass.NONE

    def test_small_is_minimal(self) -> None:
        from pmriprep.utils.dsc_preload import preload_class_from_dose
        assert preload_class_from_dose(0.05) == PreloadClass.MINIMAL
        assert preload_class_from_dose(0.10) == PreloadClass.MINIMAL

    def test_quarter_dose_is_half_dose(self) -> None:
        from pmriprep.utils.dsc_preload import preload_class_from_dose
        assert preload_class_from_dose(0.25) == PreloadClass.HALF_DOSE
        assert preload_class_from_dose(0.5) == PreloadClass.HALF_DOSE

    def test_full_is_significant(self) -> None:
        from pmriprep.utils.dsc_preload import preload_class_from_dose
        assert preload_class_from_dose(1.0) == PreloadClass.SIGNIFICANT
        assert preload_class_from_dose(0.75) == PreloadClass.SIGNIFICANT

    def test_double_is_severe(self) -> None:
        from pmriprep.utils.dsc_preload import preload_class_from_dose
        assert preload_class_from_dose(1.5) == PreloadClass.SEVERE
        assert preload_class_from_dose(2.0) == PreloadClass.SEVERE

    def test_boundary_edges(self) -> None:
        """Edges are lower-inclusive: [lo, hi)."""
        from pmriprep.utils.dsc_preload import preload_class_from_dose
        assert preload_class_from_dose(0.01) == PreloadClass.MINIMAL
        assert preload_class_from_dose(0.20) == PreloadClass.HALF_DOSE
        assert preload_class_from_dose(0.60) == PreloadClass.SIGNIFICANT
        assert preload_class_from_dose(1.15) == PreloadClass.SEVERE

    def test_negative_raises(self) -> None:
        from pmriprep.utils.dsc_preload import preload_class_from_dose
        with pytest.raises(ValueError, match="non-negative"):
            preload_class_from_dose(-0.1)

    def test_nan_raises(self) -> None:
        from pmriprep.utils.dsc_preload import preload_class_from_dose
        with pytest.raises(ValueError, match="finite"):
            preload_class_from_dose(float("nan"))


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


# =============================================================================
# K2-primary preload adequacy classifier
# =============================================================================


class TestClassifyPreloadAdequacyFromK2:
    """K2-primary classifier (dsc_preload.classify_preload_adequacy_from_k2)."""

    def test_cohort_center_is_adequate(self) -> None:
        """k2_median=0.002, leaky=0.1% (institutional cohort center) → HALF_DOSE."""
        cls, conf, reason = classify_preload_adequacy_from_k2(
            {"k2_median": 0.002, "percent_leaky_voxels": 0.1}
        )
        assert cls == PreloadClass.HALF_DOSE
        assert conf >= 0.5
        assert "adequate" in reason.lower()

    def test_boundary_below_adequate_threshold(self) -> None:
        """k2_median=0.004 just below 0.005 threshold, leaky=0.5% < 1.0% → HALF_DOSE."""
        cls, conf, _ = classify_preload_adequacy_from_k2(
            {"k2_median": 0.004, "percent_leaky_voxels": 0.5}
        )
        assert cls == PreloadClass.HALF_DOSE
        assert conf >= 0.5

    def test_partial_tier(self) -> None:
        """k2_median=0.007, leaky=2% → MINIMAL (between adequate and inadequate)."""
        cls, _, reason = classify_preload_adequacy_from_k2(
            {"k2_median": 0.007, "percent_leaky_voxels": 2.0}
        )
        assert cls == PreloadClass.MINIMAL
        assert "partial" in reason.lower()

    def test_inadequate_k2_high(self) -> None:
        """k2_median=0.02 exceeds INADEQUATE threshold → NONE."""
        cls, _, reason = classify_preload_adequacy_from_k2(
            {"k2_median": 0.02, "percent_leaky_voxels": 0.1}
        )
        assert cls == PreloadClass.NONE
        assert "inadequate" in reason.lower()

    def test_empty_dict_is_unknown(self) -> None:
        cls, conf, _ = classify_preload_adequacy_from_k2({})
        assert cls == PreloadClass.UNKNOWN
        assert conf == 0.0

    def test_none_is_unknown(self) -> None:
        cls, conf, _ = classify_preload_adequacy_from_k2(None)
        assert cls == PreloadClass.UNKNOWN
        assert conf == 0.0

    def test_missing_leaky_key_is_unknown(self) -> None:
        cls, _, _ = classify_preload_adequacy_from_k2(
            {"k2_median": 0.002}  # no percent_leaky_voxels
        )
        assert cls == PreloadClass.UNKNOWN


class TestPreloadClassHasUnknown:
    """PreloadClass.UNKNOWN is a defined member."""

    def test_unknown_exists(self) -> None:
        assert PreloadClass.UNKNOWN.value == "unknown"

    def test_unknown_not_in_dose_edges(self) -> None:
        from pmriprep.utils.dsc_preload import _PRELOAD_DOSE_EDGES
        # UNKNOWN is never produced from user dose input — should NOT be
        # in the dose-to-class binning table.
        assert PreloadClass.UNKNOWN not in _PRELOAD_DOSE_EDGES
