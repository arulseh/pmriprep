"""Unit tests for DSC AIF quality assessment and candidate scoring."""

import numpy as np
import pytest

from pmriprep.utils.dsc_aif import (
    AIFCandidateMetrics,
    AIFQualityResult,
    AIFSearchResult,
    assess_aif_quality,
    compute_candidate_metrics,
    compute_peak_time,
    score_aif_candidate,
    get_aif_thresholds,
    _compute_quality_score,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_good_aif(
    n_frames: int = 40,
    baseline_frames: int = 5,
    peak_value: float = 20.0,
    peak_frame: int = 12,
    sigma: float = 3.0,
    tail_value: float = 5.0,
) -> np.ndarray:
    """Create a synthetic AIF with known characteristics.

    Returns a curve with a clean zero baseline, Gaussian-like peak, and
    exponential washout to a specified tail level.
    """
    t = np.arange(n_frames, dtype=np.float64)
    aif = np.zeros(n_frames, dtype=np.float64)

    # Gaussian-ish peak after baseline
    bolus = peak_value * np.exp(-0.5 * ((t - peak_frame) / sigma) ** 2)
    aif[baseline_frames:] = bolus[baseline_frames:]

    # Exponential washout in the tail
    decay_start = peak_frame + 5
    if decay_start < n_frames:
        for i in range(decay_start, n_frames):
            decay = peak_value * np.exp(-0.5 * ((decay_start - peak_frame) / sigma) ** 2)
            tau = 8.0
            aif[i] = tail_value + (decay - tail_value) * np.exp(-(i - decay_start) / tau)

    return aif


# =============================================================================
# TestComputePeakTime
# =============================================================================


class TestComputePeakTime:
    """Tests for compute_peak_time."""

    def test_known_peak(self) -> None:
        """Peak at frame 7 returns 7."""
        aif = np.zeros(30, dtype=np.float64)
        aif[7] = 15.0
        assert compute_peak_time(aif) == 7

    def test_flat_signal(self) -> None:
        """All zeros returns 0 (argmax of flat array)."""
        aif = np.zeros(20, dtype=np.float64)
        assert compute_peak_time(aif) == 0


# =============================================================================
# TestComputeCandidateMetrics
# =============================================================================


class TestComputeCandidateMetrics:
    """Tests for compute_candidate_metrics."""

    def test_synthetic_aif(self) -> None:
        """Synthetic AIF with known properties returns expected metrics."""
        baseline_frames = 5
        n_frames = 30

        # Build a synthetic AIF: zero baseline, sharp peak at frame 8,
        # exponential washout to ~5.0 in the tail.
        t = np.arange(n_frames, dtype=np.float64)
        aif = np.zeros(n_frames, dtype=np.float64)

        # Gamma-variate-like shape peaking near frame 8
        aif[baseline_frames:] = 20.0 * (
            (t[baseline_frames:] - baseline_frames) / 3.0
        ) * np.exp(-(t[baseline_frames:] - baseline_frames) / 4.0)

        metrics = compute_candidate_metrics(aif, baseline_frames=baseline_frames)

        # CNR should be high (zero-noise baseline triggers epsilon guard → 100)
        assert metrics.cnr > 5.0
        # Washout should be positive (peak decays to tail)
        assert metrics.washout > 0.2
        # FWHM should be reasonable (a few frames)
        assert metrics.fwhm >= 3.0
        # Baseline CV should be near-zero (clean baseline)
        assert metrics.baseline_cv < 0.1
        # Peak time near frame 8 (gamma-variate peak = alpha*beta + t0 = 3*4+5 ≈ 8 post-baseline)
        assert 7 <= metrics.peak_time <= 10
        # Peak value around 20 * (alpha^alpha * exp(-alpha)) ≈ 20 * 1.1 * 0.05 ≈ ~4.9
        # Actually for (t/3)*exp(-t/4) at peak: peak_t = 3*4/(3+4)... let's just verify positive
        assert metrics.peak_value > 0.0


# =============================================================================
# TestAssessAIFQuality
# =============================================================================


class TestAssessAIFQuality:
    """Tests for assess_aif_quality."""

    def test_strict_pass(self) -> None:
        """Good AIF with high CNR passes strict and relaxed criteria."""
        baseline_frames = 5
        n_frames = 60
        t = np.arange(n_frames, dtype=np.float64)

        # Build AIF with controlled noise baseline and strong peak.
        # Baseline must have a positive mean so that CV = std/|mean| stays small.
        rng = np.random.default_rng(42)
        aif = np.zeros(n_frames, dtype=np.float64)
        aif[:baseline_frames] = 100.0 + 0.5 * rng.standard_normal(baseline_frames)

        # Sharp peak at frame 12, FWHM ~7-8 frames, on top of baseline
        bolus = 25.0 * np.exp(-0.5 * ((t - 12.0) / 3.0) ** 2)
        aif[baseline_frames:] = 100.0 + bolus[baseline_frames:]

        # Washout tail
        for i in range(20, n_frames):
            aif[i] = 106.0 + (aif[19] - 106.0) * np.exp(-(i - 19) / 5.0)

        result = assess_aif_quality(aif, baseline_frames=baseline_frames)

        assert result.status != "POOR"
        assert result.meets_strict is True
        assert result.meets_relaxed is True

    def test_relaxed_only(self) -> None:
        """Marginal AIF passes relaxed but not strict criteria.

        Construct an AIF where CNR is between relaxed (2.0) and strict (4.0).
        Use deterministic baseline values to control baseline_std exactly.
        """
        baseline_frames = 5
        n_frames = 60
        t = np.arange(n_frames, dtype=np.float64)

        # Deterministic baseline: mean=100, std=1.0 exactly
        # 5 values that have mean=100 and std=1.0:
        # [-1.5, -0.5, 0, 0.5, 1.5] has mean=0, std=sqrt(mean of squares)=sqrt(1.1)
        # Use simpler approach: [99, 100, 101, 99, 101] → mean=100, std=1.0
        baseline_vals = np.array([99.0, 100.0, 101.0, 99.0, 101.0])
        assert np.mean(baseline_vals) == pytest.approx(100.0)
        baseline_std = float(np.std(baseline_vals))

        aif = np.full(n_frames, 100.0, dtype=np.float64)
        aif[:baseline_frames] = baseline_vals

        # Peak offset chosen so CNR = peak_offset / baseline_std ≈ 3
        # CNR strict threshold is 4.0, relaxed is 2.0
        peak_offset = 3.0 * baseline_std
        bolus = peak_offset * np.exp(-0.5 * ((t - 12.0) / 3.0) ** 2)
        aif[baseline_frames:] = 100.0 + bolus[baseline_frames:]

        # Washout tail near baseline
        for i in range(20, n_frames):
            aif[i] = 100.5

        result = assess_aif_quality(aif, baseline_frames=baseline_frames)

        assert result.meets_strict is False
        assert result.meets_relaxed is True

    def test_poor(self) -> None:
        """Bad AIF with no bolus gets POOR status for both criteria."""
        aif = np.zeros(30, dtype=np.float64)
        # Tiny random noise, no real bolus
        rng = np.random.default_rng(0)
        aif += 0.001 * rng.standard_normal(30)

        result = assess_aif_quality(aif, baseline_frames=5)

        assert result.status == "POOR"
        assert result.meets_strict is False
        assert result.meets_relaxed is False

    def test_criteria_details(self) -> None:
        """Good AIF produces criteria_details with expected keys and PASS/WARN/POOR values."""
        aif = _make_good_aif()
        result = assess_aif_quality(aif, baseline_frames=5)

        expected_keys = {"cnr", "washout", "fwhm", "baseline_cv"}
        assert expected_keys.issubset(result.criteria_details.keys())

        valid_statuses = {"PASS", "WARN", "POOR"}
        for key in expected_keys:
            assert result.criteria_details[key] in valid_statuses, (
                f"criteria_details['{key}'] = '{result.criteria_details[key]}' "
                f"not in {valid_statuses}"
            )


# =============================================================================
# TestComputeQualityScore
# =============================================================================


class TestComputeQualityScore:
    """Tests for _compute_quality_score."""

    def test_weights(self) -> None:
        """Good metrics produce a score between 0 and 1."""
        thresholds = get_aif_thresholds("strict")
        score = _compute_quality_score(
            cnr=10.0,
            washout=0.5,
            fwhm=8,
            baseline_cv=0.03,
            thresholds=thresholds,
        )
        assert 0.0 <= score <= 1.0

    def test_bad_metrics_low_score(self) -> None:
        """Bad metrics produce a score below 0.3."""
        thresholds = get_aif_thresholds("strict")
        score = _compute_quality_score(
            cnr=0.5,
            washout=0.0,
            fwhm=25,
            baseline_cv=0.5,
            thresholds=thresholds,
        )
        assert score < 0.3


# =============================================================================
# TestScoreAIFCandidate
# =============================================================================


class TestScoreAIFCandidate:
    """Tests for score_aif_candidate."""

    def test_good_metrics_high_score(self) -> None:
        """Good candidate metrics yield a score above 0.5."""
        metrics = AIFCandidateMetrics(
            cnr=15.0,
            washout=0.5,
            fwhm=8.0,
            baseline_cv=0.03,
            peak_time=12,  # 12 - 5 = 7 frames after baseline
            peak_value=20.0,
        )
        thresholds = get_aif_thresholds("strict")
        score = score_aif_candidate(metrics, thresholds, baseline_frames=5)
        assert score > 0.5

    def test_hard_threshold_violation(self) -> None:
        """CNR below threshold returns score of exactly 0."""
        metrics = AIFCandidateMetrics(
            cnr=0.5,
            washout=0.5,
            fwhm=8.0,
            baseline_cv=0.03,
            peak_time=12,
            peak_value=20.0,
        )
        thresholds = get_aif_thresholds("strict")
        assert thresholds["cnr_min"] > 0.5  # Confirm threshold is above 0.5
        score = score_aif_candidate(metrics, thresholds, baseline_frames=5)
        assert score == 0.0

    def test_ideal_washout_scores_higher(self) -> None:
        """Washout of 0.5 (ideal) scores higher than extreme washout of 0.95."""
        thresholds = get_aif_thresholds("strict")

        metrics_ideal = AIFCandidateMetrics(
            cnr=15.0, washout=0.5, fwhm=9.0, baseline_cv=0.03,
            peak_time=12, peak_value=20.0,
        )
        metrics_extreme = AIFCandidateMetrics(
            cnr=15.0, washout=0.95, fwhm=9.0, baseline_cv=0.03,
            peak_time=12, peak_value=20.0,
        )

        score_ideal = score_aif_candidate(metrics_ideal, thresholds, baseline_frames=5)
        score_extreme = score_aif_candidate(metrics_extreme, thresholds, baseline_frames=5)

        assert score_ideal > score_extreme

    def test_ideal_fwhm_scores_higher(self) -> None:
        """FWHM of 9 (ideal) scores higher than FWHM of 14 (wide but within bounds)."""
        thresholds = get_aif_thresholds("strict")

        metrics_ideal = AIFCandidateMetrics(
            cnr=15.0, washout=0.5, fwhm=9.0, baseline_cv=0.03,
            peak_time=12, peak_value=20.0,
        )
        metrics_wide = AIFCandidateMetrics(
            cnr=15.0, washout=0.5, fwhm=14.0, baseline_cv=0.03,
            peak_time=12, peak_value=20.0,
        )

        score_ideal = score_aif_candidate(metrics_ideal, thresholds, baseline_frames=5)
        score_wide = score_aif_candidate(metrics_wide, thresholds, baseline_frames=5)

        assert score_ideal > score_wide


# =============================================================================
# TestDataclasses
# =============================================================================


class TestDataclasses:
    """Tests for AIF dataclass construction and field access."""

    def test_aif_candidate_metrics_fields(self) -> None:
        """AIFCandidateMetrics fields are accessible after construction."""
        metrics = AIFCandidateMetrics(
            cnr=10.0,
            washout=0.4,
            fwhm=7.0,
            baseline_cv=0.05,
            peak_time=8,
            peak_value=15.0,
        )
        assert metrics.cnr == 10.0
        assert metrics.washout == 0.4
        assert metrics.fwhm == 7.0
        assert metrics.baseline_cv == 0.05
        assert metrics.peak_time == 8
        assert metrics.peak_value == 15.0

    def test_aif_quality_result_fields(self) -> None:
        """AIFQualityResult fields are accessible after construction."""
        result = AIFQualityResult(
            status="PASS",
            quality_score=0.85,
            washout_fraction=0.5,
            fwhm_frames=8,
            baseline_cv=0.03,
            cnr=12.0,
            peak_value=20.0,
            peak_time=10,
            meets_strict=True,
            meets_relaxed=True,
            criteria_details={"cnr": "PASS", "washout": "PASS"},
        )
        assert result.status == "PASS"
        assert result.quality_score == 0.85
        assert result.washout_fraction == 0.5
        assert result.fwhm_frames == 8
        assert result.baseline_cv == 0.03
        assert result.cnr == 12.0
        assert result.peak_value == 20.0
        assert result.peak_time == 10
        assert result.meets_strict is True
        assert result.meets_relaxed is True
        assert "cnr" in result.criteria_details
        assert "washout" in result.criteria_details
