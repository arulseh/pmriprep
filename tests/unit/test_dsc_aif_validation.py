"""Unit tests for DSC AIF temporal and shape validation."""

import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection


class TestPSRLowerBound:
    """Tests for PSR lower bound rejection in _score_candidate_cost."""

    @pytest.fixture
    def aif_interface(self) -> AIFDetection:
        """Create minimal AIFDetection instance for testing."""
        return AIFDetection()

    def _make_metrics(self, psr: float = 80.0) -> dict:
        """Create minimal candidate metrics dict."""
        return {
            "pre_bolus_frames": 15,
            "dt": 1.5,
            "ttp": 30.0,  # 30s after start → 9s after baseline end
            "baseline_return_ratio": 15.0,
            "peak_to_baseline_ratio": 5.0,
            "secondary_peak_ratio": 40.0,
            "post_peak_monotonicity": 70.0,
            "fwhm": 6.0,
            "ph": 50.0,
            "psr": psr,
            "s0": 50.0,
            "max_allowed_ttp": 100.0,
        }

    def test_psr_below_strict_threshold_rejected(self, aif_interface: AIFDetection) -> None:
        """PSR < 40% rejected at strict level."""
        metrics = self._make_metrics(psr=30.0)
        cost = aif_interface._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0,
        )
        assert cost == float("inf")

    def test_psr_above_strict_threshold_passes(self, aif_interface: AIFDetection) -> None:
        """PSR = 50% passes at strict level."""
        metrics = self._make_metrics(psr=50.0)
        cost = aif_interface._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0,
        )
        assert cost < float("inf")

    def test_psr_below_relaxed_threshold_rejected(self, aif_interface: AIFDetection) -> None:
        """PSR < 30% rejected at relaxed level."""
        metrics = self._make_metrics(psr=20.0)
        cost = aif_interface._score_candidate_cost(
            metrics, criteria="relaxed", median_s0=200.0,
        )
        assert cost == float("inf")

    def test_psr_nan_not_rejected(self, aif_interface: AIFDetection) -> None:
        """PSR = NaN should not trigger rejection."""
        metrics = self._make_metrics(psr=float("nan"))
        cost = aif_interface._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0,
        )
        assert cost < float("inf")

    def test_psr_above_max_still_rejected(self, aif_interface: AIFDetection) -> None:
        """PSR > 110% still rejected at strict (existing PSR max check)."""
        metrics = self._make_metrics(psr=120.0)
        cost = aif_interface._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0,
        )
        assert cost == float("inf")
