"""Tests for BIC model selection enhancement gate."""

import pytest

from pmriprep.interfaces.dce_compute import _gate_bic_candidates
from pmriprep.utils.dce_constants import TOFTS_ENHANCEMENT_GATE_MM


class TestGateBicCandidates:
    """Tests for _gate_bic_candidates — Standard Tofts exclusion at low enhancement."""

    def test_tofts_excluded_below_threshold(self) -> None:
        """Standard Tofts removed when max enhancement < 0.05 mM."""
        candidates = {"patlak": "mock_p", "tofts": "mock_t", "extended_tofts": "mock_et"}
        result = _gate_bic_candidates(0.03, candidates)
        assert "tofts" not in result
        assert "patlak" in result
        assert "extended_tofts" in result

    def test_tofts_kept_above_threshold(self) -> None:
        """Standard Tofts remains when max enhancement >= 0.05 mM."""
        candidates = {"patlak": "mock_p", "tofts": "mock_t", "extended_tofts": "mock_et"}
        result = _gate_bic_candidates(0.10, candidates)
        assert "tofts" in result
        assert len(result) == 3

    def test_boundary_at_threshold(self) -> None:
        """At exactly threshold, Tofts is kept (< is strict)."""
        candidates = {"patlak": "mock_p", "tofts": "mock_t"}
        result = _gate_bic_candidates(TOFTS_ENHANCEMENT_GATE_MM, candidates)
        assert "tofts" in result

    def test_no_tofts_in_candidates(self) -> None:
        """No-op when Tofts isn't a candidate."""
        candidates = {"patlak": "mock_p", "extended_tofts": "mock_et"}
        result = _gate_bic_candidates(0.01, candidates)
        assert result == candidates

    def test_empty_candidates(self) -> None:
        """No-op on empty candidate dict."""
        result = _gate_bic_candidates(0.01, {})
        assert result == {}

    def test_only_tofts_candidate(self) -> None:
        """If Tofts is the only candidate and below threshold, returns empty."""
        candidates = {"tofts": "mock_t"}
        result = _gate_bic_candidates(0.01, candidates)
        assert result == {}

    def test_custom_threshold(self) -> None:
        """Custom threshold overrides default."""
        candidates = {"patlak": "mock_p", "tofts": "mock_t"}
        # Below custom threshold of 0.10
        result = _gate_bic_candidates(0.08, candidates, threshold=0.10)
        assert "tofts" not in result
        # Above custom threshold
        result = _gate_bic_candidates(0.12, candidates, threshold=0.10)
        assert "tofts" in result
