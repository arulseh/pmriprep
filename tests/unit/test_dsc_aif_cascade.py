"""Tests for AIF cascade minimum candidate acceptance logic."""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import (
    MIN_AIF_CANDIDATES_ACCEPT,
    _CASCADE_N_CANDIDATES_FRACTION,
)


def _make_scored(
    n: int,
) -> list[tuple[float, tuple[int, int, int], dict]]:
    """Create a list of n fake scored candidates."""
    return [
        (float(i) * 0.1, (i, 0, 0), {"snr": 10.0})
        for i in range(n)
    ]


class TestMinAifCandidatesConstant:
    """Verify the module-level constant exists and is sensible."""

    def test_value(self) -> None:
        assert MIN_AIF_CANDIDATES_ACCEPT == 3

    def test_is_int(self) -> None:
        assert isinstance(MIN_AIF_CANDIDATES_ACCEPT, int)


class TestCascadeMinCandidates:
    """Test cascade break/continue logic with MIN_AIF_CANDIDATES_ACCEPT.

    These tests exercise the cascade loop logic in isolation by simulating
    the scored-candidates list at each level and verifying which level
    is ultimately selected.
    """

    @staticmethod
    def _run_cascade(
        level_results: dict[str, list],
    ) -> tuple[str, list, bool]:
        """Simulate the cascade loop logic from _run_interface.

        Args:
            level_results: mapping from level_name to scored candidates
                returned at that level. Levels are iterated in order.

        Returns:
            (detection_level, top_candidates, broke_early)
        """
        detection_level = "synthetic"
        top_candidates: list = []
        cascade_initial_level = None
        broke_early = False

        for level_name, scored in level_results.items():
            if not scored:
                continue

            if cascade_initial_level is None:
                cascade_initial_level = level_name

            if len(scored) >= MIN_AIF_CANDIDATES_ACCEPT:
                top_candidates = scored
                detection_level = level_name
                broke_early = True
                break
            elif len(scored) > len(top_candidates):
                top_candidates = scored
                detection_level = level_name

        return detection_level, top_candidates, broke_early

    def test_sufficient_strict_breaks(self) -> None:
        """5 strict candidates → break at strict, don't continue."""
        levels = {
            "atlas_strict": _make_scored(5),
            "brain_relaxed": _make_scored(20),
        }
        level, cands, broke = self._run_cascade(levels)
        assert level == "atlas_strict"
        assert len(cands) == 5
        assert broke is True

    def test_insufficient_strict_continues_to_relaxed(self) -> None:
        """1 strict candidate → continues to relaxed (4 candidates)."""
        levels = {
            "atlas_strict": _make_scored(1),
            "brain_relaxed": _make_scored(4),
        }
        level, cands, broke = self._run_cascade(levels)
        assert level == "brain_relaxed"
        assert len(cands) == 4
        assert broke is True

    def test_two_strict_four_relaxed_uses_relaxed(self) -> None:
        """2 strict + 4 relaxed → uses relaxed (2 < MIN_AIF_CANDIDATES_ACCEPT)."""
        levels = {
            "atlas_strict": _make_scored(2),
            "brain_relaxed": _make_scored(4),
        }
        level, cands, broke = self._run_cascade(levels)
        assert level == "brain_relaxed"
        assert len(cands) == 4
        assert broke is True

    def test_all_below_minimum_keeps_best(self) -> None:
        """All levels below minimum → keeps the one with most candidates."""
        levels = {
            "atlas_strict": _make_scored(1),
            "brain_relaxed": _make_scored(2),
            "brain_permissive": _make_scored(2),
        }
        level, cands, broke = self._run_cascade(levels)
        # brain_relaxed found 2 first, permissive has 2 but not > 2
        assert level == "brain_relaxed"
        assert len(cands) == 2
        assert broke is False

    def test_no_candidates_stays_synthetic(self) -> None:
        """No candidates at any level → stays synthetic."""
        levels = {
            "atlas_strict": [],
            "brain_relaxed": [],
        }
        level, cands, broke = self._run_cascade(levels)
        assert level == "synthetic"
        assert len(cands) == 0
        assert broke is False

    def test_exactly_three_breaks(self) -> None:
        """Exactly MIN_AIF_CANDIDATES_ACCEPT candidates → breaks."""
        levels = {
            "atlas_strict": _make_scored(3),
            "brain_relaxed": _make_scored(10),
        }
        level, cands, broke = self._run_cascade(levels)
        assert level == "atlas_strict"
        assert len(cands) == 3
        assert broke is True


class TestGraduatedNCandidates:
    """Test graduated n_candidates per cascade level."""

    @staticmethod
    def _level_n(crit: str, n_candidates: int) -> int:
        """Compute level_n_candidates the same way the cascade does."""
        frac = _CASCADE_N_CANDIDATES_FRACTION.get(crit, 0.6)
        return max(MIN_AIF_CANDIDATES_ACCEPT, math.ceil(frac * n_candidates))

    def test_fraction_strict(self) -> None:
        """Strict uses full n_candidates (10 → 10)."""
        assert self._level_n("strict", 10) == 10

    def test_fraction_relaxed(self) -> None:
        """Relaxed uses ceil(0.6 * 10) = 6."""
        assert self._level_n("relaxed", 10) == 6

    def test_fraction_permissive(self) -> None:
        """Permissive uses ceil(0.4 * 10) = 4."""
        assert self._level_n("permissive", 10) == 4

    def test_fraction_degraded(self) -> None:
        """Degraded floors at MIN_AIF_CANDIDATES_ACCEPT = 3."""
        assert self._level_n("degraded", 10) == MIN_AIF_CANDIDATES_ACCEPT

    def test_small_n_floors(self) -> None:
        """n_candidates=5 → relaxed=3, permissive=3, degraded=3."""
        assert self._level_n("relaxed", 5) == 3
        assert self._level_n("permissive", 5) == 3
        assert self._level_n("degraded", 5) == MIN_AIF_CANDIDATES_ACCEPT

    def test_cascade_prefers_stricter_at_equal_count(self) -> None:
        """2 strict stashed, 2 relaxed doesn't replace (not >)."""
        levels = {
            "atlas_strict": _make_scored(2),
            "brain_relaxed": _make_scored(2),
        }
        level, cands, broke = TestCascadeMinCandidates._run_cascade(levels)
        # strict found 2 first; relaxed has 2 but not > 2, so strict kept
        assert level == "atlas_strict"
        assert len(cands) == 2
        assert broke is False
