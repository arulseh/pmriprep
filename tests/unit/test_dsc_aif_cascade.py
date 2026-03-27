"""Tests for AIF cascade minimum candidate acceptance logic."""

import math
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import (
    GAMMA_PEAK_RATIO_CASCADE_MAX,
    GAMMA_PEAK_RATIO_CASCADE_MIN,
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


class TestCascadeQualityGate:
    """Test cascade quality gate based on gamma peak ratio.

    When a tier meets MIN_AIF_CANDIDATES_ACCEPT but has a bad peak ratio,
    the cascade continues once to the next tier. The better peak ratio wins.
    """

    @staticmethod
    def _run_cascade_with_quality(
        level_results: dict[str, tuple[list, Optional[float]]],
    ) -> tuple[str, list, bool, bool]:
        """Simulate cascade loop with quality gate.

        Args:
            level_results: mapping from level_name to (scored, peak_ratio).
                peak_ratio=None means gamma fit failed for that tier.

        Returns:
            (detection_level, top_candidates, broke_early, quality_gate_triggered)
        """
        detection_level = "synthetic"
        top_candidates: list = []
        quality_stash: Optional[tuple[list, str, Optional[float]]] = None
        broke_early = False
        quality_gate_triggered = False

        for level_name, (scored, peak_ratio) in level_results.items():
            if not scored:
                continue

            if len(scored) >= MIN_AIF_CANDIDATES_ACCEPT:
                # Quality gate: check peak ratio before accepting
                if quality_stash is None:
                    if (peak_ratio is not None
                            and (peak_ratio < GAMMA_PEAK_RATIO_CASCADE_MIN
                                 or peak_ratio > GAMMA_PEAK_RATIO_CASCADE_MAX)):
                        quality_stash = (scored, level_name, peak_ratio)
                        quality_gate_triggered = True
                        continue

                if quality_stash is not None:
                    # Compare with stashed
                    stashed_pr = quality_stash[2]
                    use_current = (
                        peak_ratio is not None
                        and (stashed_pr is None
                             or abs(peak_ratio - 1.0) < abs(stashed_pr - 1.0))
                    )
                    if use_current:
                        top_candidates = scored
                        detection_level = level_name
                    else:
                        top_candidates = quality_stash[0]
                        detection_level = quality_stash[1]
                    broke_early = True
                    break

                # Good quality — accept
                top_candidates = scored
                detection_level = level_name
                broke_early = True
                break
            elif len(scored) > len(top_candidates):
                top_candidates = scored
                detection_level = level_name

        # Post-loop: use stash if loop exhausted
        if quality_stash is not None and (
            not top_candidates
            or len(top_candidates) < MIN_AIF_CANDIDATES_ACCEPT
        ):
            top_candidates = quality_stash[0]
            detection_level = quality_stash[1]

        return detection_level, top_candidates, broke_early, quality_gate_triggered

    def test_bad_peak_ratio_continues_to_next_tier(self) -> None:
        """Strict with bad peak ratio → continues to relaxed with good ratio."""
        levels = {
            "atlas_strict": (_make_scored(3), 0.84),
            "atlas_relaxed": (_make_scored(6), 0.96),
        }
        level, cands, broke, triggered = self._run_cascade_with_quality(levels)
        assert level == "atlas_relaxed"
        assert len(cands) == 6
        assert triggered is True

    def test_good_peak_ratio_breaks_normally(self) -> None:
        """Strict with good peak ratio → breaks at strict (no change)."""
        levels = {
            "atlas_strict": (_make_scored(3), 0.95),
            "atlas_relaxed": (_make_scored(6), 0.96),
        }
        level, cands, broke, triggered = self._run_cascade_with_quality(levels)
        assert level == "atlas_strict"
        assert len(cands) == 3
        assert broke is True
        assert triggered is False

    def test_bad_both_tiers_uses_closer_to_one(self) -> None:
        """Both tiers bad → picks whichever has peak ratio closer to 1.0."""
        levels = {
            "atlas_strict": (_make_scored(3), 0.84),
            "atlas_relaxed": (_make_scored(6), 0.82),
        }
        level, cands, broke, triggered = self._run_cascade_with_quality(levels)
        # 0.84 is closer to 1.0 than 0.82
        assert level == "atlas_strict"
        assert len(cands) == 3
        assert triggered is True

    def test_bad_strict_no_relaxed_candidates_uses_stashed(self) -> None:
        """Strict bad, relaxed has 0 candidates → uses stashed strict."""
        levels = {
            "atlas_strict": (_make_scored(3), 0.84),
            "atlas_relaxed": ([], None),
        }
        level, cands, broke, triggered = self._run_cascade_with_quality(levels)
        assert level == "atlas_strict"
        assert len(cands) == 3
        assert triggered is True

    def test_quality_gate_triggers_only_once(self) -> None:
        """Gate triggers at strict, compares with relaxed, never reaches permissive."""
        levels = {
            "atlas_strict": (_make_scored(3), 0.84),
            "atlas_relaxed": (_make_scored(4), 0.86),
            "atlas_permissive": (_make_scored(10), 0.98),
        }
        level, cands, broke, triggered = self._run_cascade_with_quality(levels)
        # Compares strict (0.84) vs relaxed (0.86), picks relaxed (closer to 1.0)
        assert level == "atlas_relaxed"
        assert len(cands) == 4
        assert triggered is True

    def test_fewer_candidates_better_ratio_still_wins(self) -> None:
        """Strict 3 cands pr=0.96 vs relaxed 6 cands — strict wins (good ratio)."""
        levels = {
            "atlas_strict": (_make_scored(3), 0.96),
            "atlas_relaxed": (_make_scored(6), 0.98),
        }
        level, cands, broke, triggered = self._run_cascade_with_quality(levels)
        # Good peak ratio at strict → breaks immediately, no quality gate
        assert level == "atlas_strict"
        assert len(cands) == 3
        assert triggered is False

    def test_high_side_peak_ratio_triggers_gate(self) -> None:
        """Peak ratio 1.15 (above 1.12) triggers cascade continuation."""
        levels = {
            "atlas_strict": (_make_scored(3), 1.15),
            "atlas_relaxed": (_make_scored(5), 0.98),
        }
        level, cands, broke, triggered = self._run_cascade_with_quality(levels)
        assert level == "atlas_relaxed"
        assert len(cands) == 5
        assert triggered is True
