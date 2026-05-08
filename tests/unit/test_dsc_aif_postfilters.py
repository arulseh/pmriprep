"""Unit tests for DSC AIF post-score kinetic-coherence filters.

Covers ``AIFDetection._apply_postscore_aif_filters``: the 1-TR
peak-spread filter and the tail/peak threshold. TTP causality lives
in ``_score_candidate_cost`` (upstream), not here, so it is exercised
by the cascade-level tests in ``test_dsc_aif_cascade.py``.
"""

from __future__ import annotations

import numpy as np

from pmriprep.interfaces.dsc_aif import AIFDetection
from pmriprep.utils.dsc_constants import (
    AIF_PEAK_SPREAD_MAX_FRAMES,
    AIF_TAIL_FRACTION_MAX,
)


def _vox(coord, peak_time_frame, tail_fraction_late, cost=1.0):
    """Build a (cost, coord, metrics) tuple."""
    return (
        cost,
        coord,
        {
            "peak_time_frame": float(peak_time_frame),
            "tail_fraction_late": float(tail_fraction_late),
        },
    )


class TestEmptyPool:
    def test_empty_pool_returns_zero_diagnostics(self):
        survivors, diag = AIFDetection._apply_postscore_aif_filters([])
        assert survivors == []
        assert diag["aif_n_after_filters"] == 0
        assert diag["aif_n_filtered_peakspread"] == 0
        assert diag["aif_n_filtered_tail"] == 0
        assert diag["aif_peak_time_spread_frames"] == 0


class TestPeakSpreadFilter:
    """1-TR peak-spread filter: keep voxels within +1 of earliest peak."""

    def test_peak_spread_at_1_frame_kept(self):
        # earliest=25; window=[25, 26]
        scored = [
            _vox((0, 0, 0), 25, 0.05),
            _vox((1, 0, 0), 25, 0.05),
            _vox((2, 0, 0), 26, 0.05),
            _vox((3, 0, 0), 27, 0.05),  # beyond +1, rejected
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored)
        assert len(survivors) == 3
        assert diag["aif_n_filtered_peakspread"] == 1
        assert diag["aif_peak_time_spread_frames"] == 1

    def test_earliest_reference_fixed_for_pass(self):
        """The earliest_pt reference is the input pool's minimum and is
        NOT recomputed after peak-spread or tail filters prune. So if the
        earliest voxel itself fails the tail filter (downstream), other
        voxels are still evaluated against the original earliest_pt.
        """
        # earliest=25; spread_max=26; voxel 28 fails spread regardless
        # of whether voxel 25 is later removed by tail filter.
        scored = [
            _vox((0, 0, 0), 25, 0.20),  # spread pass; tail fail
            _vox((1, 0, 0), 26, 0.05),  # spread pass; tail pass
            _vox((2, 0, 0), 28, 0.05),  # spread FAIL (28 > 25+1)
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored)
        # Only voxel at 26 survives: 25 fails tail, 28 fails peak-spread.
        assert len(survivors) == 1
        assert survivors[0][1] == (1, 0, 0)
        assert diag["aif_n_filtered_peakspread"] == 1
        assert diag["aif_n_filtered_tail"] == 1


class TestTailFilter:
    """tail_fraction_late ≤ 0.15 threshold."""

    def test_tail_at_threshold_kept(self):
        # Use the constant so the test doesn't drift if 0.15 changes.
        thr = AIF_TAIL_FRACTION_MAX
        scored = [
            _vox((0, 0, 0), 25, thr),         # exactly at threshold (kept)
            _vox((1, 0, 0), 25, thr - 0.01),  # below
            _vox((2, 0, 0), 25, thr + 0.01),  # above (rejected)
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored)
        assert len(survivors) == 2
        assert diag["aif_n_filtered_tail"] == 1
        assert diag["aif_tail_max_fraction"] == thr

    def test_nan_tail_rejected(self):
        scored = [
            (1.0, (1, 1, 1), {
                "peak_time_frame": 25.0,
                "tail_fraction_late": float("nan"),
            }),
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored)
        assert survivors == []
        assert diag["aif_n_filtered_tail"] == 1


class TestTwoFiltersCombined:
    def test_mixed_pool(self):
        # earliest = min(25, 49, 26, 28, 25, 26) = 25, spread_max = 26.
        # 49 and 28 fail spread; 26-with-0.20 fails tail; 3 survivors.
        scored = [
            _vox((0, 0, 0), 25, 0.05),  # all pass
            _vox((1, 0, 0), 49, 0.05),  # spread FAIL (49 > 26)
            _vox((2, 0, 0), 26, 0.20),  # spread pass; tail FAIL
            _vox((3, 0, 0), 28, 0.05),  # spread FAIL (28 > 26)
            _vox((4, 0, 0), 25, 0.10),  # all pass
            _vox((5, 0, 0), 26, 0.05),  # all pass
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored)
        assert len(survivors) == 3
        survivor_coords = {c for _, c, _ in survivors}
        assert survivor_coords == {(0, 0, 0), (4, 0, 0), (5, 0, 0)}
        assert diag["aif_n_after_filters"] == 3
        assert diag["aif_n_filtered_peakspread"] == 2
        assert diag["aif_n_filtered_tail"] == 1
        # max(25, 25, 26) - min(25, 25, 26) = 1
        assert diag["aif_peak_time_spread_frames"] == 1

    def test_all_rejected_returns_empty_with_diagnostics(self):
        # Two voxels: one fails spread, the other fails tail.
        scored = [
            _vox((0, 0, 0), 25, 0.50),   # spread pass, tail FAIL
            _vox((1, 0, 0), 30, 0.05),   # spread FAIL (30 > 25+1)
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored)
        assert survivors == []
        assert diag["aif_n_filtered_peakspread"] == 1
        assert diag["aif_n_filtered_tail"] == 1
        assert diag["aif_n_after_filters"] == 0


class TestConstantSanity:
    """Sanity-check the kinetic-filter constants are still wired in."""

    def test_spread_uses_constant(self):
        assert AIF_PEAK_SPREAD_MAX_FRAMES >= 1

    def test_tail_uses_constant(self):
        assert 0.0 < AIF_TAIL_FRACTION_MAX <= 0.5
