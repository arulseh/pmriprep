"""Unit tests for DSC AIF post-score kinetic+anchor filters.

Covers ``AIFDetection._apply_postscore_aif_filters``: the
tissue-TTP causality anchor, the 1-TR peak-spread filter, and the
tail/peak threshold.

Cascade-restructure tests (accumulate-to-quorum) live in
``test_dsc_aif_cascade.py`` since they exercise the full ``_run_interface``
flow; here we test the pure filter method in isolation.
"""

from __future__ import annotations

import numpy as np

from pmriprep.interfaces.dsc_aif import AIFDetection
from pmriprep.utils.dsc_constants import (
    AIF_PEAK_SPREAD_MAX_FRAMES,
    AIF_PEAK_TIME_PRE_TISSUE_FRAMES_MIN,
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
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            [], tissue_ttp_frame=50,
        )
        assert survivors == []
        assert diag["aif_n_after_filters"] == 0
        assert diag["aif_n_filtered_ttp_anchor"] == 0
        assert diag["aif_n_filtered_peakspread"] == 0
        assert diag["aif_n_filtered_tail"] == 0
        assert diag["aif_peak_time_spread_frames"] == 0


class TestTissueTTPAnchor:
    """Causality anchor: peak_t < tissue_ttp - 1 (strict)."""

    def test_voxel_peaking_at_tissue_ttp_rejected(self):
        # tissue_ttp=50, anchor threshold = 50 - 1 = 49 (strict <)
        # peak_t=50 fails (50 < 49 is False)
        scored = [_vox((1, 1, 1), peak_time_frame=50, tail_fraction_late=0.05)]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        assert survivors == []
        assert diag["aif_n_filtered_ttp_anchor"] == 1

    def test_voxel_peaking_at_tissue_ttp_minus_1_rejected(self):
        # peak_t=49 fails (49 < 49 is False — strict, not ≤)
        scored = [_vox((1, 1, 1), peak_time_frame=49, tail_fraction_late=0.05)]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        assert survivors == []
        assert diag["aif_n_filtered_ttp_anchor"] == 1

    def test_voxel_peaking_at_tissue_ttp_minus_2_kept(self):
        # peak_t=48 passes (48 < 49 is True)
        scored = [_vox((1, 1, 1), peak_time_frame=48, tail_fraction_late=0.05)]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        assert len(survivors) == 1
        assert diag["aif_n_filtered_ttp_anchor"] == 0

    def test_nan_peak_time_rejected(self):
        scored = [
            (1.0, (1, 1, 1), {
                "peak_time_frame": float("nan"),
                "tail_fraction_late": 0.05,
            }),
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        assert survivors == []
        assert diag["aif_n_filtered_ttp_anchor"] == 1


class TestPeakSpreadFilter:
    """1-TR peak-spread filter (after TTP anchor)."""

    def test_peak_spread_at_1_frame_kept(self):
        # tissue_ttp=50; all anchor-pass; earliest=25; window=[25, 26]
        scored = [
            _vox((0, 0, 0), 25, 0.05),
            _vox((1, 0, 0), 25, 0.05),
            _vox((2, 0, 0), 26, 0.05),
            _vox((3, 0, 0), 27, 0.05),  # beyond +1, rejected
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        assert len(survivors) == 3
        assert diag["aif_n_filtered_peakspread"] == 1
        assert diag["aif_peak_time_spread_frames"] == 1

    def test_earliest_reference_fixed_for_pass(self):
        """The earliest_pt reference is the post-anchor pool's minimum,
        NOT recomputed after peak-spread prunes. So if the earliest voxel
        itself fails the tail filter (downstream), other voxels are still
        evaluated against the original earliest_pt.
        """
        # earliest=25 (anchored, kept), but tail=0.20 → fails tail
        # 26 voxel: passes peak-spread (within +1) AND tail
        # Without fixed earliest, removing the 25 voxel before peak-spread
        # would shift earliest_pt to 26 and accept later voxels
        scored = [
            _vox((0, 0, 0), 25, 0.20),  # anchor pass; spread pass; tail fail
            _vox((1, 0, 0), 26, 0.05),  # anchor pass; spread pass; tail pass
            _vox((2, 0, 0), 28, 0.05),  # anchor pass; spread FAIL (28 > 25+1)
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        # Only voxel at 26 survives: 25 fails tail, 28 fails peak-spread
        # The 28 voxel is rejected because earliest_pt remains 25 (not 26
        # after the 25 voxel is dropped by tail filter — which runs AFTER
        # peak-spread).
        assert len(survivors) == 1
        assert survivors[0][1] == (1, 0, 0)
        assert diag["aif_n_filtered_peakspread"] == 1
        assert diag["aif_n_filtered_tail"] == 1


class TestTailFilter:
    """tail_fraction_late ≤ 0.15 threshold."""

    def test_tail_at_threshold_kept(self):
        # Use the constant directly so the test doesn't drift if 0.15 changes
        thr = AIF_TAIL_FRACTION_MAX
        scored = [
            _vox((0, 0, 0), 25, thr),         # exactly at threshold (kept)
            _vox((1, 0, 0), 25, thr - 0.01),  # below
            _vox((2, 0, 0), 25, thr + 0.01),  # above (rejected)
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
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
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        assert survivors == []
        assert diag["aif_n_filtered_tail"] == 1


class TestThreeFiltersCombined:
    def test_mixed_pool(self):
        # tissue_ttp=50; anchor threshold = 49 (strict <)
        scored = [
            _vox((0, 0, 0), 25, 0.05),  # all pass
            _vox((1, 0, 0), 49, 0.05),  # anchor FAIL (49 < 49 is False)
            _vox((2, 0, 0), 26, 0.20),  # spread pass; tail FAIL
            _vox((3, 0, 0), 28, 0.05),  # spread FAIL (28 > 25+1)
            _vox((4, 0, 0), 25, 0.10),  # all pass
            _vox((5, 0, 0), 26, 0.05),  # all pass
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        assert len(survivors) == 3
        survivor_coords = {c for _, c, _ in survivors}
        assert survivor_coords == {(0, 0, 0), (4, 0, 0), (5, 0, 0)}
        assert diag["aif_n_after_filters"] == 3
        assert diag["aif_n_filtered_ttp_anchor"] == 1
        assert diag["aif_n_filtered_peakspread"] == 1
        assert diag["aif_n_filtered_tail"] == 1
        # max(25, 25, 26) - min(25, 25, 26) = 1
        assert diag["aif_peak_time_spread_frames"] == 1

    def test_all_rejected_returns_empty_with_diagnostics(self):
        scored = [
            _vox((0, 0, 0), 50, 0.05),  # anchor fail
            _vox((1, 0, 0), 51, 0.05),  # anchor fail
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(
            scored, tissue_ttp_frame=50,
        )
        assert survivors == []
        assert diag["aif_n_filtered_ttp_anchor"] == 2
        assert diag["aif_n_after_filters"] == 0


class TestConstantSanity:
    """Sanity-check that the constants are wired into the filter."""

    def test_anchor_uses_constant(self):
        assert AIF_PEAK_TIME_PRE_TISSUE_FRAMES_MIN >= 1

    def test_spread_uses_constant(self):
        assert AIF_PEAK_SPREAD_MAX_FRAMES >= 1

    def test_tail_uses_constant(self):
        assert 0.0 < AIF_TAIL_FRACTION_MAX <= 0.5
