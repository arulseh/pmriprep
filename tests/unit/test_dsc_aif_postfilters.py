"""Unit tests for DSC AIF post-score kinetic-coherence filters.

Covers ``AIFDetection._apply_postscore_aif_filters``: the TR-scaled
peak-spread filter (20th-percentile reference) and the tail/peak
threshold. TTP causality lives in ``_score_candidate_cost`` (upstream),
not here, so it is exercised by the cascade-level tests in
``test_dsc_aif_cascade.py``.
"""

from __future__ import annotations

import numpy as np

from pmriprep.interfaces.dsc_aif import AIFDetection
from pmriprep.utils.dsc_constants import (
    AIF_PEAK_SPREAD_MAX_SECONDS,
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
        survivors, diag = AIFDetection._apply_postscore_aif_filters([], tr=1.5)
        assert survivors == []
        assert diag["aif_n_after_filters"] == 0
        assert diag["aif_n_filtered_peakspread"] == 0
        assert diag["aif_n_filtered_tail"] == 0
        assert diag["aif_peak_time_spread_frames"] == 0


class TestPeakSpreadFilter:
    """TR-scaled peak-spread filter with 20th-percentile reference."""

    def test_peak_spread_within_window_kept(self):
        # At TR=1.5s, max_frame_spread = ceil(3.0/1.5) = 2.
        # Pool [25, 25, 26, 27]: p20 = 25.0, spread_max = 27. All 4 pass.
        scored = [
            _vox((0, 0, 0), 25, 0.05),
            _vox((1, 0, 0), 25, 0.05),
            _vox((2, 0, 0), 26, 0.05),
            _vox((3, 0, 0), 27, 0.05),
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        assert len(survivors) == 4
        assert diag["aif_n_filtered_peakspread"] == 0
        assert diag["aif_peak_time_spread_frames"] == 2

    def test_peak_spread_outside_window_rejected(self):
        # Pool [25, 25, 26, 30]: p20 = 25.0, spread_max = 27. Voxel 30 rejected.
        scored = [
            _vox((0, 0, 0), 25, 0.05),
            _vox((1, 0, 0), 25, 0.05),
            _vox((2, 0, 0), 26, 0.05),
            _vox((3, 0, 0), 30, 0.05),
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        assert len(survivors) == 3
        assert diag["aif_n_filtered_peakspread"] == 1

    def test_earliest_reference_fixed_for_pass(self):
        """``earliest_pt`` is computed once from the input pool's
        20th-percentile and is NOT recomputed after the peak-spread or
        tail filters prune the pool. So if the early voxel itself fails
        the tail filter (downstream), other voxels are still evaluated
        against the original ``earliest_pt`` reference.
        """
        # Pool [25, 26, 28]: p20 = 25.4, spread_max = 27.4 at TR=1.5s.
        # Voxel 28 fails spread regardless of whether voxel 25 is later
        # removed by the tail filter.
        scored = [
            _vox((0, 0, 0), 25, 0.20),  # spread pass; tail fail
            _vox((1, 0, 0), 26, 0.05),  # spread pass; tail pass
            _vox((2, 0, 0), 28, 0.05),  # spread FAIL (28 > 27.4)
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        assert len(survivors) == 1
        assert survivors[0][1] == (1, 0, 0)
        assert diag["aif_n_filtered_peakspread"] == 1
        assert diag["aif_n_filtered_tail"] == 1


class TestTailFilter:
    """tail_fraction_late ≤ AIF_TAIL_FRACTION_MAX threshold."""

    def test_tail_at_threshold_kept(self):
        thr = AIF_TAIL_FRACTION_MAX
        scored = [
            _vox((0, 0, 0), 25, thr),         # exactly at threshold (kept)
            _vox((1, 0, 0), 25, thr - 0.01),  # below
            _vox((2, 0, 0), 25, thr + 0.01),  # above (rejected)
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
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
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        assert survivors == []
        assert diag["aif_n_filtered_tail"] == 1


class TestTwoFiltersCombined:
    def test_mixed_pool(self):
        # Pool [25, 49, 26, 28, 25, 26]: p20 = 25.0, spread_max = 27 at
        # TR=1.5s. Voxels 49 and 28 fail spread; voxel at 26 with tail
        # 0.20 fails tail. 3 survivors.
        scored = [
            _vox((0, 0, 0), 25, 0.05),  # all pass
            _vox((1, 0, 0), 49, 0.05),  # spread FAIL
            _vox((2, 0, 0), 26, 0.20),  # spread pass; tail FAIL
            _vox((3, 0, 0), 28, 0.05),  # spread FAIL
            _vox((4, 0, 0), 25, 0.10),  # all pass
            _vox((5, 0, 0), 26, 0.05),  # all pass
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
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
        # Pool [25, 30]: p20 = 26.0, spread_max = 28. Voxel 30 fails.
        scored = [
            _vox((0, 0, 0), 25, 0.50),   # spread pass, tail FAIL
            _vox((1, 0, 0), 30, 0.05),   # spread FAIL (30 > 28)
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        assert survivors == []
        assert diag["aif_n_filtered_peakspread"] == 1
        assert diag["aif_n_filtered_tail"] == 1
        assert diag["aif_n_after_filters"] == 0


class TestTRScaling:
    """The peak-spread limit scales with TR via
    ``ceil(AIF_PEAK_SPREAD_MAX_SECONDS / tr)``.
    """

    def test_max_frame_spread_at_each_tr(self):
        # Pool spans 6 frames; expect different survivor counts at
        # different TRs.
        scored = [
            _vox((0, 0, 0), 40, 0.05),
            _vox((1, 0, 0), 41, 0.05),
            _vox((2, 0, 0), 42, 0.05),
            _vox((3, 0, 0), 43, 0.05),
            _vox((4, 0, 0), 44, 0.05),
            _vox((5, 0, 0), 45, 0.05),
            _vox((6, 0, 0), 46, 0.05),
        ]
        # At TR=0.5s: max_frame_spread = ceil(3.0/0.5) = 6 → all pass.
        s, _ = AIFDetection._apply_postscore_aif_filters(scored, tr=0.5)
        assert len(s) == 7
        # At TR=1.0s: max_frame_spread = 3. p20 of [40..46] = 41.2,
        # spread_max ≈ 44.2 → voxels at 45, 46 rejected.
        s, _ = AIFDetection._apply_postscore_aif_filters(scored, tr=1.0)
        assert len(s) == 5
        # At TR=1.5s: max_frame_spread = 2. p20 = 41.2, spread_max ≈ 43.2
        # → voxels at 44, 45, 46 rejected.
        s, _ = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        assert len(s) == 4
        # At TR=2.0s: max_frame_spread = 2 (same as 1.5s in this case).
        s, _ = AIFDetection._apply_postscore_aif_filters(scored, tr=2.0)
        assert len(s) == 4

    def test_minimum_spread_is_one_frame(self):
        # Even at very long TR the gate retains a 1-frame minimum so
        # that two voxels in adjacent frames can coexist in a pool.
        scored = [
            _vox((0, 0, 0), 25, 0.05),
            _vox((1, 0, 0), 26, 0.05),
        ]
        s, _ = AIFDetection._apply_postscore_aif_filters(scored, tr=10.0)
        assert len(s) == 2


class TestOutlierRobustness:
    """The 20th-percentile reference is robust to a single early-peaking
    outlier voxel that ``min`` would otherwise anchor the gate to.
    """

    def test_arterial_pool_survives_one_early_outlier(self):
        # 3 co-arterial voxels at frames {46, 46, 47} plus 1 noise voxel
        # at frame 44. With min(pool)=44 the spread_max would be 46 at
        # TR=1.5s, killing voxel 47. With p20 = 45.2, spread_max = 47.2,
        # so all 3 arterial voxels survive.
        scored = [
            _vox((0, 0, 0), 44, 0.05),  # noise (early)
            _vox((1, 0, 0), 46, 0.05),  # arterial
            _vox((2, 0, 0), 46, 0.05),  # arterial
            _vox((3, 0, 0), 47, 0.05),  # arterial
        ]
        survivors, _ = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        # All arterial voxels (1, 2, 3) must be in the survivor set.
        survivor_coords = {c for _, c, _ in survivors}
        assert (1, 0, 0) in survivor_coords
        assert (2, 0, 0) in survivor_coords
        assert (3, 0, 0) in survivor_coords


class TestLateTissueRejection:
    """The filter must still reject obviously-late tissue/capillary
    voxels (peak ≥ 10 frames after the arterial pool) at every TR.
    """

    def test_late_tissue_voxel_rejected_across_tr(self):
        for tr in (0.5, 1.0, 1.5, 2.0):
            scored = [
                _vox((0, 0, 0), 40, 0.05),
                _vox((1, 0, 0), 41, 0.05),
                _vox((2, 0, 0), 41, 0.05),
                _vox((3, 0, 0), 42, 0.05),
                _vox((9, 0, 0), 60, 0.05),  # tissue: 19 frames late
            ]
            survivors, _ = AIFDetection._apply_postscore_aif_filters(scored, tr=tr)
            survivor_coords = {c for _, c, _ in survivors}
            assert (9, 0, 0) not in survivor_coords, (
                f"Late tissue voxel at frame 60 incorrectly retained at TR={tr}"
            )


class TestAllNonFinitePeakTimes:
    """Defensive: if every voxel has non-finite peak_time_frame, the
    filter must return an empty pool with zero diagnostics rather than
    propagating NaN through ``np.percentile``.
    """

    def test_all_inf_peak_times_empty_result(self):
        scored = [
            (1.0, (0, 0, 0), {
                "peak_time_frame": float("inf"),
                "tail_fraction_late": 0.05,
            }),
            (1.0, (1, 0, 0), {
                "peak_time_frame": float("inf"),
                "tail_fraction_late": 0.05,
            }),
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        assert survivors == []
        assert diag["aif_n_after_filters"] == 0

    def test_all_nan_peak_times_empty_result(self):
        scored = [
            (1.0, (0, 0, 0), {
                "peak_time_frame": float("nan"),
                "tail_fraction_late": 0.05,
            }),
        ]
        survivors, diag = AIFDetection._apply_postscore_aif_filters(scored, tr=1.5)
        assert survivors == []
        assert diag["aif_n_after_filters"] == 0


class TestConstantSanity:
    """Sanity-check the kinetic-filter constants are still wired in."""

    def test_spread_uses_constant(self):
        assert 0 < AIF_PEAK_SPREAD_MAX_SECONDS <= 10

    def test_tail_uses_constant(self):
        assert 0.0 < AIF_TAIL_FRACTION_MAX <= 0.5
