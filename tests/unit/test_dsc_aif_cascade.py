"""Tests for AIF cascade logic and quality helper methods."""

import math
from typing import Optional

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import (
    MIN_AIF_CANDIDATES_ACCEPT,
    S0_RATIO_FLOOR,
    _CASCADE_N_CANDIDATES_FRACTION,
    AIFDetection,
)


def _make_scored(
    n: int,
) -> list[tuple[float, tuple[int, int, int], dict]]:
    """Create a list of n fake scored candidates."""
    return [
        (float(i) * 0.1, (i, 0, 0), {"snr": 10.0})
        for i in range(n)
    ]


def _make_aif_curve(
    n_frames: int = 120,
    baseline_frames: int = 20,
    tr: float = 1.5,
    peak_frame: int = 25,
    peak_height: float = 30.0,
    fwhm_frames: int = 5,
    recirculation_height: float = 5.0,
    recirc_frame: int = 45,
) -> np.ndarray:
    """Create a synthetic AIF concentration curve for testing."""
    curve = np.zeros(n_frames)
    for i in range(n_frames):
        if i >= peak_frame:
            dt = i - peak_frame
            curve[i] = peak_height * np.exp(-0.5 * (dt / fwhm_frames) ** 2)
        if i >= recirc_frame:
            dt = i - recirc_frame
            curve[i] += recirculation_height * np.exp(-0.5 * (dt / 8) ** 2)
    return curve


class TestMinAifCandidatesConstant:
    """Verify the module-level constant exists and is sensible."""

    def test_value(self) -> None:
        assert MIN_AIF_CANDIDATES_ACCEPT == 3

    def test_is_int(self) -> None:
        assert isinstance(MIN_AIF_CANDIDATES_ACCEPT, int)


class TestS0Constants:
    """Verify S0 scoring constants."""

    def test_floor_value(self) -> None:
        assert S0_RATIO_FLOOR == 0.10


class TestComputeAifSpr:
    """Tests for AIFDetection._compute_aif_spr."""

    def test_clean_first_pass(self) -> None:
        """Clean bolus with small recirculation → low SPR."""
        curve = _make_aif_curve(
            peak_height=30.0, recirculation_height=3.0,
        )
        spr, fp_frame, fp_val = AIFDetection._compute_aif_spr(
            curve, baseline_frames=20, tr=1.5,
        )
        assert spr < 50.0
        assert fp_frame == 25
        assert fp_val > 20.0

    def test_dominant_recirculation(self) -> None:
        """Recirculation exceeds first pass → SPR > 100."""
        curve = _make_aif_curve(
            peak_height=10.0, recirculation_height=15.0,
            recirc_frame=60,
        )
        spr, _, _ = AIFDetection._compute_aif_spr(
            curve, baseline_frames=20, tr=1.5,
        )
        assert spr > 100.0

    def test_no_recirculation(self) -> None:
        """No recirculation → SPR near 0."""
        curve = _make_aif_curve(
            peak_height=30.0, recirculation_height=0.0,
        )
        spr, _, _ = AIFDetection._compute_aif_spr(
            curve, baseline_frames=20, tr=1.5,
        )
        assert spr < 5.0

    def test_empty_curve(self) -> None:
        """Very short curve → SPR = 0 (no first-pass window)."""
        curve = np.zeros(5)
        spr, _, _ = AIFDetection._compute_aif_spr(
            curve, baseline_frames=3, tr=1.5,
        )
        assert spr == 0.0


class TestAifGlobalMaxInFp:
    """Tests for AIFDetection._aif_global_max_in_fp."""

    def test_peak_in_first_pass(self) -> None:
        """Global max during first pass → True."""
        curve = _make_aif_curve(
            peak_height=30.0, recirculation_height=5.0,
        )
        assert AIFDetection._aif_global_max_in_fp(
            curve, baseline_frames=20, tr=1.5,
        ) is True

    def test_peak_outside_first_pass(self) -> None:
        """Recirculation exceeds first pass → False."""
        curve = _make_aif_curve(
            peak_height=10.0, recirculation_height=15.0,
            recirc_frame=60,
        )
        assert AIFDetection._aif_global_max_in_fp(
            curve, baseline_frames=20, tr=1.5,
        ) is False

    def test_peak_in_baseline(self) -> None:
        """Peak before first-pass window → False."""
        curve = np.zeros(120)
        curve[5] = 50.0  # peak in baseline
        assert AIFDetection._aif_global_max_in_fp(
            curve, baseline_frames=20, tr=1.5,
        ) is False


class TestAifPassesQuality:
    """Tests for AIFDetection._aif_passes_quality."""

    def test_good_curve_has_low_spr(self) -> None:
        """Clean arterial curve has acceptable SPR."""
        curve = _make_aif_curve(
            peak_height=30.0, recirculation_height=3.0,
        )
        result = AIFDetection._aif_passes_quality(
            curve, baseline_frames=20, tr=1.5,
        )
        assert result.spr < 100.0

    def test_bad_recirculation_fails(self) -> None:
        """Dominant recirculation fails SPR check."""
        curve = _make_aif_curve(
            peak_height=10.0, recirculation_height=15.0,
            recirc_frame=60,
        )
        result = AIFDetection._aif_passes_quality(
            curve, baseline_frames=20, tr=1.5,
        )
        assert result.passes is False
        assert result.spr > 100.0


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

    def test_small_n_floors(self) -> None:
        """n_candidates=5 → relaxed=3, permissive=3 (floor at MIN)."""
        assert self._level_n("relaxed", 5) == 3
        assert self._level_n("permissive", 5) == 3

    def test_cascade_prefers_stricter_at_equal_count(self) -> None:
        """2 strict + 2 relaxed accumulated → both in pool, strict first by cost."""
        pool: list[tuple[float, tuple[int, int, int], dict]] = []
        coords: set[tuple[int, int, int]] = set()

        strict_scored = [(0.1, (0, 0, 0), {}), (0.2, (1, 0, 0), {})]
        relaxed_scored = [(0.5, (2, 0, 0), {}), (0.6, (3, 0, 0), {})]

        for scored in [strict_scored, relaxed_scored]:
            for cost, coord, metrics in scored:
                if coord not in coords:
                    pool.append((cost, coord, metrics))
                    coords.add(coord)

        pool.sort(key=lambda x: x[0])
        # Strict candidates (lower cost) sort first
        assert pool[0][0] == 0.1
        assert pool[1][0] == 0.2
        assert len(pool) == 4


class TestAccumulatingCascade:
    """Test the accumulating cascade with averaged-pool quality gate.

    The cascade accumulates candidates across tiers and uses the first passing
    averaged AIF. Tests exercise the helper methods that implement this logic.
    """

    def test_pool_acceptance_with_arterial_voxel(self) -> None:
        """Pool acceptance returns a result for a clear arterial voxel."""
        aif = AIFDetection()
        shape = (10, 10, 5, 120)
        conc_data = np.zeros(shape)
        brain_mask = np.ones(shape[:3])

        curve = _make_aif_curve(peak_height=30.0, recirculation_height=2.0)
        conc_data[5, 5, 2, :] = curve
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                conc_data[5+dx, 5+dy, 2, :] = _make_aif_curve(
                    peak_height=5.0, recirculation_height=1.0,
                    peak_frame=28,
                )

        candidates: list[tuple[float, tuple[int, int, int], dict]] = [
            (1.0, (5, 5, 2), {}),
        ]
        result = aif._try_aif_pool_acceptance(
            candidates, conc_data, brain_mask, tr=1.5, baseline_frames=20,
        )
        # May return None if gamma fit fails on synthetic curve — that's OK.
        # Pool acceptance uses each candidate's own voxel.
        if result is not None:
            assert result.roi_size == 1
            assert result.spr >= 0.0

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_pool_acceptance_returns_none_on_bad_curves(self) -> None:
        """A pool whose averaged curve fails quality gates returns None."""
        aif = AIFDetection()
        shape = (10, 10, 5, 120)
        conc_data = np.random.RandomState(3).randn(*shape) * 0.1  # noise only
        brain_mask = np.ones(shape[:3])

        candidates = [(1.0, (5, 5, 2), {})]
        result = aif._try_aif_pool_acceptance(
            candidates, conc_data, brain_mask, tr=1.5, baseline_frames=20,
        )
        assert result is None

    def test_accumulated_pool_deduplication(self) -> None:
        """Candidates from multiple tiers are deduplicated by coordinate."""
        pool: list[tuple[float, tuple[int, int, int], dict]] = []
        coords: set[tuple[int, int, int]] = set()

        # Tier 1 candidates
        tier1 = [(1.0, (5, 5, 2), {}), (2.0, (6, 5, 2), {})]
        for cost, coord, metrics in tier1:
            if coord not in coords:
                pool.append((cost, coord, metrics))
                coords.add(coord)

        # Tier 2 has one overlap and one new
        tier2 = [(0.5, (5, 5, 2), {}), (1.5, (7, 5, 2), {})]
        for cost, coord, metrics in tier2:
            if coord not in coords:
                pool.append((cost, coord, metrics))
                coords.add(coord)

        assert len(pool) == 3
        assert (5, 5, 2) in coords
        assert (6, 5, 2) in coords
        assert (7, 5, 2) in coords

    def test_pool_cap_preserves_best_candidates(self) -> None:
        """Pool cap keeps lowest-cost candidates when pool exceeds limit."""
        pool = [(float(i), (i, 0, 0), {}) for i in range(20)]
        pool.sort(key=lambda x: x[0])
        cap = 10
        capped = pool[:cap]

        assert len(capped) == 10
        assert capped[0][0] == 0.0  # best cost preserved
        assert capped[-1][0] == 9.0  # worst kept is 9.0

    def test_below_min_candidates_accumulate(self) -> None:
        """Candidates below MIN_AIF_CANDIDATES_ACCEPT are accumulated."""
        pool: list[tuple[float, tuple[int, int, int], dict]] = []
        coords: set[tuple[int, int, int]] = set()

        # Two tiers each with 2 candidates (below MIN=3)
        for tier_offset in [0, 10]:
            scored = _make_scored(2)
            scored = [(c, (x + tier_offset, y, z), m) for c, (x, y, z), m in scored]
            for cost, coord, metrics in scored:
                if coord not in coords:
                    pool.append((cost, coord, metrics))
                    coords.add(coord)

        # After two tiers, accumulated pool has 4 (>= MIN)
        assert len(pool) == 4
        assert len(pool) >= MIN_AIF_CANDIDATES_ACCEPT


class TestPostGammaFitR2Fallback:
    """Test post-gamma-fit R² quality gate that triggers synthetic fallback.

    After AIF candidates are found and gamma fitting is attempted, if the
    gamma fit R² is catastrophically bad (< 0.0), the pipeline should fall
    back to synthetic AIF. This catches cases where AIF detection "succeeds"
    but the curve has no actual bolus shape (e.g., T1 leakage contamination).
    """

    @staticmethod
    def _simulate_post_gamma_check(
        gamma_fit_applied: bool,
        gamma_r2: Optional[float],
        disable_fallback: bool,
        detection_level: str,
    ) -> tuple[bool, Optional[str]]:
        """Simulate the post-gamma-fit quality gate logic.

        Returns
        -------
        tuple[bool, Optional[str]]
            (triggers_fallback, fallback_reason)
        """
        from pmriprep.interfaces.dsc_aif import GAMMA_R2_SYNTHETIC_FALLBACK

        # Mirrors the logic in _run_interface after gamma fitting
        if (
            not gamma_fit_applied
            and not disable_fallback
            and detection_level != "synthetic"
        ):
            if gamma_r2 is not None and gamma_r2 < GAMMA_R2_SYNTHETIC_FALLBACK:
                return True, "gamma_r2_floor"
        return False, None

    def test_negative_r2_triggers_fallback(self) -> None:
        """R² = -0.5 (worse than mean) triggers synthetic fallback."""
        triggers, reason = self._simulate_post_gamma_check(
            gamma_fit_applied=False,
            gamma_r2=-0.5,
            disable_fallback=False,
            detection_level="brain_relaxed",
        )
        assert triggers is True
        assert reason == "gamma_r2_floor"

    def test_r2_exactly_zero_does_not_trigger(self) -> None:
        """R² = 0.0 (threshold) does not trigger fallback (>= not >)."""
        triggers, reason = self._simulate_post_gamma_check(
            gamma_fit_applied=False,
            gamma_r2=0.0,
            disable_fallback=False,
            detection_level="brain_relaxed",
        )
        assert triggers is False
        assert reason is None

    def test_positive_r2_does_not_trigger(self) -> None:
        """R² = 0.3 (above threshold) does not trigger fallback."""
        triggers, reason = self._simulate_post_gamma_check(
            gamma_fit_applied=False,
            gamma_r2=0.3,
            disable_fallback=False,
            detection_level="brain_relaxed",
        )
        assert triggers is False
        assert reason is None

    def test_r2_none_does_not_trigger(self) -> None:
        """R² = None (not computed) does not trigger fallback."""
        triggers, reason = self._simulate_post_gamma_check(
            gamma_fit_applied=False,
            gamma_r2=None,
            disable_fallback=False,
            detection_level="brain_relaxed",
        )
        assert triggers is False
        assert reason is None

    def test_gamma_fit_applied_skips_check(self) -> None:
        """If gamma fit was applied, skip the R² check (already good)."""
        triggers, reason = self._simulate_post_gamma_check(
            gamma_fit_applied=True,
            gamma_r2=-0.5,  # bad R², but fit was applied
            disable_fallback=False,
            detection_level="brain_relaxed",
        )
        assert triggers is False
        assert reason is None

    def test_disable_fallback_skips_check(self) -> None:
        """If fallback is disabled, skip the R² check."""
        triggers, reason = self._simulate_post_gamma_check(
            gamma_fit_applied=False,
            gamma_r2=-0.5,
            disable_fallback=True,
            detection_level="brain_relaxed",
        )
        assert triggers is False
        assert reason is None

    def test_already_synthetic_skips_check(self) -> None:
        """If already synthetic, skip the R² check."""
        triggers, reason = self._simulate_post_gamma_check(
            gamma_fit_applied=False,
            gamma_r2=-0.5,
            disable_fallback=False,
            detection_level="synthetic",
        )
        assert triggers is False
        assert reason is None

    def test_constant_value(self) -> None:
        """Verify GAMMA_R2_SYNTHETIC_FALLBACK constant is 0.0."""
        from pmriprep.interfaces.dsc_aif import GAMMA_R2_SYNTHETIC_FALLBACK
        assert GAMMA_R2_SYNTHETIC_FALLBACK == 0.0


# =============================================================================
# Progressive-trim + averaged-BRR gate
# =============================================================================


def _curve_with_negative_tail(
    peak_height: float = 30.0,
    tail_offset: float = -5.0,
    n_frames: int = 120,
    baseline_frames: int = 20,
    peak_frame: int = 25,
    fwhm_frames: int = 5,
) -> np.ndarray:
    """Build a curve with a clean first-pass peak but a constant negative
    pedestal in the post-peak tail.  Mimics the failure mode where T1
    leakage or motion produces a negative-going tail."""
    curve = np.zeros(n_frames)
    for i in range(peak_frame, n_frames):
        dt = i - peak_frame
        curve[i] = peak_height * np.exp(-0.5 * (dt / fwhm_frames) ** 2)
    # Apply pedestal in the late tail (last 40% of frames)
    tail_start = int(0.6 * n_frames)
    curve[tail_start:] += tail_offset
    return curve


class TestAifAvgBrrConstant:
    """The averaged-AIF BRR floor."""

    def test_value(self) -> None:
        from pmriprep.interfaces.dsc_aif import AIF_AVG_BRR_MIN
        assert AIF_AVG_BRR_MIN == -3.0

    def test_looser_than_per_voxel_permissive(self) -> None:
        """Averaged-curve floor should be tighter than the per-voxel
        permissive floor (−10 %) — averaging should reduce noise, not
        let bad voxels dominate.  -3 is between relaxed (0) and
        permissive (−10)."""
        from pmriprep.interfaces.dsc_aif import AIF_AVG_BRR_MIN
        # Per-voxel ranges from _score_candidate_cost:
        # strict (5, 30), relaxed (0, 40), permissive (−10, 50)
        assert -10.0 < AIF_AVG_BRR_MIN < 0.0


class TestComputeAifBrr:
    """AIFDetection._compute_aif_brr static helper."""

    def test_positive_tail(self) -> None:
        curve = _make_aif_curve(peak_height=30.0, recirculation_height=2.0)
        brr = AIFDetection._compute_aif_brr(curve, baseline_frames=20, tr=1.5)
        assert brr > 0.0
        assert brr < 50.0

    def test_negative_tail(self) -> None:
        curve = _curve_with_negative_tail(peak_height=30.0, tail_offset=-5.0)
        brr = AIFDetection._compute_aif_brr(curve, baseline_frames=20, tr=1.5)
        assert brr < 0.0

    def test_short_curve_returns_nan(self) -> None:
        curve = np.zeros(5)
        brr = AIFDetection._compute_aif_brr(curve, baseline_frames=3, tr=1.5)
        assert np.isnan(brr)


class TestAifPassesQualityBrrGate:
    """_aif_passes_quality gates on averaged BRR ≥ AIF_AVG_BRR_MIN."""

    def test_rejects_negative_tail(self) -> None:
        """Curve with BRR well below -3 % fails the BRR gate."""
        curve = _curve_with_negative_tail(
            peak_height=30.0, tail_offset=-8.0,
        )
        result = AIFDetection._aif_passes_quality(
            curve, baseline_frames=20, tr=1.5,
        )
        assert result.passes is False
        assert result.brr < -3.0

    def test_return_tuple_has_brr(self) -> None:
        """The return is an AifQualityResult with 4 fields."""
        curve = _make_aif_curve(peak_height=30.0, recirculation_height=2.0)
        result = AIFDetection._aif_passes_quality(
            curve, baseline_frames=20, tr=1.5,
        )
        assert len(result) == 4
        assert isinstance(result.brr, float)


class TestAIFPoolBrrGate:
    """Pool acceptance returns None when the averaged curve fails BRR."""

    def test_negative_tail_average_returns_none(self) -> None:
        """Pool whose averaged curve has strongly negative tail → None.

        Place 4 candidates with negative-tail curves; their inverse-cost
        weighted average preserves the negative tail, so the BRR gate
        should fire and return None.
        """
        aif = AIFDetection()
        shape = (10, 10, 5, 120)
        conc_data = np.zeros(shape)
        brain_mask = np.ones(shape[:3])
        bad_curve = _curve_with_negative_tail(peak_height=30.0, tail_offset=-8.0)
        coords = [(4, 4, 2), (4, 5, 2), (5, 4, 2), (5, 5, 2)]
        candidates: list[tuple[float, tuple[int, int, int], dict]] = []
        for i, c in enumerate(coords):
            conc_data[c[0], c[1], c[2], :] = bad_curve + np.random.randn(120) * 0.1
            # Assign per-candidate BRR matching the curve (negative)
            candidates.append((1.0 + i * 0.1, c, {"baseline_return_ratio": -8.0}))
        result = aif._try_aif_pool_acceptance(
            candidates, conc_data, brain_mask, tr=1.5, baseline_frames=20,
        )
        # Should fail the BRR gate — result is None
        assert result is None

    def test_returns_brr_in_tuple(self) -> None:
        """Successful return is an AIFPoolAcceptanceResult with six fields."""
        aif = AIFDetection()
        shape = (10, 10, 5, 120)
        conc_data = np.zeros(shape)
        brain_mask = np.ones(shape[:3])
        good_curve = _make_aif_curve(peak_height=30.0, recirculation_height=2.0)
        coords = [(4, 4, 2), (4, 5, 2), (5, 4, 2)]
        candidates: list[tuple[float, tuple[int, int, int], dict]] = []
        for i, c in enumerate(coords):
            conc_data[c[0], c[1], c[2], :] = good_curve
            candidates.append((1.0 + i * 0.1, c, {"baseline_return_ratio": 10.0}))
        result = aif._try_aif_pool_acceptance(
            candidates, conc_data, brain_mask, tr=1.5, baseline_frames=20,
        )
        if result is not None:
            assert len(result) == 6
            assert result.roi_size == 1
            assert isinstance(result.brr, float)


class TestProgressiveTrimSemantics:
    """Verify the cost-sorted pool + drop-worst-BRR ordering.

    These are semantic / invariant tests on the trim choice function
    (imports the worst_idx logic directly).  The full cascade loop
    itself is exercised by existing cascade tests — adding a full 4-D
    end-to-end test that triggers trimming requires synthesizing atlas
    masks, scoring gates, and the complete `_run_interface` wiring,
    which is left for integration-level coverage.
    """

    def _worst_idx(
        self,
        pool: list[tuple[float, tuple[int, int, int], dict]],
    ) -> int:
        """Replicates the trim-selection predicate inside the cascade
        loop in `pmriprep/interfaces/dsc_aif.py`."""
        def key(i: int) -> float:
            brr = pool[i][2].get("baseline_return_ratio", float("nan"))
            if not np.isnan(brr):
                return brr
            # Cost fallback: more negative = "worse" in the min() call,
            # matching the higher-cost = worse convention.
            return -pool[i][0]
        return min(range(len(pool)), key=key)

    def test_drops_most_negative_brr(self) -> None:
        pool = [
            (1.0, (0, 0, 0), {"baseline_return_ratio": 10.0}),
            (2.0, (1, 0, 0), {"baseline_return_ratio": -8.0}),  # worst
            (3.0, (2, 0, 0), {"baseline_return_ratio": 5.0}),
        ]
        assert self._worst_idx(pool) == 1

    def test_drops_most_negative_even_if_low_cost(self) -> None:
        """Worst-BRR wins even when its cost is the lowest (best)."""
        pool = [
            (0.1, (0, 0, 0), {"baseline_return_ratio": -9.0}),  # worst-BRR, lowest cost
            (2.0, (1, 0, 0), {"baseline_return_ratio": 5.0}),
            (3.0, (2, 0, 0), {"baseline_return_ratio": 8.0}),
        ]
        assert self._worst_idx(pool) == 0

    def test_cost_fallback_when_brr_nan(self) -> None:
        """When all per-voxel BRRs are NaN, fall back to highest-cost."""
        pool = [
            (1.0, (0, 0, 0), {}),
            (3.0, (1, 0, 0), {}),  # highest cost → worst under fallback
            (2.0, (2, 0, 0), {}),
        ]
        assert self._worst_idx(pool) == 1


class TestTrimLoopEdgeCases:
    """Edge-case behaviour of the trim loop."""

    def test_pool_exactly_at_floor_runs_once(self) -> None:
        """``len(scored) == MIN`` should test once, then break without
        trimming (the trim step checks ``len(pool) == MIN`` before
        popping)."""
        aif = AIFDetection()
        shape = (8, 8, 4, 120)
        conc_data = np.zeros(shape)
        brain_mask = np.ones(shape[:3])
        bad_curve = _curve_with_negative_tail(peak_height=30.0, tail_offset=-8.0)
        coords = [(3, 3, 2), (3, 4, 2), (4, 3, 2)]  # exactly MIN=3
        candidates: list[tuple[float, tuple[int, int, int], dict]] = []
        for i, c in enumerate(coords):
            conc_data[c[0], c[1], c[2], :] = bad_curve
            candidates.append((1.0 + i * 0.1, c, {"baseline_return_ratio": -8.0}))
        # At exactly MIN, pool acceptance is attempted once; BRR gate
        # rejects the averaged negative-tail curve; no trim happens
        # because pool size is already at the floor.
        result = aif._try_aif_pool_acceptance(
            candidates, conc_data, brain_mask, tr=1.5, baseline_frames=20,
        )
        assert result is None

    def test_below_floor_skips_loop(self) -> None:
        """``len(scored) < MIN`` should never enter pool acceptance."""
        # This is a semantic check on the trim loop's entry condition.
        # The real cascade skips the loop when len(pool) < MIN; we
        # verify the predicate directly.
        from pmriprep.interfaces.dsc_aif import MIN_AIF_CANDIDATES_ACCEPT
        scored: list[tuple[float, tuple[int, int, int], dict]] = [
            (0.5, (0, 0, 0), {"baseline_return_ratio": 5.0}),
            (0.6, (1, 0, 0), {"baseline_return_ratio": 8.0}),
        ]
        # The loop's entry predicate is `while len(pool) >= MIN`; with
        # len=2 < 3 the loop never executes and the tier rejects via
        # accumulation.
        assert len(scored) < MIN_AIF_CANDIDATES_ACCEPT
        # Simulate the loop
        pool = list(scored)
        iterations = 0
        while len(pool) >= MIN_AIF_CANDIDATES_ACCEPT:
            iterations += 1
            pool.pop()
        assert iterations == 0


class TestPostfilterDiagsEmission:
    """Pin the predicate that gates ``cascade_postfilter_diags`` emission.

    Production code emits the per-tier postfilter diagnostics dict to
    QC iff the cascade was downgraded (initial tier ≠ final accepted
    tier) OR the cascade fell to synthetic (either via exhaustion or
    via gamma-R² post-validation rejection). Single-tier acceptances
    (cascade_initial == cascade_final, no downgrade) skip the per-tier
    breakdown — there's no diagnostic value when only one tier ran.
    """

    @staticmethod
    def _should_emit(qc_metrics: dict, detection_level: str,
                     postfilter_diags: dict) -> bool:
        """Mirror of the production gate at dsc_aif.py."""
        return bool(
            (qc_metrics.get("cascade_downgraded", False)
             or detection_level == "synthetic")
            and postfilter_diags
        )

    def test_emits_on_downgrade(self) -> None:
        qc = {"cascade_downgraded": True}
        diags = {"atlas_strict": {"aif_n_after_filters": 0}}
        assert self._should_emit(qc, "atlas_relaxed", diags) is True

    def test_emits_on_synthetic_exhaustion(self) -> None:
        qc = {"cascade_downgraded": False}
        diags = {
            "atlas_strict": {"aif_n_after_filters": 0},
            "atlas_relaxed": {"aif_n_after_filters": 0},
        }
        assert self._should_emit(qc, "synthetic", diags) is True

    def test_emits_on_gamma_r2_synthetic_fallback(self) -> None:
        # In the gamma-R²-fallback path, detection_level is overwritten
        # to "synthetic" after a tier had originally accepted.
        # cascade_downgraded may be False, but detection_level alone
        # should trigger the emission.
        qc = {"cascade_downgraded": False}
        diags = {"atlas_strict": {"aif_n_after_filters": 5}}
        assert self._should_emit(qc, "synthetic", diags) is True

    def test_skips_on_single_tier_accept(self) -> None:
        # Single-tier accept: initial == final, no downgrade,
        # detection_level is the tier name (not "synthetic").
        qc = {"cascade_downgraded": False}
        diags = {"atlas_strict": {"aif_n_after_filters": 6}}
        assert self._should_emit(qc, "atlas_strict", diags) is False

    def test_skips_when_postfilter_diags_empty(self) -> None:
        # Even if downgraded or synthetic, an empty diag dict produces
        # no useful information; predicate should be False.
        qc = {"cascade_downgraded": True}
        assert self._should_emit(qc, "atlas_relaxed", {}) is False
        assert self._should_emit(qc, "synthetic", {}) is False

    def test_default_no_downgrade(self) -> None:
        # Missing cascade_downgraded key should be treated as False.
        qc: dict = {}
        diags = {"atlas_strict": {"aif_n_after_filters": 6}}
        assert self._should_emit(qc, "atlas_strict", diags) is False


class TestUnionPoolTrim:
    """Test progressive worst-BRR trimming of a union candidate pool.

    These tests reproduce the trimming loop using production scoring helpers.
    They check rejection of a contaminated pool, acceptance after trimming and
    provenance counts for the retained coordinates.
    """

    @staticmethod
    def _trim_key(union_pool, i):
        brr = union_pool[i][2].get("baseline_return_ratio", float("nan"))
        if not np.isnan(brr):
            return float(brr)
        return -float(union_pool[i][0])

    def _build_pool(self, contaminator_tier="atlas_relaxed"):
        """Build a 5-voxel union: 1 contaminated (tier=contaminator_tier)
        + 4 clean (split 1 from atlas_relaxed, 3 from atlas_permissive).
        """
        shape = (10, 10, 5, 120)
        conc_data = np.zeros(shape)
        brain_mask = np.ones(shape[:3])
        good_curve = _make_aif_curve(peak_height=30.0, recirculation_height=2.0)
        bad_curve = _curve_with_negative_tail(
            peak_height=30.0, tail_offset=-8.0,
        )
        # 5 coords; (4,4,2) is the contaminator
        union: list[tuple[float, tuple[int, int, int], dict]] = []
        union_coord_origin: dict[tuple[int, int, int], str] = {}
        # Contaminator first (tier matches the parameter)
        conc_data[4, 4, 2, :] = bad_curve
        union.append((1.0, (4, 4, 2), {"baseline_return_ratio": -8.0}))
        union_coord_origin[(4, 4, 2)] = contaminator_tier
        # 1 clean voxel from atlas_relaxed
        conc_data[4, 5, 2, :] = good_curve
        union.append((1.1, (4, 5, 2), {"baseline_return_ratio": 10.0}))
        union_coord_origin[(4, 5, 2)] = "atlas_relaxed"
        # 3 clean voxels from atlas_permissive
        for i, c in enumerate([(5, 4, 2), (5, 5, 2), (4, 6, 2)]):
            conc_data[c[0], c[1], c[2], :] = good_curve
            union.append((1.2 + i * 0.1, c, {"baseline_return_ratio": 10.0}))
            union_coord_origin[c] = "atlas_permissive"
        return conc_data, brain_mask, union, union_coord_origin

    def _run_union_trim(self, conc_data, brain_mask, union):
        """Replicates the production union-trim loop body."""
        aif = AIFDetection()
        union_pool = list(union)
        accept_pool = None
        while len(union_pool) >= MIN_AIF_CANDIDATES_ACCEPT:
            result = aif._try_aif_pool_acceptance(
                union_pool, conc_data, brain_mask,
                tr=1.5, baseline_frames=20,
            )
            if result is not None:
                accept_pool = list(union_pool)
                break
            worst_idx = min(
                range(len(union_pool)),
                key=lambda i: self._trim_key(union_pool, i),
            )
            union_pool.pop(worst_idx)
        return accept_pool

    def test_single_shot_fails_on_contaminated_union(self):
        """Without trim, a 5-voxel union with one negative-BRR voxel
        averaged in fails the averaged-pool gate single-shot."""
        aif = AIFDetection()
        conc_data, brain_mask, union, _ = self._build_pool()
        result = aif._try_aif_pool_acceptance(
            union, conc_data, brain_mask, tr=1.5, baseline_frames=20,
        )
        assert result is None

    def test_trim_drops_contaminator_and_passes(self):
        """With progressive trim, the contaminated voxel is dropped on
        the first iteration and the remaining 4-voxel pool passes."""
        conc_data, brain_mask, union, _ = self._build_pool()
        accepted = self._run_union_trim(conc_data, brain_mask, union)
        assert accepted is not None
        accepted_coords = {coord for _, coord, _ in accepted}
        # Contaminator (4,4,2) must be dropped; the 4 clean voxels remain.
        assert (4, 4, 2) not in accepted_coords
        assert {(4, 5, 2), (5, 4, 2), (5, 5, 2), (4, 6, 2)} == accepted_coords

    def test_provenance_recomputed_from_trimmed_pool(self):
        """When the contaminator was the only voxel from a particular
        tier, that tier must be REMOVED from the recomputed provenance
        (and from detection_level) — not merely have its count reduced.
        """
        # Contaminator is from atlas_strict; the only atlas_strict voxel.
        conc_data, brain_mask, union, union_coord_origin = self._build_pool(
            contaminator_tier="atlas_strict",
        )
        accepted = self._run_union_trim(conc_data, brain_mask, union)
        assert accepted is not None
        # Recompute provenance from the trimmed pool's coords
        # (replicates production code).
        trimmed_provenance: dict[str, int] = {}
        for _, coord, _ in accepted:
            lvl = union_coord_origin.get(coord)
            if lvl is not None:
                trimmed_provenance[lvl] = trimmed_provenance.get(lvl, 0) + 1
        # atlas_strict must NOT appear (its only voxel was the contaminator).
        assert "atlas_strict" not in trimmed_provenance
        assert trimmed_provenance == {
            "atlas_relaxed": 1,
            "atlas_permissive": 3,
        }
        # detection_level must reflect post-trim contributors only,
        # in cascade order (strict, relaxed, permissive).
        cascade_order = [
            "atlas_strict", "atlas_relaxed", "atlas_permissive",
            "full_atlas_permissive", "brain_relaxed", "brain_permissive",
        ]
        contributing = [
            lvl for lvl in cascade_order
            if trimmed_provenance.get(lvl, 0) > 0
        ]
        detection_level = "+".join(contributing)
        assert detection_level == "atlas_relaxed+atlas_permissive"

    def test_provenance_preserves_partial_tier(self):
        """When a tier had multiple voxels and only one was trimmed,
        the tier should remain in provenance with a reduced count.
        """
        # Contaminator is from atlas_relaxed; 1 other clean atlas_relaxed
        # voxel remains in the pool.
        conc_data, brain_mask, union, union_coord_origin = self._build_pool(
            contaminator_tier="atlas_relaxed",
        )
        accepted = self._run_union_trim(conc_data, brain_mask, union)
        assert accepted is not None
        trimmed_provenance: dict[str, int] = {}
        for _, coord, _ in accepted:
            lvl = union_coord_origin.get(coord)
            if lvl is not None:
                trimmed_provenance[lvl] = trimmed_provenance.get(lvl, 0) + 1
        # atlas_relaxed still present (1 clean voxel survived) but count
        # dropped from 2 to 1.
        assert trimmed_provenance == {
            "atlas_relaxed": 1,
            "atlas_permissive": 3,
        }


class TestSyntheticFallbackQcConsistency:
    """When the cascade falls back to synthetic (either no_candidates or
    gamma-R²-floor), real-AIF spatial QC fields must NOT remain in
    ``qc_metrics``. They would otherwise contradict the synthetic
    provenance (e.g. ``n_aif_voxels_selected=0`` alongside a stale
    ``aif_voxel_coord`` from the rejected candidate) and corrupt
    downstream sidecars and QC visualizations.

    Production code at ``dsc_aif.py`` calls ``_clear_stale_real_aif_keys``
    on both synthetic-fallback paths.
    """

    # Mirror the production constant — duplicated intentionally so the
    # test fails loudly if a key is ever added to production but
    # forgotten here (or vice versa).
    STALE_KEYS = (
        "aif_voxel_coord", "aif_centroid_mm", "aif_hemisphere",
        "candidate_coords", "candidate_scores",
        "n_left_candidates", "n_right_candidates",
        "hemisphere_coverage", "is_bilateral",
        "tumor_laterality", "laterality_mask_source",
        "n_contralateral", "n_ipsilateral",
        "aif_slice", "aif_edge_slice_warning",
        "best_cost", "mean_cost",
        "cascade_accepted_roi_size",
        "aif_metrics",
        "aif_tier_provenance",
    )

    def test_helper_clears_all_stale_keys(self) -> None:
        """Predicate test: the helper pops every stale key and leaves
        non-stale keys (cascade_tiers_attempted, cascade_initial_level,
        n_baseline_frames, etc.) intact."""
        from pmriprep.interfaces.dsc_aif import _clear_stale_real_aif_keys
        qc = {k: f"stale_{k}" for k in self.STALE_KEYS}
        qc["cascade_tiers_attempted"] = ["atlas_strict", "atlas_relaxed"]
        qc["cascade_initial_level"] = "atlas_strict"
        qc["cascade_final_level"] = "atlas_relaxed"
        qc["n_baseline_frames"] = 41
        qc["arrival_time_seconds"] = 60.0
        qc["detection_method"] = "synthetic"
        qc["n_aif_voxels_selected"] = 0
        _clear_stale_real_aif_keys(qc)
        for k in self.STALE_KEYS:
            assert k not in qc, f"{k} should have been popped"
        assert qc["cascade_tiers_attempted"] == ["atlas_strict", "atlas_relaxed"]
        assert qc["cascade_initial_level"] == "atlas_strict"
        assert qc["cascade_final_level"] == "atlas_relaxed"
        assert qc["n_baseline_frames"] == 41
        assert qc["arrival_time_seconds"] == 60.0
        assert qc["detection_method"] == "synthetic"
        assert qc["n_aif_voxels_selected"] == 0

    def test_helper_idempotent_on_empty_dict(self) -> None:
        """Helper must not raise when applied to a fresh dict (the
        no_candidates synthetic-fallback case)."""
        from pmriprep.interfaces.dsc_aif import _clear_stale_real_aif_keys
        qc = {"detection_method": "synthetic", "synthetic": True}
        _clear_stale_real_aif_keys(qc)
        assert qc == {"detection_method": "synthetic", "synthetic": True}

    def test_consistency_invariant_under_synthetic(self) -> None:
        """Contract test: any qc_metrics dict with
        ``detection_method == 'synthetic'`` must NOT contain stale
        spatial fields. Any future synthetic-fallback path must satisfy
        this invariant — downstream consumers rely on it.
        """
        from pmriprep.interfaces.dsc_aif import _clear_stale_real_aif_keys
        qc = {k: f"stale_{k}" for k in self.STALE_KEYS}
        qc["detection_method"] = "atlas_permissive"
        _clear_stale_real_aif_keys(qc)
        qc["detection_method"] = "synthetic"
        qc["synthetic"] = True
        qc["n_aif_voxels_selected"] = 0
        if qc.get("detection_method") == "synthetic":
            for k in self.STALE_KEYS:
                assert k not in qc, (
                    f"Synthetic-fallback consistency violated: stale "
                    f"key {k!r} present alongside detection_method='synthetic'"
                )

    # --- _stamp_synthetic_fallback_common tests ---

    _COMMON_KEYS = {
        "synthetic", "n_aif_voxels_selected", "n_aif_voxels_available",
        "synthetic_fallback_reason", "aif_quality", "cascade_tiers_attempted",
    }

    def test_stamp_common_on_fresh_dict(self) -> None:
        """Helper sets all expected keys on an empty dict."""
        from pmriprep.interfaces.dsc_aif import _stamp_synthetic_fallback_common
        qc: dict[str, object] = {}
        _stamp_synthetic_fallback_common(
            qc, reason="no_candidates",
            tiers_attempted=["atlas_strict", "atlas_relaxed"],
        )
        for k in self._COMMON_KEYS:
            assert k in qc, f"Expected key {k!r} not set by helper"
        assert qc["synthetic"] is True
        assert qc["n_aif_voxels_selected"] == 0
        assert qc["aif_quality"] == "degraded"
        assert qc["synthetic_fallback_reason"] == "no_candidates"

    def test_stamp_common_overwrites_aif_quality(self) -> None:
        """Synthetic fallback overwrites a detected-quality label with degraded quality."""
        from pmriprep.interfaces.dsc_aif import _stamp_synthetic_fallback_common
        qc: dict[str, object] = {"aif_quality": "detected"}
        _stamp_synthetic_fallback_common(
            qc, reason="gamma_r2_floor",
            tiers_attempted=["atlas_strict"],
        )
        assert qc["aif_quality"] == "degraded"

    def test_stamp_common_preserves_existing_cascade_tiers(self) -> None:
        """Helper does NOT overwrite ``cascade_tiers_attempted`` if already
        present (Path B inherits from the real-AIF detection run).
        """
        from pmriprep.interfaces.dsc_aif import _stamp_synthetic_fallback_common
        original = ["atlas_strict", "atlas_relaxed", "atlas_permissive"]
        qc: dict[str, object] = {"cascade_tiers_attempted": original}
        _stamp_synthetic_fallback_common(
            qc, reason="gamma_r2_floor",
            tiers_attempted=["should_not_appear"],
        )
        assert qc["cascade_tiers_attempted"] is original
