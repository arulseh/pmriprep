"""Tests for AIF cascade logic and quality helper methods."""


import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import (
    AIFDetection,
)


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


class TestAccumulatingCascade:
    """Test the accumulating cascade with averaged-pool quality gate.

    The cascade accumulates candidates across tiers and uses the first passing
    averaged AIF. Tests exercise the helper methods that implement this logic.
    """


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
        assert result is not None
        assert len(result) == 6
        assert result.roi_size == 1
        assert isinstance(result.brr, float)


class TestUnionPoolTrim:
    """Test progressive worst-BRR trimming of a union candidate pool.

    The production pool gate rejects the contaminated average, and the
    production trim loop accepts it after removing the contaminator.
    """


    def _build_pool(self):
        """Build a five-voxel pool with one contaminated curve."""
        shape = (10, 10, 5, 120)
        conc_data = np.zeros(shape)
        brain_mask = np.ones(shape[:3])
        good_curve = _make_aif_curve(peak_height=30.0, recirculation_height=2.0)
        bad_curve = _curve_with_negative_tail(
            peak_height=30.0, tail_offset=-8.0,
        )
        # 5 coords; (4,4,2) is the contaminator
        union: list[tuple[float, tuple[int, int, int], dict]] = []
        conc_data[4, 4, 2, :] = bad_curve
        union.append((1.0, (4, 4, 2), {"baseline_return_ratio": -8.0}))
        conc_data[4, 5, 2, :] = good_curve
        union.append((1.1, (4, 5, 2), {"baseline_return_ratio": 10.0}))
        for i, c in enumerate([(5, 4, 2), (5, 5, 2), (4, 6, 2)]):
            conc_data[c[0], c[1], c[2], :] = good_curve
            union.append((1.2 + i * 0.1, c, {"baseline_return_ratio": 10.0}))
        return conc_data, brain_mask, union

    def test_single_shot_fails_on_contaminated_union(self):
        """Without trim, a 5-voxel union with one negative-BRR voxel
        averaged in fails the averaged-pool gate single-shot."""
        aif = AIFDetection()
        conc_data, brain_mask, union = self._build_pool()
        result = aif._try_aif_pool_acceptance(
            union, conc_data, brain_mask, tr=1.5, baseline_frames=20,
        )
        assert result is None

    def test_trim_drops_contaminator_and_passes(self):
        """With progressive trim, the contaminated voxel is dropped on
        the first iteration and the remaining 4-voxel pool passes."""
        conc_data, brain_mask, union = self._build_pool()
        accepted = AIFDetection()._try_trimmed_pool_acceptance(
            union, conc_data, brain_mask, tr=1.5, baseline_frames=20,
        )
        assert accepted is not None
        accepted_coords = {coord for _, coord, _ in accepted.candidates}
        # Contaminator (4,4,2) must be dropped; the 4 clean voxels remain.
        assert (4, 4, 2) not in accepted_coords
        assert {(4, 5, 2), (5, 4, 2), (5, 5, 2), (4, 6, 2)} == accepted_coords
