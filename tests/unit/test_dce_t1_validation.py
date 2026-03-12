"""Unit tests for dce_t1_validation module.

Tests tissue-segmented T1 fallback, NaN abandon threshold,
auto-correction tiers, T1CorrectionResult serialization,
and interior NaN gap-fill.
"""

import numpy as np
import pytest

from pmriprep.utils.dce_t1_validation import (
    GAPFILL_MIN_NEIGHBOR_FLOOR,
    GAPFILL_MIN_NEIGHBOR_FRACTION,
    T1CorrectionResult,
    T1GapFillResult,
    T1_TISSUE_REFERENCE,
    VFA_NAN_FRACTION_GOOD,
    VFA_NAN_FRACTION_TISSUE_FALLBACK,
    apply_t1_correction,
    build_tissue_t1_map,
    fill_t1_gaps,
    get_field_strength_key,
    get_nan_abandon_threshold,
)


# ============================================================================
# get_nan_abandon_threshold tests
# ============================================================================


class TestGetNanAbandonThreshold:
    """Tests for VFA NaN abandon threshold calculation."""

    def test_well_conditioned_returns_base_threshold(self) -> None:
        """E1 < 0.99 → standard 0.40 threshold."""
        assert get_nan_abandon_threshold(0.98) == VFA_NAN_FRACTION_TISSUE_FALLBACK

    def test_ill_conditioned_reduces_threshold(self) -> None:
        """E1 = 0.995 → threshold between 0.20 and 0.40."""
        thresh = get_nan_abandon_threshold(0.995)
        assert VFA_NAN_FRACTION_GOOD < thresh < VFA_NAN_FRACTION_TISSUE_FALLBACK

    def test_severely_ill_conditioned_returns_minimum(self) -> None:
        """E1 >= 1.0 → threshold clamped at 0.20."""
        # E1=0.999 gives 0.40 - 0.009*20 = 0.22, not yet at minimum
        assert get_nan_abandon_threshold(0.999) == pytest.approx(0.22, abs=0.01)
        # E1=1.0 → 0.40 - 0.01*20 = 0.20 (clamped)
        assert get_nan_abandon_threshold(1.0) == VFA_NAN_FRACTION_GOOD
        assert get_nan_abandon_threshold(1.001) == VFA_NAN_FRACTION_GOOD

    def test_boundary_at_0_99(self) -> None:
        """E1 = 0.99 exactly → returns 0.40."""
        assert get_nan_abandon_threshold(0.99) == pytest.approx(
            VFA_NAN_FRACTION_TISSUE_FALLBACK
        )

    def test_linear_interpolation(self) -> None:
        """Mid-point E1=0.9945 → threshold ~0.31."""
        thresh = get_nan_abandon_threshold(0.9945)
        expected = 0.40 - (0.9945 - 0.99) * 20.0  # 0.31
        assert thresh == pytest.approx(expected, abs=0.01)


# ============================================================================
# build_tissue_t1_map tests
# ============================================================================


class TestBuildTissueT1Map:
    """Tests for tissue-segmented T1 map construction."""

    def test_pure_wm_returns_wm_t1(self) -> None:
        """Pure WM probability → WM midpoint T1."""
        shape = (10, 10, 10)
        wm = np.ones(shape)
        gm = np.zeros(shape)
        t1_map = build_tissue_t1_map(wm, gm, field_strength=3.0, smooth_fwhm_vox=0)
        wm_range = T1_TISSUE_REFERENCE[3.0]["wm"]
        expected_wm_t1 = (wm_range[0] + wm_range[1]) / 2.0
        assert np.median(t1_map) == pytest.approx(expected_wm_t1, rel=0.01)

    def test_pure_gm_returns_gm_t1(self) -> None:
        """Pure GM probability → GM midpoint T1."""
        shape = (10, 10, 10)
        wm = np.zeros(shape)
        gm = np.ones(shape)
        t1_map = build_tissue_t1_map(wm, gm, field_strength=3.0, smooth_fwhm_vox=0)
        gm_range = T1_TISSUE_REFERENCE[3.0]["gm"]
        expected_gm_t1 = (gm_range[0] + gm_range[1]) / 2.0
        assert np.median(t1_map) == pytest.approx(expected_gm_t1, rel=0.01)

    def test_low_probability_uses_expected_median(self) -> None:
        """Both probs near zero → expected_median fallback."""
        shape = (10, 10, 10)
        wm = np.full(shape, 0.01)
        gm = np.full(shape, 0.01)
        t1_map = build_tissue_t1_map(wm, gm, field_strength=3.0, smooth_fwhm_vox=0)
        expected = T1_TISSUE_REFERENCE[3.0]["expected_median"]
        assert np.median(t1_map) == pytest.approx(expected, rel=0.01)

    def test_mixed_tissue_weighted_average(self) -> None:
        """50/50 WM/GM → midpoint of WM and GM T1."""
        shape = (10, 10, 10)
        wm = np.full(shape, 0.5)
        gm = np.full(shape, 0.5)
        t1_map = build_tissue_t1_map(wm, gm, field_strength=3.0, smooth_fwhm_vox=0)
        wm_t1 = sum(T1_TISSUE_REFERENCE[3.0]["wm"]) / 2.0
        gm_t1 = sum(T1_TISSUE_REFERENCE[3.0]["gm"]) / 2.0
        expected = (wm_t1 + gm_t1) / 2.0
        assert np.median(t1_map) == pytest.approx(expected, rel=0.01)

    def test_smoothing_reduces_variance(self) -> None:
        """Smoothing should reduce spatial variance."""
        shape = (20, 20, 20)
        rng = np.random.default_rng(42)
        wm = rng.random(shape)
        gm = 1.0 - wm
        t1_no_smooth = build_tissue_t1_map(wm, gm, 3.0, smooth_fwhm_vox=0)
        t1_smoothed = build_tissue_t1_map(wm, gm, 3.0, smooth_fwhm_vox=3.0)
        assert np.std(t1_smoothed) < np.std(t1_no_smooth)

    def test_no_nan_in_output(self) -> None:
        """Output should contain no NaN values."""
        shape = (10, 10, 10)
        wm = np.zeros(shape)
        gm = np.zeros(shape)
        t1_map = build_tissue_t1_map(wm, gm, field_strength=3.0)
        assert not np.any(np.isnan(t1_map))

    def test_field_strength_1_5t(self) -> None:
        """1.5T should produce lower T1 values than 3T."""
        shape = (10, 10, 10)
        wm = np.ones(shape)
        gm = np.zeros(shape)
        t1_1_5 = build_tissue_t1_map(wm, gm, 1.5, smooth_fwhm_vox=0)
        t1_3_0 = build_tissue_t1_map(wm, gm, 3.0, smooth_fwhm_vox=0)
        assert np.median(t1_1_5) < np.median(t1_3_0)

    def test_output_dtype_float32(self) -> None:
        """Output should be float32."""
        shape = (5, 5, 5)
        t1_map = build_tissue_t1_map(np.ones(shape), np.zeros(shape), 3.0)
        assert t1_map.dtype == np.float32


# ============================================================================
# T1CorrectionResult tests
# ============================================================================


class TestT1CorrectionResult:
    """Tests for T1CorrectionResult dataclass."""

    def test_to_dict_includes_nan_fraction(self) -> None:
        """nan_fraction should appear in dict when set."""
        result = T1CorrectionResult(nan_fraction=0.55)
        d = result.to_dict()
        assert "nan_fraction" in d
        assert d["nan_fraction"] == 0.55

    def test_to_dict_includes_tissue_t1_used(self) -> None:
        """tissue_t1_used should always appear in dict."""
        result = T1CorrectionResult()
        d = result.to_dict()
        assert "tissue_t1_used" in d
        assert d["tissue_t1_used"] is False

    def test_to_dict_tissue_t1_true(self) -> None:
        result = T1CorrectionResult(tissue_t1_used=True)
        d = result.to_dict()
        assert d["tissue_t1_used"] is True

    def test_to_dict_nan_fraction_none_excluded(self) -> None:
        """nan_fraction=None should not appear in dict."""
        result = T1CorrectionResult()
        d = result.to_dict()
        assert "nan_fraction" not in d


# ============================================================================
# apply_t1_correction auto-mode tests
# ============================================================================


class TestApplyT1CorrectionAuto:
    """Tests for auto-mode T1 correction with tissue fallback."""

    @pytest.fixture
    def brain_mask(self) -> np.ndarray:
        """10x10x10 brain mask (inner 6x6x6)."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        return mask

    @pytest.fixture
    def probsegs(self) -> tuple[np.ndarray, np.ndarray]:
        """WM and GM probability maps for 10x10x10 volume."""
        shape = (10, 10, 10)
        wm = np.full(shape, 0.6)
        gm = np.full(shape, 0.4)
        return wm, gm

    def test_high_nan_with_probsegs_uses_tissue(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """High NaN + probsegs → tissue_segmented correction."""
        # Create T1 map with high median (above reject_max)
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        t1_corrected, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.50, wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "tissue_segmented"
        assert result.tissue_t1_used is True
        # Brain voxels should have valid T1 values
        valid_in_brain = brain_mask & np.isfinite(t1_corrected)
        assert np.sum(valid_in_brain) > 0
        # Outside brain should be NaN (preserved for downstream masking)
        assert np.all(np.isnan(t1_corrected[~brain_mask]))

    def test_high_nan_without_probsegs_uses_uniform(
        self, brain_mask: np.ndarray,
    ) -> None:
        """High NaN + no probsegs → uniform fallback."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan

        t1_corrected, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.50,
        )
        assert result.correction_applied == "fallback"
        assert result.tissue_t1_used is False

    def test_low_nan_existing_behavior(self, brain_mask: np.ndarray) -> None:
        """Low NaN fraction → existing offset/none behavior unchanged."""
        # T1 within acceptable range
        t1_map = np.full((10, 10, 10), 1050.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.05,
        )
        assert result.correction_applied == "none"

    def test_nan_fraction_stored_in_result(self, brain_mask: np.ndarray) -> None:
        """nan_fraction should be stored in correction result."""
        t1_map = np.full((10, 10, 10), 1050.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.15,
        )
        assert result.nan_fraction == 0.15

    def test_reject_below_min_with_probsegs(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Below reject_min + high NaN + probsegs → tissue_segmented."""
        t1_map = np.full((10, 10, 10), 400.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.50, wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "tissue_segmented"

    def test_offset_mode_unchanged(self, brain_mask: np.ndarray) -> None:
        """Explicit offset mode should not use tissue fallback."""
        t1_map = np.full((10, 10, 10), 1200.0)
        t1_map[~brain_mask] = np.nan
        wm = np.full((10, 10, 10), 0.6)
        gm = np.full((10, 10, 10), 0.4)

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="offset",
            brain_mask=brain_mask,
            nan_fraction=0.50, wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "offset"

    def test_low_nan_with_high_median_offsets(
        self, brain_mask: np.ndarray,
    ) -> None:
        """Low NaN + out-of-range median but no probsegs → fallback (not tissue)."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.10,
        )
        # Low NaN without probsegs → uniform fallback
        assert result.correction_applied == "fallback"

    def test_e1_dependent_threshold_triggers_tissue(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Ill-conditioned E1 lowers threshold, triggering tissue fallback at 25% NaN."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        # E1=0.996 → threshold ~0.28, so 30% NaN triggers tissue
        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.30, wm_probseg=wm, gm_probseg=gm,
            e1_median=0.996,
        )
        assert result.correction_applied == "tissue_segmented"

    def test_well_conditioned_e1_requires_higher_nan(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Well-conditioned E1 keeps threshold at 0.40, so 30% NaN → uniform fallback."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        # E1=0.95 → threshold=0.40, so 30% NaN < threshold → uniform fallback
        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.30, wm_probseg=wm, gm_probseg=gm,
            e1_median=0.95,
        )
        assert result.correction_applied == "fallback"


# ============================================================================
# VFA conditioning QC metric computation
# ============================================================================


class TestVFAConditioning:
    """Tests for VFA conditioning assessment."""

    def test_e1_well_conditioned(self) -> None:
        """TR=50ms, T1=1000ms → E1=0.951, well-conditioned."""
        tr_s = 0.050
        t1_s = 1.0
        e1 = np.exp(-tr_s / t1_s)
        assert e1 < 0.99

    def test_e1_ill_conditioned(self) -> None:
        """TR=5ms, T1=1000ms → E1=0.995, severely ill-conditioned."""
        tr_s = 0.005
        t1_s = 1.0
        e1 = np.exp(-tr_s / t1_s)
        assert e1 > 0.995


# ============================================================================
# skip_t1 path tests (via apply_t1_correction, not interface)
# ============================================================================


class TestFieldStrengthOrderingMixedTissue:
    """Test that 50/50 WM/GM tissue T1 increases with field strength."""

    def test_t1_ordering_1_5_lt_3_lt_7(self) -> None:
        """Mixed tissue T1: 1.5T < 3T < 7T."""
        shape = (10, 10, 10)
        wm = np.full(shape, 0.5)
        gm = np.full(shape, 0.5)

        t1_1_5 = build_tissue_t1_map(wm, gm, 1.5, smooth_fwhm_vox=0)
        t1_3_0 = build_tissue_t1_map(wm, gm, 3.0, smooth_fwhm_vox=0)
        t1_7_0 = build_tissue_t1_map(wm, gm, 7.0, smooth_fwhm_vox=0)

        median_1_5 = float(np.median(t1_1_5))
        median_3_0 = float(np.median(t1_3_0))
        median_7_0 = float(np.median(t1_7_0))

        assert median_1_5 < median_3_0 < median_7_0, (
            f"Expected T1 ordering 1.5T ({median_1_5:.0f}) < "
            f"3T ({median_3_0:.0f}) < 7T ({median_7_0:.0f})"
        )


class TestReferenceTableSelfConsistency:
    """Test that derived values in T1_TISSUE_REFERENCE are self-consistent."""

    @pytest.mark.parametrize("field_key", [1.5, 3.0, 7.0])
    def test_qc_max_strict_equals_gm_upper(self, field_key: float) -> None:
        """qc_max_strict should equal the upper bound of GM range."""
        ref = T1_TISSUE_REFERENCE[field_key]
        assert ref["qc_max_strict"] == ref["gm"][1], (
            f"{field_key}T: qc_max_strict ({ref['qc_max_strict']}) "
            f"!= gm upper ({ref['gm'][1]})"
        )

    @pytest.mark.parametrize("field_key", [1.5, 3.0, 7.0])
    def test_qc_max_lenient_equals_strict_plus_sd(self, field_key: float) -> None:
        """qc_max_lenient should equal qc_max_strict + brain_sd."""
        ref = T1_TISSUE_REFERENCE[field_key]
        assert ref["qc_max_lenient"] == ref["qc_max_strict"] + ref["brain_sd"], (
            f"{field_key}T: qc_max_lenient ({ref['qc_max_lenient']}) "
            f"!= qc_max_strict + brain_sd "
            f"({ref['qc_max_strict']} + {ref['brain_sd']})"
        )

    @pytest.mark.parametrize("field_key", [1.5, 3.0, 7.0])
    def test_expected_median_equals_brain_mean(self, field_key: float) -> None:
        """expected_median should equal brain_mean."""
        ref = T1_TISSUE_REFERENCE[field_key]
        assert ref["expected_median"] == ref["brain_mean"], (
            f"{field_key}T: expected_median ({ref['expected_median']}) "
            f"!= brain_mean ({ref['brain_mean']})"
        )

    @pytest.mark.parametrize("field_key", [1.5, 3.0, 7.0])
    def test_warn_high_equals_expected_plus_sd(self, field_key: float) -> None:
        """warn_high should equal expected_median + brain_sd."""
        ref = T1_TISSUE_REFERENCE[field_key]
        assert ref["warn_high"] == ref["expected_median"] + ref["brain_sd"], (
            f"{field_key}T: warn_high ({ref['warn_high']}) "
            f"!= expected_median + brain_sd "
            f"({ref['expected_median']} + {ref['brain_sd']})"
        )


class TestSkipT1Paths:
    """Tests for skip_t1-equivalent behavior in apply_t1_correction."""

    def test_fallback_mode_returns_uniform(self) -> None:
        """Explicit fallback mode → uniform literature value."""
        shape = (5, 5, 5)
        mask = np.ones(shape, dtype=bool)
        t1_map = np.full(shape, 1050.0)

        t1_corr, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="fallback",
            brain_mask=mask,
        )
        assert result.correction_applied == "fallback"
        assert result.fallback_value_ms == T1_TISSUE_REFERENCE[3.0]["default"]

    def test_none_mode_returns_unchanged(self) -> None:
        """None mode → no changes."""
        shape = (5, 5, 5)
        mask = np.ones(shape, dtype=bool)
        t1_map = np.full(shape, 1050.0)

        t1_corr, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="none",
            brain_mask=mask,
        )
        assert result.correction_applied == "none"
        np.testing.assert_array_equal(t1_corr, t1_map)


# ============================================================================
# T1 gap-fill tests
# ============================================================================


class TestT1GapFill:
    """Tests for interior NaN gap-fill in T1 maps."""

    @pytest.fixture
    def brain_mask_20(self) -> np.ndarray:
        """20x20x20 brain mask (inner 4:16 cube)."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[4:16, 4:16, 4:16] = True
        return mask

    @pytest.fixture
    def t1_map_20(self, brain_mask_20: np.ndarray) -> np.ndarray:
        """20x20x20 T1 map with valid values in brain, NaN outside."""
        t1 = np.full((20, 20, 20), np.nan, dtype=np.float32)
        t1[brain_mask_20] = 1050.0  # typical 3T brain T1
        return t1

    def test_no_gaps_returns_unchanged(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """No NaN gaps → returns copy, method='none'."""
        t1_out, result = fill_t1_gaps(t1_map_20, brain_mask_20, field_strength=3.0)
        assert result.method == "none"
        assert result.n_nan_before == 0
        np.testing.assert_array_equal(t1_out[brain_mask_20], t1_map_20[brain_mask_20])

    def test_scattered_nans_filled_by_interp(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Scattered interior NaN voxels filled by IDW interpolation."""
        rng = np.random.default_rng(42)
        # Add spatial variation so interpolation is meaningful
        t1_map_20[brain_mask_20] = rng.normal(1050.0, 50.0, int(np.sum(brain_mask_20))).astype(np.float32)
        # Punch scattered NaN holes in interior (far from boundary)
        interior = np.zeros_like(brain_mask_20)
        interior[7:13, 7:13, 7:13] = True
        gap_coords = np.argwhere(interior)
        chosen = rng.choice(len(gap_coords), size=20, replace=False)
        for idx in chosen:
            c = tuple(gap_coords[idx])
            t1_map_20[c] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_nan_before == 20
        assert result.n_filled_interp > 0
        assert result.n_remaining == 0
        assert "interp" in result.method

    def test_large_gap_stays_nan(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Large NaN hole with < min_neighbors → stays NaN (no tissue fallback)."""
        # Create a big hole that can't be filled by IDW
        t1_map_20[6:14, 6:14, 6:14] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_filled_tissue == 0
        assert result.n_remaining > 0
        # Interior of the large hole should still be NaN
        assert np.any(np.isnan(t1_out[brain_mask_20]))

    def test_large_gap_remains_nan(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Large gap without probsegs → remains NaN (downstream PK handles)."""
        t1_map_20[6:14, 6:14, 6:14] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_filled_tissue == 0
        assert result.n_remaining > 0

    def test_outside_brain_mask_stays_nan(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Voxels outside brain mask must remain NaN."""
        # Punch some gaps in brain
        t1_map_20[8:12, 8:12, 8:12] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
        )
        assert np.all(np.isnan(t1_out[~brain_mask_20]))

    def test_valid_values_never_changed(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Original valid T1 values must not be modified (bitwise equality)."""
        # Punch a few holes
        rng = np.random.default_rng(123)
        t1_map_20[brain_mask_20] = rng.normal(1050.0, 50.0, int(np.sum(brain_mask_20))).astype(np.float32)
        original = t1_map_20.copy()
        interior = np.zeros_like(brain_mask_20)
        interior[8:12, 8:12, 8:12] = True
        gap_coords = np.argwhere(interior)
        chosen = rng.choice(len(gap_coords), size=10, replace=False)
        for idx in chosen:
            c = tuple(gap_coords[idx])
            t1_map_20[c] = np.nan

        valid_before = np.isfinite(t1_map_20) & brain_mask_20
        t1_out, _ = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        # Boundary smoothing only applies to filled voxels, not original valid voxels
        np.testing.assert_array_equal(
            t1_out[valid_before], t1_map_20[valid_before],
        )

    def test_metrics_sum_correctly(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """n_nan_before == n_filled_interp + n_filled_tissue + n_remaining."""
        t1_map_20[7:13, 7:13, 7:13] = np.nan

        _, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_filled_tissue == 0
        assert result.n_nan_before == (
            result.n_filled_interp + result.n_filled_tissue + result.n_remaining
        )

    def test_output_dtype_float32(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Output must be float32."""
        t1_map_20[8:12, 8:12, 8:12] = np.nan
        t1_out, _ = fill_t1_gaps(t1_map_20, brain_mask_20, field_strength=3.0)
        assert t1_out.dtype == np.float32

    @pytest.mark.parametrize("field_strength", [1.5, 3.0, 7.0])
    def test_field_strengths(
        self, brain_mask_20: np.ndarray, field_strength: float,
    ) -> None:
        """Gap-fill works at all supported field strengths (scattered gaps)."""
        t1 = np.full((20, 20, 20), np.nan, dtype=np.float32)
        expected = T1_TISSUE_REFERENCE[get_field_strength_key(field_strength)]["expected_median"]
        t1[brain_mask_20] = expected
        # Punch scattered single-voxel holes (well-surrounded by valid neighbors)
        rng = np.random.default_rng(42)
        interior = np.zeros_like(brain_mask_20)
        interior[7:13, 7:13, 7:13] = True
        gap_coords = np.argwhere(interior)
        chosen = rng.choice(len(gap_coords), size=10, replace=False)
        for idx in chosen:
            c = tuple(gap_coords[idx])
            t1[c] = np.nan

        t1_out, result = fill_t1_gaps(
            t1, brain_mask_20, field_strength=field_strength,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_nan_before > 0
        assert result.n_remaining == 0

    def test_t1_gap_fill_result_to_dict(self) -> None:
        """T1GapFillResult.to_dict() returns expected keys."""
        result = T1GapFillResult(
            n_nan_before=100, n_filled_interp=80,
            n_filled_tissue=0, n_remaining=20, method="interp_only",
        )
        d = result.to_dict()
        assert d["n_nan_before"] == 100
        assert d["n_filled_interp"] == 80
        assert d["n_filled_tissue"] == 0
        assert d["n_remaining"] == 20
        assert d["method"] == "interp_only"

    def test_anisotropic_voxels(
        self, brain_mask_20: np.ndarray,
    ) -> None:
        """Thick-slice DCE (anisotropic voxels) uses ellipsoidal kernel."""
        t1 = np.full((20, 20, 20), np.nan, dtype=np.float32)
        t1[brain_mask_20] = 1050.0
        # Punch a single interior gap
        t1[10, 10, 10] = np.nan

        t1_out, result = fill_t1_gaps(
            t1, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([0.5, 0.5, 5.0]),
        )
        assert result.n_nan_before == 1
        assert result.n_filled_interp == 1
        assert np.isfinite(t1_out[10, 10, 10])

    def test_single_voxel_gap(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """One NaN surrounded by valid → filled near neighbor mean."""
        t1_map_20[10, 10, 10] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_nan_before == 1
        assert result.n_filled_interp == 1
        assert result.n_remaining == 0
        # Filled value should be close to surrounding value (1050.0)
        assert t1_out[10, 10, 10] == pytest.approx(1050.0, rel=0.01)

    def test_all_nan_brain(self, brain_mask_20: np.ndarray) -> None:
        """Entire brain NaN → graceful return, n_remaining == n_nan_before."""
        t1 = np.full((20, 20, 20), np.nan, dtype=np.float32)

        t1_out, result = fill_t1_gaps(
            t1, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_nan_before == int(np.sum(brain_mask_20))
        assert result.n_remaining == result.n_nan_before
        assert result.n_filled_interp == 0
        assert result.n_filled_tissue == 0
        # All brain voxels still NaN
        assert np.all(np.isnan(t1_out[brain_mask_20]))

    def test_fractional_threshold_scales_with_resolution(self) -> None:
        """Verify min_neighbors increases with finer resolution."""
        from pmriprep.utils.dce_t1_validation import (
            GAPFILL_IDW_RADIUS_MM,
        )

        # Coarse voxels: 3mm isotropic → small ball → floor applies
        vox_coarse = np.array([3.0, 3.0, 3.0])
        radius_vox_coarse = GAPFILL_IDW_RADIUS_MM / vox_coarse
        half_coarse = np.ceil(radius_vox_coarse).astype(int)
        grid_c = np.mgrid[
            -half_coarse[0]:half_coarse[0] + 1,
            -half_coarse[1]:half_coarse[1] + 1,
            -half_coarse[2]:half_coarse[2] + 1,
        ]
        dist_c = np.sqrt(
            (grid_c[0] * vox_coarse[0]) ** 2
            + (grid_c[1] * vox_coarse[1]) ** 2
            + (grid_c[2] * vox_coarse[2]) ** 2
        )
        n_ball_coarse = int(np.sum(dist_c <= GAPFILL_IDW_RADIUS_MM)) - 1
        min_neigh_coarse = max(
            GAPFILL_MIN_NEIGHBOR_FLOOR,
            int(n_ball_coarse * GAPFILL_MIN_NEIGHBOR_FRACTION),
        )

        # Fine voxels: 1.0mm isotropic → large ball → fraction dominates
        vox_fine = np.array([1.0, 1.0, 1.0])
        radius_vox_fine = GAPFILL_IDW_RADIUS_MM / vox_fine
        half_fine = np.ceil(radius_vox_fine).astype(int)
        grid_f = np.mgrid[
            -half_fine[0]:half_fine[0] + 1,
            -half_fine[1]:half_fine[1] + 1,
            -half_fine[2]:half_fine[2] + 1,
        ]
        dist_f = np.sqrt(
            (grid_f[0] * vox_fine[0]) ** 2
            + (grid_f[1] * vox_fine[1]) ** 2
            + (grid_f[2] * vox_fine[2]) ** 2
        )
        n_ball_fine = int(np.sum(dist_f <= GAPFILL_IDW_RADIUS_MM)) - 1
        min_neigh_fine = max(
            GAPFILL_MIN_NEIGHBOR_FLOOR,
            int(n_ball_fine * GAPFILL_MIN_NEIGHBOR_FRACTION),
        )

        # Fine resolution should require more neighbors
        assert min_neigh_fine > min_neigh_coarse
        # Coarse should hit floor
        assert min_neigh_coarse == GAPFILL_MIN_NEIGHBOR_FLOOR
