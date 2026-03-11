"""Unit tests for dce_t1_validation module.

Tests tissue-segmented T1 fallback, NaN abandon threshold,
auto-correction tiers, and T1CorrectionResult serialization.
"""

import numpy as np
import pytest

from pmriprep.utils.dce_t1_validation import (
    T1CorrectionResult,
    T1_TISSUE_REFERENCE,
    VFA_NAN_FRACTION_GOOD,
    VFA_NAN_FRACTION_TISSUE_FALLBACK,
    apply_t1_correction,
    build_tissue_t1_map,
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
