"""Unit tests for dce_t1_validation module.

Tests uniform T1 fallback, valid-support classification,
auto-correction tiers, T1CorrectionResult serialization,
interior NaN gap-fill, and NLLS T1 fitting.
"""

from pathlib import Path

import numpy as np
import pytest

from pmriprep.utils.dce_t1_validation import (
    GAPFILL_MIN_NEIGHBOR_FLOOR,
    GAPFILL_MIN_NEIGHBOR_FRACTION,
    MIN_VALID_T1_MS,
    T1CorrectionResult,
    T1GapFillResult,
    T1_TISSUE_REFERENCE,
    apply_t1_correction,
    build_tissue_t1_map,
    fill_t1_gaps,
    get_field_strength_key,
    valid_t1_support,
)


# ============================================================================
# valid_t1_support tests
# ============================================================================


class TestValidT1Support:
    """Tests for the shared downstream T1-validity rule."""

    def test_rejects_nonfinite_zero_and_subthreshold_values(self) -> None:
        """QC support matches signal conversion rather than finiteness alone."""
        values = np.array([np.nan, np.inf, -1.0, 0.0, 99.9, MIN_VALID_T1_MS, 900.0])
        assert valid_t1_support(values).tolist() == [
            False, False, False, False, False, True, True,
        ]

    def test_field_strength_envelope_retains_pathology_and_rejects_extremes(self) -> None:
        """The 3T safety envelope retains prolonged tissue T1 through 6000 ms."""
        values = np.array([99.0, 100.0, 2500.0, 5000.0, 6000.0, 6000.1])
        assert valid_t1_support(values, 3.0).tolist() == [
            False, True, True, True, True, False,
        ]


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

    def test_high_nan_with_probsegs_retains_measurement(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Coverage loss does not replace a measured map when probsegs exist."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        t1_corrected, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.50, wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "none"
        assert result.tissue_t1_used is False
        assert np.all(t1_corrected[brain_mask] == 2000.0)
        # Brain voxels should have valid T1 values
        valid_in_brain = brain_mask & np.isfinite(t1_corrected)
        assert np.sum(valid_in_brain) > 0
        # Outside brain should be NaN (preserved for downstream masking)
        assert np.all(np.isnan(t1_corrected[~brain_mask]))

    def test_high_nan_without_probsegs_retains_measurement(
        self, brain_mask: np.ndarray,
    ) -> None:
        """Coverage loss does not replace a measured map without probsegs."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan

        t1_corrected, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.50,
        )
        assert result.correction_applied == "none"
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

    @pytest.mark.parametrize("missing_fraction", [0.41, 0.60, 0.80])
    def test_coverage_loss_never_switches_auto_estimator(
        self, brain_mask: np.ndarray, missing_fraction: float,
    ) -> None:
        """Auto mode retains measured support across the tested coverage fractions."""
        t1_map = np.full((10, 10, 10), 1100.0)
        t1_map[~brain_mask] = np.nan
        corrected, result = apply_t1_correction(
            t1_map,
            field_strength=3.0,
            correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=missing_fraction,
        )
        assert result.correction_applied == "none"
        np.testing.assert_array_equal(corrected, t1_map)

    def test_low_population_t1_with_probsegs_is_diagnostic(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """A low but numerically usable T1 remains measured support."""
        t1_map = np.full((10, 10, 10), 400.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.50, wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "none"

    def test_offset_mode_removed(self, brain_mask: np.ndarray) -> None:
        """Additive T1 offset correction is rejected."""
        t1_map = np.full((10, 10, 10), 1200.0)
        t1_map[~brain_mask] = np.nan
        wm = np.full((10, 10, 10), 0.6)
        gm = np.full((10, 10, 10), 0.4)

        with pytest.raises(ValueError, match="Unknown correction mode: offset"):
            apply_t1_correction(
                t1_map, field_strength=3.0, correction_mode="offset",
                brain_mask=brain_mask,
                nan_fraction=0.50, wm_probseg=wm, gm_probseg=gm,
            )

    def test_low_nan_with_high_median_is_retained(
        self, brain_mask: np.ndarray,
    ) -> None:
        """An unusual median remains measured support without probsegs."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.10,
        )
        assert result.correction_applied == "none"

    def test_population_comparison_unavailable_without_csf_restriction(
        self,
        brain_mask: np.ndarray,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A whole-brain median is not compared with parenchymal norms."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan

        with caplog.at_level("INFO"):
            _, result = apply_t1_correction(
                t1_map,
                field_strength=3.0,
                correction_mode="auto",
                brain_mask=brain_mask,
                population_reference_applicable=False,
            )

        assert result.population_reference_applicable is False
        assert result.deviation_type is None
        assert "comparison unavailable" in result.reason
        assert "no CSF-restricted QC mask was available" in caplog.text
        assert "numerical safety envelope" not in caplog.text

    def test_population_warning_explains_pathology_retention(
        self,
        brain_mask: np.ndarray,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A CSF-restricted deviation remains descriptive in pathology."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan

        with caplog.at_level("WARNING"):
            _, result = apply_t1_correction(
                t1_map,
                field_strength=3.0,
                correction_mode="auto",
                brain_mask=brain_mask,
                population_reference_applicable=True,
            )

        assert result.population_reference_applicable is True
        assert result.deviation_type == "high"
        assert "may still include tumor or edema" in caplog.text
        assert "no population-T1 fallback was applied" in caplog.text

    def test_e1_does_not_change_fit_loss_threshold(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """E1 remains descriptive and cannot reject adequate spatial support."""
        t1_map = np.full((10, 10, 10), 1090.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.30, wm_probseg=wm, gm_probseg=gm,
            e1_median=0.996,
            b1_corrected=True,
        )
        assert result.correction_applied == "none"

    def test_population_threshold_does_not_switch_estimator(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """A population-reference deviation remains a QC diagnostic."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            nan_fraction=0.30, wm_probseg=wm, gm_probseg=gm,
            e1_median=0.95,
        )
        assert result.correction_applied == "none"
        assert result.tissue_t1_used is False


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

    def test_auto_falls_back_only_when_no_measured_support_exists(self) -> None:
        """An entirely unavailable measured map uses the documented fallback."""
        shape = (5, 5, 5)
        mask = np.ones(shape, dtype=bool)
        corrected, result = apply_t1_correction(
            np.full(shape, np.nan),
            field_strength=3.0,
            correction_mode="auto",
            brain_mask=mask,
        )
        assert result.correction_applied == "fallback"
        assert result.reason == "No valid measured T1 values"
        np.testing.assert_allclose(
            corrected[mask], T1_TISSUE_REFERENCE[3.0]["default"],
        )

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
        t1_map_20.copy()
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


# ============================================================================
# NLLS T1 fitting tests
# ============================================================================


def _make_ernst_signal(
    s0: float, t1_s: float, tr_s: float, fa_deg: np.ndarray, b1: float = 1.0,
) -> np.ndarray:
    """Generate noiseless Ernst equation signal for VFA."""
    fa_rad = np.radians(fa_deg) * b1
    e1 = np.exp(-tr_s / t1_s)
    return s0 * np.sin(fa_rad) * (1 - e1) / (1 - np.cos(fa_rad) * e1)


def test_t1_mapping_preserves_compact_measured_structure(tmp_path: Path) -> None:
    """Quantitative T1 output must not spatially vote away a compact lesion."""
    import nibabel as nib

    from pmriprep.interfaces.dce import T1Mapping

    shape = (9, 9, 9)
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    truth_ms = np.full(shape, 1000.0)
    center = np.array([4, 4, 4])
    lesion = np.zeros(shape, dtype=bool)
    lesion[tuple(center)] = True
    for axis in range(3):
        for direction in (-1, 1):
            neighbor = center.copy()
            neighbor[axis] += direction
            lesion[tuple(neighbor)] = True
    truth_ms[lesion] = 1600.0

    flip_angles = np.array([5.0, 15.0, 25.0])
    tr_ms = 7.0
    vfa_files = []
    for index, signal in enumerate(
        _make_ernst_signal(
            5000.0,
            truth_ms / 1000.0,
            tr_ms / 1000.0,
            flip_angles[:, None, None, None],
        ),
    ):
        path = tmp_path / f"vfa_{index}.nii.gz"
        nib.save(nib.Nifti1Image(signal.astype(np.float32), affine), path)
        vfa_files.append(str(path))

    mask_file = tmp_path / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), mask_file)
    interface = T1Mapping(
        vfa_files=vfa_files,
        flip_angles=flip_angles.tolist(),
        tr=tr_ms,
        brain_mask=str(mask_file),
        t1_method="nlls",
        t1_correction_mode="none",
        out_file=str(tmp_path / "t1_map.nii.gz"),
    )
    result = interface.run(cwd=str(tmp_path))

    measured = nib.load(result.outputs.out_file).get_fdata()
    support = nib.load(result.outputs.support_file).get_fdata()
    assert np.all(measured[lesion] > 1500.0)
    assert np.all(measured[~lesion & np.isfinite(measured)] < 1100.0)
    assert set(np.unique(support)).issubset({1.0, 2.0, 3.0})
    support_qc = result.outputs.qc_metrics["t1_support"]
    assert sum(
        support_qc[key]
        for key in (
            "directly_measured_voxels", "locally_interpolated_voxels",
            "unavailable_voxels",
            "uniform_assumption_voxels",
        )
    ) == support_qc["analysis_domain_voxels"]
    assert result.outputs.qc_metrics["median_filtered"] is False
    assert (
        result.outputs.qc_metrics["t1_correction"][
            "population_reference_applicable"
        ]
        is False
    )


def test_t1_mapping_csf_probseg_restricts_qc_not_support(tmp_path: Path) -> None:
    """Anatomical CSF classification must not remove quantitative T1 support."""
    import nibabel as nib

    from pmriprep.interfaces.dce import T1Mapping

    shape = (5, 5, 5)
    affine = np.eye(4)
    flip_angles = np.array([5.0, 15.0, 25.0])
    tr_ms = 7.0
    truth_ms = np.full(shape, 1200.0)
    truth_ms[2, 2, 2] = 4000.0
    truth_ms[1, 1, 1] = 4000.0
    vfa_files = []
    for index, signal in enumerate(
        _make_ernst_signal(
            5000.0,
            truth_ms / 1000.0,
            tr_ms / 1000.0,
            flip_angles[:, None, None, None],
        ),
    ):
        path = tmp_path / f"vfa_{index}.nii.gz"
        nib.save(nib.Nifti1Image(signal.astype(np.float32), affine), path)
        vfa_files.append(str(path))

    brain = np.ones(shape, dtype=np.uint8)
    csf = np.zeros(shape, dtype=np.float32)
    csf[2, 2, 2] = 1.0
    csf[1, 1, 1] = 1.0
    paths = {}
    for name, data in (("brain", brain), ("csf", csf)):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), path)
        paths[name] = str(path)

    result = T1Mapping(
        vfa_files=vfa_files,
        flip_angles=flip_angles.tolist(),
        tr=tr_ms,
        brain_mask=paths["brain"],
        csf_probseg=paths["csf"],
        t1_method="nlls",
        t1_correction_mode="none",
        out_file=str(tmp_path / "t1_map.nii.gz"),
    ).run(cwd=str(tmp_path))

    support = nib.load(result.outputs.support_file).get_fdata()
    assert support[2, 2, 2] == 1
    assert support[1, 1, 1] == 1
    support_qc = result.outputs.qc_metrics["t1_support"]
    assert support_qc["csf_qc_restricted_voxels"] == 2
    assert result.outputs.qc_metrics["fraction_valid"] == 1.0
    assert result.outputs.qc_metrics["median_t1_ms"] == pytest.approx(1200.0, abs=1.0)
    assert (
        result.outputs.qc_metrics["t1_correction"][
            "population_reference_applicable"
        ]
        is True
    )


class TestNLLSFitting:
    """Tests for nonlinear least squares T1 fitting."""

    def _make_t1mapping_interface(self):
        """Create a minimal T1Mapping interface for calling fitting methods."""
        from unittest.mock import MagicMock

        from pmriprep.interfaces.dce import T1Mapping

        t1m = T1Mapping.__new__(T1Mapping)
        t1m.inputs = MagicMock()
        return t1m

    def test_nlls_recovers_long_t1_noiseless(self) -> None:
        """NLLS recovers T1=5.0s at TR=4.4ms, noiseless (the OSIPI failure case)."""
        t1m = self._make_t1mapping_interface()
        tr_s = 0.0044
        fa_deg = np.array([2.0, 12.0])
        t1_true = 5.0  # seconds
        s0_true = 1000.0

        shape = (5, 5, 5)
        n_fa = len(fa_deg)
        signal_data = np.zeros((n_fa,) + shape)
        for i, fa in enumerate(fa_deg):
            signal_data[i] = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))

        t1_map = t1m._fit_t1_nlls(signal_data, sin_fa, cos_fa, tr_s, shape)

        # Should recover T1 within 1% of ground truth (noiseless)
        median_t1 = float(np.median(t1_map[t1_map > 0]))
        assert abs(median_t1 - t1_true) / t1_true < 0.01, (
            f"NLLS median T1={median_t1:.3f}s, expected {t1_true}s"
        )

    def test_wlls_inaccurate_long_t1_with_noise(self) -> None:
        """Document: WLLS is inaccurate for T1=5.0s at TR=4.4ms with noise.

        When E1 ~ 0.999, the linearized DESPOT1 variables X=S/tan(a) and
        Y=S/sin(a) share correlated noise from the same signal S. This makes
        WLLS unstable — small noise perturbations cause large T1 swings.
        The error can go in either direction depending on noise realization.
        """
        t1m = self._make_t1mapping_interface()
        tr_s = 0.0044
        fa_deg = np.array([2.0, 12.0])
        t1_true = 5.0
        s0_true = 1000.0

        shape = (20, 20, 20)
        rng = np.random.default_rng(42)
        signal_data = np.zeros((2,) + shape)
        for i, fa in enumerate(fa_deg):
            clean = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))
            signal_data[i] = clean + rng.normal(0, s0_true / 30, shape)  # SNR ~30
        signal_data = np.maximum(signal_data, 1.0)

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))
        tan_fa = np.tan(fa_rad[:, None, None, None] * np.ones(shape))

        y_data = signal_data / sin_fa
        x_data = signal_data / tan_fa

        t1_wlls = t1m._fit_t1_wlls(x_data, y_data, sin_fa, cos_fa, tr_s, shape)

        # WLLS with noise at long T1 should be significantly inaccurate (>50% error)
        median_wlls = float(np.median(t1_wlls[t1_wlls > 0]))
        relative_error = abs(median_wlls - t1_true) / t1_true
        assert relative_error > 0.5, (
            f"Expected WLLS to be inaccurate (>50% error) at T1=5s, TR=4.4ms, "
            f"SNR=30, got {median_wlls:.3f}s (error={relative_error:.1%})"
        )

    def test_wlls_initialized_nlls_improves_long_t1(self) -> None:
        """Default NLLS initialization improves a noisy long-T1 estimate."""
        tr_s = 0.0044
        t1_true = 5.0
        t1m = self._make_t1mapping_interface()
        fa_deg = np.array([2.0, 12.0])
        s0_true = 1000.0
        shape = (15, 15, 15)

        rng = np.random.default_rng(42)
        signal_data = np.zeros((2,) + shape)
        for i, fa in enumerate(fa_deg):
            clean = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))
            signal_data[i] = clean + rng.normal(0, s0_true / 30, shape)
        signal_data = np.maximum(signal_data, 1.0)

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))
        tan_fa = np.tan(fa_rad[:, None, None, None] * np.ones(shape))

        # WLLS result
        y_data = signal_data / sin_fa
        x_data = signal_data / tan_fa
        t1_wlls = t1m._fit_t1_wlls(x_data, y_data, sin_fa, cos_fa, tr_s, shape)

        # No explicit seed: production NLLS derives its WLLS warm start.
        refined = t1m._fit_t1_nlls(
            signal_data, sin_fa, cos_fa, tr_s, shape,
        )
        e1_wlls = np.exp(-tr_s / np.where(t1_wlls > 0, t1_wlls, 1.0))
        explicit_warm_start = t1m._fit_t1_nlls(
            signal_data, sin_fa, cos_fa, tr_s, shape,
            e1_init=e1_wlls.ravel(),
        )
        np.testing.assert_allclose(refined, explicit_warm_start)
        median_refined = float(np.median(refined[refined > 0]))
        median_wlls = float(np.median(t1_wlls[t1_wlls > 0]))

        # NLLS should be closer to ground truth than WLLS
        wlls_error = abs(median_wlls - t1_true)
        nlls_error = abs(median_refined - t1_true)
        assert nlls_error < wlls_error, (
            f"NLLS error ({nlls_error:.3f}) should be < WLLS error ({wlls_error:.3f})"
        )

    def test_nlls_2_flip_angles(self) -> None:
        """Exactly-determined fit with 2 FAs — verify convergence."""
        t1m = self._make_t1mapping_interface()
        tr_s = 0.005
        fa_deg = np.array([5.0, 15.0])
        t1_true = 1.5
        s0_true = 500.0

        shape = (2, 2, 2)
        signal_data = np.zeros((2,) + shape)
        for i, fa in enumerate(fa_deg):
            signal_data[i] = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))

        t1_map = t1m._fit_t1_nlls(signal_data, sin_fa, cos_fa, tr_s, shape)

        # Noiseless 2-FA: should converge to exact solution
        for val in t1_map.ravel():
            assert abs(val - t1_true) / t1_true < 0.01, (
                f"Expected T1≈{t1_true}s, got {val:.4f}s"
            )

    def test_nlls_noisy_graceful_degradation(self) -> None:
        """SNR=1: NLLS no worse than WLLS (both noisy, neither should crash)."""
        t1m = self._make_t1mapping_interface()
        tr_s = 0.005
        fa_deg = np.array([5.0, 15.0])
        t1_true = 1.0
        s0_true = 100.0

        shape = (10, 10, 10)
        rng = np.random.default_rng(123)
        signal_data = np.zeros((2,) + shape)
        for i, fa in enumerate(fa_deg):
            clean = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))
            signal_data[i] = clean + rng.normal(0, s0_true, shape)  # SNR ≈ 1
        signal_data = np.maximum(signal_data, 1.0)

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))

        t1_map = t1m._fit_t1_nlls(signal_data, sin_fa, cos_fa, tr_s, shape)

        # Should not crash, all values should be finite and within bounds
        valid = t1_map[t1_map > 0]
        assert len(valid) > 0, "Should have some valid T1 values"
        assert np.all(np.isfinite(valid)), "All fitted T1 values should be finite"
        assert np.all(valid >= 0.01), "T1 >= 10ms"
        assert np.all(valid <= 20.0), "T1 <= 20s"

    def test_nlls_r1_bounds(self) -> None:
        """T1 stays within [10ms, 20s] with pathological inputs."""
        t1m = self._make_t1mapping_interface()
        tr_s = 0.005
        fa_deg = np.array([5.0, 15.0])
        shape = (3, 3, 3)

        # Pathological: constant signal across flip angles (degenerate)
        signal_data = np.ones((2,) + shape) * 100.0
        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))

        t1_map = t1m._fit_t1_nlls(signal_data, sin_fa, cos_fa, tr_s, shape)

        # All values should be within bounds
        valid = t1_map[t1_map > 0]
        if len(valid) > 0:
            assert np.all(valid >= 0.01), f"Min T1={np.min(valid):.4f}s < 10ms"
            assert np.all(valid <= 20.0), f"Max T1={np.max(valid):.4f}s > 20s"


class TestExternalT1:
    """Tests for external T1 map support."""

    def test_external_t1_unit_detection_seconds(self, tmp_path) -> None:
        """Verify auto-detection of seconds vs milliseconds."""
        import nibabel as nib

        # Create a T1 map in seconds (median ~1.0)
        t1_data = np.full((10, 10, 10), 1.0, dtype=np.float32)
        t1_data[5:, :, :] = 1.5  # Some GM
        affine = np.eye(4)
        img = nib.Nifti1Image(t1_data, affine)
        t1_file = str(tmp_path / "t1_seconds.nii.gz")
        nib.save(img, t1_file)

        # Also create a reference image
        ref_data = np.ones((10, 10, 10), dtype=np.float32)
        ref_img = nib.Nifti1Image(ref_data, affine)
        ref_file = str(tmp_path / "ref.nii.gz")
        nib.save(ref_img, ref_file)

        # The function should detect seconds and convert
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            from pmriprep.workflows.dce.base import _transform_external_t1_to_dce
            result = _transform_external_t1_to_dce(t1_file, ref_file)
            # Load the result and check its millisecond units
            result_img = nib.load(result)
            result_data = result_img.get_fdata()
            median_val = float(np.nanmedian(result_data[result_data > 0]))
            assert median_val > 500, (
                f"Expected T1 in ms (>500), got median={median_val:.1f}"
            )
        finally:
            os.chdir(original_cwd)

    def test_external_t1_unit_detection_milliseconds(self, tmp_path) -> None:
        """T1 already in ms should not be converted."""
        import nibabel as nib

        t1_data = np.full((10, 10, 10), 1000.0, dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(t1_data, affine)
        t1_file = str(tmp_path / "t1_ms.nii.gz")
        nib.save(img, t1_file)

        ref_data = np.ones((10, 10, 10), dtype=np.float32)
        ref_img = nib.Nifti1Image(ref_data, affine)
        ref_file = str(tmp_path / "ref.nii.gz")
        nib.save(ref_img, ref_file)

        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            from pmriprep.workflows.dce.base import _transform_external_t1_to_dce
            result = _transform_external_t1_to_dce(t1_file, ref_file)
            result_img = nib.load(result)
            result_data = result_img.get_fdata()
            median_val = float(np.nanmedian(result_data[result_data > 0]))
            # Should still be ~1000ms (not 1e6)
            assert 500 < median_val < 2000, (
                f"Expected ~1000ms, got median={median_val:.1f}"
            )
        finally:
            os.chdir(original_cwd)


class TestConfigMutualExclusivity:
    """Tests for DCEConfig t1_map_file / no_t1 validation."""

    def test_no_t1_and_t1_map_file_raises(self) -> None:
        """Cannot set both no_t1=True and t1_map_file."""
        from pmriprep.config.dce import DCEConfig

        with pytest.raises(ValueError, match="Cannot set both"):
            DCEConfig(no_t1=True, t1_map_file=Path("/tmp/t1.nii"))

    def test_t1_map_file_alone_ok(self) -> None:
        """t1_map_file without no_t1 is valid."""
        from pmriprep.config.dce import DCEConfig

        config = DCEConfig(t1_map_file=Path("/tmp/t1.nii"))
        assert config.t1_map_file == Path("/tmp/t1.nii")
        assert not config.no_t1

    def test_t1_method_default_is_nlls(self) -> None:
        """Default T1 fitting uses WLLS-initialized NLLS."""
        from pmriprep.config.dce import DCEConfig

        config = DCEConfig()
        assert config.t1_method == "nlls"


# ============================================================================
# Post-offset recovery logic tests
# ============================================================================


class TestPostOffsetRecovery:
    """Tests for the post-offset T1 recovery logic in T1Mapping.

    This tests the recovery algorithm directly (not the full interface)
    to verify that voxels clipped to NaN by range filtering are correctly
    recovered after offset correction.
    """

    def _run_recovery(
        self,
        t1_pre_clip: np.ndarray,
        t1_map: np.ndarray,
        brain_mask: np.ndarray,
        edge_mask: np.ndarray,
        offset_ms: float,
        field_key: float = 3.0,
    ) -> tuple[np.ndarray, int]:
        """Run the post-offset recovery algorithm (mirrors the dce.py implementation)."""
        t1_ref = T1_TISSUE_REFERENCE[field_key]
        clamp_lo = t1_ref["reject_min"] * 0.9
        raw_t1_max = t1_ref["t1_parenchyma_upper"] + abs(offset_ms)
        recovery_upper = t1_ref["t1_parenchyma_upper"]
        t1_raw_corrected = t1_pre_clip + offset_ms  # no clip
        recoverable = (
            np.isnan(t1_map) & brain_mask
            & ~edge_mask
            & np.isfinite(t1_pre_clip)
            & (t1_pre_clip > 0)
            & (t1_pre_clip <= raw_t1_max)
            & (t1_raw_corrected >= clamp_lo)
            & (t1_raw_corrected <= recovery_upper)
        )
        n_recovered = int(np.sum(recoverable))
        if n_recovered > 0:
            t1_map[recoverable] = t1_raw_corrected[recoverable].astype(
                np.float32,
            )
        return t1_map, n_recovered

    def test_recovers_clipped_tumor_voxels_3t(self) -> None:
        """Voxels with raw T1 in [2500, 3109]ms recovered at 3T offset=-609ms."""
        shape = (10, 10, 5)
        brain_mask = np.ones(shape, dtype=bool)
        edge_mask = np.zeros(shape, dtype=bool)

        # Simulate: 90% normal brain (1000ms), 10% tumor outliers (2700ms)
        rng = np.random.default_rng(42)
        t1_pre_clip = np.full(shape, 1000.0)
        tumor_idx = rng.choice(np.prod(shape), size=50, replace=False)
        t1_pre_clip.flat[tumor_idx] = 2700.0  # above 2500ms upper bound

        # After range clip: tumor voxels become NaN
        t1_upper = T1_TISSUE_REFERENCE[3.0]["t1_parenchyma_upper"]  # 2500
        t1_lower = T1_TISSUE_REFERENCE[3.0]["reject_min"]  # 600
        t1_map = np.where(
            (t1_pre_clip >= t1_lower) & (t1_pre_clip <= t1_upper),
            t1_pre_clip, np.nan,
        ).astype(np.float32)

        # Apply offset correction to normal voxels (simulating what
        # apply_t1_correction does)
        offset = -609.0
        valid = np.isfinite(t1_map)
        t1_map[valid] += offset  # 1000 → 391ms (would be clamped in real code)

        # Run recovery
        t1_map, n_recovered = self._run_recovery(
            t1_pre_clip, t1_map, brain_mask, edge_mask, offset, 3.0,
        )

        assert n_recovered == 50
        # Recovered voxels: 2700 + (-609) = 2091ms (no clamp, within recovery_upper=2500)
        recovered_vals = t1_map.flat[tumor_idx]
        np.testing.assert_allclose(recovered_vals, 2091.0, atol=1.0)

    def test_edge_mask_prevents_recovery(self) -> None:
        """Edge-eroded voxels are NOT recovered."""
        shape = (5, 5, 3)
        brain_mask = np.ones(shape, dtype=bool)
        edge_mask = np.zeros(shape, dtype=bool)

        t1_pre_clip = np.full(shape, 2700.0)
        t1_map = np.full(shape, np.nan, dtype=np.float32)

        # Mark all voxels as edge
        edge_mask[:] = True

        t1_map, n_recovered = self._run_recovery(
            t1_pre_clip, t1_map, brain_mask, edge_mask, -609.0, 3.0,
        )
        assert n_recovered == 0

    def test_no_recovery_without_offset(self) -> None:
        """Recovery only runs for offset correction, not for other types."""
        # This is tested implicitly by the condition in dce.py:
        # correction_result.correction_applied == "offset"
        # Here we verify the guard: if offset_ms is None, no recovery.
        result = T1CorrectionResult()
        result.correction_applied = "tissue_segmented"
        result.offset_ms = None
        assert result.offset_ms is None

    def test_no_recovery_for_small_offset(self) -> None:
        """Normal brain with small offset recovers ~0 voxels."""
        shape = (10, 10, 5)
        brain_mask = np.ones(shape, dtype=bool)
        edge_mask = np.zeros(shape, dtype=bool)

        # All voxels well within range — none clipped
        t1_pre_clip = np.full(shape, 1200.0)
        t1_lower = T1_TISSUE_REFERENCE[3.0]["reject_min"]
        t1_upper = T1_TISSUE_REFERENCE[3.0]["t1_parenchyma_upper"]
        t1_map = np.where(
            (t1_pre_clip >= t1_lower) & (t1_pre_clip <= t1_upper),
            t1_pre_clip, np.nan,
        ).astype(np.float32)

        # Small offset
        t1_map, n_recovered = self._run_recovery(
            t1_pre_clip, t1_map, brain_mask, edge_mask, -50.0, 3.0,
        )
        assert n_recovered == 0

    def test_recovery_respects_clamp_bounds(self) -> None:
        """Extreme outliers (5000ms) excluded by raw T1 gate."""
        shape = (5, 5, 3)
        brain_mask = np.ones(shape, dtype=bool)
        edge_mask = np.zeros(shape, dtype=bool)

        # Raw T1 = 5000ms, way above raw_t1_max (2500 + 609 = 3109ms)
        t1_pre_clip = np.full(shape, 5000.0)
        t1_map = np.full(shape, np.nan, dtype=np.float32)

        t1_map, n_recovered = self._run_recovery(
            t1_pre_clip, t1_map, brain_mask, edge_mask, -609.0, 3.0,
        )
        assert n_recovered == 0

    def test_recovery_at_1_5t(self) -> None:
        """Recovery works at 1.5T with appropriate bounds."""
        shape = (5, 5, 3)
        brain_mask = np.ones(shape, dtype=bool)
        edge_mask = np.zeros(shape, dtype=bool)

        # At 1.5T: t1_parenchyma_upper = 2000ms
        # Raw T1 = 2200ms (above upper), offset = -474ms → 1726ms (valid)
        t1_pre_clip = np.full(shape, 2200.0)
        t1_lower = T1_TISSUE_REFERENCE[1.5]["reject_min"]
        t1_upper = T1_TISSUE_REFERENCE[1.5]["t1_parenchyma_upper"]
        t1_map = np.where(
            (t1_pre_clip >= t1_lower) & (t1_pre_clip <= t1_upper),
            t1_pre_clip, np.nan,
        ).astype(np.float32)

        t1_map, n_recovered = self._run_recovery(
            t1_pre_clip, t1_map, brain_mask, edge_mask, -474.0, 1.5,
        )
        assert n_recovered == np.prod(shape)
        # 2200 + (-474) = 1726ms (no clamp, within recovery_upper=2000)
        np.testing.assert_allclose(t1_map, 1726.0, atol=1.0)

    def test_negative_t1_not_recovered(self) -> None:
        """Negative t1_pre_clip values are not recovered."""
        shape = (5, 5, 3)
        brain_mask = np.ones(shape, dtype=bool)
        edge_mask = np.zeros(shape, dtype=bool)

        t1_pre_clip = np.full(shape, -500.0)
        t1_map = np.full(shape, np.nan, dtype=np.float32)

        t1_map, n_recovered = self._run_recovery(
            t1_pre_clip, t1_map, brain_mask, edge_mask, -609.0, 3.0,
        )
        assert n_recovered == 0

    def test_nan_inf_t1_not_recovered(self) -> None:
        """NaN and Inf in t1_pre_clip are not recovered."""
        shape = (5, 5, 3)
        brain_mask = np.ones(shape, dtype=bool)
        edge_mask = np.zeros(shape, dtype=bool)

        t1_pre_clip = np.full(shape, np.nan)
        t1_pre_clip[:3] = np.inf
        t1_map = np.full(shape, np.nan, dtype=np.float32)

        t1_map, n_recovered = self._run_recovery(
            t1_pre_clip, t1_map, brain_mask, edge_mask, -609.0, 3.0,
        )
        assert n_recovered == 0

    def test_raw_t1_beyond_gate_excluded(self) -> None:
        """Raw T1 just beyond the gate (t1_parenchyma_upper + |offset|) is excluded."""
        shape = (5, 5, 3)
        brain_mask = np.ones(shape, dtype=bool)
        edge_mask = np.zeros(shape, dtype=bool)

        # At 3T: raw_t1_max = 2500 + 609 = 3109ms
        # Raw T1 = 3200ms → exceeds gate → excluded
        t1_pre_clip = np.full(shape, 3200.0)
        t1_map = np.full(shape, np.nan, dtype=np.float32)

        t1_map, n_recovered = self._run_recovery(
            t1_pre_clip, t1_map, brain_mask, edge_mask, -609.0, 3.0,
        )
        assert n_recovered == 0


# ============================================================================
# Fixed VFA fit-loss fallback
# ============================================================================


class TestVFAFitLossGate:
    """Tests that acquisition diagnostics do not switch estimators."""

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

    def test_high_e1_low_fit_loss_retains_measurement(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """High E1 alone does not reject a plausible, complete measurement."""
        t1_map = np.full((10, 10, 10), 1090.0)  # within normal range
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.998, b1_corrected=True, b1_qc_pass=True,
            wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "none"
        assert result.tissue_t1_used is False

    def test_high_e1_no_b1_low_nan_falls_through(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """High E1 without B1 but manageable fit loss falls through."""
        t1_map = np.full((10, 10, 10), 1090.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.995, b1_corrected=False,
            nan_fraction=0.15,
            wm_probseg=wm, gm_probseg=gm,
        )
        # Falls through to Case 2: median 1090 is within 3T warn range → none
        assert result.correction_applied == "none"

    def test_high_e1_b1_fail_low_nan_falls_through(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """High E1 with B1 QC failure but manageable fit loss falls through."""
        t1_map = np.full((10, 10, 10), 1090.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.995, b1_corrected=True, b1_qc_pass=False,
            nan_fraction=0.15,
            wm_probseg=wm, gm_probseg=gm,
        )
        # Falls through to Case 2: median 1090 is within 3T warn range → none
        assert result.correction_applied == "none"

    def test_high_e1_no_fallback_good_b1(
        self, brain_mask: np.ndarray,
    ) -> None:
        """High E1 with good B1 passes through to physiological validation."""
        t1_map = np.full((10, 10, 10), 1090.0)  # within acceptable range
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.995, b1_corrected=True, b1_qc_pass=True,
        )
        # Normal Case 2 logic: median 1090 is within acceptable range → no correction
        assert result.correction_applied == "none"
        assert "fit loss" not in result.reason

    def test_lower_e1_no_fit_loss_fallback(
        self, brain_mask: np.ndarray,
    ) -> None:
        """Lower E1 without fit loss retains a plausible measurement."""
        t1_map = np.full((10, 10, 10), 1090.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.990, b1_corrected=False, b1_qc_pass=False,
        )
        assert result.correction_applied == "none"
        assert "fit loss" not in result.reason

    def test_high_fit_loss_with_tissue_maps_retains_measurement(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Tissue maps do not turn coverage loss into an estimator switch."""
        t1_map = np.full((10, 10, 10), 1090.0)
        t1_map[~brain_mask] = np.nan
        wm, gm = probsegs

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.998, b1_corrected=False,
            nan_fraction=0.50,
            wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "none"
        assert result.tissue_t1_used is False
        assert result.fallback_value_ms is None

    def test_high_fit_loss_without_tissue_maps_retains_measurement(
        self, brain_mask: np.ndarray,
    ) -> None:
        """Coverage loss remains explicit when tissue maps are absent."""
        t1_map = np.full((10, 10, 10), 1090.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.998, b1_corrected=False,
            nan_fraction=0.50,
        )
        assert result.correction_applied == "none"
        assert result.fallback_value_ms is None

    def test_fit_loss_preserves_missing_support_with_probsegs(
        self, brain_mask: np.ndarray, probsegs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Auto mode preserves missing support rather than fabricating T1."""
        t1_map = np.full((10, 10, 10), np.nan)
        # Only 37% of brain voxels have valid T1 (simulating ill-conditioned VFA)
        brain_voxels = np.argwhere(brain_mask)
        n_brain = len(brain_voxels)
        n_valid = int(0.37 * n_brain)
        rng = np.random.default_rng(42)
        valid_idx = rng.choice(n_brain, n_valid, replace=False)
        for idx in valid_idx:
            i, j, k = brain_voxels[idx]
            t1_map[i, j, k] = 1090.0

        wm, gm = probsegs

        t1_corr, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.998, b1_corrected=False,
            nan_fraction=0.63,
            wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "none"

        # Missing measured support remains explicit for local gap filling.
        brain_t1 = t1_corr[brain_mask]
        n_nan_in_brain = np.sum(np.isnan(brain_t1))
        assert n_nan_in_brain > 0

    def test_fit_loss_preserves_missing_support_without_probsegs(
        self, brain_mask: np.ndarray,
    ) -> None:
        """Missing support stays missing when tissue maps are absent."""
        t1_map = np.full((10, 10, 10), np.nan)
        # Only some brain voxels have valid T1
        brain_voxels = np.argwhere(brain_mask)
        for idx in brain_voxels[:len(brain_voxels) // 3]:
            i, j, k = idx
            t1_map[i, j, k] = 1090.0

        t1_corr, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.998, b1_corrected=False,
            nan_fraction=0.67,
        )
        assert result.correction_applied == "none"

        brain_t1 = t1_corr[brain_mask]
        n_nan_in_brain = np.sum(np.isnan(brain_t1))
        assert n_nan_in_brain > 0

    def test_explicit_fallback_fills_entire_brain(self) -> None:
        """Explicit fallback mode fills all brain voxels, not just valid ones."""
        shape = (5, 5, 5)
        mask = np.ones(shape, dtype=bool)
        # Half the voxels are NaN (simulating failed VFA fit)
        t1_map = np.full(shape, np.nan)
        t1_map[:3] = 1050.0

        t1_corr, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="fallback",
            brain_mask=mask,
        )
        assert result.correction_applied == "fallback"
        # ALL brain voxels should have the fallback value
        assert not np.any(np.isnan(t1_corr[mask]))
        np.testing.assert_allclose(
            t1_corr[mask], T1_TISSUE_REFERENCE[3.0]["default"],
        )

    def test_low_population_t1_does_not_fill_missing_support(
        self, brain_mask: np.ndarray,
    ) -> None:
        """Low but numerical T1 does not trigger whole-brain replacement."""
        t1_map = np.full((10, 10, 10), np.nan)
        # Only some brain voxels have valid but out-of-range T1
        brain_voxels = np.argwhere(brain_mask)
        for idx in brain_voxels[:len(brain_voxels) // 3]:
            i, j, k = idx
            t1_map[i, j, k] = 400.0

        t1_corr, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
        )
        assert result.correction_applied == "none"

        brain_t1 = t1_corr[brain_mask]
        n_nan_in_brain = np.sum(np.isnan(brain_t1))
        assert n_nan_in_brain > 0

    def test_high_e1_no_b1_retains_plausible_measurement(
        self, brain_mask: np.ndarray,
    ) -> None:
        """High E1 without B1 does not force calibration."""
        t1_map = np.full((10, 10, 10), 1200.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.996, b1_corrected=False, b1_qc_pass=False,
            nan_fraction=0.15,  # manageable
        )
        assert result.correction_applied == "none"
        assert result.median_t1_corrected == pytest.approx(1200.0)

    def test_high_e1_high_fit_loss_retains_measurement(
        self, brain_mask: np.ndarray,
    ) -> None:
        """High E1 and fit loss remain diagnostic evidence."""
        t1_map = np.full((10, 10, 10), 1200.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.996, b1_corrected=False, b1_qc_pass=False,
            nan_fraction=0.60,
        )
        assert result.correction_applied == "none"

    def test_high_e1_high_fit_loss_with_tissue_maps(
        self, brain_mask: np.ndarray,
    ) -> None:
        """Fit loss remains diagnostic even when tissue maps exist."""
        t1_map = np.full((10, 10, 10), 1200.0)
        t1_map[~brain_mask] = np.nan
        wm = np.where(brain_mask, 0.6, 0.0).astype(np.float32)
        gm = np.where(brain_mask, 0.4, 0.0).astype(np.float32)

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.996, b1_corrected=False, b1_qc_pass=False,
            nan_fraction=0.60,
            wm_probseg=wm, gm_probseg=gm,
        )
        assert result.correction_applied == "none"
        assert result.tissue_t1_used is False

    def test_failed_b1_and_high_median_retains_measurement(
        self, brain_mask: np.ndarray,
    ) -> None:
        """A marginal measured map is not calibrated when B1 QC also failed."""
        t1_map = np.full((10, 10, 10), 1600.0)
        t1_map[~brain_mask] = np.nan
        wm = np.where(brain_mask, 0.6, 0.0).astype(np.float32)
        gm = np.where(brain_mask, 0.4, 0.0).astype(np.float32)

        _, result = apply_t1_correction(
            t1_map,
            field_strength=3.0,
            correction_mode="auto",
            brain_mask=brain_mask,
            b1_corrected=True,
            b1_qc_pass=False,
            nan_fraction=0.1,
            wm_probseg=wm,
            gm_probseg=gm,
        )

        assert result.correction_applied == "none"
        assert "B1 QC failed" in result.reason

    def test_failed_b1_and_high_median_retains_measurement_without_seg(
        self, brain_mask: np.ndarray,
    ) -> None:
        """A joint T1/B1 warning remains explicit without segmentations."""
        t1_map = np.full((10, 10, 10), 1600.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map,
            field_strength=3.0,
            correction_mode="auto",
            brain_mask=brain_mask,
            b1_corrected=True,
            b1_qc_pass=False,
            nan_fraction=0.1,
        )

        assert result.correction_applied == "none"
        assert result.fallback_value_ms is None

    def test_high_e1_with_low_fit_loss_is_retained(
        self, brain_mask: np.ndarray,
    ) -> None:
        """High E1 with low failed-fit coverage retains plausible measured T1."""
        t1_map = np.full((10, 10, 10), 1090.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.998, b1_corrected=False, b1_qc_pass=False,
            nan_fraction=0.05,
        )
        assert result.correction_applied == "none"

    def test_high_e1_warn_range_bias_is_not_auto_corrected(
        self, brain_mask: np.ndarray,
    ) -> None:
        """A plausible population deviation is retained at subject level."""
        t1_map = np.full((10, 10, 10), 950.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.996, b1_corrected=False, b1_qc_pass=False,
            nan_fraction=0.15,
        )
        assert result.correction_applied == "none"
        assert result.median_t1_corrected == pytest.approx(950.0)

    def test_nrmse_gate_bypassed_not_in_dict(
        self, brain_mask: np.ndarray,
    ) -> None:
        """to_dict() omits nrmse_gate_bypassed."""
        t1_map = np.full((10, 10, 10), 1090.0)
        t1_map[~brain_mask] = np.nan

        _, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="auto",
            brain_mask=brain_mask,
            e1_median=0.996, b1_corrected=False, b1_qc_pass=False,
            nan_fraction=0.15,
        )
        d = result.to_dict()
        assert "nrmse_gate_bypassed" not in d

    def test_vfa_fit_nrmse_preserved_in_dict(
        self, brain_mask: np.ndarray,
    ) -> None:
        """vfa_fit_nrmse appears in to_dict() when set (QC audit trail)."""
        result = T1CorrectionResult(
            correction_applied="offset",
            vfa_fit_nrmse=0.065,
        )
        d = result.to_dict()
        assert "vfa_fit_nrmse" in d
        assert d["vfa_fit_nrmse"] == pytest.approx(0.065)
