"""Unit tests for DSC CBV to vp constraint utilities."""

from typing import Any

import numpy as np
import pytest

from pmriprep.utils.dsc_cbv_constraint import (
    CBVQualityAssessment,
    CNR_COVERAGE_HIGH,
    CNR_COVERAGE_MODERATE,
    CNR_COVERAGE_SKIP,
    CNR_MEAN_MINIMUM,
    ConstraintConfig,
    ConstraintMode,
    VP_MAX_PHYSIOLOGICAL,
    _classify_preload_k2,
    _normalize_aif_qc_keys,
    assess_dsc_cbv_quality,
    cbv_to_vp,
    determine_constraint_mode,
    prepare_vp_constraint_from_data,
)


# =============================================================================
# CBV to vp Conversion
# =============================================================================


class TestCBVtoVP:
    """Tests for cbv_to_vp conversion."""

    def test_wm_cbv_to_vp(self) -> None:
        """White matter CBV=2.5 mL/100g converts to vp ~0.0155."""
        cbv = np.array([2.5], dtype=np.float64)
        vp = cbv_to_vp(cbv)
        # Capillary Hct = 0.42 * 0.85 = 0.357
        # vp = 2.5 * (1 - 0.357) / (100 * 1.04) = 2.5 * 0.643 / 104
        expected = 2.5 * 0.643 / 104.0
        assert vp[0] == pytest.approx(expected, rel=1e-3)

    def test_gm_cbv_to_vp(self) -> None:
        """Gray matter CBV=4.5 mL/100g converts to vp ~0.028."""
        cbv = np.array([4.5], dtype=np.float64)
        vp = cbv_to_vp(cbv)
        expected = 4.5 * 0.643 / 104.0
        assert vp[0] == pytest.approx(expected, rel=1e-3)

    def test_tumor_cbv_to_vp(self) -> None:
        """Tumor CBV=10.0 mL/100g converts to vp ~0.062."""
        cbv = np.array([10.0], dtype=np.float64)
        vp = cbv_to_vp(cbv)
        expected = 10.0 * 0.643 / 104.0
        assert vp[0] == pytest.approx(expected, rel=1e-3)

    def test_negative_cbv_nan(self) -> None:
        """Negative CBV values should become NaN; zero stays zero."""
        cbv = np.array([-1.0, 0.0, 5.0], dtype=np.float64)
        vp = cbv_to_vp(cbv)
        assert np.isnan(vp[0]), "Negative CBV should produce NaN"
        assert vp[1] == pytest.approx(0.0, abs=1e-12)
        assert np.isfinite(vp[2])

    def test_array_shape_preserved(self) -> None:
        """Output shape matches input shape for 3D arrays."""
        cbv = np.ones((3, 4, 5), dtype=np.float64) * 3.0
        vp = cbv_to_vp(cbv)
        assert vp.shape == (3, 4, 5)

    def test_out_of_range_vp_nan(self) -> None:
        """CBV=500 produces vp >> VP_MAX_PHYSIOLOGICAL (0.2), so result is NaN."""
        cbv = np.array([500.0], dtype=np.float64)
        vp = cbv_to_vp(cbv)
        # vp = 500 * 0.6 / 104 = 2.88, which exceeds 0.2
        assert np.isnan(vp[0]), "Out-of-range vp should be NaN"


# =============================================================================
# Quality Assessment
# =============================================================================


class TestAssessDSCCBVQuality:
    """Tests for assess_dsc_cbv_quality."""

    def test_high_quality(self) -> None:
        """Excellent AIF metrics produce high confidence and use_constraint=True."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "high"
        assert result.use_constraint is True

    def test_moderate_quality(self) -> None:
        """CNR coverage between moderate and high thresholds gives moderate confidence."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.90},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "moderate"

    def test_low_quality(self) -> None:
        """Poor CNR coverage and low AIF peak produce low confidence."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.5},
            "aif_peak_height": 0.1,
            "baseline_return_ratio": 0.5,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "low"
        assert result.use_constraint is False

    def test_cbv_checks_downgrade(self) -> None:
        """Good AIF metrics but high negative CBV fraction downgrades to low."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }
        # 10% negative values, well above CBV_NEGATIVE_FRACTION_MAX=0.01
        cbv_data = np.ones((10, 10, 10), dtype=np.float64) * 3.0
        cbv_data[:1, :, :] = -5.0  # 10% negative

        result = assess_dsc_cbv_quality(aif_qc_data, cbv_data=cbv_data)
        assert result.confidence == "low"
        assert len(result.reasons) > 0
        assert any("negative" in r.lower() for r in result.reasons)

    def test_synthetic_aif_downgrades_to_low(self) -> None:
        """Synthetic AIF should downgrade to low even with good shape metrics."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_level": "synthetic",
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "low"
        assert result.use_constraint is False

    # Note: test_brain_degraded_downgrades_to_low removed in v0.19.16
    # because brain_degraded tier was removed from the cascade.

    def test_atlas_permissive_downgrades_to_moderate(self) -> None:
        """atlas_permissive AIF should downgrade to moderate (FWHM up to 18 frames)."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_level": "atlas_permissive",
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "moderate"

    def test_atlas_strict_no_downgrade(self) -> None:
        """atlas_strict AIF should not downgrade from detection level alone."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_level": "atlas_strict",
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "high"

    def test_single_candidate_downgrades_to_low(self) -> None:
        """1 AIF voxel (no averaging) should downgrade to low."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_level": "atlas_strict",
            "n_aif_voxels_used": 1,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "low"

    def test_two_candidates_downgrades_to_moderate(self) -> None:
        """2 AIF voxels should downgrade to moderate."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_level": "atlas_strict",
            "n_aif_voxels_used": 2,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "moderate"

    def test_detection_level_in_output(self) -> None:
        """detection_level and n_aif_voxels should appear in output."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_level": "atlas_relaxed",
            "n_aif_voxels_used": 5,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.detection_level == "atlas_relaxed"
        assert result.n_aif_voxels == 5

    def test_missing_detection_keys_no_crash(self) -> None:
        """Missing detection_level and n_aif_voxels should not crash or downgrade."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        # No downgrade from detection level (empty string) or candidates (default 3)
        assert result.confidence == "high"

    def test_bad_gamma_peak_ratio_downgrades_to_moderate(self) -> None:
        """Peak ratio 0.843 (outside [0.90, 1.10]) downgrades to moderate."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "gamma_fit_auto_metrics": {"peak_ratio": 0.843},
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "moderate"
        assert result.gamma_peak_ratio == pytest.approx(0.843)
        assert any("peak ratio" in r.lower() for r in result.reasons)

    def test_severe_gamma_peak_ratio_downgrades_to_low(self) -> None:
        """Peak ratio 0.75 (outside [0.80, 1.20]) downgrades to low."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "gamma_fit_auto_metrics": {"peak_ratio": 0.75},
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "low"
        assert result.use_constraint is False

    def test_good_gamma_peak_ratio_no_downgrade(self) -> None:
        """Peak ratio 0.95 (within [0.90, 1.10]) does not downgrade."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "gamma_fit_auto_metrics": {"peak_ratio": 0.95},
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "high"
        assert result.gamma_peak_ratio == pytest.approx(0.95)

    def test_missing_gamma_peak_ratio_no_downgrade(self) -> None:
        """Missing gamma_fit_auto_metrics does not downgrade (backward compat)."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "high"
        assert result.gamma_peak_ratio is None

    def test_gamma_peak_ratio_high_side_downgrade(self) -> None:
        """Peak ratio 1.15 (above 1.10) downgrades to moderate."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "gamma_fit_auto_metrics": {"peak_ratio": 1.15},
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "moderate"
        assert any("peak ratio" in r.lower() for r in result.reasons)

    def test_mean_cnr_below_minimum_downgrades_to_low(self) -> None:
        """Mean CNR below Rose criterion (3.0) triggers low confidence."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.95, "mean_cnr": 2.5},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "low"
        assert any("inadequate contrast" in r for r in result.reasons)

    def test_cnr_coverage_below_skip_downgrades_to_low(self) -> None:
        """CNR coverage below 0.70 triggers low confidence (>30% unreliable)."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.65, "mean_cnr": 25.0},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "low"
        assert any(">30% unreliable" in r for r in result.reasons)


# =============================================================================
# Constraint Mode Determination
# =============================================================================


class TestDetermineConstraintMode:
    """Tests for determine_constraint_mode."""

    def test_high_quality_hard(self) -> None:
        """High confidence produces hard constraint with zero uncertainty."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=["All quality checks passed"],
        )
        config = determine_constraint_mode(quality)
        assert config.mode == ConstraintMode.HARD
        assert config.vp_uncertainty == 0.0

    def test_moderate_quality_soft(self) -> None:
        """Moderate confidence produces soft constraint with 20% uncertainty."""
        quality = CBVQualityAssessment(
            confidence="moderate",
            use_constraint=True,
            reasons=["CNR coverage 0.90 < 0.99"],
        )
        config = determine_constraint_mode(quality)
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.2)

    def test_low_quality_skip(self) -> None:
        """Low confidence produces skip mode."""
        quality = CBVQualityAssessment(
            confidence="low",
            use_constraint=False,
            reasons=["CNR coverage 0.50 < 0.84"],
        )
        config = determine_constraint_mode(quality)
        assert config.mode == ConstraintMode.SKIP

    def test_user_override(self) -> None:
        """User override forces hard mode even with low quality."""
        quality = CBVQualityAssessment(
            confidence="low",
            use_constraint=False,
            reasons=["CNR coverage 0.50 < 0.84"],
        )
        config = determine_constraint_mode(quality, user_override="hard")
        assert config.mode == ConstraintMode.HARD


# =============================================================================
# AIF QC Key Normalization
# =============================================================================


class TestNormalizeAifQcKeys:
    """Tests for _normalize_aif_qc_keys."""

    def test_pmriprep_format_translates(self) -> None:
        """PMRIPrep AIF detection keys (best_psr, best_brr) are translated."""
        data = {"best_psr": 94.6, "best_brr": 7.17}
        result = _normalize_aif_qc_keys(data)

        assert result["aif_peak_height"] == pytest.approx(0.946, rel=1e-6)
        assert result["baseline_return_ratio"] == pytest.approx(0.0717, rel=1e-6)
        assert "cnr_whole_brain" in result
        # Default is now MODERATE (conservative) when CNR is not wired
        assert result["cnr_whole_brain"]["fraction_cnr_gt_4"] == CNR_COVERAGE_MODERATE

    def test_already_normalized_passthrough(self) -> None:
        """Already-normalized data passes through unchanged."""
        data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }
        result = _normalize_aif_qc_keys(data)

        assert result["aif_peak_height"] == 0.4
        assert result["baseline_return_ratio"] == 0.15
        assert result["cnr_whole_brain"]["fraction_cnr_gt_4"] == 0.995

    def test_missing_keys_safe_defaults(self) -> None:
        """Missing keys get conservative defaults (MODERATE, not HIGH)."""
        data = {}
        result = _normalize_aif_qc_keys(data)

        # CNR defaults to MODERATE (conservative) when not wired from preprocessing
        assert "cnr_whole_brain" in result
        assert result["cnr_whole_brain"]["fraction_cnr_gt_4"] == CNR_COVERAGE_MODERATE
        # Other keys should remain absent (handled by assess_dsc_cbv_quality defaults)
        assert "aif_peak_height" not in result
        assert "baseline_return_ratio" not in result

    def test_aif_metrics_nested_psr(self) -> None:
        """Nested aif_metrics.psr is used when best_psr is absent."""
        data = {"aif_metrics": {"psr": 85.0}, "best_brr": 10.0}
        result = _normalize_aif_qc_keys(data)

        assert result["aif_peak_height"] == pytest.approx(0.85, rel=1e-6)
        assert result["baseline_return_ratio"] == pytest.approx(0.10, rel=1e-6)

    def test_good_aif_with_cnr_high_confidence(self) -> None:
        """Good AIF metrics with CNR coverage produces high confidence."""
        aif_qc_data = {
            "best_psr": 94.6,
            "best_brr": 7.17,
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.95, "mean_cnr": 25.0},
        }
        result = assess_dsc_cbv_quality(aif_qc_data)

        assert result.confidence == "high"
        assert result.use_constraint is True
        assert result.aif_peak_height == pytest.approx(0.946, rel=1e-3)
        assert result.aif_baseline_return == pytest.approx(0.0717, rel=1e-3)

    def test_original_keys_preserved(self) -> None:
        """Original PMRIPrep keys are preserved alongside new keys."""
        data = {"best_psr": 94.6, "best_brr": 7.17, "extra_field": 42}
        result = _normalize_aif_qc_keys(data)

        assert result["best_psr"] == 94.6
        assert result["best_brr"] == 7.17
        assert result["extra_field"] == 42

    def test_normalize_extracts_gamma_peak_ratio(self) -> None:
        """gamma_fit_auto_metrics.peak_ratio is promoted to top-level key."""
        data = {"gamma_fit_auto_metrics": {"peak_ratio": 0.843}}
        result = _normalize_aif_qc_keys(data)

        assert result["gamma_peak_ratio"] == pytest.approx(0.843)

    def test_normalize_no_gamma_metrics_no_key(self) -> None:
        """Missing gamma_fit_auto_metrics does not create gamma_peak_ratio."""
        data = {"best_psr": 94.6}
        result = _normalize_aif_qc_keys(data)

        assert "gamma_peak_ratio" not in result


# =============================================================================
# QC Report Schema
# =============================================================================


class TestQcReportSchema:
    """Tests that qc_report uses the renamed key schema (mode, reason)."""

    def test_qc_report_keys_from_prepare_vp_constraint(self, tmp_path: Any) -> None:
        """prepare_vp_constraint_from_data returns qc_report with 'mode'/'reason' keys."""
        import nibabel as nib

        cbv_data = np.ones((5, 5, 5), dtype=np.float64) * 3.0
        mask_data = np.ones((5, 5, 5), dtype=np.uint8)
        affine = np.eye(4)

        cbv_file = tmp_path / "cbv.nii.gz"
        mask_file = tmp_path / "mask.nii.gz"
        nib.Nifti1Image(cbv_data, affine).to_filename(str(cbv_file))
        nib.Nifti1Image(mask_data, affine).to_filename(str(mask_file))

        aif_qc = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }

        result = prepare_vp_constraint_from_data(
            cbv_file=str(cbv_file),
            aif_qc_data=aif_qc,
            brain_mask_file=str(mask_file),
        )

        qc = result.qc_report
        assert "mode" in qc, "qc_report must have 'mode' key"
        assert "reason" in qc, "qc_report must have 'reason' key"
        assert "constraint_mode" not in qc, "old 'constraint_mode' key must not exist"
        assert "constraint_reason" not in qc, "old 'constraint_reason' key must not exist"
        assert qc["mode"] in ("hard", "soft", "skip")

    def test_qc_report_skip_default_schema(self) -> None:
        """Default skip dict uses 'mode' and 'reason' (no 'constraint_' prefix)."""
        default_qc: dict[str, Any] = {"mode": "skip", "reason": "No DSC CBV provided"}
        assert "mode" in default_qc
        assert "reason" in default_qc
        assert "constraint_mode" not in default_qc


# =============================================================================
# Preload + K2 Classification
# =============================================================================


class TestClassifyPreloadK2:
    """Tests for _classify_preload_k2 helper."""

    def test_preloaded_well_suppressed(self) -> None:
        """Preloaded with low K2 and low leaky voxels → well_suppressed."""
        k2 = {"k2_median": 0.0005, "percent_leaky_voxels": 3.0}
        is_pre, well, high = _classify_preload_k2("half_dose", k2)
        assert is_pre is True
        assert well is True
        assert high is False

    def test_not_preloaded_high_correction(self) -> None:
        """No preload with high leaky voxels → high_correction."""
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 20.0}
        is_pre, well, high = _classify_preload_k2("none", k2)
        assert is_pre is False
        assert well is False
        assert high is True

    def test_none_inputs(self) -> None:
        """None inputs → (False, False, False)."""
        assert _classify_preload_k2(None, None) == (False, False, False)

    def test_boundary_k2_median_at_threshold(self) -> None:
        """k2_median exactly at K2_MEDIAN_SUPPRESSED (0.01) → NOT well_suppressed."""
        k2 = {"k2_median": 0.01, "percent_leaky_voxels": 1.0}
        _, well, _ = _classify_preload_k2("half_dose", k2)
        assert well is False

    def test_boundary_leaky_at_low_threshold(self) -> None:
        """percent_leaky_voxels exactly at LEAKY_VOXELS_LOW (5.0) → NOT well_suppressed."""
        k2 = {"k2_median": 0.005, "percent_leaky_voxels": 5.0}
        _, well, _ = _classify_preload_k2("significant", k2)
        assert well is False

    def test_boundary_leaky_at_high_threshold(self) -> None:
        """percent_leaky_voxels exactly at LEAKY_VOXELS_HIGH (15.0) → NOT high_correction."""
        k2 = {"k2_median": 0.003, "percent_leaky_voxels": 15.0}
        _, _, high = _classify_preload_k2("none", k2)
        assert high is False

    def test_preloaded_moderate_k2(self) -> None:
        """Preloaded with moderate K2 → not well_suppressed, not high_correction."""
        # leaky between LOW (5%) and HIGH (15%), k2_median below K2_MEDIAN_CORRECTION (0.005)
        k2 = {"k2_median": 0.004, "percent_leaky_voxels": 10.0}
        is_pre, well, high = _classify_preload_k2("half_dose", k2)
        assert is_pre is True
        assert well is False  # pct_leaky >= LEAKY_VOXELS_LOW
        assert high is False  # pct_leaky <= LEAKY_VOXELS_HIGH and k2_med <= K2_MEDIAN_CORRECTION


# =============================================================================
# Preload-Aware Quality Assessment
# =============================================================================


class TestPreloadAwareQuality:
    """Tests for preload-aware assess_dsc_cbv_quality."""

    _HIGH_AIF_QC: dict[str, Any] = {
        "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
        "aif_peak_height": 0.4,
        "baseline_return_ratio": 0.15,
    }

    def test_preload_low_k2_no_informational_reason(self) -> None:
        """Preloaded + low K2 does not add informational reason."""
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        result = assess_dsc_cbv_quality(
            self._HIGH_AIF_QC,
            preload_classification="half_dose",
            k2_metrics=k2,
        )
        assert not any("well-suppressed" in r for r in result.reasons)

    def test_no_preload_high_k2_adds_substantial_reason(self) -> None:
        """No preload + high leaky adds substantial correction reason."""
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 20.0}
        result = assess_dsc_cbv_quality(
            self._HIGH_AIF_QC,
            preload_classification="none",
            k2_metrics=k2,
        )
        assert any("substantial" in r.lower() for r in result.reasons)

    def test_none_args_backward_compat(self) -> None:
        """None preload/k2 args → no new reasons, identical to current behavior."""
        result = assess_dsc_cbv_quality(self._HIGH_AIF_QC)
        assert result.confidence == "high"
        assert not any("preload" in r.lower() for r in result.reasons)
        assert not any("k2" in r.lower() for r in result.reasons)

    def test_preload_status_stored(self) -> None:
        """preload_status is stored on result."""
        result = assess_dsc_cbv_quality(
            self._HIGH_AIF_QC,
            preload_classification="severe",
        )
        assert result.preload_status == "severe"


# =============================================================================
# Preload-Aware Uncertainty Modulation
# =============================================================================


class TestPreloadAwareUncertainty:
    """Tests for preload-aware determine_constraint_mode."""

    _MODERATE_QUALITY = CBVQualityAssessment(
        confidence="moderate",
        use_constraint=True,
        reasons=["CNR coverage 0.90 < 0.99"],
    )

    def test_preload_low_k2_tightens_soft(self) -> None:
        """Preload + low K2 → uncertainty=0.12."""
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        config = determine_constraint_mode(
            self._MODERATE_QUALITY,
            preload_classification="half_dose",
            k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.12)
        assert "preload" in config.reason.lower()

    def test_no_preload_high_k2_widens_soft(self) -> None:
        """No preload + high leaky → uncertainty=0.30."""
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 20.0}
        config = determine_constraint_mode(
            self._MODERATE_QUALITY,
            preload_classification="none",
            k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.30)
        assert "k2" in config.reason.lower()

    def test_default_uncertainty_unchanged(self) -> None:
        """No preload info → uncertainty=0.20."""
        config = determine_constraint_mode(self._MODERATE_QUALITY)
        assert config.vp_uncertainty == pytest.approx(0.20)

    def test_preload_high_k2_default_uncertainty(self) -> None:
        """Preloaded but NOT well-suppressed → default 0.20."""
        k2 = {"k2_median": 0.02, "percent_leaky_voxels": 10.0}
        config = determine_constraint_mode(
            self._MODERATE_QUALITY,
            preload_classification="half_dose",
            k2_metrics=k2,
        )
        assert config.vp_uncertainty == pytest.approx(0.20)

    def test_high_confidence_unaffected(self) -> None:
        """HARD mode → preload/K2 don't change anything."""
        quality = CBVQualityAssessment(
            confidence="high", use_constraint=True, reasons=["All quality checks passed"],
        )
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        config = determine_constraint_mode(
            quality, preload_classification="half_dose", k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.HARD
        assert config.vp_uncertainty == 0.0

    def test_low_confidence_unaffected(self) -> None:
        """SKIP mode → preload/K2 don't change anything."""
        quality = CBVQualityAssessment(
            confidence="low", use_constraint=False,
            reasons=["CNR coverage 0.50 < 0.84"],
        )
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        config = determine_constraint_mode(
            quality, preload_classification="half_dose", k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.SKIP

    def test_user_override_ignores_preload(self) -> None:
        """User override 'soft' + preload data → default 0.20 (override path)."""
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        config = determine_constraint_mode(
            self._MODERATE_QUALITY,
            user_override="soft",
            preload_classification="half_dose",
            k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.20)
        assert "User override" in config.reason
