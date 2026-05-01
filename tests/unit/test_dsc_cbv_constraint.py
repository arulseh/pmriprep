"""Unit tests for DSC CBV to vp constraint utilities."""

from typing import Any

import numpy as np
import pytest

from pmriprep.utils.dce_constants import (
    NAWM_VP_REFERENCE,
    VP_MEDIAN_MAX,
    VP_MEDIAN_MIN,
)
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
    ncbv_to_vp_prior,
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
            "detection_method": "synthetic",
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
            "detection_method": "atlas_permissive",
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "moderate"

    def test_atlas_strict_no_downgrade(self) -> None:
        """atlas_strict AIF should not downgrade from detection level alone."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_method": "atlas_strict",
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "high"

    def test_single_candidate_downgrades_to_low(self) -> None:
        """1 AIF voxel (no averaging) should downgrade to low."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_method": "atlas_strict",
            "n_aif_voxels_selected": 1,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "low"

    def test_two_candidates_downgrades_to_moderate(self) -> None:
        """2 AIF voxels should downgrade to moderate."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_method": "atlas_strict",
            "n_aif_voxels_selected": 2,
        }
        result = assess_dsc_cbv_quality(aif_qc_data)
        assert result.confidence == "moderate"

    def test_detection_level_in_output(self) -> None:
        """detection_level and n_aif_voxels should appear in output."""
        aif_qc_data = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
            "detection_method": "atlas_relaxed",
            "n_aif_voxels_selected": 5,
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

    def test_high_quality_soft_uses_unified_sigma(self) -> None:
        """High confidence in auto mode emits SOFT at unified σ = 0.40 (v0.27.0+).

        DSC CBV and DCE vp measure different vascular compartments; HARD
        equality is an unsupported physics claim, so ``auto`` caps at SOFT.
        σ is unified at 0.40 across confidence tiers because the residual
        scale uncertainty is biological (Hct heterogeneity + compartment
        partitioning) and does not reduce with DSC SNR.
        """
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=["All quality checks passed"],
        )
        config = determine_constraint_mode(quality)
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.40)

    def test_moderate_quality_soft(self) -> None:
        """Moderate confidence produces soft constraint with widened sigma."""
        quality = CBVQualityAssessment(
            confidence="moderate",
            use_constraint=True,
            reasons=["CNR coverage 0.90 < 0.99"],
        )
        config = determine_constraint_mode(quality)
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.40)

    def test_low_quality_skip(self) -> None:
        """Low confidence produces skip mode."""
        quality = CBVQualityAssessment(
            confidence="low",
            use_constraint=False,
            reasons=["CNR coverage 0.50 < 0.84"],
        )
        config = determine_constraint_mode(quality)
        assert config.mode == ConstraintMode.SKIP

    def test_user_override_skip(self) -> None:
        """``skip`` user override is honored regardless of quality."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=["All quality checks passed"],
        )
        config = determine_constraint_mode(quality, user_override="skip")
        assert config.mode == ConstraintMode.SKIP

    def test_user_override_soft(self) -> None:
        """``soft`` user override forces SOFT even on low quality."""
        quality = CBVQualityAssessment(
            confidence="low",
            use_constraint=False,
            reasons=["CNR coverage 0.50 < 0.84"],
        )
        config = determine_constraint_mode(quality, user_override="soft")
        assert config.mode == ConstraintMode.SOFT


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

        # nCBV ≈ 1 (NAWM reference) → vp ≈ 0.02, squarely physiological.
        ncbv_data = np.ones((5, 5, 5), dtype=np.float64)
        mask_data = np.ones((5, 5, 5), dtype=np.uint8)
        affine = np.eye(4)

        ncbv_file = tmp_path / "ncbv.nii.gz"
        mask_file = tmp_path / "mask.nii.gz"
        nib.Nifti1Image(ncbv_data, affine).to_filename(str(ncbv_file))
        nib.Nifti1Image(mask_data, affine).to_filename(str(mask_file))

        aif_qc = {
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }

        result = prepare_vp_constraint_from_data(
            ncbv_file=str(ncbv_file),
            aif_qc_data=aif_qc,
            brain_mask_file=str(mask_file),
        )

        qc = result.qc_report
        assert "mode" in qc, "qc_report must have 'mode' key"
        assert "reason" in qc, "qc_report must have 'reason' key"
        assert "constraint_mode" not in qc, "old 'constraint_mode' key must not exist"
        assert "constraint_reason" not in qc, "old 'constraint_reason' key must not exist"
        assert qc["mode"] in ("soft", "skip")

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

    def test_preload_low_k2_uses_unified_sigma(self) -> None:
        """Preload + low K2 tier uses unified σ = 0.40 (v0.27.0+).

        The preload/K2 tiers still set their reason-string suffix for QC
        provenance, but σ no longer stratifies — biological uncertainty on
        vp dominates and is DSC-SNR-independent.
        """
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        config = determine_constraint_mode(
            self._MODERATE_QUALITY,
            preload_classification="half_dose",
            k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.40)
        assert "preload" in config.reason.lower()

    def test_no_preload_high_k2_uses_unified_sigma(self) -> None:
        """No preload + high leaky uses unified σ = 0.40 (v0.27.0+).

        Previously stratified to σ=0.50; now unified with default.
        """
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 20.0}
        config = determine_constraint_mode(
            self._MODERATE_QUALITY,
            preload_classification="none",
            k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.40)
        assert "k2" in config.reason.lower()

    def test_default_uncertainty_unchanged(self) -> None:
        """No preload info → SIGMA_SOFT_DEFAULT (0.40)."""
        config = determine_constraint_mode(self._MODERATE_QUALITY)
        assert config.vp_uncertainty == pytest.approx(0.40)

    def test_preload_high_k2_default_uncertainty(self) -> None:
        """Preloaded but NOT well-suppressed → default 0.40."""
        k2 = {"k2_median": 0.02, "percent_leaky_voxels": 10.0}
        config = determine_constraint_mode(
            self._MODERATE_QUALITY,
            preload_classification="half_dose",
            k2_metrics=k2,
        )
        assert config.vp_uncertainty == pytest.approx(0.40)

    def test_high_confidence_caps_at_soft(self) -> None:
        """High confidence in auto mode emits SOFT at unified σ = 0.40 (v0.27.0+), never HARD."""
        quality = CBVQualityAssessment(
            confidence="high", use_constraint=True, reasons=["All quality checks passed"],
        )
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        config = determine_constraint_mode(
            quality, preload_classification="half_dose", k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.40)

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
        """User override 'soft' + preload data → default 0.40 (override path)."""
        k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        config = determine_constraint_mode(
            self._MODERATE_QUALITY,
            user_override="soft",
            preload_classification="half_dose",
            k2_metrics=k2,
        )
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.40)
        assert "User override" in config.reason


# =============================================================================
# NAWM-normalized CBV → vp prior
# =============================================================================


class TestNcbvToVpPrior:
    """Tests for ncbv_to_vp_prior (NAWM-anchored vp prior)."""

    def test_nawm_value_yields_reference_vp(self) -> None:
        """nCBV=1 (NAWM reference) → vp = NAWM_VP_REFERENCE."""
        ncbv = np.array([1.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv)
        assert vp[0] == pytest.approx(NAWM_VP_REFERENCE)

    def test_tumor_elevation_proportional(self) -> None:
        """nCBV=3 (hypervascular tumor) → vp = 3 × NAWM_VP_REFERENCE."""
        ncbv = np.array([3.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv)
        assert vp[0] == pytest.approx(3.0 * NAWM_VP_REFERENCE)

    def test_negative_and_nan_return_nan(self) -> None:
        """Negative and NaN nCBV values produce NaN."""
        ncbv = np.array([-0.5, np.nan, 1.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv)
        assert np.isnan(vp[0])
        assert np.isnan(vp[1])
        assert np.isfinite(vp[2])

    def test_above_physiological_max_clipped_to_nan(self) -> None:
        """nCBV large enough to exceed VP_MAX_PHYSIOLOGICAL → NaN."""
        # With NAWM_VP_REFERENCE = 0.010 (v0.27.0+) and VP_MAX_PHYSIOLOGICAL
        # = 0.2, nCBV = 25 gives vp = 0.25 which is beyond the physiological
        # cap.
        ncbv = np.array([25.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv)
        assert np.isnan(vp[0])

    def test_custom_reference(self) -> None:
        """Custom nawm_vp_reference scales the prior accordingly."""
        ncbv = np.array([1.0, 2.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv, nawm_vp_reference=0.015)
        assert vp[0] == pytest.approx(0.015)
        assert vp[1] == pytest.approx(0.030)


# =============================================================================
# vp magnitude gate (undershoot / overshoot)
# =============================================================================


def _write_nifti(path: Any, data: np.ndarray) -> str:
    import nibabel as nib  # type: ignore[import-untyped]
    img = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    nib.save(img, str(path))
    return str(path)


def _synth_aif_qc() -> dict[str, Any]:
    return {
        "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995},
        "aif_peak_height": 0.4,
        "baseline_return_ratio": 0.15,
    }


class TestVpMagnitudeGate:
    """Tests for the vp median undershoot/overshoot gate."""

    def test_undershoot_downgrades_to_skip(self, tmp_path: Any) -> None:
        """vp median below VP_MEDIAN_MIN triggers SKIP + vp_map=None."""
        # nCBV median far below 1 so vp = 0.02*nCBV falls under VP_MEDIAN_MIN.
        ncbv = np.full((8, 8, 8), 0.1, dtype=np.float64)
        mask = np.ones((8, 8, 8), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            aif_qc_data=_synth_aif_qc(),
            brain_mask_file=mask_file,
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.vp_map is None
        assert "vp_median" in result.config.reason
        assert result.qc_report.get("magnitude_gate_triggered") is True

    def test_overshoot_downgrades_to_skip(self, tmp_path: Any) -> None:
        """vp median above VP_MEDIAN_MAX triggers SKIP + vp_map=None."""
        # nCBV median ≈ 8 → vp = 0.16, above VP_MEDIAN_MAX = 0.10.
        ncbv = np.full((8, 8, 8), 8.0, dtype=np.float64)
        mask = np.ones((8, 8, 8), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            aif_qc_data=_synth_aif_qc(),
            brain_mask_file=mask_file,
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.vp_map is None
        assert result.qc_report.get("magnitude_gate_triggered") is True

    def test_physiological_range_no_downgrade(self, tmp_path: Any) -> None:
        """vp median in physiological range preserves constraint mode."""
        # nCBV ≈ 1 → vp ≈ 0.02, squarely inside [VP_MEDIAN_MIN, VP_MEDIAN_MAX].
        ncbv = np.full((8, 8, 8), 1.0, dtype=np.float64)
        mask = np.ones((8, 8, 8), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            aif_qc_data=_synth_aif_qc(),
            brain_mask_file=mask_file,
        )

        assert result.config.mode != ConstraintMode.SKIP
        assert result.vp_map is not None
        assert result.qc_report.get("magnitude_gate_triggered") is not True
        assert VP_MEDIAN_MIN <= result.qc_report["vp_median"] <= VP_MEDIAN_MAX


# =============================================================================
# nCBV integration path through prepare_vp_constraint_from_data
# =============================================================================


class TestNcbvIntegration:
    """End-to-end tests confirming the production path uses ncbv_to_vp_prior."""

    def test_prepare_vp_constraint_uses_ncbv_to_vp_prior(
        self, tmp_path: Any
    ) -> None:
        """nCBV median 1.0 yields vp_map median ≈ NAWM_VP_REFERENCE (0.02)."""
        ncbv = np.full((6, 6, 6), 1.0, dtype=np.float64)
        mask = np.ones((6, 6, 6), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            aif_qc_data=_synth_aif_qc(),
            brain_mask_file=mask_file,
        )

        assert result.vp_map is not None
        vp_median = float(np.nanmedian(result.vp_map))
        assert vp_median == pytest.approx(NAWM_VP_REFERENCE, rel=1e-6)

    def test_missing_ncbv_file_emits_skip(self, tmp_path: Any) -> None:
        """ncbv_file=None yields SKIP with skip_reason='no_nawm_mask'."""
        mask = np.ones((4, 4, 4), dtype=np.float32)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=None,
            aif_qc_data=_synth_aif_qc(),
            brain_mask_file=mask_file,
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.vp_map is None
        assert result.qc_report.get("skip_reason") == "no_nawm_mask"


# =============================================================================
# Dose-aware CNR coverage gate (v0.25.2)
# =============================================================================


class TestDoseAwareCnrGate:
    """Dose-scaled CNR coverage thresholds + AIF-tier bypass."""

    @staticmethod
    def _qc(cnr_coverage: float, detection: str, peak: float = 0.4) -> dict[str, Any]:
        return {
            "cnr_whole_brain": {
                "fraction_cnr_gt_4": cnr_coverage,
                "mean_cnr": 10.0,  # above Rose minimum
            },
            "detection_method": detection,
            "aif_peak_height": peak,
            "baseline_return_ratio": 0.15,
        }

    def test_full_dose_thresholds_unchanged(self) -> None:
        """dose=1.0 × agent=0.1 (Gd-DOTA) → thresholds match v0.25.1."""
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction
        assert _bolus_gd_fraction(1.0, 0.1) == pytest.approx(1.0)

    def test_half_dose_gddota_half_thresholds(self) -> None:
        """dose=0.5 × agent=0.1 → eff=0.5; coverage 0.55 now passes SKIP."""
        # v0.25.1 thresholds: SKIP=0.70, MODERATE=0.84, HIGH=0.95
        # At eff=0.5, SKIP becomes 0.35. Coverage=0.55 clears it.
        result = assess_dsc_cbv_quality(
            aif_qc_data=self._qc(0.55, "atlas_permissive"),
            dose=0.5,
            agent_standard_dose=0.1,
        )
        # No low-confidence downgrade from CNR (above scaled SKIP=0.35).
        assert result.confidence != "low" or all(
            "CNR coverage" not in r for r in result.reasons
        )

    def test_half_dose_vueway_scaled(self) -> None:
        """dose=0.5 × agent=0.05 (Vueway half-dose) → eff=0.25."""
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction
        assert _bolus_gd_fraction(0.5, 0.05) == pytest.approx(0.25)

    def test_atlas_strict_bypasses_skip_at_low_cnr(self) -> None:
        """atlas_strict + CNR 0.60 → moderate (not low), even at full dose."""
        result = assess_dsc_cbv_quality(
            aif_qc_data=self._qc(0.60, "atlas_strict"),
            dose=1.0,
            agent_standard_dose=0.1,
        )
        # CNR 0.60 < SKIP 0.70 at full dose, but atlas_strict bypass
        # keeps confidence at moderate instead of low.
        assert result.confidence == "moderate"

    def test_atlas_permissive_still_skips_at_low_cnr(self) -> None:
        """atlas_permissive + CNR 0.60 at full dose → low (no bypass)."""
        result = assess_dsc_cbv_quality(
            aif_qc_data=self._qc(0.60, "atlas_permissive"),
            dose=1.0,
            agent_standard_dose=0.1,
        )
        assert result.confidence == "low"

    def test_double_dose_tightens_thresholds(self) -> None:
        """dose=2.0 → eff=2.0; coverage 0.80 (passes at full-dose) now SKIPs."""
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction
        # Double dose bumps SKIP threshold from 0.70 → 1.4 (clamped implicit at 1.0).
        # Sub with coverage 0.80 hits SKIP now where it cleared at full-dose.
        assert _bolus_gd_fraction(2.0, 0.1) == pytest.approx(2.0)

    def test_very_low_effective_dose_clamp(self) -> None:
        """dose=0.25 × agent=0.04 (gadoquatrane quarter) → eff=0.1."""
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction
        # SKIP threshold becomes 0.07 — gate essentially disabled for
        # very-low-dose protocols. Documents behavior; subjects pass trivially.
        assert _bolus_gd_fraction(0.25, 0.04) == pytest.approx(0.1)


class TestDetermineConstraintModeUnknownSkips:
    """v0.25.2: UNKNOWN preload short-circuits to SKIP regardless of quality."""

    def test_unknown_preload_skips_even_with_high_quality(self) -> None:
        """Preload=UNKNOWN → SKIP even if DSC quality signals are excellent."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=["All quality checks passed"],
        )
        config = determine_constraint_mode(
            quality,
            preload_classification="unknown",
        )
        assert config.mode == ConstraintMode.SKIP
        assert "UNKNOWN" in config.reason or "unavailable" in config.reason.lower()

    def test_unknown_preload_short_circuits_before_user_override(self) -> None:
        """Even with 'auto' override, UNKNOWN → SKIP."""
        quality = CBVQualityAssessment(
            confidence="moderate",
            use_constraint=True,
            reasons=["CNR coverage marginal"],
        )
        config = determine_constraint_mode(
            quality,
            user_override="auto",
            preload_classification="unknown",
        )
        assert config.mode == ConstraintMode.SKIP

    def test_explicit_user_override_skip_still_wins_over_unknown(self) -> None:
        """UNKNOWN auto path bypasses preload check, but explicit override is honored."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=["All quality checks passed"],
        )
        config = determine_constraint_mode(
            quality,
            user_override="soft",
            preload_classification="unknown",
        )
        assert config.mode == ConstraintMode.SOFT


class TestBolusGdFractionValidation:
    """_bolus_gd_fraction rejects invalid dose (W1 from v0.25.2 review)."""

    def test_zero_dose_raises(self) -> None:
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction
        with pytest.raises(ValueError, match="dose must be positive"):
            _bolus_gd_fraction(0.0, 0.1)

    def test_negative_dose_raises(self) -> None:
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction
        with pytest.raises(ValueError, match="dose must be positive"):
            _bolus_gd_fraction(-0.1, 0.1)

    def test_zero_agent_raises(self) -> None:
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction
        with pytest.raises(ValueError, match="agent_standard_dose"):
            _bolus_gd_fraction(1.0, 0.0)


class TestCnrBypassFallThrough:
    """atlas_strict bypass only overrides SKIP; MODERATE/HIGH tiers still apply
    (S3 from v0.25.2 review)."""

    @staticmethod
    def _qc(cnr_coverage: float, detection: str) -> dict[str, Any]:
        return {
            "cnr_whole_brain": {
                "fraction_cnr_gt_4": cnr_coverage,
                "mean_cnr": 10.0,
            },
            "detection_method": detection,
            "aif_peak_height": 0.4,
            "baseline_return_ratio": 0.15,
        }

    def test_atlas_strict_in_moderate_band_still_moderate(self) -> None:
        """CNR coverage between SKIP and MODERATE thresholds with atlas_strict
        → moderate (same outcome as the bypass, correctly falls through)."""
        # At full dose SKIP=0.70, MODERATE=0.84, HIGH=0.95.
        # CNR=0.80 is between SKIP and MODERATE, atlas_strict → moderate.
        result = assess_dsc_cbv_quality(
            aif_qc_data=self._qc(0.80, "atlas_strict"),
            dose=1.0,
            agent_standard_dose=0.1,
        )
        assert result.confidence == "moderate"

    def test_atlas_strict_in_high_band_still_high(self) -> None:
        """CNR coverage above HIGH with atlas_strict → high confidence preserved."""
        result = assess_dsc_cbv_quality(
            aif_qc_data=self._qc(0.97, "atlas_strict"),
            dose=1.0,
            agent_standard_dose=0.1,
        )
        assert result.confidence == "high"


class TestSigmaUnification:
    """v0.27.0: σ unified at 0.40 across all DSC-confidence + preload/K2 tiers.

    Per physics-expert 2026-04-19: residual scale uncertainty on vp is
    dominated by biological sources (Hct heterogeneity ~15% CoV, capillary-
    vs-total vessel partitioning ~20%) which are DSC-SNR-independent.
    Stratifying σ by DSC confidence confuses orthogonal uncertainties.
    """

    def test_all_confidence_tiers_emit_unified_sigma(self) -> None:
        """High/moderate/low confidence paths (where SOFT emits) all use σ = 0.40."""
        high = CBVQualityAssessment(confidence="high", use_constraint=True, reasons=[])
        mod = CBVQualityAssessment(confidence="moderate", use_constraint=True, reasons=[])

        cfg_high = determine_constraint_mode(high)
        cfg_mod = determine_constraint_mode(mod)

        assert cfg_high.mode == ConstraintMode.SOFT
        assert cfg_mod.mode == ConstraintMode.SOFT
        assert cfg_high.vp_uncertainty == pytest.approx(0.40)
        assert cfg_mod.vp_uncertainty == pytest.approx(0.40)

    def test_all_preload_k2_tiers_emit_unified_sigma(self) -> None:
        """Preload-suppressed / high-K2 / default moderate paths all use σ = 0.40."""
        mod = CBVQualityAssessment(confidence="moderate", use_constraint=True, reasons=[])
        low_k2 = {"k2_median": 0.002, "percent_leaky_voxels": 2.0}
        high_k2 = {"k2_median": 0.002, "percent_leaky_voxels": 20.0}

        cfg_preload = determine_constraint_mode(
            mod, preload_classification="half_dose", k2_metrics=low_k2,
        )
        cfg_highk2 = determine_constraint_mode(
            mod, preload_classification="none", k2_metrics=high_k2,
        )
        cfg_default = determine_constraint_mode(mod)

        assert cfg_preload.vp_uncertainty == pytest.approx(0.40)
        assert cfg_highk2.vp_uncertainty == pytest.approx(0.40)
        assert cfg_default.vp_uncertainty == pytest.approx(0.40)


class TestNAWMVpReferenceAnchor:
    """v0.27.0: NAWM_VP_REFERENCE = 0.010 from Larsson MRM 2009;62:1270.

    Regression guard — pinning the anchor value ensures any future retuning
    carries an explicit test update + citation.
    """

    def test_nawm_vp_reference_value(self) -> None:
        from pmriprep.utils.dce_constants import NAWM_VP_REFERENCE
        assert NAWM_VP_REFERENCE == 0.010

    def test_nawm_vp_reference_inside_median_bounds(self) -> None:
        """Anchor must sit inside [VP_MEDIAN_MIN, VP_MEDIAN_MAX] so normal
        NAWM-like nCBV inputs never trip the magnitude gate.
        """
        from pmriprep.utils.dce_constants import (
            NAWM_VP_REFERENCE, VP_MEDIAN_MIN, VP_MEDIAN_MAX,
        )
        assert VP_MEDIAN_MIN < NAWM_VP_REFERENCE < VP_MEDIAN_MAX


class TestSkipReasonConstants:
    """Shared constants prevent skip_reason drift across modules."""

    def test_constants_expose_expected_values(self) -> None:
        from pmriprep.utils.dsc_cbv_constraint import (
            NO_DSC_RUN_REASON_STRING,
            SKIP_REASON_NO_DSC_RUN,
            SKIP_REASON_NO_NAWM_MASK,
            SKIP_REASON_SYNTHETIC_AIF,
            SYNTHETIC_AIF_REASON_STRING,
        )
        assert SKIP_REASON_NO_DSC_RUN == "no_dsc_run"
        assert SKIP_REASON_NO_NAWM_MASK == "no_nawm_mask"
        assert SKIP_REASON_SYNTHETIC_AIF == "synthetic_aif"
        assert NO_DSC_RUN_REASON_STRING == "No DSC data provided"
        assert "synthetic" in SYNTHETIC_AIF_REASON_STRING.lower()


# =============================================================================
# Synthetic-AIF SKIP plumbing (v0.28.3)
# =============================================================================


class TestPrepareVpConstraintSyntheticAif:
    """`prepare_vp_constraint_from_data` early-SKIPs on synthetic AIF."""

    def test_skip_on_detection_level_synthetic(self, tmp_path) -> None:
        from pmriprep.utils.dsc_cbv_constraint import (
            prepare_vp_constraint_from_data, SKIP_REASON_SYNTHETIC_AIF,
        )
        # Brain mask file required by signature; synthetic branch
        # short-circuits before it's loaded, so a placeholder suffices.
        mask_path = tmp_path / "brain_mask.nii.gz"
        mask_path.write_text("")
        # ncbv_file=None would also hit the no-NAWM SKIP, but synthetic
        # SKIP must fire FIRST — that's the precedence being tested.
        result = prepare_vp_constraint_from_data(
            ncbv_file=None,
            aif_qc_data={"detection_method": "synthetic"},
            brain_mask_file=str(mask_path),
        )
        assert result.config.mode == ConstraintMode.SKIP
        assert result.qc_report["skip_reason"] == SKIP_REASON_SYNTHETIC_AIF

    def test_skip_on_cbv_auc_denominator_is_synthetic(self, tmp_path) -> None:
        """Legacy flag (``cbv_auc_denominator_is_synthetic``) also fires."""
        from pmriprep.utils.dsc_cbv_constraint import (
            prepare_vp_constraint_from_data, SKIP_REASON_SYNTHETIC_AIF,
        )
        mask_path = tmp_path / "brain_mask.nii.gz"
        mask_path.write_text("")
        result = prepare_vp_constraint_from_data(
            ncbv_file=None,
            aif_qc_data={"cbv_auc_denominator_is_synthetic": True},
            brain_mask_file=str(mask_path),
        )
        assert result.config.mode == ConstraintMode.SKIP
        assert result.qc_report["skip_reason"] == SKIP_REASON_SYNTHETIC_AIF

    def test_synthetic_short_circuits_before_missing_ncbv(
        self, tmp_path
    ) -> None:
        """Synthetic-AIF SKIP precedes the no-NAWM-mask SKIP — the reason
        in QC must be ``synthetic_aif``, not ``no_nawm_mask``."""
        from pmriprep.utils.dsc_cbv_constraint import (
            prepare_vp_constraint_from_data,
            SKIP_REASON_SYNTHETIC_AIF,
            SKIP_REASON_NO_NAWM_MASK,
        )
        mask_path = tmp_path / "brain_mask.nii.gz"
        mask_path.write_text("")
        result = prepare_vp_constraint_from_data(
            ncbv_file=None,  # would fire NO_NAWM_MASK if synthetic missed
            aif_qc_data={"detection_method": "synthetic"},
            brain_mask_file=str(mask_path),
        )
        assert result.qc_report["skip_reason"] == SKIP_REASON_SYNTHETIC_AIF
        assert result.qc_report["skip_reason"] != SKIP_REASON_NO_NAWM_MASK

    def test_measured_aif_does_not_short_circuit(self, tmp_path) -> None:
        """Measured AIF + missing ncbv → normal NO_NAWM_MASK SKIP, not synthetic."""
        from pmriprep.utils.dsc_cbv_constraint import (
            prepare_vp_constraint_from_data, SKIP_REASON_NO_NAWM_MASK,
        )
        mask_path = tmp_path / "brain_mask.nii.gz"
        mask_path.write_text("")
        result = prepare_vp_constraint_from_data(
            ncbv_file=None,
            aif_qc_data={"detection_method": "atlas_relaxed"},
            brain_mask_file=str(mask_path),
        )
        assert result.qc_report["skip_reason"] == SKIP_REASON_NO_NAWM_MASK


class TestDetermineConstraintModeSyntheticOverride:
    """`determine_constraint_mode` forces SKIP when AIF is synthetic."""

    def _make_quality(
        self, detection_level: str = "measured"
    ) -> CBVQualityAssessment:
        return CBVQualityAssessment(
            cbv_median=1.0,
            confidence="high",
            use_constraint=True,
            detection_level=detection_level,
        )

    def test_synthetic_overrides_user_soft(self) -> None:
        quality = self._make_quality(detection_level="synthetic")
        config = determine_constraint_mode(
            dsc_quality=quality,
            user_override="soft",
        )
        assert config.mode == ConstraintMode.SKIP

    def test_synthetic_auto_still_skips(self) -> None:
        quality = self._make_quality(detection_level="synthetic")
        config = determine_constraint_mode(
            dsc_quality=quality,
            user_override="auto",
        )
        assert config.mode == ConstraintMode.SKIP

    def test_measured_soft_honored_when_aif_real(self) -> None:
        quality = self._make_quality(detection_level="atlas_strict")
        config = determine_constraint_mode(
            dsc_quality=quality,
            user_override="soft",
        )
        # Real AIF → user override honored.
        assert config.mode == ConstraintMode.SOFT
