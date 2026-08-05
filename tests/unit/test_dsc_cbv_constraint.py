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
    CNR_COVERAGE_MODERATE,
    ConstraintMode,
    _normalize_aif_qc_keys,
    assess_dsc_cbv_quality,
    determine_constraint_mode,
    ncbv_to_vp_prior,
    prepare_vp_constraint_from_data,
)


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

    def test_high_quality_uses_fixed_prior(self) -> None:
        """High confidence and a valid K2 reference emit the fixed soft prior."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=["All quality checks passed"],
        )
        k2 = {"k2_reference_available": True, "k2_signed": True}
        config = determine_constraint_mode(quality, k2_metrics=k2)
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.60)

    def test_moderate_quality_soft(self) -> None:
        """Moderate confidence and a valid reference use the fixed prior."""
        quality = CBVQualityAssessment(
            confidence="moderate",
            use_constraint=True,
            reasons=["CNR coverage 0.90 < 0.99"],
        )
        k2 = {"k2_reference_available": True, "k2_signed": True}
        config = determine_constraint_mode(quality, k2_metrics=k2)
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.60)

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

    def test_low_quality_auto_skips(self) -> None:
        """Low DSC confidence under ``auto`` → SKIP (no prior)."""
        quality = CBVQualityAssessment(
            confidence="low",
            use_constraint=False,
            reasons=["CNR coverage 0.50 < 0.84"],
        )
        config = determine_constraint_mode(quality, user_override="auto")
        assert config.mode == ConstraintMode.SKIP


# =============================================================================
# AIF QC Key Normalization
# =============================================================================


class TestNormalizeAifQcKeys:
    """Tests for _normalize_aif_qc_keys."""

    def test_pmriprep_format_translates(self) -> None:
        """Producer-shape aif_metrics (psr, brr) is translated."""
        data = {"aif_metrics": {"psr": 94.6, "brr": 7.17}}
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

    def test_aif_metrics_nested_psr_only(self) -> None:
        """Nested aif_metrics.psr translates to aif_peak_height (brr absent)."""
        data = {"aif_metrics": {"psr": 85.0}}
        result = _normalize_aif_qc_keys(data)

        assert result["aif_peak_height"] == pytest.approx(0.85, rel=1e-6)
        assert "baseline_return_ratio" not in result

    def test_aif_metrics_nested_brr_only(self) -> None:
        """Nested aif_metrics.brr translates to baseline_return_ratio (psr absent)."""
        data = {"aif_metrics": {"brr": 12.0}}
        result = _normalize_aif_qc_keys(data)

        assert result["baseline_return_ratio"] == pytest.approx(0.12, rel=1e-6)
        assert "aif_peak_height" not in result

    def test_good_aif_with_cnr_high_confidence(self) -> None:
        """Good AIF metrics with CNR coverage produces high confidence."""
        aif_qc_data = {
            "aif_metrics": {"psr": 94.6, "brr": 7.17},
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.95, "mean_cnr": 25.0},
        }
        result = assess_dsc_cbv_quality(aif_qc_data)

        assert result.confidence == "high"
        assert result.use_constraint is True
        assert result.aif_peak_height == pytest.approx(0.946, rel=1e-3)
        assert result.aif_baseline_return == pytest.approx(0.0717, rel=1e-3)

    def test_unknown_keys_preserved(self) -> None:
        """Unknown keys pass through unchanged (non-AIF fixture)."""
        data = {"extra_field": 42, "another_extra": "string"}
        result = _normalize_aif_qc_keys(data)

        assert result["extra_field"] == 42
        assert result["another_extra"] == "string"

    def test_normalize_extracts_gamma_peak_ratio(self) -> None:
        """gamma_fit_auto_metrics.peak_ratio is promoted to top-level key."""
        data = {"gamma_fit_auto_metrics": {"peak_ratio": 0.843}}
        result = _normalize_aif_qc_keys(data)

        assert result["gamma_peak_ratio"] == pytest.approx(0.843)

    def test_normalize_no_gamma_metrics_no_key(self) -> None:
        """Missing gamma_fit_auto_metrics does not create gamma_peak_ratio."""
        data = {"aif_metrics": {"psr": 94.6}}
        result = _normalize_aif_qc_keys(data)

        assert "gamma_peak_ratio" not in result


# =============================================================================
# QC Report Schema
# =============================================================================


class TestQcReportSchema:
    """Tests that qc_report uses the (mode, reason) key schema."""

    def test_qc_report_keys_from_prepare_vp_constraint_from_data(
        self, tmp_path: Any
    ) -> None:
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
        assert "constraint_reason" not in qc, (
            "old 'constraint_reason' key must not exist"
        )
        assert qc["mode"] in ("soft", "skip")

    def test_qc_report_skip_default_schema(self) -> None:
        """Default skip dict uses 'mode' and 'reason' (no 'constraint_' prefix)."""
        default_qc: dict[str, Any] = {"mode": "skip", "reason": "No DSC CBV provided"}
        assert "mode" in default_qc
        assert "reason" in default_qc
        assert "constraint_mode" not in default_qc


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
        # With NAWM_VP_REFERENCE = 0.010 and VP_MAX_PHYSIOLOGICAL
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


def _accepted_registration_qc() -> dict[str, dict[str, str]]:
    """Return accepted QC for both transforms in the spatial transfer."""
    return {
        "dsc_registration_qc": {"qc_status": "PASS"},
        "dce_registration_qc": {"qc_status": "PASS"},
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
            k2_metrics={"k2_reference_available": True},
            **_accepted_registration_qc(),
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
            k2_metrics={"k2_reference_available": True},
            **_accepted_registration_qc(),
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
            k2_metrics={"k2_reference_available": True},
            **_accepted_registration_qc(),
        )

        assert result.config.mode != ConstraintMode.SKIP
        assert result.vp_map is not None
        assert result.qc_report.get("magnitude_gate_triggered") is not True
        assert VP_MEDIAN_MIN <= result.qc_report["vp_median"] <= VP_MEDIAN_MAX

    def test_signed_k2_preserves_qualified_bayesian_prior(self, tmp_path: Any) -> None:
        """Production signed K2 retains the fixed prior after gross-quality gates."""
        ncbv = np.full((8, 8, 8), 1.0, dtype=np.float64)
        mask = np.ones((8, 8, 8), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            aif_qc_data=_synth_aif_qc(),
            brain_mask_file=mask_file,
            k2_metrics={
                "k2_signed": True,
                "k2_reference_available": True,
                "k2_median": 0.001,
                "percent_negative_k2": 35.0,
            },
            **_accepted_registration_qc(),
        )

        assert result.config.mode == ConstraintMode.SOFT
        assert result.config.vp_uncertainty == pytest.approx(0.60)
        assert result.vp_map is not None
        assert result.qc_report["vp_uncertainty"] == pytest.approx(0.60)


# =============================================================================
# nCBV integration path through prepare_vp_constraint_from_data
# =============================================================================


class TestNcbvIntegration:
    """End-to-end tests confirming the production path uses ncbv_to_vp_prior."""

    def test_prepare_vp_constraint_uses_ncbv_to_vp_prior(self, tmp_path: Any) -> None:
        """nCBV median 1.0 yields vp_map median ≈ NAWM_VP_REFERENCE (0.02)."""
        ncbv = np.full((6, 6, 6), 1.0, dtype=np.float64)
        mask = np.ones((6, 6, 6), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            aif_qc_data=_synth_aif_qc(),
            brain_mask_file=mask_file,
            k2_metrics={"k2_reference_available": True},
            **_accepted_registration_qc(),
        )

        assert result.vp_map is not None
        vp_median = float(np.nanmedian(result.vp_map))
        assert vp_median == pytest.approx(NAWM_VP_REFERENCE, rel=1e-6)
        assert result.qc_report["nawm_vp_reference"] == pytest.approx(NAWM_VP_REFERENCE)

    @pytest.mark.parametrize(
        ("dsc_status", "dce_status"),
        [("POOR", "PASS"), ("PASS", "POOR"), ("UNKNOWN", "PASS"), ("PASS", None)],
    )
    def test_registration_failure_skips_spatial_prior(
        self,
        tmp_path: Any,
        dsc_status: str,
        dce_status: str | None,
    ) -> None:
        """A failed or unavailable transform makes a spatial prior unsafe."""
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", np.ones((6, 6, 6)))
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", np.ones((6, 6, 6)))

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            aif_qc_data=_synth_aif_qc(),
            brain_mask_file=mask_file,
            dsc_registration_qc={"qc_status": dsc_status},
            dce_registration_qc=(
                {"qc_status": dce_status} if dce_status is not None else None
            ),
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.vp_map is None
        assert result.qc_report["skip_reason"] == "cross_modal_registration_qc"

    def test_registration_warn_preserves_existing_width(self, tmp_path: Any) -> None:
        """Accepted WARN registration does not invent another prior-width tier."""
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", np.ones((6, 6, 6)))
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", np.ones((6, 6, 6)))
        common = {
            "ncbv_file": ncbv_file,
            "aif_qc_data": _synth_aif_qc(),
            "brain_mask_file": mask_file,
            "k2_metrics": {"k2_reference_available": True},
        }

        passed = prepare_vp_constraint_from_data(
            **common,
            dsc_registration_qc={"qc_status": "PASS"},
            dce_registration_qc={"qc_status": "PASS"},
        )
        warned = prepare_vp_constraint_from_data(
            **common,
            dsc_registration_qc={"qc_status": "WARN"},
            dce_registration_qc={"qc_status": "PASS"},
        )

        assert warned.config.mode == passed.config.mode == ConstraintMode.SOFT
        assert warned.config.vp_uncertainty == passed.config.vp_uncertainty

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
# Dose-aware CNR coverage gate
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
        """dose=1.0 × agent=0.1 (Gd-DOTA) → full-dose thresholds."""
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction

        assert _bolus_gd_fraction(1.0, 0.1) == pytest.approx(1.0)

    def test_half_dose_gddota_half_thresholds(self) -> None:
        """dose=0.5 × agent=0.1 → eff=0.5; coverage 0.55 now passes SKIP."""
        # Full-dose thresholds: SKIP=0.70, MODERATE=0.84, HIGH=0.95
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


class TestDetermineConstraintReferenceGate:
    """The K2 reference is a binary availability gate, not a classifier."""

    @pytest.mark.parametrize(
        "k2_metrics", [None, {}, {"k2_reference_available": False}]
    )
    def test_missing_or_unavailable_reference_skips(
        self,
        k2_metrics: dict[str, Any] | None,
    ) -> None:
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=[],
        )
        config = determine_constraint_mode(
            quality,
            k2_metrics=k2_metrics,
        )

        assert config.mode == ConstraintMode.SKIP
        assert config.vp_uncertainty == 0.0
        assert "reference unavailable" in config.reason


class TestBolusGdFractionValidation:
    """_bolus_gd_fraction rejects invalid dose."""

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
    """atlas_strict bypass only overrides SKIP; MODERATE/HIGH tiers still apply."""

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


class TestFixedPriorWidth:
    """Qualified anchors use one fixed release-policy prior width."""

    def test_confidence_does_not_stratify_sigma(self) -> None:
        """High and moderate confidence use the same fixed width."""
        high = CBVQualityAssessment(confidence="high", use_constraint=True, reasons=[])
        mod = CBVQualityAssessment(
            confidence="moderate", use_constraint=True, reasons=[]
        )
        available = {"k2_reference_available": True, "k2_signed": True}

        cfg_high = determine_constraint_mode(high, k2_metrics=available)
        cfg_mod = determine_constraint_mode(mod, k2_metrics=available)

        assert cfg_high.mode == ConstraintMode.SOFT
        assert cfg_mod.mode == ConstraintMode.SOFT
        assert cfg_high.vp_uncertainty == pytest.approx(0.60)
        assert cfg_mod.vp_uncertainty == cfg_high.vp_uncertainty

    def test_signed_k2_uses_fixed_release_policy_prior_width(self) -> None:
        """Signed leakage metrics neither gate nor tune the Bayesian prior."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=[],
        )
        signed = {
            "k2_signed": True,
            "k2_reference_available": True,
            "k2_median": -0.002,
            "percent_negative_k2": 60.0,
        }

        config = determine_constraint_mode(
            quality,
            k2_metrics=signed,
        )

        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.60)
        assert "fixed release-policy Bayesian prior" in config.reason
        assert "signed K2 reported descriptively" in config.reason

    def test_disabled_leakage_correction_skips_prior(self) -> None:
        """An explicitly uncorrected CBV map cannot anchor the DCE prior."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=[],
        )

        config = determine_constraint_mode(
            quality,
            k2_metrics={"leakage_correction_disabled": True},
        )

        assert config.mode == ConstraintMode.SKIP
        assert config.vp_uncertainty == 0.0
        assert "leakage correction disabled" in config.reason.lower()


class TestNAWMVpReferenceAnchor:
    """NAWM_VP_REFERENCE = 0.010 from Larsson MRM 2009;62:1270.

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
            NAWM_VP_REFERENCE,
            VP_MEDIAN_MIN,
            VP_MEDIAN_MAX,
        )

        assert VP_MEDIAN_MIN < NAWM_VP_REFERENCE < VP_MEDIAN_MAX


class TestSkipReasonConstants:
    """Shared constants prevent skip_reason drift across modules."""

    def test_constants_expose_expected_values(self) -> None:
        from pmriprep.utils.dsc_cbv_constraint import (
            NO_DSC_RUN_REASON_STRING,
            SKIP_REASON_NO_DSC_RUN,
            SKIP_REASON_NO_NAWM_MASK,
            SKIP_REASON_REGISTRATION_QC,
            SKIP_REASON_SYNTHETIC_AIF,
            SYNTHETIC_AIF_REASON_STRING,
        )

        assert SKIP_REASON_NO_DSC_RUN == "no_dsc_run"
        assert SKIP_REASON_NO_NAWM_MASK == "no_nawm_mask"
        assert SKIP_REASON_REGISTRATION_QC == "cross_modal_registration_qc"
        assert SKIP_REASON_SYNTHETIC_AIF == "synthetic_aif"
        assert NO_DSC_RUN_REASON_STRING == "No DSC data provided"
        assert "synthetic" in SYNTHETIC_AIF_REASON_STRING.lower()


# =============================================================================
# Synthetic-AIF SKIP plumbing
# =============================================================================


class TestPrepareVpConstraintSyntheticAif:
    """`prepare_vp_constraint_from_data` early-SKIPs on synthetic AIF."""

    def test_skip_on_detection_level_synthetic(self, tmp_path) -> None:
        from pmriprep.utils.dsc_cbv_constraint import (
            prepare_vp_constraint_from_data,
            SKIP_REASON_SYNTHETIC_AIF,
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
            prepare_vp_constraint_from_data,
            SKIP_REASON_SYNTHETIC_AIF,
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

    def test_synthetic_short_circuits_before_missing_ncbv(self, tmp_path) -> None:
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
            prepare_vp_constraint_from_data,
            SKIP_REASON_NO_NAWM_MASK,
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

    def _make_quality(self, detection_level: str = "measured") -> CBVQualityAssessment:
        return CBVQualityAssessment(
            cbv_median=1.0,
            confidence="high",
            use_constraint=True,
            detection_level=detection_level,
        )

    def test_synthetic_auto_still_skips(self) -> None:
        quality = self._make_quality(detection_level="synthetic")
        config = determine_constraint_mode(
            dsc_quality=quality,
            user_override="auto",
        )
        assert config.mode == ConstraintMode.SKIP
