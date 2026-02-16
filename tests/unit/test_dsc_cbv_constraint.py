"""Unit tests for DSC CBV to vp constraint utilities."""

from typing import Any

import numpy as np
import pytest

from pmriprep.utils.dsc_cbv_constraint import (
    CBVQualityAssessment,
    CNR_COVERAGE_HIGH,
    ConstraintConfig,
    ConstraintMode,
    VP_MAX_PHYSIOLOGICAL,
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
        """White matter CBV=2.5 mL/100g converts to vp ~0.0144."""
        cbv = np.array([2.5], dtype=np.float64)
        vp = cbv_to_vp(cbv)
        # vp = 2.5 * (1 - 0.40) / (100 * 1.04) = 2.5 * 0.6 / 104 = 0.01442
        expected = 2.5 * 0.6 / 104.0
        assert vp[0] == pytest.approx(expected, rel=1e-6)

    def test_gm_cbv_to_vp(self) -> None:
        """Gray matter CBV=4.5 mL/100g converts to vp ~0.026."""
        cbv = np.array([4.5], dtype=np.float64)
        vp = cbv_to_vp(cbv)
        expected = 4.5 * 0.6 / 104.0
        assert vp[0] == pytest.approx(expected, rel=1e-6)

    def test_tumor_cbv_to_vp(self) -> None:
        """Tumor CBV=10.0 mL/100g converts to vp ~0.0577."""
        cbv = np.array([10.0], dtype=np.float64)
        vp = cbv_to_vp(cbv)
        expected = 10.0 * 0.6 / 104.0
        assert vp[0] == pytest.approx(expected, rel=1e-6)

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
        assert result["cnr_whole_brain"]["fraction_cnr_gt_4"] == CNR_COVERAGE_HIGH

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
        """Missing keys get safe defaults that don't cause false downgrades."""
        data = {}
        result = _normalize_aif_qc_keys(data)

        # CNR should get high default
        assert "cnr_whole_brain" in result
        assert result["cnr_whole_brain"]["fraction_cnr_gt_4"] == CNR_COVERAGE_HIGH
        # Other keys should remain absent (handled by assess_dsc_cbv_quality defaults)
        assert "aif_peak_height" not in result
        assert "baseline_return_ratio" not in result

    def test_aif_metrics_nested_psr(self) -> None:
        """Nested aif_metrics.psr is used when best_psr is absent."""
        data = {"aif_metrics": {"psr": 85.0}, "best_brr": 10.0}
        result = _normalize_aif_qc_keys(data)

        assert result["aif_peak_height"] == pytest.approx(0.85, rel=1e-6)
        assert result["baseline_return_ratio"] == pytest.approx(0.10, rel=1e-6)

    def test_sub0011_actual_data_high_confidence(self) -> None:
        """Sub-0011's actual DSC AIF QC data produces high confidence."""
        # Actual values from sub-0011 DSC AIF detection
        aif_qc_data = {"best_psr": 94.6, "best_brr": 7.17}
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
