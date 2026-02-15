"""Unit tests for DSC CBV to vp constraint utilities."""

import numpy as np
import pytest

from pmriprep.utils.dsc_cbv_constraint import (
    CBVQualityAssessment,
    ConstraintConfig,
    ConstraintMode,
    VP_MAX_PHYSIOLOGICAL,
    assess_dsc_cbv_quality,
    cbv_to_vp,
    determine_constraint_mode,
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
