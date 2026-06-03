"""Unit tests for DCE signal quality QC gating.

Tests for peak enhancement and baseline tSNR gating in the DCE QC
recommendation system.
"""

import pytest

from pmriprep.utils.dce_constants import (
    QC_B1_MEAN_HIGH,
    QC_B1_MEAN_LOW,
    QC_DCE_BASELINE_TSNR_MIN,
    QC_DCE_BASELINE_TSNR_WARN,
    QC_PEAK_ENHANCEMENT_MIN,
    QC_PEAK_ENHANCEMENT_WARN,
)
from pmriprep.utils.qc_standards import QCAction


class TestDCESignalQualityConstants:
    """Verify DCE signal quality constants are set correctly."""

    def test_peak_enhancement_thresholds(self) -> None:
        """Peak enhancement thresholds match physics-validated values."""
        assert QC_PEAK_ENHANCEMENT_MIN == 5.0
        assert QC_PEAK_ENHANCEMENT_WARN == 20.0

    def test_baseline_tsnr_thresholds(self) -> None:
        """Baseline tSNR thresholds match physics-validated values."""
        assert QC_DCE_BASELINE_TSNR_MIN == 3.0
        assert QC_DCE_BASELINE_TSNR_WARN == 8.0

    def test_b1_bias_thresholds(self) -> None:
        """B1+ bias thresholds match 1.5T extreme values."""
        assert QC_B1_MEAN_LOW == 0.7
        assert QC_B1_MEAN_HIGH == 1.3


class TestPeakEnhancementGating:
    """Tests for peak enhancement QC gating."""

    @staticmethod
    def _build_qc_metrics(peak_enh: float | None, aif_source: str = "detected") -> dict:
        """Build minimal QC metrics dict for testing."""
        return {
            "preprocessing": {
                "peak_enh_pct": peak_enh,
            },
            "aif_detection": {
                "aif_source": aif_source,
            },
            "pk_modeling": {},
        }

    def test_low_enhancement_exclude(self) -> None:
        """Peak enhancement < 5% triggers EXCLUDE."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=3.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.EXCLUDE
        assert "critically low" in rec["reason"].lower()
        assert "3.0%" in rec["reason"]

    def test_marginal_enhancement_review(self) -> None:
        """Peak enhancement 5-20% triggers REVIEW."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=12.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.REVIEW
        assert "marginal" in rec["reason"].lower()

    def test_good_enhancement_no_gating(self) -> None:
        """Peak enhancement >= 20% does not trigger signal quality gating."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=35.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        # Should be INCLUDE (no issues flagged)
        assert rec["action"] == QCAction.INCLUDE

    def test_population_aif_warning(self) -> None:
        """Marginal enhancement + population AIF adds injection verification warning."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=15.0, aif_source="population")
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.REVIEW
        # The warning about population AIF should be present
        assert "population" in rec["reason"].lower() or "injection" in rec["reason"].lower()


class TestBaselineTSNRGating:
    """Tests for baseline tSNR QC gating."""

    @staticmethod
    def _build_qc_metrics(baseline_tsnr: float | None) -> dict:
        """Build minimal QC metrics dict for testing."""
        return {
            "preprocessing": {
                "baseline_tsnr": baseline_tsnr,
            },
            "aif_detection": {},
            "pk_modeling": {},
        }

    def test_low_tsnr_exclude(self) -> None:
        """Baseline tSNR < 3 triggers EXCLUDE."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(baseline_tsnr=2.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.EXCLUDE
        assert "critically low" in rec["reason"].lower()
        assert "2.0" in rec["reason"]

    def test_marginal_tsnr_review(self) -> None:
        """Baseline tSNR 3-8 triggers REVIEW."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(baseline_tsnr=5.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.REVIEW
        assert "marginal" in rec["reason"].lower()

    def test_good_tsnr_no_gating(self) -> None:
        """Baseline tSNR >= 8 does not trigger signal quality gating."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(baseline_tsnr=15.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.INCLUDE


class TestCombinedGating:
    """Tests for combined signal quality metrics."""

    @staticmethod
    def _build_qc_metrics(peak_enh: float | None, baseline_tsnr: float | None) -> dict:
        """Build QC metrics dict with both signal quality metrics."""
        return {
            "preprocessing": {
                "peak_enh_pct": peak_enh,
                "baseline_tsnr": baseline_tsnr,
            },
            "aif_detection": {},
            "pk_modeling": {},
        }

    def test_both_critical_exclude(self) -> None:
        """Both metrics critically low results in EXCLUDE."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=2.0, baseline_tsnr=1.5)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.EXCLUDE

    def test_one_critical_one_marginal_exclude(self) -> None:
        """One critical, one marginal results in EXCLUDE (worst-of)."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=10.0, baseline_tsnr=1.5)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.EXCLUDE

    def test_both_marginal_review(self) -> None:
        """Both metrics marginal results in REVIEW."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=12.0, baseline_tsnr=5.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.REVIEW

    def test_both_good_include(self) -> None:
        """Both metrics good results in INCLUDE."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=40.0, baseline_tsnr=20.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.INCLUDE

    def test_missing_metrics_no_crash(self) -> None:
        """Missing signal quality metrics does not crash."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=None, baseline_tsnr=None)
        result = recommend_dce_qc(qc)

        # Should still produce a recommendation (INCLUDE if no issues)
        assert "recommendation" in result
        assert "action" in result["recommendation"]

    def test_nan_baseline_tsnr_no_gating(self) -> None:
        """NaN baseline_tsnr (from single-frame baseline) does not trigger gating."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(peak_enh=50.0, baseline_tsnr=float("nan"))
        result = recommend_dce_qc(qc)

        # NaN should not trigger EXCLUDE or REVIEW for tSNR
        rec = result["recommendation"]
        assert rec["action"] == QCAction.INCLUDE


class TestB1BiasGating:
    """Tests for B1+ bias QC gating."""

    @staticmethod
    def _build_qc_metrics(mean_b1: float | None) -> dict:
        """Build minimal QC metrics dict for testing."""
        return {
            "preprocessing": {},
            "aif_detection": {},
            "pk_modeling": {},
            "b1_mapping": {
                "mean_b1": mean_b1,
            },
        }

    def test_b1_low_triggers_review(self) -> None:
        """Mean B1+ < 0.7 triggers REVIEW."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(mean_b1=0.65)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.REVIEW
        assert "B1+" in rec["reason"]

    def test_b1_high_triggers_review(self) -> None:
        """Mean B1+ > 1.3 triggers REVIEW."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(mean_b1=1.4)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.REVIEW
        assert "B1+" in rec["reason"]

    def test_b1_normal_no_gating(self) -> None:
        """Mean B1+ within [0.7, 1.3] does not trigger gating."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(mean_b1=1.0)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.INCLUDE

    def test_b1_at_boundary_low_no_gating(self) -> None:
        """Mean B1+ exactly at 0.7 does not trigger gating."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(mean_b1=0.7)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.INCLUDE

    def test_b1_at_boundary_high_no_gating(self) -> None:
        """Mean B1+ exactly at 1.3 does not trigger gating."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(mean_b1=1.3)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.INCLUDE

    def test_b1_missing_no_crash(self) -> None:
        """Missing B1+ metrics does not crash."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(mean_b1=None)
        result = recommend_dce_qc(qc)

        assert "recommendation" in result
        assert result["recommendation"]["action"] == QCAction.INCLUDE

    def test_b1_mapping_absent_no_crash(self) -> None:
        """Entirely missing b1_mapping section does not crash."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = {
            "preprocessing": {},
            "aif_detection": {},
            "pk_modeling": {},
            # b1_mapping key absent entirely
        }
        result = recommend_dce_qc(qc)

        assert "recommendation" in result
        assert result["recommendation"]["action"] == QCAction.INCLUDE
