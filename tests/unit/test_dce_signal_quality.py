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
        """Peak enhancement thresholds match configured policy values."""
        assert QC_PEAK_ENHANCEMENT_MIN == 5.0
        assert QC_PEAK_ENHANCEMENT_WARN == 20.0

    def test_baseline_tsnr_thresholds(self) -> None:
        """Baseline tSNR thresholds match configured policy values."""
        assert QC_DCE_BASELINE_TSNR_MIN == 3.0
        assert QC_DCE_BASELINE_TSNR_WARN == 8.0

    def test_b1_bias_thresholds(self) -> None:
        """B1+ bias thresholds match 1.5T extreme values."""
        assert QC_B1_MEAN_LOW == 0.7
        assert QC_B1_MEAN_HIGH == 1.3


def test_two_cxm_suitability_warning_requests_review_not_exclusion() -> None:
    """Model suitability warnings feed the overall non-fatal recommendation."""
    from pmriprep.utils.qc_standards import recommend_dce_qc

    result = recommend_dce_qc(
        {
            "pk_modeling": {},
            "model_suitability": {
                "2cxm": {
                    "status": "WARN",
                    "reasons": [
                        "temporal_sampling_outside_auto_2cxm_eligibility"
                    ],
                }
            },
        }
    )

    assert result["recommendation"]["action"] == QCAction.REVIEW
    assert "2CXM suitability warning" in result["recommendation"]["reason"]


def test_measured_reduced_aif_is_distinct_from_population_fallback() -> None:
    """Reader-facing recommendations distinguish confidence from AIF source."""
    from pmriprep.utils.qc_standards import recommend_dce_qc

    measured = recommend_dce_qc({
        "aif_detection": {
            "aif_source": "detected",
            "aif_confidence": "reduced",
        },
    })["recommendation"]
    population = recommend_dce_qc({
        "aif_detection": {
            "aif_source": "population",
            "aif_confidence": "low",
        },
    })["recommendation"]

    assert "Measured AIF accepted with reduced confidence" in measured["reason"]
    assert "Population AIF used" not in measured["reason"]
    assert "Population AIF used" in population["reason"]
    assert "reduced confidence" not in population["reason"]


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


class TestBaselineSampleCountGating:
    """Tests for an inadequate pre-contrast sample count."""

    def test_early_bolus_fallback_excludes_quantitative_exam(self) -> None:
        """A fallback keeps derivatives but forces an EXCLUDE recommendation."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = {
            "preprocessing": {
                "baseline_status": "POOR",
                "baseline_fallback_applied": True,
                "detected_bolus_frame": 2,
                "baseline_minimum_frames": 5,
            },
            "aif_detection": {"aif_confidence": "standard"},
            "pk_modeling": {"median_residual": 0.01},
        }

        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.EXCLUDE
        assert "Inadequate pre-contrast DCE baseline" in rec["reason"]
        assert "detected frame 2" in rec["reason"]
        assert "minimum 5" in rec["reason"]
        assert "PK maps are unreliable" in rec["reason"]

    def test_normal_baseline_does_not_trigger_exclusion(self) -> None:
        """A normal detector result leaves the recommendation unchanged."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        result = recommend_dce_qc({
            "preprocessing": {"baseline_fallback_applied": False},
        })

        assert result["recommendation"]["action"] == QCAction.INCLUDE


class TestMotionAvailabilityGating:
    """Tests for unavailable motion summary QC."""

    def test_unavailable_motion_qc_requires_review(self) -> None:
        """Unknown motion is REVIEW, not a default INCLUDE."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        result = recommend_dce_qc({
            "preprocessing": {
                "motion_status": "UNKNOWN",
                "motion_qc_available": False,
            },
        })

        rec = result["recommendation"]
        assert rec["action"] == QCAction.REVIEW
        assert "Motion QC unavailable" in rec["reason"]


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
        assert result["recommendation"]["action"] == QCAction.REVIEW

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
        assert result["recommendation"]["action"] == QCAction.REVIEW


def test_displayed_tsnr_uses_only_precontrast_frames(tmp_path) -> None:
    """Enhancement dynamics do not enter the displayed DCE tSNR denominator."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")

    from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

    curve = np.array([100.0, 102.0, 98.0, 150.0, 200.0])
    signal = np.broadcast_to(curve, (3, 3, 3, curve.size)).copy()
    mask = np.ones((3, 3, 3), dtype=bool)

    _, stats = generate_preprocessing_qc_plot(
        signal,
        mask,
        baseline_frames=3,
        bolus_arrival_frame=3,
        volume_tr_s=4.0,
        output_path=str(tmp_path / "dce_qc.png"),
    )

    assert stats["baseline_tsnr"] == pytest.approx(50.0)


def test_preprocessing_qc_omits_bolus_contaminated_cov(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The QC figure must not label whole-series enhancement variance as CoV."""
    import matplotlib
    import numpy as np
    from matplotlib.axes import Axes

    matplotlib.use("Agg")

    from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

    titles: list[str] = []
    original_set_title = Axes.set_title

    def _record_title(self, label, *args, **kwargs):
        titles.append(str(label))
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", _record_title)
    curve = np.array([100.0, 101.0, 99.0, 140.0, 180.0])
    signal = np.broadcast_to(curve, (3, 3, 3, curve.size)).copy()

    _, stats = generate_preprocessing_qc_plot(
        signal,
        np.ones((3, 3, 3), dtype=bool),
        baseline_frames=3,
        bolus_arrival_frame=3,
        volume_tr_s=4.0,
        output_path=str(tmp_path / "dce_qc.png"),
    )

    assert "Baseline tSNR Map" in titles
    assert "Enhancement CNR (frame 4)" in titles
    assert not any("Coefficient of Variation" in title for title in titles)
    assert stats["mean_cnr"] == pytest.approx(80.0)


def test_preprocessing_mean_cnr_averages_voxelwise_noise_ratios(tmp_path) -> None:
    """Whole-brain CNR is a mean of voxelwise ratios, not a product of means."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")

    from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

    signal = np.array(
        [
            [[[99.0, 100.0, 101.0, 110.0]]],
            [[[90.0, 100.0, 110.0, 110.0]]],
        ]
    )
    _, stats = generate_preprocessing_qc_plot(
        signal,
        np.ones((2, 1, 1), dtype=bool),
        baseline_frames=3,
        bolus_arrival_frame=3,
        volume_tr_s=4.0,
        output_path=str(tmp_path / "dce_qc.png"),
    )

    pooled_noise = 5.5
    expected = np.mean([10.0 / pooled_noise, 10.0 / 10.0])
    assert stats["mean_cnr"] == pytest.approx(expected, abs=0.01)
