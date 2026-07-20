"""Unit tests for DCE signal quality QC gating.

Tests for peak enhancement and baseline tSNR gating in the DCE QC
recommendation system.
"""

import warnings

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

    def test_baseline_snr_thresholds(self) -> None:
        """Baseline-SNR floor and low-fraction WARN thresholds are set."""
        from pmriprep.utils.dce_constants import (
            QC_DCE_BASELINE_SNR_LOW_FRACTION_WARN,
            QC_DCE_BASELINE_SNR_MIN,
        )

        assert QC_DCE_BASELINE_SNR_MIN == 3.0
        assert QC_DCE_BASELINE_SNR_LOW_FRACTION_WARN == 0.15

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


class TestBaselineSNRLowFractionGating:
    """Report-only baseline-SNR low-fraction caution (never gates processing)."""

    @staticmethod
    def _build_qc_metrics(low_fraction: float | None) -> dict:
        """Build minimal QC metrics dict for testing."""
        preproc: dict = {}
        if low_fraction is not None:
            preproc["baseline_snr_low_fraction"] = low_fraction
        return {
            "preprocessing": preproc,
            "aif_detection": {},
            "pk_modeling": {},
        }

    def test_high_fraction_review(self) -> None:
        """Fraction above WARN triggers a REVIEW (never EXCLUDE)."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(low_fraction=0.30)
        result = recommend_dce_qc(qc)

        rec = result["recommendation"]
        assert rec["action"] == QCAction.REVIEW
        assert "rician" in rec["reason"].lower() or "baseline snr" in rec["reason"].lower()
        assert "30%" in rec["reason"]

    def test_low_fraction_no_gating(self) -> None:
        """Fraction below WARN does not flag."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(low_fraction=0.0)
        result = recommend_dce_qc(qc)

        assert result["recommendation"]["action"] == QCAction.INCLUDE

    def test_boundary_at_warn_is_review(self) -> None:
        """Exactly at the WARN fraction triggers REVIEW (box color and note aligned)."""
        from pmriprep.utils.dce_constants import (
            QC_DCE_BASELINE_SNR_LOW_FRACTION_WARN,
        )
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(low_fraction=QC_DCE_BASELINE_SNR_LOW_FRACTION_WARN)
        result = recommend_dce_qc(qc)

        assert result["recommendation"]["action"] == QCAction.REVIEW

    def test_nan_fraction_no_spurious_warn(self) -> None:
        """NaN fraction (metric unavailable) must not raise or flag."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(low_fraction=float("nan"))
        result = recommend_dce_qc(qc)

        assert result["recommendation"]["action"] == QCAction.INCLUDE

    def test_absent_fraction_no_crash(self) -> None:
        """Missing key does not raise."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(low_fraction=None)
        result = recommend_dce_qc(qc)

        assert result["recommendation"]["action"] == QCAction.INCLUDE

    def test_never_excludes(self) -> None:
        """Even a fully-low-SNR brain is REVIEW, never EXCLUDE, from this axis."""
        from pmriprep.utils.qc_standards import recommend_dce_qc

        qc = self._build_qc_metrics(low_fraction=1.0)
        result = recommend_dce_qc(qc)

        assert result["recommendation"]["action"] == QCAction.REVIEW


class TestBaselineSNRComputation:
    """The robust baseline-SNR low-fraction computed in the preprocessing plot."""

    @staticmethod
    def _signal(shape_xyz, n_low, n_high, n_const, n_t=40):
        """Build a 4D signal with planted low/high-SNR and constant voxels.

        Thermal noise is seeded Gaussian, so the successive-difference MAD
        recovers the per-frame sigma. Low-SNR voxels sit far below the SNR-3
        floor, high-SNR voxels far above; constant voxels have sigma == 0 and
        must drop out of the fraction denominator.
        """
        import numpy as np

        rng = np.random.default_rng(0)
        nx, ny, nz = shape_xyz
        n_vox = nx * ny * nz
        assert n_low + n_high + n_const <= n_vox
        flat = np.zeros((n_vox, n_t), dtype=np.float64)
        idx = 0
        for _ in range(n_low):
            flat[idx] = 5.0 + rng.normal(0.0, 20.0, n_t)   # SNR ~0.25 -> low
            idx += 1
        for _ in range(n_high):
            flat[idx] = 2000.0 + rng.normal(0.0, 10.0, n_t)  # SNR ~200 -> not low
            idx += 1
        for _ in range(n_const):
            flat[idx] = 500.0  # sigma == 0 -> excluded
            idx += 1
        return flat.reshape(nx, ny, nz, n_t)

    def test_planted_fraction(self, tmp_path) -> None:
        """Fraction matches the planted low-SNR count over finite-sigma voxels."""
        import matplotlib

        matplotlib.use("Agg")
        import numpy as np

        from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

        sig = self._signal((4, 4, 2), n_low=8, n_high=24, n_const=0)
        mask = np.ones((4, 4, 2), dtype=bool)
        _, stats = generate_preprocessing_qc_plot(
            sig, mask, baseline_frames=5, bolus_arrival_frame=8,
            volume_tr_s=4.0, output_path=str(tmp_path / "qc.png"),
        )
        # 8 low of 32 finite-sigma voxels = 0.25
        assert stats["baseline_snr_low_fraction"] == pytest.approx(0.25, abs=1e-3)

    def test_sigma_zero_voxels_excluded(self, tmp_path) -> None:
        """Constant (sigma==0) voxels leave the denominator, not counted as low."""
        import matplotlib

        matplotlib.use("Agg")
        import numpy as np

        from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

        # 8 low + 24 high finite-sigma, plus 32 constant voxels that must be excluded.
        sig = self._signal((4, 4, 4), n_low=8, n_high=24, n_const=32)
        mask = np.ones((4, 4, 4), dtype=bool)
        _, stats = generate_preprocessing_qc_plot(
            sig, mask, baseline_frames=5, bolus_arrival_frame=8,
            volume_tr_s=4.0, output_path=str(tmp_path / "qc.png"),
        )
        # Denominator excludes the 32 constant voxels: 8 / 32 = 0.25 (not 8/64).
        assert stats["baseline_snr_low_fraction"] == pytest.approx(0.25, abs=1e-3)

    def test_too_few_volumes_is_nan(self, tmp_path) -> None:
        """Fewer than 3 volumes yields NaN, never None."""
        import math

        import matplotlib

        matplotlib.use("Agg")
        import numpy as np

        from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

        sig = np.ones((4, 4, 2, 2), dtype=np.float64) * 100.0
        mask = np.ones((4, 4, 2), dtype=bool)
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            _, stats = generate_preprocessing_qc_plot(
                sig, mask, baseline_frames=1, bolus_arrival_frame=1,
                volume_tr_s=4.0, output_path=str(tmp_path / "qc.png"),
            )
        assert not warning_records
        frac = stats["baseline_snr_low_fraction"]
        assert isinstance(frac, float)
        assert math.isnan(frac)

    def test_all_sigma_zero_is_nan(self, tmp_path) -> None:
        """Enough volumes but every voxel constant (0/0) yields NaN, never 0.0."""
        import math

        import matplotlib

        matplotlib.use("Agg")
        import numpy as np

        from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

        # n_volumes >= 3 so the guard passes, but all voxels constant -> sigma==0
        # everywhere -> n_valid == 0 -> the 0/0 branch must return NaN (not 0.0).
        sig = self._signal((4, 4, 2), n_low=0, n_high=0, n_const=16)
        mask = np.ones((4, 4, 2), dtype=bool)
        _, stats = generate_preprocessing_qc_plot(
            sig, mask, baseline_frames=5, bolus_arrival_frame=8,
            volume_tr_s=4.0, output_path=str(tmp_path / "qc.png"),
        )
        frac = stats["baseline_snr_low_fraction"]
        assert isinstance(frac, float)
        assert math.isnan(frac)
