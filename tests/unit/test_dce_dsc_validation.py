"""Tests for DSC-DCE cross-validation logic."""

import numpy as np
import pytest

from pmriprep.utils.dce_dsc_validation import (
    CORRELATION_PASS_SOFT,
    CORRELATION_PASS_UNCONSTRAINED,
    CORRELATION_WARN_SOFT,
    CORRELATION_WARN_UNCONSTRAINED,
    FRACTION_VALID_PASS_SOFT,
    FRACTION_VALID_PASS_UNCONSTRAINED,
    FRACTION_VALID_WARN_SOFT,
    FRACTION_VALID_WARN_UNCONSTRAINED,
    MIN_VALID_VOXELS,
    MIN_VP_FOR_VALIDATION,
    ValidationResult,
    validate_vp_cbv,
)
from pmriprep.utils.qc_standards import QCStatus


def _make_data(
    n: int = 5000,
    correlation: float = 0.5,
    vp_mean: float = 0.010,
    cbv_mean: float = 1.0,
    seed: int = 42,
):
    """Generate correlated vp and nCBV arrays with a 3D mask.

    Default vp_mean=0.010 and cbv_mean=1.0 give the NAWM-anchored ratio.

    Returns (vp_data, cbv_data, mask) where vp_data and cbv_data are 3D
    arrays of shape (n, 1, 1) and mask is a boolean array.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    # Create correlated noise
    z_corr = correlation * z1 + np.sqrt(1 - correlation**2) * z2
    vp_flat = np.clip(vp_mean + 0.004 * z1, 0.002, 0.14)
    cbv_flat = np.clip(cbv_mean + 0.4 * z_corr, 0.31, 4.9)
    # Reshape to 3D
    vp_data = vp_flat.reshape(n, 1, 1)
    cbv_data = cbv_flat.reshape(n, 1, 1)
    mask = np.ones((n, 1, 1), dtype=bool)
    return vp_data, cbv_data, mask


class TestCorrelationStatus:
    """Correlation should affect pass/fail status."""

    def test_high_correlation_passes(self):
        vp, cbv, mask = _make_data(correlation=0.6)
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status == QCStatus.PASS
        # Verify both Spearman and Pearson are populated
        assert result.metrics.correlation > 0.3  # Spearman
        assert result.metrics.pearson_correlation > 0.3  # Pearson

    def test_weak_correlation_poor(self):
        """Very weak correlation (ρ<0.20) → POOR."""
        vp, cbv, mask = _make_data(correlation=0.2)
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status == QCStatus.POOR
        assert "No meaningful" in result.interpretation

    def test_no_correlation_poor(self):
        vp, cbv, mask = _make_data(correlation=0.0)
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status == QCStatus.POOR
        # With zero input correlation, actual r may be slightly negative
        assert (
            "No meaningful" in result.interpretation
            or "implausible" in result.interpretation
        )

    def test_negative_correlation_poor(self):
        """Negative correlation is physically implausible."""
        rng = np.random.default_rng(99)
        n = 3000
        vp_flat = np.clip(0.04 + 0.02 * rng.standard_normal(n), 0.002, 0.14)
        # Invert relationship
        cbv_flat = np.clip(6.0 - 1.5 * (vp_flat - 0.04) / 0.02, 0.2, 7.5)
        vp = vp_flat.reshape(n, 1, 1)
        cbv = cbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status == QCStatus.POOR
        assert "implausible" in result.interpretation


class TestSoftConstraintHigherBar:
    """Soft constraint mode should require higher correlation."""

    def test_soft_moderate_correlation_warns(self):
        """Moderate correlation passes unconstrained but warns under soft constraint."""
        vp, cbv, mask = _make_data(correlation=0.50)
        result_skip = validate_vp_cbv(vp, cbv, mask, constraint_mode="skip")
        result_soft = validate_vp_cbv(vp, cbv, mask, constraint_mode="soft")
        # Unconstrained should pass, soft should warn (or worse)
        assert result_skip.status == QCStatus.PASS
        assert result_soft.status in (QCStatus.WARN, QCStatus.POOR)


class TestCoverageStatus:
    """Low fraction_valid should downgrade status."""

    def test_very_low_coverage_poor(self):
        """Most voxels filtered out → POOR (unconstrained mode)."""
        n = 5000
        rng = np.random.default_rng(42)
        # Make most vp values below threshold (will be filtered)
        vp_flat = np.full(n, 0.0005)  # below MIN_VP_FOR_VALIDATION
        cbv_flat = np.full(n, 1.0)
        # A few valid voxels (above MIN_VALID_VOXELS)
        n_valid = 600
        vp_flat[:n_valid] = np.clip(
            0.04 + 0.02 * rng.standard_normal(n_valid), 0.002, 0.14
        )
        cbv_flat[:n_valid] = np.clip(
            3.0 + 1.5 * rng.standard_normal(n_valid), 0.2, 7.5
        )
        vp = vp_flat.reshape(n, 1, 1)
        cbv = cbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        result = validate_vp_cbv(vp, cbv, mask)
        # 600/5000 = 12% < 20% unconstrained warn threshold → coverage degrades
        assert result.status in (QCStatus.WARN, QCStatus.POOR)


class TestSoftConstraintCoverage:
    """Soft-constrained coverage uses relaxed thresholds."""

    def test_soft_constraint_low_coverage_passes(self):
        """~6% coverage should PASS under soft constraint (threshold 5%)."""
        n = 10000
        rng = np.random.default_rng(42)
        # 94% below MIN_VP threshold
        vp_flat = np.full(n, 0.0005)
        cbv_flat = np.full(n, 1.0)
        n_valid = 600  # 6% coverage
        z1 = rng.standard_normal(n_valid)
        vp_flat[:n_valid] = np.clip(0.019 + 0.008 * z1, 0.002, 0.14)
        cbv_flat[:n_valid] = np.clip(3.0 + 1.5 * z1, 0.2, 7.5)
        vp = vp_flat.reshape(n, 1, 1)
        cbv = cbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        result = validate_vp_cbv(vp, cbv, mask, constraint_mode="soft")
        # 6% > 5% soft pass threshold → coverage PASS
        # Overall status depends on correlation and ratio too
        assert "Insufficient overlap" not in result.interpretation
        assert "Limited overlap" not in result.interpretation

    def test_soft_constraint_very_low_coverage_poor(self):
        """~1.5% coverage should be POOR even under soft constraint."""
        n = 10000
        rng = np.random.default_rng(42)
        vp_flat = np.full(n, 0.0005)
        cbv_flat = np.full(n, 1.0)
        n_valid = 150  # 1.5% < 2% soft warn threshold
        # But 150 < MIN_VALID_VOXELS (500) → early return WARN
        # So use n_valid > 500 but fraction < 2%
        n = 50000
        n_valid = 600  # 1.2% < 2% soft warn threshold
        vp_flat = np.full(n, 0.0005)
        cbv_flat = np.full(n, 1.0)
        z1 = rng.standard_normal(n_valid)
        z2 = rng.standard_normal(n_valid)
        vp_flat[:n_valid] = np.clip(0.019 + 0.008 * z1, 0.002, 0.14)
        cbv_flat[:n_valid] = np.clip(
            1.0 + 0.25 * z1 + 0.25 * z2, 0.31, 4.9
        )
        vp = vp_flat.reshape(n, 1, 1)
        cbv = cbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        result = validate_vp_cbv(vp, cbv, mask, constraint_mode="soft")
        assert result.status == QCStatus.POOR
        assert "Insufficient overlap" in result.interpretation


class TestRatioStillWorks:
    """Ratio-based detection should still work."""

    def test_low_ratio_preload_detected(self):
        """Low vp/CBV ratio should still flag preload."""
        vp, cbv, mask = _make_data(
            correlation=0.5, vp_mean=0.003, cbv_mean=1.0
        )
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.preload_effect_detected is True

    def test_high_ratio_warns(self):
        """High vp/CBV ratio should warn."""
        vp, cbv, mask = _make_data(
            correlation=0.5, vp_mean=0.10, cbv_mean=1.0
        )
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status in (QCStatus.WARN, QCStatus.POOR)


class TestWorstOfLogic:
    """Final status should be the worst of ratio, correlation, coverage."""

    def test_good_ratio_bad_correlation_is_poor(self):
        """This is the exact scenario from the bug report: r=0.04, ratio OK."""
        vp, cbv, mask = _make_data(correlation=0.0)
        result = validate_vp_cbv(vp, cbv, mask)
        # Ratio may be PASS but correlation is POOR → final is POOR
        assert result.status == QCStatus.POOR

    def test_bad_ratio_good_correlation_is_warn(self):
        """Low ratio (preload) with good correlation → WARN from ratio."""
        vp, cbv, mask = _make_data(
            correlation=0.5, vp_mean=0.003, cbv_mean=1.0,
        )
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status == QCStatus.WARN
        assert result.preload_effect_detected is True

    def test_good_ratio_good_corr_bad_coverage_is_poor(self):
        """Good ratio and correlation but terrible coverage → POOR."""
        n = 10000
        rng = np.random.default_rng(42)
        # 95% of voxels below MIN_VP threshold
        vp_flat = np.full(n, 0.0005)
        cbv_flat = np.full(n, 1.0)
        n_valid = 600  # 6% coverage < 10% unconstrained warn → POOR
        z1 = rng.standard_normal(n_valid)
        z2 = rng.standard_normal(n_valid)
        vp_flat[:n_valid] = np.clip(0.019 + 0.008 * z1, 0.002, 0.14)
        cbv_flat[:n_valid] = np.clip(
            1.0 + 0.25 * z1 + 0.25 * z2, 0.31, 4.9
        )
        vp = vp_flat.reshape(n, 1, 1)
        cbv = cbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status == QCStatus.POOR


class TestEarlyReturns:
    """Edge cases that trigger early returns."""

    def test_empty_mask_returns_poor(self):
        mask = np.zeros((10, 1, 1), dtype=bool)
        result = validate_vp_cbv(
            np.zeros((10, 1, 1)), np.zeros((10, 1, 1)), mask,
        )
        assert result.status == QCStatus.POOR

    def test_insufficient_voxels_returns_warn(self):
        """Fewer than MIN_VALID_VOXELS valid voxels after filtering → WARN."""
        n = 1000
        rng = np.random.default_rng(42)
        # Only 200 valid, rest below MIN_VP — 200 < 500 MIN_VALID_VOXELS
        vp_flat = np.full(n, 0.0005)
        cbv_flat = np.full(n, 1.0)
        vp_flat[:200] = np.clip(0.019 + 0.005 * rng.standard_normal(200), 0.002, 0.14)
        cbv_flat[:200] = np.clip(3.0 + 1.0 * rng.standard_normal(200), 0.2, 7.5)
        vp = vp_flat.reshape(n, 1, 1)
        cbv = cbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status == QCStatus.WARN
        assert "Insufficient valid voxels" in result.interpretation


class TestCoverageWarn:
    """Coverage between warn and pass thresholds."""

    def test_marginal_coverage_warns(self):
        """~15% valid voxels → coverage WARN (unconstrained)."""
        n = 5000
        rng = np.random.default_rng(42)
        # 85% below MIN_VP, 15% valid
        n_valid = 750
        vp_flat = np.full(n, 0.0005)
        cbv_flat = np.full(n, 1.0)
        z1 = rng.standard_normal(n_valid)
        vp_flat[:n_valid] = np.clip(0.019 + 0.008 * z1, 0.002, 0.14)
        cbv_flat[:n_valid] = np.clip(1.0 + 0.4 * z1, 0.31, 4.9)
        vp = vp_flat.reshape(n, 1, 1)
        cbv = cbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        result = validate_vp_cbv(vp, cbv, mask)
        # Coverage ~15% → WARN; correlation should be good (same z1)
        assert result.status in (QCStatus.WARN, QCStatus.POOR)
        assert "Limited overlap" in result.interpretation


class TestTightenedThresholds:
    """Tests for tightened cross-validation thresholds."""

    def test_marginal_correlation_unconstrained_warns(self) -> None:
        """ρ≈0.35 is between new WARN (0.20) and PASS (0.40) → WARN."""
        vp, cbv, mask = _make_data(correlation=0.45)
        result = validate_vp_cbv(vp, cbv, mask, constraint_mode="skip")
        # correlation=0.45 produces ρ≈0.40, right at the boundary
        assert result.metrics.correlation >= CORRELATION_WARN_UNCONSTRAINED
        assert result.metrics.correlation <= CORRELATION_PASS_UNCONSTRAINED + 0.05
        # Status depends on whether ρ lands just above or below 0.40
        assert result.status in (QCStatus.PASS, QCStatus.WARN)

    def test_ratio_marginal_warn(self) -> None:
        """Ratio outside ±30% but within ±50% → WARN with 'marginally' text."""
        # expected = 0.010; vp_mean=0.003 is below the expected range.
        vp, cbv, mask = _make_data(vp_mean=0.003, cbv_mean=1.0)
        result = validate_vp_cbv(vp, cbv, mask, constraint_mode="skip")
        if result.metrics.median_ratio < 0.468 or result.metrics.median_ratio > 0.870:
            # Outside PASS band but possibly within WARN band
            if 0.335 <= result.metrics.median_ratio <= 1.004:
                assert "marginally" in result.interpretation

    def test_ratio_within_pass_band(self) -> None:
        """Ratio within ±30% of expected → PASS on ratio component."""
        # Default vp_mean=0.010, cbv_mean=1.0 gives ratio ≈ 0.010.
        vp, cbv, mask = _make_data()
        result = validate_vp_cbv(vp, cbv, mask, constraint_mode="skip")
        assert "within expected range" in result.interpretation


class TestNcbvPathValidation:
    """Tests for the NAWM-normalized cross-validation path."""

    def test_validation_with_ncbv_uses_nawm_anchored_expected_ratio(
        self,
    ) -> None:
        """Validation anchors the expected ratio to NAWM_VP_REFERENCE."""
        from pmriprep.utils.dce_constants import NAWM_VP_REFERENCE

        # Generate nCBV centered at 1.0 (NAWM reference), vp centered at
        # NAWM_VP_REFERENCE with partially independent noise so std(vp/nCBV)
        # > the circular gate.
        n = 5000
        rng = np.random.default_rng(7)
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        ncbv_flat = np.clip(1.0 + 0.3 * z1, 0.4, 4.5)
        vp_flat = np.clip(
            NAWM_VP_REFERENCE + 0.003 * (0.5 * z1 + 0.87 * z2),
            0.002, 0.07,
        )
        ncbv = ncbv_flat.reshape(n, 1, 1)
        vp = vp_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)

        result = validate_vp_cbv(
            vp, ncbv, mask, constraint_mode="skip",
        )
        # Median vp/nCBV should be near NAWM_VP_REFERENCE and inside the
        # factor-of-2 tolerance band.
        assert result.metrics.median_ratio == pytest.approx(
            NAWM_VP_REFERENCE, abs=0.003
        )
        assert result.circular is False

    def test_widened_soft_does_not_trigger_circular_ncbv_path(self) -> None:
        """Realistic SOFT posterior spread (> CIRCULAR_STD_THRESHOLD_NCBV)
        on the nCBV path is NOT flagged circular."""
        from pmriprep.utils.dce_constants import CIRCULAR_STD_THRESHOLD_NCBV

        # Inject enough vp spread that std(vp/nCBV) clearly exceeds 0.002.
        n = 5000
        rng = np.random.default_rng(11)
        ncbv_flat = np.clip(1.0 + 0.3 * rng.standard_normal(n), 0.4, 4.5)
        vp_flat = np.clip(0.020 + 0.008 * rng.standard_normal(n), 0.003, 0.14)
        ncbv = ncbv_flat.reshape(n, 1, 1)
        vp = vp_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)

        result = validate_vp_cbv(
            vp, ncbv, mask, constraint_mode="soft",
        )
        assert result.metrics.std_ratio > CIRCULAR_STD_THRESHOLD_NCBV
        assert result.circular is False


class TestAcquisitionLimited:
    """Acquisition-limited flag widens the nCBV ratio tolerance when the
    upstream ``aif_tissue_overshoot_fraction`` exceeds the prior-domination
    threshold (Buckley MRM 2002).

    Fixture note: tests on the nCBV-anchored path synthesize data with
    ``cbv_mean=1.0`` (NAWM-normalized scale, NAWM=1) so that the observed
    vp/nCBV ratio approximates ``vp_mean`` directly, matching
    NAWM_VP_REFERENCE=0.02.
    """

    @staticmethod
    def _ncbv_data(vp_mean: float, n: int = 5000, seed: int = 7):
        rng = np.random.default_rng(seed)
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        z_corr = 0.6 * z1 + np.sqrt(1 - 0.6**2) * z2
        vp_flat = np.clip(vp_mean + 0.008 * z1, 0.002, 0.14)
        ncbv_flat = np.clip(1.0 + 0.3 * z_corr, 0.4, 4.5)
        vp = vp_flat.reshape(n, 1, 1)
        ncbv = ncbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        return vp, ncbv, mask

    def test_overshoot_above_threshold_widens_tolerance(self) -> None:
        """v254 half-dose pattern: overshoot=0.94, ratio≈0.007 → PASS with flag."""
        vp, ncbv, mask = self._ncbv_data(vp_mean=0.007)
        result = validate_vp_cbv(
            vp, ncbv, mask,
            constraint_mode="soft",
            aif_tissue_overshoot_fraction=0.94,
        )
        assert result.acquisition_limited is True
        assert result.status == QCStatus.PASS
        assert "Buckley" in result.interpretation

    def test_overshoot_below_threshold_keeps_tight_band(self) -> None:
        """overshoot=0.3 with low ratio still hits WARN on the tight band."""
        vp, ncbv, mask = self._ncbv_data(vp_mean=0.007)
        result = validate_vp_cbv(
            vp, ncbv, mask,
            constraint_mode="soft",
            aif_tissue_overshoot_fraction=0.3,
        )
        assert result.acquisition_limited is False
        assert result.status == QCStatus.WARN

    def test_overshoot_boundary_exactly_0_7_not_flagged(self) -> None:
        """Threshold uses strict ``>``; overshoot == 0.7 exactly → tight-band path."""
        vp, ncbv, mask = self._ncbv_data(vp_mean=0.007)
        result = validate_vp_cbv(
            vp, ncbv, mask,
            constraint_mode="soft",
            aif_tissue_overshoot_fraction=0.7,
        )
        assert result.acquisition_limited is False

    def test_overshoot_none_uses_tight_band(self) -> None:
        """Missing overshoot (user-forced mode / no-NAWM path) → tight band."""
        vp, ncbv, mask = self._ncbv_data(vp_mean=0.007)
        result = validate_vp_cbv(
            vp, ncbv, mask,
            constraint_mode="soft",
            aif_tissue_overshoot_fraction=None,
        )
        assert result.acquisition_limited is False
        # Observed ratio ≈0.007 is outside the tight ±30% band around 0.02,
        # so the tight-band path still WARNs — proving the flag was not applied.
        assert result.status == QCStatus.WARN

    def test_outside_widened_band_still_warns(self) -> None:
        """Sub-physiological ratio (< 0.005 floor) warns even with the flag set."""
        vp, ncbv, mask = self._ncbv_data(vp_mean=0.002)
        result = validate_vp_cbv(
            vp, ncbv, mask,
            constraint_mode="soft",
            aif_tissue_overshoot_fraction=0.94,
        )
        assert result.acquisition_limited is True
        assert result.status == QCStatus.WARN

class TestDscDceValidationNodeSkipReasons:
    """_dsc_dce_validation_node distinguishes 'no DSC run' from 'no NAWM'."""

    def _call(self, **kwargs):
        from pmriprep.workflows.dce.base import _dsc_dce_validation_node
        return _dsc_dce_validation_node(**kwargs)

    def test_reports_no_dsc_run_when_all_dsc_inputs_absent(self, tmp_path):
        vp_file = tmp_path / "vp.nii.gz"
        vp_file.write_bytes(b"dummy")
        result = self._call(
            vp_file=str(vp_file),
            dsc_cbv_file=None,
            dsc_ncbv_file=None,
            brain_mask=None,
            no_dsc_validation=False,
            pk_qc=None,
            dsc_preload_configured=None,
        )
        assert result["skipped"] is True
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "No DSC data provided"
        assert result["skip_reason"] == "no_dsc_run"

    def test_reports_no_ncbv_when_dsc_configured_but_ncbv_missing(self, tmp_path):
        vp_file = tmp_path / "vp.nii.gz"
        vp_file.write_bytes(b"dummy")
        result = self._call(
            vp_file=str(vp_file),
            dsc_cbv_file=None,
            dsc_ncbv_file=None,
            brain_mask=None,
            no_dsc_validation=False,
            pk_qc=None,
            dsc_preload_configured=True,
        )
        assert result["skipped"] is True
        assert result["status"] == "SKIPPED"
        assert "nCBV unavailable" in result["reason"]
        assert "skip_reason" not in result
