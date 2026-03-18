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
    vp_mean: float = 0.019,
    cbv_mean: float = 3.0,
    seed: int = 42,
):
    """Generate correlated vp and CBV arrays with a 3D mask.

    Default vp_mean=0.019 and cbv_mean=3.0 give a median ratio near the
    expected 0.624 = (1 - 0.40) * 1.04.

    Returns (vp_data, cbv_data, mask) where vp_data and cbv_data are 3D
    arrays of shape (n, 1, 1) and mask is a boolean array.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    # Create correlated noise
    z_corr = correlation * z1 + np.sqrt(1 - correlation**2) * z2
    vp_flat = np.clip(vp_mean + 0.008 * z1, 0.002, 0.14)
    cbv_flat = np.clip(cbv_mean + 1.5 * z_corr, 0.2, 7.5)
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

    def test_weak_correlation_warns(self):
        vp, cbv, mask = _make_data(correlation=0.2)
        result = validate_vp_cbv(vp, cbv, mask)
        assert result.status == QCStatus.WARN
        assert "Weak" in result.interpretation

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
        """r=0.35 passes unconstrained but warns under soft constraint."""
        vp, cbv, mask = _make_data(correlation=0.4)
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
        cbv_flat = np.full(n, 2.0)
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
        cbv_flat = np.full(n, 2.0)
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
        cbv_flat = np.full(n, 2.0)
        n_valid = 150  # 1.5% < 2% soft warn threshold
        # But 150 < MIN_VALID_VOXELS (500) → early return WARN
        # So use n_valid > 500 but fraction < 2%
        n = 50000
        n_valid = 600  # 1.2% < 2% soft warn threshold
        vp_flat = np.full(n, 0.0005)
        cbv_flat = np.full(n, 2.0)
        z1 = rng.standard_normal(n_valid)
        vp_flat[:n_valid] = np.clip(0.019 + 0.008 * z1, 0.002, 0.14)
        cbv_flat[:n_valid] = np.clip(3.0 + 1.5 * z1, 0.2, 7.5)
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
            correlation=0.5, vp_mean=0.01, cbv_mean=4.0
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


class TestCircularDetection:
    """Hard constraint should still return CIRCULAR."""

    def test_hard_constraint_circular(self):
        vp, cbv, mask = _make_data()
        result = validate_vp_cbv(vp, cbv, mask, constraint_mode="hard")
        assert result.circular is True
        assert result.status == QCStatus.CIRCULAR


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
            correlation=0.5, vp_mean=0.01, cbv_mean=4.0,
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
        cbv_flat = np.full(n, 2.0)
        n_valid = 600  # 6% coverage < 10% unconstrained warn → POOR
        z1 = rng.standard_normal(n_valid)
        vp_flat[:n_valid] = np.clip(0.019 + 0.008 * z1, 0.002, 0.14)
        cbv_flat[:n_valid] = np.clip(3.0 + 1.5 * z1, 0.2, 7.5)
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
        cbv_flat = np.full(n, 2.0)
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
        cbv_flat = np.full(n, 2.0)
        z1 = rng.standard_normal(n_valid)
        vp_flat[:n_valid] = np.clip(0.019 + 0.008 * z1, 0.002, 0.14)
        cbv_flat[:n_valid] = np.clip(3.0 + 1.5 * z1, 0.2, 7.5)
        vp = vp_flat.reshape(n, 1, 1)
        cbv = cbv_flat.reshape(n, 1, 1)
        mask = np.ones((n, 1, 1), dtype=bool)
        result = validate_vp_cbv(vp, cbv, mask)
        # Coverage ~15% → WARN; correlation should be good (same z1)
        assert result.status in (QCStatus.WARN, QCStatus.POOR)
        assert "Limited overlap" in result.interpretation
