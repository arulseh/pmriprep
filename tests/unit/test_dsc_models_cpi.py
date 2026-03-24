"""Unit tests for CPI (Control Point Interpolation) DSC model."""

import numpy as np
import pytest

from pmriprep.models.dsc.cpi import (
    CPIModel,
    estimate_bolus_duration,
    select_n_by_bic,
    select_n_control_points,
)
from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix


class TestCPIModelConstruction:
    """Tests for CPIModel initialization and properties."""

    def test_default_n6(self) -> None:
        """Default CPIModel() has 6 control points and matching properties."""
        model = CPIModel()
        assert model.name == "cpi_6"
        assert model.n_parameters == 6
        assert model.n_control_points == 6

    def test_custom_n(self) -> None:
        """Custom n_control_points produce different n_parameters."""
        model4 = CPIModel(n_control_points=4)
        model10 = CPIModel(n_control_points=10)
        assert model4.n_parameters == 4
        assert model10.n_parameters == 10
        assert model4.name == "cpi_4"
        assert model10.name == "cpi_10"

    def test_control_point_times_evenly_spaced(self) -> None:
        """Control point times are evenly spaced across the time range."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        times = model.control_point_times
        assert len(times) == 6
        expected = np.linspace(0, 30, 6)
        np.testing.assert_allclose(times, expected, rtol=1e-10)
        assert model.time_range == (0.0, 30.0)


class TestResidueFunction:
    """Tests for CPI residue function R(t)."""

    def test_all_ones_flat_residue(self) -> None:
        """Params with all fractions 1.0 give R(t) approximately 1.0."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        # cbf=0.1, all fractions = 1.0 => control values all 1.0
        params = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0]
        t = np.linspace(0, 30, 50)
        residue = model.residue_function(params, t)
        np.testing.assert_allclose(residue, 1.0, atol=1e-6)

    def test_monotonic_decrease(self) -> None:
        """Residue with decreasing fractions is monotonically non-increasing."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        params = [0.1, 0.8, 0.6, 0.4, 0.2, 0.05]
        t = np.linspace(0, 30, 100)
        residue = model.residue_function(params, t)
        # Check that each value is >= next value (monotonically non-increasing)
        diffs = np.diff(residue)
        assert np.all(diffs <= 1e-10), "Residue should be monotonically non-increasing"

    def test_r0_equals_one(self) -> None:
        """R(0) should always equal 1.0 (dimensionless residue)."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        params = [0.15, 0.7, 0.5, 0.3, 0.1, 0.01]
        t = np.array([0.0])
        residue = model.residue_function(params, t)
        assert residue[0] == pytest.approx(1.0, abs=1e-10)


class TestEnforceConstraints:
    """Tests for constraint enforcement on control points."""

    def test_soft_clips_to_01(self) -> None:
        """Default (soft) enforce_constraints clips to [0,1], no monotonicity."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        # r2=0.9 > r1=0.8 — allowed in soft mode
        params = [0.1, 0.8, 0.9, 0.5, 0.3, 0.2]
        constrained = model.enforce_constraints(params)
        assert constrained[0] == pytest.approx(1.0)
        assert constrained[2] == pytest.approx(0.9)  # NOT capped to 0.8
        # All values in [0, 1]
        for i in range(len(constrained)):
            assert 0.0 <= constrained[i] <= 1.0

    def test_hard_enforces_monotonicity(self) -> None:
        """Hard mode caps fractions to enforce monotonic non-increase."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        params = [0.1, 0.8, 0.9, 0.5, 0.3, 0.2]
        constrained = model.enforce_constraints_hard(params)
        assert constrained[0] == pytest.approx(1.0)
        for i in range(1, len(constrained)):
            assert constrained[i] <= constrained[i - 1] + 1e-15

    def test_monotonicity_prior_penalizes_violations(self) -> None:
        """Soft monotonicity prior penalizes violations more than decrements."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        # Monotonic (good)
        params_good = [0.1, 0.8, 0.6, 0.4, 0.2, 0.1]
        # Non-monotonic (bad) — r2=0.9 > r1=0.8
        params_bad = [0.1, 0.8, 0.9, 0.4, 0.2, 0.1]
        lp_good = model.compute_monotonicity_prior(params_good)
        lp_bad = model.compute_monotonicity_prior(params_bad)
        assert lp_good > lp_bad, "Monotonic should have higher log-prior"


class TestTissueCurve:
    """Tests for tissue concentration curve computation."""

    def test_tissue_curve_shape(self) -> None:
        """Tissue curve has same length as time points."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        n = 30
        t = np.linspace(0, 30, n)
        aif = (t / 3.0) * np.exp(-t / 5.0)
        params = [0.1, 0.8, 0.6, 0.4, 0.2, 0.05]
        tissue = model.tissue_curve(params, aif, t)
        assert tissue.shape == (n,)

    def test_zero_aif_zero_tissue(self) -> None:
        """All-zero AIF produces all-zero tissue curve."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        n = 30
        t = np.linspace(0, 30, n)
        aif = np.zeros(n, dtype=np.float64)
        params = [0.1, 0.8, 0.6, 0.4, 0.2, 0.05]
        tissue = model.tissue_curve(params, aif, t)
        np.testing.assert_allclose(tissue, 0.0, atol=1e-15)


class TestDerivedParameters:
    """Tests for CBF, CBV, and MTT extraction."""

    def test_get_cbf(self) -> None:
        """get_cbf returns the first parameter."""
        model = CPIModel(n_control_points=6)
        params = [0.15, 0.8, 0.6, 0.4, 0.2, 0.1]
        assert model.get_cbf(params) == pytest.approx(0.15)

    def test_get_mtt_positive(self) -> None:
        """MTT should be positive for reasonable parameters."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        params = [0.15, 0.8, 0.6, 0.4, 0.2, 0.1]
        mtt = model.get_mtt(params)
        assert mtt > 0.0


class TestComputeFittedResidue:
    """Tests for CBF-scaled residue output."""

    def test_fitted_residue_scales_by_cbf(self) -> None:
        """compute_fitted_residue returns CBF × R(t), not dimensionless R(t)."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        params = [50.0, 0.8, 0.6, 0.4, 0.2, 0.05]
        t = np.linspace(0, 30, 50)
        residue_norm = model.residue_function(params, t)
        fitted_residue = model.compute_fitted_residue(params, t)
        np.testing.assert_allclose(fitted_residue, 50.0 * residue_norm, rtol=1e-10)

    def test_fitted_residue_peak_equals_cbf(self) -> None:
        """Peak of compute_fitted_residue equals CBF (since R(0)=1)."""
        model = CPIModel(n_control_points=6, time_range=(0, 30))
        cbf = 75.0
        params = [cbf, 0.8, 0.6, 0.4, 0.2, 0.05]
        t = np.linspace(0, 30, 50)
        fitted_residue = model.compute_fitted_residue(params, t)
        assert fitted_residue[0] == pytest.approx(cbf, rel=1e-10)


class TestSelectionFunctions:
    """Tests for automatic control point selection."""

    def test_estimate_bolus_duration(self) -> None:
        """Gaussian-like concentration with FWHM ~10 frames gives duration in [5, 20]."""
        t = np.arange(60, dtype=np.float64)
        # Gaussian centered at frame 20 with sigma ~4 (FWHM ~10 frames)
        concentration = np.exp(-0.5 * ((t - 20.0) / 4.0) ** 2)
        duration = estimate_bolus_duration(concentration, threshold_fraction=0.1)
        assert 5 <= duration <= 20, f"Duration {duration} not in [5, 20]"

    def test_select_n_control_points(self) -> None:
        """Short bolus gives low N, long bolus gives higher N, both within [4, 12]."""
        # Short bolus: 5 frames, TR=1.5
        n_short = select_n_control_points(5, tr_seconds=1.5)
        # Long bolus: 20 frames, TR=1.5
        n_long = select_n_control_points(20, tr_seconds=1.5)
        assert 4 <= n_short <= 12
        assert 4 <= n_long <= 12
        assert n_short < n_long

    def test_select_n_by_bic(self) -> None:
        """BIC selection returns valid best_n and results list."""
        n = 40
        t = np.linspace(0, 40, n)
        dt = float(t[1] - t[0])
        # Create synthetic AIF
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0
        # Create synthetic tissue: convolution with exponential residue
        residue = 0.1 * np.exp(-t / 10.0)
        matrix = build_toeplitz_matrix(aif, dt)
        tissue = matrix @ residue
        # Add small noise
        rng = np.random.default_rng(42)
        tissue += rng.normal(0, 0.001, n)

        best_n, results = select_n_by_bic(tissue, aif, t, n_candidates=(4, 5, 6, 7, 8))
        assert 4 <= best_n <= 10
        assert len(results) > 0
        # Each result should be a (n, bic, rss) tuple
        for entry in results:
            assert len(entry) == 3
