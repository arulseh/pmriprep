"""Unit tests for CPI (Control Point Interpolation) DSC model."""

import numpy as np
import pytest

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.cpi import (
    CPIModel,
    _fit_cpi_simple,
    estimate_bolus_duration,
    select_n_by_bic,
    select_n_control_points,
)
from pmriprep.utils.dsc_deconvolution import build_volterra_matrix


class TestCPIModelConstruction:
    """Tests for CPIModel initialization and properties."""

    def test_default_n6(self) -> None:
        """Default CPIModel(n_acquisition_frames=20, dt=1.5) has 6 control points and matching properties."""
        model = CPIModel(n_acquisition_frames=20, dt=1.5)
        assert model.name == "cpi_6"
        # FABBER convention: cbf + N free fractions = N+1 params.
        assert model.n_parameters == 7
        assert model.n_control_points == 6

    def test_custom_n(self) -> None:
        """Custom n_control_points produce different n_parameters."""
        model4 = CPIModel(n_control_points=4, n_acquisition_frames=20, dt=1.5)
        model10 = CPIModel(n_control_points=10, n_acquisition_frames=20, dt=1.5)
        assert model4.n_parameters == 5  # cbf + 4 fractions
        assert model10.n_parameters == 11  # cbf + 10 fractions
        assert model4.name == "cpi_4"
        assert model10.name == "cpi_10"

    def test_fraction_transforms_are_sigmoid(self) -> None:
        """CPI fraction parameters use SIGMOID (FABBER TRANSFORM_FRACTIONAL)."""
        model = CPIModel(n_acquisition_frames=20, dt=1.5)
        params = model.get_parameter_definitions()
        # CBF (index 0) is identity-transformed (FABBER: var=1e6 flat prior)
        assert params[0].transform == TransformType.IDENTITY, (
            f"CBF uses {params[0].transform}, expected IDENTITY"
        )
        for p in params[1:]:
            assert p.transform == TransformType.SIGMOID, (
                f"Parameter {p.name} uses {p.transform}, expected SIGMOID"
            )

    def test_parameter_names_multiplicative(self) -> None:
        """Parameter names use f1..fN for N CPs (FABBER convention)."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        names = model.get_parameter_names()
        assert names == ["cbf", "f1", "f2", "f3", "f4", "f5", "f6"]

    def test_control_point_times_fabber_layout(self) -> None:
        """CP times follow FABBER layout: tpos[i] = (i+1) * nt/N * dt."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        times = model.control_point_times
        assert len(times) == 6
        # (i+1)*20/6 * 1.5 for i=0..5 → 5, 10, 15, 20, 25, 30 (seconds)
        expected = (np.arange(1, 7) * 20.0 / 6.0) * 1.5
        np.testing.assert_allclose(times, expected, rtol=1e-10)
        # Terminal anchor at (nt+1)*dt = 21*1.5 = 31.5s
        assert model.terminal_time == pytest.approx(31.5)


class TestResidueFunction:
    """Tests for CPI residue function R(t)."""

    def test_all_ones_flat_residue(self) -> None:
        """Params with all fractions 1.0 give R(t) approximately 1.0."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        # cbf=0.1, all 6 fractions = 1.0 => control values all 1.0
        params = [0.1] + [1.0] * 6
        t = np.linspace(0, 30, 50)
        residue = model.residue_function(params, t)
        np.testing.assert_allclose(residue, 1.0, atol=1e-6)

    def test_monotonic_decrease(self) -> None:
        """Residue with multiplicative fractions is monotonically non-increasing."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        params = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # cbf + 6 fractions
        t = np.linspace(0, 30, 100)
        residue = model.residue_function(params, t)
        diffs = np.diff(residue)
        assert np.all(diffs <= 1e-10), "Residue should be monotonically non-increasing"

    def test_r0_equals_one(self) -> None:
        """R(0) should always equal 1.0 (FABBER prepends a (0, 1) anchor)."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        params = [0.15, 0.9, 0.7, 0.5, 0.3, 0.4, 0.6]  # cbf + 6 fractions
        t = np.array([0.0])
        residue = model.residue_function(params, t)
        assert residue[0] == pytest.approx(1.0, abs=1e-10)


class TestEnforceConstraints:
    """Tests for constraint enforcement on control points."""

    def test_cumprod_enforce_constraints(self) -> None:
        """enforce_constraints applies cumprod over N free fractions."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        # cbf + 6 fractions; enforce_constraints returns cumprod over fractions
        params = [0.1, 1.0, 0.8, 0.9, 0.5, 0.3, 0.2]
        constrained = model.enforce_constraints(params)
        assert constrained[0] == pytest.approx(1.0)
        assert constrained[1] == pytest.approx(0.8)
        assert constrained[2] == pytest.approx(0.8 * 0.9)
        assert constrained[3] == pytest.approx(0.8 * 0.9 * 0.5)
        for v in constrained:
            assert 0.0 <= v <= 1.0

    def test_multiplicative_monotonicity_guaranteed(self) -> None:
        """Random f_i in [0,1] always produce non-increasing control values."""
        rng = np.random.default_rng(42)
        model = CPIModel(n_control_points=10, n_acquisition_frames=20, dt=1.5)
        for _ in range(100):
            params = [0.1] + list(rng.uniform(0.0, 1.0, 10))  # cbf + 10 fractions
            constrained = model.enforce_constraints(params)
            diffs = np.diff(constrained)
            assert np.all(diffs <= 1e-15), "Cumprod must be non-increasing"


class TestTissueCurve:
    """Tests for tissue concentration curve computation."""

    def test_tissue_curve_shape(self) -> None:
        """Tissue curve has same length as time points."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        n = 30
        t = np.linspace(0, 30, n)
        aif = (t / 3.0) * np.exp(-t / 5.0)
        params = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # cbf + 6 fractions
        tissue = model.tissue_curve(params, aif, t)
        assert tissue.shape == (n,)

    def test_zero_aif_zero_tissue(self) -> None:
        """All-zero AIF produces all-zero tissue curve."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        n = 30
        t = np.linspace(0, 30, n)
        aif = np.zeros(n, dtype=np.float64)
        params = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        tissue = model.tissue_curve(params, aif, t)
        np.testing.assert_allclose(tissue, 0.0, atol=1e-15)


class TestDerivedParameters:
    """Tests for CBF, CBV, and MTT extraction."""

    def test_get_cbf(self) -> None:
        """get_cbf returns the first parameter."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        params = [0.15, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        assert model.get_cbf(params) == pytest.approx(0.15)

    def test_get_mtt_positive(self) -> None:
        """MTT should be positive for reasonable parameters."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        params = [0.15, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        mtt = model.get_mtt(params)
        assert mtt > 0.0


class TestComputeFittedResidue:
    """Tests for CBF-scaled residue output."""

    def test_fitted_residue_scales_by_cbf(self) -> None:
        """compute_fitted_residue returns CBF × R(t), not dimensionless R(t)."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        params = [0.08, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        t = np.linspace(0, 30, 50)
        residue_norm = model.residue_function(params, t)
        fitted_residue = model.compute_fitted_residue(params, t)
        np.testing.assert_allclose(fitted_residue, 0.08 * residue_norm, rtol=1e-10)

    def test_fitted_residue_peak_equals_cbf(self) -> None:
        """Peak of compute_fitted_residue equals CBF (since R(0)=1)."""
        model = CPIModel(n_control_points=6, n_acquisition_frames=20, dt=1.5)
        cbf = 0.12
        params = [cbf, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
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
        n_short = select_n_control_points(5)
        n_long = select_n_control_points(20)
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
        matrix = build_volterra_matrix(aif, dt)
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


class TestCPIDelay:
    """Optional bolus delay (FABBER inferdelay): param layout + forward shift."""

    @staticmethod
    def _aif(tp):
        tb = tp - 8.0
        return np.where(tb > 0, (tb / 3.0) ** 3 * np.exp(-tb / 3.0), 0.0)

    def test_delay_param_layout(self) -> None:
        """delay is the trailing param; identity transform, mean 0, var 25."""
        nt, dt, n_cp = 40, 1.5, 6
        m_none = CPIModel(n_control_points=n_cp, n_acquisition_frames=nt, dt=dt)
        m_delay = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt, infer_delay=True
        )
        m_both = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt,
            infer_cpt=True, infer_delay=True,
        )
        assert m_none.n_parameters == 1 + n_cp
        assert m_delay.n_parameters == 1 + n_cp + 1
        assert m_both.n_parameters == 1 + 2 * n_cp + 1
        assert m_delay.get_parameter_names()[-1] == "delay"
        assert m_both.get_parameter_names()[-1] == "delay"
        assert m_delay.get_initial_guess()[-1] == 0.0
        ddef = m_delay.get_parameter_definitions()[-1]
        assert ddef.name == "delay"
        assert ddef.transform == TransformType.IDENTITY
        assert ddef.transformed_prior_sigma == pytest.approx(np.sqrt(25.0))

    def test_delay_zero_matches_no_delay(self) -> None:
        """tissue_curve with delay=0 reproduces the no-delay curve exactly."""
        nt, dt, n_cp = 40, 1.5, 6
        tp = np.arange(nt) * dt
        aif = self._aif(tp)
        fr = [0.85, 0.7, 0.55, 0.4, 0.28, 0.18]
        m0 = CPIModel(n_control_points=n_cp, n_acquisition_frames=nt, dt=dt)
        md = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt, infer_delay=True
        )
        t0 = m0.tissue_curve([0.1] + fr, aif, tp)
        td0 = md.tissue_curve([0.1] + fr + [0.0], aif, tp)
        np.testing.assert_allclose(td0, t0, atol=1e-12)

    def test_delay_shifts_peak_later(self) -> None:
        """A positive delay shifts the tissue-curve peak to a later frame."""
        nt, dt, n_cp = 40, 1.5, 6
        tp = np.arange(nt) * dt
        aif = self._aif(tp)
        fr = [0.85, 0.7, 0.55, 0.4, 0.28, 0.18]
        md = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt, infer_delay=True
        )
        t0 = md.tissue_curve([0.1] + fr + [0.0], aif, tp)
        t3 = md.tissue_curve([0.1] + fr + [3.0], aif, tp)
        assert int(np.argmax(t3)) > int(np.argmax(t0))

    def test_cbv_mtt_delay_invariant(self) -> None:
        """CBV and MTT (residue integrals) are unaffected by the delay."""
        nt, dt, n_cp = 40, 1.5, 6
        tp = np.arange(nt) * dt
        fr = [0.85, 0.7, 0.55, 0.4, 0.28, 0.18]
        md = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt, infer_delay=True
        )
        assert md.get_cbv([0.1] + fr + [0.0], tp) == pytest.approx(
            md.get_cbv([0.1] + fr + [3.0], tp), abs=1e-9
        )
        assert md.get_mtt([0.1] + fr + [0.0], tp) == pytest.approx(
            md.get_mtt([0.1] + fr + [3.0], tp), abs=1e-9
        )


def test_nlls_honors_identity_parameter_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPI NLLS bounds inferred spacings before exponentiation."""
    from scipy import optimize

    captured: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    original = optimize.least_squares

    def capture_bounds(*args, **kwargs):
        captured["bounds"] = kwargs["bounds"]
        return original(*args, **kwargs)

    monkeypatch.setattr(optimize, "least_squares", capture_bounds)
    model = CPIModel(
        n_control_points=4,
        n_acquisition_frames=30,
        dt=1.5,
        infer_cpt=True,
        infer_delay=True,
    )
    time_points = np.arange(30, dtype=np.float64) * 1.5
    aif = np.exp(-0.5 * ((time_points - 9.0) / 2.0) ** 2)
    tissue = model.tissue_curve(model.get_initial_guess(), aif, time_points)

    result = _fit_cpi_simple(model, tissue, aif, time_points, sloppy=True)

    assert result is not None
    lower, upper = captured["bounds"]
    assert lower[0] == 0.0
    assert np.isposinf(upper[0])
    np.testing.assert_array_equal(lower[5:9], -5.0)
    np.testing.assert_array_equal(upper[5:9], 5.0)
    assert (lower[-1], upper[-1]) == model.get_parameter_definitions()[-1].bounds
