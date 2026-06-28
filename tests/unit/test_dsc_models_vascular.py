"""Unit tests for DSC vascular model and base dataclasses."""

import numpy as np
import pytest

from pmriprep.models.dsc.base import DSCFitResult, DSCModelParameter
from pmriprep.models.dsc.vascular import VascularModel


# =============================================================================
# DSCModelParameter
# =============================================================================


class TestDSCModelParameter:
    """Tests for DSCModelParameter dataclass methods."""

    def test_get_prior_mean(self) -> None:
        """Prior mean is midpoint of typical_range."""
        param = DSCModelParameter(
            name="test",
            bounds=(0.0, 200.0),
            typical_range=(10.0, 100.0),
        )
        assert param.get_prior_mean() == pytest.approx(55.0)

    def test_get_prior_sigma(self) -> None:
        """Prior sigma is (range_high - range_low) / 4."""
        param = DSCModelParameter(
            name="test",
            bounds=(0.0, 200.0),
            typical_range=(10.0, 100.0),
        )
        assert param.get_prior_sigma() == pytest.approx(22.5)

    def test_clip_to_bounds(self) -> None:
        """Values are clipped to parameter bounds."""
        param = DSCModelParameter(
            name="test",
            bounds=(0.0, 100.0),
            typical_range=(10.0, 90.0),
        )
        assert param.clip_to_bounds(50.0) == pytest.approx(50.0)
        assert param.clip_to_bounds(-10.0) == pytest.approx(0.0)
        assert param.clip_to_bounds(200.0) == pytest.approx(100.0)

    def test_is_valid(self) -> None:
        """Values within bounds are valid; out-of-bounds and NaN are not."""
        param = DSCModelParameter(
            name="test",
            bounds=(0.0, 100.0),
            typical_range=(10.0, 90.0),
        )
        assert param.is_valid(50.0) is True
        assert param.is_valid(-1.0) is False
        assert param.is_valid(float("nan")) is False


# =============================================================================
# DSCFitResult
# =============================================================================


class TestDSCFitResult:
    """Tests for DSCFitResult dataclass methods."""

    def test_has_uncertainty(self) -> None:
        """has_uncertainty returns True when uncertainties are populated."""
        result_with = DSCFitResult(uncertainties={"cbf": 5.0})
        assert result_with.has_uncertainty() is True

        result_without = DSCFitResult()
        assert result_without.has_uncertainty() is False

    def test_passes_convergence(self) -> None:
        """passes_convergence checks rhat, ess, and converged flag."""
        good_result = DSCFitResult(
            converged=True,
            rhat={"cbf": 1.02, "alpha": 1.05},
            ess={"cbf": 500, "alpha": 300},
        )
        assert good_result.passes_convergence() is True

        bad_rhat = DSCFitResult(
            converged=True,
            rhat={"cbf": 1.5, "alpha": 1.05},
            ess={"cbf": 500, "alpha": 300},
        )
        assert bad_rhat.passes_convergence() is False

        bad_ess = DSCFitResult(
            converged=True,
            rhat={"cbf": 1.02, "alpha": 1.05},
            ess={"cbf": 50, "alpha": 300},
        )
        assert bad_ess.passes_convergence() is False

        not_converged = DSCFitResult(
            converged=False,
            rhat={"cbf": 1.02},
            ess={"cbf": 500},
        )
        assert not_converged.passes_convergence() is False

    def test_get_parameter(self) -> None:
        """get_parameter returns value for known keys, default for missing."""
        result = DSCFitResult(parameters={"cbf": 0.08, "alpha": 3.0})
        assert result.get_parameter("cbf") == pytest.approx(0.08)
        assert result.get_parameter("missing") == pytest.approx(0.0)


# =============================================================================
# VascularModel
# =============================================================================


class TestVascularModel:
    """Tests for VascularModel."""

    def test_residue_function_shape(self) -> None:
        """Residue starts at CBF and decays monotonically for delay=0."""
        model = VascularModel()
        t = np.arange(0, 30, 0.5, dtype=np.float64)
        params = [0.08, 3.0, 2.0, 0.0]  # CBF, alpha, beta, delay
        residue = model.residue_function(params, t)

        # Starts at CBF
        assert residue[0] == pytest.approx(0.08, rel=1e-6)
        # Monotonically non-increasing (allow tiny numerical noise)
        diffs = np.diff(residue)
        assert np.all(diffs <= 1e-10), "Residue should decay monotonically"

    def test_residue_function_r0_equals_cbf(self) -> None:
        """R(0) equals CBF when delay=0."""
        model = VascularModel()
        t = np.linspace(0, 20, 50, dtype=np.float64)
        cbf = 0.12
        params = [cbf, 4.0, 1.5, 0.0]
        residue = model.residue_function(params, t)
        assert residue[0] == pytest.approx(cbf, rel=1e-10)

    def test_tissue_curve_nonnegative(self) -> None:
        """Tissue curve from synthetic AIF is non-negative everywhere."""
        model = VascularModel()
        dt = 1.5
        t = np.arange(0, 60, dt, dtype=np.float64)

        # Synthetic gamma-variate AIF
        aif = (t / 5.0) * np.exp(-t / 8.0)
        aif[0] = 0.0

        params = [0.08, 3.0, 2.0, 0.0]
        tissue = model.tissue_curve(params, aif, t)
        assert np.all(tissue >= -1e-10), "Tissue curve should be non-negative"

    def test_compute_derived_parameters(self) -> None:
        """Derived parameters match analytical formulas."""
        model = VascularModel()
        cbf, alpha, beta, delay = 0.08, 3.0, 2.0, 0.0
        params = [cbf, alpha, beta, delay]
        derived = model.compute_derived_parameters(params)

        # MTT = alpha * beta = 3 * 2 = 6
        assert derived["mtt"] == pytest.approx(6.0, rel=1e-10)
        # CBV = CBF * MTT = 0.08 * 6 = 0.48 (relative units, no /60)
        assert derived["cbv"] == pytest.approx(0.48, rel=1e-10)
        # CTH = beta * sqrt(alpha) = 2 * sqrt(3)
        assert derived["cth"] == pytest.approx(2.0 * np.sqrt(3.0), rel=1e-10)
        # RCTH = 1 / sqrt(alpha) = 1 / sqrt(3)
        assert derived["rcth"] == pytest.approx(1.0 / np.sqrt(3.0), rel=1e-10)

    def test_get_mtt_cth_rcth(self) -> None:
        """Direct convenience methods match analytical formulas."""
        model = VascularModel()
        alpha, beta = 4.0, 2.5

        assert model.get_mtt(alpha, beta) == pytest.approx(10.0, rel=1e-10)
        assert model.get_cth(alpha, beta) == pytest.approx(
            2.5 * np.sqrt(4.0), rel=1e-10
        )
        assert model.get_rcth(alpha) == pytest.approx(
            1.0 / np.sqrt(4.0), rel=1e-10
        )

    def test_validate_params(self) -> None:
        """Parameters within bounds are valid; out-of-bounds are not."""
        model = VascularModel()
        assert model.validate_params([0.08, 3.0, 2.0, 0.0]) is True
        # CBF bounds are (0.0, 1.0) — negative CBF is out of bounds
        assert model.validate_params([-1.0, 3.0, 2.0, 0.0]) is False

    def test_get_parameter_names(self) -> None:
        """Parameter names are returned in expected order."""
        model = VascularModel()
        assert model.get_parameter_names() == ["cbf", "alpha", "beta", "delay"]

    def test_residue_function_is_delay_agnostic(self) -> None:
        """The residue SHAPE does not depend on the delay parameter.

        Delay is an arterial-input arrival property, applied in tissue_curve,
        not in the residue. residue_function([..., delta]) must equal
        residue_function([..., 0]) for any delta.
        """
        model = VascularModel()
        t = np.arange(0, 60, 1.5, dtype=np.float64)
        r0 = model.residue_function([0.08, 3.0, 2.0, 0.0], t)
        for delay in (3.0, -2.0, 7.5):
            r = model.residue_function([0.08, 3.0, 2.0, delay], t)
            assert np.allclose(r, r0), f"residue shape changed at delay={delay}"

    def test_fitted_residue_cbv_independent_of_delay(self) -> None:
        """compute_fitted_residue integrates to a delay-independent CBV.

        Guards the historical footgun where a delay-shifted residue inflated
        the integral by cbf*delay (a flat-at-CBF prefix).
        """
        model = VascularModel()
        dt = 1.5
        t = np.arange(0, 90, dt, dtype=np.float64)
        cbf, alpha, beta = 0.08, 3.0, 2.0
        cbv0 = float(
            np.trapezoid(
                model.compute_fitted_residue([cbf, alpha, beta, 0.0], t), dx=dt
            )
        )
        for delay in (3.0, 6.0):
            cbv = float(
                np.trapezoid(
                    model.compute_fitted_residue(
                        [cbf, alpha, beta, delay], t
                    ),
                    dx=dt,
                )
            )
            assert cbv == pytest.approx(cbv0, rel=1e-10), (
                f"fitted_residue CBV changed at delay={delay} "
                f"(footgun would add cbf*delay={cbf * delay:.3f})"
            )

    def test_aif_upsampling_improves_forward_model_accuracy(self) -> None:
        """8x AIF upsampling brings tissue_curve closer to continuous truth.

        Can-fail in the upsampling's favour: ground-truth tissue is the
        convolution on a 16x-finer grid (continuous-time proxy) sampled to
        native dt. Production tissue_curve (8x internal upsampling) must be
        closer to that truth than a plain native-dt convolution. If
        tissue_curve did NOT upsample, c_model == c_native and this fails.
        """
        from scipy.signal import fftconvolve

        model = VascularModel()
        dt = 1.5          # clinical TR
        n = 60
        t = np.arange(n) * dt
        params = [0.08, 3.0, 2.0, 0.0]   # delay=0 isolates the convolution

        def aif_fn(tt: np.ndarray) -> np.ndarray:
            x = np.maximum(tt - 9.0, 0.0)   # sharp bolus arriving at 9 s
            a = (x ** 3) * np.exp(-x / 1.2)
            return np.asarray(a / (a.max() + 1e-12), dtype=np.float64)

        # Continuous-time truth at 16x (finer than the model's 8x).
        factor = 16
        n_fine = (n - 1) * factor + 1
        t_fine = np.linspace(0.0, float(t[-1]), n_fine)
        truth_fine = fftconvolve(
            aif_fn(t_fine), model.residue_function(params, t_fine), mode="full"
        )[:n_fine] * (dt / factor)
        truth = truth_fine[::factor]     # native grid, n points

        # Production (8x) vs plain native-dt convolution.
        aif_native = aif_fn(t)
        c_model = model.tissue_curve(params, aif_native, t)
        c_native = fftconvolve(
            aif_native, model.residue_function(params, t), mode="full"
        )[:n] * dt

        err_model = float(np.max(np.abs(c_model - truth)))
        err_native = float(np.max(np.abs(c_native - truth)))
        assert err_model < err_native, (
            f"upsampled forward model not closer to truth "
            f"(8x err={err_model:.4g} vs native err={err_native:.4g})"
        )
