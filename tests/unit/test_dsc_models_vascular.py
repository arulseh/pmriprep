"""Unit tests for DSC vascular model and base dataclasses."""

import numpy as np
import pytest

from pmriprep.models.dsc.base import DSCFitResult, DSCModelParameter
from pmriprep.models.dsc.vascular import VascularModel
from pmriprep.utils.dsc_constants import (
    MTT_BOUNDS,
    VASCULAR_ALPHA_BOUNDS,
    VASCULAR_BETA_BOUNDS,
)


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

    def test_missing_uncertainty_is_not_zero(self) -> None:
        """An unavailable uncertainty must not imply infinite precision."""
        assert np.isnan(DSCFitResult().get_uncertainty("cbf"))

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
        # Negative flow is outside the relative-amplitude domain.
        assert model.validate_params([-1.0, 3.0, 2.0, 0.0]) is False

    @pytest.mark.parametrize("mtt", [0.5, 5.0, 60.0])
    @pytest.mark.parametrize("alpha", [0.5, 3.0, 100.0])
    def test_derived_beta_envelope_contains_fitting_domain(
        self, mtt: float, alpha: float
    ) -> None:
        """Marginal model metadata must not exclude supported fitting points."""
        model = VascularModel()
        definitions = model.get_fitting_parameter_definitions()
        assert definitions[1].bounds == MTT_BOUNDS == (0.5, 60.0)
        assert definitions[2].bounds == VASCULAR_ALPHA_BOUNDS == (0.5, 100.0)
        assert VASCULAR_BETA_BOUNDS == (0.005, 120.0)
        fitting = np.array([0.06, mtt, alpha, 2.0])
        natural = model.fitting_to_model_params(fitting)
        assert model.validate_params(natural)
        np.testing.assert_allclose(model.model_to_fitting_params(natural), fitting)
        np.testing.assert_allclose(model.clip_params(natural), natural)
        assert natural[2] == pytest.approx(mtt / alpha)

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

    @pytest.mark.parametrize(
        "cbf,alpha,beta",
        [(0.08, 3.0, 1.3), (0.10, 4.0, 1.5), (0.04, 2.5, 1.9)],
    )
    def test_forward_matches_fine_truth(
        self, cbf: float, alpha: float, beta: float
    ) -> None:
        """The production Volterra forward matches the dt/100 truth in area AND shape.

        Locks in the accuracy of the shared Sourbron (1, 4, 1)/6 operator on the
        vascular forward. On a realistically-sampled arterial bolus (FWHM ~9 s,
        ~6 samples at TR 1.5 s) the measured AUC error is 0.4-0.5% and the max
        pointwise error is 0.7-1.3% of peak. AUC is checked at 1.0% (between the
        Volterra floor and the ~1.5-2.2% over-read of the rectangular
        product-integration it replaced, so it is can-fail against that
        regression without being brittle); the shape is checked at 2% so a
        regression that preserves total area but distorts the curve (peak-timing
        or kernel error) also fails. The ratios are scale-free in CBF, so the
        shape parameters set the error.
        """
        model = VascularModel()
        dt, n = 1.5, 60
        t = np.arange(n) * dt
        params = [cbf, alpha, beta, 0.0]   # delay=0 isolates the convolution

        def aif_fn(tt: np.ndarray) -> np.ndarray:
            x = np.maximum(tt - 10.0, 0.0)   # arterial bolus, realistic width
            a = (x ** 3) * np.exp(-x / 2.5)
            return np.asarray(a / (a.max() + 1e-12), dtype=np.float64)

        aif = aif_fn(t)
        # dt/100 rectangular convolution converges to the continuous integral
        # (~0.02% residual), so it is a valid continuous-truth proxy.
        oversample = 100
        t_fine = np.arange(n * oversample) * (dt / oversample)
        truth = np.convolve(
            aif_fn(t_fine), model.residue_function(params, t_fine)
        )[: n * oversample][::oversample] * (dt / oversample)

        c_model = model.tissue_curve(params, aif, t)
        auc_true = float(np.trapezoid(truth, dx=dt))
        rel = (float(np.trapezoid(c_model, dx=dt)) - auc_true) / auc_true
        assert abs(rel) < 0.01, f"vascular forward AUC error {rel:+.2%} exceeds 1%"

        # Shape guard: an area-preserving distortion (peak-timing/kernel error)
        # must still fail. Measured max pointwise error is 0.7-1.3% of peak.
        shape_err = float(np.max(np.abs(c_model - truth))) / float(truth.max())
        assert shape_err < 0.02, f"vascular forward shape error {shape_err:.2%} exceeds 2%"
