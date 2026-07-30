"""Unit tests for Bayesian DSC fitting (MAP, VI, dispatch)."""

import numpy as np
import pytest

from pmriprep.models.dsc.bayesian import (
    BayesianDSCFitter,
    fit_dsc_model,
)
from pmriprep.models.dsc.vascular import VascularModel


def _make_synthetic_data(
    cbf: float = 0.08,
    alpha: float = 3.0,
    beta: float = 2.0,
    delay: float = 0.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic vascular model tissue curve with noise.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (concentration, aif, time_points).
    """
    n = 40
    dt = 1.5
    t = np.arange(n, dtype=np.float64) * dt

    # Gamma-variate AIF
    aif = (t / 3.0) * np.exp(-t / 5.0)
    aif[0] = 0.0

    # Generate tissue curve from model
    model = VascularModel()
    params = [cbf, alpha, beta, delay]
    tissue = model.tissue_curve(params, aif, t)

    # Add noise
    rng = np.random.default_rng(seed)
    tissue += rng.normal(0, noise_std, n)

    return tissue, aif, t


class TestBayesianConstruction:
    """Tests for BayesianDSCFitter initialization."""

    def test_priors_match_model(self) -> None:
        """Number of priors matches model parameter count."""
        model = VascularModel()
        fitter = BayesianDSCFitter(model)
        assert len(fitter.priors) == model.n_parameters

    def test_prior_mean_std(self) -> None:
        """Prior means match model fitting-space parameter definitions."""
        model = VascularModel()
        fitter = BayesianDSCFitter(model)
        definitions = model.get_fitting_parameter_definitions()
        for i, defn in enumerate(definitions):
            assert fitter.priors[i].mean == pytest.approx(defn.get_prior_mean())
            assert fitter.priors[i].std == pytest.approx(defn.get_prior_sigma())

    def test_vascular_cbf_prior_is_weak_without_widening_other_priors(self) -> None:
        """Only log-CBF uses the weak regularizing transformed prior."""
        model = VascularModel()
        priors = BayesianDSCFitter(model).priors
        names = [
            definition.name
            for definition in model.get_fitting_parameter_definitions()
        ]
        transformed_stds = {
            name: prior.std_transformed
            for name, prior in zip(names, priors)
        }

        assert transformed_stds["cbf"] == 5.0
        assert transformed_stds["mtt"] == 1.0
        assert transformed_stds["alpha"] == 1.0
        assert transformed_stds["delay"] == 1.5
        assert model.get_parameter_definitions()[0].transformed_prior_sigma == 5.0


class TestLogPosteriorComponents:
    """Tests for log prior, likelihood, and posterior computation."""

    def test_log_prior_at_mean_maximum(self) -> None:
        """Log prior is highest at the prior means."""
        model = VascularModel()
        fitter = BayesianDSCFitter(model)
        means = np.array([p.mean for p in fitter.priors], dtype=np.float64)

        log_prior_at_mean = fitter._log_prior(means)

        # Evaluate at a point far from the means (still within bounds)
        offset = means.copy()
        offset[0] = 0.5  # CBF far from typical center
        offset[1] = 0.6  # alpha far from typical center
        log_prior_at_offset = fitter._log_prior(offset)

        assert log_prior_at_mean > log_prior_at_offset

    def test_log_likelihood_good_fit(self) -> None:
        """Log likelihood is higher for data matching the model."""
        model = VascularModel()
        fitter = BayesianDSCFitter(model)
        concentration, aif, t = _make_synthetic_data(noise_std=0.01)

        true_params = np.array([0.08, 3.0, 2.0, 0.0], dtype=np.float64)
        noise_std = 0.5

        # Good fit (true params)
        ll_good = fitter._log_likelihood(true_params, concentration, aif, t, noise_std)

        # Bad fit (wrong params)
        bad_params = np.array([0.5, 8.0, 0.5, 3.0], dtype=np.float64)
        ll_bad = fitter._log_likelihood(bad_params, concentration, aif, t, noise_std)

        assert ll_good > ll_bad

    def test_log_likelihood_extreme_residual_returns_negative_infinity(
        self,
    ) -> None:
        """Overflow-sized residuals are rejected without a runtime warning."""
        model = VascularModel()
        fitter = BayesianDSCFitter(model)
        concentration, aif, t = _make_synthetic_data(noise_std=0.01)
        concentration[:] = np.finfo(np.float64).max
        params = np.array([0.08, 3.0, 2.0, 0.0], dtype=np.float64)

        with np.errstate(over="raise", invalid="raise"):
            likelihood = fitter._log_likelihood(
                params, concentration, aif, t, noise_std=0.5
            )

        assert likelihood == float("-inf")

    def test_log_posterior_sum(self) -> None:
        """Log posterior equals log prior + log likelihood."""
        model = VascularModel()
        fitter = BayesianDSCFitter(model)
        concentration, aif, t = _make_synthetic_data(noise_std=0.01)

        params = np.array([0.08, 3.0, 2.0, 0.0], dtype=np.float64)
        noise_std = 0.5

        log_prior = fitter._log_prior(params)
        log_lik = fitter._log_likelihood(params, concentration, aif, t, noise_std)
        log_post = fitter._log_posterior(params, concentration, aif, t, noise_std)

        assert log_post == pytest.approx(log_prior + log_lik, rel=1e-10)


class TestMAPFitting:
    """Tests for MAP estimation."""

    def test_fit_map_synthetic(self) -> None:
        """MAP recovers CBF within 50% from synthetic vascular data."""
        model = VascularModel()
        concentration, aif, t = _make_synthetic_data(
            cbf=0.08, alpha=3.0, beta=2.0, delay=0.0, noise_std=0.1
        )

        fitter = BayesianDSCFitter(model)
        result = fitter.fit_map(concentration, aif, t, compute_uncertainties=False)

        assert result.fitting_method == "map"
        fitted_cbf = result.parameters["cbf"]
        assert fitted_cbf == pytest.approx(0.08, rel=0.5), (
            f"Fitted CBF={fitted_cbf}, expected ~0.08"
        )

    def test_fit_map_uncertainties(self) -> None:
        """MAP with compute_uncertainties=True returns non-empty uncertainties."""
        model = VascularModel()
        concentration, aif, t = _make_synthetic_data(noise_std=0.1)

        fitter = BayesianDSCFitter(model)
        result = fitter.fit_map(concentration, aif, t, compute_uncertainties=True)

        assert len(result.uncertainties) > 0
        assert result.uncertainty_source == "laplace_hessian"
        for name, val in result.uncertainties.items():
            assert val >= 0.0, f"Uncertainty for {name} is negative: {val}"

    def test_fit_map_hessian_uses_transformed_prior_units(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scalar MAP differentiates its optimized transformed objective."""
        model = VascularModel()
        concentration, aif, t = _make_synthetic_data(noise_std=0.1)
        fitter = BayesianDSCFitter(model)
        original = fitter._compute_hessian
        observed_stds: list[np.ndarray] = []

        def _record_hessian(
            params: np.ndarray,
            objective: object,
            relative_epsilon: float = 1e-5,
            prior_stds: np.ndarray | None = None,
        ) -> np.ndarray:
            assert callable(objective)
            assert prior_stds is not None
            observed_stds.append(prior_stds.copy())
            return original(
                params,
                objective,  # type: ignore[arg-type]
                relative_epsilon,
                prior_stds,
            )

        monkeypatch.setattr(fitter, "_compute_hessian", _record_hessian)
        fitter.fit_map(concentration, aif, t, compute_uncertainties=True)

        assert len(observed_stds) == 1
        assert observed_stds[0] == pytest.approx(
            [prior.std_transformed for prior in fitter.priors]
        )

    def test_fit_map_marks_uncertainty_undefined_when_hessian_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A singular Hessian does not publish prior width as posterior SD."""
        model = VascularModel()
        concentration, aif, t = _make_synthetic_data(noise_std=0.1)
        fitter = BayesianDSCFitter(model)

        def _unavailable_hessian(*args: object, **kwargs: object) -> np.ndarray:
            del args, kwargs
            return np.full((4, 4), np.nan, dtype=np.float64)

        monkeypatch.setattr(fitter, "_compute_hessian", _unavailable_hessian)
        result = fitter.fit_map(concentration, aif, t, compute_uncertainties=True)

        assert np.isnan(result.uncertainties["cbf"])
        assert result.credible_intervals == {}
        assert result.uncertainty_source == "uncertainty_unavailable"

    def test_fit_map_flat_concentration(self) -> None:
        """All-zero concentration yields CBF near zero."""
        model = VascularModel()
        n = 40
        t = np.arange(n, dtype=np.float64) * 1.5
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0
        concentration = np.zeros(n, dtype=np.float64)

        fitter = BayesianDSCFitter(model)
        result = fitter.fit_map(concentration, aif, t, compute_uncertainties=False)

        fitted_cbf = result.parameters["cbf"]
        assert fitted_cbf < 0.05, f"CBF should be near 0 for flat data, got {fitted_cbf}"


class TestMCMCPriorCoordinates:
    """Characterize the common transformed-space Bayesian prior."""

    def test_mcmc_initializes_log_cbf_in_transformed_space(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        concentration, aif, t = _make_synthetic_data(noise_std=0.1)
        fitter = BayesianDSCFitter(VascularModel())
        captured: dict[str, object] = {}

        def _capture_sampler(
            log_prob: object,
            init_positions: np.ndarray,
            n_iterations: int,
            bounds: list[tuple[float, float]],
        ) -> np.ndarray:
            assert callable(log_prob)
            captured["positions"] = init_positions.copy()
            captured["bounds"] = bounds
            return np.repeat(init_positions[None, :, :], n_iterations, axis=0)

        monkeypatch.setattr(
            "pmriprep.models.dsc.bayesian.run_ensemble_sampler", _capture_sampler
        )
        fitter.fit_mcmc(concentration, aif, t, sloppy=True)

        positions = captured["positions"]
        assert isinstance(positions, np.ndarray)
        assert np.all(positions[:, 0] < 0.0)  # log(CBF), not natural CBF
        assert captured["bounds"][0] == (-50.0, 50.0)  # type: ignore[index]


class TestVIFitting:
    """Tests for Variational Inference fitting."""

    def test_fit_vi_synthetic(self) -> None:
        """VI recovers CBF within 50% from synthetic vascular data."""
        model = VascularModel()
        concentration, aif, t = _make_synthetic_data(
            cbf=0.08, alpha=3.0, beta=2.0, delay=0.0, noise_std=0.1
        )

        fitter = BayesianDSCFitter(model)
        result = fitter.fit_vi(concentration, aif, t, n_iterations=20)

        assert result.fitting_method == "vi"
        fitted_cbf = result.parameters["cbf"]
        assert fitted_cbf == pytest.approx(0.08, rel=0.5), (
            f"VI fitted CBF={fitted_cbf}, expected ~0.08"
        )

    def test_fit_vi_has_uncertainties(self) -> None:
        """VI returns uncertainties dict with entries."""
        model = VascularModel()
        concentration, aif, t = _make_synthetic_data(noise_std=0.1)

        fitter = BayesianDSCFitter(model)
        result = fitter.fit_vi(concentration, aif, t, n_iterations=20)

        assert len(result.uncertainties) > 0


class TestDispatch:
    """Tests for fit_dsc_model convenience function."""

    def test_fit_dsc_model_map(self) -> None:
        """fit_dsc_model with method='map' returns MAP result."""
        model = VascularModel()
        concentration, aif, t = _make_synthetic_data(noise_std=0.1)

        result = fit_dsc_model(
            model, concentration, aif, t, method="map",
            compute_uncertainties=False,
        )
        assert result.fitting_method == "map"

    def test_fit_dsc_model_invalid(self) -> None:
        """fit_dsc_model with invalid method raises ValueError."""
        model = VascularModel()
        concentration, aif, t = _make_synthetic_data(noise_std=0.1)

        with pytest.raises(ValueError, match="Unknown fitting method"):
            fit_dsc_model(model, concentration, aif, t, method="invalid")
