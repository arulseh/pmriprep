"""Independent identities and accepted-state tests for isolated shared VI."""

from typing import Any

import numpy as np
import pytest
from scipy.optimize import brentq

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit
from pmriprep.models.dsc.local_vi import local_vi_state
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter
from pmriprep.models.dsc.signal_cpi.probability import GaussianPrior, GammaPrecision, local_free_energy


@pytest.mark.parametrize("observations,parameters", [(8, 3), (3, 5), (8, 1)])
@pytest.mark.parametrize("weighted", [False, True])
def test_local_factors_match_direct_covariance_and_scalar_root(
    observations: int, parameters: int, weighted: bool
) -> None:
    """Check rank-deficient and weighted cases without sharing the SVD algebra."""
    rng = np.random.default_rng(71)
    jacobian = rng.normal(size=(2, observations, parameters))
    residual = rng.normal(size=(2, observations))
    mean = rng.normal(size=(2, parameters))
    prior_mean = rng.normal(size=parameters)
    root = rng.normal(size=(parameters, parameters))
    precision = root @ root.T + np.eye(parameters)
    weights = np.linspace(0, 1, observations) if weighted else np.ones(observations)
    state = local_vi_state(mean, jacobian, residual, prior_mean, precision, weights)
    shape = 1e-6 + weights.sum() / 2
    for i in range(2):
        gram = jacobian[i].T @ (weights[:, None] * jacobian[i])
        rss = float(residual[i] @ (weights * residual[i]))

        def equation(tau: float) -> float:
            cov = np.linalg.inv(precision + tau * gram)
            return float(tau * (1e-6 + 0.5 * (rss + np.trace(cov @ gram))) - shape)

        tau = brentq(equation, 0, shape / (1e-6 + 0.5 * rss), xtol=1e-13)
        expected = np.linalg.inv(precision + tau * gram)
        np.testing.assert_allclose(state.covariance[i], expected, rtol=1e-9, atol=1e-11)
        assert state.shape * state.scale[i] == pytest.approx(tau, rel=1e-9)
        if not weighted:
            score = local_free_energy(
                residual[i],
                np.zeros(observations),
                jacobian[i],
                mean[i],
                state.covariance[i],
                GaussianPrior(
                    prior_mean, np.asarray(np.linalg.inv(precision), dtype=np.float64)
                ),
                GammaPrecision(shape, 1 / state.scale[i]),
                GammaPrecision(1e-6, 1e-6),
            )
            assert state.score[i] == pytest.approx(score, abs=1e-9)


def inputs(nonlinear: bool = False, wrong_derivative: bool = False) -> dict[str, Any]:
    """One-dimensional model with transparent mean movement and exact scores."""
    times = np.arange(8.0)
    return dict(
        model_type="state_test",
        tissue_curves=np.ones((1, 8)),
        aif=np.ones(8),
        time_points=times,
        prior_means_nat=np.array([0.0]),
        prior_stds_nat=np.array([1000.0]),
        transforms=[TransformType.IDENTITY],
        upper_bounds=[None],
        fitting_param_names=["cbf"],
        fitting_bounds=[(-100.0, 100.0)],
        initial_means_transformed=np.array([[0.2 if nonlinear else 0.0]]),
        model_adapter=DSCVIModelAdapter(
            lambda m: np.broadcast_to(m**2 if nonlinear else m, (len(m), 8)).copy(),
            lambda m: np.broadcast_to(
                (
                    2 * m
                    if nonlinear
                    else np.full_like(m, -1.0 if wrong_derivative else 1.0)
                )[:, None, :],
                (len(m), 8, 1),
            ).copy(),
        ),
        return_full_posterior=True,
        save_fitted_curves=True,
    )


def assert_consistent(args: dict[str, Any], result: Any) -> None:
    """Compare both returned factors with a fresh calculation at the saved mean."""
    posterior = result.full_posterior
    assert posterior is not None
    model = args["model_adapter"]
    state = local_vi_state(
        posterior.mean,
        model.jacobian(posterior.mean),
        args["tissue_curves"] - model.predict(posterior.mean),
        args["prior_means_nat"],
        np.diag(1 / args["prior_stds_nat"] ** 2),
        np.ones(8),
    )
    np.testing.assert_allclose(posterior.covariance, state.covariance, rtol=1e-9)
    np.testing.assert_allclose(posterior.noise_scale, state.scale, rtol=1e-9)
    np.testing.assert_array_equal(result.fitted_curves, model.predict(posterior.mean))


def test_rejected_trials_preserve_complete_state_and_do_not_converge() -> None:
    """Deliberately wrong derivatives force every proposed step uphill."""
    args = inputs(wrong_derivative=True)
    result = batched_dsc_vi_fit(**args, n_iterations=10)
    np.testing.assert_array_equal(result.params["cbf"], [0.0])
    assert not result.converged[0]
    assert_consistent(args, result)


def test_backtracked_step_has_its_own_factors_at_budget_exit() -> None:
    """A nonlinear overshoot must shrink without keeping its noise/covariance."""
    args = inputs(nonlinear=True)
    result = batched_dsc_vi_fit(**args, n_iterations=1)
    assert 0.2 < result.params["cbf"][0] < 2.5
    assert not result.converged[0]
    assert_consistent(args, result)


def test_linear_fixed_point_remains_converged_with_consistent_factors() -> None:
    """The full unshortened mean step can establish ordinary convergence."""
    args = inputs()
    result = batched_dsc_vi_fit(**args, n_iterations=10)
    assert result.converged[0]
    assert result.params["cbf"][0] == pytest.approx(1, abs=1e-6)
    assert_consistent(args, result)
