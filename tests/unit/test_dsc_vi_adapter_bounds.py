"""Correlated mean constraints for models using the shared DSC VI adapter."""

from typing import Any

import numpy as np
import pytest

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter


def _linear_problem(upper_active: bool, weighted: bool) -> dict[str, Any]:
    """Build an exact linear observation model with correlated coordinates."""
    precision = np.array([[1.0, 0.9], [0.9, 1.0]])
    jacobian = np.tile(np.linalg.cholesky(precision - 0.01 * np.eye(2)).T, (20, 1))
    start = np.array([1.0, 0.0]) if upper_active else np.zeros(2)
    truth = np.array([11.0, -5.0]) if upper_active else np.array([-10.0, 5.0])
    observations = 10.0 + jacobian @ truth
    adapter = DSCVIModelAdapter(
        lambda means: 10.0 + means @ jacobian.T,
        lambda means: np.broadcast_to(jacobian, (len(means), *jacobian.shape)),
    )
    return dict(
        model_type="linear_adapter_test",
        tissue_curves=observations[None],
        aif=np.ones(len(observations)),
        time_points=np.arange(len(observations), dtype=float),
        prior_means_nat=np.zeros(2),
        prior_stds_nat=np.full(2, 10.0),
        transforms=[TransformType.IDENTITY] * 2,
        upper_bounds=[None] * 2,
        fitting_param_names=["cbf", "coordinate"],
        fitting_bounds=[(0.0, 1.0), (-100.0, 100.0)],
        prior_means_transformed=np.zeros(2),
        prior_stds_transformed=np.full(2, 10.0),
        initial_means_transformed=start[None],
        model_adapter=adapter,
        weights=np.linspace(0.1, 1.0, len(observations)) if weighted else None,
        return_full_posterior=True,
        save_fitted_curves=True,
        n_iterations=100,
    )


@pytest.mark.parametrize("upper_active", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_correlated_bound_reoptimizes_free_coordinate(
    upper_active: bool, weighted: bool
) -> None:
    """A bound on one parameter must not leave other parameters at a bad start."""
    args = _linear_problem(upper_active, weighted)
    result = batched_dsc_vi_fit(**args)
    posterior = result.full_posterior
    assert posterior is not None and result.fitted_curves is not None
    mean = posterior.mean[0]
    assert mean[0] == pytest.approx(float(upper_active), abs=1e-12)
    assert mean[1] > 3.0 if upper_active else mean[1] < -3.0
    assert result.converged[0]

    weights = np.ones(40) if args["weights"] is None else args["weights"]
    jacobian = args["model_adapter"].jacobian(mean[None])[0]
    residual = args["tissue_curves"][0] - result.fitted_curves[0]
    precision = posterior.noise_shape[0] * posterior.noise_scale[0]
    gradient = precision * jacobian.T @ (weights * residual) - mean / 100.0
    assert abs(gradient[1]) < 1e-3
    assert gradient[0] > 0 if upper_active else gradient[0] < 0


def test_interior_and_bound_rows_share_one_batch() -> None:
    """Constraining one row leaves a feasible row's ordinary solution intact."""
    args = _linear_problem(False, False)
    interior = np.array([0.3, 0.7])
    args["tissue_curves"] = np.vstack(
        (args["tissue_curves"], args["model_adapter"].predict(interior[None]))
    )
    args["initial_means_transformed"] = np.zeros((2, 2))
    result = batched_dsc_vi_fit(**args)
    assert result.full_posterior is not None
    np.testing.assert_allclose(result.full_posterior.mean[1], interior, atol=1e-6)
    assert result.full_posterior.mean[0, 1] < -3.0
    assert np.all(result.converged)


@pytest.mark.parametrize("upper_active", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_spatial_conditional_update_uses_the_same_bound_solve(
    upper_active: bool, weighted: bool
) -> None:
    """A spatial conditional prior must not restore independent clipping."""
    args = _linear_problem(upper_active, weighted)
    args.update(
        n_iterations_override=1,
        initial_noise_precision=np.ones(1),
        spatial_prec_diag=np.array([[0.2, 0.0]]),
        spatial_eta=np.array([[0.1, 0.0]]),
        spatial_replaces_parameter_prior=True,
        spatial_parameter_mask=np.array([True, False]),
    )
    result = batched_dsc_vi_fit(**args)
    assert result.full_posterior is not None and result.fitted_curves is not None
    mean = result.full_posterior.mean[0]
    assert mean[0] == pytest.approx(float(upper_active), abs=1e-12)
    assert mean[1] > 3.0 if upper_active else mean[1] < -3.0
    weights = np.ones(40) if args["weights"] is None else args["weights"]
    jacobian = args["model_adapter"].jacobian(mean[None])[0]
    residual = args["tissue_curves"][0] - result.fitted_curves[0]
    gradient = jacobian.T @ (weights * residual) - np.array([0.2, 0.01]) * mean
    gradient[0] += 0.1
    assert abs(gradient[1]) < 1e-8
    assert gradient[0] > 0 if upper_active else gradient[0] < 0
