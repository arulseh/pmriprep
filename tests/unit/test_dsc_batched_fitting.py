"""Shared DSC adapter inference contracts."""

from typing import Any

import numpy as np
import pytest

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter


@pytest.mark.parametrize("model_name", ["vascular", "cpi", "unknown"])
def test_model_requires_explicit_adapter(model_name: str) -> None:
    """An identifier cannot select a built-in concentration solver."""
    with pytest.raises(ValueError, match="without an adapter"):
        batched_dsc_vi_fit(
            model_name, np.empty((0, 2)), np.ones(2), np.arange(2.0),
            np.array([0.1]), np.array([0.01]), [], [], ["cbf"], [(0.0, 2.0)],
        )


def linear_arguments(rows: int = 3) -> dict[str, Any]:
    """Exactly differentiable observations for generic engine invariants."""
    time = np.linspace(0, 1, 20)
    design = np.column_stack((time, np.ones_like(time)))
    truth = np.column_stack((np.linspace(.05, .1, rows), np.full(rows, .02)))
    observed = truth @ design.T + np.random.default_rng(731).normal(0, .001, (rows, 20))
    return dict(model_type="linear", tissue_curves=observed, aif=np.ones(20),
        time_points=time, prior_means_nat=np.array([.06,.02]), prior_stds_nat=np.ones(2),
        transforms=[TransformType.IDENTITY]*2, upper_bounds=[None]*2,
        fitting_param_names=["cbf","offset"], fitting_bounds=[(0,1),(-1,1)],
        model_adapter=DSCVIModelAdapter(lambda m: m @ design.T,
            lambda m: np.broadcast_to(design,(len(m),*design.shape)).copy()),
        save_fitted_curves=True, return_full_posterior=True)


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_adapter_results_and_empty_batch(rows: int) -> None:
    """Finite fits keep curves, uncertainty and row shape; empty batches stay empty."""
    result = batched_dsc_vi_fit(**linear_arguments(rows))
    assert result.params["cbf"].shape == (rows,)
    assert result.full_posterior is not None
    assert result.full_posterior.mean.shape == (rows, 2)
    if rows:
        np.testing.assert_allclose(result.params["cbf"], np.linspace(.05,.1,rows), rtol=.05)
        assert np.all(result.r_squared > .95)
        assert np.all(np.isfinite(result.uncertainties["cbf"]))
        assert result.fitted_curves.shape == (rows, 20)


def test_fractional_weights_reduce_information() -> None:
    """Uniform half-weights widen the conditional CBF uncertainty."""
    args = linear_arguments()
    full = batched_dsc_vi_fit(**args, weights=np.ones(20))
    half = batched_dsc_vi_fit(**args, weights=np.full(20,.5))
    assert np.all(half.uncertainties["cbf"] > full.uncertainties["cbf"])


def test_adapter_parallel_dispatch_is_explicit() -> None:
    """Adapter callers own multiprocessing rather than silently changing the fit."""
    with pytest.raises(ValueError, match="nprocs=1"):
        batched_dsc_vi_fit(**linear_arguments(), nprocs=2)


def test_bounded_mean_failure_is_isolated_to_one_voxel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One row's numerical bound failure does not discard the batch."""
    from pmriprep.models.bayesian_utils import TransformType
    from pmriprep.models.dsc import batched_fitting
    from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter

    original = batched_fitting._bounded_vi_mean
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("forced bounded solve failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(batched_fitting, "_bounded_vi_mean", fail_once)
    time = np.arange(6.0)
    observations = np.vstack((np.full(6, 10.0), np.full(6, 8.0)))
    adapter = DSCVIModelAdapter(
        predict=lambda means: np.repeat(means[:, :1], len(time), axis=1),
        jacobian=lambda means: np.ones((len(means), len(time), 1)),
        valid_mean=lambda means: np.all((means >= 0) & (means <= 1), axis=1),
    )
    result = batched_dsc_vi_fit(
        "test_adapter", observations, np.ones_like(time), time,
        np.array([0.5]), np.array([1.0]), [TransformType.IDENTITY], [None],
        ["cbf"], [(0.0, 1.0)], initial_means_transformed=np.full((2, 1), 0.5),
        n_iterations=2, model_adapter=adapter, return_full_posterior=True,
        save_fitted_curves=True,
    )

    assert calls >= 2
    assert result.failure_reasons == ("bounded_mean_failed", None)
    assert np.isnan(result.params["cbf"][0])
    assert np.isfinite(result.params["cbf"][1])
    assert not result.converged[0]
    assert result.full_posterior is not None
    assert np.isnan(result.full_posterior.mean[0]).all()
    assert np.isfinite(result.full_posterior.mean[1]).all()
