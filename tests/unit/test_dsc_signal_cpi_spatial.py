"""Row-specific leakage priors through the shared spatial-VB schedule."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import DSCBatchedFitResult, batched_dsc_vi_fit
from pmriprep.models.dsc.signal_cpi import spatial
from pmriprep.models.dsc.signal_cpi.probability import GaussianPrior
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter, DSCVIPosterior


def linear_arguments() -> dict[str, Any]:
    """Isolate the spatial dispatch contract with an exactly differentiable model."""
    design = np.column_stack((np.linspace(.2, 2., 12), np.ones(12)))
    observations = np.array([[2., 3.], [2.1, 3.2], [2.3, 3.4]]) @ design.T
    observations += np.random.default_rng(882).normal(0, .2, observations.shape)
    return dict(model_type="linear_spatial_check", tissue_curves=observations,
        aif=np.ones(12), time_points=np.arange(12.),
        prior_means_nat=np.array([1., 1.]), prior_stds_nat=np.array([2., 2.]),
        transforms=[TransformType.IDENTITY]*2, upper_bounds=[None]*2,
        fitting_param_names=["cbf", "leakage"], fitting_bounds=[(0., 10.), (-10., 10.)],
        prior_means_transformed=np.array([1., 1.]), prior_stds_transformed=np.array([2., 2.]),
        nlls_seeds=np.array([[1.8, 2.8], [2., 3.], [2.2, 3.2]]),
        model_adapter=DSCVIModelAdapter(lambda m: m @ design.T,
            lambda m: np.broadcast_to(design, (len(m), *design.shape)).copy()),
        save_fitted_curves=True, return_full_posterior=True,
    )


def coordinates(connected: bool = True) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[int, int, int]]:
    """Declare graph geometry explicitly, independently of row order."""
    x = np.arange(3) * (1 if connected else 2)
    return (x, np.zeros(3, int), np.zeros(3, int)), (5, 1, 1)


@pytest.mark.parametrize("connected", [False, True])
def test_equal_prior_dispatch_preserves_shared_wrapper(connected: bool) -> None:
    """The callback is bookkeeping, not a replacement spatial algorithm."""
    args = linear_arguments()
    coords, shape = coordinates(connected)
    expected = batched_dsc_spatial_vi_fit(**args, mask_coords=coords, spatial_shape=shape)
    dispatch = spatial.RowPriorDispatcher(np.tile([1., 1.], (3, 1)), np.full((3, 2), 2.),
                                           keep_full_posterior=True)
    actual = batched_dsc_spatial_vi_fit(**args, mask_coords=coords, spatial_shape=shape,
                                      local_fit=dispatch)
    assert actual.full_posterior is not None and expected.full_posterior is not None
    for name in ("mean", "covariance", "noise_shape", "noise_scale"):
        np.testing.assert_allclose(getattr(actual.full_posterior, name),
                                   getattr(expected.full_posterior, name), rtol=1e-8, atol=1e-9)
    assert dispatch.calls == (11 if connected else 1)


@pytest.mark.parametrize("weighted", [False, True])
def test_mixed_priors_survive_every_sweep_and_row_permutation(weighted: bool) -> None:
    """Only CBF's prior changes; row-specific signed leakage guidance survives."""
    args = linear_arguments()
    if weighted:
        args["weights"] = np.linspace(.1, 1., 12)
    means = np.array([[1., -2.], [1., 0.], [1., 3.]])
    stds = np.array([[2., .7], [2., 1.2], [2., 2.1]])
    coords, shape = coordinates()
    records: list[tuple[int, np.ndarray, np.ndarray]] = []

    def capture(i: int, result: DSCBatchedFitResult, prior: GaussianPrior) -> None:
        assert prior.mean[1] == pytest.approx(means[i, 1])
        assert prior.covariance[1, 1] == pytest.approx(stds[i, 1]**2)
        records.append((i, prior.mean.copy(), np.diag(prior.covariance)))

    first = spatial.RowPriorDispatcher(means, stds, capture, keep_full_posterior=True)
    result = batched_dsc_spatial_vi_fit(**args, mask_coords=coords, spatial_shape=shape, local_fit=first)
    assert len(records) == 33
    assert any(not np.isclose(cov[0], 4.) for _, _, cov in records[3:])
    permutation = np.array([2, 0, 1])
    second = spatial.RowPriorDispatcher(means[permutation], stds[permutation], keep_full_posterior=True)
    args["tissue_curves"] = args["tissue_curves"][permutation]
    args["nlls_seeds"] = args["nlls_seeds"][permutation]
    reordered = batched_dsc_spatial_vi_fit(**args, mask_coords=tuple(x[permutation] for x in coords),
                                          spatial_shape=shape, local_fit=second)
    assert result.full_posterior is not None and reordered.full_posterior is not None
    for name in ("mean", "covariance", "noise_shape", "noise_scale"):
        np.testing.assert_allclose(getattr(reordered.full_posterior, name)[np.argsort(permutation)],
                                   getattr(result.full_posterior, name), rtol=1e-8, atol=1e-9)


def test_no_edges_match_identical_vi_schedule() -> None:
    """Compare ten-update warm starts, not an unequal 100-update endpoint."""
    args = linear_arguments()
    means = np.array([[1., -2.], [1., 0.], [1., 3.]])
    stds = np.array([[2., .7], [2., 1.2], [2., 2.1]])
    dispatcher = spatial.RowPriorDispatcher(means, stds, keep_full_posterior=True)
    coords, shape = coordinates(False)
    actual = batched_dsc_spatial_vi_fit(**args, mask_coords=coords, spatial_shape=shape,
                                       local_fit=dispatcher)
    expected = []
    for i in range(3):
        row = args | dict(tissue_curves=args["tissue_curves"][i:i+1],
            nlls_seeds=args["nlls_seeds"][i:i+1], prior_means_nat=means[i], prior_stds_nat=stds[i],
            prior_means_transformed=means[i], prior_stds_transformed=stds[i], return_posteriors=True)
        expected.append(batched_dsc_vi_fit(**row))
    joined = spatial._join_results(expected)
    assert actual.full_posterior is not None and joined.full_posterior is not None
    for name in ("mean", "covariance", "noise_shape", "noise_scale"):
        np.testing.assert_array_equal(getattr(actual.full_posterior, name), getattr(joined.full_posterior, name))


def test_join_preserves_row_failure_reasons() -> None:
    """Spatial row assembly cannot discard numerical-failure provenance."""
    args = linear_arguments()
    rows = []
    for i, reason in enumerate(("local_factor_failed", None, None)):
        row_args = args | {
            "tissue_curves": args["tissue_curves"][i:i + 1],
            "nlls_seeds": args["nlls_seeds"][i:i + 1],
        }
        result = batched_dsc_vi_fit(**row_args)
        rows.append(replace(result, failure_reasons=(reason,)))
    assert spatial._join_results(rows).failure_reasons == (
        "local_factor_failed", None, None,
    )


def test_dispatcher_carries_finite_state_past_failed_spatial_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected SVB row cannot poison later spatial sweeps with NaNs."""
    args = linear_arguments()
    row_args = args | {
        "tissue_curves": args["tissue_curves"][:1],
        "nlls_seeds": args["nlls_seeds"][:1],
    }
    failed = batched_dsc_vi_fit(**row_args)
    assert failed.full_posterior is not None
    failed = replace(
        failed,
        posterior_means_transformed=np.full((1, 2), np.nan),
        posterior_cov_diag=np.full((1, 2), np.nan),
        full_posterior=DSCVIPosterior(
            np.full((1, 2), np.nan), np.full((1, 2, 2), np.nan),
            np.array([np.nan]), np.array([np.nan]),
        ),
        failure_reasons=("local_factor_failed",),
    )
    monkeypatch.setattr(spatial, "batched_dsc_vi_fit", lambda **kwargs: failed)
    dispatcher = spatial.RowPriorDispatcher(
        np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]]),
        keep_full_posterior=True,
    )
    result = dispatcher(**row_args)
    assert result.failure_reasons == ("local_factor_failed",)
    assert result.posterior_means_transformed is not None
    assert result.posterior_cov_diag is not None
    assert np.isfinite(result.posterior_means_transformed).all()
    assert np.isfinite(result.posterior_cov_diag).all()
    assert result.full_posterior is not None
    assert np.isfinite(result.full_posterior.covariance).all()
