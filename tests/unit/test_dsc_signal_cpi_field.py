"""A common CBF graph must preserve the shared spatial update exactly."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import DSCBatchedFitResult
from pmriprep.models.dsc.signal_cpi import acquisition, connection, field, spatial
from pmriprep.models.dsc.signal_cpi.probability import GaussianPrior
from pmriprep.models.dsc.signal_cpi.settings import CPIOptions
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter
from tests.unit.test_dsc_signal_cpi_spatial import coordinates, linear_arguments


def common_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """Expose only the CBF coordinates to the existing spatial wrapper."""
    common = args.copy()
    for name in ("prior_means_nat", "prior_stds_nat", "prior_means_transformed",
                 "prior_stds_transformed", "transforms", "upper_bounds",
                 "fitting_param_names", "fitting_bounds"):
        common[name] = common[name][:1]
    common["nlls_seeds"] = None
    common["model_adapter"] = None
    return common


def linear_adapter(design: np.ndarray) -> DSCVIModelAdapter:
    """Bind an observation matrix independently of the caller's row loop."""
    def predict(means: np.ndarray) -> np.ndarray:
        return np.asarray(means @ design.T)

    def jacobian(means: np.ndarray) -> np.ndarray:
        return np.broadcast_to(design, (len(means), *design.shape)).copy()

    return DSCVIModelAdapter(predict, jacobian)


@pytest.mark.parametrize("connected", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_common_field_is_identical_to_homogeneous_schedule(connected: bool, weighted: bool) -> None:
    args = linear_arguments()
    if weighted:
        args["weights"] = np.linspace(.3, 1., 12)
    coords, shape = coordinates(connected)
    means = np.array([[1., -2.], [1., 0.], [1., 3.]])
    stds = np.array([[2., .7], [2., 1.2], [2., 2.1]])
    expected_dispatch = spatial.RowPriorDispatcher(means, stds, keep_full_posterior=True)
    expected = batched_dsc_spatial_vi_fit(**args, mask_coords=coords, spatial_shape=shape,
                                         local_fit=expected_dispatch)
    local = []
    for i in range(3):
        local.append(args | dict(tissue_curves=args["tissue_curves"][i:i+1],
            nlls_seeds=args["nlls_seeds"][i:i+1], prior_means_nat=means[i], prior_stds_nat=stds[i],
            prior_means_transformed=means[i], prior_stds_transformed=stds[i]))
    states = {}

    def consume(i: int, result: DSCBatchedFitResult, prior: GaussianPrior) -> None:
        assert prior.mean[1] == pytest.approx(means[i, 1])
        assert prior.covariance[1, 1] == pytest.approx(stds[i, 1]**2)
        states[i] = result

    dispatch = field.CommonCBFDispatcher(local, consume)
    actual = batched_dsc_spatial_vi_fit(**common_arguments(args), mask_coords=coords,
                                        spatial_shape=shape, local_fit=dispatch)
    assert actual.posterior_means_transformed is not None
    assert actual.posterior_means_transformed.shape == (3, 1)
    assert expected.full_posterior is not None
    joined = spatial._join_results([states[i] for i in range(3)])
    assert joined.full_posterior is not None
    for name in ("mean", "covariance", "noise_shape", "noise_scale"):
        np.testing.assert_allclose(getattr(joined.full_posterior, name),
            getattr(expected.full_posterior, name), rtol=1e-8, atol=1e-9)
    assert dispatch.calls == expected_dispatch.calls == (11 if connected else 1)


def test_mixed_parameter_counts_keep_neighbors_and_row_ownership() -> None:
    args = linear_arguments()
    local, states = [], {}
    for i, p in enumerate((2, 3, 2)):
        design = np.column_stack((np.linspace(.2, 2., 12), np.ones(12), np.linspace(-1, 1, 12)**2))[:, :p]
        adapter = linear_adapter(design)
        mean, std = np.ones(p), np.full(p, 2.)
        local.append(args | dict(tissue_curves=args["tissue_curves"][i:i+1],
            nlls_seeds=mean[None], prior_means_nat=mean, prior_stds_nat=std,
            prior_means_transformed=mean, prior_stds_transformed=std,
            transforms=[TransformType.IDENTITY]*p, upper_bounds=[None]*p,
            fitting_param_names=["cbf"] + [f"local_{j}" for j in range(1, p)],
            fitting_bounds=[(0., 10.)] + [(-10., 10.)]*(p-1), model_adapter=adapter))
    coords, shape = coordinates()

    def consume(i: int, result: DSCBatchedFitResult, prior: GaussianPrior) -> None:
        states[i] = result
        np.testing.assert_allclose(prior.mean[1:], 1.)
        np.testing.assert_allclose(np.diag(prior.covariance)[1:], 4.)

    dispatch = field.CommonCBFDispatcher(local, consume)
    result = batched_dsc_spatial_vi_fit(**common_arguments(args), mask_coords=coords,
                                        spatial_shape=shape, local_fit=dispatch)
    assert dispatch.calls == 11
    for i, p in enumerate((2, 3, 2)):
        posterior = states[i].full_posterior
        assert posterior is not None and posterior.mean.shape == (1, p)
        assert dispatch.dispatchers[i].last_prior_variances[0, 0] != 4.
    permutation = np.array([2, 0, 1])
    reordered_args = common_arguments(args)
    reordered_args["tissue_curves"] = args["tissue_curves"][permutation]
    reordered = field.CommonCBFDispatcher([local[i] for i in permutation], lambda *args: None)
    second = batched_dsc_spatial_vi_fit(**reordered_args,
        mask_coords=tuple(x[permutation] for x in coords), spatial_shape=shape, local_fit=reordered)
    assert second.posterior_means_transformed is not None
    assert result.posterior_means_transformed is not None
    np.testing.assert_allclose(second.posterior_means_transformed[np.argsort(permutation)],
                               result.posterior_means_transformed, rtol=1e-9, atol=1e-10)


def test_actual_selected_cpi_field_matches_existing_single_family() -> None:
    time = np.arange(30.) * 3
    u = np.maximum(time - 9, 0)
    aif = 10 * u**2 * np.exp(-u/3)
    signals = (100 * np.exp(-.03 * np.stack([.2*u**2*np.exp(-u/5), .21*u**2*np.exp(-u/5)])))[:, None, None]
    a = acquisition.prepare_cpi_acquisition(signals, aif, time, .03,
        leakage_method="none", inference_method="svb",
        options=CPIOptions(dispersion="none", infer_delay=False, control_points=5))
    expected = spatial.fit_spatial_voxels(a.branches[0], keep_diagnostics=True)
    actual = field.fit_selected_field(a, keep_diagnostics=True)
    for before, after in zip(expected.voxels, actual.voxels, strict=True):
        assert after.status == before.status
        assert np.isnan(after.delay)
        assert after.family == "cpi" and len(after.selection_scores) == 1
        np.testing.assert_allclose([after.cbf, after.cbv, after.mtt],
                                   [before.cbf, before.cbv, before.mtt], rtol=1e-8, atol=1e-10)
        assert after.diagnostics is not None and before.diagnostics is not None
        for name in ("mean", "covariance", "noise_shape", "noise_scale", "prior_mean", "prior_covariance"):
            np.testing.assert_allclose(after.diagnostics.arrays[name], before.diagnostics.arrays[name],
                                       rtol=1e-8, atol=1e-10)


def test_empty_or_rejected_field_does_not_run_spatial_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    time = np.arange(30.) * 3
    signal = np.ones((1, 1, 1, 30)) * 100
    aif = np.exp(-(time-20)**2/20)
    a = acquisition.prepare_cpi_acquisition(signal, aif, time, .03,
        leakage_method="none", inference_method="svb", mask=np.zeros((1, 1, 1), bool))
    assert field.fit_selected_field(a).voxels == ()
    bad = connection.GuidedVoxel(0, "fallback_requested", ("no_valid_state",))
    monkeypatch.setattr(field, "fit_selected_voxels", lambda _: iter([bad]))
    result = field.fit_selected_field(a)
    assert result.voxels == (bad,) and result.local_calls == 0
    with pytest.raises(ValueError, match="svb"):
        field.fit_selected_field(replace(a, branches=(replace(a.branches[0], inference_method="vi"),)))
