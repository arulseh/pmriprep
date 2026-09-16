"""Vascular VI support, posterior/readout consistency and endpoint recovery."""

from typing import Any

import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import ComputeDSCMaps
from pmriprep.models.bayesian_utils import (
    TransformType,
    BayesianPrior,
    build_priors_from_params,
    from_transformed,
)
from pmriprep.models.dsc import VascularModel, fit_dsc_model
from pmriprep.models.dsc.batched_fitting import (
    _compute_jacobian_columns,
    batched_dsc_vi_fit,
    batched_vascular_forward,
)
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit


def _setup() -> tuple[dict[str, Any], list[BayesianPrior]]:
    """Use production fitting definitions and a portable synthetic AIF."""
    definitions = VascularModel().get_fitting_parameter_definitions()
    priors = build_priors_from_params(definitions)
    times = np.arange(120, dtype=float)
    u = np.maximum(times - 8.0, 0.0)
    aif = 3.0 * u * np.exp(-u / 3.0)
    kwargs = dict(
        model_type="vascular",
        aif=aif,
        time_points=times,
        prior_means_nat=np.array([p.mean for p in priors]),
        prior_stds_nat=np.array([p.std for p in priors]),
        transforms=[p.transform for p in priors],
        upper_bounds=[p.upper_bound for p in priors],
        fitting_param_names=[p.name for p in definitions],
        fitting_bounds=[p.bounds for p in definitions],
        prior_means_transformed=np.array([p.mean_transformed for p in priors]),
        prior_stds_transformed=np.array([p.std_transformed for p in priors]),
        save_fitted_curves=True,
    )
    return kwargs, priors


def _forward(values: np.ndarray, aif: np.ndarray, times: np.ndarray) -> np.ndarray:
    return batched_vascular_forward(
        values[:, 0],
        values[:, 2],
        values[:, 1] / values[:, 2],
        values[:, 3],
        aif,
        times,
    )


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_vascular_vi_posterior_and_readout_share_bounded_point(
    spatial: bool, weighted: bool
) -> None:
    """Upper/lower challenges cannot hide unbounded posterior means by clipping."""
    kwargs, priors = _setup()
    truth = np.array(
        [
            [0.06, 6.0, 3.0, 2.0],
            [0.06, 29.0, 0.6, 2.0],
            [0.06, 0.2, 3.0, 2.0],
            [0.06, 40.0, 3.0, 2.0],
            [0.06, 6.0, 0.2, 2.0],
            [0.06, 6.0, 15.0, 2.0],
            [0.06, 6.0, 3.0, -5.0],
            [0.06, 6.0, 3.0, 12.0],
            [3.0, 6.0, 3.0, 2.0],
        ]
    )
    tissue = _forward(truth, kwargs["aif"], kwargs["time_points"])
    coords = (
        np.arange(len(truth)),
        np.zeros(len(truth), dtype=int),
        np.zeros(len(truth), dtype=int),
    )
    if weighted:
        weights = np.ones(tissue.shape[1])
        weights[12:18] = 0.2
        kwargs["weights"] = weights
    if spatial:
        result = batched_dsc_spatial_vi_fit(
            **kwargs,
            tissue_curves=tissue,
            mask_coords=coords,
            spatial_shape=(len(truth), 1, 1),
        )
    else:
        result = batched_dsc_vi_fit(
            **kwargs, tissue_curves=tissue, return_posteriors=True
        )
    means = result.posterior_means_transformed
    assert means is not None
    natural = np.column_stack(
        [
            from_transformed(means[:, j], p.transform, p.upper_bound)
            for j, p in enumerate(priors)
        ]
    )
    for j, (name, (lo, hi)) in enumerate(
        zip(kwargs["fitting_param_names"], kwargs["fitting_bounds"])
    ):
        assert np.all(natural[:, j] >= lo - 1e-12)
        assert np.all(natural[:, j] <= hi + 1e-12)
        np.testing.assert_allclose(
            result.params[name], natural[:, j], atol=1e-12, rtol=1e-12
        )
        assert np.all(np.isfinite(result.uncertainties[name]))
        assert np.all(result.uncertainties[name] >= 0)
        assert result.posterior_cov_diag is not None
        scale = natural[:, j] if priors[j].transform == TransformType.LOG else 1.0
        np.testing.assert_allclose(
            result.uncertainties[name],
            np.sqrt(result.posterior_cov_diag[:, j]) * scale,
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.all(result.credible_interval_lower[name] >= lo)
        assert np.all(result.credible_interval_upper[name] <= hi)
        assert np.all(
            result.credible_interval_lower[name] <= result.credible_interval_upper[name]
        )
    prediction = _forward(natural, kwargs["aif"], kwargs["time_points"])
    np.testing.assert_allclose(result.fitted_curves, prediction, atol=1e-12, rtol=1e-12)
    r2 = 1 - np.sum((tissue - prediction) ** 2, axis=1) / np.sum(
        (tissue - tissue.mean(axis=1, keepdims=True)) ** 2, axis=1
    )
    np.testing.assert_allclose(result.r_squared, np.clip(r2, -1, 1), atol=1e-12)
    readout = ComputeDSCMaps()._batched_result_to_voxel_dicts(
        result,
        coords,
        "vascular",
        len(truth),
        1.0,
        None,
        kwargs["time_points"],
        inference="svb" if spatial else "vi",
    )
    for i, voxel in enumerate(readout):
        assert voxel["cbv"] == pytest.approx(natural[i, 0] * natural[i, 1])
        assert voxel["mtt"] == pytest.approx(natural[i, 1])
        assert voxel["needs_fallback"] == (result.r_squared[i] < 0)


@pytest.mark.parametrize("mtt,alpha,min_r2", [(40.0, 3.0, 0.979), (6.0, 15.0, 0.9999)])
def test_vascular_vi_reoptimizes_other_parameters_at_ceiling(
    mtt: float, alpha: float, min_r2: float
) -> None:
    """A feasible fit improves on merely clipping the out-of-box solution."""
    kwargs, _ = _setup()
    truth = np.array([[0.06, mtt, alpha, 2.0]])
    tissue = _forward(truth, kwargs["aif"], kwargs["time_points"])
    result = batched_dsc_vi_fit(**kwargs, tissue_curves=tissue)
    assert result.r_squared[0] > min_r2
    assert result.params["cbf"][0] > 0.063
    assert result.params["mtt" if mtt == 40 else "alpha"][0] == pytest.approx(
        30.0 if mtt == 40 else 10.0
    )


def test_vascular_vi_ordinary_recovery() -> None:
    kwargs, _ = _setup()
    truth = np.array([[0.06, 6.0, 3.0, 2.0]])
    tissue = _forward(truth, kwargs["aif"], kwargs["time_points"])
    result = batched_dsc_vi_fit(**kwargs, tissue_curves=tissue)
    fitted = np.array(
        [result.params[name][0] for name in kwargs["fitting_param_names"]]
    )
    np.testing.assert_allclose(fitted, truth[0], rtol=1e-4)
    assert result.r_squared[0] > 0.999999


@pytest.mark.parametrize("truth", [[0.06, 40.0, 3.0, 2.0], [0.06, 6.0, 15.0, 2.0]])
def test_scalar_vascular_vi_respects_same_fitting_support(truth: list[float]) -> None:
    kwargs, _ = _setup()
    tissue = _forward(np.array([truth]), kwargs["aif"], kwargs["time_points"])[0]
    result = fit_dsc_model(
        VascularModel(), tissue, kwargs["aif"], kwargs["time_points"], method="vi"
    )
    model_params = [result.parameters[p] for p in VascularModel().get_parameter_names()]
    fitted = VascularModel().model_to_fitting_params(np.array(model_params))
    for value, (lo, hi) in zip(fitted, kwargs["fitting_bounds"]):
        assert lo - 1e-12 <= value <= hi + 1e-12
    np.testing.assert_allclose(
        result.fitted_tissue_curve,
        _forward(fitted[None, :], kwargs["aif"], kwargs["time_points"])[0],
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "endpoint,transform",
    [
        (0.5, TransformType.LOG),
        (30.0, TransformType.LOG),
        (-3.0, TransformType.IDENTITY),
        (10.0, TransformType.IDENTITY),
    ],
)
@pytest.mark.parametrize("nudge", [False, True])
def test_bounded_jacobian_uses_nonzero_feasible_difference(
    endpoint: float,
    transform: TransformType,
    nudge: bool,
) -> None:
    """At either endpoint, differentiate within support rather than flattening."""
    natural = np.array([[0.06, endpoint]])
    log_parameter = transform == TransformType.LOG
    lo, hi = (0.5, 30.0) if log_parameter else (-3.0, 10.0)
    mean = np.array([[np.log(0.06), np.log(endpoint) if log_parameter else endpoint]])
    lower = np.array([-np.inf, np.log(lo) if log_parameter else lo])
    upper = np.array([np.log(2.0), np.log(hi) if log_parameter else hi])
    if nudge:
        interior = (lower[1] + upper[1]) / 2
        mean[0, 1] = np.nextafter(mean[0, 1], interior)
        natural[0, 1] = np.exp(mean[0, 1]) if log_parameter else mean[0, 1]

    def evaluate(params: np.ndarray) -> np.ndarray:
        assert np.all(params[:, 1] >= lo - 1e-12)
        assert np.all(params[:, 1] <= hi + 1e-12)
        return (params[:, 0] * params[:, 1])[:, None]

    jacobian = _compute_jacobian_columns(
        mean,
        natural,
        evaluate(natural),
        evaluate,
        [TransformType.LOG, transform],
        [None, None],
        np.array([0.0, lo]),
        np.array([2.0, hi]),
        0,
        transformed_bounds=(lower, upper),
    )
    expected = evaluate(natural) if log_parameter else np.array([[0.06]])
    np.testing.assert_allclose(jacobian[1], expected, rtol=3e-5)
