"""Vascular VI support, posterior/readout consistency and endpoint recovery."""

from dataclasses import replace
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
from pmriprep.models.dsc.vascular import _fit_vascular_nlls


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
            [0.06, 59.0, 0.6, 2.0],
            [0.06, 0.2, 3.0, 2.0],
            [0.06, 80.0, 3.0, 2.0],
            [0.06, 6.0, 0.2, 2.0],
            [0.06, 6.0, 200.0, 2.0],
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
    # Exercise reoptimization at a supplied box, independently of defaults.
    kwargs["fitting_bounds"][1] = (0.5, 30.0)
    kwargs["fitting_bounds"][2] = (0.5, 10.0)
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


@pytest.mark.parametrize("method", ["nlls", "vi", "svb"])
def test_vascular_long_mtt_headroom_recovers_joint_fit(method: str) -> None:
    """A plausible-volume 45-s curve must not rail and bias fitted flow."""
    kwargs, _ = _setup()
    truth = np.array([[0.004, 45.0, 3.0, 2.0]])
    tissue = _forward(truth, kwargs["aif"], kwargs["time_points"])
    if method == "nlls":
        model = VascularModel()
        result = _fit_vascular_nlls(
            model,
            tissue[0],
            kwargs["aif"],
            kwargs["time_points"],
        )
        assert result is not None
        fitted = model.model_to_fitting_params(
            np.array([result.parameters[p] for p in model.get_parameter_names()])
        )
    elif method == "vi":
        result = batched_dsc_vi_fit(**kwargs, tissue_curves=tissue)
        fitted = np.array(
            [result.params[name][0] for name in kwargs["fitting_param_names"]]
        )
    else:
        result = batched_dsc_spatial_vi_fit(
            **kwargs,
            tissue_curves=tissue,
            mask_coords=(np.array([0]), np.array([0]), np.array([0])),
            spatial_shape=(1, 1, 1),
        )
        fitted = np.array(
            [result.params[name][0] for name in kwargs["fitting_param_names"]]
        )

    np.testing.assert_allclose(fitted[:2], truth[0, :2], rtol=5e-3)
    assert fitted[1] < 59.0


def test_vascular_shape_support_does_not_change_gaussian_priors() -> None:
    """A wider LOG support must not implicitly change population priors."""
    definitions = VascularModel().get_fitting_parameter_definitions()
    narrower = [replace(p, bounds=(0.5, 10.0)) if p.name == "alpha" else p
                for p in definitions]
    for actual, reference in zip(
        build_priors_from_params(definitions), build_priors_from_params(narrower)
    ):
        assert actual.transform == reference.transform
        assert actual.upper_bound == reference.upper_bound
        np.testing.assert_array_equal(
            [actual.mean, actual.std, actual.mean_transformed, actual.std_transformed],
            [reference.mean, reference.std, reference.mean_transformed, reference.std_transformed],
        )


@pytest.mark.parametrize("method", ["nlls", "vi", "svb"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("alpha", [3.0, 25.0, 100.0, 200.0])
def test_vascular_narrow_distribution_recovery_and_support(
    method: str, weighted: bool, alpha: float
) -> None:
    """Resolve sampled narrow distributions; bound above-support challenges."""
    kwargs, _ = _setup()
    times = np.arange(0.0, 120.0, 0.5)
    u = np.maximum(times - 15.0, 0.0)
    aif = 5.0 * (u / 4.5) ** 3 * np.exp(3.0 - u / 1.5)
    kwargs.update(time_points=times, aif=aif)
    truth = np.tile([0.06, 8.0, alpha, 2.0], (3, 1))
    tissue = _forward(truth, aif, times)
    weights = np.ones(len(times))
    if weighted:
        weights[(times >= 19.0) & (times <= 23.0)] = 0.2
    kwargs["weights"] = weights
    model = VascularModel()
    if method == "nlls":
        result = _fit_vascular_nlls(model, tissue[0], aif, times, weights=weights)
        assert result is not None
        fitted = model.model_to_fitting_params(
            np.array([result.parameters[p] for p in model.get_parameter_names()])
        )[None, :]
        curves = np.asarray(result.fitted_tissue_curve)[None, :]
        target = tissue[:1]
    else:
        if method == "vi":
            batched = batched_dsc_vi_fit(**kwargs, tissue_curves=tissue)
        else:
            batched = batched_dsc_spatial_vi_fit(
                **kwargs, tissue_curves=tissue,
                mask_coords=(np.arange(3), np.zeros(3, dtype=int), np.zeros(3, dtype=int)),
                spatial_shape=(3, 1, 1),
            )
        fitted = np.column_stack([batched.params[p] for p in kwargs["fitting_param_names"]])
        curves = batched.fitted_curves
        target = tissue
    for j, (lo, hi) in enumerate(kwargs["fitting_bounds"]):
        assert np.all(fitted[:, j] >= lo - 1e-12)
        assert np.all(fitted[:, j] <= hi + 1e-12)
    np.testing.assert_allclose(curves, _forward(fitted, aif, times), rtol=1e-10, atol=1e-12)
    relative_rss = np.sum(weights * (curves - target) ** 2) / np.sum(weights * target ** 2)
    if alpha <= 100.0:
        np.testing.assert_allclose(fitted, truth[:len(fitted)], rtol=1e-3)
        assert relative_rss < 1e-8
    else:
        np.testing.assert_allclose(fitted[:, 2], 100.0, rtol=1e-6)
        assert relative_rss < 1e-4


@pytest.mark.parametrize("truth", [
    [0.06, 80.0, 3.0, 2.0], [0.06, 6.0, 200.0, 2.0],
    [0.06, 8.0, 25.0, 2.0], [0.06, 8.0, 100.0, 2.0],
])
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
        (60.0, TransformType.LOG),
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
    lo, hi = (0.5, 60.0) if log_parameter else (-3.0, 10.0)
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
