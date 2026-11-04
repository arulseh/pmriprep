"""Fixed-input VI reuse, model identity and same-delay fallback."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pmriprep.models.dce import fixed_inference as inference
from pmriprep.models.dce.base import FitResult
from pmriprep.models.dce.bayesian import create_vp_prior_from_dsc
from pmriprep.models.dce.delay_profiles import ProfileFamily, build_delay_profile
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.fit_errors import NumericalFitError
from pmriprep.models.dce.local_selection import family_timing
from pmriprep.models.dce.local_refinement import ConditionalInput, finish_conditional
from pmriprep.models.dce.identification import identify_fixed_point
from pmriprep.models.dce.native_exchange import predict
from pmriprep.models.dce.profile_fitting import tofts_predict


def _model(family: ProfileFamily) -> inference.FixedDelayModel:
    times = np.arange(65) / 30
    aif = 4 * np.exp(-((times - 0.35) / 0.1) ** 2) + 0.1 * (times > 0.35)
    observed = np.r_[0:9, 12:65]
    context = DelayedPchip(aif, times, 2.0, observed)
    if family == "2cxm":
        truth = predict(context, np.array([70.0, 15.0, 0.3, 0.04]))
    elif family == "patlak":
        truth = 0.08 * context.convolution(0.0) + 0.04 * context.cp
    else:
        truth = tofts_predict(context, np.array([0.2, 0.3]))
        if family == "extended_tofts":
            truth += 0.04 * context.cp
    tissue = truth + np.random.default_rng(78).normal(0, 0.0005, len(truth))
    profile = build_delay_profile(
        aif, times, np.stack([tissue, tissue]), family,
        observed=observed, delays_seconds=np.array([0.0, 2.0]),
    )
    timing = family_timing(profile, np.array([0, 1]), 0, frame_interval_seconds=2)
    assert timing.index == 1 and timing.reason is None
    seed = inference.selected_fit_result(profile, timing, 0, aif, times, tissue)
    if family == "patlak":
        assert "ve" not in seed.parameters
    with pytest.raises(ValueError, match="operator/data"):
        inference.selected_fit_result(profile, timing, 0, aif, times, tissue + 0.1)
    return inference.FixedDelayModel(family, aif, times, tissue, 2, seed, observed)


@pytest.mark.parametrize("family", ["patlak", "tofts", "extended_tofts", "2cxm"])
def test_actual_vi_keeps_selected_family_delay_and_full_history(family: ProfileFamily) -> None:
    model = _model(family)
    seed = model.fit(model.tissue, model.input_aif, model.input_times)
    item = ConditionalInput(
        family, model.context.aif, np.arange(65) / 30, model.context.observed,
        model.tissue, 2, seed, "nlls",
        np.zeros(5 if family == "2cxm" else model.n_parameters), seed.residual_sum_squares,
    )
    nlls, vi = finish_conditional([item, replace(item, inference="vi")])
    result = vi.refinement
    params = np.array(model.dict_to_params(seed.parameters))
    if family == "patlak":
        params = np.array([params[0], np.nan, params[1]])
    expected = identify_fixed_point(family, model.context, params, seed.residual_sum_squares)
    for conditional in (nlls, vi):
        np.testing.assert_array_equal(conditional.identification.effective, expected.effective)
        assert conditional.identification.reasons == expected.reasons
    assert nlls.refinement.fit.parameters == seed.parameters
    assert result.family == family and result.delay == 2
    assert result.actual_engine == "vi" and not result.fallback
    assert result.fit.converged and result.fit.uncertainties
    model.check_result_curve(result.fit)
    np.testing.assert_array_equal(
        model.fit(model.tissue, model.input_aif, model.input_times).fitted_curve,
        seed.fitted_curve,
    )
    assert len(model.context.aif) > len(model.input_aif)


def test_cached_seed_is_isolated_and_changed_data_cannot_be_reused() -> None:
    model = _model("extended_tofts")
    seed = model.fit(model.tissue, model.input_aif, model.input_times)
    seed.parameters["Ktrans"] = 999
    seed.fitted_curve[:] = 999
    assert model.fit(model.tissue, model.input_aif, model.input_times).parameters["Ktrans"] < 1
    for inputs in (
        (model.tissue + 1, model.input_aif, model.input_times, None),
        (model.tissue, model.input_aif + 1, model.input_times, None),
        (model.tissue, model.input_aif, model.input_times + 1, None),
        (model.tissue, model.input_aif, model.input_times, np.full_like(model.tissue, 0.5)),
    ):
        with pytest.raises(ValueError):
            model.fit(*inputs[:3], weights=inputs[3])
    with pytest.raises(ValueError, match="operator/data"):
        inference.FixedDelayModel(
            "extended_tofts", model.context.aif, np.arange(65) / 30,
            model.tissue, 0, model.fit(model.tissue, model.input_aif, model.input_times),
            model.context.observed,
        )


def test_vi_fallback_preserves_seed_but_does_not_hide_contract_or_process_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model("2cxm")
    seed = model.fit(model.tissue, model.input_aif, model.input_times)
    response: FitResult | Exception = NumericalFitError("nonfinite solve")

    def fit(*args: object, **kwargs: object) -> FitResult:
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(inference.BayesianPKFitter, "fit_vi", fit)
    for condition in ("numerical", "nonconverged", "ceiling"):
        if condition != "numerical":
            response = deepcopy(seed)
            response.fitting_method = "vi"
            if condition == "nonconverged":
                response.converged = False
            else:
                response.parameters["Fp"] = 500
        result = inference.refine_vi(model)
        assert result.fallback and result.actual_engine == "nlls"
        assert result.delay == 2 and result.fit.parameters == seed.parameters
        np.testing.assert_array_equal(result.fit.fitted_curve, seed.fitted_curve)
    for error in (MemoryError("allocation"), RuntimeError("backend contract")):
        response = error
        with pytest.raises(type(error)):
            inference.refine_vi(model)
    response = deepcopy(seed)
    response.model_name = "patlak"
    response.fitting_method = "vi"
    with pytest.raises(ValueError, match="family"):
        inference.refine_vi(model)
    seed.parameters["Fp"] = 500
    with pytest.raises(ValueError, match="Eligible"):
        inference.FixedDelayModel(
            "2cxm", model.context.aif, np.arange(65) / 30,
            model.tissue, 2, seed, model.context.observed,
        )


def test_prior_and_sloppy_are_forwarded_without_changing_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model("extended_tofts")
    prior = create_vp_prior_from_dsc(0.03, relative_uncertainty=0.5)

    def fit(self: inference.BayesianPKFitter, *args: object, **kwargs: object) -> FitResult:
        assert self.model is model
        assert self.priors[model.get_parameter_names().index("vp")] is prior
        assert kwargs["sloppy"] is True
        seed = model.fit(model.tissue, model.input_aif, model.input_times)
        assert kwargs["initial_parameters"] == seed.parameters
        seed.fitting_method = "vi"
        return seed

    monkeypatch.setattr(inference.BayesianPKFitter, "fit_vi", fit)
    result = inference.refine_vi(model, vp_prior=prior, sloppy=True)
    assert result.actual_engine == "vi" and result.delay == 2
