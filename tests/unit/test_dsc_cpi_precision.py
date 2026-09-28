"""Signal verification is scale-aware and independent of the fitted evaluator."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pmriprep.models.dsc.local_vi import local_vi_state
from pmriprep.models.dsc.signal_cpi.leakage import readout_state
from pmriprep.models.dsc.signal_cpi.policy import assess_fit, model_for
from pmriprep.models.dsc.signal_cpi.residue import signal_evaluator


@pytest.fixture(scope="module", params=[
    (family, scale) for family in ("cpi", "cpi_vtf") for scale in (.001, 1., 1000.)
])
def state(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Build a consistent local posterior without nonlinear fitting."""
    family, scale = request.param
    time = np.arange(30.) * 3
    elapsed = np.maximum(time-9., 0.)
    aif = 10*elapsed**2*np.exp(-elapsed/3)
    observed = scale*100*np.exp(-.03*.2*elapsed**2*np.exp(-elapsed/5))
    model = model_for(observed, aif, time, .03, 5, family=family)
    mean = model.starts()[0]
    evaluator = signal_evaluator(model)
    prediction = evaluator.predictions(mean[None])[0]
    jacobian = evaluator.jacobians(mean[None])[0]
    local = local_vi_state(
        mean[None], jacobian[None], (observed-prediction)[None], model.prior.mean,
        np.linalg.inv(model.prior.covariance), np.ones(len(time)),
    )
    fit, assessment, _ = readout_state(
        model, observed, mean, local.covariance[0], local.shape, local.scale[0],
        prediction, jacobian, 5, 0, True,
    )
    assert assessment.valid
    return dict(model=model, observed=observed, prediction=prediction, jacobian=jacobian,
                mean=mean, local=local, fit=fit)


@pytest.mark.parametrize("fraction,valid", [
    (0., True), (5e-9, True), (-5e-9, True), (2e-8, False), (-2e-8, False),
])
def test_prediction_budget_is_baseline_normalized(
    state: dict[str, Any], fraction: float, valid: bool,
) -> None:
    """Small numerical differences pass; larger discrepancies fail at every scale."""
    model, mean = state["model"], state["mean"]
    prediction = state["prediction"].copy()
    prediction[np.argmin(prediction)] += fraction*mean[1]
    local = local_vi_state(
        mean[None], state["jacobian"][None], (state["observed"]-prediction)[None],
        model.prior.mean, np.linalg.inv(model.prior.covariance), np.ones(prediction.size),
    )
    _, assessment, _ = readout_state(
        model, state["observed"], mean, local.covariance[0], local.shape, local.scale[0],
        prediction, state["jacobian"], 5, 0, True,
    )
    assert assessment.valid == valid
    assert assessment.reasons == (() if valid else ("prediction_mean_mismatch",))


def test_prediction_is_recomputed_at_returned_mean(state: dict[str, Any]) -> None:
    """A stale curve cannot be certified by its own optimized evaluator cache."""
    mean = state["mean"].copy()
    mean[0] *= 1.01
    local = state["local"]
    _, assessment, _ = readout_state(
        state["model"], state["observed"], mean, local.covariance[0], local.shape,
        local.scale[0], state["prediction"], state["jacobian"], 5, 0, True,
    )
    assert not assessment.valid
    assert "prediction_mean_mismatch" in assessment.reasons


@pytest.mark.parametrize("corruption,reason", [
    ("nonfinite", "nonfinite_state"),
    ("readout", "mean_readout_mismatch"),
    ("covariance", "nonpositive_definite_covariance"),
    ("noise", "inconsistent_local_factors"),
])
def test_other_state_checks_are_not_relaxed(
    state: dict[str, Any], corruption: str, reason: str,
) -> None:
    """Signal accuracy does not excuse invalid posterior state or measurements."""
    fit, local, model = state["fit"], state["local"], state["model"]
    prediction, covariance = state["prediction"].copy(), local.covariance[0].copy()
    noise_scale = local.scale[0]
    if corruption == "nonfinite":
        prediction[0] = np.nan
    elif corruption == "readout":
        fit = replace(fit, cbf=fit.cbf*1.01)
    elif corruption == "covariance":
        covariance[0, 0] = -1.
    else:
        noise_scale *= 2
    assessment = assess_fit(
        fit, model, state["observed"], prediction, state["jacobian"], covariance,
        local.shape, noise_scale, model.prior.covariance,
    )
    assert not assessment.valid and reason in assessment.reasons
