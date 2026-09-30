"""Signal verification is scale-aware and independent of the fitted evaluator."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from pmriprep.models.dsc.local_vi import local_vi_state
from pmriprep.models.dsc.signal_cpi.connection import fit_voxels, prepare_acquisition
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
    ("covariance_scale", "inconsistent_local_factors"),
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
    elif corruption == "covariance_scale":
        covariance *= 1.01
    else:
        noise_scale *= 2
    assessment = assess_fit(
        fit, model, state["observed"], prediction, state["jacobian"], covariance,
        local.shape, noise_scale, model.prior.covariance,
    )
    assert not assessment.valid and reason in assessment.reasons


@pytest.mark.parametrize("state", [("cpi", 1000.0)], indirect=True)
def test_directional_cbf_covariance_corruption_is_rejected(
    state: dict[str, Any],
) -> None:
    """A CBF-only variance error cannot hide behind larger covariance entries."""
    fit, local, model = state["fit"], state["local"], state["model"]
    covariance = local.covariance[0].copy()
    factor = np.sqrt(1.01)
    covariance[0, :] *= factor
    covariance[:, 0] *= factor
    residual = state["observed"] - state["prediction"]
    gram = state["jacobian"].T @ state["jacobian"]
    required_rate = 1e-6 + 0.5 * (
        residual @ residual + np.sum(covariance * gram.T)
    )
    noise_scale = 1 / required_rate
    fit = replace(fit, noise_sd=1 / np.sqrt(local.shape * noise_scale))
    assessment = assess_fit(
        fit, model, state["observed"], state["prediction"], state["jacobian"], covariance,
        local.shape, noise_scale, model.prior.covariance,
    )
    assert np.isclose(assessment.gamma_rate_ratio, 1.0, rtol=1e-6)
    assert assessment.covariance_identity_error is not None
    assert assessment.covariance_identity_error > 1e-4
    assert not assessment.valid
    assert assessment.reasons == ("inconsistent_local_factors",)


def test_clean_short_mtt_fit_is_not_rejected_by_roundoff() -> None:
    """An accurate ill-conditioned clean fit survives ppm-scale closure error."""
    time = np.arange(120, dtype=float) * 1.5
    onset, alpha, beta = 66.0, 3.0, 1.5
    elapsed = np.maximum(time - onset, 0.0)
    peak_time = alpha * beta
    aif = np.where(
        elapsed > 0,
        45 * (elapsed / peak_time) ** alpha * np.exp(alpha - elapsed / beta),
        0.0,
    )
    arterial = PchipInterpolator(time, aif, extrapolate=False)
    flow, mtt, te = 40 / 6000, 4.0, 0.035
    concentration = np.asarray([
        flow * quad(
            lambda tau: float(arterial(tau)) * np.exp(-(sample - tau) / mtt),
            0.0,
            sample,
            points=time[(time > 0) & (time < sample)],
            epsabs=1e-10,
            epsrel=1e-10,
            limit=200,
        )[0]
        for sample in time
    ])
    signal = 1000 * np.exp(-te * concentration)
    prepared = prepare_acquisition(
        signal[None, None, None], aif, time, te,
        leakage_method="none", baseline_frames=44, control_points=12,
    )
    output = list(fit_voxels(prepared))[0]
    assert output.status == "provisional"
    assert output.diagnostics is not None
    assert all(assessment.valid for assessment in output.diagnostics.assessments)
    assert output.cbf * 6000 == pytest.approx(40, rel=0.01)
    assert output.mtt == pytest.approx(4, rel=0.02)
    assert output.delay == pytest.approx(0, abs=0.01)
