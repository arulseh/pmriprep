"""Regression coverage for EES fractions and coupled tissue occupancy."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pmriprep.interfaces.dce_compute import _is_degenerate_dce_fit
from pmriprep.models.dce.batched_fitting import _apply_volume_constraint_batched, batched_lm_fit
from pmriprep.models.dce.batched_models import evaluate_model
from pmriprep.models.dce._convolution import analytic_exp_convolution
from pmriprep.models.dce.bayesian import BayesianPKFitter
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel


@pytest.fixture
def sampling() -> tuple[np.ndarray, np.ndarray]:
    """Return a uniform minute grid and a smooth concentration AIF."""
    time = np.linspace(0, 6, 181)
    aif = 5 * np.exp(-0.5 * ((time - 0.5) / 0.15) ** 2)
    return time, aif + np.where(time >= 0.5, np.exp(-(time - 0.5) / 2), 0)


@pytest.mark.parametrize("name", ["tofts", "extended_tofts", "2cxm"])
@pytest.mark.parametrize("ve", [0.0005, 1e-6])
@pytest.mark.parametrize("vp", [0.02, 0.0005])
def test_low_ve_matches_independent_ode(
    sampling: tuple[np.ndarray, np.ndarray], name: str, ve: float, vp: float,
) -> None:
    """Positive low EES volumes agree with independently driven compartments."""
    time, aif = sampling
    models = {"tofts": ToftsModel(), "extended_tofts": ExtendedToftsModel(),
              "2cxm": TwoCompartmentExchangeModel()}
    model = models[name]
    params = [40, 10, ve, vp] if name == "2cxm" else [0.1, ve]
    if name == "extended_tofts":
        params.append(vp)

    def system(t: float, c: np.ndarray) -> list[float]:
        arterial = float(np.interp(t, time, aif))
        if name == "2cxm":
            return [0.4 / vp * (arterial - c[0]) - 0.1 / vp * (c[0] - c[1]),
                    0.1 / ve * (c[0] - c[1])]
        return [0.1 * arterial - 0.1 / ve * c[0]]

    ode = solve_ivp(system, (0, 6), np.zeros(2 if name == "2cxm" else 1),
                    t_eval=time, method="Radau", rtol=1e-10, atol=1e-15,
                    max_step=float(time[1] - time[0]) / 10)
    assert ode.success
    reference = vp * ode.y[0] + ve * ode.y[1] if name == "2cxm" else ode.y[0]
    if name == "extended_tofts":
        reference = reference + vp * aif
    scalar = model.evaluate(params, aif, time)
    batch_params = {k.lower(): np.array([v]) for k, v in zip(model.get_parameter_names(), params)}
    batch = evaluate_model(name, batch_params, aif, time).tissue_curves[0]
    np.testing.assert_allclose(scalar, reference, atol=2e-10, rtol=2e-4)
    np.testing.assert_allclose(batch, scalar, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("name,truth", [
    ("tofts", [0.1, 0.95]), ("tofts", [0.1, 1.0]),
    ("extended_tofts", [0.1, 0.95, 0.05]),
    ("2cxm", [40, 10, 0.85, 0.02]),
])
def test_high_ve_recovery(
    sampling: tuple[np.ndarray, np.ndarray], name: str, truth: list[float],
) -> None:
    """Admissible high-volume curves are recoverable and retained."""
    time, aif = sampling
    model = {"tofts": ToftsModel(), "extended_tofts": ExtendedToftsModel(),
             "2cxm": TwoCompartmentExchangeModel()}[name]
    curve = model.evaluate(truth, aif, time)
    scalar = model.fit(curve, aif, time)
    batch = batched_lm_fit(name, curve[None], aif, time, max_iter=200)
    assert scalar.converged
    for key, value in zip(model.get_parameter_names(), truth):
        assert scalar.parameters[key] == pytest.approx(value, rel=0.02, abs=1e-5)
        assert batch.params[key.lower()][0] == pytest.approx(value, rel=0.02, abs=1e-5)
    assert not _is_degenerate_dce_fit(name, scalar.parameters)


def test_projection_preserves_fraction_domain() -> None:
    """Joint projection preserves the shared nonnegative fraction domain."""
    params = _apply_volume_constraint_batched(
        np.array([[40, 10, 0.01, 1.0], [40, 10, 1.0, 1.0]]),
        ["fp", "ps", "ve", "vp"],
    )
    assert np.all(params[:, 2:] >= 0)
    assert np.all(params[:, 2:].sum(axis=1) <= 1.0)


@pytest.mark.parametrize("ve,degenerate", [(0, True), (1, False), (1.000001, True), (np.nan, True)])
def test_ve_retention(ve: float, degenerate: bool) -> None:
    """Only invalid EES fractions or the unidentifiable zero limit are rejected."""
    for name in ("tofts", "extended_tofts", "2cxm"):
        assert _is_degenerate_dce_fit(
            name, {"Ktrans": 0.1, "ve": ve, "vp": 0, "Fp": 40, "PS": 10},
        ) is degenerate


def test_ve_priors_share_unit_support() -> None:
    """All fraction priors use the common unit-domain upper bound."""
    for model in (ToftsModel(), ExtendedToftsModel(), TwoCompartmentExchangeModel()):
        index = model.get_parameter_names().index("ve")
        assert model.get_bounds()[index] == (0, 1)
        assert BayesianPKFitter(model).priors[index].upper_bound == 1


@pytest.mark.parametrize("name", ["tofts", "extended_tofts", "2cxm"])
def test_zero_ve_reduced_model(sampling: tuple[np.ndarray, np.ndarray], name: str) -> None:
    """An absent EES has no exchange contribution to tissue concentration."""
    time, aif = sampling
    model = {"tofts": ToftsModel(), "extended_tofts": ExtendedToftsModel(),
             "2cxm": TwoCompartmentExchangeModel()}[name]
    params = [40, 10, 0, 0.02] if name == "2cxm" else [0.1, 0]
    expected = np.zeros_like(aif)
    if name == "extended_tofts":
        params.append(0.02)
        expected = 0.02 * aif
    if name == "2cxm":
        expected = 0.4 * analytic_exp_convolution(aif, 0.4 / 0.02, float(time[1]))
    scalar = model.evaluate(params, aif, time)
    batch_params = {k.lower(): np.array([v]) for k, v in zip(model.get_parameter_names(), params)}
    np.testing.assert_allclose(scalar, expected, atol=1e-12)
    np.testing.assert_allclose(evaluate_model(name, batch_params, aif, time).tissue_curves[0], expected, atol=1e-12)


@pytest.mark.parametrize("method", ["vi", "map"])
@pytest.mark.parametrize("ve", [0.005, 0.95])
@pytest.mark.parametrize("name", ["tofts", "2cxm"])
def test_bayesian_ve_intervals(
    sampling: tuple[np.ndarray, np.ndarray], method: str, ve: float, name: str,
) -> None:
    """Posterior intervals remain ordered inside the widened fraction support."""
    time, aif = sampling
    model = ToftsModel() if name == "tofts" else TwoCompartmentExchangeModel()
    truth = [0.1, ve] if name == "tofts" else [40, 10, ve, 0.02]
    tissue = model.evaluate(truth, aif, time)
    fitter = BayesianPKFitter(model)
    result = (fitter.fit_vi if method == "vi" else fitter.fit_map)(tissue, aif, time)
    assert result.converged
    lo, hi = result.credible_intervals["ve"]
    assert 0 <= lo <= result.parameters["ve"] <= hi <= 1
    if name == "tofts":
        assert result.parameters["ve"] == pytest.approx(ve, rel=0.05)


def test_2cxm_zero_ve_is_rejected() -> None:
    """A feasible zero-EES result is not a retained exchange-tissue estimate."""
    assert _is_degenerate_dce_fit(
        "2cxm", {"Fp": 40, "PS": 10, "ve": 0, "vp": 0.995},
    )
