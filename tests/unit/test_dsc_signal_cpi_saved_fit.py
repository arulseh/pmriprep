"""Portable regression of complete signal-CPI restart endpoints."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from pmriprep.models.dsc.signal_cpi.geometry import Model
from pmriprep.models.dsc.signal_cpi.leakage import (
    BSWPrior,
    LeakageReference,
    fit_with_prior,
)


@pytest.mark.slow
@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_complete_restart_endpoints(family: Model) -> None:
    """Fixed observations preserve every start's posterior and signal prediction."""
    time = np.arange(30.0) * 3.0
    u = np.maximum(time - 9.0, 0.0)
    aif = 10 * u**2 * np.exp(-u / 3)
    curve = 0.2 * u**2 * np.exp(-u / 5)
    integral = cumulative_trapezoid(curve, time, initial=0)
    reference = LeakageReference(
        time, integral, 0.001 * integral[:, None], "empirical_iid_region_mean", 32,
    )
    observed = 100 * np.exp(-0.03 * (curve - 0.003 * integral))
    result = fit_with_prior(
        observed, aif, time, 0.03, 5, reference, BSWPrior(0.003, 0.004), family=family,
    )
    fixture = Path(__file__).parent / "data" / "signal_cpi_saved_fits.json"
    expected = json.loads(fixture.read_text())[family]["expected"]
    actual = result.arrays | {
        "mtt": np.array([fit.mtt for fit in result.fits]),
        "iterations": np.array(result.iterations),
    }
    if family == "cpi_vtf":
        assert_gamma_state_accuracy(actual, expected, .03*reference.scale)
        assert all(assessment.valid for assessment in result.assessments)
        return
    for key, values in expected.items():
        if key in ("iterations", "converged"):
            np.testing.assert_array_equal(actual[key], values, err_msg=key)
        else:
            np.testing.assert_allclose(actual[key], values, rtol=1e-7, atol=1e-9,
                                       err_msg=key)
    assert all(assessment.valid for assessment in result.assessments)


def assert_gamma_state_accuracy(actual: dict, expected: dict, leakage_scale: float) -> None:
    """Preserve all restart measurements and uncertainty within numerical budgets.

    Iteration and winning-restart identities are not arithmetic invariants for
    near-tied nonlinear fits. Every restart must satisfy these limits, including
    exact convergence flags and unchanged initial states.
    """
    before = {name: np.asarray(value) for name, value in expected.items()}
    np.testing.assert_allclose(actual["starts"], before["starts"], rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(actual["converged"], before["converged"])
    assert 0 < actual["iterations"] <= 100
    covariance = before["covariance"]
    sd = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    new_sd = np.sqrt(np.diagonal(actual["covariance"], axis1=-2, axis2=-1))
    np.testing.assert_allclose(new_sd, sd, rtol=.001, atol=0)
    assert np.max(np.linalg.norm(actual["covariance"]-covariance, axis=(-2, -1)) /
                  np.linalg.norm(covariance, axis=(-2, -1))) <= .001
    assert np.max(abs(actual["mean"]-before["mean"]) / sd) <= .01
    for key in ("noise_shape", "noise_scale", "mtt"):
        np.testing.assert_allclose(actual[key], before[key], rtol=.001, atol=0, err_msg=key)
    # Flow, residue-derived volume and delay retain distinct physical units.
    np.testing.assert_allclose(actual["mean"][:, 0], before["mean"][:, 0], rtol=.001, atol=0)
    np.testing.assert_allclose(actual["mean"][:, 0]*actual["mtt"],
                               before["mean"][:, 0]*before["mtt"], rtol=.001, atol=0)
    np.testing.assert_allclose(actual["mean"][:, 6], before["mean"][:, 6], rtol=0, atol=.001)
    np.testing.assert_allclose(actual["mean"][:, -2]/leakage_scale,
                               before["mean"][:, -2]/leakage_scale, rtol=0, atol=1e-6)
    baseline = before["mean"][:, 1, None]
    np.testing.assert_allclose(actual["fitted_curves"]/baseline,
                               before["fitted_curves"]/baseline, rtol=0, atol=.001)
