"""Numerical and physical contracts for the independent leakage generator."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from .dsc_exchange import ExchangeParameters, apparent_delta_r2, exchange_curves
from .dsc_observations import signal_observation
from .dsc_reference import (
    analytic_residue,
    continuous_tissue,
    gamma_variate_aif,
    transport_kernel,
)


def _aif(t: Any) -> np.ndarray:
    """Smooth causal continuous blood input."""
    return gamma_variate_aif(t, onset_seconds=10.0, peak=5, alpha=2.5, beta_seconds=2)


def _step(t: Any) -> np.ndarray:
    """Unit blood input with a declared onset."""
    return np.asarray(np.asarray(t) >= 10, dtype=float)


def _parameters(**overrides: float) -> ExchangeParameters:
    """Exchange settings independent of any fitted model."""
    settings = dict(blood_flow=0.014, blood_volume=0.07, ees_volume=0.25,
                    permeability=0.0035, hematocrit=0.25)
    return ExchangeParameters(**{**settings, **overrides})


def _signal(civ: Any, cev: Any, **overrides: float) -> np.ndarray:
    """Declared effective GRE settings."""
    settings = dict(tr_seconds=1.5, te_seconds=0.045, flip_angle_degrees=60,
                    native_t1_seconds=1.8, r1_ees=4.2, r2_intravascular=44, r2_ees=5.3)
    return apparent_delta_r2(civ, cev, **{**settings, **overrides})


@pytest.mark.parametrize("hematocrit", [0.0, 0.25, 0.45])
@pytest.mark.parametrize("delay", [-2.37, 0.0, 2.37])
def test_zero_exchange_matches_independent_vascular_integral(hematocrit: float, delay: float) -> None:
    """Known AIF relaxation calibration recovers blood, not plasma, flow units."""
    params = _parameters(permeability=0, hematocrit=hematocrit)
    time = np.array([0, 5, 8, 10, 12.13, 16.7, 24.2, 45.1])
    result = exchange_curves(time, _aif, params, aif_onset_seconds=10, delay_seconds=delay)
    expected = continuous_tissue(
        time, _aif, analytic_residue("exponential", params.vascular_mtt),
        flow=params.blood_flow, aif_onset_seconds=10, delay_seconds=delay,
    ).values
    np.testing.assert_allclose(params.plasma_volume * result.plasma, expected, rtol=2e-5, atol=2e-8)
    np.testing.assert_array_equal(result.ees, 0)
    np.testing.assert_allclose(_signal(expected, np.zeros_like(time)), 44 * expected, atol=1e-12)
    assert result.refinement_error <= 1e-5
    assert params.ktrans == 0


def test_step_exchange_against_matrix_exponential_and_mass_balance() -> None:
    """An independent block exponential also integrates influx minus efflux."""
    params = _parameters()
    time = np.array([0, 10, 10.1, 11.3, 14.7, 22, 40])
    result = exchange_curves(time, _step, params, aif_onset_seconds=10)
    forcing = np.array([params.blood_flow / params.plasma_volume, 0])
    augmented = np.zeros((4, 4))
    augmented[:2, :2] = params.matrix
    augmented[:2, 2] = forcing
    augmented[3, 0] = -params.plasma_flow
    augmented[3, 2] = params.blood_flow
    oracle = np.array([expm(augmented * max(t - 10, 0)) @ [0, 0, 1, 0] for t in time])
    np.testing.assert_allclose(result.plasma, oracle[:, 0], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(result.ees, oracle[:, 1], atol=1e-9, rtol=1e-9)
    mass = params.plasma_volume * result.plasma + params.ees_volume * result.ees
    np.testing.assert_allclose(mass, oracle[:, 3], atol=1e-10)
    assert params.vascular_mtt == pytest.approx(5)
    assert params.ktrans < min(params.plasma_flow, params.permeability)


def test_continuous_exchange_against_independent_matrix_convolution() -> None:
    """Adaptive convolution of the analytic input agrees with ODE integration."""
    params = _parameters()
    time = np.array([0, 10, 11.31, 14.78, 21.93, 35.8])
    result = exchange_curves(time, _aif, params, aif_onset_seconds=10, delay_seconds=0.37)
    expected = np.zeros((2, len(time)))
    for j, t in enumerate(time):
        horizon = t - 10.37
        if horizon <= 0:
            continue
        for i in range(2):
            expected[i, j] = params.blood_flow / params.plasma_volume * quad(
                lambda lag: expm(params.matrix * lag)[i, 0] * float(_aif(t - 0.37 - lag)),
                0, horizon, epsabs=1e-11, epsrel=1e-10,
            )[0]
    np.testing.assert_allclose(result.plasma, expected[0], rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(result.ees, expected[1], rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_transport_drives_both_compartments_without_scan_renormalization(family: str) -> None:
    """A dispersed step has exactly the transport CDF as local blood input."""
    params = _parameters()
    kernel = transport_kernel(family, 2.0)
    time = np.array([0, 8, 10, 10.2, 11.7, 15.3, 24, 40])
    result = exchange_curves(
        time, _step, params, aif_onset_seconds=10, delay_seconds=-0.37, transport=kernel,
    )
    np.testing.assert_allclose(result.local_blood_aif, kernel.cumulative(time - 9.63),
                               atol=2e-6, rtol=2e-5)
    expected = np.zeros((2, len(time)))
    for j, t in enumerate(time):
        if t <= 9.63:
            continue
        for i in range(2):
            expected[i, j] = params.blood_flow / params.plasma_volume * quad(
                lambda lag: expm(params.matrix * lag)[i, 0] * float(kernel.cumulative(t - 9.63 - lag)),
                0, t - 9.63, epsabs=1e-10,
            )[0]
    np.testing.assert_allclose(result.plasma, expected[0], atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(result.ees, expected[1], atol=3e-6, rtol=3e-5)
    short = exchange_curves(time[:6], _step, params, aif_onset_seconds=10,
                            delay_seconds=-0.37, transport=kernel)
    np.testing.assert_allclose(short.plasma, result.plasma[:6], atol=3e-6)
    np.testing.assert_allclose(short.ees, result.ees[:6], atol=3e-6)


def test_gre_signal_is_full_normalized_signal_not_signed_tail() -> None:
    """Independent direct signal equation verifies the log-domain evaluation."""
    civ, cev = np.array([0, 0.3, 0.08]), np.array([0, 0.03, 0.05])
    r1 = 1 / 1.8 + 4.2 * cev
    e = np.exp(-1.5 * r1)
    e0 = np.exp(-1.5 / 1.8)
    signal_ratio = ((1 - e) / (1 - 0.5 * e)) / ((1 - e0) / (1 - 0.5 * e0))
    signal_ratio *= np.exp(-0.045 * (5.3 * cev + 44 * np.abs(civ - cev)))
    np.testing.assert_allclose(np.exp(-0.045 * _signal(civ, cev)), signal_ratio, rtol=1e-13)
    np.testing.assert_array_equal(_signal(civ, cev, r1_ees=0, r2_intravascular=0, r2_ees=0), 0)
    assert _signal([0], [0])[0] == 0


def test_t1_and_t2_effects_have_opposite_late_signal_directions() -> None:
    """The same extravascular concentration yields enhancement or signal loss."""
    t1 = _signal([0], [0.03], flip_angle_degrees=90, r2_ees=5.3)
    t2 = _signal([0], [0.03], flip_angle_degrees=30, r2_ees=30)
    assert t1[0] < 0 < t2[0]
    clean = np.array([0, 0, 0, t1[0], t2[0]])
    mask = np.arange(5) < 3
    settings = dict(te_seconds=0.045, s0=1000, noise_model="rician_signal", noise_sd=50, seed=7)
    first = signal_observation(clean, mask, **settings)
    second = signal_observation(clean, mask, **settings)
    np.testing.assert_array_equal(first.signal, second.signal)
    assert first.clean_signal[3] > 1000 > first.clean_signal[4]
    assert first.baseline_snr == 20


@pytest.mark.parametrize("field,value", [
    ("blood_flow", 0), ("blood_volume", 0), ("ees_volume", 0),
    ("permeability", -1), ("hematocrit", 1), ("blood_volume", 0.8),
    ("permeability", np.nan),
])
def test_invalid_exchange_domain_raises(field: str, value: float) -> None:
    """No physical-domain clipping changes the declared truth."""
    with pytest.raises(ValueError):
        replace(_parameters(), **{field: value})


@pytest.mark.parametrize("mean_seconds", [0.01, 1e-4])
def test_unresolved_dispersion_raises(mean_seconds: float) -> None:
    """Insufficient numerical resolution cannot be archived as accepted truth."""
    with pytest.raises(RuntimeError, match="refinement tolerance"):
        exchange_curves([0, 10.1, 11, 20], _aif, _parameters(), aif_onset_seconds=10,
                        transport=transport_kernel("lognormal", mean_seconds),
                        initial_step_seconds=1, max_refinements=2)
