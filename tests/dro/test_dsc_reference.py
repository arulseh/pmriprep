"""Verify DSC reference data without fitting a perfusion model."""

from functools import partial
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.signal import fftconvolve
from scipy.special import gamma

from . import dsc_reference as reference
from .conftest import fine_grid_forward, generate_dsc_aif_dro
from .dsc_reference import (
    ContinuousResidue,
    analytic_residue,
    continuous_tissue,
    gamma_variate_aif,
    recirculating_gamma_aif,
    spline_residue,
)


def _aif(time: Any) -> np.ndarray:
    """Evaluate the same continuous bolus at arbitrary requested times."""
    return gamma_variate_aif(
        time, onset_seconds=10.3, peak=4.0, alpha=2.5, beta_seconds=3.0,
    )


def test_gamma_aif_has_analytic_peak_and_unchanged_sample_values() -> None:
    """Changing the sample grid does not renormalize the underlying bolus."""
    aif, time = generate_dsc_aif_dro()
    function = partial(
        gamma_variate_aif, onset_seconds=15.0, peak=40.0,
        alpha=2.5, beta_seconds=3.0,
    )
    np.testing.assert_allclose(function(time), aif, rtol=2e-14, atol=1e-14)
    assert float(function(22.5)) == pytest.approx(40.0)
    np.testing.assert_array_equal(function([-4.0, 0.0, 15.0]), 0.0)
    shifted_grid = time + 0.37
    assert np.max(function(shifted_grid)) < 40.0
    np.testing.assert_array_equal(function(shifted_grid)[::3], function(shifted_grid[::3]))


def _recirculating_aif(time: Any, **overrides: Any) -> np.ndarray:
    """Evaluate an explicitly parameterized second-pass control."""
    settings = dict(
        onset_seconds=10.3, peak=4.0, alpha=2.5, beta_seconds=3.0,
        recirculation_fraction=0.35, recirculation_delay_seconds=15.0, broadening=2.0,
    )
    return recirculating_gamma_aif(time, **{**settings, **overrides})


def test_recirculation_causality_zero_fraction_and_point_sampling() -> None:
    """The first pass is unchanged and sampled values do not depend on scan extent."""
    times = np.array([-1.0, 0.0, 10.3, 11.27, 18.43, 25.3, 30.11, 48.07, 180.0])
    np.testing.assert_array_equal(_recirculating_aif(times, recirculation_fraction=0), _aif(times))
    np.testing.assert_array_equal(_recirculating_aif(times[times <= 25.3]), _aif(times[times <= 25.3]))
    assert np.all(_recirculating_aif(times[times > 25.3]) > _aif(times[times > 25.3]))
    np.testing.assert_array_equal(_recirculating_aif(times)[::2], _recirculating_aif(times[::2]))
    np.testing.assert_array_equal(_recirculating_aif(times)[:5], _recirculating_aif(times[:5]))
    for t in times:
        assert float(_recirculating_aif(t)) == float(_recirculating_aif([t])[0])


@pytest.mark.parametrize("fraction", [0.1, 0.35])
@pytest.mark.parametrize("broadening", [1.0, 2.0, 3.0])
def test_recirculation_full_area_ratio(fraction: float, broadening: float) -> None:
    """The second-pass area is fixed analytically, independently of frame spacing."""
    first_area = 4.0 * 3.0 * np.exp(2.5) * gamma(3.5) / 2.5**2.5

    def function(t: float) -> float:
        return float(_recirculating_aif(
            t, recirculation_fraction=fraction, broadening=broadening,
        ))

    before, _ = quad(function, 10.3, 25.3, epsabs=1e-9, epsrel=1e-10)
    after, _ = quad(function, 25.3, np.inf, epsabs=1e-9, epsrel=1e-10)
    assert (before + after) / first_area == pytest.approx(1 + fraction, rel=1e-9)


@pytest.mark.parametrize("dt", [1.5, 3.0])
def test_same_recirculating_input_drives_tissue_and_measured_aif(dt: float) -> None:
    """A direct fine-grid convolution checks both passes, without a tissue-only tail."""
    time = np.arange(0.0, 180.0, dt)
    residue = analytic_residue("gamma", 4.0)
    flow, delay = 0.01, 0.37
    result = continuous_tissue(
        time, _recirculating_aif, residue, flow=flow, aif_onset_seconds=10.3,
        delay_seconds=delay, aif_breakpoints=[25.3],
    )
    errors = []
    for factor in (128, 256):
        step = dt / factor
        fine_time = np.arange((len(time) - 1) * factor + 1) * step
        arterial = _recirculating_aif(fine_time - delay)
        r = residue.evaluate(fine_time)
        convolution = fftconvolve(arterial, r)[:len(fine_time)] * step
        convolution -= step / 2 * (arterial[0] * r + arterial * r[0])
        errors.append(float(np.max(np.abs(flow * convolution[::factor] - result.values))
                            / np.max(result.values)))
    assert errors[1] < errors[0]
    assert errors[1] < 5e-5
    assert residue.mtt_seconds == 4.0
    aif = _recirculating_aif(time)
    area_ratio = np.trapezoid(result.values, time) / np.trapezoid(aif, time)
    assert area_ratio == pytest.approx(flow * residue.mtt_seconds, rel=1e-3)


@pytest.mark.parametrize("overrides", [
    {"recirculation_fraction": -0.1}, {"recirculation_fraction": np.nan},
    {"recirculation_fraction": np.inf}, {"recirculation_delay_seconds": 0},
    {"recirculation_delay_seconds": -1}, {"recirculation_delay_seconds": np.nan},
    {"broadening": 0.5}, {"broadening": np.inf},
])
def test_invalid_recirculation_settings_are_rejected(overrides: dict[str, Any]) -> None:
    """A malformed second-pass definition must not silently generate reference data."""
    with pytest.raises(ValueError):
        _recirculating_aif([0, 20, 40], **overrides)


@pytest.mark.parametrize("family", ["exponential", "gamma", "box", "triangle"])
@pytest.mark.parametrize("mtt", [0.4, 4.0, 30.0])
def test_residue_causality_monotonicity_and_full_area(family: Any, mtt: float) -> None:
    """All families have unit height and the same declared full-area meaning."""
    residue = analytic_residue(family, mtt)
    time = np.r_[-1.0, np.linspace(0.0, 100.0 * mtt, 1001)]
    values = residue.evaluate(time)
    assert values[0] == 0.0
    assert values[1] == 1.0
    assert np.all(values >= 0)
    assert np.all(np.diff(values[1:]) <= 0)
    assert residue.mtt_seconds == mtt
    assert residue.area_until(100 * mtt) == pytest.approx(mtt, rel=1e-10)
    assert residue.area_until(0.0) == 0.0


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0, 8.0])
def test_gamma_shapes_preserve_full_mtt(shape: float) -> None:
    """Shape variation does not accidentally vary the specified MTT."""
    residue = analytic_residue("gamma", 4.0, gamma_shape=shape)
    assert residue.area_until(400.0) == pytest.approx(4.0, rel=1e-9)
    if shape == 1.0:
        t = np.linspace(0.0, 30.0, 100)
        np.testing.assert_allclose(residue.evaluate(t), np.exp(-t / 4.0), atol=1e-15)


@pytest.mark.parametrize("delay", [-5.0, -2.0, -0.37, 0.0, 0.37, 2.0, 5.0])
@pytest.mark.parametrize("family", ["exponential", "gamma", "box", "triangle"])
def test_convolution_matches_closed_form_with_signed_delay(family: Any, delay: float) -> None:
    """A step input has an independent closed form, including sharp residue edges."""
    time = np.r_[-1.0, 0.0, 7.11, 10.3, 10.301, 12.22, 18.9, 39.7, 80.0]
    onset, mtt, flow = 10.3, 4.0, 0.03
    residue = analytic_residue(family, mtt)
    result = continuous_tissue(
        time, lambda t: np.asarray(np.where(np.asarray(t) >= onset, 2.0, 0.0)),
        residue, flow=flow, aif_onset_seconds=onset, delay_seconds=delay,
    )
    elapsed = np.maximum(time - onset - delay, 0.0)
    if family == "exponential":
        area = -mtt * np.expm1(-elapsed / mtt)
    elif family == "gamma":
        x = elapsed / (mtt / 3.0)
        area = mtt / 3.0 * (3.0 - np.exp(-x) * (3.0 + 2.0 * x + x * x / 2.0))
    elif family == "box":
        area = np.minimum(elapsed, mtt)
    else:
        clipped = np.minimum(elapsed, 2.0 * mtt)
        area = clipped - clipped**2 / (4.0 * mtt)
    np.testing.assert_allclose(result.values, flow * 2.0 * area, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(result.values[time <= onset + delay], 0.0)
    assert np.max(result.quadrature_error) < 1e-9


def test_scan_limited_area_does_not_replace_full_truth() -> None:
    """An incompletely acquired exponential tail does not change the true MTT."""
    residue = analytic_residue("exponential", 30.0)
    assert residue.area_until(60.0) == pytest.approx(30.0 * (1.0 - np.exp(-2.0)))
    assert residue.area_until(60.0) < residue.area_until(90.0) < residue.mtt_seconds
    assert residue.mtt_seconds == 30.0


def test_spline_area_support_and_input_isolation() -> None:
    """Polynomial integration preserves normalization and does not borrow input arrays."""
    knots, values = np.array([0.0, 3.0, 6.0]), np.array([1.0, 0.5, 0.0])
    residue = spline_residue(knots, values)
    knots[1], values[1] = 1.0, 0.9
    time = np.array([-1.0, 0.0, 1.5, 3.0, 6.0, 8.0])
    np.testing.assert_allclose(residue.evaluate(time), [0, 1, 0.75, 0.5, 0, 0])
    assert residue.mtt_seconds == pytest.approx(3.0)
    assert residue.area_until(30.0) == pytest.approx(3.0)
    assert residue.area_until(3.0) == pytest.approx(2.25)


REFINEMENT_RESIDUES = [
    pytest.param(analytic_residue("exponential", 4.0), id="exponential"),
    pytest.param(analytic_residue("gamma", 4.0), id="gamma3"),
    pytest.param(analytic_residue("gamma", 4.0, gamma_shape=0.7), id="gamma0.7"),
    pytest.param(analytic_residue("triangle", 4.0), id="triangle"),
    pytest.param(spline_residue([0, 1, 3, 7, 20], [1, 0.98, 0.6, 0.15, 0]), id="pchip"),
]


@pytest.mark.parametrize("residue", REFINEMENT_RESIDUES)
@pytest.mark.parametrize("dt", [1.5, 3.0])
def test_adaptive_convolution_agrees_with_independent_grid_refinement(
    residue: ContinuousResidue, dt: float,
) -> None:
    """Two direct continuous-grid trapezoid convolutions converge to adaptive integration."""
    time = np.arange(0.0, 90.0, dt)
    delay, flow = 0.37, 0.01
    result = continuous_tissue(
        time, _aif, residue, flow=flow, aif_onset_seconds=10.3, delay_seconds=delay,
    )
    peak = float(np.max(result.values))
    errors = []
    for factor in (256, 512):
        step = dt / factor
        fine_time = np.arange((len(time) - 1) * factor + 1) * step
        arterial, r = _aif(fine_time - delay), residue.evaluate(fine_time)
        convolution = fftconvolve(arterial, r)[:len(fine_time)] * step
        convolution -= step / 2.0 * (arterial[0] * r + arterial * r[0])
        sampled = flow * convolution[::factor]
        errors.append(float(np.max(np.abs(sampled - result.values)) / peak))
    assert errors[1] < errors[0]
    assert errors[1] < 5e-5  # 0.005% of peak, below the recovery effects being assessed.
    assert np.max(result.quadrature_error) / peak < 1e-8


def test_frame_interpolation_error_persists_with_grid_refinement() -> None:
    """Increasing integration resolution cannot undo interpolation of coarse samples."""
    time = np.arange(60) * 1.5
    aif = _aif(time)
    residue = analytic_residue("exponential", 4.0)
    truth = continuous_tissue(
        time, _aif, residue, flow=1.0, aif_onset_seconds=10.3,
    ).values
    for oversample in (20, 80):
        frame_interpolated = fine_grid_forward(aif, residue.evaluate(time), 1.5, oversample)
        assert np.max(np.abs(frame_interpolated - truth)) / np.max(truth) > 1e-3


def test_explicit_aif_breakpoints_and_zero_flow() -> None:
    """A finite input pulse is integrated at its edges; zero flow stays zero."""
    def pulse(t: Any) -> np.ndarray:
        return np.asarray(1.0 if 2.0 <= float(t) < 2.2 else 0.0)

    residue = analytic_residue("box", 0.1)
    result = continuous_tissue(
        [0.0, 2.03, 2.15, 2.25, 2.3, 3.0], pulse, residue,
        flow=2.0, aif_onset_seconds=2.0, aif_breakpoints=[2.2],
    )
    np.testing.assert_allclose(result.values, [0, 0.06, 0.2, 0.1, 0, 0], atol=1e-14)
    zero = continuous_tissue([0, 1, 2], pulse, residue, flow=0.0, aif_onset_seconds=2.0)
    np.testing.assert_array_equal(zero.values, 0.0)
    np.testing.assert_array_equal(zero.quadrature_error, 0.0)


@pytest.mark.parametrize("time", [[], [0, 0], [1, 0], [0, np.nan], [[0, 1]]])
def test_invalid_sampling_is_rejected(time: Any) -> None:
    """Invalid sampling must not silently create a reference curve."""
    with pytest.raises(ValueError, match="time_seconds"):
        continuous_tissue(
            time, _aif, analytic_residue("exponential", 4), flow=1.0, aif_onset_seconds=10.3,
        )


@pytest.mark.parametrize("overrides", [
    {"flow": -1}, {"flow": np.inf}, {"delay_seconds": np.nan},
    {"aif_onset_seconds": np.inf}, {"aif_breakpoints": [np.nan]},
    {"epsabs": 0}, {"epsrel": -1},
])
def test_invalid_convolution_settings_are_rejected(overrides: dict[str, Any]) -> None:
    """Fail on invalid scale, support and numerical controls."""
    settings: dict[str, Any] = {"flow": 1.0, "aif_onset_seconds": 10.3, **overrides}
    with pytest.raises(ValueError):
        continuous_tissue([0, 20], _aif, analytic_residue("exponential", 4), **settings)


@pytest.mark.parametrize("knots,values", [
    ([0], [1]), ([1, 2], [1, 0]), ([0, 0], [1, 0]),
    ([0, 1], [1]), ([0, 1], [1, 0.1]), ([0, 1, 2], [1, 1.1, 0]),
    ([0, np.nan], [1, 0]), ([0, 1], [np.nan, 0]),
])
def test_invalid_spline_is_rejected(knots: Any, values: Any) -> None:
    """Reject noncausal, rising, unnormalized and undefined spline residues."""
    with pytest.raises(ValueError, match="Spline"):
        spline_residue(knots, values)


def test_integration_failure_is_not_saved_as_ground_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed quadrature diagnostic is an error, even with a finite returned value."""
    monkeypatch.setattr(reference, "quad", lambda *args, **kwargs: (1.0, 1.0, {}, "failure"))
    with pytest.raises(RuntimeError, match="Reference quadrature failed"):
        continuous_tissue(
            [0, 20], _aif, analytic_residue("exponential", 4), flow=1.0, aif_onset_seconds=10.3,
        )
