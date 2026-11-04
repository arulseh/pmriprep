"""Independent integration and fixed-delay fitting checks for the shared input."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.profile_fitting import fit_tofts, tofts_predict


def _input(dt: float = 7.4) -> tuple[np.ndarray, np.ndarray]:
    seconds = np.arange(0, 180, dt)
    aif = 0.04 + 5 * np.exp(-((seconds - 38) / 9) ** 2)
    aif += 0.3 * np.exp(-np.maximum(seconds - 45, 0) / 100)
    return seconds / 60, aif


def _quadrature(aif: np.ndarray, times: np.ndarray, delay: float, rate: float) -> np.ndarray:
    """Numerically integrate each segment, without polynomial moments."""
    clock = times - times[0]
    curve = PchipInterpolator(clock, aif)
    shift = delay / 60
    knots = clock + shift
    events = np.unique(np.r_[clock, knots[(knots > 0) & (knots < clock[-1])]])
    states = np.zeros(len(events))
    for j, (left, right) in enumerate(zip(events[:-1], events[1:], strict=True)):
        area = quad(
            lambda u: float(curve(np.clip(u - shift, 0, clock[-1])))
            * np.exp(-rate * (right - u)),
            left, right, epsabs=2e-13, epsrel=2e-13,
        )[0]
        states[j + 1] = states[j] * np.exp(-rate * (right - left)) + area
    return states[np.searchsorted(events, clock)]


@pytest.mark.parametrize("delay", [-7.4, 0., 5.])
def test_delayed_input_matches_independent_quadrature(delay: float) -> None:
    """Catch wrong shift sign, seconds/minutes, cubic moments and initial history."""
    times, aif = _input()
    context = DelayedPchip(aif, times, delay)
    rates = np.array([0., 1e-12, 0.7, 30.])
    expected = np.stack([_quadrature(aif, times, delay, rate) for rate in rates])
    np.testing.assert_allclose(context.convolution(rates), expected, atol=1e-11, rtol=1e-10)
    np.testing.assert_array_equal(context.convolution(rates)[:, 0], 0.)
    np.testing.assert_array_equal(context.convolution(.7), context.convolution(rates)[2])
    interpolant = PchipInterpolator(times, aif)
    np.testing.assert_allclose(context.cp, interpolant(np.clip(times - delay / 60, 0, times[-1])))


def test_held_endpoints_and_constant_input_limits() -> None:
    """Endpoint holding and zero/large-rate limits remain finite for signed shifts."""
    times, aif = _input()
    for delay in (-300., 300.):
        context = DelayedPchip(aif, times, delay)
        endpoint = aif[-1] if delay < 0 else aif[0]
        np.testing.assert_array_equal(context.cp, endpoint)
        assert (context.held_after if delay < 0 else context.held_before).all()
        np.testing.assert_allclose(context.convolution(0.), endpoint * times, atol=1e-13)
    for delay in (-300., -3.1, 0., 12., 300.):
        constant = DelayedPchip(np.full(len(times), .4), times, delay)
        for rate in (1e-12, 1., 1e12):
            expected = -.4 * np.expm1(-rate * times) / rate
            np.testing.assert_allclose(constant.convolution(rate), expected, atol=1e-13, rtol=1e-12)


def test_observation_selection_preserves_history_and_order() -> None:
    """Dropping peak observations must not discard their input contribution."""
    times, aif = _input()
    ids = np.array([20, 2, 15, 20])
    full = DelayedPchip(aif, times, -3.2)
    subset = DelayedPchip(aif, times + 4., -3.2, ids)
    ids[:] = 0
    np.testing.assert_array_equal(subset.observed, [20, 2, 15, 20])
    np.testing.assert_allclose(subset.cp, full.cp[subset.observed], atol=1e-12)
    rates = np.array([0., .3, 2.])
    np.testing.assert_allclose(subset.convolution(rates), full.convolution(rates)[:, subset.observed], atol=1e-12)
    aif[:] = 0
    assert np.any(full.cp > 0)


def test_invalid_clocks_indices_and_rates_fail_explicitly() -> None:
    times, aif = _input()
    for clock in (times[::-1], np.r_[times[:-1], times[-1] + .01]):
        with pytest.raises(ValueError, match="Uniform increasing"):
            DelayedPchip(aif, clock)
    for indices in (np.array([-1]), np.array([len(aif)]), np.array([1.2])):
        with pytest.raises(ValueError, match="observation indices"):
            DelayedPchip(aif, times, observed=indices)
    for delay in (np.nan, np.inf):
        with pytest.raises(ValueError, match="Finite matching"):
            DelayedPchip(aif, times, delay)
    with pytest.raises(ValueError, match="Finite matching"):
        DelayedPchip(np.full_like(aif, np.nan), times)
    context = DelayedPchip(aif, times)
    for rate in (-1., np.nan, np.ones((2, 2))):
        with pytest.raises(ValueError, match="rates"):
            context.convolution(rate)


@pytest.mark.parametrize("dt,delay", [(2., -1.3), (7.4, 5.)])
def test_direct_tofts_recovery_and_noisy_bounded_fit(dt: float, delay: float) -> None:
    """Direct two-parameter recovery and a competing optimizer protect the fitter."""
    times, aif = _input(dt)
    context = DelayedPchip(aif, times, delay)
    truth = np.array([[.03, .2], [.3, .4]])
    curves = np.stack([k * _quadrature(aif, times, delay, k / ve) for k, ve in truth])
    fitted = fit_tofts(context, curves)
    np.testing.assert_allclose(fitted.params, truth, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(fitted.sse, 0., atol=1e-16)
    noisy = curves + np.random.default_rng(42).normal(size=curves.shape) * .002
    profiled = fit_tofts(context, noisy)
    for index, curve in enumerate(noisy):
        alternative = least_squares(
            lambda p: tofts_predict(context, p) - curve, [.1, .3],
            bounds=([0., 1e-8], [5., 1.]),
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        assert profiled.sse[index] <= np.sum(alternative.fun**2) + 1e-10
        np.testing.assert_allclose(
            profiled.sse[index], np.sum((tofts_predict(context, profiled.params[index]) - curve)**2),
            atol=1e-15,
        )


def test_tofts_zero_empty_and_invalid_inputs() -> None:
    times, aif = _input()
    context = DelayedPchip(aif, times)
    zero = fit_tofts(context, np.zeros((1, len(times))))
    np.testing.assert_array_equal(zero.params, [[0., 0.]])
    np.testing.assert_array_equal(zero.sse, [0.])
    assert fit_tofts(context, np.empty((0, len(times)))).params.shape == (0, 2)
    with pytest.raises(ValueError, match="retained observation"):
        fit_tofts(DelayedPchip(aif, times, observed=np.array([], dtype=int)), np.empty((1, 0)))
    with pytest.raises(ValueError, match="member-by-frame"):
        fit_tofts(context, np.full((1, len(times)), np.nan))
    with pytest.raises(ValueError, match="physical bounds"):
        tofts_predict(context, np.array([-.1, .2]))
