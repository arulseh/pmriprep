"""Independent integration and fixed-delay fitting checks for the shared input."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, lsq_linear, minimize

from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.profile_fitting import (
    fit_extended_tofts,
    fit_patlak,
    fit_tofts,
    tofts_predict,
)


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


def test_patlak_box_matches_independent_linear_solver() -> None:
    """Protect interior, lower/upper edges and residuals on retained observations."""
    times, aif = _input()
    context = DelayedPchip(aif, times, -2., np.arange(2, len(times), 2))
    design = np.column_stack((context.convolution(0.), context.cp))
    curves = np.array([[.1, .03], [-.2, .2], [.2, 1.4], [5.5, .1]]) @ design.T
    curves += np.random.default_rng(7).normal(size=curves.shape) * .001
    fitted = fit_patlak(context, curves)
    assert np.isnan(fitted.params[:, 1]).all()
    for i, y in enumerate(curves):
        independent = lsq_linear(design, y, bounds=([0., 0.], [5., 1.]), tol=1e-14)
        np.testing.assert_allclose(fitted.params[i, [0, 2]], independent.x, atol=1e-8)
        np.testing.assert_allclose(fitted.sse[i], np.sum((design @ independent.x - y)**2), atol=1e-10)


@pytest.mark.parametrize("dt,delay", [(2., -1.3), (7.4, 5.)])
def test_et_recovery_feasible_benchmark_and_aif_scaling(dt: float, delay: float) -> None:
    """Catch an avoidable poorer optimum or failure of feasible scale reciprocity."""
    times, aif = _input(dt)
    context = DelayedPchip(aif, times, delay)
    truth = np.array([[.03, .2, .01], [.3, .4, .1]])
    curves = np.stack([k * _quadrature(aif, times, delay, k / ve) + vp * context.cp
                       for k, ve, vp in truth])
    fitted = fit_extended_tofts(context, curves)
    np.testing.assert_allclose(fitted.params, truth, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(fitted.sse, 0., atol=1e-12)
    noisy = curves + np.random.default_rng(12).normal(size=curves.shape) * .002
    fitted = fit_extended_tofts(context, noisy)
    assert np.all(fitted.sse <= np.sum((curves - noisy)**2, axis=1) + 1e-12)
    assert np.all(fitted.params[:, 1:].sum(axis=1) <= 1. + 1e-12)
    scaled = fit_extended_tofts(DelayedPchip(aif * 1.7, times, delay), noisy)
    np.testing.assert_allclose(scaled.params * 1.7, fitted.params, atol=1e-6)
    np.testing.assert_allclose(scaled.sse, fitted.sse, atol=1e-10)
    prediction = np.stack([k * context.convolution(k / ve) + vp * context.cp
                           for k, ve, vp in fitted.params])
    np.testing.assert_allclose(fitted.sse, np.sum((prediction - noisy)**2, axis=1), atol=1e-12)


def test_et_joint_volume_edge_matches_constrained_optimizer() -> None:
    """The coupled constraint must be optimized, not clipped after fitting."""
    times, aif = _input()
    context = DelayedPchip(aif, times, 3.5)
    y = .5 * context.convolution(.5 / .85) + .35 * context.cp

    def objective(p: np.ndarray) -> float:
        prediction = p[0] * context.convolution(p[0] / p[1]) + p[2] * context.cp
        return float(np.sum((prediction - y)**2))

    solutions = [minimize(objective, start, method="SLSQP",
                          bounds=[(0., 5.), (1e-4, 1.), (0., 1.)],
                          constraints=[{"type": "ineq", "fun": lambda p: 1. - p[1] - p[2]}],
                          options={"ftol": 1e-12, "maxiter": 300})
                 for start in ([.2, .5, .1], [.6, .7, .25])]
    assert all(result.success for result in solutions)
    fitted = fit_extended_tofts(context, y[None])
    np.testing.assert_allclose(fitted.params[0, 1:].sum(), 1., atol=1e-8)
    assert fitted.sse[0] <= min(result.fun for result in solutions) + 1e-9
    np.testing.assert_allclose(fitted.sse[0], objective(fitted.params[0]), atol=1e-12)


def test_vascular_fitters_handle_degenerate_and_invalid_batches() -> None:
    times, aif = _input()
    context = DelayedPchip(aif, times)
    for fit in (fit_patlak, fit_extended_tofts):
        zero = fit(context, np.zeros((1, len(times))))
        assert zero.params[0, 0] == 0. and zero.params[0, 2] == 0. and zero.sse[0] == 0.
        assert fit(context, np.empty((0, len(times)))).params.shape == (0, 3)
        rank_zero = fit(DelayedPchip(np.zeros_like(aif), times), np.ones((1, len(times))))
        np.testing.assert_allclose(rank_zero.sse, len(times))
        with pytest.raises(ValueError, match="member-by-frame"):
            fit(context, np.ones((1, len(times) - 1)))
        with pytest.raises(ValueError, match="member-by-frame"):
            fit(context, np.full((1, len(times)), np.nan))
        with pytest.raises(ValueError, match="retained observation"):
            fit(DelayedPchip(aif, times, observed=np.array([], dtype=int)), np.empty((1, 0)))
