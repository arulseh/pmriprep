"""Independent integral and fallback checks for continuous arterial transport."""

import numpy as np
import pytest
from typing import Literal
from scipy.integrate import quad
from scipy.special import gammainc
from scipy.interpolate import PchipInterpolator
from scipy.stats import gamma as gamma_distribution

from pmriprep.models.dsc.signal_cpi.batch import EnsembleConvolution
from pmriprep.models.dsc.signal_cpi.continuous import ContinuousCPIConvolution
from pmriprep.models.dsc.signal_cpi.dispersion import DispersionIntegrationError
from pmriprep.models.dsc.signal_cpi.transport import AssociatedConvolution, DispersedAIF, gamma_log_density
from pmriprep.models.dsc.signal_cpi import transport


def assert_convolution_accuracy(actual: np.ndarray, expected: np.ndarray) -> None:
    """Bound each curve's maximum error to 1e-8 of its reference peak."""
    scale = np.max(np.abs(expected), axis=-1, keepdims=True)
    assert np.all(scale > 0)
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=1e-8)


@pytest.mark.parametrize('mode,rate', [(0., 2.), (.15, 1.5), (2., .7), (30., 20.)])
def test_constant_aif_and_residue(mode: float, rate: float) -> None:
    """Preserve unit-area transport, acquisition-end tail and causal origin."""
    time = np.arange(0., 24., 3.)
    base = ContinuousCPIConvolution(np.ones_like(time), time)
    prepared = DispersedAIF(base, mode, rate, 48.)
    target = np.r_[np.linspace(0, 48, 83), 1e-9, 1e-6]
    shape = 1+mode*rate
    expected = gammainc(shape, rate*target)-gammainc(shape, rate*np.maximum(target-time[-1], 0.))
    assert_convolution_accuracy(prepared._raw(target), expected)
    def integral(t: np.ndarray) -> np.ndarray:
        t = np.maximum(t, 0.)
        return np.asarray(t*gammainc(shape, rate*t)-shape/rate*gammainc(shape+1, rate*t))
    expected_integral = integral(target)-integral(target-time[-1])
    actual = prepared.convolve(np.array([0., 20.]), np.ones(2), target)
    assert_convolution_accuracy(actual, expected_integral)


@pytest.mark.parametrize('interpolation', ['linear', 'pchip'])
def test_reference_signed_input_and_delays(interpolation: Literal['linear', 'pchip']) -> None:
    """Retain signed measured input, input interpolation and terminal residue."""
    time = np.array([0., .5, 2., 4., 7., 11., 18.])
    base = ContinuousCPIConvolution(np.array([1., -2., 20., 15., -1., 3., 0.]), time,
                                    aif_interpolation=interpolation)
    knots = np.tile([0., .3, 2., 9., 18.], (4, 1))
    values = np.exp(-knots/6)
    gamma = np.array([[.15, 1.5], [2., .7], [30., 20.], [0., 2.]])
    delays = np.array([-3., 2., -20., 0.])
    expected = EnsembleConvolution(base).evaluate(knots, values, np.ones(4), delays, gamma)
    actual = AssociatedConvolution(base).evaluate(knots, values, np.ones(4), delays, gamma)
    assert_convolution_accuracy(actual, expected)


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
@pytest.mark.parametrize('batch_size', [1, 8])
def test_scaling_permutation_and_cache_extension(scale: float, batch_size: int) -> None:
    """Share only the AIF/transport, not curve error scales or changing residues."""
    time = np.arange(0., 16., 2.)
    base = ContinuousCPIConvolution(scale*np.exp(-((time-5)/2)**2), time,
                                    aif_interpolation='pchip')
    knots = np.tile([0., .8, 5., 14.], (3, 1))
    values = np.exp(-knots/4)*np.array([1e-8, 1., 1e8])[:, None]
    gamma = np.tile([.15, 1.5], (3, 1))
    delay = np.array([2., 0., -5.])
    flow = np.array([.3, 1., 2.])
    expected = EnsembleConvolution(base).evaluate(knots, values, flow, delay, gamma)
    new = AssociatedConvolution(base, batch_size)
    actual = new.evaluate(knots, values, flow, delay, gamma)
    normalization = scale*np.array([1e-8, 1., 1e8])[:, None]
    assert_convolution_accuracy(actual/normalization, expected/normalization)
    permutation = np.array([2, 0, 1])
    shuffled = new.evaluate(knots[permutation], values[permutation], flow[permutation],
                           delay[permutation], gamma[permutation])
    np.testing.assert_allclose(shuffled/normalization[permutation], actual[permutation]/normalization[permutation],
                               rtol=3e-12, atol=3e-12)
    changed = values.copy()
    changed[:, 2] *= .5
    assert_convolution_accuracy(new.evaluate(knots, changed, flow, delay, gamma)/normalization,
        EnsembleConvolution(base).evaluate(knots, changed, flow, delay, gamma)/normalization)
    extended = delay-5.
    assert_convolution_accuracy(new.evaluate(knots, values, flow, extended, gamma)/normalization,
        EnsembleConvolution(base).evaluate(knots, values, flow, extended, gamma)/normalization)


@pytest.mark.parametrize('mode,rate', [(0., .001), (.001, 20.), (.5, 3.), (10., .8), (30., 20.)])
def test_independent_off_grid_cubic_aif(mode: float, rate: float) -> None:
    """Check raw integration and off-node representation against independent quad."""
    time = np.array([0., .4, 1.2, 3., 7., 15.])
    aif = np.array([.2, -.4, 5., 2., -.1, .3])
    base = ContinuousCPIConvolution(aif, time, aif_interpolation='pchip')
    prepared = DispersedAIF(base, mode, rate, 50.)
    interpolation = PchipInterpolator(time, aif)
    targets = np.array([.003, .317, 1.11, 2.79, 6.11, 14.91, 18.1, 30.47, 47.6])
    shape = 1+mode*rate
    expected = []
    for t in targets:
        upper = min(t, time[-1])
        points = time[(time > 0) & (time < upper)]
        value, error = quad(lambda u: float(interpolation(u)*gamma_distribution.pdf(t-u, shape, scale=1/rate)),
                            0., upper, points=points, epsabs=1e-12, epsrel=1e-12, limit=500)
        assert error < 2e-11
        expected.append(value)
    np.testing.assert_allclose(prepared._raw(targets)/prepared.scale,
                               np.asarray(expected)/prepared.scale, rtol=0, atol=1e-8)
    indices = np.searchsorted(prepared.left, targets, side='right')-1
    x = 2*(targets-prepared.left[indices])/(prepared.right-prepared.left)[indices]-1
    actual = np.polynomial.chebyshev.chebval(x, prepared.coefficients[indices].T, tensor=False)
    np.testing.assert_allclose(actual/prepared.scale, np.asarray(expected)/prepared.scale,
                               rtol=0, atol=1e-8)


def test_zero_input_and_invalid_parameters() -> None:
    """Zero input stays zero and invalid transport cannot produce a measurement."""
    time = np.arange(10.)
    base = ContinuousCPIConvolution(time*0, time)
    prepared = DispersedAIF(base, .15, 1.5, 20.)
    np.testing.assert_array_equal(prepared._raw(time), time*0)
    np.testing.assert_array_equal(prepared.convolve(np.array([0., 9.]), np.array([1., .1]), time), time*0)
    for args in [(-1., 1., 20.), (1., 0., 20.), (1., 1., -20.), (np.nan, 1., 20.)]:
        with pytest.raises(ValueError):
            DispersedAIF(base, *args)


@pytest.mark.parametrize('shape', [1., 1.225, 8.999, 9., 50., 601.])
def test_gamma_moments_reuse_only_exact_endpoints(shape: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated lags share CDF work; adjacent floating-point values stay distinct."""
    time = np.arange(32.) * 1.5
    target = np.r_[time, time, np.nextafter(time, np.inf)]
    lower = np.tile(target, 3)
    upper = lower + 1.5
    a = shape + np.arange(4)[:, None]
    expected = (transport.gammainc(a, lower), transport.gammainc(a, upper),
                transport.gammaincc(a, lower), transport.gammaincc(a, upper))
    original = transport.gammainc
    seen: list[np.ndarray] = []

    def record(a: np.ndarray, x: np.ndarray) -> np.ndarray:
        seen.append(x.copy())
        return original(a, x)

    monkeypatch.setattr(transport, "gammainc", record)
    actual = transport._gamma_interval_cdfs(shape, lower, upper)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    assert len(seen) == 1
    endpoints = np.r_[lower, upper]
    np.testing.assert_array_equal(seen[0][0], np.unique(endpoints))
    assert seen[0].size < endpoints.size / 4


def test_failed_quadrature_raises() -> None:
    """Unestablished numerical accuracy is never returned as an accepted curve."""
    time = np.arange(10.)
    prepared = DispersedAIF(ContinuousCPIConvolution(np.ones(10), time), .15, 1.5, 20.)
    x, w = prepared.rules[16]
    prepared.rules[16] = x, np.full_like(w, np.nan)
    with pytest.raises(DispersionIntegrationError):
        prepared._quadrature(np.array([5.]), np.array([0]), np.array([0]),
                             np.array([4.]), np.array([5.]), np.array([1e-14]))


@pytest.mark.parametrize('fraction', [0., 1e-15, 1e-8, .01, .249, .251, .9])
def test_near_origin_intervals_preserve_gamma_mass(fraction: float) -> None:
    """A lower endpoint near zero must not require endless ordinary refinements."""
    time = np.arange(10.)
    prepared = DispersedAIF(ContinuousCPIConvolution(np.ones(10), time), .15, 1.5, 20.)
    hi, lo = 1.5, fraction*1.5
    result = prepared._quadrature(np.array([2.]), np.array([0]), np.array([0]),
                                  np.array([lo]), np.array([hi]), np.array([4e-15]))
    expected = gammainc(prepared.shape, prepared.rate*hi)-gammainc(prepared.shape, prepared.rate*lo)
    np.testing.assert_allclose(result, [expected], rtol=3e-13, atol=3e-15)


@pytest.mark.parametrize('shape', [1., 1.225, 8.999, 9., 50., 601.])
def test_stable_gamma_normalization(shape: float) -> None:
    """Check the high-shape density independently across peak and tails."""
    x = np.linspace(max(.001, shape-8*np.sqrt(shape)), shape+8*np.sqrt(shape), 201)
    expected = gamma_distribution.pdf(x, shape)
    actual = np.exp(gamma_log_density(x, shape, 1.))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)
    zero = np.exp(gamma_log_density(np.array([0.]), shape, 1.))
    np.testing.assert_array_equal(zero, np.array([1. if shape == 1. else 0.]))


def test_uncertified_preparation_uses_checked_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fast-path accuracy failure falls back without changing the model output."""
    time = np.arange(0., 16., 2.)
    base = ContinuousCPIConvolution(np.exp(-((time-5)/2)**2), time, aif_interpolation='pchip')
    knots = np.tile([0., 2., 5., 14.], (2, 1))
    values = np.exp(-knots/4)
    gamma = np.tile([.15, 1.5], (2, 1))
    delay, flow = np.array([0., -1.]), np.ones(2)
    expected = EnsembleConvolution(base).evaluate(knots, values, flow, delay, gamma)

    def fail(self: DispersedAIF) -> None:
        raise DispersionIntegrationError('uncertified interpolation')

    monkeypatch.setattr(DispersedAIF, '_prepare', fail)
    new = AssociatedConvolution(base)
    actual = new.evaluate(knots, values, flow, delay, gamma)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
    assert new.stats['failed_preparations'] == 1
    assert new.fallback_rows == 2
    np.testing.assert_array_equal(new.evaluate(knots, values, flow, delay, gamma), actual)
    assert new.stats['failed_preparations'] == 1
