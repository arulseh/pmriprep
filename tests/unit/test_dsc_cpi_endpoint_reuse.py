"""Gamma endpoint geometry reuse never reuses another kernel's CDF values."""

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi.continuous import ContinuousCPIConvolution
from pmriprep.models.dsc.signal_cpi.transport import (
    DispersedAIF, GammaEndpointCache, _gamma_interval_cdfs,
)


@pytest.mark.parametrize('shape,rate', [(1., .05), (1.1, 1.5), (5., 2.), (601., 20.)])
@pytest.mark.parametrize('scale', [1e-10, 1., 1e10])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cdf_values_are_exact_for_each_kernel_and_cutoff(
    shape: float, rate: float, scale: float, dtype: type[np.float32] | type[np.float64],
) -> None:
    """Clipping and scaling may merge endpoints without changing their CDFs."""
    lower = (scale*np.array([0., .001, .1, 1., np.nextafter(1., 2.), 2.])).astype(dtype)
    upper = (scale*np.array([.1, 1., 2., 2.01, 10., 100.])).astype(dtype)
    cache = GammaEndpointCache()
    for limit in scale*np.array([3., 8., 100., 101., 2.01]):
        limit = float(limit)
        for changed_rate in (rate, np.nextafter(rate, np.inf), rate*1.01):
            expected = _gamma_interval_cdfs(shape, changed_rate*lower,
                                             changed_rate*np.minimum(upper, limit))
            actual = cache.evaluate(shape, changed_rate, lower, upper, float(limit))
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)
            assert len(cache.entries) == int(dtype is np.float64)
    if dtype is np.float32:
        return
    layout = next(iter(cache.entries.values()))
    assert not layout.points.flags.writeable and not layout.inverse.flags.writeable
    lower[1] += scale*.001
    cache.evaluate(shape, rate, lower, upper, 3*scale)
    assert len(cache.entries) == 2


def test_empty_endpoints_remain_empty() -> None:
    """No active arterial intervals require no CDF or posterior contribution."""
    cache = GammaEndpointCache()
    actual = cache.evaluate(1., 1., np.array([]), np.array([]), 3.)
    assert all(array.shape == (4, 0) for array in actual)


def test_endpoint_cache_is_bounded_and_precision_sensitive() -> None:
    """Keep aggregate storage bounded across distinct kernels and acquisitions."""
    cache = GammaEndpointCache()
    lower, upper = np.arange(10.), np.arange(10.)+1
    for offset in range(25):
        cache.evaluate(2., 1., lower+offset, upper+offset, 100.)
    assert len(cache.entries) == 16
    assert cache.nbytes == sum(v.points.nbytes+v.inverse.nbytes+sum(map(len, k))
                              for k, v in cache.entries.items())
    # Half-open endpoints are all distinct, so the payload exceeds eight MB.
    lower = np.arange(260_000.)*2
    upper = lower+1
    before = cache.nbytes
    cache.evaluate(2., 1., lower, upper, 1e6)
    assert cache.nbytes == before and len(cache.entries) == 16
    cache = GammaEndpointCache()
    lower, upper = np.arange(70_000.)*2, np.arange(70_000.)*2+1
    for offset in range(5):
        cache.evaluate(2., 1., lower+offset*.1, upper+offset*.1, 1e6)
    assert 0 < len(cache.entries) < 5 and cache.nbytes <= 8_000_000
    cache = GammaEndpointCache()
    for dtype in (np.float32, np.float64):
        cache.evaluate(2., 1., np.arange(5., dtype=dtype), np.arange(5., dtype=dtype)+1, 10.)
    assert len(cache.entries) == 1


@pytest.mark.parametrize('mode,rate', [(0., 1.5), (.15, 1.5), (2., 2.), (30., 20.), (1., .05)])
def test_complete_preparation_matches_without_endpoint_cache(mode: float, rate: float) -> None:
    """All coefficients, independent checks and resulting curves remain exact."""
    time = np.array([0., .4, 1.2, 3., 7., 15.])
    base = ContinuousCPIConvolution(np.array([.2, -.4, 5., 2., -.1, .3]), time,
                                    aif_interpolation='pchip')
    cache = GammaEndpointCache()
    for changed in (rate, rate*1.001):
        cold = DispersedAIF(base, mode, changed, 25.)
        cached = DispersedAIF(base, mode, changed, 25., endpoint_cache=cache)
        for name in ('left', 'right', 'coefficients', 'error'):
            np.testing.assert_array_equal(getattr(cached, name), getattr(cold, name))
        knots, values = np.array([0., 1., 5., 15.]), np.array([1., .9, .4, .1])
        target = np.r_[-3., 0., np.linspace(.001, 25., 100)]
        np.testing.assert_array_equal(cached.convolve(knots, values, target),
                                       cold.convolve(knots, values, target))
