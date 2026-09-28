"""Cached arterial geometry preserves the complete plain convolution arithmetic."""

from typing import Literal

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from pmriprep.models.dsc.signal_cpi.batch import EnsembleConvolution
from pmriprep.models.dsc.signal_cpi.continuous import ContinuousCPIConvolution, polynomial_overlaps
from pmriprep.models.dsc.signal_cpi import native


@pytest.fixture(autouse=True)
def python_geometry_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """These cache-identity checks characterize the Python fallback implementation."""
    monkeypatch.setattr(native, "_kernel", None)


def uncached_plain(base: ContinuousCPIConvolution, knots: np.ndarray, values: np.ndarray,
                   target: np.ndarray) -> np.ndarray:
    """Characterize batched polynomial arithmetic without any persistent geometry."""
    coefficients = np.stack([np.column_stack((PchipInterpolator(x, y).c, [0., 0., 0., y[-1]]))
                             for x, y in zip(knots, values, strict=True)])
    rows, frames, ai, ri, lo, hi = polynomial_overlaps(base.left, base.right, knots, target)
    half = (hi-lo)/2
    points = (hi+lo)[:, None]/2+half[:, None]*base.nodes
    local = points-knots[rows, ri, None]
    c = coefficients[rows, :, ri]
    residue = ((c[:, 0, None]*local+c[:, 1, None])*local+c[:, 2, None])*local+c[:, 3, None]
    alocal = target[rows, frames, None]-points-base.left[ai, None]
    if base.aif_coefficients is None:
        aif = base.amplitude[ai, None]+base.slope[ai, None]*alocal
    else:
        c = base.aif_coefficients[:, ai]
        aif = ((c[0, :, None]*alocal+c[1, :, None])*alocal+c[2, :, None])*alocal+c[3, :, None]
    # Match each voxel's reduction shape so BLAS uses the same summation order.
    integral = np.empty(rows.size)
    for row in range(len(knots)):
        selected = rows == row
        integral[selected] = half[selected]*((aif[selected]*residue[selected]) @ base.weights)
    return np.bincount(rows*target.shape[1]+frames, weights=integral,
                       minlength=target.size).reshape(target.shape)


@pytest.mark.parametrize('interpolation', ['linear', 'pchip'])
@pytest.mark.parametrize('points', [2, 5, 12])
@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_cached_geometry_retains_exact_arithmetic(
    interpolation: Literal['linear', 'pchip'], points: int, scale: float,
) -> None:
    """Changing amplitudes reuses geometry; changed knots/delays never reuse it."""
    rng = np.random.default_rng(92031)
    time = np.r_[0., np.cumsum(rng.uniform(.3, 2., 20))]
    aif = rng.normal(size=time.size)*scale
    aif[3:8] = 0.
    base = ContinuousCPIConvolution(aif, time, aif_interpolation=interpolation)
    convolution = EnsembleConvolution(base)
    knots = np.tile(np.linspace(0., time[-1], points), (8, 1))
    target = np.tile(time, (8, 1))
    target[3] += 3.2
    target[4] -= 2.7
    if points > 2:
        knots[5, 1] = np.nextafter(knots[5, 1], np.inf)
    values = np.exp(-knots/rng.uniform(1., 12., (8, 1)))
    expected = uncached_plain(base, knots, values, target)
    np.testing.assert_array_equal(convolution._plain(knots, values, target), expected)
    cached = dict(convolution._geometry_cache)
    values[:, 1:] *= .8
    updated = uncached_plain(base, knots, values, target)
    np.testing.assert_array_equal(convolution._plain(knots, values, target), updated)
    assert all(convolution._geometry_cache[key] is geometry for key, geometry in cached.items())
    reverse = np.arange(7, -1, -1)
    np.testing.assert_array_equal(convolution._plain(knots[reverse], values[reverse], target[reverse]),
                                   updated[reverse])
    np.testing.assert_array_equal(convolution._plain(knots, values, target[:, :0]), np.empty((8, 0)))


def test_geometry_cache_is_owned_and_bounded() -> None:
    """Changing caller arrays cannot corrupt cached geometry or grow it indefinitely."""
    time = np.arange(30.)
    base = ContinuousCPIConvolution(np.sin(time), time, aif_interpolation='pchip')
    convolution = EnsembleConvolution(base)
    knots = np.array([0., 1., 4., 10., 29.])
    target = time.copy()
    first = convolution._plain_geometry(knots, target)
    keys = list(convolution._geometry_cache)
    knots[1] = 1.01
    target += .01
    second = convolution._plain_geometry(knots, target)
    assert second is not first
    assert convolution._geometry_cache[keys[0]] is first
    assert all(not array.flags.writeable for geometry in (first, second)
               for array in (geometry.frames, geometry.residue, geometry.half, geometry.local, geometry.aif))
    for i in range(20):
        convolution._plain_geometry(knots, target+i*.05)
    assert len(convolution._geometry_cache) == 16
    assert keys[0] not in convolution._geometry_cache
    assert convolution._geometry_bytes == sum(value.nbytes+sum(map(len, key))
        for key, value in convolution._geometry_cache.items())
    assert convolution._geometry_bytes <= 8_000_000


def test_large_geometry_does_not_exceed_cache_byte_budget() -> None:
    """A long acquisition can bypass caching without changing numerical evaluation."""
    time = np.arange(480.)
    convolution = EnsembleConvolution(ContinuousCPIConvolution(np.ones_like(time), time,
                                                               aif_interpolation='pchip'))
    geometry = convolution._plain_geometry(np.array([0., 479.]), time)
    assert geometry.nbytes > 8_000_000
    assert not convolution._geometry_cache and convolution._geometry_bytes == 0


def test_aggregate_geometry_storage_evicts_before_entry_limit() -> None:
    """Several medium-sized entries cannot exceed the independent byte ceiling."""
    time = np.arange(270.)
    convolution = EnsembleConvolution(ContinuousCPIConvolution(np.ones_like(time), time,
                                                               aif_interpolation='pchip'))
    for offset in range(5):
        convolution._plain_geometry(np.array([0., 269.]), time+offset*.2)
    assert 0 < len(convolution._geometry_cache) < 5
    assert convolution._geometry_bytes <= 8_000_000


def test_geometry_keys_preserve_numeric_precision() -> None:
    """Different input dtypes do not inherit intermediate arithmetic from a cache."""
    time = np.arange(10.)
    convolution = EnsembleConvolution(ContinuousCPIConvolution(np.ones_like(time), time))
    knots = np.array([0., 1., 9.])
    first = convolution._plain_geometry(knots, time)
    second = convolution._plain_geometry(knots.astype(np.float32), time.astype(np.float32))
    assert first is not second and len(convolution._geometry_cache) == 2


def test_zero_aif_keeps_empty_geometry_and_zero_predictions() -> None:
    """Zero arterial input remains exactly zero for every nonzero residue."""
    time = np.arange(10.)
    convolution = EnsembleConvolution(ContinuousCPIConvolution(time*0., time))
    knots = np.array([[0., 9.]])
    values = np.array([[1., .1]])
    np.testing.assert_array_equal(convolution._plain(knots, values, time[None]), time[None]*0.)
