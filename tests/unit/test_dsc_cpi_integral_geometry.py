"""Aggregate-bounded gamma geometry reuse preserves complete integral arithmetic."""

from dataclasses import fields, replace
import weakref

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from pmriprep.models.dsc.signal_cpi.continuous import (
    IntegralGeometryCache, PolynomialIntegral, polynomial_overlaps,
)
from pmriprep.models.dsc.signal_cpi import native


@pytest.fixture(autouse=True)
def python_geometry_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise exact Python geometry arithmetic independently of native dispatch."""
    monkeypatch.setattr(native, "_kernel", None)


def uncached(integral: PolynomialIntegral, knots: np.ndarray,
             coefficients: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Characterize the combined moment/quadrature calculation before reuse."""
    rows, frames, ai, ri, lo, hi = polynomial_overlaps(integral.left, integral.right, knots, target)
    full = ((lo == target[rows, frames]-integral.right[ai])
            & (hi == target[rows, frames]-integral.left[ai]))
    result = np.empty(lo.size)
    selected = ai[full]
    local = (hi[full]+lo[full])/2-knots[rows[full], ri[full]]
    c = coefficients[rows[full], :, ri[full]].T
    h, m = integral.half[selected], integral.moments[:, selected]
    result[full] = (
        (((c[0]*local+c[1])*local+c[2])*local+c[3])*m[0]
        + ((3*c[0]*local+2*c[1])*local+c[2])*h*m[1]
        + (3*c[0]*local+c[1])*h**2*m[2] + c[0]*h**3*m[3])
    partial = ~full
    half = (hi[partial]-lo[partial])/2
    points = (hi[partial]+lo[partial])[:, None]/2+half[:, None]*integral.nodes
    local = points-knots[rows[partial], ri[partial], None]
    c = coefficients[rows[partial], :, ri[partial]].T[:, :, None]
    residue = ((c[0]*local+c[1])*local+c[2])*local+c[3]
    alocal = target[rows[partial], frames[partial], None]-points-integral.left[ai[partial], None]
    aif = integral._values(ai[partial], alocal)
    result[partial] = half*((aif*residue) @ integral.weights)
    return np.bincount(rows*target.shape[1]+frames, weights=result,
                       minlength=target.size).reshape(target.shape)


@pytest.mark.parametrize('degree,chebyshev', [(1, False), (3, False), (15, True)])
@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_exact_cached_integral_arithmetic(degree: int, chebyshev: bool, scale: float) -> None:
    """Changed amplitudes share geometry, including gaps, delay signs and tails."""
    rng = np.random.default_rng(7301)
    edges = np.r_[0., np.cumsum(rng.uniform(.2, 3., 30))]
    active = rng.random(30) > .2
    left, right = edges[:-1][active], edges[1:][active]
    arterial = rng.normal(size=(degree+1, left.size))*scale
    nodes, weights = np.polynomial.legendre.leggauss((degree+5)//2)
    cache = IntegralGeometryCache()
    integral = PolynomialIntegral(left, right, arterial, nodes, weights,
                                  chebyshev=chebyshev, geometry_cache=cache)
    knots = np.c_[np.zeros(3), np.cumsum(rng.uniform(.01, 12., (3, 4)), axis=1)]
    values = np.exp(-knots/6)
    coefficients = np.stack([np.c_[PchipInterpolator(x, y).c, [0., 0., 0., y[-1]]]
                             for x, y in zip(knots, values, strict=True)])
    target = rng.uniform(-10., edges[-1]+20, (3, 25))
    target[:, :4] = [-5., 0., right[0], 100.]
    for multiplier in (1., .8):
        changed = coefficients*multiplier
        np.testing.assert_array_equal(integral.evaluate(knots, changed, target),
                                       uncached(integral, knots, changed, target))
        assert len(cache.entries) == 1
    original = next(iter(cache.entries.values()))
    assert all(not getattr(original, field.name).flags.writeable for field in fields(original))
    knots[0, 1] = np.nextafter(knots[0, 1], np.inf)
    target[1] += .001
    np.testing.assert_array_equal(integral.evaluate(knots, coefficients, target),
                                   uncached(integral, knots, coefficients, target))
    assert len(cache.entries) == 2
    np.testing.assert_array_equal(integral.evaluate(knots, coefficients, target[:, :0]),
                                   np.empty((3, 0)))


def test_cache_is_shared_bounded_and_does_not_retain_models() -> None:
    """The byte limit applies across models and their tokens cannot alias."""
    cache = IntegralGeometryCache()
    nodes, weights = np.polynomial.legendre.leggauss(4)
    knots, target = np.array([[0., 2., 10.]]), np.arange(20.)[None]
    references = []
    for i in range(24):
        integral = PolynomialIntegral(np.array([0., 1.]), np.array([1., 4.]),
                                      np.full((4, 2), i+1.), nodes, weights, geometry_cache=cache)
        references.append(weakref.ref(integral))
        geometry = integral._geometry(knots, target)
        assert geometry is integral._geometry(knots, target)
    assert len(cache.entries) == 16
    assert all(ref() is None for ref in references[:-1])
    assert cache.nbytes == sum(g.nbytes+len(k[3])+len(k[6]) for k, g in cache.entries.items())
    key = next(iter(cache.entries))
    larger = replace(geometry, moments=np.zeros((4, 150_000)))
    cache.put(key, larger)
    assert cache.entries[key] is larger
    assert cache.nbytes <= 8_000_000
    cache.put((object(), *key[1:]), larger)
    assert len(cache.entries) == 1 and cache.nbytes <= 8_000_000
    huge = replace(geometry, moments=np.zeros((4, 260_000)))
    cache.put((object(), *key[1:]), huge)
    assert len(cache.entries) == 1 and cache.nbytes <= 8_000_000


def test_geometry_keys_include_dtype_and_shape() -> None:
    """Coordinate shapes and precisions are part of the exact cache identity."""
    cache = IntegralGeometryCache()
    nodes, weights = np.polynomial.legendre.leggauss(4)
    integral = PolynomialIntegral(np.array([0.]), np.array([5.]), np.ones((4, 1)),
                                  nodes, weights, geometry_cache=cache)
    knots = np.array([[0., 3., 10.], [0., 3., 10.]])
    target = np.arange(12.).reshape(2, 6)
    a = integral._geometry(knots, target)
    b = integral._geometry(knots, target[:, ::-1][:, ::-1])
    assert a is b
    key = next(iter(cache.entries))
    assert key[1] == knots.shape and key[4] == target.shape
    assert integral._geometry(knots, target[:, :3]) is not a
    assert integral._geometry(knots.astype(np.float32), target.astype(np.float32)) is not a
    assert len(cache.entries) == 3
