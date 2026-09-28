"""Derivative neighbors share integration geometry, not curves or certification."""

from typing import Any

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi import transport
from pmriprep.models.dsc.signal_cpi.continuous import ContinuousCPIConvolution
from pmriprep.models.dsc.signal_cpi.dispersion import DispersionIntegrationError
from pmriprep.models.dsc.signal_cpi.transport import AssociatedConvolution, DispersedAIF


def artery(scale: float = 1.) -> ContinuousCPIConvolution:
    """Signed irregular samples do not assume smooth or nonnegative measured AIFs."""
    time = np.array([0., .4, 1.2, 3., 7., 15.])
    return ContinuousCPIConvolution(scale*np.array([.2, -.4, 5., 2., -.1, .3]), time,
                                    aif_interpolation='pchip')


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
@pytest.mark.parametrize('horizon', [12., 25., 50.])
def test_seed_geometry_recomputes_values_and_checks(scale: float, horizon: float) -> None:
    """A new kernel independently certifies every reused or extended interval."""
    base = artery(scale)
    previous = DispersedAIF(base, .15, 1.5, 25.)
    edges = np.r_[previous.left, previous.right[-1]]
    seed = edges.copy()
    new = DispersedAIF(base, .1501, 1.5, horizon, refinement_edges=edges)
    cold = DispersedAIF(base, .1501, 1.5, horizon)
    np.testing.assert_array_equal(edges, seed)
    assert new.raw_points > 0
    assert np.all(new.error <= new.interpolation_tolerance)
    assert new.right[-1] == horizon
    assert np.all(np.isin(seed[seed < horizon], new.left))
    # Changing the caller's seed must not mutate the prepared representation.
    edges[:] = 0
    np.testing.assert_array_equal(new.refinement_edges, seed)
    knots, values = np.array([0., .8, 4., 15.]), np.array([1., .8, .3, .1])
    target = np.r_[-3., 0., np.linspace(.003, horizon, 83)]
    expected = cold.convolve(knots, values, target)/scale
    normalization = np.max(np.abs(expected))
    np.testing.assert_allclose(new.convolve(knots, values, target)/scale/normalization,
                               expected/normalization, rtol=0, atol=1e-8)
    assert not np.array_equal(new._raw(target[2:]), previous._raw(target[2:]))


@pytest.mark.parametrize('edges', [np.array([0., 25.]), np.linspace(0., 25., 20002)])
def test_coarse_or_oversized_seed_retains_independent_budget(edges: np.ndarray) -> None:
    """A geometry hint cannot bypass refinement or exhaust its segment allowance."""
    base = artery()
    cold = DispersedAIF(base, .15, 1.5, 25.)
    seeded = DispersedAIF(base, .15, 1.5, 25., refinement_edges=edges)
    for name in ('left', 'right', 'coefficients', 'error'):
        np.testing.assert_array_equal(getattr(seeded, name), getattr(cold, name))


@pytest.mark.parametrize('edges', [np.ones((2, 2)), np.array([0., np.nan]),
                                 np.array([-1., 2.]), np.array([0., 0.]),
                                 np.array([2., 1.])])
def test_invalid_seed_raises(edges: np.ndarray) -> None:
    """Only ordered finite interval geometry may seed preparation."""
    with pytest.raises(ValueError, match='refinement edges'):
        DispersedAIF(artery(), .15, 1.5, 25., refinement_edges=edges)


def test_only_single_coordinate_stencil_neighbors_seed() -> None:
    """Distant parameters and zero modes do not accidentally reuse nearby values."""
    convolution = AssociatedConvolution(artery())
    key = (.15, 1.5)
    previous = DispersedAIF(convolution.base, *key, 25.)
    convolution.cache[key] = previous
    expected = np.r_[previous.left, previous.right[-1]]
    for neighbor in ((.1501, 1.5), (.15, 1.501)):
        np.testing.assert_array_equal(convolution._refinement_edges(neighbor), expected)
    for different in (key, (.15, 2.), (.16, 1.5), (.1501, 1.501), (0., 1.5)):
        assert convolution._refinement_edges(different) is None
    assert len(convolution.cache) == 1


def test_uncertified_seed_retries_cold_before_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warm-start failure cannot make a formerly certified input fail."""
    base = artery()
    convolution = AssociatedConvolution(base)
    convolution.cache[(.15, 1.5)] = DispersedAIF(base, .15, 1.5, 25.)
    knots = np.array([[0., .8, 4., 15.]])
    values = np.array([[1., .8, .3, .1]])
    target = base.time[None]
    gamma = np.array([[.1501, 1.5]])
    expected = DispersedAIF(base, *gamma[0], 25.).convolve(knots[0], values[0], target[0])
    calls = []
    original = transport.DispersedAIF

    def constructor(*args: Any, **kwargs: Any) -> DispersedAIF:
        warm = kwargs.get('refinement_edges') is not None
        calls.append(warm)
        if warm:
            raise DispersionIntegrationError('uncertified seed')
        return original(*args, **kwargs)

    monkeypatch.setattr(transport, 'DispersedAIF', constructor)
    actual = convolution._gamma(knots, values, target, gamma)
    np.testing.assert_allclose(actual[0], expected, rtol=3e-12, atol=3e-12)
    assert calls == [True, False]
    assert convolution.stats['seed_retries'] == 1
    assert convolution.stats['seeded_maps'] == 0
    assert convolution.fallback_rows == 0 and not convolution.failed
