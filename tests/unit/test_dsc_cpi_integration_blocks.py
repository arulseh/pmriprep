"""Interval batching bounds temporaries without changing adaptive decisions."""

from typing import Any

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi import transport
from pmriprep.models.dsc.signal_cpi.continuous import ContinuousCPIConvolution
from pmriprep.models.dsc.signal_cpi.dispersion import DispersionIntegrationError
from pmriprep.models.dsc.signal_cpi.transport import DispersedAIF, GammaEndpointCache


def artery() -> ContinuousCPIConvolution:
    """Signed irregular AIF with early support and a nonzero final sample."""
    return ContinuousCPIConvolution(
        np.array([.2, -.4, 5., 2., -.1, .3]), np.array([0., .4, 1.2, 3., 7., 15.]),
        aif_interpolation='pchip',
    )


@pytest.mark.parametrize('mode,rate', [(0., 1.5), (.05, .05), (.15, 1.5), (2., 2.), (30., 20.)])
@pytest.mark.parametrize('cached', [False, True])
def test_blocked_preparation_matches_unsplit_values_and_checks(
    mode: float, rate: float, cached: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both block boundaries and exact endpoint caches preserve each integral."""
    base = artery()
    monkeypatch.setattr(transport, '_INTERVAL_BLOCK_SIZE', 1_000_000)
    reference = DispersedAIF(base, mode, rate, 25.,
                             endpoint_cache=GammaEndpointCache() if cached else None)
    target = np.r_[-3., 0., np.linspace(.003, 25., 43)]
    knots, values = np.array([0., .8, 4., 15.]), np.array([1., .8, .3, .1])
    expected = reference.convolve(knots, values, target)
    monkeypatch.setattr(transport, '_INTERVAL_BLOCK_SIZE', 31)
    blocked = DispersedAIF(base, mode, rate, 25.,
                           endpoint_cache=GammaEndpointCache() if cached else None)
    for name in ('left', 'right', 'coefficients', 'error'):
        np.testing.assert_array_equal(getattr(blocked, name), getattr(reference, name))
    assert blocked.raw_points == reference.raw_points
    assert blocked.analytic_intervals == reference.analytic_intervals
    assert blocked.quadrature_intervals == reference.quadrature_intervals
    np.testing.assert_array_equal(blocked.convolve(knots, values, target), expected)


def test_moment_and_node_work_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-block cap applies before allocation of each large work array."""
    monkeypatch.setattr(transport, '_INTERVAL_BLOCK_SIZE', 17)
    moment_rows, node_rows = [], []
    original_cdf = transport._gamma_interval_cdfs
    original_nodes = DispersedAIF._quadrature_values

    def cdf(shape: float, lower: np.ndarray, upper: np.ndarray) -> Any:
        moment_rows.append(len(lower))
        return original_cdf(shape, lower, upper)

    def nodes(self: DispersedAIF, order: int, target: np.ndarray, rows: np.ndarray,
              segments: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        node_rows.append(len(rows))
        return original_nodes(self, order, target, rows, segments, left, right)

    monkeypatch.setattr(transport, '_gamma_interval_cdfs', cdf)
    monkeypatch.setattr(DispersedAIF, '_quadrature_values', nodes)
    kernel = DispersedAIF(artery(), .15, 1.5, 25.)
    # Exercise quadrature independently of whether these moments need fallback.
    kernel._quadrature(np.array([.2]), np.zeros(100, dtype=int), np.zeros(100, dtype=int),
                       np.zeros(100), np.full(100, .1), np.array([4e-9]))
    assert len(moment_rows) > 1 and max(moment_rows) <= 17
    assert len(node_rows) > 1 and max(node_rows) <= 17


def test_error_checks_remain_per_target_not_per_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small errors in separate blocks must not evade the shared target budget."""
    prepared = DispersedAIF(artery(), .15, 1.5, 25.)
    monkeypatch.setattr(transport, '_INTERVAL_BLOCK_SIZE', 2)
    calls = []

    def artificial(self: DispersedAIF, order: int, target: np.ndarray, rows: np.ndarray,
                    segments: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        calls.append(order)
        if len(calls) > 4:
            return np.full(len(rows), np.nan)
        return np.full(len(rows), .1 if order == 32 else 0.)

    monkeypatch.setattr(DispersedAIF, '_quadrature_values', artificial)
    # Each two-interval block's estimated error is .8; the target total is 1.2.
    # Neither block may independently accept the shared budget of 1.
    with pytest.raises(DispersionIntegrationError, match='accuracy'):
        prepared._quadrature(np.array([1.]), np.zeros(3, dtype=int), np.zeros(3, dtype=int),
                             np.array([0., .1, .2]), np.array([.1, .2, .3]), np.array([1.]))
    assert len(calls) > 4


def test_empty_active_intervals_remain_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty moment buffers neither reuse stale cache data nor invoke quadrature."""
    monkeypatch.setattr(transport, '_INTERVAL_BLOCK_SIZE', 3)
    prepared = DispersedAIF(artery(), .15, 1.5, 25., endpoint_cache=GammaEndpointCache())
    np.testing.assert_array_equal(prepared._raw(np.array([-3., 0.])), [0., 0.])
    assert prepared._raw(np.array([])).shape == (0,)
