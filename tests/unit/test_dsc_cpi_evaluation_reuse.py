"""Exact sparse geometry and bounded arithmetic preserve CPI evaluation."""

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from pmriprep.models.dsc.batched_fitting import _assemble_JtWJ
from pmriprep.models.dsc.signal_cpi.continuous import PolynomialIntegral, polynomial_overlaps
from pmriprep.models.dsc.signal_cpi.residue import initial_slopes


@pytest.mark.parametrize("points", [2, 5, 12])
@pytest.mark.parametrize("frames", [0, 5, 120])
def test_sparse_overlap_matches_dense_order_and_limits(points: int, frames: int) -> None:
    """Keep gaps, touching edges, signed/out-of-window times and constant tails."""
    rng = np.random.default_rng(2711)
    edges = np.r_[0., np.cumsum(rng.uniform(.2, 3., 50))]
    active = rng.random(50) > .2
    left, right = edges[:-1][active], edges[1:][active]
    knots = np.c_[np.zeros(8), np.cumsum(rng.uniform(.1, 20., (8, points-1)), axis=1)]
    target = rng.uniform(-10., edges[-1]+50, (8, frames))
    if frames >= 5:
        target[:, :5] = np.array([-10., 0., left[0], right[0], 500.])
    lo = np.maximum(target[:, :, None, None]-right[None, None, :, None], knots[:, None, None])
    ends = np.c_[knots[:, 1:], np.full(8, np.inf)]
    hi = np.minimum(target[:, :, None, None]-left[None, None, :, None], ends[:, None, None])
    indices = np.nonzero(hi > lo)
    reference = (*indices, lo[indices], hi[indices])
    actual = polynomial_overlaps(left, right, knots, target)
    for expected, result in zip(reference, actual, strict=True):
        np.testing.assert_array_equal(result, expected)
    empty = polynomial_overlaps(left[:0], right[:0], knots, target)
    assert all(x.size == 0 for x in empty)


@pytest.mark.parametrize("points", [2, 5, 12])
def test_origin_slope_matches_scipy(points: int) -> None:
    """Both sign-changing and monotone rows follow the same endpoint rule."""
    rng = np.random.default_rng(2711)
    knots = np.c_[np.zeros(100), np.cumsum(np.exp(rng.uniform(-3., 3., (100, points-1))), axis=1)]
    values = rng.normal(size=knots.shape)
    values[0] = 1.
    values[1, :2] = 1.
    values[2:] = np.exp(-knots[2:]/5)
    expected = np.array([PchipInterpolator(x, y).c[2, 0] for x, y in zip(knots, values, strict=True)])
    np.testing.assert_array_equal(initial_slopes(knots, values), expected)


@pytest.mark.parametrize("degree,chebyshev", [(1, False), (3, False), (15, True)])
def test_whole_interval_moments_match_unsplit_gaussian_rule(degree: int, chebyshev: bool) -> None:
    """Centered moments preserve signed input, local clipping, tails and scales."""
    rng = np.random.default_rng(2711)
    edges = np.r_[0., np.cumsum(rng.uniform(.01, 3., 50))]
    left, right = edges[:-1], edges[1:]
    arterial = rng.normal(size=(degree+1, len(left)))
    nodes, weights = np.polynomial.legendre.leggauss((degree+5)//2)
    integral = PolynomialIntegral(left, right, arterial, nodes, weights, chebyshev=chebyshev)
    knots = np.c_[np.zeros(6), np.cumsum(rng.uniform(.1, 20., (6, 4)), axis=1)]
    values = np.exp(-knots/rng.uniform(2., 20., (6, 1)))
    values *= np.geomspace(1e-12, 1e12, 6)[:, None]
    coefficients = np.stack([np.c_[PchipInterpolator(x, y).c, [0., 0., 0., y[-1]]]
                             for x, y in zip(knots, values, strict=True)])
    targets = rng.uniform(-10., edges[-1]+100, (6, 30))
    rows, frames, ai, ri, lo, hi = polynomial_overlaps(left, right, knots, targets)
    half = (hi-lo)/2
    points = (hi+lo)[:, None]/2+half[:, None]*nodes
    local = points-knots[rows, ri, None]
    c = coefficients[rows, :, ri].T[:, :, None]
    residue = ((c[0]*local+c[1])*local+c[2])*local+c[3]
    alocal = targets[rows, frames, None]-points-left[ai, None]
    if chebyshev:
        aif = np.polynomial.chebyshev.chebval(2*alocal/(right-left)[ai, None]-1,
                                             arterial[:, ai, None], tensor=False)
    else:
        aif = np.polyval(arterial[:, ai, None], alocal)
    reference = np.bincount(rows*targets.shape[1]+frames,
        weights=half*((aif*residue) @ weights), minlength=targets.size).reshape(targets.shape)
    actual = integral.evaluate(knots, coefficients, targets)
    scale = np.max(abs(reference), axis=1)[:, None]
    np.testing.assert_allclose(actual/scale, reference/scale, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("parameters", [1, 4, 16, 131])
@pytest.mark.parametrize("voxels", [0, 1, 3, 25])
@pytest.mark.parametrize("fortran", [True, False])
def test_blocked_gram_preserves_columnwise_arithmetic(parameters: int, voxels: int, fortran: bool) -> None:
    """Keep weighted precision/residual sums and symmetry, including rank defects."""
    rng = np.random.default_rng(2711)
    columns = list(rng.normal(size=(parameters, voxels, 21)))
    if fortran:
        columns = [np.asfortranarray(column) for column in columns]
    columns[-1] *= 0
    residual = rng.normal(size=(voxels, 21))
    weights = rng.uniform(0., 1., 21)
    weights[::3] = 0.
    expected = np.zeros((voxels, parameters, parameters))
    projected = np.zeros((voxels, parameters))
    for j in range(parameters):
        weighted = weights[None]*columns[j]
        projected[:, j] = np.sum(weighted*residual, axis=1)
        for k in range(j, parameters):
            expected[:, j, k] = expected[:, k, j] = np.sum(weighted*columns[k], axis=1)
    actual, rhs = _assemble_JtWJ(columns, residual, weights)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(rhs, projected)
