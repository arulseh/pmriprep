"""Physical residue support, continuous readouts and complete slope derivatives."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PPoly, PchipInterpolator

from pmriprep.models.dsc.signal_cpi.continuous import ContinuousCPIConvolution
from pmriprep.models.dsc.signal_cpi.derivatives import plan_stencil
from pmriprep.models.dsc.signal_cpi.geometry import Model
from pmriprep.models.dsc.signal_cpi.policy import candidate_orders, model_for
from pmriprep.models.dsc.signal_cpi.residue import (
    PhysicalSignalCPI, PhysicalSignalEvaluator, changed_residue, slope_ratio,
    slope_basis_integral,
)
from pmriprep.models.dsc.signal_cpi.signal import MovingLayoutSignalCPI


def make_model(family: Model, dt: float = 1.5) -> PhysicalSignalCPI:
    """Portable measured input, with no parameter-recovery tuning."""
    time = np.arange(24.) * dt
    aif = np.interp(time / dt, np.linspace(0., 23., 9),
                    [.2, 1., 2., 1., .4, .2, .1, .05, .02])
    return PhysicalSignalCPI(100 * np.exp(-.04 * aif), aif, time, dt, .04, 4, family)


def assert_monotone(polynomial: PPoly) -> None:
    """Check derivative extrema analytically, including flat cubic segments."""
    for index, width in enumerate(np.diff(polynomial.x)):
        a, b, c, _ = polynomial.c[:, index]
        positions = [0., float(width)]
        if a != 0 and 0 < -b / (3 * a) < width:
            positions.append(float(-b / (3 * a)))
        x = np.asarray(positions)
        derivative = (3 * a * x + 2 * b) * x + c
        scale = max(abs(3*a*width**2), abs(2*b*width), abs(c), 1.)
        assert np.max(derivative) <= 2e-13 * scale
    assert polynomial(0) == pytest.approx(1., abs=1e-14)
    assert polynomial(polynomial.x[-1]) >= -1e-13
    assert polynomial.derivative()(0) <= 1e-13


@pytest.mark.parametrize("alpha", [0., .4, 1., 2.9, 3.])
def test_entire_cubic_nonincreasing(alpha: float) -> None:
    """Cover irregular spacing, flat segments and very small amplitudes."""
    rng = np.random.default_rng(250925)
    for count in (4, 6, 12):
        for _ in range(30):
            knots = np.r_[0., np.cumsum(np.exp(rng.uniform(-2., 3., count-1)))]
            ratios = rng.choice([0., -1e-12, -.03, -.5, -3., -24.], count-1)
            values = np.r_[1., np.exp(np.cumsum(ratios))]
            polynomial = changed_residue(knots, values, alpha)
            assert_monotone(polynomial)
            assert 0 < polynomial.integrate(0, knots[-1]) <= knots[-1] * (1+1e-13)


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
@pytest.mark.parametrize("dt", [1., 1.5, 3.])
def test_priors_starts_and_coordinate_order(family: Model, dt: float) -> None:
    """The extra slope preserves the declared count, priors and initial curves."""
    model = make_model(family, dt)
    aif = -np.log(model.observed / 100) / model.te
    base = MovingLayoutSignalCPI(model.observed, aif, model.time, dt, model.te,
        model.n_points, family, integration="continuous", aif_interpolation="pchip")
    np.testing.assert_array_equal(model.prior.mean[:-1], base.prior.mean)
    np.testing.assert_array_equal(model.prior.covariance[:-1, :-1], base.prior.covariance)
    assert model.prior.mean[-1] == model.prior.covariance[-1, -1] == 1.
    assert model.lower[-1] == 0. and model.upper[-1] == 3.
    np.testing.assert_array_equal(model.upper[2:model.delay_index], 0.)
    for start, previous in zip(model.starts(), base.starts(), strict=True):
        np.testing.assert_allclose(start[:-1], previous, rtol=1e-12, atol=1e-12)
        assert start[-1] == slope_ratio(model.knot_times(start), model.knot_values(start))
        np.testing.assert_allclose(model.predict(start), base.predict(previous), rtol=1e-11)
        assert_monotone(model.residue(start))
    flat = model.starts()[0]
    flat[2:model.delay_index] = 0.
    assert model.valid_mean(flat)
    assert model.knot_values(flat)[-1] == 1.  # Endpoint amplitude remains free.
    np.testing.assert_allclose(model.residue(flat).integrate(0, model.duration), model.duration)


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_positive_slope_and_rising_amplitudes_rejected(family: Model) -> None:
    model = make_model(family)
    for index in range(2, model.delay_index):
        bad = model.starts()[0]
        bad[index] = np.nextafter(0., 1.)
        assert not model.valid_mean(bad) and not model.derivative_domain(bad)
    for alpha in (-1e-8, 3+1e-8, np.nan, np.inf):
        bad = model.starts()[0]
        bad[-1] = alpha
        assert not model.valid_mean(bad)


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_full_forward_derivatives_and_readout(family: Model) -> None:
    """Compare batched and scalar operators plus independent differences/integrals."""
    model = make_model(family)
    means = np.repeat(np.asarray(model.starts()[:1]), 2, axis=0)
    means[:, 2:model.delay_index] = [-.2, -.5, -.7]
    means[:, -1] = [.4, 2.5]
    means[:, model.delay_index] = [-.2, .4]
    evaluator = PhysicalSignalEvaluator(model)
    actual = evaluator.predictions(means)
    expected = np.stack([model.predict(m) for m in means])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    derivatives = evaluator.jacobians(means)
    for index in range(means.shape[1]):
        step = 1e-5 * max(1., abs(means[0, index]))
        left, right = means[0].copy(), means[0].copy()
        left[index] -= step
        right[index] += step
        expected_column = (model.predict(right) - model.predict(left)) / (2*step)
        np.testing.assert_allclose(derivatives[0, :, index], expected_column,
                                   rtol=3e-4, atol=2e-7)
    for mean in means:
        knots, values = model.knot_times(mean), model.knot_values(mean)
        base = PchipInterpolator(knots, values)
        width, delta = model.slope_change(mean)
        expected_area = base.integrate(0, model.duration) + delta * width**2 / 12
        actual_area = model.residue(mean).integrate(0, model.duration)
        assert actual_area == pytest.approx(expected_area, rel=1e-13)
        numerical = quad(lambda t: float(model.residue(mean)(t)), 0, model.duration,
                          points=knots[1:-1], epsabs=1e-11)[0]
        assert actual_area == pytest.approx(numerical, rel=1e-12)
    np.testing.assert_allclose(evaluator.predictions(means[::-1]), actual[::-1], rtol=1e-12)
    no_cache = PhysicalSignalEvaluator(model, cache_size=0)
    np.testing.assert_allclose(no_cache.predictions(means), actual, rtol=1e-12)
    assert not no_cache.cache
    assert evaluator.predictions(np.empty((0, means.shape[1]))).shape == (0, len(model.time))


def test_plain_exact_reuse_preserves_derivatives_and_owned_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flow/S0/slope changes share unit curves, never distinct shape or timing."""
    def rowwise_basis(operator: ContinuousCPIConvolution, widths: np.ndarray,
                      delays: np.ndarray) -> np.ndarray:
        # Isolate cache identity from BLAS's batch-size-dependent reduction order.
        if not len(widths):
            return slope_basis_integral(operator, widths, delays)
        return np.concatenate([slope_basis_integral(operator, widths[i:i+1], delays[i:i+1])
                               for i in range(len(widths))])

    monkeypatch.setattr("pmriprep.models.dsc.signal_cpi.residue.slope_basis_integral", rowwise_basis)
    model = make_model("cpi")
    evaluator = PhysicalSignalEvaluator(model)
    fresh = PhysicalSignalEvaluator(model, cache_size=0)
    original = evaluator.convolution.evaluate
    calls: list[int] = []

    def record(*args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluator.convolution, "evaluate", record)
    means = np.tile(model.starts()[0], (3, 1))
    means[:, 0] *= [1., 2., .5]
    means[:, 1] *= [1., .5, 2.]
    means[:, -1] = [.1, 1., 2.9]
    first = evaluator.predictions(means)
    np.testing.assert_array_equal(first, fresh.predictions(means))
    assert calls == [1]
    first[:] = -1000
    np.testing.assert_array_equal(evaluator.predictions(means), fresh.predictions(means))
    assert calls == [1]
    np.testing.assert_array_equal(evaluator.jacobians(means), fresh.jacobians(means))
    changed = means.copy()
    changed[0, model.delay_index] += .05
    changed[1, 2] -= .01
    changed[2, model.spacing_indices.start] += .01
    np.testing.assert_array_equal(evaluator.predictions(changed), fresh.predictions(changed))
    bounded = PhysicalSignalEvaluator(model, cache_size=2)
    np.testing.assert_array_equal(bounded.predictions(changed), fresh.predictions(changed))
    assert len(bounded.cache) == 2
    assert all(not value.flags.writeable for value in bounded.cache.values())


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
@pytest.mark.parametrize("alpha", [0., 3.])
def test_physical_boundary_derivatives(family: Model, alpha: float) -> None:
    model = make_model(family)
    mean = model.starts()[0]
    mean[2], mean[-1] = 0., alpha
    stencil = plan_stencil(mean, 2, 1., model.derivative_domain)
    assert not stencil.central and stencil.step < 0
    assert all(model.derivative_domain(p) and p[2] < 0 for p in stencil.points)
    assert np.all(np.isfinite(PhysicalSignalEvaluator(model).jacobian(mean)))


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
@pytest.mark.parametrize("frames", [20, 21, 25, 26, 60])
def test_count_accounts_for_slope(family: Model, frames: int) -> None:
    extra = 3 if family == "cpi_vtf" else 1
    counts = candidate_orders(frames, model=family)
    assert all(2*n + extra < frames for n in counts)
    assert counts[-1] == min(12, (frames - extra - 1) // 2)
    model = make_model(family)
    aif = -np.log(model.observed / 100) / model.te
    assert isinstance(model_for(model.observed, aif, model.time, model.te, 4, family=family),
                      PhysicalSignalCPI)
