"""Independent continuous-dispersion integrals, support and limit checks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import gammainc
from scipy.stats import gamma

from pmriprep.models.dsc.signal_cpi import dispersion
from pmriprep.models.dsc.signal_cpi.dispersion import ContinuousDispersedCPI, DispersionIntegrationError


def positive_gamma_moment(
    target: np.ndarray, power: int, peak: float, rate: float
) -> np.ndarray:
    """Evaluate E[(t-U)_+**power] through independent incomplete moments."""
    from math import comb

    x = np.maximum(target, 0)
    shape = 1 + rate * peak
    result = np.zeros_like(x)
    factor = 1.0
    for index in range(power + 1):
        if index:
            factor *= (shape + index - 1) / rate
        result += (
            (-1) ** index
            * comb(power, index)
            * x ** (power - index)
            * factor
            * gammainc(shape + index, rate * x)
        )
    return result


@pytest.mark.parametrize("peak,rate", [(0, 0.2), (0.05, 0.05), (2, 0.7), (30, 20)])
@pytest.mark.parametrize("delay", [-3.25, 0.0, 2.75])
@pytest.mark.parametrize("dt", [0.5, 1.0, 3.0])
def test_constant_aif_and_residue_match_closed_form(
    peak: float, rate: float, delay: float, dt: float
) -> None:
    """Check the origin, kernel envelope, signed delay and beyond-scan support."""
    time = np.arange(0.0, 42.0 + dt / 2, dt)
    operator = ContinuousDispersedCPI(np.ones_like(time), time)
    actual = operator.evaluate(np.array([0, 42.0]), np.ones(2), 1.7, delay, peak, rate)
    target = time - delay
    expected = 1.7 * (
        positive_gamma_moment(target, 1, peak, rate)
        - positive_gamma_moment(target - time[-1], 1, peak, rate)
    )
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(actual[target <= 0], 0)


def physical_time_reference(
    aif: np.ndarray,
    time: np.ndarray,
    knots: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    peak: float,
    rate: float,
    interpolation: str,
) -> np.ndarray:
    """Convolve gamma with residue first using physical-time adaptive integrals."""
    residue = PchipInterpolator(knots, values)
    artery = PchipInterpolator(time, aif)
    density = gamma(1 + rate * peak, scale=1 / rate)

    def dispersed_residue(elapsed: float) -> float:
        if elapsed <= 0:
            return 0.0
        cuts = elapsed - knots
        return float(
            quad(
                lambda u: float(residue(min(elapsed - u, knots[-1]))) * density.pdf(u),
                0,
                elapsed,
                points=cuts[(cuts > 0) & (cuts < elapsed)],
                epsabs=2e-11,
                epsrel=2e-10,
                limit=100,
            )[0]
        )

    def concentration(target: float) -> float:
        end = min(target, time[-1])
        if end <= 0:
            return 0.0
        return float(
            quad(
                lambda a: (
                    (
                        float(artery(a))
                        if interpolation == "pchip"
                        else float(np.interp(a, time, aif))
                    )
                    * dispersed_residue(target - a)
                ),
                0,
                end,
                points=time[(time > 0) & (time < end)],
                epsabs=2e-10,
                epsrel=2e-9,
                limit=100,
            )[0]
        )

    return np.array([concentration(t) for t in targets])


@pytest.mark.parametrize("interpolation", ["linear", "pchip"])
@pytest.mark.parametrize("delay", [-1.3, 0.0, 2.2])
def test_matches_independent_reordered_physical_time_integral(
    interpolation: str, delay: float
) -> None:
    """Neither reference convolution uses the implementation's probability grid."""
    time = np.array([0, 1, 3, 5, 8.0])
    aif = np.array([0.2, 1, 2, 0.4, 0.1])
    knots, values = np.array([0, 0.8, 2.4, 5.0]), np.array([1, 0.8, 0.35, 0.1])
    op = ContinuousDispersedCPI(aif, time, aif_interpolation=interpolation)
    actual = op.evaluate(knots, values, 0.03, delay, 1.2, 0.7)
    indices = np.array([0, 2, 4])
    expected = 0.03 * physical_time_reference(
        aif, time, knots, values, time[indices] - delay, 1.2, 0.7, interpolation
    )
    np.testing.assert_allclose(actual[indices], expected, rtol=2e-9, atol=2e-11)


def test_truncated_gamma_mass_is_not_renormalized() -> None:
    """A long transport tail cannot be folded back into a short acquisition."""
    time = np.linspace(0, 3, 10)
    op = ContinuousDispersedCPI(np.ones_like(time), time)
    actual = op.evaluate(np.array([0, 3.0]), np.ones(2), 1, 0, 30, 0.05)
    expected = positive_gamma_moment(time, 1, 30, 0.05)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-14)
    captured = gamma.cdf(3, 2.5, scale=20)
    assert captured < 0.01
    assert actual[-1] < 0.01 * expected[-1] / captured


def test_complete_transport_preserves_bolus_area() -> None:
    """Unit residue eventually accumulates the full AIF area after transport."""
    time = np.array([0, 1, 2, 3, 10, 50, 2000.0])
    aif = np.array([0, 1, 1, 0, 0, 0, 0.0])
    actual = ContinuousDispersedCPI(aif, time).evaluate(
        np.array([0, 2000.0]), np.ones(2), 0.03, 0, 30, 0.05
    )
    area = PchipInterpolator(time, aif).integrate(0, time[-1])
    assert actual[-1] == pytest.approx(0.03 * area, rel=2e-12)
    assert actual[4] < actual[-1] / 10


def test_zero_flow_aif_and_absent_causal_support() -> None:
    """Empty inputs and not-yet-arrived tissue have exactly zero concentration."""
    time, knots = np.arange(10.0), np.array([0, 9.0])
    values = np.array([1, 0.2])
    empty = ContinuousDispersedCPI(np.zeros_like(time), time)
    op = ContinuousDispersedCPI(np.ones_like(time), time)
    np.testing.assert_array_equal(empty.evaluate(knots, values, 1, 0, 2, 0.7), 0)
    np.testing.assert_array_equal(op.evaluate(knots, values, 0, 0, 2, 0.7), 0)
    np.testing.assert_array_equal(op.evaluate(knots, values, 1, 12, 2, 0.7), 0)


def test_zero_mode_alone_is_not_zero_dispersion() -> None:
    """An exponential kernel remains dispersed until its rate tends to infinity."""
    time = np.arange(0.0, 20.0)
    knots, values = np.array([0, 19.0]), np.ones(2)
    op = ContinuousDispersedCPI(np.ones_like(time), time)
    actual = op.evaluate(knots, values, 1, 0, 0, 0.5)
    expected = time + 2 * np.expm1(-time / 2)
    np.testing.assert_allclose(actual, expected, atol=2e-11)
    assert actual[1] < time[1] / 2


@pytest.mark.parametrize("delay", [-1.3, 0.0, 2.2])
def test_subframe_kernel_tends_to_plain_cpi_without_a_frame_offset(
    delay: float,
) -> None:
    """Increasing rate resolves a delta limit even far below the sample interval."""
    time = np.arange(0.0, 20.0, 3.0)
    aif = 1 + time * np.exp(-time / 3)
    knots, values = np.array([0, 1.2, 5, 18.0]), np.array([1, 0.8, 0.2, 0.1])
    op = ContinuousDispersedCPI(aif, time)
    plain = op.base.evaluate(knots, values, 0.02, delay)
    errors = []
    for rate in (100.0, 10000.0, 1000000.0):
        actual = op.evaluate(knots, values, 0.02, delay, 0, rate)
        errors.append(np.max(abs(actual - plain)))
    assert errors[1] < errors[0] / 50
    assert errors[2] < errors[1] / 50
    assert errors[2] < 1e-7


def test_units_and_amplitude_scaling() -> None:
    """Mode scales with time and sharpness inversely, with no acquisition constant."""
    time = np.array([0, 0.5, 2, 5, 8.0])
    aif = np.array([1, 3, 2, 0.3, 0.0])
    knots, values = np.array([0, 1, 8.0]), np.array([1, 0.5, 0.05])
    actual = ContinuousDispersedCPI(aif, time).evaluate(knots, values, 0.2, -1, 2, 0.7)
    scaled = ContinuousDispersedCPI(3 * aif, 60 * time).evaluate(
        60 * knots, values, 0.2 / 180, -60, 120, 0.7 / 60
    )
    np.testing.assert_allclose(actual, scaled, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("peak,rate", [(-1, 2), (1, 0), (1, np.inf), (np.nan, 1)])
def test_invalid_kernel_parameters_raise(peak: float, rate: float) -> None:
    """Do not silently clip an invalid transport distribution."""
    time = np.arange(5.0)
    with pytest.raises(ValueError):
        ContinuousDispersedCPI(np.ones_like(time), time).evaluate(
            np.array([0, 4.0]), np.ones(2), 1, 0, peak, rate
        )


def test_failed_quadrature_does_not_become_a_valid_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmet integration criterion must be visible to the caller."""
    monkeypatch.setattr(
        dispersion,
        "quad_vec",
        lambda *args, **kwargs: (
            np.ones(5),
            0.1,
            SimpleNamespace(success=False, message="forced failure"),
        ),
    )
    time = np.arange(5.0)
    with pytest.raises(DispersionIntegrationError, match="forced failure"):
        ContinuousDispersedCPI(np.ones_like(time), time).evaluate(
            np.array([0, 4.0]), np.ones(2), 1, 0, 2, 0.7
        )
