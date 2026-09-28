"""Gamma derivative identity, scaled stencils and certified auxiliary failure."""

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi.dispersion import DispersionIntegrationError
from pmriprep.models.dsc.signal_cpi.residue import PhysicalSignalCPI, PhysicalSignalEvaluator
from pmriprep.models.dsc.signal_cpi.transport import AssociatedConvolution


def gamma_case(mode: float, rate: float) -> tuple[PhysicalSignalCPI, np.ndarray]:
    """Use signed input and both active kernel bounds without recovery tuning."""
    time = np.arange(36.)*1.5
    aif = np.interp(time, [0., 4., 8., 12., 20., time[-1]], [.2, -.1, 5., 2., -.1, .05])
    model = PhysicalSignalCPI(100*np.exp(-.04*aif), aif, time, 1.5, .04, 5, 'cpi_vtf')
    mean = model.starts()[0]
    mean[:2] = [.1, 100.]
    mean[2:model.delay_index] = [-.2, -.4, -.6, -.8]
    mean[model.delay_index] = -.3
    mean[-3:-1] = np.log([mode, rate])
    mean[-1] = .4
    assert model.valid_mean(mean)
    return model, mean


@pytest.mark.parametrize('mode,rate', [(.05, .05), (.15, 1.5), (2., 2.), (30., 20.), (30., .05)])
def test_rate_identity_matches_independent_scalar_forward(mode: float, rate: float) -> None:
    """Keep narrow-kernel cancellation controlled, including one-sided means."""
    model, mean = gamma_case(mode, rate)
    original = mean.copy()
    evaluator = PhysicalSignalEvaluator(model)
    actual = evaluator.jacobian(mean)
    index = model.slope_index - 1
    step = 1e-4*max(1., abs(mean[index]))
    points = np.tile(mean, (4, 1))
    points[:, index] += np.array([-2., -1., 1., 2.])*step
    # Scalar continuous quadrature is independent of the prepared batched AIF.
    # Positive gamma kernels remain mathematically defined beyond fit bounds.
    curves = np.stack([model.predict(point) for point in points])
    expected = (curves[0]-8*curves[1]+8*curves[2]-curves[3])/(12*step)
    np.testing.assert_allclose(actual[:, index], expected, rtol=3e-5, atol=2e-7)
    np.testing.assert_array_equal(mean, original)
    convolution = evaluator.convolution
    assert isinstance(convolution, AssociatedConvolution)
    assert convolution.stats['maps'] == 6
    assert convolution.fallback_rows == 0


def test_auxiliary_failure_retries_original_stencils(monkeypatch: pytest.MonkeyPatch) -> None:
    """An uncertified adjacent shape cannot turn a supported mean into a failed fit."""
    model, mean = gamma_case(30., 2.)
    reference = PhysicalSignalEvaluator(model)._chain_rule_jacobians(mean[None], None)
    evaluator = PhysicalSignalEvaluator(model)
    original = evaluator.concentration
    failed = []

    def reject_auxiliary(means: np.ndarray) -> np.ndarray:
        if np.any(means[:, model.slope_index-2] > model.upper[model.slope_index-2]):
            failed.append(True)
            raise DispersionIntegrationError('auxiliary accuracy not established')
        return original(means)

    monkeypatch.setattr(evaluator, 'concentration', reject_auxiliary)
    actual = evaluator.jacobians(mean[None])
    assert failed == [True]
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize('error', [ValueError, DispersionIntegrationError])
def test_unresolved_errors_propagate(error: type[Exception], monkeypatch: pytest.MonkeyPatch) -> None:
    """Only an integration failure is retried; neither kind becomes a derivative."""
    model, mean = gamma_case(2., 2.)
    evaluator = PhysicalSignalEvaluator(model)
    calls = []

    def fail(means: np.ndarray) -> np.ndarray:
        calls.append(True)
        raise error('unresolved')

    monkeypatch.setattr(evaluator, 'concentration', fail)
    with pytest.raises(error, match='unresolved'):
        evaluator.jacobians(mean[None])
    assert len(calls) == (2 if error is DispersionIntegrationError else 1)


def test_empty_and_invalid_parameter_rows() -> None:
    """The optimized path retains the evaluator's input contract."""
    model, mean = gamma_case(2., 2.)
    evaluator = PhysicalSignalEvaluator(model)
    assert evaluator.jacobians(np.empty((0, len(mean)))).shape == (0, len(model.time), len(mean))
    for values in (mean, mean[None, :-1], np.full((1, len(mean)), np.nan)):
        with pytest.raises(ValueError, match='finite parameter matrix'):
            evaluator.jacobians(values)
