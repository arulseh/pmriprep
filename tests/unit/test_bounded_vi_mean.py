"""Coordinate-invariant box constraints for the shared Gaussian mean update."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.models import bayesian_utils
from pmriprep.models.bayesian_utils import _bounded_vi_mean


@pytest.mark.parametrize("exponent", [0, 6, 12, 18])
@pytest.mark.parametrize("upper_active", [False, True])
def test_bounded_mean_is_invariant_to_parameter_units(
    exponent: int, upper_active: bool
) -> None:
    """An active bound reoptimizes correlated coordinates at disparate scales."""
    precision = np.array([[2.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 3.0]])
    mean = np.array([2.0 if upper_active else -1.0, 0.2, 0.3])
    lower, upper = np.array([0.0, -4.0, -4.0]), np.array([1.0, 4.0, 4.0])
    expected = np.array([1.0, 1.0, 0.3]) if upper_active else np.array([0.0, -0.6, 0.3])
    units = 10.0 ** np.array([-exponent, 0.0, exponent])
    actual = _bounded_vi_mean(
        precision / units[:, None] / units[None, :],
        (precision @ mean) / units,
        mean * units,
        lower * units,
        upper * units,
    )
    np.testing.assert_allclose(actual / units, expected, rtol=1e-11, atol=1e-11)


def test_large_offset_and_stiff_nuisance_preserve_bounded_solution() -> None:
    """A tightly specified nuisance must not drive an unrelated baseline to zero."""
    precision = np.diag([1.0, 1e30, 2.0])
    precision[0, 2] = precision[2, 0] = 0.8
    mean = np.array([1000.0, 0.0, 3.0])
    result = _bounded_vi_mean(
        precision, precision @ mean, mean,
        np.array([-np.inf, -np.inf, 0.0]), np.array([np.inf, np.inf, 1.0]),
    )
    np.testing.assert_allclose(result, [1001.6, 0.0, 1.0], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("failed_proposal", [False, True])
def test_feasible_and_nonfinite_unconstrained_proposals(failed_proposal: bool) -> None:
    """Feasible updates retain identity and nonfinite input is recomputed stably."""
    precision = np.diag([1.0, 1e30])
    mean = np.array([0.2, 0.3])
    proposal = np.full(2, np.nan) if failed_proposal else mean
    result = _bounded_vi_mean(precision, precision @ mean, proposal, np.zeros(2), np.ones(2))
    np.testing.assert_allclose(result, mean, rtol=1e-12, atol=1e-12)
    if not failed_proposal:
        assert result is mean


def test_nonfinite_proposal_still_applies_active_bound() -> None:
    """Recovering the unconstrained mean does not bypass its constraints."""
    precision = np.array([[2.0, 0.8], [0.8, 1.0]])
    mean = np.array([2.0, 0.2])
    units = np.array([1e-12, 1e12])
    result = _bounded_vi_mean(
        precision / units[:, None] / units[None, :], (precision @ mean) / units,
        np.full(2, np.nan), np.array([0.0, -4.0]) * units, np.array([1.0, 4.0]) * units,
    )
    np.testing.assert_allclose(result / units, [1.0, 1.0], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("success,answer", [(False, 0.0), (True, np.nan)])
def test_failed_constrained_solve_is_not_silently_clipped(
    monkeypatch: pytest.MonkeyPatch, success: bool, answer: float
) -> None:
    """A failed numerical update remains an error, not a plausible bound value."""
    monkeypatch.setattr(
        bayesian_utils.optimize, "lsq_linear",
        lambda *args, **kwargs: SimpleNamespace(success=success, x=np.array([answer])),
    )
    with pytest.raises(RuntimeError, match="Bound-constrained"):
        _bounded_vi_mean(np.eye(1), np.array([2.0]), np.array([2.0]),
                         np.zeros(1), np.ones(1))
