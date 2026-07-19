"""Validation tests for the DCE pharmacokinetic-model time axis."""

from __future__ import annotations

import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import _validate_dce_time_axis
from pmriprep.models.dce.batched_models import _validate_uniform_time_points


def test_uniform_matching_axis_is_accepted() -> None:
    """A finite, uniform, length-matched axis is returned as float64."""
    result = _validate_dce_time_axis([0.0, 0.1, 0.2], 3, 3)
    np.testing.assert_allclose(result, [0.0, 0.1, 0.2])
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    ("time_points", "message"),
    [
        ([[0.0, 0.1], [0.2, 0.3]], "one-dimensional"),
        ([0.0], "at least two"),
        ([0.0, np.nan, 0.2], "finite"),
        ([0.0, 0.1, np.inf], "finite"),
        ([0.0, 0.1, 0.1], "strictly increasing"),
        ([0.0, 0.2, 0.1], "strictly increasing"),
        ([0.0, 0.1, 0.25], "Nonuniform"),
    ],
)
def test_invalid_axis_is_rejected(
    time_points: object,
    message: str,
) -> None:
    """Unsupported time-axis shapes and sampling fail clearly."""
    with pytest.raises(ValueError, match=message):
        _validate_dce_time_axis(time_points, 3, 3)


@pytest.mark.parametrize(("n_frames", "n_aif"), [(2, 3), (3, 2), (4, 4)])
def test_length_mismatch_is_rejected(n_frames: int, n_aif: int) -> None:
    """The time vector must match both DCE frames and AIF samples."""
    with pytest.raises(ValueError, match="lengths must match"):
        _validate_dce_time_axis([0.0, 0.1, 0.2], n_frames, n_aif)


def test_batched_models_reject_nonuniform_axis() -> None:
    """Direct batched-model callers cannot bypass the uniformity requirement."""
    with pytest.raises(ValueError, match="Nonuniform"):
        _validate_uniform_time_points(np.array([0.0, 0.1, 0.25]))


def test_batched_models_reject_late_nonincreasing_interval() -> None:
    """Every interval, rather than only the first, must be positive."""
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_uniform_time_points(np.array([0.0, 0.1, 0.05]))
