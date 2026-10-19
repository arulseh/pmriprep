"""Expected accuracy limitations for specific deterministic DSC fixtures."""

import numpy as np
import numpy.typing as npt
import pytest


class AccuracyTargetError(AssertionError):
    """Finite estimates exceed a fixture's stated accuracy bounds."""


def assert_accuracy(
    errors: npt.ArrayLike,
    upper: npt.ArrayLike,
    *,
    lower: float = -np.inf,
    label: str,
) -> None:
    """Check all finite errors before classifying an accuracy limitation.

    Parameters
    ----------
    errors : array-like
        Scalar or per-metric errors to compare with the bounds.
    upper : array-like
        Exclusive upper bounds, broadcastable to the errors.
    lower : float, optional
        Exclusive lower bound.
    label : str
        Metric or comparison description for failure messages.

    Raises
    ------
    AssertionError
        If the error array is empty or contains nonfinite values.
    AccuracyTargetError
        If a finite error falls outside its bounds.
    """
    values = np.asarray(errors, dtype=float)
    assert values.size and np.all(np.isfinite(values)), f"{label}: invalid errors {values}"
    if not np.all((values > lower) & (values < upper)):
        raise AccuracyTargetError(f"{label}: errors {values} outside ({lower}, {upper})")


P2_NO_CORNER = pytest.mark.xfail(
    strict=True,
    raises=AccuracyTargetError,
    reason=(
        "This noiseless fixture has no interior curvature maximum. P2 selects "
        "its upper-endpoint default (10*sigma_max), substantially underestimating "
        "CBF and residue area."
    ),
)

P2_UPPER_CORNER = pytest.mark.xfail(
    strict=True,
    raises=AccuracyTargetError,
    reason=(
        "This noiseless dispersed fixture has two curvature maxima. P2 selects "
        "the maximum nearest the upper lambda bound, underestimating CBF by "
        "about 21% and exceeding the 16% recovery target."
    ),
)

OSVD_RESIDUE_MTT = pytest.mark.xfail(
    strict=True,
    raises=AccuracyTargetError,
    reason=(
        "For this noiseless fixture at 1.5-s sampling, the 1% truncation floor "
        "and block-circulant discretization underestimate the residue peak. "
        "Full-period area divided by that peak exceeds the MTT recovery target."
    ),
)
