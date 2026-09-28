"""Tests for bounded DSC parametric NLLS fallback semantics."""

import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import (
    _parametric_fit_needs_fallback,
    _parametric_outputs_are_finite,
)
from pmriprep.models.dsc.base import DSCFitResult


def test_good_budget_limited_nlls_is_quality_gated() -> None:
    """A stable endpoint is retained when its fitted curve passes R² QC."""
    result = DSCFitResult(
        r_squared=0.95,
        converged=False,
        fitting_method="nlls",
        evaluation_budget_exhausted=True,
    )

    assert not _parametric_fit_needs_fallback(result, "nlls")


def test_low_positive_r2_budget_limited_nlls_is_retained() -> None:
    """A low positive R² is not treated as a reliability threshold."""
    result = DSCFitResult(
        r_squared=0.2,
        converged=False,
        fitting_method="nlls",
        evaluation_budget_exhausted=True,
    )

    assert not _parametric_fit_needs_fallback(result, "nlls")


def test_negative_r2_budget_limited_nlls_falls_back() -> None:
    """A fit worse than the temporal-mean predictor is replaced."""
    result = DSCFitResult(
        r_squared=-0.01,
        converged=False,
        fitting_method="nlls",
        evaluation_budget_exhausted=True,
    )

    assert _parametric_fit_needs_fallback(result, "nlls")


def test_zero_r2_is_retained() -> None:
    """The minimum R² boundary is inclusive."""
    result = DSCFitResult(r_squared=0.0, converged=True, fitting_method="vi")

    assert not _parametric_fit_needs_fallback(result, "vi")


@pytest.mark.parametrize("r_squared", [float("nan"), float("inf")])
def test_nonfinite_r2_falls_back(r_squared: float) -> None:
    """Undefined fit quality cannot silently retain a parametric estimate."""
    result = DSCFitResult(
        r_squared=r_squared,
        converged=True,
        fitting_method="vi",
    )

    assert _parametric_fit_needs_fallback(result, "vi")


def test_nonfinite_parameter_or_prediction_falls_back() -> None:
    """Finite R² cannot rescue a non-finite quantitative result."""
    bad_parameter = DSCFitResult(
        parameters={"cbf": float("nan")},
        r_squared=0.9,
        converged=True,
        fitting_method="vi",
    )
    bad_prediction = DSCFitResult(
        parameters={"cbf": 0.1},
        r_squared=0.9,
        converged=True,
        fitting_method="vi",
        fitted_tissue_curve=np.array([0.0, np.inf]),
    )

    assert _parametric_fit_needs_fallback(bad_parameter, "vi")
    assert _parametric_fit_needs_fallback(bad_prediction, "vi")


@pytest.mark.parametrize("field", ["cbf", "cbv", "mtt", "tmax", "alpha", "beta", "cth"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_derived_output_falls_back(field: str, invalid: float) -> None:
    """Every required vascular output is checked, including fitted timing."""
    output = dict(cbf=0.1, cbv=0.4, mtt=4.0, tmax=0.0, alpha=2.0, beta=2.0, cth=2.8)
    assert _parametric_outputs_are_finite(output)
    output[field] = invalid
    assert not _parametric_outputs_are_finite(output)


def test_nonbudget_nlls_failure_falls_back() -> None:
    """Other solver failures remain distinct from controlled termination."""
    result = DSCFitResult(
        r_squared=0.95,
        converged=False,
        fitting_method="nlls",
        evaluation_budget_exhausted=False,
    )

    assert _parametric_fit_needs_fallback(result, "nlls")
