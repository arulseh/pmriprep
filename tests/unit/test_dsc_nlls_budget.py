"""Tests for bounded DSC parametric NLLS fallback semantics."""

from pmriprep.interfaces.dsc_compute import _parametric_fit_needs_fallback
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


def test_poor_budget_limited_nlls_still_falls_back() -> None:
    """Budget exhaustion never bypasses the fitted-curve quality gate."""
    result = DSCFitResult(
        r_squared=0.2,
        converged=False,
        fitting_method="nlls",
        evaluation_budget_exhausted=True,
    )

    assert _parametric_fit_needs_fallback(result, "nlls")


def test_nonbudget_nlls_failure_falls_back() -> None:
    """Other solver failures remain distinct from controlled termination."""
    result = DSCFitResult(
        r_squared=0.95,
        converged=False,
        fitting_method="nlls",
        evaluation_budget_exhausted=False,
    )

    assert _parametric_fit_needs_fallback(result, "nlls")
