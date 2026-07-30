"""Tests for inference-specific uncertainty provenance."""

import pytest

from pmriprep.utils.uncertainty_provenance import build_uncertainty_provenance


@pytest.mark.parametrize(
    ("inference", "method", "uncertainty_method", "spatial"),
    [
        (
            "nlls",
            "nonlinear_least_squares",
            "local_jacobian_covariance",
            False,
        ),
        ("vi", "variational_bayes", "variational_posterior_covariance", False),
        (
            "svb",
            "spatial_variational_bayes",
            "spatial_variational_posterior_covariance",
            True,
        ),
        ("map", "maximum_a_posteriori", "laplace_approximation", False),
        ("mcmc", "markov_chain_monte_carlo", "posterior_samples", False),
    ],
)
def test_inference_metadata(
    inference: str,
    method: str,
    uncertainty_method: str,
    spatial: bool,
) -> None:
    """Each inference engine has an explicit uncertainty interpretation."""
    result = build_uncertainty_provenance(
        inference, "cbf_std", modality="DSC"
    )

    assert result["EstimationMethod"] == inference
    assert result["InferenceMethod"] == method
    assert result["UncertaintyEstimationMethod"] == uncertainty_method
    assert result["SpatialPriorApplied"] is spatial
    assert result["ConditionalOnSelectedModel"] is True
    assert result["IncludesModelSelectionUncertainty"] is False


def test_nlls_statistics_are_frequentist() -> None:
    """NLLS absolute uncertainty is identified as a standard error."""
    standard_error = build_uncertainty_provenance(
        "nlls", "ktrans_uncertainty", modality="DCE"
    )
    assert standard_error["UncertaintyStatistic"] == "parameter_standard_error"

def test_svb_spatial_prior_is_described() -> None:
    """SVB sidecars identify the spatial prior and neighborhood."""
    result = build_uncertainty_provenance("svb", "cbf_std", modality="DSC")

    assert result["UncertaintyStatistic"] == "posterior_standard_deviation"
    assert result["SpatialPriorModel"] == "Penny M-prior"
    assert result["SpatialNeighborhood"] == "3D face-connected"
    assert result["SpatialUpdateScheme"] == "synchronous_batched_backtracking"
    assert result["SpatiallyRegularizedParameters"] == ["cbf"]


def test_map_hessian_failure_is_disclosed() -> None:
    """MAP provenance discloses undefined uncertainty after Hessian failure."""
    result = build_uncertainty_provenance(
        "map", "ktrans_uncertainty", modality="DCE"
    )

    assert result["UncertaintyStatistic"] == "posterior_standard_deviation"
    assert result["UncertaintyFallbackBehavior"].startswith("NaN")
