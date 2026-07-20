"""Tests for inference-specific uncertainty provenance."""

import pytest

from pmriprep.utils.uncertainty_provenance import build_uncertainty_provenance


@pytest.mark.parametrize(
    ("inference", "method", "uncertainty_method", "spatial"),
    [
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


def test_svb_spatial_prior_is_described() -> None:
    """SVB sidecars identify the spatial prior and neighborhood."""
    result = build_uncertainty_provenance("svb", "cbf_cv", modality="DSC")

    assert result["UncertaintyStatistic"] == "coefficient_of_variation"
    assert result["SpatialPriorModel"] == "Penny M-prior"
    assert result["SpatialNeighborhood"] == "3D face-connected"


def test_map_fallback_is_disclosed() -> None:
    """MAP provenance discloses its Hessian-failure fallback semantics."""
    result = build_uncertainty_provenance(
        "map", "ktrans_uncertainty", modality="DCE"
    )

    assert result["UncertaintyStatistic"] == "posterior_standard_deviation"
    assert "prior standard deviations" in result["UncertaintyFallbackBehavior"]

    interval = build_uncertainty_provenance(
        "map", "ktrans_ci95_lower", modality="DCE"
    )
    assert interval["CredibleIntervalConstruction"].endswith(
        "or_full_parameter_bounds_fallback"
    )


def test_dce_interval_is_normal_approximation_bounded_to_parameters() -> None:
    """DCE writer provenance matches its bounded normal approximation."""
    result = build_uncertainty_provenance(
        "vi", "ktrans_ci95_lower", modality="DCE"
    )

    assert result["CredibleIntervalProbability"] == 0.95
    assert result["CredibleIntervalConstruction"].startswith(
        "mean_plus_or_minus_1.96"
    )
    assert result["CredibleIntervalBounding"] == "parameter_bounds"


def test_dsc_mcmc_interval_uses_sample_quantiles() -> None:
    """DSC MCMC intervals are distinguished from Gaussian approximations."""
    result = build_uncertainty_provenance(
        "mcmc", "cbfCI95upper", modality="DSC"
    )

    assert result["CredibleIntervalConstruction"] == "posterior_sample_quantiles"
    assert result["CredibleIntervalBounding"] == "nonnegative_lower_bound"


def test_dce_mcmc_interval_uses_sample_quantiles() -> None:
    """DCE MCMC provenance follows the retained posterior quantiles."""
    result = build_uncertainty_provenance(
        "mcmc", "ktrans_ci95_upper", modality="DCE"
    )

    assert result["CredibleIntervalConstruction"] == "posterior_sample_quantiles"
    assert result["CredibleIntervalBounding"] == "parameter_bounds"
