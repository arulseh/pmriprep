"""Provenance descriptions for Bayesian uncertainty derivatives."""

from __future__ import annotations


_INFERENCE_METADATA: dict[str, dict[str, object]] = {
    "vi": {
        "InferenceMethod": "variational_bayes",
        "UncertaintyEstimationMethod": "variational_posterior_covariance",
        "PosteriorApproximation": "Gaussian",
        "SpatialPriorApplied": False,
    },
    "svb": {
        "InferenceMethod": "spatial_variational_bayes",
        "UncertaintyEstimationMethod": "spatial_variational_posterior_covariance",
        "PosteriorApproximation": "Gaussian",
        "SpatialPriorApplied": True,
        "SpatialPriorModel": "Penny M-prior",
        "SpatialNeighborhood": "3D face-connected",
    },
    "map": {
        "InferenceMethod": "maximum_a_posteriori",
        "UncertaintyEstimationMethod": "laplace_approximation",
        "PosteriorApproximation": "local_Gaussian",
        "SpatialPriorApplied": False,
        "UncertaintyFallbackBehavior": (
            "prior standard deviations and full parameter bounds when the "
            "local Hessian covariance is unavailable"
        ),
    },
    "mcmc": {
        "InferenceMethod": "markov_chain_monte_carlo",
        "UncertaintyEstimationMethod": "posterior_samples",
        "PosteriorApproximation": "posterior_samples",
        "SpatialPriorApplied": False,
    },
}


def build_uncertainty_provenance(
    inference: str,
    map_key: str,
    modality: str,
) -> dict[str, object]:
    """Describe how one Bayesian uncertainty derivative was constructed.

    Parameters
    ----------
    inference
        Inference-engine identifier from the processing configuration.
    map_key
        Internal derivative key identifying standard deviation, coefficient
        of variation, or a 95% credible-interval bound.
    modality
        Perfusion modality (``"DCE"`` or ``"DSC"``), used to describe the
        modality-specific interval bounding performed by the map writer.

    Returns
    -------
    dict
        JSON-serializable provenance fields for the derivative sidecar.
    """
    normalized_inference = str(inference or "unknown").lower()
    metadata = dict(
        _INFERENCE_METADATA.get(
            normalized_inference,
            {
                "InferenceMethod": normalized_inference,
                "UncertaintyEstimationMethod": "unknown",
                "SpatialPriorApplied": False,
            },
        )
    )
    metadata["EstimationMethod"] = normalized_inference

    key_lower = map_key.lower()
    if "cv" in key_lower:
        statistic = "coefficient_of_variation"
        legacy_method = "coefficient_of_variation"
    elif "ci95" in key_lower:
        statistic = "credible_interval_bound"
        legacy_method = "95pct_credible_interval"
    else:
        statistic = "posterior_standard_deviation"
        legacy_method = "std"
    metadata["UncertaintyStatistic"] = statistic
    # Preserve the established field for downstream sidecar consumers.
    metadata["UncertaintyMethod"] = legacy_method

    if statistic == "credible_interval_bound":
        metadata["CredibleIntervalProbability"] = 0.95
        if normalized_inference == "mcmc":
            construction = "posterior_sample_quantiles"
        elif normalized_inference == "map":
            construction = (
                "laplace_mean_plus_or_minus_1.96_posterior_standard_deviations_"
                "or_full_parameter_bounds_fallback"
            )
        else:
            construction = "mean_plus_or_minus_1.96_posterior_standard_deviations"
        metadata["CredibleIntervalConstruction"] = construction
        if modality.upper() == "DCE":
            metadata["CredibleIntervalBounding"] = "parameter_bounds"
        elif modality.upper() == "DSC":
            metadata["CredibleIntervalBounding"] = "nonnegative_lower_bound"

    return metadata
