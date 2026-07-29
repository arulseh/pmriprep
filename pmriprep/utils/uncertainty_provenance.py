"""Provenance descriptions for parameter-uncertainty derivatives."""

from __future__ import annotations


_INFERENCE_METADATA: dict[str, dict[str, object]] = {
    "nlls": {
        "InferenceMethod": "nonlinear_least_squares",
        "UncertaintyEstimationMethod": "local_jacobian_covariance",
        "UncertaintyApproximation": "local_linear_Gaussian",
        "SpatialPriorApplied": False,
    },
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
        "SpatialUpdateScheme": "synchronous_batched_backtracking",
        "SpatiallyRegularizedParameters": ["cbf"],
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
    """Describe how one parameter-uncertainty derivative was constructed.

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
    metadata["ConditionalOnSelectedModel"] = True
    metadata["IncludesModelSelectionUncertainty"] = False
    if modality.upper() == "DCE":
        metadata["ConditionalOnConvertedConcentration"] = True
        metadata["ConditionalOnSuppliedAIF"] = True
        metadata["ExcludedUncertaintySources"] = [
            "model_selection",
            "baseline_estimation",
            "T1_mapping",
            "B1_mapping",
            "signal_to_concentration_conversion",
        ]

    statistic = (
        "parameter_standard_error"
        if normalized_inference == "nlls"
        else "posterior_standard_deviation"
    )
    metadata["UncertaintyStatistic"] = statistic

    return metadata
