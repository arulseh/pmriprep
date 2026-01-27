"""PMRIPrep parametric models.

This package contains parametric models for DSC and DCE perfusion MRI
analysis, including Bayesian fitting capabilities and GPU-accelerated
deconvolution.

Subpackages:
    dsc: DSC perfusion models (vascular, CPI, Bayesian fitting, GPU deconvolution)
    dce: DCE pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM)
"""

from pmriprep.models.dsc import (
    BayesianDSCFitter,
    BayesianPrior,
    CPIModel,
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    GPUDeconvolutionResult,
    VascularModel,
    batched_svd_deconvolve,
    batched_svd_deconvolve_circulant,
    batched_svd_deconvolve_standard,
    estimate_bolus_duration,
    fit_dsc_model,
    get_optimal_batch_size,
    select_n_by_bic,
    select_n_control_points,
)

from pmriprep.models.dce import (
    FitResult,
    ModelParameter,
    PKModel,
)

__all__ = [
    # DSC models
    "BayesianDSCFitter",
    "BayesianPrior",
    "CPIModel",
    "DSCFitResult",
    "DSCModel",
    "DSCModelParameter",
    "GPUDeconvolutionResult",
    "VascularModel",
    "batched_svd_deconvolve",
    "batched_svd_deconvolve_circulant",
    "batched_svd_deconvolve_standard",
    "estimate_bolus_duration",
    "fit_dsc_model",
    "get_optimal_batch_size",
    "select_n_by_bic",
    "select_n_control_points",
    # DCE models
    "FitResult",
    "ModelParameter",
    "PKModel",
]
