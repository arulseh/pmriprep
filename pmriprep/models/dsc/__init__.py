"""DSC perfusion parametric models.

This module contains parametric models for DSC perfusion analysis,
including the vascular (gamma-variate) model and Control Point
Interpolation (CPI) model, along with Bayesian fitting capabilities
and GPU-accelerated deconvolution.
"""

from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
)
from pmriprep.models.dsc.bayesian import (
    BayesianDSCFitter,
    BayesianPrior,
    fit_dsc_model,
)
from pmriprep.models.dsc.cpi import (
    CPIModel,
    estimate_bolus_duration,
    select_n_by_bic,
    select_n_control_points,
)
from pmriprep.models.dsc.gpu_deconvolution import (
    GPUDeconvolutionResult,
    batched_svd_deconvolve,
    batched_svd_deconvolve_circulant,
    batched_svd_deconvolve_standard,
    get_optimal_batch_size,
)
from pmriprep.models.dsc.vascular import VascularModel

__all__ = [
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
]
