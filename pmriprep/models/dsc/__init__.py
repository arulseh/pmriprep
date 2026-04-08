"""DSC perfusion parametric models.

This module contains parametric models for DSC perfusion analysis,
including the vascular (gamma-variate) model and Control Point
Interpolation (CPI) model, along with Bayesian fitting capabilities
and batched deconvolution.
"""

from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    ParamsLike,
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
from pmriprep.models.dsc.batched_deconvolution import (
    DeconvolutionResult,
    batched_svd_deconvolve,
    batched_svd_deconvolve_circulant,
    batched_svd_deconvolve_standard,
    get_optimal_batch_size,
)
from pmriprep.models.dsc.batched_fitting import (
    DSCBatchedFitResult,
    batched_cpi_forward,
    batched_dsc_vi_fit,
    batched_vascular_forward,
)
from pmriprep.models.dsc.vascular import VascularModel

__all__ = [
    "BayesianDSCFitter",
    "BayesianPrior",
    "CPIModel",
    "DSCBatchedFitResult",
    "DSCFitResult",
    "DSCModel",
    "DSCModelParameter",
    "ParamsLike",
    "DeconvolutionResult",
    "VascularModel",
    "batched_cpi_forward",
    "batched_dsc_vi_fit",
    "batched_svd_deconvolve",
    "batched_svd_deconvolve_circulant",
    "batched_svd_deconvolve_standard",
    "batched_vascular_forward",
    "estimate_bolus_duration",
    "fit_dsc_model",
    "get_optimal_batch_size",
    "select_n_by_bic",
    "select_n_control_points",
]
