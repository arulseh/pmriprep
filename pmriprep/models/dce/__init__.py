"""DCE pharmacokinetic models.

This module provides pharmacokinetic (PK) models for DCE-MRI analysis,
including Standard Tofts, Extended Tofts, Patlak, and Two-Compartment
Exchange models, along with Bayesian fitting capabilities and GPU-accelerated
batch model evaluation and fitting.

Classes:
    ModelParameter: Parameter definition with bounds and prior information.
    FitResult: Result from fitting a PK model to DCE data.
    PKModel: Abstract base class for pharmacokinetic models.
    ToftsModel: Standard Tofts pharmacokinetic model.
    ExtendedToftsModel: Extended Tofts pharmacokinetic model with vp.
    PatlakModel: Patlak pharmacokinetic model (no backflux).
    TwoCompartmentExchangeModel: Two-Compartment Exchange model (2CXM).
    BayesianPKFitter: Bayesian inference for PK models (MAP, VI, MCMC).
    fit_pk_model: Convenience function for Bayesian fitting.
    GPUModelResult: Result from GPU batched model evaluation.
    GPUToftsResult: Backward-compatible alias for GPUModelResult.
    GPUFitResult: Result from GPU batched LM fitting.

GPU Functions:
    batched_tofts_model_gpu: GPU-accelerated Standard Tofts model evaluation.
    batched_extended_tofts_model_gpu: GPU-accelerated Extended Tofts evaluation.
    batched_patlak_model_gpu: GPU-accelerated Patlak model evaluation.
    batched_2cxm_model_gpu: GPU-accelerated 2CXM model evaluation.
    evaluate_model_gpu: Convenience wrapper for GPU model evaluation.
    batched_lm_fit: Batched Levenberg-Marquardt fitting for any model.
    batched_patlak_lstsq: Linear least-squares Patlak fitting.
"""

from pmriprep.models.dce.base import (
    EPSILON_RSS,
    FitResult,
    ModelParameter,
    PKModel,
)
from pmriprep.models.dce.bayesian import (
    BayesianPKFitter,
    BayesianPrior,
    create_vp_prior_from_dsc,
    fit_pk_model,
)
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.gpu_convolution import (
    GPUModelResult,
    GPUToftsResult,
    batched_2cxm_model_gpu,
    batched_extended_tofts_model_gpu,
    batched_patlak_model_gpu,
    batched_tofts_model_gpu,
    evaluate_model_gpu,
    get_optimal_batch_size,
)
from pmriprep.models.dce.gpu_fitting import (
    GPUFitResult,
    batched_lm_fit,
    batched_patlak_lstsq,
)
from pmriprep.models.dce.patlak import PatlakModel
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel

__all__ = [
    "BayesianPKFitter",
    "BayesianPrior",
    "create_vp_prior_from_dsc",
    "EPSILON_RSS",
    "ExtendedToftsModel",
    "FitResult",
    "fit_pk_model",
    "GPUFitResult",
    "GPUModelResult",
    "GPUToftsResult",
    "ModelParameter",
    "PatlakModel",
    "PKModel",
    "ToftsModel",
    "TwoCompartmentExchangeModel",
    "batched_2cxm_model_gpu",
    "batched_extended_tofts_model_gpu",
    "batched_lm_fit",
    "batched_patlak_lstsq",
    "batched_patlak_model_gpu",
    "batched_tofts_model_gpu",
    "evaluate_model_gpu",
    "get_optimal_batch_size",
]
