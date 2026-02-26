"""DCE pharmacokinetic models.

This module provides pharmacokinetic (PK) models for DCE-MRI analysis,
including Standard Tofts, Extended Tofts, Patlak, and Two-Compartment
Exchange models, along with Bayesian fitting capabilities and GPU-accelerated
batch model evaluation.

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
    GPUToftsResult: Result from GPU batched Tofts model evaluation.

GPU Functions:
    batched_tofts_model_gpu: GPU-accelerated Standard Tofts model evaluation.
    batched_extended_tofts_model_gpu: GPU-accelerated Extended Tofts evaluation.
    evaluate_model_gpu: Convenience wrapper for GPU model evaluation.
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
    GPUToftsResult,
    batched_extended_tofts_model_gpu,
    batched_tofts_model_gpu,
    evaluate_model_gpu,
    get_optimal_batch_size,
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
    "GPUToftsResult",
    "ModelParameter",
    "PatlakModel",
    "PKModel",
    "ToftsModel",
    "TwoCompartmentExchangeModel",
    "batched_extended_tofts_model_gpu",
    "batched_tofts_model_gpu",
    "evaluate_model_gpu",
    "get_optimal_batch_size",
]
