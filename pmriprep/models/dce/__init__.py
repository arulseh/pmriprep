"""DCE pharmacokinetic models.

This module provides pharmacokinetic (PK) models for DCE-MRI analysis,
including Standard Tofts, Extended Tofts, Patlak, and Two-Compartment
Exchange models, along with Bayesian fitting capabilities and batched
model evaluation and fitting.

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
    ModelResult: Result from batched model evaluation.

Batched Functions:
    batched_tofts_model: Batched Standard Tofts model evaluation.
    batched_extended_tofts_model: Batched Extended Tofts evaluation.
    batched_patlak_model: Batched Patlak model evaluation.
    batched_2cxm_model: Batched 2CXM model evaluation.
    evaluate_model: Convenience wrapper for batched model evaluation.
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
from pmriprep.models.dce.batched_models import (
    ModelResult,
    batched_2cxm_model,
    batched_extended_tofts_model,
    batched_patlak_model,
    batched_tofts_model,
    evaluate_model,
)
from pmriprep.models.dce.batched_fitting import (
    FitResult as BatchedFitResult,
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
    "BatchedFitResult",
    "ModelResult",
    "ModelParameter",
    "PatlakModel",
    "PKModel",
    "ToftsModel",
    "TwoCompartmentExchangeModel",
    "batched_2cxm_model",
    "batched_extended_tofts_model",
    "batched_lm_fit",
    "batched_patlak_lstsq",
    "batched_patlak_model",
    "batched_tofts_model",
    "evaluate_model",
]
