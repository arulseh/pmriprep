"""DSC perfusion parametric models.

This module contains parametric models for DSC perfusion analysis,
including the vascular (gamma-variate) model and Control Point
Interpolation (CPI) model, along with Bayesian fitting capabilities.
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
from pmriprep.models.dsc.batched_fitting import (
    DSCBatchedFitResult,
    batched_dsc_vi_fit,
    batched_vascular_forward,
)
from pmriprep.models.dsc.vascular import VascularModel

__all__ = [
    "BayesianDSCFitter",
    "BayesianPrior",
    "DSCBatchedFitResult",
    "DSCFitResult",
    "DSCModel",
    "DSCModelParameter",
    "ParamsLike",
    "VascularModel",
    "batched_dsc_vi_fit",
    "batched_vascular_forward",
    "fit_dsc_model",
]
