"""PMRIPrep parametric models.

This package contains parametric models for DSC and DCE perfusion MRI
analysis, including Bayesian fitting capabilities.

Subpackages:
    dsc: DSC perfusion models (vascular, CPI, Bayesian fitting)
    dce: DCE pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM)
"""

from pmriprep.models.dsc import (
    BayesianDSCFitter,
    BayesianPrior,
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    VascularModel,
    fit_dsc_model,
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
    "DSCFitResult",
    "DSCModel",
    "DSCModelParameter",
    "VascularModel",
    "fit_dsc_model",
    # DCE models
    "FitResult",
    "ModelParameter",
    "PKModel",
]
