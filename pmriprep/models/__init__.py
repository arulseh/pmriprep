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
    CPIModel,
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    VascularModel,
    estimate_bolus_duration,
    fit_dsc_model,
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
    "VascularModel",
    "estimate_bolus_duration",
    "fit_dsc_model",
    "select_n_by_bic",
    "select_n_control_points",
    # DCE models
    "FitResult",
    "ModelParameter",
    "PKModel",
]
