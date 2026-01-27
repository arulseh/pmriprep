"""PMRIPrep parametric models.

This package contains parametric models for DSC and DCE perfusion MRI
analysis, including Bayesian fitting capabilities.

Subpackages:
    dsc: DSC perfusion models (vascular, CPI)
    dce: DCE pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM)
"""

from pmriprep.models.dsc import (
    CPIModel,
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    VascularModel,
    estimate_bolus_duration,
    select_n_by_bic,
    select_n_control_points,
)

__all__ = [
    "CPIModel",
    "DSCFitResult",
    "DSCModel",
    "DSCModelParameter",
    "VascularModel",
    "estimate_bolus_duration",
    "select_n_by_bic",
    "select_n_control_points",
]
