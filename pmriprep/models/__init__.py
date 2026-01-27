"""PMRIPrep parametric models.

This package contains parametric models for DSC and DCE perfusion MRI
analysis, including Bayesian fitting capabilities.

Subpackages:
    dsc: DSC perfusion models (vascular, CPI)
    dce: DCE pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM)
"""

from pmriprep.models.dsc import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    VascularModel,
)

__all__ = [
    "DSCFitResult",
    "DSCModel",
    "DSCModelParameter",
    "VascularModel",
]
