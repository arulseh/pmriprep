"""PMRIPrep parametric models.

This package contains parametric models for DSC and DCE perfusion MRI
analysis, including Bayesian fitting capabilities.

Subpackages:
    dsc: Signal-domain CPI and shared Bayesian fitting
    dce: DCE pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM)
"""

from pmriprep.models.dce import (
    FitResult,
    ModelParameter,
    PKModel,
)

__all__ = [
    # DCE models
    "FitResult",
    "ModelParameter",
    "PKModel",
]
