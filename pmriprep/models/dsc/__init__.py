"""DSC perfusion parametric models.

This module contains parametric models for DSC perfusion analysis,
including the vascular (gamma-variate) model and Control Point
Interpolation (CPI) model.
"""

from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
)

__all__ = [
    "DSCFitResult",
    "DSCModel",
    "DSCModelParameter",
]
