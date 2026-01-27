"""DCE pharmacokinetic models.

This module provides pharmacokinetic (PK) models for DCE-MRI analysis,
including Standard Tofts, Extended Tofts, Patlak, and Two-Compartment
Exchange models, along with Bayesian fitting capabilities.

Classes:
    ModelParameter: Parameter definition with bounds and prior information.
    FitResult: Result from fitting a PK model to DCE data.
    PKModel: Abstract base class for pharmacokinetic models.
"""

from pmriprep.models.dce.base import (
    FitResult,
    ModelParameter,
    PKModel,
)

__all__ = [
    "FitResult",
    "ModelParameter",
    "PKModel",
]
