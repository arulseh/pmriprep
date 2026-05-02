"""Shared base dataclasses for DCE and DSC pharmacokinetic models.

Both modalities define a parameter dataclass with identical fields and
methods (bounds, typical_range, transform, prior helpers). This module
provides the canonical implementation; ``models/dce/base.py`` and
``models/dsc/base.py`` subclass it.
"""

from dataclasses import dataclass

import numpy as np

from pmriprep.models.bayesian_utils import TransformType


@dataclass
class BaseModelParameter:
    """Definition of a model parameter with bounds and prior information.

    Attributes:
        name: Parameter identifier (e.g., 'Ktrans', 'cbf', 'alpha').
        bounds: Tuple of (lower, upper) hard bounds for the parameter.
        typical_range: Tuple of (low, high) values representing the typical
            physiological range. Used for prior construction in Bayesian
            fitting (midpoint = prior mean, range/4 = prior sigma).
        transform: Parameter space transform for Bayesian fitting.
            LOG for strictly positive params, LOG_LOGIT for bounded fractions,
            IDENTITY for params that can be negative.
        transformed_prior_sigma: Optional override for the prior standard
            deviation in transformed space. When set, bypasses the
            delta-method approximation from typical_range. Units are in the
            space defined by ``transform`` (log-space for LOG, log-odds for
            LOG_LOGIT). FABBER convention uses 1.0 for LOG parameters.
        description: Human-readable description of the parameter.
        units: Physical units of the parameter.
    """

    name: str
    bounds: tuple[float, float]
    typical_range: tuple[float, float]
    transform: TransformType = TransformType.IDENTITY
    transformed_prior_sigma: float | None = None
    description: str = ""
    units: str = ""

    def get_prior_sigma(self) -> float:
        """Compute prior standard deviation from typical range.

        Uses the rule that the typical_range should span approximately
        ±2 standard deviations (95% of the prior mass), so:
        sigma = (typical_range[1] - typical_range[0]) / 4.0

        Returns:
            Prior standard deviation for Gaussian prior construction.
        """
        return (self.typical_range[1] - self.typical_range[0]) / 4.0

    def get_prior_mean(self) -> float:
        """Compute prior mean from typical range midpoint.

        Returns:
            Prior mean (center of typical range).
        """
        return (self.typical_range[0] + self.typical_range[1]) / 2.0

    def clip_to_bounds(self, value: float) -> float:
        """Clip a value to the parameter bounds.

        Args:
            value: Parameter value to clip.

        Returns:
            Value clipped to (bounds[0], bounds[1]).
        """
        return float(np.clip(value, self.bounds[0], self.bounds[1]))

    def is_valid(self, value: float) -> bool:
        """Check if a value is within bounds.

        Args:
            value: Parameter value to check.

        Returns:
            True if value is within bounds.
        """
        return self.bounds[0] <= value <= self.bounds[1]
