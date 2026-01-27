"""DSC model base classes.

Provides base classes for DSC parametric models including parameter
definitions, fit results, and abstract model interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt


@dataclass
class DSCModelParameter:
    """Definition of a model parameter with bounds and prior information.

    Attributes:
        name: Parameter identifier (e.g., 'cbf', 'alpha', 'beta').
        bounds: Tuple of (lower, upper) hard bounds for the parameter.
        typical_range: Tuple of (low, high) values representing the typical
            physiological range. Used for prior construction in Bayesian fitting.
        log_transform: Whether to use log-transformation during fitting.
            Useful for parameters that span orders of magnitude.
        description: Human-readable description of the parameter.
        units: Physical units of the parameter.
    """

    name: str
    bounds: tuple[float, float]
    typical_range: tuple[float, float]
    log_transform: bool = False
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


@dataclass
class DSCFitResult:
    """Result from fitting a DSC model to tissue concentration data.

    Contains fitted parameter values, uncertainty estimates (for Bayesian
    fitting), credible intervals, and goodness-of-fit metrics.

    Attributes:
        # Fitted parameter values (dict: parameter_name -> value)
        parameters: Dictionary mapping parameter names to fitted values.

        # Uncertainty estimates (for Bayesian methods)
        uncertainties: Dictionary mapping parameter names to standard deviations.
            Only populated for MAP with Laplace approximation, VI, or MCMC.

        # Credible intervals (for Bayesian methods)
        credible_intervals: Dictionary mapping parameter names to (lower, upper)
            tuples representing 95% credible intervals. Only populated for
            Bayesian methods.

        # Goodness-of-fit metrics
        r_squared: Coefficient of determination (R²) for the fit.
        residual_sum_squares: Sum of squared residuals.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.

        # MCMC diagnostics (only for MCMC fitting)
        rhat: Dictionary of Gelman-Rubin R-hat values per parameter.
            Values should be < 1.1 for convergence.
        ess: Dictionary of effective sample sizes per parameter.
            Values should be > 100 for reliable inference.

        # Model information
        model_name: Name of the model used for fitting.
        n_parameters: Number of free parameters in the model.
        n_observations: Number of time points used in fitting.
        converged: Whether the fitting algorithm converged.
        fitting_method: Method used ('nlls', 'map', 'vi', 'mcmc').

        # Fitted curves
        fitted_residue: Fitted residue function R(t).
        fitted_tissue_curve: Fitted tissue concentration curve C(t).
    """

    # Fitted parameter values
    parameters: dict[str, float] = field(default_factory=dict)

    # Uncertainty estimates (Bayesian)
    uncertainties: dict[str, float] = field(default_factory=dict)

    # Credible intervals (Bayesian)
    credible_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Goodness-of-fit metrics
    r_squared: float = 0.0
    residual_sum_squares: float = 0.0
    aic: float = float("inf")
    bic: float = float("inf")

    # MCMC diagnostics
    rhat: dict[str, float] = field(default_factory=dict)
    ess: dict[str, int] = field(default_factory=dict)

    # Model information
    model_name: str = ""
    n_parameters: int = 0
    n_observations: int = 0
    converged: bool = False
    fitting_method: str = ""

    # Fitted curves
    fitted_residue: Optional[npt.NDArray[np.floating]] = None
    fitted_tissue_curve: Optional[npt.NDArray[np.floating]] = None

    def has_uncertainty(self) -> bool:
        """Check if uncertainty estimates are available.

        Returns:
            True if uncertainties are populated.
        """
        return len(self.uncertainties) > 0

    def has_credible_intervals(self) -> bool:
        """Check if credible intervals are available.

        Returns:
            True if credible intervals are populated.
        """
        return len(self.credible_intervals) > 0

    def passes_convergence(self, rhat_max: float = 1.1, ess_min: int = 100) -> bool:
        """Check if MCMC convergence diagnostics pass.

        Args:
            rhat_max: Maximum acceptable R-hat value.
            ess_min: Minimum acceptable effective sample size.

        Returns:
            True if converged and all R-hat < rhat_max and all ESS > ess_min,
            or True if not MCMC (no diagnostics to check).
        """
        if not self.rhat or not self.ess:
            # Not MCMC, or no diagnostics available
            return self.converged

        all_rhat_ok = all(r < rhat_max for r in self.rhat.values())
        all_ess_ok = all(e > ess_min for e in self.ess.values())

        return self.converged and all_rhat_ok and all_ess_ok

    def get_parameter(self, name: str, default: float = 0.0) -> float:
        """Get a parameter value by name.

        Args:
            name: Parameter name.
            default: Default value if parameter not found.

        Returns:
            Parameter value or default.
        """
        return self.parameters.get(name, default)

    def get_uncertainty(self, name: str, default: float = 0.0) -> float:
        """Get uncertainty (std) for a parameter.

        Args:
            name: Parameter name.
            default: Default value if uncertainty not available.

        Returns:
            Parameter uncertainty (standard deviation) or default.
        """
        return self.uncertainties.get(name, default)

    def get_credible_interval(
        self, name: str, default: tuple[float, float] = (0.0, 0.0)
    ) -> tuple[float, float]:
        """Get credible interval for a parameter.

        Args:
            name: Parameter name.
            default: Default interval if not available.

        Returns:
            (lower, upper) 95% credible interval or default.
        """
        return self.credible_intervals.get(name, default)


class DSCModel(ABC):
    """Abstract base class for DSC parametric models.

    Defines the interface for DSC perfusion models that compute tissue
    concentration curves from the convolution of an arterial input function
    (AIF) with a residue function R(t).

    The tissue concentration is: C_tissue(t) = AIF(t) ⊗ R(t)

    Subclasses must implement:
        - residue_function(): Compute R(t) from model parameters
        - tissue_curve(): Compute C_tissue(t) = AIF ⊗ R(t)
        - get_parameter_definitions(): Return parameter specifications
        - get_parameter_names(): Return list of parameter names
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for identification."""
        pass

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of free parameters in the model."""
        pass

    @abstractmethod
    def get_parameter_definitions(self) -> Sequence[DSCModelParameter]:
        """Get parameter definitions for this model.

        Returns:
            List of DSCModelParameter objects defining each parameter's
            bounds, typical range, and transformation properties.
        """
        pass

    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names.

        Returns:
            List of parameter name strings in the order expected by
            residue_function() and tissue_curve().
        """
        pass

    @abstractmethod
    def residue_function(
        self,
        params: Sequence[float],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute the residue function R(t) from model parameters.

        The residue function represents the fraction of contrast agent
        remaining in the tissue at time t after an ideal impulse input.

        Args:
            params: Parameter values in the order from get_parameter_names().
            time_points: Time points (seconds) at which to evaluate R(t).

        Returns:
            Residue function values R(t) at each time point.
        """
        pass

    @abstractmethod
    def tissue_curve(
        self,
        params: Sequence[float],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute tissue concentration curve by convolving AIF with residue.

        C_tissue(t) = AIF(t) ⊗ R(t)

        Args:
            params: Parameter values in the order from get_parameter_names().
            aif: Arterial input function concentration values.
            time_points: Time points (seconds) at which to evaluate C(t).

        Returns:
            Tissue concentration values C(t) at each time point.
        """
        pass

    def get_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization.

        Returns:
            List of (lower, upper) bound tuples for each parameter.
        """
        return [p.bounds for p in self.get_parameter_definitions()]

    def get_initial_guess(self) -> list[float]:
        """Get initial parameter guess from prior means.

        Returns:
            List of initial parameter values (center of typical range).
        """
        return [p.get_prior_mean() for p in self.get_parameter_definitions()]

    def validate_params(self, params: Sequence[float]) -> bool:
        """Check if parameters are within bounds.

        Args:
            params: Parameter values to validate.

        Returns:
            True if all parameters are within their defined bounds.
        """
        definitions = self.get_parameter_definitions()
        if len(params) != len(definitions):
            return False
        return all(d.is_valid(p) for d, p in zip(definitions, params))

    def clip_params(self, params: Sequence[float]) -> list[float]:
        """Clip parameters to bounds.

        Args:
            params: Parameter values to clip.

        Returns:
            List of parameter values clipped to bounds.
        """
        definitions = self.get_parameter_definitions()
        return [d.clip_to_bounds(p) for d, p in zip(definitions, params)]

    def params_to_dict(self, params: Sequence[float]) -> dict[str, float]:
        """Convert parameter array to named dictionary.

        Args:
            params: Parameter values in order from get_parameter_names().

        Returns:
            Dictionary mapping parameter names to values.
        """
        names = self.get_parameter_names()
        return dict(zip(names, params))

    def dict_to_params(self, param_dict: dict[str, float]) -> list[float]:
        """Convert parameter dictionary to array.

        Args:
            param_dict: Dictionary mapping parameter names to values.

        Returns:
            List of parameter values in order from get_parameter_names().
        """
        names = self.get_parameter_names()
        return [param_dict[name] for name in names]
