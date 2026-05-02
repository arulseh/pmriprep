"""DSC model base classes.

Provides base classes for DSC parametric models including parameter
definitions, fit results, and abstract model interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

from pmriprep.models._base import BaseModelParameter
from pmriprep.models.bayesian_utils import TransformType  # re-export for callers

__all__ = [
    "DSCModelParameter",
    "DSCFitResult",
    "DSCModel",
    "ParamsLike",
    "TransformType",
]

#: Type alias for model parameter vectors — accepts both Python sequences
#: and numpy arrays (scipy optimizers return ndarrays).
ParamsLike = Union[Sequence[float], npt.NDArray[np.floating]]


@dataclass
class DSCModelParameter(BaseModelParameter):
    """DSC parametric model parameter — see BaseModelParameter."""

    pass


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
    fitted_residue: npt.NDArray[np.floating] | None = None
    fitted_tissue_curve: npt.NDArray[np.floating] | None = None

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
        params: ParamsLike,
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
        params: ParamsLike,
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

    def get_fitting_parameter_definitions(self) -> Sequence['DSCModelParameter']:
        """Parameter definitions for Bayesian fitting space.

        Override when the fitting parameterization differs from the model
        parameterization (e.g., VascularModel: alpha/beta -> MTT/alpha).
        Default: returns get_parameter_definitions() (no reparameterization).

        Returns:
            Sequence of DSCModelParameter objects for fitting-space parameters.
        """
        return self.get_parameter_definitions()

    def fitting_to_model_params(
        self, fitting_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        """Convert fitting-space parameters to model-space parameters.

        Default: identity (no conversion).

        Args:
            fitting_params: Parameter values in fitting space.

        Returns:
            Parameter values in model space.
        """
        return np.asarray(fitting_params, dtype=np.float64)

    def model_to_fitting_params(
        self, model_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        """Convert model-space parameters to fitting-space parameters.

        Default: identity (no conversion).

        Args:
            model_params: Parameter values in model space.

        Returns:
            Parameter values in fitting space.
        """
        return np.asarray(model_params, dtype=np.float64)

    def compute_nlls_seed(
        self,
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating] | None:
        """Compute NLLS parameter seed for Bayesian methods.

        Models with expensive or high-dimensional parameterizations can
        override this to provide a quick NLLS fit as a starting point for
        VI/MAP, preventing poor initialization from collapsing the noise
        precision estimate.

        Args:
            concentration: Measured tissue concentration curve.
            aif: Arterial input function.
            time_points: Time points (seconds).

        Returns:
            Parameter array from NLLS fit, or None if not available.
        """
        return None

    def compute_monotonicity_prior(
        self,
        params: ParamsLike,
        sigma_delta: float = 0.15,
        violation_scale: float = 10.0,
    ) -> float:
        """Monotonicity log-prior for Bayesian fitting. Returns 0.0.

        Vestigial — CPI now guarantees monotonicity structurally via
        multiplicative fractions (cumulative product). No model overrides
        this method.
        """
        return 0.0

    def compute_monotonicity_precision(
        self,
        sigma_delta: float = 0.15,
    ) -> npt.NDArray[np.floating] | None:
        """Monotonicity precision matrix for VI. Returns None.

        Vestigial — CPI now guarantees monotonicity structurally.
        Callers guard with ``if mono_prec is not None``.
        """
        return None

    def compute_monotonicity_eta_correction(
        self,
        sigma_anchor: float = 3.0,
    ) -> npt.NDArray[np.floating] | None:
        """Monotonicity eta correction for VI. Returns None.

        Vestigial — CPI now guarantees monotonicity structurally.
        Callers guard with ``if mono_eta is not None``.
        """
        return None

    def compute_fitted_residue(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute the residue function for fitted output storage.

        By default returns residue_function() directly. Models where
        residue_function() returns a dimensionless form (e.g. CPI) should
        override this to include the appropriate scaling (e.g. CBF × R).

        Args:
            params: Parameter values.
            time_points: Time points (seconds).

        Returns:
            Residue function suitable for downstream CBF/CBV extraction.
        """
        return self.residue_function(params, time_points)

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

    def validate_params(self, params: ParamsLike) -> bool:
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

    def clip_params(self, params: ParamsLike) -> list[float]:
        """Clip parameters to bounds.

        Args:
            params: Parameter values to clip.

        Returns:
            List of parameter values clipped to bounds.
        """
        definitions = self.get_parameter_definitions()
        return [d.clip_to_bounds(p) for d, p in zip(definitions, params)]

    def params_to_dict(self, params: ParamsLike) -> dict[str, float]:
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
