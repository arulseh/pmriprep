"""DCE PK model base classes.

Provides base classes for DCE pharmacokinetic models including parameter
definitions, fit results with model selection criteria, and abstract model
interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Final, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from pmriprep.models.bayesian_utils import TransformType

# Type alias for parameter arrays (accepts both list and ndarray)
ParamsType = Union[Sequence[float], npt.NDArray[np.floating]]

# Floor for RSS in BIC/AIC computation: a perfect fit (RSS=0) should yield
# the lowest (most negative) information criterion, not inf.
EPSILON_RSS: Final[float] = 1e-30


@dataclass
class ModelParameter:
    """Definition of a model parameter with bounds and prior information.

    Attributes:
        name: Parameter identifier (e.g., 'Ktrans', 've', 'vp').
        bounds: Tuple of (lower, upper) hard bounds for the parameter.
        typical_range: Tuple of (low, high) values representing the typical
            physiological range. Used for prior construction in Bayesian fitting.
        transform: Parameter space transform for Bayesian fitting.
            LOG for strictly positive params, LOG_LOGIT for bounded fractions,
            IDENTITY for params that can be negative.
        transformed_prior_sigma: Optional override for the prior standard
            deviation in transformed space. See DSCModelParameter for details.
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


@dataclass
class FitResult:
    """Result from fitting a PK model to DCE tissue concentration data.

    Contains fitted parameter values, uncertainty estimates (for Bayesian
    fitting), credible intervals, goodness-of-fit metrics, and model
    selection criteria.

    Attributes:
        # Fitted parameter values (dict: parameter_name -> value)
        parameters: Dictionary mapping parameter names to fitted values.

        # Uncertainty estimates (for Bayesian methods)
        uncertainties: Dictionary mapping parameter names to standard deviations.
            Only populated for MAP with Laplace approximation or MCMC.

        # Credible intervals (for Bayesian methods)
        credible_intervals: Dictionary mapping parameter names to (lower, upper)
            tuples representing 95% credible intervals. Only populated for
            Bayesian methods.

        # Goodness-of-fit metrics
        r_squared: Coefficient of determination (R²) for the fit.
        residual_sum_squares: Sum of squared residuals.
        aic: Akaike Information Criterion for model selection.
        bic: Bayesian Information Criterion for model selection.

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
        fitting_method: Method used ('nlls', 'map', 'mcmc').

        # Fitted curve
        fitted_curve: Fitted tissue concentration curve C_t(t).
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

    # Fitted curve
    fitted_curve: Optional[npt.NDArray[np.floating]] = None

    # Fit quality flag: "good", "poor_convergence", or "high_uncertainty"
    quality_flag: str = "good"

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

    def is_better_than(self, other: "FitResult", criterion: str = "aic") -> bool:
        """Compare fit quality with another result using model selection criterion.

        Args:
            other: Another FitResult to compare against.
            criterion: Model selection criterion ('aic', 'bic', 'r_squared').

        Returns:
            True if this result is better than other.

        Raises:
            ValueError: If criterion is not recognized.
        """
        if criterion == "aic":
            return self.aic < other.aic
        elif criterion == "bic":
            return self.bic < other.bic
        elif criterion == "r_squared":
            return self.r_squared > other.r_squared
        else:
            raise ValueError(f"Unknown criterion: {criterion}")


class PKModel(ABC):
    """Abstract base class for DCE pharmacokinetic models.

    Defines the interface for DCE pharmacokinetic models that compute tissue
    concentration curves from the convolution of a plasma input function (AIF)
    with model-specific impulse response functions.

    Subclasses must implement:
        - evaluate(): Compute tissue concentration C_t(t) from parameters and AIF
        - get_parameter_definitions(): Return parameter specifications
        - fit(): Fit model to observed data

    The general Tofts family models follow:
        C_t(t) = vp × Cp(t) + Ktrans × [Cp ⊗ exp(-kep×t)]

    where:
        C_t = tissue concentration
        Cp = plasma concentration (AIF)
        Ktrans = volume transfer constant (min⁻¹)
        ve = extravascular extracellular volume fraction
        vp = plasma volume fraction
        kep = Ktrans / ve = efflux rate constant (min⁻¹)
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
    def get_parameter_definitions(self) -> Sequence[ModelParameter]:
        """Get parameter definitions for this model.

        Returns:
            List of ModelParameter objects defining each parameter's
            bounds, typical range, and transformation properties.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        params: ParamsType,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate the tissue concentration model.

        Computes C_t(t) for the given parameters and arterial input function.

        Args:
            params: Parameter values in the order from get_parameter_names().
            aif: Arterial input function plasma concentration values Cp(t).
            time_points: Time points (minutes) at which to evaluate C_t.

        Returns:
            Tissue concentration values C_t(t) at each time point.
        """
        pass

    @abstractmethod
    def fit(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        initial_guess: Optional[Sequence[float]] = None,
        weights: Optional[npt.NDArray[np.floating]] = None,
    ) -> FitResult:
        """Fit the model to observed tissue concentration data.

        Args:
            tissue_curve: Observed tissue concentration values C_t(t).
            aif: Arterial input function plasma concentration values Cp(t).
            time_points: Time points (minutes) corresponding to measurements.
            initial_guess: Optional initial parameter values. If None, uses
                default initial guess from get_initial_guess().
            weights: Optional per-frame weights for motion-weighted fitting.
                Values in [0, 1] where 1.0 = full weight. If None, uniform
                weighting is used.

        Returns:
            FitResult containing fitted parameters and diagnostics.
        """
        pass

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names.

        Returns:
            List of parameter name strings in the order expected by evaluate().
        """
        return [p.name for p in self.get_parameter_definitions()]

    def get_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization.

        Returns:
            List of (lower, upper) bound tuples for each parameter.
        """
        return [p.bounds for p in self.get_parameter_definitions()]

    def _apply_vp_bounds_override(
        self,
        bounds: list[tuple[float, float]],
        vp_bounds: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Override vp bounds in a bounds list.

        Args:
            bounds: List of (lo, hi) per parameter from get_bounds().
            vp_bounds: New (lo, hi) for the vp parameter.

        Returns:
            Modified bounds list with vp entry replaced.
        """
        names = self.get_parameter_names()
        if "vp" in names:
            idx = names.index("vp")
            bounds = list(bounds)
            bounds[idx] = vp_bounds
        return bounds

    def get_initial_guess(self) -> list[float]:
        """Get initial parameter guess from prior means.

        Returns:
            List of initial parameter values (center of typical range).
        """
        return [p.get_prior_mean() for p in self.get_parameter_definitions()]

    def estimate_initial_guess(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> list[float]:
        """Estimate data-driven initial parameters from the data.

        Override in subclasses for model-specific heuristics.
        Default falls back to prior midpoints via get_initial_guess().

        Args:
            tissue_curve: Observed tissue concentration C_t(t).
            aif: Arterial input function Cp(t).
            time_points: Time points in minutes.

        Returns:
            List of initial parameter values.
        """
        return self.get_initial_guess()

    def validate_params(self, params: ParamsType) -> bool:
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

    def clip_params(self, params: ParamsType) -> list[float]:
        """Clip parameters to bounds.

        Args:
            params: Parameter values to clip.

        Returns:
            List of parameter values clipped to bounds.
        """
        definitions = self.get_parameter_definitions()
        return [d.clip_to_bounds(p) for d, p in zip(definitions, params)]

    def params_to_dict(self, params: ParamsType) -> dict[str, float]:
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

    def compute_derived_parameters(
        self, params: ParamsType
    ) -> dict[str, float]:
        """Compute derived parameters from fitted values.

        Override in subclasses to compute model-specific derived parameters
        like kep = Ktrans/ve for Tofts models.

        Args:
            params: Fitted parameter values.

        Returns:
            Dictionary of derived parameter names to values.
            Default implementation returns empty dict.
        """
        return {}

    def compute_aic(
        self,
        residual_sum_squares: float,
        n_observations: int,
    ) -> float:
        """Compute Akaike Information Criterion.

        AIC = n × ln(RSS/n) + 2k

        Args:
            residual_sum_squares: Sum of squared residuals.
            n_observations: Number of data points.

        Returns:
            AIC value. Lower is better.
        """
        n = n_observations
        k = self.n_parameters
        if n <= 0:
            return float("inf")
        rss_safe = max(residual_sum_squares, EPSILON_RSS)
        return float(n * np.log(rss_safe / n) + 2 * k)

    def compute_bic(
        self,
        residual_sum_squares: float,
        n_observations: int,
    ) -> float:
        """Compute Bayesian Information Criterion.

        BIC = n × ln(RSS/n) + k × ln(n)

        Args:
            residual_sum_squares: Sum of squared residuals.
            n_observations: Number of data points.

        Returns:
            BIC value. Lower is better.
        """
        n = n_observations
        k = self.n_parameters
        if n <= 0:
            return float("inf")
        rss_safe = max(residual_sum_squares, EPSILON_RSS)
        return float(n * np.log(rss_safe / n) + k * np.log(n))

    def compute_r_squared(
        self,
        observed: npt.NDArray[np.floating],
        predicted: npt.NDArray[np.floating],
    ) -> float:
        """Compute coefficient of determination (R²).

        R² = 1 - SS_res / SS_tot

        Args:
            observed: Observed data values.
            predicted: Model-predicted values.

        Returns:
            R² value. Higher is better (max 1.0).
        """
        ss_res = float(np.sum((observed - predicted) ** 2))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot
