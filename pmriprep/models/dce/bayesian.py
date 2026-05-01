"""Bayesian inference for DCE pharmacokinetic models.

Provides Bayesian fitting methods for DCE models including:
- MAP (Maximum A Posteriori) estimation with Laplace approximation
- VI (Variational Inference) with FABBER-style linearized VB
- MCMC (Markov Chain Monte Carlo) with ensemble sampling

The BayesianPKFitter class works with any model implementing the PKModel
interface, including ToftsModel, ExtendedToftsModel, PatlakModel, and
TwoCompartmentExchangeModel.

The constraint ve + vp <= 1.0 is enforced during sampling for models
that have both ve and vp parameters.

References:
    Tofts PS, Brix G, Buckley DL, et al. (1999).
    "Estimating kinetic parameters from dynamic contrast-enhanced
    T1-weighted MRI of a diffusable tracer."
    J Magn Reson Imaging 10(3):223-232.

    Goodman J, Weare J. (2010).
    "Ensemble samplers with affine invariance."
    Comm App Math Comp Sci 5(1):65-80.

    Chappell MA, Groves AR, Whitcher B, Woolrich MW (2009).
    "Variational Bayesian Inference for a Nonlinear Forward Model."
    IEEE Trans Med Imaging 28(1):94-104.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy import optimize  # type: ignore[import-untyped]
from scipy.optimize import Bounds  # type: ignore[import-untyped]

from pmriprep.models.bayesian_utils import (
    BayesianPrior,
    TransformType,
    build_priors_from_params,
    compute_goodness_of_fit,
    compute_log_prior,
    compute_per_walker_ess,
    compute_split_rhat,
    from_transformed,
    log_jacobian_det,
    run_ensemble_sampler,
    to_transformed,
    validate_bayesian_inputs,
)
from pmriprep.models.dce.base import (
    FitResult,
    PKModel,
)

# Re-export BayesianPrior for backward compatibility
__all__ = [
    "BayesianPKFitter",
    "BayesianPrior",
    "create_vp_prior_from_dsc",
    "fit_pk_model",
]


# =============================================================================
# Bayesian Inference Constants
# =============================================================================

#: Default number of MCMC samples (after burn-in)
DEFAULT_MCMC_SAMPLES: int = 500

#: Default maximum iterations for MAP optimization
DEFAULT_MAP_MAX_ITER: int = 1000

#: Maximum acceptable R-hat for MCMC convergence
QC_RHAT_MAX: float = 1.1

#: Minimum acceptable effective sample size
QC_ESS_MIN: int = 100

#: Physiological noise floor for DCE concentration (~0.001 mM at tSNR=50).
#: Prevents profile likelihood sigma from collapsing to near-zero when
#: the model fits data very well, which would make log-likelihood overwhelm
#: the prior and reduce MAP to unconstrained MLE.
EPSILON_SIGMA_CONCENTRATION: float = 0.001


class BayesianPKFitter:
    """Bayesian inference for DCE pharmacokinetic models.

    Provides three fitting methods:
    - fit_map(): MAP estimation with Laplace approximation for uncertainties
    - fit_vi(): FABBER-style linearized Variational Bayes
    - fit_mcmc(): Full MCMC sampling with ensemble sampler

    The fitter constructs Gaussian priors from the model's parameter definitions,
    using the typical_range to determine prior mean and standard deviation.

    For models with both ve and vp parameters (Extended Tofts, 2CXM), the
    constraint ve + vp <= 1.0 is enforced during sampling.

    Example:
        >>> model = ExtendedToftsModel()
        >>> fitter = BayesianPKFitter(model)
        >>> result = fitter.fit_map(tissue_curve, aif, time_points)
        >>> print(result.parameters)
        >>> print(result.uncertainties)

    Attributes:
        model: PK model to fit (ToftsModel, ExtendedToftsModel, etc.)
        priors: Gaussian priors for each parameter
        noise_std: Assumed noise standard deviation (estimated if not provided)
    """

    def __init__(
        self,
        model: PKModel,
        noise_std: float | None = None,
        vp_prior_override: BayesianPrior | None = None,
    ) -> None:
        """Initialize the Bayesian fitter.

        Args:
            model: PK model implementing the PKModel interface.
            noise_std: Optional noise standard deviation. If None, will be
                estimated from the data during fitting.
            vp_prior_override: Optional informative prior for vp parameter,
                typically derived from DSC CBV. If provided, replaces the
                default vp prior built from model parameter definitions.
        """
        self.model = model
        self.noise_std = noise_std

        # Build priors from model parameter definitions
        self.priors = build_priors_from_params(model.get_parameter_definitions())

        # Check if model has both ve and vp (needs volume constraint)
        param_names = model.get_parameter_names()
        self._has_volume_constraint = "ve" in param_names and "vp" in param_names
        self._ve_index = param_names.index("ve") if "ve" in param_names else -1
        self._vp_index = param_names.index("vp") if "vp" in param_names else -1

        # Override vp prior if provided (for DSC CBV soft constraint)
        if vp_prior_override is not None and self._vp_index >= 0:
            self.priors[self._vp_index] = vp_prior_override

    @staticmethod
    def _validate_inputs(
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> None:
        """Validate input arrays for NaN/Inf values and length consistency.

        Args:
            tissue_curve: Measured tissue concentration time course.
            aif: Arterial input function.
            time_points: Time points (minutes).

        Raises:
            ValueError: If inputs contain NaN/Inf or lengths don't match.
        """
        validate_bayesian_inputs(
            ("tissue_curve", tissue_curve),
            ("aif", aif),
            ("time_points", time_points),
        )

    def _estimate_noise_std(
        self,
        tissue_curve: npt.NDArray[np.floating],
        baseline_frames: int = 5,
    ) -> float:
        """Estimate noise standard deviation from baseline variability.

        Uses the standard deviation of the first few baseline frames
        as an estimate of measurement noise.

        Args:
            tissue_curve: Measured tissue concentration time course.
            baseline_frames: Number of baseline frames to use.

        Returns:
            Estimated noise standard deviation.
        """
        if len(tissue_curve) < baseline_frames:
            baseline_frames = max(2, len(tissue_curve) // 2)

        baseline = tissue_curve[:baseline_frames]
        noise_std = float(np.std(baseline))

        # Minimum noise level for numerical stability
        return max(noise_std, EPSILON_SIGMA_CONCENTRATION)

    def _log_prior(
        self, params: npt.NDArray[np.floating]
    ) -> float:
        """Compute log prior probability.

        Uses simple Gaussian priors for all parameters (no log transform).

        Args:
            params: Parameter values.

        Returns:
            Log prior probability (unnormalized).
        """
        return compute_log_prior(params, self.priors)

    def _check_volume_constraint(
        self, params: npt.NDArray[np.floating]
    ) -> bool:
        """Check if ve + vp <= 1.0 constraint is satisfied.

        Args:
            params: Parameter values.

        Returns:
            True if constraint is satisfied (or not applicable).
        """
        if not self._has_volume_constraint:
            return True

        ve = params[self._ve_index]
        vp = params[self._vp_index]

        return bool((ve + vp) <= 1.0)

    def _apply_volume_constraint(
        self, params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Apply the ve + vp <= 1.0 constraint.

        If the sum exceeds 1.0, scales both ve and vp proportionally
        to satisfy the constraint while maintaining their ratio.

        Args:
            params: Parameter values.

        Returns:
            Constrained parameter values.
        """
        if not self._has_volume_constraint:
            return params

        params = params.copy()
        ve = params[self._ve_index]
        vp = params[self._vp_index]

        total_volume = ve + vp
        if total_volume > 1.0:
            # Scale down proportionally
            scale_factor = 0.999 / total_volume
            params[self._ve_index] = ve * scale_factor
            params[self._vp_index] = vp * scale_factor

        return params

    def _log_likelihood(
        self,
        params: npt.NDArray[np.floating],
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        noise_std: float,
        weights: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute log likelihood using profile likelihood noise estimation.

        Noise standard deviation is estimated from residuals at each
        evaluation (profile likelihood).
        This self-tunes as the model approaches the data, giving a
        balanced likelihood-prior tradeoff.

        When weights are provided, the profile likelihood uses weighted
        residuals to estimate noise variance, and the effective sample size
        is sum(weights) instead of n. This downweights high-motion frames.

        Note: With profile likelihood, the log-likelihood reduces to
        -n * log(std(residuals)) + const, equivalent to penalized least
        squares with the Gaussian prior providing L2 regularization.
        Laplace approximation uncertainties are approximate under profile
        likelihood -- use for relative voxel comparison, not calibrated CIs.

        Args:
            params: Parameter values.
            tissue_curve: Measured tissue concentration.
            aif: Arterial input function.
            time_points: Time points (minutes).
            noise_std: Unused (kept for API compatibility).
            weights: Optional per-frame weights in [0, 1]. If provided,
                inflates noise variance for high-FD frames.

        Returns:
            Log likelihood (unnormalized).
        """
        # Compute model prediction
        try:
            predicted = self.model.evaluate(list(params), aif, time_points)
        except (ValueError, RuntimeError):
            return float("-inf")

        # Reject parameter sets that produce NaN/inf predictions
        if not np.all(np.isfinite(predicted)):
            return float("-inf")

        # Residuals
        residuals = tissue_curve - predicted

        if weights is not None:
            # Weighted profile likelihood: inflate noise variance for high-FD frames
            sigma_val = float(np.sqrt(np.mean(weights * residuals**2)))
            n_eff = float(np.sum(weights))
        else:
            sigma_val = float(np.std(residuals))
            n_eff = float(len(tissue_curve))

        if not np.isfinite(sigma_val):
            return float("-inf")
        sigma = max(sigma_val, EPSILON_SIGMA_CONCENTRATION)

        # Gaussian log likelihood
        if weights is not None:
            log_lik = -n_eff * np.log(sigma) - 0.5 * np.sum(
                weights * (residuals / sigma) ** 2
            )
        else:
            n = len(tissue_curve)
            log_lik = -n * np.log(sigma) - 0.5 * np.sum((residuals / sigma) ** 2)

        return float(log_lik)

    def _log_likelihood_with_sigma(
        self,
        params: npt.NDArray[np.floating],
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        sigma: float,
        weights: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute log likelihood with explicit noise sigma (for MCMC).

        Unlike _log_likelihood which uses profile likelihood, this method
        uses the provided sigma directly, suitable for MCMC where sigma
        is sampled as a free parameter.

        Args:
            params: Parameter values.
            tissue_curve: Measured tissue concentration.
            aif: Arterial input function.
            time_points: Time points (minutes).
            sigma: Noise standard deviation.
            weights: Optional per-frame weights in [0, 1]. If provided,
                uses weighted Gaussian log-likelihood with effective
                sample size n_eff = sum(weights).

        Returns:
            Log likelihood (unnormalized).
        """
        try:
            predicted = self.model.evaluate(list(params), aif, time_points)
        except (ValueError, RuntimeError):
            return float("-inf")

        if not np.all(np.isfinite(predicted)):
            return float("-inf")

        residuals = tissue_curve - predicted
        if weights is not None:
            n_eff = float(np.sum(weights))
            log_lik = -0.5 * n_eff * np.log(2 * np.pi * sigma**2)
            log_lik -= 0.5 * np.sum(weights * residuals**2) / (sigma**2)
        else:
            n = len(tissue_curve)
            log_lik = -0.5 * n * np.log(2 * np.pi * sigma**2)
            log_lik -= 0.5 * np.sum(residuals**2) / (sigma**2)

        return float(log_lik)

    def _log_posterior(
        self,
        params: npt.NDArray[np.floating],
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        noise_std: float,
        weights: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute log posterior probability.

        log p(θ|y) ∝ log p(y|θ) + log p(θ)

        Args:
            params: Parameter values.
            tissue_curve: Measured tissue concentration.
            aif: Arterial input function.
            time_points: Time points (minutes).
            noise_std: Noise standard deviation.
            weights: Optional per-frame weights for motion-weighted fitting.

        Returns:
            Log posterior probability (unnormalized).
        """
        # Check bounds
        if not self.model.validate_params(params):
            return float("-inf")

        # Check volume constraint
        if not self._check_volume_constraint(params):
            return float("-inf")

        log_prior = self._log_prior(params)
        if not np.isfinite(log_prior):
            return float("-inf")

        log_lik = self._log_likelihood(
            params, tissue_curve, aif, time_points, noise_std,
            weights=weights,
        )

        posterior = log_prior + log_lik
        return posterior if np.isfinite(posterior) else float("-inf")

    def _compute_hessian(
        self,
        params: npt.NDArray[np.floating],
        objective: Callable[[npt.NDArray[np.floating]], float],
        relative_epsilon: float = 1e-5,
    ) -> npt.NDArray[np.floating]:
        """Compute Hessian matrix using finite differences.

        Uses central differences for second derivatives with adaptive
        per-parameter step sizes that scale with parameter magnitude.
        When perturbations push parameters into invalid regions (causing
        non-finite objective values), falls back to using prior curvature
        for those entries.

        Args:
            params: Parameter values at which to evaluate Hessian.
            objective: Negative log posterior function.
            relative_epsilon: Relative step size for finite differences.
                Actual step for parameter i is max(relative_epsilon,
                relative_epsilon * |params[i]|).

        Returns:
            Hessian matrix (n_params x n_params).
        """
        n = len(params)
        hessian = np.zeros((n, n), dtype=np.float64)

        # Compute per-parameter step sizes
        eps = np.array([
            max(relative_epsilon, relative_epsilon * abs(params[i]))
            for i in range(n)
        ])

        for i in range(n):
            for j in range(i, n):
                # Perturb parameters
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()

                params_pp[i] += eps[i]
                params_pp[j] += eps[j]
                params_pm[i] += eps[i]
                params_pm[j] -= eps[j]
                params_mp[i] -= eps[i]
                params_mp[j] += eps[j]
                params_mm[i] -= eps[i]
                params_mm[j] -= eps[j]

                # Second derivative
                f_pp = objective(params_pp)
                f_pm = objective(params_pm)
                f_mp = objective(params_mp)
                f_mm = objective(params_mm)

                # Check for non-finite values before arithmetic
                # This handles cases where perturbations violate constraints
                # (e.g., ve + vp > 1.0) causing objective to return inf
                if all(np.isfinite([f_pp, f_pm, f_mp, f_mm])):
                    h_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * eps[i] * eps[j])
                    # Reject extreme Hessian entries that would make
                    # covariance near-singular, producing unreliable
                    # Laplace approximation uncertainties.
                    # With profile-likelihood objectives (30-120 time points,
                    # sigma ~ 0.001-0.01), individual entries are typically
                    # O(10^3-10^6). The 1e8 threshold provides ~100x headroom
                    # above expected maxima.
                    if abs(h_ij) < 1e8:
                        hessian[i, j] = h_ij
                    elif i == j:
                        prior_std = self.priors[i].std
                        if prior_std > 0:
                            hessian[i, j] = 1.0 / (prior_std**2)
                        else:
                            hessian[i, j] = 1.0
                    # Off-diagonal remains 0 (assumes independent parameters)
                else:
                    # Fall back to prior curvature when perturbations hit invalid region
                    # Use 1/prior_variance as diagonal Hessian approximation
                    if i == j:
                        prior_std = self.priors[i].std
                        if prior_std > 0:
                            hessian[i, j] = 1.0 / (prior_std**2)
                        else:
                            hessian[i, j] = 1.0  # Default regularization
                    # Off-diagonal remains 0 (assumes independent parameters)

                hessian[j, i] = hessian[i, j]

        return hessian

    def _compute_laplace_uncertainties(
        self,
        params: npt.NDArray[np.floating],
        objective: Callable[[npt.NDArray[np.floating]], float],
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        """Compute uncertainties using Laplace approximation.

        The Laplace approximation uses the inverse Hessian at the MAP
        estimate as an approximation to the posterior covariance.

        Args:
            params: MAP parameter estimates.
            objective: Negative log posterior function.

        Returns:
            Tuple of:
                - uncertainties: dict mapping param names to std deviations
                - credible_intervals: dict mapping param names to 95% CIs
        """
        names = self.model.get_parameter_names()
        uncertainties: dict[str, float] = {}
        credible_intervals: dict[str, tuple[float, float]] = {}

        try:
            # Compute Hessian
            hessian = self._compute_hessian(params, objective)

            # Add regularization for stability
            reg = 1e-6 * np.eye(len(params))
            hessian_reg = hessian + reg

            # Inverse Hessian gives covariance approximation
            covariance = np.linalg.inv(hessian_reg)

            # Extract standard deviations
            variances = np.diag(covariance)
            stds = np.sqrt(np.maximum(variances, 0.0))

            for i, name in enumerate(names):
                std = float(stds[i])
                uncertainties[name] = std

                # 95% credible interval (mean ± 1.96 std)
                ci_low = float(params[i] - 1.96 * std)
                ci_high = float(params[i] + 1.96 * std)

                # Clip to bounds
                bounds = self.model.get_bounds()
                ci_low = max(ci_low, bounds[i][0])
                ci_high = min(ci_high, bounds[i][1])

                credible_intervals[name] = (ci_low, ci_high)

        except np.linalg.LinAlgError:
            # Hessian not invertible; return large uncertainties
            bounds = self.model.get_bounds()
            for i, name in enumerate(names):
                # Use prior std as uncertainty estimate
                uncertainties[name] = self.priors[i].std
                ci_low = bounds[i][0]
                ci_high = bounds[i][1]
                credible_intervals[name] = (ci_low, ci_high)

        return uncertainties, credible_intervals

    def fit_map(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        max_iter: int = DEFAULT_MAP_MAX_ITER,
        compute_uncertainties: bool = True,
        weights: npt.NDArray[np.floating] | None = None,
        sloppy: bool = False,
    ) -> FitResult:
        """Fit model using MAP estimation with Laplace approximation.

        Maximum A Posteriori finds the mode of the posterior distribution.
        Uncertainties are estimated using the Laplace approximation
        (inverse Hessian at the mode).

        Args:
            tissue_curve: Measured tissue concentration curve (mM).
            aif: Arterial input function (mM).
            time_points: Time points (minutes).
            max_iter: Maximum optimizer iterations.
            compute_uncertainties: Whether to compute Laplace uncertainties.
                Set to False to skip the expensive Hessian computation.
            weights: Optional per-frame weights for motion-weighted fitting.
                Values in [0, 1] where 1.0 = full weight.
            sloppy: Use fast, low-quality settings for testing.

        Returns:
            FitResult with parameters, uncertainties, and credible intervals.
        """
        if sloppy:
            max_iter = 100
            compute_uncertainties = False
        self._validate_inputs(tissue_curve, aif, time_points)

        noise_std = 1.0  # Unused; profile likelihood estimates from residuals

        bounds = self.model.get_bounds()
        n_params = self.model.n_parameters

        # Build transform info
        transforms = [p.transform for p in self.priors]
        upper_bounds_t = [p.upper_bound for p in self.priors]

        # Objective in transformed space.
        # Prior is evaluated as a Gaussian directly on transformed params
        # (== log-normal on natural-space LOG params, == logit-normal on
        # natural-space LOG_LOGIT params). This matches the VI path's
        # convention and the FABBER convention. Evaluating the prior in
        # natural space and adding a Jacobian recovered the MAP of a
        # *different* prior distribution, so MAP and VI returned different
        # posterior modes for the same model when the prior was informative.
        def neg_log_posterior_t(params_t: npt.NDArray[np.floating]) -> float:
            log_prior_t = compute_log_prior(
                params_t, self.priors, transformed=True
            )
            if not np.isfinite(log_prior_t):
                return 1e30

            # Inverse-transform for likelihood (model evaluates natural params)
            params_nat = np.array([
                float(from_transformed(params_t[i], transforms[i], upper_bounds_t[i]))
                for i in range(n_params)
            ], dtype=np.float64)

            if not self.model.validate_params(params_nat):
                return 1e30
            if not self._check_volume_constraint(params_nat):
                return 1e30

            log_lik = self._log_likelihood(
                params_nat, tissue_curve, aif, time_points, noise_std,
                weights=weights,
            )
            if not np.isfinite(log_lik):
                return 1e30

            return min(-(log_prior_t + log_lik), 1e30)

        # Multi-start: try NLLS warm-start, prior mean, and data-driven guess.
        # Use Nelder-Mead (gradient-free) because the profile likelihood
        # has non-smooth gradients that cause L-BFGS-B to stall.
        x0_prior_nat = np.array(self.model.get_initial_guess(), dtype=np.float64)
        x0_data_nat = np.array(
            self.model.estimate_initial_guess(tissue_curve, aif, time_points),
            dtype=np.float64,
        )

        # NLLS warm-start: use model.fit() (curve_fit TRF) as third start point
        x0_nlls_nat = x0_prior_nat.copy()
        try:
            nlls_result = self.model.fit(
                tissue_curve, aif, time_points, weights=weights,
            )
            if nlls_result.converged:
                param_names = self.model.get_parameter_names()
                x0_nlls_nat = np.array(
                    [nlls_result.parameters.get(p, x0_prior_nat[i])
                     for i, p in enumerate(param_names)],
                    dtype=np.float64,
                )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass

        # Clip natural-space start points, then transform to unconstrained space
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        def _to_t(x_nat: npt.NDArray) -> npt.NDArray:
            x_nat = np.clip(x_nat, lower, upper)
            return np.array([
                float(to_transformed(x_nat[i], transforms[i], upper_bounds_t[i]))
                for i in range(n_params)
            ], dtype=np.float64)

        x0_nlls_t = _to_t(x0_nlls_nat)
        x0_prior_t = _to_t(x0_prior_nat)
        x0_data_t = _to_t(x0_data_nat)

        best_fun = float("inf")
        converged = False
        fitted_params_t = x0_prior_t

        for x0 in [x0_nlls_t, x0_prior_t, x0_data_t]:
            try:
                result = optimize.minimize(
                    neg_log_posterior_t,
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": max_iter,
                        "xatol": 1e-3 if sloppy else 1e-6,
                        "fatol": 1e-3 if sloppy else 1e-6,
                    },
                )
                if result.fun < best_fun:
                    best_fun = result.fun
                    converged = result.success and result.fun < 1e20
                    fitted_params_t = np.array(result.x, dtype=np.float64)
            except (ValueError, RuntimeError):
                continue

        # Inverse-transform to natural space
        fitted_params = np.array([
            float(from_transformed(fitted_params_t[i], transforms[i], upper_bounds_t[i]))
            for i in range(n_params)
        ], dtype=np.float64)

        # Apply volume constraint (natural space, no clipping)
        fitted_params = self._apply_volume_constraint(fitted_params)  # type: ignore[assignment]

        # Compute Laplace approximation uncertainties.
        # Hessian is taken on the transformed-space objective so its
        # prior contribution matches what MAP optimized; the resulting
        # transformed-space std is back-transformed to natural space
        # via the delta method (jac = exp(log|d theta_nat / d theta_t|)).
        if compute_uncertainties:
            try:
                hessian_t = self._compute_hessian(
                    fitted_params_t, neg_log_posterior_t
                )
                cov_t = np.linalg.inv(
                    hessian_t + 1e-6 * np.eye(n_params)
                )
                stds_t = np.sqrt(np.maximum(np.diag(cov_t), 0.0))

                names = self.model.get_parameter_names()
                bounds_natural = self.model.get_bounds()
                uncertainties = {}
                credible_intervals = {}
                for i, name in enumerate(names):
                    theta_nat = float(fitted_params[i])
                    jac = float(np.exp(log_jacobian_det(
                        theta_nat, transforms[i], upper_bounds_t[i]
                    )))
                    std_nat = float(stds_t[i]) * jac
                    uncertainties[name] = std_nat
                    ci_low = max(theta_nat - 1.96 * std_nat, bounds_natural[i][0])
                    ci_high = min(theta_nat + 1.96 * std_nat, bounds_natural[i][1])
                    credible_intervals[name] = (ci_low, ci_high)
            except np.linalg.LinAlgError:
                # Hessian not invertible; fall back to prior std as
                # uncertainty estimate (matches _compute_laplace_uncertainties).
                names = self.model.get_parameter_names()
                bounds_natural = self.model.get_bounds()
                uncertainties = {
                    name: self.priors[i].std
                    for i, name in enumerate(names)
                }
                credible_intervals = {
                    name: (bounds_natural[i][0], bounds_natural[i][1])
                    for i, name in enumerate(names)
                }
        else:
            uncertainties = {}
            credible_intervals = {}

        # Compute fitted curve
        fitted_curve = self.model.evaluate(list(fitted_params), aif, time_points)

        # Compute goodness of fit
        residuals = tissue_curve - fitted_curve
        rss = float(np.sum(residuals**2))
        r_squared = self.model.compute_r_squared(tissue_curve, fitted_curve)

        # Information criteria
        n = len(tissue_curve)
        k = self.model.n_parameters
        aic = self.model.compute_aic(rss, n)
        bic = self.model.compute_bic(rss, n)

        # Compute derived parameters
        param_dict = self.model.params_to_dict(list(fitted_params))
        derived = self.model.compute_derived_parameters(list(fitted_params))
        param_dict.update(derived)

        return FitResult(
            parameters=param_dict,
            uncertainties=uncertainties,
            credible_intervals=credible_intervals,
            r_squared=r_squared,
            residual_sum_squares=rss,
            aic=aic,
            bic=bic,
            model_name=self.model.name,
            n_parameters=k,
            n_observations=n,
            converged=converged,
            fitting_method="map",
            fitted_curve=fitted_curve,
        )

    def fit_vi(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        n_iterations: int = 30,
        noise_prior: tuple[float, float] | None = None,
        weights: npt.NDArray[np.floating] | None = None,
        sloppy: bool = False,
    ) -> FitResult:
        """Fit model using FABBER-style linearized Variational Bayes.

        Implements the linearized VB algorithm from Chappell et al. (2009,
        IEEE TMI 28(1):94-104). At each iteration the forward model is
        linearized around the current posterior mean, enabling analytic
        Gaussian posterior updates. Noise precision is treated as a conjugate
        Gamma random variable.

        Args:
            tissue_curve: Measured tissue concentration curve (mM).
            aif: Arterial input function (mM).
            time_points: Time points (minutes).
            n_iterations: Maximum VB iterations (default 20).
            noise_prior: Optional (c0, s0) Gamma prior on noise precision
                tau = 1/sigma^2. If None, uses non-informative prior.
                If noise_std was provided to the constructor, uses an
                informative prior centered on that value.
            weights: Optional per-frame motion weights (1.0 = full weight).
                Weights modify data fidelity terms (J^T W J, k^T W k) but
                NOT the observation count in Gamma shape/ELBO (Chappell 2009).
            sloppy: Use fast, low-quality settings for testing.

        Returns:
            FitResult with parameters, uncertainties, and credible intervals.

        References:
            Chappell MA, Groves AR, Whitcher B, Woolrich MW (2009).
            "Variational Bayesian Inference for a Nonlinear Forward Model."
            IEEE Trans Med Imaging 28(1):94-104.
        """
        if sloppy:
            n_iterations = 8
        self._validate_inputs(tissue_curve, aif, time_points)

        n_params = self.model.n_parameters
        n_obs = len(tissue_curve)
        bounds = self.model.get_bounds()
        param_names = self.model.get_parameter_names()
        y = np.asarray(tissue_curve, dtype=np.float64)

        # Motion weights for data fidelity terms
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
        else:
            w = None

        # Build transform info from priors
        transforms = [p.transform for p in self.priors]
        upper_bounds_t = [p.upper_bound for p in self.priors]

        # Prior precision matrix in transformed space
        prior_means_t = np.array(
            [p.mean_transformed for p in self.priors], dtype=np.float64
        )
        prior_stds_t = np.array(
            [p.std_transformed for p in self.priors], dtype=np.float64
        )
        prior_prec = np.diag(1.0 / (prior_stds_t**2))  # Λ₀

        # Noise precision prior: Gamma(c0, s0) in shape-scale form.
        # E[tau] = c0 * s0 (matching DSC convention).
        if noise_prior is not None:
            c0, s0 = noise_prior
        elif self.noise_std is not None:
            tau_prior = 1.0 / (self.noise_std**2)
            c0 = 10.0
            s0 = tau_prior / c0  # so E[tau] = c0 * s0 = tau_prior
        else:
            c0 = 1e-3
            s0 = 1e3

        # Initialize posterior mean at prior mean (natural), clipped to bounds,
        # then transform to fitting space
        m_nat = np.array([p.mean for p in self.priors], dtype=np.float64)
        for i in range(n_params):
            m_nat[i] = np.clip(m_nat[i], bounds[i][0], bounds[i][1])

        m = np.array([
            float(to_transformed(m_nat[i], transforms[i], upper_bounds_t[i]))
            for i in range(n_params)
        ], dtype=np.float64)

        # Initialize noise posterior
        c = c0
        s = s0
        tau_exp = c * s  # E[tau] = shape * scale

        # Best free energy tracking with step-size reduction
        best_free_energy = float("-inf")
        best_m = m.copy()
        best_cov = np.diag(prior_stds_t**2)
        best_c = c
        best_s = s
        step_scale = 1.0

        converged = False
        fe_plateau_count = 0

        for iteration in range(n_iterations):
            # Inverse-transform m to natural space for forward model
            m_nat = np.array([
                float(from_transformed(m[k], transforms[k], upper_bounds_t[k]))
                for k in range(n_params)
            ], dtype=np.float64)

            # 1. Evaluate forward model at current mean
            try:
                g_m = self.model.evaluate(list(m_nat), aif, time_points)
            except (ValueError, RuntimeError):
                break
            g_m = np.asarray(g_m, dtype=np.float64)

            # 2. Compute Jacobian via central finite differences in
            # transformed space
            J = np.zeros((n_obs, n_params), dtype=np.float64)
            deltas = np.maximum(np.abs(m) * 1e-5, 1e-10)
            for j in range(n_params):
                # Only inverse-transform the perturbed param (not all n_params)
                p_nat = m_nat.copy()
                n_nat = m_nat.copy()
                p_nat[j] = float(from_transformed(m[j] + deltas[j], transforms[j], upper_bounds_t[j]))
                n_nat[j] = float(from_transformed(m[j] - deltas[j], transforms[j], upper_bounds_t[j]))
                try:
                    g_plus = self.model.evaluate(list(p_nat), aif, time_points)
                    g_minus = self.model.evaluate(list(n_nat), aif, time_points)
                except (ValueError, RuntimeError):
                    continue
                J[:, j] = (np.asarray(g_plus) - np.asarray(g_minus)) / (2 * deltas[j])

            # Residual
            k_res = y - g_m

            # 3. Update posterior precision: Λ = Λ₀ + <τ> J^T W J
            if w is not None:
                JtWJ = J.T @ (w[:, np.newaxis] * J)
            else:
                JtWJ = J.T @ J
            post_prec = prior_prec + tau_exp * JtWJ

            # 4. Update posterior covariance
            try:
                post_cov = np.linalg.inv(post_prec)
            except np.linalg.LinAlgError:
                post_prec += 1e-6 * np.eye(n_params)
                post_cov = np.linalg.inv(post_prec)

            # 5. Update posterior mean (in transformed space)
            if w is not None:
                rhs = prior_prec @ prior_means_t + tau_exp * J.T @ (w * (k_res + J @ m))
            else:
                rhs = prior_prec @ prior_means_t + tau_exp * J.T @ (k_res + J @ m)
            m_new = post_cov @ rhs

            # NO CLIPPING — transformed space is unconstrained

            # 6. Update noise precision posterior
            c = c0 + n_obs / 2.0
            m_new_nat = np.array([
                float(from_transformed(m_new[k], transforms[k], upper_bounds_t[k]))
                for k in range(n_params)
            ], dtype=np.float64)
            try:
                g_new = self.model.evaluate(list(m_new_nat), aif, time_points)
            except (ValueError, RuntimeError):
                g_new = g_m
            k_new = y - np.asarray(g_new, dtype=np.float64)
            trace_term = float(np.trace(post_cov @ JtWJ))
            if w is not None:
                weighted_kk = float(k_new @ (w * k_new))
            else:
                weighted_kk = float(k_new @ k_new)
            s = 1.0 / (1.0 / s0 + 0.5 * (weighted_kk + trace_term))
            tau_exp = c * s

            # 7. Compute free energy (analytic ELBO)
            dm = m_new - prior_means_t
            sign_cov, logdet_cov = np.linalg.slogdet(post_cov)
            sign_prior_prec, logdet_prior_prec = np.linalg.slogdet(prior_prec)
            logdet_prior_cov = -logdet_prior_prec
            sign_prior = sign_prior_prec

            free_energy = (
                0.5 * n_obs * (np.log(tau_exp) - np.log(2 * np.pi))
                - 0.5 * tau_exp * (weighted_kk + trace_term)
                - 0.5 * float(dm @ prior_prec @ dm)
                - 0.5 * float(np.trace(prior_prec @ post_cov))
                + 0.5 * logdet_cov * sign_cov
                - 0.5 * logdet_prior_cov * sign_prior
                + 0.5 * n_params
            )

            # Gamma KL: -KL(q(tau) || p(tau)) for shape-scale Gamma
            from scipy.special import digamma, gammaln
            gamma_kl = -((c - c0) * digamma(c) - gammaln(c) + gammaln(c0)
                          + c0 * (np.log(s0) - np.log(s))
                          + c * (s / s0 - 1.0))
            free_energy += gamma_kl

            # Step-size reduction: if free energy decreased, interpolate
            if free_energy > best_free_energy:
                best_free_energy = free_energy
                best_m = m_new.copy()
                best_cov = post_cov.copy()
                best_c = c
                best_s = s
                step_scale = 1.0
            else:
                step_scale *= 0.5
                if step_scale < 1e-4:
                    break
                m_new = m + step_scale * (m_new - m)

            # Convergence check
            if iteration > 0:
                param_change = float(np.max(np.abs(m_new - m)))
                if param_change < 1e-3:  # 0.1% relative for log-params
                    converged = True
                    m = m_new
                    break
                # Free energy convergence: 2 consecutive negligible improvements
                if (iteration > 2
                        and best_free_energy > float("-inf")
                        and abs(free_energy - best_free_energy)
                        < 1e-4 * abs(best_free_energy)):
                    fe_plateau_count += 1
                    if fe_plateau_count >= 2:
                        converged = True
                        m = m_new
                        break
                else:
                    fe_plateau_count = 0

            m = m_new

        # Inverse-transform best parameters to natural space
        fitted_params = np.array([
            float(from_transformed(best_m[k], transforms[k], upper_bounds_t[k]))
            for k in range(n_params)
        ], dtype=np.float64)

        fitted_stds_t = np.sqrt(np.maximum(np.diag(best_cov), 0.0))
        noise_std_posterior = 1.0 / np.sqrt(max(best_c * best_s, 1e-300))

        # Build uncertainties and credible intervals.
        # Back-transform covariance using delta method:
        # std_natural ≈ std_transformed * |dtheta/dtheta'|
        uncertainties: dict[str, float] = {}
        credible_intervals: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(param_names):
            std_t = float(fitted_stds_t[i])
            theta_nat = float(fitted_params[i])
            jac = float(np.exp(log_jacobian_det(
                theta_nat, transforms[i], upper_bounds_t[i]
            )))
            std_nat = std_t * jac
            uncertainties[name] = std_nat
            ci_low = max(float(theta_nat - 1.96 * std_nat), bounds[i][0])
            ci_high = min(float(theta_nat + 1.96 * std_nat), bounds[i][1])
            credible_intervals[name] = (ci_low, ci_high)

        uncertainties["noise_std"] = noise_std_posterior

        # Compute fitted curve
        fitted_curve = self.model.evaluate(list(fitted_params), aif, time_points)

        # Compute goodness of fit
        gof = compute_goodness_of_fit(
            tissue_curve, fitted_curve, self.model.n_parameters
        )

        # Compute derived parameters
        param_dict = self.model.params_to_dict(list(fitted_params))
        derived = self.model.compute_derived_parameters(list(fitted_params))
        param_dict.update(derived)

        return FitResult(
            parameters=param_dict,
            uncertainties=uncertainties,
            credible_intervals=credible_intervals,
            r_squared=gof["r_squared"],
            residual_sum_squares=gof["rss"],
            aic=gof["aic"],
            bic=gof["bic"],
            model_name=self.model.name,
            n_parameters=self.model.n_parameters,
            n_observations=n_obs,
            converged=converged or step_scale >= 1e-4,
            fitting_method="vi",
            fitted_curve=fitted_curve,
        )

    def fit_mcmc(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        n_samples: int = DEFAULT_MCMC_SAMPLES,
        n_walkers: int | None = None,
        n_burnin: int | None = None,
        sloppy: bool = False,
        weights: npt.NDArray[np.floating] | None = None,
    ) -> FitResult:
        """Fit model using MCMC with ensemble sampler.

        Uses an affine-invariant ensemble sampler (similar to emcee).
        Noise sigma is sampled as a free parameter (log_sigma) with a
        HalfNormal prior, following Gelman et al. (BDA3, Ch. 3).

        The constraint ve + vp <= 1.0 is enforced during sampling for
        models with both ve and vp parameters.

        Args:
            tissue_curve: Measured tissue concentration curve (mM).
            aif: Arterial input function (mM).
            time_points: Time points (minutes).
            n_samples: Number of samples to draw (after burn-in).
            n_walkers: Number of ensemble walkers (default: 2 * n_dims + 2).
            n_burnin: Number of burn-in samples (default: n_samples // 2).
            sloppy: Use fast, low-quality settings for testing.
            weights: Optional per-frame weights for motion-weighted fitting.
                Values in [0, 1] where 1.0 = full weight.

        Returns:
            FitResult with parameters, uncertainties, credible intervals,
            and MCMC diagnostics (R-hat, ESS).
        """
        if sloppy:
            n_samples = 50
        self._validate_inputs(tissue_curve, aif, time_points)

        # Estimate baseline noise for walker initialization and prior scale
        noise_std_baseline = self.noise_std
        if noise_std_baseline is None:
            noise_std_baseline = self._estimate_noise_std(tissue_curve)

        n_model_params = self.model.n_parameters
        # Total sampled dimensions: model params + log_sigma
        n_dims = n_model_params + 1
        bounds = self.model.get_bounds()
        param_names = self.model.get_parameter_names()

        # Set defaults
        if n_walkers is None:
            n_walkers = max(2 * n_dims + 2, 10)
        if n_burnin is None:
            n_burnin = n_samples // 2

        n_total = n_samples + n_burnin

        # Log posterior including sampled sigma
        # State vector: [model_params..., log_sigma]
        def log_prob_with_sigma(
            state: npt.NDArray[np.floating],
        ) -> float:
            model_params = state[:n_model_params]
            log_sigma = state[n_model_params]
            sigma = np.exp(log_sigma)

            # Check model parameter bounds
            if not self.model.validate_params(model_params):
                return float("-inf")

            # Check volume constraint
            if not self._check_volume_constraint(model_params):
                return float("-inf")

            # sigma must be positive (guaranteed by exp) but guard extremes
            if sigma < 1e-15 or sigma > 1e6:
                return float("-inf")

            # Prior on model parameters
            log_p = self._log_prior(model_params)
            if not np.isfinite(log_p):
                return float("-inf")

            # HalfNormal prior on sigma: p(sigma) ∝ exp(-sigma^2 / (2*scale^2))
            # Plus Jacobian for log_sigma: p(log_sigma) = p(sigma) * sigma
            log_p -= (sigma**2) / (2.0 * noise_std_baseline**2)
            log_p += log_sigma  # Jacobian |d(sigma)/d(log_sigma)| = sigma

            # Likelihood with sampled sigma (not profile)
            log_lik = self._log_likelihood_with_sigma(
                model_params, tissue_curve, aif, time_points, sigma,
                weights=weights,
            )

            return log_p + log_lik  # type: ignore[no-any-return]

        # Initialize walkers near MAP estimate
        map_result = self.fit_map(
            tissue_curve, aif, time_points, max_iter=100,
            compute_uncertainties=False,
        )
        map_params_list = self.model.dict_to_params(
            {k: v for k, v in map_result.parameters.items() if k in param_names}
        )
        map_params = np.array(map_params_list, dtype=np.float64)

        # Bounds for sampler: model bounds + log_sigma bounds
        sampler_bounds = bounds + [(-15.0, 15.0)]  # log_sigma

        # Initialize walkers with small perturbations around MAP
        init_positions = np.zeros((n_walkers, n_dims))
        log_sigma_init = np.log(noise_std_baseline)
        for i in range(n_walkers):
            for j in range(n_model_params):
                scale = self.priors[j].std * 0.1
                perturbed = map_params[j] + scale * np.random.randn()
                perturbed = np.clip(perturbed, bounds[j][0], bounds[j][1])
                init_positions[i, j] = perturbed
            # log_sigma: small perturbation around baseline estimate
            init_positions[i, n_model_params] = (
                log_sigma_init + 0.1 * np.random.randn()
            )

            # Apply volume constraint to initialization
            if self._has_volume_constraint:
                init_positions[i, :n_model_params] = self._apply_volume_constraint(
                    init_positions[i, :n_model_params]
                )

        # Run ensemble sampler (shared utility)
        samples = run_ensemble_sampler(
            log_prob_with_sigma, init_positions, n_total, sampler_bounds
        )

        # Discard burn-in
        samples = samples[n_burnin:, :, :]  # (n_samples, n_walkers, n_dims)

        # Flatten chains for analysis
        flat_samples = samples.reshape(-1, n_dims)

        # Separate model params and sigma
        model_samples = flat_samples[:, :n_model_params]
        sigma_samples = np.exp(flat_samples[:, n_model_params])

        # Posterior summaries for model parameters
        posterior_mean = np.mean(model_samples, axis=0)
        posterior_std = np.std(model_samples, axis=0)
        ci_low = np.percentile(model_samples, 2.5, axis=0)
        ci_high = np.percentile(model_samples, 97.5, axis=0)

        parameters: dict[str, float] = {}
        uncertainties: dict[str, float] = {}
        credible_intervals: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(param_names):
            parameters[name] = float(posterior_mean[i])
            uncertainties[name] = float(posterior_std[i])
            credible_intervals[name] = (float(ci_low[i]), float(ci_high[i]))

        # Add sigma posterior summary
        parameters["noise_std"] = float(np.mean(sigma_samples))
        uncertainties["noise_std"] = float(np.std(sigma_samples))
        credible_intervals["noise_std"] = (
            float(np.percentile(sigma_samples, 2.5)),
            float(np.percentile(sigma_samples, 97.5)),
        )

        # Convergence diagnostics (shared utilities: split R-hat, per-walker ESS)
        rhat = compute_split_rhat(samples)
        ess = compute_per_walker_ess(samples)

        rhat_dict: dict[str, float] = {}
        ess_dict: dict[str, int] = {}
        all_names = param_names + ["log_sigma"]
        for i, name in enumerate(all_names):
            rhat_dict[name] = float(rhat[i])
            ess_dict[name] = int(ess[i])

        all_rhat_ok = all(r < QC_RHAT_MAX for r in rhat)
        all_ess_ok = all(e > QC_ESS_MIN for e in ess)
        converged = all_rhat_ok and all_ess_ok
        quality_flag = "good" if converged else "poor_convergence"

        # Compute fitted curve using posterior mean
        fitted_params = posterior_mean
        fitted_curve = self.model.evaluate(list(fitted_params), aif, time_points)

        # Compute goodness of fit
        gof = compute_goodness_of_fit(
            tissue_curve, fitted_curve, self.model.n_parameters
        )

        # Compute derived parameters
        derived = self.model.compute_derived_parameters(list(fitted_params))
        parameters.update(derived)

        return FitResult(
            parameters=parameters,
            uncertainties=uncertainties,
            credible_intervals=credible_intervals,
            r_squared=gof["r_squared"],
            residual_sum_squares=gof["rss"],
            aic=gof["aic"],
            bic=gof["bic"],
            rhat=rhat_dict,
            ess=ess_dict,
            model_name=self.model.name,
            n_parameters=self.model.n_parameters,
            n_observations=len(tissue_curve),
            converged=converged,
            fitting_method="mcmc",
            fitted_curve=fitted_curve,
            quality_flag=quality_flag,
        )


def create_vp_prior_from_dsc(
    vp_value: float,
    relative_uncertainty: float = 0.2,
) -> BayesianPrior:
    """Create an informative vp prior from DSC-derived plasma volume.

    The prior is a Gaussian centered on the DSC-derived vp value with
    standard deviation proportional to the value (relative uncertainty).

    A minimum std of 0.001 prevents degenerate priors when vp is very small.

    Args:
        vp_value: DSC-derived vp value for this voxel.
        relative_uncertainty: Relative uncertainty (fraction). Default 0.2
            means +/- 20% of the vp value.

    Returns:
        BayesianPrior with informative mean and std for vp.
    """
    std = max(vp_value * relative_uncertainty, 0.001)
    # vp uses LOG_LOGIT transform; compute transformed prior stats
    from pmriprep.models.bayesian_utils import transform_prior
    ub = 0.3  # VP_BOUNDS.max default
    t_mean, t_std = transform_prior(
        max(vp_value, 1e-10), std, TransformType.LOG_LOGIT, ub
    )
    return BayesianPrior(
        mean=vp_value,
        std=std,
        transform=TransformType.LOG_LOGIT,
        upper_bound=ub,
        mean_transformed=t_mean,
        std_transformed=t_std,
    )


def fit_pk_model(
    model: PKModel,
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    method: str = "vi",
    mcmc_samples: int = DEFAULT_MCMC_SAMPLES,
    noise_std: float | None = None,
    sloppy: bool = False,
) -> FitResult:
    """Convenience function to fit a PK model using specified Bayesian method.

    Args:
        model: PK model to fit.
        tissue_curve: Measured tissue concentration (mM).
        aif: Arterial input function (mM).
        time_points: Time points (minutes).
        method: Fitting method: 'map', 'vi', or 'mcmc'.
        mcmc_samples: Number of MCMC samples (for method='mcmc').
        noise_std: Optional noise standard deviation.
        sloppy: Use fast, low-quality settings for testing.

    Returns:
        FitResult with fitted parameters and uncertainties.

    Raises:
        ValueError: If method is not recognized.
    """
    fitter = BayesianPKFitter(model, noise_std=noise_std)

    if method.lower() == "map":
        return fitter.fit_map(
            tissue_curve, aif, time_points, sloppy=sloppy,
        )
    elif method.lower() == "vi":
        return fitter.fit_vi(
            tissue_curve, aif, time_points, sloppy=sloppy,
        )
    elif method.lower() == "mcmc":
        return fitter.fit_mcmc(
            tissue_curve, aif, time_points, n_samples=mcmc_samples,
            sloppy=sloppy,
        )
    else:
        raise ValueError(
            f"Unknown fitting method: {method}. Use 'map', 'vi', or 'mcmc'."
        )
