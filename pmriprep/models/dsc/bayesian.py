"""Bayesian inference for DSC perfusion models.

Provides Bayesian fitting methods for DSC models including:
- MAP (Maximum A Posteriori) estimation with Laplace approximation
- VI (Variational Inference) with mean-field approximation
- MCMC (Markov Chain Monte Carlo) with ensemble sampling

The BayesianDSCFitter class works with any model implementing the DSCModel
interface, including VascularModel and CPIModel.

References:
    Mouridsen K, Friston K, Hajnal N, et al. (2006).
    "Bayesian estimation of cerebral perfusion using a physiological
    model of microvasculature." NeuroImage 33(2):570-579.

    Woolrich MW, Jbabdi S, Patenaude B, et al. (2009).
    "Bayesian analysis of neuroimaging data in FSL."
    NeuroImage 45(1):S173-S186.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy import optimize  # type: ignore[import-untyped]

from pmriprep.models.bayesian_utils import (
    BayesianPrior,
    build_priors_from_params,
    compute_goodness_of_fit,
    compute_log_prior,
    compute_per_walker_ess,
    compute_split_rhat,
    run_ensemble_sampler,
    validate_bayesian_inputs,
)
from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
)
from pmriprep.utils.dsc_constants import (
    MAP_MAX_ITER,
    MCMC_SAMPLES,
    QC_ESS_MIN,
    QC_RHAT_MAX,
)

# Re-export BayesianPrior for backward compatibility
__all__ = ["BayesianDSCFitter", "BayesianPrior", "fit_dsc_model"]


class BayesianDSCFitter:
    """Bayesian inference for DSC perfusion models.

    Provides three fitting methods:
    - fit_map(): MAP estimation with Laplace approximation for uncertainties
    - fit_vi(): Variational inference with mean-field Gaussian approximation
    - fit_mcmc(): Full MCMC sampling with ensemble sampler

    The fitter constructs Gaussian priors from the model's parameter definitions,
    using the typical_range to determine prior mean and standard deviation.

    Example:
        >>> model = VascularModel()
        >>> fitter = BayesianDSCFitter(model)
        >>> result = fitter.fit_map(concentration, aif, time_points)
        >>> print(result.parameters)
        >>> print(result.uncertainties)

    Attributes:
        model: DSC model to fit (VascularModel, CPIModel, etc.)
        priors: Gaussian priors for each parameter
        noise_std: Assumed noise standard deviation (estimated if not provided)
    """

    def __init__(
        self,
        model: DSCModel,
        noise_std: float | None = None,
    ) -> None:
        """Initialize the Bayesian fitter.

        Args:
            model: DSC model implementing the DSCModel interface.
            noise_std: Optional noise standard deviation. If None, will be
                estimated from the data during fitting.
        """
        self.model = model
        self.noise_std = noise_std

        # Build priors from model parameter definitions
        self.priors = build_priors_from_params(model.get_parameter_definitions())

    @staticmethod
    def _validate_inputs(
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> None:
        """Validate input arrays for NaN/Inf values and length consistency.

        Args:
            concentration: Measured concentration time course.
            aif: Arterial input function.
            time_points: Time points (seconds).

        Raises:
            ValueError: If inputs contain NaN/Inf or lengths don't match.
        """
        validate_bayesian_inputs(
            ("concentration", concentration),
            ("aif", aif),
            ("time_points", time_points),
        )

    def _estimate_noise_std(
        self,
        concentration: npt.NDArray[np.floating],
        baseline_frames: int = 5,
    ) -> float:
        """Estimate noise standard deviation from baseline variability.

        Uses the standard deviation of the first few baseline frames
        as an estimate of measurement noise.

        Args:
            concentration: Measured concentration time course.
            baseline_frames: Number of baseline frames to use.

        Returns:
            Estimated noise standard deviation.
        """
        if len(concentration) < baseline_frames:
            baseline_frames = max(2, len(concentration) // 2)

        baseline = concentration[:baseline_frames]
        noise_std = float(np.std(baseline))

        # Minimum noise level for numerical stability
        # perf-plus uses EPSILON_SIGNAL * 100 = 1e-7; the 1e-10 floor
        # (EPSILON_CONCENTRATION) makes the likelihood overwhelm the prior
        return max(noise_std, 1e-7)

    def _log_prior(
        self, params: npt.NDArray[np.floating]
    ) -> float:
        """Compute log prior probability.

        Uses truncated normal priors matching perf-plus: simple Gaussian in
        natural parameter space, hard-clipped to parameter bounds.  The
        log_transform flag is NOT applied here (perf-plus ignores it in its
        prior computation).

        Args:
            params: Parameter values.

        Returns:
            Log prior probability (unnormalized).
        """
        return compute_log_prior(params, self.priors)

    def _log_likelihood(
        self,
        params: npt.NDArray[np.floating],
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        noise_std: float,
        weights: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute log likelihood.

        Assumes independent Gaussian noise on each time point, optionally
        weighted by per-frame motion weights.

        Args:
            params: Parameter values.
            concentration: Measured concentration.
            aif: Arterial input function.
            time_points: Time points (seconds).
            noise_std: Noise standard deviation.
            weights: Optional per-frame weights (1.0 = full, <1.0 = downweighted).

        Returns:
            Log likelihood (unnormalized).
        """
        # Compute model prediction
        try:
            predicted = self.model.tissue_curve(params, aif, time_points)
        except (ValueError, RuntimeError):
            return float("-inf")

        # Residuals
        residuals = concentration - predicted

        # Gaussian log likelihood (weighted if motion weights provided)
        n = len(concentration)
        log_lik = -0.5 * n * np.log(2 * np.pi * noise_std**2)
        if weights is not None:
            log_lik -= 0.5 * np.sum(weights * residuals**2) / (noise_std**2)
        else:
            log_lik -= 0.5 * np.sum(residuals**2) / (noise_std**2)

        return float(log_lik)

    def _log_posterior(
        self,
        params: npt.NDArray[np.floating],
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        noise_std: float,
        weights: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute log posterior probability.

        log p(θ|y) ∝ log p(y|θ) + log p(θ)

        Args:
            params: Parameter values.
            concentration: Measured concentration.
            aif: Arterial input function.
            time_points: Time points (seconds).
            noise_std: Noise standard deviation.
            weights: Optional per-frame motion weights.

        Returns:
            Log posterior probability (unnormalized).
        """
        # Check bounds
        if not self.model.validate_params(params):
            return float("-inf")

        log_prior = self._log_prior(params)
        if not np.isfinite(log_prior):
            return float("-inf")

        log_lik = self._log_likelihood(
            params, concentration, aif, time_points, noise_std,
            weights=weights,
        )

        return log_prior + log_lik

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

                if all(np.isfinite([f_pp, f_pm, f_mp, f_mm])):
                    h_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * eps[i] * eps[j])
                    if abs(h_ij) < 1e8:
                        hessian[i, j] = h_ij
                    elif i == j:
                        prior_std = self.priors[i].std
                        if prior_std > 0:
                            hessian[i, j] = 1.0 / (prior_std**2)
                        else:
                            hessian[i, j] = 1.0
                else:
                    # Fall back to prior curvature for invalid regions
                    if i == j:
                        prior_std = self.priors[i].std
                        if prior_std > 0:
                            hessian[i, j] = 1.0 / (prior_std**2)
                        else:
                            hessian[i, j] = 1.0

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

    def _compute_gof(
        self,
        concentration: npt.NDArray[np.floating],
        fitted_tissue: npt.NDArray[np.floating],
    ) -> dict[str, float]:
        """Compute goodness-of-fit metrics.

        Args:
            concentration: Measured concentration.
            fitted_tissue: Model-predicted concentration.

        Returns:
            Dictionary with r_squared, rss, aic, bic.
        """
        return compute_goodness_of_fit(
            concentration, fitted_tissue, self.model.n_parameters
        )

    def fit_map(
        self,
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        max_iter: int = MAP_MAX_ITER,
        compute_uncertainties: bool = True,
        weights: npt.NDArray[np.floating] | None = None,
        sloppy: bool = False,
    ) -> DSCFitResult:
        """Fit model using MAP estimation with Laplace approximation.

        Maximum A Posteriori finds the mode of the posterior distribution.
        Uncertainties are estimated using the Laplace approximation
        (inverse Hessian at the mode).

        Args:
            concentration: Measured tissue concentration curve.
            aif: Arterial input function.
            time_points: Time points (seconds).
            max_iter: Maximum optimizer iterations.
            compute_uncertainties: Whether to compute Laplace uncertainties.
                Set to False to skip the expensive Hessian computation.
            weights: Optional per-frame motion weights (1.0 = full weight).
            sloppy: Use fast, low-quality settings for testing.

        Returns:
            DSCFitResult with parameters, uncertainties, and credible intervals.
        """
        if sloppy:
            max_iter = 100
            compute_uncertainties = False
        self._validate_inputs(concentration, aif, time_points)

        # Estimate noise if not provided
        noise_std = self.noise_std
        if noise_std is None:
            noise_std = self._estimate_noise_std(concentration)

        # Initial guesses for multi-start optimization
        x0_prior = np.array(self.model.get_initial_guess(), dtype=np.float64)
        bounds = self.model.get_bounds()
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        # Data-driven initial guess from concentration curve
        x0_data = x0_prior.copy()
        try:
            peak_conc = float(np.max(concentration))
            if peak_conc > 0:
                # Scale CBF-like parameters based on peak concentration
                param_names = self.model.get_parameter_names()
                for i, name in enumerate(param_names):
                    if "cbf" in name.lower():
                        x0_data[i] = min(peak_conc * 10.0, upper[i])
        except (ValueError, IndexError):
            pass

        # Clip start points to bounds
        x0_prior = np.clip(x0_prior, lower, upper)
        x0_data = np.clip(x0_data, lower, upper)

        # Objective: negative log posterior
        def neg_log_posterior(params: npt.NDArray[np.floating]) -> float:
            val = -self._log_posterior(
                params, concentration, aif, time_points, noise_std,
                weights=weights,
            )
            return min(val, 1e20)

        # Multi-start optimization (matching DCE approach)
        best_fun = float("inf")
        converged = False
        fitted_params = x0_prior

        for x0 in [x0_prior, x0_data]:
            try:
                result = optimize.minimize(
                    neg_log_posterior,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": max_iter,
                        "disp": False,
                        "ftol": 1e-3 if sloppy else 1e-6,
                    },
                )
                if result.fun < best_fun:
                    best_fun = result.fun
                    converged = result.success and result.fun < 1e20
                    fitted_params = np.array(result.x, dtype=np.float64)
            except (ValueError, RuntimeError):
                continue

        # Clip to bounds
        fitted_params = np.array(
            self.model.clip_params(fitted_params), dtype=np.float64
        )

        # Compute Laplace approximation uncertainties
        if compute_uncertainties:
            uncertainties, credible_intervals = self._compute_laplace_uncertainties(
                fitted_params, neg_log_posterior
            )
        else:
            uncertainties = {}
            credible_intervals = {}

        # Compute fitted curves
        fitted_residue = self.model.residue_function(fitted_params, time_points)
        fitted_tissue = self.model.tissue_curve(fitted_params, aif, time_points)

        gof = self._compute_gof(concentration, fitted_tissue)

        return DSCFitResult(
            parameters=self.model.params_to_dict(fitted_params),
            uncertainties=uncertainties,
            credible_intervals=credible_intervals,
            r_squared=gof["r_squared"],
            residual_sum_squares=gof["rss"],
            aic=gof["aic"],
            bic=gof["bic"],
            model_name=self.model.name,
            n_parameters=self.model.n_parameters,
            n_observations=len(concentration),
            converged=converged,
            fitting_method="map",
            fitted_residue=fitted_residue,
            fitted_tissue_curve=fitted_tissue,
        )

    def fit_vi(
        self,
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        n_iterations: int = 20,
        noise_prior: tuple[float, float] | None = None,
        weights: npt.NDArray[np.floating] | None = None,
        sloppy: bool = False,
    ) -> DSCFitResult:
        """Fit model using FABBER-style linearized Variational Bayes.

        Implements the linearized VB algorithm from Chappell et al. (2009,
        IEEE TMI 28(1):94-104). At each iteration the forward model is
        linearized around the current posterior mean, enabling analytic
        Gaussian posterior updates. Noise precision is treated as a conjugate
        Gamma random variable.

        This is ~125-500x faster than the previous black-box VI
        (90-180 forward model calls vs ~45,000).

        Args:
            concentration: Measured tissue concentration curve.
            aif: Arterial input function.
            time_points: Time points (seconds).
            n_iterations: Maximum VB iterations (default 20).
            noise_prior: Optional (c0, s0) Gamma prior on noise precision
                tau = 1/sigma^2. If None, uses non-informative prior.
                If noise_std was provided to the constructor, uses an
                informative prior centered on that value.
            weights: Optional per-frame motion weights (1.0 = full weight).
            sloppy: Use fast, low-quality settings for testing.

        Returns:
            DSCFitResult with parameters, uncertainties, and credible intervals.

        References:
            Chappell MA, Groves AR, Whitcher B, Woolrich MW (2009).
            "Variational Bayesian Inference for a Nonlinear Forward Model."
            IEEE Trans Med Imaging 28(1):94-104.
        """
        if sloppy:
            n_iterations = 5
        self._validate_inputs(concentration, aif, time_points)

        n_params = self.model.n_parameters
        n_obs = len(concentration)
        bounds = self.model.get_bounds()
        param_names = self.model.get_parameter_names()
        y = np.asarray(concentration, dtype=np.float64)

        # Prior precision matrix (diagonal, from prior stds)
        prior_means = np.array([p.mean for p in self.priors], dtype=np.float64)
        prior_stds = np.array([p.std for p in self.priors], dtype=np.float64)
        prior_prec = np.diag(1.0 / (prior_stds**2))  # Λ₀

        # Noise precision prior: Gamma(c0, s0) with E[tau] = c0 * s0
        # Following FABBER convention: Gamma parameterized so E[tau] = c/s
        if noise_prior is not None:
            c0, s0 = noise_prior
        elif self.noise_std is not None:
            # Informative prior centered on provided noise_std
            tau_prior = 1.0 / (self.noise_std**2)
            c0 = 10.0
            s0 = c0 / tau_prior  # so E[tau] = c0/s0 = tau_prior
        else:
            # Non-informative prior
            c0 = 1e-3
            s0 = 1e3

        # Initialize posterior mean at prior mean, clipped to bounds
        m = prior_means.copy()
        for i in range(n_params):
            m[i] = np.clip(m[i], bounds[i][0], bounds[i][1])

        # Initialize noise posterior
        c = c0
        s = s0
        tau_exp = c / s  # E[tau] = c/s

        # Best free energy tracking for trial mode
        best_free_energy = float("-inf")
        best_m = m.copy()
        best_cov = np.diag(prior_stds**2)
        best_c = c
        best_s = s
        n_trials = 0
        max_trials = 10

        converged = False

        for iteration in range(n_iterations):
            # 1. Evaluate forward model at current mean
            try:
                g_m = self.model.tissue_curve(m, aif, time_points)
            except (ValueError, RuntimeError):
                break
            g_m = np.asarray(g_m, dtype=np.float64)

            # 2. Compute Jacobian via central finite differences
            J = np.zeros((n_obs, n_params), dtype=np.float64)
            for j in range(n_params):
                delta = max(abs(m[j]) * 1e-5, 1e-10)
                m_plus = m.copy()
                m_minus = m.copy()
                m_plus[j] += delta
                m_minus[j] -= delta
                # Clip perturbations to bounds
                m_plus[j] = np.clip(m_plus[j], bounds[j][0], bounds[j][1])
                m_minus[j] = np.clip(m_minus[j], bounds[j][0], bounds[j][1])
                actual_delta = m_plus[j] - m_minus[j]
                if actual_delta < 1e-15:
                    continue
                try:
                    g_plus = self.model.tissue_curve(m_plus, aif, time_points)
                    g_minus = self.model.tissue_curve(m_minus, aif, time_points)
                except (ValueError, RuntimeError):
                    continue
                J[:, j] = (np.asarray(g_plus) - np.asarray(g_minus)) / actual_delta

            # Residual
            k_res = y - g_m  # k = y - g(m)

            # 3. Update posterior precision: Λ = Λ₀ + <τ> J^T W J
            if weights is not None:
                JtWJ = J.T @ (weights[:, np.newaxis] * J)
            else:
                JtWJ = J.T @ J
            post_prec = prior_prec + tau_exp * JtWJ

            # 4. Update posterior covariance
            try:
                post_cov = np.linalg.inv(post_prec)
            except np.linalg.LinAlgError:
                # Regularize if singular
                post_prec += 1e-6 * np.eye(n_params)
                post_cov = np.linalg.inv(post_prec)

            # 5. Update posterior mean:
            # m_new = Σ (Λ₀ m₀ + <τ> J^T W (k + J m))
            if weights is not None:
                rhs = prior_prec @ prior_means + tau_exp * J.T @ (weights * (k_res + J @ m))
            else:
                rhs = prior_prec @ prior_means + tau_exp * J.T @ (k_res + J @ m)
            m_new = post_cov @ rhs

            # Clip to bounds
            for i in range(n_params):
                m_new[i] = np.clip(m_new[i], bounds[i][0], bounds[i][1])

            # 6. Update noise precision posterior: Gamma(c, s)
            # c = c₀ + N/2
            c = c0 + n_obs / 2.0
            # Residual at new mean
            try:
                g_new = self.model.tissue_curve(m_new, aif, time_points)
            except (ValueError, RuntimeError):
                g_new = g_m
            k_new = y - np.asarray(g_new, dtype=np.float64)
            # 1/s = 1/s₀ + 0.5 * (k^T W k + tr(Σ J^T W J))
            trace_term = float(np.trace(post_cov @ JtWJ))
            if weights is not None:
                weighted_kk = float(k_new @ (weights * k_new))
            else:
                weighted_kk = float(k_new @ k_new)
            s = 1.0 / (1.0 / s0 + 0.5 * (weighted_kk + trace_term))
            tau_exp = c * s  # E[tau] for Gamma with this parameterization

            # 7. Compute free energy (analytic ELBO) for convergence
            # F = -0.5 * [<tau> * (<k^T k> + tr(Σ J^T J))
            #     + tr(Λ₀ Σ) + (m - m₀)^T Λ₀ (m - m₀)
            #     - log|Σ| + log|Λ₀⁻¹|
            #     - N log(<tau>) + ...]  (up to constants)
            dm = m_new - prior_means
            sign_cov, logdet_cov = np.linalg.slogdet(post_cov)
            sign_prior, logdet_prior_cov = np.linalg.slogdet(
                np.diag(prior_stds**2)
            )

            free_energy = (
                0.5 * n_obs * (np.log(tau_exp) - np.log(2 * np.pi))
                - 0.5 * tau_exp * (weighted_kk + trace_term)
                - 0.5 * float(dm @ prior_prec @ dm)
                - 0.5 * float(np.trace(prior_prec @ post_cov))
                + 0.5 * logdet_cov * sign_cov
                - 0.5 * logdet_prior_cov * sign_prior
                + 0.5 * n_params  # entropy constant
            )

            # Trial mode: revert if free energy decreased
            if free_energy > best_free_energy:
                best_free_energy = free_energy
                best_m = m_new.copy()
                best_cov = post_cov.copy()
                best_c = c
                best_s = s
                n_trials = 0
            else:
                n_trials += 1
                if n_trials >= max_trials:
                    break

            # Convergence check
            if iteration > 0:
                param_change = float(np.max(np.abs(m_new - m)))
                if param_change < 1e-6:
                    converged = True
                    m = m_new
                    break

            m = m_new

        # Use best parameters found
        fitted_params = best_m
        fitted_stds = np.sqrt(np.maximum(np.diag(best_cov), 0.0))
        noise_std_posterior = 1.0 / np.sqrt(max(best_c * best_s, 1e-300))

        # Build uncertainties and credible intervals
        uncertainties: dict[str, float] = {}
        credible_intervals: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(param_names):
            std = float(fitted_stds[i])
            uncertainties[name] = std
            ci_low = max(float(fitted_params[i] - 1.96 * std), bounds[i][0])
            ci_high = min(float(fitted_params[i] + 1.96 * std), bounds[i][1])
            credible_intervals[name] = (ci_low, ci_high)

        # Add noise std to uncertainties for diagnostics
        uncertainties["noise_std"] = noise_std_posterior

        # Compute fitted curves
        fitted_residue = self.model.residue_function(fitted_params, time_points)
        fitted_tissue = self.model.tissue_curve(fitted_params, aif, time_points)

        gof = self._compute_gof(concentration, fitted_tissue)

        return DSCFitResult(
            parameters=self.model.params_to_dict(fitted_params),
            uncertainties=uncertainties,
            credible_intervals=credible_intervals,
            r_squared=gof["r_squared"],
            residual_sum_squares=gof["rss"],
            aic=gof["aic"],
            bic=gof["bic"],
            model_name=self.model.name,
            n_parameters=self.model.n_parameters,
            n_observations=n_obs,
            converged=converged or n_trials < max_trials,
            fitting_method="vi",
            fitted_residue=fitted_residue,
            fitted_tissue_curve=fitted_tissue,
        )

    def fit_mcmc(
        self,
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        n_samples: int = MCMC_SAMPLES,
        n_walkers: int | None = None,
        n_burnin: int | None = None,
        weights: npt.NDArray[np.floating] | None = None,
        sloppy: bool = False,
    ) -> DSCFitResult:
        """Fit model using MCMC with ensemble sampler.

        Uses an affine-invariant ensemble sampler (similar to emcee).
        Noise sigma is sampled as a free parameter (log_sigma) with a
        HalfNormal prior, following Gelman et al. (BDA3, Ch. 3).

        Args:
            concentration: Measured tissue concentration curve.
            aif: Arterial input function.
            time_points: Time points (seconds).
            n_samples: Number of samples to draw (after burn-in).
            n_walkers: Number of ensemble walkers (default: 2 * n_dims + 2).
            n_burnin: Number of burn-in samples (default: n_samples // 2).
            weights: Optional per-frame motion weights (1.0 = full weight).
            sloppy: Use fast, low-quality settings for testing.

        Returns:
            DSCFitResult with parameters, uncertainties, credible intervals,
            and MCMC diagnostics (R-hat, ESS).
        """
        if sloppy:
            n_samples = 50
        self._validate_inputs(concentration, aif, time_points)

        # Estimate baseline noise for walker initialization and prior scale
        noise_std_baseline = self.noise_std
        if noise_std_baseline is None:
            noise_std_baseline = self._estimate_noise_std(concentration)

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

            # Likelihood with sampled sigma
            log_lik = self._log_likelihood(
                model_params, concentration, aif, time_points, sigma,
                weights=weights,
            )

            return log_p + log_lik  # type: ignore[no-any-return]

        # Initialize walkers near MAP estimate
        map_result = self.fit_map(
            concentration, aif, time_points, max_iter=100,
            compute_uncertainties=False, weights=weights,
        )
        map_params_list = self.model.dict_to_params(map_result.parameters)
        map_params = np.array(map_params_list, dtype=np.float64)

        # Bounds for sampler: model bounds + log_sigma bounds
        sampler_bounds = bounds + [(-15.0, 15.0)]  # log_sigma

        # Initialize walkers with small perturbations
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

        # Convergence diagnostics (shared utilities)
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

        # Fitted curves using posterior mean of model params
        fitted_residue = self.model.residue_function(posterior_mean, time_points)
        fitted_tissue = self.model.tissue_curve(posterior_mean, aif, time_points)

        gof = self._compute_gof(concentration, fitted_tissue)

        return DSCFitResult(
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
            n_observations=len(concentration),
            converged=converged,
            fitting_method="mcmc",
            fitted_residue=fitted_residue,
            fitted_tissue_curve=fitted_tissue,
        )


def fit_dsc_model(
    model: DSCModel,
    concentration: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    method: str = "map",
    mcmc_samples: int = MCMC_SAMPLES,
    noise_std: float | None = None,
    compute_uncertainties: bool = True,
    weights: npt.NDArray[np.floating] | None = None,
    sloppy: bool = False,
) -> DSCFitResult:
    """Convenience function to fit a DSC model using specified Bayesian method.

    Args:
        model: DSC model to fit.
        concentration: Measured tissue concentration.
        aif: Arterial input function.
        time_points: Time points (seconds).
        method: Fitting method: 'map', 'vi', or 'mcmc'.
        mcmc_samples: Number of MCMC samples (for method='mcmc').
        noise_std: Optional noise standard deviation.
        compute_uncertainties: Whether to compute Laplace uncertainties for MAP.
        weights: Optional per-frame motion weights (1.0 = full weight).
        sloppy: Use fast, low-quality settings for testing.

    Returns:
        DSCFitResult with fitted parameters and uncertainties.

    Raises:
        ValueError: If method is not recognized.
    """
    fitter = BayesianDSCFitter(model, noise_std=noise_std)

    if method.lower() == "map":
        return fitter.fit_map(
            concentration, aif, time_points,
            compute_uncertainties=compute_uncertainties,
            weights=weights,
            sloppy=sloppy,
        )
    elif method.lower() == "vi":
        return fitter.fit_vi(
            concentration, aif, time_points, weights=weights,
            sloppy=sloppy,
        )
    elif method.lower() == "mcmc":
        return fitter.fit_mcmc(
            concentration, aif, time_points, n_samples=mcmc_samples,
            weights=weights,
            sloppy=sloppy,
        )
    else:
        raise ValueError(
            f"Unknown fitting method: {method}. Use 'map', 'vi', or 'mcmc'."
        )
