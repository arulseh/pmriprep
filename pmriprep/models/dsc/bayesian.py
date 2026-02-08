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

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy import optimize  # type: ignore[import-untyped]

from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
)
from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
    MAP_MAX_ITER,
    MCMC_SAMPLES,
    QC_ESS_MIN,
    QC_RHAT_MAX,
)


@dataclass
class BayesianPrior:
    """Gaussian prior specification for a model parameter.

    Attributes:
        mean: Prior mean.
        std: Prior standard deviation.
        log_transform: Whether parameter is fit in log space.
    """

    mean: float
    std: float
    log_transform: bool = False


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
        noise_std: Optional[float] = None,
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
        self.priors = self._build_priors(model.get_parameter_definitions())

    def _build_priors(
        self, param_defs: Sequence[DSCModelParameter]
    ) -> list[BayesianPrior]:
        """Build Gaussian priors from parameter definitions.

        Prior mean is the center of typical_range.
        Prior std is (typical_range[1] - typical_range[0]) / 4.0,
        which places 95% of prior mass within the typical range.

        Args:
            param_defs: List of DSCModelParameter objects.

        Returns:
            List of BayesianPrior objects.
        """
        priors = []
        for param in param_defs:
            prior = BayesianPrior(
                mean=param.get_prior_mean(),
                std=param.get_prior_sigma(),
                log_transform=param.log_transform,
            )
            priors.append(prior)
        return priors

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
        return max(noise_std, EPSILON_CONCENTRATION)

    def _log_prior(
        self, params: npt.NDArray[np.floating]
    ) -> float:
        """Compute log prior probability.

        Uses Gaussian priors for each parameter, optionally in log space.

        Args:
            params: Parameter values.

        Returns:
            Log prior probability (unnormalized).
        """
        log_p = 0.0

        for i, (p, prior) in enumerate(zip(params, self.priors)):
            # Transform to log space if needed
            if prior.log_transform:
                if p <= 0:
                    return float("-inf")
                p_trans = np.log(p)
                mean_trans = np.log(max(prior.mean, EPSILON_CONCENTRATION))
                # Jacobian term for log transform
                log_p -= np.log(p)
            else:
                p_trans = p
                mean_trans = prior.mean

            # Gaussian prior
            if prior.std > 0:
                log_p -= 0.5 * ((p_trans - mean_trans) / prior.std) ** 2

        return log_p

    def _log_likelihood(
        self,
        params: npt.NDArray[np.floating],
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        noise_std: float,
    ) -> float:
        """Compute log likelihood.

        Assumes independent Gaussian noise on each time point.

        Args:
            params: Parameter values.
            concentration: Measured concentration.
            aif: Arterial input function.
            time_points: Time points (seconds).
            noise_std: Noise standard deviation.

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

        # Gaussian log likelihood
        n = len(concentration)
        log_lik = -0.5 * n * np.log(2 * np.pi * noise_std**2)
        log_lik -= 0.5 * np.sum(residuals**2) / (noise_std**2)

        return float(log_lik)

    def _log_posterior(
        self,
        params: npt.NDArray[np.floating],
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        noise_std: float,
    ) -> float:
        """Compute log posterior probability.

        log p(θ|y) ∝ log p(y|θ) + log p(θ)

        Args:
            params: Parameter values.
            concentration: Measured concentration.
            aif: Arterial input function.
            time_points: Time points (seconds).
            noise_std: Noise standard deviation.

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
            params, concentration, aif, time_points, noise_std
        )

        return log_prior + log_lik

    def _compute_hessian(
        self,
        params: npt.NDArray[np.floating],
        objective: Callable[[npt.NDArray[np.floating]], float],
        epsilon: float = 1e-5,
    ) -> npt.NDArray[np.floating]:
        """Compute Hessian matrix using finite differences.

        Uses central differences for second derivatives.

        Args:
            params: Parameter values at which to evaluate Hessian.
            objective: Negative log posterior function.
            epsilon: Step size for finite differences.

        Returns:
            Hessian matrix (n_params x n_params).
        """
        n = len(params)
        hessian = np.zeros((n, n), dtype=np.float64)

        f0 = objective(params)

        for i in range(n):
            for j in range(i, n):
                # Perturb parameters
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()

                params_pp[i] += epsilon
                params_pp[j] += epsilon
                params_pm[i] += epsilon
                params_pm[j] -= epsilon
                params_mp[i] -= epsilon
                params_mp[j] += epsilon
                params_mm[i] -= epsilon
                params_mm[j] -= epsilon

                # Second derivative
                with np.errstate(invalid="ignore"):
                    f_pp = objective(params_pp)
                    f_pm = objective(params_pm)
                    f_mp = objective(params_mp)
                    f_mm = objective(params_mm)

                    if all(np.isfinite([f_pp, f_pm, f_mp, f_mm])):
                        hessian[i, j] = (
                            (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                        )
                    else:
                        hessian[i, j] = 0.0
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
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        max_iter: int = MAP_MAX_ITER,
        compute_uncertainties: bool = True,
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

        Returns:
            DSCFitResult with parameters, uncertainties, and credible intervals.
        """
        # Estimate noise if not provided
        noise_std = self.noise_std
        if noise_std is None:
            noise_std = self._estimate_noise_std(concentration)

        # Initial guess
        x0 = np.array(self.model.get_initial_guess(), dtype=np.float64)
        bounds = self.model.get_bounds()

        # Objective: negative log posterior
        def neg_log_posterior(params: npt.NDArray[np.floating]) -> float:
            return -self._log_posterior(
                params, concentration, aif, time_points, noise_std
            )

        # Optimize
        try:
            result = optimize.minimize(
                neg_log_posterior,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iter, "disp": False},
            )
            converged = result.success
            fitted_params = np.array(result.x, dtype=np.float64)

            # Clip to bounds
            fitted_params = np.array(
                self.model.clip_params(fitted_params), dtype=np.float64
            )

        except (ValueError, RuntimeError):
            converged = False
            fitted_params = x0

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

        # Compute goodness of fit
        residuals = concentration - fitted_tissue
        rss = float(np.sum(residuals**2))
        ss_tot = float(np.sum((concentration - np.mean(concentration))**2))
        r_squared = 1.0 - (rss / max(ss_tot, EPSILON_CONCENTRATION))

        # Information criteria
        n = len(concentration)
        k = self.model.n_parameters
        if rss > 0:
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)
        else:
            aic = float("-inf")
            bic = float("-inf")

        return DSCFitResult(
            parameters=self.model.params_to_dict(fitted_params),
            uncertainties=uncertainties,
            credible_intervals=credible_intervals,
            r_squared=r_squared,
            residual_sum_squares=rss,
            aic=float(aic),
            bic=float(bic),
            model_name=self.model.name,
            n_parameters=k,
            n_observations=n,
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
        n_iterations: int = 500,
        learning_rate: float = 0.01,
    ) -> DSCFitResult:
        """Fit model using variational inference.

        Uses mean-field variational inference with Gaussian approximation.
        Optimizes the ELBO (Evidence Lower Bound) to approximate the posterior.

        Args:
            concentration: Measured tissue concentration curve.
            aif: Arterial input function.
            time_points: Time points (seconds).
            n_iterations: Number of VI optimization iterations.
            learning_rate: Learning rate for gradient updates.

        Returns:
            DSCFitResult with parameters, uncertainties, and credible intervals.
        """
        # Estimate noise if not provided
        noise_std = self.noise_std
        if noise_std is None:
            noise_std = self._estimate_noise_std(concentration)

        n_params = self.model.n_parameters
        bounds = self.model.get_bounds()
        param_names = self.model.get_parameter_names()

        # Initialize variational parameters (mean and log-std for each param)
        # Start from prior means
        v_means = np.array([p.mean for p in self.priors], dtype=np.float64)
        v_log_stds = np.log(np.array([p.std for p in self.priors], dtype=np.float64))

        # ELBO estimation using Monte Carlo samples
        n_samples = 10  # Samples per iteration for gradient estimation

        def sample_params(
            means: npt.NDArray[np.floating],
            log_stds: npt.NDArray[np.floating],
        ) -> npt.NDArray[np.floating]:
            """Sample from variational distribution."""
            stds = np.exp(log_stds)
            samples = means + stds * np.random.randn(n_params)
            # Clip to bounds
            for i in range(n_params):
                samples[i] = np.clip(samples[i], bounds[i][0], bounds[i][1])
            return samples

        def estimate_elbo(
            means: npt.NDArray[np.floating],
            log_stds: npt.NDArray[np.floating],
        ) -> float:
            """Estimate ELBO using Monte Carlo."""
            elbo_sum = 0.0
            stds = np.exp(log_stds)

            for _ in range(n_samples):
                params = sample_params(means, log_stds)

                # Log posterior
                log_post = self._log_posterior(
                    params, concentration, aif, time_points, noise_std
                )

                # Log variational density (Gaussian)
                log_q = -0.5 * np.sum(
                    ((params - means) / stds) ** 2 + 2 * log_stds + np.log(2 * np.pi)
                )

                elbo_sum += log_post - log_q

            return float(elbo_sum / n_samples)

        # Optimize using gradient ascent with numerical gradients
        best_elbo = float("-inf")
        best_means = v_means.copy()
        best_log_stds = v_log_stds.copy()

        epsilon = 1e-4  # For numerical gradients

        for iteration in range(n_iterations):
            # Compute gradients numerically
            grad_means = np.zeros(n_params)
            grad_log_stds = np.zeros(n_params)

            elbo_current = estimate_elbo(v_means, v_log_stds)

            for i in range(n_params):
                # Gradient w.r.t. mean
                means_plus = v_means.copy()
                means_plus[i] += epsilon
                grad_means[i] = (
                    estimate_elbo(means_plus, v_log_stds) - elbo_current
                ) / epsilon

                # Gradient w.r.t. log_std
                log_stds_plus = v_log_stds.copy()
                log_stds_plus[i] += epsilon
                grad_log_stds[i] = (
                    estimate_elbo(v_means, log_stds_plus) - elbo_current
                ) / epsilon

            # Update parameters (gradient ascent)
            v_means += learning_rate * grad_means
            v_log_stds += learning_rate * grad_log_stds

            # Clip means to bounds
            for i in range(n_params):
                v_means[i] = np.clip(v_means[i], bounds[i][0], bounds[i][1])

            # Clip log_stds to reasonable range
            v_log_stds = np.clip(v_log_stds, -10.0, 5.0)

            # Track best
            if elbo_current > best_elbo:
                best_elbo = elbo_current
                best_means = v_means.copy()
                best_log_stds = v_log_stds.copy()

        # Final results
        fitted_params = best_means
        fitted_stds = np.exp(best_log_stds)

        # Build uncertainties and credible intervals
        uncertainties: dict[str, float] = {}
        credible_intervals: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(param_names):
            std = float(fitted_stds[i])
            uncertainties[name] = std

            # 95% CI
            ci_low = max(float(fitted_params[i] - 1.96 * std), bounds[i][0])
            ci_high = min(float(fitted_params[i] + 1.96 * std), bounds[i][1])
            credible_intervals[name] = (ci_low, ci_high)

        # Compute fitted curves
        fitted_residue = self.model.residue_function(fitted_params, time_points)
        fitted_tissue = self.model.tissue_curve(fitted_params, aif, time_points)

        # Compute goodness of fit
        residuals = concentration - fitted_tissue
        rss = float(np.sum(residuals**2))
        ss_tot = float(np.sum((concentration - np.mean(concentration))**2))
        r_squared = 1.0 - (rss / max(ss_tot, EPSILON_CONCENTRATION))

        # Information criteria
        n = len(concentration)
        k = self.model.n_parameters
        if rss > 0:
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)
        else:
            aic = float("-inf")
            bic = float("-inf")

        return DSCFitResult(
            parameters=self.model.params_to_dict(fitted_params),
            uncertainties=uncertainties,
            credible_intervals=credible_intervals,
            r_squared=r_squared,
            residual_sum_squares=rss,
            aic=float(aic),
            bic=float(bic),
            model_name=self.model.name,
            n_parameters=k,
            n_observations=n,
            converged=True,  # VI always "converges" (may not reach global optimum)
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
        n_walkers: Optional[int] = None,
        n_burnin: Optional[int] = None,
    ) -> DSCFitResult:
        """Fit model using MCMC with ensemble sampler.

        Uses an affine-invariant ensemble sampler (similar to emcee).
        Provides full posterior samples and convergence diagnostics.

        Args:
            concentration: Measured tissue concentration curve.
            aif: Arterial input function.
            time_points: Time points (seconds).
            n_samples: Number of samples to draw (after burn-in).
            n_walkers: Number of ensemble walkers (default: 2 * n_params + 2).
            n_burnin: Number of burn-in samples (default: n_samples // 2).

        Returns:
            DSCFitResult with parameters, uncertainties, credible intervals,
            and MCMC diagnostics (R-hat, ESS).
        """
        # Estimate noise if not provided
        noise_std = self.noise_std
        if noise_std is None:
            noise_std = self._estimate_noise_std(concentration)

        n_params = self.model.n_parameters
        bounds = self.model.get_bounds()
        param_names = self.model.get_parameter_names()

        # Set defaults
        if n_walkers is None:
            n_walkers = max(2 * n_params + 2, 10)
        if n_burnin is None:
            n_burnin = n_samples // 2

        # Total iterations
        n_total = n_samples + n_burnin

        # Log posterior function
        def log_prob(params: npt.NDArray[np.floating]) -> float:
            return self._log_posterior(
                params, concentration, aif, time_points, noise_std
            )

        # Initialize walkers near MAP estimate
        # First, get a rough MAP estimate
        map_result = self.fit_map(
            concentration, aif, time_points, max_iter=100,
            compute_uncertainties=False,
        )
        map_params = self.model.dict_to_params(map_result.parameters)
        map_params = np.array(map_params, dtype=np.float64)

        # Initialize walkers with small perturbations around MAP
        init_positions = np.zeros((n_walkers, n_params))
        for i in range(n_walkers):
            for j in range(n_params):
                # Small Gaussian perturbation
                scale = self.priors[j].std * 0.1
                perturbed = map_params[j] + scale * np.random.randn()
                # Clip to bounds
                perturbed = np.clip(perturbed, bounds[j][0], bounds[j][1])
                init_positions[i, j] = perturbed

        # Run ensemble sampler (simple implementation)
        # Using stretch move similar to Goodman & Weare (2010)
        samples = self._run_ensemble_sampler(
            log_prob, init_positions, n_total, bounds
        )

        # Discard burn-in
        samples = samples[n_burnin:, :, :]  # Shape: (n_samples, n_walkers, n_params)

        # Flatten chains for analysis
        flat_samples = samples.reshape(-1, n_params)

        # Compute posterior mean and std
        posterior_mean = np.mean(flat_samples, axis=0)
        posterior_std = np.std(flat_samples, axis=0)

        # Compute credible intervals (2.5th and 97.5th percentiles)
        ci_low = np.percentile(flat_samples, 2.5, axis=0)
        ci_high = np.percentile(flat_samples, 97.5, axis=0)

        # Build dictionaries
        parameters: dict[str, float] = {}
        uncertainties: dict[str, float] = {}
        credible_intervals: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(param_names):
            parameters[name] = float(posterior_mean[i])
            uncertainties[name] = float(posterior_std[i])
            credible_intervals[name] = (float(ci_low[i]), float(ci_high[i]))

        # Compute convergence diagnostics
        rhat = self._compute_rhat(samples)
        ess = self._compute_ess(samples)

        # Build diagnostic dicts
        rhat_dict: dict[str, float] = {}
        ess_dict: dict[str, int] = {}
        for i, name in enumerate(param_names):
            rhat_dict[name] = float(rhat[i])
            ess_dict[name] = int(ess[i])

        # Check convergence
        all_rhat_ok = all(r < QC_RHAT_MAX for r in rhat)
        all_ess_ok = all(e > QC_ESS_MIN for e in ess)
        converged = all_rhat_ok and all_ess_ok

        # Compute fitted curves using posterior mean
        fitted_params = posterior_mean
        fitted_residue = self.model.residue_function(fitted_params, time_points)
        fitted_tissue = self.model.tissue_curve(fitted_params, aif, time_points)

        # Compute goodness of fit
        residuals = concentration - fitted_tissue
        rss = float(np.sum(residuals**2))
        ss_tot = float(np.sum((concentration - np.mean(concentration))**2))
        r_squared = 1.0 - (rss / max(ss_tot, EPSILON_CONCENTRATION))

        # Information criteria (using posterior mean)
        n = len(concentration)
        k = self.model.n_parameters
        if rss > 0:
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)
        else:
            aic = float("-inf")
            bic = float("-inf")

        return DSCFitResult(
            parameters=parameters,
            uncertainties=uncertainties,
            credible_intervals=credible_intervals,
            r_squared=r_squared,
            residual_sum_squares=rss,
            aic=float(aic),
            bic=float(bic),
            rhat=rhat_dict,
            ess=ess_dict,
            model_name=self.model.name,
            n_parameters=k,
            n_observations=n,
            converged=converged,
            fitting_method="mcmc",
            fitted_residue=fitted_residue,
            fitted_tissue_curve=fitted_tissue,
        )

    def _run_ensemble_sampler(
        self,
        log_prob: Callable[[npt.NDArray[np.floating]], float],
        init_positions: npt.NDArray[np.floating],
        n_iterations: int,
        bounds: list[tuple[float, float]],
        a: float = 2.0,
    ) -> npt.NDArray[np.floating]:
        """Run affine-invariant ensemble sampler.

        Implements the stretch move from Goodman & Weare (2010).

        Args:
            log_prob: Log posterior function.
            init_positions: Initial walker positions (n_walkers x n_params).
            n_iterations: Number of iterations.
            bounds: Parameter bounds.
            a: Stretch move scale parameter (default 2.0).

        Returns:
            Samples array (n_iterations x n_walkers x n_params).
        """
        n_walkers, n_params = init_positions.shape

        # Storage for samples
        samples = np.zeros((n_iterations, n_walkers, n_params))
        samples[0] = init_positions

        # Current positions and log probs
        current_pos = init_positions.copy()
        current_lp = np.array([log_prob(current_pos[j]) for j in range(n_walkers)])

        for iteration in range(1, n_iterations):
            # Update each walker
            for j in range(n_walkers):
                # Select random walker from complementary ensemble
                others = np.concatenate([np.arange(j), np.arange(j + 1, n_walkers)])
                k = np.random.choice(others)

                # Stretch move
                # z ~ 1/sqrt(z) on [1/a, a]
                u = np.random.rand()
                z = ((a - 1.0) * u + 1.0) ** 2 / a

                # Proposal
                proposal = current_pos[k] + z * (current_pos[j] - current_pos[k])

                # Check bounds
                in_bounds = True
                for i in range(n_params):
                    if proposal[i] < bounds[i][0] or proposal[i] > bounds[i][1]:
                        in_bounds = False
                        break

                if in_bounds:
                    # Compute log prob of proposal
                    proposal_lp = log_prob(proposal)

                    # Acceptance probability
                    log_ratio = (n_params - 1) * np.log(z) + proposal_lp - current_lp[j]

                    # Accept/reject
                    if np.log(np.random.rand()) < log_ratio:
                        current_pos[j] = proposal
                        current_lp[j] = proposal_lp

            samples[iteration] = current_pos

        return samples

    def _compute_rhat(
        self, samples: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Compute Gelman-Rubin R-hat diagnostic.

        R-hat compares within-chain and between-chain variance.
        Values < 1.1 indicate convergence.

        Args:
            samples: MCMC samples (n_samples x n_walkers x n_params).

        Returns:
            R-hat values for each parameter.
        """
        n_samples, n_walkers, n_params = samples.shape

        rhat = np.zeros(n_params)

        for p in range(n_params):
            # Treat each walker as a separate chain
            chains = samples[:, :, p].T  # (n_walkers, n_samples)

            n = n_samples
            m = n_walkers

            # Chain means
            chain_means = np.mean(chains, axis=1)  # (n_walkers,)

            # Overall mean
            overall_mean = np.mean(chain_means)

            # Between-chain variance
            B = n * np.var(chain_means, ddof=1)

            # Within-chain variance
            W = np.mean(np.var(chains, axis=1, ddof=1))

            # Posterior variance estimate
            var_plus = ((n - 1) / n) * W + (1 / n) * B

            # R-hat
            if W > 0:
                rhat[p] = np.sqrt(var_plus / W)
            else:
                rhat[p] = 1.0

        return rhat

    def _compute_ess(
        self, samples: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.integer]:
        """Compute effective sample size (ESS).

        ESS accounts for autocorrelation in the chains.

        Args:
            samples: MCMC samples (n_samples x n_walkers x n_params).

        Returns:
            ESS values for each parameter.
        """
        n_samples, n_walkers, n_params = samples.shape

        ess = np.zeros(n_params, dtype=np.int64)

        for p in range(n_params):
            # Flatten all chains for this parameter
            flat = samples[:, :, p].flatten()

            # Compute autocorrelation
            n = len(flat)
            if n < 2:
                ess[p] = n
                continue

            # Normalize
            x = flat - np.mean(flat)
            var = np.var(x)

            if var < EPSILON_CONCENTRATION:
                ess[p] = n
                continue

            # Compute autocorrelation up to max lag
            max_lag = min(n // 2, 1000)
            acf = np.zeros(max_lag)

            for k in range(max_lag):
                acf[k] = np.mean(x[: n - k] * x[k:]) / var

            # Integrated autocorrelation time using initial positive sequence
            # Sum pairs of consecutive autocorrelations until they go negative
            tau_int = 1.0
            for k in range(1, max_lag - 1, 2):
                pair_sum = acf[k] + acf[k + 1]
                if pair_sum > 0:
                    tau_int += pair_sum
                else:
                    break

            # ESS = n / (2 * tau_int)
            ess[p] = max(1, int(n / (2 * tau_int)))

        return ess


def fit_dsc_model(
    model: DSCModel,
    concentration: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    method: str = "map",
    mcmc_samples: int = MCMC_SAMPLES,
    noise_std: Optional[float] = None,
    compute_uncertainties: bool = True,
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
        )
    elif method.lower() == "vi":
        return fitter.fit_vi(concentration, aif, time_points)
    elif method.lower() == "mcmc":
        return fitter.fit_mcmc(
            concentration, aif, time_points, n_samples=mcmc_samples
        )
    else:
        raise ValueError(
            f"Unknown fitting method: {method}. Use 'map', 'vi', or 'mcmc'."
        )
