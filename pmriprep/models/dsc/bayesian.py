"""Bayesian inference for DSC perfusion models.

Provides Bayesian fitting methods for DSC models including:
- MAP (Maximum A Posteriori) estimation with Laplace approximation
- VI (Variational Inference) with mean-field approximation
- MCMC (Markov Chain Monte Carlo) with ensemble sampling

The BayesianDSCFitter class works with any model implementing the DSCModel
interface, including VascularModel and CPIModel.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy import optimize  # type: ignore[import-untyped]

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

    Examples
    --------
        >>> model = VascularModel()  # doctest: +SKIP
        >>> fitter = BayesianDSCFitter(model)  # doctest: +SKIP
        >>> result = fitter.fit_map(concentration, aif, time_points)  # doctest: +SKIP
        >>> print(result.parameters)  # doctest: +SKIP
        >>> print(result.uncertainties)  # doctest: +SKIP

    Attributes
    ----------
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

        Parameters
        ----------
        model : DSCModel
            DSC model implementing the DSCModel interface.
        noise_std : float or None, optional
            Optional noise standard deviation. If None, will be
            estimated from the data during fitting.
        """
        self.model = model
        self.noise_std = noise_std

        # Build priors from fitting-space parameter definitions.
        # For models without reparameterization (e.g. CPI), this is identical
        # to get_parameter_definitions(). For VascularModel, this uses the
        # (CBF, MTT, alpha, delay) parameterization instead of (CBF, alpha,
        # beta, delay).
        self.priors = build_priors_from_params(model.get_fitting_parameter_definitions())

    @staticmethod
    def _validate_inputs(
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> None:
        """Validate input arrays for NaN/Inf values and length consistency.

        Parameters
        ----------
        concentration : npt.NDArray[np.floating]
            Measured concentration time course.
        aif : npt.NDArray[np.floating]
            Arterial input function.
        time_points : npt.NDArray[np.floating]
            Time points (seconds).

        Raises
        ------
        ValueError
            If inputs contain NaN/Inf or lengths don't match.
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

        Parameters
        ----------
        concentration : npt.NDArray[np.floating]
            Measured concentration time course.
        baseline_frames : int, optional
            Number of baseline frames to use.

        Returns
        -------
        float
            Estimated noise standard deviation.
        """
        if len(concentration) < baseline_frames:
            baseline_frames = max(2, len(concentration) // 2)

        baseline = concentration[:baseline_frames]
        noise_std = float(np.std(baseline))

        # Minimum noise level for numerical stability
        # 1e-7 floor (EPSILON_SIGNAL * 100) prevents the likelihood from
        # overwhelming the prior (1e-10 EPSILON_CONCENTRATION is too small)
        return max(noise_std, 1e-7)

    def _log_prior(
        self, params: npt.NDArray[np.floating]
    ) -> float:
        """Compute log prior probability.

        Uses truncated normal priors: simple Gaussian in natural parameter
        space, hard-clipped to parameter bounds.  Parameter transforms
        (LOG, LOG_LOGIT) are applied in VI/MAP fitting loops, not here.

        Parameters
        ----------
        params : npt.NDArray[np.floating]
            Parameter values.

        Returns
        -------
        float
            Log prior probability (unnormalized).
        """
        lp = compute_log_prior(params, self.priors)
        lp += self.model.compute_monotonicity_prior(params)
        return lp

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

        Parameters
        ----------
        params : npt.NDArray[np.floating]
            Parameter values.
        concentration : npt.NDArray[np.floating]
            Measured concentration.
        aif : npt.NDArray[np.floating]
            Arterial input function.
        time_points : npt.NDArray[np.floating]
            Time points (seconds).
        noise_std : float
            Noise standard deviation.
        weights : npt.NDArray[np.floating] or None, optional
            Optional per-frame weights (1.0 = full, <1.0 = downweighted).

        Returns
        -------
        float
            Log likelihood (unnormalized).
        """
        # Convert fitting-space params to model-space for tissue_curve.
        # For non-reparameterized models this is identity (no-op).
        params_model = self.model.fitting_to_model_params(params)

        # Compute model prediction
        try:
            predicted = self.model.tissue_curve(params_model, aif, time_points)
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

        Parameters
        ----------
        params : npt.NDArray[np.floating]
            Parameter values.
        concentration : npt.NDArray[np.floating]
            Measured concentration.
        aif : npt.NDArray[np.floating]
            Arterial input function.
        time_points : npt.NDArray[np.floating]
            Time points (seconds).
        noise_std : float
            Noise standard deviation.
        weights : npt.NDArray[np.floating] or None, optional
            Optional per-frame motion weights.

        Returns
        -------
        float
            Log posterior probability (unnormalized).
        """
        # Check bounds in fitting space
        fitting_defs = self.model.get_fitting_parameter_definitions()
        if len(params) != len(fitting_defs):
            return float("-inf")
        if not all(d.is_valid(float(p)) for d, p in zip(fitting_defs, params)):
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
        prior_stds: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute Hessian matrix using finite differences.

        Uses central differences for second derivatives with adaptive
        per-parameter step sizes that scale with parameter magnitude.
        When perturbations push parameters into invalid regions (causing
        non-finite objective values), falls back to using prior curvature
        for those entries.

        Parameters
        ----------
        params : npt.NDArray[np.floating]
            Parameter values at which to evaluate Hessian.
        objective : Callable[[npt.NDArray[np.floating]], float]
            Negative log posterior function.
        relative_epsilon : float, optional
            Relative step size for finite differences.
            Actual step for parameter i is max(relative_epsilon,
            relative_epsilon * |params[i]|).
        prior_stds : npt.NDArray[np.floating] or None, optional
            Prior standard deviations expressed in the same coordinates as
            ``params``. Natural fitting-space deviations are used by default.

        Returns
        -------
        npt.NDArray[np.floating]
            Hessian matrix (n_params x n_params).
        """
        n = len(params)
        hessian = np.zeros((n, n), dtype=np.float64)
        fallback_stds = (
            np.asarray(prior_stds, dtype=np.float64)
            if prior_stds is not None
            else np.asarray([prior.std for prior in self.priors], dtype=np.float64)
        )
        if fallback_stds.shape != (n,):
            raise ValueError("prior_stds must have one value per parameter")

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
                        prior_std = float(fallback_stds[i])
                        if prior_std > 0:
                            hessian[i, j] = 1.0 / (prior_std**2)
                        else:
                            hessian[i, j] = 1.0
                else:
                    # Fall back to prior curvature for invalid regions
                    if i == j:
                        prior_std = float(fallback_stds[i])
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
        Parameters and Hessian are in fitting space; names and bounds
        use fitting-space definitions.

        Parameters
        ----------
        params : npt.NDArray[np.floating]
            MAP parameter estimates (fitting space).
        objective : Callable[[npt.NDArray[np.floating]], float]
            Negative log posterior function (fitting space).

        Returns
        -------
        uncertainties : dict
            dict mapping fitting param names to std deviations
        credible_intervals : dict
            dict mapping fitting param names to 95% CIs
        """
        fitting_defs = self.model.get_fitting_parameter_definitions()
        names = [p.name for p in fitting_defs]
        fitting_bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
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

                # Clip to fitting-space bounds
                ci_low = max(ci_low, fitting_bounds[i][0])
                ci_high = min(ci_high, fitting_bounds[i][1])

                credible_intervals[name] = (ci_low, ci_high)

        except np.linalg.LinAlgError:
            # Hessian not invertible; return large uncertainties
            for i, name in enumerate(names):
                # Use prior std as uncertainty estimate
                uncertainties[name] = self.priors[i].std
                ci_low = fitting_bounds[i][0]
                ci_high = fitting_bounds[i][1]
                credible_intervals[name] = (ci_low, ci_high)

        return uncertainties, credible_intervals

    def _compute_gof(
        self,
        concentration: npt.NDArray[np.floating],
        fitted_tissue: npt.NDArray[np.floating],
    ) -> dict[str, float]:
        """Compute goodness-of-fit metrics.

        Parameters
        ----------
        concentration : npt.NDArray[np.floating]
            Measured concentration.
        fitted_tissue : npt.NDArray[np.floating]
            Model-predicted concentration.

        Returns
        -------
        dict
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

        Parameters
        ----------
        concentration : npt.NDArray[np.floating]
            Measured tissue concentration curve.
        aif : npt.NDArray[np.floating]
            Arterial input function.
        time_points : npt.NDArray[np.floating]
            Time points (seconds).
        max_iter : int, optional
            Maximum optimizer iterations.
        compute_uncertainties : bool, optional
            Whether to compute Laplace uncertainties.
            Set to False to skip the expensive Hessian computation.
        weights : npt.NDArray[np.floating] or None, optional
            Optional per-frame motion weights (1.0 = full weight).
        sloppy : bool, optional
            Use fast, low-quality settings for testing.

        Returns
        -------
        DSCFitResult
            DSCFitResult with parameters, uncertainties, and credible intervals.
        """
        if sloppy:
            max_iter = 100
            compute_uncertainties = False
        self._validate_inputs(concentration, aif, time_points)

        # Estimate noise if not provided
        noise_std: float = self.noise_std if self.noise_std is not None else self._estimate_noise_std(concentration)

        # Build transform info from fitting-space priors
        transforms = [p.transform for p in self.priors]
        upper_bounds_t = [p.upper_bound for p in self.priors]
        fitting_defs = self.model.get_fitting_parameter_definitions()
        n_params = len(fitting_defs)

        # Fitting-space bounds
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        fitting_param_names = [p.name for p in fitting_defs]

        # Initial guesses: convert model-space guess to fitting space
        x0_prior_nat = self.model.model_to_fitting_params(
            np.array(self.model.get_initial_guess(), dtype=np.float64)
        )

        # Data-driven initial guess from concentration curve
        x0_data_nat = x0_prior_nat.copy()
        try:
            peak_conc = float(np.max(concentration))
            if peak_conc > 0:
                for i, name in enumerate(fitting_param_names):
                    if "cbf" in name.lower():
                        x0_data_nat[i] = min(peak_conc * 10.0, upper[i])
        except (ValueError, IndexError):
            pass

        x0_prior_nat = np.clip(x0_prior_nat, lower, upper)
        x0_data_nat = np.clip(x0_data_nat, lower, upper)

        # Transform start points to unconstrained space
        x0_prior_t = np.array([
            float(to_transformed(x0_prior_nat[i], transforms[i], upper_bounds_t[i]))
            for i in range(n_params)
        ], dtype=np.float64)
        x0_data_t = np.array([
            float(to_transformed(x0_data_nat[i], transforms[i], upper_bounds_t[i]))
            for i in range(n_params)
        ], dtype=np.float64)

        n_obs = len(concentration)
        n_sigma_iter = 1 if sloppy else 3
        best_fun = float("inf")
        converged = False
        fitted_params_t = x0_prior_t

        for sigma_iter in range(n_sigma_iter):
            # Objective in transformed space
            _noise_std = noise_std

            def neg_log_posterior_t(params_t: npt.NDArray[np.floating]) -> float:
                # Inverse-transform to fitting-space natural values
                params_nat = np.array([
                    float(from_transformed(params_t[i], transforms[i], upper_bounds_t[i]))
                    for i in range(n_params)
                ], dtype=np.float64)
                log_prior_t = compute_log_prior(
                    params_t, self.priors, transformed=True
                )
                if not np.isfinite(log_prior_t):
                    return 1e20
                log_lik = self._log_likelihood(
                    params_nat, concentration, aif, time_points, _noise_std,
                    weights=weights,
                )
                if not np.isfinite(log_lik):
                    return 1e20
                return min(-(log_prior_t + log_lik), 1e20)

            # Multi-start trials. LOG/LOG_LOGIT params are unconstrained
            # in transformed space; IDENTITY params need explicit bounds.
            opt_bounds = [
                (bounds[i][0], bounds[i][1])
                if transforms[i] == TransformType.IDENTITY
                else (None, None)
                for i in range(n_params)
            ]
            starts = [x0_prior_t, x0_data_t]
            if sigma_iter > 0:
                starts.append(fitted_params_t.copy())

            for x0 in starts:
                try:
                    result = optimize.minimize(
                        neg_log_posterior_t,
                        x0,
                        method="L-BFGS-B",
                        bounds=opt_bounds,
                        options={
                            "maxiter": max_iter,
                            "ftol": 1e-3 if sloppy else 1e-6,
                        },
                    )
                    if result.fun < best_fun:
                        best_fun = result.fun
                        converged = result.success and result.fun < 1e20
                        fitted_params_t = np.array(result.x, dtype=np.float64)
                except (ValueError, RuntimeError):
                    continue

            # Inverse-transform for profile likelihood update
            fitted_params = np.array([
                float(from_transformed(fitted_params_t[i], transforms[i], upper_bounds_t[i]))
                for i in range(n_params)
            ], dtype=np.float64)

            # Profile likelihood: update noise_std from residuals
            if sigma_iter < n_sigma_iter - 1:
                try:
                    fitted_model_params = self.model.fitting_to_model_params(fitted_params)
                    fitted_tissue = self.model.tissue_curve(
                        fitted_model_params, aif, time_points
                    )
                    rss = float(np.sum((concentration - fitted_tissue) ** 2))
                    noise_std = max(np.sqrt(rss / n_obs), 1e-7)
                except (ValueError, RuntimeError):
                    pass

        # Final inverse-transform to fitting space (no clipping needed)
        fitted_fitting_params = np.array([
            float(from_transformed(fitted_params_t[i], transforms[i], upper_bounds_t[i]))
            for i in range(n_params)
        ], dtype=np.float64)

        # Convert fitting-space params to model-space for output
        fitted_model_params = self.model.fitting_to_model_params(fitted_fitting_params)

        # Compute Laplace approximation uncertainties from the exact
        # transformed-space objective optimized above, then propagate its
        # marginal standard deviations into natural fitting space.
        if compute_uncertainties:
            try:
                hessian_t = self._compute_hessian(
                    fitted_params_t,
                    neg_log_posterior_t,
                    prior_stds=np.asarray(
                        [prior.std_transformed for prior in self.priors],
                        dtype=np.float64,
                    ),
                )
                if not np.all(np.isfinite(hessian_t)):
                    raise np.linalg.LinAlgError("non-finite Hessian")
                covariance_t = np.linalg.inv(
                    hessian_t + 1e-6 * np.eye(n_params)
                )
                if not np.all(np.isfinite(covariance_t)):
                    raise np.linalg.LinAlgError("non-finite covariance")
                stds_t = np.sqrt(np.maximum(np.diag(covariance_t), 0.0))
                uncertainties = {}
                credible_intervals = {}
                for i, name in enumerate(fitting_param_names):
                    theta_nat = float(fitted_fitting_params[i])
                    jac = float(np.exp(log_jacobian_det(
                        theta_nat, transforms[i], upper_bounds_t[i]
                    )))
                    std_nat = float(stds_t[i]) * jac
                    uncertainties[name] = std_nat
                    credible_intervals[name] = (
                        max(theta_nat - 1.96 * std_nat, bounds[i][0]),
                        min(theta_nat + 1.96 * std_nat, bounds[i][1]),
                    )
            except np.linalg.LinAlgError:
                uncertainties = {
                    name: self.priors[i].std
                    for i, name in enumerate(fitting_param_names)
                }
                credible_intervals = {
                    name: bounds[i]
                    for i, name in enumerate(fitting_param_names)
                }
        else:
            uncertainties = {}
            credible_intervals = {}

        # Compute fitted curves using model-space params
        fitted_residue = self.model.compute_fitted_residue(fitted_model_params, time_points)
        fitted_tissue = self.model.tissue_curve(fitted_model_params, aif, time_points)

        gof = self._compute_gof(concentration, fitted_tissue)

        return DSCFitResult(
            parameters=self.model.params_to_dict(fitted_model_params),
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
        n_iterations: int = 30,
        noise_prior: tuple[float, float] | None = None,
        weights: npt.NDArray[np.floating] | None = None,
        sloppy: bool = False,
    ) -> DSCFitResult:
        """Fit model using FABBER-style linearized Variational Bayes.

        Implements the linearized VB algorithm. At each iteration the forward model is
        linearized around the current posterior mean, enabling analytic
        Gaussian posterior updates. Noise precision is treated as a conjugate
        Gamma random variable.

        This is ~125-500x faster than the previous black-box VI
        (90-180 forward model calls vs ~45,000).

        Parameters
        ----------
        concentration : npt.NDArray[np.floating]
            Measured tissue concentration curve.
        aif : npt.NDArray[np.floating]
            Arterial input function.
        time_points : npt.NDArray[np.floating]
            Time points (seconds).
        n_iterations : int, optional
            Maximum VB iterations (default 20).
        noise_prior : tuple[float, float] or None, optional
            Optional (c0, s0) Gamma prior on noise precision
            tau = 1/sigma^2. If None, uses non-informative prior.
            If noise_std was provided to the constructor, uses an
            informative prior centered on that value.
        weights : npt.NDArray[np.floating] or None, optional
            Optional per-frame motion weights (1.0 = full weight).
        sloppy : bool, optional
            Use fast, low-quality settings for testing.

        Returns
        -------
        DSCFitResult
            DSCFitResult with parameters, uncertainties, and credible intervals.
        """
        if sloppy:
            n_iterations = 8
        self._validate_inputs(concentration, aif, time_points)

        fitting_defs = self.model.get_fitting_parameter_definitions()
        n_params = len(fitting_defs)
        n_obs = len(concentration)
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        param_names = [p.name for p in fitting_defs]
        y = np.asarray(concentration, dtype=np.float64)

        # Build transform info from priors
        transforms = [p.transform for p in self.priors]
        upper_bounds = [p.upper_bound for p in self.priors]

        # Prior precision matrix in transformed space
        prior_means_t = np.array(
            [p.mean_transformed for p in self.priors], dtype=np.float64
        )
        prior_stds_t = np.array(
            [p.std_transformed for p in self.priors], dtype=np.float64
        )
        prior_prec = np.diag(1.0 / (prior_stds_t**2))  # Λ_diag

        mono_prec = self.model.compute_monotonicity_precision()
        mono_eta = self.model.compute_monotonicity_eta_correction()
        if mono_prec is not None:
            prior_prec = prior_prec + mono_prec  # Λ_combined

        # Noise precision prior: Gamma(c0, s0) in shape-scale form.
        # E[tau] = c0 * s0. Update: 1/s_new = 1/s_0 + 0.5 * data_term.
        if noise_prior is not None:
            c0, s0 = noise_prior
        elif self.noise_std is not None:
            # Informative prior centered on provided noise_std
            tau_prior = 1.0 / (self.noise_std**2)
            c0 = 10.0
            s0 = tau_prior / c0  # so E[tau] = c0 * s0 = tau_prior
        else:
            # Non-informative prior
            c0 = 1e-3
            s0 = 1e3

        # Initialize posterior mean from model's initial guess, converted to
        # fitting space. For CPI, get_initial_guess() provides exponential-decay
        # fractions; fitting_to_model is identity so no conversion needed.
        m_nat = self.model.model_to_fitting_params(
            np.array(self.model.get_initial_guess(), dtype=np.float64)
        )
        for i in range(n_params):
            m_nat[i] = np.clip(m_nat[i], bounds[i][0], bounds[i][1])

        # Seed from NLLS if model provides one (e.g. CPI). This prevents
        # tau collapse when the default initial guess is far from truth.
        # NLLS seed is in model space; convert to fitting space.
        nlls_seed = self.model.compute_nlls_seed(y, aif, time_points)
        if nlls_seed is not None:
            m_nat = self.model.model_to_fitting_params(nlls_seed)
            for i in range(n_params):
                m_nat[i] = np.clip(m_nat[i], bounds[i][0], bounds[i][1])

        # Transform initial guess to fitting space
        m = np.array([
            float(to_transformed(m_nat[i], transforms[i], upper_bounds[i]))
            for i in range(n_params)
        ], dtype=np.float64)

        # Initialize noise posterior
        c = c0
        s = s0
        tau_exp = c * s  # E[tau] = shape * scale

        # If NLLS-seeded, initialize tau from residual variance to prevent
        # the Gamma update from collapsing tau on the first iteration.
        if nlls_seed is not None:
            try:
                m_nat_model = self.model.fitting_to_model_params(m_nat)
                g_init = self.model.tissue_curve(m_nat_model, aif, time_points)
                init_var = float(np.sum((y - np.asarray(g_init))**2)) / n_obs
                if init_var > 1e-20:
                    tau_exp = 1.0 / init_var
            except (ValueError, RuntimeError):
                pass

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
            # Inverse-transform m to fitting-space natural params
            m_nat = np.array([
                float(from_transformed(m[k], transforms[k], upper_bounds[k]))
                for k in range(n_params)
            ], dtype=np.float64)

            # Convert fitting-space to model-space for forward model
            m_model = self.model.fitting_to_model_params(m_nat)

            # 1. Evaluate forward model at current mean
            try:
                g_m = self.model.tissue_curve(m_model, aif, time_points)
            except (ValueError, RuntimeError):
                break
            g_m = np.asarray(g_m, dtype=np.float64)

            # 2. Compute Jacobian via central finite differences in
            # transformed space. Perturb m' (transformed), inverse-transform,
            # convert to model space, evaluate forward model.
            J = np.zeros((n_obs, n_params), dtype=np.float64)
            deltas = np.maximum(np.abs(m) * 1e-5, 1e-10)
            for j in range(n_params):
                # Perturb only param j in transformed space
                m_plus_j = m[j] + deltas[j]
                m_minus_j = m[j] - deltas[j]
                # For IDENTITY params, clip perturbations to natural bounds.
                # LOG/LOG_LOGIT are unconstrained in transformed space.
                if transforms[j] == TransformType.IDENTITY:
                    m_plus_j = min(m_plus_j, bounds[j][1])
                    m_minus_j = max(m_minus_j, bounds[j][0])
                # Only inverse-transform the perturbed param (not all n_params)
                p_nat = m_nat.copy()
                n_nat = m_nat.copy()
                p_nat[j] = float(from_transformed(m_plus_j, transforms[j], upper_bounds[j]))
                n_nat[j] = float(from_transformed(m_minus_j, transforms[j], upper_bounds[j]))
                # Convert to model space for tissue_curve
                m_plus_model = self.model.fitting_to_model_params(p_nat)
                m_minus_model = self.model.fitting_to_model_params(n_nat)
                try:
                    g_plus = self.model.tissue_curve(m_plus_model, aif, time_points)
                    g_minus = self.model.tissue_curve(m_minus_model, aif, time_points)
                except (ValueError, RuntimeError):
                    continue
                actual_delta = m_plus_j - m_minus_j
                if actual_delta > 0:
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

            # 5. Update posterior mean (in transformed space):
            # m_new = Σ (Λ₀ m₀' + <τ> J^T W (k + J m'))
            if weights is not None:
                rhs = prior_prec @ prior_means_t + tau_exp * J.T @ (weights * (k_res + J @ m))
            else:
                rhs = prior_prec @ prior_means_t + tau_exp * J.T @ (k_res + J @ m)
            if mono_eta is not None:
                rhs = rhs + mono_eta
            m_new = post_cov @ rhs

            # For LOG/LOG_LOGIT, transformed space is unconstrained (bounds
            # enforced by inverse transform). For IDENTITY, clip to bounds.
            for i in range(n_params):
                if transforms[i] == TransformType.IDENTITY:
                    m_new[i] = np.clip(m_new[i], bounds[i][0], bounds[i][1])

            # 6. Update noise precision posterior: Gamma(c, s)
            # c = c₀ + N/2
            c = c0 + n_obs / 2.0
            # Residual at new mean (inverse-transform, then convert to model space)
            m_new_nat = np.array([
                float(from_transformed(m_new[k], transforms[k], upper_bounds[k]))
                for k in range(n_params)
            ], dtype=np.float64)
            m_new_model = self.model.fitting_to_model_params(m_new_nat)
            try:
                g_new = self.model.tissue_curve(m_new_model, aif, time_points)
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
            dm = m_new - prior_means_t
            sign_cov, logdet_cov = np.linalg.slogdet(post_cov)
            # Use combined prior precision (includes monotonicity if present)
            sign_prior_prec, logdet_prior_prec = np.linalg.slogdet(prior_prec)
            # log|Σ₀| = log|Λ₀⁻¹| = -log|Λ₀|
            logdet_prior_cov = -logdet_prior_prec
            sign_prior = sign_prior_prec

            free_energy = (
                0.5 * n_obs * (np.log(tau_exp) - np.log(2 * np.pi))
                - 0.5 * tau_exp * (weighted_kk + trace_term)
                - 0.5 * float(dm @ prior_prec @ dm)
                - 0.5 * float(np.trace(prior_prec @ post_cov))
                + 0.5 * logdet_cov * sign_cov
                - 0.5 * logdet_prior_cov * sign_prior
                + 0.5 * n_params  # entropy constant
            )

            # Gamma KL: -KL(q(tau) || p(tau)) for shape-scale Gamma
            # (E[tau] = c*s). Only affects convergence/step-size, not
            # parameter updates. Large constant offset with non-informative
            # prior (gammaln(0.001) ~ 999) — only deltas matter.
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
                step_scale = 1.0  # reset on success
            else:
                # Reduce step: interpolate between old and new
                step_scale *= 0.5
                if step_scale < 1e-4:
                    break  # give up
                m_new = m + step_scale * (m_new - m)
                for i in range(n_params):
                    if transforms[i] == TransformType.IDENTITY:
                        m_new[i] = np.clip(m_new[i], bounds[i][0], bounds[i][1])

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

        # Inverse-transform best parameters to fitting-space natural params
        fitted_fitting_params = np.array([
            float(from_transformed(best_m[k], transforms[k], upper_bounds[k]))
            for k in range(n_params)
        ], dtype=np.float64)

        # Convert to model-space for output
        fitted_model_params = self.model.fitting_to_model_params(fitted_fitting_params)

        fitted_stds_t = np.sqrt(np.maximum(np.diag(best_cov), 0.0))
        noise_std_posterior = 1.0 / np.sqrt(max(best_c * best_s, 1e-300))

        # Build uncertainties and credible intervals in fitting space.
        # Back-transform covariance using delta method:
        # std_natural ≈ std_transformed * |dtheta/dtheta'|
        uncertainties: dict[str, float] = {}
        credible_intervals: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(param_names):
            std_t = float(fitted_stds_t[i])
            theta_nat = float(fitted_fitting_params[i])
            jac = float(np.exp(log_jacobian_det(
                theta_nat, transforms[i], upper_bounds[i]
            )))
            std_nat = std_t * jac
            uncertainties[name] = std_nat
            ci_low = max(float(theta_nat - 1.96 * std_nat), bounds[i][0])
            ci_high = min(float(theta_nat + 1.96 * std_nat), bounds[i][1])
            credible_intervals[name] = (ci_low, ci_high)

        # Add noise std to uncertainties for diagnostics
        uncertainties["noise_std"] = noise_std_posterior

        # Compute fitted curves using model-space params
        fitted_residue = self.model.compute_fitted_residue(fitted_model_params, time_points)
        fitted_tissue = self.model.tissue_curve(fitted_model_params, aif, time_points)

        gof = self._compute_gof(concentration, fitted_tissue)

        return DSCFitResult(
            parameters=self.model.params_to_dict(fitted_model_params),
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

        Parameters
        ----------
        concentration : npt.NDArray[np.floating]
            Measured tissue concentration curve.
        aif : npt.NDArray[np.floating]
            Arterial input function.
        time_points : npt.NDArray[np.floating]
            Time points (seconds).
        n_samples : int, optional
            Number of samples to draw (after burn-in).
        n_walkers : int or None, optional
            Number of ensemble walkers (default: 2 * n_dims + 2).
        n_burnin : int or None, optional
            Number of burn-in samples (default: n_samples // 2).
        weights : npt.NDArray[np.floating] or None, optional
            Optional per-frame motion weights (1.0 = full weight).
        sloppy : bool, optional
            Use fast, low-quality settings for testing.

        Returns
        -------
        DSCFitResult
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

        fitting_defs = self.model.get_fitting_parameter_definitions()
        n_fitting_params = len(fitting_defs)
        # Total sampled dimensions: fitting params + log_sigma
        n_dims = n_fitting_params + 1
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        fitting_param_names = [p.name for p in fitting_defs]

        # Set defaults
        if n_walkers is None:
            n_walkers = max(2 * n_dims + 2, 10)
        if n_burnin is None:
            n_burnin = n_samples // 2

        n_total = n_samples + n_burnin

        # Log posterior including sampled sigma
        # State vector: [fitting_params..., log_sigma]
        def log_prob_with_sigma(
            state: npt.NDArray[np.floating],
        ) -> float:
            fitting_params = state[:n_fitting_params]
            log_sigma = state[n_fitting_params]
            sigma = np.exp(log_sigma)

            # Check fitting-space parameter bounds
            if not all(
                d.is_valid(float(p))
                for d, p in zip(fitting_defs, fitting_params)
            ):
                return float("-inf")

            # sigma must be positive (guaranteed by exp) but guard extremes
            if sigma < 1e-15 or sigma > 1e6:
                return float("-inf")

            # Prior on fitting-space parameters
            log_p = self._log_prior(fitting_params)
            if not np.isfinite(log_p):
                return float("-inf")

            # HalfNormal prior on sigma: p(sigma) ∝ exp(-sigma^2 / (2*scale^2))
            # Plus Jacobian for log_sigma: p(log_sigma) = p(sigma) * sigma
            log_p -= (sigma**2) / (2.0 * noise_std_baseline**2)
            log_p += log_sigma  # Jacobian |d(sigma)/d(log_sigma)| = sigma

            # Likelihood with sampled sigma (fitting_to_model conversion
            # happens inside _log_likelihood)
            log_lik = self._log_likelihood(
                fitting_params, concentration, aif, time_points, sigma,
                weights=weights,
            )

            return log_p + log_lik  # type: ignore[no-any-return]

        # Initialize walkers near MAP estimate (already in model space)
        map_result = self.fit_map(
            concentration, aif, time_points, max_iter=100,
            compute_uncertainties=False, weights=weights,
        )
        # Convert MAP result (model space) to fitting space for initialization
        map_model_params = np.array(
            self.model.dict_to_params(map_result.parameters), dtype=np.float64
        )
        map_fitting_params = self.model.model_to_fitting_params(map_model_params)

        # Bounds for sampler: fitting bounds + log_sigma bounds
        sampler_bounds = bounds + [(-15.0, 15.0)]  # log_sigma

        # Initialize walkers with small perturbations in fitting space
        init_positions = np.zeros((n_walkers, n_dims))
        log_sigma_init = np.log(noise_std_baseline)
        for i in range(n_walkers):
            for j in range(n_fitting_params):
                scale = self.priors[j].std * 0.1
                perturbed = map_fitting_params[j] + scale * np.random.randn()
                perturbed = np.clip(perturbed, bounds[j][0], bounds[j][1])
                init_positions[i, j] = perturbed
            # log_sigma: small perturbation around baseline estimate
            init_positions[i, n_fitting_params] = (
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

        # Separate fitting params and sigma
        fitting_samples = flat_samples[:, :n_fitting_params]
        sigma_samples = np.exp(flat_samples[:, n_fitting_params])

        # Convert each sample from fitting space to model space
        model_samples = np.array([
            self.model.fitting_to_model_params(s) for s in fitting_samples
        ])

        # Posterior summaries for model parameters
        posterior_mean_model = np.mean(model_samples, axis=0)
        posterior_std_model = np.std(model_samples, axis=0)
        ci_low = np.percentile(model_samples, 2.5, axis=0)
        ci_high = np.percentile(model_samples, 97.5, axis=0)

        model_param_names = self.model.get_parameter_names()
        parameters: dict[str, float] = {}
        uncertainties: dict[str, float] = {}
        credible_intervals: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(model_param_names):
            parameters[name] = float(posterior_mean_model[i])
            uncertainties[name] = float(posterior_std_model[i])
            credible_intervals[name] = (float(ci_low[i]), float(ci_high[i]))

        # Add fitting-space summaries (e.g. "mtt") alongside model-space keys
        fitting_std = np.std(fitting_samples, axis=0)
        fitting_ci_low = np.percentile(fitting_samples, 2.5, axis=0)
        fitting_ci_high = np.percentile(fitting_samples, 97.5, axis=0)
        for i, name in enumerate(fitting_param_names):
            if name not in uncertainties:
                uncertainties[name] = float(fitting_std[i])
                credible_intervals[name] = (
                    float(fitting_ci_low[i]), float(fitting_ci_high[i]),
                )

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
        all_names = fitting_param_names + ["log_sigma"]
        for i, name in enumerate(all_names):
            rhat_dict[name] = float(rhat[i])
            ess_dict[name] = int(ess[i])

        all_rhat_ok = all(r < QC_RHAT_MAX for r in rhat)
        all_ess_ok = all(e > QC_ESS_MIN for e in ess)
        converged = all_rhat_ok and all_ess_ok

        # Fitted curves using posterior mean of model params
        fitted_residue = self.model.compute_fitted_residue(
            posterior_mean_model, time_points
        )
        fitted_tissue = self.model.tissue_curve(
            posterior_mean_model, aif, time_points
        )

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
    method: str = "vi",
    mcmc_samples: int = MCMC_SAMPLES,
    noise_std: float | None = None,
    compute_uncertainties: bool = True,
    weights: npt.NDArray[np.floating] | None = None,
    sloppy: bool = False,
) -> DSCFitResult:
    """Fit a DSC model using specified Bayesian method.

    Parameters
    ----------
    model : DSCModel
        DSC model to fit.
    concentration : npt.NDArray[np.floating]
        Measured tissue concentration.
    aif : npt.NDArray[np.floating]
        Arterial input function.
    time_points : npt.NDArray[np.floating]
        Time points (seconds).
    method : str, optional
        Fitting method: 'map', 'vi', or 'mcmc'.
    mcmc_samples : int, optional
        Number of MCMC samples (for method='mcmc').
    noise_std : float or None, optional
        Optional noise standard deviation.
    compute_uncertainties : bool, optional
        Whether to compute Laplace uncertainties for MAP.
    weights : npt.NDArray[np.floating] or None, optional
        Optional per-frame motion weights (1.0 = full weight).
    sloppy : bool, optional
        Use fast, low-quality settings for testing.

    Returns
    -------
    DSCFitResult
        DSCFitResult with fitted parameters and uncertainties.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    fitter = BayesianDSCFitter(model, noise_std=noise_std)

    method_lower = method.lower()
    # Spatial VB collapses to plain VI at the per-voxel scalar level
    # (spatial coupling is handled by the batched_dsc_spatial_vi_fit wrapper).
    if method_lower == "svb":
        method_lower = "vi"

    if method_lower == "nlls":
        # FABBER-parity NLLS lives only on the CPI path
        from pmriprep.models.dsc.cpi import CPIModel, _fit_cpi_simple
        if not isinstance(model, CPIModel):
            raise ValueError(
                "method='nlls' is only supported for CPIModel; got "
                f"{type(model).__name__}. Use 'map' or 'vi' for other models."
            )
        result = _fit_cpi_simple(model, concentration, aif, time_points,
                                 weights=weights, sloppy=sloppy)
        if result is None:
            raise RuntimeError("CPI NLLS fit failed to converge")
        return result
    elif method_lower == "map":
        return fitter.fit_map(
            concentration, aif, time_points,
            compute_uncertainties=compute_uncertainties,
            weights=weights,
            sloppy=sloppy,
        )
    elif method_lower == "vi":
        return fitter.fit_vi(
            concentration, aif, time_points, weights=weights,
            sloppy=sloppy,
        )
    elif method_lower == "mcmc":
        return fitter.fit_mcmc(
            concentration, aif, time_points, n_samples=mcmc_samples,
            weights=weights,
            sloppy=sloppy,
        )
    else:
        raise ValueError(
            f"Unknown fitting method: {method}. "
            "Use 'nlls', 'map', 'vi', 'svb', or 'mcmc'."
        )
