"""Bayesian inference for DCE pharmacokinetic models.

Provides Bayesian fitting methods for DCE models including:
- MAP (Maximum A Posteriori) estimation with Laplace approximation
- VI (Variational Inference) with FABBER-style linearized VB
- MCMC (Markov Chain Monte Carlo) with ensemble sampling

The BayesianPKFitter class works with any model implementing the PKModel
interface, including ToftsModel, ExtendedToftsModel, PatlakModel, and
TwoCompartmentExchangeModel.

The shared constraint ve + vp <= VP_VE_TOTAL_MAX is enforced during sampling
for models that have both ve and vp parameters.
"""

from typing import Callable, Mapping

import numpy as np
import numpy.typing as npt
from scipy import optimize  # type: ignore[import-untyped]
from scipy.special import digamma, gammaln  # type: ignore[import-untyped]

from pmriprep.models.bayesian_utils import (
    BayesianPrior,
    TransformType,
    build_priors_from_params,
    compute_goodness_of_fit,
    compute_log_prior,
    compute_ensemble_ess,
    from_transformed,
    log_jacobian_det,
    run_ensemble_sampler,
    to_transformed,
    transformed_gaussian_interval,
    validate_bayesian_inputs,
)
from pmriprep.models.dce.base import (
    FitResult,
    PKModel,
)
from pmriprep.utils.dce_constants import VP_BOUNDS, VP_VE_TOTAL_MAX

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

#: Absolute local free-energy change for VI objective convergence (nats).
VI_FREE_ENERGY_TOL: float = 0.01

#: Threshold for independently generated R-hat diagnostics.
QC_RHAT_MAX: float = 1.1

#: Minimum acceptable effective sample size
QC_ESS_MIN: int = 100

#: Physiological noise floor for DCE concentration (~0.001 mM at tSNR=50).
#: Prevents profile likelihood sigma from collapsing to near-zero when
#: the model fits data very well, which would make log-likelihood overwhelm
#: the prior and reduce MAP to unconstrained MLE.
EPSILON_SIGMA_CONCENTRATION: float = 0.001


def _bounded_vi_mean(
    precision: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    unconstrained: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Minimize the local Gaussian mean objective within transformed bounds.

    Parameters
    ----------
    precision : numpy.ndarray
        Positive-definite local posterior precision matrix.
    rhs : numpy.ndarray
        Linear term in the quadratic mean objective.
    unconstrained : numpy.ndarray
        Unconstrained mean update.
    lower, upper : numpy.ndarray
        Bounds in the same coordinates as the mean and precision.

    Returns
    -------
    numpy.ndarray
        Bound-constrained mean, with all free coordinates reoptimized.

    Raises
    ------
    RuntimeError
        If the constrained solve fails or returns a nonfinite mean.
    """
    if np.all(np.isfinite(unconstrained)) and np.all(
        (unconstrained >= lower) & (unconstrained <= upper)
    ):
        return unconstrained
    factor = np.linalg.cholesky(precision)
    target = np.linalg.solve(factor, rhs)
    result = optimize.lsq_linear(
        factor.T, target, bounds=(lower, upper), method="bvls", tol=1e-10,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Bound-constrained VI mean update failed")
    return np.asarray(result.x, dtype=np.float64)


def _vi_noise_and_free_energy(
    mean: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    gram: npt.NDArray[np.float64],
    weighted_rss: float,
    prior_mean: npt.NDArray[np.float64],
    prior_precision: npt.NDArray[np.float64],
    noise_prior: tuple[float, float],
    n_eff: float,
) -> tuple[float, float, float]:
    """Evaluate a VI trial's noise posterior and local free-energy score.

    Parameters
    ----------
    mean, covariance : numpy.ndarray
        Trial Gaussian parameters in transformed space.
    gram : numpy.ndarray
        Weighted Jacobian Gram matrix evaluated at the trial mean.
    weighted_rss : float
        Weighted residual sum of squares evaluated at the trial mean.
    prior_mean, prior_precision : numpy.ndarray
        Gaussian parameter prior in transformed space.
    noise_prior : tuple of float
        Gamma shape and scale of the noise precision prior.
    n_eff : float
        Effective number of observations.

    Returns
    -------
    c, s, free_energy : float
        Trial Gamma shape, scale, and approximate local free-energy score.
    """
    c0, s0 = noise_prior
    c = c0 + n_eff / 2.0
    trace_term = float(np.trace(covariance @ gram))
    s = 1.0 / (1.0 / s0 + 0.5 * (weighted_rss + trace_term))
    tau_exp = c * s
    dm = mean - prior_mean
    sign_cov, logdet_cov = np.linalg.slogdet(covariance)
    sign_prior, logdet_prior_prec = np.linalg.slogdet(prior_precision)
    free_energy = (
        0.5 * n_eff * (np.log(tau_exp) - np.log(2 * np.pi))
        - 0.5 * tau_exp * (weighted_rss + trace_term)
        - 0.5 * float(dm @ prior_precision @ dm)
        - 0.5 * float(np.trace(prior_precision @ covariance))
        + 0.5 * logdet_cov * sign_cov
        + 0.5 * logdet_prior_prec * sign_prior
        + 0.5 * mean.size
    )
    gamma_kl = -(
        (c - c0) * digamma(c) - gammaln(c) + gammaln(c0)
        + c0 * (np.log(s0) - np.log(s)) + c * (s / s0 - 1.0)
    )
    return float(c), float(s), float(free_energy + gamma_kl)


class BayesianPKFitter:
    """Bayesian inference for DCE pharmacokinetic models.

    Provides three fitting methods:
    - fit_map(): MAP estimation with Laplace approximation for uncertainties
    - fit_vi(): FABBER-style linearized Variational Bayes
    - fit_mcmc(): Full MCMC sampling with ensemble sampler

    The fitter constructs Gaussian priors from the model's parameter definitions,
    using the typical_range to determine prior mean and standard deviation.

    For models with both ve and vp parameters (Extended Tofts, 2CXM), the
    shared joint-volume constraint is enforced during sampling.

    Attributes
    ----------
    model : PKModel
        PK model to fit (ToftsModel, ExtendedToftsModel, etc.)
    priors : list of BayesianPrior
        Gaussian priors for each parameter
    noise_std : float or None
        Assumed noise standard deviation (estimated if not provided)

    Examples
    --------
    >>> model = ExtendedToftsModel()  # doctest: +SKIP
    >>> fitter = BayesianPKFitter(model)  # doctest: +SKIP
    >>> result = fitter.fit_map(tissue_curve, aif, time_points)  # doctest: +SKIP
    >>> print(result.parameters)  # doctest: +SKIP
    >>> print(result.uncertainties)  # doctest: +SKIP
    """

    def __init__(
        self,
        model: PKModel,
        noise_std: float | None = None,
        vp_prior_override: BayesianPrior | None = None,
    ) -> None:
        """Initialize the Bayesian fitter.

        Parameters
        ----------
        model : PKModel
            PK model implementing the PKModel interface.
        noise_std : float or None, optional
            Optional noise standard deviation. If None, will be
            estimated from the data during fitting.
        vp_prior_override : BayesianPrior or None, optional
            Optional informative prior for vp parameter,
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

        Parameters
        ----------
        tissue_curve : numpy.ndarray
            Measured tissue concentration time course.
        aif : numpy.ndarray
            Arterial input function.
        time_points : numpy.ndarray
            Time points (minutes).

        Raises
        ------
        ValueError
            If inputs contain NaN/Inf or lengths don't match.
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

        Parameters
        ----------
        tissue_curve : numpy.ndarray
            Measured tissue concentration time course.
        baseline_frames : int, optional
            Number of baseline frames to use.

        Returns
        -------
        float
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

        Parameters
        ----------
        params : numpy.ndarray
            Parameter values.

        Returns
        -------
        float
            Log prior probability (unnormalized).
        """
        return compute_log_prior(params, self.priors)

    def _check_volume_constraint(
        self, params: npt.NDArray[np.floating]
    ) -> bool:
        """Check whether the shared joint-volume constraint is satisfied.

        Parameters
        ----------
        params : numpy.ndarray
            Parameter values.

        Returns
        -------
        bool
            True if constraint is satisfied (or not applicable).
        """
        if not self._has_volume_constraint:
            return True

        ve = params[self._ve_index]
        vp = params[self._vp_index]

        return bool((ve + vp) <= VP_VE_TOTAL_MAX + 1e-12)

    def _apply_volume_constraint(
        self, params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Apply the ve + vp <= VP_VE_TOTAL_MAX constraint.

        If the sum exceeds the constraint, scales both ve and vp
        proportionally while maintaining their ratio.

        Parameters
        ----------
        params : numpy.ndarray
            Parameter values.

        Returns
        -------
        numpy.ndarray
            Constrained parameter values.
        """
        if not self._has_volume_constraint:
            return params

        params = params.copy()
        ve = params[self._ve_index]
        vp = params[self._vp_index]

        total_volume = ve + vp
        if total_volume > VP_VE_TOTAL_MAX:
            scale_factor = VP_VE_TOTAL_MAX / total_volume
            params[self._ve_index] = ve * scale_factor
            params[self._vp_index] = vp * scale_factor

        return params

    def _log_likelihood(
        self,
        params: npt.NDArray[np.floating],
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
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

        Parameters
        ----------
        params : numpy.ndarray
            Parameter values.
        tissue_curve : numpy.ndarray
            Measured tissue concentration.
        aif : numpy.ndarray
            Arterial input function.
        time_points : numpy.ndarray
            Time points (minutes).
        weights : numpy.ndarray or None, optional
            Optional per-frame weights in [0, 1]. If provided,
            inflates noise variance for high-FD frames.

        Returns
        -------
        float
            Log likelihood (unnormalized).

        Notes
        -----
        With profile likelihood, the log-likelihood reduces to
        -n * log(std(residuals)) + const, equivalent to penalized least
        squares with the Gaussian prior providing L2 regularization.
        Laplace approximation uncertainties are approximate under profile
        likelihood -- use for relative voxel comparison, not calibrated CIs.
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

        Parameters
        ----------
        params : numpy.ndarray
            Parameter values.
        tissue_curve : numpy.ndarray
            Measured tissue concentration.
        aif : numpy.ndarray
            Arterial input function.
        time_points : numpy.ndarray
            Time points (minutes).
        sigma : float
            Noise standard deviation.
        weights : numpy.ndarray or None, optional
            Optional per-frame weights in [0, 1]. If provided,
            uses weighted Gaussian log-likelihood with effective
            sample size n_eff = sum(weights).

        Returns
        -------
        float
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
        params : numpy.ndarray
            Parameter values at which to evaluate Hessian.
        objective : callable
            Negative log posterior function.
        relative_epsilon : float, optional
            Relative step size for finite differences.
            Actual step for parameter i is max(relative_epsilon,
            relative_epsilon * |params[i]|).
        prior_stds : numpy.ndarray or None, optional
            Prior standard deviations expressed in the same coordinates as
            ``params``. Natural-space prior deviations are used by default.

        Returns
        -------
        numpy.ndarray
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
                    # Caveat: at very low permeability (NAWM, Ktrans~0.001
                    # with σ~0.001 mM and N≈100), the diagonal can sit close
                    # to 1e8; uncertainties for those voxels are dominated
                    # by the prior fallback. Acceptable: NAWM is a reference
                    # region, not a clinical-decision target.
                    if abs(h_ij) < 1e8:
                        hessian[i, j] = h_ij
                    elif i == j:
                        prior_std = float(fallback_stds[i])
                        if prior_std > 0:
                            hessian[i, j] = 1.0 / (prior_std**2)
                        else:
                            hessian[i, j] = 1.0
                    # Off-diagonal remains 0 (assumes independent parameters)
                else:
                    # Fall back to prior curvature when perturbations hit invalid region
                    # Use 1/prior_variance as diagonal Hessian approximation
                    if i == j:
                        prior_std = float(fallback_stds[i])
                        if prior_std > 0:
                            hessian[i, j] = 1.0 / (prior_std**2)
                        else:
                            hessian[i, j] = 1.0  # Default regularization
                    # Off-diagonal remains 0 (assumes independent parameters)

                hessian[j, i] = hessian[i, j]

        return hessian

    def fit_map(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        max_iter: int = DEFAULT_MAP_MAX_ITER,
        compute_uncertainties: bool = True,
        weights: npt.NDArray[np.floating] | None = None,
        sloppy: bool = False,
        initial_parameters: Mapping[str, float] | None = None,
    ) -> FitResult:
        """Fit model using MAP estimation with Laplace approximation.

        Maximum A Posteriori finds the mode of the posterior distribution.
        Uncertainties are estimated using the Laplace approximation
        (inverse Hessian at the mode).

        Parameters
        ----------
        tissue_curve : numpy.ndarray
            Measured tissue concentration curve (mM).
        aif : numpy.ndarray
            Arterial input function (mM).
        time_points : numpy.ndarray
            Time points (minutes).
        max_iter : int, optional
            Maximum optimizer iterations.
        compute_uncertainties : bool, optional
            Whether to compute Laplace uncertainties.
            Set to False to skip the expensive Hessian computation.
        weights : numpy.ndarray or None, optional
            Optional per-frame weights for motion-weighted fitting.
            Values in [0, 1] where 1.0 = full weight.
        sloppy : bool, optional
            Use fast, low-quality settings for testing.
        initial_parameters : mapping of str to float or None, optional
            Natural-space parameters used for the NLLS warm-start candidate.
            When omitted or incomplete, the scalar NLLS fit is computed.

        Returns
        -------
        FitResult
            FitResult with parameters, uncertainties, and credible intervals.
        """
        if sloppy:
            max_iter = 100
            compute_uncertainties = False
        self._validate_inputs(tissue_curve, aif, time_points)

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
                params_nat, tissue_curve, aif, time_points,
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

        # NLLS warm-start candidate. A preceding batched fit may supply this
        # candidate; otherwise compute it with the scalar model implementation.
        x0_nlls_nat = x0_prior_nat.copy()
        param_names = self.model.get_parameter_names()
        if initial_parameters is not None:
            supplied = np.array(
                [initial_parameters.get(name, np.nan) for name in param_names],
                dtype=np.float64,
            )
            if np.all(np.isfinite(supplied)):
                x0_nlls_nat = supplied
            else:
                initial_parameters = None

        if initial_parameters is None:
            try:
                nlls_result = self.model.fit(
                    tissue_curve, aif, time_points, weights=weights,
                )
                if nlls_result.converged:
                    x0_nlls_nat = np.array(
                        [
                            nlls_result.parameters.get(p, x0_prior_nat[i])
                            for i, p in enumerate(param_names)
                        ],
                        dtype=np.float64,
                    )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass

        # Clip natural-space start points, then transform to unconstrained space
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        def _to_t(x_nat: npt.NDArray) -> npt.NDArray:
            x_nat = np.clip(x_nat, lower, upper)
            # Project the seed onto the ve + vp <= 1 feasible set so the simplex
            # starts inside the posterior support (the ve + vp > 1 region is a
            # hard 1e30 barrier); starting on it would stall Nelder-Mead.
            x_nat = np.asarray(self._apply_volume_constraint(x_nat), dtype=np.float64)
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
                    fitted_params_t,
                    neg_log_posterior_t,
                    prior_stds=np.asarray(
                        [prior.std_transformed for prior in self.priors],
                        dtype=np.float64,
                    ),
                )
                # _compute_hessian uses finite differences and can return
                # non-finite entries when perturbations land on the
                # boundary objective (1e30) or violate constraints.
                # np.linalg.inv on a non-finite matrix returns silently
                # NaN-laden output rather than raising, so guard explicitly
                # to route into the prior-std fallback below.
                if not np.all(np.isfinite(hessian_t)):
                    raise np.linalg.LinAlgError("non-finite Hessian")
                cov_t = np.linalg.inv(
                    hessian_t + 1e-6 * np.eye(n_params)
                )
                if not np.all(np.isfinite(cov_t)):
                    raise np.linalg.LinAlgError("non-finite covariance")
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
                    credible_intervals[name] = transformed_gaussian_interval(
                        float(fitted_params_t[i]),
                        float(stds_t[i]),
                        transforms[i],
                        upper_bounds_t[i],
                        bounds_natural[i],
                    )
                uncertainty_source = "laplace_hessian"
            except np.linalg.LinAlgError:
                # A prior width is not a posterior uncertainty estimate. Keep
                # the fitted parameters, but mark their local spread undefined.
                names = self.model.get_parameter_names()
                uncertainties = {
                    name: float("nan") for name in names
                }
                credible_intervals = {}
                uncertainty_source = "hessian_unavailable"
        else:
            uncertainties = {}
            credible_intervals = {}
            uncertainty_source = "not_computed"

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
            uncertainty_source=uncertainty_source,
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
        initial_parameters: Mapping[str, float] | None = None,
    ) -> FitResult:
        """Fit model using FABBER-style linearized Variational Bayes.

        Implements the linearized VB algorithm. At each iteration the forward model is
        linearized around the current posterior mean, enabling analytic
        Gaussian posterior updates. Noise precision is treated as a conjugate
        Gamma random variable.

        Parameters
        ----------
        tissue_curve : numpy.ndarray
            Measured tissue concentration curve (mM).
        aif : numpy.ndarray
            Arterial input function (mM).
        time_points : numpy.ndarray
            Time points (minutes).
        n_iterations : int, optional
            Maximum VB iterations (default 30).
        noise_prior : tuple of float or None, optional
            Optional (c0, s0) Gamma prior on noise precision
            tau = 1/sigma^2. If None, uses an informative prior centered
            on the measured pre-bolus noise (or ``noise_std`` when supplied
            to the constructor).
        weights : numpy.ndarray or None, optional
            Optional per-frame motion weights (1.0 = full weight).
            Weights modify data fidelity terms and contribute
            ``sum(weights)`` effective observations to the noise posterior.
        sloppy : bool, optional
            Use fast, low-quality settings for testing.
        initial_parameters : mapping of str to float or None, optional
            Natural-space parameter values used to initialize the variational
            linearization. When omitted, an NLLS warm start is computed.

        Returns
        -------
        FitResult
            FitResult with parameters, uncertainties, and credible intervals.

        Notes
        -----
        Finite rate bounds constrain each local quadratic mean update.
        Covariance remains a local Gaussian approximation, with reported
        marginal intervals intersected with model support; this is not an
        exact truncated-posterior uncertainty calculation at an active bound.
        Backtracking trials are evaluated before acceptance. Convergence
        requires an accepted trial and either a small full proposed mean
        update or absolute free-energy changes below ``VI_FREE_ENERGY_TOL``
        for both the full proposal and accepted trial. Shrinking an update
        alone cannot establish convergence. An exhausted line search without
        accepted progress is not itself evidence of convergence.

        """
        if sloppy:
            n_iterations = 8
        self._validate_inputs(tissue_curve, aif, time_points)

        n_params = self.model.n_parameters
        n_obs = len(tissue_curve)
        bounds = self.model.get_bounds()
        param_names = self.model.get_parameter_names()
        transforms = [prior.transform for prior in self.priors]
        upper_bounds_t = [prior.upper_bound for prior in self.priors]
        y = np.asarray(tissue_curve, dtype=np.float64)

        # Motion weights for data fidelity terms
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
        else:
            w = None
        n_eff = float(np.sum(w)) if w is not None else float(n_obs)

        # Fractions already have bounded transforms. Log-transformed rates
        # still require their finite model ceilings in transformed coordinates.
        lower_t = np.full(n_params, -np.inf)
        upper_t = np.full(n_params, np.inf)
        for i, (transform, (lo, hi)) in enumerate(zip(transforms, bounds)):
            if transform == TransformType.LOG:
                lower_t[i] = np.log(lo) if lo > 0 else -np.inf
                upper_t[i] = np.log(hi)
            elif transform == TransformType.IDENTITY:
                lower_t[i], upper_t[i] = lo, hi

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
            baseline_noise_std = self._estimate_noise_std(tissue_curve)
            tau_prior = 1.0 / (baseline_noise_std**2)
            c0 = 10.0
            s0 = tau_prior / c0

        # Warm-start the variational linearization from the data likelihood.
        # Priors remain unchanged; this only avoids linearizing a nonlinear PK
        # model around a distant prior mean, which can trap low-uptake curves
        # at a poor local approximation.
        m_nat = np.array([p.mean for p in self.priors], dtype=np.float64)
        if initial_parameters is not None:
            supplied = np.array(
                [initial_parameters.get(name, np.nan) for name in param_names],
                dtype=np.float64,
            )
            if np.all(np.isfinite(supplied)):
                m_nat = supplied
            else:
                initial_parameters = None

        if initial_parameters is None:
            try:
                warm_result = self.model.fit(
                    tissue_curve,
                    aif,
                    time_points,
                    weights=weights,
                )
                if warm_result.converged:
                    m_nat = np.array(
                        [
                            warm_result.parameters.get(name, m_nat[index])
                            for index, name in enumerate(param_names)
                        ],
                        dtype=np.float64,
                    )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                data_guess = self.model.estimate_initial_guess(
                    tissue_curve,
                    aif,
                    time_points,
                )
                m_nat = np.asarray(data_guess, dtype=np.float64)
        for i in range(n_params):
            m_nat[i] = np.clip(m_nat[i], bounds[i][0], bounds[i][1])
        m_nat = np.asarray(self._apply_volume_constraint(m_nat), dtype=np.float64)

        m = np.array([
            float(to_transformed(m_nat[i], transforms[i], upper_bounds_t[i]))
            for i in range(n_params)
        ], dtype=np.float64)

        # Initialize noise posterior
        c = c0
        s = s0
        tau_exp = c * s  # E[tau] = shape * scale

        # Only accepted trials may update the returned posterior or the next
        # iteration's linearization and noise precision.
        best_free_energy = float("-inf")
        best_m = m.copy()
        best_cov = np.diag(prior_stds_t**2)
        best_c = c
        best_s = s

        def jacobian(
            mean: npt.NDArray[np.floating],
            natural: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Evaluate bound-aware finite differences at the supplied mean."""
            J = np.zeros((n_obs, n_params), dtype=np.float64)
            deltas = np.maximum(np.abs(mean) * 1e-5, 1e-10)
            for j in range(n_params):
                # Only inverse-transform the perturbed param (not all n_params)
                p_nat = natural.copy()
                n_nat = natural.copy()
                plus_t = min(mean[j] + deltas[j], upper_t[j])
                minus_t = max(mean[j] - deltas[j], lower_t[j])
                p_nat[j] = float(np.clip(from_transformed(plus_t, transforms[j], upper_bounds_t[j]), *bounds[j]))
                n_nat[j] = float(np.clip(from_transformed(minus_t, transforms[j], upper_bounds_t[j]), *bounds[j]))
                g_plus = self.model.evaluate(list(p_nat), aif, time_points)
                g_minus = self.model.evaluate(list(n_nat), aif, time_points)
                J[:, j] = (np.asarray(g_plus) - np.asarray(g_minus)) / (plus_t - minus_t)
            if not np.all(np.isfinite(J)):
                raise ValueError("Nonfinite VI trial Jacobian")
            return J

        converged = False
        try:
            g_m = np.asarray(self.model.evaluate(list(m_nat), aif, time_points), dtype=np.float64)
            J = jacobian(m, m_nat)
        except (ValueError, RuntimeError):
            n_iterations = 0
        for iteration in range(n_iterations):
            # Residual
            k_res = y - g_m

            # Update posterior precision: Λ = Λ₀ + <τ> J^T W J
            if w is not None:
                JtWJ = J.T @ (w[:, np.newaxis] * J)
            else:
                JtWJ = J.T @ J
            post_prec = prior_prec + tau_exp * JtWJ

            # Update posterior covariance
            try:
                post_cov = np.asarray(np.linalg.inv(post_prec), dtype=np.float64)
            except np.linalg.LinAlgError:
                post_prec += 1e-6 * np.eye(n_params)
                post_cov = np.asarray(np.linalg.inv(post_prec), dtype=np.float64)

            # Update posterior mean (in transformed space)
            if w is not None:
                rhs = prior_prec @ prior_means_t + tau_exp * J.T @ (w * (k_res + J @ m))
            else:
                rhs = prior_prec @ prior_means_t + tau_exp * J.T @ (k_res + J @ m)
            m_new = post_cov @ rhs
            try:
                m_new = _bounded_vi_mean(post_prec, rhs, m_new, lower_t, upper_t)
            except (RuntimeError, np.linalg.LinAlgError):
                break

            # Each trial gets its own forward/Jacobian evaluation and Gamma
            # update. The covariance is the proposed local Gaussian update;
            # rejected trials must not alter the accepted state.
            previous_free_energy = best_free_energy
            full_free_energy = float("nan")
            step_scale = 1.0
            accepted = False
            while step_scale >= 1e-4:
                trial_m = m + step_scale * (m_new - m)
                trial_nat = np.array([
                    float(np.clip(from_transformed(trial_m[k], transforms[k], upper_bounds_t[k]), *bounds[k]))
                    for k in range(n_params)
                ], dtype=np.float64)
                trial_nat = np.asarray(
                    self._apply_volume_constraint(trial_nat), dtype=np.float64,
                )
                trial_m = np.array([
                    float(to_transformed(trial_nat[k], transforms[k], upper_bounds_t[k]))
                    for k in range(n_params)
                ], dtype=np.float64)
                if step_scale == 1.0:
                    full_param_change = float(np.max(np.abs(trial_m - m)))
                try:
                    g_trial = self.model.evaluate(list(trial_nat), aif, time_points)
                    trial_J = jacobian(trial_m, trial_nat)
                except (ValueError, RuntimeError):
                    step_scale *= 0.5
                    continue
                residual = y - np.asarray(g_trial, dtype=np.float64)
                weighted_rss = float(residual @ (w * residual if w is not None else residual))
                trial_gram = trial_J.T @ (w[:, np.newaxis] * trial_J if w is not None else trial_J)
                trial_c, trial_s, free_energy = _vi_noise_and_free_energy(
                    trial_m, post_cov, trial_gram, weighted_rss,
                    prior_means_t, prior_prec, (c0, s0), n_eff,
                )
                if step_scale == 1.0:
                    full_free_energy = free_energy
                if np.isfinite(free_energy) and free_energy >= best_free_energy:
                    best_free_energy = free_energy
                    best_m = trial_m.copy()
                    best_cov = post_cov.copy()
                    best_c, best_s = trial_c, trial_s
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                break

            # Convergence check
            # Check the undamped proposal as well as the accepted improvement:
            # backtracking can make the latter tiny far from a stable score.
            objective_stable = (
                abs(full_free_energy - previous_free_energy) < VI_FREE_ENERGY_TOL
                and abs(best_free_energy - previous_free_energy) < VI_FREE_ENERGY_TOL
            )
            if iteration > 0 and (full_param_change < 1e-3 or objective_stable):
                converged = True
                break
            m = best_m.copy()
            tau_exp = best_c * best_s
            g_m = np.asarray(g_trial, dtype=np.float64)
            J = trial_J

        # Inverse-transform best parameters to natural space
        fitted_params = np.array([
            float(np.clip(from_transformed(best_m[k], transforms[k], upper_bounds_t[k]), *bounds[k]))
            for k in range(n_params)
        ], dtype=np.float64)
        fitted_params = np.asarray(
            self._apply_volume_constraint(fitted_params),
            dtype=np.float64,
        )

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
            credible_intervals[name] = transformed_gaussian_interval(
                float(best_m[i]),
                std_t,
                transforms[i],
                upper_bounds_t[i],
                bounds[i],
            )

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
            converged=converged,
            fitting_method="vi",
            fitted_curve=fitted_curve,
            quality_flag="good" if converged else "poor_convergence",
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
        HalfNormal prior.

        The shared joint-volume constraint is enforced during sampling for
        models with both ve and vp parameters.

        Parameters
        ----------
        tissue_curve : numpy.ndarray
            Measured tissue concentration curve (mM).
        aif : numpy.ndarray
            Arterial input function (mM).
        time_points : numpy.ndarray
            Time points (minutes).
        n_samples : int, optional
            Number of samples to draw (after burn-in).
        n_walkers : int or None, optional
            Number of ensemble walkers (default: 2 * n_dims + 2).
        n_burnin : int or None, optional
            Number of burn-in samples (default: n_samples // 2).
        sloppy : bool, optional
            Use fast, low-quality settings for testing.
        weights : numpy.ndarray or None, optional
            Optional per-frame weights for motion-weighted fitting.
            Values in [0, 1] where 1.0 = full weight.

        Returns
        -------
        FitResult
            FitResult with parameters, uncertainties, credible intervals,
            and autocorrelation-based ensemble ESS diagnostics. R-hat is
            unavailable because walkers within one ensemble are coupled.
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
        transforms = [prior.transform for prior in self.priors]
        upper_bounds_t = [prior.upper_bound for prior in self.priors]

        # Set defaults
        if n_walkers is None:
            n_walkers = max(2 * n_dims + 2, 10)
        if n_burnin is None:
            n_burnin = n_samples // 2

        n_total = n_samples + n_burnin

        # Log posterior including sampled sigma
        # State vector: [transformed model params..., log_sigma]
        def log_prob_with_sigma(
            state: npt.NDArray[np.floating],
        ) -> float:
            model_params_t = state[:n_model_params]
            model_params = np.asarray([
                from_transformed(
                    model_params_t[i], transforms[i], upper_bounds_t[i]
                )
                for i in range(n_model_params)
            ], dtype=np.float64)
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
            log_p = compute_log_prior(
                model_params_t, self.priors, transformed=True
            )
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
            compute_uncertainties=False, weights=weights,
        )
        map_params_list = self.model.dict_to_params(
            {k: v for k, v in map_result.parameters.items() if k in param_names}
        )
        map_params = np.array(map_params_list, dtype=np.float64)
        map_params_t = np.asarray([
            to_transformed(map_params[i], transforms[i], upper_bounds_t[i])
            for i in range(n_model_params)
        ], dtype=np.float64)

        # Transformed parameters are unconstrained in principle. Finite guards
        # keep proposals numerically safe; identity transforms retain their
        # natural model bounds.
        sampler_bounds = [
            bounds[i] if transforms[i] == TransformType.IDENTITY else (-50.0, 50.0)
            for i in range(n_model_params)
        ] + [(-15.0, 15.0)]

        # Initialize walkers with small perturbations around MAP
        init_positions = np.zeros((n_walkers, n_dims))
        log_sigma_init = np.log(noise_std_baseline)
        for i in range(n_walkers):
            for j in range(n_model_params):
                scale = self.priors[j].std_transformed * 0.1
                perturbed = map_params_t[j] + scale * np.random.randn()
                perturbed = np.clip(
                    perturbed, sampler_bounds[j][0], sampler_bounds[j][1]
                )
                init_positions[i, j] = perturbed
            # log_sigma: small perturbation around baseline estimate
            init_positions[i, n_model_params] = (
                log_sigma_init + 0.1 * np.random.randn()
            )

            # Apply volume constraint to initialization
            if self._has_volume_constraint:
                initial_nat = np.asarray([
                    from_transformed(
                        init_positions[i, j], transforms[j], upper_bounds_t[j]
                    )
                    for j in range(n_model_params)
                ], dtype=np.float64)
                initial_nat = np.asarray(
                    self._apply_volume_constraint(initial_nat), dtype=np.float64
                )
                init_positions[i, :n_model_params] = np.asarray([
                    to_transformed(
                        initial_nat[j], transforms[j], upper_bounds_t[j]
                    )
                    for j in range(n_model_params)
                ], dtype=np.float64)

        # Run ensemble sampler (shared utility)
        samples = run_ensemble_sampler(
            log_prob_with_sigma, init_positions, n_total, sampler_bounds
        )

        # Discard burn-in
        samples = samples[n_burnin:, :, :]  # (n_samples, n_walkers, n_dims)

        # Flatten chains for analysis
        flat_samples = samples.reshape(-1, n_dims)

        # Separate model params and sigma
        transformed_samples = flat_samples[:, :n_model_params]
        model_samples = np.column_stack([
            np.asarray(from_transformed(
                transformed_samples[:, i], transforms[i], upper_bounds_t[i]
            ), dtype=np.float64)
            for i in range(n_model_params)
        ])
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

        # Coupled ensemble walkers are not independent chains, so R-hat is
        # intentionally unavailable. Use time-axis autocorrelation instead.
        ess, _tau, reliable = compute_ensemble_ess(samples)
        rhat_dict: dict[str, float] = {}
        ess_dict: dict[str, int] = {}
        all_names = param_names + ["log_sigma"]
        for i, name in enumerate(all_names):
            ess_dict[name] = int(ess[i])

        all_ess_ok = all(e > QC_ESS_MIN for e in ess)
        converged = bool(np.all(reliable)) and all_ess_ok
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

    Parameters
    ----------
    vp_value : float
        DSC-derived vp value for this voxel.
    relative_uncertainty : float, optional
        Relative uncertainty (fraction). Default 0.2
        means +/- 20% of the vp value.

    Returns
    -------
    BayesianPrior
        BayesianPrior with informative mean and std for vp.
    """
    std = max(vp_value * relative_uncertainty, 0.001)
    # vp uses LOG_LOGIT transform; compute transformed prior stats
    from pmriprep.models.bayesian_utils import transform_prior
    ub = VP_BOUNDS.max
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
    """Fit a PK model using the specified Bayesian method (convenience wrapper).

    Parameters
    ----------
    model : PKModel
        PK model to fit.
    tissue_curve : numpy.ndarray
        Measured tissue concentration (mM).
    aif : numpy.ndarray
        Arterial input function (mM).
    time_points : numpy.ndarray
        Time points (minutes).
    method : str, optional
        Fitting method: 'map', 'vi', or 'mcmc'.
    mcmc_samples : int, optional
        Number of MCMC samples (for method='mcmc').
    noise_std : float or None, optional
        Optional noise standard deviation.
    sloppy : bool, optional
        Use fast, low-quality settings for testing.

    Returns
    -------
    FitResult
        FitResult with fitted parameters and uncertainties.

    Raises
    ------
    ValueError
        If method is not recognized.
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
