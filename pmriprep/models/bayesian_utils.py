"""Shared Bayesian inference utilities for DSC and DCE perfusion models.

Provides generic MCMC sampling, convergence diagnostics, and statistical
utilities used by both BayesianDSCFitter and BayesianPKFitter.
"""

import enum
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import expit  # numerically stable sigmoid


# ---------------------------------------------------------------------------
# Parameter transform types
# ---------------------------------------------------------------------------


class TransformType(enum.Enum):
    """Parameter space transforms for Bayesian fitting.

    IDENTITY: No transform (unconstrained or can-be-negative params).
    LOG: Natural logarithm for strictly positive params (CBF, Ktrans, Fp, etc).
    LOG_LOGIT: Log-odds for bounded fraction params (ve, vp).
    Requires an upper bound: theta' = ln(theta / (ub - theta)).
    SIGMOID: FABBER TRANSFORM_FRACTIONAL — natural value lives in (0, 1)
    with no explicit upper bound; theta = 1/(1+exp(theta')) so the
    fitting-space prior mean=0.5 corresponds to natural=0.5. Used for
    CPI control-point fractions.

    """

    IDENTITY = "identity"
    LOG = "log"
    LOG_LOGIT = "log_logit"
    SIGMOID = "sigmoid"


# ---------------------------------------------------------------------------
# Transform utility functions
# ---------------------------------------------------------------------------


def to_transformed(
    theta: float | npt.NDArray[np.floating],
    transform: TransformType,
    upper_bound: float | None = None,
) -> float | npt.NDArray[np.floating]:
    """Map from natural parameter space to unconstrained transformed space.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Parameter value(s) in natural space.
    transform : TransformType
        Transform type.
    upper_bound : float, optional
        Upper bound for LOG_LOGIT (required).

    Returns
    -------
    float or numpy.ndarray
        Parameter value(s) in transformed space.
    """
    if transform == TransformType.LOG:
        result: float | npt.NDArray[np.floating] = np.log(np.maximum(theta, 1e-20))
        return result
    elif transform == TransformType.LOG_LOGIT:
        if upper_bound is None:
            raise ValueError("upper_bound required for LOG_LOGIT transform")
        t = np.clip(theta, 1e-10, upper_bound - 1e-10)
        result = np.log(t / (upper_bound - t))
        return result
    elif transform == TransformType.SIGMOID:
        # FABBER TRANSFORM_FRACTIONAL: theta = 1/(1+exp(theta'))
        # Inverse: theta' = log((1 - theta) / theta)
        t = np.clip(theta, 1e-10, 1.0 - 1e-10)
        result = np.log((1.0 - t) / t)
        return result
    return theta  # IDENTITY


def from_transformed(
    theta_prime: float | npt.NDArray[np.floating],
    transform: TransformType,
    upper_bound: float | None = None,
) -> float | npt.NDArray[np.floating]:
    """Map from unconstrained transformed space to natural parameter space.

    Parameters
    ----------
    theta_prime : float or numpy.ndarray
        Parameter value(s) in transformed space.
    transform : TransformType
        Transform type.
    upper_bound : float, optional
        Upper bound for LOG_LOGIT (required).

    Returns
    -------
    float or numpy.ndarray
        Parameter value(s) in natural space.
    """
    if transform == TransformType.LOG:
        # Overflow protection: exp(709) ≈ 8.2e307, exp(710) overflows
        result: float | npt.NDArray[np.floating] = np.exp(np.clip(theta_prime, -500, 500))
        return result
    elif transform == TransformType.LOG_LOGIT:
        if upper_bound is None:
            raise ValueError("upper_bound required for LOG_LOGIT transform")
        # Numerically stable sigmoid via scipy.special.expit
        result = upper_bound * expit(np.clip(theta_prime, -500, 500))
        return result
    elif transform == TransformType.SIGMOID:
        # FABBER TRANSFORM_FRACTIONAL: theta = 1/(1+exp(theta'))
        # = expit(-theta')
        result = expit(-np.clip(theta_prime, -50, 50))
        return result
    return theta_prime  # IDENTITY


def transformed_gaussian_interval(
    mean_transformed: float,
    std_transformed: float,
    transform: TransformType,
    upper_bound: float | None = None,
    natural_bounds: tuple[float, float] | None = None,
    z: float = 1.96,
) -> tuple[float, float]:
    """Map a symmetric transformed Gaussian interval to natural space.

    Parameters
    ----------
    mean_transformed : float
        Posterior mean or mode in transformed coordinates.
    std_transformed : float
        Marginal posterior standard deviation in transformed coordinates.
    transform : TransformType
        Parameter transform.
    upper_bound : float or None, optional
        Natural-space upper bound required by ``LOG_LOGIT``.
    natural_bounds : tuple of float or None, optional
        Additional natural-space bounds used for identity transforms and
        finite model constraints.
    z : float, optional
        Gaussian quantile multiplier; 1.96 gives a nominal 95% interval.

    Returns
    -------
    tuple of float
        Ordered lower and upper interval limits in natural coordinates.
        Both limits are NaN if the interval has no overlap with model support.
    """
    endpoints = np.asarray(from_transformed(
        np.asarray([
            mean_transformed - z * std_transformed,
            mean_transformed + z * std_transformed,
        ]),
        transform,
        upper_bound,
    ), dtype=np.float64)
    lower = float(np.min(endpoints))
    upper = float(np.max(endpoints))
    if natural_bounds is not None:
        lower = max(lower, natural_bounds[0])
        upper = min(upper, natural_bounds[1])
    if lower > upper:
        return float("nan"), float("nan")
    return lower, upper


def to_transformed_batch(
    values: npt.NDArray[np.floating],
    transform: TransformType,
    upper_bound: float | None = None,
) -> npt.NDArray[np.float64]:
    """Vectorized forward transform for (N,) array of one parameter.

    Maps from natural parameter space to unconstrained transformed space,
    with NaN guards for numerical safety in batched fitting.

    Parameters
    ----------
    values : numpy.ndarray
        Parameter values in natural space, shape (N,).
    transform : TransformType
        Transform type for this parameter.
    upper_bound : float, optional
        Upper bound for LOG_LOGIT (required).

    Returns
    -------
    numpy.ndarray
        Parameter values in transformed space, shape (N,).
    """
    if transform == TransformType.LOG:
        result = np.log(np.maximum(values, 1e-20))
    elif transform == TransformType.LOG_LOGIT:
        if upper_bound is None:
            raise ValueError("upper_bound required for LOG_LOGIT transform")
        t = np.clip(values, 1e-10, upper_bound - 1e-10)
        result = np.log(t / (upper_bound - t))
    elif transform == TransformType.SIGMOID:
        t = np.clip(values, 1e-10, 1.0 - 1e-10)
        result = np.log((1.0 - t) / t)
    else:
        result = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(np.asarray(result, dtype=np.float64), nan=0.0)


def from_transformed_batch(
    values: npt.NDArray[np.floating],
    transform: TransformType,
    upper_bound: float | None = None,
) -> npt.NDArray[np.float64]:
    """Vectorized inverse transform for (N,) array of one parameter.

    Maps from unconstrained transformed space to natural parameter space,
    with NaN guards for numerical safety in batched fitting.

    Parameters
    ----------
    values : numpy.ndarray
        Parameter values in transformed space, shape (N,).
    transform : TransformType
        Transform type for this parameter.
    upper_bound : float, optional
        Upper bound for LOG_LOGIT (required).

    Returns
    -------
    numpy.ndarray
        Parameter values in natural space, shape (N,).
    """
    if transform == TransformType.LOG:
        clipped = np.clip(values, -500, 500)
        result = np.exp(clipped)
    elif transform == TransformType.LOG_LOGIT:
        if upper_bound is None:
            raise ValueError("upper_bound required for LOG_LOGIT transform")
        clipped = np.clip(values, -500, 500)
        result = upper_bound * expit(clipped)
    elif transform == TransformType.SIGMOID:
        clipped = np.clip(values, -50, 50)
        result = expit(-clipped)
    else:
        result = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(np.asarray(result, dtype=np.float64), nan=0.0)


def log_jacobian_det(
    theta: float | npt.NDArray[np.floating],
    transform: TransformType,
    upper_bound: float | None = None,
) -> float | npt.NDArray[np.floating]:
    """Log absolute Jacobian determinant ``|dtheta/dtheta'|``.

    Used to correct the MAP objective when optimizing in transformed space::

        neg_log_posterior_transformed = neg_log_posterior_natural - log|J|

    For ``log``: ``|J| = theta``, so ``log|J| = log(theta)``.
    For ``log-logit``: ``|J| = theta*(ub-theta)/ub``, so ``log|J| = log(theta) + log(ub-theta) - log(ub)``.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Parameter value(s) in natural space.
    transform : TransformType
        Transform type.
    upper_bound : float, optional
        Upper bound for LOG_LOGIT.

    Returns
    -------
    float or numpy.ndarray
        Log absolute Jacobian determinant.
    """
    if transform == TransformType.LOG:
        result: float | npt.NDArray[np.floating] = np.log(np.maximum(np.abs(theta), 1e-20))
        return result
    elif transform == TransformType.LOG_LOGIT:
        if upper_bound is None:
            raise ValueError("upper_bound required for LOG_LOGIT transform")
        t = np.clip(theta, 1e-10, upper_bound - 1e-10)
        result = np.log(t) + np.log(upper_bound - t) - np.log(upper_bound)
        return result
    elif transform == TransformType.SIGMOID:
        # |dtheta/dtheta'| = theta * (1 - theta)
        t = np.clip(theta, 1e-10, 1.0 - 1e-10)
        result = np.log(t) + np.log(1.0 - t)
        return result
    return 0.0  # IDENTITY


def inverse_transform_all(
    params_t: npt.NDArray[np.floating],
    transforms: list[TransformType],
    upper_bounds: list[float | None],
) -> npt.NDArray[np.float64]:
    """Inverse-transform all parameters from transformed to natural space.

    Parameters
    ----------
    params_t : numpy.ndarray
        Parameter values in transformed space.
    transforms : list of TransformType
        Per-parameter transform types.
    upper_bounds : list of float or None
        Per-parameter upper bounds (for LOG_LOGIT).

    Returns
    -------
    numpy.ndarray
        Parameter values in natural space.
    """
    return np.array([
        float(from_transformed(params_t[i], transforms[i], upper_bounds[i]))
        for i in range(len(params_t))
    ], dtype=np.float64)


def forward_transform_all(
    params_nat: npt.NDArray[np.floating],
    transforms: list[TransformType],
    upper_bounds: list[float | None],
) -> npt.NDArray[np.float64]:
    """Transform all parameters from natural to transformed space.

    Parameters
    ----------
    params_nat : numpy.ndarray
        Parameter values in natural space.
    transforms : list of TransformType
        Per-parameter transform types.
    upper_bounds : list of float or None
        Per-parameter upper bounds (for LOG_LOGIT).

    Returns
    -------
    numpy.ndarray
        Parameter values in transformed space.
    """
    return np.array([
        float(to_transformed(params_nat[i], transforms[i], upper_bounds[i]))
        for i in range(len(params_nat))
    ], dtype=np.float64)


def transform_prior(
    mean: float,
    std: float,
    transform: TransformType,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    """Transform prior mean/std from natural to transformed space (delta method).

    Uses first-order Taylor approximation. Accurate when std << mean
    (CV < ~0.5). For log: mu'=ln(mu), sigma'=sigma/mu.
    For log-logit: mu'=ln(mu/(ub-mu)), sigma'=sigma*ub/(mu*(ub-mu)).

    Parameters
    ----------
    mean : float
        Prior mean in natural space.
    std : float
        Prior std in natural space.
    transform : TransformType
        Transform type.
    upper_bound : float, optional
        Upper bound for LOG_LOGIT.

    Returns
    -------
    tuple of float
        Tuple of (transformed_mean, transformed_std).
    """
    if transform == TransformType.LOG:
        if mean <= 0:
            raise ValueError(f"Log transform requires positive mean, got {mean}")
        return float(np.log(mean)), float(std / mean)
    elif transform == TransformType.LOG_LOGIT:
        if upper_bound is None:
            raise ValueError("upper_bound required for LOG_LOGIT transform")
        # Guard against boundary means
        m = np.clip(mean, 1e-10, upper_bound - 1e-10)
        m_prime = float(np.log(m / (upper_bound - m)))
        s_prime = float(std * upper_bound / (m * (upper_bound - m)))
        return m_prime, s_prime
    elif transform == TransformType.SIGMOID:
        # theta = 1/(1+exp(theta')); dtheta'/dtheta = -1/(theta*(1-theta))
        m = float(np.clip(mean, 1e-10, 1.0 - 1e-10))
        m_prime = float(np.log((1.0 - m) / m))
        s_prime = float(std / (m * (1.0 - m)))
        return m_prime, s_prime
    return mean, std  # IDENTITY


# ---------------------------------------------------------------------------
# Protocol for parameter definitions (DSCModelParameter / ModelParameter)
# ---------------------------------------------------------------------------

class ParameterDef(Protocol):
    """Protocol matching both DSCModelParameter and DCE ModelParameter."""

    transform: TransformType
    bounds: tuple[float, float]
    transformed_prior_sigma: float | None

    def get_prior_mean(self) -> float:
        """Return the prior mean in natural space."""
        ...

    def get_prior_sigma(self) -> float:
        """Return the prior standard deviation in natural space."""
        ...


# ---------------------------------------------------------------------------
# BayesianPrior dataclass (single source of truth)
# ---------------------------------------------------------------------------

@dataclass
class BayesianPrior:
    """Gaussian prior specification for a model parameter.

    Stores prior in both natural and transformed space. The fitting
    code operates in transformed space; natural-space values are kept
    for reporting and diagnostics.

    Attributes
    ----------
    mean : float
        Prior mean in natural space.
    std : float
        Prior standard deviation in natural space.
    transform : TransformType
        Parameter space transform type.
    upper_bound : float or None
        Upper bound for LOG_LOGIT transform (None otherwise).
    mean_transformed : float
        Prior mean in transformed space.
    std_transformed : float
        Prior std in transformed space.
    """

    mean: float
    std: float
    transform: TransformType = TransformType.IDENTITY
    upper_bound: float | None = None
    mean_transformed: float = 0.0
    std_transformed: float = 1.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_bayesian_inputs(
    *arrays: tuple[str, npt.NDArray[np.floating]],
) -> None:
    """Validate input arrays for NaN/Inf values and length consistency.

    Parameters
    ----------
    *arrays : tuple of (str, numpy.ndarray)
        Tuples of (name, array) to validate. All arrays must have
        the same length and contain only finite values.

    Raises
    ------
    ValueError
        If inputs contain NaN/Inf or lengths don't match.
    """
    if not arrays:
        return

    for name, arr in arrays:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf values")

    first_len = len(arrays[0][1])
    for name, arr in arrays[1:]:
        if len(arr) != first_len:
            raise ValueError(
                f"Input array lengths must match: {arrays[0][0]}={first_len}, "
                f"{name}={len(arr)}"
            )


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

def build_priors_from_params(
    param_defs: Sequence[ParameterDef],
) -> list[BayesianPrior]:
    """Build Gaussian priors from parameter definitions.

    Prior mean is the center of typical_range.
    Prior std is (typical_range[1] - typical_range[0]) / 4.0,
    which places 95% of prior mass within the typical range.

    Priors are also transformed to the fitting space using the delta
    method (first-order Taylor approximation).

    Parameters
    ----------
    param_defs : sequence of ParameterDef
        List of parameter objects implementing ParameterDef.

    Returns
    -------
    list of BayesianPrior
        List of BayesianPrior objects with both natural and transformed
        prior mean/std populated.
    """
    priors = []
    for param in param_defs:
        nat_mean = param.get_prior_mean()
        nat_std = param.get_prior_sigma()
        tfm = param.transform
        ub = param.bounds[1] if tfm == TransformType.LOG_LOGIT else None

        # SIGMOID has natural range (0, 1) — no upper-bound metadata stored
        # but transform_prior handles it via the implicit (0, 1) clip.
        t_mean, t_std = transform_prior(nat_mean, nat_std, tfm, ub)

        # Allow explicit override of transformed-space sigma
        # (e.g., FABBER convention: sigma_log=1.0 for LOG parameters)
        if param.transformed_prior_sigma is not None:
            t_std = param.transformed_prior_sigma

        prior = BayesianPrior(
            mean=nat_mean,
            std=nat_std,
            transform=tfm,
            upper_bound=ub,
            mean_transformed=t_mean,
            std_transformed=t_std,
        )
        priors.append(prior)
    return priors


# ---------------------------------------------------------------------------
# Log prior
# ---------------------------------------------------------------------------

def compute_log_prior(
    params: npt.NDArray[np.floating],
    priors: list[BayesianPrior],
    transformed: bool = False,
) -> float:
    """Compute log prior probability.

    Parameters
    ----------
    params : numpy.ndarray
        Parameter values (in natural or transformed space).
    priors : list of BayesianPrior
        Prior specifications.
    transformed : bool, optional
        If True, params are in transformed space and the
        prior is evaluated using transformed mean/std.

    Returns
    -------
    float
        Log prior probability (unnormalized).
    """
    log_p = 0.0
    for p, prior in zip(params, priors):
        if transformed:
            mean = prior.mean_transformed
            std = prior.std_transformed
        else:
            mean = prior.mean
            std = prior.std
        if std > 0:
            log_p -= 0.5 * ((p - mean) / std) ** 2
    return log_p


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------

def compute_goodness_of_fit(
    observed: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
    n_params: int,
    epsilon: float = 1e-300,
) -> dict[str, float]:
    """Compute goodness-of-fit metrics.

    Parameters
    ----------
    observed : numpy.ndarray
        Measured concentration/signal.
    predicted : numpy.ndarray
        Model-predicted values.
    n_params : int
        Number of fitted parameters.
    epsilon : float, optional
        Floor for log(rss/n) to avoid log(0).

    Returns
    -------
    dict
        Dictionary with r_squared, rss, aic, bic.
    """
    residuals = observed - predicted
    rss = float(np.sum(residuals**2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 - (rss / max(ss_tot, 1e-10))

    n = len(observed)
    k = n_params
    if rss > 0:
        aic = n * np.log(max(rss / n, epsilon)) + 2 * k
        bic = n * np.log(max(rss / n, epsilon)) + k * np.log(n)
    else:
        aic = float("-inf")
        bic = float("-inf")

    return {
        "r_squared": r_squared,
        "rss": rss,
        "aic": float(aic),
        "bic": float(bic),
    }


# ---------------------------------------------------------------------------
# Ensemble sampler
# ---------------------------------------------------------------------------

def run_ensemble_sampler(
    log_prob: Callable[[npt.NDArray[np.floating]], float],
    init_positions: npt.NDArray[np.floating],
    n_iterations: int,
    bounds: list[tuple[float, float]],
    a: float = 2.0,
) -> npt.NDArray[np.floating]:
    """Run affine-invariant ensemble sampler.

    Implements the stretch move.

    Parameters
    ----------
    log_prob : callable
        Log posterior function.
    init_positions : numpy.ndarray
        Initial walker positions (n_walkers x n_params).
    n_iterations : int
        Number of iterations.
    bounds : list of tuple of float
        Parameter bounds.
    a : float, optional
        Stretch move scale parameter (default 2.0).

    Returns
    -------
    numpy.ndarray
        Samples array (n_iterations x n_walkers x n_params).
    """
    n_walkers, n_params = init_positions.shape

    samples = np.zeros((n_iterations, n_walkers, n_params))
    samples[0] = init_positions

    current_pos = init_positions.copy()
    current_lp = np.array([log_prob(current_pos[j]) for j in range(n_walkers)])

    for iteration in range(1, n_iterations):
        for j in range(n_walkers):
            others = np.concatenate([np.arange(j), np.arange(j + 1, n_walkers)])
            k = np.random.choice(others)

            u = np.random.rand()
            z = ((a - 1.0) * u + 1.0) ** 2 / a

            proposal = current_pos[k] + z * (current_pos[j] - current_pos[k])

            in_bounds = True
            for i in range(n_params):
                if proposal[i] < bounds[i][0] or proposal[i] > bounds[i][1]:
                    in_bounds = False
                    break

            if in_bounds:
                proposal_lp = log_prob(proposal)
                log_ratio = (
                    (n_params - 1) * np.log(z) + proposal_lp - current_lp[j]
                )
                if np.log(np.random.rand()) < log_ratio:
                    current_pos[j] = proposal
                    current_lp[j] = proposal_lp

        samples[iteration] = current_pos

    return samples


def compute_ensemble_ess(
    samples: npt.NDArray[np.floating],
    epsilon: float = 1e-10,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Estimate ensemble ESS from integrated autocorrelation time.

    Autocovariance is averaged across walkers at each time lag, preserving
    the sampler's time axis. Walkers are not treated as independent chains.
    Reliability requires at least 50 estimated autocorrelation times in the
    retained time series.

    Parameters
    ----------
    samples : numpy.ndarray
        MCMC samples (n_samples x n_walkers x n_params).
    epsilon : float, optional
        Minimum variance threshold below which a parameter is treated as
        constant and its diagnostic is marked unreliable.

    Returns
    -------
    ess : numpy.ndarray
        Effective sample sizes for each parameter.
    tau : numpy.ndarray
        Integrated autocorrelation-time estimates in retained iterations.
    reliable : numpy.ndarray
        Whether the retained chain is at least 50 autocorrelation times long.
    """
    n_steps, n_walkers, n_params = samples.shape
    ess = np.ones(n_params, dtype=np.int64)
    tau = np.full(n_params, np.inf, dtype=np.float64)
    reliable = np.zeros(n_params, dtype=bool)

    for p in range(n_params):
        centered = samples[:, :, p] - np.mean(samples[:, :, p], axis=0)
        variance = float(np.mean(centered**2))
        if n_steps < 4 or variance < epsilon:
            continue
        max_lag = min(n_steps // 2, 500)
        acf = np.empty(max_lag, dtype=np.float64)
        acf[0] = 1.0
        for lag in range(1, max_lag):
            acf[lag] = float(np.mean(
                centered[:-lag, :] * centered[lag:, :]
            )) / variance
        tau_p = 1.0
        for lag in range(1, max_lag - 1, 2):
            pair_sum = acf[lag] + acf[lag + 1]
            if pair_sum <= 0.0:
                break
            tau_p += 2.0 * pair_sum
        tau[p] = max(tau_p, 1.0)
        ess[p] = max(1, int(n_steps * n_walkers / tau[p]))
        reliable[p] = n_steps >= 50.0 * tau[p]

    return ess, tau, reliable
