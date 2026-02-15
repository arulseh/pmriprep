"""Shared Bayesian inference utilities for DSC and DCE perfusion models.

Provides generic MCMC sampling, convergence diagnostics, and statistical
utilities used by both BayesianDSCFitter and BayesianPKFitter.

References:
    Goodman J, Weare J (2010). CAMCOS 5(1):65-80. — Ensemble sampler
    Chappell MA et al. (2009). IEEE TMI 28(1):94-104. — FABBER VB
    Vehtari A et al. (2021). Bayesian Analysis 16(2):667-718. — Split R-hat
"""

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Protocol for parameter definitions (DSCModelParameter / ModelParameter)
# ---------------------------------------------------------------------------

class ParameterDef(Protocol):
    """Protocol matching both DSCModelParameter and DCE ModelParameter."""

    log_transform: bool

    def get_prior_mean(self) -> float: ...
    def get_prior_sigma(self) -> float: ...


# ---------------------------------------------------------------------------
# BayesianPrior dataclass (single source of truth)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_bayesian_inputs(
    *arrays: tuple[str, npt.NDArray[np.floating]],
) -> None:
    """Validate input arrays for NaN/Inf values and length consistency.

    Args:
        *arrays: Tuples of (name, array) to validate. All arrays must have
            the same length and contain only finite values.

    Raises:
        ValueError: If inputs contain NaN/Inf or lengths don't match.
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

    Args:
        param_defs: List of parameter objects implementing ParameterDef.

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


# ---------------------------------------------------------------------------
# Log prior
# ---------------------------------------------------------------------------

def compute_log_prior(
    params: npt.NDArray[np.floating],
    priors: list[BayesianPrior],
) -> float:
    """Compute log prior probability.

    Uses truncated normal priors: simple Gaussian in natural parameter
    space, hard-clipped to parameter bounds. The log_transform flag is
    NOT applied (matches perf-plus behavior).

    Args:
        params: Parameter values.
        priors: Prior specifications.

    Returns:
        Log prior probability (unnormalized).
    """
    log_p = 0.0
    for p, prior in zip(params, priors):
        if prior.std > 0:
            log_p -= 0.5 * ((p - prior.mean) / prior.std) ** 2
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

    Args:
        observed: Measured concentration/signal.
        predicted: Model-predicted values.
        n_params: Number of fitted parameters.
        epsilon: Floor for log(rss/n) to avoid log(0).

    Returns:
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


# ---------------------------------------------------------------------------
# Split R-hat (Vehtari et al. 2021)
# ---------------------------------------------------------------------------

def compute_split_rhat(
    samples: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute split-R-hat diagnostic for ensemble samplers.

    For affine-invariant ensemble samplers, walkers are coupled through
    the complementary ensemble and are NOT independent chains.
    Standard per-walker R-hat is inappropriate.

    Instead, concatenate all walkers into a single chain and split
    in half, then compute R-hat between the two halves (recommended
    by emcee documentation and Vehtari et al. 2021).

    Args:
        samples: MCMC samples (n_samples x n_walkers x n_params).

    Returns:
        Split R-hat values for each parameter.
    """
    _, _, n_params = samples.shape

    rhat = np.zeros(n_params)

    for p in range(n_params):
        full_chain = samples[:, :, p].flatten()

        mid = len(full_chain) // 2
        chains = np.array([full_chain[:mid], full_chain[mid : 2 * mid]])

        n = chains.shape[1]

        chain_means = np.mean(chains, axis=1)

        B = n * np.var(chain_means, ddof=1)
        W = np.mean(np.var(chains, axis=1, ddof=1))

        var_plus = ((n - 1) / n) * W + (1 / n) * B

        if W > 0:
            rhat[p] = np.sqrt(var_plus / W)
        else:
            rhat[p] = 1.0

    return rhat


# ---------------------------------------------------------------------------
# Per-walker ESS
# ---------------------------------------------------------------------------

def compute_per_walker_ess(
    samples: npt.NDArray[np.floating],
    epsilon: float = 1e-10,
) -> npt.NDArray[np.integer]:
    """Compute effective sample size (ESS) per-walker then sum.

    For ensemble samplers, computing ESS per-walker and summing
    avoids underestimating autocorrelation that occurs when
    correlated walkers are flattened into a single sequence.

    Args:
        samples: MCMC samples (n_samples x n_walkers x n_params).
        epsilon: Minimum variance threshold below which chain is
            treated as constant (ESS = n).

    Returns:
        ESS values for each parameter.
    """
    _, n_walkers, n_params = samples.shape

    ess = np.zeros(n_params, dtype=np.int64)

    for p in range(n_params):
        ess_total = 0

        for w in range(n_walkers):
            chain = samples[:, w, p]
            n = len(chain)

            if n < 2:
                ess_total += n
                continue

            x = chain - np.mean(chain)
            var = float(np.var(x))

            if var < epsilon:
                ess_total += n
                continue

            max_lag = min(n // 2, 500)
            acf = np.zeros(max_lag)
            for k in range(max_lag):
                acf[k] = float(np.mean(x[: n - k] * x[k:])) / var

            tau_int = 1.0
            for k in range(1, max_lag - 1, 2):
                pair_sum = acf[k] + acf[k + 1]
                if pair_sum > 0:
                    tau_int += pair_sum
                else:
                    break

            ess_total += max(1, int(n / (2 * tau_int)))

        ess[p] = max(1, ess_total)

    return ess
