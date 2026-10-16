"""Signal-CPI prior, noise and local-score definitions; no inference loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, gammaln

Array = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianPrior:
    """Normalized joint Gaussian prior, in the inferred coordinates."""

    mean: Array
    covariance: Array


@dataclass(frozen=True)
class GammaPrecision:
    """Gamma distribution over observation precision, using shape/rate."""

    shape: float
    rate: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.shape) or not np.isfinite(self.rate):
            raise ValueError("Gamma shape and rate must be finite")
        if self.shape <= 0 or self.rate <= 0:
            raise ValueError("Gamma shape and rate must be positive")

    @property
    def mean(self) -> float:
        """Return expected precision, not reciprocal expected variance."""
        return self.shape / self.rate

    @property
    def expected_log(self) -> float:
        """Return the expected log precision."""
        return float(digamma(self.shape) - np.log(self.rate))


def solve_spd(matrix: Array, rhs: Array) -> Array:
    """Solve a positive-definite system after diagonal unit rescaling."""
    scale = np.sqrt(np.diag(matrix))
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("matrix must have finite positive diagonal")
    scaled = matrix / scale[:, None] / scale[None, :]
    factor = cho_factor(scaled, lower=True, check_finite=True)
    if rhs.ndim == 1:
        return np.asarray(cho_solve(factor, rhs / scale) / scale)
    return np.asarray(cho_solve(factor, rhs / scale[:, None]) / scale[:, None])


def _logdet(matrix: Array) -> float:
    """Return a positive-definite log determinant."""
    factor = np.linalg.cholesky(matrix)
    return float(2 * np.log(np.diag(factor)).sum())


def local_free_energy(
    observed: Array,
    prediction: Array,
    jacobian: Array,
    mean: Array,
    covariance: Array,
    prior: GaussianPrior,
    noise: GammaPrecision | None,
    noise_prior: GammaPrecision,
    *,
    known_precision: float | None = None,
    weights: Array | None = None,
) -> float:
    """Evaluate normalized local expected likelihood minus both KL terms.

    Parameters
    ----------
    observed, prediction : ndarray
        Observations and nonlinear prediction at the current mean.
    jacobian : ndarray
        Signal derivative in inferred coordinates.
    mean, covariance : ndarray
        Joint Gaussian variational posterior.
    prior : GaussianPrior
        Proper Gaussian prior in the same coordinates.
    noise, noise_prior : GammaPrecision
        Gamma posterior and prior, in shape/rate convention.
    known_precision : float, optional
        Diagnostic fixed-noise likelihood; no noise KL is included.
    weights : ndarray, optional
        Validated frame likelihood weights. Their sum is the effective count.

    Returns
    -------
    float
        Local variational free energy, retaining all normalization terms.

    Raises
    ------
    ValueError
        If noise is unknown but no Gamma posterior is supplied.
    """
    if known_precision is None:
        if noise is None:
            raise ValueError("unknown noise requires a Gamma posterior")
        precision, log_precision = noise.mean, noise.expected_log
        a, b, a0, b0 = noise.shape, noise.rate, noise_prior.shape, noise_prior.rate
        noise_kl = (
            (a - a0) * digamma(a)
            - gammaln(a)
            + gammaln(a0)
            + a0 * (np.log(b) - np.log(b0))
            + a * (b0 / b - 1)
        )
    else:
        precision, log_precision = known_precision, np.log(known_precision)
        noise_kl = 0.0
    residual = observed - prediction
    expected_error = residual @ residual + np.einsum(
        "ni,ij,nj->", jacobian, covariance, jacobian
    )
    count = float(observed.size)
    if weights is not None:
        expected_error = np.sum(weights * residual**2) + np.einsum(
            "n,ni,ij,nj->", weights, jacobian, covariance, jacobian
        )
        count = float(np.sum(weights))
    likelihood = 0.5 * (
        count * (log_precision - np.log(2 * np.pi)) - precision * expected_error
    )
    delta = mean - prior.mean
    prior_precision = solve_spd(prior.covariance, np.eye(mean.size))
    parameter_kl = 0.5 * (
        np.trace(prior_precision @ covariance)
        + delta @ prior_precision @ delta
        - mean.size
        + _logdet(prior.covariance)
        - _logdet(covariance)
    )
    return float(likelihood - parameter_kl - noise_kl)


@dataclass(frozen=True)
class SharedFit:
    """One production VI restart, including unsuccessful fits."""

    voxel: int
    order: int
    restart: int
    converged: bool
    score: float
    cbf: float
    cbv: float
    mtt: float
    delay: float
    s0: float
    noise_sd: float
    rss: float
    mean: list[float]
