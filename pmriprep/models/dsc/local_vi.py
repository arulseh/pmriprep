"""Consistent local Gaussian/Gamma factors at a fixed nonlinear-model mean."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import digamma, gammaln

Array = npt.NDArray[np.float64]


class LocalVIFactorError(RuntimeError):
    """Raised when numerical linear algebra cannot form local VI factors."""


@dataclass(frozen=True)
class LocalVIState:
    """Local covariance, Gamma shape/scale, and normalized variational score."""

    covariance: Array
    shape: float
    scale: Array
    score: Array


def conditional_mean_gain(
    step: Array,
    prediction_change: Array,
    residual: Array,
    centered_mean: Array,
    prior_precision: Array,
    noise_precision: Array,
    weights: Array,
) -> Array:
    """Evaluate mean-step improvement with the current noise precision fixed.

    Parameters
    ----------
    step, centered_mean : ndarray
        Proposed mean displacement and current mean minus prior mean, (N, P).
    prediction_change, residual : ndarray
        Trial minus current prediction and current data residual, (N, T).
    prior_precision : ndarray
        Gaussian prior precision, (P, P).
    noise_precision : ndarray
        Current expected Gamma precision, (N,).
    weights : ndarray
        Nonnegative observation weights, (T,).

    Returns
    -------
    ndarray
        Conditional log-likelihood-plus-prior improvement for each row.

    Notes
    -----
    Covariance, noise and the linearized uncertainty penalty are held fixed
    during this mean substep. Their terms cancel. After accepting a mean,
    relinearize and settle its factors separately. This gain is not a change
    in the fully relinearized local evidence approximation.
    """
    data_gain = noise_precision * np.sum(
        weights[None, :] * prediction_change * (residual - 0.5 * prediction_change),
        axis=1,
    )
    prior_loss = np.einsum(
        "ni,ij,nj->n", step, prior_precision, centered_mean + 0.5 * step
    )
    return np.asarray(data_gain - prior_loss, dtype=np.float64)


def local_vi_state(
    mean: Array,
    jacobian: Array,
    residual: Array,
    prior_mean: Array,
    prior_precision: Array,
    weights: Array,
) -> LocalVIState:
    """Settle both local factors without changing the supplied mean.

    Parameters
    ----------
    mean : ndarray
        Inferred-coordinate means, shape (N, P).
    jacobian : ndarray
        Derivatives at these means, shape (N, T, P).
    residual : ndarray
        Observed minus predicted data at these means, shape (N, T).
    prior_mean, prior_precision : ndarray
        Proper Gaussian prior in the same coordinates.
    weights : ndarray
        Nonnegative observation weights, shape (T,).

    Returns
    -------
    LocalVIState
        Mutually consistent local Gaussian/Gamma factors and their score.

    Notes
    -----
    Prior-whitened singular values yield a monotonic scalar equation for
    expected noise precision. Solving it enforces covariance and noise updates
    together, including rank-deficient likelihoods. The score remains a local
    linearization, not exact nonlinear evidence or a truncated posterior.
    """
    if (
        not all(np.all(np.isfinite(x)) for x in (prior_mean, prior_precision, weights))
        or np.any(weights < 0)
        or np.sum(weights) <= 0
    ):
        raise ValueError("finite prior factors and nonnegative weights required")
    if not all(np.all(np.isfinite(x)) for x in (mean, jacobian, residual)):
        raise LocalVIFactorError("local VI mean, Jacobian, or residual is non-finite")
    n_parameters = mean.shape[1]
    root_precision = np.linalg.cholesky(prior_precision)
    prior_root = np.linalg.solve(root_precision.T, np.eye(n_parameters))
    weighted = jacobian * np.sqrt(weights)[None, :, None]
    whitened = weighted @ prior_root
    if not np.all(np.isfinite(whitened)):
        raise LocalVIFactorError("local VI prior-whitened Jacobian is non-finite")
    try:
        _, singular, vectors = np.linalg.svd(
            whitened, full_matrices=jacobian.shape[1] < n_parameters
        )
    except np.linalg.LinAlgError as exc:
        raise LocalVIFactorError("local VI singular-value decomposition failed") from exc
    eigenvalues = np.zeros((len(mean), n_parameters))
    eigenvalues[:, : singular.shape[1]] = singular**2
    shape0, rate0 = 1e-6, 1e-6
    n_eff = float(np.sum(weights))
    shape = shape0 + n_eff / 2
    rss = np.sum(weights[None, :] * residual**2, axis=1)
    base_rate = rate0 + 0.5 * rss
    lower = np.zeros(len(mean))
    upper = shape / base_rate
    precision = upper.copy()
    for _ in range(96):
        precision = 0.5 * (lower + upper)
        product = precision[:, None] * eigenvalues
        equation = (
            precision * base_rate
            + 0.5 * np.sum(product / (1 + product), axis=1)
            - shape
        )
        lower = np.where(equation < 0, precision, lower)
        upper = np.where(equation >= 0, precision, upper)
        if np.all(upper - lower <= 1e-10 * np.maximum(precision, np.finfo(float).tiny)):
            break
    else:
        raise LocalVIFactorError("local noise/covariance factors did not settle")
    precision = 0.5 * (lower + upper)
    denominator = 1 + precision[:, None] * eigenvalues
    basis = prior_root @ np.swapaxes(vectors, 1, 2)
    scaled = basis / np.sqrt(denominator)[:, None, :]
    covariance = scaled @ np.swapaxes(scaled, 1, 2)
    trace = np.sum(eigenvalues / denominator, axis=1)
    scale = precision / shape
    logdet_prior_precision = 2 * np.sum(np.log(np.diag(root_precision)))
    logdet_covariance = -logdet_prior_precision - np.sum(np.log(denominator), axis=1)
    delta = mean - prior_mean[None, :]
    parameter_kl = 0.5 * (
        np.sum(1 / denominator, axis=1)
        + np.einsum("ni,ij,nj->n", delta, prior_precision, delta)
        - n_parameters
        - logdet_prior_precision
        - logdet_covariance
    )
    rate = 1 / scale
    noise_kl = (
        (shape - shape0) * digamma(shape)
        - gammaln(shape)
        + gammaln(shape0)
        + shape0 * np.log(rate / rate0)
        + shape * (rate0 / rate - 1)
    )
    likelihood = 0.5 * (
        n_eff * (digamma(shape) + np.log(scale) - np.log(2 * np.pi))
        - precision * (rss + trace)
    )
    return LocalVIState(covariance, shape, scale, likelihood - parameter_kl - noise_kl)
