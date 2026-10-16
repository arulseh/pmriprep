"""Model interfaces for the shared batched DSC variational fitter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pmriprep.models.bayesian_utils import TransformType, from_transformed_batch

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DSCVIModelAdapter:
    """Supply model evaluations without replacing the variational update loop.

    Parameters
    ----------
    predict : callable
        Natural-coordinate parameter rows (N, P) to observations (N, T).
    jacobian : callable
        Natural-coordinate derivatives with shape (N, T, P). The fitter
        applies the coordinate-transform chain rule.
    valid_mean : callable, optional
        Return one boolean per natural-coordinate row. The domain must be
        convex along transformed-coordinate segments between feasible means.
        This constrains fitted means, not the support of a Gaussian posterior.
    reset_state : callable, optional
        Clear transient evaluator state before an independent fit. This keeps
        process payloads and results independent of earlier fitted rows.
    """

    predict: Callable[[Array], Array]
    jacobian: Callable[[Array], Array]
    valid_mean: Callable[[Array], npt.NDArray[np.bool_]] | None = None
    reset_state: Callable[[], None] | None = None

    def reset_transient_state(self) -> None:
        """Discard evaluator caches without changing the model or its priors."""
        if self.reset_state is not None:
            self.reset_state()

    def admissible(self, parameters: Array) -> npt.NDArray[np.bool_]:
        """Validate the domain callback's shape and boolean result."""
        finite: npt.NDArray[np.bool_] = np.asarray(
            np.all(np.isfinite(parameters), axis=1), dtype=np.bool_
        )
        if self.valid_mean is None:
            return finite
        valid = np.asarray(self.valid_mean(parameters))
        if valid.shape != finite.shape or valid.dtype != np.bool_:
            raise ValueError("model domain must return one boolean per row")
        return np.asarray(finite & valid, dtype=np.bool_)


@dataclass(frozen=True)
class DSCVIPosterior:
    """Optional complete local factors in fitting coordinates.

    Covariance has shape (N, P, P), not cross-voxel covariance. Noise precision
    uses Gamma shape/scale. Unfitted rows contain NaN. These factors are local
    Gaussian approximations, including at constrained means.
    """

    mean: Array
    covariance: Array
    noise_shape: Array
    noise_scale: Array


def feasible_mean_update(
    previous: Array,
    proposed: Array,
    adapter: DSCVIModelAdapter,
    transforms: list[TransformType],
    upper_bounds: list[float | None],
) -> Array:
    """Backtrack only infeasible proposals toward their feasible current means.

    Parameters
    ----------
    previous, proposed : ndarray
        Current and proposed transformed means, shape (N, P).
    adapter : DSCVIModelAdapter
        Model domain check in natural coordinates.
    transforms, upper_bounds : list
        Coordinate-transform definitions used by the shared fitter.

    Returns
    -------
    ndarray
        Feasible transformed proposals; already feasible rows are unchanged.
        Rows with no representable feasible step retain the previous mean.

    Raises
    ------
    ValueError
        If the proposed mean is nonfinite or the current mean is inadmissible.
    """
    if not np.all(np.isfinite(proposed)):
        raise ValueError("non-finite variational mean proposal")

    def natural(values: Array) -> Array:
        result = np.empty_like(values)
        for j, transform in enumerate(transforms):
            result[:, j] = from_transformed_batch(
                values[:, j], transform, upper_bounds[j]
            )
        return result

    if not np.all(adapter.admissible(natural(previous))):
        raise ValueError("current mean violates the model domain")
    candidate = proposed.copy()
    for _ in range(64):
        valid = adapter.admissible(natural(candidate))
        if np.all(valid):
            return candidate
        candidate[~valid] = previous[~valid] + 0.5 * (
            candidate[~valid] - previous[~valid]
        )
    valid = adapter.admissible(natural(candidate))
    candidate[~valid] = previous[~valid]
    return candidate
