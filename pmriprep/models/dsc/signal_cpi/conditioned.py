"""Fixed-arrival conditioning without a zero-width inference coordinate."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly

from .batch import SignalBatchEvaluator
from .probability import GaussianPrior
from .residue import PhysicalSignalCPI, PhysicalSignalEvaluator
from .signal import MovingLayoutSignalCPI


class FixedDelaySignalCPI(MovingLayoutSignalCPI):
    """Condition arrival at zero, retaining all other coordinates and priors.

    Parameters
    ----------
    full : PhysicalSignalCPI
        Unconditioned model for the same acquisition and residue family.

    Notes
    -----
    The posterior contains only inferred coordinates. There is no singular
    zero-variance delay row, and the delay prior contributes no score term.
    """

    def __init__(self, full: PhysicalSignalCPI) -> None:
        self.full = full
        self.keep = np.delete(np.arange(full.prior.mean.size), full.delay_index)
        self.prior = GaussianPrior(full.prior.mean[self.keep],
                                   full.prior.covariance[np.ix_(self.keep, self.keep)])
        self.lower, self.upper = full.lower[self.keep], full.upper[self.keep]
        self.observed, self.time = full.observed, full.time
        self.dt, self.te, self.duration = full.dt, full.te, full.duration
        self.model, self.n_points = full.model, full.n_points
        self.continuous, self.dispersion = full.continuous, full.dispersion
        self.delay_index = -1

    def expand(self, means: np.ndarray) -> np.ndarray:
        """Restore the fixed coordinate for scalar or batched forward evaluation."""
        if means.shape[-1:] != self.prior.mean.shape:
            raise ValueError("fixed-delay coordinates do not match the model")
        result = np.zeros((*means.shape[:-1], self.full.prior.mean.size))
        result[..., self.keep] = means
        return result

    def starts(self) -> tuple[np.ndarray, ...]:
        """Preserve start identities while removing the inferred delay."""
        return tuple(mean[self.keep] for mean in self.full.starts())

    def valid_mean(self, mean: np.ndarray) -> bool:
        """Check the full physical domain after restoring zero arrival."""
        return mean.shape == self.prior.mean.shape and self.full.valid_mean(self.expand(mean))

    def derivative_domain(self, mean: np.ndarray) -> bool:
        """Retain ordered-spacing differentiation with delay conditioned at zero."""
        return mean.shape == self.prior.mean.shape and self.full.derivative_domain(self.expand(mean))

    def knot_times(self, mean: np.ndarray) -> np.ndarray:
        """Read intrinsic knot times in their original coordinate order."""
        return self.full.knot_times(self.expand(mean))

    def knot_values(self, mean: np.ndarray) -> np.ndarray:
        """Read intrinsic amplitudes independently of fixed arrival."""
        return self.full.knot_values(self.expand(mean))

    def residue(self, mean: np.ndarray) -> PPoly:
        """Use the same modified polynomial for fixed-arrival perfusion."""
        return self.full.residue(self.expand(mean))

    def concentration(self, mean: np.ndarray) -> np.ndarray:
        """Predict vascular concentration at the supplied zero arrival."""
        return self.full.concentration(self.expand(mean))

    def arrival_delay(self, mean: np.ndarray) -> float:
        """Return the conditioning value, not an estimated timing measurement."""
        return 0.0


class FixedDelayEvaluator(SignalBatchEvaluator):
    """Project complete physical predictions/Jacobians onto inferred coordinates."""

    def __init__(self, model: FixedDelaySignalCPI) -> None:
        self.conditioned = model
        self.model = model
        self.full_evaluator = PhysicalSignalEvaluator(model.full)

    def clear_transient_state(self) -> None:
        """Discard the complete evaluator's transient response history."""
        self.full_evaluator.clear_transient_state()

    def concentration(self, means: np.ndarray) -> np.ndarray:
        """Evaluate without a timing parameter in the fitted state."""
        return self.full_evaluator.concentration(self.conditioned.expand(means))

    def predictions(self, means: np.ndarray) -> np.ndarray:
        """Restore zero arrival only for forward evaluation."""
        return self.full_evaluator.predictions(self.conditioned.expand(means))

    def jacobians(self, means: np.ndarray) -> np.ndarray:
        """Remove the delay column, retaining all inferred cross-covariances."""
        return self.full_evaluator.jacobians(self.conditioned.expand(means))[:, :, self.conditioned.keep]
