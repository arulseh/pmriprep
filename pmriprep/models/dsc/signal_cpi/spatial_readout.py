"""Worker-owned readout of the final conditional spatial CPI posterior."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from pmriprep.models.dsc.batched_fitting import DSCBatchedFitResult
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter

from .connection import GuidedVoxel, _restore_fitted_signal, classify_result
from .geometry import Model
from .leakage import BSWCenteredModel, LeakageBatch, readout_state
from .policy import select_valid_restarts
from .probability import GaussianPrior
from .signal import MovingLayoutSignalCPI


@dataclass(frozen=True)
class SpatialReadout:
    """One voxel's output identity and shared model, without a full image buffer.

    The actual signal and conditional prior accompany the local update. Full
    factors are released after readout unless diagnostics are explicitly kept.
    """

    model: BSWCenteredModel | MovingLayoutSignalCPI
    flat_index: int
    order: int
    bsw_k2: float
    family: Model
    selection_scores: tuple[tuple[str, float | None], ...]
    leakage_method: Literal["bsw", "none"]
    infer_delay: bool
    weights: np.ndarray | None
    keep_diagnostics: bool
    signal_scale: float

    def __call__(
        self, result: DSCBatchedFitResult, prior: GaussianPrior,
        adapter: DSCVIModelAdapter, signal: np.ndarray,
    ) -> GuidedVoxel:
        """Read the complete local state using the worker's existing evaluator.

        Parameters
        ----------
        result, prior
            Local update and the conditional prior used for that update.
        adapter : DSCVIModelAdapter
            Worker-local evaluator, with any final-mean derivative cache intact.
        signal : ndarray
            Internally normalized measured signal for this voxel.

        Returns
        -------
        GuidedVoxel
            Assessed measurement or explicit local numerical-failure request.
        """
        failure = result.failure_reasons[0] if result.failure_reasons else None
        if failure is not None:
            return GuidedVoxel(
                self.flat_index, "fallback_requested", (failure,), bsw_k2=self.bsw_k2,
                family=self.family, selection_scores=self.selection_scores,
                leakage_method=self.leakage_method,
            )
        posterior, prediction = result.full_posterior, result.fitted_curves
        assert posterior is not None and prediction is not None
        jacobian = adapter.jacobian(posterior.mean)
        fit, assessment, inferred = readout_state(
            self.model, signal, posterior.mean[0], posterior.covariance[0],
            float(posterior.noise_shape[0]), float(posterior.noise_scale[0]),
            prediction[0], jacobian[0], self.order, 0, bool(result.converged[0]),
            weights=self.weights, prior=prior,
        )
        arrays = dict(mean=posterior.mean, covariance=posterior.covariance,
            noise_shape=posterior.noise_shape, noise_scale=posterior.noise_scale,
            fitted_curves=prediction, jacobian=jacobian, prior_mean=prior.mean,
            prior_covariance=prior.covariance)
        if self.weights is not None:
            arrays["weights"] = self.weights
        diagnostic = LeakageBatch(
            [fit], [assessment], select_valid_restarts([fit], [assessment], 1),
            [inferred] if self.leakage_method == "bsw" else [], arrays, 0.,
            result.n_iterations, self.leakage_method, self.infer_delay,
        )
        row = replace(classify_result(self.flat_index, self.bsw_k2, diagnostic),
                      family=self.family, selection_scores=self.selection_scores)
        row = _restore_fitted_signal(row, self.signal_scale)
        return row if self.keep_diagnostics else replace(row, diagnostics=None)
