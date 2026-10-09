"""Optional voxel-specific BSW priors through the shared DSC spatial-VB wrapper."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import (
    DSCBatchedFitResult, _build_vi_update_prior_terms, batched_dsc_vi_fit,
)
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter, DSCVIPosterior

from .connection import (
    GuidedAcquisition,
    GuidedVoxel,
    _restore_fitted_signal,
    classify_result,
    fit_voxels,
)
from .batch import SignalBatchEvaluator
from .residue import signal_evaluator
from .leakage import BSWCenteredModel, BSWPrior, LeakageBatch, LeakageBatchEvaluator, readout_state
from .policy import model_for, select_valid_restarts
from .probability import Array, GaussianPrior
from .signal import MovingLayoutSignalCPI


def _join_results(rows: list[DSCBatchedFitResult]) -> DSCBatchedFitResult:
    """Assemble independent local updates in their unchanged spatial row order."""
    first = rows[0]

    def array(name: str) -> np.ndarray | None:
        values = [getattr(row, name) for row in rows]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("inconsistent optional local-update outputs")
        return np.concatenate(values)

    def mapping(name: str) -> dict[str, np.ndarray]:
        keys = getattr(first, name).keys()
        if any(getattr(row, name).keys() != keys for row in rows):
            raise ValueError("inconsistent local-update output schema")
        return {key: np.concatenate([getattr(row, name)[key] for row in rows]) for key in keys}

    posterior = None
    if any(row.full_posterior is not None for row in rows):
        if any(row.full_posterior is None for row in rows):
            raise ValueError("inconsistent full posterior outputs")
        posterior = DSCVIPosterior(*(
            np.concatenate([getattr(row.full_posterior, name) for row in rows])
            for name in ("mean", "covariance", "noise_shape", "noise_scale")
        ))
    return DSCBatchedFitResult(
        params=mapping("params"), r_squared=np.concatenate([r.r_squared for r in rows]),
        converged=np.concatenate([r.converged for r in rows]), fitted_curves=array("fitted_curves"),
        n_iterations=max(r.n_iterations for r in rows), model_name=first.model_name,
        uncertainties=mapping("uncertainties"), posterior_means_transformed=array("posterior_means_transformed"),
        posterior_cov_diag=array("posterior_cov_diag"),
        credible_interval_lower=mapping("credible_interval_lower"),
        credible_interval_upper=mapping("credible_interval_upper"), full_posterior=posterior,
        failure_reasons=tuple(
            reason for row in rows for reason in row.failure_reasons
        ),
    )


class RowPriorDispatcher:
    """Dispatch each synchronous local update with its own independent prior.

    Parameters
    ----------
    means, stds : ndarray
        Independent transformed prior vectors, shaped (voxel, parameter).
        This signal candidate uses identity parameter transforms.
    on_result : callable, optional
        Consume a complete single-voxel state and its actual conditional prior
        before its full covariance is released. Called in spatial row order.
    keep_full_posterior : bool, optional
        Retain full covariance across the field for small diagnostic checks.
        Normal connection keeps only marginal variances between sweeps.

    Notes
    -----
    This dispatches the same shared VI fitter; it has no optimization or spatial
    update loop of its own. Every row reads the same frozen sweep inputs.
    """

    def __init__(
        self, means: Array, stds: Array,
        on_result: Callable[[int, DSCBatchedFitResult, GaussianPrior], None] | None = None,
        *, keep_full_posterior: bool = False,
    ) -> None:
        if (means.ndim != 2 or means.shape != stds.shape or not len(means)
                or not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds))
                or np.any(stds <= 0)):
            raise ValueError("finite matching row priors with positive standard deviations required")
        self.means, self.stds = means.copy(), stds.copy()
        self.on_result, self.keep_full_posterior = on_result, keep_full_posterior
        self.calls = 0
        self.warm_means: Array | None = None
        self.last_prior_means = means.copy()
        self.last_prior_variances = stds**2

    def __call__(self, **kwargs: Any) -> DSCBatchedFitResult:
        """Preserve voxel prior/initial-state ownership through one shared update."""
        n, p = self.means.shape
        if (kwargs["tissue_curves"].shape[0] != n or len(kwargs["transforms"]) != p
                or any(t != TransformType.IDENTITY for t in kwargs["transforms"])
                or kwargs.get("mono_prec") is not None or kwargs.get("mono_eta") is not None):
            raise ValueError("dispatcher requires aligned signal rows and identity coordinates")
        sliced = ("tissue_curves", "nlls_seeds", "initial_means_transformed", "initial_cov_diag",
                  "initial_noise_precision", "spatial_prec_diag", "spatial_eta")
        rows = []
        for i in range(n):
            arguments = kwargs.copy()
            for name in sliced:
                if arguments.get(name) is not None:
                    arguments[name] = arguments[name][i:i + 1]
            arguments.update(prior_means_nat=self.means[i], prior_stds_nat=self.stds[i],
                             prior_means_transformed=self.means[i], prior_stds_transformed=self.stds[i],
                             return_full_posterior=True, return_posteriors=True,
                             save_fitted_curves=True)
            result = batched_dsc_vi_fit(**arguments)
            failure_reason = result.failure_reasons[0] if result.failure_reasons else None
            if failure_reason is not None:
                carry_mean = arguments.get("initial_means_transformed")
                if carry_mean is None:
                    carry_mean = arguments.get("nlls_seeds")
                if carry_mean is None:
                    carry_mean = self.means[i:i + 1]
                carry_cov = arguments.get("initial_cov_diag")
                if carry_cov is None:
                    carry_cov = self.stds[i:i + 1] ** 2
                carry_mean = np.asarray(carry_mean, dtype=np.float64)
                carry_cov = np.asarray(carry_cov, dtype=np.float64)
                posterior = result.full_posterior
                if posterior is not None:
                    noise_shape = np.where(
                        np.isfinite(posterior.noise_shape), posterior.noise_shape, 1.0,
                    )
                    noise_scale = np.where(
                        np.isfinite(posterior.noise_scale), posterior.noise_scale, 1.0,
                    )
                    posterior = DSCVIPosterior(
                        carry_mean.copy(),
                        np.asarray([np.diag(carry_cov[0])], dtype=np.float64),
                        noise_shape,
                        noise_scale,
                    )
                result = replace(
                    result,
                    posterior_means_transformed=carry_mean.copy(),
                    posterior_cov_diag=carry_cov.copy(),
                    full_posterior=posterior,
                )
            precision, eta = _build_vi_update_prior_terms(
                np.diag(1 / self.stds[i]**2), self.means[i], arguments.get("mono_prec"),
                arguments.get("mono_eta"),
                spatial_replaces_parameter_prior=arguments.get("spatial_replaces_parameter_prior", False),
                spatial_parameter_mask=arguments.get("spatial_parameter_mask"),
            )
            if arguments.get("spatial_prec_diag") is not None:
                precision += np.diag(arguments["spatial_prec_diag"][0])
            if arguments.get("spatial_eta") is not None:
                eta += arguments["spatial_eta"][0]
            prior = GaussianPrior(np.asarray(np.linalg.solve(precision, eta), dtype=np.float64),
                                   np.asarray(np.linalg.inv(precision), dtype=np.float64))
            self.last_prior_means[i] = prior.mean
            self.last_prior_variances[i] = np.diag(prior.covariance)
            if self.on_result is not None:
                self.on_result(i, result, prior)
            if not self.keep_full_posterior:
                result = replace(result, full_posterior=None)
            rows.append(result)
        joined = _join_results(rows)
        if self.calls == 0:
            assert joined.posterior_means_transformed is not None
            self.warm_means = np.array(joined.posterior_means_transformed, dtype=np.float64)
        self.calls += 1
        return joined


@dataclass(frozen=True)
class SpatialGuidedResult:
    """Field outputs, actual local priors and schedule provenance; not global evidence."""

    voxels: tuple[GuidedVoxel, ...]
    spatial_indices: np.ndarray
    warm_means: Array | None
    posterior_means: Array | None
    conditional_prior_means: Array | None
    conditional_prior_variances: Array | None
    local_calls: int
    spatial_sweeps: int
    weights: Array | None = None


def fit_spatial_voxels(
    acquisition: GuidedAcquisition, *, keep_diagnostics: bool = False,
) -> SpatialGuidedResult:
    """Fit the complete eligible image field with the existing CBF-only M-prior.

    Parameters
    ----------
    acquisition : GuidedAcquisition
        Complete prepared acquisition explicitly requesting svb. Ordinary
        multi-start VI selects one endpoint per voxel before spatial fitting.
        Missing/invalid/fallback-requested ordinary fits do not enter the graph.
    keep_diagnostics : bool, optional
        Retain each final full state for small qualification fields. False
        assesses full covariance per voxel then releases it, retaining scalar
        assessment and the marginal arrays needed by the shared spatial loop.

    Returns
    -------
    SpatialGuidedResult
        Image-indexed outputs and conditional-prior/schedule provenance.
        No independent spatial chunking or alternate fallback is performed.
    """
    if acquisition.inference_method != "svb":
        raise ValueError("spatial fitting requires an explicit svb acquisition")
    a = acquisition
    outputs: dict[int, GuidedVoxel] = {}
    indices, seeds, prior_means, prior_stds = [], [], [], []
    for row in fit_voxels(replace(a, inference_method="vi")):
        if row.status in ("retained", "provisional"):
            assert row.diagnostics is not None
            chosen = row.diagnostics.rows[0]["selected"]
            indices.append(row.flat_index)
            seeds.append(chosen.mean)
            prior_means.append(row.diagnostics.arrays["prior_mean"])
            prior_stds.append(np.sqrt(np.diag(row.diagnostics.arrays["prior_covariance"])))
        outputs[row.flat_index] = row if keep_diagnostics else replace(row, diagnostics=None)
    index_array = np.asarray(indices, dtype=int)
    if not indices:
        return SpatialGuidedResult(tuple(outputs.values()), index_array, None, None, None, None, 0, 0,
                                   a.weights)
    assert a.decision is not None
    count = a.decision.selected
    first = indices[0]
    vascular = model_for(a.signals[first], a.aif, a.time, a.te, count, family=a.family,
                         infer_delay=a.infer_delay)
    model: BSWCenteredModel | MovingLayoutSignalCPI
    evaluator: LeakageBatchEvaluator | SignalBatchEvaluator
    if a.leakage_method == "bsw":
        assert a.reference is not None
        model = BSWCenteredModel(vascular, a.reference, BSWPrior(a.k2[first], a.k2_sd[first]))
        evaluator = LeakageBatchEvaluator(model)
    else:
        model = vascular
        evaluator = signal_evaluator(model)
    adapter = DSCVIModelAdapter(evaluator.predictions, evaluator.jacobians,
                               lambda m: np.asarray([model.valid_mean(x) for x in m], dtype=bool))
    p = len(model.prior.mean)

    def consume(i: int, result: DSCBatchedFitResult, prior: GaussianPrior) -> None:
        """Assess the actual conditional local state before releasing its covariance."""
        posterior, prediction = result.full_posterior, result.fitted_curves
        assert posterior is not None and prediction is not None
        failure_reason = result.failure_reasons[0] if result.failure_reasons else None
        if failure_reason is not None:
            outputs[indices[i]] = GuidedVoxel(
                indices[i], "fallback_requested", (failure_reason,),
                bsw_k2=float(a.k2[indices[i]]), family=a.family,
                leakage_method=a.leakage_method,
            )
            return
        jacobian = evaluator.jacobians(posterior.mean)
        fit, assessment, leakage = readout_state(
            model, a.signals[indices[i]], posterior.mean[0], posterior.covariance[0],
            float(posterior.noise_shape[0]), float(posterior.noise_scale[0]),
            prediction[0], jacobian[0], count, 0, bool(result.converged[0]),
            weights=a.weights, prior=prior,
        )
        arrays = dict(
            mean=posterior.mean, covariance=posterior.covariance, noise_shape=posterior.noise_shape,
            noise_scale=posterior.noise_scale, fitted_curves=prediction, jacobian=jacobian,
            prior_mean=prior.mean, prior_covariance=prior.covariance,
        )
        if keep_diagnostics and a.weights is not None:
            arrays["weights"] = a.weights
        diagnostic = LeakageBatch([fit], [assessment], select_valid_restarts([fit], [assessment], 1),
                                   [leakage] if a.leakage_method == "bsw" else [], arrays, 0.,
                                   result.n_iterations, a.leakage_method, a.infer_delay)
        row = classify_result(indices[i], float(a.k2[indices[i]]), diagnostic)
        row = replace(_restore_fitted_signal(row, a.signal_scale), family=a.family)
        outputs[indices[i]] = row if keep_diagnostics else replace(row, diagnostics=None)

    dispatcher = RowPriorDispatcher(np.asarray(prior_means), np.asarray(prior_stds), consume)
    coordinates = np.unravel_index(index_array, a.spatial_shape)
    shape = (a.spatial_shape[0], a.spatial_shape[1], a.spatial_shape[2])
    result = batched_dsc_spatial_vi_fit(
        model_type="signal_cpi_leakage" if a.leakage_method == "bsw" else "signal_cpi",
        tissue_curves=a.signals[index_array], aif=a.aif,
        time_points=a.time, mask_coords=(coordinates[0], coordinates[1], coordinates[2]),
        spatial_shape=shape, prior_means_nat=model.prior.mean,
        prior_stds_nat=np.sqrt(np.diag(model.prior.covariance)),
        prior_means_transformed=model.prior.mean,
        prior_stds_transformed=np.sqrt(np.diag(model.prior.covariance)),
        transforms=[TransformType.IDENTITY] * p, upper_bounds=[None] * p,
        fitting_param_names=["cbf", "s0"] + [f"coordinate_{i}" for i in range(2, p)],
        fitting_bounds=list(zip(model.lower, model.upper, strict=True)), nlls_seeds=np.asarray(seeds),
        weights=a.weights, model_adapter=adapter, save_fitted_curves=True, local_fit=dispatcher,
        sloppy=a.sloppy,
    )
    posterior_means = (None if result.posterior_means_transformed is None
                       else np.asarray(result.posterior_means_transformed, dtype=np.float64))
    return SpatialGuidedResult(tuple(outputs[i] for i in sorted(outputs)), index_array,
                               dispatcher.warm_means, posterior_means,
                               dispatcher.last_prior_means, dispatcher.last_prior_variances,
                               dispatcher.calls, max(0, dispatcher.calls - 1), a.weights)
