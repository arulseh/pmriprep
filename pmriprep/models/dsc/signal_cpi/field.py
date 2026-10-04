"""One CBF spatial graph across selected CPI transport families."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable

import numpy as np

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import DSCBatchedFitResult
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter

from .acquisition import CPIAcquisition, fit_selected_voxels
from .batch import SignalBatchEvaluator
from .connection import GuidedVoxel
from .geometry import Model
from .leakage import BSWCenteredModel, BSWPrior, LeakageBatchEvaluator
from .policy import model_for
from .probability import GaussianPrior
from .progress import _CPIProgress
from .residue import signal_evaluator
from .signal import MovingLayoutSignalCPI
from .spatial import RowPriorDispatcher, SpatialGuidedResult, _join_results
from .spatial_readout import SpatialReadout
from .execution import CPIExecutor


@dataclass(frozen=True)
class _SpatialTask:
    """One local update with optional final readout and factor retention."""

    arguments: dict[str, Any]
    readout: SpatialReadout | None
    keep_full_posterior: bool


@dataclass(frozen=True)
class _SpatialUpdate:
    """Marginal spatial state and an optional worker-computed output voxel."""

    result: DSCBatchedFitResult
    prior: GaussianPrior
    voxel: GuidedVoxel | None


def _fit_spatial_row(task: _SpatialTask) -> _SpatialUpdate:
    """Run one frozen-neighbor update with the unchanged shared row dispatcher."""
    arguments = task.arguments
    arguments["model_adapter"].reset_transient_state()
    captured: list[tuple[DSCBatchedFitResult, GaussianPrior]] = []

    def consume(_: int, result: DSCBatchedFitResult, prior: GaussianPrior) -> None:
        captured.append((result, prior))

    dispatch = RowPriorDispatcher(arguments["prior_means_transformed"][None],
                                  arguments["prior_stds_transformed"][None], consume)
    dispatch(**arguments)
    result, prior = captured[0]
    voxel = (None if task.readout is None else task.readout(
        result, prior, arguments["model_adapter"], arguments["tissue_curves"][0]))
    if not task.keep_full_posterior:
        result = replace(result, full_posterior=None)
    return _SpatialUpdate(result, prior, voxel)


class CommonCBFDispatcher:
    """Expose only CBF to the shared spatial schedule; retain private local states.

    Parameters
    ----------
    arguments : list of dict
        Per-row shared-VI arguments with one signal, prior, model and seed each.
        The first coordinate must be CBF in identity coordinates. Other
        coordinates may differ in number and meaning between rows.
    on_result : callable, optional
        Consume the complete local posterior and actual conditional prior.
    executor : CPIExecutor, optional
        Parallelize local updates, retaining one synchronous whole-field sweep.
    readouts : list of SpatialReadout, optional
        Final-pass output contexts. Use begin_pass with the fixed spatial
        schedule to evaluate these only on its last pass.
    on_readout : callable, optional
        Consume a final worker-owned voxel without recalculating its factors.

    Notes
    -----
    All neighbor contributions are computed by the existing spatial wrapper
    before this dispatcher runs. Local fits never consume a neighbor updated
    in the same sweep. Only marginal state is stored between sweeps, as in
    the shared schedule; full covariance is available during the callback.
    """

    def __init__(
        self, arguments: list[dict[str, Any]],
        on_result: Callable[[int, DSCBatchedFitResult, GaussianPrior], None] | None = None,
        *, executor: CPIExecutor | None = None,
        readouts: list[SpatialReadout] | None = None,
        on_readout: Callable[[GuidedVoxel], None] | None = None,
    ) -> None:
        if not arguments:
            raise ValueError("at least one local model required")
        self.arguments = arguments
        self.executor = executor
        if ((readouts is None) != (on_readout is None)
                or (readouts is not None and len(readouts) != len(arguments))):
            raise ValueError("one readout per row and a matching consumer required")
        self.on_result = on_result
        self.readouts, self.on_readout = readouts, on_readout
        self.readout_enabled = False
        self.progress: _CPIProgress | None = None
        self.dispatchers = []
        for args in arguments:
            p = len(args["fitting_param_names"])
            if (args["fitting_param_names"][0] != "cbf"
                    or args["transforms"] != [TransformType.IDENTITY] * p
                    or args["tissue_curves"].shape[0] != 1):
                raise ValueError("one identity-coordinate row beginning with CBF required")
            self.dispatchers.append(RowPriorDispatcher(
                args["prior_means_transformed"][None], args["prior_stds_transformed"][None]))
        self.states: list[dict[str, np.ndarray]] = []
        self.warm_means: np.ndarray | None = None
        self.calls = 0

    def begin_pass(self, index: int, sweeps: int) -> None:
        """Select final-pass readout and start a fresh voxel progress counter.

        Parameters
        ----------
        index, sweeps : int
            Zero denotes warm start, followed by sweeps one through sweeps.
            Both zero denotes the independent-VI pass on a graph without edges.
        """
        self.readout_enabled = index == sweeps
        label = (f"SVB sweep {index}/{sweeps}" if index else
                 "SVB warm start" if sweeps else "SVB ordinary VI")
        self.progress = _CPIProgress(len(self.arguments), label=label)

    def __call__(self, **kwargs: Any) -> DSCBatchedFitResult:
        """Run the shared local updater with this sweep's frozen CBF inputs."""
        n = len(self.arguments)
        if kwargs["fitting_param_names"] != ["cbf"] or len(kwargs["tissue_curves"]) != n:
            raise ValueError("spatial schedule must expose the common CBF field only")
        rows, next_states, tasks = [], [], []
        for i, local in enumerate(self.arguments):
            args = kwargs | local
            p = len(local["fitting_param_names"])
            for name in ("spatial_prec_diag", "spatial_eta"):
                if kwargs.get(name) is not None:
                    args[name] = np.zeros((1, p))
                    args[name][0, 0] = kwargs[name][i, 0]
            if kwargs.get("spatial_parameter_mask") is not None:
                args["spatial_parameter_mask"] = np.arange(p) == 0
            if self.calls:
                args.update(self.states[i], nlls_seeds=None)
                # The common state must be the same frozen state held privately.
                for name in ("initial_means_transformed", "initial_cov_diag"):
                    if not np.array_equal(args[name][:, :1], kwargs[name][i:i+1]):
                        raise ValueError("spatial and local CBF states are misaligned")
            readout = self.readouts[i] if self.readouts is not None and self.readout_enabled else None
            tasks.append(_SpatialTask(args, readout, self.on_result is not None))
        for i, update in enumerate(self._results(tasks)):
            result, prior = update.result, update.prior
            dispatch = self.dispatchers[i]
            dispatch.last_prior_means[0] = prior.mean
            dispatch.last_prior_variances[0] = np.diag(prior.covariance)
            if self.on_result is not None:
                self.on_result(i, result, prior)
            if update.voxel is not None:
                assert self.on_readout is not None
                self.on_readout(update.voxel)
            if dispatch.calls == 0:
                assert result.posterior_means_transformed is not None
                dispatch.warm_means = np.array(result.posterior_means_transformed, dtype=np.float64)
            dispatch.calls += 1
            if self.progress is not None:
                self.progress.advance()
            mean, variance = result.posterior_means_transformed, result.posterior_cov_diag
            assert mean is not None and variance is not None
            noise = result.uncertainties["noise_std"]
            precision = np.zeros_like(noise)
            valid = np.isfinite(noise) & (noise > 0)
            precision[valid] = 1 / noise[valid]**2
            next_states.append(dict(initial_means_transformed=mean,
                initial_cov_diag=variance, initial_noise_precision=precision))
            rows.append(replace(result, params={"cbf": mean[:, 0]},
                uncertainties={"noise_std": noise}, posterior_means_transformed=mean[:, :1],
                posterior_cov_diag=variance[:, :1], credible_interval_lower={},
                credible_interval_upper={}, full_posterior=None))
        self.states = next_states
        combined = _join_results(rows)
        if self.calls == 0:
            assert combined.posterior_means_transformed is not None
            self.warm_means = combined.posterior_means_transformed.copy()
        self.calls += 1
        return combined

    def _results(self, tasks: list[_SpatialTask]) -> Iterator[_SpatialUpdate]:
        """Consume local updates in spatial order with bounded worker payloads."""
        def clean_tasks() -> Iterator[_SpatialTask]:
            for task in tasks:
                # The executor serializes lazily. Clear the shared evaluator
                # immediately before each snapshot so caches never travel with
                # a per-voxel task.
                task.arguments["model_adapter"].reset_transient_state()
                yield task

        if self.executor is None:
            yield from map(_fit_spatial_row, clean_tasks())
        else:
            yield from self.executor.map(_fit_spatial_row, clean_tasks())


def fit_selected_field(
    acquisition: CPIAcquisition, *, keep_diagnostics: bool = False,
    executor: CPIExecutor | None = None,
    on_selection_result: Callable[[], None] | None = None,
) -> SpatialGuidedResult:
    """Select by ordinary VI, then share CBF spatial information across families.

    Parameters
    ----------
    acquisition : CPIAcquisition
        Complete image, explicitly requesting SVB. Family selection precedes
        spatial inference and is not reranked using local conditional scores.
    keep_diagnostics : bool, optional
        Retain full final local posterior states for small diagnostic fields.
    executor : CPIExecutor, optional
        Reuse the pool for ordinary selection and synchronous local updates.
    on_selection_result : callable, optional
        Controller-side callback after each initial independent voxel fit.

    Returns
    -------
    SpatialGuidedResult
        Image-indexed measurements and common-CBF schedule provenance. Private
        parameters are available in per-voxel diagnostics, not padded arrays.
    """
    if any(branch.inference_method != "svb" for branch in acquisition.branches):
        raise ValueError("complete field must explicitly request svb")
    a = acquisition.branches[0]
    branches = {b.family: b for b in acquisition.branches}
    outputs, indices, arguments, readouts = {}, [], [], []
    shared: dict[Model, tuple[BSWCenteredModel | MovingLayoutSignalCPI,
                             LeakageBatchEvaluator | SignalBatchEvaluator, DSCVIModelAdapter]] = {}
    selected = (
        fit_selected_voxels(acquisition, on_result=on_selection_result)
        if executor is None
        else fit_selected_voxels(
            acquisition,
            executor=executor,
            on_result=on_selection_result,
        )
    )
    for row in selected:
        outputs[row.flat_index] = row if keep_diagnostics else replace(row, diagnostics=None)
        if row.status not in ("retained", "provisional"):
            continue
        assert row.family is not None and row.diagnostics is not None
        branch = branches[row.family]
        assert branch.decision is not None
        i = row.flat_index
        if row.family not in shared:
            vascular = model_for(a.signals[i], a.aif, a.time, a.te, branch.decision.selected,
                                 family=row.family, infer_delay=a.infer_delay)
            model: BSWCenteredModel | MovingLayoutSignalCPI = vascular
            if a.leakage_method == "bsw":
                assert a.reference is not None
                model = BSWCenteredModel(vascular, a.reference,
                                         BSWPrior(float(a.k2[i]), float(a.k2_sd[i])))
            evaluator = LeakageBatchEvaluator(model) if isinstance(model, BSWCenteredModel) else signal_evaluator(model)
            adapter = DSCVIModelAdapter(evaluator.predictions, evaluator.jacobians,
                                       partial(_admissible, model), evaluator.clear_transient_state)
            shared[row.family] = (model, evaluator, adapter)
        model, evaluator, adapter = shared[row.family]
        mean = row.diagnostics.arrays["prior_mean"]
        std = np.sqrt(np.diag(row.diagnostics.arrays["prior_covariance"]))
        p = len(mean)
        arguments.append(dict(model_type="signal_cpi_leakage" if a.leakage_method == "bsw" else "signal_cpi",
            tissue_curves=a.signals[i:i+1], prior_means_nat=mean, prior_stds_nat=std,
            prior_means_transformed=mean, prior_stds_transformed=std,
            transforms=[TransformType.IDENTITY] * p, upper_bounds=[None] * p,
            fitting_param_names=["cbf", "s0"] + [f"coordinate_{j}" for j in range(2, p)],
            fitting_bounds=list(zip(model.lower, model.upper, strict=True)),
            nlls_seeds=np.asarray([row.diagnostics.rows[0]["selected"].mean]), model_adapter=adapter))
        indices.append(i)
        readouts.append(SpatialReadout(model, i, branch.decision.selected, float(a.k2[i]),
            row.family, row.selection_scores, a.leakage_method, a.infer_delay,
            a.weights, keep_diagnostics))
    index_array = np.asarray(indices, dtype=int)
    if not indices:
        return SpatialGuidedResult(tuple(outputs.values()), index_array, None, None, None, None, 0, 0,
                                   a.weights)

    final_outputs: dict[int, GuidedVoxel] = {}

    def consume(row: GuidedVoxel) -> None:
        final_outputs[row.flat_index] = row

    dispatch = CommonCBFDispatcher(arguments, executor=executor,
                                   readouts=readouts, on_readout=consume)
    coordinates = np.unravel_index(index_array, a.spatial_shape)
    result = batched_dsc_spatial_vi_fit(model_type="signal_cpi", tissue_curves=a.signals[index_array],
        aif=a.aif, time_points=a.time, mask_coords=(coordinates[0], coordinates[1], coordinates[2]),
        spatial_shape=(a.spatial_shape[0], a.spatial_shape[1], a.spatial_shape[2]),
        prior_means_nat=arguments[0]["prior_means_nat"][:1],
        prior_stds_nat=arguments[0]["prior_stds_nat"][:1],
        transforms=[TransformType.IDENTITY], upper_bounds=[None], fitting_param_names=["cbf"],
        fitting_bounds=arguments[0]["fitting_bounds"][:1], weights=a.weights,
        save_fitted_curves=True, local_fit=dispatch, sloppy=a.sloppy,
        on_pass_start=dispatch.begin_pass)
    if set(final_outputs) != set(indices):
        raise RuntimeError("spatial schedule did not produce every final voxel readout")
    outputs.update(final_outputs)
    return SpatialGuidedResult(tuple(outputs[i] for i in sorted(outputs)), index_array,
        dispatch.warm_means, (None if result.posterior_means_transformed is None else
                             np.asarray(result.posterior_means_transformed, dtype=np.float64)),
        np.concatenate([d.last_prior_means[:, :1] for d in dispatch.dispatchers]),
        np.concatenate([d.last_prior_variances[:, :1] for d in dispatch.dispatchers]),
        dispatch.calls, max(0, dispatch.calls - 1), a.weights)


def _admissible(model: BSWCenteredModel | MovingLayoutSignalCPI, means: np.ndarray) -> np.ndarray:
    """Keep the domain callback bound to its own model rather than a loop local."""
    return np.asarray([model.valid_mean(m) for m in means], dtype=bool)
