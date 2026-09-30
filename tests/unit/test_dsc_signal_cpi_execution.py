"""CPI process limits preserve voxel ownership, numerical fits and spatial edges."""

from dataclasses import dataclass, replace
from concurrent.futures import Future
import os
from pathlib import Path
import pickle
import time as clock
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from pmriprep.models.dsc.signal_cpi import acquisition, execution, field, maps
from pmriprep.models.dsc.signal_cpi.connection import GuidedVoxel
from pmriprep.models.dsc.signal_cpi.settings import CPIOptions
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter
from tests.unit.test_dsc_cpi_production import guidance_for
from tests.unit.test_dsc_signal_cpi_field import common_arguments
from tests.unit.test_dsc_signal_cpi_spatial import coordinates, linear_arguments


def _worker_info(value: int) -> tuple[int, int, list[int]]:
    """Force overlapping jobs and inspect the actual numerical library limits."""
    clock.sleep(.1)
    return value, os.getpid(), [p["num_threads"] for p in threadpool_info()]


def _failure(value: int) -> int:
    """Propagate one failing voxel rather than silently substituting a fit."""
    if value == 0:
        raise RuntimeError("owned worker failure")
    clock.sleep(2)
    return value


def _resources(n: int) -> execution.CPIResources:
    """Resolve limits for a small field without a scheduler memory cap."""
    return execution.plan_resources(n, 8, 30, 5)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_actual_workers_are_bounded_and_restore_parent_threads(
    workers: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No nested numerical threading; output order does not depend on completion."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    with threadpool_limits(limits=2):
        before = threadpool_info()
        with execution.CPIExecutor(_resources(workers)) as executor:
            result = list(executor.map(_worker_info, range(2 * workers)))
            children = [] if executor.pool is None else list(executor.pool._processes.values())
        assert [r[0] for r in result] == list(range(2 * workers))
        pids = {r[1] for r in result}
        assert len(pids) == workers
        assert (os.getpid() in pids) == (workers == 1)
        assert all(n == 1 for _, _, threads in result for n in threads)
        assert not any(p.is_alive() for p in children)
        assert threadpool_info() == before


@pytest.mark.parametrize("ordered", [True, False])
def test_failed_and_abandoned_work_reaps_workers(monkeypatch: pytest.MonkeyPatch, ordered: bool) -> None:
    """A real worker exception or closed iterator cannot leave fitting running."""
    terminated = []
    original = execution.terminate_process_pool

    def terminate(pool: Any) -> None:
        terminated.extend(pool._processes.values())
        original(pool)

    monkeypatch.setattr(execution, "terminate_process_pool", terminate)
    for fails in (True, False):
        executor = execution.CPIExecutor(_resources(2))
        children = []
        with executor:
            iterator = executor.map(_failure if fails else _worker_info, range(6), ordered=ordered)
            if fails:
                with pytest.raises(RuntimeError, match="owned worker failure"):
                    next(iterator)
            else:
                next(iterator)
                assert executor.pool is not None
                children = list(executor.pool._processes.values())
                iterator.close()
            assert executor.pool is None
        assert not any(p.is_alive() for p in children)
    assert terminated and not any(p.is_alive() for p in terminated)


def test_submission_is_bounded() -> None:
    """Consuming one result must not eagerly enqueue the complete image."""
    produced = []

    def values() -> Any:
        for i in range(50):
            produced.append(i)
            yield i

    with execution.CPIExecutor(_resources(2)) as executor:
        iterator = executor.map(_worker_info, values())
        next(iterator)
        assert len(produced) == 2
        iterator.close()


def test_unordered_work_refills_before_slow_first_result() -> None:
    """A stalled first fit must not idle workers or enqueue the whole image."""
    class Pool:
        def __init__(self) -> None:
            self.first: Future[int] = Future()
            self.submitted: list[int] = []

        def submit(self, function: Any, payload: bytes) -> Future[int]:
            _, value = pickle.loads(payload)
            self.submitted.append(value)
            if value == 0:
                return self.first
            future: Future[int] = Future()
            future.set_result(value)
            if value == 2:
                self.first.set_result(0)
            return future

        def shutdown(self, **kwargs: Any) -> None:
            pass

    with execution.CPIExecutor(_resources(2)) as executor:
        pool = Pool()
        executor.pool = pool  # type: ignore[assignment]
        iterator = executor.map(_worker_info, range(50), ordered=False)
        first = next(iterator)
        assert first == 1
        assert pool.submitted == [0, 1, 2]
        assert sorted([first, *iterator]) == list(range(50))


def test_memory_budget_reduces_workers_or_reports_insufficient_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler allocation accounts for parent memory and child overhead."""
    monkeypatch.setattr(execution.psutil, "Process",
                        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=200_000_000)))
    assert execution.plan_resources(8, 20, 30, 5, memory_gb=2.3).processes == 2
    assert execution.plan_resources(8, 20, 30, 5, memory_gb=1.5).processes == 1
    assert execution.plan_resources(8, 20, 30, 5, memory_gb=1).processes == 1
    assert execution.plan_resources(8, 1, 30, 5).processes == 1
    with pytest.raises(MemoryError, match="--mem-gb"):
        execution.plan_resources(8, 20, 30, 5, memory_gb=.25)
    with pytest.raises(MemoryError):
        execution.plan_resources(8, 20, 30, 5, memory_gb=1, extra_parent_bytes=300_000_000)


def test_workspace_includes_adaptive_headroom_and_grows_with_acquisition() -> None:
    """Normal curves reserve cache/workspace headroom; longer curves scale up."""
    ordinary = execution.plan_resources(4, 100, 120, 5)
    assert ordinary.estimated_worker_gb == 1.024
    assert execution.plan_resources(4, 100, 400, 12).estimated_worker_gb > ordinary.estimated_worker_gb


def test_workflow_allocation_does_not_halve_large_cpi_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconstrained large-memory node can use every requested CPI process."""
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.workflows.dsc.base import init_dsc_preproc_wf

    requested = 82
    base_memory = 1.2
    wf = init_dsc_preproc_wf(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        subject_id="test",
        session_id="test",
        config=DSCConfig(method="cpi", num_processes=requested, sloppy=True),
        mem_gb=base_memory,
        mem_limit_gb=200.0,
    )
    node = wf.get_node("dsc_deconvolution_wf.compute_maps")
    expected = (
        base_memory
        + execution.CPI_CONTROLLER_MEMORY_GB
        + requested * execution.estimate_worker_memory_gb()
    )
    assert node.mem_gb == pytest.approx(expected)
    assert node.inputs.fitting_memory_gb == pytest.approx(expected)

    monkeypatch.setattr(
        execution.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=3_300_000_000)),
    )
    resources = execution.plan_resources(
        requested,
        81_306,
        120,
        6,
        memory_gb=node.inputs.fitting_memory_gb,
        extra_parent_bytes=90_000_000,
    )
    assert resources.processes == requested


@pytest.mark.parametrize("workers,memory", [(0, None), (True, None), (1, 0), (1, np.nan)])
def test_invalid_execution_limits_raise(workers: Any, memory: Any) -> None:
    """Invalid limits are configuration errors, not failed voxels."""
    with pytest.raises(ValueError):
        execution.plan_resources(workers, 2, 30, 5, memory_gb=memory)


def prepared(
    leakage: str = "none", dispersion: str = "none", *, inference: str = "vi",
    sloppy: bool = True, infer_delay: bool = True,
) -> acquisition.CPIAcquisition:
    """Small distinct voxel priors, temporal weights, and original signal input."""
    t = np.arange(30.) * 3
    u = np.maximum(t - 9, 0)
    aif = 10 * u**2 * np.exp(-u/3)
    raw = np.stack([.2*u**2*np.exp(-u/5), .21*u**2*np.exp(-u/5)])[:, None, None]
    signal = 100 * np.exp(-.03 * raw)
    files = {"time": t, "raw": raw}
    guidance = guidance_for(files) if leakage == "bsw" else None
    return acquisition.prepare_cpi_acquisition(signal, aif, t, .03, guidance,
        leakage_method=leakage, inference_method=inference, baseline_frames=4,  # type: ignore[arg-type]
        temporal_weights=np.linspace(.8, 1., 30), sloppy=sloppy,
        options=CPIOptions(control_points=5, infer_delay=infer_delay,
                           dispersion=dispersion, fallback="none"))  # type: ignore[arg-type]


def compare_rows(expected: list[GuidedVoxel], actual: list[GuidedVoxel]) -> None:
    """All starts, posteriors and selected outputs agree, not just the best CBF."""
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual, strict=True):
        assert (left.flat_index, left.status, left.family, left.reasons) == (
            right.flat_index, right.status, right.family, right.reasons)
        np.testing.assert_allclose([left.cbf, left.cbv, left.mtt, left.delay, left.cbf_std, left.fitted_k2],
                                   [right.cbf, right.cbv, right.mtt, right.delay, right.cbf_std, right.fitted_k2],
                                   rtol=1e-8, atol=1e-10)
        assert left.assessment is not None and right.assessment is not None
        assert left.assessment.converged == right.assessment.converged
        assert left.diagnostics is not None and right.diagnostics is not None
        for key in ("mean", "covariance", "noise_shape", "noise_scale", "fitted_curves",
                    "prior_mean", "prior_covariance"):
            np.testing.assert_allclose(left.diagnostics.arrays[key], right.diagnostics.arrays[key],
                                       rtol=1e-8, atol=1e-10, err_msg=key)
        assert left.diagnostics.iterations == right.diagnostics.iterations
        np.testing.assert_allclose([x[1] for x in left.selection_scores],
                                   [x[1] for x in right.selection_scores], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("leakage,dispersion,delay", [
    ("none", "none", False), ("bsw", "none", True),
    ("none", "gamma", True), ("bsw", "auto", True),
])
def test_vi_parallel_matches_same_voxel_priors_and_all_starts(
    leakage: str, dispersion: str, delay: bool,
) -> None:
    """Changing worker count/order cannot change count, guidance or selection."""
    a = prepared(leakage, dispersion, infer_delay=delay)
    expected = list(acquisition.fit_selected_voxels(a, [1, 0]))
    for n in (1, 2, 4):
        completed: list[None] = []
        with execution.CPIExecutor(_resources(n)) as executor:
            actual = list(
                acquisition.fit_selected_voxels(
                    a,
                    [1, 0],
                    executor=executor,
                    on_result=lambda: completed.append(None),
                )
            )
        assert len(completed) == 2
        compare_rows(expected, actual)
        assert all(r.diagnostics.iterations <= 8 for r in actual if r.diagnostics)
    with execution.CPIExecutor(_resources(2)) as executor:
        unordered = list(acquisition.fit_selected_voxels(a, [1, 0], executor=executor, ordered=False))
    compare_rows(sorted(expected, key=lambda r: r.flat_index),
                 sorted(unordered, key=lambda r: r.flat_index))
    one = acquisition._voxel_acquisition(a, 1)
    for branch, original in zip(one.branches, a.branches, strict=True):
        assert branch.signals.shape == (1, 30)
        assert branch.decision == original.decision
        assert branch.reference is original.reference


@pytest.mark.parametrize("leakage,dispersion", [
    ("none", "none"), ("bsw", "none"), ("none", "gamma"),
])
def test_svb_parallel_preserves_complete_graph_and_sloppy_schedule(
    leakage: str, dispersion: str,
) -> None:
    """Parallel local updates still share the same frozen neighbor contributions."""
    a = prepared(leakage, dispersion, inference="svb", infer_delay=False)
    expected = field.fit_selected_field(a, keep_diagnostics=True)
    assert 0 < expected.spatial_sweeps <= 6
    with execution.CPIExecutor(_resources(2)) as executor:
        actual = field.fit_selected_field(a, keep_diagnostics=True, executor=executor)
    assert actual.spatial_sweeps == expected.spatial_sweeps
    compare_rows(list(expected.voxels), list(actual.voxels))
    for name in ("warm_means", "posterior_means", "conditional_prior_means", "conditional_prior_variances"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=1e-8, atol=1e-10)
        if dispersion == "gamma":
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    if dispersion == "gamma":
        for before, after in zip(expected.voxels, actual.voxels, strict=True):
            assert before.diagnostics is not None and after.diagnostics is not None
            for name in ("mean", "covariance", "noise_shape", "noise_scale", "fitted_curves",
                         "prior_mean", "prior_covariance"):
                np.testing.assert_array_equal(
                    after.diagnostics.arrays[name], before.diagnostics.arrays[name], err_msg=name,
                )


def test_map_workers_keep_predictions_and_provenance(tmp_path: Path) -> None:
    """Map assembly consumes fits at their original locations without a QC refit."""
    a = prepared()
    raw = -np.log(a.branches[0].signals.reshape(2, 1, 1, 30) / 100) / .03
    before = maps.fit_cpi_maps(a, raw, raw, oi_target=.035, prediction_file=tmp_path / "one.npy")
    after = maps.fit_cpi_maps(a, raw, raw, oi_target=.035, prediction_file=tmp_path / "two.npy", n_procs=2)
    for name in ("cbf", "cbv", "mtt", "delay", "cbf_std", "r_squared", "predictions", "fitted_k2",
                 "fallback_mask", "dispersion", "converged"):
        np.testing.assert_allclose(getattr(after, name), getattr(before, name), rtol=1e-8, atol=1e-10)
    assert after.qc["execution"]["processes"] == 2
    assert after.qc["execution"]["sloppy"] is True
    assert {k: v for k, v in before.qc.items() if k != "execution"} == {
        k: v for k, v in after.qc.items() if k != "execution"}


def test_invalid_voxel_keeps_global_identity() -> None:
    """Subset slicing must not turn rejection into a fit or an outside-mask voxel."""
    a = prepared()
    first = replace(a.branches[0], valid_signal=np.array([False, True]))
    a = replace(a, branches=(first,))
    with execution.CPIExecutor(_resources(2)) as executor:
        rows = list(acquisition.fit_selected_voxels(a, [1, 0], executor=executor))
    assert rows[0].flat_index == 1
    assert rows[1].flat_index == 0 and rows[1].status == "fallback_requested"
    assert rows[1].reasons == ("invalid_measured_signal",)


@dataclass
class _LinearModel:
    """Pickle-safe, exactly differentiable mixed-dimensional test adapter."""

    design: np.ndarray

    def predict(self, means: np.ndarray) -> np.ndarray:
        return np.asarray(means @ self.design.T)

    def jacobian(self, means: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.design, (len(means), *self.design.shape)).copy()


@pytest.mark.parametrize("workers,connected", [(2, False), (2, True), (4, True)])
def test_mixed_spatial_dimensions_keep_cross_family_edges(workers: int, connected: bool) -> None:
    """Different local models must still exchange CBF information at their border."""
    args = linear_arguments()
    args["weights"] = np.linspace(.7, 1., 12)
    local = []
    for i, p in enumerate((2, 3, 2)):
        design = np.column_stack((np.linspace(.2, 2., 12), np.ones(12),
                                  np.linspace(-1, 1, 12)**2))[:, :p]
        model = _LinearModel(design)
        mean, std = np.ones(p), np.linspace(1., 2., p)
        local.append(args | dict(tissue_curves=args["tissue_curves"][i:i+1], nlls_seeds=mean[None],
            prior_means_nat=mean, prior_stds_nat=std, prior_means_transformed=mean,
            prior_stds_transformed=std, transforms=args["transforms"][:1] * p,
            upper_bounds=[None]*p, fitting_param_names=["cbf"] + [f"local_{j}" for j in range(1, p)],
            fitting_bounds=[(0., 10.)] + [(-10., 10.)]*(p-1),
            model_adapter=DSCVIModelAdapter(model.predict, model.jacobian)))
    coords, shape = coordinates(connected)
    before, after = [], []
    serial = field.CommonCBFDispatcher(local, lambda i, r, p: before.append((i, r, p)))
    expected = batched_dsc_spatial_vi_fit(**common_arguments(args), mask_coords=coords,
                                         spatial_shape=shape, local_fit=serial)
    with execution.CPIExecutor(_resources(workers)) as executor:
        parallel = field.CommonCBFDispatcher(local, lambda i, r, p: after.append((i, r, p)),
                                             executor=executor)
        actual = batched_dsc_spatial_vi_fit(**common_arguments(args), mask_coords=coords,
                                           spatial_shape=shape, local_fit=parallel)
    assert parallel.calls == serial.calls == (11 if connected else 1)
    np.testing.assert_allclose(actual.posterior_means_transformed, expected.posterior_means_transformed,
                               rtol=1e-8, atol=1e-10)
    assert len(before) == len(after)
    for (i, old, old_prior), (j, new, new_prior) in zip(before, after, strict=True):
        assert i == j
        for name in ("mean", "covariance", "noise_shape", "noise_scale"):
            np.testing.assert_allclose(getattr(new.full_posterior, name),
                                       getattr(old.full_posterior, name), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(new_prior.mean, old_prior.mean, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(new_prior.covariance, old_prior.covariance, rtol=1e-8, atol=1e-10)


def test_workflow_reserves_the_same_memory_and_processes_given_to_cpi() -> None:
    """Workers use the node's share of the workflow budget, not a second budget."""
    from pmriprep.workflows.dsc.base import init_dsc_deconvolution_wf

    wf = init_dsc_deconvolution_wf(dsc_map_kwargs=dict(method="cpi", n_procs=4, sloppy=True),
                                    mem_gb=1.5)
    node = wf.get_node("compute_maps")
    assert node.n_procs == node.inputs.n_procs == 4
    assert node.mem_gb == node.inputs.fitting_memory_gb == 1.5
    assert node.inputs.sloppy


@pytest.mark.parametrize("memory_limit", [1.5, 4.0])
def test_cpi_worker_allocation_respects_workflow_memory_limit(tmp_path: Path, memory_limit: float) -> None:
    """The complete DSC workflow must not reserve a second uncapped worker budget."""
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.workflows.dsc.base import init_dsc_preproc_wf

    wf = init_dsc_preproc_wf(bids_dir=tmp_path, output_dir=tmp_path / "out",
        subject_id="test", session_id="test",
        config=DSCConfig(method="cpi", num_processes=4, sloppy=True),
        mem_gb=1., mem_limit_gb=memory_limit)
    node = wf.get_node("dsc_deconvolution_wf.compute_maps")
    assert node.n_procs == node.inputs.n_procs == 4
    uncapped = 1. + execution.CPI_CONTROLLER_MEMORY_GB + 4 * execution.estimate_worker_memory_gb()
    assert node.mem_gb == node.inputs.fitting_memory_gb == min(uncapped, memory_limit)
    assert node.inputs.sloppy
