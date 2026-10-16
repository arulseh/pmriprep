"""Bounded process execution for independent CPI fits and local SVB updates."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import AbstractContextManager
from dataclasses import dataclass
import multiprocessing as mp
import os
import pickle
from types import TracebackType
from typing import Any, TypeVar

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from pmriprep.utils.parallel import limit_worker_threads
from pmriprep.utils.processes import terminate_process_pool

T = TypeVar("T")
R = TypeVar("R")

# The workflow reserves this in addition to image-buffer estimates and the
# per-worker fitting allocation. It covers the controller interpreter, loaded
# numerical libraries, and image-indexed bookkeeping that remain resident while
# the spawned fitting pool is active.
CPI_CONTROLLER_MEMORY_GB = 4.0


@dataclass(frozen=True)
class CPIResources:
    """Execution limits; memory figures are estimates, not OS-enforced quotas."""

    requested_processes: int
    processes: int
    threads_per_process: int
    memory_gb: float | None
    estimated_worker_gb: float


def estimate_worker_memory_gb(n_frames: int = 120, n_points: int = 12) -> float:
    """Estimate resident memory for one CPI fitting worker.

    Parameters
    ----------
    n_frames : int, optional
        Number of acquisition frames.
    n_points : int, optional
        Maximum number of residue control points.

    Returns
    -------
    float
        Estimated worker memory in decimal gigabytes.

    Raises
    ------
    ValueError
        If the frame or control-point count is non-positive.
    """
    if min(n_frames, n_points) < 1:
        raise ValueError("CPI execution dimensions must be positive")
    workspace = max(768_000_000, 8 * n_frames**2 * n_points * 64)
    return (256_000_000 + workspace) / 1e9


def plan_resources(
    n_procs: int, n_voxels: int, n_frames: int, n_points: int,
    *, memory_gb: float | None = None, extra_parent_bytes: int = 0,
) -> CPIResources:
    """Limit worker count using the node allocation and resident parent memory.

    Parameters
    ----------
    n_procs, n_voxels, n_frames, n_points : int
        Requested processes and acquisition sizes. Numerical libraries use one
        thread per fitting process, within both the CPU and OMP thread limits.
    memory_gb : float, optional
        Same allocation reserved for this node by the workflow scheduler.
    extra_parent_bytes : int, optional
        Estimated additional image maps and spatial state, not yet allocated.

    Returns
    -------
    CPIResources
        Worker cap accounting for independent interpreter/workspace overhead.
        A process pool never holds copies of the complete signal image.

    Raises
    ------
    ValueError
        If process count, memory allowance or execution dimensions are invalid.
    MemoryError
        If the allocation cannot hold the estimated serial working set.

    Notes
    -----
    The workspace allowance scales with acquisition length and knot count for
    polynomial overlap/quadrature arrays. Adaptive integration and allocator
    behavior prevent this estimate from being a hard resident-memory guarantee.
    """
    if isinstance(n_procs, bool) or not isinstance(n_procs, int) or n_procs < 1:
        raise ValueError("n_procs must be a positive integer")
    if memory_gb is not None and (not np.isfinite(memory_gb) or memory_gb <= 0):
        raise ValueError("fitting memory allocation must be finite and positive")
    if min(n_voxels, extra_parent_bytes) < 0 or min(n_frames, n_points) < 1:
        raise ValueError("invalid CPI execution dimensions")
    # Include cached kernels, refinement state and allocator headroom as well
    # as the bounded interval-node temporaries and acquisition-sized geometry.
    worker_bytes = round(estimate_worker_memory_gb(n_frames, n_points) * 1e9)
    workspace = worker_bytes - 256_000_000
    workers = min(n_procs, max(1, n_voxels))
    if memory_gb is not None and n_voxels:
        remaining = memory_gb * 1e9 - psutil.Process().memory_info().rss - extra_parent_bytes
        if remaining < workspace:
            raise MemoryError(
                "CPI fitting memory allocation is below the estimated serial working set; "
                "increase --mem-gb or reduce concurrent work"
            )
        workers = min(workers, max(1, int(remaining // worker_bytes)))
    return CPIResources(n_procs, workers, 1, memory_gb, worker_bytes / 1e9)


def _initialize_worker() -> None:
    """Apply thread caps after spawn, including already-loaded BLAS libraries."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    limit_worker_threads()


def _run_serialized(payload: bytes) -> Any:
    """Execute an internally created task snapshot, never an external pickle."""
    function, value = pickle.loads(payload)
    return function(value)


class CPIExecutor(AbstractContextManager["CPIExecutor"]):
    """Reuse one process pool with bounded submission and owned cleanup.

    Parameters
    ----------
    resources : CPIResources
        Resolved process/thread/memory limits. Serial execution uses the same
        tasks and numerical thread limit as a worker and restores the parent
        thread configuration on exit.
    """

    def __init__(self, resources: CPIResources) -> None:
        self.resources = resources
        self.pool: ProcessPoolExecutor | None = None
        self._threads: Any = None

    def __enter__(self) -> CPIExecutor:
        self._threads = threadpool_limits(limits=self.resources.threads_per_process)
        return self

    def map(self, function: Callable[[T], R], values: Iterable[T],
            *, ordered: bool = True) -> Iterator[R]:
        """Yield results with at most one queued job per worker.

        Only small per-voxel inputs should be submitted. Errors and interrupted
        or abandoned iterators terminate owned workers rather than leaving fits
        running after a failed interface. Ordered callers retain input order.
        Image-indexed consumers can set ordered=False to release a completed
        voxel immediately and keep its worker busy behind a slow earlier fit.
        """
        if self.resources.processes == 1:
            for value in values:
                yield function(value)
            return
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=self.resources.processes, mp_context=mp.get_context("spawn"),
                initializer=_initialize_worker,
            )
        pending: deque[Future[R]] = deque()
        iterator = iter(values)
        try:
            for _ in range(self.resources.processes):
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                pending.append(self.pool.submit(_run_serialized, pickle.dumps((function, value))))
            while pending:
                if ordered:
                    result = pending.popleft().result()
                    yield result
                else:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    future = next(f for f in pending if f in completed)
                    pending.remove(future)
                    result = future.result()
                try:
                    value = next(iterator)
                except StopIteration:
                    pass
                else:
                    pending.append(self.pool.submit(_run_serialized, pickle.dumps((function, value))))
                if not ordered:
                    yield result
        except BaseException:
            self._terminate()
            raise

    def _terminate(self) -> None:
        """Stop only this executor's workers and release its ownership."""
        if self.pool is not None:
            terminate_process_pool(self.pool)
            self.pool = None

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc_type is not None:
                self._terminate()
            elif self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
                self.pool = None
        finally:
            if self._threads is not None:
                self._threads.restore_original_limits()
