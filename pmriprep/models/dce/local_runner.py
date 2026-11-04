"""Bounded conditional fitting and local selection for complete image slices."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import get_context
from types import TracebackType

import numpy as np
from threadpoolctl import threadpool_limits

from pmriprep.models.dce.delay_profiles import (
    MAX_LOCAL_DELAY_SECONDS,
    DelayProfile,
    ProfileFamily,
    build_delay_profile,
)
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.local_refinement import ConditionalInput, ConditionalResult, finish_conditional
from pmriprep.models.dce.local_selection import (
    LocalSelection,
    SelectionMode,
    in_plane_neighborhoods,
    profile_families,
    select_pool,
)
from pmriprep.utils.parallel import limit_worker_threads
from pmriprep.utils.processes import terminate_process_pool


@dataclass
class SliceFits:
    """Completed profiles and choices in the slice's original member order.

    Attributes
    ----------
    coordinates : ndarray
        Copy of the supplied eligible integer voxel coordinates.
    radius_mm : float or None
        Effective physical radius; None when delay correction is disabled.
    neighborhoods : list of ndarray
        Full-slice member indices, or singletons when delay is fixed to zero.
    profiles : dict
        Finalized per-family profiles, including original observation identities
        and retry evidence. Retained for downstream uncertainty and diagnostics.
    selections : list of LocalSelection
        One result per input member, before inference and final retention/QC.
    """

    coordinates: np.ndarray
    radius_mm: float | None
    neighborhoods: list[np.ndarray]
    profiles: dict[ProfileFamily, DelayProfile]
    selections: list[LocalSelection]


@dataclass
class _ProfileBlock:
    """Picklable input for one bounded block, sharing original frame identities."""

    aif: np.ndarray
    times: np.ndarray
    curves: np.ndarray
    observed: np.ndarray
    grid: np.ndarray
    families: list[ProfileFamily]


def _fit_profile_block(block: _ProfileBlock) -> dict[ProfileFamily, DelayProfile]:
    """Build all requested profiles without creating another worker pool."""
    with threadpool_limits(limits=1):
        return {
            family: build_delay_profile(
                block.aif, block.times, block.curves, family,
                observed=block.observed, delays_seconds=block.grid,
            )
            for family in block.families
        }


def _scatter_block(
    destination: dict[ProfileFamily, DelayProfile],
    block: dict[ProfileFamily, DelayProfile],
    start: int,
    total: int,
) -> None:
    """Copy one block to slice arrays, translating retry member identities."""
    for family, source in block.items():
        count = source.sse.shape[1]
        if family not in destination:
            shape = (len(source.delays_seconds), total)
            destination[family] = replace(
                source,
                params=np.empty((*shape, source.params.shape[2]), dtype=source.params.dtype),
                sse=np.empty(shape, dtype=source.sse.dtype),
                status=np.empty(shape, dtype=source.status.dtype),
                evaluations=np.empty(shape, dtype=source.evaluations.dtype),
                optimality=np.empty(shape, dtype=source.optimality.dtype),
                seed_index=np.empty(shape, dtype=source.seed_index.dtype),
                retries={},
                failures={},
            )
        target = destination[family]
        for name in ("params", "sse", "status", "evaluations", "optimality", "seed_index"):
            getattr(target, name)[:, start:start + count] = getattr(source, name)
        target.retries.update(
            {(delay, start + member): attempts for (delay, member), attempts in source.retries.items()}
        )
        target.failures.update({start + member: failure for member, failure in source.failures.items()})


class LocalDelayRunner:
    """Reuse one bounded worker pool across complete slices.

    Parameters
    ----------
    n_procs : int
        Positive process budget already resolved by the calling pipeline node.
        One uses in-process fitting. Each process uses one BLAS/OpenMP thread.
    chunk_size : int
        Maximum members per fitting task; it does not limit spatial support.

    Notes
    -----
    Use as a context manager. Process each complete eligible slice, consume its
    profiles for inference/QC/output scattering, then release the result before
    processing the next slice. The runner retains no completed results. At most
    n_procs tasks are outstanding, limiting extra profile storage to bounded
    blocks rather than a second complete slice or volume. Classified local
    numerical failures remain recorded in their family profiles. Unexpected
    worker/backend exceptions abort and reap the pool.
    """

    def __init__(self, n_procs: int = 1, chunk_size: int = 128) -> None:
        for name, value in (("n_procs", n_procs), ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.n_procs = n_procs
        self.chunk_size = chunk_size
        self._executor: ProcessPoolExecutor | None = None
        self._active = False
        self._used = False

    def __enter__(self) -> LocalDelayRunner:
        """Start a single-use context; the pool is shared by all slice calls."""
        if self._used:
            raise RuntimeError("Local delay runner contexts cannot be reused.")
        self._used = True
        if self.n_procs > 1:
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_procs,
                mp_context=get_context("spawn"),
                initializer=limit_worker_threads,
            )
        self._active = True
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_value: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        """Close normally or terminate owned workers after an abnormal exit."""
        self._close(abort=exc_type is not None)

    def _close(self, *, abort: bool) -> None:
        """Release the pool and mark the runner unusable, including after failure."""
        executor, self._executor = self._executor, None
        self._active = False
        if executor is not None:
            if abort:
                terminate_process_pool(executor)
            else:
                executor.shutdown(wait=True, cancel_futures=True)

    def fit_slice(
        self,
        aif: np.ndarray,
        times: np.ndarray,
        curves: np.ndarray,
        coordinates: np.ndarray,
        affine: np.ndarray,
        *,
        mode: SelectionMode = "auto",
        observed: np.ndarray | None = None,
        delay_correction: bool = True,
        sloppy: bool = False,
    ) -> SliceFits:
        """Fit once per member/family, then pool across the complete slice.

        Parameters
        ----------
        aif, times : ndarray
            Full plasma input in mM and original acquisition times in minutes.
        curves : ndarray
            Finite member-by-retained-observation tissue concentrations in mM.
        coordinates : ndarray
            All input-fitting-eligible voxel indices in one slice, in curve order.
            Caller must not restrict membership with a statistics ROI or fit QC.
        affine : ndarray
            Original voxel-to-mm affine.
        mode : str
            Automatic selection or explicitly requested kinetic family.
        observed : ndarray, optional
            Common retained original frame indices, in curve-column order.
        delay_correction, sloppy : bool
            Full 0:0.1:12-s search by default; sloppy uses 0:1:12 seconds.
            Disabled correction uses only zero, irrespective of sloppy.

        Returns
        -------
        SliceFits
            Complete profiles and selections in input order. No map retention,
            inference refinement or identification masking is applied here.
        """
        if not self._active:
            raise RuntimeError("Use an active LocalDelayRunner context.")
        aif, times = np.asarray(aif, float), np.asarray(times, float)
        context = DelayedPchip(aif, times, 0.0, observed)
        indices = context.observed
        families = profile_families(mode, float((times[1] - times[0]) * 60))
        y, xyz = np.asarray(curves, float), np.asarray(coordinates)
        if (
            y.ndim != 2 or y.shape[1] != len(indices) or not np.isfinite(y).all()
            or len(indices) < 2 or len(np.unique(indices)) != len(indices)
            or xyz.shape != (len(y), 3)
        ):
            raise ValueError("Slice curves/coordinates and unique common observations must match.")
        # Geometry validation is also required when no spatial pooling is used.
        radius, neighborhoods = in_plane_neighborhoods(xyz, affine)
        spatial_radius: float | None = radius if delay_correction else None
        if not delay_correction:
            neighborhoods = [np.array([i], dtype=np.intp) for i in range(len(y))]
        scale = 1 if sloppy else 10
        grid = (
            np.arange(int(round(MAX_LOCAL_DELAY_SECONDS * scale)) + 1) / scale
            if delay_correction else np.array([0.0])
        )
        profiles: dict[ProfileFamily, DelayProfile] = {}
        try:
            self._build_slice(aif, times, y, indices, grid, families, profiles)
            with threadpool_limits(limits=1):
                selections = [
                    select_pool(profiles, aif, times, members, i, mode)
                    for i, members in enumerate(neighborhoods)
                ]
        except BaseException:
            self._close(abort=True)
            raise
        return SliceFits(xyz.copy(), spatial_radius, neighborhoods, profiles, selections)

    def _build_slice(
        self, aif: np.ndarray, times: np.ndarray, curves: np.ndarray,
        observed: np.ndarray, grid: np.ndarray, families: list[ProfileFamily],
        profiles: dict[ProfileFamily, DelayProfile],
    ) -> None:
        """Assemble completed blocks without queueing the whole slice at once."""
        total = len(curves)
        starts = iter(range(0, max(total, 1), self.chunk_size))

        def block(start: int) -> _ProfileBlock:
            """Copy only the next bounded input block for worker serialization."""
            return _ProfileBlock(
                aif, times, curves[start:start + self.chunk_size].copy(),
                observed, grid, families,
            )

        if self._executor is None:
            for start in starts:
                _scatter_block(profiles, _fit_profile_block(block(start)), start, total)
            return
        pending: deque[tuple[int, Future[dict[ProfileFamily, DelayProfile]]]] = deque()
        for _ in range(self.n_procs):
            next_start = next(starts, None)
            if next_start is not None:
                pending.append((next_start, self._executor.submit(_fit_profile_block, block(next_start))))

        while pending:
            start, future = pending.popleft()
            try:
                _scatter_block(profiles, future.result(), start, total)
            except Exception as error:
                raise RuntimeError(f"Local delay fitting failed for members starting at {start}.") from error
            # Future retains its result until released; drop it before queuing more.
            del future
            next_start = next(starts, None)
            if next_start is not None:
                pending.append((next_start, self._executor.submit(_fit_profile_block, block(next_start))))

    def refine_selected(self, inputs: list[ConditionalInput]) -> list[ConditionalResult]:
        """Reuse the bounded pool for NLLS identification and fixed-delay inference.

        Parameters
        ----------
        inputs : list of ConditionalInput
            Eligible selected fits from the current slice only, in voxel order.

        Returns
        -------
        list of ConditionalResult
            Conditional results in the same order. No new model/delay search or
            process pool is created, and unexpected failures abort owned workers.
        """
        if not self._active:
            raise RuntimeError("Use an active LocalDelayRunner context.")
        starts = iter(range(0, len(inputs), self.chunk_size))
        results: list[ConditionalResult] = []
        pending: deque[Future[list[ConditionalResult]]] = deque()
        try:
            if self._executor is None:
                for serial_start in starts:
                    results.extend(finish_conditional(inputs[serial_start:serial_start + self.chunk_size]))
                return results
            for _ in range(self.n_procs):
                start = next(starts, None)
                if start is not None:
                    pending.append(self._executor.submit(finish_conditional, inputs[start:start + self.chunk_size]))
            while pending:
                future = pending.popleft()
                results.extend(future.result())
                del future
                start = next(starts, None)
                if start is not None:
                    pending.append(self._executor.submit(finish_conditional, inputs[start:start + self.chunk_size]))
        except BaseException:
            self._close(abort=True)
            raise
        return results
