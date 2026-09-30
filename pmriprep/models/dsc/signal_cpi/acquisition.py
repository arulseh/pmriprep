"""Single-CPI acquisition policy with plain/gamma model selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterator, Literal, Sequence

import numpy as np

from pmriprep.utils.bsw_guidance import BSWGuidance
from .connection import GuidedAcquisition, GuidedVoxel, fit_voxels, prepare_acquisition, requested_count
from .leakage import leakage_count
from .policy import automatic_count
from .settings import CPIOptions
from .execution import CPIExecutor


@dataclass(frozen=True)
class CPIAcquisition:
    """One immutable acquisition and its requested internal transport branches."""

    options: CPIOptions
    branches: tuple[GuidedAcquisition, ...]


def prepare_cpi_acquisition(
    signals: np.ndarray, aif: np.ndarray, time: np.ndarray, te: float,
    guidance: BSWGuidance | None = None,
    *, options: CPIOptions = CPIOptions(), inference_method: str = "vi",
    temporal_weights: np.ndarray | None = None,
    leakage_method: Literal["bsw", "none"] = "bsw",
    mask: np.ndarray | None = None, baseline_frames: int | None = None,
    sloppy: bool = False,
) -> CPIAcquisition:
    """Prepare family-specific counts while sharing the owned acquisition arrays.

    Parameters
    ----------
    signals, aif, time, te, guidance
        Original measured input and exact optional BSW guidance.
    options : CPIOptions
        Count, timing, transport selection and fallback policy.
    inference_method : {"vi", "svb"}
        Shared ordinary or spatial variational inference.
    temporal_weights, leakage_method, mask, baseline_frames
        Likelihood weights, leakage policy, spatial support and baseline window.
    sloppy : bool, optional
        Use the shared fast testing schedule in every branch and spatial sweep.

    Returns
    -------
    CPIAcquisition
        One or two branches; automatic selection changes no priors or starts.
    """
    first = prepare_acquisition(signals, aif, time, te, guidance,
        inference_method=inference_method, temporal_weights=temporal_weights,
        family="cpi_vtf" if options.dispersion == "gamma" else "cpi",
        leakage_method=leakage_method, mask=mask, baseline_frames=baseline_frames,
        control_points=options.control_points, infer_delay=options.infer_delay, sloppy=sloppy)
    if options.dispersion != "auto":
        return CPIAcquisition(options, (first,))
    pool = first.fit_support & first.valid_signal
    decision = None
    if np.any(pool):
        count = leakage_count if leakage_method == "bsw" else automatic_count
        assert first.decision is not None
        decision = count(first.signals[pool], first.aif, first.time, first.te,
                         family="cpi_vtf", baseline_frames=first.decision.baseline_frames)
        decision = requested_count(decision, options.control_points)
    return CPIAcquisition(options, (first, replace(first, family="cpi_vtf", decision=decision)))


def choose_transport(rows: Sequence[GuidedVoxel]) -> GuidedVoxel:
    """Select the higher ordinary-VI score, preferring plain on exact ties.

    Parameters
    ----------
    rows : sequence of GuidedVoxel
        One or both transport branches for the same spatial voxel, before SVB.

    Returns
    -------
    GuidedVoxel
        Selected valid/provisional fit, or rejection with all branch reasons.
        A failed branch cannot override a valid alternative.
    """
    if (not rows or len({r.flat_index for r in rows}) != 1
            or any(r.family not in ("cpi", "cpi_vtf") for r in rows)
            or len({r.family for r in rows}) != len(rows)):
        raise ValueError("distinct CPI branches for one spatial voxel required")
    scored: list[tuple[float, GuidedVoxel]] = []
    scores = []
    for row in rows:
        score = None
        if row.status in ("retained", "provisional"):
            if row.diagnostics is None or row.diagnostics.rows[0]["selected"] is None:
                raise ValueError("transport selection requires the complete ordinary-VI score")
            score = float(row.diagnostics.rows[0]["selected"].score)
            if not np.isfinite(score):
                raise ValueError("a retained CPI state cannot have a nonfinite score")
            scored.append((score, row))
        scores.append((str(row.family), score))
    if scored:
        winner = max(scored, key=lambda pair: (pair[0], pair[1].family == "cpi"))[1]
    else:
        winner = next((r for r in rows if r.family == "cpi"), rows[0])
        reasons = tuple(dict.fromkeys(reason for row in rows for reason in row.reasons))
        winner = replace(winner, family=None, reasons=reasons)
    return replace(winner, selection_scores=tuple(scores))


def fit_selected_voxels(
    acquisition: CPIAcquisition, indices: Sequence[int] | None = None,
    *, executor: CPIExecutor | None = None, ordered: bool = True,
    on_result: Callable[[], None] | None = None,
) -> Iterator[GuidedVoxel]:
    """Fit ordinary-VI branches in lockstep and release each losing posterior.

    Parameters
    ----------
    acquisition : CPIAcquisition
        Full prepared acquisition; spatial callers use this for initial selection.
    indices : sequence of int, optional
        Requested independent voxel order. Counts remain acquisition-wide.
    executor : CPIExecutor, optional
        Bounded executor for independent voxel selection. Each task receives
        one signal and its priors, not a copy of the full image.
    ordered : bool, optional
        Preserve requested voxel order. Image-indexed consumers may disable
        ordering to refill workers as soon as any voxel finishes.
    on_result : callable, optional
        Controller-side callback after each completed selected voxel.

    Yields
    ------
    GuidedVoxel
        Selected result with both branch scores and its unchanged spatial index.
    """
    if executor is not None:
        selected = (np.flatnonzero(acquisition.branches[0].fit_support).tolist()
                    if indices is None else list(indices))
        n = len(acquisition.branches[0].signals)
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                or not 0 <= i < n for i in selected) or len(set(selected)) != len(selected)):
            raise ValueError("unique in-range integer spatial indices required")
        for row in executor.map(
            _fit_selected_row,
            ((_voxel_acquisition(acquisition, i), int(i)) for i in selected),
            ordered=ordered,
        ):
            if on_result is not None:
                on_result()
            yield row
        return
    iterators = [fit_voxels(replace(a, inference_method="vi"), indices)
                 for a in acquisition.branches]
    for rows in zip(*iterators, strict=True):
        # Rejected voxels also retain the identity of the evaluated branch.
        owned = [replace(row, family=branch.family)
                 for row, branch in zip(rows, acquisition.branches, strict=True)]
        chosen = choose_transport(owned)
        if on_result is not None:
            on_result()
        yield chosen


def _voxel_acquisition(acquisition: CPIAcquisition, index: int) -> CPIAcquisition:
    """Slice only spatial arrays; keep acquisition-wide count and reference."""
    branches = tuple(replace(a, spatial_shape=(1, 1, 1), signals=a.signals[index:index+1],
        k2=a.k2[index:index+1], k2_sd=a.k2_sd[index:index+1],
        guidance_status=None if a.guidance_status is None else a.guidance_status[index:index+1],
        valid_signal=a.valid_signal[index:index+1], support=a.fit_support[index:index+1],
        inference_method="vi") for a in acquisition.branches)
    return replace(acquisition, branches=branches)


def _fit_selected_row(task: tuple[CPIAcquisition, int]) -> GuidedVoxel:
    """Fit one selected branch pair and restore its original image identity."""
    acquisition, index = task
    row = next(fit_selected_voxels(acquisition, [0]))
    assessment = None if row.assessment is None else replace(row.assessment, voxel=index)
    return replace(row, flat_index=index, assessment=assessment)
