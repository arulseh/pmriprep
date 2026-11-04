"""Conditional kinetic profiles with post-sweep native fit finalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.cached_input import CachedPchip
from pmriprep.models.dce.fit_errors import NumericalFitError
from pmriprep.models.dce.native_exchange import ExchangeFit, NativeInput, fit_native
from pmriprep.models.dce.profile_fitting import (
    fit_extended_tofts,
    fit_patlak,
    fit_tofts,
)

ProfileFamily = Literal["patlak", "tofts", "extended_tofts", "2cxm"]
MAX_LOCAL_DELAY_SECONDS: float = 12.0


@dataclass
class ExchangeRetry:
    """Evidence from one full-start retry at a fixed delay.

    Attributes
    ----------
    budget : int
        Residual evaluation budget per start.
    source, candidate : ExchangeFit
        Best fit before the attempt and the returned retry fit, respectively.
    retained : bool
        Whether the candidate replaced the source. A worse SSE never wins
        merely because the candidate converged.
    """

    budget: int
    source: ExchangeFit
    candidate: ExchangeFit
    retained: bool


@dataclass
class ProfileFailure:
    """Reason one member's family profile is unavailable for all delays.

    Attributes
    ----------
    delay_index : int
        Point at which the numerical failure occurred.
    phase : {'sweep', 'retry'}
        Whether the original sweep or post-sweep retry failed.
    reason : str
        Explicit numerical failure, distinct from finite nonconvergence.
    """

    delay_index: int
    phase: Literal["sweep", "retry"]
    reason: str


@dataclass
class DelayProfile:
    """One model's completed conditional fits on a common observation set.

    Attributes
    ----------
    family : str
        Kinetic model name.
    delays_seconds, observed : ndarray
        Actual increasing delay grid and retained original frame indices.
    params : ndarray
        Delay-by-member-by-parameter array. Tofts uses Ktrans/ve, Patlak and
        ET use Ktrans/ve/vp (Patlak ve is NaN), and 2CXM uses Fp/PS/ve/vp.
        Ktrans is min^-1; Fp/PS are mL/min/100 mL; volumes are fractions.
    sse, status : ndarray
        Delay-by-member residual SSE and numerical status after retries.
        Native statuses retain their solver meaning. For profiled linear/rate
        fits, 1 means a finite numerical result. Neither implies fit QC,
        physiological eligibility or parameter identification.
    evaluations, optimality : ndarray
        Original-sweep native evaluation counts and final selected gradient
        norms. Counts exclude retries, whose evidence is in ``retries``.
        Counts are -1 and optimality NaN for non-native model families.
    seed_index : ndarray
        Original continuation predecessor, or -1 for no continuation. Repairs
        do not alter this array or propagate into later sweep points.
    retries : dict
        (Delay index, member index) to completed ordered retry attempts. Finite
        unconverged fits retain their own nonpositive status and actual SSE.
    failures : dict
        Member index to known numerical failure. Affected members have NaN
        parameters/SSE and status zero at every delay, excluding their family
        profile from pooling. Other members and families remain available.
    """

    family: ProfileFamily
    delays_seconds: np.ndarray
    observed: np.ndarray
    params: np.ndarray
    sse: np.ndarray
    status: np.ndarray
    evaluations: np.ndarray
    optimality: np.ndarray
    seed_index: np.ndarray
    retries: dict[tuple[int, int], tuple[ExchangeRetry, ...]]
    failures: dict[int, ProfileFailure] = field(default_factory=dict)


def _invalidate_member(
    profile: DelayProfile, member: int, delay: int,
    phase: Literal["sweep", "retry"], reason: str,
) -> None:
    """Record a local numerical failure without making up residual evidence."""
    profile.failures[member] = ProfileFailure(delay, phase, reason)
    profile.params[:, member] = np.nan
    profile.sse[:, member] = np.nan
    profile.status[:, member] = 0
    profile.optimality[:, member] = np.nan


def retry_exchange_fit(
    context: NativeInput, curve: np.ndarray, initial: ExchangeFit
) -> tuple[ExchangeFit, tuple[ExchangeRetry, ...]]:
    """Retry a finite failed fit without changing its delay or observation set.

    Parameters
    ----------
    context : NativeInput
        Fixed-delay plasma input and original observation support.
    curve : ndarray
        Tissue concentration in mM at those observations.
    initial : ExchangeFit
        Original conditional fit. Converged fits require no retry.

    Returns
    -------
    best : ExchangeFit
        Lowest finite SSE among the original and retries. Exact SSE ties
        prefer convergence; otherwise the earlier result is retained.
    attempts : tuple of ExchangeRetry
        Full-start attempts at budgets 180 then 720. The best-so-far physical
        parameters seed each retry; stop once that retained fit converges.

    Raises
    ------
    ValueError
        If the original result has no finite, nonnegative SSE or parameters.
    NumericalFitError
        If a retry cannot produce a finite numerical result. The caller must
        mark this member's family profile unavailable rather than infer flatness.
    """
    if (
        np.shape(initial.params) != (4,)
        or not np.isfinite(initial.params).all()
        or not np.isfinite(initial.sse)
        or initial.sse < 0
    ):
        raise ValueError("Retry requires a finite original exchange fit.")
    best = initial
    attempts = []
    for budget in (180, 720):
        if best.status > 0:
            break
        try:
            candidate = fit_native(
                context, curve, previous=best.params.copy(), max_nfev=budget
            )
        except NumericalFitError as error:
            raise NumericalFitError(f"Retry budget {budget}: {error}") from error
        if candidate.sse < 0:
            raise RuntimeError("Native retry returned a negative residual sum of squares.")
        if not np.isfinite(candidate.sse) or not np.isfinite(candidate.params).all():
            raise NumericalFitError(f"Retry budget {budget}: nonfinite native fit result")
        retained = bool(
            candidate.sse < best.sse
            or (
                candidate.sse == best.sse
                and candidate.status > 0
                and best.status <= 0
            )
        )
        attempts.append(ExchangeRetry(budget, best, candidate, retained))
        if retained:
            best = candidate
    return best, tuple(attempts)


def build_delay_profile(
    aif: np.ndarray,
    times: np.ndarray,
    curves: np.ndarray,
    family: ProfileFamily,
    *,
    observed: np.ndarray | None = None,
    delays_seconds: np.ndarray | None = None,
) -> DelayProfile:
    """Fit a member block at every delay, then repair failed native points.

    Parameters
    ----------
    aif, times : ndarray
        Full plasma input in mM and uniform acquisition times in minutes.
    curves : ndarray
        Finite member-by-retained-observation tissue concentrations in mM.
        No per-member frame filtering is performed.
    family : {'patlak', 'tofts', 'extended_tofts', '2cxm'}
        Requested conditional model, independent of automatic admission.
    observed : ndarray, optional
        Unique original frame indices in curve-column order. Omitted frames
        remain part of the input/convolution history.
    delays_seconds : ndarray, optional
        Increasing grid starting at zero, inside [0,12] seconds. Default is
        0:0.1:12. A single zero fixes delay; 0:1:12 supports test-mode profiling.

    Returns
    -------
    DelayProfile
        Finalized family profiles with real fit statuses and retry evidence.

    Notes
    -----
    Callers bound memory by supplying member blocks and schedule parallel work
    outside this function. No pool, model selection, masking or kinetic salvage
    occurs here. A classified numerical failure invalidates only that member's
    family profile, with its reason and point identity. Other exceptions abort
    the block; none is replaced by a fabricated objective or flat profile.
    """
    parameter_count = {"patlak": 3, "tofts": 2, "extended_tofts": 3, "2cxm": 4}
    if family not in parameter_count:
        raise ValueError(f"Unsupported conditional model: {family}")
    grid = np.array(
        np.arange(int(round(MAX_LOCAL_DELAY_SECONDS * 10)) + 1) / 10
        if delays_seconds is None else delays_seconds,
        dtype=float,
        copy=True,
    )
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or grid[0] != 0
        or grid[-1] > MAX_LOCAL_DELAY_SECONDS
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError(
            f"Delay grid must increase from zero within [0,{MAX_LOCAL_DELAY_SECONDS:g}] seconds."
        )
    initial_context = DelayedPchip(aif, times, 0.0, observed)
    indices = initial_context.observed
    if not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("Unique nonempty original observation indices required.")
    y = np.asarray(curves, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(indices) or not np.isfinite(y).all():
        raise ValueError("Finite member curves on the common observation set required.")
    shape = (len(grid), len(y))
    profile = DelayProfile(
        family,
        grid,
        indices.copy(),
        np.full((*shape, parameter_count[family]), np.nan),
        np.full(shape, np.nan),
        np.zeros(shape, dtype=np.int32),
        np.full(shape, -1, dtype=np.int64),
        np.full(shape, np.nan),
        np.full(shape, -1, dtype=np.int32),
        {},
    )
    if not len(y):
        return profile

    for i, delay in enumerate(grid):
        if family == "2cxm":
            native_context = NativeInput(aif, times, float(delay), indices)
            for member, curve in enumerate(y):
                if member in profile.failures:
                    continue
                seed = profile.params[i - 1, member].copy() if i else None
                try:
                    result = fit_native(native_context, curve, previous=seed)
                except NumericalFitError as error:
                    _invalidate_member(profile, member, i, "sweep", str(error))
                    continue
                except RuntimeError as error:
                    raise RuntimeError(
                        f"2CXM profile failed at delay {delay:g} s, member {member}: {error}"
                    ) from error
                if result.sse < 0:
                    raise RuntimeError("Native fit returned a negative residual sum of squares.")
                if (
                    not np.isfinite(result.params).all()
                    or not np.isfinite(result.sse)
                ):
                    _invalidate_member(profile, member, i, "sweep", "Invalid native fit result")
                    continue
                profile.params[i, member] = result.params
                profile.sse[i, member], profile.status[i, member] = (
                    result.sse,
                    result.status,
                )
                profile.evaluations[i, member] = result.nfev
                profile.optimality[i, member] = result.optimality
                profile.seed_index[i, member] = i - 1
        else:
            input_type = DelayedPchip if family == "patlak" else CachedPchip
            context = input_type(aif, times, float(delay), indices)
            fitted = (
                fit_patlak(context, y)
                if family == "patlak"
                else fit_tofts(context, y)
                if family == "tofts"
                else fit_extended_tofts(context, y)
            )
            required = fitted.params[:, [0, 2]] if family == "patlak" else fitted.params
            if np.any(fitted.sse < 0):
                raise RuntimeError(f"{family} returned a negative residual sum of squares.")
            invalid = (
                ~np.isfinite(required).all(axis=1)
                | ~np.isfinite(fitted.sse)
            )
            profile.params[i], profile.sse[i], profile.status[i] = (
                fitted.params,
                fitted.sse,
                1,
            )
            for member in np.flatnonzero(invalid).tolist():
                if member not in profile.failures:
                    _invalidate_member(
                        profile, int(member), i, "sweep", "Invalid profiled fit result"
                    )
            # Later batched results cannot restore an already incomplete member.
            for member in profile.failures:
                profile.params[i, member] = np.nan
                profile.sse[i, member] = np.nan
                profile.status[i, member] = 0

    if family == "2cxm":
        for point in np.argwhere(profile.status <= 0):
            i, member = int(point[0]), int(point[1])
            if member in profile.failures:
                continue
            native_context = NativeInput(aif, times, float(grid[i]), indices)
            initial = ExchangeFit(
                profile.params[i, member].copy(),
                float(profile.sse[i, member]),
                int(profile.status[i, member]),
                int(profile.evaluations[i, member]),
                float(profile.optimality[i, member]),
            )
            try:
                best, attempts = retry_exchange_fit(native_context, y[member], initial)
            except NumericalFitError as error:
                _invalidate_member(profile, int(member), int(i), "retry", str(error))
                continue
            except RuntimeError as error:
                raise RuntimeError(
                    f"2CXM retry failed at delay {grid[i]:g} s, member {member}: {error}"
                ) from error
            profile.params[i, member] = best.params
            profile.sse[i, member], profile.status[i, member] = best.sse, best.status
            profile.optimality[i, member] = best.optimality
            profile.retries[(int(i), int(member))] = attempts
    return profile
