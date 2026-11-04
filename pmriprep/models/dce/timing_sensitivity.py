"""Conditional NLLS timing diagnostics, separate from selection and central QC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmriprep.models.dce.delay_profiles import DelayProfile, ExchangeRetry, ProfileFamily, retry_exchange_fit
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.fit_errors import NumericalFitError
from pmriprep.models.dce.fixed_inference import evaluate_fixed_kinetics, selected_fit_result
from pmriprep.models.dce.identification import IdentificationResult, identify_fixed_point, identify_profile_point
from pmriprep.models.dce.local_selection import FamilyTiming
from pmriprep.models.dce.native_exchange import fit_native
from pmriprep.models.dce.profile_fitting import ProfileFit, ToftsFit, fit_extended_tofts, fit_patlak, fit_tofts
from pmriprep.models.dce.retention import is_degenerate_fit


@dataclass
class TimingOffset:
    """Raw perturbed fit and interpretable changes for one signed offset.

    Attributes
    ----------
    delay_seconds : float
        Actual diagnostic offset; never clipped to the selection grid.
    parameters : ndarray
        Raw physical profile coordinates, or NaN after a numerical failure.
    sse, status, evaluations : float or int
        Raw fitting evidence; evaluations are -1 for rate/linear fits or failure.
    retries : tuple of ExchangeRetry
        Ordered native retry evidence; empty for other families.
    retained : bool
        Numerical retention of this diagnostic, independent of central retention.
    identification : IdentificationResult or None
        Conditional diagnostic identification, unavailable after numerical failure.
    raw_changes, changes, available : ndarray
        Signed primary-parameter changes, their usable subset (NaN otherwise),
        and per-parameter availability in TimingSensitivity.names order.
    reasons : tuple of tuple of str
        Central/perturbed retention or identification reasons for unavailability.
    objective_change, objective_reason : float and str or None
        n*log(SSE_side/SSE_selected), or NaN with a zero-SSE/numerical reason.
    held_before, held_after : int
        Counts of original tissue observations using held input endpoints.
    failure_reason, failure_detail : str or None
        Numerical failure, nonconvergence or degeneracy, with optional detail.
    """

    delay_seconds: float
    parameters: np.ndarray
    sse: float
    status: int
    evaluations: int
    retries: tuple[ExchangeRetry, ...]
    retained: bool
    identification: IdentificationResult | None
    raw_changes: np.ndarray
    changes: np.ndarray
    available: np.ndarray
    reasons: tuple[tuple[str, ...], ...]
    objective_change: float
    objective_reason: str | None
    held_before: int
    held_after: int
    failure_reason: str | None
    failure_detail: str | None


@dataclass
class TimingSensitivity:
    """Minus/plus-one-frame NLLS diagnostics at the selected kinetic family.

    Attributes
    ----------
    family, names : str and tuple of str
        Unchanged family and applicable primary output names.
    selected_delay_seconds, frame_interval_seconds : float
        Selected offset and original acquisition interval, not a filtered interval.
    observations : ndarray
        Original retained indices shared by central and both perturbed fits.
    central_identification : IdentificationResult
        Selected NLLS identification, including its own-family profile excursion.
    central_retained : bool
        Central numerical retention, not the final binary QC mask.
    offsets : tuple of TimingOffset
        Minus then plus one frame. No diagnostic result can modify the centre.
    """

    family: ProfileFamily
    names: tuple[str, ...]
    selected_delay_seconds: float
    frame_interval_seconds: float
    observations: np.ndarray
    central_identification: IdentificationResult
    central_retained: bool
    offsets: tuple[TimingOffset, TimingOffset]


def timing_sensitivity(
    profile: DelayProfile, timing: FamilyTiming, member: int,
    aif: np.ndarray, times: np.ndarray, tissue: np.ndarray,
) -> TimingSensitivity:
    """Refit two signed offsets without selecting models or changing central QC.

    Parameters
    ----------
    profile, timing : DelayProfile and FamilyTiming
        Finalized selected-family profile and numerically available centre timing.
    member : int
        Centre index in the profile.
    aif, times : ndarray
        Complete plasma input and original uniform acquisition times in minutes.
    tissue : ndarray
        Original tissue observations in profile.observed order.

    Returns
    -------
    TimingSensitivity
        Raw NLLS fits and per-parameter usable signed changes. Both endpoints
        must be numerically retained and pass the adopted identification screen.
        VI reporting uses this same NLLS diagnostic, not posterior differences.

    Notes
    -----
    Selected CX kinetics supplement the generic starts; retries use the existing
    fixed-delay rule. Only numerical failures become unavailable diagnostics.
    Structural input, backend and process errors propagate to the caller.
    """
    central = identify_profile_point(profile, member, timing.index, aif, times)
    seed = selected_fit_result(profile, timing, member, aif, times, tissue)
    family = profile.family
    retained = not is_degenerate_fit(family, seed.parameters)
    params = profile.params[timing.index, member].copy()
    selected_delay = float(profile.delays_seconds[timing.index])
    interval = float((times[1] - times[0]) * 60)
    selected_sse = float(profile.sse[timing.index, member])
    tissue = np.asarray(tissue, dtype=float)
    sides = []
    for delay in (selected_delay - interval, selected_delay + interval):
        context = DelayedPchip(aif, times, delay, profile.observed)
        p, sse, status, evaluations = np.full_like(params, np.nan), np.nan, 0, -1
        attempts: tuple[ExchangeRetry, ...] = ()
        ident = None
        side_retained = False
        failure = detail = None
        try:
            if family == "2cxm":
                initial = fit_native(context, tissue, previous=params)
                fit, attempts = retry_exchange_fit(context, tissue, initial)
                p, sse, status = fit.params, fit.sse, fit.status
                evaluations = initial.nfev + sum(a.candidate.nfev for a in attempts)
            else:
                result: ProfileFit | ToftsFit
                if family == "patlak":
                    result = fit_patlak(context, tissue[None])
                elif family == "tofts":
                    result = fit_tofts(context, tissue[None])
                else:
                    result = fit_extended_tofts(context, tissue[None])
                p, sse, status = result.params[0], float(result.sse[0]), 1
            physical = p[[0, 2]] if family == "patlak" else p
            if not np.isfinite(physical).all() or not np.isfinite(sse) or sse < 0:
                raise NumericalFitError("Nonfinite diagnostic fit or invalid residual sum.")
            predicted = evaluate_fixed_kinetics(family, context, physical)
            reconstructed = float(np.sum((predicted - tissue) ** 2))
            if not np.isfinite(predicted).all() or not np.isclose(sse, reconstructed, rtol=0, atol=1e-12):
                raise RuntimeError("Diagnostic fit does not match the fixed operator/data.")
            ident = identify_fixed_point(family, context, p, sse)
            side_retained = status > 0 and not is_degenerate_fit(family, dict(zip(ident.names, ident.values)))
            if status <= 0:
                failure = "nonconverged"
            elif not side_retained:
                failure = "degenerate_fit"
        except (NumericalFitError, FloatingPointError, np.linalg.LinAlgError) as error:
            failure, detail = "numerical_failure", str(error)
        raw = np.full(len(central.primary), np.nan)
        reasons = []
        for i, name in enumerate(central.primary):
            why = []
            if not retained:
                why.append("central_not_retained")
            why.extend("central_" + r for r in central.reasons[central.names.index(name)])
            if not side_retained:
                why.append("perturbed_" + (failure or "not_retained"))
            if ident is not None:
                j = ident.names.index(name)
                raw[i] = ident.values[j] - central.values[central.names.index(name)]
                why.extend("perturbed_" + r for r in ident.reasons[j])
            reasons.append(tuple(why))
        available = np.array([not why for why in reasons])
        objective, objective_reason = np.nan, None
        if not np.isfinite(sse) or failure == "numerical_failure":
            objective_reason = "numerical_failure"
        elif selected_sse == 0:
            objective_reason = "zero_selected_sse"
        elif sse == 0:
            objective_reason = "zero_perturbed_sse"
        else:
            objective = len(profile.observed) * (np.log(sse) - np.log(selected_sse))
        sides.append(TimingOffset(
            delay, p.copy(), sse, status, evaluations, attempts, side_retained, ident,
            raw, np.where(available, raw, np.nan), available, tuple(reasons),
            float(objective), objective_reason, int(context.held_before.sum()),
            int(context.held_after.sum()), failure, detail,
        ))
    return TimingSensitivity(
        family, central.primary, selected_delay, interval, profile.observed.copy(),
        central, retained, (sides[0], sides[1]),
    )
