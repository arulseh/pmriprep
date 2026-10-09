"""Acquisition preparation and ownership-safe signal fitting with optional leakage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Literal, Sequence, cast

import numpy as np
from scipy.integrate import cumulative_trapezoid

from pmriprep.models.dsc.local_vi import LocalVIFactorError
from pmriprep.utils.bsw_guidance import BSWGuidance, BSWGuidanceStatus

from .leakage import (
    BSWPrior, LeakageBatch, LeakageReference, fit_with_prior, fit_without_leakage, leakage_count,
)
from .geometry import Model
from .policy import CountDecision, FitAssessment, automatic_count
from .probability import Array, SharedFit
from .weights import checked_weights
from .settings import CPIOptions
from .signal import SIGNAL_BASELINE_PRIOR_MEAN


def _readonly(value: np.ndarray) -> np.ndarray:
    """Own a read-only snapshot so caller mutation cannot change a prepared fit."""
    result = value.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class GuidedAcquisition:
    """Validated signal acquisition with one count and optional BSW priors.

    Arrays use flattened spatial order, matching a C-order image reshape.
    Signals are divided by one acquisition-wide scale before inference;
    ``signal_scale`` converts fitted signals back to the input magnitude units.
    Reference uncertainty is conditional on the selected BSW membership.
    This object does not hold corrected concentration or execute fallback.
    """

    spatial_shape: tuple[int, ...]
    signals: Array
    aif: Array
    time: Array
    te: float
    k2: Array
    k2_sd: Array
    guidance_status: np.ndarray | None
    valid_signal: np.ndarray
    reference: LeakageReference | None
    reference_reason: str | None
    decision: CountDecision | None
    inference_method: Literal["vi", "svb"] = "vi"
    weights: Array | None = None
    family: Model = "cpi"
    leakage_method: Literal["bsw", "none"] = "bsw"
    support: np.ndarray | None = None
    infer_delay: bool = True
    sloppy: bool = False
    signal_scale: float = 1.0

    @property
    def fit_support(self) -> np.ndarray:
        """Return image support independently of coefficient availability."""
        if self.support is not None:
            return self.support
        if self.guidance_status is None:
            raise ValueError("fitting support is required without BSW guidance")
        return np.asarray(self.guidance_status != BSWGuidanceStatus.OUTSIDE_SUPPORT, dtype=bool)


@dataclass(frozen=True)
class GuidedVoxel:
    """Selected native-unit measurements or an explicit replacement request.

    CBF and CBV are the forward model's flow and volume fraction before the
    compute interface's scaling; MTT and arrival delay are seconds. K2 is s^-1.
    Rejected measurements are NaN; diagnostics retain the candidate endpoint.
    No fallback is executed here, and a provisional fit is not automatically
    rejected. Numerical acceptance does not establish biological accuracy.
    The assessment uses the image's flat index; detailed restart diagnostics
    retain their single-voxel local ownership coordinates.
    CBF standard deviation uses the same native units as CBF. The owned signal
    prediction is restored to the input magnitude units and is a lightweight
    readout of the selected state, not a new fit.
    """

    flat_index: int
    status: Literal["retained", "provisional", "fallback_requested", "outside_support"]
    reasons: tuple[str, ...]
    cbf: float = np.nan
    cbv: float = np.nan
    mtt: float = np.nan
    delay: float = np.nan
    bsw_k2: float = np.nan
    fitted_k2: float = np.nan
    assessment: FitAssessment | None = None
    diagnostics: LeakageBatch | None = None
    cbf_std: float = np.nan
    fitted_signal: Array | None = None
    leakage_method: Literal["bsw", "none"] = "bsw"
    family: Model | None = None
    selection_scores: tuple[tuple[str, float | None], ...] = ()


def _normalize_signals(
    rows: Array,
    pool: np.ndarray,
    baseline_frames: int | None,
) -> tuple[Array, float]:
    """Put one acquisition on CPI's numerical signal scale.

    The multiplier is common to every voxel and frame, so signal ratios,
    delta-R2-star, tissue contrast and all perfusion measurements are
    unchanged. The robust prebolus reference only adapts arbitrary scanner
    magnitude units to the finite S0 prior used by signal-domain inference.
    """
    if baseline_frames is None or not np.any(pool):
        return _readonly(rows), 1.0
    baselines = np.mean(rows[pool, :baseline_frames], axis=1)
    reference = float(np.median(baselines))
    scale = reference / SIGNAL_BASELINE_PRIOR_MEAN
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("CPI signal normalization requires a positive finite baseline")
    normalized = rows / scale
    if np.any(np.isfinite(rows) & ~np.isfinite(normalized)):
        raise ValueError("CPI signal normalization produced nonfinite values")
    return _readonly(normalized), scale


def _restore_fitted_signal(row: GuidedVoxel, signal_scale: float) -> GuidedVoxel:
    """Restore a selected prediction to the acquisition's input units."""
    if row.fitted_signal is None or signal_scale == 1.0:
        return row
    return replace(row, fitted_signal=_readonly(row.fitted_signal * signal_scale))


def prepare_acquisition(
    signals: Array,
    aif: Array,
    time: Array,
    te: float,
    guidance: BSWGuidance | None = None,
    *,
    inference_method: str = "vi",
    temporal_weights: Array | None = None,
    family: Model = "cpi",
    leakage_method: Literal["bsw", "none"] = "bsw",
    mask: np.ndarray | None = None,
    baseline_frames: int | None = None,
    control_points: int = 0,
    infer_delay: bool = True,
    sloppy: bool = False,
) -> GuidedAcquisition:
    """Prepare measured signals with BSW guidance or without leakage estimation.

    Parameters
    ----------
    signals : ndarray
        Original measured signal, shaped (x, y, z, frame), not corrected signal.
    aif, time : ndarray
        Delta-R2-star AIF and uniform acquisition seconds starting at zero.
    te : float
        Echo time in seconds; no sequence-specific default is supplied.
    guidance : BSWGuidance, optional
        Reference and signed per-voxel coefficient/SE from the same correction.
        Required for bsw; ignored completely for none.
    inference_method : str, optional
        Ordinary VI or spatial VB. Spatial VB uses the complete field through
        fit_spatial_voxels, not independently fitted subsets.
    temporal_weights : ndarray, optional
        Frame likelihood downweights. BSW guidance, count and starts retain
        their original unweighted input semantics.
    family : {"cpi", "cpi_vtf"}, optional
        Plain CPI or CPI with gamma arterial transport and two extra inferred
        coordinates. Control-point feasibility accounts for this choice.
    leakage_method : {"bsw", "none"}, optional
        Infer leakage with BSW guidance, or omit leakage and reference-related
        coordinates entirely. Disabled leakage is not a zero-valued estimate.
    mask : ndarray, optional
        Boolean spatial fitting mask. Without it, use BSW support for bsw or
        all image voxels for none. With bsw, intersect with its existing support.
    baseline_frames : int, optional
        Prebolus window for automatic count selection. BSW uses its recorded
        window and requires agreement if supplied here. Without leakage, use
        this window or the AIF-derived initializer window.
    control_points : int, optional
        Zero selects automatically. A manual count must be supported by the
        acquisition; it is never silently clipped to another count.
    infer_delay : bool, optional
        Infer arrival or condition it at zero without a delay posterior.
    sloppy : bool, optional
        Use the shared reduced-iteration test schedule, not quantitative defaults.

    Returns
    -------
    GuidedAcquisition
        Immutable input snapshots and a single automatic acquisition count,
        reusable for arbitrary processing order and subsets. Bad voxel signals
        remain identifiable without contaminating the count-selection pool.
    """
    if inference_method not in ("vi", "svb"):
        raise ValueError("signal CPI supports vi or svb only")
    CPIOptions(control_points=control_points, infer_delay=infer_delay)
    if family not in ("cpi", "cpi_vtf"):
        raise ValueError("family must be cpi or cpi_vtf")
    if leakage_method not in ("bsw", "none"):
        raise ValueError("leakage_method must be bsw or none")
    signals, aif, time = (np.asarray(v, dtype=float) for v in (signals, aif, time))
    if signals.ndim != 4 or any(n == 0 for n in signals.shape):
        raise ValueError("nonempty spatial-by-frame signal image required")
    frames = signals.shape[-1]
    weights = checked_weights(temporal_weights, frames)
    if (frames < 20 or aif.shape != (frames,) or time.shape != (frames,)
            or not np.all(np.isfinite(aif)) or not np.all(np.isfinite(time))
            or np.max(aif) <= 0 or time[0] != 0 or np.any(np.diff(time) <= 0)
            or not np.isfinite(te) or te <= 0):
        raise ValueError("finite matching AIF/time and positive TE required")
    if mask is None:
        support = np.ones(signals.shape[:-1], dtype=bool)
    else:
        support = np.asarray(mask)
        if support.dtype != np.bool_ or support.shape != signals.shape[:-1]:
            raise ValueError("boolean mask on the signal spatial grid required")
    if baseline_frames is not None and (
        isinstance(baseline_frames, (bool, np.bool_))
        or not isinstance(baseline_frames, (int, np.integer))
        or not 0 < baseline_frames < frames
    ):
        raise ValueError("baseline_frames must be an interior positive integer")
    rows = signals.reshape(-1, frames)
    valid = np.all(np.isfinite(rows) & (rows > 0), axis=1)
    if leakage_method == "none":
        if not np.allclose(np.diff(time), time[1] - time[0], rtol=1e-10, atol=1e-12):
            raise ValueError("uniform acquisition timing required")
        pool = valid & support.ravel()
        decision = automatic_count(
            rows[pool], aif, time, te, baseline_frames=baseline_frames, family=family,
        ) if np.any(pool) else None
        decision = requested_count(decision, control_points)
        missing = np.full(rows.shape[0], np.nan)
        normalized, signal_scale = _normalize_signals(
            rows, pool, None if decision is None else decision.baseline_frames,
        )
        return GuidedAcquisition(
            signals.shape[:-1], normalized, _readonly(aif), _readonly(time), float(te),
            _readonly(missing), _readonly(missing), None, _readonly(valid), None, None, decision,
            cast(Literal["vi", "svb"], inference_method), weights, family, "none",
            _readonly(support.ravel()), infer_delay, sloppy, signal_scale=signal_scale,
        )
    if guidance is None:
        raise ValueError("BSW guidance required when leakage_method is bsw")
    if (not np.isfinite(guidance.tr) or guidance.tr <= 0
            or not np.allclose(np.diff(time), guidance.tr, rtol=1e-10, atol=1e-12)):
        raise ValueError("exact BSW timing required")
    if (isinstance(guidance.baseline_frames, (bool, np.bool_))
            or not isinstance(guidance.baseline_frames, (int, np.integer))
            or not 0 < guidance.baseline_frames < frames
            or guidance.degrees_of_freedom != frames - 2):
        raise ValueError("BSW baseline window or regression degrees of freedom mismatch")
    if baseline_frames is not None and baseline_frames != guidance.baseline_frames:
        raise ValueError("baseline_frames must match the BSW baseline window")
    if any(v.shape != signals.shape[:-1]
           for v in (guidance.k2, guidance.k2_sd, guidance.status)):
        raise ValueError("signal and BSW guidance spatial shapes must agree")
    if not np.all(np.isin(guidance.status, list(BSWGuidanceStatus))):
        raise ValueError("unknown BSW guidance status")
    available = guidance.available
    if np.any(available & (~np.isfinite(guidance.k2) | ~np.isfinite(guidance.k2_sd)
                           | (guidance.k2_sd <= 0))):
        raise ValueError("available BSW guidance must contain finite K2 and positive SE")
    reference = None
    if guidance.reference_reason is None:
        curve, integral, basis = (
            guidance.reference_curve, guidance.reference_integral, guidance.reference_basis
        )
        if (curve is None or integral is None or basis is None
                or curve.shape != (frames,) or integral.shape != (frames,)
                or not np.all(np.isfinite(curve))
                or not np.allclose(integral, cumulative_trapezoid(curve, time, initial=0),
                                   rtol=1e-10, atol=1e-12)):
            raise ValueError("BSW reference curve and integral must agree on the acquisition")
        reference = LeakageReference(time, integral, basis, "empirical_iid_region_mean",
                                     guidance.reference_count)
    elif np.any(available):
        raise ValueError("available voxel prior cannot have unavailable reference guidance")
    if guidance.reference_mask is not None:
        if (guidance.reference_mask.shape != signals.shape[:-1]
                or np.count_nonzero(guidance.reference_mask) != guidance.reference_count):
            raise ValueError("BSW reference membership/count mismatch")
    status = guidance.status.ravel()
    support = support.ravel() & (status != BSWGuidanceStatus.OUTSIDE_SUPPORT)
    pool = valid & support
    decision = leakage_count(rows[pool], aif, time, te,
                             baseline_frames=guidance.baseline_frames,
                             family=family) if np.any(pool) else None
    decision = requested_count(decision, control_points)
    normalized, signal_scale = _normalize_signals(
        rows, pool, None if decision is None else decision.baseline_frames,
    )
    return GuidedAcquisition(
        signals.shape[:-1], normalized, _readonly(aif), _readonly(time), float(te),
        _readonly(guidance.k2.ravel()), _readonly(guidance.k2_sd.ravel()),
        _readonly(status), _readonly(valid), reference, guidance.reference_reason, decision,
        cast(Literal["vi", "svb"], inference_method), weights, family, "bsw", _readonly(support),
        infer_delay, sloppy, signal_scale=signal_scale,
    )


def requested_count(decision: CountDecision | None, count: int) -> CountDecision | None:
    """Resolve a manual count inside the same frozen acquisition support."""
    if count == 0 or decision is None:
        return decision
    if count not in decision.allowed:
        raise ValueError(f"cpi_control_points={count} is infeasible; supported counts: {decision.allowed}")
    return replace(decision, requested=count, selected=count, selection_method="manual",
                   restrictions=())


def classify_result(index: int, bsw_k2: float, result: LeakageBatch) -> GuidedVoxel:
    """Convert a checked selection to native units without discarding its diagnostics.

    Flow within one float64 epsilon of zero in native inverse-second units is
    numerical boundary contact, not a positive perfusion measurement. This
    output-only tolerance is about 1.3e-12 in the model's scaled CBF units; it
    is not a physiological flow floor or an uncertainty threshold. At that
    boundary, the residue shape does not supply a measured transit time.
    """
    if result.leakage_method == "none":
        bsw_k2 = np.nan
    row = result.rows[0]
    fit: SharedFit | None = row["selected"]
    assessment: FitAssessment | None = row["assessment"]
    prediction = None
    cbf_std = np.nan
    if fit is not None:
        if "owners" in result.arrays and "restarts" in result.arrays:
            matches = np.flatnonzero(
                (np.asarray(result.arrays["owners"]) == fit.voxel)
                & (np.asarray(result.arrays["restarts"]) == fit.restart)
            )
            if len(matches) != 1:
                raise ValueError("selected CPI fit has ambiguous restart ownership")
            selected = int(matches[0])
        else:
            selected = next(i for i, value in enumerate(result.fits)
                            if value.restart == fit.restart and value.voxel == fit.voxel)
        if "fitted_curves" in result.arrays:
            prediction = _readonly(result.arrays["fitted_curves"][selected])
        if "covariance" in result.arrays:
            variance = float(result.arrays["covariance"][selected, 0, 0])
            if np.isfinite(variance) and variance >= 0:
                cbf_std = float(np.sqrt(variance))
    if assessment is not None:
        assessment = replace(assessment, voxel=index)
    reasons: list[str] = []
    if fit is None or assessment is None:
        numerical_failures = tuple(
            dict.fromkeys(
                reason for reason in result.arrays.get("failure_reasons", ())
                if isinstance(reason, str)
            )
        )
        reasons.extend(numerical_failures or ("no_valid_variational_state",))
    else:
        if not assessment.valid:
            reasons.extend(assessment.reasons or ("invalid_variational_state",))
        if not np.all(np.isfinite([fit.cbf, fit.cbv, fit.mtt, fit.delay])):
            reasons.append("nonfinite_required_measurement")
        if fit.cbf <= 0 or fit.cbv <= 0 or fit.mtt <= 0:
            reasons.append("nonpositive_perfusion_measurement")
        elif fit.cbf / 6000 <= np.finfo(np.float64).eps:
            reasons.append("flow_at_zero_bound")
        if (assessment.r_squared is None or not np.isfinite(assessment.r_squared)
                or assessment.r_squared < 0):
            reasons.append("inadequate_signal_fit")
    if reasons:
        return GuidedVoxel(index, "fallback_requested", tuple(reasons), bsw_k2=bsw_k2,
                           assessment=assessment, diagnostics=result, fitted_signal=prediction,
                           leakage_method=result.leakage_method)
    assert fit is not None and assessment is not None
    fitted_k2 = np.nan
    if result.leakage_method == "bsw":
        leakage = next(x for x in result.leakage if x["restart"] == fit.restart)
        fitted_k2 = float(leakage["k2"])
    return GuidedVoxel(
        index, "retained" if fit.converged else "provisional", (),
        fit.cbf / 6000, fit.cbv / 100, fit.mtt, fit.delay if result.infer_delay else np.nan, bsw_k2,
        fitted_k2, assessment, result, cbf_std, prediction, result.leakage_method,
    )


def fit_voxels(
    acquisition: GuidedAcquisition, indices: Sequence[int] | None = None,
) -> Iterator[GuidedVoxel]:
    """Yield independent voxel results with one unchanged acquisition-level count.

    Parameters
    ----------
    acquisition : GuidedAcquisition
        Prepared full input pool, before dividing work into subsets.
    indices : sequence of int, optional
        Unique flat spatial indices in requested processing order. By default,
        fit acquisition support only. Each result owns only that voxel's posterior;
        consume and release results to bound full-covariance memory use.

    Yields
    ------
    GuidedVoxel
        Measurement or explicit fallback request with the original image index.

    Notes
    -----
    Unexpected engine exceptions propagate; they do not masquerade as ordinary
    failed voxels. No global model patch or prior shared across voxels is used.
    """
    a = acquisition
    if a.inference_method != "vi":
        raise ValueError("use fit_spatial_voxels for the complete spatial field")
    if indices is None:
        indices = np.flatnonzero(a.fit_support).tolist()
    if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            or not 0 <= i < len(a.signals) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("unique in-range integer spatial indices required")
    for raw_index in indices:
        i = int(raw_index)
        if not a.fit_support[i]:
            reason = "outside_bsw_support" if a.leakage_method == "bsw" else "outside_fit_support"
            yield GuidedVoxel(i, "outside_support", (reason,), leakage_method=a.leakage_method)
            continue
        reasons = []
        if not a.valid_signal[i]:
            reasons.append("invalid_measured_signal")
        if a.leakage_method == "bsw":
            assert a.guidance_status is not None
            status = BSWGuidanceStatus(int(a.guidance_status[i]))
            if status != BSWGuidanceStatus.AVAILABLE:
                reasons.append("bsw_" + status.name.lower())
                if a.reference_reason is not None:
                    reasons.append(a.reference_reason)
        if reasons:
            yield GuidedVoxel(i, "fallback_requested", tuple(reasons), bsw_k2=float(a.k2[i]),
                              leakage_method=a.leakage_method)
            continue
        assert a.decision is not None
        try:
            if a.leakage_method == "none":
                result = fit_without_leakage(
                    a.signals[i], a.aif, a.time, a.te, a.decision.selected,
                    weights=a.weights, family=a.family, infer_delay=a.infer_delay,
                    sloppy=a.sloppy,
                )
            else:
                assert a.reference is not None
                result = fit_with_prior(
                    a.signals[i], a.aif, a.time, a.te, a.decision.selected,
                    a.reference, BSWPrior(float(a.k2[i]), float(a.k2_sd[i])),
                    weights=a.weights, family=a.family, infer_delay=a.infer_delay,
                    sloppy=a.sloppy,
                )
        except LocalVIFactorError:
            yield GuidedVoxel(
                i, "fallback_requested", ("local_factor_failed",),
                bsw_k2=float(a.k2[i]), leakage_method=a.leakage_method,
            )
            continue
        row = classify_result(i, float(a.k2[i]), result)
        yield replace(_restore_fitted_signal(row, a.signal_scale), family=a.family)
