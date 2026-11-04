"""Primary-parameter identification from conditional NLLS delay profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pmriprep.models.dce.conditional_covariance import covariance, feasible, jacobian
from pmriprep.models.dce.delay_profiles import DelayProfile, ProfileFamily
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.fixed_inference import evaluate_fixed_kinetics
from pmriprep.models.dce.local_selection import validate_profile
from pmriprep.utils.dce_constants import FP_BOUNDS, KTRANS_BOUNDS, PS_BOUNDS, VE_BOUNDS, VP_BOUNDS

PRIMARY_PARAMETERS: dict[ProfileFamily, tuple[str, ...]] = {
    "patlak": ("Ktrans", "vp"),
    "tofts": ("Ktrans", "ve"),
    "extended_tofts": ("Ktrans", "ve", "vp"),
    "2cxm": ("Ktrans", "ve", "vp"),
}


def identification_policy() -> dict[str, object]:
    """Describe the binary-mask identification rule for derivative sidecars.

    Returns
    -------
    dict
        Selected-NLLS basis, numerical conventions and interpretation limits.
        Individual Fp/PS warnings remain separate from the primary conjunction.
    """
    return {
        "basis": "selected_nlls",
        "description": (
            "QC combines fit-quality checks on the reported estimate with "
            "data-identification checks evaluated at the corresponding NLLS fit. "
            "VI posterior uncertainty is reported separately and is not used "
            "for identification screening."
        ),
        "primary_parameters": {name: list(parameters) for name, parameters in PRIMARY_PARAMETERS.items()},
        "individual_fp_ps_required": False,
        "effective_uncertainty": "sqrt(conditional_SE^2 + profile_excursion^2)",
        "maximum_effective_relative_uncertainty_inclusive": 0.5,
        "profile_excursion": "maximum own-family departure from selected value over Q <= min(Q) + 1",
        "profile_objective": "n * log(max(SSE, 1e-30)); n is the number of observed frames",
        "covariance": "SSE/(n-k) times inverse full kinetic information; conditional on AIF, family and delay",
        "minimum_information_rcond_inclusive": 1e-10,
        "maximum_half_step_se_relative_change_inclusive": 1e-3,
        "box_bound_exclusion": "endpoints or distance < 2 SE; lower and upper reasons separate",
        "overlapping_reasons": (
            "For positive estimates with finite SE and a zero lower bound, "
            "value < 2 SE and SE > 50% of value are equivalent. Thus "
            "lower_box_bound and relative_uncertainty can describe the same "
            "failure and must not be added. lower_box_bound includes a "
            "precision margin; it does not imply optimizer contact with the bound."
        ),
        "joint_volume_exclusion": "slack <= 2 SE, using full covariance",
        "missing_evidence": "fails identification; never treated as zero uncertainty",
        "interpretation": (
            "Screening convention, not an accuracy guarantee or certification of Fp/PS or model adequacy. "
            "Expected near-zero values can fail; QC-filtered low-permeability summaries may be biased upward. "
            "Quantitative maps and retention are unchanged by this screen."
        ),
    }


@dataclass
class IdentificationResult:
    """NLLS identification evidence, independent of retention and fit-quality QC.

    Attributes
    ----------
    names, primary : tuple of str
        Reported parameters and the subset required by the binary QC rule.
    values, se, half_se, spread, effective : ndarray
        Physical estimates, conditional SE at two derivative step sizes,
        own-family profile excursions and sqrt(SE squared + excursion squared).
    covariance : ndarray
        Full covariance in physical kinetic coordinates, not output coordinates.
    rcond, half_rcond : float
        Normalized information reciprocal conditions at the two step sizes.
    covariance_reason, half_covariance_reason : str
        Availability or rank-failure reason for each covariance calculation.
    schemes, half_schemes : tuple of str
        Per-coordinate feasible derivative stencils.
    step_change : float
        Maximum relative SE change, including joint-volume slack when applicable.
    slack, slack_se : float
        Joint-volume slack and its full-covariance SE; NaN if inapplicable.
    reasons : tuple of tuple of str
        Per-output failures; an empty tuple means identification passed.
        Lower/upper box margins are distinct from joint-volume and precision flags.
    """

    names: tuple[str, ...]
    primary: tuple[str, ...]
    values: np.ndarray
    se: np.ndarray
    half_se: np.ndarray
    spread: np.ndarray
    effective: np.ndarray
    covariance: np.ndarray
    rcond: float
    half_rcond: float
    covariance_reason: str
    half_covariance_reason: str
    schemes: tuple[str, ...]
    half_schemes: tuple[str, ...]
    step_change: float
    slack: float
    slack_se: float
    reasons: tuple[tuple[str, ...], ...]

    @property
    def passed(self) -> np.ndarray:
        """Return each output's identification status, not its retention status."""
        return np.array([not reason for reason in self.reasons], dtype=bool)

    @property
    def primary_pass(self) -> bool:
        """Require every primary output; individual Fp/PS warnings are separate."""
        return all(not self.reasons[self.names.index(name)] for name in self.primary)


def _covariance(derivative: np.ndarray, sse: float) -> tuple[np.ndarray, float, str]:
    """Report numerical decomposition failure without treating missing SE as zero."""
    try:
        return covariance(derivative, sse)
    except np.linalg.LinAlgError:
        size = derivative.shape[1]
        return np.full((size, size), np.nan), 0.0, "covariance_unavailable"


def evaluate_identification(
    p: np.ndarray, upper: np.ndarray, volumes: tuple[int, ...],
    predict: Callable[[np.ndarray], np.ndarray], sse: float,
    values: np.ndarray, gradients: np.ndarray, output_upper: np.ndarray,
    spread: np.ndarray, joint_outputs: tuple[int, ...], *,
    names: tuple[str, ...], primary: tuple[str, ...],
) -> IdentificationResult:
    """Evaluate conditional NLLS precision, profile excursions and bound margins.

    Parameters
    ----------
    p, upper : ndarray
        Physical kinetic coordinates and upper bounds; lower bounds are zero.
    volumes : tuple of int
        Coordinates subject to the joint-volume constraint, or empty.
    predict : callable
        Fixed-family, fixed-delay prediction on the original observations.
    sse : float
        Unweighted NLLS residual sum of squares at these coordinates.
    values, gradients, output_upper : ndarray
        Reported outputs, their output-by-coordinate gradients and upper bounds.
    spread : ndarray
        Maximum own-family output excursion in the one-unit profile set.
        Missing profile evidence is NaN, never zero.
    joint_outputs : tuple of int
        Outputs to which the joint-volume warning applies.
    names, primary : tuple of str
        Unique output names and nonempty subset required for identification.

    Returns
    -------
    IdentificationResult
        All outputs retain their reasons. Global covariance/rank/SE-instability
        failures affect every output even when an individual output is optional.
        This calculation neither modifies inputs nor combines fit-quality QC.
    """
    p, upper, values, gradients, output_upper, spread = (
        np.asarray(a, dtype=float) for a in (p, upper, values, gradients, output_upper, spread)
    )
    if (
        p.ndim != 1 or not len(p) or upper.shape != p.shape
        or values.shape != (len(names),) or gradients.shape != (len(names), len(p))
        or output_upper.shape != values.shape or spread.shape != values.shape
        or len(set(names)) != len(names) or not primary or not set(primary) <= set(names)
        or not np.isfinite(upper).all() or np.any(upper <= 0)
        or not np.isfinite(output_upper).all() or np.any(output_upper <= 0)
        or any(i < 0 or i >= len(p) for i in volumes)
        or any(i < 0 or i >= len(names) for i in joint_outputs)
        or np.any(spread < 0)
    ):
        raise ValueError("Aligned kinetic/output coordinates, bounds and primary subset required.")
    if feasible(p, upper, volumes):
        derivative, schemes = jacobian(p, upper, volumes, predict)
        half, half_schemes = jacobian(p, upper, volumes, predict, .5)
    else:
        derivative = half = np.full((0, len(p)), np.nan)
        schemes = half_schemes = ["unavailable"] * len(p)
    cov, rcond, reason = _covariance(derivative, sse)
    half_cov, half_rcond, half_reason = _covariance(half, sse)
    se = np.sqrt(np.maximum(np.diag(gradients @ cov @ gradients.T), 0))
    half_se = np.sqrt(np.maximum(np.diag(gradients @ half_cov @ gradients.T), 0))
    slack = 1 - float(p[list(volumes)].sum()) if volumes else np.nan
    slack_se = half_slack_se = np.nan
    if volumes:
        g = np.zeros(len(p))
        g[list(volumes)] = -1
        slack_se = float(np.sqrt(max(float(g @ cov @ g), 0)))
        half_slack_se = float(np.sqrt(max(float(g @ half_cov @ g), 0)))
    finite = bool(reason == half_reason == "ok" and np.isfinite(se).all() and np.isfinite(half_se).all())
    change = np.nan
    if finite:
        a, b = se, half_se
        if volumes:
            a, b = np.r_[a, slack_se], np.r_[b, half_slack_se]
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        if finite:
            denominator = np.maximum(abs(a), abs(b))
            change = float(np.divide(abs(a - b), denominator, out=np.zeros_like(a), where=denominator > 0).max())
    effective = np.sqrt(se**2 + spread**2)
    reasons = []
    for i, value in enumerate(values):
        why = []
        if not finite:
            why.append("rank_deficient" if "rank_deficient" in (reason, half_reason) else "covariance_unavailable")
        elif change > 1e-3:
            why.append("derivative_unstable")
        if value <= 0 or (np.isfinite(se[i]) and value < 2 * se[i]):
            why.append("lower_box_bound")
        if value >= output_upper[i] or (np.isfinite(se[i]) and output_upper[i] - value < 2 * se[i]):
            why.append("upper_box_bound")
        if i in joint_outputs and np.isfinite(slack_se) and slack <= 2 * slack_se:
            why.append("joint_volume_bound")
        if np.isfinite(se[i]) and se[i] > .5 * abs(value):
            why.append("relative_uncertainty")
        elif np.isfinite(effective[i]) and effective[i] > .5 * abs(value):
            why.append("profile_spread")
        if not np.isfinite(value) or not np.isfinite(spread[i]):
            why.append("output_unavailable")
        reasons.append(tuple(why))
    return IdentificationResult(
        names, primary, values.copy(), se, half_se, spread.copy(), effective,
        cov, rcond, half_rcond, reason, half_reason, tuple(schemes), tuple(half_schemes),
        change, slack, slack_se, tuple(reasons),
    )


def _outputs(family: ProfileFamily, parameters: np.ndarray) -> np.ndarray:
    """Convert one or multiple profile points to reported physical coordinates."""
    if family == "patlak":
        return parameters[..., [0, 2]].copy()
    if family != "2cxm":
        return parameters.copy()
    fp, ps, ve, vp = np.moveaxis(parameters, -1, 0)
    total = fp + ps
    kt = np.divide(fp * ps, total, out=np.full_like(total, np.nan), where=total > 0) / 100
    return np.stack([kt, ve, vp, fp, ps], axis=-1)


def identify_profile_point(
    profile: DelayProfile, member: int, index: int,
    aif: np.ndarray, times: np.ndarray,
) -> IdentificationResult:
    """Assess the selected NLLS point without fitting, selecting or masking data.

    Parameters
    ----------
    profile : DelayProfile
        Own-family conditional profiles and common original observation indices.
        Finite nonconverged points remain measured profile evidence; numerical
        retention is the caller's separate responsibility.
    member, index : int
        Centre member and selected delay indices, not necessarily the free minimum.
    aif, times : ndarray
        Full plasma input and original uniform acquisition times in minutes.

    Returns
    -------
    IdentificationResult
        Identification at the selected NLLS estimate. Profile excursions use
        Q <= min(Q) + 1, with Q = n*log(max(SSE, 1e-30)); n counts observations,
        never neighbors or the full input history. An incomplete profile has
        unavailable excursions. Fixed-zero profiles have zero excursion.

    Notes
    -----
    This evaluates data identification at the selected NLLS fit for both NLLS
    and VI reporting. VI posterior uncertainty is separate. The final QC mask
    also requires fit-quality checks on the reported estimate; this function
    does not apply those checks, replace estimates or change model admission.
    """
    spread = profile_excursion(profile, member, index)
    context = DelayedPchip(aif, times, float(profile.delays_seconds[index]), profile.observed)
    return identify_fixed_point(
        profile.family, context, profile.params[index, member],
        float(profile.sse[index, member]), spread=spread,
    )


def profile_excursion(profile: DelayProfile, member: int, index: int) -> np.ndarray:
    """Extract compact own-family identification evidence before slice release.

    Parameters
    ----------
    profile : DelayProfile
        Conditional family profiles on common original observations.
    member, index : int
        Voxel and selected delay indices.

    Returns
    -------
    ndarray
        Maximum output departure within one Q unit of this voxel's profile
        minimum, in reported-output order. Missing evidence is NaN. Only this
        vector, not the whole slice profile, needs transport to a worker.
    """
    validate_profile(profile)
    if not 0 <= member < profile.sse.shape[1] or not 0 <= index < len(profile.delays_seconds):
        raise ValueError("Existing member and selected delay indices required.")
    family = profile.family
    parameters = profile.params[:, member]
    outputs = _outputs(family, parameters)
    values = outputs[index]
    sse = profile.sse[:, member]
    complete = bool(
        member not in profile.failures and np.isfinite(profile.status[:, member]).all()
        and np.isfinite(outputs).all() and np.isfinite(sse).all() and (sse >= 0).all()
    )
    spread = np.full_like(values, np.nan)
    if complete:
        objective = len(profile.observed) * np.log(np.maximum(sse, 1e-30))
        keep = objective <= objective.min() + 1
        spread = np.max(abs(outputs[keep] - values), axis=0)
    return np.asarray(spread)


def identify_fixed_point(
    family: ProfileFamily, context: DelayedPchip, params: np.ndarray, sse: float,
    *, spread: np.ndarray | None = None,
) -> IdentificationResult:
    """Evaluate data identification at one fixed, possibly signed, input delay.

    Parameters
    ----------
    family : str
        Selected kinetic family; no model selection is performed.
    context : DelayedPchip
        Full input history and original observations at the fixed offset.
    params : ndarray
        Physical profile coordinates: Ktrans/ve for Tofts, Ktrans/ve/vp for
        ET or Patlak (Patlak's ve is absent), Fp/PS/ve/vp for 2CXM.
    sse : float
        Unweighted residual sum of squares on these observations.
    spread : ndarray, optional
        Own-family excursions in reported-output order. Omit for a purely
        conditional diagnostic at a prescribed offset; its excursion is zero.
        Unavailable profile excursions must be supplied as NaN, not omitted.

    Returns
    -------
    IdentificationResult
        The same parameter mapping and numerical screen used for selected fits.
        A diagnostic's result does not alter central identification or retention.
    """
    counts = {"patlak": 3, "tofts": 2, "extended_tofts": 3, "2cxm": 4}
    p = np.asarray(params, dtype=float).copy()
    if family not in counts or p.shape != (counts[family],):
        raise ValueError("Physical profile coordinates for a supported family required.")
    values = _outputs(family, p)
    if spread is None:
        spread = np.zeros_like(values)
    names = PRIMARY_PARAMETERS[family]
    upper = np.array([KTRANS_BOUNDS.max, VE_BOUNDS.max, VP_BOUNDS.max])
    volumes: tuple[int, ...] = () if family in ("patlak", "tofts") else (1, 2)
    joint_outputs = volumes
    if family == "2cxm":
        names = ("Ktrans", "ve", "vp", "Fp", "PS")
        output_upper = np.r_[upper, FP_BOUNDS.max, PS_BOUNDS.max]
        upper = np.array([FP_BOUNDS.max, PS_BOUNDS.max, VE_BOUNDS.max, VP_BOUNDS.max])
        volumes, joint_outputs = (2, 3), (1, 2)
        gradient = np.eye(4)[[0, 2, 3, 0, 1]]
        total = p[0] + p[1]
        gradient[0] = [p[1]**2 / total**2 / 100, p[0]**2 / total**2 / 100, 0, 0] if total > 0 else np.full(4, np.nan)
    else:
        if family == "patlak":
            p, upper = p[[0, 2]], upper[[0, 2]]
        elif family == "tofts":
            upper = upper[:2]
        output_upper = upper
        gradient = np.eye(len(p))
    return evaluate_identification(
        p, upper, volumes, lambda point: evaluate_fixed_kinetics(family, context, point),
        sse, values, gradient, output_upper, spread, joint_outputs,
        names=names, primary=PRIMARY_PARAMETERS[family],
    )
