"""Conditional uncertainty and optional VI for already selected voxel fits."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from threadpoolctl import threadpool_limits

from pmriprep.models.dce.base import FitResult
from pmriprep.models.dce.bayesian import create_vp_prior_from_dsc
from pmriprep.models.dce.conditional_covariance import covariance, jacobian
from pmriprep.models.dce.delay_profiles import ProfileFamily
from pmriprep.models.dce.fixed_inference import FixedDelayModel, Refinement, refine_vi
from pmriprep.models.dce.identification import IdentificationResult, identify_fixed_point


@dataclass
class ConditionalInput:
    """Eligible selected NLLS fit and the unchanged observations used to obtain it.

    Attributes
    ----------
    profile_spread : ndarray
        Own-family excursion in reported-output order, extracted before slice
        release. NaN denotes unavailable evidence, not zero uncertainty.
    profile_sse : float
        Stored selected-profile SSE for identification. The seed carries its
        independently reconstructed SSE for inference; both agree within the
        fixed-input contract but can differ at floating-point precision.
    """

    family: ProfileFamily
    aif: np.ndarray
    times: np.ndarray
    observed: np.ndarray
    tissue: np.ndarray
    delay: float
    seed: FitResult
    inference: Literal["nlls", "vi"]
    profile_spread: np.ndarray
    profile_sse: float
    vp_prior: float | None = None
    prior_relative_uncertainty: float = 0.0
    sloppy: bool = False


@dataclass
class ConditionalResult:
    """Reported inference and separate selected-NLLS identification evidence.

    Timings are worker elapsed seconds, not parallel wall time. Identification
    never changes the refinement, its parameters, or its retention.
    """

    refinement: Refinement
    identification: IdentificationResult
    identification_seconds: float
    refinement_seconds: float


def finish_conditional(block: list[ConditionalInput]) -> list[ConditionalResult]:
    """Compute conditional NLLS SEs, then optionally refine at unchanged timing.

    Parameters
    ----------
    block : list of ConditionalInput
        Bounded group of eligible selected fits. Prior qualification is owned
        by the caller; this function cannot change model or observation support.

    Returns
    -------
    list of ConditionalResult
        Conditional estimates and NLLS identification in input order.
        NLLS uncertainty uses SSE/(n-k) and the full kinetic Jacobian. It does
        not include uncertainty in the selected delay or arterial input.
    """
    results: list[ConditionalResult] = []
    with threadpool_limits(limits=1):
        for item in block:
            started = perf_counter()
            model = FixedDelayModel(
                item.family, item.aif, item.times, item.tissue, item.delay,
                item.seed, item.observed,
            )
            params = np.asarray(model.dict_to_params(item.seed.parameters))
            identification_started = perf_counter()
            profile_params = np.array([params[0], np.nan, params[1]]) if item.family == "patlak" else params
            identification = identify_fixed_point(
                item.family, model.context, profile_params,
                item.profile_sse, spread=item.profile_spread,
            )
            identification_seconds = perf_counter() - identification_started
            upper = np.asarray([b[1] for b in model.get_bounds()])
            names = model.get_parameter_names()
            volumes = tuple(names.index(n) for n in ("ve", "vp") if n in names)
            # Only the reversible vascular models have a joint-volume bound.
            volumes = volumes if len(volumes) == 2 else ()
            derivative, _ = jacobian(
                params, upper, volumes,
                lambda p: model.evaluate(p, model.input_aif, model.input_times),
            )
            try:
                cov, _, reason = covariance(derivative, item.seed.residual_sum_squares)
            except np.linalg.LinAlgError:
                cov, reason = np.full((len(params), len(params)), np.nan), "covariance_unavailable"
            seed = model.fit(model.tissue, model.input_aif, model.input_times)
            seed.uncertainties = dict(zip(names, np.sqrt(np.maximum(np.diag(cov), 0))))
            seed.uncertainty_source = "conditional_jacobian" if reason == "ok" else reason
            model = FixedDelayModel(
                item.family, item.aif, item.times, item.tissue, item.delay,
                seed, item.observed,
            )
            if item.inference == "nlls":
                refined = Refinement(seed, "nlls", "nlls", item.family, item.delay, False, None)
            elif item.inference == "vi":
                prior = (
                    create_vp_prior_from_dsc(item.vp_prior, relative_uncertainty=item.prior_relative_uncertainty)
                    if item.vp_prior is not None and "vp" in names else None
                )
                refined = refine_vi(model, vp_prior=prior, sloppy=item.sloppy)
            else:
                raise ValueError("Only NLLS and VI pipeline inference are supported.")
            results.append(ConditionalResult(
                refined, identification, identification_seconds,
                perf_counter() - started - identification_seconds,
            ))
    return results
