"""Conditional uncertainty and optional VI for already selected voxel fits."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from threadpoolctl import threadpool_limits

from pmriprep.models.dce.base import FitResult
from pmriprep.models.dce.bayesian import create_vp_prior_from_dsc
from pmriprep.models.dce.conditional_covariance import covariance, jacobian
from pmriprep.models.dce.delay_profiles import ProfileFamily
from pmriprep.models.dce.fixed_inference import FixedDelayModel, Refinement, refine_vi


@dataclass
class ConditionalInput:
    """Eligible selected NLLS fit and the unchanged observations used to obtain it."""

    family: ProfileFamily
    aif: np.ndarray
    times: np.ndarray
    observed: np.ndarray
    tissue: np.ndarray
    delay: float
    seed: FitResult
    inference: Literal["nlls", "vi"]
    vp_prior: float | None = None
    prior_relative_uncertainty: float = 0.0
    sloppy: bool = False


def finish_conditional(block: list[ConditionalInput]) -> list[Refinement]:
    """Compute conditional NLLS SEs, then optionally refine at unchanged timing.

    Parameters
    ----------
    block : list of ConditionalInput
        Bounded group of eligible selected fits. Prior qualification is owned
        by the caller; this function cannot change model or observation support.

    Returns
    -------
    list of Refinement
        Conditional estimates in input order, with inference/fallback provenance.
        NLLS uncertainty uses SSE/(n-k) and the full kinetic Jacobian. It does
        not include uncertainty in the selected delay or arterial input.
    """
    results: list[Refinement] = []
    with threadpool_limits(limits=1):
        for item in block:
            model = FixedDelayModel(
                item.family, item.aif, item.times, item.tissue, item.delay,
                item.seed, item.observed,
            )
            params = np.asarray(model.dict_to_params(item.seed.parameters))
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
                results.append(Refinement(seed, "nlls", "nlls", item.family, item.delay, False, None))
            elif item.inference == "vi":
                prior = (
                    create_vp_prior_from_dsc(item.vp_prior, relative_uncertainty=item.prior_relative_uncertainty)
                    if item.vp_prior is not None and "vp" in names else None
                )
                results.append(refine_vi(model, vp_prior=prior, sloppy=item.sloppy))
            else:
                raise ValueError("Only NLLS and VI pipeline inference are supported.")
    return results
