"""Conditional VI refinement of a selected family and effective input delay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

from pmriprep.models.dce.base import FitResult, ModelParameter, ParamsType, PKModel
from pmriprep.models.dce.bayesian import BayesianPKFitter, BayesianPrior
from pmriprep.models.dce.delay_profiles import DelayProfile, ProfileFamily
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.fit_errors import NumericalFitError
from pmriprep.models.dce.local_selection import FamilyTiming
from pmriprep.models.dce.native_exchange import predict
from pmriprep.models.dce.patlak import PatlakModel
from pmriprep.models.dce.profile_fitting import tofts_predict
from pmriprep.models.dce.retention import is_degenerate_fit
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel

_MODELS: dict[ProfileFamily, Callable[[], PKModel]] = {
    "patlak": PatlakModel,
    "tofts": ToftsModel,
    "extended_tofts": ExtendedToftsModel,
    "2cxm": TwoCompartmentExchangeModel,
}


def _evaluate(family: str, context: DelayedPchip, params: np.ndarray) -> np.ndarray:
    """Evaluate only kinetics, retaining the full fixed delayed input history."""
    if family == "2cxm":
        return predict(context, params)
    if family == "patlak":
        return np.asarray(params[0] * context.convolution(0.0) + params[1] * context.cp)
    curve = tofts_predict(context, params[:2])
    return curve if family == "tofts" else np.asarray(curve + params[2] * context.cp)


def selected_fit_result(
    profile: DelayProfile, timing: FamilyTiming, member: int,
    aif: np.ndarray, times: np.ndarray, tissue: np.ndarray,
) -> FitResult:
    """Translate selected conditional kinetics without refitting or reselection.

    Parameters
    ----------
    profile, timing : DelayProfile and FamilyTiming
        Matching finalized family profile and centre selection.
    member : int
        Centre index in profile arrays.
    aif, times : ndarray
        Full input in mM and original acquisition times in minutes.
    tissue : ndarray
        Same member's retained observations in mM, in profile.observed order.

    Returns
    -------
    FitResult
        Reconstructed NLLS estimate and curve with the selected BIC. Numerical
        success does not imply retention or identification; callers must apply
        retention before using this result as an inference seed.
    """
    index = timing.index
    if (
        timing.reason is not None or index < 0 or index >= len(profile.delays_seconds)
        or member < 0 or member >= profile.sse.shape[1]
        or profile.status[index, member] <= 0 or member in profile.failures
    ):
        raise ValueError("A numerically available selected family fit is required.")
    base = _MODELS[profile.family]()
    params = profile.params[index, member].copy()
    if profile.family == "patlak":
        params = params[[0, 2]]
    context = DelayedPchip(aif, times, float(profile.delays_seconds[index]), profile.observed)
    curve = _evaluate(profile.family, context, params)
    tissue = np.asarray(tissue)
    if tissue.shape != curve.shape or not np.isfinite(tissue).all():
        raise ValueError("Selected tissue must match the retained observation support.")
    sse = float(np.sum((curve - tissue) ** 2))
    if not np.isfinite(curve).all() or not np.isclose(
        sse, profile.sse[index, member], rtol=0.0, atol=1e-12,
    ):
        raise ValueError("Selected profile does not match the fixed operator/data.")
    values = base.params_to_dict(params)
    values.update(base.compute_derived_parameters(params))
    return FitResult(
        parameters=values, fitted_curve=curve, residual_sum_squares=sse,
        r_squared=base.compute_r_squared(tissue, curve),
        aic=base.compute_aic(sse, len(tissue)), bic=timing.bic,
        model_name=profile.family, n_parameters=base.n_parameters,
        n_observations=len(tissue), converged=True, fitting_method="nlls",
    )


class FixedDelayModel(PKModel):
    """Bind kinetic inference to one input, observation set and eligible NLLS fit.

    Parameters
    ----------
    family : str
        Already selected or explicitly requested kinetic family.
    aif, times : ndarray
        Full plasma input in mM and original acquisition times in minutes.
    tissue : ndarray
        Retained tissue concentrations, in observed-index order.
    delay : float
        Fixed effective offset in seconds; not an inference parameter.
    seed : FitResult
        Eligible converged NLLS fit from the same family, offset and observations.
    observed : ndarray, optional
        Retained original indices; intervening input history is preserved.

    Notes
    -----
    Model parameter definitions, priors and derived quantities come from the
    existing kinetic classes. Evaluation uses the shared delayed PCHIP input.
    Cached fit calls return independent copies and cannot start an undelayed fit.
    """

    def __init__(
        self, family: ProfileFamily, aif: np.ndarray, times: np.ndarray,
        tissue: np.ndarray, delay: float, seed: FitResult,
        observed: np.ndarray | None = None,
    ) -> None:
        self.base = _MODELS[family]()
        self._delay = float(delay)
        self.context = DelayedPchip(aif, times, delay, observed)
        indices = self.context.observed
        if len(indices) < 2 or len(np.unique(indices)) != len(indices):
            raise ValueError("Unique original observation indices required.")
        self.input_aif = np.array(np.asarray(aif)[indices], copy=True)
        self.input_times = np.array(np.asarray(times)[indices], copy=True)
        self.tissue = np.array(tissue, dtype=float, copy=True)
        self._seed = deepcopy(seed)
        if not self.valid_result(seed) or seed.fitting_method != "nlls":
            raise ValueError("Eligible same-model NLLS seed required.")
        self.check_result_curve(seed)
        for array in (self.input_aif, self.input_times, self.tissue):
            array.setflags(write=False)

    @property
    def delay(self) -> float:
        """Selected offset, excluded from inferred parameters."""
        return self._delay

    @property
    def name(self) -> str:
        """Selected family name."""
        return self.base.name

    @property
    def n_parameters(self) -> int:
        """Kinetic dimension only; inference is conditional on timing."""
        return self.base.n_parameters

    def get_parameter_definitions(self) -> Sequence[ModelParameter]:
        """Preserve kinetic bounds, units, transformations and prior definitions."""
        return self.base.get_parameter_definitions()

    def compute_derived_parameters(self, params: ParamsType) -> dict[str, float]:
        """Preserve the selected family's derived-parameter definitions."""
        return self.base.compute_derived_parameters(params)

    def _check_inputs(self, aif: np.ndarray, times: np.ndarray) -> None:
        """Reject changed input samples or observation times."""
        if not np.array_equal(aif, self.input_aif) or not np.array_equal(times, self.input_times):
            raise ValueError("Fixed input/observation contract changed.")

    def evaluate(self, params: ParamsType, aif: np.ndarray, time_points: np.ndarray) -> np.ndarray:
        """Evaluate kinetics with unchanged offset and complete input history."""
        self._check_inputs(aif, time_points)
        return _evaluate(self.name, self.context, np.asarray(params))

    def fit(
        self, tissue_curve: np.ndarray, aif: np.ndarray, time_points: np.ndarray,
        initial_guess: Sequence[float] | None = None, weights: np.ndarray | None = None,
    ) -> FitResult:
        """Return a copy of the selected NLLS fit for identical data/unit weights.

        The optional initial_guess is not used: this is cached reuse, not a new
        optimization. Changed observations or weighting cannot reuse the seed.
        """
        self._check_inputs(aif, time_points)
        if not np.array_equal(tissue_curve, self.tissue):
            raise ValueError("Cannot reuse NLLS for changed observations.")
        if weights is not None and not np.array_equal(weights, np.ones_like(self.tissue)):
            raise ValueError("Only unit temporal weights are supported.")
        return deepcopy(self._seed)

    def valid_result(self, result: FitResult) -> bool:
        """Check numerical/parameter retention, independently of identification QC."""
        params = np.array([result.parameters.get(n, np.nan) for n in self.get_parameter_names()])
        return bool(
            result.model_name == self.name and result.converged
            and result.n_observations == len(self.tissue)
            and result.n_parameters == self.n_parameters
            and np.isfinite(params).all() and self.validate_params(params)
            and not is_degenerate_fit(self.name, result.parameters)
            and np.isfinite(self.tissue).all()
            and result.fitted_curve is not None
            and np.shape(result.fitted_curve) == self.tissue.shape
            and np.isfinite(result.fitted_curve).all()
            and np.isfinite(result.residual_sum_squares) and result.residual_sum_squares >= 0
        )

    def check_result_curve(self, result: FitResult) -> None:
        """Reject results from a different forward operator, offset or tissue curve."""
        expected = self.evaluate(self.dict_to_params(result.parameters), self.input_aif, self.input_times)
        if result.fitted_curve is None or not np.allclose(result.fitted_curve, expected, rtol=1e-10, atol=1e-12) or not np.isclose(
            float(np.sum((expected - self.tissue) ** 2)), result.residual_sum_squares,
            rtol=0.0, atol=1e-12,
        ):
            raise ValueError("Fit result violates the fixed operator/data contract.")


@dataclass(frozen=True)
class Refinement:
    """Retained kinetics with fixed-selection and fallback provenance."""

    fit: FitResult
    requested_engine: Literal["vi"]
    actual_engine: Literal["vi", "nlls"]
    family: str
    delay: float
    fallback: bool
    reason: str | None


def refine_vi(
    model: FixedDelayModel, *, vp_prior: BayesianPrior | None = None, sloppy: bool = False,
) -> Refinement:
    """Refine kinetics at fixed timing, or retain the eligible same-delay NLLS fit.

    Parameters
    ----------
    model : FixedDelayModel
        Checked operator/data and eligible selected NLLS seed.
    vp_prior : BayesianPrior, optional
        Already qualified DSC prior; it cannot change family or timing.
    sloppy : bool
        Use the existing test-mode VI iteration budget.

    Returns
    -------
    Refinement
        Retained fit and requested/actual inference provenance. Uncertainty is
        conditional on the selected family/input/delay; pooled timing uncertainty
        is not included. Unexpected programming/backend errors propagate.
    """
    seed = model.fit(model.tissue, model.input_aif, model.input_times)
    fitter = BayesianPKFitter(model, vp_prior_override=vp_prior)
    reason: str | None = None
    try:
        result = fitter.fit_vi(
            model.tissue, model.input_aif, model.input_times,
            initial_parameters=seed.parameters, sloppy=sloppy,
        )
    except (NumericalFitError, FloatingPointError, np.linalg.LinAlgError) as error:
        reason = type(error).__name__
    else:
        if result.model_name != model.name or result.fitting_method != "vi":
            raise ValueError("Refinement changed the selected family or inference engine.")
        if model.valid_result(result):
            model.check_result_curve(result)
            return Refinement(deepcopy(result), "vi", "vi", model.name, model.delay, False, None)
        reason = "nonconverged" if not result.converged else "invalid_result"
    return Refinement(seed, "vi", "nlls", model.name, model.delay, True, reason)
