"""Explicit BSW-guided signal CPI through the shared DSC VI engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
from scipy.interpolate import PPoly

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter
from .residue import signal_evaluator
from .policy import CountDecision, FitAssessment, assess_fit, automatic_count, model_for, select_valid_restarts
from .probability import GammaPrecision, GaussianPrior, SharedFit, local_free_energy
from .geometry import Model
from .signal import MovingLayoutSignalCPI
from .weights import checked_weights


@dataclass(frozen=True)
class LeakageReference:
    """Gaussian uncertainty in the cumulative region-mean reference.

    Parameters
    ----------
    time : ndarray
        Acquisition seconds, starting at zero.
    mean : ndarray
        Cumulative integral of reference delta-R2-star.
    basis : ndarray
        Covariance square root, shaped (frame, rank).
    mode : str
        Explicitly conditional or empirical iid-region-mean uncertainty.
    n_voxels : int
        Reference sample size; zero for a supplied conditional curve.
    """

    time: np.ndarray
    mean: np.ndarray
    basis: np.ndarray
    mode: Literal["conditional_reference", "empirical_iid_region_mean"]
    n_voxels: int

    def __post_init__(self) -> None:
        """Validate geometry and retain immutable numeric copies."""
        time, mean, basis = (np.asarray(x, dtype=float) for x in
                             (self.time, self.mean, self.basis))
        if (time.ndim != 1 or time.size < 2 or mean.shape != time.shape
                or basis.ndim != 2 or basis.shape[0] != time.size
                or not all(np.all(np.isfinite(x)) for x in (time, mean, basis))
                or time[0] != 0 or np.any(np.diff(time) <= 0)
                or mean[0] != 0 or np.any(basis[0] != 0)
                or np.max(np.abs(mean)) <= np.finfo(float).eps):
            raise ValueError("finite, nonzero reference with zero integral origin required")
        if self.mode not in ("conditional_reference", "empirical_iid_region_mean"):
            raise ValueError("unknown reference uncertainty mode")
        if (isinstance(self.n_voxels, bool) or not isinstance(self.n_voxels, int)
                or (self.mode == "empirical_iid_region_mean" and self.n_voxels < 2)
                or (self.mode == "conditional_reference"
                    and (self.n_voxels != 0 or basis.shape[1] != 0))):
            raise ValueError("reference uncertainty and sample count disagree")
        for name, value in zip(("time", "mean", "basis"), (time, mean, basis), strict=True):
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def scale(self) -> float:
        """Return the positive reference-integral normalization."""
        return float(np.max(np.abs(self.mean)))

    @property
    def rank(self) -> int:
        """Return the number of independently informed nuisance directions."""
        return int(self.basis.shape[1])


def leakage_count(
    signals: np.ndarray, aif: np.ndarray, time: np.ndarray, te: float,
    *, family: Model = "cpi", baseline_frames: int | None = None,
) -> CountDecision:
    """Apply the frozen count rule with one extra primary leakage coordinate."""
    decision = automatic_count(signals, aif, time, te, family=family,
                               baseline_frames=baseline_frames)
    extra = 4 if family == "cpi_vtf" else 2
    allowed = tuple(n for n in decision.allowed if 2 * n + extra < len(time))
    if not allowed:
        raise ValueError("no supported leakage count for this acquisition")
    selected = min(allowed, key=lambda n: (abs(n - decision.requested), n))
    reasons = decision.restrictions
    if decision.requested in decision.allowed and decision.requested not in allowed:
        reasons += ("leakage_parameter_observation_limit",)
    return replace(decision, selected=selected, allowed=allowed, restrictions=reasons)


class LeakageSignalModel:
    """Compose intrinsic signal CPI with signed leakage and reference nuisances.

    Parameters
    ----------
    vascular : MovingLayoutSignalCPI
        Checked signal-domain vascular forward model.
    reference : LeakageReference
        Reference integral and its measured Gaussian uncertainty.
    rho_sd : float, optional
        Prototype zero-centered leakage prior SD in log-signal effect units.

    Notes
    -----
    Reference nuisance coordinates have independent unit Gaussian priors.
    Mean validation bounds the leakage exponent to protect signal evaluation.
    """

    def __init__(self, vascular: MovingLayoutSignalCPI, reference: LeakageReference,
                 *, rho_sd: float = 1.0) -> None:
        if not np.array_equal(vascular.time, reference.time):
            raise ValueError("vascular and reference acquisition grids must agree")
        if not np.isfinite(rho_sd) or rho_sd <= 0:
            raise ValueError("rho_sd must be finite and positive")
        self.vascular, self.reference = vascular, reference
        self.observed, self.duration = vascular.observed, vascular.duration
        self.delay_index = vascular.delay_index
        self.rho_index = vascular.prior.mean.size
        variance = np.r_[np.diag(vascular.prior.covariance), rho_sd ** 2,
                         np.ones(reference.rank)]
        self.prior = GaussianPrior(np.r_[vascular.prior.mean, np.zeros(1 + reference.rank)],
                                   np.diag(variance))
        self.lower = np.r_[vascular.lower, -20.0, np.full(reference.rank, -np.inf)]
        self.upper = np.r_[vascular.upper, 20.0, np.full(reference.rank, np.inf)]
        self.q = reference.mean / reference.scale
        self.basis = reference.basis / reference.scale

    def valid_mean(self, mean: np.ndarray) -> bool:
        """Require feasible vascular coordinates and finite bounded log effect."""
        if (mean.shape != self.prior.mean.shape or not np.all(np.isfinite(mean))
                or np.any(mean < self.lower) or np.any(mean > self.upper)
                or not self.vascular.valid_mean(mean[:self.rho_index])):
            return False
        with np.errstate(over="ignore", invalid="ignore"):
            effect = mean[self.rho_index] * (self.q + self.basis @ mean[self.rho_index + 1:])
        return bool(np.all(np.isfinite(effect)) and np.max(np.abs(effect)) <= 20)

    def starts(self) -> tuple[np.ndarray, ...]:
        """Preserve every data-derived start, appending no-effect/reference means."""
        return tuple(np.r_[x, np.zeros(1 + self.reference.rank)]
                     for x in self.vascular.starts())

    def knot_times(self, mean: np.ndarray) -> np.ndarray:
        """Return the intrinsic residue geometry, excluding leakage coordinates."""
        return self.vascular.knot_times(mean[:self.rho_index])

    def knot_values(self, mean: np.ndarray) -> np.ndarray:
        """Return the intrinsic residue values, excluding leakage coordinates."""
        return self.vascular.knot_values(mean[:self.rho_index])

    def residue(self, mean: np.ndarray) -> PPoly:
        """Return the intrinsic polynomial, excluding leakage coordinates."""
        return self.vascular.residue(mean[:self.rho_index])

    def arrival_delay(self, mean: np.ndarray) -> float:
        """Read arrival from the vascular part without confusing a fixed coordinate."""
        return self.vascular.arrival_delay(mean[:self.rho_index])

    def predict(self, mean: np.ndarray) -> np.ndarray:
        """Evaluate scalar signal without invoking the batched implementation."""
        q = self.q + self.basis @ mean[self.rho_index + 1:]
        return cast(np.ndarray, self.vascular.predict(mean[:self.rho_index]) * np.exp(mean[self.rho_index] * q))


class LeakageBatchEvaluator:
    """Batched product-rule extension with no duplicated convolution or VI loop."""

    def __init__(self, model: LeakageSignalModel) -> None:
        self.model = model
        self.vascular = signal_evaluator(model.vascular)

    def clear_transient_state(self) -> None:
        """Discard vascular evaluator history between independent fits."""
        self.vascular.clear_transient_state()

    def components(self, means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return uncertain normalized reference and signed multiplicative effect."""
        model = self.model
        if means.ndim != 2 or means.shape[1:] != model.prior.mean.shape:
            raise ValueError("parameter matrix dimensions do not match the model")
        if not np.all(np.isfinite(means)):
            raise ValueError("finite parameter matrix required")
        q = model.q[None, :] + means[:, model.rho_index + 1:] @ model.basis.T
        return q, np.exp(means[:, model.rho_index, None] * q)

    def predictions(self, means: np.ndarray) -> np.ndarray:
        """Predict the measured signal at all voxel/start means."""
        _, factor = self.components(means)
        return cast(np.ndarray, self.vascular.predictions(means[:, :self.model.rho_index]) * factor)

    def jacobians(self, means: np.ndarray) -> np.ndarray:
        """Differentiate vascular, leakage and all reference coordinates jointly."""
        q, factor = self.components(means)
        index = self.model.rho_index
        prediction = self.vascular.predictions(means[:, :index]) * factor
        vascular = self.vascular.jacobians(means[:, :index]) * factor[:, :, None]
        rho = (prediction * q)[:, :, None]
        nuisance = (prediction * means[:, index, None])[:, :, None] * self.model.basis[None]
        return np.concatenate((vascular, rho, nuisance), axis=2)


@dataclass
class LeakageBatch:
    """Complete checked marginal fits and full posterior, including failures."""

    fits: list[SharedFit]
    assessments: list[FitAssessment]
    rows: list[dict[str, Any]]
    leakage: list[dict[str, float | int]]
    arrays: dict[str, Any]
    seconds: float
    iterations: int
    leakage_method: Literal["bsw", "none"] = "bsw"
    infer_delay: bool = True


@dataclass(frozen=True)
class BSWPrior:
    """Signed K2 in inverse seconds and its positive conditional OLS standard error."""

    k2: float
    k2_sd: float


class BSWCenteredModel(LeakageSignalModel):
    """Use a declared empirical coefficient prior, with K2 inferred jointly with perfusion."""

    def __init__(self, vascular: MovingLayoutSignalCPI, reference: LeakageReference,
                 prior: BSWPrior) -> None:
        scale = vascular.te * reference.scale
        if not np.isfinite(prior.k2) or not np.isfinite(prior.k2_sd) or prior.k2_sd <= 0:
            raise ValueError("finite prior mean and positive SD required")
        super().__init__(vascular, reference, rho_sd=prior.k2_sd * scale)
        mean = self.prior.mean.copy()
        mean[self.rho_index] = prior.k2 * scale
        self.prior = GaussianPrior(mean, self.prior.covariance)


def readout_state(
    model: BSWCenteredModel | MovingLayoutSignalCPI, observed: np.ndarray, mean: np.ndarray,
    covariance: np.ndarray, shape: float, scale: float, prediction: np.ndarray,
    jacobian: np.ndarray, order: int, restart: int, converged: bool,
    *, weights: np.ndarray | None = None, prior: GaussianPrior | None = None,
) -> tuple[SharedFit, FitAssessment, dict[str, float | int]]:
    """Read and assess one VI state using its actual independent/conditional prior.

    Parameters
    ----------
    model : BSWCenteredModel or MovingLayoutSignalCPI
        Signal forward model and physical coordinate definitions. A vascular
        model alone has no inferred leakage or reference coordinates.
    observed, mean, covariance, prediction, jacobian : ndarray
        One voxel's measured signal and complete returned local state.
    shape, scale : float
        Gamma noise precision in shape/scale convention.
    order, restart : int
        Acquisition count and initialization identifier.
    converged : bool
        Solver convergence flag, passed through unchanged.
    weights : ndarray, optional
        Frame likelihood weights used by the solver.
    prior : GaussianPrior, optional
        Actual final conditional prior for spatial fitting. Defaults to the
        model's independent prior. The score uses this voxel's conditional prior.

    Returns
    -------
    fit, assessment, leakage : SharedFit, FitAssessment, dict
        Research-unit readout, numerical assessment and separate fitted K2.
        The leakage record is empty when leakage estimation is disabled.
    """
    actual_prior = model.prior if prior is None else prior
    noise = GammaPrecision(float(shape), 1 / float(scale))
    score = local_free_energy(observed, prediction, jacobian, mean, covariance,
                              actual_prior, noise, GammaPrecision(1e-6, 1e-6), weights=weights)
    mtt = float(model.residue(mean).integrate(0, model.duration))
    fit = SharedFit(0, order, restart, converged, score, float(mean[0] * 6000),
                    float(mean[0] * mtt * 100), mtt, model.arrival_delay(mean),
                    float(mean[1]), float(np.sqrt(1 / noise.mean)),
                    float(np.sum((observed - prediction) ** 2)), mean.tolist())
    assessed = assess_fit(fit, model, observed, prediction, jacobian, covariance,
                           float(shape), float(scale), actual_prior.covariance, weights=weights)
    vascular = model.vascular if isinstance(model, BSWCenteredModel) else model
    vascular_mean = mean[:model.rho_index] if isinstance(model, BSWCenteredModel) else mean
    from .conditioned import FixedDelaySignalCPI
    fixed = isinstance(vascular, FixedDelaySignalCPI)
    if isinstance(vascular, FixedDelaySignalCPI):
        vascular_mean = vascular.expand(vascular_mean)
        vascular = vascular.full
    warnings = list(assessed.warnings)
    if not fixed and (np.isclose(vascular_mean[vascular.delay_index], vascular.lower[vascular.delay_index],
                                 rtol=1e-5, atol=1e-8)
                      or np.isclose(vascular_mean[vascular.delay_index], vascular.upper[vascular.delay_index],
                                    rtol=1e-5, atol=1e-8)):
        warnings.append("arrival_at_bound")
    spacing = vascular.spacing_indices
    if (np.any(np.isclose(vascular_mean[spacing], vascular.lower[spacing], rtol=1e-5, atol=1e-8))
            or np.any(np.isclose(vascular_mean[spacing], vascular.upper[spacing], rtol=1e-5, atol=1e-8))
            or np.isclose(np.diff(vascular.knot_times(vascular_mean))[-1], vascular.dt/3,
                          rtol=1e-5, atol=1e-8)):
        warnings.append("spacing_at_bound")
    assessed = replace(assessed, warnings=tuple(warnings))
    if not isinstance(model, BSWCenteredModel):
        return fit, assessed, {}
    rho = float(mean[model.rho_index])
    rho_sd = float(np.sqrt(covariance[model.rho_index, model.rho_index]))
    normalization = model.vascular.te * model.reference.scale
    leakage: dict[str, float | int] = dict(voxel=0, restart=restart, rho=rho,
        k2=rho / normalization, rho_sd=rho_sd, k2_sd=rho_sd / normalization)
    return fit, assessed, leakage


def fit_with_prior(
    signal: np.ndarray, aif: np.ndarray, time: np.ndarray, te: float, order: int,
    reference: LeakageReference, prior: BSWPrior,
    *, weights: np.ndarray | None = None, family: Model = "cpi", infer_delay: bool = True,
    sloppy: bool = False,
) -> LeakageBatch:
    """Fit one voxel's explicit BSW prior, batching its data-derived starts.

    Parameters
    ----------
    signal, aif, time : ndarray
        Original measured signal, delta-R2-star AIF and acquisition seconds.
    te : float
        Echo time in seconds.
    order : int
        Acquisition-level control-point count.
    reference : LeakageReference
        Exact BSW reference integral and empirical covariance factor.
    prior : BSWPrior
        This voxel's signed coefficient and conditional standard error.
    weights : ndarray, optional
        Shared frame likelihood weights; BSW guidance and starts are unchanged.
    family : {"cpi", "cpi_vtf"}, optional
        Plain intrinsic residue or intrinsic residue with gamma arterial
        transport. Both include the same separately guided leakage term.
    infer_delay : bool, optional
        Infer arrival or condition it at zero, removing the delay coordinate.
    sloppy : bool, optional
        Use the shared eight-iteration testing schedule.

    Returns
    -------
    LeakageBatch
        All restart states, assessments and selected state. The shared engine
        receives exactly one prior; independent voxels require separate calls.

    Raises
    ------
    ValueError
        If the signal and K2 prior do not describe exactly one voxel.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("one signal and its own BSW prior required")
    weights = checked_weights(weights, signal.size)
    model = BSWCenteredModel(model_for(signal, aif, time, te, order, family=family,
                                      infer_delay=infer_delay),
                             reference, prior)
    return _fit_model(signal, aif, time, order, model, weights=weights,
                      infer_delay=infer_delay, sloppy=sloppy)


def fit_without_leakage(
    signal: np.ndarray, aif: np.ndarray, time: np.ndarray, te: float, order: int,
    *, weights: np.ndarray | None = None, family: Model = "cpi", infer_delay: bool = True,
    sloppy: bool = False,
) -> LeakageBatch:
    """Fit measured signal without a leakage parameter or reference tissue.

    Parameters
    ----------
    signal, aif, time : ndarray
        Original measured signal, delta-R2-star AIF and acquisition seconds.
    te : float
        Echo time in seconds.
    order : int
        Acquisition-level control-point count.
    weights : ndarray, optional
        Frame likelihood weights for the shared variational engine.
    family : {"cpi", "cpi_vtf"}, optional
        Intrinsic CPI with optional gamma arterial transport.
    infer_delay : bool, optional
        Infer arrival or condition it at zero, removing the delay coordinate.
    sloppy : bool, optional
        Use the shared eight-iteration testing schedule.

    Returns
    -------
    LeakageBatch
        Full vascular posterior and restart diagnostics. The leakage record
        is empty: no zero-valued leakage measurement or prior is substituted.

    Raises
    ------
    ValueError
        If the input does not describe exactly one measured signal.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("one measured signal required")
    weights = checked_weights(weights, signal.size)
    model = model_for(signal, aif, time, te, order, family=family, infer_delay=infer_delay)
    return _fit_model(signal, aif, time, order, model, weights=weights,
                      infer_delay=infer_delay, sloppy=sloppy)


def _fit_model(
    signal: np.ndarray, aif: np.ndarray, time: np.ndarray, order: int,
    model: BSWCenteredModel | MovingLayoutSignalCPI,
    *, weights: np.ndarray | None, infer_delay: bool = True, sloppy: bool = False,
) -> LeakageBatch:
    """Batch one model's starts through the shared DSC variational engine."""
    signals = signal[None, :]
    models = [model]
    iterations = 100
    model = models[0]
    initial, owners, restarts = [], [], []
    for voxel, current in enumerate(models):
        for restart, start in enumerate(current.starts()):
            initial.append(start)
            owners.append(voxel)
            restarts.append(restart)
    means = np.asarray(initial)
    observations = signals[owners]
    has_leakage = isinstance(model, BSWCenteredModel)
    evaluator = (LeakageBatchEvaluator(model) if isinstance(model, BSWCenteredModel)
                 else signal_evaluator(model))
    adapter = DSCVIModelAdapter(evaluator.predictions, evaluator.jacobians,
                               lambda x: np.asarray([model.valid_mean(m) for m in x]))
    stds = np.sqrt(np.diag(model.prior.covariance))
    n_parameters = means.shape[1]
    started = perf_counter()
    result = batched_dsc_vi_fit(
        model_type="signal_cpi_leakage" if has_leakage else "signal_cpi",
        tissue_curves=observations, aif=aif,
        time_points=time, prior_means_nat=model.prior.mean, prior_stds_nat=stds,
        transforms=[TransformType.IDENTITY] * n_parameters,
        upper_bounds=[None] * n_parameters,
        fitting_param_names=["cbf", "s0"] + [f"coordinate_{i}" for i in range(2, n_parameters)],
        fitting_bounds=list(zip(model.lower, model.upper, strict=True)),
        prior_means_transformed=model.prior.mean, prior_stds_transformed=stds,
        initial_means_transformed=means, n_iterations=iterations,
        model_adapter=adapter, return_full_posterior=True, save_fitted_curves=True,
        weights=weights, sloppy=sloppy,
    )
    posterior, prediction = result.full_posterior, result.fitted_curves
    assert posterior is not None and prediction is not None
    failure_reasons = result.failure_reasons or (None,) * len(posterior.mean)
    successful = np.array([reason is None for reason in failure_reasons], dtype=bool)
    jacobian = np.full(
        (len(posterior.mean), signal.size, posterior.mean.shape[1]), np.nan,
        dtype=np.float64,
    )
    if np.any(successful):
        jacobian[successful] = evaluator.jacobians(posterior.mean[successful])
    fits, assessments, leakage = [], [], []
    for i, mean in enumerate(posterior.mean):
        if not successful[i]:
            continue
        fit, assessed, inferred_leakage = readout_state(
            model, observations[i], mean, posterior.covariance[i],
            float(posterior.noise_shape[i]), float(posterior.noise_scale[i]),
            prediction[i], jacobian[i], order, restarts[i], bool(result.converged[i]), weights=weights,
        )
        fits.append(fit)
        assessments.append(assessed)
        if has_leakage:
            leakage.append(inferred_leakage)
    arrays = dict(mean=posterior.mean, covariance=posterior.covariance,
                  noise_shape=posterior.noise_shape, noise_scale=posterior.noise_scale,
                  fitted_curves=prediction, jacobian=jacobian, starts=means,
                  prior_mean=model.prior.mean, prior_covariance=model.prior.covariance,
                  converged=result.converged,
                  owners=np.asarray(owners, dtype=np.int64),
                  restarts=np.asarray(restarts, dtype=np.int64),
                  failure_reasons=np.asarray(failure_reasons, dtype=object))
    if isinstance(model, BSWCenteredModel):
        arrays.update(reference_mean=model.reference.mean, reference_basis=model.reference.basis)
    if weights is not None:
        arrays["weights"] = weights
    rows = select_valid_restarts(fits, assessments, len(signals))
    for voxel, row in enumerate(rows):
        row["n_failed_starts"] = sum(
            owner == voxel and reason is not None
            for owner, reason in zip(owners, failure_reasons, strict=True)
        )
    return LeakageBatch(fits, assessments, rows, leakage, arrays,
                        perf_counter() - started, result.n_iterations,
                        "bsw" if has_leakage else "none", infer_delay)
