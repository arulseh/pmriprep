"""Automatic acquisition settings and numerical fit assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from scipy.interpolate import PPoly

from pmriprep.models.dsc.signal_cpi.count import select_n_control_points
from .probability import GaussianPrior, SharedFit
from .precision import SIGNAL_BASELINE_RELATIVE_ERROR
from .geometry import Model, _validate
from .signal import MovingLayoutSignalCPI
from .residue import PhysicalSignalCPI
from .weights import checked_weights

_LOCAL_FACTOR_COVARIANCE_RTOL = np.sqrt(np.finfo(np.float64).eps)
_LOCAL_FACTOR_IDENTITY_ATOL = 1e-4


def candidate_orders(n_frames: int, *, model: Model = "cpi") -> tuple[int, ...]:
    """Return every supported count with more observations than mean parameters."""
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)):
        raise ValueError("frame count must be an integer")
    if n_frames < 20:
        raise ValueError("signal CPI requires at least 20 frames")
    if model not in ("cpi", "cpi_vtf"):
        raise ValueError("model must be cpi or cpi_vtf")
    extra_parameters = 3 if model == "cpi_vtf" else 1
    return tuple(range(5, min(12, (n_frames - 1 - extra_parameters) // 2) + 1))


def factor_diagnostic(
    covariance: np.ndarray,
    jacobian: np.ndarray,
    prior_covariance: np.ndarray,
    residual: np.ndarray,
    shape: float,
    scale: float,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Check fixed-mean Gaussian/Gamma coordinate identities at returned data.

    These are local-linear identities, not a claim of nonlinear posterior
    calibration. No factor is substituted into a fit or selection result.
    """
    precision = shape * scale
    gram = jacobian.T @ jacobian if weights is None else jacobian.T @ (weights[:, None] * jacobian)
    trace = float(np.sum(covariance * gram.T))
    rate = 1.0 / scale
    rss = float(residual @ residual) if weights is None else float(np.sum(weights * residual**2))
    required_rate = 1e-6 + 0.5 * (rss + trace)
    prior_precision = np.linalg.inv(prior_covariance)
    required_precision = prior_precision + precision * gram
    root = np.linalg.cholesky(covariance)
    whitened = root.T @ required_precision @ root
    root_prior_precision = np.linalg.cholesky(prior_precision)
    prior_root = np.linalg.solve(
        root_prior_precision.T, np.eye(root_prior_precision.shape[0]),
    )
    weighted_jacobian = (
        jacobian if weights is None else np.sqrt(weights)[:, None] * jacobian
    )
    _, singular, vectors = np.linalg.svd(weighted_jacobian @ prior_root)
    basis = prior_root @ vectors.T
    expected_scaled = basis / np.sqrt(1 + precision * singular**2)[None, :]
    expected_covariance = expected_scaled @ expected_scaled.T
    covariance_relative_error = float(
        np.linalg.norm(covariance - expected_covariance, ord="fro")
        / np.linalg.norm(expected_covariance, ord="fro")
    )
    return {
        "stored_gamma_rate": rate,
        "fixed_covariance_required_gamma_rate": required_rate,
        "gamma_rate_ratio": required_rate / rate,
        "current_jacobian_covariance_trace": trace,
        "residual_sum_squares": rss,
        "covariance_precision_identity_max_eigen_error": float(
            np.max(np.abs(np.linalg.eigvalsh(whitened) - 1.0))
        ),
        "covariance_relative_error": covariance_relative_error,
    }


class AssessedSignalModel(Protocol):
    """Forward/readout contract needed to check a complete local VI state."""

    observed: np.ndarray
    prior: GaussianPrior
    duration: float
    delay_index: int

    def valid_mean(self, mean: np.ndarray) -> bool:
        """Return whether the inferred coordinates are admissible."""
        ...

    def predict(self, mean: np.ndarray) -> np.ndarray:
        """Evaluate the scalar signal forward model."""
        ...

    def knot_times(self, mean: np.ndarray) -> np.ndarray:
        """Return intrinsic residue knot times."""
        ...

    def knot_values(self, mean: np.ndarray) -> np.ndarray:
        """Return intrinsic residue knot amplitudes."""
        ...

    def residue(self, mean: np.ndarray) -> PPoly:
        """Return the full intrinsic polynomial, including its initial slope."""
        ...

    def arrival_delay(self, mean: np.ndarray) -> float:
        """Return the inferred or conditioned timing used by the model."""
        ...


@dataclass(frozen=True)
class CountDecision:
    """Requested and feasible acquisition-level counts, with restriction reasons."""

    requested: int
    selected: int
    allowed: tuple[int, ...]
    baseline_frames: int
    baseline_source: str
    restrictions: tuple[str, ...]
    selection_method: str = "automatic"


def model_for(
    signal: np.ndarray, aif: np.ndarray, time: np.ndarray, te: float, count: int,
    *, family: Model = "cpi", infer_delay: bool = True,
) -> MovingLayoutSignalCPI:
    """Construct the declared continuous cubic CPI family without fitting."""
    if family not in ("cpi", "cpi_vtf"):
        raise ValueError("family must be cpi or cpi_vtf")
    if not isinstance(infer_delay, bool):
        raise TypeError("infer_delay must be a boolean")
    observed, checked_aif, checked_time, dt = _validate(signal, aif, time, te)
    full = PhysicalSignalCPI(observed, checked_aif, checked_time, dt, te, count, family)
    if infer_delay:
        return full
    from .conditioned import FixedDelaySignalCPI

    return FixedDelaySignalCPI(full)


def automatic_count(
    signals: np.ndarray,
    aif: np.ndarray,
    time: np.ndarray,
    te: float,
    *,
    baseline_frames: int | None = None,
    family: Model = "cpi",
) -> CountDecision:
    """Choose one supported count from measured signals without ground truth.

    Parameters
    ----------
    signals : ndarray
        Positive measured signals, shaped (voxel, frame), in one input pool.
    aif, time : ndarray
        Arterial delta-R2-star and acquisition times in seconds.
    te : float
        Echo time in seconds.
    baseline_frames : int, optional
        Acquisition-supplied prebolus window. Without it, use the AIF-based
        window employed by the existing research initializer.
    family : {"cpi", "cpi_vtf"}, optional
        Declared model, accounting for gamma transport's two extra coordinates.
        This does not select a family using the observations or ground truth.

    Returns
    -------
    CountDecision
        Requested and used counts with explicit baseline and domain provenance.

    Notes
    -----
    Concentration is used only by the scan-level decay heuristic, not as the
    fitting target. A count restriction is an experimental support limit,
    not a demonstration that this is the optimal number of parameters.
    """
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2 or signals.shape[0] == 0:
        raise ValueError("a nonempty voxel-by-frame signal pool is required")
    for signal in signals:
        _validate(signal, aif, time, te)
    if not np.all(np.isfinite(aif)) or np.max(aif) <= 0:
        raise ValueError("finite AIF with a positive bolus required")
    source = "acquisition"
    if baseline_frames is None:
        baseline_frames = max(
            4, min(signals.shape[1] // 5, int(np.argmax(aif > 0.05 * np.max(aif))))
        )
        source = "aif_initializer_rule"
    if (
        isinstance(baseline_frames, (bool, np.bool_))
        or not isinstance(baseline_frames, (int, np.integer))
        or not 0 < baseline_frames < signals.shape[1]
    ):
        raise ValueError("baseline_frames must be an interior positive integer")
    baseline = signals[:, :baseline_frames].mean(axis=1)
    concentration = -np.log(signals / baseline[:, None]) / te
    requested = int(select_n_control_points(
        mean_tissue=concentration.mean(axis=0), dt=float(time[1] - time[0]),
    ))
    counts = candidate_orders(signals.shape[1], model=family)
    allowed = []
    for count in counts:
        model = model_for(signals[0], aif, time, te, count, family=family)
        if model.valid_mean(model.prior.mean):
            allowed.append(count)
    if not allowed:
        raise ValueError("acquisition has no supported count with feasible prior geometry")
    selected = min(allowed, key=lambda n: (abs(n - requested), n))
    restrictions = []
    if requested not in counts:
        restrictions.append("tested_count_or_observation_limit")
    elif requested not in allowed:
        restrictions.append("spacing_prior_geometry")
    return CountDecision(
        requested, selected, tuple(allowed), int(baseline_frames), source,
        tuple(restrictions),
    )


@dataclass(frozen=True)
class FitAssessment:
    """Numerical validity separate from convergence and curve-fit diagnostics."""

    voxel: int
    restart: int
    valid: bool
    converged: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    r_squared: float | None
    gamma_rate_ratio: float | None
    covariance_identity_error: float | None
    covariance_relative_error: float | None = None


def assess_fit(
    fit: SharedFit,
    model: AssessedSignalModel,
    observed: np.ndarray,
    prediction: np.ndarray,
    jacobian: np.ndarray,
    covariance: np.ndarray,
    shape: float,
    scale: float,
    prior_covariance: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> FitAssessment:
    """Assess a complete ordinary-VI state without certifying its accuracy.

    Parameters
    ----------
    fit, model : SharedFit, AssessedSignalModel
        Original fit record and its unchanged forward model.
    observed, prediction, jacobian, covariance : ndarray
        Measured signal and complete posterior evaluated at the returned mean.
    shape, scale : float
        Gamma noise-precision parameters in shape/scale convention.
    prior_covariance : ndarray
        Gaussian covariance actually used by VI, including the current
        conditional spatial prior when assessing a spatial update.
    weights : ndarray, optional
        Frame likelihood weights used by the fit and noise update. Signal R2
        and the reported RSS remain unweighted whole-curve diagnostics.

    Returns
    -------
    FitAssessment
        Hard numerical failure reasons and independent descriptive warnings.
    """
    weights = checked_weights(weights, observed.size)
    effective_count = observed.size if weights is None else float(weights.sum())
    reasons: list[str] = []
    warnings: list[str] = []
    mean = np.asarray(fit.mean)
    if not model.valid_mean(mean):
        reasons.append("invalid_mean_domain")
    finite = all(np.all(np.isfinite(x)) for x in (
        mean, observed, prediction, jacobian, covariance, prior_covariance,
        [shape, scale, fit.score, fit.cbf, fit.cbv, fit.mtt, fit.delay,
         fit.s0, fit.noise_sd, fit.rss],
    ))
    if not finite:
        reasons.append("nonfinite_state")
    expected_p = model.prior.mean.size
    dimensions = (
        observed.shape == prediction.shape == model.observed.shape
        and jacobian.shape == (observed.size, expected_p)
        and covariance.shape == prior_covariance.shape == (expected_p, expected_p)
    )
    if not dimensions:
        reasons.append("invalid_state_shape")
    if shape <= 0 or scale <= 0 or fit.noise_sd <= 0:
        reasons.append("invalid_noise_precision")
    if fit.cbf < 0 or fit.cbv < 0 or fit.mtt <= 0 or fit.s0 <= 0:
        reasons.append("invalid_readout")
    ratio = identity = covariance_error = r_squared = None
    if finite and dimensions:
        if model.valid_mean(mean):
            if not np.allclose(prediction, model.predict(mean), rtol=0,
                               atol=SIGNAL_BASELINE_RELATIVE_ERROR * mean[1]):
                reasons.append("prediction_mean_mismatch")
            mtt = float(model.residue(mean).integrate(0, model.duration))
            if not np.allclose(
                [fit.cbf, fit.mtt, fit.delay, fit.s0],
                [mean[0] * 6000, mtt, model.arrival_delay(mean), mean[1]],
                rtol=1e-10, atol=1e-10,
            ):
                reasons.append("mean_readout_mismatch")
        residual = observed - prediction
        rss = float(residual @ residual)
        total = float(np.sum((observed - observed.mean()) ** 2))
        if total > 0:
            r_squared = 1.0 - rss / total
        if rss > total:
            warnings.append("worse_than_constant_signal")
        if not np.isclose(fit.rss, rss, rtol=1e-8, atol=1e-8):
            reasons.append("residual_readout_mismatch")
        if not np.isclose(fit.cbv, fit.cbf * fit.mtt / 60, rtol=1e-10, atol=1e-10):
            reasons.append("central_volume_identity")
        try:
            np.linalg.cholesky(covariance)
            np.linalg.cholesky(prior_covariance)
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
                reasons.append("asymmetric_covariance")
            if shape > 0 and scale > 0:
                if not np.isclose(fit.noise_sd, 1 / np.sqrt(shape * scale), rtol=1e-10):
                    reasons.append("noise_readout_mismatch")
                diagnostic = factor_diagnostic(
                    covariance, jacobian, prior_covariance, residual, shape, scale, weights
                )
                ratio = diagnostic["gamma_rate_ratio"]
                identity = diagnostic["covariance_precision_identity_max_eigen_error"]
                covariance_error = diagnostic["covariance_relative_error"]
                if (
                    not np.isclose(ratio, 1.0, rtol=1e-6, atol=0)
                    or not np.isfinite(identity)
                    or identity > _LOCAL_FACTOR_IDENTITY_ATOL
                    or not np.isfinite(covariance_error)
                    or covariance_error > _LOCAL_FACTOR_COVARIANCE_RTOL
                    or not np.isclose(shape, 1e-6 + effective_count / 2, rtol=1e-10)
                ):
                    reasons.append("inconsistent_local_factors")
        except np.linalg.LinAlgError:
            reasons.append("nonpositive_definite_covariance")
    if not fit.converged:
        warnings.append("not_converged")
    if fit.cbf == 0:
        warnings.append("zero_flow")
    return FitAssessment(
        fit.voxel, fit.restart, not reasons, fit.converged,
        tuple(reasons), tuple(warnings), r_squared, ratio, identity, covariance_error,
    )


def select_valid_restarts(
    fits: Sequence[SharedFit], assessments: Sequence[FitAssessment], n_voxels: int
) -> list[dict[str, Any]]:
    """Select within one count, preserving failure/provisional voxel records."""
    if len(fits) != len(assessments) or len({f.order for f in fits}) > 1:
        raise ValueError("aligned assessments at exactly one count required")
    pairs = list(zip(fits, assessments, strict=True))
    seen: set[tuple[int, int]] = set()
    for fit, assessment in pairs:
        key = (fit.voxel, fit.restart)
        if (
            key != (assessment.voxel, assessment.restart) or key in seen
            or not 0 <= fit.voxel < n_voxels
        ):
            raise ValueError("invalid or duplicated voxel/restart ownership")
        seen.add(key)
    rows = []
    for voxel in range(n_voxels):
        valid = [(f, a) for f, a in pairs if f.voxel == voxel and a.valid]
        best = max(valid, key=lambda pair: (pair[0].score, -pair[0].restart)) if valid else None
        rows.append({
            "voxel": voxel,
            "status": "failed" if best is None else (
                "converged" if best[0].converged else "provisional"
            ),
            "selected": None if best is None else best[0],
            "assessment": None if best is None else best[1],
            "n_valid_starts": len(valid),
        })
    return rows
