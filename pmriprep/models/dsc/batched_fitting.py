"""Shared batched VI/SVB support for explicit forward-model adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Final, Optional

import numpy as np
import numpy.typing as npt
from scipy.special import digamma, gammaln

from pmriprep.models.bayesian_utils import (
    TransformType,
    _bounded_vi_mean,
    from_transformed_batch,
    log_jacobian_det,
    to_transformed_batch,
    transform_prior,
)
from pmriprep.models.dsc.vi_adapter import (
    DSCVIModelAdapter,
    DSCVIPosterior,
    feasible_mean_update,
)
from pmriprep.models.dsc.local_vi import (
    LocalVIFactorError,
    LocalVIState,
    conditional_mean_gain,
    local_vi_state,
)
from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
)

logger = logging.getLogger("nipype.interface")

#: Numerical floor used as a divide guard in batched VI fits
#: (R² denominators, sigma² floors, Jacobian column scaling). Set well
#: below any physical scale so it only fires on truly degenerate inputs.
#: Not a physics constant.
_EPSILON_NUMERICAL: Final[float] = 1e-30


def _isolated_local_vi_state(
    mean: npt.NDArray[np.float64],
    jacobian: npt.NDArray[np.float64],
    residual: npt.NDArray[np.float64],
    prior_mean: npt.NDArray[np.float64],
    prior_precision: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[LocalVIState, npt.NDArray[np.bool_]]:
    """Settle local factors while identifying only numerically failed rows."""
    try:
        return (
            local_vi_state(mean, jacobian, residual, prior_mean, prior_precision, weights),
            np.zeros(len(mean), dtype=bool),
        )
    except LocalVIFactorError:
        pass

    covariance = np.full((len(mean), mean.shape[1], mean.shape[1]), np.nan)
    scale = np.full(len(mean), np.nan)
    score = np.full(len(mean), np.nan)
    failed = np.zeros(len(mean), dtype=bool)
    shape = 1e-6 + float(np.sum(weights)) / 2
    for row in range(len(mean)):
        row_slice = slice(row, row + 1)
        try:
            state = local_vi_state(
                mean[row_slice], jacobian[row_slice], residual[row_slice],
                prior_mean, prior_precision, weights,
            )
        except LocalVIFactorError:
            failed[row] = True
            continue
        covariance[row] = state.covariance[0]
        scale[row] = state.scale[0]
        score[row] = state.score[0]
        shape = state.shape
    return LocalVIState(covariance, shape, scale, score), failed


def _build_vi_update_prior_terms(
    prior_prec: npt.NDArray[np.floating],
    prior_means_t: npt.NDArray[np.floating],
    mono_prec: Optional[npt.NDArray[np.floating]],
    mono_eta: Optional[npt.NDArray[np.floating]],
    *,
    spatial_replaces_parameter_prior: bool,
    spatial_parameter_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build parameter-update prior terms for ordinary VI or FABBER M-prior.

    Parameters
    ----------
    prior_prec : npt.NDArray[np.floating]
        Complete independent-prior precision, including any structural
        monotonicity precision.
    prior_means_t : npt.NDArray[np.floating]
        Independent-prior means in transformed space.
    mono_prec : npt.NDArray[np.floating], optional
        Structural monotonicity precision.
    mono_eta : npt.NDArray[np.floating], optional
        Structural monotonicity information-vector contribution.
    spatial_replaces_parameter_prior : bool
        Whether a boundary-corrected FABBER M-prior replaces the independent
        parameter prior.
    spatial_parameter_mask : npt.NDArray[np.bool_], optional
        Parameters receiving the M-prior. ``None`` selects every parameter.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Base precision and information vector to which per-voxel spatial
        contributions are added.
    """
    n_params = len(prior_means_t)
    if spatial_replaces_parameter_prior:
        selected = (
            np.ones(n_params, dtype=bool)
            if spatial_parameter_mask is None
            else np.asarray(spatial_parameter_mask, dtype=bool)
        )
        structural_prec = (
            np.asarray(mono_prec, dtype=np.float64)
            if mono_prec is not None
            else np.zeros((n_params, n_params), dtype=np.float64)
        )
        independent_prec = np.asarray(prior_prec, dtype=np.float64) - structural_prec
        independent_prec = independent_prec.copy()
        selected_idx = np.flatnonzero(selected)
        independent_prec[selected_idx, selected_idx] = 0.0
        update_prec = independent_prec + structural_prec
        update_eta = update_prec @ np.asarray(prior_means_t, dtype=np.float64)
        if mono_eta is not None:
            update_eta = update_eta + mono_eta
        return update_prec, update_eta

    update_prec = np.asarray(prior_prec, dtype=np.float64)
    update_eta = update_prec @ np.asarray(prior_means_t, dtype=np.float64)
    if mono_eta is not None:
        update_eta = update_eta + mono_eta
    return update_prec, update_eta


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DSCBatchedFitResult:
    """Result from DSC batched fitting.

    Attributes
    ----------
        params: Parameter arrays, param_name -> (N,).
        r_squared: R-squared values, shape (N,).
        converged: Convergence flags, shape (N,).
        fitted_curves: Fitted tissue curves, shape (N, T) or None.
        n_iterations: Number of fitting iterations performed.
        model_name: Name of the VI adapter.
        uncertainties: Parameter uncertainties, param_name -> (N,).
        full_posterior: Optional complete local Gaussian/Gamma factors.
        failure_reasons: Per-voxel numerical failure reason, or ``None``.
    """

    # Use Mapping (covariant) rather than dict (invariant) so callers can
    # pass dicts keyed by narrower numpy dtypes (e.g. float64 at construction)
    # without mypy rejecting the subtype. All current consumers only read
    # from params/uncertainties — no write-back needed.
    params: Mapping[str, npt.NDArray[np.floating]]
    r_squared: npt.NDArray[np.floating]
    converged: npt.NDArray[np.bool_]
    fitted_curves: Optional[npt.NDArray[np.floating]]
    n_iterations: int
    model_name: str
    uncertainties: Mapping[str, npt.NDArray[np.floating]] = field(default_factory=dict)
    posterior_means_transformed: Optional[npt.NDArray[np.floating]] = None
    posterior_cov_diag: Optional[npt.NDArray[np.floating]] = None
    credible_interval_lower: Mapping[str, npt.NDArray[np.floating]] = field(
        default_factory=dict
    )
    credible_interval_upper: Mapping[str, npt.NDArray[np.floating]] = field(
        default_factory=dict
    )
    full_posterior: DSCVIPosterior | None = None
    failure_reasons: tuple[str | None, ...] = ()



def _assemble_JtWJ(
    J_columns: list[npt.NDArray[np.floating]],
    k_res: npt.NDArray[np.floating],
    w: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Assemble J^T W J and J^T W k from Jacobian columns.

    Parameters
    ----------
    J_columns : list[npt.NDArray[np.floating]]
        List of P arrays, each shape (n_active, T).
    k_res : npt.NDArray[np.floating]
        Residuals, shape (n_active, T).
    w : npt.NDArray[np.floating]
        Per-frame weights, shape (T,).

    Returns
    -------
    JtWJ : npt.NDArray[np.floating]
        shape (n_active, P, P).
    JtWk : npt.NDArray[np.floating]
        shape (n_active, P).
    """
    n_params = len(J_columns)
    n_active = J_columns[0].shape[0]
    JtWJ = np.zeros((n_active, n_params, n_params))
    JtWk = np.zeros((n_active, n_params))
    if n_params >= 16 and all(column.flags.c_contiguous for column in J_columns):
        jacobian = np.asarray(J_columns)
        weighted = w[None, None, :] * jacobian
        JtWk[:] = np.sum(weighted * k_res[None], axis=2).T
        rows, columns = np.triu_indices(n_params)
        # Bound the two gathered column blocks and their product together.
        pair_bytes = max(1, 3 * n_active * k_res.shape[1] * jacobian.itemsize)
        block_size = max(1, min(64, 8_000_000 // pair_bytes))
        for first in range(0, rows.size, block_size):
            js, ks = rows[first:first+block_size], columns[first:first+block_size]
            values = np.sum(weighted[js] * jacobian[ks], axis=2).T
            JtWJ[:, js, ks] = values
            JtWJ[:, ks, js] = values
        return JtWJ, JtWk
    for j in range(n_params):
        wJ_j = w[None, :] * J_columns[j]
        JtWk[:, j] = np.sum(wJ_j * k_res, axis=1)
        for k in range(j, n_params):
            val = np.sum(wJ_j * J_columns[k], axis=1)
            JtWJ[:, j, k] = val
            JtWJ[:, k, j] = val
    return JtWJ, JtWk


def _compute_r_squared(
    observed: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute R-squared for each voxel.

    Parameters
    ----------
    observed : npt.NDArray[np.floating]
        Observed curves, shape (N, T).
    predicted : npt.NDArray[np.floating]
        Predicted curves, shape (N, T).

    Returns
    -------
    npt.NDArray[np.floating]
        R-squared values, shape (N,).
    """
    ss_res = np.sum((observed - predicted) ** 2, axis=1)
    ss_tot = np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, _EPSILON_NUMERICAL)
    result: npt.NDArray[np.floating] = np.clip(r2, -1.0, 1.0)
    return result


def batched_dsc_vi_fit(
    model_type: str,
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    prior_means_nat: npt.NDArray[np.floating],
    prior_stds_nat: npt.NDArray[np.floating],
    transforms: list[TransformType],
    upper_bounds: list[float | None],
    fitting_param_names: list[str],
    fitting_bounds: list[tuple[float, float]],
    mono_prec: Optional[npt.NDArray[np.floating]] = None,
    mono_eta: Optional[npt.NDArray[np.floating]] = None,
    nlls_seeds: Optional[npt.NDArray[np.floating]] = None,
    prior_means_transformed: Optional[npt.NDArray[np.floating]] = None,
    prior_stds_transformed: Optional[npt.NDArray[np.floating]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    n_iterations: int = 10,
    sloppy: bool = False,
    save_fitted_curves: bool = False,
    spatial_prec_diag: Optional[npt.NDArray[np.floating]] = None,
    spatial_eta: Optional[npt.NDArray[np.floating]] = None,
    spatial_replaces_parameter_prior: bool = False,
    spatial_parameter_mask: Optional[npt.NDArray[np.bool_]] = None,
    initial_means_transformed: Optional[npt.NDArray[np.floating]] = None,
    initial_cov_diag: Optional[npt.NDArray[np.floating]] = None,
    initial_noise_precision: Optional[npt.NDArray[np.floating]] = None,
    return_posteriors: bool = False,
    n_iterations_override: Optional[int] = None,
    nprocs: int = 1,
    model_adapter: DSCVIModelAdapter | None = None,
    return_full_posterior: bool = False,
) -> DSCBatchedFitResult:
    """Batched Variational Bayes fitting for DSC perfusion models.

    Implements the linearized VB algorithm vectorized across all voxels.
    Requires an explicitly supplied model adapter.

    Model bounds constrain the local Gaussian mean updates, including
    spatial updates. Covariance remains a local Gaussian approximation;
    intervals at active bounds are not exact truncated-posterior intervals.

    Parameters
    ----------
    model_type : str
        Identifier of the supplied model adapter.
    tissue_curves : npt.NDArray[np.floating]
        Observed tissue curves, shape (N, T).
    aif : npt.NDArray[np.floating]
        Arterial input function, shape (T,).
    time_points : npt.NDArray[np.floating]
        Time points (seconds), shape (T,).
    prior_means_nat : npt.NDArray[np.floating]
        Prior means in natural (fitting) space, shape (P,).
    prior_stds_nat : npt.NDArray[np.floating]
        Prior stds in natural (fitting) space, shape (P,).
    transforms : list[TransformType]
        Per-parameter transform types, length P.
    upper_bounds : list[float | None]
        Per-parameter upper bounds for LOG_LOGIT, length P.
    fitting_param_names : list[str]
        Parameter names in fitting space, length P.
    fitting_bounds : list[tuple[float, float]]
        Per-parameter bounds in fitting space, length P.
    mono_prec : npt.NDArray[np.floating], optional
        Structural monotonicity precision matrix, shape (P, P) or None.
    nlls_seeds : npt.NDArray[np.floating], optional
        NLLS seed parameters for initialization, shape (N, P) or None.
    prior_means_transformed : npt.NDArray[np.floating], optional
        Pre-computed prior means in transformed space,
        shape (P,). If provided, bypasses internal transform_prior() call.
        Use this to preserve transformed_prior_sigma overrides from
        BayesianPrior objects (e.g., FABBER sigma=1.0 convention).
    prior_stds_transformed : npt.NDArray[np.floating], optional
        Pre-computed prior stds in transformed space,
        shape (P,). Must be provided alongside prior_means_transformed.
    weights : npt.NDArray[np.floating], optional
        Per-frame motion weights, shape (T,). None = uniform.
    n_iterations : int, optional
        Maximum VB iterations (default 20).
    sloppy : bool, optional
        Use fast, low-quality settings (8 iterations).
    save_fitted_curves : bool, optional
        Whether to store fitted curves in result.
    spatial_prec_diag : npt.NDArray[np.floating], optional
        Per-voxel spatial precision diagonal, shape (N, P).
        N must match tissue_curves.shape[0].
    spatial_eta : npt.NDArray[np.floating], optional
        Per-voxel spatial mean contribution, shape (N, P).
    spatial_replaces_parameter_prior : bool, optional
        Replace the independent parameter prior with ``spatial_prec_diag`` and
        ``spatial_eta``, matching FABBER's boundary-corrected M-prior. Structural
        Structural monotonicity terms remain active. Requires both spatial arrays.
    spatial_parameter_mask : npt.NDArray[np.bool_], optional
        Boolean parameter selector for M-prior replacement, shape (P,).
    initial_means_transformed : npt.NDArray[np.floating], optional
        Initial posterior means in transformed space, shape (N, P). Used by
        stateful spatial-VB sweeps.
    initial_cov_diag : npt.NDArray[np.floating], optional
        Initial posterior covariance diagonal in transformed space, shape
        (N, P). Retained when a spatial proposal fails conditional-objective
        backtracking.
    initial_noise_precision : npt.NDArray[np.floating], optional
        Initial expected noise precision, shape (N,). Used by stateful
        spatial-VB sweeps.
    return_posteriors : bool, optional
        If True, populate posterior_means_transformed and
        posterior_cov_diag on the result.
    n_iterations_override : int, optional
        When set, overrides n_iterations and sloppy,
        and disables ELBO-based step-size reduction. Used by spatial VB
        wrapper where the ELBO is invalidated by spatial terms.
    nprocs : int, optional
        Must be one; callers schedule adapter fits externally.
    model_adapter : DSCVIModelAdapter, optional
        Observation prediction, natural-coordinate Jacobian and mean domain.
        Overrides the built-in forward model and concentration-specific flow
        derivative. Requires single-process execution; evaluation remains batched.
    return_full_posterior : bool, optional
        Retain full within-voxel covariance and Gamma noise factors for
        diagnostics/model comparison. Requires single-process execution.

    Returns
    -------
    DSCBatchedFitResult
        DSCBatchedFitResult with fitted parameters, R², convergence, uncertainties.
    """
    if nprocs != 1:
        raise ValueError("model adapters and full posterior output require nprocs=1")
    if model_adapter is None:
        raise ValueError(f"Unsupported DSC model without an adapter: {model_type}")

    tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)
    n_voxels, n_obs = tissue_curves.shape
    n_params = len(fitting_param_names)

    if spatial_replaces_parameter_prior and (
        spatial_prec_diag is None or spatial_eta is None
    ):
        raise ValueError(
            "Spatial M-prior replacement requires both spatial_prec_diag "
            "and spatial_eta"
        )
    if spatial_replaces_parameter_prior and n_iterations_override is None:
        raise ValueError(
            "Spatial M-prior replacement requires spatial one-update mode "
            "(n_iterations_override must be set)"
        )
    if spatial_parameter_mask is not None and (
        np.shape(spatial_parameter_mask) != (n_params,)
    ):
        raise ValueError(
            f"spatial_parameter_mask must have shape ({n_params},)"
        )
    if initial_means_transformed is not None and (
        np.shape(initial_means_transformed) != (n_voxels, n_params)
    ):
        raise ValueError(
            "initial_means_transformed must have shape "
            f"({n_voxels}, {n_params})"
        )
    if initial_cov_diag is not None and (
        np.shape(initial_cov_diag) != (n_voxels, n_params)
    ):
        raise ValueError(
            f"initial_cov_diag must have shape ({n_voxels}, {n_params})"
        )
    if initial_noise_precision is not None and (
        np.shape(initial_noise_precision) != (n_voxels,)
    ):
        raise ValueError(
            f"initial_noise_precision must have shape ({n_voxels},)"
        )

    if n_iterations_override is not None:
        n_iterations = n_iterations_override
    elif sloppy:
        n_iterations = 8
    # Disables ELBO step-size/convergence (spatial_vb.py is sole consumer)
    _spatial_mode = n_iterations_override is not None
    _consistent_mode = model_adapter is not None and not _spatial_mode
    if _consistent_mode and (spatial_prec_diag is not None or spatial_eta is not None):
        raise ValueError("consistent local factors require ordinary non-spatial VI")
    if _consistent_mode and n_iterations < 1:
        raise ValueError("consistent local fitting requires at least one iteration")

    # Pre-filter: exclude voxels with negligible signal
    voxel_max = np.max(np.abs(tissue_curves), axis=1)
    valid_mask = voxel_max > EPSILON_CONCENTRATION
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    # Initialize output arrays
    out_params = {name: np.zeros(n_voxels) for name in fitting_param_names}
    out_r2 = np.full(n_voxels, -1.0)
    out_converged = np.zeros(n_voxels, dtype=bool)
    out_uncertainties = {name: np.full(n_voxels, np.nan) for name in fitting_param_names}
    out_uncertainties["noise_std"] = np.full(n_voxels, np.nan)

    if n_valid == 0:
        return DSCBatchedFitResult(
            params=out_params, r_squared=out_r2, converged=out_converged,
            fitted_curves=None, n_iterations=0, model_name=model_type,
            uncertainties=out_uncertainties,
            full_posterior=DSCVIPosterior(
                np.full((n_voxels, n_params), np.nan),
                np.full((n_voxels, n_params, n_params), np.nan),
                np.full(n_voxels, np.nan),
                np.full(n_voxels, np.nan),
            ) if return_full_posterior else None,
            failure_reasons=(None,) * n_voxels,
        )

    y = tissue_curves[valid_idx]  # (N_valid, T)

    # --- Prior setup ---
    # Use pre-transformed means/stds if provided (preserves transformed_prior_sigma
    # overrides from transformed-coordinate priors).
    if prior_means_transformed is not None and prior_stds_transformed is not None:
        prior_means_t = np.asarray(prior_means_transformed, dtype=np.float64)
        prior_stds_t = np.asarray(prior_stds_transformed, dtype=np.float64)
    else:
        prior_means_t = np.zeros(n_params, dtype=np.float64)
        prior_stds_t = np.zeros(n_params, dtype=np.float64)
        for j in range(n_params):
            prior_means_t[j], prior_stds_t[j] = transform_prior(
                float(prior_means_nat[j]), float(prior_stds_nat[j]),
                transforms[j], upper_bounds[j],
            )
    prior_prec = np.diag(1.0 / (prior_stds_t ** 2))

    # Add Structural monotonicity penalty
    if mono_prec is not None:
        prior_prec = prior_prec + mono_prec

    # FABBER's boundary-corrected M-prior replaces the independent parameter
    # prior rather than adding to it. Structural monotonicity terms remain.
    update_prior_prec, update_prior_eta = _build_vi_update_prior_terms(
        prior_prec,
        prior_means_t,
        mono_prec,
        mono_eta,
        spatial_replaces_parameter_prior=spatial_replaces_parameter_prior,
        spatial_parameter_mask=spatial_parameter_mask,
    )

    # Noise precision prior — FABBER non-informative defaults
    # (fabber_core/noisemodel_white.cc: c=1e-6, b=1e6).
    c0 = 1e-6
    s0 = 1e6
    # Posterior init for Gamma noise (FABBER posterior init: c=50, s=1e-8)
    c_post_init = 50.0
    s_post_init = 1e-8

    # --- Bounds ---
    bounds_lo = np.array([b[0] for b in fitting_bounds])
    bounds_hi = np.array([b[1] for b in fitting_bounds])

    transformed_bounds = None
    if model_adapter is not None:
        lower_t = np.full(n_params, -np.inf)
        upper_t = np.full(n_params, np.inf)
        for j, (transform, (lo, hi)) in enumerate(zip(transforms, fitting_bounds)):
            if transform == TransformType.LOG:
                lower_t[j] = np.log(lo) if lo > 0 else -np.inf
                upper_t[j] = np.log(hi)
            elif transform == TransformType.IDENTITY:
                lower_t[j], upper_t[j] = lo, hi
        transformed_bounds = (lower_t, upper_t)

    # --- Initialize posterior mean ---
    m_nat = np.tile(prior_means_nat, (n_valid, 1)).astype(np.float64)
    for j in range(n_params):
        m_nat[:, j] = np.clip(m_nat[:, j], bounds_lo[j], bounds_hi[j])

    if initial_means_transformed is not None:
        initial_t = np.asarray(initial_means_transformed, dtype=np.float64)[valid_idx]
        for j in range(n_params):
            m_nat[:, j] = from_transformed_batch(
                initial_t[:, j], transforms[j], upper_bounds[j],
            )
            m_nat[:, j] = np.clip(m_nat[:, j], bounds_lo[j], bounds_hi[j])

    # Override with NLLS seeds where available
    if nlls_seeds is not None:
        valid_seeds = nlls_seeds[valid_idx]
        has_seed = np.any(np.abs(valid_seeds) > 0, axis=1)
        if np.any(has_seed):
            m_nat[has_seed] = valid_seeds[has_seed]
            for j in range(n_params):
                m_nat[:, j] = np.clip(m_nat[:, j], bounds_lo[j], bounds_hi[j])

    # Transform to fitting space
    m = np.zeros((n_valid, n_params), dtype=np.float64)
    for j in range(n_params):
        m[:, j] = to_transformed_batch(m_nat[:, j], transforms[j], upper_bounds[j])

    # --- Forward model dispatch ---
    def _evaluate(params_fitting: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Evaluate forward model from fitting-space natural parameters."""
        values = np.asarray(params_fitting, dtype=np.float64)
        if not np.all(model_adapter.admissible(values)):
            raise ValueError("forward parameters violate the model domain")
        prediction = np.asarray(model_adapter.predict(values), dtype=np.float64)
        if prediction.shape != (values.shape[0], n_obs) or not np.all(
            np.isfinite(prediction)
        ):
            raise ValueError("model prediction must be finite with shape (N, T)")
        return prediction

    if model_adapter is not None and not np.all(model_adapter.admissible(m_nat)):
        raise ValueError("initial mean violates the model domain")

    # --- Noise posterior init ---
    # FABBER's GetInitialPosterior c=50, s=1e-8 gives tau ~ 5e-7 — far below
    # typical DSC concentration-residual precision (~1/0.01^2 = 1e4). To get
    # the first VB iteration off the ground, seed tau from residuals at the
    # initial-guess forward, which mirrors what FABBER's noise posterior
    # converges to within 1-2 iterations anyway.
    try:
        g_init = _evaluate(m_nat)
        init_var = np.sum((y - g_init) ** 2, axis=1) / n_obs
        good_var = init_var > 1e-20
        tau_exp = np.full(n_valid, c_post_init * s_post_init)
        tau_exp[good_var] = 1.0 / init_var[good_var]
    except (ValueError, RuntimeError):
        tau_exp = np.full(n_valid, c_post_init * s_post_init)

    if initial_noise_precision is not None:
        initial_tau = np.asarray(initial_noise_precision, dtype=np.float64)[valid_idx]
        good_tau = np.isfinite(initial_tau) & (initial_tau > 0)
        tau_exp[good_tau] = initial_tau[good_tau]

    c = np.full(n_valid, c_post_init, dtype=np.float64)
    s = np.full(n_valid, s_post_init, dtype=np.float64)
    if initial_noise_precision is not None:
        np.divide(tau_exp, c, out=s)

    # --- State tracking ---
    best_fe = np.full(n_valid, -np.inf)
    best_m = m.copy()
    best_cov_diag: npt.NDArray[np.float64] = np.tile(
        prior_stds_t ** 2, (n_valid, 1),
    )
    if initial_cov_diag is not None:
        best_cov_diag = np.asarray(initial_cov_diag, dtype=np.float64)[valid_idx].copy()
    best_cov_full = None
    if return_full_posterior:
        best_cov_full = np.zeros((n_valid, n_params, n_params), dtype=np.float64)
        best_cov_full[:, range(n_params), range(n_params)] = best_cov_diag
    best_c = c.copy()
    best_s = s.copy()
    step_scale = np.ones(n_valid)

    active = np.ones(n_valid, dtype=bool)
    converged = np.zeros(n_valid, dtype=bool)
    failure_reasons: list[str | None] = [None] * n_valid
    fe_plateau_count = np.zeros(n_valid, dtype=np.int32)

    w = np.ones(n_obs, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    n_eff = float(np.sum(w))

    # CBF index for analytic Jacobian (constant across iterations)
    cbf_idx = fitting_param_names.index("cbf")
    if transforms[cbf_idx] not in (TransformType.LOG, TransformType.IDENTITY):
        raise ValueError(
            f"Analytic CBF Jacobian requires LOG or IDENTITY transform, "
            f"got {transforms[cbf_idx]} for param '{fitting_param_names[cbf_idx]}'"
        )

    # Constant prior precision log-determinant
    _sign_prior_prec, _logdet_prior_prec = np.linalg.slogdet(prior_prec)
    _logdet_prior_cov = -_logdet_prior_prec

    last_iteration = 0

    def _jacobian_columns(
        means: npt.NDArray[np.floating],
        natural: npt.NDArray[np.floating],
        prediction: npt.NDArray[np.floating],
    ) -> list[npt.NDArray[np.floating]]:
        """Evaluate every derivative at the supplied mean and prediction."""
        natural_jacobian = np.asarray(
            model_adapter.jacobian(np.asarray(natural, dtype=np.float64)), dtype=np.float64
        )
        if natural_jacobian.shape != (len(means), n_obs, n_params) or not np.all(
            np.isfinite(natural_jacobian)
        ):
            raise ValueError("model Jacobian must be finite with shape (N, T, P)")
        return [
            natural_jacobian[:, :, j] * np.broadcast_to(
                np.exp(log_jacobian_det(natural[:, j], transforms[j], upper_bounds[j])),
                (len(means),),
            )[:, None]
            for j in range(n_params)
        ]

    local_prior_mean = np.asarray(
        np.linalg.solve(update_prior_prec, update_prior_eta) if _consistent_mode else prior_means_t,
        dtype=np.float64,
    )

    for iteration in range(n_iterations):
        last_iteration = iteration
        n_active = int(np.sum(active))
        if n_active == 0:
            break

        a_idx = np.where(active)[0]
        m_a = m[a_idx]
        y_a = y[a_idx]
        tau_a = tau_exp[a_idx]

        # 1. Inverse-transform to fitting-space natural params
        m_nat_a = np.zeros_like(m_a)
        for j in range(n_params):
            m_nat_a[:, j] = from_transformed_batch(
                m_a[:, j], transforms[j], upper_bounds[j],
            )

        # 2. Forward model at current mean
        g_m = _evaluate(m_nat_a)

        # 3. Jacobian: analytic CBF + forward FD for remaining params
        J_columns = _jacobian_columns(m_a, m_nat_a, g_m)

        # Residual
        k_res = y_a - g_m

        # 4. Build JtWJ via column-loop
        JtWJ, JtWk = _assemble_JtWJ(J_columns, k_res, w)

        if _consistent_mode:
            current_state, factor_failed = _isolated_local_vi_state(
                m_a, np.stack(J_columns, axis=-1), k_res,
                local_prior_mean, update_prior_prec, w,
            )
            if np.any(factor_failed):
                failed_indices = a_idx[factor_failed]
                for failed_index in failed_indices:
                    failure_reasons[int(failed_index)] = "local_factor_failed"
                active[failed_indices] = False
                converged[failed_indices] = False
                logger.debug(
                    "DSC VI local factors failed for %d fit row(s)",
                    len(failed_indices),
                )
                # Retry the surviving rows together on the next iteration so
                # every downstream array remains aligned to the active set.
                continue
            tau_a = current_state.shape * current_state.scale
            best_fe[a_idx] = current_state.score
            best_m[a_idx] = m_a
            best_cov_diag[a_idx] = np.diagonal(current_state.covariance, axis1=1, axis2=2)
            if best_cov_full is not None:
                best_cov_full[a_idx] = current_state.covariance
            best_c[a_idx] = current_state.shape
            best_s[a_idx] = current_state.scale

        # 5. Posterior precision and covariance
        post_prec = update_prior_prec[None, :, :] + tau_a[:, None, None] * JtWJ
        if spatial_prec_diag is not None:
            # (N_total, P) -> valid subset -> active subset
            sp_a = spatial_prec_diag[valid_idx][a_idx]
            post_prec[:, range(n_params), range(n_params)] += sp_a

        post_cov = current_state.covariance if _consistent_mode else np.linalg.inv(post_prec)
        bad_cov = ~np.isfinite(post_cov).all(axis=(1, 2))
        if np.any(bad_cov):
            post_prec[bad_cov] += 1e-6 * np.eye(n_params)
            post_cov[bad_cov] = np.linalg.inv(post_prec[bad_cov])

        # 6. Posterior mean update
        Jm = np.zeros((n_active, n_obs))
        for j in range(n_params):
            Jm += J_columns[j] * m_a[:, j:j + 1]

        lam0_mu0 = update_prior_eta
        data_term = np.zeros((n_active, n_params))
        for j in range(n_params):
            data_term[:, j] = np.sum(
                w[None, :] * J_columns[j] * (k_res + Jm), axis=1,
            )
        rhs = lam0_mu0[None, :] + tau_a[:, None] * data_term
        if spatial_eta is not None:
            sp_eta_a = spatial_eta[valid_idx][a_idx]
            rhs = rhs + sp_eta_a

        m_new = np.einsum("nij,nj->ni", post_cov, rhs)
        if _consistent_mode:
            mean_gradient = tau_a[:, None] * JtWk - (
                m_a - local_prior_mean
            ) @ update_prior_prec
            m_new = m_a + np.einsum("nij,nj->ni", post_cov, mean_gradient)

        if transformed_bounds is not None:
            outside = np.any(
                ~np.isfinite(m_new) | (m_new < lower_t) | (m_new > upper_t), axis=1
            )
            for i in np.flatnonzero(outside):
                try:
                    m_new[i] = _bounded_vi_mean(
                        post_prec[i], rhs[i], m_new[i], lower_t, upper_t,
                    )
                except RuntimeError as exc:
                    failed_index = int(a_idx[i])
                    failure_reasons[failed_index] = "bounded_mean_failed"
                    logger.debug(
                        "DSC VI bounded mean failed for fit row %d: %s",
                        int(valid_idx[failed_index]), exc,
                    )
                    active[failed_index] = False
                    converged[failed_index] = False
                    m_new[i] = m_a[i]
        else:
            # IDENTITY params: transformed and natural coordinates coincide.
            for j in range(n_params):
                if transforms[j] == TransformType.IDENTITY:
                    m_new[:, j] = np.clip(m_new[:, j], bounds_lo[j], bounds_hi[j])

        full_param_change = np.max(np.abs(m_new - m_a), axis=1)
        if model_adapter is not None:
            m_new = feasible_mean_update(
                m_a, m_new, model_adapter, transforms, upper_bounds
            )

        if _consistent_mode:
            proposal = m_new.copy()
            accepted = np.zeros(n_active, dtype=bool)
            step = np.ones(n_active)
            for _ in range(14):
                pending = np.flatnonzero(~accepted)
                if not len(pending):
                    break
                trial = m_a[pending] + step[pending, None] * (proposal[pending] - m_a[pending])
                natural = np.column_stack([
                    from_transformed_batch(trial[:, j], transforms[j], upper_bounds[j])
                    for j in range(n_params)
                ])
                assert model_adapter is not None
                invalid_trial = ~model_adapter.admissible(natural)
                if np.any(invalid_trial):
                    full_step = invalid_trial & (step[pending] == 1.0)
                    trial[full_step] = proposal[pending[full_step]]
                    trial[invalid_trial] = feasible_mean_update(
                        m_a[pending[invalid_trial]], trial[invalid_trial],
                        model_adapter, transforms, upper_bounds,
                    )
                    for j in range(n_params):
                        natural[invalid_trial, j] = from_transformed_batch(
                            trial[invalid_trial, j], transforms[j], upper_bounds[j]
                        )
                prediction = _evaluate(natural)
                gain = conditional_mean_gain(
                    trial - m_a[pending], prediction - g_m[pending], k_res[pending],
                    m_a[pending] - local_prior_mean, update_prior_prec, tau_a[pending], w,
                )
                improves = np.isfinite(gain) & (gain >= 0)
                local = pending[improves]
                if not len(local):
                    step[pending] *= 0.5
                    continue
                columns = _jacobian_columns(trial[improves], natural[improves], prediction[improves])
                trial_state, factor_failed = _isolated_local_vi_state(
                    trial[improves], np.stack(columns, axis=-1), y_a[local] - prediction[improves],
                    local_prior_mean, update_prior_prec, w,
                )
                settled = ~factor_failed
                settled_local = local[settled]
                global_indices = a_idx[settled_local]
                if len(settled_local):
                    best_m[global_indices] = trial[improves][settled]
                    best_fe[global_indices] = trial_state.score[settled]
                    best_cov_diag[global_indices] = np.diagonal(
                        trial_state.covariance[settled], axis1=1, axis2=2,
                    )
                    if best_cov_full is not None:
                        best_cov_full[global_indices] = trial_state.covariance[settled]
                    best_c[global_indices] = trial_state.shape
                    best_s[global_indices] = trial_state.scale[settled]
                    accepted[settled_local] = True
                step[local[factor_failed]] *= 0.5
                step[pending[~improves]] *= 0.5
            newly_converged = accepted & (full_param_change < 1e-3)
            converged[a_idx[newly_converged]] = True
            active[a_idx[~accepted | newly_converged]] = False
            failed_now = np.array(
                [failure_reasons[index] is not None for index in a_idx], dtype=bool
            )
            converged[a_idx[failed_now]] = False
            active[a_idx[failed_now]] = False
            m[a_idx] = best_m[a_idx]
            c[a_idx] = best_c[a_idx]
            s[a_idx] = best_s[a_idx]
            tau_exp[a_idx] = best_c[a_idx] * best_s[a_idx]
            continue

        # Spatial linearization can propose a poor full step for nonlinear CPI
        # residue parameters. Accept only steps that improve the conditional
        # likelihood-plus-M-prior objective for the current neighbor state.
        spatial_accepted = np.ones(n_active, dtype=bool)
        if _spatial_mode:
            effective_prec = np.broadcast_to(
                update_prior_prec, (n_active, n_params, n_params)
            ).copy()
            effective_eta = np.broadcast_to(
                update_prior_eta, (n_active, n_params)
            ).copy()
            if spatial_prec_diag is not None:
                effective_prec[:, range(n_params), range(n_params)] += sp_a
            if spatial_eta is not None:
                effective_eta += sp_eta_a

            def _conditional_objective(
                values_t: npt.NDArray[np.floating],
                prediction: npt.NDArray[np.floating],
            ) -> npt.NDArray[np.float64]:
                residual = y_a - prediction
                data_term = -0.5 * tau_a * np.sum(
                    w[None, :] * residual**2, axis=1
                )
                quadratic = np.einsum(
                    "ni,nij,nj->n", values_t, effective_prec, values_t
                )
                information = np.sum(effective_eta * values_t, axis=1)
                return np.asarray(
                    data_term - 0.5 * quadratic + information,
                    dtype=np.float64,
                )

            old_objective = _conditional_objective(m_a, g_m)
            raw_m_new = m_new.copy()
            step = np.ones(n_active, dtype=np.float64)
            accepted = np.zeros(n_active, dtype=bool)
            g_new = g_m.copy()
            for _ in range(14):
                trial_m = m_a + step[:, None] * (raw_m_new - m_a)
                for j in range(n_params):
                    if transforms[j] == TransformType.IDENTITY:
                        trial_m[:, j] = np.clip(
                            trial_m[:, j], bounds_lo[j], bounds_hi[j]
                        )
                trial_nat = np.zeros_like(trial_m)
                for j in range(n_params):
                    trial_nat[:, j] = from_transformed_batch(
                        trial_m[:, j], transforms[j], upper_bounds[j]
                    )
                trial_prediction = _evaluate(trial_nat)
                trial_objective = _conditional_objective(
                    trial_m, trial_prediction
                )
                newly_accepted = (
                    ~accepted
                    & np.isfinite(trial_objective)
                    & (trial_objective >= old_objective)
                )
                if np.any(newly_accepted):
                    m_new[newly_accepted] = trial_m[newly_accepted]
                    g_new[newly_accepted] = trial_prediction[newly_accepted]
                    accepted[newly_accepted] = True
                if np.all(accepted):
                    break
                step[~accepted] *= 0.5
            spatial_accepted = accepted
            if np.any(~accepted):
                m_new[~accepted] = m_a[~accepted]
                g_new[~accepted] = g_m[~accepted]
        else:
            m_new_nat = np.zeros_like(m_new)
            for j in range(n_params):
                m_new_nat[:, j] = from_transformed_batch(
                    m_new[:, j], transforms[j], upper_bounds[j],
                )
            g_new = _evaluate(m_new_nat)

        # 7. Noise posterior update
        k_new = y_a - g_new

        c_new = c0 + n_eff / 2.0
        trace_term = np.sum(
            post_cov * np.transpose(JtWJ, (0, 2, 1)), axis=(1, 2),
        )
        weighted_kk = np.sum(w[None, :] * k_new ** 2, axis=1)
        s_new = 1.0 / (1.0 / s0 + 0.5 * (weighted_kk + trace_term))
        tau_new = c_new * s_new

        # 8. Free energy (ELBO)
        dm = m_new - prior_means_t[None, :]
        sign_cov, logdet_cov = np.linalg.slogdet(post_cov)

        dm_prec_dm = np.sum(dm * (dm @ prior_prec), axis=1)
        trace_prior_cov = np.sum(
            prior_prec[None, :, :] * np.transpose(post_cov, (0, 2, 1)),
            axis=(1, 2),
        )

        free_energy = (
            0.5 * n_eff * (np.log(tau_new) - np.log(2 * np.pi))
            - 0.5 * tau_new * (weighted_kk + trace_term)
            - 0.5 * dm_prec_dm
            - 0.5 * trace_prior_cov
            + 0.5 * logdet_cov * sign_cov
            - 0.5 * _logdet_prior_cov * _sign_prior_prec
            + 0.5 * n_params
        )

        gamma_kl = -(
            (c_new - c0) * digamma(c_new)
            - gammaln(c_new) + gammaln(c0)
            + c0 * (np.log(s0) - np.log(s_new))
            + c_new * (s_new / s0 - 1.0)
        )
        free_energy += gamma_kl

        # 9. Step-size reduction
        if _spatial_mode:
            accepted_global = a_idx[spatial_accepted]
            if len(accepted_global) > 0:
                accepted_local = np.where(spatial_accepted)[0]
                best_m[accepted_global] = m_new[accepted_local]
                best_cov_diag[accepted_global] = np.diagonal(
                    post_cov[accepted_local], axis1=1, axis2=2
                )
                if best_cov_full is not None:
                    best_cov_full[accepted_global] = post_cov[accepted_local]
                best_c[accepted_global] = c_new
                best_s[accepted_global] = s_new[accepted_local]
        else:
            prev_best_fe = best_fe[a_idx].copy()
            improved = free_energy > prev_best_fe

            imp_local = np.where(improved)[0]
            if len(imp_local) > 0:
                imp_global = a_idx[imp_local]
                best_fe[imp_global] = free_energy[imp_local]
                best_m[imp_global] = m_new[imp_local]
                best_cov_diag[imp_global] = np.diagonal(
                    post_cov[imp_local], axis1=1, axis2=2,
                )
                if best_cov_full is not None:
                    best_cov_full[imp_global] = post_cov[imp_local]
                best_c[imp_global] = c_new
                best_s[imp_global] = s_new[imp_local]
                step_scale[imp_global] = 1.0

            not_imp_local = np.where(~improved)[0]
            if len(not_imp_local) > 0:
                not_imp_global = a_idx[not_imp_local]
                step_scale[not_imp_global] *= 0.5
                m_new[not_imp_local] = (
                    m_a[not_imp_local]
                    + step_scale[not_imp_global, None]
                    * (m_new[not_imp_local] - m_a[not_imp_local])
                )
                # Re-clip IDENTITY params after interpolation
                for j in range(n_params):
                    if transforms[j] == TransformType.IDENTITY:
                        m_new[not_imp_local, j] = np.clip(
                            m_new[not_imp_local, j], bounds_lo[j], bounds_hi[j],
                        )
                exhausted = step_scale[not_imp_global] < 1e-4
                if np.any(exhausted):
                    active[not_imp_global[exhausted]] = False

        # 10. Convergence checks
        if _spatial_mode:
            pass
        elif iteration > 0:
            param_change = np.max(np.abs(m_new - m_a), axis=1)
            param_conv = param_change < 1e-3

            if iteration > 2:
                fe_near = (
                    (prev_best_fe > -np.inf)
                    & (np.abs(free_energy - prev_best_fe) < 1e-4 * np.abs(prev_best_fe))
                )
                fe_plateau_count[a_idx] = np.where(
                    fe_near, fe_plateau_count[a_idx] + 1, 0,
                )
            fe_conv = fe_plateau_count[a_idx] >= 2

            newly_conv = param_conv | fe_conv
            if np.any(newly_conv):
                conv_global = a_idx[newly_conv]
                converged[conv_global] = True
                active[conv_global] = False

        failed_now = np.array(
            [failure_reasons[index] is not None for index in a_idx], dtype=bool
        )
        converged[a_idx[failed_now]] = False
        active[a_idx[failed_now]] = False

        # Update state
        still_active = active[a_idx]
        if np.any(still_active):
            sa_local = np.where(still_active)[0]
            sa_global = a_idx[sa_local]
            m[sa_global] = m_new[sa_local]
            c[sa_global] = c_new
            s[sa_global] = s_new[sa_local]
            tau_exp[sa_global] = tau_new[sa_local]

        if iteration % 5 == 0:
            logger.info(
                "DSC batched VI iter %d: %d/%d active, %d at FE plateau",
                iteration, int(np.sum(active)), n_valid, int(np.sum(converged)),
            )

    # The fixed 10-iteration VI schedule makes the FE-plateau count
    # informational only. Final fit quality is gated downstream by R².
    logger.info(
        "DSC batched VI finished %d-iteration schedule (n=%d): "
        "%d at FE plateau, %d numerical failures",
        last_iteration + 1, n_voxels, int(np.sum(converged)),
        sum(reason is not None for reason in failure_reasons),
    )

    # --- Post-convergence: inverse-transform best params ---
    fitted_nat = np.zeros((n_valid, n_params), dtype=np.float64)
    for j in range(n_params):
        fitted_nat[:, j] = from_transformed_batch(
            best_m[:, j], transforms[j], upper_bounds[j],
        )

    # Clip to bounds
    for j in range(n_params):
        fitted_nat[:, j] = np.clip(fitted_nat[:, j], bounds_lo[j], bounds_hi[j])

    if model_adapter is not None and _spatial_mode:
        prediction = _evaluate(fitted_nat)
        jacobian = np.stack(
            _jacobian_columns(best_m, fitted_nat, prediction), axis=-1
        )
        for spatial_row in range(n_valid):
            effective_precision = update_prior_prec.copy()
            effective_information = update_prior_eta.copy()
            if spatial_prec_diag is not None:
                effective_precision += np.diag(spatial_prec_diag[valid_idx[spatial_row]])
            if spatial_eta is not None:
                effective_information += spatial_eta[valid_idx[spatial_row]]
            effective_mean = np.asarray(
                np.linalg.solve(effective_precision, effective_information), dtype=np.float64
            )
            row_slice = slice(spatial_row, spatial_row + 1)
            try:
                state = local_vi_state(
                    best_m[row_slice], jacobian[row_slice], y[row_slice] - prediction[row_slice],
                    effective_mean, effective_precision, w,
                )
            except LocalVIFactorError:
                failure_reasons[spatial_row] = "local_factor_failed"
                converged[spatial_row] = False
                logger.debug(
                    "DSC spatial VI local factors failed for fit row %d",
                    int(valid_idx[spatial_row]),
                )
                continue
            best_cov_diag[spatial_row] = np.diag(state.covariance[0])
            if best_cov_full is not None:
                best_cov_full[spatial_row] = state.covariance[0]
            best_c[spatial_row], best_s[spatial_row] = state.shape, state.scale[0]

    # --- Uncertainties via delta method ---
    out_ci_lower = {
        name: np.zeros(n_voxels, dtype=np.float64)
        for name in fitting_param_names
    }
    out_ci_upper = {
        name: np.zeros(n_voxels, dtype=np.float64)
        for name in fitting_param_names
    }
    for j, name in enumerate(fitting_param_names):
        std_t = np.sqrt(np.maximum(best_cov_diag[:, j], 0.0))
        jac = np.exp(log_jacobian_det(fitted_nat[:, j], transforms[j], upper_bounds[j]))
        out_uncertainties[name][valid_idx] = std_t * jac
        endpoint_a = from_transformed_batch(
            best_m[:, j] - 1.96 * std_t, transforms[j], upper_bounds[j]
        )
        endpoint_b = from_transformed_batch(
            best_m[:, j] + 1.96 * std_t, transforms[j], upper_bounds[j]
        )
        out_ci_lower[name][valid_idx] = np.clip(
            np.minimum(endpoint_a, endpoint_b), bounds_lo[j], bounds_hi[j]
        )
        out_ci_upper[name][valid_idx] = np.clip(
            np.maximum(endpoint_a, endpoint_b), bounds_lo[j], bounds_hi[j]
        )

    out_uncertainties["noise_std"][valid_idx] = 1.0 / np.sqrt(
        np.maximum(best_c * best_s, 1e-300)
    )

    # --- Final forward evaluation for R² ---
    final_pred = _evaluate(fitted_nat)
    r2_valid = _compute_r_squared(y, final_pred)

    # Scatter results
    for j, name in enumerate(fitting_param_names):
        out_params[name][valid_idx] = fitted_nat[:, j]
    out_r2[valid_idx] = r2_valid
    out_converged[valid_idx] = converged
    failed_valid = np.array([reason is not None for reason in failure_reasons], dtype=bool)
    failed_output = valid_idx[failed_valid]
    if len(failed_output) and not _spatial_mode:
        for name in fitting_param_names:
            out_params[name][failed_output] = np.nan
            out_uncertainties[name][failed_output] = np.nan
            out_ci_lower[name][failed_output] = np.nan
            out_ci_upper[name][failed_output] = np.nan
        out_uncertainties["noise_std"][failed_output] = np.nan
        out_r2[failed_output] = np.nan
        out_converged[failed_output] = False

    out_fitted = None
    if save_fitted_curves:
        out_fitted = np.zeros((n_voxels, n_obs), dtype=np.float64)
        out_fitted[valid_idx] = final_pred
        if not _spatial_mode:
            out_fitted[failed_output] = np.nan

    out_post_means = None
    out_post_cov = None
    if return_posteriors:
        out_post_means = np.zeros((n_voxels, n_params), dtype=np.float64)
        out_post_cov = np.zeros((n_voxels, n_params), dtype=np.float64)
        out_post_means[valid_idx] = best_m
        out_post_cov[valid_idx] = best_cov_diag
        if not _spatial_mode:
            out_post_means[failed_output] = np.nan
            out_post_cov[failed_output] = np.nan

    full_posterior = None
    if best_cov_full is not None:
        full_mean = np.full((n_voxels, n_params), np.nan)
        full_cov = np.full((n_voxels, n_params, n_params), np.nan)
        full_shape = np.full(n_voxels, np.nan)
        full_scale = np.full(n_voxels, np.nan)
        full_mean[valid_idx] = best_m
        full_cov[valid_idx] = best_cov_full
        full_shape[valid_idx] = best_c
        full_scale[valid_idx] = best_s
        if not _spatial_mode:
            full_mean[failed_output] = np.nan
            full_cov[failed_output] = np.nan
            full_shape[failed_output] = np.nan
            full_scale[failed_output] = np.nan
        full_posterior = DSCVIPosterior(full_mean, full_cov, full_shape, full_scale)

    out_failure_reasons: list[str | None] = [None] * n_voxels
    for local_index, output_index in enumerate(valid_idx):
        out_failure_reasons[int(output_index)] = failure_reasons[local_index]

    return DSCBatchedFitResult(
        params=out_params, r_squared=out_r2, converged=out_converged,
        fitted_curves=out_fitted, n_iterations=last_iteration + 1,
        model_name=model_type, uncertainties=out_uncertainties,
        credible_interval_lower=out_ci_lower,
        credible_interval_upper=out_ci_upper,
        posterior_means_transformed=out_post_means,
        posterior_cov_diag=out_post_cov,
        full_posterior=full_posterior,
        failure_reasons=tuple(out_failure_reasons),
    )
