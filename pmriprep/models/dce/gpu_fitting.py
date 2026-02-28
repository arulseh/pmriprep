"""GPU-accelerated batched Levenberg-Marquardt solver for DCE PK fitting.

This module provides a generic batched LM optimizer that uses GPU-accelerated
forward models to fit all voxels in parallel. It supports all four DCE PK
models (Tofts, Extended Tofts, Patlak, 2CXM) with proper constraint handling,
weighted least squares, and uncertainty estimation.

The key insight is that while each LM iteration requires solving a small
per-voxel normal equation (2-4 unknowns), the forward model evaluations
(the expensive part) can be batched across all voxels on the GPU.

References:
    Marquardt DW. An algorithm for least-squares estimation of nonlinear
    parameters. J Soc Indust Appl Math. 1963;11(2):431-441.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from pmriprep.models.dce.gpu_convolution import evaluate_model_gpu
from pmriprep.utils.dce_constants import (
    EPSILON_VE,
    EPSILON_VP,
    FP_BOUNDS,
    KTRANS_BOUNDS,
    PS_BOUNDS,
    VE_BOUNDS,
    VP_BOUNDS,
)

logger = logging.getLogger("nipype.interface")

# Model parameter configurations
_MODEL_PARAMS: dict[str, list[str]] = {
    "tofts": ["ktrans", "ve"],
    "extended_tofts": ["ktrans", "ve", "vp"],
    "patlak": ["ktrans", "vp"],
    "2cxm": ["fp", "ps", "ve", "vp"],
}

_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "ktrans": (KTRANS_BOUNDS.min, KTRANS_BOUNDS.max),
    "ve": (VE_BOUNDS.min, VE_BOUNDS.max),
    "vp": (VP_BOUNDS.min, VP_BOUNDS.max),
    "fp": (FP_BOUNDS.min, FP_BOUNDS.max),
    "ps": (PS_BOUNDS.min, PS_BOUNDS.max),
}

# Parameter-specific finite difference floors (scale-aware)
_PARAM_FD_FLOORS: dict[str, float] = {
    "ktrans": 1e-5,
    "ve": 1e-4,
    "vp": 1e-4,
    "fp": 0.01,
    "ps": 0.01,
}


@dataclass
class GPUFitResult:
    """Result from GPU batched fitting.

    Attributes:
        params: Parameter arrays, param_name → (N,).
        r_squared: R-squared values, shape (N,).
        converged: Convergence flags, shape (N,).
        fitted_curves: Fitted concentration curves, shape (N, T) or None.
        n_iterations: Number of LM iterations performed.
        model_name: Name of the fitted model.
        uncertainties: Parameter uncertainties, param_name → (N,).
            NaN for non-converged voxels.
    """

    params: dict[str, npt.NDArray[np.floating]]
    r_squared: npt.NDArray[np.floating]
    converged: npt.NDArray[np.bool_]
    fitted_curves: Optional[npt.NDArray[np.floating]]
    n_iterations: int
    model_name: str
    uncertainties: dict[str, npt.NDArray[np.floating]] = field(default_factory=dict)


def _compute_r_squared(
    observed: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute R-squared for each voxel.

    Args:
        observed: Observed curves, shape (N, T).
        predicted: Predicted curves, shape (N, T).

    Returns:
        R-squared values, shape (N,).
    """
    ss_res = np.sum((observed - predicted) ** 2, axis=1)
    ss_tot = np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-30)
    return np.asarray(np.clip(r2, a_min=-1.0, a_max=1.0))


def _build_param_dict(
    model_name: str,
    params_matrix: npt.NDArray[np.floating],
    param_names: list[str],
) -> dict[str, npt.NDArray[np.floating]]:
    """Convert (N, P) params matrix to a dict for evaluate_model_gpu.

    Args:
        model_name: Model name.
        params_matrix: Parameter matrix, shape (N, P).
        param_names: Parameter names for this model.

    Returns:
        Dict mapping parameter names to (N,) arrays.
    """
    return {name: params_matrix[:, j] for j, name in enumerate(param_names)}


def _apply_volume_constraint_batched(
    params: npt.NDArray[np.floating],
    param_names: list[str],
) -> npt.NDArray[np.floating]:
    """Apply ve + vp <= 1.0 constraint by proportional scaling.

    Only applies to models that have both ve and vp.

    Args:
        params: Parameter matrix, shape (N, P).
        param_names: Parameter names.

    Returns:
        Params with volume constraint applied.
    """
    if "ve" not in param_names or "vp" not in param_names:
        return params

    ve_idx = param_names.index("ve")
    vp_idx = param_names.index("vp")

    total = params[:, ve_idx] + params[:, vp_idx]
    violating = total > 1.0

    if np.any(violating):
        params = params.copy()
        scale = 1.0 / np.maximum(total[violating], 1e-30)
        params[violating, ve_idx] *= scale
        params[violating, vp_idx] *= scale

    return params


def batched_patlak_lstsq(
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    fixed_vp: Optional[npt.NDArray[np.floating]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    save_fitted_curves: bool = False,
    compute_uncertainties: bool = False,
) -> GPUFitResult:
    """Fit Patlak model using linear least squares.

    Uses np.linalg.lstsq (QR-based) for numerical stability.

    Args:
        tissue_curves: Observed tissue curves, shape (N, T).
        aif: Arterial input function, shape (T,).
        time_points: Time points in minutes, shape (T,).
        fixed_vp: Fixed vp values, shape (N,). If provided, solves
            1-parameter (Ktrans only) via dot product.
        weights: Per-frame motion weights, shape (T,).
        save_fitted_curves: Whether to store fitted curves in result.
        compute_uncertainties: Whether to compute parameter uncertainties.

    Returns:
        GPUFitResult with fitted Patlak parameters.
    """
    tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)

    n_voxels, n_timepoints = tissue_curves.shape

    # Compute AIF integral (cumulative trapezoidal)
    dts = np.diff(time_points)
    aif_integral = np.zeros_like(aif)
    aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dts)

    # Apply weights if provided
    w = np.ones(n_timepoints) if weights is None else np.sqrt(np.maximum(weights, 1e-10))

    if fixed_vp is not None:
        # 1-parameter solve: subtract vp*Cp, solve for Ktrans
        residual = tissue_curves - fixed_vp[:, None] * aif[None, :]
        w_aif_int = w * aif_integral
        denom = np.sum(w_aif_int ** 2)
        ktrans = np.sum(residual * (w ** 2)[None, :] * aif_integral[None, :], axis=1)
        ktrans /= max(denom, 1e-30)
        ktrans = np.clip(ktrans, *_PARAM_BOUNDS["ktrans"])
        vp_out = fixed_vp.copy()
        n_free = 1
    else:
        # 2-parameter solve: [aif_integral, aif] → [Ktrans, vp]
        A = np.stack([w * aif_integral, w * aif], axis=1)  # (T, 2)
        Y = (tissue_curves * w[None, :]).T  # (T, N)
        params_2xN, _, _, _ = np.linalg.lstsq(A, Y, rcond=1e-10)  # (2, N)
        ktrans = np.clip(params_2xN[0], *_PARAM_BOUNDS["ktrans"])
        vp_out = np.clip(params_2xN[1], *_PARAM_BOUNDS["vp"])
        n_free = 2

    # Compute fitted curves and R²
    fitted = vp_out[:, None] * aif[None, :] + ktrans[:, None] * aif_integral[None, :]
    r_squared = _compute_r_squared(tissue_curves, fitted)
    converged = np.ones(n_voxels, dtype=bool)

    # Uncertainties
    uncertainties: dict[str, npt.NDArray[np.floating]] = {}
    if compute_uncertainties:
        residuals = tissue_curves - fitted
        sigma2 = np.sum((w[None, :] * residuals) ** 2, axis=1) / max(n_timepoints - n_free, 1)
        if fixed_vp is not None:
            denom_val = max(np.sum((w * aif_integral) ** 2), 1e-30)
            ktrans_var = sigma2 / denom_val
            uncertainties["ktrans"] = np.sqrt(np.maximum(ktrans_var, 0.0))
            uncertainties["vp"] = np.zeros(n_voxels)
        else:
            A_full = np.stack([w * aif_integral, w * aif], axis=1)
            ATA = A_full.T @ A_full
            try:
                ATA_inv = np.linalg.inv(ATA)
                uncertainties["ktrans"] = np.sqrt(np.maximum(sigma2 * ATA_inv[0, 0], 0.0))
                uncertainties["vp"] = np.sqrt(np.maximum(sigma2 * ATA_inv[1, 1], 0.0))
            except np.linalg.LinAlgError:
                uncertainties["ktrans"] = np.full(n_voxels, np.nan)
                uncertainties["vp"] = np.full(n_voxels, np.nan)

    return GPUFitResult(
        params={"ktrans": ktrans, "vp": vp_out},
        r_squared=r_squared,
        converged=converged,
        fitted_curves=fitted if save_fitted_curves else None,
        n_iterations=1,
        model_name="patlak",
        uncertainties=uncertainties,
    )


def _batched_initial_guess(
    model_name: str,
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    param_names: list[str],
) -> npt.NDArray[np.floating]:
    """Generate vectorized initial parameter guesses.

    Uses data-driven heuristics specific to each model type.

    Args:
        model_name: Model name.
        tissue_curves: Observed curves, shape (N, T).
        aif: AIF, shape (T,).
        time_points: Time points, shape (T,).
        param_names: Parameter names for this model.

    Returns:
        Initial guess matrix, shape (N, P).
    """
    n_voxels = tissue_curves.shape[0]
    n_params = len(param_names)
    n_timepoints = len(aif)

    # Compute shared quantities
    dts = np.diff(time_points)
    aif_integral = np.zeros_like(aif)
    aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dts)

    aif_peak = np.max(aif)
    aif_peak_idx = np.argmax(aif)

    # Patlak regression for Ktrans estimate
    aif_int_sq = np.sum(aif_integral ** 2)
    if aif_int_sq > 1e-30:
        tissue_dot_int = np.sum(tissue_curves * aif_integral[None, :], axis=1)
        ktrans_init = np.clip(tissue_dot_int / aif_int_sq, 0.001, 2.0)
    else:
        ktrans_init = np.full(n_voxels, 0.05)

    # vp from peak ratio
    if aif_peak > 1e-10:
        peak_start = max(0, int(aif_peak_idx) - 2)
        peak_end = min(n_timepoints, int(aif_peak_idx) + 3)
        tissue_peaks = np.max(tissue_curves[:, peak_start:peak_end], axis=1)
        vp_init = np.clip(tissue_peaks / aif_peak, 0.005, 0.25)
    else:
        vp_init = np.full(n_voxels, 0.05)

    # ve from late-phase ratio
    late_start = min(n_timepoints - 1, max(int(aif_peak_idx) + 10, int(0.7 * n_timepoints)))
    if late_start < n_timepoints - 1:
        late_tissue = np.mean(tissue_curves[:, late_start:], axis=1)
        late_aif = max(np.mean(aif[late_start:]), 1e-10)
        ve_init = np.clip(late_tissue / late_aif, 0.05, 0.7)
    else:
        ve_init = np.full(n_voxels, 0.2)

    # Build param matrix
    params = np.zeros((n_voxels, n_params), dtype=np.float64)
    for j, name in enumerate(param_names):
        if name == "ktrans":
            params[:, j] = ktrans_init
        elif name == "ve":
            params[:, j] = ve_init
        elif name == "vp":
            params[:, j] = vp_init
        elif name == "fp":
            # 2CXM: Fp_init = Ktrans * 120 (matching CPU two_compartment_exchange.py)
            params[:, j] = np.clip(ktrans_init * 120.0, 1.0, 150.0)
        elif name == "ps":
            # 2CXM: PS_init = Ktrans * 60
            params[:, j] = np.clip(ktrans_init * 60.0, 0.5, 80.0)

    return params


def batched_lm_fit(
    model_name: str,
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    fixed_vp: Optional[npt.NDArray[np.floating]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    max_iter: int = 100,
    ftol: float = 1e-6,
    xtol: float = 1e-5,
    save_fitted_curves: bool = False,
    compute_uncertainties: bool = False,
) -> GPUFitResult:
    """Batched Levenberg-Marquardt fitting for any DCE PK model.

    Fits all voxels simultaneously using GPU-accelerated forward model
    evaluations. The LM normal equations are solved on CPU (small per-voxel
    systems), while the expensive forward model calls are batched on GPU.

    Args:
        model_name: Model to fit ('tofts', 'extended_tofts', 'patlak', '2cxm').
        tissue_curves: Observed tissue curves, shape (N, T).
        aif: Arterial input function, shape (T,).
        time_points: Time points in minutes, shape (T,).
        fixed_vp: Fixed vp values for constrained fitting, shape (N,).
            For extended_tofts/patlak: pre-subtract vp*Cp.
            For 2cxm: pass as constant (vp enters eigenvalues nonlinearly).
        weights: Per-frame motion weights, shape (T,).
        max_iter: Maximum LM iterations. Default 100.
        ftol: Relative cost function convergence tolerance. Default 1e-6.
        xtol: Relative parameter convergence tolerance. Default 1e-5.
        save_fitted_curves: Whether to store final fitted curves.
        compute_uncertainties: Whether to compute parameter uncertainties
            from the final Jacobian.

    Returns:
        GPUFitResult with fitted parameters, R², and convergence flags.
    """
    tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)

    n_voxels, n_timepoints = tissue_curves.shape

    # Determine free parameters based on model and constraints
    all_param_names = list(_MODEL_PARAMS[model_name])

    # Handle vp constraint
    vp_subtracted = False
    if fixed_vp is not None:
        if model_name in ("extended_tofts", "patlak"):
            tissue_curves = tissue_curves - fixed_vp[:, None] * aif[None, :]
            all_param_names = [p for p in all_param_names if p != "vp"]
            vp_subtracted = True
        elif model_name == "2cxm":
            all_param_names = [p for p in all_param_names if p != "vp"]

    free_param_names = all_param_names
    n_free = len(free_param_names)

    # Get bounds for free parameters
    bounds_lo = np.array([_PARAM_BOUNDS[p][0] for p in free_param_names])
    bounds_hi = np.array([_PARAM_BOUNDS[p][1] for p in free_param_names])

    # Initialize parameters from original (un-subtracted) tissue curves
    init_curves = tissue_curves
    if vp_subtracted and fixed_vp is not None:
        init_curves = tissue_curves + fixed_vp[:, None] * aif[None, :]
    params = _batched_initial_guess(
        model_name, init_curves, aif, time_points, free_param_names,
    )

    for j in range(n_free):
        params[:, j] = np.clip(params[:, j], bounds_lo[j], bounds_hi[j])

    # Weight array: raw weights (not sqrt). LM objective minimizes sum(w_i * r_i^2).
    # Patlak takes sqrt(w) before lstsq, which after squaring gives the same WLS cost.
    w = np.ones(n_timepoints) if weights is None else np.asarray(weights, dtype=np.float64)

    # LM state
    lam = np.full(n_voxels, 1e-3)
    active = np.ones(n_voxels, dtype=bool)
    best_cost = np.full(n_voxels, np.inf)
    converged_flags = np.zeros(n_voxels, dtype=bool)

    # FD floors for each free parameter
    fd_floors = np.array([_PARAM_FD_FLOORS.get(p, 1e-6) for p in free_param_names])

    # Minimum iterations before convergence checks
    _MIN_ITER = 3

    def _build_full_params_idx(
        free_params: npt.NDArray[np.floating],
        global_indices: npt.NDArray[np.intp],
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Build full parameter dict using global indices for fixed_vp."""
        full_names = _MODEL_PARAMS[model_name]
        result: dict[str, npt.NDArray[np.floating]] = {}
        free_idx = 0
        for name in full_names:
            if name in free_param_names:
                result[name] = free_params[:, free_idx]
                free_idx += 1
            elif name == "vp" and fixed_vp is not None:
                result[name] = fixed_vp[global_indices]
            else:
                raise ValueError(f"Missing parameter '{name}' for model '{model_name}'")
        return result

    def _evaluate_idx(
        free_params: npt.NDArray[np.floating],
        global_indices: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating]:
        """Evaluate forward model, using global indices for fixed_vp lookup."""
        param_dict = _build_full_params_idx(free_params, global_indices)
        predicted = evaluate_model_gpu(
            model_name, param_dict, aif, time_points, use_gpu=False,
        ).tissue_curves
        if vp_subtracted and fixed_vp is not None:
            predicted = predicted - fixed_vp[global_indices, None] * aif[None, :]
        return predicted

    target = tissue_curves
    last_iteration = 0

    for iteration in range(max_iter):
        last_iteration = iteration
        n_active = int(np.sum(active))
        if n_active == 0:
            break

        active_idx = np.where(active)[0]
        p_active = params[active_idx]
        t_active = target[active_idx]

        # Forward evaluation
        predicted = _evaluate_idx(p_active, active_idx)
        residuals = t_active - predicted
        cost = 0.5 * np.sum(w[None, :] * residuals ** 2, axis=1)

        # Update best cost on first iteration
        if iteration == 0:
            best_cost[active_idx] = cost

        # Jacobian via central differences
        J_columns = []
        for j in range(n_free):
            delta_j = np.maximum(np.abs(p_active[:, j]) * 1e-4, fd_floors[j])

            p_plus = p_active.copy()
            p_plus[:, j] += delta_j
            p_plus[:, j] = np.clip(p_plus[:, j], bounds_lo[j], bounds_hi[j])

            p_minus = p_active.copy()
            p_minus[:, j] -= delta_j
            p_minus[:, j] = np.clip(p_minus[:, j], bounds_lo[j], bounds_hi[j])

            actual_delta = p_plus[:, j] - p_minus[:, j]
            actual_delta = np.maximum(actual_delta, fd_floors[j])

            pred_plus = _evaluate_idx(p_plus, active_idx)
            pred_minus = _evaluate_idx(p_minus, active_idx)

            J_columns.append((pred_plus - pred_minus) / actual_delta[:, None])

        # Build normal equations
        JtJ = np.zeros((n_active, n_free, n_free))
        Jtr = np.zeros((n_active, n_free))

        for j in range(n_free):
            wJ_j = w[None, :] * J_columns[j]
            Jtr[:, j] = np.sum(wJ_j * residuals, axis=1)
            for k in range(j, n_free):
                val = np.sum(wJ_j * J_columns[k], axis=1)
                JtJ[:, j, k] = val
                JtJ[:, k, j] = val

        # Damping
        lam_active = lam[active_idx]
        for j in range(n_free):
            diag_val = np.maximum(JtJ[:, j, j].copy(), 1e-12)
            JtJ[:, j, j] += lam_active * diag_val

        # Solve normal equations (batched, with condition-number pre-filter)
        dp = np.zeros((n_active, n_free))
        conds = np.linalg.cond(JtJ)
        well_conditioned = conds < 1e12
        if np.any(well_conditioned):
            good = np.where(well_conditioned)[0]
            dp[good] = np.linalg.solve(
                JtJ[good], Jtr[good, :, np.newaxis],
            )[:, :, 0]
        # Poorly conditioned voxels: use lstsq fallback (more robust)
        ill_conditioned = ~well_conditioned
        if np.any(ill_conditioned):
            bad = np.where(ill_conditioned)[0]
            for bi in bad:
                try:
                    dp[bi] = np.linalg.lstsq(JtJ[bi], Jtr[bi], rcond=1e-8)[0]
                except np.linalg.LinAlgError:
                    pass  # keep dp=0

        # Trial step
        p_trial = p_active + dp
        for j in range(n_free):
            p_trial[:, j] = np.clip(p_trial[:, j], bounds_lo[j], bounds_hi[j])

        p_trial = _apply_volume_constraint_batched(p_trial, free_param_names)

        # Evaluate trial
        pred_trial = _evaluate_idx(p_trial, active_idx)
        residuals_trial = t_active - pred_trial
        cost_trial = 0.5 * np.sum(w[None, :] * residuals_trial ** 2, axis=1)

        # Accept/reject
        accepted = cost_trial < cost

        # Convergence checks BEFORE state update (need pre-step costs)
        if iteration >= _MIN_ITER and np.any(accepted):
            acc_local = np.where(accepted)[0]
            acc_global = active_idx[acc_local]

            # ftol: relative cost improvement (cost is pre-step, cost_trial is post)
            old_cost_acc = cost[acc_local]
            new_cost_acc = cost_trial[acc_local]
            cost_change = old_cost_acc - new_cost_acc  # positive since accepted
            ftol_check = (cost_change / np.maximum(old_cost_acc, 1e-30)) < ftol

            # xtol: relative parameter step size
            dp_acc = dp[acc_local]
            p_acc = p_active[acc_local]
            dp_norm = np.linalg.norm(dp_acc, axis=1)
            p_norm = np.linalg.norm(p_acc, axis=1)
            xtol_check = (dp_norm / (p_norm + 1e-12)) < xtol

            acc_converged = ftol_check | xtol_check

            conv_global_idx = acc_global[acc_converged]
            converged_flags[conv_global_idx] = True
            active[conv_global_idx] = False

        # Update params and lambda AFTER convergence check
        params[active_idx[accepted]] = p_trial[accepted]
        lam[active_idx[accepted]] = np.maximum(lam[active_idx[accepted]] / 10.0, 1e-10)
        lam[active_idx[~accepted]] = np.minimum(lam[active_idx[~accepted]] * 10.0, 1e10)
        best_cost[active_idx[accepted]] = cost_trial[accepted]

        if iteration % 20 == 0:
            logger.info(
                "GPU LM iter %d: %d/%d active, %d converged",
                iteration, int(np.sum(active)), n_voxels, int(np.sum(converged_flags)),
            )

    logger.info(
        "GPU LM finished after %d iters: %d/%d converged",
        last_iteration + 1, int(np.sum(converged_flags)), n_voxels,
    )

    # Final evaluation for all voxels
    all_param_dict = _build_full_params_idx(params, np.arange(n_voxels))
    all_predicted = evaluate_model_gpu(
        model_name, all_param_dict, aif, time_points, use_gpu=False,
    ).tissue_curves

    if vp_subtracted and fixed_vp is not None:
        original_tissue = target + fixed_vp[:, None] * aif[None, :]
        final_r2 = _compute_r_squared(original_tissue, all_predicted)
        final_fitted = all_predicted if save_fitted_curves else None
    else:
        final_r2 = _compute_r_squared(tissue_curves, all_predicted)
        final_fitted = all_predicted if save_fitted_curves else None

    # Build output params dict
    output_params: dict[str, npt.NDArray[np.floating]] = {}
    full_names = _MODEL_PARAMS[model_name]
    free_idx = 0
    for name in full_names:
        if name in free_param_names:
            output_params[name] = params[:, free_idx]
            free_idx += 1
        elif name == "vp" and fixed_vp is not None:
            output_params[name] = fixed_vp.copy()

    # Uncertainty estimation for converged voxels
    uncertainties: dict[str, npt.NDArray[np.floating]] = {}
    if compute_uncertainties:
        for name in full_names:
            uncertainties[name] = np.full(n_voxels, np.nan)

        conv_idx = np.where(converged_flags)[0]
        if len(conv_idx) > 0:
            p_conv = params[conv_idx]
            t_conv = target[conv_idx]

            # Jacobian at final params
            J_cols_final = []
            for j in range(n_free):
                delta_j = np.maximum(np.abs(p_conv[:, j]) * 1e-4, fd_floors[j])

                pp = p_conv.copy()
                pp[:, j] += delta_j
                pp[:, j] = np.clip(pp[:, j], bounds_lo[j], bounds_hi[j])

                pm = p_conv.copy()
                pm[:, j] -= delta_j
                pm[:, j] = np.clip(pm[:, j], bounds_lo[j], bounds_hi[j])

                actual_d = np.maximum(pp[:, j] - pm[:, j], fd_floors[j])

                pred_p = _evaluate_idx(pp, conv_idx)
                pred_m = _evaluate_idx(pm, conv_idx)
                J_cols_final.append((pred_p - pred_m) / actual_d[:, None])

            n_conv = len(conv_idx)
            JtJ_final = np.zeros((n_conv, n_free, n_free))
            for j in range(n_free):
                wJ_j = w[None, :] * J_cols_final[j]
                for k in range(j, n_free):
                    val = np.sum(wJ_j * J_cols_final[k], axis=1)
                    JtJ_final[:, j, k] = val
                    JtJ_final[:, k, j] = val

            pred_conv = _evaluate_idx(p_conv, conv_idx)
            res_conv = t_conv - pred_conv
            sigma2 = np.sum(w[None, :] * res_conv ** 2, axis=1) / max(n_timepoints - n_free, 1)

            conds = np.linalg.cond(JtJ_final)
            well_conditioned = conds < 1e10
            if np.any(well_conditioned):
                good_idx = np.where(well_conditioned)[0]
                cov_batch = sigma2[good_idx, None, None] * np.linalg.inv(
                    JtJ_final[good_idx],
                )
                for j, name in enumerate(free_param_names):
                    uncertainties[name][conv_idx[good_idx]] = np.sqrt(
                        np.maximum(cov_batch[:, j, j], 0.0),
                    )

            if fixed_vp is not None and "vp" in uncertainties:
                uncertainties["vp"][:] = 0.0

    return GPUFitResult(
        params=output_params,
        r_squared=final_r2,
        converged=converged_flags,
        fitted_curves=final_fitted,
        n_iterations=last_iteration + 1 if n_voxels > 0 else 0,
        model_name=model_name,
        uncertainties=uncertainties,
    )
