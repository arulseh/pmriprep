"""Batched Levenberg-Marquardt solver for DCE PK fitting.

This module provides a generic batched LM optimizer that uses vectorized
forward models to fit all voxels in parallel. It supports all four DCE PK
models (Tofts, Extended Tofts, Patlak, 2CXM) with proper constraint handling,
weighted least squares, and uncertainty estimation.

The key insight is that while each LM iteration requires solving a small
per-voxel normal equation (2-4 unknowns), the forward model evaluations
(the expensive part) can be batched across all voxels.

References
----------
Marquardt DW. An algorithm for least-squares estimation of nonlinear
parameters. J Soc Indust Appl Math. 1963;11(2):431-441.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Optional

import numpy as np
import numpy.typing as npt

from pmriprep.models.dce.batched_models import evaluate_model
from pmriprep.utils.dce_constants import (
    FP_BOUNDS,
    KTRANS_BOUNDS,
    PS_BOUNDS,
    VE_BOUNDS,
    VP_BOUNDS,
    VP_VE_TOTAL_MAX,
)

logger = logging.getLogger("nipype.interface")

#: Numerical floor used as a divide/log guard in batched fits (R²
#: denominators, normal-equation denominators, ftol scaling). Set well
#: below any physical scale (~24 orders below a noise-floor RSS in mM²)
#: so it only fires on truly degenerate inputs. Not a physics constant.
_EPSILON_NUMERICAL: Final[float] = 1e-30

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
class FitResult:
    """Result from batched fitting.

    Attributes
    ----------
    params : dict of str to numpy.ndarray
        Parameter arrays, param_name → (N,).
    r_squared : numpy.ndarray
        R-squared values, shape (N,).
    converged : numpy.ndarray
        Convergence flags, shape (N,).
    fitted_curves : numpy.ndarray or None
        Fitted concentration curves, shape (N, T) or None.
    n_iterations : int
        Number of LM iterations performed.
    model_name : str
        Name of the fitted model.
    uncertainties : dict of str to numpy.ndarray
        Parameter uncertainties, param_name → (N,).
        NaN for non-converged voxels.
    vp_at_bound : numpy.ndarray or None
        Per-voxel indicator (0/1) of whether vp ended at a
        bound in the active-set SOFT NLLS path. None for HARD / SKIP /
        unconstrained / Bayesian fits. Shape (N,).
    """

    params: dict[str, npt.NDArray[np.floating]]
    r_squared: npt.NDArray[np.floating]
    converged: npt.NDArray[np.bool_]
    fitted_curves: Optional[npt.NDArray[np.floating]]
    n_iterations: int
    model_name: str
    uncertainties: dict[str, npt.NDArray[np.floating]] = field(default_factory=dict)
    vp_at_bound: Optional[npt.NDArray[np.floating]] = None


def concat_fit_results(
    results: list[FitResult],
    n_total: int,
) -> FitResult:
    """Concatenate per-chunk FitResults along the voxel axis.

    Used by the chunked-multiproc NLLS fitting path: workers return a
    FitResult for their slice of voxels, and the parent stitches them
    back into one whole-volume FitResult preserving voxel order.

    Invariants (fail-fast via ``assert``):
    - All chunks share the same ``model_name``.
    - ``fitted_curves`` is either None on every chunk or non-None on
      every chunk (mixed → AssertionError, caller bug).
    - ``uncertainties`` key sets match across chunks.
    - The concatenated length equals ``n_total``.

    ``n_iterations`` is reported as the max across chunks
    (representative — logging only; no downstream decision depends on it).

    Parameters
    ----------
    results : list of FitResult
        Non-empty list of per-chunk FitResults in voxel order.
    n_total : int
        Expected total voxel count across all chunks
        (sanity-checked against the concatenated length).

    Returns
    -------
    FitResult
        A single FitResult spanning all n_total voxels.
    """
    assert len(results) > 0, "concat_fit_results requires at least one chunk"
    model_name = results[0].model_name
    first_params_keys = set(results[0].params.keys())
    first_unc_keys = set(results[0].uncertainties.keys())
    first_has_curves = results[0].fitted_curves is not None
    for r in results[1:]:
        assert r.model_name == model_name, (
            f"chunk model_name mismatch: {r.model_name} vs {model_name}"
        )
        assert set(r.params.keys()) == first_params_keys, (
            "chunk params key set mismatch"
        )
        assert set(r.uncertainties.keys()) == first_unc_keys, (
            "chunk uncertainties key set mismatch"
        )
        assert (r.fitted_curves is not None) == first_has_curves, (
            "fitted_curves must be either None on all chunks or non-None on all"
        )

    params = {
        k: np.concatenate([r.params[k] for r in results]) for k in first_params_keys
    }
    r_squared = np.concatenate([r.r_squared for r in results])
    converged = np.concatenate([r.converged for r in results])
    fitted_curves: Optional[npt.NDArray[np.floating]]
    if first_has_curves:
        fitted_curves = np.concatenate(
            [r.fitted_curves for r in results], axis=0  # type: ignore[misc]
        )
    else:
        fitted_curves = None
    uncertainties = {
        k: np.concatenate([r.uncertainties[k] for r in results])
        for k in first_unc_keys
    }
    n_iterations = max(r.n_iterations for r in results)

    # vp_at_bound: either None on every chunk or non-None on every chunk
    first_has_atbound = results[0].vp_at_bound is not None
    for r in results[1:]:
        assert (r.vp_at_bound is not None) == first_has_atbound, (
            "vp_at_bound must be either None on all chunks or non-None on all"
        )
    vp_at_bound: Optional[npt.NDArray[np.floating]]
    if first_has_atbound:
        vp_at_bound = np.concatenate(
            [r.vp_at_bound for r in results]  # type: ignore[misc]
        )
    else:
        vp_at_bound = None

    assert r_squared.shape[0] == n_total, (
        f"concatenated length {r_squared.shape[0]} != expected {n_total}"
    )

    return FitResult(
        params=params,
        r_squared=r_squared,
        converged=converged,
        fitted_curves=fitted_curves,
        n_iterations=n_iterations,
        model_name=model_name,
        uncertainties=uncertainties,
        vp_at_bound=vp_at_bound,
    )


def _compute_r_squared(
    observed: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute R-squared for each voxel.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed curves, shape (N, T).
    predicted : numpy.ndarray
        Predicted curves, shape (N, T).

    Returns
    -------
    numpy.ndarray
        R-squared values, shape (N,).
    """
    ss_res = np.sum((observed - predicted) ** 2, axis=1)
    ss_tot = np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, _EPSILON_NUMERICAL)
    return np.asarray(np.clip(r2, a_min=-1.0, a_max=1.0))


def _build_param_dict(
    model_name: str,
    params_matrix: npt.NDArray[np.floating],
    param_names: list[str],
) -> dict[str, npt.NDArray[np.floating]]:
    """Convert (N, P) params matrix to a dict for evaluate_model.

    Parameters
    ----------
    model_name : str
        Model name.
    params_matrix : numpy.ndarray
        Parameter matrix, shape (N, P).
    param_names : list of str
        Parameter names for this model.

    Returns
    -------
    dict
        Dict mapping parameter names to (N,) arrays.
    """
    return {name: params_matrix[:, j] for j, name in enumerate(param_names)}


def _apply_volume_constraint_batched(
    params: npt.NDArray[np.float64],
    param_names: list[str],
) -> npt.NDArray[np.float64]:
    """Apply ve + vp <= 1.0 constraint by proportional scaling.

    Only applies to models that have both ve and vp.

    Parameters
    ----------
    params : numpy.ndarray
        Parameter matrix, shape (N, P).
    param_names : list of str
        Parameter names.

    Returns
    -------
    numpy.ndarray
        Params with volume constraint applied.
    """
    if "ve" not in param_names or "vp" not in param_names:
        return params

    ve_idx = param_names.index("ve")
    vp_idx = param_names.index("vp")

    total = params[:, ve_idx] + params[:, vp_idx]
    violating = total > VP_VE_TOTAL_MAX

    if np.any(violating):
        params = params.copy()
        scale = VP_VE_TOTAL_MAX / np.maximum(total[violating], _EPSILON_NUMERICAL)
        params[violating, ve_idx] *= scale
        params[violating, vp_idx] *= scale

    return params


def batched_patlak_lstsq(
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    fixed_vp: Optional[npt.NDArray[np.floating]] = None,
    vp_uncertainty: Optional[float] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    save_fitted_curves: bool = False,
    compute_uncertainties: bool = False,
) -> FitResult:
    """Fit Patlak model using linear least squares with optional vp constraint.

    Uses np.linalg.lstsq (QR-based) for numerical stability.

    Three modes determined by ``fixed_vp`` and ``vp_uncertainty``:

    - ``fixed_vp is None`` → unconstrained 2-param solve (Ktrans, vp).
    - ``fixed_vp`` provided, ``vp_uncertainty is None`` → HARD: vp fixed at
      ``fixed_vp``, 1-param Ktrans-only solve.
    - ``fixed_vp`` provided, ``vp_uncertainty > 0`` → SOFT box constraint:
      vp constrained to ``[fixed_vp·(1−σ), fixed_vp·(1+σ)]`` per voxel via
      active-set two-pass:
        Pass 1: unconstrained 2-param solve on all voxels (closed-form lstsq).
        Pass 2: for voxels where unconstrained vp* strictly violates a bound,
                clamp vp at the violated bound and re-solve for Ktrans only
                (closed-form 1-param). Merged per-voxel via ``np.where``.
      Voxels with ``fixed_vp == 0`` are treated as unconstrained (active-mask).

    ``vp_uncertainty == 0.0`` raises ``ValueError`` — pass ``None`` for HARD.

    Parameters
    ----------
    tissue_curves : numpy.ndarray
        Observed tissue curves, shape (N, T).
    aif : numpy.ndarray
        Arterial input function, shape (T,).
    time_points : numpy.ndarray
        Time points in minutes, shape (T,).
    fixed_vp : numpy.ndarray or None, optional
        Fixed vp values, shape (N,). See modes above.
    vp_uncertainty : float or None, optional
        Relative uncertainty σ for the SOFT box constraint.
        Must be positive or None.
    weights : numpy.ndarray or None, optional
        Per-frame motion weights, shape (T,).
    save_fitted_curves : bool, optional
        Whether to store fitted curves in result.
    compute_uncertainties : bool, optional
        Whether to compute parameter uncertainties.

    Returns
    -------
    FitResult
        FitResult with fitted Patlak parameters. ``vp_at_bound`` is set
        (per-voxel 0/1) only for the SOFT box mode; None otherwise.
    """
    if vp_uncertainty is not None and vp_uncertainty == 0.0:
        raise ValueError(
            "vp_uncertainty must be positive for SOFT constraint; pass None for HARD"
        )

    tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)

    n_voxels, n_timepoints = tissue_curves.shape

    # Compute AIF integral (cumulative trapezoidal)
    dts = np.diff(time_points)
    aif_integral = np.zeros_like(aif)
    aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dts)

    # Apply weights if provided
    w = np.ones(n_timepoints) if weights is None else np.sqrt(np.maximum(weights, 1e-10))

    vp_at_bound: Optional[npt.NDArray[np.floating]] = None

    soft_box = (
        fixed_vp is not None
        and vp_uncertainty is not None
        and vp_uncertainty > 0
    )

    if soft_box:
        # SOFT box constraint via active-set two-pass.
        assert fixed_vp is not None  # for type narrowing
        assert vp_uncertainty is not None

        # Pass 1: unconstrained 2-param solve on all voxels.
        A = np.stack([w * aif_integral, w * aif], axis=1)  # (T, 2)
        Y = (tissue_curves * w[None, :]).T  # (T, N)
        params_2xN, _, _, _ = np.linalg.lstsq(A, Y, rcond=1e-10)
        ktrans_unc = params_2xN[0]
        vp_unc = params_2xN[1]

        # Per-voxel bounds. fixed_vp == 0 → inactive (acts unconstrained).
        active = fixed_vp > 0
        vp_lo = np.where(
            active,
            fixed_vp * (1.0 - vp_uncertainty),
            _PARAM_BOUNDS["vp"][0],
        )
        vp_hi = np.where(
            active,
            fixed_vp * (1.0 + vp_uncertainty),
            _PARAM_BOUNDS["vp"][1],
        )

        # Strict-inequality violation detection (grazing the bound = interior).
        violates_lo = active & (vp_unc < vp_lo)
        violates_hi = active & (vp_unc > vp_hi)
        violates = violates_lo | violates_hi

        # Pass 2: for violators, clamp vp at violated bound and 1-param
        # Ktrans-only solve. Computed for all voxels then merged by np.where
        # (closed-form, cheap enough to do unconditionally).
        clamped_vp = np.where(
            violates_lo,
            vp_lo,
            np.where(violates_hi, vp_hi, vp_unc),
        )
        residual_clamped = tissue_curves - clamped_vp[:, None] * aif[None, :]
        denom = max(np.sum((w * aif_integral) ** 2), _EPSILON_NUMERICAL)
        ktrans_clamped = np.sum(
            residual_clamped * (w ** 2)[None, :] * aif_integral[None, :], axis=1
        )
        ktrans_clamped /= denom
        ktrans_clamped = np.clip(ktrans_clamped, *_PARAM_BOUNDS["ktrans"])

        ktrans = np.where(
            violates,
            ktrans_clamped,
            np.clip(ktrans_unc, *_PARAM_BOUNDS["ktrans"]),
        )
        vp_out = np.where(
            violates,
            clamped_vp,
            np.clip(vp_unc, *_PARAM_BOUNDS["vp"]),
        )
        vp_at_bound = violates.astype(np.float32)
        n_free = 2  # Pass-1 effective dof for interior voxels; pragmatic.
    elif fixed_vp is not None:
        # HARD: 1-parameter solve: subtract vp*Cp, solve for Ktrans.
        residual = tissue_curves - fixed_vp[:, None] * aif[None, :]
        w_aif_int = w * aif_integral
        denom = np.sum(w_aif_int ** 2)
        ktrans = np.sum(residual * (w ** 2)[None, :] * aif_integral[None, :], axis=1)
        ktrans /= max(denom, _EPSILON_NUMERICAL)
        ktrans = np.clip(ktrans, *_PARAM_BOUNDS["ktrans"])
        vp_out = fixed_vp.copy()
        n_free = 1
    else:
        # Unconstrained: 2-parameter solve [aif_integral, aif] → [Ktrans, vp].
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
        # For SOFT box and HARD (vp fixed or clamped at a bound), uncertainty
        # in vp is 0 for voxels at a bound and data-driven elsewhere; a tight
        # closed-form formula across the mixed population is not meaningful,
        # so report the Ktrans-only σ and zero vp uncertainty for HARD and
        # at-bound voxels. Interior SOFT voxels get the 2-param uncertainty.
        if soft_box and vp_at_bound is not None:
            A_full = np.stack([w * aif_integral, w * aif], axis=1)
            ATA = A_full.T @ A_full
            try:
                ATA_inv = np.linalg.inv(ATA)
                uncertainties["ktrans"] = np.sqrt(
                    np.maximum(sigma2 * ATA_inv[0, 0], 0.0)
                )
                unc_vp_interior = np.sqrt(
                    np.maximum(sigma2 * ATA_inv[1, 1], 0.0)
                )
                uncertainties["vp"] = np.where(
                    vp_at_bound > 0, 0.0, unc_vp_interior
                )
            except np.linalg.LinAlgError:
                uncertainties["ktrans"] = np.full(n_voxels, np.nan)
                uncertainties["vp"] = np.full(n_voxels, np.nan)
        elif fixed_vp is not None:
            denom_val = max(np.sum((w * aif_integral) ** 2), _EPSILON_NUMERICAL)
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

    return FitResult(
        params={"ktrans": ktrans, "vp": vp_out},
        r_squared=r_squared,
        converged=converged,
        fitted_curves=fitted if save_fitted_curves else None,
        n_iterations=1,
        model_name="patlak",
        uncertainties=uncertainties,
        vp_at_bound=vp_at_bound,
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

    Parameters
    ----------
    model_name : str
        Model name.
    tissue_curves : numpy.ndarray
        Observed curves, shape (N, T).
    aif : numpy.ndarray
        AIF, shape (T,).
    time_points : numpy.ndarray
        Time points, shape (T,).
    param_names : list of str
        Parameter names for this model.

    Returns
    -------
    numpy.ndarray
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
    if aif_int_sq > _EPSILON_NUMERICAL:
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


def _batched_lm_fit_soft_box(
    model_name: str,
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    fixed_vp: npt.NDArray[np.floating],
    vp_uncertainty: float,
    weights: Optional[npt.NDArray[np.floating]],
    max_iter: int,
    ftol: float,
    xtol: float,
    save_fitted_curves: bool,
    compute_uncertainties: bool,
) -> FitResult:
    """Active-set two-pass SOFT box constraint for batched LM.

    Pass 1: unconstrained batched LM (vp in free-param set). Extract
    unconstrained vp*.

    Pass 2: for voxels where vp* strictly violates the per-voxel bound
    ``[fixed_vp·(1−σ), fixed_vp·(1+σ)]``, clamp vp at the violated bound
    and re-run batched LM with vp fixed (reduced problem). Mirrors the KKT
    solution for bounded NLLS and the Patlak active-set in
    ``batched_patlak_lstsq``.

    Edge-case handling:
    - ``fixed_vp == 0`` per voxel → inactive (treated as unconstrained;
      Pass 2 never fires).
    - Pass-1 NaN vp → mark voxel ``converged=False``, keep Pass-1 params,
      skip Pass 2 for that voxel.
    - Pass-2 non-convergence → fall back to (vp clamped at bound, Pass-1
      Ktrans/ve), flag ``converged=False`` so BIC=inf downstream.
    - Strict inequalities (vp* < vp_lo or vp* > vp_hi) for Pass-2 trigger;
      grazing the bound (vp* == bound) counts as interior (matches scipy KKT).
    """
    n_voxels = tissue_curves.shape[0]

    # Pass 1: fully unconstrained (all params free, no fixed_vp).
    pass1 = batched_lm_fit(
        model_name, tissue_curves, aif, time_points,
        fixed_vp=None, vp_uncertainty=None,
        weights=weights, max_iter=max_iter, ftol=ftol, xtol=xtol,
        save_fitted_curves=save_fitted_curves,
        compute_uncertainties=compute_uncertainties,
    )

    vp_unc = pass1.params.get("vp")
    if vp_unc is None:
        # Model has no vp parameter — shouldn't reach here given the caller
        # gate (model_name ∈ {extended_tofts, patlak, 2cxm}), but degrade
        # gracefully: just return Pass 1.
        pass1.vp_at_bound = np.zeros(n_voxels, dtype=np.float32)
        return pass1

    # Per-voxel bounds. fixed_vp == 0 → inactive (acts unconstrained).
    active = fixed_vp > 0
    vp_lo = np.where(
        active, fixed_vp * (1.0 - vp_uncertainty), _PARAM_BOUNDS["vp"][0],
    )
    vp_hi = np.where(
        active, fixed_vp * (1.0 + vp_uncertainty), _PARAM_BOUNDS["vp"][1],
    )

    # Pass-1 NaN: NaN comparisons are False, so such voxels don't enter
    # `violates` naturally; handle explicitly by forcing converged=False.
    pass1_nan = ~np.isfinite(vp_unc)

    # Strict-inequality violation detection.
    violates_lo = active & ~pass1_nan & (vp_unc < vp_lo)
    violates_hi = active & ~pass1_nan & (vp_unc > vp_hi)
    violates = violates_lo | violates_hi

    clamped_vp = np.where(
        violates_lo, vp_lo, np.where(violates_hi, vp_hi, vp_unc),
    )

    # No violators: return Pass 1 with vp_at_bound all zeros, plus NaN fixup.
    if not np.any(violates):
        merged_converged = pass1.converged.copy()
        merged_converged[pass1_nan] = False
        pass1.converged = merged_converged
        pass1.vp_at_bound = np.zeros(n_voxels, dtype=np.float32)
        return pass1

    # Pass 2: re-fit violators with vp fixed at the clamped bound (HARD on subset).
    viol_idx = np.where(violates)[0]
    tc_viol = tissue_curves[viol_idx]
    fixed_vp_viol = clamped_vp[viol_idx]
    weights_viol = weights  # per-frame weights, shape (T,), independent of voxels

    pass2 = batched_lm_fit(
        model_name, tc_viol, aif, time_points,
        fixed_vp=fixed_vp_viol, vp_uncertainty=None,  # HARD on subset
        weights=weights_viol, max_iter=max_iter, ftol=ftol, xtol=xtol,
        save_fitted_curves=save_fitted_curves,
        compute_uncertainties=compute_uncertainties,
    )

    # Merge per-voxel: interior voxels keep Pass-1; violators take Pass-2.
    merged_params: dict[str, npt.NDArray[np.floating]] = {
        k: v.copy() for k, v in pass1.params.items()
    }
    for name, arr in pass2.params.items():
        if name in merged_params:
            merged_params[name][viol_idx] = arr
        else:
            merged_params[name] = np.zeros(n_voxels, dtype=arr.dtype)
            merged_params[name][viol_idx] = arr

    # Ensure vp at the violated bound for Pass-2 voxels (Pass-2 returns
    # fixed_vp_viol unchanged, but belt-and-suspenders in case the HARD path
    # ever modifies it — the clamped_vp[viol_idx] is the authoritative value).
    merged_params["vp"][viol_idx] = clamped_vp[viol_idx]

    merged_converged = pass1.converged.copy()
    merged_converged[viol_idx] = pass2.converged
    merged_converged[pass1_nan] = False

    # Pass-2 non-convergence fallback: keep vp at bound (already set), and
    # for Ktrans/ve use Pass-1 values (since Pass-2 didn't converge, its
    # Ktrans/ve are unreliable). Already have Pass-1 values in merged_params
    # prior to the pass2 overwrite loop — re-apply for non-converged Pass-2.
    pass2_not_conv_local = ~pass2.converged
    if np.any(pass2_not_conv_local):
        notconv_local_idx = np.where(pass2_not_conv_local)[0]
        notconv_global_idx = viol_idx[notconv_local_idx]
        for name in pass2.params:
            if name == "vp":
                continue  # vp stays at bound per plan
            if name in pass1.params:
                merged_params[name][notconv_global_idx] = (
                    pass1.params[name][notconv_global_idx]
                )

    merged_r2 = pass1.r_squared.copy()
    merged_r2[viol_idx] = pass2.r_squared

    merged_fitted: Optional[npt.NDArray[np.floating]] = None
    if save_fitted_curves:
        if pass1.fitted_curves is not None and pass2.fitted_curves is not None:
            merged_fitted = pass1.fitted_curves.copy()
            merged_fitted[viol_idx] = pass2.fitted_curves

    merged_unc: dict[str, npt.NDArray[np.floating]] = {}
    if compute_uncertainties:
        for name, arr in pass1.uncertainties.items():
            merged_unc[name] = arr.copy()
            if name in pass2.uncertainties:
                merged_unc[name][viol_idx] = pass2.uncertainties[name]

    return FitResult(
        params=merged_params,
        r_squared=merged_r2,
        converged=merged_converged,
        fitted_curves=merged_fitted,
        n_iterations=max(pass1.n_iterations, pass2.n_iterations),
        model_name=model_name,
        uncertainties=merged_unc,
        vp_at_bound=violates.astype(np.float32),
    )


def batched_lm_fit(
    model_name: str,
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    fixed_vp: Optional[npt.NDArray[np.floating]] = None,
    vp_uncertainty: Optional[float] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    max_iter: int = 100,
    ftol: float = 1e-6,
    xtol: float = 1e-5,
    save_fitted_curves: bool = False,
    compute_uncertainties: bool = False,
) -> FitResult:
    """Batched Levenberg-Marquardt fitting for any DCE PK model.

    Fits all voxels simultaneously using vectorized forward model
    evaluations. The LM normal equations are solved on CPU (small per-voxel
    systems), while the expensive forward model calls are batched.

    Constraint modes (determined by ``fixed_vp`` and ``vp_uncertainty``):

    - ``fixed_vp is None`` → unconstrained fit (all model params free).
    - ``fixed_vp`` provided, ``vp_uncertainty is None`` → HARD: vp fixed at
      ``fixed_vp``; for extended_tofts/patlak, pre-subtracts vp*Cp; for 2cxm,
      vp enters eigenvalues nonlinearly.
    - ``fixed_vp`` provided, ``vp_uncertainty > 0`` → SOFT box constraint
      ``vp ∈ [fixed_vp·(1−σ), fixed_vp·(1+σ)]`` per voxel, implemented as
      active-set two-pass (Pass 1: unconstrained; Pass 2: for violators,
      clamp vp at violated bound and re-fit reduced problem). Voxels with
      ``fixed_vp == 0`` are treated as unconstrained (active-mask).

    ``vp_uncertainty == 0.0`` raises ``ValueError`` — pass ``None`` for HARD.

    Parameters
    ----------
    model_name : str
        Model to fit ('tofts', 'extended_tofts', 'patlak', '2cxm').
    tissue_curves : numpy.ndarray
        Observed tissue curves, shape (N, T).
    aif : numpy.ndarray
        Arterial input function, shape (T,).
    time_points : numpy.ndarray
        Time points in minutes, shape (T,).
    fixed_vp : numpy.ndarray or None, optional
        Fixed vp values for constrained fitting, shape (N,).
        See modes above.
    vp_uncertainty : float or None, optional
        Relative uncertainty σ for the SOFT box constraint.
        Must be positive or None.
    weights : numpy.ndarray or None, optional
        Per-frame motion weights, shape (T,).
    max_iter : int, optional
        Maximum LM iterations. Default 100.
    ftol : float, optional
        Relative cost function convergence tolerance. Default 1e-6.
    xtol : float, optional
        Relative parameter convergence tolerance. Default 1e-5.
    save_fitted_curves : bool, optional
        Whether to store final fitted curves.
    compute_uncertainties : bool, optional
        Whether to compute parameter uncertainties
        from the final Jacobian.

    Returns
    -------
    FitResult
        FitResult with fitted parameters, R², and convergence flags. The
        ``vp_at_bound`` field is set (per-voxel 0/1) only for the SOFT box
        mode; None otherwise.
    """
    if vp_uncertainty is not None and vp_uncertainty == 0.0:
        raise ValueError(
            "vp_uncertainty must be positive for SOFT constraint; pass None for HARD"
        )

    # SOFT box constraint: active-set two-pass dispatch. Only meaningful for
    # models that have a vp parameter.
    if (
        fixed_vp is not None
        and vp_uncertainty is not None
        and vp_uncertainty > 0
        and model_name in ("extended_tofts", "patlak", "2cxm")
    ):
        return _batched_lm_fit_soft_box(
            model_name, tissue_curves, aif, time_points,
            fixed_vp, vp_uncertainty, weights,
            max_iter, ftol, xtol,
            save_fitted_curves, compute_uncertainties,
        )

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
        predicted = evaluate_model(
            model_name, param_dict, aif, time_points,
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
            ftol_check = (cost_change / np.maximum(old_cost_acc, _EPSILON_NUMERICAL)) < ftol

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
                "Batched LM iter %d: %d/%d active, %d converged",
                iteration, int(np.sum(active)), n_voxels, int(np.sum(converged_flags)),
            )

    logger.info(
        "Batched LM finished after %d iters: %d/%d converged",
        last_iteration + 1, int(np.sum(converged_flags)), n_voxels,
    )

    # Final evaluation for all voxels
    all_param_dict = _build_full_params_idx(params, np.arange(n_voxels))
    all_predicted = evaluate_model(
        model_name, all_param_dict, aif, time_points,
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

    return FitResult(
        params=output_params,
        r_squared=final_r2,
        converged=converged_flags,
        fitted_curves=final_fitted,
        n_iterations=last_iteration + 1 if n_voxels > 0 else 0,
        model_name=model_name,
        uncertainties=uncertainties,
    )
