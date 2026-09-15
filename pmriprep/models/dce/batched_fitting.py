"""Batched Levenberg-Marquardt solver for DCE PK fitting.

This module provides a generic batched LM optimizer that uses vectorized
forward models to fit all voxels in parallel. It supports all four DCE PK
models (Tofts, Extended Tofts, Patlak, 2CXM) with proper constraint handling,
weighted least squares, and uncertainty estimation.

Each LM iteration solves a small per-voxel normal equation (2-4 unknowns).
Forward-model evaluations are batched across all voxels.

All model parameters are estimated within their physical bounds. DSC-informed
vp priors are applied separately during Bayesian refinement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Final, Optional

import numpy as np
import numpy.typing as npt

from pmriprep.models.dce.batched_models import evaluate_model
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.two_compartment_exchange import (
    TwoCompartmentExchangeModel,
)
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
    """

    params: dict[str, npt.NDArray[np.floating]]
    r_squared: npt.NDArray[np.floating]
    converged: npt.NDArray[np.bool_]
    fitted_curves: Optional[npt.NDArray[np.floating]]
    n_iterations: int
    model_name: str
    uncertainties: dict[str, npt.NDArray[np.floating]] = field(default_factory=dict)


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
      every chunk; mixed values are invalid input.
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
    params: npt.NDArray[Any],
    param_names: list[str],
) -> npt.NDArray[np.float64]:
    """Apply the joint tissue-volume constraint.

    Free ``ve`` and ``vp`` are proportionally projected.

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
    params = np.asarray(params, dtype=np.float64)
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
    weights: Optional[npt.NDArray[np.floating]] = None,
    save_fitted_curves: bool = False,
    compute_uncertainties: bool = False,
) -> FitResult:
    """Fit Patlak using linear least squares within physical parameter bounds.

    Uses np.linalg.lstsq (QR-based) for numerical stability.

    Parameters
    ----------
    tissue_curves : numpy.ndarray
        Observed tissue curves, shape (N, T).
    aif : numpy.ndarray
        Arterial input function, shape (T,).
    time_points : numpy.ndarray
        Time points in minutes, shape (T,).
    weights : numpy.ndarray or None, optional
        Per-frame motion weights, shape (T,).
    save_fitted_curves : bool, optional
        Whether to store fitted curves in result.
    compute_uncertainties : bool, optional
        Whether to compute parameter uncertainties.

    Returns
    -------
    FitResult
        FitResult with fitted Patlak parameters.
    """
    tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)

    n_voxels, n_timepoints = tissue_curves.shape

    # Compute AIF integral (cumulative trapezoidal)
    dts = np.diff(time_points)
    aif_integral = np.zeros_like(aif)
    aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dts)

    # Apply weights if provided
    w = np.ones(n_timepoints) if weights is None else np.sqrt(np.maximum(weights, 1e-10))

    def _solve_bounded(
        vp_lower: npt.NDArray[np.floating],
        vp_upper: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Solve the two-parameter weighted box-constrained linear problem."""
        weighted_integral = w * aif_integral
        weighted_aif = w * aif
        weighted_tissue = tissue_curves * w[None, :]
        design = np.column_stack([weighted_integral, weighted_aif])
        unconstrained, _, _, _ = np.linalg.lstsq(
            design, weighted_tissue.T, rcond=1e-10
        )

        k_lo, k_hi = _PARAM_BOUNDS["ktrans"]
        candidates_k = [
            np.clip(unconstrained[0], k_lo, k_hi),
        ]
        candidates_vp = [
            np.clip(unconstrained[1], vp_lower, vp_upper),
        ]

        aif_denom = max(
            float(np.dot(weighted_aif, weighted_aif)),
            _EPSILON_NUMERICAL,
        )
        for k_boundary in (k_lo, k_hi):
            vp_candidate = np.sum(
                (
                    weighted_tissue
                    - k_boundary * weighted_integral[None, :]
                )
                * weighted_aif[None, :],
                axis=1,
            ) / aif_denom
            candidates_k.append(np.full(n_voxels, k_boundary))
            candidates_vp.append(
                np.clip(vp_candidate, vp_lower, vp_upper)
            )

        integral_denom = max(
            float(np.dot(weighted_integral, weighted_integral)),
            _EPSILON_NUMERICAL,
        )
        for vp_boundary in (vp_lower, vp_upper):
            k_candidate = np.sum(
                (
                    weighted_tissue
                    - vp_boundary[:, None] * weighted_aif[None, :]
                )
                * weighted_integral[None, :],
                axis=1,
            ) / integral_denom
            candidates_k.append(np.clip(k_candidate, k_lo, k_hi))
            candidates_vp.append(vp_boundary)

        k_stack = np.stack(candidates_k)
        vp_stack = np.stack(candidates_vp)
        tissue_integral = weighted_tissue @ weighted_integral
        tissue_aif = weighted_tissue @ weighted_aif
        tissue_squared = np.sum(weighted_tissue**2, axis=1)
        integral_squared = float(np.dot(weighted_integral, weighted_integral))
        aif_squared = float(np.dot(weighted_aif, weighted_aif))
        integral_aif = float(np.dot(weighted_integral, weighted_aif))
        costs = (
            tissue_squared[None, :]
            - 2.0 * k_stack * tissue_integral[None, :]
            - 2.0 * vp_stack * tissue_aif[None, :]
            + k_stack**2 * integral_squared
            + 2.0 * k_stack * vp_stack * integral_aif
            + vp_stack**2 * aif_squared
        )
        best = np.argmin(costs, axis=0)
        voxel_indices = np.arange(n_voxels)
        return (
            k_stack[best, voxel_indices],
            vp_stack[best, voxel_indices],
        )

    vp_lo = np.full(n_voxels, _PARAM_BOUNDS["vp"][0])
    vp_hi = np.full(n_voxels, _PARAM_BOUNDS["vp"][1])
    ktrans, vp_out = _solve_bounded(vp_lo, vp_hi)

    # Compute fitted curves and R²
    fitted = vp_out[:, None] * aif[None, :] + ktrans[:, None] * aif_integral[None, :]
    r_squared = _compute_r_squared(tissue_curves, fitted)
    converged = np.ones(n_voxels, dtype=bool)

    # Uncertainties
    uncertainties: dict[str, npt.NDArray[np.floating]] = {}
    if compute_uncertainties:
        residuals = tissue_curves - fitted
        ktrans_at_bound = (
            np.isclose(
                ktrans, _PARAM_BOUNDS["ktrans"][0], rtol=0.0, atol=1e-12
            )
            | np.isclose(
                ktrans, _PARAM_BOUNDS["ktrans"][1], rtol=0.0, atol=1e-12
            )
        )
        vp_is_at_bound = (
            np.isclose(
                vp_out, _PARAM_BOUNDS["vp"][0], rtol=0.0, atol=1e-12
            )
            | np.isclose(
                vp_out, _PARAM_BOUNDS["vp"][1], rtol=0.0, atol=1e-12
            )
        )
        free_counts = (
            (~ktrans_at_bound).astype(np.int8)
            + (~vp_is_at_bound).astype(np.int8)
        )
        sigma_denominator = np.maximum(n_timepoints - free_counts, 1)
        sigma2 = (
            np.sum((w[None, :] * residuals) ** 2, axis=1)
            / sigma_denominator
        )
        # A symmetric local standard error is undefined for an active-bound
        # parameter. Re-estimated free coefficients use their reduced
        # one-parameter covariance, while interior voxels use the full
        # two-parameter form.
        A_full = np.stack([w * aif_integral, w * aif], axis=1)
        ATA = A_full.T @ A_full
        try:
            ATA_inv = np.linalg.inv(ATA)
            ktrans_full = np.sqrt(
                np.maximum(sigma2 * ATA_inv[0, 0], 0.0)
            )
            vp_full = np.sqrt(
                np.maximum(sigma2 * ATA_inv[1, 1], 0.0)
            )
            ktrans_reduced = np.sqrt(
                np.maximum(
                    sigma2
                    / max(
                        float(np.dot(w * aif_integral, w * aif_integral)),
                        _EPSILON_NUMERICAL,
                    ),
                    0.0,
                )
            )
            vp_reduced = np.sqrt(
                np.maximum(
                    sigma2
                    / max(
                        float(np.dot(w * aif, w * aif)),
                        _EPSILON_NUMERICAL,
                    ),
                    0.0,
                )
            )
            uncertainties["ktrans"] = np.where(
                ktrans_at_bound,
                np.nan,
                np.where(vp_is_at_bound, ktrans_reduced, ktrans_full),
            )
            uncertainties["vp"] = np.where(
                vp_is_at_bound,
                np.nan,
                np.where(ktrans_at_bound, vp_reduced, vp_full),
            )
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

    # Match the scalar exchange-model initializers. A distant starting point
    # can lead the ill-conditioned objective to an incorrect boundary solution.
    if model_name in ("2cxm", "extended_tofts"):
        tissue_peak_indices = np.argmax(tissue_curves, axis=1)
        start = int(aif_peak_idx) + 1
        ends = np.minimum(
            np.maximum(tissue_peak_indices, start + 3),
            n_timepoints - 1,
        )
        ktrans_init = np.full(n_voxels, 0.1, dtype=np.float64)
        for end in np.unique(ends):
            if int(end) - start < 3:
                continue
            idx = np.arange(start, int(end) + 1)
            if float(np.max(np.abs(aif_integral[idx]))) < 1e-9:
                continue
            design = np.column_stack([aif_integral[idx], aif[idx]])
            rows = np.where(ends == end)[0]
            try:
                solution = np.linalg.lstsq(
                    design,
                    tissue_curves[rows][:, idx].T,
                    rcond=None,
                )[0]
            except np.linalg.LinAlgError:
                continue
            slopes = solution[0]
            finite = np.isfinite(slopes)
            ktrans_init[rows[finite]] = np.maximum(slopes[finite], 0.001)

        if aif_peak > 1e-6:
            vp_init = np.clip(
                np.max(tissue_curves, axis=1) / aif_peak,
                _PARAM_BOUNDS["vp"][0],
                _PARAM_BOUNDS["vp"][1],
            )
        else:
            vp_init = np.full(n_voxels, 0.02)

        late_start = min(int(aif_peak_idx) + 12, n_timepoints - 5)
        if late_start < n_timepoints - 3:
            mean_aif = float(np.mean(aif[late_start:]))
            if mean_aif > 0.05 * aif_peak:
                ve_init = np.clip(
                    np.mean(tissue_curves[:, late_start:], axis=1) / mean_aif,
                    VE_BOUNDS.min,
                    VE_BOUNDS.max,
                )
            else:
                ve_init = np.full(n_voxels, 0.2)
        else:
            ve_init = np.full(n_voxels, 0.2)

        initial = {
            "ktrans": np.clip(
                ktrans_init,
                _PARAM_BOUNDS["ktrans"][0] + 1e-6,
                _PARAM_BOUNDS["ktrans"][1],
            ),
            "fp": np.clip(
                ktrans_init * 120.0,
                _PARAM_BOUNDS["fp"][0] + 1e-3,
                _PARAM_BOUNDS["fp"][1],
            ),
            "ps": np.clip(
                ktrans_init * 60.0,
                _PARAM_BOUNDS["ps"][0] + 1e-3,
                _PARAM_BOUNDS["ps"][1],
            ),
            "ve": ve_init,
            "vp": vp_init,
        }
        return np.column_stack([initial[name] for name in param_names])

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


def _scaled_lm_step(
    gram: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    jacobian_columns: list[npt.NDArray[np.float64]],
    residuals: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    damping: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve damped LM steps in column-normalized parameter coordinates.

    Parameters
    ----------
    gram : numpy.ndarray
        Weighted Jacobian Gram matrices, shape (voxels, parameters, parameters).
    rhs : numpy.ndarray
        Weighted Jacobian-residual products, shape (voxels, parameters).
    jacobian_columns : list of numpy.ndarray
        Jacobian columns, each shape (voxels, observations).
    residuals : numpy.ndarray
        Observed minus predicted curves, shape (voxels, observations).
    weights : numpy.ndarray
        Observation weights.
    damping : numpy.ndarray
        Per-voxel LM damping factors.

    Returns
    -------
    numpy.ndarray
        Steps in natural parameter units. Failed solves return zero steps.
    """
    n_voxels, n_parameters = rhs.shape
    scale = np.sqrt(np.maximum(np.diagonal(gram, axis1=1, axis2=2), 1e-12))
    normalized = gram / scale[:, :, None] / scale[:, None, :]
    for j in range(n_parameters):
        normalized[:, j, j] += damping
    normalized_rhs = rhs / scale
    step = np.zeros((n_voxels, n_parameters), dtype=np.float64)
    try:
        return np.linalg.solve(normalized, normalized_rhs[:, :, None])[:, :, 0] / scale
    except np.linalg.LinAlgError:
        pass
    # On direct-solve failure, use the augmented Jacobian rather than its Gram
    # matrix to avoid squaring the condition number.
    for index in range(n_voxels):
        jacobian = np.column_stack([column[index] for column in jacobian_columns])
        design = np.vstack([
            np.sqrt(weights)[:, None] * jacobian / scale[index],
            np.sqrt(damping[index]) * np.eye(n_parameters),
        ])
        target = np.concatenate([np.sqrt(weights) * residuals[index], np.zeros(n_parameters)])
        try:
            step[index] = np.linalg.lstsq(design, target, rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
    return step / scale


def batched_lm_fit(
    model_name: str,
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
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
        FitResult with fitted parameters, R², and convergence flags.
    """
    tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)

    n_voxels, n_timepoints = tissue_curves.shape

    free_param_names = list(_MODEL_PARAMS[model_name])
    n_free = len(free_param_names)

    # Get bounds for free parameters
    bounds_lo = np.array(
        [_PARAM_BOUNDS[name][0] for name in free_param_names]
    )
    bounds_hi = np.array(
        [_PARAM_BOUNDS[name][1] for name in free_param_names]
    )

    params = _batched_initial_guess(
        model_name, tissue_curves, aif, time_points, free_param_names,
    )

    for j in range(n_free):
        params[:, j] = np.clip(params[:, j], bounds_lo[j], bounds_hi[j])
    params = _apply_volume_constraint_batched(
        params,
        free_param_names,
    )

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

    def _evaluate(
        free_params: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate the forward model for the selected rows."""
        param_dict = _build_param_dict(model_name, free_params, free_param_names)
        predicted = evaluate_model(
            model_name, param_dict, aif, time_points,
        ).tissue_curves
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
        predicted = _evaluate(p_active)
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
            p_plus = _apply_volume_constraint_batched(
                p_plus,
                free_param_names,
            )
            p_minus = _apply_volume_constraint_batched(
                p_minus,
                free_param_names,
            )

            actual_delta = p_plus[:, j] - p_minus[:, j]
            actual_delta = np.maximum(actual_delta, fd_floors[j])

            pred_plus = _evaluate(p_plus)
            pred_minus = _evaluate(p_minus)

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

        dp = _scaled_lm_step(JtJ, Jtr, J_columns, residuals, w, lam[active_idx])

        # Trial step
        p_trial = p_active + dp
        for j in range(n_free):
            p_trial[:, j] = np.clip(p_trial[:, j], bounds_lo[j], bounds_hi[j])

        p_trial = _apply_volume_constraint_batched(
            p_trial,
            free_param_names,
        )

        # Evaluate trial
        pred_trial = _evaluate(p_trial)
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
    full_names = _MODEL_PARAMS[model_name]
    all_param_dict = _build_param_dict(model_name, params, free_param_names)
    all_predicted = evaluate_model(
        model_name, all_param_dict, aif, time_points,
    ).tissue_curves

    boundary_uncertainties: dict[str, npt.NDArray[np.float64]] = {}
    needs_boundary_refit = model_name in ("extended_tofts", "2cxm")
    if needs_boundary_refit:
        volume = all_param_dict["ve"] + all_param_dict["vp"]
        at_box_bound = np.any(
            np.isclose(params, bounds_lo[None, :], rtol=0.0, atol=1e-10)
            | np.isclose(params, bounds_hi[None, :], rtol=0.0, atol=1e-10),
            axis=1,
        )
        at_volume_bound = np.isclose(
            volume, VP_VE_TOTAL_MAX, rtol=0.0, atol=1e-10
        )
        boundary_indices = np.where(
            at_volume_bound
            if model_name == "extended_tofts"
            else at_box_bound | at_volume_bound
        )[0]
        if boundary_indices.size:
            scalar_model = (
                ExtendedToftsModel()
                if model_name == "extended_tofts"
                else TwoCompartmentExchangeModel()
            )
            boundary_uncertainties = {
                name: np.full(n_voxels, np.nan, dtype=np.float64)
                for name in full_names
            }
            for index in boundary_indices:
                initial = [float(all_param_dict[name][index]) for name in full_names]
                result = scalar_model.fit(
                    tissue_curves[index],
                    aif,
                    time_points,
                    initial_guess=initial,
                    weights=weights,
                )
                if model_name == "2cxm":
                    # A zero-volume seed can trap the refit in a reduced
                    # model. Also try the normal data-driven initialization,
                    # retaining only the better weighted objective.
                    restarted = scalar_model.fit(
                        tissue_curves[index], aif, time_points, weights=weights,
                    )
                    if restarted.converged:
                        parameter_names = ("Fp", "PS", "ve", "vp")
                        restarted_curve = scalar_model.evaluate(
                            [restarted.parameters[name] for name in parameter_names], aif, time_points,
                        )
                        seeded_curve = scalar_model.evaluate(
                            [result.parameters[name] for name in parameter_names], aif, time_points,
                        )
                        if not result.converged or np.sum(
                            w * (tissue_curves[index] - restarted_curve) ** 2
                        ) < np.sum(w * (tissue_curves[index] - seeded_curve) ** 2):
                            result = restarted
                if not result.converged:
                    continue
                result_names = (
                    ("Ktrans", "ve", "vp")
                    if model_name == "extended_tofts"
                    else ("Fp", "PS", "ve", "vp")
                )
                candidate = np.array(
                    [result.parameters[name] for name in result_names],
                    dtype=np.float64,
                )
                candidate_curve = scalar_model.evaluate(
                    candidate.tolist(),
                    aif,
                    time_points,
                )
                old_rss = float(
                    np.sum(w * (tissue_curves[index] - all_predicted[index]) ** 2)
                )
                new_rss = float(
                    np.sum(w * (tissue_curves[index] - candidate_curve) ** 2)
                )
                if new_rss > old_rss * (1.0 + 1e-12):
                    continue
                params[index] = candidate
                all_predicted[index] = candidate_curve
                converged_flags[index] = True
                for name in full_names:
                    uncertainty = result.uncertainties.get(
                        {"ktrans": "Ktrans", "fp": "Fp", "ps": "PS"}.get(
                            name, name
                        ),
                        np.nan,
                    )
                    boundary_uncertainties[name][index] = uncertainty

    final_r2 = _compute_r_squared(tissue_curves, all_predicted)
    final_fitted = all_predicted if save_fitted_curves else None

    # Build output params dict
    output_params = _build_param_dict(model_name, params, free_param_names)

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
                pp = _apply_volume_constraint_batched(
                    pp,
                    free_param_names,
                )
                pm = _apply_volume_constraint_batched(
                    pm,
                    free_param_names,
                )

                actual_d = np.maximum(pp[:, j] - pm[:, j], fd_floors[j])

                pred_p = _evaluate(pp)
                pred_m = _evaluate(pm)
                J_cols_final.append((pred_p - pred_m) / actual_d[:, None])

            n_conv = len(conv_idx)
            JtJ_final = np.zeros((n_conv, n_free, n_free))
            for j in range(n_free):
                wJ_j = w[None, :] * J_cols_final[j]
                for k in range(j, n_free):
                    val = np.sum(wJ_j * J_cols_final[k], axis=1)
                    JtJ_final[:, j, k] = val
                    JtJ_final[:, k, j] = val

            pred_conv = _evaluate(p_conv)
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

        for name, values in boundary_uncertainties.items():
            finite = np.isfinite(values)
            uncertainties[name][finite] = values[finite]

    return FitResult(
        params=output_params,
        r_squared=final_r2,
        converged=converged_flags,
        fitted_curves=final_fitted,
        n_iterations=last_iteration + 1 if n_voxels > 0 else 0,
        model_name=model_name,
        uncertainties=uncertainties,
    )
