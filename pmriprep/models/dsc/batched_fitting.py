"""Batched fitting for DSC perfusion models.

This module provides batched forward model evaluation and Variational Bayes
fitting for DSC-MRI parametric models (CPI and Vascular). The key optimization
is vectorizing forward model evaluations across all voxels simultaneously.

Supported models:
- **CPI**: Control-point interpolation with PCHIP residue functions.
    Vectorized Fritsch-Carlson PCHIP on CPU + shared Toeplitz matmul.
- **Vascular**: Gamma-variate residue with FFT convolution + delay shift.

References:
    Chappell MA, Groves AR, Whitcher B, Woolrich MW (2009).
    Variational Bayesian Inference for a Nonlinear Forward Model.
    IEEE Trans Med Imaging 28(1):94-104.

    Mouridsen K, Friston K, Hjort N, et al. (2006). Bayesian estimation
    of cerebral perfusion using a physiological model of microvasculature.
    NeuroImage 33(2):570-579.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.special import gammainc

from pmriprep.models.bayesian_utils import (
    TransformType,
    from_transformed_batch,
    log_jacobian_det,
    to_transformed_batch,
    transform_prior,
)
from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION

logger = logging.getLogger("nipype.interface")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DSCBatchedFitResult:
    """Result from DSC batched fitting.

    Attributes:
        params: Parameter arrays, param_name -> (N,).
        r_squared: R-squared values, shape (N,).
        converged: Convergence flags, shape (N,).
        fitted_curves: Fitted tissue curves, shape (N, T) or None.
        n_iterations: Number of VI iterations performed.
        model_name: Name of the fitted model ('cpi' or 'vascular').
        uncertainties: Parameter uncertainties, param_name -> (N,).
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


# ---------------------------------------------------------------------------
# Batched Vascular forward model
# ---------------------------------------------------------------------------


def batched_vascular_forward(
    cbf: npt.NDArray[np.floating],
    alpha: npt.NDArray[np.floating],
    beta: npt.NDArray[np.floating],
    delay: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Batched Vascular model forward evaluation for N voxels.

    Computes tissue concentration via FFT convolution of AIF with
    gamma-variate residue function, with post-hoc delay shift.

    Args:
        cbf: CBF values, shape (N,).
        alpha: Gamma shape, shape (N,).
        beta: Gamma scale (seconds), shape (N,).
        delay: Bolus arrival delay (seconds), shape (N,).
        aif: Arterial input function, shape (T,).
        time_points: Time points (seconds), shape (T,).

    Returns:
        Tissue concentration curves, shape (N, T).
    """
    n_voxels = len(cbf)
    n_times = len(time_points)
    dt = float(time_points[1] - time_points[0]) if n_times > 1 else 1.0

    # Guard parameters
    alpha_safe = np.maximum(alpha, 1e-10)
    beta_safe = np.maximum(beta, 1e-10)

    # Compute residue WITHOUT delay: R(t) = CBF * [1 - gammainc(alpha, t/beta)]
    # Shape: (N, T)
    x = time_points[None, :] / beta_safe[:, None]  # (N, T)
    gamma_vals = gammainc(alpha_safe[:, None], x)  # (N, T) — vectorized
    R_no_delay = cbf[:, None] * (1.0 - gamma_vals)

    # FFT convolution: C = AIF ⊗ R for each voxel
    # Use batch FFT: pad to avoid circular artifacts
    pad_len = 2 * n_times
    aif_padded = np.zeros(pad_len)
    aif_padded[:n_times] = aif

    R_padded = np.zeros((n_voxels, pad_len))
    R_padded[:, :n_times] = R_no_delay

    aif_fft = np.fft.fft(aif_padded)
    R_fft = np.fft.fft(R_padded, axis=1)
    conv_full = np.fft.ifft(aif_fft[None, :] * R_fft, axis=1).real
    conv = conv_full[:, :n_times] * dt  # (N, T)

    # Apply delay as post-hoc shift with linear interpolation (vectorized)
    delay_frac = delay / dt
    n_shift = np.floor(delay_frac).astype(int)
    f_shift = delay_frac - n_shift  # fractional part, (N,)

    indices = np.arange(n_times)[None, :]  # (1, T)
    src_all = indices - n_shift[:, None]   # (N, T)
    row_idx = np.arange(n_voxels)[:, None]

    # Integer-shift contribution: (1 - f) * conv[i - n_shift]
    valid_lo = (src_all >= 0) & (src_all < n_times)
    src_clipped = np.clip(src_all, 0, n_times - 1)
    C_t = np.where(
        valid_lo,
        (1.0 - f_shift[:, None]) * conv[row_idx, src_clipped],
        0.0,
    )

    # Fractional contribution: f * conv[i - n_shift - 1]
    src_hi = src_all - 1
    valid_hi = (src_hi >= 0) & (src_hi < n_times)
    src_hi_clipped = np.clip(src_hi, 0, n_times - 1)
    C_t += np.where(
        valid_hi,
        f_shift[:, None] * conv[row_idx, src_hi_clipped],
        0.0,
    )

    return C_t


# ---------------------------------------------------------------------------
# Batched CPI forward model
# ---------------------------------------------------------------------------


def batched_cpi_forward(
    params: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    control_point_times: npt.NDArray[np.floating],
    toeplitz_matrix: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Batched CPI model forward evaluation for N voxels.

    Computes tissue concentration via PCHIP-interpolated residue
    convolved with AIF using a precomputed Toeplitz matrix.

    C_tissue(t) = CBF * (Toeplitz @ R_norm(t))

    PCHIP interpolation is vectorized across voxels using
    Fritsch-Carlson monotone slope computation.

    Args:
        params: Parameter matrix, shape (N, P) where P = n_control_points.
            Column 0 = CBF, columns 1..P-1 = multiplicative fractions f1..f_{P-1}.
        aif: Arterial input function, shape (T,).
        time_points: Time points (seconds), shape (T,).
        control_point_times: Control point time locations, shape (K,).
        toeplitz_matrix: Precomputed Toeplitz convolution matrix, shape (T, T).

    Returns:
        Tissue concentration curves, shape (N, T).
    """
    n_voxels = params.shape[0]
    n_times = len(time_points)
    n_cp = len(control_point_times)

    cbf = params[:, 0]  # (N,)

    # Cumulative product of multiplicative fractions: R(t_k) = f1*f2*...*fk
    cp_vals = np.ones((n_voxels, n_cp), dtype=np.float64)
    fracs = np.clip(params[:, 1:], 0.0, 1.0)  # (N, K-1)
    cp_vals[:, 1:] = np.cumprod(fracs, axis=1)

    # Vectorized PCHIP interpolation
    R_norm = _batched_pchip(control_point_times, cp_vals, time_points)
    R_norm = np.clip(R_norm, 0.0, 1.0)  # (N, T)

    # Toeplitz matmul: C = CBF * (A @ R_norm^T)^T
    # Memory batching: (T,T) @ (T,N) costs T*N*8 bytes for the result.
    # For N=200K, T=100: ~160MB. Batch to keep under ~500MB.
    _MAX_BATCH = 20000  # ~16MB per batch at T=100
    tissue = np.zeros((n_voxels, n_times), dtype=np.float64)
    for start in range(0, n_voxels, _MAX_BATCH):
        end = min(start + _MAX_BATCH, n_voxels)
        tissue[start:end] = cbf[start:end, None] * (
            toeplitz_matrix @ R_norm[start:end].T
        ).T

    return tissue


def _batched_pchip(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    xi: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Vectorized PCHIP interpolation for N sets of y-values.

    Uses Fritsch-Carlson monotone slopes computed on (N, K) arrays
    simultaneously, avoiding per-voxel scipy.interpolate calls.

    Args:
        x: Knot positions, shape (K,). Must be strictly increasing.
        y: Knot values, shape (N, K).
        xi: Evaluation points, shape (T,).

    Returns:
        Interpolated values, shape (N, T).
    """
    n_voxels, n_knots = y.shape
    n_eval = len(xi)

    # Compute intervals and slopes
    h = np.diff(x)  # (K-1,)
    delta = np.diff(y, axis=1) / h[None, :]  # (N, K-1)

    # Fritsch-Carlson monotone slopes
    d = np.zeros_like(y)  # (N, K)

    # Interior slopes
    for k in range(1, n_knots - 1):
        d1 = delta[:, k - 1]
        d2 = delta[:, k]
        # Where signs differ or either is zero → slope = 0
        same_sign = (d1 * d2) > 0
        # Harmonic mean for monotone interpolation
        d[:, k] = np.where(
            same_sign,
            2.0 * d1 * d2 / (d1 + d2 + 1e-300),
            0.0,
        )

    # Endpoint slopes (one-sided)
    d[:, 0] = delta[:, 0]
    d[:, -1] = delta[:, -1]

    # Clamp endpoint slopes for monotonicity
    for endpoint, side in [(0, 0), (-1, -1)]:
        sign_match = d[:, endpoint] * delta[:, side] > 0
        d[:, endpoint] = np.where(sign_match, d[:, endpoint], 0.0)

    # Find intervals for each evaluation point
    # np.searchsorted returns index of rightmost knot <= xi
    indices = np.searchsorted(x, xi, side='right') - 1
    indices = np.clip(indices, 0, n_knots - 2)  # (T,)

    # Hermite basis evaluation
    t_local = (xi - x[indices]) / h[indices]  # (T,) normalized [0, 1]
    t_local = np.clip(t_local, 0.0, 1.0)

    # Hermite basis functions
    t2 = t_local ** 2
    t3 = t_local ** 3
    h00 = 2 * t3 - 3 * t2 + 1      # (T,)
    h10 = t3 - 2 * t2 + t_local     # (T,)
    h01 = -2 * t3 + 3 * t2          # (T,)
    h11 = t3 - t2                    # (T,)

    # Gather values at interval boundaries for all voxels
    y0 = y[:, indices]              # (N, T)
    y1 = y[:, indices + 1]          # (N, T)
    d0 = d[:, indices] * h[indices] # (N, T) — scale slopes by interval width
    d1 = d[:, indices + 1] * h[indices]

    # Evaluate: result = h00*y0 + h10*d0 + h01*y1 + h11*d1
    result: npt.NDArray[np.floating] = (
        h00[None, :] * y0
        + h10[None, :] * d0
        + h01[None, :] * y1
        + h11[None, :] * d1
    )

    # Exponential decay extrapolation beyond last knot (matches CPI scalar path).
    # Uses last two knot values to estimate decay rate.
    beyond_mask = xi > x[-1]
    if np.any(beyond_mask) and n_knots >= 2:
        r_last = y[:, -1]    # (N,)
        r_prev = y[:, -2]    # (N,)
        dt_knot = x[-1] - x[-2]
        # Decay rate from last two points: lambda = ln(r_prev / r_last) / dt
        safe_last = np.maximum(r_last, 1e-10)
        safe_prev = np.maximum(r_prev, 1e-10)
        decay_rate = np.clip(
            np.log(safe_prev / safe_last) / dt_knot, 0.01, 1.0,
        )  # (N,)
        t_excess = xi[beyond_mask] - x[-1]  # (M,) where M = sum(beyond_mask)
        result[:, beyond_mask] = r_last[:, None] * np.exp(
            -decay_rate[:, None] * t_excess[None, :]
        )

    return result


# ---------------------------------------------------------------------------
# Batched DSC Variational Inference
# ---------------------------------------------------------------------------


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
    control_point_times: Optional[npt.NDArray[np.floating]] = None,
    toeplitz_matrix: Optional[npt.NDArray[np.floating]] = None,
    nlls_seeds: Optional[npt.NDArray[np.floating]] = None,
    prior_means_transformed: Optional[npt.NDArray[np.floating]] = None,
    prior_stds_transformed: Optional[npt.NDArray[np.floating]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    n_iterations: int = 30,
    sloppy: bool = False,
    save_fitted_curves: bool = False,
    spatial_prec_diag: Optional[npt.NDArray[np.floating]] = None,
    spatial_eta: Optional[npt.NDArray[np.floating]] = None,
    return_posteriors: bool = False,
    n_iterations_override: Optional[int] = None,
) -> DSCBatchedFitResult:
    """Batched Variational Bayes fitting for DSC perfusion models.

    Implements the linearized VB algorithm from Chappell et al. (2009)
    vectorized across all voxels. Supports both CPI and Vascular models.

    Args:
        model_type: 'cpi' or 'vascular'.
        tissue_curves: Observed tissue curves, shape (N, T).
        aif: Arterial input function, shape (T,).
        time_points: Time points (seconds), shape (T,).
        prior_means_nat: Prior means in natural (fitting) space, shape (P,).
        prior_stds_nat: Prior stds in natural (fitting) space, shape (P,).
        transforms: Per-parameter transform types, length P.
        upper_bounds: Per-parameter upper bounds for LOG_LOGIT, length P.
        fitting_param_names: Parameter names in fitting space, length P.
        fitting_bounds: Per-parameter bounds in fitting space, length P.
        mono_prec: CPI monotonicity precision matrix, shape (P, P) or None.
        control_point_times: CPI control point times, shape (K,) or None.
        toeplitz_matrix: Precomputed Toeplitz matrix, shape (T, T) or None.
        nlls_seeds: NLLS seed parameters for initialization, shape (N, P) or None.
        prior_means_transformed: Pre-computed prior means in transformed space,
            shape (P,). If provided, bypasses internal transform_prior() call.
            Use this to preserve transformed_prior_sigma overrides from
            BayesianPrior objects (e.g., FABBER sigma=1.0 convention).
        prior_stds_transformed: Pre-computed prior stds in transformed space,
            shape (P,). Must be provided alongside prior_means_transformed.
        weights: Per-frame motion weights, shape (T,). None = uniform.
        n_iterations: Maximum VB iterations (default 20).
        sloppy: Use fast, low-quality settings (8 iterations).
        save_fitted_curves: Whether to store fitted curves in result.
        spatial_prec_diag: Per-voxel spatial precision diagonal, shape (N, P).
            N must match tissue_curves.shape[0].
        spatial_eta: Per-voxel spatial mean contribution, shape (N, P).
        return_posteriors: If True, populate posterior_means_transformed and
            posterior_cov_diag on the result.
        n_iterations_override: When set, overrides n_iterations and sloppy,
            and disables ELBO-based step-size reduction. Used by spatial VB
            wrapper where the ELBO is invalidated by spatial terms.

    Returns:
        DSCBatchedFitResult with fitted parameters, R², convergence, uncertainties.
    """
    from scipy.special import digamma, gammaln

    tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)
    n_voxels, n_obs = tissue_curves.shape
    n_params = len(fitting_param_names)

    if n_iterations_override is not None:
        n_iterations = n_iterations_override
    elif sloppy:
        n_iterations = 8
    # Disables ELBO step-size/convergence (spatial_vb.py is sole consumer)
    _spatial_mode = n_iterations_override is not None

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
        )

    y = tissue_curves[valid_idx]  # (N_valid, T)

    # --- Prior setup ---
    # Use pre-transformed means/stds if provided (preserves transformed_prior_sigma
    # overrides from BayesianPrior, e.g., FABBER sigma=1.0 convention for Vascular).
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

    # Add CPI monotonicity penalty
    if mono_prec is not None:
        prior_prec = prior_prec + mono_prec

    # Noise precision prior
    c0 = 1e-3
    s0 = 1e3

    # --- Bounds ---
    bounds_lo = np.array([b[0] for b in fitting_bounds])
    bounds_hi = np.array([b[1] for b in fitting_bounds])

    # --- Initialize posterior mean ---
    m_nat = np.tile(prior_means_nat, (n_valid, 1)).astype(np.float64)
    for j in range(n_params):
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
        if model_type == "vascular":
            # Fitting space: (cbf, mtt, alpha, delay) → model: (cbf, alpha, beta, delay)
            cbf_v = params_fitting[:, 0]
            mtt_v = params_fitting[:, 1]
            alpha_v = params_fitting[:, 2]
            delay_v = params_fitting[:, 3]
            beta_v = mtt_v / np.maximum(alpha_v, 1e-10)
            return batched_vascular_forward(
                cbf_v, alpha_v, beta_v, delay_v, aif, time_points,
            )
        else:  # cpi
            if control_point_times is None or toeplitz_matrix is None:
                raise ValueError("CPI model requires control_point_times and toeplitz_matrix")
            return batched_cpi_forward(
                params_fitting, aif, time_points,
                control_point_times, toeplitz_matrix,
            )

    # --- Noise initialization from NLLS seed residuals ---
    tau_exp = np.full(n_valid, c0 * s0)
    if nlls_seeds is not None:
        try:
            g_init = _evaluate(m_nat)
            init_var = np.sum((y - g_init) ** 2, axis=1) / n_obs
            good_var = init_var > 1e-20
            tau_exp[good_var] = 1.0 / init_var[good_var]
        except (ValueError, RuntimeError):
            pass

    c = np.full(n_valid, c0, dtype=np.float64)
    s = np.full(n_valid, s0, dtype=np.float64)

    # --- State tracking ---
    best_fe = np.full(n_valid, -np.inf)
    best_m = m.copy()
    best_cov_diag = np.tile(prior_stds_t ** 2, (n_valid, 1))
    best_c = c.copy()
    best_s = s.copy()
    step_scale = np.ones(n_valid)

    active = np.ones(n_valid, dtype=bool)
    converged = np.zeros(n_valid, dtype=bool)
    fe_plateau_count = np.zeros(n_valid, dtype=np.int32)

    w = np.ones(n_obs, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    _FD_REL = 1e-5
    _FD_FLOOR = 1e-10

    # Constant prior precision log-determinant
    _sign_prior_prec, _logdet_prior_prec = np.linalg.slogdet(prior_prec)
    _logdet_prior_cov = -_logdet_prior_prec

    last_iteration = 0

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

        # 3. Jacobian via central FD in transformed space
        deltas = np.maximum(np.abs(m_a) * _FD_REL, _FD_FLOOR)
        J_columns: list[npt.NDArray[np.floating]] = []

        for j in range(n_params):
            nat_plus = m_nat_a.copy()
            nat_plus[:, j] = from_transformed_batch(
                m_a[:, j] + deltas[:, j], transforms[j], upper_bounds[j],
            )
            nat_minus = m_nat_a.copy()
            nat_minus[:, j] = from_transformed_batch(
                m_a[:, j] - deltas[:, j], transforms[j], upper_bounds[j],
            )

            # For IDENTITY params, clip perturbations to bounds
            if transforms[j] == TransformType.IDENTITY:
                nat_plus[:, j] = np.clip(nat_plus[:, j], bounds_lo[j], bounds_hi[j])
                nat_minus[:, j] = np.clip(nat_minus[:, j], bounds_lo[j], bounds_hi[j])

            g_plus = _evaluate(nat_plus)
            g_minus = _evaluate(nat_minus)

            actual_delta = (m_a[:, j] + deltas[:, j]) - (m_a[:, j] - deltas[:, j])
            # For IDENTITY with clipping, recompute actual delta
            if transforms[j] == TransformType.IDENTITY:
                actual_delta = nat_plus[:, j] - nat_minus[:, j]
                actual_delta = np.maximum(actual_delta, _FD_FLOOR)

            J_columns.append(
                (g_plus - g_minus) / (actual_delta[:, None])
            )

        # Residual
        k_res = y_a - g_m

        # 4. Build JtWJ via column-loop
        JtWJ = np.zeros((n_active, n_params, n_params))
        JtWk = np.zeros((n_active, n_params))
        for j in range(n_params):
            wJ_j = w[None, :] * J_columns[j]
            JtWk[:, j] = np.sum(wJ_j * k_res, axis=1)
            for k in range(j, n_params):
                val = np.sum(wJ_j * J_columns[k], axis=1)
                JtWJ[:, j, k] = val
                JtWJ[:, k, j] = val

        # 5. Posterior precision and covariance
        post_prec = prior_prec[None, :, :] + tau_a[:, None, None] * JtWJ
        if spatial_prec_diag is not None:
            # (N_total, P) -> valid subset -> active subset
            sp_a = spatial_prec_diag[valid_idx][a_idx]
            post_prec[:, range(n_params), range(n_params)] += sp_a

        post_cov = np.linalg.inv(post_prec)
        bad_cov = ~np.isfinite(post_cov).all(axis=(1, 2))
        if np.any(bad_cov):
            post_prec[bad_cov] += 1e-6 * np.eye(n_params)
            post_cov[bad_cov] = np.linalg.inv(post_prec[bad_cov])

        # 6. Posterior mean update
        Jm = np.zeros((n_active, n_obs))
        for j in range(n_params):
            Jm += J_columns[j] * m_a[:, j:j + 1]

        lam0_mu0 = prior_prec @ prior_means_t
        if mono_eta is not None:
            lam0_mu0 = lam0_mu0 + mono_eta
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

        # IDENTITY params: clip to bounds in transformed space (= natural space)
        for j in range(n_params):
            if transforms[j] == TransformType.IDENTITY:
                m_new[:, j] = np.clip(m_new[:, j], bounds_lo[j], bounds_hi[j])

        # 7. Noise posterior update
        m_new_nat = np.zeros_like(m_new)
        for j in range(n_params):
            m_new_nat[:, j] = from_transformed_batch(
                m_new[:, j], transforms[j], upper_bounds[j],
            )

        g_new = _evaluate(m_new_nat)
        k_new = y_a - g_new

        c_new = c0 + n_obs / 2.0
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
            0.5 * n_obs * (np.log(tau_new) - np.log(2 * np.pi))
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
            + c0 * (np.log(s_new) - np.log(s0))
            + c_new * (s0 / s_new - 1.0)
        )
        free_energy += gamma_kl

        # 9. Step-size reduction
        if _spatial_mode:
            # Spatial mode: ELBO is invalid (spatial terms not in KL).
            # Always accept the update; convergence monitored at outer level.
            best_m[a_idx] = m_new
            best_cov_diag[a_idx] = np.diagonal(post_cov, axis1=1, axis2=2)
            best_c[a_idx] = c_new
            best_s[a_idx] = s_new
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
                "DSC batched VI iter %d: %d/%d active, %d converged",
                iteration, int(np.sum(active)), n_valid, int(np.sum(converged)),
            )

    logger.info(
        "DSC batched VI finished after %d iters: %d/%d converged (of %d total)",
        last_iteration + 1, int(np.sum(converged)), n_valid, n_voxels,
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

    # --- Uncertainties via delta method ---
    for j, name in enumerate(fitting_param_names):
        std_t = np.sqrt(np.maximum(best_cov_diag[:, j], 0.0))
        jac = np.exp(log_jacobian_det(fitted_nat[:, j], transforms[j], upper_bounds[j]))
        out_uncertainties[name][valid_idx] = std_t * jac

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

    out_fitted = None
    if save_fitted_curves:
        out_fitted = np.zeros((n_voxels, n_obs), dtype=np.float64)
        out_fitted[valid_idx] = final_pred

    out_post_means = None
    out_post_cov = None
    if return_posteriors:
        out_post_means = np.zeros((n_voxels, n_params), dtype=np.float64)
        out_post_cov = np.zeros((n_voxels, n_params), dtype=np.float64)
        out_post_means[valid_idx] = best_m
        out_post_cov[valid_idx] = best_cov_diag

    return DSCBatchedFitResult(
        params=out_params, r_squared=out_r2, converged=out_converged,
        fitted_curves=out_fitted, n_iterations=last_iteration + 1,
        model_name=model_type, uncertainties=out_uncertainties,
        posterior_means_transformed=out_post_means,
        posterior_cov_diag=out_post_cov,
    )
