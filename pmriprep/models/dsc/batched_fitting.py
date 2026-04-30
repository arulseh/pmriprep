"""Batched fitting for DSC perfusion models.

This module provides batched forward model evaluation, Variational Bayes
fitting, and MAP fitting for DSC-MRI parametric models (CPI and Vascular).
The key optimization is vectorizing forward model evaluations across all
voxels simultaneously.

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
from collections.abc import Callable, Mapping
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
        n_iterations: Number of fitting iterations performed.
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


def _append_terminal_anchor(
    cp_times: npt.NDArray[np.floating],
    cp_vals: npt.NDArray[np.floating],
    t_terminal: float | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Append terminal R=0 anchor (and prepend t=0 anchor) for shared CP grid.

    Mirrors FABBER's CPI boundary conditions: prepend (t=0, amp=1) when the
    first CP sits beyond t=0, and append (t_terminal, amp=0).

    Args:
        cp_times: Control point times, shape (K,).
        cp_vals: Control point values, shape (N, K).
        t_terminal: Optional explicit terminal time. If None, defaults to
            ``cp_times[-1] + (cp_times[-1] - cp_times[-2])``.

    Returns:
        Extended (cp_times, cp_vals) with anchors applied.
    """
    if t_terminal is None:
        spacing = float(cp_times[-1] - cp_times[-2]) if len(cp_times) >= 2 else 1.0
        t_terminal = float(cp_times[-1]) + spacing

    n_voxels = cp_vals.shape[0]
    if float(cp_times[0]) > 0.0:
        cp_times_ext = np.concatenate([[0.0], cp_times, [t_terminal]])
        ones = np.ones((n_voxels, 1), dtype=cp_vals.dtype)
        zeros = np.zeros((n_voxels, 1), dtype=cp_vals.dtype)
        cp_vals_ext = np.concatenate([ones, cp_vals, zeros], axis=1)
    else:
        cp_times_ext = np.append(cp_times, t_terminal)
        zeros = np.zeros((n_voxels, 1), dtype=cp_vals.dtype)
        cp_vals_ext = np.concatenate([cp_vals, zeros], axis=1)
    return cp_times_ext, cp_vals_ext


def _append_terminal_anchor_per_voxel(
    cp_times: npt.NDArray[np.floating],
    cp_vals: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Per-voxel variant of ``_append_terminal_anchor`` (for ``infer_cpt``).

    Args:
        cp_times: Control point times per voxel, shape (N, K).
        cp_vals: Control point values, shape (N, K).

    Returns:
        Extended (cp_times, cp_vals) each shape (N, K+2): prepended (0, 1)
        and appended (t_terminal, 0) per voxel.
    """
    n_voxels, n_cp = cp_vals.shape
    last_spacing = (cp_times[:, -1] - cp_times[:, -2]) if n_cp >= 2 else np.ones(n_voxels)
    t_term = cp_times[:, -1] + last_spacing  # (N,)

    zeros = np.zeros((n_voxels, 1), dtype=cp_vals.dtype)
    ones = np.ones((n_voxels, 1), dtype=cp_vals.dtype)
    leading_t = np.zeros((n_voxels, 1), dtype=cp_times.dtype)
    cp_times_ext = np.concatenate([leading_t, cp_times, t_term[:, None]], axis=1)
    cp_vals_ext = np.concatenate([ones, cp_vals, zeros], axis=1)
    return cp_times_ext, cp_vals_ext


def batched_cpi_forward(
    params: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    control_point_times: npt.NDArray[np.floating],
    toeplitz_matrix: npt.NDArray[np.floating],
    t_terminal: float | None = None,
) -> npt.NDArray[np.floating]:
    """Batched CPI model forward evaluation for N voxels.

    Computes tissue concentration via PCHIP-interpolated residue
    convolved with AIF using a precomputed Toeplitz matrix.

    C_tissue(t) = CBF * (Toeplitz @ R_norm(t))

    Supports both fixed and per-voxel control point times. When
    ``params`` has 2*n_cp - 1 columns, columns ``[n_cp..2n_cp-2]``
    are interpreted as log-spacings between CPs (FABBER ``infer-cpt``)
    and per-voxel CP times are reconstructed via cumulative-sum.

    Args:
        params: Parameter matrix, shape (N, P).
            P = n_cp: column 0 = CBF, columns 1..n_cp-1 = fractions.
            P = 2*n_cp - 1: above plus columns [n_cp..2*n_cp-2] = log-spacings.
        aif: Arterial input function, shape (T,).
        time_points: Time points (seconds), shape (T,).
        control_point_times: Default control point time locations, shape (K,).
        toeplitz_matrix: Precomputed Toeplitz convolution matrix, shape (T, T).
        t_terminal: Terminal anchor time (seconds) for fixed-CP layout.

    Returns:
        Tissue concentration curves, shape (N, T).
    """
    n_voxels = params.shape[0]
    n_times = len(time_points)
    n_cp = len(control_point_times)
    # FABBER layout: params = [cbf, f_1..f_N] (no infer_cpt) or
    # [cbf, f_1..f_N, s_1..s_N] (infer_cpt). All N fractions are free;
    # the (t=0, amp=1) BC is added at PCHIP-construction time.
    has_log_spacings = params.shape[1] == 1 + 2 * n_cp

    cbf = params[:, 0]  # (N,)

    # Cumulative product of multiplicative fractions: R(tpos[k]) = f1*...*fk
    fracs = np.clip(params[:, 1 : 1 + n_cp], 0.0, 1.0)  # (N, n_cp)
    cp_vals = np.cumprod(fracs, axis=1)

    if has_log_spacings:
        log_s = params[:, 1 + n_cp :]  # (N, n_cp)
        spacings = np.exp(np.clip(log_s, -50.0, 50.0))
        per_voxel_cp_times = np.cumsum(spacings, axis=1)
        cp_times_ext, cp_vals_ext = _append_terminal_anchor_per_voxel(
            per_voxel_cp_times, cp_vals
        )
        R_norm = _batched_pchip_per_voxel(cp_times_ext, cp_vals_ext, time_points)
    else:
        cp_times_ext, cp_vals_ext = _append_terminal_anchor(
            control_point_times, cp_vals, t_terminal=t_terminal,
        )
        R_norm = _batched_pchip(cp_times_ext, cp_vals_ext, time_points)

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

    return result


def _batched_pchip_per_voxel(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    xi: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Per-voxel PCHIP for ``infer_cpt`` (CP times vary per voxel).

    Args:
        x: Knot positions per voxel, shape (N, K). Must be strictly
            increasing along axis=1.
        y: Knot values per voxel, shape (N, K).
        xi: Evaluation points (shared), shape (T,).

    Returns:
        Interpolated values, shape (N, T).
    """
    n_voxels, n_knots = y.shape
    n_eval = len(xi)

    h = np.diff(x, axis=1)  # (N, K-1)
    h = np.where(h <= 0.0, 1e-12, h)
    delta = np.diff(y, axis=1) / h  # (N, K-1)

    d = np.zeros_like(y)  # (N, K)

    # Interior slopes — Fritsch-Carlson harmonic mean
    if n_knots >= 3:
        d1 = delta[:, :-1]  # (N, K-2)
        d2 = delta[:, 1:]
        same_sign = (d1 * d2) > 0
        d[:, 1:-1] = np.where(
            same_sign,
            2.0 * d1 * d2 / (d1 + d2 + 1e-300),
            0.0,
        )

    # Endpoint slopes (one-sided)
    d[:, 0] = delta[:, 0]
    d[:, -1] = delta[:, -1]

    # Clamp endpoint slopes for monotonicity
    sign0 = d[:, 0] * delta[:, 0] > 0
    d[:, 0] = np.where(sign0, d[:, 0], 0.0)
    signN = d[:, -1] * delta[:, -1] > 0
    d[:, -1] = np.where(signN, d[:, -1], 0.0)

    # Per-voxel intervals: searchsorted broadcast in a small Python loop
    # (n_voxels can be 100k+ but n_eval ~50 so this is dominated by numpy ops)
    result = np.zeros((n_voxels, n_eval), dtype=np.float64)
    for v in range(n_voxels):
        xv = x[v]
        idx = np.searchsorted(xv, xi, side='right') - 1
        idx = np.clip(idx, 0, n_knots - 2)
        hv = h[v, idx]
        t_local = (xi - xv[idx]) / hv
        t_local = np.clip(t_local, 0.0, 1.0)

        t2 = t_local ** 2
        t3 = t_local ** 3
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t_local
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        y0 = y[v, idx]
        y1 = y[v, idx + 1]
        d0 = d[v, idx] * hv
        d1v = d[v, idx + 1] * hv
        result[v] = h00 * y0 + h10 * d0 + h01 * y1 + h11 * d1v

    return result


# ---------------------------------------------------------------------------
# Shared helpers for batched fitting (VI and MAP)
# ---------------------------------------------------------------------------

_FD_REL = 1e-5
_FD_FLOOR = 1e-10


def _compute_jacobian_columns(
    m_a: npt.NDArray[np.floating],
    m_nat_a: npt.NDArray[np.floating],
    g_m: npt.NDArray[np.floating],
    evaluate_fn: "Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]",
    transforms: list[TransformType],
    upper_bounds: list[float | None],
    bounds_lo: npt.NDArray[np.floating],
    bounds_hi: npt.NDArray[np.floating],
    cbf_idx: int,
    fitting_param_names: list[str] | None = None,
) -> list[npt.NDArray[np.floating]]:
    """Compute Jacobian columns: analytical CBF + forward FD for others.

    For CBF parameter:
    - LOG transform: ∂g/∂(log cbf) = g (since g = cbf · h(rest)).
    - IDENTITY transform: ∂g/∂cbf = g / cbf = h(rest).

    For log-spacing parameters (name starts with "s") under IDENTITY transform,
    a looser FD step is used because PCHIP knot-shift roundoff dominates the
    default 1e-5 step.

    Args:
        m_a: Transformed parameters, shape (n_active, P).
        m_nat_a: Natural-space parameters, shape (n_active, P).
        g_m: Forward model at current params, shape (n_active, T).
        evaluate_fn: Batched forward model callable.
        transforms: Per-parameter transform types.
        upper_bounds: Per-parameter upper bounds for LOG_LOGIT.
        bounds_lo: Lower bounds in natural space, shape (P,).
        bounds_hi: Upper bounds in natural space, shape (P,).
        cbf_idx: Index of CBF parameter (uses analytical Jacobian).
        fitting_param_names: Optional parameter names; used to detect
            log-spacing columns (starting with "s") for relaxed FD step.

    Returns:
        List of P arrays, each shape (n_active, T).
    """
    n_params = m_a.shape[1]
    deltas = np.maximum(np.abs(m_a) * _FD_REL, _FD_FLOOR)

    # Looser FD step for log-spacing columns to clear PCHIP knot-shift roundoff.
    if fitting_param_names is not None:
        for j in range(n_params):
            name = fitting_param_names[j]
            if (
                transforms[j] == TransformType.IDENTITY
                and j != cbf_idx
                and name.startswith("s")
            ):
                deltas[:, j] = np.maximum(np.abs(m_a[:, j]) * 1e-4, 1e-3)

    J_columns: list[npt.NDArray[np.floating]] = []

    for j in range(n_params):
        if j == cbf_idx:
            if transforms[j] == TransformType.LOG:
                J_columns.append(g_m)
            else:  # IDENTITY
                cbf_vals = m_nat_a[:, cbf_idx, None]
                J_columns.append(g_m / np.maximum(cbf_vals, 1e-30))
            continue

        nat_plus = m_nat_a.copy()
        nat_plus[:, j] = from_transformed_batch(
            m_a[:, j] + deltas[:, j], transforms[j], upper_bounds[j],
        )

        if transforms[j] == TransformType.IDENTITY:
            nat_plus[:, j] = np.clip(nat_plus[:, j], bounds_lo[j], bounds_hi[j])

        g_plus = evaluate_fn(nat_plus)

        if transforms[j] == TransformType.IDENTITY:
            actual_delta = nat_plus[:, j] - m_nat_a[:, j]
            actual_delta = np.maximum(np.abs(actual_delta), _FD_FLOOR) * np.sign(
                np.where(actual_delta == 0.0, 1.0, actual_delta)
            )
        else:
            actual_delta = deltas[:, j]

        J_columns.append((g_plus - g_m) / actual_delta[:, None])

    return J_columns


def _assemble_JtWJ(
    J_columns: list[npt.NDArray[np.floating]],
    k_res: npt.NDArray[np.floating],
    w: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Assemble J^T W J and J^T W k from Jacobian columns.

    Args:
        J_columns: List of P arrays, each shape (n_active, T).
        k_res: Residuals, shape (n_active, T).
        w: Per-frame weights, shape (T,).

    Returns:
        JtWJ: shape (n_active, P, P).
        JtWk: shape (n_active, P).
    """
    n_params = len(J_columns)
    n_active = J_columns[0].shape[0]
    JtWJ = np.zeros((n_active, n_params, n_params))
    JtWk = np.zeros((n_active, n_params))
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
    cpi_t_terminal: Optional[float] = None,
    nlls_seeds: Optional[npt.NDArray[np.floating]] = None,
    prior_means_transformed: Optional[npt.NDArray[np.floating]] = None,
    prior_stds_transformed: Optional[npt.NDArray[np.floating]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    n_iterations: int = 10,
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
                t_terminal=cpi_t_terminal,
            )

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

    c = np.full(n_valid, c_post_init, dtype=np.float64)
    s = np.full(n_valid, s_post_init, dtype=np.float64)

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
        J_columns = _compute_jacobian_columns(
            m_a, m_nat_a, g_m, _evaluate,
            transforms, upper_bounds, bounds_lo, bounds_hi, cbf_idx,
            fitting_param_names=fitting_param_names,
        )

        # Residual
        k_res = y_a - g_m

        # 4. Build JtWJ via column-loop
        JtWJ, JtWk = _assemble_JtWJ(J_columns, k_res, w)

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
            + c0 * (np.log(s0) - np.log(s_new))
            + c_new * (s_new / s0 - 1.0)
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


# ---------------------------------------------------------------------------
# Batched DSC MAP (Maximum A Posteriori)
# ---------------------------------------------------------------------------


def _run_map_gauss_newton(
    y: npt.NDArray[np.floating],
    m_init: npt.NDArray[np.floating],
    sigma_sq_init: npt.NDArray[np.floating],
    prior_means_t: npt.NDArray[np.floating],
    prior_prec: npt.NDArray[np.floating],
    mono_eta: Optional[npt.NDArray[np.floating]],
    transforms: list[TransformType],
    upper_bounds: list[float | None],
    bounds_lo: npt.NDArray[np.floating],
    bounds_hi: npt.NDArray[np.floating],
    cbf_idx: int,
    w: npt.NDArray[np.floating],
    n_eff: float,
    evaluate_fn: "Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]",
    n_iterations: int,
    n_noise_iterations: int,
    fitting_param_names: list[str] | None = None,
) -> tuple[
    npt.NDArray[np.floating],  # best_m (n_valid, P)
    npt.NDArray[np.floating],  # best_nlp (n_valid,)
    npt.NDArray[np.floating],  # best_sigma_sq (n_valid,)
    npt.NDArray[np.bool_],     # converged (n_valid,)
    int,                       # last_iteration
]:
    """Run Gauss-Newton MAP optimization with profile likelihood noise.

    Args:
        y: Observed curves, shape (n_valid, T).
        m_init: Initial transformed parameters, shape (n_valid, P).
        sigma_sq_init: Initial noise variance, shape (n_valid,).
        prior_means_t: Prior means in transformed space, shape (P,).
        prior_prec: Prior precision matrix, shape (P, P).
        mono_eta: CPI monotonicity bias, shape (P,) or None.
        transforms: Per-parameter transform types.
        upper_bounds: Per-parameter LOG_LOGIT upper bounds.
        bounds_lo: Lower bounds in natural space, shape (P,).
        bounds_hi: Upper bounds in natural space, shape (P,).
        cbf_idx: Index of CBF parameter.
        w: Per-frame weights, shape (T,).
        n_eff: Effective number of observations for noise estimation.
        evaluate_fn: Batched forward model.
        n_iterations: Gauss-Newton iterations per noise loop.
        n_noise_iterations: Profile likelihood outer iterations.

    Returns:
        Tuple of (best_m, best_nlp, best_sigma_sq, converged, last_iteration).
    """
    n_valid = y.shape[0]
    n_obs = y.shape[1]
    n_params = m_init.shape[1]

    m = m_init.copy()
    sigma_sq = sigma_sq_init.copy()

    best_nlp = np.full(n_valid, np.inf)
    best_m = m.copy()
    best_sigma_sq = sigma_sq.copy()

    converged = np.zeros(n_valid, dtype=bool)
    last_iteration = 0

    for sigma_iter in range(n_noise_iterations):
        tau = 1.0 / np.maximum(sigma_sq, 1e-30)

        # Reset per-iteration state (NLP values from previous sigma are
        # invalid because tau changed)
        best_nlp[:] = np.inf
        active = ~converged.copy()
        step_scale = np.ones(n_valid)
        nlp_plateau_count = np.zeros(n_valid, dtype=np.int32)
        prev_nlp = np.full(n_valid, np.inf)

        for iteration in range(n_iterations):
            last_iteration = sigma_iter * n_iterations + iteration
            n_active = int(np.sum(active))
            if n_active == 0:
                break

            a_idx = np.where(active)[0]
            m_a = m[a_idx]
            y_a = y[a_idx]
            tau_a = tau[a_idx]

            # 1. Inverse-transform
            m_nat_a = np.zeros_like(m_a)
            for j in range(n_params):
                m_nat_a[:, j] = from_transformed_batch(
                    m_a[:, j], transforms[j], upper_bounds[j],
                )

            # 2. Forward model
            g_m = evaluate_fn(m_nat_a)

            # 3. Jacobian
            J_columns = _compute_jacobian_columns(
                m_a, m_nat_a, g_m, evaluate_fn,
                transforms, upper_bounds, bounds_lo, bounds_hi, cbf_idx,
                fitting_param_names=fitting_param_names,
            )

            # 4. Residual
            k_res = y_a - g_m

            # 5. JtWJ
            JtWJ, _JtWk = _assemble_JtWJ(J_columns, k_res, w)

            # 6. Posterior precision and covariance
            post_prec = prior_prec[None, :, :] + tau_a[:, None, None] * JtWJ
            post_cov = np.linalg.inv(post_prec)
            bad_cov = ~np.isfinite(post_cov).all(axis=(1, 2))
            if np.any(bad_cov):
                post_prec[bad_cov] += 1e-6 * np.eye(n_params)
                post_cov[bad_cov] = np.linalg.inv(post_prec[bad_cov])

            # 7. Gauss-Newton mean update
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
            m_new = np.einsum("nij,nj->ni", post_cov, rhs)

            # Clip IDENTITY params
            for j in range(n_params):
                if transforms[j] == TransformType.IDENTITY:
                    m_new[:, j] = np.clip(m_new[:, j], bounds_lo[j], bounds_hi[j])

            # 8. Compute negative log posterior
            m_new_nat = np.zeros_like(m_new)
            for j in range(n_params):
                m_new_nat[:, j] = from_transformed_batch(
                    m_new[:, j], transforms[j], upper_bounds[j],
                )
            g_new = evaluate_fn(m_new_nat)
            k_new = y_a - g_new
            weighted_rss = np.sum(w[None, :] * k_new ** 2, axis=1)
            dm = m_new - prior_means_t[None, :]
            prior_term = np.sum(dm * (dm @ prior_prec), axis=1)
            nlp = 0.5 * tau_a * weighted_rss + 0.5 * prior_term

            # 9. Step-size reduction (objective-based)
            cur_best = best_nlp[a_idx].copy()
            improved = nlp < cur_best

            imp_local = np.where(improved)[0]
            if len(imp_local) > 0:
                imp_global = a_idx[imp_local]
                best_nlp[imp_global] = nlp[imp_local]
                best_m[imp_global] = m_new[imp_local]
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
                for j in range(n_params):
                    if transforms[j] == TransformType.IDENTITY:
                        m_new[not_imp_local, j] = np.clip(
                            m_new[not_imp_local, j], bounds_lo[j], bounds_hi[j],
                        )
                exhausted = step_scale[not_imp_global] < 1e-4
                if np.any(exhausted):
                    active[not_imp_global[exhausted]] = False

            # 10. Convergence: param change OR NLP plateau
            if iteration > 0:
                param_change = np.max(np.abs(m_new - m_a), axis=1)
                param_conv = param_change < 1e-3

                cur_prev = prev_nlp[a_idx]
                nlp_near = (
                    np.isfinite(cur_prev)
                    & (np.abs(nlp - cur_prev) < 1e-4 * np.maximum(np.abs(cur_prev), 1.0))
                )
                nlp_plateau_count[a_idx] = np.where(
                    nlp_near, nlp_plateau_count[a_idx] + 1, 0,
                )
                nlp_conv = nlp_plateau_count[a_idx] >= 2

                newly_conv = param_conv | nlp_conv
                if np.any(newly_conv):
                    conv_global = a_idx[newly_conv]
                    converged[conv_global] = True
                    active[conv_global] = False

            prev_nlp[a_idx] = nlp

            # Update state for still-active voxels
            still_active = active[a_idx]
            if np.any(still_active):
                sa_local = np.where(still_active)[0]
                sa_global = a_idx[sa_local]
                m[sa_global] = m_new[sa_local]

        # Profile likelihood noise update
        if sigma_iter < n_noise_iterations - 1:
            best_nat = np.zeros_like(best_m)
            for j in range(n_params):
                best_nat[:, j] = from_transformed_batch(
                    best_m[:, j], transforms[j], upper_bounds[j],
                )
            g_best = evaluate_fn(best_nat)
            residuals = y - g_best
            sigma_sq = np.sum(w[None, :] * residuals ** 2, axis=1) / n_eff
            sigma_sq = np.maximum(sigma_sq, 1e-30)
            best_sigma_sq = sigma_sq.copy()

            # Warm-start next outer iteration from best solution
            m = best_m.copy()
            converged[:] = False

    # Final sigma from best solution
    best_nat_final = np.zeros_like(best_m)
    for j in range(n_params):
        best_nat_final[:, j] = from_transformed_batch(
            best_m[:, j], transforms[j], upper_bounds[j],
        )
    g_final = evaluate_fn(best_nat_final)
    residuals_final = y - g_final
    best_sigma_sq = np.sum(w[None, :] * residuals_final ** 2, axis=1) / n_eff
    best_sigma_sq = np.maximum(best_sigma_sq, 1e-30)

    # Re-evaluate NLP at best solution with final sigma
    weighted_rss_final = np.sum(w[None, :] * residuals_final ** 2, axis=1)
    dm_final = best_m - prior_means_t[None, :]
    prior_term_final = np.sum(dm_final * (dm_final @ prior_prec), axis=1)
    best_nlp = 0.5 / best_sigma_sq * weighted_rss_final + 0.5 * prior_term_final

    # Mark unconverged voxels that are at a good solution
    good_fit = _compute_r_squared(y, g_final) > 0.0
    converged |= good_fit

    return best_m, best_nlp, best_sigma_sq, converged, last_iteration


def batched_dsc_map_fit(
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
    cpi_t_terminal: Optional[float] = None,
    nlls_seeds: Optional[npt.NDArray[np.floating]] = None,
    prior_means_transformed: Optional[npt.NDArray[np.floating]] = None,
    prior_stds_transformed: Optional[npt.NDArray[np.floating]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    n_iterations: int = 15,
    n_noise_iterations: int = 3,
    sloppy: bool = False,
    save_fitted_curves: bool = False,
    compute_uncertainties: bool = True,
) -> DSCBatchedFitResult:
    """Batched MAP fitting for DSC perfusion models.

    Implements Gauss-Newton MAP optimization with profile likelihood noise
    estimation, vectorized across all voxels. Uses the same forward models
    and Jacobian infrastructure as batched VI.

    Does not support spatial VB (SVB).

    Args:
        model_type: 'cpi' or 'vascular'.
        tissue_curves: Observed tissue curves, shape (N, T).
        aif: Arterial input function, shape (T,).
        time_points: Time points (seconds), shape (T,).
        prior_means_nat: Prior means in natural space, shape (P,).
        prior_stds_nat: Prior stds in natural space, shape (P,).
        transforms: Per-parameter transform types, length P.
        upper_bounds: Per-parameter upper bounds for LOG_LOGIT, length P.
        fitting_param_names: Parameter names in fitting space, length P.
        fitting_bounds: Per-parameter bounds in fitting space, length P.
        mono_prec: CPI monotonicity precision matrix, shape (P, P) or None.
        mono_eta: CPI monotonicity bias, shape (P,) or None.
        control_point_times: CPI control point times, shape (K,) or None.
        toeplitz_matrix: Precomputed Toeplitz matrix, shape (T, T) or None.
        nlls_seeds: NLLS seed parameters for multi-start init, shape (N, P)
            or None. Used as second start point alongside prior means.
        prior_means_transformed: Pre-computed prior means in transformed space,
            shape (P,). Bypasses internal transform_prior() call.
        prior_stds_transformed: Pre-computed prior stds in transformed space,
            shape (P,). Must be provided alongside prior_means_transformed.
        weights: Per-frame motion weights, shape (T,). None = uniform.
        n_iterations: Gauss-Newton iterations per noise loop (default 15).
        n_noise_iterations: Profile likelihood outer loops (default 3).
        sloppy: Fast mode: 8 iters, 1 noise loop, no uncertainties.
        save_fitted_curves: Whether to store fitted curves in result.
        compute_uncertainties: Whether to compute Laplace uncertainties.

    Returns:
        DSCBatchedFitResult with fitted parameters, R², convergence,
        and Laplace uncertainties.
    """
    tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)
    n_voxels, n_obs = tissue_curves.shape
    n_params = len(fitting_param_names)

    if sloppy:
        n_iterations = 8
        n_noise_iterations = 1
        compute_uncertainties = False

    # Pre-filter: exclude voxels with negligible signal
    voxel_max = np.max(np.abs(tissue_curves), axis=1)
    valid_mask = voxel_max > EPSILON_CONCENTRATION
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    # Initialize output arrays
    out_params = {name: np.zeros(n_voxels) for name in fitting_param_names}
    out_r2 = np.full(n_voxels, -1.0)
    out_converged = np.zeros(n_voxels, dtype=bool)
    out_uncertainties: dict[str, npt.NDArray[np.floating]] = {
        name: np.full(n_voxels, np.nan) for name in fitting_param_names
    }
    out_uncertainties["noise_std"] = np.full(n_voxels, np.nan)

    if n_valid == 0:
        return DSCBatchedFitResult(
            params=out_params, r_squared=out_r2, converged=out_converged,
            fitted_curves=None, n_iterations=0, model_name=model_type,
            uncertainties=out_uncertainties,
        )

    y = tissue_curves[valid_idx]  # (n_valid, T)

    # --- Prior setup ---
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
    prior_prec_mat = np.diag(1.0 / (prior_stds_t ** 2))
    if mono_prec is not None:
        prior_prec_mat = prior_prec_mat + mono_prec

    # --- Bounds ---
    bounds_lo = np.array([b[0] for b in fitting_bounds])
    bounds_hi = np.array([b[1] for b in fitting_bounds])

    w = np.ones(n_obs, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    n_eff = float(np.sum(w)) if weights is not None else float(n_obs)

    # CBF index for analytic Jacobian
    cbf_idx = fitting_param_names.index("cbf")
    if transforms[cbf_idx] not in (TransformType.LOG, TransformType.IDENTITY):
        raise ValueError(
            f"Analytic CBF Jacobian requires LOG or IDENTITY transform, "
            f"got {transforms[cbf_idx]} for param '{fitting_param_names[cbf_idx]}'"
        )

    # --- Forward model dispatch ---
    def _evaluate(params_fitting: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        if model_type == "vascular":
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
                t_terminal=cpi_t_terminal,
            )

    # --- Multi-start initialization ---
    # Start 0: prior means
    m0_nat = np.tile(prior_means_nat, (n_valid, 1)).astype(np.float64)
    for j in range(n_params):
        m0_nat[:, j] = np.clip(m0_nat[:, j], bounds_lo[j], bounds_hi[j])
    m0_t = np.zeros((n_valid, n_params), dtype=np.float64)
    for j in range(n_params):
        m0_t[:, j] = to_transformed_batch(m0_nat[:, j], transforms[j], upper_bounds[j])

    # Initial noise from prior-mean residuals
    try:
        g0 = _evaluate(m0_nat)
        sigma_sq_init = np.sum(w[None, :] * (y - g0) ** 2, axis=1) / n_eff
        sigma_sq_init = np.maximum(sigma_sq_init, 1e-30)
    except (ValueError, RuntimeError):
        sigma_sq_init = np.ones(n_valid, dtype=np.float64)

    # Start 1: NLLS seeds (if available and different from prior)
    has_second_start = False
    if not sloppy and nlls_seeds is not None:
        valid_seeds = nlls_seeds[valid_idx]
        has_seed = np.any(np.abs(valid_seeds) > 0, axis=1)
        if np.any(has_seed):
            has_second_start = True
            m1_nat = m0_nat.copy()
            m1_nat[has_seed] = valid_seeds[has_seed]
            for j in range(n_params):
                m1_nat[:, j] = np.clip(m1_nat[:, j], bounds_lo[j], bounds_hi[j])
            m1_t = np.zeros((n_valid, n_params), dtype=np.float64)
            for j in range(n_params):
                m1_t[:, j] = to_transformed_batch(m1_nat[:, j], transforms[j], upper_bounds[j])

            # Noise from seed residuals
            try:
                g1 = _evaluate(m1_nat)
                sigma_sq_seed = np.sum(w[None, :] * (y - g1) ** 2, axis=1) / n_eff
                sigma_sq_seed = np.maximum(sigma_sq_seed, 1e-30)
            except (ValueError, RuntimeError):
                sigma_sq_seed = sigma_sq_init.copy()

    # For sloppy with NLLS seeds, use seeds as single start
    if sloppy and nlls_seeds is not None:
        valid_seeds = nlls_seeds[valid_idx]
        has_seed = np.any(np.abs(valid_seeds) > 0, axis=1)
        if np.any(has_seed):
            m0_nat[has_seed] = valid_seeds[has_seed]
            for j in range(n_params):
                m0_nat[:, j] = np.clip(m0_nat[:, j], bounds_lo[j], bounds_hi[j])
            for j in range(n_params):
                m0_t[:, j] = to_transformed_batch(m0_nat[:, j], transforms[j], upper_bounds[j])
            try:
                g0 = _evaluate(m0_nat)
                sigma_sq_init = np.sum(w[None, :] * (y - g0) ** 2, axis=1) / n_eff
                sigma_sq_init = np.maximum(sigma_sq_init, 1e-30)
            except (ValueError, RuntimeError):
                pass

    logger.info("DSC batched MAP: fitting %s on %d voxels (multi-start=%s)",
                model_type, n_valid, has_second_start)

    # --- Run start 0 ---
    best_m, best_nlp, best_sigma_sq, converged, total_iters = _run_map_gauss_newton(
        y, m0_t, sigma_sq_init, prior_means_t, prior_prec_mat, mono_eta,
        transforms, upper_bounds, bounds_lo, bounds_hi, cbf_idx, w, n_eff,
        _evaluate, n_iterations, n_noise_iterations,
        fitting_param_names=fitting_param_names,
    )

    # --- Run start 1 (if available) ---
    if has_second_start:
        m1_best, nlp1, sigma1, conv1, iters1 = _run_map_gauss_newton(
            y, m1_t, sigma_sq_seed, prior_means_t, prior_prec_mat, mono_eta,
            transforms, upper_bounds, bounds_lo, bounds_hi, cbf_idx, w, n_eff,
            _evaluate, n_iterations, n_noise_iterations,
        )
        total_iters = max(total_iters, iters1)

        # Select better start per voxel
        start1_better = nlp1 < best_nlp
        if np.any(start1_better):
            best_m[start1_better] = m1_best[start1_better]
            best_nlp[start1_better] = nlp1[start1_better]
            best_sigma_sq[start1_better] = sigma1[start1_better]
            converged[start1_better] |= conv1[start1_better]

    logger.info(
        "DSC batched MAP finished: %d/%d converged (of %d total)",
        int(np.sum(converged)), n_valid, n_voxels,
    )

    # --- Post-convergence: inverse-transform ---
    fitted_nat = np.zeros((n_valid, n_params), dtype=np.float64)
    for j in range(n_params):
        fitted_nat[:, j] = from_transformed_batch(
            best_m[:, j], transforms[j], upper_bounds[j],
        )
    for j in range(n_params):
        fitted_nat[:, j] = np.clip(fitted_nat[:, j], bounds_lo[j], bounds_hi[j])

    # --- Laplace uncertainties ---
    if compute_uncertainties:
        final_tau = 1.0 / np.maximum(best_sigma_sq, 1e-30)
        g_final_unc = _evaluate(fitted_nat)
        J_final = _compute_jacobian_columns(
            best_m, fitted_nat, g_final_unc, _evaluate,
            transforms, upper_bounds, bounds_lo, bounds_hi, cbf_idx,
            fitting_param_names=fitting_param_names,
        )
        JtWJ_final, _ = _assemble_JtWJ(J_final, y - g_final_unc, w)

        H = prior_prec_mat[None, :, :] + final_tau[:, None, None] * JtWJ_final
        cov = np.linalg.inv(H)
        bad_cov = ~np.isfinite(cov).all(axis=(1, 2))
        if np.any(bad_cov):
            H[bad_cov] += 1e-6 * np.eye(n_params)
            cov[bad_cov] = np.linalg.inv(H[bad_cov])

        for j, name in enumerate(fitting_param_names):
            std_t = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2)[:, j], 0.0))
            jac = np.exp(log_jacobian_det(fitted_nat[:, j], transforms[j], upper_bounds[j]))
            out_uncertainties[name][valid_idx] = std_t * jac

    out_uncertainties["noise_std"][valid_idx] = np.sqrt(best_sigma_sq)

    # --- R² and scatter results ---
    final_pred = _evaluate(fitted_nat)
    r2_valid = _compute_r_squared(y, final_pred)

    for j, name in enumerate(fitting_param_names):
        out_params[name][valid_idx] = fitted_nat[:, j]
    out_r2[valid_idx] = r2_valid
    out_converged[valid_idx] = converged

    out_fitted = None
    if save_fitted_curves:
        out_fitted = np.zeros((n_voxels, n_obs), dtype=np.float64)
        out_fitted[valid_idx] = final_pred

    return DSCBatchedFitResult(
        params=out_params, r_squared=out_r2, converged=out_converged,
        fitted_curves=out_fitted, n_iterations=total_iters + 1,
        model_name=model_type, uncertainties=out_uncertainties,
    )
