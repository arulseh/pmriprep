"""Spatial Variational Bayes for DSC perfusion (Groves et al. 2009).

Wraps the standard batched VI to add MRF spatial priors that couple
neighboring voxels' parameters, reducing noise without explicit smoothing.
"""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import sparse

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import DSCBatchedFitResult, batched_dsc_vi_fit

logger = logging.getLogger(__name__)

_INNER_ITERATIONS = 4


def _build_neighborhood_graph(
    mask_coords: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    spatial_shape: tuple[int, int, int],
) -> tuple[sparse.csr_matrix, npt.NDArray[np.int64]]:
    """Build sparse 6-connected adjacency from mask coordinates.

    Args:
        mask_coords: (x, y, z) arrays of masked voxel coordinates.
        spatial_shape: (nx, ny, nz) volume dimensions.

    Returns:
        Tuple of (adjacency CSR matrix (N×N), degree vector (N,)).
    """
    n_voxels = len(mask_coords[0])
    nx, ny, nz = spatial_shape

    ijk = np.column_stack(mask_coords)
    flat_idx = ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2]

    flat_to_voxel = np.full(nx * ny * nz, -1, dtype=np.int64)
    flat_to_voxel[flat_idx] = np.arange(n_voxels)

    offsets = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=np.int64)

    rows = []
    cols = []
    for d in offsets:
        nbr = ijk + d[None, :]
        valid = (
            (nbr[:, 0] >= 0) & (nbr[:, 0] < nx)
            & (nbr[:, 1] >= 0) & (nbr[:, 1] < ny)
            & (nbr[:, 2] >= 0) & (nbr[:, 2] < nz)
        )
        nbr_flat = np.where(valid, nbr[:, 0] * (ny * nz) + nbr[:, 1] * nz + nbr[:, 2], 0)
        nbr_voxel = np.where(valid, flat_to_voxel[nbr_flat], -1)
        has_nbr = nbr_voxel >= 0
        src = np.where(has_nbr)[0]
        dst = nbr_voxel[has_nbr]
        rows.append(src)
        cols.append(dst)

    rows_all = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    cols_all = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    data = np.ones(len(rows_all), dtype=np.float64)
    adj = sparse.csr_matrix((data, (rows_all, cols_all)), shape=(n_voxels, n_voxels))

    degree = np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)
    return adj, degree


def _compute_spatial_contributions(
    posteriors: npt.NDArray[np.floating],
    adj: sparse.csr_matrix,
    degree: npt.NDArray[np.int64],
    aK: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute per-voxel spatial precision and eta under FABBER M-prior.

    M-prior (Penny et al. 2005): spatial precision = aK * (nn + 1e-8),
    eta = aK * (adj @ posteriors), where nn is the neighbour count.

    Args:
        posteriors: Posterior means in transformed space, (N, P).
        adj: Sparse adjacency matrix (N, N).
        degree: Degree vector (N,).
        aK: Per-parameter spatial precision hyperparameter, (P,).

    Returns:
        (spatial_prec_diag (N, P), spatial_eta (N, P)).
    """
    n_voxels, n_params = posteriors.shape
    prec_diag = np.zeros((n_voxels, n_params), dtype=np.float64)
    eta = np.zeros((n_voxels, n_params), dtype=np.float64)

    L_diag = degree.astype(np.float64) + 1e-8
    for k in range(n_params):
        nbr_sum = np.asarray(adj @ posteriors[:, k]).ravel()
        prec_diag[:, k] = aK[k] * L_diag
        eta[:, k] = aK[k] * nbr_sum

    return prec_diag, eta


def _update_aK_penny(
    posteriors: npt.NDArray[np.floating],
    cov_diag: npt.NDArray[np.floating],
    adj: sparse.csr_matrix,
    degree: npt.NDArray[np.int64],
    q1: float = 10.0,
    q2: float = 1.0,
) -> npt.NDArray[np.float64]:
    """FABBER M-prior aK closed-form update (Penny et al. 2005).

    Posterior on the spatial-precision hyperparameter aK is Gamma; the
    update has the closed form:
        gK = 1 / (0.5 * trace(L Σ_k) + 0.5 * m_k^T L m_k + 1/q1)
        hK = 0.5 * N + q2
        aK = gK * hK

    where L = diag(nn + 1e-8) - adj is the graph Laplacian.

    Args:
        posteriors: Posterior means in transformed space, (N, P).
        cov_diag: Posterior covariance diagonal, (N, P).
        adj: Sparse adjacency matrix (N, N).
        degree: Degree vector (N,).
        q1: Gamma hyperprior shape (FABBER default 10).
        q2: Gamma hyperprior rate (FABBER default 1).

    Returns:
        Updated aK values, (P,).
    """
    n_voxels, n_params = posteriors.shape
    L_diag = degree.astype(np.float64) + 1e-8

    aK = np.zeros(n_params, dtype=np.float64)
    for k in range(n_params):
        m_k = posteriors[:, k]
        Lm = L_diag * m_k - np.asarray(adj @ m_k).ravel()
        trace_term = 0.5 * float(np.sum(L_diag * cov_diag[:, k]))
        term2 = 0.5 * float(m_k @ Lm)
        gK = 1.0 / max(trace_term + term2 + 1.0 / q1, 1e-30)
        hK = 0.5 * n_voxels + q2
        aK[k] = gK * hK

    return aK


def batched_dsc_spatial_vi_fit(
    model_type: str,
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    mask_coords: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    spatial_shape: tuple[int, int, int],
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
    sloppy: bool = False,
    save_fitted_curves: bool = False,
    n_spatial_sweeps: int = 20,
    spatial_tol: float = 0.01,
    aK_init: float = 10.0,
) -> DSCBatchedFitResult:
    """Spatial VB wrapper around batched VI.

    Iterates spatial sweeps, updating MRF neighbor priors between calls
    to the standard VB function. Convergence is monitored via relative
    lambda change at the outer level.

    Args:
        model_type: 'cpi' or 'vascular'.
        tissue_curves: Observed tissue curves, shape (N, T).
        aif: Arterial input function, shape (T,).
        time_points: Time points (seconds), shape (T,).
        mask_coords: (x, y, z) coordinate arrays for masked voxels.
        spatial_shape: (nx, ny, nz) volume dimensions.
        prior_means_nat: Prior means in natural space, shape (P,).
        prior_stds_nat: Prior stds in natural space, shape (P,).
        transforms: Per-parameter transform types.
        upper_bounds: Per-parameter upper bounds for LOG_LOGIT.
        fitting_param_names: Parameter names in fitting space.
        fitting_bounds: Per-parameter bounds in fitting space.
        mono_prec: Monotonicity precision matrix or None.
        mono_eta: Monotonicity eta correction or None.
        control_point_times: CPI control point times or None.
        toeplitz_matrix: Precomputed Toeplitz matrix or None.
        nlls_seeds: NLLS seed parameters or None.
        prior_means_transformed: Pre-computed transformed prior means.
        prior_stds_transformed: Pre-computed transformed prior stds.
        weights: Per-frame motion weights or None.
        sloppy: Use fast settings (fewer sweeps and inner iterations).
        save_fitted_curves: Store fitted curves in result.
        n_spatial_sweeps: Maximum number of spatial sweeps.
        spatial_tol: Relative lambda convergence tolerance.
        lambda_init: Initial lambda value for all parameters.

    Returns:
        DSCBatchedFitResult with fitted parameters from spatial VB.
    """
    n_params = len(fitting_param_names)

    if sloppy:
        n_spatial_sweeps = min(n_spatial_sweeps, 6)

    adj, degree = _build_neighborhood_graph(mask_coords, spatial_shape)
    has_any_neighbors = (degree > 0).any()

    if not has_any_neighbors:
        logger.info("SVB: no voxel neighbors found, falling back to standard VI")
        return batched_dsc_vi_fit(
            model_type=model_type,
            tissue_curves=tissue_curves,
            aif=aif,
            time_points=time_points,
            prior_means_nat=prior_means_nat,
            prior_stds_nat=prior_stds_nat,
            transforms=transforms,
            upper_bounds=upper_bounds,
            fitting_param_names=fitting_param_names,
            fitting_bounds=fitting_bounds,
            mono_prec=mono_prec,
            mono_eta=mono_eta,
            control_point_times=control_point_times,
            toeplitz_matrix=toeplitz_matrix,
            cpi_t_terminal=cpi_t_terminal,
            nlls_seeds=nlls_seeds,
            prior_means_transformed=prior_means_transformed,
            prior_stds_transformed=prior_stds_transformed,
            weights=weights,
            sloppy=sloppy,
            save_fitted_curves=save_fitted_curves,
        )

    aK = np.full(n_params, aK_init, dtype=np.float64)
    inner_iters = _INNER_ITERATIONS if not sloppy else 2

    # Initial standard VI run to get starting posteriors
    logger.info("SVB: initial VI sweep (no spatial prior)")
    result = batched_dsc_vi_fit(
        model_type=model_type,
        tissue_curves=tissue_curves,
        aif=aif,
        time_points=time_points,
        prior_means_nat=prior_means_nat,
        prior_stds_nat=prior_stds_nat,
        transforms=transforms,
        upper_bounds=upper_bounds,
        fitting_param_names=fitting_param_names,
        fitting_bounds=fitting_bounds,
        mono_prec=mono_prec,
        mono_eta=mono_eta,
        control_point_times=control_point_times,
        toeplitz_matrix=toeplitz_matrix,
        cpi_t_terminal=cpi_t_terminal,
        nlls_seeds=nlls_seeds,
        prior_means_transformed=prior_means_transformed,
        prior_stds_transformed=prior_stds_transformed,
        weights=weights,
        sloppy=sloppy,
        save_fitted_curves=False,
        return_posteriors=True,
    )
    posteriors = result.posterior_means_transformed
    cov_diag = result.posterior_cov_diag
    assert posteriors is not None and cov_diag is not None

    for sweep in range(n_spatial_sweeps):
        aK_old = aK.copy()
        aK = _update_aK_penny(posteriors, cov_diag, adj, degree)
        sp_prec, sp_eta = _compute_spatial_contributions(posteriors, adj, degree, aK)

        logger.debug(
            "SVB sweep %d/%d: aK = %s",
            sweep + 1, n_spatial_sweeps,
            np.array2string(aK, precision=3, separator=", "),
        )

        save_curves = save_fitted_curves and (sweep == n_spatial_sweeps - 1)
        result = batched_dsc_vi_fit(
            model_type=model_type,
            tissue_curves=tissue_curves,
            aif=aif,
            time_points=time_points,
            prior_means_nat=prior_means_nat,
            prior_stds_nat=prior_stds_nat,
            transforms=transforms,
            upper_bounds=upper_bounds,
            fitting_param_names=fitting_param_names,
            fitting_bounds=fitting_bounds,
            mono_prec=mono_prec,
            mono_eta=mono_eta,
            control_point_times=control_point_times,
            toeplitz_matrix=toeplitz_matrix,
            cpi_t_terminal=cpi_t_terminal,
            nlls_seeds=None,
            prior_means_transformed=prior_means_transformed,
            prior_stds_transformed=prior_stds_transformed,
            weights=weights,
            save_fitted_curves=save_curves,
            spatial_prec_diag=sp_prec,
            spatial_eta=sp_eta,
            return_posteriors=True,
            n_iterations_override=inner_iters,
        )
        posteriors = result.posterior_means_transformed
        cov_diag = result.posterior_cov_diag
        assert posteriors is not None and cov_diag is not None

        rel_change = np.max(np.abs(aK - aK_old) / np.maximum(aK, 1e-10))
        if rel_change < spatial_tol:
            logger.info("SVB converged at sweep %d (rel_change=%.4f)", sweep + 1, rel_change)
            break
    else:
        logger.info(
            "SVB reached max sweeps (%d), final rel_change=%.4f",
            n_spatial_sweeps, rel_change,
        )

    if save_fitted_curves and result.fitted_curves is None:
        final = batched_dsc_vi_fit(
            model_type=model_type,
            tissue_curves=tissue_curves,
            aif=aif,
            time_points=time_points,
            prior_means_nat=prior_means_nat,
            prior_stds_nat=prior_stds_nat,
            transforms=transforms,
            upper_bounds=upper_bounds,
            fitting_param_names=fitting_param_names,
            fitting_bounds=fitting_bounds,
            mono_prec=mono_prec,
            mono_eta=mono_eta,
            control_point_times=control_point_times,
            toeplitz_matrix=toeplitz_matrix,
            cpi_t_terminal=cpi_t_terminal,
            nlls_seeds=None,
            prior_means_transformed=prior_means_transformed,
            prior_stds_transformed=prior_stds_transformed,
            weights=weights,
            save_fitted_curves=True,
            spatial_prec_diag=sp_prec,
            spatial_eta=sp_eta,
            n_iterations_override=inner_iters,
        )
        result = final

    return result
