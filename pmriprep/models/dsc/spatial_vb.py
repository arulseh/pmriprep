"""Spatial Variational Bayes for DSC perfusion.

Wraps the standard batched VI to add MRF spatial priors that couple
neighboring voxels' parameters, reducing noise without explicit smoothing.
"""

import logging
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy import sparse

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.batched_fitting import DSCBatchedFitResult, batched_dsc_vi_fit
from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter
from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION

logger = logging.getLogger("nipype.interface")

def _build_neighborhood_graph(
    mask_coords: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    spatial_shape: tuple[int, int, int],
) -> tuple[sparse.csr_matrix, npt.NDArray[np.int64]]:
    """Build sparse 6-connected adjacency from mask coordinates.

    Parameters
    ----------
    mask_coords : tuple[npt.NDArray, npt.NDArray, npt.NDArray]
        (x, y, z) arrays of masked voxel coordinates.
    spatial_shape : tuple[int, int, int]
        (nx, ny, nz) volume dimensions.

    Returns
    -------
    tuple[sparse.csr_matrix, npt.NDArray[np.int64]]
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

    M-prior: spatial precision = aK * (nn + 1e-8),
    eta = aK * (adj @ posteriors), where nn is the neighbour count.

    Parameters
    ----------
    posteriors : npt.NDArray[np.floating]
        Posterior means in transformed space, (N, P).
    adj : sparse.csr_matrix
        Sparse adjacency matrix (N, N).
    degree : npt.NDArray[np.int64]
        Degree vector (N,).
    aK : npt.NDArray[np.floating]
        Per-parameter spatial precision hyperparameter, (P,).

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
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


def _exclude_inadmissible_neighbors(
    adj: sparse.csr_matrix,
    fit_admissible: npt.NDArray[np.bool_],
) -> tuple[sparse.csr_matrix, npt.NDArray[np.int64]]:
    """Remove fit-inadmissible voxels from every spatial neighbor list.

    Parameters
    ----------
    adj : sparse.csr_matrix
        Full masked-voxel adjacency.
    fit_admissible : npt.NDArray[np.bool_]
        Boolean fit-admissibility vector, shape (N,).

    Returns
    -------
    tuple[sparse.csr_matrix, npt.NDArray[np.int64]]
        Pruned adjacency and its updated degree vector.
    """
    keep = sparse.diags(np.asarray(fit_admissible, dtype=np.float64))
    pruned = (keep @ adj @ keep).tocsr()
    pruned.eliminate_zeros()
    degree = np.asarray(pruned.sum(axis=1)).ravel().astype(np.int64)
    return pruned, degree


def _update_aK_penny(
    posteriors: npt.NDArray[np.floating],
    cov_diag: npt.NDArray[np.floating],
    adj: sparse.csr_matrix,
    degree: npt.NDArray[np.int64],
    q1: float = 10.0,
    q2: float = 1.0,
) -> npt.NDArray[np.float64]:
    """FABBER M-prior aK closed-form update.

    Posterior on the spatial-precision hyperparameter aK is Gamma; the
    update has the closed form:
        gK = 1 / (0.5 * trace(L Σ_k) + 0.5 * m_k^T L m_k + 1/q1)
        hK = 0.5 * N + q2
        aK = gK * hK

    where L = diag(nn + 1e-8) - adj is the graph Laplacian.

    Parameters
    ----------
    posteriors : npt.NDArray[np.floating]
        Posterior means in transformed space, (N, P).
    cov_diag : npt.NDArray[np.floating]
        Posterior covariance diagonal, (N, P).
    adj : sparse.csr_matrix
        Sparse adjacency matrix (N, N).
    degree : npt.NDArray[np.int64]
        Degree vector (N,).
    q1 : float, optional
        Gamma hyperprior shape (FABBER default 10).
    q2 : float, optional
        Gamma hyperprior rate (FABBER default 1).

    Returns
    -------
    npt.NDArray[np.float64]
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
    nlls_seeds: Optional[npt.NDArray[np.floating]] = None,
    prior_means_transformed: Optional[npt.NDArray[np.floating]] = None,
    prior_stds_transformed: Optional[npt.NDArray[np.floating]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    sloppy: bool = False,
    save_fitted_curves: bool = False,
    n_spatial_sweeps: int = 10,
    spatial_tol: Optional[float] = None,
    model_adapter: DSCVIModelAdapter | None = None,
    return_full_posterior: bool = False,
    local_fit: Callable[..., DSCBatchedFitResult] | None = None,
    on_pass_start: Callable[[int, int], None] | None = None,
) -> DSCBatchedFitResult:
    """Spatial VB wrapper around batched VI.

    Warm-starts the vectorized solver with ordinary VI, then iterates stateful
    CBF sweeps using FABBER's boundary-corrected M-prior equations. Each
    spatial sweep performs one synchronous batched parameter/noise update and
    retains the posterior state. Nonlinear proposals are backtracked against
    the likelihood-plus-M-prior conditional objective before acceptance.

    Parameters
    ----------
    model_type : str
        Identifier of the supplied VI adapter.
    tissue_curves : npt.NDArray[np.floating]
        Observed tissue curves, shape (N, T).
    aif : npt.NDArray[np.floating]
        Arterial input function, shape (T,).
    time_points : npt.NDArray[np.floating]
        Time points (seconds), shape (T,).
    mask_coords : tuple[npt.NDArray, npt.NDArray, npt.NDArray]
        (x, y, z) coordinate arrays for masked voxels.
    spatial_shape : tuple[int, int, int]
        (nx, ny, nz) volume dimensions.
    prior_means_nat : npt.NDArray[np.floating]
        Prior means in natural space, shape (P,).
    prior_stds_nat : npt.NDArray[np.floating]
        Prior stds in natural space, shape (P,).
    transforms : list[TransformType]
        Per-parameter transform types.
    upper_bounds : list[float | None]
        Per-parameter upper bounds for LOG_LOGIT.
    fitting_param_names : list[str]
        Parameter names in fitting space.
    fitting_bounds : list[tuple[float, float]]
        Per-parameter bounds in fitting space.
    mono_prec : npt.NDArray[np.floating], optional
        Monotonicity precision matrix or None.
    mono_eta : npt.NDArray[np.floating], optional
        Monotonicity eta correction or None.
    nlls_seeds : npt.NDArray[np.floating], optional
        NLLS seed parameters or None.
    prior_means_transformed : npt.NDArray[np.floating], optional
        Pre-computed transformed prior means.
    prior_stds_transformed : npt.NDArray[np.floating], optional
        Pre-computed transformed prior stds.
    weights : npt.NDArray[np.floating], optional
        Per-frame motion weights or None.
    sloppy : bool, optional
        Use fast settings (fewer sweeps and inner iterations).
    save_fitted_curves : bool, optional
        Store fitted curves in result.
    n_spatial_sweeps : int, optional
        Maximum number of spatial sweeps.
    spatial_tol : float, optional
        Optional relative ``aK`` convergence tolerance. ``None`` (default)
        runs ten spatial sweeps after the ordinary-VI warm start.
    model_adapter : DSCVIModelAdapter, optional
        Model evaluations and mean domain passed to the shared batched fitter.
    return_full_posterior : bool, optional
        Retain complete local factors on the final result, not cross-voxel
        covariance. Default False avoids additional covariance storage.
    local_fit : callable, optional
        Explicit local-update dispatcher with the batched fitter's keyword
        contract, for models with voxel-specific independent priors. It must
        preserve row order and synchronous sweep state. None uses the shared
        batched fitter directly; the spatial algorithm is identical.
    on_pass_start : callable, optional
        Receive (pass index, configured sweeps) before each local update.
        Index zero denotes warm start; (0, 0) denotes the independent-VI pass
        on a graph without edges. The observer does not change the schedule.

    Returns
    -------
    DSCBatchedFitResult
        DSCBatchedFitResult with fitted parameters from spatial VB.
    """
    fit_local = batched_dsc_vi_fit if local_fit is None else local_fit
    n_params = len(fitting_param_names)
    spatial_parameter_mask = np.asarray(
        [name == "cbf" for name in fitting_param_names], dtype=bool,
    )
    if not np.any(spatial_parameter_mask):
        raise ValueError("SVB requires a fitted 'cbf' parameter")

    if n_spatial_sweeps < 1:
        raise ValueError("n_spatial_sweeps must be at least 1")

    if sloppy:
        n_spatial_sweeps = min(n_spatial_sweeps, 6)

    adj, degree = _build_neighborhood_graph(mask_coords, spatial_shape)
    finite_curves = np.nan_to_num(
        tissue_curves, nan=0.0, posinf=0.0, neginf=0.0,
    )
    fit_admissible = np.max(np.abs(finite_curves), axis=1) > EPSILON_CONCENTRATION
    adj, degree = _exclude_inadmissible_neighbors(adj, fit_admissible)
    has_any_neighbors = (degree > 0).any()

    if not has_any_neighbors:
        logger.info("SVB: no voxel neighbors found, falling back to standard VI")
        if on_pass_start is not None:
            on_pass_start(0, 0)
        return fit_local(
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
            nlls_seeds=nlls_seeds,
            prior_means_transformed=prior_means_transformed,
            prior_stds_transformed=prior_stds_transformed,
            weights=weights,
            sloppy=sloppy,
            save_fitted_curves=save_fitted_curves,
            model_adapter=model_adapter,
            return_full_posterior=return_full_posterior,
        )

    aK = np.zeros(n_params, dtype=np.float64)
    logger.info("SVB: ordinary-VI warm start")
    if on_pass_start is not None:
        on_pass_start(0, n_spatial_sweeps)
    started = perf_counter()
    result = fit_local(
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
        nlls_seeds=nlls_seeds,
        prior_means_transformed=prior_means_transformed,
        prior_stds_transformed=prior_stds_transformed,
        weights=weights,
        sloppy=sloppy,
        save_fitted_curves=False,
        return_posteriors=True,
        model_adapter=model_adapter,
    )
    logger.info("SVB warm start completed in %.1f s", perf_counter() - started)
    warm_means = result.posterior_means_transformed
    warm_cov_diag = result.posterior_cov_diag
    assert warm_means is not None and warm_cov_diag is not None
    posteriors = np.asarray(warm_means, dtype=np.float64)
    cov_diag = np.asarray(warm_cov_diag, dtype=np.float64)
    noise_std = result.uncertainties.get("noise_std")
    noise_precision: Optional[npt.NDArray[np.float64]] = None
    if noise_std is not None:
        noise_precision = np.zeros_like(noise_std, dtype=np.float64)
        valid_noise = np.isfinite(noise_std) & (noise_std > 0)
        noise_precision[valid_noise] = 1.0 / noise_std[valid_noise] ** 2

    sweep_seconds: list[float] = []
    for sweep in range(n_spatial_sweeps):
        aK_old = aK.copy()
        updated_aK = _update_aK_penny(posteriors, cov_diag, adj, degree)
        aK[spatial_parameter_mask] = updated_aK[spatial_parameter_mask]
        sp_prec, sp_eta = _compute_spatial_contributions(
            posteriors, adj, degree, aK
        )

        if sweep_seconds:
            remaining = sum(sweep_seconds) / len(sweep_seconds) * (n_spatial_sweeps - sweep)
            logger.info("SVB sweep %d/%d; estimated remaining %.1f s",
                        sweep + 1, n_spatial_sweeps, remaining)
        else:
            logger.info("SVB sweep %d/%d", sweep + 1, n_spatial_sweeps)
        logger.debug(
            "SVB sweep %d/%d: aK = %s",
            sweep + 1, n_spatial_sweeps,
            np.array2string(aK, precision=3, separator=", "),
        )

        if on_pass_start is not None:
            on_pass_start(sweep + 1, n_spatial_sweeps)
        started = perf_counter()
        result = fit_local(
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
            nlls_seeds=None,
            prior_means_transformed=prior_means_transformed,
            prior_stds_transformed=prior_stds_transformed,
            weights=weights,
            save_fitted_curves=save_fitted_curves,
            spatial_prec_diag=sp_prec,
            spatial_eta=sp_eta,
            spatial_replaces_parameter_prior=True,
            spatial_parameter_mask=spatial_parameter_mask,
            initial_means_transformed=posteriors,
            initial_cov_diag=cov_diag,
            initial_noise_precision=noise_precision,
            return_posteriors=True,
            n_iterations_override=1,
            model_adapter=model_adapter,
            return_full_posterior=return_full_posterior,
        )
        sweep_seconds.append(perf_counter() - started)
        logger.info("SVB sweep %d/%d completed in %.1f s",
                    sweep + 1, n_spatial_sweeps, sweep_seconds[-1])
        result_means = result.posterior_means_transformed
        result_cov_diag = result.posterior_cov_diag
        assert result_means is not None and result_cov_diag is not None
        posteriors = np.asarray(result_means, dtype=np.float64)
        cov_diag = np.asarray(result_cov_diag, dtype=np.float64)
        noise_std = result.uncertainties.get("noise_std")
        if noise_std is not None:
            noise_precision = np.zeros_like(noise_std, dtype=np.float64)
            valid_noise = np.isfinite(noise_std) & (noise_std > 0)
            noise_precision[valid_noise] = 1.0 / noise_std[valid_noise] ** 2

        rel_change = np.max(np.abs(aK - aK_old) / np.maximum(aK, 1e-10))
        if spatial_tol is not None and sweep > 0 and rel_change < spatial_tol:
            logger.info("SVB converged at sweep %d (rel_change=%.4f)", sweep + 1, rel_change)
            break
    else:
        logger.info(
            "SVB reached max sweeps (%d), final rel_change=%.4f",
            n_spatial_sweeps, rel_change,
        )

    result.n_iterations = sweep + 1
    return result
