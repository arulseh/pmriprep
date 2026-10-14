"""DSC deconvolution methods.

SVD-based deconvolution methods for DSC-MRI perfusion quantification,
including standard SVD (sSVD), circulant SVD (cSVD), oscillation-index
limited SVD (oSVD), and Tikhonov regularization.

The deconvolution problem is:
    C_tissue(t) = CBF × [R(t) ⊗ AIF(t)]

where R(t) is the fraction of tracer remaining in tissue at time t.
The solvers return the flow-scaled residue h(t) = CBF × R(t).
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import linalg

from pmriprep.utils.dsc_constants import (
    DEFAULT_LAMBDA_RANGE,
    DEFAULT_OI_THRESHOLD,
    DEFAULT_OSVD_GRID_STEPS,
    DEFAULT_SVD_THRESHOLD,
    DEFAULT_TIKHONOV_PENALTY_P,
    EPSILON_CONCENTRATION,
    LCURVE_N_LAMBDAS,
)

logger = logging.getLogger(__name__)


@dataclass
class DeconvolutionResultBase:
    """Base result fields shared by all deconvolution methods.

    Attributes
    ----------
    residue : np.ndarray
        Flow-scaled residue h(t) = CBF * R(t).
    cbf : float
        Cerebral blood flow, the maximum of h(t).
    cbv : float
        Residue-integral CBV (integral of h(t)); feeds the deconvolution
        rel_cbv map and the parametric oSVD fallback. The primary reported
        CBV is the AUC ratio computed in the interface.
    mtt : float
        Residue transit time, CBV/CBF. The reported SVD MTT is computed in
        the interface as AUC-CBV / CBF.
    """

    residue: npt.NDArray[np.floating]
    cbf: float
    cbv: float
    mtt: float

    def get_tmax(self, dt: float = 1.0) -> float:
        """Get time to maximum of residue function.

        Parameters
        ----------
        dt : float
            Time step in seconds (default 1.0 for frame units).

        Returns
        -------
        float
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(np.argmax(self.residue)) * dt


@dataclass
class DeconvolutionResult(DeconvolutionResultBase):
    """Result from standard SVD (sSVD) deconvolution.

    Attributes
    ----------
    threshold_used : float
        SVD truncation threshold as a fraction of the largest singular value.
    n_singular_values : int
        Number of singular values retained after truncation.
    condition_number : float
        Condition number of the retained singular spectrum.
    """

    threshold_used: float
    n_singular_values: int
    condition_number: float


def build_toeplitz_matrix(
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Build the (1, 4, 1)/6 convolution matrix from the AIF.

    Creates the lower-triangular Toeplitz matrix A such that
    ``C_tissue = A @ h``, where h is the flow-scaled residue. The integral is discretized by
    pre-smoothing the AIF on its interior with the (1, 4, 1)/6 three-point
    stencil (endpoints left unsmoothed), then forming the lower-triangular
    Toeplitz of the smoothed AIF scaled by ``dt``. This quadrature leaves the
    first column and diagonal distinct from the Volterra operator.

    Parameters
    ----------
    aif : np.ndarray
        Arterial input function concentration curve, shape (n_timepoints,).
        Must contain at least 2 samples.
    dt : float
        Time step in seconds for integration.

    Returns
    -------
    np.ndarray
        Toeplitz matrix of shape (n_timepoints, n_timepoints).
    """
    from scipy.linalg import toeplitz  # type: ignore[import-untyped]

    n = len(aif)
    if n < 2:
        raise ValueError(f"aif must contain at least 2 samples, got {n}")
    aif64 = np.asarray(aif, dtype=np.float64)

    # (1, 4, 1)/6 pre-smoothed AIF on the interior; endpoints left unsmoothed.
    aif_smooth = aif64.copy()
    aif_smooth[1:-1] = (aif64[:-2] + 4.0 * aif64[1:-1] + aif64[2:]) / 6.0

    # Lower-triangular Toeplitz of the smoothed AIF, scaled by dt.
    matrix = toeplitz(aif_smooth, np.zeros(n, dtype=np.float64))

    return matrix * dt  # type: ignore[no-any-return]


def build_volterra_matrix(
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Build the Volterra (1, 4, 1)/6 convolution matrix from the AIF.

    Creates the lower-triangular matrix A such that ``C_tissue = A @ h``,
    where h is the flow-scaled residue,
    discretizing the convolution integral with the product-integration Volterra
    quadrature: the interior uses the (1, 4, 1)/6 stencil centred on the lag,
    the first column uses the two-point rule ``(2*aif[i] + aif[i-1])/6``, and
    the main diagonal after row zero is the constant
    ``(2*aif[0] + aif[1])/6``. Row zero is zero, as the integral has zero
    duration there. The whole matrix is scaled by ``dt``. Standard SVD and
    Tikhonov use this operator.

    Parameters
    ----------
    aif : np.ndarray
        Arterial input function concentration curve, shape (n_timepoints,).
        Must contain at least 2 samples.
    dt : float
        Time step in seconds for integration.

    Returns
    -------
    np.ndarray
        Volterra convolution matrix of shape (n_timepoints, n_timepoints).
    """
    from scipy.linalg import toeplitz  # type: ignore[import-untyped]

    n = len(aif)
    if n < 2:
        raise ValueError(f"aif must contain at least 2 samples, got {n}")
    aif64 = np.asarray(aif, dtype=np.float64)

    # Toeplitz kernel k[m] = A[r, r - m]: the constant diagonal A_0^+ at m=0, and
    # the (1, 4, 1)/6 stencil centred on the lag for the interior (m >= 1). The
    # last kernel entry only lands at (n-1, 0), which the first-column rule
    # overwrites, so it is left zero.
    kern = np.zeros(n, dtype=np.float64)
    kern[0] = (2.0 * aif64[0] + aif64[1]) / 6.0
    if n > 2:
        kern[1 : n - 1] = (aif64[:-2] + 4.0 * aif64[1:-1] + aif64[2:]) / 6.0
    matrix = toeplitz(kern, np.zeros(n, dtype=np.float64))

    # First column: two-point rule A_i^- = (2*aif[i] + aif[i-1])/6.
    aif_prev = np.empty(n, dtype=np.float64)
    aif_prev[0] = 0.0
    aif_prev[1:] = aif64[:-1]
    matrix[:, 0] = (2.0 * aif64 + aif_prev) / 6.0

    # The convolution at t=0 has zero duration, so the first row is zero.
    matrix[0, 0] = 0.0

    return matrix * dt  # type: ignore[no-any-return]


def _validate_svd_inputs(
    tissue: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float,
) -> None:
    """Require finite selected curves, a usable AIF and positive sampling."""
    if tissue.ndim == 0 or tissue.shape[-1] == 0:
        raise ValueError("Arrays cannot be empty")
    if aif.ndim != 1 or tissue.shape[-1] != aif.size:
        raise ValueError("tissue curves and aif must have the same length")
    if aif.size < 2:
        raise ValueError("At least two time samples are required")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.all(np.isfinite(aif)) or not np.all(np.isfinite(tissue)):
        raise ValueError("Tissue curves and aif must contain only finite values")
    if np.max(np.abs(aif)) < EPSILON_CONCENTRATION:
        raise ValueError("AIF is near-zero. Check AIF detection.")


def _prepare_svd_batch_inputs(
    tissue: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float,
    mask: Optional[npt.NDArray[np.bool_]],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Validate shared inputs and separate finite from nonfinite selected curves."""
    if tissue.ndim < 2:
        raise ValueError("tissue_curves must have spatial and time dimensions")
    spatial_shape = tissue.shape[:-1]
    if mask is not None and mask.shape != spatial_shape:
        raise ValueError("mask shape must match the tissue spatial dimensions")
    mask_flat = (
        np.ones(int(np.prod(spatial_shape)), dtype=bool)
        if mask is None else np.asarray(mask, dtype=bool).ravel()
    )
    curves = tissue.reshape((mask_flat.size, tissue.shape[-1]))
    finite = np.all(np.isfinite(curves), axis=-1)
    valid = mask_flat & finite
    invalid = mask_flat & ~finite
    _validate_svd_inputs(curves[valid], aif, dt)
    return curves, valid, invalid


def _validate_svd_threshold(threshold: float) -> None:
    """Require a finite fractional singular-value cutoff."""
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")


def _validate_osvd_settings(
    oi_target: float, threshold_min: float, threshold_max: float, max_iterations: int,
) -> None:
    """Require a positive OI target and a nonempty ordered threshold grid."""
    if not np.isfinite(oi_target) or oi_target <= 0:
        raise ValueError("oi_target must be finite and positive")
    if not 0 < threshold_min <= threshold_max < 1:
        raise ValueError("threshold bounds must satisfy 0 < threshold_min <= threshold_max < 1")
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer")


def svd_deconvolve_standard(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    threshold: float = DEFAULT_SVD_THRESHOLD,
) -> DeconvolutionResult:
    """Perform standard SVD deconvolution (sSVD).

    Performs deconvolution by constructing the Volterra (product-integration)
    convolution matrix from the AIF and solving via truncated SVD. Singular
    values at or below a fraction of the maximum are discarded.

    The deconvolution equation is:
        C_tissue(t) = CBF × ∫_0^t R(t-τ) × AIF(τ) dτ

    In matrix form, with flow-scaled residue h = CBF × R:
        C = A @ h

    where A is the Volterra (product-integration) matrix built from AIF.

    Parameters
    ----------
    tissue_curve : np.ndarray
        Tissue concentration curve, shape (n_timepoints,).
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,).
        Must have same length as tissue_curve.
    dt : float, optional
        Time step in seconds. Default 1.0 gives results in frame units.
        Provide actual TR for timing parameters (MTT, Tmax) in seconds.
    threshold : float, optional
        SVD truncation threshold as fraction of maximum singular
        value. Singular values at or below threshold × max(s) are discarded.
        Default is 0.2 (20% of max). Higher values = more regularization.

    Returns
    -------
    DeconvolutionResult
        Signed full-N residue and its finite trapezoidal CBV integral.
        Residue MTT is CBV/CBF, or NaN for nonpositive area or a peak at or
        below the concentration tolerance. The interface reports AUC-CBV/CBF
        as primary MTT.

    Raises
    ------
    ValueError
        If the curves are not finite one-dimensional arrays of equal length
        with at least two samples, the AIF is near-zero, dt is not finite and
        positive, or threshold is outside (0, 1).

    Notes
    -----
    CBF is the maximum of the flow-scaled residue. This method is sensitive
    to bolus delay, which can underestimate CBF.

    """
    tissue_curve = np.asarray(tissue_curve, dtype=np.float64)
    aif = np.asarray(aif, dtype=np.float64)
    if tissue_curve.ndim != 1:
        raise ValueError("tissue_curve must be one-dimensional")
    _validate_svd_inputs(tissue_curve, aif, dt)
    _validate_svd_threshold(threshold)

    # Build Volterra (product-integration) convolution matrix
    matrix = build_volterra_matrix(aif, dt)

    # Perform SVD
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Retain singular values strictly above the cutoff.
    s_max = s[0] if len(s) > 0 else 1.0
    truncation_threshold = threshold * s_max

    # Count retained singular values
    n_retained = int(np.sum(s > truncation_threshold))
    if n_retained == 0:
        n_retained = 1

    # Create truncated inverse singular values
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] > truncation_threshold:
            s_inv[i] = 1.0 / s[i]

    # Compute condition number of truncated matrix
    if n_retained > 0 and s[n_retained - 1] > 0:
        condition_number = float(s[0] / s[n_retained - 1])
    else:
        condition_number = float("inf")

    # Apply the truncated inverse through its SVD factors.
    residue = vh.T @ (s_inv * (u.T @ tissue_curve))

    # Extract perfusion parameters
    cbf = float(np.max(residue))

    # CBV via trapezoidal integration of residue
    cbv = float(np.trapezoid(residue, dx=dt))

    # MTT from central volume theorem: MTT = CBV / CBF
    if cbf > EPSILON_CONCENTRATION and cbv > 0:
        mtt = cbv / cbf
    else:
        mtt = np.nan

    return DeconvolutionResult(
        residue=residue,
        cbf=cbf,
        cbv=cbv,
        mtt=mtt,
        threshold_used=threshold,
        n_singular_values=n_retained,
        condition_number=condition_number,
    )


def svd_deconvolve_standard_batch(
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    threshold: float = DEFAULT_SVD_THRESHOLD,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # residues
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
]:
    """Batch sSVD deconvolution for multiple voxels.

    Efficiently deconvolves multiple tissue curves by precomputing
    the SVD decomposition of the Volterra (product-integration) matrix once.

    Parameters
    ----------
    tissue_curves : np.ndarray
        Tissue concentration curves, shape (n_voxels, n_timepoints)
        or (x, y, z, n_timepoints) for 4D data.
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,).
    dt : float, optional
        Time step in seconds.
    threshold : float, optional
        SVD truncation threshold (default 0.2).
    mask : np.ndarray, optional
        Optional boolean mask of voxels to process. If None, process all.
        Shape must match spatial dimensions of tissue_curves.

    Returns
    -------
    tuple
        Signed full-N residues and spatial cbf, cbv and mtt arrays. CBV uses
        finite trapezoidal integration. Residue MTT is NaN for nonpositive
        area or a peak at or below the concentration tolerance; excluded
        voxels remain zero. Nonfinite selected tissue curves receive NaN
        outputs without affecting other voxels.

    Raises
    ------
    ValueError
        If spatial/time dimensions, mask shape or AIF are invalid, dt is not
        finite and positive, or threshold is outside (0, 1).
    """
    tissue_curves = np.asarray(tissue_curves, dtype=np.float64)
    aif = np.asarray(aif, dtype=np.float64)
    tissue_2d, mask_flat, invalid = _prepare_svd_batch_inputs(tissue_curves, aif, dt, mask)
    _validate_svd_threshold(threshold)
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]
    spatial_shape = original_shape[:-1]
    n_voxels = mask_flat.size

    # Build Volterra (product-integration) matrix and compute SVD once
    matrix = build_volterra_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Truncate singular values
    s_max = s[0] if len(s) > 0 else 1.0
    truncation_threshold = threshold * s_max
    n_retained = max(1, int(np.sum(s > truncation_threshold)))

    # Create truncated inverse
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] > truncation_threshold:
            s_inv[i] = 1.0 / s[i]

    # Initialize output arrays
    residues = np.zeros((n_voxels, n_timepoints), dtype=np.float64)
    cbf = np.zeros(n_voxels, dtype=np.float64)
    cbv = np.zeros(n_voxels, dtype=np.float64)
    mtt = np.zeros(n_voxels, dtype=np.float64)
    for values in (residues, cbf, cbv, mtt):
        values[invalid] = np.nan

    # Process masked voxels
    # Vectorized computation: R = vh.T @ (s_inv[:, None] * (u.T @ C.T))
    # For multiple voxels: R_batch = vh.T @ diag(s_inv) @ u.T @ C_batch.T
    masked_indices = np.where(mask_flat)[0]
    if len(masked_indices) > 0:
        tissue_masked = tissue_2d[masked_indices, :]

        # Batch deconvolution: (n_time, n_time) @ (n_time, n_vox) = (n_time, n_vox)
        # u.T @ tissue.T: (n_time, n_time) @ (n_time, n_vox) = (n_time, n_vox)
        ut_tissue = u.T @ tissue_masked.T  # (n_time, n_vox)
        scaled = s_inv[:, None] * ut_tissue  # (n_time, n_vox)
        residues_masked = (vh.T @ scaled).T  # (n_vox, n_time)

        # Extract parameters
        cbf_masked = np.max(residues_masked, axis=1)

        # CBV via trapezoidal integration
        cbv_masked = np.trapezoid(residues_masked, dx=dt, axis=1)

        # MTT = CBV / CBF
        with np.errstate(divide="ignore", invalid="ignore"):
            mtt_masked = np.where(
                (cbf_masked > EPSILON_CONCENTRATION) & (cbv_masked > 0),
                cbv_masked / cbf_masked,
                np.nan,
            )

        # Store results
        residues[masked_indices, :] = residues_masked
        cbf[masked_indices] = cbf_masked
        cbv[masked_indices] = cbv_masked
        mtt[masked_indices] = mtt_masked

    # Reshape to original spatial dimensions
    if tissue_curves.ndim > 2:
        residues = residues.reshape(original_shape)  # type: ignore[assignment]
        cbf = cbf.reshape(spatial_shape)  # type: ignore[assignment]
        cbv = cbv.reshape(spatial_shape)  # type: ignore[assignment]
        mtt = mtt.reshape(spatial_shape)  # type: ignore[assignment]

    return residues, cbf, cbv, mtt


# =============================================================================
# Circulant SVD (cSVD) - Delay-Insensitive Deconvolution
# =============================================================================


@dataclass
class CirculantDeconvolutionResult(DeconvolutionResultBase):
    """Result from circulant SVD (cSVD) deconvolution.

    Attributes
    ----------
    delay : float
        Signed residue-peak lag in seconds relative to the AIF.
    delay_frames : int
        Residue-peak lag in frames, with indices N through 2N-1 wrapped negative.
    threshold_used : float
        SVD truncation threshold as a fraction of the largest singular value.
    n_singular_values : int
        Number of singular values retained after truncation.
    condition_number : float
        Condition number of the retained singular spectrum.
    """

    delay: float
    delay_frames: int
    threshold_used: float
    n_singular_values: int
    condition_number: float

    def get_tmax(self, dt: float = 1.0) -> float:
        """Convert the signed residue-peak lag to time units.

        Parameters
        ----------
        dt : float
            Time step in seconds (default 1.0 for frame units).

        Returns
        -------
        float
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(self.delay_frames) * dt


def build_block_circulant_matrix(
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Build the (1, 4, 1)/6 block-circulant convolution matrix from the AIF.

    Constructs a 2N x 2N circulant (periodic) convolution matrix whose
    generating column is the AIF pre-smoothed with the (1, 4, 1)/6 stencil
    and zero-padded to length 2N. The residue peak can occur anywhere in this
    period; indices N through 2N-1 represent negative lags.

    The generating column applies the (1, 4, 1)/6 stencil on the interior,
    one-sided stencils consistent with zero extension at both AIF endpoints,
    and carries the trailing stencil mass one sample into the zero-pad region
    (index N).

    Parameters
    ----------
    aif : np.ndarray
        Arterial input function concentration curve, shape (n_timepoints,).
        Must contain at least 2 samples.
    dt : float
        Time step in seconds for integration weighting.

    Returns
    -------
    np.ndarray
        Block-circulant matrix of shape (2N, 2N) where N = len(aif).
    """
    n = len(aif)
    if n < 2:
        raise ValueError(f"aif must contain at least 2 samples, got {n}")
    m = 2 * n
    aif64 = np.asarray(aif, dtype=np.float64)

    # (1, 4, 1)/6 pre-smoothed AIF as the circulant generating column,
    # zero-padded to length 2N. The trailing stencil mass leaks one sample
    # into the pad region (index N). Both AIF endpoints use one-sided stencils
    # consistent with zero extension.
    col = np.zeros(m, dtype=np.float64)
    col[0] = (4.0 * aif64[0] + aif64[1]) / 6.0
    col[1 : n - 1] = (aif64[: n - 2] + 4.0 * aif64[1 : n - 1] + aif64[2:n]) / 6.0
    col[n - 1] = (aif64[n - 2] + 4.0 * aif64[n - 1]) / 6.0
    col[n] = aif64[n - 1] / 6.0

    matrix: npt.NDArray[np.floating] = linalg.circulant(col) * dt

    return matrix


def svd_deconvolve_circulant(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    threshold: float = DEFAULT_SVD_THRESHOLD,
) -> CirculantDeconvolutionResult:
    """Circulant SVD deconvolution (cSVD) for delay-insensitive estimation.

    Solve for a signed flow-scaled residue using a block-circulant matrix
    and a fixed singular-value cutoff. Both curves are zero-padded to 2N.
    The residue peak defines the lag: indices below N give nonnegative lags,
    and indices N through 2N-1 wrap to negative lags.

    Parameters
    ----------
    tissue_curve : np.ndarray
        Tissue concentration curve, shape (n_timepoints,).
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,).
        Must have same length as tissue_curve.
    dt : float, optional
        Time step in seconds. Default 1.0 gives delay in frame units.
        Use the acquisition frame interval for timing in seconds.
    threshold : float, optional
        SVD truncation threshold as fraction of maximum singular
        value. Singular values at or below threshold × max(s) are discarded.
        Default is 0.2 (20% of max). Higher values = more regularization.

    Returns
    -------
    CirculantDeconvolutionResult
        Signed residue of length 2N, CBF from its maximum, and CBV from the
        trapezoidal integral of N samples starting at the peak and wrapping
        around the period. MTT is CBV/CBF when CBF exceeds the concentration
        tolerance, otherwise zero. Delay is the signed residue-peak lag.

    Raises
    ------
    ValueError
        If curves differ in length, contain fewer than two samples, or
        threshold is outside (0, 1).
    """
    # Validate inputs
    if len(tissue_curve) != len(aif):
        raise ValueError(
            f"tissue_curve and aif must have same length, "
            f"got {len(tissue_curve)} and {len(aif)}"
        )

    if len(tissue_curve) == 0:
        raise ValueError("Arrays cannot be empty")

    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    n = len(tissue_curve)
    m = 2 * n  # Padded length

    # Zero-pad tissue curve to length 2N
    tissue_padded = np.zeros(m, dtype=np.float64)
    tissue_padded[:n] = tissue_curve

    # Build block-circulant matrix
    matrix = build_block_circulant_matrix(aif, dt)

    # Perform SVD
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Retain singular values strictly above the cutoff.
    s_max = s[0] if len(s) > 0 else 1.0
    truncation_threshold = threshold * s_max

    # Count retained singular values
    n_retained = int(np.sum(s > truncation_threshold))
    if n_retained == 0:
        n_retained = 1

    # Create truncated inverse singular values
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] > truncation_threshold:
            s_inv[i] = 1.0 / s[i]

    # Compute condition number of truncated matrix
    if n_retained > 0 and s[n_retained - 1] > 0:
        condition_number = float(s[0] / s[n_retained - 1])
    else:
        condition_number = float("inf")

    # Solve for residue function: R = V @ S_inv @ U^T @ C
    residue = vh.T @ (s_inv * (u.T @ tissue_padded))

    # Extract delay from position of maximum
    # Zero-delay is at index 0; indices >= n wrap to negative delays.
    peak_position = int(np.argmax(residue))
    if peak_position < n:
        delay_frames = peak_position
    else:
        delay_frames = peak_position - m
    delay_seconds = float(delay_frames) * dt

    # Integrate an N-sample window starting at the peak and wrapping at 2N.
    causal_indices = (peak_position + np.arange(n)) % m
    causal_residue = residue[causal_indices]

    cbf = float(np.max(causal_residue))

    cbv = float(np.trapezoid(causal_residue, dx=dt))

    # MTT from central volume theorem: MTT = CBV / CBF
    if cbf > EPSILON_CONCENTRATION:
        mtt = cbv / cbf
    else:
        mtt = 0.0

    return CirculantDeconvolutionResult(
        residue=residue,
        cbf=cbf,
        cbv=cbv,
        mtt=mtt,
        delay=delay_seconds,
        delay_frames=delay_frames,
        threshold_used=threshold,
        n_singular_values=n_retained,
        condition_number=condition_number,
    )


def svd_deconvolve_circulant_batch(
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    threshold: float = DEFAULT_SVD_THRESHOLD,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # residues
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
    npt.NDArray[np.floating],  # delay (seconds)
]:
    """Batch cSVD deconvolution for multiple voxels.

    Efficiently deconvolves multiple tissue curves using the circulant SVD
    approach by precomputing the SVD decomposition once.

    Parameters
    ----------
    tissue_curves : np.ndarray
        Tissue concentration curves, shape (n_voxels, n_timepoints)
        or (x, y, z, n_timepoints) for 4D data.
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,).
    dt : float, optional
        Time step in seconds.
    threshold : float, optional
        SVD truncation threshold (default 0.2).
    mask : np.ndarray, optional
        Optional boolean mask of voxels to process. If None, process all.
        Shape must match spatial dimensions of tissue_curves.

    Returns
    -------
    tuple
        Signed residues have shape (..., 2N); cbf, cbv, mtt and delay match
        the input spatial shape. CBV integrates N samples from each residue
        peak with periodic wrapping, as in the scalar cSVD function. MTT is
        CBV/CBF when CBF exceeds the concentration tolerance, otherwise zero.
        Excluded voxels remain zero.
    """
    # Store original shape for reshaping output
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]
    m = 2 * n_timepoints  # Padded length
    tissue_2d: npt.NDArray[np.floating]

    # Reshape to 2D (n_voxels, n_timepoints)
    if tissue_curves.ndim > 2:
        spatial_shape = original_shape[:-1]
        n_voxels = int(np.prod(spatial_shape))
        tissue_2d = tissue_curves.reshape(n_voxels, n_timepoints)
    else:
        spatial_shape = (original_shape[0],)
        n_voxels = original_shape[0]
        tissue_2d = tissue_curves

    # Flatten mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
    else:
        mask_flat = np.ones(n_voxels, dtype=bool)

    # Build block-circulant matrix and compute SVD once
    matrix = build_block_circulant_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Truncate singular values
    s_max = s[0] if len(s) > 0 else 1.0
    truncation_threshold = threshold * s_max
    n_retained = max(1, int(np.sum(s > truncation_threshold)))

    # Create truncated inverse
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] > truncation_threshold:
            s_inv[i] = 1.0 / s[i]

    # Initialize output arrays
    residues = np.zeros((n_voxels, m), dtype=np.float64)
    cbf = np.zeros(n_voxels, dtype=np.float64)
    cbv = np.zeros(n_voxels, dtype=np.float64)
    mtt = np.zeros(n_voxels, dtype=np.float64)
    delay = np.zeros(n_voxels, dtype=np.float64)

    # Process masked voxels
    masked_indices = np.where(mask_flat)[0]
    if len(masked_indices) > 0:
        tissue_masked = tissue_2d[masked_indices, :]

        # Zero-pad tissue curves to length 2N
        tissue_padded = np.zeros((len(masked_indices), m), dtype=np.float64)
        tissue_padded[:, :n_timepoints] = tissue_masked

        # Batch deconvolution: R_batch = vh.T @ diag(s_inv) @ u.T @ C_batch.T
        ut_tissue = u.T @ tissue_padded.T  # (m, n_vox)
        scaled = s_inv[:, None] * ut_tissue  # (m, n_vox)
        residues_masked = (vh.T @ scaled).T  # (n_vox, m)

        # Extract delays from peak positions (zero-delay at index 0)
        peak_positions = np.argmax(residues_masked, axis=1)
        delay_frames = np.where(
            peak_positions < n_timepoints,
            peak_positions,
            peak_positions - m,
        )
        delay_masked = delay_frames.astype(np.float64) * dt

        # Each integration window starts at its voxel's peak and wraps at 2N.
        vox_idx = np.arange(len(masked_indices))[:, None]
        time_idx = (peak_positions[:, None] + np.arange(n_timepoints)[None, :]) % m
        causal_residues = residues_masked[vox_idx, time_idx]  # (n_vox, N)

        cbf_masked = np.max(causal_residues, axis=1)

        cbv_masked = np.trapezoid(causal_residues, dx=dt, axis=1)

        # MTT = CBV / CBF
        with np.errstate(divide="ignore", invalid="ignore"):
            mtt_masked = np.where(
                cbf_masked > EPSILON_CONCENTRATION,
                cbv_masked / cbf_masked,
                0.0,
            )

        # Store results
        residues[masked_indices, :] = residues_masked
        cbf[masked_indices] = cbf_masked
        cbv[masked_indices] = cbv_masked
        mtt[masked_indices] = mtt_masked
        delay[masked_indices] = delay_masked

    # Reshape to original spatial dimensions
    if tissue_curves.ndim > 2:
        residue_shape = spatial_shape + (m,)
        residues = residues.reshape(residue_shape)  # type: ignore[assignment]
        cbf = cbf.reshape(spatial_shape)  # type: ignore[assignment]
        cbv = cbv.reshape(spatial_shape)  # type: ignore[assignment]
        mtt = mtt.reshape(spatial_shape)  # type: ignore[assignment]
        delay = delay.reshape(spatial_shape)  # type: ignore[assignment]

    return residues, cbf, cbv, mtt, delay


# =============================================================================
# Oscillation-Index Limited SVD (oSVD)
# =============================================================================


@dataclass
class OscillationDeconvolutionResult(DeconvolutionResultBase):
    """Result from oscillation-index limited SVD (oSVD) deconvolution.

    Attributes
    ----------
    delay : float
        Signed residue-peak lag in seconds relative to the AIF.
    delay_frames : int
        Residue-peak lag in frames, with indices N through 2N-1 wrapped negative.
    oscillation_index : float
        Sum of absolute interior second differences divided by L * max(residue),
        where L = 2N is the residue length.
    threshold_used : float
        Selected singular-value cutoff as a fraction of the largest value.
    n_singular_values : int
        Number of singular values retained after truncation.
    condition_number : float
        Condition number of the retained singular spectrum.
    oi_target : float
        Target oscillation index.
    converged : bool
        Whether the threshold search found a residue meeting the target.
    """

    delay: float
    delay_frames: int
    oscillation_index: float
    threshold_used: float
    n_singular_values: int
    condition_number: float
    oi_target: float
    converged: bool

    def get_tmax(self, dt: float = 1.0) -> float:
        """Convert the signed residue-peak lag to time units.

        Parameters
        ----------
        dt : float
            Time step in seconds (default 1.0 for frame units).

        Returns
        -------
        float
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(self.delay_frames) * dt


def compute_oscillation_index(residue: npt.NDArray[np.floating]) -> float:
    """Compute the oscillation index of a residue function.

    The oscillation index (OI) measures oscillation in the residue function as
    the summed absolute second-order difference (discrete curvature) normalized
    by the number of points and the maximum value:

        OI = (1/L) × Σ|R(k) − 2R(k−1) + R(k−2)| / max(R)

    where L is the number of points. Differences use interior samples only,
    without a periodic closure term.

    Parameters
    ----------
    residue : np.ndarray
        Residue function, shape (n_timepoints,).

    Returns
    -------
    float
        Oscillation index (dimensionless). Returns 0.0 if the peak is at or
        below the concentration tolerance or there are fewer than three points.

    """
    # Handle edge cases (the second difference needs >= 3 points)
    if len(residue) < 3:
        return 0.0

    max_residue = float(np.max(residue))
    if max_residue <= EPSILON_CONCENTRATION:
        return 0.0

    # Summed absolute second-order difference (discrete curvature), normalized
    # by length and maximum: OI = sum|R[k] − 2R[k-1] + R[k-2]| / (L * max).
    diff2 = np.abs(np.diff(residue, n=2))
    oi = float(np.sum(diff2)) / (len(residue) * max_residue)

    return oi


def _deconvolve_with_threshold(
    tissue_padded: npt.NDArray[np.floating],
    threshold: float,
    u: npt.NDArray[np.floating],
    s: npt.NDArray[np.floating],
    vh: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], float, int, float, int]:
    """Deconvolve with a given threshold using block-circulant SVD (helper).

    OI is computed on the full signed 2N residue, which is also returned
    unchanged for parameter extraction.

    Parameters
    ----------
    tissue_padded : np.ndarray
        Zero-padded tissue curve, length 2N.
    threshold : float
        SVD truncation threshold (fraction of max singular value).
    u, s, vh : np.ndarray
        Pre-computed SVD of block-circulant matrix.

    Returns
    -------
    tuple
        Tuple of (residue, oscillation_index, n_retained,
        condition_number, peak_position).
    """
    # Retain singular values strictly above the cutoff.
    s_max = s[0] if len(s) > 0 else 1.0
    truncation_threshold = threshold * s_max

    # Count retained singular values
    n_retained = int(np.sum(s > truncation_threshold))
    if n_retained == 0:
        n_retained = 1

    # Create truncated inverse singular values
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] > truncation_threshold:
            s_inv[i] = 1.0 / s[i]

    # Compute condition number of truncated matrix
    if n_retained > 0 and s[n_retained - 1] > 0:
        condition_number = float(s[0] / s[n_retained - 1])
    else:
        condition_number = float("inf")

    # Solve: R = Vh^T @ (S_inv * (U^T @ C_padded))
    residue = vh.T @ (s_inv * (u.T @ tissue_padded))

    # Peak (for the delay) is the global maximum of the residue.
    peak_position = int(np.argmax(residue))

    # Evaluate oscillation over the full signed residue, including its tail.
    oi = compute_oscillation_index(residue)

    return residue, oi, n_retained, condition_number, peak_position


def svd_deconvolve_oscillation(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    oi_target: float = DEFAULT_OI_THRESHOLD,
    threshold_min: float = 0.01,
    threshold_max: float = 0.50,
    max_iterations: int = DEFAULT_OSVD_GRID_STEPS,
) -> OscillationDeconvolutionResult:
    """Oscillation-index limited SVD deconvolution (oSVD).

    Uses a block-circulant matrix for delay-insensitive
    deconvolution with adaptive OI-based threshold selection.

    The oscillation index is defined as::

        OI = (1/L) * sum|R(k) - 2R(k-1) + R(k-2)| / max(R)

    computed on the full signed residue, with L = 2N and interior second
    differences. The area integral includes the circular closure interval.

    A linear search from low to high threshold finds the first (lowest)
    threshold that achieves OI <= oi_target. If none does, the upper-threshold
    result is retained with converged=False.

    Parameters
    ----------
    tissue_curve : np.ndarray
        Tissue concentration curve, shape (n_timepoints,).
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,).
        Must have same length as tissue_curve.
    dt : float, optional
        Time step in seconds. Default 1.0 gives results in frame units.
    oi_target : float, optional
        Target oscillation index threshold. Default :data:`DEFAULT_OI_THRESHOLD`.
    threshold_min : float, optional
        Minimum SVD threshold to consider. Default 0.01.
    threshold_max : float, optional
        Maximum SVD threshold to consider. Default 0.50.
    max_iterations : int, optional
        Number of evenly-spaced thresholds. Default :data:`DEFAULT_OSVD_GRID_STEPS`.

    Returns
    -------
    OscillationDeconvolutionResult
        OscillationDeconvolutionResult with 2N residue, delay, and
        signed full-period CBV from dt * sum(residue). Residue MTT is this
        area divided by CBF, or NaN for a nonpositive area or a peak at or
        below the concentration tolerance.

    Raises
    ------
    ValueError
        If the curves are not finite one-dimensional arrays of equal length
        with at least two samples, the AIF is near-zero, dt or oi_target is
        not finite and positive, the bounds do not satisfy
        0 < threshold_min <= threshold_max < 1, or max_iterations is not a
        positive integer.

    """
    tissue_curve = np.asarray(tissue_curve, dtype=np.float64)
    aif = np.asarray(aif, dtype=np.float64)
    if tissue_curve.ndim != 1:
        raise ValueError("tissue_curve must be one-dimensional")
    _validate_svd_inputs(tissue_curve, aif, dt)
    _validate_osvd_settings(oi_target, threshold_min, threshold_max, max_iterations)

    n = len(tissue_curve)
    m = 2 * n

    # Zero-pad tissue curve to 2N
    tissue_padded = np.zeros(m, dtype=np.float64)
    tissue_padded[:n] = tissue_curve

    # Build block-circulant matrix and compute SVD once
    matrix = build_block_circulant_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Search increasing thresholds for the first OI at or below target.
    thresholds = np.linspace(threshold_min, threshold_max, max_iterations)

    best_residue: Optional[npt.NDArray[np.floating]] = None
    best_oi = float("inf")
    best_threshold = threshold_max
    best_n_ret = 0
    best_cond = float("inf")
    best_peak = 0
    converged = False

    for thresh in thresholds:
        residue, oi, n_ret, cond, peak = _deconvolve_with_threshold(
            tissue_padded, thresh, u, s, vh,
        )

        if oi <= oi_target:
            best_residue = residue
            best_oi = oi
            best_threshold = thresh
            best_n_ret = n_ret
            best_cond = cond
            best_peak = peak
            converged = True
            break

    # If no threshold achieved target, use maximum regularization
    if best_residue is None:
        best_residue, best_oi, best_n_ret, best_cond, best_peak = (
            _deconvolve_with_threshold(
                tissue_padded, threshold_max, u, s, vh,
            )
        )
        best_threshold = threshold_max
        converged = False

    # Extract delay from peak position (zero-delay at index 0)
    if best_peak < n:
        delay_frames = best_peak
    else:
        delay_frames = best_peak - m
    delay = float(delay_frames) * dt

    cbf = float(np.max(best_residue))
    cbv = float(dt * np.sum(best_residue))
    mtt = cbv / cbf if cbf > EPSILON_CONCENTRATION and cbv > 0 else np.nan

    return OscillationDeconvolutionResult(
        residue=best_residue,
        cbf=cbf,
        cbv=cbv,
        mtt=mtt,
        delay=delay,
        delay_frames=delay_frames,
        oscillation_index=best_oi,
        threshold_used=best_threshold,
        n_singular_values=best_n_ret,
        condition_number=best_cond,
        oi_target=oi_target,
        converged=converged,
    )


def svd_deconvolve_oscillation_batch(
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    oi_target: float = DEFAULT_OI_THRESHOLD,
    threshold_min: float = 0.01,
    threshold_max: float = 0.50,
    max_iterations: int = DEFAULT_OSVD_GRID_STEPS,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # residues (2N per voxel)
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
    npt.NDArray[np.floating],  # oscillation_index
    npt.NDArray[np.floating],  # threshold_used
    npt.NDArray[np.floating],  # delay (seconds)
]:
    """Batch oSVD deconvolution using block-circulant matrix.

    Each voxel gets an individually optimized threshold via OI-based
    linear search. Returns 2N residues and delay maps.

    Parameters
    ----------
    tissue_curves : np.ndarray
        Tissue concentration curves, shape (n_voxels, n_timepoints)
        or (x, y, z, n_timepoints) for 4D data.
    aif : np.ndarray
        Arterial input function, shape (n_timepoints,).
    dt : float, optional
        Time step in seconds.
    oi_target : float, optional
        Target oscillation index threshold. Default :data:`DEFAULT_OI_THRESHOLD`.
    threshold_min : float, optional
        Minimum SVD threshold. Default 0.01.
    threshold_max : float, optional
        Maximum SVD threshold. Default 0.50.
    max_iterations : int, optional
        Number of linear search steps per voxel. Default :data:`DEFAULT_OSVD_GRID_STEPS`.
    mask : np.ndarray, optional
        Optional boolean mask of voxels to process.

    Returns
    -------
    tuple
        Tuple of (residues, cbf, cbv, mtt, oscillation_index, threshold_used,
        delay) arrays. Residues have time dimension 2N. All spatial arrays
        match the input spatial shape. CBV is the signed full-period area,
        dt * sum(residue); residue MTT is NaN for a nonpositive area or a peak
        at or below the concentration tolerance. Excluded voxels remain zero.
        Nonfinite selected tissue curves receive NaN outputs without affecting
        other voxels. For finite in-mask OI values, oscillation_index <= oi_target
        indicates that the target was met. Results above target retain the
        upper-threshold fit.

    Raises
    ------
    ValueError
        If spatial/time dimensions or mask shape are invalid, or the AIF,
        sampling or search settings fail scalar validation.
    """
    tissue_curves = np.asarray(tissue_curves, dtype=np.float64)
    aif = np.asarray(aif, dtype=np.float64)
    tissue_2d, mask_flat, invalid = _prepare_svd_batch_inputs(tissue_curves, aif, dt, mask)
    _validate_osvd_settings(oi_target, threshold_min, threshold_max, max_iterations)
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]
    m = 2 * n_timepoints
    spatial_shape = original_shape[:-1]
    n_voxels = mask_flat.size

    # Build block-circulant matrix and compute SVD once
    matrix = build_block_circulant_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Initialize output arrays (residues are 2N long)
    residues = np.zeros((n_voxels, m), dtype=np.float64)
    cbf = np.zeros(n_voxels, dtype=np.float64)
    cbv = np.zeros(n_voxels, dtype=np.float64)
    mtt = np.zeros(n_voxels, dtype=np.float64)
    oscillation_index = np.zeros(n_voxels, dtype=np.float64)
    threshold_used = np.zeros(n_voxels, dtype=np.float64)
    delay = np.zeros(n_voxels, dtype=np.float64)
    for values in (residues, cbf, cbv, mtt, oscillation_index, threshold_used, delay):
        values[invalid] = np.nan

    masked_indices = np.where(mask_flat)[0]
    thresholds = np.linspace(threshold_min, threshold_max, max_iterations)

    for idx in masked_indices:
        # Zero-pad tissue to 2N
        tissue_padded = np.zeros(m, dtype=np.float64)
        tissue_padded[:n_timepoints] = tissue_2d[idx, :]

        # Search increasing thresholds for the first OI at or below target.
        best_residue = None
        best_oi = float("inf")
        best_threshold = threshold_max
        best_peak = 0

        for thresh in thresholds:
            residue, oi, n_ret, cond, peak = _deconvolve_with_threshold(
                tissue_padded, thresh, u, s, vh,
            )
            if oi <= oi_target:
                best_residue = residue
                best_oi = oi
                best_threshold = thresh
                best_peak = peak
                break

        # If no threshold achieved target, use maximum regularization
        if best_residue is None:
            best_residue, best_oi, _, _, best_peak = _deconvolve_with_threshold(
                tissue_padded, threshold_max, u, s, vh,
            )
            best_threshold = threshold_max

        # Extract delay (zero-delay at index 0)
        if best_peak < n_timepoints:
            delay_frames = best_peak
        else:
            delay_frames = best_peak - m
        residues[idx, :] = best_residue
        cbf[idx] = float(np.max(best_residue))
        cbv[idx] = float(dt * np.sum(best_residue))
        mtt[idx] = (
            cbv[idx] / cbf[idx]
            if cbf[idx] > EPSILON_CONCENTRATION and cbv[idx] > 0 else np.nan
        )
        oscillation_index[idx] = best_oi
        threshold_used[idx] = best_threshold
        delay[idx] = float(delay_frames) * dt

    # Reshape to original spatial dimensions
    if tissue_curves.ndim > 2:
        residue_shape = spatial_shape + (m,)
        residues = residues.reshape(residue_shape)  # type: ignore[assignment]
        cbf = cbf.reshape(spatial_shape)  # type: ignore[assignment]
        cbv = cbv.reshape(spatial_shape)  # type: ignore[assignment]
        mtt = mtt.reshape(spatial_shape)  # type: ignore[assignment]
        oscillation_index = oscillation_index.reshape(spatial_shape)  # type: ignore[assignment]
        threshold_used = threshold_used.reshape(spatial_shape)  # type: ignore[assignment]
        delay = delay.reshape(spatial_shape)  # type: ignore[assignment]

    return residues, cbf, cbv, mtt, oscillation_index, threshold_used, delay


# =============================================================================
# Tikhonov Regularization
# =============================================================================


@dataclass
class TikhonovDeconvolutionResult(DeconvolutionResultBase):
    """Signed Tikhonov residue and full-N perfusion diagnostics.

    Attributes
    ----------
    lambda_optimal : float
        Discrete L-curve selection, or zero for an inactive tissue curve.
    residual_norm : float
        Squared data residual norm of the unconstrained fit.
    solution_norm : float
        Squared penalty norm of the unconstrained fit.
    selection_status : TikhonovSelection
        Interior curvature sign, upper-endpoint selection, or inactive input.
    """

    lambda_optimal: float
    residual_norm: float
    solution_norm: float
    selection_status: "TikhonovSelection"

    @property
    def lcurve_corner_found(self) -> bool:
        """Whether a strict interior curvature maximum was selected."""
        return self.selection_status in (
            TikhonovSelection.POSITIVE_CORNER,
            TikhonovSelection.NONPOSITIVE_CORNER,
        )


class TikhonovSelection(IntEnum):
    """Mutually exclusive outcomes of discrete L-curve selection."""

    INACTIVE = 0
    POSITIVE_CORNER = 1
    NONPOSITIVE_CORNER = 2
    UPPER_ENDPOINT = 3
    INVALID_INPUT = 4


def _build_modified_first_derivative_matrix(
    n: int, p: int = DEFAULT_TIKHONOV_PENALTY_P
) -> npt.NDArray[np.float64]:
    """Build a leading magnitude penalty followed by first differences.

    Parameters
    ----------
    n : int
        Number of residue samples.
    p : int
        Number of leading samples penalized by magnitude rather than slope.

    Returns
    -------
    np.ndarray
        Square penalty matrix, with an identity block on the leading samples.
    """
    penalty = np.zeros((n, n), dtype=np.float64)
    p_eff = min(p, n)
    leading = np.arange(p_eff)
    penalty[leading, leading] = 1.0
    tail = np.arange(p_eff, n)
    penalty[tail, tail - 1] = -1.0
    penalty[tail, tail] = 1.0
    return penalty


@dataclass
class _TikhonovSystem:
    """Shared standard-form decomposition for one AIF and penalty."""

    matrix: npt.NDArray[np.float64]
    penalty: npt.NDArray[np.float64]
    left: npt.NDArray[np.float64]
    singular: npt.NDArray[np.float64]
    back: npt.NDArray[np.float64]
    unpenalized: npt.NDArray[np.float64]
    projection: npt.NDArray[np.float64]
    lambdas: npt.NDArray[np.float64]


def _validate_tikhonov_inputs(
    tissue: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float,
    lambda_min: float,
    lambda_max: float,
) -> None:
    """Require finite curves, a positive frame interval and ordered bounds."""
    if tissue.ndim == 0 or tissue.shape[-1] == 0:
        raise ValueError("Arrays cannot be empty")
    if aif.ndim != 1 or tissue.shape[-1] != aif.size:
        raise ValueError("tissue curves and aif must have the same length")
    if aif.size < 2:
        raise ValueError("At least two time samples are required")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not (np.isfinite(lambda_min) and np.isfinite(lambda_max)
            and 0 < lambda_min < lambda_max):
        raise ValueError("lambda_min and lambda_max must be finite, positive, and ordered")
    if not np.all(np.isfinite(aif)) or not np.all(np.isfinite(tissue)):
        raise ValueError("Tissue curves and aif must contain only finite values")


def _prepare_tikhonov_system(
    aif: npt.NDArray[np.floating],
    dt: float,
    lambda_min: float,
    lambda_max: float,
    penalty: Optional[npt.NDArray[np.floating]] = None,
) -> _TikhonovSystem:
    """Reduce the full-N causal forward model and penalty to standard form."""
    matrix = np.asarray(build_volterra_matrix(aif, dt), dtype=np.float64)
    n = len(aif)
    if not np.any(matrix):
        raise ValueError("AIF produces a degenerate convolution matrix")
    if penalty is None:
        active_penalty = _build_modified_first_derivative_matrix(n)
    else:
        if penalty.ndim != 2 or penalty.shape[1] != len(aif):
            raise ValueError("Penalty columns must match the AIF length")
        if not np.all(np.isfinite(penalty)):
            raise ValueError("Penalty must contain only finite values")
        active_penalty = np.asarray(penalty, dtype=np.float64)

    projection = np.eye(len(aif), dtype=np.float64)
    unpenalized = np.zeros((n, n), dtype=np.float64)
    if np.array_equal(active_penalty, _build_modified_first_derivative_matrix(n)):
        transform = linalg.solve_triangular(
            active_penalty, np.eye(n), lower=True,
        )
    else:
        transform = linalg.pinv(active_penalty)
        null = linalg.null_space(active_penalty)
        if null.shape[1]:
            null_forward = matrix @ null
            if np.linalg.matrix_rank(null_forward) != null.shape[1]:
                raise ValueError("Unpenalized residue components are not identifiable")
            unpenalized = null @ linalg.pinv(null_forward)
            projection -= matrix @ unpenalized
            transform -= unpenalized @ matrix @ transform

    left, singular, right = linalg.svd(matrix @ transform, full_matrices=False)
    scale = float(linalg.svdvals(matrix)[0])
    lambdas = scale * np.logspace(
        np.log10(lambda_min), np.log10(lambda_max), LCURVE_N_LAMBDAS,
    )
    return _TikhonovSystem(
        matrix, active_penalty, left, singular, transform @ right.T,
        unpenalized, projection, lambdas,
    )


def _tikhonov_lcurve(
    singular: npt.NDArray[np.floating],
    coefficients_squared: npt.NDArray[np.floating],
    perpendicular_squared: npt.NDArray[np.floating],
    lambdas: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64],
    npt.NDArray[np.float64], npt.NDArray[np.float64],
]:
    """Evaluate norms and signed log-log curvature using analytic derivatives.

    Parameters
    ----------
    singular : np.ndarray
        Singular values of the standard-form forward matrix.
    coefficients_squared : np.ndarray
        Squared data coefficients, shape (n_components, n_voxels).
    perpendicular_squared : np.ndarray
        Squared residual outside the left singular-vector space, per voxel.
    lambdas : np.ndarray
        Positive regularization parameters.

    Returns
    -------
    rho, eta, curvature, curvature_derivative : np.ndarray
        Data and penalty norms and signed curvature, each shaped
        (n_lambdas, n_voxels). The derivative is with respect to log lambda.
        Undefined curvature is negative infinity.
    """
    lam2 = np.atleast_1d(lambdas)[:, None] ** 2
    s2 = singular[None, :] ** 2
    fraction = lam2 / (s2 + lam2)
    first = 2 * fraction * (1 - fraction)
    second = 4 * fraction * (1 - fraction) * (1 - 2 * fraction)
    third = 8 * fraction * (1 - fraction) * (1 - 6 * fraction + 6 * fraction ** 2)
    weight = singular[None, :] / (s2 + lam2)
    weight_first = -2 * fraction * weight
    weight_second = 4 * fraction * (2 * fraction - 1) * weight
    weight_third = -8 * fraction * (1 - 6 * fraction + 6 * fraction ** 2) * weight
    rho2 = (fraction ** 2) @ coefficients_squared + perpendicular_squared
    eta2 = (weight ** 2) @ coefficients_squared
    rho_first = (2 * fraction * first) @ coefficients_squared
    eta_first = (2 * weight * weight_first) @ coefficients_squared
    rho_second = (2 * (first ** 2 + fraction * second)) @ coefficients_squared
    eta_second = (2 * (weight_first ** 2 + weight * weight_second)) @ coefficients_squared
    rho_third = (2 * (3 * first * second + fraction * third)) @ coefficients_squared
    eta_third = (2 * (3 * weight_first * weight_second + weight * weight_third)) @ coefficients_squared
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dx = rho_first / (2 * rho2)
        dy = eta_first / (2 * eta2)
        ddx = rho_second / (2 * rho2) - 2 * dx ** 2
        ddy = eta_second / (2 * eta2) - 2 * dy ** 2
        dddx = rho_third / (2 * rho2) - 3 * rho_second * rho_first / (2 * rho2 ** 2) + (rho_first / rho2) ** 3
        dddy = eta_third / (2 * eta2) - 3 * eta_second * eta_first / (2 * eta2 ** 2) + (eta_first / eta2) ** 3
        speed2 = dx ** 2 + dy ** 2
        numerator = dx * ddy - dy * ddx
        curvature = numerator / speed2 ** 1.5
        derivative = ((dx * dddy - dy * dddx) * speed2 - 3 * numerator * (dx * ddx + dy * ddy)) / speed2 ** 2.5
    return (
        np.sqrt(rho2), np.sqrt(eta2),
        np.where(np.isfinite(curvature), curvature, -np.inf), derivative,
    )


def _select_lcurve_corner(
    curvature: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int8]]:
    """Select the strict interior maximum nearest the upper grid endpoint.

    Parameters
    ----------
    curvature : np.ndarray
        Signed curvature, shape (n_lambdas, n_voxels). Undefined samples
        cannot bracket an extremum.

    Returns
    -------
    indices, status : np.ndarray
        Per-voxel grid indices and selection codes. Either curvature sign is
        eligible; no strict interior maximum selects the upper endpoint.
    """
    eligible = np.zeros_like(curvature, dtype=bool)
    eligible[1:-1] = (
        np.isfinite(curvature[:-2])
        & np.isfinite(curvature[1:-1])
        & np.isfinite(curvature[2:])
        & (curvature[1:-1] > curvature[:-2])
        & (curvature[1:-1] > curvature[2:])
    )
    found = np.any(eligible, axis=0)
    indices = curvature.shape[0] - 1 - np.argmax(eligible[::-1], axis=0)
    chosen = curvature[indices, np.arange(curvature.shape[1])]
    status = np.full(found.shape, TikhonovSelection.UPPER_ENDPOINT, dtype=np.int8)
    status[found & (chosen > 0)] = TikhonovSelection.POSITIVE_CORNER
    status[found & (chosen <= 0)] = TikhonovSelection.NONPOSITIVE_CORNER
    return indices, status


def _fit_tikhonov_curves(
    curves: npt.NDArray[np.floating],
    system: _TikhonovSystem,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int8]]:
    """Fit signed full-N residues using discrete L-curve selection."""
    projected = system.projection @ curves.T
    coefficients = system.left.T @ projected
    perpendicular = np.sum((projected - system.left @ coefficients) ** 2, axis=0)
    _, _, curvature, _ = _tikhonov_lcurve(
        system.singular, coefficients ** 2, perpendicular, system.lambdas,
    )
    picks, status = _select_lcurve_corner(curvature)
    selected = system.lambdas[picks].copy()

    weights = system.singular[:, None] / (
        system.singular[:, None] ** 2 + selected[None, :] ** 2
    )
    residue = (system.back @ (weights * coefficients) + system.unpenalized @ curves.T).T
    inactive = ~np.any(curves != 0.0, axis=1)
    selected[inactive] = 0.0
    status[inactive] = TikhonovSelection.INACTIVE
    return residue, selected, status


def _tikhonov_solve(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float,
    lambda_min: float,
    lambda_max: float,
    L: npt.NDArray[np.floating],
) -> TikhonovDeconvolutionResult:
    """Solve one curve and integrate its full signed finite residue."""
    tissue = np.asarray(tissue_curve, dtype=np.float64)
    arterial = np.asarray(aif, dtype=np.float64)
    if tissue.ndim != 1:
        raise ValueError("tissue_curve must be one-dimensional")
    _validate_tikhonov_inputs(tissue, arterial, dt, lambda_min, lambda_max)
    system = _prepare_tikhonov_system(arterial, dt, lambda_min, lambda_max, L)
    raw, selected, status = _fit_tikhonov_curves(tissue[None, :], system)
    residual_norm = float(np.sum((system.matrix @ raw[0] - tissue) ** 2))
    solution_norm = float(np.sum((system.penalty @ raw[0]) ** 2))
    residue = raw[0]
    cbf = float(np.max(residue))
    cbv = float(np.trapezoid(residue, dx=dt))
    return TikhonovDeconvolutionResult(
        residue=residue, cbf=cbf, cbv=cbv,
        mtt=cbv / cbf if cbf > EPSILON_CONCENTRATION and cbv > 0 else np.nan,
        lambda_optimal=float(selected[0]), residual_norm=residual_norm,
        solution_norm=solution_norm, selection_status=TikhonovSelection(status[0]),
    )


def tikhonov_deconvolve(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    lambda_min: float = DEFAULT_LAMBDA_RANGE.min,
    lambda_max: float = DEFAULT_LAMBDA_RANGE.max,
) -> TikhonovDeconvolutionResult:
    """Perform first-order Tikhonov deconvolution with L-curve selection.

    The forward model uses Volterra product integration. A modified
    first-difference penalty constrains the magnitude of the leading two
    residue samples and differences between later samples. Selection uses the
    strict interior curvature maximum nearest the upper lambda bound, allowing
    either curvature sign. The sweep uses 100 log-spaced values without
    continuous refinement. If no such maximum exists, the upper endpoint is
    selected. Zero tissue curves are inactive and have zero lambda/residue.

    Parameters
    ----------
    tissue_curve, aif : np.ndarray
        Finite one-dimensional concentration curves of equal length.
    dt : float, optional
        Positive frame interval in seconds.
    lambda_min, lambda_max : float, optional
        Positive, increasing search bounds as fractions of the largest
        singular value of the forward matrix.

    Returns
    -------
    TikhonovDeconvolutionResult
        Signed flow-scaled residue over all N lags. CBV is its finite
        trapezoidal integral; residue MTT is CBV/CBF, or NaN for nonpositive
        area or a peak at or below the concentration tolerance. Lags without
        forward sensitivity are determined by the penalty and do not recover
        an unobserved physiological tail. The interface reports AUC-CBV/CBF
        as its primary MTT. This estimator is sensitive to bolus delay.

    Raises
    ------
    ValueError
        If curves, sampling or bounds are invalid, or the forward matrix is zero.
    """
    tissue = np.asarray(tissue_curve, dtype=np.float64)
    if tissue.ndim != 1:
        raise ValueError("tissue_curve must be one-dimensional")
    return _tikhonov_solve(
        tissue, aif, dt, lambda_min, lambda_max,
        _build_modified_first_derivative_matrix(len(tissue)),
    )


def tikhonov_deconvolve_batch(
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    lambda_min: float = DEFAULT_LAMBDA_RANGE.min,
    lambda_max: float = DEFAULT_LAMBDA_RANGE.max,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating], npt.NDArray[np.floating],
    npt.NDArray[np.floating], npt.NDArray[np.floating],
    npt.NDArray[np.floating], npt.NDArray[np.int8],
]:
    """Fit Tikhonov residues using a shared standard-form SVD.

    Parameters
    ----------
    tissue_curves : np.ndarray
        Concentration curves with spatial dimensions followed by time.
    aif : np.ndarray
        Finite one-dimensional arterial concentration curve.
    dt : float, optional
        Positive frame interval in seconds.
    lambda_min, lambda_max : float, optional
        Positive, increasing search bounds as fractions of the forward
        matrix's largest singular value.
    mask : np.ndarray, optional
        Boolean mask matching the spatial dimensions. Only finite selected
        curves are fitted; excluded outputs are zero.

    Returns
    -------
    residue, cbf, cbv, mtt, lambda_optimal, selection_status : np.ndarray
        Signed residues retain all N samples. CBV uses finite trapezoidal
        integration; residue MTT is NaN for a nonpositive area or a peak at or
        below the concentration tolerance. Other arrays have the input spatial
        shape. Status codes are given by TikhonovSelection; zero tissue curves
        and excluded voxels are inactive, with zero lambda/residue. Excluded
        MTT values are zero. Selection uses the same discrete L-curve rule as the
        scalar function. Nonfinite selected tissue curves have INVALID_INPUT
        status and NaN residue, parameters and lambda, without affecting other
        voxels.

    Raises
    ------
    ValueError
        If spatial/time dimensions, mask shape, AIF, sampling or bounds are
        invalid, or the shared forward matrix is zero.
    """
    tissue = np.asarray(tissue_curves, dtype=np.float64)
    arterial = np.asarray(aif, dtype=np.float64)
    if tissue.ndim < 2:
        raise ValueError("tissue_curves must have spatial and time dimensions")
    spatial_shape = tissue.shape[:-1]
    if mask is not None and mask.shape != spatial_shape:
        raise ValueError("mask shape must match the tissue spatial dimensions")
    mask_flat = np.ones(int(np.prod(spatial_shape)), dtype=bool) if mask is None else np.asarray(mask, dtype=bool).ravel()
    curves = tissue.reshape((mask_flat.size, tissue.shape[-1]))
    finite = np.all(np.isfinite(curves), axis=-1)
    valid = mask_flat & finite
    invalid = mask_flat & ~finite
    _validate_tikhonov_inputs(curves[valid], arterial, dt, lambda_min, lambda_max)
    system = _prepare_tikhonov_system(arterial, dt, lambda_min, lambda_max)
    residue = np.zeros_like(curves)
    cbf = np.zeros(mask_flat.size)
    cbv = np.zeros(mask_flat.size)
    mtt = np.zeros(mask_flat.size)
    selected = np.zeros(mask_flat.size)
    status = np.zeros(mask_flat.size, dtype=np.int8)
    for values in (residue, cbf, cbv, mtt, selected):
        values[invalid] = np.nan
    status[invalid] = TikhonovSelection.INVALID_INPUT
    indices = np.flatnonzero(valid)
    for start in range(0, indices.size, 256):
        subset = indices[start:start + 256]
        active, selected[subset], status[subset] = _fit_tikhonov_curves(curves[subset], system)
        residue[subset] = active
        cbf[subset] = np.max(active, axis=1)
        cbv[subset] = np.trapezoid(active, dx=dt, axis=1)
        mtt[subset] = np.divide(
            cbv[subset], cbf[subset], out=np.full(len(subset), np.nan),
            where=(cbf[subset] > EPSILON_CONCENTRATION) & (cbv[subset] > 0),
        )
    return (
        residue.reshape(tissue.shape), cbf.reshape(spatial_shape),
        cbv.reshape(spatial_shape), mtt.reshape(spatial_shape),
        selected.reshape(spatial_shape), status.reshape(spatial_shape),
    )
