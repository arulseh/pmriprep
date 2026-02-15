"""DSC deconvolution methods.

SVD-based deconvolution methods for DSC-MRI perfusion quantification,
including standard SVD (sSVD), circulant SVD (cSVD), oscillation-index
limited SVD (oSVD), and Tikhonov regularization.

The deconvolution problem is:
    C_tissue(t) = CBF × [R(t) ⊗ AIF(t)]

where R(t) is the residue function (impulse response), which encodes
the tissue microvascular transit time distribution.

References:
    Østergaard et al. (1996) High resolution measurement of cerebral
    blood flow using intravascular tracer bolus passages. Part I.
    Magn Reson Med 36(5):715-725.

    Wu et al. (2003) Tracer arrival timing-insensitive technique for
    estimating flow in MR perfusion-weighted imaging using singular
    value decomposition with a block-circulant deconvolution matrix.
    Magn Reson Med 50(1):164-174.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import linalg
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar

from pmriprep.utils.dsc_constants import (
    DEFAULT_LAMBDA_RANGE,
    DEFAULT_OI_THRESHOLD,
    DEFAULT_SVD_THRESHOLD,
    EPSILON_CONCENTRATION,
)


@dataclass
class DeconvolutionResult:
    """Result from SVD deconvolution.

    Attributes:
        residue: Residue function R(t) with same length as input curves.
            The maximum value corresponds to CBF.
        cbf: Cerebral blood flow, max(R(t)).
        cbv: Cerebral blood volume, integral of R(t).
        mtt: Mean transit time (seconds), CBV/CBF via central volume theorem.
        threshold_used: SVD truncation threshold that was applied.
        n_singular_values: Number of singular values retained after truncation.
        condition_number: Condition number of the truncated matrix.
    """

    residue: npt.NDArray[np.floating]
    cbf: float
    cbv: float
    mtt: float
    threshold_used: float
    n_singular_values: int
    condition_number: float

    def get_tmax(self, dt: float = 1.0) -> float:
        """Get time to maximum of residue function.

        Args:
            dt: Time step in seconds (default 1.0 for frame units).

        Returns:
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(np.argmax(self.residue)) * dt


def build_toeplitz_matrix(
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Build Toeplitz convolution matrix from AIF.

    Creates the lower-triangular Toeplitz matrix A such that:
        C_tissue = A @ R

    where the convolution integral is discretized using trapezoidal
    weighting, consistent with the Volterra discretization described
    by Østergaard (1996) and UMMPerfusion (Sourbron PMC3597952).

    Trapezoidal weighting applies 0.5× weight to the first and last
    points of each row's integration interval, and 1.0× to interior
    points.

    Args:
        aif: Arterial input function concentration curve, shape (n_timepoints,).
        dt: Time step in seconds for integration.

    Returns:
        Toeplitz matrix of shape (n_timepoints, n_timepoints).

    References:
        Østergaard L, et al. Magn Reson Med. 1996;36(5):715-725.
        Sourbron S, Luypaert R. Phys Med Biol. 2007;52:6761-6783.
    """
    n = len(aif)
    matrix = np.zeros((n, n), dtype=np.float64)

    # Build lower-triangular Toeplitz matrix with trapezoidal weighting
    for i in range(n):
        for j in range(i + 1):
            # Trapezoidal rule: half-weight at endpoints (j==0, j==i)
            if j == 0 or j == i:
                weight = 0.5
            else:
                weight = 1.0
            matrix[i, j] = aif[i - j] * weight * dt

    return matrix


def svd_deconvolve_standard(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    threshold: float = DEFAULT_SVD_THRESHOLD,
    apply_non_negativity: bool = True,
) -> DeconvolutionResult:
    """Standard SVD deconvolution (sSVD) using the Østergaard method.

    Performs deconvolution by constructing a Toeplitz convolution matrix
    from the AIF and solving via truncated SVD. Singular values below
    a fraction of the maximum are set to zero for regularization.

    The deconvolution equation is:
        C_tissue(t) = CBF × ∫_0^t R(t-τ) × AIF(τ) dτ

    In matrix form:
        C = A @ R

    where A is the Toeplitz matrix built from AIF.

    Args:
        tissue_curve: Tissue concentration curve, shape (n_timepoints,).
        aif: Arterial input function, shape (n_timepoints,).
            Must have same length as tissue_curve.
        dt: Time step in seconds. Default 1.0 gives results in frame units.
            For absolute quantification, provide actual TR.
        threshold: SVD truncation threshold as fraction of maximum singular
            value. Singular values below threshold × max(s) are set to zero.
            Default is 0.2 (20% of max). Higher values = more regularization.
        apply_non_negativity: If True, apply non-negativity constraint to
            residue function by clipping negative values to zero.
            Default is True.

    Returns:
        DeconvolutionResult containing residue function and perfusion parameters.

    Raises:
        ValueError: If tissue_curve and aif have different lengths.
        ValueError: If arrays are empty.
        ValueError: If threshold is not in (0, 1).

    Notes:
        - CBF is extracted as max(R(t))
        - CBV is computed as integral of R(t) using trapezoidal rule
        - MTT = CBV / CBF (central volume theorem)
        - This method is NOT delay-insensitive; delays cause CBF underestimation
        - For delay-insensitive deconvolution, use svd_deconvolve_circulant()

    References:
        Østergaard L, Weisskoff RM, Chesler DA, Gyldensted C, Rosen BR.
        High resolution measurement of cerebral blood flow using
        intravascular tracer bolus passages. Part I: Mathematical approach
        and statistical analysis. Magn Reson Med. 1996;36(5):715-725.
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

    # Build Toeplitz convolution matrix
    matrix = build_toeplitz_matrix(aif, dt)

    # Perform SVD
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Truncate singular values below threshold
    s_max = s[0] if len(s) > 0 else 1.0
    truncation_threshold = threshold * s_max

    # Count retained singular values
    n_retained = int(np.sum(s > truncation_threshold))
    if n_retained == 0:
        n_retained = 1  # Keep at least one singular value

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
    # More numerically stable: R = vh.T @ (s_inv * (u.T @ tissue_curve))
    residue = vh.T @ (s_inv * (u.T @ tissue_curve))

    # Apply non-negativity constraint
    if apply_non_negativity:
        residue = np.maximum(residue, 0.0)

    # Extract perfusion parameters
    cbf = float(np.max(residue))

    # CBV via trapezoidal integration of residue (Sourbron & Luypaert 2007)
    cbv = float(np.trapezoid(residue, dx=dt))

    # MTT from central volume theorem: MTT = CBV / CBF
    if cbf > EPSILON_CONCENTRATION:
        mtt = cbv / cbf
    else:
        mtt = 0.0

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
    apply_non_negativity: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # residues
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
]:
    """Batch sSVD deconvolution for multiple voxels.

    Efficiently deconvolves multiple tissue curves by precomputing
    the SVD decomposition of the Toeplitz matrix once.

    Args:
        tissue_curves: Tissue concentration curves, shape (n_voxels, n_timepoints)
            or (x, y, z, n_timepoints) for 4D data.
        aif: Arterial input function, shape (n_timepoints,).
        dt: Time step in seconds.
        threshold: SVD truncation threshold (default 0.2).
        apply_non_negativity: Apply non-negativity constraint on residue.
        mask: Optional boolean mask of voxels to process. If None, process all.
            Shape must match spatial dimensions of tissue_curves.

    Returns:
        Tuple of (residues, cbf, cbv, mtt) arrays with same spatial shape as input.
    """
    # Store original shape for reshaping output
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]

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

    # Build Toeplitz matrix and compute SVD once
    matrix = build_toeplitz_matrix(aif, dt)
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

        # Apply non-negativity
        if apply_non_negativity:
            residues_masked = np.maximum(residues_masked, 0.0)

        # Extract parameters
        cbf_masked = np.max(residues_masked, axis=1)

        # CBV via trapezoidal integration (Sourbron & Luypaert 2007)
        cbv_masked = np.trapezoid(residues_masked, dx=dt, axis=1)

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

    # Reshape to original spatial dimensions
    if tissue_curves.ndim > 2:
        residues = residues.reshape(original_shape)
        cbf = cbf.reshape(spatial_shape)
        cbv = cbv.reshape(spatial_shape)
        mtt = mtt.reshape(spatial_shape)

    return residues, cbf, cbv, mtt


# =============================================================================
# Circulant SVD (cSVD) - Delay-Insensitive Deconvolution
# =============================================================================


@dataclass
class CirculantDeconvolutionResult:
    """Result from circulant SVD deconvolution.

    Extends the standard DeconvolutionResult with delay estimation,
    which is uniquely available from the block-circulant approach.

    Attributes:
        residue: Residue function R(t) with length 2N-1 where N is input length.
            The maximum value corresponds to CBF. The delay is encoded in the
            position of the maximum.
        cbf: Cerebral blood flow, max(R(t)).
        cbv: Cerebral blood volume, integral of the positive portion of R(t).
        mtt: Mean transit time (seconds), CBV/CBF via central volume theorem.
        delay: Estimated delay between AIF and tissue in seconds.
            Positive = tissue delayed relative to AIF.
            Negative = tissue arrives before AIF (e.g., proximal to AIF sampling site).
        delay_frames: Delay in frame units (before multiplication by dt).
        threshold_used: SVD truncation threshold that was applied.
        n_singular_values: Number of singular values retained after truncation.
        condition_number: Condition number of the truncated matrix.
    """

    residue: npt.NDArray[np.floating]
    cbf: float
    cbv: float
    mtt: float
    delay: float
    delay_frames: int
    threshold_used: float
    n_singular_values: int
    condition_number: float

    def get_tmax(self, dt: float = 1.0) -> float:
        """Get time to maximum of residue function relative to delay.

        For cSVD, this returns the time from bolus arrival (accounting for delay)
        to the maximum of the residue function.

        Args:
            dt: Time step in seconds (default 1.0 for frame units).

        Returns:
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(np.argmax(self.residue)) * dt


def build_block_circulant_matrix(
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Build block-circulant convolution matrix for delay-insensitive deconvolution.

    Creates a circulant matrix by zero-padding the AIF to length 2N-1 and
    constructing a circulant (periodic) convolution matrix. This allows
    the deconvolution to estimate arbitrary positive or negative delays
    between the AIF and tissue curve.

    The block-circulant matrix approach treats the convolution as circular,
    so that delays "wrap around" rather than causing boundary artifacts.

    Args:
        aif: Arterial input function concentration curve, shape (n_timepoints,).
        dt: Time step in seconds for integration weighting.

    Returns:
        Block-circulant matrix of shape (2N-1, 2N-1) where N = len(aif).

    References:
        Wu O, Østergaard L, Weisskoff RM, Benner T, Rosen BR, Sorensen AG.
        Tracer arrival timing-insensitive technique for estimating flow in
        MR perfusion-weighted imaging using singular value decomposition
        with a block-circulant deconvolution matrix.
        Magn Reson Med. 2003;50(1):164-174.
    """
    n = len(aif)
    # Padded length for full delay range
    m = 2 * n - 1

    # Zero-pad AIF to length 2N-1
    aif_padded = np.zeros(m, dtype=np.float64)
    aif_padded[:n] = aif

    # Build circulant matrix where each row is a cyclic shift
    # Row i has aif_padded shifted by i positions
    matrix = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            # Circulant: element (i,j) = aif_padded[(i-j) mod m]
            idx = (i - j) % m
            matrix[i, j] = aif_padded[idx]

    # Apply rectangular integration weighting for circulant matrix.
    # The block-circulant structure uses uniform (rectangular) weighting
    # because the cyclic shift pattern is incompatible with trapezoidal
    # endpoint weighting.
    matrix *= dt

    return matrix


def svd_deconvolve_circulant(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    threshold: float = DEFAULT_SVD_THRESHOLD,
    apply_non_negativity: bool = True,
) -> CirculantDeconvolutionResult:
    """Circulant SVD deconvolution (cSVD) for delay-insensitive estimation.

    Performs deconvolution using a block-circulant matrix approach that is
    insensitive to delays between the AIF and tissue curve. The method
    zero-pads both the AIF and tissue curve to length 2N-1 and constructs
    a circulant convolution matrix, allowing the residue function to peak
    at any time point (positive or negative delay).

    The deconvolution equation is:
        C_tissue(t) = CBF × ∫_{-∞}^{∞} R(t-τ) × AIF(τ) dτ

    The delay is extracted from the position of the residue maximum:
        delay = argmax(R(t)) - (N-1)

    where N is the original curve length and (N-1) is the zero-delay position.

    Args:
        tissue_curve: Tissue concentration curve, shape (n_timepoints,).
        aif: Arterial input function, shape (n_timepoints,).
            Must have same length as tissue_curve.
        dt: Time step in seconds. Default 1.0 gives delay in frame units.
            For absolute timing, provide actual TR.
        threshold: SVD truncation threshold as fraction of maximum singular
            value. Singular values below threshold × max(s) are set to zero.
            Default is 0.2 (20% of max). Higher values = more regularization.
        apply_non_negativity: If True, apply non-negativity constraint to
            residue function by clipping negative values to zero.
            Default is True.

    Returns:
        CirculantDeconvolutionResult containing residue function, perfusion
        parameters, and estimated delay.

    Raises:
        ValueError: If tissue_curve and aif have different lengths.
        ValueError: If arrays are empty.
        ValueError: If threshold is not in (0, 1).

    Notes:
        - The residue function has length 2N-1, allowing delays from -(N-1) to +(N-1)
        - CBF is extracted as max(R(t))
        - CBV is computed as integral of R(t) using trapezoidal rule
        - MTT = CBV / CBF (central volume theorem)
        - Delay is estimated from the position of max(R(t))
        - For large delays, consider if the delay is physiologically reasonable

    References:
        Wu O, Østergaard L, Weisskoff RM, Benner T, Rosen BR, Sorensen AG.
        Tracer arrival timing-insensitive technique for estimating flow in
        MR perfusion-weighted imaging using singular value decomposition
        with a block-circulant deconvolution matrix.
        Magn Reson Med. 2003;50(1):164-174.
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
    m = 2 * n - 1  # Padded length

    # Zero-pad tissue curve to length 2N-1
    tissue_padded = np.zeros(m, dtype=np.float64)
    tissue_padded[:n] = tissue_curve

    # Build block-circulant matrix
    matrix = build_block_circulant_matrix(aif, dt)

    # Perform SVD
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Truncate singular values below threshold
    s_max = s[0] if len(s) > 0 else 1.0
    truncation_threshold = threshold * s_max

    # Count retained singular values
    n_retained = int(np.sum(s > truncation_threshold))
    if n_retained == 0:
        n_retained = 1  # Keep at least one singular value

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

    # Apply non-negativity constraint
    if apply_non_negativity:
        residue = np.maximum(residue, 0.0)

    # Extract perfusion parameters
    cbf = float(np.max(residue))

    # Extract delay from position of maximum
    # The zero-delay position is at index (n-1) in the padded residue
    # Delay = peak_position - (n-1)
    peak_position = int(np.argmax(residue))
    delay_frames = peak_position - (n - 1)
    delay_seconds = float(delay_frames) * dt

    # CBV via trapezoidal integration of residue (Sourbron & Luypaert 2007)
    cbv = float(np.trapezoid(residue, dx=dt))

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
    apply_non_negativity: bool = True,
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

    Args:
        tissue_curves: Tissue concentration curves, shape (n_voxels, n_timepoints)
            or (x, y, z, n_timepoints) for 4D data.
        aif: Arterial input function, shape (n_timepoints,).
        dt: Time step in seconds.
        threshold: SVD truncation threshold (default 0.2).
        apply_non_negativity: Apply non-negativity constraint on residue.
        mask: Optional boolean mask of voxels to process. If None, process all.
            Shape must match spatial dimensions of tissue_curves.

    Returns:
        Tuple of (residues, cbf, cbv, mtt, delay) arrays.
        - residues has shape (..., 2N-1) where N is the number of timepoints
        - cbf, cbv, mtt, delay have the same spatial shape as input
    """
    # Store original shape for reshaping output
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]
    m = 2 * n_timepoints - 1  # Padded length

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

        # Zero-pad tissue curves to length 2N-1
        tissue_padded = np.zeros((len(masked_indices), m), dtype=np.float64)
        tissue_padded[:, :n_timepoints] = tissue_masked

        # Batch deconvolution: R_batch = vh.T @ diag(s_inv) @ u.T @ C_batch.T
        ut_tissue = u.T @ tissue_padded.T  # (m, n_vox)
        scaled = s_inv[:, None] * ut_tissue  # (m, n_vox)
        residues_masked = (vh.T @ scaled).T  # (n_vox, m)

        # Apply non-negativity
        if apply_non_negativity:
            residues_masked = np.maximum(residues_masked, 0.0)

        # Extract parameters
        cbf_masked = np.max(residues_masked, axis=1)

        # Extract delays from peak positions
        peak_positions = np.argmax(residues_masked, axis=1)
        delay_frames = peak_positions - (n_timepoints - 1)
        delay_masked = delay_frames.astype(np.float64) * dt

        # CBV via trapezoidal integration (Sourbron & Luypaert 2007)
        cbv_masked = np.trapezoid(residues_masked, dx=dt, axis=1)

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
        residues = residues.reshape(residue_shape)
        cbf = cbf.reshape(spatial_shape)
        cbv = cbv.reshape(spatial_shape)
        mtt = mtt.reshape(spatial_shape)
        delay = delay.reshape(spatial_shape)

    return residues, cbf, cbv, mtt, delay


# =============================================================================
# Oscillation-Index Limited SVD (oSVD)
# =============================================================================


@dataclass
class OscillationDeconvolutionResult:
    """Result from oscillation-index limited SVD deconvolution.

    Extends the standard DeconvolutionResult with oscillation index information,
    which is uniquely available from the oSVD approach.

    Attributes:
        residue: Residue function R(t) with same length as input curves.
            The maximum value corresponds to CBF.
        cbf: Cerebral blood flow, max(R(t)).
        cbv: Cerebral blood volume, integral of R(t).
        mtt: Mean transit time (seconds), CBV/CBF via central volume theorem.
        oscillation_index: The computed oscillation index of the residue function.
            OI = (1/N) × Σ|R(t+1) - R(t)| / max(R).
            Lower values indicate smoother residue functions.
        threshold_used: The optimal SVD truncation threshold found by binary search.
        n_singular_values: Number of singular values retained after truncation.
        condition_number: Condition number of the truncated matrix.
        oi_target: The target oscillation index threshold that was used.
        converged: Whether the binary search converged to find a valid threshold.
    """

    residue: npt.NDArray[np.floating]
    cbf: float
    cbv: float
    mtt: float
    oscillation_index: float
    threshold_used: float
    n_singular_values: int
    condition_number: float
    oi_target: float
    converged: bool

    def get_tmax(self, dt: float = 1.0) -> float:
        """Get time to maximum of residue function.

        Args:
            dt: Time step in seconds (default 1.0 for frame units).

        Returns:
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(np.argmax(self.residue)) * dt


def compute_oscillation_index(residue: npt.NDArray[np.floating]) -> float:
    """Compute the oscillation index of a residue function.

    The oscillation index (OI) measures the smoothness of the residue function.
    It is defined as the mean absolute first-order difference normalized by
    the maximum value:

        OI = (1/N) × Σ|R(t+1) - R(t)| / max(R)

    where N is the number of timepoints.

    A lower OI indicates a smoother residue function with fewer oscillations.
    Typical target: OI ≤ 0.065 provides good balance between noise suppression
    and preserving physiological detail.

    Args:
        residue: Residue function, shape (n_timepoints,).

    Returns:
        Oscillation index (dimensionless, typically in range 0-0.5).
        Returns 0.0 if residue is all zeros.

    References:
        Wu O, Østergaard L, Korber M, Yenari M, Weisskoff RM, Sorensen AG.
        Oscillation index: a novel technique for measurement of cerebral
        blood flow using an oscillation-index-driven deconvolution method.
        ISMRM 2000.
    """
    # Handle edge cases
    if len(residue) < 2:
        return 0.0

    max_residue = float(np.max(residue))
    if max_residue <= EPSILON_CONCENTRATION:
        return 0.0

    # Compute first-order differences
    diff = np.abs(np.diff(residue))

    # Sum of absolute differences normalized by length and maximum
    # This matches perf-plus: OI = sum(diff) / (N * max_residue)
    # Using len(residue) (not len(diff)) as the normalizer
    oi = float(np.sum(diff)) / (len(residue) * max_residue)

    return oi


def _deconvolve_with_threshold(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float,
    threshold: float,
    matrix: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    s: npt.NDArray[np.floating],
    vh: npt.NDArray[np.floating],
    apply_non_negativity: bool,
) -> tuple[npt.NDArray[np.floating], float, int, float]:
    """Helper function to deconvolve with a given threshold.

    Args:
        tissue_curve: Tissue concentration curve.
        aif: Arterial input function.
        dt: Time step.
        threshold: SVD truncation threshold.
        matrix: Pre-computed Toeplitz matrix.
        u, s, vh: Pre-computed SVD decomposition.
        apply_non_negativity: Whether to clip negative values.

    Returns:
        Tuple of (residue, oscillation_index, n_retained, condition_number).
    """
    # Truncate singular values below threshold
    s_max = s[0] if len(s) > 0 else 1.0
    truncation_threshold = threshold * s_max

    # Count retained singular values
    n_retained = int(np.sum(s > truncation_threshold))
    if n_retained == 0:
        n_retained = 1  # Keep at least one singular value

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
    residue = vh.T @ (s_inv * (u.T @ tissue_curve))

    # Apply non-negativity constraint
    if apply_non_negativity:
        residue = np.maximum(residue, 0.0)

    # Compute oscillation index
    oi = compute_oscillation_index(residue)

    return residue, oi, n_retained, condition_number


def svd_deconvolve_oscillation(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    oi_target: float = DEFAULT_OI_THRESHOLD,
    threshold_min: float = 0.01,
    threshold_max: float = 0.50,
    max_iterations: int = 20,
    apply_non_negativity: bool = True,
) -> OscillationDeconvolutionResult:
    """Oscillation-index limited SVD deconvolution (oSVD).

    Performs deconvolution using a truncated SVD approach where the truncation
    threshold is automatically selected to keep the oscillation index (OI)
    of the residue function at or below a target value. This provides adaptive
    regularization that balances noise suppression against preserving
    physiological detail.

    The oscillation index is defined as:
        OI = (1/N) × Σ|R(t+1) - R(t)| / max(R)

    A linear search from low to high threshold is used to find the first
    threshold that achieves OI ≤ oi_target, matching the perf-plus
    implementation.

    Strategy:
        - Lower threshold = more singular values retained = more detail but more noise
        - Higher threshold = fewer singular values = smoother but may lose detail
        - Linear search finds the first (lowest) threshold where OI ≤ target

    Args:
        tissue_curve: Tissue concentration curve, shape (n_timepoints,).
        aif: Arterial input function, shape (n_timepoints,).
            Must have same length as tissue_curve.
        dt: Time step in seconds. Default 1.0 gives results in frame units.
            For absolute quantification, provide actual TR.
        oi_target: Target oscillation index threshold. Residue functions with
            OI above this value are considered too oscillatory. Default is 0.065.
        threshold_min: Minimum SVD threshold to consider. Default 0.01.
        threshold_max: Maximum SVD threshold to consider. Default 0.50.
        max_iterations: Number of evenly-spaced thresholds in linear
            search. Default 20.
        apply_non_negativity: If True, apply non-negativity constraint to
            residue function by clipping negative values to zero.
            Default is True.

    Returns:
        OscillationDeconvolutionResult containing residue function, perfusion
        parameters, oscillation index, and optimal threshold.

    Raises:
        ValueError: If tissue_curve and aif have different lengths.
        ValueError: If arrays are empty.
        ValueError: If oi_target is not positive.

    Notes:
        - CBF is extracted as max(R(t))
        - CBV is computed as integral of R(t) using trapezoidal rule
        - MTT = CBV / CBF (central volume theorem)
        - If OI cannot be reduced to oi_target even at threshold_max, returns
          the result at threshold_max with converged=False
        - This method is NOT delay-insensitive; for delay estimation, consider
          applying oSVD after cSVD delay correction

    References:
        Wu O, Østergaard L, Korber M, Yenari M, Weisskoff RM, Sorensen AG.
        Oscillation index: a novel technique for measurement of cerebral
        blood flow using an oscillation-index-driven deconvolution method.
        ISMRM 2000.
    """
    # Validate inputs
    if len(tissue_curve) != len(aif):
        raise ValueError(
            f"tissue_curve and aif must have same length, "
            f"got {len(tissue_curve)} and {len(aif)}"
        )

    if len(tissue_curve) == 0:
        raise ValueError("Arrays cannot be empty")

    if oi_target <= 0:
        raise ValueError(f"oi_target must be positive, got {oi_target}")

    n = len(tissue_curve)

    # Build Toeplitz convolution matrix and compute SVD once
    matrix = build_toeplitz_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Linear search for optimal threshold (matches perf-plus)
    # Search from LOW to HIGH, take the FIRST threshold where OI ≤ oi_target.
    # This finds the lowest acceptable threshold (least regularization).
    n_steps = max_iterations
    thresholds = np.linspace(threshold_min, threshold_max, n_steps)

    best_residue: Optional[npt.NDArray[np.floating]] = None
    best_oi = float("inf")
    best_threshold = threshold_max
    best_n_ret = 0
    best_cond = float("inf")
    converged = False

    for thresh in thresholds:
        residue, oi, n_ret, cond = _deconvolve_with_threshold(
            tissue_curve, aif, dt, thresh, matrix, u, s, vh,
            apply_non_negativity,
        )

        if oi <= oi_target:
            best_residue = residue
            best_oi = oi
            best_threshold = thresh
            best_n_ret = n_ret
            best_cond = cond
            converged = True
            break

    # If no threshold achieved target, use maximum regularization
    if best_residue is None:
        best_residue, best_oi, best_n_ret, best_cond = (
            _deconvolve_with_threshold(
                tissue_curve, aif, dt, threshold_max, matrix, u, s, vh,
                apply_non_negativity,
            )
        )
        best_threshold = threshold_max
        converged = False

    # Extract perfusion parameters from best result
    cbf = float(np.max(best_residue))
    cbv = float(np.trapezoid(best_residue, dx=dt))
    mtt = cbv / cbf if cbf > EPSILON_CONCENTRATION else 0.0

    return OscillationDeconvolutionResult(
        residue=best_residue,
        cbf=cbf,
        cbv=cbv,
        mtt=mtt,
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
    max_iterations: int = 20,
    apply_non_negativity: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # residues
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
    npt.NDArray[np.floating],  # oscillation_index
    npt.NDArray[np.floating],  # threshold_used
]:
    """Batch oSVD deconvolution for multiple voxels.

    Deconvolves multiple tissue curves using the oscillation-index limited
    SVD approach. Each voxel gets an individually optimized threshold.

    Note: This function processes voxels sequentially because each voxel
    may require a different optimal threshold. For faster processing with
    a fixed threshold, use svd_deconvolve_standard_batch().

    Args:
        tissue_curves: Tissue concentration curves, shape (n_voxels, n_timepoints)
            or (x, y, z, n_timepoints) for 4D data.
        aif: Arterial input function, shape (n_timepoints,).
        dt: Time step in seconds.
        oi_target: Target oscillation index threshold. Default 0.065.
        threshold_min: Minimum SVD threshold. Default 0.01.
        threshold_max: Maximum SVD threshold. Default 0.50.
        max_iterations: Number of linear search steps per voxel. Default 20.
        apply_non_negativity: Apply non-negativity constraint on residue.
        mask: Optional boolean mask of voxels to process. If None, process all.
            Shape must match spatial dimensions of tissue_curves.

    Returns:
        Tuple of (residues, cbf, cbv, mtt, oscillation_index, threshold_used) arrays.
        All output arrays have the same spatial shape as input (excluding time dimension).
    """
    # Store original shape for reshaping output
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]

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

    # Build Toeplitz matrix and compute SVD once (shared across all voxels)
    matrix = build_toeplitz_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Initialize output arrays
    residues = np.zeros((n_voxels, n_timepoints), dtype=np.float64)
    cbf = np.zeros(n_voxels, dtype=np.float64)
    cbv = np.zeros(n_voxels, dtype=np.float64)
    mtt = np.zeros(n_voxels, dtype=np.float64)
    oscillation_index = np.zeros(n_voxels, dtype=np.float64)
    threshold_used = np.zeros(n_voxels, dtype=np.float64)

    # Process each masked voxel with individual threshold optimization
    masked_indices = np.where(mask_flat)[0]

    # Pre-compute linear threshold grid (shared across all voxels)
    thresholds = np.linspace(threshold_min, threshold_max, max_iterations)

    for idx in masked_indices:
        tissue_curve = tissue_2d[idx, :]

        # Linear search: LOW→HIGH, stop at FIRST threshold where OI ≤ target
        best_residue = None
        best_oi = float("inf")
        best_threshold = threshold_max

        for thresh in thresholds:
            residue, oi, n_ret, cond = _deconvolve_with_threshold(
                tissue_curve, aif, dt, thresh, matrix, u, s, vh,
                apply_non_negativity,
            )
            if oi <= oi_target:
                best_residue = residue
                best_oi = oi
                best_threshold = thresh
                break

        # If no threshold achieved target, use maximum regularization
        if best_residue is None:
            best_residue, best_oi, _, _ = _deconvolve_with_threshold(
                tissue_curve, aif, dt, threshold_max, matrix, u, s, vh,
                apply_non_negativity,
            )
            best_threshold = threshold_max

        # Store results for this voxel
        residues[idx, :] = best_residue
        cbf[idx] = float(np.max(best_residue))
        cbv[idx] = float(np.trapezoid(best_residue, dx=dt))
        if cbf[idx] > EPSILON_CONCENTRATION:
            mtt[idx] = cbv[idx] / cbf[idx]
        oscillation_index[idx] = best_oi
        threshold_used[idx] = best_threshold

    # Reshape to original spatial dimensions
    if tissue_curves.ndim > 2:
        residues = residues.reshape(original_shape)
        cbf = cbf.reshape(spatial_shape)
        cbv = cbv.reshape(spatial_shape)
        mtt = mtt.reshape(spatial_shape)
        oscillation_index = oscillation_index.reshape(spatial_shape)
        threshold_used = threshold_used.reshape(spatial_shape)

    return residues, cbf, cbv, mtt, oscillation_index, threshold_used


# =============================================================================
# Tikhonov Regularization
# =============================================================================


@dataclass
class TikhonovDeconvolutionResult:
    """Result from Tikhonov regularized deconvolution.

    Extends the standard DeconvolutionResult with Tikhonov-specific information.

    Attributes:
        residue: Residue function R(t) with same length as input curves.
            The maximum value corresponds to CBF.
        cbf: Cerebral blood flow, max(R(t)).
        cbv: Cerebral blood volume, integral of R(t).
        mtt: Mean transit time (seconds), CBV/CBF via central volume theorem.
        lambda_optimal: The optimal regularization parameter λ found by GCV.
        gcv_score: The GCV score at the optimal λ.
        n_lambda_evaluated: Number of λ values evaluated during optimization.
        residual_norm: ||AR - C||² at the solution.
        regularization_norm: ||R||² at the solution.
    """

    residue: npt.NDArray[np.floating]
    cbf: float
    cbv: float
    mtt: float
    lambda_optimal: float
    gcv_score: float
    n_lambda_evaluated: int
    residual_norm: float
    regularization_norm: float

    def get_tmax(self, dt: float = 1.0) -> float:
        """Get time to maximum of residue function.

        Args:
            dt: Time step in seconds (default 1.0 for frame units).

        Returns:
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(np.argmax(self.residue)) * dt


def _compute_gcv_score(
    tissue_curve: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    s: npt.NDArray[np.floating],
    vh: npt.NDArray[np.floating],
    lambda_val: float,
) -> tuple[float, npt.NDArray[np.floating]]:
    """Compute Generalized Cross-Validation score for a given λ.

    The GCV function is defined as:
        GCV(λ) = ||AR_λ - C||² / (tr(I - A × A_λ^†))²

    where A_λ^† is the Tikhonov-regularized pseudoinverse.

    For SVD-based computation:
        GCV(λ) = Σ[(u_i^T C)² × λ²/(s_i² + λ²)²] / [Σ(λ²/(s_i² + λ²))]²

    Args:
        tissue_curve: Tissue concentration curve.
        u, s, vh: SVD decomposition of the Toeplitz matrix.
        lambda_val: Regularization parameter λ.

    Returns:
        Tuple of (GCV score, residue function at this λ).
    """
    n = len(tissue_curve)
    lambda_sq = lambda_val ** 2
    s_sq = s ** 2

    # Filter factors: f_i = s_i² / (s_i² + λ²)
    filter_factors = s_sq / (s_sq + lambda_sq)

    # Compute residue via Tikhonov regularization
    # R = V @ diag(s / (s² + λ²)) @ U^T @ C
    s_reg = s / (s_sq + lambda_sq)
    ut_c = u.T @ tissue_curve  # (n,)
    residue = vh.T @ (s_reg * ut_c)

    # Residual part of GCV: Σ[(u_i^T C)² × (1 - f_i)²]
    # = Σ[(u_i^T C)² × (λ²/(s_i² + λ²))²]
    residual_terms = (ut_c ** 2) * ((1 - filter_factors) ** 2)
    residual_sum = float(np.sum(residual_terms))

    # Denominator: (Σ(1 - f_i))² = (Σ(λ²/(s_i² + λ²)))²
    trace_term = float(np.sum(1 - filter_factors))

    # GCV = residual_sum / trace_term²
    if trace_term > 1e-12:
        gcv = residual_sum / (trace_term ** 2)
    else:
        gcv = float("inf")

    return gcv, residue


def tikhonov_deconvolve(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    lambda_min: float = DEFAULT_LAMBDA_RANGE.min,
    lambda_max: float = DEFAULT_LAMBDA_RANGE.max,
    n_lambda: int = 50,
    apply_non_negativity: bool = True,
) -> TikhonovDeconvolutionResult:
    """Tikhonov regularized deconvolution with GCV parameter selection.

    Performs deconvolution by solving the regularized least squares problem:
        minimize ||AR - C||² + λ²||R||²

    where A is the Toeplitz convolution matrix from the AIF, R is the residue
    function, C is the tissue curve, and λ is the regularization parameter.

    The optimal λ is selected using Generalized Cross-Validation (GCV), which
    minimizes the leave-one-out cross-validation error:
        GCV(λ) = ||AR_λ - C||² / (tr(I - A × A_λ^†))²

    The GCV score is minimized over λ using bounded scalar optimization
    in log10-space (scipy.optimize.minimize_scalar).

    Args:
        tissue_curve: Tissue concentration curve, shape (n_timepoints,).
        aif: Arterial input function, shape (n_timepoints,).
            Must have same length as tissue_curve.
        dt: Time step in seconds. Default 1.0 gives results in frame units.
            For absolute quantification, provide actual TR.
        lambda_min: Minimum λ value to consider. Default 1e-4.
        lambda_max: Maximum λ value to consider. Default 1.0.
        n_lambda: Unused. Kept for API compatibility.
        apply_non_negativity: If True, apply non-negativity constraint to
            residue function by clipping negative values to zero.
            Default is True.

    Returns:
        TikhonovDeconvolutionResult containing residue function, perfusion
        parameters, and optimal λ.

    Raises:
        ValueError: If tissue_curve and aif have different lengths.
        ValueError: If arrays are empty.
        ValueError: If lambda_min >= lambda_max.

    Notes:
        - CBF is extracted as max(R(t))
        - CBV is computed as integral of R(t) using trapezoidal rule
        - MTT = CBV / CBF (central volume theorem)
        - Tikhonov regularization provides smooth solutions by penalizing
          the norm of the residue function
        - GCV provides automatic parameter selection without requiring a
          noise estimate
        - This method is NOT delay-insensitive; consider cSVD for delay estimation

    References:
        Hansen PC. Rank-Deficient and Discrete Ill-Posed Problems: Numerical
        Aspects of Linear Inversion. SIAM, 1998.

        Golub GH, Heath M, Wahba G. Generalized cross-validation as a method
        for choosing a good ridge parameter. Technometrics. 1979;21(2):215-223.
    """
    # Validate inputs
    if len(tissue_curve) != len(aif):
        raise ValueError(
            f"tissue_curve and aif must have same length, "
            f"got {len(tissue_curve)} and {len(aif)}"
        )

    if len(tissue_curve) == 0:
        raise ValueError("Arrays cannot be empty")

    if lambda_min >= lambda_max:
        raise ValueError(
            f"lambda_min must be less than lambda_max, "
            f"got {lambda_min} >= {lambda_max}"
        )

    n = len(tissue_curve)

    # Build Toeplitz convolution matrix
    matrix = build_toeplitz_matrix(aif, dt)

    # Perform SVD
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Find optimal λ via continuous GCV minimization in log10-space
    def gcv_objective(log_lambda: float) -> float:
        lam = 10.0 ** log_lambda
        gcv, _ = _compute_gcv_score(tissue_curve, u, s, vh, lam)
        return gcv

    try:
        result = minimize_scalar(
            gcv_objective,
            bounds=(np.log10(lambda_min), np.log10(lambda_max)),
            method="bounded",
        )
        if result.success and np.isfinite(result.x):
            best_lambda = 10.0 ** result.x
        else:
            best_lambda = 0.1
    except Exception:
        best_lambda = 0.1

    best_gcv, best_residue = _compute_gcv_score(
        tissue_curve, u, s, vh, best_lambda
    )

    # Apply non-negativity constraint
    if apply_non_negativity:
        best_residue = np.maximum(best_residue, 0.0)

    # Compute residual and regularization norms
    predicted = matrix @ best_residue
    residual_norm = float(np.sum((predicted - tissue_curve) ** 2))
    regularization_norm = float(np.sum(best_residue ** 2))

    # Extract perfusion parameters
    cbf = float(np.max(best_residue))

    # CBV via trapezoidal integration of residue (Sourbron & Luypaert 2007)
    cbv = float(np.trapezoid(best_residue, dx=dt))

    # MTT from central volume theorem: MTT = CBV / CBF
    if cbf > EPSILON_CONCENTRATION:
        mtt = cbv / cbf
    else:
        mtt = 0.0

    return TikhonovDeconvolutionResult(
        residue=best_residue,
        cbf=cbf,
        cbv=cbv,
        mtt=mtt,
        lambda_optimal=float(best_lambda),
        gcv_score=float(best_gcv),
        n_lambda_evaluated=n_lambda,
        residual_norm=residual_norm,
        regularization_norm=regularization_norm,
    )


def tikhonov_deconvolve_batch(
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    lambda_min: float = DEFAULT_LAMBDA_RANGE.min,
    lambda_max: float = DEFAULT_LAMBDA_RANGE.max,
    n_lambda: int = 50,
    apply_non_negativity: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # residues
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
    npt.NDArray[np.floating],  # lambda_optimal
]:
    """Batch Tikhonov deconvolution for multiple voxels.

    Deconvolves multiple tissue curves using Tikhonov regularization with
    GCV-based parameter selection. Each voxel gets an individually optimized λ.

    Note: This function processes voxels sequentially because each voxel
    may require a different optimal λ.

    Args:
        tissue_curves: Tissue concentration curves, shape (n_voxels, n_timepoints)
            or (x, y, z, n_timepoints) for 4D data.
        aif: Arterial input function, shape (n_timepoints,).
        dt: Time step in seconds.
        lambda_min: Minimum λ value. Default 1e-4.
        lambda_max: Maximum λ value. Default 1.0.
        n_lambda: Unused. Kept for API compatibility.
        apply_non_negativity: Apply non-negativity constraint on residue.
        mask: Optional boolean mask of voxels to process. If None, process all.
            Shape must match spatial dimensions of tissue_curves.

    Returns:
        Tuple of (residues, cbf, cbv, mtt, lambda_optimal) arrays.
        All output arrays have the same spatial shape as input (excluding time dimension).
    """
    # Store original shape for reshaping output
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]

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

    # Build Toeplitz matrix and compute SVD once (shared across all voxels)
    matrix = build_toeplitz_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Initialize output arrays
    residues = np.zeros((n_voxels, n_timepoints), dtype=np.float64)
    cbf = np.zeros(n_voxels, dtype=np.float64)
    cbv = np.zeros(n_voxels, dtype=np.float64)
    mtt = np.zeros(n_voxels, dtype=np.float64)
    lambda_optimal = np.zeros(n_voxels, dtype=np.float64)

    # Log10 bounds for minimize_scalar
    log_lambda_min = np.log10(lambda_min)
    log_lambda_max = np.log10(lambda_max)

    # Process each masked voxel with individual λ optimization via minimize_scalar
    masked_indices = np.where(mask_flat)[0]
    for idx in masked_indices:
        tissue_curve = tissue_2d[idx, :]

        def gcv_objective(log_lambda: float) -> float:
            lam = 10.0 ** log_lambda
            gcv, _ = _compute_gcv_score(tissue_curve, u, s, vh, lam)
            return gcv

        try:
            result = minimize_scalar(
                gcv_objective,
                bounds=(log_lambda_min, log_lambda_max),
                method="bounded",
            )
            if result.success and np.isfinite(result.x):
                best_lambda = 10.0 ** result.x
            else:
                best_lambda = 0.1
        except Exception:
            best_lambda = 0.1

        _, best_residue = _compute_gcv_score(tissue_curve, u, s, vh, best_lambda)

        # Apply non-negativity
        if apply_non_negativity:
            best_residue = np.maximum(best_residue, 0.0)

        # Store results for this voxel
        residues[idx, :] = best_residue
        cbf[idx] = float(np.max(best_residue))
        cbv[idx] = float(np.trapezoid(best_residue, dx=dt))
        if cbf[idx] > EPSILON_CONCENTRATION:
            mtt[idx] = cbv[idx] / cbf[idx]
        lambda_optimal[idx] = float(best_lambda)

    # Reshape to original spatial dimensions
    if tissue_curves.ndim > 2:
        residues = residues.reshape(original_shape)
        cbf = cbf.reshape(spatial_shape)
        cbv = cbv.reshape(spatial_shape)
        mtt = mtt.reshape(spatial_shape)
        lambda_optimal = lambda_optimal.reshape(spatial_shape)

    return residues, cbf, cbv, mtt, lambda_optimal


# =============================================================================
# Arterial Dispersion Correction
# =============================================================================


@dataclass
class DispersionCorrectionResult:
    """Result from arterial dispersion correction.

    Dispersion correction accounts for the broadening and attenuation of
    the bolus as it travels from the AIF sampling site to the tissue of interest.
    This is modeled using a vascular transport function (VTF).

    Attributes:
        aif_corrected: The dispersion-corrected AIF, with same length as input AIF.
            This represents the estimated AIF at the tissue site.
        vtf: The vascular transport function used for correction.
        auc_original: AUC of the original AIF (dose preserved).
        auc_corrected: AUC of the corrected AIF (should match original).
        peak_original: Peak value of the original AIF.
        peak_corrected: Peak value of the corrected AIF (typically higher).
        fwhm_original: FWHM of the original AIF in frames.
        fwhm_corrected: FWHM of the corrected AIF (typically narrower).
        dispersion_factor: Ratio of corrected to original peak (typically 1.15-1.30).
    """

    aif_corrected: npt.NDArray[np.floating]
    vtf: npt.NDArray[np.floating]
    auc_original: float
    auc_corrected: float
    peak_original: float
    peak_corrected: float
    fwhm_original: float
    fwhm_corrected: float
    dispersion_factor: float


def _gamma_variate_vtf(
    t: npt.NDArray[np.floating],
    alpha: float,
    beta: float,
    delay: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Generate a gamma-variate vascular transport function (VTF).

    The VTF models the probability distribution of transit times through
    the arterial tree between the AIF sampling site and the tissue.

    Model: h(t) = ((t - delay) / beta)^alpha × exp(-(t - delay) / beta) / (beta × Gamma(alpha + 1))

    This is a normalized gamma distribution (integrates to 1) representing
    the probability density of transit times.

    Args:
        t: Time points in seconds (or frames if dt=1), shape (n_timepoints,).
        alpha: Shape parameter controlling the skewness of the VTF.
            Higher alpha = more symmetric distribution.
            Typical range: 0.5-2.0 for arterial dispersion.
        beta: Scale parameter in seconds (or frames), controlling the
            mean transit time and width.
            MTT_vtf = (alpha + 1) × beta
            Typical range: 0.5-2.0 seconds.
        delay: Onset delay in same units as t. Default 0.

    Returns:
        Normalized VTF with same shape as t. Integrates to approximately 1.
    """
    # Shift time by delay
    t_shifted = t - delay

    # Output array (zeros where t <= delay)
    vtf = np.zeros_like(t, dtype=np.float64)

    # Valid time points
    valid = t_shifted > 0
    t_valid = t_shifted[valid]

    # Gamma distribution: (t/beta)^alpha * exp(-t/beta) / (beta * Gamma(alpha+1))
    # This is the PDF of a gamma distribution with shape (alpha+1) and scale beta
    from scipy.special import gamma as gamma_func

    normalization = beta * gamma_func(alpha + 1)
    if normalization > EPSILON_CONCENTRATION:
        vtf[valid] = (
            np.power(t_valid / beta, alpha)
            * np.exp(-t_valid / beta)
            / normalization
        )

    return vtf


def _compute_fwhm_frames(curve: npt.NDArray[np.floating]) -> float:
    """Compute FWHM of a curve in frame units.

    Args:
        curve: 1D array representing the curve.

    Returns:
        FWHM in frames. Returns 0 if curve is flat or invalid.
    """
    if len(curve) < 3:
        return 0.0

    peak = float(np.max(curve))
    if peak <= EPSILON_CONCENTRATION:
        return 0.0

    half_max = peak / 2.0

    # Find indices where curve crosses half-max
    above_half = curve >= half_max

    if not np.any(above_half):
        return 0.0

    # Find first and last indices above half-max
    indices = np.where(above_half)[0]
    first_idx = int(indices[0])
    last_idx = int(indices[-1])

    # FWHM in frames
    return float(last_idx - first_idx + 1)


def _deconvolve_aif_with_vtf(
    aif: npt.NDArray[np.floating],
    vtf: npt.NDArray[np.floating],
    dt: float,
    threshold: float = 0.2,
) -> npt.NDArray[np.floating]:
    """Deconvolve the measured AIF with the VTF to recover the true local AIF.

    The measured AIF is the convolution of the true (local) AIF with the VTF:
        AIF_measured = AIF_true ⊗ VTF

    Deconvolving removes the dispersion effect:
        AIF_true = deconvolve(AIF_measured, VTF)

    Args:
        aif: Measured AIF concentration curve, shape (n_timepoints,).
        vtf: Vascular transport function, shape (n_timepoints,).
        dt: Time step in seconds.
        threshold: SVD truncation threshold for regularization.

    Returns:
        Dispersion-corrected (true local) AIF.
    """
    n = len(aif)

    # Build Toeplitz matrix from VTF
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            # Trapezoidal rule weighting
            if j == 0 or j == i:
                weight = 0.5
            else:
                weight = 1.0
            matrix[i, j] = vtf[i - j] * weight * dt

    # SVD of VTF convolution matrix
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

    # Deconvolve: AIF_true = V @ S_inv @ U^T @ AIF_measured
    aif_corrected = vh.T @ (s_inv * (u.T @ aif))

    # Apply non-negativity
    aif_corrected = np.maximum(aif_corrected, 0.0)

    return np.asarray(aif_corrected, dtype=np.float64)


def apply_dispersion_correction(
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    vtf_alpha: float = 1.0,
    vtf_beta: float = 1.5,
    vtf_delay: float = 0.0,
    svd_threshold: float = 0.2,
    preserve_auc: bool = True,
) -> DispersionCorrectionResult:
    """Apply vascular transport function (VTF) based arterial dispersion correction.

    Arterial dispersion occurs as the contrast bolus travels through the
    arterial tree from the AIF sampling site (typically large arteries like
    the MCA or carotid) to the tissue microvasculature. The bolus broadens
    and attenuates due to:

    1. Different path lengths through the vascular network
    2. Laminar flow velocity profiles
    3. Mixing at branch points

    This effect causes CBF underestimation, particularly in distal territories
    (15-30% underestimation in watershed regions).

    The correction models dispersion using a gamma-variate VTF and deconvolves
    the measured AIF to recover the estimated local AIF at the tissue site.

    Model:
        AIF_measured(t) = AIF_true(t) ⊗ VTF(t)
        AIF_true(t) = deconvolve(AIF_measured(t), VTF(t))

    where VTF is a gamma distribution modeling transit time distribution
    through the arterial tree.

    Args:
        aif: Measured arterial input function, shape (n_timepoints,).
            Can be in concentration (mM) or ΔR2* (s⁻¹) units.
        dt: Time step in seconds. Default 1.0 gives results in frame units.
        vtf_alpha: Shape parameter for the VTF gamma distribution.
            Higher values = more symmetric VTF.
            Typical range: 0.5-2.0. Default 1.0.
        vtf_beta: Scale parameter for the VTF in seconds (or frames if dt=1).
            Controls the mean arterial transit time: MTT_vtf = (alpha+1) × beta.
            Typical range: 0.5-2.0 seconds. Default 1.5.
        vtf_delay: Onset delay of the VTF in same units as dt.
            Default 0.0.
        svd_threshold: SVD truncation threshold for deconvolution.
            Higher = more regularization = smoother result.
            Default 0.2.
        preserve_auc: If True, scale the corrected AIF to preserve the total
            dose (AUC) of the original AIF. This ensures that CBV remains
            unaffected by the correction. Default True.

    Returns:
        DispersionCorrectionResult containing corrected AIF and metrics.

    Raises:
        ValueError: If aif is empty.
        ValueError: If vtf_alpha or vtf_beta are not positive.

    Notes:
        - The correction increases the AIF peak by 15-30% typically
        - FWHM is reduced (narrower, sharper bolus)
        - Total AUC is preserved to maintain accurate CBV
        - CBF increases in distal territories after correction
        - The effect is larger for regions far from the AIF sampling site
        - For proximal regions, correction may over-correct

    Impact on perfusion parameters:
        - CBF: Increases 15-30% in distal (watershed) territories
        - CBV: Unchanged if preserve_auc=True
        - MTT: Decreases proportionally (MTT = CBV/CBF)
        - Tmax: May decrease due to sharper bolus

    References:
        Calamante F, Gadian DG, Connelly A.
        Delay and dispersion effects in dynamic susceptibility contrast MRI:
        simulations using singular value decomposition.
        Magn Reson Med. 2000;44(3):466-473.

        Willats L, Calamante F.
        The 39 steps: evading error and deciphering the secrets for
        accurate dynamic susceptibility contrast MRI.
        NMR Biomed. 2013;26(8):913-931.
    """
    # Validate inputs
    if len(aif) == 0:
        raise ValueError("AIF array cannot be empty")

    if vtf_alpha <= 0:
        raise ValueError(f"vtf_alpha must be positive, got {vtf_alpha}")

    if vtf_beta <= 0:
        raise ValueError(f"vtf_beta must be positive, got {vtf_beta}")

    n = len(aif)

    # Create time vector
    t = np.arange(n, dtype=np.float64) * dt

    # Generate VTF
    vtf = _gamma_variate_vtf(t, vtf_alpha, vtf_beta, vtf_delay)

    # Normalize VTF to integrate to 1 (probability distribution)
    vtf_integral = trapezoid(vtf, dx=dt)
    if vtf_integral > EPSILON_CONCENTRATION:
        vtf = vtf / vtf_integral

    # Compute original AIF metrics
    auc_original = float(trapezoid(aif, dx=dt))
    peak_original = float(np.max(aif))
    fwhm_original = _compute_fwhm_frames(aif)

    # Deconvolve AIF with VTF
    aif_corrected = _deconvolve_aif_with_vtf(aif, vtf, dt, svd_threshold)

    # Preserve AUC (dose) if requested
    if preserve_auc:
        auc_corrected_raw = float(trapezoid(aif_corrected, dx=dt))
        if auc_corrected_raw > EPSILON_CONCENTRATION:
            scale_factor = auc_original / auc_corrected_raw
            aif_corrected = aif_corrected * scale_factor

    # Compute corrected AIF metrics
    auc_corrected = float(trapezoid(aif_corrected, dx=dt))
    peak_corrected = float(np.max(aif_corrected))
    fwhm_corrected = _compute_fwhm_frames(aif_corrected)

    # Dispersion factor (ratio of corrected to original peak)
    if peak_original > EPSILON_CONCENTRATION:
        dispersion_factor = peak_corrected / peak_original
    else:
        dispersion_factor = 1.0

    return DispersionCorrectionResult(
        aif_corrected=aif_corrected,
        vtf=vtf,
        auc_original=auc_original,
        auc_corrected=auc_corrected,
        peak_original=peak_original,
        peak_corrected=peak_corrected,
        fwhm_original=fwhm_original,
        fwhm_corrected=fwhm_corrected,
        dispersion_factor=dispersion_factor,
    )


def apply_dispersion_correction_batch(
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    vtf_alpha: float = 1.0,
    vtf_beta: float = 1.5,
    vtf_delay: float = 0.0,
    svd_threshold: float = 0.2,
    preserve_auc: bool = True,
) -> npt.NDArray[np.floating]:
    """Convenience function to apply dispersion correction and return only the corrected AIF.

    This is a simplified interface for batch processing pipelines that only
    need the corrected AIF without the full metrics.

    Args:
        aif: Measured arterial input function, shape (n_timepoints,).
        dt: Time step in seconds.
        vtf_alpha: VTF shape parameter. Default 1.0.
        vtf_beta: VTF scale parameter in seconds. Default 1.5.
        vtf_delay: VTF onset delay. Default 0.0.
        svd_threshold: SVD truncation threshold. Default 0.2.
        preserve_auc: Preserve total AUC. Default True.

    Returns:
        Dispersion-corrected AIF, shape (n_timepoints,).
    """
    result = apply_dispersion_correction(
        aif=aif,
        dt=dt,
        vtf_alpha=vtf_alpha,
        vtf_beta=vtf_beta,
        vtf_delay=vtf_delay,
        svd_threshold=svd_threshold,
        preserve_auc=preserve_auc,
    )
    return result.aif_corrected


# =============================================================================
# Perfusion Parameter Extraction
# =============================================================================


@dataclass
class PerfusionParameters:
    """Perfusion parameters extracted from a residue function.

    These parameters are derived from the residue (impulse response) function
    obtained via deconvolution of the tissue concentration curve with the AIF.

    Attributes:
        cbf: Cerebral blood flow = max(R(t)).
            Units depend on input: typically mL/100g/min for absolute quantification.
        cbv: Cerebral blood volume = ∫R(t)dt.
            Units: mL/100g for absolute quantification.
        mtt: Mean transit time = CBV / CBF (central volume theorem).
            Units: seconds.
        tmax: Time to maximum of residue function = argmax(R(t)) × dt.
            Units: seconds. Indicates delay between AIF and tissue response.
            Clinical significance: Tmax > 6s may indicate penumbra.
        ttp: Time to peak of residue function in frame units (before dt scaling).
            Integer index of the maximum in the residue function.
        auc: Area under the residue curve (same as CBV but kept for clarity).
        r_max: Maximum value of the residue function (same as CBF).
    """

    cbf: float
    cbv: float
    mtt: float
    tmax: float
    ttp: int
    auc: float
    r_max: float


def extract_perfusion_params(
    residue: npt.NDArray[np.floating],
    dt: float = 1.0,
) -> dict[str, float]:
    """Extract perfusion parameters from a residue function.

    Given a residue function R(t) from deconvolution, extracts the standard
    DSC perfusion parameters:

    - CBF = max(R(t))
        Cerebral blood flow, the maximum of the residue function.

    - CBV = ∫R(t)dt
        Cerebral blood volume, the integral of the residue function.

    - MTT = CBV / CBF
        Mean transit time via the central volume theorem.

    - Tmax = argmax(R(t)) × dt
        Time to maximum of the residue function (in seconds if dt provided).

    This function is useful when you have a residue function from any source
    (not necessarily from the DeconvolutionResult dataclass) and want to
    extract the standard perfusion parameters.

    Args:
        residue: Residue function R(t), shape (n_timepoints,).
            This should be the output from deconvolution (sSVD, cSVD, oSVD,
            Tikhonov, or any other method).
        dt: Time step in seconds. Default 1.0 gives Tmax in frame units.
            For proper temporal units, provide the actual TR (repetition time).

    Returns:
        Dictionary containing:
            - 'cbf': Cerebral blood flow = max(R(t))
            - 'cbv': Cerebral blood volume = ∫R(t)dt
            - 'mtt': Mean transit time = CBV/CBF (seconds)
            - 'tmax': Time to maximum = argmax(R(t)) × dt (seconds)
            - 'ttp': Time to peak in frame units (integer index)
            - 'auc': Area under curve (same as CBV)
            - 'r_max': Maximum residue value (same as CBF)

    Raises:
        ValueError: If residue array is empty.

    Notes:
        - CBF and CBV units depend on the input residue units:
            - If residue is from relative deconvolution: unitless or arbitrary
            - If residue includes r2* scaling: concentration units
            - For absolute CBF in mL/100g/min, additional scaling factors needed
        - MTT is always in seconds (or frames if dt=1.0)
        - Tmax > 6 seconds is often used as a clinical threshold for penumbra
        - For cSVD results, the delay is already accounted for in the residue

    Example:
        >>> import numpy as np
        >>> from pmriprep.utils.dsc_deconvolution import (
        ...     svd_deconvolve_standard,
        ...     extract_perfusion_params,
        ... )
        >>> # Generate example data
        >>> aif = np.exp(-np.arange(60) / 5.0) * 10
        >>> tissue = np.convolve(aif, np.exp(-np.arange(60) / 10.0))[:60]
        >>> # Deconvolve
        >>> result = svd_deconvolve_standard(tissue, aif, dt=1.5)
        >>> # Extract parameters (alternative to using result.cbf, etc.)
        >>> params = extract_perfusion_params(result.residue, dt=1.5)
        >>> print(f"CBF: {params['cbf']:.2f}, MTT: {params['mtt']:.2f}s")

    References:
        Østergaard L, Weisskoff RM, Chesler DA, Gyldensted C, Rosen BR.
        High resolution measurement of cerebral blood flow using
        intravascular tracer bolus passages. Part I: Mathematical approach
        and statistical analysis. Magn Reson Med. 1996;36(5):715-725.

        Central Volume Theorem:
        MTT = CBV / CBF (Meier P, Zierler KL. J Appl Physiol. 1954;6:731-744)
    """
    # Validate input
    if len(residue) == 0:
        raise ValueError("Residue array cannot be empty")

    # CBF = max(R(t))
    cbf = float(np.max(residue))

    # CBV = ∫R(t)dt using trapezoidal integration (Sourbron & Luypaert 2007)
    cbv = float(np.trapezoid(residue, dx=dt))

    # MTT = CBV / CBF (central volume theorem)
    # Guard against division by zero
    if cbf > EPSILON_CONCENTRATION:
        mtt = cbv / cbf
    else:
        mtt = 0.0

    # Tmax = argmax(R(t)) × dt
    # Time to peak in frame units (integer index)
    ttp = int(np.argmax(residue))
    # Time to maximum in seconds (or whatever units dt is in)
    tmax = float(ttp) * dt

    return {
        "cbf": cbf,
        "cbv": cbv,
        "mtt": mtt,
        "tmax": tmax,
        "ttp": ttp,
        "auc": cbv,  # AUC is the same as CBV
        "r_max": cbf,  # r_max is the same as CBF
    }


def extract_perfusion_params_batch(
    residues: npt.NDArray[np.floating],
    dt: float = 1.0,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
    npt.NDArray[np.floating],  # tmax
]:
    """Extract perfusion parameters from multiple residue functions.

    Batch version of extract_perfusion_params() for processing multiple
    voxels efficiently.

    Args:
        residues: Residue functions, shape (n_voxels, n_timepoints) or
            (x, y, z, n_timepoints) for 4D data.
        dt: Time step in seconds. Default 1.0.
        mask: Optional boolean mask of voxels to process. If None, process all.
            Shape must match spatial dimensions of residues.

    Returns:
        Tuple of (cbf, cbv, mtt, tmax) arrays with same spatial shape as input.
            - cbf: Cerebral blood flow = max(R(t))
            - cbv: Cerebral blood volume = ∫R(t)dt
            - mtt: Mean transit time = CBV/CBF
            - tmax: Time to maximum = argmax(R(t)) × dt
    """
    # Store original shape for reshaping output
    original_shape = residues.shape
    n_timepoints = original_shape[-1]

    # Reshape to 2D (n_voxels, n_timepoints)
    if residues.ndim > 2:
        spatial_shape = original_shape[:-1]
        n_voxels = int(np.prod(spatial_shape))
        residues_2d = residues.reshape(n_voxels, n_timepoints)
    else:
        spatial_shape = (original_shape[0],)
        n_voxels = original_shape[0]
        residues_2d = residues

    # Flatten mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
    else:
        mask_flat = np.ones(n_voxels, dtype=bool)

    # Initialize output arrays
    cbf = np.zeros(n_voxels, dtype=np.float64)
    cbv = np.zeros(n_voxels, dtype=np.float64)
    mtt = np.zeros(n_voxels, dtype=np.float64)
    tmax = np.zeros(n_voxels, dtype=np.float64)

    # Process masked voxels
    masked_indices = np.where(mask_flat)[0]
    if len(masked_indices) > 0:
        residues_masked = residues_2d[masked_indices, :]

        # CBF = max(R(t))
        cbf_masked = np.max(residues_masked, axis=1)

        # CBV = ∫R(t)dt using trapezoidal integration (Sourbron & Luypaert 2007)
        cbv_masked = np.trapezoid(residues_masked, dx=dt, axis=1)

        # MTT = CBV / CBF
        with np.errstate(divide="ignore", invalid="ignore"):
            mtt_masked = np.where(
                cbf_masked > EPSILON_CONCENTRATION,
                cbv_masked / cbf_masked,
                0.0,
            )

        # Tmax = argmax(R(t)) × dt
        ttp_masked = np.argmax(residues_masked, axis=1)
        tmax_masked = ttp_masked.astype(np.float64) * dt

        # Store results
        cbf[masked_indices] = cbf_masked
        cbv[masked_indices] = cbv_masked
        mtt[masked_indices] = mtt_masked
        tmax[masked_indices] = tmax_masked

    # Reshape to original spatial dimensions
    if residues.ndim > 2:
        cbf = cbf.reshape(spatial_shape)
        cbv = cbv.reshape(spatial_shape)
        mtt = mtt.reshape(spatial_shape)
        tmax = tmax.reshape(spatial_shape)

    return cbf, cbv, mtt, tmax
