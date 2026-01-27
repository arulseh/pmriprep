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

from pmriprep.utils.dsc_constants import (
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

    where the convolution integral is discretized using the
    trapezoidal rule.

    Args:
        aif: Arterial input function concentration curve, shape (n_timepoints,).
        dt: Time step in seconds for integration.

    Returns:
        Toeplitz matrix of shape (n_timepoints, n_timepoints).
    """
    n = len(aif)
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            # Trapezoidal rule weighting
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
    n_retained = int(np.sum(s >= truncation_threshold))
    if n_retained == 0:
        n_retained = 1  # Keep at least one singular value

    # Create truncated inverse singular values
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] >= truncation_threshold:
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

    # CBV via trapezoidal integration of residue
    cbv = float(trapezoid(residue, dx=dt))

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
    n_retained = max(1, int(np.sum(s >= truncation_threshold)))

    # Create truncated inverse
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] >= truncation_threshold:
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

        # CBV via trapezoidal integration
        cbv_masked = trapezoid(residues_masked, dx=dt, axis=1)

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

    # Apply trapezoidal integration weighting
    # For circulant matrix, we apply uniform weighting scaled by dt
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
    n_retained = int(np.sum(s >= truncation_threshold))
    if n_retained == 0:
        n_retained = 1  # Keep at least one singular value

    # Create truncated inverse singular values
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] >= truncation_threshold:
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

    # CBV via trapezoidal integration of residue
    cbv = float(trapezoid(residue, dx=dt))

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
    n_retained = max(1, int(np.sum(s >= truncation_threshold)))

    # Create truncated inverse
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] >= truncation_threshold:
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

        # CBV via trapezoidal integration
        cbv_masked = trapezoid(residues_masked, dx=dt, axis=1)

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

    # Mean absolute difference normalized by maximum
    oi = float(np.mean(diff)) / max_residue

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
    n_retained = int(np.sum(s >= truncation_threshold))
    if n_retained == 0:
        n_retained = 1  # Keep at least one singular value

    # Create truncated inverse singular values
    s_inv = np.zeros_like(s)
    for i in range(n_retained):
        if s[i] >= truncation_threshold:
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

    A binary search is used to find the optimal threshold that achieves
    OI ≤ oi_target while using the minimum regularization necessary.

    Strategy:
        - Lower threshold = more singular values retained = more detail but more noise
        - Higher threshold = fewer singular values = smoother but may lose detail
        - Binary search finds the smallest threshold where OI ≤ target

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
        max_iterations: Maximum iterations for binary search. Default 20.
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

    # Binary search for optimal threshold
    # We want the LOWEST threshold that achieves OI ≤ oi_target
    # (lowest threshold = least regularization = most detail preserved)
    lo = threshold_min
    hi = threshold_max
    converged = False

    # First, check if we can achieve target at all
    residue_hi, oi_hi, n_ret_hi, cond_hi = _deconvolve_with_threshold(
        tissue_curve, aif, dt, hi, matrix, u, s, vh, apply_non_negativity
    )

    if oi_hi > oi_target:
        # Even maximum threshold doesn't achieve target OI
        # Return result at maximum threshold
        cbf = float(np.max(residue_hi))
        cbv = float(trapezoid(residue_hi, dx=dt))
        mtt = cbv / cbf if cbf > EPSILON_CONCENTRATION else 0.0

        return OscillationDeconvolutionResult(
            residue=residue_hi,
            cbf=cbf,
            cbv=cbv,
            mtt=mtt,
            oscillation_index=oi_hi,
            threshold_used=hi,
            n_singular_values=n_ret_hi,
            condition_number=cond_hi,
            oi_target=oi_target,
            converged=False,
        )

    # Check minimum threshold
    residue_lo, oi_lo, n_ret_lo, cond_lo = _deconvolve_with_threshold(
        tissue_curve, aif, dt, lo, matrix, u, s, vh, apply_non_negativity
    )

    if oi_lo <= oi_target:
        # Even minimum threshold achieves target OI
        # Return result at minimum threshold (maximum detail)
        cbf = float(np.max(residue_lo))
        cbv = float(trapezoid(residue_lo, dx=dt))
        mtt = cbv / cbf if cbf > EPSILON_CONCENTRATION else 0.0

        return OscillationDeconvolutionResult(
            residue=residue_lo,
            cbf=cbf,
            cbv=cbv,
            mtt=mtt,
            oscillation_index=oi_lo,
            threshold_used=lo,
            n_singular_values=n_ret_lo,
            condition_number=cond_lo,
            oi_target=oi_target,
            converged=True,
        )

    # Binary search for optimal threshold
    # Invariant: oi_lo > oi_target, oi_hi <= oi_target
    best_residue = residue_hi
    best_oi = oi_hi
    best_threshold = hi
    best_n_ret = n_ret_hi
    best_cond = cond_hi

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0

        residue_mid, oi_mid, n_ret_mid, cond_mid = _deconvolve_with_threshold(
            tissue_curve, aif, dt, mid, matrix, u, s, vh, apply_non_negativity
        )

        if oi_mid <= oi_target:
            # This threshold works, try lower (less regularization)
            hi = mid
            best_residue = residue_mid
            best_oi = oi_mid
            best_threshold = mid
            best_n_ret = n_ret_mid
            best_cond = cond_mid
        else:
            # Need more regularization
            lo = mid

        # Check for convergence
        if hi - lo < 0.001:
            converged = True
            break
    else:
        # Loop completed without break - still converged to best found
        converged = True

    # Extract perfusion parameters from best result
    cbf = float(np.max(best_residue))
    cbv = float(trapezoid(best_residue, dx=dt))
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
        max_iterations: Maximum binary search iterations per voxel. Default 20.
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
    for idx in masked_indices:
        tissue_curve = tissue_2d[idx, :]

        # Binary search for this voxel
        lo = threshold_min
        hi = threshold_max

        # Check bounds
        residue_hi, oi_hi, n_ret_hi, cond_hi = _deconvolve_with_threshold(
            tissue_curve, aif, dt, hi, matrix, u, s, vh, apply_non_negativity
        )

        if oi_hi > oi_target:
            # Even max threshold doesn't achieve target
            best_residue = residue_hi
            best_oi = oi_hi
            best_threshold = hi
        else:
            residue_lo, oi_lo, n_ret_lo, cond_lo = _deconvolve_with_threshold(
                tissue_curve, aif, dt, lo, matrix, u, s, vh, apply_non_negativity
            )

            if oi_lo <= oi_target:
                # Min threshold achieves target
                best_residue = residue_lo
                best_oi = oi_lo
                best_threshold = lo
            else:
                # Binary search
                best_residue = residue_hi
                best_oi = oi_hi
                best_threshold = hi

                for _ in range(max_iterations):
                    mid = (lo + hi) / 2.0
                    residue_mid, oi_mid, _, _ = _deconvolve_with_threshold(
                        tissue_curve, aif, dt, mid, matrix, u, s, vh, apply_non_negativity
                    )

                    if oi_mid <= oi_target:
                        hi = mid
                        best_residue = residue_mid
                        best_oi = oi_mid
                        best_threshold = mid
                    else:
                        lo = mid

                    if hi - lo < 0.001:
                        break

        # Store results for this voxel
        residues[idx, :] = best_residue
        cbf[idx] = float(np.max(best_residue))
        cbv[idx] = float(trapezoid(best_residue, dx=dt))
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
