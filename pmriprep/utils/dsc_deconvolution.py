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

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import linalg

from pmriprep.utils.dsc_constants import (
    DEFAULT_LAMBDA_RANGE,
    DEFAULT_OI_THRESHOLD,
    DEFAULT_SVD_THRESHOLD,
    EPSILON_CONCENTRATION,
    LCURVE_CURVATURE_FALLBACK,
    LCURVE_N_LAMBDAS,
)

logger = logging.getLogger(__name__)


@dataclass
class DeconvolutionResultBase:
    """Base result fields shared by all deconvolution methods.

    Attributes:
        residue: Residue function R(t).
        cbf: Cerebral blood flow, max(R(t)).
        cbv: Cerebral blood volume, integral of R(t).
        mtt: Mean transit time (seconds), CBV/CBF via central volume theorem.
    """

    residue: npt.NDArray[np.floating]
    cbf: float
    cbv: float
    mtt: float

    def get_tmax(self, dt: float = 1.0) -> float:
        """Get time to maximum of residue function.

        Args:
            dt: Time step in seconds (default 1.0 for frame units).

        Returns:
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(np.argmax(self.residue)) * dt


@dataclass
class DeconvolutionResult(DeconvolutionResultBase):
    """Result from standard SVD (sSVD) deconvolution.

    Attributes:
        threshold_used: SVD truncation threshold that was applied.
        n_singular_values: Number of singular values retained after truncation.
        condition_number: Condition number of the truncated matrix.
    """

    threshold_used: float
    n_singular_values: int
    condition_number: float


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
            Provide actual TR for timing parameters (MTT, Tmax) in seconds.
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

    Attributes:
        delay: Estimated delay between AIF and tissue in seconds.
            Positive = tissue delayed relative to AIF.
            Negative = tissue arrives before AIF.
        delay_frames: Delay in frame units (before multiplication by dt).
        threshold_used: SVD truncation threshold that was applied.
        n_singular_values: Number of singular values retained after truncation.
        condition_number: Condition number of the truncated matrix.
    """

    delay: float
    delay_frames: int
    threshold_used: float
    n_singular_values: int
    condition_number: float

    def get_tmax(self, dt: float = 1.0) -> float:
        """Get Tmax for cSVD (= delay).

        For cSVD, Tmax represents the time delay of the tissue residue
        function relative to the AIF (Calamante, Stroke 2010). This is
        the standard definition used in DEFUSE-3 and DAWN trials.

        Args:
            dt: Time step in seconds (default 1.0 for frame units).

        Returns:
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(self.delay_frames) * dt


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
    m = 2 * n - 1

    # Zero-pad AIF to length 2N-1 for full delay range
    aif_padded = np.zeros(m, dtype=np.float64)
    aif_padded[:n] = aif

    # Rectangular weighting (cyclic shifts are incompatible with trapezoidal)
    matrix: npt.NDArray[np.floating] = linalg.circulant(aif_padded) * dt

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
        if argmax(R) < N:  delay = argmax(R)          (positive delay)
        else:              delay = argmax(R) - (2N-1)  (negative / wrap-around)

    where N is the original curve length and 2N-1 is the padded length.
    Index 0 in the block-circulant residue corresponds to zero delay
    (Wu et al. 2003, Eq. 5).

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

    # Extract delay from position of maximum (Wu et al. 2003)
    # Zero-delay is at index 0; indices >= n wrap to negative delays.
    peak_position = int(np.argmax(residue))
    if peak_position < n:
        delay_frames = peak_position
    else:
        delay_frames = peak_position - m
    delay_seconds = float(delay_frames) * dt

    # Extract N causal points starting at peak position
    # (Østergaard 1996: CBV = integral over causal response only)
    causal_indices = (peak_position + np.arange(n)) % m
    causal_residue = residue[causal_indices]

    # Extract perfusion parameters from causal portion
    cbf = float(np.max(causal_residue))

    # CBV via trapezoidal integration of causal residue (Østergaard 1996)
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

        # Extract delays from peak positions (Wu et al. 2003; zero-delay at index 0)
        peak_positions = np.argmax(residues_masked, axis=1)
        delay_frames = np.where(
            peak_positions < n_timepoints,
            peak_positions,
            peak_positions - m,
        )
        delay_masked = delay_frames.astype(np.float64) * dt

        # Extract N causal points from each voxel's 2N-1 residue
        # (Østergaard 1996: CBV = integral over causal response only)
        vox_idx = np.arange(len(masked_indices))[:, None]
        time_idx = (peak_positions[:, None] + np.arange(n_timepoints)[None, :]) % m
        causal_residues = residues_masked[vox_idx, time_idx]  # (n_vox, N)

        # Extract parameters from causal portion
        cbf_masked = np.max(causal_residues, axis=1)

        # CBV via trapezoidal integration of causal residue (Østergaard 1996)
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

    Attributes:
        delay: Tracer arrival delay in seconds.
        delay_frames: Tracer arrival delay in frame units.
        oscillation_index: OI = (1/N) * sum|R(t+1) - R(t)| / max(R).
        threshold_used: The optimal SVD truncation threshold.
        n_singular_values: Number of singular values retained after truncation.
        condition_number: Condition number of the truncated matrix.
        oi_target: The target oscillation index threshold that was used.
        converged: Whether the search converged to find a valid threshold.
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
        """Get time to maximum of residue function (delay-based).

        Args:
            dt: Time step in seconds (default 1.0 for frame units).

        Returns:
            Tmax in seconds (or frames if dt=1.0).
        """
        return float(self.delay_frames) * dt


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
    # OI = sum(diff) / (N * max_residue)
    # Using len(residue) (not len(diff)) as the normalizer
    oi = float(np.sum(diff)) / (len(residue) * max_residue)

    return oi


def _deconvolve_with_threshold(
    tissue_padded: npt.NDArray[np.floating],
    n_original: int,
    threshold: float,
    u: npt.NDArray[np.floating],
    s: npt.NDArray[np.floating],
    vh: npt.NDArray[np.floating],
    apply_non_negativity: bool,
) -> tuple[npt.NDArray[np.floating], float, int, float, int]:
    """Helper: deconvolve with a given threshold using block-circulant SVD.

    Args:
        tissue_padded: Zero-padded tissue curve, length 2N-1.
        n_original: Original (unpadded) number of timepoints N.
        threshold: SVD truncation threshold (fraction of max singular value).
        u, s, vh: Pre-computed SVD of block-circulant matrix.
        apply_non_negativity: Whether to clip negative values.

    Returns:
        Tuple of (residue_2N1, oscillation_index, n_retained,
        condition_number, peak_position).
    """
    m = len(tissue_padded)

    # Truncate singular values below threshold
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

    # Apply non-negativity constraint
    if apply_non_negativity:
        residue = np.maximum(residue, 0.0)

    # Extract peak and N causal points for OI computation
    peak_position = int(np.argmax(residue))
    causal_indices = (peak_position + np.arange(n_original)) % m
    causal_residue = residue[causal_indices]

    # OI computed on causal subset only (Wu 2003)
    oi = compute_oscillation_index(causal_residue)

    return residue, oi, n_retained, condition_number, peak_position


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

    Uses a block-circulant matrix (Wu 2003) for delay-insensitive
    deconvolution with adaptive OI-based threshold selection.

    The oscillation index is defined as:
        OI = (1/N) * sum|R(t+1) - R(t)| / max(R)
    computed on the N causal points extracted from the 2N-1 circulant residue.

    A linear search from low to high threshold finds the first (lowest)
    threshold that achieves OI <= oi_target.

    Args:
        tissue_curve: Tissue concentration curve, shape (n_timepoints,).
        aif: Arterial input function, shape (n_timepoints,).
            Must have same length as tissue_curve.
        dt: Time step in seconds. Default 1.0 gives results in frame units.
        oi_target: Target oscillation index threshold. Default is 0.065.
        threshold_min: Minimum SVD threshold to consider. Default 0.01.
        threshold_max: Maximum SVD threshold to consider. Default 0.50.
        max_iterations: Number of evenly-spaced thresholds. Default 20.
        apply_non_negativity: If True, clip negative residue values to zero.

    Returns:
        OscillationDeconvolutionResult with 2N-1 residue, delay, and
        perfusion parameters derived from the N causal points.

    Raises:
        ValueError: If tissue_curve and aif have different lengths.
        ValueError: If arrays are empty.
        ValueError: If oi_target is not positive.

    References:
        Wu O, Ostergaard L, Weisskoff RM, et al. Tracer arrival
        timing-insensitive technique for estimating flow in MR
        perfusion-weighted imaging using singular value decomposition
        with a block-circulant deconvolution matrix. MRM 2003;50:164-174.
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
    m = 2 * n - 1

    if np.max(np.abs(aif)) < EPSILON_CONCENTRATION:
        raise ValueError(
            f"AIF is near-zero (max={np.max(np.abs(aif)):.2e}). "
            "Check AIF detection."
        )

    # Zero-pad tissue curve to 2N-1
    tissue_padded = np.zeros(m, dtype=np.float64)
    tissue_padded[:n] = tissue_curve

    # Build block-circulant matrix and compute SVD once
    matrix = build_block_circulant_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Linear search for optimal threshold (LOW→HIGH, first acceptable)
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
            tissue_padded, n, thresh, u, s, vh, apply_non_negativity,
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
                tissue_padded, n, threshold_max, u, s, vh,
                apply_non_negativity,
            )
        )
        best_threshold = threshold_max
        converged = False

    # Extract delay from peak position (Wu et al. 2003; zero-delay at index 0)
    if best_peak < n:
        delay_frames = best_peak
    else:
        delay_frames = best_peak - m
    delay = float(delay_frames) * dt

    # Extract N causal points for perfusion parameters
    causal_indices = (best_peak + np.arange(n)) % m
    causal_residue = best_residue[causal_indices]

    cbf = float(np.max(causal_residue))
    cbv = float(np.trapezoid(causal_residue, dx=dt))
    mtt = cbv / cbf if cbf > EPSILON_CONCENTRATION else 0.0

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
    max_iterations: int = 20,
    apply_non_negativity: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # residues (2N-1 per voxel)
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
    npt.NDArray[np.floating],  # oscillation_index
    npt.NDArray[np.floating],  # threshold_used
    npt.NDArray[np.floating],  # delay (seconds)
]:
    """Batch oSVD deconvolution using block-circulant matrix (Wu 2003).

    Each voxel gets an individually optimized threshold via OI-based
    linear search. Returns 2N-1 residues and delay maps.

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
        mask: Optional boolean mask of voxels to process.

    Returns:
        Tuple of (residues, cbf, cbv, mtt, oscillation_index, threshold_used,
        delay) arrays. Residues have time dimension 2N-1. All spatial arrays
        match the input spatial shape.
    """
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]
    m = 2 * n_timepoints - 1

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

    if np.max(np.abs(aif)) < EPSILON_CONCENTRATION:
        raise ValueError(
            f"AIF is near-zero (max={np.max(np.abs(aif)):.2e}). "
            "Check AIF detection."
        )

    # Build block-circulant matrix and compute SVD once
    matrix = build_block_circulant_matrix(aif, dt)
    u, s, vh = linalg.svd(matrix, full_matrices=False)

    # Initialize output arrays (residues are 2N-1 long)
    residues = np.zeros((n_voxels, m), dtype=np.float64)
    cbf = np.zeros(n_voxels, dtype=np.float64)
    cbv = np.zeros(n_voxels, dtype=np.float64)
    mtt = np.zeros(n_voxels, dtype=np.float64)
    oscillation_index = np.zeros(n_voxels, dtype=np.float64)
    threshold_used = np.zeros(n_voxels, dtype=np.float64)
    delay = np.zeros(n_voxels, dtype=np.float64)

    masked_indices = np.where(mask_flat)[0]
    thresholds = np.linspace(threshold_min, threshold_max, max_iterations)

    for idx in masked_indices:
        # Zero-pad tissue to 2N-1
        tissue_padded = np.zeros(m, dtype=np.float64)
        tissue_padded[:n_timepoints] = tissue_2d[idx, :]

        # Linear search: LOW→HIGH, stop at FIRST threshold where OI ≤ target
        best_residue = None
        best_oi = float("inf")
        best_threshold = threshold_max
        best_peak = 0

        for thresh in thresholds:
            residue, oi, n_ret, cond, peak = _deconvolve_with_threshold(
                tissue_padded, n_timepoints, thresh, u, s, vh,
                apply_non_negativity,
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
                tissue_padded, n_timepoints, threshold_max, u, s, vh,
                apply_non_negativity,
            )
            best_threshold = threshold_max

        # Extract delay (Wu et al. 2003; zero-delay at index 0)
        if best_peak < n_timepoints:
            delay_frames = best_peak
        else:
            delay_frames = best_peak - m
        causal_indices = (best_peak + np.arange(n_timepoints)) % m
        causal_residue = best_residue[causal_indices]

        residues[idx, :] = best_residue
        cbf[idx] = float(np.max(causal_residue))
        cbv[idx] = float(np.trapezoid(causal_residue, dx=dt))
        if cbf[idx] > EPSILON_CONCENTRATION:
            mtt[idx] = cbv[idx] / cbf[idx]
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
    """Result from Tikhonov regularised deconvolution.

    First-derivative penalty operator with L-curve corner detection
    (Hansen 1998). Replaces the legacy zeroth-order + GCV implementation
    that failed on noisy DSC data.

    Attributes:
        lambda_optimal: λ at the L-curve corner (or fallback default).
        residual_norm: ||A·r - C||² at the unclipped solution.
        solution_norm: ||L·r||² at the unclipped solution (where L is
            the first-derivative operator).
        lcurve_corner_found: True if the L-curve has a detectable corner
            (curvature ≥ ``LCURVE_CURVATURE_FALLBACK``); False if the
            curve was too flat / monotone and the caller fell back to a
            default λ.
    """

    lambda_optimal: float
    residual_norm: float
    solution_norm: float
    lcurve_corner_found: bool = False


def _build_first_derivative_matrix(n: int) -> npt.NDArray[np.float64]:
    """Build the (n-1, n) forward-difference operator L.

    L[i, i] = -1, L[i, i+1] = +1, used for first-order Tikhonov:
        min ||A·r - C||² + λ² ||L·r||²

    The penalty ||L·r||² is the sum of squared first differences,
    appropriate for residue functions that should vary smoothly.
    """
    L = np.zeros((n - 1, n), dtype=np.float64)
    idx = np.arange(n - 1)
    L[idx, idx] = -1.0
    L[idx, idx + 1] = 1.0
    return L


def _precompute_tikhonov_inverses(
    AtA: npt.NDArray[np.floating],
    LtL: npt.NDArray[np.floating],
    lambdas: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Precompute (A^T A + λ² L^T L)^{-1} for each λ in the sweep.

    Uses Cholesky-based ``np.linalg.solve`` against the identity, which is
    numerically stable: the system ``A^T A + λ² L^T L`` is symmetric
    positive definite for any λ > 0 even though L^T L is rank-deficient
    on its own. This avoids the ill-conditioning of the
    generalised-eigenvalue pencil (which has near-zero L^T L modes
    aligned with near-zero A^T A modes for DSC Toeplitz matrices).

    Returns an array of shape (n_lambdas, n, n).
    """
    n = AtA.shape[0]
    eye_n = np.eye(n)
    inverses = np.zeros((len(lambdas), n, n), dtype=np.float64)
    for k, lam in enumerate(lambdas):
        M = AtA + (lam ** 2) * LtL
        inverses[k] = np.linalg.solve(M, eye_n)
    return inverses


def _select_lambda_lcurve(
    residual_norms: npt.NDArray[np.floating],
    solution_norms: npt.NDArray[np.floating],
) -> tuple[int, bool]:
    """Triangle-method L-curve corner detection (Hansen 1998 + Castellanos 2002).

    Returns (best_index, corner_found). ``corner_found=False`` indicates
    the L-curve is approximately monotone — the caller should fall back
    to a conservative default λ.
    """
    log_rho = np.log10(np.maximum(residual_norms, 1e-30))
    log_eta = np.log10(np.maximum(solution_norms, 1e-30))
    n = len(log_rho)
    curvatures = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        P_i = np.array([log_rho[i], log_eta[i]])
        P_lt = np.array([log_rho[i - 1], log_eta[i - 1]])
        P_rt = np.array([log_rho[i + 1], log_eta[i + 1]])
        a = float(np.linalg.norm(P_lt - P_i))
        b = float(np.linalg.norm(P_i - P_rt))
        c = float(np.linalg.norm(P_lt - P_rt))
        s = 0.5 * (a + b + c)
        area2 = max(s * (s - a) * (s - b) * (s - c), 0.0)
        if a * b * c > 1e-30:
            curvatures[i] = 4.0 * np.sqrt(area2) / (a * b * c)
    best_idx = int(np.argmax(curvatures))
    corner_found = bool(curvatures[best_idx] >= LCURVE_CURVATURE_FALLBACK)
    return best_idx, corner_found


def tikhonov_deconvolve(
    tissue_curve: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    lambda_min: float = DEFAULT_LAMBDA_RANGE.min,
    lambda_max: float = DEFAULT_LAMBDA_RANGE.max,
    apply_non_negativity: bool = True,
) -> TikhonovDeconvolutionResult:
    """First-order Tikhonov regularised deconvolution with L-curve λ selection.

    Solves the regularised least squares problem with a first-derivative
    smoothness penalty:
        minimize ||A·r - C||² + λ²||L·r||²

    where ``A`` is the Toeplitz convolution matrix built from the AIF, ``r``
    is the residue function, and ``L`` is the (n-1, n) forward-difference
    operator that penalises oscillations rather than residue magnitude.
    First-order Tikhonov is more appropriate than zeroth-order for DSC
    residues (which physically should vary smoothly) — see Wirestam 2000,
    Calamante 2003.

    The optimal λ is selected via the L-curve corner method (Hansen 1998):
    sweep λ on a log grid, record (||A·r_λ - C||, ||L·r_λ||), find the
    point of maximum curvature in log-log space.

    Args:
        tissue_curve: Tissue concentration curve, shape (n_timepoints,).
        aif: Arterial input function, shape (n_timepoints,). Must match
            ``tissue_curve`` length.
        dt: Frame period in seconds. Default 1.0 gives frame-unit timings;
            provide the actual TR for MTT/Tmax in seconds.
        lambda_min, lambda_max: Bounds of the log-spaced λ sweep,
            **expressed as fractions of σ_max(A)** (the largest singular
            value of the AIF Toeplitz matrix). The natural scale for
            first-order Tikhonov is the operator norm; absolute λ values
            are AIF-dependent and not portable. Defaults from
            ``DEFAULT_LAMBDA_RANGE`` = (1e-3, 10.0).
        apply_non_negativity: If True, clip negative residue values to zero
            after solving.

    Returns:
        TikhonovDeconvolutionResult with the residue, perfusion parameters,
        chosen λ, and a flag indicating whether the L-curve had a detectable
        corner.

    References:
        Hansen PC. Rank-Deficient and Discrete Ill-Posed Problems. SIAM, 1998.
        Wirestam R et al. MRM 43:691-700 (2000).
        Calamante F et al. MRM 50:1237-1247 (2003).
        Castellanos JL et al. App Numer Math 43:359-373 (2002).
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
    tissue = tissue_curve.astype(np.float64, copy=False)

    A = build_toeplitz_matrix(aif, dt)
    L = _build_first_derivative_matrix(n)
    AtA = A.T @ A
    LtL = L.T @ L
    AtC = A.T @ tissue

    sigma_max = float(linalg.svdvals(A)[0])
    if sigma_max <= 0.0:
        raise ValueError(
            "AIF produces a degenerate Toeplitz matrix (σ_max = 0)."
        )

    # λ_min, λ_max are interpreted as fractions of σ_max(A). First-order
    # Tikhonov has no problem-independent λ scale; absolute values are
    # AIF-dependent and not portable across protocols. The 4-decade
    # default sweep keeps the system well-conditioned at the low end
    # (where AtA alone is rank-deficient for DSC Toeplitz operators).
    lambdas = np.logspace(
        np.log10(lambda_min), np.log10(lambda_max), LCURVE_N_LAMBDAS
    ) * sigma_max
    residual_norms = np.zeros(LCURVE_N_LAMBDAS, dtype=np.float64)
    solution_norms = np.zeros(LCURVE_N_LAMBDAS, dtype=np.float64)
    residues_sweep = np.zeros((LCURVE_N_LAMBDAS, n), dtype=np.float64)

    for k, lam in enumerate(lambdas):
        M = AtA + (lam ** 2) * LtL
        r = np.linalg.solve(M, AtC)
        residues_sweep[k] = r
        residual_norms[k] = float(np.linalg.norm(A @ r - tissue))
        solution_norms[k] = float(np.linalg.norm(L @ r))

    best_idx, corner_found = _select_lambda_lcurve(
        residual_norms, solution_norms
    )

    if corner_found:
        best_lambda = float(lambdas[best_idx])
        best_residue = residues_sweep[best_idx].copy()
        best_residual_norm_unsq = residual_norms[best_idx]
        best_solution_norm_unsq = solution_norms[best_idx]
    else:
        # Hansen 1998 default for cornerless / monotone L-curves
        best_lambda = 0.1 * sigma_max
        M = AtA + (best_lambda ** 2) * LtL
        best_residue = np.linalg.solve(M, AtC)
        best_residual_norm_unsq = float(np.linalg.norm(A @ best_residue - tissue))
        best_solution_norm_unsq = float(np.linalg.norm(L @ best_residue))

    residual_norm = float(best_residual_norm_unsq ** 2)
    solution_norm = float(best_solution_norm_unsq ** 2)

    if apply_non_negativity:
        best_residue = np.maximum(best_residue, 0.0)

    cbf = float(np.max(best_residue))
    cbv = float(np.trapezoid(best_residue, dx=dt))
    mtt = cbv / cbf if cbf > EPSILON_CONCENTRATION else 0.0

    return TikhonovDeconvolutionResult(
        residue=best_residue,
        cbf=cbf,
        cbv=cbv,
        mtt=mtt,
        lambda_optimal=float(best_lambda),
        residual_norm=residual_norm,
        solution_norm=solution_norm,
        lcurve_corner_found=bool(corner_found),
    )


def tikhonov_deconvolve_batch(
    tissue_curves: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float = 1.0,
    lambda_min: float = DEFAULT_LAMBDA_RANGE.min,
    lambda_max: float = DEFAULT_LAMBDA_RANGE.max,
    apply_non_negativity: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[
    npt.NDArray[np.floating],  # residues
    npt.NDArray[np.floating],  # cbf
    npt.NDArray[np.floating],  # cbv
    npt.NDArray[np.floating],  # mtt
    npt.NDArray[np.floating],  # lambda_optimal
]:
    """Batch first-order Tikhonov deconvolution with L-curve λ selection.

    Vectorised counterpart of :func:`tikhonov_deconvolve`. The Toeplitz
    matrix ``A`` and its first-derivative pencil ``(A^T A, L^T L)``
    eigendecomposition are computed once and shared across all voxels;
    each voxel runs its own L-curve corner search.

    Args:
        tissue_curves: Tissue concentration curves, shape
            (n_voxels, n_timepoints) or (x, y, z, n_timepoints) for 4D data.
        aif: Arterial input function, shape (n_timepoints,).
        dt: Time step in seconds.
        lambda_min: Minimum λ value (log-grid lower bound).
        lambda_max: Maximum λ value (log-grid upper bound).
        apply_non_negativity: Apply non-negativity constraint on residue.
        mask: Optional boolean mask of voxels to process. If None, process all.
            Shape must match spatial dimensions of tissue_curves.

    Returns:
        Tuple of (residues, cbf, cbv, mtt, lambda_optimal) arrays.
        All output arrays have the same spatial shape as input (excluding time dimension).
    """
    original_shape = tissue_curves.shape
    n_timepoints = original_shape[-1]

    if tissue_curves.ndim > 2:
        spatial_shape = original_shape[:-1]
        n_voxels = int(np.prod(spatial_shape))
        tissue_2d = tissue_curves.reshape(n_voxels, n_timepoints).astype(
            np.float64, copy=False
        )
    else:
        spatial_shape = (original_shape[0],)
        n_voxels = original_shape[0]
        tissue_2d = tissue_curves.astype(np.float64, copy=False)

    if mask is not None:
        mask_flat = mask.flatten()
    else:
        mask_flat = np.ones(n_voxels, dtype=bool)

    A = build_toeplitz_matrix(aif, dt)
    L = _build_first_derivative_matrix(n_timepoints)
    AtA = A.T @ A
    LtL = L.T @ L

    sigma_max = float(linalg.svdvals(A)[0])
    if sigma_max <= 0.0:
        raise ValueError(
            "AIF produces a degenerate Toeplitz matrix (σ_max = 0)."
        )
    fallback_lambda = 0.1 * sigma_max

    # λ bounds are fractions of σ_max(A); see scalar tikhonov_deconvolve
    # for the rationale. Scale once and reuse for the precomputed inverses.
    lambdas = np.logspace(
        np.log10(lambda_min), np.log10(lambda_max), LCURVE_N_LAMBDAS
    ) * sigma_max

    # Shared precompute: (A^T A + λ² L^T L)^{-1} for each λ. Each system is
    # SPD for any λ > 0, so np.linalg.solve is rock-solid even where
    # AtA alone is rank-deficient (typical for DSC Toeplitz matrices).
    inverses = _precompute_tikhonov_inverses(AtA, LtL, lambdas)
    M_fallback = AtA + (fallback_lambda ** 2) * LtL

    residues = np.zeros((n_voxels, n_timepoints), dtype=np.float64)
    cbf = np.zeros(n_voxels, dtype=np.float64)
    cbv = np.zeros(n_voxels, dtype=np.float64)
    mtt = np.zeros(n_voxels, dtype=np.float64)
    lambda_optimal = np.zeros(n_voxels, dtype=np.float64)

    masked_indices = np.where(mask_flat)[0]
    for idx in masked_indices:
        tissue = tissue_2d[idx]
        AtC = A.T @ tissue

        # R[k, :] = inverses[k] @ AtC — shape (LCURVE_N_LAMBDAS, n)
        R = inverses @ AtC

        AR = R @ A.T
        LR = R @ L.T
        residual_norms = np.linalg.norm(AR - tissue, axis=1)
        solution_norms = np.linalg.norm(LR, axis=1)

        best_idx, corner_found = _select_lambda_lcurve(
            residual_norms, solution_norms
        )

        if corner_found:
            best_lambda = float(lambdas[best_idx])
            best_residue = R[best_idx].copy()
        else:
            best_lambda = fallback_lambda
            best_residue = np.linalg.solve(M_fallback, AtC)

        if apply_non_negativity:
            best_residue = np.maximum(best_residue, 0.0)

        residues[idx, :] = best_residue
        cbf[idx] = float(np.max(best_residue))
        cbv[idx] = float(np.trapezoid(best_residue, dx=dt))
        if cbf[idx] > EPSILON_CONCENTRATION:
            mtt[idx] = cbv[idx] / cbf[idx]
        lambda_optimal[idx] = best_lambda

    if tissue_curves.ndim > 2:
        residues = residues.reshape(original_shape)  # type: ignore[assignment]
        cbf = cbf.reshape(spatial_shape)  # type: ignore[assignment]
        cbv = cbv.reshape(spatial_shape)  # type: ignore[assignment]
        mtt = mtt.reshape(spatial_shape)  # type: ignore[assignment]
        lambda_optimal = lambda_optimal.reshape(spatial_shape)  # type: ignore[assignment]

    return residues, cbf, cbv, mtt, lambda_optimal

