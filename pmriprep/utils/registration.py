"""Registration metric calculations for PMRIPrep.

This module provides functions for calculating registration quality metrics
including DICE coefficient, correlation, normalized mutual information (NMI),
and AIF preservation metrics.  Also provides shared registration utilities
(header alignment, DICE-based path selection) used by both DSC and DCE.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Registration path selection: path B must exceed path A by this margin
DICE_SELECTION_MARGIN: float = 0.01

# Percentile threshold for automatic mask creation in DICE selection
MASK_THRESHOLD_PERCENTILE: int = 5

# Epsilon to prevent division by zero in DICE selection
DICE_EPSILON: float = 1e-9


def compare_transforms(
    mat1: NDArray[np.floating],
    mat2: NDArray[np.floating],
    radius: float = 70.0,
) -> float:
    """Compute maximum displacement between two FSL affine transforms.

    Evaluates both transforms at six test points (cube face midpoints at
    distance ``radius`` from the origin) and returns the maximum Euclidean
    distance between the transformed positions.

    Adapted from niworkflows ``compare_xforms()``.

    Parameters
    ----------
    mat1 : np.ndarray
        First 4x4 affine matrix (FSL format).
    mat2 : np.ndarray
        Second 4x4 affine matrix (FSL format).
    radius : float
        Distance (mm) of test points from the origin.
        Default 70mm approximates the brain radius.

    Returns
    -------
    float
        Maximum displacement in mm across all test points.

    Raises
    ------
    ValueError
        If matrices are not 4x4.
    """
    if mat1.shape != (4, 4) or mat2.shape != (4, 4):
        raise ValueError(
            f"Affine matrices must be 4x4, got {mat1.shape} and {mat2.shape}"
        )

    # Cube face midpoints at distance `radius` from origin
    points = np.array([
        [radius, 0.0, 0.0],
        [-radius, 0.0, 0.0],
        [0.0, radius, 0.0],
        [0.0, -radius, 0.0],
        [0.0, 0.0, radius],
        [0.0, 0.0, -radius],
    ])

    # Apply both transforms to all test points
    # Homogeneous coordinates: append ones column
    ones = np.ones((points.shape[0], 1))
    pts_h = np.hstack([points, ones])  # (6, 4)

    # Transform: (4x4) @ (4xN) → (4xN)
    transformed1 = (mat1 @ pts_h.T).T[:, :3]  # (6, 3)
    transformed2 = (mat2 @ pts_h.T).T[:, :3]  # (6, 3)

    # Euclidean distance at each test point
    displacements = np.sqrt(np.sum((transformed1 - transformed2) ** 2, axis=1))

    return float(np.max(displacements))


def compare_transforms_from_files(
    mat1_path: Union[str, Path],
    mat2_path: Union[str, Path],
    radius: float = 70.0,
) -> float:
    """Compute maximum displacement between two FSL affine matrix files.

    Parameters
    ----------
    mat1_path : Path or str
        Path to first FSL .mat file.
    mat2_path : Path or str
        Path to second FSL .mat file.
    radius : float
        Distance (mm) of test points from origin (default: 70mm).

    Returns
    -------
    float
        Maximum displacement in mm.

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    """
    mat1_path = Path(mat1_path)
    mat2_path = Path(mat2_path)

    if not mat1_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {mat1_path}")
    if not mat2_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {mat2_path}")

    mat1 = np.loadtxt(mat1_path)
    mat2 = np.loadtxt(mat2_path)

    return compare_transforms(mat1, mat2, radius=radius)


def calculate_dice(
    mask1: NDArray[np.floating],
    mask2: NDArray[np.floating],
    threshold: float = 0.5,
) -> float:
    """Calculate DICE coefficient for two binary masks.

    ``DICE = 2 * |A ∩ B| / (|A| + |B|)``

    Parameters
    ----------
    mask1 : np.ndarray
        First mask array (will be binarized).
    mask2 : np.ndarray
        Second mask array (will be binarized).
    threshold : float
        Threshold for binarization of non-binary masks.

    Returns
    -------
    float
        DICE coefficient in range [0, 1]. Returns 0.0 if both masks are empty.

    Raises
    ------
    ValueError
        If masks have different shapes.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(
            f"Mask shapes must match: {mask1.shape} vs {mask2.shape}"
        )

    # Binarize masks
    bin1 = mask1 > threshold
    bin2 = mask2 > threshold

    # Calculate intersection and sums
    intersection: int = int(np.sum(bin1 & bin2))
    sum_masks: int = int(np.sum(bin1)) + int(np.sum(bin2))

    # Handle empty masks
    if sum_masks == 0:
        return 0.0

    dice = 2.0 * intersection / sum_masks
    return dice


def calculate_dice_from_files(
    mask1_path: Union[str, Path],
    mask2_path: Union[str, Path],
    threshold: float = 0.5,
) -> float:
    """Calculate DICE coefficient from two NIfTI mask files.

    Parameters
    ----------
    mask1_path : Path or str
        Path to first mask NIfTI file.
    mask2_path : Path or str
        Path to second mask NIfTI file.
    threshold : float
        Threshold for binarization.

    Returns
    -------
    float
        DICE coefficient in range [0, 1].

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    ValueError
        If masks have different shapes.
    """
    import nibabel as nib

    mask1_path = Path(mask1_path)
    mask2_path = Path(mask2_path)

    if not mask1_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask1_path}")
    if not mask2_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask2_path}")

    img1 = nib.load(mask1_path)
    img2 = nib.load(mask2_path)

    data1: NDArray[np.floating] = np.asarray(
        img1.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    data2: NDArray[np.floating] = np.asarray(
        img2.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )

    return calculate_dice(data1, data2, threshold=threshold)


def calculate_correlation(
    image1: NDArray[np.floating],
    image2: NDArray[np.floating],
    mask: Optional[NDArray[np.floating]] = None,
    threshold: float = 0.5,
) -> float:
    """Calculate Pearson correlation coefficient between two images.

    Optionally restricted to voxels within a mask.

    Parameters
    ----------
    image1 : np.ndarray
        First image array.
    image2 : np.ndarray
        Second image array (must match shape of image1).
    mask : np.ndarray, optional
        Optional mask array. If provided, correlation is computed
        only for voxels where mask > threshold.
    threshold : float
        Threshold for mask binarization.

    Returns
    -------
    float
        Pearson correlation coefficient in range [-1, 1].
        Returns 0.0 if insufficient valid voxels.

    Raises
    ------
    ValueError
        If images have different shapes.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image shapes must match: {image1.shape} vs {image2.shape}"
        )

    # Flatten images
    flat1: NDArray[np.floating] = image1.ravel()
    flat2: NDArray[np.floating] = image2.ravel()

    # Apply mask if provided
    if mask is not None:
        if mask.shape != image1.shape:
            raise ValueError(
                f"Mask shape must match images: {mask.shape} vs {image1.shape}"
            )
        mask_flat = mask.ravel() > threshold
        flat1 = flat1[mask_flat]
        flat2 = flat2[mask_flat]

    # Check we have enough valid voxels
    if len(flat1) < 2:
        return 0.0

    # Remove NaN/Inf values
    valid = np.isfinite(flat1) & np.isfinite(flat2)
    flat1 = flat1[valid]
    flat2 = flat2[valid]

    if len(flat1) < 2:
        return 0.0

    # Calculate correlation using numpy
    # corrcoef returns 2x2 matrix; correlation is at [0,1]
    corr_matrix = np.corrcoef(flat1, flat2)
    correlation = float(corr_matrix[0, 1])

    # Handle NaN result (can happen with constant arrays)
    if not np.isfinite(correlation):
        return 0.0

    return correlation


def calculate_correlation_from_files(
    image1_path: Union[str, Path],
    image2_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5,
) -> float:
    """Calculate correlation coefficient from two NIfTI image files.

    Parameters
    ----------
    image1_path : Path or str
        Path to first image NIfTI file.
    image2_path : Path or str
        Path to second image NIfTI file.
    mask_path : Path or str, optional
        Optional path to mask NIfTI file.
    threshold : float
        Threshold for mask binarization.

    Returns
    -------
    float
        Pearson correlation coefficient in range [-1, 1].

    Raises
    ------
    FileNotFoundError
        If any required file does not exist.
    ValueError
        If images have different shapes.
    """
    import nibabel as nib

    image1_path = Path(image1_path)
    image2_path = Path(image2_path)

    if not image1_path.exists():
        raise FileNotFoundError(f"Image file not found: {image1_path}")
    if not image2_path.exists():
        raise FileNotFoundError(f"Image file not found: {image2_path}")

    img1 = nib.load(image1_path)
    img2 = nib.load(image2_path)

    data1: NDArray[np.floating] = np.asarray(
        img1.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    data2: NDArray[np.floating] = np.asarray(
        img2.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )

    mask_data: Optional[NDArray[np.floating]] = None
    if mask_path is not None:
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask_img = nib.load(mask_path)
        mask_data = np.asarray(
            mask_img.get_fdata(),  # type: ignore[attr-defined]
            dtype=np.float64,
        )

    return calculate_correlation(data1, data2, mask=mask_data, threshold=threshold)


def calculate_nmi(
    image1: NDArray[np.floating],
    image2: NDArray[np.floating],
    mask: Optional[NDArray[np.floating]] = None,
    threshold: float = 0.5,
    n_bins: int = 64,
) -> float:
    """Calculate normalized mutual information between two images.

    NMI = 2 * I(A,B) / (H(A) + H(B))

    where I(A,B) is mutual information and H(A), H(B) are marginal entropies.

    Parameters
    ----------
    image1 : np.ndarray
        First image array.
    image2 : np.ndarray
        Second image array (must match shape of image1).
    mask : np.ndarray, optional
        Optional mask array. If provided, NMI is computed
        only for voxels where mask > threshold.
    threshold : float
        Threshold for mask binarization.
    n_bins : int
        Number of histogram bins for entropy calculation.

    Returns
    -------
    float
        Normalized mutual information in range [0, 2].
        0 indicates no mutual information, 2 indicates perfect correlation.
        Returns 0.0 if insufficient valid voxels.

    Raises
    ------
    ValueError
        If images have different shapes.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image shapes must match: {image1.shape} vs {image2.shape}"
        )

    # Flatten images
    flat1: NDArray[np.float64] = image1.ravel().astype(np.float64)
    flat2: NDArray[np.float64] = image2.ravel().astype(np.float64)

    # Apply mask if provided
    if mask is not None:
        if mask.shape != image1.shape:
            raise ValueError(
                f"Mask shape must match images: {mask.shape} vs {image1.shape}"
            )
        mask_flat = mask.ravel() > threshold
        flat1 = flat1[mask_flat]
        flat2 = flat2[mask_flat]

    # Remove NaN/Inf values
    valid = np.isfinite(flat1) & np.isfinite(flat2)
    flat1 = flat1[valid]
    flat2 = flat2[valid]

    # Check we have enough valid voxels
    if len(flat1) < n_bins:
        return 0.0

    # Compute 2D histogram (joint distribution)
    # Use data range for binning
    range1 = (float(np.min(flat1)), float(np.max(flat1)))
    range2 = (float(np.min(flat2)), float(np.max(flat2)))

    # Handle constant arrays
    if range1[0] == range1[1] or range2[0] == range2[1]:
        return 0.0

    hist_2d, _, _ = np.histogram2d(
        flat1, flat2, bins=n_bins, range=[range1, range2]
    )

    # Normalize to get joint probability
    joint_prob = hist_2d / float(np.sum(hist_2d))

    # Compute marginal probabilities
    p1 = np.sum(joint_prob, axis=1)
    p2 = np.sum(joint_prob, axis=0)

    # Compute entropies (with epsilon to avoid log(0))
    epsilon = 1e-10

    # Marginal entropy H(A)
    p1_nonzero = p1[p1 > epsilon]
    h1 = -float(np.sum(p1_nonzero * np.log(p1_nonzero)))

    # Marginal entropy H(B)
    p2_nonzero = p2[p2 > epsilon]
    h2 = -float(np.sum(p2_nonzero * np.log(p2_nonzero)))

    # Joint entropy H(A,B)
    joint_nonzero = joint_prob[joint_prob > epsilon]
    h_joint = -float(np.sum(joint_nonzero * np.log(joint_nonzero)))

    # Mutual information I(A,B) = H(A) + H(B) - H(A,B)
    mi = h1 + h2 - h_joint

    # Normalized mutual information
    sum_h = h1 + h2
    if sum_h < epsilon:
        return 0.0

    nmi = 2.0 * mi / sum_h

    # Clamp to valid range [0, 2]
    return float(max(0.0, min(2.0, nmi)))


def calculate_nmi_from_files(
    image1_path: Union[str, Path],
    image2_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5,
    n_bins: int = 64,
) -> float:
    """Calculate NMI from two NIfTI image files.

    Parameters
    ----------
    image1_path : Path or str
        Path to first image NIfTI file.
    image2_path : Path or str
        Path to second image NIfTI file.
    mask_path : Path or str, optional
        Optional path to mask NIfTI file.
    threshold : float
        Threshold for mask binarization.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Normalized mutual information in range [0, 2].

    Raises
    ------
    FileNotFoundError
        If any required file does not exist.
    ValueError
        If images have different shapes.
    """
    import nibabel as nib

    image1_path = Path(image1_path)
    image2_path = Path(image2_path)

    if not image1_path.exists():
        raise FileNotFoundError(f"Image file not found: {image1_path}")
    if not image2_path.exists():
        raise FileNotFoundError(f"Image file not found: {image2_path}")

    img1 = nib.load(image1_path)
    img2 = nib.load(image2_path)

    data1: NDArray[np.floating] = np.asarray(
        img1.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    data2: NDArray[np.floating] = np.asarray(
        img2.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )

    mask_data: Optional[NDArray[np.floating]] = None
    if mask_path is not None:
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask_img = nib.load(mask_path)
        mask_data = np.asarray(
            mask_img.get_fdata(),  # type: ignore[attr-defined]
            dtype=np.float64,
        )

    return calculate_nmi(data1, data2, mask=mask_data, threshold=threshold, n_bins=n_bins)


def compute_aif_preservation(
    aif_before: NDArray[np.floating],
    aif_after: NDArray[np.floating],
) -> float:
    """Compute AIF preservation ratio after registration/transformation.

    This metric quantifies how well the arterial input function (AIF)
    peak is preserved after spatial transformation. A ratio close to 1.0
    indicates good preservation.

    Parameters
    ----------
    aif_before : np.ndarray
        AIF curve before transformation (1D array, time series).
    aif_after : np.ndarray
        AIF curve after transformation (1D array, same length).

    Returns
    -------
    float
        Peak preservation ratio: peak_after / peak_before.
        Returns 1.0 if peaks are identical, <1.0 if peak is reduced,
        >1.0 if peak is increased. Returns 0.0 if before peak is zero.

    Raises
    ------
    ValueError
        If arrays have different lengths or are not 1D.
    """
    if aif_before.ndim != 1 or aif_after.ndim != 1:
        raise ValueError("AIF arrays must be 1D time series")

    if len(aif_before) != len(aif_after):
        raise ValueError(
            f"AIF lengths must match: {len(aif_before)} vs {len(aif_after)}"
        )

    # Get peak values (for DSC, peak is the maximum concentration)
    peak_before = float(np.max(aif_before))
    peak_after = float(np.max(aif_after))

    # Handle zero peak
    if peak_before == 0.0:
        return 0.0

    ratio = peak_after / peak_before
    return ratio


def compute_aif_preservation_with_auc(
    aif_before: NDArray[np.floating],
    aif_after: NDArray[np.floating],
    dt: float = 1.0,
) -> dict:
    """Compute comprehensive AIF preservation metrics.

    Computes both peak ratio and AUC (area under curve) ratio to
    provide a complete picture of AIF preservation after transformation.

    Parameters
    ----------
    aif_before : np.ndarray
        AIF curve before transformation (1D array).
    aif_after : np.ndarray
        AIF curve after transformation (1D array).
    dt : float
        Time step between samples (for AUC calculation).

    Returns
    -------
    dict
        Dictionary with:
            - peak_ratio: peak_after / peak_before
            - auc_ratio: auc_after / auc_before
            - peak_time_shift: Difference in peak time indices
            - quality: 'good' if both ratios in [0.9, 1.1], else 'degraded'

    Raises
    ------
    ValueError
        If arrays have different lengths or are not 1D.
    """
    if aif_before.ndim != 1 or aif_after.ndim != 1:
        raise ValueError("AIF arrays must be 1D time series")

    if len(aif_before) != len(aif_after):
        raise ValueError(
            f"AIF lengths must match: {len(aif_before)} vs {len(aif_after)}"
        )

    # Peak values and locations
    peak_before = float(np.max(aif_before))
    peak_after = float(np.max(aif_after))
    peak_idx_before = int(np.argmax(aif_before))
    peak_idx_after = int(np.argmax(aif_after))

    # AUC using trapezoidal rule
    # Use trapezoid (numpy 2.0+) or trapz (numpy <2.0)
    try:
        auc_before = float(np.trapezoid(aif_before, dx=dt))  # type: ignore[attr-defined]
        auc_after = float(np.trapezoid(aif_after, dx=dt))  # type: ignore[attr-defined]
    except AttributeError:
        auc_before = float(np.trapz(aif_before, dx=dt))  # type: ignore[attr-defined]
        auc_after = float(np.trapz(aif_after, dx=dt))  # type: ignore[attr-defined]

    # Calculate ratios
    if peak_before > 0:
        peak_ratio = peak_after / peak_before
    else:
        peak_ratio = 0.0

    if auc_before > 0:
        auc_ratio = auc_after / auc_before
    else:
        auc_ratio = 0.0

    # Peak time shift
    peak_time_shift = peak_idx_after - peak_idx_before

    # Quality assessment
    good_peak = 0.9 <= peak_ratio <= 1.1
    good_auc = 0.9 <= auc_ratio <= 1.1
    quality = "good" if (good_peak and good_auc) else "degraded"

    return {
        "peak_ratio": peak_ratio,
        "auc_ratio": auc_ratio,
        "peak_time_shift": peak_time_shift,
        "quality": quality,
    }


def _fsl_vox2mm(img: Any) -> NDArray[np.floating]:
    """Compute FSL's internal voxel-to-mm scaling matrix.

    FSL uses a specific convention: if the sform determinant is positive
    (neurological convention), the x-axis is flipped and an offset is
    added so that the origin stays at the first voxel corner.

    Parameters
    ----------
    img : Any
        Nibabel image object (Nifti1Image or similar).

    Returns
    -------
    np.ndarray
        4x4 scaling matrix in FSL mm coordinates.
    """
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    shape = np.array(img.shape[:3], dtype=np.float64)

    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = zooms[0]
    mat[1, 1] = zooms[1]
    mat[2, 2] = zooms[2]

    # If sform determinant is positive, FSL flips x-axis
    if np.linalg.det(img.affine[:3, :3]) > 0:
        mat[0, 0] = -zooms[0]
        mat[0, 3] = (shape[0] - 1) * zooms[0]

    return mat


def compute_header_init_matrix(
    moving_file: Union[str, Path],
    reference_file: Union[str, Path],
    output_dir: Union[str, Path],
) -> str:
    """Compute FLIRT-format matrix encoding header-based alignment.

    Derives the affine mapping from the moving image's voxel space to
    the reference image's voxel space using NIfTI sform/qform headers,
    expressed in FSL's internal coordinate convention.  The resulting
    ``.mat`` file can be passed as ``-init`` to FLIRT so that the
    coarse search starts near the header-aligned solution.

    The formula is:

        M_flirt = S_ref @ inv(W_ref) @ W_mov @ inv(S_mov)

    where ``S`` is FSL's voxel-to-mm scaling matrix and ``W`` is the
    NIfTI sform affine.

    Parameters
    ----------
    moving_file : Path or str
        Path to the moving image (e.g. T1w).
    reference_file : Path or str
        Path to the reference image (e.g. DSC mean).
    output_dir : Path or str
        Directory for the output ``.mat`` file.

    Returns
    -------
    str
        Path to the FLIRT-format ``.mat`` file.
    """
    import nibabel as nib

    mov_img = nib.load(str(moving_file))
    ref_img = nib.load(str(reference_file))

    S_mov = _fsl_vox2mm(mov_img)
    S_ref = _fsl_vox2mm(ref_img)
    W_mov = np.asarray(mov_img.affine, dtype=np.float64)  # type: ignore[attr-defined]
    W_ref = np.asarray(ref_img.affine, dtype=np.float64)  # type: ignore[attr-defined]

    # M_flirt = S_ref @ inv(W_ref) @ W_mov @ inv(S_mov)
    init_mat = S_ref @ np.linalg.inv(W_ref) @ W_mov @ np.linalg.inv(S_mov)

    out_path = str(Path(output_dir) / "header_init.mat")
    np.savetxt(out_path, init_mat, fmt="%.10f")

    return out_path


def write_ras_transform_to_fsl(
    ras_matrix: NDArray[np.floating],
    moving_file: Union[str, Path],
    reference_file: Union[str, Path],
    output_file: Union[str, Path],
) -> str:
    """Serialize a world/RAS affine as an FSL matrix for specific image grids."""
    from nitransforms.linear import Affine  # type: ignore[import-untyped]

    transform = Affine(np.asarray(ras_matrix), reference=str(reference_file))
    transform.to_filename(str(output_file), fmt="fsl", moving=str(moving_file))
    return str(output_file)


def reframe_fsl_transform(
    matrix_file: Union[str, Path],
    matrix_moving_file: Union[str, Path],
    output_moving_file: Union[str, Path],
    reference_file: Union[str, Path],
    output_file: Union[str, Path],
) -> str:
    """Express an FSL affine for a different moving-image grid."""
    from nitransforms.linear import Affine  # type: ignore[import-untyped]

    transform = Affine.from_filename(
        str(matrix_file),
        fmt="fsl",
        reference=str(reference_file),
        moving=str(matrix_moving_file),
    )
    return write_ras_transform_to_fsl(
        transform.matrix,
        output_moving_file,
        reference_file,
        output_file,
    )


def compute_dice_for_selection(
    registered_file: Union[str, Path],
    reference_file: Union[str, Path],
) -> float:
    """Compute DICE coefficient for registration path selection.

    Creates rough brain masks from both images by thresholding at a low
    percentile of positive voxels, then computes DICE overlap.  This is
    intentionally simple: the masks need only be consistent enough to
    discriminate a good registration from a bad one.

    Parameters
    ----------
    registered_file : Path or str
        Path to registered image.
    reference_file : Path or str
        Path to reference image.

    Returns
    -------
    float
        DICE coefficient in [0, 1].  Returns 0.0 on failure.
    """
    import nibabel as nib

    registered_file = Path(registered_file)
    reference_file = Path(reference_file)

    if not registered_file.exists():
        return 0.0

    try:
        reg_data = nib.load(str(registered_file)).get_fdata()  # type: ignore[attr-defined]
        ref_data = nib.load(str(reference_file)).get_fdata()  # type: ignore[attr-defined]

        reg_nonzero = reg_data[reg_data > 0]
        ref_nonzero = ref_data[ref_data > 0]

        if len(reg_nonzero) == 0 or len(ref_nonzero) == 0:
            return 0.0

        reg_thresh = np.percentile(reg_nonzero, MASK_THRESHOLD_PERCENTILE)
        ref_thresh = np.percentile(ref_nonzero, MASK_THRESHOLD_PERCENTILE)

        reg_mask = reg_data > reg_thresh
        ref_mask = ref_data > ref_thresh

        intersection = np.logical_and(reg_mask, ref_mask).sum()
        dice = 2.0 * intersection / (
            reg_mask.sum() + ref_mask.sum() + DICE_EPSILON
        )

        return float(dice)
    except Exception as exc:
        logger.warning(
            "DICE computation failed for %s vs %s: %s",
            registered_file,
            reference_file,
            exc,
        )
        return 0.0


def transform_mask_to_grid(
    mask_file: Optional[Union[str, Path]],
    reference_file: Optional[Union[str, Path]],
    transform_matrix: Optional[Union[str, Path]],
    out_dir: Optional[Path] = None,
) -> str:
    """Transform a binary mask to a target grid using a FLIRT matrix.

    Resamples a binary mask onto the grid of ``reference_file`` by
    applying ``transform_matrix`` (a FLIRT 4x4 affine in source-to-target
    convention) with nearest-neighbour interpolation, then thresholds,
    binarises, and fills holes via fslmaths.

    A same-space shortcut is taken when the mask and reference share
    geometry (matching shape and affine within 1e-4): the mask is
    binarised and hole-filled in place without invoking FLIRT.

    Parameters
    ----------
    mask_file : Path or str, optional
        Path to source mask.  May be ``None`` or a Nipype
        ``Undefined``; in either case the function returns ``""``.
    reference_file : Path or str, optional
        Path to reference image whose grid the output
        should match.
    transform_matrix : Path or str, optional
        Path to FLIRT 4x4 affine matrix mapping the
        mask's source frame into the reference frame.
    out_dir : Path, optional
        Directory for output and temporary files.  Defaults to
        ``Path.cwd()``.

    Returns
    -------
    str
        Path to the transformed mask, or ``""`` if any input is unset
        or the mask file does not exist.  Downstream Nipype
        ``traits.Str`` ports reject ``None``, so ``""`` is the correct
        empty sentinel.
    """
    import subprocess

    import nibabel as nib
    from nipype.interfaces.base import isdefined
    from scipy.ndimage import binary_fill_holes

    if not isdefined(mask_file) or mask_file is None:
        return ""
    if not isdefined(reference_file) or reference_file is None:
        return ""
    if not isdefined(transform_matrix) or transform_matrix is None:
        return ""

    mask_path = Path(mask_file)
    if not mask_path.is_file():
        return ""

    out_dir = out_dir or Path.cwd()
    out_file = str(out_dir / "brain_mask_dce.nii.gz")

    mask_img = nib.load(str(mask_file))
    ref_img = nib.load(str(reference_file))
    if mask_img.shape[:3] == ref_img.shape[:3]:  # type: ignore[attr-defined]
        affine_diff = float(np.abs(mask_img.affine - ref_img.affine).max())  # type: ignore[attr-defined]
        if affine_diff < 1e-4:
            mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]
            mask_bin = (mask_data > 0.5).astype(np.int16)
            mask_filled = binary_fill_holes(mask_bin).astype(np.int16)
            out_img = nib.Nifti1Image(mask_filled, ref_img.affine)  # type: ignore[attr-defined]
            nib.save(out_img, out_file)
            return out_file

    tmp_file = str(out_dir / "_tmp_brain_mask_dce.nii.gz")
    cmd = [
        "flirt",
        "-in", str(mask_file),
        "-ref", str(reference_file),
        "-out", tmp_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "nearestneighbour",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    cmd_fillh = [
        "fslmaths", tmp_file,
        "-thr", "0.5", "-bin", "-fillh", out_file,
    ]
    subprocess.run(cmd_fillh, check=True, capture_output=True)

    try:
        Path(tmp_file).unlink(missing_ok=True)
    except Exception:
        pass

    return out_file
