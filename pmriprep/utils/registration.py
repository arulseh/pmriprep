"""Registration metric calculations for PMRIPrep.

This module provides functions for calculating registration quality metrics
including DICE coefficient, correlation, normalized mutual information (NMI),
and AIF preservation metrics.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


def compare_transforms(
    mat1: NDArray[np.floating],
    mat2: NDArray[np.floating],
    radius: float = 70.0,
) -> float:
    """Compute maximum displacement between two FSL affine transforms.

    Evaluates both transforms at six test points (cube face midpoints at
    distance ``radius`` from the origin) and returns the maximum Euclidean
    distance between the transformed positions.

    Adapted from niworkflows ``compare_xforms()`` (Esteban et al., 2019).

    Args:
        mat1: First 4x4 affine matrix (FSL format).
        mat2: Second 4x4 affine matrix (FSL format).
        radius: Distance (mm) of test points from the origin.
            Default 70mm approximates the brain radius.

    Returns:
        Maximum displacement in mm across all test points.

    Raises:
        ValueError: If matrices are not 4x4.
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

    Args:
        mat1_path: Path to first FSL .mat file.
        mat2_path: Path to second FSL .mat file.
        radius: Distance (mm) of test points from origin (default: 70mm).

    Returns:
        Maximum displacement in mm.

    Raises:
        FileNotFoundError: If either file does not exist.
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

    DICE = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        mask1: First mask array (will be binarized).
        mask2: Second mask array (will be binarized).
        threshold: Threshold for binarization of non-binary masks.

    Returns:
        DICE coefficient in range [0, 1]. Returns 0.0 if both masks are empty.

    Raises:
        ValueError: If masks have different shapes.
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

    Args:
        mask1_path: Path to first mask NIfTI file.
        mask2_path: Path to second mask NIfTI file.
        threshold: Threshold for binarization.

    Returns:
        DICE coefficient in range [0, 1].

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If masks have different shapes.
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

    Args:
        image1: First image array.
        image2: Second image array (must match shape of image1).
        mask: Optional mask array. If provided, correlation is computed
              only for voxels where mask > threshold.
        threshold: Threshold for mask binarization.

    Returns:
        Pearson correlation coefficient in range [-1, 1].
        Returns 0.0 if insufficient valid voxels.

    Raises:
        ValueError: If images have different shapes.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image shapes must match: {image1.shape} vs {image2.shape}"
        )

    # Flatten images
    flat1 = image1.ravel()
    flat2 = image2.ravel()

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

    Args:
        image1_path: Path to first image NIfTI file.
        image2_path: Path to second image NIfTI file.
        mask_path: Optional path to mask NIfTI file.
        threshold: Threshold for mask binarization.

    Returns:
        Pearson correlation coefficient in range [-1, 1].

    Raises:
        FileNotFoundError: If any required file does not exist.
        ValueError: If images have different shapes.
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

    Args:
        image1: First image array.
        image2: Second image array (must match shape of image1).
        mask: Optional mask array. If provided, NMI is computed
              only for voxels where mask > threshold.
        threshold: Threshold for mask binarization.
        n_bins: Number of histogram bins for entropy calculation.

    Returns:
        Normalized mutual information in range [0, 2].
        0 indicates no mutual information, 2 indicates perfect correlation.
        Returns 0.0 if insufficient valid voxels.

    Raises:
        ValueError: If images have different shapes.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image shapes must match: {image1.shape} vs {image2.shape}"
        )

    # Flatten images
    flat1 = image1.ravel().astype(np.float64)
    flat2 = image2.ravel().astype(np.float64)

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

    Args:
        image1_path: Path to first image NIfTI file.
        image2_path: Path to second image NIfTI file.
        mask_path: Optional path to mask NIfTI file.
        threshold: Threshold for mask binarization.
        n_bins: Number of histogram bins.

    Returns:
        Normalized mutual information in range [0, 2].

    Raises:
        FileNotFoundError: If any required file does not exist.
        ValueError: If images have different shapes.
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

    Args:
        aif_before: AIF curve before transformation (1D array, time series).
        aif_after: AIF curve after transformation (1D array, same length).

    Returns:
        Peak preservation ratio: peak_after / peak_before.
        Returns 1.0 if peaks are identical, <1.0 if peak is reduced,
        >1.0 if peak is increased. Returns 0.0 if before peak is zero.

    Raises:
        ValueError: If arrays have different lengths or are not 1D.
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

    Args:
        aif_before: AIF curve before transformation (1D array).
        aif_after: AIF curve after transformation (1D array).
        dt: Time step between samples (for AUC calculation).

    Returns:
        Dictionary with:
            - peak_ratio: peak_after / peak_before
            - auc_ratio: auc_after / auc_before
            - peak_time_shift: Difference in peak time indices
            - quality: 'good' if both ratios in [0.9, 1.1], else 'degraded'

    Raises:
        ValueError: If arrays have different lengths or are not 1D.
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
