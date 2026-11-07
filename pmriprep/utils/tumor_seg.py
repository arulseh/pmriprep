"""HD-GLIO tumor mask extraction utilities for PMRIPrep.

This module provides utilities for extracting binary masks from HD-GLIO
multi-class segmentation outputs and creating NAWM ROIs.

Functions
---------
extract_tumor_masks
    Extract binary masks for enhancing, non-enhancing, and whole tumor
create_nawm_roi
    Create supratentorial normal-appearing white matter ROI for normalization
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.nifti import physical_voxel_volume_mm3

EPSILON_VOXEL_SIZE = 1e-6      # Guard for zero voxel sizes
MAX_KERNEL_RADIUS_VOXELS = 200  # Memory safety cap for kernel generation
NAWM_LATERALITY_FRACTION = 0.60
NAWM_SYMMETRY_MAX_ANGLE_DEGREES = 25.0
NAWM_SYMMETRY_MAX_SHIFT_MM = 10.0


def extract_tumor_masks(
    segmentation_file: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
    prefix: str = "",
) -> dict[str, str]:
    """Extract binary tumor masks from HD-GLIO multi-class segmentation.

    HD-GLIO outputs multi-class segmentation with the following labels:
    - 0: Background (non-tumor)
    - 1: Non-enhancing tumor (necrosis, non-enhancing core)
    - 2: Enhancing tumor (contrast-enhancing tumor)

    This function creates separate binary masks for each class and their
    combinations, which are needed for various downstream analyses.

    Parameters
    ----------
    segmentation_file : str or Path
        Path to HD-GLIO segmentation NIfTI file (multi-class labels 0, 1, 2)
    output_dir : str or Path, optional
        Output directory for extracted masks. If None, uses same directory
        as input file.
    prefix : str, optional
        Prefix for output filenames (e.g., ``"sub-001_ses-01_"``)

    Returns
    -------
    dict[str, str]
        Dictionary mapping mask type to output file path:
        - "enhancing": Path to enhancing tumor mask (label 2)
        - "non_enhancing": Path to non-enhancing tumor mask (label 1)
        - "whole_tumor": Path to whole tumor mask (labels 1 + 2)

    Examples
    --------
    >>> masks = extract_tumor_masks("hdglio_seg.nii.gz", output_dir="/out")  # doctest: +SKIP
    >>> masks["enhancing"]  # doctest: +SKIP
    '/out/desc-enhancing_mask.nii.gz'
    >>> masks["whole_tumor"]  # doctest: +SKIP
    '/out/desc-wholetumor_mask.nii.gz'

    Notes
    -----
    The masks are saved as uint8 NIfTI files with values 0 (background)
    and 1 (tumor). The affine and header are preserved from the input
    segmentation file.

    For BIDS compatibility, mask files use the `desc-` entity:
    - desc-enhancing_mask.nii.gz
    - desc-nonenhancing_mask.nii.gz
    - desc-wholetumor_mask.nii.gz
    """
    import nibabel as nib  # type: ignore[import-untyped]

    # Load segmentation
    seg_path = Path(segmentation_file)
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()  # type: ignore[attr-defined]
    affine = seg_img.affine  # type: ignore[attr-defined]
    header = seg_img.header  # type: ignore[attr-defined]

    # Determine output directory
    if output_dir is None:
        out_dir = seg_path.parent
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Create binary masks
    # Label 1: Non-enhancing tumor
    non_enhancing_mask = (seg_data == 1).astype(np.uint8)

    # Label 2: Enhancing tumor
    enhancing_mask = (seg_data == 2).astype(np.uint8)

    # Whole tumor: Labels 1 + 2
    whole_tumor_mask = ((seg_data == 1) | (seg_data == 2)).astype(np.uint8)

    # Save masks
    output_files: dict[str, str] = {}

    mask_definitions = [
        ("enhancing", enhancing_mask, "desc-enhancing_mask.nii.gz"),
        ("non_enhancing", non_enhancing_mask, "desc-nonenhancing_mask.nii.gz"),
        ("whole_tumor", whole_tumor_mask, "desc-wholetumor_mask.nii.gz"),
    ]

    for mask_name, mask_data, filename in mask_definitions:
        # Add prefix if provided
        if prefix:
            out_filename = f"{prefix}{filename}"
        else:
            out_filename = filename

        out_path = out_dir / out_filename

        # Create NIfTI image preserving header info
        mask_img = nib.Nifti1Image(mask_data, affine, header)  # type: ignore[attr-defined]
        nib.save(mask_img, out_path)

        output_files[mask_name] = str(out_path)

    return output_files


def _create_spherical_kernel(
    radius_mm: float,
    voxel_sizes: tuple[float, float, float],
) -> npt.NDArray[np.bool_]:
    """Create a spherical structuring element with anisotropic voxel support.

    Creates a 3D binary kernel representing a sphere of the specified radius
    in millimeters, properly accounting for non-isotropic voxel dimensions.

    Parameters
    ----------
    radius_mm : float
        Radius of the sphere in millimeters
    voxel_sizes : tuple of float
        Voxel dimensions (x, y, z) in millimeters

    Returns
    -------
    ndarray of bool
        3D binary array representing the spherical kernel

    Raises
    ------
    ValueError
        If a kernel radius exceeds ``MAX_KERNEL_RADIUS_VOXELS`` on any axis.

    Notes
    -----
    The kernel size in each dimension is calculated as:
        size = 2 * ceil(radius_mm / voxel_size) + 1

    For anisotropic voxels (e.g., 1x1x3 mm), the kernel will have different
    numbers of voxels in each dimension to approximate a true sphere.

    Examples
    --------
    >>> kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 1.0))
    >>> kernel.shape
    (11, 11, 11)

    >>> kernel = _create_spherical_kernel(5.0, (1.0, 1.0, 3.0))
    >>> kernel.shape
    (11, 11, 5)
    """
    vx = max(float(voxel_sizes[0]), EPSILON_VOXEL_SIZE)
    vy = max(float(voxel_sizes[1]), EPSILON_VOXEL_SIZE)
    vz = max(float(voxel_sizes[2]), EPSILON_VOXEL_SIZE)

    # Calculate kernel size in voxels for each dimension
    # Need at least radius_mm / voxel_size voxels on each side of center
    nx = int(np.ceil(radius_mm / vx))
    ny = int(np.ceil(radius_mm / vy))
    nz = int(np.ceil(radius_mm / vz))

    # Safety cap: prevent enormous kernels from tiny voxels
    if nx > MAX_KERNEL_RADIUS_VOXELS or ny > MAX_KERNEL_RADIUS_VOXELS or nz > MAX_KERNEL_RADIUS_VOXELS:
        raise ValueError(
            f"Kernel radius exceeds safety cap ({MAX_KERNEL_RADIUS_VOXELS} voxels): "
            f"nx={nx}, ny={ny}, nz={nz} for radius_mm={radius_mm}, "
            f"voxel_sizes=({vx}, {vy}, {vz})"
        )

    # Total kernel size (diameter + 1 for center)
    size_x = 2 * nx + 1
    size_y = 2 * ny + 1
    size_z = 2 * nz + 1

    # Create coordinate grids centered at the kernel center
    # Coordinates are in millimeters (float64 to prevent overflow with small voxels)
    x = np.arange(size_x, dtype=np.float64) - nx
    y = np.arange(size_y, dtype=np.float64) - ny
    z = np.arange(size_z, dtype=np.float64) - nz

    # Scale by voxel sizes to get physical coordinates
    xx, yy, zz = np.meshgrid(x * vx, y * vy, z * vz, indexing="ij")

    # Calculate distance from center in mm
    distance = np.sqrt(xx**2 + yy**2 + zz**2)

    # Create spherical kernel
    kernel: npt.NDArray[np.bool_] = distance <= radius_mm

    return kernel


def dilate_mask_from_array(
    mask_data: npt.NDArray[np.bool_],
    radius_mm: float,
    voxel_sizes: tuple[float, float, float],
) -> npt.NDArray[np.bool_]:
    """Dilate a binary mask array using a spherical structuring element.

    This is a convenience function for dilating mask data directly without
    file I/O, useful for in-memory processing pipelines.

    Parameters
    ----------
    mask_data : ndarray of bool
        3D binary mask array
    radius_mm : float
        Dilation radius in millimeters
    voxel_sizes : tuple of float
        Voxel dimensions (x, y, z) in millimeters

    Returns
    -------
    ndarray of bool
        Dilated binary mask array

    Examples
    --------
    >>> mask = np.zeros((100, 100, 50), dtype=bool)
    >>> mask[40:60, 40:60, 20:30] = True
    >>> dilated = dilate_mask_from_array(mask, 5.0, (1.0, 1.0, 2.0))
    >>> bool(dilated.sum() > mask.sum())
    True

    """
    from scipy import ndimage  # type: ignore[import-untyped]

    # Create spherical kernel
    kernel = _create_spherical_kernel(radius_mm, voxel_sizes)

    # Apply dilation
    dilated: npt.NDArray[np.bool_] = ndimage.binary_dilation(mask_data, structure=kernel)

    return dilated


def erode_mask_from_array(
    mask_data: npt.NDArray[np.bool_],
    radius_mm: float,
    voxel_sizes: tuple[float, float, float],
) -> npt.NDArray[np.bool_]:
    """Erode a binary mask using a single-pass spherical structuring element.

    Uses a single erosion pass with an ellipsoidal kernel of the given
    physical radius, rather than iterative 1-voxel erosions.

    Parameters
    ----------
    mask_data : ndarray of bool
        3D binary mask array
    radius_mm : float
        Erosion radius in millimeters
    voxel_sizes : tuple of float
        Voxel dimensions (x, y, z) in millimeters

    Returns
    -------
    ndarray of bool
        Eroded binary mask array
    """
    from scipy import ndimage  # type: ignore[import-untyped]

    if radius_mm <= 0:
        return mask_data.copy()  # type: ignore[no-any-return]

    kernel = _create_spherical_kernel(radius_mm, voxel_sizes)
    eroded: npt.NDArray[np.bool_] = ndimage.binary_erosion(mask_data, structure=kernel)

    return eroded


def get_mask_volume_ml(
    mask_file: Union[str, Path],
) -> float:
    """Calculate volume of a binary mask in milliliters.

    Parameters
    ----------
    mask_file : str or Path
        Path to binary mask NIfTI file

    Returns
    -------
    float
        Volume of non-zero voxels in milliliters

    Examples
    --------
    >>> volume = get_mask_volume_ml("tumor_mask.nii.gz")  # doctest: +SKIP
    >>> print(f"Tumor volume: {volume:.1f} mL")  # doctest: +SKIP
    Tumor volume: 42.5 mL

    Notes
    -----
    Volume is calculated as:
        volume_ml = n_voxels * voxel_volume_mm3 / 1000
    """
    import nibabel as nib  # type: ignore[import-untyped]

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]
    voxel_volume_mm3 = physical_voxel_volume_mm3(mask_img.affine)  # type: ignore[attr-defined]

    # Count non-zero voxels
    n_voxels = int(np.sum(mask_data > 0.5))

    # Convert to mL (1 mL = 1000 mm³)
    volume_ml = n_voxels * voxel_volume_mm3 / 1000.0

    return volume_ml


def get_mask_statistics(
    mask_file: Union[str, Path],
) -> dict[str, float]:
    """Get comprehensive statistics for a binary mask.

    Parameters
    ----------
    mask_file : str or Path
        Path to binary mask NIfTI file

    Returns
    -------
    dict
        Dictionary with mask statistics:
        - n_voxels: Number of non-zero voxels
        - volume_ml: Volume in milliliters
        - volume_mm3: Volume in cubic millimeters
        - bounding_box: (min_x, max_x, min_y, max_y, min_z, max_z)
        - centroid_voxels: (x, y, z) center of mass in voxel coordinates
        - voxel_sizes: (vx, vy, vz) voxel dimensions in mm

    Examples
    --------
    >>> stats = get_mask_statistics("tumor_mask.nii.gz")  # doctest: +SKIP
    >>> print(f"Volume: {stats['volume_ml']:.1f} mL")  # doctest: +SKIP
    >>> print(f"Centroid: {stats['centroid_voxels']}")  # doctest: +SKIP
    """
    import nibabel as nib  # type: ignore[import-untyped]
    from scipy import ndimage  # type: ignore[import-untyped]

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]
    zooms = mask_img.header.get_zooms()[:3]  # type: ignore[attr-defined]

    # Binary mask
    binary_mask = mask_data > 0.5

    # Voxel count and volume
    n_voxels = int(np.sum(binary_mask))
    voxel_volume_mm3 = physical_voxel_volume_mm3(mask_img.affine)  # type: ignore[attr-defined]
    volume_mm3 = n_voxels * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0

    # Bounding box
    if n_voxels > 0:
        nonzero = np.argwhere(binary_mask)
        bbox_min = tuple(int(x) for x in nonzero.min(axis=0))
        bbox_max = tuple(int(x) for x in nonzero.max(axis=0))
        bounding_box = (
            bbox_min[0], bbox_max[0],
            bbox_min[1], bbox_max[1],
            bbox_min[2], bbox_max[2],
        )

        # Center of mass
        centroid = ndimage.center_of_mass(binary_mask)
        centroid_voxels = tuple(float(c) for c in centroid)
    else:
        bounding_box = (0, 0, 0, 0, 0, 0)
        centroid_voxels = (0.0, 0.0, 0.0)

    return {
        "n_voxels": float(n_voxels),
        "volume_ml": volume_ml,
        "volume_mm3": volume_mm3,
        "bounding_box": bounding_box,  # type: ignore[dict-item]
        "centroid_voxels": centroid_voxels,  # type: ignore[dict-item]
        "voxel_sizes": tuple(float(z) for z in zooms),  # type: ignore[dict-item]
    }


@dataclass
class TumorAnalysis:
    """Tumor compartment and laterality used for NAWM selection."""

    is_supratentorial: bool
    laterality: Literal["left", "right", "bilateral"]
    centroid_voxels: tuple[float, float, float]
    volume_ml: float
    compartment: Literal["supratentorial", "infratentorial", "unavailable"]
    supratentorial_fraction: float
    cerebellar_fraction: float
    other_fraction: float
    atlas_coverage_fraction: float
    left_fraction: float
    right_fraction: float
    laterality_method: Literal["brain_symmetry", "world_x_fallback"]
    midline_angle_degrees: float
    symmetry_dice: float


@dataclass
class _MidlineEstimate:
    """Subject-space midsagittal plane estimated from brain symmetry."""

    signed_distance: npt.NDArray[np.float64]
    angle_degrees: float
    symmetry_dice: float


def estimate_brain_symmetry_midline(
    brain_mask: npt.NDArray[np.bool_],
    affine: npt.NDArray[np.float64],
) -> Optional[_MidlineEstimate]:
    """Estimate a subject-space midsagittal plane from axial brain symmetry.

    Parameters
    ----------
    brain_mask : ndarray of bool
        Native-space three-dimensional brain support.
    affine : ndarray
        Voxel-to-RAS transform for the brain mask.

    Returns
    -------
    _MidlineEstimate or None
        Signed distances in millimetres, plane angle and symmetry overlap.
        Positive distances point toward RAS right. Returns None when support
        or geometry is insufficient for the estimate.
    """
    from scipy.ndimage import map_coordinates  # type: ignore[import-untyped]

    brain = np.asarray(brain_mask, dtype=bool)
    projection = np.any(brain, axis=2)
    points = np.argwhere(projection)
    if points.shape[0] < 10:
        return None

    affine_array = np.asarray(affine, dtype=float)
    first_axis = affine_array[:3, 0]
    second_axis = affine_array[:3, 1]
    first_norm = float(np.linalg.norm(first_axis))
    if first_norm <= 0:
        return None
    basis_first = first_axis / first_norm
    second_orthogonal = second_axis - np.dot(second_axis, basis_first) * basis_first
    second_norm = float(np.linalg.norm(second_orthogonal))
    if second_norm <= 0:
        return None
    basis_second = second_orthogonal / second_norm
    basis = np.column_stack((basis_first, basis_second))
    plane_matrix = basis.T @ affine_array[:3, :2]
    plane_offset = basis.T @ affine_array[:3, 3]
    try:
        inverse_plane_matrix = np.linalg.inv(plane_matrix)
    except np.linalg.LinAlgError:
        return None

    # Score on a bounded grid so native 1-mm T1w matrices do not turn this
    # one-time geometric estimate into a large workflow cost. The final plane
    # is evaluated on the full image grid below.
    sample_step = max(1, int(np.ceil(max(projection.shape) / 96)))
    sample_i = np.arange(0, projection.shape[0], sample_step)
    sample_j = np.arange(0, projection.shape[1], sample_step)
    sample_ii, sample_jj = np.meshgrid(sample_i, sample_j, indexing="ij")
    search_grid = np.column_stack((sample_ii.ravel(), sample_jj.ravel()))
    search_grid_plane = search_grid @ plane_matrix.T + plane_offset
    projection_sample = projection[np.ix_(sample_i, sample_j)]
    brain_plane = points @ plane_matrix.T + plane_offset
    center = np.mean(brain_plane, axis=0)
    projection_float = projection.astype(float)

    def _search(
        angles: npt.NDArray[np.float64],
        shifts: npt.NDArray[np.float64],
    ) -> tuple[float, float, float, npt.NDArray[np.float64], float]:
        best: Optional[tuple[float, float, float, npt.NDArray[np.float64], float]] = (
            None
        )
        for angle_degrees in angles:
            radians = np.deg2rad(angle_degrees)
            normal = np.asarray([np.cos(radians), np.sin(radians)], dtype=float)
            center_projection = float(center @ normal)
            for shift_mm in shifts:
                position = center_projection + float(shift_mm)
                signed = search_grid_plane @ normal - position
                reflected_plane = search_grid_plane - 2.0 * signed[:, None] * normal
                reflected_voxels = (
                    reflected_plane - plane_offset
                ) @ inverse_plane_matrix.T
                reflected = map_coordinates(
                    projection_float,
                    [reflected_voxels[:, 0], reflected_voxels[:, 1]],
                    order=1,
                    mode="constant",
                    cval=0.0,
                ).reshape(projection_sample.shape)
                denominator = float(np.sum(projection_sample) + np.sum(reflected))
                overlap = (
                    2.0 * float(np.sum(projection_sample * reflected)) / denominator
                    if denominator > 0
                    else 0.0
                )
                candidate = (
                    overlap,
                    float(angle_degrees),
                    position,
                    normal,
                    float(shift_mm),
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            raise RuntimeError("Brain-symmetry search produced no result")
        return best

    coarse = _search(
        np.arange(
            -NAWM_SYMMETRY_MAX_ANGLE_DEGREES,
            NAWM_SYMMETRY_MAX_ANGLE_DEGREES + 0.01,
            2.0,
        ),
        np.arange(-NAWM_SYMMETRY_MAX_SHIFT_MM, NAWM_SYMMETRY_MAX_SHIFT_MM + 0.01, 1.0),
    )
    overlap, angle_degrees, position, normal, _ = _search(
        np.arange(coarse[1] - 2.0, coarse[1] + 2.01, 0.25),
        np.arange(coarse[4] - 1.0, coarse[4] + 1.01, 0.25),
    )
    # Orient positive distance toward world-space right (RAS +x).
    if float((basis @ normal)[0]) < 0:
        normal = -normal
        position = -position
    full_grid = np.indices(projection.shape, dtype=float).reshape(2, -1).T
    full_grid_plane = full_grid @ plane_matrix.T + plane_offset
    signed_2d = (full_grid_plane @ normal - position).reshape(projection.shape)
    signed_3d = np.broadcast_to(signed_2d[:, :, None], brain.shape).copy()
    return _MidlineEstimate(signed_3d, angle_degrees, overlap)


def _analyze_tumor_anatomy(
    tumor_mask: npt.NDArray[np.bool_],
    affine: npt.NDArray[np.float64],
    supratentorial_mask: Optional[npt.NDArray[np.bool_]] = None,
    cerebellar_mask: Optional[npt.NDArray[np.bool_]] = None,
    midline: Optional[_MidlineEstimate] = None,
) -> TumorAnalysis:
    """Classify tumor compartment by atlas overlap and laterality by symmetry."""
    from scipy import ndimage  # type: ignore[import-untyped]

    n_voxels = int(np.sum(tumor_mask))
    centroid_voxels = (
        tuple(float(value) for value in ndimage.center_of_mass(tumor_mask))
        if n_voxels
        else (0.0, 0.0, 0.0)
    )
    voxel_volume_mm3 = physical_voxel_volume_mm3(affine)
    volume_ml = n_voxels * voxel_volume_mm3 / 1000.0

    if n_voxels and supratentorial_mask is not None and cerebellar_mask is not None:
        supra_voxels = int(np.sum(tumor_mask & supratentorial_mask))
        cereb_voxels = int(np.sum(tumor_mask & cerebellar_mask))
        covered = supratentorial_mask | cerebellar_mask
        other_voxels = int(np.sum(tumor_mask & ~covered))
        supratentorial_fraction = supra_voxels / n_voxels
        cerebellar_fraction = cereb_voxels / n_voxels
        other_fraction = other_voxels / n_voxels
        atlas_coverage_fraction = (supra_voxels + cereb_voxels) / n_voxels
        is_supratentorial = supratentorial_fraction > max(
            cerebellar_fraction, other_fraction
        )
        compartment: Literal["supratentorial", "infratentorial", "unavailable"] = (
            "supratentorial" if is_supratentorial else "infratentorial"
        )
    else:
        supratentorial_fraction = float("nan")
        cerebellar_fraction = float("nan")
        other_fraction = float("nan")
        atlas_coverage_fraction = float("nan")
        is_supratentorial = True
        compartment = "unavailable"

    laterality_mask = tumor_mask
    if supratentorial_mask is not None:
        laterality_mask = tumor_mask & supratentorial_mask
    left_mask = _create_hemisphere_mask(tumor_mask.shape, affine, "left", midline)
    right_mask = _create_hemisphere_mask(tumor_mask.shape, affine, "right", midline)
    left_voxels = int(np.sum(laterality_mask & left_mask))
    right_voxels = int(np.sum(laterality_mask & right_mask))
    total_voxels = left_voxels + right_voxels
    if total_voxels == 0:
        left_fraction = right_fraction = 0.0
        laterality: Literal["left", "right", "bilateral"] = "bilateral"
    else:
        left_fraction = left_voxels / total_voxels
        right_fraction = right_voxels / total_voxels
        if left_fraction > NAWM_LATERALITY_FRACTION:
            laterality = "left"
        elif right_fraction > NAWM_LATERALITY_FRACTION:
            laterality = "right"
        else:
            laterality = "bilateral"

    return TumorAnalysis(
        is_supratentorial=is_supratentorial,
        laterality=laterality,
        centroid_voxels=centroid_voxels,  # type: ignore[arg-type]
        volume_ml=volume_ml,
        compartment=compartment,
        supratentorial_fraction=supratentorial_fraction,
        cerebellar_fraction=cerebellar_fraction,
        other_fraction=other_fraction,
        atlas_coverage_fraction=atlas_coverage_fraction,
        left_fraction=left_fraction,
        right_fraction=right_fraction,
        laterality_method=(
            "brain_symmetry" if midline is not None else "world_x_fallback"
        ),
        midline_angle_degrees=(
            midline.angle_degrees if midline is not None else float("nan")
        ),
        symmetry_dice=(midline.symmetry_dice if midline is not None else float("nan")),
    )


def _create_hemisphere_mask(
    shape: tuple[int, int, int],
    affine: npt.NDArray[np.float64],
    hemisphere: Literal["left", "right"],
    midline: Optional[_MidlineEstimate] = None,
) -> npt.NDArray[np.bool_]:
    """Create a hemisphere mask from a subject midline or world-x fallback.

    Parameters
    ----------
    shape : tuple of int
        Image dimensions (i, j, k)
    affine : ndarray
        4x4 affine transformation matrix
    hemisphere : {"left", "right"}
        Which hemisphere to select (RAS: left = x < 0, right = x > 0)
    midline : _MidlineEstimate, optional
        Subject-space midsagittal plane. If unavailable, world x=0 is used.

    Raises
    ------
    ValueError
        If a supplied midline estimate has a different shape from the mask.

    Notes
    -----
    Voxels exactly on the selected midsagittal plane are excluded from both
    masks because they are not unambiguously ipsilateral or contralateral.

    Returns
    -------
    ndarray of bool
        Binary mask for the selected hemisphere
    """
    if midline is not None:
        if midline.signed_distance.shape != shape:
            raise ValueError("Midline estimate shape does not match requested mask")
        return (
            midline.signed_distance < 0
            if hemisphere == "left"
            else midline.signed_distance > 0
        )

    # World-x fallback when a brain-derived symmetry plane is unavailable.
    i_coords = np.arange(shape[0])
    j_coords = np.arange(shape[1])
    k_coords = np.arange(shape[2])
    ii, jj, kk = np.meshgrid(i_coords, j_coords, k_coords, indexing="ij")

    # Transform to physical x coordinate
    x_physical = (
        affine[0, 0] * ii + affine[0, 1] * jj + affine[0, 2] * kk + affine[0, 3]
    )

    # RAS: +x = right, -x = left
    if hemisphere == "left":
        mask: npt.NDArray[np.bool_] = x_physical < 0
    else:
        mask = x_physical > 0

    return mask


def _transform_atlas_to_t1w(
    atlas_path: str,
    reference_path: str,
    t1w_to_mni_xfm: str,
) -> Optional[str]:
    """Transform MNI atlas to T1w space via FLIRT.

    Parameters
    ----------
    atlas_path : str
        Path to MNI-space atlas NIfTI file
    reference_path : str
        Path to T1w reference image (for output geometry)
    t1w_to_mni_xfm : str
        Path to T1w-to-MNI FLIRT transform (.mat file). Inverted
        internally to obtain MNI-to-T1w before applying.

    Returns
    -------
    str or None
        Path to transformed atlas in T1w space, or None if transform fails
    """
    import subprocess
    import tempfile

    inv_path: Optional[str] = None
    try:
        # Invert T1w-to-MNI to get MNI-to-T1w
        inv_file = tempfile.NamedTemporaryFile(suffix="_inv.mat", delete=False)
        inv_path = inv_file.name
        inv_file.close()

        subprocess.run(
            [
                "convert_xfm",
                "-omat", inv_path,
                "-inverse", t1w_to_mni_xfm,
            ],
            check=True,
            capture_output=True,
        )

        out_file = tempfile.NamedTemporaryFile(
            suffix="_atlas_t1w.nii.gz", delete=False
        )
        out_path = out_file.name
        out_file.close()

        subprocess.run(
            [
                "flirt",
                "-in", atlas_path,
                "-ref", reference_path,
                "-applyxfm",
                "-init", inv_path,
                "-interp", "nearestneighbour",
                "-out", out_path,
            ],
            check=True,
            capture_output=True,
        )

        return out_path

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    finally:
        # Clean up inverted transform temp file
        if inv_path is not None:
            tmp_inv = Path(inv_path)
            if tmp_inv.exists():
                tmp_inv.unlink()


def create_nawm_roi(
    tumor_mask_file: Union[str, Path],
    wm_probseg_file: Union[str, Path],
    output_file: Union[str, Path, None] = None,
    wm_threshold: float = 0.9,
    tumor_margin_mm: float = 20.0,
    nawm_erosion: int = 2,
    nawm_min_voxels: int = 500,
    t1w_to_mni_xfm: Optional[str] = None,
    reference_image: Optional[str] = None,
    region_masks_dir: Optional[str] = None,
    brain_mask_file: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Create a supratentorial NAWM ROI for perfusion normalization.

    This function creates an NAWM (Normal Appearing White Matter) ROI for
    use in normalizing perfusion maps. For supratentorial tumors, NAWM is
    selected contralaterally or from the lower-burden hemisphere. For
    infratentorial tumors, both supratentorial hemispheres are retained.

    When atlas files and the MNI transform are available, the whole tumor is
    classified by overlap with transformed supratentorial and cerebellar masks.
    The reference ROI itself is always restricted to supratentorial white matter.

    The order of operations follows the reference implementation:
    brain_mask -> WM threshold -> region mask -> tumor exclusion -> erosion,
    with an erosion-only adaptive fallback (2 -> 1 -> 0 voxels).

    Parameters
    ----------
    tumor_mask_file : str or Path
        Path to whole tumor binary mask NIfTI file (from HD-GLIO)
    wm_probseg_file : str or Path
        Path to white matter probability segmentation from FSL FAST
    output_file : str or Path, optional
        Path to output NAWM mask file. If None, generates filename by
        appending "_nawm" to tumor mask filename.
    wm_threshold : float, default=0.9
        Minimum white matter probability for inclusion in NAWM ROI
    tumor_margin_mm : float, default=20.0
        Exclusion margin (mm) around tumor to avoid peritumoral effects
    nawm_erosion : int, default=2
        Erosion level in voxels. Converted to mm as
        ``nawm_erosion * min(voxel_sizes)`` for single-pass ellipsoidal
        erosion (from HDGLIOConfig.nawm_erosion).
    nawm_min_voxels : int, default=500
        Minimum required voxels in final NAWM ROI
        (from HDGLIOConfig.nawm_min_voxels)
    t1w_to_mni_xfm : str, optional
        Path to T1w-to-MNI transform (.mat). Required for atlas-based
        region restriction and tumor-compartment classification.
    reference_image : str, optional
        Path to T1w reference image for atlas transformation.
    region_masks_dir : str, optional
        Directory containing the full MNI supratentorial and cerebellar region
        masks (``Supra.nii.gz`` and ``Cerebellum.nii.gz``).
    brain_mask_file : str or Path, optional
        Path to brain mask NIfTI file. Used as the initial base constraint
        (Step 1). If None, all voxels are considered.

    Returns
    -------
    dict
        Dictionary containing:
        - "output_file": Path to output NAWM mask file
        - "analysis": Dictionary with analysis details:

            - "tumor_laterality": "left", "right", or "bilateral"
            - "compartment_label": "supratentorial", "infratentorial", or
              "unavailable"
            - "f_supra", "f_cereb", "f_other": tumor overlap fractions
            - "selected_hemisphere": "left", "right", or "both"
            - "nawm_compartment": always "supratentorial"
            - "n_voxels": Final voxel count in NAWM ROI
            - "volume_ml": NAWM ROI volume in milliliters
            - "fallback_used": bool indicating if adaptive fallback was needed
            - "atlas_used": bool indicating if atlas restriction was applied
            - "valid": bool indicating if minimum voxel count was met
            - "erosion_voxels_final": int, final erosion level used
            - "erosion_mm_final": float, final erosion radius in mm
            - "qc_steps": dict with per-step voxel counts

    Examples
    --------
    >>> result = create_nawm_roi(  # doctest: +SKIP
    ...     "tumor_mask.nii.gz",
    ...     "wm_probseg.nii.gz",
    ...     nawm_erosion=2,
    ...     nawm_min_voxels=500
    ... )
    >>> print(f"NAWM voxels: {result['analysis']['n_voxels']}")  # doctest: +SKIP
    NAWM voxels: 2847

    Raises
    ------
    ValueError
        If the WM probability map or supplied brain mask has a different
        spatial shape from the tumor mask.

    Notes
    -----
    The NAWM selection algorithm:

    1. Base mask: brain_mask (if provided) or np.ones()
    2. WM threshold: intersect with wm_data >= 0.9
    3. Region mask: restrict to supratentorial tissue and, for supratentorial
       tumors, the selected hemisphere
    4. Tumor exclusion: subtract dilated tumor (20mm margin)
    5. Erosion: SINGLE pass, ellipsoidal kernel of nawm_erosion * min(voxel_sizes) mm
    6. Adaptive fallback: erosion-only reduction (2 -> 1 -> 0 voxels)

    This does NOT relax the WM threshold, tumor margin, or hemisphere
    constraints; only erosion is reduced.
    If still below nawm_min_voxels after 0-erosion, the mask is saved
    with ``valid=False`` rather than raising an error.
    """
    import logging

    import nibabel as nib  # type: ignore[import-untyped]

    logger = logging.getLogger(__name__)

    # Load tumor mask
    tumor_path = Path(tumor_mask_file)
    tumor_img = nib.load(tumor_path)
    tumor_data = tumor_img.get_fdata()  # type: ignore[attr-defined]
    tumor_mask = tumor_data > 0.5
    affine: npt.NDArray[np.float64] = tumor_img.affine  # type: ignore[attr-defined]
    header = tumor_img.header  # type: ignore[attr-defined]

    # Load WM probability map
    wm_img = nib.load(wm_probseg_file)
    wm_data = wm_img.get_fdata()  # type: ignore[attr-defined]

    # Verify dimensions match
    if tumor_mask.shape != wm_data.shape:
        raise ValueError(
            f"Dimension mismatch: tumor mask {tumor_mask.shape} vs "
            f"WM probseg {wm_data.shape}"
        )

    # Get voxel sizes
    zooms = header.get_zooms()[:3]  # type: ignore[attr-defined]
    voxel_sizes: tuple[float, float, float] = (
        float(zooms[0]),
        float(zooms[1]),
        float(zooms[2]),
    )

    # Load the brain mask before laterality analysis so the midsagittal plane
    # is estimated in native T1w geometry rather than assumed to be world x=0.
    if brain_mask_file is not None:
        brain_img = nib.load(brain_mask_file)
        brain_data = brain_img.get_fdata()  # type: ignore[attr-defined]
        brain_mask = brain_data > 0.5
        if brain_mask.shape != tumor_mask.shape:
            raise ValueError(
                f"Dimension mismatch: brain mask {brain_mask.shape} vs "
                f"tumor mask {tumor_mask.shape}"
            )
    else:
        brain_mask = np.ones(tumor_mask.shape, dtype=bool)

    midline = (
        estimate_brain_symmetry_midline(brain_mask, affine)
        if brain_mask_file is not None
        else None
    )
    warnings: list[str] = []
    if midline is None:
        warnings.append("symmetry_midline_unavailable_world_x_fallback")
        logger.warning(
            "Brain-symmetry midline unavailable; NAWM laterality uses world x=0"
        )

    # Transform both full atlas compartments into T1w space. The small overlap
    # in the source masks is assigned to cerebellum so it cannot enter the
    # supratentorial normalization ROI or be double-counted during classification.
    atlas_used = False
    supra_mask: Optional[npt.NDArray[np.bool_]] = None
    cerebellar_mask: Optional[npt.NDArray[np.bool_]] = None
    if region_masks_dir and t1w_to_mni_xfm and reference_image:
        transformed_masks: dict[str, npt.NDArray[np.bool_]] = {}
        for name in ("Supra", "Cerebellum"):
            atlas_path = Path(region_masks_dir) / f"{name}.nii.gz"
            if not atlas_path.exists():
                continue
            transformed_atlas = _transform_atlas_to_t1w(
                str(atlas_path), reference_image, t1w_to_mni_xfm
            )
            if transformed_atlas is None:
                continue
            try:
                atlas_data = nib.load(transformed_atlas).get_fdata()  # type: ignore[attr-defined]
                atlas_binary = atlas_data > 0.5
                if atlas_binary.shape == tumor_mask.shape:
                    transformed_masks[name] = atlas_binary
            finally:
                tmp_atlas = Path(transformed_atlas)
                if tmp_atlas.exists():
                    tmp_atlas.unlink()
        cerebellar_mask = transformed_masks.get("Cerebellum")
        supra_mask = transformed_masks.get("Supra")
        if supra_mask is not None:
            if cerebellar_mask is not None:
                supra_mask = supra_mask & ~cerebellar_mask
            atlas_used = True
    if not atlas_used:
        warnings.append("supratentorial_atlas_unavailable")
        logger.warning(
            "Supratentorial atlas restriction unavailable; NAWM selection "
            "falls back to a hemisphere-only mask"
        )

    analysis = _analyze_tumor_anatomy(
        tumor_mask,
        affine,
        supratentorial_mask=supra_mask,
        cerebellar_mask=cerebellar_mask,
        midline=midline,
    )

    # Determine which hemisphere to use for NAWM
    # For unilateral tumors: select contralateral hemisphere
    # For bilateral tumors: select hemisphere with less tumor volume
    if analysis.compartment == "infratentorial":
        selected_hemisphere: Literal["left", "right", "both"] = "both"
        hemisphere_selection_rule = "bilateral_supratentorial_for_infratentorial_tumor"
        warnings.append("infratentorial_tumor_uses_bilateral_supratentorial_nawm")
        hemisphere_mask = np.ones(tumor_mask.shape, dtype=bool)
    elif analysis.laterality == "left":
        selected_hemisphere = "right"
        hemisphere_selection_rule = "contralateral_to_lateralized_supratentorial_tumor"
        hemisphere_mask = _create_hemisphere_mask(
            tumor_mask.shape, affine, selected_hemisphere, midline
        )
    elif analysis.laterality == "right":
        selected_hemisphere = "left"
        hemisphere_selection_rule = "contralateral_to_lateralized_supratentorial_tumor"
        hemisphere_mask = _create_hemisphere_mask(
            tumor_mask.shape, affine, selected_hemisphere, midline
        )
    else:
        # Bilateral supratentorial tumor: choose the lower-burden hemisphere.
        hemi_left = _create_hemisphere_mask(tumor_mask.shape, affine, "left", midline)
        hemi_right = _create_hemisphere_mask(tumor_mask.shape, affine, "right", midline)
        tumor_for_burden = (
            tumor_mask & supra_mask if supra_mask is not None else tumor_mask
        )
        left_tumor = int(np.sum(tumor_for_burden & hemi_left))
        right_tumor = int(np.sum(tumor_for_burden & hemi_right))
        selected_hemisphere = "left" if left_tumor <= right_tumor else "right"
        hemisphere_selection_rule = "lower_supratentorial_tumor_burden"
        hemisphere_mask = _create_hemisphere_mask(
            tumor_mask.shape, affine, selected_hemisphere, midline
        )

    region_mask = hemisphere_mask
    if supra_mask is not None:
        region_mask = supra_mask & hemisphere_mask

    # ---------------------------------------------------------------
    # Sequential order of operations
    # ---------------------------------------------------------------
    qc_steps: dict[str, int] = {}

    # Step 1: Base mask — brain_mask (if provided) or all voxels
    nawm_mask = brain_mask.copy()
    qc_steps["initial_voxels"] = int(np.sum(nawm_mask))

    # Step 2: WM threshold — intersect with wm_data >= wm_threshold
    nawm_mask = nawm_mask & (wm_data >= wm_threshold)
    qc_steps["after_wm_threshold"] = int(np.sum(nawm_mask))

    # Step 3: Region mask — intersect with atlas or hemisphere mask
    nawm_mask = nawm_mask & region_mask
    qc_steps["after_region_mask"] = int(np.sum(nawm_mask))

    # Step 4: Tumor exclusion — subtract dilated tumor
    dilated_tumor = dilate_mask_from_array(tumor_mask, tumor_margin_mm, voxel_sizes)  # type: ignore[arg-type]
    nawm_mask = nawm_mask & ~dilated_tumor
    qc_steps["after_tumor_exclusion"] = int(np.sum(nawm_mask))

    # Step 5: Erosion — single pass, ellipsoidal kernel
    # Convert nawm_erosion (voxel count) to mm: N × min(voxel_sizes)
    min_vox = min(voxel_sizes)
    erosion_mm = nawm_erosion * min_vox
    erosion_voxels_final = nawm_erosion
    erosion_mm_final = erosion_mm

    pre_erosion_mask = nawm_mask.copy()
    if erosion_mm > 0:
        nawm_mask = erode_mask_from_array(nawm_mask, erosion_mm, voxel_sizes)  # type: ignore[arg-type]
    qc_steps["after_erosion"] = int(np.sum(nawm_mask))

    # Step 6: Adaptive fallback — erosion-only reduction (2→1→0)
    n_voxels = int(np.sum(nawm_mask))
    fallback_used = False

    if n_voxels < nawm_min_voxels and nawm_erosion > 0:
        fallback_used = True
        # Try reducing erosion: step down from nawm_erosion-1 to 0
        for reduced_erosion in range(nawm_erosion - 1, -1, -1):
            reduced_mm = reduced_erosion * min_vox
            if reduced_mm > 0:
                nawm_mask = erode_mask_from_array(
                    pre_erosion_mask, reduced_mm, voxel_sizes
                )  # type: ignore[arg-type]
            else:
                nawm_mask = pre_erosion_mask.copy()
            n_voxels = int(np.sum(nawm_mask))
            erosion_voxels_final = reduced_erosion
            erosion_mm_final = reduced_mm
            if n_voxels >= nawm_min_voxels:
                break

    qc_steps["after_adaptive_erosion"] = int(np.sum(nawm_mask))
    n_voxels = int(np.sum(nawm_mask))

    # Quantitative use requires both sufficient size and a verified
    # supratentorial restriction. A hemisphere-only fallback remains useful
    # for visual QC but can include posterior-fossa white matter.
    sufficient_size = n_voxels >= nawm_min_voxels
    valid = sufficient_size and atlas_used
    if not sufficient_size:
        logger.warning(
            "NAWM ROI has only %d voxels (minimum: %d). "
            "Mask saved but marked valid=False. "
            "This may indicate extensive tumor involvement or WM abnormality.",
            n_voxels,
            nawm_min_voxels,
        )
    elif not atlas_used:
        logger.warning(
            "NAWM mask saved for QC but excluded from quantitative use because "
            "the supratentorial atlas restriction was unavailable"
        )

    if not atlas_used:
        exclusion_reason: Optional[str] = "supratentorial_atlas_unavailable"
    elif not sufficient_size:
        exclusion_reason = "insufficient_voxels_after_erosion_relaxation"
    else:
        exclusion_reason = None

    # Calculate final volume
    voxel_volume_mm3 = physical_voxel_volume_mm3(affine)
    volume_ml = n_voxels * voxel_volume_mm3 / 1000.0

    # Determine output path
    if output_file is None:
        if tumor_path.name.endswith(".nii.gz"):
            base = tumor_path.name[:-7]
            ext = ".nii.gz"
        elif tumor_path.name.endswith(".nii"):
            base = tumor_path.stem
            ext = ".nii"
        else:
            base = tumor_path.stem
            ext = tumor_path.suffix
        out_path = tumor_path.parent / f"{base}_nawm{ext}"
    else:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save NAWM mask
    nawm_img = nib.Nifti1Image(  # type: ignore[attr-defined]
        nawm_mask.astype(np.uint8),
        affine,
        header,
    )
    nib.save(nawm_img, out_path)

    return {
        "output_file": str(out_path),
        "analysis": {
            "tumor_laterality": analysis.laterality,
            "compartment_label": analysis.compartment,
            "f_supra": analysis.supratentorial_fraction,
            "f_cereb": analysis.cerebellar_fraction,
            "f_other": analysis.other_fraction,
            "atlas_coverage": analysis.atlas_coverage_fraction,
            "left_fraction": analysis.left_fraction,
            "right_fraction": analysis.right_fraction,
            "laterality_method": analysis.laterality_method,
            "midline_angle_degrees": analysis.midline_angle_degrees,
            "symmetry_dice": analysis.symmetry_dice,
            "selected_hemisphere": selected_hemisphere,
            "hemisphere_selection_rule": hemisphere_selection_rule,
            "nawm_compartment": (
                "supratentorial" if atlas_used else "hemisphere_unrestricted"
            ),
            "warnings": warnings,
            "n_voxels": float(n_voxels),
            "volume_ml": volume_ml,
            "fallback_used": fallback_used,
            "atlas_used": atlas_used,
            "valid": valid,
            "minimum_required_voxels": nawm_min_voxels,
            "quantitative_use_exclusion_reason": exclusion_reason,
            "erosion_voxels_final": erosion_voxels_final,
            "erosion_mm_final": erosion_mm_final,
            "qc_steps": qc_steps,
        },
    }
