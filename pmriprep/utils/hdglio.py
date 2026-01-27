"""HD-GLIO tumor mask extraction and dilation utilities for PMRIPrep.

This module provides utilities for extracting binary masks from HD-GLIO
multi-class segmentation outputs and applying morphological operations
like dilation with proper handling of voxel anisotropy.

Functions
---------
extract_tumor_masks
    Extract binary masks for enhancing, non-enhancing, and whole tumor
dilate_mask
    Dilate a binary mask using a spherical kernel with anisotropy support
"""

from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt


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
        Prefix for output filenames (e.g., "sub-001_ses-01_")

    Returns
    -------
    dict[str, str]
        Dictionary mapping mask type to output file path:
        - "enhancing": Path to enhancing tumor mask (label 2)
        - "non_enhancing": Path to non-enhancing tumor mask (label 1)
        - "whole_tumor": Path to whole tumor mask (labels 1 + 2)

    Examples
    --------
    >>> masks = extract_tumor_masks("hdglio_seg.nii.gz", output_dir="/out")
    >>> masks["enhancing"]
    '/out/desc-enhancing_mask.nii.gz'
    >>> masks["whole_tumor"]
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
    vx, vy, vz = voxel_sizes

    # Calculate kernel size in voxels for each dimension
    # Need at least radius_mm / voxel_size voxels on each side of center
    nx = int(np.ceil(radius_mm / vx))
    ny = int(np.ceil(radius_mm / vy))
    nz = int(np.ceil(radius_mm / vz))

    # Total kernel size (diameter + 1 for center)
    size_x = 2 * nx + 1
    size_y = 2 * ny + 1
    size_z = 2 * nz + 1

    # Create coordinate grids centered at the kernel center
    # Coordinates are in millimeters
    x = np.arange(size_x) - nx
    y = np.arange(size_y) - ny
    z = np.arange(size_z) - nz

    # Scale by voxel sizes to get physical coordinates
    xx, yy, zz = np.meshgrid(x * vx, y * vy, z * vz, indexing="ij")

    # Calculate distance from center in mm
    distance = np.sqrt(xx**2 + yy**2 + zz**2)

    # Create spherical kernel
    kernel: npt.NDArray[np.bool_] = distance <= radius_mm

    return kernel


def dilate_mask(
    mask_file: Union[str, Path],
    radius_mm: float,
    output_file: Union[str, Path, None] = None,
) -> str:
    """Dilate a binary mask using a spherical structuring element.

    Applies morphological dilation to a binary mask using a sphere of
    the specified radius in millimeters. The dilation properly accounts
    for anisotropic voxel dimensions by creating an appropriately shaped
    structuring element.

    Parameters
    ----------
    mask_file : str or Path
        Path to input binary mask NIfTI file
    radius_mm : float
        Dilation radius in millimeters. This is the radius of the
        spherical structuring element used for dilation.
    output_file : str or Path, optional
        Path to output dilated mask. If None, generates filename by
        appending "_dilated" to input filename.

    Returns
    -------
    str
        Path to the output dilated mask file

    Examples
    --------
    >>> dilated = dilate_mask("tumor_mask.nii.gz", radius_mm=15.0)
    >>> dilated
    'tumor_mask_dilated.nii.gz'

    >>> dilated = dilate_mask("mask.nii.gz", 15.0, "mask_15mm.nii.gz")
    >>> dilated
    'mask_15mm.nii.gz'

    Notes
    -----
    The dilation is performed using scipy.ndimage.binary_dilation with
    a spherical structuring element. For anisotropic voxels, the kernel
    shape is adjusted to maintain a spherical dilation in physical space.

    The dilation radius in HD-GLIO workflows is typically 15mm (default
    from HDGLIOConfig.lesion_dilation), which creates a safety margin
    around the tumor for registration cost function weighting.

    See Also
    --------
    _create_spherical_kernel : Create the spherical structuring element
    scipy.ndimage.binary_dilation : The underlying dilation operation
    """
    import nibabel as nib  # type: ignore[import-untyped]
    from scipy import ndimage  # type: ignore[import-untyped]

    # Load mask
    mask_path = Path(mask_file)
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]
    affine = mask_img.affine  # type: ignore[attr-defined]
    header = mask_img.header  # type: ignore[attr-defined]

    # Get voxel sizes from header
    zooms = header.get_zooms()[:3]  # type: ignore[attr-defined]
    voxel_sizes = tuple(float(z) for z in zooms)

    # Create spherical kernel respecting anisotropy
    kernel = _create_spherical_kernel(radius_mm, voxel_sizes)  # type: ignore[arg-type]

    # Binarize mask (in case it has non-binary values)
    binary_mask = mask_data > 0.5

    # Apply dilation
    dilated_mask = ndimage.binary_dilation(binary_mask, structure=kernel)

    # Determine output path
    if output_file is None:
        # Generate output filename
        if mask_path.name.endswith(".nii.gz"):
            base = mask_path.name[:-7]
            ext = ".nii.gz"
        elif mask_path.name.endswith(".nii"):
            base = mask_path.stem
            ext = ".nii"
        else:
            base = mask_path.stem
            ext = mask_path.suffix

        out_path = mask_path.parent / f"{base}_dilated{ext}"
    else:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dilated mask as uint8
    dilated_img = nib.Nifti1Image(  # type: ignore[attr-defined]
        dilated_mask.astype(np.uint8),
        affine,
        header,
    )
    nib.save(dilated_img, out_path)

    return str(out_path)


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
    >>> dilated.sum() > mask.sum()
    True

    See Also
    --------
    dilate_mask : File-based version with NIfTI I/O
    """
    from scipy import ndimage  # type: ignore[import-untyped]

    # Create spherical kernel
    kernel = _create_spherical_kernel(radius_mm, voxel_sizes)

    # Apply dilation
    dilated: npt.NDArray[np.bool_] = ndimage.binary_dilation(mask_data, structure=kernel)

    return dilated


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
    >>> volume = get_mask_volume_ml("tumor_mask.nii.gz")
    >>> print(f"Tumor volume: {volume:.1f} mL")
    Tumor volume: 42.5 mL

    Notes
    -----
    Volume is calculated as:
        volume_ml = n_voxels * voxel_volume_mm3 / 1000
    """
    import nibabel as nib  # type: ignore[import-untyped]

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]
    zooms = mask_img.header.get_zooms()[:3]  # type: ignore[attr-defined]

    # Calculate voxel volume in mm³
    voxel_volume_mm3 = float(np.prod(zooms))

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
    >>> stats = get_mask_statistics("tumor_mask.nii.gz")
    >>> print(f"Volume: {stats['volume_ml']:.1f} mL")
    >>> print(f"Centroid: {stats['centroid_voxels']}")
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
    voxel_volume_mm3 = float(np.prod(zooms))
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
