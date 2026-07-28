"""HD-GLIO tumor mask extraction utilities for PMRIPrep.

This module provides utilities for extracting binary masks from HD-GLIO
multi-class segmentation outputs and creating NAWM ROIs.

Functions
---------
extract_tumor_masks
    Extract binary masks for enhancing, non-enhancing, and whole tumor
create_nawm_roi
    Create contralateral normal-appearing white matter ROI for CBV normalization
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

EPSILON_VOXEL_SIZE = 1e-6      # Guard for zero voxel sizes
MAX_KERNEL_RADIUS_VOXELS = 200  # Memory safety cap for kernel generation


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


@dataclass
class TumorAnalysis:
    """Results from analyzing tumor anatomical location.

    Attributes
    ----------
    is_supratentorial : bool
        True if tumor is supratentorial (above tentorium)
    laterality : Literal["left", "right", "bilateral"]
        Hemisphere containing majority of tumor mass
    centroid_voxels : tuple[float, float, float]
        Center of mass in voxel coordinates
    volume_ml : float
        Tumor volume in milliliters
    centroid_mni : tuple[float, float, float], optional
        Center of mass in MNI coordinates (if MNI transform available)
    """

    is_supratentorial: bool
    laterality: Literal["left", "right", "bilateral"]
    centroid_voxels: tuple[float, float, float]
    volume_ml: float
    centroid_mni: Optional[tuple[float, float, float]] = field(default=None)


def _analyze_tumor_anatomy(
    tumor_mask: npt.NDArray[np.bool_],
    affine: npt.NDArray[np.float64],
    voxel_sizes: tuple[float, float, float],
    t1w_to_mni_xfm: Optional[str] = None,
) -> TumorAnalysis:
    """Analyze tumor location to determine anatomy for NAWM selection.

    Determines whether tumor is supratentorial or infratentorial based on
    centroid position, and which hemisphere contains the majority of tumor
    mass for contralateral NAWM selection.

    Parameters
    ----------
    tumor_mask : ndarray of bool
        3D binary mask of whole tumor
    affine : ndarray
        4x4 affine transformation matrix
    voxel_sizes : tuple of float
        Voxel dimensions (x, y, z) in millimeters
    t1w_to_mni_xfm : str, optional
        Path to T1w-to-MNI transform (FLIRT .mat format). If provided,
        the centroid is transformed to MNI space for supratentorial
        classification using z > -15mm (matching source MNI_TENTORIUM_Z).
        If not provided, falls back to RAS z > -20mm heuristic.

    Returns
    -------
    TumorAnalysis
        Analysis results including location and laterality

    Notes
    -----
    Laterality is determined by comparing tumor volume in each hemisphere.
    The midline is assumed to be at x=0 in RAS coordinates. If tumor spans
    both hemispheres relatively equally (<60% in either), it's marked as
    bilateral.

    When t1w_to_mni_xfm is provided, supratentorial determination uses
    z > -15mm in MNI space (MNI_TENTORIUM_Z from source). Otherwise,
    falls back to z > -20mm in RAS space.
    """
    from scipy import ndimage  # type: ignore[import-untyped]

    # Guard: empty tumor mask
    if not np.any(tumor_mask):
        return TumorAnalysis(
            is_supratentorial=True,
            laterality="bilateral",
            centroid_voxels=(0.0, 0.0, 0.0),
            volume_ml=0.0,
            centroid_mni=None,
        )

    # MNI tentorium z-threshold
    MNI_TENTORIUM_Z = -15.0

    # Get centroid in voxel coordinates
    centroid_voxels = ndimage.center_of_mass(tumor_mask)

    # Convert centroid to physical (RAS) coordinates
    # Affine transforms from voxel to physical space
    centroid_vox_homog = np.array([*centroid_voxels, 1.0])
    centroid_ras = affine @ centroid_vox_homog
    centroid_x, centroid_y, centroid_z = centroid_ras[:3]

    # Determine supratentorial vs infratentorial
    centroid_mni: Optional[tuple[float, float, float]] = None
    if t1w_to_mni_xfm is not None:
        # Transform centroid to MNI space for accurate classification
        mni_coords = _transform_point_to_mni(
            centroid_ras[:3], t1w_to_mni_xfm
        )
        if mni_coords is not None:
            centroid_mni = mni_coords
            is_supratentorial = mni_coords[2] > MNI_TENTORIUM_Z
        else:
            # Transform failed, fall back to RAS heuristic
            is_supratentorial = centroid_z > -20.0
    else:
        # No MNI transform available, use RAS heuristic
        is_supratentorial = centroid_z > -20.0

    # Calculate volume
    voxel_volume_mm3 = float(np.prod(voxel_sizes))
    n_voxels = int(np.sum(tumor_mask))
    volume_ml = n_voxels * voxel_volume_mm3 / 1000.0

    # Determine laterality by comparing volume in each hemisphere
    # Create coordinate grid for x position (RAS)
    shape = tumor_mask.shape
    i_coords = np.arange(shape[0])
    j_coords = np.arange(shape[1])
    k_coords = np.arange(shape[2])
    ii, jj, kk = np.meshgrid(i_coords, j_coords, k_coords, indexing="ij")

    # Transform to physical x coordinate
    # x_physical = affine[0,0]*i + affine[0,1]*j + affine[0,2]*k + affine[0,3]
    x_physical = (
        affine[0, 0] * ii + affine[0, 1] * jj + affine[0, 2] * kk + affine[0, 3]
    )

    # Count tumor voxels in each hemisphere (RAS: +x = right, -x = left)
    left_voxels = int(np.sum(tumor_mask & (x_physical < 0)))
    right_voxels = int(np.sum(tumor_mask & (x_physical > 0)))
    total_voxels = left_voxels + right_voxels

    if total_voxels == 0:
        laterality: Literal["left", "right", "bilateral"] = "bilateral"
    else:
        left_fraction = left_voxels / total_voxels
        right_fraction = right_voxels / total_voxels

        # If >60% in one hemisphere, assign laterality
        if left_fraction > 0.6:
            laterality = "left"
        elif right_fraction > 0.6:
            laterality = "right"
        else:
            laterality = "bilateral"

    return TumorAnalysis(
        is_supratentorial=is_supratentorial,
        laterality=laterality,
        centroid_voxels=tuple(float(c) for c in centroid_voxels),  # type: ignore[arg-type]
        volume_ml=volume_ml,
        centroid_mni=centroid_mni,
    )


def _create_hemisphere_mask(
    shape: tuple[int, int, int],
    affine: npt.NDArray[np.float64],
    hemisphere: Literal["left", "right"],
) -> npt.NDArray[np.bool_]:
    """Create a mask for one hemisphere based on RAS x coordinate.

    Parameters
    ----------
    shape : tuple of int
        Image dimensions (i, j, k)
    affine : ndarray
        4x4 affine transformation matrix
    hemisphere : {"left", "right"}
        Which hemisphere to select (RAS: left = x < 0, right = x > 0)

    Returns
    -------
    ndarray of bool
        Binary mask for the selected hemisphere
    """
    # Create coordinate grid
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


def _transform_point_to_mni(
    point_ras: npt.NDArray[np.float64],
    t1w_to_mni_xfm: str,
) -> Optional[tuple[float, float, float]]:
    """Transform a point from T1w RAS space to MNI space.

    Applies the T1w-to-MNI transform (FLIRT .mat format) to convert
    a point from native T1w space to MNI space.

    Parameters
    ----------
    point_ras : ndarray
        3-element array of RAS coordinates in T1w space
    t1w_to_mni_xfm : str
        Path to T1w-to-MNI FLIRT transform (.mat file)

    Returns
    -------
    tuple of float or None
        (x, y, z) in MNI space, or None if transform fails
    """
    try:
        # Load T1w-to-MNI transform directly
        t1w_to_mni = np.loadtxt(t1w_to_mni_xfm)

        # Apply transform: MNI = T1w_to_MNI @ [x, y, z, 1]
        point_homog = np.array([point_ras[0], point_ras[1], point_ras[2], 1.0])
        mni_coords = t1w_to_mni @ point_homog

        return (float(mni_coords[0]), float(mni_coords[1]), float(mni_coords[2]))

    except (FileNotFoundError, OSError, ValueError):
        return None


def _select_region_atlas(
    analysis: TumorAnalysis,
    region_masks_dir: str,
) -> Optional[str]:
    """Select an MNI anatomical region mask for the tumor location.

    For supratentorial tumors, selects a supratentorial atlas region on the
    contralateral side. For infratentorial tumors, selects a cerebellar atlas.

    Parameters
    ----------
    analysis : TumorAnalysis
        Results from tumor anatomy analysis
    region_masks_dir : str
        Directory containing supratentorial and cerebellar region masks.

    Returns
    -------
    str or None
        Path to selected atlas file, or None if not found

    Notes
    -----
    Atlas selection logic:
    - Supratentorial left tumor  -> Supra_R.nii.gz (right supra for NAWM)
    - Supratentorial right tumor -> Supra_L.nii.gz (left supra for NAWM)
    - Supratentorial bilateral   -> Supra.nii.gz (both hemispheres of region)
    - Infratentorial left tumor  -> Cerebellum_R.nii.gz
    - Infratentorial right tumor -> Cerebellum_L.nii.gz
    - Infratentorial bilateral   -> Cerebellum.nii.gz
    """
    masks_path = Path(region_masks_dir)

    if analysis.is_supratentorial:
        region = "Supra"
    else:
        region = "Cerebellum"

    # Select contralateral side
    if analysis.laterality == "left":
        atlas_name = f"{region}_R.nii.gz"
    elif analysis.laterality == "right":
        atlas_name = f"{region}_L.nii.gz"
    else:
        # Bilateral: use both hemispheres of the region
        atlas_name = f"{region}.nii.gz"

    atlas_path = masks_path / atlas_name
    if atlas_path.exists():
        return str(atlas_path)

    # Fallback: try full-region atlas (no hemisphere suffix)
    if analysis.laterality != "bilateral":
        fallback_name = ("Supra" if analysis.is_supratentorial else "Cerebellum") + ".nii.gz"
        fallback_path = masks_path / fallback_name
        if fallback_path.exists():
            return str(fallback_path)

    return None


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
    """Create contralateral normal-appearing white matter ROI for CBV normalization.

    This function creates an NAWM (Normal Appearing White Matter) ROI for
    use in normalizing DSC CBV maps. The NAWM is selected from the hemisphere
    contralateral to the tumor to avoid contamination by peritumoral edema
    or infiltrating tumor cells.

    When atlas files and MNI transform are provided, NAWM is restricted to the
    anatomically appropriate region (supratentorial or cerebellar) rather than
    just the contralateral hemisphere, preventing cerebellar WM from contaminating
    NAWM estimates for supratentorial tumors.

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
        region restriction and MNI-space tumor classification.
    reference_image : str, optional
        Path to T1w reference image for atlas transformation.
    region_masks_dir : str, optional
        Directory containing MNI supratentorial and cerebellar region masks
        (e.g., Supra_L.nii.gz, Cerebellum_R.nii.gz).
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
            - "tumor_supratentorial": bool
            - "selected_hemisphere": "left" or "right"
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

    Notes
    -----
    The NAWM selection algorithm:

    1. Base mask: brain_mask (if provided) or np.ones()
    2. WM threshold: intersect with wm_data >= 0.9
    3. Region mask: intersect with atlas or hemisphere mask
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
    voxel_sizes = tuple(float(z) for z in zooms)

    # Analyze tumor anatomy (with MNI transform if available)
    analysis = _analyze_tumor_anatomy(
        tumor_mask, affine, voxel_sizes,  # type: ignore[arg-type]
        t1w_to_mni_xfm=t1w_to_mni_xfm,
    )

    # Determine which hemisphere to use for NAWM
    # For unilateral tumors: select contralateral hemisphere
    # For bilateral tumors: select hemisphere with less tumor volume
    if analysis.laterality == "left":
        selected_hemisphere: Literal["left", "right"] = "right"
    elif analysis.laterality == "right":
        selected_hemisphere = "left"
    else:
        # Bilateral: count voxels in each hemisphere and pick the one with less tumor
        hemi_left = _create_hemisphere_mask(tumor_mask.shape, affine, "left")
        hemi_right = _create_hemisphere_mask(tumor_mask.shape, affine, "right")
        left_tumor = int(np.sum(tumor_mask & hemi_left))
        right_tumor = int(np.sum(tumor_mask & hemi_right))
        selected_hemisphere = "left" if left_tumor <= right_tumor else "right"

    # Create hemisphere mask (default region constraint)
    hemisphere_mask = _create_hemisphere_mask(tumor_mask.shape, affine, selected_hemisphere)

    # --- Atlas-based region restriction ---
    atlas_used = False
    region_mask = hemisphere_mask

    if region_masks_dir and t1w_to_mni_xfm and reference_image:
        atlas_path = _select_region_atlas(analysis, region_masks_dir)
        if atlas_path is not None:
            transformed_atlas = _transform_atlas_to_t1w(
                atlas_path, reference_image, t1w_to_mni_xfm
            )
            if transformed_atlas is not None:
                try:
                    atlas_img = nib.load(transformed_atlas)
                    atlas_data = atlas_img.get_fdata()  # type: ignore[attr-defined]
                    atlas_binary = atlas_data > 0.5
                    if atlas_binary.shape == tumor_mask.shape:
                        region_mask = atlas_binary
                        atlas_used = True
                finally:
                    tmp_atlas = Path(transformed_atlas)
                    if tmp_atlas.exists():
                        tmp_atlas.unlink()

    # ---------------------------------------------------------------
    # Sequential order of operations
    # ---------------------------------------------------------------
    qc_steps: dict[str, int] = {}

    # Step 1: Base mask — brain_mask (if provided) or all voxels
    if brain_mask_file is not None:
        brain_img = nib.load(brain_mask_file)
        brain_data = brain_img.get_fdata()  # type: ignore[attr-defined]
        nawm_mask = brain_data > 0.5
    else:
        nawm_mask = np.ones(tumor_mask.shape, dtype=bool)
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
                nawm_mask = erode_mask_from_array(pre_erosion_mask, reduced_mm, voxel_sizes)  # type: ignore[arg-type]
            else:
                nawm_mask = pre_erosion_mask.copy()
            n_voxels = int(np.sum(nawm_mask))
            erosion_voxels_final = reduced_erosion
            erosion_mm_final = reduced_mm
            if n_voxels >= nawm_min_voxels:
                break

    qc_steps["after_adaptive_erosion"] = int(np.sum(nawm_mask))
    n_voxels = int(np.sum(nawm_mask))

    # Determine validity — save mask even if below minimum
    valid = n_voxels >= nawm_min_voxels
    if not valid:
        logger.warning(
            "NAWM ROI has only %d voxels (minimum: %d). "
            "Mask saved but marked valid=False. "
            "This may indicate extensive tumor involvement or WM abnormality.",
            n_voxels,
            nawm_min_voxels,
        )

    # Calculate final volume
    voxel_volume_mm3 = float(np.prod(voxel_sizes))
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
            "tumor_supratentorial": analysis.is_supratentorial,
            "selected_hemisphere": selected_hemisphere,
            "n_voxels": float(n_voxels),
            "volume_ml": volume_ml,
            "fallback_used": fallback_used,
            "atlas_used": atlas_used,
            "valid": valid,
            "erosion_voxels_final": erosion_voxels_final,
            "erosion_mm_final": erosion_mm_final,
            "qc_steps": qc_steps,
        },
    }
