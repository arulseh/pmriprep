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
create_nawm_roi
    Create contralateral normal-appearing white matter ROI for CBV normalization
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

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
    mni_to_t1w_xfm: Optional[str] = None,
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
    mni_to_t1w_xfm : str, optional
        Path to MNI-to-T1w transform (FLIRT .mat format). If provided,
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

    When mni_to_t1w_xfm is provided, supratentorial determination uses
    z > -15mm in MNI space (MNI_TENTORIUM_Z from source). Otherwise,
    falls back to z > -20mm in RAS space.
    """
    from scipy import ndimage  # type: ignore[import-untyped]

    # MNI tentorium z-threshold (matches source perf-plus)
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
    if mni_to_t1w_xfm is not None:
        # Transform centroid to MNI space for accurate classification
        mni_coords = _transform_point_to_mni(
            centroid_ras[:3], mni_to_t1w_xfm
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

    # Count tumor voxels in each hemisphere (x < 0 is right, x > 0 is left in RAS)
    left_voxels = int(np.sum(tumor_mask & (x_physical > 0)))
    right_voxels = int(np.sum(tumor_mask & (x_physical < 0)))
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
        Which hemisphere to select (left = x > 0, right = x < 0 in RAS)

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

    # In RAS: left hemisphere has x > 0, right hemisphere has x < 0
    if hemisphere == "left":
        mask: npt.NDArray[np.bool_] = x_physical > 0
    else:
        mask = x_physical < 0

    return mask


def _transform_point_to_mni(
    point_ras: npt.NDArray[np.float64],
    mni_to_t1w_xfm: str,
) -> Optional[tuple[float, float, float]]:
    """Transform a point from T1w RAS space to MNI space.

    Uses the inverse of the MNI-to-T1w transform (FLIRT .mat format)
    to convert a point from native T1w space to MNI space.

    Parameters
    ----------
    point_ras : ndarray
        3-element array of RAS coordinates in T1w space
    mni_to_t1w_xfm : str
        Path to MNI-to-T1w FLIRT transform (.mat file)

    Returns
    -------
    tuple of float or None
        (x, y, z) in MNI space, or None if transform fails
    """
    import subprocess
    import tempfile

    try:
        # Invert the MNI-to-T1w transform to get T1w-to-MNI
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            t1w_to_mni_path = tmp.name

        subprocess.run(
            [
                "convert_xfm",
                "-omat", t1w_to_mni_path,
                "-inverse", mni_to_t1w_xfm,
            ],
            check=True,
            capture_output=True,
        )

        # Load the inverted transform
        t1w_to_mni = np.loadtxt(t1w_to_mni_path)

        # Apply transform: MNI = T1w_to_MNI @ [x, y, z, 1]
        point_homog = np.array([point_ras[0], point_ras[1], point_ras[2], 1.0])
        mni_coords = t1w_to_mni @ point_homog

        return (float(mni_coords[0]), float(mni_coords[1]), float(mni_coords[2]))

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    finally:
        # Clean up temp file
        tmp_path = Path(t1w_to_mni_path)  # type: ignore[possibly-undefined]
        if tmp_path.exists():
            tmp_path.unlink()


def _select_region_atlas(
    analysis: TumorAnalysis,
    atlases_dir: str,
) -> Optional[str]:
    """Select MNI lobar atlas file based on tumor location and laterality.

    For supratentorial tumors, selects a supratentorial atlas region on the
    contralateral side. For infratentorial tumors, selects a cerebellar atlas.

    Parameters
    ----------
    analysis : TumorAnalysis
        Results from tumor anatomy analysis
    atlases_dir : str
        Path to directory containing MNI lobar atlas NIfTI files

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
    atlases_path = Path(atlases_dir)

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

    atlas_path = atlases_path / atlas_name
    if atlas_path.exists():
        return str(atlas_path)

    return None


def _transform_atlas_to_t1w(
    atlas_path: str,
    reference_path: str,
    mni_to_t1w_xfm: str,
) -> Optional[str]:
    """Transform MNI atlas to T1w space via FLIRT.

    Parameters
    ----------
    atlas_path : str
        Path to MNI-space atlas NIfTI file
    reference_path : str
        Path to T1w reference image (for output geometry)
    mni_to_t1w_xfm : str
        Path to MNI-to-T1w FLIRT transform (.mat file)

    Returns
    -------
    str or None
        Path to transformed atlas in T1w space, or None if transform fails
    """
    import subprocess
    import tempfile

    try:
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
                "-init", mni_to_t1w_xfm,
                "-interp", "nearestneighbour",
                "-out", out_path,
            ],
            check=True,
            capture_output=True,
        )

        return out_path

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def create_nawm_roi(
    tumor_mask_file: Union[str, Path],
    wm_probseg_file: Union[str, Path],
    output_file: Union[str, Path, None] = None,
    wm_threshold: float = 0.9,
    tumor_margin_mm: float = 20.0,
    nawm_erosion: int = 2,
    nawm_min_voxels: int = 500,
    mni_to_t1w_xfm: Optional[str] = None,
    reference_image: Optional[str] = None,
    atlases_dir: Optional[str] = None,
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
        Number of erosion iterations to clean up partial volume effects
        (from HDGLIOConfig.nawm_erosion)
    nawm_min_voxels : int, default=500
        Minimum required voxels in final NAWM ROI
        (from HDGLIOConfig.nawm_min_voxels)
    mni_to_t1w_xfm : str, optional
        Path to MNI-to-T1w transform (.mat). Required for atlas-based
        region restriction and MNI-space tumor classification.
    reference_image : str, optional
        Path to T1w reference image for atlas transformation.
    atlases_dir : str, optional
        Path to directory containing MNI lobar atlas NIfTI files
        (e.g., Supra_L.nii.gz, Cerebellum_R.nii.gz).

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

    Raises
    ------
    ValueError
        If no suitable NAWM region can be found even with adaptive fallback

    Examples
    --------
    >>> result = create_nawm_roi(
    ...     "tumor_mask.nii.gz",
    ...     "wm_probseg.nii.gz",
    ...     nawm_erosion=2,
    ...     nawm_min_voxels=500
    ... )
    >>> print(f"NAWM voxels: {result['analysis']['n_voxels']}")
    NAWM voxels: 2847

    Notes
    -----
    The NAWM selection algorithm:

    1. Analyze tumor anatomy to determine laterality (left/right/bilateral)
    2. Select contralateral hemisphere:
       - For unilateral tumors: opposite hemisphere
       - For bilateral tumors: select hemisphere with less tumor volume
    3. If atlas restriction is available:
       - Transform MNI lobar atlas to T1w space
       - Replace hemisphere mask with region-specific atlas mask
    4. Apply WM restriction: probseg >= wm_threshold (default 0.9)
    5. Exclude voxels within tumor_margin_mm (default 20mm) of tumor
    6. Apply erosion (nawm_erosion iterations) to clean up partial volume
    7. If insufficient voxels, apply adaptive fallback:
       - Reduce WM threshold to 0.7
       - Reduce erosion to 1 iteration
       - Reduce tumor margin to 15mm
       - If still insufficient, include ipsilateral hemisphere with
         increased tumor margin (30mm)

    References
    ----------
    Prah MA, Schmainda KM. rCBV estimation in tumors: comparison of methods
    for normalization to normal appearing white matter. J Magn Reson Imaging.
    2024.
    """
    import nibabel as nib  # type: ignore[import-untyped]
    from scipy import ndimage  # type: ignore[import-untyped]

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
        mni_to_t1w_xfm=mni_to_t1w_xfm,
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
    # If atlas files and transform are available, replace hemisphere mask
    # with a region-specific atlas mask for more accurate NAWM selection
    atlas_used = False
    region_mask = hemisphere_mask

    if atlases_dir and mni_to_t1w_xfm and reference_image:
        atlas_path = _select_region_atlas(analysis, atlases_dir)
        if atlas_path is not None:
            transformed_atlas = _transform_atlas_to_t1w(
                atlas_path, reference_image, mni_to_t1w_xfm
            )
            if transformed_atlas is not None:
                try:
                    atlas_img = nib.load(transformed_atlas)
                    atlas_data = atlas_img.get_fdata()  # type: ignore[attr-defined]
                    # Binarize (nearest-neighbour should give 0/1, but guard)
                    atlas_binary = atlas_data > 0.5
                    if atlas_binary.shape == tumor_mask.shape:
                        region_mask = atlas_binary
                        atlas_used = True
                finally:
                    # Clean up temporary atlas file
                    tmp_atlas = Path(transformed_atlas)
                    if tmp_atlas.exists():
                        tmp_atlas.unlink()

    # Create dilated tumor exclusion mask
    # Dilate tumor by tumor_margin_mm to create exclusion zone
    dilated_tumor = dilate_mask_from_array(tumor_mask, tumor_margin_mm, voxel_sizes)  # type: ignore[arg-type]

    def _compute_nawm(
        wm_thresh: float,
        n_erosion: int,
        margin_mask: npt.NDArray[np.bool_],
        hemi_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.bool_]:
        """Compute NAWM mask with given parameters."""
        # WM probability threshold
        wm_mask = wm_data >= wm_thresh

        # Combine constraints
        nawm = wm_mask & hemi_mask & ~margin_mask

        # Apply erosion to clean up partial volume effects
        if n_erosion > 0:
            # Create spherical erosion kernel (1 voxel radius)
            erosion_kernel = _create_spherical_kernel(
                min(voxel_sizes), voxel_sizes  # type: ignore[arg-type]
            )
            for _ in range(n_erosion):
                nawm = ndimage.binary_erosion(nawm, structure=erosion_kernel)

        return nawm  # type: ignore[return-value]

    # Try standard parameters first (using region_mask which may be atlas-based)
    nawm_mask = _compute_nawm(wm_threshold, nawm_erosion, dilated_tumor, region_mask)
    n_voxels = int(np.sum(nawm_mask))
    fallback_used = False

    # Adaptive fallback if insufficient voxels
    if n_voxels < nawm_min_voxels:
        fallback_used = True

        # Fallback 1: Reduce WM threshold and erosion
        nawm_mask = _compute_nawm(0.7, 1, dilated_tumor, region_mask)
        n_voxels = int(np.sum(nawm_mask))

        if n_voxels < nawm_min_voxels:
            # Fallback 2: Reduce tumor margin
            dilated_tumor_reduced = dilate_mask_from_array(
                tumor_mask, 15.0, voxel_sizes  # type: ignore[arg-type]
            )
            nawm_mask = _compute_nawm(0.7, 1, dilated_tumor_reduced, region_mask)
            n_voxels = int(np.sum(nawm_mask))

        if n_voxels < nawm_min_voxels and atlas_used:
            # Fallback 2b: Drop atlas restriction, use hemisphere mask
            nawm_mask = _compute_nawm(0.7, 1, dilated_tumor, hemisphere_mask)
            n_voxels = int(np.sum(nawm_mask))
            if n_voxels >= nawm_min_voxels:
                atlas_used = False

        if n_voxels < nawm_min_voxels:
            # Fallback 3: Include both hemispheres with larger margin
            dilated_tumor_large = dilate_mask_from_array(
                tumor_mask, 30.0, voxel_sizes  # type: ignore[arg-type]
            )
            # Full brain (both hemispheres)
            full_brain_mask = np.ones(tumor_mask.shape, dtype=bool)
            nawm_mask = _compute_nawm(0.7, 1, dilated_tumor_large, full_brain_mask)
            n_voxels = int(np.sum(nawm_mask))

        if n_voxels < nawm_min_voxels:
            # Final fallback: no erosion, minimal WM threshold
            nawm_mask = _compute_nawm(0.5, 0, dilated_tumor_large, full_brain_mask)  # type: ignore[possibly-undefined]
            n_voxels = int(np.sum(nawm_mask))

        if n_voxels < nawm_min_voxels:
            raise ValueError(
                f"Unable to create NAWM ROI with at least {nawm_min_voxels} voxels. "
                f"Found only {n_voxels} voxels after all fallback attempts. "
                "This may indicate extensive tumor involvement or WM abnormality."
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
        },
    }
