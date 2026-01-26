"""NIfTI I/O and metadata utilities for PMRIPrep."""

import json
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def load_nifti(
    path: Union[str, Path],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load a NIfTI file and return data array and affine matrix.

    Args:
        path: Path to the NIfTI file (.nii or .nii.gz).

    Returns:
        Tuple of (data array, affine matrix).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid NIfTI.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    img = nib.load(filepath)
    data: NDArray[np.floating] = np.asarray(
        img.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    affine: NDArray[np.floating] = np.asarray(
        img.affine,  # type: ignore[attr-defined]
        dtype=np.float64,
    )

    return data, affine


def save_nifti(
    data: NDArray[np.floating],
    affine: NDArray[np.floating],
    output_path: Union[str, Path],
    reference: Optional[Union[str, Path]] = None,
    dtype: Optional[np.dtype] = None,
) -> Path:
    """Save data as a NIfTI file with proper header handling.

    If a reference image is provided, its header information (sform, qform,
    data type hints) will be used as a template. Otherwise, a minimal
    header is created.

    Args:
        data: Image data array.
        affine: 4x4 affine transformation matrix.
        output_path: Path for the output NIfTI file.
        reference: Optional path to a reference NIfTI for header template.
        dtype: Optional output data type. If None, uses input data dtype
               or reference header suggestion.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)

    # Determine output dtype
    if dtype is not None:
        out_dtype = dtype
    elif reference is not None:
        ref_img = nib.load(reference)
        out_dtype = ref_img.header.get_data_dtype()  # type: ignore[union-attr,attr-defined]
    else:
        out_dtype = data.dtype

    # Cast data to output dtype
    out_data = np.asarray(data, dtype=out_dtype)

    # Create NIfTI image
    if reference is not None:
        # Use reference header as template
        ref_img = nib.load(reference)
        header = ref_img.header.copy()  # type: ignore[union-attr]
        # Update shape-related fields
        header.set_data_shape(out_data.shape)  # type: ignore[attr-defined]
        header.set_data_dtype(out_dtype)  # type: ignore[attr-defined]
        img = nib.Nifti1Image(out_data, affine, header)
    else:
        # Create new image with minimal header
        img = nib.Nifti1Image(out_data, affine)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    nib.save(img, output_path)

    return output_path


def get_voxel_sizes(path: Union[str, Path]) -> Tuple[float, float, float]:
    """Get voxel dimensions from a NIfTI file header.

    Args:
        path: Path to the NIfTI file.

    Returns:
        Tuple of (voxel_x, voxel_y, voxel_z) sizes in mm.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    img = nib.load(filepath)
    # Get zooms (voxel sizes)
    zooms = img.header.get_zooms()  # type: ignore[union-attr,attr-defined]

    # Return first 3 dimensions (x, y, z)
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def get_tr(
    nifti_path: Union[str, Path],
    json_path: Optional[Union[str, Path]] = None,
) -> Optional[float]:
    """Get repetition time (TR) from NIfTI header or JSON sidecar.

    The JSON sidecar takes precedence if provided and contains
    'RepetitionTime'. Otherwise, the TR from the NIfTI header
    is returned (for 4D images only).

    Args:
        nifti_path: Path to the NIfTI file.
        json_path: Optional path to JSON sidecar. If None, attempts to
                   find sidecar at same location with .json extension.

    Returns:
        TR in seconds, or None if not found.
    """
    nifti_path = Path(nifti_path)

    # Try JSON sidecar first
    if json_path is None:
        # Look for sidecar with same name
        json_path = nifti_path.with_suffix("").with_suffix(".json")
        if nifti_path.name.endswith(".nii.gz"):
            json_path = Path(str(nifti_path)[:-7] + ".json")

    json_path = Path(json_path)

    if json_path.exists():
        with open(json_path) as f:
            sidecar = json.load(f)
        if "RepetitionTime" in sidecar:
            return float(sidecar["RepetitionTime"])

    # Fall back to NIfTI header
    if not nifti_path.exists():
        return None

    img = nib.load(nifti_path)
    header = img.header  # type: ignore[union-attr]

    # Check if this is a 4D image
    shape = header.get_data_shape()  # type: ignore[attr-defined]
    if len(shape) < 4:
        return None

    # Get zooms - 4th dimension is TR for 4D images
    zooms = header.get_zooms()  # type: ignore[attr-defined]
    if len(zooms) >= 4:
        tr = float(zooms[3])
        # Sanity check: TR should be positive and reasonable (0.1-30 seconds)
        if 0.1 <= tr <= 30.0:
            return tr

    # Check xyzt_units for time unit
    xyzt_units = header.get_xyzt_units()  # type: ignore[attr-defined]
    if len(xyzt_units) >= 2:
        time_unit = xyzt_units[1]
        # If time unit is 'sec' or 'msec', use pixdim[4]
        pixdim = header["pixdim"]  # type: ignore[index]
        if len(pixdim) >= 5:
            tr = float(pixdim[4])
            if time_unit == "msec":
                tr = tr / 1000.0
            if 0.1 <= tr <= 30.0:
                return tr

    return None


def check_dimensions_match(
    path1: Union[str, Path],
    path2: Union[str, Path],
    check_affine: bool = True,
    tolerance: float = 1e-5,
) -> bool:
    """Check if two NIfTI images have matching dimensions.

    Args:
        path1: Path to the first NIfTI file.
        path2: Path to the second NIfTI file.
        check_affine: Whether to also check affine matrices match.
        tolerance: Tolerance for floating-point comparisons.

    Returns:
        True if dimensions (and optionally affines) match.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    path1 = Path(path1)
    path2 = Path(path2)

    if not path1.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path1}")
    if not path2.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path2}")

    img1 = nib.load(path1)
    img2 = nib.load(path2)

    # Check shape (first 3 dimensions for spatial matching)
    shape1 = img1.header.get_data_shape()[:3]  # type: ignore[union-attr,attr-defined]
    shape2 = img2.header.get_data_shape()[:3]  # type: ignore[union-attr,attr-defined]

    if shape1 != shape2:
        return False

    if check_affine:
        affine1 = np.asarray(img1.affine)  # type: ignore[attr-defined]
        affine2 = np.asarray(img2.affine)  # type: ignore[attr-defined]
        if not np.allclose(affine1, affine2, atol=tolerance):
            return False

    return True


def resample_to_target(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    output_path: Union[str, Path],
    interpolation: Literal["nearest", "linear", "cubic"] = "linear",
) -> Path:
    """Resample a source image to match a target image's geometry.

    Uses nilearn for resampling with specified interpolation method.

    Args:
        source_path: Path to the source NIfTI file to resample.
        target_path: Path to the target NIfTI file defining output geometry.
        output_path: Path for the resampled output file.
        interpolation: Interpolation method - 'nearest' for masks/labels,
                       'linear' for continuous data, 'cubic' for high-quality.

    Returns:
        Path to the resampled output file.

    Raises:
        FileNotFoundError: If source or target file does not exist.
        ValueError: If interpolation method is not recognized.
    """
    # Import nilearn here to avoid import overhead when not needed
    from nilearn.image import resample_to_img

    source_path = Path(source_path)
    target_path = Path(target_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source NIfTI file not found: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target NIfTI file not found: {target_path}")

    # Map interpolation to nilearn order
    interp_map = {
        "nearest": "nearest",
        "linear": "linear",
        "cubic": "linear",  # nilearn uses 'continuous' for spline, but linear is safer
    }

    if interpolation not in interp_map:
        raise ValueError(
            f"Interpolation must be 'nearest', 'linear', or 'cubic', got: {interpolation}"
        )

    source_img = nib.load(source_path)
    target_img = nib.load(target_path)

    # Resample
    resampled_img = resample_to_img(
        source_img,
        target_img,
        interpolation=interp_map[interpolation],
    )

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    nib.save(resampled_img, output_path)

    return output_path
