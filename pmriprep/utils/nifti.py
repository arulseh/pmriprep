"""NIfTI filename and orientation utilities."""

from pathlib import Path
from typing import TypeVar, Union

import numpy as np

_ImageT = TypeVar("_ImageT")


def reorient_img_to_ras(img: _ImageT) -> _ImageT:
    """Reorient a NIfTI image to canonical RAS orientation.

    Reorders the data array and affine together so the voxel axes run R, A, S;
    physical (world) positions are preserved. Idempotent: an image already in
    RAS is returned unchanged (no copy). Works on 3D and 4D data (only the first
    three axes are permuted/flipped).

    Parameters
    ----------
    img : nibabel spatial image
        Input image in any orientation.

    Returns
    -------
    nibabel spatial image
        The image reoriented to RAS (same type as the input), or the input image
        itself if it was already RAS.
    """
    from nibabel.orientations import (  # type: ignore[import-untyped]
        axcodes2ornt,
        io_orientation,
        ornt_transform,
    )

    ornt = io_orientation(img.affine)  # type: ignore[attr-defined]
    ras_ornt = axcodes2ornt(("R", "A", "S"))
    if np.array_equal(ornt, ras_ornt):
        return img
    return img.as_reoriented(ornt_transform(ornt, ras_ornt))  # type: ignore[attr-defined, no-any-return]


def nifti_gz_name(filename: Union[str, Path], suffix: str) -> str:
    """Generate a .nii.gz output name from a NIfTI input path.

    Strips .nii.gz or .nii extension, appends the suffix, and always
    returns .nii.gz. This ensures consistency with FSLOUTPUTTYPE=NIFTI_GZ,
    which makes FSL tools write .nii.gz regardless of the specified output.

    Parameters
    ----------
    filename : Path or str
        Input NIfTI file path.
    suffix : str
        Suffix to append (e.g., "_reoriented", "_brain").

    Returns
    -------
    str
        Filename string with suffix and .nii.gz extension.
    """
    name = Path(filename).name
    if name.endswith(".nii.gz"):
        base = name[:-7]
    elif name.endswith(".nii"):
        base = name[:-4]
    else:
        base = Path(filename).stem
    return f"{base}{suffix}.nii.gz"
