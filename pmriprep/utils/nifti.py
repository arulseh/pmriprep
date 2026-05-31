"""NIfTI filename utilities."""

from pathlib import Path
from typing import Union


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
