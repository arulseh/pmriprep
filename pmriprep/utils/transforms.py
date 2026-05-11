"""Spatial transform format conversion utilities."""

from pathlib import Path
from typing import Union

import numpy as np


def fsl_to_x5(
    mat_path: Union[str, Path],
    reference: Union[str, Path],
    moving: Union[str, Path],
) -> str:
    """Convert an FSL FLIRT .mat to an ITK composite .h5 transform.

    Parameters
    ----------
    mat_path : str or Path
        Path to an FSL FLIRT 4x4 affine matrix file.
    reference : str or Path
        Path to the reference (fixed) NIfTI image used for registration.
    moving : str or Path
        Path to the moving NIfTI image used for registration.

    Returns
    -------
    str
        Path to the output .h5 file (same stem as input, .h5 extension),
        or ``""`` if *mat_path* is a sentinel (empty/None).
    """
    if not mat_path:
        return ""

    import h5py  # type: ignore[import-untyped]
    from nitransforms.io.itk import ITKLinearTransform  # type: ignore[import-untyped]
    from nitransforms.linear import Affine  # type: ignore[import-untyped]

    mat_path = Path(mat_path)
    h5_path = mat_path.with_suffix(".h5")

    xfm = Affine.from_filename(
        str(mat_path), fmt="fsl", reference=str(reference), moving=str(moving),
    )

    itk_xfm = ITKLinearTransform.from_ras(xfm.matrix)
    params_4x4 = np.asarray(itk_xfm["parameters"], dtype=np.float64)
    offset = np.asarray(itk_xfm["offset"], dtype=np.float64)
    itk_params = np.concatenate([params_4x4[:3, :3].ravel(), params_4x4[:3, 3]])

    identity_params = np.array(
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64,
    )
    with h5py.File(str(h5_path), "w") as f:
        grp = f.create_group("TransformGroup")
        g0 = grp.create_group("0")
        g0.create_dataset(
            "TransformType",
            data=np.array([b"AffineTransform_double_3_3"]),
        )
        g0.create_dataset("TransformParameters", data=itk_params)
        g0.create_dataset("TransformFixedParameters", data=offset)
        g1 = grp.create_group("1")
        g1.create_dataset(
            "TransformType",
            data=np.array([b"AffineTransform_double_3_3"]),
        )
        g1.create_dataset("TransformParameters", data=identity_params)
        g1.create_dataset("TransformFixedParameters", data=np.zeros(3, dtype=np.float64))

    return str(h5_path)
