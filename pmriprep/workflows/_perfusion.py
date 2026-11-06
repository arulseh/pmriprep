"""Serializable helpers shared by DSC and DCE workflows."""


def fsl_to_x5_node(mat_path: str, reference: str, moving: str) -> str:
    """Convert an FSL affine to an ITK X5 transform in a Nipype worker."""
    from pmriprep.utils.transforms import fsl_to_x5

    return fsl_to_x5(mat_path, reference, moving)


def smooth_masked_maps(
    map_files: dict[str, str],
    brain_mask_file: str | None,
    fwhm: float,
) -> dict[str, str]:
    """Smooth quantitative maps over their finite in-brain support.

    Parameters
    ----------
    map_files : dict
        Mapping from map name to NIfTI path.
    brain_mask_file : str or None
        Brain-mask NIfTI on the same grid as every input map.
    fwhm : float
        Gaussian full-width at half-maximum in millimetres.

    Returns
    -------
    dict
        Mapping from each available input name to its smoothed NIfTI path.

    Raises
    ------
    ValueError
        If a processed map and brain mask have different shapes or affines.

    Notes
    -----
    Normalized convolution uses only finite source voxels inside the brain.
    Undefined source voxels remain undefined and outside-brain voxels are zero.
    """
    import logging
    import subprocess
    from pathlib import Path
    from typing import cast

    import nibabel as nib
    import numpy as np
    from nipype.interfaces.base import isdefined

    logger = logging.getLogger("nipype.workflow")
    results: dict[str, str] = {}
    if not map_files:
        logger.warning("Smoothing requested but no quantitative map files are available")
        return results
    if not brain_mask_file or not isdefined(brain_mask_file):
        logger.warning("Smoothing skipped because the brain mask is unavailable")
        return results
    if not Path(brain_mask_file).is_file():
        logger.warning("Smoothing skipped because the brain mask file does not exist")
        return results

    sigma = fwhm / 2.355
    for name, filepath in map_files.items():
        if not filepath or not isdefined(filepath):
            continue
        if isinstance(filepath, str) and filepath in ("", "None"):
            continue
        if not Path(filepath).is_file():
            continue

        out_path = str(Path.cwd() / f"{Path(filepath).stem}_fwhm{int(fwhm)}.nii.gz")
        img = cast(nib.Nifti1Image, nib.load(filepath))
        data = img.get_fdata()
        mask_img = cast(nib.Nifti1Image, nib.load(brain_mask_file))
        brain_mask = mask_img.get_fdata() > 0
        if brain_mask.shape != data.shape:
            raise ValueError(
                f"Brain mask shape {brain_mask.shape} does not match {name} map {data.shape}"
            )
        if not np.allclose(mask_img.affine, img.affine):
            raise ValueError(f"Brain mask affine does not match {name} map affine")

        source_support = np.isfinite(data) & brain_mask
        valid_mask = source_support.astype(np.float32)
        numerator = np.where(source_support, data, 0.0).astype(np.float32)
        vmask_path = str(Path.cwd() / f"{name}_vmask.nii.gz")
        num_path = str(Path.cwd() / f"{name}_num.nii.gz")
        den_path = str(Path.cwd() / f"{name}_den.nii.gz")
        smooth_num_path = str(Path.cwd() / f"{name}_smooth_num.nii.gz")
        nib.save(nib.Nifti1Image(valid_mask, img.affine, img.header), vmask_path)
        nib.save(nib.Nifti1Image(numerator, img.affine, img.header), num_path)

        logger.info("Brain-masked normalized convolution for '%s'", name)
        try:
            subprocess.run(
                ["fslmaths", num_path, "-s", str(sigma), smooth_num_path],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["fslmaths", vmask_path, "-s", str(sigma), den_path],
                check=True,
                capture_output=True,
            )
            smooth_num = cast(nib.Nifti1Image, nib.load(smooth_num_path)).get_fdata()
            smooth_den = cast(nib.Nifti1Image, nib.load(den_path)).get_fdata()
            smoothed = np.full(data.shape, np.nan, dtype=np.float32)
            calculable = source_support & (smooth_den > np.finfo(np.float32).eps)
            smoothed[calculable] = (smooth_num[calculable] / smooth_den[calculable]).astype(
                np.float32
            )
            smoothed[~brain_mask] = 0.0
            out_header = nib.Nifti1Header.from_header(img.header)
            out_header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(smoothed, img.affine, out_header), out_path)
        finally:
            for temporary in (vmask_path, num_path, den_path, smooth_num_path):
                Path(temporary).unlink(missing_ok=True)

        results[name] = out_path
    return results


def select_aif_exclusion_mask(
    user_lesion_mask: str | None = None,
    transformed_lesion_mask: str | None = None,
) -> str:
    """Select the lesion mask used to exclude AIF candidates.

    Parameters
    ----------
    user_lesion_mask : str or None
        User-supplied lesion mask in native perfusion space.
    transformed_lesion_mask : str or None
        Tumor-segmentation mask transformed to native perfusion space.

    Returns
    -------
    str
        The user mask when present, otherwise the transformed mask or ``""``.
    """
    from nipype.interfaces.base import isdefined

    if isdefined(user_lesion_mask) and user_lesion_mask not in (None, ""):
        return str(user_lesion_mask)
    if (
        isdefined(transformed_lesion_mask)
        and transformed_lesion_mask not in (None, "")
    ):
        return str(transformed_lesion_mask)
    return ""


def validate_native_mask(
    mask_file: str | None = None,
    reference_file: str | None = None,
    mask_label: str = "mask",
) -> str:
    """Validate a 3-D mask against its native perfusion image grid.

    Parameters
    ----------
    mask_file : str or None
        Path to a three-dimensional mask NIfTI.
    reference_file : str or None
        Path to the perfusion reference image.
    mask_label : str
        Human-readable label used in validation errors.

    Returns
    -------
    str
        The validated mask path, or ``""`` when no mask is supplied.

    Raises
    ------
    ValueError
        If the mask is not 3-D, or its shape or affine does not match the
        reference grid.
    """
    from nipype.interfaces.base import isdefined

    if not isdefined(mask_file) or mask_file in (None, ""):
        return ""

    import nibabel as nib
    import numpy as np
    from pmriprep.utils.lesion_stats import validate_mask_ndim

    validate_mask_ndim(str(mask_file), source_name=mask_label)
    mask_img = nib.load(str(mask_file))
    ref_img = nib.load(str(reference_file))
    if mask_img.shape[:3] != ref_img.shape[:3]:  # type: ignore[attr-defined]
        raise ValueError(
            f"User-supplied {mask_label} shape {mask_img.shape[:3]} does not "  # type: ignore[attr-defined]
            f"match perfusion volume shape {ref_img.shape[:3]}. "  # type: ignore[attr-defined]
            "Masks under config must be in perfusion native space."
        )
    affine_diff = float(
        np.abs(mask_img.affine - ref_img.affine).max()  # type: ignore[attr-defined]
    )
    if affine_diff > 1.0:
        raise ValueError(
            f"User-supplied {mask_label} affine differs from perfusion volume "
            f"by {affine_diff:.2f} (max element). "
            "Masks under config must be in perfusion native space."
        )
    return str(mask_file)
