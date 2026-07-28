"""Serializable helpers shared by DSC and DCE workflows."""


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
    """Validate that a mask matches its native perfusion image grid.

    Parameters
    ----------
    mask_file : str or None
        Path to the mask NIfTI.
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
        If the mask shape or affine does not match the reference grid.
    """
    from nipype.interfaces.base import isdefined

    if not isdefined(mask_file) or mask_file in (None, ""):
        return ""

    import nibabel as nib
    import numpy as np

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
