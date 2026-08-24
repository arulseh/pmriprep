"""Descriptive tumor-relative laterality for detected AIF candidates."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt

TumorLaterality = Literal["left", "right", "bilateral"]


def describe_tumor_relative_aif_candidates(
    candidates: Sequence[tuple[int, int, int]],
    affine: npt.NDArray[Any],
    lesion_mask: npt.NDArray[Any] | None,
    *,
    enhancing_mask: npt.NDArray[Any] | None = None,
    midline_tolerance_mm: float = 10.0,
) -> dict[str, str | int]:
    """Describe candidate laterality relative to a lesion mask.

    Parameters
    ----------
    candidates : sequence of tuple of int
        Final AIF candidate voxel coordinates.
    affine : ndarray
        Voxel-to-RAS affine for the candidates and masks.
    lesion_mask : ndarray or None
        Lesion mask used when no non-empty enhancing mask is available.
    enhancing_mask : ndarray or None, optional
        Preferred mask for tumor laterality classification.
    midline_tolerance_mm : float, optional
        Minimum bilateral extent on both sides of the RAS midline.

    Returns
    -------
    dict
        Descriptive QC fields. Contralateral and ipsilateral counts are
        included only for unilateral tumors, where those terms are defined.

    Notes
    -----
    These fields do not alter AIF candidate scores, selection, or acceptance.
    """
    laterality_mask: npt.NDArray[Any] | None = None
    mask_source: str | None = None
    if enhancing_mask is not None and np.any(enhancing_mask > 0.5):
        laterality_mask = enhancing_mask
        mask_source = "enhancing"
    elif lesion_mask is not None and np.any(lesion_mask > 0.5):
        laterality_mask = lesion_mask
        mask_source = "lesion"

    if laterality_mask is None or mask_source is None:
        return {}

    lesion_coords = np.argwhere(laterality_mask > 0.5)
    homogeneous = np.column_stack(
        [lesion_coords, np.ones(len(lesion_coords), dtype=float)]
    )
    world_x = (affine @ homogeneous.T).T[:, 0]
    center_x = float(np.mean(world_x))
    laterality: TumorLaterality
    if (
        float(np.min(world_x)) < -midline_tolerance_mm
        and float(np.max(world_x)) > midline_tolerance_mm
    ):
        laterality = "bilateral"
    else:
        laterality = "left" if center_x < 0 else "right"

    result: dict[str, str | int] = {
        "tumor_laterality": laterality,
        "laterality_mask_source": mask_source,
    }
    if laterality == "bilateral":
        return result

    tumor_is_left = laterality == "left"
    n_contralateral = 0
    n_ipsilateral = 0
    for coordinate in candidates:
        world = affine @ np.asarray([*coordinate, 1.0], dtype=float)
        candidate_is_left = bool(world[0] < 0)
        if candidate_is_left == tumor_is_left:
            n_ipsilateral += 1
        else:
            n_contralateral += 1
    result["n_contralateral"] = n_contralateral
    result["n_ipsilateral"] = n_ipsilateral
    return result
