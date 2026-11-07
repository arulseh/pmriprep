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
    brain_mask: npt.NDArray[Any] | None = None,
    midline_tolerance_mm: float = 10.0,
) -> dict[str, str | int | float]:
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
    brain_mask : ndarray or None, optional
        Native-space brain mask for the symmetry-derived midsagittal plane.
        When supplied, candidate hemisphere counts use the same plane as
        tumor-relative counts. An unavailable estimate uses an explicitly
        labelled world-x fallback.
    midline_tolerance_mm : float, optional
        Minimum bilateral extent on both sides of the selected midline.

    Returns
    -------
    dict
        Descriptive QC fields. Contralateral and ipsilateral counts are
        included only for unilateral tumors, where those terms are defined.

    Notes
    -----
    These fields do not alter AIF candidate scores, selection, or acceptance.
    Candidates exactly on the plane are assigned to the right for counting.
    """
    from pmriprep.utils.tumor_seg import estimate_brain_symmetry_midline

    midline = (
        estimate_brain_symmetry_midline(np.asarray(brain_mask, dtype=bool), affine)
        if brain_mask is not None else None
    )

    def signed_distances(coordinates: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if midline is not None:
            return np.asarray(midline.signed_distance[tuple(coordinates.T)])
        return np.asarray(coordinates @ affine[0, :3] + affine[0, 3])

    result: dict[str, str | int | float] = {}
    candidate_left = signed_distances(np.asarray(candidates, dtype=int).reshape(-1, 3)) < 0
    if brain_mask is not None:
        n_left = int(np.count_nonzero(candidate_left))
        n_right = len(candidates) - n_left
        result.update(
            n_left_candidates=n_left, n_right_candidates=n_right,
            is_bilateral=bool(n_left and n_right),
            hemisphere_coverage=("bilateral" if n_left and n_right else
                                 "unilateral_left" if n_left else
                                 "unilateral_right" if n_right else "unavailable"),
            laterality_midline_method="brain_symmetry" if midline is not None else "world_x_fallback",
        )
        if midline is not None:
            result["laterality_midline_angle_degrees"] = midline.angle_degrees
        if len(candidates):
            result["aif_hemisphere"] = "left" if candidate_left[0] else "right"
    laterality_mask: npt.NDArray[Any] | None = None
    mask_source: str | None = None
    if enhancing_mask is not None and np.any(enhancing_mask > 0.5):
        laterality_mask = enhancing_mask
        mask_source = "enhancing"
    elif lesion_mask is not None and np.any(lesion_mask > 0.5):
        laterality_mask = lesion_mask
        mask_source = "lesion"

    if laterality_mask is None or mask_source is None:
        return result

    lesion_coords = np.argwhere(laterality_mask > 0.5)
    lesion_distances = signed_distances(lesion_coords)
    center_x = float(np.mean(lesion_distances))
    laterality: TumorLaterality
    if (
        float(np.min(lesion_distances)) < -midline_tolerance_mm
        and float(np.max(lesion_distances)) > midline_tolerance_mm
    ):
        laterality = "bilateral"
    else:
        laterality = "left" if center_x < 0 else "right"

    result.update({
        "tumor_laterality": laterality,
        "laterality_mask_source": mask_source,
    })
    if laterality == "bilateral":
        return result

    tumor_is_left = laterality == "left"
    n_contralateral = 0
    n_ipsilateral = 0
    for candidate_is_left in candidate_left:
        if candidate_is_left == tumor_is_left:
            n_ipsilateral += 1
        else:
            n_contralateral += 1
    result["n_contralateral"] = n_contralateral
    result["n_ipsilateral"] = n_ipsilateral
    return result
