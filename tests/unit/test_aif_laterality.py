"""Tests for shared descriptive AIF candidate laterality."""

import numpy as np
import pytest

from pmriprep.utils.aif_laterality import describe_tumor_relative_aif_candidates


def _centered_affine() -> np.ndarray:
    """Return an identity-orientation affine with the x midline at voxel 10."""
    affine = np.eye(4)
    affine[0, 3] = -10.0
    return affine


def test_unilateral_counts_use_final_candidate_coordinates() -> None:
    """Candidate counts are relative to a unilateral lesion in RAS space."""
    lesion = np.zeros((21, 3, 3), dtype=np.uint8)
    lesion[2:5, 1, 1] = 1

    result = describe_tumor_relative_aif_candidates(
        [(3, 1, 1), (15, 1, 1), (18, 1, 1)],
        _centered_affine(),
        lesion,
    )

    assert result == {
        "tumor_laterality": "left",
        "laterality_mask_source": "lesion",
        "n_contralateral": 2,
        "n_ipsilateral": 1,
    }


def test_enhancing_mask_is_preferred_and_bilateral_counts_are_omitted() -> None:
    """An enhancing mask controls classification without invented counts."""
    lesion = np.zeros((21, 3, 3), dtype=np.uint8)
    lesion[2:5, 1, 1] = 1
    enhancing = np.zeros_like(lesion)
    enhancing[0, 1, 1] = 1
    enhancing[20, 1, 1] = 1

    result = describe_tumor_relative_aif_candidates(
        [(3, 1, 1), (18, 1, 1)],
        _centered_affine(),
        lesion,
        enhancing_mask=enhancing,
        midline_tolerance_mm=5.0,
    )

    assert result == {
        "tumor_laterality": "bilateral",
        "laterality_mask_source": "enhancing",
    }


def test_missing_or_empty_mask_emits_no_laterality_claim() -> None:
    """Laterality provenance is absent when no non-empty mask exists."""
    empty = np.zeros((5, 5, 5), dtype=np.uint8)
    assert describe_tumor_relative_aif_candidates([], np.eye(4), None) == {}
    assert describe_tumor_relative_aif_candidates([], np.eye(4), empty) == {}


@pytest.mark.parametrize("x_direction", [1, -1])
def test_brain_midline_counts_ignore_scanner_translation(x_direction: int) -> None:
    """Off-centre and reversed-axis images keep coherent anatomical counts."""
    x, y, z = np.indices((25, 31, 5))
    brain = ((x - 12) / 9) ** 2 + ((y - 15) / 13) ** 2 + ((z - 2) / 3) ** 2 < 1
    lesion = np.zeros(brain.shape, dtype=np.uint8)
    lesion[6:9, 13:17, 1:4] = 1
    affine = np.diag([2.0 * x_direction, 2.0, 2.0, 1.0])
    candidates = [(7, 15, 2), (16, 15, 2), (18, 15, 2)]
    first = describe_tumor_relative_aif_candidates(
        candidates, affine, lesion, brain_mask=brain,
    )
    affine[:3, 3] = [100, -37, 12]
    shifted = describe_tumor_relative_aif_candidates(
        candidates, affine, lesion, brain_mask=brain,
    )
    assert shifted == first
    assert shifted["laterality_midline_method"] == "brain_symmetry"
    assert shifted["n_left_candidates"] == (1 if x_direction > 0 else 2)
    assert shifted["n_right_candidates"] == (2 if x_direction > 0 else 1)
    assert shifted["n_ipsilateral"] == 1
    assert shifted["n_contralateral"] == 2
