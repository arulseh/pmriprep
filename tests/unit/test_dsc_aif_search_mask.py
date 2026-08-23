"""Tests for DSC AIF atlas and whole-brain search-mask preparation."""

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from pmriprep.interfaces.dsc_aif import AIFDetection


def test_atlas_edge_exclusion_uses_physical_inplane_distance() -> None:
    """Atlas support changes with physical sampling rather than voxel count."""
    brain = np.zeros((21, 21, 1), dtype=np.uint8)
    brain[1:20, 1:20, :] = 1
    atlas = np.ones_like(brain)

    one_mm = AIFDetection._prepare_search_mask(
        brain,
        atlas,
        None,
        voxel_sizes=(1.0, 1.0, 4.0),
    )
    two_mm = AIFDetection._prepare_search_mask(
        brain,
        atlas,
        None,
        voxel_sizes=(2.0, 2.0, 4.0),
    )

    assert not one_mm[4, 10, 0]
    assert two_mm[4, 10, 0]
    assert one_mm[8, 10, 0]


def test_atlas_edge_distance_ignores_enclosed_mask_holes() -> None:
    """A mask defect remains excluded without creating an exclusion rim."""
    brain = np.zeros((23, 23, 1), dtype=np.uint8)
    brain[1:22, 1:22, :] = 1
    brain[11, 11, 0] = 0
    atlas = np.ones_like(brain)

    search = AIFDetection._prepare_search_mask(
        brain,
        atlas,
        None,
        voxel_sizes=(1.0, 1.0, 4.0),
    )

    assert not search[11, 11, 0]
    assert search[11, 10, 0]
    assert search[10, 11, 0]


def test_empty_atlas_interior_falls_back_to_unreduced_intersection() -> None:
    """Small atlas support remains searchable when the 8 mm cut empties it."""
    brain = np.ones((5, 5, 1), dtype=np.uint8)
    atlas = np.zeros_like(brain)
    atlas[2, 2, 0] = 1

    search = AIFDetection._prepare_search_mask(
        brain,
        atlas,
        None,
        voxel_sizes=(1.0, 1.0, 4.0),
    )

    np.testing.assert_array_equal(search, atlas.astype(bool))


def test_whole_brain_tier_retains_four_iteration_inplane_erosion() -> None:
    """The atlas change does not alter whole-brain fallback support."""
    brain = np.ones((15, 15, 3), dtype=np.uint8)
    structure = generate_binary_structure(3, 1)
    structure[:, :, 0] = 0
    structure[:, :, 2] = 0
    expected = binary_erosion(brain, structure=structure, iterations=4)

    search = AIFDetection._prepare_search_mask(brain, None, None)

    np.testing.assert_array_equal(search, expected)
