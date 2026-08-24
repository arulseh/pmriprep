"""Tests for DSC AIF atlas and whole-brain search-mask preparation."""

import numpy as np
import pytest

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


def test_whole_brain_edge_exclusion_uses_physical_inplane_distance() -> None:
    """Whole-brain support uses the same physical edge depth as atlas tiers."""
    brain = np.zeros((21, 21, 3), dtype=np.uint8)
    brain[1:20, 1:20, :] = 1

    one_mm = AIFDetection._prepare_search_mask(
        brain, None, None, voxel_sizes=(1.0, 1.0, 4.0),
    )
    two_mm = AIFDetection._prepare_search_mask(
        brain, None, None, voxel_sizes=(2.0, 2.0, 4.0),
    )

    assert not one_mm[4, 10, 1]
    assert two_mm[4, 10, 1]
    assert one_mm[8, 10, 1]


def _ellipsoid(
    shape: tuple[int, int, int],
    angle_degrees: float = 0.0,
) -> np.ndarray:
    x, y = np.indices(shape[:2], dtype=float)
    x -= (shape[0] - 1) / 2.0
    y -= (shape[1] - 1) / 2.0
    angle = np.deg2rad(angle_degrees)
    across = np.cos(angle) * x + np.sin(angle) * y
    anterior = -np.sin(angle) * x + np.cos(angle) * y
    projection = (across / 40.0) ** 2 + (anterior / 50.0) ** 2 <= 1.0
    return np.broadcast_to(projection[:, :, None], shape).astype(np.uint8)


def test_symmetry_plane_tracks_inplane_rotation() -> None:
    """The posterior midline zone follows a rotated brain projection."""
    brain = _ellipsoid((101, 121, 9), angle_degrees=12.0)

    zone, angle, overlap = AIFDetection._brain_symmetry_midline_zone(
        brain, np.eye(4),
    )

    assert zone is not None
    assert angle == pytest.approx(12.0, abs=1.0)
    assert overlap > 0.98


def test_whole_brain_zone_preserves_anterior_midline_and_edge_slices() -> None:
    """The venous band is posterior-only and edge slices stay excluded."""
    brain = _ellipsoid((101, 121, 9))
    zone, _, _ = AIFDetection._brain_symmetry_midline_zone(brain, np.eye(4))

    search = AIFDetection._prepare_search_mask(
        brain,
        None,
        None,
        voxel_sizes=(1.0, 1.0, 1.0),
        apply_posterior_midline_exclusion=True,
        posterior_midline_zone=zone,
    )

    assert not search[50, 25, 4]
    assert search[50, 100, 4]
    assert not np.any(search[:, :, 0])
    assert not np.any(search[:, :, -1])


def test_degenerate_symmetry_uses_posterior_scanner_axis_fallback() -> None:
    """Unavailable symmetry preserves an explicit posterior-only exclusion."""
    brain = _ellipsoid((101, 121, 9))
    zone, angle, overlap = AIFDetection._brain_symmetry_midline_zone(
        brain, np.zeros((4, 4), dtype=float),
    )

    search = AIFDetection._prepare_search_mask(
        brain,
        None,
        None,
        voxel_sizes=(1.0, 1.0, 1.0),
        apply_posterior_midline_exclusion=True,
        posterior_midline_zone=zone,
    )

    assert zone is None
    assert np.isnan(angle)
    assert np.isnan(overlap)
    assert not search[50, 25, 4]
    assert search[50, 100, 4]


def test_spatial_exclusions_do_not_empty_a_small_roi() -> None:
    """Midline and edge-slice cuts retain a non-empty supplied ROI."""
    brain = np.ones((21, 21, 5), dtype=np.uint8)
    posterior_midline = np.zeros_like(brain)
    posterior_midline[10, 5, 2] = 1
    zone = np.zeros_like(brain)
    zone[10, :12, :] = 1

    midline_search = AIFDetection._prepare_search_mask(
        brain,
        posterior_midline,
        None,
        voxel_sizes=(1.0, 1.0, 1.0),
        apply_posterior_midline_exclusion=True,
        posterior_midline_zone=zone,
    )

    edge_slice = np.zeros_like(brain)
    edge_slice[10, 15, 0] = 1
    edge_search = AIFDetection._prepare_search_mask(
        brain,
        edge_slice,
        None,
        voxel_sizes=(1.0, 1.0, 1.0),
    )

    assert midline_search[10, 5, 2]
    assert edge_search[10, 15, 0]
