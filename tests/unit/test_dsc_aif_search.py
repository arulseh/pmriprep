"""Unit tests for DSC AIF search pipeline."""

import numpy as np
import pytest

from pmriprep.utils.dsc_aif import (
    AIFCandidate,
    AIFSearchResult,
    atlas_based_aif_search,
    _search_candidates_in_mask,
    exclude_near_lesion,
    get_contralateral_mask,
    get_aif_thresholds,
)


# =============================================================================
# Helper
# =============================================================================


def _make_synthetic_concentration_4d(
    shape: tuple[int, int, int, int] = (10, 10, 10, 30),
    hot_voxels: list[tuple[int, int, int]] | None = None,
    baseline_frames: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create 4D concentration with optional hot arterial voxels.

    Parameters
    ----------
    shape : tuple
        Spatial + temporal shape (x, y, z, t).
    hot_voxels : list of (x, y, z) tuples, optional
        Voxel coordinates that receive a gamma-variate AIF-like signal.
    baseline_frames : int
        Number of clean baseline frames.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        4D float64 array.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    data = rng.normal(0, 0.1, shape)
    data[:, :, :, :baseline_frames] = 0.0  # Clean baseline

    if hot_voxels is not None:
        for x, y, z in hot_voxels:
            t = np.arange(shape[3], dtype=np.float64)
            aif = 20.0 * ((t - baseline_frames) / 3.0) * np.exp(
                -(t - baseline_frames) / 4.0
            )
            aif[:baseline_frames] = 0.0
            aif = np.maximum(aif, 0.0)
            data[x, y, z, :] = aif

    return data.astype(np.float64)


# =============================================================================
# TestAtlasBasedAIFSearch
# =============================================================================


class TestAtlasBasedAIFSearch:
    """Tests for atlas_based_aif_search."""

    def test_atlas_strict_success(self) -> None:
        """Hot voxels inside atlas mask are detected at atlas_strict level."""
        shape = (10, 10, 10, 30)
        hot = [(3, 3, 3), (3, 4, 3), (4, 3, 3), (4, 4, 3), (3, 3, 4)]
        data = _make_synthetic_concentration_4d(shape=shape, hot_voxels=hot)

        brain_mask = np.ones(shape[:3], dtype=np.uint8)
        atlas_mask = np.zeros(shape[:3], dtype=np.uint8)
        for x, y, z in hot:
            atlas_mask[x, y, z] = 1

        result = atlas_based_aif_search(
            concentration_data=data,
            brain_mask=brain_mask,
            atlas_mask=atlas_mask,
            baseline_frames=5,
            n_candidates=5,
        )

        assert result.detection_level == "atlas_strict"
        assert len(result.candidates) > 0

    def test_fallback_to_whole_brain(self) -> None:
        """Without an atlas mask the search falls back to whole_brain level."""
        shape = (10, 10, 10, 30)
        hot = [(5, 5, 5)]
        data = _make_synthetic_concentration_4d(shape=shape, hot_voxels=hot)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        result = atlas_based_aif_search(
            concentration_data=data,
            brain_mask=brain_mask,
            atlas_mask=None,
            baseline_frames=5,
        )

        assert result.detection_level.startswith("whole_brain")

    def test_no_candidates_flat_data(self) -> None:
        """All-zero concentration produces zero passed candidates."""
        shape = (10, 10, 10, 30)
        data = np.zeros(shape, dtype=np.float64)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        result = atlas_based_aif_search(
            concentration_data=data,
            brain_mask=brain_mask,
            atlas_mask=None,
            baseline_frames=5,
        )

        assert result.n_candidates_passed == 0

    def test_n_candidates_limit(self) -> None:
        """Number of returned candidates respects n_candidates limit."""
        shape = (10, 10, 10, 30)
        # 20 hot voxels spread in a plane
        hot = [(i, j, 5) for i in range(2, 7) for j in range(2, 6)]
        data = _make_synthetic_concentration_4d(shape=shape, hot_voxels=hot)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        result = atlas_based_aif_search(
            concentration_data=data,
            brain_mask=brain_mask,
            atlas_mask=None,
            baseline_frames=5,
            n_candidates=5,
        )

        assert len(result.candidates) <= 5

    def test_aif_curve_shape(self) -> None:
        """Returned AIF curve has the correct temporal dimension."""
        shape = (10, 10, 10, 30)
        hot = [(5, 5, 5)]
        data = _make_synthetic_concentration_4d(shape=shape, hot_voxels=hot)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        result = atlas_based_aif_search(
            concentration_data=data,
            brain_mask=brain_mask,
            atlas_mask=None,
            baseline_frames=5,
        )

        assert result.aif_curve.shape == (30,)

    def test_aif_mask_binary(self) -> None:
        """Returned AIF mask contains only 0 and 1 values."""
        shape = (10, 10, 10, 30)
        hot = [(5, 5, 5)]
        data = _make_synthetic_concentration_4d(shape=shape, hot_voxels=hot)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        result = atlas_based_aif_search(
            concentration_data=data,
            brain_mask=brain_mask,
            atlas_mask=None,
            baseline_frames=5,
        )

        unique_vals = set(np.unique(result.aif_mask))
        assert unique_vals.issubset({0, 1})

    def test_qc_metrics_keys(self) -> None:
        """Result qc_metrics is a dict (possibly empty when no candidates)."""
        shape = (10, 10, 10, 30)
        data = np.zeros(shape, dtype=np.float64)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        result = atlas_based_aif_search(
            concentration_data=data,
            brain_mask=brain_mask,
            atlas_mask=None,
            baseline_frames=5,
        )

        assert isinstance(result.qc_metrics, dict)


# =============================================================================
# TestSearchCandidatesInMask
# =============================================================================


class TestSearchCandidatesInMask:
    """Tests for _search_candidates_in_mask."""

    def test_known_good_voxels(self) -> None:
        """Hot voxels within the mask are returned as candidates."""
        shape = (10, 10, 10, 30)
        hot = [(3, 3, 3), (4, 4, 4)]
        data = _make_synthetic_concentration_4d(shape=shape, hot_voxels=hot)

        mask = np.zeros(shape[:3], dtype=np.uint8)
        for x, y, z in hot:
            mask[x, y, z] = 1

        thresholds = get_aif_thresholds("relaxed")
        candidates, n_evaluated = _search_candidates_in_mask(
            data, mask, thresholds, baseline_frames=5, max_candidates=10,
        )

        assert n_evaluated > 0
        assert len(candidates) > 0
        assert all(isinstance(c, AIFCandidate) for c in candidates)

    def test_empty_mask(self) -> None:
        """All-zero mask returns no candidates and zero evaluated."""
        shape = (10, 10, 10, 30)
        data = _make_synthetic_concentration_4d(shape=shape)
        mask = np.zeros(shape[:3], dtype=np.uint8)

        thresholds = get_aif_thresholds("strict")
        candidates, n_evaluated = _search_candidates_in_mask(
            data, mask, thresholds, baseline_frames=5, max_candidates=10,
        )

        assert candidates == []
        assert n_evaluated == 0


# =============================================================================
# TestExcludeNearLesion
# =============================================================================


class TestExcludeNearLesion:
    """Tests for exclude_near_lesion."""

    def test_exclude_near_lesion(self) -> None:
        """Voxels near lesion are excluded while distant ones remain."""
        search_mask = np.ones((10, 10, 10), dtype=np.uint8)
        lesion_mask = np.zeros((10, 10, 10), dtype=np.uint8)
        lesion_mask[5, 5, 5] = 1

        result = exclude_near_lesion(
            search_mask,
            lesion_mask,
            voxel_sizes=(1.0, 1.0, 1.0),
            min_distance_mm=3.0,
        )

        # Near lesion should be excluded
        assert result[5, 5, 5] == 0
        assert result[5, 5, 6] == 0  # distance 1 mm < 3 mm

        # Far corners should remain
        assert result[0, 0, 0] == 1  # distance ~8.66 mm > 3 mm

    def test_no_lesion(self) -> None:
        """All-zero lesion mask preserves the search mask.

        When there is no lesion, inverted = (lesion <= 0) = all-ones.
        distance_transform_edt of an all-ones array (no zero reference points)
        yields large distances, so all voxels pass the distance threshold
        and the search mask is effectively preserved.
        """
        search_mask = np.ones((10, 10, 10), dtype=np.uint8)
        lesion_mask = np.zeros((10, 10, 10), dtype=np.uint8)

        result = exclude_near_lesion(
            search_mask,
            lesion_mask,
            voxel_sizes=(1.0, 1.0, 1.0),
            min_distance_mm=3.0,
        )

        # Most voxels should be preserved (those far enough from edges)
        # With a 10x10x10 volume the interior voxels all have distance > 3
        assert np.sum(result) > 900


# =============================================================================
# TestGetContralateralMask
# =============================================================================


class TestGetContralateralMask:
    """Tests for get_contralateral_mask."""

    def test_left_lesion_right_mask(self) -> None:
        """Lesion on left (x < midline) produces contralateral mask on right."""
        brain_mask = np.ones((20, 20, 20), dtype=np.uint8)
        lesion_mask = np.zeros((20, 20, 20), dtype=np.uint8)
        # Lesion centered at x=3 (left of midline x=10)
        lesion_mask[2:5, 8:12, 8:12] = 1

        result = get_contralateral_mask(brain_mask, lesion_mask)

        # Right hemisphere (x >= 10) should have nonzero values
        assert np.sum(result[10:, :, :]) > 0
        # Left hemisphere (x < 10) should be all zero
        assert np.sum(result[:10, :, :]) == 0
