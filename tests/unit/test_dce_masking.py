"""Unit tests for DCE masking operations.

Tests fill-then-erode T1 edge erosion, enhancement gating,
NaN-aware median filtering, slab-aware erosion, hemispheric asymmetry,
and B1 fill-then-erode interior mask construction.
"""

import numpy as np
import pytest
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    generate_binary_structure,
)

from pmriprep.interfaces.dce import _b1_interior_mask, _nanmedian_filter
from pmriprep.utils.dce_constants import MIN_ENHANCEMENT_MM, SLAB_EROSION_MAX_SLICES


# ============================================================================
# Shared test helpers
# ============================================================================


def _erode_slab_aware(valid_filled: np.ndarray) -> np.ndarray:
    """Slab-aware erosion (production algorithm)."""
    n_slices = valid_filled.shape[2]
    if n_slices <= SLAB_EROSION_MAX_SLICES:
        struct_2d = generate_binary_structure(2, 1)
        eroded = np.zeros_like(valid_filled)
        for z in range(n_slices):
            eroded[:, :, z] = binary_erosion(
                valid_filled[:, :, z], structure=struct_2d, iterations=1,
            )
    else:
        struct = generate_binary_structure(3, 1)
        eroded = binary_erosion(valid_filled, structure=struct, iterations=1)
    return eroded


# ============================================================================
# Fill-then-erode T1 edge erosion
# ============================================================================


class TestFillThenErodeT1:
    """Tests for the fill-then-erode T1 edge erosion logic.

    Reproduces the pattern from dce.py lines ~1620-1629:
        valid_filled = binary_fill_holes(valid_t1_mask)
        eroded = binary_erosion(valid_filled, structure=struct, iterations=1)
        edge_mask = valid_t1_mask & ~eroded
    """

    @staticmethod
    def _edge_mask(valid_t1_mask: np.ndarray) -> np.ndarray:
        """Compute edge mask using the production algorithm."""
        struct = generate_binary_structure(3, 1)  # 6-connectivity
        valid_filled = binary_fill_holes(valid_t1_mask)
        eroded = binary_erosion(valid_filled, structure=struct, iterations=1)
        return valid_t1_mask & ~eroded

    def test_internal_hole_not_eroded(self) -> None:
        """Internal hole should not widen after fill-then-erode."""
        mask = np.ones((11, 11, 11), dtype=bool)
        # Punch a small internal hole
        mask[5, 5, 5] = False

        edge = self._edge_mask(mask)

        # The hole voxel was never valid, so it can't appear in edge_mask
        assert not edge[5, 5, 5]
        # Neighbors of the hole should NOT be eroded (they are interior
        # after fill_holes closes the hole)
        assert not edge[4, 5, 5]
        assert not edge[6, 5, 5]

    def test_solid_mask_standard_erosion(self) -> None:
        """Solid mask should behave like standard 6-conn erosion."""
        mask = np.ones((11, 11, 11), dtype=bool)

        edge = self._edge_mask(mask)

        # Center is interior, not edge
        assert not edge[5, 5, 5]
        # Face of cube is edge
        assert edge[0, 5, 5]
        assert edge[5, 0, 5]

    def test_large_internal_void_not_widened(self) -> None:
        """Large internal void (e.g. tumor cavity) should not grow."""
        mask = np.ones((15, 15, 15), dtype=bool)
        # 3x3x3 internal void
        mask[6:9, 6:9, 6:9] = False

        edge = self._edge_mask(mask)

        # Voxels adjacent to the void should NOT be marked as edge
        # (fill_holes closes the void, so these are interior after erosion)
        assert not edge[5, 7, 7]
        assert not edge[9, 7, 7]

    def test_default_fill_closes_diagonal_escape(self) -> None:
        """Default (6-conn background) fill closes a diagonal escape path.

        binary_fill_holes default uses 6-conn structure for background
        connectivity: diagonal-only escape paths don't count as connected
        to the border, so the void is filled. A 26-conn structure WOULD
        see the diagonal path and leave the void open.

        The production code uses the default, which is the more aggressive
        fill — correct for closing CSF/noisy dropout holes.
        """
        mask = np.ones((9, 9, 9), dtype=bool)
        # Diagonal channel: each step differs in all 3 coordinates.
        # 26-connected but NOT 6-connected to the border.
        mask[4, 4, 4] = False
        mask[3, 3, 3] = False
        mask[2, 2, 2] = False
        mask[1, 1, 1] = False
        mask[0, 0, 0] = False  # reaches exterior face

        # Default (6-conn bg): diagonal NOT connected → void is filled
        filled_default = binary_fill_holes(mask)
        assert filled_default[4, 4, 4]

        # 26-conn bg: diagonal IS connected → void is NOT filled
        struct_26 = generate_binary_structure(3, 3)
        filled_26 = binary_fill_holes(mask, structure=struct_26)
        assert not filled_26[4, 4, 4]


# ============================================================================
# Enhancement gate
# ============================================================================


class TestEnhancementGate:
    """Tests for the enhancement gate: max(conc) > MIN_ENHANCEMENT_MM.

    The threshold is 1e-8 mM (from dce_constants.py).
    """

    THRESHOLD = 1e-8

    def test_peak_above_threshold_passes(self) -> None:
        """Voxel with peak > 1e-8 should pass the gate."""
        conc = np.array([0.0, 0.0, 1e-7, 0.0])
        assert np.max(conc) > self.THRESHOLD

    def test_peak_at_threshold_fails(self) -> None:
        """Voxel with peak == 1e-8 should fail (strict >)."""
        conc = np.array([0.0, 0.0, 1e-8, 0.0])
        assert not (np.max(conc) > self.THRESHOLD)

    def test_all_zero_fails(self) -> None:
        """All-zero concentration should be filtered."""
        conc = np.zeros(20)
        assert not (np.max(conc) > self.THRESHOLD)

    def test_tiny_positive_passes(self) -> None:
        """Tiny positive enhancement (above threshold) should pass."""
        conc = np.array([0.0, 2e-8])
        assert np.max(conc) > self.THRESHOLD


# ============================================================================
# B1 fill-then-erode interior mask
# ============================================================================


class TestB1FillThenErode:
    """Tests for B1 smoothing interior mask construction.

    Uses the extracted _b1_interior_mask() helper from dce.py.
    """

    @staticmethod
    def _interior(valid: np.ndarray) -> np.ndarray:
        """Compute interior mask using the production helper."""
        return _b1_interior_mask(valid)

    def test_hole_does_not_shrink_interior(self) -> None:
        """Internal hole should not cause interior voxels to be lost."""
        solid = np.ones((15, 15, 15), dtype=bool)
        with_hole = solid.copy()
        with_hole[7, 7, 7] = False

        interior_solid = self._interior(solid)
        interior_holed = self._interior(with_hole)

        # The hole voxel itself is not valid, so not in either interior
        assert not interior_holed[7, 7, 7]
        # But neighbors of the hole should still be interior (fill closes it)
        assert interior_holed[7, 7, 8]
        # Interior count should differ by at most the hole voxel
        diff = int(np.sum(interior_solid)) - int(np.sum(interior_holed))
        assert diff <= 1

    def test_invalid_voxels_excluded_by_intersection(self) -> None:
        """Voxels not in original valid mask must not appear in interior.

        This is the critical & valid intersection: after fill_holes, the
        filled region may include voxels where b1_map = 1.0 (default/invalid).
        These must not contaminate the Gaussian-weighted reference.
        """
        mask = np.ones((15, 15, 15), dtype=bool)
        # Create a band of invalid voxels (simulating bad B1+ slice)
        mask[7, :, :] = False

        interior = self._interior(mask)

        # No voxel in the invalid band should be in interior
        assert not np.any(interior[7, :, :])

    def test_solid_mask_matches_simple_erosion(self) -> None:
        """For a solid mask, fill-then-erode should equal simple erosion."""
        mask = np.ones((15, 15, 15), dtype=bool)

        interior_new = self._interior(mask)
        interior_old = binary_erosion(mask, iterations=2)

        np.testing.assert_array_equal(interior_new, interior_old)


# ============================================================================
# NaN-aware median filter
# ============================================================================


class TestNanMedianFilter:
    """Tests for finite-neighborhood T1 median filtering."""

    def test_boundary_uses_only_observed_neighbors(self) -> None:
        """NaN background must not vote as a repeated population median."""
        t1_map = np.full((7, 7, 7), np.nan)
        t1_map[2:5, 2:5, 2:5] = 900.0
        t1_map[3, 3, 3] = 1500.0

        result = _nanmedian_filter(t1_map)

        assert result[2, 2, 2] == pytest.approx(900.0)
        assert np.isnan(result[1, 2, 2])

    def test_does_not_create_global_median_rail(self) -> None:
        """Sparse holes must not create exact whole-map median plateaus."""
        values = np.arange(343, dtype=float).reshape(7, 7, 7) + 700.0
        values[[0, 1, 5, 6], :, :] = np.nan
        global_median = float(np.nanmedian(values))

        result = _nanmedian_filter(values)

        assert np.count_nonzero(result == global_median) < 4

    def test_interior_voxels_unchanged(self) -> None:
        """A constant interior remains unchanged."""
        t1_map = np.full((11, 11, 11), 1000.0)
        # NaN border only on faces
        t1_map[0, :, :] = np.nan
        t1_map[-1, :, :] = np.nan
        t1_map[:, 0, :] = np.nan
        t1_map[:, -1, :] = np.nan
        t1_map[:, :, 0] = np.nan
        t1_map[:, :, -1] = np.nan

        result = _nanmedian_filter(t1_map)
        np.testing.assert_array_equal(result[3:-3, 3:-3, 3:-3], 1000.0)

    def test_all_nan_returns_nan(self) -> None:
        """All-NaN input should remain all-NaN."""
        t1_map = np.full((5, 5, 5), np.nan)
        result = _nanmedian_filter(t1_map)
        assert np.all(np.isnan(result))


# ============================================================================
# Slab-aware erosion
# ============================================================================


class TestSlabAwareErosion:
    """Tests for slab-aware erosion: 2D per-slice for thin slabs, 3D for thick."""

    def test_thin_slab_preserves_top_bottom_slices(self) -> None:
        """2D erosion for thin slab (<=30 slices) preserves top/bottom slice interior."""
        mask = np.ones((20, 20, 24), dtype=bool)

        eroded = _erode_slab_aware(mask)

        # Interior of top/bottom slices should be preserved with 2D erosion
        assert eroded[10, 10, 0], "Top slice interior lost with 2D erosion"
        assert eroded[10, 10, 23], "Bottom slice interior lost with 2D erosion"

        # 3D erosion would have removed these
        struct_3d = generate_binary_structure(3, 1)
        eroded_3d = binary_erosion(mask, structure=struct_3d, iterations=1)
        assert not eroded_3d[10, 10, 0], "3D erosion should remove top slice"
        assert not eroded_3d[10, 10, 23], "3D erosion should remove bottom slice"

    def test_thin_slab_still_erodes_in_plane(self) -> None:
        """2D erosion still removes in-plane boundary voxels."""
        mask = np.ones((20, 20, 10), dtype=bool)

        eroded = _erode_slab_aware(mask)

        # In-plane edges should be eroded
        assert not eroded[0, 10, 5], "In-plane edge should be eroded"
        assert not eroded[10, 0, 5], "In-plane edge should be eroded"
        # Interior should be preserved
        assert eroded[10, 10, 5]

    def test_thick_volume_uses_3d_erosion(self) -> None:
        """Volumes with >30 slices should use standard 3D erosion."""
        mask = np.ones((40, 40, 40), dtype=bool)

        eroded = _erode_slab_aware(mask)

        # 3D erosion removes top/bottom slices
        assert not eroded[20, 20, 0], "3D erosion should remove top slice"
        assert not eroded[20, 20, 39], "3D erosion should remove bottom slice"
        # Interior preserved
        assert eroded[20, 20, 20]

    def test_boundary_n_slices_30_uses_2d(self) -> None:
        """Exactly 30 slices should use 2D erosion (<=30 threshold)."""
        mask = np.ones((15, 15, 30), dtype=bool)

        eroded = _erode_slab_aware(mask)

        # Top/bottom slice interior should be preserved (2D mode)
        assert eroded[7, 7, 0]
        assert eroded[7, 7, 29]

    def test_boundary_n_slices_31_uses_3d(self) -> None:
        """31 slices should use 3D erosion (>30 threshold)."""
        mask = np.ones((15, 15, 31), dtype=bool)

        eroded = _erode_slab_aware(mask)

        # Top/bottom slice should be eroded (3D mode)
        assert not eroded[7, 7, 0]
        assert not eroded[7, 7, 30]


# ============================================================================
# Hemispheric asymmetry QC
# ============================================================================


class TestHemisphericAsymmetry:
    """Tests for hemispheric asymmetry QC metric computation."""

    @staticmethod
    def _compute_asymmetry(
        t1_map: np.ndarray, brain_mask: np.ndarray,
    ) -> dict:
        """Reproduce the production hemispheric asymmetry logic."""
        shape = t1_map.shape
        midplane = shape[0] // 2
        right_valid = brain_mask[:midplane] & np.isfinite(t1_map[:midplane])
        left_valid = brain_mask[midplane:] & np.isfinite(t1_map[midplane:])
        right_t1 = t1_map[:midplane][right_valid]
        left_t1 = t1_map[midplane:][left_valid]
        if len(right_t1) > 100 and len(left_t1) > 100:
            med_r = float(np.median(right_t1))
            med_l = float(np.median(left_t1))
            denom = (med_r + med_l) / 2.0
            asym = abs(med_r - med_l) / denom if denom > 0 else 0.0
            return {
                "asymmetry_index": round(float(asym), 4),
                "median_right_ms": round(med_r, 1),
                "median_left_ms": round(med_l, 1),
                "warn": bool(asym > 0.15),
                "fail": bool(asym > 0.30),
            }
        return {}

    def test_symmetric_brain_low_asymmetry(self) -> None:
        """Symmetric T1 map should have near-zero asymmetry index."""
        rng = np.random.default_rng(42)
        t1_map = np.full((20, 20, 20), np.nan)
        brain_mask = np.zeros((20, 20, 20), dtype=bool)
        # Fill both hemispheres with similar T1 values
        brain_mask[2:18, 2:18, 2:18] = True
        t1_map[brain_mask] = 1000.0 + rng.normal(0, 20, size=int(np.sum(brain_mask)))

        result = self._compute_asymmetry(t1_map, brain_mask)

        assert result["asymmetry_index"] < 0.05
        assert not result["warn"]
        assert not result["fail"]

    def test_asymmetric_brain_triggers_warn(self) -> None:
        """20% median difference should trigger warn but not fail."""
        t1_map = np.full((20, 20, 20), np.nan)
        brain_mask = np.zeros((20, 20, 20), dtype=bool)
        brain_mask[2:18, 2:18, 2:18] = True
        # Right hemisphere: T1=1000ms, Left hemisphere: T1=1200ms (20% diff)
        t1_map[2:10, 2:18, 2:18] = 1000.0
        t1_map[10:18, 2:18, 2:18] = 1200.0

        result = self._compute_asymmetry(t1_map, brain_mask)

        assert result["warn"]
        assert not result["fail"]

    def test_severe_asymmetry_triggers_fail(self) -> None:
        """40% median difference should trigger fail."""
        t1_map = np.full((20, 20, 20), np.nan)
        brain_mask = np.zeros((20, 20, 20), dtype=bool)
        brain_mask[2:18, 2:18, 2:18] = True
        # Right: 800ms, Left: 1200ms (~40% diff)
        t1_map[2:10, 2:18, 2:18] = 800.0
        t1_map[10:18, 2:18, 2:18] = 1200.0

        result = self._compute_asymmetry(t1_map, brain_mask)

        assert result["fail"]

    def test_insufficient_voxels_returns_empty(self) -> None:
        """Fewer than 100 voxels in a hemisphere should return empty dict."""
        t1_map = np.full((10, 10, 10), np.nan)
        brain_mask = np.zeros((10, 10, 10), dtype=bool)
        # Only a few voxels in each hemisphere
        brain_mask[2:4, 2:4, 2:4] = True
        brain_mask[6:8, 6:8, 6:8] = True
        t1_map[brain_mask] = 1000.0

        result = self._compute_asymmetry(t1_map, brain_mask)

        assert result == {}


# ============================================================================
# Additional edge-case tests
# ============================================================================


class TestNanMedianFilterEdgeCases:
    """Edge-case tests for finite-neighborhood median filtering."""

    def test_wide_t1_range_no_bias(self) -> None:
        """Filtering should not bias boundaries toward another tissue type."""
        rng = np.random.default_rng(123)
        t1_map = np.full((12, 12, 12), np.nan)
        center = np.array([6, 6, 6])
        for ix in range(12):
            for iy in range(12):
                for iz in range(12):
                    dist = np.linalg.norm(np.array([ix, iy, iz]) - center)
                    if dist <= 4:
                        # Assign tissue-like T1: WM=800, GM=1400, CSF=4000
                        if dist <= 2:
                            t1_map[ix, iy, iz] = 4000.0 + rng.normal(0, 100)
                        elif dist <= 3:
                            t1_map[ix, iy, iz] = 1400.0 + rng.normal(0, 50)
                        else:
                            t1_map[ix, iy, iz] = 800.0 + rng.normal(0, 30)

        result = _nanmedian_filter(t1_map.copy())

        # Boundary WM voxels (dist ~4) should remain in WM range, not be
        # pulled toward the median (~1400ms GM)
        brain = np.isfinite(t1_map)
        boundary_wm = brain & (result > 0) & (result < 1000)
        assert int(np.sum(boundary_wm)) > 0, "Should have boundary WM voxels"
        # No voxel should become negative
        assert not np.any(result[brain] < 0)

    def test_single_valid_voxel_preserved(self) -> None:
        """A single valid voxel surrounded by NaN remains unchanged."""
        t1_map = np.full((5, 5, 5), np.nan)
        t1_map[2, 2, 2] = 1500.0

        result = _nanmedian_filter(t1_map.copy())
        assert result[2, 2, 2] == pytest.approx(1500.0)


class TestSlabAwareErosionEdgeCases:
    """Edge-case tests for slab-aware erosion."""

    def test_spherical_mask_thin_slab(self) -> None:
        """Spherical brain mask in thin slab — curved boundary eroded in-plane."""
        mask = np.zeros((20, 20, 15), dtype=bool)
        center = np.array([10, 10, 7])
        for ix in range(20):
            for iy in range(20):
                for iz in range(15):
                    if np.linalg.norm(np.array([ix, iy, iz]) - center) <= 6:
                        mask[ix, iy, iz] = True

        eroded = _erode_slab_aware(mask)

        # Interior should be preserved
        assert eroded[10, 10, 7]
        # In-plane boundary should be eroded (voxel at edge of sphere)
        mid_slice = mask[:, :, 7].copy()
        mid_eroded = eroded[:, :, 7]
        boundary = mid_slice & ~mid_eroded
        assert int(np.sum(boundary)) > 0, "Should erode in-plane boundary"
        # Each slice with data should retain some interior voxels after 2D erosion
        # (sphere radius 6 means inner slices have radius >= 3, enough to survive)
        for z in range(15):
            if np.sum(mask[:, :, z]) > 8:  # slices with enough data
                assert np.any(eroded[:, :, z]), (
                    f"Slice {z} lost all voxels despite having "
                    f"{int(np.sum(mask[:, :, z]))} input voxels"
                )

    def test_empty_slice_in_middle(self) -> None:
        """Empty slice in the middle of slab — 2D erosion handles gracefully."""
        mask = np.ones((15, 15, 10), dtype=bool)
        # Clear an entire slice
        mask[:, :, 5] = False

        eroded = _erode_slab_aware(mask)

        # Empty slice stays empty
        assert not np.any(eroded[:, :, 5])
        # Neighboring slices still have interior
        assert eroded[7, 7, 4]
        assert eroded[7, 7, 6]


class TestEnhancementGateNegative:
    """Test that negative concentrations fail the enhancement gate."""

    THRESHOLD = MIN_ENHANCEMENT_MM

    def test_negative_concentration_fails(self) -> None:
        """Negative concentration (noise/baseline drift) should fail the gate."""
        conc = np.array([0.0, -0.01, -0.005, 0.0])
        assert not (np.max(conc) > self.THRESHOLD)

    def test_mixed_negative_positive_below_threshold_fails(self) -> None:
        """Mix of negative and tiny positive (below threshold) fails."""
        conc = np.array([-0.01, 0.0, 1e-9, -0.005])
        assert not (np.max(conc) > self.THRESHOLD)

    def test_mixed_negative_positive_above_threshold_passes(self) -> None:
        """Negative values don't prevent passing if peak exceeds threshold."""
        conc = np.array([-0.01, 0.0, 1e-7, -0.005])
        assert np.max(conc) > self.THRESHOLD
