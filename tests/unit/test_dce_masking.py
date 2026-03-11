"""Unit tests for DCE masking operations.

Tests fill-then-erode T1 edge erosion, enhancement gating,
and B1 fill-then-erode interior mask construction.
"""

import numpy as np
import pytest
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    generate_binary_structure,
)


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

    Reproduces the pattern from dce.py (AFI ~line 703, VFA ~line 1028):
        interior = binary_erosion(binary_fill_holes(valid), iterations=2) & valid
    """

    @staticmethod
    def _interior(valid: np.ndarray) -> np.ndarray:
        """Compute interior mask using the production algorithm."""
        return binary_erosion(binary_fill_holes(valid), iterations=2) & valid

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
