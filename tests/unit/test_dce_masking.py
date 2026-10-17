"""Tests for B1 smoothing interior-mask construction."""

import numpy as np
from scipy.ndimage import binary_erosion

from pmriprep.interfaces.dce import _b1_interior_mask


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
