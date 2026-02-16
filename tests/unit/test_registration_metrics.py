"""Unit tests for registration quality metrics."""

import numpy as np
import pytest

from pmriprep.utils.registration import (
    calculate_correlation,
    calculate_dice,
    calculate_nmi,
    compute_aif_preservation,
    compute_aif_preservation_with_auc,
)


class TestCalculateDice:
    """Tests for DICE coefficient calculation."""

    def test_perfect_overlap(self) -> None:
        """Identical binary masks produce DICE = 1.0."""
        mask = np.ones((10, 10), dtype=np.float64)
        assert calculate_dice(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Disjoint masks produce DICE = 0.0."""
        mask1 = np.zeros((10, 10), dtype=np.float64)
        mask2 = np.zeros((10, 10), dtype=np.float64)
        mask1[:5, :] = 1.0
        mask2[5:, :] = 1.0
        assert calculate_dice(mask1, mask2) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """50% overlap gives expected DICE value."""
        # mask1: rows 0-7, mask2: rows 4-7 (overlap = rows 4-7 = 4 rows)
        # |A| = 8*10 = 80, |B| = 4*10 = 40, |A & B| = 4*10 = 40
        # DICE = 2*40 / (80 + 40) = 80/120 = 2/3
        mask1 = np.zeros((10, 10), dtype=np.float64)
        mask2 = np.zeros((10, 10), dtype=np.float64)
        mask1[:8, :] = 1.0
        mask2[4:8, :] = 1.0
        assert calculate_dice(mask1, mask2) == pytest.approx(2.0 / 3.0)

    def test_both_empty(self) -> None:
        """Two empty masks produce DICE = 0.0."""
        mask = np.zeros((10, 10), dtype=np.float64)
        assert calculate_dice(mask, mask) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self) -> None:
        """Different shaped masks raise ValueError."""
        mask1 = np.ones((10, 10), dtype=np.float64)
        mask2 = np.ones((10, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="shapes must match"):
            calculate_dice(mask1, mask2)


class TestCalculateCorrelation:
    """Tests for Pearson correlation coefficient."""

    def test_perfect_correlation(self) -> None:
        """Identical arrays produce correlation = 1.0."""
        image = np.random.default_rng(42).random((10, 10)).astype(np.float64)
        assert calculate_correlation(image, image) == pytest.approx(1.0)

    def test_anti_correlated(self) -> None:
        """Negated array produces correlation close to -1.0."""
        rng = np.random.default_rng(42)
        image = rng.random((10, 10)).astype(np.float64)
        assert calculate_correlation(image, -image) == pytest.approx(-1.0)

    def test_with_mask(self) -> None:
        """Correlation computed only over masked voxels."""
        rng = np.random.default_rng(42)
        image1 = rng.random((10, 10)).astype(np.float64)
        image2 = rng.random((10, 10)).astype(np.float64)

        # Create a mask covering only the top half
        mask = np.zeros((10, 10), dtype=np.float64)
        mask[:5, :] = 1.0

        corr_masked = calculate_correlation(image1, image2, mask=mask)

        # Manually compute correlation over masked region
        flat1 = image1[:5, :].ravel()
        flat2 = image2[:5, :].ravel()
        expected = float(np.corrcoef(flat1, flat2)[0, 1])
        assert corr_masked == pytest.approx(expected)

    def test_constant_arrays_return_zero(self) -> None:
        """Constant arrays (zero variance) return 0.0 instead of NaN."""
        image = np.ones((10, 10), dtype=np.float64)
        assert calculate_correlation(image, image) == pytest.approx(0.0)


class TestCalculateNMI:
    """Tests for normalized mutual information."""

    def test_identical_images_high_nmi(self) -> None:
        """Identical images produce NMI = 1.0 (H(A,B) = H(A) = H(B))."""
        rng = np.random.default_rng(42)
        image = rng.random((100, 100)).astype(np.float64)
        nmi = calculate_nmi(image, image)
        assert nmi == pytest.approx(1.0, abs=0.05)

    def test_uncorrelated_lower_nmi(self) -> None:
        """Independent random images produce lower NMI."""
        rng = np.random.default_rng(42)
        image1 = rng.random((100, 100)).astype(np.float64)
        image2 = rng.random((100, 100)).astype(np.float64)
        nmi_identical = calculate_nmi(image1, image1)
        nmi_random = calculate_nmi(image1, image2)
        assert nmi_random < nmi_identical

    def test_with_mask(self) -> None:
        """NMI is computed only over masked voxels."""
        rng = np.random.default_rng(42)
        image1 = rng.random((100, 100)).astype(np.float64)
        image2 = image1 * 2.0 + 0.5  # linearly related

        mask = np.zeros((100, 100), dtype=np.float64)
        mask[:50, :] = 1.0

        nmi = calculate_nmi(image1, image2, mask=mask)
        # Linearly related images should have NMI close to 1.0
        assert nmi == pytest.approx(1.0, abs=0.05)


class TestComputeAifPreservation:
    """Tests for AIF peak preservation ratio."""

    def test_perfect_preservation(self) -> None:
        """Identical AIFs produce peak ratio of 1.0."""
        aif = np.array([0.0, 0.5, 1.0, 0.8, 0.3], dtype=np.float64)
        assert compute_aif_preservation(aif, aif) == pytest.approx(1.0)

    def test_zero_peak_returns_zero(self) -> None:
        """Zero peak in before-AIF returns 0.0."""
        aif_before = np.zeros(5, dtype=np.float64)
        aif_after = np.array([0.0, 0.1, 0.2, 0.1, 0.0], dtype=np.float64)
        assert compute_aif_preservation(aif_before, aif_after) == pytest.approx(0.0)

    def test_reduced_peak(self) -> None:
        """Halved AIF produces peak ratio of 0.5."""
        aif_before = np.array([0.0, 0.5, 1.0, 0.8, 0.3], dtype=np.float64)
        aif_after = aif_before * 0.5
        assert compute_aif_preservation(aif_before, aif_after) == pytest.approx(0.5)


class TestComputeAifPreservationWithAuc:
    """Tests for comprehensive AIF preservation metrics."""

    def test_good_preservation(self) -> None:
        """Nearly identical AIFs yield quality='good'."""
        aif = np.array([0.0, 0.5, 1.0, 0.8, 0.3], dtype=np.float64)
        result = compute_aif_preservation_with_auc(aif, aif)
        assert result["quality"] == "good"
        assert result["peak_ratio"] == pytest.approx(1.0)
        assert result["auc_ratio"] == pytest.approx(1.0)
        assert result["peak_time_shift"] == 0

    def test_degraded_preservation(self) -> None:
        """Substantially reduced AIF yields quality='degraded'."""
        aif_before = np.array([0.0, 0.5, 1.0, 0.8, 0.3], dtype=np.float64)
        aif_after = aif_before * 0.5
        result = compute_aif_preservation_with_auc(aif_before, aif_after)
        assert result["quality"] == "degraded"
        assert result["peak_ratio"] == pytest.approx(0.5)

    def test_peak_time_shift(self) -> None:
        """Shifted AIF reports correct peak time shift."""
        aif_before = np.array([0.0, 0.0, 1.0, 0.5, 0.0], dtype=np.float64)
        aif_after = np.array([0.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float64)
        result = compute_aif_preservation_with_auc(aif_before, aif_after)
        # Peak moves from index 2 to index 3
        assert result["peak_time_shift"] == 1
