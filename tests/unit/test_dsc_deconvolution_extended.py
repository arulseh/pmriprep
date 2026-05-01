"""Extended unit tests for DSC deconvolution methods.

Tests for oSVD, Tikhonov, block-circulant matrix, and batch methods.
"""

import numpy as np
import pytest

from pmriprep.utils.dsc_deconvolution import (
    build_block_circulant_matrix,
    build_toeplitz_matrix,
    compute_oscillation_index,
    svd_deconvolve_circulant,
    svd_deconvolve_circulant_batch,
    svd_deconvolve_oscillation,
    svd_deconvolve_oscillation_batch,
    tikhonov_deconvolve,
    tikhonov_deconvolve_batch,
)


def _make_synthetic_data(
    n: int = 30,
    dt: float = 1.5,
    true_cbf: float = 2.0,
    true_mtt: float = 8.0,
    seed: int = 42,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Create synthetic AIF + tissue curve pair with known ground truth."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) * dt
    # Gamma-variate AIF
    aif = (t / 3.0) * np.exp(-t / 5.0)
    aif[0] = 0.0
    # Exponential residue
    true_residue = true_cbf * np.exp(-t / true_mtt)
    # Create tissue via Toeplitz convolution
    matrix = build_toeplitz_matrix(aif, dt=dt)
    tissue = matrix @ true_residue
    return aif, tissue, t, true_residue


# =============================================================================
# Block-Circulant Matrix Tests
# =============================================================================


class TestBlockCirculantMatrix:
    """Tests for build_block_circulant_matrix."""

    def test_block_circulant_matrix_size(self) -> None:
        """AIF of length N produces a matrix of shape (2N-1, 2N-1)."""
        n = 10
        aif = np.arange(n, dtype=np.float64) * 0.5
        matrix = build_block_circulant_matrix(aif, dt=1.0)
        expected_size = 2 * n - 1
        assert matrix.shape == (expected_size, expected_size)

    def test_block_circulant_matrix_circulant_structure(self) -> None:
        """Each row is a cyclic shift of the previous row."""
        n = 8
        aif = np.arange(1, n + 1, dtype=np.float64)
        dt = 1.0
        matrix = build_block_circulant_matrix(aif, dt=dt)
        m = 2 * n - 1

        # For a circulant matrix C, C[i, j] = C[0, (j - i) mod m]
        # Check rows 0, 1, 2 against this pattern
        for i in range(3):
            for j in range(m):
                expected = matrix[0, (j - i) % m]
                assert matrix[i, j] == pytest.approx(expected, rel=1e-12), (
                    f"matrix[{i},{j}] = {matrix[i, j]}, "
                    f"expected {expected} (cyclic shift of row 0)"
                )


# =============================================================================
# oSVD Single Voxel Tests
# =============================================================================


class TestOSVDSingleVoxel:
    """Tests for svd_deconvolve_oscillation (oSVD)."""

    def test_osvd_converged(self) -> None:
        """Synthetic data produces a converged result with OI at or near target."""
        aif, tissue, _, _ = _make_synthetic_data()
        result = svd_deconvolve_oscillation(
            tissue, aif, dt=1.5, oi_target=0.065,
        )
        assert result.converged is True
        # OI should be at or below the target (with some tolerance for the
        # discrete threshold grid)
        assert result.oscillation_index <= result.oi_target + 0.01

    def test_osvd_stricter_oi_higher_threshold(self) -> None:
        """Stricter OI target requires equal or higher SVD threshold."""
        aif, tissue, _, _ = _make_synthetic_data()

        result_relaxed = svd_deconvolve_oscillation(
            tissue, aif, dt=1.5, oi_target=0.10,
        )
        result_strict = svd_deconvolve_oscillation(
            tissue, aif, dt=1.5, oi_target=0.03,
        )

        # A stricter OI target needs more regularization (higher threshold)
        assert result_strict.threshold_used >= result_relaxed.threshold_used

    def test_osvd_recovers_cbf(self) -> None:
        """oSVD recovers CBF within 50% of the known ground truth."""
        true_cbf = 2.0
        aif, tissue, _, _ = _make_synthetic_data(true_cbf=true_cbf)
        result = svd_deconvolve_oscillation(tissue, aif, dt=1.5)
        assert result.cbf == pytest.approx(true_cbf, rel=0.5)

    def test_osvd_validation(self) -> None:
        """Empty arrays and length mismatch raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            svd_deconvolve_oscillation(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="same length"):
            svd_deconvolve_oscillation(
                np.ones(10, dtype=np.float64),
                np.ones(15, dtype=np.float64),
            )


# =============================================================================
# Tikhonov Single Voxel Tests
# =============================================================================


class TestTikhonovSingleVoxel:
    """Tests for tikhonov_deconvolve."""

    def test_tikhonov_recovers_cbf(self) -> None:
        """Tikhonov recovers CBF within 50% of the known ground truth."""
        true_cbf = 2.0
        aif, tissue, _, _ = _make_synthetic_data(true_cbf=true_cbf)
        result = tikhonov_deconvolve(tissue, aif, dt=1.5)
        assert result.cbf == pytest.approx(true_cbf, rel=0.5)

    def test_tikhonov_lcurve_result_finite(self) -> None:
        """L-curve result fields are finite and consistent."""
        aif, tissue, _, _ = _make_synthetic_data()
        result = tikhonov_deconvolve(tissue, aif, dt=1.5)
        assert np.isfinite(result.residual_norm)
        assert np.isfinite(result.solution_norm)
        assert result.residual_norm >= 0
        assert result.solution_norm >= 0
        assert result.lambda_optimal > 0
        assert isinstance(result.lcurve_corner_found, bool)

    def test_tikhonov_validation(self) -> None:
        """Empty arrays, length mismatch, and invalid lambda raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            tikhonov_deconvolve(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="same length"):
            tikhonov_deconvolve(
                np.ones(10, dtype=np.float64),
                np.ones(15, dtype=np.float64),
            )
        with pytest.raises(ValueError, match="lambda_min"):
            tikhonov_deconvolve(
                np.ones(10, dtype=np.float64),
                np.ones(10, dtype=np.float64),
                lambda_min=1.0,
                lambda_max=0.5,
            )

    def test_tikhonov_higher_regularization_smoother(self) -> None:
        """Higher regularization (larger lambda) produces a smoother residue.

        Uses noisy data so the effect of regularization on oscillation is
        clearly visible.
        """
        n = 30
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0
        true_residue = 2.0 * np.exp(-t / 8.0)
        matrix = build_toeplitz_matrix(aif, dt=dt)
        rng = np.random.default_rng(99)
        tissue = matrix @ true_residue + rng.normal(0, 0.5, n)

        # Strong regularization: narrow range near lambda_max
        result_strong = tikhonov_deconvolve(
            tissue, aif, dt=dt, lambda_min=0.5, lambda_max=1.0,
        )
        # Weak regularization: narrow range near lambda_min
        result_weak = tikhonov_deconvolve(
            tissue, aif, dt=dt, lambda_min=1e-6, lambda_max=1e-4,
        )

        oi_strong = compute_oscillation_index(result_strong.residue)
        oi_weak = compute_oscillation_index(result_weak.residue)

        # With noisy data, higher regularization should give smoother residue
        assert oi_strong <= oi_weak


# =============================================================================
# Batch Methods Tests
# =============================================================================


class TestBatchMethods:
    """Tests for batch deconvolution methods matching single-voxel results."""

    def _make_batch_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Create 3 different tissue curves with a common AIF."""
        n = 30
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0
        tissues = np.zeros((3, n), dtype=np.float64)
        matrix = build_toeplitz_matrix(aif, dt=dt)
        for i in range(3):
            residue = (1.0 + i * 0.5) * np.exp(-t / (5.0 + i * 2.0))
            tissues[i, :] = matrix @ residue
        return tissues, aif, dt

    def test_circulant_batch_matches_single(self) -> None:
        """Batch cSVD produces the same CBF/CBV/MTT as looped single calls."""
        tissues, aif, dt = self._make_batch_data()
        threshold = 0.2

        single_cbf = np.zeros(3)
        single_cbv = np.zeros(3)
        single_mtt = np.zeros(3)
        for i in range(3):
            result = svd_deconvolve_circulant(
                tissues[i], aif, dt=dt, threshold=threshold,
            )
            single_cbf[i] = result.cbf
            single_cbv[i] = result.cbv
            single_mtt[i] = result.mtt

        _, batch_cbf, batch_cbv, batch_mtt, _ = svd_deconvolve_circulant_batch(
            tissues, aif, dt=dt, threshold=threshold,
        )

        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-8)
        np.testing.assert_allclose(batch_cbv, single_cbv, rtol=1e-8)
        np.testing.assert_allclose(batch_mtt, single_mtt, rtol=1e-8)

    def test_oscillation_batch_matches_single(self) -> None:
        """Batch oSVD produces the same CBF/CBV/MTT as looped single calls."""
        tissues, aif, dt = self._make_batch_data()

        single_cbf = np.zeros(3)
        single_cbv = np.zeros(3)
        single_mtt = np.zeros(3)
        for i in range(3):
            result = svd_deconvolve_oscillation(tissues[i], aif, dt=dt)
            single_cbf[i] = result.cbf
            single_cbv[i] = result.cbv
            single_mtt[i] = result.mtt

        _, batch_cbf, batch_cbv, batch_mtt, _, _, _ = (
            svd_deconvolve_oscillation_batch(tissues, aif, dt=dt)
        )

        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-8)
        np.testing.assert_allclose(batch_cbv, single_cbv, rtol=1e-8)
        np.testing.assert_allclose(batch_mtt, single_mtt, rtol=1e-8)

    def test_tikhonov_batch_matches_single(self) -> None:
        """Batch Tikhonov produces the same CBF/CBV/MTT as looped single calls."""
        tissues, aif, dt = self._make_batch_data()

        single_cbf = np.zeros(3)
        single_cbv = np.zeros(3)
        single_mtt = np.zeros(3)
        for i in range(3):
            result = tikhonov_deconvolve(tissues[i], aif, dt=dt)
            single_cbf[i] = result.cbf
            single_cbv[i] = result.cbv
            single_mtt[i] = result.mtt

        _, batch_cbf, batch_cbv, batch_mtt, _, _ = tikhonov_deconvolve_batch(
            tissues, aif, dt=dt,
        )

        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-8)
        np.testing.assert_allclose(batch_cbv, single_cbv, rtol=1e-8)
        np.testing.assert_allclose(batch_mtt, single_mtt, rtol=1e-8)

    def test_oscillation_batch_with_mask(self) -> None:
        """Masked voxels have zero CBF; unmasked voxels match single-voxel."""
        n = 30
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0

        tissues = np.zeros((5, n), dtype=np.float64)
        matrix = build_toeplitz_matrix(aif, dt=dt)
        for i in range(5):
            residue = (1.0 + i * 0.3) * np.exp(-t / (6.0 + i))
            tissues[i, :] = matrix @ residue

        mask = np.array([True, True, False, True, False])

        _, batch_cbf, _, _, _, _, _ = svd_deconvolve_oscillation_batch(
            tissues, aif, dt=dt, mask=mask,
        )

        # Masked-out voxels should have zero CBF
        assert batch_cbf[2] == 0.0
        assert batch_cbf[4] == 0.0

        # Unmasked voxels should match single-voxel results
        for i in [0, 1, 3]:
            single = svd_deconvolve_oscillation(tissues[i], aif, dt=dt)
            assert batch_cbf[i] == pytest.approx(single.cbf, rel=1e-8)
