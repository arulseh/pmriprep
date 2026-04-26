"""Extended unit tests for DSC deconvolution methods.

Tests for oSVD, Tikhonov, block-circulant matrix, dispersion correction,
batch methods, and perfusion parameter extraction.
"""

import numpy as np
import pytest

from pmriprep.utils.dsc_deconvolution import (
    apply_dispersion_correction,
    build_block_circulant_matrix,
    build_toeplitz_matrix,
    compute_oscillation_index,
    extract_perfusion_params,
    extract_perfusion_params_batch,
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

    def test_tikhonov_gcv_score_finite(self) -> None:
        """GCV score is finite and positive; lambda_optimal is positive."""
        aif, tissue, _, _ = _make_synthetic_data()
        result = tikhonov_deconvolve(tissue, aif, dt=1.5)
        assert np.isfinite(result.gcv_score)
        assert result.gcv_score > 0
        assert result.lambda_optimal > 0

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

        _, batch_cbf, batch_cbv, batch_mtt, _ = tikhonov_deconvolve_batch(
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


# =============================================================================
# Dispersion Correction Tests
# =============================================================================


class TestDispersionCorrection:
    """Tests for apply_dispersion_correction."""

    def _make_dispersed_aif(self) -> np.ndarray:
        """Create a gamma-variate AIF that has some inherent width."""
        n = 60
        t = np.arange(n, dtype=np.float64)
        aif = (t / 5.0) * np.exp(-t / 8.0)
        aif[0] = 0.0
        return aif

    def test_dispersion_correction_narrower_aif(self) -> None:
        """Corrected AIF has a higher peak than original (sharper bolus)."""
        aif = self._make_dispersed_aif()
        result = apply_dispersion_correction(aif, dt=1.0)

        # Corrected peak should be higher (dispersion removed = sharper)
        assert result.peak_corrected >= result.peak_original

    def test_dispersion_correction_preserve_auc(self) -> None:
        """With preserve_auc=True, corrected AUC matches original within 20%."""
        aif = self._make_dispersed_aif()
        result = apply_dispersion_correction(aif, dt=1.0, preserve_auc=True)

        ratio = result.auc_corrected / result.auc_original
        assert ratio == pytest.approx(1.0, abs=0.2)

    def test_dispersion_correction_result_fields(self) -> None:
        """DispersionCorrectionResult has all expected fields; dispersion_factor >= ~1.0."""
        aif = self._make_dispersed_aif()
        result = apply_dispersion_correction(aif, dt=1.0)

        # Check all fields exist and are reasonable
        assert result.aif_corrected is not None
        assert len(result.aif_corrected) == len(aif)
        assert result.vtf is not None
        assert result.auc_original > 0
        assert result.auc_corrected > 0
        assert result.peak_original > 0
        assert result.peak_corrected > 0
        assert result.fwhm_original >= 0
        assert result.fwhm_corrected >= 0
        # Dispersion factor should be >= ~1.0 (corrected peak >= original)
        assert result.dispersion_factor >= 0.95


# =============================================================================
# Perfusion Parameter Extraction Tests
# =============================================================================


class TestPerfusionParamExtraction:
    """Tests for extract_perfusion_params and extract_perfusion_params_batch."""

    def test_extract_perfusion_params_exponential(self) -> None:
        """Known exponential residue yields expected CBF, MTT, and CBV."""
        true_cbf = 2.0
        true_mtt = 8.0
        dt = 1.5
        n = 40
        t = np.arange(n, dtype=np.float64) * dt
        residue = true_cbf * np.exp(-t / true_mtt)

        params = extract_perfusion_params(residue, dt=dt)

        # CBF = max(R) should be exactly true_cbf (peak at t=0)
        assert params["cbf"] == pytest.approx(true_cbf, rel=1e-10)
        assert params["r_max"] == pytest.approx(true_cbf, rel=1e-10)

        # MTT should be within 50% of true_mtt (trapezoidal integration
        # of a truncated exponential introduces some error)
        assert params["mtt"] == pytest.approx(true_mtt, rel=0.5)

        # CBV = integral of R(t) dt = CBF * MTT for exponential
        # (approximately, due to truncation)
        expected_cbv = true_cbf * true_mtt * (1 - np.exp(-t[-1] / true_mtt))
        assert params["cbv"] == pytest.approx(expected_cbv, rel=0.1)

        # TTP should be 0 (peak at first frame)
        assert params["ttp"] == 0
        assert params["tmax"] == pytest.approx(0.0, abs=1e-10)

    def test_extract_perfusion_params_keys(self) -> None:
        """Returned dict has all expected keys."""
        residue = np.exp(-np.arange(20, dtype=np.float64) / 5.0)
        params = extract_perfusion_params(residue, dt=1.0)

        expected_keys = {"cbf", "cbv", "mtt", "tmax", "ttp", "auc", "r_max"}
        assert set(params.keys()) == expected_keys

    def test_extract_perfusion_params_batch_matches_single(self) -> None:
        """Batch extraction matches looped single-voxel extraction."""
        dt = 1.5
        n = 30
        t = np.arange(n, dtype=np.float64) * dt

        residues = np.zeros((3, n), dtype=np.float64)
        for i in range(3):
            residues[i, :] = (1.0 + i) * np.exp(-t / (5.0 + i * 3.0))

        # Single-voxel extraction
        single_cbf = np.zeros(3)
        single_cbv = np.zeros(3)
        single_mtt = np.zeros(3)
        single_tmax = np.zeros(3)
        for i in range(3):
            p = extract_perfusion_params(residues[i], dt=dt)
            single_cbf[i] = p["cbf"]
            single_cbv[i] = p["cbv"]
            single_mtt[i] = p["mtt"]
            single_tmax[i] = p["tmax"]

        # Batch extraction
        batch_cbf, batch_cbv, batch_mtt, batch_tmax = (
            extract_perfusion_params_batch(residues, dt=dt)
        )

        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-10)
        np.testing.assert_allclose(batch_cbv, single_cbv, rtol=1e-10)
        np.testing.assert_allclose(batch_mtt, single_mtt, rtol=1e-10)
        np.testing.assert_allclose(batch_tmax, single_tmax, rtol=1e-10)
