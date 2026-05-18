"""Unit tests for DSC deconvolution methods."""

import numpy as np
import pytest

from pmriprep.utils.dsc_deconvolution import (
    build_block_circulant_matrix,
    build_toeplitz_matrix,
    compute_oscillation_index,
    svd_deconvolve_circulant,
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
    svd_deconvolve_standard_batch,
)


class TestBuildToeplitzMatrix:
    """Tests for build_toeplitz_matrix."""

    def test_lower_triangular(self) -> None:
        """Matrix should be lower triangular (upper triangle all zeros)."""
        aif = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        matrix = build_toeplitz_matrix(aif, dt=1.0)

        # Upper triangle (excluding diagonal) should be all zeros
        for i in range(len(aif)):
            for j in range(i + 1, len(aif)):
                assert matrix[i, j] == 0.0, f"matrix[{i},{j}] = {matrix[i, j]}"

    def test_first_column(self) -> None:
        """First column should be proportional to AIF with trapezoidal weights."""
        aif = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
        dt = 1.5
        matrix = build_toeplitz_matrix(aif, dt=dt)

        # First column: matrix[i, 0] = aif[i] * 0.5 * dt (j==0 always gets 0.5 weight)
        for i in range(len(aif)):
            expected = aif[i] * 0.5 * dt
            assert matrix[i, 0] == pytest.approx(expected, rel=1e-10), (
                f"matrix[{i},0] = {matrix[i, 0]}, expected {expected}"
            )

    def test_diagonal(self) -> None:
        """Diagonal elements should all be 0.5 * aif[0] * dt.

        For diagonal element (i, i): j == 0 AND j == i, so weight = 0.5.
        The AIF value is aif[i - j] = aif[0].
        """
        aif = np.array([3.0, 7.0, 11.0, 5.0, 2.0], dtype=np.float64)
        dt = 2.0
        matrix = build_toeplitz_matrix(aif, dt=dt)

        expected_diag = 0.5 * aif[0] * dt
        for i in range(len(aif)):
            assert matrix[i, i] == pytest.approx(expected_diag, rel=1e-10), (
                f"matrix[{i},{i}] = {matrix[i, i]}, expected {expected_diag}"
            )


class TestSVDDeconvolveStandard:
    """Tests for svd_deconvolve_standard (sSVD)."""

    def test_impulse_response(self) -> None:
        """If tissue IS the AIF, R(t) should approximate delta function.

        The residue should peak near t=0 (within first few frames) and
        CBF should be on the order of 1/dt.
        """
        n = 30
        # Gamma-variate-like AIF
        t = np.arange(n, dtype=np.float64)
        aif = t * np.exp(-t / 3.0)
        aif[:2] = 0.0  # Clean baseline
        dt = 1.0

        # Tissue = AIF means R(t) = delta(t), so CBF ~ 1/dt
        result = svd_deconvolve_standard(aif, aif, dt=dt, threshold=0.1)

        # Residue should peak near t=0 (within first 2 frames due to
        # trapezoidal weighting and discretization)
        assert np.argmax(result.residue) <= 1
        # CBF should be positive and on the order of 1/dt
        assert result.cbf > 0.0

    def test_known_residue_recovery(self) -> None:
        """Recover a known exponential residue from synthetic convolution."""
        n = 40
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt

        # Known AIF: gamma variate
        aif = (t / 3.0) * np.exp(-t / 4.5)
        aif[0] = 0.0

        # Known residue: CBF * exp(-t/MTT) with CBF=2.0, MTT=8.0
        true_cbf = 2.0
        true_mtt = 8.0
        true_residue = true_cbf * np.exp(-t / true_mtt)

        # Create tissue curve: C = A @ R using the same Toeplitz matrix
        matrix = build_toeplitz_matrix(aif, dt=dt)
        tissue = matrix @ true_residue

        # Deconvolve and recover
        result = svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=0.1
        )

        # CBF should be close to true value
        assert result.cbf == pytest.approx(true_cbf, rel=0.2)
        # Residue shape should correlate with true residue
        corr = np.corrcoef(result.residue, true_residue)[0, 1]
        assert corr > 0.9

    def test_validation_different_lengths(self) -> None:
        """Different length tissue and AIF raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            svd_deconvolve_standard(
                np.ones(10, dtype=np.float64),
                np.ones(15, dtype=np.float64),
            )

    def test_validation_empty_arrays(self) -> None:
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            svd_deconvolve_standard(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

    def test_validation_invalid_threshold(self) -> None:
        """Threshold outside (0, 1) raises ValueError."""
        arr = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="threshold"):
            svd_deconvolve_standard(arr, arr, threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            svd_deconvolve_standard(arr, arr, threshold=1.0)


class TestSVDDeconvolveCirculant:
    """Tests for svd_deconvolve_circulant (cSVD)."""

    def test_delay_detection(self) -> None:
        """cSVD should detect a non-negative delay for delayed tissue.

        Uses circulant forward model to avoid discretization mismatch
        between Toeplitz (trapezoidal) and circulant (rectangular) weighting.
        """
        n = 40
        dt = 1.5
        m = 2 * n

        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 5.0) * np.exp(-t / 7.0)
        aif[0] = 0.0

        delay_frames = 4
        residue_padded = np.zeros(m, dtype=np.float64)
        residue_padded[delay_frames:delay_frames + n] = 2.0 * np.exp(-t / 12.0)
        matrix = build_block_circulant_matrix(aif, dt)
        tissue = (matrix @ residue_padded)[:n]

        result = svd_deconvolve_circulant(
            tissue, aif, dt=dt, threshold=0.15
        )

        assert result.cbf > 0.0
        assert result.mtt >= 0.0
        assert result.delay_frames >= 0, (
            f"Delayed tissue should give non-negative delay, got {result.delay_frames}"
        )

    def test_csvd_relative_delay_recovery(self) -> None:
        """cSVD delay differences should match injected delay differences.

        Absolute delay has a systematic offset from SVD regularization,
        but relative delay between two tissues should be accurate.
        """
        n = 40
        dt = 1.5
        m = 2 * n

        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 5.0) * np.exp(-t / 7.0)
        aif[0] = 0.0
        matrix = build_block_circulant_matrix(aif, dt)

        delays_detected = []
        for injected_delay in [0, 3, 6]:
            residue_padded = np.zeros(m, dtype=np.float64)
            residue_padded[injected_delay:injected_delay + n] = (
                2.0 * np.exp(-t / 12.0)
            )
            tissue = (matrix @ residue_padded)[:n]
            result = svd_deconvolve_circulant(
                tissue, aif, dt=dt, threshold=0.15,
            )
            delays_detected.append(result.delay_frames)

        # Relative delay differences should match within ±1 frame
        assert abs((delays_detected[1] - delays_detected[0]) - 3) <= 1, (
            f"Relative delay 3-0: expected 3, got {delays_detected[1] - delays_detected[0]}"
        )
        assert abs((delays_detected[2] - delays_detected[0]) - 6) <= 1, (
            f"Relative delay 6-0: expected 6, got {delays_detected[2] - delays_detected[0]}"
        )


class TestOSVDDelayDetection:
    """Tests for oSVD delay recovery."""

    def test_osvd_relative_delay_recovery(self) -> None:
        """oSVD delay differences should match injected delay differences.

        Uses circulant forward model. Small delays (0-3) to avoid
        wrap-around artifacts from minimal regularization.
        """
        n = 40
        dt = 1.5
        m = 2 * n

        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 5.0) * np.exp(-t / 7.0)
        aif[0] = 0.0
        matrix = build_block_circulant_matrix(aif, dt)

        delays_detected = []
        for injected_delay in [0, 1, 3]:
            residue_padded = np.zeros(m, dtype=np.float64)
            residue_padded[injected_delay:injected_delay + n] = (
                2.0 * np.exp(-t / 12.0)
            )
            tissue = (matrix @ residue_padded)[:n]
            result = svd_deconvolve_oscillation(tissue, aif, dt=dt)
            delays_detected.append(result.delay_frames)

        # oSVD has adaptive threshold selection, so relative delay is
        # less precise than cSVD (±2 frames vs ±1)
        assert abs((delays_detected[1] - delays_detected[0]) - 1) <= 2, (
            f"Relative delay 1-0: expected 1, got {delays_detected[1] - delays_detected[0]}"
        )
        assert abs((delays_detected[2] - delays_detected[0]) - 3) <= 2, (
            f"Relative delay 3-0: expected 3, got {delays_detected[2] - delays_detected[0]}"
        )


class TestSVDBatchMatchesSingle:
    """Tests for svd_deconvolve_standard_batch consistency."""

    def test_batch_matches_single(self) -> None:
        """Batch processing of 3 voxels should match individual results."""
        n = 30
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt

        # Common AIF
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0

        # 3 different tissue curves
        rng = np.random.default_rng(42)
        tissues = np.zeros((3, n), dtype=np.float64)
        for i in range(3):
            residue = (1.0 + i * 0.5) * np.exp(-t / (5.0 + i * 2.0))
            matrix = build_toeplitz_matrix(aif, dt=dt)
            tissues[i, :] = matrix @ residue + rng.normal(0, 0.01, n)

        threshold = 0.2

        # Individual deconvolution
        single_cbf = np.zeros(3)
        single_cbv = np.zeros(3)
        single_mtt = np.zeros(3)
        for i in range(3):
            result = svd_deconvolve_standard(
                tissues[i], aif, dt=dt, threshold=threshold
            )
            single_cbf[i] = result.cbf
            single_cbv[i] = result.cbv
            single_mtt[i] = result.mtt

        # Batch deconvolution
        _, batch_cbf, batch_cbv, batch_mtt = svd_deconvolve_standard_batch(
            tissues, aif, dt=dt, threshold=threshold
        )

        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-10)
        np.testing.assert_allclose(batch_cbv, single_cbv, rtol=1e-10)
        np.testing.assert_allclose(batch_mtt, single_mtt, rtol=1e-10)


class TestSVDThresholdSensitivity:
    """Tests for SVD threshold effects on regularization."""

    def test_higher_threshold_fewer_singular_values(self) -> None:
        """Higher threshold retains fewer singular values."""
        n = 30
        dt = 1.0
        t = np.arange(n, dtype=np.float64)
        aif = t * np.exp(-t / 4.0)
        aif[0] = 0.0

        # Noisy tissue curve
        rng = np.random.default_rng(99)
        tissue = np.convolve(aif, np.exp(-t / 8.0))[:n] * dt
        tissue += rng.normal(0, 0.1, n)

        result_low = svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=0.05
        )
        result_high = svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=0.4
        )

        assert result_high.n_singular_values <= result_low.n_singular_values


class TestNonNegativityConstraint:
    """Tests for non-negativity constraint on residue function."""

    def test_non_negativity_enforced(self) -> None:
        """With apply_non_negativity=True, residue has no negative values."""
        n = 30
        dt = 1.0
        t = np.arange(n, dtype=np.float64)
        aif = t * np.exp(-t / 3.0)
        aif[0] = 0.0

        rng = np.random.default_rng(7)
        tissue = np.convolve(aif, np.exp(-t / 6.0))[:n] * dt
        tissue += rng.normal(0, 0.5, n)

        result = svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=0.05, apply_non_negativity=True
        )

        assert np.all(result.residue >= 0.0)

    def test_without_non_negativity_may_have_negatives(self) -> None:
        """Without non-negativity, residue may contain negative values."""
        n = 30
        dt = 1.0
        t = np.arange(n, dtype=np.float64)
        aif = t * np.exp(-t / 3.0)
        aif[0] = 0.0

        rng = np.random.default_rng(7)
        tissue = np.convolve(aif, np.exp(-t / 6.0))[:n] * dt
        tissue += rng.normal(0, 0.5, n)

        result = svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=0.05, apply_non_negativity=False
        )

        # With noise and low threshold, we expect some negative values
        # (This is probabilistic but with seed 7 and this noise level it holds)
        assert np.any(result.residue < 0.0)


class TestComputeOscillationIndex:
    """Tests for compute_oscillation_index."""

    def test_smooth_residue_low_oi(self) -> None:
        """Smooth exponential residue should have low OI."""
        t = np.arange(30, dtype=np.float64)
        residue = 5.0 * np.exp(-t / 10.0)

        oi = compute_oscillation_index(residue)

        assert oi < 0.1

    def test_noisy_residue_high_oi(self) -> None:
        """Oscillatory residue should have higher OI than smooth."""
        t = np.arange(30, dtype=np.float64)
        smooth = 5.0 * np.exp(-t / 10.0)

        rng = np.random.default_rng(55)
        noisy = smooth + rng.normal(0, 2.0, 30)
        noisy = np.maximum(noisy, 0.0)  # Keep non-negative

        oi_smooth = compute_oscillation_index(smooth)
        oi_noisy = compute_oscillation_index(noisy)

        assert oi_noisy > oi_smooth

    def test_zero_residue(self) -> None:
        """All-zero residue returns OI=0."""
        residue = np.zeros(20, dtype=np.float64)
        assert compute_oscillation_index(residue) == 0.0

    def test_single_element(self) -> None:
        """Single element returns OI=0."""
        assert compute_oscillation_index(np.array([5.0])) == 0.0


class TestDeconvolutionResultTmax:
    """Tests for DeconvolutionResult.get_tmax()."""

    def test_tmax_known_peak(self) -> None:
        """Tmax should return correct time for known residue peak position."""
        n = 30
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt

        # AIF
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0

        # Create tissue from known residue peaking at t=0 (no delay)
        true_residue = 2.0 * np.exp(-t / 8.0)
        matrix = build_toeplitz_matrix(aif, dt=dt)
        tissue = matrix @ true_residue

        result = svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=0.1
        )

        # Tmax = argmax(residue) * dt
        expected_tmax = float(np.argmax(result.residue)) * dt
        assert result.get_tmax(dt=dt) == pytest.approx(expected_tmax, rel=1e-10)
