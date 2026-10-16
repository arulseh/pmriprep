"""Unit tests for DSC deconvolution methods."""

import numpy as np
import pytest

from pmriprep.utils.dsc_deconvolution import (
    build_block_circulant_matrix,
    build_toeplitz_matrix,
    build_volterra_matrix,
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
        """First column = (1, 4, 1)/6-smoothed AIF (interior) times dt.

        Endpoints are left unsmoothed: matrix[0, 0] = aif[0]*dt and
        matrix[n-1, 0] = aif[n-1]*dt.
        """
        aif = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
        dt = 1.5
        matrix = build_toeplitz_matrix(aif, dt=dt)

        smooth = aif.copy()
        smooth[1:-1] = (aif[:-2] + 4.0 * aif[1:-1] + aif[2:]) / 6.0
        for i in range(len(aif)):
            expected = smooth[i] * dt
            assert matrix[i, 0] == pytest.approx(expected, rel=1e-10), (
                f"matrix[{i},0] = {matrix[i, 0]}, expected {expected}"
            )

    def test_diagonal(self) -> None:
        """Diagonal elements all equal aif[0] * dt.

        The diagonal is the generating column's first entry (aif[i - j] = aif[0]),
        i.e. the unsmoothed first AIF sample.
        """
        aif = np.array([3.0, 7.0, 11.0, 5.0, 2.0], dtype=np.float64)
        dt = 2.0
        matrix = build_toeplitz_matrix(aif, dt=dt)

        expected_diag = aif[0] * dt
        for i in range(len(aif)):
            assert matrix[i, i] == pytest.approx(expected_diag, rel=1e-10), (
                f"matrix[{i},{i}] = {matrix[i, i]}, expected {expected_diag}"
            )

    def test_interior_141_stencil(self) -> None:
        """Interior first-column entries use the (1, 4, 1)/6 stencil."""
        aif = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 4.0], dtype=np.float64)
        dt = 1.0
        matrix = build_toeplitz_matrix(aif, dt=dt)
        for i in range(1, len(aif) - 1):
            expected = (aif[i - 1] + 4.0 * aif[i] + aif[i + 1]) / 6.0 * dt
            assert matrix[i, 0] == pytest.approx(expected, rel=1e-12)

    def test_requires_two_samples(self) -> None:
        """A single-sample AIF cannot be smoothed and must raise."""
        with pytest.raises(ValueError):
            build_toeplitz_matrix(np.array([1.0]), dt=1.0)


class TestBuildBlockCirculantMatrix:
    """Tests for build_block_circulant_matrix ((1, 4, 1)/6 generating column)."""

    def test_generating_column_141_and_boundary(self) -> None:
        """First column = (1,4,1)/6-smoothed AIF + one-sided + pad-leak terms."""
        aif = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 4.0], dtype=np.float64)
        n = len(aif)
        dt = 1.5
        matrix = build_block_circulant_matrix(aif, dt=dt)
        col = matrix[:, 0] / dt  # undo the dt scaling

        assert col[0] == pytest.approx((4 * aif[0] + aif[1]) / 6)
        for k in range(1, n - 1):
            assert col[k] == pytest.approx((aif[k - 1] + 4 * aif[k] + aif[k + 1]) / 6)
        assert col[n - 1] == pytest.approx((aif[n - 2] + 4 * aif[n - 1]) / 6)  # one-sided
        assert col[n] == pytest.approx(aif[n - 1] / 6)  # pad-leak into index N
        assert np.allclose(col[n + 1:], 0.0)  # zero pad beyond

    def test_shape_and_circulant(self) -> None:
        """Matrix is 2N x 2N and circulant (each column a cyclic shift)."""
        aif = np.array([1.0, 5.0, 2.0, 9.0], dtype=np.float64)
        n = len(aif)
        matrix = build_block_circulant_matrix(aif, dt=1.0)
        assert matrix.shape == (2 * n, 2 * n)
        assert np.allclose(matrix[:, 1], np.roll(matrix[:, 0], 1))

    @pytest.mark.parametrize("aif", [
        [0.6, 0.0], [0.0, 0.9, 0.4], [-0.2, 0.1, 1.0, -0.3],
    ])
    def test_zero_extended_convolution(self, aif: list[float]) -> None:
        """The operator matches independent smoothing and periodic convolution."""
        arterial = np.array(aif)
        size = 2 * arterial.size
        kernel = np.zeros(size)
        smoothed = np.convolve(arterial, np.array([1, 4, 1]) / 6)[1:]
        kernel[:smoothed.size] = smoothed
        residue = np.linspace(-0.2, 1, size)
        expected = np.array([
            sum(kernel[(i - j) % size] * residue[j] for j in range(size))
            for i in range(size)
        ]) * 1.5
        np.testing.assert_allclose(build_block_circulant_matrix(arterial, 1.5) @ residue, expected)

    def test_requires_two_samples(self) -> None:
        with pytest.raises(ValueError):
            build_block_circulant_matrix(np.array([1.0]), dt=1.0)


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

        # Create tissue curve: C = A @ R with the Volterra (product-integration)
        # operator that svd_deconvolve_standard inverts (matched forward).
        matrix = build_volterra_matrix(aif, dt=dt)
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

        Uses the same circulant operator to generate and deconvolve the curve.
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

        Uses the circulant forward model and searches the full residue for
        its peak after oscillation-based threshold selection.
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


class TestSignedResidue:
    """Unconstrained sSVD residues retain negative samples and their area."""

    def test_signed_solution_matches_truncated_inverse(self) -> None:
        """The default returned residue equals the linear truncated inverse."""
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix
        n = 30
        dt = 1.0
        t = np.arange(n, dtype=np.float64)
        aif = t * np.exp(-t / 3.0)
        aif[0] = 0.0

        rng = np.random.default_rng(7)
        tissue = np.convolve(aif, np.exp(-t / 6.0))[:n] * dt
        tissue += rng.normal(0, 0.5, n)

        result = svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=0.05
        )

        reference = np.linalg.pinv(build_volterra_matrix(aif, dt), rcond=0.05) @ tissue
        np.testing.assert_allclose(result.residue, reference, rtol=1e-10, atol=1e-12)
        assert result.cbv == pytest.approx(np.trapezoid(reference, dx=dt))

    def test_default_residue_preserves_negative_samples(self) -> None:
        """Noise-driven negative residue samples survive parameter extraction."""
        n = 30
        dt = 1.0
        t = np.arange(n, dtype=np.float64)
        aif = t * np.exp(-t / 3.0)
        aif[0] = 0.0

        rng = np.random.default_rng(7)
        tissue = np.convolve(aif, np.exp(-t / 6.0))[:n] * dt
        tissue += rng.normal(0, 0.5, n)

        result = svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=0.05
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

    def test_two_elements(self) -> None:
        """Two elements return OI=0 (the second difference needs >= 3 points)."""
        assert compute_oscillation_index(np.array([5.0, 3.0])) == 0.0

    def test_second_difference_formula(self) -> None:
        """OI is summed |R[k]-2R[k-1]+R[k-2]| / (L*max)."""
        residue = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        # second differences = -2, +2, -2 -> sum|.| = 6; OI = 6/(5*1) = 1.2
        assert compute_oscillation_index(residue) == pytest.approx(1.2)


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
