"""Unit tests for GPU/batched SVD deconvolution."""

import numpy as np
import pytest

from pmriprep.models.dsc.gpu_deconvolution import (
    GPUDeconvolutionResult,
    batched_svd_deconvolve,
    batched_svd_deconvolve_circulant,
    batched_svd_deconvolve_standard,
    get_optimal_batch_size,
)
from pmriprep.utils.dsc_deconvolution import (
    svd_deconvolve_circulant_batch,
    svd_deconvolve_standard_batch,
)


def _make_synthetic_batch(
    n_voxels: int = 5,
    n_timepoints: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create synthetic tissue curves and AIF for batch testing.

    Returns:
        Tuple of (tissue_curves, aif, dt).
    """
    rng = np.random.default_rng(seed)
    dt = 1.5
    t = np.arange(n_timepoints, dtype=np.float64) * dt

    # Gamma-variate AIF
    aif = (t / 3.0) * np.exp(-t / 5.0)
    aif[0] = 0.0

    # Create tissue curves via convolution with exponential residues
    tissue_curves = np.zeros((n_voxels, n_timepoints), dtype=np.float64)
    for i in range(n_voxels):
        cbf = 1.0 + i * 0.5
        mtt = 5.0 + i * 2.0
        residue = cbf * np.exp(-t / mtt)
        tissue_curves[i, :] = np.convolve(aif, residue, mode="full")[:n_timepoints] * dt
        tissue_curves[i, :] += rng.normal(0, 0.01, n_timepoints)

    return tissue_curves, aif, dt


class TestStandardMethod:
    """Tests for standard (sSVD) GPU batched deconvolution."""

    def test_standard_method_shapes(self) -> None:
        """Output shapes match n_voxels for standard method."""
        tissue, aif, dt = _make_synthetic_batch(n_voxels=5, n_timepoints=30)

        result = batched_svd_deconvolve(
            tissue, aif, dt=dt, method="standard", use_gpu=False,
        )

        assert result.cbf.shape == (5,)
        assert result.cbv.shape == (5,)
        assert result.mtt.shape == (5,)
        assert result.tmax.shape == (5,)
        assert result.used_gpu is False
        assert result.delay is None  # Standard method has no delay

    def test_standard_matches_cpu_batch(self) -> None:
        """GPU standard deconvolution (CPU fallback) matches CPU batch results."""
        tissue, aif, dt = _make_synthetic_batch(n_voxels=5, n_timepoints=30)
        threshold = 0.2

        # CPU batch reference
        _, cpu_cbf, cpu_cbv, cpu_mtt = svd_deconvolve_standard_batch(
            tissue, aif, dt=dt, threshold=threshold,
        )

        # GPU function with CPU fallback
        gpu_result = batched_svd_deconvolve_standard(
            tissue, aif, dt=dt, threshold=threshold, use_gpu=False,
        )

        np.testing.assert_allclose(gpu_result.cbf, cpu_cbf, rtol=1e-10)
        np.testing.assert_allclose(gpu_result.cbv, cpu_cbv, rtol=1e-10)
        np.testing.assert_allclose(gpu_result.mtt, cpu_mtt, rtol=1e-10)


class TestCirculantMethod:
    """Tests for circulant (cSVD) GPU batched deconvolution."""

    def test_circulant_method_has_delay(self) -> None:
        """Circulant method returns delay estimates."""
        tissue, aif, dt = _make_synthetic_batch(n_voxels=5, n_timepoints=30)

        result = batched_svd_deconvolve(
            tissue, aif, dt=dt, method="circulant", use_gpu=False,
        )

        assert result.delay is not None
        assert result.delay.shape == (5,)

    def test_circulant_matches_cpu_batch(self) -> None:
        """GPU circulant deconvolution (CPU fallback) matches CPU batch results."""
        tissue, aif, dt = _make_synthetic_batch(n_voxels=5, n_timepoints=30)
        threshold = 0.2

        # CPU batch reference
        _, cpu_cbf, cpu_cbv, cpu_mtt, cpu_delay = svd_deconvolve_circulant_batch(
            tissue, aif, dt=dt, threshold=threshold,
        )

        # GPU function with CPU fallback
        gpu_result = batched_svd_deconvolve_circulant(
            tissue, aif, dt=dt, threshold=threshold, use_gpu=False,
        )

        np.testing.assert_allclose(gpu_result.cbf, cpu_cbf, rtol=1e-10)
        np.testing.assert_allclose(gpu_result.cbv, cpu_cbv, rtol=1e-10)
        np.testing.assert_allclose(gpu_result.mtt, cpu_mtt, rtol=1e-10)

    def test_csvd_tmax_equals_delay(self) -> None:
        """For cSVD, Tmax should equal delay (Calamante, Stroke 2010)."""
        tissue, aif, dt = _make_synthetic_batch(n_voxels=5, n_timepoints=30)

        result = batched_svd_deconvolve_circulant(
            tissue, aif, dt=dt, use_gpu=False,
        )

        assert result.delay is not None
        np.testing.assert_array_equal(result.tmax, result.delay)


class TestMaskAndResidues:
    """Tests for mask and residue functionality."""

    def test_with_mask(self) -> None:
        """Masked voxels have CBF = 0."""
        tissue, aif, dt = _make_synthetic_batch(n_voxels=5, n_timepoints=30)
        mask = np.array([True, True, False, True, False])

        result = batched_svd_deconvolve(
            tissue, aif, dt=dt, method="standard", mask=mask, use_gpu=False,
        )

        # Masked-out voxels (indices 2 and 4) should be zero
        assert result.cbf[2] == 0.0
        assert result.cbf[4] == 0.0
        # Unmasked voxels should be positive
        assert result.cbf[0] > 0.0
        assert result.cbf[1] > 0.0
        assert result.cbf[3] > 0.0

    def test_return_residues(self) -> None:
        """return_residues=True returns residue array with correct shape."""
        n_voxels = 5
        n_timepoints = 30
        tissue, aif, dt = _make_synthetic_batch(
            n_voxels=n_voxels, n_timepoints=n_timepoints,
        )

        result = batched_svd_deconvolve(
            tissue, aif, dt=dt, method="standard",
            return_residues=True, use_gpu=False,
        )

        assert result.residues is not None
        assert result.residues.shape == (n_voxels, n_timepoints)


class TestBatchSize:
    """Tests for batch size estimation."""

    def test_get_optimal_batch_size(self) -> None:
        """Without GPU, returns CPU default of 10000."""
        batch_size = get_optimal_batch_size(n_timepoints=30, method="standard")
        assert isinstance(batch_size, int)
        assert batch_size >= 100


class TestGPUDeconvolutionResultDataclass:
    """Tests for the GPUDeconvolutionResult dataclass."""

    def test_gpu_result_dataclass(self) -> None:
        """All fields are accessible on a manually constructed result."""
        result = GPUDeconvolutionResult(
            cbf=np.array([1.0, 2.0]),
            cbv=np.array([0.5, 1.0]),
            mtt=np.array([3.0, 4.0]),
            tmax=np.array([1.5, 3.0]),
            residues=None,
            delay=None,
            n_voxels_processed=2,
            used_gpu=False,
            threshold_used=0.2,
            n_singular_values_retained=10,
        )

        assert result.cbf.shape == (2,)
        assert result.n_voxels_processed == 2
        assert result.used_gpu is False
        assert result.threshold_used == pytest.approx(0.2)
        assert result.n_singular_values_retained == 10
        assert result.residues is None
        assert result.delay is None
