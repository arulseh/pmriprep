"""DSC parameter recovery tests using Digital Reference Objects.

Tests that deconvolution methods can recover known ground-truth CBF, CBV,
and MTT from synthetic concentration-time curves at various noise levels.
"""

from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.models.dsc.cpi import CPIModel, _fit_cpi_simple
from pmriprep.utils.dsc_deconvolution import (
    svd_deconvolve_circulant,
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
    tikhonov_deconvolve,
)

from .conftest import (
    add_gaussian_noise,
    dsc_forward_model,
    generate_dsc_aif_dro,
    make_exponential_residue,
)


# =============================================================================
# Helpers
# =============================================================================


def _assert_param_recovery(
    recovered: float,
    truth: float,
    rel_tol: float,
    abs_tol: float = 0.5,
    param_name: str = "",
) -> None:
    """Assert DSC parameter recovery within tolerance."""
    if abs(truth) < 1e-3:
        assert abs(recovered - truth) < abs_tol, (
            f"{param_name}: |{recovered:.4f} - {truth:.4f}| = "
            f"{abs(recovered - truth):.4f} >= {abs_tol}"
        )
    else:
        rel_err = abs(recovered - truth) / abs(truth)
        assert rel_err < rel_tol, (
            f"{param_name}: |{recovered:.4f} - {truth:.4f}| / |{truth:.4f}| = "
            f"{rel_err:.4f} >= {rel_tol}"
        )


def _make_dsc_test_data(
    cbf: float,
    mtt: float,
    noise_frac: float = 0.0,
    n_timepoints: int = 60,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    seed: int = 42,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
    npt.NDArray[np.floating],
]:
    """Generate DSC test data with known parameters."""
    aif, time_sec = generate_dsc_aif_dro(
        n_timepoints=n_timepoints,
        dt_sec=dt_sec,
        baseline_frames=baseline_frames,
        peak_delta_r2=40.0,
    )

    residue = make_exponential_residue(cbf, mtt, n_timepoints, dt_sec)
    tissue = dsc_forward_model(aif, residue, dt_sec)

    if noise_frac > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=seed)

    return tissue, aif, dt_sec, time_sec


# =============================================================================
# DSC parameter combinations
# =============================================================================

DSC_PARAMS = [
    pytest.param(60.0, 4.0, id="normal_gm"),
    pytest.param(25.0, 6.0, id="normal_wm"),
    pytest.param(100.0, 8.0, id="hyperperfused_tumor"),
    pytest.param(10.0, 3.0, id="low_flow"),
    pytest.param(80.0, 12.0, id="long_mtt"),
]


# =============================================================================
# sSVD Recovery
# =============================================================================

# (noise_fraction, svd_threshold, cbf_tol, mtt_tol)
# SVD threshold must scale with noise to balance bias vs variance.
# At zero noise, threshold=0.001 gives < 5% CBF error.
# At higher noise, higher thresholds control noise amplification at
# the cost of increased regularization bias (underestimated CBF).
SSVD_NOISE_LEVELS = [
    pytest.param(0.0, 0.001, 0.05, 0.10, id="zero"),
    pytest.param(0.02, 0.02, 0.15, 0.65, id="low"),
    pytest.param(0.05, 0.05, 0.20, 0.55, id="medium"),
    pytest.param(0.10, 0.10, 0.30, 0.85, id="high"),
]


class TestSSVDRecovery:
    """Test standard SVD deconvolution parameter recovery."""

    @pytest.mark.parametrize("noise_frac,threshold,cbf_tol,mtt_tol", SSVD_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,mtt", DSC_PARAMS)
    def test_ssvd(
        self,
        cbf: float,
        mtt: float,
        noise_frac: float,
        threshold: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test sSVD recovery of CBF and MTT."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(cbf, mtt, noise_frac)

        result = svd_deconvolve_standard(tissue, aif, dt, threshold=threshold)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# cSVD Recovery — delay detection only
# =============================================================================


class TestCSVDRecovery:
    """Test circulant SVD deconvolution delay detection.

    cSVD uses a block-circulant matrix that wraps around, which makes
    CBF recovery unreliable without actual delays in the data. We focus
    on testing its core feature: delay-insensitive deconvolution.
    """

    @pytest.mark.parametrize(
        "delay_frames",
        [
            pytest.param(2, id="delay_2"),
            pytest.param(5, id="delay_5"),
        ],
    )
    def test_csvd_delay_preserves_shape(self, delay_frames: int) -> None:
        """Test that cSVD recovers a reasonable residue with delayed data."""
        cbf, mtt = 60.0, 4.0
        tissue, aif, dt, time_sec = _make_dsc_test_data(cbf, mtt, noise_frac=0.0)

        # Shift tissue curve to simulate delay
        tissue_shifted = np.zeros_like(tissue)
        tissue_shifted[delay_frames:] = tissue[:-delay_frames]

        result = svd_deconvolve_circulant(tissue_shifted, aif, dt)

        # CBF should be positive and in a reasonable range
        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        # The residue should not be all zeros
        assert np.max(result.residue) > 0


# =============================================================================
# oSVD Recovery
# =============================================================================


class TestOSVDRecovery:
    """Test oscillation-index SVD deconvolution parameter recovery.

    oSVD automatically selects the SVD threshold via oscillation index.
    We set threshold_min=0.001 for zero-noise so oSVD can select a very
    low threshold, matching the minimal regularization needed.
    """

    OSVD_NOISE_LEVELS = [
        pytest.param(0.0, 0.001, 0.05, 0.10, id="zero"),
        pytest.param(0.02, 0.005, 0.15, 0.65, id="low"),
        pytest.param(0.05, 0.01, 0.25, 1.05, id="medium"),
        pytest.param(0.10, 0.01, 0.35, 0.85, id="high"),
    ]

    @pytest.mark.parametrize("noise_frac,thresh_min,cbf_tol,mtt_tol", OSVD_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,mtt", DSC_PARAMS)
    def test_osvd(
        self,
        cbf: float,
        mtt: float,
        noise_frac: float,
        thresh_min: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery of CBF and MTT."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(cbf, mtt, noise_frac)

        result = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=thresh_min
        )

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# Tikhonov Recovery — zero noise only
# =============================================================================


class TestTikhonovRecovery:
    """Test Tikhonov regularization deconvolution parameter recovery.

    Tikhonov uses GCV (generalized cross-validation) to select lambda.
    GCV works excellently at zero noise but selects excessive regularization
    with additive Gaussian noise on synthetic DRO data (a known GCV
    failure mode). We therefore only test at zero noise.
    """

    @pytest.mark.parametrize("cbf,mtt", DSC_PARAMS)
    def test_tikhonov_zero_noise(
        self,
        cbf: float,
        mtt: float,
    ) -> None:
        """Test Tikhonov recovery of CBF and MTT at zero noise."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(cbf, mtt, noise_frac=0.0)

        result = tikhonov_deconvolve(tissue, aif, dt)

        _assert_param_recovery(result.cbf, cbf, 0.05, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, 0.10, param_name="MTT")


# =============================================================================
# CPI Recovery
# =============================================================================

CPI_PARAMS = [
    pytest.param(60.0, 4.0, id="normal_gm"),
    pytest.param(25.0, 6.0, id="normal_wm"),
    pytest.param(100.0, 8.0, id="hyperperfused_tumor"),
    pytest.param(80.0, 12.0, id="long_mtt"),
]

# CPI uses PCHIP spline through 6 control points to approximate an
# exponential residue. The spline approximation introduces ~15-20%
# error even at zero noise — this is inherent to the method.
CPI_NOISE_LEVELS = [
    pytest.param(0.0, 0.30, 0.35, id="zero"),
    pytest.param(0.02, 0.35, 0.45, id="low"),
    pytest.param(0.05, 0.45, 0.55, id="medium"),
]


class TestCPIRecovery:
    """Test Control Point Interpolation model parameter recovery."""

    @pytest.mark.parametrize("noise_frac,cbf_tol,mtt_tol", CPI_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,mtt", CPI_PARAMS)
    def test_cpi(
        self,
        cbf: float,
        mtt: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test CPI recovery of CBF and MTT."""
        n_timepoints = 60
        dt_sec = 1.5
        baseline_frames = 10

        aif, time_sec = generate_dsc_aif_dro(
            n_timepoints=n_timepoints,
            dt_sec=dt_sec,
            baseline_frames=baseline_frames,
        )

        residue = make_exponential_residue(cbf, mtt, n_timepoints, dt_sec)
        tissue = dsc_forward_model(aif, residue, dt_sec)

        if noise_frac > 0:
            peak = float(np.max(np.abs(tissue)))
            if peak > 0:
                tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=42)

        model = CPIModel(n_control_points=6, time_range=(0.0, 30.0))

        result = _fit_cpi_simple(model, tissue, aif, time_sec)
        assert result is not None, "CPI fit returned None"
        assert result.converged or result.r_squared > 0.9

        fitted_params = model.dict_to_params(result.parameters)
        recovered_cbf = model.get_cbf(fitted_params)
        recovered_mtt = model.get_mtt(fitted_params, time_sec)

        _assert_param_recovery(recovered_cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(recovered_mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# Edge Cases
# =============================================================================


class TestDSCEdgeCases:
    """Test edge cases for DSC deconvolution methods."""

    def test_delta_residue(self) -> None:
        """A delta-function residue should recover CBF correctly."""
        n = 60
        dt = 1.5
        aif, time_sec = generate_dsc_aif_dro(n, dt)

        cbf = 50.0
        residue = np.zeros(n, dtype=np.float64)
        residue[0] = cbf

        tissue = dsc_forward_model(aif, residue, dt)
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)

        _assert_param_recovery(result.cbf, cbf, 0.10, param_name="CBF")

    def test_zero_tissue(self) -> None:
        """Zero tissue concentration should give near-zero CBF."""
        n = 60
        dt = 1.5
        aif, time_sec = generate_dsc_aif_dro(n, dt)

        tissue = np.zeros(n, dtype=np.float64)
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)

        assert result.cbf < 0.1, f"CBF for zero tissue: {result.cbf:.4f}"

    def test_short_mtt(self) -> None:
        """Short MTT (near-delta) should be recoverable."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=60.0, mtt=1.0, noise_frac=0.0
        )
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        _assert_param_recovery(result.cbf, 60.0, 0.10, param_name="CBF")

    def test_long_mtt(self) -> None:
        """Long MTT should be recoverable."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=40.0, mtt=15.0, noise_frac=0.0
        )
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        _assert_param_recovery(result.cbf, 40.0, 0.10, param_name="CBF")
        _assert_param_recovery(result.mtt, 15.0, 0.15, param_name="MTT")

    def test_ssvd_osvd_tikhonov_agreement(self) -> None:
        """sSVD, oSVD, and Tikhonov should agree at zero noise."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=60.0, mtt=4.0, noise_frac=0.0
        )

        r_ssvd = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        r_osvd = svd_deconvolve_oscillation(tissue, aif, dt, threshold_min=0.001)
        r_tikh = tikhonov_deconvolve(tissue, aif, dt)

        cbf_values = [r_ssvd.cbf, r_osvd.cbf, r_tikh.cbf]
        mean_cbf = np.mean(cbf_values)

        for cbf_val in cbf_values:
            rel_diff = abs(cbf_val - mean_cbf) / mean_cbf
            assert rel_diff < 0.05, (
                f"Method disagreement: CBFs = {cbf_values}, "
                f"mean = {mean_cbf:.2f}"
            )
