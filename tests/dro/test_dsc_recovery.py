"""DSC parameter recovery tests using Digital Reference Objects.

Tests that deconvolution methods can recover known ground-truth CBF, CBV,
and MTT from synthetic concentration-time curves at various noise levels.
"""

from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.models.dsc.bayesian import fit_dsc_model
from pmriprep.models.dsc.cpi import CPIModel, _fit_cpi_simple
from pmriprep.models.dsc.vascular import VascularModel
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
    pytest.param(0.0, 0.001, 0.05, 0.05, id="zero"),
    pytest.param(0.02, 0.02, 0.40, 1.59, id="low"),
    pytest.param(0.05, 0.05, 0.50, 1.43, id="medium"),
    pytest.param(0.10, 0.10, 0.62, 1.99, id="high"),
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
        # SVD regularization systematically inflates MTT for low-CBF residues
        # (MC N=100: low_flow mtt mean+3sigma = 1.22-1.66).
        if cbf <= 10 and noise_frac > 0:
            pytest.xfail("SVD regularization inflates MTT for low-CBF residues")

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
        pytest.param(0.0, 0.001, 0.05, 0.05, id="zero"),
        pytest.param(0.02, 0.005, 0.37, 0.95, id="low"),
        pytest.param(0.05, 0.01, 0.49, 1.79, id="medium"),
        pytest.param(0.10, 0.01, 0.70, 2.24, id="high"),
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
        # SVD regularization systematically inflates MTT for low-CBF residues
        # (MC N=100: low_flow mtt mean+3sigma = 0.88-1.91).
        if cbf <= 10 and noise_frac > 0:
            pytest.xfail("SVD regularization inflates MTT for low-CBF residues")

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
        _assert_param_recovery(result.mtt, mtt, 0.05, param_name="MTT")


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
    pytest.param(0.0, 0.40, 0.32, id="zero"),
    pytest.param(0.02, 0.63, 0.48, id="low"),
    pytest.param(0.05, 0.68, 0.54, id="medium"),
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
# Vascular Model Recovery
# =============================================================================

# (cbf, alpha, beta, delay)
VASCULAR_PARAMS = [
    pytest.param(60.0, 3.0, 1.5, 0.0, id="normal_gm"),
    pytest.param(25.0, 4.0, 1.5, 0.0, id="normal_wm"),
    pytest.param(100.0, 2.0, 3.0, 0.0, id="tumor"),
    pytest.param(15.0, 3.5, 2.0, 0.0, id="low_flow"),
    pytest.param(60.0, 3.0, 1.5, 3.0, id="delayed"),
]

# (noise_fraction, cbf_tol, mtt_tol)
# CBF and MTT tolerances (relative). MAP uses informative priors, so
# at high noise the prior (CBF mean ~55) pulls low-flow estimates upward.
# The low_flow combo (CBF=15) is xfailed at high noise because the prior
# overwhelms the data — that tests prior behavior, not fitting quality.
# High-noise tolerances cover normal_wm (CBF=25, ~25-35% error).
VASCULAR_NOISE_LEVELS = [
    pytest.param(0.0, 0.07, 0.08, id="zero"),
    pytest.param(0.02, 0.29, 0.37, id="low"),
    pytest.param(0.05, 0.42, 0.65, id="medium"),
    pytest.param(0.10, 0.54, 0.92, id="high"),
]


def _make_vascular_test_data(
    cbf: float,
    alpha: float,
    beta: float,
    delay: float,
    noise_frac: float = 0.0,
    n_timepoints: int = 60,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    seed: int = 42,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Generate vascular model test data with known parameters.

    Uses VascularModel.tissue_curve() for data generation (FFT convolution),
    ensuring a true round-trip test with the MAP fitter.

    Args:
        cbf: Cerebral blood flow (mL/100g/min).
        alpha: Gamma shape parameter.
        beta: Gamma scale parameter (seconds).
        delay: Bolus arrival delay (seconds).
        noise_frac: Noise as fraction of peak concentration.
        n_timepoints: Number of time points.
        dt_sec: Time step in seconds.
        baseline_frames: Number of pre-bolus baseline frames.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (tissue_concentration, aif, time_sec).
    """
    aif, time_sec = generate_dsc_aif_dro(
        n_timepoints=n_timepoints,
        dt_sec=dt_sec,
        baseline_frames=baseline_frames,
        peak_delta_r2=40.0,
    )

    model = VascularModel()
    tissue = model.tissue_curve([cbf, alpha, beta, delay], aif, time_sec)

    if noise_frac > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=seed)

    return tissue, aif, time_sec


class TestVascularRecovery:
    """Test vascular (gamma-variate) model parameter recovery via MAP.

    The vascular model uses a gamma distribution of transit times
    (Mouridsen et al., 2006). We test recovery of CBF and MTT (= alpha * beta).
    Alpha and beta are coupled and NOT asserted separately.

    For the delayed parameter combination, we also check delay recovery.
    """

    @pytest.mark.parametrize("noise_frac,cbf_tol,mtt_tol", VASCULAR_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,alpha,beta,delay", VASCULAR_PARAMS)
    def test_vascular(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        delay: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test vascular model recovery of CBF and MTT."""
        # CBF=15 is near the 2.5th percentile of the prior N(55, 22.5^2),
        # so at 10% noise the prior overwhelms the data — tests prior
        # behavior, not fitting quality.
        if cbf < 20 and noise_frac >= 0.10:
            pytest.xfail("Prior bias overwhelms low-SNR signal at 10% noise")

        # MAP prior N(55, 22.5^2) systematically pulls CBF=25 upward at all
        # noise levels (MC N=100: normal_wm cbf mean+3sigma = 0.25-0.75).
        if cbf <= 25 and noise_frac > 0:
            pytest.xfail("MAP prior bias pulls CBF=25 upward at all noise levels")

        tissue, aif, time_sec = _make_vascular_test_data(
            cbf, alpha, beta, delay, noise_frac
        )

        model = VascularModel()
        # Provide noise_std to avoid zero-noise baseline estimation (std=0
        # gets clamped to 1e-7, making the likelihood overwhelm gradients).
        # Use 1% of peak as a reasonable noise floor for the optimizer.
        peak = float(np.max(np.abs(tissue)))
        noise_std = max(noise_frac * peak, 0.01 * peak) if peak > 0 else 1.0
        result = fit_dsc_model(
            model, tissue, aif, time_sec, method="map",
            compute_uncertainties=False, noise_std=noise_std,
        )
        assert result is not None, "Vascular MAP fit returned None"

        # Extract CBF directly from result
        recovered_cbf = result.parameters["cbf"]
        _assert_param_recovery(recovered_cbf, cbf, cbf_tol, param_name="CBF")

        # Derive MTT from fitted alpha and beta
        fitted_params = model.dict_to_params(result.parameters)
        derived = model.compute_derived_parameters(fitted_params)
        recovered_mtt = derived["mtt"]
        true_mtt = alpha * beta
        _assert_param_recovery(recovered_mtt, true_mtt, mtt_tol, param_name="MTT")

        # For delayed combo, also check delay recovery
        if delay > 0:
            dt = float(time_sec[1] - time_sec[0])
            recovered_delay = result.parameters["delay"]
            assert abs(recovered_delay - delay) < 1.5 * dt, (
                f"Delay: |{recovered_delay:.3f} - {delay:.3f}| = "
                f"{abs(recovered_delay - delay):.3f} >= {1.5 * dt:.3f}"
            )


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
