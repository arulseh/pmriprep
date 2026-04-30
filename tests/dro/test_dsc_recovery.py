"""DSC parameter recovery tests using Digital Reference Objects.

Tests that deconvolution methods can recover known ground-truth CBF, CBV,
and MTT from synthetic concentration-time curves at various noise levels.
"""

from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.models.bayesian_utils import build_priors_from_params
from pmriprep.models.dsc.batched_fitting import batched_dsc_map_fit
from pmriprep.models.dsc.bayesian import fit_dsc_model
from pmriprep.models.dsc.cpi import CPIModel
from pmriprep.models.dsc.vascular import VascularModel
from pmriprep.utils.dsc_deconvolution import (
    build_toeplitz_matrix,
    svd_deconvolve_circulant,
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
    tikhonov_deconvolve,
)

from .conftest import (
    add_gaussian_noise,
    dsc_forward_model,
    dsc_forward_model_circulant,
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
    forward_model: Any = None,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
    npt.NDArray[np.floating],
]:
    """Generate DSC test data with known parameters."""
    if forward_model is None:
        forward_model = dsc_forward_model
    aif, time_sec = generate_dsc_aif_dro(
        n_timepoints=n_timepoints,
        dt_sec=dt_sec,
        baseline_frames=baseline_frames,
        peak_delta_r2=40.0,
    )

    residue = make_exponential_residue(cbf, mtt, n_timepoints, dt_sec)
    tissue = forward_model(aif, residue, dt_sec)

    if noise_frac > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=seed)

    return tissue, aif, dt_sec, time_sec


# =============================================================================
# DSC parameter combinations
# =============================================================================

DSC_PARAMS = [
    pytest.param(0.10, 4.0, id="normal_gm"),
    pytest.param(0.04, 6.0, id="normal_wm"),
    pytest.param(0.16, 8.0, id="hyperperfused_tumor"),
    pytest.param(0.015, 3.0, id="low_flow"),
    pytest.param(0.13, 12.0, id="long_mtt"),
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
        if cbf <= 0.015 and noise_frac > 0:
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

    cSVD uses a block-circulant matrix (Wu 2003). Forward model uses
    the matched circulant matrix to isolate algorithm behavior from
    discretization mismatch.
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
        cbf, mtt = 0.10, 4.0
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf, mtt, noise_frac=0.0,
            forward_model=dsc_forward_model_circulant,
        )

        # Shift tissue curve to simulate delay
        tissue_shifted = np.zeros_like(tissue)
        tissue_shifted[delay_frames:] = tissue[:-delay_frames]

        result = svd_deconvolve_circulant(tissue_shifted, aif, dt)

        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        assert np.max(result.residue) > 0
        assert abs(result.delay_frames - delay_frames) <= 3, (
            f"Expected delay ~{delay_frames}, got {result.delay_frames}"
        )


# =============================================================================
# oSVD Recovery
# =============================================================================


class TestOSVDRecovery:
    """Test oscillation-index SVD deconvolution parameter recovery.

    oSVD uses a block-circulant matrix (Wu 2003). Forward model uses
    the matched circulant matrix to isolate algorithm behavior from
    discretization mismatch.
    """

    @pytest.mark.parametrize("cbf,mtt", DSC_PARAMS)
    def test_osvd_zero_noise(self, cbf: float, mtt: float) -> None:
        """oSVD recovers CBF and MTT at zero noise."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf, mtt, 0.0, forward_model=dsc_forward_model_circulant,
        )
        result = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=1e-6
        )
        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        assert result.mtt > 0, f"MTT should be positive, got {result.mtt}"
        _assert_param_recovery(result.cbf, cbf, 0.10, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, 0.10, param_name="MTT")

    OSVD_NOISY_LEVELS = [
        pytest.param(0.02, 0.005, 0.46, 0.95, id="low"),
        pytest.param(0.05, 0.01, 0.52, 1.79, id="medium"),
        pytest.param(0.10, 0.01, 0.70, 2.24, id="high"),
    ]

    @pytest.mark.parametrize("noise_frac,thresh_min,cbf_tol,mtt_tol", OSVD_NOISY_LEVELS)
    @pytest.mark.parametrize("cbf,mtt", DSC_PARAMS)
    def test_osvd_noisy(
        self,
        cbf: float,
        mtt: float,
        noise_frac: float,
        thresh_min: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery of CBF and MTT with noise."""
        if cbf <= 0.015:
            pytest.xfail("SVD regularization inflates MTT for low-CBF residues")

        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf, mtt, noise_frac,
            forward_model=dsc_forward_model_circulant,
        )
        result = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=thresh_min
        )
        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")

    @pytest.mark.parametrize(
        "delay_frames",
        [
            pytest.param(2, id="delay_2"),
            pytest.param(5, id="delay_5"),
        ],
    )
    def test_osvd_delay_preserves_shape(self, delay_frames: int) -> None:
        """Test that oSVD recovers a reasonable residue with delayed data."""
        cbf, mtt = 0.10, 4.0
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf, mtt, noise_frac=0.0,
            forward_model=dsc_forward_model_circulant,
        )

        tissue_shifted = np.zeros_like(tissue)
        tissue_shifted[delay_frames:] = tissue[:-delay_frames]

        result = svd_deconvolve_oscillation(
            tissue_shifted, aif, dt, threshold_min=0.001
        )
        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        assert np.max(result.residue) > 0
        assert abs(result.delay_frames - delay_frames) <= 2, (
            f"Expected delay ~{delay_frames}, got {result.delay_frames}"
        )


# =============================================================================
# Tikhonov Recovery — zero noise only
# =============================================================================


class TestTikhonovRecovery:
    """Test Tikhonov regularization deconvolution parameter recovery.

    Uses first-order (smoothness) Tikhonov with L-curve corner λ
    selection (Hansen 1998) on gamma-variate residues, which match the
    DSC manuscript figure tissue model. Tolerances are mean+3σ from
    20-seed Monte Carlo on the production defaults
    (``DEFAULT_LAMBDA_RANGE = (1e-3, 10) · σ_max(A)``,
    ``LCURVE_CURVATURE_FALLBACK = 2.0``).

    First-order Tikhonov has a known smoothing bias on sharply-peaked
    (e.g. exponential) residues, which is why the recovery test runs on
    the gamma-variate residue family used in production validation —
    not the exponential residue used by sSVD/oSVD/cSVD tests above.
    """

    # (cbf, alpha, beta, mtt = α·β) — gamma-variate residue (Vascular model).
    GAMMA_PARAMS = [
        pytest.param(0.10, 3.0, 1.5, 4.5, id="normal_gm"),
        pytest.param(0.04, 4.0, 1.5, 6.0, id="normal_wm"),
        pytest.param(0.16, 2.0, 3.0, 6.0, id="tumor"),
        pytest.param(0.13, 3.0, 4.0, 12.0, id="long_mtt"),
    ]

    # Tolerances per (noise_frac, cbf_tol, mtt_tol). MC mean+3σ:
    #   noiseless: max |CBF| 5.4%, max |MTT| 5.4% → 8% covers all
    #   2% noise:  CBF max 6.2+3·2.7=14%, MTT max 16+3·9=43%
    #   5% noise:  CBF max 14.1+3·4.6=28%, MTT max 29.3+3·13=68%
    NOISE_LEVELS = [
        pytest.param(0.0,  0.10, 0.10, id="zero"),
        pytest.param(0.02, 0.20, 0.50, id="low"),
        pytest.param(0.05, 0.35, 0.75, id="medium"),
    ]

    @pytest.mark.parametrize("cbf,alpha,beta,mtt", GAMMA_PARAMS)
    @pytest.mark.parametrize("noise_frac,cbf_tol,mtt_tol", NOISE_LEVELS)
    def test_tikhonov(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        mtt: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Tikhonov recovers CBF and MTT on gamma-variate residues."""
        aif, time_sec = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=10,
            peak_delta_r2=40.0,
        )
        dt = 1.5
        residue = VascularModel().residue_function(
            [cbf, alpha, beta, 0.0], time_sec
        )
        tissue = dsc_forward_model(aif, residue, dt)
        if noise_frac > 0:
            peak = float(np.max(np.abs(tissue)))
            tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=42)

        result = tikhonov_deconvolve(tissue, aif, dt)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# Vascular Model Recovery
# =============================================================================

# (cbf, alpha, beta, delay)
VASCULAR_PARAMS = [
    pytest.param(0.10, 3.0, 1.5, 0.0, id="normal_gm"),
    pytest.param(0.04, 4.0, 1.5, 0.0, id="normal_wm"),
    pytest.param(0.16, 2.0, 3.0, 0.0, id="tumor"),
    pytest.param(0.02, 3.5, 2.0, 0.0, id="low_flow"),
    pytest.param(0.10, 3.0, 1.5, 3.0, id="delayed"),
]

# (noise_fraction, cbf_tol, mtt_tol)
# CBF and MTT tolerances (relative). MAP uses informative priors, so
# at high noise the prior pulls low-flow estimates toward prior mean.
# High-noise tolerances cover normal_wm (~25-35% error).
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
        # CBF=0.02 is near the edge of the log-space prior, so at 10%
        # noise the prior overwhelms the data.
        if cbf < 0.03 and noise_frac >= 0.10:
            pytest.xfail("Prior bias overwhelms low-SNR signal at 10% noise")

        # MAP prior systematically pulls low-CBF tissues toward prior mean
        # at all noise levels.
        if cbf <= 0.04 and noise_frac > 0:
            pytest.xfail("MAP prior bias pulls low CBF toward prior mean")

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
# CPI MAP Recovery on Gamma-Variate Residues
# =============================================================================


def _make_cpi_gamma_test_data(
    cbf: float,
    alpha: float,
    beta: float,
    noise_frac: float = 0.0,
    n_timepoints: int = 60,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    seed: int = 42,
    forward_model: Any = None,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
]:
    """Generate CPI test data with gamma-variate residue.

    Uses VascularModel.residue_function() to create the gamma-variate
    residue R(t) = CBF * [1 - gammainc(alpha, t/beta)], then convolves
    with the DRO AIF via the specified forward model.

    Returns:
        Tuple of (tissue_concentration, aif, time_sec, true_mtt).
    """
    if forward_model is None:
        forward_model = dsc_forward_model
    aif, time_sec = generate_dsc_aif_dro(
        n_timepoints=n_timepoints,
        dt_sec=dt_sec,
        baseline_frames=baseline_frames,
    )

    model = VascularModel()
    residue = model.residue_function([cbf, alpha, beta, 0.0], time_sec)
    tissue = forward_model(aif, residue, dt_sec)

    if noise_frac > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=seed)

    true_mtt = alpha * beta
    return tissue, aif, time_sec, true_mtt


# =============================================================================
# CPI Bayesian (MAP) Recovery on Gamma-Variate Residues
# =============================================================================

# CPI MAP with soft monotonicity + wide priors (Fabber-harmonized).
# VI is now competitive with MAP (within 2%) after harmonization.

CPI_BAYESIAN_PARAMS = [
    pytest.param(0.10, 3.0, 1.5, id="normal_gm"),    # MTT=4.5s
    pytest.param(0.04, 4.0, 1.5, id="normal_wm"),     # MTT=6.0s
    pytest.param(0.16, 2.0, 3.0, id="tumor"),         # MTT=6.0s
    pytest.param(0.02, 3.5, 2.0, id="low_flow"),      # MTT=7.0s
    pytest.param(0.13, 3.0, 4.0, id="long_mtt"),      # MTT=12.0s
]

# (noise_fraction, cbf_tol, mtt_tol)
# v0.34 FABBER-parity tolerances per Mehndiratta 2013 / Chappell 2009 +
# physics review: FABBER's loose sigmoid fraction prior systematically
# inflates MTT (5-20%) and CBF/CBV are biased by the wider full-acquisition
# CP layout. PMRIPrep §6 numerical-change table.
CPI_BAYESIAN_NOISE_LEVELS = [
    pytest.param(0.0, 0.30, 0.25, id="zero"),
    pytest.param(0.02, 0.30, 0.25, id="low"),
    pytest.param(0.05, 0.32, 0.30, id="medium"),
]


class TestCPIBayesianGammaRecovery:
    """Test CPI MAP recovery on gamma-variate residues with noise.

    MAP with soft monotonicity prior and wide fraction priors
    (Fabber-harmonized). Tested on gamma-variate residues at 0%, 2%,
    and 5% noise.
    """

    @pytest.mark.parametrize(
        "noise_frac,cbf_tol,mtt_tol", CPI_BAYESIAN_NOISE_LEVELS
    )
    @pytest.mark.parametrize("cbf,alpha,beta", CPI_BAYESIAN_PARAMS)
    def test_cpi_map(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test CPI MAP recovery of CBF and MTT."""
        true_mtt = alpha * beta

        if cbf <= 0.02:
            pytest.xfail(
                f"CBF prior overwhelms low-flow signal (CBF={cbf})"
            )

        tissue, aif, time_sec, true_mtt = _make_cpi_gamma_test_data(
            cbf, alpha, beta, noise_frac
        )

        n_cp = 10
        t_span = 30.0
        model = CPIModel(
            n_control_points=n_cp,
            n_acquisition_frames=int(len(time_sec)),
            dt=float(time_sec[1] - time_sec[0]),
        )

        # Pass noise_std; use 1% of peak as floor for zero-noise DRO
        peak = float(np.max(np.abs(tissue)))
        noise_std = max(noise_frac * peak, 0.01 * peak) if peak > 0 else 1.0

        result = fit_dsc_model(
            model, tissue, aif, time_sec, method="map",
            compute_uncertainties=False, noise_std=noise_std,
        )
        assert result is not None, "CPI MAP fit returned None"

        fitted_params = model.dict_to_params(result.parameters)
        recovered_cbf = model.get_cbf(fitted_params)
        recovered_mtt = model.get_mtt(fitted_params, time_sec)

        _assert_param_recovery(recovered_cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(
            recovered_mtt, true_mtt, mtt_tol, param_name="MTT"
        )


class TestCPIVIGammaRecovery:
    """Test CPI VI recovery on gamma-variate residues with noise.

    After switching CPI fractions to LOG_LOGIT transforms, VI should
    match MAP accuracy at all noise levels.
    """

    @pytest.mark.parametrize(
        "noise_frac,cbf_tol,mtt_tol", CPI_BAYESIAN_NOISE_LEVELS
    )
    @pytest.mark.parametrize("cbf,alpha,beta", CPI_BAYESIAN_PARAMS)
    def test_cpi_vi(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test CPI VI recovery of CBF and MTT."""
        true_mtt = alpha * beta

        if cbf <= 0.02:
            pytest.xfail(
                f"CBF prior overwhelms low-flow signal (CBF={cbf})"
            )

        tissue, aif, time_sec, true_mtt = _make_cpi_gamma_test_data(
            cbf, alpha, beta, noise_frac
        )

        n_cp = 10
        t_span = 30.0
        model = CPIModel(
            n_control_points=n_cp,
            n_acquisition_frames=int(len(time_sec)),
            dt=float(time_sec[1] - time_sec[0]),
        )

        peak = float(np.max(np.abs(tissue)))
        noise_std = max(noise_frac * peak, 0.01 * peak) if peak > 0 else 1.0

        result = fit_dsc_model(
            model, tissue, aif, time_sec, method="vi",
            compute_uncertainties=False, noise_std=noise_std,
        )
        assert result is not None, "CPI VI fit returned None"

        fitted_params = model.dict_to_params(result.parameters)
        recovered_cbf = model.get_cbf(fitted_params)
        recovered_mtt = model.get_mtt(fitted_params, time_sec)

        _assert_param_recovery(recovered_cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(
            recovered_mtt, true_mtt, mtt_tol, param_name="MTT"
        )


# =============================================================================
# Batched MAP Recovery (vectorized Gauss-Newton path)
# =============================================================================


BATCHED_MAP_CPI_PARAMS = [
    pytest.param(0.10, 3.0, 1.5, id="normal_gm"),
    pytest.param(0.04, 4.0, 1.5, id="normal_wm"),
    pytest.param(0.16, 2.0, 3.0, id="tumor"),
    pytest.param(0.13, 3.0, 4.0, id="long_mtt"),
]

BATCHED_MAP_NOISE_LEVELS = [
    pytest.param(0.0, 0.30, 0.25, id="zero"),
    pytest.param(0.02, 0.30, 0.25, id="low"),
    pytest.param(0.05, 0.32, 0.30, id="medium"),
]


class TestBatchedMAPVascularRecovery:
    """Test batched MAP recovery for vascular model on DRO data.

    Validates that the vectorized Gauss-Newton MAP path produces the same
    quality of parameter recovery as the per-voxel scipy MAP.
    """

    @pytest.mark.parametrize("noise_frac,cbf_tol,mtt_tol", VASCULAR_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,alpha,beta,delay", VASCULAR_PARAMS)
    def test_batched_vascular_map(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        delay: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        if cbf < 0.03 and noise_frac >= 0.10:
            pytest.xfail("Prior bias overwhelms low-SNR signal at 10% noise")
        if cbf <= 0.04 and noise_frac > 0:
            pytest.xfail("MAP prior bias pulls low CBF toward prior mean")

        n_seeds = 5
        tissues = []
        for seed in range(n_seeds):
            t, aif, time_sec = _make_vascular_test_data(
                cbf, alpha, beta, delay, noise_frac, seed=seed,
            )
            tissues.append(t)
        tissue_curves = np.stack(tissues)

        model = VascularModel()
        fitting_defs = list(model.get_fitting_parameter_definitions())
        priors = build_priors_from_params(fitting_defs)

        result = batched_dsc_map_fit(
            model_type="vascular",
            tissue_curves=tissue_curves,
            aif=aif,
            time_points=time_sec,
            prior_means_nat=np.array([p.mean for p in priors]),
            prior_stds_nat=np.array([p.std for p in priors]),
            transforms=[p.transform for p in priors],
            upper_bounds=[p.upper_bound for p in priors],
            fitting_param_names=[p.name for p in fitting_defs],
            fitting_bounds=[(p.bounds[0], p.bounds[1]) for p in fitting_defs],
            prior_means_transformed=np.array([p.mean_transformed for p in priors]),
            prior_stds_transformed=np.array([p.std_transformed for p in priors]),
            compute_uncertainties=False,
        )

        conv = result.converged
        assert np.sum(conv) >= 3, f"Only {np.sum(conv)}/{n_seeds} converged"

        recovered_cbf = float(np.median(result.params["cbf"][conv]))
        _assert_param_recovery(recovered_cbf, cbf, cbf_tol, param_name="CBF")

        true_mtt = alpha * beta
        recovered_mtt = float(np.median(result.params["mtt"][conv]))
        _assert_param_recovery(recovered_mtt, true_mtt, mtt_tol, param_name="MTT")


class TestBatchedMAPCPIRecovery:
    """Test batched MAP recovery for CPI model on DRO data.

    Uses the same gamma-variate ground truth and tolerances as the
    per-voxel CPI MAP tests.
    """

    @pytest.mark.parametrize(
        "noise_frac,cbf_tol,mtt_tol", BATCHED_MAP_NOISE_LEVELS
    )
    @pytest.mark.parametrize("cbf,alpha,beta", BATCHED_MAP_CPI_PARAMS)
    def test_batched_cpi_map(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        true_mtt = alpha * beta

        if cbf <= 0.02:
            pytest.xfail(f"CBF prior overwhelms low-flow signal (CBF={cbf})")

        n_seeds = 5
        tissues = []
        for seed in range(n_seeds):
            t, aif, time_sec, _ = _make_cpi_gamma_test_data(
                cbf, alpha, beta, noise_frac, seed=seed,
            )
            tissues.append(t)
        tissue_curves = np.stack(tissues)

        n_cp = 10
        t_span = 30.0
        dt = float(time_sec[1] - time_sec[0])
        model = CPIModel(
            n_control_points=n_cp,
            n_acquisition_frames=int(len(time_sec)),
            dt=float(time_sec[1] - time_sec[0]),
        )
        fitting_defs = list(model.get_fitting_parameter_definitions())
        priors = build_priors_from_params(fitting_defs)
        cp_times = np.array(model._control_point_times, dtype=np.float64)
        toeplitz = build_toeplitz_matrix(aif, dt)

        result = batched_dsc_map_fit(
            model_type="cpi",
            tissue_curves=tissue_curves,
            aif=aif,
            time_points=time_sec,
            prior_means_nat=np.array([p.mean for p in priors]),
            prior_stds_nat=np.array([p.std for p in priors]),
            transforms=[p.transform for p in priors],
            upper_bounds=[p.upper_bound for p in priors],
            fitting_param_names=[p.name for p in fitting_defs],
            fitting_bounds=[(p.bounds[0], p.bounds[1]) for p in fitting_defs],
            control_point_times=cp_times,
            toeplitz_matrix=toeplitz,
            prior_means_transformed=np.array([p.mean_transformed for p in priors]),
            prior_stds_transformed=np.array([p.std_transformed for p in priors]),
            compute_uncertainties=False,
        )

        conv = result.converged
        assert np.sum(conv) >= 3, f"Only {np.sum(conv)}/{n_seeds} converged"

        for i in range(n_seeds):
            if not conv[i]:
                continue
            p_dict = {n: float(result.params[n][i]) for n in [p.name for p in fitting_defs]}
            p_list = model.dict_to_params(p_dict)
            recovered_cbf = model.get_cbf(p_list)
            recovered_mtt = model.get_mtt(p_list, time_sec)
            _assert_param_recovery(recovered_cbf, cbf, cbf_tol, param_name="CBF")
            _assert_param_recovery(recovered_mtt, true_mtt, mtt_tol, param_name="MTT")


# =============================================================================
# SVD Recovery on Gamma-Variate Residues (cross-residue validation)
# =============================================================================

# SVD methods are model-free and should generalize across residue shapes.
# Gamma-variate residues have a plateau before decay (R'(0)=0 for alpha>1),
# which SVD regularization can truncate, causing higher CBF overestimation
# than on exponential residues.

SVD_GAMMA_PARAMS = [
    pytest.param(0.10, 3.0, 1.5, id="normal_gm"),    # MTT=4.5s
    pytest.param(0.04, 4.0, 1.5, id="normal_wm"),     # MTT=6.0s
    pytest.param(0.16, 2.0, 3.0, id="tumor"),         # MTT=6.0s
    pytest.param(0.13, 3.0, 4.0, id="long_mtt"),      # MTT=12.0s
]

# Tolerances wider than exponential: SVD regularization truncates the
# gamma-variate plateau, causing systematic CBF overestimation.
# Zero-noise: deterministic — sSVD long_mtt has 19% CBF, 15% MTT from
# threshold truncating the gamma plateau. MTT 15.1% sets the floor.
# 5% noise: MC N=20, mean+3σ. Binding: oSVD long_mtt CBF 56%,
# oSVD normal_gm MTT 85%. Large MTT errors are inherent to SVD on
# non-exponential residues (Mouridsen 2006).
SVD_GAMMA_NOISE_LEVELS = [
    pytest.param(0.0, 0.20, 0.16, id="zero"),     # CBF 19%, MTT 15.1% (deterministic)
    pytest.param(0.05, 0.57, 0.86, id="medium"),   # MC: CBF m+3σ=56%, MTT m+3σ=85%
]


class TestSVDGammaRecovery:
    """Test SVD deconvolution on gamma-variate residues.

    Validates that model-free SVD methods generalize beyond the exponential
    residues they are traditionally benchmarked on. This matches fig2
    panel b data.
    """

    @pytest.mark.parametrize("noise_frac,cbf_tol,mtt_tol", SVD_GAMMA_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,alpha,beta", SVD_GAMMA_PARAMS)
    def test_ssvd_gamma(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test sSVD recovery of CBF and MTT from gamma-variate residue."""
        tissue, aif, time_sec, true_mtt = _make_cpi_gamma_test_data(
            cbf, alpha, beta, noise_frac
        )

        dt = float(time_sec[1] - time_sec[0])
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.05)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, true_mtt, mtt_tol, param_name="MTT")

    OSVD_GAMMA_NOISE_LEVELS = [
        pytest.param(0.0, 0.10, 0.30, id="zero"),
        pytest.param(0.05, 0.57, 0.86, id="medium"),
    ]

    @pytest.mark.parametrize("noise_frac,cbf_tol,mtt_tol", OSVD_GAMMA_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,alpha,beta", SVD_GAMMA_PARAMS)
    def test_osvd_gamma(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery of CBF and MTT from gamma-variate residue."""
        tissue, aif, time_sec, true_mtt = _make_cpi_gamma_test_data(
            cbf, alpha, beta, noise_frac,
            forward_model=dsc_forward_model_circulant,
        )

        dt = float(time_sec[1] - time_sec[0])
        result = svd_deconvolve_oscillation(tissue, aif, dt, threshold_min=0.01)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, true_mtt, mtt_tol, param_name="MTT")


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

        cbf = 0.08
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

        assert result.cbf < 1e-4, f"CBF for zero tissue: {result.cbf:.6f}"

    def test_short_mtt(self) -> None:
        """Short MTT (near-delta) should be recoverable."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=0.10, mtt=1.0, noise_frac=0.0
        )
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        _assert_param_recovery(result.cbf, 0.10, 0.10, param_name="CBF")

    def test_long_mtt(self) -> None:
        """Long MTT should be recoverable."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=0.065, mtt=15.0, noise_frac=0.0
        )
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        _assert_param_recovery(result.cbf, 0.065, 0.10, param_name="CBF")
        _assert_param_recovery(result.mtt, 15.0, 0.15, param_name="MTT")

    def test_ssvd_tikhonov_agreement(self) -> None:
        """sSVD and Tikhonov (both Toeplitz) should agree at zero noise.

        Tolerance is 25%: sSVD with threshold=0.001 is essentially the
        unregularised pseudo-inverse, while first-order Tikhonov + L-curve
        always applies a smoothness penalty even at zero noise (its
        L-curve has no real corner there, so it falls back to the
        Hansen 1998 default λ = 0.1·σ_max). The smoothing biases CBF
        down ~5–25% on exponential residues.
        """
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=0.10, mtt=4.0, noise_frac=0.0
        )

        r_ssvd = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        r_tikh = tikhonov_deconvolve(tissue, aif, dt)

        cbf_values = [r_ssvd.cbf, r_tikh.cbf]
        mean_cbf = float(np.mean(cbf_values))

        for cbf_val in cbf_values:
            rel_diff = abs(cbf_val - mean_cbf) / mean_cbf
            assert rel_diff < 0.25, (
                f"Method disagreement: CBFs = {cbf_values}, "
                f"mean = {mean_cbf:.2f}"
            )
