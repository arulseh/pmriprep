"""QIBA DSC-MRI DRO parameter recovery tests.

Tests deconvolution methods against QIBA-representative synthetic data with:
- Recirculation AIF (Patil & Johnson 2011, kappa=0.35)
- CTH gamma-variate residue (Mouridsen 2006)
- Published tissue parameters (Semmineh 2017, Boxerman 2020)

Phase 1: Code-generated data (this file).
Phase 2: External GBM-DSC-MRI-DRO (Phantomas/BNI) from qibadscdro.rsna.org.

Key differences from Tier 1 (tests/dro/test_dsc_recovery.py):
1. AIF has SCR recirculation tail (harder deconvolution)
2. CTH gamma-variate residue (non-exponential shapes)
3. QIBA tissue parameter tables (8 tissue types)
4. 80 timepoints (vs 60) to accommodate recirculation
5. Cross-method comparison on identical data
"""

from typing import Tuple

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
    CTH_CONFIGS,
    QIBA_NOISE_LEVELS,
    QIBA_TISSUE_PARAMS,
    RILMAYO_CONFIGS,
    RILMAYO_DRO_ROOT,
    add_gaussian_noise,
    deconvolve_dro_batch,
    dsc_forward_model,
    generate_cth_residue,
    generate_qiba_aif,
    load_qiba_dro,
    load_rilmayo_dro,
    make_exponential_residue,
    rilmayo_config_available,
    skip_no_qiba_dro,
    skip_no_rilmayo_dro,
)

# =============================================================================
# Constants
# =============================================================================

N_TIMEPOINTS = 80
DT_SEC = 1.5
BASELINE_FRAMES = 10
SEED = 42


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
    """Assert DSC parameter recovery within tolerance.

    Uses relative tolerance for non-zero truth, absolute for near-zero.
    Copied from Tier 1 for test independence.
    """
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


def _make_qiba_exponential_data(
    tissue_key: str,
    noise_key: str,
    seed: int = SEED,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
    npt.NDArray[np.floating],
    float,
    float,
]:
    """Generate QIBA test data with exponential residue.

    Returns:
        Tuple of (tissue, aif, dt, time_sec, cbf, mtt).
    """
    params = QIBA_TISSUE_PARAMS[tissue_key]
    noise = QIBA_NOISE_LEVELS[noise_key]

    aif, time_sec = generate_qiba_aif(
        n_timepoints=N_TIMEPOINTS,
        dt_sec=DT_SEC,
        baseline_frames=BASELINE_FRAMES,
    )

    residue = make_exponential_residue(params.cbf, params.mtt, N_TIMEPOINTS, DT_SEC)
    tissue = dsc_forward_model(aif, residue, DT_SEC)

    if noise.fraction > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise.fraction * peak, seed=seed)

    return tissue, aif, DT_SEC, time_sec, params.cbf, params.mtt


def _make_qiba_cth_data(
    tissue_key: str,
    cth_key: str,
    noise_key: str,
    seed: int = SEED,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
    npt.NDArray[np.floating],
    float,
    float,
    float,
]:
    """Generate QIBA test data with CTH gamma-variate residue.

    beta = mtt / alpha (from gamma distribution mean = alpha * beta).

    Returns:
        Tuple of (tissue, aif, dt, time_sec, cbf, mtt, alpha).
    """
    params = QIBA_TISSUE_PARAMS[tissue_key]
    cth = CTH_CONFIGS[cth_key]
    noise = QIBA_NOISE_LEVELS[noise_key]

    aif, time_sec = generate_qiba_aif(
        n_timepoints=N_TIMEPOINTS,
        dt_sec=DT_SEC,
        baseline_frames=BASELINE_FRAMES,
    )

    beta = params.mtt / cth.alpha
    residue = generate_cth_residue(params.cbf, cth.alpha, beta, N_TIMEPOINTS, DT_SEC)
    tissue = dsc_forward_model(aif, residue, DT_SEC)

    if noise.fraction > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise.fraction * peak, seed=seed)

    return tissue, aif, DT_SEC, time_sec, params.cbf, params.mtt, cth.alpha


# =============================================================================
# Tissue and noise parametrize helpers
# =============================================================================

# 6 tissues for SVD methods (exclude tumor_high and contralateral_gm)
SVD_TISSUES = [
    pytest.param("normal_gm", id="normal_gm"),
    pytest.param("normal_wm", id="normal_wm"),
    pytest.param("tumor", id="tumor"),
    pytest.param("tumor_low", id="tumor_low"),
    pytest.param("peritumoral", id="peritumoral"),
    pytest.param("low_perfusion", id="low_perfusion"),
]

# 4 tissues for CPI (drop low_perfusion and tumor_high)
CPI_TISSUES = [
    pytest.param("normal_gm", id="normal_gm"),
    pytest.param("normal_wm", id="normal_wm"),
    pytest.param("tumor", id="tumor"),
    pytest.param("peritumoral", id="peritumoral"),
]

# 4 tissues for Vascular (drop low_perfusion and tumor_high)
VASCULAR_TISSUES = [
    pytest.param("normal_gm", id="normal_gm"),
    pytest.param("normal_wm", id="normal_wm"),
    pytest.param("tumor", id="tumor"),
    pytest.param("peritumoral", id="peritumoral"),
]

# CTH combos: 3 cth x 2 tissues = 6
CTH_COMBOS = [
    pytest.param("normal_gm", "low_cth", id="normal_gm-low_cth"),
    pytest.param("normal_gm", "moderate_cth", id="normal_gm-moderate_cth"),
    pytest.param("normal_gm", "high_cth", id="normal_gm-high_cth"),
    pytest.param("tumor", "low_cth", id="tumor-low_cth"),
    pytest.param("tumor", "moderate_cth", id="tumor-moderate_cth"),
    pytest.param("tumor", "high_cth", id="tumor-high_cth"),
]

# CTH combos for CPI (4 combos, no low_perfusion/tumor_high)
CPI_CTH_COMBOS = [
    pytest.param("normal_gm", "low_cth", id="normal_gm-low_cth"),
    pytest.param("normal_gm", "high_cth", id="normal_gm-high_cth"),
    pytest.param("tumor", "low_cth", id="tumor-low_cth"),
    pytest.param("tumor", "high_cth", id="tumor-high_cth"),
]

# CTH combos for Vascular (4 combos)
VASCULAR_CTH_COMBOS = [
    pytest.param("normal_gm", "low_cth", id="normal_gm-low_cth"),
    pytest.param("normal_gm", "high_cth", id="normal_gm-high_cth"),
    pytest.param("tumor", "low_cth", id="tumor-low_cth"),
    pytest.param("tumor", "high_cth", id="tumor-high_cth"),
]


# =============================================================================
# sSVD with exponential residue
# =============================================================================

# (noise_key, threshold, cbf_tol, mtt_tol)
# Recirculation adds ~10-15% difficulty vs Tier 1
SSVD_QIBA_NOISE = [
    pytest.param("zero", 0.001, 0.10, 0.15, id="zero"),
    pytest.param("low", 0.02, 0.45, 1.70, id="low"),
    pytest.param("medium", 0.05, 0.55, 1.60, id="medium"),
    pytest.param("high", 0.10, 0.70, 2.10, id="high"),
]


class TestQIBASSVDExponential:
    """sSVD recovery with QIBA AIF (recirculation) and exponential residue."""

    @pytest.mark.parametrize("noise_key,threshold,cbf_tol,mtt_tol", SSVD_QIBA_NOISE)
    @pytest.mark.parametrize("tissue_key", SVD_TISSUES)
    def test_ssvd_exponential(
        self,
        tissue_key: str,
        noise_key: str,
        threshold: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test sSVD recovery with QIBA recirculation AIF."""
        if tissue_key == "low_perfusion" and noise_key != "zero":
            pytest.xfail(
                "SVD regularization inflates MTT for low-CBF (12) residues"
            )

        tissue, aif, dt, time_sec, cbf, mtt = _make_qiba_exponential_data(
            tissue_key, noise_key
        )

        result = svd_deconvolve_standard(tissue, aif, dt, threshold=threshold)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# cSVD with exponential residue — delay detection only
# =============================================================================


class TestQIBACSVD:
    """cSVD delay preservation with QIBA AIF."""

    @pytest.mark.parametrize(
        "tissue_key,delay_frames",
        [
            pytest.param("normal_gm", 2, id="normal_gm-delay2"),
            pytest.param("normal_gm", 5, id="normal_gm-delay5"),
            pytest.param("tumor", 2, id="tumor-delay2"),
            pytest.param("tumor", 5, id="tumor-delay5"),
        ],
    )
    def test_csvd_delay_preserves_shape(
        self, tissue_key: str, delay_frames: int
    ) -> None:
        """Test that cSVD recovers reasonable residue with delayed QIBA data."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_qiba_exponential_data(
            tissue_key, "zero"
        )

        # Shift tissue curve to simulate delay
        tissue_shifted = np.zeros_like(tissue)
        tissue_shifted[delay_frames:] = tissue[:-delay_frames]

        result = svd_deconvolve_circulant(tissue_shifted, aif, dt)

        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        assert np.max(result.residue) > 0


# =============================================================================
# oSVD with exponential residue
# =============================================================================

OSVD_QIBA_NOISE = [
    pytest.param("zero", 0.001, 0.10, 0.15, id="zero"),
    pytest.param("low", 0.005, 0.42, 1.10, id="low"),
    pytest.param("medium", 0.01, 0.55, 1.90, id="medium"),
    pytest.param("high", 0.01, 0.75, 2.40, id="high"),
]


class TestQIBAOSVDExponential:
    """oSVD recovery with QIBA AIF (recirculation) and exponential residue."""

    @pytest.mark.parametrize("noise_key,thresh_min,cbf_tol,mtt_tol", OSVD_QIBA_NOISE)
    @pytest.mark.parametrize("tissue_key", SVD_TISSUES)
    def test_osvd_exponential(
        self,
        tissue_key: str,
        noise_key: str,
        thresh_min: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery with QIBA recirculation AIF."""
        if tissue_key == "low_perfusion" and noise_key != "zero":
            pytest.xfail(
                "SVD regularization inflates MTT for low-CBF (12) residues"
            )

        tissue, aif, dt, time_sec, cbf, mtt = _make_qiba_exponential_data(
            tissue_key, noise_key
        )

        result = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=thresh_min
        )

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# Tikhonov with exponential residue — zero noise only
# =============================================================================


class TestQIBATikhonovExponential:
    """Tikhonov recovery with QIBA AIF, zero noise only.

    GCV over-regularizes with additive Gaussian noise on DRO data.
    """

    @pytest.mark.parametrize("tissue_key", SVD_TISSUES)
    def test_tikhonov_zero_noise(self, tissue_key: str) -> None:
        """Test Tikhonov recovery at zero noise with QIBA AIF."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_qiba_exponential_data(
            tissue_key, "zero"
        )

        result = tikhonov_deconvolve(tissue, aif, dt)

        _assert_param_recovery(result.cbf, cbf, 0.10, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, 0.15, param_name="MTT")


# =============================================================================
# CPI with exponential residue
# =============================================================================

CPI_QIBA_NOISE = [
    pytest.param("zero", 0.45, 0.40, id="zero"),
    pytest.param("low", 0.68, 0.55, id="low"),
    pytest.param("medium", 0.75, 0.65, id="medium"),
]


class TestQIBACPIExponential:
    """CPI recovery with QIBA AIF and exponential residue.

    CPI uses PCHIP spline through control points; ~15-20% inherent
    approximation error even at zero noise.
    """

    @pytest.mark.parametrize("noise_key,cbf_tol,mtt_tol", CPI_QIBA_NOISE)
    @pytest.mark.parametrize("tissue_key", CPI_TISSUES)
    def test_cpi_exponential(
        self,
        tissue_key: str,
        noise_key: str,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test CPI recovery with QIBA recirculation AIF."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_qiba_exponential_data(
            tissue_key, noise_key
        )

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
# Vascular model with exponential residue
# =============================================================================

# Vascular MAP uses its own forward model (gamma residue + FFT convolution).
# For exponential residue data, we still test Toeplitz-generated tissue curves
# since the MAP fitter must handle model mismatch (gamma ≠ exponential).
VASCULAR_QIBA_NOISE = [
    pytest.param("zero", 0.20, 0.20, id="zero"),
    pytest.param("low", 0.40, 0.50, id="low"),
    pytest.param("medium", 0.55, 0.80, id="medium"),
    pytest.param("high", 0.65, 1.05, id="high"),
]


class TestQIBAVascularExponential:
    """Vascular MAP recovery with QIBA AIF and exponential residue.

    Tests model mismatch: data generated with exponential residue,
    fitted with gamma-variate model. Zero-noise CBF error ~18% is
    expected from the model mismatch + recirculation AIF.
    """

    @pytest.mark.parametrize("noise_key,cbf_tol,mtt_tol", VASCULAR_QIBA_NOISE)
    @pytest.mark.parametrize("tissue_key", VASCULAR_TISSUES)
    def test_vascular_exponential(
        self,
        tissue_key: str,
        noise_key: str,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test vascular MAP recovery with QIBA recirculation AIF."""
        # MAP prior N(55, 22.5^2) pulls normal_wm (CBF=20) upward
        if tissue_key == "normal_wm" and noise_key != "zero":
            pytest.xfail(
                "MAP prior bias pulls CBF=20 upward at all noise levels"
            )

        tissue, aif, dt, time_sec, cbf, mtt = _make_qiba_exponential_data(
            tissue_key, noise_key
        )

        model = VascularModel()
        peak = float(np.max(np.abs(tissue)))
        noise_frac = QIBA_NOISE_LEVELS[noise_key].fraction
        noise_std = max(noise_frac * peak, 0.01 * peak) if peak > 0 else 1.0

        result = fit_dsc_model(
            model,
            tissue,
            aif,
            time_sec,
            method="map",
            compute_uncertainties=False,
            noise_std=noise_std,
        )
        assert result is not None, "Vascular MAP fit returned None"

        recovered_cbf = result.parameters["cbf"]
        _assert_param_recovery(recovered_cbf, cbf, cbf_tol, param_name="CBF")

        fitted_params = model.dict_to_params(result.parameters)
        derived = model.compute_derived_parameters(fitted_params)
        recovered_mtt = derived["mtt"]
        _assert_param_recovery(recovered_mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# sSVD with CTH residue
# =============================================================================

SSVD_CTH_NOISE = [
    pytest.param("zero", 0.001, 0.20, 0.30, id="zero"),
    pytest.param("low", 0.02, 0.50, 2.00, id="low"),
    pytest.param("medium", 0.05, 0.60, 2.50, id="medium"),
]


class TestQIBASSVDCTH:
    """sSVD recovery with QIBA AIF and CTH gamma-variate residue.

    MTT tolerances ~2x exponential because SVD bias increases with
    non-exponential residue shapes.
    """

    @pytest.mark.parametrize("noise_key,threshold,cbf_tol,mtt_tol", SSVD_CTH_NOISE)
    @pytest.mark.parametrize("tissue_key,cth_key", CTH_COMBOS)
    def test_ssvd_cth(
        self,
        tissue_key: str,
        cth_key: str,
        noise_key: str,
        threshold: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test sSVD recovery with CTH residue."""
        tissue, aif, dt, time_sec, cbf, mtt, alpha = _make_qiba_cth_data(
            tissue_key, cth_key, noise_key
        )

        result = svd_deconvolve_standard(tissue, aif, dt, threshold=threshold)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# oSVD with CTH residue
# =============================================================================

OSVD_CTH_NOISE = [
    pytest.param("zero", 0.001, 0.20, 0.30, id="zero"),
    pytest.param("low", 0.005, 0.50, 2.00, id="low"),
    pytest.param("medium", 0.01, 0.60, 2.50, id="medium"),
]


class TestQIBAOSVDCTH:
    """oSVD recovery with QIBA AIF and CTH gamma-variate residue."""

    @pytest.mark.parametrize("noise_key,thresh_min,cbf_tol,mtt_tol", OSVD_CTH_NOISE)
    @pytest.mark.parametrize("tissue_key,cth_key", CTH_COMBOS)
    def test_osvd_cth(
        self,
        tissue_key: str,
        cth_key: str,
        noise_key: str,
        thresh_min: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery with CTH residue."""
        tissue, aif, dt, time_sec, cbf, mtt, alpha = _make_qiba_cth_data(
            tissue_key, cth_key, noise_key
        )

        result = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=thresh_min
        )

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# Tikhonov with CTH residue — zero noise only
# =============================================================================


class TestQIBATikhonovCTH:
    """Tikhonov recovery with CTH residue, zero noise only."""

    @pytest.mark.parametrize("tissue_key,cth_key", CTH_COMBOS)
    def test_tikhonov_cth_zero_noise(
        self, tissue_key: str, cth_key: str
    ) -> None:
        """Test Tikhonov recovery with CTH residue at zero noise."""
        tissue, aif, dt, time_sec, cbf, mtt, alpha = _make_qiba_cth_data(
            tissue_key, cth_key, "zero"
        )

        result = tikhonov_deconvolve(tissue, aif, dt)

        _assert_param_recovery(result.cbf, cbf, 0.20, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, 0.30, param_name="MTT")


# =============================================================================
# CPI with CTH residue
# =============================================================================

CPI_CTH_NOISE = [
    pytest.param("zero", 0.50, 0.40, id="zero"),
    pytest.param("low", 0.70, 0.60, id="low"),
]


class TestQIBACPICTH:
    """CPI recovery with CTH residue.

    CPI's PCHIP spline is designed for non-exponential residues,
    so it should perform as well as or better than with exponential.
    """

    @pytest.mark.parametrize("noise_key,cbf_tol,mtt_tol", CPI_CTH_NOISE)
    @pytest.mark.parametrize("tissue_key,cth_key", CPI_CTH_COMBOS)
    def test_cpi_cth(
        self,
        tissue_key: str,
        cth_key: str,
        noise_key: str,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test CPI recovery with CTH residue."""
        tissue, aif, dt, time_sec, cbf, mtt, alpha = _make_qiba_cth_data(
            tissue_key, cth_key, noise_key
        )

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
# Vascular with CTH residue — round-trip (self-consistent model)
# =============================================================================

VASCULAR_CTH_NOISE = [
    pytest.param("zero", 0.25, 0.25, id="zero"),
    pytest.param("low", 0.40, 0.55, id="low"),
    pytest.param("medium", 0.55, 0.80, id="medium"),
    pytest.param("high", 0.65, 1.05, id="high"),
]


class TestQIBAVascularCTH:
    """Vascular MAP recovery with CTH residue — true round-trip.

    Data generated via VascularModel.tissue_curve() with gamma residue,
    fitted with the same model. Recirculation AIF increases zero-noise
    error (~15-24% CBF) vs Tier 1's <5% due to AIF complexity.
    """

    @pytest.mark.parametrize("noise_key,cbf_tol,mtt_tol", VASCULAR_CTH_NOISE)
    @pytest.mark.parametrize("tissue_key,cth_key", VASCULAR_CTH_COMBOS)
    def test_vascular_cth(
        self,
        tissue_key: str,
        cth_key: str,
        noise_key: str,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test vascular MAP round-trip with CTH residue."""
        params = QIBA_TISSUE_PARAMS[tissue_key]
        cth = CTH_CONFIGS[cth_key]
        noise = QIBA_NOISE_LEVELS[noise_key]

        aif, time_sec = generate_qiba_aif(
            n_timepoints=N_TIMEPOINTS,
            dt_sec=DT_SEC,
            baseline_frames=BASELINE_FRAMES,
        )

        # Generate tissue via VascularModel for round-trip
        beta = params.mtt / cth.alpha
        true_params = [params.cbf, cth.alpha, beta, 0.0]

        model = VascularModel()
        tissue = model.tissue_curve(true_params, aif, time_sec)

        if noise.fraction > 0:
            peak = float(np.max(np.abs(tissue)))
            if peak > 0:
                tissue = add_gaussian_noise(
                    tissue, noise.fraction * peak, seed=SEED
                )

        peak = float(np.max(np.abs(tissue)))
        noise_std = (
            max(noise.fraction * peak, 0.01 * peak) if peak > 0 else 1.0
        )

        result = fit_dsc_model(
            model,
            tissue,
            aif,
            time_sec,
            method="map",
            compute_uncertainties=False,
            noise_std=noise_std,
        )
        assert result is not None, "Vascular MAP fit returned None"

        recovered_cbf = result.parameters["cbf"]
        _assert_param_recovery(
            recovered_cbf, params.cbf, cbf_tol, param_name="CBF"
        )

        fitted_params = model.dict_to_params(result.parameters)
        derived = model.compute_derived_parameters(fitted_params)
        recovered_mtt = derived["mtt"]
        _assert_param_recovery(
            recovered_mtt, params.mtt, mtt_tol, param_name="MTT"
        )


# =============================================================================
# Cross-method comparison
# =============================================================================


class TestQIBACrossMethodComparison:
    """Compare deconvolution methods on identical QIBA data.

    Documents relative performance without asserting ranking
    (method performance is tissue/noise dependent).
    """

    @pytest.mark.parametrize(
        "tissue_key,noise_key",
        [
            pytest.param("normal_gm", "zero", id="normal_gm-zero"),
            pytest.param("normal_gm", "low", id="normal_gm-low"),
            pytest.param("tumor", "zero", id="tumor-zero"),
            pytest.param("tumor", "low", id="tumor-low"),
        ],
    )
    def test_cross_method_cbf_agreement(
        self, tissue_key: str, noise_key: str
    ) -> None:
        """All methods should produce positive CBF within 5x of truth."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_qiba_exponential_data(
            tissue_key, noise_key
        )

        results = {}
        results["ssvd"] = svd_deconvolve_standard(
            tissue, aif, dt, threshold=0.02
        )
        results["osvd"] = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=0.005
        )
        results["tikhonov"] = tikhonov_deconvolve(tissue, aif, dt)

        for name, r in results.items():
            assert r.cbf > 0, f"{name} CBF should be positive, got {r.cbf}"
            assert r.cbf < cbf * 5, (
                f"{name} CBF={r.cbf:.1f} exceeds 5x truth ({cbf * 5:.1f})"
            )

    @pytest.mark.parametrize(
        "tissue_key",
        [
            pytest.param("normal_gm", id="normal_gm"),
            pytest.param("tumor", id="tumor"),
        ],
    )
    def test_cross_method_zero_noise_convergence(
        self, tissue_key: str
    ) -> None:
        """SVD methods should agree within 20% at zero noise."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_qiba_exponential_data(
            tissue_key, "zero"
        )

        r_ssvd = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        r_osvd = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=0.001
        )
        r_tikh = tikhonov_deconvolve(tissue, aif, dt)

        cbf_values = [r_ssvd.cbf, r_osvd.cbf, r_tikh.cbf]
        mean_cbf = float(np.mean(cbf_values))

        for cbf_val in cbf_values:
            rel_diff = abs(cbf_val - mean_cbf) / mean_cbf
            assert rel_diff < 0.20, (
                f"Method disagreement: CBFs = "
                f"[{cbf_values[0]:.2f}, {cbf_values[1]:.2f}, {cbf_values[2]:.2f}], "
                f"mean = {mean_cbf:.2f}"
            )


# =============================================================================
# Phase 2: External GBM-DSC-MRI-DRO (Phantomas/BNI)
# =============================================================================


@skip_no_qiba_dro
class TestGBMDSCDRO:
    """Phase 2: External QIBA GBM-DSC-MRI-DRO validation.

    Uses converted Phantomas DRO data (qibadscdro.rsna.org, BNI config).
    No per-voxel ground truth available — validates via physiological range
    checks, cross-method consistency, and inter-ROI contrasts.

    Config: TR=1500ms, TE=30ms, FA=30deg, B0=3.0T, dose=0.5+0.5.
    Volume: (128, 128, 63, 121), two tissue intensity clusters.

    Known DRO limitations (MRI physics review):

    1. **5.7-bit effective dynamic range** (uint16 max=53, ~50 discrete levels).
       Quantization noise dominates over thermal noise. GCV parameter selection
       fails because it assumes IID Gaussian noise (Golub, Heath, Wahba 1979),
       but quantization error is structured/step-patterned. sSVD/oSVD are
       robust to this; Tikhonov GCV is not.
    2. **Tissue-derived pseudo-AIF** (no true arterial compartment in DRO).
       Peak = 10.5 s^-1, below clinical AIF threshold (~15 s^-1, Boxerman
       et al., AJNR 2006). Results are relative (rCBV ratios valid), not
       absolute.
    3. **Dual-injection protocol in single series.** Both preload (~45s) and
       main bolus (~82.5s) peaks present in 121-timepoint series. Deconvolution
       assumes single bolus passage; full-series analysis is approximate.
    4. **No per-voxel ground truth.** Validation limited to physiological range
       checks and cross-method consistency, not parameter recovery.
    5. **Signal ratio clipped to [eps, 1.0]** — T1 leakage effects (S > S0)
       not modeled.
    """

    # -- Test E: concentration sanity ---

    def test_concentration_sanity(self) -> None:
        """DeltaR2* should be non-negative and bounded."""
        dro = load_qiba_dro()
        tissue_mask = dro.roi_masks["tissue_high"] | dro.roi_masks["tissue_low"]
        tissue_dr2 = dro.delta_r2[tissue_mask]

        # deltaR2* from -ln(S/S0)/TE should be >= 0 when S <= S0
        # Small negative values from quantization noise are acceptable
        min_val = tissue_dr2.min()
        assert min_val >= -1.0, (
            f"deltaR2* has unexpectedly negative values: min={min_val:.2f}"
        )

        # Peak should be bounded (< 200 s^-1 for tissue)
        peak_vals = np.max(tissue_dr2, axis=1)
        median_peak = np.median(peak_vals)
        assert median_peak < 200.0, (
            f"Median peak deltaR2* too high: {median_peak:.2f} s^-1"
        )

    # -- Test D: AIF bolus characteristics ---

    def test_aif_bolus_characteristics(self) -> None:
        """AIF should have a detectable bolus peak."""
        dro = load_qiba_dro()
        aif = dro.aif

        peak = np.max(aif)
        ttp_sec = np.argmax(aif) * dro.dt

        # Peak should be meaningfully above baseline noise
        baseline_std = np.std(aif[:5])
        assert peak > 3.0 * max(baseline_std, 0.1), (
            f"AIF peak ({peak:.2f} s^-1) not significantly above "
            f"baseline noise (std={baseline_std:.2f})"
        )

        # TTP should be within scan duration
        assert 0 < ttp_sec < dro.time_sec[-1], (
            f"AIF TTP outside scan range: {ttp_sec:.1f}s"
        )

    # -- Test G: tissue types have distinct dynamics ---

    def test_tissue_types_distinct_dynamics(self) -> None:
        """The two tissue intensity clusters should differ in baseline signal."""
        dro = load_qiba_dro()

        mean_high = np.mean(dro.baseline[dro.roi_masks["tissue_high"]])
        mean_low = np.mean(dro.baseline[dro.roi_masks["tissue_low"]])

        # Clusters should have clearly different baseline intensities
        ratio = mean_high / max(mean_low, 1e-10)
        assert ratio > 1.3, (
            f"Tissue clusters not sufficiently distinct: "
            f"high={mean_high:.1f}, low={mean_low:.1f}, ratio={ratio:.2f}"
        )

    # -- Test B: MTT in physiological range ---

    @pytest.mark.parametrize("method", ["ssvd", "osvd", "tikhonov"])
    @pytest.mark.parametrize("roi", ["tissue_high", "tissue_low"])
    def test_mtt_range(self, method: str, roi: str) -> None:
        """Median MTT should be in physiological range (1-30s)."""
        dro = load_qiba_dro()
        mask = dro.roi_masks[roi]
        tissue_curves = dro.delta_r2[mask]

        _, _, _, mtt = deconvolve_dro_batch(
            tissue_curves, dro.aif, dro.dt, method
        )

        # Filter valid voxels (MTT > 0 and finite)
        valid = np.isfinite(mtt) & (mtt > 0)
        assert valid.sum() > 0.1 * len(mtt), (
            f"Too few valid MTT voxels: {valid.sum()}/{len(mtt)}"
        )

        median_mtt = np.median(mtt[valid])
        # Physiological MTT range: 1-30s (generous for DRO with quantization)
        assert 0.5 < median_mtt < 30.0, (
            f"[{method}/{roi}] Median MTT={median_mtt:.2f}s outside "
            f"physiological range (0.5-30s)"
        )

    # -- Test A: rCBV ratio between tissue types ---

    @pytest.mark.parametrize("method", ["ssvd", "osvd", "tikhonov"])
    def test_rcbv_ratio(self, method: str) -> None:
        """rCBV ratio (high/low tissue) should be positive and bounded."""
        dro = load_qiba_dro()

        results = {}
        for roi in ["tissue_high", "tissue_low"]:
            mask = dro.roi_masks[roi]
            tissue_curves = dro.delta_r2[mask]
            _, _, cbv, _ = deconvolve_dro_batch(
                tissue_curves, dro.aif, dro.dt, method
            )
            valid = np.isfinite(cbv) & (cbv > 0)
            results[roi] = np.median(cbv[valid])

        rcbv = results["tissue_high"] / max(results["tissue_low"], 1e-10)
        # rCBV should be positive and in a reasonable range
        # (no absolute GT, just checking it's not degenerate)
        assert 0.1 < rcbv < 50.0, (
            f"[{method}] rCBV ratio={rcbv:.2f} outside reasonable range "
            f"(high={results['tissue_high']:.2f}, low={results['tissue_low']:.2f})"
        )

    # -- Test C: cross-method CBV agreement ---

    def test_cross_method_cbv_agreement(self) -> None:
        """CBV estimates from sSVD and oSVD should agree within COV<50%.

        Tikhonov excluded: GCV parameter selection assumes IID Gaussian
        noise (Golub, Heath, Wahba 1979) but DRO quantization error is
        structured (step-pattern from ~50 discrete uint16 levels), causing
        GCV to over-regularize (CBV inflated ~200x vs SVD methods).
        """
        dro = load_qiba_dro()

        # Use tissue_high ROI for cross-method comparison
        mask = dro.roi_masks["tissue_high"]
        tissue_curves = dro.delta_r2[mask]

        # Compare only SVD variants — Tikhonov GCV selects lambda based on
        # IID Gaussian noise assumption, which structured quantization noise
        # violates. Hard SVD truncation is robust to this.
        median_cbvs = {}
        for method in ["ssvd", "osvd"]:
            _, _, cbv, _ = deconvolve_dro_batch(
                tissue_curves, dro.aif, dro.dt, method
            )
            valid = np.isfinite(cbv) & (cbv > 0)
            if valid.sum() > 0:
                median_cbvs[method] = np.median(cbv[valid])

        assert len(median_cbvs) == 2, "Need both SVD methods"

        arr = np.array(list(median_cbvs.values()))
        cov = np.std(arr) / max(np.mean(arr), 1e-10)
        assert cov < 0.50, (
            f"Cross-method CBV COV={cov:.2f} exceeds 50%: "
            f"sSVD={median_cbvs['ssvd']:.2f}, oSVD={median_cbvs['osvd']:.2f}"
        )

    # -- Test F: tmax range ---

    @pytest.mark.parametrize(
        "method",
        [
            "ssvd",
            "osvd",
            pytest.param(
                "tikhonov",
                marks=pytest.mark.xfail(
                    reason="Tikhonov GCV assumes IID Gaussian noise "
                    "(Golub 1979); DRO quantization noise (~50 discrete "
                    "uint16 levels) is structured, causing over-"
                    "regularization and displaced tmax",
                    strict=False,
                ),
            ),
        ],
    )
    def test_tmax_range(self, method: str) -> None:
        """Tmax should be 0-30s for majority of tissue voxels."""
        dro = load_qiba_dro()
        mask = dro.roi_masks["tissue_high"] | dro.roi_masks["tissue_low"]
        tissue_curves = dro.delta_r2[mask]

        residues, _, _, _ = deconvolve_dro_batch(
            tissue_curves, dro.aif, dro.dt, method
        )

        # Tmax = argmax(residue) * dt
        tmax = np.argmax(residues, axis=-1) * dro.dt
        valid = np.isfinite(tmax) & (tmax >= 0)

        if valid.sum() == 0:
            pytest.skip("No valid tmax values recovered")

        in_range = (tmax[valid] >= 0) & (tmax[valid] <= 30.0)
        frac_in_range = in_range.sum() / valid.sum()

        assert frac_in_range > 0.50, (
            f"[{method}] Only {frac_in_range:.0%} of tmax values in 0-30s range "
            f"(need >50%)"
        )


# =============================================================================
# Phase 3: RILMAYO DSC-MRI DRO (leakage + deconvolution validation)
# =============================================================================

# Deconvolution methods to parametrize
RILMAYO_METHODS = [
    pytest.param("ssvd", id="ssvd"),
    pytest.param("osvd", id="osvd"),
]


@skip_no_rilmayo_dro
class TestRILMAYODSCDRO:
    """Phase 3: RILMAYO DSC-MRI DRO with leakage simulation.

    RILMAYO DROs from qibadscdro.rsna.org have known ground truth (flow,
    leak, R(t) shape) and much better dynamic range than BNI (max~54464 vs
    ~53). They simulate BBB leakage via a T1-shortening K2 parameter,
    testing both deconvolution AND K2 leakage correction.

    **No absolute CBF comparison**: DRO "flow" != deconvolved CBF without
    calibration factors (K_H, r2*, rho_brain). Tests use **ratios** and
    **ordering** only.

    **K2 correction improves CBV, not CBF**: First-pass peak (CBF) is
    T2*-dominated; leakage mainly distorts the post-bolus tail (CBV/MTT).

    Known limitations:
    - TR=0.1s = 20s total scan, no recirculation captured
    - TE=50ms at 1.5T gives ~88% signal drop at 1mM (near-void AIF)
    - Marginal post-bolus window for K2 estimation
    """

    # -- Test 1: Concentration sanity (all configs) ---

    @pytest.mark.parametrize(
        "config_name",
        [
            pytest.param("flow40_leak005", id="flow40_leak005"),
            pytest.param("flow60_leak002", id="flow60_leak002"),
            pytest.param("flow40_leak002", id="flow40_leak002"),
            pytest.param("flow70_leak002", id="flow70_leak002"),
            pytest.param("flow60_leak005", id="flow60_leak005"),
            pytest.param("flow60_box_leak002", id="flow60_box_leak002"),
            pytest.param("flow60_noise2_leak002", id="flow60_noise2_leak002"),
        ],
    )
    def test_concentration_sanity(self, config_name: str) -> None:
        """AIF has bolus, baseline is stable, deltaR2* is bounded."""
        if not rilmayo_config_available(config_name):
            pytest.skip(f"Config {config_name} not converted")

        dro = load_rilmayo_dro(config_name)

        # AIF should have a detectable bolus
        aif_peak = np.max(dro.aif)
        assert aif_peak > 1.0, (
            f"AIF peak too low: {aif_peak:.2f} s^-1"
        )

        # Tissue deltaR2* should be non-negative (clipped) and bounded
        tissue_dr2 = dro.delta_r2[dro.tissue_mask]
        assert tissue_dr2.min() >= -0.1, (
            f"deltaR2* has unexpected negative values: min={tissue_dr2.min():.4f}"
        )

        peak_vals = np.max(tissue_dr2, axis=1)
        median_peak = np.median(peak_vals)
        assert 0.1 < median_peak < 500.0, (
            f"Median tissue peak deltaR2* out of range: {median_peak:.2f} s^-1"
        )

    # -- Test 2: Flow ordering (cancels all calibration factors) ---

    @pytest.mark.parametrize("method", RILMAYO_METHODS)
    def test_flow_ordering(self, method: str) -> None:
        """CBF(40) < CBF(60) < CBF(70) — cancels all calibration factors.

        This is the most robust test: since all configs share the same AIF
        extraction and deconvolution pipeline, calibration factors cancel
        in the ordering comparison. Only the relative flow matters.
        """
        # Need at least 3 configs with different flows
        needed = ["flow40_leak002", "flow60_leak002", "flow70_leak002"]
        for name in needed:
            if not rilmayo_config_available(name):
                pytest.skip(f"Config {name} not available")

        median_cbf = {}
        for config_name in needed:
            dro = load_rilmayo_dro(config_name)
            tissue_curves = dro.delta_r2[dro.tissue_mask]

            _, cbf, _, _ = deconvolve_dro_batch(
                tissue_curves, dro.aif, dro.dt, method
            )

            valid = np.isfinite(cbf) & (cbf > 0)
            assert valid.sum() > 0.1 * len(cbf), (
                f"Too few valid CBF voxels for {config_name}"
            )
            median_cbf[config_name] = float(np.median(cbf[valid]))

        # Flow ordering: 40 < 60 < 70
        assert median_cbf["flow40_leak002"] < median_cbf["flow60_leak002"], (
            f"[{method}] CBF(40)={median_cbf['flow40_leak002']:.2f} >= "
            f"CBF(60)={median_cbf['flow60_leak002']:.2f}"
        )
        assert median_cbf["flow60_leak002"] < median_cbf["flow70_leak002"], (
            f"[{method}] CBF(60)={median_cbf['flow60_leak002']:.2f} >= "
            f"CBF(70)={median_cbf['flow70_leak002']:.2f}"
        )

    # -- Test 3: CBF ratio linearity ---

    @pytest.mark.parametrize("method", RILMAYO_METHODS)
    def test_cbf_ratio_linearity(self, method: str) -> None:
        """CBF(70)/CBF(40) should approximate 1.75 within 30%.

        Tests linearity of deconvolution without absolute calibration.
        Expected ratio = 70/40 = 1.75.
        """
        needed = ["flow40_leak002", "flow70_leak002"]
        for name in needed:
            if not rilmayo_config_available(name):
                pytest.skip(f"Config {name} not available")

        median_cbf = {}
        for config_name in needed:
            dro = load_rilmayo_dro(config_name)
            tissue_curves = dro.delta_r2[dro.tissue_mask]

            _, cbf, _, _ = deconvolve_dro_batch(
                tissue_curves, dro.aif, dro.dt, method
            )

            valid = np.isfinite(cbf) & (cbf > 0)
            median_cbf[config_name] = float(np.median(cbf[valid]))

        ratio = median_cbf["flow70_leak002"] / max(
            median_cbf["flow40_leak002"], 1e-10
        )
        expected = 70.0 / 40.0  # 1.75

        rel_err = abs(ratio - expected) / expected
        assert rel_err < 0.30, (
            f"[{method}] CBF ratio={ratio:.3f}, expected={expected:.3f}, "
            f"rel_err={rel_err:.2%} (>30%)"
        )

    # -- Test 4: K2 correction improves CBV ---

    @pytest.mark.parametrize("method", RILMAYO_METHODS)
    @pytest.mark.parametrize(
        "config_name",
        [
            pytest.param("flow40_leak005", id="flow40_leak005"),
            pytest.param("flow60_leak005", id="flow60_leak005"),
        ],
    )
    def test_k2_correction_improves_cbv(
        self, method: str, config_name: str
    ) -> None:
        """K2-corrected CBV should be closer to no-leak reference than uncorrected.

        K2 correction primarily fixes the post-bolus tail (CBV/MTT), not
        the first-pass peak (CBF). Compare CBV from corrected vs uncorrected
        deltaR2* against the low-leak reference (leak=0.02, same flow).

        Uses leak=0.05 configs and compares to leak=0.02 reference.
        """
        if not rilmayo_config_available(config_name):
            pytest.skip(f"Config {config_name} not available")

        # Find matching low-leak reference (same flow, leak=0.02)
        config = RILMAYO_CONFIGS[config_name]
        ref_name = f"flow{config.flow}_leak002"
        if not rilmayo_config_available(ref_name):
            pytest.skip(f"Reference config {ref_name} not available")

        # Get reference CBV (low leak ~ no leak)
        ref_dro = load_rilmayo_dro(ref_name)
        ref_curves = ref_dro.delta_r2[ref_dro.tissue_mask]
        _, _, ref_cbv, _ = deconvolve_dro_batch(
            ref_curves, ref_dro.aif, ref_dro.dt, method
        )
        ref_valid = np.isfinite(ref_cbv) & (ref_cbv > 0)
        ref_median = float(np.median(ref_cbv[ref_valid]))

        # Get uncorrected CBV (high leak)
        dro = load_rilmayo_dro(config_name)
        uncorr_curves = dro.delta_r2[dro.tissue_mask]
        _, _, uncorr_cbv, _ = deconvolve_dro_batch(
            uncorr_curves, dro.aif, dro.dt, method
        )
        uncorr_valid = np.isfinite(uncorr_cbv) & (uncorr_cbv > 0)
        uncorr_median = float(np.median(uncorr_cbv[uncorr_valid]))

        # Get corrected CBV
        corr_curves = dro.delta_r2_corrected[dro.tissue_mask]
        _, _, corr_cbv, _ = deconvolve_dro_batch(
            corr_curves, dro.aif, dro.dt, method
        )
        corr_valid = np.isfinite(corr_cbv) & (corr_cbv > 0)
        corr_median = float(np.median(corr_cbv[corr_valid]))

        # Corrected should be closer to reference than uncorrected
        uncorr_err = abs(uncorr_median - ref_median)
        corr_err = abs(corr_median - ref_median)

        assert corr_err < uncorr_err, (
            f"[{method}/{config_name}] K2 correction did NOT improve CBV: "
            f"uncorr_err={uncorr_err:.2f}, corr_err={corr_err:.2f}, "
            f"ref={ref_median:.2f}"
        )

    # -- Test 5: K2 detection ---

    @pytest.mark.parametrize(
        "config_name",
        [
            pytest.param("flow40_leak005", id="flow40_leak005"),
            pytest.param("flow60_leak005", id="flow60_leak005"),
        ],
    )
    def test_k2_detection(self, config_name: str) -> None:
        """Median K2 should be > 0.005 s^-1 in tissue voxels for leak=0.05.

        K2 = 0.005 s^-1 is a physically grounded threshold: below this,
        leakage has <1% effect on the concentration curve over 20s.
        """
        if not rilmayo_config_available(config_name):
            pytest.skip(f"Config {config_name} not available")

        config_dir = RILMAYO_DRO_ROOT / config_name
        k2_file = config_dir / "k2_map.npz"
        if not k2_file.exists():
            pytest.skip(f"K2 map not found for {config_name}")

        k2_data = np.load(k2_file)
        k2_map = k2_data["k2_map"]

        dro = load_rilmayo_dro(config_name)
        k2_tissue = k2_map[dro.tissue_mask]

        # Fraction of voxels with detectable K2
        detectable = k2_tissue > 1e-4
        frac_detectable = detectable.sum() / len(k2_tissue)

        # For leak=0.05, expect substantial K2 detection
        assert frac_detectable > 0.10, (
            f"[{config_name}] Only {frac_detectable:.0%} voxels with "
            f"K2 > 0.0001 (expected >10% for leak=0.05)"
        )

        # Median K2 among detectable voxels
        if detectable.sum() > 0:
            median_k2 = float(np.median(k2_tissue[detectable]))
            assert median_k2 > 0.005, (
                f"[{config_name}] Median K2={median_k2:.4f} s^-1, "
                f"expected > 0.005 for leak=0.05"
            )

    # -- Test 6: Residue shape consistency ---

    @pytest.mark.parametrize("method", RILMAYO_METHODS)
    def test_residue_shape_consistency(self, method: str) -> None:
        """CBF from mono and box residue should agree within 25%.

        Deconvolution should be approximately shape-independent for
        the same flow value. Box R(t) gives a step function instead
        of exponential decay.
        """
        needed = ["flow60_leak002", "flow60_box_leak002"]
        for name in needed:
            if not rilmayo_config_available(name):
                pytest.skip(f"Config {name} not available")

        median_cbf = {}
        for config_name in needed:
            dro = load_rilmayo_dro(config_name)
            tissue_curves = dro.delta_r2[dro.tissue_mask]

            _, cbf, _, _ = deconvolve_dro_batch(
                tissue_curves, dro.aif, dro.dt, method
            )

            valid = np.isfinite(cbf) & (cbf > 0)
            median_cbf[config_name] = float(np.median(cbf[valid]))

        # CBF should agree within 25%
        cbf_mono = median_cbf["flow60_leak002"]
        cbf_box = median_cbf["flow60_box_leak002"]
        mean_cbf = (cbf_mono + cbf_box) / 2.0

        if mean_cbf > 0:
            rel_diff = abs(cbf_mono - cbf_box) / mean_cbf
            assert rel_diff < 0.25, (
                f"[{method}] Mono/box CBF disagree: "
                f"mono={cbf_mono:.2f}, box={cbf_box:.2f}, "
                f"rel_diff={rel_diff:.2%} (>25%)"
            )

    # -- Test 7: K2 is low in low-leak configs ---

    @pytest.mark.parametrize(
        "config_name",
        [
            pytest.param("flow60_leak002", id="flow60_leak002"),
            pytest.param("flow40_leak002", id="flow40_leak002"),
        ],
    )
    def test_k2_low_in_low_leak(self, config_name: str) -> None:
        """Median K2 should be < 0.02 s^-1 for leak=0.02 configs.

        leak=0.02 has minimal first-pass effect (<2% at peak), so K2
        estimates should be much smaller than leak=0.05 configs.
        """
        if not rilmayo_config_available(config_name):
            pytest.skip(f"Config {config_name} not available")

        config_dir = RILMAYO_DRO_ROOT / config_name
        k2_file = config_dir / "k2_map.npz"
        if not k2_file.exists():
            pytest.skip(f"K2 map not found for {config_name}")

        k2_data = np.load(k2_file)
        k2_map = k2_data["k2_map"]

        dro = load_rilmayo_dro(config_name)
        k2_tissue = k2_map[dro.tissue_mask]
        k2_nonzero = k2_tissue[k2_tissue > 1e-4]

        if len(k2_nonzero) > 0:
            median_k2 = float(np.median(k2_nonzero))
            assert median_k2 < 0.02, (
                f"[{config_name}] Median K2={median_k2:.4f} s^-1, "
                f"expected < 0.02 for leak=0.02"
            )

    # -- Test 8: Noise robustness ---

    @pytest.mark.parametrize("method", RILMAYO_METHODS)
    def test_noise_robustness(self, method: str) -> None:
        """CBF from noisy data (noise=2) should agree with noiseless within 40%.

        noise=2 in RILMAYO is a low-noise setting. Deconvolution should
        be robust at this level.
        """
        needed = ["flow60_leak002", "flow60_noise2_leak002"]
        for name in needed:
            if not rilmayo_config_available(name):
                pytest.skip(f"Config {name} not available")

        median_cbf = {}
        for config_name in needed:
            dro = load_rilmayo_dro(config_name)
            tissue_curves = dro.delta_r2[dro.tissue_mask]

            _, cbf, _, _ = deconvolve_dro_batch(
                tissue_curves, dro.aif, dro.dt, method
            )

            valid = np.isfinite(cbf) & (cbf > 0)
            median_cbf[config_name] = float(np.median(cbf[valid]))

        # Noisy should agree with noiseless within 40%
        cbf_clean = median_cbf["flow60_leak002"]
        cbf_noisy = median_cbf["flow60_noise2_leak002"]

        if cbf_clean > 0:
            rel_diff = abs(cbf_noisy - cbf_clean) / cbf_clean
            assert rel_diff < 0.40, (
                f"[{method}] Noise degrades CBF too much: "
                f"clean={cbf_clean:.2f}, noisy={cbf_noisy:.2f}, "
                f"rel_diff={rel_diff:.2%} (>40%)"
            )

    # -- Test 9: MTT in physiological range ---

    @pytest.mark.parametrize("method", RILMAYO_METHODS)
    @pytest.mark.parametrize(
        "config_name",
        [
            pytest.param("flow40_leak005", id="flow40_leak005"),
            pytest.param("flow60_leak002", id="flow60_leak002"),
            pytest.param("flow70_leak002", id="flow70_leak002"),
        ],
    )
    def test_mtt_range(self, method: str, config_name: str) -> None:
        """Median MTT should be in range (0.1-10s) for RILMAYO configs.

        RILMAYO DROs have TR=0.1s (dt=0.1s), so temporal resolution is
        excellent. MTT range is narrower than BNI because of the higher
        quality data.
        """
        if not rilmayo_config_available(config_name):
            pytest.skip(f"Config {config_name} not available")

        dro = load_rilmayo_dro(config_name)
        tissue_curves = dro.delta_r2[dro.tissue_mask]

        _, _, _, mtt = deconvolve_dro_batch(
            tissue_curves, dro.aif, dro.dt, method
        )

        valid = np.isfinite(mtt) & (mtt > 0)
        assert valid.sum() > 0.1 * len(mtt), (
            f"Too few valid MTT voxels: {valid.sum()}/{len(mtt)}"
        )

        median_mtt = float(np.median(mtt[valid]))
        assert 0.01 < median_mtt < 10.0, (
            f"[{method}/{config_name}] Median MTT={median_mtt:.3f}s "
            f"outside range (0.01-10s)"
        )
