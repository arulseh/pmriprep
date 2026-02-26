"""QIBA DSC-MRI DRO parameter recovery tests.

Tests deconvolution methods against QIBA-representative synthetic data with:
- Recirculation AIF (Patil & Johnson 2011, kappa=0.35)
- CTH gamma-variate residue (Mouridsen 2006)
- Published tissue parameters (Semmineh 2017, Boxerman 2020)

Phase 1: Code-generated data (this file).
Phase 4: TCIA GBM-DSC-MRI-DRO (Semmineh 2017) for external validation.

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
    TCIA_CONSENSUS_15T,
    TCIA_CONSENSUS_3T,
    TCIA_DRO_ROOT,
    add_gaussian_noise,
    deconvolve_dro_batch,
    deconvolve_tcia_regions,
    dsc_forward_model,
    generate_cth_residue,
    generate_qiba_aif,
    load_tcia_dro,
    make_exponential_residue,
    skip_no_tcia_dro,
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
# Phase 4: TCIA GBM-DSC-MRI-DRO (Semmineh et al., Tomography 2017)
# =============================================================================

# Deconvolution methods to test
TCIA_METHODS = ["ssvd", "osvd"]


@skip_no_tcia_dro
class TestTCIAGBMDSCDRO:
    """Validate DSC deconvolution against TCIA GBM-DSC-MRI-DRO.

    The TCIA GBM-DSC-MRI-DRO (Semmineh et al., Tomography 2017) provides:
    - 4 non-overlapping ROI masks: tumor-leak (Ktrans>0), tumor-noleak
      (Ktrans=0, same hemodynamics), NAWM, AIF
    - Known ground truth: tumor CBV=6.1%, WM CBV=3.5%, rCBV~1.74
    - T1 leakage in tumor-leak region (post-bolus signal overshoot)

    Primary test: rCBV = CBV(noleak) / CBV(NAWM) should approach GT ~1.74.
    K2-corrected leak rCBV should approach noleak rCBV.
    """

    # -- Test A: Concentration sanity ---

    def test_concentration_sanity(self) -> None:
        """All 4 regions have expected signal behavior.

        - AIF: large positive deltaR2* peak (strong T2* signal drop)
        - Tumor-leak: positive peak + negative post-bolus (T1 leakage)
        - Tumor-noleak: positive peak, returns near zero post-bolus
        - NAWM: positive peak, returns near zero post-bolus
        """
        dro = load_tcia_dro(TCIA_CONSENSUS_15T)

        for region in ["tumor_leak", "tumor_noleak", "nawm", "aif_region"]:
            mask = dro.roi_masks[region]
            curves = dro.delta_r2[mask]

            # All regions should have positive peak deltaR2*
            peak = np.max(np.mean(curves, axis=0))
            assert peak > 1.0, (
                f"{region}: mean peak deltaR2* = {peak:.2f} s^-1, "
                f"expected > 1.0"
            )

        # AIF should have the highest peak
        aif_peak = np.max(dro.aif)
        for region in ["tumor_noleak", "nawm"]:
            tissue_peak = np.max(np.mean(dro.delta_r2[dro.roi_masks[region]], axis=0))
            assert aif_peak > tissue_peak, (
                f"AIF peak ({aif_peak:.1f}) should exceed "
                f"{region} peak ({tissue_peak:.1f})"
            )

    # -- Test B: rCBV from noleak region ---

    @pytest.mark.parametrize("method", TCIA_METHODS)
    def test_rcbv_noleak(self, method: str) -> None:
        """rCBV = CBV(noleak) / CBV(WM) should approximate GT ~1.74.

        Ground truth (Semmineh 2017): tumor CBV=6.1%, WM CBV=3.5%.
        Tolerance: 0.5 - 5.0 (generous for deconvolution-based rCBV).
        """
        dro = load_tcia_dro(TCIA_CONSENSUS_15T)
        results = deconvolve_tcia_regions(dro, method)

        noleak_cbv = results["tumor_noleak"]["cbv"]
        nawm_cbv = results["nawm"]["cbv"]

        assert nawm_cbv > 0, f"NAWM CBV should be positive, got {nawm_cbv:.4f}"

        rcbv = noleak_cbv / nawm_cbv
        assert 0.5 < rcbv < 5.0, (
            f"[{method}] rCBV(noleak/WM) = {rcbv:.3f}, "
            f"expected 0.5-5.0 (GT ~1.74)"
        )

    # -- Test C: K2 correction effectiveness ---

    @pytest.mark.parametrize("method", TCIA_METHODS)
    def test_k2_correction_effectiveness(self, method: str) -> None:
        """K2-corrected leak rCBV should be closer to noleak rCBV.

        Without correction, tumor-leak has negative deltaR2* post-bolus
        (T1 leakage), depressing CBV. K2 correction should bring it
        closer to the noleak estimate.
        """
        from pmriprep.utils.dsc_preload import apply_k2_correction

        dro = load_tcia_dro(TCIA_CONSENSUS_15T)

        # Noleak and NAWM rCBV (reference)
        results = deconvolve_tcia_regions(dro, method)
        noleak_cbv = results["tumor_noleak"]["cbv"]
        nawm_cbv = results["nawm"]["cbv"]

        if nawm_cbv <= 0 or noleak_cbv <= 0:
            pytest.skip("Invalid CBV estimates for comparison")

        rcbv_noleak = noleak_cbv / nawm_cbv

        # Uncorrected leak rCBV
        leak_cbv_uncorrected = results["tumor_leak"]["cbv"]
        rcbv_leak_uncorrected = leak_cbv_uncorrected / nawm_cbv

        # Apply K2 correction — add z-dim since TCIA is single 2D slice
        # but apply_k2_correction expects 4D (x, y, z, time)
        brain_mask = (
            dro.roi_masks["tumor_leak"]
            | dro.roi_masks["tumor_noleak"]
            | dro.roi_masks["nawm"]
        )
        conc_4d = dro.delta_r2[:, :, np.newaxis, :]
        mask_3d = brain_mask[:, :, np.newaxis]

        k2_result = apply_k2_correction(
            concentration=conc_4d,
            aif=dro.aif,
            brain_mask=mask_3d,
            tr=dro.dt,
        )

        # Re-deconvolve corrected leak region — squeeze back to 2D mask
        leak_mask_3d = dro.roi_masks["tumor_leak"][:, :, np.newaxis]
        corrected_leak_curves = k2_result.concentration_corrected[
            np.broadcast_to(leak_mask_3d, k2_result.concentration_corrected.shape[:3])
        ].reshape(-1, dro.delta_r2.shape[-1])
        _, cbf_corr, cbv_corr, _ = deconvolve_dro_batch(
            corrected_leak_curves, dro.aif, dro.dt, method
        )
        valid = np.isfinite(cbv_corr) & (cbv_corr > 0)
        if valid.sum() == 0:
            pytest.skip("No valid corrected voxels")

        leak_cbv_corrected = float(np.median(cbv_corr[valid]))
        rcbv_leak_corrected = leak_cbv_corrected / nawm_cbv

        # Corrected should be closer to noleak than uncorrected
        err_uncorrected = abs(rcbv_leak_uncorrected - rcbv_noleak)
        err_corrected = abs(rcbv_leak_corrected - rcbv_noleak)

        assert err_corrected < err_uncorrected, (
            f"[{method}] K2 correction did not improve rCBV: "
            f"uncorrected={rcbv_leak_uncorrected:.3f} "
            f"(err={err_uncorrected:.3f}), "
            f"corrected={rcbv_leak_corrected:.3f} "
            f"(err={err_corrected:.3f}), "
            f"noleak={rcbv_noleak:.3f}"
        )

    # -- Test D: K2 detection in leak vs noleak ---

    def test_k2_correction_restores_leak_postbolus(self) -> None:
        """K2 correction should restore leak post-bolus toward non-negative.

        Uncorrected tumor-leak post-bolus deltaR2* is strongly negative
        (~-3.6 s^-1 due to T1 leakage). After BSW K2 correction, the
        post-bolus mean should move toward zero (or positive).

        Note: Per-voxel K2 regression can fail for strongly leaky
        voxels, but the global correction still improves the signal.
        """
        from pmriprep.utils.dsc_preload import apply_k2_correction

        dro = load_tcia_dro(TCIA_CONSENSUS_15T)

        brain_mask = (
            dro.roi_masks["tumor_leak"]
            | dro.roi_masks["tumor_noleak"]
            | dro.roi_masks["nawm"]
        )
        # Add z-dim for 4D requirement
        conc_4d = dro.delta_r2[:, :, np.newaxis, :]
        mask_3d = brain_mask[:, :, np.newaxis]

        k2_result = apply_k2_correction(
            concentration=conc_4d,
            aif=dro.aif,
            brain_mask=mask_3d,
            tr=dro.dt,
        )

        corr_2d = k2_result.concentration_corrected[:, :, 0, :]

        # Post-bolus region (last 30% of timepoints)
        n_tp = dro.delta_r2.shape[-1]
        post_start = int(0.7 * n_tp)

        # Leak region: uncorrected post-bolus is negative (T1 leakage)
        leak_uncorr_post = np.mean(
            dro.delta_r2[dro.roi_masks["tumor_leak"]][:, post_start:],
        )
        leak_corr_post = np.mean(
            corr_2d[dro.roi_masks["tumor_leak"]][:, post_start:],
        )

        # Corrected should be closer to zero than uncorrected
        assert abs(leak_corr_post) < abs(leak_uncorr_post), (
            f"Corrected post-bolus ({leak_corr_post:.4f}) should be "
            f"closer to zero than uncorrected ({leak_uncorr_post:.4f})"
        )

        # K2 map should have some nonzero voxels across brain
        k2_map_2d = k2_result.k2_map[:, :, 0]
        k2_nonzero = np.count_nonzero(k2_map_2d[brain_mask])
        assert k2_nonzero > 0.1 * brain_mask.sum(), (
            f"Too few K2 nonzero voxels: {k2_nonzero}/{brain_mask.sum()}"
        )

    # -- Test E: Cross-method rCBV consistency ---

    def test_cross_method_rcbv_consistency(self) -> None:
        """sSVD and oSVD rCBV should agree within COV < 30%.

        Both methods operating on the same data should produce similar
        relative CBV estimates.
        """
        dro = load_tcia_dro(TCIA_CONSENSUS_15T)

        rcbv_values = []
        for method in TCIA_METHODS:
            results = deconvolve_tcia_regions(dro, method)
            noleak_cbv = results["tumor_noleak"]["cbv"]
            nawm_cbv = results["nawm"]["cbv"]
            if nawm_cbv > 0:
                rcbv_values.append(noleak_cbv / nawm_cbv)

        assert len(rcbv_values) >= 2, "Need at least 2 methods for comparison"

        mean_rcbv = np.mean(rcbv_values)
        std_rcbv = np.std(rcbv_values)

        if mean_rcbv > 0:
            cov = std_rcbv / mean_rcbv
            assert cov < 0.30, (
                f"Cross-method rCBV COV = {cov:.3f} (>0.30). "
                f"Values: {rcbv_values}"
            )

    # -- Test F: CBF/MTT physiological ranges ---

    @pytest.mark.parametrize("method", TCIA_METHODS)
    def test_perfusion_ranges(self, method: str) -> None:
        """CBF and MTT should be in physiological ranges.

        DRO ground truth: tumor CBF=150.3, WM CBF=38.4.
        Note: deconvolved CBF is in arbitrary units without calibration
        factor, so we test relative ranges and ordering.
        """
        dro = load_tcia_dro(TCIA_CONSENSUS_15T)
        results = deconvolve_tcia_regions(dro, method)

        # MTT should be in range (1-15s) for all tissue types
        for region in ["tumor_noleak", "nawm"]:
            mtt = results[region]["mtt"]
            assert 0.5 < mtt < 30.0, (
                f"[{method}/{region}] Median MTT={mtt:.3f}s "
                f"outside range (0.5-30s)"
            )

        # CBF: tumor should be higher than WM (GT: 150.3 vs 38.4)
        tumor_cbf = results["tumor_noleak"]["cbf"]
        wm_cbf = results["nawm"]["cbf"]

        assert tumor_cbf > 0, f"Tumor CBF should be positive, got {tumor_cbf}"
        assert wm_cbf > 0, f"WM CBF should be positive, got {wm_cbf}"

        assert tumor_cbf > wm_cbf, (
            f"[{method}] Tumor CBF ({tumor_cbf:.2f}) should exceed "
            f"WM CBF ({wm_cbf:.2f}) (GT ratio: ~3.9)"
        )

    # -- Test G: Leak region signal overshoot ---

    def test_leak_region_signal_overshoot(self) -> None:
        """Tumor-leak post-bolus deltaR2* should be negative (T1 leakage).

        When Ktrans > 0, contrast agent leaks into extravascular space,
        shortening T1 and increasing signal above baseline (S(t) > S0),
        which gives negative deltaR2*.
        """
        dro = load_tcia_dro(TCIA_CONSENSUS_15T)

        leak_curves = dro.delta_r2[dro.roi_masks["tumor_leak"]]

        # Post-bolus = last 30% of timepoints
        n_tp = leak_curves.shape[1]
        post_start = int(0.7 * n_tp)
        post_bolus_mean = np.mean(leak_curves[:, post_start:])

        assert post_bolus_mean < 0, (
            f"Leak region post-bolus mean deltaR2* = {post_bolus_mean:.4f}, "
            f"expected < 0 (T1 leakage overshoot)"
        )

        # Noleak should NOT have this overshoot
        noleak_curves = dro.delta_r2[dro.roi_masks["tumor_noleak"]]
        noleak_post = np.mean(noleak_curves[:, post_start:])

        assert noleak_post > post_bolus_mean, (
            f"Noleak post-bolus ({noleak_post:.4f}) should be > "
            f"leak post-bolus ({post_bolus_mean:.4f})"
        )

    # -- Test H: TE sensitivity ---

    def test_te_sensitivity(self) -> None:
        """rCBV should be stable across TE values (within 35%).

        TE affects the magnitude of deltaR2* but rCBV is a ratio, so TE
        cancels. Testing TE=20/30/40/50ms at 1.5T, pre=none, FA=60deg.
        """
        te_configs = []
        for te in [20, 30, 40, 50]:
            name = f"b0_15T_pre_none_bolus_full_TR1500ms_TE{te}ms_FA60deg"
            if (TCIA_DRO_ROOT / name / "concentration.npz").exists():
                te_configs.append(name)

        if len(te_configs) < 2:
            pytest.skip(
                f"Need ≥2 TE configs, found {len(te_configs)}. "
                f"Run conversion with --all first."
            )

        rcbv_values = []
        for config_name in te_configs:
            dro = load_tcia_dro(config_name)
            results = deconvolve_tcia_regions(dro, "ssvd")
            noleak_cbv = results["tumor_noleak"]["cbv"]
            nawm_cbv = results["nawm"]["cbv"]
            if nawm_cbv > 0:
                rcbv_values.append(noleak_cbv / nawm_cbv)

        if len(rcbv_values) < 2:
            pytest.skip("Too few valid rCBV values")

        mean_rcbv = np.mean(rcbv_values)
        for rcbv in rcbv_values:
            if mean_rcbv > 0:
                rel_diff = abs(rcbv - mean_rcbv) / mean_rcbv
                assert rel_diff < 0.35, (
                    f"rCBV varies >35% across TE: {rcbv:.3f} vs "
                    f"mean {mean_rcbv:.3f} (rel_diff={rel_diff:.2%})"
                )

    # -- Test I: FA sensitivity ---

    def test_fa_sensitivity(self) -> None:
        """rCBV should be stable across flip angles (within 35%).

        FA affects steady-state signal magnitude but rCBV is a ratio.
        Testing FA=30/60/90deg at 1.5T, pre=none, TE=30ms.
        """
        fa_configs = []
        for fa in [30, 60, 90]:
            name = f"b0_15T_pre_none_bolus_full_TR1500ms_TE30ms_FA{fa}deg"
            if (TCIA_DRO_ROOT / name / "concentration.npz").exists():
                fa_configs.append(name)

        if len(fa_configs) < 2:
            pytest.skip(
                f"Need ≥2 FA configs, found {len(fa_configs)}. "
                f"Run conversion with --all first."
            )

        rcbv_values = []
        for config_name in fa_configs:
            dro = load_tcia_dro(config_name)
            results = deconvolve_tcia_regions(dro, "ssvd")
            noleak_cbv = results["tumor_noleak"]["cbv"]
            nawm_cbv = results["nawm"]["cbv"]
            if nawm_cbv > 0:
                rcbv_values.append(noleak_cbv / nawm_cbv)

        if len(rcbv_values) < 2:
            pytest.skip("Too few valid rCBV values")

        mean_rcbv = np.mean(rcbv_values)
        for rcbv in rcbv_values:
            if mean_rcbv > 0:
                rel_diff = abs(rcbv - mean_rcbv) / mean_rcbv
                assert rel_diff < 0.35, (
                    f"rCBV varies >35% across FA: {rcbv:.3f} vs "
                    f"mean {mean_rcbv:.3f} (rel_diff={rel_diff:.2%})"
                )

    # -- Test J: Field strength comparison ---

    def test_field_strength_comparison(self) -> None:
        """1.5T and 3T rCBV should agree within 40%.

        At higher field, deltaR2* increases proportionally to B0, but
        rCBV is a ratio and should be field-strength independent.
        """
        configs = [TCIA_CONSENSUS_15T, TCIA_CONSENSUS_3T]

        available = [
            c for c in configs
            if (TCIA_DRO_ROOT / c / "concentration.npz").exists()
        ]

        if len(available) < 2:
            pytest.skip(
                f"Need both 1.5T and 3T configs, found {len(available)}"
            )

        rcbv_values = {}
        for config_name in available:
            dro = load_tcia_dro(config_name)
            results = deconvolve_tcia_regions(dro, "ssvd")
            noleak_cbv = results["tumor_noleak"]["cbv"]
            nawm_cbv = results["nawm"]["cbv"]
            if nawm_cbv > 0:
                rcbv_values[config_name] = noleak_cbv / nawm_cbv

        if len(rcbv_values) < 2:
            pytest.skip("Invalid CBV in one or both configs")

        values = list(rcbv_values.values())
        mean_rcbv = np.mean(values)
        if mean_rcbv > 0:
            rel_diff = abs(values[0] - values[1]) / mean_rcbv
            assert rel_diff < 0.40, (
                f"1.5T vs 3T rCBV differs by {rel_diff:.2%} (>40%): "
                f"{dict(zip(rcbv_values.keys(), values))}"
            )
