"""DSC recovery with generated recirculation AIFs and tissue residue curves."""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest

from tests.dsc_accuracy import OSVD_RESIDUE_MTT, P2_NO_CORNER, assert_accuracy

from pmriprep.utils.dsc_deconvolution import (
    TikhonovSelection,
    svd_deconvolve_circulant,
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
    tikhonov_deconvolve,
)

from .conftest import add_gaussian_noise, fine_grid_forward, make_exponential_residue
from .dsc_recirculation import (
    GAMMA_RESIDUE_SHAPES,
    NOISE_LEVELS,
    TISSUE_PARAMS,
    generate_gamma_residue,
    generate_recirculation_aif,
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


def _make_exponential_data(
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
    """Generate synthetic test data with exponential residue.

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float]
        Tuple of (tissue, aif, dt, time_sec, cbf, mtt).
    """
    params = TISSUE_PARAMS[tissue_key]
    noise = NOISE_LEVELS[noise_key]

    aif, time_sec = generate_recirculation_aif(
        n_timepoints=N_TIMEPOINTS,
        dt_sec=DT_SEC,
        baseline_frames=BASELINE_FRAMES,
    )

    residue = make_exponential_residue(params.cbf, params.mtt, N_TIMEPOINTS, DT_SEC)
    tissue = fine_grid_forward(aif, residue, DT_SEC)

    if noise.fraction > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise.fraction * peak, seed=seed)

    return tissue, aif, DT_SEC, time_sec, params.cbf, params.mtt


def _make_gamma_residue_data(
    tissue_key: str,
    shape_key: str,
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
    """Generate synthetic test data with a gamma transit-time survival residue.

    beta = mtt / alpha (from gamma distribution mean = alpha * beta).

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]
        Tuple of (tissue, aif, dt, time_sec, cbf, mtt, alpha).
    """
    params = TISSUE_PARAMS[tissue_key]
    shape = GAMMA_RESIDUE_SHAPES[shape_key]
    noise = NOISE_LEVELS[noise_key]

    aif, time_sec = generate_recirculation_aif(
        n_timepoints=N_TIMEPOINTS,
        dt_sec=DT_SEC,
        baseline_frames=BASELINE_FRAMES,
    )

    beta = params.mtt / shape.alpha
    residue = generate_gamma_residue(params.cbf, shape.alpha, beta, N_TIMEPOINTS, DT_SEC)
    tissue = fine_grid_forward(aif, residue, DT_SEC)

    if noise.fraction > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise.fraction * peak, seed=seed)

    return tissue, aif, DT_SEC, time_sec, params.cbf, params.mtt, shape.alpha


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

# Gamma residue combinations: 3 shapes x 2 tissues = 6
GAMMA_RESIDUE_COMBOS = [
    pytest.param("normal_gm", "low_dispersion", id="normal_gm-low_dispersion"),
    pytest.param("normal_gm", "moderate_dispersion", id="normal_gm-moderate_dispersion"),
    pytest.param("normal_gm", "high_dispersion", id="normal_gm-high_dispersion"),
    pytest.param("tumor", "low_dispersion", id="tumor-low_dispersion"),
    pytest.param("tumor", "moderate_dispersion", id="tumor-moderate_dispersion"),
    pytest.param("tumor", "high_dispersion", id="tumor-high_dispersion"),
]

# =============================================================================
# sSVD with exponential residue
# =============================================================================

# (noise_key, threshold, cbf_tol, mtt_tol)
SSVD_NOISE = [
    pytest.param("zero", 0.001, 0.10, 0.15, id="zero"),
    pytest.param("low", 0.02, 0.45, 1.70, id="low"),
    pytest.param("medium", 0.05, 0.55, 1.60, id="medium"),
    pytest.param("high", 0.10, 0.70, 2.10, id="high"),
]


class TestSSVDExponential:
    """sSVD recovery with synthetic AIF (recirculation) and exponential residue."""

    @pytest.mark.parametrize("noise_key,threshold,cbf_tol,mtt_tol", SSVD_NOISE)
    @pytest.mark.parametrize("tissue_key", SVD_TISSUES)
    def test_ssvd_exponential(
        self,
        tissue_key: str,
        noise_key: str,
        threshold: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test sSVD recovery with synthetic recirculation AIF."""

        tissue, aif, dt, time_sec, cbf, mtt = _make_exponential_data(
            tissue_key, noise_key
        )

        result = svd_deconvolve_standard(tissue, aif, dt, threshold=threshold)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# cSVD with exponential residue — delay detection only
# =============================================================================


class TestCSVD:
    """cSVD delay preservation with synthetic AIF."""

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
        """Test that cSVD recovers reasonable residue with delayed synthetic data."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_exponential_data(
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

# noise_key, threshold_min, cbf_tol, mtt_tol. threshold_min is the production
# oSVD search floor (the default the pipeline runs with); all tiers use it so the
# test exercises the deployed operating point rather than a finer-grid override.
OSVD_NOISY = [
    pytest.param("low", 0.01, 0.49, 1.10, id="low"),
    pytest.param("medium", 0.01, 0.55, 1.90, id="medium"),
    pytest.param("high", 0.01, 0.75, 2.40, id="high"),
]


class TestOSVDExponential:
    """oSVD recovery with synthetic AIF (recirculation) and exponential residue.

    oSVD uses an OI-regularized block-circulant matrix. Tissue is generated
    from frame-interpolated AIF/residue curves, whose interpolation assumptions
    are shared with some fitted operators. Full-period residue MTT carries
    the residue-peak error from regularization and temporal sampling.
    """

    @pytest.mark.parametrize("tissue_key", SVD_TISSUES)
    def test_osvd_zero_noise(self, tissue_key: str) -> None:
        """oSVD produces finite positive CBF and residue MTT at zero noise."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_exponential_data(
            tissue_key, "zero"
        )
        result = svd_deconvolve_oscillation(tissue, aif, dt)
        assert np.isfinite(result.cbf) and np.isfinite(result.mtt)
        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        assert result.mtt > 0, f"MTT should be positive, got {result.mtt}"

    @OSVD_RESIDUE_MTT
    @pytest.mark.parametrize("tissue_key", SVD_TISSUES)
    def test_osvd_zero_noise_mtt(self, tissue_key: str) -> None:
        """Evaluate the 20% residue-MTT recovery target at zero noise."""
        tissue, aif, dt, _, _, mtt = _make_exponential_data(tissue_key, "zero")
        result = svd_deconvolve_oscillation(tissue, aif, dt)
        assert np.isfinite(result.mtt) and result.mtt > 0
        assert_accuracy(abs(result.mtt - mtt) / mtt, 0.20, label="oSVD residue-MTT")

    @pytest.mark.parametrize("noise_key,thresh_min,cbf_tol,mtt_tol", OSVD_NOISY)
    @pytest.mark.parametrize("tissue_key", SVD_TISSUES)
    def test_osvd_noisy(
        self,
        tissue_key: str,
        noise_key: str,
        thresh_min: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery with synthetic recirculation AIF and noise."""

        tissue, aif, dt, time_sec, cbf, mtt = _make_exponential_data(
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


class TestTikhonovExponential:
    """Tikhonov recovery with synthetic AIF, zero noise — the non-dispersed corner.

    These exact-data fixtures have no interior L-curve curvature maximum.
    The L-curve rule selects its upper-endpoint default, strongly suppressing the peak.
    """

    @P2_NO_CORNER
    @pytest.mark.parametrize("tissue_key", SVD_TISSUES)
    def test_tikhonov_zero_noise(self, tissue_key: str) -> None:
        """Test Tikhonov recovery at zero noise with synthetic AIF."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_exponential_data(
            tissue_key, "zero"
        )

        result = tikhonov_deconvolve(tissue, aif, dt)
        assert result.selection_status == TikhonovSelection.UPPER_ENDPOINT
        assert np.all(np.isfinite([result.cbf, result.mtt])) and result.cbf > 0 and result.mtt > 0
        assert_accuracy(
            [abs(result.cbf - cbf) / cbf, abs(result.mtt - mtt) / mtt],
            [0.35, 0.58], label="Tikhonov CBF, residue-MTT",
        )


# =============================================================================
# sSVD with gamma transit-time survival residue
# =============================================================================

SSVD_GAMMA_RESIDUE_NOISE = [
    pytest.param("zero", 0.001, 0.20, 0.30, id="zero"),
    pytest.param("low", 0.02, 0.50, 2.00, id="low"),
    pytest.param("medium", 0.05, 0.60, 2.50, id="medium"),
]


class TestSSVDGammaResidue:
    """sSVD recovery with a synthetic AIF and gamma transit-time survival residue.

    MTT tolerances ~2x exponential because SVD bias increases with
    non-exponential residue shapes.
    """

    @pytest.mark.parametrize("noise_key,threshold,cbf_tol,mtt_tol", SSVD_GAMMA_RESIDUE_NOISE)
    @pytest.mark.parametrize("tissue_key,shape_key", GAMMA_RESIDUE_COMBOS)
    def test_ssvd_gamma_residue(
        self,
        tissue_key: str,
        shape_key: str,
        noise_key: str,
        threshold: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test sSVD recovery with a gamma transit-time survival residue."""
        tissue, aif, dt, time_sec, cbf, mtt, alpha = _make_gamma_residue_data(
            tissue_key, shape_key, noise_key
        )

        result = svd_deconvolve_standard(tissue, aif, dt, threshold=threshold)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# oSVD with gamma transit-time survival residue
# =============================================================================

OSVD_GAMMA_RESIDUE_NOISY = [
    pytest.param("low", 0.01, 0.50, 2.00, id="low"),
    pytest.param("medium", 0.01, 0.60, 2.50, id="medium"),
]


class TestOSVDGammaResidue:
    """oSVD recovery with a synthetic AIF and gamma transit-time survival residue.

    Evaluated against frame-interpolated fine-grid reference curves. Full-period
    residue MTT combines the signed area and regularized peak. The fixed recovery
    bounds characterize these residue shapes and acquisition conditions.
    """

    @pytest.mark.parametrize("tissue_key,shape_key", GAMMA_RESIDUE_COMBOS)
    def test_osvd_gamma_residue_zero_noise(
        self, tissue_key: str, shape_key: str
    ) -> None:
        """oSVD produces positive CBF and reasonable MTT at zero noise."""
        tissue, aif, dt, time_sec, cbf, mtt, alpha = _make_gamma_residue_data(
            tissue_key, shape_key, "zero"
        )
        # Production default threshold_min (0.01); see test_osvd_zero_noise.
        result = svd_deconvolve_oscillation(tissue, aif, dt)
        _assert_param_recovery(result.cbf, cbf, 0.25, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, 0.55, param_name="MTT")

    @pytest.mark.parametrize("noise_key,thresh_min,cbf_tol,mtt_tol", OSVD_GAMMA_RESIDUE_NOISY)
    @pytest.mark.parametrize("tissue_key,shape_key", GAMMA_RESIDUE_COMBOS)
    def test_osvd_gamma_residue_noisy(
        self,
        tissue_key: str,
        shape_key: str,
        noise_key: str,
        thresh_min: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery with a gamma transit-time survival residue and noise."""
        tissue, aif, dt, time_sec, cbf, mtt, alpha = _make_gamma_residue_data(
            tissue_key, shape_key, noise_key
        )

        result = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=thresh_min
        )

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# Tikhonov with gamma transit-time survival residue, zero noise only
# =============================================================================


class TestTikhonovGammaResidue:
    """Tikhonov recovery with a gamma transit-time survival residue, zero noise only.

    These exact-data fixtures select the upper lambda endpoint without an
    interior corner.
    """

    @P2_NO_CORNER
    @pytest.mark.parametrize("tissue_key,shape_key", GAMMA_RESIDUE_COMBOS)
    def test_tikhonov_gamma_residue_zero_noise(
        self, tissue_key: str, shape_key: str
    ) -> None:
        """Test Tikhonov recovery with a gamma transit-time survival residue at zero noise."""
        tissue, aif, dt, time_sec, cbf, mtt, alpha = _make_gamma_residue_data(
            tissue_key, shape_key, "zero"
        )

        result = tikhonov_deconvolve(tissue, aif, dt)
        assert result.selection_status == TikhonovSelection.UPPER_ENDPOINT
        assert np.all(np.isfinite([result.cbf, result.mtt])) and result.cbf > 0 and result.mtt > 0
        assert_accuracy(
            [abs(result.cbf - cbf) / cbf, abs(result.mtt - mtt) / mtt],
            [0.30, 0.50], label="Tikhonov CBF, residue-MTT",
        )


# =============================================================================
# Cross-method comparison
# =============================================================================


class TestCrossMethodComparison:
    """Compare deconvolution methods on identical synthetic data.

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
        tissue, aif, dt, time_sec, cbf, mtt = _make_exponential_data(
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

    @P2_NO_CORNER
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
        """Evaluate the 20% CBF agreement target for causal Volterra methods."""
        tissue, aif, dt, time_sec, cbf, mtt = _make_exponential_data(
            tissue_key, "zero"
        )

        r_ssvd = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        r_tikh = tikhonov_deconvolve(tissue, aif, dt)
        assert r_tikh.selection_status == TikhonovSelection.UPPER_ENDPOINT

        cbf_values = np.array([r_ssvd.cbf, r_tikh.cbf])
        assert np.all(np.isfinite(cbf_values)) and np.all(cbf_values > 0)
        mean_cbf = float(np.mean(cbf_values))
        assert_accuracy(abs(cbf_values - mean_cbf) / mean_cbf, 0.20, label="sSVD/Tikhonov CBF")
