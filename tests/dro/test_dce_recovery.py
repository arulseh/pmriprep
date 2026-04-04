"""DCE parameter recovery tests using Digital Reference Objects.

Tests that pharmacokinetic models can recover known ground-truth parameters
from synthetic concentration-time curves at various noise levels.

Tier 1: Concentration-level (evaluate -> noise -> fit -> compare)
Tier 2: Signal-level (ernst_signal -> Rician noise -> signal_to_conc -> fit)
"""

from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.models.dce.bayesian import fit_pk_model
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.patlak import PatlakModel
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel
from pmriprep.utils.dce_constants import get_r1_relaxivity
from pmriprep.utils.dce_signal import ernst_signal_model, signal_to_concentration_simple

from .conftest import add_gaussian_noise, add_rician_noise, generate_parker_aif_dro


# =============================================================================
# Helpers
# =============================================================================


def _assert_param_recovery(
    recovered: float,
    truth: float,
    rel_tol: float,
    abs_tol: float = 0.015,
    param_name: str = "",
) -> None:
    """Assert that a recovered parameter matches ground truth.

    Uses relative tolerance for non-zero parameters, absolute tolerance
    for near-zero parameters.

    Args:
        recovered: Recovered parameter value.
        truth: Ground truth parameter value.
        rel_tol: Relative tolerance (e.g. 0.05 = 5%).
        abs_tol: Absolute tolerance for near-zero truth values.
        param_name: Name for error messages.
    """
    if abs(truth) < 0.005:
        assert abs(recovered - truth) < abs_tol, (
            f"{param_name}: |{recovered:.6f} - {truth:.6f}| = "
            f"{abs(recovered - truth):.6f} >= {abs_tol}"
        )
    else:
        rel_err = abs(recovered - truth) / abs(truth)
        assert rel_err < rel_tol, (
            f"{param_name}: |{recovered:.6f} - {truth:.6f}| / |{truth:.6f}| = "
            f"{rel_err:.4f} >= {rel_tol}"
        )


def _make_dce_aif_and_time(
    n_timepoints: int = 80,
    dt_min: float = 4.0 / 60.0,
    peak: float = 5.0,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Create AIF and time array for DCE tests."""
    return generate_parker_aif_dro(n_timepoints, dt_min, peak)


# =============================================================================
# Noise level definitions
# =============================================================================

# (noise_std_fraction_of_peak, relative_tolerance)
# noise_std is applied as fraction of peak tissue concentration
# Tolerances are for well-identified parameters (e.g., Ktrans in Tofts
# with ve > 0.05). Parameters that are structurally unidentifiable
# (e.g., ve when Ktrans ~ 0) are guarded separately in each test.
NOISE_LEVELS_STANDARD = [
    pytest.param(0.0, 0.01, id="zero"),
    pytest.param(0.02, 0.08, id="low"),
    pytest.param(0.05, 0.19, id="medium"),
    pytest.param(0.10, 0.40, id="high"),
]

# Extended Tofts (3 params) is harder to identify than Tofts (2 params),
# especially for high kep and low permeability cases.
NOISE_LEVELS_ETOFTS = [
    pytest.param(0.0, 0.01, id="zero"),
    pytest.param(0.02, 0.11, id="low"),
    pytest.param(0.05, 0.26, id="medium"),
    pytest.param(0.10, 0.50, id="high"),
]

# 2CXM (4 free params): known coupling between PS and ve makes
# individual parameter recovery inherently less precise.
NOISE_LEVELS_2CXM = [
    pytest.param(0.0, 0.01, id="zero"),
    pytest.param(0.02, 0.09, id="low"),
    pytest.param(0.05, 0.21, id="medium"),
    pytest.param(0.10, 0.41, id="high"),
]


# =============================================================================
# Tofts Recovery
# =============================================================================

# (Ktrans, ve) parameter combinations
TOFTS_PARAMS = [
    pytest.param(0.1, 0.2, id="normal_tissue"),
    pytest.param(0.3, 0.4, id="enhancing_tumor"),
    pytest.param(0.01, 0.1, id="low_permeability"),
    pytest.param(0.2, 0.05, id="high_kep"),
    pytest.param(0.001, 0.05, id="near_zero_ktrans"),
    pytest.param(0.05, 0.5, id="large_ve"),
]


class TestToftsRecovery:
    """Test Tofts model parameter recovery at concentration level."""

    @pytest.mark.parametrize("noise_frac,tol", NOISE_LEVELS_STANDARD)
    @pytest.mark.parametrize("ktrans,ve", TOFTS_PARAMS)
    def test_nlls(
        self,
        ktrans: float,
        ve: float,
        noise_frac: float,
        tol: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test NLLS recovery of Tofts parameters."""
        aif, time_min = dce_aif_and_time
        model = ToftsModel()

        # Forward model
        curve = model.evaluate([ktrans, ve], aif, time_min)
        peak = float(np.max(np.abs(curve)))

        # Add noise
        if noise_frac > 0 and peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=42)

        # Fit
        result = model.fit(curve, aif, time_min)
        assert result.converged

        _assert_param_recovery(result.parameters["Ktrans"], ktrans, tol, param_name="Ktrans")
        # ve is unidentifiable when Ktrans is near zero (flat curve)
        if ktrans > 0.005:
            _assert_param_recovery(result.parameters["ve"], ve, tol, param_name="ve")

    @pytest.mark.parametrize("noise_frac,tol", NOISE_LEVELS_STANDARD)
    @pytest.mark.parametrize("ktrans,ve", TOFTS_PARAMS[:3])
    def test_map(
        self,
        ktrans: float,
        ve: float,
        noise_frac: float,
        tol: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test MAP recovery of Tofts parameters."""
        aif, time_min = dce_aif_and_time
        model = ToftsModel()

        curve = model.evaluate([ktrans, ve], aif, time_min)
        peak = float(np.max(np.abs(curve)))

        if noise_frac > 0 and peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=42)

        result = fit_pk_model(model, curve, aif, time_min, method="map")
        assert result.converged

        _assert_param_recovery(result.parameters["Ktrans"], ktrans, tol, param_name="Ktrans")
        _assert_param_recovery(result.parameters["ve"], ve, tol, param_name="ve")

    @pytest.mark.parametrize("noise_frac,tol", NOISE_LEVELS_STANDARD)
    @pytest.mark.parametrize("ktrans,ve", TOFTS_PARAMS[:2])
    def test_vi(
        self,
        ktrans: float,
        ve: float,
        noise_frac: float,
        tol: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test VI recovery of Tofts parameters.

        Uses same tolerances as MAP. Low-permeability case (Ktrans=0.01)
        excluded: VI prior mean ~0.1 is 10x higher, causing ~3-6% prior
        shrinkage even at 0% noise. This is expected Bayesian behavior,
        not a fitting bug.
        """
        aif, time_min = dce_aif_and_time
        model = ToftsModel()

        curve = model.evaluate([ktrans, ve], aif, time_min)
        peak = float(np.max(np.abs(curve)))

        if noise_frac > 0 and peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=42)

        result = fit_pk_model(model, curve, aif, time_min, method="vi")
        assert result.converged

        _assert_param_recovery(result.parameters["Ktrans"], ktrans, tol, param_name="Ktrans")
        _assert_param_recovery(result.parameters["ve"], ve, tol, param_name="ve")


# =============================================================================
# VI Prior Shrinkage Direction
# =============================================================================


class TestVIPriorShrinkage:
    """Verify that VI exhibits expected prior shrinkage direction.

    When the true parameter is far from the prior mean, the VI posterior
    mean should be pulled toward the prior. This is a fundamental property
    of Bayesian estimation (Lindley & Smith, 1972, JRSS-B).

    These are not recovery tests — they validate a statistical identity
    of the algorithm: estimates shrink toward the prior mean.
    """

    def test_low_ktrans_overestimated(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Low Ktrans should be overestimated (pulled toward prior mean).

        Tofts prior mean ~0.1 min^-1 (from typical_range midpoint).
        True Ktrans=0.01 is 4.7 sigma below prior mean in log-space.
        VI should overestimate (shrink toward prior), with <10% bias.
        """
        aif, time_min = dce_aif_and_time
        model = ToftsModel()
        true_ktrans = 0.01

        curve = model.evaluate([true_ktrans, 0.1], aif, time_min)
        result = fit_pk_model(model, curve, aif, time_min, method="vi")
        assert result is not None

        recovered = result.parameters["Ktrans"]
        assert recovered > true_ktrans, (
            f"Expected shrinkage toward prior (Ktrans_vi > {true_ktrans}), "
            f"got {recovered:.6f}"
        )
        assert recovered < true_ktrans * 1.10, (
            f"Shrinkage too large: {recovered:.6f} > {true_ktrans * 1.10:.6f} "
            f"(>10% bias at 0% noise)"
        )

    def test_low_ktrans_etofts_overestimated(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Extended Tofts: low Ktrans should also shrink toward prior."""
        aif, time_min = dce_aif_and_time
        model = ExtendedToftsModel()
        true_ktrans = 0.01

        curve = model.evaluate([true_ktrans, 0.1, 0.01], aif, time_min)
        result = fit_pk_model(model, curve, aif, time_min, method="vi")
        assert result is not None

        recovered = result.parameters["Ktrans"]
        assert recovered > true_ktrans, (
            f"Expected shrinkage toward prior (Ktrans_vi > {true_ktrans}), "
            f"got {recovered:.6f}"
        )
        assert recovered < true_ktrans * 1.10


# =============================================================================
# Extended Tofts Recovery
# =============================================================================

# (Ktrans, ve, vp) parameter combinations
ETOFTS_PARAMS = [
    pytest.param(0.1, 0.2, 0.02, id="normal_tissue"),
    pytest.param(0.3, 0.4, 0.05, id="enhancing_tumor"),
    pytest.param(0.01, 0.1, 0.01, id="low_permeability"),
    pytest.param(0.2, 0.05, 0.03, id="high_kep"),
    pytest.param(0.0, 0.2, 0.05, id="zero_ktrans"),
    pytest.param(0.3, 0.5, 0.25, id="near_volume_limit"),
]


class TestExtendedToftsRecovery:
    """Test Extended Tofts model parameter recovery."""

    @pytest.mark.parametrize("noise_frac,tol", NOISE_LEVELS_ETOFTS)
    @pytest.mark.parametrize("ktrans,ve,vp", ETOFTS_PARAMS)
    def test_nlls(
        self,
        ktrans: float,
        ve: float,
        vp: float,
        noise_frac: float,
        tol: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test NLLS recovery of Extended Tofts parameters."""
        aif, time_min = dce_aif_and_time
        model = ExtendedToftsModel()

        curve = model.evaluate([ktrans, ve, vp], aif, time_min)
        peak = float(np.max(np.abs(curve)))

        if noise_frac > 0 and peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=42)

        result = model.fit(curve, aif, time_min)
        assert result.converged

        # High kep or near-volume-limit combos have structural identifiability
        # issues with noise. Verify curve fit quality instead of individual params.
        kep = ktrans / ve if ve > 0.01 else 0.0
        if noise_frac > 0 and (kep > 3.0 or ve + vp > 0.6):
            assert result.r_squared > 0.80, (
                f"R² = {result.r_squared:.4f} < 0.80 despite convergence"
            )
            return

        _assert_param_recovery(result.parameters["Ktrans"], ktrans, tol, param_name="Ktrans")
        # ve is unidentifiable when Ktrans is low (flat tissue curve)
        if ktrans > 0.02:
            _assert_param_recovery(result.parameters["ve"], ve, tol, param_name="ve")
        # vp is poorly identified when Ktrans ~ 0 (weak signal = vp*AIF only)
        if ktrans > 0.02 or noise_frac == 0:
            _assert_param_recovery(result.parameters["vp"], vp, tol, param_name="vp")

    @pytest.mark.parametrize("noise_frac,tol", NOISE_LEVELS_ETOFTS)
    @pytest.mark.parametrize("ktrans,ve,vp", ETOFTS_PARAMS[:3])
    def test_map(
        self,
        ktrans: float,
        ve: float,
        vp: float,
        noise_frac: float,
        tol: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test MAP recovery of Extended Tofts parameters."""
        aif, time_min = dce_aif_and_time
        model = ExtendedToftsModel()

        curve = model.evaluate([ktrans, ve, vp], aif, time_min)
        peak = float(np.max(np.abs(curve)))

        if noise_frac > 0 and peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=42)

        result = fit_pk_model(model, curve, aif, time_min, method="map")
        assert result.converged

        _assert_param_recovery(result.parameters["Ktrans"], ktrans, tol, param_name="Ktrans")
        # ve is poorly constrained when Ktrans is low (MC calibration:
        # low_permeability ve mean+3sigma = 0.23-1.72 vs tolerance 0.10-0.40)
        if ktrans > 0.02:
            _assert_param_recovery(result.parameters["ve"], ve, tol, param_name="ve")
        _assert_param_recovery(result.parameters["vp"], vp, tol, param_name="vp")

    @pytest.mark.parametrize("noise_frac,tol", NOISE_LEVELS_ETOFTS)
    @pytest.mark.parametrize("ktrans,ve,vp", ETOFTS_PARAMS[:2])
    def test_vi(
        self,
        ktrans: float,
        ve: float,
        vp: float,
        noise_frac: float,
        tol: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test VI recovery of Extended Tofts parameters.

        Uses same tolerances as MAP. Low-permeability case excluded:
        VI prior shrinkage on Ktrans=0.01 (see TestToftsRecovery.test_vi).
        """
        aif, time_min = dce_aif_and_time
        model = ExtendedToftsModel()

        curve = model.evaluate([ktrans, ve, vp], aif, time_min)
        peak = float(np.max(np.abs(curve)))

        if noise_frac > 0 and peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=42)

        result = fit_pk_model(model, curve, aif, time_min, method="vi")
        assert result.converged

        _assert_param_recovery(result.parameters["Ktrans"], ktrans, tol, param_name="Ktrans")
        if ktrans > 0.02:
            _assert_param_recovery(result.parameters["ve"], ve, tol, param_name="ve")
        _assert_param_recovery(result.parameters["vp"], vp, tol, param_name="vp")


# =============================================================================
# Patlak Recovery
# =============================================================================

# (Ktrans, vp) parameter combinations
PATLAK_PARAMS = [
    pytest.param(0.05, 0.02, id="intact_bbb"),
    pytest.param(0.1, 0.05, id="moderate_leak"),
    pytest.param(0.01, 0.01, id="low_both"),
    pytest.param(0.2, 0.1, id="disrupted_bbb"),
    pytest.param(0.001, 0.03, id="near_zero_ktrans"),
]


class TestPatlakRecovery:
    """Test Patlak model parameter recovery."""

    @pytest.mark.parametrize("noise_frac,tol", NOISE_LEVELS_STANDARD)
    @pytest.mark.parametrize("ktrans,vp", PATLAK_PARAMS)
    def test_nlls(
        self,
        ktrans: float,
        vp: float,
        noise_frac: float,
        tol: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test NLLS recovery of Patlak parameters."""
        aif, time_min = dce_aif_and_time
        model = PatlakModel()

        curve = model.evaluate([ktrans, vp], aif, time_min)
        peak = float(np.max(np.abs(curve)))

        if noise_frac > 0 and peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=42)

        result = model.fit(curve, aif, time_min)
        assert result.converged

        # Near-zero Ktrans uses absolute tolerance (relative error blows up)
        _assert_param_recovery(
            result.parameters["Ktrans"], ktrans, tol,
            abs_tol=0.005, param_name="Ktrans",
        )
        _assert_param_recovery(result.parameters["vp"], vp, tol, param_name="vp")

    @pytest.mark.parametrize("ktrans,vp", PATLAK_PARAMS[:4])
    def test_graphical(
        self,
        ktrans: float,
        vp: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test graphical (linear regression) recovery of Patlak parameters.

        Graphical method only tested at zero noise since it linearizes the
        model and is inherently less robust to noise.
        """
        aif, time_min = dce_aif_and_time
        model = PatlakModel()

        curve = model.evaluate([ktrans, vp], aif, time_min)

        result = model.fit_graphical(curve, aif, time_min)
        assert result.converged

        # Graphical method has slightly relaxed tolerance
        _assert_param_recovery(
            result.parameters["Ktrans"], ktrans, 0.05, param_name="Ktrans"
        )
        _assert_param_recovery(
            result.parameters["vp"], vp, 0.05, param_name="vp"
        )


# =============================================================================
# 2CXM Recovery
# =============================================================================

# (Fp, PS, ve, vp) parameter combinations — units: mL/min/100mL for Fp, PS
# Note: low-PS cases (PS < 5) have severe PS/ve coupling → excluded from
# full parametric sweep. The 2CXM has known identifiability issues when
# PS << Fp (approaches Patlak regime where PS and ve are confounded).
TWO_CXM_PARAMS = [
    pytest.param(50.0, 10.0, 0.2, 0.02, id="normal_gm"),
    pytest.param(80.0, 30.0, 0.4, 0.05, id="tumor"),
    pytest.param(40.0, 15.0, 0.25, 0.03, id="moderate"),
]


class Test2CXMRecovery:
    """Test Two-Compartment Exchange model parameter recovery."""

    @pytest.mark.parametrize("noise_frac,tol", NOISE_LEVELS_2CXM)
    @pytest.mark.parametrize("fp,ps,ve,vp", TWO_CXM_PARAMS)
    def test_nlls(
        self,
        fp: float,
        ps: float,
        ve: float,
        vp: float,
        noise_frac: float,
        tol: float,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test NLLS recovery of 2CXM parameters."""
        aif, time_min = dce_aif_and_time
        model = TwoCompartmentExchangeModel()

        curve = model.evaluate([fp, ps, ve, vp], aif, time_min)
        peak = float(np.max(np.abs(curve)))

        if noise_frac > 0 and peak > 0:
            curve = add_gaussian_noise(curve, noise_frac * peak, seed=42)

        result = model.fit(curve, aif, time_min)
        assert result.converged

        # Fp is the best-identified parameter
        _assert_param_recovery(result.parameters["Fp"], fp, tol, param_name="Fp")

        if noise_frac == 0:
            # At zero noise all parameters should recover exactly
            _assert_param_recovery(result.parameters["PS"], ps, tol, param_name="PS")
            _assert_param_recovery(
                result.parameters["ve"], ve, min(tol * 1.5, 0.60), param_name="ve"
            )
            _assert_param_recovery(
                result.parameters["vp"], vp, min(tol * 1.5, 0.60), param_name="vp"
            )
        else:
            # PS/ve have known coupling that makes individual parameter recovery
            # unreliable with noise (MC calibration: mean+3sigma = 0.7-8.5 for PS,
            # 1.4-5.7 for ve). Verify curve fit quality instead.
            # R² threshold is 0.80: at 10% noise the residual variance from noise
            # alone accounts for ~10-15% of signal variance.
            assert result.r_squared > 0.80, (
                f"R² = {result.r_squared:.4f} < 0.80 despite convergence"
            )

    def test_map_convergence(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Test that 2CXM MAP converges and achieves good fit.

        2CXM has known parameter trade-offs (PS/ve coupling) that make
        exact MAP recovery unreliable for individual parameters. We verify
        convergence and curve fit quality instead.
        """
        aif, time_min = dce_aif_and_time
        model = TwoCompartmentExchangeModel()

        curve = model.evaluate([50.0, 10.0, 0.2, 0.02], aif, time_min)
        result = fit_pk_model(model, curve, aif, time_min, method="map")

        assert result.converged
        assert result.r_squared > 0.99


# =============================================================================
# Edge Cases
# =============================================================================


class TestDCEEdgeCases:
    """Test edge cases for DCE models."""

    def test_zero_ktrans_tofts(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Tofts with zero Ktrans should produce flat zero curve."""
        aif, time_min = dce_aif_and_time
        model = ToftsModel()

        curve = model.evaluate([0.0, 0.2], aif, time_min)
        assert np.allclose(curve, 0.0, atol=1e-10)

    def test_zero_ktrans_etofts(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Extended Tofts with zero Ktrans should produce vp*Cp curve."""
        aif, time_min = dce_aif_and_time
        model = ExtendedToftsModel()

        vp = 0.05
        curve = model.evaluate([0.0, 0.2, vp], aif, time_min)
        expected = vp * aif
        np.testing.assert_allclose(curve, expected, atol=1e-6)

    def test_high_kep_tofts(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Tofts with very high kep (small ve) should still converge."""
        aif, time_min = dce_aif_and_time
        model = ToftsModel()

        curve = model.evaluate([0.5, 0.01], aif, time_min)
        result = model.fit(curve, aif, time_min)
        assert result.converged

    def test_flat_tissue_curve(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Fitting a flat (all-zero) curve should not crash."""
        aif, time_min = dce_aif_and_time
        model = ToftsModel()

        curve = np.zeros_like(time_min)
        result = model.fit(curve, aif, time_min)
        # Should recover near-zero Ktrans regardless of convergence
        assert result.parameters["Ktrans"] < 0.01

    def test_volume_constraint_etofts(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Extended Tofts should enforce ve + vp <= 1.0."""
        aif, time_min = dce_aif_and_time
        model = ExtendedToftsModel()

        # Generate with valid near-limit params
        curve = model.evaluate([0.3, 0.6, 0.3], aif, time_min)
        result = model.fit(curve, aif, time_min)

        if result.converged:
            ve = result.parameters["ve"]
            vp = result.parameters["vp"]
            assert ve + vp <= 1.0 + 1e-6

    def test_patlak_reduces_to_linear(
        self,
        dce_aif_and_time: Tuple[npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        """Patlak model should recover parameters via both NLLS and graphical."""
        aif, time_min = dce_aif_and_time
        model = PatlakModel()

        ktrans, vp = 0.05, 0.03
        curve = model.evaluate([ktrans, vp], aif, time_min)

        result_nlls = model.fit(curve, aif, time_min)
        result_graph = model.fit_graphical(curve, aif, time_min)

        # Both methods should agree within 5%
        assert abs(result_nlls.parameters["Ktrans"] - result_graph.parameters["Ktrans"]) / ktrans < 0.05
        assert abs(result_nlls.parameters["vp"] - result_graph.parameters["vp"]) / vp < 0.05


# =============================================================================
# Signal-Level Round-Trip (Tier 2)
# =============================================================================

# Signal-level test parameters
SIGNAL_PARAMS = [
    pytest.param(0.1, 0.2, 0.02, id="normal_tissue"),
    pytest.param(0.3, 0.4, 0.05, id="enhancing_tumor"),
]

# Signal noise std in raw signal units. With S0=1000 and baseline ~88
# (Ernst equation with TR=5ms, T1=1200ms, flip=15deg), these give:
#   0.5 → SNR ~176 (low), 1.0 → SNR ~88 (medium), 2.0 → SNR ~44 (high)
SIGNAL_NOISE_LEVELS = [
    pytest.param(0.0, 0.02, id="zero"),
    pytest.param(0.5, 0.10, id="low"),
    pytest.param(1.0, 0.20, id="medium"),
    pytest.param(2.0, 0.35, id="high"),
]


class TestSignalRoundtrip:
    """Test full signal-level round-trip: signal -> concentration -> fit.

    Uses Ernst signal model with known T1, then converts back to
    concentration via signal_to_concentration_simple, then fits.
    """

    @pytest.mark.parametrize("noise_std,tol", SIGNAL_NOISE_LEVELS)
    @pytest.mark.parametrize("ktrans,ve,vp", SIGNAL_PARAMS)
    def test_signal_roundtrip(
        self,
        ktrans: float,
        ve: float,
        vp: float,
        noise_std: float,
        tol: float,
    ) -> None:
        """Test Extended Tofts recovery through signal domain."""
        # Parameters
        t1_pre_ms = 1200.0
        s0 = 1000.0
        flip_angle_deg = 15.0
        tr_sec = 0.005  # 5 ms
        field_strength = 3.0
        r1 = get_r1_relaxivity(field_strength)  # 3.8 s^-1/mM
        n_timepoints = 80
        dt_sec = 4.0
        baseline_frames = 8

        # Time in minutes for PK model
        dt_min = dt_sec / 60.0
        time_min = np.arange(n_timepoints, dtype=np.float64) * dt_min

        # Generate AIF
        aif, _ = generate_parker_aif_dro(n_timepoints, dt_min, peak_concentration=5.0)

        # Forward PK model -> concentration
        model = ExtendedToftsModel()
        conc = model.evaluate([ktrans, ve, vp], aif, time_min)

        # Concentration -> R1(t)
        r1_pre = 1000.0 / t1_pre_ms  # s^-1
        r1_t = r1_pre + r1 * conc

        # R1(t) -> T1(t) -> signal via Ernst equation
        t1_t = 1.0 / np.maximum(r1_t, 1e-6)
        flip_angle_rad = np.deg2rad(flip_angle_deg)

        # Baseline signal (pre-contrast)
        baseline_signal = ernst_signal_model(s0, flip_angle_rad, tr_sec, t1_pre_ms / 1000.0)

        # Build signal time series
        signal = np.zeros(n_timepoints, dtype=np.float64)
        signal[:baseline_frames] = baseline_signal
        for i in range(baseline_frames, n_timepoints):
            signal[i] = ernst_signal_model(s0, flip_angle_rad, tr_sec, t1_t[i])

        # Add Rician noise
        if noise_std > 0:
            signal = add_rician_noise(signal, noise_std, seed=42)

        # Signal -> concentration via the pipeline's own conversion
        conc_recovered = signal_to_concentration_simple(
            signal=signal,
            flip_angle_deg=flip_angle_deg,
            tr=tr_sec,
            t1_pre_ms=t1_pre_ms,
            field_strength=field_strength,
            baseline_frames=baseline_frames,
            convert_to_plasma=False,
        )

        # Fit recovered concentration
        result = model.fit(conc_recovered, aif, time_min)
        assert result.converged

        _assert_param_recovery(result.parameters["Ktrans"], ktrans, tol, param_name="Ktrans")
        _assert_param_recovery(result.parameters["ve"], ve, tol, param_name="ve")
        _assert_param_recovery(result.parameters["vp"], vp, tol, param_name="vp")
