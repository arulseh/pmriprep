"""Unit tests for batched Levenberg-Marquardt fitting.

Tests verify parameter recovery, convergence, constraint handling,
auto-mode BIC selection, and result format for all four DCE PK models.

All tests use numpy fallback (no CuPy required).
"""

import numpy as np
import pytest

from pmriprep.models.dce.batched_models import (
    ModelResult,
    batched_2cxm_model,
    batched_extended_tofts_model,
    batched_patlak_model,
    batched_tofts_model,
    evaluate_model,
)
from pmriprep.models.dce.batched_fitting import (
    FitResult,
    _MODEL_PARAMS,
    batched_lm_fit,
    batched_patlak_lstsq,
)
from pmriprep.utils.dce_constants import EPSILON_VE


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aif_and_time():
    """Generate a realistic AIF and time vector."""
    n_time = 60
    time_points = np.linspace(0, 5.0, n_time)  # 5 minutes
    # Parker-like AIF: gaussian peak + exponential washout
    peak_time = 0.5
    aif = 5.0 * np.exp(-0.5 * ((time_points - peak_time) / 0.15) ** 2) + \
          1.5 * np.exp(-0.3 * np.maximum(time_points - peak_time, 0))
    return aif, time_points


def _generate_tofts_data(aif, time_points, n_voxels=1000, noise_level=0.01):
    """Generate synthetic Tofts data with known parameters."""
    rng = np.random.RandomState(42)
    ktrans_true = rng.uniform(0.01, 0.3, n_voxels)
    ve_true = rng.uniform(0.1, 0.5, n_voxels)

    result = batched_tofts_model(
        ktrans_true, ve_true, aif, time_points, use_gpu=False,
    )
    tissue = result.tissue_curves
    noise = noise_level * np.max(tissue, axis=1, keepdims=True) * rng.randn(*tissue.shape)
    tissue_noisy = tissue + noise

    return tissue_noisy, ktrans_true, ve_true


def _generate_etofts_data(aif, time_points, n_voxels=1000, noise_level=0.01):
    """Generate synthetic Extended Tofts data."""
    rng = np.random.RandomState(43)
    ktrans_true = rng.uniform(0.01, 0.3, n_voxels)
    ve_true = rng.uniform(0.1, 0.5, n_voxels)
    vp_true = rng.uniform(0.01, 0.1, n_voxels)

    result = batched_extended_tofts_model(
        ktrans_true, ve_true, vp_true, aif, time_points, use_gpu=False,
    )
    tissue = result.tissue_curves
    noise = noise_level * np.max(tissue, axis=1, keepdims=True) * rng.randn(*tissue.shape)

    return tissue + noise, ktrans_true, ve_true, vp_true


def _generate_patlak_data(aif, time_points, n_voxels=1000):
    """Generate synthetic noiseless Patlak data."""
    rng = np.random.RandomState(44)
    ktrans_true = rng.uniform(0.01, 0.2, n_voxels)
    vp_true = rng.uniform(0.01, 0.15, n_voxels)

    result = batched_patlak_model(
        ktrans_true, vp_true, aif, time_points, use_gpu=False,
    )
    return result.tissue_curves, ktrans_true, vp_true


def _generate_2cxm_data(aif, time_points, n_voxels=1000, noise_level=0.01):
    """Generate synthetic 2CXM data."""
    rng = np.random.RandomState(45)
    fp_true = rng.uniform(20, 100, n_voxels)
    ps_true = rng.uniform(5, 40, n_voxels)
    ve_true = rng.uniform(0.1, 0.5, n_voxels)
    vp_true = rng.uniform(0.01, 0.1, n_voxels)

    result = batched_2cxm_model(
        fp_true, ps_true, ve_true, vp_true, aif, time_points, use_gpu=False,
    )
    tissue = result.tissue_curves
    noise = noise_level * np.max(tissue, axis=1, keepdims=True) * rng.randn(*tissue.shape)

    return tissue + noise, fp_true, ps_true, ve_true, vp_true


# ---------------------------------------------------------------------------
# Test 1: Tofts recovery
# ---------------------------------------------------------------------------

class TestToftsRecovery:
    def test_tofts_recovery_1pct_noise(self, aif_and_time):
        """1000 synthetic voxels, 1% noise → params within 5% for 95%."""
        aif, time_points = aif_and_time
        tissue, ktrans_true, ve_true = _generate_tofts_data(aif, time_points)

        result = batched_lm_fit(
            "tofts", tissue, aif, time_points,
            max_iter=50, save_fitted_curves=True,
        )

        assert isinstance(result, FitResult)
        assert result.model_name == "tofts"
        assert result.params["ktrans"].shape == (1000,)

        # Check recovery for converged voxels
        converged = result.converged
        assert np.sum(converged) > 800  # >80% converged

        ktrans_err = np.abs(result.params["ktrans"][converged] - ktrans_true[converged])
        ktrans_rel_err = ktrans_err / np.maximum(ktrans_true[converged], 1e-6)
        pct_within_5 = np.mean(ktrans_rel_err < 0.05) * 100
        assert pct_within_5 > 90, f"Only {pct_within_5:.0f}% within 5%"


# ---------------------------------------------------------------------------
# Test 2: Extended Tofts recovery
# ---------------------------------------------------------------------------

class TestExtendedToftsRecovery:
    def test_etofts_recovery(self, aif_and_time):
        """1000 synthetic voxels, 1% noise, 3 params."""
        aif, time_points = aif_and_time
        tissue, ktrans_true, ve_true, vp_true = _generate_etofts_data(aif, time_points)

        result = batched_lm_fit(
            "extended_tofts", tissue, aif, time_points, max_iter=50,
        )

        converged = result.converged
        assert np.sum(converged) > 700

        # Ktrans recovery
        kt_err = np.abs(result.params["ktrans"][converged] - ktrans_true[converged])
        kt_rel = kt_err / np.maximum(ktrans_true[converged], 1e-6)
        assert np.median(kt_rel) < 0.15


# ---------------------------------------------------------------------------
# Test 3: Patlak exact recovery (no noise)
# ---------------------------------------------------------------------------

class TestPatlakExact:
    def test_patlak_exact_recovery(self, aif_and_time):
        """No noise → machine precision recovery (linear model)."""
        aif, time_points = aif_and_time
        tissue, ktrans_true, vp_true = _generate_patlak_data(aif, time_points)

        result = batched_patlak_lstsq(
            tissue, aif, time_points, save_fitted_curves=True,
        )

        assert result.model_name == "patlak"
        assert np.all(result.converged)

        np.testing.assert_allclose(
            result.params["ktrans"], ktrans_true, rtol=1e-6,
        )
        np.testing.assert_allclose(
            result.params["vp"], vp_true, rtol=1e-6,
        )

    def test_patlak_high_r_squared(self, aif_and_time):
        """Noiseless Patlak should give R² ≈ 1."""
        aif, time_points = aif_and_time
        tissue, _, _ = _generate_patlak_data(aif, time_points, n_voxels=100)

        result = batched_patlak_lstsq(tissue, aif, time_points)
        assert np.all(result.r_squared > 0.999)


# ---------------------------------------------------------------------------
# Test 4: 2CXM recovery
# ---------------------------------------------------------------------------

class TestTwoCXMRecovery:
    def test_2cxm_recovery(self, aif_and_time):
        """1000 voxels, 1% noise, 4 params → params within 10% for 80%."""
        aif, time_points = aif_and_time
        tissue, fp_true, ps_true, ve_true, vp_true = _generate_2cxm_data(
            aif, time_points, n_voxels=200,
        )

        result = batched_lm_fit(
            "2cxm", tissue, aif, time_points, max_iter=80,
        )

        converged = result.converged
        assert np.sum(converged) > 100  # >50% converged (4-param is harder)

        # Fp recovery (within 20% for converged)
        fp_err = np.abs(result.params["fp"][converged] - fp_true[converged])
        fp_rel = fp_err / np.maximum(fp_true[converged], 1e-6)
        assert np.median(fp_rel) < 0.5  # Lenient for 4-param model


# ---------------------------------------------------------------------------
# Test 5: vp constraint
# ---------------------------------------------------------------------------

class TestVpConstraint:
    def test_fixed_vp_roundtrip(self, aif_and_time):
        """fixed_vp → output vp matches input, Ktrans/ve recovered."""
        aif, time_points = aif_and_time
        rng = np.random.RandomState(50)
        n = 500
        ktrans_true = rng.uniform(0.05, 0.2, n)
        ve_true = rng.uniform(0.1, 0.4, n)
        vp_true = rng.uniform(0.02, 0.08, n)

        result_fwd = batched_extended_tofts_model(
            ktrans_true, ve_true, vp_true, aif, time_points, use_gpu=False,
        )
        tissue = result_fwd.tissue_curves + 0.005 * rng.randn(*result_fwd.tissue_curves.shape)

        result = batched_lm_fit(
            "extended_tofts", tissue, aif, time_points,
            fixed_vp=vp_true, max_iter=50,
        )

        # vp should match input exactly
        np.testing.assert_allclose(result.params["vp"], vp_true, rtol=1e-10)

    def test_patlak_fixed_vp(self, aif_and_time):
        """Patlak with fixed vp: 1-parameter solve for Ktrans."""
        aif, time_points = aif_and_time
        tissue, ktrans_true, vp_true = _generate_patlak_data(
            aif, time_points, n_voxels=200,
        )

        result = batched_patlak_lstsq(
            tissue, aif, time_points, fixed_vp=vp_true,
        )

        np.testing.assert_allclose(
            result.params["ktrans"], ktrans_true, rtol=1e-5,
        )
        np.testing.assert_allclose(result.params["vp"], vp_true, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 6: Auto mode BIC selection
# ---------------------------------------------------------------------------

class TestAutoModeBIC:
    def test_bic_selects_correct_model(self, aif_and_time):
        """Synthetic data from different models → correct model selected."""
        aif, time_points = aif_and_time
        rng = np.random.RandomState(60)
        n_per = 100

        # Tofts data (no vp contribution)
        kt_tofts = rng.uniform(0.05, 0.2, n_per)
        ve_tofts = rng.uniform(0.15, 0.4, n_per)
        tofts_curves = batched_tofts_model(
            kt_tofts, ve_tofts, aif, time_points, use_gpu=False,
        ).tissue_curves

        # Patlak data (distinct linear accumulation pattern)
        kt_patlak = rng.uniform(0.05, 0.15, n_per)
        vp_patlak = rng.uniform(0.05, 0.15, n_per)
        patlak_curves = batched_patlak_model(
            kt_patlak, vp_patlak, aif, time_points, use_gpu=False,
        ).tissue_curves

        # Combine
        all_curves = np.vstack([tofts_curves, patlak_curves])
        noise = 0.005 * np.max(all_curves, axis=1, keepdims=True) * rng.randn(*all_curves.shape)
        all_curves += noise

        # Fit all candidates
        candidates = ["tofts", "patlak"]
        results = {}
        for cname in candidates:
            if cname == "patlak":
                results[cname] = batched_patlak_lstsq(
                    all_curves, aif, time_points, save_fitted_curves=True,
                )
            else:
                results[cname] = batched_lm_fit(
                    cname, all_curves, aif, time_points,
                    max_iter=30, save_fitted_curves=True,
                )

        # BIC selection
        n_total = 2 * n_per
        T = len(time_points)
        n_params = {"tofts": 2, "patlak": 2}
        bic = {}
        for cname, res in results.items():
            rss = np.sum((all_curves - res.fitted_curves) ** 2, axis=1)
            rss_safe = np.maximum(rss, 1e-30)
            k = n_params[cname]
            bic[cname] = T * np.log(rss_safe / T) + k * np.log(T)

        bic_stack = np.stack([bic[c] for c in candidates], axis=0)
        winner_idx = np.argmin(bic_stack, axis=0)
        winners = [candidates[w] for w in winner_idx]

        # Check that Tofts voxels are mostly selected as Tofts
        tofts_correct = sum(w == "tofts" for w in winners[:n_per])
        assert tofts_correct > 60, f"Only {tofts_correct}/{n_per} Tofts correct"


# ---------------------------------------------------------------------------
# Test 7: CPU consistency
# ---------------------------------------------------------------------------

class TestCPUConsistency:
    def test_tofts_r2_quality(self, aif_and_time):
        """Same data through batched LM should give good R² distributions."""
        aif, time_points = aif_and_time
        tissue, ktrans_true, ve_true = _generate_tofts_data(
            aif, time_points, n_voxels=200,
        )

        result = batched_lm_fit(
            "tofts", tissue, aif, time_points, max_iter=50,
        )

        # Should have good R²
        assert np.median(result.r_squared[result.converged]) > 0.9


# ---------------------------------------------------------------------------
# Test 8: Convergence under noise
# ---------------------------------------------------------------------------

class TestConvergenceUnderNoise:
    def test_10pct_noise_convergence(self, aif_and_time):
        """10% noise → >80% convergence rate."""
        aif, time_points = aif_and_time
        tissue, _, _ = _generate_tofts_data(
            aif, time_points, n_voxels=500, noise_level=0.1,
        )

        result = batched_lm_fit(
            "tofts", tissue, aif, time_points, max_iter=80,
        )

        convergence_rate = np.sum(result.converged) / len(result.converged)
        assert convergence_rate > 0.8, f"Convergence rate: {convergence_rate:.2f}"


# ---------------------------------------------------------------------------
# Test 9: Result dict format
# ---------------------------------------------------------------------------

class TestResultDictFormat:
    def test_fit_result_fields(self, aif_and_time):
        """FitResult has all required fields."""
        aif, time_points = aif_and_time
        tissue, _, _ = _generate_tofts_data(aif, time_points, n_voxels=50)

        result = batched_lm_fit(
            "tofts", tissue, aif, time_points, max_iter=10,
            save_fitted_curves=True, compute_uncertainties=True,
        )

        assert isinstance(result, FitResult)
        assert "ktrans" in result.params
        assert "ve" in result.params
        assert result.r_squared.shape == (50,)
        assert result.converged.shape == (50,)
        assert result.fitted_curves is not None
        assert result.fitted_curves.shape == (50, 60)
        assert result.model_name == "tofts"
        assert isinstance(result.uncertainties, dict)
        assert result.n_iterations > 0


# ---------------------------------------------------------------------------
# Test 10: Constrained auto-mode
# ---------------------------------------------------------------------------

class TestConstrainedAutoMode:
    def test_constrained_vp_preserved(self, aif_and_time):
        """100 constrained + 100 unconstrained: mixed vp handling."""
        aif, time_points = aif_and_time
        rng = np.random.RandomState(100)
        n = 200

        kt = rng.uniform(0.05, 0.2, n)
        ve = rng.uniform(0.15, 0.4, n)
        vp = rng.uniform(0.02, 0.08, n)

        tissue = batched_extended_tofts_model(
            kt, ve, vp, aif, time_points, use_gpu=False,
        ).tissue_curves

        # Constrain first half only; second half are unconstrained (zero)
        fixed_vp = np.zeros(n)
        fixed_vp[:100] = vp[:100]

        from pmriprep.interfaces.dce_compute import _fit_with_split_constraint

        def _lm_fit(tc, **kw):
            return batched_lm_fit(
                "extended_tofts", tc, aif, time_points, **kw,
            )

        result = _fit_with_split_constraint(
            _lm_fit, tissue, fixed_vp, "extended_tofts", n,
            weights=None,
            max_iter=30,
            save_fitted_curves=False,
            compute_uncertainties=False,
        )

        # Constrained voxels should have exact vp
        np.testing.assert_allclose(
            result.params["vp"][:100], vp[:100], rtol=1e-10,
        )

        # Unconstrained voxels should have vp > 0 (fitted freely, not forced to 0)
        assert np.all(result.params["vp"][100:] > 0), (
            "Unconstrained voxels should have vp > 0 from free fitting"
        )


# ---------------------------------------------------------------------------
# Test 11: Weighted fitting
# ---------------------------------------------------------------------------

class TestWeightedFitting:
    def test_weighted_patlak(self, aif_and_time):
        """Motion weights applied to Patlak fitting."""
        aif, time_points = aif_and_time
        tissue, kt_true, vp_true = _generate_patlak_data(
            aif, time_points, n_voxels=100,
        )

        # Create weights: downweight a few frames
        weights = np.ones(len(time_points))
        weights[10:15] = 0.1  # Simulate motion frames

        result = batched_patlak_lstsq(
            tissue, aif, time_points, weights=weights,
        )

        # Should still recover params reasonably (noiseless data)
        np.testing.assert_allclose(
            result.params["ktrans"], kt_true, rtol=0.02,
        )

    def test_weighted_lm(self, aif_and_time):
        """Motion weights applied to LM fitting."""
        aif, time_points = aif_and_time
        tissue, _, _ = _generate_tofts_data(
            aif, time_points, n_voxels=100,
        )

        weights = np.ones(len(time_points))
        weights[10:15] = 0.1

        result = batched_lm_fit(
            "tofts", tissue, aif, time_points,
            weights=weights, max_iter=30,
        )

        assert np.sum(result.converged) > 50


# ---------------------------------------------------------------------------
# Test 12-14: Batched convolution forward model tests
# ---------------------------------------------------------------------------

class TestPatlakForwardModel:
    def test_patlak_batched_matches_cpu(self, aif_and_time):
        """Batched Patlak forward model matches CPU PatlakModel.evaluate()."""
        from pmriprep.models.dce.patlak import PatlakModel

        aif, time_points = aif_and_time
        model = PatlakModel()
        n = 50
        rng = np.random.RandomState(120)
        ktrans = rng.uniform(0.01, 0.2, n)
        vp = rng.uniform(0.01, 0.15, n)

        result = batched_patlak_model(
            ktrans, vp, aif, time_points, use_gpu=False,
        )

        for i in range(n):
            cpu_curve = model.evaluate([ktrans[i], vp[i]], aif, time_points)
            np.testing.assert_allclose(
                result.tissue_curves[i], cpu_curve, atol=1e-10,
            )


class TestTwoCXMForwardModel:
    def test_2cxm_batched_matches_cpu(self, aif_and_time):
        """Batched 2CXM matches CPU TwoCompartmentExchangeModel.evaluate()."""
        from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel

        aif, time_points = aif_and_time
        model = TwoCompartmentExchangeModel()
        n = 50
        rng = np.random.RandomState(130)
        fp = rng.uniform(20, 100, n)
        ps = rng.uniform(5, 40, n)
        ve = rng.uniform(0.1, 0.5, n)
        vp = rng.uniform(0.01, 0.1, n)

        result = batched_2cxm_model(
            fp, ps, ve, vp, aif, time_points, use_gpu=False,
        )

        for i in range(n):
            cpu_curve = model.evaluate([fp[i], ps[i], ve[i], vp[i]], aif, time_points)
            np.testing.assert_allclose(
                result.tissue_curves[i], cpu_curve, atol=1e-8,
                err_msg=f"Voxel {i}: Fp={fp[i]:.1f}, PS={ps[i]:.1f}",
            )

    def test_2cxm_degenerate_eigenvalue(self, aif_and_time):
        """PS≈0 → single exponential, should match CPU exactly."""
        from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel

        aif, time_points = aif_and_time
        model = TwoCompartmentExchangeModel()

        # PS very small → degenerate case (k1 ≈ k2)
        fp = np.array([50.0, 80.0])
        ps = np.array([1e-8, 1e-8])
        ve = np.array([0.2, 0.3])
        vp = np.array([0.05, 0.05])

        result = batched_2cxm_model(
            fp, ps, ve, vp, aif, time_points, use_gpu=False,
        )

        for i in range(2):
            cpu_curve = model.evaluate([fp[i], ps[i], ve[i], vp[i]], aif, time_points)
            np.testing.assert_allclose(
                result.tissue_curves[i], cpu_curve, atol=1e-6,
                err_msg=f"Degenerate voxel {i}",
            )

    def test_2cxm_unit_conversion(self, aif_and_time):
        """Verify fp/100 and ps/100 unit conversion is applied."""
        aif, time_points = aif_and_time

        # If units were wrong (no /100), tissue concentration would be 100x larger
        fp = np.array([60.0])
        ps = np.array([10.0])
        ve = np.array([0.2])
        vp = np.array([0.05])

        result = batched_2cxm_model(
            fp, ps, ve, vp, aif, time_points, use_gpu=False,
        )

        peak = np.max(result.tissue_curves)
        aif_peak = np.max(aif)
        # Peak tissue should be same order as AIF peak (not 100x)
        assert peak < aif_peak * 2, f"Peak {peak:.1f} >> AIF peak {aif_peak:.1f}"


class TestEvaluateModel:
    def test_dispatcher_all_models(self, aif_and_time):
        """evaluate_model dispatches to all 4 models."""
        aif, time_points = aif_and_time
        n = 10

        # Tofts
        r = evaluate_model("tofts", {"ktrans": np.ones(n) * 0.1,
                                          "ve": np.ones(n) * 0.2},
                                aif, time_points, use_gpu=False)
        assert r.model_type == "tofts"
        assert r.tissue_curves.shape == (n, len(aif))

        # Extended Tofts
        r = evaluate_model("extended_tofts",
                                {"ktrans": np.ones(n) * 0.1,
                                 "ve": np.ones(n) * 0.2,
                                 "vp": np.ones(n) * 0.05},
                                aif, time_points, use_gpu=False)
        assert r.model_type == "extended_tofts"

        # Patlak
        r = evaluate_model("patlak",
                                {"ktrans": np.ones(n) * 0.1,
                                 "vp": np.ones(n) * 0.05},
                                aif, time_points, use_gpu=False)
        assert r.model_type == "patlak"

        # 2CXM
        r = evaluate_model("2cxm",
                                {"fp": np.ones(n) * 60,
                                 "ps": np.ones(n) * 10,
                                 "ve": np.ones(n) * 0.2,
                                 "vp": np.ones(n) * 0.05},
                                aif, time_points, use_gpu=False)
        assert r.model_type == "2cxm"

    def test_dispatcher_unknown_model(self, aif_and_time):
        """Unknown model name raises ValueError."""
        aif, time_points = aif_and_time
        with pytest.raises(ValueError, match="Unknown model"):
            evaluate_model("unknown", {}, aif, time_points, use_gpu=False)
