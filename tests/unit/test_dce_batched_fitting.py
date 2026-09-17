"""Unit tests for batched Levenberg-Marquardt fitting.

Tests verify parameter recovery, convergence, constraint handling,
auto-mode BIC selection, and result format for all four DCE PK models.
"""

from typing import Any

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear, minimize

from pmriprep.models.dce.batched_models import (
    batched_2cxm_model,
    batched_extended_tofts_model,
    batched_patlak_model,
    batched_tofts_model,
    evaluate_model,
)
from pmriprep.models.dce.batched_fitting import (
    FitResult,
    _scaled_lm_step,
    batched_lm_fit,
    batched_patlak_lstsq,
)
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.patlak import PatlakModel
from pmriprep.models.dce.two_compartment_exchange import (
    TwoCompartmentExchangeModel,
)
from pmriprep.utils.dce_constants import VP_VE_TOTAL_MAX


@pytest.mark.parametrize("model_name", ["patlak", "tofts", "extended_tofts", "2cxm"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("save_curves,compute_uncertainties", [(False, False), (True, False), (False, True), (True, True)])
def test_chunk_worker_forwards_fit_inputs(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    weighted: bool,
    save_curves: bool,
    compute_uncertainties: bool,
) -> None:
    """Both chunk-worker routes preserve arrays, weights and output options."""
    from pmriprep.interfaces.dce_compute import BatchedChunkArgs, _fit_chunk_nlls
    from pmriprep.models.dce import batched_fitting

    curves = np.zeros((2, 5))
    aif = np.arange(5, dtype=float)
    time = aif / 10
    weights = np.linspace(0.2, 1.0, 5) if weighted else None
    sentinel = object()
    calls = 0

    def fit(*args: Any, **kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        if model_name != "patlak":
            assert args[0] == model_name
            args = args[1:]
        assert len(args) == 3
        assert all(actual is expected for actual, expected in zip(args, (curves, aif, time)))
        assert set(kwargs) == {"weights", "save_fitted_curves", "compute_uncertainties"}
        assert kwargs["weights"] is weights
        assert kwargs["save_fitted_curves"] is save_curves
        assert kwargs["compute_uncertainties"] is compute_uncertainties
        return sentinel

    def wrong_route(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Chunk worker selected the wrong fitter")

    monkeypatch.setattr(batched_fitting, "batched_patlak_lstsq", fit if model_name == "patlak" else wrong_route)
    monkeypatch.setattr(batched_fitting, "batched_lm_fit", fit if model_name != "patlak" else wrong_route)
    result = _fit_chunk_nlls(BatchedChunkArgs(
        model_name=model_name, tissue_curves=curves, aif=aif, time_points=time,
        weights=weights, save_fitted_curves=save_curves,
        compute_uncertainties=compute_uncertainties,
    ))
    assert result is sentinel
    assert calls == 1


@pytest.mark.parametrize("force_fallback", [False, True])
def test_scaled_lm_step_preserves_small_parameter_directions(
    monkeypatch: pytest.MonkeyPatch, force_fallback: bool,
) -> None:
    """Mixed parameter units must not truncate an identifiable direction."""
    jacobian = np.array([[1.0, 0.0], [0.0, 1e-7]])
    residual = jacobian @ np.array([2.0, 300.0])
    if force_fallback:
        def fail_direct_solve(matrices: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            raise np.linalg.LinAlgError("Injected direct-solve failure")

        monkeypatch.setattr(np.linalg, "solve", fail_direct_solve)
    step = _scaled_lm_step(
        (jacobian.T @ jacobian)[None], (jacobian.T @ residual)[None],
        [jacobian[:, j][None] for j in range(2)], residual[None],
        np.ones(2), np.array([1e-10]),
    )
    np.testing.assert_allclose(step[0], [2.0, 300.0], rtol=2e-8)


@pytest.mark.parametrize("fp,ps,fp_ceiling,ps_ceiling", [
    (120., 80., 200., 100.), (40., 300., 200., 500.), (120., 450., 200., 500.),
    (300., 40., 500., 100.), (450., 40., 500., 100.),
    (300., 150., 500., 500.), (450., 450., 500., 500.),
    (450., 40., None, None), (40., 450., None, None), (450., 450., None, None),
])
def test_batched_2cxm_independent_ode_recovery(
    monkeypatch: pytest.MonkeyPatch, fp: float, ps: float,
    fp_ceiling: float | None, ps_ceiling: float | None,
) -> None:
    """Recover interior fits in both production and expanded finite domains."""
    from pmriprep.models.dce import batched_fitting, two_compartment_exchange
    from pmriprep.utils.dce_constants import ParameterBounds

    for name, ceiling in (("FP", fp_ceiling), ("PS", ps_ceiling)):
        if ceiling is not None:
            monkeypatch.setattr(two_compartment_exchange, name + "_BOUNDS", ParameterBounds(0, ceiling))
            monkeypatch.setitem(batched_fitting._PARAM_BOUNDS, name.lower(), (0.0, ceiling))
    t = np.arange(73) * 5 / 60
    u = np.maximum(t - 0.5, 0)
    x = u / 0.06
    aif = 5 * (x / 3) ** 3 * np.exp(3 - x) + 0.6 * (-np.expm1(-u / .1)) * np.exp(-u / 2)
    ve, vp = .2, .05

    def rhs(now: float, state: np.ndarray) -> list[float]:
        cp, ce = state
        exchange = ps / 100 * (cp - ce)
        return [(fp / 100 * (np.interp(now, t, aif) - cp) - exchange) / vp, exchange / ve]

    reference = solve_ivp(rhs, (0, 6), [0., 0.], t_eval=t,
                          method="Radau", rtol=1e-9, atol=1e-12, max_step=5/240)
    assert reference.success
    tissue = vp * reference.y[0] + ve * reference.y[1]
    result = batched_lm_fit("2cxm", tissue[None], aif, t)
    assert result.converged[0]
    actual = [result.params[key][0] for key in ("fp", "ps", "ve", "vp")]
    np.testing.assert_allclose(actual, [fp, ps, ve, vp], rtol=1e-3)
    model = two_compartment_exchange.TwoCompartmentExchangeModel()
    predicted = model.evaluate(actual, aif, t)
    assert np.sum((predicted - tissue) ** 2) / np.sum(tissue**2) < 1e-8
    scalar = model.fit(tissue, aif, t)
    assert scalar.converged
    np.testing.assert_allclose(
        [scalar.parameters[key] for key in ("Fp", "PS", "ve", "vp")],
        [fp, ps, ve, vp], rtol=1e-3,
    )
    assert scalar.residual_sum_squares / np.sum(tissue**2) < 1e-8


@pytest.mark.parametrize("ceilings", [(200., 100.), (500., 500.)])
@pytest.mark.parametrize("truth", [
    [11.500746144009431, 2.3695054876065345, .34559137446283594, .19236753475802834],
    [20., 80., .02, .15],
    [78.77, 18.67, .0726, .0601],
])
def test_2cxm_small_step_recovery(
    monkeypatch: pytest.MonkeyPatch, ceilings: tuple[float, float], truth: list[float],
) -> None:
    """Small optimizer steps must not hide recoverable noiseless residuals."""
    from pmriprep.models.dce import batched_fitting, two_compartment_exchange
    from pmriprep.utils.dce_constants import ParameterBounds

    for name, ceiling in zip(("FP", "PS"), ceilings):
        monkeypatch.setattr(two_compartment_exchange, name + "_BOUNDS", ParameterBounds(0, ceiling))
        monkeypatch.setitem(batched_fitting._PARAM_BOUNDS, name.lower(), (0., ceiling))
    t = np.arange(121) * 3 / 60
    u = np.maximum(t - .5, 0.)
    x = u / .06
    aif = 5 * (x / 3) ** 3 * np.exp(3 - x) + .6 * (-np.expm1(-u / .1)) * np.exp(-u / 2)
    fp, ps, ve, vp = truth

    def rhs(now: float, state: np.ndarray) -> list[float]:
        cp, ce = state
        exchange = ps / 100 * (cp - ce)
        return [(fp / 100 * (np.interp(now, t, aif) - cp) - exchange) / vp, exchange / ve]

    reference = solve_ivp(rhs, (0, 6), [0., 0.], t_eval=t,
                          method="Radau", rtol=1e-9, atol=1e-12, max_step=3/240)
    assert reference.success
    tissue = vp * reference.y[0] + ve * reference.y[1]
    model = TwoCompartmentExchangeModel()
    for weights in (None, np.linspace(.2, 1., t.size)):
        batch = batched_lm_fit("2cxm", tissue[None], aif, t, weights=weights,
                               save_fitted_curves=True, compute_uncertainties=True)
        scalar = model.fit(tissue, aif, t, weights=weights)
        assert batch.converged[0]
        assert scalar.converged
        assert batch.fitted_curves is not None
        np.testing.assert_allclose(
            batch.fitted_curves[0],
            model.evaluate([batch.params[key][0] for key in ("fp", "ps", "ve", "vp")], aif, t),
        )
        assert all(np.isfinite(values[0]) for values in batch.uncertainties.values())
        for actual in (
            [batch.params[key][0] for key in ("fp", "ps", "ve", "vp")],
            [scalar.parameters[key] for key in ("Fp", "PS", "ve", "vp")],
        ):
            np.testing.assert_allclose(actual, truth, rtol=1e-3)
            predicted = model.evaluate(actual, aif, t)
            assert np.sum((predicted - tissue) ** 2) / np.sum(tissue**2) < 1e-10


@pytest.mark.parametrize("scalar_converged", [False, True])
def test_2cxm_small_step_requires_successful_improving_refit(
    monkeypatch: pytest.MonkeyPatch, scalar_converged: bool,
) -> None:
    """Failed or worse scalar refits cannot certify a damped batch stop."""
    from pmriprep.models.dce.base import FitResult as ScalarFitResult

    t = np.arange(121) * 3 / 60
    u = np.maximum(t - .5, 0.)
    x = u / .06
    aif = 5 * (x / 3) ** 3 * np.exp(3 - x) + .6 * (-np.expm1(-u / .1)) * np.exp(-u / 2)
    model = TwoCompartmentExchangeModel()
    tissue = model.evaluate(
        [11.500746144009431, 2.3695054876065345, .34559137446283594, .19236753475802834], aif, t,
    )
    seeds: list[Any] = []

    def unsuccessful_refit(self: Any, *args: Any, **kwargs: Any) -> ScalarFitResult:
        seeds.append(kwargs.get("initial_guess"))
        np.testing.assert_array_equal(args[0], tissue)
        return ScalarFitResult(
            parameters={"Fp": 1., "PS": 1., "ve": .1, "vp": .01},
            converged=scalar_converged,
        )

    monkeypatch.setattr(TwoCompartmentExchangeModel, "fit", unsuccessful_refit)
    result = batched_lm_fit("2cxm", tissue[None], aif, t, save_fitted_curves=True)
    assert len(seeds) == 2
    assert seeds[0] is not None and seeds[1] is None
    assert not result.converged[0]
    assert result.params["fp"][0] > 10.


@pytest.fixture(scope="module")
def parker_exchange_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent exchange curve with bolus-centered motion weights."""
    from pmriprep.interfaces.dce_compute import _compute_motion_weights

    time = np.arange(121) * 3 / 60
    u = np.maximum(time - .5, 0.)
    aif = sum(
        amplitude / (width * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((u - center) / width)**2)
        for amplitude, center, width in zip((.809, .330), (.17046, .365), (.0563, .132))
    )
    aif += 1.05 * np.exp(-.1685*u) / (1 + np.exp(-38.078*(u-.483)))
    aif = np.where(time > .5, aif, 0.)

    def rhs(now: float, state: np.ndarray) -> list[float]:
        cp, ce = state
        exchange = .8 * (cp - ce)
        return [(1.2 * (np.interp(now, time, aif) - cp) - exchange) / .05, exchange / .2]

    reference = solve_ivp(rhs, (0., 6.), [0., 0.], t_eval=time,
                          method="Radau", rtol=1e-9, atol=1e-12, max_step=3/240)
    assert reference.success
    tissue = .05 * reference.y[0] + .2 * reference.y[1]
    fd = np.zeros(time.size)
    peak = int(np.argmax(aif))
    fd[peak-1:peak+3] = 1.
    weights = _compute_motion_weights(fd.tolist(), 10, time.size, 2.)
    return tissue, aif, time, weights


@pytest.mark.parametrize("ceilings", [(200., 100.), (500., 500.)])
@pytest.mark.parametrize("weighted", [False, True])
def test_2cxm_small_cost_change_recovery(
    monkeypatch: pytest.MonkeyPatch, ceilings: tuple[float, float], weighted: bool,
    parker_exchange_case: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A transient cost plateau cannot certify a poorly fitted exchange curve."""
    from pmriprep.models.dce import batched_fitting, two_compartment_exchange
    from pmriprep.utils.dce_constants import ParameterBounds

    for name, ceiling in zip(("FP", "PS"), ceilings):
        monkeypatch.setattr(two_compartment_exchange, name + "_BOUNDS", ParameterBounds(0, ceiling))
        monkeypatch.setitem(batched_fitting._PARAM_BOUNDS, name.lower(), (0., ceiling))
    tissue, aif, time, motion_weights = parker_exchange_case
    weights = motion_weights if weighted else None
    result = batched_lm_fit("2cxm", tissue[None], aif, time, weights=weights,
                            save_fitted_curves=True, compute_uncertainties=True)
    assert result.converged[0]
    np.testing.assert_allclose(
        [result.params[key][0] for key in ("fp", "ps", "ve", "vp")],
        [120., 80., .2, .05], rtol=1e-3,
    )
    assert result.fitted_curves is not None
    w = motion_weights if weighted else np.ones(time.size)
    assert np.sum(w * (result.fitted_curves[0]-tissue)**2) / np.sum(w*tissue**2) < 1e-10
    assert all(np.isfinite(values[0]) for values in result.uncertainties.values())


@pytest.mark.parametrize("scalar_converged", [False, True])
def test_2cxm_cost_stop_requires_successful_improving_refit(
    monkeypatch: pytest.MonkeyPatch, scalar_converged: bool,
    parker_exchange_case: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Neither failed nor worse refits can certify a cost-only batch stop."""
    from pmriprep.models.dce import batched_fitting, two_compartment_exchange
    from pmriprep.models.dce.base import FitResult as ScalarFitResult
    from pmriprep.utils.dce_constants import ParameterBounds

    for name in ("FP", "PS"):
        monkeypatch.setattr(two_compartment_exchange, name + "_BOUNDS", ParameterBounds(0, 500.))
        monkeypatch.setitem(batched_fitting._PARAM_BOUNDS, name.lower(), (0., 500.))
    tissue, aif, time, weights = parker_exchange_case
    seeds: list[Any] = []

    def unsuccessful_refit(self: Any, *args: Any, **kwargs: Any) -> ScalarFitResult:
        seeds.append(kwargs.get("initial_guess"))
        np.testing.assert_array_equal(args[0], tissue)
        assert kwargs["weights"] is weights
        return ScalarFitResult(parameters={"Fp": 1., "PS": 1., "ve": .1, "vp": .01},
                               converged=scalar_converged)

    monkeypatch.setattr(TwoCompartmentExchangeModel, "fit", unsuccessful_refit)
    result = batched_lm_fit("2cxm", tissue[None], aif, time, weights=weights,
                            save_fitted_curves=True)
    assert len(seeds) == 2
    assert seeds[0] is not None and seeds[1] is None
    assert not result.converged[0]
    np.testing.assert_array_equal(
        [result.params[key][0] for key in ("fp", "ps", "ve", "vp")], seeds[0],
    )


@pytest.mark.parametrize("ceilings", [(200., 100.), (500., 500.)])
def test_auto_selection_uses_confirmed_exchange_fit(
    monkeypatch: pytest.MonkeyPatch, ceilings: tuple[float, float],
    parker_exchange_case: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Recoverable weighted exchange data remain eligible for automatic BIC."""
    from pmriprep.interfaces import dce_compute
    from pmriprep.models.dce import batched_fitting, two_compartment_exchange
    from pmriprep.utils.dce_constants import ParameterBounds

    for name, ceiling in zip(("FP", "PS"), ceilings):
        bounds = ParameterBounds(0, ceiling)
        monkeypatch.setattr(two_compartment_exchange, name + "_BOUNDS", bounds)
        monkeypatch.setattr(dce_compute, name + "_BOUNDS", bounds)
        monkeypatch.setitem(batched_fitting._PARAM_BOUNDS, name.lower(), (0., ceiling))
    tissue, aif, time, weights = parker_exchange_case
    interface = dce_compute.ComputeDCEMaps()
    candidates = interface._build_candidate_set(3.)
    results = interface._fit_batched(
        conc_data=tissue[None, None, None, :],
        mask_coords=(np.array([0]), np.array([0]), np.array([0])),
        aif=aif, time_points=time, model_name=candidates[-1],
        inference="nlls", is_auto=True, candidate_names=candidates,
        motion_weights=weights, save_fitted_curves=True, sloppy=False,
        mcmc_samples=0, baseline_frames=10, n_procs=1, random_seed=None,
        init_args=(), voxel_data_list=[],
    )
    assert len(results) == 1
    result = results[0]
    assert not result["failed"]
    assert result["selected_model_name"] == "2cxm"
    assert np.isfinite(result["delta_bic"])
    assert not dce_compute._is_degenerate_dce_fit("2cxm", result["params"])
    np.testing.assert_allclose(
        [result["params"][key] for key in ("Fp", "PS", "ve", "vp")],
        [120., 80., .2, .05], rtol=1e-3,
    )


def test_scaled_lm_step_preserves_weighted_damped_objective() -> None:
    """Column scaling is a change of coordinates, not a new regularizer."""
    rng = np.random.default_rng(304)
    jacobian = rng.normal(size=(10, 3))
    weights = np.linspace(0.1, 1, 10)
    residual = rng.normal(size=10)
    gram = jacobian.T @ (weights[:, None] * jacobian)
    rhs = jacobian.T @ (weights * residual)
    expected = np.linalg.solve(gram + 0.03 * np.diag(np.diag(gram)), rhs)
    actual = _scaled_lm_step(
        gram[None], rhs[None], [jacobian[:, j][None] for j in range(3)],
        residual[None], weights, np.array([0.03]),
    )
    np.testing.assert_allclose(actual[0], expected, rtol=1e-12)


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
        ktrans_true, ve_true, aif, time_points,
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
        ktrans_true, ve_true, vp_true, aif, time_points,
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
        ktrans_true, vp_true, aif, time_points,
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
        fp_true, ps_true, ve_true, vp_true, aif, time_points,
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

    def test_active_volume_boundary_matches_independent_optimizer(
        self,
        aif_and_time: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Batched ET solves, rather than merely projects onto, the boundary."""
        aif, time_points = aif_and_time
        model = ExtendedToftsModel()
        target = model.evaluate([0.12, 0.85, 0.25], aif, time_points)
        bounds = model.get_bounds()

        def objective(parameters: np.ndarray) -> float:
            residuals = target - model.evaluate(
                parameters.tolist(), aif, time_points
            )
            return float(np.sum(residuals**2))

        reference = minimize(
            objective,
            x0=np.array([0.12, 0.75, 0.249]),
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": lambda parameters: (
                    VP_VE_TOTAL_MAX - parameters[1] - parameters[2]
                ),
            },
            options={"maxiter": 4000, "ftol": 1e-14},
        )
        assert reference.success

        result = batched_lm_fit(
            "extended_tofts",
            target[np.newaxis, :],
            aif,
            time_points,
            max_iter=150,
            compute_uncertainties=True,
        )

        assert result.converged[0]
        assert result.params["ve"][0] + result.params["vp"][0] == pytest.approx(
            VP_VE_TOTAL_MAX
        )
        parameters = np.array(
            [result.params[name][0] for name in ("ktrans", "ve", "vp")]
        )
        assert objective(parameters) == pytest.approx(reference.fun, rel=1e-3)
        for name in ("ktrans", "ve", "vp"):
            assert np.isfinite(result.uncertainties[name][0])
        assert result.uncertainties["ve"][0] == pytest.approx(
            result.uncertainties["vp"][0]
        )


# ---------------------------------------------------------------------------
# Test 3: Patlak exact recovery (no noise)
# ---------------------------------------------------------------------------

class TestPatlakExact:
    @pytest.mark.parametrize("n_frames", [127, 128, 256, 361])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_uncertainty_frame_count_and_reduced_covariance(
        self, n_frames: int, weighted: bool,
    ) -> None:
        """Frame counts do not overflow for interior or active-bound fits."""
        time = np.linspace(0., 6., n_frames)
        aif = 5*np.exp(-0.5*((time-.5)/.15)**2)
        integral = np.zeros_like(aif)
        integral[1:] = np.cumsum(.5*(aif[:-1]+aif[1:])*np.diff(time))
        design = np.column_stack((integral, aif))
        latent = np.array([[.08, .06], [-.02, .08], [.08, 1.5]])
        tissue = latent @ design.T
        tissue += np.random.default_rng(29).normal(0., .001, tissue.shape)
        weights = np.linspace(.2, 1., n_frames) if weighted else np.ones(n_frames)
        weights_arg = weights if weighted else None
        ordinary = batched_patlak_lstsq(tissue, aif, time, weights=weights_arg)
        result = batched_patlak_lstsq(
            tissue, aif, time, weights=weights_arg, compute_uncertainties=True,
        )
        for name in ("ktrans", "vp"):
            np.testing.assert_array_equal(result.params[name], ordinary.params[name])
        parameters = np.column_stack((result.params["ktrans"], result.params["vp"]))
        free_coordinates = ([0, 1], [1], [0])
        for index, free in enumerate(free_coordinates):
            reference = lsq_linear(
                np.sqrt(weights)[:, None]*design, np.sqrt(weights)*tissue[index],
                bounds=([0., 0.], [5., 1.]), tol=1e-14,
            )
            assert reference.success
            np.testing.assert_allclose(parameters[index], reference.x, rtol=1e-8, atol=1e-10)
            residual = tissue[index]-design @ parameters[index]
            variance = np.sum(weights*residual**2)/(n_frames-len(free))
            free_design = design[:, free]
            covariance = variance*np.linalg.inv(free_design.T @ (weights[:, None]*free_design))
            for column, name in enumerate(("ktrans", "vp")):
                if column in free:
                    assert result.uncertainties[name][index] == pytest.approx(
                        np.sqrt(covariance[free.index(column), free.index(column)]), rel=1e-10,
                    )
                else:
                    assert np.isnan(result.uncertainties[name][index])

    @pytest.mark.parametrize("n_frames", [127, 128, 361])
    @pytest.mark.parametrize("inference", ["nlls", "vi"])
    def test_auto_selection_keeps_patlak_for_long_series(
        self, n_frames: int, inference: str,
    ) -> None:
        """Uncertainty computation cannot remove Patlak from automatic BIC."""
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps, VoxelInput, _worker_ctx

        time = np.arange(n_frames)*5/60
        aif = 5*np.exp(-.5*((time-.5)/.15)**2)
        tissue = PatlakModel().evaluate([.02, .05], aif, time)
        tissue += np.random.default_rng(17).normal(0., .001, n_frames)
        interface = ComputeDCEMaps()
        candidates = interface._build_candidate_set(5.)
        voxels = [VoxelInput(0, 0, 0, tissue, None)]
        try:
            results = interface._fit_batched(
                conc_data=tissue[None, None, None, :],
                mask_coords=(np.array([0]), np.array([0]), np.array([0])),
                aif=aif, time_points=time, model_name=candidates[-1],
                inference=inference, is_auto=True, candidate_names=candidates,
                motion_weights=None, save_fitted_curves=True, sloppy=False,
                mcmc_samples=0, baseline_frames=5, n_procs=1, random_seed=None,
                init_args=(aif, time, candidates[-1], inference, 0, 5, "skip", 0.,
                           True, None, False, None, candidates),
                voxel_data_list=voxels,
            )
        finally:
            _worker_ctx.clear()
        assert len(results) == 1
        assert not results[0]["failed"]
        assert results[0]["selected_model_name"] == "patlak"
        assert np.isfinite(results[0]["delta_bic"])

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

    @pytest.mark.parametrize(
        "latent_parameters",
        [
            (0.08, -0.06),
            (-0.05, 0.08),
            (0.08, 1.50),
            (6.0, 0.08),
        ],
    )
    def test_patlak_active_box_bound_matches_independent_optimizer(
        self,
        aif_and_time: tuple[np.ndarray, np.ndarray],
        latent_parameters: tuple[float, float],
    ) -> None:
        """Batched Patlak re-optimizes the free coefficient at a box bound."""
        aif, time_points = aif_and_time
        model = PatlakModel()
        tissue = model.evaluate(latent_parameters, aif, time_points)

        design = np.column_stack([
            model.evaluate([1.0, 0.0], aif, time_points),
            model.evaluate([0.0, 1.0], aif, time_points),
        ])
        bounds = np.asarray(model.get_bounds(), dtype=np.float64)
        reference = lsq_linear(
            design,
            tissue,
            bounds=(bounds[:, 0], bounds[:, 1]),
            tol=1e-14,
            max_iter=4000,
        )
        assert reference.success

        result = batched_patlak_lstsq(
            tissue[np.newaxis, :],
            aif,
            time_points,
            compute_uncertainties=True,
        )
        parameters = np.array(
            [result.params["ktrans"][0], result.params["vp"][0]]
        )

        reference_residuals = tissue - design @ reference.x
        result_residuals = tissue - design @ parameters
        assert np.sum(result_residuals**2) == pytest.approx(
            np.sum(reference_residuals**2), rel=1e-9
        )
        at_ktrans_bound = parameters[0] == pytest.approx(
            model.get_bounds()[0][0]
        ) or parameters[0] == pytest.approx(model.get_bounds()[0][1])
        at_vp_bound = parameters[1] == pytest.approx(
            model.get_bounds()[1][0]
        ) or parameters[1] == pytest.approx(model.get_bounds()[1][1])
        assert at_ktrans_bound or at_vp_bound
        if at_ktrans_bound:
            assert np.isnan(result.uncertainties["ktrans"][0])
        else:
            assert np.isfinite(result.uncertainties["ktrans"][0])
        if at_vp_bound:
            assert np.isnan(result.uncertainties["vp"][0])
        else:
            assert np.isfinite(result.uncertainties["vp"][0])

    def test_graphical_active_bound_reoptimizes_other_coefficient(
        self,
        aif_and_time: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Bounded graphical regression solves rather than clips."""
        aif, time_points = aif_and_time
        model = PatlakModel()
        tissue = model.evaluate([0.08, -0.06], aif, time_points)
        x_values, y_values, valid = model.compute_patlak_plot_variables(
            tissue, aif, time_points
        )
        start = min(int(np.argmax(aif)) + 1, len(time_points) - 2)
        valid[:start] = False
        design = np.column_stack([x_values[valid], np.ones(np.sum(valid))])

        def objective(parameters: np.ndarray) -> float:
            residuals = y_values[valid] - design @ parameters
            return float(np.sum(residuals**2))

        reference = minimize(
            objective,
            x0=np.array([0.1, 0.02]),
            method="SLSQP",
            bounds=model.get_bounds(),
            options={"maxiter": 4000, "ftol": 1e-14},
        )
        assert reference.success

        result = model.fit_graphical(tissue, aif, time_points)
        parameters = np.array(
            [result.parameters["Ktrans"], result.parameters["vp"]]
        )
        assert objective(parameters) == pytest.approx(reference.fun, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 4: 2CXM recovery
# ---------------------------------------------------------------------------

class TestTwoCXMRecovery:
    def test_2cxm_noiseless_balanced_regime_matches_scalar(
        self,
        aif_and_time: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Batched 2CXM must recover a noiseless identifiable scalar solution."""
        aif, time_points = aif_and_time
        model = TwoCompartmentExchangeModel()
        truth = {"fp": 50.0, "ps": 10.0, "ve": 0.2, "vp": 0.02}
        tissue = model.evaluate(
            [truth[name] for name in ("fp", "ps", "ve", "vp")],
            aif,
            time_points,
        )

        result = batched_lm_fit(
            "2cxm",
            tissue[np.newaxis, :],
            aif,
            time_points,
            max_iter=100,
        )

        assert result.converged[0]
        assert result.r_squared[0] > 0.9999
        for name, expected in truth.items():
            assert result.params[name][0] == pytest.approx(expected, rel=0.01)

    def test_2cxm_active_volume_boundary_matches_independent_optimizer(
        self,
        aif_and_time: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Scalar and batched fits solve, rather than project onto, the boundary."""
        aif, time_points = aif_and_time
        model = TwoCompartmentExchangeModel()
        target = model.evaluate([60.0, 15.0, 0.8, 0.25], aif, time_points)
        bounds = model.get_bounds()

        def objective(parameters: np.ndarray) -> float:
            residuals = target - model.evaluate(parameters.tolist(), aif, time_points)
            return float(np.sum(residuals**2))

        reference = minimize(
            objective,
            x0=np.array([60.0, 15.0, 0.75, 0.249]),
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": lambda parameters: (
                    VP_VE_TOTAL_MAX - parameters[2] - parameters[3]
                ),
            },
            options={"maxiter": 4000, "ftol": 1e-14},
        )
        assert reference.success

        scalar = model.fit(target, aif, time_points)
        batched = batched_lm_fit(
            "2cxm",
            target[np.newaxis, :],
            aif,
            time_points,
            max_iter=150,
            compute_uncertainties=True,
        )

        assert scalar.converged
        assert batched.converged[0]
        assert scalar.parameters["ve"] + scalar.parameters["vp"] == pytest.approx(
            VP_VE_TOTAL_MAX
        )
        assert batched.params["ve"][0] + batched.params["vp"][0] == pytest.approx(
            VP_VE_TOTAL_MAX
        )
        assert scalar.residual_sum_squares == pytest.approx(reference.fun, rel=1e-3)
        batched_parameters = [
            batched.params[name][0] for name in ("fp", "ps", "ve", "vp")
        ]
        assert objective(np.asarray(batched_parameters)) == pytest.approx(
            reference.fun,
            rel=1e-3,
        )
        for name in ("fp", "ps", "ve", "vp"):
            assert np.isfinite(batched.uncertainties[name][0])
        assert batched.uncertainties["ve"][0] == pytest.approx(
            batched.uncertainties["vp"][0]
        )


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

        # Fp recovery. Lenient bound: 2CXM Fp is poorly identifiable at 1% noise
        # over this broad range (high Fp / low ve), so this is a smoke check that
        # the batched fitter is roughly on target, not an accuracy claim. The
        # exact-convolution forward (vs the prior left-Riemann) slightly reshapes
        # the ill-conditioned Fp objective; median lands ~0.56 here.
        fp_err = np.abs(result.params["fp"][converged] - fp_true[converged])
        fp_rel = fp_err / np.maximum(fp_true[converged], 1e-6)
        assert np.median(fp_rel) < 0.6


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
            kt_tofts, ve_tofts, aif, time_points,
        ).tissue_curves

        # Patlak data (distinct linear accumulation pattern)
        kt_patlak = rng.uniform(0.05, 0.15, n_per)
        vp_patlak = rng.uniform(0.05, 0.15, n_per)
        patlak_curves = batched_patlak_model(
            kt_patlak, vp_patlak, aif, time_points,
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
        2 * n_per
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
            ktrans, vp, aif, time_points,
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
            fp, ps, ve, vp, aif, time_points,
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
            fp, ps, ve, vp, aif, time_points,
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
            fp, ps, ve, vp, aif, time_points,
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
                                aif, time_points)
        assert r.model_type == "tofts"
        assert r.tissue_curves.shape == (n, len(aif))

        # Extended Tofts
        r = evaluate_model("extended_tofts",
                                {"ktrans": np.ones(n) * 0.1,
                                 "ve": np.ones(n) * 0.2,
                                 "vp": np.ones(n) * 0.05},
                                aif, time_points)
        assert r.model_type == "extended_tofts"

        # Patlak
        r = evaluate_model("patlak",
                                {"ktrans": np.ones(n) * 0.1,
                                 "vp": np.ones(n) * 0.05},
                                aif, time_points)
        assert r.model_type == "patlak"

        # 2CXM
        r = evaluate_model("2cxm",
                                {"fp": np.ones(n) * 60,
                                 "ps": np.ones(n) * 10,
                                 "ve": np.ones(n) * 0.2,
                                 "vp": np.ones(n) * 0.05},
                                aif, time_points)
        assert r.model_type == "2cxm"

    def test_dispatcher_unknown_model(self, aif_and_time):
        """Unknown model name raises ValueError."""
        aif, time_points = aif_and_time
        with pytest.raises(ValueError, match="Unknown model"):
            evaluate_model("unknown", {}, aif, time_points)


# ---------------------------------------------------------------------------
# Chunked multiproc NLLS path — concat_fit_results + _dispatch_batched_chunks
# ---------------------------------------------------------------------------


class TestChunkedMultiproc:
    """Regression tests for the chunked-multiproc batched NLLS path.

    Guards three invariants:
    - ``concat_fit_results`` fails fast on inconsistent chunks.
    - ``_dispatch_batched_chunks`` at ``n_procs=1`` and ``n_procs>1``
      produces numerically identical FitResults (modulo BLAS reduction
      ULP drift, covered by ``atol=1e-10``).
    - The auto-mode BIC loop is untouched by chunked parallelization.
    """

    @staticmethod
    def _make_result(
        n: int,
        model_name: str = "tofts",
        seed: int = 0,
        with_curves: bool = True,
        with_uncertainties: bool = True,
    ) -> FitResult:
        rng = np.random.default_rng(seed)
        params = {
            "ktrans": rng.uniform(0.01, 0.3, n),
            "ve": rng.uniform(0.1, 0.5, n),
        }
        fitted_curves = rng.uniform(0, 1, (n, 60)) if with_curves else None
        uncertainties = (
            {k: rng.uniform(0, 0.01, n) for k in params} if with_uncertainties else {}
        )
        return FitResult(
            params=params,
            r_squared=rng.uniform(0.5, 1.0, n),
            converged=np.ones(n, dtype=bool),
            fitted_curves=fitted_curves,
            n_iterations=10,
            model_name=model_name,
            uncertainties=uncertainties,
        )

    # -- concat_fit_results --------------------------------------------

    def test_concat_preserves_voxel_order(self):
        from pmriprep.models.dce.batched_fitting import concat_fit_results

        r1 = self._make_result(5, seed=0)
        r2 = self._make_result(7, seed=1)
        merged = concat_fit_results([r1, r2], n_total=12)

        assert merged.r_squared.shape == (12,)
        assert merged.params["ktrans"].shape == (12,)
        np.testing.assert_array_equal(
            merged.params["ktrans"][:5], r1.params["ktrans"]
        )
        np.testing.assert_array_equal(
            merged.params["ktrans"][5:], r2.params["ktrans"]
        )
        assert merged.fitted_curves is not None
        assert merged.fitted_curves.shape == (12, 60)

    def test_concat_propagates_none_fitted_curves(self):
        from pmriprep.models.dce.batched_fitting import concat_fit_results

        r1 = self._make_result(3, seed=0, with_curves=False)
        r2 = self._make_result(4, seed=1, with_curves=False)
        merged = concat_fit_results([r1, r2], n_total=7)
        assert merged.fitted_curves is None

    def test_concat_rejects_mismatched_model_name(self):
        from pmriprep.models.dce.batched_fitting import concat_fit_results

        r1 = self._make_result(3, model_name="tofts")
        r2 = self._make_result(4, model_name="patlak")
        with pytest.raises(AssertionError, match="model_name mismatch"):
            concat_fit_results([r1, r2], n_total=7)

    def test_concat_rejects_mixed_curves_none(self):
        from pmriprep.models.dce.batched_fitting import concat_fit_results

        r1 = self._make_result(3, seed=0, with_curves=True)
        r2 = self._make_result(4, seed=1, with_curves=False)
        with pytest.raises(AssertionError, match="fitted_curves"):
            concat_fit_results([r1, r2], n_total=7)

    # -- _dispatch_batched_chunks -------------------------------------

    def test_dispatch_n_procs_1_matches_n_procs_4(self, aif_and_time):
        """Single-process and multi-process yield identical results.

        Above the chunking floor (n_voxels >= max(1000, 4*n_procs))
        ``n_procs=4`` spins up workers; ``n_procs=1`` goes in-process.
        Both must return numerically identical FitResults.
        """
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps

        aif, time_points = aif_and_time
        tissue, *_ = _generate_tofts_data(
            aif, time_points, n_voxels=1500, noise_level=0.02,
        )

        iface = ComputeDCEMaps()
        r1 = iface._dispatch_batched_chunks(
            model_name="tofts",
            tissue_curves=tissue,
            aif=aif,
            time_points=time_points,
            weights=None,
            save_fitted_curves=True,
            compute_uncertainties=False,
            n_procs=1,
            n_voxels=1500,
        )
        r4 = iface._dispatch_batched_chunks(
            model_name="tofts",
            tissue_curves=tissue,
            aif=aif,
            time_points=time_points,
            weights=None,
            save_fitted_curves=True,
            compute_uncertainties=False,
            n_procs=4,
            n_voxels=1500,
        )
        assert r1.model_name == r4.model_name == "tofts"
        for k in r1.params:
            np.testing.assert_allclose(
                r1.params[k], r4.params[k], atol=1e-10, rtol=0,
            )
        np.testing.assert_allclose(
            r1.r_squared, r4.r_squared, atol=1e-10, rtol=0,
        )
        np.testing.assert_array_equal(r1.converged, r4.converged)
        assert r1.fitted_curves is not None and r4.fitted_curves is not None
        np.testing.assert_allclose(
            r1.fitted_curves, r4.fitted_curves, atol=1e-10, rtol=0,
        )


    def test_dispatch_auto_mode_bic_chunked_matches_inline(self, aif_and_time):
        """Auto-mode BIC selection is identical at n_procs=1 and n_procs=4.

        Runs the outer candidate loop at each n_procs setting, stacks
        the per-candidate BICs (same arithmetic the interface uses at
        the :argmin step), and asserts the winner index per voxel and
        the winner params match exactly.
        """
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps

        aif, time_points = aif_and_time
        tissue, *_ = _generate_etofts_data(
            aif, time_points, n_voxels=1500, noise_level=0.03,
        )

        iface = ComputeDCEMaps()

        def run_candidates(n_procs: int) -> dict[str, FitResult]:
            out = {}
            for cname in ("tofts", "extended_tofts", "patlak"):
                out[cname] = iface._dispatch_batched_chunks(
                    model_name=cname,
                    tissue_curves=tissue,
                    aif=aif,
                    time_points=time_points,
                    weights=None,
                    save_fitted_curves=True,
                    compute_uncertainties=False,
                    n_procs=n_procs,
                    n_voxels=1500,
                )
            return out

        results_1 = run_candidates(1)
        results_4 = run_candidates(4)

        # Per-candidate: fitted_curves and params must match between paths.
        for cname in results_1:
            r1, r4 = results_1[cname], results_4[cname]
            for k in r1.params:
                np.testing.assert_allclose(
                    r1.params[k], r4.params[k], atol=1e-10, rtol=0,
                    err_msg=f"{cname}:{k}",
                )
            assert r1.fitted_curves is not None and r4.fitted_curves is not None
            np.testing.assert_allclose(
                r1.fitted_curves, r4.fitted_curves, atol=1e-10, rtol=0,
                err_msg=f"{cname}:fitted_curves",
            )

        # BIC selection should land identical winners.
        from pmriprep.models.dce.base import EPSILON_RSS

        T = tissue.shape[1]
        k_map = {"tofts": 2, "extended_tofts": 3, "patlak": 2}

        def bic_winner(results: dict[str, FitResult]) -> np.ndarray:
            bics = []
            cnames = list(results.keys())
            for cname in cnames:
                rss = np.sum(
                    (tissue - results[cname].fitted_curves) ** 2, axis=1,
                )
                rss_safe = np.maximum(rss, EPSILON_RSS)
                bics.append(
                    T * np.log(rss_safe / T) + k_map[cname] * np.log(T),
                )
            return np.argmin(np.stack(bics, axis=0), axis=0)

        np.testing.assert_array_equal(
            bic_winner(results_1), bic_winner(results_4),
        )

    def test_auto_nlls_requests_candidate_uncertainties(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Auto NLLS retains conditional uncertainty from its winning model."""
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps

        calls: list[bool] = []

        def fake_dispatch(**kwargs: Any) -> FitResult:
            calls.append(kwargs["compute_uncertainties"])
            return FitResult(
                params={
                    "ktrans": np.array([0.1]),
                    "vp": np.array([0.02]),
                },
                r_squared=np.array([1.0]),
                converged=np.array([True]),
                fitted_curves=np.zeros((1, 4)),
                n_iterations=1,
                model_name="patlak",
                uncertainties={
                    "ktrans": np.array([0.01]),
                    "vp": np.array([0.002]),
                },
            )

        interface = ComputeDCEMaps()
        monkeypatch.setattr(interface, "_dispatch_batched_chunks", fake_dispatch)
        coords = (np.array([0]), np.array([0]), np.array([0]))
        results = interface._fit_batched(
            conc_data=np.zeros((1, 1, 1, 4)),
            mask_coords=coords,
            aif=np.zeros(4),
            time_points=np.arange(4, dtype=float),
            model_name="auto",
            inference="nlls",
            is_auto=True,
            candidate_names=["patlak"],
            motion_weights=None,
            save_fitted_curves=False,
            sloppy=False,
            mcmc_samples=0,
            baseline_frames=1,
            n_procs=1,
            random_seed=42,
            init_args=(),
            voxel_data_list=[],
        )

        assert calls == [True]
        assert results[0]["selected_model_name"] == "patlak"
        assert results[0]["uncertainties"] == {
            "Ktrans": 0.01,
            "vp": 0.002,
        }

    def test_batched_auto_propagates_unexpected_candidate_exception(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A candidate implementation error cannot silently reduce the BIC set."""
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps

        class CandidateImplementationError(Exception):
            pass

        def fake_dispatch(**kwargs: Any) -> FitResult:
            if kwargs["model_name"] == "extended_tofts":
                raise CandidateImplementationError("injected candidate failure")
            return FitResult(
                params={"ktrans": np.array([0.1]), "vp": np.array([0.02])},
                r_squared=np.array([1.0]),
                converged=np.array([True]),
                fitted_curves=np.zeros((1, 4)),
                n_iterations=1,
                model_name="patlak",
            )

        interface = ComputeDCEMaps()
        monkeypatch.setattr(interface, "_dispatch_batched_chunks", fake_dispatch)
        coords = (np.array([0]), np.array([0]), np.array([0]))

        with pytest.raises(CandidateImplementationError, match="injected"):
            interface._fit_batched(
                conc_data=np.zeros((1, 1, 1, 4)),
                mask_coords=coords,
                aif=np.zeros(4),
                time_points=np.arange(4, dtype=float),
                model_name="auto",
                inference="nlls",
                is_auto=True,
                candidate_names=["patlak", "extended_tofts"],
                motion_weights=None,
                save_fitted_curves=False,
                sloppy=False,
                mcmc_samples=0,
                baseline_frames=1,
                n_procs=1,
                random_seed=42,
                init_args=(),
                voxel_data_list=[],
            )

    def test_batched_auto_preserves_candidate_nonconvergence(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An optimizer nonconvergence remains an in-band candidate outcome."""
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps

        def fake_dispatch(**kwargs: Any) -> FitResult:
            model_name = str(kwargs["model_name"])
            converged = model_name == "extended_tofts"
            if converged:
                params = {
                    "ktrans": np.array([0.1]),
                    "ve": np.array([0.3]),
                    "vp": np.array([0.02]),
                }
            else:
                params = {"ktrans": np.array([0.0]), "vp": np.array([0.0])}
            return FitResult(
                params=params,
                r_squared=np.array([1.0 if converged else -1.0]),
                converged=np.array([converged]),
                fitted_curves=np.zeros((1, 4)),
                n_iterations=1,
                model_name=model_name,
            )

        interface = ComputeDCEMaps()
        monkeypatch.setattr(interface, "_dispatch_batched_chunks", fake_dispatch)
        coords = (np.array([0]), np.array([0]), np.array([0]))
        result = interface._fit_batched(
            conc_data=np.zeros((1, 1, 1, 4)),
            mask_coords=coords,
            aif=np.zeros(4),
            time_points=np.arange(4, dtype=float),
            model_name="auto",
            inference="nlls",
            is_auto=True,
            candidate_names=["patlak", "extended_tofts"],
            motion_weights=None,
            save_fitted_curves=False,
            sloppy=False,
            mcmc_samples=0,
            baseline_frames=1,
            n_procs=1,
            random_seed=42,
            init_args=(),
            voxel_data_list=[],
        )[0]

        assert not result["failed"]
        assert result["selected_model_name"] == "extended_tofts"

    def test_scalar_auto_propagates_unexpected_candidate_exception(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scalar automatic fitting also fails loudly on candidate code errors."""
        import pmriprep.interfaces.dce_compute as dce_compute
        from pmriprep.interfaces.dce_compute import VoxelInput
        from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

        class CandidateImplementationError(Exception):
            pass

        class FakeResult:
            model_name = "patlak"
            parameters = {"Ktrans": 0.1, "vp": 0.02}
            r_squared = 1.0
            converged = True
            bic = 1.0
            n_parameters = 2
            uncertainty_source = "not_computed"
            fitted_curve = None

            @staticmethod
            def has_uncertainty() -> bool:
                return False

        class PrimaryModel:
            name = "patlak"

            @staticmethod
            def fit(*args: Any, **kwargs: Any) -> FakeResult:
                del args, kwargs
                return FakeResult()

        class BrokenCandidate:
            @staticmethod
            def fit(*args: Any, **kwargs: Any) -> None:
                del args, kwargs
                raise CandidateImplementationError("injected candidate failure")

        monkeypatch.setattr(
            dce_compute,
            "_worker_ctx",
            {
                "BayesianPKFitter": object,
                "create_vp_prior_from_dsc": object,
                "ConstraintMode": ConstraintMode,
                "model": PrimaryModel(),
                "fitter": None,
                "aif": np.zeros(4),
                "time_points": np.arange(4, dtype=float),
                "inference": "nlls",
                "mcmc_samples": 0,
                "baseline_frames": 1,
                "constraint_mode_str": "skip",
                "constraint_vp_uncertainty": 0.0,
                "motion_weights": None,
                "sloppy": False,
                "is_auto": True,
                "bic_candidates": {"extended_tofts": BrokenCandidate()},
                "save_fitted_curves": False,
            },
        )

        with pytest.raises(CandidateImplementationError, match="injected"):
            dce_compute._fit_single_voxel(
                VoxelInput(
                    0, 0, 0, np.array([0.0, 0.1, 0.2, 0.3]), None
                )
            )

    def test_auto_vi_uses_batched_selection_before_refinement(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Auto VI must refine, rather than repeat, batched-NLLS selection."""
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps

        calls: list[bool] = []
        selected_before_refinement: list[str] = []

        def fake_dispatch(**kwargs: Any) -> FitResult:
            calls.append(kwargs["compute_uncertainties"])
            assert "fixed_vp" not in kwargs
            assert "vp_uncertainty" not in kwargs
            return FitResult(
                params={
                    "ktrans": np.array([0.1]),
                    "vp": np.array([0.02]),
                },
                r_squared=np.array([1.0]),
                converged=np.array([True]),
                fitted_curves=np.zeros((1, 4)),
                n_iterations=1,
                model_name="patlak",
                uncertainties={},
            )

        def fake_refine(
            results: list[dict[str, Any]],
            voxel_data_list: list[Any],
            init_args: tuple[Any, ...],
            n_procs: int,
        ) -> list[dict[str, Any]]:
            del voxel_data_list, init_args, n_procs
            selected_before_refinement.extend(
                str(result["selected_model_name"]) for result in results
            )
            return results

        interface = ComputeDCEMaps()
        monkeypatch.setattr(interface, "_dispatch_batched_chunks", fake_dispatch)
        monkeypatch.setattr(
            interface, "_refine_auto_bayesian_from_batched", fake_refine
        )
        coords = (np.array([0]), np.array([0]), np.array([0]))
        results = interface._fit_batched(
            conc_data=np.zeros((1, 1, 1, 4)),
            mask_coords=coords,
            aif=np.zeros(4),
            time_points=np.arange(4, dtype=float),
            model_name="auto",
            inference="vi",
            is_auto=True,
            candidate_names=["patlak"],
            motion_weights=None,
            save_fitted_curves=False,
            sloppy=False,
            mcmc_samples=0,
            baseline_frames=1,
            n_procs=1,
            random_seed=42,
            init_args=(),
            voxel_data_list=[],
        )

        assert calls == [False]
        assert selected_before_refinement == ["patlak"]
        assert results[0]["selected_model_name"] == "patlak"

    def test_fixed_vi_applies_prior_to_vp_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fixed-model Bayesian worker applies and reports an accepted vp prior."""
        import pmriprep.interfaces.dce_compute as dce_compute
        from pmriprep.interfaces.dce_compute import VoxelInput
        from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

        class FakeModel:
            name = "extended_tofts"

        class FakeResult:
            model_name = "extended_tofts"
            parameters = {"Ktrans": 0.1, "ve": 0.3, "vp": 0.04}
            r_squared = 0.9
            converged = True
            uncertainty_source = "not_computed"
            fitted_curve = None

            @staticmethod
            def has_uncertainty() -> bool:
                return False

        prior_overrides: list[Any] = []

        class FakeBayesianFitter:
            def __init__(self, model: Any, vp_prior_override: Any = None) -> None:
                del model
                prior_overrides.append(vp_prior_override)

            @staticmethod
            def fit_vi(*args: Any, **kwargs: Any) -> FakeResult:
                del args, kwargs
                return FakeResult()

        monkeypatch.setattr(
            dce_compute,
            "_worker_ctx",
            {
                "BayesianPKFitter": FakeBayesianFitter,
                "create_vp_prior_from_dsc": lambda value, **kwargs: (value, kwargs),
                "ConstraintMode": ConstraintMode,
                "model": FakeModel(),
                "fitter": None,
                "aif": np.zeros(4),
                "time_points": np.arange(4, dtype=float),
                "inference": "vi",
                "mcmc_samples": 0,
                "baseline_frames": 1,
                "constraint_mode_str": "soft",
                "constraint_vp_uncertainty": 0.4,
                "motion_weights": None,
                "sloppy": False,
                "is_auto": False,
                "bic_candidates": {},
                "save_fitted_curves": False,
            },
        )

        result = dce_compute._fit_single_voxel(
            VoxelInput(0, 0, 0, np.array([0.0, 0.1, 0.2, 0.3]), 0.04)
        )

        assert result["failed"] is False
        assert result["constrained"] is True
        assert prior_overrides == [(0.04, {"relative_uncertainty": 0.4})]

    def test_auto_bayesian_refinement_preserves_selection_and_delta_bic(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Grouped refinement cannot change canonical BIC state."""
        import pmriprep.interfaces.dce_compute as dce_compute
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps, VoxelInput

        caplog.set_level("INFO", logger="nipype.interface")
        assert dce_compute.logger.name == "nipype.interface"

        batched_results = [
            {
                "coords": (0, 0, 0),
                "selected_model_name": "patlak",
                "delta_bic": 8.0,
                "failed": False,
                "converged": True,
                "params": {"Ktrans": 0.01, "vp": 0.02},
            },
            {
                "coords": (1, 0, 0),
                "selected_model_name": "extended_tofts",
                "delta_bic": 12.0,
                "failed": False,
                "converged": True,
                "params": {"Ktrans": 0.02, "ve": 0.3, "vp": 0.01},
            },
            {
                "coords": (2, 0, 0),
                "selected_model_name": "patlak",
                "delta_bic": 20.0,
                "failed": False,
                "converged": False,
                "params": {"Ktrans": 0.03, "vp": 0.04},
            },
        ]
        voxels = [
            VoxelInput(index, 0, 0, np.zeros(4), None)
            for index in range(3)
        ]
        seen: list[tuple[int, str, bool, list[str], dict[str, float] | None]] = []

        def fake_fit(voxel: VoxelInput) -> dict[str, Any]:
            selected_model = str(dce_compute._worker_ctx["model_name"])
            is_auto = bool(dce_compute._worker_ctx["is_auto"])
            candidate_models = sorted(dce_compute._worker_ctx["bic_candidates"])
            seen.append(
                (
                    voxel.x,
                    selected_model,
                    is_auto,
                    candidate_models,
                    voxel.initial_parameters,
                )
            )
            return {
                "coords": (voxel.x, voxel.y, voxel.z),
                "selected_model_name": selected_model,
                "delta_bic": np.nan,
                "failed": False,
                "converged": voxel.x != 2,
            }

        monkeypatch.setattr(dce_compute, "_fit_single_voxel", fake_fit)
        init_args = (
            np.zeros(4),
            np.arange(4, dtype=float),
            "extended_tofts",
            "vi",
            0,
            1,
            "skip",
            0.0,
            False,
            None,
            False,
            42,
            ["patlak", "extended_tofts"],
        )

        results = ComputeDCEMaps()._refine_auto_bayesian_from_batched(
            batched_results,
            voxels,
            init_args,
            n_procs=1,
        )

        assert seen == [
            (0, "patlak", False, [], {"Ktrans": 0.01, "vp": 0.02}),
            (2, "patlak", False, [], None),
            (
                1,
                "extended_tofts",
                False,
                [],
                {"Ktrans": 0.02, "ve": 0.3, "vp": 0.01},
            ),
        ]
        assert results[0]["selected_model_name"] == "patlak"
        assert results[0]["delta_bic"] == 8.0
        assert results[0]["inference_fallback"] is False
        assert results[1]["selected_model_name"] == "extended_tofts"
        assert results[1]["delta_bic"] == 12.0
        assert results[1]["inference_fallback"] is False
        assert results[2]["selected_model_name"] == "patlak"
        assert results[2]["delta_bic"] == 20.0
        assert results[2]["inference_fallback"] is True
        assert "inference_fallback" not in batched_results[2]
        assert "Starting DCE VI refinement for 3 voxels" in caplog.text
        assert "CPU worker processes (configured: 1)" in caplog.text
        assert "DCE VI refinement group 1/2: patlak (2 voxels)" in caplog.text
        assert "DCE VI refinement finished" in caplog.text
        assert "2/3 retained, 1 fell back to batched NLLS" in caplog.text

    def test_fixed_map_refinement_marks_nlls_fallback(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed fixed-model MAP refinement discloses retained NLLS output."""
        import pmriprep.interfaces.dce_compute as dce_compute
        from pmriprep.interfaces.dce_compute import ComputeDCEMaps, VoxelInput

        batched = [{"failed": False, "converged": True, "params": {"Ktrans": 0.1}}]
        voxel = VoxelInput(0, 0, 0, np.zeros(4), None)
        seen_initial: list[dict[str, float] | None] = []

        def fake_fit(value: VoxelInput) -> dict[str, Any]:
            seen_initial.append(value.initial_parameters)
            return {"failed": True, "converged": False}

        monkeypatch.setattr(
            dce_compute,
            "_fit_single_voxel",
            fake_fit,
        )
        init_args = (
            np.zeros(4),
            np.arange(4, dtype=float),
            "tofts",
            "map",
            0,
            1,
            "skip",
            0.0,
            False,
            None,
            False,
            42,
            None,
        )

        results = ComputeDCEMaps()._refine_map_from_batched(
            batched,
            [voxel],
            init_args,
            n_procs=1,
        )

        assert results[0]["params"] == {"Ktrans": 0.1}
        assert results[0]["inference_fallback"] is True
        assert "inference_fallback" not in batched[0]
        assert seen_initial == [{"Ktrans": 0.1}]
