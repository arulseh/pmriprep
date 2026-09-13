"""Regression tests for plasma-fraction support and boundary retention."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear

from pmriprep.interfaces.dce_compute import _is_degenerate_dce_fit
from pmriprep.models.dce.batched_fitting import batched_lm_fit, batched_patlak_lstsq
from pmriprep.models.dce.bayesian import BayesianPKFitter, create_vp_prior_from_dsc
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.patlak import PatlakModel
from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel
from pmriprep.utils.dce_constants import VP_BOUNDS, VP_VE_TOTAL_MAX


@pytest.fixture
def sampling() -> tuple[np.ndarray, np.ndarray]:
    """Return a concentration AIF and sampling times in minutes."""
    time = np.linspace(0.0, 5.0, 101)
    aif = 5 * np.exp(-0.5 * ((time - 0.5) / 0.15) ** 2)
    aif += np.where(time >= 0.5, np.exp(-(time - 0.5) / 2), 0)
    return time, aif


@pytest.mark.parametrize("vp", [0.0, 0.0005, 0.3, 0.5, 0.999999, 1.0])
def test_patlak_recovers_full_fraction_domain(
    sampling: tuple[np.ndarray, np.ndarray],
    vp: float,
) -> None:
    """Independent linear recovery agrees with both production NLLS solvers."""
    time, aif = sampling
    design = np.column_stack((cumulative_trapezoid(aif, time, initial=0), aif))
    tissue = design @ [0.05, vp]
    reference = lsq_linear(design, tissue, bounds=([0, 0], [5, 1]), tol=1e-12)
    scalar = PatlakModel().fit(tissue, aif, time)
    batch = batched_patlak_lstsq(tissue[None, :], aif, time)
    assert scalar.converged
    for params in (
        scalar.parameters,
        {"Ktrans": batch.params["ktrans"][0], "vp": batch.params["vp"][0]},
    ):
        np.testing.assert_allclose(
            [params["Ktrans"], params["vp"]], reference.x, atol=1e-5, rtol=1e-4
        )
        assert not _is_degenerate_dce_fit("patlak", params)


@pytest.mark.parametrize("vp", [0.0005, 0.5])
@pytest.mark.parametrize("method", ["vi", "map"])
def test_bayesian_vp_interval_contains_estimate(
    sampling: tuple[np.ndarray, np.ndarray],
    vp: float,
    method: str,
) -> None:
    """Plasma-fraction posteriors have ordered intervals across the domain."""
    time, aif = sampling
    tissue = PatlakModel().evaluate([0.05, vp], aif, time)
    fitter = BayesianPKFitter(PatlakModel())
    fit = fitter.fit_vi if method == "vi" else fitter.fit_map
    result = fit(tissue, aif, time)
    assert result.converged
    if vp < 0.001:
        # MAP includes a likelihood noise floor and need not equal noiseless truth.
        assert 0 < result.parameters["vp"] < 0.001
    else:
        assert result.parameters["vp"] == pytest.approx(vp, rel=0.02)
    if method == "vi":
        assert result.parameters["vp"] == pytest.approx(vp, abs=5e-5, rel=0.02)
    lo, hi = result.credible_intervals["vp"]
    assert 0 <= lo <= result.parameters["vp"] <= hi <= 1


def test_priors_share_fraction_support() -> None:
    """Default and DSC-informed priors use the same plasma-fraction domain."""
    assert VP_BOUNDS == (0.0, 1.0)
    for model in (PatlakModel(), ExtendedToftsModel(), TwoCompartmentExchangeModel()):
        fitter = BayesianPKFitter(model)
        prior = fitter.priors[model.get_parameter_names().index("vp")]
        assert prior.upper_bound == 1.0
    prior = create_vp_prior_from_dsc(0.5, relative_uncertainty=0.6)
    assert prior.upper_bound == 1.0
    assert prior.mean == 0.5
    assert prior.std == 0.3


@pytest.mark.parametrize("vp", [0.0, 0.3, 0.5, 1.0])
def test_patlak_retains_fraction_endpoints(vp: float) -> None:
    """Contact with the plasma-fraction boundary is not itself degeneracy."""
    assert not _is_degenerate_dce_fit("patlak", {"Ktrans": 0.1, "vp": vp})


@pytest.mark.parametrize("vp", [-1e-6, 1.000001, np.nan, np.inf])
def test_invalid_fractions_are_rejected(vp: float) -> None:
    """Nonfinite and out-of-domain fractions cannot enter parameter maps."""
    for name in ("patlak", "extended_tofts", "2cxm"):
        assert _is_degenerate_dce_fit(
            name, {"Ktrans": 0.1, "ve": 0.2, "Fp": 40, "PS": 10, "vp": vp}
        )


@pytest.mark.parametrize(
    "name,truth",
    [
        ("extended_tofts", [0.1, 0.2, 0.5]),
        ("2cxm", [40.0, 10.0, 0.2, 0.5]),
    ],
)
def test_exchange_models_recover_high_vp(
    sampling: tuple[np.ndarray, np.ndarray],
    name: str,
    truth: list[float],
) -> None:
    """Both exchange-model solvers recover plasma fractions above 0.3."""
    time, aif = sampling
    model = (
        ExtendedToftsModel()
        if name == "extended_tofts"
        else TwoCompartmentExchangeModel()
    )
    tissue = model.evaluate(truth, aif, time)
    scalar = model.fit(tissue, aif, time)
    batch = batched_lm_fit(name, tissue[None, :], aif, time, max_iter=200)
    assert scalar.converged
    assert scalar.parameters["vp"] == pytest.approx(0.5, rel=0.02)
    assert batch.params["vp"][0] == pytest.approx(0.5, rel=0.02)
    assert not _is_degenerate_dce_fit(name, scalar.parameters)
    assert scalar.parameters["ve"] + scalar.parameters["vp"] <= VP_VE_TOTAL_MAX


def test_joint_volume_still_rejects_impossible_fits() -> None:
    """Broad individual fractions do not permit impossible tissue occupancy."""
    for name in ("extended_tofts", "2cxm"):
        assert _is_degenerate_dce_fit(
            name, {"Ktrans": 0.1, "ve": 0.3, "vp": 0.8, "Fp": 40, "PS": 10}
        )


@pytest.mark.parametrize("name,vp", [
    ("extended_tofts", 1.0),
    ("extended_tofts", VP_VE_TOTAL_MAX),
    ("2cxm", 0.995),
])
def test_fixed_vp_cannot_leave_empty_volume_domain(
    sampling: tuple[np.ndarray, np.ndarray],
    name: str,
    vp: float,
) -> None:
    """Fixed plasma fractions must leave room for the model's EES domain."""
    time, aif = sampling
    model = (
        ExtendedToftsModel()
        if name == "extended_tofts"
        else TwoCompartmentExchangeModel()
    )
    with pytest.raises(ValueError, match="joint tissue-volume"):
        model.fit_constrained(aif, aif, time, fixed_vp=vp)
    with pytest.raises(ValueError, match="joint tissue-volume"):
        batched_lm_fit(name, aif[None, :], aif, time, fixed_vp=np.array([vp]))


def test_2cxm_zero_vp_forward_limit(sampling: tuple[np.ndarray, np.ndarray]) -> None:
    """Zero plasma volume approaches the reduced exchange-model limit."""
    from pmriprep.models.dce.tofts import ToftsModel
    from pmriprep.models.dce.batched_models import batched_2cxm_model

    time, aif = sampling
    fp, ps, ve = 40.0, 10.0, 0.2
    ktrans = fp * ps / (fp + ps) / 100.0
    expected = ToftsModel().evaluate([ktrans, ve], aif, time)
    scalar = TwoCompartmentExchangeModel().evaluate([fp, ps, ve, 0], aif, time)
    batch = batched_2cxm_model(
        np.array([fp]), np.array([ps]), np.array([ve]), np.array([0.0]), aif, time
    )
    np.testing.assert_allclose(scalar, expected, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(batch.tissue_curves[0], scalar, atol=1e-10)


@pytest.mark.parametrize("vp", [0.0005, 0.5, 0.999999])
def test_patlak_initializer_spans_fraction_domain(
    sampling: tuple[np.ndarray, np.ndarray], vp: float,
) -> None:
    """Data-driven initialization preserves low and high plasma fractions."""
    time, aif = sampling
    model = PatlakModel()
    tissue = model.evaluate([0.05, vp], aif, time)
    initial = model.estimate_initial_guess(tissue, aif, time)
    assert initial[1] == pytest.approx(vp, abs=1e-6)
