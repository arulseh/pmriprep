"""Tests for BIC scores, automatic model-selection decisions and PK priors."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import (
    ComputeDCEMaps,
    _candidate_bic_for_selection,
    _compute_auto_bic,
    _select_auto_bic_winners,
)
from pmriprep.models.dce import (
    ExtendedToftsModel,
    PatlakModel,
    ToftsModel,
)

_N_PARAMETERS = {
    "patlak": 2,
    "extended_tofts": 3,
    "2cxm": 4,
}


@pytest.fixture
def time_points() -> np.ndarray:
    """Time vector: 60 frames at 7.6 s spacing (typical DCE)."""
    return np.arange(60, dtype=np.float64) * (7.6 / 60.0)  # minutes


@pytest.fixture
def aif(time_points: np.ndarray) -> np.ndarray:
    """Simple population AIF: gamma-variate-like."""
    t = time_points * 60.0  # seconds
    t_peak = 30.0  # s
    aif_curve = np.zeros_like(t)
    post_bolus = t > 10.0
    t_shifted = t[post_bolus] - 10.0
    aif_curve[post_bolus] = (
        5.0 * (t_shifted / t_peak) * np.exp(1.0 - t_shifted / t_peak)
    )
    return aif_curve


class TestBICPerfectFit:
    """BIC must produce lowest value when RSS=0 (perfect fit)."""

    def test_bic_perfect_fit_lower_than_nonzero_rss(self) -> None:
        """RSS=0 should produce lower BIC than RSS>0 for same model."""
        model = PatlakModel()
        n_obs = 60

        bic_perfect = model.compute_bic(0.0, n_obs)
        bic_nonzero = model.compute_bic(1e-6, n_obs)

        assert np.isfinite(bic_perfect), "BIC for RSS=0 must be finite"
        assert bic_perfect < bic_nonzero, (
            f"BIC for RSS=0 ({bic_perfect:.1f}) must be lower than "
            f"BIC for RSS=1e-6 ({bic_nonzero:.1f})"
        )

    def test_aic_perfect_fit_lower_than_nonzero_rss(self) -> None:
        """RSS=0 should produce lower AIC than RSS>0 for same model."""
        model = ExtendedToftsModel()
        n_obs = 60

        aic_perfect = model.compute_aic(0.0, n_obs)
        aic_nonzero = model.compute_aic(1e-6, n_obs)

        assert np.isfinite(aic_perfect), "AIC for RSS=0 must be finite"
        assert aic_perfect < aic_nonzero

    def test_bic_zero_observations_returns_inf(self) -> None:
        """Zero observations should still return inf."""
        model = PatlakModel()
        assert model.compute_bic(0.0, 0) == float("inf")


class TestBICPenaltyDirection:
    """Models with fewer params should have lower BIC for equal RSS."""

    def test_patlak_lower_bic_equal_rss(self) -> None:
        """With same RSS, Patlak BIC < Extended Tofts BIC (fewer params)."""
        patlak = PatlakModel()
        etofts = ExtendedToftsModel()
        rss = 0.01
        n_obs = 60

        bic_patlak = patlak.compute_bic(rss, n_obs)
        bic_etofts = etofts.compute_bic(rss, n_obs)

        assert bic_patlak < bic_etofts, (
            f"Patlak BIC ({bic_patlak:.2f}) should be less than "
            f"Extended Tofts BIC ({bic_etofts:.2f}) for equal RSS"
        )

    def test_tofts_lower_bic_than_etofts_equal_rss(self) -> None:
        """Tofts (k=2) should beat Extended Tofts (k=3) for equal RSS."""
        tofts = ToftsModel()
        etofts = ExtendedToftsModel()
        rss = 0.01
        n_obs = 60

        bic_tofts = tofts.compute_bic(rss, n_obs)
        bic_etofts = etofts.compute_bic(rss, n_obs)

        assert bic_tofts < bic_etofts

    def test_etofts_can_win_with_lower_rss(self) -> None:
        """Extended Tofts should win when its RSS is sufficiently lower."""
        patlak = PatlakModel()
        etofts = ExtendedToftsModel()
        n_obs = 60

        bic_patlak = patlak.compute_bic(0.1, n_obs)
        bic_etofts = etofts.compute_bic(0.001, n_obs)

        assert bic_etofts < bic_patlak


class TestBICSelectsPatlakLowKtrans:
    """For low-Ktrans tissue (NAWM), Patlak should win via BIC."""

    def test_bic_selects_patlak_for_nawm(
        self, time_points: np.ndarray, aif: np.ndarray
    ) -> None:
        """Synthetic voxel with Ktrans=0.005: Patlak should win."""
        patlak = PatlakModel()
        true_ktrans = 0.005  # min⁻¹ (NAWM range)
        true_vp = 0.02
        tissue_curve = patlak.evaluate(
            [true_ktrans, true_vp], aif, time_points
        )
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.001, size=tissue_curve.shape)
        tissue_curve = np.maximum(tissue_curve + noise, 0.0)

        etofts = ExtendedToftsModel()
        patlak_result = patlak.fit(tissue_curve, aif, time_points)
        etofts_result = etofts.fit(tissue_curve, aif, time_points)

        assert patlak_result.bic < etofts_result.bic, (
            f"Patlak BIC ({patlak_result.bic:.2f}) should be lower than "
            f"Extended Tofts BIC ({etofts_result.bic:.2f}) for Ktrans=0.005"
        )

    def test_bic_selects_patlak_for_near_zero_ktrans(
        self, time_points: np.ndarray, aif: np.ndarray
    ) -> None:
        """Synthetic voxel with Ktrans≈0: Patlak should definitely win."""
        patlak = PatlakModel()
        tissue_curve = patlak.evaluate([0.001, 0.01], aif, time_points)
        rng = np.random.default_rng(123)
        noise = rng.normal(0, 0.0005, size=tissue_curve.shape)
        tissue_curve = np.maximum(tissue_curve + noise, 0.0)

        etofts = ExtendedToftsModel()
        patlak_result = patlak.fit(tissue_curve, aif, time_points)
        etofts_result = etofts.fit(tissue_curve, aif, time_points)

        assert patlak_result.bic < etofts_result.bic


class TestBICSelectsEToftsHighKtrans:
    """For high-Ktrans tissue (tumor), Extended Tofts should win."""

    def test_bic_selects_etofts_for_enhancing_tumor(
        self, time_points: np.ndarray, aif: np.ndarray
    ) -> None:
        """Synthetic voxel with Ktrans=0.1, ve=0.3: ETofts should win."""
        etofts = ExtendedToftsModel()
        true_params = [0.1, 0.3, 0.05]  # Ktrans, ve, vp
        tissue_curve = etofts.evaluate(true_params, aif, time_points)
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 0.005, size=tissue_curve.shape)
        tissue_curve = np.maximum(tissue_curve + noise, 0.0)

        patlak = PatlakModel()
        etofts_result = etofts.fit(tissue_curve, aif, time_points)
        patlak_result = patlak.fit(tissue_curve, aif, time_points)

        assert etofts_result.bic < patlak_result.bic, (
            f"Extended Tofts BIC ({etofts_result.bic:.2f}) should be lower "
            f"than Patlak BIC ({patlak_result.bic:.2f}) for enhancing tumor"
        )

    def test_bic_selects_etofts_for_high_ve(
        self, time_points: np.ndarray, aif: np.ndarray
    ) -> None:
        """High-ve tissue needs Extended Tofts backflux term."""
        etofts = ExtendedToftsModel()
        true_params = [0.2, 0.5, 0.03]  # Ktrans, ve, vp
        tissue_curve = etofts.evaluate(true_params, aif, time_points)
        rng = np.random.default_rng(77)
        noise = rng.normal(0, 0.01, size=tissue_curve.shape)
        tissue_curve = np.maximum(tissue_curve + noise, 0.0)

        patlak = PatlakModel()
        etofts_result = etofts.fit(tissue_curve, aif, time_points)
        patlak_result = patlak.fit(tissue_curve, aif, time_points)

        assert etofts_result.bic < patlak_result.bic


class TestFitResultIsBetterThan:
    """FitResult.is_better_than works for BIC comparison."""

    def test_lower_bic_is_better(
        self, time_points: np.ndarray, aif: np.ndarray
    ) -> None:
        """Patlak result with lower BIC is_better_than Extended Tofts."""
        patlak = PatlakModel()
        tissue_curve = patlak.evaluate([0.003, 0.02], aif, time_points)

        patlak_result = patlak.fit(tissue_curve, aif, time_points)
        etofts = ExtendedToftsModel()
        etofts_result = etofts.fit(tissue_curve, aif, time_points)

        assert patlak_result.is_better_than(etofts_result, "bic")


class TestBuildCandidateSet:
    """Candidate set depends on volume TR."""

    @pytest.mark.parametrize(
        ("volume_tr", "expected"),
        [
            (2.0, ["patlak", "extended_tofts", "2cxm"]),
            (2.0000000000000004, ["patlak", "extended_tofts", "2cxm"]),
            (2.00001, ["patlak", "extended_tofts"]),
            (7.6, ["patlak", "extended_tofts"]),
        ],
    )
    def test_candidate_policy(
        self, volume_tr: float, expected: list[str],
    ) -> None:
        """2CXM uses the 2 s boundary with numerical tolerance only."""
        candidates = ComputeDCEMaps()._build_candidate_set(volume_tr)

        assert candidates == expected


class TestNModelBIC:
    """N-model BIC comparison selects the best model."""


    def test_auto_bic_backflux_data(
        self, time_points: np.ndarray, aif: np.ndarray
    ) -> None:
        """Data with backflux (washout): ETofts should win over Patlak."""
        patlak = PatlakModel()
        etofts = ExtendedToftsModel()
        tofts = ToftsModel()

        # Generate data with clear washout (kep = Ktrans/ve = 0.1/0.3 = 0.33)
        tissue_curve = tofts.evaluate([0.1, 0.3], aif, time_points)
        rng = np.random.default_rng(55)
        tissue_curve += rng.normal(0, 0.005, size=tissue_curve.shape)
        tissue_curve = np.maximum(tissue_curve, 0.0)

        results = [
            patlak.fit(tissue_curve, aif, time_points),
            etofts.fit(tissue_curve, aif, time_points),
        ]
        assert all(result.converged for result in results)
        names = [result.model_name for result in results]
        winners, _, valid = _select_auto_bic_winners(
            names, {r.model_name: np.array([r.bic]) for r in results}, _N_PARAMETERS,
        )
        assert valid[0]
        assert names[winners[0]] == "extended_tofts"


class TestAutoBICDecisionSeam:
    """The production auto-mode ranking contract is independently testable."""

    def test_upgrade_evidence_is_measured_against_simplest(self) -> None:
        """Close complex models do not force a fallback to a poor Patlak fit."""
        names = ["patlak", "extended_tofts", "2cxm"]
        bics = {
            "patlak": np.array([0.0]),
            "extended_tofts": np.array([-20.0]),
            "2cxm": np.array([-22.0]),
        }

        winner, delta, valid = _select_auto_bic_winners(
            names, bics, _N_PARAMETERS
        )

        assert valid[0]
        assert names[winner[0]] == "2cxm"
        assert delta[0] == pytest.approx(2.0)

    def test_weak_improvement_downgrades_to_simplest(self) -> None:
        names = ["patlak", "extended_tofts"]
        bics = {
            "patlak": np.array([0.0]),
            "extended_tofts": np.array([-4.0]),
        }

        winner, delta, valid = _select_auto_bic_winners(
            names, bics, _N_PARAMETERS
        )

        assert valid[0]
        assert names[winner[0]] == "patlak"
        assert np.isnan(delta[0])

    def test_failed_simplest_does_not_block_complex_selection(self) -> None:
        names = ["patlak", "extended_tofts", "2cxm"]
        bics = {
            "patlak": np.array([np.inf]),
            "extended_tofts": np.array([-10.0]),
            "2cxm": np.array([-12.0]),
        }

        winner, delta, valid = _select_auto_bic_winners(
            names, bics, _N_PARAMETERS
        )

        assert valid[0]
        assert names[winner[0]] == "2cxm"
        assert delta[0] == pytest.approx(2.0)

    def test_nonconverged_candidate_bic_is_excluded(self) -> None:
        candidate = SimpleNamespace(bic=-100.0, converged=False)

        assert np.isinf(_candidate_bic_for_selection(candidate))

    def test_all_failed_candidates_are_invalid(self) -> None:
        names = ["patlak", "extended_tofts", "2cxm"]
        bics = {name: np.array([np.inf]) for name in names}

        _, delta, valid = _select_auto_bic_winners(
            names, bics, _N_PARAMETERS
        )

        assert not valid[0]
        assert np.isnan(delta[0])

    def test_motion_weighted_bic_uses_fitted_likelihood(self) -> None:
        """A downweighted outlier cannot re-enter through unweighted BIC."""
        tissue = np.zeros((1, 20), dtype=np.float64)
        patlak_fit = np.zeros_like(tissue)
        patlak_fit[0, 5] = 10.0
        et_fit = np.full_like(tissue, 0.2)
        converged = np.ones(1, dtype=bool)
        weights = np.ones(20, dtype=np.float64)
        weights[5] = 1e-4

        weighted_patlak = _compute_auto_bic(
            tissue, patlak_fit, converged, 2, weights
        )
        weighted_et = _compute_auto_bic(
            tissue, et_fit, converged, 3, weights
        )
        unweighted_patlak = _compute_auto_bic(
            tissue, patlak_fit, converged, 2
        )
        unweighted_et = _compute_auto_bic(
            tissue, et_fit, converged, 3
        )

        assert weighted_patlak[0] < weighted_et[0]
        assert unweighted_et[0] < unweighted_patlak[0]


class TestPatlakKtransPrior:
    """Patlak Ktrans prior should match its validity domain (no backflux)."""

    def test_patlak_ktrans_prior_below_01(self) -> None:
        """Patlak Ktrans prior mean should be below 0.1 (no-backflux domain)."""
        patlak = PatlakModel()
        ktrans_param = patlak.get_parameter_definitions()[0]
        assert ktrans_param.get_prior_mean() < 0.1

    def test_patlak_ktrans_prior_range(self) -> None:
        """Patlak Ktrans typical_range should be (0.001, 0.1)."""
        patlak = PatlakModel()
        ktrans_param = patlak.get_parameter_definitions()[0]
        assert ktrans_param.typical_range == (0.001, 0.1)

    def test_tofts_ktrans_prior_range(self) -> None:
        """Tofts Ktrans prior typical range is (0.001, 0.2)."""
        tofts = ToftsModel()
        ktrans_param = tofts.get_parameter_definitions()[0]
        assert ktrans_param.typical_range == (0.001, 0.2)

    def test_etofts_ktrans_prior_range(self) -> None:
        """Extended Tofts Ktrans prior typical range is (0.001, 0.2)."""
        etofts = ExtendedToftsModel()
        ktrans_param = etofts.get_parameter_definitions()[0]
        assert ktrans_param.typical_range == (0.001, 0.2)
