"""Characterization tests for DCE fit-result scattering decisions."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import (
    ComputeDCEMaps,
    VoxelInput,
    _account_retained_dce_fit,
    _accumulate_mcmc_diagnostics,
    _compute_vp_prior_departure,
    _fraction_fxl_risk,
    _is_degenerate_dce_fit,
    _resolve_dce_fit_quality,
    _initialize_dce_qc_metrics,
    _scatter_degenerate_dce_fit,
    _scatter_dce_fit_artifacts,
    _scatter_dce_fit_results,
    _scatter_dce_parameters,
    _scatter_dce_uncertainties,
    _summarize_dce_categorical_qc,
    _summarize_dce_fitted_parameters,
    _summarize_dce_fit_outcomes,
    _summarize_dce_r2,
    _summarize_map_uncertainty_sources,
    _summarize_inference_fallback,
    _summarize_mcmc_diagnostics,
    _summarize_vp_prior_departures,
)
from pmriprep.models.dce.bayesian import QC_ESS_MIN, QC_RHAT_MAX
from pmriprep.utils.dsc_cbv_constraint import ConstraintMode
from pmriprep.utils.dce_constants import (
    FP_BOUNDS,
    KTRANS_BOUNDS,
    PS_BOUNDS,
    VE_BOUNDS,
    VP_BOUNDS,
)


@pytest.mark.parametrize(
    ("converged", "base_quality", "kep_railed", "expected"),
    [
        (False, "good", False, "poor_convergence"),
        (False, "high_uncertainty", True, "poor_convergence"),
        (True, "good", True, "unreliable_high_kep"),
        (True, "high_uncertainty", True, "high_uncertainty"),
        (True, "good", False, "good"),
    ],
)
def test_resolve_dce_fit_quality_prioritizes_convergence(
    converged: bool,
    base_quality: str,
    kep_railed: bool,
    expected: str,
) -> None:
    """Nonconvergence cannot remain good or be hidden by a secondary flag."""
    assert (
        _resolve_dce_fit_quality(converged, base_quality, kep_railed)
        == expected
    )


@pytest.mark.parametrize(
    ("model", "params", "expected"),
    [
        ("tofts", {"Ktrans": 0.1, "ve": 0.3}, False),
        ("tofts", {"Ktrans": 0.0, "ve": 0.3}, True),
        ("tofts", {"Ktrans": KTRANS_BOUNDS.max, "ve": 0.3}, True),
        ("tofts", {"Ktrans": 0.1, "ve": VE_BOUNDS.max}, True),
        ("patlak", {"Ktrans": 0.0, "vp": 0.03}, False),
        ("patlak", {"Ktrans": -1e-6, "vp": 0.03}, True),
        ("patlak", {"Ktrans": 0.1, "vp": 0.0}, True),
        ("extended_tofts", {"Ktrans": 0.0, "ve": 0.3, "vp": 0.03}, True),
        ("extended_tofts", {"Ktrans": 0.1, "ve": 0.3, "vp": VP_BOUNDS.max}, True),
        ("2cxm", {"Fp": 1.0, "PS": 0.0, "ve": 0.3, "vp": 0.03}, True),
        ("2cxm", {"Fp": 0.0, "PS": 0.1, "ve": 0.3, "vp": 0.03}, True),
        ("2cxm", {"Fp": FP_BOUNDS.max, "PS": 0.1, "ve": 0.3, "vp": 0.03}, True),
        ("2cxm", {"Fp": 1.0, "PS": PS_BOUNDS.max, "ve": 0.3, "vp": 0.03}, True),
        ("2cxm", {"Fp": 1.0, "PS": 0.1, "ve": 0.8, "vp": 0.03}, True),
        ("2cxm", {"Fp": 1.0, "PS": 0.1, "ve": 0.3, "vp": VP_BOUNDS.max}, True),
    ],
)
def test_degeneracy_rules_preserve_model_specific_boundaries(
    model: str,
    params: dict[str, float],
    expected: bool,
) -> None:
    """Exact physiological bounds are rejected independently of prior use."""
    assert _is_degenerate_dce_fit(model, params) is expected


def _uncertainty_maps() -> SimpleNamespace:
    """Build the mutable uncertainty fields consumed by the scatter helper."""
    return SimpleNamespace(
        ktrans_std=np.zeros((1, 1, 1)),
        ve_std=np.zeros((1, 1, 1)),
        vp_std=np.zeros((1, 1, 1)),
        fp_std=np.zeros((1, 1, 1)),
        ps_std=np.zeros((1, 1, 1)),
        sigma_factor={"Ktrans": 2.0, "ve": 1.0, "vp": 3.0, "Fp": 4.0, "PS": 5.0},
        ktrans=np.zeros((1, 1, 1)),
        ve=np.zeros((1, 1, 1)),
        vp=np.zeros((1, 1, 1)),
        kep=np.zeros((1, 1, 1)),
        r_squared=np.zeros((1, 1, 1)),
        fp=np.zeros((1, 1, 1)),
        ps=np.zeros((1, 1, 1)),
        extraction_fraction=np.zeros((1, 1, 1)),
        optimizer_converged=np.zeros((1, 1, 1), dtype=bool),
        model_selection=np.zeros((1, 1, 1), dtype=np.int8),
        model_selection_codes={"tofts": 2},
        delta_bic=np.full((1, 1, 1), np.nan),
        fitted_curves=None,
        quality_flag=np.zeros((1, 1, 1), dtype=np.int8),
        quality_flag_codes={"good": 1},
    )


def test_uncertainty_scatter_applies_model_fields_and_inflation() -> None:
    """2CXM omits derived Ktrans but writes direct-parameter uncertainty."""
    maps = _uncertainty_maps()
    _scatter_dce_uncertainties(
        maps,
        (0, 0, 0),
        "2cxm",
        {"Ktrans": 1.0, "ve": 2.0, "vp": 3.0, "Fp": 4.0, "PS": 5.0},
    )
    assert maps.ktrans_std[0, 0, 0] == 0.0
    assert maps.ve_std[0, 0, 0] == 2.0
    assert maps.vp_std[0, 0, 0] == 9.0
    assert maps.fp_std[0, 0, 0] == 16.0
    assert maps.ps_std[0, 0, 0] == 25.0


def test_uncertainty_scatter_leaves_nonapplicable_patlak_fields_zero() -> None:
    """Patlak writes Ktrans/vp uncertainty but not ve, Fp, or PS."""
    maps = _uncertainty_maps()
    _scatter_dce_uncertainties(
        maps,
        (0, 0, 0),
        "patlak",
        {"Ktrans": 1.0, "ve": 2.0, "vp": 3.0, "Fp": 4.0, "PS": 5.0},
    )
    assert maps.ktrans_std[0, 0, 0] == 2.0
    assert maps.vp_std[0, 0, 0] == 9.0
    assert maps.ve_std[0, 0, 0] == 0.0
    assert maps.fp_std[0, 0, 0] == 0.0
    assert maps.ps_std[0, 0, 0] == 0.0


def test_fit_artifact_scatter_writes_curve_and_quality() -> None:
    """All non-parameter artifacts use the same voxel coordinate."""
    maps = SimpleNamespace(
        fitted_curves=np.full((1, 1, 1, 3), np.nan, dtype=np.float32),
        quality_flag=np.zeros((1, 1, 1), dtype=np.int8),
        quality_flag_codes={"good": 1, "high_uncertainty": 3},
    )
    _scatter_dce_fit_artifacts(
        maps,
        (0, 0, 0),
        {
            "fitted_curve": np.array([0.0, 0.2, 0.1]),
            "quality_flag": "high_uncertainty",
        },
    )
    np.testing.assert_array_equal(
        maps.fitted_curves[0, 0, 0],
        np.array([0.0, 0.2, 0.1], dtype=np.float32),
    )
    assert maps.quality_flag[0, 0, 0] == 3


def test_fit_artifact_scatter_defaults_without_optional_curve() -> None:
    """Missing optional values retain no curve and default to a good label."""
    maps = SimpleNamespace(
        fitted_curves=None,
        quality_flag=np.zeros((1, 1, 1), dtype=np.int8),
        quality_flag_codes={"good": 1},
    )
    _scatter_dce_fit_artifacts(maps, (0, 0, 0), {})
    assert maps.quality_flag[0, 0, 0] == 1


def test_parameter_scatter_respects_patlak_applicability() -> None:
    """Patlak writes Ktrans/vp/R² while leaving ve/kep and 2CXM maps untouched."""
    maps = _uncertainty_maps()
    _scatter_dce_parameters(
        maps,
        (0, 0, 0),
        "patlak",
        {"Ktrans": 0.1, "ve": 0.3, "vp": 0.04, "kep": 0.2},
        0.8,
    )
    assert maps.ktrans[0, 0, 0] == 0.1
    assert maps.vp[0, 0, 0] == 0.04
    assert maps.r_squared[0, 0, 0] == 0.8
    assert maps.ve[0, 0, 0] == 0.0
    assert maps.kep[0, 0, 0] == 0.0
    assert maps.fp[0, 0, 0] == 0.0


def test_parameter_scatter_writes_all_2cxm_fields() -> None:
    """2CXM writes every shared and model-specific parameter."""
    maps = _uncertainty_maps()
    params = {
        "Ktrans": 0.1,
        "ve": 0.3,
        "vp": 0.04,
        "kep": 0.2,
        "Fp": 20.0,
        "PS": 5.0,
        "E": 0.2,
    }
    _scatter_dce_parameters(maps, (0, 0, 0), "2cxm", params, 0.9)
    assert maps.ktrans[0, 0, 0] == 0.1
    assert maps.ve[0, 0, 0] == 0.3
    assert maps.vp[0, 0, 0] == 0.04
    assert maps.kep[0, 0, 0] == 0.2
    assert maps.fp[0, 0, 0] == 20.0
    assert maps.ps[0, 0, 0] == 5.0
    assert maps.extraction_fraction[0, 0, 0] == 0.2


def test_degenerate_scatter_uses_auto_nan_for_applicable_patlak_maps() -> None:
    """Auto Patlak rejection marks Ktrans/vp non-applicable without touching ve."""
    maps = _uncertainty_maps()
    maps.ktrans[:] = 1.0
    maps.ve[:] = 2.0
    maps.vp[:] = 3.0
    _scatter_degenerate_dce_fit(maps, (0, 0, 0), "patlak", -0.2)
    assert np.isnan(maps.ktrans[0, 0, 0])
    assert np.isnan(maps.vp[0, 0, 0])
    assert maps.ve[0, 0, 0] == 2.0
    assert maps.r_squared[0, 0, 0] == -0.2


def test_degenerate_scatter_uses_fixed_nan_for_tofts_maps() -> None:
    """Fixed Tofts rejection marks applicable parameters as missing."""
    maps = _uncertainty_maps()
    maps.ktrans[:] = 1.0
    maps.ve[:] = 2.0
    maps.kep[:] = 3.0
    maps.vp[:] = 4.0
    _scatter_degenerate_dce_fit(maps, (0, 0, 0), "tofts", 0.1)
    assert np.isnan(maps.ktrans[0, 0, 0])
    assert np.isnan(maps.ve[0, 0, 0])
    assert np.isnan(maps.kep[0, 0, 0])
    assert maps.vp[0, 0, 0] == 4.0
    assert maps.r_squared[0, 0, 0] == 0.1


def test_retained_fixed_fit_preserves_genuine_zero_parameter() -> None:
    """A retained numerical zero remains distinct from an unfitted sentinel."""
    maps = _uncertainty_maps()

    _scatter_dce_parameters(
        maps,
        (0, 0, 0),
        "patlak",
        {"Ktrans": 0.0, "vp": 0.02},
        0.8,
    )

    assert maps.ktrans[0, 0, 0] == 0.0
    assert np.isfinite(maps.ktrans[0, 0, 0])
    assert maps.vp[0, 0, 0] == pytest.approx(0.02)


def test_vp_prior_departure_uses_relative_sigma_and_floor() -> None:
    """Bayesian SOFT departure uses relative uncertainty with a 0.001 floor."""
    assert _compute_vp_prior_departure(
        ConstraintMode.SOFT, "map", "extended_tofts", 0.04, 0.06, 0.25,
    ) == pytest.approx(2.0)
    assert _compute_vp_prior_departure(
        ConstraintMode.SOFT, "vi", "patlak", 0.001, 0.003, 0.2,
    ) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("mode", "inference", "model", "prior"),
    [
        (ConstraintMode.SKIP, "map", "extended_tofts", 0.04),
        (ConstraintMode.SOFT, "nlls", "extended_tofts", 0.04),
        (ConstraintMode.SOFT, "map", "tofts", 0.04),
        (ConstraintMode.SOFT, "map", "extended_tofts", None),
        (ConstraintMode.SOFT, "map", "extended_tofts", 0.0),
    ],
)
def test_vp_prior_departure_skips_inapplicable_paths(
    mode: ConstraintMode,
    inference: str,
    model: str,
    prior: float | None,
) -> None:
    """Departure QC exists only for positive SOFT priors in Bayesian vp models."""
    assert _compute_vp_prior_departure(
        mode, inference, model, prior, 0.06, 0.25,
    ) is None


def test_accumulate_mcmc_diagnostics_groups_parameters_and_counts_voxel() -> None:
    """One non-empty R-hat record counts once and groups R-hat/ESS by parameter."""
    rhat: dict[str, list[float]] = {"Ktrans": [1.01]}
    ess: dict[str, list[float]] = {}
    increment = _accumulate_mcmc_diagnostics(
        {
            "rhat": {"Ktrans": 1.02, "vp": 1.03},
            "ess": {"Ktrans": 250.0, "vp": 180.0},
        },
        rhat,
        ess,
    )
    assert increment == 1
    assert rhat == {"Ktrans": [1.01, 1.02], "vp": [1.03]}
    assert ess == {"Ktrans": [250.0], "vp": [180.0]}


def test_accumulate_mcmc_diagnostics_retains_ess_without_rhat() -> None:
    """Ensemble ESS identifies MCMC output when R-hat is unavailable."""
    rhat: dict[str, list[float]] = {}
    ess: dict[str, list[float]] = {}
    increment = _accumulate_mcmc_diagnostics(
        {"rhat": {}, "ess": {"Ktrans": 250.0}}, rhat, ess,
    )
    assert increment == 1
    assert rhat == {}
    assert ess == {"Ktrans": [250.0]}


def test_summarize_map_uncertainty_sources_counts_actual_outcomes() -> None:
    """MAP QC separates Hessian covariance, unavailable spread, and no estimate."""
    summary = _summarize_map_uncertainty_sources(
        [
            "laplace_hessian",
            "hessian_unavailable",
            "hessian_unavailable",
            "not_computed",
        ]
    )

    assert summary == {
        "n_retained_voxels": 4,
        "n_evaluated": 3,
        "n_laplace_hessian": 1,
        "n_hessian_unavailable": 2,
        "n_not_computed": 1,
        "hessian_unavailable_fraction": pytest.approx(2 / 3),
    }


def test_summarize_inference_fallback_names_denominator() -> None:
    """Bayesian fallback QC reports its engines, count, and retained fraction."""
    assert _summarize_inference_fallback("nlls", 8, 0) is None
    assert _summarize_inference_fallback("vi", 8, 2) == {
        "requested_inference": "vi",
        "fallback_inference": "nlls",
        "n_retained_voxels": 8,
        "n_nlls_fallback": 2,
        "fallback_fraction_of_retained": 0.25,
    }


def test_retained_fit_accounting_tracks_constrained_convergence() -> None:
    """A converged constrained fit contributes constrained count and R²."""
    voxel = VoxelInput(0, 0, 0, np.zeros(3), 0.04)
    update = _account_retained_dce_fit(
        voxel,
        {"constrained": True, "converged": True, "r_squared": 0.8},
        ConstraintMode.SOFT,
    )
    assert update.n_constrained == 1
    assert update.n_unconstrained_fallback == 0
    assert update.r_squared == 0.8


def test_retained_fit_accounting_tracks_unconstrained_failure_fallback() -> None:
    """A failed retained fit with an available prior records fallback and no R²."""
    voxel = VoxelInput(0, 0, 0, np.zeros(3), 0.04)
    update = _account_retained_dce_fit(
        voxel,
        {"constrained": False, "converged": False, "r_squared": -0.2},
        ConstraintMode.SOFT,
    )
    assert update.n_constrained == 0
    assert update.n_unconstrained_fallback == 1
    assert update.r_squared is None


def test_retained_fit_accounting_skip_mode_never_marks_fallback() -> None:
    """A voxel prior is not a fallback source when the constraint mode is SKIP."""
    voxel = VoxelInput(0, 0, 0, np.zeros(3), 0.04)
    update = _account_retained_dce_fit(
        voxel,
        {"constrained": False, "converged": True, "r_squared": 0.5},
        ConstraintMode.SKIP,
    )
    assert update.n_unconstrained_fallback == 0


def test_summarize_mcmc_diagnostics_preserves_threshold_fractions() -> None:
    """MCMC summary uses inclusive R-hat and strict ESS failure thresholds."""
    summary = _summarize_mcmc_diagnostics(
        2,
        {"Ktrans": [QC_RHAT_MAX - 0.1, QC_RHAT_MAX]},
        {"Ktrans": [QC_ESS_MIN - 1.0, QC_ESS_MIN]},
    )
    assert summary is not None
    assert summary["n_mcmc_voxels"] == 2
    assert summary["rhat_median_Ktrans"] == pytest.approx(QC_RHAT_MAX - 0.05)
    assert summary["rhat_frac_poor_Ktrans"] == 0.5
    assert summary["ess_median_Ktrans"] == pytest.approx(QC_ESS_MIN - 0.5)
    assert summary["ess_frac_low_Ktrans"] == 0.5


def test_summarize_mcmc_diagnostics_returns_none_without_voxels() -> None:
    """No MCMC voxels omit the diagnostics block entirely."""
    assert _summarize_mcmc_diagnostics(0, {}, {}) is None


def test_summarize_vp_prior_departures_preserves_strict_categories() -> None:
    """Prior-dominated and strong-departure fractions use strict inequalities."""
    summary = _summarize_vp_prior_departures([0.4, 0.5, 2.0, 3.0])
    assert summary["z_departure_median"] == 1.25
    assert summary["z_departure_p90"] == pytest.approx(2.7)
    assert summary["frac_prior_dominated"] == 0.25
    assert summary["frac_strong_departure"] == 0.25


def test_summarize_vp_prior_departures_empty_input_omits_metrics() -> None:
    """No applicable SOFT-prior voxels add no departure QC keys."""
    assert _summarize_vp_prior_departures([]) == {}


def test_summarize_dce_categorical_qc_counts_models_and_quality() -> None:
    """Auto categorical QC reports candidate fractions and in-domain labels."""
    model_selection = np.array([[[1, 3], [3, 0]]])
    quality = np.array([[[1, 2], [3, 4]]])
    fitting_mask = np.array([[[True, True], [True, False]]])
    summary = _summarize_dce_categorical_qc(
        True,
        ["patlak", "extended_tofts"],
        model_selection,
        {"patlak": 1, "extended_tofts": 3},
        quality,
        fitting_mask,
    )
    assert summary["model_selection"] == "auto"
    assert summary["n_patlak_selected"] == 1
    assert summary["patlak_fraction"] == pytest.approx(1 / 3)
    assert summary["n_extended_tofts_selected"] == 2
    assert summary["extended_tofts_fraction"] == pytest.approx(2 / 3)
    assert summary["quality_n_good"] == 1
    assert summary["quality_n_poor_convergence"] == 1
    assert summary["quality_n_high_uncertainty"] == 1
    assert summary["quality_n_unreliable_high_kep"] == 0


def test_summarize_dce_categorical_qc_fixed_mode_omits_candidates() -> None:
    """Fixed-model QC omits automatic candidate counts and fractions."""
    summary = _summarize_dce_categorical_qc(
        False,
        [],
        np.zeros((1, 1, 1), dtype=np.int8),
        {},
        np.ones((1, 1, 1), dtype=np.int8),
        np.ones((1, 1, 1), dtype=bool),
    )
    assert summary["model_selection"] == "fixed"
    assert "candidate_models" not in summary
    assert summary["quality_n_good"] == 1


def test_summarize_dce_fitted_parameters_preserves_applicability() -> None:
    """Fitted summaries retain finite and positive parameter rules."""
    fitting_mask = np.ones((1, 2, 2), dtype=bool)
    summary = _summarize_dce_fitted_parameters(
        fitting_mask,
        np.array([[[0.1, 0.3], [0.0, np.nan]]]),
        np.array([[[0.2, np.nan], [0.8, 0.9]]]),
        np.array([[[0.04, 0.06], [0.08, 0.1]]]),
        np.array([[[20.0, -1.0], [30.0, 40.0]]]),
        np.array([[[2.0, np.nan], [3.0, 4.0]]]),
        True,
        True,
        True,
    )
    assert summary["ktrans_mean"] == pytest.approx(0.2)
    assert summary["ktrans_median"] == pytest.approx(0.2)
    assert sum(summary["ktrans_histogram"]) == 2
    assert summary["ve_mean"] == pytest.approx(0.2)
    assert sum(summary["ve_histogram"]) == 1
    assert summary["vp_mean"] == pytest.approx(0.05)
    assert summary["fp_median"] == pytest.approx(20.0)
    assert summary["ps_median"] == pytest.approx(2.0)


def test_summarize_dce_fitted_parameters_empty_fit_omits_metrics() -> None:
    """No positive finite Ktrans values produce no fitted-map metrics."""
    shape = (1, 1, 2)
    empty = np.zeros(shape)
    assert _summarize_dce_fitted_parameters(
        np.ones(shape, dtype=bool),
        np.array([[[0.0, np.nan]]]),
        empty,
        empty,
        empty,
        empty,
        True,
        True,
        True,
    ) == {}


def test_fraction_fxl_risk_uses_peak_fitted_concentration() -> None:
    """FXL prevalence uses fitted voxels and a strict 0.8 mM threshold."""
    concentration = np.array(
        [[[[0.1, 0.8, 0.2], [0.1, 0.9, 0.3], [0.2, 1.2, 0.4]]]]
    )
    fitting_mask = np.array([[[True, True, False]]])
    assert _fraction_fxl_risk(concentration, fitting_mask) == pytest.approx(0.5)


def test_fraction_fxl_risk_empty_finite_domain_is_zero() -> None:
    """An empty fitting domain retains the serialized zero prevalence."""
    concentration = np.zeros((1, 1, 1, 2))
    assert _fraction_fxl_risk(
        concentration,
        np.zeros((1, 1, 1), dtype=bool),
    ) == 0.0


def test_summarize_dce_r2_preserves_population_statistics() -> None:
    """R-squared summaries retain mean, population SD, and median."""
    assert _summarize_dce_r2([0.2, 0.5, 0.8]) == pytest.approx(
        {"r2_mean": 0.5, "r2_std": np.std([0.2, 0.5, 0.8]), "r2_median": 0.5}
    )
    assert _summarize_dce_r2([]) == {}


def test_summarize_dce_fit_outcomes_preserves_counters() -> None:
    """Fit outcome assembly exposes named stages and explicit denominators."""
    reasons = {"optimizer_failed": 2}
    summary = _summarize_dce_fit_outcomes(
        10, 20, 7, 1, 1, 1, 6, 3, 1, 1, reasons,
    )
    accounting = summary["voxel_accounting"]
    assert accounting["fitting_eligible"] == {
        "count": 10,
        "fraction_of_brain": 0.5,
    }
    assert accounting["optimizer_converged"]["count"] == 7
    assert accounting["prefit_rejected"]["count"] == 1
    assert accounting["execution_exception"]["count"] == 1
    assert accounting["fit_unsuccessful"]["count"] == 1
    assert "execution_failed" not in accounting
    assert accounting["postfit_degenerate"]["count"] == 1
    assert accounting["retained_nondegenerate"]["count"] == 6
    assert (
        accounting["prefit_rejected"]["count"]
        + accounting["execution_exception"]["count"]
        + accounting["fit_unsuccessful"]["count"]
        + accounting["postfit_degenerate"]["count"]
        + accounting["retained_nondegenerate"]["count"]
        == accounting["attempted"]["count"]
    )
    assert summary["failure_reasons"] is reasons


def test_summarize_dce_fit_outcomes_zero_voxels_has_zero_rate() -> None:
    """An empty fit domain retains guarded zero stage fractions."""
    summary = _summarize_dce_fit_outcomes(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}
    )
    accounting = summary["voxel_accounting"]
    assert accounting["optimizer_converged"]["fraction_of_attempted"] == 0.0


def test_scatter_dce_fit_results_counts_failures_before_map_access() -> None:
    """Failed fits preserve reason counts without mutating output maps."""
    voxels = [VoxelInput(0, 0, 0, np.zeros(3), None) for _ in range(3)]
    results = [
        {
            "failed": True,
            "failure_reason": "failed_no_enhancement",
            "failure_stage": "prefit",
        },
        {
            "failed": True,
            "failure_reason": "exception",
            "failure_stage": "execution",
        },
        {"failed": True},
    ]
    summary = _scatter_dce_fit_results(
        voxels,
        results,
        SimpleNamespace(),
        "tofts",
        False,
        "nlls",
        ConstraintMode.SKIP,
        0.0,
    )
    assert summary.n_prefit_rejected == 1
    assert summary.n_execution_exceptions == 1
    assert summary.n_fit_unsuccessful == 1
    assert summary.failure_reasons == {
        "failed_no_enhancement": 1,
        "exception": 1,
        "unknown": 1,
    }
    assert summary.n_optimizer_converged == 0
    assert summary.r2_values == []


def test_nonconverged_very_poor_r2_is_unsuccessful_not_execution_error() -> None:
    """A completed fixed-model fit with very poor R² has fit-stage provenance."""
    fit_result = SimpleNamespace(
        params={"Ktrans": np.array([0.1]), "ve": np.array([0.3])},
        r_squared=np.array([-0.75]),
        converged=np.array([False]),
        uncertainties={},
        fitted_curves=None,
        vp_at_bound=None,
    )

    result = ComputeDCEMaps()._fit_result_to_dicts(
        fit_result,
        np.array([[0, 0, 0]]),
        "tofts",
    )[0]

    assert result["failed"] is True
    assert result["failure_stage"] == "fit"
    assert result["failure_reason"] == "nonconverged_very_poor_r2"


def test_map_uncertainty_accounting_excludes_postfit_rejection() -> None:
    """MAP retained-voxel accounting excludes a degenerate rejected fit."""
    voxels = [
        VoxelInput(0, 0, 0, np.zeros(3), None),
        VoxelInput(0, 0, 0, np.zeros(3), None),
    ]
    base_result = {
        "failed": False,
        "coords": (0, 0, 0),
        "selected_model_name": "tofts",
        "delta_bic": np.nan,
        "constrained": False,
        "r_squared": 0.9,
        "converged": True,
        "uncertainties": {},
        "fitted_curve": None,
        "quality_flag": "good",
        "rhat": {},
        "ess": {},
    }
    rejected = {
        **base_result,
        "params": {"Ktrans": KTRANS_BOUNDS.max, "ve": 0.3},
        "uncertainty_source": "hessian_unavailable",
    }
    retained = {
        **base_result,
        "params": {"Ktrans": 0.1, "ve": 0.3},
        "uncertainty_source": "laplace_hessian",
        "inference_fallback": True,
    }

    summary = _scatter_dce_fit_results(
        voxels,
        [rejected, retained],
        _uncertainty_maps(),
        "tofts",
        False,
        "map",
        ConstraintMode.SKIP,
        0.0,
    )

    assert summary.n_rejected_postfit == 1
    assert summary.map_uncertainty_sources == ["laplace_hessian"]
    assert summary.n_inference_fallback == 1


def test_initialize_dce_qc_metrics_preserves_pre_fit_state() -> None:
    """Initial QC state retains counts, settings, and alignment provenance."""
    prepared = SimpleNamespace(
        mask=np.array([[[1, 1, 0]]]),
        fitting_mask=np.array([[[True, False, False]]]),
        is_auto=True,
        inference="map",
        tsnr_threshold=12.0,
        baseline_frames=6,
        selection_tr_seconds=2.5,
        n_timepoints=80,
    )
    constraint_qc = {"mode": "soft"}
    bat_correction = {"applied": True}
    summary = _initialize_dce_qc_metrics(
        prepared, "extended_tofts", constraint_qc, bat_correction, 3.0
    )
    assert summary == {
        "model": "auto",
        "inference": "map",
        "tsnr_threshold": 12.0,
        "baseline_frames": 6,
        "selection_tr_seconds": 2.5,
        "acquisition_duration_seconds": 197.5,
        "two_cxm_auto_sampling_eligible": True,
        "dsc_cbv_constraint": constraint_qc,
        "bat_correction": bat_correction,
    }
