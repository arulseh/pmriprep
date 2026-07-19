"""Characterization tests for DCE fit-result scattering decisions."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import (
    VoxelInput,
    _account_retained_dce_fit,
    _accumulate_mcmc_diagnostics,
    _compute_vp_prior_departure,
    _is_degenerate_dce_fit,
    _scatter_degenerate_dce_fit,
    _scatter_dce_fit_artifacts,
    _scatter_dce_parameters,
    _scatter_dce_uncertainties,
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
    ("model", "params", "constrained", "expected"),
    [
        ("tofts", {"Ktrans": 0.1, "ve": 0.3}, False, False),
        ("tofts", {"Ktrans": 0.0, "ve": 0.3}, False, True),
        ("tofts", {"Ktrans": KTRANS_BOUNDS.max, "ve": 0.3}, False, True),
        ("tofts", {"Ktrans": 0.1, "ve": VE_BOUNDS.max}, False, True),
        ("patlak", {"Ktrans": 0.0, "vp": 0.03}, False, False),
        ("patlak", {"Ktrans": -1e-6, "vp": 0.03}, False, True),
        ("patlak", {"Ktrans": 0.1, "vp": 0.0}, False, True),
        ("patlak", {"Ktrans": 0.1, "vp": 0.0}, True, False),
        ("extended_tofts", {"Ktrans": 0.0, "ve": 0.3, "vp": 0.03}, True, False),
        ("extended_tofts", {"Ktrans": 0.0, "ve": 0.3, "vp": 0.03}, False, True),
        ("extended_tofts", {"Ktrans": 0.1, "ve": 0.3, "vp": VP_BOUNDS.max}, False, True),
        ("extended_tofts", {"Ktrans": 0.1, "ve": 0.3, "vp": VP_BOUNDS.max}, True, False),
        ("2cxm", {"Fp": 1.0, "PS": 0.0, "ve": 0.3, "vp": 0.03}, False, False),
        ("2cxm", {"Fp": 0.0, "PS": 0.1, "ve": 0.3, "vp": 0.03}, False, True),
        ("2cxm", {"Fp": FP_BOUNDS.max, "PS": 0.1, "ve": 0.3, "vp": 0.03}, False, True),
        ("2cxm", {"Fp": 1.0, "PS": PS_BOUNDS.max, "ve": 0.3, "vp": 0.03}, False, True),
    ],
)
def test_degeneracy_rules_preserve_model_specific_boundaries(
    model: str,
    params: dict[str, float],
    constrained: bool,
    expected: bool,
) -> None:
    """Exact bounds and constrained-fit exceptions remain unchanged."""
    assert _is_degenerate_dce_fit(model, params, constrained) is expected


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
    )


def test_uncertainty_scatter_applies_model_fields_and_inflation() -> None:
    """2CXM writes every uncertainty using its parameter-specific factor."""
    maps = _uncertainty_maps()
    _scatter_dce_uncertainties(
        maps,
        (0, 0, 0),
        "2cxm",
        {"Ktrans": 1.0, "ve": 2.0, "vp": 3.0, "Fp": 4.0, "PS": 5.0},
    )
    assert maps.ktrans_std[0, 0, 0] == 2.0
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


def test_fit_artifact_scatter_writes_curve_bound_and_quality() -> None:
    """All non-parameter artifacts use the same voxel coordinate."""
    maps = SimpleNamespace(
        fitted_curves=np.zeros((1, 1, 1, 3), dtype=np.float32),
        vp_at_bound=np.zeros((1, 1, 1), dtype=np.float32),
        quality_flag=np.zeros((1, 1, 1), dtype=np.int8),
        quality_flag_codes={"good": 1, "high_uncertainty": 3},
    )
    _scatter_dce_fit_artifacts(
        maps,
        (0, 0, 0),
        {
            "fitted_curve": np.array([0.0, 0.2, 0.1]),
            "vp_at_bound": 1.0,
            "quality_flag": "high_uncertainty",
        },
    )
    np.testing.assert_array_equal(
        maps.fitted_curves[0, 0, 0],
        np.array([0.0, 0.2, 0.1], dtype=np.float32),
    )
    assert maps.vp_at_bound[0, 0, 0] == 1.0
    assert maps.quality_flag[0, 0, 0] == 3


def test_fit_artifact_scatter_defaults_without_optional_curve() -> None:
    """Missing optional values retain no curve and default to a good label."""
    maps = SimpleNamespace(
        fitted_curves=None,
        vp_at_bound=np.ones((1, 1, 1), dtype=np.float32),
        quality_flag=np.zeros((1, 1, 1), dtype=np.int8),
        quality_flag_codes={"good": 1},
    )
    _scatter_dce_fit_artifacts(maps, (0, 0, 0), {})
    assert maps.vp_at_bound[0, 0, 0] == 0.0
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
    _scatter_degenerate_dce_fit(maps, (0, 0, 0), "patlak", True, -0.2)
    assert np.isnan(maps.ktrans[0, 0, 0])
    assert np.isnan(maps.vp[0, 0, 0])
    assert maps.ve[0, 0, 0] == 2.0
    assert maps.r_squared[0, 0, 0] == -0.2


def test_degenerate_scatter_uses_fixed_zero_for_tofts_maps() -> None:
    """Fixed Tofts rejection restores zero sentinels for Ktrans/ve/kep."""
    maps = _uncertainty_maps()
    maps.ktrans[:] = 1.0
    maps.ve[:] = 2.0
    maps.kep[:] = 3.0
    maps.vp[:] = 4.0
    _scatter_degenerate_dce_fit(maps, (0, 0, 0), "tofts", False, 0.1)
    assert maps.ktrans[0, 0, 0] == 0.0
    assert maps.ve[0, 0, 0] == 0.0
    assert maps.kep[0, 0, 0] == 0.0
    assert maps.vp[0, 0, 0] == 4.0
    assert maps.r_squared[0, 0, 0] == 0.1


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


def test_accumulate_mcmc_diagnostics_ignores_ess_without_rhat() -> None:
    """An empty R-hat record remains a non-MCMC voxel even if ESS is present."""
    rhat: dict[str, list[float]] = {}
    ess: dict[str, list[float]] = {}
    increment = _accumulate_mcmc_diagnostics(
        {"rhat": {}, "ess": {"Ktrans": 250.0}}, rhat, ess,
    )
    assert increment == 0
    assert rhat == {}
    assert ess == {}


def test_retained_fit_accounting_tracks_constrained_convergence() -> None:
    """A converged constrained fit contributes constrained count and R²."""
    voxel = VoxelInput(0, 0, 0, np.zeros(3), 0.04)
    update = _account_retained_dce_fit(
        voxel,
        {"constrained": True, "converged": True, "r_squared": 0.8},
        ConstraintMode.SOFT,
    )
    assert update.n_converged == 1
    assert update.n_failed == 0
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
    assert update.n_converged == 0
    assert update.n_failed == 1
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
