"""Characterization tests for DCE fit-result scattering decisions."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import (
    _is_degenerate_dce_fit,
    _scatter_degenerate_dce_fit,
    _scatter_dce_fit_artifacts,
    _scatter_dce_parameters,
    _scatter_dce_uncertainties,
)
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
