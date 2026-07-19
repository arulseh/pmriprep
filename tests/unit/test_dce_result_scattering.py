"""Characterization tests for DCE fit-result scattering decisions."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import (
    _is_degenerate_dce_fit,
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
