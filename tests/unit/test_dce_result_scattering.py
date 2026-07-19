"""Characterization tests for DCE fit-result scattering decisions."""

import pytest

from pmriprep.interfaces.dce_compute import _is_degenerate_dce_fit
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
