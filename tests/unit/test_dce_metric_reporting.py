"""Finite derived-value reporting and primary-rate validity regressions."""

import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import (
    _build_dce_qc_mask,
    _compute_derived_params,
    _is_degenerate_dce_fit,
)
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel
from pmriprep.utils.dce_metrics import compute_kep


@pytest.mark.parametrize("ktrans,ve,expected", [
    (0.1, 0.01, 10), (0.1001, 0.01, 10.01),
    (0.0025, 0.0005, 5), (0.1, 0.0005, 200),
    (0.1, 1e-8, 1e7),
    (0, 0.2, 0),
])
def test_finite_ratios_are_reported_without_qc_classification(
    ktrans: float, ve: float, expected: float,
) -> None:
    """Derived maps report finite ratios throughout the positive-volume domain."""
    for model in (ToftsModel(), ExtendedToftsModel()):
        assert model.compute_derived_parameters([ktrans, ve, 0.02])["kep"] == pytest.approx(expected)
        params = {"Ktrans": ktrans, "ve": ve, "vp": 0.02}
        _compute_derived_params(params, model.name)
        assert params["kep"] == pytest.approx(expected)


@pytest.mark.parametrize("ktrans,ve", [
    (np.nan, 0.2), (np.inf, 0.2), (-1, 0.2), (0.1, np.nan),
    (0.1, np.inf), (0.1, 0), (0.1, -0.1), (1e308, 1e-308), (1e30, 1e-30),
])
def test_invalid_or_overflow_ratio_is_nan(ktrans: float, ve: float) -> None:
    """Undefined and unrepresentable ratios remain missing."""
    assert np.isnan(compute_kep(ktrans, ve))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_primary_rates_are_rejected(value: float) -> None:
    """A nonfinite primary rate cannot survive the post-fit predicate."""
    for name in ("patlak", "tofts", "extended_tofts"):
        assert _is_degenerate_dce_fit(name, {"Ktrans": value, "ve": 0.2, "vp": 0.02})
    for key in ("Fp", "PS"):
        params = {"Fp": 40, "PS": 10, "ve": 0.2, "vp": 0.02}
        params[key] = value
        assert _is_degenerate_dce_fit("2cxm", params)


@pytest.mark.parametrize("ve", [0.0005, 0.005, 0.02])
def test_2cxm_derived_ratio_parity(ve: float) -> None:
    """2CXM scalar and batched-result reporting preserve the same finite ratio."""
    scalar = TwoCompartmentExchangeModel().compute_derived_parameters([40, 10, ve, 0.02])
    batch = {"Fp": 40, "PS": 10, "ve": ve, "vp": 0.02}
    _compute_derived_params(batch, "2cxm")
    assert scalar["kep"] == pytest.approx(0.08 / ve)
    assert batch["kep"] == pytest.approx(scalar["kep"])


@pytest.mark.parametrize("name", ["tofts", "extended_tofts", "2cxm"])
@pytest.mark.parametrize("ve,ktrans,accepted", [
    (0.2, 0.1, True), (0.005, 0.1, False),
    (0.0005, 0.1, False), (1e-8, 0.1, False),
    (0.0005, 0.0025, True),
])
def test_kep_range_directly_gates_qc_mask(
    name: str, ve: float, ktrans: float, accepted: bool,
) -> None:
    """Extreme finite ratios cannot pass the explicit kep criterion."""
    params = {"Ktrans": ktrans, "ve": ve, "vp": 0.02,
              "Fp": ktrans * 200, "PS": ktrans * 200}
    _compute_derived_params(params, name)
    result = _build_dce_qc_mask(
        fitting_mask=np.array([True]),
        optimizer_converged=np.array([True]),
        retained_nondegenerate=np.array([True]),
        tsnr=np.array([100.0]),
        signal_qc_available=False, ktrans=np.array([params["Ktrans"]]),
        r_squared=np.array([0.99]), ve=np.array([ve]), vp=np.array([0.02]),
        kep=np.array([params["kep"]]),
        model_selection=np.array([1]), model_selection_codes={name: 1},
        is_auto=False, model_name=name,
    )
    assert bool(result[0]) is accepted
    assert np.isfinite(params["kep"])
