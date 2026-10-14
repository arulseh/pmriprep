"""Optional CPI fallback replaces measurements only when oSVD supplies a result."""

from dataclasses import replace
from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi import acquisition, connection, fallback
from pmriprep.models.dsc.signal_cpi.settings import CPIOptions
from pmriprep.utils.dsc_deconvolution import svd_deconvolve_oscillation


@pytest.fixture
def prepared() -> tuple[acquisition.CPIAcquisition, np.ndarray]:
    time = np.arange(30.) * 3
    u = np.maximum(time - 9, 0)
    aif = 10*u**2*np.exp(-u/3)
    raw = np.broadcast_to(.2*u**2*np.exp(-u/5), (3, 1, 1, 30)).copy()
    a = acquisition.prepare_cpi_acquisition(100*np.exp(-.03*raw), aif, time, .03,
        leakage_method="none", options=CPIOptions(dispersion="none"))
    return a, raw


def rejected(index: int = 0) -> connection.GuidedVoxel:
    return connection.GuidedVoxel(index, "fallback_requested", ("inadequate_signal_fit",),
                                  family="cpi")


def resolve(a: acquisition.CPIAcquisition, raw: np.ndarray, rows: list[connection.GuidedVoxel],
            corrected: np.ndarray | None = None) -> list[fallback.CPIOutputVoxel]:
    return list(fallback.resolve_fallbacks(a, iter(rows), raw_concentration=raw,
        corrected_concentration=corrected, oi_target=.035))


@pytest.mark.parametrize("status", ["retained", "provisional"])
def test_valid_or_unfinished_fit_never_triggers_fallback(prepared: Any, status: Any,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    a, raw = prepared
    row = connection.GuidedVoxel(0, status, (), .01, .04, 4., np.nan,
                                  cbf_std=.001, fitted_k2=.002)
    monkeypatch.setattr(fallback, "svd_deconvolve_oscillation", lambda *a, **k: pytest.fail("unexpected refit"))
    result = resolve(a, raw, [row])[0]
    assert result.source == "cpi" and not result.fallback_attempted
    assert result.cbf == .01 and result.cbf_std == .001 and result.fitted_k2 == .002
    assert result.fit is row and np.isnan(result.delay)


def test_disabled_fallback_preserves_missing_values_and_reasons(prepared: Any,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    a, raw = prepared
    a = replace(a, options=replace(a.options, fallback="none"))
    monkeypatch.setattr(fallback, "svd_deconvolve_oscillation", lambda *a, **k: pytest.fail("unexpected oSVD"))
    row = rejected()
    result = resolve(a, raw, [row])[0]
    assert result.source == "missing" and result.fallback_reason == "disabled"
    assert result.fit is row and result.fit.reasons == row.reasons
    assert not result.fallback_attempted
    assert all(np.isnan(getattr(result, name)) for name in ("cbf", "cbv", "mtt", "delay", "cbf_std", "fitted_k2"))


def test_none_uses_raw_not_corrected_curve_and_real_osvd(prepared: Any) -> None:
    a, raw = prepared
    expected = svd_deconvolve_oscillation(raw[0, 0, 0], a.branches[0].aif, 3., oi_target=.035)
    result = resolve(a, raw, [rejected()], corrected=raw*100)[0]
    assert result.source == "osvd" and result.fallback_attempted
    assert result.fallback_reason is None
    assert result.osvd_oi_target_met == expected.converged
    for name in ("cbf", "cbv", "mtt"):
        assert getattr(result, name) == getattr(expected, name)
    assert np.isnan(result.delay)
    assert np.isnan(result.cbf_std) and np.isnan(result.fitted_k2)


def test_bsw_uses_exact_corrected_curve(prepared: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    a, raw = prepared
    a = replace(a, branches=(replace(a.branches[0], leakage_method="bsw"),))
    corrected = raw*2

    def fit(tissue: np.ndarray, aif: np.ndarray, dt: float, **kwargs: Any) -> Any:
        np.testing.assert_array_equal(tissue, corrected[0, 0, 0])
        return SimpleNamespace(cbf=.01, cbv=.04, mtt=4., delay=3., converged=True)

    monkeypatch.setattr(fallback, "svd_deconvolve_oscillation", fit)
    assert resolve(a, raw, [rejected()], corrected)[0].source == "osvd"
    with pytest.raises(ValueError, match="leakage policy"):
        resolve(a, raw, [rejected()])


def test_unmet_osvd_target_keeps_valid_replacement(prepared: Any) -> None:
    """Target failure is a qualification, not a failed replacement."""
    a, raw = prepared
    fit = svd_deconvolve_oscillation(raw[0, 0, 0], a.branches[0].aif, 3., oi_target=1e-10)
    assert not fit.converged
    result = list(fallback.resolve_fallbacks(
        a, iter([rejected()]), raw_concentration=raw,
        corrected_concentration=None, oi_target=1e-10,
    ))[0]
    assert result.source == "osvd" and result.fallback_attempted
    assert result.osvd_oi_target_met is False and result.fallback_reason is None
    for name in ("cbf", "cbv", "mtt"):
        assert np.isfinite(getattr(result, name))
        assert getattr(result, name) == getattr(fit, name)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.])
def test_inadmissible_concentration_is_not_attempted(prepared: Any, bad: float) -> None:
    a, raw = prepared
    raw[0] = bad
    result = resolve(a, raw, [rejected()])[0]
    assert result.source == "missing" and not result.fallback_attempted
    assert result.fallback_reason == "inadmissible_concentration"


@pytest.mark.parametrize("fault", ["linear_algebra", "floating_point", "nonfinite", "zero"])
def test_failed_osvd_does_not_set_substitution_mask(prepared: Any, fault: str,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    a, raw = prepared

    def fit(*args: Any, **kwargs: Any) -> Any:
        if fault == "linear_algebra":
            raise np.linalg.LinAlgError("no solve")
        if fault == "floating_point":
            raise FloatingPointError("overflow")
        return SimpleNamespace(cbf=np.nan if fault == "nonfinite" else 0., cbv=.04, mtt=4., delay=0.)

    monkeypatch.setattr(fallback, "svd_deconvolve_oscillation", fit)
    result = resolve(a, raw, [rejected()])[0]
    assert result.source == "missing" and result.fallback_attempted
    assert result.fallback_reason is not None and np.isnan(result.cbf)


def test_programming_error_propagates_and_output_indices_are_checked(prepared: Any,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    a, raw = prepared

    def bad_call(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("broken solver interface")

    monkeypatch.setattr(fallback, "svd_deconvolve_oscillation", bad_call)
    with pytest.raises(TypeError, match="broken solver"):
        resolve(a, raw, [rejected()])
    with pytest.raises(ValueError, match="indices"):
        resolve(a, raw, [rejected(3)])
    outside = connection.GuidedVoxel(1, "outside_support", ("outside_fit_support",))
    assert resolve(a, raw, [outside])[0].source == "outside_support"
