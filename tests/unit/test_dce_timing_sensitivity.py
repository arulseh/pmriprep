"""Signed offsets, raw evidence, and availability independent of central QC."""

from dataclasses import replace

import numpy as np
import pytest

from pmriprep.models.dce import timing_sensitivity as diagnostics
from pmriprep.models.dce.delay_profiles import DelayProfile, ProfileFamily, build_delay_profile
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.fit_errors import NumericalFitError
from pmriprep.models.dce.fixed_inference import evaluate_fixed_kinetics
from pmriprep.models.dce.local_selection import FamilyTiming, family_timing


def _case(family: ProfileFamily, delay: float) -> tuple[DelayProfile, FamilyTiming, np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(65) / 30
    aif = .02 + 5 * np.exp(-((times - .3) / .1)**2) + .2 * (times > .5)
    observed = np.r_[0:10, 13:65]
    p = {"patlak": [.08, .04], "tofts": [.2, .3],
         "extended_tofts": [.2, .3, .04], "2cxm": [60., 12., .3, .06]}[family]
    context = DelayedPchip(aif, times, delay, observed)
    tissue = evaluate_fixed_kinetics(family, context, np.array(p))
    tissue += np.random.default_rng(23).normal(0, .0002, len(tissue))
    profile = build_delay_profile(aif, times, tissue[None], family,
                                  observed=observed, delays_seconds=np.array([0., 6., 12.]))
    timing = family_timing(profile, np.array([0]), 0, frame_interval_seconds=2)
    assert profile.delays_seconds[timing.index] == delay and timing.reason is None
    return profile, timing, aif, times, tissue


@pytest.mark.parametrize("family,delay", [
    ("patlak", 0.), ("tofts", 12.), ("extended_tofts", 0.), ("2cxm", 12.), ("2cxm", 0.),
])
def test_two_real_refits_preserve_family_history_and_central_fit(
    family: ProfileFamily, delay: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, timing, aif, times, tissue = _case(family, delay)
    before = [a.copy() for a in (profile.params, profile.sse, profile.status, aif, times, tissue)]
    starts = []
    native = diagnostics.fit_native

    def track(context: DelayedPchip, curve: np.ndarray, **kwargs: object) -> object:
        starts.append(np.asarray(kwargs["previous"]).copy())
        return native(context, curve, **kwargs)

    monkeypatch.setattr(diagnostics, "fit_native", track)
    result = diagnostics.timing_sensitivity(profile, timing, 0, aif, times, tissue)
    assert result.family == family and result.frame_interval_seconds == 2
    assert [s.delay_seconds for s in result.offsets] == [delay - 2, delay + 2]
    np.testing.assert_array_equal(result.observations, profile.observed)
    if family == "2cxm":
        assert len(starts) == 2
        for start in starts:
            np.testing.assert_array_equal(start, profile.params[timing.index, 0])
    for side in result.offsets:
        context = DelayedPchip(aif, times, side.delay_seconds, profile.observed)
        assert side.identification is not None
        p = side.parameters[[0, 2]] if family == "patlak" else side.parameters
        curve = evaluate_fixed_kinetics(family, context, p)
        assert side.sse == pytest.approx(float(np.sum((curve - tissue)**2)), abs=1e-12)
        assert side.objective_reason is None
        assert side.objective_change == pytest.approx(len(tissue) * np.log(side.sse / profile.sse[timing.index, 0]))
        assert side.held_before == int(context.held_before.sum())
        assert side.held_after == int(context.held_after.sum())
        for i, name in enumerate(result.names):
            j = side.identification.names.index(name)
            central = result.central_identification
            assert side.raw_changes[i] == side.identification.values[j] - central.values[central.names.index(name)]
        np.testing.assert_array_equal(np.isnan(side.changes), ~side.available)
        assert side.available.tolist() == [not r for r in side.reasons]
    for saved, actual in zip(before, (profile.params, profile.sse, profile.status, aif, times, tissue)):
        np.testing.assert_array_equal(saved, actual)


def test_diagnostic_failures_do_not_change_centre_or_hide_raw_values(monkeypatch: pytest.MonkeyPatch) -> None:
    profile, timing, aif, times, tissue = _case("patlak", 0.)
    original = diagnostics.identify_fixed_point
    central = diagnostics.identify_profile_point(profile, 0, timing.index, aif, times)

    def weak_vp(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        return replace(result, reasons=((), ("relative_uncertainty",)))

    monkeypatch.setattr(diagnostics, "identify_fixed_point", weak_vp)
    result = diagnostics.timing_sensitivity(profile, timing, 0, aif, times, tissue)
    for side in result.offsets:
        assert side.retained and np.isfinite(side.raw_changes).all()
        assert side.available.tolist() == [True, False]
        assert np.isnan(side.changes[1])
        assert side.reasons[1] == ("perturbed_relative_uncertainty",)
    np.testing.assert_array_equal(result.central_identification.passed, central.passed)
    fitter = diagnostics.fit_patlak

    def failed(context: DelayedPchip, curves: np.ndarray) -> object:
        if context.delay < 0:
            raise NumericalFitError("test numerical failure")
        return fitter(context, curves)

    monkeypatch.setattr(diagnostics, "fit_patlak", failed)
    result = diagnostics.timing_sensitivity(profile, timing, 0, aif, times, tissue)
    assert result.offsets[0].failure_reason == "numerical_failure"
    assert result.offsets[0].objective_reason == "numerical_failure"
    assert not result.offsets[0].available.any()
    assert result.offsets[1].retained

    def broken(*args: object) -> object:
        raise RuntimeError("backend contract")

    monkeypatch.setattr(diagnostics, "fit_patlak", broken)
    with pytest.raises(RuntimeError, match="backend contract"):
        diagnostics.timing_sensitivity(profile, timing, 0, aif, times, tissue)


def test_exact_zero_residual_does_not_invent_objective_or_usable_sensitivity() -> None:
    times = np.arange(20) / 30
    aif = np.exp(-((times - .2) / .1)**2)
    tissue = np.zeros(len(times))
    profile = build_delay_profile(aif, times, tissue[None], "patlak", delays_seconds=np.array([0.]))
    timing = family_timing(profile, np.array([0]), 0, frame_interval_seconds=2)
    result = diagnostics.timing_sensitivity(profile, timing, 0, aif, times, tissue)
    for side in result.offsets:
        assert side.sse == 0 and side.objective_reason == "zero_selected_sse"
        assert np.isnan(side.objective_change)
        np.testing.assert_array_equal(side.raw_changes, np.zeros(2))
        assert not side.available.any() and np.isnan(side.changes).all()
