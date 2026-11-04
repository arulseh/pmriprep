"""Identification thresholds, family mapping and immutable profile consumption."""

from dataclasses import replace

import numpy as np
import pytest

from pmriprep.models.dce import identification as ident
from pmriprep.models.dce.delay_profiles import DelayProfile
from pmriprep.models.dce.delayed import DelayedPchip


def _profile(family: str, params: np.ndarray, sse: np.ndarray | None = None,
             observed: np.ndarray | None = None) -> DelayProfile:
    params = np.atleast_2d(params)
    count = len(params)
    return DelayProfile(
        family, np.arange(count, dtype=float),
        np.arange(60) if observed is None else observed,
        params[:, None].copy(), np.full((count, 1), 1e-8) if sse is None else sse[:, None],
        np.ones((count, 1), dtype=int), np.full((count, 1), -1, dtype=int),
        np.full((count, 1), np.nan), np.full((count, 1), -1, dtype=int), {},
    )


def _input() -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(60) / 60
    return .01 + 5 * np.exp(-((times - .3) / .08)**2), times


@pytest.mark.parametrize("family,params,names", [
    ("patlak", [.2, np.nan, .05], ("Ktrans", "vp")),
    ("tofts", [.2, .3], ("Ktrans", "ve")),
    ("extended_tofts", [.2, .3, .05], ("Ktrans", "ve", "vp")),
    ("2cxm", [60., 12., .3, .06], ("Ktrans", "ve", "vp", "Fp", "PS")),
])
def test_family_mapping_and_fixed_zero_do_not_change_estimates(
    family: str, params: list[float], names: tuple[str, ...],
) -> None:
    profile = _profile(family, np.array(params))
    before = profile.params.copy(), profile.sse.copy(), profile.status.copy()
    result = ident.identify_profile_point(profile, 0, 0, *_input())
    assert result.names == names
    assert result.primary == (names[:3] if family == "2cxm" else names)
    assert result.primary_pass
    np.testing.assert_array_equal(result.spread, np.zeros(len(names)))
    assert np.isfinite(result.se).all()
    if family == "2cxm":
        np.testing.assert_allclose(result.values, [.1, .3, .06, 60., 12.])
        # Delta method must retain the flow/permeability covariance cross term.
        gradient = np.array([12.**2/72.**2/100, 60.**2/72.**2/100, 0, 0])
        assert result.se[0] == pytest.approx(np.sqrt(gradient @ result.covariance @ gradient))
    for original, current in zip(before, (profile.params, profile.sse, profile.status)):
        np.testing.assert_array_equal(original, current)


def test_profile_excursion_uses_free_minimum_and_observation_count() -> None:
    aif, times = _input()
    observed = np.array([0, 10, 20, 30, 40, 59])
    profile = _profile("patlak", np.array([[.9, np.nan, .1], [.2, np.nan, .1], [1., np.nan, .1]]),
                       np.exp(np.array([0., .5, 2.]) / len(observed)), observed)
    result = ident.identify_profile_point(profile, 0, 2, aif, times)
    np.testing.assert_allclose(result.spread, [.8, 0.])
    # Independent linear covariance uses all AIF history at retained observations.
    context = DelayedPchip(aif, times, 2., observed)
    design = np.column_stack([context.convolution(0.), context.cp])
    expected = profile.sse[2, 0] / (len(observed) - 2) * np.linalg.inv(design.T @ design)
    np.testing.assert_allclose(result.covariance, expected, rtol=1e-8, atol=1e-9)
    profile.sse[0, 0] = np.nan
    unavailable = ident.identify_profile_point(profile, 0, 2, aif, times)
    assert not unavailable.primary_pass
    assert np.isnan(unavailable.spread).all()
    assert all("output_unavailable" in reasons for reasons in unavailable.reasons)


def _linear_screen(values: np.ndarray, spread: np.ndarray, volumes: tuple[int, ...] = ()) -> ident.IdentificationResult:
    design = np.r_[np.eye(2), np.eye(2)]
    return ident.evaluate_identification(
        values, np.ones(2), volumes, lambda p: design @ p, .1,
        values, np.eye(2), np.ones(2), spread, volumes,
        names=("a", "b"), primary=("a", "b"),
    )


def test_exact_box_precision_and_joint_bound_inequalities(monkeypatch: pytest.MonkeyPatch) -> None:
    cov = np.eye(2) / 64
    monkeypatch.setattr(ident, "_covariance", lambda *args: (cov.copy(), 1., "ok"))
    at = _linear_screen(np.array([.25, .75]), np.zeros(2))
    assert at.primary_pass  # Exactly two SE at either box edge, exactly 50% SE.
    below = _linear_screen(np.array([np.nextafter(.25, 0.), .75]), np.zeros(2))
    assert "lower_box_bound" in below.reasons[0]
    above = _linear_screen(np.array([.25, np.nextafter(.75, 1.)]), np.zeros(2))
    assert "upper_box_bound" in above.reasons[1]
    # A scaled 3-4-5 triangle gives an exactly representable 50% effective SE.
    cov = np.eye(2) * .1875**2
    exact = _linear_screen(np.array([.625, .5]), np.array([.25, 0.]))
    assert "profile_spread" not in exact.reasons[0]
    excess = _linear_screen(np.array([.625, .5]), np.array([np.nextafter(.25, 1.), 0.]))
    assert "profile_spread" in excess.reasons[0]
    cov = np.array([[1., -.5], [-.5, 1.]]) / 64
    joint = _linear_screen(np.array([.25, .5]), np.zeros(2), (0, 1))
    assert joint.slack == 2 * joint.slack_se
    assert all("joint_volume_bound" in why for why in joint.reasons)


def test_optional_exchange_outputs_do_not_exempt_global_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile("2cxm", np.array([499.9, 10., .2, .05]))
    cov = np.diag([25., .01, 1e-6, 1e-6])
    monkeypatch.setattr(ident, "_covariance", lambda *args: (cov.copy(), 1., "ok"))
    result = ident.identify_profile_point(profile, 0, 0, *_input())
    assert "upper_box_bound" in result.reasons[result.names.index("Fp")]
    assert result.primary_pass and not result.passed.all()
    # A global numerical failure still affects primary outputs, despite optional Fp/PS.
    monkeypatch.setattr(ident, "_covariance", lambda *args: (np.full((4, 4), np.nan), 0., "rank_deficient"))
    failed = ident.identify_profile_point(profile, 0, 0, *_input())
    assert not failed.primary_pass and not failed.passed.any()
    assert all("rank_deficient" in why for why in failed.reasons)
    calls = iter([cov, cov * 1.01])
    monkeypatch.setattr(ident, "_covariance", lambda *args: (next(calls), 1., "ok"))
    unstable = ident.identify_profile_point(profile, 0, 0, *_input())
    assert not unstable.primary_pass
    assert all("derivative_unstable" in why for why in unstable.reasons)


def test_invalid_profile_geometry_is_not_a_qc_failure() -> None:
    profile = _profile("tofts", np.array([.2, .3]))
    with pytest.raises(ValueError, match="indices"):
        ident.identify_profile_point(profile, 0, 1, *_input())
    with pytest.raises(ValueError, match="Aligned"):
        ident.identify_profile_point(replace(profile, observed=np.array([0, 0])), 0, 0, *_input())
