"""Local physical support, nested evidence and capped model-admission contracts."""

import numpy as np
import pytest

from pmriprep.models.dce import local_selection as selection
from pmriprep.models.dce.automatic_admission import (
    exchange_admission,
    ExchangeAdmission,
)
from pmriprep.models.dce.delay_profiles import DelayProfile
from pmriprep.models.dce.native_exchange import NativeInput


def _profile(family: str, sse: np.ndarray, n: int = 20) -> DelayProfile:
    values = np.asarray(sse, dtype=float)
    physical = {
        "patlak": [0.2, np.nan, 0.03],
        "extended_tofts": [0.2, 0.3, 0.03],
        "tofts": [0.2, 0.3],
        "2cxm": [60.0, 15.0, 0.3, 0.03],
    }[family]
    return DelayProfile(
        family,
        np.arange(len(values)) / 10,
        np.arange(n),
        np.broadcast_to(physical, (*values.shape, len(physical))).copy(),
        values.copy(),
        np.ones(values.shape, dtype=int),
        np.full(values.shape, -1),
        np.full(values.shape, np.nan),
        np.full(values.shape, -1),
        {},
    )


def test_physical_disc_uses_affine_floor_and_single_slice() -> None:
    xyz = np.array([[i, j, 2] for i in range(7) for j in range(7)])
    for basis in (
        np.diag([0.8, 0.8, 4.0]),
        np.diag([3.0, 2.0, 1.0]),
        np.array([[0.0, -1.2, 0.3], [0.8, 0.5, 0.1], [0.2, 0.0, 3.0]]),
    ):
        affine = np.eye(4)
        affine[:3, :3] = basis
        affine[:3, 3] = [300.0, -25.0, 100.0]
        radius, pools = selection.in_plane_neighborhoods(xyz, affine)
        assert radius == max(
            2.0, np.sqrt(sum(np.linalg.norm(basis[:, i]) ** 2 for i in (0, 1)))
        )
        for i, pool in enumerate(pools):
            expected = [
                j
                for j in range(len(xyz))
                if np.linalg.norm(basis @ (xyz[j] - xyz[i])) <= radius + 1e-6
            ]
            np.testing.assert_array_equal(pool, expected)
            assert i in pool
    with pytest.raises(ValueError, match="one slice"):
        selection.in_plane_neighborhoods(np.array([[0, 0, 0], [0, 0, 1]]), np.eye(4))
    with pytest.raises(ValueError, match="one slice"):
        selection.in_plane_neighborhoods(np.array([[0, 0, 0], [0, 0, 0]]), np.eye(4))


def test_complete_support_singleton_and_selected_failure_are_distinct() -> None:
    p = _profile("patlak", [[10.0, 10.0], [1.0, 1.0], [2.0, 2.0]])
    pair = np.array([0, 1])
    chosen = selection.family_timing(p, pair, 0, frame_interval_seconds=2.0)
    assert chosen.reference == chosen.index == 1 and not chosen.free
    assert chosen.support == 2 and chosen.fallback is None
    p.sse[0, 1] = np.nan
    singleton = selection.family_timing(p, pair, 0, frame_interval_seconds=2.0)
    assert singleton.reference == 0 and singleton.index == 1 and singleton.free
    assert singleton.support == 1 and singleton.fallback == "insufficient_support"
    missing = selection.family_timing(p, pair, 1, frame_interval_seconds=2.0)
    assert missing.index == -1 and missing.reason == "incomplete_family_profile"
    p.sse[0, 1] = 10.0
    p.status[1, 0] = 0
    failed = selection.family_timing(p, pair, 0, frame_interval_seconds=2.0)
    assert failed.index == 1 and failed.status == 0 and failed.support == 2
    assert failed.reason == "selected_nonconverged"


def test_flat_threshold_is_strict_and_ties_choose_lowest_grid_point() -> None:
    p = _profile("tofts", [[np.e, np.e], [1.0, 1.0], [1.0, 1.0]], n=3)
    q = 3 * np.log(p.sse).sum(axis=1)
    assert np.ptp(q) == 6
    exact = selection.family_timing(p, np.array([0, 1]), 0, frame_interval_seconds=2.0)
    assert exact.reference == exact.index == exact.best == 1
    assert exact.fallback is None
    p.sse[0] = np.exp((6 - 1e-10) / 6)
    below = selection.family_timing(p, np.array([0, 1]), 0, frame_interval_seconds=2.0)
    assert below.reference == 0 and below.fallback == "flat_profile"


def test_boundary_diagnostics_use_original_interval_and_are_report_only() -> None:
    p = _profile("tofts", np.full((121, 2), np.exp(2)), n=3)
    p.observed[:] = [0, 2, 4]
    p.sse[101:] = 1.0
    p.sse[-2] = 0.9
    # The neighbour fixes the shared reference at zero despite the centre's
    # free optimum being one grid step below the upper bound.
    p.sse[:, 1] = 100.0
    p.sse[0, 1] = 1e-10
    timing = selection.family_timing(p, np.array([0, 1]), 0, frame_interval_seconds=2.0)
    assert timing.best == 119 and timing.index == timing.reference == 0
    assert not timing.free and timing.reference_at_bound
    assert timing.free_optimum_near_upper and not timing.free_optimum_near_lower
    assert timing.upper_edge_evidence == 6.0 and timing.upper_edge == "truncated"
    p.sse[100, 0] = np.exp((6 - 1e-10) / 3)
    flat = selection.family_timing(p, np.array([0, 1]), 0, frame_interval_seconds=2.0)
    assert flat.upper_edge == "flat_edge"
    assert (flat.index, flat.reference, flat.free) == (timing.index, timing.reference, timing.free)
    unavailable = selection.family_timing(p, np.array([0, 1]), 0, frame_interval_seconds=13.0)
    assert unavailable.free_optimum_near_upper and unavailable.upper_edge is None
    assert unavailable.boundary_unavailable_reason == "no_pre_edge_samples"
    p.sse[:, 0] = 10.0
    p.sse[1, 0] = 1.0
    lower = selection.family_timing(p, np.array([0, 1]), 0, frame_interval_seconds=2.0)
    assert lower.free_optimum_near_lower and not lower.free_optimum_near_upper
    assert lower.upper_edge is None and np.isnan(lower.upper_edge_evidence)


def test_boundary_diagnostics_are_unavailable_without_a_full_search() -> None:
    for sse, reason in (
        ([[1.0]], "delay_fixed"),
        ([[1.0], [2.0]], "incomplete_delay_range"),
        ([[np.nan], [1.0]], "incomplete_family_profile"),
    ):
        p = _profile("patlak", sse)
        timing = selection.family_timing(p, np.array([0]), 0, frame_interval_seconds=2.0)
        assert timing.free_optimum_near_lower is timing.free_optimum_near_upper is None
        assert timing.reference_at_bound is None and timing.upper_edge is None
        assert np.isnan(timing.upper_edge_evidence)
        assert timing.boundary_unavailable_reason == reason


def test_family_guard_boundary_without_double_delay_penalty() -> None:
    n = 20
    times = np.arange(n) / 12
    aif = np.ones(n)
    rng = np.random.default_rng(83)
    for ratio in np.r_[
        np.exp((np.log(n) - np.array([6.0, 6.0 - 1e-10, 6.0 + 1e-10])) / n),
        rng.uniform(0.1, 2.0, 25),
    ]:
        profiles = {
            "patlak": _profile("patlak", [[1.0]]),
            "extended_tofts": _profile("extended_tofts", [[ratio]]),
        }
        actual = selection.select_pool(profiles, aif, times, np.array([0]), 0)
        improvement = actual.timing["patlak"].bic - actual.timing["extended_tofts"].bic
        expected = "extended_tofts" if improvement >= 6.0 else "patlak"
        assert actual.family == expected
        expected_delta = improvement if improvement >= 6 else (-improvement if improvement <= 0 else np.nan)
        np.testing.assert_equal(actual.delta_bic, expected_delta)
        assert actual.timing[actual.family].fallback == "fixed_zero"


def test_admission_detection_boundaries_and_strict_residual_gain() -> None:
    times = np.arange(70) / 30
    aif = 5 * np.exp(-(((times - 0.4) / 0.15) ** 2))
    context = NativeInput(aif, times, 1.0)
    p = np.array([0.2, 0.3, 0.05])
    accepted = exchange_admission(context, p, 1e-5, 1e-7)
    assert accepted.passed and np.isfinite(accepted.se).all()
    assert accepted.rcond > 0
    for direction in (-1.0, 1.0):
        cx = 1e-5 / np.exp((np.log(len(times)) + direction * 1e-8) / len(times))
        gate = exchange_admission(context, p, 1e-5, cx)
        assert gate.g1 and gate.g2 == (direction > 0)
    noisy = exchange_admission(context, p, 10.0, 0.01)
    assert not noisy.g1 and noisy.g2
    bound = exchange_admission(context, np.array([0.2, 0.3, 0.0]), 1e-5, 1e-7)
    assert not bound.g1 and "unavailable" in bound.reason


@pytest.mark.parametrize(
    "dt,eligible", [(2.0, True), (2.0 + 1e-6, True), (2.0 + 1.1e-6, False)]
)
def test_automatic_cap_and_explicit_family_independence(
    dt: float, eligible: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = {
        name: _profile(name, [[sse]])
        for name, sse in [("patlak", 1.0), ("extended_tofts", 0.1), ("2cxm", 0.001)]
    }
    monkeypatch.setattr(
        selection,
        "exchange_admission",
        lambda *args: ExchangeAdmission(
            True, True, "admitted", np.ones(3), 1.0, 0.0, 10.0
        ),
    )
    times, aif = np.arange(20) * dt / 60, np.ones(20)
    automatic = selection.select_pool(profiles, aif, times, np.array([0]), 0)
    assert automatic.family == ("2cxm" if eligible else "extended_tofts")
    explicit = selection.select_pool(profiles, aif, times, np.array([0]), 0, "2cxm")
    assert explicit.family == "2cxm" and explicit.admission is None
    profiles["2cxm"].status[0, 0] = 0
    failed = selection.select_pool(profiles, aif, times, np.array([0]), 0, "2cxm")
    assert (
        failed.family == "2cxm"
        and failed.timing["2cxm"].reason == "selected_nonconverged"
    )


def test_mismatched_observations_and_incomplete_automatic_candidates() -> None:
    profiles = {name: _profile(name, [[1.0]]) for name in ("patlak", "extended_tofts")}
    aif, times, pool = np.ones(21), np.arange(21) / 12, np.array([0])
    profiles["extended_tofts"].observed += 1
    with pytest.raises(ValueError, match="observation identity"):
        selection.select_pool(profiles, aif, times, pool, 0)
    profiles["extended_tofts"].observed -= 1
    for profile in profiles.values():
        profile.sse[0, 0] = np.nan
    unavailable = selection.select_pool(profiles, aif, times, pool, 0)
    assert unavailable.family is None
    assert all(
        value.reason == "incomplete_family_profile"
        for value in unavailable.timing.values()
    )
    with pytest.raises(ValueError, match="Distinct"):
        selection.family_timing(
            profiles["patlak"], np.array([0, 0]), 0, frame_interval_seconds=2.0
        )
