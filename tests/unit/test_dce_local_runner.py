"""Complete-slice support, bounded execution and worker-failure contracts."""

from concurrent.futures import Future

import numpy as np
import pytest

from pmriprep.models.dce import local_runner as runner
from pmriprep.models.dce.delay_profiles import (
    DelayProfile, ExchangeRetry, ProfileFailure, build_delay_profile,
)
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.local_selection import select_pool
from pmriprep.models.dce.native_exchange import ExchangeFit
from pmriprep.models.dce.fixed_inference import selected_fit_result
from pmriprep.models.dce.local_refinement import ConditionalInput
from pmriprep.models.dce.identification import identify_profile_point, profile_excursion


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(35) * 4.76 / 60
    aif = 0.01 + 4 * np.exp(-(((times - 0.4) / 0.12) ** 2))
    observed = np.r_[0:7, 9:35]
    curves = []
    for i in range(5):
        context = DelayedPchip(aif, times, 2.0 + i / 10, observed)
        curves.append((0.12 + i / 30) * context.convolution(0.7) + 0.04 * context.cp)
    coordinates = np.array([[2, 0, 3], [0, 0, 3], [1, 1, 3], [1, 0, 3], [9, 0, 3]])
    return aif, times, np.asarray(curves), coordinates, observed


def _compare_profiles(actual: DelayProfile, expected: DelayProfile) -> None:
    for name in ("delays_seconds", "observed", "status", "seed_index", "evaluations"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("params", "sse", "optimality"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), atol=1e-10, rtol=1e-8,
        )
    assert actual.retries.keys() == expected.retries.keys()
    assert actual.failures == expected.failures
    for key in actual.retries:
        for a, b in zip(actual.retries[key], expected.retries[key], strict=True):
            assert a.budget == b.budget and a.retained == b.retained
            for name in ("source", "candidate"):
                af, bf = getattr(a, name), getattr(b, name)
                assert (af.sse, af.status, af.nfev, af.optimality) == (
                    bf.sse, bf.status, bf.nfev, bf.optimality,
                )
                np.testing.assert_array_equal(af.params, bf.params)


def test_chunk_boundaries_and_reused_pool_do_not_change_slice_results() -> None:
    aif, times, curves, xyz, observed = _data()
    profiles = {}
    for family in ("patlak", "extended_tofts"):
        parts = [
            build_delay_profile(aif, times, curves[start:start + 2], family, observed=observed)
            for start in range(0, len(curves), 2)
        ]
        profile = parts[0]
        for name in ("params", "sse", "status", "evaluations", "optimality", "seed_index"):
            setattr(profile, name, np.concatenate([getattr(p, name) for p in parts], axis=1))
        assert not any(p.retries for p in parts)
        profiles[family] = profile
    serial_refinements = {}
    for processes in (1, 2):
        with runner.LocalDelayRunner(processes, 2) as engine:
            pool = engine._executor
            for z in (3, 7):
                xyz[:, 2] = z
                result = engine.fit_slice(aif, times, curves, xyz, np.eye(4), observed=observed)
                assert engine._executor is pool
                assert set(result.profiles) == {"patlak", "extended_tofts"}
                np.testing.assert_array_equal(result.coordinates, xyz)
                # Support crosses both execution chunks; isolated voxel stays singleton.
                assert len(result.neighborhoods[2]) == 4
                assert result.neighborhoods[-1].tolist() == [4]
                for name, profile in result.profiles.items():
                    _compare_profiles(profile, profiles[name])
                    np.testing.assert_array_equal(profile.params, profiles[name].params)
                    np.testing.assert_array_equal(profile.sse, profiles[name].sse)
                for i, members in enumerate(result.neighborhoods):
                    expected = select_pool(profiles, aif, times, members, i)
                    actual = result.selections[i]
                    assert actual.family == expected.family
                    for name in actual.timing:
                        a, b = actual.timing[name], expected.timing[name]
                        for field in ("index", "reference", "free", "support", "fallback", "status"):
                            assert getattr(a, field) == getattr(b, field)
                        np.testing.assert_allclose(a.bic, b.bic, rtol=0, atol=1e-8)
                selected = []
                for i, choice in enumerate(result.selections):
                    family = choice.family
                    timing = choice.timing[family]
                    profile = result.profiles[family]
                    seed = selected_fit_result(profile, timing, i, aif, times, curves[i])
                    selected.append(ConditionalInput(
                        family, aif, times, observed, curves[i],
                        float(profile.delays_seconds[timing.index]), seed,
                        "vi" if i % 2 else "nlls",
                        profile_excursion(profile, i, timing.index),
                        float(profile.sse[timing.index, i]),
                    ))
                final = engine.refine_selected(selected)
                for i, item in enumerate(final):
                    choice = result.selections[i]
                    expected = identify_profile_point(
                        result.profiles[choice.family], i, choice.timing[choice.family].index, aif, times,
                    )
                    np.testing.assert_array_equal(item.identification.effective, expected.effective)
                    assert item.identification.reasons == expected.reasons
                assert engine._executor is pool
                if processes == 1:
                    serial_refinements[z] = final
                else:
                    for first, second in zip(serial_refinements[z], final, strict=True):
                        np.testing.assert_array_equal(first.identification.effective, second.identification.effective)
                        first, second = first.refinement, second.refinement
                        assert first.fit.parameters == second.fit.parameters
                        assert first.actual_engine == second.actual_engine
                        assert first.fallback == second.fallback
                        np.testing.assert_array_equal(first.fit.fitted_curve, second.fit.fitted_curve)
                        for name in first.fit.uncertainties:
                            np.testing.assert_array_equal(first.fit.uncertainties[name], second.fit.uncertainties[name])
                del result


@pytest.mark.parametrize("family", ["patlak", "tofts", "2cxm"])
def test_explicit_coarse_and_fixed_zero_use_the_requested_family(family: str) -> None:
    aif, times, curves, xyz, observed = _data()
    with runner.LocalDelayRunner(2, 1) as engine:
        for enabled in (True, False):
            grid = np.arange(13.0) if enabled else np.array([0.0])
            result = engine.fit_slice(
                aif, times, curves[:2], xyz[:2], np.eye(4),
                mode=family, observed=observed, sloppy=True, delay_correction=enabled,
            )
            expected = build_delay_profile(
                aif, times, curves[:2], family, observed=observed, delays_seconds=grid,
            )
            assert set(result.profiles) == {family}
            _compare_profiles(result.profiles[family], expected)
            assert all(s.family == family for s in result.selections)
            if not enabled:
                assert result.radius_mm is None
                assert [s.timing[family].support for s in result.selections] == [1, 1]
                assert all(s.timing[family].fallback == "fixed_zero" for s in result.selections)


def test_block_assembly_translates_retry_members_without_dropping_evidence() -> None:
    aif, times, curves, _, observed = _data()
    source = build_delay_profile(
        aif, times, curves[:2], "2cxm", observed=observed, delays_seconds=np.array([0.0]),
    )
    original = ExchangeFit(np.array([50.0, 10.0, 0.3, 0.02]), 2.0, 0, 180, 1.0)
    retried = ExchangeFit(original.params.copy(), 1.0, 1, 20, 0.1)
    evidence = (ExchangeRetry(180, original, retried, True),)
    source.retries[(0, 1)] = evidence
    failure = ProfileFailure(0, "retry", "isolated numerical failure")
    source.failures[1] = failure
    destination = {}
    runner._scatter_block(destination, {"2cxm": source}, 0, 4)
    runner._scatter_block(destination, {"2cxm": source}, 2, 4)
    assert destination["2cxm"].retries == {(0, 1): evidence, (0, 3): evidence}
    assert destination["2cxm"].failures == {1: failure, 3: failure}
    np.testing.assert_array_equal(destination["2cxm"].seed_index[:, 2:], source.seed_index)


def test_pending_work_is_bounded_and_worker_failure_aborts_the_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"outstanding": 0, "peak": 0, "submitted": 0, "terminated": False}

    class CountedFuture(Future):
        def result(self, timeout: float | None = None) -> dict:
            state["outstanding"] -= 1
            return super().result(timeout)

    class Executor:
        def __init__(self, **kwargs: object) -> None:
            pass

        def submit(self, function: object, block: object) -> CountedFuture:
            state["outstanding"] += 1
            state["submitted"] += 1
            state["peak"] = max(state["peak"], state["outstanding"])
            future = CountedFuture()
            if state["submitted"] == 3:
                future.set_exception(RuntimeError("worker defect"))
            else:
                future.set_result(function(block))
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(runner, "terminate_process_pool", lambda pool: state.update(terminated=True))
    aif, times, curves, xyz, observed = _data()
    with runner.LocalDelayRunner(2, 1) as engine:
        with pytest.raises(RuntimeError, match="starting at 2") as caught:
            engine.fit_slice(
                aif, times, curves, xyz, np.eye(4), mode="patlak",
                observed=observed, delay_correction=False,
            )
        assert "worker defect" in str(caught.value.__cause__)
        assert state["peak"] == 2 and state["submitted"] == 4
        assert state["terminated"]
        with pytest.raises(RuntimeError, match="active"):
            engine.fit_slice(aif, times, curves, xyz, np.eye(4), observed=observed)
