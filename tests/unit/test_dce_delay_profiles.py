"""Conditional-profile assembly and retry ordering, without selection policy."""

import numpy as np
import pytest

from pmriprep.models.dce import delay_profiles as profiles
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.cached_input import CachedPchip
from pmriprep.models.dce.fit_errors import NumericalFitError
from pmriprep.models.dce.native_exchange import (
    ExchangeFit,
    NativeInput,
    fit_native,
    predict,
)
from pmriprep.models.dce.profile_fitting import (
    fit_extended_tofts,
    fit_patlak,
    fit_tofts,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(45) / 30
    aif = 0.02 + 4 * np.exp(-(((times - 0.35) / 0.12) ** 2))
    observed = np.r_[0:8, 10:45]
    context = NativeInput(aif, times, 0.1, observed)
    curves = np.array([predict(context, np.array([70.0, 15.0, 0.3, 0.04]))])
    return aif, times, curves, observed


@pytest.mark.parametrize("family", ["patlak", "tofts", "extended_tofts", "2cxm"])
def test_profiles_equal_direct_fits_with_full_input_history(family: str) -> None:
    aif, times, curves, observed = _inputs()
    grid = np.array([0.0, 0.1, 0.2])
    actual = profiles.build_delay_profile(
        aif, times, curves, family, observed=observed, delays_seconds=grid
    )
    np.testing.assert_array_equal(actual.observed, observed)
    np.testing.assert_array_equal(actual.delays_seconds, grid)
    previous = None
    for i, delay in enumerate(grid):
        if family == "2cxm":
            fit = fit_native(
                NativeInput(aif, times, delay, observed), curves[0], previous
            )
            previous = fit.params.copy()
            assert actual.evaluations[i, 0] == fit.nfev
            assert actual.seed_index[i, 0] == i - 1
            assert actual.status[i, 0] == fit.status > 0
            assert actual.optimality[i, 0] == fit.optimality
            np.testing.assert_array_equal(actual.params[i, 0], fit.params)
        else:
            fitter = {
                "patlak": fit_patlak,
                "tofts": fit_tofts,
                "extended_tofts": fit_extended_tofts,
            }[family]
            input_type = DelayedPchip if family == "patlak" else CachedPchip
            fit = fitter(input_type(aif, times, delay, observed), curves)
            np.testing.assert_array_equal(actual.params[i], fit.params)
            assert actual.evaluations[i, 0] == -1
            assert actual.status[i, 0] == 1
        np.testing.assert_array_equal(actual.sse[i], np.atleast_1d(fit.sse))
    assert not actual.retries


def test_zero_default_coarse_and_empty_profiles_preserve_grid_identity() -> None:
    aif, times, curves, observed = _inputs()
    dense = profiles.build_delay_profile(
        aif, times, curves, "patlak", observed=observed
    )
    zero = profiles.build_delay_profile(
        aif, times, curves, "patlak", observed=observed, delays_seconds=np.array([0.0])
    )
    coarse = profiles.build_delay_profile(
        aif, times, curves, "patlak", observed=observed, delays_seconds=np.arange(13.0)
    )
    np.testing.assert_array_equal(dense.delays_seconds, np.arange(121) / 10)
    np.testing.assert_array_equal(zero.params[0], dense.params[0])
    np.testing.assert_array_equal(coarse.params, dense.params[::10])
    empty = profiles.build_delay_profile(aif, times, np.empty((0, len(times))), "2cxm")
    assert empty.params.shape == (121, 0, 4) and not empty.retries


def test_repairs_follow_sweep_and_never_change_later_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aif, times, curve, observed = _inputs()
    calls = []
    returned = []

    def fake(
        context: NativeInput,
        y: np.ndarray,
        previous: np.ndarray | None = None,
        max_nfev: int = 180,
    ) -> ExchangeFit:
        index = len(calls)
        calls.append(
            (context.delay, None if previous is None else previous.copy(), max_nfev)
        )
        if index < 6:
            fit = ExchangeFit(
                np.array([20.0 + index, 10.0, 0.2, 0.02]),
                5.0,
                0 if index in (0, 3) else 1,
                10,
                0.5,
            )
        else:
            sse, status = [(4.0, 0), (4.5, 1), (5.0, 1)][index - 6]
            fit = ExchangeFit(
                np.array([70.0 + index, 10.0, 0.2, 0.02]), sse, status, 20, 0.2
            )
        returned.append(fit)
        return fit

    monkeypatch.setattr(profiles, "fit_native", fake)
    actual = profiles.build_delay_profile(
        aif,
        times,
        np.repeat(curve, 2, axis=0),
        "2cxm",
        observed=observed,
        delays_seconds=np.array([0.0, 0.1, 0.2]),
    )
    assert len(calls) == 9
    for i in range(2, 6):
        np.testing.assert_array_equal(calls[i][1], returned[i - 2].params)
    np.testing.assert_array_equal(calls[6][1], returned[0].params)
    np.testing.assert_array_equal(calls[7][1], returned[6].params)
    np.testing.assert_array_equal(calls[8][1], returned[3].params)
    assert [c[2] for c in calls[6:]] == [180, 720, 180]
    assert actual.status[0, 0] == 0 and actual.sse[0, 0] == 4.0
    assert actual.status[1, 1] == 1 and actual.sse[1, 1] == 5.0
    assert [a.retained for a in actual.retries[(0, 0)]] == [True, False]
    assert actual.retries[(1, 1)][0].retained
    np.testing.assert_array_equal(actual.evaluations, np.full((3, 2), 10))
    np.testing.assert_array_equal(actual.seed_index, [[-1, -1], [0, 0], [1, 1]])


def test_retry_classifies_nonfinite_result_and_does_not_retry_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aif, times, curves, observed = _inputs()
    context = NativeInput(aif, times, 0.0, observed)
    initial = ExchangeFit(np.array([40.0, 10.0, 0.2, 0.02]), 1.0, 0, 10, 0.1)
    calls = []

    def unusable(*args: object, **kwargs: object) -> ExchangeFit:
        calls.append(kwargs)
        return ExchangeFit(initial.params.copy(), np.nan, 1, 5, 0.1)

    monkeypatch.setattr(profiles, "fit_native", unusable)
    with pytest.raises(NumericalFitError, match="budget 180: nonfinite"):
        profiles.retry_exchange_fit(context, curves[0], initial)
    converged = ExchangeFit(initial.params.copy(), initial.sse, 1, 10, 0.1)
    best, attempts = profiles.retry_exchange_fit(context, curves[0], converged)
    assert best is converged and not attempts and len(calls) == 1


def test_invalid_profiles_and_native_errors_do_not_become_flat_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aif, times, curves, observed = _inputs()
    for grid in ([], [1.0], [0.0, 0.0], [0.0, np.nan], [0.0, 13.0], [[0.0, 1.0]]):
        with pytest.raises(ValueError, match="Delay grid"):
            profiles.build_delay_profile(
                aif, times, curves, "patlak", delays_seconds=grid
            )
    with pytest.raises(ValueError, match="observation indices"):
        profiles.build_delay_profile(
            aif, times, curves, "patlak", observed=np.array([0, 0])
        )
    with pytest.raises(ValueError, match="common observation"):
        profiles.build_delay_profile(
            aif, times, np.full((1, len(times)), np.nan), "patlak"
        )

    def failed(*args: object, **kwargs: object) -> ExchangeFit:
        raise RuntimeError("solver failure")

    monkeypatch.setattr(profiles, "fit_native", failed)
    with pytest.raises(RuntimeError, match="delay 0 s, member 0: solver failure"):
        profiles.build_delay_profile(aif, times, curves, "2cxm", observed=observed)


def test_local_numerical_failure_invalidates_only_its_family_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pmriprep.models.dce.local_selection import select_pool

    aif, times, curve, observed = _inputs()
    curves = np.repeat(curve, 3, axis=0)
    grid = np.array([0.0, 0.1, 0.2])
    for phase in ("sweep", "retry"):
        calls = []

        def fit(*args: object, **kwargs: object) -> ExchangeFit:
            call = len(calls)
            calls.append(kwargs)
            if (phase == "sweep" and call == 4) or (phase == "retry" and call == 9):
                raise NumericalFitError("isolated numerical failure")
            return ExchangeFit(
                np.array([60.0, 12.0, 0.3, 0.06]),
                1.0, 0 if phase == "retry" and call == 4 else 1, 10, 0.01,
            )

        monkeypatch.setattr(profiles, "fit_native", fit)
        profile = profiles.build_delay_profile(
            aif, times, curves, "2cxm", observed=observed, delays_seconds=grid,
        )
        assert profile.failures.keys() == {1}
        failure = profile.failures[1]
        assert failure.phase == phase and failure.delay_index == 1
        assert "isolated numerical failure" in failure.reason
        assert len(calls) == (8 if phase == "sweep" else 10)
        assert np.isnan(profile.params[:, 1]).all() and np.isnan(profile.sse[:, 1]).all()
        assert (profile.status[:, 1] == 0).all()
        assert np.isfinite(profile.params[:, [0, 2]]).all()
        assert (profile.status[:, [0, 2]] == 1).all()
        selected = select_pool({"2cxm": profile}, aif, times, np.arange(3), 1, "2cxm")
        assert selected.family == "2cxm"
        assert selected.timing["2cxm"].reason == "incomplete_family_profile"
        assert selected.timing["2cxm"].support == 2
        healthy = select_pool({"2cxm": profile}, aif, times, np.arange(3), 0, "2cxm")
        assert healthy.timing["2cxm"].reason is None


def test_nonnative_bad_row_cannot_reenter_pooling_at_a_later_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pmriprep.models.dce.profile_fitting import ProfileFit

    aif, times, curve, observed = _inputs()
    calls = []

    def fit(context: DelayedPchip, y: np.ndarray) -> ProfileFit:
        calls.append(context.delay)
        params = np.tile([0.2, np.nan, 0.03], (len(y), 1))
        sse = np.ones(len(y))
        if len(calls) == 2:
            sse[1] = np.nan
        return ProfileFit(params, sse)

    monkeypatch.setattr(profiles, "fit_patlak", fit)
    actual = profiles.build_delay_profile(
        aif, times, np.repeat(curve, 3, axis=0), "patlak", observed=observed,
        delays_seconds=np.array([0.0, 0.1, 0.2]),
    )
    assert actual.failures.keys() == {1}
    assert np.isnan(actual.sse[:, 1]).all() and (actual.status[:, 1] == 0).all()
    np.testing.assert_array_equal(actual.sse[:, [0, 2]], np.ones((3, 2)))
    assert (actual.status[:, [0, 2]] == 1).all()
