"""Compiled 2CXM equations, optimizer and native memory-boundary checks."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from pmriprep.models.dce._native_backend import Evaluator, solve, svd
from pmriprep.models.dce.fit_errors import NumericalFitError
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.native_exchange import (
    NativeInput,
    decode,
    encode,
    fit_native,
    predict,
)


def _context(delay: float = 3.5) -> NativeInput:
    times = np.arange(100) / 30.0
    aif = 0.03 + 5 * np.exp(-(((times - 0.6) / 0.15) ** 2)) + 0.2 * (times > 0.8)
    return NativeInput(aif, times, delay)


def test_native_prediction_and_fit_match_independent_exchange_odes() -> None:
    """Check physical units and model recovery without using its eigenvalue formula."""
    context = _context()
    truth = np.array([60.0, 12.0, 0.3, 0.06])
    fp, ps, ve, vp = truth
    fp, ps = fp / 100.0, ps / 100.0

    def rhs(t: float, state: np.ndarray) -> list[float]:
        cp = float(context.poly(np.clip(t - context.delay, 0.0, context.t[-1])))
        plasma, tissue = state
        return [
            fp * cp - (fp + ps) / vp * plasma + ps / ve * tissue,
            ps / vp * plasma - ps / ve * tissue,
        ]

    ode = solve_ivp(
        rhs,
        [0.0, context.t[-1]],
        [0.0, 0.0],
        t_eval=context.t,
        rtol=1e-11,
        atol=1e-12,
        max_step=context.h / 5,
    )
    assert ode.success
    curve = ode.y.sum(axis=0)
    evaluator = Evaluator(context, curve)
    native = evaluator.evaluate(encode(truth))[0]
    np.testing.assert_allclose(native, curve, rtol=1e-7, atol=2e-9)
    fit = fit_native(context, curve)
    assert fit.status > 0 and fit.sse < 1e-12
    np.testing.assert_allclose(fit.params, truth, rtol=1e-4, atol=1e-5)


def test_native_jacobian_and_single_start_match_scipy() -> None:
    context = _context()
    curve = predict(context, np.array([60.0, 12.0, 0.3, 0.06]))
    evaluator = Evaluator(context, curve)
    x = np.array([0.8, 0.2, 0.4, 0.1])
    pred, _, jac = evaluator.evaluate(x, True)
    np.testing.assert_allclose(pred, predict(context, decode(x)), atol=1e-13)
    delta = 1e-5 * np.maximum(abs(x), 0.01)
    expected = np.column_stack(
        [
            (
                predict(context, decode(x + np.eye(4)[j] * delta[j]))
                - predict(context, decode(x))
            )
            / delta[j]
            / evaluator.scale
            for j in range(4)
        ]
    )
    np.testing.assert_allclose(jac, expected, rtol=1e-7, atol=2e-9)
    native = solve(evaluator, x)
    reference = least_squares(
        evaluator.residual,
        x,
        jac=evaluator.jacobian,
        bounds=(np.zeros(4), [5.0, 5.0, 1.0, 1.0]),
        x_scale="jac",
        ftol=2e-9,
        xtol=2e-9,
        gtol=2e-9,
        max_nfev=180,
    )
    assert native.status > 0 and reference.status > 0
    np.testing.assert_allclose(native.x, reference.x, atol=1e-6)
    np.testing.assert_allclose(native.cost, reference.cost, atol=1e-12)


@pytest.mark.parametrize("delay", [0.0, 3.5, 12.0])
def test_native_input_preserves_history_and_owns_outputs(delay: float) -> None:
    base = _context(delay)
    ids = np.array([1, 8, 23, 70, 99])
    native = NativeInput(base.aif, base.t, delay, ids)
    common = DelayedPchip(base.aif, base.t, delay, ids)
    for name in ("coefficients", "k", "widths", "knots", "partial"):
        np.testing.assert_array_equal(getattr(native, name), getattr(common, name))
    rates = np.array([0.0, 1e-12, 0.5, 30.0, 1e5])
    np.testing.assert_array_equal(
        DelayedPchip.convolution(native, rates), common.convolution(rates)
    )
    first = native.convolution(rates)
    saved = first.copy()
    native.convolution(rates * 2)
    np.testing.assert_array_equal(first, saved)
    np.testing.assert_allclose(first, common.convolution(rates), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first, base.convolution(rates)[:, ids], atol=1e-12)
    with pytest.raises(ValueError, match="nonnegative delay"):
        NativeInput(base.aif, base.t, -1.0)


def test_reused_evaluator_outputs_and_budget_are_independent() -> None:
    context = _context()
    y = predict(context, np.array([60.0, 12.0, 0.3, 0.06]))
    evaluator = Evaluator(context, y)
    x = np.array([0.6, 0.12, 0.3, 0.1])
    untouched = x.copy()
    one = solve(evaluator, x, max_nfev=1, trace=True)
    saved = one.x.copy()
    assert one.status == 0 and one.nfev == 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: solve(evaluator, x).x, range(4)))
    for value in results[1:]:
        np.testing.assert_array_equal(value, results[0])
    np.testing.assert_array_equal(one.x, saved)
    np.testing.assert_array_equal(x, untouched)
    bound = solve(evaluator, np.array([0.0, 5.0, 0.0, 1.0]), max_nfev=1)
    np.testing.assert_array_equal(bound.x, [1e-10, 5.0 - 5e-10, 1e-10, 1.0 - 1e-10])


def test_native_boundary_rejects_invalid_shapes_indices_and_controls() -> None:
    context = _context()
    evaluator = Evaluator(context, np.zeros(len(context.t)))
    for initial in (
        np.zeros(3),
        np.array([np.nan, 1.0, 1.0, 1.0]),
        np.array([6.0, 1.0, 1.0, 1.0]),
    ):
        with pytest.raises(ValueError, match="initial coordinates"):
            solve(evaluator, initial)
    for kwargs in ({"max_nfev": 0}, {"max_nfev": 1.5}, {"ftol": np.nan}):
        with pytest.raises(ValueError):
            solve(evaluator, np.ones(4), **kwargs)
    with pytest.raises(ValueError, match="curve matching"):
        Evaluator(context, np.zeros(len(context.t) - 1))
    with pytest.raises(ValueError, match="previous"):
        fit_native(context, np.zeros(len(context.t)), previous=np.zeros(3))
    context._interval_index[0] = len(context.t)
    with pytest.raises(ValueError, match="indices"):
        Evaluator(context, np.zeros(len(context.t)))


def test_native_svd_uses_installed_lapack_with_correct_layout() -> None:
    matrix = np.random.default_rng(4).normal(size=(12, 4))
    u, singular, vt = svd(matrix)
    np.testing.assert_allclose((u * singular) @ vt, matrix, atol=1e-13)
    np.testing.assert_allclose(
        singular, np.linalg.svd(matrix, compute_uv=False), atol=1e-13
    )
    for invalid in (np.ones((2, 4)), np.full((8, 4), np.nan)):
        with pytest.raises(ValueError):
            svd(invalid)


def test_native_errors_distinguish_local_numerics_from_backend_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pmriprep.models.dce import _native_backend as backend

    context = _context()
    evaluator = Evaluator(context, np.ones(len(context.t)))
    for code in (-13, -14, -15, -10, -11, -12, -99):
        monkeypatch.setattr(backend, "SOLVE", lambda *args, code=code: code)
        expected = (
            NumericalFitError if code in (-13, -14, -15)
            else MemoryError if code == -11 else RuntimeError
        )
        with pytest.raises(expected) as caught:
            solve(evaluator, np.ones(4))
        assert type(caught.value) is expected

    def nonfinite(*args: object) -> int:
        for i in range(4):
            args[9][i] = np.nan
        for i in range(7):
            args[10][i] = 0.0
        return 0

    monkeypatch.setattr(backend, "SOLVE", nonfinite)
    with pytest.raises(NumericalFitError, match="Nonfinite native result"):
        solve(evaluator, np.ones(4))


@pytest.mark.parametrize("interface", ["hyp1f1", "lapack"])
def test_cli_rejects_incompatible_native_interface_before_bids_indexing(
    interface: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pmriprep.cli import run
    from pmriprep.models.dce import _native_backend as backend

    args = run._build_parser().parse_args(
        [str(tmp_path / "bids"), str(tmp_path / "out"), "participant"]
    )
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(run.plogger, "init", lambda **kwargs: None)
    messages = []
    monkeypatch.setattr(run.logger, "error", lambda *values: messages.append(values))
    monkeypatch.setattr("bids.layout.BIDSLayout",
                        lambda *a, **kw: pytest.fail("input indexing started"))
    special = backend.cython_special.__pyx_capi__
    lapack = backend.cython_lapack.__pyx_capi__
    capsules, key, wrong_signature = (
        (special, "__pyx_fuse_1hyp1f1", lapack["dgesdd"])
        if interface == "hyp1f1"
        else (lapack, "dgesdd", special["__pyx_fuse_1hyp1f1"])
    )
    for replacement in (None, wrong_signature):
        if replacement is None:
            monkeypatch.delitem(capsules, key)
        else:
            monkeypatch.setitem(capsules, key, replacement)
        with pytest.raises(SystemExit) as error:
            run.run_pmriprep_wf(args)
        assert error.value.code == 1
        assert "Native DCE backend unavailable" in messages[-1][0]
        assert "SciPy" in str(messages[-1][1])


@pytest.mark.parametrize("option", ["--skip-dce", "--report-only"])
def test_cli_without_dce_does_not_require_native_backend(
    option: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pmriprep.cli import run
    from pmriprep.models.dce import _native_backend as backend

    args = run._build_parser().parse_args(
        [str(tmp_path / "bids"), str(tmp_path / "out"), "participant", option]
    )
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(run.plogger, "init", lambda **kwargs: None)
    monkeypatch.setattr(backend, "check_backend",
                        lambda: pytest.fail("DCE backend checked despite skip"))

    class IndexingReached(Exception):
        pass

    def stop_at_indexing(*args: object, **kwargs: object) -> None:
        raise IndexingReached

    monkeypatch.setattr("bids.layout.BIDSLayout", stop_at_indexing)
    with pytest.raises(IndexingReached):
        run.run_pmriprep_wf(args)
