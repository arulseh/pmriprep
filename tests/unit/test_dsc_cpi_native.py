"""Native convolution parity, buffer safety, fallback and spawned-worker checks."""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator
from threadpoolctl import threadpool_info

from pmriprep.models.dsc.signal_cpi import native
from pmriprep.models.dsc.signal_cpi.batch import EnsembleConvolution
from pmriprep.models.dsc.signal_cpi.continuous import ContinuousCPIConvolution, PolynomialIntegral
from pmriprep.models.dsc.signal_cpi.execution import CPIExecutor, plan_resources


@pytest.fixture
def kernel() -> Any:
    if native._kernel is None:
        pytest.skip("compiled kernel requires a package build")
    return native._kernel


@pytest.mark.parametrize("interpolation", ["linear", "pchip"])
@pytest.mark.parametrize("points", [2, 5, 12])
@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_plain_matches_independent_python_and_scalar(
    kernel: Any, interpolation: str, points: int, scale: float,
) -> None:
    rng = np.random.default_rng(92031)
    time = np.r_[0., np.cumsum(rng.uniform(.3, 2., 20))]
    aif = rng.normal(size=time.size)*scale
    aif[3:8] = 0.
    base = ContinuousCPIConvolution(aif, time, aif_interpolation=interpolation)
    engine = EnsembleConvolution(base)
    python_engine = EnsembleConvolution(base)
    knots = np.tile(np.linspace(0., time[-1], points), (8, 1))
    target = np.tile(time, (8, 1))
    target[3] += 3.2
    target[4] -= 2.7
    target[6] += 100.
    target[7] = target[7, ::-1]
    if points > 2:
        knots[5, 1] = np.nextafter(knots[5, 1], np.inf)
    values = np.exp(-knots/rng.uniform(1., 12., (8, 1)))
    envelope = time[-1]*max(scale, np.max(abs(aif)))
    for factor in (1., .8):
        changed = values*factor
        actual = engine._plain(knots, changed, target)
        expected = python_engine._plain_python(knots, changed, target)
        reference = np.stack([base.evaluate_at(x, y, 1., t) for x, y, t in
                              zip(knots, changed, target, strict=True)])
        assert np.max(abs(actual-expected))/envelope <= 2e-12
        assert np.max(abs(actual-reference))/envelope <= 2e-12
        np.testing.assert_array_equal(actual[target <= 0], 0.)
        np.testing.assert_array_equal(engine._plain(knots[::-1], changed[::-1], target[::-1]),
                                       actual[::-1])
        assert engine._plain(knots, changed, target[:, :0]).shape == (8, 0)
    assert not engine._geometry_cache  # Prove these calls exercised compiled dispatch.


@pytest.mark.parametrize("degree,chebyshev", [(1, False), (3, False), (15, True), (23, True)])
@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_prepared_integral_matches_python(kernel: Any, degree: int, chebyshev: bool, scale: float) -> None:
    rng = np.random.default_rng(7301)
    edges = np.r_[0., np.cumsum(rng.uniform(.2, 3., 30))]
    active = rng.random(30) > .2
    left, right = edges[:-1][active], edges[1:][active]
    arterial = rng.normal(size=(degree+1, left.size))*scale
    nodes, weights = np.polynomial.legendre.leggauss((degree+5)//2)
    engine = PolynomialIntegral(left, right, arterial, nodes, weights, chebyshev=chebyshev)
    knots = np.c_[np.zeros(3), np.cumsum(rng.uniform(.01, 12., (3, 4)), axis=1)]
    values = np.exp(-knots/6)
    coefficients = np.stack([np.c_[PchipInterpolator(x, y).c, [0., 0., 0., y[-1]]]
                             for x, y in zip(knots, values, strict=True)])
    target = rng.uniform(-10., edges[-1]+20, (3, 25))
    target[:, :4] = [-5., 0., right[0], 100.]
    for multiplier in (1., .8):
        changed = coefficients*multiplier
        actual = engine.evaluate(knots, changed, target)
        expected = engine._evaluate_python(knots, changed, target)
        assert np.max(abs(actual-expected))/(scale*edges[-1]*100) <= 2e-12
        np.testing.assert_array_equal(actual[target <= 0], 0.)
        assert engine.evaluate(knots, changed, target[:, :0]).shape == (3, 0)


def arguments() -> list[Any]:
    """Return independently owned native buffers with a known constant integral."""
    coefficients = np.zeros((1, 4, 2))
    coefficients[:, 3] = 1.
    return [np.array([[0., 3.]]), coefficients, np.array([[1., 2.]]),
            np.array([0.]), np.array([3.]), np.array([[1.]]), np.array([0.]),
            np.array([2.]), None, np.full((1, 2), -123.), False]


@pytest.mark.parametrize("index", list(range(8)) + [9])
@pytest.mark.parametrize("kind", ["shape", "dtype", "unaligned", "strided"])
def test_direct_extension_rejects_unsafe_buffers(kernel: Any, index: int, kind: str) -> None:
    args = arguments()
    value = args[index]
    if kind == "shape":
        args[index] = value.reshape(-1, 1, 1, 1)
    elif kind == "dtype":
        args[index] = value.astype(np.float32)
    elif kind == "unaligned":
        args[index] = np.ndarray(value.shape, dtype=np.float64,
                                 buffer=bytearray(value.nbytes+1), offset=1)
        args[index][:] = value
    else:
        shape = (*value.shape[:-1], max(2, value.shape[-1])*2)
        args[index] = np.zeros(shape)[..., ::2]
        assert not args[index].flags.c_contiguous
    with pytest.raises((ValueError, BufferError)):
        kernel.convolve(*args)


@pytest.mark.parametrize("index", range(8))
def test_nonfinite_native_input_is_rejected_before_writing(kernel: Any, index: int) -> None:
    args = arguments()
    args[index].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        kernel.convolve(*args)
    np.testing.assert_array_equal(args[9], -123.)


@pytest.mark.parametrize("case", ["knots", "origin", "interval", "moments", "alias", "readonly", "endian"])
def test_domain_and_output_guards(kernel: Any, case: str) -> None:
    args = arguments()
    if case == "knots":
        args[0][0, 1] = 0.
    elif case == "origin":
        args[0][0, 0] = 1.
    elif case == "interval":
        args[4][0] = -1.
    elif case == "moments":
        args[8] = np.ones((3, 1))
    elif case == "alias":
        args[9] = args[2]
    elif case == "readonly":
        args[9].flags.writeable = False
    else:
        args[0] = args[0].astype(np.dtype("float64").newbyteorder("S"))
    with pytest.raises((ValueError, BufferError)):
        kernel.convolve(*args)


def test_buffers_released_after_success_and_failure(kernel: Any) -> None:
    args = arguments()
    before = [sys.getrefcount(a) for a in args]
    for _ in range(10):
        kernel.convolve(*args)
        args[0][0, 0] = 1.
        with pytest.raises(ValueError):
            kernel.convolve(*args)
        args[0][0, 0] = 0.
    assert [sys.getrefcount(a) for a in args] == before
    np.testing.assert_array_equal(args[9], args[2])


def test_dispatch_copies_only_layout_and_preserves_readonly_inputs(kernel: Any) -> None:
    args = arguments()
    expected = args[2].copy()
    for i in range(8):
        args[i].flags.writeable = False
    result = native.convolve(*args[:8])
    np.testing.assert_array_equal(result, expected)
    args[2] = args[2][:, ::-1]
    np.testing.assert_array_equal(native.convolve(*args[:8]), expected[:, ::-1])
    for i in range(8):
        args[i] = args[i].astype(np.float32)
    assert native.convolve(*args[:8]) is None


def test_unavailable_native_uses_python_once_and_keeps_dtype_semantics(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(native, "_kernel", None)
    monkeypatch.setattr(native, "_warned", False)
    caplog.set_level(logging.WARNING)
    time = np.arange(10.)
    engine = EnsembleConvolution(ContinuousCPIConvolution(np.sin(time), time))
    for dtype in (np.float32, np.float64):
        x = np.array([[0., 9.]], dtype=dtype)
        y = np.array([[1., .1]], dtype=dtype)
        t = time[None].astype(dtype)
        np.testing.assert_array_equal(engine._plain(x, y, t), engine._plain_python(x, y, t))
    assert sum("Compiled CPI convolution is unavailable" in r.message for r in caplog.records) == 1


def test_native_failure_is_not_silently_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any) -> None:
        raise RuntimeError("native failure")
    monkeypatch.setattr(native, "_kernel", SimpleNamespace(convolve=fail))
    with pytest.raises(RuntimeError, match="native failure"):
        native.convolve(*arguments()[:8])


def _native_worker(_: int) -> tuple[int, bool, list[float], list[int]]:
    result = native.convolve(*arguments()[:8])
    assert result is not None
    return os.getpid(), native.available(), result.ravel().tolist(), [
        row["num_threads"] for row in threadpool_info()
    ]


def test_spawned_workers_use_native_with_resource_limits(kernel: Any) -> None:
    with CPIExecutor(plan_resources(2, 2, 30, 5)) as executor:
        results = list(executor.map(_native_worker, range(2), ordered=False))
    assert all(pid != os.getpid() and available and values == [1., 2.]
               and all(t == 1 for t in threads) for pid, available, values, threads in results)
