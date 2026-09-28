"""Portable native preparation parity, buffer lifetime and fallback checks."""

from __future__ import annotations

import gc
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi import native
from pmriprep.models.dsc.signal_cpi.continuous import ContinuousCPIConvolution
from pmriprep.models.dsc.signal_cpi.execution import CPIExecutor, plan_resources
from pmriprep.models.dsc.signal_cpi.transport import DispersedAIF, GammaEndpointCache


@pytest.fixture
def kernel() -> Any:
    if native._kernel is None:
        pytest.skip('compiled preparation requires a package build')
    assert hasattr(native._kernel, 'interval_layout')
    assert hasattr(native._kernel, 'contract_moments')
    return native._kernel


@pytest.mark.parametrize('shape', [1., 1.0025, 1.225, 5., 601.])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_intervals_moments_and_roundoff_match_python(
    kernel: Any, monkeypatch: pytest.MonkeyPatch, shape: float, cached: bool, scale: float,
) -> None:
    rng = np.random.default_rng(5001)
    time = np.r_[0., np.cumsum(rng.uniform(.05, 2., 20))]
    base = ContinuousCPIConvolution(scale*rng.normal(size=len(time)), time, aif_interpolation='pchip')
    prepared = DispersedAIF(base, (shape-1)/20., 20., 50.,
                             endpoint_cache=GammaEndpointCache() if cached else None)
    target = np.r_[-3., 0., rng.uniform(0., 70., 137), time, np.nextafter(time, np.inf)]
    actual = native.interval_layout(target, base.left, base.right,
                                    prepared.kernel_lower, prepared.kernel_upper)
    assert actual is not None
    rows, segments, lo, hi, bits = actual
    old_lo = np.maximum(0., target[:, None]-base.right[None])
    old_hi = np.minimum(prepared.kernel_upper, np.maximum(0., target[:, None]-base.left[None]))
    active = (old_hi > old_lo) & (old_hi > prepared.kernel_lower)
    expected_rows, expected_segments = np.nonzero(active)
    for a, b in zip((rows, segments, lo, hi),
                    (expected_rows, expected_segments, old_lo[active], old_hi[active]), strict=True):
        np.testing.assert_array_equal(a, b)
        assert not a.flags.writeable
    assert bits == np.packbits(active).tobytes()
    actual_value, actual_error = prepared._moment_blocks(target, rows, segments, lo, hi, (b'parity',))
    with monkeypatch.context() as patch:
        patch.setattr(native, 'contract_moments', lambda *a: None)
        expected, old_error = prepared._moment_blocks(target, rows, segments, lo, hi, (b'parity',))
    assert np.all(abs(actual_value-expected) <= actual_error+old_error)
    np.testing.assert_allclose(actual_error, old_error, rtol=2e-15, atol=0.)
    budget = prepared.raw_tolerance-2*prepared.tail_probability*prepared.scale
    threshold = budget/np.maximum(1, np.bincount(rows, minlength=len(target))[rows])
    np.testing.assert_array_equal(actual_error <= threshold, old_error <= threshold)


@pytest.mark.parametrize('mode,rate', [(0., .001), (.05, .05), (.15, 1.5), (2., 2.), (30., 20.)])
@pytest.mark.parametrize('interpolation', ['linear', 'pchip'])
@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_whole_preparation_matches_python(
    kernel: Any, monkeypatch: pytest.MonkeyPatch, mode: float, rate: float,
    interpolation: str, scale: float,
) -> None:
    time = np.array([0., .4, 1.2, 3., 7., 15.])
    base = ContinuousCPIConvolution(scale*np.array([.2, -.4, 5., 2., -.1, .3]), time,
                                    aif_interpolation=interpolation)
    target = np.r_[-3., 0., np.linspace(.001, 50., 47)]
    knots, values = np.array([0., .8, 4., 15.]), np.array([1., .8, .3, .1])
    with monkeypatch.context() as patch:
        patch.setattr(native, 'interval_layout', lambda *a: None)
        patch.setattr(native, 'contract_moments', lambda *a: None)
        reference = DispersedAIF(base, mode, rate, 50., endpoint_cache=GammaEndpointCache())
        raw = reference._raw(target)
        curve = reference.convolve(knots, values, target)
    candidate = DispersedAIF(base, mode, rate, 50., endpoint_cache=GammaEndpointCache())
    np.testing.assert_allclose(candidate._raw(target)/reference.scale, raw/reference.scale,
                               rtol=0, atol=1e-11)
    np.testing.assert_allclose(candidate.convolve(knots, values, target)/(reference.scale*50),
                               curve/(reference.scale*50), rtol=0, atol=1e-11)
    for name in ('raw_points', 'analytic_intervals', 'quadrature_intervals'):
        assert getattr(candidate, name) == getattr(reference, name)
    np.testing.assert_array_equal(candidate.left, reference.left)
    np.testing.assert_array_equal(candidate.right, reference.right)


def moment_arguments() -> list[np.ndarray]:
    """Return typed, independently owned inputs with simple constant moments."""
    coefficients = np.zeros((4, 3))
    coefficients[3] = 1.
    return [np.array([1., 2., 3.]), np.arange(3, dtype=np.intp), coefficients,
            np.ones(4), np.zeros((4, 3)), np.full((4, 3), .25),
            np.ones((4, 3)), np.full((4, 3), .75)]


@pytest.mark.parametrize('index', range(8))
@pytest.mark.parametrize('kind', ['rank', 'dtype', 'unaligned', 'strided', 'endian'])
def test_moment_buffers_are_checked(kernel: Any, index: int, kind: str) -> None:
    args = moment_arguments()
    value = args[index]
    if kind == 'rank':
        args[index] = value.reshape(-1, 1, 1)
    elif kind == 'dtype':
        args[index] = value.astype(np.uintp if index == 1 else np.float32)
    elif kind == 'unaligned':
        args[index] = np.ndarray(value.shape, dtype=value.dtype,
                                 buffer=bytearray(value.nbytes+1), offset=1)
        args[index][:] = value
    elif kind == 'endian':
        args[index] = value.astype(value.dtype.newbyteorder('S'))
    else:
        args[index] = np.zeros((*value.shape[:-1], 6), dtype=value.dtype)[..., ::2]
    with pytest.raises((ValueError, BufferError)):
        kernel.contract_moments(*args)


@pytest.mark.parametrize('case', ['negative', 'large', 'segments', 'coefficients', 'factors', 'cdf'])
def test_moment_domain_is_checked_before_indexing(kernel: Any, case: str) -> None:
    args = moment_arguments()
    if case in ('negative', 'large'):
        args[1][0] = -1 if case == 'negative' else 3
    elif case == 'segments':
        args[1] = args[1][:2]
    elif case == 'coefficients':
        args[2] = args[2][:3]
    elif case == 'factors':
        args[3] = args[3][:3]
    else:
        args[4] = np.zeros((4, 4))
    with pytest.raises(ValueError):
        kernel.contract_moments(*args)


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('kind', ['rank', 'dtype', 'nan', 'strided', 'unaligned'])
def test_layout_buffers_are_checked(kernel: Any, index: int, kind: str) -> None:
    args = [np.arange(3.), np.arange(3.), np.arange(3.)+1]
    if kind == 'rank':
        args[index] = args[index][None]
    elif kind == 'dtype':
        args[index] = args[index].astype(np.float32)
    elif kind == 'nan':
        args[index][0] = np.nan
    elif kind == 'strided':
        args[index] = np.arange(6.)[::2]
    else:
        args[index] = np.ndarray((3,), dtype=np.float64, buffer=bytearray(25), offset=1)
        args[index][:] = np.arange(3.)
    with pytest.raises((ValueError, BufferError)):
        kernel.interval_layout(*args, 0., 10.)


@pytest.mark.parametrize('lower,upper', [(-1., 1.), (1., 1.), (2., 1.), (0., np.inf), (np.nan, 1.)])
def test_layout_cutoffs_are_checked(kernel: Any, lower: float, upper: float) -> None:
    with pytest.raises(ValueError, match='cutoffs'):
        kernel.interval_layout(np.arange(3.), np.arange(3.), np.arange(3.)+1, lower, upper)


def test_layout_edges_and_empty_domains(kernel: Any) -> None:
    time = np.arange(9.)
    for right in (time[1:-1], time[:-1]):
        with pytest.raises(ValueError):
            kernel.interval_layout(time, time[:-1], right, 0., 100.)
    for target in (np.array([]), np.array([-1e300, -1., 0., 1e300])):
        result = native.interval_layout(target, time[:-1], time[1:], 1e-20, 100.)
        assert result is not None
        assert all(a.size == 0 for a in result[:4])
        assert result[4] == bytes((len(target)*8+7)//8)


def test_cdf_layouts_ownership_and_buffer_release(kernel: Any) -> None:
    args = moment_arguments()
    # Different C/F layouts can be read in the same call, including readonly inputs.
    for i in (4, 6):
        args[i] = np.asfortranarray(args[i])
    for a in args:
        a.flags.writeable = False
    before = [sys.getrefcount(a) for a in args]
    for _ in range(10):
        actual = native.contract_moments(*args)
        assert actual is not None
        np.testing.assert_array_equal(actual[0], .25)
        with pytest.raises(ValueError):
            kernel.contract_moments(*args[:3], np.empty(0), *args[4:])
    assert [sys.getrefcount(a) for a in args] == before
    del args
    gc.collect()
    assert all(not a.flags.writeable and isinstance(a.base, bytes) for a in actual)
    np.testing.assert_array_equal(actual[0], .25)
    arrays = [np.arange(3.), np.arange(3.), np.arange(3.)+1]
    before = [sys.getrefcount(a) for a in arrays]
    for _ in range(10):
        kernel.interval_layout(*arrays, 0., 10.)
        with pytest.raises(ValueError):
            kernel.interval_layout(arrays[0], arrays[1], arrays[1], 0., 10.)
    assert [sys.getrefcount(a) for a in arrays] == before


def test_dispatch_layout_copy_dtype_fallback_and_empty_moments(kernel: Any) -> None:
    args = moment_arguments()
    for i in range(4, 8):
        args[i] = np.repeat(args[i], 2, axis=1)[:, ::2]
    result = native.contract_moments(*args)
    assert result is not None
    np.testing.assert_array_equal(result[0], .25)
    args[0] = args[0].astype(np.float32)
    assert native.contract_moments(*args) is None
    assert native.interval_layout(np.arange(3, dtype=np.float32), np.arange(3.), np.arange(3.)+1, 0., 10.) is None
    empty = native.contract_moments(np.empty(0), np.empty(0, dtype=np.intp), np.empty((4, 0)),
                                    np.ones(4), *[np.empty((4, 0)) for _ in range(4)])
    assert empty is not None and all(a.shape == (0,) for a in empty)


def test_nonfinite_moments_reach_existing_guard(kernel: Any) -> None:
    args = moment_arguments()
    args[4][0, 0] = np.nan
    result = native.contract_moments(*args)
    assert result is not None
    assert np.isnan(result[0][0]) and np.isnan(result[1][0])


def test_unavailable_and_failed_native_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, '_kernel', None)
    assert native.contract_moments(*moment_arguments()) is None
    assert native.interval_layout(np.arange(3.), np.arange(3.), np.arange(3.)+1, 0., 10.) is None
    def fail(*args: Any) -> None:
        raise RuntimeError('native failure')
    monkeypatch.setattr(native, '_kernel', SimpleNamespace(interval_layout=fail, contract_moments=fail))
    with pytest.raises(RuntimeError, match='native failure'):
        native.contract_moments(*moment_arguments())
    with pytest.raises(RuntimeError, match='native failure'):
        native.interval_layout(np.arange(3.), np.arange(3.), np.arange(3.)+1, 0., 10.)


def _prepare_worker(_: int) -> tuple[int, float]:
    base = ContinuousCPIConvolution(np.ones(10), np.arange(10.))
    prepared = DispersedAIF(base, .15, 1.5, 20.)
    assert native.available()
    return os.getpid(), float(prepared._raw(np.array([5.]))[0])


def test_preparation_in_spawned_workers(kernel: Any) -> None:
    expected = _prepare_worker(0)[1]
    with CPIExecutor(plan_resources(2, 2, 30, 5)) as executor:
        results = list(executor.map(_prepare_worker, range(2), ordered=False))
    assert all(pid != os.getpid() and value == expected for pid, value in results)
