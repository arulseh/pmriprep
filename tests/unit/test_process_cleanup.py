"""Regression coverage for PMRIPrep-owned child-process cleanup."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pmriprep.interfaces.hdglio import _terminate_process_group


class _FakeProcess:
    """Minimal ``Popen`` stand-in recording waits."""

    pid = 1234

    def __init__(self, *, time_out_once: bool = False) -> None:
        self.time_out_once = time_out_once
        self.wait_calls: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.time_out_once:
            self.time_out_once = False
            raise subprocess.TimeoutExpired("hd_glio_predict", timeout)
        return 0


def test_process_group_cleanup_terminates_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graceful cleanup targets the group and waits for its leader."""
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: sent.append((pid, sig)))
    process = _FakeProcess()

    _terminate_process_group(process, timeout=2.0)

    assert sent == [(1234, signal.SIGTERM)]
    assert process.wait_calls == [2.0]


def test_process_group_cleanup_escalates_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child ignoring SIGTERM is killed and then reaped without a timeout."""
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: sent.append((pid, sig)))
    process = _FakeProcess(time_out_once=True)

    _terminate_process_group(process, timeout=2.0)

    assert sent == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]
    assert process.wait_calls == [2.0, None]


def test_process_group_cleanup_reaps_already_exited_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exit race still performs a final wait to collect child status."""
    def _missing_process(pid: int, sig: signal.Signals) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("os.killpg", _missing_process)
    process = _FakeProcess()

    _terminate_process_group(process)

    assert process.wait_calls == [None]


@pytest.mark.parametrize(
    "relative_path,expected_count",
    [
        ("pmriprep/interfaces/dsc_compute.py", 1),
        ("pmriprep/interfaces/dce_compute.py", 3),
    ],
)
def test_fitting_executors_join_workers(
    relative_path: str,
    expected_count: int,
) -> None:
    """Fitting pools synchronously join workers on every exit path."""
    root = Path(__file__).parents[2]
    source = (root / relative_path).read_text()

    assert "shutdown(wait=False" not in source
    assert source.count("shutdown(wait=True, cancel_futures=True)") == expected_count


def test_hdglio_interrupt_cleans_isolated_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """HD-GLIO interruption cleans its isolated group and remains an interrupt."""
    from pmriprep.interfaces.hdglio import HDGLIOSegment

    popen_kwargs: dict[str, Any] = {}
    cleaned: list[Any] = []
    process = _FakeProcess()

    def _communicate() -> tuple[str, str]:
        raise KeyboardInterrupt

    process.communicate = _communicate  # type: ignore[attr-defined]

    def _popen(*args: Any, **kwargs: Any) -> _FakeProcess:
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(
        "pmriprep.interfaces.hdglio._terminate_process_group",
        lambda child: cleaned.append(child),
    )
    monkeypatch.setattr(HDGLIOSegment, "_get_output_dir", lambda self: tmp_path)
    monkeypatch.setattr(HDGLIOSegment, "_build_command", lambda self: "hd_glio_predict")

    with pytest.raises(KeyboardInterrupt):
        HDGLIOSegment()._run_interface(object())

    assert popen_kwargs["start_new_session"] is True
    assert cleaned == [process]


def test_hdglio_cleanup_failure_preserves_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cleanup errors do not replace the interruption that initiated cleanup."""
    from pmriprep.interfaces.hdglio import HDGLIOSegment

    process = _FakeProcess()

    def _communicate() -> tuple[str, str]:
        raise KeyboardInterrupt

    def _cleanup_failure(child: Any) -> None:
        raise RuntimeError("cleanup failed")

    process.communicate = _communicate  # type: ignore[attr-defined]
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        "pmriprep.interfaces.hdglio._terminate_process_group",
        _cleanup_failure,
    )
    monkeypatch.setattr(HDGLIOSegment, "_get_output_dir", lambda self: tmp_path)
    monkeypatch.setattr(HDGLIOSegment, "_build_command", lambda self: "hd_glio_predict")

    with pytest.raises(KeyboardInterrupt):
        HDGLIOSegment()._run_interface(object())
