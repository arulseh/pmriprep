"""Regression coverage for PMRIPrep-owned child-process cleanup."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pmriprep.utils.processes import (
    run_isolated_command,
    terminate_process_group,
    terminate_process_pool,
)


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

    terminate_process_group(process, timeout=2.0)  # type: ignore[arg-type]

    assert sent == [(1234, signal.SIGTERM)]
    assert process.wait_calls == [2.0]


def test_process_group_cleanup_escalates_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child ignoring SIGTERM is killed and then reaped without a timeout."""
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: sent.append((pid, sig)))
    process = _FakeProcess(time_out_once=True)

    terminate_process_group(process, timeout=2.0)  # type: ignore[arg-type]

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

    terminate_process_group(process)  # type: ignore[arg-type]

    assert process.wait_calls == [None]


@pytest.mark.parametrize(
    ("relative_path", "cleanup_call"),
    [
        ("pmriprep/interfaces/dsc_compute.py", "terminate_process_pool(executor)"),
        ("pmriprep/interfaces/dce_compute.py", "terminate_process_pool(executor)"),
        ("pmriprep/models/dsc/batched_fitting.py", "terminate_process_pool(pool)"),
    ],
)
def test_fitting_executors_join_workers(
    relative_path: str,
    cleanup_call: str,
) -> None:
    """Fitting pools force active workers down on abnormal exits."""
    root = Path(__file__).parents[2]
    source = (root / relative_path).read_text()

    assert "shutdown(wait=False" not in source
    executor_count = source.count("ProcessPoolExecutor(")
    cleanup_count = source.count(cleanup_call)
    assert executor_count > 0
    assert cleanup_count == executor_count


class _FakePoolProcess:
    """Process-pool worker stand-in with controllable TERM behavior."""

    def __init__(self, *, exits_on_terminate: bool) -> None:
        self.alive = True
        self.exits_on_terminate = exits_on_terminate
        self.terminated = 0
        self.killed = 0
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated += 1
        if self.exits_on_terminate:
            self.alive = False

    def kill(self) -> None:
        self.killed += 1
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


class _FakeExecutor:
    """ProcessPoolExecutor stand-in exposing its owned worker processes."""

    def __init__(self, processes: list[_FakePoolProcess]) -> None:
        self._processes = {index: process for index, process in enumerate(processes)}
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(self, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


def test_process_pool_cleanup_terminates_joins_and_escalates() -> None:
    """Aborted fitting pools reap workers, killing only TERM-resistant ones."""
    graceful = _FakePoolProcess(exits_on_terminate=True)
    resistant = _FakePoolProcess(exits_on_terminate=False)
    executor = _FakeExecutor([graceful, resistant])

    terminate_process_pool(executor, timeout=0.0)  # type: ignore[arg-type]

    assert graceful.terminated == resistant.terminated == 1
    assert graceful.killed == 0
    assert resistant.killed == 1
    assert graceful.join_calls == [0.0, None]
    assert resistant.join_calls == [0.0, None]
    assert executor.shutdown_calls == [(True, True)]


def test_isolated_command_interrupt_cleans_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interruption cleans the isolated group and remains an interrupt."""
    popen_kwargs: dict[str, Any] = {}
    cleaned: list[Any] = []
    process = _FakeProcess()

    def _communicate(timeout: float | None = None) -> tuple[str, str]:
        raise KeyboardInterrupt

    process.communicate = _communicate  # type: ignore[attr-defined]

    def _popen(*args: Any, **kwargs: Any) -> _FakeProcess:
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(
        "pmriprep.utils.processes.terminate_process_group",
        lambda child, timeout: cleaned.append(child),
    )

    with pytest.raises(KeyboardInterrupt):
        run_isolated_command(["hd_glio_predict"])

    assert popen_kwargs["start_new_session"] is True
    assert cleaned == [process]


def test_isolated_command_cleanup_failure_preserves_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup errors do not replace the interruption that initiated cleanup."""
    process = _FakeProcess()

    def _communicate(timeout: float | None = None) -> tuple[str, str]:
        raise KeyboardInterrupt

    def _cleanup_failure(child: Any, timeout: float) -> None:
        raise RuntimeError("cleanup failed")

    process.communicate = _communicate  # type: ignore[attr-defined]
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        "pmriprep.utils.processes.terminate_process_group",
        _cleanup_failure,
    )

    with pytest.raises(KeyboardInterrupt):
        run_isolated_command(["hd_glio_predict"])
