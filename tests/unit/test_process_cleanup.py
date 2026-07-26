"""Regression coverage for PMRIPrep-owned child-process cleanup."""

from __future__ import annotations

import signal
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from pmriprep.utils.processes import (
    enable_child_subreaper,
    run_isolated_command,
    terminate_descendant_processes,
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


class _FakeDescendant:
    """Minimal psutil process stand-in recording delivered signals."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.signals: list[signal.Signals] = []
        self.killed = 0

    def send_signal(self, sig: signal.Signals) -> None:
        self.signals.append(sig)

    def kill(self) -> None:
        self.killed += 1


def test_descendant_cleanup_forwards_interrupt_then_escalates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow interruption reaches nested workers and kills survivors."""
    parent = _FakeDescendant(10)
    worker = _FakeDescendant(11)
    wait_calls: list[tuple[list[_FakeDescendant], float]] = []
    reaped: list[bool] = []

    class _CurrentProcess:
        def children(self, recursive: bool) -> list[_FakeDescendant]:
            assert recursive is True
            return [parent, worker]

    def _wait_procs(
        processes: list[_FakeDescendant],
        timeout: float,
    ) -> tuple[list[_FakeDescendant], list[_FakeDescendant]]:
        wait_calls.append((processes, timeout))
        return ([], [worker]) if len(wait_calls) == 1 else ([worker], [])

    monkeypatch.setattr("pmriprep.utils.processes.psutil.Process", _CurrentProcess)
    monkeypatch.setattr("pmriprep.utils.processes.psutil.wait_procs", _wait_procs)
    monkeypatch.setattr(
        "pmriprep.utils.processes._reap_exited_children",
        lambda: reaped.append(True),
    )

    terminate_descendant_processes(interrupted=True, timeout=2.0)

    assert parent.signals == worker.signals == [signal.SIGINT]
    assert parent.killed == 0
    assert worker.killed == 1
    assert wait_calls == [([parent, worker], 2.0), ([worker], 2.0)]
    assert reaped == [True]


def test_enable_child_subreaper_is_platform_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Linux platforms do not attempt the Linux-specific prctl call."""
    monkeypatch.setattr("pmriprep.utils.processes.sys.platform", "darwin")

    assert enable_child_subreaper() is False


def test_workflow_interrupt_cleans_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI forwards an interrupted Nipype run to descendant cleanup."""
    from pmriprep.cli.run import _run_workflow

    calls: list[tuple[str, Any]] = []
    subreaper_calls: list[bool] = []
    cleanup_calls: list[bool] = []

    class _Workflow:
        def run(self, *, plugin: str, plugin_args: Any = None) -> None:
            calls.append((plugin, plugin_args))
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "pmriprep.utils.processes.enable_child_subreaper",
        lambda: subreaper_calls.append(True),
    )
    monkeypatch.setattr(
        "pmriprep.utils.processes.terminate_descendant_processes",
        lambda *, interrupted: cleanup_calls.append(interrupted),
    )

    with pytest.raises(KeyboardInterrupt):
        _run_workflow(_Workflow(), Namespace(nprocs=1, use_plugin=None))

    assert calls == [("Linear", None)]
    assert subreaper_calls == [True]
    assert cleanup_calls == [True]


def test_workflow_cleanup_failure_preserves_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleanup error never replaces the initiating KeyboardInterrupt."""
    from pmriprep.cli.run import _run_workflow

    class _Workflow:
        def run(self, *, plugin: str, plugin_args: Any = None) -> None:
            raise KeyboardInterrupt

    def _cleanup_failure(*, interrupted: bool) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        "pmriprep.utils.processes.enable_child_subreaper",
        lambda: True,
    )
    monkeypatch.setattr(
        "pmriprep.utils.processes.terminate_descendant_processes",
        _cleanup_failure,
    )

    with pytest.raises(KeyboardInterrupt):
        _run_workflow(_Workflow(), Namespace(nprocs=1, use_plugin=None))


@pytest.mark.parametrize("use_gpu", [False, True])
def test_multiproc_gpu_slots_follow_constructed_workflow(
    monkeypatch: pytest.MonkeyPatch, use_gpu: bool,
) -> None:
    """YAML-resolved node state, not absent CLI device flags, controls slots."""
    from pmriprep.cli.run import _run_workflow

    calls: list[tuple[str, dict[str, Any] | None]] = []

    class _Node:
        pass

    node = _Node()
    node.use_gpu = use_gpu

    class _Workflow:
        def _get_all_nodes(self):
            return [node]

        def run(self, *, plugin: str, plugin_args: Any = None) -> None:
            calls.append((plugin, plugin_args))

    monkeypatch.setattr("pmriprep.utils.processes.enable_child_subreaper", lambda: True)
    monkeypatch.setattr("pmriprep.utils.gpu.detect_gpu_count", lambda: 2)
    args = Namespace(nprocs=4, mem_gb=None, use_plugin=None)

    _run_workflow(_Workflow(), args)

    expected = {"n_procs": 4}
    if use_gpu:
        expected["n_gpu_procs"] = 2
    assert calls == [("MultiProc", expected)]


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
    assert "return list(executor.map(" not in source
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
