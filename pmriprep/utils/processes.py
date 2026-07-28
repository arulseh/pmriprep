"""Lifecycle helpers for PMRIPrep-owned external processes."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import psutil

logger = logging.getLogger(__name__)


def enable_child_subreaper() -> bool:
    """Adopt orphaned descendants so PMRIPrep can reap them on Linux.

    This is primarily needed in containers whose PID 1 does not reap zombies.
    Failure is non-fatal because descendant termination still works without
    adoption, although reaping orphaned grandchildren then remains PID 1's
    responsibility.

    Returns
    -------
    bool
        ``True`` when Linux accepted the subreaper request.
    """
    if not sys.platform.startswith("linux"):
        return False

    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
    except (AttributeError, OSError) as error:
        logger.debug("Could not enable child subreaping: %s", error)
        return False
    return True


def _reap_exited_children() -> None:
    """Collect all immediately available child exit statuses."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def terminate_descendant_processes(
    *,
    interrupted: bool = False,
    timeout: float = 5.0,
) -> None:
    """Stop and reap the current PMRIPrep process's descendants.

    Parameters
    ----------
    interrupted : bool
        Forward ``SIGINT`` first so Python workers can run their own cleanup
        handlers. Other abnormal exits begin with ``SIGTERM``.
    timeout : float
        Total grace period before escalation to ``SIGKILL``.
    """
    try:
        descendants = psutil.Process().children(recursive=True)
    except (psutil.Error, OSError) as error:
        logger.warning("Could not enumerate PMRIPrep child processes: %s", error)
        return
    if not descendants:
        _reap_exited_children()
        return

    first_signal = signal.SIGINT if interrupted else signal.SIGTERM
    for process in reversed(descendants):
        try:
            process.send_signal(first_signal)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            logger.warning("Could not signal child process %d: %s", process.pid, error)

    _, alive = psutil.wait_procs(descendants, timeout=timeout)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            logger.warning("Could not kill child process %d: %s", process.pid, error)
    if alive:
        psutil.wait_procs(alive, timeout=timeout)
    _reap_exited_children()


def terminate_process_pool(
    executor: ProcessPoolExecutor,
    *,
    timeout: float = 5.0,
) -> None:
    """Terminate and synchronously reap workers after an abnormal pool exit.

    ``ProcessPoolExecutor.shutdown(cancel_futures=True)`` cannot cancel work
    that is already running. Snapshotting the owned worker processes before
    shutdown permits bounded cleanup when fitting is interrupted or raises.

    Parameters
    ----------
    executor : concurrent.futures.ProcessPoolExecutor
        Pool whose active workers belong to the current PMRIPrep process.
    timeout : float
        Total seconds to allow workers to exit after ``SIGTERM`` before
        escalation to ``SIGKILL``.
    """
    processes = list(getattr(executor, "_processes", {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()

    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))

    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join()

    executor.shutdown(wait=True, cancel_futures=True)


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    timeout: float = 5.0,
) -> None:
    """Terminate a subprocess group and synchronously reap its leader.

    Parameters
    ----------
    process : subprocess.Popen
        Leader of a subprocess group created with ``start_new_session=True``.
    timeout : float
        Seconds to allow graceful termination before forcing termination.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_isolated_command(
    command: Sequence[str],
    *,
    timeout: float | None = None,
    cleanup_timeout: float = 5.0,
) -> subprocess.CompletedProcess[str]:
    """Run a command in an isolated process group with bounded cleanup.

    Parameters
    ----------
    command : sequence of str
        Executable and arguments passed directly to ``Popen``.
    timeout : float or None
        Maximum execution time, or no timeout when ``None``.
    cleanup_timeout : float
        Grace period after termination before killing the process group.

    Returns
    -------
    subprocess.CompletedProcess
        Captured command result. Nonzero exits are returned to the caller.
    """
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        try:
            terminate_process_group(process, timeout=cleanup_timeout)
        except Exception as cleanup_error:
            logger.warning(
                "Failed to fully reap subprocess group for %s: %s",
                command[0],
                cleanup_error,
            )
        for stream_name in ("stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except Exception as close_error:
                    logger.debug(
                        "Could not close child %s stream for %s: %s",
                        stream_name, command[0], close_error,
                    )
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
