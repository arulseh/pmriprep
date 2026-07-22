"""Lifecycle helpers for PMRIPrep-owned external processes."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from collections.abc import Sequence

logger = logging.getLogger(__name__)


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
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
