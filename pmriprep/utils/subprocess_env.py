"""Subprocess environment helpers for GPU inference nodes.

Provides thread-limiting environment variables for nnU-Net/PyTorch
subprocesses (HD-BET, HD-GLIO). When multiple GPU inference nodes run
concurrently, each nnU-Net process spawns background workers that can
exhaust thread/PID limits. These variables constrain each process to
a single CPU worker, preventing ``pthread_create`` EAGAIN errors.
"""

import os

NNUNET_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "nnUNet_def_n_proc": "1",
}
"""Thread-limiting env vars for nnU-Net/PyTorch inference subprocesses."""

NNUNET_THREAD_ENV_STR: str = " ".join(
    f"{k}={v}" for k, v in NNUNET_THREAD_ENV.items()
)
"""Shell-formatted string for use in ``bash -c`` / ``Popen(shell=True)``."""


def get_nnunet_subprocess_env() -> dict[str, str]:
    """Return ``os.environ`` merged with nnU-Net thread-limiting vars.

    Returns:
        Environment dictionary suitable for ``subprocess.run(env=...)``.
    """
    return {**os.environ, **NNUNET_THREAD_ENV}
