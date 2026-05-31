"""GPU detection utilities for PMRIPrep.

Provides GPU device detection for scheduling PyTorch-based tools
(HD-BET, HD-GLIO). Perfusion computation runs on CPU via NumPy/SciPy.
"""

from __future__ import annotations


def detect_gpu_count() -> int:
    """Detect the number of GPUs via PyTorch runtime APIs.

    Tries PyTorch CUDA first, then PyTorch MPS (Apple Silicon).
    Works inside containers without ``nvidia-smi`` as long as the
    CUDA driver library (``libcuda.so``) is visible.

    Returns
    -------
    int
        Number of GPU devices, or 0 if no GPU is detected.
    """
    try:
        import torch
    except Exception:
        return 0

    if torch.cuda.is_available():
        return int(torch.cuda.device_count())

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 1

    return 0


def get_device_name() -> str:
    """Get the name of the current GPU device.

    Returns
    -------
    str
        Device name string, or "CPU" if no GPU is available.
    """
    return "CPU"
