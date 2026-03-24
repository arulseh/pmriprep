"""Torch device detection utilities for PMRIPrep."""

import logging
import platform
import sys

logger = logging.getLogger(__name__)


def resolve_device_uses_gpu(device_str: str) -> bool:
    """Resolve ``'auto'`` device and return whether it uses GPU.

    Args:
        device_str: Device string (``'auto'``, ``'cuda'``, ``'mps'``, or ``'cpu'``).

    Returns:
        True if the resolved device is a GPU (CUDA or MPS).
    """
    if device_str == "auto":
        device_str = detect_torch_device()
    return device_str in ("cuda", "mps")


def detect_torch_device() -> str:
    """Auto-detect the best available torch device.

    Checks for CUDA GPU first, then Apple MPS, then falls back to CPU.
    If torch is not importable, uses platform detection as a fallback
    (macOS assumed to have MPS).

    Returns
    -------
    str
        Device name: ``'cuda'``, ``'mps'``, or ``'cpu'``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Torch device: CUDA GPU detected")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Torch device: Apple MPS detected")
            return "mps"
    except ImportError:
        pass

    # Fallback: check platform for macOS (likely has MPS)
    if sys.platform == "darwin":
        logger.info("Torch device: macOS detected, assuming MPS")
        return "mps"

    logger.info("Torch device: No GPU detected, using CPU")
    return "cpu"
