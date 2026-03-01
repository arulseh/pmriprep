"""Torch device detection utilities for PMRIPrep."""

import logging
import sys

logger = logging.getLogger(__name__)


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
