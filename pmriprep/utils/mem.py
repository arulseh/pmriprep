"""Memory estimation utilities for workflow scheduling."""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _system_ram_gb() -> float:
    """Total system RAM in GB (stdlib only, works on Linux/macOS)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return 16.0  # conservative fallback


def estimate_4d_mem_gb(
    nifti_path: Path,
    n_buffers: int = 4,
    mem_limit_gb: Optional[float] = None,
) -> float:
    """Estimate peak memory for processing a 4D NIfTI.

    Reads only the NIfTI header (no data loaded). Assumes float64
    (8 bytes) since nibabel upcasts on load regardless of on-disk dtype.

    Args:
        nifti_path: Path to 4D NIfTI file.
        n_buffers: Number of concurrent array copies in memory.
            DSC deconvolution: 4 (conc + signal + recomputed + transient).
            DCE PK fitting: 5 (conc + signal + model arrays + Bayesian).
        mem_limit_gb: User-specified memory limit (``--mem-gb``).
            When None, caps at 90% of system RAM.

    Returns:
        Estimated memory in GB, capped so the node is always schedulable.
    """
    import nibabel as nib

    shape = nib.load(nifti_path).header.get_data_shape()
    n_voxels = shape[0] * shape[1] * shape[2]
    n_timepoints = shape[3] if len(shape) > 3 else 1
    raw_estimate = (n_voxels * n_timepoints * 8 * n_buffers) / 1e9

    # Warn before capping so user sees the true requirement
    sys_ram_gb = _system_ram_gb()
    if raw_estimate > sys_ram_gb * 0.8:
        logger.warning(
            "Dataset requires ~%.1f GB but system has %.1f GB. "
            "Consider --nprocs 1 or a machine with more RAM.",
            raw_estimate, sys_ram_gb,
        )

    # Cap so the node is always schedulable
    cap = (mem_limit_gb * 0.9) if mem_limit_gb else (sys_ram_gb * 0.9)
    estimated = min(raw_estimate, cap)

    return max(estimated, 0.5)  # floor at 0.5 GB
