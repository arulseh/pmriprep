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


def cap_mem_gb(value: float, mem_limit_gb: Optional[float] = None) -> float:
    """Cap a mem_gb value so the node is schedulable by nipype.

    Args:
        value: Desired mem_gb for the node.
        mem_limit_gb: User-specified ``--mem-gb``. When None, caps at
            90% of system RAM.

    Returns:
        mem_gb clamped to [0.5, limit].
    """
    limit = mem_limit_gb if mem_limit_gb is not None else (_system_ram_gb() * 0.9)
    return max(0.5, min(value, limit))


def estimate_4d_mem_gb(
    nifti_path: Path,
    n_buffers: int = 4,
    mem_limit_gb: Optional[float] = None,
) -> tuple[float, float]:
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
        Tuple of (capped_estimate, raw_estimate) in GB.
        capped_estimate is used for nipype scheduling.
        raw_estimate is the true memory requirement for user warnings.
    """
    import nibabel as nib

    shape = nib.load(nifti_path).header.get_data_shape()  # type: ignore[attr-defined]
    n_voxels = shape[0] * shape[1] * shape[2]
    n_timepoints = shape[3] if len(shape) > 3 else 1
    raw_estimate = (n_voxels * n_timepoints * 8 * n_buffers) / 1e9

    return cap_mem_gb(raw_estimate, mem_limit_gb), raw_estimate
