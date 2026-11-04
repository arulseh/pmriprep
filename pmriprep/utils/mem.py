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

    Parameters
    ----------
    value : float
        Desired mem_gb for the node.
    mem_limit_gb : float, optional
        User-specified ``--mem-gb``. When None, caps at
        90% of system RAM.

    Returns
    -------
    float
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

    Parameters
    ----------
    nifti_path : Path
        Path to 4D NIfTI file.
    n_buffers : int
        Number of concurrent array copies in memory.
        DSC deconvolution: 4 (conc + signal + recomputed + transient).
        DCE PK fitting: 5 (conc + signal + model arrays + Bayesian).
    mem_limit_gb : float, optional
        User-specified memory limit (``--mem-gb``).
        When None, caps at 90% of system RAM.

    Returns
    -------
    tuple of float
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


def estimate_dce_mem_gb(
    nifti_path: Path, model: str, n_procs: int,
    delay_correction: bool = True, sloppy: bool = False,
) -> float:
    """Reserve volume buffers, complete-slice profiles and bounded worker space.

    Parameters
    ----------
    nifti_path : Path
        Dynamic input; only its header is read.
    model : str
        Requested family or automatic mode. Auto reserves all three families.
    n_procs : int
        Resolved process budget, each with one numerical-library thread.
    delay_correction, sloppy : bool
        Select one, thirteen or 121 conditional delay points.

    Returns
    -------
    float
        Uncapped scheduling estimate in GB. Worker/library overhead is estimated
        at 0.35 GB per process plus 0.5 GB in the parent, in addition to arrays.
        This is a reservation, not an enforced operating-system memory limit.
    """
    import nibabel as nib
    from pmriprep.models.dce.delay_profiles import MAX_LOCAL_DELAY_SECONDS

    shape = nib.load(nifti_path).header.get_data_shape()  # type: ignore[attr-defined]
    workers = max(1, n_procs)
    points = int(MAX_LOCAL_DELAY_SECONDS * (1 if sloppy else 10)) + 1 if delay_correction else 1
    counts = {"patlak": 3, "tofts": 2, "extended_tofts": 3, "2cxm": 4}
    families = ["patlak", "extended_tofts", "2cxm"] if model == "auto" else [model]
    # Parameters plus SSE, status, evaluations, optimality and seed index.
    point_bytes = sum(8 * counts[f] + 32 for f in families)
    # Full plane is an upper bound on eligible brain members; two block copies
    # cover worker results and their serialized transport while assembling.
    profile_bytes = (shape[0] * shape[1] + 2 * 128 * workers) * points * point_bytes
    _, buffers = estimate_4d_mem_gb(nifti_path, n_buffers=5)
    return float(buffers + profile_bytes / 1e9 + 0.5 + 0.35 * workers)
