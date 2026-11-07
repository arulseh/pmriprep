"""Shared motion parameter utilities for PMRIPrep.

Calculate framewise displacement (FD) and QC summaries from rigid-body
motion parameters in the common FSL-compatible six-column format.
"""

from typing import Any

import numpy as np
import numpy.typing as npt


def calculate_framewise_displacement(
    motion_params: npt.NDArray[Any],
    head_radius: float = 50.0,
) -> npt.NDArray[Any]:
    """Calculate framewise displacement from motion parameters.

    Framewise displacement (FD) is calculated as the sum of absolute
    displacements, with rotations converted to mm using head radius.

    FD formula: ``|Δtrans| + r × |Δrot|``

    Parameters
    ----------
    motion_params : np.ndarray
        Motion parameters array with shape (N, 6).
        Columns: rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm).
        Parameters use the common FSL-compatible convention.
    head_radius : float
        Head radius in mm for converting rotations to displacements (default: 50mm).

    Returns
    -------
    np.ndarray
        Framewise displacement for each frame (length N).

    Notes
    -----
    FD is undefined for the first volume because it has no predecessor, so the
    returned numerical array uses 0 at that position. Public confounds encode
    it as ``n/a``.
    """
    motion_params = np.asarray(motion_params, dtype=np.float64)

    if motion_params.ndim == 1:
        motion_params = motion_params.reshape(-1, 6)

    n_frames = motion_params.shape[0]
    if n_frames < 2:
        return np.zeros(n_frames, dtype=np.float64)

    # Calculate frame-to-frame differences
    diff = np.diff(motion_params, axis=0)

    # Rotations (columns 0-2) in radians, translations (columns 3-5) in mm
    rot_diff = diff[:, :3]  # radians
    trans_diff = diff[:, 3:]  # mm

    # Convert rotations to mm displacement using head radius
    # Arc length = radius × angle (radians)
    rot_disp = np.abs(rot_diff) * head_radius  # mm

    # Sum absolute displacements: |Δtrans| + r×|Δrot|
    fd = np.sum(np.abs(trans_diff), axis=1) + np.sum(rot_disp, axis=1)

    # Prepend 0 for first frame (reference)
    fd_full = np.zeros(n_frames, dtype=np.float64)
    fd_full[1:] = fd

    return fd_full


def summarize_motion(
    motion_params: npt.NDArray[Any],
    threshold: float = 0.5,
    head_radius: float = 50.0,
) -> dict[str, Any]:
    """Summarize finite motion parameters using a common FD convention.

    Parameters
    ----------
    motion_params : ndarray
        Shape (frames, 6): radians followed by millimetres.
    threshold : float, optional
        Strict FD exceedance threshold in millimetres.
    head_radius : float, optional
        Radius converting angular displacement to millimetres.

    Returns
    -------
    dict
        Unrounded FD, transition mean, maximum, exceedances and composite QC.
        Percentages use all frames; the mean excludes the initial placeholder.
    """
    from pmriprep.utils.qc_standards import get_composite_motion_status

    params = np.atleast_2d(np.asarray(motion_params, dtype=float))
    if params.ndim != 2 or params.shape[1] != 6 or params.shape[0] == 0 or not np.isfinite(params).all():
        raise ValueError("Motion parameters must be a finite nonempty (frames, 6) array")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Motion threshold must be finite and nonnegative")
    if not np.isfinite(head_radius) or head_radius <= 0:
        raise ValueError("Motion head radius must be finite and positive")
    fd = calculate_framewise_displacement(params, head_radius)
    mean = float(np.mean(fd[1:])) if len(fd) > 1 else 0.0
    maximum = float(np.max(fd))
    indices = np.flatnonzero(fd > threshold).tolist()
    percent = 100.0 * len(indices) / len(fd)
    return {
        "fd_list": fd.tolist(), "fd_mean": mean, "fd_max": maximum,
        "n_outliers": len(indices), "outlier_frames": indices,
        "percent_outliers": percent,
        "fd_threshold": threshold, "head_radius_mm": head_radius,
        "n_volumes": len(fd), "motion_qc_available": True,
        "motion_status": get_composite_motion_status(mean, maximum, percent),
    }
