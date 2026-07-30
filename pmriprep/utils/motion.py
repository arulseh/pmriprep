"""Shared motion parameter utilities for PMRIPrep.

Provides functions to calculate framewise displacement (FD) and
root-mean-square displacement (RMSD) from rigid-body motion parameters
in FSL mcflirt `.par` format.
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
        This is FSL mcflirt output format.
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
