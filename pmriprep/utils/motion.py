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

    FD formula: |Δtrans| + r × |Δrot|

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
    FD is undefined for the first frame (reference), so we set it to 0.
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


def calculate_rms_displacement(
    motion_params: npt.NDArray[Any],
    head_radius: float = 50.0,
) -> npt.NDArray[Any]:
    """Calculate RMS displacement from motion parameters.

    RMS displacement combines rotations and translations into a single
    measure of total displacement from the reference position.

    Parameters
    ----------
    motion_params : np.ndarray
        Motion parameters array with shape (N, 6).
        Columns: rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm).
    head_radius : float
        Head radius in mm for converting rotations to displacements (default: 50mm).

    Returns
    -------
    np.ndarray
        RMS displacement for each frame relative to reference (length N).

    Notes
    -----
    First frame RMS is always 0 (it's the reference).
    """
    motion_params = np.asarray(motion_params, dtype=np.float64)

    if motion_params.ndim == 1:
        motion_params = motion_params.reshape(-1, 6)

    # Rotations (columns 0-2) in radians, translations (columns 3-5) in mm
    rot = motion_params[:, :3]  # radians
    trans = motion_params[:, 3:]  # mm

    # Convert rotations to mm displacement using head radius
    rot_disp = rot * head_radius  # mm

    # RMS displacement: sqrt(sum of squared displacements / 6)
    all_disp = np.hstack([rot_disp, trans])  # (N, 6)
    rms = np.sqrt(np.mean(all_disp ** 2, axis=1))

    return rms  # type: ignore[no-any-return]
