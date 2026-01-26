"""QC visualization utilities.

This module provides quality control visualization functions for
registration assessment, motion correction QC, and other pipeline
outputs. All visualizations use the colorblind-safe palette from
qc_standards.py.

References:
    - Power et al. (2012) for framewise displacement
    - Paul Tol color schemes for accessibility
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from .qc_standards import (
    COLORBLIND_PALETTE,
    MOTION_FD_THRESHOLDS,
    get_qc_status,
    get_status_color,
)


def get_radiological_slice(
    data: npt.NDArray[Any],
    axis: int = 2,
    slice_idx: Optional[int] = None,
) -> npt.NDArray[Any]:
    """Extract a 2D slice from a 3D volume in radiological orientation.

    Radiological convention:
    - Axial (axis=2): Left on right side of image, anterior at top
    - Coronal (axis=1): Left on right side of image, superior at top
    - Sagittal (axis=0): Anterior on left, superior at top

    Args:
        data: 3D numpy array (x, y, z)
        axis: Slice axis (0=sagittal, 1=coronal, 2=axial). Default is axial.
        slice_idx: Index of slice to extract. If None, uses middle slice.

    Returns:
        2D numpy array with proper radiological orientation (flipped/rotated
        as needed for display)

    Raises:
        ValueError: If data is not 3D or axis is invalid
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D")

    if axis not in (0, 1, 2):
        raise ValueError(f"Axis must be 0, 1, or 2, got {axis}")

    # Get slice index (default to middle)
    if slice_idx is None:
        slice_idx = data.shape[axis] // 2

    # Extract slice along the specified axis
    slice_2d: npt.NDArray[Any]
    if axis == 0:  # Sagittal (YZ plane)
        slice_2d = data[slice_idx, :, :]
        # Rotate 90° and flip to get anterior-left, superior-top
        slice_2d = np.rot90(slice_2d, k=1)
    elif axis == 1:  # Coronal (XZ plane)
        slice_2d = data[:, slice_idx, :]
        # Flip left-right for radiological, rotate for superior-top
        slice_2d = np.rot90(slice_2d, k=1)
        slice_2d = np.fliplr(slice_2d)
    else:  # axis == 2, Axial (XY plane)
        slice_2d = data[:, :, slice_idx]
        # Flip left-right for radiological convention, rotate for display
        slice_2d = np.rot90(slice_2d, k=1)
        slice_2d = np.fliplr(slice_2d)

    return slice_2d


def calculate_framewise_displacement(
    motion_params: npt.NDArray[Any],
    radius: float = 50.0,
) -> npt.NDArray[np.float64]:
    """Calculate framewise displacement from motion parameters.

    Framewise displacement (FD) quantifies head motion between consecutive
    frames. Rotation parameters are converted to mm using the specified
    radius (typically 50mm for head radius).

    FD = |Δtx| + |Δty| + |Δtz| + r×|Δrx| + r×|Δry| + r×|Δrz|

    Args:
        motion_params: Motion parameters array of shape (N, 6) where columns
            are [tx, ty, tz, rx, ry, rz] (translations in mm, rotations in
            radians) or shape (N, 12) for affine parameters (uses first 6).
        radius: Head radius in mm for converting rotations to displacements.
            Default is 50.0mm.

    Returns:
        1D array of framewise displacement values (length N, with first
        element set to 0 as FD is undefined for first frame).

    Raises:
        ValueError: If motion_params has wrong shape
    """
    motion_params = np.asarray(motion_params, dtype=np.float64)

    if motion_params.ndim != 2:
        raise ValueError(f"Expected 2D array, got {motion_params.ndim}D")

    if motion_params.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns (tx,ty,tz,rx,ry,rz), "
            f"got {motion_params.shape[1]}"
        )

    # Extract the 6 rigid-body parameters
    params = motion_params[:, :6]

    # Calculate frame-to-frame differences
    diff = np.diff(params, axis=0)

    # Translations are direct displacements (columns 0-2)
    trans_disp = np.abs(diff[:, :3])

    # Rotations converted to arc length: displacement = r × θ (columns 3-5)
    rot_disp = np.abs(diff[:, 3:6]) * radius

    # Sum all displacements
    fd = np.sum(trans_disp, axis=1) + np.sum(rot_disp, axis=1)

    # Prepend 0 for the first frame (no previous frame to compare)
    fd = np.concatenate([[0.0], fd])

    return fd


def plot_registration_qc_filmstrip(
    fixed: npt.NDArray[Any],
    moving: npt.NDArray[Any],
    output_path: Union[str, Path],
    mask: Optional[npt.NDArray[Any]] = None,
    title: Optional[str] = None,
    n_slices: int = 7,
    contour_color: Optional[str] = None,
    axis: int = 2,
) -> Path:
    """Create a filmstrip visualization for registration QC.

    Generates a multi-slice filmstrip showing the fixed image with
    contours of the moving image overlaid. This allows visual assessment
    of registration quality.

    Args:
        fixed: 3D reference/fixed image array
        moving: 3D moving/registered image array (same shape as fixed)
        output_path: Path to save the output PNG image
        mask: Optional 3D brain mask to restrict display region
        title: Optional title for the figure
        n_slices: Number of slices to display in the filmstrip (default 7)
        contour_color: Color for contour overlay. If None, uses palette primary.
        axis: Slice axis (0=sagittal, 1=coronal, 2=axial). Default is axial.

    Returns:
        Path to the saved figure

    Raises:
        ValueError: If image shapes don't match or arrays aren't 3D
    """
    # Import matplotlib here to avoid import overhead when not needed
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    fixed = np.asarray(fixed)
    moving = np.asarray(moving)

    if fixed.ndim != 3 or moving.ndim != 3:
        raise ValueError("Both fixed and moving images must be 3D")

    if fixed.shape != moving.shape:
        raise ValueError(
            f"Shape mismatch: fixed {fixed.shape} != moving {moving.shape}"
        )

    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != fixed.shape:
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match image shape {fixed.shape}"
            )

    # Use colorblind-safe palette color for contours
    if contour_color is None:
        contour_color = COLORBLIND_PALETTE.primary

    # Calculate slice positions (evenly spaced through the volume)
    n_total = fixed.shape[axis]
    # Exclude edge slices (10% margin on each side)
    margin = int(n_total * 0.1)
    slice_indices = np.linspace(margin, n_total - margin - 1, n_slices, dtype=int)

    # Create the figure
    fig, axes = plt.subplots(1, n_slices, figsize=(n_slices * 2, 2.5))

    # Handle single slice case
    if n_slices == 1:
        axes = [axes]

    # Determine intensity range from fixed image (exclude zeros if masked)
    if mask is not None:
        valid_voxels = fixed[mask > 0]
    else:
        valid_voxels = fixed[fixed > 0]

    vmin: float
    vmax: float
    if len(valid_voxels) > 0:
        vmin = float(np.percentile(valid_voxels, 1))
        vmax = float(np.percentile(valid_voxels, 99))
    else:
        vmin, vmax = 0.0, 1.0

    for ax_idx, slice_idx in enumerate(slice_indices):
        ax = axes[ax_idx]

        # Get slices in radiological orientation
        fixed_slice = get_radiological_slice(fixed, axis=axis, slice_idx=int(slice_idx))
        moving_slice = get_radiological_slice(moving, axis=axis, slice_idx=int(slice_idx))

        # Display the fixed image as background
        ax.imshow(
            fixed_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Overlay contours of the moving image
        # Use multiple contour levels for better visualization
        if np.any(moving_slice > 0):
            low_pct = float(np.percentile(moving_slice[moving_slice > 0], 30))
            high_pct = float(np.percentile(moving_slice[moving_slice > 0], 70))
        else:
            low_pct, high_pct = 0.3, 0.7

        contour_levels = np.linspace(low_pct, high_pct, 3)

        # Only draw contours if there's meaningful data
        if np.any(moving_slice > 0):
            ax.contour(
                moving_slice,
                levels=contour_levels,
                colors=[contour_color],
                linewidths=0.5,
                alpha=0.8,
            )

        ax.axis("off")
        ax.set_title(f"z={slice_idx}", fontsize=8)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=10, y=1.02)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path


def plot_motion_qc(
    motion_params: npt.NDArray[Any],
    output_path: Union[str, Path],
    tr: float = 1.0,
    title: Optional[str] = None,
    radius: float = 50.0,
) -> Path:
    """Create a motion QC plot showing framewise displacement over time.

    Generates a two-panel figure showing:
    1. Top panel: Raw motion parameters (translations and rotations)
    2. Bottom panel: Framewise displacement with threshold lines

    Args:
        motion_params: Motion parameters array of shape (N, 6) where columns
            are [tx, ty, tz, rx, ry, rz]
        output_path: Path to save the output PNG image
        tr: Repetition time in seconds for x-axis time labeling
        title: Optional title for the figure
        radius: Head radius in mm for FD calculation (default 50.0)

    Returns:
        Path to the saved figure

    Raises:
        ValueError: If motion_params has wrong shape
    """
    # Import matplotlib here to avoid import overhead when not needed
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    motion_params = np.asarray(motion_params)

    if motion_params.ndim != 2 or motion_params.shape[1] < 6:
        raise ValueError(
            f"Expected shape (N, 6) or (N, 12), got {motion_params.shape}"
        )

    # Extract the 6 rigid-body parameters
    params = motion_params[:, :6]
    n_frames = params.shape[0]

    # Calculate time axis
    time = np.arange(n_frames) * tr

    # Calculate framewise displacement
    fd = calculate_framewise_displacement(params, radius=radius)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Colors for motion parameters
    trans_colors = [COLORBLIND_PALETTE.primary, COLORBLIND_PALETTE.secondary, COLORBLIND_PALETTE.success]
    rot_colors = [COLORBLIND_PALETTE.error, COLORBLIND_PALETTE.warning, "#AA3377"]  # Purple for rz

    # Top panel: Raw motion parameters
    # Translations
    labels_trans = ["tx", "ty", "tz"]
    for i, (label, color) in enumerate(zip(labels_trans, trans_colors)):
        ax1.plot(time, params[:, i], label=label, color=color, linewidth=1)

    # Rotations (convert to degrees for display)
    labels_rot = ["rx", "ry", "rz"]
    for i, (label, color) in enumerate(zip(labels_rot, rot_colors)):
        ax1.plot(
            time,
            np.rad2deg(params[:, i + 3]),
            label=label,
            color=color,
            linewidth=1,
            linestyle="--",
        )

    ax1.set_ylabel("Motion (mm / deg)")
    ax1.legend(loc="upper right", ncol=2, fontsize=8)
    ax1.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Motion Parameters")

    # Bottom panel: Framewise displacement
    # Determine QC status for coloring
    mean_fd = float(np.mean(fd))
    qc_status = get_qc_status(mean_fd, MOTION_FD_THRESHOLDS, higher_is_better=False)
    fd_color = get_status_color(qc_status)

    ax2.fill_between(time, fd, alpha=0.3, color=fd_color)
    ax2.plot(time, fd, color=fd_color, linewidth=1)

    # Add threshold lines
    ax2.axhline(
        y=MOTION_FD_THRESHOLDS.pass_threshold,
        color=COLORBLIND_PALETTE.success,
        linestyle="--",
        linewidth=1,
        label=f"Pass (<{MOTION_FD_THRESHOLDS.pass_threshold}mm)",
    )
    ax2.axhline(
        y=MOTION_FD_THRESHOLDS.warn_threshold,
        color=COLORBLIND_PALETTE.warning,
        linestyle="--",
        linewidth=1,
        label=f"Warn (<{MOTION_FD_THRESHOLDS.warn_threshold}mm)",
    )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("FD (mm)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Add summary statistics
    max_fd: float = float(np.max(fd))
    outlier_count: int = int(np.sum(fd > MOTION_FD_THRESHOLDS.warn_threshold))
    outlier_frac: float = outlier_count / len(fd) * 100

    stats_text = (
        f"Mean FD: {mean_fd:.3f}mm | Max FD: {max_fd:.3f}mm | "
        f"Outliers: {outlier_count} ({outlier_frac:.1f}%) | Status: {qc_status}"
    )
    ax2.set_title(f"Framewise Displacement\n{stats_text}", fontsize=9)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path
