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
        motion_params: Motion parameters array of shape (N, 6) in MCFLIRT
            format where columns are [rx, ry, rz, tx, ty, tz] (rotations
            in radians, translations in mm), or shape (N, 12) for affine
            parameters (uses first 6).
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
            f"Expected at least 6 columns (rx,ry,rz,tx,ty,tz), "
            f"got {motion_params.shape[1]}"
        )

    # Extract the 6 rigid-body parameters
    params = motion_params[:, :6]

    # Calculate frame-to-frame differences
    diff = np.diff(params, axis=0)

    # MCFLIRT format: cols 0-2 = rotations (radians), cols 3-5 = translations (mm)
    rot_disp = np.abs(diff[:, :3]) * radius
    trans_disp = np.abs(diff[:, 3:6])

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
    voxel_sizes: Optional[tuple[float, float, float]] = None,
) -> Path:
    """Create a 3-row orthogonal filmstrip for registration QC.

    Generates a multi-slice filmstrip with three rows (axial, coronal,
    sagittal) showing the fixed image as grayscale background with
    Sobel edge-detected boundaries of the moving image overlaid in color.
    This matches the perf-plus approach (fslmaths -edge -bin) for clean
    structural boundary visualization.

    Args:
        fixed: 3D background image array (e.g. DSC/DCE mean)
        moving: 3D image for edge overlay (e.g. T1w in functional space)
        output_path: Path to save the output PNG image
        mask: Optional 3D brain mask to restrict display region
        title: Optional title for the figure
        n_slices: Number of slices per row (default 7)
        contour_color: Color for contour overlay. If None, uses palette primary.
        voxel_sizes: Voxel dimensions (dx, dy, dz) in mm for correct aspect
            ratios. If None, uses equal aspect.

    Returns:
        Path to the saved figure

    Raises:
        ValueError: If image shapes don't match or arrays aren't 3D
    """
    # Import matplotlib/scipy here to avoid import overhead when not needed
    import matplotlib.colors as mcolors  # type: ignore[import-untyped]
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    from scipy.ndimage import sobel

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
        # Brain-extract the overlay so contours only appear inside the brain
        moving = moving.copy()
        moving[mask == 0] = 0

    # Use colorblind-safe palette color for contours
    if contour_color is None:
        contour_color = COLORBLIND_PALETTE.error

    # Row definitions: (label, axis, aspect_ratio)
    # Aspect ratios account for anisotropic voxels in each view plane
    row_defs: list[tuple[str, int]] = [
        ("Axial", 2),
        ("Coronal", 1),
        ("Sagittal", 0),
    ]

    # Compute per-plane aspect ratios from voxel sizes
    # After get_radiological_slice rotation, the displayed axes are:
    #   Axial (axis=2):    rows=y, cols=x  -> aspect = dy/dx
    #   Coronal (axis=1):  rows=z, cols=x  -> aspect = dz/dx
    #   Sagittal (axis=0): rows=z, cols=y  -> aspect = dz/dy
    if voxel_sizes is not None:
        dx, dy, dz = voxel_sizes
        aspect_ratios = [dy / dx, dz / dx, dz / dy]
    else:
        aspect_ratios = [1.0, 1.0, 1.0]

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

    # Create the figure: 3 rows x n_slices columns
    fig, axes = plt.subplots(
        3, n_slices, figsize=(n_slices * 2.2, 3 * 2.5),
    )

    # Handle single slice case (axes would be 1D)
    if n_slices == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (label, axis) in enumerate(row_defs):
        aspect = aspect_ratios[row_idx]

        # Calculate slice positions (evenly spaced, 10% margin each side)
        n_total = fixed.shape[axis]
        margin = int(n_total * 0.1)
        slice_indices = np.linspace(
            margin, n_total - margin - 1, n_slices, dtype=int,
        )

        # Axis label mapping for slice coordinate labels
        axis_labels = {2: "z", 1: "y", 0: "x"}
        coord_label = axis_labels[axis]

        for col_idx, slice_idx in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]

            # Get slices in radiological orientation
            fixed_slice = get_radiological_slice(
                fixed, axis=axis, slice_idx=int(slice_idx),
            )
            moving_slice = get_radiological_slice(
                moving, axis=axis, slice_idx=int(slice_idx),
            )

            # Display the fixed image as background
            ax.imshow(
                fixed_slice,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                aspect=aspect,
            )

            # Overlay edge-detected boundaries of the moving image
            # Sobel edge detection (Python equivalent of fslmaths -edge)
            # produces clean structural boundaries instead of noisy contours
            edges_x = sobel(moving_slice, axis=0)
            edges_y = sobel(moving_slice, axis=1)
            edge_mag = np.sqrt(edges_x**2 + edges_y**2)
            if np.any(edge_mag > 0):
                threshold = float(
                    np.percentile(edge_mag[edge_mag > 0], 70),
                )
                edge_mask = edge_mag > threshold
                # Create RGBA overlay: colored edges with transparency
                overlay = np.zeros((*edge_mask.shape, 4))
                rgba = mcolors.to_rgba(contour_color)
                overlay[edge_mask] = (*rgba[:3], 0.8)
                ax.imshow(overlay, aspect=aspect)

            ax.axis("off")
            ax.set_title(f"{coord_label}={slice_idx}", fontsize=7)

    plt.tight_layout()

    # Row labels — placed after tight_layout so positions are stable
    for row_idx, (label, _axis) in enumerate(row_defs):
        # Get the bounding box of the leftmost subplot in this row
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            bbox.x0 - 0.01, (bbox.y0 + bbox.y1) / 2, label,
            fontsize=9, fontweight="bold", rotation=90,
            ha="right", va="center",
        )

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=10, y=1.02)

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
    """Create a 3-panel motion QC plot matching perf-plus layout.

    Generates a three-panel figure showing:
    1. Top panel: Translation parameters (mm)
    2. Middle panel: Rotation parameters (degrees)
    3. Bottom panel: Framewise displacement with threshold and outlier shading

    Args:
        motion_params: Motion parameters array of shape (N, 6) in MCFLIRT
            format where columns are [rx, ry, rz, tx, ty, tz] (rotations
            in radians, translations in mm), or shape (N, 12) for affine
            parameters (uses first 6).
        output_path: Path to save the output PNG image
        tr: Repetition time in seconds (unused, x-axis is volume number)
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

    # Extract the 6 rigid-body parameters (MCFLIRT: rx, ry, rz, tx, ty, tz)
    params = motion_params[:, :6]
    n_frames = params.shape[0]
    volumes = np.arange(n_frames)

    # Calculate framewise displacement
    fd = calculate_framewise_displacement(params, radius=radius)
    fd_threshold = MOTION_FD_THRESHOLDS.warn_threshold  # 0.5mm

    # Perf-plus color scheme
    color_1 = "#EE7733"  # Orange
    color_2 = "#0077BB"  # Blue
    color_3 = "#009988"  # Teal
    color_fd = "#0077BB"  # Blue for FD fill
    color_thresh = "#CC3311"  # Red for threshold

    # Create figure with three subplots, shared x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
    )

    # --- Subplot 1: Translations (cols 3-5) ---
    ax1.plot(volumes, params[:, 3], label="X (LR)", color=color_1, linewidth=1)
    ax1.plot(volumes, params[:, 4], label="Y (AP)", color=color_2, linewidth=1)
    ax1.plot(volumes, params[:, 5], label="Z (SI)", color=color_3, linewidth=1)
    ax1.set_ylabel("Translation (mm)")
    ax1.legend(loc="upper right", fontsize=8, ncol=3)
    ax1.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Translation Parameters")

    # --- Subplot 2: Rotations (cols 0-2, converted to degrees) ---
    ax2.plot(
        volumes, np.rad2deg(params[:, 0]),
        label="Pitch (X)", color=color_1, linewidth=1,
    )
    ax2.plot(
        volumes, np.rad2deg(params[:, 1]),
        label="Roll (Y)", color=color_2, linewidth=1,
    )
    ax2.plot(
        volumes, np.rad2deg(params[:, 2]),
        label="Yaw (Z)", color=color_3, linewidth=1,
    )
    ax2.set_ylabel("Rotation (\u00b0)")
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Rotation Parameters")

    # --- Subplot 3: Framewise Displacement ---
    ax3.fill_between(volumes, fd, alpha=0.3, color=color_fd)
    ax3.plot(volumes, fd, color=color_fd, linewidth=1)

    # Threshold line
    ax3.axhline(
        y=fd_threshold, color=color_thresh, linestyle="--", linewidth=1,
        label=f"Threshold ({fd_threshold} mm)",
    )

    # Red shading for frames exceeding threshold
    outlier_mask = fd > fd_threshold
    for i in range(n_frames):
        if outlier_mask[i]:
            ax3.axvspan(i - 0.5, i + 0.5, alpha=0.15, color=color_thresh)

    # Y-axis limits
    max_fd = float(np.max(fd))
    ax3.set_ylim(0, max(max_fd * 1.1, fd_threshold * 1.5))

    ax3.set_xlabel("Volume")
    ax3.set_ylabel("FD (mm)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Framewise Displacement")

    # Metrics box (bottom-left)
    mean_fd = float(np.mean(fd))
    outlier_count = int(np.sum(outlier_mask))
    qc_status = get_qc_status(
        mean_fd, MOTION_FD_THRESHOLDS, higher_is_better=False,
    )
    stats_text = (
        f"Status: {qc_status}\n"
        f"Mean FD: {mean_fd:.3f} mm\n"
        f"Max FD: {max_fd:.3f} mm\n"
        f"Outliers (>{fd_threshold} mm): {outlier_count}"
    )
    ax3.text(
        0.02, 0.05, stats_text, transform=ax3.transAxes,
        fontsize=8, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    # Overall title
    if title:
        fig.suptitle(title, fontsize=11)

    fig.tight_layout(rect=[0, 0.12, 1, 0.985])

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path
