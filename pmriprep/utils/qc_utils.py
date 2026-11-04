"""QC visualization utilities.

This module provides quality control visualization functions for
registration assessment, motion correction QC, and other pipeline
outputs. All visualizations use the colorblind-safe palette from
qc_standards.py.
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from .qc_standards import (
    COLORBLIND_PALETTE,
)


def qc_figure_title(subject_id: str, session_id: str, title: str) -> str:
    """Build a standardized QC figure title.

    Parameters
    ----------
    subject_id : str
        Subject ID (without 'sub-' prefix).
    session_id : str
        Session ID (without 'ses-' prefix).
    title : str
        The main title text.

    Returns
    -------
    str
        Formatted title like "sub-X | ses-Y | Title".
    """
    parts = []
    if subject_id:
        parts.append(f"sub-{subject_id}")
    if session_id:
        parts.append(f"ses-{session_id}")
    parts.append(title)
    return " | ".join(parts)


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

    Parameters
    ----------
    data : npt.NDArray[Any]
        3D numpy array (x, y, z)
    axis : int, optional
        Slice axis (0=sagittal, 1=coronal, 2=axial). Default is axial.
    slice_idx : int, optional
        Index of slice to extract. If None, uses middle slice.

    Returns
    -------
    npt.NDArray[Any]
        2D numpy array with proper radiological orientation (flipped/rotated
        as needed for display)

    Raises
    ------
    ValueError
        If data is not 3D or axis is invalid
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

    Thin wrapper around :func:`pmriprep.utils.motion.calculate_framewise_displacement`
    that accepts the ``radius`` keyword (used by QC plotting code) and
    handles 12-column affine parameter arrays by extracting the first 6 columns.

    Parameters
    ----------
    motion_params : npt.NDArray[Any]
        Motion parameters array of shape (N, 6) in the common FSL-compatible
        format where columns are [rx, ry, rz, tx, ty, tz] (rotations
        in radians, translations in mm), or shape (N, 12) for affine
        parameters (uses first 6).
    radius : float, optional
        Head radius in mm for converting rotations to displacements.
        Default is 50.0mm.

    Returns
    -------
    npt.NDArray[np.float64]
        1D array of framewise displacement values (length N, with first
        element set to 0 as FD is undefined for first frame).

    Raises
    ------
    ValueError
        If motion_params has wrong shape
    """
    from pmriprep.utils.motion import (
        calculate_framewise_displacement as _calculate_fd,
    )

    motion_params = np.asarray(motion_params, dtype=np.float64)

    if motion_params.ndim != 2:
        raise ValueError(f"Expected 2D array, got {motion_params.ndim}D")

    if motion_params.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns (rx,ry,rz,tx,ty,tz), "
            f"got {motion_params.shape[1]}"
        )

    # Extract the 6 rigid-body parameters (handles 12-column affine)
    params = motion_params[:, :6]

    return _calculate_fd(params, head_radius=radius)  # type: ignore[return-value]


def _compute_brain_slice_indices(
    data: npt.NDArray[Any],
    axis: int,
    n_slices: int,
    mask: Optional[npt.NDArray[Any]] = None,
) -> npt.NDArray[np.intp]:
    """Compute evenly-spaced slice indices within the brain extent.

    Projects the mask (or nonzero data) onto the given axis to find the
    range of slices containing brain, then returns ``n_slices`` evenly
    spaced indices within that range.  Follows the niworkflows
    ``cuts_from_bbox()`` convention of excluding the very edge slices.

    Parameters
    ----------
    data : npt.NDArray[Any]
        3D volume used as fallback when *mask* is None.
    axis : int
        Projection axis (0=sagittal, 1=coronal, 2=axial).
    n_slices : int
        Number of slice indices to return.
    mask : npt.NDArray[Any], optional
        Optional binary brain mask.  When provided, the extent is
        derived from the mask; otherwise from ``data > 0``.

    Returns
    -------
    npt.NDArray[np.intp]
        1-D integer array of slice indices (length *n_slices*).
    """
    if mask is not None:
        binary = np.asarray(mask) > 0
    else:
        binary = np.asarray(data) > 0

    # Sum over all axes *except* the target axis → 1-D profile
    sum_axes = tuple(i for i in range(3) if i != axis)
    projection = np.sum(binary, axis=sum_axes)

    # Threshold at 5 % of peak to exclude noisy edge voxels
    threshold = 0.05 * float(np.max(projection)) if np.any(projection) else 0.0
    (indices,) = np.where(projection > threshold)

    if len(indices) >= 2:
        smin = int(indices[0])
        smax = int(indices[-1])
    else:
        # Fallback: full range with 10 % margin
        n_total = data.shape[axis]
        margin = int(n_total * 0.1)
        smin = margin
        smax = n_total - margin - 1

    # linspace with +2 then trim edges (niworkflows convention)
    slices = np.linspace(smin, smax, n_slices + 2)[1:-1].astype(np.intp)
    return slices


def plot_registration_qc_filmstrip(
    fixed: npt.NDArray[Any],
    moving: npt.NDArray[Any],
    output_path: Union[str, Path],
    mask: Optional[npt.NDArray[Any]] = None,
    moving_mask: Optional[npt.NDArray[Any]] = None,
    title: Optional[str] = None,
    n_slices: int = 7,
    contour_color: Optional[str] = None,
    voxel_sizes: Optional[tuple[float, float, float]] = None,
) -> Path:
    """Create a 3-row orthogonal filmstrip for registration QC.

    Generates a multi-slice filmstrip with three rows (axial, coronal,
    sagittal) showing the fixed image as grayscale background with
    contour lines of the moving image overlaid in color on a dark background.
    When masks are available, the fixed-image brain boundary is shown as a
    solid contour and the registered moving-image brain boundary as a dashed
    contour. This distinguishes outer-boundary agreement from internal tissue
    contrast.

    Parameters
    ----------
    fixed : npt.NDArray[Any]
        3D background image array (e.g. DSC/DCE mean)
    moving : npt.NDArray[Any]
        3D image for edge overlay (e.g. T1w in functional space)
    output_path : str or Path
        Path to save the output PNG image
    mask : npt.NDArray[Any], optional
        Optional fixed-image 3D brain mask used for slice selection, intensity
        scaling, and a solid outer-boundary contour.
    moving_mask : npt.NDArray[Any], optional
        Optional registered moving-image 3D brain mask shown as a dashed
        outer-boundary contour.
    title : str, optional
        Optional title for the figure
    n_slices : int, optional
        Number of slices per row (default 7)
    contour_color : str, optional
        Color for contour overlay. If None, uses palette primary.
    voxel_sizes : tuple of float, optional
        Voxel dimensions (dx, dy, dz) in mm for correct aspect
        ratios. If None, uses equal aspect.

    Returns
    -------
    Path
        Path to the saved figure

    Raises
    ------
    ValueError
        If image shapes don't match or arrays aren't 3D
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
    if moving_mask is not None:
        moving_mask = np.asarray(moving_mask)
        if moving_mask.shape != fixed.shape:
            raise ValueError(
                f"Moving mask shape {moving_mask.shape} "
                f"doesn't match image shape {fixed.shape}"
            )

    # Use colorblind-safe palette color for contours (cyan on dark bg)
    if contour_color is None:
        contour_color = COLORBLIND_PALETTE.secondary

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

    # Create the figure: 3 rows x n_slices columns (dark background)
    fig, axes = plt.subplots(
        3, n_slices, figsize=(n_slices * 2.2, 3 * 2.5),
        facecolor="black",
    )

    # Handle single slice case (axes would be 1D)
    if n_slices == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (label, axis) in enumerate(row_defs):
        aspect = aspect_ratios[row_idx]

        # Data-driven slice selection within brain extent
        slice_indices = _compute_brain_slice_indices(
            fixed, axis=axis, n_slices=n_slices, mask=mask,
        )

        # Axis label mapping for slice coordinate labels
        axis_labels = {2: "z", 1: "y", 0: "x"}
        coord_label = axis_labels[axis]

        for col_idx, slice_idx in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")

            # Get slices in radiological orientation
            fixed_slice = get_radiological_slice(
                fixed, axis=axis, slice_idx=int(slice_idx),
            )
            moving_slice = get_radiological_slice(
                moving, axis=axis, slice_idx=int(slice_idx),
            )
            fixed_mask_slice = (
                get_radiological_slice(mask, axis=axis, slice_idx=int(slice_idx))
                if mask is not None else None
            )
            moving_mask_slice = (
                get_radiological_slice(
                    moving_mask, axis=axis, slice_idx=int(slice_idx),
                )
                if moving_mask is not None else None
            )

            # Display the fixed image as background
            ax.imshow(
                fixed_slice,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                aspect=aspect,
            )

            # Overlay contour lines of the moving image
            # matplotlib contour() gives consistent thin lines matching
            # fmriprep's add_contours(linewidths=0.5)
            if np.any(moving_slice > 0):
                moving_valid = (
                    moving_slice[(fixed_mask_slice > 0) & (moving_slice > 0)]
                    if fixed_mask_slice is not None
                    else moving_slice[moving_slice > 0]
                )
                if len(moving_valid) > 0:
                    val_range = float(np.ptp(moving_valid))
                    if val_range < 1e-6:
                        # Binary / edge image: single level at midpoint
                        levels = [float(np.max(moving_valid)) * 0.5]
                    else:
                        # Continuous image: single level at median for
                        # clean tissue boundary visualization
                        levels = [float(np.percentile(moving_valid, 50))]
                    ax.contour(
                        moving_slice, levels=levels,
                        colors=[contour_color], linewidths=0.8,
                    )

            if fixed_mask_slice is not None and np.any(fixed_mask_slice > 0):
                ax.contour(
                    fixed_mask_slice, levels=[0.5],
                    colors=[COLORBLIND_PALETTE.warning], linewidths=1.5,
                )

            if moving_mask_slice is not None and np.any(moving_mask_slice > 0):
                ax.contour(
                    moving_mask_slice, levels=[0.5],
                    colors=[COLORBLIND_PALETTE.success], linewidths=1.2,
                    linestyles="dashed",
                )

            ax.axis("off")
            ax.set_title(
                f"{coord_label}={slice_idx}", fontsize=7, color="white",
            )

    plt.tight_layout()

    # Row labels — placed after tight_layout so positions are stable
    for row_idx, (label, _axis) in enumerate(row_defs):
        # Get the bounding box of the leftmost subplot in this row
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            bbox.x0 - 0.01, (bbox.y0 + bbox.y1) / 2, label,
            fontsize=9, fontweight="bold", rotation=90,
            ha="right", va="center", color="white",
        )

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=10, y=1.02, color="white")

    if mask is not None or moving_mask is not None:
        from matplotlib.lines import Line2D  # type: ignore[import-untyped]

        legend_handles = [
            Line2D([0], [0], color=contour_color, linewidth=0.8,
                   label="Moving-image intensity"),
        ]
        if mask is not None:
            legend_handles.append(
                Line2D(
                    [0], [0], color=COLORBLIND_PALETTE.warning,
                    linewidth=1.5, label="Fixed brain mask",
                )
            )
        if moving_mask is not None:
            legend_handles.append(
                Line2D(
                    [0], [0], color=COLORBLIND_PALETTE.success,
                    linewidth=1.2, linestyle="dashed",
                    label="Registered moving brain mask",
                )
            )
        legend = fig.legend(
            handles=legend_handles, loc="lower center", ncol=len(legend_handles),
            bbox_to_anchor=(0.5, -0.015), frameon=False, fontsize=7,
        )
        for text in legend.get_texts():
            text.set_color("white")

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    return output_path


def plot_segmentation_qc_filmstrip(
    background: npt.NDArray[Any],
    label_masks: list[tuple[str, npt.NDArray[Any], str]],
    output_path: Union[str, Path],
    mask: Optional[npt.NDArray[Any]] = None,
    title: Optional[str] = None,
    n_slices: int = 7,
    voxel_sizes: Optional[tuple[float, float, float]] = None,
) -> Path:
    """Create a 3-row orthogonal filmstrip for segmentation QC.

    Generates a multi-slice filmstrip with three rows (axial, coronal,
    sagittal) showing the background image as grayscale with colored
    contour outlines from one or more binary label masks overlaid.

    Parameters
    ----------
    background : npt.NDArray[Any]
        3D background image array (e.g. T1w post-contrast).
    label_masks : list of tuple
        List of (name, mask_3d, color) tuples. Each mask
        is a binary 3D array; contour edges are drawn in the given color.
    output_path : str or Path
        Path to save the output PNG image.
    mask : npt.NDArray[Any], optional
        Optional 3D brain mask to restrict display region.
    title : str, optional
        Optional title for the figure.
    n_slices : int, optional
        Number of slices per row (default 7).
    voxel_sizes : tuple of float, optional
        Voxel dimensions (dx, dy, dz) in mm for correct aspect
        ratios. If None, uses equal aspect.

    Returns
    -------
    Path
        Path to the saved figure.

    Raises
    ------
    ValueError
        If the background array is not 3D.
    """
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    from matplotlib.patches import Patch  # type: ignore[import-untyped]

    background = np.asarray(background)
    if background.ndim != 3:
        raise ValueError(f"Expected 3D background array, got {background.ndim}D")

    # Row definitions: (label, axis)
    row_defs: list[tuple[str, int]] = [
        ("Axial", 2),
        ("Coronal", 1),
        ("Sagittal", 0),
    ]

    # Compute per-plane aspect ratios from voxel sizes
    if voxel_sizes is not None:
        dx, dy, dz = voxel_sizes
        aspect_ratios = [dy / dx, dz / dx, dz / dy]
    else:
        aspect_ratios = [1.0, 1.0, 1.0]

    # Determine intensity range from background image
    if mask is not None:
        mask = np.asarray(mask)
        valid_voxels = background[mask > 0]
    else:
        valid_voxels = background[background > 0]

    if len(valid_voxels) > 0:
        vmin = float(np.percentile(valid_voxels, 1))
        vmax = float(np.percentile(valid_voxels, 99))
    else:
        vmin, vmax = 0.0, 1.0

    # Validate and collect label masks for contour overlay
    mask_data: list[tuple[str, npt.NDArray[np.bool_], str]] = []
    for name, lmask, color in label_masks:
        lmask = np.asarray(lmask, dtype=bool)
        if lmask.shape != background.shape:
            continue
        if not np.any(lmask):
            continue
        mask_data.append((name, lmask, color))

    # Create the figure: 3 rows x n_slices columns (dark background)
    fig, axes = plt.subplots(
        3, n_slices, figsize=(n_slices * 2.2, 3 * 2.5),
        facecolor="black",
    )
    if n_slices == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (label, axis) in enumerate(row_defs):
        aspect = aspect_ratios[row_idx]

        # Data-driven slice selection within brain extent
        slice_indices = _compute_brain_slice_indices(
            background, axis=axis, n_slices=n_slices, mask=mask,
        )

        axis_labels = {2: "z", 1: "y", 0: "x"}
        coord_label = axis_labels[axis]

        for col_idx, slice_idx in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")

            bg_slice = get_radiological_slice(
                background, axis=axis, slice_idx=int(slice_idx),
            )
            ax.imshow(bg_slice, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)

            # Overlay each label's contour
            for _name, lmask, color in mask_data:
                mask_slice = get_radiological_slice(
                    lmask.astype(np.float64), axis=axis, slice_idx=int(slice_idx),
                )
                if np.any(mask_slice > 0):
                    ax.contour(
                        mask_slice, levels=[0.5], colors=[color], linewidths=1.5,
                    )

            ax.axis("off")
            ax.set_title(
                f"{coord_label}={slice_idx}", fontsize=7, color="white",
            )

    plt.tight_layout()

    # Row labels
    for row_idx, (label, _axis) in enumerate(row_defs):
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            bbox.x0 - 0.01, (bbox.y0 + bbox.y1) / 2, label,
            fontsize=9, fontweight="bold", rotation=90,
            ha="right", va="center", color="white",
        )

    # Color legend below the figure
    if mask_data:
        legend_handles = [
            Patch(facecolor=color, edgecolor=color, label=name)
            for name, _lmask, color in mask_data
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=8,
            frameon=True,
            fancybox=True,
            shadow=False,
            bbox_to_anchor=(0.5, -0.02),
            labelcolor="white",
            facecolor="black",
            edgecolor="gray",
        )

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=10, y=1.02, color="white")

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    return output_path


def plot_motion_qc(
    motion_params: npt.NDArray[Any],
    output_path: Union[str, Path],
    tr: float = 1.0,
    title: Optional[str] = None,
    radius: float = 50.0,
    subject_id: str = "",
    session_id: str = "",
    note: Optional[str] = None,
    fd_threshold: float = 0.5,
) -> tuple[Path, dict[str, Any]]:
    """Create a 3-panel motion QC plot.

    Generates a three-panel figure showing:
    1. Top panel: Translation parameters (mm)
    2. Middle panel: Rotation parameters (degrees)
    3. Bottom panel: Framewise displacement with threshold and outlier shading

    Parameters
    ----------
    motion_params : npt.NDArray[Any]
        Motion parameters array of shape (N, 6) in the common FSL-compatible
        format where columns are [rx, ry, rz, tx, ty, tz] (rotations
        in radians, translations in mm), or shape (N, 12) for affine
        parameters (uses first 6).
    output_path : str or Path
        Path to save the output PNG image
    tr : float, optional
        Repetition time in seconds (unused, x-axis is volume number)
    title : str, optional
        Optional title for the figure
    radius : float, optional
        Head radius in mm for FD calculation (default 50.0)
    subject_id, session_id : str, optional
        Participant and session identifiers for the figure title.
    note : str, optional
        Caption describing motion estimation and transform application.
    fd_threshold : float, optional
        Configured strict FD exceedance threshold in millimetres.

    Returns
    -------
    tuple
        Tuple of (path to saved figure, motion metrics dict with
        ``fd_mean``, ``fd_max``, ``n_outliers``, ``percent_outliers``,
        ``fd_threshold``).

    Raises
    ------
    ValueError
        If motion_params has wrong shape
    """
    # Import matplotlib here to avoid import overhead when not needed
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    from pmriprep.utils.motion import summarize_motion

    motion_params = np.atleast_2d(np.asarray(motion_params))

    if motion_params.ndim != 2 or motion_params.shape[1] < 6:
        raise ValueError(
            f"Expected shape (N, 6) or (N, 12), got {motion_params.shape}"
        )

    # Extract rotations in radians followed by translations in millimetres.
    params = motion_params[:, :6]
    n_frames = params.shape[0]
    volumes = np.arange(n_frames)

    # Calculate framewise displacement
    motion_metrics = summarize_motion(params, fd_threshold, radius)
    fd = np.asarray(motion_metrics["fd_list"])

    # Color scheme
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

    # Overall title
    if title:
        fig.suptitle(title, fontsize=11)
    elif subject_id or session_id:
        fig.suptitle(qc_figure_title(subject_id, session_id, "Motion QC"), fontsize=11)

    fig.tight_layout(rect=(0, 0.12, 1, 0.985))

    # Describe motion estimation and whether correction was applied.
    if note:
        fig.text(
            0.5, 0.04, note, ha="center", va="center", fontsize=9,
            color="#CC3311", wrap=True,
        )

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path, motion_metrics
