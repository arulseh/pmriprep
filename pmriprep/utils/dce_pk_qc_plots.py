"""DCE pharmacokinetic QC visualization plots.

This module provides three QC plot functions for DCE perfusion processing:
1. Preprocessing QC — signal quality, enhancement, tSNR, CV
2. PK modeling QC — model fit at example voxel, AIF, Ktrans map
3. PK parameter maps — multi-slice spatial parameter maps

All visualizations use the colorblind-safe palette from qc_standards.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("nipype.interface")


def _select_qc_voxel(
    ktrans: npt.NDArray[Any],
    brain_mask: npt.NDArray[np.bool_],
    qc_mask: npt.NDArray[Any] | None,
    binary_erosion: Any,
) -> tuple[int, int, int]:
    """Select an example voxel for PK modeling QC display.

    Tiered selection to avoid mask edges and noisy voxels:
        1. Eroded QC mask — voxel nearest median Ktrans among good-fit
           voxels, eroded to stay away from edges
        2. QC mask (uneroded) — if erosion leaves nothing
        3. Eroded brain mask with nonzero Ktrans
        4. Brain mask center (ultimate fallback)

    Args:
        ktrans: 3D Ktrans map.
        brain_mask: 3D boolean brain mask.
        qc_mask: 3D QC mask (physiological + R² filtered). May be None.
        binary_erosion: scipy.ndimage.binary_erosion function.

    Returns:
        (x, y, z) voxel coordinates.
    """
    def _near_median(search_mask: npt.NDArray[np.bool_]) -> tuple[int, int, int] | None:
        """Find voxel closest to median Ktrans within search_mask."""
        vals = ktrans[search_mask]
        if len(vals) == 0 or not np.any(vals > 0):
            return None
        target = float(np.median(vals[vals > 0]))
        candidate_mask = search_mask & (ktrans > 0)
        diffs = np.full(ktrans.shape, np.inf)
        diffs[candidate_mask] = np.abs(ktrans[candidate_mask] - target)
        flat_idx = int(np.argmin(diffs))
        idx = np.unravel_index(flat_idx, ktrans.shape)
        return int(idx[0]), int(idx[1]), int(idx[2])

    # Tier 1: Eroded QC mask (best — avoids edges, only good-fit voxels)
    if qc_mask is not None:
        qc_bool = qc_mask > 0.5
        eroded_qc = binary_erosion(qc_bool, iterations=2)
        if isinstance(eroded_qc, np.ndarray) and np.any(eroded_qc):
            result = _near_median(eroded_qc)
            if result is not None:
                return result

        # Tier 2: Uneroded QC mask
        if np.any(qc_bool):
            result = _near_median(qc_bool)
            if result is not None:
                return result

    # Tier 3: Eroded brain mask with nonzero Ktrans
    eroded_brain = binary_erosion(brain_mask, iterations=3)
    if isinstance(eroded_brain, np.ndarray):
        eroded_with_ktrans = eroded_brain & (ktrans > 0)
        if np.any(eroded_with_ktrans):
            result = _near_median(eroded_with_ktrans)
            if result is not None:
                return result

    # Tier 4: Fallback — center of brain mask
    coords = np.where(brain_mask)
    mid = len(coords[0]) // 2
    return int(coords[0][mid]), int(coords[1][mid]), int(coords[2][mid])


def generate_preprocessing_qc_plot(
    dce_signal_4d: npt.NDArray[Any],
    brain_mask: npt.NDArray[Any],
    baseline_frames: int,
    bolus_arrival_frame: int,
    volume_tr_s: float,
    subject_id: str = "",
    session_id: str = "",
    output_path: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Generate a 2-row preprocessing QC plot for DCE data.

    Layout (GridSpec 2×4, graphs span 2 cols, images span 1):
        Row 0: Global Signal Time Course (0-1) | Peak Signal (2) | Enhancement Map (3)
        Row 1: Signal Intensity Histogram (0-1) | tSNR Map (2) | CV Map (3)

    FD and summary text are omitted — FD is shown in the dedicated
    motion_qc.png, and summary stats are rendered as HTML metric boxes.

    Args:
        dce_signal_4d: 4D raw DCE signal array (x, y, z, t).
        brain_mask: 3D boolean brain mask.
        baseline_frames: Number of baseline (pre-bolus) frames.
        bolus_arrival_frame: Frame index of bolus arrival.
        volume_tr_s: Temporal resolution between volumes in seconds.
        subject_id: Subject identifier for title.
        session_id: Session identifier for title.
        output_path: Path to save PNG. If None, uses CWD.

    Returns:
        Tuple of (path to saved QC plot PNG, stats dict with
        ``peak_enh_pct`` and ``mean_tsnr``).
    """
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    from matplotlib.gridspec import GridSpec

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice

    mask_bool = brain_mask > 0.5
    n_volumes = dce_signal_4d.shape[3]

    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 4, figure=fig, wspace=0.30, hspace=0.35)

    title = "DCE Preprocessing QC"
    if subject_id:
        title += f" — sub-{subject_id}"
    if session_id:
        title += f" ses-{session_id}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # --- (0, 0-1) Global Signal Time Course ---
    ax = fig.add_subplot(gs[0, 0:2])
    global_signal = np.array([
        np.mean(dce_signal_4d[..., t][mask_bool])
        for t in range(n_volumes)
    ])
    time_s = np.arange(n_volumes) * volume_tr_s
    ax.plot(time_s, global_signal, color=COLORBLIND_PALETTE.primary, linewidth=1.2)
    # Baseline shading
    if baseline_frames > 0:
        ax.axvspan(0, (baseline_frames - 0.5) * volume_tr_s, alpha=0.15,
                   color=COLORBLIND_PALETTE.success, label="Baseline")
        ax.axvline((baseline_frames - 0.5) * volume_tr_s, color="gray",
                   linestyle="--", linewidth=0.8, label="Baseline end")
    # Bolus arrival
    if bolus_arrival_frame > 0:
        ax.axvline(bolus_arrival_frame * volume_tr_s, color="#EE7733",
                   linestyle="--", linewidth=1.0, label="Bolus arrival")
    # Peak enhancement annotation
    baseline_mean = np.mean(global_signal[:max(1, baseline_frames)])
    peak_val = np.max(global_signal)
    peak_enh_pct = (peak_val - baseline_mean) / max(baseline_mean, 1e-6) * 100.0
    ax.set_title("Global Signal Time Course", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Signal")
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.02, 0.95, f"Peak enhancement: {peak_enh_pct:.1f}%",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- (0, 2) Peak Signal Image ---
    ax = fig.add_subplot(gs[0, 2])
    peak_frame = int(np.argmax(global_signal))
    peak_vol = dce_signal_4d[..., peak_frame]
    mid_slice = get_radiological_slice(peak_vol, axis=2)
    brain_vals = peak_vol[mask_bool]
    vmax = float(np.percentile(brain_vals, 99)) if brain_vals.size > 0 else 1.0
    ax.imshow(mid_slice, cmap="gray", vmin=0, vmax=vmax, aspect="equal")
    ax.set_title(f"Peak Signal (frame {peak_frame})", fontsize=10)
    ax.text(0.02, 0.5, "R", transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold", va="center")
    ax.text(0.98, 0.5, "L", transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold", va="center", ha="right")
    ax.axis("off")

    # --- (0, 3) Enhancement Map (%) ---
    ax = fig.add_subplot(gs[0, 3])
    baseline_vol = np.mean(dce_signal_4d[..., :max(1, baseline_frames)], axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        enhancement = np.where(
            baseline_vol > 1.0,
            (peak_vol - baseline_vol) / baseline_vol * 100.0,
            0.0,
        )
    enhancement[~mask_bool] = np.nan
    enh_slice = get_radiological_slice(enhancement, axis=2)
    im = ax.imshow(enh_slice, cmap="hot", vmin=0, vmax=100, aspect="equal")
    ax.set_title("Enhancement Map (%)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

    # --- (1, 0-1) Signal Intensity Histogram ---
    ax = fig.add_subplot(gs[1, 0:2])
    baseline_vals = dce_signal_4d[..., 0][mask_bool]
    peak_vals = dce_signal_4d[..., peak_frame][mask_bool]
    all_vals = np.concatenate([baseline_vals, peak_vals])
    hist_min = float(np.percentile(all_vals, 1))
    hist_max = float(np.percentile(all_vals, 99))
    bins = np.linspace(hist_min, hist_max, 80)
    ax.hist(baseline_vals, bins=bins, alpha=0.5, density=True,
            color=COLORBLIND_PALETTE.primary, label="Baseline (frame 0)")
    ax.hist(peak_vals, bins=bins, alpha=0.5, density=True,
            color=COLORBLIND_PALETTE.error, label=f"Peak (frame {peak_frame})")
    ax.set_title("Signal Intensity Distribution", fontsize=10)
    ax.set_xlabel("Signal Intensity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # --- (1, 2) tSNR Map ---
    ax = fig.add_subplot(gs[1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        temporal_mean = np.mean(dce_signal_4d, axis=3)
        temporal_std = np.std(dce_signal_4d, axis=3)
        tsnr = np.where(temporal_std > 1e-6, temporal_mean / temporal_std, 0.0)
    tsnr[~mask_bool] = np.nan
    mean_tsnr = float(np.nanmean(tsnr[mask_bool]))
    tsnr_slice = get_radiological_slice(tsnr, axis=2)
    tsnr_vmax = float(np.nanpercentile(tsnr[mask_bool], 95)) if np.any(mask_bool) else 50.0
    im = ax.imshow(tsnr_slice, cmap="hot", vmin=0, vmax=tsnr_vmax, aspect="equal")
    ax.set_title("tSNR Map", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="tSNR")

    # --- (1, 3) CV Map ---
    ax = fig.add_subplot(gs[1, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(temporal_mean > 1e-6, temporal_std / temporal_mean, 0.0)
    cv[~mask_bool] = np.nan
    cv_slice = get_radiological_slice(cv, axis=2)
    im = ax.imshow(cv_slice, cmap="viridis", vmin=0, vmax=0.2, aspect="equal")
    ax.set_title("Coefficient of Variation", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CV")

    if output_path is None:
        output_path = str(Path.cwd() / "dce_preprocessing_qc.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = {
        "peak_enh_pct": round(peak_enh_pct, 1),
        "mean_tsnr": round(mean_tsnr, 1),
    }
    return str(out), stats


def generate_pk_modeling_qc_plot(
    conc_data: npt.NDArray[Any],
    signal_data: npt.NDArray[Any] | None,
    aif: npt.NDArray[Any],
    time_points_s: npt.NDArray[Any],
    ktrans: npt.NDArray[Any],
    ve: npt.NDArray[Any] | None,
    vp: npt.NDArray[Any] | None,
    r2_map: npt.NDArray[Any],
    brain_mask: npt.NDArray[Any],
    fitted_curves: npt.NDArray[Any] | None,
    model_name: str,
    baseline_frames: int,
    qc_mask: npt.NDArray[Any] | None = None,
    qc_fitted_curve: npt.NDArray[Any] | None = None,
    fp: npt.NDArray[Any] | None = None,
    ps: npt.NDArray[Any] | None = None,
    subject_id: str = "",
    session_id: str = "",
    output_path: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate a 2x2 PK modeling QC plot.

    Panels:
        (0,0) Raw signal at example voxel with S0 baseline
        (0,1) Tissue concentration with model fit curve and R^2 badge
        (1,0) Arterial Input Function (AIF)
        (1,1) Ktrans map with example voxel marker

    Example voxel selection (tiered):
        1. Enhancing voxels within eroded QC mask (top-25% Ktrans, good R²)
        2. QC mask voxels (good fit, physiological values)
        3. Eroded brain mask with nonzero Ktrans
        4. Brain mask center (fallback)

    Args:
        conc_data: 4D concentration array (x, y, z, t).
        signal_data: 4D raw signal array (optional, for signal panel).
        aif: 1D AIF concentration array.
        time_points_s: 1D time array in seconds.
        ktrans: 3D Ktrans map.
        ve: 3D ve map.
        vp: 3D vp map (None if model has no vp).
        r2_map: 3D R^2 goodness-of-fit map.
        brain_mask: 3D boolean brain mask.
        fitted_curves: 4D fitted curves (optional).
        model_name: Name of the PK model used.
        baseline_frames: Number of baseline frames.
        qc_mask: 3D QC mask (physiological bounds + R² filter). Optional.
        qc_fitted_curve: Pre-computed 1D fitted curve for QC voxel. Optional.
        fp: 3D Fp map for 2CXM (optional).
        ps: 3D PS map for 2CXM (optional).
        subject_id: Subject identifier for title.
        session_id: Session identifier for title.
        output_path: Path to save PNG. If None, uses CWD.

    Returns:
        Tuple of (path to saved QC plot PNG, voxel metrics dict).
    """
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    from scipy.ndimage import binary_erosion

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice

    mask_bool = brain_mask > 0.5

    # --- Select example voxel (tiered) ---
    vx, vy, vz = _select_qc_voxel(ktrans, mask_bool, qc_mask, binary_erosion)

    # Extract voxel data
    voxel_conc = conc_data[vx, vy, vz, :]
    voxel_r2 = float(r2_map[vx, vy, vz])
    voxel_ktrans = float(ktrans[vx, vy, vz])
    voxel_ve = float(ve[vx, vy, vz]) if ve is not None else None
    voxel_vp = float(vp[vx, vy, vz]) if vp is not None else None

    # Get fitted curve for the example voxel
    fitted_curve: Optional[npt.NDArray[Any]] = None
    if qc_fitted_curve is not None and len(qc_fitted_curve) > 0:
        fitted_curve = qc_fitted_curve
    elif fitted_curves is not None:
        fitted_curve = fitted_curves[vx, vy, vz, :]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f"PK Modeling QC — {model_name}"
    if subject_id:
        title += f" — sub-{subject_id}"
    if session_id:
        title += f" ses-{session_id}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # --- (0,0) Signal Time Course ---
    ax = axes[0, 0]
    if signal_data is not None:
        voxel_signal = signal_data[vx, vy, vz, :]
        ax.plot(time_points_s, voxel_signal, "o-", color=COLORBLIND_PALETTE.primary,
                markersize=3, linewidth=1, label="Signal")
        # S0 baseline
        s0 = float(np.mean(voxel_signal[:max(1, baseline_frames)]))
        ax.axhline(y=s0, color=COLORBLIND_PALETTE.error, linestyle="--",
                   linewidth=1, label=f"S0 = {s0:.0f}")
        ax.set_ylabel("Signal Intensity")
    else:
        ax.plot(time_points_s, voxel_conc, "o-", color=COLORBLIND_PALETTE.primary,
                markersize=3, linewidth=1, label="Concentration")
        ax.set_ylabel("Concentration (mM)")
    ax.set_title(f"Voxel ({vx}, {vy}, {vz})", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Concentration + Model Fit ---
    ax = axes[0, 1]
    ax.plot(time_points_s, voxel_conc, "o", color=COLORBLIND_PALETTE.primary,
            markersize=4, label="Measured", alpha=0.7)
    if fitted_curve is not None and np.any(fitted_curve != 0):
        ax.plot(time_points_s[:len(fitted_curve)], fitted_curve,
                "-", color=COLORBLIND_PALETTE.error, linewidth=1.5, label="Model fit")
    ax.set_title("Tissue Concentration & Model Fit", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration (mM)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,0) AIF ---
    ax = axes[1, 0]
    aif_time = time_points_s[:len(aif)]
    ax.plot(aif_time, aif, color=COLORBLIND_PALETTE.secondary, linewidth=1.5)
    ax.fill_between(aif_time, aif, alpha=0.2, color=COLORBLIND_PALETTE.secondary)
    ax.set_title("Arterial Input Function", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("AIF Concentration (mM)")
    ax.grid(True, alpha=0.3)

    # --- (1,1) Ktrans Map with voxel marker ---
    ax = axes[1, 1]
    ktrans_display = ktrans.copy().astype(float)
    ktrans_display[~mask_bool] = np.nan
    ktrans_slice = get_radiological_slice(ktrans_display, axis=2, slice_idx=vz)
    ktrans_nonzero = ktrans[mask_bool & (ktrans > 0)]
    if len(ktrans_nonzero) > 0:
        kt_vmax = float(np.percentile(ktrans_nonzero, 95))
    else:
        kt_vmax = 0.5
    im = ax.imshow(ktrans_slice, cmap="hot", vmin=0, vmax=kt_vmax, aspect="equal")
    ax.set_title(f"Ktrans Map (z={vz})", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="min\u207b\u00b9")
    # R/L markers
    ax.text(0.02, 0.5, "R", transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold", va="center")
    ax.text(0.98, 0.5, "L", transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold", va="center", ha="right")
    # Voxel marker — transform (vx, vy) to radiological slice coordinates
    # get_radiological_slice for axial: rot90 + fliplr
    # After rot90: (row, col) = (shape[0]-1-col_orig, row_orig)
    # After fliplr on rotated: col -> shape[1]-1-col
    # Original: row_orig = vx, col_orig = vy
    # After rot90: row = shape[0]-1 - vy, col = vx
    # After fliplr: col = (shape of rotated)[1] - 1 - vx
    ny = ktrans.shape[1]
    marker_row = ny - 1 - vy
    marker_col = ktrans.shape[0] - 1 - vx
    ax.plot(marker_col, marker_row, "*", color="cyan", markersize=12, markeredgecolor="black",
            markeredgewidth=0.5)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    if output_path is None:
        output_path = str(Path.cwd() / "pk_modeling_qc.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Build voxel metrics for HTML rendering
    voxel_metrics: dict[str, Any] = {
        "qc_voxel": f"({vx}, {vy}, {vz})",
        "qc_voxel_ktrans": round(voxel_ktrans, 4),
        "qc_voxel_fit_r2": round(voxel_r2, 4),
    }
    if voxel_ve is not None:
        voxel_metrics["qc_voxel_ve"] = round(voxel_ve, 4)
    if voxel_vp is not None:
        voxel_metrics["qc_voxel_vp"] = round(voxel_vp, 4)
    if fp is not None:
        voxel_metrics["qc_voxel_fp"] = round(float(fp[vx, vy, vz]), 4)
    if ps is not None:
        voxel_metrics["qc_voxel_ps"] = round(float(ps[vx, vy, vz]), 4)

    return str(out), voxel_metrics


def generate_pk_parameter_maps_qc_plot(
    param_maps: dict[str, npt.NDArray[Any]],
    brain_mask: npt.NDArray[Any],
    subject_id: str = "",
    session_id: str = "",
    output_path: str | None = None,
) -> str:
    """Generate a multi-row parameter map QC plot.

    Each row shows one PK parameter across 3 evenly-spaced axial slices
    (25%, 50%, 75% of brain extent). Per-row colorbar with unit labels.

    Args:
        param_maps: Dict mapping parameter name to 3D array.
            Expected keys: "Ktrans", "ve", optionally "vp".
        brain_mask: 3D boolean brain mask.
        subject_id: Subject identifier for title.
        session_id: Session identifier for title.
        output_path: Path to save PNG. If None, uses CWD.

    Returns:
        Path to the saved QC plot PNG.
    """
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    from pmriprep.utils.qc_utils import get_radiological_slice

    mask_bool = brain_mask > 0.5

    # Parameter display config: (colormap, vmin, vmax_mode, unit_label)
    # vmax_mode: "pct95" = 95th percentile of nonzero, "fixed" = use vmax directly
    param_config: dict[str, tuple[str, float, str, float | None, str]] = {
        "Ktrans": ("hot", 0.0, "pct95", None, "min\u207b\u00b9"),
        "ve": ("viridis", 0.0, "fixed", 1.0, "fraction"),
        "vp": ("plasma", 0.0, "pct95", None, "fraction"),
        "Fp": ("inferno", 0.0, "pct95", None, "mL/100g/min"),
        "PS": ("magma", 0.0, "pct95", None, "mL/100g/min"),
    }

    # Filter to params that exist in the input
    params_to_plot = [(name, param_maps[name]) for name in param_config if name in param_maps]
    n_params = len(params_to_plot)

    if n_params == 0:
        logger.warning("No parameter maps provided for QC plot")
        return ""

    # Determine slice positions (25%, 50%, 75% of brain z-extent)
    z_coords = np.where(mask_bool)[2]
    if len(z_coords) == 0:
        z_min, z_max = 0, brain_mask.shape[2] - 1
    else:
        z_min, z_max = int(z_coords.min()), int(z_coords.max())
    z_positions = np.linspace(z_min, z_max, 5, dtype=int)[1:4]  # 25%, 50%, 75%

    n_cols = 3
    fig, axes = plt.subplots(n_params, n_cols, figsize=(4.5 * n_cols, 4 * n_params))

    # Handle single-row case
    if n_params == 1:
        axes = axes[np.newaxis, :]

    title = "PK Parameter Maps"
    if subject_id:
        title += f" — sub-{subject_id}"
    if session_id:
        title += f" ses-{session_id}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    for row_idx, (name, data) in enumerate(params_to_plot):
        cmap, vmin, vmax_mode, vmax_fixed, unit_label = param_config[name]

        # Determine vmax
        data_masked = data[mask_bool & (data > 0)]
        if vmax_mode == "pct95" and len(data_masked) > 0:
            vmax = float(np.percentile(data_masked, 95))
        elif vmax_fixed is not None:
            vmax = vmax_fixed
        else:
            vmax = 1.0

        # Compute statistics
        if len(data_masked) > 0:
            mu = float(np.mean(data_masked))
            sigma = float(np.std(data_masked))
        else:
            mu, sigma = 0.0, 0.0

        # Prepare display data (mask non-brain as NaN)
        display = data.copy().astype(float)
        display[~mask_bool] = np.nan

        for col_idx, z_idx in enumerate(z_positions):
            ax = axes[row_idx, col_idx]
            slc = get_radiological_slice(display, axis=2, slice_idx=int(z_idx))
            im = ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.axis("off")

            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(f"z = {z_idx}", fontsize=9)

        # Row label on left
        axes[row_idx, 0].text(
            -0.08, 0.5, f"{name}\n(\u03bc={mu:.4f}, \u03c3={sigma:.4f})",
            transform=axes[row_idx, 0].transAxes,
            fontsize=9, fontweight="bold", rotation=90,
            ha="right", va="center",
        )

        # Per-row colorbar
        cbar = fig.colorbar(im, ax=axes[row_idx, :].tolist(), fraction=0.03,
                            pad=0.06, label=f"{name} ({unit_label})")
        cbar.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0.06, 0.02, 0.92, 0.96))

    if output_path is None:
        output_path = str(Path.cwd() / "pk_parameter_maps_qc.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return str(out)
