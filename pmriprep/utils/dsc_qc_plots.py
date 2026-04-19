"""DSC perfusion QC visualization plots.

This module provides QC plot functions for DSC perfusion processing:
1. Preprocessing QC — signal quality, signal drop, tSNR, CNR
2. Parameter maps QC — multi-slice spatial parameter maps
3. Fit QC — model fit at example voxel (all deconvolution methods)
4. Leakage correction QC — K2, AUC decomposition, correction effect

All visualizations use the colorblind-safe palette from qc_standards.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dsc_constants import QC_CNR_THRESHOLD

logger = logging.getLogger("nipype.interface")


def _select_dsc_qc_voxel(
    cbf: npt.NDArray[Any],
    brain_mask: npt.NDArray[np.bool_],
    binary_erosion: Any,
    enhancing_mask: npt.NDArray[Any] | None = None,
) -> tuple[int, int, int]:
    """Select an example voxel for DSC fit QC display.

    Tiered selection (adapted from DCE _select_qc_voxel for CBF):
        1. Enhancing tumor mask — clinically most relevant region
        2. Eroded brain mask with nonzero CBF — voxel nearest median CBF
        3. Brain mask center (fallback)

    Args:
        cbf: 3D CBF map.
        brain_mask: 3D boolean brain mask.
        binary_erosion: scipy.ndimage.binary_erosion function.
        enhancing_mask: 3D enhancing tumor mask (optional). In native
            DSC space. When available, voxel selection strongly prefers
            the enhancing region.

    Returns:
        (x, y, z) voxel coordinates.
    """
    def _near_median(search_mask: npt.NDArray[np.bool_]) -> tuple[int, int, int] | None:
        vals = cbf[search_mask]
        if len(vals) == 0 or not np.any(vals > 0):
            return None
        target = float(np.median(vals[vals > 0]))
        candidate_mask = search_mask & (cbf > 0)
        diffs = np.full(cbf.shape, np.inf)
        diffs[candidate_mask] = np.abs(cbf[candidate_mask] - target)
        flat_idx = int(np.argmin(diffs))
        idx = np.unravel_index(flat_idx, cbf.shape)
        return int(idx[0]), int(idx[1]), int(idx[2])

    # Tier 1: Enhancing tumor with nonzero CBF
    if enhancing_mask is not None:
        enh_bool = enhancing_mask > 0.5
        enh_with_cbf = enh_bool & (cbf > 0)
        if np.any(enh_with_cbf):
            result = _near_median(enh_with_cbf)
            if result is not None:
                return result

    # Tier 2: Eroded brain mask with nonzero CBF
    eroded_brain = binary_erosion(brain_mask, iterations=3)
    if isinstance(eroded_brain, np.ndarray):
        eroded_with_cbf = eroded_brain & (cbf > 0)
        if np.any(eroded_with_cbf):
            result = _near_median(eroded_with_cbf)
            if result is not None:
                return result

    # Tier 3: Fallback — center of brain mask
    coords = np.where(brain_mask)
    mid = len(coords[0]) // 2
    return int(coords[0][mid]), int(coords[1][mid]), int(coords[2][mid])


def generate_dsc_preprocessing_qc_plot(
    dsc_signal_4d: npt.NDArray[Any],
    brain_mask: npt.NDArray[Any],
    baseline_frames: int,
    volume_tr_s: float,
    te_s: float | None = None,
    subject_id: str = "",
    session_id: str = "",
    output_path: str | None = None,
    voxel_size_mm: tuple[float, float, float] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate a 2-row preprocessing QC plot for DSC data.

    Layout (GridSpec 2x4, graphs span 2 cols, images span 1):
        Row 0: Global Signal Time Course (0-1) | Minimum Signal (2) | Signal Drop Map (3)
        Row 1: Signal Intensity Histogram (0-1) | tSNR Map (2) | CNR Map (3)

    Key DSC differences from DCE:
    - Global signal shows DROP (not rise)
    - "Minimum Signal Image" replaces "Peak Signal Image"
    - "Signal Drop Map (%)" replaces "Enhancement Map (%)"
    - CNR map replaces CV map

    Args:
        dsc_signal_4d: 4D raw DSC signal array (x, y, z, t).
        brain_mask: 3D boolean brain mask.
        baseline_frames: Number of baseline (pre-bolus) frames.
        volume_tr_s: Temporal resolution between volumes in seconds.
        subject_id: Subject identifier for title.
        session_id: Session identifier for title.
        output_path: Path to save PNG. If None, uses CWD.

    Returns:
        Tuple of (path to saved QC plot PNG, stats dict with
        ``peak_drop_pct``, ``mean_tsnr``, ``mean_cnr``, ``fraction_cnr_gt_4``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice

    mask_bool = brain_mask > 0.5
    n_volumes = dsc_signal_4d.shape[3]

    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 4, figure=fig, wspace=0.30, hspace=0.35)

    from pmriprep.utils.qc_utils import qc_figure_title
    fig.suptitle(
        qc_figure_title(subject_id, session_id, "DSC Preprocessing QC"),
        fontsize=13, fontweight="bold", y=0.98,
    )

    # --- (0, 0-1) Global Signal Time Course (signal DROP) ---
    ax = fig.add_subplot(gs[0, 0:2])
    global_signal = np.array([
        np.mean(dsc_signal_4d[..., t][mask_bool])
        for t in range(n_volumes)
    ])
    time_s = np.arange(n_volumes) * volume_tr_s
    ax.plot(time_s, global_signal, color=COLORBLIND_PALETTE.primary, linewidth=1.2)
    # S0 estimation window shading
    if baseline_frames > 0:
        ax.axvspan(0, (baseline_frames - 0.5) * volume_tr_s, alpha=0.15,
                   color=COLORBLIND_PALETTE.success, label="S0 window")
    # Peak signal drop annotation
    baseline_mean = np.mean(global_signal[:max(1, baseline_frames)])
    min_val = np.min(global_signal)
    peak_drop_pct = (baseline_mean - min_val) / max(baseline_mean, 1e-6) * 100.0
    ax.set_title("Global Signal Time Course", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Signal")
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.02, 0.95, f"Peak signal drop: {peak_drop_pct:.1f}%",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- (0, 2) Minimum Signal Image (at TTM) ---
    ax = fig.add_subplot(gs[0, 2])
    min_frame = int(np.argmin(global_signal))
    min_vol = dsc_signal_4d[..., min_frame]
    mid_slice = get_radiological_slice(min_vol, axis=2)
    brain_vals = min_vol[mask_bool]
    vmax = float(np.percentile(brain_vals, 99)) if brain_vals.size > 0 else 1.0
    ax.imshow(mid_slice, cmap="gray", vmin=0, vmax=vmax, aspect="equal")
    ax.set_title(f"Minimum Signal (frame {min_frame})", fontsize=10)
    ax.text(0.02, 0.5, "R", transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold", va="center")
    ax.text(0.98, 0.5, "L", transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold", va="center", ha="right")
    ax.axis("off")

    # --- (0, 3) Signal Drop Map (%) ---
    ax = fig.add_subplot(gs[0, 3])
    baseline_vol = np.mean(dsc_signal_4d[..., :max(1, baseline_frames)], axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal_drop = np.where(
            baseline_vol > 1.0,
            (baseline_vol - min_vol) / baseline_vol * 100.0,
            0.0,
        )
    signal_drop[~mask_bool] = np.nan
    drop_slice = get_radiological_slice(signal_drop, axis=2)
    im = ax.imshow(drop_slice, cmap="hot", vmin=0, vmax=100, aspect="equal")
    ax.set_title("Signal Drop Map (%)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

    # --- (1, 0-1) Signal Intensity Histogram ---
    ax = fig.add_subplot(gs[1, 0:2])
    baseline_vals = dsc_signal_4d[..., 0][mask_bool]
    min_vals = dsc_signal_4d[..., min_frame][mask_bool]
    all_vals = np.concatenate([baseline_vals, min_vals])
    hist_min = float(np.percentile(all_vals, 1))
    hist_max = float(np.percentile(all_vals, 99))
    bins = np.linspace(hist_min, hist_max, 80).tolist()
    ax.hist(baseline_vals, bins=bins, alpha=0.5, density=True,
            color=COLORBLIND_PALETTE.primary, label="Baseline (frame 0)")
    ax.hist(min_vals, bins=bins, alpha=0.5, density=True,
            color=COLORBLIND_PALETTE.error, label=f"Minimum (frame {min_frame})")
    ax.set_title("Signal Intensity Distribution", fontsize=10)
    ax.set_xlabel("Signal Intensity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # --- (1, 2) tSNR Map ---
    ax = fig.add_subplot(gs[1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        temporal_mean = np.mean(dsc_signal_4d, axis=3)
        temporal_std = np.std(dsc_signal_4d, axis=3)
        tsnr = np.where(temporal_std > 1e-6, temporal_mean / temporal_std, 0.0)
    tsnr[~mask_bool] = np.nan
    mean_tsnr = float(np.nanmean(tsnr[mask_bool]))
    tsnr_slice = get_radiological_slice(tsnr, axis=2)
    tsnr_vmax = float(np.nanpercentile(tsnr[mask_bool], 95)) if np.any(mask_bool) else 50.0
    im = ax.imshow(tsnr_slice, cmap="hot", vmin=0, vmax=tsnr_vmax, aspect="equal")
    ax.set_title("tSNR Map", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="tSNR")

    # --- (1, 3) CNR Map ---
    ax = fig.add_subplot(gs[1, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_std_3d = np.std(dsc_signal_4d[..., :max(1, baseline_frames)], axis=3)
        peak_drop_3d = baseline_vol - min_vol
        cnr = np.where(baseline_std_3d > 1e-6, peak_drop_3d / baseline_std_3d, 0.0)
    cnr[~mask_bool] = np.nan
    mean_cnr = float(np.nanmean(cnr[mask_bool]))
    # CNR coverage metric: fraction of brain voxels with CNR > threshold (Rose criterion)
    n_brain_voxels = np.sum(mask_bool)
    if n_brain_voxels > 0:
        fraction_cnr_gt_4 = float(np.sum(cnr[mask_bool] > QC_CNR_THRESHOLD) / n_brain_voxels)
    else:
        fraction_cnr_gt_4 = 0.0
    cnr_slice = get_radiological_slice(cnr, axis=2)
    cnr_vmax = float(np.nanpercentile(cnr[mask_bool], 95)) if np.any(mask_bool) else 50.0
    im = ax.imshow(cnr_slice, cmap="hot", vmin=0, vmax=cnr_vmax, aspect="equal")
    ax.set_title("CNR Map", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CNR")

    if output_path is None:
        output_path = str(Path.cwd() / "dsc_preprocessing_qc.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats: dict[str, Any] = {
        "peak_drop_pct": round(peak_drop_pct, 1),
        "mean_tsnr": round(mean_tsnr, 1),
        "mean_cnr": round(mean_cnr, 1),
        "fraction_cnr_gt_4": round(fraction_cnr_gt_4, 3),
        "tr_ms": round(volume_tr_s * 1000, 2),
        "te_ms": round(te_s * 1000, 2) if te_s is not None else None,
        "n_volumes": n_volumes,
        "n_brain_voxels": int(mask_bool.sum()),
        "voxel_size_mm": (
            [float(v) for v in voxel_size_mm] if voxel_size_mm is not None else None
        ),
    }
    return str(out), stats


def generate_dsc_parameter_maps_qc_plot(
    param_maps: dict[str, npt.NDArray[Any]],
    brain_mask: npt.NDArray[Any],
    subject_id: str = "",
    session_id: str = "",
    output_path: str | None = None,
) -> str:
    """Generate a multi-row parameter map QC plot for DSC.

    Each row shows one parameter across 3 evenly-spaced axial slices
    (25%, 50%, 75% of brain extent). Per-row colorbar with unit labels.

    Args:
        param_maps: Dict mapping parameter name to 3D array.
            Expected keys: "rCBF", "rCBV", "MTT", "Tmax", "PSR", "TTPᶜ".
        brain_mask: 3D boolean brain mask.
        subject_id: Subject identifier for title.
        session_id: Session identifier for title.
        output_path: Path to save PNG. If None, uses CWD.

    Returns:
        Path to the saved QC plot PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from pmriprep.utils.qc_utils import get_radiological_slice

    mask_bool = brain_mask > 0.5

    # Parameter display config: (colormap, vmin, vmax_mode, vmax_fixed, unit_label)
    param_config: dict[str, tuple[str, float, str, float | None, str]] = {
        "rCBF": ("hot", 0.0, "pct98", None, "s\u207b\u00b9"),
        "rCBV": ("hot", 0.0, "pct98", None, "dimensionless"),
        "MTT": ("viridis", 0.0, "fixed", 30.0, "seconds"),
        "Tmax": ("plasma", 0.0, "fixed", 30.0, "seconds"),
        "PSR": ("coolwarm", 0.0, "fixed", 200.0, "%"),
        "TTP\u1d9c": ("viridis", 0.0, "fixed", 20.0, "seconds"),
    }

    # Filter to params that exist in the input
    params_to_plot = [(name, param_maps[name]) for name in param_config if name in param_maps]
    n_params = len(params_to_plot)

    if n_params == 0:
        logger.warning("No parameter maps provided for DSC QC plot")
        return ""

    # Determine slice positions (25%, 50%, 75% of brain z-extent)
    z_coords = np.where(mask_bool)[2]
    if len(z_coords) == 0:
        z_min, z_max = 0, brain_mask.shape[2] - 1
    else:
        z_min, z_max = int(z_coords.min()), int(z_coords.max())
    z_positions = np.linspace(z_min, z_max, 5, dtype=int)[1:4]

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10, 2.8 * n_params))
    gs = GridSpec(
        n_params, 8, figure=fig,
        width_ratios=[0.4, 1, 1, 1, 1, 1, 1, 0.06],
        wspace=0.05, hspace=0.15,
    )

    from pmriprep.utils.qc_utils import qc_figure_title
    fig.suptitle(
        qc_figure_title(subject_id, session_id, "DSC Perfusion Maps"),
        fontsize=13, fontweight="bold", y=0.99,
    )

    for row_idx, (name, data) in enumerate(params_to_plot):
        cmap, vmin, vmax_mode, vmax_fixed, unit_label = param_config[name]

        # Determine vmax
        data_masked = data[mask_bool & np.isfinite(data)]
        if vmax_mode.startswith("pct") and len(data_masked) > 0:
            pct = int(vmax_mode[3:])
            vmax = float(np.percentile(data_masked, pct))
            vmax = max(vmax, vmin + 1e-3)
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

        # Row label axis (col 0)
        label_ax = fig.add_subplot(gs[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.5, 0.5, f"{name}\n(\u03bc={mu:.4f}, \u03c3={sigma:.4f})",
            transform=label_ax.transAxes,
            fontsize=9, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

        # Image axes (3 pairs of merged columns)
        im = None
        slice_spans = [(1, 3), (3, 5), (5, 7)]
        for col_idx, (z_idx, (c_start, c_end)) in enumerate(
            zip(z_positions, slice_spans)
        ):
            ax = fig.add_subplot(gs[row_idx, c_start:c_end])
            slc = get_radiological_slice(display, axis=2, slice_idx=int(z_idx))
            if name == "PSR":
                norm = TwoSlopeNorm(vcenter=100.0, vmin=vmin, vmax=vmax)
                im = ax.imshow(slc, cmap=cmap, norm=norm, aspect="equal")
            else:
                im = ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.axis("off")

            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(f"z = {z_idx}", fontsize=9)

            # R/L labels on first and last image axes
            if col_idx == 0:
                ax.text(
                    0.02, 0.5, "R", transform=ax.transAxes,
                    fontsize=9, color="white", fontweight="bold", va="center",
                )
            if col_idx == 2:
                ax.text(
                    0.98, 0.5, "L", transform=ax.transAxes,
                    fontsize=9, color="white", fontweight="bold",
                    va="center", ha="right",
                )

        # Colorbar axis (col 7)
        assert im is not None  # always set by z_positions loop
        cax = fig.add_subplot(gs[row_idx, 7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"{name} ({unit_label})", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    if output_path is None:
        output_path = str(Path.cwd() / "dsc_parameter_maps_qc.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return str(out)


def generate_dsc_fit_qc_plot(
    tissue_concentration: npt.NDArray[Any],
    model_prediction: npt.NDArray[Any] | None,
    aif: npt.NDArray[Any],
    time_points: npt.NDArray[Any],
    cbf_map: npt.NDArray[Any],
    brain_mask: npt.NDArray[Any],
    qc_voxel: tuple[int, int, int],
    qc_voxel_cbf: float,
    r_squared: float = float("nan"),
    method: str = "",
    subject_id: str = "",
    session_id: str = "",
    output_path: str | None = None,
    raw_signal: npt.NDArray[Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate a 2x2 fit QC plot for DSC deconvolution methods.

    For parametric methods (vascular, CPI), shows the per-voxel model fit.
    For SVD methods (sSVD, cSVD, oSVD, tikhonov), shows the reconstructed
    concentration from AIF convolved with the recovered residue function.

    Panels:
        (0,0) Raw signal intensity at example voxel (shows bolus signal drop)
        (0,1) Model fit curve overlaid on tissue concentration + R^2 badge
        (1,0) AIF concentration curve
        (1,1) CBF map at voxel slice with marker

    Args:
        tissue_concentration: 1D measured concentration at QC voxel.
        model_prediction: 1D model-fitted concentration at QC voxel.
        aif: 1D AIF concentration array.
        time_points: 1D time points array (seconds).
        cbf_map: 3D CBF map (for spatial context panel).
        brain_mask: 3D boolean brain mask.
        qc_voxel: (x, y, z) coordinates of the QC voxel.
        qc_voxel_cbf: CBF value at the QC voxel.
        r_squared: R-squared of the fit at the QC voxel.
        method: Deconvolution method name.
        subject_id: Subject identifier for title.
        session_id: Session identifier for title.
        output_path: Path to save PNG. If None, uses CWD.
        raw_signal: 1D raw signal intensity at QC voxel (optional).

    Returns:
        Tuple of (path to saved QC plot PNG, voxel metrics dict).
    """
    import matplotlib.pyplot as plt

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice

    mask_bool = brain_mask > 0.5
    vx, vy, vz = qc_voxel
    voxel_conc = tissue_concentration
    voxel_cbf = qc_voxel_cbf
    voxel_r2 = r_squared
    time_points_s = time_points
    fitted_curve = model_prediction

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    from pmriprep.utils.qc_utils import qc_figure_title
    fig.suptitle(
        qc_figure_title(subject_id, session_id, f"DSC Fit QC ({method})"),
        fontsize=13, fontweight="bold", y=0.98,
    )

    # --- (0,0) Signal Intensity (or fallback to Concentration) ---
    ax = axes[0, 0]
    if raw_signal is not None and len(raw_signal) > 0:
        ax.plot(time_points_s[:len(raw_signal)], raw_signal, "-o",
                color=COLORBLIND_PALETTE.primary, markersize=3, linewidth=1,
                alpha=0.7)
        ax.set_title(f"Signal Intensity \u2014 Voxel ({vx}, {vy}, {vz})", fontsize=10)
        ax.set_ylabel("Signal (a.u.)")
    else:
        ax.plot(time_points_s, voxel_conc, "o", color=COLORBLIND_PALETTE.primary,
                markersize=4, label="Measured", alpha=0.7)
        ax.set_title(f"Tissue Concentration \u2014 Voxel ({vx}, {vy}, {vz})", fontsize=10)
        ax.set_ylabel("Concentration (a.u.)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    # --- (0,1) Model Fit ---
    ax = axes[0, 1]
    ax.plot(time_points_s, voxel_conc, "o", color=COLORBLIND_PALETTE.primary,
            markersize=4, label="Measured", alpha=0.7)
    if fitted_curve is not None and np.any(fitted_curve != 0):
        ax.plot(time_points_s[:len(fitted_curve)], fitted_curve,
                "-", color=COLORBLIND_PALETTE.error, linewidth=1.5, label="Model fit")
    # R^2 badge
    if not np.isnan(voxel_r2):
        badge_color = COLORBLIND_PALETTE.success if voxel_r2 > 0.7 else COLORBLIND_PALETTE.warning
        ax.text(0.98, 0.97, f"R\u00b2 = {voxel_r2:.3f}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="right", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_color, alpha=0.9))
    ax.set_title("Model Fit", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration (a.u.)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,0) AIF ---
    ax = axes[1, 0]
    aif_time = time_points_s[:len(aif)]
    ax.plot(aif_time, aif, color=COLORBLIND_PALETTE.secondary, linewidth=1.5)
    ax.fill_between(aif_time, aif, alpha=0.2, color=COLORBLIND_PALETTE.secondary)
    ax.set_title("Arterial Input Function", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("AIF Concentration (a.u.)")
    ax.grid(True, alpha=0.3)

    # --- (1,1) CBF Map with voxel marker ---
    ax = axes[1, 1]
    cbf_display = cbf_map.copy().astype(float)
    cbf_display[~mask_bool] = np.nan
    cbf_slice = get_radiological_slice(cbf_display, axis=2, slice_idx=vz)
    cbf_nonzero = cbf_map[mask_bool & (cbf_map > 0)]
    if len(cbf_nonzero) > 0:
        cbf_vmax = float(np.percentile(cbf_nonzero, 95))
    else:
        cbf_vmax = 1.0
    im = ax.imshow(cbf_slice, cmap="hot", vmin=0, vmax=cbf_vmax, aspect="equal")
    ax.set_title(f"CBF Map (z={vz})", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="s\u207b\u00b9")
    # R/L markers
    ax.text(0.02, 0.5, "R", transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold", va="center")
    ax.text(0.98, 0.5, "L", transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold", va="center", ha="right")
    # Voxel marker — transform to radiological slice coordinates
    ny = cbf_map.shape[1]
    marker_row = ny - 1 - vy
    marker_col = cbf_map.shape[0] - 1 - vx
    ax.plot(marker_col, marker_row, "*", color="cyan", markersize=12,
            markeredgecolor="black", markeredgewidth=0.5)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    if output_path is None:
        output_path = str(Path.cwd() / "dsc_fit_qc.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    voxel_metrics: dict[str, Any] = {
        "qc_voxel": f"({vx}, {vy}, {vz})",
        "qc_voxel_cbf": round(voxel_cbf, 4),
        "qc_voxel_r2": round(voxel_r2, 4) if not np.isnan(voxel_r2) else None,
    }
    return str(out), voxel_metrics


def generate_dsc_leakage_qc_plot(
    k2_map: npt.NDArray[Any],
    auc_raw: npt.NDArray[Any],
    auc_corr: npt.NDArray[Any],
    auc_diff: npt.NDArray[Any],
    brain_mask: npt.NDArray[Any],
    conc_raw_4d: npt.NDArray[Any] | None = None,
    conc_corr_4d: npt.NDArray[Any] | None = None,
    psr: npt.NDArray[Any] | None = None,
    tr: float = 1.5,
    subject_id: str = "",
    session_id: str = "",
    output_path: str | None = None,
    method: str = "bsw",
    kep_used: float | None = None,
) -> str:
    """Generate a 2x3 leakage correction QC plot for DSC.

    Layout:
        (0,0) K2 Map — spatial distribution of leakage rate
        (0,1) AUC Raw (pre-K2) map
        (0,2) AUC Corrected (post-K2) map
        (1,0) AUC Diff map — leakage contribution
        (1,1) Example voxel: raw vs corrected concentration
        (1,2) Scatter: K2 vs PSR (if available)

    Only generated when K2 correction was applied.

    Args:
        k2_map: 3D K2 leakage rate map.
        auc_raw: 3D AUC raw (pre-K2) map.
        auc_corr: 3D AUC corrected (post-K2) map.
        auc_diff: 3D AUC diff (leakage contribution) map.
        brain_mask: 3D boolean brain mask.
        conc_raw_4d: 4D pre-K2 concentration (optional, for voxel curves).
        conc_corr_4d: 4D post-K2 concentration (optional, for voxel curves).
        psr: 3D PSR map (optional, for K2 vs PSR scatter).
        tr: Repetition time in seconds.
        subject_id: Subject identifier for title.
        session_id: Session identifier for title.
        output_path: Path to save PNG. If None, uses CWD.
        method: K2 correction method ("bsw" or "bidirectional").
        kep_used: kep value used for bidirectional correction (s⁻¹).

    Returns:
        Path to the saved QC plot PNG.
    """
    import matplotlib.pyplot as plt

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice

    mask_bool = brain_mask > 0.5

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    from pmriprep.utils.qc_utils import qc_figure_title

    # Build title with method info
    method_label = method.upper() if method else "BSW"
    if method == "bidirectional" and kep_used is not None:
        title_suffix = f"DSC Leakage QC (Bidirectional, kep={kep_used:.3f} s⁻¹)"
    else:
        title_suffix = f"DSC Leakage QC ({method_label})"

    fig.suptitle(
        qc_figure_title(subject_id, session_id, title_suffix),
        fontsize=13, fontweight="bold", y=0.98,
    )

    # --- (0,0) K2 Map ---
    ax = axes[0, 0]
    k2_display = k2_map.copy().astype(float)
    k2_display[~mask_bool] = np.nan
    k2_slice = get_radiological_slice(k2_display, axis=2)
    k2_vals = k2_map[mask_bool & (k2_map > 0)]
    k2_vmax = float(np.percentile(k2_vals, 95)) if len(k2_vals) > 0 else 0.1
    im = ax.imshow(k2_slice, cmap="hot", vmin=0, vmax=k2_vmax, aspect="equal")
    ax.set_title("K2 Map (leakage rate)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="s\u207b\u00b9")

    # --- (0,1) AUC Raw ---
    ax = axes[0, 1]
    auc_raw_display = auc_raw.copy().astype(float)
    auc_raw_display[~mask_bool] = np.nan
    auc_raw_slice = get_radiological_slice(auc_raw_display, axis=2)
    auc_raw_vals = auc_raw[mask_bool & (auc_raw > 0)]
    auc_vmax = float(np.percentile(auc_raw_vals, 95)) if len(auc_raw_vals) > 0 else 1.0
    im = ax.imshow(auc_raw_slice, cmap="hot", vmin=0, vmax=auc_vmax, aspect="equal")
    ax.set_title("AUC Raw (pre-K2)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- (0,2) AUC Corrected ---
    ax = axes[0, 2]
    auc_corr_display = auc_corr.copy().astype(float)
    auc_corr_display[~mask_bool] = np.nan
    auc_corr_slice = get_radiological_slice(auc_corr_display, axis=2)
    im = ax.imshow(auc_corr_slice, cmap="hot", vmin=0, vmax=auc_vmax, aspect="equal")
    ax.set_title("AUC Corrected (post-K2)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- (1,0) AUC Diff ---
    ax = axes[1, 0]
    auc_diff_display = auc_diff.copy().astype(float)
    auc_diff_display[~mask_bool] = np.nan
    auc_diff_slice = get_radiological_slice(auc_diff_display, axis=2)
    diff_vals = np.abs(auc_diff[mask_bool])
    diff_vmax = float(np.percentile(diff_vals, 95)) if len(diff_vals) > 0 else 0.5
    im = ax.imshow(auc_diff_slice, cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax,
                   aspect="equal")
    ax.set_title("AUC Diff (leakage contribution)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- (1,1) Example Voxel Curves ---
    ax = axes[1, 1]
    if conc_raw_4d is not None and conc_corr_4d is not None:
        # Find voxel near 90th percentile K2 (representative leakage,
        # avoiding degenerate max-K2 fits from OLS outliers)
        k2_masked = k2_map.copy()
        k2_masked[~mask_bool] = 0
        k2_vals = k2_masked[mask_bool]
        p90 = float(np.percentile(k2_vals, 90)) if len(k2_vals) > 0 else 0.0
        # Find the voxel closest to the 90th percentile value
        diffs = np.abs(k2_masked - p90)
        diffs[~mask_bool] = np.inf
        max_k2_idx = np.unravel_index(np.argmin(diffs), k2_masked.shape)
        ex, ey, ez = int(max_k2_idx[0]), int(max_k2_idx[1]), int(max_k2_idx[2])
        n_t = conc_raw_4d.shape[3]
        time_s = np.arange(n_t) * tr
        ax.plot(time_s, conc_raw_4d[ex, ey, ez, :],
                color=COLORBLIND_PALETTE.error, linewidth=1.2, label="Raw")
        ax.plot(time_s, conc_corr_4d[ex, ey, ez, :],
                color=COLORBLIND_PALETTE.primary, linewidth=1.2, label="Corrected")
        ax.set_title(f"Voxel ({ex},{ey},{ez}) — K2={k2_map[ex,ey,ez]:.4f}", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No 4D data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_title("Example Voxel Curves", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration (a.u.)")
    ax.grid(True, alpha=0.3)

    # --- (1,2) K2 vs PSR Scatter ---
    ax = axes[1, 2]
    if psr is not None:
        k2_vals_scatter = k2_map[mask_bool]
        psr_vals_scatter = psr[mask_bool]
        # Subsample for performance
        n_points = min(5000, len(k2_vals_scatter))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(k2_vals_scatter), size=n_points, replace=False)
        ax.scatter(k2_vals_scatter[indices], psr_vals_scatter[indices],
                   alpha=0.3, s=2, color=COLORBLIND_PALETTE.primary)
        ax.set_xlabel("K2 (s\u207b\u00b9)")
        ax.set_ylabel("PSR (%)")
        ax.set_title("K2 vs PSR", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No PSR data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_title("K2 vs PSR", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    if output_path is None:
        output_path = str(Path.cwd() / "dsc_leakage_qc.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return str(out)
