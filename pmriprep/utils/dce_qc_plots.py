"""QC visualization for VFA T1 mapping.

Generates a 3x3 diagnostic plot for VFA T1 mapping quality assessment:

Row 1: VFA images and signal ratio
Row 2: T1 map, M0 map, synthetic T1w contrast
Row 3: T1 histogram, M0 histogram, T1 vs M0 scatter

Uses the colorblind-safe palette from qc_standards.py.

Ported from perf-plus dce_vfa_t1_mapping.py with adaptations for
NiPreps-style visualization conventions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("pmriprep.utils.dce_qc_plots")


def generate_vfa_t1_qc_plot(
    vfa_data: List[npt.NDArray[np.floating]],
    flip_angles: List[float],
    t1_map: npt.NDArray[np.floating],
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    field_strength: float = 3.0,
    output_path: Optional[str] = None,
    correction_info: Optional[Dict[str, Any]] = None,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Generate a 3x3 QC plot for VFA T1 mapping.

    Parameters
    ----------
    vfa_data : list of ndarray
        List of 3D VFA signal arrays (one per flip angle).
    flip_angles : list of float
        Flip angles in degrees corresponding to vfa_data.
    t1_map : ndarray
        T1 map in milliseconds.
    brain_mask : ndarray, optional
        Brain mask for windowing and histogram analysis.
    field_strength : float
        Magnetic field strength in Tesla (default: 3.0).
    output_path : str, optional
        Output path for the PNG file. If None, uses cwd.
    correction_info : dict, optional
        T1 correction info dictionary for annotation.

    Returns
    -------
    str
        Path to the saved QC plot PNG file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice

    if output_path is None:
        output_path = str(Path.cwd() / "vfa_t1_qc.png")

    n_fa = len(vfa_data)

    # Select middle axial slice
    mid_slice = t1_map.shape[2] // 2

    # Get T1 reference ranges (needed for windowing and histograms)
    from pmriprep.utils.dce_t1_validation import T1_TISSUE_REFERENCE, get_field_strength_key

    field_key = get_field_strength_key(field_strength)
    t1_ref = T1_TISSUE_REFERENCE[field_key]

    # Determine display window for T1 map (exclude CSF)
    if brain_mask is not None:
        valid_t1 = t1_map[brain_mask > 0.5]
    else:
        valid_t1 = t1_map[~np.isnan(t1_map)]
    valid_t1 = valid_t1[(valid_t1 > t1_ref["reject_min"]) & (valid_t1 < 2500)]

    if len(valid_t1) > 0:
        t1_vmin = float(np.percentile(valid_t1, 1))
        t1_vmax = float(np.percentile(valid_t1, 99))
    else:
        t1_vmin, t1_vmax = 0.0, 3000.0

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    from pmriprep.utils.qc_utils import qc_figure_title
    fig.suptitle(
        qc_figure_title(subject_id, session_id, "VFA T1 Mapping QC"),
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ---- Row 1: VFA images and signal ratio ----
    # Show first two VFA images and their ratio
    for i in range(min(n_fa, 2)):
        ax = axes[0, i]
        sl = get_radiological_slice(vfa_data[i], axis=2, slice_idx=mid_slice)
        vmax = float(np.percentile(sl[sl > 0], 99)) if np.any(sl > 0) else 1.0
        ax.imshow(sl, cmap="gray", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(f"VFA {flip_angles[i]:.0f}\u00b0", fontsize=11)
        ax.axis("off")

    # Signal ratio
    ax = axes[0, 2]
    if n_fa >= 2:
        eps = 1e-6
        ratio = vfa_data[1] / np.maximum(vfa_data[0], eps)
        ratio_sl = get_radiological_slice(ratio, axis=2, slice_idx=mid_slice)
        # Window ratio around expected range
        ratio_valid = ratio_sl[(ratio_sl > 0.1) & (ratio_sl < 10)]
        if len(ratio_valid) > 0:
            r_vmin = float(np.percentile(ratio_valid, 5))
            r_vmax = float(np.percentile(ratio_valid, 95))
        else:
            r_vmin, r_vmax = 0.5, 3.0
        im = ax.imshow(ratio_sl, cmap="coolwarm", vmin=r_vmin, vmax=r_vmax, origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(
            f"Signal Ratio (S({flip_angles[1]:.0f}\u00b0)/S({flip_angles[0]:.0f}\u00b0))",
            fontsize=10,
        )
    else:
        ax.text(0.5, 0.5, "Need 2+ flip angles\nfor ratio", ha="center", va="center",
                transform=ax.transAxes)
    ax.axis("off")

    # ---- Row 2: T1 map, M0 estimate, synthetic T1w ----
    # T1 map
    ax = axes[1, 0]
    t1_sl = get_radiological_slice(t1_map, axis=2, slice_idx=mid_slice)
    im = ax.imshow(t1_sl, cmap="plasma", vmin=t1_vmin, vmax=t1_vmax, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ms")
    ax.set_title("T1 Map", fontsize=11)
    ax.axis("off")

    # M0 estimate (from first VFA image / sin(alpha))
    ax = axes[1, 1]
    if n_fa >= 1:
        fa_rad = np.radians(flip_angles[0])
        sin_fa = np.sin(fa_rad)
        m0_est = vfa_data[0] / max(sin_fa, 1e-6)
        m0_sl = get_radiological_slice(m0_est, axis=2, slice_idx=mid_slice)
        m0_valid = m0_sl[m0_sl > 0]
        if len(m0_valid) > 0:
            m0_vmax = float(np.percentile(m0_valid, 99))
        else:
            m0_vmax = 1.0
        ax.imshow(m0_sl, cmap="gray", vmin=0, vmax=m0_vmax, origin="upper")
        ax.set_title("M0 Estimate (S/sin(\u03b1))", fontsize=11)
    ax.axis("off")

    # Synthetic T1w contrast
    ax = axes[1, 2]
    if n_fa >= 1:
        tr_assumed = 0.010  # 10ms TR typical for VFA
        t1_s = np.clip(t1_map, 100, 5000) / 1000.0
        synth = m0_est * (1.0 - np.exp(-tr_assumed / t1_s))
        synth_sl = get_radiological_slice(synth, axis=2, slice_idx=mid_slice)
        synth_valid = synth_sl[synth_sl > 0]
        if len(synth_valid) > 0:
            s_vmax = float(np.percentile(synth_valid, 99))
        else:
            s_vmax = 1.0
        ax.imshow(synth_sl, cmap="gray", vmin=0, vmax=s_vmax, origin="upper")
        ax.set_title("Synthetic T1w", fontsize=11)
    ax.axis("off")

    # ---- Row 3: Histograms and scatter ----
    # T1 histogram
    ax = axes[2, 0]
    if brain_mask is not None:
        hist_t1 = t1_map[(brain_mask > 0.5) & (t1_map > t1_ref["reject_min"]) & (t1_map < 3000)]
    else:
        hist_t1 = t1_map[(t1_map > t1_ref["reject_min"]) & (t1_map < 3000)]

    if len(hist_t1) > 0:
        # Percentile trimming for display (matches spatial T1 map windowing)
        p1, p99 = np.percentile(hist_t1, [1, 99])
        t1_display = hist_t1[(hist_t1 >= p1) & (hist_t1 <= p99)]
        if len(t1_display) > 0:
            n, _, _ = ax.hist(t1_display, bins=100, color=COLORBLIND_PALETTE.primary,
                              alpha=0.7, edgecolor="none", density=True)
            # Cap y-axis so outlier edge-voxel bars don't compress the main peak
            nonzero = n[n > 1e-10]
            if len(nonzero) > 5:
                ax.set_ylim(0, float(np.percentile(nonzero, 95)) * 1.5)
        # Reference ranges and median use full hist_t1, not trimmed
        wm_range = t1_ref["wm"]
        gm_range = t1_ref["gm"]
        ax.axvspan(wm_range[0], wm_range[1], alpha=0.15, color=COLORBLIND_PALETTE.success,
                   label=f"WM ({wm_range[0]}-{wm_range[1]}ms)")
        ax.axvspan(gm_range[0], gm_range[1], alpha=0.15, color=COLORBLIND_PALETTE.warning,
                   label=f"GM ({gm_range[0]}-{gm_range[1]}ms)")
        median_val = float(np.median(hist_t1))
        ax.axvline(median_val, color=COLORBLIND_PALETTE.error, linestyle="--",
                   label=f"Median: {median_val:.0f}ms")
        # Ensure reference bands and median are visible
        display_min = min(p1, wm_range[0], median_val) - 50
        display_max = max(p99, gm_range[1], median_val) + 50
        ax.set_xlim(display_min, display_max)
        ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("T1 (ms)")
    ax.set_ylabel("Density")
    ax.set_title("T1 Distribution (Brain)", fontsize=11)

    # M0 histogram
    ax = axes[2, 1]
    if n_fa >= 1:
        if brain_mask is not None:
            valid_voxels = (brain_mask > 0.5) & np.isfinite(t1_map) & (t1_map > t1_ref["reject_min"])
            hist_m0 = m0_est[valid_voxels & (m0_est > 0)]
        else:
            hist_m0 = m0_est[m0_est > 0]
        # IQR-based outlier rejection for display
        if len(hist_m0) > 0:
            q1, q3 = np.percentile(hist_m0, [25, 75])
            iqr = q3 - q1
            m0_display = hist_m0[(hist_m0 >= q1 - 1.5 * iqr) & (hist_m0 <= q3 + 1.5 * iqr)]
            if len(m0_display) > 0:
                n_m0, _, _ = ax.hist(m0_display, bins=80, color=COLORBLIND_PALETTE.secondary,
                                     alpha=0.7, edgecolor="none", density=True)
                nonzero_m0 = n_m0[n_m0 > 1e-10]
                if len(nonzero_m0) > 5:
                    ax.set_ylim(0, float(np.percentile(nonzero_m0, 95)) * 1.5)
    ax.set_xlabel("M0 (a.u.)")
    ax.set_ylabel("Density")
    ax.set_title("M0 Distribution (Brain)", fontsize=11)

    # T1 vs M0 scatter
    ax = axes[2, 2]
    if n_fa >= 1 and brain_mask is not None:
        scatter_mask = (brain_mask > 0.5) & np.isfinite(t1_map) & (t1_map > t1_ref["reject_min"]) & (t1_map < 3000) & (m0_est > 0)
        t1_sc = t1_map[scatter_mask]
        m0_sc = m0_est[scatter_mask]
        if len(t1_sc) > 1000:
            # Subsample for performance
            idx = np.random.default_rng(42).choice(len(t1_sc), 1000, replace=False)
            t1_sc = t1_sc[idx]
            m0_sc = m0_sc[idx]
        if len(t1_sc) > 0:
            ax.scatter(t1_sc, m0_sc, s=1, alpha=0.3, c=COLORBLIND_PALETTE.primary, rasterized=True)
    ax.set_xlabel("T1 (ms)")
    ax.set_ylabel("M0 (a.u.)")
    ax.set_title("T1 vs M0", fontsize=11)

    # Add correction annotation if provided
    if correction_info is not None:
        correction_text = (
            f"Correction: {correction_info.get('correction_applied', 'none')} | "
            f"Original median: {correction_info.get('median_t1_original', 0):.0f}ms | "
            f"Corrected: {correction_info.get('median_t1_corrected', 0):.0f}ms"
        )
        fig.text(0.5, 0.01, correction_text, ha="center", fontsize=9,
                 style="italic", color="gray")

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("VFA T1 QC plot saved to %s", output_path)
    return output_path


def generate_b1_qc_plot(
    s1: npt.NDArray[np.floating],
    s2: npt.NDArray[np.floating],
    signal_ratio: npt.NDArray[np.floating],
    b1_map: npt.NDArray[np.floating],
    nominal_flip_angle: float,
    signal_mask: npt.NDArray[np.bool_],
    valid_mask: npt.NDArray[np.bool_],
    mean_b1: float = 1.0,
    std_b1: float = 0.0,
    output_path: Optional[str] = None,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Generate a 6-panel QC plot for AFI B1+ mapping.

    Layout (2 rows x 3 columns):
        Row 1: TR1 image, TR2 image, Signal ratio (S_TR1/S_TR2)
        Row 2: B1+ map, Actual flip angle map, B1+ histogram

    Parameters
    ----------
    s1 : ndarray
        3D TR1 (short TR) signal array.
    s2 : ndarray
        3D TR2 (long TR) signal array.
    signal_ratio : ndarray
        3D S1/S2 ratio array.
    b1_map : ndarray
        3D B1+ correction factor map.
    nominal_flip_angle : float
        Nominal flip angle in degrees.
    signal_mask : ndarray
        Boolean mask of voxels with valid signal.
    valid_mask : ndarray
        Boolean mask of voxels used in B1+ calculation.
    mean_b1 : float
        Mean B1+ value in brain.
    std_b1 : float
        Standard deviation of B1+ in brain.
    output_path : str, optional
        Output path for the PNG file. If None, uses cwd.

    Returns
    -------
    str
        Path to the saved QC plot PNG file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice

    if output_path is None:
        output_path = str(Path.cwd() / "b1_qc.png")

    # Select middle axial slice
    mid_slice = b1_map.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    from pmriprep.utils.qc_utils import qc_figure_title
    fig.suptitle(
        qc_figure_title(subject_id, session_id, "AFI B1+ Mapping QC"),
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ---- Panel 1: TR1 image ----
    ax = axes[0, 0]
    sl = get_radiological_slice(s1, axis=2, slice_idx=mid_slice)
    vmax = float(np.percentile(sl[sl > 0], 99)) if np.any(sl > 0) else 1.0
    ax.imshow(sl, cmap="gray", vmin=0, vmax=vmax, origin="upper")
    ax.set_title("TR1 (short TR)", fontsize=11)
    ax.axis("off")

    # ---- Panel 2: TR2 image ----
    ax = axes[0, 1]
    sl = get_radiological_slice(s2, axis=2, slice_idx=mid_slice)
    vmax = float(np.percentile(sl[sl > 0], 99)) if np.any(sl > 0) else 1.0
    ax.imshow(sl, cmap="gray", vmin=0, vmax=vmax, origin="upper")
    ax.set_title("TR2 (long TR)", fontsize=11)
    ax.axis("off")

    # ---- Panel 3: Signal ratio ----
    ax = axes[0, 2]
    ratio_sl = get_radiological_slice(signal_ratio, axis=2, slice_idx=mid_slice)
    mask_sl = get_radiological_slice(signal_mask.astype(float), axis=2, slice_idx=mid_slice)
    ratio_display = np.where(mask_sl > 0.5, ratio_sl, np.nan)
    im = ax.imshow(
        ratio_display, cmap="coolwarm", vmin=0.2, vmax=0.8, origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Signal Ratio (S\u2081/S\u2082)", fontsize=11)
    ax.axis("off")

    # ---- Panel 4: B1+ map ----
    ax = axes[1, 0]
    b1_sl = get_radiological_slice(b1_map, axis=2, slice_idx=mid_slice)
    valid_sl = get_radiological_slice(valid_mask.astype(float), axis=2, slice_idx=mid_slice)
    b1_display = np.where(valid_sl > 0.5, b1_sl, np.nan)
    im = ax.imshow(
        b1_display, cmap="plasma", vmin=0.5, vmax=1.5, origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("B1+ Map", fontsize=11)
    ax.axis("off")

    # ---- Panel 5: Actual flip angle map ----
    ax = axes[1, 1]
    fa_map = b1_map * nominal_flip_angle
    fa_sl = get_radiological_slice(fa_map, axis=2, slice_idx=mid_slice)
    fa_display = np.where(valid_sl > 0.5, fa_sl, np.nan)
    expected_fa = nominal_flip_angle
    im = ax.imshow(
        fa_display, cmap="jet",
        vmin=expected_fa * 0.5, vmax=expected_fa * 1.5,
        origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="\u00b0")
    ax.set_title(f"Actual FA (nominal={expected_fa:.0f}\u00b0)", fontsize=11)
    ax.axis("off")

    # ---- Panel 6: B1+ histogram ----
    ax = axes[1, 2]
    b1_brain = b1_map[valid_mask]
    b1_brain = b1_brain[(b1_brain > 0.5) & (b1_brain < 2.0)]

    if len(b1_brain) > 0:
        ax.hist(
            b1_brain, bins=80, range=(0.3, 1.8),
            color=COLORBLIND_PALETTE.primary, alpha=0.7,
            edgecolor="none", density=True,
        )
        # Reference zones
        ax.axvspan(0.7, 1.5, alpha=0.10, color=COLORBLIND_PALETTE.success,
                   label="Typical (0.7\u20131.5)")
        ax.axvspan(1.5, 1.7, alpha=0.10, color=COLORBLIND_PALETTE.warning,
                   label="Dielectric (1.5\u20131.7)")
        # Reference lines
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0,
                   label="Nominal (1.0)")
        hist_mean = float(np.mean(b1_brain))
        hist_median = float(np.median(b1_brain))
        hist_std = float(np.std(b1_brain))
        ax.axvline(hist_mean, color=COLORBLIND_PALETTE.error, linestyle="-",
                   linewidth=1.5, label=f"Mean: {hist_mean:.3f}")
        ax.axvline(hist_median, color=COLORBLIND_PALETTE.secondary, linestyle=":",
                   linewidth=1.5, label=f"Median: {hist_median:.3f}")
        ax.legend(fontsize=7, loc="upper left")
    else:
        hist_mean = mean_b1
        hist_std = std_b1
        ax.text(0.5, 0.5, "No valid B1+ voxels\nin brain mask",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)

    ax.set_xlabel("B1+ Correction Factor")
    ax.set_ylabel("Density")
    ax.set_title(
        f"B1+ Distribution (mean={hist_mean:.3f}, std={hist_std:.3f})",
        fontsize=10,
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("B1+ QC plot saved to %s", output_path)
    return output_path


def generate_b1_precomputed_qc_plot(
    b1_raw: npt.NDArray[np.floating],
    b1_map: npt.NDArray[np.floating],
    valid_mask: npt.NDArray[np.bool_],
    nominal_prep_fa: float,
    mean_b1: float = 1.0,
    std_b1: float = 0.0,
    output_path: Optional[str] = None,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Generate a QC plot for pre-computed B1+ mapping (e.g., tfl_b1map).

    Layout (2 rows, GridSpec):
        Row 1: Raw FA map (x0.1 deg), B1+ ratio map, Actual FA map
        Row 2: Valid mask overlay, B1+ histogram (spans 2 cols)

    Parameters
    ----------
    b1_raw : ndarray
        3D raw pixel values from pre-computed B1+ map (0.1 degree units).
    b1_map : ndarray
        3D B1+ correction factor map (ratio, after scaling and smoothing).
    valid_mask : ndarray
        Boolean mask of voxels with valid B1+ data.
    nominal_prep_fa : float
        Nominal preparation/saturation flip angle in degrees.
    mean_b1 : float
        Mean B1+ value in brain.
    std_b1 : float
        Standard deviation of B1+ in brain.
    output_path : str, optional
        Output path for the PNG file. If None, uses cwd.
    subject_id : str
        Subject ID for figure title.
    session_id : str
        Session ID for figure title.

    Returns
    -------
    str
        Path to the saved QC plot PNG file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice

    if output_path is None:
        output_path = str(Path.cwd() / "b1_qc.png")

    mid_slice = b1_map.shape[2] // 2

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    from pmriprep.utils.qc_utils import qc_figure_title
    fig.suptitle(
        qc_figure_title(subject_id, session_id, "tfl_b1map B1+ Mapping QC"),
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Shared slice data
    raw_sl = get_radiological_slice(b1_raw, axis=2, slice_idx=mid_slice)
    mask_sl = get_radiological_slice(valid_mask.astype(float), axis=2, slice_idx=mid_slice)
    b1_sl = get_radiological_slice(b1_map, axis=2, slice_idx=mid_slice)

    # ---- Panel [0,0]: Raw pixel map ----
    ax = fig.add_subplot(gs[0, 0])
    raw_display = np.where(mask_sl > 0.5, raw_sl, np.nan)
    expected_center = nominal_prep_fa * 10.0
    im = ax.imshow(
        raw_display, cmap="viridis",
        vmin=expected_center * 0.5, vmax=expected_center * 1.5,
        origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="0.1\u00b0")
    ax.set_title(f"Raw FA Map (\u00d70.1\u00b0, nom={nominal_prep_fa:.0f}\u00b0)", fontsize=11)
    ax.axis("off")

    # ---- Panel [0,1]: B1+ ratio map ----
    ax = fig.add_subplot(gs[0, 1])
    b1_display = np.where(mask_sl > 0.5, b1_sl, np.nan)
    im = ax.imshow(
        b1_display, cmap="plasma", vmin=0.5, vmax=1.5, origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("B1+ Ratio Map", fontsize=11)
    ax.axis("off")

    # ---- Panel [0,2]: Actual FA map ----
    ax = fig.add_subplot(gs[0, 2])
    fa_map = b1_map * nominal_prep_fa
    fa_sl = get_radiological_slice(fa_map, axis=2, slice_idx=mid_slice)
    fa_display = np.where(mask_sl > 0.5, fa_sl, np.nan)
    im = ax.imshow(
        fa_display, cmap="jet",
        vmin=nominal_prep_fa * 0.5, vmax=nominal_prep_fa * 1.5,
        origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="\u00b0")
    ax.set_title(f"Actual FA (nominal={nominal_prep_fa:.0f}\u00b0)", fontsize=11)
    ax.axis("off")

    # ---- Panel [1,0]: Valid mask overlay ----
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(raw_sl, cmap="gray", vmin=0,
              vmax=float(np.percentile(raw_sl[raw_sl > 0], 99)) if np.any(raw_sl > 0) else 1.0,
              origin="upper")
    mask_overlay = np.ma.masked_where(mask_sl < 0.5, mask_sl)  # type: ignore[attr-defined]
    ax.imshow(mask_overlay, cmap="Reds", alpha=0.3, origin="upper")
    ax.set_title("Valid Mask Overlay", fontsize=11)
    ax.axis("off")

    # ---- Panel [1, 1:3]: B1+ histogram (spans 2 columns) ----
    ax = fig.add_subplot(gs[1, 1:3])
    b1_brain = b1_map[valid_mask]
    b1_brain = b1_brain[(b1_brain > 0.5) & (b1_brain < 2.0)]

    if len(b1_brain) > 0:
        ax.hist(
            b1_brain, bins=80, range=(0.3, 1.8),
            color=COLORBLIND_PALETTE.primary, alpha=0.7,
            edgecolor="none", density=True,
        )
        # Reference zones
        ax.axvspan(0.7, 1.5, alpha=0.10, color=COLORBLIND_PALETTE.success,
                   label="Typical (0.7\u20131.5)")
        ax.axvspan(1.5, 1.7, alpha=0.10, color=COLORBLIND_PALETTE.warning,
                   label="Dielectric (1.5\u20131.7)")
        # Reference lines
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0,
                   label="Nominal (1.0)")
        hist_mean = float(np.mean(b1_brain))
        hist_median = float(np.median(b1_brain))
        hist_std = float(np.std(b1_brain))
        ax.axvline(hist_mean, color=COLORBLIND_PALETTE.error, linestyle="-",
                   linewidth=1.5, label=f"Mean: {hist_mean:.3f}")
        ax.axvline(hist_median, color=COLORBLIND_PALETTE.secondary, linestyle=":",
                   linewidth=1.5, label=f"Median: {hist_median:.3f}")
        ax.legend(fontsize=7, loc="upper left")
    else:
        hist_mean = mean_b1
        hist_std = std_b1
        ax.text(0.5, 0.5, "No valid B1+ voxels",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)

    ax.set_xlabel("B1+ Correction Factor")
    ax.set_ylabel("Density")
    ax.set_title(
        f"B1+ Distribution (mean={hist_mean:.3f}, std={hist_std:.3f})",
        fontsize=10,
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Pre-computed B1+ QC plot saved to %s", output_path)
    return output_path
