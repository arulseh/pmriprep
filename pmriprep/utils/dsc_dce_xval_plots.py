"""DSC-DCE cross-validation visualization.

Generates a 3-panel QC figure for DSC-DCE cross-validation:
  Panel 1: vp vs CBV scatter (converted to common CBV-equivalent units)
  Panel 2: Bland-Altman plot (in CBV-equivalent units)
  Panel 3: Spatial ratio map (axial slice, diverging colormap)

When the cross-validation is circular (vp deterministically derived from
CBV), all three panels display informational text instead of plots.

Uses the colorblind-safe palette from qc_standards.py.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_constants import (
    MAX_NCBV_FOR_VALIDATION,
    MIN_NCBV_FOR_VALIDATION,
    NAWM_VP_REFERENCE,
)
from pmriprep.utils.dce_dsc_validation import (
    MAX_VP_FOR_VALIDATION,
    MIN_VP_FOR_VALIDATION,
)

logger = logging.getLogger(__name__)


def generate_xval_plot(
    vp_data: npt.NDArray[Any],
    cbv_data: npt.NDArray[Any],
    mask: npt.NDArray[Any],
    validation_result: dict[str, Any],
    subject_id: str = "",
    session_id: str = "",
    output_path: Optional[str] = None,
    model_selection: Optional[npt.NDArray[Any]] = None,
) -> str:
    """Generate a 3-panel cross-validation QC figure.

    Parameters
    ----------
    vp_data : npt.NDArray[Any]
        3D vp map (fraction, 0-1).
    cbv_data : npt.NDArray[Any]
        3D dimensionless NAWM-normalized CBV (nCBV) map.
    mask : npt.NDArray[Any]
        3D brain mask.
    validation_result : dict[str, Any]
        Validation result dict from validate_vp_cbv.
    subject_id : str, optional
        BIDS subject ID.
    session_id : str, optional
        BIDS session ID.
    output_path : str, optional
        Output PNG path. If None, uses cwd.
    model_selection : npt.NDArray[Any], optional
        Optional 3D model selection map (int8).
        When provided, scatter points are colored by model
        (0=unfitted, 1=Patlak, 2=Tofts, 3=ExtTofts, 4=2CXM).

    Returns
    -------
    str
        Path to saved PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.colors import TwoSlopeNorm  # noqa: E402

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice, qc_figure_title

    if output_path is None:
        output_path = str(Path.cwd() / "dsc_dce_xval.png")

    expected_ratio = NAWM_VP_REFERENCE
    cbv_min = MIN_NCBV_FOR_VALIDATION
    cbv_max = MAX_NCBV_FOR_VALIDATION
    cbv_axis_label = "DSC nCBV (NAWM-normalized)"
    cbv_from_vp_label = "DCE vp-derived nCBV"
    mean_axis_label = "Mean nCBV"
    ratio_label = "vp/nCBV ratio"
    cbv_unit_factor = 1.0

    # Extract valid voxels — two masks:
    #   valid_plot: all finite nonzero brain voxels (gray background layer)
    #   valid_metrics: physiological bounds subset (colored layer + metrics)
    mask_bool = mask.astype(bool)
    vp_masked = vp_data[mask_bool]
    cbv_masked = cbv_data[mask_bool]

    valid_plot = (
        np.isfinite(vp_masked)
        & np.isfinite(cbv_masked)
        & (vp_masked > 0)
        & (cbv_masked > 0)
    )
    valid_metrics = (
        valid_plot
        & (vp_masked > MIN_VP_FOR_VALIDATION)
        & (cbv_masked > cbv_min)
        & (vp_masked < MAX_VP_FOR_VALIDATION)
        & (cbv_masked < cbv_max)
    )
    vp_valid = vp_masked[valid_metrics]
    cbv_valid = cbv_masked[valid_metrics]

    # Out-of-bounds voxels for gray background
    oob = valid_plot & ~valid_metrics
    vp_oob = vp_masked[oob]
    cbv_oob = cbv_masked[oob]

    # Extract model selection for in-bounds voxels (if provided)
    model_sel_valid: Optional[npt.NDArray[Any]] = None
    if model_selection is not None:
        model_sel_valid = model_selection[mask_bool][valid_metrics]

    # Project vp onto the dimensionless nCBV scale for comparison.
    if expected_ratio > 0:
        cbv_from_vp = vp_valid * cbv_unit_factor / expected_ratio
        cbv_from_vp_oob = vp_oob * cbv_unit_factor / expected_ratio
    else:
        cbv_from_vp = vp_valid * cbv_unit_factor
        cbv_from_vp_oob = vp_oob * cbv_unit_factor

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        qc_figure_title(subject_id, session_id, "DSC-DCE Cross-Validation"),
        fontsize=12,
        fontweight="bold",
    )

    is_circular = validation_result.get("circular", False)
    has_metrics = len(vp_valid) >= 10

    # ---- Panel 1: Scatter (vp-derived CBV vs DSC CBV) ----
    ax = axes[0]
    if is_circular or np.sum(valid_plot) < 10:
        ax.text(
            0.5, 0.5,
            "vp derived from CBV\nscatter not informative"
            if is_circular else "Insufficient valid voxels",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0"),
        )
        ax.set_title("vp vs CBV Scatter", fontsize=10)
    else:
        # Subsample for performance
        rng = np.random.default_rng(42)

        # Gray background: out-of-bounds voxels
        if len(cbv_oob) > 0:
            n_oob = min(5000, len(cbv_oob))
            idx_oob = rng.choice(len(cbv_oob), size=n_oob, replace=False)
            ax.scatter(
                cbv_oob[idx_oob], cbv_from_vp_oob[idx_oob],
                s=1, alpha=0.12, color="0.6", rasterized=True, zorder=1,
            )

        # Colored foreground: in-bounds voxels
        if has_metrics:
            n_pts = min(5000, len(cbv_valid))
            idx = rng.choice(len(cbv_valid), size=n_pts, replace=False)

            if model_sel_valid is not None:
                from pmriprep.utils.qc_standards import EXTENDED_PALETTE

                _MODEL_NAMES = {1: "Patlak", 2: "Tofts", 3: "Ext. Tofts", 4: "2CXM"}
                _MODEL_COLORS = {
                    1: EXTENDED_PALETTE[0],
                    2: EXTENDED_PALETTE[2],
                    3: EXTENDED_PALETTE[4],
                    4: EXTENDED_PALETTE[5],
                }
                ms_sub = model_sel_valid[idx]
                for code, name in _MODEL_NAMES.items():
                    sel = ms_sub == code
                    if np.any(sel):
                        ax.scatter(
                            cbv_valid[idx][sel], cbv_from_vp[idx][sel],
                            s=2, alpha=0.3, color=_MODEL_COLORS[code],
                            label=name, rasterized=True, zorder=2,
                        )
                ax.legend(fontsize=7, markerscale=3, loc="lower right")
            else:
                ax.scatter(
                    cbv_valid[idx], cbv_from_vp[idx],
                    s=2, alpha=0.3, color=COLORBLIND_PALETTE.primary,
                    rasterized=True, zorder=2,
                )

            lim_max = max(np.percentile(cbv_valid[idx], 98),
                          np.percentile(cbv_from_vp[idx], 98))
            ax.plot([0, lim_max], [0, lim_max], "--",
                    color=COLORBLIND_PALETTE.error, linewidth=1,
                    label="identity")

            metrics = validation_result.get("metrics", {})
            rho = metrics.get("correlation", 0.0)
            r_pearson = metrics.get("pearson_correlation", 0.0)
            ax.text(
                0.05, 0.92, f"\u03c1 = {rho:.3f} (r = {r_pearson:.3f})",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        else:
            if len(cbv_oob) > 0:
                lim_max = max(np.percentile(cbv_oob, 98),
                              np.percentile(cbv_from_vp_oob, 98))
            else:
                lim_max = 1.0

        ax.set_xlabel(cbv_axis_label, fontsize=9)
        ax.set_ylabel(cbv_from_vp_label, fontsize=9)
        ax.set_title("vp vs CBV Scatter", fontsize=10)
        ax.set_xlim(0, lim_max * 1.05)
        ax.set_ylim(0, lim_max * 1.05)

    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Bland-Altman ----
    ax = axes[1]
    if is_circular or np.sum(valid_plot) < 10:
        ax.text(
            0.5, 0.5,
            "vp derived from CBV\nBland-Altman not informative"
            if is_circular else "Insufficient valid voxels",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0"),
        )
        ax.set_title("Bland-Altman", fontsize=10)
    else:
        rng = np.random.default_rng(42)

        # Gray background: out-of-bounds BA points
        if len(cbv_oob) > 0:
            n_oob = min(5000, len(cbv_oob))
            idx_oob = rng.choice(len(cbv_oob), size=n_oob, replace=False)
            ba_mean_oob = (cbv_from_vp_oob[idx_oob] + cbv_oob[idx_oob]) / 2.0
            ba_diff_oob = cbv_from_vp_oob[idx_oob] - cbv_oob[idx_oob]
            ax.scatter(
                ba_mean_oob, ba_diff_oob,
                s=1, alpha=0.12, color="0.6", rasterized=True, zorder=1,
            )

        # Colored foreground: in-bounds BA points
        if has_metrics:
            n_pts = min(5000, len(cbv_valid))
            idx = rng.choice(len(cbv_valid), size=n_pts, replace=False)

            ba_mean = (cbv_from_vp[idx] + cbv_valid[idx]) / 2.0
            ba_diff = cbv_from_vp[idx] - cbv_valid[idx]

            if model_sel_valid is not None:
                from pmriprep.utils.qc_standards import EXTENDED_PALETTE

                _MODEL_NAMES = {1: "Patlak", 2: "Tofts", 3: "Ext. Tofts", 4: "2CXM"}
                _MODEL_COLORS = {
                    1: EXTENDED_PALETTE[0],
                    2: EXTENDED_PALETTE[2],
                    3: EXTENDED_PALETTE[4],
                    4: EXTENDED_PALETTE[5],
                }
                ms_sub = model_sel_valid[idx]
                for code, name in _MODEL_NAMES.items():
                    sel = ms_sub == code
                    if np.any(sel):
                        ax.scatter(
                            ba_mean[sel], ba_diff[sel],
                            s=2, alpha=0.3, color=_MODEL_COLORS[code],
                            label=name, rasterized=True, zorder=2,
                        )
                ax.legend(fontsize=7, markerscale=3, loc="lower right")
            else:
                ax.scatter(
                    ba_mean, ba_diff,
                    s=2, alpha=0.3, color=COLORBLIND_PALETTE.primary,
                    rasterized=True, zorder=2,
                )

            mean_bias = float(np.mean(ba_diff))
            sd_diff = float(np.std(ba_diff))
            ax.axhline(mean_bias, color=COLORBLIND_PALETTE.error,
                        linewidth=1, label=f"bias={mean_bias:.2f}")
            ax.axhline(mean_bias + 1.96 * sd_diff,
                        color=COLORBLIND_PALETTE.warning,
                        linewidth=0.8, linestyle="--")
            ax.axhline(mean_bias - 1.96 * sd_diff,
                        color=COLORBLIND_PALETTE.warning,
                        linewidth=0.8, linestyle="--")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

            ax.text(
                0.95, 0.92, f"bias={mean_bias:.2f}\nLoA=+/-{1.96*sd_diff:.2f}",
                transform=ax.transAxes, fontsize=8, ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel(mean_axis_label, fontsize=9)
        ax.set_ylabel("Difference (DCE - DSC)", fontsize=9)
        ax.set_title("Bland-Altman", fontsize=10)

    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Spatial ratio map ----
    ax = axes[2]
    if is_circular or np.sum(valid_plot) < 10:
        ax.text(
            0.5, 0.5,
            "vp derived from CBV\nratio map uniform by construction"
            if is_circular else "Insufficient valid voxels",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0"),
        )
        ax.set_title("Ratio Map", fontsize=10)
        ax.axis("off")
    else:
        ratio_3d = np.full(vp_data.shape, np.nan)
        valid_3d = (
            mask_bool
            & (vp_data > MIN_VP_FOR_VALIDATION)
            & (cbv_data > cbv_min)
            & np.isfinite(vp_data)
            & np.isfinite(cbv_data)
        )
        cbv_safe = np.maximum(cbv_data, cbv_min)
        ratio_3d[valid_3d] = (
            vp_data[valid_3d] * cbv_unit_factor
        ) / cbv_safe[valid_3d]

        # Get axial slice with most valid voxels
        n_valid_per_slice = np.sum(valid_3d, axis=(0, 1))
        if np.any(n_valid_per_slice > 0):
            best_slice = int(np.argmax(n_valid_per_slice))
        else:
            best_slice = ratio_3d.shape[2] // 2  # type: ignore[misc]

        slice_2d = get_radiological_slice(ratio_3d, axis=2, slice_idx=best_slice)

        # Diverging colormap centered at expected ratio
        # Set NaN (background) to black so it's distinct from coolwarm's
        # near-white center (RGB ~0.87), which represents the expected ratio.
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(color="black")
        # Colorbar span is a multiplicative factor of the expected ratio so
        # it works on both scales (0.669 absolute, 0.02 nCBV) without
        # saturating one path or compressing the other.
        half_span = expected_ratio * 0.75
        vmin = max(0.0, expected_ratio - half_span)
        vmax = expected_ratio + half_span
        norm = TwoSlopeNorm(vcenter=expected_ratio, vmin=vmin, vmax=vmax)
        im = ax.imshow(
            slice_2d,
            cmap=cmap,
            norm=norm,
            aspect="equal",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(ratio_label, fontsize=8)

        ax.set_title(f"Ratio Map (z={best_slice})", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("DSC-DCE cross-validation plot saved: %s", output_path)
    return output_path
