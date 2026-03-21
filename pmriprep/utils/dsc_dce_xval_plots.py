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

from pmriprep.utils.dce_dsc_validation import (
    HEMATOCRIT_VP_CBV,
    MAX_CBV_FOR_VALIDATION,
    MAX_VP_FOR_VALIDATION,
    MIN_CBV_FOR_VALIDATION,
    MIN_VP_FOR_VALIDATION,
    get_expected_ratio,
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
    constraint_mode: Optional[str] = None,
    hematocrit: float = HEMATOCRIT_VP_CBV,
) -> str:
    """Generate a 3-panel cross-validation QC figure.

    Args:
        vp_data: 3D vp map (fraction, 0-1).
        cbv_data: 3D CBV map (mL/100g).
        mask: 3D brain mask.
        validation_result: Validation result dict from validate_vp_cbv.
        subject_id: BIDS subject ID.
        session_id: BIDS session ID.
        output_path: Output PNG path. If None, uses cwd.
        constraint_mode: CBV constraint mode (hard/soft/skip/None).
        hematocrit: Hematocrit for unit conversion.

    Returns:
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

    expected_ratio = get_expected_ratio(hematocrit)

    # Extract valid voxels
    mask_bool = mask.astype(bool)
    vp_masked = vp_data[mask_bool]
    cbv_masked = cbv_data[mask_bool]

    valid = (
        (vp_masked > MIN_VP_FOR_VALIDATION)
        & (cbv_masked > MIN_CBV_FOR_VALIDATION)
        & (vp_masked < MAX_VP_FOR_VALIDATION)
        & (cbv_masked < MAX_CBV_FOR_VALIDATION)
        & np.isfinite(vp_masked)
        & np.isfinite(cbv_masked)
    )
    vp_valid = vp_masked[valid]
    cbv_valid = cbv_masked[valid]

    # Convert vp to CBV-equivalent units: cbv_from_vp = vp * 100 / ((1-Hct) * rho)
    # But since CBV is already in mL/100g and vp*100 is in mL/100mL,
    # the conversion is: cbv_from_vp = vp * 100 / expected_ratio
    cbv_from_vp = vp_valid * 100.0 / expected_ratio if expected_ratio > 0 else vp_valid * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        qc_figure_title(subject_id, session_id, "DSC-DCE Cross-Validation"),
        fontsize=12,
        fontweight="bold",
    )

    is_circular = validation_result.get("circular", False)

    # ---- Panel 1: Scatter (vp-derived CBV vs DSC CBV) ----
    ax = axes[0]
    if is_circular or len(vp_valid) < 10:
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
        n_pts = min(5000, len(cbv_valid))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cbv_valid), size=n_pts, replace=False)

        ax.scatter(
            cbv_valid[idx], cbv_from_vp[idx],
            s=2, alpha=0.3, color=COLORBLIND_PALETTE.primary,
            rasterized=True,
        )

        # Identity line
        lim_max = max(np.percentile(cbv_valid[idx], 98),
                      np.percentile(cbv_from_vp[idx], 98))
        ax.plot([0, lim_max], [0, lim_max], "--",
                color=COLORBLIND_PALETTE.error, linewidth=1,
                label="identity")

        # Annotate correlation
        metrics = validation_result.get("metrics", {})
        corr = metrics.get("correlation", 0.0)
        ax.text(
            0.05, 0.92, f"r = {corr:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("DSC CBV (mL/100g)", fontsize=9)
        ax.set_ylabel("DCE vp-derived CBV (mL/100g)", fontsize=9)
        ax.set_title("vp vs CBV Scatter", fontsize=10)
        ax.set_xlim(0, lim_max * 1.05)
        ax.set_ylim(0, lim_max * 1.05)

    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Bland-Altman ----
    ax = axes[1]
    if is_circular or len(vp_valid) < 10:
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
        n_pts = min(5000, len(cbv_valid))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cbv_valid), size=n_pts, replace=False)

        ba_mean = (cbv_from_vp[idx] + cbv_valid[idx]) / 2.0
        ba_diff = cbv_from_vp[idx] - cbv_valid[idx]

        ax.scatter(
            ba_mean, ba_diff,
            s=2, alpha=0.3, color=COLORBLIND_PALETTE.primary,
            rasterized=True,
        )

        # Mean bias and limits of agreement
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

        ax.set_xlabel("Mean CBV (mL/100g)", fontsize=9)
        ax.set_ylabel("Difference (DCE - DSC)", fontsize=9)
        ax.set_title("Bland-Altman", fontsize=10)

        ax.text(
            0.95, 0.92, f"bias={mean_bias:.2f}\nLoA=+/-{1.96*sd_diff:.2f}",
            transform=ax.transAxes, fontsize=8, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Spatial ratio map ----
    ax = axes[2]
    if is_circular or len(vp_valid) < 10:
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
            & (cbv_data > MIN_CBV_FOR_VALIDATION)
            & np.isfinite(vp_data)
            & np.isfinite(cbv_data)
        )
        cbv_safe = np.maximum(cbv_data, MIN_CBV_FOR_VALIDATION)
        ratio_3d[valid_3d] = (vp_data[valid_3d] * 100.0) / cbv_safe[valid_3d]

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
        vmin = max(0, expected_ratio - 0.5)
        vmax = expected_ratio + 0.5
        norm = TwoSlopeNorm(vcenter=expected_ratio, vmin=vmin, vmax=vmax)
        im = ax.imshow(
            slice_2d,
            cmap=cmap,
            norm=norm,
            aspect="equal",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("(vp*100)/CBV ratio", fontsize=8)

        ax.set_title(f"Ratio Map (z={best_slice})", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("DSC-DCE cross-validation plot saved: %s", output_path)
    return output_path
