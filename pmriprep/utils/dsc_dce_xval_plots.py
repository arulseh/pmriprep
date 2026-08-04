"""DSC-DCE vascular-concordance visualizations."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_dsc_validation import (
    CONCORDANCE_SMOOTHING_SIGMA_MM,
    build_vp_cbv_validation_mask,
    support_aware_smooth,
)


def generate_xval_plot(
    vp_data: npt.NDArray[Any],
    cbv_data: npt.NDArray[Any],
    brain_mask: npt.NDArray[Any],
    validation_result: dict[str, Any],
    *,
    domain_mask: Optional[npt.NDArray[Any]] = None,
    domain_label: str = "Whole brain",
    metrics_key: str = "metrics",
    voxel_sizes_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    subject_id: str = "",
    session_id: str = "",
    output_path: Optional[str] = None,
    model_selection: Optional[npt.NDArray[Any]] = None,
) -> str:
    """Generate one three-panel vascular-concordance QC figure.

    The whole-brain and optional lesion-region domains are rendered as
    separate figures. The spatial panel retains the brain and region outlines
    so that a restricted-domain difference map remains anatomically legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.colors import TwoSlopeNorm  # noqa: E402

    from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
    from pmriprep.utils.qc_utils import get_radiological_slice, qc_figure_title

    if output_path is None:
        output_path = str(Path.cwd() / "dsc_dce_xval.png")

    normalization = validation_result.get("normalization", {})
    vp_nawm = float(normalization.get("dce_vp_nawm_median", 0.0))
    cbv_nawm = float(normalization.get("dsc_ncbv_nawm_median", 0.0))
    if min(vp_nawm, cbv_nawm) <= 0:
        raise ValueError("DSC-DCE plot requires positive NAWM normalization medians")

    vp = np.asarray(vp_data, dtype=np.float64) / vp_nawm
    cbv = np.asarray(cbv_data, dtype=np.float64) / cbv_nawm
    sigma_vox = tuple(
        CONCORDANCE_SMOOTHING_SIGMA_MM / max(float(size), 1e-6)
        for size in voxel_sizes_mm
    )
    vp = support_aware_smooth(vp, sigma_vox)  # type: ignore[arg-type]
    cbv = support_aware_smooth(cbv, sigma_vox)  # type: ignore[arg-type]

    brain = np.asarray(brain_mask, dtype=bool)
    domain = brain if domain_mask is None else np.asarray(domain_mask, dtype=bool)
    if domain.shape != brain.shape:
        raise ValueError("Concordance domain and brain-mask geometry must match")
    valid = build_vp_cbv_validation_mask(vp, cbv, domain)
    vp_values = vp[valid]
    cbv_values = cbv[valid]
    model_values = model_selection[valid] if model_selection is not None else None
    is_prior_informed = bool(validation_result.get("prior_informed", False))
    prior_sd = validation_result.get("prior_relative_sd")
    prior_relative_sd = float(prior_sd) if isinstance(prior_sd, (int, float)) else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    title = "DSC-informed vp Prior Diagnostic" if is_prior_informed else "DSC-DCE Vascular Concordance"
    fig.suptitle(
        qc_figure_title(subject_id, session_id, f"{title} — {domain_label}"),
        fontsize=12,
        fontweight="bold",
    )
    enough = vp_values.size >= 10
    rng = np.random.default_rng(42)
    sample = (
        rng.choice(vp_values.size, size=min(5000, vp_values.size), replace=False)
        if enough
        else np.array([], dtype=int)
    )

    ax = axes[0]
    if enough:
        if model_values is None:
            ax.scatter(
                cbv_values[sample], vp_values[sample], s=2, alpha=0.3,
                color=COLORBLIND_PALETTE.primary, rasterized=True,
            )
        else:
            from pmriprep.utils.qc_standards import EXTENDED_PALETTE

            names = {1: "Patlak", 2: "Tofts", 3: "Ext. Tofts", 4: "2CXM"}
            colors = {1: EXTENDED_PALETTE[0], 2: EXTENDED_PALETTE[2], 3: EXTENDED_PALETTE[4], 4: EXTENDED_PALETTE[5]}
            selected_models = model_values[sample]
            for code, name in names.items():
                chosen = selected_models == code
                if np.any(chosen):
                    ax.scatter(
                        cbv_values[sample][chosen], vp_values[sample][chosen],
                        s=2, alpha=0.3, color=colors[code], label=name,
                        rasterized=True,
                    )
        limit = max(
            float(np.percentile(cbv_values[sample], 98)),
            float(np.percentile(vp_values[sample], 98)),
            1e-6,
        )
        ax.plot(
            [0, limit], [0, limit], "--", color=COLORBLIND_PALETTE.error,
            linewidth=1, label="Identity",
        )
        if is_prior_informed and prior_relative_sd is not None:
            x_line = np.linspace(0, limit, 100)
            ax.fill_between(
                x_line, x_line * (1 - prior_relative_sd), x_line * (1 + prior_relative_sd),
                color=COLORBLIND_PALETTE.warning, alpha=0.18,
                label=f"Prior mean ± 1 SD ({prior_relative_sd:.0%})",
            )
        ax.legend(fontsize=7, markerscale=3, loc="upper right", framealpha=0.9)
        metrics = validation_result.get(metrics_key, {})
        ax.text(
            0.96, 0.08,
            "Prior-informed" if is_prior_informed else (
                f"ρ = {float(metrics.get('correlation', 0)):.3f}"
            ),
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.set_xlim(0, limit * 1.05)
        ax.set_ylim(0, limit * 1.05)
    else:
        ax.text(0.5, 0.5, "Insufficient valid voxels", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("DSC relative CBV (NAWM=1)", fontsize=9)
    ax.set_ylabel("DCE relative vp (NAWM=1)", fontsize=9)
    ax.set_title("vp vs CBV Scatter", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if enough and is_prior_informed:
        departure = vp_values / cbv_values - 1.0
        departure = departure[np.isfinite(departure)]
        ax.hist(departure, bins=60, color=COLORBLIND_PALETTE.primary, alpha=0.8)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        if prior_relative_sd is not None:
            ax.axvline(-prior_relative_sd, color=COLORBLIND_PALETTE.warning, linewidth=1)
            ax.axvline(prior_relative_sd, color=COLORBLIND_PALETTE.warning, linewidth=1)
        ax.set_xlabel("(vp − DSC-informed vp) / DSC-informed vp", fontsize=9)
        ax.set_ylabel("Voxels", fontsize=9)
        ax.set_title("Fractional Prior Departure", fontsize=10)
    elif enough:
        means = (vp_values[sample] + cbv_values[sample]) / 2
        differences = vp_values[sample] - cbv_values[sample]
        ax.scatter(means, differences, s=2, alpha=0.3, color=COLORBLIND_PALETTE.primary, rasterized=True)
        bias = float(np.mean(differences))
        sd = float(np.std(differences))
        ax.axhline(
            bias, color=COLORBLIND_PALETTE.error, linewidth=1,
            label="Mean bias",
        )
        ax.axhline(
            bias + 1.96 * sd, color=COLORBLIND_PALETTE.warning,
            linewidth=0.8, linestyle="--", label="95% limits of agreement",
        )
        ax.axhline(
            bias - 1.96 * sd, color=COLORBLIND_PALETTE.warning,
            linewidth=0.8, linestyle="--",
        )
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":", label="Zero")
        ax.set_xlim(0, max(float(np.percentile(means, 98)), 1e-6))
        diff_limit = max(float(np.percentile(np.abs(differences), 98)), 1e-6)
        ax.set_ylim(-diff_limit, diff_limit)
        ax.set_xlabel("Mean relative vascular volume", fontsize=9)
        ax.set_ylabel("Difference (DCE − DSC)", fontsize=9)
        ax.set_title("Bland-Altman", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "Insufficient valid voxels", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Bland-Altman", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if enough:
        difference = np.full(vp.shape, np.nan)
        difference[valid] = vp[valid] - cbv[valid]
        best_slice = int(np.argmax(np.sum(valid, axis=(0, 1))))
        display = get_radiological_slice(difference, axis=2, slice_idx=best_slice)
        brain_slice = get_radiological_slice(
            brain.astype(float), axis=2, slice_idx=best_slice
        )
        domain_slice = get_radiological_slice(
            domain.astype(float), axis=2, slice_idx=best_slice
        )
        finite = np.abs(difference[valid])
        vmax = max(float(np.percentile(finite, 95)), 0.1)
        ax.set_facecolor("black")
        image = ax.imshow(display, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), interpolation="nearest")
        ax.contour(brain_slice, levels=[0.5], colors=["0.65"], linewidths=0.8)
        if domain_mask is not None:
            ax.contour(domain_slice, levels=[0.5], colors=["white"], linewidths=1.0)
        fig.colorbar(image, ax=ax, shrink=0.75, label="DCE relative vp − DSC relative CBV")
        ax.set_title(f"Spatial Difference — {domain_label}", fontsize=10)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Insufficient valid voxels", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
