"""AIF QC visualization interface for DCE-MRI.

This module implements Nipype interfaces for generating QC plots of the
detected AIF: a curve time series plot and a 3-pane location overlay.

Adapted from DSC AIF QC with DCE-specific modifications:
- Concentration in mM (not relative signal change)
- DCE-specific QC metrics: final_peak_mm, final_fwhm_seconds, washout_mean, cnr_mean
- Title labels say "DCE" instead of "DSC"
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
    isdefined,
)


def _voxel_to_mni(
    coords_vox: npt.NDArray[Any],
    img: Any,
    t1w_to_native_xfm: str,
    t1w_to_mni_xfm: str,
) -> npt.NDArray[np.floating]:
    """Transform voxel coordinates to MNI mm via FLIRT matrices.

    Chain: voxel → FSL mm → T1w mm → MNI FSL mm → nilearn MNI mm.
    """
    zooms = np.abs(np.array(img.header.get_zooms()[:3]))
    shape = np.array(img.shape[:3])
    fsl_v2m = np.diag(list(zooms) + [1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        fsl_v2m[0, 0] = -zooms[0]
        fsl_v2m[0, 3] = (shape[0] - 1) * zooms[0]

    t1w_to_native = np.loadtxt(t1w_to_native_xfm)
    native_to_t1w = np.linalg.inv(t1w_to_native)
    t1w_to_mni = np.loadtxt(t1w_to_mni_xfm)

    coords_hom = np.hstack([coords_vox, np.ones((len(coords_vox), 1))])
    coords_native_mm = (fsl_v2m @ coords_hom.T).T[:, :3]
    coords_t1w = (
        native_to_t1w[:3, :3] @ coords_native_mm.T + native_to_t1w[:3, 3:4]
    ).T
    coords_mni_fsl = (
        t1w_to_mni[:3, :3] @ coords_t1w.T + t1w_to_mni[:3, 3:4]
    ).T

    return np.column_stack([
        90.0 - coords_mni_fsl[:, 0],
        coords_mni_fsl[:, 1] - 126.0,
        coords_mni_fsl[:, 2] - 72.0,
    ])


class DCEAIFQCPlotInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCEAIFQCPlot interface."""

    aif_curve_file = File(
        exists=True,
        mandatory=True,
        desc="1D AIF curve text file (concentration in mM)",
    )
    aif_mask_file = File(
        exists=True,
        mandatory=False,
        desc="3D AIF voxel mask NIfTI (not provided for population AIF)",
    )
    dce_file = File(
        exists=True,
        mandatory=True,
        desc="4D DCE data (motion-corrected) for background in location overlay",
    )
    brain_mask_file = File(
        exists=True,
        mandatory=True,
        desc="3D brain mask NIfTI",
    )
    qc_metrics = traits.Dict(
        mandatory=True,
        desc="QC metrics dictionary from AIFDetection",
    )
    tr = traits.Float(
        1.5,
        usedefault=True,
        desc="Temporal resolution (volume TR) in seconds",
    )
    subject_id = traits.Str(
        "",
        usedefault=True,
        desc="Subject ID for plot title",
    )
    session_id = traits.Str(
        "",
        usedefault=True,
        desc="Session ID for plot title",
    )
    t1w_to_dce_xfm = File(
        exists=True,
        mandatory=False,
        desc="FLIRT matrix: T1w to DCE space (for glass brain transform)",
    )
    t1w_to_mni_xfm = File(
        exists=True,
        mandatory=False,
        desc="FLIRT matrix: T1w to MNI space (for glass brain transform)",
    )


class DCEAIFQCPlotOutputSpec(TraitedSpec):
    """Output specification for DCEAIFQCPlot interface."""

    out_aif_plot = File(exists=True, desc="AIF curve QC plot PNG")
    out_aif_location = File(desc="AIF location overlay PNG (None for population AIF)")
    out_aif_candidates = File(desc="AIF candidates glass brain PNG")


class DCEAIFQCPlot(BaseInterface):
    """Generate AIF QC visualizations for DCE-MRI.

    Produces two QC plots:
    1. AIF curve time series with peak marker, baseline shading, and quality metrics
    2. 3-pane location overlay (axial, coronal, sagittal) with AIF ROI
    """

    input_spec = DCEAIFQCPlotInputSpec
    output_spec = DCEAIFQCPlotOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        import nibabel as nib  # type: ignore[import-untyped]

        # Load data
        aif_curve = np.loadtxt(self.inputs.aif_curve_file)

        dce_img = nib.load(self.inputs.dce_file)
        dce_data = dce_img.get_fdata()  # type: ignore[attr-defined]

        brain_mask_img = nib.load(self.inputs.brain_mask_file)
        brain_mask = brain_mask_img.get_fdata()  # type: ignore[attr-defined]

        voxel_sizes = tuple(float(z) for z in dce_img.header.get_zooms()[:3])  # type: ignore[attr-defined,union-attr]

        tr = float(self.inputs.tr)
        qc = self.inputs.qc_metrics if isdefined(self.inputs.qc_metrics) else {}
        sub = self.inputs.subject_id or ""
        ses = self.inputs.session_id or ""
        detection_method = qc.get("detection_method", "unknown")

        # Generate plots
        out_dir = Path.cwd()

        # Always generate curve plot
        aif_plot_path = str(out_dir / "aif_curve_qc.png")
        self._plot_aif_curve(
            aif_curve, tr, qc, sub, ses, aif_plot_path,
        )

        # Only generate location plot if voxel-based detection was used
        aif_location_path: str | None = None
        if detection_method not in ("population", "external") and isdefined(self.inputs.aif_mask_file):
            aif_mask_img = nib.load(self.inputs.aif_mask_file)
            aif_mask = aif_mask_img.get_fdata()  # type: ignore[attr-defined]

            aif_location_path = str(out_dir / "aif_location_qc.png")
            self._plot_aif_location(
                dce_data, aif_mask, brain_mask, voxel_sizes,
                qc, sub, ses, aif_location_path,
            )

        # Generate glass brain candidates plot if transforms are available
        aif_candidates_path: str | None = None
        has_xfms = (
            isdefined(self.inputs.t1w_to_dce_xfm)
            and isdefined(self.inputs.t1w_to_mni_xfm)
        )
        if (
            detection_method not in ("population", "external")
            and has_xfms
            and qc.get("candidate_coords")
        ):
            aif_candidates_path = str(out_dir / "aif_candidates_qc.png")
            plot_aif_candidates_glass_brain(
                candidate_coords=qc["candidate_coords"],
                candidate_scores=qc.get("candidate_scores", []),
                native_img=dce_img,
                t1w_to_native_xfm=self.inputs.t1w_to_dce_xfm,
                t1w_to_mni_xfm=self.inputs.t1w_to_mni_xfm,
                subject_id=sub,
                session_id=ses,
                output_path=aif_candidates_path,
                modality="DCE",
            )

        self._results: dict[str, Any] = {
            "out_aif_plot": aif_plot_path,
        }
        if aif_location_path is not None:
            self._results["out_aif_location"] = aif_location_path
        if aif_candidates_path is not None:
            self._results["out_aif_candidates"] = aif_candidates_path

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        if hasattr(self, "_results"):
            outputs.update(self._results)
        return outputs

    # -----------------------------------------------------------------
    # Plot 1: AIF Curve
    # -----------------------------------------------------------------

    @staticmethod
    def _plot_aif_curve(
        aif_curve: npt.NDArray[Any],
        tr: float,
        qc: dict[str, Any],
        subject_id: str,
        session_id: str,
        output_path: str,
    ) -> None:
        """Plot AIF concentration curve with quality annotations.

        Adapted from DSC AIF QC with DCE-specific metrics:
        - Concentration in mM
        - DCE-specific metrics: peak_mm, fwhm_seconds, washout, cnr
        """
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        from pmriprep.utils.qc_standards import COLORBLIND_PALETTE

        n_frames = len(aif_curve)
        time = np.arange(n_frames) * tr

        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            # Plot AIF curve with dot markers
            ax.plot(
                time, aif_curve,
                color=COLORBLIND_PALETTE.primary,
                linewidth=1.5,
                marker="o",
                markersize=3,
                label="AIF",
            )

            # Peak marker
            peak_idx = int(np.argmax(aif_curve))
            ax.scatter(
                time[peak_idx], aif_curve[peak_idx],
                color=COLORBLIND_PALETTE.error,
                s=80, zorder=5, label="Peak",
            )

            # TTP marker (time-to-peak)
            peak_time_s = time[peak_idx]
            ax.axvline(
                x=peak_time_s, color=COLORBLIND_PALETTE.error,
                linestyle="--", linewidth=1.2, label=f"TTP = {peak_time_s:.1f}s",
            )

            # Baseline region shading
            baseline_frames = qc.get("baseline_frames", 5)
            baseline_end_t = (baseline_frames - 1) * tr
            ax.axvspan(0, baseline_end_t, alpha=0.08, color="gray", label=f"Baseline ({baseline_frames} fr)")

            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Concentration (mM)", fontsize=11)  # DCE uses mM
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Title
            sub_label = f"sub-{subject_id}" if subject_id else ""
            ses_label = f" | ses-{session_id}" if session_id else ""
            ax.set_title(f"{sub_label}{ses_label} | DCE AIF Quality", fontsize=12)

            plt.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 2: AIF Location Overlay
    # -----------------------------------------------------------------

    @staticmethod
    def _plot_aif_location(
        dce_data: npt.NDArray[Any],
        aif_mask: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        voxel_sizes: tuple[float, ...],
        qc: dict[str, Any],
        subject_id: str,
        session_id: str,
        output_path: str,
    ) -> None:
        """Plot 3-pane AIF location overlay on temporal mean DCE.

        Shows AIF voxel locations overlaid on the temporal mean DCE image
        in axial, coronal, and sagittal views centered on the best-ranked
        candidate voxel (first entry in candidate_coords), with fallback
        to mask centroid.
        """
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        from pmriprep.utils.qc_utils import get_radiological_slice

        # Compute temporal mean for background
        if dce_data.ndim == 4:
            bg = np.mean(dce_data, axis=-1)
        else:
            bg = dce_data.copy()

        # Prefer best candidate voxel (first entry in ranked candidate_coords)
        cand_coords = qc.get("candidate_coords", [])
        if len(cand_coords) > 0:
            best = cand_coords[0]
            cx, cy, cz = int(best[0]), int(best[1]), int(best[2])
        else:
            # Fallback to mask centroid
            aif_coords = np.argwhere(aif_mask > 0.5)
            if len(aif_coords) == 0:
                brain_coords = np.argwhere(brain_mask > 0.5)
                if len(brain_coords) > 0:
                    centroid = brain_coords.mean(axis=0).astype(int)
                else:
                    centroid = np.array([s // 2 for s in bg.shape[:3]])
            else:
                centroid = aif_coords.mean(axis=0).astype(int)
            cx, cy, cz = int(centroid[0]), int(centroid[1]), int(centroid[2])

        # Voxel aspect ratios
        dx, dy, dz = voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]

        # Set up figure
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        try:
            # Intensity range from background
            valid = bg[brain_mask > 0.5] if np.any(brain_mask > 0.5) else bg[bg > 0]
            if len(valid) > 0:
                vmin = float(np.percentile(valid, 1))
                vmax = float(np.percentile(valid, 99))
            else:
                vmin, vmax = 0.0, 1.0

            # Panel configs: (axis_id, slice_idx, title, aspect, lr_labels)
            # Aspect ratios account for post-rotation axis mapping from
            # get_radiological_slice (rot90 + fliplr).
            panels = [
                (2, cz, f"Axial (z={cz})", dy / dx, ("R", "L")),      # Axial: x vs y
                (1, cy, f"Coronal (y={cy})", dz / dx, ("R", "L")),     # Coronal: x vs z
                (0, cx, f"Sagittal (x={cx})", dz / dy, (None, None)),  # Sagittal: y vs z
            ]

            for ax_plt, (axis_id, slice_idx, title, aspect, lr) in zip(axes, panels):
                # Background slice
                bg_slice = get_radiological_slice(bg, axis=axis_id, slice_idx=slice_idx)
                ax_plt.imshow(bg_slice, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)

                # AIF mask overlay
                mask_slice = get_radiological_slice(
                    aif_mask.astype(float), axis=axis_id, slice_idx=slice_idx,
                )
                if np.any(mask_slice > 0.5):
                    masked = np.ma.masked_where(mask_slice < 0.5, mask_slice)  # type: ignore[no-untyped-call]
                    ax_plt.imshow(masked, cmap="hot", alpha=0.4, aspect=aspect)

                # Crosshair at best candidate / centroid
                # Map 3D coord to 2D slice position after radiological transform.
                # get_radiological_slice applies: rot90(k=1) then fliplr (axial/coronal)
                # or rot90(k=1) only (sagittal). See qc_utils.py for details.
                h, w = bg_slice.shape
                if axis_id == 2:  # Axial
                    cross_x = w - 1 - cx  # Flipped L-R for radiological
                    cross_y = h - 1 - cy  # Rotated
                elif axis_id == 1:  # Coronal
                    cross_x = w - 1 - cx  # Flipped L-R for radiological
                    cross_y = h - 1 - cz  # Rotated
                else:  # Sagittal
                    cross_x = cy  # Y on horizontal
                    cross_y = h - 1 - cz  # Z on vertical (rotated)

                ax_plt.axhline(y=cross_y, color="cyan", linestyle="--", linewidth=0.8, alpha=0.5)
                ax_plt.axvline(x=cross_x, color="cyan", linestyle="--", linewidth=0.8, alpha=0.5)

                # Candidate markers (DSC style: numbered filled circles)
                if len(cand_coords) > 0:
                    candidate_colors = [
                        "#EE3377", "#FFDD00", "#33BBEE", "#009988",
                        "#EE7733", "#CC3311", "#0077BB", "#BBBBBB",
                    ]
                    for i, coord_3d in enumerate(cand_coords):
                        cand_x = int(coord_3d[0])
                        cand_y = int(coord_3d[1])
                        cand_z = int(coord_3d[2])

                        # Check if candidate is within ±1 slice of this view
                        if axis_id == 2 and abs(cand_z - slice_idx) <= 1:
                            px = w - 1 - cand_x
                            py = h - 1 - cand_y
                        elif axis_id == 1 and abs(cand_y - slice_idx) <= 1:
                            px = w - 1 - cand_x
                            py = h - 1 - cand_z
                        elif axis_id == 0 and abs(cand_x - slice_idx) <= 1:
                            px = cand_y
                            py = h - 1 - cand_z
                        else:
                            continue

                        color = candidate_colors[i % len(candidate_colors)]
                        size = 120 if i == 0 else 80
                        ax_plt.scatter(
                            [px], [py], c=color, s=size, marker="o",
                            edgecolors="black", linewidths=1.5, zorder=10,
                        )
                        if len(cand_coords) > 1:
                            ax_plt.annotate(
                                str(i + 1), (px, py), color="white", fontsize=7,
                                ha="center", va="center", fontweight="bold",
                                zorder=11,
                            )

                # L/R labels (radiological: L on right, R on left)
                if lr[0] is not None:
                    ax_plt.text(
                        0.02, 0.5, lr[0], transform=ax_plt.transAxes,
                        fontsize=12, color="white", fontweight="bold",
                        verticalalignment="center",
                    )
                    ax_plt.text(
                        0.98, 0.5, lr[1], transform=ax_plt.transAxes,
                        fontsize=12, color="white", fontweight="bold",
                        verticalalignment="center", horizontalalignment="right",
                    )

                ax_plt.set_title(title, fontsize=11)
                ax_plt.axis("off")

            # Overall title
            sub_label = f"sub-{subject_id}" if subject_id else ""
            ses_label = f" | ses-{session_id}" if session_id else ""
            fig.suptitle(f"{sub_label}{ses_label} | DCE AIF Location", fontsize=12)

            plt.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)

def plot_aif_candidates_glass_brain(
    candidate_coords: list[list[int]],
    candidate_scores: list[float],
    native_img: Any,
    t1w_to_native_xfm: str,
    t1w_to_mni_xfm: str,
    subject_id: str,
    session_id: str,
    output_path: str,
    modality: str = "DCE",
) -> None:
    """Plot AIF candidates on a glass brain in MNI space.

    Args:
        candidate_coords: Voxel coordinates [[x, y, z], ...].
        candidate_scores: Quality scores per candidate (0-1).
        native_img: NIfTI image in native (DCE/DSC) space.
        t1w_to_native_xfm: FLIRT matrix file (T1w → native space).
        t1w_to_mni_xfm: FLIRT matrix file (T1w → MNI space).
        subject_id: Subject ID for title.
        session_id: Session ID for title.
        output_path: Output PNG path.
        modality: "DCE" or "DSC" for title.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from nilearn.plotting import plot_markers  # type: ignore[import-untyped]

    coords_vox = np.array(candidate_coords)
    n = len(coords_vox)
    if n == 0:
        return

    scores = np.array(candidate_scores) if candidate_scores else np.ones(n)

    coords_mni = _voxel_to_mni(
        coords_vox, native_img, t1w_to_native_xfm, t1w_to_mni_xfm,
    )

    fig = plt.figure(figsize=(14, 5), facecolor="black")
    gs = GridSpec(
        1, 4, figure=fig,
        width_ratios=[1.0, 0.9, 1.1, 0.04], wspace=0.05,
    )

    for i, mode in enumerate(("z", "y", "x")):
        ax = fig.add_subplot(gs[i])
        plot_markers(
            node_values=scores,
            node_coords=coords_mni,
            node_cmap="plasma",
            node_size=140,
            display_mode=mode,
            axes=ax,
            colorbar=False,
            radiological=True,
            black_bg=True,
            node_vmin=0.0,
            node_vmax=1.0,
        )

    cb_height = 0.6
    cb_bottom = (1.0 - cb_height) / 2.0
    cb_ax = fig.add_axes((0.92, cb_bottom, 0.012, cb_height))
    # matplotlib.colormaps[...] is the non-deprecated accessor (>=3.7).
    # cm.get_cmap was scheduled for removal in matplotlib 3.11.
    import matplotlib as mpl
    sm = cm.ScalarMappable(
        cmap=mpl.colormaps["plasma"],
        norm=mcolors.Normalize(0, 1),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label("Quality Score", color="white", fontsize=12)
    cb.ax.yaxis.set_tick_params(color="white")
    if cb.outline is not None:
        cb.outline.set_edgecolor("white")  # type: ignore[operator]
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=10)

    sub_label = f"sub-{subject_id}" if subject_id else ""
    ses_label = f" | ses-{session_id}" if session_id else ""
    fig.suptitle(
        f"{sub_label}{ses_label} | {modality} AIF Sources (n={n})",
        fontsize=12, color="white", y=0.95,
    )

    try:
        fig.savefig(
            output_path, dpi=150, bbox_inches="tight", facecolor="black",
        )
    finally:
        plt.close(fig)
