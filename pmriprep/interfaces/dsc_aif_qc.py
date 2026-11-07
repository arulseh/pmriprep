"""AIF QC visualization interface for DSC-MRI.

This module implements Nipype interfaces for generating QC plots of the
detected AIF: a curve time series plot and a 3-pane location overlay.
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


class AIFQCPlotInputSpec(BaseInterfaceInputSpec):
    """Input specification for AIFQCPlot interface."""

    aif_curve_file = File(
        exists=True,
        desc="Accepted AIF curve text file, when available",
    )
    aif_mask_file = File(
        exists=True,
        mandatory=False,
        desc="3D AIF voxel mask NIfTI (provided for measured AIFs)",
    )
    dsc_file = File(
        exists=True,
        mandatory=True,
        desc="4D DSC data (motion-corrected) for background in location overlay",
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
        desc="Repetition time in seconds",
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
    gamma_aif_curve_file = File(exists=True, desc="Diagnostic gamma-fit curve")
    rejected_aif_curve_file = File(exists=True, desc="Rejected measured curve for QC")
    rejected_aif_mask_file = File(exists=True, desc="Matching rejected-candidate voxel mask")
    t1w_to_dsc_xfm = File(
        exists=True,
        mandatory=False,
        desc="FLIRT matrix: T1w to DSC space (for glass brain transform)",
    )
    t1w_to_mni_xfm = File(
        exists=True,
        mandatory=False,
        desc="FLIRT matrix: T1w to MNI space (for glass brain transform)",
    )


class AIFQCPlotOutputSpec(TraitedSpec):
    """Output specification for AIFQCPlot interface."""

    out_aif_plot = File(exists=True, desc="AIF curve QC plot PNG")
    out_aif_location = File(desc="AIF location overlay PNG (absent without spatial support)")
    out_aif_candidates = File(desc="AIF candidates glass brain PNG")
    out_aif_metrics = traits.Dict(desc="AIF quality metrics for HTML report")


class AIFQCPlot(BaseInterface):
    """Generate AIF QC visualizations for DSC-MRI.

    Produces two QC plots:
    1. AIF curve time series with TTP/arrival markers and quality metrics
    2. 3-pane location overlay (axial, coronal, sagittal) with AIF ROI
    """

    input_spec = AIFQCPlotInputSpec
    output_spec = AIFQCPlotOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        import nibabel as nib  # type: ignore[import-untyped]

        # Load data
        accepted = isdefined(self.inputs.aif_curve_file) and bool(self.inputs.aif_curve_file)
        rejected = not accepted and isdefined(self.inputs.rejected_aif_curve_file)
        curve_file = self.inputs.aif_curve_file if accepted else self.inputs.rejected_aif_curve_file
        aif_curve = np.loadtxt(curve_file) if accepted or rejected else None

        dsc_img = nib.load(self.inputs.dsc_file)
        dsc_data = dsc_img.get_fdata()  # type: ignore[attr-defined]

        brain_mask_img = nib.load(self.inputs.brain_mask_file)
        brain_mask = brain_mask_img.get_fdata()  # type: ignore[attr-defined]

        voxel_sizes = tuple(float(z) for z in dsc_img.header.get_zooms()[:3])  # type: ignore[attr-defined,union-attr]

        tr = float(self.inputs.tr)
        qc = self.inputs.qc_metrics if isdefined(self.inputs.qc_metrics) else {}
        sub = self.inputs.subject_id or ""
        ses = self.inputs.session_id or ""
        detection_level = qc.get("detection_method", "unknown")

        display_qc = dict(qc)
        if rejected:
            display_qc.update(qc.get("rejected_candidate") or {})
            display_qc["candidate_rejected"] = True
        gamma_curve = (
            np.loadtxt(self.inputs.gamma_aif_curve_file)
            if isdefined(self.inputs.gamma_aif_curve_file) and self.inputs.gamma_aif_curve_file
            else None
        )
        out_dir = Path.cwd()
        aif_plot_path = None
        if aif_curve is not None:
            aif_plot_path = str(out_dir / "aif_curve_qc.png")
            self._plot_aif_curve(
                aif_curve, tr, display_qc, sub, ses, aif_plot_path,
                gamma_aif_curve=gamma_curve, rejected=rejected,
            )
        mask_file = self.inputs.rejected_aif_mask_file if rejected else self.inputs.aif_mask_file
        aif_location_path: str | None = None
        if detection_level != "external" and isdefined(mask_file) and mask_file:
            mask_img: Any = nib.load(mask_file)
            aif_mask = mask_img.get_fdata()
            if np.any(aif_mask > 0):
                aif_location_path = str(out_dir / "aif_location_qc.png")
                self._plot_aif_location(
                    dsc_data, aif_mask, brain_mask, voxel_sizes,
                    display_qc, sub, ses, aif_location_path,
                )

        # Build AIF metrics dict for HTML report. Producer (aif_detect)
        # populates the canonical nested ``aif_metrics`` block on its
        # qc_metrics output; this node augments that block with
        # gamma-fit annotations and rounds for
        # display.
        upstream = qc.get("aif_metrics") or {}
        aif_metrics: dict[str, Any] = {}
        if "peak_conc" in upstream:
            aif_metrics["peak_conc"] = round(float(upstream["peak_conc"]), 2)
        if "ttp" in upstream:
            aif_metrics["ttp"] = round(float(upstream["ttp"]), 1)
        if "fwhm" in upstream:
            aif_metrics["fwhm"] = round(float(upstream["fwhm"]), 1)
        if "baseline_return_ratio_percent" in upstream:
            aif_metrics["baseline_return_ratio_percent"] = round(float(upstream["baseline_return_ratio_percent"]), 1)
        if "signal_recovery_percent" in upstream:
            aif_metrics["signal_recovery_percent"] = round(float(upstream["signal_recovery_percent"]), 1)
        if "peak_height_percent" in upstream:
            aif_metrics["peak_height_percent"] = round(float(upstream["peak_height_percent"]), 1)

        # Generate glass brain candidates plot if transforms are available
        aif_candidates_path: str | None = None
        has_xfms = (
            isdefined(self.inputs.t1w_to_dsc_xfm)
            and isdefined(self.inputs.t1w_to_mni_xfm)
        )
        if (
            detection_level != "external"
            and has_xfms
            and display_qc.get("candidate_coords")
        ):
            from pmriprep.interfaces.dce_aif_qc import (
                plot_aif_candidates_glass_brain,
            )

            aif_candidates_path = str(out_dir / "aif_candidates_qc.png")
            plot_aif_candidates_glass_brain(
                candidate_coords=display_qc["candidate_coords"],
                candidate_scores=display_qc.get("candidate_scores", []),
                native_img=dsc_img,
                t1w_to_native_xfm=self.inputs.t1w_to_dsc_xfm,
                t1w_to_mni_xfm=self.inputs.t1w_to_mni_xfm,
                subject_id=sub,
                session_id=ses,
                output_path=aif_candidates_path,
                modality="DSC",
            )

        self._results: dict[str, Any] = {
            "out_aif_metrics": aif_metrics,
        }
        if aif_plot_path is not None:
            self._results["out_aif_plot"] = aif_plot_path
        if aif_location_path is not None:
            self._results["out_aif_location"] = aif_location_path
        if aif_candidates_path is not None:
            self._results["out_aif_candidates"] = aif_candidates_path

        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        """Return the plots available for this detection outcome."""
        return dict(getattr(self, "_results", {}))

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
        gamma_aif_curve: Optional[npt.NDArray[Any]] = None,
        rejected: bool = False,
    ) -> None:
        """Plot AIF concentration curve with quality annotations.

        The measured curve and optional diagnostic gamma fit retain their
        own labels. Rejected candidates are shown only as QC evidence.
        """
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        from pmriprep.utils.qc_standards import COLORBLIND_PALETTE

        n_frames = len(aif_curve)
        time = np.arange(n_frames) * tr

        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            label = "Rejected candidate" if rejected else (
                "Supplied AIF" if qc.get("detection_method") == "external" else "Measured AIF"
            )
            ax.plot(time, aif_curve, color=COLORBLIND_PALETTE.primary,
                    linewidth=1.5, marker="o", markersize=3, label=label)
            if gamma_aif_curve is not None:
                r2 = qc.get("gamma_fit_r_squared")
                fit_label = f"Gamma fit (QC, R²={r2:.3f})" if r2 is not None else "Gamma fit (QC)"
                ax.plot(time, gamma_aif_curve, color=COLORBLIND_PALETTE.error,
                        linewidth=1.5, linestyle="--", alpha=0.6, label=fit_label)
                start = qc.get("gamma_fit_evaluation_start_frame")
                end = qc.get("gamma_fit_evaluation_end_frame")
                if start is not None and end is not None:
                    ax.text(0, -0.18, f"Gamma R² window: frames {start}–{end - 1}",
                            transform=ax.transAxes, fontsize=8, color="0.4")

            # Peak marker
            peak_idx = int(np.argmax(aif_curve))
            ax.scatter(
                time[peak_idx], aif_curve[peak_idx],
                color=COLORBLIND_PALETTE.error,
                s=80, zorder=5, label="Peak",
            )

            # TTP marker (sourced from canonical aif_metrics block)
            ttp_s = qc.get("aif_metrics", {}).get("ttp")
            if ttp_s is not None and ttp_s > 0:
                ax.axvline(
                    x=ttp_s, color=COLORBLIND_PALETTE.error,
                    linestyle="--", linewidth=1.2, label=f"TTP = {ttp_s:.1f}s",
                )

            # Arrival time marker
            arrival_s = qc.get("arrival_time_seconds", None)
            if arrival_s is not None and arrival_s > 0:
                ax.axvline(
                    x=arrival_s, color=COLORBLIND_PALETTE.success,
                    linestyle="--", linewidth=1.2, label=f"Arrival = {arrival_s:.1f}s",
                )

            # Baseline region shading
            baseline_frames = qc.get("n_baseline_frames", 5)
            baseline_end_t = (baseline_frames - 1) * tr
            ax.axvspan(0, baseline_end_t, alpha=0.08, color="gray", label=f"Baseline ({baseline_frames} fr)")

            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel(r"$\Delta$R$_2^*$ (s$^{-1}$)", fontsize=11)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

            sub_label = f"sub-{subject_id}" if subject_id else ""
            ses_label = f" | ses-{session_id}" if session_id else ""
            ax.set_title(f"{sub_label}{ses_label} | DSC AIF", fontsize=12)

            plt.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 2: AIF Location Overlay
    # -----------------------------------------------------------------

    @staticmethod
    def _plot_aif_location(
        dsc_data: npt.NDArray[Any],
        aif_mask: npt.NDArray[Any],
        brain_mask: npt.NDArray[Any],
        voxel_sizes: tuple[float, ...],
        qc: dict[str, Any],
        subject_id: str,
        session_id: str,
        output_path: str,
    ) -> None:
        """Plot 3-pane AIF location overlay on temporal mean DSC."""
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        from pmriprep.utils.qc_utils import get_radiological_slice

        # Compute temporal mean for background
        if dsc_data.ndim == 4:
            bg = np.mean(dsc_data, axis=-1)
        else:
            bg = dsc_data.copy()

        # Prefer best candidate voxel coord (guaranteed to be an actual mask voxel)
        aif_voxel = qc.get("aif_voxel_coord", None)
        if aif_voxel is not None and len(aif_voxel) == 3:
            cx, cy, cz = int(aif_voxel[0]), int(aif_voxel[1]), int(aif_voxel[2])
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
                    ax_plt.imshow(masked, cmap="hot", vmin=0, vmax=1, alpha=0.4, aspect=aspect)

                # Crosshair at centroid
                # Map 3D centroid to 2D slice position after radiological transform.
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

                # Candidate markers (numbered circles)
                all_coords = qc.get("candidate_coords", [])
                if len(all_coords) > 0:
                    candidate_colors = [
                        "#EE3377", "#FFDD00", "#33BBEE", "#009988",
                        "#EE7733", "#CC3311", "#0077BB", "#BBBBBB",
                    ]
                    for i, coord_3d in enumerate(all_coords):
                        cand_x = int(coord_3d[0])
                        cand_y = int(coord_3d[1])
                        cand_z = int(coord_3d[2])

                        # Check if candidate is within ±1 slice of this view
                        if axis_id == 2 and abs(cand_z - slice_idx) <= 1:
                            px = w - 1 - cand_x  # radiological flip
                            py = h - 1 - cand_y
                        elif axis_id == 1 and abs(cand_y - slice_idx) <= 1:
                            px = w - 1 - cand_x
                            py = h - 1 - cand_z
                        elif axis_id == 0 and abs(cand_x - slice_idx) <= 1:
                            px = cand_y
                            py = h - 1 - cand_z
                        else:
                            continue  # Not in this view's slice range

                        color = candidate_colors[i % len(candidate_colors)]
                        size = 120 if i == 0 else 80
                        ax_plt.scatter(
                            [px], [py], c=color, s=size, marker="o",
                            edgecolors="black", linewidths=1.5, zorder=10,
                        )
                        if len(all_coords) > 1:
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
            location_label = "Candidate voxels" if qc.get("candidate_rejected") else "DSC AIF Location"
            fig.suptitle(f"{sub_label}{ses_label} | {location_label}", fontsize=12)

            plt.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)
