"""AIF QC visualization interface for DSC-MRI.

This module implements Nipype interfaces for generating QC plots of the
detected AIF: a curve time series plot and a 3-pane location overlay.

Matches perf-plus ``plot_aif`` and ``plot_aif_location_overlay`` outputs.
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
        mandatory=True,
        desc="1D AIF curve text file",
    )
    aif_mask_file = File(
        exists=True,
        mandatory=True,
        desc="3D AIF voxel mask NIfTI",
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


class AIFQCPlotOutputSpec(TraitedSpec):
    """Output specification for AIFQCPlot interface."""

    out_aif_plot = File(exists=True, desc="AIF curve QC plot PNG")
    out_aif_location = File(exists=True, desc="AIF location overlay PNG")


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
        aif_curve = np.loadtxt(self.inputs.aif_curve_file)
        aif_mask_img = nib.load(self.inputs.aif_mask_file)
        aif_mask = aif_mask_img.get_fdata()  # type: ignore[attr-defined]

        dsc_img = nib.load(self.inputs.dsc_file)
        dsc_data = dsc_img.get_fdata()  # type: ignore[attr-defined]

        brain_mask_img = nib.load(self.inputs.brain_mask_file)
        brain_mask = brain_mask_img.get_fdata()  # type: ignore[attr-defined]

        voxel_sizes = tuple(float(z) for z in dsc_img.header.get_zooms()[:3])  # type: ignore[attr-defined,union-attr]

        tr = float(self.inputs.tr)
        qc = self.inputs.qc_metrics if isdefined(self.inputs.qc_metrics) else {}
        sub = self.inputs.subject_id or ""
        ses = self.inputs.session_id or ""

        # Generate plots
        out_dir = Path.cwd()

        aif_plot_path = str(out_dir / "aif_curve_qc.png")
        self._plot_aif_curve(
            aif_curve, tr, qc, sub, ses, aif_plot_path,
        )

        aif_location_path = str(out_dir / "aif_location_qc.png")
        self._plot_aif_location(
            dsc_data, aif_mask, brain_mask, voxel_sizes,
            qc, sub, ses, aif_location_path,
        )

        self._results = {
            "out_aif_plot": aif_plot_path,
            "out_aif_location": aif_location_path,
        }

        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        outputs = self.output_spec().get()  # type: ignore[misc]
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

        Matches perf-plus ``plot_aif`` output style.
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

            # TTP marker
            ttp_s = qc.get("best_ttp", None)
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
            baseline_frames = qc.get("baseline_frames_used", qc.get("n_baseline_frames", 5))
            baseline_end_t = baseline_frames * tr
            ax.axvspan(0, baseline_end_t, alpha=0.08, color="gray", label=f"Baseline ({baseline_frames} fr)")

            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Concentration (a.u.)", fontsize=11)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Title
            sub_label = f"sub-{subject_id}" if subject_id else ""
            ses_label = f" | ses-{session_id}" if session_id else ""
            ax.set_title(f"{sub_label}{ses_label} | DSC AIF Quality", fontsize=12)

            # Metrics text box
            detection_level = qc.get("detection_level", "unknown")
            peak_conc = float(np.max(aif_curve))
            fwhm = qc.get("best_fwhm", float("nan"))
            brr = qc.get("best_brr", float("nan"))
            psr = qc.get("best_psr", float("nan"))

            metrics_lines = [
                f"Detection: {detection_level}",
                f"Peak conc: {peak_conc:.2f}",
            ]
            if ttp_s is not None:
                metrics_lines.append(f"TTP: {ttp_s:.1f}s")
            if not np.isnan(fwhm):
                metrics_lines.append(f"FWHM: {fwhm:.1f}s")
            if arrival_s is not None:
                metrics_lines.append(f"Arrival: {arrival_s:.1f}s")
            if not np.isnan(brr):
                metrics_lines.append(f"BRR: {brr:.1f}%")
            if not np.isnan(psr):
                metrics_lines.append(f"PSR: {psr:.1f}%")

            metrics_text = "\n".join(metrics_lines)
            props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85)
            ax.text(
                0.02, 0.97, metrics_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=props,
            )

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
        """Plot 3-pane AIF location overlay on temporal mean DSC.

        Matches perf-plus ``plot_aif_location_overlay`` output style.
        """
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        from pmriprep.utils.qc_utils import get_radiological_slice

        # Compute temporal mean for background
        if dsc_data.ndim == 4:
            bg = np.mean(dsc_data, axis=-1)
        else:
            bg = dsc_data.copy()

        # Find AIF centroid for crosshair placement
        aif_coords = np.argwhere(aif_mask > 0.5)
        if len(aif_coords) == 0:
            # No AIF voxels — use center of brain
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

                # Magenta marker at centroid
                ax_plt.plot(cross_x, cross_y, "m+", markersize=12, markeredgewidth=2)

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
            fig.suptitle(f"{sub_label}{ses_label} | DSC AIF Location", fontsize=12)

            plt.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)
