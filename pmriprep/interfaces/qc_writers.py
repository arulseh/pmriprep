"""QC writers and shared DataSink subclass for the per-pipeline factories.

The four ``init_*_datasinks_wf`` factories use
:class:`niworkflows.interfaces.bids.DerivativesDataSink` for every NIfTI +
sidecar pair. This module hosts the small bespoke pieces that don't fit the
DataSink model:

- :class:`PMRIPrepDataSink`: subclass that fixes ``out_path_base="pmriprep"``.
- :class:`AnatQCWriter`: writes ``desc-anatQC_quality.json`` and copies
  the anat figure filmstrips into ``figures/``.
- :func:`init_dataset_description_node`: factory for the per-session
  ``dataset_description.json`` barrier node.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, List, Optional

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from niworkflows.interfaces.bids import (  # type: ignore[import-untyped]
    DerivativesDataSink as _NiworkflowsDDS,
)


class PMRIPrepDataSink(_NiworkflowsDDS):
    """``DerivativesDataSink`` with ``out_path_base="pmriprep"`` baked in."""

    out_path_base = "pmriprep"


# ---------------------------------------------------------------------------
# Anat QC writer
# ---------------------------------------------------------------------------


class AnatQCWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)
    qc_metrics = traits.Dict(desc="Anatomical QC metrics dict")

    # Modality-presence flags (drive the warnings block in the QC JSON).
    # Mirrors the legacy AnatDerivativeWriter._data_dependent_inputs check.
    has_t1w_pre = traits.Bool(False, usedefault=True)
    has_t1w_postcon = traits.Bool(False, usedefault=True)
    has_t2w = traits.Bool(False, usedefault=True)
    has_flair = traits.Bool(False, usedefault=True)
    has_tissue_seg = traits.Bool(False, usedefault=True)

    # Figure filmstrips (all optional)
    brain_mask_filmstrip = File(exists=True)
    tissue_seg_filmstrip = File(exists=True)
    acpc_filmstrip = File(exists=True)
    coreg_filmstrip_postcon = File(exists=True)
    coreg_filmstrip_t2w = File(exists=True)
    coreg_filmstrip_flair = File(exists=True)
    coreg_filmstrip_flair_postcon = File(exists=True)


class AnatQCWriterOutputSpec(TraitedSpec):
    qc_json = File(desc="Path to anatQC_quality.json (empty if no metrics)")
    figures_dir = traits.Str(desc="Path to per-session figures/ directory")


class AnatQCWriter(SimpleInterface):
    """Writes the anat ``desc-anatQC_quality.json`` and copies filmstrips.

    Replaces the QC-and-figures portion of the bespoke
    ``AnatDerivativeWriter._run_interface``.
    """

    input_spec = AnatQCWriterInputSpec
    output_spec = AnatQCWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.report_utils import figure_filename

        output_dir = Path(self.inputs.output_dir)
        subject = self.inputs.subject_id
        session = self.inputs.session_id or ""

        prefix = f"sub-{subject}"
        if session:
            prefix += f"_ses-{session}"

        ses_dir = output_dir / "pmriprep" / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        ses_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------
        # QC JSON (with warnings derived from modality-presence flags)
        # ---------------------------------------------------------------
        warnings_list: List[str] = []
        if not self.inputs.has_tissue_seg:
            warnings_list.append(
                "Tissue segmentation (WM/GM/CSF probability maps) not available"
            )
        if not self.inputs.has_flair:
            warnings_list.append("FLAIR not available")
        if not self.inputs.has_t2w:
            warnings_list.append("T2w not available")
        # Postcon coreg is only expected when both pre and post T1w are present;
        # in postcon-only mode the postcon IS the reference, so no coreg warning.
        if (
            self.inputs.has_t1w_pre
            and self.inputs.has_t1w_postcon
            and not isdefined(self.inputs.coreg_filmstrip_postcon)
        ):
            warnings_list.append("Post-contrast T1w coregistration not available")

        qc_metrics = self.inputs.qc_metrics
        self._results["qc_json"] = ""
        if isdefined(qc_metrics) and qc_metrics:
            payload = dict(qc_metrics)
            if warnings_list:
                payload["warnings"] = warnings_list
            qc_path = ses_dir / f"{prefix}_desc-anatQC_quality.json"
            qc_path.write_text(json.dumps(payload, indent=2, default=str))
            self._results["qc_json"] = str(qc_path)

        # ---------------------------------------------------------------
        # Figures
        # ---------------------------------------------------------------
        figures_dir = ses_dir / "figures"
        fig_inputs = {
            "brain_mask": self.inputs.brain_mask_filmstrip,
            "tissue_seg": self.inputs.tissue_seg_filmstrip,
            "acpc": self.inputs.acpc_filmstrip,
            "coreg_postcon": self.inputs.coreg_filmstrip_postcon,
            "coreg_t2w": self.inputs.coreg_filmstrip_t2w,
            "coreg_flair": self.inputs.coreg_filmstrip_flair,
            "coreg_flair_postcon": self.inputs.coreg_filmstrip_flair_postcon,
        }
        for fig_key, src in fig_inputs.items():
            if not isdefined(src) or not src:
                continue
            figures_dir.mkdir(parents=True, exist_ok=True)
            fname = figure_filename(subject, session, fig_key, "anat")
            shutil.copy2(src, figures_dir / fname)

        self._results["figures_dir"] = str(figures_dir)
        return runtime


# ---------------------------------------------------------------------------
# Dataset description barrier
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tumor-segmentation QC writer
# ---------------------------------------------------------------------------


def _write_tumor_volumes_txt(qc_metrics: dict, path: Path) -> None:
    """Write tumor volume statistics to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seg_val = qc_metrics.get("segmentation", {}).get("validation", {})
    total_vol = seg_val.get("total_tumor_volume_ml", "N/A")
    enhancing_vol = seg_val.get("enhancing_volume_ml", "N/A")
    non_enhancing_vol = seg_val.get("non_enhancing_volume_ml", "N/A")
    total_voxels = seg_val.get("total_tumor_voxels", "N/A")
    enhancing_voxels = seg_val.get("enhancing_voxels", "N/A")
    non_enhancing_voxels = seg_val.get("non_enhancing_voxels", "N/A")
    voxel_vol = seg_val.get("voxel_volume_mm3", "N/A")

    def _fmt_vol(vol_ml: object) -> str:
        if vol_ml == "N/A" or vol_ml is None:
            return "N/A"
        val = float(str(vol_ml))
        return f"{val:.3f} mL ({val * 1000.0:.2f} mm3)"

    enhancing_frac = "N/A"
    if (
        enhancing_vol != "N/A"
        and total_vol != "N/A"
        and float(total_vol) > 0
    ):
        enhancing_frac = f"{float(enhancing_vol) / float(total_vol):.3f}"

    lines = [
        "Tumor Volume Statistics",
        "=" * 50,
        "",
        "Volumes:",
        f"  Total tumor:          {_fmt_vol(total_vol)}",
        f"  Enhancing tumor:      {_fmt_vol(enhancing_vol)}",
        f"  Non-enhancing tumor:  {_fmt_vol(non_enhancing_vol)}",
        "",
        "Voxel counts:",
        f"  Total tumor:          {total_voxels}",
        f"  Enhancing tumor:      {enhancing_voxels}",
        f"  Non-enhancing tumor:  {non_enhancing_voxels}",
        "",
        f"Voxel volume:           {voxel_vol} mm3",
        f"Enhancing fraction:     {enhancing_frac}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_tumor_lesion_stats_json(
    segmentation_file: str,
    subject: str,
    session: str,
    output_path: Path,
) -> None:
    """Write the per-region tumor-volume stats JSON read straight from the dseg."""
    from datetime import datetime, timezone

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    import pmriprep
    from pmriprep.utils.lesion_stats import compute_region_volume

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(segmentation_file)
    seg_data = np.asarray(img.dataobj, dtype=np.int16)  # type: ignore[attr-defined]
    zooms = img.header.get_zooms()[:3]  # type: ignore[attr-defined]
    voxel_sizes = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    enh_count, enh_mm3, enh_ml = compute_region_volume(seg_data == 2, voxel_sizes)
    nenh_count, nenh_mm3, nenh_ml = compute_region_volume(seg_data == 1, voxel_sizes)
    wt_count, wt_mm3, wt_ml = compute_region_volume(seg_data > 0, voxel_sizes)

    enhancing_fraction: Optional[float] = None
    if wt_ml > 0:
        enhancing_fraction = round(enh_ml / wt_ml, 4)

    result: dict = {
        "Description": "Tumor segmentation lesion volume statistics",
        "GeneratedBy": [{"Name": "pmriprep", "Version": pmriprep.__version__}],
        "GeneratedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Subject": f"sub-{subject}",
        "Session": f"ses-{session}" if session else "",
        "SegmentationSource": Path(segmentation_file).name,
        "SegmentationMethod": "HD-GLIO",
        "TotalLesionVoxels": wt_count,
        "Regions": {
            "contrast_enhancing_tumor": {
                "voxel_count": enh_count,
                "volume_mm3": round(enh_mm3, 2),
                "volume_ml": round(enh_ml, 3),
            },
            "non_enhancing_tumor": {
                "voxel_count": nenh_count,
                "volume_mm3": round(nenh_mm3, 2),
                "volume_ml": round(nenh_ml, 3),
            },
            "whole_tumor": {
                "voxel_count": wt_count,
                "volume_mm3": round(wt_mm3, 2),
                "volume_ml": round(wt_ml, 3),
            },
        },
        "derived_metrics": {"enhancing_fraction": enhancing_fraction},
    }
    output_path.write_text(json.dumps(result, indent=2, default=str))


class TumorSegQCWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)
    qc_metrics = traits.Dict(desc="Combined tumor-seg QC metrics")
    segmentation = File(
        exists=True, mandatory=True,
        desc="Multi-class tumor segmentation NIfTI (consumed for lesionStats)",
    )

    # Figure filmstrips (optional)
    seg_filmstrip_postcon = File(exists=True)
    seg_filmstrip_flair = File(exists=True)
    nawm_filmstrip_flair = File(exists=True)


class TumorSegQCWriterOutputSpec(TraitedSpec):
    qc_json = File(desc="Path to desc-tumorsegQC_quality.json (empty if no metrics)")
    lesion_stats_json = File(desc="Path to desc-lesionStats.json")
    volumes_txt = File(desc="Path to desc-tumorseg_volumes.txt (empty if no metrics)")


class TumorSegQCWriter(SimpleInterface):
    """Writes tumor-seg QC JSON, volumes.txt, lesionStats.json + filmstrip copies."""

    input_spec = TumorSegQCWriterInputSpec
    output_spec = TumorSegQCWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.report_utils import figure_filename

        output_dir = Path(self.inputs.output_dir)
        subject = self.inputs.subject_id
        session = self.inputs.session_id or ""

        prefix = f"sub-{subject}"
        if session:
            prefix += f"_ses-{session}"

        ses_dir = output_dir / "pmriprep" / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        anat_dir = ses_dir / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # QC JSON + volumes.txt (both gated on qc_metrics defined)
        qc_metrics = self.inputs.qc_metrics
        self._results["qc_json"] = ""
        self._results["volumes_txt"] = ""
        if isdefined(qc_metrics) and qc_metrics:
            qc_path = ses_dir / f"{prefix}_desc-tumorsegQC_quality.json"
            qc_path.write_text(json.dumps(qc_metrics, indent=2, default=str))
            self._results["qc_json"] = str(qc_path)

            volumes_path = anat_dir / f"{prefix}_desc-tumorseg_volumes.txt"
            _write_tumor_volumes_txt(qc_metrics, volumes_path)
            self._results["volumes_txt"] = str(volumes_path)

        # lesionStats.json (always written from segmentation NIfTI)
        lesion_path = ses_dir / f"{prefix}_space-anat_desc-lesionStats.json"
        _write_tumor_lesion_stats_json(
            segmentation_file=self.inputs.segmentation,
            subject=subject,
            session=session,
            output_path=lesion_path,
        )
        self._results["lesion_stats_json"] = str(lesion_path)

        # Figures
        figures_dir = ses_dir / "figures"
        fig_inputs = {
            "seg_postcon": self.inputs.seg_filmstrip_postcon,
            "seg_flair": self.inputs.seg_filmstrip_flair,
            "nawm_flair": self.inputs.nawm_filmstrip_flair,
        }
        for fig_key, src in fig_inputs.items():
            if not isdefined(src) or not src:
                continue
            figures_dir.mkdir(parents=True, exist_ok=True)
            fname = figure_filename(subject, session, fig_key, "tumor_seg")
            shutil.copy2(src, figures_dir / fname)

        return runtime


# ---------------------------------------------------------------------------
# Dataset description barrier
# ---------------------------------------------------------------------------


def write_dataset_description_node(output_dir, bids_dir, _trigger=None):
    """Nipype-Function: write ``dataset_description.json`` to the pipeline root.

    Joined on every per-pipeline DataSink's ``out_file`` so it runs exactly
    once per session. ``_trigger`` is the joined input — value unused.
    """
    from pathlib import Path
    from pmriprep.utils.outputs import write_dataset_description

    pipeline_dir = Path(output_dir) / "pmriprep"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_description(pipeline_dir, bids_dir)
    return str(pipeline_dir / "dataset_description.json")
