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
# AIF timeseries sidecar writer (Phase 2a)
# ---------------------------------------------------------------------------


def _write_aif_timeseries_sidecar(
    *,
    modality: str,
    aif_txt: Path,
    qc_metrics: Optional[dict],
    bids_dir: str,
    source_path: Optional[str],
    subject: str,
    session: str,
) -> None:
    """Write the JSON sidecar next to a copied AIF curve TXT.

    Reads ``qc_metrics["aif_detection"]`` (the producer-shape ``aif_qc``
    dict) and emits flat ``AIF*``-prefixed top-level keys via
    :func:`pmriprep.utils.sidecar_builders.build_aif_timeseries_sidecar`.
    Silently skips when ``qc_metrics`` is missing or empty — the TXT
    still exists; the sidecar is opt-in metadata.
    """
    if not qc_metrics:
        return
    aif_qc = qc_metrics.get("aif_detection") if isinstance(qc_metrics, dict) else None
    if not aif_qc:
        return

    from pmriprep.utils.sidecar_builders import build_aif_timeseries_sidecar

    sources = [source_path] if source_path else []
    sidecar = build_aif_timeseries_sidecar(
        modality=modality,
        source_paths=sources,
        bids_dir=bids_dir or ".",
        subject=subject,
        session=session or None,
        aif_qc=aif_qc,
    )
    json_path = aif_txt.with_suffix(".json")
    json_path.write_text(json.dumps(sidecar, indent=2, default=str))


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
# DSC QC writer
# ---------------------------------------------------------------------------


_DSC_FIGURE_KEYS = {
    "dsc_brain_mask": "brain_mask_filmstrip",
    "motion": "motion_qc_plot",
    "aif_curve": "aif_curve_plot",
    "aif_location": "aif_location_plot",
    "aif_candidates": "aif_candidates_plot",
    "registration": "registration_filmstrip",
    "dsc_preprocessing_qc": "dsc_preprocessing_qc_plot",
    "dsc_param_maps_qc": "dsc_param_maps_qc_plot",
    "dsc_fit_qc": "dsc_fit_qc_plot",
    "dsc_leakage_qc": "dsc_leakage_qc_plot",
}


class DSCQCWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)
    qc_metrics = traits.Dict(desc="Merged DSC QC metrics dict")
    provenance_dict = traits.Dict(desc="DSC physics provenance dict (drives warnings)")
    preflight_warnings = traits.List(traits.Str, value=[], usedefault=True)
    bids_dir = traits.Str("", usedefault=True, desc="BIDS rawdata root for AIF sidecar Sources URIs")
    source_dsc = File(exists=False, desc="Raw DSC NIfTI for AIF sidecar Sources field")

    has_brain_mask_dsc = traits.Bool(False, usedefault=True)
    has_nawm_mask_dsc = traits.Bool(False, usedefault=True)
    has_lesion_mask_dsc = traits.Bool(False, usedefault=True)
    has_dsc_to_t1w_xfm = traits.Bool(False, usedefault=True)

    aif_curve = File(exists=True, desc="AIF curve text file")
    motion_params = File(exists=True, desc=".par motion file (converted to confounds TSV)")
    lesion_stats_json = File(exists=True)
    roi_stats_json = File(exists=True)

    brain_mask_filmstrip = File(exists=True)
    motion_qc_plot = File(exists=True)
    aif_curve_plot = File(exists=True)
    aif_location_plot = File(exists=True)
    aif_candidates_plot = File(exists=True)
    registration_filmstrip = File(exists=True)
    dsc_preprocessing_qc_plot = File(exists=True)
    dsc_param_maps_qc_plot = File(exists=True)
    dsc_fit_qc_plot = File(exists=True)
    dsc_leakage_qc_plot = File(exists=True)


class DSCQCWriterOutputSpec(TraitedSpec):
    qc_json = File(desc="Path to desc-dscQC_quality.json (empty if no metrics)")
    aif_curve_txt = File(desc="Path to copied AIF curve TXT (empty if not provided)")
    confounds_tsv = File(desc="Path to derived confounds TSV (empty if no motion params)")


class DSCQCWriter(SimpleInterface):
    """Writes the DSC QC JSON, AIF curve copy, confounds TSV/JSON, lesion/roi
    stats JSON copies, and figure filmstrips.
    """

    input_spec = DSCQCWriterInputSpec
    output_spec = DSCQCWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.outputs import (
            par_to_confounds_tsv,
            write_confounds_sidecar,
        )
        from pmriprep.utils.qc_standards import recommend_dsc_qc
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
        perf_dir = ses_dir / "perf"
        perf_dir.mkdir(parents=True, exist_ok=True)

        self._results["aif_curve_txt"] = ""
        if isdefined(self.inputs.aif_curve) and self.inputs.aif_curve:
            dst = perf_dir / f"{prefix}_desc-dscAif_timeseries.txt"
            shutil.copy2(self.inputs.aif_curve, dst)
            self._results["aif_curve_txt"] = str(dst)
            _write_aif_timeseries_sidecar(
                modality="DSC",
                aif_txt=dst,
                qc_metrics=self.inputs.qc_metrics if isdefined(self.inputs.qc_metrics) else None,
                bids_dir=self.inputs.bids_dir if isdefined(self.inputs.bids_dir) else "",
                source_path=(
                    self.inputs.source_dsc
                    if isdefined(self.inputs.source_dsc) and self.inputs.source_dsc
                    else None
                ),
                subject=subject,
                session=session,
            )

        self._results["confounds_tsv"] = ""
        if isdefined(self.inputs.motion_params) and self.inputs.motion_params:
            tsv_path = perf_dir / f"{prefix}_desc-dscConfounds_timeseries.tsv"
            json_path = perf_dir / f"{prefix}_desc-dscConfounds_timeseries.json"
            par_to_confounds_tsv(self.inputs.motion_params, tsv_path)
            write_confounds_sidecar(json_path)
            self._results["confounds_tsv"] = str(tsv_path)

        if isdefined(self.inputs.lesion_stats_json) and self.inputs.lesion_stats_json:
            shutil.copy2(
                self.inputs.lesion_stats_json,
                ses_dir / f"{prefix}_space-dsc_desc-lesionStats.json",
            )
        if isdefined(self.inputs.roi_stats_json) and self.inputs.roi_stats_json:
            shutil.copy2(
                self.inputs.roi_stats_json,
                ses_dir / f"{prefix}_space-dsc_desc-roiStats.json",
            )

        figures_dir = ses_dir / "figures"
        for fig_key, attr in _DSC_FIGURE_KEYS.items():
            src = getattr(self.inputs, attr)
            if not isdefined(src) or not src:
                continue
            figures_dir.mkdir(parents=True, exist_ok=True)
            fname = figure_filename(subject, session, fig_key, "dsc")
            shutil.copy2(src, figures_dir / fname)

        self._results["qc_json"] = ""
        qc_metrics = self.inputs.qc_metrics
        if isdefined(qc_metrics) and qc_metrics:
            qc = dict(qc_metrics)
            warnings_list: list = []
            if not self.inputs.has_brain_mask_dsc:
                warnings_list.append("Brain mask in DSC space not available")
            if not self.inputs.has_nawm_mask_dsc:
                warnings_list.append("NAWM mask in DSC space not available")
            if not self.inputs.has_lesion_mask_dsc:
                warnings_list.append("Lesion mask in DSC space not available")
            if not self.inputs.has_dsc_to_t1w_xfm:
                warnings_list.append("DSC to T1w registration transform not available")
            if warnings_list:
                qc["warnings"] = warnings_list

            provenance = self.inputs.provenance_dict
            if isdefined(provenance) and provenance:
                from pmriprep.utils.contrast_agent import collect_default_warnings

                prov_warnings = collect_default_warnings(
                    provenance, "DSC", qc_metrics=qc,
                )
                if prov_warnings:
                    qc["provenance_warnings"] = prov_warnings

            from pmriprep.utils.contrast_agent import collect_aif_fallback_warning

            aif_warn = collect_aif_fallback_warning(qc, "DSC")
            if aif_warn:
                qc.setdefault("acquisition_warnings", []).append(aif_warn)

            pf = self.inputs.preflight_warnings
            if isdefined(pf) and pf:
                qc.setdefault("acquisition_warnings", []).extend(list(pf))

            enhanced = recommend_dsc_qc(qc)
            qc_path = ses_dir / f"{prefix}_desc-dscQC_quality.json"
            qc_path.write_text(json.dumps(enhanced, indent=2, default=str))
            self._results["qc_json"] = str(qc_path)

        return runtime


# ---------------------------------------------------------------------------
# DCE QC writer
# ---------------------------------------------------------------------------


_DCE_FIGURE_KEYS = {
    "dce_brain_mask": "brain_mask_filmstrip",
    "motion": "motion_qc_plot",
    "aif_curve": "aif_curve_plot",
    "aif_location": "aif_location_plot",
    "aif_candidates": "aif_candidates_plot",
    "registration": "registration_filmstrip",
    "b1": "b1_qc_plot",
    "t1": "t1_qc_plot",
    "preprocessing_qc": "preprocessing_qc_plot",
    "pk_modeling_qc": "pk_modeling_qc_plot",
    "pk_param_maps_qc": "pk_param_maps_qc_plot",
    "model_selection_qc": "model_selection_qc_plot",
    "dsc_dce_xval": "dsc_dce_xval_plot",
}


class DCEQCWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)
    qc_metrics = traits.Dict(desc="Merged DCE QC metrics dict")
    provenance_dict = traits.Dict(desc="DCE physics provenance (drives warnings)")
    preflight_warnings = traits.List(traits.Str, value=[], usedefault=True)
    bids_dir = traits.Str("", usedefault=True, desc="BIDS rawdata root for AIF sidecar Sources URIs")
    source_dce = File(exists=False, desc="Raw DCE NIfTI for AIF sidecar Sources field")

    has_t1_map = traits.Bool(False, usedefault=True)
    has_b1_map = traits.Bool(False, usedefault=True)
    has_brain_mask = traits.Bool(False, usedefault=True)
    has_nawm_mask = traits.Bool(False, usedefault=True)
    has_lesion_mask_dce = traits.Bool(False, usedefault=True)
    has_dce_to_t1w_xfm = traits.Bool(False, usedefault=True)

    aif_curve = File(exists=True, desc="AIF curve text file")
    motion_params = File(exists=True, desc=".par motion file (→ confounds TSV)")
    lesion_stats_json = File(exists=True)
    roi_stats_json = File(exists=True)

    brain_mask_filmstrip = File(exists=True)
    motion_qc_plot = File(exists=True)
    aif_curve_plot = File(exists=True)
    aif_location_plot = File(exists=True)
    aif_candidates_plot = File(exists=True)
    registration_filmstrip = File(exists=True)
    b1_qc_plot = File(exists=True)
    t1_qc_plot = File(exists=True)
    preprocessing_qc_plot = File(exists=True)
    pk_modeling_qc_plot = File(exists=True)
    pk_param_maps_qc_plot = File(exists=True)
    model_selection_qc_plot = File(exists=True)
    dsc_dce_xval_plot = File(exists=True)


class DCEQCWriterOutputSpec(TraitedSpec):
    qc_json = File(desc="Path to desc-dceQC_quality.json (empty if no metrics)")
    aif_curve_txt = File(desc="Path to copied AIF curve TXT (empty if not provided)")
    confounds_tsv = File(desc="Path to derived confounds TSV (empty if no motion params)")


class DCEQCWriter(SimpleInterface):
    """DCE QC JSON, AIF curve copy, confounds TSV/JSON, lesion/ROI stats
    copies, figure copies, and stale-figure cleanup on rerun.
    """

    input_spec = DCEQCWriterInputSpec
    output_spec = DCEQCWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.outputs import (
            par_to_confounds_tsv,
            write_confounds_sidecar,
        )
        from pmriprep.utils.qc_standards import recommend_dce_qc
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
        perf_dir = ses_dir / "perf"
        perf_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = ses_dir / "figures"

        self._results["aif_curve_txt"] = ""
        if isdefined(self.inputs.aif_curve) and self.inputs.aif_curve:
            dst = perf_dir / f"{prefix}_desc-dceAif_timeseries.txt"
            shutil.copy2(self.inputs.aif_curve, dst)
            self._results["aif_curve_txt"] = str(dst)
            _write_aif_timeseries_sidecar(
                modality="DCE",
                aif_txt=dst,
                qc_metrics=self.inputs.qc_metrics if isdefined(self.inputs.qc_metrics) else None,
                bids_dir=self.inputs.bids_dir if isdefined(self.inputs.bids_dir) else "",
                source_path=(
                    self.inputs.source_dce
                    if isdefined(self.inputs.source_dce) and self.inputs.source_dce
                    else None
                ),
                subject=subject,
                session=session,
            )

        self._results["confounds_tsv"] = ""
        if isdefined(self.inputs.motion_params) and self.inputs.motion_params:
            tsv_path = perf_dir / f"{prefix}_desc-dceConfounds_timeseries.tsv"
            json_path = perf_dir / f"{prefix}_desc-dceConfounds_timeseries.json"
            par_to_confounds_tsv(self.inputs.motion_params, tsv_path)
            write_confounds_sidecar(json_path)
            self._results["confounds_tsv"] = str(tsv_path)

        if isdefined(self.inputs.lesion_stats_json) and self.inputs.lesion_stats_json:
            shutil.copy2(
                self.inputs.lesion_stats_json,
                ses_dir / f"{prefix}_space-dce_desc-lesionStats.json",
            )
        if isdefined(self.inputs.roi_stats_json) and self.inputs.roi_stats_json:
            shutil.copy2(
                self.inputs.roi_stats_json,
                ses_dir / f"{prefix}_space-dce_desc-roiStats.json",
            )

        # Stale AIF location figure cleanup (population/external AIF)
        qc_metrics = self.inputs.qc_metrics
        if isdefined(qc_metrics) and qc_metrics:
            method = (
                qc_metrics.get("aif_detection", {})
                .get("detection_method", "unknown")
            )
            if method in ("population", "external"):
                stale = figures_dir / figure_filename(
                    subject, session, "aif_location", "dce",
                )
                if stale.exists():
                    stale.unlink()

        for fig_key, attr in _DCE_FIGURE_KEYS.items():
            src = getattr(self.inputs, attr)
            if not isdefined(src) or not src:
                continue
            figures_dir.mkdir(parents=True, exist_ok=True)
            fname = figure_filename(subject, session, fig_key, "dce")
            shutil.copy2(src, figures_dir / fname)

        # QC JSON enrichment + write
        self._results["qc_json"] = ""
        if isdefined(qc_metrics) and qc_metrics:
            qc = dict(qc_metrics)
            warnings_list: list = []
            if not self.inputs.has_b1_map:
                warnings_list.append("B1+ transmit field map not available")
            if not self.inputs.has_t1_map:
                warnings_list.append("T1 relaxation time map not available")
            if not self.inputs.has_brain_mask:
                warnings_list.append("Brain mask in DCE space not available")
            if not self.inputs.has_nawm_mask:
                warnings_list.append("NAWM mask in DCE space not available")
            if not self.inputs.has_lesion_mask_dce:
                warnings_list.append("Lesion mask in DCE space not available")
            if not self.inputs.has_dce_to_t1w_xfm:
                warnings_list.append("DCE to T1w registration transform not available")
            if warnings_list:
                qc["warnings"] = warnings_list

            provenance = self.inputs.provenance_dict
            if isdefined(provenance) and provenance:
                from pmriprep.utils.contrast_agent import collect_default_warnings
                prov_warnings = collect_default_warnings(provenance, "DCE")
                if prov_warnings:
                    qc["provenance_warnings"] = prov_warnings

            from pmriprep.utils.contrast_agent import collect_aif_fallback_warning
            aif_warn = collect_aif_fallback_warning(qc, "DCE")
            if aif_warn:
                qc.setdefault("acquisition_warnings", []).append(aif_warn)
            pf = self.inputs.preflight_warnings
            if isdefined(pf) and pf:
                qc.setdefault("acquisition_warnings", []).extend(list(pf))

            enhanced = recommend_dce_qc(qc)
            qc_path = ses_dir / f"{prefix}_desc-dceQC_quality.json"
            qc_path.write_text(json.dumps(enhanced, indent=2, default=str))
            self._results["qc_json"] = str(qc_path)

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
