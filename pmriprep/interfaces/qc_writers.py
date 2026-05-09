"""Figure writers and shared DataSink subclass for the per-pipeline factories.

The four ``init_*_datasinks_wf`` factories use
:class:`niworkflows.interfaces.bids.DerivativesDataSink` for every NIfTI +
sidecar pair. This module hosts the small bespoke pieces that don't fit the
DataSink model:

- :class:`PMRIPrepDataSink`: subclass that fixes ``out_path_base="pmriprep"``.
- :class:`AnatFiguresWriter` / :class:`TumorSegFiguresWriter` /
  :class:`DSCFiguresWriter` / :class:`DCEFiguresWriter`: copy the
  per-pipeline figure filmstrips into ``figures/`` and write the
  AIF-curve TSV companion sidecar (DSC/DCE). All QC summary JSONs live
  on the canonical per-image sidecars (the ``QC*`` flat keys on the
  preproc / probseg / dseg sidecars).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

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


def _optional_figure_file(desc: str = "") -> traits.Either:
    """File-trait that also accepts ``None`` from upstream Function nodes.

    Figure-producing Function nodes (e.g. ``_dsc_dce_xval_plot_node``)
    legitimately return Python ``None`` when their figure is not
    applicable. Nipype's ``File`` validator calls ``Path(value)``
    unconditionally, which raises ``TraitError`` on ``None`` and stalls
    the DAG via a downstream ``FileNotFoundError`` on the missing
    ``result_*.pklz``. Wrapping ``File(exists=True)`` in
    ``traits.Either(..., traits.Enum(None))`` lets the value through;
    each writer's ``_run_interface`` already skips falsy/undefined inputs
    via ``if not isdefined(src) or not src: continue``.
    """
    return traits.Either(
        File(exists=True), traits.Enum(None), desc=desc,
    )


# ---------------------------------------------------------------------------
# Anat QC writer
# ---------------------------------------------------------------------------


class AnatFiguresWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)

    # Figure filmstrips (all optional)
    brain_mask_filmstrip = _optional_figure_file()
    tissue_seg_filmstrip = _optional_figure_file()
    acpc_filmstrip = _optional_figure_file()
    coreg_filmstrip_postcon = _optional_figure_file()
    coreg_filmstrip_t2w = _optional_figure_file()
    coreg_filmstrip_flair = _optional_figure_file()
    coreg_filmstrip_flair_postcon = _optional_figure_file()


class AnatFiguresWriterOutputSpec(TraitedSpec):
    figures_dir = traits.Str(desc="Path to per-session figures/ directory")


class AnatFiguresWriter(SimpleInterface):
    """Copies the anat figure filmstrips into ``figures/``.

    QC summary fields (recommendation / warnings) live on the canonical
    ``desc-preproc_T1w.json`` anchor sidecar as flat ``QC*`` keys; this
    writer no longer emits a session-root ``desc-anatQC_quality.json``.
    """

    input_spec = AnatFiguresWriterInputSpec
    output_spec = AnatFiguresWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.report_utils import figure_filename

        output_dir = Path(self.inputs.output_dir)
        subject = self.inputs.subject_id
        session = self.inputs.session_id or ""

        ses_dir = output_dir / "pmriprep" / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        ses_dir.mkdir(parents=True, exist_ok=True)

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


def _write_tumor_clinical_context_json(
    segmentation_file: str,
    subject: str,
    session: str,
    output_path: Path,
) -> None:
    """Write per-region tumor-volume context JSON to the anat/ scope.

    Replaces the legacy session-root ``space-anat_desc-lesionStats.json``
    orphan with an anat-scoped clinical-context sidecar carrying region
    voxel counts, volumes (mm³ / mL), and the enhancing fraction. The
    canonical voxel-count/volume table lives in
    ``<prefix>_space-T1w_desc-tumorseg_dseg.tsv`` (BIDS-Common-Derivatives
    dseg lookup).
    """
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
        "Description": (
            "Whole-tumor clinical context (volumes + enhancing fraction). "
            "Per-region voxel counts and volumes are also available in "
            "the companion desc-tumorseg_dseg.tsv."
        ),
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


class TumorSegResumeWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)
    qc_metrics = traits.Dict(
        desc="Producer-shape qc_metrics envelope (preprocessing/segmentation/nawm).",
    )


class TumorSegResumeWriterOutputSpec(TraitedSpec):
    out_file = File(
        desc="Path to anat/<prefix>_desc-tumorResume_pmriprep.json",
    )


class TumorSegResumeWriter(SimpleInterface):
    """Writes the tumor-segmentation resume sidecar.

    The sidecar carries the producer-shape ``qc_metrics`` envelope
    (preprocessing / segmentation / nawm blocks) so the HTML aggregator
    can recover the nested shape that ``section_tumor_seg.html``
    consumes.
    """

    input_spec = TumorSegResumeWriterInputSpec
    output_spec = TumorSegResumeWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.sidecar_builders import build_tumor_resume_sidecar

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

        qc_metrics = (
            dict(self.inputs.qc_metrics)
            if isdefined(self.inputs.qc_metrics)
            and isinstance(self.inputs.qc_metrics, dict)
            else None
        )
        sidecar = build_tumor_resume_sidecar(qc_metrics=qc_metrics)

        out_path = anat_dir / f"{prefix}_desc-tumorResume_pmriprep.json"
        out_path.write_text(json.dumps(sidecar, indent=2, default=str))
        self._results["out_file"] = str(out_path)
        return runtime


class TumorSegFiguresWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)
    segmentation = File(
        exists=True, mandatory=True,
        desc="Multi-class tumor segmentation NIfTI",
    )

    # Figure filmstrips (optional)
    seg_filmstrip_postcon = _optional_figure_file()
    seg_filmstrip_flair = _optional_figure_file()
    nawm_filmstrip_flair = _optional_figure_file()


class TumorSegFiguresWriterOutputSpec(TraitedSpec):
    clinical_context_json = File(
        desc="Path to anat/<prefix>_desc-tumorClinicalContext.json",
    )


class TumorSegFiguresWriter(SimpleInterface):
    """Writes the tumor-clinical-context JSON + tumor figure filmstrips.

    The legacy session-root ``desc-tumorsegQC_quality.json``,
    ``desc-tumorseg_volumes.txt``, and ``space-anat_desc-lesionStats.json``
    files are gone after v0.46.0. Per-region voxel counts and volumes
    live in the canonical ``desc-tumorseg_dseg.tsv`` (BIDS dseg lookup
    table); whole-tumor clinical context (enhancing fraction + summary
    volumes) is now anat-scoped at
    ``anat/<prefix>_desc-tumorClinicalContext.json``.
    """

    input_spec = TumorSegFiguresWriterInputSpec
    output_spec = TumorSegFiguresWriterOutputSpec

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

        clinical_context_path = (
            anat_dir / f"{prefix}_desc-tumorClinicalContext.json"
        )
        _write_tumor_clinical_context_json(
            segmentation_file=self.inputs.segmentation,
            subject=subject,
            session=session,
            output_path=clinical_context_path,
        )
        self._results["clinical_context_json"] = str(clinical_context_path)

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


class DSCFiguresWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)
    qc_metrics = traits.Dict(desc="Merged DSC QC metrics (drives AIF sidecar only)")
    bids_dir = traits.Str("", usedefault=True, desc="BIDS rawdata root for AIF sidecar Sources URIs")
    source_dsc = File(exists=False, desc="Raw DSC NIfTI for AIF sidecar Sources field")

    aif_curve = File(exists=True, desc="AIF curve text file")
    motion_params = File(exists=True, desc=".par motion file (converted to confounds TSV)")

    brain_mask_filmstrip = _optional_figure_file()
    motion_qc_plot = _optional_figure_file()
    aif_curve_plot = _optional_figure_file()
    aif_location_plot = _optional_figure_file()
    aif_candidates_plot = _optional_figure_file()
    registration_filmstrip = _optional_figure_file()
    dsc_preprocessing_qc_plot = _optional_figure_file()
    dsc_param_maps_qc_plot = _optional_figure_file()
    dsc_fit_qc_plot = _optional_figure_file()
    dsc_leakage_qc_plot = _optional_figure_file()


class DSCFiguresWriterOutputSpec(TraitedSpec):
    aif_curve_txt = File(desc="Path to copied AIF curve TXT (empty if not provided)")
    confounds_tsv = File(desc="Path to derived confounds TSV (empty if no motion params)")


class DSCFiguresWriter(SimpleInterface):
    """Copies the DSC AIF curve + confounds TSV/JSON + figure filmstrips.

    QC summary fields (recommendation / warnings) live on
    ``desc-dscPreproc_dsc.json`` as flat ``QC*`` keys; per-map lesion
    stats live on the per-PK-map sidecars as ``LesionStats`` fragments.
    This writer no longer emits a session-root
    ``desc-dscQC_quality.json`` or copies the per-map lesion/ROI stats
    JSONs.
    """

    input_spec = DSCFiguresWriterInputSpec
    output_spec = DSCFiguresWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.outputs import (
            par_to_confounds_tsv,
            write_confounds_sidecar,
        )
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
                qc_metrics=(
                    self.inputs.qc_metrics
                    if isdefined(self.inputs.qc_metrics) else None
                ),
                bids_dir=(
                    self.inputs.bids_dir
                    if isdefined(self.inputs.bids_dir) else ""
                ),
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

        figures_dir = ses_dir / "figures"
        for fig_key, attr in _DSC_FIGURE_KEYS.items():
            src = getattr(self.inputs, attr)
            if not isdefined(src) or not src:
                continue
            figures_dir.mkdir(parents=True, exist_ok=True)
            fname = figure_filename(subject, session, fig_key, "dsc")
            shutil.copy2(src, figures_dir / fname)

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


class DCEFiguresWriterInputSpec(BaseInterfaceInputSpec):
    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)
    qc_metrics = traits.Dict(desc="Merged DCE QC metrics (drives AIF sidecar only)")
    bids_dir = traits.Str("", usedefault=True, desc="BIDS rawdata root for AIF sidecar Sources URIs")
    source_dce = File(exists=False, desc="Raw DCE NIfTI for AIF sidecar Sources field")

    aif_curve = File(exists=True, desc="AIF curve text file")
    motion_params = File(exists=True, desc=".par motion file (→ confounds TSV)")

    brain_mask_filmstrip = _optional_figure_file()
    motion_qc_plot = _optional_figure_file()
    aif_curve_plot = _optional_figure_file()
    aif_location_plot = _optional_figure_file()
    aif_candidates_plot = _optional_figure_file()
    registration_filmstrip = _optional_figure_file()
    b1_qc_plot = _optional_figure_file()
    t1_qc_plot = _optional_figure_file()
    preprocessing_qc_plot = _optional_figure_file()
    pk_modeling_qc_plot = _optional_figure_file()
    pk_param_maps_qc_plot = _optional_figure_file()
    model_selection_qc_plot = _optional_figure_file()
    dsc_dce_xval_plot = _optional_figure_file()


class DCEFiguresWriterOutputSpec(TraitedSpec):
    aif_curve_txt = File(desc="Path to copied AIF curve TXT (empty if not provided)")
    confounds_tsv = File(desc="Path to derived confounds TSV (empty if no motion params)")


class DCEFiguresWriter(SimpleInterface):
    """Copies the DCE AIF curve + confounds TSV/JSON + figure filmstrips.

    QC summary fields (recommendation / warnings) live on
    ``desc-dcePreproc_dce.json`` as flat ``QC*`` keys; per-PK-map lesion
    stats live on the per-map sidecars as ``LesionStats`` fragments.
    This writer no longer emits a session-root
    ``desc-dceQC_quality.json`` or copies the per-map lesion/ROI stats
    JSONs. It still cleans up stale ``aif_location`` figures on rerun
    when AIF detection falls back to population/external.
    """

    input_spec = DCEFiguresWriterInputSpec
    output_spec = DCEFiguresWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.outputs import (
            par_to_confounds_tsv,
            write_confounds_sidecar,
        )
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
                qc_metrics=(
                    self.inputs.qc_metrics
                    if isdefined(self.inputs.qc_metrics) else None
                ),
                bids_dir=(
                    self.inputs.bids_dir
                    if isdefined(self.inputs.bids_dir) else ""
                ),
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
