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

import numpy as np

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
    """Input specification for AnatFiguresWriter interface."""

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
    """Output specification for AnatFiguresWriter interface."""

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


def _write_aif_timeseries_tsv(out_path: Path, values: np.ndarray, modality: str) -> None:
    """Write an AIF curve as a single-column ``_timeseries.tsv``.

    The column header is sourced from
    :func:`pmriprep.utils.sidecar_builders.aif_timeseries_columns` — the same
    single source the JSON sidecar's ``Columns`` is built from — so the TSV
    header and the sidecar cannot drift (``delta_r2_star`` for DSC ΔR2* in 1/s,
    ``plasma_concentration_mM`` for DCE plasma [Gd] in mM).
    """
    from pmriprep.utils.sidecar_builders import aif_timeseries_columns

    header = aif_timeseries_columns(modality)[0]
    np.savetxt(str(out_path), values, header=header, comments="")


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




class TumorSegFiguresWriterInputSpec(BaseInterfaceInputSpec):
    """Input specification for TumorSegFiguresWriter interface."""

    output_dir = traits.Either(traits.Str, File(exists=True), mandatory=True)
    subject_id = traits.Str(mandatory=True)
    session_id = traits.Str("", usedefault=True)

    # Figure filmstrips (optional)
    seg_filmstrip_postcon = _optional_figure_file()
    seg_filmstrip_flair = _optional_figure_file()
    nawm_filmstrip_flair = _optional_figure_file()


class TumorSegFiguresWriterOutputSpec(TraitedSpec):
    """Output specification for TumorSegFiguresWriter interface."""

    done = traits.Str(desc="Sentinel for Nipype barrier synchronization")


class TumorSegFiguresWriter(SimpleInterface):
    """Copies tumor figure filmstrips into ``figures/``.

    Per-region voxel counts, volumes, and the enhancing fraction live in
    the ``desc-tumorseg_dseg.json`` sidecar (``TumorVolumetrics`` block).
    """

    input_spec = TumorSegFiguresWriterInputSpec
    output_spec = TumorSegFiguresWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.report_utils import figure_filename

        output_dir = Path(self.inputs.output_dir)
        subject = self.inputs.subject_id
        session = self.inputs.session_id or ""

        ses_dir = output_dir / "pmriprep" / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"

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

        self._results["done"] = "done"
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
    """Input specification for DSCFiguresWriter interface."""

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
    """Output specification for DSCFiguresWriter interface."""

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
        bids_dir = self.inputs.bids_dir if isdefined(self.inputs.bids_dir) else ""
        source_path = (
            self.inputs.source_dsc
            if isdefined(self.inputs.source_dsc) and self.inputs.source_dsc
            else None
        )

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
            dst = perf_dir / f"{prefix}_desc-dscAif_timeseries.tsv"
            vals = np.loadtxt(str(self.inputs.aif_curve), dtype=np.float64)
            _write_aif_timeseries_tsv(dst, vals, "DSC")
            self._results["aif_curve_txt"] = str(dst)
            _write_aif_timeseries_sidecar(
                modality="DSC",
                aif_txt=dst,
                qc_metrics=(
                    self.inputs.qc_metrics
                    if isdefined(self.inputs.qc_metrics) else None
                ),
                bids_dir=bids_dir,
                source_path=source_path,
                subject=subject,
                session=session,
            )

        self._results["confounds_tsv"] = ""
        if isdefined(self.inputs.motion_params) and self.inputs.motion_params:
            tsv_path = perf_dir / f"{prefix}_desc-dscConfounds_timeseries.tsv"
            json_path = perf_dir / f"{prefix}_desc-dscConfounds_timeseries.json"
            par_to_confounds_tsv(self.inputs.motion_params, tsv_path)
            write_confounds_sidecar(json_path, source_path=source_path, bids_dir=bids_dir)
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
    """Input specification for DCEFiguresWriter interface."""

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
    """Output specification for DCEFiguresWriter interface."""

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
        bids_dir = self.inputs.bids_dir if isdefined(self.inputs.bids_dir) else ""
        source_path = (
            self.inputs.source_dce
            if isdefined(self.inputs.source_dce) and self.inputs.source_dce
            else None
        )

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
            dst = perf_dir / f"{prefix}_desc-dceAif_timeseries.tsv"
            vals = np.loadtxt(str(self.inputs.aif_curve), dtype=np.float64)
            _write_aif_timeseries_tsv(dst, vals, "DCE")
            self._results["aif_curve_txt"] = str(dst)
            _write_aif_timeseries_sidecar(
                modality="DCE",
                aif_txt=dst,
                qc_metrics=(
                    self.inputs.qc_metrics
                    if isdefined(self.inputs.qc_metrics) else None
                ),
                bids_dir=bids_dir,
                source_path=source_path,
                subject=subject,
                session=session,
            )

        self._results["confounds_tsv"] = ""
        if isdefined(self.inputs.motion_params) and self.inputs.motion_params:
            tsv_path = perf_dir / f"{prefix}_desc-dceConfounds_timeseries.tsv"
            json_path = perf_dir / f"{prefix}_desc-dceConfounds_timeseries.json"
            par_to_confounds_tsv(self.inputs.motion_params, tsv_path)
            write_confounds_sidecar(json_path, source_path=source_path, bids_dir=bids_dir)
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
