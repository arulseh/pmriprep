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
