"""Nipype interface for unified HTML report generation.

This module provides a Nipype interface for generating a single unified
NiPreps-style HTML QC report per session, combining all pipeline sections
(anat, tumor_seg, dsc, dce) into one document.
"""

from pathlib import Path
from typing import Any

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)


class GenerateUnifiedReportInputSpec(BaseInterfaceInputSpec):
    """Input specification for GenerateUnifiedReport."""

    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")
    output_dir = traits.Str(mandatory=True, desc="Output directory (BIDS derivatives root)")

    # Which pipelines ran (controls which sections appear)
    ran_anat = traits.Bool(False, usedefault=True, desc="Whether anat pipeline ran")
    ran_tumor_seg = traits.Bool(False, usedefault=True, desc="Whether tumor segmentation pipeline ran")
    ran_dsc = traits.Bool(False, usedefault=True, desc="Whether dsc pipeline ran")
    ran_dce = traits.Bool(False, usedefault=True, desc="Whether dce pipeline ran")

    # Barrier strings: connected from writers to enforce execution ordering.
    # Each carries the qc_json path from its writer, ensuring the writer
    # finishes before the report node runs.
    anat_done = traits.Str("", usedefault=True, desc="Barrier from anat writer")
    tumor_seg_done = traits.Str("", usedefault=True, desc="Barrier from tumor_seg writer")
    dsc_done = traits.Str("", usedefault=True, desc="Barrier from dsc writer")
    dce_done = traits.Str("", usedefault=True, desc="Barrier from dce writer")

    # CLI command for reproducibility
    command_string = traits.Str("", usedefault=True, desc="CLI command used to run PMRIPrep")


class GenerateUnifiedReportOutputSpec(TraitedSpec):
    """Output specification for GenerateUnifiedReport."""

    report_file = File(desc="Generated unified HTML report path")


class GenerateUnifiedReport(BaseInterface):
    """Generate a single unified HTML QC report for all pipelines in a session.

    This interface creates one HTML report that combines QC sections for
    all pipelines that ran (anat, tumor_seg, dsc, dce), with a navigation
    bar and summary section. The report lives inside the session directory
    so figure paths are simply ``figures/<filename>``.

    Barrier inputs (``anat_done``, ``tumor_seg_done``, etc.) carry the
    ``qc_json`` path from each writer, enforcing execution order without
    needing explicit data dependencies.
    """

    input_spec = GenerateUnifiedReportInputSpec
    output_spec = GenerateUnifiedReportOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.report_utils import generate_unified_report

        output_dir = Path(self.inputs.output_dir)
        subject_id = self.inputs.subject_id
        session_id = self.inputs.session_id

        # Build session directory path
        ses_dir = output_dir / "pmriprep" / f"sub-{subject_id}"
        if session_id:
            ses_dir = ses_dir / f"ses-{session_id}"

        figures_dir = str(ses_dir / "figures")

        # Build report output path
        if session_id:
            report_name = f"sub-{subject_id}_ses-{session_id}.html"
        else:
            report_name = f"sub-{subject_id}.html"
        output_path = str(ses_dir / report_name)

        ran_pipelines = {
            "anat": self.inputs.ran_anat,
            "tumor_seg": self.inputs.ran_tumor_seg,
            "dsc": self.inputs.ran_dsc,
            "dce": self.inputs.ran_dce,
        }

        self._report_file = generate_unified_report(
            subject_id=subject_id,
            session_id=session_id,
            output_path=output_path,
            ses_dir=str(ses_dir),
            figures_dir=figures_dir,
            ran_pipelines=ran_pipelines,
            command_string=self.inputs.command_string,
        )

        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["report_file"] = getattr(self, "_report_file", "")
        return outputs
