"""Nipype interface for HTML report generation.

This module provides a Nipype interface for generating NiPreps-style
HTML QC reports that reference external figure files via relative paths.
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


class GenerateReportInputSpec(BaseInterfaceInputSpec):
    """Input specification for GenerateReport."""

    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")
    pipeline = traits.Str(mandatory=True, desc="Pipeline name ('dsc', 'dce', 'anat', or 'hdglio')")
    output_dir = traits.Str(mandatory=True, desc="Output directory for the report")

    # qc_json enforces execution order (writer runs before report)
    qc_json = traits.Str(mandatory=True, desc="QC metrics JSON file path")


class GenerateReportOutputSpec(TraitedSpec):
    """Output specification for GenerateReport."""

    report_file = File(desc="Generated HTML report path")


class GenerateReport(BaseInterface):
    """Generate NiPreps-style HTML QC report with external figure files.

    This interface creates an HTML report that references figure PNGs
    via relative paths (``figures/<bids-name>.png``).  Figures are
    expected to already exist in the ``figures/`` subdirectory of the
    pipeline session output directory, written there by the
    corresponding derivative writer.

    The ``qc_json`` input enforces execution order so the writer
    (which creates both the QC JSON and the figures) always runs
    before this report node.
    """

    input_spec = GenerateReportInputSpec
    output_spec = GenerateReportOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.report_utils import generate_qc_report

        # Build output path following BIDS conventions
        output_dir = Path(self.inputs.output_dir)
        pipeline = self.inputs.pipeline
        subject_id = self.inputs.subject_id
        session_id = self.inputs.session_id

        # Build BIDS-compliant output path
        report_dir = output_dir / "pmriprep"
        if session_id:
            report_name = f"sub-{subject_id}_ses-{session_id}_{pipeline}.html"
        else:
            report_name = f"sub-{subject_id}_{pipeline}.html"

        output_path = report_dir / report_name

        # NOTE: figures_dir must match the path used by the derivative writers.
        # report_utils.py computes a relative path from the report location
        # (derivative root) to this directory using os.path.relpath().
        ses_dir = output_dir / "pmriprep" / f"sub-{subject_id}"
        if session_id:
            ses_dir = ses_dir / f"ses-{session_id}"
        figures_dir = str(ses_dir / "figures")

        # Generate the report
        self._report_file = generate_qc_report(
            subject_id=subject_id,
            session_id=session_id,
            pipeline=pipeline,
            qc_json_path=self.inputs.qc_json,
            output_path=str(output_path),
            figures_dir=figures_dir,
        )

        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["report_file"] = self._report_file
        return outputs
