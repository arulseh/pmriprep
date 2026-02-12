"""Nipype interface for HTML report generation.

This module provides a Nipype interface for generating NiPreps-style
HTML QC reports with embedded visualizations.
"""

from pathlib import Path
from typing import Any

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
    isdefined,
)


class GenerateReportInputSpec(BaseInterfaceInputSpec):
    """Input specification for GenerateReport."""

    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")
    pipeline = traits.Str(mandatory=True, desc="Pipeline name ('dsc' or 'dce')")
    output_dir = traits.Str(mandatory=True, desc="Output directory for the report")

    qc_json = traits.Str(mandatory=True, desc="QC metrics JSON file path")
    motion_qc_plot = traits.Str(desc="Motion QC plot PNG path")
    aif_curve_plot = traits.Str(desc="AIF curve plot PNG path")
    aif_location_plot = traits.Str(desc="AIF location plot PNG path")
    registration_filmstrip = traits.Str(desc="Registration filmstrip PNG path")
    b1_qc_plot = traits.Str(desc="B1+ mapping QC plot PNG path")
    t1_qc_plot = traits.Str(desc="VFA T1 mapping QC plot PNG path")

    # DCE QC plots
    preprocessing_qc_plot = traits.Str(desc="DCE preprocessing QC plot PNG path")
    pk_modeling_qc_plot = traits.Str(desc="PK modeling QC plot PNG path")
    pk_param_maps_qc_plot = traits.Str(desc="PK parameter maps QC plot PNG path")

    # HD-GLIO segmentation filmstrips
    seg_filmstrip_postcon = traits.Str(desc="Tumor segmentation on T1w postcon filmstrip PNG path")
    seg_filmstrip_flair = traits.Str(desc="Tumor segmentation on FLAIR filmstrip PNG path")
    nawm_filmstrip_flair = traits.Str(desc="NAWM ROI on FLAIR filmstrip PNG path")

    # Anatomical pipeline filmstrips
    brain_mask_filmstrip = traits.Str(desc="Brain mask QC filmstrip PNG path")
    tissue_seg_filmstrip = traits.Str(desc="Tissue segmentation QC filmstrip PNG path")
    acpc_filmstrip = traits.Str(desc="ACPC alignment QC filmstrip PNG path")
    coreg_filmstrip_postcon = traits.Str(desc="T1w postcon coregistration filmstrip PNG path")
    coreg_filmstrip_t2w = traits.Str(desc="T2w coregistration filmstrip PNG path")
    coreg_filmstrip_flair = traits.Str(desc="FLAIR coregistration filmstrip PNG path")
    coreg_filmstrip_pdw = traits.Str(desc="PDw coregistration filmstrip PNG path")


class GenerateReportOutputSpec(TraitedSpec):
    """Output specification for GenerateReport."""

    report_file = File(desc="Generated HTML report path")


class GenerateReport(BaseInterface):
    """Generate NiPreps-style HTML QC report.

    This interface creates a self-contained HTML report with embedded
    images for visual quality control. The report includes:

    - Overall QC recommendation (include/review/exclude)
    - Motion quality metrics and visualization
    - AIF detection quality plots
    - Registration quality filmstrip
    - Pipeline-specific metrics (deconvolution for DSC, PK modeling for DCE)

    All images are embedded as base64 to create a standalone HTML file.
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
        if session_id:
            report_dir = (
                output_dir
                / f"{pipeline}_pipeline"
                / f"sub-{subject_id}"
                / f"ses-{session_id}"
            )
            report_name = f"sub-{subject_id}_ses-{session_id}_{pipeline}.html"
        else:
            report_dir = (
                output_dir
                / f"{pipeline}_pipeline"
                / f"sub-{subject_id}"
            )
            report_name = f"sub-{subject_id}_{pipeline}.html"

        output_path = report_dir / report_name

        # Get optional inputs
        motion_qc_plot = None
        if isdefined(self.inputs.motion_qc_plot) and self.inputs.motion_qc_plot:
            if Path(self.inputs.motion_qc_plot).exists():
                motion_qc_plot = self.inputs.motion_qc_plot

        aif_curve_plot = None
        if isdefined(self.inputs.aif_curve_plot) and self.inputs.aif_curve_plot:
            if Path(self.inputs.aif_curve_plot).exists():
                aif_curve_plot = self.inputs.aif_curve_plot

        aif_location_plot = None
        if isdefined(self.inputs.aif_location_plot) and self.inputs.aif_location_plot:
            if Path(self.inputs.aif_location_plot).exists():
                aif_location_plot = self.inputs.aif_location_plot

        registration_filmstrip = None
        if isdefined(self.inputs.registration_filmstrip) and self.inputs.registration_filmstrip:
            if Path(self.inputs.registration_filmstrip).exists():
                registration_filmstrip = self.inputs.registration_filmstrip

        b1_qc_plot = None
        if isdefined(self.inputs.b1_qc_plot) and self.inputs.b1_qc_plot:
            if Path(self.inputs.b1_qc_plot).exists():
                b1_qc_plot = self.inputs.b1_qc_plot

        t1_qc_plot = None
        if isdefined(self.inputs.t1_qc_plot) and self.inputs.t1_qc_plot:
            if Path(self.inputs.t1_qc_plot).exists():
                t1_qc_plot = self.inputs.t1_qc_plot

        preprocessing_qc_plot = None
        if isdefined(self.inputs.preprocessing_qc_plot) and self.inputs.preprocessing_qc_plot:
            if Path(self.inputs.preprocessing_qc_plot).exists():
                preprocessing_qc_plot = self.inputs.preprocessing_qc_plot

        pk_modeling_qc_plot = None
        if isdefined(self.inputs.pk_modeling_qc_plot) and self.inputs.pk_modeling_qc_plot:
            if Path(self.inputs.pk_modeling_qc_plot).exists():
                pk_modeling_qc_plot = self.inputs.pk_modeling_qc_plot

        pk_param_maps_qc_plot = None
        if isdefined(self.inputs.pk_param_maps_qc_plot) and self.inputs.pk_param_maps_qc_plot:
            if Path(self.inputs.pk_param_maps_qc_plot).exists():
                pk_param_maps_qc_plot = self.inputs.pk_param_maps_qc_plot

        # HD-GLIO filmstrips
        seg_filmstrip_postcon = None
        if isdefined(self.inputs.seg_filmstrip_postcon) and self.inputs.seg_filmstrip_postcon:
            if Path(self.inputs.seg_filmstrip_postcon).exists():
                seg_filmstrip_postcon = self.inputs.seg_filmstrip_postcon

        seg_filmstrip_flair = None
        if isdefined(self.inputs.seg_filmstrip_flair) and self.inputs.seg_filmstrip_flair:
            if Path(self.inputs.seg_filmstrip_flair).exists():
                seg_filmstrip_flair = self.inputs.seg_filmstrip_flair

        nawm_filmstrip_flair = None
        if isdefined(self.inputs.nawm_filmstrip_flair) and self.inputs.nawm_filmstrip_flair:
            if Path(self.inputs.nawm_filmstrip_flair).exists():
                nawm_filmstrip_flair = self.inputs.nawm_filmstrip_flair

        # Anatomical filmstrips
        brain_mask_filmstrip = None
        if isdefined(self.inputs.brain_mask_filmstrip) and self.inputs.brain_mask_filmstrip:
            if Path(self.inputs.brain_mask_filmstrip).exists():
                brain_mask_filmstrip = self.inputs.brain_mask_filmstrip

        tissue_seg_filmstrip = None
        if isdefined(self.inputs.tissue_seg_filmstrip) and self.inputs.tissue_seg_filmstrip:
            if Path(self.inputs.tissue_seg_filmstrip).exists():
                tissue_seg_filmstrip = self.inputs.tissue_seg_filmstrip

        acpc_filmstrip = None
        if isdefined(self.inputs.acpc_filmstrip) and self.inputs.acpc_filmstrip:
            if Path(self.inputs.acpc_filmstrip).exists():
                acpc_filmstrip = self.inputs.acpc_filmstrip

        coreg_filmstrip_postcon = None
        if isdefined(self.inputs.coreg_filmstrip_postcon) and self.inputs.coreg_filmstrip_postcon:
            if Path(self.inputs.coreg_filmstrip_postcon).exists():
                coreg_filmstrip_postcon = self.inputs.coreg_filmstrip_postcon

        coreg_filmstrip_t2w = None
        if isdefined(self.inputs.coreg_filmstrip_t2w) and self.inputs.coreg_filmstrip_t2w:
            if Path(self.inputs.coreg_filmstrip_t2w).exists():
                coreg_filmstrip_t2w = self.inputs.coreg_filmstrip_t2w

        coreg_filmstrip_flair = None
        if isdefined(self.inputs.coreg_filmstrip_flair) and self.inputs.coreg_filmstrip_flair:
            if Path(self.inputs.coreg_filmstrip_flair).exists():
                coreg_filmstrip_flair = self.inputs.coreg_filmstrip_flair

        coreg_filmstrip_pdw = None
        if isdefined(self.inputs.coreg_filmstrip_pdw) and self.inputs.coreg_filmstrip_pdw:
            if Path(self.inputs.coreg_filmstrip_pdw).exists():
                coreg_filmstrip_pdw = self.inputs.coreg_filmstrip_pdw

        # Generate the report
        self._report_file = generate_qc_report(
            subject_id=subject_id,
            session_id=session_id,
            pipeline=pipeline,
            qc_json_path=self.inputs.qc_json,
            output_path=str(output_path),
            motion_qc_plot=motion_qc_plot,
            aif_curve_plot=aif_curve_plot,
            aif_location_plot=aif_location_plot,
            registration_filmstrip=registration_filmstrip,
            b1_qc_plot=b1_qc_plot,
            t1_qc_plot=t1_qc_plot,
            brain_mask_filmstrip=brain_mask_filmstrip,
            tissue_seg_filmstrip=tissue_seg_filmstrip,
            acpc_filmstrip=acpc_filmstrip,
            coreg_filmstrip_postcon=coreg_filmstrip_postcon,
            coreg_filmstrip_t2w=coreg_filmstrip_t2w,
            coreg_filmstrip_flair=coreg_filmstrip_flair,
            coreg_filmstrip_pdw=coreg_filmstrip_pdw,
            seg_filmstrip_postcon=seg_filmstrip_postcon,
            seg_filmstrip_flair=seg_filmstrip_flair,
            nawm_filmstrip_flair=nawm_filmstrip_flair,
            preprocessing_qc_plot=preprocessing_qc_plot,
            pk_modeling_qc_plot=pk_modeling_qc_plot,
            pk_param_maps_qc_plot=pk_param_maps_qc_plot,
        )

        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["report_file"] = self._report_file
        return outputs
