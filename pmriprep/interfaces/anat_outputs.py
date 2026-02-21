"""BIDS-Derivatives compliant anatomical output writer interface.

This module implements a custom Nipype interface that replaces DataSink
for writing anatomical pipeline outputs with BIDS-compliant filenames,
JSON sidecars, and QC files.
"""

import logging
from pathlib import Path
from typing import Any

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    isdefined,
    traits,
)

from pmriprep.utils.outputs import copy_derivative, has_input, write_json_sidecar

logger = logging.getLogger(__name__)


class AnatDerivativeWriterInputSpec(BaseInterfaceInputSpec):
    """Input specification for AnatDerivativeWriter."""

    # Configuration (mandatory)
    output_dir = traits.Str(mandatory=True, desc="Root output derivatives directory")
    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")
    bids_dir = traits.Str(mandatory=True, desc="BIDS dataset root directory")

    # Core anatomical outputs (mandatory)
    t1w_preproc = File(exists=True, mandatory=True, desc="Preprocessed T1w image")
    brain_mask = File(exists=True, mandatory=True, desc="Brain mask")
    acpc_xfm = File(exists=True, mandatory=True, desc="ACPC alignment transform")

    # Tissue probability maps (optional — unavailable in postcon-only mode
    # where FAST segmentation is skipped due to gadolinium contamination)
    wm_probseg = File(desc="WM probability map")
    gm_probseg = File(desc="GM probability map")
    csf_probseg = File(desc="CSF probability map")

    # Optional coregistered images
    t1w_postcon_coreg = File(desc="Post-contrast T1w coregistered to precon")
    t1w_postcon_to_t1w_xfm = File(desc="PostCon to T1w transform matrix")
    t2w_coreg = File(desc="T2w coregistered to T1w")
    t2w_to_t1w_xfm = File(desc="T2w to T1w transform matrix")
    flair_coreg = File(desc="FLAIR coregistered to T1w")
    flair_to_t1w_xfm = File(desc="FLAIR to T1w transform matrix")
    pdw_coreg = File(desc="PDw coregistered to T1w")
    pdw_to_t1w_xfm = File(desc="PDw to T1w transform matrix")
    t1w_subtraction = File(desc="T1w subtraction image (post - pre)")

    # Source paths for JSON sidecar Sources field
    source_t1w = File(desc="Original T1w source file path")
    source_t1w_postcon = File(desc="Original T1w post-contrast source path")
    source_t2w = File(desc="Original T2w source path")
    source_flair = File(desc="Original FLAIR source path")
    source_pdw = File(desc="Original PDw source path")

    # QC filmstrips (optional)
    brain_mask_filmstrip = traits.Str(desc="Brain mask QC filmstrip PNG path")
    tissue_seg_filmstrip = traits.Str(desc="Tissue segmentation QC filmstrip PNG path")
    acpc_filmstrip = traits.Str(desc="ACPC alignment QC filmstrip PNG path")
    coreg_filmstrip_postcon = traits.Str(desc="T1w postcon coregistration filmstrip PNG path")
    coreg_filmstrip_t2w = traits.Str(desc="T2w coregistration filmstrip PNG path")
    coreg_filmstrip_flair = traits.Str(desc="FLAIR coregistration filmstrip PNG path")
    coreg_filmstrip_pdw = traits.Str(desc="PDw coregistration filmstrip PNG path")

    # QC metrics
    qc_metrics = traits.Dict(desc="QC metrics dictionary from brain extraction")


class AnatDerivativeWriterOutputSpec(TraitedSpec):
    """Output specification for AnatDerivativeWriter."""

    out_dir = traits.Str(desc="Path to the output session directory")
    qc_json = traits.Str(desc="Path to the QC JSON file")


class AnatDerivativeWriter(BaseInterface):
    """Write anatomical pipeline outputs with BIDS-compliant naming.

    This interface replaces Nipype DataSink for anatomical outputs,
    producing:
    - BIDS-compliant filenames (e.g., sub-XX_ses-XX_space-anat_desc-preproc_T1w.nii.gz)
    - JSON sidecars paired with every NIfTI file
    - dataset_description.json at the derivative root
    - QC JSON file in the session directory

    The output directory structure follows::

        {output_dir}/pmriprep/
            dataset_description.json
            sub-{subject}/ses-{session}/
                anat/
                    sub-XX_ses-XX_space-anat_desc-preproc_T1w.nii.gz
                    sub-XX_ses-XX_space-anat_desc-preproc_T1w.json
                    sub-XX_ses-XX_space-anat_desc-brain_mask.nii.gz
                    from-anat_to-ACPC_mode-image_xfm.mat
                    ...
                sub-XX_ses-XX_desc-anatQC_quality.json
                figures/
                    ...
    """

    input_spec = AnatDerivativeWriterInputSpec
    output_spec = AnatDerivativeWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from pmriprep.utils.anat_outputs import (
            build_mask_sidecar,
            build_preproc_sidecar,
            build_probseg_sidecar,
            get_xfm_filename,
        )

        output_dir = Path(self.inputs.output_dir)
        subject = self.inputs.subject_id
        session = self.inputs.session_id
        bids_dir = self.inputs.bids_dir

        # Build prefix for filenames
        prefix = f"sub-{subject}"
        if session:
            prefix += f"_ses-{session}"

        # Create directory structure
        pipeline_dir = output_dir / "pmriprep"
        ses_dir = pipeline_dir / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        anat_dir = ses_dir / "anat"

        anat_dir.mkdir(parents=True, exist_ok=True)

        # Determine source T1w path for sidecars
        source_t1w = self._get_source_path("source_t1w")

        # ---------------------------------------------------------------
        # 1. Core NIfTI + JSON pairs
        # ---------------------------------------------------------------

        # T1w preprocessed
        copy_derivative(
            self.inputs.t1w_preproc,
            anat_dir / f"{prefix}_space-anat_desc-preproc_T1w.nii.gz",
        )
        if source_t1w:
            sidecar = build_preproc_sidecar(
                description="Preprocessed and brain-extracted T1w image in native space (pre-contrast)",
                source_path=source_t1w,
                bids_dir=bids_dir,
                skull_stripped=True,
            )
            write_json_sidecar(
                sidecar,
                anat_dir / f"{prefix}_space-anat_desc-preproc_T1w.json",
            )

        # Brain mask
        copy_derivative(
            self.inputs.brain_mask,
            anat_dir / f"{prefix}_space-anat_desc-brain_mask.nii.gz",
        )
        if source_t1w:
            sidecar = build_mask_sidecar(
                description="Brain mask from HD-BET in native anat space",
                source_path=source_t1w,
                bids_dir=bids_dir,
            )
            write_json_sidecar(
                sidecar,
                anat_dir / f"{prefix}_space-anat_desc-brain_mask.json",
            )

        # Tissue probability maps (optional — skipped in postcon-only mode)
        tissue_map = {
            "CSF": "csf_probseg",
            "GM": "gm_probseg",
            "WM": "wm_probseg",
        }
        for tissue_type, input_name in tissue_map.items():
            if not has_input(self.inputs,input_name):
                continue
            probseg_path = getattr(self.inputs, input_name)
            copy_derivative(
                probseg_path,
                anat_dir / f"{prefix}_space-anat_label-{tissue_type}_probseg.nii.gz",
            )
            if source_t1w:
                sidecar = build_probseg_sidecar(
                    tissue_type=tissue_type,
                    source_path=source_t1w,
                    bids_dir=bids_dir,
                )
                write_json_sidecar(
                    sidecar,
                    anat_dir / f"{prefix}_space-anat_label-{tissue_type}_probseg.json",
                )

        # ---------------------------------------------------------------
        # 2. Optional coregistered images
        # ---------------------------------------------------------------

        # Post-contrast T1w
        if has_input(self.inputs,"t1w_postcon_coreg"):
            copy_derivative(
                self.inputs.t1w_postcon_coreg,
                anat_dir / f"{prefix}_ce-GAD_space-anat_desc-preproc_T1w.nii.gz",
            )
            source_postcon = self._get_source_path("source_t1w_postcon")
            if source_postcon:
                sidecar = build_preproc_sidecar(
                    description="Preprocessed and brain-extracted post-contrast T1w image in native space",
                    source_path=source_postcon,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                write_json_sidecar(
                    sidecar,
                    anat_dir / f"{prefix}_ce-GAD_space-anat_desc-preproc_T1w.json",
                )

        # T2w
        if has_input(self.inputs,"t2w_coreg"):
            copy_derivative(
                self.inputs.t2w_coreg,
                anat_dir / f"{prefix}_space-anat_desc-preproc_T2w.nii.gz",
            )
            source_t2w = self._get_source_path("source_t2w")
            if source_t2w:
                sidecar = build_preproc_sidecar(
                    description="Preprocessed and brain-extracted T2w image in native anat space",
                    source_path=source_t2w,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                write_json_sidecar(
                    sidecar,
                    anat_dir / f"{prefix}_space-anat_desc-preproc_T2w.json",
                )

        # FLAIR
        if has_input(self.inputs,"flair_coreg"):
            copy_derivative(
                self.inputs.flair_coreg,
                anat_dir / f"{prefix}_space-anat_desc-preproc_FLAIR.nii.gz",
            )
            source_flair = self._get_source_path("source_flair")
            if source_flair:
                sidecar = build_preproc_sidecar(
                    description="Preprocessed and brain-extracted FLAIR image in native anat space",
                    source_path=source_flair,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                write_json_sidecar(
                    sidecar,
                    anat_dir / f"{prefix}_space-anat_desc-preproc_FLAIR.json",
                )

        # PDw
        if has_input(self.inputs,"pdw_coreg"):
            copy_derivative(
                self.inputs.pdw_coreg,
                anat_dir / f"{prefix}_space-anat_desc-preproc_PDw.nii.gz",
            )
            source_pdw = self._get_source_path("source_pdw")
            if source_pdw:
                sidecar = build_preproc_sidecar(
                    description="Preprocessed and brain-extracted proton density weighted image in native anat space",
                    source_path=source_pdw,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                write_json_sidecar(
                    sidecar,
                    anat_dir / f"{prefix}_space-anat_desc-preproc_PDw.json",
                )

        # T1w subtraction
        if has_input(self.inputs,"t1w_subtraction"):
            copy_derivative(
                self.inputs.t1w_subtraction,
                anat_dir / f"{prefix}_space-anat_desc-subtraction_T1w.nii.gz",
            )
            source_postcon = self._get_source_path("source_t1w_postcon")
            if source_postcon:
                sidecar = build_preproc_sidecar(
                    description="Post-contrast minus pre-contrast subtraction map",
                    source_path=source_postcon,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                write_json_sidecar(
                    sidecar,
                    anat_dir / f"{prefix}_space-anat_desc-subtraction_T1w.json",
                )

        # ---------------------------------------------------------------
        # 3. Transform files
        # ---------------------------------------------------------------

        # ACPC transform (always present)
        acpc_fname = get_xfm_filename(subject, session, "anat", "ACPC")
        copy_derivative(self.inputs.acpc_xfm, anat_dir / acpc_fname)

        # Optional transforms
        if has_input(self.inputs,"t1w_postcon_to_t1w_xfm"):
            fname = get_xfm_filename(subject, session, "T1wCE", "anat")
            copy_derivative(self.inputs.t1w_postcon_to_t1w_xfm, anat_dir / fname)

        if has_input(self.inputs,"t2w_to_t1w_xfm"):
            fname = get_xfm_filename(subject, session, "T2w", "anat")
            copy_derivative(self.inputs.t2w_to_t1w_xfm, anat_dir / fname)

        if has_input(self.inputs,"flair_to_t1w_xfm"):
            fname = get_xfm_filename(subject, session, "FLAIR", "anat")
            copy_derivative(self.inputs.flair_to_t1w_xfm, anat_dir / fname)

        if has_input(self.inputs,"pdw_to_t1w_xfm"):
            fname = get_xfm_filename(subject, session, "PDw", "anat")
            copy_derivative(self.inputs.pdw_to_t1w_xfm, anat_dir / fname)

        # ---------------------------------------------------------------
        # 4. Dataset description
        # ---------------------------------------------------------------
        from pmriprep.utils.outputs import write_dataset_description
        write_dataset_description(pipeline_dir, bids_dir)

        # ---------------------------------------------------------------
        # 5. QC JSON
        # ---------------------------------------------------------------
        # Collect processing warnings for missing data-dependent inputs
        _data_dependent_inputs = {
            "wm_probseg": "Tissue segmentation (WM/GM/CSF probability maps)",
            "t1w_postcon_coreg": "Post-contrast T1w",
            "flair_coreg": "FLAIR",
            "t2w_coreg": "T2w",
            "pdw_coreg": "PDw",
        }
        warnings: list[str] = []
        for attr, label in _data_dependent_inputs.items():
            if not has_input(self.inputs, attr):
                msg = f"{label} not available"
                logger.warning(msg)
                warnings.append(msg)

        qc_metrics = self.inputs.qc_metrics
        self._qc_json = ""
        if isdefined(qc_metrics) and qc_metrics:
            if warnings:
                qc_metrics = dict(qc_metrics)
                qc_metrics["warnings"] = warnings
            qc_json_path = ses_dir / f"{prefix}_desc-anatQC_quality.json"
            write_json_sidecar(qc_metrics, qc_json_path)
            self._qc_json = str(qc_json_path)

        # ---------------------------------------------------------------
        # 6. QC figures → figures/ with BIDS names
        # ---------------------------------------------------------------
        from pmriprep.utils.report_utils import figure_filename

        figures_dir = ses_dir / "figures"

        _fig_inputs = {
            "brain_mask": "brain_mask_filmstrip",
            "tissue_seg": "tissue_seg_filmstrip",
            "acpc": "acpc_filmstrip",
            "coreg_postcon": "coreg_filmstrip_postcon",
            "coreg_t2w": "coreg_filmstrip_t2w",
            "coreg_flair": "coreg_filmstrip_flair",
            "coreg_pdw": "coreg_filmstrip_pdw",
        }
        for fig_key, attr in _fig_inputs.items():
            if has_input(self.inputs,attr):
                figures_dir.mkdir(parents=True, exist_ok=True)
                fname = figure_filename(subject, session, fig_key, "anat")
                copy_derivative(getattr(self.inputs, attr), figures_dir / fname)

        self._out_dir = str(ses_dir)
        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        outputs = self.output_spec().get()  # type: ignore[misc]
        outputs["out_dir"] = getattr(self, "_out_dir", "")
        outputs["qc_json"] = getattr(self, "_qc_json", "")
        return outputs  # type: ignore[no-any-return]

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

    def _get_source_path(self, attr_name: str) -> str:
        """Get a source path from inputs, returning empty string if undefined."""
        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        val = getattr(self.inputs, attr_name, None)
        if val is None or not isdefined(val):
            return ""
        if isinstance(val, str) and val != "" and Path(val).exists():
            return val
        return ""

