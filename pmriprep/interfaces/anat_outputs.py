"""BIDS-Derivatives compliant anatomical output writer interface.

This module implements a custom Nipype interface that replaces DataSink
for writing anatomical pipeline outputs with BIDS-compliant filenames,
JSON sidecars, and QC files.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)


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
    wm_probseg = File(exists=True, mandatory=True, desc="WM probability map")
    gm_probseg = File(exists=True, mandatory=True, desc="GM probability map")
    csf_probseg = File(exists=True, mandatory=True, desc="CSF probability map")
    acpc_xfm = File(exists=True, mandatory=True, desc="ACPC alignment transform")

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

    # QC metrics
    qc_metrics = traits.Dict(desc="QC metrics dictionary from brain extraction")


class AnatDerivativeWriterOutputSpec(TraitedSpec):
    """Output specification for AnatDerivativeWriter."""

    out_dir = traits.Str(desc="Path to the output session directory")


class AnatDerivativeWriter(BaseInterface):
    """Write anatomical pipeline outputs with BIDS-compliant naming.

    This interface replaces Nipype DataSink for anatomical outputs,
    producing:
    - BIDS-compliant filenames (e.g., sub-XX_ses-XX_space-T1w_desc-preproc_T1w.nii.gz)
    - JSON sidecars paired with every NIfTI file
    - dataset_description.json at the pipeline root
    - QC JSON files in a qc/ subdirectory

    The output directory structure follows::

        {output_dir}/anat_pipeline/
            dataset_description.json
            sub-{subject}/ses-{session}/
                anat/
                    sub-XX_ses-XX_space-T1w_desc-preproc_T1w.nii.gz
                    sub-XX_ses-XX_space-T1w_desc-preproc_T1w.json
                    sub-XX_ses-XX_space-T1w_desc-brain_mask.nii.gz
                    ...
                xfm/
                    from-T1w_to-ACPC_mode-image_xfm.mat
                    ...
                qc/
                    brain_extraction_qc.json
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
        pipeline_dir = output_dir / "anat_pipeline"
        ses_dir = pipeline_dir / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        anat_dir = ses_dir / "anat"
        xfm_dir = ses_dir / "xfm"
        qc_dir = ses_dir / "qc"

        anat_dir.mkdir(parents=True, exist_ok=True)
        xfm_dir.mkdir(parents=True, exist_ok=True)
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Determine source T1w path for sidecars
        source_t1w = self._get_source_path("source_t1w")

        # ---------------------------------------------------------------
        # 1. Core NIfTI + JSON pairs
        # ---------------------------------------------------------------

        # T1w preprocessed
        self._copy_nifti(
            self.inputs.t1w_preproc,
            anat_dir / f"{prefix}_space-T1w_desc-preproc_T1w.nii.gz",
        )
        if source_t1w:
            sidecar = build_preproc_sidecar(
                description="Preprocessed and brain-extracted T1w image in native space (pre-contrast)",
                source_path=source_t1w,
                bids_dir=bids_dir,
                skull_stripped=True,
            )
            self._write_json(
                sidecar,
                anat_dir / f"{prefix}_space-T1w_desc-preproc_T1w.json",
            )

        # Brain mask
        self._copy_nifti(
            self.inputs.brain_mask,
            anat_dir / f"{prefix}_space-T1w_desc-brain_mask.nii.gz",
        )
        if source_t1w:
            sidecar = build_mask_sidecar(
                description="Brain mask from HD-BET in native T1w space",
                source_path=source_t1w,
                bids_dir=bids_dir,
            )
            self._write_json(
                sidecar,
                anat_dir / f"{prefix}_space-T1w_desc-brain_mask.json",
            )

        # Tissue probability maps
        tissue_map = {
            "CSF": self.inputs.csf_probseg,
            "GM": self.inputs.gm_probseg,
            "WM": self.inputs.wm_probseg,
        }
        for tissue_type, probseg_path in tissue_map.items():
            self._copy_nifti(
                probseg_path,
                anat_dir / f"{prefix}_space-T1w_label-{tissue_type}_probseg.nii.gz",
            )
            if source_t1w:
                sidecar = build_probseg_sidecar(
                    tissue_type=tissue_type,
                    source_path=source_t1w,
                    bids_dir=bids_dir,
                )
                self._write_json(
                    sidecar,
                    anat_dir / f"{prefix}_space-T1w_label-{tissue_type}_probseg.json",
                )

        # ---------------------------------------------------------------
        # 2. Optional coregistered images
        # ---------------------------------------------------------------

        # Post-contrast T1w
        if self._has_input("t1w_postcon_coreg"):
            self._copy_nifti(
                self.inputs.t1w_postcon_coreg,
                anat_dir / f"{prefix}_ce-GAD_space-T1w_desc-preproc_T1w.nii.gz",
            )
            source_postcon = self._get_source_path("source_t1w_postcon")
            if source_postcon:
                sidecar = build_preproc_sidecar(
                    description="Preprocessed and brain-extracted post-contrast T1w image in native space",
                    source_path=source_postcon,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                self._write_json(
                    sidecar,
                    anat_dir / f"{prefix}_ce-GAD_space-T1w_desc-preproc_T1w.json",
                )

        # T2w
        if self._has_input("t2w_coreg"):
            self._copy_nifti(
                self.inputs.t2w_coreg,
                anat_dir / f"{prefix}_space-T1w_desc-preproc_T2w.nii.gz",
            )
            source_t2w = self._get_source_path("source_t2w")
            if source_t2w:
                sidecar = build_preproc_sidecar(
                    description="Preprocessed and brain-extracted T2w image in native T1w space",
                    source_path=source_t2w,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                self._write_json(
                    sidecar,
                    anat_dir / f"{prefix}_space-T1w_desc-preproc_T2w.json",
                )

        # FLAIR
        if self._has_input("flair_coreg"):
            self._copy_nifti(
                self.inputs.flair_coreg,
                anat_dir / f"{prefix}_space-T1w_desc-preproc_FLAIR.nii.gz",
            )
            source_flair = self._get_source_path("source_flair")
            if source_flair:
                sidecar = build_preproc_sidecar(
                    description="Preprocessed and brain-extracted FLAIR image in native T1w space",
                    source_path=source_flair,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                self._write_json(
                    sidecar,
                    anat_dir / f"{prefix}_space-T1w_desc-preproc_FLAIR.json",
                )

        # PDw
        if self._has_input("pdw_coreg"):
            self._copy_nifti(
                self.inputs.pdw_coreg,
                anat_dir / f"{prefix}_space-T1w_desc-preproc_PDw.nii.gz",
            )
            source_pdw = self._get_source_path("source_pdw")
            if source_pdw:
                sidecar = build_preproc_sidecar(
                    description="Preprocessed and brain-extracted proton density weighted image in native T1w space",
                    source_path=source_pdw,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                self._write_json(
                    sidecar,
                    anat_dir / f"{prefix}_space-T1w_desc-preproc_PDw.json",
                )

        # T1w subtraction
        if self._has_input("t1w_subtraction"):
            self._copy_nifti(
                self.inputs.t1w_subtraction,
                anat_dir / f"{prefix}_space-T1w_desc-subtraction_T1w.nii.gz",
            )
            source_postcon = self._get_source_path("source_t1w_postcon")
            if source_postcon:
                sidecar = build_preproc_sidecar(
                    description="Post-contrast minus pre-contrast subtraction map",
                    source_path=source_postcon,
                    bids_dir=bids_dir,
                    skull_stripped=True,
                )
                self._write_json(
                    sidecar,
                    anat_dir / f"{prefix}_space-T1w_desc-subtraction_T1w.json",
                )

        # ---------------------------------------------------------------
        # 3. Transform files
        # ---------------------------------------------------------------

        # ACPC transform (always present)
        acpc_fname = get_xfm_filename(subject, session, "T1w", "ACPC")
        self._copy_file(self.inputs.acpc_xfm, xfm_dir / acpc_fname)

        # Optional transforms
        if self._has_input("t1w_postcon_to_t1w_xfm"):
            fname = get_xfm_filename(subject, session, "T1wPostCon", "T1w")
            self._copy_file(self.inputs.t1w_postcon_to_t1w_xfm, xfm_dir / fname)

        if self._has_input("t2w_to_t1w_xfm"):
            fname = get_xfm_filename(subject, session, "T2w", "T1w")
            self._copy_file(self.inputs.t2w_to_t1w_xfm, xfm_dir / fname)

        if self._has_input("flair_to_t1w_xfm"):
            fname = get_xfm_filename(subject, session, "FLAIR", "T1w")
            self._copy_file(self.inputs.flair_to_t1w_xfm, xfm_dir / fname)

        if self._has_input("pdw_to_t1w_xfm"):
            fname = get_xfm_filename(subject, session, "PDw", "T1w")
            self._copy_file(self.inputs.pdw_to_t1w_xfm, xfm_dir / fname)

        # ---------------------------------------------------------------
        # 4. Dataset description
        # ---------------------------------------------------------------
        self._write_dataset_description(pipeline_dir)

        # ---------------------------------------------------------------
        # 5. QC JSON
        # ---------------------------------------------------------------
        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        qc_metrics = self.inputs.qc_metrics
        if isdefined(qc_metrics) and qc_metrics:
            self._write_json(qc_metrics, qc_dir / "brain_extraction_qc.json")

        self._out_dir = str(ses_dir)
        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        outputs = self.output_spec().get()  # type: ignore[misc]
        outputs["out_dir"] = getattr(self, "_out_dir", "")
        return outputs

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

    def _has_input(self, name: str) -> bool:
        """Check if an optional input is defined and points to an existing file."""
        val = getattr(self.inputs, name, None)
        if val is None:
            return False
        # Nipype traits use _Undefined for unset values
        try:
            from nipype.interfaces.base import Undefined  # type: ignore[import-untyped]
            if val is Undefined or val == Undefined:
                return False
        except ImportError:
            pass
        if isinstance(val, str) and val == "":
            return False
        if isinstance(val, str):
            return Path(val).exists()
        return False

    def _get_source_path(self, attr_name: str) -> str:
        """Get a source path from inputs, returning empty string if undefined."""
        val = getattr(self.inputs, attr_name, None)
        if val is None:
            return ""
        try:
            from nipype.interfaces.base import Undefined  # type: ignore[import-untyped]
            if val is Undefined or val == Undefined:
                return ""
        except ImportError:
            pass
        if isinstance(val, str) and val != "" and Path(val).exists():
            return val
        return ""

    @staticmethod
    def _copy_file(src: str, dst: Path) -> None:
        """Copy a file to the destination path."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    # Alias for readability in NIfTI-specific contexts
    _copy_nifti = _copy_file

    @staticmethod
    def _write_json(data: dict, path: Path) -> None:
        """Write a dictionary as a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _write_dataset_description(pipeline_dir: Path) -> None:
        """Write dataset_description.json at the pipeline root."""
        import pmriprep

        desc_path = pipeline_dir / "dataset_description.json"
        if desc_path.exists():
            return

        pipeline_dir.mkdir(parents=True, exist_ok=True)
        description = {
            "Name": "PMRIPrep - Anatomical Pipeline",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "pmriprep",
                    "Version": pmriprep.__version__,
                    "Description": "Anatomical preprocessing including brain extraction and segmentation",
                }
            ],
            "HowToAcknowledge": (
                f"Please cite pmriprep (version {pmriprep.__version__}) "
                "when using these outputs in publications."
            ),
            "License": "CC-BY-4.0",
        }
        with open(desc_path, "w") as f:
            json.dump(description, f, indent=2)
