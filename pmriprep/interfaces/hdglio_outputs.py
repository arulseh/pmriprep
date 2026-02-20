"""BIDS-Derivatives compliant HD-GLIO output writer interface.

This module implements a custom Nipype interface that writes HD-GLIO
tumor segmentation outputs with BIDS-compliant filenames, JSON sidecars,
and QC files. Outputs are split between hdglio_pipeline/ (segmentation
outputs) and anat_pipeline/ (lesion and NAWM masks for downstream use).
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


class HDGLIODerivativeWriterInputSpec(BaseInterfaceInputSpec):
    """Input specification for HDGLIODerivativeWriter."""

    # Configuration (mandatory)
    output_dir = traits.Str(mandatory=True, desc="Root output derivatives directory")
    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")
    bids_dir = traits.Str(mandatory=True, desc="BIDS dataset root directory")

    # HD-GLIO segmentation outputs (mandatory)
    segmentation = File(
        exists=True, mandatory=True, desc="HD-GLIO multi-class segmentation label map"
    )
    enhancing_mask = File(
        exists=True, mandatory=True, desc="Binary enhancing tumor mask (label 2)"
    )
    non_enhancing_mask = File(
        exists=True, mandatory=True, desc="Binary non-enhancing tumor mask (label 1)"
    )
    whole_tumor_mask = File(
        exists=True, mandatory=True, desc="Binary whole tumor mask (labels 1+2)"
    )

    # Downstream masks (mandatory)
    dilated_lesion_mask = File(
        exists=True, mandatory=True, desc="Dilated lesion mask for registration"
    )
    nawm_mask = File(
        exists=True, mandatory=True, desc="Contralateral NAWM mask for CBV normalization"
    )

    # QC filmstrips (optional)
    seg_filmstrip_postcon = traits.Str(desc="Tumor segmentation on T1w postcon filmstrip PNG path")
    seg_filmstrip_flair = traits.Str(desc="Tumor segmentation on FLAIR filmstrip PNG path")
    nawm_filmstrip_flair = traits.Str(desc="NAWM ROI on FLAIR filmstrip PNG path")

    # QC metrics
    qc_metrics = traits.Dict(desc="Combined QC metrics from all HD-GLIO steps")


class HDGLIODerivativeWriterOutputSpec(TraitedSpec):
    """Output specification for HDGLIODerivativeWriter."""

    out_dir = traits.Str(desc="Path to the hdglio_pipeline output session directory")
    qc_json = traits.Str(desc="Path to the hdglio_qc.json QC metrics file")
    lesion_stats_json = traits.Str(desc="Path to the lesion-stats.json file")


class HDGLIODerivativeWriter(BaseInterface):
    """Write HD-GLIO pipeline outputs with BIDS-compliant naming.

    This interface writes HD-GLIO tumor segmentation outputs to BIDS
    derivative directories, producing:

    - BIDS-compliant filenames for segmentation maps and binary masks
    - JSON sidecars paired with every NIfTI file
    - Lesion and NAWM masks written to anat_pipeline/ for downstream use
    - Volume statistics text file
    - dataset_description.json at the pipeline root
    - QC JSON files in a qc/ subdirectory

    The output directory structure follows::

        {output_dir}/hdglio_pipeline/
            dataset_description.json
            sub-{subject}/ses-{session}/
                anat/
                    sub-XX_ses-XX_space-anat_desc-hdglio_dseg.nii.gz
                    sub-XX_ses-XX_space-anat_desc-hdglio_dseg.json
                    sub-XX_ses-XX_space-anat_desc-enhancing_mask.nii.gz
                    sub-XX_ses-XX_space-anat_desc-nonenhancing_mask.nii.gz
                    sub-XX_ses-XX_space-anat_desc-wholetumor_mask.nii.gz
                    sub-XX_ses-XX_desc-hdglio_volumes.txt
                qc/
                    hdglio_qc.json

        {output_dir}/anat_pipeline/
            sub-{subject}/ses-{session}/
                anat/
                    sub-XX_ses-XX_space-anat_desc-lesion_mask.nii.gz
                    sub-XX_ses-XX_space-anat_desc-lesiondilated15mm_mask.nii.gz
                    sub-XX_ses-XX_space-anat_desc-nawm_mask.nii.gz
    """

    input_spec = HDGLIODerivativeWriterInputSpec
    output_spec = HDGLIODerivativeWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        output_dir = Path(self.inputs.output_dir)
        subject = self.inputs.subject_id
        session = self.inputs.session_id

        # Build prefix for filenames
        prefix = f"sub-{subject}"
        if session:
            prefix += f"_ses-{session}"

        # --- Create directory structure ---
        hdglio_dir = output_dir / "hdglio_pipeline"
        hdglio_ses_dir = hdglio_dir / f"sub-{subject}"
        if session:
            hdglio_ses_dir = hdglio_ses_dir / f"ses-{session}"
        hdglio_anat_dir = hdglio_ses_dir / "anat"
        hdglio_qc_dir = hdglio_ses_dir / "qc"

        anat_dir = output_dir / "anat_pipeline"
        anat_ses_dir = anat_dir / f"sub-{subject}"
        if session:
            anat_ses_dir = anat_ses_dir / f"ses-{session}"
        anat_anat_dir = anat_ses_dir / "anat"

        hdglio_anat_dir.mkdir(parents=True, exist_ok=True)
        hdglio_qc_dir.mkdir(parents=True, exist_ok=True)
        anat_anat_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------
        # 1. HD-GLIO segmentation outputs → hdglio_pipeline/
        # ---------------------------------------------------------------

        # Segmentation label map (dseg)
        self._copy_nifti(
            self.inputs.segmentation,
            hdglio_anat_dir / f"{prefix}_space-anat_desc-hdglio_dseg.nii.gz",
        )
        self._write_json(
            {
                "Description": "HD-GLIO deep learning tumor segmentation",
                "Manual": False,
                "SegmentationLabels": {
                    "0": "Background",
                    "1": "Non-enhancing tumor",
                    "2": "Enhancing tumor",
                },
                "Method": "HD-GLIO (Isensee et al.)",
            },
            hdglio_anat_dir / f"{prefix}_space-anat_desc-hdglio_dseg.json",
        )

        # Enhancing tumor mask
        self._copy_nifti(
            self.inputs.enhancing_mask,
            hdglio_anat_dir / f"{prefix}_space-anat_desc-enhancing_mask.nii.gz",
        )
        self._write_json(
            {
                "Description": "Binary mask of enhancing tumor (HD-GLIO label 2)",
                "Type": "Brain",
                "Method": "Derived from HD-GLIO segmentation",
            },
            hdglio_anat_dir / f"{prefix}_space-anat_desc-enhancing_mask.json",
        )

        # Non-enhancing tumor mask
        self._copy_nifti(
            self.inputs.non_enhancing_mask,
            hdglio_anat_dir / f"{prefix}_space-anat_desc-nonenhancing_mask.nii.gz",
        )
        self._write_json(
            {
                "Description": "Binary mask of non-enhancing tumor (HD-GLIO label 1)",
                "Type": "Brain",
                "Method": "Derived from HD-GLIO segmentation",
            },
            hdglio_anat_dir / f"{prefix}_space-anat_desc-nonenhancing_mask.json",
        )

        # Whole tumor mask
        self._copy_nifti(
            self.inputs.whole_tumor_mask,
            hdglio_anat_dir / f"{prefix}_space-anat_desc-wholetumor_mask.nii.gz",
        )
        self._write_json(
            {
                "Description": "Binary mask of whole tumor (HD-GLIO labels 1+2)",
                "Type": "Brain",
                "Method": "Derived from HD-GLIO segmentation",
            },
            hdglio_anat_dir / f"{prefix}_space-anat_desc-wholetumor_mask.json",
        )

        # ---------------------------------------------------------------
        # 2. Downstream masks → anat_pipeline/
        # ---------------------------------------------------------------

        # Lesion mask (copy of whole tumor mask for downstream use)
        self._copy_nifti(
            self.inputs.whole_tumor_mask,
            anat_anat_dir / f"{prefix}_space-anat_desc-lesion_mask.nii.gz",
        )
        self._write_json(
            {
                "Description": "Lesion mask from HD-GLIO whole tumor segmentation",
                "Type": "Brain",
                "Method": "HD-GLIO segmentation (labels 1+2)",
            },
            anat_anat_dir / f"{prefix}_space-anat_desc-lesion_mask.json",
        )

        # Dilated lesion mask
        self._copy_nifti(
            self.inputs.dilated_lesion_mask,
            anat_anat_dir / f"{prefix}_space-anat_desc-lesiondilated15mm_mask.nii.gz",
        )
        self._write_json(
            {
                "Description": "Lesion mask dilated by 15mm for registration cost function weighting",
                "Type": "Brain",
                "Method": "Spherical dilation of HD-GLIO whole tumor mask",
                "DilationRadiusMM": 15.0,
            },
            anat_anat_dir / f"{prefix}_space-anat_desc-lesiondilated15mm_mask.json",
        )

        # NAWM mask
        self._copy_nifti(
            self.inputs.nawm_mask,
            anat_anat_dir / f"{prefix}_space-anat_desc-nawm_mask.nii.gz",
        )
        self._write_json(
            {
                "Description": "Contralateral normal-appearing white matter mask for CBV normalization",
                "Type": "Brain",
                "Method": "Contralateral WM thresholded at p>=0.9, excluding tumor margin",
            },
            anat_anat_dir / f"{prefix}_space-anat_desc-nawm_mask.json",
        )

        # ---------------------------------------------------------------
        # 3. Volume statistics text file
        # ---------------------------------------------------------------
        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        qc_metrics = self.inputs.qc_metrics
        if isdefined(qc_metrics) and qc_metrics:
            volumes_path = (
                hdglio_anat_dir / f"{prefix}_desc-hdglio_volumes.txt"
            )
            self._write_volumes_txt(qc_metrics, volumes_path)

        # ---------------------------------------------------------------
        # 3b. Lesion statistics JSON
        # ---------------------------------------------------------------
        stats_dir = hdglio_ses_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        lesion_stats_path = (
            stats_dir / f"{prefix}_space-anat_lesion-stats.json"
        )
        self._write_lesion_stats_json(
            segmentation_file=self.inputs.segmentation,
            subject=subject,
            session=session,
            output_path=lesion_stats_path,
        )
        self._lesion_stats_json = str(lesion_stats_path)

        # ---------------------------------------------------------------
        # 4. Dataset description
        # ---------------------------------------------------------------
        self._write_dataset_description(hdglio_dir)

        # ---------------------------------------------------------------
        # 4b. QC figures → figures/ with BIDS names
        # ---------------------------------------------------------------
        from pmriprep.utils.report_utils import figure_filename

        figures_dir = hdglio_ses_dir / "figures"

        _fig_inputs = {
            "seg_postcon": "seg_filmstrip_postcon",
            "seg_flair": "seg_filmstrip_flair",
            "nawm_flair": "nawm_filmstrip_flair",
        }
        for fig_key, attr in _fig_inputs.items():
            val = getattr(self.inputs, attr, None)
            if val is not None and isdefined(val) and val and Path(val).exists():
                figures_dir.mkdir(parents=True, exist_ok=True)
                fname = figure_filename(subject, session, fig_key, "hdglio")
                shutil.copy2(str(val), str(figures_dir / fname))

        # ---------------------------------------------------------------
        # 5. QC JSON
        # ---------------------------------------------------------------
        qc_json_path = hdglio_qc_dir / "hdglio_qc.json"
        if isdefined(qc_metrics) and qc_metrics:
            self._write_json(qc_metrics, qc_json_path)

        self._out_dir = str(hdglio_ses_dir)
        self._qc_json = str(qc_json_path)
        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_dir"] = getattr(self, "_out_dir", "")
        outputs["qc_json"] = getattr(self, "_qc_json", "")
        outputs["lesion_stats_json"] = getattr(self, "_lesion_stats_json", "")
        return outputs

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

    @staticmethod
    def _copy_nifti(src: str, dst: Path) -> None:
        """Copy a NIfTI file to the destination path."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    @staticmethod
    def _write_json(data: dict, path: Path) -> None:
        """Write a dictionary as a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _write_dataset_description(pipeline_dir: Path) -> None:
        """Write dataset_description.json at the hdglio_pipeline root."""
        import pmriprep

        desc_path = pipeline_dir / "dataset_description.json"
        if desc_path.exists():
            return

        pipeline_dir.mkdir(parents=True, exist_ok=True)
        description = {
            "Name": "PMRIPrep - HD-GLIO Tumor Segmentation Pipeline",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "pmriprep",
                    "Version": pmriprep.__version__,
                    "Description": (
                        "HD-GLIO deep learning tumor segmentation "
                        "with mask extraction and NAWM ROI creation"
                    ),
                }
            ],
            "HowToAcknowledge": (
                f"Please cite pmriprep (version {pmriprep.__version__}) "
                "and HD-GLIO (Isensee et al.) when using these outputs."
            ),
            "License": "CC-BY-4.0",
        }
        with open(desc_path, "w") as f:
            json.dump(description, f, indent=2)

    @staticmethod
    def _write_volumes_txt(qc_metrics: dict, path: Path) -> None:
        """Write tumor volume statistics to a text file.

        Parameters
        ----------
        qc_metrics : dict
            Combined QC metrics containing segmentation volume info.
        path : Path
            Output text file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract volume data from QC metrics (nested under validation)
        seg_val = qc_metrics.get("segmentation", {}).get("validation", {})
        total_vol = seg_val.get("total_tumor_volume_ml", "N/A")
        enhancing_vol = seg_val.get("enhancing_volume_ml", "N/A")
        non_enhancing_vol = seg_val.get("non_enhancing_volume_ml", "N/A")
        total_voxels = seg_val.get("total_tumor_voxels", "N/A")
        enhancing_voxels = seg_val.get("enhancing_voxels", "N/A")
        non_enhancing_voxels = seg_val.get("non_enhancing_voxels", "N/A")
        voxel_vol = seg_val.get("voxel_volume_mm3", "N/A")

        def _fmt_vol(vol_ml: object) -> str:
            """Format volume as 'X.XX mL (Y.YY mm³)' or 'N/A'."""
            if vol_ml == "N/A" or vol_ml is None:
                return "N/A"
            val = float(str(vol_ml))
            mm3 = val * 1000.0
            return f"{val:.3f} mL ({mm3:.2f} mm³)"

        # Compute enhancing fraction
        enhancing_frac = "N/A"
        if (
            enhancing_vol != "N/A"
            and total_vol != "N/A"
            and float(total_vol) > 0
        ):
            enhancing_frac = f"{float(enhancing_vol) / float(total_vol):.3f}"

        lines = [
            "HD-GLIO Tumor Volume Statistics",
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
            f"Voxel volume:           {voxel_vol} mm³",
            f"Enhancing fraction:     {enhancing_frac}",
        ]

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _write_lesion_stats_json(
        segmentation_file: str,
        subject: str,
        session: str,
        output_path: Path,
    ) -> None:
        """Write BIDS-compliant lesion statistics JSON from segmentation.

        Reads the HD-GLIO segmentation NIfTI directly and computes
        per-region volumes using ``compute_region_volume``.

        Parameters
        ----------
        segmentation_file : str
            Path to HD-GLIO multi-label segmentation NIfTI.
        subject : str
            Subject ID (without ``sub-`` prefix).
        session : str
            Session ID (without ``ses-`` prefix).
        output_path : Path
            Output JSON file path.
        """
        from datetime import datetime, timezone

        import nibabel as nib
        import numpy as np

        import pmriprep
        from pmriprep.utils.lesion_stats import compute_region_volume

        output_path.parent.mkdir(parents=True, exist_ok=True)

        img: nib.Nifti1Image = nib.load(segmentation_file)  # type: ignore[assignment]
        seg_data = np.asarray(img.dataobj, dtype=np.int16)
        zooms = img.header.get_zooms()[:3]
        voxel_sizes: tuple[float, float, float] = (
            float(zooms[0]),
            float(zooms[1]),
            float(zooms[2]),
        )

        # Compute per-region volumes
        enhancing_mask = seg_data == 2
        non_enhancing_mask = seg_data == 1
        whole_tumor_mask = seg_data > 0

        enh_count, enh_mm3, enh_ml = compute_region_volume(
            enhancing_mask, voxel_sizes
        )
        nenh_count, nenh_mm3, nenh_ml = compute_region_volume(
            non_enhancing_mask, voxel_sizes
        )
        wt_count, wt_mm3, wt_ml = compute_region_volume(
            whole_tumor_mask, voxel_sizes
        )

        # Enhancing fraction
        enhancing_fraction: float | None = None
        if wt_ml > 0:
            enhancing_fraction = round(enh_ml / wt_ml, 4)

        result: dict[str, object] = {
            "Description": "HD-GLIO tumor segmentation lesion volume statistics",
            "GeneratedBy": f"pmriprep {pmriprep.__version__}",
            "GeneratedAt": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
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
            "derived_metrics": {
                "enhancing_fraction": enhancing_fraction,
            },
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
