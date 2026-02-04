"""BIDS-Derivatives compliant DCE perfusion output writer interface.

This module implements a custom Nipype interface for writing DCE pipeline
outputs with BIDS-compliant filenames, JSON sidecars, and QC files.
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
    isdefined,
)


class DCEDerivativeWriterInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCEDerivativeWriter."""

    # Configuration (mandatory)
    output_dir = traits.Str(mandatory=True, desc="Root output derivatives directory")
    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")
    bids_dir = traits.Str(mandatory=True, desc="BIDS dataset root directory")

    # Core pharmacokinetic parameter maps (mandatory)
    ktrans_map = File(exists=True, mandatory=True, desc="Ktrans (volume transfer constant) map")
    ve_map = File(exists=True, mandatory=True, desc="ve (EES volume fraction) map")
    kep_map = File(exists=True, mandatory=True, desc="kep (rate constant) map")

    # Optional PK parameter (only for extended Tofts/2CXM)
    vp_map = traits.Str(desc="vp (plasma volume fraction) map path")

    # T1/B1 mapping outputs (optional)
    t1_map = traits.Str(desc="T1 map path")
    b1_map = traits.Str(desc="B1+ map path")

    # Motion-corrected data (optional)
    dce_moco = traits.Str(desc="Motion-corrected 4D DCE path")
    dce_mean = traits.Str(desc="Mean DCE image path")
    motion_params = traits.Str(desc="Motion parameters file path")

    # AIF outputs (optional)
    aif_curve = traits.Str(desc="AIF curve text file path")
    aif_mask = traits.Str(desc="AIF detection mask path")

    # Registration (optional)
    dce_to_t1w_xfm = traits.Str(desc="DCE to T1w transform matrix path")

    # Uncertainty outputs (Bayesian, optional)
    ktrans_uncertainty = traits.Str(desc="Ktrans uncertainty map path")
    ve_uncertainty = traits.Str(desc="ve uncertainty map path")
    vp_uncertainty = traits.Str(desc="vp uncertainty map path")

    # QC metrics (optional)
    qc_metrics = traits.Dict(desc="Merged QC metrics dictionary")

    # Structural underlays (registered to DCE space, optional)
    t1w_in_dce = traits.Str(desc="CE-T1w registered to DCE space NIfTI path")
    flair_in_dce = traits.Str(desc="FLAIR registered to DCE space NIfTI path")

    # Lesion statistics (conditional on lesion mask, optional)
    lesion_stats_json = traits.Str(desc="Lesion statistics JSON file path")
    lesion_stats_txt = traits.Str(desc="Lesion statistics text summary path")
    lesion_mask_dce = traits.Str(desc="Lesion mask in DCE space path")

    # Processing config for sidecar (optional)
    pk_model = traits.Str(desc="Pharmacokinetic model name")
    config_dict = traits.Dict(desc="DCE processing config as dictionary")

    # QC plots (optional)
    aif_curve_plot = traits.Str(desc="AIF curve QC plot PNG path")
    aif_location_plot = traits.Str(desc="AIF location overlay PNG path")
    registration_filmstrip = traits.Str(desc="Registration QC filmstrip PNG path")


class DCEDerivativeWriterOutputSpec(TraitedSpec):
    """Output specification for DCEDerivativeWriter."""

    out_dir = traits.Str(desc="Path to the output session directory")


class DCEDerivativeWriter(BaseInterface):
    """Write DCE perfusion pipeline outputs with BIDS-compliant naming.

    Output directory structure::

        {output_dir}/dce_pipeline/
            dataset_description.json
            sub-{subject}/ses-{session}/
                perf/
                    sub-XX_ses-XX_space-DCE_res-X.XXxX.XX_model-{model}_Ktrans.nii.gz
                    sub-XX_ses-XX_space-DCE_res-X.XXxX.XX_model-{model}_Ktrans.json
                    ...
                anat/
                    sub-XX_ses-XX_space-DCE_res-X.XXxX.XX_T1map.nii.gz
                    ...
                xfm/
                    sub-XX_ses-XX_from-DCE_to-T1w_mode-image_xfm.mat
                qc/
                    dce_qc.json
                stats/
                    sub-XX_ses-XX_space-DCE_lesion-stats.json
    """

    input_spec = DCEDerivativeWriterInputSpec
    output_spec = DCEDerivativeWriterOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        import nibabel as nib  # type: ignore[import-untyped]

        output_dir = Path(self.inputs.output_dir)
        subject = self.inputs.subject_id
        session = self.inputs.session_id
        bids_dir = self.inputs.bids_dir

        # Build prefix
        prefix = f"sub-{subject}"
        if session:
            prefix += f"_ses-{session}"

        # Build resolution string from Ktrans map header (always present)
        ref_img = nib.load(self.inputs.ktrans_map)
        voxel_sizes = ref_img.header.get_zooms()[:3]  # type: ignore[attr-defined]
        res_str = "x".join(f"{v:.2f}" for v in voxel_sizes)

        # Get model name for filename - extract resolved model from qc_metrics if "auto"
        pk_model = self.inputs.pk_model
        if not isdefined(pk_model) or not pk_model or pk_model == "auto":
            # Extract resolved model from qc_metrics["pk_modeling"]["model"]
            qc_metrics = self.inputs.qc_metrics
            if (
                isdefined(qc_metrics)
                and qc_metrics
                and "pk_modeling" in qc_metrics
                and "model" in qc_metrics["pk_modeling"]
            ):
                pk_model = qc_metrics["pk_modeling"]["model"]
            else:
                pk_model = "unknown"

        # Build spatial entity string
        spatial = f"space-DCE_res-{res_str}"
        model_entity = f"model-{pk_model}"

        # Create directory structure
        pipeline_dir = output_dir / "dce_pipeline"
        ses_dir = pipeline_dir / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        perf_dir = ses_dir / "perf"
        anat_dir = ses_dir / "anat"
        xfm_dir = ses_dir / "xfm"
        qc_dir = ses_dir / "qc"
        stats_dir = ses_dir / "stats"

        perf_dir.mkdir(parents=True, exist_ok=True)

        # Build base sidecar from config
        base_sidecar = self._build_base_sidecar(bids_dir)

        # -----------------------------------------------------------
        # 1. Core PK parameter maps (mandatory)
        # -----------------------------------------------------------
        core_maps = {
            "Ktrans": ("ktrans_map", "Volume transfer constant (min⁻¹)"),
            "ve": ("ve_map", "Extravascular extracellular space volume fraction"),
            "kep": ("kep_map", "Efflux rate constant (min⁻¹)"),
        }
        for map_name, (attr, description) in core_maps.items():
            src = getattr(self.inputs, attr)
            fname = f"{prefix}_{spatial}_{model_entity}_{map_name}"
            self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": description}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 2. Optional vp map (extended Tofts/2CXM only)
        # -----------------------------------------------------------
        if self._has_input("vp_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_vp"
            self._copy_nifti(self.inputs.vp_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Plasma volume fraction"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3. Uncertainty maps (Bayesian inference only)
        # -----------------------------------------------------------
        uncertainty_maps = {
            "Ktrans": ("ktrans_uncertainty", "Ktrans uncertainty (posterior std)"),
            "ve": ("ve_uncertainty", "ve uncertainty (posterior std)"),
            "vp": ("vp_uncertainty", "vp uncertainty (posterior std)"),
        }
        for map_name, (attr, description) in uncertainty_maps.items():
            if self._has_input(attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_{model_entity}_desc-uncertainty_{map_name}"
                self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 4. T1/B1 mapping outputs
        # -----------------------------------------------------------
        if self._has_input("t1_map"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_T1map"
            self._copy_nifti(self.inputs.t1_map, anat_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "T1 relaxation time map (ms)"}
            self._write_json(sidecar, anat_dir / f"{fname}.json")

        if self._has_input("b1_map"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_B1map"
            self._copy_nifti(self.inputs.b1_map, anat_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "B1+ transmit field map (ratio)"}
            self._write_json(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 5. Motion-corrected DCE and reference
        # -----------------------------------------------------------
        if self._has_input("dce_moco"):
            fname = f"{prefix}_{spatial}_desc-mc_dce"
            self._copy_nifti(self.inputs.dce_moco, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Motion-corrected 4D DCE time series"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("dce_mean"):
            fname = f"{prefix}_{spatial}_desc-ref_dce"
            self._copy_nifti(self.inputs.dce_mean, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Mean DCE reference image"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("motion_params"):
            self._copy_file(
                self.inputs.motion_params,
                perf_dir / f"{prefix}_desc-mc_motion.par",
            )

        # -----------------------------------------------------------
        # 6. AIF outputs
        # -----------------------------------------------------------
        if self._has_input("aif_curve"):
            self._copy_file(
                self.inputs.aif_curve,
                qc_dir / f"{prefix}_desc-aif_timeseries.txt",
            )
            qc_dir.mkdir(parents=True, exist_ok=True)

        if self._has_input("aif_mask"):
            self._copy_nifti(
                self.inputs.aif_mask,
                perf_dir / f"{prefix}_{spatial}_desc-aif_mask.nii.gz",
            )

        # -----------------------------------------------------------
        # 7. Registration transform
        # -----------------------------------------------------------
        if self._has_input("dce_to_t1w_xfm"):
            xfm_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(
                self.inputs.dce_to_t1w_xfm,
                xfm_dir / f"{prefix}_from-DCE_to-T1w_mode-image_xfm.mat",
            )

        # -----------------------------------------------------------
        # 8. Structural underlays (in DCE space)
        # -----------------------------------------------------------
        if self._has_input("t1w_in_dce"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_ce-GAD_T1w"
            self._copy_nifti(self.inputs.t1w_in_dce, anat_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "CE-T1w registered to DCE space"}
            self._write_json(sidecar, anat_dir / f"{fname}.json")

        if self._has_input("flair_in_dce"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_FLAIR"
            self._copy_nifti(self.inputs.flair_in_dce, anat_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "FLAIR registered to DCE space"}
            self._write_json(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 9. Lesion statistics (conditional)
        # -----------------------------------------------------------
        if self._has_input("lesion_stats_json"):
            stats_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(
                self.inputs.lesion_stats_json,
                stats_dir / f"{prefix}_{spatial}_lesion-stats.json",
            )

        if self._has_input("lesion_stats_txt"):
            stats_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(
                self.inputs.lesion_stats_txt,
                stats_dir / f"{prefix}_{spatial}_lesion-stats.txt",
            )

        if self._has_input("lesion_mask_dce"):
            roi_dir = ses_dir / "roi"
            roi_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_desc-lesion_mask"
            self._copy_nifti(self.inputs.lesion_mask_dce, roi_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Lesion mask registered to DCE space"}
            self._write_json(sidecar, roi_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 10. QC outputs
        # -----------------------------------------------------------
        qc_metrics = self.inputs.qc_metrics
        if isdefined(qc_metrics) and qc_metrics:
            qc_dir.mkdir(parents=True, exist_ok=True)
            # Add NiPreps-style recommendation to QC metrics
            enhanced_qc = self._add_qc_recommendation(qc_metrics)
            self._write_json(enhanced_qc, qc_dir / "dce_qc.json")

        # Copy QC plots
        if self._has_input("aif_curve_plot"):
            qc_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(self.inputs.aif_curve_plot, qc_dir / "aif_curve_qc.png")
        if self._has_input("aif_location_plot"):
            qc_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(self.inputs.aif_location_plot, qc_dir / "aif_location_qc.png")
        if self._has_input("registration_filmstrip"):
            qc_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(
                self.inputs.registration_filmstrip, qc_dir / "registration_qc_filmstrip.png"
            )

        # -----------------------------------------------------------
        # 11. Dataset description
        # -----------------------------------------------------------
        self._write_dataset_description(pipeline_dir)

        self._out_dir = str(ses_dir)
        return runtime

    def _list_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
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
        if not isdefined(val):
            return False
        if isinstance(val, str):
            if val in ("", "None", "."):
                return False
            return Path(val).exists()
        return False

    def _build_base_sidecar(self, bids_dir: str) -> dict[str, Any]:
        """Build base JSON sidecar with processing metadata."""
        sidecar: dict[str, Any] = {
            "Pipeline": "PMRIPrep DCE",
            "Space": "DCE (native perfusion space)",
        }

        # Add model info
        pk_model = self.inputs.pk_model
        if isdefined(pk_model) and pk_model:
            sidecar["PharmacokineticModel"] = pk_model

        # Add config summary
        config_dict = self.inputs.config_dict
        if isdefined(config_dict) and config_dict:
            sidecar["PharmacokineticModel"] = config_dict.get("model", "unknown")
            sidecar["BayesianInference"] = config_dict.get("bayesian", "none")

        return sidecar

    @staticmethod
    def _copy_file(src: str, dst: Path) -> None:
        """Copy a file to the destination path."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

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
    def _add_qc_recommendation(qc_metrics: dict[str, Any]) -> dict[str, Any]:
        """Add NiPreps-style QC recommendation to metrics.

        Evaluates the QC metrics and adds an overall recommendation
        for data inclusion/exclusion, following NiPreps conventions.

        Args:
            qc_metrics: Original QC metrics dictionary.

        Returns:
            Enhanced QC metrics with 'recommendation' field added.
        """
        from pmriprep.utils.qc_standards import (
            QCAction,
            QCRecommendation,
            aggregate_qc_recommendations,
        )

        recommendations: list[QCRecommendation] = []
        enhanced = dict(qc_metrics)

        # Evaluate AIF quality if available
        aif_qc = qc_metrics.get("aif", {})
        quality_score = aif_qc.get("quality_score")
        if quality_score is not None:
            if quality_score >= 0.7:
                aif_rec = QCRecommendation(QCAction.INCLUDE, f"AIF quality good ({quality_score:.2f})", 0)
            elif quality_score >= 0.4:
                aif_rec = QCRecommendation(QCAction.REVIEW, f"AIF quality marginal ({quality_score:.2f})", 1)
            else:
                aif_rec = QCRecommendation(QCAction.EXCLUDE, f"AIF quality poor ({quality_score:.2f})", 3)
            recommendations.append(aif_rec)

        # Evaluate PK model fit if available
        pk_qc = qc_metrics.get("pk_modeling", {})
        fit_residual = pk_qc.get("median_residual")
        if fit_residual is not None:
            if fit_residual < 0.1:
                fit_rec = QCRecommendation(QCAction.INCLUDE, f"PK fit residual acceptable ({fit_residual:.3f})", 0)
            elif fit_residual < 0.2:
                fit_rec = QCRecommendation(QCAction.REVIEW, f"PK fit residual marginal ({fit_residual:.3f})", 1)
            else:
                fit_rec = QCRecommendation(QCAction.EXCLUDE, f"PK fit residual high ({fit_residual:.3f})", 2)
            recommendations.append(fit_rec)

        # Evaluate physiological plausibility of Ktrans
        ktrans_median = pk_qc.get("ktrans_median")
        if ktrans_median is not None:
            # Typical brain Ktrans: 0.001-0.01 min^-1 (healthy), up to 0.5 min^-1 (tumor)
            if 0.0001 < ktrans_median < 1.0:
                kt_rec = QCRecommendation(QCAction.INCLUDE, f"Ktrans physiologically plausible ({ktrans_median:.4f})", 0)
            else:
                kt_rec = QCRecommendation(QCAction.REVIEW, f"Ktrans outside typical range ({ktrans_median:.4f})", 1)
            recommendations.append(kt_rec)

        # Aggregate recommendations
        overall = aggregate_qc_recommendations(recommendations)
        enhanced["recommendation"] = {
            "action": overall.action,
            "reason": overall.reason,
            "severity": overall.severity,
        }

        return enhanced

    @staticmethod
    def _write_dataset_description(pipeline_dir: Path) -> None:
        """Write dataset_description.json at the pipeline root."""
        import pmriprep

        desc_path = pipeline_dir / "dataset_description.json"
        if desc_path.exists():
            return

        pipeline_dir.mkdir(parents=True, exist_ok=True)
        description = {
            "Name": "PMRIPrep - DCE Perfusion Pipeline",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "pmriprep",
                    "Version": pmriprep.__version__,
                    "Description": "DCE perfusion processing including pharmacokinetic modeling",
                }
            ],
            "License": "CC-BY-4.0",
        }
        with open(desc_path, "w") as f:
            json.dump(description, f, indent=2)
