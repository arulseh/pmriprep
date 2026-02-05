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

    # Bayesian CV and CI outputs (optional)
    ktrans_cv = traits.Str(desc="Ktrans coefficient of variation map path")
    ve_cv = traits.Str(desc="ve coefficient of variation map path")
    vp_cv = traits.Str(desc="vp coefficient of variation map path")
    ktrans_ci95_lower = traits.Str(desc="Ktrans 95% CI lower bound map path")
    ktrans_ci95_upper = traits.Str(desc="Ktrans 95% CI upper bound map path")
    ve_ci95_lower = traits.Str(desc="ve 95% CI lower bound map path")
    ve_ci95_upper = traits.Str(desc="ve 95% CI upper bound map path")
    vp_ci95_lower = traits.Str(desc="vp 95% CI lower bound map path")
    vp_ci95_upper = traits.Str(desc="vp 95% CI upper bound map path")

    # Model-free maps (optional)
    iauc60_map = traits.Str(desc="IAUC at 60s map path")
    iauc90_map = traits.Str(desc="IAUC at 90s map path")
    iauc120_map = traits.Str(desc="IAUC at 120s map path")
    peak_enhancement_map = traits.Str(desc="Peak enhancement map path")
    time_to_peak_map = traits.Str(desc="Time to peak map path")
    ser_map = traits.Str(desc="Signal enhancement ratio map path")

    # QC maps (optional)
    tsnr_map = traits.Str(desc="tSNR map path")
    qc_mask = traits.Str(desc="QC mask (good fit voxels) path")
    r2_map = traits.Str(desc="R² goodness-of-fit map path")
    quality_flag_map = traits.Str(desc="Quality flag map path")

    # Tortuosity-corrected output (optional)
    ktrans_tortuosity_corrected = traits.Str(desc="Ktrans with tortuosity correction map path")

    # Normalized maps (optional, requires NAWM mask)
    nktrans_map = traits.Either(traits.Str, None, desc="nKtrans (NAWM-normalized Ktrans) map path")
    nvp_map = traits.Either(traits.Str, None, desc="nvp (NAWM-normalized vp) map path")
    niauc60_map = traits.Either(traits.Str, None, desc="nIAUC60 (NAWM-normalized IAUC at 60s) map path")

    # Smoothed maps (optional, dict of name->path)
    smooth_pk_maps = traits.Dict(desc="Smoothed PK maps dict")
    smooth_model_free_maps = traits.Dict(desc="Smoothed model-free maps dict")
    smooth_uncertainty_maps = traits.Dict(desc="Smoothed uncertainty maps dict (Std, CV, CI95)")
    smooth_tortuosity_maps = traits.Dict(desc="Smoothed tortuosity-corrected maps dict")

    # QC metrics (optional)
    qc_metrics = traits.Dict(desc="Merged QC metrics dictionary")

    # Structural underlays (registered to DCE space, optional)
    t1w_in_dce = traits.Str(desc="CE-T1w registered to DCE space NIfTI path")
    flair_in_dce = traits.Str(desc="FLAIR registered to DCE space NIfTI path")

    # Lesion statistics (conditional on lesion mask, optional)
    lesion_stats_json = traits.Str(desc="Lesion statistics JSON file path")
    lesion_stats_txt = traits.Str(desc="Lesion statistics text summary path")
    lesion_mask_dce = traits.Str(desc="Lesion mask in DCE space path")

    # Anatomical masks in DCE space (optional)
    brain_mask = traits.Str(desc="Brain mask in DCE space path")
    nawm_mask = traits.Either(traits.Str, None, desc="NAWM mask in DCE space path")

    # Processing config for sidecar (optional)
    pk_model = traits.Str(desc="Pharmacokinetic model name")
    config_dict = traits.Dict(desc="DCE processing config as dictionary")

    # QC plots (optional)
    aif_curve_plot = traits.Str(desc="AIF curve QC plot PNG path")
    aif_location_plot = traits.Str(desc="AIF location overlay PNG path")
    registration_filmstrip = traits.Str(desc="Registration QC filmstrip PNG path")
    motion_qc_plot = traits.Str(desc="Motion QC plot PNG path")


class DCEDerivativeWriterOutputSpec(TraitedSpec):
    """Output specification for DCEDerivativeWriter."""

    out_dir = traits.Str(desc="Path to the output session directory")
    qc_json = traits.Str(desc="Path to the QC JSON file")


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
        # 3b. Coefficient of Variation maps (Bayesian inference only)
        # -----------------------------------------------------------
        cv_maps = {
            "Ktrans": ("ktrans_cv", "Ktrans coefficient of variation (std/mean)"),
            "ve": ("ve_cv", "ve coefficient of variation (std/mean)"),
            "vp": ("vp_cv", "vp coefficient of variation (std/mean)"),
        }
        for map_name, (attr, description) in cv_maps.items():
            if self._has_input(attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_{model_entity}_desc-cv_{map_name}"
                self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3c. 95% Credible Interval maps (Bayesian inference only)
        # -----------------------------------------------------------
        ci_maps = {
            "Ktrans": [
                ("ktrans_ci95_lower", "Ktrans 95% credible interval lower bound"),
                ("ktrans_ci95_upper", "Ktrans 95% credible interval upper bound"),
            ],
            "ve": [
                ("ve_ci95_lower", "ve 95% credible interval lower bound"),
                ("ve_ci95_upper", "ve 95% credible interval upper bound"),
            ],
            "vp": [
                ("vp_ci95_lower", "vp 95% credible interval lower bound"),
                ("vp_ci95_upper", "vp 95% credible interval upper bound"),
            ],
        }
        for map_name, bounds in ci_maps.items():
            for attr, description in bounds:
                if self._has_input(attr):
                    src = getattr(self.inputs, attr)
                    bound_type = "ci95lower" if "lower" in attr else "ci95upper"
                    fname = f"{prefix}_{spatial}_{model_entity}_desc-{bound_type}_{map_name}"
                    self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                    sidecar = {**base_sidecar, "Description": description}
                    self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3d. Model-free maps (IAUC, peak, TTP, SER)
        # -----------------------------------------------------------
        model_free_maps = {
            "IAUC60": ("iauc60_map", "Initial Area Under Curve at 60s (mM·min)"),
            "IAUC90": ("iauc90_map", "Initial Area Under Curve at 90s (mM·min)"),
            "IAUC120": ("iauc120_map", "Initial Area Under Curve at 120s (mM·min)"),
            "peak": ("peak_enhancement_map", "Peak enhancement percentage"),
            "TTP": ("time_to_peak_map", "Time to peak (seconds)"),
            "SER": ("ser_map", "Signal Enhancement Ratio (washout characterization)"),
        }
        for map_name, (attr, description) in model_free_maps.items():
            if self._has_input(attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-modelfree_{map_name}"
                self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3e. QC maps (tSNR, QC mask, R², quality flag)
        # -----------------------------------------------------------
        if self._has_input("tsnr_map"):
            fname = f"{prefix}_{spatial}_desc-tSNR_perfmap"
            self._copy_nifti(self.inputs.tsnr_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Temporal signal-to-noise ratio"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("qc_mask"):
            fname = f"{prefix}_{spatial}_desc-qc_mask"
            self._copy_nifti(self.inputs.qc_mask, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "QC mask (1=good fit, 0=unfitted/poor)"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("r2_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-R2_perfmap"
            self._copy_nifti(self.inputs.r2_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "R² goodness of fit"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("quality_flag_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-qualityflag_perfmap"
            self._copy_nifti(self.inputs.quality_flag_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "Quality flag map (0=unfitted, 1=good, 2=poor_convergence, 3=high_uncertainty)",
            }
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3f. Tortuosity-corrected Ktrans
        # -----------------------------------------------------------
        if self._has_input("ktrans_tortuosity_corrected"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-tortuositycorr_Ktrans"
            self._copy_nifti(self.inputs.ktrans_tortuosity_corrected, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "Ktrans corrected for tissue tortuosity using DWI-derived ADC",
            }
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3f2. Normalized maps (nKtrans, nvp, nIAUC60) - requires NAWM mask
        # -----------------------------------------------------------
        if self._has_input("nktrans_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_nKtrans"
            self._copy_nifti(self.inputs.nktrans_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "Ktrans normalized by NAWM (normal-appearing white matter) mean",
            }
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("nvp_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_nvp"
            self._copy_nifti(self.inputs.nvp_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "vp normalized by NAWM (normal-appearing white matter) mean",
            }
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("niauc60_map"):
            fname = f"{prefix}_{spatial}_desc-modelfree_nIAUC60"
            self._copy_nifti(self.inputs.niauc60_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "IAUC at 60s normalized by NAWM (normal-appearing white matter) mean",
            }
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3g. Smoothed maps (conditional on smoothing being applied)
        # -----------------------------------------------------------
        smooth_pk = self.inputs.smooth_pk_maps
        if isdefined(smooth_pk) and smooth_pk:
            for map_name, src_path in smooth_pk.items():
                if src_path and Path(src_path).exists():
                    # Extract FWHM from filename (e.g., "_fwhm3.nii.gz")
                    import re
                    fwhm_match = re.search(r"_fwhm(\d+)", str(src_path))
                    fwhm_str = f"fwhm{fwhm_match.group(1)}" if fwhm_match else "fwhmN"
                    fname = f"{prefix}_{spatial}_{model_entity}_desc-{fwhm_str}_{map_name}"
                    self._copy_nifti(src_path, perf_dir / f"{fname}.nii.gz")
                    sidecar = {
                        **base_sidecar,
                        "Description": f"Smoothed {map_name} map",
                    }
                    self._write_json(sidecar, perf_dir / f"{fname}.json")

        smooth_mf = self.inputs.smooth_model_free_maps
        if isdefined(smooth_mf) and smooth_mf:
            for map_name, src_path in smooth_mf.items():
                if src_path and Path(src_path).exists():
                    import re
                    fwhm_match = re.search(r"_fwhm(\d+)", str(src_path))
                    fwhm_str = f"fwhm{fwhm_match.group(1)}" if fwhm_match else "fwhmN"
                    fname = f"{prefix}_{spatial}_desc-{fwhm_str}-modelfree_{map_name}"
                    self._copy_nifti(src_path, perf_dir / f"{fname}.nii.gz")
                    sidecar = {
                        **base_sidecar,
                        "Description": f"Smoothed model-free {map_name} map",
                    }
                    self._write_json(sidecar, perf_dir / f"{fname}.json")

        # Smoothed uncertainty maps (Bayesian std, CV, CI95)
        smooth_uncertainty = self.inputs.smooth_uncertainty_maps
        if isdefined(smooth_uncertainty) and smooth_uncertainty:
            for map_name, src_path in smooth_uncertainty.items():
                if src_path and Path(src_path).exists():
                    import re
                    fwhm_match = re.search(r"_fwhm(\d+)", str(src_path))
                    fwhm_str = f"fwhm{fwhm_match.group(1)}" if fwhm_match else "fwhmN"
                    # Map names like "Ktrans_std" -> "Std_Ktrans", "Ktrans_cv" -> "CV_Ktrans"
                    # "Ktrans_ci95lower" -> "CI95lower_Ktrans"
                    if "_std" in map_name:
                        param = map_name.replace("_std", "")
                        desc_part = f"{model_entity}Std"
                        out_name = param
                    elif "_cv" in map_name:
                        param = map_name.replace("_cv", "")
                        desc_part = f"{model_entity}CV"
                        out_name = param
                    elif "_ci95lower" in map_name:
                        param = map_name.replace("_ci95lower", "")
                        desc_part = f"{model_entity}CI95lower"
                        out_name = param
                    elif "_ci95upper" in map_name:
                        param = map_name.replace("_ci95upper", "")
                        desc_part = f"{model_entity}CI95upper"
                        out_name = param
                    else:
                        desc_part = model_entity
                        out_name = map_name
                    fname = f"{prefix}_{spatial}_desc-{fwhm_str}-{desc_part}_{out_name}"
                    self._copy_nifti(src_path, perf_dir / f"{fname}.nii.gz")
                    sidecar = {
                        **base_sidecar,
                        "Description": f"Smoothed {map_name} uncertainty map",
                    }
                    self._write_json(sidecar, perf_dir / f"{fname}.json")

        # Smoothed tortuosity-corrected maps
        smooth_tort = self.inputs.smooth_tortuosity_maps
        if isdefined(smooth_tort) and smooth_tort:
            for map_name, src_path in smooth_tort.items():
                if src_path and Path(src_path).exists():
                    import re
                    fwhm_match = re.search(r"_fwhm(\d+)", str(src_path))
                    fwhm_str = f"fwhm{fwhm_match.group(1)}" if fwhm_match else "fwhmN"
                    # map_name is "Ktrans_tortuositycorr"
                    fname = f"{prefix}_{spatial}_desc-{fwhm_str}-tortuositycorr_Ktrans"
                    self._copy_nifti(src_path, perf_dir / f"{fname}.nii.gz")
                    sidecar = {
                        **base_sidecar,
                        "Description": "Smoothed Ktrans corrected for tissue tortuosity",
                    }
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
        # 4b. Anatomical masks in DCE space
        # -----------------------------------------------------------
        if self._has_input("brain_mask"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_desc-brain_mask"
            self._copy_nifti(self.inputs.brain_mask, anat_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Brain mask in DCE space"}
            self._write_json(sidecar, anat_dir / f"{fname}.json")

        if self._has_input("nawm_mask"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_desc-nawm_mask"
            self._copy_nifti(self.inputs.nawm_mask, anat_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Normal-appearing white matter mask in DCE space"}
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
        self._qc_json = ""
        if isdefined(qc_metrics) and qc_metrics:
            qc_dir.mkdir(parents=True, exist_ok=True)
            # Add NiPreps-style recommendation to QC metrics
            enhanced_qc = self._add_qc_recommendation(qc_metrics)
            qc_json_path = qc_dir / "dce_qc.json"
            self._write_json(enhanced_qc, qc_json_path)
            self._qc_json = str(qc_json_path)

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

        if self._has_input("motion_qc_plot"):
            qc_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(
                self.inputs.motion_qc_plot, qc_dir / "motion_qc.png"
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
        outputs["qc_json"] = getattr(self, "_qc_json", "")
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
