"""BIDS-Derivatives compliant DSC perfusion output writer interface.

This module implements a custom Nipype interface for writing DSC pipeline
outputs with BIDS-compliant filenames, JSON sidecars, and QC files.
"""

import logging
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

from pmriprep.utils.outputs import copy_derivative, has_input, write_json_sidecar
from pmriprep.utils.provenance import get_bids_derivative_provenance

logger = logging.getLogger(__name__)


class DSCDerivativeWriterInputSpec(BaseInterfaceInputSpec):
    """Input specification for DSCDerivativeWriter."""

    # Configuration (mandatory)
    output_dir = traits.Str(mandatory=True, desc="Root output derivatives directory")
    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")
    bids_dir = traits.Str(mandatory=True, desc="BIDS dataset root directory")

    # Quantification-independent maps (mandatory)
    mtt_map = File(exists=True, mandatory=True, desc="MTT map")
    tmax_map = File(exists=True, mandatory=True, desc="Tmax map")
    psr_map = File(exists=True, mandatory=True, desc="PSR map")
    ph_map = File(exists=True, mandatory=True, desc="PH map")

    # Additional quantification-independent (optional — use Str for safe handling)
    ttp_map = traits.Either(traits.Str, None, desc="TTP map path")
    ttp_corrected_map = traits.Either(traits.Str, None, desc="AIF-corrected TTP map path")
    rttp_map = traits.Either(traits.Str, None, desc="NAWM-relative TTP map path")
    cnr_map = traits.Either(traits.Str, None, desc="CNR map path")
    auc_diff_map = traits.Either(traits.Str, None, desc="AUC diff (leakage contribution) map path")
    snr_map = traits.Either(traits.Str, None, desc="SNR (baseline) map path")
    tsnr_map = traits.Either(traits.Str, None, desc="tSNR (temporal) map path")

    # Relative maps (always produced)
    rel_cbf_map = traits.Either(traits.Str, None, desc="Relative CBF map path")
    rel_cbv_map = traits.Either(traits.Str, None, desc="Relative CBV map path")
    rel_auc_raw_map = traits.Either(traits.Str, None, desc="Relative AUC raw map path")
    rel_auc_corr_map = traits.Either(traits.Str, None, desc="Relative AUC corrected map path")

    # Normalized maps (optional)
    ncbv_map = traits.Either(traits.Str, None, desc="Normalized CBV map path")
    ncbf_map = traits.Either(traits.Str, None, desc="Normalized CBF map path")

    # Vascular model parameters (optional)
    alpha_map = traits.Either(traits.Str, None, desc="Alpha parameter map path (vascular only)")
    beta_map = traits.Either(traits.Str, None, desc="Beta parameter map path (vascular only)")
    cth_map = traits.Either(traits.Str, None, desc="CTH (capillary transit time heterogeneity) map path (vascular only)")

    # Bayesian uncertainty maps (optional)
    cbf_std_map = traits.Either(traits.Str, None, desc="CBF uncertainty (std) map path (Bayesian only)")
    cbv_std_map = traits.Either(traits.Str, None, desc="CBV uncertainty (std) map path (Bayesian only)")
    cbf_cv_map = traits.Either(traits.Str, None, desc="CBF coefficient of variation path (Bayesian only)")
    cbv_cv_map = traits.Either(traits.Str, None, desc="CBV coefficient of variation path (Bayesian only)")
    cbf_ci95_lower_map = traits.Either(traits.Str, None, desc="CBF 95% CI lower bound path (VI/MCMC only)")
    cbf_ci95_upper_map = traits.Either(traits.Str, None, desc="CBF 95% CI upper bound path (VI/MCMC only)")
    cbv_ci95_lower_map = traits.Either(traits.Str, None, desc="CBV 95% CI lower bound path (VI/MCMC only)")
    cbv_ci95_upper_map = traits.Either(traits.Str, None, desc="CBV 95% CI upper bound path (VI/MCMC only)")

    # Leakage correction (optional)
    k2_map = traits.Either(traits.Str, None, desc="K2 leakage correction map path")
    preload_qc = traits.Dict(desc="Preload/K2 QC metrics dictionary")

    # Motion-corrected data
    dsc_moco = traits.Either(traits.Str, None, desc="Motion-corrected 4D DSC path")
    dsc_mean = traits.Either(traits.Str, None, desc="Mean DSC image path")
    motion_params = traits.Either(traits.Str, None, desc="Motion parameters file path (.par)")

    # AIF outputs
    aif_curve = traits.Either(traits.Str, None, desc="AIF curve text file path")
    aif_mask = traits.Either(traits.Str, None, desc="AIF detection mask path")

    # Registration
    dsc_to_t1w_xfm = traits.Either(traits.Str, None, desc="DSC to T1w transform matrix path")
    registration_filmstrip = traits.Either(traits.Str, None, desc="Registration QC filmstrip PNG path")

    # QC metrics
    qc_metrics = traits.Dict(desc="Merged QC metrics dictionary")

    # Source DSC file for sidecar
    source_dsc = traits.Either(traits.Str, None, desc="Original raw DSC source file path")

    # Processing config for sidecar
    config_dict = traits.Dict(desc="DSC processing config as dictionary")

    # Smoothed map dicts (optional — each is a dict of name->filepath)
    smooth_hemodynamic = traits.Dict(desc="Smoothed hemodynamic maps (CBF, CBV, MTT, nCBF, nCBV)")
    smooth_auc = traits.Dict(desc="Smoothed AUC corrected maps (relative)")
    smooth_fwhm = traits.Float(0.0, usedefault=True, desc="Smoothing FWHM in mm")

    # Structural underlays (registered to DSC space)
    t1w_in_dsc = traits.Either(traits.Str, None, desc="T1w registered to DSC space NIfTI path")
    flair_in_dsc = traits.Either(traits.Str, None, desc="FLAIR registered to DSC space NIfTI path")
    t2w_in_dsc = traits.Either(traits.Str, None, desc="T2w registered to DSC space NIfTI path")
    t1w_to_dsc_xfm = traits.Either(traits.Str, None, desc="T1w-to-DSC transform matrix path")

    # AIF QC plots
    aif_curve_plot = traits.Either(traits.Str, None, desc="AIF curve QC plot PNG path")
    aif_location_plot = traits.Either(traits.Str, None, desc="AIF location overlay PNG path")

    # Brain mask filmstrip
    brain_mask_filmstrip = traits.Either(traits.Str, None, desc="Brain mask QC filmstrip PNG path")

    # Motion QC plot
    motion_qc_plot = traits.Either(traits.Str, None, desc="Motion QC plot PNG path")

    # DSC QC plots
    dsc_preprocessing_qc_plot = traits.Either(traits.Str, None, desc="DSC preprocessing QC plot PNG path")
    dsc_param_maps_qc_plot = traits.Either(traits.Str, None, desc="DSC parameter maps QC plot PNG path")
    dsc_fit_qc_plot = traits.Either(traits.Str, None, desc="DSC fit QC plot PNG path (parametric methods only)")
    dsc_leakage_qc_plot = traits.Either(traits.Str, None, desc="DSC leakage correction QC plot PNG path")

    # Masks in DSC space
    brain_mask_dsc = traits.Either(traits.Str, None, desc="Brain mask in DSC space")
    nawm_mask_dsc = traits.Either(traits.Str, None, desc="NAWM mask in DSC space")

    # Structural underlay labeling
    has_postcon_t1w = traits.Bool(False, usedefault=True,
                                  desc="Whether t1w_in_dsc is post-contrast CE-T1w")

    # Lesion statistics
    lesion_stats_json = traits.Either(traits.Str, None, desc="Lesion statistics JSON file path")
    lesion_mask_dsc = traits.Either(traits.Str, None, desc="Lesion mask in DSC space path")

    # ROI statistics (from user-supplied roi_mask, optional)
    roi_stats_json = traits.Either(traits.Str, None, desc="ROI statistics JSON file path")
    roi_mask_dsc = traits.Either(traits.Str, None, desc="ROI mask in DSC space path")


class DSCDerivativeWriterOutputSpec(TraitedSpec):
    """Output specification for DSCDerivativeWriter."""

    out_dir = traits.Str(desc="Path to the output session directory")
    qc_json = traits.Str(desc="Path to the QC JSON file")


class DSCDerivativeWriter(BaseInterface):
    """Write DSC perfusion pipeline outputs with BIDS-compliant naming.

    Output directory structure::

        {output_dir}/pmriprep/
            dataset_description.json
            sub-{subject}/ses-{session}/
                perf/
                    sub-XX_ses-XX_space-DSC_desc-{desc}_{suffix}.nii.gz
                    sub-XX_ses-XX_space-DSC_desc-{desc}_{suffix}.json
                    ...
                anat/
                    sub-XX_ses-XX_space-DSC_{T1w|FLAIR}.nii.gz
                    sub-XX_ses-XX_from-{src}_to-{dst}_mode-image_xfm.mat
                    sub-XX_ses-XX_space-DSC_desc-lesion_mask.nii.gz
                figures/
                    {BIDS figure files}
                sub-XX_ses-XX_desc-dscQC_quality.json
                sub-XX_ses-XX_space-DSC_desc-lesionStats.json
    """

    input_spec = DSCDerivativeWriterInputSpec
    output_spec = DSCDerivativeWriterOutputSpec

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

        # Load reference image for voxel size metadata (always present)
        ref_img = nib.load(self.inputs.mtt_map)
        voxel_sizes = ref_img.header.get_zooms()[:3]  # type: ignore[attr-defined]

        # Build spatial entity string (no res- entity)
        spatial = "space-DSC"

        # Create directory structure (unified output dir)
        pipeline_dir = output_dir / "pmriprep"
        ses_dir = pipeline_dir / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        perf_dir = ses_dir / "perf"
        anat_dir = ses_dir / "anat"

        perf_dir.mkdir(parents=True, exist_ok=True)
        anat_dir.mkdir(parents=True, exist_ok=True)

        # Build base sidecar from config
        base_sidecar = self._build_base_sidecar(bids_dir, voxel_sizes)

        # -----------------------------------------------------------
        # 1. Relative maps (ALWAYS produced)
        # -----------------------------------------------------------
        rel_maps = {
            "cbf": ("rel_cbf_map", "Cerebral blood flow (relative)", "dscRel"),
            "cbv": ("rel_cbv_map", "Cerebral blood volume (relative)", "dscRel"),
            "auc": ("rel_auc_raw_map", "AUC raw concentration (relative)", "dscRel"),
            "auc_corr": ("rel_auc_corr_map", "AUC corrected concentration (relative)", "dscRelCorrected"),
        }
        rel_units = {"cbf": "1/min", "cbv": "arbitrary"}
        for map_suffix, (attr, description, desc_val) in rel_maps.items():
            if has_input(self.inputs,attr):
                src = getattr(self.inputs, attr)
                # For auc_corr, the BIDS suffix is just "auc"
                bids_suffix = "auc" if map_suffix == "auc_corr" else map_suffix
                fname = f"{prefix}_{spatial}_desc-{desc_val}_{bids_suffix}"
                copy_derivative(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                if bids_suffix in rel_units:
                    sidecar["Units"] = rel_units[bids_suffix]
                write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 2. Core maps (desc-dsc)
        # -----------------------------------------------------------
        core_maps = {
            "mtt": ("mtt_map", "Mean transit time", "s"),
            "tmax": ("tmax_map", "Time to maximum of residue function", "s"),
            "psr": ("psr_map", "Percent signal recovery", "%"),
            "ph": ("ph_map", "Peak height", "arbitrary"),
        }
        for map_name, (attr, description, units) in core_maps.items():
            src = getattr(self.inputs, attr)
            fname = f"{prefix}_{spatial}_desc-dsc_{map_name}"
            copy_derivative(src, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": description, "Units": units}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # Additional optional maps with per-map desc values
        optional_maps: dict[str, tuple[str, str, str, str]] = {
            # key: (attr, description, desc_value, bids_suffix)
            "ttp": ("ttp_map", "Time to peak concentration", "dsc", "ttp"),
            "ttpCorrected": ("ttp_corrected_map",
                             "Time to peak, corrected for AIF bolus arrival delay. "
                             "Comparable across patients and centers.",
                             "dscAifCorrected", "ttp"),
            "rttp": ("rttp_map",
                     "Time to peak relative to contralateral NAWM median. "
                     "Positive values indicate delayed perfusion.",
                     "dscNawmRel", "ttp"),
            "cnr": ("cnr_map", "Contrast-to-noise ratio", "dsc", "cnr"),
            "auc_diff": ("auc_diff_map", "Leakage contribution (AUC_raw - AUC_corr)",
                         "dscLeakage", "auc"),
            "snr": ("snr_map", "Signal-to-noise ratio (baseline)", "dsc", "snr"),
            "tsnr": ("tsnr_map", "Temporal signal-to-noise ratio", "dsc", "tsnr"),
        }
        # Units for optional maps
        optional_units: dict[str, str] = {
            "cnr": "arbitrary",
            "snr": "arbitrary",
            "tsnr": "arbitrary",
        }
        # Units and reference points for timing maps
        timing_sidecars: dict[str, dict[str, str]] = {
            "ttp": {"Units": "seconds", "ReferencePoint": "start of acquisition (volume 0)"},
            "ttpCorrected": {"Units": "seconds", "ReferencePoint": "AIF bolus arrival time"},
            "rttp": {"Units": "seconds", "ReferencePoint": "contralateral NAWM median TTP"},
        }
        for map_key, (attr, description, desc_val, bids_suffix) in optional_maps.items():
            if has_input(self.inputs,attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-{desc_val}_{bids_suffix}"
                copy_derivative(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                if map_key in timing_sidecars:
                    sidecar.update(timing_sidecars[map_key])
                if map_key in optional_units:
                    sidecar["Units"] = optional_units[map_key]
                write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 4. Normalized maps (optional)
        # -----------------------------------------------------------
        if has_input(self.inputs,"ncbv_map"):
            fname = f"{prefix}_{spatial}_desc-dscNawmNorm_cbv"
            copy_derivative(self.inputs.ncbv_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Normalized CBV (NAWM-normalized)", "Units": "ratio"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"ncbf_map"):
            fname = f"{prefix}_{spatial}_desc-dscNawmNorm_cbf"
            copy_derivative(self.inputs.ncbf_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Normalized CBF (NAWM-normalized)", "Units": "ratio"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 5. Leakage correction map (optional)
        # -----------------------------------------------------------
        if has_input(self.inputs,"k2_map"):
            fname = f"{prefix}_{spatial}_desc-dsc_k2"
            copy_derivative(self.inputs.k2_map, perf_dir / f"{fname}.nii.gz")
            # Build K2 sidecar with method info from preload QC
            preload_qc = self.inputs.preload_qc if isdefined(self.inputs.preload_qc) else {}
            sidecar = {
                **base_sidecar,
                "Description": "K2 leakage correction map",
                "Units": "1/s",
                "LeakageCorrectionMethod": preload_qc.get("k2_method", "bsw"),
            }
            if preload_qc.get("kep_used") is not None:
                sidecar["KepUsed"] = preload_qc["kep_used"]
                sidecar["KepUnits"] = "1/s"
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 5b. Vascular model parameters (optional)
        # -----------------------------------------------------------
        for vmap, (attr, desc) in {
            "alpha": ("alpha_map", "Vascular model alpha parameter"),
            "beta": ("beta_map", "Vascular model beta parameter"),
            "cth": ("cth_map", "Capillary transit time heterogeneity"),
        }.items():
            if has_input(self.inputs,attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-dsc_{vmap}"
                copy_derivative(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": desc}
                write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 5c. Bayesian uncertainty maps (optional)
        # -----------------------------------------------------------
        bayesian_uncertainty_maps: dict[str, tuple[str, str, str, str]] = {
            # key: (attr, description, desc_value, bids_suffix)
            "cbf_std": ("cbf_std_map", "CBF uncertainty (posterior std)",
                        "dscUncertainty", "cbf"),
            "cbv_std": ("cbv_std_map", "CBV uncertainty (posterior std)",
                        "dscUncertainty", "cbv"),
            "cbfCV": ("cbf_cv_map", "CBF coefficient of variation",
                      "dscCV", "cbf"),
            "cbvCV": ("cbv_cv_map", "CBV coefficient of variation",
                      "dscCV", "cbv"),
            "cbfCI95lower": ("cbf_ci95_lower_map", "CBF 95% credible interval lower bound",
                             "dscCI95lower", "cbf"),
            "cbfCI95upper": ("cbf_ci95_upper_map", "CBF 95% credible interval upper bound",
                             "dscCI95upper", "cbf"),
            "cbvCI95lower": ("cbv_ci95_lower_map", "CBV 95% credible interval lower bound",
                             "dscCI95lower", "cbv"),
            "cbvCI95upper": ("cbv_ci95_upper_map", "CBV 95% credible interval upper bound",
                             "dscCI95upper", "cbv"),
        }
        for umap_key, (attr, desc, desc_val, bids_suffix) in bayesian_uncertainty_maps.items():
            if has_input(self.inputs,attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-{desc_val}_{bids_suffix}"
                copy_derivative(src, perf_dir / f"{fname}.nii.gz")
                # Determine estimation method from config
                est_method = "unknown"
                cfg = self.inputs.config_dict
                if isdefined(cfg) and cfg:
                    est_method = cfg.get("bayesian", "unknown")
                uncertainty_type = "std"
                if "CV" in umap_key:
                    uncertainty_type = "coefficient_of_variation"
                elif "CI95" in umap_key:
                    uncertainty_type = "95pct_credible_interval"
                sidecar = {
                    **base_sidecar,
                    "Description": desc,
                    "EstimationMethod": est_method,
                    "UncertaintyMethod": uncertainty_type,
                }
                write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 6. Smoothed maps (if smooth_fwhm > 0)
        # -----------------------------------------------------------
        fwhm = float(self.inputs.smooth_fwhm)
        if fwhm > 0:
            sm_label = f"Sm{int(fwhm)}"

            # Smoothed hemodynamic maps: CBF, CBV, MTT, nCBF, nCBV
            smooth_hemo = self.inputs.smooth_hemodynamic
            if isdefined(smooth_hemo) and smooth_hemo:
                hemo_descriptions: dict[str, tuple[str, str, str]] = {
                    # key: (description, desc_prefix, bids_suffix)
                    "mtt": ("Mean transit time (smoothed)",
                            f"dsc{sm_label}", "mtt"),
                    "rel_cbf": ("Cerebral blood flow, relative (smoothed)",
                                f"dscRel{sm_label}", "cbf"),
                    "rel_cbv": ("Cerebral blood volume, relative (smoothed)",
                                f"dscRel{sm_label}", "cbv"),
                    "ncbf": ("Normalized CBF (smoothed)",
                             f"dscNawmNorm{sm_label}", "cbf"),
                    "ncbv": ("Normalized CBV (smoothed)",
                             f"dscNawmNorm{sm_label}", "cbv"),
                }
                for name, filepath in smooth_hemo.items():
                    if filepath and Path(filepath).exists():
                        if name in hemo_descriptions:
                            desc_text, desc_prefix, map_suffix = hemo_descriptions[name]
                        else:
                            desc_text = f"{name} (smoothed)"
                            desc_prefix = f"dsc{sm_label}"
                            # Strip quantification prefix for BIDS suffix
                            map_suffix = name.replace("rel_", "").replace("abs_", "")
                        fname = f"{prefix}_{spatial}_desc-{desc_prefix}_{map_suffix}"
                        copy_derivative(filepath, perf_dir / f"{fname}.nii.gz")
                        sidecar = {**base_sidecar, "Description": desc_text,
                                   "SmoothingFWHM": fwhm,
                                   "SmoothingType": "Gaussian",
                                   "SmoothingMasked": True,
                                   "IntendedFor": "visualization",
                                   "Note": "Use non-smoothed maps for quantitative ROI analysis"}
                        write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

            # Smoothed AUC corrected maps (CBV surrogate)
            smooth_a = self.inputs.smooth_auc
            if isdefined(smooth_a) and smooth_a:
                auc_descriptions: dict[str, tuple[str, str]] = {
                    # key: (description, desc_prefix)
                    "rel_auc_corr": ("Relative AUC corrected (smoothed)",
                                     f"dscRelCorrected{sm_label}"),
                }
                for name, filepath in smooth_a.items():
                    if filepath and Path(filepath).exists():
                        if name in auc_descriptions:
                            desc_text, desc_prefix = auc_descriptions[name]
                        else:
                            desc_text = f"{name} (smoothed)"
                            desc_prefix = f"dsc{sm_label}"
                        fname = f"{prefix}_{spatial}_desc-{desc_prefix}_auc"
                        copy_derivative(filepath, perf_dir / f"{fname}.nii.gz")
                        sidecar = {**base_sidecar, "Description": desc_text,
                                   "SmoothingFWHM": fwhm,
                                   "SmoothingType": "Gaussian",
                                   "SmoothingMasked": True,
                                   "IntendedFor": "visualization",
                                   "Note": "Use non-smoothed maps for quantitative ROI analysis"}
                        write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 7. Motion-corrected DSC and reference
        # -----------------------------------------------------------
        if has_input(self.inputs,"dsc_moco"):
            fname = f"{prefix}_{spatial}_desc-dscPreproc_dsc"
            copy_derivative(self.inputs.dsc_moco, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Motion-corrected 4D DSC time series"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"dsc_mean"):
            fname = f"{prefix}_{spatial}_desc-dscMean_ref"
            copy_derivative(self.inputs.dsc_mean, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Mean DSC reference image"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs, "motion_params"):
            from pmriprep.utils.outputs import (
                par_to_confounds_tsv,
                write_confounds_sidecar,
            )

            confounds_stem = f"{prefix}_desc-dscConfounds_timeseries"
            par_to_confounds_tsv(
                self.inputs.motion_params,
                perf_dir / f"{confounds_stem}.tsv",
            )
            write_confounds_sidecar(perf_dir / f"{confounds_stem}.json")

        # -----------------------------------------------------------
        # 8. AIF outputs
        # -----------------------------------------------------------
        if has_input(self.inputs,"aif_curve"):
            copy_derivative(
                self.inputs.aif_curve,
                perf_dir / f"{prefix}_desc-dscAif_timeseries.txt",
            )
        if has_input(self.inputs,"aif_mask"):
            copy_derivative(
                self.inputs.aif_mask,
                perf_dir / f"{prefix}_{spatial}_desc-dscAif_mask.nii.gz",
            )

        # -----------------------------------------------------------
        # 8b. DSC-space masks (written to anat/)
        # -----------------------------------------------------------
        minimal_sidecar = self._build_base_sidecar(bids_dir, voxel_sizes, minimal=True)

        if has_input(self.inputs,"brain_mask_dsc"):
            fname = f"{prefix}_{spatial}_desc-brain_mask"
            copy_derivative(self.inputs.brain_mask_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "Brain mask in DSC space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        if has_input(self.inputs,"nawm_mask_dsc"):
            fname = f"{prefix}_{spatial}_desc-nawm_mask"
            copy_derivative(self.inputs.nawm_mask_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "Normal-appearing white matter mask in DSC space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 9. Registration transforms (written to anat/)
        # -----------------------------------------------------------
        if has_input(self.inputs,"dsc_to_t1w_xfm"):
            copy_derivative(
                self.inputs.dsc_to_t1w_xfm,
                anat_dir / f"{prefix}_from-DSC_to-anat_mode-image_xfm.mat",
            )

        if has_input(self.inputs,"t1w_to_dsc_xfm"):
            copy_derivative(
                self.inputs.t1w_to_dsc_xfm,
                anat_dir / f"{prefix}_from-anat_to-DSC_mode-image_xfm.mat",
            )

        # -----------------------------------------------------------
        # 10. QC outputs (written to session root)
        # -----------------------------------------------------------
        # Collect processing warnings for missing data-dependent inputs
        _data_dependent_inputs = {
            "brain_mask_dsc": "Brain mask in DSC space",
            "nawm_mask_dsc": "NAWM mask in DSC space",
            "lesion_mask_dsc": "Lesion mask in DSC space",
            "dsc_to_t1w_xfm": "DSC to T1w registration transform",
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
            # Add NiPreps-style recommendation to QC metrics
            enhanced_qc = self._add_qc_recommendation(qc_metrics)
            qc_json_path = ses_dir / f"{prefix}_desc-dscQC_quality.json"
            write_json_sidecar(enhanced_qc, qc_json_path)
            self._qc_json = str(qc_json_path)

        # -----------------------------------------------------------
        # 10b. QC figures -> figures/ with BIDS names
        # -----------------------------------------------------------
        from pmriprep.utils.report_utils import figure_filename

        figures_dir = ses_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        _fig_inputs = {
            "dsc_brain_mask": "brain_mask_filmstrip",
            "motion": "motion_qc_plot",
            "aif_curve": "aif_curve_plot",
            "aif_location": "aif_location_plot",
            "registration": "registration_filmstrip",
            "dsc_preprocessing_qc": "dsc_preprocessing_qc_plot",
            "dsc_param_maps_qc": "dsc_param_maps_qc_plot",
            "dsc_fit_qc": "dsc_fit_qc_plot",
            "dsc_leakage_qc": "dsc_leakage_qc_plot",
        }
        for fig_key, attr in _fig_inputs.items():
            if has_input(self.inputs,attr):
                fname = figure_filename(subject, session, fig_key, "dsc")
                copy_derivative(getattr(self.inputs, attr), figures_dir / fname)

        # -----------------------------------------------------------
        # 10c. Structural underlays (in DSC space, written to anat/)
        # -----------------------------------------------------------
        if has_input(self.inputs,"t1w_in_dsc"):
            if self.inputs.has_postcon_t1w:
                label = "ce-GAD_T1w"
                description = "CE-T1w registered to DSC space"
            else:
                label = "T1w"
                description = "T1w registered to DSC space"
            fname = f"{prefix}_{spatial}_{label}"
            copy_derivative(self.inputs.t1w_in_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": description}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        if has_input(self.inputs,"flair_in_dsc"):
            fname = f"{prefix}_{spatial}_FLAIR"
            copy_derivative(self.inputs.flair_in_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "FLAIR registered to DSC space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        if has_input(self.inputs,"t2w_in_dsc"):
            fname = f"{prefix}_{spatial}_T2w"
            copy_derivative(self.inputs.t2w_in_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "T2w registered to DSC space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 10d. Lesion outputs (written to session root / anat)
        # -----------------------------------------------------------
        if has_input(self.inputs,"lesion_stats_json"):
            copy_derivative(
                self.inputs.lesion_stats_json,
                ses_dir / f"{prefix}_space-DSC_desc-lesionStats.json",
            )

        if has_input(self.inputs,"lesion_mask_dsc"):
            fname = f"{prefix}_{spatial}_desc-lesion_mask"
            copy_derivative(self.inputs.lesion_mask_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "Lesion mask registered to DSC space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 10e. ROI statistics and mask (stats extraction only)
        # -----------------------------------------------------------
        if has_input(self.inputs, "roi_stats_json"):
            copy_derivative(
                self.inputs.roi_stats_json,
                ses_dir / f"{prefix}_space-DSC_desc-roiStats.json",
            )

        if has_input(self.inputs, "roi_mask_dsc"):
            fname = f"{prefix}_{spatial}_desc-roi_mask"
            copy_derivative(self.inputs.roi_mask_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "ROI mask registered to DSC space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 11. Dataset description (shared utility)
        # -----------------------------------------------------------
        from pmriprep.utils.outputs import write_dataset_description

        write_dataset_description(pipeline_dir, bids_dir)

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

    def _build_base_sidecar(
        self, bids_dir: str, voxel_sizes: tuple, minimal: bool = False,
    ) -> dict[str, Any]:
        """Build base JSON sidecar with processing metadata.

        Args:
            bids_dir: Path to the BIDS dataset root.
            voxel_sizes: Voxel dimensions from the reference image header.
            minimal: If True, omit perfusion-specific fields
                (DeconvolutionMethod, BayesianInference, PreBolus).
                Use for masks, structural underlays, and non-perfusion maps.

        Returns:
            Base sidecar dictionary with common metadata fields.
        """
        sidecar: dict[str, Any] = {
            **get_bids_derivative_provenance(),
            "Space": "DSC (native perfusion space)",
            "Resolution": [float(v) for v in voxel_sizes],
        }

        # Add source file reference
        source_dsc = getattr(self.inputs, "source_dsc", None)
        if source_dsc and isdefined(source_dsc) and source_dsc:
            try:
                bids_root = Path(bids_dir)
                src_path = Path(source_dsc)
                if src_path.is_relative_to(bids_root):
                    rel = src_path.relative_to(bids_root)
                    sidecar["Sources"] = [f"bids::{rel}"]
                else:
                    sidecar["Sources"] = [str(source_dsc)]
            except (ValueError, TypeError):
                sidecar["Sources"] = [str(source_dsc)]

        # Add config summary (only for perfusion maps)
        if not minimal:
            config_dict = self.inputs.config_dict
            if isdefined(config_dict) and config_dict:
                method = config_dict.get("method", "unknown")
                sidecar["DeconvolutionMethod"] = method
                if method in ("cpi", "vascular"):
                    sidecar["BayesianInference"] = config_dict.get("bayesian")
                sidecar["PreBolus"] = config_dict.get("pre_bolus", 5)

        return sidecar

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
            QCStatus,
            aggregate_qc_recommendations,
        )

        recommendations: list[QCRecommendation] = []
        enhanced = dict(qc_metrics)

        # Evaluate AIF detection quality
        aif_qc = qc_metrics.get("aif", {})
        detection_level = aif_qc.get("detection_level", "unknown")
        if detection_level in ("atlas_strict", "atlas_relaxed"):
            aif_rec = QCRecommendation(QCAction.INCLUDE, f"AIF detected ({detection_level})", 0)
        elif detection_level in ("atlas_permissive", "brain_relaxed", "brain_permissive"):
            aif_rec = QCRecommendation(QCAction.REVIEW, f"AIF detection marginal ({detection_level})", 1)
        # Note: brain_degraded removed in v0.19.16 - cascade now goes to synthetic
        elif detection_level == "external":
            aif_rec = QCRecommendation(QCAction.REVIEW, "External AIF provided; verify curve quality", 1)
        elif detection_level == "synthetic":
            aif_rec = QCRecommendation(QCAction.EXCLUDE, "Synthetic AIF used; verify clinically", 3)
        elif detection_level in ("none", "unknown"):
            aif_rec = QCRecommendation(QCAction.EXCLUDE, f"AIF detection failed ({detection_level})", 3)
        else:
            aif_rec = QCRecommendation(QCAction.REVIEW, f"AIF detection: {detection_level}", 1)
        recommendations.append(aif_rec)

        # Evaluate motion QC using composite motion_status
        motion_qc = qc_metrics.get("motion", {})
        motion_status = motion_qc.get("motion_status")
        if motion_status is not None:
            fd_mean = motion_qc.get("fd_mean")
            fd_max = motion_qc.get("fd_max")
            pct_outliers = motion_qc.get("percent_outliers")
            parts = []
            if fd_mean is not None:
                parts.append(f"mean FD={fd_mean:.2f}mm")
            if fd_max is not None:
                parts.append(f"max FD={fd_max:.2f}mm")
            if pct_outliers is not None:
                parts.append(f"outliers={pct_outliers:.1f}%")
            detail = f"({', '.join(parts)})" if parts else ""

            if motion_status == QCStatus.PASS:
                motion_rec = QCRecommendation(QCAction.INCLUDE, f"Motion acceptable {detail}", 0)
            elif motion_status == QCStatus.WARN:
                motion_rec = QCRecommendation(QCAction.REVIEW, f"Motion marginal {detail}", 1)
            elif motion_status == QCStatus.POOR:
                motion_rec = QCRecommendation(QCAction.EXCLUDE, f"Excessive motion {detail}", 3)
            else:
                motion_rec = None
            if motion_rec is not None:
                recommendations.append(motion_rec)

        # Evaluate CNR if available
        cnr_qc = qc_metrics.get("cnr_whole_brain", {})
        cnr_fraction = cnr_qc.get("fraction_cnr_gt_4")
        if cnr_fraction is not None:
            if cnr_fraction >= 0.84:
                cnr_rec = QCRecommendation(QCAction.INCLUDE, f"CNR excellent ({cnr_fraction:.0%})", 0)
            elif cnr_fraction >= 0.65:
                cnr_rec = QCRecommendation(QCAction.REVIEW, f"CNR marginal ({cnr_fraction:.0%})", 1)
            else:
                cnr_rec = QCRecommendation(QCAction.EXCLUDE, f"CNR poor ({cnr_fraction:.0%})", 3)
            recommendations.append(cnr_rec)

        # Aggregate recommendations
        overall = aggregate_qc_recommendations(recommendations)
        enhanced["recommendation"] = {
            "action": overall.action,
            "reason": overall.reason,
            "severity": overall.severity,
        }

        return enhanced
