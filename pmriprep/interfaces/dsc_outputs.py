"""BIDS-Derivatives compliant DSC perfusion output writer interface.

This module implements a custom Nipype interface for writing DSC pipeline
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
    ttp_map = traits.Str(desc="TTP map path")
    cnr_map = traits.Str(desc="CNR map path")
    auc_diff_map = traits.Str(desc="AUC diff (leakage contribution) map path")
    snr_map = traits.Str(desc="SNR (baseline) map path")
    tsnr_map = traits.Str(desc="tSNR (temporal) map path")

    # Relative maps (always produced)
    rel_cbf_map = traits.Str(desc="Relative CBF map path")
    rel_cbv_map = traits.Str(desc="Relative CBV map path")
    rel_auc_raw_map = traits.Str(desc="Relative AUC raw map path")
    rel_auc_corr_map = traits.Str(desc="Relative AUC corrected map path")

    # Absolute maps (conditional on gating — may be empty)
    abs_cbf_map = traits.Str(desc="Absolute CBF map path")
    abs_cbv_map = traits.Str(desc="Absolute CBV map path")
    abs_auc_raw_map = traits.Str(desc="Absolute AUC raw map path")
    abs_auc_corr_map = traits.Str(desc="Absolute AUC corrected map path")

    # Normalized maps (optional)
    ncbv_map = traits.Str(desc="Normalized CBV map path")
    ncbf_map = traits.Str(desc="Normalized CBF map path")

    # Vascular model parameters (optional)
    alpha_map = traits.Str(desc="Alpha parameter map path (vascular only)")
    beta_map = traits.Str(desc="Beta parameter map path (vascular only)")
    cth_map = traits.Str(desc="CTH (capillary transit time heterogeneity) map path (vascular only)")

    # Bayesian uncertainty maps (optional)
    cbf_std_map = traits.Str(desc="CBF uncertainty (std) map path (Bayesian only)")
    cbv_std_map = traits.Str(desc="CBV uncertainty (std) map path (Bayesian only)")
    cbf_cv_map = traits.Str(desc="CBF coefficient of variation path (Bayesian only)")
    cbv_cv_map = traits.Str(desc="CBV coefficient of variation path (Bayesian only)")
    cbf_ci95_lower_map = traits.Str(desc="CBF 95% CI lower bound path (VI/MCMC only)")
    cbf_ci95_upper_map = traits.Str(desc="CBF 95% CI upper bound path (VI/MCMC only)")
    cbv_ci95_lower_map = traits.Str(desc="CBV 95% CI lower bound path (VI/MCMC only)")
    cbv_ci95_upper_map = traits.Str(desc="CBV 95% CI upper bound path (VI/MCMC only)")

    # Leakage correction (optional)
    k2_map = traits.Str(desc="K2 leakage correction map path")

    # Motion-corrected data
    dsc_moco = traits.Str(desc="Motion-corrected 4D DSC path")
    dsc_mean = traits.Str(desc="Mean DSC image path")

    # AIF outputs
    aif_curve = traits.Str(desc="AIF curve text file path")
    aif_mask = traits.Str(desc="AIF detection mask path")

    # Registration
    dsc_to_t1w_xfm = traits.Str(desc="DSC to T1w transform matrix path")
    registration_filmstrip = traits.Str(desc="Registration QC filmstrip PNG path")

    # QC metrics
    qc_metrics = traits.Dict(desc="Merged QC metrics dictionary")

    # Source DSC file for sidecar
    source_dsc = traits.Str(desc="Original raw DSC source file path")

    # Processing config for sidecar
    config_dict = traits.Dict(desc="DSC processing config as dictionary")

    # Smoothed map dicts (optional — each is a dict of name→filepath)
    smooth_hemodynamic = traits.Dict(desc="Smoothed hemodynamic maps (CBF, CBV, MTT, nCBF, nCBV)")
    smooth_auc = traits.Dict(desc="Smoothed AUC corrected maps (rel + abs)")
    smooth_fwhm = traits.Float(0.0, usedefault=True, desc="Smoothing FWHM in mm")

    # Structural underlays (registered to DSC space)
    t1w_in_dsc = traits.Str(desc="CE-T1w registered to DSC space NIfTI path")
    flair_in_dsc = traits.Str(desc="FLAIR registered to DSC space NIfTI path")
    t1w_to_dsc_xfm = traits.Str(desc="T1w-to-DSC transform matrix path")

    # AIF QC plots
    aif_curve_plot = traits.Str(desc="AIF curve QC plot PNG path")
    aif_location_plot = traits.Str(desc="AIF location overlay PNG path")

    # Motion QC plot
    motion_qc_plot = traits.Str(desc="Motion QC plot PNG path")

    # Lesion statistics
    lesion_stats_json = traits.Str(desc="Lesion statistics JSON file path")
    lesion_stats_txt = traits.Str(desc="Lesion statistics text summary path")
    lesion_mask_dsc = traits.Str(desc="Lesion mask in DSC space path")


class DSCDerivativeWriterOutputSpec(TraitedSpec):
    """Output specification for DSCDerivativeWriter."""

    out_dir = traits.Str(desc="Path to the output session directory")
    qc_json = traits.Str(desc="Path to the QC JSON file")


class DSCDerivativeWriter(BaseInterface):
    """Write DSC perfusion pipeline outputs with BIDS-compliant naming.

    Output directory structure::

        {output_dir}/dsc_pipeline/
            dataset_description.json
            sub-{subject}/ses-{session}/
                perf/
                    sub-XX_ses-XX_space-DSC_res-X.XXxX.XX_desc-{desc}_{map}.nii.gz
                    sub-XX_ses-XX_space-DSC_res-X.XXxX.XX_desc-{desc}_{map}.json
                    ...
                xfm/
                    sub-XX_ses-XX_from-DSC_to-T1w_mode-image_xfm.mat
                qc/
                    dsc_qc.json
                    registration_qc_filmstrip.png
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

        # Build resolution string from MTT map header (always present)
        ref_img = nib.load(self.inputs.mtt_map)
        voxel_sizes = ref_img.header.get_zooms()[:3]  # type: ignore[attr-defined]
        res_str = "x".join(f"{v:.2f}" for v in voxel_sizes)

        # Build spatial entity string
        spatial = f"space-DSC_res-{res_str}"

        # Create directory structure
        pipeline_dir = output_dir / "dsc_pipeline"
        ses_dir = pipeline_dir / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        perf_dir = ses_dir / "perf"
        xfm_dir = ses_dir / "xfm"
        qc_dir = ses_dir / "qc"

        perf_dir.mkdir(parents=True, exist_ok=True)
        xfm_dir.mkdir(parents=True, exist_ok=True)
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Build base sidecar from config
        base_sidecar = self._build_base_sidecar(bids_dir)

        # -----------------------------------------------------------
        # 1. Relative maps (ALWAYS produced)
        # -----------------------------------------------------------
        rel_maps = {
            "cbf": ("rel_cbf_map", "Cerebral blood flow (relative)"),
            "cbv": ("rel_cbv_map", "Cerebral blood volume (relative)"),
            "auc_raw": ("rel_auc_raw_map", "AUC raw concentration (relative)"),
            "auc_corr": ("rel_auc_corr_map", "AUC corrected concentration (relative)"),
        }
        rel_units = {"cbf": "1/min", "cbv": "arbitrary"}
        for map_name, (attr, description) in rel_maps.items():
            if self._has_input(attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-rel_{map_name}"
                self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                if map_name in rel_units:
                    sidecar["Units"] = rel_units[map_name]
                self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 2. Absolute maps (CONDITIONAL — only if gating passed)
        # -----------------------------------------------------------
        abs_maps = {
            "cbf": ("abs_cbf_map", "Cerebral blood flow (mL/100g/min)"),
            "cbv": ("abs_cbv_map", "Cerebral blood volume (mL/100g)"),
            "auc_raw": ("abs_auc_raw_map", "AUC raw concentration (mM·s)"),
            "auc_corr": ("abs_auc_corr_map", "AUC corrected concentration (mM·s)"),
        }
        abs_units = {"cbf": "mL/100g/min", "cbv": "mL/100g", "auc_raw": "mM*s", "auc_corr": "mM*s"}
        for map_name, (attr, description) in abs_maps.items():
            if self._has_input(attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-abs_{map_name}"
                self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                if map_name in abs_units:
                    sidecar["Units"] = abs_units[map_name]
                self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3. Quantification-independent maps (desc-dscproc)
        # -----------------------------------------------------------
        core_maps = {
            "mtt": ("mtt_map", "Mean transit time"),
            "tmax": ("tmax_map", "Time to maximum of residue function"),
            "psr": ("psr_map", "Percent signal recovery"),
            "ph": ("ph_map", "Peak height"),
        }
        for map_name, (attr, description) in core_maps.items():
            src = getattr(self.inputs, attr)
            fname = f"{prefix}_{spatial}_desc-dscproc_{map_name}"
            self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": description}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # Additional desc-dscproc maps (optional)
        optional_dscproc = {
            "ttp": ("ttp_map", "Time to peak concentration"),
            "cnr": ("cnr_map", "Contrast-to-noise ratio"),
            "auc_diff": ("auc_diff_map", "Leakage contribution (AUC_raw - AUC_corr)"),
            "snr": ("snr_map", "Signal-to-noise ratio (baseline)"),
            "tsnr": ("tsnr_map", "Temporal signal-to-noise ratio"),
        }
        for map_name, (attr, description) in optional_dscproc.items():
            if self._has_input(attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-dscproc_{map_name}"
                self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 4. Normalized maps (optional)
        # -----------------------------------------------------------
        if self._has_input("ncbv_map"):
            fname = f"{prefix}_{spatial}_desc-dscproc_ncbv"
            self._copy_nifti(self.inputs.ncbv_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Normalized CBV (NAWM-normalized)"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("ncbf_map"):
            fname = f"{prefix}_{spatial}_desc-dscproc_ncbf"
            self._copy_nifti(self.inputs.ncbf_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Normalized CBF (NAWM-normalized)"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 5. Leakage correction map (optional)
        # -----------------------------------------------------------
        if self._has_input("k2_map"):
            fname = f"{prefix}_{spatial}_desc-dscproc_k2"
            self._copy_nifti(self.inputs.k2_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "K2 leakage correction map"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 5b. Vascular model parameters (optional)
        # -----------------------------------------------------------
        for vmap, (attr, desc) in {
            "alpha": ("alpha_map", "Vascular model alpha parameter"),
            "beta": ("beta_map", "Vascular model beta parameter"),
            "cth": ("cth_map", "Capillary transit time heterogeneity"),
        }.items():
            if self._has_input(attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-dscproc_{vmap}"
                self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": desc}
                self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 5c. Bayesian uncertainty maps (optional)
        # -----------------------------------------------------------
        bayesian_uncertainty_maps = {
            "cbf_std": ("cbf_std_map", "CBF uncertainty (posterior std)"),
            "cbv_std": ("cbv_std_map", "CBV uncertainty (posterior std)"),
            "cbfCV": ("cbf_cv_map", "CBF coefficient of variation"),
            "cbvCV": ("cbv_cv_map", "CBV coefficient of variation"),
            "cbfCI95lower": ("cbf_ci95_lower_map", "CBF 95% credible interval lower bound"),
            "cbfCI95upper": ("cbf_ci95_upper_map", "CBF 95% credible interval upper bound"),
            "cbvCI95lower": ("cbv_ci95_lower_map", "CBV 95% credible interval lower bound"),
            "cbvCI95upper": ("cbv_ci95_upper_map", "CBV 95% credible interval upper bound"),
        }
        for umap, (attr, desc) in bayesian_uncertainty_maps.items():
            if self._has_input(attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-dscproc_{umap}"
                self._copy_nifti(src, perf_dir / f"{fname}.nii.gz")
                # Determine estimation method from config
                est_method = "unknown"
                cfg = self.inputs.config_dict
                if isdefined(cfg) and cfg:
                    est_method = cfg.get("bayesian", "unknown")
                uncertainty_type = "std"
                if "CV" in umap:
                    uncertainty_type = "coefficient_of_variation"
                elif "CI95" in umap:
                    uncertainty_type = "95pct_credible_interval"
                sidecar = {
                    **base_sidecar,
                    "Description": desc,
                    "EstimationMethod": est_method,
                    "UncertaintyMethod": uncertainty_type,
                }
                self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 6. Smoothed maps (if smooth_fwhm > 0)
        # -----------------------------------------------------------
        fwhm = float(self.inputs.smooth_fwhm)
        if fwhm > 0:
            fwhm_label = f"fwhm{int(fwhm)}"

            # Smoothed hemodynamic maps: CBF, CBV, MTT, nCBF, nCBV
            smooth_hemo = self.inputs.smooth_hemodynamic
            if isdefined(smooth_hemo) and smooth_hemo:
                hemo_descriptions = {
                    "mtt": ("Mean transit time (smoothed)", f"{fwhm_label}"),
                    "rel_cbf": ("Cerebral blood flow, relative (smoothed)", f"rel{fwhm_label.capitalize()}"),
                    "rel_cbv": ("Cerebral blood volume, relative (smoothed)", f"rel{fwhm_label.capitalize()}"),
                    "abs_cbf": ("Cerebral blood flow, mL/100g/min (smoothed)", f"abs{fwhm_label.capitalize()}"),
                    "abs_cbv": ("Cerebral blood volume, mL/100g (smoothed)", f"abs{fwhm_label.capitalize()}"),
                    "ncbf": ("Normalized CBF (smoothed)", f"{fwhm_label}"),
                    "ncbv": ("Normalized CBV (smoothed)", f"{fwhm_label}"),
                }
                for name, filepath in smooth_hemo.items():
                    if filepath and Path(filepath).exists():
                        desc_text, desc_prefix = hemo_descriptions.get(
                            name, (f"{name} (smoothed)", fwhm_label))
                        # Strip quantification prefix from map suffix for BIDS naming
                        map_suffix = name.replace("rel_", "").replace("abs_", "")
                        fname = f"{prefix}_{spatial}_desc-{desc_prefix}_{map_suffix}"
                        self._copy_nifti(filepath, perf_dir / f"{fname}.nii.gz")
                        sidecar = {**base_sidecar, "Description": desc_text,
                                   "SmoothingFWHM": fwhm,
                                   "SmoothingType": "Gaussian",
                                   "SmoothingMasked": True,
                                   "IntendedFor": "visualization",
                                   "Note": "Use non-smoothed maps for quantitative ROI analysis"}
                        self._write_json(sidecar, perf_dir / f"{fname}.json")

            # Smoothed AUC corrected maps (CBV surrogate)
            smooth_a = self.inputs.smooth_auc
            if isdefined(smooth_a) and smooth_a:
                auc_descriptions = {
                    "rel_auc_corr": ("Relative AUC corrected (smoothed)", f"rel{fwhm_label.capitalize()}"),
                    "abs_auc_corr": ("Absolute AUC corrected, mM·s (smoothed)", f"abs{fwhm_label.capitalize()}"),
                }
                for name, filepath in smooth_a.items():
                    if filepath and Path(filepath).exists():
                        desc_text, desc_prefix = auc_descriptions.get(
                            name, (f"{name} (smoothed)", fwhm_label))
                        fname = f"{prefix}_{spatial}_desc-{desc_prefix}_auc_corr"
                        self._copy_nifti(filepath, perf_dir / f"{fname}.nii.gz")
                        sidecar = {**base_sidecar, "Description": desc_text,
                                   "SmoothingFWHM": fwhm,
                                   "SmoothingType": "Gaussian",
                                   "SmoothingMasked": True,
                                   "IntendedFor": "visualization",
                                   "Note": "Use non-smoothed maps for quantitative ROI analysis"}
                        self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 7. Motion-corrected DSC and reference
        # -----------------------------------------------------------
        if self._has_input("dsc_moco"):
            fname = f"{prefix}_{spatial}_desc-mc_dsc"
            self._copy_nifti(self.inputs.dsc_moco, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Motion-corrected 4D DSC time series"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        if self._has_input("dsc_mean"):
            fname = f"{prefix}_{spatial}_desc-ref_dsc"
            self._copy_nifti(self.inputs.dsc_mean, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Mean DSC reference image"}
            self._write_json(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 8. AIF outputs
        # -----------------------------------------------------------
        if self._has_input("aif_curve"):
            self._copy_file(
                self.inputs.aif_curve,
                perf_dir / f"{prefix}_desc-aif_timeseries.txt",
            )
        if self._has_input("aif_mask"):
            self._copy_nifti(
                self.inputs.aif_mask,
                perf_dir / f"{prefix}_{spatial}_desc-aif_mask.nii.gz",
            )

        # -----------------------------------------------------------
        # 9. Registration transform
        # -----------------------------------------------------------
        if self._has_input("dsc_to_t1w_xfm"):
            self._copy_file(
                self.inputs.dsc_to_t1w_xfm,
                xfm_dir / f"{prefix}_from-DSC_to-T1w_mode-image_xfm.mat",
            )

        # -----------------------------------------------------------
        # 10. QC outputs
        # -----------------------------------------------------------
        if self._has_input("registration_filmstrip"):
            self._copy_file(
                self.inputs.registration_filmstrip,
                qc_dir / "registration_qc_filmstrip.png",
            )

        qc_metrics = self.inputs.qc_metrics
        self._qc_json = ""
        if isdefined(qc_metrics) and qc_metrics:
            # Add NiPreps-style recommendation to QC metrics
            enhanced_qc = self._add_qc_recommendation(qc_metrics)
            qc_json_path = qc_dir / "dsc_qc.json"
            self._write_json(enhanced_qc, qc_json_path)
            self._qc_json = str(qc_json_path)

        # -----------------------------------------------------------
        # 10b. AIF QC plots
        # -----------------------------------------------------------
        if self._has_input("aif_curve_plot"):
            self._copy_file(
                self.inputs.aif_curve_plot,
                qc_dir / "aif_curve_qc.png",
            )
        if self._has_input("aif_location_plot"):
            self._copy_file(
                self.inputs.aif_location_plot,
                qc_dir / "aif_location_qc.png",
            )

        if self._has_input("motion_qc_plot"):
            qc_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(
                self.inputs.motion_qc_plot,
                qc_dir / "motion_qc.png",
            )

        # -----------------------------------------------------------
        # 10c. Structural underlays (in DSC space)
        # -----------------------------------------------------------
        anat_dir = ses_dir / "anat"
        if self._has_input("t1w_in_dsc"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_ce-GAD_T1w"
            self._copy_nifti(self.inputs.t1w_in_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "CE-T1w registered to DSC space"}
            self._write_json(sidecar, anat_dir / f"{fname}.json")

        if self._has_input("flair_in_dsc"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_FLAIR"
            self._copy_nifti(self.inputs.flair_in_dsc, anat_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "FLAIR registered to DSC space"}
            self._write_json(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 10d. Lesion statistics (conditional)
        # -----------------------------------------------------------
        if self._has_input("lesion_stats_json"):
            stats_dir = ses_dir / "stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(
                self.inputs.lesion_stats_json,
                stats_dir / f"{prefix}_{spatial}_lesion-stats.json",
            )
        if self._has_input("lesion_stats_txt"):
            stats_dir = ses_dir / "stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            self._copy_file(
                self.inputs.lesion_stats_txt,
                stats_dir / f"{prefix}_{spatial}_lesion-stats.txt",
            )
        if self._has_input("lesion_mask_dsc"):
            roi_dir = ses_dir / "roi"
            roi_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_desc-lesion_mask"
            self._copy_nifti(self.inputs.lesion_mask_dsc, roi_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Lesion mask registered to DSC space"}
            self._write_json(sidecar, roi_dir / f"{fname}.json")

        if self._has_input("t1w_to_dsc_xfm"):
            self._copy_file(
                self.inputs.t1w_to_dsc_xfm,
                xfm_dir / f"{prefix}_from-T1w_to-DSC_mode-image_xfm.mat",
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
            "Pipeline": "PMRIPrep DSC",
            "Space": "DSC (native perfusion space)",
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

        # Add config summary
        config_dict = self.inputs.config_dict
        if isdefined(config_dict) and config_dict:
            sidecar["DeconvolutionMethod"] = config_dict.get("method", "unknown")
            sidecar["BayesianInference"] = config_dict.get("bayesian", "none")
            sidecar["PreBolus"] = config_dict.get("pre_bolus", 5)

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
            QCStatus,
            aggregate_qc_recommendations,
        )

        recommendations: list[QCRecommendation] = []
        enhanced = dict(qc_metrics)

        # Evaluate AIF detection quality
        aif_qc = qc_metrics.get("aif", {})
        detection_level = aif_qc.get("detection_level", "unknown")
        level_to_rec = {
            "atlas": QCRecommendation(QCAction.INCLUDE, "AIF detected via atlas method", 0),
            "relaxed": QCRecommendation(QCAction.REVIEW, "AIF detected via relaxed criteria", 1),
            "synthetic": QCRecommendation(QCAction.EXCLUDE, "Synthetic AIF used; verify clinically", 3),
        }
        aif_rec = level_to_rec.get(
            detection_level,
            QCRecommendation(QCAction.REVIEW, f"AIF detection: {detection_level}", 1),
        )
        recommendations.append(aif_rec)

        # Evaluate motion QC if available
        motion_qc = qc_metrics.get("motion", {})
        fd_mean = motion_qc.get("fd_mean")
        if fd_mean is not None:
            if fd_mean < 0.3:
                motion_rec = QCRecommendation(QCAction.INCLUDE, f"Motion acceptable (FD={fd_mean:.2f}mm)", 0)
            elif fd_mean < 0.5:
                motion_rec = QCRecommendation(QCAction.REVIEW, f"Motion marginal (FD={fd_mean:.2f}mm)", 1)
            else:
                motion_rec = QCRecommendation(QCAction.EXCLUDE, f"Excessive motion (FD={fd_mean:.2f}mm)", 3)
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

    @staticmethod
    def _write_dataset_description(pipeline_dir: Path) -> None:
        """Write dataset_description.json at the pipeline root."""
        import pmriprep

        desc_path = pipeline_dir / "dataset_description.json"
        if desc_path.exists():
            return

        pipeline_dir.mkdir(parents=True, exist_ok=True)
        description = {
            "Name": "PMRIPrep - DSC Perfusion Pipeline",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "pmriprep",
                    "Version": pmriprep.__version__,
                    "Description": "DSC perfusion processing including deconvolution and parameter extraction",
                }
            ],
            "License": "CC-BY-4.0",
        }
        with open(desc_path, "w") as f:
            json.dump(description, f, indent=2)
