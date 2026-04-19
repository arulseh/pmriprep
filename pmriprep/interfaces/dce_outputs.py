"""BIDS-Derivatives compliant DCE perfusion output writer interface.

This module implements a custom Nipype interface for writing DCE pipeline
outputs with BIDS-compliant filenames, JSON sidecars, and QC files.
"""

import logging
import re
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

from pmriprep.utils.outputs import (
    copy_derivative,
    has_input,
    sanitize_model_entity,
    write_dataset_description,
    write_json_sidecar,
)
from pmriprep.utils.provenance import get_bids_derivative_provenance

logger = logging.getLogger(__name__)


class DCEDerivativeWriterInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCEDerivativeWriter."""

    # Configuration (mandatory)
    output_dir = traits.Str(mandatory=True, desc="Root output derivatives directory")
    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")
    bids_dir = traits.Str(mandatory=True, desc="BIDS dataset root directory")

    # Core pharmacokinetic parameter maps
    ktrans_map = File(exists=True, mandatory=True, desc="Ktrans (volume transfer constant) map")
    ve_map = traits.Either(traits.Str, None, desc="ve (EES volume fraction) map path")
    kep_map = traits.Either(traits.Str, None, desc="kep (rate constant) map path")

    # Optional PK parameter (only for extended Tofts/2CXM)
    vp_map = traits.Either(traits.Str, None, desc="vp (plasma volume fraction) map path")

    # T1/B1 mapping outputs (optional)
    t1_map = traits.Either(traits.Str, None, desc="T1 map path")
    b1_map = traits.Either(traits.Str, None, desc="B1+ map path")

    # Motion-corrected data (optional)
    dce_moco = traits.Either(traits.Str, None, desc="Motion-corrected 4D DCE path")
    dce_mean = traits.Either(traits.Str, None, desc="Mean DCE image path")
    motion_params = traits.Either(traits.Str, None, desc="Motion parameters file path")

    # AIF outputs (optional)
    aif_curve = traits.Either(traits.Str, None, desc="AIF curve text file path")
    aif_mask = traits.Either(traits.Str, None, desc="AIF detection mask path")

    # Registration (optional)
    dce_to_t1w_xfm = traits.Either(traits.Str, None, desc="DCE to T1w transform matrix path")

    # Uncertainty outputs (Bayesian, optional)
    ktrans_uncertainty = traits.Either(traits.Str, None, desc="Ktrans uncertainty map path")
    ve_uncertainty = traits.Either(traits.Str, None, desc="ve uncertainty map path")
    vp_uncertainty = traits.Either(traits.Str, None, desc="vp uncertainty map path")
    fp_uncertainty = traits.Either(traits.Str, None, desc="Fp uncertainty map path (2CXM only)")
    ps_uncertainty = traits.Either(traits.Str, None, desc="PS uncertainty map path (2CXM only)")

    # Bayesian CV and CI outputs (optional)
    ktrans_cv = traits.Either(traits.Str, None, desc="Ktrans coefficient of variation map path")
    ve_cv = traits.Either(traits.Str, None, desc="ve coefficient of variation map path")
    vp_cv = traits.Either(traits.Str, None, desc="vp coefficient of variation map path")
    ktrans_ci95_lower = traits.Either(traits.Str, None, desc="Ktrans 95% CI lower bound map path")
    ktrans_ci95_upper = traits.Either(traits.Str, None, desc="Ktrans 95% CI upper bound map path")
    ve_ci95_lower = traits.Either(traits.Str, None, desc="ve 95% CI lower bound map path")
    ve_ci95_upper = traits.Either(traits.Str, None, desc="ve 95% CI upper bound map path")
    vp_ci95_lower = traits.Either(traits.Str, None, desc="vp 95% CI lower bound map path")
    vp_ci95_upper = traits.Either(traits.Str, None, desc="vp 95% CI upper bound map path")

    # Model-free maps (optional)
    iauc60_map = traits.Either(traits.Str, None, desc="IAUC at 60s map path")
    iauc90_map = traits.Either(traits.Str, None, desc="IAUC at 90s map path")
    iauc120_map = traits.Either(traits.Str, None, desc="IAUC at 120s map path")
    peak_enhancement_map = traits.Either(traits.Str, None, desc="Peak enhancement map path")
    time_to_peak_map = traits.Either(traits.Str, None, desc="Time to peak map path")
    ser_map = traits.Either(traits.Str, None, desc="Signal enhancement ratio map path")
    wash_in_slope_map = traits.Either(traits.Str, None, desc="Wash-in slope map path")
    wash_out_slope_map = traits.Either(traits.Str, None, desc="Wash-out slope map path")
    bat_map = traits.Either(traits.Str, None, desc="Bolus arrival time map path")
    curve_type_map = traits.Either(traits.Str, None, desc="Curve type classification map path")

    # QC maps (optional)
    tsnr_map = traits.Either(traits.Str, None, desc="tSNR map path")
    qc_mask = traits.Either(traits.Str, None, desc="QC mask (good fit voxels) path")
    r2_map = traits.Either(traits.Str, None, desc="R-squared goodness-of-fit map path")
    quality_flag_map = traits.Either(traits.Str, None, desc="Quality flag map path")
    model_selection_map = traits.Either(traits.Str, None, desc="Model selection map path (BIC mode)")
    delta_bic_map = traits.Either(traits.Str, None, desc="Delta-BIC confidence map path")

    # Tortuosity-corrected output (optional)
    ktrans_tortuosity_corrected = traits.Either(traits.Str, None, desc="Ktrans with tortuosity correction map path")

    # Normalized maps (optional, requires NAWM mask)
    nktrans_map = traits.Either(traits.Str, None, desc="nKtrans (NAWM-normalized Ktrans) map path")
    niauc60_map = traits.Either(traits.Str, None, desc="nIAUC60 (NAWM-normalized IAUC at 60s) map path")

    # Smoothed maps (optional, dict of name->path)
    smooth_pk_maps = traits.Dict(desc="Smoothed PK maps dict")
    smooth_model_free_maps = traits.Dict(desc="Smoothed model-free maps dict")
    smooth_uncertainty_maps = traits.Dict(desc="Smoothed uncertainty std maps dict")
    smooth_tortuosity_maps = traits.Dict(desc="Smoothed tortuosity-corrected maps dict")

    # QC metrics (optional)
    qc_metrics = traits.Dict(desc="Merged QC metrics dictionary")

    # Structural underlays (registered to DCE space, optional)
    t1w_in_dce = traits.Either(traits.Str, None, desc="T1w registered to DCE space NIfTI path")
    flair_in_dce = traits.Either(traits.Str, None, desc="FLAIR registered to DCE space NIfTI path")
    t2w_in_dce = traits.Either(traits.Str, None, desc="T2w registered to DCE space NIfTI path")

    # Structural underlay labeling
    has_postcon_t1w = traits.Bool(False, usedefault=True,
                                  desc="Whether t1w_in_dce is post-contrast CE-T1w")

    # Lesion statistics (conditional on lesion mask, optional)
    lesion_stats_json = traits.Either(traits.Str, None, desc="Lesion statistics JSON file path")
    lesion_mask_dce = traits.Either(traits.Str, None, desc="Lesion mask in DCE space path")

    # ROI statistics (from user-supplied roi_mask, optional)
    roi_stats_json = traits.Either(traits.Str, None, desc="ROI statistics JSON file path")
    roi_mask_dce = traits.Either(traits.Str, None, desc="ROI mask in DCE space path")

    # Anatomical masks in DCE space (optional)
    brain_mask = traits.Either(traits.Str, None, desc="Brain mask in DCE space path")
    nawm_mask = traits.Either(traits.Str, None, desc="NAWM mask in DCE space path")

    # Source DCE file for sidecar
    source_dce = traits.Either(traits.Str, None, desc="Original raw DCE source file path")

    # Processing config for sidecar (optional)
    pk_model = traits.Either(traits.Str, None, desc="Pharmacokinetic model name")
    config_dict = traits.Dict(desc="DCE processing config as dictionary")
    provenance_dict = traits.Dict(desc="Physics provenance metadata for output sidecars")

    # QC plots (optional)
    brain_mask_filmstrip = traits.Either(traits.Str, None, desc="Brain mask QC filmstrip PNG path")
    aif_curve_plot = traits.Either(traits.Str, None, desc="AIF curve QC plot PNG path")
    aif_location_plot = traits.Either(traits.Str, None, desc="AIF location overlay PNG path")
    aif_candidates_plot = traits.Either(traits.Str, None, desc="AIF candidates glass brain PNG path")
    registration_filmstrip = traits.Either(traits.Str, None, desc="Registration QC filmstrip PNG path")
    motion_qc_plot = traits.Either(traits.Str, None, desc="Motion QC plot PNG path")
    b1_qc_plot = traits.Either(traits.Str, None, desc="B1+ mapping QC plot PNG path")
    t1_qc_plot = traits.Either(traits.Str, None, desc="VFA T1 mapping QC plot PNG path")
    preprocessing_qc_plot = traits.Either(traits.Str, None, desc="DCE preprocessing QC plot PNG path")
    pk_modeling_qc_plot = traits.Either(traits.Str, None, desc="PK modeling QC plot PNG path")
    pk_param_maps_qc_plot = traits.Either(traits.Str, None, desc="PK parameter maps QC plot PNG path")
    model_selection_qc_plot = traits.Either(traits.Str, None, desc="Model selection QC plot PNG path")
    dsc_dce_xval_plot = traits.Either(traits.Str, None, desc="DSC-DCE cross-validation QC plot PNG path")


class DCEDerivativeWriterOutputSpec(TraitedSpec):
    """Output specification for DCEDerivativeWriter."""

    out_dir = traits.Str(desc="Path to the output session directory")
    qc_json = traits.Str(desc="Path to the QC JSON file")


class DCEDerivativeWriter(BaseInterface):
    """Write DCE perfusion pipeline outputs with BIDS-compliant naming.

    Output directory structure::

        {output_dir}/pmriprep/
            dataset_description.json
            sub-{subject}/ses-{session}/
                perf/
                    sub-XX_ses-XX_space-DCE_model-{model}_desc-dce_Ktrans.nii.gz
                    sub-XX_ses-XX_space-DCE_model-{model}_desc-dce_Ktrans.json
                    ...
                anat/
                    sub-XX_ses-XX_space-DCE_T1map.nii.gz
                    sub-XX_ses-XX_from-DCE_to-anat_mode-image_xfm.mat
                    ...
                sub-XX_ses-XX_desc-dceQC_quality.json
                sub-XX_ses-XX_space-DCE_desc-lesionStats.json
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

        # Load reference image for voxel sizes (used in sidecars)
        ref_img = nib.load(self.inputs.ktrans_map)
        voxel_sizes = ref_img.header.get_zooms()[:3]  # type: ignore[attr-defined]

        # Get model name for filename
        pk_model = self.inputs.pk_model
        if not isdefined(pk_model) or not pk_model:
            pk_model = "unknown"

        # Build spatial and model entity strings
        spatial = "space-DCE"
        model_entity = f"model-{sanitize_model_entity(pk_model)}"

        # Create directory structure (unified output dir)
        pipeline_dir = output_dir / "pmriprep"
        ses_dir = pipeline_dir / f"sub-{subject}"
        if session:
            ses_dir = ses_dir / f"ses-{session}"
        perf_dir = ses_dir / "perf"
        anat_dir = ses_dir / "anat"

        perf_dir.mkdir(parents=True, exist_ok=True)

        # Remove stale model-specific files from previous runs
        sanitized_model = sanitize_model_entity(pk_model)
        for stale in perf_dir.glob("*_model-*"):
            if f"_model-{sanitized_model}_" not in stale.name:
                logger.warning("Removing stale output: %s", stale.name)
                stale.unlink()

        # Build base sidecar from config (pass resolved pk_model)
        base_sidecar = self._build_base_sidecar(bids_dir, voxel_sizes, pk_model=pk_model)

        # -----------------------------------------------------------
        # 1. Core PK parameter maps
        # -----------------------------------------------------------
        fname = f"{prefix}_{spatial}_{model_entity}_desc-dce_Ktrans"
        copy_derivative(self.inputs.ktrans_map, perf_dir / f"{fname}.nii.gz")
        sidecar = {**base_sidecar, "Description": "Volume transfer constant (1/min)", "Units": "1/min"}
        write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"ve_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dce_ve"
            copy_derivative(self.inputs.ve_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Extravascular extracellular space volume fraction", "Units": "fraction"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"kep_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dce_kep"
            copy_derivative(self.inputs.kep_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Efflux rate constant (1/min)", "Units": "1/min"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 2. Optional vp map (extended Tofts/2CXM only)
        # -----------------------------------------------------------
        if has_input(self.inputs,"vp_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dce_vp"
            copy_derivative(self.inputs.vp_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Plasma volume fraction", "Units": "fraction"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3. Uncertainty maps (Bayesian inference only)
        # -----------------------------------------------------------
        uncertainty_maps = {
            "Ktrans": ("ktrans_uncertainty", "Ktrans uncertainty (posterior std)"),
            "ve": ("ve_uncertainty", "ve uncertainty (posterior std)"),
            "vp": ("vp_uncertainty", "vp uncertainty (posterior std)"),
            "Fp": ("fp_uncertainty", "Fp uncertainty (posterior std, 2CXM only)"),
            "PS": ("ps_uncertainty", "PS uncertainty (posterior std, 2CXM only)"),
        }
        for map_name, (attr, description) in uncertainty_maps.items():
            if has_input(self.inputs,attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_{model_entity}_desc-dceUncertainty_{map_name}"
                copy_derivative(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3b. Coefficient of Variation maps (Bayesian inference only)
        # -----------------------------------------------------------
        cv_maps = {
            "Ktrans": ("ktrans_cv", "Ktrans coefficient of variation (std/mean)"),
            "ve": ("ve_cv", "ve coefficient of variation (std/mean)"),
            "vp": ("vp_cv", "vp coefficient of variation (std/mean)"),
        }
        for map_name, (attr, description) in cv_maps.items():
            if has_input(self.inputs,attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_{model_entity}_desc-dceCV_{map_name}"
                copy_derivative(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description}
                write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

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
                if has_input(self.inputs,attr):
                    src = getattr(self.inputs, attr)
                    bound_type = "dceCI95lower" if "lower" in attr else "dceCI95upper"
                    fname = f"{prefix}_{spatial}_{model_entity}_desc-{bound_type}_{map_name}"
                    copy_derivative(src, perf_dir / f"{fname}.nii.gz")
                    sidecar = {**base_sidecar, "Description": description}
                    write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3d. Model-free maps (IAUC, peak enhancement, TTP, SER)
        # -----------------------------------------------------------
        model_free_maps = {
            "IAUC60": ("iauc60_map", "Initial Area Under Curve at 60s (mM*min)", "mM*min"),
            "IAUC90": ("iauc90_map", "Initial Area Under Curve at 90s (mM*min)", "mM*min"),
            "IAUC120": ("iauc120_map", "Initial Area Under Curve at 120s (mM*min)", "mM*min"),
            "peakEnhancement": ("peak_enhancement_map", "Peak enhancement percentage", "%"),
            "ttp": ("time_to_peak_map", "Time to peak (seconds)", "s"),
            "SER": ("ser_map", "Signal Enhancement Ratio (washout characterization)", "ratio"),
            "washInSlope": ("wash_in_slope_map", "Maximum upslope of concentration during wash-in (mM/min)", "mM/min"),
            "washOutSlope": ("wash_out_slope_map", "Linear washout slope (mM/min, negative=washout)", "mM/min"),
            "BAT": ("bat_map", "Bolus arrival time (seconds)", "s"),
            "curveType": ("curve_type_map", "Curve type classification (1=persistent, 2=plateau, 3=washout)", "categorical"),
        }
        for map_name, (attr, description, units) in model_free_maps.items():
            if has_input(self.inputs, attr):
                src = getattr(self.inputs, attr)
                fname = f"{prefix}_{spatial}_desc-dce_{map_name}"
                copy_derivative(src, perf_dir / f"{fname}.nii.gz")
                sidecar = {**base_sidecar, "Description": description, "Units": units}
                # Enhanced sidecars with method metadata
                if map_name == "washInSlope":
                    sidecar.update({
                        "SmoothingMethod": "Savitzky-Golay",
                        "SmoothingWindow": 5,
                        "SmoothingPolyorder": 2,
                        "InputCurve": "concentration",
                    })
                elif map_name == "washOutSlope":
                    sidecar.update({
                        "PostPeakStartSeconds": 30.0,
                        "PostPeakEndSeconds": 120.0,
                        "MinPoints": 4,
                        "RegressionMethod": "ordinary_least_squares",
                    })
                elif map_name == "BAT":
                    sidecar.update({
                        "DetectionMethod": "threshold_then_gradient_fallback",
                        "ConfidenceThreshold": "high",
                    })
                elif map_name == "curveType":
                    sidecar.update({
                        "PersistentThreshold": 0.9,
                        "WashoutThreshold": 1.1,
                        "ValueMapping": {
                            "0": "unclassified",
                            "1": "persistent",
                            "2": "plateau",
                            "3": "washout",
                        },
                    })
                write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3e. QC maps (tSNR, QC mask, R-squared, quality flag)
        # -----------------------------------------------------------
        if has_input(self.inputs,"tsnr_map"):
            fname = f"{prefix}_{spatial}_desc-dce_tsnr"
            copy_derivative(self.inputs.tsnr_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Temporal signal-to-noise ratio", "Units": "arbitrary"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"qc_mask"):
            fname = f"{prefix}_{spatial}_desc-dceQc_mask"
            copy_derivative(self.inputs.qc_mask, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "QC mask (1=good fit, 0=unfitted/poor)"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"r2_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dce_R2"
            copy_derivative(self.inputs.r2_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "R2 goodness of fit"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"quality_flag_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dceQualityFlag_mask"
            copy_derivative(self.inputs.quality_flag_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "Quality flag map (0=unfitted, 1=good, 2=poor_convergence, 3=high_uncertainty)",
            }
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        _pk_qc = self.inputs.qc_metrics.get("pk_modeling", {}) if isdefined(self.inputs.qc_metrics) and self.inputs.qc_metrics else {}

        if has_input(self.inputs, "model_selection_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dceModelSelection_dseg"
            copy_derivative(self.inputs.model_selection_map, perf_dir / f"{fname}.nii.gz")
            ms_sidecar = {
                **self._build_base_sidecar(bids_dir, voxel_sizes, pk_model=pk_model, minimal=True),
                "Description": "Per-voxel BIC model selection: complexity-ordered encoding",
                "LabelEncoding": {
                    "0": "unfitted",
                    "1": "patlak",
                    "2": "tofts",
                    "3": "extended_tofts",
                    "4": "2cxm",
                },
            }
            _ms_regions = _pk_qc.get("model_selection_regions")
            if _ms_regions:
                ms_sidecar["ModelDistribution"] = _ms_regions
            write_json_sidecar(ms_sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs, "delta_bic_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dceDeltaBIC_stat"
            copy_derivative(self.inputs.delta_bic_map, perf_dir / f"{fname}.nii.gz")
            dbic_sidecar = {
                **self._build_base_sidecar(bids_dir, voxel_sizes, pk_model=pk_model, minimal=True),
                "Description": (
                    "BIC confidence: runner-up minus winner. "
                    ">2 = positive, >6 = strong, >10 = very strong "
                    "(Kass & Raftery 1995)"
                ),
            }
            _r2 = _pk_qc.get("r2_median")
            _conv = _pk_qc.get("convergence_rate")
            if _r2 is not None:
                dbic_sidecar["R2Median"] = _r2
            if _conv is not None:
                dbic_sidecar["ConvergenceRate"] = _conv
            write_json_sidecar(dbic_sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3f. Tortuosity-corrected Ktrans
        # -----------------------------------------------------------
        if has_input(self.inputs,"ktrans_tortuosity_corrected"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dceTortuosityCorr_Ktrans"
            copy_derivative(self.inputs.ktrans_tortuosity_corrected, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "Ktrans corrected for tissue tortuosity using DWI-derived ADC",
            }
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3f2. Normalized maps (nKtrans, nIAUC60) - requires NAWM mask
        # -----------------------------------------------------------
        if has_input(self.inputs,"nktrans_map"):
            fname = f"{prefix}_{spatial}_{model_entity}_desc-dceNawmNorm_Ktrans"
            copy_derivative(self.inputs.nktrans_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "Ktrans normalized by NAWM (normal-appearing white matter) mean",
                "Units": "ratio",
            }
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"niauc60_map"):
            fname = f"{prefix}_{spatial}_desc-dceNawmNorm_IAUC60"
            copy_derivative(self.inputs.niauc60_map, perf_dir / f"{fname}.nii.gz")
            sidecar = {
                **base_sidecar,
                "Description": "IAUC at 60s normalized by NAWM (normal-appearing white matter) mean",
                "Units": "ratio",
            }
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 3g. Smoothed maps (conditional on smoothing being applied)
        # -----------------------------------------------------------
        def _smoothing_sidecar(
            description: str, fwhm_mm: str, extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Build JSON sidecar for a smoothed map with QIBA-recommended fields."""
            s: dict[str, Any] = {
                **base_sidecar,
                "Description": description,
                "SmoothingFWHM": float(fwhm_mm) if fwhm_mm.replace(".", "").isdigit() else fwhm_mm,
                "SmoothingType": "Gaussian",
                "SmoothingMasked": True,
                "IntendedFor": "visualization",
                "Note": "Use non-smoothed maps for quantitative ROI analysis",
            }
            if extra:
                s.update(extra)
            return s

        def _extract_fwhm(src_path: str) -> tuple[str, str]:
            """Extract FWHM string and numeric value from smoothed filename."""
            m = re.search(r"_fwhm(\d+(?:\.\d+)?)", str(src_path))
            if m:
                return f"Sm{m.group(1)}", m.group(1)
            return "SmN", "N"

        _pk_units: dict[str, str] = {
            "ktrans": "1/min", "ve": "fraction", "kep": "1/min", "vp": "fraction",
            "fp": "mL/100g/min", "ps": "mL/100g/min",
        }
        _mf_units: dict[str, str] = {
            "iauc60": "mM*min", "iauc90": "mM*min", "iauc120": "mM*min",
            "peak_enhancement": "%", "ser": "ratio",
            "wash_in_slope": "mM/min", "wash_out_slope": "mM/min",
        }

        smooth_pk = self.inputs.smooth_pk_maps
        if isdefined(smooth_pk) and smooth_pk:
            for map_name, src_path in smooth_pk.items():
                if src_path and Path(src_path).exists():
                    fwhm_str, fwhm_mm = _extract_fwhm(src_path)
                    fname = f"{prefix}_{spatial}_{model_entity}_desc-dce{fwhm_str}_{map_name}"
                    copy_derivative(src_path, perf_dir / f"{fname}.nii.gz")
                    _extra = {"Units": _pk_units[map_name]} if map_name in _pk_units else None
                    sidecar = _smoothing_sidecar(f"Smoothed {map_name} map", fwhm_mm, extra=_extra)
                    write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        smooth_mf = self.inputs.smooth_model_free_maps
        if isdefined(smooth_mf) and smooth_mf:
            for map_name, src_path in smooth_mf.items():
                if src_path and Path(src_path).exists():
                    fwhm_str, fwhm_mm = _extract_fwhm(src_path)
                    fname = f"{prefix}_{spatial}_desc-dce{fwhm_str}_{map_name}"
                    copy_derivative(src_path, perf_dir / f"{fname}.nii.gz")
                    _extra = {"Units": _mf_units[map_name]} if map_name in _mf_units else None
                    sidecar = _smoothing_sidecar(
                        f"Smoothed model-free {map_name} map", fwhm_mm, extra=_extra,
                    )
                    write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # Smoothed uncertainty maps (Bayesian posterior std only)
        smooth_uncertainty = self.inputs.smooth_uncertainty_maps
        if isdefined(smooth_uncertainty) and smooth_uncertainty:
            for map_name, src_path in smooth_uncertainty.items():
                if src_path and Path(src_path).exists():
                    fwhm_str, fwhm_mm = _extract_fwhm(src_path)
                    param = map_name.replace("_std", "")
                    fname = f"{prefix}_{spatial}_{model_entity}_desc-dceUncertainty{fwhm_str}_{param}"
                    copy_derivative(src_path, perf_dir / f"{fname}.nii.gz")
                    _extra = {"Units": _pk_units[param]} if param in _pk_units else None
                    sidecar = _smoothing_sidecar(
                        f"Smoothed {map_name} uncertainty map", fwhm_mm, extra=_extra,
                    )
                    write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # Smoothed tortuosity-corrected maps
        smooth_tort = self.inputs.smooth_tortuosity_maps
        if isdefined(smooth_tort) and smooth_tort:
            for map_name, src_path in smooth_tort.items():
                if src_path and Path(src_path).exists():
                    fwhm_str, fwhm_mm = _extract_fwhm(src_path)
                    fname = f"{prefix}_{spatial}_{model_entity}_desc-dceTortuosityCorr{fwhm_str}_Ktrans"
                    copy_derivative(src_path, perf_dir / f"{fname}.nii.gz")
                    sidecar = _smoothing_sidecar(
                        "Smoothed Ktrans corrected for tissue tortuosity", fwhm_mm,
                        extra={"Units": "1/min"},
                    )
                    write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 4. T1/B1 mapping outputs
        # -----------------------------------------------------------
        minimal_sidecar = self._build_base_sidecar(
            bids_dir, voxel_sizes, pk_model=pk_model, minimal=True,
        )

        if has_input(self.inputs,"t1_map"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_T1map"
            copy_derivative(self.inputs.t1_map, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "T1 relaxation time map (ms)", "Units": "ms"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        if has_input(self.inputs,"b1_map"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_B1plusmap"
            copy_derivative(self.inputs.b1_map, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "B1+ transmit field map (ratio)", "Units": "ratio"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 4b. Anatomical masks in DCE space
        # -----------------------------------------------------------
        if has_input(self.inputs,"brain_mask"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_desc-brain_mask"
            copy_derivative(self.inputs.brain_mask, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "Brain mask in DCE space", "SkullStripped": True}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        if has_input(self.inputs,"nawm_mask"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_desc-nawm_mask"
            copy_derivative(self.inputs.nawm_mask, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "Normal-appearing white matter mask in DCE space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 5. Motion-corrected DCE and reference
        # -----------------------------------------------------------
        if has_input(self.inputs,"dce_moco"):
            fname = f"{prefix}_{spatial}_desc-dcePreproc_dce"
            copy_derivative(self.inputs.dce_moco, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Motion-corrected 4D DCE time series"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"dce_mean"):
            fname = f"{prefix}_{spatial}_desc-dceMean_ref"
            copy_derivative(self.inputs.dce_mean, perf_dir / f"{fname}.nii.gz")
            sidecar = {**base_sidecar, "Description": "Mean DCE reference image"}
            write_json_sidecar(sidecar, perf_dir / f"{fname}.json")

        if has_input(self.inputs,"motion_params"):
            from pmriprep.utils.outputs import (
                par_to_confounds_tsv,
                write_confounds_sidecar,
            )

            confounds_stem = f"{prefix}_desc-dceConfounds_timeseries"
            par_to_confounds_tsv(
                self.inputs.motion_params,
                perf_dir / f"{confounds_stem}.tsv",
            )
            write_confounds_sidecar(perf_dir / f"{confounds_stem}.json")

        # -----------------------------------------------------------
        # 6. AIF outputs
        # -----------------------------------------------------------
        if has_input(self.inputs,"aif_curve"):
            copy_derivative(
                self.inputs.aif_curve,
                perf_dir / f"{prefix}_desc-dceAif_timeseries.txt",
            )

        if has_input(self.inputs,"aif_mask"):
            copy_derivative(
                self.inputs.aif_mask,
                perf_dir / f"{prefix}_{spatial}_desc-dceAif_mask.nii.gz",
            )

        # -----------------------------------------------------------
        # 7. Registration transform (written to anat/)
        # -----------------------------------------------------------
        if has_input(self.inputs,"dce_to_t1w_xfm"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            copy_derivative(
                self.inputs.dce_to_t1w_xfm,
                anat_dir / f"{prefix}_from-DCE_to-anat_mode-image_xfm.mat",
            )

        # -----------------------------------------------------------
        # 8. Structural underlays (in DCE space)
        # -----------------------------------------------------------
        if has_input(self.inputs,"t1w_in_dce"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            if self.inputs.has_postcon_t1w:
                label = "ce-GAD_T1w"
                description = "CE-T1w registered to DCE space"
            else:
                label = "T1w"
                description = "T1w registered to DCE space"
            fname = f"{prefix}_{spatial}_{label}"
            copy_derivative(self.inputs.t1w_in_dce, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": description}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        if has_input(self.inputs,"flair_in_dce"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_FLAIR"
            copy_derivative(self.inputs.flair_in_dce, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "FLAIR registered to DCE space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        if has_input(self.inputs,"t2w_in_dce"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_T2w"
            copy_derivative(self.inputs.t2w_in_dce, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "T2w registered to DCE space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 9. Lesion statistics and mask (written to session root / anat)
        # -----------------------------------------------------------
        if has_input(self.inputs,"lesion_stats_json"):
            copy_derivative(
                self.inputs.lesion_stats_json,
                ses_dir / f"{prefix}_{spatial}_desc-lesionStats.json",
            )

        if has_input(self.inputs,"lesion_mask_dce"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_desc-lesion_mask"
            copy_derivative(self.inputs.lesion_mask_dce, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "Lesion mask registered to DCE space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 9b. ROI statistics and mask (stats extraction only)
        # -----------------------------------------------------------
        if has_input(self.inputs, "roi_stats_json"):
            copy_derivative(
                self.inputs.roi_stats_json,
                ses_dir / f"{prefix}_{spatial}_desc-roiStats.json",
            )

        if has_input(self.inputs, "roi_mask_dce"):
            anat_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix}_{spatial}_desc-roi_mask"
            copy_derivative(self.inputs.roi_mask_dce, anat_dir / f"{fname}.nii.gz")
            sidecar = {**minimal_sidecar, "Description": "ROI mask registered to DCE space"}
            write_json_sidecar(sidecar, anat_dir / f"{fname}.json")

        # -----------------------------------------------------------
        # 10. QC outputs (written to session root)
        # -----------------------------------------------------------
        # Collect processing warnings for missing data-dependent inputs
        _data_dependent_inputs = {
            "b1_map": "B1+ transmit field map",
            "t1_map": "T1 relaxation time map",
            "brain_mask": "Brain mask in DCE space",
            "nawm_mask": "NAWM mask in DCE space",
            "lesion_mask_dce": "Lesion mask in DCE space",
            "dce_to_t1w_xfm": "DCE to T1w registration transform",
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
            qc_metrics = dict(qc_metrics)
            if warnings:
                qc_metrics["warnings"] = warnings
            # Add provenance warnings (structurally separate from data warnings)
            provenance = self.inputs.provenance_dict
            if isdefined(provenance) and provenance:
                from pmriprep.utils.contrast_agent import collect_default_warnings
                prov_warnings = collect_default_warnings(provenance, "DCE")
                if prov_warnings:
                    qc_metrics["provenance_warnings"] = prov_warnings
            enhanced_qc = self._add_qc_recommendation(qc_metrics)
            qc_json_path = ses_dir / f"{prefix}_desc-dceQC_quality.json"
            write_json_sidecar(enhanced_qc, qc_json_path)
            self._qc_json = str(qc_json_path)

        # Copy QC figures to figures/ with BIDS names
        from pmriprep.utils.report_utils import figure_filename

        figures_dir = ses_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Clean stale AIF location figure from prior runs when not generated this run
        aif_detection_method = self.inputs.qc_metrics.get(
            "aif_detection", {}
        ).get("detection_method", "unknown") if isdefined(qc_metrics) and qc_metrics else "unknown"
        if aif_detection_method in ("population", "external"):
            stale = figures_dir / figure_filename(
                subject, session, "aif_location", "dce"
            )
            if stale.exists():
                logger.info("Removing stale AIF location figure (method=%s): %s", aif_detection_method, stale.name)
                stale.unlink()

        _fig_inputs = {
            "dce_brain_mask": "brain_mask_filmstrip",
            "motion": "motion_qc_plot",
            "aif_curve": "aif_curve_plot",
            "aif_location": "aif_location_plot",
            "aif_candidates": "aif_candidates_plot",
            "registration": "registration_filmstrip",
            "b1": "b1_qc_plot",
            "t1": "t1_qc_plot",
            "preprocessing_qc": "preprocessing_qc_plot",
            "pk_modeling_qc": "pk_modeling_qc_plot",
            "pk_param_maps_qc": "pk_param_maps_qc_plot",
            "model_selection_qc": "model_selection_qc_plot",
            "dsc_dce_xval": "dsc_dce_xval_plot",
        }
        for fig_key, attr in _fig_inputs.items():
            if has_input(self.inputs,attr):
                fname = figure_filename(subject, session, fig_key, "dce")
                copy_derivative(getattr(self.inputs, attr), figures_dir / fname)

        # -----------------------------------------------------------
        # 11. Dataset description (shared utility)
        # -----------------------------------------------------------
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
        self,
        bids_dir: str,
        voxel_sizes: tuple[float, ...],
        pk_model: str | None = None,
        minimal: bool = False,
    ) -> dict[str, Any]:
        """Build base JSON sidecar with processing metadata.

        Args:
            bids_dir: Path to the source BIDS dataset root.
            voxel_sizes: Voxel dimensions (x, y, z) from reference image header.
            pk_model: Resolved pharmacokinetic model name. If provided, used
                instead of config_dict value (which may be "auto").
            minimal: If True, omit perfusion-specific fields
                (PharmacokineticModel, BayesianInference).
                Use for masks, structural underlays, and non-perfusion maps.

        Returns:
            Dictionary with shared sidecar fields.
        """
        sidecar: dict[str, Any] = {
            **get_bids_derivative_provenance(),
            "SpatialReference": "orig",
            "Resolution": [float(v) for v in voxel_sizes[:3]],
        }

        # Add source file reference
        source_dce = getattr(self.inputs, "source_dce", None)
        if source_dce and isdefined(source_dce) and source_dce:
            try:
                bids_root = Path(bids_dir)
                src_path = Path(source_dce)
                if src_path.is_relative_to(bids_root):
                    rel = src_path.relative_to(bids_root)
                    sidecar["Sources"] = [f"bids::{rel}"]
                else:
                    sidecar["Sources"] = [str(source_dce)]
            except (ValueError, TypeError):
                sidecar["Sources"] = [str(source_dce)]

        # Add physics provenance (PerfusionProcessing namespace)
        provenance = self.inputs.provenance_dict
        if isdefined(provenance) and provenance:
            sidecar.update(provenance)

        # Enrich with dynamic QC data (only for perfusion maps)
        if not minimal:
            qc = self.inputs.qc_metrics
            if isdefined(provenance) and provenance and isdefined(qc) and qc:
                from pmriprep.utils.provenance import enrich_provenance_from_qc
                enriched = enrich_provenance_from_qc(sidecar, qc, "DCE")
                sidecar.update(enriched)

            sidecar["KtransConvention"] = "plasma_Tofts1999"
            hct_used = None
            if isdefined(qc) and qc:
                hct_used = qc.get("aif_detection", {}).get("hematocrit")
            if hct_used is not None:
                sidecar["KtransConventionHct"] = float(hct_used)

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

        # Evaluate AIF quality if available
        aif_qc = qc_metrics.get("aif_detection", {})
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

        # Evaluate physiological plausibility of Ktrans using mean (not median).
        ktrans_mean = pk_qc.get("ktrans_mean")
        if ktrans_mean is not None:
            if 0.0001 < ktrans_mean < 1.0:
                kt_rec = QCRecommendation(QCAction.INCLUDE, f"Ktrans physiologically plausible (mean={ktrans_mean:.4f})", 0)
            else:
                kt_rec = QCRecommendation(QCAction.REVIEW, f"Ktrans outside typical range (mean={ktrans_mean:.4f})", 1)
            recommendations.append(kt_rec)

        # Evaluate motion QC using composite motion_status
        preprocessing_qc = qc_metrics.get("preprocessing", {})
        motion_status = preprocessing_qc.get("motion_status")
        if motion_status is not None:
            fd_mean = preprocessing_qc.get("fd_mean")
            fd_max = preprocessing_qc.get("fd_max")
            pct_outliers = preprocessing_qc.get("percent_outliers")
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
                motion_rec = QCRecommendation(QCAction.REVIEW, f"Excessive motion {detail}", 2)
            else:
                motion_rec = None
            if motion_rec is not None:
                recommendations.append(motion_rec)

        # Evaluate signal quality (peak enhancement, baseline tSNR, B1+ bias)
        from pmriprep.utils.dce_constants import (
            QC_B1_MEAN_HIGH,
            QC_B1_MEAN_LOW,
            QC_DCE_BASELINE_TSNR_MIN,
            QC_DCE_BASELINE_TSNR_WARN,
            QC_PEAK_ENHANCEMENT_MIN,
            QC_PEAK_ENHANCEMENT_WARN,
        )

        peak_enh = preprocessing_qc.get("peak_enh_pct")
        baseline_tsnr = preprocessing_qc.get("baseline_tsnr")
        aif_source = aif_qc.get("aif_source", "")

        # Peak enhancement gating
        if peak_enh is not None:
            if peak_enh < QC_PEAK_ENHANCEMENT_MIN:
                enh_rec = QCRecommendation(
                    QCAction.EXCLUDE,
                    f"Peak enhancement critically low ({peak_enh:.1f}% < {QC_PEAK_ENHANCEMENT_MIN}%, possible failed injection)",
                    3,
                )
                recommendations.append(enh_rec)
            elif peak_enh < QC_PEAK_ENHANCEMENT_WARN:
                enh_rec = QCRecommendation(
                    QCAction.REVIEW,
                    f"Peak enhancement marginal ({peak_enh:.1f}%)",
                    1,
                )
                recommendations.append(enh_rec)
                # Population AIF cannot verify injection quality
                if aif_source == "population":
                    recommendations.append(
                        QCRecommendation(
                            QCAction.REVIEW,
                            "Injection quality cannot be verified with population AIF",
                            1,
                        )
                    )

        # Baseline tSNR gating
        if baseline_tsnr is not None:
            if baseline_tsnr < QC_DCE_BASELINE_TSNR_MIN:
                tsnr_rec = QCRecommendation(
                    QCAction.EXCLUDE,
                    f"Baseline tSNR critically low ({baseline_tsnr:.1f} < {QC_DCE_BASELINE_TSNR_MIN})",
                    3,
                )
                recommendations.append(tsnr_rec)
            elif baseline_tsnr < QC_DCE_BASELINE_TSNR_WARN:
                tsnr_rec = QCRecommendation(
                    QCAction.REVIEW,
                    f"Baseline tSNR marginal ({baseline_tsnr:.1f})",
                    1,
                )
                recommendations.append(tsnr_rec)

        # Evaluate B1+ bias if available
        b1_qc = qc_metrics.get("b1_mapping", {})
        mean_b1 = b1_qc.get("mean_b1")
        if mean_b1 is not None:
            if mean_b1 < QC_B1_MEAN_LOW or mean_b1 > QC_B1_MEAN_HIGH:
                b1_rec = QCRecommendation(
                    QCAction.REVIEW,
                    f"B1+ bias outside expected range (mean={mean_b1:.2f}, expected {QC_B1_MEAN_LOW}-{QC_B1_MEAN_HIGH})",
                    1,
                )
                recommendations.append(b1_rec)

        # Evaluate DSC-DCE cross-validation if available
        xval = qc_metrics.get("dsc_dce_validation", {})
        if (
            not xval.get("skipped", False)
            and not xval.get("circular", False)
        ):
            xval_status = xval.get("status")
            xval_metrics = xval.get("metrics", {})
            corr = xval_metrics.get("correlation", 0.0)
            if xval_status == QCStatus.WARN:
                preload = xval.get("preload_effect_detected", False)
                detail = (
                    "low vp/CBV ratio"
                    if preload
                    else "ratio outside range"
                )
                recommendations.append(
                    QCRecommendation(
                        QCAction.REVIEW,
                        f"DSC-DCE: {detail} (r={corr:.2f})",
                        1,
                    )
                )
            elif xval_status == QCStatus.PASS:
                recommendations.append(
                    QCRecommendation(
                        QCAction.INCLUDE,
                        f"DSC-DCE validation passed (r={corr:.2f})",
                        0,
                    )
                )

        # Aggregate recommendations
        overall = aggregate_qc_recommendations(recommendations)
        enhanced["recommendation"] = {
            "action": overall.action,
            "reason": overall.reason,
            "severity": overall.severity,
        }

        return enhanced
