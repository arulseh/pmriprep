"""DSC (Dynamic Susceptibility Contrast) preprocessing workflow for PMRIPrep.

This module implements the main DSC perfusion processing workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Any, Optional, Union

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]
from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]


from pmriprep.config.dsc import DSCConfig
from pmriprep.data import get_arterial_atlas, get_arterial_atlas_cascade
from pmriprep.interfaces.common import Reorient
from pmriprep.interfaces.dsc import MotionCorrection, DSCRegistration
from pmriprep.interfaces.dsc_aif import AIFDetection
from pmriprep.interfaces.dsc_aif_qc import AIFQCPlot
from pmriprep.interfaces.dsc_compute import ComputeDSCMaps

# FSL CommandLine wrappers, Function-node helpers, and the (formerly embedded)
# smoothing dict-builders now live in ``_helpers.py``; re-exported here so
# ``init_dsc_preproc_wf`` and external importers keep resolving them from
# ``...dsc.base``.
from pmriprep.workflows.dsc._helpers import (  # noqa: F401
    ConvertXFM,
    FLIRTApplyXFM,
    _build_auc_corr_dict,
    _build_hemodynamic_dict,
    _compute_prebolus_mean,
    _create_perfusion_brain_mask_filmstrip,
    _create_registration_filmstrip,
    _detect_and_correct_preload,
    _extract_preload_class,
    _fsl_to_x5_node,
    _generate_dsc_fit_qc,
    _generate_dsc_leakage_qc,
    _generate_dsc_param_maps_qc,
    _generate_dsc_preprocessing_qc,
    _generate_motion_qc_plot,
    _histogram_match_t1w_to_dsc,
    _merge_qc_metrics,
    _normalize_cbf,
    _normalize_cbv,
    _select_aif_exclusion_mask,
    _signal_to_concentration,
    _smooth_maps,
    _transform_brain_mask_to_dsc,
    _transform_lesion_mask_to_dsc,
    _transform_nawm_to_dsc,
    _transform_structural_to_dsc,
    _validate_native_mask,
)



def init_dsc_concentration_aif_wf(
    *,
    baseline_frames: int,
    aif_method: str,
    n_candidates: int,
    lesion_distance: float,
    disable_fallback: bool,
    dose: float,
    agent_standard_dose: float,
    gd_multiplicity: int,
    external_aif_file: Optional[str],
    aif_mask_file: Optional[str],
    mem_gb: float,
    name: str = "dsc_conc_aif_wf",
) -> pe.Workflow:
    """Build the DSC concentration + AIF-detection sub-workflow.

    Converts the motion-corrected DSC series to ΔR2* concentration (s⁻¹) and
    detects the arterial input function.

    Parameters
    ----------
    baseline_frames : int
        Pre-bolus baseline frames (S0 estimation; also the AIF baseline).
    aif_method : str
        AIF detection method ("auto" or "synthetic").
    n_candidates : int
        Number of AIF candidate voxels to average.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
    disable_fallback : bool
        Disable the synthetic-AIF fallback when detection fails.
    dose, agent_standard_dose : float
        Contrast dose multiplier and the agent's standard dose (mmol Gd/kg).
    gd_multiplicity : int
        Gadolinium multiplicity for the contrast agent.
    external_aif_file : str or None
        External AIF curve path (bypasses detection) or None.
    aif_mask_file : str or None
        User-supplied AIF mask path (overrides atlas mask) or None.
    mem_gb : float
        Memory estimate (GB) for the heavy signal-conversion node.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The concentration/AIF sub-workflow.

    Inputs
    ------
    dsc_moco
        Motion-corrected DSC 4D series.
    te, tr, field_strength
        Acquisition parameters.
    brain_mask_file
        DSC-space brain mask.
    lesion_mask_file, enhancing_mask_file, atlas_mask_file, full_atlas_mask_file
        AIF exclusion / search masks (``full_atlas_mask_file`` is fed by the
        parent only when the posterior-circulation atlas tier is installed; an
        unfed inputnode field collapses to a no-op).

    Outputs
    -------
    concentration_file, signal_file
        ΔR2* concentration and raw signal.
    out_aif_curve, out_measured_aif_curve, out_aif_mask, out_aif_coord,
    detection_level, aif_baseline_frames, qc_metrics, out_raw_aif_curve,
    gamma_fit_applied
        AIF detection products and QC.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dsc_moco",
                "te",
                "tr",
                "field_strength",
                "brain_mask_file",
                "lesion_mask_file",
                "enhancing_mask_file",
                "atlas_mask_file",
                "full_atlas_mask_file",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "concentration_file",
                "signal_file",
                "out_aif_curve",
                "out_measured_aif_curve",
                "out_aif_mask",
                "out_aif_coord",
                "detection_level",
                "aif_baseline_frames",
                "qc_metrics",
                "out_raw_aif_curve",
                "gamma_fit_applied",
            ]
        ),
        name="outputnode",
    )

    # Node 2: Signal to Concentration Conversion (always ΔR2*, s⁻¹)
    signal_to_conc = pe.Node(
        niu.Function(
            input_names=["dsc_file", "te", "baseline_frames"],
            output_names=["concentration_file", "signal_file"],
            function=_signal_to_concentration,
        ),
        name="signal_to_conc",
        mem_gb=mem_gb,
    )
    signal_to_conc.inputs.baseline_frames = baseline_frames

    # Node 3: AIF Detection (uses uncorrected concentration; arterial BBB intact)
    aif_detect = pe.Node(
        AIFDetection(
            aif_method=aif_method,
            criteria="strict",
            n_candidates=n_candidates,
            lesion_distance=lesion_distance,
            disable_fallback=disable_fallback,
            baseline_frames=baseline_frames,
            dose=dose,
            agent_standard_dose=agent_standard_dose,
            gd_multiplicity=gd_multiplicity,
        ),
        name="aif_detect",
    )
    if external_aif_file is not None:
        aif_detect.inputs.external_aif_file = external_aif_file
    if aif_mask_file is not None:
        aif_detect.inputs.aif_mask_file = aif_mask_file

    workflow.connect([
        (inputnode, signal_to_conc, [
            ("dsc_moco", "dsc_file"),
            ("te", "te"),
        ]),
        (signal_to_conc, aif_detect, [
            ("concentration_file", "concentration_file"),
            ("signal_file", "signal_file"),
        ]),
        (inputnode, aif_detect, [
            ("tr", "tr"),
            ("field_strength", "field_strength"),
            ("brain_mask_file", "brain_mask_file"),
            ("lesion_mask_file", "lesion_mask_file"),
            ("enhancing_mask_file", "enhancing_mask_file"),
            ("atlas_mask_file", "atlas_mask_file"),
            ("full_atlas_mask_file", "full_atlas_mask_file"),
        ]),
        (signal_to_conc, outputnode, [
            ("concentration_file", "concentration_file"),
            ("signal_file", "signal_file"),
        ]),
        (aif_detect, outputnode, [
            ("out_aif_curve", "out_aif_curve"),
            ("out_measured_aif_curve", "out_measured_aif_curve"),
            ("out_aif_mask", "out_aif_mask"),
            ("out_aif_coord", "out_aif_coord"),
            ("detection_level", "detection_level"),
            ("aif_baseline_frames", "aif_baseline_frames"),
            ("qc_metrics", "qc_metrics"),
            ("out_raw_aif_curve", "out_raw_aif_curve"),
            ("gamma_fit_applied", "gamma_fit_applied"),
        ]),
    ])
    return workflow


def init_dsc_motion_wf(
    *,
    motion_thresh: float,
    baseline_frames: int,
    subject_id: str,
    session_id: Optional[str],
    mem_gb: float,
    name: str = "dsc_motion_wf",
) -> pe.Workflow:
    """Build the DSC motion-correction sub-workflow.

    Reorients the raw DSC series to RAS, runs motion correction, and computes a
    pre-bolus mean reference (used downstream for brain extraction, registration,
    and mask/atlas resampling), plus a motion-QC plot.

    Parameters
    ----------
    motion_thresh : float
        Framewise-displacement threshold (mm) for motion outliers.
    baseline_frames : int
        Number of pre-bolus baseline frames for the pre-bolus mean.
    subject_id : str
        BIDS subject label (for the motion-QC plot annotation).
    session_id : str or None
        BIDS session label (for the motion-QC plot annotation).
    mem_gb : float
        Memory estimate (GB) for the heavy motion-correction node.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The DSC motion sub-workflow.

    Inputs
    ------
    dsc_file
        Raw DSC 4D timeseries.
    tr
        Repetition time (s), for the motion-QC plot.

    Outputs
    -------
    dsc_moco
        Motion-corrected DSC 4D series.
    dsc_mean
        Mean of the motion-corrected series.
    motion_params
        Per-volume motion parameters.
    fd_values
        Framewise-displacement values.
    motion_qc_metrics
        Motion QC metrics.
    prebolus_mean_file
        Pre-bolus mean reference volume.
    motion_qc_plot_file
        Motion-QC plot (PNG).
    motion_summary
        Motion summary metrics for the merged QC.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["dsc_file", "tr"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dsc_moco",
                "dsc_mean",
                "motion_params",
                "fd_values",
                "motion_qc_metrics",
                "prebolus_mean_file",
                "motion_qc_plot_file",
                "motion_summary",
            ]
        ),
        name="outputnode",
    )

    # Node 0: Reorient DSC to RAS orientation
    dsc_reorient = pe.Node(
        Reorient(),
        name="dsc_reorient",
    )

    # Node 1: Motion Correction
    motion_correct = pe.Node(
        MotionCorrection(
            ref_vol="mean",
            motion_thresh=motion_thresh,
        ),
        name="motion_correct",
        mem_gb=mem_gb,
    )

    # Node 1a: Motion QC Plot
    motion_qc_plot = pe.Node(
        niu.Function(
            input_names=["motion_params_file", "tr", "subject_id", "session_id"],
            output_names=["plot_file", "motion_summary"],
            function=_generate_motion_qc_plot,
        ),
        name="motion_qc_plot",
    )
    motion_qc_plot.inputs.subject_id = subject_id
    motion_qc_plot.inputs.session_id = session_id or ""

    # Node 1b: Pre-Bolus Mean Reference
    prebolus_mean = pe.Node(
        niu.Function(
            input_names=["moco_file", "baseline_frames"],
            output_names=["prebolus_mean_file"],
            function=_compute_prebolus_mean,
        ),
        name="prebolus_mean",
    )
    prebolus_mean.inputs.baseline_frames = baseline_frames

    workflow.connect([
        (inputnode, dsc_reorient, [("dsc_file", "in_file")]),
        (dsc_reorient, motion_correct, [("out_file", "in_file")]),
        (motion_correct, prebolus_mean, [("out_file", "moco_file")]),
        (motion_correct, motion_qc_plot, [("out_params", "motion_params_file")]),
        (inputnode, motion_qc_plot, [("tr", "tr")]),
        (motion_correct, outputnode, [
            ("out_file", "dsc_moco"),
            ("mean_img", "dsc_mean"),
            ("out_params", "motion_params"),
            ("fd", "fd_values"),
            ("qc_metrics", "motion_qc_metrics"),
        ]),
        (prebolus_mean, outputnode, [("prebolus_mean_file", "prebolus_mean_file")]),
        (motion_qc_plot, outputnode, [
            ("plot_file", "motion_qc_plot_file"),
            ("motion_summary", "motion_summary"),
        ]),
    ])
    return workflow


def init_dsc_preload_wf(
    *,
    preload: Union[str, float],
    agent_standard_dose: float,
    baseline_frames: int,
    leakage_method: str,
    kep: Optional[float],
    mem_gb: float,
    name: str = "dsc_preload_wf",
) -> pe.Workflow:
    """Build the DSC preload-detection + K2 leakage-correction sub-workflow.

    Detects/corrects contrast preload and computes the K2 leakage map (which
    requires the AIF), then extracts the preload classification.

    Parameters
    ----------
    preload : "auto" or float
        Administered preload dose multiplier ("auto" to detect).
    agent_standard_dose : float
        Agent's standard dose (mmol Gd/kg).
    baseline_frames : int
        Pre-bolus baseline frames.
    leakage_method : str
        K2 leakage-correction method ("bsw" or "bidirectional").
    kep : float or None
        EES efflux rate (s⁻¹) for bidirectional correction, or None.
    mem_gb : float
        Memory estimate (GB) for the heavy correction node.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The preload/K2 sub-workflow.

    Inputs
    ------
    concentration_file, signal_file, aif_file, brain_mask_file, te,
    field_strength, tr
        ΔR2* concentration + raw signal, AIF curve, DSC brain mask, and
        acquisition parameters.

    Outputs
    -------
    corrected_concentration_file
        Leakage-corrected concentration.
    k2_map_file
        K2 leakage map.
    qc_dict
        Preload/leakage QC.
    preload_classification
        Discrete preload class (for K2 uncertainty tuning downstream).
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "concentration_file",
                "signal_file",
                "aif_file",
                "brain_mask_file",
                "te",
                "field_strength",
                "tr",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "corrected_concentration_file",
                "k2_map_file",
                "qc_dict",
                "preload_classification",
            ]
        ),
        name="outputnode",
    )

    # Node 4: Preload Detection and K2 Correction (needs the AIF)
    preload_correct = pe.Node(
        niu.Function(
            input_names=[
                "concentration_file",
                "aif_file",
                "brain_mask_file",
                "te",
                "field_strength",
                "tr",
                "preload",
                "signal_file",
                "baseline_frames",
                "leakage_method",
                "kep",
                "agent_standard_dose",
            ],
            output_names=["corrected_concentration_file", "k2_map_file", "qc_dict"],
            function=_detect_and_correct_preload,
        ),
        name="preload_correct",
        mem_gb=mem_gb,
    )
    preload_correct.inputs.preload = preload
    preload_correct.inputs.agent_standard_dose = agent_standard_dose
    preload_correct.inputs.baseline_frames = baseline_frames
    preload_correct.inputs.leakage_method = leakage_method
    preload_correct.inputs.kep = kep

    # Node 4b: Extract preload classification from QC dict
    extract_preload = pe.Node(
        niu.Function(
            input_names=["qc_dict"],
            output_names=["preload_classification"],
            function=_extract_preload_class,
        ),
        name="extract_preload",
    )

    workflow.connect([
        (inputnode, preload_correct, [
            ("concentration_file", "concentration_file"),
            ("signal_file", "signal_file"),
            ("aif_file", "aif_file"),
            ("brain_mask_file", "brain_mask_file"),
            ("te", "te"),
            ("field_strength", "field_strength"),
            ("tr", "tr"),
        ]),
        (preload_correct, extract_preload, [("qc_dict", "qc_dict")]),
        (preload_correct, outputnode, [
            ("corrected_concentration_file", "corrected_concentration_file"),
            ("k2_map_file", "k2_map_file"),
            ("qc_dict", "qc_dict"),
        ]),
        (extract_preload, outputnode, [("preload_classification", "preload_classification")]),
    ])
    return workflow


def init_dsc_preproc_wf(
    bids_dir: Union[str, Path],
    output_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[DSCConfig] = None,
    random_seed: Optional[int] = None,
    mem_gb: Optional[float] = None,
    mem_limit_gb: Optional[float] = None,
    gd_multiplicity: int = 1,
    name: str = "dsc_preproc_wf",
) -> pe.Workflow:
    """Initialize DSC perfusion preprocessing workflow.

    This workflow performs DSC perfusion processing including motion correction,
    signal-to-concentration conversion, AIF detection, deconvolution, and
    perfusion parameter extraction.

    Parameters
    ----------
    bids_dir : Path or str
        Path to BIDS dataset root directory.
    output_dir : Path or str
        Path to output derivatives directory (e.g., /project/derivatives).
    subject_id : str
        BIDS subject identifier (without 'sub-' prefix).
    session_id : str, optional
        BIDS session identifier (without 'ses-' prefix).
    config : DSCConfig, optional
        DSC processing configuration. Uses defaults if not provided.
    name : str
        Workflow name (default: "dsc_preproc_wf").

    Returns
    -------
    pe.Workflow
        Nipype workflow for DSC perfusion processing.

    Inputs
    ------
    dsc_file
        Path to raw 4D DSC perfusion image.
    t1w_reference
        Path to T1-weighted reference image for registration.
    t1w_underlay
        T1w image to transform to DSC space as anatomical underlay.
        CE-T1w if post-contrast available, otherwise native T1w.
    brain_mask
        Path to brain mask in T1w space.
    wm_probseg
        Path to white matter probability map for NAWM mask.
    nawm_mask
        Path to pre-computed NAWM mask for CBV normalization.
    te
        Echo time in seconds (default from config or 0.03).
    field_strength
        Magnetic field strength in Tesla (default 3.0).
    tr
        Repetition time in seconds (default 1.5).

    Outputs
    -------
    dsc_moco
        Path to motion-corrected DSC image.
    dsc_mean
        Path to mean DSC image across time.
    motion_params
        Path to motion parameters file.
    rel_cbf_map, rel_cbv_map
        Relative cerebral blood flow/volume maps (always produced).
    rel_auc_raw_map, rel_auc_corr_map
        Relative AUC maps (always produced).
    mtt_map, tmax_map
        Mean transit time and time-to-maximum maps.
    psr_map, ph_map
        Percent signal recovery and peak height maps.
    ttp_map, cnr_map
        Time-to-peak and contrast-to-noise ratio maps.
    auc_diff_map
        AUC leakage contribution (raw - corrected).
    snr_map, tsnr_map
        SNR (baseline) and tSNR (temporal) quality metric maps.
    aif_curve
        Path to detected/used AIF curve.
    aif_mask
        Path to AIF detection mask.
    k2_map
        Path to K2 leakage correction map.
    ncbv_map, ncbf_map
        Normalized CBV/CBF maps (if NAWM mask available).
    smooth_quant_indep, smooth_rel, smooth_abs
        Smoothed map dictionaries (if smooth_fwhm > 0).
    dsc_to_t1w_xfm
        Path to DSC to T1w transform.
    qc_metrics
        Dictionary of QC metrics from processing steps.

    Notes
    -----
    The workflow follows this processing flow:
    1. Motion correction (mcflirt)
    2. Signal-to-concentration conversion (ΔR2*)
    3. AIF detection (atlas-based → automatic → synthetic fallback)
    4. Preload detection and K2 leakage correction (if needed)
    5. Deconvolution (method from config: sSVD, oSVD, tikhonov, vascular, cpi)
    6. Perfusion parameter extraction (CBF, CBV, MTT, Tmax, PSR, PH)
    7. Registration to T1w space
    8. CBV normalization (if NAWM mask available)

    See Also
    --------
    pmriprep.config.dsc.DSCConfig : Configuration options for DSC processing.
    """
    # Use default config if not provided
    if config is None:
        config = DSCConfig()

    # Store parameters for use in workflow nodes
    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)
    derivatives_dir = output_dir       # interfaces that append "pmriprep" internally

    # Create workflow
    workflow = pe.Workflow(name=name)

    # -------------------------------------------------------------------
    # Input Node
    # Define all possible inputs for DSC processing
    # -------------------------------------------------------------------
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dsc_file",  # Raw 4D DSC perfusion image
                "t1w_reference",  # T1w reference for registration
                "t1w_underlay",  # precontrast (or single-source) T1w underlay
                "t1w_postcon_underlay",  # postcontrast T1w underlay (independent of t1w_underlay)
                "brain_mask",  # Brain mask in T1w space
                "wm_probseg",  # White matter probability map (optional)
                "whole_tumor_mask",  # Whole-tumor mask for AIF exclusion + lesion stats (optional, T1w space)
                "enhancing_mask",  # Enhancing tumor mask for laterality (optional, T1w space)
                "tumor_segmentation",  # Multi-label tumor segmentation for lesion stats (optional, T1w space)
                "user_lesion_mask",  # User-supplied lesion mask in DSC native space (optional)
                "nawm_mask",  # NAWM mask for normalization (optional)
                "nawm_needs_transform",  # True if nawm_mask is in T1w space (tumor seg)
                "roi_mask",  # Generic ROI mask in DSC native space (optional)
                "t1w_to_mni_xfm",  # T1w to MNI transform (from ACPC alignment)
                "te",  # Echo time in seconds
                "field_strength",  # Magnetic field strength
                "tr",  # Repetition time in seconds
                "flair_in_t1w",  # FLAIR coregistered to T1w space (from anat)
                "t2w_in_t1w",  # T2w coregistered to T1w space (from anat)
            ]
        ),
        name="inputnode",
    )

    # Fallback defaults — overridden by BIDS sidecar values in base.py
    inputnode.inputs.te = 0.035  # 35ms echo time (typical DSC-MRI at 3T)
    inputnode.inputs.field_strength = 3.0  # 3T
    inputnode.inputs.tr = 1.5  # 1.5s TR
    inputnode.inputs.nawm_needs_transform = True

    # -------------------------------------------------------------------
    # Output Node
    # Define all outputs from DSC processing
    # -------------------------------------------------------------------
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Motion-corrected data
                "dsc_moco",
                "dsc_mean",
                "motion_params",
                # Quantification-independent parameter maps
                "mtt_map",
                "tmax_map",
                "psr_map",
                "ph_map",
                "ttp_map",
                "ttp_corrected_map",
                "rttp_map",
                "cnr_map",
                # AUC decomposition
                "auc_diff_map",
                # Vascular model parameters (optional)
                "alpha_map",
                "beta_map",
                "cth_map",
                # Bayesian uncertainty maps (optional)
                "cbf_std_map",
                "cbv_std_map",
                "cbf_cv_map",
                "cbv_cv_map",
                "cbf_ci95_lower_map",
                "cbf_ci95_upper_map",
                "cbv_ci95_lower_map",
                "cbv_ci95_upper_map",
                # Parametric-fit diagnostics (CPI/Vascular only)
                "parametric_r2_map",
                "fallback_mask",
                # Relative maps (always produced)
                "rel_cbf_map",
                "rel_cbv_map",
                "rel_cbv_auc_map",
                "rel_auc_raw_map",
                "rel_auc_corr_map",
                # Quality metrics
                "snr_map",
                "tsnr_map",
                # AIF outputs
                "aif_curve",
                "aif_mask",
                "aif_qc",
                # Leakage correction
                "k2_map",
                # Normalized maps
                "ncbv_map",
                "ncbf_map",
                # Registration
                "dsc_to_t1w_xfm",
                "t1w_to_dsc_xfm",
                "dsc_to_t1w_h5",
                "t1w_to_dsc_h5",
                "registration_filmstrip",
                # Structural underlays
                "t1w_in_dsc",
                "t1w_postcon_in_dsc",
                "flair_in_dsc",
                "t2w_in_dsc",
                # AIF QC plots
                "aif_curve_plot",
                "aif_location_plot",
                "aif_candidates_plot",
                # Smoothed maps (populated only if smooth_fwhm > 0)
                "smooth_hemodynamic",
                "smooth_auc",
                # QC metrics
                "qc_metrics",
                "preload_qc",
                "preprocessing_stats",
                "motion_qc",
                "aif_metrics",
                # Masks in DSC space
                "brain_mask_dsc",
                "nawm_mask_dsc",
                # Lesion statistics (conditional on lesion mask)
                "lesion_stats_json",
                "lesion_mask_dsc",
                # User lesion statistics (conditional on user_lesion_mask)
                "user_lesion_stats_json",
                "user_lesion_mask_dsc",
                # ROI statistics (conditional on roi_mask)
                "roi_stats_json",
                "roi_mask_dsc",
                # Motion QC
                "motion_qc_plot",
                # DSC QC plots
                "dsc_preprocessing_qc_plot",
                "dsc_param_maps_qc_plot",
                "dsc_fit_qc_plot",
                "dsc_leakage_qc_plot",
                # Brain mask filmstrip
                "brain_mask_filmstrip",
            ]
        ),
        name="outputnode",
    )

    # Cap static mem_gb values to respect --mem-gb
    from pmriprep.utils.mem import cap_mem_gb

    _static_mem = cap_mem_gb(2.0, mem_limit_gb)
    _heavy_mem = mem_gb if mem_gb is not None else cap_mem_gb(4.0, mem_limit_gb)

    # -------------------------------------------------------------------
    # Motion sub-workflow: reorient -> motion-correct -> pre-bolus mean (+ QC)
    # -------------------------------------------------------------------
    motion_wf = init_dsc_motion_wf(
        motion_thresh=config.motion_thresh,
        baseline_frames=config.pre_bolus,
        subject_id=subject_id,
        session_id=session_id,
        mem_gb=_heavy_mem,
    )
    workflow.connect([
        (inputnode, motion_wf, [
            ("dsc_file", "inputnode.dsc_file"),
            ("tr", "inputnode.tr"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Node 1b: HD-BET Brain Extraction (for registration quality)
    # HD-BET extracts DSC brain so registration uses brain-extracted
    # images on both sides. The final processing mask comes from
    # transforming the T1w brain mask through the registration.
    # -------------------------------------------------------------------
    from pmriprep.interfaces.dsc_brain import DSCBrainExtract
    from pmriprep.utils.device import resolve_device_uses_gpu

    hdbet_dsc = pe.Node(
        DSCBrainExtract(),
        name="hdbet_dsc",
        mem_gb=_static_mem,
    )
    hdbet_dsc.inputs.hdbet_device = config.hdbet_device
    hdbet_dsc.inputs.hdbet_conda_env = config.hdbet_conda_env or ""
    hdbet_dsc.inputs.use_gpu = resolve_device_uses_gpu(config.hdbet_device)

    # -------------------------------------------------------------------
    # Node 1b2: Histogram Match T1w to DSC (for FLIRT convergence)
    # T1w and DSC have inverted contrasts. Without histogram matching,
    # FLIRT's normcorr coarse-search converges to a wrong local optimum.
    # Histogram matching improves cross-contrast FLIRT convergence.
    # -------------------------------------------------------------------
    hist_match_t1w = pe.Node(
        niu.Function(
            input_names=["t1w_file", "dsc_brain_file", "brain_mask_file"],
            output_names=["matched_t1w_file"],
            function=_histogram_match_t1w_to_dsc,
        ),
        name="hist_match_t1w",
    )

    # -------------------------------------------------------------------
    # Node 1c: Transform T1w Brain Mask to DSC Space
    # NOTE: T1w brain mask transform kept for future use. To reinstate,
    # uncomment and reconnect downstream consumers.
    # -------------------------------------------------------------------
    # transform_brain_mask = pe.Node(
    #     niu.Function(
    #         input_names=["brain_mask_file", "reference_file", "transform_matrix",
    #                      "hdbet_mask_file"],
    #         output_names=["dsc_brain_mask_file", "dice_score"],
    #         function=_transform_brain_mask_to_dsc,
    #     ),
    #     name="transform_brain_mask",
    # )

    # -------------------------------------------------------------------
    # Concentration + AIF sub-workflow (signal->ΔR2* conversion + AIF detection)
    # -------------------------------------------------------------------
    _external_aif = str(config.aif_file) if config.aif_file is not None else None
    _aif_mask: Optional[str] = None
    if config.aif_mask_file is not None:
        from pmriprep.utils.lesion_stats import validate_mask_ndim
        validate_mask_ndim(config.aif_mask_file, "dsc.aif_mask_file")
        _aif_mask = str(config.aif_mask_file)
    conc_aif_wf = init_dsc_concentration_aif_wf(
        baseline_frames=config.pre_bolus,
        aif_method=config.aif_method,
        n_candidates=config.n_aif_candidates,
        lesion_distance=config.lesion_distance,
        disable_fallback=config.disable_aif_fallback,
        dose=config.dose,
        agent_standard_dose=config.agent_standard_dose,
        gd_multiplicity=gd_multiplicity,
        external_aif_file=_external_aif,
        aif_mask_file=_aif_mask,
        mem_gb=_heavy_mem,
    )

    # -------------------------------------------------------------------
    # Node 4: Preload Detection and K2 Correction
    # K2 correction needs the AIF, so runs after AIF detection
    # -------------------------------------------------------------------
    preload_wf = init_dsc_preload_wf(
        preload=config.preload,
        agent_standard_dose=config.agent_standard_dose,
        baseline_frames=config.pre_bolus,
        leakage_method=config.leakage_method,
        kep=config.kep,
        mem_gb=_heavy_mem,
    )

    # -------------------------------------------------------------------
    # Node 5: DSC Map Computation (Deconvolution)
    # -------------------------------------------------------------------
    dsc_map_kwargs: dict[str, Any] = dict(
        method=config.method,
        svd_threshold=0.2,  # Default SVD threshold
        oi_threshold=0.065,  # Default OI threshold for oSVD
        cpi_control_points=config.cpi_control_points,
        cpi_infer_cpt=config.cpi_infer_cpt,
        inference=config.inference,
        mcmc_samples=config.mcmc_samples,
        n_procs=config.num_processes if config.num_processes > 0 else 1,
        # baseline_frames wired dynamically from conc_aif_wf.outputnode.aif_baseline_frames
        fallback_baseline_frames=config.pre_bolus,
        tail_frames=5,  # Post-bolus frames for PSR
        sloppy=config.sloppy,
    )
    if random_seed is not None:
        dsc_map_kwargs["random_seed"] = random_seed
    compute_maps = pe.Node(
        ComputeDSCMaps(**dsc_map_kwargs),
        name="compute_maps",
        mem_gb=_heavy_mem,
    )

    # -------------------------------------------------------------------
    # Node 6: DSC to T1w Registration
    # -------------------------------------------------------------------
    register = pe.Node(
        DSCRegistration(
            registration=config.registration,
            dof=6,  # Rigid body
            cost="normmi",
        ),
        name="register",
        mem_gb=_static_mem,
    )

    # -------------------------------------------------------------------
    # Node 6b: Transform Chain for MNI Atlas to DSC Space
    # Creates MNI→DSC transform by concatenating:
    #   MNI→T1w (inverted T1w→MNI) + T1w→DSC (register output)
    # -------------------------------------------------------------------

    # Invert T1w→MNI to get MNI→T1w
    invert_mni_xfm = pe.Node(
        ConvertXFM(invert_xfm=True),
        name="invert_mni_xfm",
    )

    # Invert T1w→DSC to get DSC→T1w (for output labeling)
    invert_dsc_xfm = pe.Node(
        ConvertXFM(invert_xfm=True),
        name="invert_dsc_xfm",
    )

    # Concatenate: MNI→T1w + T1w→DSC = MNI→DSC
    concat_mni_to_dsc = pe.Node(
        ConvertXFM(concat_xfm=True),
        name="concat_mni_to_dsc",
    )

    # Transform arterial atlas from MNI to DSC space
    # Use nearest neighbor interpolation for binary mask
    transform_atlas = pe.Node(
        FLIRTApplyXFM(
            apply_xfm=True,
            interp="nearestneighbour",
        ),
        name="transform_atlas",
    )
    # Cascade branch resolves to MCA_ACA_dil (core MCA/ACA, ~2 mm margin).
    # Fallback resolves to the dilated full MRA mask (full_mask_dil) when
    # the cascade has fewer than 2 tiers installed.
    try:
        atlas_cascade = get_arterial_atlas_cascade("2mm")
        if len(atlas_cascade) >= 2:
            transform_atlas.inputs.in_file = str(atlas_cascade[1])  # MCA_ACA_dil
        else:
            transform_atlas.inputs.in_file = str(get_arterial_atlas("2mm", "dil"))
    except (FileNotFoundError, IndexError):
        import logging as _logging
        _logging.getLogger("pmriprep.workflows.dsc").warning(
            "Arterial atlas not found. AIF detection will use whole-brain search."
        )

    # Transform full arterial atlas (undilated, incl. posterior circulation)
    transform_full_atlas = pe.Node(
        FLIRTApplyXFM(
            apply_xfm=True,
            interp="nearestneighbour",
        ),
        name="transform_full_atlas",
    )
    try:
        transform_full_atlas.inputs.in_file = str(get_arterial_atlas("2mm", ""))
    except FileNotFoundError:
        pass

    # -------------------------------------------------------------------
    # Node 6c: Transform NAWM mask from T1w to DSC space
    # -------------------------------------------------------------------
    transform_nawm = pe.Node(
        niu.Function(
            input_names=["nawm_mask_file", "reference_file", "transform_matrix",
                         "needs_transform"],
            output_names=["nawm_mask_dsc_file"],
            function=_transform_nawm_to_dsc,
        ),
        name="transform_nawm",
    )

    # -------------------------------------------------------------------
    # Node 6c-ii: Validate user-supplied masks against DSC reference grid
    # -------------------------------------------------------------------
    validate_roi_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "mask_label"],
            output_names=["validated_mask"],
            function=_validate_native_mask,
        ),
        name="validate_roi_mask",
    )
    validate_roi_mask.inputs.mask_label = "dsc.roi_mask"

    validate_user_lesion_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "mask_label"],
            output_names=["validated_mask"],
            function=_validate_native_mask,
        ),
        name="validate_user_lesion_mask",
    )
    validate_user_lesion_mask.inputs.mask_label = "dsc.lesion_mask"

    # -------------------------------------------------------------------
    # Node 6d: Select tumor mask for lesion statistics
    # Priority: tumor_segmentation (multi-label) > whole_tumor > None
    # -------------------------------------------------------------------
    from pmriprep.utils.lesion_stats import combine_tumor_masks

    combine_masks = pe.Node(
        niu.Function(
            input_names=[
                "tumor_segmentation_file",
                "whole_tumor_mask_file",
                "reference_file",
            ],
            output_names=["combined_mask_file", "mask_origin"],
            function=combine_tumor_masks,
        ),
        name="combine_masks",
    )

    # -------------------------------------------------------------------
    # Node 6e: Transform combined lesion mask from T1w to DSC space
    # -------------------------------------------------------------------
    transform_lesion = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dsc_file"],
            function=_transform_lesion_mask_to_dsc,
        ),
        name="transform_lesion",
    )

    # -------------------------------------------------------------------
    # Node 6e-b: Transform enhancing mask from T1w to DSC space
    # -------------------------------------------------------------------
    transform_enhancing_mask = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dsc_file"],
            function=_transform_lesion_mask_to_dsc,
        ),
        name="transform_enhancing_mask",
    )

    # -------------------------------------------------------------------
    # Node 6f: Lesion Statistics Extraction
    # Conditional - only run if lesion mask is available
    # -------------------------------------------------------------------
    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

    lesion_stats = pe.Node(
        DSCLesionStats(),
        name="lesion_stats",
    )
    lesion_stats.inputs.subject_id = subject_id
    lesion_stats.inputs.session_id = session_id

    # -------------------------------------------------------------------
    # Node 7: CBV Normalization
    # -------------------------------------------------------------------
    normalize = pe.Node(
        niu.Function(
            input_names=[
                "cbv_file",
                "brain_mask_file",
                "derivatives_dir",
                "subject_id",
                "session_id",
                "nawm_mask_file",
            ],
            output_names=["ncbv_file", "normalization_factor"],
            function=_normalize_cbv,
        ),
        name="normalize",
    )
    normalize.inputs.derivatives_dir = str(derivatives_dir)
    normalize.inputs.subject_id = subject_id
    normalize.inputs.session_id = session_id

    # -------------------------------------------------------------------
    # Node 7b: CBF Normalization (same method as CBV normalization)
    # -------------------------------------------------------------------
    normalize_cbf = pe.Node(
        niu.Function(
            input_names=[
                "cbf_file",
                "brain_mask_file",
                "derivatives_dir",
                "subject_id",
                "session_id",
                "nawm_mask_file",
            ],
            output_names=["ncbf_file", "normalization_factor"],
            function=_normalize_cbf,
        ),
        name="normalize_cbf",
    )
    normalize_cbf.inputs.derivatives_dir = str(derivatives_dir)
    normalize_cbf.inputs.subject_id = subject_id
    normalize_cbf.inputs.session_id = session_id

    # -------------------------------------------------------------------
    # Node 7c: Smoothing (conditional on config.smooth_fwhm > 0)
    # -------------------------------------------------------------------
    smooth_hemodynamic = None
    smooth_auc = None

    if config.smooth_fwhm > 0:
        # Smoothing for hemodynamic maps: CBF, CBV, MTT, nCBF, nCBV
        # Physics-approved: intensive quantities that vary smoothly
        smooth_hemodynamic = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_maps,
            ),
            name="smooth_hemodynamic",
        )
        smooth_hemodynamic.inputs.fwhm = config.smooth_fwhm

        # Smoothing for leakage-corrected AUC (CBV surrogate)
        # Only auc_corr; auc_raw contains leakage artifact
        smooth_auc = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_maps,
            ),
            name="smooth_auc",
        )
        smooth_auc.inputs.fwhm = config.smooth_fwhm

    # -------------------------------------------------------------------
    # Node 8: Merge QC Metrics
    # -------------------------------------------------------------------
    # Determine AIF source for boilerplate text generation
    if config.aif_file is not None:
        aif_source = "external"
    elif config.aif_method == "synthetic":
        aif_source = "synthetic"
    else:
        aif_source = "auto"

    merge_qc = pe.Node(
        niu.Function(
            input_names=[
                "motion_qc",
                "preload_qc",
                "aif_qc",
                "deconv_qc",
                "registration_qc",
                "normalization_factor",
                "motion_summary",
                "aif_metrics",
                "preprocessing_stats",
                "aif_source",
                "registration_bbr",
            ],
            output_names=["merged_qc"],
            function=_merge_qc_metrics,
        ),
        name="merge_qc",
    )
    merge_qc.inputs.aif_source = aif_source
    merge_qc.inputs.registration_bbr = config.registration == "flirt_bbr"

    # -------------------------------------------------------------------
    # Node 9: Registration QC Filmstrip
    # -------------------------------------------------------------------
    reg_filmstrip = pe.Node(
        niu.Function(
            input_names=["dsc_mean_file", "registered_t1w_file", "brain_mask_file",
                         "subject_id", "session_id"],
            output_names=["filmstrip_file"],
            function=_create_registration_filmstrip,
        ),
        name="reg_filmstrip",
    )
    reg_filmstrip.inputs.subject_id = subject_id
    reg_filmstrip.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9b: AIF QC Plots
    # -------------------------------------------------------------------
    aif_qc_plot = pe.Node(
        AIFQCPlot(
            subject_id=subject_id,
            session_id=session_id or "",
        ),
        name="aif_qc_plot",
    )

    # -------------------------------------------------------------------
    # Node 9c: DSC Preprocessing QC Plot
    # -------------------------------------------------------------------
    dsc_preproc_qc = pe.Node(
        niu.Function(
            input_names=[
                "dsc_signal_file", "brain_mask_file",
                "baseline_frames",
                "tr", "te", "subject_id", "session_id",
            ],
            output_names=["plot_file", "stats"],
            function=_generate_dsc_preprocessing_qc,
        ),
        name="dsc_preproc_qc",
    )
    dsc_preproc_qc.inputs.baseline_frames = config.pre_bolus  # S0 baseline (pipeline value)
    dsc_preproc_qc.inputs.subject_id = subject_id
    dsc_preproc_qc.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9d: DSC Parameter Maps QC Plot
    # -------------------------------------------------------------------
    dsc_param_maps_qc = pe.Node(
        niu.Function(
            input_names=[
                "rel_cbf_file", "rel_cbv_file", "mtt_file",
                "tmax_file", "psr_file", "ttp_file",
                "brain_mask_file", "subject_id", "session_id",
            ],
            output_names=["plot_file"],
            function=_generate_dsc_param_maps_qc,
        ),
        name="dsc_param_maps_qc",
    )
    dsc_param_maps_qc.inputs.subject_id = subject_id
    dsc_param_maps_qc.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9e: DSC Leakage Correction QC Plot (conditional on K2)
    # -------------------------------------------------------------------
    dsc_leakage_qc = pe.Node(
        niu.Function(
            input_names=[
                "k2_map_file", "auc_raw_file", "auc_corr_file",
                "auc_diff_file", "brain_mask_file",
                "concentration_file", "corrected_concentration_file",
                "psr_file", "tr", "subject_id", "session_id",
                "preload_qc",
            ],
            output_names=["plot_file"],
            function=_generate_dsc_leakage_qc,
        ),
        name="dsc_leakage_qc",
    )
    dsc_leakage_qc.inputs.subject_id = subject_id
    dsc_leakage_qc.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9e2: DSC Fit QC Plot (all deconvolution methods)
    # -------------------------------------------------------------------
    dsc_fit_qc = pe.Node(
        niu.Function(
            input_names=[
                "qc_fit_data_file", "brain_mask_file", "cbf_file",
                "subject_id", "session_id",
            ],
            output_names=["plot_file"],
            function=_generate_dsc_fit_qc,
        ),
        name="dsc_fit_qc",
    )
    dsc_fit_qc.inputs.subject_id = subject_id
    dsc_fit_qc.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9f: Transform structural underlays to DSC Space
    # register.out_matrix is T1w→DSC, so we use it directly.
    # Underlays are conditional based on config.underlays list.
    # Uses Function interface to handle undefined inputs gracefully.
    # -------------------------------------------------------------------
    transform_t1_to_dsc = None
    if "t1w_precon" in config.underlays:
        transform_t1_to_dsc = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dsc,
            ),
            name="transform_t1_to_dsc",
        )
        transform_t1_to_dsc.inputs.output_name = "t1w_in_dsc.nii.gz"

    transform_t1_postcon_to_dsc = None
    if "t1w_postcon" in config.underlays:
        transform_t1_postcon_to_dsc = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dsc,
            ),
            name="transform_t1_postcon_to_dsc",
        )
        transform_t1_postcon_to_dsc.inputs.output_name = "t1w_postcon_in_dsc.nii.gz"

    transform_flair_to_dsc = None
    if "flair" in config.underlays:
        transform_flair_to_dsc = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dsc,
            ),
            name="transform_flair_to_dsc",
        )
        transform_flair_to_dsc.inputs.output_name = "flair_in_dsc.nii.gz"

    transform_t2w_to_dsc = None
    if "t2w" in config.underlays:
        transform_t2w_to_dsc = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dsc,
            ),
            name="transform_t2w_to_dsc",
        )
        transform_t2w_to_dsc.inputs.output_name = "t2w_in_dsc.nii.gz"

    # -------------------------------------------------------------------
    # Node 10: Provenance Logging
    # -------------------------------------------------------------------
    import dataclasses
    config_dict = dataclasses.asdict(config)
    # Convert Path objects to strings for JSON serialization
    for k, v in config_dict.items():
        if isinstance(v, Path):
            config_dict[k] = str(v)

    # -------------------------------------------------------------------
    # Workflow Connections
    # -------------------------------------------------------------------

    # Pre-Bolus Mean to HD-BET Brain Extraction
    # HD-BET on pre-bolus mean for brain-extracted DSC (improves registration)
    workflow.connect([
        (motion_wf, hdbet_dsc, [("outputnode.prebolus_mean_file", "dsc_mean_file")]),
    ])

    # Brain mask filmstrip QC (after HD-BET)
    dsc_brain_mask_qc = pe.Node(
        niu.Function(
            input_names=["mean_file", "brain_mask_file", "subject_id", "session_id"],
            output_names=["out_file"],
            function=_create_perfusion_brain_mask_filmstrip,
        ),
        name="dsc_brain_mask_qc",
    )
    dsc_brain_mask_qc.inputs.subject_id = subject_id
    dsc_brain_mask_qc.inputs.session_id = session_id or ""
    workflow.connect([
        (motion_wf, dsc_brain_mask_qc, [
            ("outputnode.prebolus_mean_file", "mean_file"),
        ]),
        (hdbet_dsc, dsc_brain_mask_qc, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # NOTE: transform_brain_mask disconnected — using HD-BET mask directly.
    # To reinstate T1w-transformed mask, uncomment the node definition
    # above and restore these connections:
    # workflow.connect([
    #     (inputnode, transform_brain_mask, [("brain_mask", "brain_mask_file")]),
    #     (prebolus_mean, transform_brain_mask, [("prebolus_mean_file", "reference_file")]),
    #     (register, transform_brain_mask, [("out_matrix", "transform_matrix")]),
    #     (hdbet_dsc, transform_brain_mask, [("dsc_brain_mask_file", "hdbet_mask_file")]),
    # ])

    # Motion + acquisition params into the concentration/AIF sub-workflow
    workflow.connect([
        (motion_wf, conc_aif_wf, [("outputnode.dsc_moco", "inputnode.dsc_moco")]),
        (inputnode, conc_aif_wf, [
            ("te", "inputnode.te"),
        ]),
    ])

    # AIF exclusion mask: user-supplied lesion mask overrides tumor seg
    select_aif_exclusion = pe.Node(
        niu.Function(
            input_names=["user_lesion_mask", "transformed_lesion_mask"],
            output_names=["out_mask"],
            function=_select_aif_exclusion_mask,
        ),
        name="select_aif_exclusion",
    )
    workflow.connect([
        (validate_user_lesion_mask, select_aif_exclusion, [
            ("validated_mask", "user_lesion_mask"),
        ]),
        (transform_lesion, select_aif_exclusion, [
            ("lesion_mask_dsc_file", "transformed_lesion_mask"),
        ]),
    ])

    # Signal Conversion to AIF Detection
    # Use DSC-space brain mask (not T1w-space) for proper geometry match
    # Pass raw signal for S0-based partial volume rejection
    workflow.connect([
        (hdbet_dsc, conc_aif_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
        ]),
        (select_aif_exclusion, conc_aif_wf, [
            ("out_mask", "inputnode.lesion_mask_file"),
        ]),
        (transform_enhancing_mask, conc_aif_wf, [
            ("lesion_mask_dsc_file", "inputnode.enhancing_mask_file"),
        ]),
        (inputnode, conc_aif_wf, [
            ("tr", "inputnode.tr"),
            ("field_strength", "inputnode.field_strength"),
        ]),
    ])

    # AIF Detection and Signal Conversion to Preload Correction
    # Use DSC-space brain mask for proper geometry match
    workflow.connect([
        (conc_aif_wf, preload_wf, [
            ("outputnode.concentration_file", "inputnode.concentration_file"),
            ("outputnode.signal_file", "inputnode.signal_file"),
            ("outputnode.out_aif_curve", "inputnode.aif_file"),
        ]),
        (hdbet_dsc, preload_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
        ]),
        (inputnode, preload_wf, [
            ("te", "inputnode.te"),
            ("field_strength", "inputnode.field_strength"),
            ("tr", "inputnode.tr"),
        ]),
    ])

    # Preload Correction to DSC Map Computation
    # Use DSC-space brain mask for proper geometry match
    workflow.connect([
        (preload_wf, compute_maps, [
            ("outputnode.corrected_concentration_file", "concentration_file"),
            ("outputnode.k2_map_file", "k2_map_input_file"),
        ]),
        (conc_aif_wf, compute_maps, [
            ("outputnode.out_aif_curve", "aif_file"),
            ("outputnode.out_measured_aif_curve", "measured_aif_file"),
            ("outputnode.out_aif_mask", "aif_mask_file"),
            ("outputnode.out_aif_coord", "aif_best_coord"),
            ("outputnode.detection_level", "aif_detection_level"),
            ("outputnode.aif_baseline_frames", "baseline_frames"),
        ]),
        (hdbet_dsc, compute_maps, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
        (transform_enhancing_mask, compute_maps, [
            ("lesion_mask_dsc_file", "enhancing_mask_file"),
        ]),
        (inputnode, compute_maps, [
            ("tr", "tr"),
            ("te", "te"),
            ("field_strength", "field_strength"),
        ]),
        (motion_wf, compute_maps, [
            ("outputnode.fd_values", "fd_values"),
        ]),
        (conc_aif_wf, compute_maps, [
            ("outputnode.signal_file", "signal_file"),
            ("outputnode.concentration_file", "raw_concentration_file"),
        ]),
        (preload_wf, compute_maps, [
            ("outputnode.preload_classification", "preload_classification"),
        ]),
    ])

    # Histogram match T1w to DSC for registration convergence
    # Brain mask applied to T1w before matching
    workflow.connect([
        (inputnode, hist_match_t1w, [
            ("t1w_reference", "t1w_file"),
            ("brain_mask", "brain_mask_file"),
        ]),
        (hdbet_dsc, hist_match_t1w, [("dsc_brain_file", "dsc_brain_file")]),
    ])

    # DSC Map Computation to Registration
    # Use brain-extracted DSC (from HD-BET) and histogram-matched T1w
    workflow.connect([
        (hist_match_t1w, register, [("matched_t1w_file", "t1w_file")]),
        (inputnode, register, [
            ("brain_mask", "brain_mask"),
            ("wm_probseg", "wm_seg"),
        ]),
        (hdbet_dsc, register, [("dsc_brain_file", "dsc_mean")]),
    ])

    # Transform Chain: Create MNI→DSC transform for atlas
    # Step 1: Invert T1w→MNI to get MNI→T1w
    workflow.connect([
        (inputnode, invert_mni_xfm, [("t1w_to_mni_xfm", "in_file")]),
    ])

    # Step 2: Invert T1w→DSC to get DSC→T1w (for output labeling)
    workflow.connect([
        (register, invert_dsc_xfm, [("out_matrix", "in_file")]),
    ])

    # Step 3: Concatenate MNI→T1w + T1w→DSC = MNI→DSC
    workflow.connect([
        (invert_mni_xfm, concat_mni_to_dsc, [("out_file", "in_file")]),
        (register, concat_mni_to_dsc, [("out_matrix", "in_file2")]),
    ])

    # Step 4: Transform arterial atlas from MNI to DSC space
    # Use pre-bolus mean as reference for output geometry
    workflow.connect([
        (concat_mni_to_dsc, transform_atlas, [("out_file", "in_matrix_file")]),
        (motion_wf, transform_atlas, [("outputnode.prebolus_mean_file", "reference")]),
    ])

    # Connect transformed atlas to AIF detection
    workflow.connect([
        (transform_atlas, conc_aif_wf, [("out_file", "inputnode.atlas_mask_file")]),
    ])

    # Transform + connect full atlas (posterior circulation tier)
    if isdefined(transform_full_atlas.inputs.in_file):
        workflow.connect([
            (concat_mni_to_dsc, transform_full_atlas, [("out_file", "in_matrix_file")]),
            (motion_wf, transform_full_atlas, [("outputnode.prebolus_mean_file", "reference")]),
            (transform_full_atlas, conc_aif_wf, [("out_file", "inputnode.full_atlas_mask_file")]),
        ])

    # Transform NAWM mask to DSC space (FLIRT when from tumor seg, passthrough
    # when user-supplied in DSC native space)
    workflow.connect([
        (inputnode, transform_nawm, [
            ("nawm_mask", "nawm_mask_file"),
            ("nawm_needs_transform", "needs_transform"),
        ]),
        (motion_wf, transform_nawm, [("outputnode.prebolus_mean_file", "reference_file")]),
        (register, transform_nawm, [("out_matrix", "transform_matrix")]),
    ])

    # Validate user-supplied masks against DSC reference grid
    workflow.connect([
        (inputnode, validate_roi_mask, [("roi_mask", "mask_file")]),
        (motion_wf, validate_roi_mask, [("outputnode.prebolus_mean_file", "reference_file")]),
        (inputnode, validate_user_lesion_mask, [("user_lesion_mask", "mask_file")]),
        (motion_wf, validate_user_lesion_mask, [
            ("outputnode.prebolus_mean_file", "reference_file"),
        ]),
    ])

    # Wire NAWM mask to compute_maps for rTTP normalization
    workflow.connect([
        (transform_nawm, compute_maps, [
            ("nawm_mask_dsc_file", "nawm_mask_file"),
        ]),
    ])

    # DSC Map Computation to CBV Normalization (uses relative CBV)
    # Use DSC-space brain mask for fallback normalization
    workflow.connect([
        (compute_maps, normalize, [("out_rel_cbv_auc", "cbv_file")]),
        (transform_nawm, normalize, [
            ("nawm_mask_dsc_file", "nawm_mask_file"),
        ]),
        (hdbet_dsc, normalize, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # DSC Map Computation to CBF Normalization (uses relative CBF)
    workflow.connect([
        (compute_maps, normalize_cbf, [("out_rel_cbf", "cbf_file")]),
        (transform_nawm, normalize_cbf, [
            ("nawm_mask_dsc_file", "nawm_mask_file"),
        ]),
        (hdbet_dsc, normalize_cbf, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # Registration QC filmstrip
    workflow.connect([
        (motion_wf, reg_filmstrip, [("outputnode.prebolus_mean_file", "dsc_mean_file")]),
        (register, reg_filmstrip, [("out_file", "registered_t1w_file")]),
        (hdbet_dsc, reg_filmstrip, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # AIF QC plots
    workflow.connect([
        (conc_aif_wf, aif_qc_plot, [
            ("outputnode.out_aif_curve", "aif_curve_file"),
            ("outputnode.out_aif_mask", "aif_mask_file"),
            ("outputnode.qc_metrics", "qc_metrics"),
            ("outputnode.out_raw_aif_curve", "raw_aif_curve_file"),
            ("outputnode.gamma_fit_applied", "gamma_fit_applied"),
        ]),
        (motion_wf, aif_qc_plot, [("outputnode.dsc_moco", "dsc_file")]),
        (hdbet_dsc, aif_qc_plot, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, aif_qc_plot, [
            ("tr", "tr"),
            ("t1w_to_mni_xfm", "t1w_to_mni_xfm"),
        ]),
        (register, aif_qc_plot, [
            ("out_matrix", "t1w_to_dsc_xfm"),
        ]),
    ])

    # DSC Preprocessing QC
    workflow.connect([
        (motion_wf, dsc_preproc_qc, [("outputnode.dsc_moco", "dsc_signal_file")]),
        (hdbet_dsc, dsc_preproc_qc, [("dsc_brain_mask_file", "brain_mask_file")]),
        (inputnode, dsc_preproc_qc, [("tr", "tr"), ("te", "te")]),
    ])

    # DSC Parameter Maps QC
    workflow.connect([
        (compute_maps, dsc_param_maps_qc, [
            ("out_rel_cbf", "rel_cbf_file"),
            ("out_rel_cbv_auc", "rel_cbv_file"),
            ("out_mtt", "mtt_file"),
            ("out_tmax", "tmax_file"),
            ("out_psr", "psr_file"),
            ("out_ttp_corrected", "ttp_file"),
        ]),
        (hdbet_dsc, dsc_param_maps_qc, [("dsc_brain_mask_file", "brain_mask_file")]),
    ])

    # DSC Leakage Correction QC
    workflow.connect([
        (compute_maps, dsc_leakage_qc, [
            ("out_k2", "k2_map_file"),
            ("out_auc_raw", "auc_raw_file"),
            ("out_auc_corr", "auc_corr_file"),
            ("out_auc_diff", "auc_diff_file"),
            ("out_psr", "psr_file"),
        ]),
        (hdbet_dsc, dsc_leakage_qc, [("dsc_brain_mask_file", "brain_mask_file")]),
        (conc_aif_wf, dsc_leakage_qc, [("outputnode.concentration_file", "concentration_file")]),
        (preload_wf, dsc_leakage_qc, [
            ("outputnode.corrected_concentration_file", "corrected_concentration_file"),
            ("outputnode.qc_dict", "preload_qc"),
        ]),
        (inputnode, dsc_leakage_qc, [("tr", "tr")]),
    ])

    # DSC Fit QC (all deconvolution methods)
    workflow.connect([
        (compute_maps, dsc_fit_qc, [
            ("out_qc_fit_data", "qc_fit_data_file"),
            ("out_rel_cbf", "cbf_file"),
        ]),
        (hdbet_dsc, dsc_fit_qc, [("dsc_brain_mask_file", "brain_mask_file")]),
    ])

    # -------------------------------------------------------------------
    # Lesion mask selection and transform (conditional)
    # -------------------------------------------------------------------
    # Select best available tumor mask for lesion stats
    # Priority: tumor_segmentation (multi-label) > whole_tumor
    workflow.connect([
        (inputnode, combine_masks, [
            ("tumor_segmentation", "tumor_segmentation_file"),
            ("whole_tumor_mask", "whole_tumor_mask_file"),
        ]),
        (motion_wf, combine_masks, [("outputnode.prebolus_mean_file", "reference_file")]),
    ])

    # Transform combined lesion mask from T1w to DSC space
    workflow.connect([
        (combine_masks, transform_lesion, [("combined_mask_file", "lesion_mask_file")]),
        (motion_wf, transform_lesion, [("outputnode.prebolus_mean_file", "reference_file")]),
        (register, transform_lesion, [("out_matrix", "transform_matrix")]),
    ])

    # Transform enhancing mask from T1w to DSC space
    workflow.connect([
        (inputnode, transform_enhancing_mask, [("enhancing_mask", "lesion_mask_file")]),
        (motion_wf, transform_enhancing_mask, [("outputnode.prebolus_mean_file", "reference_file")]),
        (register, transform_enhancing_mask, [("out_matrix", "transform_matrix")]),
    ])

    # Lesion statistics extraction
    # Connect lesion mask in DSC space
    workflow.connect([
        (transform_lesion, lesion_stats, [("lesion_mask_dsc_file", "lesion_mask")]),
        (combine_masks, lesion_stats, [("mask_origin", "mask_origin")]),
        (transform_nawm, lesion_stats, [("nawm_mask_dsc_file", "nawm_mask")]),
        (inputnode, lesion_stats, [("field_strength", "field_strength_t")]),
    ])
    # Connect perfusion maps
    workflow.connect([
        (compute_maps, lesion_stats, [
            ("out_rel_cbf", "cbf_map"),
            ("out_rel_cbv_auc", "cbv_map"),
            ("out_mtt", "mtt_map"),
            ("out_tmax", "tmax_map"),
            ("out_psr", "psr_map"),
            ("out_ph", "ph_map"),
            ("out_ttp", "ttp_map"),
            ("out_ttp_corrected", "ttp_corrected_map"),
            ("out_rttp", "rttp_map"),
            ("out_auc_corr", "auc_corr_map"),
            ("out_auc_diff", "auc_diff_map"),
            ("out_rel_auc_corr", "rel_auc_corr_map"),
        ]),
        (normalize, lesion_stats, [("ncbv_file", "ncbv_map")]),
        (normalize_cbf, lesion_stats, [("ncbf_file", "ncbf_map")]),
        (compute_maps, lesion_stats, [("out_k2", "k2_map")]),
        # AIF provenance — surface PerfusionProcessing.AIF block in
        # the lesion-stats sidecar.
        (conc_aif_wf, lesion_stats, [("outputnode.qc_metrics", "aif_qc")]),
    ])
    # Connect lesion stats outputs to outputnode
    workflow.connect([
        (transform_lesion, outputnode, [("lesion_mask_dsc_file", "lesion_mask_dsc")]),
        (lesion_stats, outputnode, [
            ("stats_json", "lesion_stats_json"),
        ]),
    ])

    # -------------------------------------------------------------------
    # ROI statistics (conditional on roi_mask input)
    # roi_mask is user-supplied in DSC native space — no FLIRT transform.
    # -------------------------------------------------------------------
    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats as _DSCLesionStatsROI

    roi_stats = pe.Node(
        _DSCLesionStatsROI(force_binary=True),
        name="roi_stats",
    )
    roi_stats.inputs.subject_id = subject_id
    roi_stats.inputs.session_id = session_id

    # Connect validated ROI mask
    workflow.connect([
        (validate_roi_mask, roi_stats, [("validated_mask", "lesion_mask")]),
        (transform_nawm, roi_stats, [("nawm_mask_dsc_file", "nawm_mask")]),
        (inputnode, roi_stats, [("field_strength", "field_strength_t")]),
        (compute_maps, roi_stats, [
            ("out_rel_cbf", "cbf_map"),
            ("out_rel_cbv_auc", "cbv_map"),
            ("out_mtt", "mtt_map"),
            ("out_tmax", "tmax_map"),
            ("out_psr", "psr_map"),
            ("out_ph", "ph_map"),
            ("out_ttp", "ttp_map"),
            ("out_ttp_corrected", "ttp_corrected_map"),
            ("out_rttp", "rttp_map"),
            ("out_auc_corr", "auc_corr_map"),
            ("out_auc_diff", "auc_diff_map"),
            ("out_rel_auc_corr", "rel_auc_corr_map"),
        ]),
        (normalize, roi_stats, [("ncbv_file", "ncbv_map")]),
        (normalize_cbf, roi_stats, [("ncbf_file", "ncbf_map")]),
        (compute_maps, roi_stats, [("out_k2", "k2_map")]),
        (conc_aif_wf, roi_stats, [("outputnode.qc_metrics", "aif_qc")]),
    ])
    # Connect ROI stats outputs to outputnode
    workflow.connect([
        (validate_roi_mask, outputnode, [("validated_mask", "roi_mask_dsc")]),
        (roi_stats, outputnode, [
            ("stats_json", "roi_stats_json"),
        ]),
    ])

    # -------------------------------------------------------------------
    # User lesion statistics (conditional on user_lesion_mask input)
    # user_lesion_mask is in DSC native space — no FLIRT transform.
    # Additive: runs alongside tumor seg lesion_stats when both exist.
    # -------------------------------------------------------------------
    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats as _DSCLesionStatsUser

    user_lesion_stats = pe.Node(
        _DSCLesionStatsUser(force_binary=True),
        name="user_lesion_stats",
    )
    user_lesion_stats.inputs.subject_id = subject_id
    user_lesion_stats.inputs.session_id = session_id

    workflow.connect([
        (validate_user_lesion_mask, user_lesion_stats, [
            ("validated_mask", "lesion_mask"),
        ]),
        (transform_nawm, user_lesion_stats, [("nawm_mask_dsc_file", "nawm_mask")]),
        (inputnode, user_lesion_stats, [("field_strength", "field_strength_t")]),
        (compute_maps, user_lesion_stats, [
            ("out_rel_cbf", "cbf_map"),
            ("out_rel_cbv_auc", "cbv_map"),
            ("out_mtt", "mtt_map"),
            ("out_tmax", "tmax_map"),
            ("out_psr", "psr_map"),
            ("out_ph", "ph_map"),
            ("out_ttp", "ttp_map"),
            ("out_ttp_corrected", "ttp_corrected_map"),
            ("out_rttp", "rttp_map"),
            ("out_auc_corr", "auc_corr_map"),
            ("out_auc_diff", "auc_diff_map"),
            ("out_rel_auc_corr", "rel_auc_corr_map"),
        ]),
        (normalize, user_lesion_stats, [("ncbv_file", "ncbv_map")]),
        (normalize_cbf, user_lesion_stats, [("ncbf_file", "ncbf_map")]),
        (compute_maps, user_lesion_stats, [("out_k2", "k2_map")]),
        (conc_aif_wf, user_lesion_stats, [("outputnode.qc_metrics", "aif_qc")]),
    ])
    workflow.connect([
        (validate_user_lesion_mask, outputnode, [
            ("validated_mask", "user_lesion_mask_dsc"),
        ]),
        (user_lesion_stats, outputnode, [
            ("stats_json", "user_lesion_stats_json"),
        ]),
    ])

    # Expose DSC-space masks on outputnode
    workflow.connect([
        (hdbet_dsc, outputnode, [("dsc_brain_mask_file", "brain_mask_dsc")]),
        (transform_nawm, outputnode, [("nawm_mask_dsc_file", "nawm_mask_dsc")]),
    ])

    # Transform structural underlays to DSC space (conditional)
    # register.out_matrix is T1w→DSC
    # Use prebolus_mean (full FOV) as reference for complete DSC geometry
    if transform_t1_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_t1_to_dsc, [("t1w_underlay", "structural_file")]),
            (motion_wf, transform_t1_to_dsc, [("outputnode.prebolus_mean_file", "reference_file")]),
            (register, transform_t1_to_dsc, [("out_matrix", "transform_matrix")]),
        ])

    if transform_t1_postcon_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_t1_postcon_to_dsc, [("t1w_postcon_underlay", "structural_file")]),
            (motion_wf, transform_t1_postcon_to_dsc, [("outputnode.prebolus_mean_file", "reference_file")]),
            (register, transform_t1_postcon_to_dsc, [("out_matrix", "transform_matrix")]),
        ])

    if transform_flair_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_flair_to_dsc, [("flair_in_t1w", "structural_file")]),
            (motion_wf, transform_flair_to_dsc, [("outputnode.prebolus_mean_file", "reference_file")]),
            (register, transform_flair_to_dsc, [("out_matrix", "transform_matrix")]),
        ])

    if transform_t2w_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_t2w_to_dsc, [("t2w_in_t1w", "structural_file")]),
            (motion_wf, transform_t2w_to_dsc, [("outputnode.prebolus_mean_file", "reference_file")]),
            (register, transform_t2w_to_dsc, [("out_matrix", "transform_matrix")]),
        ])

    # -------------------------------------------------------------------
    # Smoothing connections (only if smooth_fwhm > 0)
    # -------------------------------------------------------------------
    if smooth_hemodynamic is not None:
        # Build dict of hemodynamic maps for smoothing:
        # MTT, rel_CBF, rel_CBV, plus optional abs_CBF, abs_CBV, nCBF, nCBV
        # Physics-approved: intensive quantities that vary smoothly across tissue.
        # NOT smoothed: Tmax, TTP, delay (timing), PSR, PH, K2, auc_diff (BBB/leakage)

        build_hemo_dict = pe.Node(
            niu.Function(
                input_names=[
                    "mtt", "rel_cbf", "rel_cbv",
                    "ncbf", "ncbv",
                ],
                output_names=["map_dict"],
                function=_build_hemodynamic_dict,
            ),
            name="build_hemo_dict",
        )
        workflow.connect([
            (compute_maps, build_hemo_dict, [
                ("out_mtt", "mtt"),
                ("out_rel_cbf", "rel_cbf"),
                ("out_rel_cbv_auc", "rel_cbv"),
            ]),
            (normalize_cbf, build_hemo_dict, [("ncbf_file", "ncbf")]),
            (normalize, build_hemo_dict, [("ncbv_file", "ncbv")]),
            (build_hemo_dict, smooth_hemodynamic, [("map_dict", "map_files")]),
            (hdbet_dsc, smooth_hemodynamic, [
                ("dsc_brain_mask_file", "brain_mask_file"),
            ]),
        ])

        # Build dict of leakage-corrected AUC maps for smoothing (CBV surrogate).
        # Only auc_corr is smoothed; auc_raw contains leakage artifact.

        build_auc_dict = pe.Node(
            niu.Function(
                input_names=["rel_auc_corr"],
                output_names=["map_dict"],
                function=_build_auc_corr_dict,
            ),
            name="build_auc_dict",
        )
        workflow.connect([
            (compute_maps, build_auc_dict, [
                ("out_rel_auc_corr", "rel_auc_corr"),
            ]),
            (build_auc_dict, smooth_auc, [("map_dict", "map_files")]),
            (hdbet_dsc, smooth_auc, [
                ("dsc_brain_mask_file", "brain_mask_file"),
            ]),
        ])

    # Merge QC metrics
    workflow.connect([
        (motion_wf, merge_qc, [("outputnode.motion_qc_metrics", "motion_qc")]),
        (preload_wf, merge_qc, [("outputnode.qc_dict", "preload_qc")]),
        (conc_aif_wf, merge_qc, [("outputnode.qc_metrics", "aif_qc")]),
        (compute_maps, merge_qc, [("qc_metrics", "deconv_qc")]),
        (register, merge_qc, [("qc_metrics", "registration_qc")]),
        (normalize, merge_qc, [("normalization_factor", "normalization_factor")]),
        (motion_wf, merge_qc, [("outputnode.motion_summary", "motion_summary")]),
        (aif_qc_plot, merge_qc, [("out_aif_metrics", "aif_metrics")]),
        (dsc_preproc_qc, merge_qc, [("stats", "preprocessing_stats")]),
    ])

    # Connect to Output Node
    workflow.connect([
        # Motion correction outputs
        (motion_wf, outputnode, [
            ("outputnode.dsc_moco", "dsc_moco"),
            ("outputnode.dsc_mean", "dsc_mean"),
            ("outputnode.motion_params", "motion_params"),
        ]),
        # Quantification-independent maps
        (compute_maps, outputnode, [
            ("out_mtt", "mtt_map"),
            ("out_tmax", "tmax_map"),
            ("out_psr", "psr_map"),
            ("out_ph", "ph_map"),
            ("out_ttp", "ttp_map"),
            ("out_ttp_corrected", "ttp_corrected_map"),
            ("out_rttp", "rttp_map"),
            ("out_cnr", "cnr_map"),
            ("out_auc_diff", "auc_diff_map"),
        ]),
        # Relative maps
        (compute_maps, outputnode, [
            ("out_rel_cbf", "rel_cbf_map"),
            ("out_rel_cbv", "rel_cbv_map"),
            ("out_rel_cbv_auc", "rel_cbv_auc_map"),
            ("out_rel_auc_raw", "rel_auc_raw_map"),
            ("out_rel_auc_corr", "rel_auc_corr_map"),
        ]),
        # Quality metrics
        (compute_maps, outputnode, [
            ("out_snr", "snr_map"),
            ("out_tsnr", "tsnr_map"),
        ]),
        # Vascular model parameters (optional — undefined for SVD methods)
        (compute_maps, outputnode, [
            ("out_alpha", "alpha_map"),
            ("out_beta", "beta_map"),
            ("out_cth", "cth_map"),
        ]),
        # Bayesian uncertainty maps (only for parametric methods: cpi, vascular)
        (compute_maps, outputnode, [
            ("out_cbf_std", "cbf_std_map"),
            ("out_cbv_std", "cbv_std_map"),
            ("out_cbf_cv", "cbf_cv_map"),
            ("out_cbv_cv", "cbv_cv_map"),
            ("out_cbf_ci95_lower", "cbf_ci95_lower_map"),
            ("out_cbf_ci95_upper", "cbf_ci95_upper_map"),
            ("out_cbv_ci95_lower", "cbv_ci95_lower_map"),
            ("out_cbv_ci95_upper", "cbv_ci95_upper_map"),
        ]),
        # Parametric-fit diagnostics (CPI/Vascular only;
        # undefined for SVD methods)
        (compute_maps, outputnode, [
            ("out_parametric_r2", "parametric_r2_map"),
            ("out_fallback_mask", "fallback_mask"),
        ]),
        # AIF outputs
        (conc_aif_wf, outputnode, [
            ("outputnode.out_aif_curve", "aif_curve"),
            ("outputnode.out_aif_mask", "aif_mask"),
            ("outputnode.qc_metrics", "aif_qc"),
        ]),
        # Leakage correction (use recomputed K2 from compute_maps when baseline differs)
        (compute_maps, outputnode, [("out_k2", "k2_map")]),
        # Normalized CBV and CBF
        (normalize, outputnode, [("ncbv_file", "ncbv_map")]),
        (normalize_cbf, outputnode, [("ncbf_file", "ncbf_map")]),
        # Registration (.mat for internal consumers)
        (invert_dsc_xfm, outputnode, [("out_file", "dsc_to_t1w_xfm")]),
        (register, outputnode, [("out_matrix", "t1w_to_dsc_xfm")]),
        # Registration (.h5 for derivatives)
        (register, outputnode, [("out_h5", "t1w_to_dsc_h5")]),
        (reg_filmstrip, outputnode, [("filmstrip_file", "registration_filmstrip")]),
    ])

    # Convert inverted DSC→T1w .mat to .h5 (ref/mov swapped for inverse)
    convert_inv_h5 = pe.Node(
        niu.Function(
            input_names=["mat_path", "reference", "moving"],
            output_names=["out_h5"],
            function=_fsl_to_x5_node,
        ),
        name="convert_inv_dsc_h5",
    )
    workflow.connect([
        (invert_dsc_xfm, convert_inv_h5, [("out_file", "mat_path")]),
        (inputnode, convert_inv_h5, [("t1w_reference", "reference")]),
        (hdbet_dsc, convert_inv_h5, [("dsc_brain_file", "moving")]),
        (convert_inv_h5, outputnode, [("out_h5", "dsc_to_t1w_h5")]),
    ])

    # Structural underlay connections (conditional)
    if transform_t1_to_dsc is not None:
        workflow.connect([
            (transform_t1_to_dsc, outputnode, [("out_file", "t1w_in_dsc")]),
        ])

    if transform_t1_postcon_to_dsc is not None:
        workflow.connect([
            (transform_t1_postcon_to_dsc, outputnode, [("out_file", "t1w_postcon_in_dsc")]),
        ])

    if transform_flair_to_dsc is not None:
        workflow.connect([
            (transform_flair_to_dsc, outputnode, [("out_file", "flair_in_dsc")]),
        ])

    if transform_t2w_to_dsc is not None:
        workflow.connect([
            (transform_t2w_to_dsc, outputnode, [("out_file", "t2w_in_dsc")]),
        ])

    workflow.connect([
        # AIF QC plots
        (aif_qc_plot, outputnode, [
            ("out_aif_plot", "aif_curve_plot"),
            ("out_aif_location", "aif_location_plot"),
            ("out_aif_candidates", "aif_candidates_plot"),
        ]),
        # Motion QC plot
        (motion_wf, outputnode, [("outputnode.motion_qc_plot_file", "motion_qc_plot")]),
        # DSC QC plots
        (dsc_preproc_qc, outputnode, [
            ("plot_file", "dsc_preprocessing_qc_plot"),
            ("stats", "preprocessing_stats"),
        ]),
        (dsc_param_maps_qc, outputnode, [("plot_file", "dsc_param_maps_qc_plot")]),
        (dsc_leakage_qc, outputnode, [("plot_file", "dsc_leakage_qc_plot")]),
        # Brain mask filmstrip
        (dsc_brain_mask_qc, outputnode, [("out_file", "brain_mask_filmstrip")]),
    ])

    workflow.connect([
        (dsc_fit_qc, outputnode, [("plot_file", "dsc_fit_qc_plot")]),
    ])

    workflow.connect([
        # QC metrics
        (merge_qc, outputnode, [("merged_qc", "qc_metrics")]),
        (preload_wf, outputnode, [("outputnode.qc_dict", "preload_qc")]),
        (motion_wf, outputnode, [("outputnode.motion_qc_metrics", "motion_qc")]),
        (aif_qc_plot, outputnode, [("out_aif_metrics", "aif_metrics")]),
    ])

    # Smoothed outputs (only if smooth_fwhm > 0)
    if smooth_hemodynamic is not None:
        workflow.connect([
            (smooth_hemodynamic, outputnode, [("smoothed_files", "smooth_hemodynamic")]),
            (smooth_auc, outputnode, [("smoothed_files", "smooth_auc")]),
        ])

    # Store config in workflow for reference by child nodes
    workflow.config_dsc = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
