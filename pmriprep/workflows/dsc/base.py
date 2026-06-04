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


def init_dsc_registration_wf(
    *,
    registration: str,
    mem_gb: float,
    name: str = "dsc_registration_wf",
) -> pe.Workflow:
    """Build the DSC<->T1w registration + MNI->DSC transform-chain sub-workflow.

    Registers the DSC reference to T1w, and builds the MNI->DSC transform by
    concatenating MNI->T1w (inverted T1w->MNI) with T1w->DSC.

    Parameters
    ----------
    registration : str
        Registration strategy ("flirt" or "flirt_bbr").
    mem_gb : float
        Memory estimate (GB) for the registration node.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The registration sub-workflow.

    Inputs
    ------
    matched_t1w_file
        Histogram-matched T1w (moving image's target).
    brain_mask, wm_probseg
        T1w-space brain mask and WM segmentation (for BBR).
    dsc_brain
        Brain-extracted DSC reference.
    t1w_to_mni_xfm
        T1w->MNI transform (inverted to MNI->T1w internally).

    Outputs
    -------
    t1w_to_dsc_matrix
        T1w->DSC FLIRT matrix (register.out_matrix).
    registered_t1w
        T1w resampled into DSC space.
    t1w_to_dsc_h5
        T1w->DSC transform as ITK .h5.
    qc_metrics
        Registration QC.
    dsc_to_t1w_matrix
        DSC->T1w FLIRT matrix (inverse).
    mni_to_dsc_matrix
        MNI->DSC concatenated transform (for atlas resampling).
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "matched_t1w_file",
                "brain_mask",
                "wm_probseg",
                "dsc_brain",
                "t1w_to_mni_xfm",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w_to_dsc_matrix",
                "registered_t1w",
                "t1w_to_dsc_h5",
                "qc_metrics",
                "dsc_to_t1w_matrix",
                "mni_to_dsc_matrix",
            ]
        ),
        name="outputnode",
    )

    # Node 6: DSC to T1w Registration
    register = pe.Node(
        DSCRegistration(
            registration=registration,
            dof=6,  # Rigid body
            cost="normmi",
        ),
        name="register",
        mem_gb=mem_gb,
    )

    # MNI->DSC transform chain: invert T1w->MNI, invert T1w->DSC, concatenate
    invert_mni_xfm = pe.Node(ConvertXFM(invert_xfm=True), name="invert_mni_xfm")
    invert_dsc_xfm = pe.Node(ConvertXFM(invert_xfm=True), name="invert_dsc_xfm")
    concat_mni_to_dsc = pe.Node(ConvertXFM(concat_xfm=True), name="concat_mni_to_dsc")

    workflow.connect([
        (inputnode, register, [
            ("matched_t1w_file", "t1w_file"),
            ("brain_mask", "brain_mask"),
            ("wm_probseg", "wm_seg"),
            ("dsc_brain", "dsc_mean"),
        ]),
        (inputnode, invert_mni_xfm, [("t1w_to_mni_xfm", "in_file")]),
        (register, invert_dsc_xfm, [("out_matrix", "in_file")]),
        (invert_mni_xfm, concat_mni_to_dsc, [("out_file", "in_file")]),
        (register, concat_mni_to_dsc, [("out_matrix", "in_file2")]),
        (register, outputnode, [
            ("out_matrix", "t1w_to_dsc_matrix"),
            ("out_file", "registered_t1w"),
            ("out_h5", "t1w_to_dsc_h5"),
            ("qc_metrics", "qc_metrics"),
        ]),
        (invert_dsc_xfm, outputnode, [("out_file", "dsc_to_t1w_matrix")]),
        (concat_mni_to_dsc, outputnode, [("out_file", "mni_to_dsc_matrix")]),
    ])
    return workflow


def init_dsc_deconvolution_wf(
    *,
    dsc_map_kwargs: dict[str, Any],
    mem_gb: float,
    name: str = "dsc_deconvolution_wf",
) -> pe.Workflow:
    """Build the DSC deconvolution (perfusion-map computation) sub-workflow.

    Wraps :class:`~pmriprep.interfaces.dsc_compute.ComputeDSCMaps`, which performs
    residue-function deconvolution to produce CBF/CBV/MTT/Tmax and leakage (K2)
    maps from the leakage-corrected concentration time series and the AIF.

    Parameters
    ----------
    dsc_map_kwargs : dict
        Keyword arguments forwarded verbatim to ``ComputeDSCMaps`` (method,
        thresholds, inference settings, ``random_seed`` when set, etc.). Built by
        the orchestrator from the run configuration.
    mem_gb : float
        Memory estimate (GB) for the deconvolution node.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The deconvolution sub-workflow.

    Inputs
    ------
    concentration_file, signal_file, raw_concentration_file
        Leakage-corrected and raw concentration / signal time series.
    aif_file, measured_aif_file, aif_mask_file, aif_best_coord, aif_detection_level, baseline_frames
        Arterial input function and detection provenance.
    k2_map_input_file, preload_classification
        Preload-detection leakage inputs.
    brain_mask_file, enhancing_mask_file, nawm_mask_file
        DSC-space masks (brain, enhancing tumor, NAWM for rTTP normalization).
    tr, te, field_strength, fd_values
        Acquisition parameters and framewise displacement.

    Outputs
    -------
    out_rel_cbf, out_rel_cbv, out_rel_cbv_auc, out_rel_auc_raw, out_rel_auc_corr
        Relative perfusion and AUC maps.
    out_mtt, out_tmax, out_ttp, out_ttp_corrected, out_rttp
        Transit-time and timing maps.
    out_psr, out_ph
        Percentage signal recovery and peak height.
    out_auc_raw, out_auc_corr, out_auc_diff, out_k2
        Leakage (blood-brain-barrier) maps.
    out_cnr, out_snr, out_tsnr
        Quality maps.
    out_alpha, out_beta, out_cth
        Vascular-model parameters (undefined for SVD methods).
    out_cbf_std, out_cbv_std, out_cbf_cv, out_cbv_cv, out_cbf_ci95_lower, out_cbf_ci95_upper, out_cbv_ci95_lower, out_cbv_ci95_upper
        Bayesian uncertainty maps (parametric methods only).
    out_parametric_r2, out_fallback_mask
        Parametric-fit diagnostics (CPI / vascular only).
    out_qc_fit_data, qc_metrics
        Deconvolution QC-fit data and the QC ``qc_metrics`` dict.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "concentration_file",
                "k2_map_input_file",
                "aif_file",
                "measured_aif_file",
                "aif_mask_file",
                "aif_best_coord",
                "aif_detection_level",
                "baseline_frames",
                "brain_mask_file",
                "enhancing_mask_file",
                "tr",
                "te",
                "field_strength",
                "fd_values",
                "signal_file",
                "raw_concentration_file",
                "preload_classification",
                "nawm_mask_file",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "out_rel_cbf",
                "out_rel_cbv",
                "out_rel_cbv_auc",
                "out_rel_auc_raw",
                "out_rel_auc_corr",
                "out_mtt",
                "out_tmax",
                "out_psr",
                "out_ph",
                "out_ttp",
                "out_ttp_corrected",
                "out_rttp",
                "out_cnr",
                "out_auc_raw",
                "out_auc_corr",
                "out_auc_diff",
                "out_k2",
                "out_snr",
                "out_tsnr",
                "out_alpha",
                "out_beta",
                "out_cth",
                "out_cbf_std",
                "out_cbv_std",
                "out_cbf_cv",
                "out_cbv_cv",
                "out_cbf_ci95_lower",
                "out_cbf_ci95_upper",
                "out_cbv_ci95_lower",
                "out_cbv_ci95_upper",
                "out_parametric_r2",
                "out_fallback_mask",
                "out_qc_fit_data",
                "qc_metrics",
            ]
        ),
        name="outputnode",
    )

    compute_maps = pe.Node(
        ComputeDSCMaps(**dsc_map_kwargs),
        name="compute_maps",
        mem_gb=mem_gb,
    )

    workflow.connect([
        (inputnode, compute_maps, [
            ("concentration_file", "concentration_file"),
            ("k2_map_input_file", "k2_map_input_file"),
            ("aif_file", "aif_file"),
            ("measured_aif_file", "measured_aif_file"),
            ("aif_mask_file", "aif_mask_file"),
            ("aif_best_coord", "aif_best_coord"),
            ("aif_detection_level", "aif_detection_level"),
            ("baseline_frames", "baseline_frames"),
            ("brain_mask_file", "brain_mask_file"),
            ("enhancing_mask_file", "enhancing_mask_file"),
            ("tr", "tr"),
            ("te", "te"),
            ("field_strength", "field_strength"),
            ("fd_values", "fd_values"),
            ("signal_file", "signal_file"),
            ("raw_concentration_file", "raw_concentration_file"),
            ("preload_classification", "preload_classification"),
            ("nawm_mask_file", "nawm_mask_file"),
        ]),
        (compute_maps, outputnode, [
            ("out_rel_cbf", "out_rel_cbf"),
            ("out_rel_cbv", "out_rel_cbv"),
            ("out_rel_cbv_auc", "out_rel_cbv_auc"),
            ("out_rel_auc_raw", "out_rel_auc_raw"),
            ("out_rel_auc_corr", "out_rel_auc_corr"),
            ("out_mtt", "out_mtt"),
            ("out_tmax", "out_tmax"),
            ("out_psr", "out_psr"),
            ("out_ph", "out_ph"),
            ("out_ttp", "out_ttp"),
            ("out_ttp_corrected", "out_ttp_corrected"),
            ("out_rttp", "out_rttp"),
            ("out_cnr", "out_cnr"),
            ("out_auc_raw", "out_auc_raw"),
            ("out_auc_corr", "out_auc_corr"),
            ("out_auc_diff", "out_auc_diff"),
            ("out_k2", "out_k2"),
            ("out_snr", "out_snr"),
            ("out_tsnr", "out_tsnr"),
            ("out_alpha", "out_alpha"),
            ("out_beta", "out_beta"),
            ("out_cth", "out_cth"),
            ("out_cbf_std", "out_cbf_std"),
            ("out_cbv_std", "out_cbv_std"),
            ("out_cbf_cv", "out_cbf_cv"),
            ("out_cbv_cv", "out_cbv_cv"),
            ("out_cbf_ci95_lower", "out_cbf_ci95_lower"),
            ("out_cbf_ci95_upper", "out_cbf_ci95_upper"),
            ("out_cbv_ci95_lower", "out_cbv_ci95_lower"),
            ("out_cbv_ci95_upper", "out_cbv_ci95_upper"),
            ("out_parametric_r2", "out_parametric_r2"),
            ("out_fallback_mask", "out_fallback_mask"),
            ("out_qc_fit_data", "out_qc_fit_data"),
            ("qc_metrics", "qc_metrics"),
        ]),
    ])
    return workflow


def init_dsc_normalization_wf(
    *,
    derivatives_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str],
    name: str = "dsc_normalization_wf",
) -> pe.Workflow:
    """Build the DSC CBV / CBF normalization sub-workflow.

    Normalizes the relative CBV and CBF maps to a reference tissue (NAWM when
    available, else a whole-brain fallback), producing nCBV / nCBF maps and the
    shared normalization factor. CBV and CBF use the same normalization method.

    Parameters
    ----------
    derivatives_dir : str or pathlib.Path
        Derivatives output directory (forwarded to the normalization nodes).
    subject_id : str
        BIDS subject label.
    session_id : str or None
        BIDS session label.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The normalization sub-workflow.

    Inputs
    ------
    cbv_file
        Relative CBV map to normalize.
    cbf_file
        Relative CBF map to normalize.
    brain_mask_file
        DSC-space brain mask (whole-brain fallback reference).
    nawm_mask_file
        NAWM mask (preferred normalization reference).

    Outputs
    -------
    ncbv_file
        Normalized CBV map.
    ncbf_file
        Normalized CBF map.
    normalization_factor
        The CBV normalization factor (surfaced for QC).
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["cbv_file", "cbf_file", "brain_mask_file", "nawm_mask_file"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["ncbv_file", "ncbf_file", "normalization_factor"]
        ),
        name="outputnode",
    )

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

    workflow.connect([
        (inputnode, normalize, [
            ("cbv_file", "cbv_file"),
            ("brain_mask_file", "brain_mask_file"),
            ("nawm_mask_file", "nawm_mask_file"),
        ]),
        (inputnode, normalize_cbf, [
            ("cbf_file", "cbf_file"),
            ("brain_mask_file", "brain_mask_file"),
            ("nawm_mask_file", "nawm_mask_file"),
        ]),
        (normalize, outputnode, [
            ("ncbv_file", "ncbv_file"),
            ("normalization_factor", "normalization_factor"),
        ]),
        (normalize_cbf, outputnode, [
            ("ncbf_file", "ncbf_file"),
        ]),
    ])
    return workflow


def init_dsc_smoothing_wf(
    *,
    fwhm: float,
    name: str = "dsc_smoothing_wf",
) -> pe.Workflow:
    """Build the DSC perfusion-map smoothing sub-workflow.

    Spatially smooths the intensive hemodynamic maps (MTT, rel-CBF, rel-CBV,
    nCBF, nCBV) and the leakage-corrected relative AUC map (a CBV surrogate)
    within the brain mask. Timing / leakage maps (Tmax, TTP, PSR, PH, K2,
    auc_diff) are deliberately NOT smoothed. The orchestrator instantiates this
    sub-workflow only when ``config.smooth_fwhm > 0``.

    Parameters
    ----------
    fwhm : float
        Smoothing-kernel full-width-at-half-maximum (mm).
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The smoothing sub-workflow.

    Inputs
    ------
    mtt, rel_cbf, rel_cbv, ncbf, ncbv
        Hemodynamic maps (assembled into one dict internally before smoothing).
    rel_auc_corr
        Leakage-corrected relative AUC map (CBV surrogate).
    brain_mask_file
        DSC-space brain mask (smoothing is restricted to brain voxels).

    Outputs
    -------
    smoothed_hemodynamic
        Smoothed hemodynamic map files.
    smoothed_auc
        Smoothed AUC map file(s).
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "mtt",
                "rel_cbf",
                "rel_cbv",
                "ncbf",
                "ncbv",
                "rel_auc_corr",
                "brain_mask_file",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["smoothed_hemodynamic", "smoothed_auc"]
        ),
        name="outputnode",
    )

    # Hemodynamic maps: CBF, CBV, MTT, nCBF, nCBV
    # Physics-approved: intensive quantities that vary smoothly
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
    smooth_hemodynamic = pe.Node(
        niu.Function(
            input_names=["map_files", "brain_mask_file", "fwhm"],
            output_names=["smoothed_files"],
            function=_smooth_maps,
        ),
        name="smooth_hemodynamic",
    )
    smooth_hemodynamic.inputs.fwhm = fwhm

    # Leakage-corrected AUC (CBV surrogate); only auc_corr is smoothed
    build_auc_dict = pe.Node(
        niu.Function(
            input_names=["rel_auc_corr"],
            output_names=["map_dict"],
            function=_build_auc_corr_dict,
        ),
        name="build_auc_dict",
    )
    smooth_auc = pe.Node(
        niu.Function(
            input_names=["map_files", "brain_mask_file", "fwhm"],
            output_names=["smoothed_files"],
            function=_smooth_maps,
        ),
        name="smooth_auc",
    )
    smooth_auc.inputs.fwhm = fwhm

    workflow.connect([
        (inputnode, build_hemo_dict, [
            ("mtt", "mtt"),
            ("rel_cbf", "rel_cbf"),
            ("rel_cbv", "rel_cbv"),
            ("ncbf", "ncbf"),
            ("ncbv", "ncbv"),
        ]),
        (build_hemo_dict, smooth_hemodynamic, [("map_dict", "map_files")]),
        (inputnode, smooth_hemodynamic, [("brain_mask_file", "brain_mask_file")]),
        (inputnode, build_auc_dict, [("rel_auc_corr", "rel_auc_corr")]),
        (build_auc_dict, smooth_auc, [("map_dict", "map_files")]),
        (inputnode, smooth_auc, [("brain_mask_file", "brain_mask_file")]),
        (smooth_hemodynamic, outputnode, [("smoothed_files", "smoothed_hemodynamic")]),
        (smooth_auc, outputnode, [("smoothed_files", "smoothed_auc")]),
    ])
    return workflow


def init_dsc_lesion_stats_wf(
    *,
    subject_id: str,
    session_id: Optional[str],
    name: str = "dsc_lesion_stats_wf",
) -> pe.Workflow:
    """Build the DSC lesion / ROI / user-mask statistics sub-workflow.

    Extracts per-region perfusion statistics from the DSC parameter maps within
    three (independently optional) masks: the tumor-segmentation lesion mask, a
    user-supplied ROI mask, and a user-supplied lesion mask. The perfusion maps,
    NAWM reference, field strength, and AIF QC are shared across all three
    extractions; the ROI and user-lesion extractions force a binary mask.

    Parameters
    ----------
    subject_id : str
        BIDS subject label.
    session_id : str or None
        BIDS session label.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The lesion-statistics sub-workflow.

    Inputs
    ------
    cbf_map, cbv_map, mtt_map, tmax_map, ttp_map, ttp_corrected_map, rttp_map
        DSC perfusion and transit-time maps (shared by all three nodes).
    psr_map, ph_map
        Percentage signal recovery and peak height.
    auc_corr_map, auc_diff_map, rel_auc_corr_map, k2_map
        Leakage maps.
    ncbv_map, ncbf_map
        Normalized CBV / CBF maps.
    nawm_mask
        NAWM reference mask.
    field_strength_t
        Field strength (Tesla).
    aif_qc
        AIF QC-metrics dict (surfaced into the stats sidecars).
    tumor_lesion_mask, mask_origin
        Tumor-segmentation lesion mask (DSC space) and its provenance label.
    roi_mask
        Validated user ROI mask (DSC native space).
    user_mask
        Validated user lesion mask (DSC native space).

    Outputs
    -------
    lesion_stats_json
        Tumor-lesion statistics sidecar.
    roi_stats_json
        User-ROI statistics sidecar.
    user_lesion_stats_json
        User-lesion statistics sidecar.
    """
    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "cbf_map",
                "cbv_map",
                "mtt_map",
                "tmax_map",
                "psr_map",
                "ph_map",
                "ttp_map",
                "ttp_corrected_map",
                "rttp_map",
                "auc_corr_map",
                "auc_diff_map",
                "rel_auc_corr_map",
                "k2_map",
                "ncbv_map",
                "ncbf_map",
                "nawm_mask",
                "field_strength_t",
                "aif_qc",
                "tumor_lesion_mask",
                "mask_origin",
                "roi_mask",
                "user_mask",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["lesion_stats_json", "roi_stats_json", "user_lesion_stats_json"]
        ),
        name="outputnode",
    )

    lesion_stats = pe.Node(DSCLesionStats(), name="lesion_stats")
    lesion_stats.inputs.subject_id = subject_id
    lesion_stats.inputs.session_id = session_id

    roi_stats = pe.Node(DSCLesionStats(force_binary=True), name="roi_stats")
    roi_stats.inputs.subject_id = subject_id
    roi_stats.inputs.session_id = session_id

    user_lesion_stats = pe.Node(
        DSCLesionStats(force_binary=True), name="user_lesion_stats"
    )
    user_lesion_stats.inputs.subject_id = subject_id
    user_lesion_stats.inputs.session_id = session_id

    # Perfusion maps + NAWM + field strength + AIF QC are shared by all three
    # stats nodes; the lesion mask differs (tumor-seg / ROI / user), and only
    # the tumor-seg node receives the mask-origin provenance label.
    shared_inputs = [
        ("cbf_map", "cbf_map"),
        ("cbv_map", "cbv_map"),
        ("mtt_map", "mtt_map"),
        ("tmax_map", "tmax_map"),
        ("psr_map", "psr_map"),
        ("ph_map", "ph_map"),
        ("ttp_map", "ttp_map"),
        ("ttp_corrected_map", "ttp_corrected_map"),
        ("rttp_map", "rttp_map"),
        ("auc_corr_map", "auc_corr_map"),
        ("auc_diff_map", "auc_diff_map"),
        ("rel_auc_corr_map", "rel_auc_corr_map"),
        ("k2_map", "k2_map"),
        ("ncbv_map", "ncbv_map"),
        ("ncbf_map", "ncbf_map"),
        ("nawm_mask", "nawm_mask"),
        ("field_strength_t", "field_strength_t"),
        ("aif_qc", "aif_qc"),
    ]
    workflow.connect([
        (inputnode, lesion_stats, shared_inputs + [
            ("tumor_lesion_mask", "lesion_mask"),
            ("mask_origin", "mask_origin"),
        ]),
        (inputnode, roi_stats, shared_inputs + [("roi_mask", "lesion_mask")]),
        (inputnode, user_lesion_stats, shared_inputs + [("user_mask", "lesion_mask")]),
        (lesion_stats, outputnode, [("stats_json", "lesion_stats_json")]),
        (roi_stats, outputnode, [("stats_json", "roi_stats_json")]),
        (user_lesion_stats, outputnode, [("stats_json", "user_lesion_stats_json")]),
    ])
    return workflow


def init_dsc_qc_wf(
    *,
    subject_id: str,
    session_id: Optional[str],
    aif_source: str,
    registration_bbr: bool,
    baseline_frames: int,
    name: str = "dsc_qc_wf",
) -> pe.Workflow:
    """Build the DSC quality-control / reports sub-workflow.

    Generates the per-stage QC figures (registration filmstrip, AIF plots,
    preprocessing, parameter-map, leakage, fit, and brain-mask filmstrips) and
    merges the per-sub-workflow QC metric dicts into a single ``qc_metrics``
    payload for the reports/sidecars.

    Parameters
    ----------
    subject_id : str
        BIDS subject label.
    session_id : str or None
        BIDS session label.
    aif_source : str
        AIF provenance label for the merged QC ("external", "synthetic", "auto").
    registration_bbr : bool
        Whether boundary-based registration was used (recorded in merged QC).
    baseline_frames : int
        Pre-bolus baseline frame count (S0) for the preprocessing QC plot.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The QC sub-workflow.

    Inputs
    ------
    motion_qc_metrics, motion_summary, preload_qc_dict, aif_qc_metrics, deconv_qc_metrics, registration_qc_metrics, normalization_factor
        Per-sub-workflow QC metric dicts / scalars merged into ``qc_metrics``.
    prebolus_mean_file, dsc_moco, registered_t1w, brain_mask_file
        Images for the registration / preprocessing / brain-mask filmstrips.
    out_aif_curve, out_aif_mask, out_raw_aif_curve, gamma_fit_applied, concentration_file, corrected_concentration_file
        AIF, concentration, and leakage-QC inputs.
    rel_cbf, rel_cbv_auc, mtt, tmax, psr, ttp_corrected, k2, auc_raw, auc_corr, auc_diff, qc_fit_data
        DSC parameter / leakage / fit maps for the QC plots.
    t1w_to_dsc_matrix, t1w_to_mni_xfm, tr, te
        Transforms and acquisition parameters.

    Outputs
    -------
    qc_metrics
        Merged QC-metrics payload.
    registration_filmstrip, brain_mask_filmstrip
        Registration and brain-mask filmstrip figures.
    aif_curve_plot, aif_location_plot, aif_candidates_plot, aif_metrics
        AIF QC figures and metrics.
    dsc_preprocessing_qc_plot, preprocessing_stats, dsc_param_maps_qc_plot, dsc_leakage_qc_plot, dsc_fit_qc_plot
        DSC preprocessing / parameter-map / leakage / fit QC figures and stats.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "motion_qc_metrics",
                "motion_summary",
                "prebolus_mean_file",
                "dsc_moco",
                "preload_qc_dict",
                "corrected_concentration_file",
                "aif_qc_metrics",
                "out_aif_curve",
                "out_aif_mask",
                "out_raw_aif_curve",
                "gamma_fit_applied",
                "concentration_file",
                "deconv_qc_metrics",
                "rel_cbf",
                "rel_cbv_auc",
                "mtt",
                "tmax",
                "psr",
                "ttp_corrected",
                "k2",
                "auc_raw",
                "auc_corr",
                "auc_diff",
                "qc_fit_data",
                "registration_qc_metrics",
                "registered_t1w",
                "t1w_to_dsc_matrix",
                "normalization_factor",
                "brain_mask_file",
                "tr",
                "te",
                "t1w_to_mni_xfm",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "registration_filmstrip",
                "aif_curve_plot",
                "aif_location_plot",
                "aif_candidates_plot",
                "aif_metrics",
                "dsc_preprocessing_qc_plot",
                "preprocessing_stats",
                "dsc_param_maps_qc_plot",
                "dsc_leakage_qc_plot",
                "brain_mask_filmstrip",
                "dsc_fit_qc_plot",
                "qc_metrics",
            ]
        ),
        name="outputnode",
    )

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
    merge_qc.inputs.registration_bbr = registration_bbr

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

    aif_qc_plot = pe.Node(
        AIFQCPlot(
            subject_id=subject_id,
            session_id=session_id or "",
        ),
        name="aif_qc_plot",
    )

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
    dsc_preproc_qc.inputs.baseline_frames = baseline_frames
    dsc_preproc_qc.inputs.subject_id = subject_id
    dsc_preproc_qc.inputs.session_id = session_id or ""

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
        (inputnode, merge_qc, [
            ("motion_qc_metrics", "motion_qc"),
            ("motion_summary", "motion_summary"),
            ("preload_qc_dict", "preload_qc"),
            ("aif_qc_metrics", "aif_qc"),
            ("deconv_qc_metrics", "deconv_qc"),
            ("registration_qc_metrics", "registration_qc"),
            ("normalization_factor", "normalization_factor"),
        ]),
        (inputnode, reg_filmstrip, [
            ("prebolus_mean_file", "dsc_mean_file"),
            ("registered_t1w", "registered_t1w_file"),
            ("brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, aif_qc_plot, [
            ("dsc_moco", "dsc_file"),
            ("aif_qc_metrics", "qc_metrics"),
            ("out_aif_curve", "aif_curve_file"),
            ("out_aif_mask", "aif_mask_file"),
            ("out_raw_aif_curve", "raw_aif_curve_file"),
            ("gamma_fit_applied", "gamma_fit_applied"),
            ("t1w_to_dsc_matrix", "t1w_to_dsc_xfm"),
            ("brain_mask_file", "brain_mask_file"),
            ("tr", "tr"),
            ("t1w_to_mni_xfm", "t1w_to_mni_xfm"),
        ]),
        (inputnode, dsc_preproc_qc, [
            ("dsc_moco", "dsc_signal_file"),
            ("brain_mask_file", "brain_mask_file"),
            ("tr", "tr"),
            ("te", "te"),
        ]),
        (inputnode, dsc_param_maps_qc, [
            ("rel_cbf", "rel_cbf_file"),
            ("rel_cbv_auc", "rel_cbv_file"),
            ("mtt", "mtt_file"),
            ("tmax", "tmax_file"),
            ("psr", "psr_file"),
            ("ttp_corrected", "ttp_file"),
            ("brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, dsc_leakage_qc, [
            ("preload_qc_dict", "preload_qc"),
            ("corrected_concentration_file", "corrected_concentration_file"),
            ("concentration_file", "concentration_file"),
            ("psr", "psr_file"),
            ("k2", "k2_map_file"),
            ("auc_raw", "auc_raw_file"),
            ("auc_corr", "auc_corr_file"),
            ("auc_diff", "auc_diff_file"),
            ("brain_mask_file", "brain_mask_file"),
            ("tr", "tr"),
        ]),
        (inputnode, dsc_fit_qc, [
            ("rel_cbf", "cbf_file"),
            ("qc_fit_data", "qc_fit_data_file"),
            ("brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, dsc_brain_mask_qc, [
            ("prebolus_mean_file", "mean_file"),
            ("brain_mask_file", "brain_mask_file"),
        ]),
        (aif_qc_plot, merge_qc, [("out_aif_metrics", "aif_metrics")]),
        (dsc_preproc_qc, merge_qc, [("stats", "preprocessing_stats")]),
        (merge_qc, outputnode, [("merged_qc", "qc_metrics")]),
        (reg_filmstrip, outputnode, [("filmstrip_file", "registration_filmstrip")]),
        (aif_qc_plot, outputnode, [
            ("out_aif_plot", "aif_curve_plot"),
            ("out_aif_location", "aif_location_plot"),
            ("out_aif_candidates", "aif_candidates_plot"),
            ("out_aif_metrics", "aif_metrics"),
        ]),
        (dsc_preproc_qc, outputnode, [
            ("plot_file", "dsc_preprocessing_qc_plot"),
            ("stats", "preprocessing_stats"),
        ]),
        (dsc_param_maps_qc, outputnode, [("plot_file", "dsc_param_maps_qc_plot")]),
        (dsc_leakage_qc, outputnode, [("plot_file", "dsc_leakage_qc_plot")]),
        (dsc_fit_qc, outputnode, [("plot_file", "dsc_fit_qc_plot")]),
        (dsc_brain_mask_qc, outputnode, [("out_file", "brain_mask_filmstrip")]),
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
    deconvolution_wf = init_dsc_deconvolution_wf(
        dsc_map_kwargs=dsc_map_kwargs,
        mem_gb=_heavy_mem,
    )

    # -------------------------------------------------------------------
    # Registration sub-workflow: DSC<->T1w + MNI->DSC transform chain
    # -------------------------------------------------------------------
    registration_wf = init_dsc_registration_wf(
        registration=config.registration,
        mem_gb=_static_mem,
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
    # Node 6f: Lesion / ROI / user-mask statistics extraction
    # -------------------------------------------------------------------
    lesion_stats_wf = init_dsc_lesion_stats_wf(
        subject_id=subject_id,
        session_id=session_id,
    )

    normalization_wf = init_dsc_normalization_wf(
        derivatives_dir=derivatives_dir,
        subject_id=subject_id,
        session_id=session_id,
    )

    # -------------------------------------------------------------------
    # Node 7c: Smoothing (conditional on config.smooth_fwhm > 0)
    # -------------------------------------------------------------------
    smoothing_wf = None
    if config.smooth_fwhm > 0:
        smoothing_wf = init_dsc_smoothing_wf(fwhm=config.smooth_fwhm)

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

    qc_wf = init_dsc_qc_wf(
        subject_id=subject_id,
        session_id=session_id,
        aif_source=aif_source,
        registration_bbr=config.registration == "flirt_bbr",
        baseline_frames=config.pre_bolus,
    )

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
        (preload_wf, deconvolution_wf, [
            ("outputnode.corrected_concentration_file", "inputnode.concentration_file"),
            ("outputnode.k2_map_file", "inputnode.k2_map_input_file"),
        ]),
        (conc_aif_wf, deconvolution_wf, [
            ("outputnode.out_aif_curve", "inputnode.aif_file"),
            ("outputnode.out_measured_aif_curve", "inputnode.measured_aif_file"),
            ("outputnode.out_aif_mask", "inputnode.aif_mask_file"),
            ("outputnode.out_aif_coord", "inputnode.aif_best_coord"),
            ("outputnode.detection_level", "inputnode.aif_detection_level"),
            ("outputnode.aif_baseline_frames", "inputnode.baseline_frames"),
        ]),
        (hdbet_dsc, deconvolution_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
        ]),
        (transform_enhancing_mask, deconvolution_wf, [
            ("lesion_mask_dsc_file", "inputnode.enhancing_mask_file"),
        ]),
        (inputnode, deconvolution_wf, [
            ("tr", "inputnode.tr"),
            ("te", "inputnode.te"),
            ("field_strength", "inputnode.field_strength"),
        ]),
        (motion_wf, deconvolution_wf, [
            ("outputnode.fd_values", "inputnode.fd_values"),
        ]),
        (conc_aif_wf, deconvolution_wf, [
            ("outputnode.signal_file", "inputnode.signal_file"),
            ("outputnode.concentration_file", "inputnode.raw_concentration_file"),
        ]),
        (preload_wf, deconvolution_wf, [
            ("outputnode.preload_classification", "inputnode.preload_classification"),
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

    # Feed the registration sub-workflow
    workflow.connect([
        (hist_match_t1w, registration_wf, [("matched_t1w_file", "inputnode.matched_t1w_file")]),
        (inputnode, registration_wf, [
            ("brain_mask", "inputnode.brain_mask"),
            ("wm_probseg", "inputnode.wm_probseg"),
            ("t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
        ]),
        (hdbet_dsc, registration_wf, [("dsc_brain_file", "inputnode.dsc_brain")]),
    ])

    # Step 4: Transform arterial atlas from MNI to DSC space
    # Use pre-bolus mean as reference for output geometry
    workflow.connect([
        (registration_wf, transform_atlas, [("outputnode.mni_to_dsc_matrix", "in_matrix_file")]),
        (motion_wf, transform_atlas, [("outputnode.prebolus_mean_file", "reference")]),
    ])

    # Connect transformed atlas to AIF detection
    workflow.connect([
        (transform_atlas, conc_aif_wf, [("out_file", "inputnode.atlas_mask_file")]),
    ])

    # Transform + connect full atlas (posterior circulation tier)
    if isdefined(transform_full_atlas.inputs.in_file):
        workflow.connect([
            (registration_wf, transform_full_atlas, [("outputnode.mni_to_dsc_matrix", "in_matrix_file")]),
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
        (registration_wf, transform_nawm, [("outputnode.t1w_to_dsc_matrix", "transform_matrix")]),
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
        (transform_nawm, deconvolution_wf, [
            ("nawm_mask_dsc_file", "inputnode.nawm_mask_file"),
        ]),
    ])

    # Feed the normalization sub-workflow (rel-CBV / rel-CBF + masks).
    # The NAWM (preferred) and brain (fallback) masks fan out to both the CBV
    # and CBF normalizers inside the sub-workflow.
    workflow.connect([
        (deconvolution_wf, normalization_wf, [
            ("outputnode.out_rel_cbv_auc", "inputnode.cbv_file"),
            ("outputnode.out_rel_cbf", "inputnode.cbf_file"),
        ]),
        (transform_nawm, normalization_wf, [
            ("nawm_mask_dsc_file", "inputnode.nawm_mask_file"),
        ]),
        (hdbet_dsc, normalization_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
        ]),
    ])

    # Feed the QC sub-workflow: per-stage QC metrics from every
    # sub-workflow, perfusion/leakage maps, the DSC brain mask, and
    # acquisition parameters.
    workflow.connect([
        (motion_wf, qc_wf, [
            ("outputnode.motion_qc_metrics", "inputnode.motion_qc_metrics"),
            ("outputnode.motion_summary", "inputnode.motion_summary"),
            ("outputnode.prebolus_mean_file", "inputnode.prebolus_mean_file"),
            ("outputnode.dsc_moco", "inputnode.dsc_moco"),
        ]),
        (preload_wf, qc_wf, [
            ("outputnode.qc_dict", "inputnode.preload_qc_dict"),
            ("outputnode.corrected_concentration_file", "inputnode.corrected_concentration_file"),
        ]),
        (conc_aif_wf, qc_wf, [
            ("outputnode.qc_metrics", "inputnode.aif_qc_metrics"),
            ("outputnode.out_aif_curve", "inputnode.out_aif_curve"),
            ("outputnode.out_aif_mask", "inputnode.out_aif_mask"),
            ("outputnode.out_raw_aif_curve", "inputnode.out_raw_aif_curve"),
            ("outputnode.gamma_fit_applied", "inputnode.gamma_fit_applied"),
            ("outputnode.concentration_file", "inputnode.concentration_file"),
        ]),
        (deconvolution_wf, qc_wf, [
            ("outputnode.qc_metrics", "inputnode.deconv_qc_metrics"),
            ("outputnode.out_rel_cbf", "inputnode.rel_cbf"),
            ("outputnode.out_rel_cbv_auc", "inputnode.rel_cbv_auc"),
            ("outputnode.out_mtt", "inputnode.mtt"),
            ("outputnode.out_tmax", "inputnode.tmax"),
            ("outputnode.out_psr", "inputnode.psr"),
            ("outputnode.out_ttp_corrected", "inputnode.ttp_corrected"),
            ("outputnode.out_k2", "inputnode.k2"),
            ("outputnode.out_auc_raw", "inputnode.auc_raw"),
            ("outputnode.out_auc_corr", "inputnode.auc_corr"),
            ("outputnode.out_auc_diff", "inputnode.auc_diff"),
            ("outputnode.out_qc_fit_data", "inputnode.qc_fit_data"),
        ]),
        (registration_wf, qc_wf, [
            ("outputnode.qc_metrics", "inputnode.registration_qc_metrics"),
            ("outputnode.registered_t1w", "inputnode.registered_t1w"),
            ("outputnode.t1w_to_dsc_matrix", "inputnode.t1w_to_dsc_matrix"),
        ]),
        (normalization_wf, qc_wf, [("outputnode.normalization_factor", "inputnode.normalization_factor")]),
        (hdbet_dsc, qc_wf, [("dsc_brain_mask_file", "inputnode.brain_mask_file")]),
        (inputnode, qc_wf, [
            ("tr", "inputnode.tr"),
            ("te", "inputnode.te"),
            ("t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
        ]),
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
        (registration_wf, transform_lesion, [("outputnode.t1w_to_dsc_matrix", "transform_matrix")]),
    ])

    # Transform enhancing mask from T1w to DSC space
    workflow.connect([
        (inputnode, transform_enhancing_mask, [("enhancing_mask", "lesion_mask_file")]),
        (motion_wf, transform_enhancing_mask, [("outputnode.prebolus_mean_file", "reference_file")]),
        (registration_wf, transform_enhancing_mask, [("outputnode.t1w_to_dsc_matrix", "transform_matrix")]),
    ])

    # -------------------------------------------------------------------
    # Lesion / ROI / user-mask statistics: feed the stats sub-workflow.
    # The perfusion maps + NAWM + field strength + AIF QC are shared by all
    # three stats nodes (fanned out inside the sub-wf); the three lesion masks
    # (tumor-seg, user ROI, user lesion) route to their respective nodes.
    # -------------------------------------------------------------------
    workflow.connect([
        (deconvolution_wf, lesion_stats_wf, [
            ("outputnode.out_rel_cbf", "inputnode.cbf_map"),
            ("outputnode.out_rel_cbv_auc", "inputnode.cbv_map"),
            ("outputnode.out_mtt", "inputnode.mtt_map"),
            ("outputnode.out_tmax", "inputnode.tmax_map"),
            ("outputnode.out_psr", "inputnode.psr_map"),
            ("outputnode.out_ph", "inputnode.ph_map"),
            ("outputnode.out_ttp", "inputnode.ttp_map"),
            ("outputnode.out_ttp_corrected", "inputnode.ttp_corrected_map"),
            ("outputnode.out_rttp", "inputnode.rttp_map"),
            ("outputnode.out_auc_corr", "inputnode.auc_corr_map"),
            ("outputnode.out_auc_diff", "inputnode.auc_diff_map"),
            ("outputnode.out_rel_auc_corr", "inputnode.rel_auc_corr_map"),
            ("outputnode.out_k2", "inputnode.k2_map"),
        ]),
        (normalization_wf, lesion_stats_wf, [
            ("outputnode.ncbv_file", "inputnode.ncbv_map"),
            ("outputnode.ncbf_file", "inputnode.ncbf_map"),
        ]),
        (conc_aif_wf, lesion_stats_wf, [("outputnode.qc_metrics", "inputnode.aif_qc")]),
        (transform_nawm, lesion_stats_wf, [("nawm_mask_dsc_file", "inputnode.nawm_mask")]),
        (inputnode, lesion_stats_wf, [("field_strength", "inputnode.field_strength_t")]),
        (transform_lesion, lesion_stats_wf, [("lesion_mask_dsc_file", "inputnode.tumor_lesion_mask")]),
        (combine_masks, lesion_stats_wf, [("mask_origin", "inputnode.mask_origin")]),
        (validate_roi_mask, lesion_stats_wf, [("validated_mask", "inputnode.roi_mask")]),
        (validate_user_lesion_mask, lesion_stats_wf, [("validated_mask", "inputnode.user_mask")]),
    ])

    # Stats sub-workflow outputs + the DSC-space masks they were computed on
    # (the masks come from transform/validate nodes that stay in the parent).
    workflow.connect([
        (transform_lesion, outputnode, [("lesion_mask_dsc_file", "lesion_mask_dsc")]),
        (validate_roi_mask, outputnode, [("validated_mask", "roi_mask_dsc")]),
        (validate_user_lesion_mask, outputnode, [("validated_mask", "user_lesion_mask_dsc")]),
        (lesion_stats_wf, outputnode, [
            ("outputnode.lesion_stats_json", "lesion_stats_json"),
            ("outputnode.roi_stats_json", "roi_stats_json"),
            ("outputnode.user_lesion_stats_json", "user_lesion_stats_json"),
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
            (registration_wf, transform_t1_to_dsc, [("outputnode.t1w_to_dsc_matrix", "transform_matrix")]),
        ])

    if transform_t1_postcon_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_t1_postcon_to_dsc, [("t1w_postcon_underlay", "structural_file")]),
            (motion_wf, transform_t1_postcon_to_dsc, [("outputnode.prebolus_mean_file", "reference_file")]),
            (registration_wf, transform_t1_postcon_to_dsc, [("outputnode.t1w_to_dsc_matrix", "transform_matrix")]),
        ])

    if transform_flair_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_flair_to_dsc, [("flair_in_t1w", "structural_file")]),
            (motion_wf, transform_flair_to_dsc, [("outputnode.prebolus_mean_file", "reference_file")]),
            (registration_wf, transform_flair_to_dsc, [("outputnode.t1w_to_dsc_matrix", "transform_matrix")]),
        ])

    if transform_t2w_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_t2w_to_dsc, [("t2w_in_t1w", "structural_file")]),
            (motion_wf, transform_t2w_to_dsc, [("outputnode.prebolus_mean_file", "reference_file")]),
            (registration_wf, transform_t2w_to_dsc, [("outputnode.t1w_to_dsc_matrix", "transform_matrix")]),
        ])

    # -------------------------------------------------------------------
    # Smoothing connections (only if smooth_fwhm > 0)
    # -------------------------------------------------------------------
    if smoothing_wf is not None:
        # nCBF/nCBV come from the normalization sub-wf; the hemodynamic +
        # AUC-corr maps from deconvolution; brain mask fans out to both
        # smoothers inside the sub-workflow.
        workflow.connect([
            (deconvolution_wf, smoothing_wf, [
                ("outputnode.out_mtt", "inputnode.mtt"),
                ("outputnode.out_rel_cbf", "inputnode.rel_cbf"),
                ("outputnode.out_rel_cbv_auc", "inputnode.rel_cbv"),
                ("outputnode.out_rel_auc_corr", "inputnode.rel_auc_corr"),
            ]),
            (normalization_wf, smoothing_wf, [
                ("outputnode.ncbf_file", "inputnode.ncbf"),
                ("outputnode.ncbv_file", "inputnode.ncbv"),
            ]),
            (hdbet_dsc, smoothing_wf, [
                ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
            ]),
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
        (deconvolution_wf, outputnode, [
            ("outputnode.out_mtt", "mtt_map"),
            ("outputnode.out_tmax", "tmax_map"),
            ("outputnode.out_psr", "psr_map"),
            ("outputnode.out_ph", "ph_map"),
            ("outputnode.out_ttp", "ttp_map"),
            ("outputnode.out_ttp_corrected", "ttp_corrected_map"),
            ("outputnode.out_rttp", "rttp_map"),
            ("outputnode.out_cnr", "cnr_map"),
            ("outputnode.out_auc_diff", "auc_diff_map"),
        ]),
        # Relative maps
        (deconvolution_wf, outputnode, [
            ("outputnode.out_rel_cbf", "rel_cbf_map"),
            ("outputnode.out_rel_cbv", "rel_cbv_map"),
            ("outputnode.out_rel_cbv_auc", "rel_cbv_auc_map"),
            ("outputnode.out_rel_auc_raw", "rel_auc_raw_map"),
            ("outputnode.out_rel_auc_corr", "rel_auc_corr_map"),
        ]),
        # Quality metrics
        (deconvolution_wf, outputnode, [
            ("outputnode.out_snr", "snr_map"),
            ("outputnode.out_tsnr", "tsnr_map"),
        ]),
        # Vascular model parameters (optional — undefined for SVD methods)
        (deconvolution_wf, outputnode, [
            ("outputnode.out_alpha", "alpha_map"),
            ("outputnode.out_beta", "beta_map"),
            ("outputnode.out_cth", "cth_map"),
        ]),
        # Bayesian uncertainty maps (only for parametric methods: cpi, vascular)
        (deconvolution_wf, outputnode, [
            ("outputnode.out_cbf_std", "cbf_std_map"),
            ("outputnode.out_cbv_std", "cbv_std_map"),
            ("outputnode.out_cbf_cv", "cbf_cv_map"),
            ("outputnode.out_cbv_cv", "cbv_cv_map"),
            ("outputnode.out_cbf_ci95_lower", "cbf_ci95_lower_map"),
            ("outputnode.out_cbf_ci95_upper", "cbf_ci95_upper_map"),
            ("outputnode.out_cbv_ci95_lower", "cbv_ci95_lower_map"),
            ("outputnode.out_cbv_ci95_upper", "cbv_ci95_upper_map"),
        ]),
        # Parametric-fit diagnostics (CPI/Vascular only;
        # undefined for SVD methods)
        (deconvolution_wf, outputnode, [
            ("outputnode.out_parametric_r2", "parametric_r2_map"),
            ("outputnode.out_fallback_mask", "fallback_mask"),
        ]),
        # AIF outputs
        (conc_aif_wf, outputnode, [
            ("outputnode.out_aif_curve", "aif_curve"),
            ("outputnode.out_aif_mask", "aif_mask"),
            ("outputnode.qc_metrics", "aif_qc"),
        ]),
        # Leakage correction (use recomputed K2 from compute_maps when baseline differs)
        (deconvolution_wf, outputnode, [("outputnode.out_k2", "k2_map")]),
        # Normalized CBV and CBF
        (normalization_wf, outputnode, [("outputnode.ncbv_file", "ncbv_map")]),
        (normalization_wf, outputnode, [("outputnode.ncbf_file", "ncbf_map")]),
        # Registration (.mat for internal consumers)
        (registration_wf, outputnode, [("outputnode.dsc_to_t1w_matrix", "dsc_to_t1w_xfm")]),
        (registration_wf, outputnode, [("outputnode.t1w_to_dsc_matrix", "t1w_to_dsc_xfm")]),
        # Registration (.h5 for derivatives)
        (registration_wf, outputnode, [("outputnode.t1w_to_dsc_h5", "t1w_to_dsc_h5")]),
        (qc_wf, outputnode, [("outputnode.registration_filmstrip", "registration_filmstrip")]),
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
        (registration_wf, convert_inv_h5, [("outputnode.dsc_to_t1w_matrix", "mat_path")]),
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
        (qc_wf, outputnode, [
            ("outputnode.aif_curve_plot", "aif_curve_plot"),
            ("outputnode.aif_location_plot", "aif_location_plot"),
            ("outputnode.aif_candidates_plot", "aif_candidates_plot"),
        ]),
        # Motion QC plot
        (motion_wf, outputnode, [("outputnode.motion_qc_plot_file", "motion_qc_plot")]),
        # DSC QC plots
        (qc_wf, outputnode, [
            ("outputnode.dsc_preprocessing_qc_plot", "dsc_preprocessing_qc_plot"),
            ("outputnode.preprocessing_stats", "preprocessing_stats"),
        ]),
        (qc_wf, outputnode, [("outputnode.dsc_param_maps_qc_plot", "dsc_param_maps_qc_plot")]),
        (qc_wf, outputnode, [("outputnode.dsc_leakage_qc_plot", "dsc_leakage_qc_plot")]),
        # Brain mask filmstrip
        (qc_wf, outputnode, [("outputnode.brain_mask_filmstrip", "brain_mask_filmstrip")]),
    ])

    workflow.connect([
        (qc_wf, outputnode, [("outputnode.dsc_fit_qc_plot", "dsc_fit_qc_plot")]),
    ])

    workflow.connect([
        # QC metrics
        (qc_wf, outputnode, [("outputnode.qc_metrics", "qc_metrics")]),
        (preload_wf, outputnode, [("outputnode.qc_dict", "preload_qc")]),
        (motion_wf, outputnode, [("outputnode.motion_qc_metrics", "motion_qc")]),
        (qc_wf, outputnode, [("outputnode.aif_metrics", "aif_metrics")]),
    ])

    # Smoothed outputs (only if smooth_fwhm > 0)
    if smoothing_wf is not None:
        workflow.connect([
            (smoothing_wf, outputnode, [
                ("outputnode.smoothed_hemodynamic", "smooth_hemodynamic"),
                ("outputnode.smoothed_auc", "smooth_auc"),
            ]),
        ])

    # Store config in workflow for reference by child nodes
    workflow.config_dsc = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
