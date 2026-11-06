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
from pmriprep.utils.dsc_constants import (
    DEFAULT_OI_THRESHOLD,
    DEFAULT_SVD_THRESHOLD,
)
from pmriprep.workflows._perfusion import (
    fsl_to_x5_node as _fsl_to_x5_node,
    select_aif_exclusion_mask as _select_aif_exclusion_mask,
    validate_native_mask as _validate_native_mask,
)

# FSL CommandLine wrappers and Function-node helpers used by the DSC graph.
from pmriprep.workflows.dsc._helpers import (  # noqa: F401
    ConvertXFM,
    FLIRTApplyXFM,
    _apply_dsc_leakage_policy,
    _build_hemodynamic_dict,
    _compute_mean_4d,
    _compute_prebolus_mean,
    _create_perfusion_brain_mask_filmstrip,
    _create_registration_filmstrip,
    _generate_dsc_fit_qc,
    _generate_dsc_leakage_qc,
    _generate_dsc_param_maps_qc,
    _generate_dsc_preprocessing_qc,
    _generate_motion_qc_plot,
    _histogram_match_t1w_to_dsc,
    _merge_qc_metrics,
    _normalize_cbf,
    _normalize_cbv,
    _signal_to_concentration,
    _smooth_maps,
    _transform_brain_mask_to_dsc,
    _transform_lesion_mask_to_dsc,
    _transform_nawm_to_dsc,
    _transform_structural_to_dsc,
)


def init_dsc_concentration_aif_wf(
    *,
    baseline_frames: int,
    n_candidates: int,
    lesion_distance: float,
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
    n_candidates : int
        Number of AIF candidate voxels to average.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
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
    out_aif_curve, out_aif_mask, out_aif_coord,
    detection_level, aif_baseline_frames, qc_metrics, out_gamma_aif_curve, out_rejected_aif_curve, out_rejected_aif_mask,
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
                "out_aif_mask",
                "out_aif_coord",
                "detection_level",
                "aif_baseline_frames",
                "qc_metrics",
                "out_gamma_aif_curve",
                "out_rejected_aif_curve",
                "out_rejected_aif_mask",
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
            criteria="strict",
            n_candidates=n_candidates,
            lesion_distance=lesion_distance,
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
            ("out_aif_mask", "out_aif_mask"),
            ("out_aif_coord", "out_aif_coord"),
            ("detection_level", "detection_level"),
            ("aif_baseline_frames", "aif_baseline_frames"),
            ("qc_metrics", "qc_metrics"),
            ("out_gamma_aif_curve", "out_gamma_aif_curve"),
            ("out_rejected_aif_curve", "out_rejected_aif_curve"),
            ("out_rejected_aif_mask", "out_rejected_aif_mask"),
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
    apply_moco: bool = True,
    name: str = "dsc_motion_wf",
) -> pe.Workflow:
    """Build the DSC motion-correction sub-workflow.

    Reorients the raw DSC series to RAS, runs motion correction, and computes a
    pre-bolus mean reference (used downstream for brain extraction, registration,
    and mask/atlas resampling), plus a motion-QC plot.

    When ``apply_moco`` is False, ANTs estimates input motion for QC,
    while the reoriented original series is passed downstream without resampling.

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
    apply_moco : bool
        Apply motion correction (default True). When False, the reoriented
        series is used downstream and ANTs runs for measurement only.
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
            motion_thresh=motion_thresh,
            baseline_frames=baseline_frames,
            apply_motion_correction=apply_moco,
        ),
        name="motion_correct",
        mem_gb=mem_gb,
    )

    # Node 1a: Motion QC Plot
    motion_qc_plot = pe.Node(
        niu.Function(
            input_names=[
                "motion_params_file", "tr", "subject_id", "session_id",
                "motion_applied", "motion_thresh",
            ],
            output_names=["plot_file", "motion_summary"],
            function=_generate_motion_qc_plot,
        ),
        name="motion_qc_plot",
    )
    motion_qc_plot.inputs.subject_id = subject_id
    motion_qc_plot.inputs.session_id = session_id or ""
    motion_qc_plot.inputs.motion_applied = apply_moco
    motion_qc_plot.inputs.motion_thresh = motion_thresh

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

    # Connections common to both branches: motion_correct always runs so
    # motion params / FD / QC report estimated input motion regardless of apply_moco.
    workflow.connect([
        (inputnode, dsc_reorient, [("dsc_file", "in_file")]),
        (dsc_reorient, motion_correct, [("out_file", "in_file")]),
        (motion_correct, motion_qc_plot, [("out_params", "motion_params_file")]),
        (inputnode, motion_qc_plot, [("tr", "tr")]),
        (motion_correct, outputnode, [
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

    if apply_moco:
        # Process the realigned series and its mean.
        workflow.connect([
            (motion_correct, prebolus_mean, [("out_file", "moco_file")]),
            (motion_correct, outputnode, [
                ("out_file", "dsc_moco"),
                ("mean_img", "dsc_mean"),
            ]),
        ])
    else:
        # Process the reoriented (uncorrected) series; ANTs output is
        # measurement-only. dsc_mean is recomputed from the passthrough series
        # so the registration/datasink reference matches the quantified data.
        reorient_mean = pe.Node(
            niu.Function(
                input_names=["in_file"],
                output_names=["mean_file"],
                function=_compute_mean_4d,
            ),
            name="reorient_mean",
        )
        workflow.connect([
            (dsc_reorient, prebolus_mean, [("out_file", "moco_file")]),
            (dsc_reorient, outputnode, [("out_file", "dsc_moco")]),
            (dsc_reorient, reorient_mean, [("out_file", "in_file")]),
            (reorient_mean, outputnode, [("mean_file", "dsc_mean")]),
        ])

    return workflow


def init_dsc_leakage_wf(
    *,
    preload: Union[str, float],
    agent_standard_dose: float,
    baseline_frames: int,
    leakage_method: str,
    mem_gb: float,
    return_guidance: bool = False,
    name: str = "dsc_leakage_wf",
) -> pe.Workflow:
    """Build the DSC leakage-correction sub-workflow.

    Applies the configured leakage policy, records administered-preload
    provenance, and computes the K2 map when BSW is selected.

    Parameters
    ----------
    preload : "unknown" or float
        Administered preload dose multiplier, or ``"unknown"``.
    agent_standard_dose : float
        Agent's standard dose (mmol Gd/kg).
    baseline_frames : int
        Pre-bolus baseline frames.
    leakage_method : str
        Leakage-correction method ("bsw" or "none").
    mem_gb : float
        Memory estimate (GB) for the heavy correction node.
    return_guidance : bool
        Preserve the exact BSW reference and uncertainty for signal CPI.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The leakage-correction sub-workflow.

    Inputs
    ------
    concentration_file, signal_file, brain_mask_file, te,
    field_strength, tr
        ΔR2* concentration, raw signal, DSC brain mask, and acquisition
        parameters.

    Outputs
    -------
    corrected_concentration_file
        Leakage-corrected concentration.
    k2_map_file
        K2 leakage map.
    guidance_file
        Exact BSW guidance archive, or empty when not requested.
    qc_dict
        Leakage QC and preload provenance.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "concentration_file",
                "signal_file",
                "brain_mask_file",
                "te",
                "field_strength",
                "tr",
                "baseline_frames",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "corrected_concentration_file",
                "k2_map_file",
                "guidance_file",
                "qc_dict",
            ]
        ),
        name="outputnode",
    )

    leakage_correct = pe.Node(
        niu.Function(
            input_names=[
                "concentration_file",
                "brain_mask_file",
                "te",
                "field_strength",
                "tr",
                "preload",
                "signal_file",
                "baseline_frames",
                "leakage_method",
                "agent_standard_dose",
                "return_guidance",
            ],
            output_names=["corrected_concentration_file", "k2_map_file", "qc_dict", "guidance_file"],
            function=_apply_dsc_leakage_policy,
        ),
        name="leakage_correct",
        mem_gb=mem_gb,
    )
    leakage_correct.inputs.preload = preload
    leakage_correct.inputs.agent_standard_dose = agent_standard_dose
    leakage_correct.inputs.baseline_frames = baseline_frames
    leakage_correct.inputs.leakage_method = leakage_method
    leakage_correct.inputs.return_guidance = return_guidance

    workflow.connect([
        (inputnode, leakage_correct, [
            ("concentration_file", "concentration_file"),
            ("signal_file", "signal_file"),
            ("brain_mask_file", "brain_mask_file"),
            ("te", "te"),
            ("field_strength", "field_strength"),
            ("tr", "tr"),
            ("baseline_frames", "baseline_frames"),
        ]),
        (leakage_correct, outputnode, [
            ("corrected_concentration_file", "corrected_concentration_file"),
            ("k2_map_file", "k2_map_file"),
            ("guidance_file", "guidance_file"),
        ]),
        (leakage_correct, outputnode, [("qc_dict", "qc_dict")]),
    ])
    return workflow


def init_dsc_registration_wf(
    *,
    mem_gb: float,
    name: str = "dsc_registration_wf",
) -> pe.Workflow:
    """Build the DSC<->T1w registration + MNI->DSC transform-chain sub-workflow.

    Registers the DSC reference to T1w, and builds the MNI->DSC transform by
    concatenating MNI->T1w (inverted T1w->MNI) with T1w->DSC.

    Parameters
    ----------
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
    brain_mask
        T1w-space brain mask used for registration QC.
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
                "native_brain_mask",
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
            ("native_brain_mask", "native_brain_mask"),
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
    concentration-domain deconvolution for SVD methods and original-signal
    fitting for CPI. Outputs include CBF, CBV, MTT, method-specific timing,
    and leakage diagnostics, using the supplied AIF and BSW guidance.

    Parameters
    ----------
    dsc_map_kwargs : dict
        Keyword arguments forwarded verbatim to ``ComputeDSCMaps`` (method,
        thresholds, inference settings, ``random_seed`` when set, etc.). Built by
        the orchestrator from the run configuration. CPI additionally receives
        the node's memory allocation for its worker cap.
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
    aif_file, aif_detection_level, baseline_frames
        Arterial input function and detection provenance.
    k2_map_input_file, bsw_guidance_file
        Leakage-correction coefficient map and exact CPI guidance archive.
    brain_mask_file, enhancing_mask_file
        DSC-space brain and enhancing-tumor masks.
    tr, te
        Acquisition timing and echo time.

    Outputs
    -------
    out_rel_cbf, out_rel_cbv, out_rel_cbv_auc
        Relative perfusion maps.
    out_mtt, out_tmax, out_ttp
        Transit-time and timing maps.
    out_psr
        Percentage signal recovery.
    out_auc_raw, out_auc_corr, out_auc_diff, out_k2
        Leakage (blood-brain-barrier) maps.
    out_cnr, out_tsnr
        Quality maps.
    out_cbf_std
        Conditional CBF uncertainty map (parametric methods only).
    out_parametric_r2, out_fallback_mask
        Parametric-fit diagnostics (CPI only).
    out_qc_fit_data, qc_metrics
        Deconvolution QC-fit data and the QC ``qc_metrics`` dict.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "concentration_file",
                "k2_map_input_file",
                "bsw_guidance_file",
                "aif_file",
                "aif_detection_level",
                "baseline_frames",
                "brain_mask_file",
                "enhancing_mask_file",
                "tr",
                "te",
                "signal_file",
                "raw_concentration_file",
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
                "out_tissue_auc",
                "out_mtt",
                "out_tmax",
                "out_psr",
                "out_ttp",
                "out_cnr",
                "out_auc_raw",
                "out_auc_corr",
                "out_auc_diff",
                "out_k2",
                "out_tsnr",
                "out_cbf_std",
                "out_parametric_r2",
                "out_fallback_mask",
                "out_qc_fit_data",
                "qc_metrics",
            ]
        ),
        name="outputnode",
    )

    if dsc_map_kwargs.get("method") == "cpi":
        dsc_map_kwargs = dsc_map_kwargs | {"fitting_memory_gb": mem_gb}
    compute_maps = pe.Node(
        ComputeDSCMaps(**dsc_map_kwargs),
        name="compute_maps",
        mem_gb=mem_gb,
        n_procs=max(1, int(dsc_map_kwargs.get("n_procs", 1))),
    )

    workflow.connect([
        (inputnode, compute_maps, [
            ("concentration_file", "concentration_file"),
            ("k2_map_input_file", "k2_map_input_file"),
            ("bsw_guidance_file", "bsw_guidance_file"),
            ("aif_file", "aif_file"),
            ("aif_detection_level", "aif_detection_level"),
            ("baseline_frames", "baseline_frames"),
            ("brain_mask_file", "brain_mask_file"),
            ("enhancing_mask_file", "enhancing_mask_file"),
            ("tr", "tr"),
            ("te", "te"),
            ("signal_file", "signal_file"),
            ("raw_concentration_file", "raw_concentration_file"),
        ]),
        (compute_maps, outputnode, [
            ("out_rel_cbf", "out_rel_cbf"),
            ("out_rel_cbv", "out_rel_cbv"),
            ("out_rel_cbv_auc", "out_rel_cbv_auc"),
            ("out_tissue_auc", "out_tissue_auc"),
            ("out_mtt", "out_mtt"),
            ("out_tmax", "out_tmax"),
            ("out_psr", "out_psr"),
            ("out_ttp", "out_ttp"),
            ("out_cnr", "out_cnr"),
            ("out_auc_raw", "out_auc_raw"),
            ("out_auc_corr", "out_auc_corr"),
            ("out_auc_diff", "out_auc_diff"),
            ("out_k2", "out_k2"),
            ("out_tsnr", "out_tsnr"),
            ("out_cbf_std", "out_cbf_std"),
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

    Normalizes relative CBV and CBF to NAWM, producing nCBV / nCBF maps and the
    shared normalization factor. Without a valid NAWM mask, normalized maps are
    omitted. CBV and CBF use the same normalization method.

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
    nawm_mask_file
        NAWM normalization mask.
    brain_mask_file
        DSC analysis support, intersected with the NAWM reference.

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
            fields=["cbv_file", "cbf_file", "nawm_mask_file", "brain_mask_file"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "ncbv_file",
                "ncbf_file",
                "normalization_factor",
                "cbf_normalization_factor",
            ]
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
                "derivatives_dir",
                "subject_id",
                "session_id",
                "nawm_mask_file",
                "brain_mask_file",
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
                "derivatives_dir",
                "subject_id",
                "session_id",
                "nawm_mask_file",
                "brain_mask_file",
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
            ("nawm_mask_file", "nawm_mask_file"),
            ("brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, normalize_cbf, [
            ("cbf_file", "cbf_file"),
            ("nawm_mask_file", "nawm_mask_file"),
            ("brain_mask_file", "brain_mask_file"),
        ]),
        (normalize, outputnode, [
            ("ncbv_file", "ncbv_file"),
            ("normalization_factor", "normalization_factor"),
        ]),
        (normalize_cbf, outputnode, [
            ("ncbf_file", "ncbf_file"),
            ("normalization_factor", "cbf_normalization_factor"),
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
    nCBF, nCBV) within the brain mask. Timing and recovery maps (Tmax, TTP, PSR,
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
    brain_mask_file
        DSC-space brain mask (smoothing is restricted to brain voxels).

    Outputs
    -------
    smoothed_hemodynamic
        Smoothed hemodynamic map files.
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
                "brain_mask_file",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["smoothed_hemodynamic"]
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
        (smooth_hemodynamic, outputnode, [("smoothed_files", "smoothed_hemodynamic")]),
    ])
    return workflow


def init_dsc_lesion_stats_wf(
    *,
    subject_id: str,
    session_id: Optional[str],
    deconvolution_method: str = "oSVD",
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
    deconvolution_method : str
        Method used to label the shared timing-map input as either Tmax or
        AIF-relative arrival delay in regional statistics.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The lesion-statistics sub-workflow.

    Inputs
    ------
    cbf_map, cbv_map, mtt_map, tmax_map, ttp_map
        DSC perfusion and transit-time maps (shared by all three nodes).
        ``cbv_map`` is the deconvolution residue-integral CBV.
    cbv_auc_map
        AUC-based CBV, using the configured leakage correction.
    k2_map
        Signed leakage coefficient.
    psr_map
        Percentage signal recovery.
    ncbv_map, ncbf_map
        Normalized CBV / CBF maps.
    brain_mask
        Perfusion analysis support; outside voxels are excluded from quantitative summaries.
    nawm_mask
        NAWM reference mask.
    fallback_mask
        Binary mask of successful CPI-to-oSVD replacements.
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
                "cbv_auc_map",
                "mtt_map",
                "tmax_map",
                "psr_map",
                "ttp_map",
                "k2_map",
                "ncbv_map",
                "ncbf_map",
                "brain_mask",
                "nawm_mask",
                "fallback_mask",
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
    lesion_stats.inputs.session_id = session_id or ""

    roi_stats = pe.Node(DSCLesionStats(force_binary=True), name="roi_stats")
    roi_stats.inputs.subject_id = subject_id
    roi_stats.inputs.session_id = session_id or ""

    user_lesion_stats = pe.Node(
        DSCLesionStats(force_binary=True), name="user_lesion_stats"
    )
    timing_map_name = (
        "arrival_delay"
        if deconvolution_method == "cpi"
        else "tmax"
    )
    for stats_node in (lesion_stats, roi_stats, user_lesion_stats):
        stats_node.inputs.timing_map_name = timing_map_name
    user_lesion_stats.inputs.subject_id = subject_id
    user_lesion_stats.inputs.session_id = session_id or ""

    # Perfusion maps + NAWM + AIF QC are shared by all three
    # stats nodes; the lesion mask differs (tumor-seg / ROI / user), and only
    # the tumor-seg node receives the mask-origin provenance label.
    shared_inputs = [
        ("cbf_map", "cbf_map"),
        ("cbv_map", "cbv_map"),
        ("cbv_auc_map", "cbv_auc_map"),
        ("mtt_map", "mtt_map"),
        ("tmax_map", "tmax_map"),
        ("psr_map", "psr_map"),
        ("ttp_map", "ttp_map"),
        ("k2_map", "k2_map"),
        ("ncbv_map", "ncbv_map"),
        ("ncbf_map", "ncbf_map"),
        ("brain_mask", "brain_mask"),
        ("nawm_mask", "nawm_mask"),
        ("fallback_mask", "fallback_mask"),
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
    baseline_frames: int,
    deconvolution_method: str = "oSVD",
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
        AIF provenance label for the merged QC ("external", "auto").
    baseline_frames : int
        Initial S0 estimation frame count marked in the preprocessing QC plot.
    deconvolution_method : str
        Method used to label the timing map in the parameter-map report.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The QC sub-workflow.

    Inputs
    ------
    motion_qc_metrics, motion_summary, preload_qc_dict, aif_qc_metrics, deconv_qc_metrics, registration_qc_metrics, normalization_factor, cbf_normalization_factor
        Per-sub-workflow QC metric dicts / scalars merged into ``qc_metrics``.
    prebolus_mean_file, dsc_moco, registered_t1w, brain_mask_file
        Images for the registration / preprocessing / brain-mask filmstrips.
    analysis_baseline_frames
        Finalized baseline frame count used for preprocessing QC metrics and
        marked on the global signal time course.
    out_aif_curve, out_aif_mask, out_gamma_aif_curve, out_rejected_aif_curve, out_rejected_aif_mask, concentration_file, corrected_concentration_file
        AIF, concentration, and leakage-QC inputs.
    rel_cbf, rel_cbv_auc, mtt, tmax, psr, ttp, k2, auc_raw, auc_corr, auc_diff, qc_fit_data
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
                "analysis_baseline_frames",
                "out_aif_curve",
                "out_aif_mask",
                "out_gamma_aif_curve",
                "out_rejected_aif_curve",
                "out_rejected_aif_mask",
                "concentration_file",
                "deconv_qc_metrics",
                "rel_cbf",
                "rel_cbv_auc",
                "ncbv",
                "mtt",
                "tmax",
                "psr",
                "ttp",
                "k2",
                "auc_raw",
                "auc_corr",
                "auc_diff",
                "qc_fit_data",
                "registration_qc_metrics",
                "brain_extraction_qc_metrics",
                "registered_t1w",
                "registered_t1w_mask",
                "t1w_to_dsc_matrix",
                "normalization_factor",
                "cbf_normalization_factor",
                "brain_mask_file",
                "nawm_mask_file",
                "nawm_exclusion_reason",
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
                "brain_extraction_qc",
                "normalization_factor",
                "cbf_normalization_factor",
                "motion_summary",
                "aif_metrics",
                "preprocessing_stats",
                "aif_source",
                "auc_raw_file",
                "auc_corr_file",
                "nawm_mask_file",
                "nawm_exclusion_reason",
            ],
            output_names=["merged_qc"],
            function=_merge_qc_metrics,
        ),
        name="merge_qc",
    )
    merge_qc.inputs.aif_source = aif_source

    reg_filmstrip = pe.Node(
        niu.Function(
            input_names=[
                "dsc_mean_file", "registered_t1w_file", "brain_mask_file",
                "registered_t1w_mask_file", "subject_id", "session_id",
            ],
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
                "analysis_baseline_frames",
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
                "rel_cbf_file", "rel_cbv_file", "ncbv_file", "mtt_file",
                "tmax_file", "psr_file", "ttp_file",
                "brain_mask_file", "timing_label", "subject_id", "session_id",
            ],
            output_names=["plot_file"],
            function=_generate_dsc_param_maps_qc,
        ),
        name="dsc_param_maps_qc",
    )
    dsc_param_maps_qc.inputs.subject_id = subject_id
    dsc_param_maps_qc.inputs.session_id = session_id or ""
    dsc_param_maps_qc.inputs.timing_label = (
        "Arrival delay"
        if deconvolution_method == "cpi"
        else "Tmax"
    )

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
            ("brain_extraction_qc_metrics", "brain_extraction_qc"),
            ("normalization_factor", "normalization_factor"),
            ("cbf_normalization_factor", "cbf_normalization_factor"),
            ("auc_raw", "auc_raw_file"),
            ("auc_corr", "auc_corr_file"),
            ("nawm_mask_file", "nawm_mask_file"),
            ("nawm_exclusion_reason", "nawm_exclusion_reason"),
        ]),
        (inputnode, reg_filmstrip, [
            ("prebolus_mean_file", "dsc_mean_file"),
            ("registered_t1w", "registered_t1w_file"),
            ("registered_t1w_mask", "registered_t1w_mask_file"),
            ("brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, aif_qc_plot, [
            ("dsc_moco", "dsc_file"),
            ("aif_qc_metrics", "qc_metrics"),
            ("out_aif_curve", "aif_curve_file"),
            ("out_aif_mask", "aif_mask_file"),
            ("out_gamma_aif_curve", "gamma_aif_curve_file"),
            ("out_rejected_aif_curve", "rejected_aif_curve_file"),
            ("out_rejected_aif_mask", "rejected_aif_mask_file"),
            ("t1w_to_dsc_matrix", "t1w_to_dsc_xfm"),
            ("brain_mask_file", "brain_mask_file"),
            ("tr", "tr"),
            ("t1w_to_mni_xfm", "t1w_to_mni_xfm"),
        ]),
        (inputnode, dsc_preproc_qc, [
            ("dsc_moco", "dsc_signal_file"),
            ("brain_mask_file", "brain_mask_file"),
            ("analysis_baseline_frames", "analysis_baseline_frames"),
            ("tr", "tr"),
            ("te", "te"),
        ]),
        (inputnode, dsc_param_maps_qc, [
            ("rel_cbf", "rel_cbf_file"),
            ("rel_cbv_auc", "rel_cbv_file"),
            ("ncbv", "ncbv_file"),
            ("mtt", "mtt_file"),
            ("tmax", "tmax_file"),
            ("psr", "psr_file"),
            ("ttp", "ttp_file"),
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


def init_dsc_lesion_masks_wf(
    *, name: str = "dsc_lesion_masks_wf",
) -> pe.Workflow:
    """Prepare native and T1w-derived lesion masks for DSC consumers."""
    from pmriprep.utils.lesion_stats import combine_tumor_masks

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "roi_mask", "user_lesion_mask", "tumor_segmentation",
            "whole_tumor_mask", "enhancing_mask", "reference_file",
            "transform_matrix",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "roi_mask", "user_lesion_mask", "tumor_lesion_mask",
            "enhancing_mask", "aif_exclusion_mask", "mask_origin",
        ]),
        name="outputnode",
    )
    validate_roi_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "mask_label"],
            output_names=["validated_mask"], function=_validate_native_mask,
        ),
        name="validate_roi_mask",
    )
    validate_roi_mask.inputs.mask_label = "dsc.roi_mask"
    validate_user_lesion_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "mask_label"],
            output_names=["validated_mask"], function=_validate_native_mask,
        ),
        name="validate_user_lesion_mask",
    )
    validate_user_lesion_mask.inputs.mask_label = "dsc.lesion_mask"
    combine_masks = pe.Node(
        niu.Function(
            input_names=[
                "tumor_segmentation_file", "whole_tumor_mask_file", "reference_file",
            ],
            output_names=["combined_mask_file", "mask_origin"],
            function=combine_tumor_masks,
        ),
        name="combine_masks",
    )
    transform_lesion = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dsc_file"], function=_transform_lesion_mask_to_dsc,
        ),
        name="transform_lesion",
    )
    transform_enhancing_mask = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dsc_file"], function=_transform_lesion_mask_to_dsc,
        ),
        name="transform_enhancing_mask",
    )
    select_aif_exclusion = pe.Node(
        niu.Function(
            input_names=["user_lesion_mask", "transformed_lesion_mask"],
            output_names=["out_mask"], function=_select_aif_exclusion_mask,
        ),
        name="select_aif_exclusion",
    )
    workflow.connect([
        (inputnode, validate_roi_mask, [
            ("roi_mask", "mask_file"), ("reference_file", "reference_file"),
        ]),
        (inputnode, validate_user_lesion_mask, [
            ("user_lesion_mask", "mask_file"), ("reference_file", "reference_file"),
        ]),
        (inputnode, combine_masks, [
            ("tumor_segmentation", "tumor_segmentation_file"),
            ("whole_tumor_mask", "whole_tumor_mask_file"),
            ("reference_file", "reference_file"),
        ]),
        (combine_masks, transform_lesion, [("combined_mask_file", "lesion_mask_file")]),
        (inputnode, transform_lesion, [
            ("reference_file", "reference_file"), ("transform_matrix", "transform_matrix"),
        ]),
        (inputnode, transform_enhancing_mask, [
            ("enhancing_mask", "lesion_mask_file"),
            ("reference_file", "reference_file"), ("transform_matrix", "transform_matrix"),
        ]),
        (validate_user_lesion_mask, select_aif_exclusion, [
            ("validated_mask", "user_lesion_mask"),
        ]),
        (transform_lesion, select_aif_exclusion, [
            ("lesion_mask_dsc_file", "transformed_lesion_mask"),
        ]),
        (validate_roi_mask, outputnode, [("validated_mask", "roi_mask")]),
        (validate_user_lesion_mask, outputnode, [("validated_mask", "user_lesion_mask")]),
        (transform_lesion, outputnode, [("lesion_mask_dsc_file", "tumor_lesion_mask")]),
        (transform_enhancing_mask, outputnode, [("lesion_mask_dsc_file", "enhancing_mask")]),
        (select_aif_exclusion, outputnode, [("out_mask", "aif_exclusion_mask")]),
        (combine_masks, outputnode, [("mask_origin", "mask_origin")]),
    ])
    return workflow


def init_dsc_underlays_wf(
    *, underlays: list[str], name: str = "dsc_underlays_wf",
) -> pe.Workflow:
    """Transform the requested structural underlays into DSC space."""
    workflow = pe.Workflow(name=name)
    input_fields = [
        "t1w_underlay", "t1w_postcon_underlay", "t1w_subtraction_underlay",
        "flair_in_t1w", "flair_postcon_in_t1w", "t2w_in_t1w",
        "reference_file", "transform_matrix",
    ]
    output_fields = [
        "t1w_in_dsc", "t1w_postcon_in_dsc", "t1w_subtraction_in_dsc",
        "flair_in_dsc", "flair_postcon_in_dsc", "t2w_in_dsc",
    ]
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields), name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=output_fields), name="outputnode")
    specifications = {
        "t1w_precon": ("transform_t1_to_dsc", "t1w_underlay", "t1w_in_dsc.nii.gz", "t1w_in_dsc"),
        "t1w_postcon": (
            "transform_t1_postcon_to_dsc", "t1w_postcon_underlay",
            "t1w_postcon_in_dsc.nii.gz", "t1w_postcon_in_dsc",
        ),
        "t1w_subtraction": (
            "transform_t1_subtraction_to_dsc", "t1w_subtraction_underlay",
            "t1w_subtraction_in_dsc.nii.gz", "t1w_subtraction_in_dsc",
        ),
        "flair_precon": (
            "transform_flair_to_dsc", "flair_in_t1w",
            "flair_in_dsc.nii.gz", "flair_in_dsc",
        ),
        "flair_postcon": (
            "transform_flair_postcon_to_dsc", "flair_postcon_in_t1w",
            "flair_postcon_in_dsc.nii.gz", "flair_postcon_in_dsc",
        ),
        "t2w": ("transform_t2w_to_dsc", "t2w_in_t1w", "t2w_in_dsc.nii.gz", "t2w_in_dsc"),
    }
    for key, (node_name, input_field, output_name, output_field) in specifications.items():
        if key not in underlays:
            continue
        transform = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"], function=_transform_structural_to_dsc,
            ),
            name=node_name,
        )
        transform.inputs.output_name = output_name
        workflow.connect([
            (inputnode, transform, [
                (input_field, "structural_file"), ("reference_file", "reference_file"),
                ("transform_matrix", "transform_matrix"),
            ]),
            (transform, outputnode, [("out_file", output_field)]),
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
    random_seed : int, optional
        Seed forwarded to the perfusion-map interface.
    mem_gb : float, optional
        Memory estimate in GB for heavy processing nodes. None uses the
        workflow's fallback estimate.
    mem_limit_gb : float, optional
        Upper bound in GB for per-node memory requests.
    gd_multiplicity : int, optional
        Number of gadolinium ions per contrast-agent molecule as metadata.
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
        Echo time in seconds. The master workflow supplies the BIDS value;
        direct construction initializes this input to 0.035.
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
    mtt_map, tmax_map
        Mean transit time and time-to-maximum maps.
    psr_map
        Percent signal recovery map.
    ttp_map, cnr_map
        AIF-arrival-referenced time-to-peak and contrast-to-noise ratio maps.
    tsnr_map
        Baseline temporal-SNR quality map.
    aif_curve
        Path to detected/used AIF curve.
    aif_mask
        Path to AIF detection mask.
    ncbv_map, ncbf_map
        Normalized CBV/CBF maps (if NAWM mask available).
    smooth_hemodynamic
        Smoothed hemodynamic-map dictionary (if smooth_fwhm > 0).
    dsc_to_t1w_xfm
        Path to DSC to T1w transform.
    qc_metrics
        Dictionary of QC metrics from processing steps.

    Notes
    -----
    The workflow follows this processing flow:
    1. Motion correction (ANTs)
    2. Signal-to-concentration conversion (ΔR2*)
    3. AIF detection (atlas-based → automatic → partial outputs)
    4. Leakage correction and preload-provenance recording
    5. Deconvolution (method from config: sSVD, oSVD, tikhonov, cpi)
    6. Perfusion parameter extraction (CBF, CBV, MTT, Tmax, PSR)
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
                "t1w_subtraction_underlay",  # post-pre subtraction underlay (precon-anchored T1w space)
                "brain_mask",  # Brain mask in T1w space
                "wm_probseg",  # White matter probability map (optional)
                "whole_tumor_mask",  # Whole-tumor mask for AIF exclusion + lesion stats (optional, T1w space)
                "enhancing_mask",  # Enhancing tumor mask for laterality (optional, T1w space)
                "tumor_segmentation",  # Multi-label tumor segmentation for lesion stats (optional, T1w space)
                "user_lesion_mask",  # User-supplied lesion mask in DSC native space (optional)
                "nawm_mask",  # NAWM mask for normalization (optional)
                "nawm_exclusion_reason",  # Reason an auto-generated NAWM mask is QC-only
                "nawm_needs_transform",  # True if nawm_mask is in T1w space (tumor seg)
                "roi_mask",  # Generic ROI mask in DSC native space (optional)
                "t1w_to_mni_xfm",  # T1w to MNI transform (from ACPC alignment)
                "te",  # Echo time in seconds
                "field_strength",  # Magnetic field strength
                "tr",  # Repetition time in seconds
                "flair_in_t1w",  # pre-contrast FLAIR coregistered to T1w space (from anat)
                "flair_postcon_in_t1w",  # post-contrast FLAIR coregistered to T1w space (from anat)
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
    inputnode.inputs.nawm_exclusion_reason = ""

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
                "ttp_map",
                "cnr_map",
                # Leakage correction
                "k2_map",
                # Conditional parametric CBF uncertainty map (optional)
                "cbf_std_map",
                # Parametric-fit diagnostics (CPI only)
                "parametric_r2_map",
                "fallback_mask",
                # Relative maps (always produced)
                "rel_cbf_map",
                "rel_cbv_map",
                "rel_cbv_auc_map",
                # Quality metrics
                "tsnr_map",
                # AIF outputs
                "aif_curve",
                "aif_mask",
                "aif_qc",
                # Normalized maps
                "ncbv_map",
                "ncbf_map",
                # Registration
                "dsc_to_t1w_xfm",
                "t1w_to_dsc_xfm",
                "dsc_to_t1w_h5",
                "t1w_to_dsc_h5",
                "registration_filmstrip",
                "registration_qc",
                # Structural underlays
                "t1w_in_dsc",
                "t1w_postcon_in_dsc",
                "t1w_subtraction_in_dsc",
                "flair_in_dsc",
                "flair_postcon_in_dsc",
                "t2w_in_dsc",
                # AIF QC plots
                "aif_curve_plot",
                "aif_location_plot",
                "aif_candidates_plot",
                # Smoothed maps (populated only if smooth_fwhm > 0)
                "smooth_hemodynamic",
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
        apply_moco=config.motion_correct,
    )
    workflow.connect([
        (inputnode, motion_wf, [
            ("dsc_file", "inputnode.dsc_file"),
            ("tr", "inputnode.tr"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Node 1b: Native DSC brain extraction for processing and registration
    # The extraction node supplies the processing mask from the DSC image,
    # including when an extraction fallback is used. The transformed T1w
    # mask is used separately for registration QC.
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
    hdbet_dsc.inputs.sloppy = config.sloppy

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
    # Node 1c: Transform T1w Brain Mask to DSC Space for registration QC
    # -------------------------------------------------------------------
    transform_brain_mask_qc = pe.Node(
        niu.Function(
            input_names=[
                "brain_mask_file", "reference_file", "transform_matrix",
                "hdbet_mask_file",
            ],
            output_names=["dsc_brain_mask_file", "dice_score"],
            function=_transform_brain_mask_to_dsc,
        ),
        name="transform_brain_mask_qc",
    )

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
        n_candidates=config.n_aif_candidates,
        lesion_distance=config.lesion_distance,
        dose=config.dose,
        agent_standard_dose=config.agent_standard_dose,
        gd_multiplicity=gd_multiplicity,
        external_aif_file=_external_aif,
        aif_mask_file=_aif_mask,
        mem_gb=_heavy_mem,
    )

    # -------------------------------------------------------------------
    # Node 4: Leakage correction and preload provenance
    # K2 correction uses the finalized baseline from AIF detection
    # -------------------------------------------------------------------
    leakage_wf = init_dsc_leakage_wf(
        preload=config.preload,
        agent_standard_dose=config.agent_standard_dose,
        baseline_frames=config.pre_bolus,
        leakage_method=config.leakage_method,
        return_guidance=config.method == "cpi" and config.leakage_method == "bsw",
        mem_gb=_heavy_mem,
    )

    # -------------------------------------------------------------------
    # Node 5: DSC Map Computation (Deconvolution)
    # -------------------------------------------------------------------
    dsc_map_kwargs: dict[str, Any] = dict(
        method=config.method,
        svd_threshold=DEFAULT_SVD_THRESHOLD,
        oi_threshold=DEFAULT_OI_THRESHOLD,
        cpi_control_points=config.cpi_control_points,
        cpi_infer_delay=config.cpi_infer_delay,
        cpi_dispersion=config.cpi_dispersion,
        cpi_fallback=config.cpi_fallback,
        inference=config.inference,
        leakage_method=config.leakage_method,
        n_procs=config.num_processes if config.num_processes > 0 else 1,
        # baseline_frames wired dynamically from conc_aif_wf.outputnode.aif_baseline_frames
        tail_frames=10,
        sloppy=config.sloppy,
    )
    if random_seed is not None:
        dsc_map_kwargs["random_seed"] = random_seed
    if config.method == "cpi":
        from pmriprep.models.dsc.signal_cpi.execution import (
            CPI_CONTROLLER_MEMORY_GB,
            estimate_worker_memory_gb,
        )

        cpi_memory_gb = (
            _heavy_mem
            + CPI_CONTROLLER_MEMORY_GB
            + estimate_worker_memory_gb() * dsc_map_kwargs["n_procs"]
        )
        deconvolution_memory_gb = cap_mem_gb(cpi_memory_gb, mem_limit_gb)
    else:
        deconvolution_memory_gb = _heavy_mem
    deconvolution_wf = init_dsc_deconvolution_wf(
        dsc_map_kwargs=dsc_map_kwargs,
        mem_gb=deconvolution_memory_gb,
    )

    # -------------------------------------------------------------------
    # Registration sub-workflow: DSC<->T1w + MNI->DSC transform chain
    # -------------------------------------------------------------------
    registration_wf = init_dsc_registration_wf(
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
            "The arterial atlas is unavailable for the DSC AIF search."
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

    lesion_masks_wf = init_dsc_lesion_masks_wf()

    # -------------------------------------------------------------------
    # Node 6f: Lesion / ROI / user-mask statistics extraction
    # -------------------------------------------------------------------
    lesion_stats_wf = init_dsc_lesion_stats_wf(
        subject_id=subject_id,
        session_id=session_id,
        deconvolution_method=config.method,
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
    else:
        aif_source = "auto"

    qc_wf = init_dsc_qc_wf(
        subject_id=subject_id,
        session_id=session_id,
        aif_source=aif_source,
        baseline_frames=config.pre_bolus,
        deconvolution_method=config.method,
    )

    # -------------------------------------------------------------------
    # Node 9f: Transform structural underlays to DSC Space
    # register.out_matrix is T1w→DSC, so we use it directly.
    # Underlays are conditional based on config.underlays list.
    # Uses Function interface to handle undefined inputs gracefully.
    # -------------------------------------------------------------------
    underlays_wf = init_dsc_underlays_wf(underlays=config.underlays)

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

    workflow.connect([
        (inputnode, transform_brain_mask_qc, [
            ("brain_mask", "brain_mask_file"),
        ]),
        (motion_wf, transform_brain_mask_qc, [
            ("outputnode.prebolus_mean_file", "reference_file"),
        ]),
        (registration_wf, transform_brain_mask_qc, [
            ("outputnode.t1w_to_dsc_matrix", "transform_matrix"),
        ]),
        (hdbet_dsc, transform_brain_mask_qc, [
            ("dsc_brain_mask_file", "hdbet_mask_file"),
        ]),
    ])

    # Motion + acquisition params into the concentration/AIF sub-workflow
    workflow.connect([
        (motion_wf, conc_aif_wf, [("outputnode.dsc_moco", "inputnode.dsc_moco")]),
        (inputnode, conc_aif_wf, [
            ("te", "inputnode.te"),
        ]),
    ])

    # Prepare lesion masks and AIF exclusion: user mask overrides tumor seg.
    workflow.connect([
        (inputnode, lesion_masks_wf, [
            ("roi_mask", "inputnode.roi_mask"),
            ("user_lesion_mask", "inputnode.user_lesion_mask"),
            ("tumor_segmentation", "inputnode.tumor_segmentation"),
            ("whole_tumor_mask", "inputnode.whole_tumor_mask"),
            ("enhancing_mask", "inputnode.enhancing_mask"),
        ]),
        (motion_wf, lesion_masks_wf, [
            ("outputnode.prebolus_mean_file", "inputnode.reference_file"),
        ]),
        (registration_wf, lesion_masks_wf, [
            ("outputnode.t1w_to_dsc_matrix", "inputnode.transform_matrix"),
        ]),
    ])

    # Signal Conversion to AIF Detection
    # Use DSC-space brain mask (not T1w-space) for proper geometry match
    # Pass raw signal for S0-based partial volume rejection
    workflow.connect([
        (hdbet_dsc, conc_aif_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
        ]),
        (lesion_masks_wf, conc_aif_wf, [
            ("outputnode.aif_exclusion_mask", "inputnode.lesion_mask_file"),
        ]),
        (lesion_masks_wf, conc_aif_wf, [
            ("outputnode.enhancing_mask", "inputnode.enhancing_mask_file"),
        ]),
        (inputnode, conc_aif_wf, [
            ("tr", "inputnode.tr"),
            ("field_strength", "inputnode.field_strength"),
        ]),
    ])

    # Signal conversion to leakage correction
    # Use DSC-space brain mask for proper geometry match
    workflow.connect([
        (conc_aif_wf, leakage_wf, [
            ("outputnode.concentration_file", "inputnode.concentration_file"),
            ("outputnode.signal_file", "inputnode.signal_file"),
            ("outputnode.aif_baseline_frames", "inputnode.baseline_frames"),
        ]),
        (hdbet_dsc, leakage_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
        ]),
        (inputnode, leakage_wf, [
            ("te", "inputnode.te"),
            ("field_strength", "inputnode.field_strength"),
            ("tr", "inputnode.tr"),
        ]),
    ])

    # Leakage correction to DSC map computation
    # Use DSC-space brain mask for proper geometry match
    workflow.connect([
        (leakage_wf, deconvolution_wf, [
            ("outputnode.corrected_concentration_file", "inputnode.concentration_file"),
            ("outputnode.k2_map_file", "inputnode.k2_map_input_file"),
            ("outputnode.guidance_file", "inputnode.bsw_guidance_file"),
        ]),
        (conc_aif_wf, deconvolution_wf, [
            ("outputnode.out_aif_curve", "inputnode.aif_file"),
            ("outputnode.detection_level", "inputnode.aif_detection_level"),
            ("outputnode.aif_baseline_frames", "inputnode.baseline_frames"),
        ]),
        (hdbet_dsc, deconvolution_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
        ]),
        (lesion_masks_wf, deconvolution_wf, [
            ("outputnode.enhancing_mask", "inputnode.enhancing_mask_file"),
        ]),
        (inputnode, deconvolution_wf, [
            ("tr", "inputnode.tr"),
            ("te", "inputnode.te"),
        ]),
        (conc_aif_wf, deconvolution_wf, [
            ("outputnode.signal_file", "inputnode.signal_file"),
            ("outputnode.concentration_file", "inputnode.raw_concentration_file"),
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
            ("t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
        ]),
        (hdbet_dsc, registration_wf, [
            ("dsc_brain_file", "inputnode.dsc_brain"),
            ("dsc_brain_mask_file", "inputnode.native_brain_mask"),
        ]),
    ])

    # Transform an available arterial atlas using the pre-bolus reference.
    if isdefined(transform_atlas.inputs.in_file):
        workflow.connect([
            (registration_wf, transform_atlas, [("outputnode.mni_to_dsc_matrix", "in_matrix_file")]),
            (motion_wf, transform_atlas, [("outputnode.prebolus_mean_file", "reference")]),
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

    # Feed relative perfusion maps and the NAWM reference to normalization.
    workflow.connect([
        (hdbet_dsc, normalization_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
        ]),
        (deconvolution_wf, normalization_wf, [
            ("outputnode.out_tissue_auc", "inputnode.cbv_file"),
            ("outputnode.out_rel_cbf", "inputnode.cbf_file"),
        ]),
        (transform_nawm, normalization_wf, [
            ("nawm_mask_dsc_file", "inputnode.nawm_mask_file"),
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
        (leakage_wf, qc_wf, [
            ("outputnode.qc_dict", "inputnode.preload_qc_dict"),
            ("outputnode.corrected_concentration_file", "inputnode.corrected_concentration_file"),
        ]),
        (transform_nawm, qc_wf, [
            ("nawm_mask_dsc_file", "inputnode.nawm_mask_file"),
        ]),
        (conc_aif_wf, qc_wf, [
            ("outputnode.qc_metrics", "inputnode.aif_qc_metrics"),
            ("outputnode.aif_baseline_frames", "inputnode.analysis_baseline_frames"),
            ("outputnode.out_aif_curve", "inputnode.out_aif_curve"),
            ("outputnode.out_aif_mask", "inputnode.out_aif_mask"),
            ("outputnode.out_gamma_aif_curve", "inputnode.out_gamma_aif_curve"),
            ("outputnode.out_rejected_aif_curve", "inputnode.out_rejected_aif_curve"),
            ("outputnode.out_rejected_aif_mask", "inputnode.out_rejected_aif_mask"),
            ("outputnode.concentration_file", "inputnode.concentration_file"),
        ]),
        (deconvolution_wf, qc_wf, [
            ("outputnode.qc_metrics", "inputnode.deconv_qc_metrics"),
            ("outputnode.out_rel_cbf", "inputnode.rel_cbf"),
            ("outputnode.out_rel_cbv_auc", "inputnode.rel_cbv_auc"),
            ("outputnode.out_mtt", "inputnode.mtt"),
            ("outputnode.out_tmax", "inputnode.tmax"),
            ("outputnode.out_psr", "inputnode.psr"),
            ("outputnode.out_ttp", "inputnode.ttp"),
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
        (transform_brain_mask_qc, qc_wf, [
            ("dsc_brain_mask_file", "inputnode.registered_t1w_mask"),
        ]),
        (normalization_wf, qc_wf, [
            ("outputnode.ncbv_file", "inputnode.ncbv"),
            ("outputnode.normalization_factor", "inputnode.normalization_factor"),
            ("outputnode.cbf_normalization_factor", "inputnode.cbf_normalization_factor"),
        ]),
        (hdbet_dsc, qc_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask_file"),
            ("qc_metrics", "inputnode.brain_extraction_qc_metrics"),
        ]),
        (inputnode, qc_wf, [
            ("tr", "inputnode.tr"),
            ("te", "inputnode.te"),
            ("t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
            ("nawm_exclusion_reason", "inputnode.nawm_exclusion_reason"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Lesion / ROI / user-mask statistics: feed the stats sub-workflow.
    # The perfusion maps + NAWM + field strength + AIF QC are shared by all
    # three stats nodes (fanned out inside the sub-wf); the three lesion masks
    # (tumor-seg, user ROI, user lesion) route to their respective nodes.
    # -------------------------------------------------------------------
    workflow.connect([
        (hdbet_dsc, lesion_stats_wf, [
            ("dsc_brain_mask_file", "inputnode.brain_mask"),
        ]),
        (deconvolution_wf, lesion_stats_wf, [
            ("outputnode.out_rel_cbf", "inputnode.cbf_map"),
            ("outputnode.out_rel_cbv", "inputnode.cbv_map"),
            ("outputnode.out_rel_cbv_auc", "inputnode.cbv_auc_map"),
            ("outputnode.out_mtt", "inputnode.mtt_map"),
            ("outputnode.out_tmax", "inputnode.tmax_map"),
            ("outputnode.out_psr", "inputnode.psr_map"),
            ("outputnode.out_ttp", "inputnode.ttp_map"),
            ("outputnode.out_k2", "inputnode.k2_map"),
            ("outputnode.out_fallback_mask", "inputnode.fallback_mask"),
        ]),
        (normalization_wf, lesion_stats_wf, [
            ("outputnode.ncbv_file", "inputnode.ncbv_map"),
            ("outputnode.ncbf_file", "inputnode.ncbf_map"),
        ]),
        (conc_aif_wf, lesion_stats_wf, [("outputnode.qc_metrics", "inputnode.aif_qc")]),
        (transform_nawm, lesion_stats_wf, [("nawm_mask_dsc_file", "inputnode.nawm_mask")]),
        (lesion_masks_wf, lesion_stats_wf, [
            ("outputnode.tumor_lesion_mask", "inputnode.tumor_lesion_mask"),
            ("outputnode.mask_origin", "inputnode.mask_origin"),
            ("outputnode.roi_mask", "inputnode.roi_mask"),
            ("outputnode.user_lesion_mask", "inputnode.user_mask"),
        ]),
    ])

    # Stats sub-workflow outputs + the DSC-space masks they were computed on
    # (the masks come from transform/validate nodes that stay in the parent).
    workflow.connect([
        (lesion_masks_wf, outputnode, [
            ("outputnode.tumor_lesion_mask", "lesion_mask_dsc"),
            ("outputnode.roi_mask", "roi_mask_dsc"),
            ("outputnode.user_lesion_mask", "user_lesion_mask_dsc"),
        ]),
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
    underlay_inputs = {
        "t1w_precon": "t1w_underlay",
        "t1w_postcon": "t1w_postcon_underlay",
        "t1w_subtraction": "t1w_subtraction_underlay",
        "flair_precon": "flair_in_t1w",
        "flair_postcon": "flair_postcon_in_t1w",
        "t2w": "t2w_in_t1w",
    }
    requested_underlay_inputs = [
        (field, f"inputnode.{field}")
        for key, field in underlay_inputs.items()
        if key in config.underlays
    ]
    if requested_underlay_inputs:
        workflow.connect([
            (inputnode, underlays_wf, requested_underlay_inputs),
            (motion_wf, underlays_wf, [
                ("outputnode.prebolus_mean_file", "inputnode.reference_file"),
            ]),
            (registration_wf, underlays_wf, [
                ("outputnode.t1w_to_dsc_matrix", "inputnode.transform_matrix"),
            ]),
        ])

    # -------------------------------------------------------------------
    # Smoothing connections (only if smooth_fwhm > 0)
    # -------------------------------------------------------------------
    if smoothing_wf is not None:
        # nCBF/nCBV come from the normalization sub-wf; the hemodynamic +
        # brain mask fans out to the smoother inside the sub-workflow.
        workflow.connect([
            (deconvolution_wf, smoothing_wf, [
                ("outputnode.out_mtt", "inputnode.mtt"),
                ("outputnode.out_rel_cbf", "inputnode.rel_cbf"),
                ("outputnode.out_rel_cbv_auc", "inputnode.rel_cbv"),
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
            ("outputnode.out_ttp", "ttp_map"),
            ("outputnode.out_cnr", "cnr_map"),
        ]),
        # Relative maps
        (deconvolution_wf, outputnode, [
            ("outputnode.out_rel_cbf", "rel_cbf_map"),
            ("outputnode.out_rel_cbv", "rel_cbv_map"),
            ("outputnode.out_rel_cbv_auc", "rel_cbv_auc_map"),
        ]),
        # Quality metrics
        (deconvolution_wf, outputnode, [
            ("outputnode.out_tsnr", "tsnr_map"),
        ]),
        # Leakage-correction coefficient (defined only for BSW)
        (deconvolution_wf, outputnode, [
            ("outputnode.out_k2", "k2_map"),
        ]),
        # Conditional CBF uncertainty (only for CPI)
        (deconvolution_wf, outputnode, [
            ("outputnode.out_cbf_std", "cbf_std_map"),
        ]),
        # Parametric-fit diagnostics (CPI only;
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
        # Normalized CBV and CBF
        (normalization_wf, outputnode, [("outputnode.ncbv_file", "ncbv_map")]),
        (normalization_wf, outputnode, [("outputnode.ncbf_file", "ncbf_map")]),
        # Registration (.mat for internal consumers)
        (registration_wf, outputnode, [("outputnode.dsc_to_t1w_matrix", "dsc_to_t1w_xfm")]),
        (registration_wf, outputnode, [("outputnode.t1w_to_dsc_matrix", "t1w_to_dsc_xfm")]),
        (registration_wf, outputnode, [("outputnode.qc_metrics", "registration_qc")]),
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

    # Structural underlay connections (conditional inside the sub-workflow)
    underlay_outputs = {
        "t1w_precon": "t1w_in_dsc",
        "t1w_postcon": "t1w_postcon_in_dsc",
        "t1w_subtraction": "t1w_subtraction_in_dsc",
        "flair_precon": "flair_in_dsc",
        "flair_postcon": "flair_postcon_in_dsc",
        "t2w": "t2w_in_dsc",
    }
    requested_underlay_outputs = [
        (f"outputnode.{field}", field)
        for key, field in underlay_outputs.items()
        if key in config.underlays
    ]
    if requested_underlay_outputs:
        workflow.connect([(underlays_wf, outputnode, requested_underlay_outputs)])

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
        (leakage_wf, outputnode, [("outputnode.qc_dict", "preload_qc")]),
        (motion_wf, outputnode, [("outputnode.motion_qc_metrics", "motion_qc")]),
        (qc_wf, outputnode, [("outputnode.aif_metrics", "aif_metrics")]),
    ])

    # Smoothed outputs (only if smooth_fwhm > 0)
    if smoothing_wf is not None:
        workflow.connect([
            (smoothing_wf, outputnode, [
                ("outputnode.smoothed_hemodynamic", "smooth_hemodynamic"),
            ]),
        ])

    # Store config in workflow for reference by child nodes
    workflow.config_dsc = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
