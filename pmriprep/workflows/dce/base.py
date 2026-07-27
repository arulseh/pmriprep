"""DCE (Dynamic Contrast Enhanced) preprocessing workflow for PMRIPrep.

This module implements the main DCE perfusion processing workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Any, Optional, Union

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.config.dce import DCEConfig
from pmriprep.interfaces.common import Reorient, reorient_list_to_ras
from pmriprep.interfaces.dce import (
    B1Mapping,
    T1Mapping,
    VFAPreprocess,
    DCEPreprocess,
    DCEBrainExtract,
    DCERegistration,
)
from pmriprep.interfaces.dce_aif import AIFDetection
from pmriprep.interfaces.dce_aif_qc import DCEAIFQCPlot
from pmriprep.interfaces.dce_compute import ComputeDCEMaps
from pmriprep.data import get_arterial_atlas_cascade
from pmriprep.workflows._perfusion import (
    select_aif_exclusion_mask as _select_aif_exclusion_mask,
    validate_native_mask as _validate_native_mask,
)

# Function-node and construction helpers used by the DCE workflow graph.
from pmriprep.workflows.dce._helpers import (  # noqa: F401
    _build_model_free_map_dict,
    _build_pk_map_dict,
    _collect_qc_metrics,
    _compute_early_mean,
    _create_aif_search_mask,
    _create_dce_registration_filmstrip,
    _dsc_dce_validation_node,
    _dsc_dce_xval_plot_node,
    _external_t1_alignment_plot_node,
    _generate_dce_motion_qc_plot,
    _generate_dce_pk_fit_qc,
    _generate_dce_pk_maps_qc,
    _generate_preprocessing_qc,
    _normalize_by_nawm,
    _resample_dsc_cbv_to_dce,
    _resample_dsc_ncbv_to_dce,
    _signal_to_concentration_node,
    _derive_smoothed_kep,
    _smooth_dce_maps,
    _transform_atlas_cascade_to_dce,
    _transform_atlas_to_dce,
    _transform_b1_to_dce,
    _transform_b1_to_vfa,
    _transform_external_t1_to_dce,
    _transform_lesion_mask_to_dce,
    _transform_mask_to_dce,
    _transform_mask_to_dce_conditional,
    _transform_probseg_to_dce,
    _transform_structural_to_dce,
    _transform_t1_to_dce,
)


def init_dce_pk_modeling_wf(
    *,
    pk_kwargs: dict[str, Any],
    mem_gb: float,
    name: str = "dce_pk_modeling_wf",
) -> pe.Workflow:
    """Build the DCE pharmacokinetic-modeling sub-workflow.

    Wraps :class:`~pmriprep.interfaces.dce_compute.ComputeDCEMaps`, which fits the
    configured tracer-kinetic model (Tofts / extended-Tofts / 2CXM, NLLS or
    Bayesian) to the concentration time series and AIF, producing Ktrans, ve, vp,
    kep, IAUC, model-free, and uncertainty maps.

    Parameters
    ----------
    pk_kwargs : dict
        Keyword arguments forwarded verbatim to ``ComputeDCEMaps`` (model,
        inference, mcmc_samples, thresholds, constraint mode, hematocrit,
        ``random_seed`` when set, etc.). Built by the orchestrator from the run
        configuration.
    mem_gb : float
        Memory estimate (GB) for the PK-modeling node.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The PK-modeling sub-workflow.

    Inputs
    ------
    concentration_file, aif_file, aif_detection_method, aif_uncertainty_inflation_factor
        Tissue concentration time series and arterial input function (with
        detection provenance).
    brain_mask_file, dce_signal_file, baseline_frames, fd_values, bolus_arrival_frame
        DCE brain mask, raw signal (for tSNR), baseline/bolus timing, and motion.
    tr, volume_tr, dsc_dose, dsc_agent_standard_dose
        Acquisition timing and DSC dose scaling (defaults apply on DCE-only runs).
    dsc_ncbv_file
        NAWM-normalized DSC CBV resampled to DCE space for the vp prior.
    tumor_mask_file
        Enhancing-tumor mask (DCE grid) — tissue-arrival reference for AIF BAT
        pre-alignment; empty when tumor segmentation is unavailable.

    Outputs
    -------
    PK parameter maps (Ktrans, ve, vp, kep), IAUC, model-free and
    Bayesian-uncertainty maps, fit diagnostics, and the ``qc_metrics`` dict --
    one outputnode field per distinct ``ComputeDCEMaps`` output consumed
    downstream.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "concentration_file",
                "aif_file",
                "aif_detection_method",
                "aif_uncertainty_inflation_factor",
                "brain_mask_file",
                "volume_tr",
                "dsc_dose",
                "dsc_agent_standard_dose",
                "dce_signal_file",
                "baseline_frames",
                "fd_values",
                "bolus_arrival_frame",
                "dsc_ncbv_file",
                "tumor_mask_file",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "out_ktrans",
                "out_iauc60",
                "out_vp",
                "qc_metrics",
                "out_model_selection",
                "out_ve",
                "out_fp",
                "out_ps",
                "out_e",
                "out_delta_bic",
                "out_vp_at_bound_map",
                "out_r2",
                "out_qc_mask",
                "out_fitting_mask",
                "out_optimizer_converged_mask",
                "out_fitted_curves",
                "out_kep",
                "out_iauc90",
                "out_iauc120",
                "out_peak_enhancement",
                "out_time_to_peak",
                "out_ser",
                "out_wash_in_slope",
                "out_wash_out_slope",
                "out_ktrans_std",
                "out_ve_std",
                "out_vp_std",
                "out_fp_std",
                "out_ps_std",
                "out_bat",
                "out_curve_type",
                "out_tsnr",
                "out_quality_flag",
                "out_ktrans_cv",
                "out_ve_cv",
                "out_vp_cv",
                "out_fp_cv",
                "out_ps_cv",
                "out_ktrans_ci95_lower",
                "out_ktrans_ci95_upper",
                "out_ve_ci95_lower",
                "out_ve_ci95_upper",
                "out_vp_ci95_lower",
                "out_vp_ci95_upper",
                "out_fp_ci95_lower",
                "out_fp_ci95_upper",
                "out_ps_ci95_lower",
                "out_ps_ci95_upper",
            ]
        ),
        name="outputnode",
    )

    pk_modeling = pe.Node(
        ComputeDCEMaps(**pk_kwargs),
        name="pk_modeling",
        mem_gb=mem_gb,
    )

    workflow.connect([
        (inputnode, pk_modeling, [
            ("concentration_file", "concentration_file"),
            ("aif_file", "aif_file"),
            ("aif_detection_method", "aif_detection_method"),
            ("aif_uncertainty_inflation_factor", "aif_uncertainty_inflation_factor"),
            ("brain_mask_file", "brain_mask_file"),
            ("volume_tr", "volume_tr"),
            ("dsc_dose", "dsc_dose"),
            ("dsc_agent_standard_dose", "dsc_agent_standard_dose"),
            ("dce_signal_file", "dce_signal_file"),
            ("baseline_frames", "baseline_frames"),
            ("fd_values", "fd_values"),
            ("bolus_arrival_frame", "bolus_arrival_frame"),
            ("dsc_ncbv_file", "dsc_ncbv_file"),
            ("tumor_mask_file", "tumor_mask_file"),
        ]),
        (pk_modeling, outputnode, [
            ("out_ktrans", "out_ktrans"),
            ("out_iauc60", "out_iauc60"),
            ("out_vp", "out_vp"),
            ("qc_metrics", "qc_metrics"),
            ("out_model_selection", "out_model_selection"),
            ("out_ve", "out_ve"),
            ("out_fp", "out_fp"),
            ("out_ps", "out_ps"),
            ("out_e", "out_e"),
            ("out_delta_bic", "out_delta_bic"),
            ("out_vp_at_bound_map", "out_vp_at_bound_map"),
            ("out_r2", "out_r2"),
            ("out_qc_mask", "out_qc_mask"),
            ("out_fitting_mask", "out_fitting_mask"),
            (
                "out_optimizer_converged_mask",
                "out_optimizer_converged_mask",
            ),
            ("out_fitted_curves", "out_fitted_curves"),
            ("out_kep", "out_kep"),
            ("out_iauc90", "out_iauc90"),
            ("out_iauc120", "out_iauc120"),
            ("out_peak_enhancement", "out_peak_enhancement"),
            ("out_time_to_peak", "out_time_to_peak"),
            ("out_ser", "out_ser"),
            ("out_wash_in_slope", "out_wash_in_slope"),
            ("out_wash_out_slope", "out_wash_out_slope"),
            ("out_ktrans_std", "out_ktrans_std"),
            ("out_ve_std", "out_ve_std"),
            ("out_vp_std", "out_vp_std"),
            ("out_fp_std", "out_fp_std"),
            ("out_ps_std", "out_ps_std"),
            ("out_bat", "out_bat"),
            ("out_curve_type", "out_curve_type"),
            ("out_tsnr", "out_tsnr"),
            ("out_quality_flag", "out_quality_flag"),
            ("out_ktrans_cv", "out_ktrans_cv"),
            ("out_ve_cv", "out_ve_cv"),
            ("out_vp_cv", "out_vp_cv"),
            ("out_fp_cv", "out_fp_cv"),
            ("out_ps_cv", "out_ps_cv"),
            ("out_ktrans_ci95_lower", "out_ktrans_ci95_lower"),
            ("out_ktrans_ci95_upper", "out_ktrans_ci95_upper"),
            ("out_ve_ci95_lower", "out_ve_ci95_lower"),
            ("out_ve_ci95_upper", "out_ve_ci95_upper"),
            ("out_vp_ci95_lower", "out_vp_ci95_lower"),
            ("out_vp_ci95_upper", "out_vp_ci95_upper"),
            ("out_fp_ci95_lower", "out_fp_ci95_lower"),
            ("out_fp_ci95_upper", "out_fp_ci95_upper"),
            ("out_ps_ci95_lower", "out_ps_ci95_lower"),
            ("out_ps_ci95_upper", "out_ps_ci95_upper"),
        ]),
    ])
    return workflow


def init_dce_normalization_wf(
    *,
    name: str = "dce_normalization_wf",
) -> pe.Workflow:
    """Build the DCE NAWM-normalization sub-workflow.

    Normalizes the IAUC60 map to normal-appearing white matter (NAWM).

    Parameters
    ----------
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        The normalization sub-workflow.

    Inputs
    ------
    iauc60_map
        IAUC60 map to normalize.
    nawm_mask
        NAWM reference mask (DCE space).

    Outputs
    -------
    niauc60
        NAWM-normalized IAUC60 map.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["iauc60_map", "nawm_mask"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["niauc60"]),
        name="outputnode",
    )

    normalize_iauc60 = pe.Node(
        niu.Function(
            input_names=["param_map", "nawm_mask"],
            output_names=["out_file"],
            function=_normalize_by_nawm,
        ),
        name="normalize_iauc60",
    )

    workflow.connect([
        (inputnode, normalize_iauc60, [
            ("iauc60_map", "param_map"),
            ("nawm_mask", "nawm_mask"),
        ]),
        (normalize_iauc60, outputnode, [("out_file", "niauc60")]),
    ])
    return workflow


def init_dce_smoothing_wf(
    *,
    fwhm: float,
    name: str = "dce_smoothing_wf",
) -> pe.Workflow:
    """Build the DCE perfusion-map smoothing sub-workflow.

    Spatially smooths the independently estimated PK maps (Ktrans, ve, vp) and
    model-free maps (IAUC60/90/120, peak, SER, wash-in/out) within the brain
    mask. Smoothed Kep is then derived as smoothed Ktrans divided by smoothed
    ve. Parameter-uncertainty maps are deliberately not smoothed. The
    orchestrator instantiates this sub-workflow only when
    ``config.smooth_fwhm > 0``.

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
    ktrans, ve, vp
        Independently estimated PK parameter maps.
    kep_validity
        Native Kep map used only to preserve its defined voxel domain.
    iauc60, iauc90, iauc120, peak, ser, wash_in_slope, wash_out_slope
        Model-free maps.
    brain_mask_file
        DCE-space brain mask (smoothing restricted to brain voxels).

    Outputs
    -------
    smoothed_pk
        Smoothed PK map files.
    smoothed_model_free
        Smoothed model-free map files.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "ktrans",
                "ve",
                "vp",
                "kep_validity",
                "iauc60",
                "iauc90",
                "iauc120",
                "peak",
                "ser",
                "wash_in_slope",
                "wash_out_slope",
                "brain_mask_file",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["smoothed_pk", "smoothed_model_free"]
        ),
        name="outputnode",
    )

    build_pk_dict = pe.Node(
        niu.Function(
            input_names=["ktrans", "ve", "vp"],
            output_names=["map_dict"],
            function=_build_pk_map_dict,
        ),
        name="build_pk_dict",
    )
    smooth_pk = pe.Node(
        niu.Function(
            input_names=["map_files", "brain_mask_file", "fwhm"],
            output_names=["smoothed_files"],
            function=_smooth_dce_maps,
        ),
        name="smooth_pk_maps",
    )
    smooth_pk.inputs.fwhm = fwhm

    build_mf_dict = pe.Node(
        niu.Function(
            input_names=["iauc60", "iauc90", "iauc120", "peak", "ser", "wash_in_slope", "wash_out_slope"],
            output_names=["map_dict"],
            function=_build_model_free_map_dict,
        ),
        name="build_model_free_dict",
    )
    smooth_model_free = pe.Node(
        niu.Function(
            input_names=["map_files", "brain_mask_file", "fwhm"],
            output_names=["smoothed_files"],
            function=_smooth_dce_maps,
        ),
        name="smooth_model_free_maps",
    )
    smooth_model_free.inputs.fwhm = fwhm

    derive_kep = pe.Node(
        niu.Function(
            input_names=[
                "smoothed_pk", "native_kep_file", "brain_mask_file",
                "epsilon_ve", "max_kep",
            ],
            output_names=["smoothed_pk_with_kep"],
            function=_derive_smoothed_kep,
        ),
        name="derive_smoothed_kep",
    )
    from pmriprep.utils.dce_constants import EPSILON_VE, KEP_BOUNDS

    derive_kep.inputs.epsilon_ve = EPSILON_VE
    derive_kep.inputs.max_kep = KEP_BOUNDS.max

    workflow.connect([
        (inputnode, build_pk_dict, [
            ("ktrans", "ktrans"),
            ("ve", "ve"),
            ("vp", "vp"),
        ]),
        (build_pk_dict, smooth_pk, [("map_dict", "map_files")]),
        (inputnode, smooth_pk, [("brain_mask_file", "brain_mask_file")]),
        (inputnode, build_mf_dict, [
            ("iauc60", "iauc60"),
            ("iauc90", "iauc90"),
            ("iauc120", "iauc120"),
            ("peak", "peak"),
            ("ser", "ser"),
            ("wash_in_slope", "wash_in_slope"),
            ("wash_out_slope", "wash_out_slope"),
        ]),
        (build_mf_dict, smooth_model_free, [("map_dict", "map_files")]),
        (inputnode, smooth_model_free, [("brain_mask_file", "brain_mask_file")]),
        (smooth_pk, derive_kep, [("smoothed_files", "smoothed_pk")]),
        (inputnode, derive_kep, [("kep_validity", "native_kep_file")]),
        (inputnode, derive_kep, [("brain_mask_file", "brain_mask_file")]),
        (derive_kep, outputnode, [("smoothed_pk_with_kep", "smoothed_pk")]),
        (smooth_model_free, outputnode, [("smoothed_files", "smoothed_model_free")]),
    ])
    return workflow


def init_dce_lesion_stats_wf(
    *,
    subject_id: str,
    session_id: Optional[str],
    name: str = "dce_lesion_stats_wf",
) -> pe.Workflow:
    """Build the DCE lesion / ROI / user-mask statistics sub-workflow.

    Extracts per-region statistics from the DCE PK and model-free maps within
    three (independently optional) masks: the tumor-segmentation lesion mask, a
    user-supplied ROI mask, and a user-supplied lesion mask. The maps, NAWM
    reference are shared across all three extractions; the
    ROI and user-lesion extractions force a binary mask.

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
    ktrans_map, ve_map, vp_map, kep_map
        PK parameter maps (shared by all three nodes).
    iauc60_map, iauc90_map, iauc120_map, peak_enhancement_map, ttp_map, ser_map, wash_in_map, wash_out_map
        Model-free maps.
    niauc60_map
        NAWM-normalized IAUC60 map.
    nawm_mask
        NAWM reference mask.
    tumor_lesion_mask, mask_origin
        Tumor-segmentation lesion mask (DCE space) and its provenance label.
    roi_mask
        Validated user ROI mask (DCE native space).
    user_mask
        Validated user lesion mask (DCE native space).

    Outputs
    -------
    lesion_stats_json
        Tumor-lesion statistics sidecar.
    roi_stats_json
        User-ROI statistics sidecar.
    user_lesion_stats_json
        User-lesion statistics sidecar.
    """
    from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "ktrans_map",
                "ve_map",
                "vp_map",
                "kep_map",
                "iauc60_map",
                "iauc90_map",
                "iauc120_map",
                "peak_enhancement_map",
                "ttp_map",
                "ser_map",
                "wash_in_map",
                "wash_out_map",
                "niauc60_map",
                "nawm_mask",
                "qc_mask",
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

    lesion_stats = pe.Node(DCELesionStats(), name="lesion_stats")
    lesion_stats.inputs.subject_id = subject_id
    lesion_stats.inputs.session_id = session_id

    user_lesion_stats = pe.Node(
        DCELesionStats(force_binary=True), name="user_lesion_stats"
    )
    user_lesion_stats.inputs.subject_id = subject_id
    user_lesion_stats.inputs.session_id = session_id

    roi_stats = pe.Node(DCELesionStats(force_binary=True), name="roi_stats")
    roi_stats.inputs.subject_id = subject_id
    roi_stats.inputs.session_id = session_id

    # PK / model-free / normalized maps + NAWM are shared by all
    # three stats nodes; the lesion mask differs (tumor-seg / ROI / user), and
    # only the tumor-seg node receives the mask-origin provenance label.
    shared_inputs = [
        ("ktrans_map", "ktrans_map"),
        ("ve_map", "ve_map"),
        ("vp_map", "vp_map"),
        ("kep_map", "kep_map"),
        ("iauc60_map", "iauc60_map"),
        ("iauc90_map", "iauc90_map"),
        ("iauc120_map", "iauc120_map"),
        ("peak_enhancement_map", "peak_enhancement_map"),
        ("ttp_map", "ttp_map"),
        ("ser_map", "ser_map"),
        ("wash_in_map", "wash_in_map"),
        ("wash_out_map", "wash_out_map"),
        ("niauc60_map", "niauc60_map"),
        ("nawm_mask", "nawm_mask"),
        ("qc_mask", "qc_mask"),
    ]
    workflow.connect([
        (inputnode, lesion_stats, shared_inputs + [
            ("tumor_lesion_mask", "lesion_mask"),
            ("mask_origin", "mask_origin"),
        ]),
        (inputnode, user_lesion_stats, shared_inputs + [("user_mask", "lesion_mask")]),
        (inputnode, roi_stats, shared_inputs + [("roi_mask", "lesion_mask")]),
        (lesion_stats, outputnode, [("stats_json", "lesion_stats_json")]),
        (user_lesion_stats, outputnode, [("stats_json", "user_lesion_stats_json")]),
        (roi_stats, outputnode, [("stats_json", "roi_stats_json")]),
    ])
    return workflow


def init_dce_t1_b1_wf(
    *,
    config: DCEConfig,
    subject_id: str,
    session_id: Optional[str],
    name: str = "dce_t1_b1_wf",
) -> pe.Workflow:
    """Build pre-registration DCE T1/B1 preparation.

    Parameters
    ----------
    config : DCEConfig
        DCE configuration controlling T1 source and B1 correction modes.
    subject_id : str
        BIDS subject identifier used in interface provenance.
    session_id : str or None
        Optional BIDS session identifier used in interface provenance.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        Workflow exposing B1/VFA/T1 maps, QC metrics, and plots through an
        explicit outputnode. Inactive outputs remain undefined for external-T1
        and no-T1 modes.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "vfa_files", "vfa_flip_angles", "vfa_tr", "afi_4d_file",
            "afi_tr1", "afi_tr2", "afi_flip_angle", "brain_mask",
            "field_strength", "dce_file", "precomputed_b1_file",
            "b1_nominal_prep_fa", "b1_map_encoding", "csf_probseg",
            "wm_probseg", "gm_probseg",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "b1_map", "b1_qc", "afi_registration_qc_plot", "b1_qc_plot",
            "t1_map", "t1_support_map", "t1_qc", "vfa_qc",
            "vfa_registration_qc_plot", "t1_qc_plot",
        ]),
        name="outputnode",
    )
    workflow.add_nodes([inputnode, outputnode])

    use_external_t1 = config.t1_map_file is not None
    use_vfa_t1 = not config.no_t1 and not use_external_t1
    apply_b1 = config.b1_method != "none"
    prepare_b1 = use_vfa_t1 or apply_b1

    if prepare_b1:
        b1_mapping = pe.Node(
            B1Mapping(skip_b1=(config.b1_method == "none")),
            name="b1_mapping",
        )
        b1_mapping.inputs.subject_id = subject_id
        b1_mapping.inputs.session_id = session_id or ""
        workflow.connect([
            (inputnode, b1_mapping, [
                ("afi_4d_file", "afi_4d_file"), ("afi_tr1", "tr1"),
                ("afi_tr2", "tr2"), ("brain_mask", "brain_mask"),
                ("field_strength", "field_strength"),
                ("afi_flip_angle", "nominal_flip_angle"),
                ("dce_file", "reference_file"),
                ("precomputed_b1_file", "precomputed_b1_file"),
                ("b1_nominal_prep_fa", "b1_nominal_prep_fa"),
                ("b1_map_encoding", "b1_map_encoding"),
            ]),
            (b1_mapping, outputnode, [
                ("out_file", "b1_map"), ("qc_metrics", "b1_qc"),
                ("registration_qc_plot", "afi_registration_qc_plot"),
                ("qc_plot", "b1_qc_plot"),
            ]),
        ])

    if use_vfa_t1:
        vfa_reorient = pe.Node(
            niu.Function(
                input_names=["in_files"], output_names=["out_file"],
                function=reorient_list_to_ras,
            ),
            name="vfa_reorient",
        )
        vfa_preprocess = pe.Node(
            VFAPreprocess(skip_coregistration=False), name="vfa_preprocess",
        )
        transform_b1_to_vfa = pe.Node(
            niu.Function(
                input_names=["b1_map_file", "vfa_ref_file"],
                output_names=["out_file"], function=_transform_b1_to_vfa,
            ),
            name="transform_b1_to_vfa",
        )
        t1_mapping = pe.Node(
            T1Mapping(
                skip_t1=False,
                t1_correction_mode=config.t1_correction_mode,
                t1_method=config.t1_method,
            ),
            name="t1_mapping",
        )
        t1_mapping.inputs.subject_id = subject_id
        t1_mapping.inputs.session_id = session_id or ""
        workflow.connect([
            (inputnode, vfa_reorient, [("vfa_files", "in_files")]),
            (vfa_reorient, vfa_preprocess, [("out_file", "vfa_files")]),
            (inputnode, vfa_preprocess, [("vfa_flip_angles", "flip_angles")]),
            (b1_mapping, transform_b1_to_vfa, [("out_file", "b1_map_file")]),
            (vfa_preprocess, transform_b1_to_vfa, [("reference_file", "vfa_ref_file")]),
            (vfa_preprocess, t1_mapping, [("out_files", "vfa_files")]),
            (inputnode, t1_mapping, [
                ("vfa_flip_angles", "flip_angles"), ("vfa_tr", "tr"),
                ("brain_mask", "brain_mask"), ("field_strength", "field_strength"),
                ("csf_probseg", "csf_probseg"), ("wm_probseg", "wm_probseg"),
                ("gm_probseg", "gm_probseg"),
            ]),
            (transform_b1_to_vfa, t1_mapping, [("out_file", "b1_map")]),
            (b1_mapping, t1_mapping, [("qc_pass", "b1_qc_pass")]),
            (t1_mapping, outputnode, [
                ("out_file", "t1_map"), ("support_file", "t1_support_map"),
                ("qc_metrics", "t1_qc"),
                ("qc_plot", "t1_qc_plot"),
            ]),
            (vfa_preprocess, outputnode, [
                ("qc_metrics", "vfa_qc"),
                ("registration_qc_plot", "vfa_registration_qc_plot"),
            ]),
        ])

    return workflow


def init_dce_native_input_wf(
    *, name: str = "dce_native_input_wf",
) -> pe.Workflow:
    """Build DCE-native input reorientation and early-reference preparation.

    Parameters
    ----------
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        Workflow exposing the RAS-oriented 4D DCE series and its early
        pre-contrast mean image through an explicit outputnode.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["dce_file"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["dce_file", "early_mean"]), name="outputnode",
    )
    dce_reorient = pe.Node(Reorient(), name="dce_reorient")
    dce_early_mean = pe.Node(
        niu.Function(
            input_names=["dce_file"], output_names=["out_file"],
            function=_compute_early_mean,
        ),
        name="dce_early_mean",
    )
    workflow.connect([
        (inputnode, dce_reorient, [("dce_file", "in_file")]),
        (dce_reorient, dce_early_mean, [("out_file", "dce_file")]),
        (dce_reorient, outputnode, [("out_file", "dce_file")]),
        (dce_early_mean, outputnode, [("out_file", "early_mean")]),
    ])
    return workflow


def init_dce_brain_extraction_wf(
    *,
    config: DCEConfig,
    mem_gb: float,
    name: str = "dce_brain_extraction_wf",
) -> pe.Workflow:
    """Build native-space DCE brain extraction.

    Parameters
    ----------
    config : DCEConfig
        DCE configuration providing HD-BET device and environment settings.
    mem_gb : float
        Memory estimate in GB for the extraction node.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        Workflow exposing the extracted brain, native brain mask, method, and
        QC metrics through an explicit outputnode.
    """
    from pmriprep.utils.device import resolve_device_uses_gpu

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["early_mean"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_brain", "out_mask", "method_used", "qc_metrics"]),
        name="outputnode",
    )
    brain_extract = pe.Node(
        DCEBrainExtract(
            method="hdbet",
            bet_frac=0.3,
            fill_holes=True,
            hdbet_device=config.hdbet_device,
            hdbet_conda_env=config.hdbet_conda_env or "",
            use_gpu=resolve_device_uses_gpu(config.hdbet_device),
        ),
        name="dce_brain_extract",
        mem_gb=mem_gb,
    )
    workflow.connect([
        (inputnode, brain_extract, [("early_mean", "dce_mean")]),
        (brain_extract, outputnode, [
            ("out_brain", "out_brain"), ("out_mask", "out_mask"),
            ("method_used", "method_used"), ("qc_metrics", "qc_metrics"),
        ]),
    ])
    return workflow


def init_dce_preprocessing_wf(
    *,
    config: DCEConfig,
    mem_gb: float,
    name: str = "dce_preprocessing_wf",
) -> pe.Workflow:
    """Build DCE motion correction and baseline/bolus detection.

    Parameters
    ----------
    config : DCEConfig
        DCE configuration controlling baseline detection and motion handling.
    mem_gb : float
        Memory estimate in GB for the preprocessing node.
    name : str
        Workflow name.

    Returns
    -------
    workflow : nipype.pipeline.engine.Workflow
        Workflow exposing the eight preprocessing results consumed downstream.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["dce_file", "brain_mask", "volume_tr"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "out_file", "out_params", "mean_img", "baseline_mean_img",
            "baseline_frames", "bolus_arrival_frame", "fd", "qc_metrics",
        ]),
        name="outputnode",
    )
    preprocess = pe.Node(
        DCEPreprocess(
            baseline_frames=config.baseline_frames,
            apply_motion_correction=config.motion_correct,
            motion_thresh=config.motion_thresh,
        ),
        name="dce_preprocess",
        mem_gb=mem_gb,
    )
    fields = [
        "out_file", "out_params", "mean_img", "baseline_mean_img",
        "baseline_frames", "bolus_arrival_frame", "fd", "qc_metrics",
    ]
    workflow.connect([
        (inputnode, preprocess, [
            ("dce_file", "in_file"), ("brain_mask", "brain_mask"),
            ("volume_tr", "volume_tr"),
        ]),
        (preprocess, outputnode, [(field, field) for field in fields]),
    ])
    return workflow


def init_dce_registration_wf(
    *, mem_gb: float, name: str = "dce_registration_wf",
) -> pe.Workflow:
    """Build DCE-to-T1w registration with its consumed output contract."""
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["dce_mean", "t1w_file", "brain_mask"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_file", "out_matrix", "out_h5", "qc_metrics"]),
        name="outputnode",
    )
    registration = pe.Node(DCERegistration(), name="dce_registration", mem_gb=mem_gb)
    workflow.connect([
        (inputnode, registration, [
            ("dce_mean", "dce_mean"), ("t1w_file", "t1w_file"),
            ("brain_mask", "brain_mask"),
        ]),
        (registration, outputnode, [
            ("out_file", "out_file"), ("out_matrix", "out_matrix"),
            ("out_h5", "out_h5"), ("qc_metrics", "qc_metrics"),
        ]),
    ])
    return workflow


def init_dce_atlas_transform_wf(
    *, name: str = "dce_atlas_transform_wf",
) -> pe.Workflow:
    """Build the ordered arterial-atlas cascade transform into DCE space."""
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "t1w_to_mni_xfm", "t1w_to_dce_xfm", "reference_file",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["out_files"]), name="outputnode")
    transform = pe.Node(
        niu.Function(
            input_names=[
                "atlas_files", "t1w_to_mni_xfm", "t1w_to_dce_xfm", "reference_file",
            ],
            output_names=["out_files"],
            function=_transform_atlas_cascade_to_dce,
        ),
        name="transform_atlas_cascade",
    )
    atlas_cascade = get_arterial_atlas_cascade("2mm")
    if atlas_cascade:
        transform.inputs.atlas_files = [str(path) for path in atlas_cascade]
    else:
        import logging as _logging
        _logging.getLogger("pmriprep.workflows.dce").warning(
            "No arterial atlases found. AIF detection will use whole-brain search."
        )
    workflow.connect([
        (inputnode, transform, [
            ("t1w_to_mni_xfm", "t1w_to_mni_xfm"),
            ("t1w_to_dce_xfm", "t1w_to_dce_xfm"),
            ("reference_file", "reference_file"),
        ]),
        (transform, outputnode, [("out_files", "out_files")]),
    ])
    return workflow


def init_dce_anatomical_masks_wf(
    *, has_tissue_seg: bool, name: str = "dce_anatomical_masks_wf",
) -> pe.Workflow:
    """Transform anatomical brain and tissue maps into DCE space."""
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "brain_mask", "csf_probseg", "wm_probseg", "gm_probseg",
            "reference_file", "transform_matrix",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "brain_mask", "csf_probseg", "wm_probseg", "gm_probseg",
        ]),
        name="outputnode",
    )
    transform_brain_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "transform_matrix"],
            output_names=["out_file"], function=_transform_mask_to_dce,
        ),
        name="transform_brain_mask",
    )
    workflow.connect([
        (inputnode, transform_brain_mask, [
            ("brain_mask", "mask_file"), ("reference_file", "reference_file"),
            ("transform_matrix", "transform_matrix"),
        ]),
        (transform_brain_mask, outputnode, [("out_file", "brain_mask")]),
    ])
    if has_tissue_seg:
        for tissue in ("csf", "wm", "gm"):
            transform = pe.Node(
                niu.Function(
                    input_names=["probseg_file", "reference_file", "transform_matrix"],
                    output_names=["out_file"], function=_transform_probseg_to_dce,
                ),
                name=f"transform_{tissue}_probseg",
            )
            workflow.connect([
                (inputnode, transform, [
                    (f"{tissue}_probseg", "probseg_file"),
                    ("reference_file", "reference_file"),
                    ("transform_matrix", "transform_matrix"),
                ]),
                (transform, outputnode, [("out_file", f"{tissue}_probseg")]),
            ])
    return workflow


def init_dce_dsc_resampling_wf(
    *, name: str = "dce_dsc_resampling_wf",
) -> pe.Workflow:
    """Resample relative DSC CBV and normalized DSC CBV into DCE space."""
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "dsc_cbv_file", "dsc_ncbv_file", "dsc_to_t1w_xfm",
            "t1w_to_dce_xfm", "dce_reference",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["dsc_cbv_file", "dsc_ncbv_file"]),
        name="outputnode",
    )
    for label, function in (
        ("cbv", _resample_dsc_cbv_to_dce),
        ("ncbv", _resample_dsc_ncbv_to_dce),
    ):
        node = pe.Node(
            niu.Function(
                input_names=[
                    "dsc_cbv_file", "dsc_to_t1w_xfm", "t1w_to_dce_xfm",
                    "dce_reference",
                ],
                output_names=["cbv_in_dce"], function=function,
            ),
            name=f"resample_dsc_{label}",
        )
        workflow.connect([
            (inputnode, node, [
                (f"dsc_{label}_file", "dsc_cbv_file"),
                ("dsc_to_t1w_xfm", "dsc_to_t1w_xfm"),
                ("t1w_to_dce_xfm", "t1w_to_dce_xfm"),
                ("dce_reference", "dce_reference"),
            ]),
            (node, outputnode, [("cbv_in_dce", f"dsc_{label}_file")]),
        ])
    return workflow


def init_dce_dsc_validation_wf(
    *,
    config: DCEConfig,
    subject_id: str,
    session_id: Optional[str],
    name: str = "dce_dsc_validation_wf",
) -> pe.Workflow:
    """Build DSC-DCE quantitative cross-validation and its QC plot."""
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "vp_file", "dsc_cbv_file", "dsc_ncbv_file", "brain_mask",
            "pk_qc", "dsc_preload_configured", "model_selection_file",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["validation_result", "plot_file"]),
        name="outputnode",
    )
    validation = pe.Node(
        niu.Function(
            input_names=[
                "vp_file", "dsc_cbv_file", "dsc_ncbv_file", "brain_mask",
                "no_dsc_validation", "pk_qc", "dsc_preload_configured",
            ],
            output_names=["validation_result"], function=_dsc_dce_validation_node,
        ),
        name="dsc_dce_validation",
    )
    validation.inputs.no_dsc_validation = config.no_dsc_validation
    plot = pe.Node(
        niu.Function(
            input_names=[
                "vp_file", "dsc_ncbv_file", "brain_mask", "validation_result",
                "subject_id", "session_id", "model_selection_file",
            ],
            output_names=["plot_file"], function=_dsc_dce_xval_plot_node,
        ),
        name="dsc_dce_xval_plot",
    )
    plot.inputs.subject_id = subject_id
    plot.inputs.session_id = session_id or ""
    plot.inputs.model_selection_file = None
    workflow.connect([
        (inputnode, validation, [
            ("vp_file", "vp_file"), ("dsc_cbv_file", "dsc_cbv_file"),
            ("dsc_ncbv_file", "dsc_ncbv_file"), ("brain_mask", "brain_mask"),
            ("pk_qc", "pk_qc"),
            ("dsc_preload_configured", "dsc_preload_configured"),
        ]),
        (inputnode, plot, [
            ("vp_file", "vp_file"), ("dsc_ncbv_file", "dsc_ncbv_file"),
            ("brain_mask", "brain_mask"),
        ]),
        (validation, plot, [("validation_result", "validation_result")]),
        (validation, outputnode, [("validation_result", "validation_result")]),
        (plot, outputnode, [("plot_file", "plot_file")]),
    ])
    if config.model == "auto":
        workflow.connect([
            (inputnode, plot, [("model_selection_file", "model_selection_file")]),
        ])
    return workflow


def init_dce_underlays_wf(
    *, config: DCEConfig, name: str = "dce_underlays_wf",
) -> pe.Workflow:
    """Transform configured structural underlays from T1w into DCE space."""
    specs = (
        ("t1w_precon", "t1w_underlay", "t1w_in_dce", "t1w_in_dce.nii.gz"),
        ("t1w_postcon", "t1w_postcon_underlay", "t1w_postcon_in_dce", "t1w_postcon_in_dce.nii.gz"),
        ("t1w_subtraction", "t1w_subtraction_underlay", "t1w_subtraction_in_dce", "t1w_subtraction_in_dce.nii.gz"),
        ("flair_precon", "flair_in_t1w", "flair_in_dce", "flair_in_dce.nii.gz"),
        ("flair_postcon", "flair_postcon_in_t1w", "flair_postcon_in_dce", "flair_postcon_in_dce.nii.gz"),
        ("t2w", "t2w_in_t1w", "t2w_in_dce", "t2w_in_dce.nii.gz"),
    )
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            *(spec[1] for spec in specs), "reference_file", "transform_matrix",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[spec[2] for spec in specs]), name="outputnode",
    )
    workflow.add_nodes([inputnode, outputnode])
    for key, input_field, output_field, output_name in specs:
        if key not in config.underlays:
            continue
        transform = pe.Node(
            niu.Function(
                input_names=[
                    "structural_file", "reference_file", "transform_matrix", "output_name",
                ],
                output_names=["out_file"], function=_transform_structural_to_dce,
            ),
            name=f"transform_{output_field.removesuffix('_in_dce')}_underlay",
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


def init_dce_qc_collection_wf(
    *, config: DCEConfig, name: str = "dce_qc_collection_wf",
) -> pe.Workflow:
    """Aggregate DCE processing, modeling, validation, and display QC."""
    fields = [
        "b1_qc", "t1_qc", "vfa_qc", "signal_conversion_qc", "preprocess_qc",
        "registration_qc", "aif_qc",
        "pk_qc", "validation_result", "preproc_display", "pk_fit_qc_metrics",
        "pk_maps_qc_metrics", "vp_at_bound_fractions", "nawm_available",
    ]
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=fields), name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=["qc_metrics"]), name="outputnode")
    collect = pe.Node(
        niu.Function(
            input_names=[
                *fields, "use_external_t1", "no_t1", "b1_method",
            ],
            output_names=["qc_metrics"], function=_collect_qc_metrics,
        ),
        name="collect_qc_metrics",
    )
    use_external_t1 = config.t1_map_file is not None
    collect.inputs.use_external_t1 = use_external_t1
    collect.inputs.no_t1 = config.no_t1
    collect.inputs.b1_method = config.b1_method
    use_vfa_t1 = not config.no_t1 and not use_external_t1
    need_b1 = use_vfa_t1 or config.b1_method != "none"
    if not use_vfa_t1:
        collect.inputs.t1_qc = None
        collect.inputs.vfa_qc = None
    if not need_b1:
        collect.inputs.b1_qc = None
    connected = [field for field in fields if not (
        (field == "t1_qc" and not use_vfa_t1)
        or (field == "vfa_qc" and not use_vfa_t1)
        or (field == "b1_qc" and not need_b1)
    )]
    workflow.connect([
        (inputnode, collect, [(field, field) for field in connected]),
        (collect, outputnode, [("qc_metrics", "qc_metrics")]),
    ])
    return workflow


def init_dce_lesion_masks_wf(
    *, name: str = "dce_lesion_masks_wf",
) -> pe.Workflow:
    """Prepare NAWM, ROI, lesion, and tumor masks for DCE consumers."""
    from pmriprep.utils.lesion_stats import combine_tumor_masks

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "nawm_mask", "nawm_needs_transform", "roi_mask", "user_lesion_mask",
            "enhancing_mask", "non_enhancing_mask", "whole_tumor_mask",
            "tumor_segmentation", "reference_file", "transform_matrix",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "nawm_mask", "roi_mask", "user_lesion_mask", "enhancing_mask",
            "non_enhancing_mask", "aif_exclusion_mask", "lesion_mask",
            "mask_origin",
        ]),
        name="outputnode",
    )
    transform_nawm_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "transform_matrix", "needs_transform"],
            output_names=["out_file"], function=_transform_mask_to_dce_conditional,
        ),
        name="transform_nawm_mask",
    )
    validate_roi_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "mask_label"],
            output_names=["validated_mask"], function=_validate_native_mask,
        ),
        name="validate_roi_mask",
    )
    validate_roi_mask.inputs.mask_label = "dce.roi_mask"
    validate_user_lesion_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "mask_label"],
            output_names=["validated_mask"], function=_validate_native_mask,
        ),
        name="validate_user_lesion_mask",
    )
    validate_user_lesion_mask.inputs.mask_label = "dce.lesion_mask"
    transforms: dict[str, pe.Node] = {}
    for label in ("non_enhancing", "enhancing"):
        transforms[label] = pe.Node(
            niu.Function(
                input_names=["mask_file", "reference_file", "transform_matrix"],
                output_names=["out_file"], function=_transform_mask_to_dce,
            ),
            name=f"transform_{label}_mask",
        )
    transform_lesion = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dce_file"], function=_transform_lesion_mask_to_dce,
        ),
        name="transform_lesion",
    )
    select_aif_exclusion = pe.Node(
        niu.Function(
            input_names=["user_lesion_mask", "transformed_lesion_mask"],
            output_names=["out_mask"], function=_select_aif_exclusion_mask,
        ),
        name="select_aif_exclusion",
    )
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
    transform_stats_lesion = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dce_file"], function=_transform_lesion_mask_to_dce,
        ),
        name="transform_stats_lesion",
    )
    workflow.connect([
        (inputnode, transform_nawm_mask, [
            ("nawm_mask", "mask_file"), ("nawm_needs_transform", "needs_transform"),
            ("reference_file", "reference_file"), ("transform_matrix", "transform_matrix"),
        ]),
        (inputnode, validate_roi_mask, [
            ("roi_mask", "mask_file"), ("reference_file", "reference_file"),
        ]),
        (inputnode, validate_user_lesion_mask, [
            ("user_lesion_mask", "mask_file"), ("reference_file", "reference_file"),
        ]),
        (inputnode, transforms["non_enhancing"], [
            ("non_enhancing_mask", "mask_file"), ("reference_file", "reference_file"),
            ("transform_matrix", "transform_matrix"),
        ]),
        (inputnode, transforms["enhancing"], [
            ("enhancing_mask", "mask_file"), ("reference_file", "reference_file"),
            ("transform_matrix", "transform_matrix"),
        ]),
        (inputnode, transform_lesion, [
            ("whole_tumor_mask", "lesion_mask_file"), ("reference_file", "reference_file"),
            ("transform_matrix", "transform_matrix"),
        ]),
        (validate_user_lesion_mask, select_aif_exclusion, [
            ("validated_mask", "user_lesion_mask"),
        ]),
        (transform_lesion, select_aif_exclusion, [
            ("lesion_mask_dce_file", "transformed_lesion_mask"),
        ]),
        (inputnode, combine_masks, [
            ("tumor_segmentation", "tumor_segmentation_file"),
            ("whole_tumor_mask", "whole_tumor_mask_file"),
            ("reference_file", "reference_file"),
        ]),
        (combine_masks, transform_stats_lesion, [("combined_mask_file", "lesion_mask_file")]),
        (inputnode, transform_stats_lesion, [
            ("reference_file", "reference_file"), ("transform_matrix", "transform_matrix"),
        ]),
        (transform_nawm_mask, outputnode, [("out_file", "nawm_mask")]),
        (validate_roi_mask, outputnode, [("validated_mask", "roi_mask")]),
        (validate_user_lesion_mask, outputnode, [("validated_mask", "user_lesion_mask")]),
        (transforms["enhancing"], outputnode, [("out_file", "enhancing_mask")]),
        (transforms["non_enhancing"], outputnode, [("out_file", "non_enhancing_mask")]),
        (select_aif_exclusion, outputnode, [("out_mask", "aif_exclusion_mask")]),
        (transform_stats_lesion, outputnode, [("lesion_mask_dce_file", "lesion_mask")]),
        (combine_masks, outputnode, [("mask_origin", "mask_origin")]),
    ])
    return workflow


def init_dce_preproc_wf(
    bids_dir: Union[str, Path],
    output_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[DCEConfig] = None,
    has_tissue_seg: bool = True,
    random_seed: Optional[int] = None,
    mem_gb: Optional[float] = None,
    mem_limit_gb: Optional[float] = None,
    name: str = "dce_preproc_wf",
) -> pe.Workflow:
    """Initialize the DCE (Dynamic Contrast Enhanced) preprocessing workflow.

    This workflow performs pharmacokinetic modeling of DCE-MRI data to extract
    permeability and perfusion parameters. It includes T1 mapping, AIF detection,
    signal to concentration conversion, and pharmacokinetic model fitting.

    The workflow structure follows NiPreps/fMRIPrep conventions with defined
    input and output nodes that can be connected to other workflows.

    Parameters
    ----------
    bids_dir : str or Path
        Path to the BIDS dataset root directory.
    output_dir : str or Path
        Path to output derivatives directory (e.g., /project/derivatives).
    subject_id : str
        Subject identifier (without "sub-" prefix).
    session_id : str, optional
        Session identifier (without "ses-" prefix).
    config : DCEConfig, optional
        Configuration for DCE processing. If None, default configuration is used.
    has_tissue_seg : bool
        Whether tissue segmentation maps (CSF/WM probseg) are available.
        False in postcon-only mode where FAST is skipped. Default: True.
    random_seed : int, optional
        Random seed for reproducible PK model fitting (Bayesian inference).
    mem_gb : float, optional
        Memory estimate (GB) for heavy nodes. If None, uses a capped default.
    mem_limit_gb : float, optional
        Upper bound (GB) used to cap per-node memory estimates.
    name : str
        Name of the workflow (default: "dce_preproc_wf").

    Returns
    -------
    pe.Workflow
        A Nipype workflow with defined inputs and outputs.

    Inputs
    ------
    dce_file
        Path to 4D DCE NIfTI file.
    t1w_reference
        Path to T1-weighted reference image (typically post-contrast T1w).
    brain_mask
        Path to brain mask NIfTI file.
    wm_probseg
        Path to white matter probability map for tissue-aware processing.
    vfa_files
        Paths to VFA (Variable Flip Angle) images for T1 mapping.
    vfa_flip_angles
        Flip angles (degrees) for each VFA image.
    afi_tr1
        TR1 value (ms) for AFI B1+ mapping.
    afi_tr2
        TR2 value (ms) for AFI B1+ mapping.
    dsc_cbv_file
        Path to DSC-derived CBV map for vp constraint (DSC native space).
        Will be resampled to DCE space via dsc_to_t1w_xfm + t1w_to_dce_xfm.
    dsc_to_t1w_xfm
        Path to DSC→T1w FLIRT transform matrix (for CBV resampling).
    dsc_aif_qc_file
        Path to DSC AIF QC JSON for CBV quality assessment.
    flip_angle
        DCE nominal flip angle in degrees.
    tr
        DCE repetition time in seconds.
    te
        DCE echo time in seconds.
    field_strength
        Magnetic field strength in Tesla.
    r1_relaxivity
        r1 relaxivity of contrast agent (L/mmol/s). If None, use lookup.

    Outputs
    -------
    dce_moco
        Path to motion-corrected 4D DCE NIfTI.
    dce_mean
        Path to mean DCE image (temporal average).
    motion_params
        Path to motion parameters file.
    b1_map
        Path to B1+ map from AFI acquisition.
    t1_map
        Path to T1 map from VFA acquisition.
    ktrans_map
        Path to Ktrans (transfer constant) map.
    ve_map
        Path to ve (extravascular extracellular volume fraction) map.
        Not produced by Patlak model.
    vp_map
        Path to vp (plasma volume fraction) map (extended Tofts/2CXM).
    kep_map
        Path to kep (rate constant) map. Not produced by Patlak model.
    aif_curve
        Path to AIF curve text file.
    aif_mask
        Path to AIF detection mask.
    dce_to_t1w_xfm
        Path to DCE to T1w transformation matrix.
    ktrans_uncertainty
        Path to Ktrans uncertainty map (if Bayesian).
    ve_uncertainty
        Path to ve uncertainty map (if Bayesian).
    qc_metrics
        Dictionary of QC metrics from all processing steps.

    Notes
    -----
    Processing flow:
    1. B1+ mapping from AFI or pre-computed map (optional, skip if config.b1_method="none")
    2. T1 mapping from VFA with B1+ correction (skip if config.no_t1=True)
    3. Motion correction of DCE time series
    4. DCE to T1w registration
    5. AIF detection (auto or population)
    6. Signal to concentration conversion
    7. Pharmacokinetic model fitting (Tofts, Extended Tofts, Patlak, 2CXM)
    8. DSC-DCE cross-validation (optional)

    The workflow accepts DCEConfig for all parameters including model selection,
    Bayesian inference options, and processing flags.
    """
    # Use default config if not provided
    if config is None:
        config = DCEConfig()

    # Create the workflow
    workflow = pe.Workflow(name=name)

    # Store metadata on workflow for reference
    workflow.config_dce = config  # type: ignore[attr-defined]
    workflow.bids_dir = str(bids_dir)  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    # Define input node with all possible inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Core DCE inputs
                "dce_file",
                "t1w_reference",
                "brain_mask",
                "wm_probseg",
                "gm_probseg",
                "csf_probseg",
                # Tumor mask inputs for lesion statistics
                "whole_tumor_mask",  # Undilated whole-tumor mask (fallback for lesion stats)
                "enhancing_mask",  # Enhancing tumor mask for QC voxel selection (optional)
                "non_enhancing_mask",  # Non-enhancing tumor mask for model selection QC (optional)
                "tumor_segmentation",  # Multi-label tumor segmentation for lesion stats (optional)
                # VFA T1 mapping inputs
                "vfa_files",
                "vfa_flip_angles",
                "vfa_tr",
                # AFI B1+ mapping inputs (4D only)
                "afi_4d_file",
                "afi_tr1",
                "afi_tr2",
                "afi_flip_angle",
                # Pre-computed B1+ mapping inputs (TB1TFL or TB1map)
                "precomputed_b1_file",
                "b1_nominal_prep_fa",
                "b1_map_encoding",
                # DSC integration inputs
                "dsc_cbv_file",       # Relative CBV for QC
                "dsc_ncbv_file",      # NAWM-normalized CBV (nCBV) — vp prior input
                "dsc_to_t1w_xfm",
                "dsc_aif_qc_file",
                # DSC dose parameters — scale CNR coverage thresholds
                "dsc_dose",                # bolus dose multiplier
                "dsc_agent_standard_dose", # standard dose (mmol Gd/kg)
                # Acquisition parameters
                "flip_angle",
                "tr",  # Pulse TR for SPGR signal equation
                "volume_tr",  # Volume TR for model auto-selection (temporal resolution)
                "te",
                "mr_acquisition_type",  # BIDS MRAcquisitionType ('3D'/'2D')
                "field_strength",
                "r1_relaxivity",
                "agent_standard_dose",  # Standard clinical dose (mmol/kg) for this agent
                # Registration transforms (from anat pipeline)
                "t1w_to_mni_xfm",  # T1w to MNI transform for atlas-guided AIF
                # Structural underlay inputs
                "t1w_underlay",  # precontrast (or single-source) T1w underlay
                "t1w_postcon_underlay",  # postcontrast T1w underlay (independent of t1w_underlay)
                "t1w_subtraction_underlay",  # post-pre subtraction underlay (precon-anchored T1w space)
                "flair_in_t1w",  # pre-contrast FLAIR coregistered to T1w space (from anat)
                "flair_postcon_in_t1w",  # post-contrast FLAIR coregistered to T1w space (from anat)
                "t2w_in_t1w",  # T2w coregistered to T1w space (from anat)
                # NAWM mask input (for normalization)
                "nawm_mask",
                "nawm_needs_transform",  # True if nawm_mask is in T1w space (tumor seg)
                # Generic ROI mask in DCE native space (optional)
                "roi_mask",
                # User-supplied lesion mask in DCE native space for QC bars
                "user_lesion_mask",
            ]
        ),
        name="inputnode",
    )

    # Default DSC inputs to None (overridden when DSC workflow is connected)
    inputnode.inputs.dsc_cbv_file = None
    inputnode.inputs.dsc_ncbv_file = None
    inputnode.inputs.dsc_to_t1w_xfm = None
    # Default dose = 1.0 (full-dose Gd-DOTA); overridden from config.dsc
    # at the workflows/base.py integration point when DSC is processed
    # in the same session.
    inputnode.inputs.dsc_dose = 1.0
    inputnode.inputs.dsc_agent_standard_dose = 0.1

    inputnode.inputs.nawm_needs_transform = True

    # Default structural underlay inputs to None
    inputnode.inputs.t1w_underlay = None
    inputnode.inputs.t1w_postcon_underlay = None
    inputnode.inputs.t1w_subtraction_underlay = None
    inputnode.inputs.t2w_in_t1w = None

    # Define output node with all possible outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Motion-corrected outputs
                "dce_moco",
                "dce_mean",
                "motion_params",
                # T1/B1 mapping outputs
                "b1_map",
                "t1_map",
                "t1_support_map",
                # Pharmacokinetic parameter maps
                "ktrans_map",
                "ve_map",
                "vp_map",
                "kep_map",
                "fp_map",
                "ps_map",
                "e_map",
                # Model-free maps
                "iauc60_map",
                "iauc90_map",
                "iauc120_map",
                "peak_enhancement_map",
                "time_to_peak_map",
                "ser_map",
                "wash_in_slope_map",
                "wash_out_slope_map",
                "bat_map",
                "curve_type_map",
                # QC maps
                "tsnr_map",
                "qc_mask",
                "r2_map",
                "quality_flag_map",
                "model_selection_map",
                "delta_bic_map",
                # AIF outputs
                "aif_curve",
                "aif_mask",
                # Registration outputs
                "dce_to_t1w_xfm",
                # Inference-specific uncertainty outputs
                "ktrans_uncertainty",
                "ve_uncertainty",
                "vp_uncertainty",
                "fp_uncertainty",
                "ps_uncertainty",
                # Relative uncertainty and interval outputs
                "ktrans_cv",
                "ve_cv",
                "vp_cv",
                "fp_cv",
                "ps_cv",
                "ktrans_ci95_lower",
                "ktrans_ci95_upper",
                "ve_ci95_lower",
                "ve_ci95_upper",
                "vp_ci95_lower",
                "vp_ci95_upper",
                "fp_ci95_lower",
                "fp_ci95_upper",
                "ps_ci95_lower",
                "ps_ci95_upper",
                # Smoothed maps (populated only if config.smooth_fwhm > 0)
                "smooth_pk_maps",
                "smooth_model_free_maps",
                # Normalized maps (requires NAWM mask)
                "niauc60_map",
                # QC metrics
                "qc_metrics",
                # QC plots
                "registration_qc_filmstrip",
                "aif_curve_qc_plot",
                "aif_location_qc_plot",
                "aif_candidates_qc_plot",
                "vfa_registration_qc_plot",  # optional, only if VFA coregistration occurred
                "afi_registration_qc_plot",  # optional, only if AFI coregistration occurred
                "b1_qc_plot",
                "t1_qc_plot",
                # Structural underlays
                "t1w_in_dce",
                "t1w_postcon_in_dce",
                "t1w_subtraction_in_dce",
                "flair_in_dce",
                "flair_postcon_in_dce",
                "t2w_in_dce",
                # Lesion statistics (conditional on lesion mask)
                "lesion_stats_json",
                "lesion_mask_dce",
                # User lesion statistics (conditional on user_lesion_mask)
                "user_lesion_stats_json",
                "user_lesion_mask_dce",
                # ROI statistics (conditional on roi_mask)
                "roi_stats_json",
                "roi_mask_dce",
                # Motion QC
                "motion_qc_plot",
                # DCE QC plots
                "preprocessing_qc_plot",
                "pk_modeling_qc_plot",
                "pk_param_maps_qc_plot",
                "model_selection_qc_plot",
                # Brain mask filmstrip
                "brain_mask_filmstrip",
                # DSC-DCE cross-validation plot
                "dsc_dce_xval_plot",
                # Anatomical masks in DCE space
                "brain_mask",
                "nawm_mask",
            ]
        ),
        name="outputnode",
    )

    # ==========================================================================
    # Node 0: Reorient all raw BIDS inputs to RAS orientation
    # Cap static mem_gb values to respect --mem-gb
    from pmriprep.utils.mem import cap_mem_gb

    _static_mem = cap_mem_gb(2.0, mem_limit_gb)
    _heavy_mem = mem_gb if mem_gb is not None else cap_mem_gb(6.0, mem_limit_gb)

    # ==========================================================================
    # DCE timeseries (4D) - required input
    dce_native_input_wf = init_dce_native_input_wf()
    workflow.connect([
        (inputnode, dce_native_input_wf, [("dce_file", "inputnode.dce_file")]),
    ])

    # NOTE: AFI reorientation is handled inside B1Mapping to avoid
    # issues with nipype passing None to interfaces that expect files.

    # ==========================================================================
    # Nodes 1-2: VFA/B1/T1 pipeline (conditional)
    # ==========================================================================
    # Three T1 source modes:
    #   1. VFA T1 fitting (default): full VFA→B1→T1 chain
    #   2. External T1 map: skip VFA fitting, keep B1 for concentration conversion
    #   3. no_t1: skip T1 mapping, use field-strength default T1
    use_external_t1 = config.t1_map_file is not None
    use_vfa_t1 = not config.no_t1 and not use_external_t1
    # B1 corrects the DCE excitation flip angle independently of the T1 source.
    need_b1 = use_vfa_t1 or config.b1_method != "none"

    t1_b1_wf = init_dce_t1_b1_wf(
        config=config,
        subject_id=subject_id,
        session_id=session_id,
    )
    if need_b1 or use_vfa_t1:
        workflow.connect([
            (inputnode, t1_b1_wf, [
                ("vfa_files", "inputnode.vfa_files"),
                ("vfa_flip_angles", "inputnode.vfa_flip_angles"),
                ("vfa_tr", "inputnode.vfa_tr"),
                ("afi_4d_file", "inputnode.afi_4d_file"),
                ("afi_tr1", "inputnode.afi_tr1"),
                ("afi_tr2", "inputnode.afi_tr2"),
                ("afi_flip_angle", "inputnode.afi_flip_angle"),
                ("brain_mask", "inputnode.brain_mask"),
                ("field_strength", "inputnode.field_strength"),
                ("dce_file", "inputnode.dce_file"),
                ("precomputed_b1_file", "inputnode.precomputed_b1_file"),
                ("b1_nominal_prep_fa", "inputnode.b1_nominal_prep_fa"),
                ("b1_map_encoding", "inputnode.b1_map_encoding"),
            ]),
        ])

    # ==========================================================================
    # Node 3: DCE Preprocessing (motion correction + baseline detection)
    # ==========================================================================
    # Node 3a-pre: Compute early mean for brain extraction
    # Mean of first 3 pre-contrast volumes provides a clean structural image
    # for HD-BET, before motion correction. This gives a DCE-space brain mask
    # for baseline detection (the anat brain mask is in T1w space and may not
    # match DCE dimensions for thin-slab acquisitions).
    # ==========================================================================
    dce_preprocessing_wf = init_dce_preprocessing_wf(
        config=config,
        mem_gb=_heavy_mem,
    )
    workflow.connect([
        (dce_native_input_wf, dce_preprocessing_wf, [
            ("outputnode.dce_file", "inputnode.dce_file"),
        ]),
        (inputnode, dce_preprocessing_wf, [
            ("volume_tr", "inputnode.volume_tr"),
        ]),
    ])

    # ==========================================================================
    # Node 3b: Motion QC Plot
    # Generate motion QC visualization from motion parameters.
    # ==========================================================================
    motion_qc_plot = pe.Node(
        niu.Function(
            input_names=[
                "motion_params_file", "tr", "subject_id", "session_id",
                "motion_applied",
            ],
            output_names=["plot_file"],
            function=_generate_dce_motion_qc_plot,
        ),
        name="motion_qc_plot",
    )
    motion_qc_plot.inputs.subject_id = subject_id
    motion_qc_plot.inputs.session_id = session_id or ""
    motion_qc_plot.inputs.motion_applied = config.motion_correct
    # Connect motion QC plot
    workflow.connect([
        (dce_preprocessing_wf, motion_qc_plot, [
            ("outputnode.out_params", "motion_params_file"),
        ]),
        (inputnode, motion_qc_plot, [("volume_tr", "tr")]),
    ])

    # ==========================================================================
    # Node 3c: DCE Brain Extraction (native space)
    # ==========================================================================
    # Extract brain from DCE mean image using HD-BET with BET fallback.
    # This provides a native DCE-space brain mask that:
    # 1. Accurately reflects the DCE field of view
    # 2. Avoids registration-induced edge artifacts
    # 3. Works directly in DCE native space
    dce_brain_extraction_wf = init_dce_brain_extraction_wf(
        config=config,
        mem_gb=_static_mem,
    )

    # Brain extraction runs on the early pre-contrast mean (before motion
    # correction). The mask is then used by DCEPreprocess for baseline
    # detection and by all downstream nodes.
    workflow.connect([
        (dce_native_input_wf, dce_brain_extraction_wf, [
            ("outputnode.early_mean", "inputnode.early_mean"),
        ]),
        (dce_brain_extraction_wf, dce_preprocessing_wf, [
            ("outputnode.out_mask", "inputnode.brain_mask"),
        ]),
    ])

    # Brain mask filmstrip QC (after brain extraction)
    from pmriprep.workflows.dsc._helpers import _create_perfusion_brain_mask_filmstrip

    dce_brain_mask_qc = pe.Node(
        niu.Function(
            input_names=["mean_file", "brain_mask_file", "subject_id", "session_id"],
            output_names=["out_file"],
            function=_create_perfusion_brain_mask_filmstrip,
        ),
        name="dce_brain_mask_qc",
    )
    dce_brain_mask_qc.inputs.subject_id = subject_id
    dce_brain_mask_qc.inputs.session_id = session_id or ""
    workflow.connect([
        (dce_preprocessing_wf, dce_brain_mask_qc, [
            ("outputnode.baseline_mean_img", "mean_file"),
        ]),
        (dce_brain_extraction_wf, dce_brain_mask_qc, [
            ("outputnode.out_mask", "brain_mask_file"),
        ]),
    ])

    # ==========================================================================
    # Node 3d: DCE Preprocessing QC Plot
    # ==========================================================================
    preprocessing_qc = pe.Node(
        niu.Function(
            input_names=[
                "dce_file", "brain_mask_file",
                "baseline_frames", "bolus_arrival_frame", "tr_ms",
                "volume_tr_s", "te", "subject_id", "session_id",
            ],
            output_names=["plot_file", "preproc_display_stats"],
            function=_generate_preprocessing_qc,
        ),
        name="preprocessing_qc",
    )
    preprocessing_qc.inputs.subject_id = subject_id
    preprocessing_qc.inputs.session_id = session_id or ""

    workflow.connect([
        (dce_preprocessing_wf, preprocessing_qc, [
            ("outputnode.out_file", "dce_file"),
            ("outputnode.baseline_frames", "baseline_frames"),
            ("outputnode.bolus_arrival_frame", "bolus_arrival_frame"),
        ]),
        (dce_brain_extraction_wf, preprocessing_qc, [
            ("outputnode.out_mask", "brain_mask_file"),
        ]),
        (inputnode, preprocessing_qc, [
            ("tr", "tr_ms"),
            ("volume_tr", "volume_tr_s"),
            ("te", "te"),
        ]),
    ])

    # ==========================================================================
    # Node 4: DCE to T1w Registration
    # ==========================================================================
    dce_registration_wf = init_dce_registration_wf(mem_gb=_static_mem)

    # Connect registration inputs
    workflow.connect([
        (inputnode, dce_registration_wf, [
            ("t1w_reference", "inputnode.t1w_file"),
            ("brain_mask", "inputnode.brain_mask"),
        ]),
        (dce_brain_extraction_wf, dce_registration_wf, [
            ("outputnode.out_brain", "inputnode.dce_mean"),
        ]),
    ])

    # ==========================================================================
    # Node 4b: Transform brain mask to DCE space
    # ==========================================================================
    # The brain mask from anat is in T1w space. After registration gives us
    # the T1w→DCE transform, we resample the mask to DCE space.
    dce_anatomical_masks_wf = init_dce_anatomical_masks_wf(
        has_tissue_seg=has_tissue_seg,
    )
    workflow.connect([
        (inputnode, dce_anatomical_masks_wf, [
            ("brain_mask", "inputnode.brain_mask"),
            ("csf_probseg", "inputnode.csf_probseg"),
            ("wm_probseg", "inputnode.wm_probseg"),
            ("gm_probseg", "inputnode.gm_probseg"),
        ]),
        (dce_preprocessing_wf, dce_anatomical_masks_wf, [
            ("outputnode.mean_img", "inputnode.reference_file"),
        ]),
        (dce_registration_wf, dce_anatomical_masks_wf, [
            ("outputnode.out_matrix", "inputnode.transform_matrix"),
        ]),
    ])

    # ==========================================================================
    # Node 4b1b: Transform CSF probseg to DCE space
    # ==========================================================================
    # CSF probability map needs trilinear interpolation (continuous values)
    # and clipping to [0, 1] instead of binarization + hole-filling.
    # Skip entirely in postcon-only mode where FAST segmentation is unavailable.
    if has_tissue_seg and use_vfa_t1:
        workflow.connect([
            (dce_anatomical_masks_wf, t1_b1_wf, [
                ("outputnode.csf_probseg", "inputnode.csf_probseg"),
                ("outputnode.wm_probseg", "inputnode.wm_probseg"),
                ("outputnode.gm_probseg", "inputnode.gm_probseg"),
            ]),
        ])

    # ==========================================================================
    # Node 4b2: Create conservative AIF search mask (Gap 5)
    # ==========================================================================
    # Intersect T1w-transformed mask with native DCE brain mask to exclude
    # non-brain regions. Uses proper brain extraction (HD-BET) instead of
    # simple Otsu thresholding for better accuracy.
    create_aif_mask = pe.Node(
        niu.Function(
            input_names=[
                "t1w_mask_in_dce",
                "dce_native_mask",
            ],
            output_names=["out_file"],
            function=_create_aif_search_mask,
        ),
        name="create_aif_mask",
    )

    workflow.connect([
        (dce_anatomical_masks_wf, create_aif_mask, [
            ("outputnode.brain_mask", "t1w_mask_in_dce"),
        ]),
        # Use native DCE brain mask from HD-BET extraction
        (dce_brain_extraction_wf, create_aif_mask, [
            ("outputnode.out_mask", "dce_native_mask"),
        ]),
    ])

    # ==========================================================================
    # Node 4b3: Transform MNI atlas cascade to DCE space (Gap 6) - optional
    # ==========================================================================
    # Atlas-guided AIF search with tiered cascade (MCA_ACA → CoW → full)
    # for reduced tissue contamination compared to a single broad mask.
    # Only runs if atlas files and t1w_to_mni_xfm are provided.
    dce_atlas_transform_wf = init_dce_atlas_transform_wf()

    workflow.connect([
        (inputnode, dce_atlas_transform_wf, [
            ("t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
        ]),
        (dce_registration_wf, dce_atlas_transform_wf, [
            ("outputnode.out_matrix", "inputnode.t1w_to_dce_xfm"),
        ]),
        (dce_preprocessing_wf, dce_atlas_transform_wf, [
            ("outputnode.mean_img", "inputnode.reference_file"),
        ]),
    ])

    # ==========================================================================
    # Nodes 4c-4d: Transform T1/B1 maps to DCE space (conditional)
    # ==========================================================================
    if use_vfa_t1:
        transform_t1_to_dce = pe.Node(
            niu.Function(
                input_names=["t1_map_file", "reference_file"],
                output_names=["out_file"],
                function=_transform_t1_to_dce,
            ),
            name="transform_t1_to_dce",
        )

        workflow.connect([
            (t1_b1_wf, transform_t1_to_dce, [
                ("outputnode.t1_map", "t1_map_file"),
            ]),
            (dce_preprocessing_wf, transform_t1_to_dce, [
                ("outputnode.mean_img", "reference_file"),
            ]),
        ])

    if need_b1:
        transform_b1_to_dce = pe.Node(
            niu.Function(
                input_names=["b1_map_file", "reference_file"],
                output_names=["out_file"],
                function=_transform_b1_to_dce,
            ),
            name="transform_b1_to_dce",
        )

        workflow.connect([
            (t1_b1_wf, transform_b1_to_dce, [
                ("outputnode.b1_map", "b1_map_file"),
            ]),
            (dce_preprocessing_wf, transform_b1_to_dce, [
                ("outputnode.mean_img", "reference_file"),
            ]),
        ])

    if use_external_t1:
        # External T1: validate units, resample to DCE space
        external_t1_to_dce = pe.Node(
            niu.Function(
                input_names=["t1_map_file", "reference_file"],
                output_names=["out_file"],
                function=_transform_external_t1_to_dce,
            ),
            name="external_t1_to_dce",
        )
        external_t1_to_dce.inputs.t1_map_file = str(config.t1_map_file)
        workflow.connect([
            (dce_preprocessing_wf, external_t1_to_dce, [
                ("outputnode.mean_img", "reference_file"),
            ]),
        ])

        external_t1_qc_plot = pe.Node(
            niu.Function(
                input_names=[
                    "t1_map_file", "dce_reference_file", "brain_mask_file",
                    "subject_id", "session_id",
                ],
                output_names=["plot_file"],
                function=_external_t1_alignment_plot_node,
            ),
            name="external_t1_qc_plot",
        )
        external_t1_qc_plot.inputs.subject_id = subject_id
        external_t1_qc_plot.inputs.session_id = session_id or ""
        workflow.connect([
            (external_t1_to_dce, external_t1_qc_plot, [("out_file", "t1_map_file")]),
            (dce_preprocessing_wf, external_t1_qc_plot, [
                ("outputnode.mean_img", "dce_reference_file"),
            ]),
            (dce_brain_extraction_wf, external_t1_qc_plot, [
                ("outputnode.out_mask", "brain_mask_file"),
            ]),
        ])

    # ==========================================================================
    # Node 5: Signal to Concentration Conversion
    # ==========================================================================
    # This is implemented as a Function node to call the signal_to_concentration
    # utility with proper parameters
    signal_to_conc = pe.Node(
        niu.Function(
            input_names=[
                "dce_file",
                "brain_mask_file",
                "t1_map_file",
                "b1_map_file",
                "flip_angle",
                "tr",
                "field_strength",
                "r1_relaxivity",
                "baseline_frames",
                "no_t1",
                "external_t1",
            ],
            output_names=["concentration_file", "conversion_qc"],
            function=_signal_to_concentration_node,
        ),
        name="signal_to_concentration",
        mem_gb=_heavy_mem,
    )

    # Pass config.no_t1 to the node
    signal_to_conc.inputs.no_t1 = config.no_t1
    signal_to_conc.inputs.external_t1 = use_external_t1

    # Use T1 and B1 maps resampled to DCE space
    workflow.connect([
        (dce_preprocessing_wf, signal_to_conc, [
            ("outputnode.out_file", "dce_file"),
            ("outputnode.baseline_frames", "baseline_frames"),
        ]),
        (dce_brain_extraction_wf, signal_to_conc, [
            ("outputnode.out_mask", "brain_mask_file"),
        ]),
        (inputnode, signal_to_conc, [
            ("flip_angle", "flip_angle"),
            ("tr", "tr"),
            ("field_strength", "field_strength"),
            ("r1_relaxivity", "r1_relaxivity"),
        ]),
    ])

    if use_vfa_t1:
        workflow.connect([
            (transform_t1_to_dce, signal_to_conc, [
                ("out_file", "t1_map_file"),
            ]),
            (transform_b1_to_dce, signal_to_conc, [
                ("out_file", "b1_map_file"),
            ]),
        ])
    elif use_external_t1:
        workflow.connect([
            (external_t1_to_dce, signal_to_conc, [
                ("out_file", "t1_map_file"),
            ]),
        ])
        if need_b1:
            workflow.connect([
                (transform_b1_to_dce, signal_to_conc, [
                    ("out_file", "b1_map_file"),
                ]),
            ])
        else:
            signal_to_conc.inputs.b1_map_file = None
    else:  # no_t1
        signal_to_conc.inputs.t1_map_file = None
        if need_b1:
            workflow.connect([
                (transform_b1_to_dce, signal_to_conc, [
                    ("out_file", "b1_map_file"),
                ]),
            ])
        else:
            signal_to_conc.inputs.b1_map_file = None

    # ==========================================================================
    # Node 6: AIF Detection
    # ==========================================================================
    aif_detection = pe.Node(
        AIFDetection(
            aif_method=config.aif_method,
            aif_peak_floor=config.aif_peak_floor,
            n_candidates=config.n_aif_candidates,
            lesion_distance=config.lesion_distance,
            baseline_frames=config.baseline_frames,
            dose=config.dose,
            hematocrit=config.hematocrit,
        ),
        name="aif_detection",
    )

    # Set target_peak if specified in config
    if config.target_peak is not None:
        aif_detection.inputs.target_peak = config.target_peak

    # Wire external AIF file if provided (bypasses detection entirely)
    if config.aif_file is not None:
        aif_detection.inputs.external_aif_file = str(config.aif_file)

    # Wire user-supplied AIF mask if provided (overrides atlas mask)
    if config.aif_mask_file is not None:
        from pmriprep.utils.lesion_stats import validate_mask_ndim
        validate_mask_ndim(config.aif_mask_file, "dce.aif_mask_file")
        aif_detection.inputs.aif_mask_file = str(config.aif_mask_file)

    workflow.connect([
        (signal_to_conc, aif_detection, [
            ("concentration_file", "concentration_file"),
        ]),
        # Use conservative AIF search mask (Gap 5) instead of plain transformed mask
        (create_aif_mask, aif_detection, [
            ("out_file", "brain_mask_file"),
        ]),
        # Connect atlas cascade for tiered AIF search (Gap 6)
        (dce_atlas_transform_wf, aif_detection, [
            ("outputnode.out_files", "atlas_mask_files"),
        ]),
        (inputnode, aif_detection, [
            ("volume_tr", "tr"),  # Use volume TR for time array, not pulse TR
            ("field_strength", "field_strength"),
            ("agent_standard_dose", "agent_standard_dose"),
            ("mr_acquisition_type", "mr_acquisition_type"),
        ]),
        # Wire auto-detected baseline_frames from dce_preprocess
        (dce_preprocessing_wf, aif_detection, [
            ("outputnode.baseline_frames", "baseline_frames"),
        ]),
    ])

    # ==========================================================================
    # Node 7: PK Modeling (ComputeDCEMaps)
    # ==========================================================================
    pk_kwargs: dict[str, Any] = dict(
        model=config.model,
        inference=config.inference,
        mcmc_samples=config.mcmc_samples,
        tsnr_threshold=config.tsnr_threshold,
        constraint_mode=config.cbv_constraint_mode,
        n_procs=max(1, config.num_processes),
        sloppy=config.sloppy,
        aif_bat_alignment=config.aif_bat_alignment,
    )
    if random_seed is not None:
        pk_kwargs["random_seed"] = random_seed
    pk_wf = init_dce_pk_modeling_wf(
        pk_kwargs=pk_kwargs,
        mem_gb=_heavy_mem,
    )

    workflow.connect([
        (signal_to_conc, pk_wf, [
            ("concentration_file", "inputnode.concentration_file"),
        ]),
        (aif_detection, pk_wf, [
            ("out_aif_curve", "inputnode.aif_file"),
            ("detection_method", "inputnode.aif_detection_method"),
            ("aif_uncertainty_inflation_factor", "inputnode.aif_uncertainty_inflation_factor"),
        ]),
        # Use native DCE brain mask (from HD-BET) for PK modeling
        # This avoids registration-induced edge artifacts and ensures
        # the mask accurately reflects the DCE field of view
        (dce_brain_extraction_wf, pk_wf, [
            ("outputnode.out_mask", "inputnode.brain_mask_file"),
        ]),
        (inputnode, pk_wf, [
            ("volume_tr", "inputnode.volume_tr"),  # For model auto-selection
            # DSC dose parameters scale CNR coverage thresholds in the
            # vp constraint quality gate. Defaults (1.0, 0.1) apply on
            # DCE-only runs.
            ("dsc_dose", "inputnode.dsc_dose"),
            ("dsc_agent_standard_dose", "inputnode.dsc_agent_standard_dose"),
        ]),
        # Raw DCE signal for tSNR computation
        # Also wire baseline_frames so pk_modeling uses the same value
        # as signal_to_concentration (avoids re-auto-detection mismatch)
        (dce_preprocessing_wf, pk_wf, [
            ("outputnode.out_file", "inputnode.dce_signal_file"),
            ("outputnode.baseline_frames", "inputnode.baseline_frames"),
            ("outputnode.fd", "inputnode.fd_values"),
            ("outputnode.bolus_arrival_frame", "inputnode.bolus_arrival_frame"),
        ]),
    ])

    # ==========================================================================
    # Node 7b: Resample DSC CBV to DCE space (for vp constraint)
    # ==========================================================================
    # DSC CBV (128x128x19, 1.88mm) is in DSC native space, but DCE processing
    # operates in DCE native space (256x256x26, 0.94mm). Resample DSC CBV
    # via: DSC → T1w (existing xfm) → DCE (existing xfm).
    dce_dsc_resampling_wf = init_dce_dsc_resampling_wf()
    workflow.connect([
        (inputnode, dce_dsc_resampling_wf, [
            ("dsc_cbv_file", "inputnode.dsc_cbv_file"),
            ("dsc_ncbv_file", "inputnode.dsc_ncbv_file"),
            ("dsc_to_t1w_xfm", "inputnode.dsc_to_t1w_xfm"),
        ]),
        (dce_registration_wf, dce_dsc_resampling_wf, [
            ("outputnode.out_matrix", "inputnode.t1w_to_dce_xfm"),
        ]),
        (dce_preprocessing_wf, dce_dsc_resampling_wf, [
            ("outputnode.mean_img", "inputnode.dce_reference"),
        ]),
        (dce_dsc_resampling_wf, pk_wf, [
            ("outputnode.dsc_ncbv_file", "inputnode.dsc_ncbv_file"),
        ]),
    ])

    # ==========================================================================
    # Node 8: Transform NAWM mask to DCE space (for normalization)
    # FLIRT when from tumor seg (T1w space), passthrough when user-supplied
    # ==========================================================================
    dce_lesion_masks_wf = init_dce_lesion_masks_wf()
    workflow.connect([
        (inputnode, dce_lesion_masks_wf, [
            ("nawm_mask", "inputnode.nawm_mask"),
            ("nawm_needs_transform", "inputnode.nawm_needs_transform"),
            ("roi_mask", "inputnode.roi_mask"),
            ("user_lesion_mask", "inputnode.user_lesion_mask"),
            ("enhancing_mask", "inputnode.enhancing_mask"),
            ("non_enhancing_mask", "inputnode.non_enhancing_mask"),
            ("whole_tumor_mask", "inputnode.whole_tumor_mask"),
            ("tumor_segmentation", "inputnode.tumor_segmentation"),
        ]),
        (dce_preprocessing_wf, dce_lesion_masks_wf, [
            ("outputnode.mean_img", "inputnode.reference_file"),
        ]),
        (dce_registration_wf, dce_lesion_masks_wf, [
            ("outputnode.out_matrix", "inputnode.transform_matrix"),
        ]),
        (dce_lesion_masks_wf, pk_wf, [
            ("outputnode.enhancing_mask", "inputnode.tumor_mask_file"),
        ]),
    ])

    # User-supplied lesion mask is already in DCE native space — no FLIRT.

    # ==========================================================================
    # Node 8a: NAWM Normalization (nKtrans, nIAUC60)
    # ==========================================================================
    normalization_wf = init_dce_normalization_wf()

    # Normalize the model-free IAUC60 map to the NAWM reference.
    workflow.connect([
        (pk_wf, normalization_wf, [
            ("outputnode.out_iauc60", "inputnode.iauc60_map"),
        ]),
        (dce_lesion_masks_wf, normalization_wf, [
            ("outputnode.nawm_mask", "inputnode.nawm_mask"),
        ]),
    ])

    # ==========================================================================
    # Node 9: DSC-DCE Cross-Validation (optional - skip if config.no_dsc_validation=True)
    # ==========================================================================
    dce_dsc_validation_wf = init_dce_dsc_validation_wf(
        config=config, subject_id=subject_id, session_id=session_id,
    )
    # dsc_preload_configured is set from base.py (None when DSC not configured)
    # pk_qc is the full pk_modeling.qc_metrics dict; the function extracts
    # dsc_cbv_constraint mode and relative prior width.
    # dsc_cbv_file is provided by resample_dsc_cbv node (None when DSC unavailable)
    # dsc_ncbv_file is provided by resample_dsc_ncbv (None when NAWM unavailable);
    # when present, validation uses the nCBV-anchored expected ratio.

    workflow.connect([
        (pk_wf, dce_dsc_validation_wf, [
            ("outputnode.out_vp", "inputnode.vp_file"),
            ("outputnode.qc_metrics", "inputnode.pk_qc"),
        ]),
        (dce_anatomical_masks_wf, dce_dsc_validation_wf, [
            ("outputnode.brain_mask", "inputnode.brain_mask"),
        ]),
        # Feed resampled DSC CBV (deferred from Node 7c)
        (dce_dsc_resampling_wf, dce_dsc_validation_wf, [
            ("outputnode.dsc_cbv_file", "inputnode.dsc_cbv_file"),
            ("outputnode.dsc_ncbv_file", "inputnode.dsc_ncbv_file"),
        ]),
    ])

    # ==========================================================================
    # Node 9a: DSC-DCE Cross-Validation Plot
    # ==========================================================================
    # Color scatter by model when model selection is auto
    if config.model == "auto":
        workflow.connect([(pk_wf, dce_dsc_validation_wf, [
            ("outputnode.out_model_selection", "inputnode.model_selection_file"),
        ])])

    # ==========================================================================
    # Node 9a2: DCE PK QC Plots (lightweight wrappers over saved NIfTIs)
    # ==========================================================================
    dce_pk_maps_qc = pe.Node(
        niu.Function(
            input_names=[
                "ktrans_file", "ve_file", "vp_file", "fp_file", "ps_file",
                "brain_mask_file", "qc_mask_file",
                "model_selection_file", "delta_bic_file",
                "enhancing_mask_file", "non_enhancing_mask_file",
                "nawm_mask_file",
                "user_lesion_mask_file", "roi_mask_file",
                "vp_at_bound_map_file",
                "model_name", "subject_id", "session_id",
            ],
            output_names=[
                "pk_param_maps_qc_plot",
                "model_selection_qc_plot",
                "model_selection_regions",
                "vp_at_bound_fractions",
            ],
            function=_generate_dce_pk_maps_qc,
        ),
        name="dce_pk_maps_qc",
    )
    dce_pk_maps_qc.inputs.model_name = config.model
    dce_pk_maps_qc.inputs.subject_id = subject_id
    dce_pk_maps_qc.inputs.session_id = session_id or ""

    workflow.connect([
        (pk_wf, dce_pk_maps_qc, [
            ("outputnode.out_ktrans", "ktrans_file"),
            ("outputnode.out_ve", "ve_file"),
            ("outputnode.out_vp", "vp_file"),
            ("outputnode.out_fp", "fp_file"),
            ("outputnode.out_ps", "ps_file"),
            ("outputnode.out_qc_mask", "qc_mask_file"),
            ("outputnode.out_model_selection", "model_selection_file"),
            ("outputnode.out_delta_bic", "delta_bic_file"),
            ("outputnode.out_vp_at_bound_map", "vp_at_bound_map_file"),
        ]),
        (dce_brain_extraction_wf, dce_pk_maps_qc, [
            ("outputnode.out_mask", "brain_mask_file"),
        ]),
        (dce_lesion_masks_wf, dce_pk_maps_qc, [
            ("outputnode.enhancing_mask", "enhancing_mask_file"),
        ]),
        (dce_lesion_masks_wf, dce_pk_maps_qc, [
            ("outputnode.non_enhancing_mask", "non_enhancing_mask_file"),
        ]),
        (dce_lesion_masks_wf, dce_pk_maps_qc, [
            ("outputnode.nawm_mask", "nawm_mask_file"),
        ]),
        (dce_lesion_masks_wf, dce_pk_maps_qc, [
            ("outputnode.user_lesion_mask", "user_lesion_mask_file"),
        ]),
    ])

    dce_pk_fit_qc = pe.Node(
        niu.Function(
            input_names=[
                "concentration_file", "dce_signal_file", "aif_file",
                "brain_mask_file",
                "ktrans_file", "ve_file", "vp_file", "fp_file", "ps_file",
                "r2_file", "fitting_mask_file",
                "optimizer_converged_mask_file", "quality_flag_file",
                "qc_mask_file", "fitted_curves_file",
                "model_selection_file", "enhancing_mask_file",
                "non_enhancing_mask_file", "nawm_mask_file",
                "volume_tr", "baseline_frames",
                "model_name", "subject_id", "session_id",
            ],
            output_names=["pk_modeling_qc_plot", "pk_voxel_metrics"],
            function=_generate_dce_pk_fit_qc,
        ),
        name="dce_pk_fit_qc",
    )
    dce_pk_fit_qc.inputs.model_name = config.model
    dce_pk_fit_qc.inputs.subject_id = subject_id
    dce_pk_fit_qc.inputs.session_id = session_id or ""

    workflow.connect([
        (signal_to_conc, dce_pk_fit_qc, [
            ("concentration_file", "concentration_file"),
        ]),
        (dce_preprocessing_wf, dce_pk_fit_qc, [
            ("outputnode.out_file", "dce_signal_file"),
            ("outputnode.baseline_frames", "baseline_frames"),
        ]),
        (aif_detection, dce_pk_fit_qc, [
            ("out_aif_curve", "aif_file"),
        ]),
        (dce_brain_extraction_wf, dce_pk_fit_qc, [
            ("outputnode.out_mask", "brain_mask_file"),
        ]),
        (pk_wf, dce_pk_fit_qc, [
            ("outputnode.out_ktrans", "ktrans_file"),
            ("outputnode.out_ve", "ve_file"),
            ("outputnode.out_vp", "vp_file"),
            ("outputnode.out_fp", "fp_file"),
            ("outputnode.out_ps", "ps_file"),
            ("outputnode.out_r2", "r2_file"),
            ("outputnode.out_fitting_mask", "fitting_mask_file"),
            (
                "outputnode.out_optimizer_converged_mask",
                "optimizer_converged_mask_file",
            ),
            ("outputnode.out_quality_flag", "quality_flag_file"),
            ("outputnode.out_qc_mask", "qc_mask_file"),
            ("outputnode.out_fitted_curves", "fitted_curves_file"),
            ("outputnode.out_model_selection", "model_selection_file"),
        ]),
        (inputnode, dce_pk_fit_qc, [
            ("volume_tr", "volume_tr"),
        ]),
        (dce_lesion_masks_wf, dce_pk_fit_qc, [
            ("outputnode.enhancing_mask", "enhancing_mask_file"),
        ]),
        (dce_lesion_masks_wf, dce_pk_fit_qc, [
            ("outputnode.non_enhancing_mask", "non_enhancing_mask_file"),
        ]),
        (dce_lesion_masks_wf, dce_pk_fit_qc, [
            ("outputnode.nawm_mask", "nawm_mask_file"),
        ]),
    ])

    # ==========================================================================
    # Node 9b: Smoothing (conditional on config.smooth_fwhm > 0)
    # ==========================================================================
    smoothing_wf = None
    if config.smooth_fwhm > 0:
        smoothing_wf = init_dce_smoothing_wf(fwhm=config.smooth_fwhm)

    if smoothing_wf is not None:
        # PK and model-free maps from PK modeling; the DCE brain mask fans out
        # to both smoothers inside the sub-workflow.
        workflow.connect([
            (pk_wf, smoothing_wf, [
                ("outputnode.out_ktrans", "inputnode.ktrans"),
                ("outputnode.out_ve", "inputnode.ve"),
                ("outputnode.out_vp", "inputnode.vp"),
                ("outputnode.out_kep", "inputnode.kep_validity"),
                ("outputnode.out_iauc60", "inputnode.iauc60"),
                ("outputnode.out_iauc90", "inputnode.iauc90"),
                ("outputnode.out_iauc120", "inputnode.iauc120"),
                ("outputnode.out_peak_enhancement", "inputnode.peak"),
                ("outputnode.out_ser", "inputnode.ser"),
                ("outputnode.out_wash_in_slope", "inputnode.wash_in_slope"),
                ("outputnode.out_wash_out_slope", "inputnode.wash_out_slope"),
            ]),
            (dce_brain_extraction_wf, smoothing_wf, [
                ("outputnode.out_mask", "inputnode.brain_mask_file"),
            ]),
        ])

    # ==========================================================================
    # Node 10: Registration QC Filmstrip
    # ==========================================================================
    reg_filmstrip = pe.Node(
        niu.Function(
            input_names=[
                "dce_mean_file", "registered_t1w_file", "brain_mask_file",
                "registered_t1w_mask_file", "subject_id", "session_id",
            ],
            output_names=["filmstrip_file"],
            function=_create_dce_registration_filmstrip,
        ),
        name="reg_filmstrip",
    )
    reg_filmstrip.inputs.subject_id = subject_id
    reg_filmstrip.inputs.session_id = session_id or ""

    workflow.connect([
        (dce_preprocessing_wf, reg_filmstrip, [
            ("outputnode.mean_img", "dce_mean_file"),
        ]),
        (dce_registration_wf, reg_filmstrip, [
            ("outputnode.out_file", "registered_t1w_file"),
        ]),
        (dce_brain_extraction_wf, reg_filmstrip, [
            ("outputnode.out_mask", "brain_mask_file"),
        ]),
        (dce_anatomical_masks_wf, reg_filmstrip, [
            ("outputnode.brain_mask", "registered_t1w_mask_file"),
        ]),
    ])

    # ==========================================================================
    # Node 10b: AIF QC Plots
    # ==========================================================================
    aif_qc_plot = pe.Node(
        DCEAIFQCPlot(
            subject_id=subject_id,
            session_id=session_id or "",
        ),
        name="aif_qc_plot",
    )

    workflow.connect([
        (aif_detection, aif_qc_plot, [
            ("out_aif_curve", "aif_curve_file"),
            ("out_aif_mask", "aif_mask_file"),
            ("qc_metrics", "qc_metrics"),
        ]),
        (dce_preprocessing_wf, aif_qc_plot, [
            ("outputnode.out_file", "dce_file"),
        ]),
        (dce_brain_extraction_wf, aif_qc_plot, [
            ("outputnode.out_mask", "brain_mask_file"),
        ]),
        (inputnode, aif_qc_plot, [
            ("volume_tr", "tr"),
            ("t1w_to_mni_xfm", "t1w_to_mni_xfm"),
        ]),
        (dce_registration_wf, aif_qc_plot, [
            ("outputnode.out_matrix", "t1w_to_dce_xfm"),
        ]),
    ])

    # ==========================================================================
    # Node 11: Collect QC Metrics
    # ==========================================================================
    dce_qc_collection_wf = init_dce_qc_collection_wf(config=config)

    if use_vfa_t1:
        workflow.connect([
            (t1_b1_wf, dce_qc_collection_wf, [
                ("outputnode.b1_qc", "inputnode.b1_qc"),
                ("outputnode.t1_qc", "inputnode.t1_qc"),
                ("outputnode.vfa_qc", "inputnode.vfa_qc"),
            ]),
        ])
    elif need_b1:
        workflow.connect([
            (t1_b1_wf, dce_qc_collection_wf, [
                ("outputnode.b1_qc", "inputnode.b1_qc"),
            ]),
        ])

    workflow.connect([
        (signal_to_conc, dce_qc_collection_wf, [
            ("conversion_qc", "inputnode.signal_conversion_qc"),
        ]),
        (dce_preprocessing_wf, dce_qc_collection_wf, [
            ("outputnode.qc_metrics", "inputnode.preprocess_qc"),
        ]),
        (dce_registration_wf, dce_qc_collection_wf, [
            ("outputnode.qc_metrics", "inputnode.registration_qc"),
        ]),
        (aif_detection, dce_qc_collection_wf, [
            ("qc_metrics", "inputnode.aif_qc"),
        ]),
        (pk_wf, dce_qc_collection_wf, [
            ("outputnode.qc_metrics", "inputnode.pk_qc"),
        ]),
        (dce_dsc_validation_wf, dce_qc_collection_wf, [
            ("outputnode.validation_result", "inputnode.validation_result"),
        ]),
        (preprocessing_qc, dce_qc_collection_wf, [
            ("preproc_display_stats", "inputnode.preproc_display"),
        ]),
        # Wire QC metrics from the DCE PK plot nodes
        (dce_pk_fit_qc, dce_qc_collection_wf, [
            ("pk_voxel_metrics", "inputnode.pk_fit_qc_metrics"),
        ]),
        (dce_pk_maps_qc, dce_qc_collection_wf, [
            ("model_selection_regions", "inputnode.pk_maps_qc_metrics"),
            ("vp_at_bound_fractions", "inputnode.vp_at_bound_fractions"),
        ]),
        # out_file is a path (truthy) when normalization ran, "" otherwise
        (normalization_wf, dce_qc_collection_wf, [
            ("outputnode.niauc60", "inputnode.nawm_available"),
        ]),
    ])

    # ==========================================================================
    # Node 11a: Lesion Statistics (conditional on lesion mask availability)
    # ==========================================================================
    workflow.connect([
        (dce_lesion_masks_wf, aif_detection, [
            ("outputnode.aif_exclusion_mask", "lesion_mask_file"),
        ]),
    ])

    # Lesion / ROI / user-mask statistics extraction
    lesion_stats_wf = init_dce_lesion_stats_wf(
        subject_id=subject_id,
        session_id=session_id,
    )

    # PK + normalized maps + NAWM + field strength are shared by all three
    # stats nodes (fanned out inside the sub-wf); the three lesion masks
    # (tumor-seg, user ROI, user lesion) route to their respective nodes.
    workflow.connect([
        (pk_wf, lesion_stats_wf, [
            ("outputnode.out_ktrans", "inputnode.ktrans_map"),
            ("outputnode.out_ve", "inputnode.ve_map"),
            ("outputnode.out_vp", "inputnode.vp_map"),
            ("outputnode.out_kep", "inputnode.kep_map"),
            ("outputnode.out_iauc60", "inputnode.iauc60_map"),
            ("outputnode.out_iauc90", "inputnode.iauc90_map"),
            ("outputnode.out_iauc120", "inputnode.iauc120_map"),
            ("outputnode.out_peak_enhancement", "inputnode.peak_enhancement_map"),
            ("outputnode.out_time_to_peak", "inputnode.ttp_map"),
            ("outputnode.out_ser", "inputnode.ser_map"),
            ("outputnode.out_wash_in_slope", "inputnode.wash_in_map"),
            ("outputnode.out_wash_out_slope", "inputnode.wash_out_map"),
            ("outputnode.out_qc_mask", "inputnode.qc_mask"),
        ]),
        (normalization_wf, lesion_stats_wf, [
            ("outputnode.niauc60", "inputnode.niauc60_map"),
        ]),
        (dce_lesion_masks_wf, lesion_stats_wf, [
            ("outputnode.nawm_mask", "inputnode.nawm_mask"),
            ("outputnode.lesion_mask", "inputnode.tumor_lesion_mask"),
            ("outputnode.mask_origin", "inputnode.mask_origin"),
            ("outputnode.user_lesion_mask", "inputnode.user_mask"),
            ("outputnode.roi_mask", "inputnode.roi_mask"),
        ]),
    ])

    workflow.connect([
        (dce_lesion_masks_wf, outputnode, [
            ("outputnode.user_lesion_mask", "user_lesion_mask_dce"),
        ]),
        (lesion_stats_wf, outputnode, [
            ("outputnode.user_lesion_stats_json", "user_lesion_stats_json"),
        ]),
    ])

    # ==========================================================================
    # Node 12: Structural underlays (conditional on config.underlays)
    # ==========================================================================
    dce_underlays_wf = init_dce_underlays_wf(config=config)
    if config.underlays:
        workflow.connect([
            (inputnode, dce_underlays_wf, [
                ("t1w_underlay", "inputnode.t1w_underlay"),
                ("t1w_postcon_underlay", "inputnode.t1w_postcon_underlay"),
                ("t1w_subtraction_underlay", "inputnode.t1w_subtraction_underlay"),
                ("flair_in_t1w", "inputnode.flair_in_t1w"),
                ("flair_postcon_in_t1w", "inputnode.flair_postcon_in_t1w"),
                ("t2w_in_t1w", "inputnode.t2w_in_t1w"),
            ]),
            (dce_preprocessing_wf, dce_underlays_wf, [
                ("outputnode.mean_img", "inputnode.reference_file"),
            ]),
            (dce_registration_wf, dce_underlays_wf, [
                ("outputnode.out_matrix", "inputnode.transform_matrix"),
            ]),
        ])

    # ==========================================================================
    # Connect outputs
    # ==========================================================================
    workflow.connect([
        # Motion-corrected outputs
        (dce_preprocessing_wf, outputnode, [
            ("outputnode.out_file", "dce_moco"),
            ("outputnode.mean_img", "dce_mean"),
            ("outputnode.out_params", "motion_params"),
        ]),
        # Pharmacokinetic parameter maps
        (pk_wf, outputnode, [
            ("outputnode.out_ktrans", "ktrans_map"),
            ("outputnode.out_ve", "ve_map"),
            ("outputnode.out_vp", "vp_map"),
            ("outputnode.out_kep", "kep_map"),
            ("outputnode.out_fp", "fp_map"),
            ("outputnode.out_ps", "ps_map"),
            ("outputnode.out_e", "e_map"),
        ]),
        # Model-free maps
        (pk_wf, outputnode, [
            ("outputnode.out_iauc60", "iauc60_map"),
            ("outputnode.out_iauc90", "iauc90_map"),
            ("outputnode.out_iauc120", "iauc120_map"),
            ("outputnode.out_peak_enhancement", "peak_enhancement_map"),
            ("outputnode.out_time_to_peak", "time_to_peak_map"),
            ("outputnode.out_ser", "ser_map"),
            ("outputnode.out_wash_in_slope", "wash_in_slope_map"),
            ("outputnode.out_wash_out_slope", "wash_out_slope_map"),
            ("outputnode.out_bat", "bat_map"),
            ("outputnode.out_curve_type", "curve_type_map"),
        ]),
        # QC maps
        (pk_wf, outputnode, [
            ("outputnode.out_tsnr", "tsnr_map"),
            ("outputnode.out_qc_mask", "qc_mask"),
            ("outputnode.out_r2", "r2_map"),
            ("outputnode.out_quality_flag", "quality_flag_map"),
            ("outputnode.out_model_selection", "model_selection_map"),
            ("outputnode.out_delta_bic", "delta_bic_map"),
        ]),
        # Inference-specific uncertainty maps
        (pk_wf, outputnode, [
            ("outputnode.out_ktrans_std", "ktrans_uncertainty"),
            ("outputnode.out_ve_std", "ve_uncertainty"),
            ("outputnode.out_vp_std", "vp_uncertainty"),
            ("outputnode.out_fp_std", "fp_uncertainty"),
            ("outputnode.out_ps_std", "ps_uncertainty"),
        ]),
        # Relative uncertainty and interval maps
        (pk_wf, outputnode, [
            ("outputnode.out_ktrans_cv", "ktrans_cv"),
            ("outputnode.out_ve_cv", "ve_cv"),
            ("outputnode.out_vp_cv", "vp_cv"),
            ("outputnode.out_fp_cv", "fp_cv"),
            ("outputnode.out_ps_cv", "ps_cv"),
            ("outputnode.out_ktrans_ci95_lower", "ktrans_ci95_lower"),
            ("outputnode.out_ktrans_ci95_upper", "ktrans_ci95_upper"),
            ("outputnode.out_ve_ci95_lower", "ve_ci95_lower"),
            ("outputnode.out_ve_ci95_upper", "ve_ci95_upper"),
            ("outputnode.out_vp_ci95_lower", "vp_ci95_lower"),
            ("outputnode.out_vp_ci95_upper", "vp_ci95_upper"),
            ("outputnode.out_fp_ci95_lower", "fp_ci95_lower"),
            ("outputnode.out_fp_ci95_upper", "fp_ci95_upper"),
            ("outputnode.out_ps_ci95_lower", "ps_ci95_lower"),
            ("outputnode.out_ps_ci95_upper", "ps_ci95_upper"),
        ]),
        # AIF outputs
        (aif_detection, outputnode, [
            ("out_aif_curve", "aif_curve"),
            ("out_aif_mask", "aif_mask"),
        ]),
        # Registration outputs (.h5 for derivatives)
        (dce_registration_wf, outputnode, [
            ("outputnode.out_h5", "dce_to_t1w_xfm"),
        ]),
        # QC metrics
        (dce_qc_collection_wf, outputnode, [
            ("outputnode.qc_metrics", "qc_metrics"),
        ]),
        # QC plots - registration filmstrip
        (reg_filmstrip, outputnode, [
            ("filmstrip_file", "registration_qc_filmstrip"),
        ]),
        # QC plots - AIF QC
        (aif_qc_plot, outputnode, [
            ("out_aif_plot", "aif_curve_qc_plot"),
            ("out_aif_location", "aif_location_qc_plot"),
            ("out_aif_candidates", "aif_candidates_qc_plot"),
        ]),
        # Motion QC plot
        (motion_qc_plot, outputnode, [
            ("plot_file", "motion_qc_plot"),
        ]),
        # Brain mask filmstrip
        (dce_brain_mask_qc, outputnode, [
            ("out_file", "brain_mask_filmstrip"),
        ]),
        # DCE QC plots
        (preprocessing_qc, outputnode, [
            ("plot_file", "preprocessing_qc_plot"),
        ]),
        (dce_pk_fit_qc, outputnode, [
            ("pk_modeling_qc_plot", "pk_modeling_qc_plot"),
        ]),
        (dce_pk_maps_qc, outputnode, [
            ("pk_param_maps_qc_plot", "pk_param_maps_qc_plot"),
            ("model_selection_qc_plot", "model_selection_qc_plot"),
        ]),
        # DSC-DCE cross-validation plot
        (dce_dsc_validation_wf, outputnode, [
            ("outputnode.plot_file", "dsc_dce_xval_plot"),
        ]),
    ])

    # T1/B1/VFA output connections (conditional on T1 source)
    if use_vfa_t1:
        workflow.connect([
            (transform_b1_to_dce, outputnode, [("out_file", "b1_map")]),
            (t1_b1_wf, outputnode, [
                ("outputnode.t1_map", "t1_map"),
                ("outputnode.t1_support_map", "t1_support_map"),
                ("outputnode.vfa_registration_qc_plot", "vfa_registration_qc_plot"),
                ("outputnode.afi_registration_qc_plot", "afi_registration_qc_plot"),
                ("outputnode.b1_qc_plot", "b1_qc_plot"),
                ("outputnode.t1_qc_plot", "t1_qc_plot"),
            ]),
        ])
    elif need_b1:
        workflow.connect([
            (transform_b1_to_dce, outputnode, [("out_file", "b1_map")]),
            (t1_b1_wf, outputnode, [
                ("outputnode.afi_registration_qc_plot", "afi_registration_qc_plot"),
                ("outputnode.b1_qc_plot", "b1_qc_plot"),
            ]),
        ])
    if use_external_t1:
        workflow.connect([
            (external_t1_to_dce, outputnode, [("out_file", "t1_map")]),
            (external_t1_qc_plot, outputnode, [("plot_file", "t1_qc_plot")]),
        ])

    # Structural underlay output connections (conditional)
    for key, field in (
        ("t1w_precon", "t1w_in_dce"),
        ("t1w_postcon", "t1w_postcon_in_dce"),
        ("t1w_subtraction", "t1w_subtraction_in_dce"),
        ("flair_precon", "flair_in_dce"),
        ("flair_postcon", "flair_postcon_in_dce"),
        ("t2w", "t2w_in_dce"),
    ):
        if key in config.underlays:
            workflow.connect([
                (dce_underlays_wf, outputnode, [(f"outputnode.{field}", field)]),
            ])

    # Lesion statistics output connections
    # Use the combined tumor mask (with multi-label support) for lesion_mask_dce output
    workflow.connect([
        (dce_lesion_masks_wf, outputnode, [
            ("outputnode.lesion_mask", "lesion_mask_dce"),
        ]),
        (lesion_stats_wf, outputnode, [
            ("outputnode.lesion_stats_json", "lesion_stats_json"),
        ]),
    ])

    # Connect validated ROI mask to the PK-maps QC overlay
    workflow.connect([
        (dce_lesion_masks_wf, dce_pk_maps_qc, [
            ("outputnode.roi_mask", "roi_mask_file"),
        ]),
    ])
    # Connect ROI stats outputs to outputnode
    workflow.connect([
        (dce_lesion_masks_wf, outputnode, [
            ("outputnode.roi_mask", "roi_mask_dce"),
        ]),
        (lesion_stats_wf, outputnode, [
            ("outputnode.roi_stats_json", "roi_stats_json"),
        ]),
    ])

    # Smoothing output connections (only if config.smooth_fwhm > 0)
    if smoothing_wf is not None:
        workflow.connect([
            (smoothing_wf, outputnode, [
                ("outputnode.smoothed_pk", "smooth_pk_maps"),
                ("outputnode.smoothed_model_free", "smooth_model_free_maps"),
            ]),
        ])

    # Normalized map output connections (requires NAWM mask from tumor segmentation)
    workflow.connect([
        (normalization_wf, outputnode, [("outputnode.niauc60", "niauc60_map")]),
    ])

    # Anatomical masks in DCE space output connections
    # Use native DCE brain mask from HD-BET extraction for output
    workflow.connect([
        (dce_brain_extraction_wf, outputnode, [
            ("outputnode.out_mask", "brain_mask"),
        ]),
        (dce_lesion_masks_wf, outputnode, [
            ("outputnode.nawm_mask", "nawm_mask"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Provenance logging (mirrors DSC write_provenance node)
    # -------------------------------------------------------------------
    import dataclasses
    config_dict = dataclasses.asdict(config)
    for k, v in config_dict.items():
        if isinstance(v, Path):
            config_dict[k] = str(v)

    return workflow
