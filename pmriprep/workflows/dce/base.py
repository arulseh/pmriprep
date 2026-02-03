"""DCE (Dynamic Contrast Enhanced) preprocessing workflow for PMRIPrep.

This module implements the main DCE perfusion processing workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Any, Optional, Union

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.config.dce import DCEConfig
from pmriprep.interfaces.dce import (
    B1Mapping,
    T1Mapping,
    VFAPreprocess,
    DCEPreprocess,
    DCERegistration,
)
from pmriprep.interfaces.dce_aif import AIFDetection
from pmriprep.interfaces.dce_compute import ComputeDCEMaps


def init_dce_preproc_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[DCEConfig] = None,
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
    subject_id : str
        Subject identifier (without "sub-" prefix).
    session_id : str, optional
        Session identifier (without "ses-" prefix).
    config : DCEConfig, optional
        Configuration for DCE processing. If None, default configuration is used.
    name : str
        Name of the workflow (default: "dce_preproc_wf").

    Returns
    -------
    workflow : pe.Workflow
        A Nipype workflow with defined inputs and outputs.

    Inputs (from inputnode)
    -----------------------
    dce_file : str
        Path to 4D DCE NIfTI file.
    t1w_reference : str
        Path to T1-weighted reference image (typically post-contrast T1w).
    brain_mask : str
        Path to brain mask NIfTI file.
    wm_probseg : str
        Path to white matter probability map for BBR registration.
    lesion_mask : str, optional
        Path to lesion mask for registration weighting and AIF exclusion.
    vfa_files : list of str
        Paths to VFA (Variable Flip Angle) images for T1 mapping.
    vfa_flip_angles : list of float
        Flip angles (degrees) for each VFA image.
    afi_tr1_file : str, optional
        Path to AFI TR1 image for B1+ mapping.
    afi_tr2_file : str, optional
        Path to AFI TR2 image for B1+ mapping.
    afi_tr1 : float, optional
        TR1 value (ms) for AFI B1+ mapping.
    afi_tr2 : float, optional
        TR2 value (ms) for AFI B1+ mapping.
    dsc_cbv_file : str, optional
        Path to DSC-derived CBV map for vp constraint.
    dsc_aif_qc_file : str, optional
        Path to DSC AIF QC JSON for CBV quality assessment.
    dwi_file : str, optional
        Path to DWI image for tortuosity correction.
    dwi_bval : float, optional
        b-value for DWI tortuosity calculation.
    flip_angle : float
        DCE nominal flip angle in degrees.
    tr : float
        DCE repetition time in seconds.
    te : float
        DCE echo time in seconds.
    field_strength : float
        Magnetic field strength in Tesla.
    r1_relaxivity : float, optional
        r1 relaxivity of contrast agent (L/mmol/s). If None, use lookup.

    Outputs (from outputnode)
    -------------------------
    dce_moco : str
        Path to motion-corrected 4D DCE NIfTI.
    dce_mean : str
        Path to mean DCE image (temporal average).
    motion_params : str
        Path to motion parameters file.
    b1_map : str, optional
        Path to B1+ map from AFI acquisition.
    t1_map : str
        Path to T1 map from VFA acquisition.
    ktrans_map : str
        Path to Ktrans (transfer constant) map.
    ve_map : str
        Path to ve (extravascular extracellular volume fraction) map.
    vp_map : str, optional
        Path to vp (plasma volume fraction) map (extended Tofts/2CXM).
    kep_map : str
        Path to kep (rate constant) map.
    aif_curve : str
        Path to AIF curve text file.
    aif_mask : str
        Path to AIF detection mask.
    dce_to_t1w_xfm : str
        Path to DCE to T1w transformation matrix.
    ktrans_uncertainty : str, optional
        Path to Ktrans uncertainty map (if Bayesian).
    ve_uncertainty : str, optional
        Path to ve uncertainty map (if Bayesian).
    qc_metrics : dict
        Dictionary of QC metrics from all processing steps.

    Notes
    -----
    Processing flow:
    1. B1+ mapping from AFI (optional, skip if config.no_b1=True)
    2. T1 mapping from VFA with B1+ correction (skip if config.no_t1=True)
    3. Motion correction of DCE time series
    4. DCE to T1w registration
    5. AIF detection (auto or population)
    6. Signal to concentration conversion
    7. Pharmacokinetic model fitting (Tofts, Extended Tofts, Patlak, 2CXM)
    8. Tortuosity correction (optional, requires DWI)
    9. DSC-DCE cross-validation (optional)

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
                "csf_probseg",
                "lesion_mask",
                # VFA T1 mapping inputs
                "vfa_files",
                "vfa_flip_angles",
                "vfa_tr",
                # AFI B1+ mapping inputs
                "afi_tr1_file",
                "afi_tr2_file",
                "afi_4d_file",
                "afi_tr1",
                "afi_tr2",
                # DSC integration inputs
                "dsc_cbv_file",
                "dsc_aif_qc_file",
                # DWI tortuosity inputs
                "dwi_file",
                "dwi_bval",
                # Acquisition parameters
                "flip_angle",
                "tr",
                "te",
                "field_strength",
                "r1_relaxivity",
                # Registration transforms (from anat pipeline)
                "t1w_to_mni_xfm",  # T1w to MNI transform for atlas-guided AIF
                # Atlas inputs (optional)
                "atlas_mask_file",  # MNI-space atlas mask for AIF search territories
                # Structural underlay inputs
                "flair_in_t1w",  # FLAIR coregistered to T1w space (from anat)
            ]
        ),
        name="inputnode",
    )

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
                # Pharmacokinetic parameter maps
                "ktrans_map",
                "ve_map",
                "vp_map",
                "kep_map",
                # AIF outputs
                "aif_curve",
                "aif_mask",
                # Registration outputs
                "dce_to_t1w_xfm",
                # Uncertainty outputs (Bayesian)
                "ktrans_uncertainty",
                "ve_uncertainty",
                "vp_uncertainty",
                # QC metrics
                "qc_metrics",
                # Structural underlays
                "t1w_in_dce",
                "flair_in_dce",
                # Lesion statistics (conditional on lesion mask)
                "lesion_stats_json",
                "lesion_stats_txt",
                "lesion_mask_dce",
            ]
        ),
        name="outputnode",
    )

    # ==========================================================================
    # Node 1: B1+ Mapping (optional - skip if config.no_b1=True)
    # ==========================================================================
    b1_mapping = pe.Node(
        B1Mapping(
            skip_b1=config.no_b1,
        ),
        name="b1_mapping",
    )

    # Connect B1 mapping inputs
    workflow.connect([
        (inputnode, b1_mapping, [
            ("afi_tr1_file", "tr1_file"),
            ("afi_tr2_file", "tr2_file"),
            ("afi_4d_file", "afi_4d_file"),
            ("afi_tr1", "tr1"),
            ("afi_tr2", "tr2"),
            ("brain_mask", "brain_mask"),
        ]),
    ])

    # ==========================================================================
    # Node 1b: VFA Preprocessing (coregistration + quality assessment)
    # ==========================================================================
    vfa_preprocess = pe.Node(
        VFAPreprocess(
            skip_coregistration=False,
            skip_flip_validation=config.skip_flip_validation,
        ),
        name="vfa_preprocess",
    )

    workflow.connect([
        (inputnode, vfa_preprocess, [
            ("vfa_files", "vfa_files"),
            ("vfa_flip_angles", "flip_angles"),
            ("vfa_tr", "tr"),
            ("field_strength", "field_strength"),
        ]),
    ])

    # ==========================================================================
    # Node 1c: Transform B1+ map to VFA space (for T1 fitting)
    # ==========================================================================
    transform_b1_to_vfa = pe.Node(
        niu.Function(
            input_names=["b1_map_file", "vfa_ref_file"],
            output_names=["out_file"],
            function=_transform_b1_to_vfa,
        ),
        name="transform_b1_to_vfa",
    )

    workflow.connect([
        (b1_mapping, transform_b1_to_vfa, [
            ("out_file", "b1_map_file"),
        ]),
        (vfa_preprocess, transform_b1_to_vfa, [
            ("reference_file", "vfa_ref_file"),
        ]),
    ])

    # ==========================================================================
    # Node 2: T1 Mapping (optional - skip if config.no_t1=True)
    # ==========================================================================
    t1_mapping = pe.Node(
        T1Mapping(
            skip_t1=config.no_t1,
            lenient_qc=config.lenient_t1_qc,
            t1_correction_mode=config.t1_correction,
        ),
        name="t1_mapping",
    )

    # Connect T1 mapping inputs — use coregistered VFA and resampled B1
    workflow.connect([
        (vfa_preprocess, t1_mapping, [
            ("out_files", "vfa_files"),
        ]),
        (inputnode, t1_mapping, [
            ("vfa_flip_angles", "flip_angles"),
            ("vfa_tr", "tr"),
            ("brain_mask", "brain_mask"),
            ("field_strength", "field_strength"),
            ("csf_probseg", "csf_probseg"),
        ]),
        # Connect B1 map resampled to VFA space
        (transform_b1_to_vfa, t1_mapping, [
            ("out_file", "b1_map"),
        ]),
    ])

    # ==========================================================================
    # Node 3: DCE Preprocessing (motion correction + baseline detection)
    # ==========================================================================
    dce_preprocess = pe.Node(
        DCEPreprocess(
            baseline_frames=config.baseline_frames,
        ),
        name="dce_preprocess",
    )

    # Connect DCE preprocessing inputs
    workflow.connect([
        (inputnode, dce_preprocess, [
            ("dce_file", "in_file"),
            ("brain_mask", "brain_mask"),
        ]),
    ])

    # ==========================================================================
    # Node 4: DCE to T1w Registration
    # ==========================================================================
    dce_registration = pe.Node(
        DCERegistration(
            skip_bbr=config.skip_bbr,
            lesion_dilation=config.lesion_dilation,
            reference_type=config.reference,  # "postcon" or "precon"
        ),
        name="dce_registration",
    )

    # Connect registration inputs
    workflow.connect([
        (inputnode, dce_registration, [
            ("t1w_reference", "t1w_file"),
            ("lesion_mask", "lesion_mask"),
            ("brain_mask", "brain_mask"),
            ("wm_probseg", "wm_seg"),  # For BBR mode
        ]),
        (dce_preprocess, dce_registration, [
            ("mean_img", "dce_mean"),
        ]),
    ])

    # ==========================================================================
    # Node 4b: Transform brain mask to DCE space
    # ==========================================================================
    # The brain mask from anat is in T1w space. After registration gives us
    # the T1w→DCE transform, we resample the mask to DCE space.
    transform_brain_mask = pe.Node(
        niu.Function(
            input_names=[
                "mask_file",
                "reference_file",
                "transform_matrix",
            ],
            output_names=["out_file"],
            function=_transform_mask_to_dce,
        ),
        name="transform_brain_mask",
    )

    workflow.connect([
        (inputnode, transform_brain_mask, [
            ("brain_mask", "mask_file"),
        ]),
        (dce_preprocess, transform_brain_mask, [
            ("mean_img", "reference_file"),
        ]),
        (dce_registration, transform_brain_mask, [
            ("out_matrix", "transform_matrix"),
        ]),
    ])

    # ==========================================================================
    # Node 4b2: Create conservative AIF search mask (Gap 5)
    # ==========================================================================
    # Intersect T1w-transformed mask with DCE-native mask to exclude
    # non-brain regions that may be included in thin DCE slabs.
    create_aif_mask = pe.Node(
        niu.Function(
            input_names=[
                "t1w_mask_in_dce",
                "dce_mean",
            ],
            output_names=["out_file"],
            function=_create_aif_search_mask,
        ),
        name="create_aif_mask",
    )

    workflow.connect([
        (transform_brain_mask, create_aif_mask, [
            ("out_file", "t1w_mask_in_dce"),
        ]),
        (dce_preprocess, create_aif_mask, [
            ("mean_img", "dce_mean"),
        ]),
    ])

    # ==========================================================================
    # Node 4b3: Transform MNI atlas to DCE space (Gap 6) - optional
    # ==========================================================================
    # Atlas-guided AIF search (MCA/ACA territories) for improved AIF detection
    # Only runs if atlas_mask_file and t1w_to_mni_xfm are provided
    transform_atlas_to_dce = pe.Node(
        niu.Function(
            input_names=[
                "atlas_file",
                "t1w_to_mni_xfm",
                "t1w_to_dce_xfm",
                "reference_file",
            ],
            output_names=["out_file"],
            function=_transform_atlas_to_dce,
        ),
        name="transform_atlas_to_dce",
    )

    workflow.connect([
        (inputnode, transform_atlas_to_dce, [
            ("atlas_mask_file", "atlas_file"),
            ("t1w_to_mni_xfm", "t1w_to_mni_xfm"),
        ]),
        (dce_registration, transform_atlas_to_dce, [
            ("out_matrix", "t1w_to_dce_xfm"),
        ]),
        (dce_preprocess, transform_atlas_to_dce, [
            ("mean_img", "reference_file"),
        ]),
    ])

    # ==========================================================================
    # Node 4c: Transform T1 map to DCE space
    # ==========================================================================
    transform_t1_to_dce = pe.Node(
        niu.Function(
            input_names=["t1_map_file", "reference_file"],
            output_names=["out_file"],
            function=_transform_t1_to_dce,
        ),
        name="transform_t1_to_dce",
    )

    workflow.connect([
        (t1_mapping, transform_t1_to_dce, [
            ("out_file", "t1_map_file"),
        ]),
        (dce_preprocess, transform_t1_to_dce, [
            ("mean_img", "reference_file"),
        ]),
    ])

    # ==========================================================================
    # Node 4d: Transform B1+ map to DCE space (for signal_to_concentration)
    # ==========================================================================
    transform_b1_to_dce = pe.Node(
        niu.Function(
            input_names=["b1_map_file", "reference_file"],
            output_names=["out_file"],
            function=_transform_b1_to_dce,
        ),
        name="transform_b1_to_dce",
    )

    workflow.connect([
        (b1_mapping, transform_b1_to_dce, [
            ("out_file", "b1_map_file"),
        ]),
        (dce_preprocess, transform_b1_to_dce, [
            ("mean_img", "reference_file"),
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
                "t1_map_file",
                "b1_map_file",
                "flip_angle",
                "tr",
                "field_strength",
                "r1_relaxivity",
                "baseline_frames",
                "no_t1",
            ],
            output_names=["concentration_file"],
            function=_signal_to_concentration_node,
        ),
        name="signal_to_concentration",
    )

    # Pass config.no_t1 to the node
    signal_to_conc.inputs.no_t1 = config.no_t1

    # Use T1 and B1 maps resampled to DCE space
    workflow.connect([
        (dce_preprocess, signal_to_conc, [
            ("out_file", "dce_file"),
            ("baseline_frames", "baseline_frames"),
        ]),
        (transform_t1_to_dce, signal_to_conc, [
            ("out_file", "t1_map_file"),
        ]),
        (transform_b1_to_dce, signal_to_conc, [
            ("out_file", "b1_map_file"),
        ]),
        (inputnode, signal_to_conc, [
            ("flip_angle", "flip_angle"),
            ("tr", "tr"),
            ("field_strength", "field_strength"),
            ("r1_relaxivity", "r1_relaxivity"),
        ]),
    ])

    # ==========================================================================
    # Node 6: AIF Detection
    # ==========================================================================
    aif_detection = pe.Node(
        AIFDetection(
            aif_method=config.aif_method,
            scale_aif=config.scale_aif,
            n_candidates=10,  # Default
            lesion_distance=config.lesion_distance,
            prefer_contralateral=config.prefer_contralateral,
            baseline_frames=config.baseline_frames if config.baseline_frames > 0 else 5,
        ),
        name="aif_detection",
    )

    # Set target_peak if specified in config
    if config.target_peak is not None:
        aif_detection.inputs.target_peak = config.target_peak

    workflow.connect([
        (signal_to_conc, aif_detection, [
            ("concentration_file", "concentration_file"),
        ]),
        # Use conservative AIF search mask (Gap 5) instead of plain transformed mask
        (create_aif_mask, aif_detection, [
            ("out_file", "brain_mask_file"),
        ]),
        # Connect atlas mask for atlas-guided AIF search (Gap 6)
        (transform_atlas_to_dce, aif_detection, [
            ("out_file", "atlas_mask_file"),
        ]),
        (inputnode, aif_detection, [
            ("lesion_mask", "lesion_mask_file"),
            ("tr", "tr"),
            ("field_strength", "field_strength"),
        ]),
    ])

    # ==========================================================================
    # Node 7: PK Modeling (ComputeDCEMaps)
    # ==========================================================================
    pk_modeling = pe.Node(
        ComputeDCEMaps(
            model=config.model,
            bayesian=config.bayesian,
            mcmc_samples=config.mcmc_samples,
            tsnr_threshold=config.tsnr_threshold,
            use_gpu=config.use_gpu,
            save_uncertainty=config.save_uncertainty,
            constraint_mode=config.cbv_constraint_mode,
            hematocrit=config.hematocrit,
        ),
        name="pk_modeling",
    )

    workflow.connect([
        (signal_to_conc, pk_modeling, [
            ("concentration_file", "concentration_file"),
        ]),
        (aif_detection, pk_modeling, [
            ("out_aif_curve", "aif_file"),
        ]),
        (transform_brain_mask, pk_modeling, [
            ("out_file", "brain_mask_file"),
        ]),
        (inputnode, pk_modeling, [
            ("tr", "tr"),
            ("dsc_cbv_file", "dsc_cbv_file"),
            ("dsc_aif_qc_file", "dsc_aif_qc_file"),
        ]),
        # Use T1 map resampled to DCE space
        (transform_t1_to_dce, pk_modeling, [
            ("out_file", "t1_map_file"),
        ]),
    ])

    # ==========================================================================
    # Node 7b: Register DWI to DCE space (Gap 1 fix)
    # ==========================================================================
    # DWI must be registered to DCE space for proper voxel-wise tortuosity
    # correction. Chain: DWI → T1w (FLIRT 6-DOF) → DCE (via existing xfm)
    register_dwi_to_dce = pe.Node(
        niu.Function(
            input_names=[
                "dwi_file",
                "t1w_file",
                "t1w_to_dce_xfm",
                "dce_reference",
            ],
            output_names=["dwi_in_dce", "dwi_to_dce_xfm"],
            function=_register_dwi_to_dce,
        ),
        name="register_dwi_to_dce",
    )

    workflow.connect([
        (inputnode, register_dwi_to_dce, [
            ("dwi_file", "dwi_file"),
            ("t1w_reference", "t1w_file"),
        ]),
        (dce_registration, register_dwi_to_dce, [
            ("out_matrix", "t1w_to_dce_xfm"),
        ]),
        (dce_preprocess, register_dwi_to_dce, [
            ("mean_img", "dce_reference"),
        ]),
    ])

    # ==========================================================================
    # Node 8: Tortuosity Correction (optional - skip if config.no_tortuosity=True)
    # ==========================================================================
    # Tortuosity correction requires DWI data
    # Implemented as a conditional function node
    tortuosity_corr = pe.Node(
        niu.Function(
            input_names=[
                "ktrans_file",
                "dwi_file",
                "dwi_bval",
                "brain_mask",
                "no_tortuosity",
            ],
            output_names=["ktrans_corrected_file", "was_corrected", "qc_plot"],
            function=_tortuosity_correction_node,
        ),
        name="tortuosity_correction",
    )

    tortuosity_corr.inputs.no_tortuosity = config.no_tortuosity

    workflow.connect([
        (pk_modeling, tortuosity_corr, [
            ("out_ktrans", "ktrans_file"),
        ]),
        # Use registered DWI instead of raw DWI (Gap 1 fix)
        (register_dwi_to_dce, tortuosity_corr, [
            ("dwi_in_dce", "dwi_file"),
        ]),
        (inputnode, tortuosity_corr, [
            ("dwi_bval", "dwi_bval"),
        ]),
        (transform_brain_mask, tortuosity_corr, [
            ("out_file", "brain_mask"),
        ]),
    ])

    # ==========================================================================
    # Node 9: DSC-DCE Cross-Validation (optional - skip if config.no_dsc_validation=True)
    # ==========================================================================
    dsc_dce_validation = pe.Node(
        niu.Function(
            input_names=[
                "vp_file",
                "dsc_cbv_file",
                "brain_mask",
                "no_dsc_validation",
            ],
            output_names=["validation_result"],
            function=_dsc_dce_validation_node,
        ),
        name="dsc_dce_validation",
    )

    dsc_dce_validation.inputs.no_dsc_validation = config.no_dsc_validation

    workflow.connect([
        (pk_modeling, dsc_dce_validation, [
            ("out_vp", "vp_file"),
        ]),
        (inputnode, dsc_dce_validation, [
            ("dsc_cbv_file", "dsc_cbv_file"),
        ]),
        (transform_brain_mask, dsc_dce_validation, [
            ("out_file", "brain_mask"),
        ]),
    ])

    # ==========================================================================
    # Node 10: Collect QC Metrics
    # ==========================================================================
    collect_qc = pe.Node(
        niu.Function(
            input_names=[
                "b1_qc",
                "t1_qc",
                "preprocess_qc",
                "registration_qc",
                "aif_qc",
                "pk_qc",
                "validation_result",
            ],
            output_names=["qc_metrics"],
            function=_collect_qc_metrics,
        ),
        name="collect_qc_metrics",
    )

    workflow.connect([
        (b1_mapping, collect_qc, [
            ("qc_metrics", "b1_qc"),
        ]),
        (t1_mapping, collect_qc, [
            ("qc_metrics", "t1_qc"),
        ]),
        (dce_preprocess, collect_qc, [
            ("qc_metrics", "preprocess_qc"),
        ]),
        (dce_registration, collect_qc, [
            ("qc_metrics", "registration_qc"),
        ]),
        (aif_detection, collect_qc, [
            ("qc_metrics", "aif_qc"),
        ]),
        (pk_modeling, collect_qc, [
            ("qc_metrics", "pk_qc"),
        ]),
        (dsc_dce_validation, collect_qc, [
            ("validation_result", "validation_result"),
        ]),
    ])

    # ==========================================================================
    # Node 11: Lesion Statistics (conditional on lesion mask availability)
    # ==========================================================================
    # Transform lesion mask from T1w to DCE space
    transform_lesion = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dce_file"],
            function=_transform_lesion_mask_to_dce,
        ),
        name="transform_lesion",
    )

    workflow.connect([
        (inputnode, transform_lesion, [("lesion_mask", "lesion_mask_file")]),
        (dce_preprocess, transform_lesion, [("mean_img", "reference_file")]),
        (dce_registration, transform_lesion, [("out_matrix", "transform_matrix")]),
    ])

    # Lesion statistics extraction
    from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

    lesion_stats = pe.Node(
        DCELesionStats(),
        name="lesion_stats",
    )
    lesion_stats.inputs.subject_id = subject_id
    lesion_stats.inputs.session_id = session_id
    lesion_stats.inputs.output_dir = str(Path(bids_dir) / "derivatives" / "dce_pipeline")

    workflow.connect([
        (transform_lesion, lesion_stats, [("lesion_mask_dce_file", "lesion_mask")]),
        (pk_modeling, lesion_stats, [
            ("out_ktrans", "ktrans_map"),
            ("out_ve", "ve_map"),
            ("out_vp", "vp_map"),
            ("out_kep", "kep_map"),
        ]),
    ])

    # ==========================================================================
    # Node 12: Structural underlays (conditional on config.underlays)
    # ==========================================================================
    transform_t1w_underlay = None
    if "t1w" in config.underlays:
        transform_t1w_underlay = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dce,
            ),
            name="transform_t1w_underlay",
        )
        transform_t1w_underlay.inputs.output_name = "t1w_in_dce.nii.gz"

    transform_flair_underlay = None
    if "flair" in config.underlays:
        transform_flair_underlay = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dce,
            ),
            name="transform_flair_underlay",
        )
        transform_flair_underlay.inputs.output_name = "flair_in_dce.nii.gz"

    # Connect structural underlay transforms
    if transform_t1w_underlay is not None:
        workflow.connect([
            (inputnode, transform_t1w_underlay, [("t1w_reference", "structural_file")]),
            (dce_preprocess, transform_t1w_underlay, [("mean_img", "reference_file")]),
            (dce_registration, transform_t1w_underlay, [("out_matrix", "transform_matrix")]),
        ])

    if transform_flair_underlay is not None:
        workflow.connect([
            (inputnode, transform_flair_underlay, [("flair_in_t1w", "structural_file")]),
            (dce_preprocess, transform_flair_underlay, [("mean_img", "reference_file")]),
            (dce_registration, transform_flair_underlay, [("out_matrix", "transform_matrix")]),
        ])

    # ==========================================================================
    # Connect outputs
    # ==========================================================================
    workflow.connect([
        # Motion-corrected outputs
        (dce_preprocess, outputnode, [
            ("out_file", "dce_moco"),
            ("mean_img", "dce_mean"),
            ("out_params", "motion_params"),
        ]),
        # T1/B1 mapping outputs
        (b1_mapping, outputnode, [
            ("out_file", "b1_map"),
        ]),
        (t1_mapping, outputnode, [
            ("out_file", "t1_map"),
        ]),
        # Pharmacokinetic parameter maps
        (pk_modeling, outputnode, [
            ("out_ktrans", "ktrans_map"),
            ("out_ve", "ve_map"),
            ("out_vp", "vp_map"),
            ("out_kep", "kep_map"),
        ]),
        # Uncertainty maps (if Bayesian)
        (pk_modeling, outputnode, [
            ("out_ktrans_std", "ktrans_uncertainty"),
            ("out_ve_std", "ve_uncertainty"),
            ("out_vp_std", "vp_uncertainty"),
        ]),
        # AIF outputs
        (aif_detection, outputnode, [
            ("out_aif_curve", "aif_curve"),
            ("out_aif_mask", "aif_mask"),
        ]),
        # Registration outputs
        (dce_registration, outputnode, [
            ("out_matrix", "dce_to_t1w_xfm"),
        ]),
        # QC metrics
        (collect_qc, outputnode, [
            ("qc_metrics", "qc_metrics"),
        ]),
    ])

    # Structural underlay output connections (conditional)
    if transform_t1w_underlay is not None:
        workflow.connect([
            (transform_t1w_underlay, outputnode, [("out_file", "t1w_in_dce")]),
        ])

    if transform_flair_underlay is not None:
        workflow.connect([
            (transform_flair_underlay, outputnode, [("out_file", "flair_in_dce")]),
        ])

    # Lesion statistics output connections
    workflow.connect([
        (transform_lesion, outputnode, [("lesion_mask_dce_file", "lesion_mask_dce")]),
        (lesion_stats, outputnode, [
            ("stats_json", "lesion_stats_json"),
            ("stats_txt", "lesion_stats_txt"),
        ]),
    ])

    return workflow


# =============================================================================
# Helper Functions for Function Nodes
# =============================================================================


def _transform_mask_to_dce(
    mask_file=None,
    reference_file=None,
    transform_matrix=None,
):
    """Transform brain mask from T1w space to DCE space.

    Uses FSL FLIRT with nearest-neighbour interpolation to resample
    the brain mask into DCE space. Includes:
    - Same-space shortcut: skip transform if mask and ref share geometry (Gap 10)
    - Hole-filling: apply -thr 0.5 -bin -fillh after transform (Gap 4)

    Args:
        mask_file: Path to brain mask in T1w space (or None).
        reference_file: Path to DCE reference image for output geometry.
        transform_matrix: Path to T1w→DCE FLIRT matrix.

    Returns:
        Path to transformed brain mask in DCE space, or None if no mask.
    """
    if mask_file is None:
        return None

    from pathlib import Path
    import subprocess

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    mask_path = Path(mask_file)
    if not mask_path.exists():
        return None

    out_dir = Path.cwd()
    out_file = str(out_dir / "brain_mask_dce.nii.gz")

    # Same-space shortcut (Gap 10): skip transform if already aligned
    mask_img = nib.load(mask_file)
    ref_img = nib.load(reference_file)
    if mask_img.shape[:3] == ref_img.shape[:3]:
        affine_diff = float(np.abs(mask_img.affine - ref_img.affine).max())
        if affine_diff < 1e-4:
            # Already in same space, just copy with hole-filling
            mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]
            # Apply threshold and binarize
            mask_bin = (mask_data > 0.5).astype(np.int16)
            # Fill holes using scipy
            from scipy.ndimage import binary_fill_holes  # type: ignore[import-untyped]
            mask_filled = binary_fill_holes(mask_bin).astype(np.int16)
            # Save
            out_img = nib.Nifti1Image(mask_filled, ref_img.affine)
            nib.save(out_img, out_file)
            return out_file

    # Transform with FLIRT
    tmp_file = str(out_dir / "_tmp_brain_mask_dce.nii.gz")
    cmd = [
        "flirt",
        "-in", str(mask_file),
        "-ref", str(reference_file),
        "-out", tmp_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "nearestneighbour",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Hole-filling (Gap 4): threshold, binarize, fill holes
    # Using fslmaths: -thr 0.5 -bin -fillh
    cmd_fillh = [
        "fslmaths", tmp_file,
        "-thr", "0.5", "-bin", "-fillh", out_file,
    ]
    subprocess.run(cmd_fillh, check=True, capture_output=True)

    # Clean up temp file
    try:
        Path(tmp_file).unlink(missing_ok=True)
    except Exception:
        pass

    return out_file


def _transform_lesion_mask_to_dce(
    lesion_mask_file=None,
    reference_file=None,
    transform_matrix=None,
):
    """Transform lesion mask from T1w space to DCE space.

    Uses FSL FLIRT with nearest-neighbour interpolation to resample
    the lesion mask into DCE space, preserving integer label values.

    Args:
        lesion_mask_file: Path to lesion mask in T1w space (or None).
        reference_file: Path to DCE reference image for output geometry.
        transform_matrix: Path to T1w→DCE FLIRT matrix.

    Returns:
        Path to transformed lesion mask in DCE space, or None if no mask.
    """
    if lesion_mask_file is None:
        return None

    from pathlib import Path
    import subprocess

    lesion_path = Path(lesion_mask_file)
    if not lesion_path.exists():
        return None

    out_dir = Path.cwd()
    out_file = str(out_dir / "lesion_mask_dce.nii.gz")

    cmd = [
        "flirt",
        "-in", str(lesion_mask_file),
        "-ref", str(reference_file),
        "-out", out_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "nearestneighbour",  # Preserve integer labels
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    return out_file


def _create_aif_search_mask(
    t1w_mask_in_dce=None,
    dce_mean=None,
):
    """Create conservative AIF search mask (Gap 5).

    Creates an intersection of T1w-transformed mask with a DCE-native
    mask to exclude non-brain regions (e.g., temporal bone) that may
    be incorrectly included by HD-BET on thin DCE slabs.

    1. Create DCE-native mask using Otsu thresholding on DCE mean
    2. Intersect with T1w-transformed mask
    3. Fill holes

    Args:
        t1w_mask_in_dce: Path to T1w brain mask transformed to DCE space.
        dce_mean: Path to DCE mean image for native mask generation.

    Returns:
        Path to conservative AIF search mask, or None if inputs missing.
    """
    if t1w_mask_in_dce is None or dce_mean is None:
        return None

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np
    from scipy.ndimage import binary_fill_holes  # type: ignore[import-untyped]
    from skimage.filters import threshold_otsu  # type: ignore[import-untyped]

    t1w_mask_path = Path(t1w_mask_in_dce)
    if not t1w_mask_path.exists():
        return None

    # Load DCE mean
    dce_img = nib.load(dce_mean)
    dce_data = dce_img.get_fdata()  # type: ignore[attr-defined]

    # Create DCE-native mask using Otsu thresholding
    # Use only positive values for threshold calculation
    dce_positive = dce_data[dce_data > 0]
    if len(dce_positive) < 100:
        # Not enough data, just return T1w mask
        return str(t1w_mask_in_dce)

    try:
        thresh = threshold_otsu(dce_positive)
    except ValueError:
        # Otsu failed, use percentile
        thresh = np.percentile(dce_positive, 25)

    # Use 50% of Otsu threshold for more inclusive mask
    dce_mask = (dce_data > thresh * 0.5).astype(np.int16)

    # Load T1w mask
    t1w_mask_img = nib.load(t1w_mask_in_dce)
    t1w_mask_data = t1w_mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    # Intersect masks
    aif_mask = np.logical_and(t1w_mask_data, dce_mask).astype(np.int16)

    # Fill holes
    aif_mask = binary_fill_holes(aif_mask).astype(np.int16)

    # Save
    out_file = str(Path.cwd() / "aif_search_mask.nii.gz")
    out_img = nib.Nifti1Image(aif_mask, dce_img.affine)
    nib.save(out_img, out_file)

    return out_file


def _transform_atlas_to_dce(
    atlas_file=None,
    t1w_to_mni_xfm=None,
    t1w_to_dce_xfm=None,
    reference_file=None,
):
    """Transform MNI atlas mask to DCE space via T1w (Gap 6).

    Creates a concatenated transform MNI→T1w→DCE and applies it to the
    atlas mask. Includes morphological refinement (dilate-dilate-erode)
    to clean up edges after interpolation.

    Args:
        atlas_file: Path to MNI-space atlas mask (e.g., MCA/ACA territories).
        t1w_to_mni_xfm: Path to T1w→MNI FLIRT matrix (from anat pipeline).
        t1w_to_dce_xfm: Path to T1w→DCE FLIRT matrix (from DCE registration).
        reference_file: Path to DCE reference image for output geometry.

    Returns:
        Path to atlas mask in DCE space, or None if inputs missing.
    """
    if atlas_file is None or t1w_to_mni_xfm is None:
        return None
    if t1w_to_dce_xfm is None or reference_file is None:
        return None

    from pathlib import Path
    import subprocess

    atlas_path = Path(atlas_file)
    if not atlas_path.exists():
        return None

    out_dir = Path.cwd()

    # Step 1: Invert T1w→MNI to get MNI→T1w
    mni_to_t1w_xfm = str(out_dir / "_tmp_mni_to_t1w.mat")
    invert_cmd = [
        "convert_xfm", "-omat", mni_to_t1w_xfm,
        "-inverse", str(t1w_to_mni_xfm),
    ]
    result = subprocess.run(invert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    # Step 2: Concatenate MNI→T1w + T1w→DCE = MNI→DCE
    mni_to_dce_xfm = str(out_dir / "_tmp_mni_to_dce.mat")
    concat_cmd = [
        "convert_xfm", "-omat", mni_to_dce_xfm,
        "-concat", str(t1w_to_dce_xfm), mni_to_t1w_xfm,
    ]
    result = subprocess.run(concat_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up
        Path(mni_to_t1w_xfm).unlink(missing_ok=True)
        return None

    # Step 3: Apply transform with nearest-neighbour interpolation
    tmp_atlas = str(out_dir / "_tmp_atlas_dce.nii.gz")
    flirt_cmd = [
        "flirt",
        "-in", str(atlas_file),
        "-ref", str(reference_file),
        "-out", tmp_atlas,
        "-init", mni_to_dce_xfm,
        "-applyxfm",
        "-interp", "nearestneighbour",
    ]
    result = subprocess.run(flirt_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up
        Path(mni_to_t1w_xfm).unlink(missing_ok=True)
        Path(mni_to_dce_xfm).unlink(missing_ok=True)
        return None

    # Step 4: Morphological refinement - dilate-dilate-erode (1mm sphere)
    # This cleans up edges after interpolation artifacts
    out_file = str(out_dir / "atlas_mask_dce.nii.gz")
    morph_cmd = [
        "fslmaths", tmp_atlas,
        "-kernel", "sphere", "1",
        "-dilM", "-dilM", "-ero",
        "-bin", out_file,
    ]
    result = subprocess.run(morph_cmd, capture_output=True, text=True)

    # Clean up temp files
    Path(mni_to_t1w_xfm).unlink(missing_ok=True)
    Path(mni_to_dce_xfm).unlink(missing_ok=True)
    Path(tmp_atlas).unlink(missing_ok=True)

    if result.returncode != 0:
        return None

    return out_file


def _signal_to_concentration_node(
    dce_file,
    t1_map_file,
    b1_map_file,
    flip_angle,
    tr,
    field_strength,
    r1_relaxivity,
    baseline_frames,
    no_t1,
):
    """Function node for signal to concentration conversion.

    This function wraps the signal_to_concentration utility for use in
    a Nipype workflow.
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    from pmriprep.utils.dce_signal import signal_to_concentration

    # Load DCE data
    dce_img = nib.load(dce_file)
    dce_data = dce_img.get_fdata()  # type: ignore[attr-defined]
    affine = dce_img.affine  # type: ignore[attr-defined]
    header = dce_img.header  # type: ignore[attr-defined]

    # Load T1 map if not skipping T1
    t1_map = None
    if not no_t1 and t1_map_file:
        try:
            t1_img = nib.load(t1_map_file)
            t1_map = t1_img.get_fdata()  # type: ignore[attr-defined]
        except Exception as exc:
            import logging as _logging

            _logging.getLogger("pmriprep.workflows.dce").warning(
                "Failed to load T1 map (%s) — using default T1", exc,
            )

    # Load B1 map — clip to DCE-appropriate range [0.5, 1.5] (Gap 8).
    # Tighter than VFA T1 mapping range because signal-to-concentration
    # conversion is more sensitive to B1+ errors.
    b1_map = None
    if b1_map_file:
        try:
            b1_img = nib.load(b1_map_file)
            b1_map = b1_img.get_fdata()  # type: ignore[attr-defined]
            b1_map = np.clip(b1_map, 0.5, 1.5)
        except Exception as exc:
            import logging as _logging

            _logging.getLogger("pmriprep.workflows.dce").warning(
                "Failed to load B1 map (%s) — using no B1 correction", exc,
            )

    # Convert signal to concentration
    result = signal_to_concentration(
        signal=dce_data,
        flip_angle_deg=flip_angle,
        tr=tr,
        t1_map=t1_map,
        b1_map=b1_map,
        r1_relaxivity=r1_relaxivity,
        field_strength=field_strength,
        baseline_frames=baseline_frames if baseline_frames > 0 else 5,
        convert_to_plasma=True,
    )

    # Save concentration file
    out_file = str(Path.cwd() / "dce_concentration.nii.gz")
    conc_img = nib.Nifti1Image(result.concentration.astype(np.float32), affine, header)
    nib.save(conc_img, out_file)

    return out_file


def _tortuosity_correction_node(
    ktrans_file,
    dwi_file,
    dwi_bval,
    brain_mask,
    no_tortuosity,
):
    """Function node for tortuosity correction.

    Returns the (possibly corrected) Ktrans file, whether correction was applied,
    and path to QC plot.

    Includes:
    - Gap 3: B-value auto-detection from BIDS .bval sidecar
    - Gap 5: QC visualization (6-panel figure)
    - Gap 7: JSON sidecar with provenance metadata
    """
    import json
    import shutil
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    from pmriprep.utils.dce_tortuosity import (
        D_FREE_GD,
        calculate_and_apply_tortuosity_correction,
        generate_tortuosity_qc_plot,
    )

    # If skipping tortuosity or no DWI data, return original Ktrans
    if no_tortuosity or dwi_file is None:
        # Just copy the file to output directory for consistency
        out_file = str(Path.cwd() / "ktrans_final.nii.gz")
        shutil.copy(ktrans_file, out_file)
        return out_file, False, None

    # Gap 4: Auto-detect b-values from BIDS sidecar if not provided
    effective_bval = dwi_bval
    if effective_bval is None:
        bval_file = str(dwi_file).replace(".nii.gz", ".bval").replace(".nii", ".bval")
        if Path(bval_file).exists():
            try:
                bvals = np.loadtxt(bval_file)
                # Select b0 (<50) and high-b (800-1200 s/mm²)
                b0_idx = np.where(bvals < 50)[0]
                high_b_idx = np.where((bvals >= 800) & (bvals <= 1200))[0]
                if len(b0_idx) > 0 and len(high_b_idx) > 0:
                    effective_bval = float(bvals[high_b_idx[0]])
            except Exception:
                pass

    # If still no b-value, cannot proceed
    if effective_bval is None:
        out_file = str(Path.cwd() / "ktrans_final.nii.gz")
        shutil.copy(ktrans_file, out_file)
        return out_file, False, None

    try:
        # Load files
        ktrans_img = nib.load(ktrans_file)
        ktrans_data = ktrans_img.get_fdata()  # type: ignore[attr-defined]

        dwi_img = nib.load(dwi_file)
        dwi_data = dwi_img.get_fdata()  # type: ignore[attr-defined]

        mask_img = nib.load(brain_mask)
        mask_data = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

        # Create b_values array: [0, effective_bval]
        # Assume dwi_file is 4D with b0 and b>0
        b_values = np.array([0.0, effective_bval])

        # Apply tortuosity correction
        correction_result, tortuosity_result = calculate_and_apply_tortuosity_correction(
            ktrans=ktrans_data,
            dwi_signal=dwi_data,
            b_values=b_values,
            mask=mask_data,
        )

        # Save corrected Ktrans
        out_file = str(Path.cwd() / "ktrans_tortuosity_corrected.nii.gz")
        corrected_img = nib.Nifti1Image(
            correction_result.ktrans_corrected.astype(np.float32),
            ktrans_img.affine,  # type: ignore[attr-defined]
            ktrans_img.header,  # type: ignore[attr-defined]
        )
        nib.save(corrected_img, out_file)

        # Gap 5: Generate QC visualization
        qc_plot = None
        try:
            qc_plot_path = str(Path.cwd() / "tortuosity_qc.png")
            qc_plot = generate_tortuosity_qc_plot(
                adc_map=tortuosity_result.adc,
                tortuosity_map=tortuosity_result.tortuosity,
                ktrans_orig=correction_result.ktrans_original,
                ktrans_corr=correction_result.ktrans_corrected,
                mask=mask_data,
                out_file=qc_plot_path,
            )
        except Exception:
            pass  # QC plot is optional

        # Gap 7: Write JSON sidecar with provenance metadata
        sidecar = {
            "Description": "Ktrans corrected for tissue tortuosity using DWI-derived ADC",
            "Sources": [str(ktrans_file), str(dwi_file)],
            "TortuosityCorrection": True,
            "TortuosityMethod": "simple",
            "FreeDiffusionCoefficient": D_FREE_GD,
            "FreeDiffusionCoefficientUnits": "mm²/s",
            "BValue": effective_bval,
            "BValueUnits": "s/mm²",
            "Units": "min^-1",
        }
        # Add QC metrics if available
        if tortuosity_result.qc_metrics is not None:
            sidecar["QCMetrics"] = tortuosity_result.qc_metrics

        sidecar_path = out_file.replace(".nii.gz", ".json")
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        return out_file, correction_result.was_corrected, qc_plot

    except Exception:
        # If correction fails, return original Ktrans
        out_file = str(Path.cwd() / "ktrans_final.nii.gz")
        shutil.copy(ktrans_file, out_file)
        return out_file, False, None


def _dsc_dce_validation_node(
    vp_file,
    dsc_cbv_file,
    brain_mask,
    no_dsc_validation,
):
    """Function node for DSC-DCE cross-validation.

    Returns a dictionary with validation results.
    """
    from pmriprep.utils.dce_dsc_validation import validate_vp_cbv_from_files

    # If skipping validation or missing data, return empty result
    if no_dsc_validation or vp_file is None or dsc_cbv_file is None:
        return {
            "skipped": True,
            "status": "skipped",
            "reason": "Validation skipped or missing data",
        }

    try:
        result = validate_vp_cbv_from_files(
            vp_path=vp_file,
            cbv_path=dsc_cbv_file,
            mask_path=brain_mask,
        )
        return result.to_dict()
    except Exception as e:
        return {
            "skipped": True,
            "status": "error",
            "reason": str(e),
        }


def _collect_qc_metrics(
    b1_qc,
    t1_qc,
    preprocess_qc,
    registration_qc,
    aif_qc,
    pk_qc,
    validation_result,
):
    """Collect all QC metrics into a single dictionary."""
    qc_metrics = {}

    if b1_qc is not None:
        qc_metrics["b1_mapping"] = b1_qc
    if t1_qc is not None:
        qc_metrics["t1_mapping"] = t1_qc
    if preprocess_qc is not None:
        qc_metrics["preprocessing"] = preprocess_qc
    if registration_qc is not None:
        qc_metrics["registration"] = registration_qc
    if aif_qc is not None:
        qc_metrics["aif_detection"] = aif_qc
    if pk_qc is not None:
        qc_metrics["pk_modeling"] = pk_qc
    if validation_result is not None:
        qc_metrics["dsc_dce_validation"] = validation_result

    return qc_metrics


def _transform_b1_to_vfa(b1_map_file=None, vfa_ref_file=None):
    """Resample B1+ map from AFI space to VFA space.

    Uses nilearn continuous interpolation via NIfTI affine headers.
    Passes through None unchanged if B1 mapping was skipped.

    Args:
        b1_map_file: Path to B1+ map in AFI space (or None).
        vfa_ref_file: Path to VFA reference image for output geometry.

    Returns:
        Path to B1+ map resampled to VFA space, or original if same space.
    """
    if b1_map_file is None or vfa_ref_file is None:
        return b1_map_file

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    b1_img = nib.load(b1_map_file)
    ref_img = nib.load(vfa_ref_file)

    # Check if already in the same space (same shape + affine)
    if b1_img.shape[:3] == ref_img.shape[:3]:
        affine_diff = float(np.abs(b1_img.affine - ref_img.affine).max())
        if affine_diff < 1e-4:
            return b1_map_file

    from nilearn.image import resample_to_img  # type: ignore[import-untyped]

    resampled = resample_to_img(b1_img, ref_img, interpolation="continuous")
    out_file = str(Path.cwd() / "b1_map_vfa_space.nii.gz")
    nib.save(resampled, out_file)
    return out_file


def _transform_b1_to_dce(b1_map_file=None, reference_file=None):
    """Resample B1+ map to DCE space for signal-to-concentration.

    Uses nilearn continuous interpolation via NIfTI affine headers.

    Args:
        b1_map_file: Path to B1+ map (or None).
        reference_file: Path to DCE mean image for output geometry.

    Returns:
        Path to B1+ map resampled to DCE space, or original if same space.
    """
    if b1_map_file is None or reference_file is None:
        return b1_map_file

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    b1_img = nib.load(b1_map_file)
    ref_img = nib.load(reference_file)

    # Check if already in the same space
    if b1_img.shape[:3] == ref_img.shape[:3]:
        affine_diff = float(np.abs(b1_img.affine - ref_img.affine).max())
        if affine_diff < 1e-4:
            return b1_map_file

    from nilearn.image import resample_to_img  # type: ignore[import-untyped]

    resampled = resample_to_img(b1_img, ref_img, interpolation="continuous")
    out_file = str(Path.cwd() / "b1_map_dce_space.nii.gz")
    nib.save(resampled, out_file)
    return out_file


def _register_dwi_to_dce(
    dwi_file=None,
    t1w_file=None,
    t1w_to_dce_xfm=None,
    dce_reference=None,
):
    """Register DWI to DCE space via T1w (Gap 1 fix).

    Performs a two-step registration chain:
        DWI → T1w (FLIRT, 6-DOF, normmi) → DCE (apply existing xfm)

    This ensures the ADC map is properly aligned with Ktrans for
    voxel-wise tortuosity correction.

    Args:
        dwi_file: Path to DWI image (4D with b0 and b>0 volumes).
        t1w_file: Path to T1w reference image.
        t1w_to_dce_xfm: Path to T1w→DCE FLIRT transform matrix
            (from DCE registration step).
        dce_reference: Path to DCE reference image for output geometry.

    Returns:
        Tuple of (dwi_in_dce, dwi_to_dce_xfm):
            - dwi_in_dce: Path to DWI resampled to DCE space
            - dwi_to_dce_xfm: Path to concatenated transform matrix
    """
    if dwi_file is None or t1w_file is None:
        return None, None
    if t1w_to_dce_xfm is None or dce_reference is None:
        return None, None

    from pathlib import Path
    import subprocess

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    dwi_path = Path(dwi_file)
    if not dwi_path.exists():
        return None, None

    out_dir = Path.cwd()

    # For 4D DWI, extract b0 volume for registration
    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()  # type: ignore[attr-defined]

    if dwi_data.ndim == 4:
        # Use first volume (assumed b0) for registration
        b0_data = dwi_data[..., 0]
        b0_img = nib.Nifti1Image(b0_data, dwi_img.affine)
        b0_file = str(out_dir / "_tmp_dwi_b0.nii.gz")
        nib.save(b0_img, b0_file)
        reg_input = b0_file
    else:
        reg_input = str(dwi_file)

    # Step 1: DWI → T1w registration (6-DOF rigid, normalized mutual info)
    dwi_to_t1w_xfm = str(out_dir / "dwi_to_t1w.mat")
    flirt_cmd = [
        "flirt",
        "-in", reg_input,
        "-ref", str(t1w_file),
        "-omat", dwi_to_t1w_xfm,
        "-dof", "6",
        "-cost", "normmi",
        "-searchrx", "-30", "30",
        "-searchry", "-30", "30",
        "-searchrz", "-30", "30",
    ]
    result = subprocess.run(flirt_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up temp file
        if dwi_data.ndim == 4:
            Path(b0_file).unlink(missing_ok=True)
        return None, None

    # Step 2: Concatenate DWI→T1w + T1w→DCE = DWI→DCE
    dwi_to_dce_xfm = str(out_dir / "dwi_to_dce.mat")
    concat_cmd = [
        "convert_xfm",
        "-omat", dwi_to_dce_xfm,
        "-concat", str(t1w_to_dce_xfm), dwi_to_t1w_xfm,
    ]
    result = subprocess.run(concat_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up temp files
        if dwi_data.ndim == 4:
            Path(b0_file).unlink(missing_ok=True)
        Path(dwi_to_t1w_xfm).unlink(missing_ok=True)
        return None, None

    # Step 3: Apply transform to full 4D DWI (spline interpolation)
    dwi_in_dce = str(out_dir / "dwi_in_dce.nii.gz")
    apply_cmd = [
        "flirt",
        "-in", str(dwi_file),
        "-ref", str(dce_reference),
        "-out", dwi_in_dce,
        "-init", dwi_to_dce_xfm,
        "-applyxfm",
        "-interp", "spline",
    ]
    result = subprocess.run(apply_cmd, capture_output=True, text=True)

    # Clean up temp files
    if dwi_data.ndim == 4:
        Path(b0_file).unlink(missing_ok=True)
    Path(dwi_to_t1w_xfm).unlink(missing_ok=True)

    if result.returncode != 0:
        Path(dwi_to_dce_xfm).unlink(missing_ok=True)
        return None, None

    return dwi_in_dce, dwi_to_dce_xfm


def _transform_t1_to_dce(t1_map_file=None, reference_file=None):
    """Resample T1 map from VFA space to DCE space.

    Uses nilearn continuous interpolation via NIfTI affine headers.
    This is critical to avoid shape mismatches in signal_to_concentration
    and pk_modeling.

    Args:
        t1_map_file: Path to T1 map in VFA space (or None).
        reference_file: Path to DCE mean image for output geometry.

    Returns:
        Path to T1 map resampled to DCE space, or original if same space.
    """
    if t1_map_file is None or reference_file is None:
        return t1_map_file

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    t1_img = nib.load(t1_map_file)
    ref_img = nib.load(reference_file)

    # Check if already in the same space
    if t1_img.shape[:3] == ref_img.shape[:3]:
        affine_diff = float(np.abs(t1_img.affine - ref_img.affine).max())
        if affine_diff < 1e-4:
            return t1_map_file

    from nilearn.image import resample_to_img  # type: ignore[import-untyped]

    resampled = resample_to_img(t1_img, ref_img, interpolation="continuous")
    out_file = str(Path.cwd() / "t1_map_dce_space.nii.gz")
    nib.save(resampled, out_file)
    return out_file


def _transform_structural_to_dce(
    structural_file=None,
    reference_file=None,
    transform_matrix=None,
    output_name="structural_in_dce.nii.gz",
):
    """Transform structural image from T1w space to DCE space via FLIRT.

    Args:
        structural_file: Path to structural image in T1w space (or None).
        reference_file: Path to DCE reference image for output geometry.
        transform_matrix: Path to T1w→DCE FLIRT matrix.
        output_name: Output filename.

    Returns:
        Path to transformed structural image in DCE space, or None if inputs missing.
    """
    if structural_file is None or reference_file is None or transform_matrix is None:
        return None

    from pathlib import Path
    import subprocess

    if not Path(structural_file).exists():
        return None

    out_file = str(Path.cwd() / output_name)
    subprocess.run(
        [
            "flirt",
            "-in", str(structural_file),
            "-ref", str(reference_file),
            "-out", out_file,
            "-init", str(transform_matrix),
            "-applyxfm",
            "-interp", "spline",
        ],
        check=True,
        capture_output=True,
    )
    return out_file
