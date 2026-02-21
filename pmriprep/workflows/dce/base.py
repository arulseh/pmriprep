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
from pmriprep.data import get_arterial_atlas, get_arterial_atlas_cascade


def init_dce_preproc_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[DCEConfig] = None,
    has_tissue_seg: bool = True,
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
    has_tissue_seg : bool
        Whether tissue segmentation maps (CSF/WM probseg) are available.
        False in postcon-only mode where FAST is skipped. Default: True.
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
        Path to DSC-derived CBV map for vp constraint (DSC native space).
        Will be resampled to DCE space via dsc_to_t1w_xfm + t1w_to_dce_xfm.
    dsc_to_t1w_xfm : str, optional
        Path to DSC→T1w FLIRT transform matrix (for CBV resampling).
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
    ve_map : str, optional
        Path to ve (extravascular extracellular volume fraction) map.
        Not produced by Patlak model.
    vp_map : str, optional
        Path to vp (plasma volume fraction) map (extended Tofts/2CXM).
    kep_map : str, optional
        Path to kep (rate constant) map. Not produced by Patlak model.
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
                # Tumor mask inputs for lesion statistics
                "whole_tumor_mask",  # Undilated whole-tumor mask (fallback for lesion stats)
                "enhancing_mask",  # Enhancing tumor mask for QC voxel selection (optional)
                "hdglio_segmentation",  # HD-GLIO multi-label segmentation for lesion stats (optional)
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
                "afi_flip_angle",
                # DSC integration inputs
                "dsc_cbv_file",
                "dsc_to_t1w_xfm",
                "dsc_aif_qc_file",
                # DWI tortuosity inputs
                "dwi_file",
                "dwi_bval",
                # Acquisition parameters
                "flip_angle",
                "tr",  # Pulse TR for SPGR signal equation
                "volume_tr",  # Volume TR for model auto-selection (temporal resolution)
                "te",
                "field_strength",
                "r1_relaxivity",
                "agent_r1_known",  # True when r1 is from confirmed agent lookup
                "agent_standard_dose",  # Standard clinical dose (mmol/kg) for this agent
                # Registration transforms (from anat pipeline)
                "t1w_to_mni_xfm",  # T1w to MNI transform for atlas-guided AIF
                # Atlas inputs (optional)
                "atlas_mask_file",  # MNI-space atlas mask for AIF search territories
                # Structural underlay inputs
                "flair_in_t1w",  # FLAIR coregistered to T1w space (from anat)
                # NAWM mask input (from hdglio, for normalization)
                "nawm_mask",  # NAWM mask in T1w space for nKtrans, nIAUC60
            ]
        ),
        name="inputnode",
    )

    # Default DSC inputs to None (overridden when DSC workflow is connected)
    inputnode.inputs.dsc_cbv_file = None
    inputnode.inputs.dsc_to_t1w_xfm = None

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
                # Model-free maps
                "iauc60_map",
                "iauc90_map",
                "iauc120_map",
                "peak_enhancement_map",
                "time_to_peak_map",
                "ser_map",
                # QC maps
                "tsnr_map",
                "qc_mask",
                "r2_map",
                "quality_flag_map",
                # AIF outputs
                "aif_curve",
                "aif_mask",
                # Registration outputs
                "dce_to_t1w_xfm",
                # Uncertainty outputs (Bayesian)
                "ktrans_uncertainty",
                "ve_uncertainty",
                "vp_uncertainty",
                "fp_uncertainty",
                "ps_uncertainty",
                # Bayesian CV and CI outputs
                "ktrans_cv",
                "ve_cv",
                "vp_cv",
                "ktrans_ci95_lower",
                "ktrans_ci95_upper",
                "ve_ci95_lower",
                "ve_ci95_upper",
                "vp_ci95_lower",
                "vp_ci95_upper",
                # Tortuosity-corrected output
                "ktrans_tortuosity_corrected",
                # Smoothed maps (populated only if config.smooth_fwhm > 0)
                "smooth_pk_maps",
                "smooth_model_free_maps",
                "smooth_uncertainty_maps",
                "smooth_tortuosity_maps",
                # Normalized maps (requires NAWM mask)
                "nktrans_map",
                "niauc60_map",
                # QC metrics
                "qc_metrics",
                # QC plots
                "registration_qc_filmstrip",
                "aif_curve_qc_plot",
                "aif_location_qc_plot",
                "vfa_registration_qc_plot",  # optional, only if VFA coregistration occurred
                "afi_registration_qc_plot",  # optional, only if AFI coregistration occurred
                "b1_qc_plot",
                "t1_qc_plot",
                # Structural underlays
                "t1w_in_dce",
                "flair_in_dce",
                # Lesion statistics (conditional on lesion mask)
                "lesion_stats_json",
                "lesion_stats_txt",
                "lesion_mask_dce",
                # Motion QC
                "motion_qc_plot",
                # DCE QC plots
                "preprocessing_qc_plot",
                "pk_modeling_qc_plot",
                "pk_param_maps_qc_plot",
                # Anatomical masks in DCE space
                "brain_mask",
                "nawm_mask",
            ]
        ),
        name="outputnode",
    )

    # ==========================================================================
    # Node 0: Reorient all raw BIDS inputs to RAS orientation
    # ==========================================================================
    # DCE timeseries (4D) - required input
    dce_reorient = pe.Node(
        Reorient(),
        name="dce_reorient",
    )

    workflow.connect([
        (inputnode, dce_reorient, [("dce_file", "in_file")]),
    ])

    # NOTE: AFI and DWI reorientation is handled inside their respective
    # processing functions (B1Mapping, _register_dwi_to_dce) to avoid
    # issues with nipype passing None to interfaces that expect files.

    # ==========================================================================
    # Nodes 1-2: VFA/B1/T1 pipeline (conditional — skipped when no_t1=True)
    # ==========================================================================
    # When no VFA data is available (no_t1=True), the entire VFA→B1→T1 chain
    # is skipped. signal_to_concentration uses field-strength default T1 and
    # nominal flip angles, producing semi-quantitative results per QIBA
    # DCE-MRI profile (Tofts et al., 1999; QIBA v2.0, 2019).
    if not config.no_t1:
        # VFA images (list of 3D) - reorient to RAS
        vfa_reorient = pe.Node(
            niu.Function(
                input_names=["in_files"],
                output_names=["out_file"],
                function=reorient_list_to_ras,
            ),
            name="vfa_reorient",
        )

        workflow.connect([
            (inputnode, vfa_reorient, [("vfa_files", "in_files")]),
        ])

        # Node 1: B1+ Mapping (optional — skip_b1 disables correction)
        b1_mapping = pe.Node(
            B1Mapping(
                skip_b1=config.no_b1,
            ),
            name="b1_mapping",
        )

        workflow.connect([
            (inputnode, b1_mapping, [
                ("afi_tr1_file", "tr1_file"),
                ("afi_tr2_file", "tr2_file"),
                ("afi_4d_file", "afi_4d_file"),
                ("afi_tr1", "tr1"),
                ("afi_tr2", "tr2"),
                ("brain_mask", "brain_mask"),
                ("field_strength", "field_strength"),
                ("afi_flip_angle", "nominal_flip_angle"),
            ]),
        ])

        b1_mapping.inputs.subject_id = subject_id
        b1_mapping.inputs.session_id = session_id or ""

        # Node 1b: VFA Preprocessing (coregistration + quality assessment)
        vfa_preprocess = pe.Node(
            VFAPreprocess(
                skip_coregistration=False,
                skip_flip_validation=config.skip_flip_validation,
            ),
            name="vfa_preprocess",
        )

        workflow.connect([
            (vfa_reorient, vfa_preprocess, [("out_file", "vfa_files")]),
            (inputnode, vfa_preprocess, [
                ("vfa_flip_angles", "flip_angles"),
                ("vfa_tr", "tr"),
                ("field_strength", "field_strength"),
            ]),
        ])

        # Node 1c: Transform B1+ map to VFA space (for T1 fitting)
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

        # Node 2: T1 Mapping
        t1_mapping = pe.Node(
            T1Mapping(
                skip_t1=False,
                lenient_qc=config.lenient_t1_qc,
                t1_correction_mode=config.t1_correction,
            ),
            name="t1_mapping",
        )

        workflow.connect([
            (vfa_preprocess, t1_mapping, [
                ("out_files", "vfa_files"),
            ]),
            (inputnode, t1_mapping, [
                ("vfa_flip_angles", "flip_angles"),
                ("vfa_tr", "tr"),
                ("brain_mask", "brain_mask"),
                ("field_strength", "field_strength"),
            ]),
            (transform_b1_to_vfa, t1_mapping, [
                ("out_file", "b1_map"),
            ]),
            (b1_mapping, t1_mapping, [
                ("qc_pass", "b1_qc_pass"),
            ]),
        ])

        t1_mapping.inputs.subject_id = subject_id
        t1_mapping.inputs.session_id = session_id or ""

    # ==========================================================================
    # Node 3: DCE Preprocessing (motion correction + baseline detection)
    # ==========================================================================
    dce_preprocess = pe.Node(
        DCEPreprocess(
            baseline_frames=config.baseline_frames,
        ),
        name="dce_preprocess",
    )

    # Connect DCE preprocessing inputs (using reoriented DCE file)
    workflow.connect([
        (dce_reorient, dce_preprocess, [("out_file", "in_file")]),
        (inputnode, dce_preprocess, [("brain_mask", "brain_mask")]),
    ])

    # ==========================================================================
    # Node 3b: Motion QC Plot
    # Generate motion QC visualization from motion parameters.
    # ==========================================================================
    motion_qc_plot = pe.Node(
        niu.Function(
            input_names=["motion_params_file", "out_dir", "tr"],
            output_names=["plot_file"],
            function=_generate_dce_motion_qc_plot,
        ),
        name="motion_qc_plot",
    )
    motion_qc_plot.inputs.out_dir = str(Path(bids_dir) / "derivatives" / "dce_pipeline")

    # Connect motion QC plot
    workflow.connect([
        (dce_preprocess, motion_qc_plot, [("out_params", "motion_params_file")]),
        (inputnode, motion_qc_plot, [("volume_tr", "tr")]),
    ])

    # ==========================================================================
    # Node 3c: DCE Brain Extraction (native space)
    # ==========================================================================
    # Extract brain from DCE mean image using HD-BET with BET fallback.
    # This provides a native DCE-space brain mask that:
    # 1. Accurately reflects the DCE field of view
    # 2. Avoids registration-induced edge artifacts
    # 3. Matches perf-plus behavior (lines 360-382)
    dce_brain_extract = pe.Node(
        DCEBrainExtract(
            method="hdbet",  # Prefer HD-BET, fallback to BET
            bet_frac=0.3,    # Lower threshold for DCE contrast
            fill_holes=True,
            hdbet_device=config.hdbet_device,
        ),
        name="dce_brain_extract",
    )

    workflow.connect([
        (dce_preprocess, dce_brain_extract, [("baseline_mean_img", "dce_mean")]),
    ])

    # ==========================================================================
    # Node 3d: DCE Preprocessing QC Plot
    # ==========================================================================
    preprocessing_qc = pe.Node(
        niu.Function(
            input_names=[
                "dce_file", "brain_mask_file",
                "baseline_frames", "bolus_arrival_frame", "tr_ms",
                "volume_tr_s", "subject_id", "session_id",
            ],
            output_names=["plot_file", "preproc_display_stats"],
            function=_generate_preprocessing_qc,
        ),
        name="preprocessing_qc",
    )
    preprocessing_qc.inputs.subject_id = subject_id
    preprocessing_qc.inputs.session_id = session_id or ""

    workflow.connect([
        (dce_preprocess, preprocessing_qc, [
            ("out_file", "dce_file"),
            ("baseline_frames", "baseline_frames"),
            ("bolus_arrival_frame", "bolus_arrival_frame"),
        ]),
        (dce_brain_extract, preprocessing_qc, [
            ("out_mask", "brain_mask_file"),
        ]),
        (inputnode, preprocessing_qc, [
            ("tr", "tr_ms"),
            ("volume_tr", "volume_tr_s"),
        ]),
    ])

    # ==========================================================================
    # Node 4: DCE to T1w Registration
    # ==========================================================================
    dce_registration = pe.Node(
        DCERegistration(
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
        ]),
        (dce_brain_extract, dce_registration, [
            ("out_brain", "dce_mean"),  # Brain-extracted baseline mean
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
    # Node 4b1b: Transform CSF probseg to DCE space
    # ==========================================================================
    # CSF probability map needs trilinear interpolation (continuous values)
    # and clipping to [0, 1] instead of binarization + hole-filling.
    # Skip entirely in postcon-only mode where FAST segmentation is unavailable.
    if has_tissue_seg:
        transform_csf_probseg = pe.Node(
            niu.Function(
                input_names=["probseg_file", "reference_file", "transform_matrix"],
                output_names=["out_file"],
                function=_transform_probseg_to_dce,
            ),
            name="transform_csf_probseg",
        )

        workflow.connect([
            (inputnode, transform_csf_probseg, [("csf_probseg", "probseg_file")]),
            (dce_preprocess, transform_csf_probseg, [("mean_img", "reference_file")]),
            (dce_registration, transform_csf_probseg, [("out_matrix", "transform_matrix")]),
        ])

        # Wire transformed CSF probseg to T1 mapping only when active
        if not config.no_t1:
            workflow.connect([
                (transform_csf_probseg, t1_mapping, [("out_file", "csf_probseg")]),
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
        (transform_brain_mask, create_aif_mask, [
            ("out_file", "t1w_mask_in_dce"),
        ]),
        # Use native DCE brain mask from HD-BET extraction
        (dce_brain_extract, create_aif_mask, [
            ("out_mask", "dce_native_mask"),
        ]),
    ])

    # ==========================================================================
    # Node 4b3: Transform MNI atlas cascade to DCE space (Gap 6) - optional
    # ==========================================================================
    # Atlas-guided AIF search with tiered cascade (MCA_ACA → CoW → full)
    # for reduced tissue contamination compared to a single broad mask.
    # Only runs if atlas files and t1w_to_mni_xfm are provided.
    transform_atlas_cascade = pe.Node(
        niu.Function(
            input_names=[
                "atlas_files",
                "t1w_to_mni_xfm",
                "t1w_to_dce_xfm",
                "reference_file",
            ],
            output_names=["out_files"],
            function=_transform_atlas_cascade_to_dce,
        ),
        name="transform_atlas_cascade",
    )

    # Load tiered arterial atlas cascade (tightest → broadest)
    atlas_cascade = get_arterial_atlas_cascade("2mm")
    if atlas_cascade:
        transform_atlas_cascade.inputs.atlas_files = [str(p) for p in atlas_cascade]
    else:
        import warnings
        warnings.warn(
            "No arterial atlases found. AIF detection will use whole-brain search."
        )

    workflow.connect([
        (inputnode, transform_atlas_cascade, [
            ("t1w_to_mni_xfm", "t1w_to_mni_xfm"),
        ]),
        (dce_registration, transform_atlas_cascade, [
            ("out_matrix", "t1w_to_dce_xfm"),
        ]),
        (dce_preprocess, transform_atlas_cascade, [
            ("mean_img", "reference_file"),
        ]),
    ])

    # ==========================================================================
    # Nodes 4c-4d: Transform T1/B1 maps to DCE space (conditional)
    # ==========================================================================
    if not config.no_t1:
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
        (inputnode, signal_to_conc, [
            ("flip_angle", "flip_angle"),
            ("tr", "tr"),
            ("field_strength", "field_strength"),
            ("r1_relaxivity", "r1_relaxivity"),
        ]),
    ])

    if not config.no_t1:
        workflow.connect([
            (transform_t1_to_dce, signal_to_conc, [
                ("out_file", "t1_map_file"),
            ]),
            (transform_b1_to_dce, signal_to_conc, [
                ("out_file", "b1_map_file"),
            ]),
        ])
    else:
        signal_to_conc.inputs.t1_map_file = None
        signal_to_conc.inputs.b1_map_file = None

    # ==========================================================================
    # Node 6: AIF Detection
    # ==========================================================================
    aif_detection = pe.Node(
        AIFDetection(
            aif_method=config.aif_method,
            scale_aif=config.scale_aif,
            n_candidates=config.n_aif_candidates,
            lesion_distance=config.lesion_distance,
            baseline_frames=config.baseline_frames,
            dose=config.dose,
        ),
        name="aif_detection",
    )

    # Set target_peak if specified in config
    if config.target_peak is not None:
        aif_detection.inputs.target_peak = config.target_peak

    # Wire user-supplied AIF mask if provided (overrides atlas mask)
    if config.aif_mask_file is not None:
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
        (transform_atlas_cascade, aif_detection, [
            ("out_files", "atlas_mask_files"),
        ]),
        (inputnode, aif_detection, [
            ("volume_tr", "tr"),  # Use volume TR for time array, not pulse TR
            ("field_strength", "field_strength"),
            ("r1_relaxivity", "r1_relaxivity"),
            ("agent_r1_known", "agent_r1_known"),
            ("agent_standard_dose", "agent_standard_dose"),
        ]),
        # Wire auto-detected baseline_frames from dce_preprocess
        (dce_preprocess, aif_detection, [
            ("baseline_frames", "baseline_frames"),
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
            constraint_mode=config.cbv_constraint_mode,
            hematocrit=config.hematocrit,
            n_procs=max(1, config.num_processes),
        ),
        name="pk_modeling",
    )

    workflow.connect([
        (signal_to_conc, pk_modeling, [
            ("concentration_file", "concentration_file"),
        ]),
        (aif_detection, pk_modeling, [
            ("out_aif_curve", "aif_file"),
            ("detection_method", "aif_detection_method"),
        ]),
        # Use native DCE brain mask (from HD-BET) for PK modeling
        # This avoids registration-induced edge artifacts and ensures
        # the mask accurately reflects the DCE field of view
        (dce_brain_extract, pk_modeling, [
            ("out_mask", "brain_mask_file"),
        ]),
        (inputnode, pk_modeling, [
            ("tr", "tr"),
            ("volume_tr", "volume_tr"),  # For model auto-selection
            ("enhancing_mask", "enhancing_mask_file"),
        ]),
        # Raw DCE signal for tSNR computation (matches perf-plus)
        # Also wire baseline_frames so pk_modeling uses the same value
        # as signal_to_concentration (avoids re-auto-detection mismatch)
        (dce_preprocess, pk_modeling, [
            ("out_file", "dce_signal_file"),
            ("baseline_frames", "baseline_frames"),
            ("fd", "fd_values"),
            ("bolus_arrival_frame", "bolus_arrival_frame"),
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

    # Connect DWI registration inputs (reorientation handled inside _register_dwi_to_dce)
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
    # Node 7c: Resample DSC CBV to DCE space (for vp constraint)
    # ==========================================================================
    # DSC CBV (128x128x19, 1.88mm) is in DSC native space, but DCE processing
    # operates in DCE native space (256x256x26, 0.94mm). Resample DSC CBV
    # via: DSC → T1w (existing xfm) → DCE (existing xfm), matching perf-plus.
    resample_dsc_cbv = pe.Node(
        niu.Function(
            input_names=[
                "dsc_cbv_file",
                "dsc_to_t1w_xfm",
                "t1w_to_dce_xfm",
                "dce_reference",
            ],
            output_names=["cbv_in_dce"],
            function=_resample_dsc_cbv_to_dce,
        ),
        name="resample_dsc_cbv",
    )

    workflow.connect([
        (inputnode, resample_dsc_cbv, [
            ("dsc_cbv_file", "dsc_cbv_file"),
            ("dsc_to_t1w_xfm", "dsc_to_t1w_xfm"),
        ]),
        (dce_registration, resample_dsc_cbv, [
            ("out_matrix", "t1w_to_dce_xfm"),
        ]),
        (dce_preprocess, resample_dsc_cbv, [
            ("mean_img", "dce_reference"),
        ]),
        # Feed resampled CBV to PK modeling (vp constraint)
        (resample_dsc_cbv, pk_modeling, [
            ("cbv_in_dce", "dsc_cbv_file"),
        ]),
    ])
    # Note: resample_dsc_cbv → dsc_dce_validation connected after Node 9 creation

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
        # Use native DCE brain mask for tortuosity correction
        (dce_brain_extract, tortuosity_corr, [
            ("out_mask", "brain_mask"),
        ]),
    ])

    # ==========================================================================
    # Node 8b: Transform NAWM mask to DCE space (for normalization)
    # ==========================================================================
    transform_nawm_mask = pe.Node(
        niu.Function(
            input_names=["mask_file", "reference_file", "transform_matrix"],
            output_names=["out_file"],
            function=_transform_mask_to_dce,
        ),
        name="transform_nawm_mask",
    )

    workflow.connect([
        (inputnode, transform_nawm_mask, [("nawm_mask", "mask_file")]),
        (dce_preprocess, transform_nawm_mask, [("mean_img", "reference_file")]),
        (dce_registration, transform_nawm_mask, [("out_matrix", "transform_matrix")]),
    ])

    # ==========================================================================
    # Node 8c: NAWM Normalization (nKtrans, nIAUC60)
    # ==========================================================================
    normalize_ktrans = pe.Node(
        niu.Function(
            input_names=["param_map", "nawm_mask"],
            output_names=["out_file"],
            function=_normalize_by_nawm,
        ),
        name="normalize_ktrans",
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
        (pk_modeling, normalize_ktrans, [("out_ktrans", "param_map")]),
        (transform_nawm_mask, normalize_ktrans, [("out_file", "nawm_mask")]),
        (pk_modeling, normalize_iauc60, [("out_iauc60", "param_map")]),
        (transform_nawm_mask, normalize_iauc60, [("out_file", "nawm_mask")]),
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
    # dsc_cbv_file is provided by resample_dsc_cbv node (None when DSC unavailable)

    workflow.connect([
        (pk_modeling, dsc_dce_validation, [
            ("out_vp", "vp_file"),
        ]),
        (transform_brain_mask, dsc_dce_validation, [
            ("out_file", "brain_mask"),
        ]),
        # Feed resampled DSC CBV (deferred from Node 7c)
        (resample_dsc_cbv, dsc_dce_validation, [
            ("cbv_in_dce", "dsc_cbv_file"),
        ]),
    ])

    # ==========================================================================
    # Node 9b: Smoothing (conditional on config.smooth_fwhm > 0)
    # ==========================================================================
    smooth_pk = None
    smooth_model_free = None
    smooth_uncertainty = None
    smooth_tortuosity = None

    if config.smooth_fwhm > 0:
        # Build dict of PK maps for smoothing
        build_pk_dict = pe.Node(
            niu.Function(
                input_names=["ktrans", "ve", "vp", "kep"],
                output_names=["map_dict"],
                function=_build_pk_map_dict,
            ),
            name="build_pk_dict",
        )

        workflow.connect([
            (pk_modeling, build_pk_dict, [
                ("out_ktrans", "ktrans"),
                ("out_ve", "ve"),
                ("out_vp", "vp"),
                ("out_kep", "kep"),
            ]),
        ])

        # Smoothing for PK maps
        smooth_pk = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_dce_maps,
            ),
            name="smooth_pk_maps",
        )
        smooth_pk.inputs.fwhm = config.smooth_fwhm

        workflow.connect([
            (build_pk_dict, smooth_pk, [("map_dict", "map_files")]),
            (dce_brain_extract, smooth_pk, [("out_mask", "brain_mask_file")]),
        ])

        # Build dict of model-free maps for smoothing
        build_mf_dict = pe.Node(
            niu.Function(
                input_names=["iauc60", "iauc90", "iauc120", "peak", "ttp", "ser"],
                output_names=["map_dict"],
                function=_build_model_free_map_dict,
            ),
            name="build_model_free_dict",
        )

        workflow.connect([
            (pk_modeling, build_mf_dict, [
                ("out_iauc60", "iauc60"),
                ("out_iauc90", "iauc90"),
                ("out_iauc120", "iauc120"),
                ("out_peak_enhancement", "peak"),
                ("out_time_to_peak", "ttp"),
                ("out_ser", "ser"),
            ]),
        ])

        # Smoothing for model-free maps
        smooth_model_free = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_dce_maps,
            ),
            name="smooth_model_free_maps",
        )
        smooth_model_free.inputs.fwhm = config.smooth_fwhm

        workflow.connect([
            (build_mf_dict, smooth_model_free, [("map_dict", "map_files")]),
            (dce_brain_extract, smooth_model_free, [("out_mask", "brain_mask_file")]),
        ])

        # Build dict of uncertainty std maps for smoothing (Bayesian outputs)
        # Only std maps are smoothed; CV and CI95 are excluded (see docstring)
        build_uncertainty_dict = pe.Node(
            niu.Function(
                input_names=[
                    "ktrans_std", "ve_std", "vp_std",
                    "fp_std", "ps_std",
                ],
                output_names=["map_dict"],
                function=_build_uncertainty_map_dict,
            ),
            name="build_uncertainty_dict",
        )

        workflow.connect([
            (pk_modeling, build_uncertainty_dict, [
                ("out_ktrans_std", "ktrans_std"),
                ("out_ve_std", "ve_std"),
                ("out_vp_std", "vp_std"),
                ("out_fp_std", "fp_std"),
                ("out_ps_std", "ps_std"),
            ]),
        ])

        # Smoothing for uncertainty maps
        smooth_uncertainty = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_dce_maps,
            ),
            name="smooth_uncertainty_maps",
        )
        smooth_uncertainty.inputs.fwhm = config.smooth_fwhm

        workflow.connect([
            (build_uncertainty_dict, smooth_uncertainty, [("map_dict", "map_files")]),
            (dce_brain_extract, smooth_uncertainty, [("out_mask", "brain_mask_file")]),
        ])

        # Build dict of tortuosity-corrected Ktrans for smoothing
        build_tortuosity_dict = pe.Node(
            niu.Function(
                input_names=["ktrans_tortuosity_corrected"],
                output_names=["map_dict"],
                function=_build_tortuosity_map_dict,
            ),
            name="build_tortuosity_dict",
        )

        workflow.connect([
            (tortuosity_corr, build_tortuosity_dict, [
                ("ktrans_corrected_file", "ktrans_tortuosity_corrected"),
            ]),
        ])

        # Smoothing for tortuosity-corrected Ktrans
        smooth_tortuosity = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_dce_maps,
            ),
            name="smooth_tortuosity_maps",
        )
        smooth_tortuosity.inputs.fwhm = config.smooth_fwhm

        workflow.connect([
            (build_tortuosity_dict, smooth_tortuosity, [("map_dict", "map_files")]),
            (dce_brain_extract, smooth_tortuosity, [("out_mask", "brain_mask_file")]),
        ])

    # ==========================================================================
    # Node 10: Registration QC Filmstrip
    # ==========================================================================
    reg_filmstrip = pe.Node(
        niu.Function(
            input_names=["dce_mean_file", "registered_t1w_file", "brain_mask_file",
                         "subject_id", "session_id"],
            output_names=["filmstrip_file"],
            function=_create_dce_registration_filmstrip,
        ),
        name="reg_filmstrip",
    )
    reg_filmstrip.inputs.subject_id = subject_id
    reg_filmstrip.inputs.session_id = session_id or ""

    workflow.connect([
        (dce_preprocess, reg_filmstrip, [("mean_img", "dce_mean_file")]),
        (dce_registration, reg_filmstrip, [("out_file", "registered_t1w_file")]),
        (dce_brain_extract, reg_filmstrip, [("out_mask", "brain_mask_file")]),
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
        (dce_preprocess, aif_qc_plot, [("out_file", "dce_file")]),
        (dce_brain_extract, aif_qc_plot, [("out_mask", "brain_mask_file")]),
        (inputnode, aif_qc_plot, [("volume_tr", "tr")]),
    ])

    # ==========================================================================
    # Node 11: Collect QC Metrics
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
                "preproc_display",
            ],
            output_names=["qc_metrics"],
            function=_collect_qc_metrics,
        ),
        name="collect_qc_metrics",
    )

    if not config.no_t1:
        workflow.connect([
            (b1_mapping, collect_qc, [
                ("qc_metrics", "b1_qc"),
            ]),
            (t1_mapping, collect_qc, [
                ("qc_metrics", "t1_qc"),
            ]),
        ])
    else:
        collect_qc.inputs.b1_qc = None
        collect_qc.inputs.t1_qc = None

    workflow.connect([
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
        (preprocessing_qc, collect_qc, [
            ("preproc_display_stats", "preproc_display"),
        ]),
    ])

    # ==========================================================================
    # Node 11: Lesion Statistics (conditional on lesion mask availability)
    # ==========================================================================
    # Transform dilated lesion mask from T1w to DCE space (for AIF exclusion)
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

    # Connect transformed lesion mask to AIF detection (must be in DCE space)
    workflow.connect([
        (transform_lesion, aif_detection, [
            ("lesion_mask_dce_file", "lesion_mask_file"),
        ]),
    ])

    # Select best available tumor mask for lesion statistics
    # Priority: hdglio_segmentation (multi-label) > whole_tumor
    from pmriprep.utils.lesion_stats import combine_tumor_masks

    combine_masks = pe.Node(
        niu.Function(
            input_names=[
                "hdglio_segmentation_file",
                "whole_tumor_mask_file",
                "reference_file",
            ],
            output_names=["combined_mask_file"],
            function=combine_tumor_masks,
        ),
        name="combine_masks",
    )

    workflow.connect([
        (inputnode, combine_masks, [
            ("hdglio_segmentation", "hdglio_segmentation_file"),
            ("whole_tumor_mask", "whole_tumor_mask_file"),
        ]),
        (dce_preprocess, combine_masks, [("mean_img", "reference_file")]),
    ])

    # Transform combined tumor mask from T1w to DCE space (for lesion stats)
    transform_stats_lesion = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dce_file"],
            function=_transform_lesion_mask_to_dce,
        ),
        name="transform_stats_lesion",
    )

    workflow.connect([
        (combine_masks, transform_stats_lesion, [("combined_mask_file", "lesion_mask_file")]),
        (dce_preprocess, transform_stats_lesion, [("mean_img", "reference_file")]),
        (dce_registration, transform_stats_lesion, [("out_matrix", "transform_matrix")]),
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
        (transform_stats_lesion, lesion_stats, [("lesion_mask_dce_file", "lesion_mask")]),
        (transform_nawm_mask, lesion_stats, [("out_file", "nawm_mask")]),
        (pk_modeling, lesion_stats, [
            ("out_ktrans", "ktrans_map"),
            ("out_ve", "ve_map"),
            ("out_vp", "vp_map"),
            ("out_kep", "kep_map"),
            ("out_iauc60", "iauc60_map"),
            ("out_peak_enhancement", "peak_enhancement_map"),
            ("out_time_to_peak", "ttp_map"),
        ]),
        (normalize_ktrans, lesion_stats, [("out_file", "nktrans_map")]),
        (normalize_iauc60, lesion_stats, [("out_file", "niauc60_map")]),
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
        # Pharmacokinetic parameter maps
        (pk_modeling, outputnode, [
            ("out_ktrans", "ktrans_map"),
            ("out_ve", "ve_map"),
            ("out_vp", "vp_map"),
            ("out_kep", "kep_map"),
        ]),
        # Model-free maps
        (pk_modeling, outputnode, [
            ("out_iauc60", "iauc60_map"),
            ("out_iauc90", "iauc90_map"),
            ("out_iauc120", "iauc120_map"),
            ("out_peak_enhancement", "peak_enhancement_map"),
            ("out_time_to_peak", "time_to_peak_map"),
            ("out_ser", "ser_map"),
        ]),
        # QC maps
        (pk_modeling, outputnode, [
            ("out_tsnr", "tsnr_map"),
            ("out_qc_mask", "qc_mask"),
            ("out_r2", "r2_map"),
            ("out_quality_flag", "quality_flag_map"),
        ]),
        # Uncertainty maps (if Bayesian)
        (pk_modeling, outputnode, [
            ("out_ktrans_std", "ktrans_uncertainty"),
            ("out_ve_std", "ve_uncertainty"),
            ("out_vp_std", "vp_uncertainty"),
            ("out_fp_std", "fp_uncertainty"),
            ("out_ps_std", "ps_uncertainty"),
        ]),
        # Bayesian CV and CI maps (if Bayesian)
        (pk_modeling, outputnode, [
            ("out_ktrans_cv", "ktrans_cv"),
            ("out_ve_cv", "ve_cv"),
            ("out_vp_cv", "vp_cv"),
            ("out_ktrans_ci95_lower", "ktrans_ci95_lower"),
            ("out_ktrans_ci95_upper", "ktrans_ci95_upper"),
            ("out_ve_ci95_lower", "ve_ci95_lower"),
            ("out_ve_ci95_upper", "ve_ci95_upper"),
            ("out_vp_ci95_lower", "vp_ci95_lower"),
            ("out_vp_ci95_upper", "vp_ci95_upper"),
        ]),
        # Tortuosity-corrected Ktrans
        (tortuosity_corr, outputnode, [
            ("ktrans_corrected_file", "ktrans_tortuosity_corrected"),
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
        # QC plots - registration filmstrip
        (reg_filmstrip, outputnode, [
            ("filmstrip_file", "registration_qc_filmstrip"),
        ]),
        # QC plots - AIF QC
        (aif_qc_plot, outputnode, [
            ("out_aif_plot", "aif_curve_qc_plot"),
            ("out_aif_location", "aif_location_qc_plot"),
        ]),
        # Motion QC plot
        (motion_qc_plot, outputnode, [
            ("plot_file", "motion_qc_plot"),
        ]),
        # DCE QC plots
        (preprocessing_qc, outputnode, [
            ("plot_file", "preprocessing_qc_plot"),
        ]),
        (pk_modeling, outputnode, [
            ("pk_modeling_qc_plot", "pk_modeling_qc_plot"),
            ("pk_param_maps_qc_plot", "pk_param_maps_qc_plot"),
        ]),
    ])

    # T1/B1/VFA output connections (conditional on T1 mapping being active)
    if not config.no_t1:
        workflow.connect([
            (b1_mapping, outputnode, [
                ("out_file", "b1_map"),
            ]),
            (t1_mapping, outputnode, [
                ("out_file", "t1_map"),
            ]),
            # QC plots - VFA registration
            (vfa_preprocess, outputnode, [
                ("registration_qc_plot", "vfa_registration_qc_plot"),
            ]),
            # QC plots - AFI registration and B1
            (b1_mapping, outputnode, [
                ("registration_qc_plot", "afi_registration_qc_plot"),
                ("qc_plot", "b1_qc_plot"),
            ]),
            # QC plots - VFA T1 mapping
            (t1_mapping, outputnode, [
                ("qc_plot", "t1_qc_plot"),
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
    # Use the combined tumor mask (with multi-label support) for lesion_mask_dce output
    workflow.connect([
        (transform_stats_lesion, outputnode, [("lesion_mask_dce_file", "lesion_mask_dce")]),
        (lesion_stats, outputnode, [
            ("stats_json", "lesion_stats_json"),
            ("stats_txt", "lesion_stats_txt"),
        ]),
    ])

    # Smoothing output connections (only if config.smooth_fwhm > 0)
    if smooth_pk is not None:
        workflow.connect([
            (smooth_pk, outputnode, [("smoothed_files", "smooth_pk_maps")]),
            (smooth_model_free, outputnode, [("smoothed_files", "smooth_model_free_maps")]),
            (smooth_uncertainty, outputnode, [("smoothed_files", "smooth_uncertainty_maps")]),
            (smooth_tortuosity, outputnode, [("smoothed_files", "smooth_tortuosity_maps")]),
        ])

    # Normalized map output connections (requires NAWM mask from hdglio)
    workflow.connect([
        (normalize_ktrans, outputnode, [("out_file", "nktrans_map")]),
        (normalize_iauc60, outputnode, [("out_file", "niauc60_map")]),
    ])

    # Anatomical masks in DCE space output connections
    # Use native DCE brain mask from HD-BET extraction for output
    workflow.connect([
        (dce_brain_extract, outputnode, [("out_mask", "brain_mask")]),
        (transform_nawm_mask, outputnode, [("out_file", "nawm_mask")]),
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
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(mask_file) or mask_file is None:
        return None
    if not isdefined(reference_file) or reference_file is None:
        return None
    if not isdefined(transform_matrix) or transform_matrix is None:
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


def _transform_probseg_to_dce(
    probseg_file=None,
    reference_file=None,
    transform_matrix=None,
):
    """Transform a probability segmentation map from T1w space to DCE space.

    Uses FSL FLIRT with trilinear interpolation (appropriate for continuous
    probability maps, unlike nearest-neighbour for binary masks). Clips output
    to [0, 1] to correct slight over/undershoot from interpolation.

    Args:
        probseg_file: Path to probability map in T1w space (or None).
        reference_file: Path to DCE reference image for output geometry.
        transform_matrix: Path to T1w→DCE FLIRT matrix.

    Returns:
        Path to transformed probability map in DCE space, or None if no input.
    """
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(probseg_file) or probseg_file is None:
        return None
    if not isdefined(reference_file) or reference_file is None:
        return None
    if not isdefined(transform_matrix) or transform_matrix is None:
        return None

    from pathlib import Path
    import subprocess

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    probseg_path = Path(probseg_file)
    if not probseg_path.exists():
        return None

    out_dir = Path.cwd()
    out_file = str(out_dir / "csf_probseg_dce.nii.gz")

    # Same-space shortcut: skip transform if already aligned
    probseg_img = nib.load(probseg_file)
    ref_img = nib.load(reference_file)
    if probseg_img.shape[:3] == ref_img.shape[:3]:
        affine_diff = float(np.abs(probseg_img.affine - ref_img.affine).max())
        if affine_diff < 1e-4:
            # Already in same space, just copy (clip to [0, 1])
            data = probseg_img.get_fdata()  # type: ignore[attr-defined]
            data = np.clip(data, 0.0, 1.0).astype(np.float32)
            out_img = nib.Nifti1Image(data, ref_img.affine)
            nib.save(out_img, out_file)
            return out_file

    # Transform with FLIRT using trilinear interpolation
    tmp_file = str(out_dir / "_tmp_csf_probseg_dce.nii.gz")
    cmd = [
        "flirt",
        "-in", str(probseg_file),
        "-ref", str(reference_file),
        "-out", tmp_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "trilinear",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Clip to [0, 1] — FLIRT trilinear can produce slight over/undershoot
    tmp_img = nib.load(tmp_file)
    data = tmp_img.get_fdata()  # type: ignore[attr-defined]
    data = np.clip(data, 0.0, 1.0).astype(np.float32)
    out_img = nib.Nifti1Image(data, tmp_img.affine)
    nib.save(out_img, out_file)

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
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(lesion_mask_file) or lesion_mask_file is None:
        return None
    if not isdefined(reference_file) or reference_file is None:
        return None
    if not isdefined(transform_matrix) or transform_matrix is None:
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
    dce_native_mask=None,
):
    """Create conservative AIF search mask (Gap 5).

    Creates an intersection of T1w-transformed mask with the native DCE
    brain mask (from HD-BET or BET) to exclude non-brain regions that may
    be incorrectly included in one mask but not the other.

    The intersection approach ensures:
    1. Regions outside T1w brain coverage are excluded
    2. Regions outside DCE brain extraction are excluded
    3. Only voxels in both masks are used for AIF search

    Args:
        t1w_mask_in_dce: Path to T1w brain mask transformed to DCE space.
        dce_native_mask: Path to native DCE brain mask from DCEBrainExtract.

    Returns:
        Path to conservative AIF search mask, or None if inputs missing.
    """
    if t1w_mask_in_dce is None:
        # If no T1w mask, fall back to native DCE mask
        return dce_native_mask

    if dce_native_mask is None:
        # If no native DCE mask, fall back to T1w mask
        return t1w_mask_in_dce

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np
    from scipy.ndimage import binary_fill_holes  # type: ignore[import-untyped]

    t1w_mask_path = Path(t1w_mask_in_dce)
    dce_mask_path = Path(dce_native_mask)

    if not t1w_mask_path.exists():
        return str(dce_native_mask)
    if not dce_mask_path.exists():
        return str(t1w_mask_in_dce)

    # Load masks
    t1w_mask_img = nib.load(t1w_mask_in_dce)
    t1w_mask_data = t1w_mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    dce_mask_img = nib.load(dce_native_mask)
    dce_mask_data = dce_mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    # Intersect masks - conservative approach
    aif_mask = np.logical_and(t1w_mask_data, dce_mask_data).astype(np.int16)

    # Fill holes in the intersection
    aif_mask = binary_fill_holes(aif_mask).astype(np.int16)

    # Save
    out_file = str(Path.cwd() / "aif_search_mask.nii.gz")
    out_img = nib.Nifti1Image(aif_mask, t1w_mask_img.affine)  # type: ignore[attr-defined]
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
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(atlas_file) or atlas_file is None:
        return None
    if not isdefined(t1w_to_mni_xfm) or t1w_to_mni_xfm is None:
        return None
    if not isdefined(t1w_to_dce_xfm) or t1w_to_dce_xfm is None:
        return None
    if not isdefined(reference_file) or reference_file is None:
        return None

    from pathlib import Path
    import subprocess

    atlas_path = Path(atlas_file)
    if not atlas_path.exists():
        return None

    out_dir = Path.cwd()
    # Use atlas-specific temp filenames to avoid collisions in cascade mode
    atlas_stem = atlas_path.name.replace(".nii.gz", "").replace(".nii", "")

    # Step 1: Invert T1w→MNI to get MNI→T1w
    mni_to_t1w_xfm = str(out_dir / f"_tmp_{atlas_stem}_mni_to_t1w.mat")
    invert_cmd = [
        "convert_xfm", "-omat", mni_to_t1w_xfm,
        "-inverse", str(t1w_to_mni_xfm),
    ]
    result = subprocess.run(invert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    # Step 2: Concatenate MNI→T1w + T1w→DCE = MNI→DCE
    mni_to_dce_xfm = str(out_dir / f"_tmp_{atlas_stem}_mni_to_dce.mat")
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
    tmp_atlas = str(out_dir / f"_tmp_{atlas_stem}_dce_raw.nii.gz")
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
    # Derive unique output name from atlas filename to support cascade transforms
    atlas_stem = Path(atlas_file).name.replace(".nii.gz", "").replace(".nii", "")
    out_file = str(out_dir / f"{atlas_stem}_dce.nii.gz")
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


def _transform_atlas_cascade_to_dce(
    atlas_files=None,
    t1w_to_mni_xfm=None,
    t1w_to_dce_xfm=None,
    reference_file=None,
):
    """Transform a list of MNI atlas masks to DCE space via T1w.

    Computes the MNI→DCE composite transform once, then applies it to each
    atlas in the cascade, preserving ordering (tightest → broadest).

    Note: This function is self-contained because Nipype Function nodes
    serialize and execute functions in isolation — calls to sibling
    functions in the same module are not in scope at runtime.

    Args:
        atlas_files: List of MNI-space atlas mask paths (ordered tightest first).
        t1w_to_mni_xfm: Path to T1w→MNI FLIRT matrix.
        t1w_to_dce_xfm: Path to T1w→DCE FLIRT matrix.
        reference_file: Path to DCE reference image for output geometry.

    Returns:
        List of atlas mask paths in DCE space. Empty if inputs are missing
        or all transforms fail.
    """
    from pathlib import Path
    import subprocess

    from nipype.interfaces.base import isdefined

    if not atlas_files:
        return []

    # Validate shared inputs
    for inp in (t1w_to_mni_xfm, t1w_to_dce_xfm, reference_file):
        if not isdefined(inp) or inp is None:
            return []

    out_dir = Path.cwd()

    # Step 1: Compute composite MNI→DCE transform once (shared across atlases)
    mni_to_t1w_xfm = str(out_dir / "_tmp_cascade_mni_to_t1w.mat")
    invert_cmd = [
        "convert_xfm", "-omat", mni_to_t1w_xfm,
        "-inverse", str(t1w_to_mni_xfm),
    ]
    result = subprocess.run(invert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    mni_to_dce_xfm = str(out_dir / "_tmp_cascade_mni_to_dce.mat")
    concat_cmd = [
        "convert_xfm", "-omat", mni_to_dce_xfm,
        "-concat", str(t1w_to_dce_xfm), mni_to_t1w_xfm,
    ]
    result = subprocess.run(concat_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Path(mni_to_t1w_xfm).unlink(missing_ok=True)
        return []

    # Step 2: Apply the composite transform to each atlas
    out_files = []
    for atlas_file in atlas_files:
        atlas_path = Path(atlas_file)
        if not atlas_path.exists():
            continue

        atlas_stem = atlas_path.name.replace(".nii.gz", "").replace(".nii", "")
        tmp_atlas = str(out_dir / f"_tmp_{atlas_stem}_dce_raw.nii.gz")

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
            Path(tmp_atlas).unlink(missing_ok=True)
            continue

        # Morphological refinement: dilate-dilate-erode (1mm sphere)
        out_file = str(out_dir / f"{atlas_stem}_dce.nii.gz")
        morph_cmd = [
            "fslmaths", tmp_atlas,
            "-kernel", "sphere", "1",
            "-dilM", "-dilM", "-ero",
            "-bin", out_file,
        ]
        result = subprocess.run(morph_cmd, capture_output=True, text=True)
        Path(tmp_atlas).unlink(missing_ok=True)

        if result.returncode == 0:
            out_files.append(out_file)

    # Clean up shared temp files
    Path(mni_to_t1w_xfm).unlink(missing_ok=True)
    Path(mni_to_dce_xfm).unlink(missing_ok=True)

    return out_files


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

    # Load B1 map — clip to DCE-appropriate field-strength-dependent range.
    # Tighter than VFA T1 mapping range because signal-to-concentration
    # conversion is more sensitive to B1+ errors.
    from pmriprep.utils.dce_constants import get_b1_thresholds

    b1_thresh = get_b1_thresholds(field_strength)
    b1_map = None
    if b1_map_file:
        try:
            b1_img = nib.load(b1_map_file)
            b1_map = b1_img.get_fdata()  # type: ignore[attr-defined]
            b1_map = np.clip(b1_map, b1_thresh.dce_clip_low, b1_thresh.dce_clip_high)
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
        convert_to_plasma=False,
    )

    # Zero out concentration for voxels with invalid T1 (NaN, sub-physiological).
    # After resampling, out-of-FOV voxels may have T1=NaN which produces
    # extreme concentration values that corrupt AIF detection thresholds.
    if t1_map is not None:
        invalid_t1 = ~np.isfinite(t1_map) | (t1_map < 100.0)  # T1 in ms
        result.concentration[invalid_t1] = 0.0

    # Save concentration file
    out_file = str(Path.cwd() / "dce_concentration.nii.gz")
    conc_img = nib.Nifti1Image(result.concentration.astype(np.float32), affine, header)
    nib.save(conc_img, out_file)

    return out_file


def _tortuosity_correction_node(
    ktrans_file,
    dwi_file,
    dwi_bval=None,
    brain_mask=None,
    no_tortuosity=False,
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
    vp_file=None,
    dsc_cbv_file=None,
    brain_mask=None,
    no_dsc_validation=False,
):
    """Function node for DSC-DCE cross-validation.

    Returns a dictionary with validation results.
    """
    from pmriprep.utils.dce_dsc_validation import validate_vp_cbv_from_files

    # If skipping validation or missing data, return empty result
    if no_dsc_validation or not vp_file or not dsc_cbv_file:
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
    preproc_display=None,
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

    # Merge display stats (TR, tSNR, peak enh) into preprocessing metrics
    if preproc_display is not None:
        qc_metrics.setdefault("preprocessing", {}).update(preproc_display)

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


def _resample_dsc_cbv_to_dce(
    dsc_cbv_file: str | None = None,
    dsc_to_t1w_xfm: str | None = None,
    t1w_to_dce_xfm: str | None = None,
    dce_reference: str | None = None,
) -> str | None:
    """Resample DSC CBV map to DCE native space.

    Chains two existing transforms: DSC→T1w + T1w→DCE = DSC→DCE,
    then applies the combined transform with trilinear interpolation
    (not spline, to avoid negative CBV values).

    This matches the perf-plus approach (dce_pipeline.sh:2664-2688).

    Args:
        dsc_cbv_file: Path to DSC-derived relative CBV map (DSC native space).
        dsc_to_t1w_xfm: Path to DSC→T1w FLIRT transform matrix.
        t1w_to_dce_xfm: Path to T1w→DCE FLIRT transform matrix
            (from DCE registration step).
        dce_reference: Path to DCE reference image for output geometry.

    Returns:
        Path to CBV map resampled to DCE space, or empty string if inputs
        are missing (empty string avoids nipype File trait validation errors
        when None flows through workflow connections).
    """
    import logging
    import os

    _log = logging.getLogger("pmriprep.workflows.dce")

    if not dsc_cbv_file or not os.path.isfile(str(dsc_cbv_file)):
        return ""
    if not dsc_to_t1w_xfm or not os.path.isfile(str(dsc_to_t1w_xfm)):
        return ""
    if not t1w_to_dce_xfm or not os.path.isfile(str(t1w_to_dce_xfm)):
        return ""
    if not dce_reference or not os.path.isfile(str(dce_reference)):
        return ""

    import subprocess

    out_dir = os.getcwd()
    dsc_to_dce_xfm = os.path.join(out_dir, "dsc_to_dce.mat")
    cbv_in_dce = os.path.join(out_dir, "dsc_cbv_in_dce.nii.gz")

    # Chain: DSC→T1w + T1w→DCE = DSC→DCE
    result = subprocess.run(
        [
            "convert_xfm",
            "-omat", dsc_to_dce_xfm,
            "-concat", str(t1w_to_dce_xfm), str(dsc_to_t1w_xfm),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _log.warning(
            "convert_xfm failed (rc=%d): %s", result.returncode, result.stderr
        )
        return ""

    # Apply combined transform with trilinear interpolation
    result = subprocess.run(
        [
            "flirt",
            "-in", str(dsc_cbv_file),
            "-ref", str(dce_reference),
            "-out", cbv_in_dce,
            "-init", dsc_to_dce_xfm,
            "-applyxfm",
            "-interp", "trilinear",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _log.warning(
            "FLIRT resampling of DSC CBV failed (rc=%d): %s",
            result.returncode, result.stderr,
        )
        return ""

    return cbv_in_dce


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

    # Reorient DWI to RAS before registration
    dwi_base = dwi_path.name.replace(".nii.gz", "").replace(".nii", "")
    dwi_reoriented = str(out_dir / f"{dwi_base}_reoriented.nii.gz")
    try:
        subprocess.run(
            ["fslreorient2std", str(dwi_file), dwi_reoriented],
            check=True,
            capture_output=True,
        )
        dwi_file = dwi_reoriented
    except subprocess.CalledProcessError:
        pass  # Use original file if reorientation fails

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

    # Restore NaN for voxels that were NaN before resampling.
    # nilearn's continuous interpolation replaces NaN/out-of-FOV with 0,
    # which downstream causes T1=0 → R1=inf → concentration artifacts.
    data = resampled.get_fdata()
    n_invalid = np.sum(data <= 0)
    if n_invalid > 0:
        data[data <= 0] = np.nan
        resampled = nib.Nifti1Image(
            data.astype(np.float32), resampled.affine, resampled.header,
        )

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
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(structural_file) or structural_file is None:
        return None
    if not isdefined(reference_file) or reference_file is None:
        return None
    if not isdefined(transform_matrix) or transform_matrix is None:
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


def _generate_dce_motion_qc_plot(motion_params_file, out_dir, tr):
    """Generate motion QC plot from DCE motion parameters.

    Creates a two-panel figure showing raw motion parameters and
    framewise displacement over time.

    Args:
        motion_params_file: Path to motion parameters text file.
        out_dir: Output directory for the plot.
        tr: Repetition time in seconds.

    Returns:
        Path to the generated motion QC plot PNG.
    """
    import numpy as np
    from pathlib import Path
    from pmriprep.utils.qc_utils import plot_motion_qc

    motion_params = np.loadtxt(motion_params_file)
    out_path = Path(out_dir) / "motion_qc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_motion_qc(motion_params, out_path, tr=tr, title="DCE Motion QC")
    return str(out_path)


def _generate_preprocessing_qc(
    dce_file,
    brain_mask_file,
    baseline_frames,
    bolus_arrival_frame,
    tr_ms,
    volume_tr_s,
    subject_id="",
    session_id="",
):
    """Generate DCE preprocessing QC plot.

    Thin wrapper that loads NIfTI data and delegates to
    generate_preprocessing_qc_plot().

    Args:
        dce_file: Path to 4D motion-corrected DCE NIfTI.
        brain_mask_file: Path to brain mask NIfTI.
        baseline_frames: Number of baseline frames.
        bolus_arrival_frame: Bolus arrival frame index.
        tr_ms: Pulse TR in seconds (from inputnode ``tr``).
        volume_tr_s: Volume TR in seconds.
        subject_id: Subject identifier.
        session_id: Session identifier.

    Returns:
        Tuple of (plot_file path, preproc_display_stats dict).
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]

    from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

    dce_data = nib.load(dce_file).get_fdata()
    brain_mask = nib.load(brain_mask_file).get_fdata() > 0.5

    # Convert TR from seconds (inputnode) to milliseconds
    tr_ms_val = tr_ms * 1000.0 if tr_ms is not None else 0.0
    volume_tr_val = volume_tr_s if volume_tr_s is not None else 1.0

    out_path = str(Path(dce_file).parent / "dce_preprocessing_qc.png")

    plot_file, stats = generate_preprocessing_qc_plot(
        dce_signal_4d=dce_data,
        brain_mask=brain_mask,
        baseline_frames=baseline_frames,
        bolus_arrival_frame=bolus_arrival_frame,
        volume_tr_s=volume_tr_val,
        subject_id=subject_id,
        session_id=session_id,
        output_path=out_path,
    )

    preproc_display_stats = {
        "tr_ms": tr_ms_val,
        "volume_tr_s": volume_tr_val,
        "n_volumes": dce_data.shape[3],
        **stats,
    }
    return plot_file, preproc_display_stats


def _create_dce_registration_filmstrip(
    dce_mean_file=None,
    registered_t1w_file=None,
    brain_mask_file=None,
    subject_id="",
    session_id="",
):
    """Create registration QC filmstrip for DCE-T1w registration.

    Generates a multi-slice filmstrip showing the DCE mean image with
    contours of the registered T1w image overlaid. This allows visual
    assessment of registration quality.

    Args:
        dce_mean_file: Path to DCE mean image.
        registered_t1w_file: Path to T1w image registered to DCE space.
        brain_mask_file: Path to brain mask in DCE space (optional).

    Returns:
        Path to registration QC filmstrip PNG, or None if inputs missing.
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]

    from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip, qc_figure_title

    if dce_mean_file is None or registered_t1w_file is None:
        return None

    if not Path(dce_mean_file).exists():
        return None
    if not Path(registered_t1w_file).exists():
        return None

    # Load data
    dce_img = nib.load(dce_mean_file)
    dce_data = dce_img.get_fdata()  # type: ignore[attr-defined]
    voxel_sizes = tuple(dce_img.header.get_zooms()[:3])

    t1w_img = nib.load(registered_t1w_file)
    t1w_data = t1w_img.get_fdata()  # type: ignore[attr-defined]

    # Load mask if provided
    mask = None
    if brain_mask_file is not None and Path(brain_mask_file).exists():
        mask_img = nib.load(brain_mask_file)
        mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    # Generate filmstrip — DCE as background, T1w edges overlaid
    out_path = Path.cwd() / "registration_qc_filmstrip.png"
    plot_registration_qc_filmstrip(
        fixed=dce_data,
        moving=t1w_data,
        output_path=out_path,
        mask=mask,
        title=qc_figure_title(subject_id, session_id, "DCE-T1w Registration"),
        voxel_sizes=voxel_sizes,
    )

    return str(out_path)


def _smooth_dce_maps(map_files, brain_mask_file, fwhm):
    """Apply Gaussian smoothing to a dict of DCE maps.

    Uses fslmaths -s for smoothing and -mas for re-masking.

    Args:
        map_files: Dict mapping map name to file path.
        brain_mask_file: Path to brain mask for re-masking after smoothing.
        fwhm: Smoothing FWHM in mm.

    Returns:
        Dict mapping map name to smoothed file path.
    """
    import subprocess
    from pathlib import Path

    from nipype.interfaces.base import isdefined

    results = {}
    if not map_files:
        return results
    if not brain_mask_file or not isdefined(brain_mask_file):
        return results
    if not Path(brain_mask_file).exists():
        return results

    sigma = fwhm / 2.355
    for name, filepath in map_files.items():
        if not filepath or not isdefined(filepath):
            continue
        if isinstance(filepath, str) and filepath in ("", "None"):
            continue
        if not Path(filepath).exists():
            continue
        out_path = str(Path.cwd() / f"{Path(filepath).stem}_fwhm{int(fwhm)}.nii.gz")
        subprocess.run(
            ["fslmaths", filepath, "-s", str(sigma), "-mas", brain_mask_file, out_path],
            check=True,
            capture_output=True,
        )
        results[name] = out_path
    return results


def _build_pk_map_dict(ktrans=None, ve=None, vp=None, kep=None):
    """Build dictionary of PK maps for smoothing."""
    from nipype.interfaces.base import isdefined

    d = {}
    if ktrans and isdefined(ktrans):
        d["ktrans"] = ktrans
    if ve and isdefined(ve):
        d["ve"] = ve
    if kep and isdefined(kep):
        d["kep"] = kep
    if vp and isdefined(vp):
        d["vp"] = vp
    return d


def _build_model_free_map_dict(
    iauc60=None, iauc90=None, iauc120=None, peak=None, ttp=None, ser=None,
):
    """Build dictionary of model-free maps for smoothing.

    Note: TTP is excluded from smoothing because it is a discrete timing
    parameter (quantized to TR). Smoothing creates non-physical intermediate
    time values. The ttp parameter is accepted but not included in the dict.
    """
    from nipype.interfaces.base import isdefined

    d = {}
    if iauc60 and isdefined(iauc60):
        d["iauc60"] = iauc60
    if iauc90 and isdefined(iauc90):
        d["iauc90"] = iauc90
    if iauc120 and isdefined(iauc120):
        d["iauc120"] = iauc120
    if peak and isdefined(peak):
        d["peak_enhancement"] = peak
    if ser and isdefined(ser):
        d["ser"] = ser
    return d


def _build_uncertainty_map_dict(
    ktrans_std=None, ve_std=None, vp_std=None, fp_std=None, ps_std=None,
):
    """Build dictionary of uncertainty std maps for smoothing.

    Only posterior std maps are smoothed. CV (std/mean ratio) and CI95 bounds
    are excluded because smoothing a ratio or derived statistic is physically
    incorrect -- it conflates spatial averaging with statistical inference.
    """
    from nipype.interfaces.base import isdefined

    d = {}
    if ktrans_std and isdefined(ktrans_std):
        d["Ktrans_std"] = ktrans_std
    if ve_std and isdefined(ve_std):
        d["ve_std"] = ve_std
    if vp_std and isdefined(vp_std):
        d["vp_std"] = vp_std
    # 2CXM-specific uncertainty maps
    if fp_std and isdefined(fp_std):
        d["Fp_std"] = fp_std
    if ps_std and isdefined(ps_std):
        d["PS_std"] = ps_std
    return d


def _build_tortuosity_map_dict(ktrans_tortuosity_corrected=None):
    """Build dictionary for tortuosity-corrected Ktrans for smoothing."""
    from nipype.interfaces.base import isdefined

    d = {}
    if ktrans_tortuosity_corrected and isdefined(ktrans_tortuosity_corrected):
        d["Ktrans_tortuositycorr"] = ktrans_tortuosity_corrected
    return d


def _normalize_by_nawm(param_map=None, nawm_mask=None):
    """Normalize parameter map by NAWM mean.

    Args:
        param_map: Path to parameter map NIfTI file.
        nawm_mask: Path to NAWM mask NIfTI file in the same space.

    Returns:
        Path to normalized parameter map, or None if normalization failed.
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np
    from nipype.interfaces.base import isdefined

    if not isdefined(nawm_mask) or nawm_mask is None:
        return None
    if not isdefined(param_map) or param_map is None:
        return None
    if not Path(nawm_mask).exists():
        return None
    if not Path(param_map).exists():
        return None

    param_img = nib.load(param_map)
    nawm_img = nib.load(nawm_mask)

    param_data = param_img.get_fdata()  # type: ignore[attr-defined]
    nawm_data = nawm_img.get_fdata() > 0  # type: ignore[attr-defined]

    # Check shapes match
    if param_data.shape[:3] != nawm_data.shape[:3]:
        return None

    # Compute NAWM mean
    nawm_values = param_data[nawm_data]
    nawm_values = nawm_values[np.isfinite(nawm_values) & (nawm_values > 0)]
    if len(nawm_values) < 10:
        # Not enough valid NAWM voxels
        return None

    nawm_mean = np.mean(nawm_values)
    if nawm_mean <= 0:
        return None

    # Normalize
    normalized = param_data / nawm_mean

    # Save
    param_stem = Path(param_map).name.replace(".nii.gz", "").replace(".nii", "")
    out_file = str(Path.cwd() / f"n{param_stem}.nii.gz")
    nib.save(
        nib.Nifti1Image(normalized.astype(np.float32), param_img.affine),
        out_file,
    )
    return out_file
