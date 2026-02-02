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
        (transform_brain_mask, aif_detection, [
            ("out_file", "brain_mask_file"),
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
            output_names=["ktrans_corrected_file", "was_corrected"],
            function=_tortuosity_correction_node,
        ),
        name="tortuosity_correction",
    )

    tortuosity_corr.inputs.no_tortuosity = config.no_tortuosity

    workflow.connect([
        (pk_modeling, tortuosity_corr, [
            ("out_ktrans", "ktrans_file"),
        ]),
        (inputnode, tortuosity_corr, [
            ("dwi_file", "dwi_file"),
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
    the brain mask into DCE space.

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

    mask_path = Path(mask_file)
    if not mask_path.exists():
        return None

    out_dir = Path.cwd()
    out_file = str(out_dir / "brain_mask_dce.nii.gz")

    cmd = [
        "flirt",
        "-in", str(mask_file),
        "-ref", str(reference_file),
        "-out", out_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "nearestneighbour",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

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

    Returns the (possibly corrected) Ktrans file and whether correction was applied.
    """
    import shutil
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    from pmriprep.utils.dce_tortuosity import calculate_and_apply_tortuosity_correction

    # If skipping tortuosity or no DWI data, return original Ktrans
    if no_tortuosity or dwi_file is None or dwi_bval is None:
        # Just copy the file to output directory for consistency
        out_file = str(Path.cwd() / "ktrans_final.nii.gz")
        shutil.copy(ktrans_file, out_file)
        return out_file, False

    try:
        # Load files
        ktrans_img = nib.load(ktrans_file)
        ktrans_data = ktrans_img.get_fdata()  # type: ignore[attr-defined]

        dwi_img = nib.load(dwi_file)
        dwi_data = dwi_img.get_fdata()  # type: ignore[attr-defined]

        mask_img = nib.load(brain_mask)
        mask_data = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

        # Create b_values array: [0, dwi_bval]
        # Assume dwi_file is 4D with b0 and b1
        b_values = np.array([0.0, dwi_bval])

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

        return out_file, correction_result.was_corrected

    except Exception:
        # If correction fails, return original Ktrans
        out_file = str(Path.cwd() / "ktrans_final.nii.gz")
        shutil.copy(ktrans_file, out_file)
        return out_file, False


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
