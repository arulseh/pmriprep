"""Anatomical preprocessing workflow for PMRIPrep.

This module implements the main anatomical preprocessing workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Any, Optional, Union

from nipype.interfaces import utility as niu  # type: ignore[import-untyped]
from nipype.pipeline import engine as pe  # type: ignore[import-untyped]

from pmriprep.config.anat import AnatConfig
from pmriprep.interfaces.anat import (
    ACPCAlignment,
    ApplyMask,
    BiasCorrection,
    BrainExtraction,
    Coregistration,
    ReorientAndCrop,
    T1wSubtraction,
    TissueSegmentation,
)


def _merge_qc_metrics(brain_qc):
    """Merge QC metrics from preprocessing steps.

    Parameters
    ----------
    brain_qc : dict
        QC metrics from brain extraction.

    Returns
    -------
    dict
        Merged QC metrics dictionary.
    """
    merged = {}

    # Add brain extraction QC
    if brain_qc:
        merged["brain_extraction"] = brain_qc

    return merged


def _determine_processing_mode(t1w_precon, t1w_postcon):
    """Determine reference T1w and processing mode.

    Parameters
    ----------
    t1w_precon : str or None
        Path to T1w pre-contrast image.
    t1w_postcon : str or None
        Path to T1w post-contrast image.

    Returns
    -------
    reference_t1w : str
        Path to T1w image to use as reference.
    is_postcon_only : bool
        True if only post-contrast T1w is available.
    """
    from pathlib import Path

    has_precon = t1w_precon is not None and Path(t1w_precon).exists()
    has_postcon = t1w_postcon is not None and Path(t1w_postcon).exists()

    if has_precon:
        return t1w_precon, False  # Use precon as reference, normal mode
    elif has_postcon:
        return t1w_postcon, True  # Use postcon, flag postcon-only mode
    else:
        raise ValueError("At least one T1w image (precon or postcon) is required")


def _passthrough_if_undefined(in_file=None):
    """Return empty string if input is undefined or doesn't exist.

    Parameters
    ----------
    in_file : str or None
        Input file path.

    Returns
    -------
    str
        File path if it exists, empty string otherwise.

    Note
    ----
    Returns empty string instead of None because nipype Function nodes
    serialize None as the string "None" which confuses downstream nodes.
    """
    from pathlib import Path

    if in_file is None:
        return ""
    if not Path(in_file).exists():
        return ""
    return in_file


def _check_same_sequence(pdw_json=None, t2w_json=None):
    """Check if PDw and T2w are from the same multi-echo sequence.

    When PDw and T2w are acquired together in a TSE sequence (different echoes),
    PDw typically registers better to T1w than T2w. In this case, we can register
    PDw and apply the same transform to T2w.

    Parameters
    ----------
    pdw_json : str or None
        Path to PDw JSON sidecar file.
    t2w_json : str or None
        Path to T2w JSON sidecar file.

    Returns
    -------
    bool
        True if PDw and T2w are from the same sequence (same SeriesNumber,
        different EchoNumber), False otherwise.
    """
    import json
    from pathlib import Path

    if not pdw_json or not t2w_json:
        return False

    pdw_path = Path(pdw_json)
    t2w_path = Path(t2w_json)

    if not pdw_path.exists() or not t2w_path.exists():
        return False

    try:
        with open(pdw_path) as f:
            pdw_meta = json.load(f)
        with open(t2w_path) as f:
            t2w_meta = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    # Check if same SeriesNumber (same acquisition)
    pdw_series = pdw_meta.get("SeriesNumber")
    t2w_series = t2w_meta.get("SeriesNumber")

    if pdw_series is None or t2w_series is None:
        return False

    same_series = pdw_series == t2w_series

    # Check if different EchoNumber (different echoes in same sequence)
    pdw_echo = pdw_meta.get("EchoNumber")
    t2w_echo = t2w_meta.get("EchoNumber")

    if pdw_echo is None or t2w_echo is None:
        return False

    different_echo = pdw_echo != t2w_echo

    return same_series and different_echo


def init_anat_preproc_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[AnatConfig] = None,
    has_t1w_postcon: bool = False,
    has_t2w: bool = False,
    has_flair: bool = False,
    has_pdw: bool = False,
    name: str = "anat_preproc_wf",
) -> pe.Workflow:
    """Initialize anatomical preprocessing workflow.

    This workflow performs anatomical preprocessing for perfusion MRI,
    including reorientation, bias correction, cross-modal coregistration,
    brain extraction, tissue segmentation, T1w subtraction,
    and optional ACPC alignment.

    Parameters
    ----------
    bids_dir : Path or str
        Path to BIDS dataset root directory.
    subject_id : str
        BIDS subject identifier (without 'sub-' prefix).
    session_id : str, optional
        BIDS session identifier (without 'ses-' prefix).
    config : AnatConfig, optional
        Anatomical preprocessing configuration. Uses defaults if not provided.
    has_t1w_postcon : bool
        Whether T1w post-contrast image is available.
    has_t2w : bool
        Whether T2w image is available.
    has_flair : bool
        Whether FLAIR image is available.
    has_pdw : bool
        Whether PDw image is available.
    name : str
        Workflow name (default: "anat_preproc_wf").

    Returns
    -------
    pe.Workflow
        Nipype workflow for anatomical preprocessing.

    Workflow Inputs
    ---------------
    inputnode.t1w : str
        Path to native T1-weighted pre-contrast image.
    inputnode.t1w_postcon : str, optional
        Path to post-contrast T1-weighted image.
    inputnode.t2w : str, optional
        Path to T2-weighted image.
    inputnode.flair : str, optional
        Path to FLAIR image.
    inputnode.pdw : str, optional
        Path to proton density weighted image.
    inputnode.lesion_mask : str, optional
        Path to lesion mask for registration cost function weighting.

    Workflow Outputs
    ----------------
    outputnode.t1w_preproc : str
        Path to preprocessed T1-weighted image (bias-corrected, brain-extracted).
    outputnode.brain_mask : str
        Path to brain mask in T1w space.
    outputnode.wm_probseg : str
        Path to white matter probability map (None if postcon-only mode).
    outputnode.gm_probseg : str
        Path to gray matter probability map (None if postcon-only mode).
    outputnode.csf_probseg : str
        Path to CSF probability map (None if postcon-only mode).
    outputnode.t1w_acpc : str, optional
        Path to ACPC-aligned T1w (if not skipped).
    outputnode.acpc_xfm : str, optional
        Path to ACPC alignment transform.
    outputnode.t1w_to_mni_xfm : str, optional
        Path to T1w to MNI transform.
    outputnode.t1w_postcon_coreg : str, optional
        T1w post-contrast coregistered to pre-contrast.
    outputnode.t2w_coreg : str, optional
        T2w coregistered to T1w.
    outputnode.flair_coreg : str, optional
        FLAIR coregistered to T1w.
    outputnode.pdw_coreg : str, optional
        PDw coregistered to T1w.
    outputnode.t1w_postcon_to_t1w_xfm : str, optional
        T1w postcon to T1w transform.
    outputnode.t2w_to_t1w_xfm : str, optional
        T2w to T1w transform.
    outputnode.flair_to_t1w_xfm : str, optional
        FLAIR to T1w transform.
    outputnode.pdw_to_t1w_xfm : str, optional
        PDw to T1w transform.
    outputnode.t1w_subtraction : str, optional
        T1w subtraction image (post - pre).
    outputnode.is_postcon_only : bool
        Flag indicating postcon-only mode.
    outputnode.qc_metrics : dict
        Dictionary of QC metrics from preprocessing steps.

    Notes
    -----
    The workflow follows this processing flow:

    1. Determine processing mode (precon reference vs postcon-only)
    2. Reorient and crop T1w reference
    3. Reorient and crop secondary modalities (in parallel)
    4. Bias correction on T1w reference
    5. Bias correction on secondary modalities (in parallel)
    6. Cross-modal coregistration to T1w (before brain extraction)
    7. Brain extraction (HD-BET or ANTs)
    8. Apply brain mask to all coregistered images
    9. Tissue segmentation (FSL FAST) - SKIPPED if postcon-only
    10. T1w subtraction (when both pre and post available)
    11. Optional ACPC alignment (rigid to MNI)

    In postcon-only mode:
    - T1w post-contrast is used as the reference
    - FAST segmentation is SKIPPED (gadolinium disrupts WM/GM boundaries)
    - T1w subtraction is SKIPPED (requires both pre and post)
    - A warning is logged about reduced accuracy
    """
    # Use default config if not provided
    if config is None:
        config = AnatConfig()

    # Store parameters for use in workflow nodes
    bids_dir = Path(bids_dir)

    # Create workflow
    workflow = pe.Workflow(name=name)

    # Define input node with all possible anatomical inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t1w_postcon",
                "t2w",
                "flair",
                "pdw",
                "lesion_mask",
            ]
        ),
        name="inputnode",
    )

    # Define output node with all preprocessing outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Core outputs
                "t1w_preproc",
                "brain_mask",
                "wm_probseg",
                "gm_probseg",
                "csf_probseg",
                "t1w_acpc",
                "acpc_xfm",
                "t1w_to_mni_xfm",
                # Coregistered modalities
                "t1w_postcon_coreg",
                "t2w_coreg",
                "flair_coreg",
                "pdw_coreg",
                # Coregistration transforms
                "t1w_postcon_to_t1w_xfm",
                "t2w_to_t1w_xfm",
                "flair_to_t1w_xfm",
                "pdw_to_t1w_xfm",
                # Subtraction
                "t1w_subtraction",
                # Processing mode flag
                "is_postcon_only",
                # QC
                "qc_metrics",
            ]
        ),
        name="outputnode",
    )

    # -------------------------------------------------------------------
    # Step 0: Determine processing mode
    # Check if we have precon (normal) or only postcon (limited mode)
    # -------------------------------------------------------------------
    determine_mode = pe.Node(
        niu.Function(
            input_names=["t1w_precon", "t1w_postcon"],
            output_names=["reference_t1w", "is_postcon_only"],
            function=_determine_processing_mode,
        ),
        name="determine_mode",
    )

    workflow.connect([
        (inputnode, determine_mode, [
            ("t1w", "t1w_precon"),
            ("t1w_postcon", "t1w_postcon"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Step 1: Reorient and Crop - T1w reference
    # Reorient to standard orientation and crop FOV using robustfov
    # -------------------------------------------------------------------
    reorient = pe.Node(
        ReorientAndCrop(),
        name="reorient",
    )

    workflow.connect([
        (determine_mode, reorient, [("reference_t1w", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # Step 2: Bias Correction - T1w reference
    # Apply N4 bias field correction, optionally preceded by denoising
    # -------------------------------------------------------------------
    bias_correct = pe.Node(
        BiasCorrection(
            denoise=config.denoise,
        ),
        name="bias_correct",
    )

    workflow.connect([
        (reorient, bias_correct, [("out_file", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # Step 3-4: Secondary modality preprocessing and coregistration
    # Only create nodes for modalities that are available
    # -------------------------------------------------------------------

    # Initialize node references (will be set if modality is available)
    coreg_postcon = None
    coreg_t2w = None
    coreg_flair = None
    coreg_pdw = None

    if config.coregister_modalities:
        # T1w post-contrast preprocessing and coregistration
        if has_t1w_postcon:
            reorient_postcon = pe.Node(ReorientAndCrop(), name="reorient_postcon")
            bias_correct_postcon = pe.Node(
                BiasCorrection(denoise=config.denoise), name="bias_correct_postcon"
            )
            coreg_postcon = pe.Node(
                Coregistration(modality="T1_postcon", dof=6), name="coreg_postcon"
            )
            workflow.connect([
                (inputnode, reorient_postcon, [("t1w_postcon", "in_file")]),
                (reorient_postcon, bias_correct_postcon, [("out_file", "in_file")]),
                (bias_correct_postcon, coreg_postcon, [("out_file", "in_file")]),
                (bias_correct, coreg_postcon, [("out_file", "reference")]),
                (inputnode, coreg_postcon, [("lesion_mask", "lesion_mask")]),
            ])

        # T2w preprocessing and coregistration
        if has_t2w:
            reorient_t2w = pe.Node(ReorientAndCrop(), name="reorient_t2w")
            bias_correct_t2w = pe.Node(
                BiasCorrection(denoise=config.denoise), name="bias_correct_t2w"
            )
            coreg_t2w = pe.Node(
                Coregistration(modality="T2w", dof=6), name="coreg_t2w"
            )
            workflow.connect([
                (inputnode, reorient_t2w, [("t2w", "in_file")]),
                (reorient_t2w, bias_correct_t2w, [("out_file", "in_file")]),
                (bias_correct_t2w, coreg_t2w, [("out_file", "in_file")]),
                (bias_correct, coreg_t2w, [("out_file", "reference")]),
                (inputnode, coreg_t2w, [("lesion_mask", "lesion_mask")]),
            ])

        # FLAIR preprocessing and coregistration
        if has_flair:
            reorient_flair = pe.Node(ReorientAndCrop(), name="reorient_flair")
            bias_correct_flair = pe.Node(
                BiasCorrection(denoise=config.denoise), name="bias_correct_flair"
            )
            coreg_flair = pe.Node(
                Coregistration(modality="FLAIR", dof=6), name="coreg_flair"
            )
            workflow.connect([
                (inputnode, reorient_flair, [("flair", "in_file")]),
                (reorient_flair, bias_correct_flair, [("out_file", "in_file")]),
                (bias_correct_flair, coreg_flair, [("out_file", "in_file")]),
                (bias_correct, coreg_flair, [("out_file", "reference")]),
                (inputnode, coreg_flair, [("lesion_mask", "lesion_mask")]),
            ])

        # PDw preprocessing and coregistration
        if has_pdw:
            reorient_pdw = pe.Node(ReorientAndCrop(), name="reorient_pdw")
            bias_correct_pdw = pe.Node(
                BiasCorrection(denoise=config.denoise), name="bias_correct_pdw"
            )
            coreg_pdw = pe.Node(
                Coregistration(modality="PDw", dof=6), name="coreg_pdw"
            )
            workflow.connect([
                (inputnode, reorient_pdw, [("pdw", "in_file")]),
                (reorient_pdw, bias_correct_pdw, [("out_file", "in_file")]),
                (bias_correct_pdw, coreg_pdw, [("out_file", "in_file")]),
                (bias_correct, coreg_pdw, [("out_file", "reference")]),
                (inputnode, coreg_pdw, [("lesion_mask", "lesion_mask")]),
            ])

    # -------------------------------------------------------------------
    # Step 5: Brain Extraction
    # Extract brain using HD-BET (default) or ANTs
    # -------------------------------------------------------------------
    brain_extract = pe.Node(
        BrainExtraction(
            tool=config.tool,
        ),
        name="brain_extract",
    )

    workflow.connect([
        (bias_correct, brain_extract, [("out_file", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # Step 6: Apply brain mask to coregistered images
    # Only create mask nodes for modalities that exist
    # -------------------------------------------------------------------

    if config.coregister_modalities:
        if has_t1w_postcon and coreg_postcon is not None:
            mask_postcon = pe.Node(ApplyMask(), name="mask_postcon")
            workflow.connect([
                (coreg_postcon, mask_postcon, [("out_file", "in_file")]),
                (brain_extract, mask_postcon, [("out_mask", "mask_file")]),
            ])

        if has_t2w and coreg_t2w is not None:
            mask_t2w = pe.Node(ApplyMask(), name="mask_t2w")
            workflow.connect([
                (coreg_t2w, mask_t2w, [("out_file", "in_file")]),
                (brain_extract, mask_t2w, [("out_mask", "mask_file")]),
            ])

        if has_flair and coreg_flair is not None:
            mask_flair = pe.Node(ApplyMask(), name="mask_flair")
            workflow.connect([
                (coreg_flair, mask_flair, [("out_file", "in_file")]),
                (brain_extract, mask_flair, [("out_mask", "mask_file")]),
            ])

        if has_pdw and coreg_pdw is not None:
            mask_pdw = pe.Node(ApplyMask(), name="mask_pdw")
            workflow.connect([
                (coreg_pdw, mask_pdw, [("out_file", "in_file")]),
                (brain_extract, mask_pdw, [("out_mask", "mask_file")]),
            ])

    # -------------------------------------------------------------------
    # Step 7: Tissue Segmentation (conditional on precon availability)
    # Segment brain into WM/GM/CSF using FSL FAST
    # SKIPPED in postcon-only mode (gadolinium disrupts tissue boundaries)
    # -------------------------------------------------------------------

    # Node to conditionally run FAST based on processing mode
    def _run_fast_if_normal_mode(in_file, is_postcon_only):
        """Run FAST only if we have precon (not postcon-only mode).

        Returns input file if normal mode, None if postcon-only.
        """
        if is_postcon_only:
            import warnings
            warnings.warn(
                "Postcon-only mode: Skipping FAST segmentation. "
                "Gadolinium contrast disrupts WM/GM intensity boundaries, "
                "making tissue segmentation unreliable."
            )
            return None
        return in_file

    check_fast_mode = pe.Node(
        niu.Function(
            input_names=["in_file", "is_postcon_only"],
            output_names=["out_file"],
            function=_run_fast_if_normal_mode,
        ),
        name="check_fast_mode",
    )

    tissue_segment = pe.Node(
        TissueSegmentation(),
        name="tissue_segment",
    )

    workflow.connect([
        (brain_extract, check_fast_mode, [("out_file", "in_file")]),
        (determine_mode, check_fast_mode, [("is_postcon_only", "is_postcon_only")]),
        (check_fast_mode, tissue_segment, [("out_file", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # Step 8: T1w Subtraction (when both pre and post available)
    # -------------------------------------------------------------------

    if config.create_subtraction and config.coregister_modalities and has_t1w_postcon:
        if coreg_postcon is not None:
            create_subtraction = pe.Node(
                T1wSubtraction(),
                name="create_subtraction",
            )

            # Use FLIRT-coregistered postcon image
            workflow.connect([
                (coreg_postcon, create_subtraction, [("out_file", "postcon_file")]),
                (bias_correct, create_subtraction, [("out_file", "precon_file")]),
                (brain_extract, create_subtraction, [("out_mask", "brain_mask")]),
            ])

    # -------------------------------------------------------------------
    # Step 9: ACPC Alignment (optional)
    # Rigid alignment to MNI template for standardized orientation
    # -------------------------------------------------------------------
    acpc_align = pe.Node(
        ACPCAlignment(
            skip_acpc=config.skip_acpc,
        ),
        name="acpc_align",
    )

    workflow.connect([
        (brain_extract, acpc_align, [("out_file", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # QC Metrics Merger
    # Combine QC metrics from all preprocessing steps
    # -------------------------------------------------------------------
    merge_qc = pe.Node(
        niu.Function(
            input_names=["brain_qc"],
            output_names=["qc_metrics"],
            function=_merge_qc_metrics,
        ),
        name="merge_qc",
    )

    workflow.connect([
        (brain_extract, merge_qc, [("qc_metrics", "brain_qc")]),
    ])

    # -------------------------------------------------------------------
    # Connect outputs to output node
    # -------------------------------------------------------------------

    # Core outputs
    workflow.connect([
        # Preprocessed T1w is the brain-extracted image
        (brain_extract, outputnode, [
            ("out_file", "t1w_preproc"),
            ("out_mask", "brain_mask"),
        ]),
        # Tissue probability maps from segmentation
        (tissue_segment, outputnode, [
            ("wm_probseg", "wm_probseg"),
            ("gm_probseg", "gm_probseg"),
            ("csf_probseg", "csf_probseg"),
        ]),
        # ACPC alignment outputs
        (acpc_align, outputnode, [
            ("out_file", "t1w_acpc"),
            ("out_matrix", "acpc_xfm"),
            # ACPC transform is also the T1w to MNI transform (rigid)
            ("out_matrix", "t1w_to_mni_xfm"),
        ]),
        # Processing mode flag
        (determine_mode, outputnode, [("is_postcon_only", "is_postcon_only")]),
        # QC metrics
        (merge_qc, outputnode, [("qc_metrics", "qc_metrics")]),
    ])

    # Coregistered modalities and transforms (only for available modalities)
    if config.coregister_modalities:
        if has_t1w_postcon and coreg_postcon is not None:
            workflow.connect([
                (mask_postcon, outputnode, [("out_file", "t1w_postcon_coreg")]),
                (coreg_postcon, outputnode, [("out_matrix", "t1w_postcon_to_t1w_xfm")]),
            ])
        if has_t2w and coreg_t2w is not None:
            workflow.connect([
                (mask_t2w, outputnode, [("out_file", "t2w_coreg")]),
                (coreg_t2w, outputnode, [("out_matrix", "t2w_to_t1w_xfm")]),
            ])
        if has_flair and coreg_flair is not None:
            workflow.connect([
                (mask_flair, outputnode, [("out_file", "flair_coreg")]),
                (coreg_flair, outputnode, [("out_matrix", "flair_to_t1w_xfm")]),
            ])
        if has_pdw and coreg_pdw is not None:
            workflow.connect([
                (mask_pdw, outputnode, [("out_file", "pdw_coreg")]),
                (coreg_pdw, outputnode, [("out_matrix", "pdw_to_t1w_xfm")]),
            ])

    # Subtraction image
    if config.create_subtraction and config.coregister_modalities and has_t1w_postcon:
        workflow.connect([
            (create_subtraction, outputnode, [("out_file", "t1w_subtraction")]),
        ])

    # Store config in workflow for reference by child nodes
    workflow.config_anat = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
