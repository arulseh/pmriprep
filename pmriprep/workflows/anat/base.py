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
    BiasCorrection,
    BrainExtraction,
    ReorientAndCrop,
    TissueSegmentation,
)


def _merge_qc_metrics(
    brain_qc: dict[str, Any],
) -> dict[str, Any]:
    """Merge QC metrics from multiple preprocessing steps.

    Parameters
    ----------
    brain_qc : dict
        QC metrics from brain extraction.

    Returns
    -------
    dict
        Merged QC metrics dictionary.
    """
    merged: dict[str, Any] = {}

    # Add brain extraction QC
    if brain_qc:
        merged["brain_extraction"] = brain_qc

    return merged


def init_anat_preproc_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[AnatConfig] = None,
    name: str = "anat_preproc_wf",
) -> pe.Workflow:
    """Initialize anatomical preprocessing workflow.

    This workflow performs anatomical preprocessing for perfusion MRI,
    including reorientation, bias correction, brain extraction,
    tissue segmentation, and optional ACPC alignment.

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
    name : str
        Workflow name (default: "anat_preproc_wf").

    Returns
    -------
    pe.Workflow
        Nipype workflow for anatomical preprocessing.

    Workflow Inputs
    ---------------
    inputnode.t1w : str
        Path to native T1-weighted image.
    inputnode.t1w_postcon : str, optional
        Path to post-contrast T1-weighted image.
    inputnode.t2w : str, optional
        Path to T2-weighted image.
    inputnode.flair : str, optional
        Path to FLAIR image.
    inputnode.lesion_mask : str, optional
        Path to lesion mask for registration cost function weighting.

    Workflow Outputs
    ----------------
    outputnode.t1w_preproc : str
        Path to preprocessed T1-weighted image (bias-corrected, brain-extracted).
    outputnode.brain_mask : str
        Path to brain mask in T1w space.
    outputnode.wm_probseg : str
        Path to white matter probability map.
    outputnode.gm_probseg : str
        Path to gray matter probability map.
    outputnode.csf_probseg : str
        Path to CSF probability map.
    outputnode.t1w_acpc : str, optional
        Path to ACPC-aligned T1w (if not skipped).
    outputnode.acpc_xfm : str, optional
        Path to ACPC alignment transform.
    outputnode.t1w_to_mni_xfm : str, optional
        Path to T1w to MNI transform.
    outputnode.qc_metrics : dict
        Dictionary of QC metrics from preprocessing steps.

    Notes
    -----
    The workflow follows this processing flow:
    1. Reorient and crop (robustfov)
    2. Bias field correction (N4) with optional denoising
    3. Brain extraction (HD-BET or ANTs)
    4. Tissue segmentation (FSL FAST)
    5. Optional ACPC alignment (rigid to MNI)

    Note: Coregistration of other modalities (T1w postcon, T2w, FLAIR) to T1w
    is handled separately in the workflow connecting step since it requires
    the bias-corrected T1w as reference.
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
                "lesion_mask",
            ]
        ),
        name="inputnode",
    )

    # Define output node with all preprocessing outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w_preproc",
                "brain_mask",
                "wm_probseg",
                "gm_probseg",
                "csf_probseg",
                "t1w_acpc",
                "acpc_xfm",
                "t1w_to_mni_xfm",
                "qc_metrics",
            ]
        ),
        name="outputnode",
    )

    # -------------------------------------------------------------------
    # Step 1: Reorient and Crop
    # Reorient to standard orientation and crop FOV using robustfov
    # -------------------------------------------------------------------
    reorient = pe.Node(
        ReorientAndCrop(),
        name="reorient",
    )

    # -------------------------------------------------------------------
    # Step 2: Bias Correction (with optional denoising)
    # Apply N4 bias field correction, optionally preceded by denoising
    # -------------------------------------------------------------------
    bias_correct = pe.Node(
        BiasCorrection(
            denoise=config.denoise,
        ),
        name="bias_correct",
    )

    # -------------------------------------------------------------------
    # Step 3: Brain Extraction
    # Extract brain using HD-BET (default) or ANTs
    # -------------------------------------------------------------------
    brain_extract = pe.Node(
        BrainExtraction(
            tool=config.tool,
        ),
        name="brain_extract",
    )

    # -------------------------------------------------------------------
    # Step 4: Tissue Segmentation
    # Segment brain into WM/GM/CSF using FSL FAST
    # -------------------------------------------------------------------
    tissue_segment = pe.Node(
        TissueSegmentation(),
        name="tissue_segment",
    )

    # -------------------------------------------------------------------
    # Step 5: ACPC Alignment (optional)
    # Rigid alignment to MNI template for standardized orientation
    # -------------------------------------------------------------------
    acpc_align = pe.Node(
        ACPCAlignment(
            skip_acpc=config.skip_acpc,
        ),
        name="acpc_align",
    )

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

    # -------------------------------------------------------------------
    # Wire all nodes together
    # Flow: reorient → bias_correct → brain_extract → segment → acpc
    # -------------------------------------------------------------------

    # Input → Reorient
    workflow.connect([
        (inputnode, reorient, [("t1w", "in_file")]),
    ])

    # Reorient → Bias Correction
    workflow.connect([
        (reorient, bias_correct, [("out_file", "in_file")]),
    ])

    # Bias Correction → Brain Extraction
    workflow.connect([
        (bias_correct, brain_extract, [("out_file", "in_file")]),
    ])

    # Brain Extraction → Tissue Segmentation
    workflow.connect([
        (brain_extract, tissue_segment, [("out_file", "in_file")]),
    ])

    # Brain Extraction → ACPC Alignment (brain-extracted image)
    workflow.connect([
        (brain_extract, acpc_align, [("out_file", "in_file")]),
    ])

    # Brain Extraction QC → QC Merger
    workflow.connect([
        (brain_extract, merge_qc, [("qc_metrics", "brain_qc")]),
    ])

    # -------------------------------------------------------------------
    # Connect outputs to output node
    # -------------------------------------------------------------------
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
        # QC metrics
        (merge_qc, outputnode, [
            ("qc_metrics", "qc_metrics"),
        ]),
    ])

    # Store config in workflow for reference by child nodes
    workflow.config_anat = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
