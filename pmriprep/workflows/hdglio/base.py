"""HD-GLIO tumor segmentation workflow for PMRIPrep.

This module implements the HD-GLIO deep learning tumor segmentation workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Any, Optional, Union

from nipype.interfaces import utility as niu  # type: ignore[import-untyped]
from nipype.pipeline import engine as pe  # type: ignore[import-untyped]

from pmriprep.config.hdglio import HDGLIOConfig
from pmriprep.interfaces.hdglio import HDGLIOPrep, HDGLIOSegment


def _extract_masks_node(
    segmentation_file: str,
    prefix: str,
) -> dict[str, str]:
    """Node function to extract tumor masks from HD-GLIO segmentation.

    This function wraps the extract_tumor_masks utility for use in a
    Nipype Function node.

    Parameters
    ----------
    segmentation_file : str
        Path to HD-GLIO segmentation NIfTI file
    prefix : str
        Prefix for output filenames

    Returns
    -------
    dict
        Dictionary with paths to extracted masks:
        - enhancing: Path to enhancing tumor mask
        - non_enhancing: Path to non-enhancing tumor mask
        - whole_tumor: Path to whole tumor mask
    """
    from pmriprep.utils.hdglio import extract_tumor_masks

    return extract_tumor_masks(
        segmentation_file=segmentation_file,
        output_dir=None,  # Use same directory as input file
        prefix=prefix,
    )


def _dilate_mask_node(
    mask_file: str,
    radius_mm: float,
) -> str:
    """Node function to dilate a tumor mask.

    This function wraps the dilate_mask utility for use in a
    Nipype Function node.

    Parameters
    ----------
    mask_file : str
        Path to input binary mask
    radius_mm : float
        Dilation radius in millimeters

    Returns
    -------
    str
        Path to the dilated mask file
    """
    from pmriprep.utils.hdglio import dilate_mask

    return dilate_mask(
        mask_file=mask_file,
        radius_mm=radius_mm,
        output_file=None,  # Auto-generate filename
    )


def _create_nawm_node(
    tumor_mask_file: str,
    wm_probseg_file: str,
    nawm_erosion: int,
    nawm_min_voxels: int,
) -> dict[str, Any]:
    """Node function to create NAWM ROI.

    This function wraps the create_nawm_roi utility for use in a
    Nipype Function node.

    Parameters
    ----------
    tumor_mask_file : str
        Path to whole tumor mask
    wm_probseg_file : str
        Path to WM probability segmentation
    nawm_erosion : int
        Erosion iterations for NAWM mask
    nawm_min_voxels : int
        Minimum voxels required in NAWM ROI

    Returns
    -------
    dict
        Dictionary with output file path and analysis details
    """
    from pmriprep.utils.hdglio import create_nawm_roi

    return create_nawm_roi(
        tumor_mask_file=tumor_mask_file,
        wm_probseg_file=wm_probseg_file,
        output_file=None,  # Auto-generate filename
        nawm_erosion=nawm_erosion,
        nawm_min_voxels=nawm_min_voxels,
    )


def _generate_qc_metrics(
    prep_qc: dict[str, Any],
    segment_qc: dict[str, Any],
    nawm_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Combine QC metrics from all HD-GLIO processing steps.

    Parameters
    ----------
    prep_qc : dict
        QC metrics from HDGLIOPrep
    segment_qc : dict
        QC metrics from HDGLIOSegment
    nawm_analysis : dict
        Analysis results from NAWM ROI creation

    Returns
    -------
    dict
        Combined QC metrics dictionary
    """
    return {
        "preprocessing": prep_qc,
        "segmentation": segment_qc,
        "nawm": nawm_analysis,
    }


def init_hdglio_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[HDGLIOConfig] = None,
    name: str = "hdglio_wf",
) -> pe.Workflow:
    """Initialize HD-GLIO tumor segmentation workflow.

    This workflow performs deep learning-based glioma segmentation using
    HD-GLIO, including preprocessing, segmentation, mask extraction,
    dilation, and NAWM ROI creation.

    Parameters
    ----------
    bids_dir : Path or str
        Path to BIDS dataset root directory.
    subject_id : str
        BIDS subject identifier (without 'sub-' prefix).
    session_id : str, optional
        BIDS session identifier (without 'ses-' prefix).
    config : HDGLIOConfig, optional
        HD-GLIO configuration. Uses defaults if not provided.
    name : str
        Workflow name (default: "hdglio_wf").

    Returns
    -------
    pe.Workflow
        Nipype workflow for HD-GLIO tumor segmentation.

    Workflow Inputs
    ---------------
    inputnode.t1w_pre : str
        Path to pre-contrast T1-weighted image (brain-extracted).
    inputnode.t1w_postcon : str
        Path to post-contrast T1-weighted image (ce-GAD, brain-extracted).
    inputnode.t2w : str
        Path to T2-weighted image (brain-extracted).
    inputnode.flair : str
        Path to FLAIR image (brain-extracted).
    inputnode.brain_mask : str
        Path to brain mask from anatomical preprocessing.
    inputnode.wm_probseg : str
        Path to white matter probability map for NAWM ROI creation.

    Workflow Outputs
    ----------------
    outputnode.segmentation : str
        Path to HD-GLIO segmentation output (labels: 0=bg, 1=non-enhancing, 2=enhancing).
    outputnode.enhancing_mask : str
        Path to enhancing tumor mask (label 2).
    outputnode.non_enhancing_mask : str
        Path to non-enhancing tumor mask (label 1).
    outputnode.whole_tumor_mask : str
        Path to whole tumor mask (labels 1 + 2).
    outputnode.dilated_lesion_mask : str
        Path to dilated lesion mask for registration cost function weighting.
    outputnode.nawm_mask : str
        Path to contralateral NAWM mask for CBV normalization.
    outputnode.qc_metrics : dict
        Dictionary of QC metrics from segmentation.

    Notes
    -----
    The workflow follows this processing flow:
    1. Prepare images for HD-GLIO (NO bias correction, NO intensity normalization)
    2. Run HD-GLIO deep learning segmentation
    3. Extract individual tumor masks (enhancing, non-enhancing, whole)
    4. Dilate lesion mask for registration use
    5. Create contralateral NAWM ROI for CBV normalization

    HD-GLIO requires all 4 modalities: T1w-pre, T1w-ce-GAD, T2w, FLAIR.
    """
    # Use default config if not provided
    if config is None:
        config = HDGLIOConfig()

    # Store parameters for use in workflow nodes
    bids_dir = Path(bids_dir)

    # Build BIDS prefix for output filenames
    bids_prefix = f"sub-{subject_id}_"
    if session_id:
        bids_prefix += f"ses-{session_id}_"

    # Create workflow
    workflow = pe.Workflow(name=name)

    # Define input node with required HD-GLIO inputs
    # HD-GLIO requires: T1w-pre, T1w-ce-GAD (post-contrast), T2w, FLAIR
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w_pre",
                "t1w_postcon",
                "t2w",
                "flair",
                "brain_mask",
                "wm_probseg",
            ]
        ),
        name="inputnode",
    )

    # Define output node with all segmentation outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "segmentation",
                "enhancing_mask",
                "non_enhancing_mask",
                "whole_tumor_mask",
                "dilated_lesion_mask",
                "nawm_mask",
                "qc_metrics",
            ]
        ),
        name="outputnode",
    )

    # Store config in workflow for reference by child nodes
    workflow.config_hdglio = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    # -------------------------------------------------------------------
    # Step 1: HD-GLIO Preprocessing (US-078)
    # - Minimal preprocessing: brain extraction only (NO bias correction)
    # - Verify all 4 modalities present and dimension-matched
    # -------------------------------------------------------------------
    hdglio_prep = pe.Node(
        HDGLIOPrep(),
        name="hdglio_prep",
    )

    # Connect inputs to prep node
    workflow.connect([
        (inputnode, hdglio_prep, [
            ("t1w_pre", "t1w_pre"),
            ("t1w_postcon", "t1w_postcon"),
            ("t2w", "t2w"),
            ("flair", "flair"),
            ("brain_mask", "brain_mask"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Step 2: HD-GLIO Segmentation (US-079)
    # - Run hd_glio_predict with config.device (cuda/mps/cpu)
    # - Output labels: 0=background, 1=non-enhancing, 2=enhancing
    # -------------------------------------------------------------------
    hdglio_segment = pe.Node(
        HDGLIOSegment(device=config.device),
        name="hdglio_segment",
    )

    # Connect prepped images to segmentation node
    workflow.connect([
        (hdglio_prep, hdglio_segment, [
            ("t1w_pre_prepped", "t1w_pre"),
            ("t1w_postcon_prepped", "t1w_postcon"),
            ("t2w_prepped", "t2w"),
            ("flair_prepped", "flair"),
        ]),
    ])

    # Connect segmentation output to workflow outputnode
    workflow.connect([
        (hdglio_segment, outputnode, [("segmentation", "segmentation")]),
    ])

    # -------------------------------------------------------------------
    # Step 3: Mask Extraction (US-080)
    # - Extract enhancing, non-enhancing, and whole tumor masks
    # -------------------------------------------------------------------
    extract_masks = pe.Node(
        niu.Function(
            input_names=["segmentation_file", "prefix"],
            output_names=["masks"],
            function=_extract_masks_node,
        ),
        name="extract_masks",
    )
    extract_masks.inputs.prefix = bids_prefix

    # Connect segmentation to mask extraction
    workflow.connect([
        (hdglio_segment, extract_masks, [("segmentation", "segmentation_file")]),
    ])

    # Extract individual mask paths from the masks dict using Function nodes
    select_enhancing = pe.Node(
        niu.Function(
            input_names=["masks"],
            output_names=["mask_path"],
            function=lambda masks: masks["enhancing"],
        ),
        name="select_enhancing",
    )
    select_non_enhancing = pe.Node(
        niu.Function(
            input_names=["masks"],
            output_names=["mask_path"],
            function=lambda masks: masks["non_enhancing"],
        ),
        name="select_non_enhancing",
    )
    select_whole_tumor = pe.Node(
        niu.Function(
            input_names=["masks"],
            output_names=["mask_path"],
            function=lambda masks: masks["whole_tumor"],
        ),
        name="select_whole_tumor",
    )

    workflow.connect([
        (extract_masks, select_enhancing, [("masks", "masks")]),
        (extract_masks, select_non_enhancing, [("masks", "masks")]),
        (extract_masks, select_whole_tumor, [("masks", "masks")]),
    ])

    # Connect individual mask paths to output node
    workflow.connect([
        (select_enhancing, outputnode, [("mask_path", "enhancing_mask")]),
        (select_non_enhancing, outputnode, [("mask_path", "non_enhancing_mask")]),
        (select_whole_tumor, outputnode, [("mask_path", "whole_tumor_mask")]),
    ])

    # -------------------------------------------------------------------
    # Step 4: Mask Dilation (US-080)
    # - Dilate whole tumor mask with config.lesion_dilation (default: 15mm)
    # -------------------------------------------------------------------
    dilate_lesion = pe.Node(
        niu.Function(
            input_names=["mask_file", "radius_mm"],
            output_names=["dilated_mask"],
            function=_dilate_mask_node,
        ),
        name="dilate_lesion",
    )
    dilate_lesion.inputs.radius_mm = config.lesion_dilation

    # Connect whole tumor mask to dilation
    workflow.connect([
        (select_whole_tumor, dilate_lesion, [("mask_path", "mask_file")]),
    ])

    # Connect dilated mask to output
    workflow.connect([
        (dilate_lesion, outputnode, [("dilated_mask", "dilated_lesion_mask")]),
    ])

    # -------------------------------------------------------------------
    # Step 5: NAWM ROI Creation (US-081)
    # - Create contralateral NAWM mask for CBV normalization
    # - Use WM probability >= 0.9, exclude tumor + 20mm margin
    # - Apply config.nawm_erosion (default: 2 voxels)
    # - Ensure >= config.nawm_min_voxels (default: 500)
    # -------------------------------------------------------------------
    create_nawm = pe.Node(
        niu.Function(
            input_names=[
                "tumor_mask_file",
                "wm_probseg_file",
                "nawm_erosion",
                "nawm_min_voxels",
            ],
            output_names=["nawm_result"],
            function=_create_nawm_node,
        ),
        name="create_nawm",
    )
    create_nawm.inputs.nawm_erosion = config.nawm_erosion
    create_nawm.inputs.nawm_min_voxels = config.nawm_min_voxels

    # Connect whole tumor mask and WM probseg to NAWM creation
    workflow.connect([
        (select_whole_tumor, create_nawm, [("mask_path", "tumor_mask_file")]),
        (inputnode, create_nawm, [("wm_probseg", "wm_probseg_file")]),
    ])

    # Extract NAWM mask path from result dict
    extract_nawm_path = pe.Node(
        niu.Function(
            input_names=["nawm_result"],
            output_names=["nawm_mask"],
            function=lambda nawm_result: nawm_result["output_file"],
        ),
        name="extract_nawm_path",
    )

    workflow.connect([
        (create_nawm, extract_nawm_path, [("nawm_result", "nawm_result")]),
        (extract_nawm_path, outputnode, [("nawm_mask", "nawm_mask")]),
    ])

    # -------------------------------------------------------------------
    # Step 6: Combine QC Metrics
    # -------------------------------------------------------------------
    combine_qc = pe.Node(
        niu.Function(
            input_names=["prep_qc", "segment_qc", "nawm_analysis"],
            output_names=["qc_metrics"],
            function=_generate_qc_metrics,
        ),
        name="combine_qc",
    )

    # Extract NAWM analysis from result dict
    extract_nawm_analysis = pe.Node(
        niu.Function(
            input_names=["nawm_result"],
            output_names=["analysis"],
            function=lambda nawm_result: nawm_result["analysis"],
        ),
        name="extract_nawm_analysis",
    )

    workflow.connect([
        (create_nawm, extract_nawm_analysis, [("nawm_result", "nawm_result")]),
        (hdglio_prep, combine_qc, [("qc_metrics", "prep_qc")]),
        (hdglio_segment, combine_qc, [("qc_metrics", "segment_qc")]),
        (extract_nawm_analysis, combine_qc, [("analysis", "nawm_analysis")]),
        (combine_qc, outputnode, [("qc_metrics", "qc_metrics")]),
    ])

    return workflow
