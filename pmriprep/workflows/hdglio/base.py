"""HD-GLIO tumor segmentation workflow for PMRIPrep.

This module implements the HD-GLIO deep learning tumor segmentation workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Optional, Union

from nipype.interfaces import utility as niu  # type: ignore[import-untyped]
from nipype.pipeline import engine as pe  # type: ignore[import-untyped]

from pmriprep.config.hdglio import HDGLIOConfig


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
    # Processing nodes will be added in US-078 to US-081
    # Node placeholders for documentation:
    #
    # Step 1: HD-GLIO Prep (US-078)
    # - Minimal preprocessing: brain extraction only (NO bias correction)
    # - Verify all 4 modalities present and dimension-matched
    #
    # Step 2: HD-GLIO Segmentation (US-079)
    # - Run hd_glio_predict with config.device (cuda/mps/cpu)
    # - Output labels: 0=background, 1=non-enhancing, 2=enhancing
    #
    # Step 3: Mask Extraction (US-080)
    # - Extract enhancing, non-enhancing, and whole tumor masks
    #
    # Step 4: Mask Dilation (US-080)
    # - Dilate lesion mask with config.lesion_dilation (default: 15mm)
    #
    # Step 5: NAWM ROI Creation (US-081)
    # - Create contralateral NAWM mask for CBV normalization
    # - Use WM probability > 0.9, exclude tumor + 20mm margin
    # - Apply config.nawm_erosion (default: 2 voxels)
    # - Ensure >= config.nawm_min_voxels (default: 500)
    #
    # Wiring will be completed in US-082
    # -------------------------------------------------------------------

    # Note: Actual node definitions and connections will be added in US-078-082
    # For now, the workflow skeleton defines the input/output interface

    return workflow
