"""Anatomical preprocessing workflow for PMRIPrep.

This module implements the main anatomical preprocessing workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Optional, Union

from nipype.interfaces import utility as niu  # type: ignore[import-untyped]
from nipype.pipeline import engine as pe  # type: ignore[import-untyped]

from pmriprep.config.anat import AnatConfig


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
    2. Bias field correction (N4)
    3. Optional denoising (ANTs DenoiseImage)
    4. Coregistration of other modalities to T1w
    5. Brain extraction (HD-BET or ANTs)
    6. Tissue segmentation (FSL FAST)
    7. Optional ACPC alignment (rigid to MNI)

    Processing steps are implemented in separate interfaces (see US-020 to US-025).
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

    # --- Processing nodes will be added in subsequent user stories ---
    #
    # US-020: ReorientAndCrop (fslreorient2std + robustfov)
    # US-021: BiasCorrection (N4BiasFieldCorrection + optional denoising)
    # US-022: Coregistration (FLIRT to T1w space)
    # US-023: BrainExtraction (HD-BET or ANTs)
    # US-024: TissueSegmentation (FSL FAST)
    # US-025: ACPCAlignment (FLIRT to MNI)
    # US-027: Wire all nodes together

    # Placeholder pass-through connection for initial skeleton
    # This will be replaced with proper node connections in US-027
    workflow.connect([
        (inputnode, outputnode, [
            ("t1w", "t1w_preproc"),  # Placeholder: pass input to output
        ]),
    ])

    # Store config in workflow for reference by child nodes
    workflow.config_anat = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
