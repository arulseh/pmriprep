"""DSC (Dynamic Susceptibility Contrast) preprocessing workflow for PMRIPrep.

This module implements the main DSC perfusion processing workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Optional, Union

from nipype.interfaces import utility as niu  # type: ignore[import-untyped]
from nipype.pipeline import engine as pe  # type: ignore[import-untyped]

from pmriprep.config.dsc import DSCConfig


def init_dsc_preproc_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[DSCConfig] = None,
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
    subject_id : str
        BIDS subject identifier (without 'sub-' prefix).
    session_id : str, optional
        BIDS session identifier (without 'ses-' prefix).
    config : DSCConfig, optional
        DSC processing configuration. Uses defaults if not provided.
    name : str
        Workflow name (default: "dsc_preproc_wf").

    Returns
    -------
    pe.Workflow
        Nipype workflow for DSC perfusion processing.

    Workflow Inputs
    ---------------
    inputnode.dsc_file : str
        Path to raw 4D DSC perfusion image.
    inputnode.t1w_reference : str
        Path to T1-weighted reference image for registration.
    inputnode.brain_mask : str
        Path to brain mask in T1w space.
    inputnode.wm_probseg : str, optional
        Path to white matter probability map for NAWM mask.
    inputnode.lesion_mask : str, optional
        Path to lesion mask for registration cost weighting and AIF exclusion.
    inputnode.nawm_mask : str, optional
        Path to pre-computed NAWM mask for CBV normalization.

    Workflow Outputs
    ----------------
    outputnode.dsc_moco : str
        Path to motion-corrected DSC image.
    outputnode.dsc_mean : str
        Path to mean DSC image across time.
    outputnode.motion_params : str
        Path to motion parameters file.
    outputnode.cbf_map : str
        Path to cerebral blood flow map.
    outputnode.cbv_map : str
        Path to cerebral blood volume map.
    outputnode.mtt_map : str
        Path to mean transit time map.
    outputnode.tmax_map : str
        Path to time-to-maximum map.
    outputnode.psr_map : str
        Path to percent signal recovery map.
    outputnode.ph_map : str
        Path to peak height map.
    outputnode.aif_curve : str
        Path to detected/used AIF curve.
    outputnode.aif_mask : str
        Path to AIF detection mask.
    outputnode.k2_map : str, optional
        Path to K2 leakage correction map (if save_k2_map=True).
    outputnode.ncbv_map : str, optional
        Path to normalized CBV map (if NAWM mask available).
    outputnode.dsc_to_t1w_xfm : str
        Path to DSC to T1w transform.
    outputnode.qc_metrics : dict
        Dictionary of QC metrics from processing steps.

    Notes
    -----
    The workflow follows this processing flow:
    1. Motion correction (mcflirt)
    2. Signal-to-concentration conversion (ΔR2*)
    3. Preload detection and K2 leakage correction (if needed)
    4. AIF detection (atlas-based → automatic → synthetic fallback)
    5. Deconvolution (method from config: sSVD, cSVD, oSVD, tikhonov, vascular, cpi)
    6. Perfusion parameter extraction (CBF, CBV, MTT, Tmax, PSR, PH)
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
                "brain_mask",  # Brain mask in T1w space
                "wm_probseg",  # White matter probability map (optional)
                "lesion_mask",  # Lesion mask (optional)
                "nawm_mask",  # NAWM mask for normalization (optional)
            ]
        ),
        name="inputnode",
    )

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
                # Perfusion parameter maps
                "cbf_map",
                "cbv_map",
                "mtt_map",
                "tmax_map",
                "psr_map",
                "ph_map",
                # AIF outputs
                "aif_curve",
                "aif_mask",
                # Leakage correction
                "k2_map",
                # Normalized maps
                "ncbv_map",
                # Registration
                "dsc_to_t1w_xfm",
                # QC metrics
                "qc_metrics",
            ]
        ),
        name="outputnode",
    )

    # -------------------------------------------------------------------
    # Processing Nodes (to be implemented in subsequent user stories)
    # -------------------------------------------------------------------
    # The following nodes will be added in subsequent user stories:
    #
    # US-029: MotionCorrection node (mcflirt)
    # US-030: SignalToConcentration node (ΔR2* conversion)
    # US-031: PSR and peak height computation
    # US-032: PreloadDetection node
    # US-033: K2LeakageCorrection node
    # US-034-038: AIFDetection node and utilities
    # US-039-044: Deconvolution nodes (sSVD, cSVD, oSVD, tikhonov, etc.)
    # US-052: ComputeDSCMaps node
    # US-053: DSCRegistration node
    # US-054: CBV normalization
    #
    # The workflow wiring will be completed in US-055.

    # Store config in workflow for reference by child nodes
    workflow.config_dsc = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
