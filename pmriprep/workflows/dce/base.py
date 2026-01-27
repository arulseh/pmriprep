"""DCE (Dynamic Contrast Enhanced) preprocessing workflow for PMRIPrep.

This module implements the main DCE perfusion processing workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Optional, Union

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.config.dce import DCEConfig


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
                "lesion_mask",
                # VFA T1 mapping inputs
                "vfa_files",
                "vfa_flip_angles",
                # AFI B1+ mapping inputs
                "afi_tr1_file",
                "afi_tr2_file",
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

    # Add input and output nodes to workflow
    # Processing nodes will be added and connected in US-076 (Wire DCE workflow)
    # This includes:
    # - B1 mapping (US-057)
    # - T1 mapping (US-058)
    # - DCE preprocessing (US-071)
    # - AIF detection (US-060, US-061, US-062)
    # - Signal conversion (US-059)
    # - PK modeling (US-064 through US-072)
    # - Registration (US-075)
    # - Tortuosity correction (US-073)
    # - DSC-DCE validation (US-074)

    return workflow
