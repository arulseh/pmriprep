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
    ApplyTransform,
    BiasCorrection,
    BrainExtraction,
    Coregistration,
    ReorientAndCrop,
    T1wSubtraction,
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


def _determine_processing_mode(t1w_precon=None, t1w_postcon=None):
    """Determine reference T1w and processing mode using confidence scoring.

    Uses _determine_processing_mode_v2 from structural_reference module for
    physics-validated confidence-based selection. Returns backward-compatible
    2-tuple (reference_t1w, is_postcon_ref) for the workflow node interface.

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
    is_postcon_ref : bool
        True if post-contrast T1w is the reference.
    """
    from pmriprep.utils.structural_reference import _determine_processing_mode_v2

    reference_t1w, is_postcon_ref, _fast_t1w = _determine_processing_mode_v2(
        t1w_precon, t1w_postcon,
    )
    return reference_t1w, is_postcon_ref


def _create_brain_mask_filmstrip(t1w_file, brain_mask_file):
    """Create brain extraction QC filmstrip showing mask boundary on T1w.

    Loads bias-corrected T1w as background and overlays the brain mask
    contour (solid red line at level=0.5) to show whether the mask boundary
    follows the cortical surface. Uses the full binary mask with
    ``plot_segmentation_qc_filmstrip`` for a clean solid contour matching
    fMRIPrep style.

    Parameters
    ----------
    t1w_file : str
        Path to bias-corrected T1w image (pre-extraction).
    brain_mask_file : str
        Path to brain mask.

    Returns
    -------
    str
        Path to output filmstrip PNG.
    """
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from pmriprep.utils.qc_utils import plot_segmentation_qc_filmstrip

    out_dir = Path.cwd()
    output_path = str(out_dir / "brain_mask_qc_filmstrip.png")

    t1w_img = nib.load(t1w_file)
    t1w_data = t1w_img.get_fdata()
    mask_data = nib.load(brain_mask_file).get_fdata().astype(bool)
    voxel_sizes = tuple(t1w_img.header.get_zooms()[:3])

    # Pass full binary mask — contour at level=0.5 gives solid line
    label_masks = [("Brain mask", mask_data.astype(np.float64), "#EE6677")]

    plot_segmentation_qc_filmstrip(
        background=t1w_data,
        label_masks=label_masks,
        output_path=output_path,
        mask=mask_data,
        title="Brain Extraction QC",
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _create_acpc_registration_filmstrip(t1w_acpc_file):
    """Create ACPC-to-MNI registration QC filmstrip.

    Shows ACPC-aligned T1w as background with MNI template edges overlaid
    to assess the quality of the rigid alignment.

    Parameters
    ----------
    t1w_acpc_file : str
        Path to ACPC-aligned T1w (on MNI 1mm grid).

    Returns
    -------
    str
        Path to output filmstrip PNG.
    """
    import os
    from pathlib import Path

    import nibabel as nib

    from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip

    out_dir = Path.cwd()
    output_path = str(out_dir / "acpc_qc_filmstrip.png")

    t1w_img = nib.as_closest_canonical(nib.load(t1w_acpc_file))
    t1w_data = t1w_img.get_fdata()
    voxel_sizes = tuple(t1w_img.header.get_zooms()[:3])

    # Load MNI152 template as the reference
    fsldir = os.environ.get("FSLDIR", "/usr/local/fsl")
    mni_path = os.path.join(fsldir, "data", "standard", "MNI152_T1_1mm_brain.nii.gz")
    mni_data = nib.as_closest_canonical(nib.load(mni_path)).get_fdata()

    # Mask from the T1w data (already brain-extracted)
    mask = t1w_data > 0

    plot_registration_qc_filmstrip(
        fixed=t1w_data,
        moving=mni_data,
        output_path=output_path,
        mask=mask,
        title="ACPC-MNI Alignment QC",
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _create_coreg_qc_filmstrip(coregistered_file, reference_file, brain_mask_file, title):
    """Create cross-modal coregistration QC filmstrip.

    Shows masked coregistered image as background with brain-extracted T1w
    structural edges overlaid for visual QC of cross-modal registration.

    Parameters
    ----------
    coregistered_file : str
        Path to coregistered and masked modality image (background).
    reference_file : str
        Path to brain-extracted T1w reference (edge overlay).
    brain_mask_file : str
        Path to brain mask.
    title : str
        Title for the filmstrip (e.g. "T1w PostCon-to-T1w").

    Returns
    -------
    str
        Path to output filmstrip PNG.
    """
    from pathlib import Path

    import nibabel as nib

    from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip

    # Use a sanitized title for the filename
    safe_name = title.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path.cwd()
    output_path = str(out_dir / f"coreg_qc_{safe_name}.png")

    coreg_img = nib.load(coregistered_file)
    coreg_data = coreg_img.get_fdata()
    ref_data = nib.load(reference_file).get_fdata()
    mask_data = nib.load(brain_mask_file).get_fdata()
    voxel_sizes = tuple(coreg_img.header.get_zooms()[:3])

    plot_registration_qc_filmstrip(
        fixed=coreg_data,
        moving=ref_data,
        output_path=output_path,
        mask=mask_data,
        title=title,
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _create_tissue_seg_filmstrip(t1w_file, wm_probseg_file, gm_probseg_file, csf_probseg_file):
    """Create tissue segmentation QC filmstrip showing WM/GM/CSF contours on T1w.

    Thresholds each probability map at p>0.5 to create binary masks,
    then overlays colored contour edges on brain-extracted T1w.

    Parameters
    ----------
    t1w_file : str
        Path to brain-extracted T1w image (background).
    wm_probseg_file : str or None
        Path to white matter probability map.
    gm_probseg_file : str or None
        Path to gray matter probability map.
    csf_probseg_file : str or None
        Path to CSF probability map.

    Returns
    -------
    str or None
        Path to output filmstrip PNG, or None if tissue maps are unavailable.
    """
    if wm_probseg_file is None:
        return None

    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from pmriprep.utils.qc_utils import plot_segmentation_qc_filmstrip

    out_dir = Path.cwd()
    output_path = str(out_dir / "tissue_seg_qc_filmstrip.png")

    t1w_img = nib.load(t1w_file)
    t1w_data = t1w_img.get_fdata()
    voxel_sizes = tuple(t1w_img.header.get_zooms()[:3])

    # Threshold probability maps at 0.5 to get binary masks
    wm_mask = (nib.load(wm_probseg_file).get_fdata() > 0.5).astype(np.float64)
    csf_mask = (nib.load(csf_probseg_file).get_fdata() > 0.5).astype(np.float64)

    # Two boundary lines separate three compartments:
    # WM contour = WM/GM interface, CSF contour = GM/CSF interface.
    # GM is implicit (region between the two boundaries).
    label_masks = [
        ("WM", wm_mask, "#4477AA"),
        ("CSF", csf_mask, "#EE6677"),
    ]

    plot_segmentation_qc_filmstrip(
        background=t1w_data,
        label_masks=label_masks,
        output_path=output_path,
        mask=(t1w_data > 0),
        title="Tissue Segmentation QC",
        voxel_sizes=voxel_sizes,
    )

    return output_path


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
    inputnode.t1w_precon : str, optional
        Path to T1-weighted pre-contrast image.
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
        Path to white matter probability map (None if postcon is reference).
    outputnode.gm_probseg : str
        Path to gray matter probability map (None if postcon is reference).
    outputnode.csf_probseg : str
        Path to CSF probability map (None if postcon is reference).
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
    outputnode.is_postcon_ref : bool
        Flag indicating postcon is the structural reference.
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

    When postcon is the reference:
    - T1w post-contrast is used as the structural reference
    - Hybrid FAST: if precon meets quality gate, it is coregistered to
      postcon space for tissue segmentation; otherwise FAST is skipped
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
                "t1w_precon",
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
                # Tumor segmentation prepared inputs (no bias correction)
                "tumor_seg_t1w_pre",
                "tumor_seg_t1w_postcon",
                "tumor_seg_t2w",
                "tumor_seg_flair",
                # Processing mode flag
                "is_postcon_ref",
                # QC
                "qc_metrics",
                # QC filmstrips
                "brain_mask_filmstrip",
                "acpc_filmstrip",
                "coreg_filmstrip_postcon",
                "coreg_filmstrip_t2w",
                "coreg_filmstrip_flair",
                "coreg_filmstrip_pdw",
                "tissue_seg_filmstrip",
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
            output_names=["reference_t1w", "is_postcon_ref"],
            function=_determine_processing_mode,
        ),
        name="determine_mode",
    )

    workflow.connect([
        (inputnode, determine_mode, [
            ("t1w_precon", "t1w_precon"),
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

    # T1w post-contrast preprocessing and coregistration
    if has_t1w_postcon:
        reorient_postcon = pe.Node(ReorientAndCrop(), name="reorient_postcon")
        bias_correct_postcon = pe.Node(
            BiasCorrection(denoise=config.denoise), name="bias_correct_postcon"
        )
        coreg_postcon = pe.Node(
            Coregistration(modality="T1_postcon", dof=6), name="coreg_postcon"
        )
        coreg_postcon_conns = [
            (inputnode, reorient_postcon, [("t1w_postcon", "in_file")]),
            (reorient_postcon, bias_correct_postcon, [("out_file", "in_file")]),
            (bias_correct_postcon, coreg_postcon, [("out_file", "in_file")]),
            (bias_correct, coreg_postcon, [("out_file", "reference")]),
        ]
        if config.use_lesion_mask:
            coreg_postcon_conns.append(
                (inputnode, coreg_postcon, [("lesion_mask", "lesion_mask")])
            )
        workflow.connect(coreg_postcon_conns)

    # T2w preprocessing and coregistration
    if has_t2w:
        reorient_t2w = pe.Node(
            ReorientAndCrop(crop_fov=False), name="reorient_t2w"
        )
        bias_correct_t2w = pe.Node(
            BiasCorrection(denoise=config.denoise), name="bias_correct_t2w"
        )
        coreg_t2w = pe.Node(
            Coregistration(modality="T2w", dof=6), name="coreg_t2w"
        )
        coreg_t2w_conns = [
            (inputnode, reorient_t2w, [("t2w", "in_file")]),
            (reorient_t2w, bias_correct_t2w, [("out_file", "in_file")]),
            (bias_correct_t2w, coreg_t2w, [("out_file", "in_file")]),
            (bias_correct, coreg_t2w, [("out_file", "reference")]),
        ]
        if config.use_lesion_mask:
            coreg_t2w_conns.append(
                (inputnode, coreg_t2w, [("lesion_mask", "lesion_mask")])
            )
        workflow.connect(coreg_t2w_conns)

    # FLAIR preprocessing and coregistration
    if has_flair:
        reorient_flair = pe.Node(
            ReorientAndCrop(crop_fov=False), name="reorient_flair"
        )
        bias_correct_flair = pe.Node(
            BiasCorrection(denoise=config.denoise), name="bias_correct_flair"
        )
        coreg_flair = pe.Node(
            Coregistration(modality="FLAIR", dof=6), name="coreg_flair"
        )
        coreg_flair_conns = [
            (inputnode, reorient_flair, [("flair", "in_file")]),
            (reorient_flair, bias_correct_flair, [("out_file", "in_file")]),
            (bias_correct_flair, coreg_flair, [("out_file", "in_file")]),
            (bias_correct, coreg_flair, [("out_file", "reference")]),
        ]
        if config.use_lesion_mask:
            coreg_flair_conns.append(
                (inputnode, coreg_flair, [("lesion_mask", "lesion_mask")])
            )
        workflow.connect(coreg_flair_conns)

    # PDw preprocessing and coregistration
    if has_pdw:
        reorient_pdw = pe.Node(
            ReorientAndCrop(crop_fov=False), name="reorient_pdw"
        )
        bias_correct_pdw = pe.Node(
            BiasCorrection(denoise=config.denoise), name="bias_correct_pdw"
        )
        coreg_pdw = pe.Node(
            Coregistration(modality="PDw", dof=6), name="coreg_pdw"
        )
        coreg_pdw_conns = [
            (inputnode, reorient_pdw, [("pdw", "in_file")]),
            (reorient_pdw, bias_correct_pdw, [("out_file", "in_file")]),
            (bias_correct_pdw, coreg_pdw, [("out_file", "in_file")]),
            (bias_correct, coreg_pdw, [("out_file", "reference")]),
        ]
        if config.use_lesion_mask:
            coreg_pdw_conns.append(
                (inputnode, coreg_pdw, [("lesion_mask", "lesion_mask")])
            )
        workflow.connect(coreg_pdw_conns)

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

    if has_t1w_postcon:
        mask_postcon = pe.Node(ApplyMask(), name="mask_postcon")
        workflow.connect([
            (coreg_postcon, mask_postcon, [("out_file", "in_file")]),
            (brain_extract, mask_postcon, [("out_mask", "mask_file")]),
        ])

    if has_t2w:
        mask_t2w = pe.Node(ApplyMask(), name="mask_t2w")
        workflow.connect([
            (coreg_t2w, mask_t2w, [("out_file", "in_file")]),
            (brain_extract, mask_t2w, [("out_mask", "mask_file")]),
        ])

    if has_flair:
        mask_flair = pe.Node(ApplyMask(), name="mask_flair")
        workflow.connect([
            (coreg_flair, mask_flair, [("out_file", "in_file")]),
            (brain_extract, mask_flair, [("out_mask", "mask_file")]),
        ])

    if has_pdw:
        mask_pdw = pe.Node(ApplyMask(), name="mask_pdw")
        workflow.connect([
            (coreg_pdw, mask_pdw, [("out_file", "in_file")]),
            (brain_extract, mask_pdw, [("out_mask", "mask_file")]),
        ])

    # -------------------------------------------------------------------
    # Step 6b: HD-GLIO input preparation (no bias correction track)
    # Reuse coregistration transforms on reoriented (uncorrected) images,
    # then skull-strip. HD-GLIO expects coregistered, skull-stripped,
    # but NOT bias-corrected images (the DL model was trained on raw).
    # -------------------------------------------------------------------
    if has_t1w_postcon and has_t2w and has_flair:

        # T1w pre: reoriented → mask with brain_extract mask
        tumor_seg_mask_t1w = pe.Node(ApplyMask(), name="tumor_seg_mask_t1w")
        workflow.connect([
            (reorient, tumor_seg_mask_t1w, [("out_file", "in_file")]),
            (brain_extract, tumor_seg_mask_t1w, [("out_mask", "mask_file")]),
            (tumor_seg_mask_t1w, outputnode, [("out_file", "tumor_seg_t1w_pre")]),
        ])

        # T1w postcon: reoriented → apply coreg xfm → mask
        tumor_seg_xfm_postcon = pe.Node(
            ApplyTransform(), name="tumor_seg_xfm_postcon"
        )
        tumor_seg_mask_postcon = pe.Node(ApplyMask(), name="tumor_seg_mask_postcon")
        workflow.connect([
            (reorient_postcon, tumor_seg_xfm_postcon, [("out_file", "in_file")]),
            (reorient, tumor_seg_xfm_postcon, [("out_file", "reference")]),
            (coreg_postcon, tumor_seg_xfm_postcon, [("out_matrix", "in_matrix")]),
            (tumor_seg_xfm_postcon, tumor_seg_mask_postcon, [("out_file", "in_file")]),
            (brain_extract, tumor_seg_mask_postcon, [("out_mask", "mask_file")]),
            (tumor_seg_mask_postcon, outputnode, [
                ("out_file", "tumor_seg_t1w_postcon"),
            ]),
        ])

        # T2w: reoriented → apply coreg xfm → mask
        tumor_seg_xfm_t2w = pe.Node(ApplyTransform(), name="tumor_seg_xfm_t2w")
        tumor_seg_mask_t2w = pe.Node(ApplyMask(), name="tumor_seg_mask_t2w")
        workflow.connect([
            (reorient_t2w, tumor_seg_xfm_t2w, [("out_file", "in_file")]),
            (reorient, tumor_seg_xfm_t2w, [("out_file", "reference")]),
            (coreg_t2w, tumor_seg_xfm_t2w, [("out_matrix", "in_matrix")]),
            (tumor_seg_xfm_t2w, tumor_seg_mask_t2w, [("out_file", "in_file")]),
            (brain_extract, tumor_seg_mask_t2w, [("out_mask", "mask_file")]),
            (tumor_seg_mask_t2w, outputnode, [("out_file", "tumor_seg_t2w")]),
        ])

        # FLAIR: reoriented → apply coreg xfm → mask
        tumor_seg_xfm_flair = pe.Node(ApplyTransform(), name="tumor_seg_xfm_flair")
        tumor_seg_mask_flair = pe.Node(ApplyMask(), name="tumor_seg_mask_flair")
        workflow.connect([
            (reorient_flair, tumor_seg_xfm_flair, [("out_file", "in_file")]),
            (reorient, tumor_seg_xfm_flair, [("out_file", "reference")]),
            (coreg_flair, tumor_seg_xfm_flair, [("out_matrix", "in_matrix")]),
            (tumor_seg_xfm_flair, tumor_seg_mask_flair, [("out_file", "in_file")]),
            (brain_extract, tumor_seg_mask_flair, [("out_mask", "mask_file")]),
            (tumor_seg_mask_flair, outputnode, [("out_file", "tumor_seg_flair")]),
        ])

    # -------------------------------------------------------------------
    # Step 7: Tissue Segmentation (conditional on precon availability)
    # Segment brain into WM/GM/CSF using FSL FAST
    # SKIPPED in postcon-only mode (gadolinium disrupts tissue boundaries)
    # -------------------------------------------------------------------

    # Node to route FAST input: handles normal, hybrid, and skip modes.
    # In hybrid mode (postcon reference + acceptable precon), FAST runs
    # on a coregistered precon image prepared by a parallel path below.
    def _route_fast_input(in_file, t1w_precon=None, t1w_postcon=None, is_postcon_ref=False):
        """Route FAST input based on processing mode and precon quality.

        Normal mode: FAST on brain-extracted precon (in_file).
        Hybrid mode: returns None here — hybrid path wired separately.
        Skip mode: returns None (no acceptable precon for FAST).
        """
        import logging
        import warnings

        _log = logging.getLogger("pmriprep.workflows.anat")

        if not is_postcon_ref:
            # Normal mode: FAST on brain-extracted precon (passed as in_file)
            return in_file

        # Postcon-only mode: check if precon is available for hybrid FAST
        from pmriprep.utils.structural_reference import _determine_processing_mode_v2

        _, _, fast_t1w = _determine_processing_mode_v2(t1w_precon, t1w_postcon)

        if fast_t1w is None:
            warnings.warn(
                "FAST segmentation skipped — no acceptable pre-contrast T1w. "
                "Tissue probability maps unavailable. Downstream features "
                "requiring WM/GM/CSF segmentation (BBR registration, NAWM "
                "normalization, CSF exclusion) will be limited.",
                stacklevel=2,
            )
            return None

        # Hybrid mode: return None here.
        # The hybrid path (coreg precon → mask → FAST) is wired separately.
        # This signals the tissue_segment node to expect input from the
        # hybrid path via the merge node below.
        _log.info(
            "Hybrid FAST mode: postcon is reference, FAST will run on "
            "coregistered precon."
        )
        return None

    check_fast_mode = pe.Node(
        niu.Function(
            input_names=["in_file", "t1w_precon", "t1w_postcon", "is_postcon_ref"],
            output_names=["out_file"],
            function=_route_fast_input,
        ),
        name="check_fast_mode",
    )

    def _guarded_tissue_segment(in_file):
        """Run FAST tissue segmentation, or return None triplet if skipped.

        Returns (None, None, None) when in_file is None (Case 4: postcon
        reference and precon fails quality gate). Otherwise runs
        TissueSegmentation and returns (wm, gm, csf) probability maps.
        """
        if in_file is None:
            return None, None, None
        from pmriprep.interfaces.anat import TissueSegmentation

        result = TissueSegmentation(in_file=in_file).run()
        return (
            result.outputs.wm_probseg,
            result.outputs.gm_probseg,
            result.outputs.csf_probseg,
        )

    tissue_segment = pe.Node(
        niu.Function(
            input_names=["in_file"],
            output_names=["wm_probseg", "gm_probseg", "csf_probseg"],
            function=_guarded_tissue_segment,
        ),
        name="tissue_segment",
    )

    workflow.connect([
        (brain_extract, check_fast_mode, [("out_file", "in_file")]),
        (inputnode, check_fast_mode, [
            ("t1w_precon", "t1w_precon"),
            ("t1w_postcon", "t1w_postcon"),
        ]),
        (determine_mode, check_fast_mode, [("is_postcon_ref", "is_postcon_ref")]),
    ])

    # -------------------------------------------------------------------
    # Step 7b: Hybrid FAST path (postcon reference + acceptable precon)
    # When postcon wins as reference, this single Function node prepares
    # the precon T1w for FAST: reorient → bias correct → coreg to postcon
    # → apply postcon brain mask.  Returns None immediately when the
    # hybrid path should not run (no postcon ref, or precon fails gate),
    # avoiding File(exists=True) crashes on CommandLine nodes.
    # -------------------------------------------------------------------

    def _run_hybrid_fast_chain(
        t1w_precon=None, t1w_postcon=None, is_postcon_ref=False,
        reference_file=None, brain_mask=None, denoise=True,
    ):
        """Run full hybrid FAST preprocessing chain or return None.

        When postcon is reference and precon passes the quality gate,
        run: reorient → bias correct → coregister to postcon → mask.
        Otherwise return None immediately (no CommandLine nodes touched).
        """
        if not is_postcon_ref:
            return None

        from pmriprep.utils.structural_reference import _determine_processing_mode_v2

        _, _, fast_t1w = _determine_processing_mode_v2(t1w_precon, t1w_postcon)
        if fast_t1w is None:
            return None

        from pmriprep.interfaces.anat import (
            ApplyMask,
            BiasCorrection,
            Coregistration,
            ReorientAndCrop,
        )

        reoriented = ReorientAndCrop(in_file=fast_t1w).run().outputs.out_file
        bias_corrected = BiasCorrection(
            in_file=reoriented, denoise=denoise,
        ).run().outputs.out_file
        coregistered = Coregistration(
            in_file=bias_corrected, reference=reference_file,
            modality="T1w", dof=6,
        ).run().outputs.out_file
        masked = ApplyMask(
            in_file=coregistered, mask_file=brain_mask,
        ).run().outputs.out_file
        return masked

    hybrid_fast_chain = pe.Node(
        niu.Function(
            input_names=[
                "t1w_precon", "t1w_postcon", "is_postcon_ref",
                "reference_file", "brain_mask", "denoise",
            ],
            output_names=["out_file"],
            function=_run_hybrid_fast_chain,
        ),
        name="hybrid_fast_chain",
    )
    hybrid_fast_chain.inputs.denoise = config.denoise

    workflow.connect([
        (inputnode, hybrid_fast_chain, [
            ("t1w_precon", "t1w_precon"),
            ("t1w_postcon", "t1w_postcon"),
        ]),
        (determine_mode, hybrid_fast_chain, [
            ("is_postcon_ref", "is_postcon_ref"),
        ]),
        (bias_correct, hybrid_fast_chain, [("out_file", "reference_file")]),
        (brain_extract, hybrid_fast_chain, [("out_mask", "brain_mask")]),
    ])

    # Merge: pick normal FAST input or hybrid FAST input
    def _merge_fast_inputs(normal_input, hybrid_input):
        """Select whichever FAST input is available."""
        if normal_input is not None:
            return normal_input
        return hybrid_input

    merge_fast = pe.Node(
        niu.Function(
            input_names=["normal_input", "hybrid_input"],
            output_names=["out_file"],
            function=_merge_fast_inputs,
        ),
        name="merge_fast",
    )

    workflow.connect([
        (check_fast_mode, merge_fast, [("out_file", "normal_input")]),
        (hybrid_fast_chain, merge_fast, [("out_file", "hybrid_input")]),
        (merge_fast, tissue_segment, [("out_file", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # Step 8: T1w Subtraction (when both pre and post available)
    # -------------------------------------------------------------------

    if config.create_subtraction and has_t1w_postcon:
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
    # Step 9: ACPC Alignment
    # Rigid alignment to MNI template for standardized orientation
    # -------------------------------------------------------------------
    acpc_align = pe.Node(
        ACPCAlignment(),
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
    # QC Filmstrips
    # Generate visual QC overlays for brain extraction, ACPC alignment,
    # and cross-modal coregistrations
    # -------------------------------------------------------------------

    # Brain mask filmstrip (always)
    brain_mask_filmstrip = pe.Node(
        niu.Function(
            input_names=["t1w_file", "brain_mask_file"],
            output_names=["filmstrip_file"],
            function=_create_brain_mask_filmstrip,
        ),
        name="brain_mask_filmstrip",
    )
    workflow.connect([
        (bias_correct, brain_mask_filmstrip, [("out_file", "t1w_file")]),
        (brain_extract, brain_mask_filmstrip, [("out_mask", "brain_mask_file")]),
    ])

    # Tissue segmentation filmstrip (only when FAST runs, i.e. not postcon-only)
    tissue_seg_filmstrip_node = pe.Node(
        niu.Function(
            input_names=["t1w_file", "wm_probseg_file", "gm_probseg_file",
                         "csf_probseg_file"],
            output_names=["filmstrip_file"],
            function=_create_tissue_seg_filmstrip,
        ),
        name="tissue_seg_filmstrip",
    )
    workflow.connect([
        (brain_extract, tissue_seg_filmstrip_node, [("out_file", "t1w_file")]),
        (tissue_segment, tissue_seg_filmstrip_node, [
            ("wm_probseg", "wm_probseg_file"),
            ("gm_probseg", "gm_probseg_file"),
            ("csf_probseg", "csf_probseg_file"),
        ]),
    ])

    # ACPC alignment filmstrip
    acpc_filmstrip_node = pe.Node(
        niu.Function(
            input_names=["t1w_acpc_file"],
            output_names=["filmstrip_file"],
            function=_create_acpc_registration_filmstrip,
        ),
        name="acpc_filmstrip",
    )
    workflow.connect([
        (acpc_align, acpc_filmstrip_node, [("out_file", "t1w_acpc_file")]),
    ])

    # Cross-modal coregistration filmstrips (one per available modality)
    coreg_qc_postcon = None
    coreg_qc_t2w = None
    coreg_qc_flair = None
    coreg_qc_pdw = None

    if has_t1w_postcon:
        coreg_qc_postcon = pe.Node(
            niu.Function(
                input_names=["coregistered_file", "reference_file",
                             "brain_mask_file", "title"],
                output_names=["filmstrip_file"],
                function=_create_coreg_qc_filmstrip,
            ),
            name="coreg_qc_postcon",
        )
        coreg_qc_postcon.inputs.title = "T1w PostCon to T1w"
        workflow.connect([
            (mask_postcon, coreg_qc_postcon, [("out_file", "coregistered_file")]),
            (brain_extract, coreg_qc_postcon, [
                ("out_file", "reference_file"),
                ("out_mask", "brain_mask_file"),
            ]),
        ])

    if has_t2w:
        coreg_qc_t2w = pe.Node(
            niu.Function(
                input_names=["coregistered_file", "reference_file",
                             "brain_mask_file", "title"],
                output_names=["filmstrip_file"],
                function=_create_coreg_qc_filmstrip,
            ),
            name="coreg_qc_t2w",
        )
        coreg_qc_t2w.inputs.title = "T2w to T1w"
        workflow.connect([
            (mask_t2w, coreg_qc_t2w, [("out_file", "coregistered_file")]),
            (brain_extract, coreg_qc_t2w, [
                ("out_file", "reference_file"),
                ("out_mask", "brain_mask_file"),
            ]),
        ])

    if has_flair:
        coreg_qc_flair = pe.Node(
            niu.Function(
                input_names=["coregistered_file", "reference_file",
                             "brain_mask_file", "title"],
                output_names=["filmstrip_file"],
                function=_create_coreg_qc_filmstrip,
            ),
            name="coreg_qc_flair",
        )
        coreg_qc_flair.inputs.title = "FLAIR to T1w"
        workflow.connect([
            (mask_flair, coreg_qc_flair, [("out_file", "coregistered_file")]),
            (brain_extract, coreg_qc_flair, [
                ("out_file", "reference_file"),
                ("out_mask", "brain_mask_file"),
            ]),
        ])

    if has_pdw:
        coreg_qc_pdw = pe.Node(
            niu.Function(
                input_names=["coregistered_file", "reference_file",
                             "brain_mask_file", "title"],
                output_names=["filmstrip_file"],
                function=_create_coreg_qc_filmstrip,
            ),
            name="coreg_qc_pdw",
        )
        coreg_qc_pdw.inputs.title = "PDw to T1w"
        workflow.connect([
            (mask_pdw, coreg_qc_pdw, [("out_file", "coregistered_file")]),
            (brain_extract, coreg_qc_pdw, [
                ("out_file", "reference_file"),
                ("out_mask", "brain_mask_file"),
            ]),
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
        (determine_mode, outputnode, [("is_postcon_ref", "is_postcon_ref")]),
        # QC metrics
        (merge_qc, outputnode, [("qc_metrics", "qc_metrics")]),
        # Brain mask filmstrip (always present)
        (brain_mask_filmstrip, outputnode, [
            ("filmstrip_file", "brain_mask_filmstrip"),
        ]),
        # Tissue segmentation filmstrip (skipped in postcon-only mode via None inputs)
        (tissue_seg_filmstrip_node, outputnode, [
            ("filmstrip_file", "tissue_seg_filmstrip"),
        ]),
    ])

    # ACPC filmstrip
    if acpc_filmstrip_node is not None:
        workflow.connect([
            (acpc_filmstrip_node, outputnode, [
                ("filmstrip_file", "acpc_filmstrip"),
            ]),
        ])

    # Cross-modal coregistration filmstrips
    if coreg_qc_postcon is not None:
        workflow.connect([
            (coreg_qc_postcon, outputnode, [
                ("filmstrip_file", "coreg_filmstrip_postcon"),
            ]),
        ])
    if coreg_qc_t2w is not None:
        workflow.connect([
            (coreg_qc_t2w, outputnode, [
                ("filmstrip_file", "coreg_filmstrip_t2w"),
            ]),
        ])
    if coreg_qc_flair is not None:
        workflow.connect([
            (coreg_qc_flair, outputnode, [
                ("filmstrip_file", "coreg_filmstrip_flair"),
            ]),
        ])
    if coreg_qc_pdw is not None:
        workflow.connect([
            (coreg_qc_pdw, outputnode, [
                ("filmstrip_file", "coreg_filmstrip_pdw"),
            ]),
        ])

    # Coregistered modalities and transforms (only for available modalities)
    if has_t1w_postcon:
        workflow.connect([
            (mask_postcon, outputnode, [("out_file", "t1w_postcon_coreg")]),
            (coreg_postcon, outputnode, [("out_matrix", "t1w_postcon_to_t1w_xfm")]),
        ])
    if has_t2w:
        workflow.connect([
            (mask_t2w, outputnode, [("out_file", "t2w_coreg")]),
            (coreg_t2w, outputnode, [("out_matrix", "t2w_to_t1w_xfm")]),
        ])
    if has_flair:
        workflow.connect([
            (mask_flair, outputnode, [("out_file", "flair_coreg")]),
            (coreg_flair, outputnode, [("out_matrix", "flair_to_t1w_xfm")]),
        ])
    if has_pdw:
        workflow.connect([
            (mask_pdw, outputnode, [("out_file", "pdw_coreg")]),
            (coreg_pdw, outputnode, [("out_matrix", "pdw_to_t1w_xfm")]),
        ])

    # Subtraction image
    if config.create_subtraction and has_t1w_postcon:
        workflow.connect([
            (create_subtraction, outputnode, [("out_file", "t1w_subtraction")]),
        ])

    # Store config in workflow for reference by child nodes
    workflow.config_anat = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
