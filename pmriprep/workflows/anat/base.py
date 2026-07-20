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


def _merge_qc_metrics(
    brain_qc: dict[str, object] | None,
    denoising: bool = False,
) -> dict[str, object]:
    """Merge QC metrics from preprocessing steps.

    Parameters
    ----------
    brain_qc : dict
        QC metrics from brain extraction.
    denoising : bool
        Whether denoising was applied during preprocessing.

    Returns
    -------
    dict
        Merged QC metrics dictionary.
    """
    merged: dict[str, object] = {}

    # Add brain extraction QC
    if brain_qc:
        merged["brain_extraction"] = brain_qc

    # Add preprocessing settings
    merged["preprocessing"] = {
        "denoising": denoising,
    }

    return merged


def _determine_processing_mode(
    t1w_precon: str | None = None,
    t1w_postcon: str | None = None,
) -> tuple[str, bool]:
    """Determine reference T1w and processing mode using confidence scoring.

    Uses confidence-based structural-reference selection and returns the
    reference image with its contrast state.

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


def _create_brain_mask_filmstrip(
    t1w_file: str,
    brain_mask_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
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

    from pmriprep.utils.qc_utils import (
        plot_segmentation_qc_filmstrip,
        qc_figure_title,
    )

    out_dir = Path.cwd()
    output_path = str(out_dir / "brain_mask_qc_filmstrip.png")

    t1w_img = nib.load(t1w_file)
    t1w_data = t1w_img.get_fdata()  # type: ignore[attr-defined]
    mask_data = nib.load(brain_mask_file).get_fdata().astype(bool)  # type: ignore[attr-defined]
    voxel_sizes = tuple(t1w_img.header.get_zooms()[:3])  # type: ignore[attr-defined]

    # Pass full binary mask — contour at level=0.5 gives solid line
    label_masks = [("Brain mask", mask_data.astype(np.float64), "#EE6677")]

    plot_segmentation_qc_filmstrip(
        background=t1w_data,
        label_masks=label_masks,
        output_path=output_path,
        mask=mask_data,
        title=qc_figure_title(subject_id, session_id, "Brain Extraction QC"),
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _create_acpc_registration_filmstrip(
    t1w_acpc_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
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
    from pathlib import Path

    import nibabel as nib

    from pmriprep.data import get_mni_brain_template
    from pmriprep.utils.qc_utils import (
        plot_registration_qc_filmstrip,
        qc_figure_title,
    )

    out_dir = Path.cwd()
    output_path = str(out_dir / "acpc_qc_filmstrip.png")

    t1w_img = nib.as_closest_canonical(nib.load(t1w_acpc_file))
    t1w_data = t1w_img.get_fdata()
    voxel_sizes = tuple(t1w_img.header.get_zooms()[:3])

    mni_path = str(get_mni_brain_template())
    mni_data = nib.as_closest_canonical(nib.load(mni_path)).get_fdata()

    # Mask from the T1w data (already brain-extracted)
    mask = t1w_data > 0

    plot_registration_qc_filmstrip(
        fixed=t1w_data,
        moving=mni_data,
        output_path=output_path,
        mask=mask,
        title=qc_figure_title(subject_id, session_id, "ACPC-MNI Alignment QC"),
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _create_coreg_qc_filmstrip(
    coregistered_file: str,
    reference_file: str,
    brain_mask_file: str,
    title: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
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

    from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip, qc_figure_title

    # Use a sanitized title for the filename
    safe_name = title.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path.cwd()
    output_path = str(out_dir / f"coreg_qc_{safe_name}.png")

    coreg_img = nib.load(coregistered_file)
    coreg_data = coreg_img.get_fdata()  # type: ignore[attr-defined]
    ref_data = nib.load(reference_file).get_fdata()  # type: ignore[attr-defined]
    mask_data = nib.load(brain_mask_file).get_fdata()  # type: ignore[attr-defined]
    voxel_sizes = tuple(coreg_img.header.get_zooms()[:3])  # type: ignore[attr-defined]

    plot_registration_qc_filmstrip(
        fixed=coreg_data,
        moving=ref_data,
        output_path=output_path,
        mask=mask_data,
        title=qc_figure_title(subject_id, session_id, title),
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _create_tissue_seg_filmstrip(
    t1w_file: str,
    wm_probseg_file: str,
    gm_probseg_file: str,
    csf_probseg_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
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

    from pmriprep.utils.qc_utils import (
        plot_segmentation_qc_filmstrip,
        qc_figure_title,
    )

    out_dir = Path.cwd()
    output_path = str(out_dir / "tissue_seg_qc_filmstrip.png")

    t1w_img = nib.load(t1w_file)
    t1w_data = t1w_img.get_fdata()  # type: ignore[attr-defined]
    voxel_sizes = tuple(t1w_img.header.get_zooms()[:3])  # type: ignore[attr-defined]

    # Threshold probability maps at 0.5 to get binary masks
    wm_mask = (nib.load(wm_probseg_file).get_fdata() > 0.5).astype(np.float64)  # type: ignore[attr-defined]
    csf_mask = (nib.load(csf_probseg_file).get_fdata() > 0.5).astype(np.float64)  # type: ignore[attr-defined]

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
        title=qc_figure_title(subject_id, session_id, "Tissue Segmentation QC"),
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _passthrough_if_undefined(in_file: Optional[str] = None) -> str:
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


def _route_fast_input(
    in_file: str | None,
    t1w_precon: str | None = None,
    t1w_postcon: str | None = None,
    is_postcon_ref: bool = False,
) -> str | None:
    """Route FAST input based on processing mode and precon quality.

    Normal mode: FAST on brain-extracted precon (in_file).
    Hybrid mode: returns None here — hybrid path wired separately.
    Skip mode: returns None (no acceptable precon for FAST).
    """
    import logging

    _log = logging.getLogger("pmriprep.workflows.anat")

    if not is_postcon_ref:
        # Normal mode: FAST on brain-extracted precon (passed as in_file)
        return in_file

    # Postcon-only mode: check if precon is available for hybrid FAST
    from pmriprep.utils.structural_reference import _determine_processing_mode_v2

    _, _, fast_t1w = _determine_processing_mode_v2(t1w_precon, t1w_postcon)

    if fast_t1w is None:
        _log.warning(
            "FAST segmentation skipped — no acceptable pre-contrast T1w. "
            "Tissue probability maps unavailable. Downstream features "
            "requiring WM/GM/CSF segmentation (BBR registration, NAWM "
            "normalization, CSF exclusion) will be limited.",
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


def _guarded_tissue_segment(
    in_file: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Run FAST tissue segmentation, or return None triplet if skipped.

    Returns ``(None, None, None)`` without a segmentation input. Otherwise
    returns WM, GM, and CSF probability maps.
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


def _run_hybrid_fast_chain(
    t1w_precon: str | None = None,
    t1w_postcon: str | None = None,
    is_postcon_ref: bool = False,
    reference_file: str | None = None,
    brain_mask: str | None = None,
    denoise: bool = True,
) -> str | None:
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
    masked: str = ApplyMask(
        in_file=coregistered, mask_file=brain_mask,
    ).run().outputs.out_file
    return masked


def _merge_fast_inputs(
    normal_input: str | None,
    hybrid_input: str | None,
) -> str | None:
    """Select whichever FAST input is available."""
    if normal_input is not None:
        return normal_input
    return hybrid_input


def init_anat_tissue_segmentation_wf(
    *,
    denoise: bool,
    name: str = "anat_tissue_segmentation_wf",
) -> pe.Workflow:
    """Initialize conditional anatomical tissue segmentation.

    Parameters
    ----------
    denoise : bool
        Whether to denoise the precontrast image in the hybrid FAST chain.
    name : str
        Workflow name.

    Returns
    -------
    pe.Workflow
        Workflow exposing WM, GM, and CSF probability maps.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "in_file", "t1w_precon", "t1w_postcon", "is_postcon_ref",
            "reference_file", "brain_mask",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["wm_probseg", "gm_probseg", "csf_probseg"]),
        name="outputnode",
    )
    check_fast_mode = pe.Node(
        niu.Function(
            input_names=["in_file", "t1w_precon", "t1w_postcon", "is_postcon_ref"],
            output_names=["out_file"],
            function=_route_fast_input,
        ),
        name="check_fast_mode",
    )
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
    hybrid_fast_chain.inputs.denoise = denoise
    merge_fast = pe.Node(
        niu.Function(
            input_names=["normal_input", "hybrid_input"],
            output_names=["out_file"],
            function=_merge_fast_inputs,
        ),
        name="merge_fast",
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
        (inputnode, check_fast_mode, [
            ("in_file", "in_file"),
            ("t1w_precon", "t1w_precon"),
            ("t1w_postcon", "t1w_postcon"),
            ("is_postcon_ref", "is_postcon_ref"),
        ]),
        (inputnode, hybrid_fast_chain, [
            ("t1w_precon", "t1w_precon"),
            ("t1w_postcon", "t1w_postcon"),
            ("is_postcon_ref", "is_postcon_ref"),
            ("reference_file", "reference_file"),
            ("brain_mask", "brain_mask"),
        ]),
        (check_fast_mode, merge_fast, [("out_file", "normal_input")]),
        (hybrid_fast_chain, merge_fast, [("out_file", "hybrid_input")]),
        (merge_fast, tissue_segment, [("out_file", "in_file")]),
        (tissue_segment, outputnode, [
            ("wm_probseg", "wm_probseg"),
            ("gm_probseg", "gm_probseg"),
            ("csf_probseg", "csf_probseg"),
        ]),
    ])
    return workflow


def init_anat_secondary_modalities_wf(
    *,
    config: AnatConfig,
    has_t1w_postcon: bool,
    has_t2w: bool,
    has_flair: bool,
    has_flair_postcon: bool,
    name: str = "anat_secondary_modalities_wf",
) -> pe.Workflow:
    """Initialize preprocessing for optional anatomical modalities.

    Parameters
    ----------
    config : AnatConfig
        Anatomical preprocessing configuration.
    has_t1w_postcon, has_t2w, has_flair, has_flair_postcon : bool
        Construction-time availability flags for each secondary modality.
    name : str
        Workflow name.

    Returns
    -------
    pe.Workflow
        Workflow exposing reoriented, coregistered, masked, and transform outputs.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "reference", "brain_mask", "t1w_postcon", "t2w", "flair",
            "flair_postcon",
        ]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "t1w_postcon_reoriented", "t1w_postcon_coreg", "t1w_postcon_xfm",
            "t1w_postcon_h5", "t1w_postcon_masked", "t2w_reoriented",
            "t2w_coreg", "t2w_xfm", "t2w_h5", "t2w_masked",
            "flair_reoriented", "flair_coreg", "flair_xfm", "flair_h5",
            "flair_masked", "flair_postcon_reoriented", "flair_postcon_coreg",
            "flair_postcon_xfm", "flair_postcon_h5", "flair_postcon_masked",
        ]),
        name="outputnode",
    )
    workflow.add_nodes([inputnode, outputnode])
    bias_kwargs: dict[str, Any] = {"denoise": config.denoise}
    if config.sloppy:
        bias_kwargs.update(
            shrink_factor=8, convergence="[20x20,0.001]", denoise=False,
        )

    modalities = [
        ("postcon", "t1w_postcon", has_t1w_postcon, "T1_postcon", True),
        ("t2w", "t2w", has_t2w, "T2w", False),
        ("flair", "flair", has_flair, "FLAIR", False),
        ("flair_postcon", "flair_postcon", has_flair_postcon, "FLAIR", False),
    ]
    for suffix, field, enabled, modality, crop_fov in modalities:
        if not enabled:
            continue
        reorient_kwargs = {} if crop_fov else {"crop_fov": False}
        reorient = pe.Node(
            ReorientAndCrop(**reorient_kwargs), name=f"reorient_{suffix}"
        )
        bias_correct = pe.Node(
            BiasCorrection(**bias_kwargs), name=f"bias_correct_{suffix}"
        )
        coreg = pe.Node(
            Coregistration(modality=modality, dof=6), name=f"coreg_{suffix}"
        )
        mask = pe.Node(ApplyMask(), name=f"mask_{suffix}")
        workflow.connect([
            (inputnode, reorient, [(field, "in_file")]),
            (reorient, bias_correct, [("out_file", "in_file")]),
            (bias_correct, coreg, [("out_file", "in_file")]),
            (inputnode, coreg, [("reference", "reference")]),
            (coreg, mask, [("out_file", "in_file")]),
            (inputnode, mask, [("brain_mask", "mask_file")]),
            (reorient, outputnode, [("out_file", f"{field}_reoriented")]),
            (coreg, outputnode, [
                ("out_file", f"{field}_coreg"),
                ("out_matrix", f"{field}_xfm"),
                ("out_h5", f"{field}_h5"),
            ]),
            (mask, outputnode, [("out_file", f"{field}_masked")]),
        ])
    return workflow


def init_anat_preproc_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[AnatConfig] = None,
    has_t1w_postcon: bool = False,
    has_t2w: bool = False,
    has_flair: bool = False,
    has_flair_postcon: bool = False,
    mem_limit_gb: Optional[float] = None,
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
        Whether pre-contrast FLAIR image is available.
    has_flair_postcon : bool
        Whether post-contrast FLAIR image is available.
    name : str
        Workflow name (default: "anat_preproc_wf").

    Returns
    -------
    pe.Workflow
        Nipype workflow for anatomical preprocessing.

    Inputs
    ------
    t1w_precon
        Path to T1-weighted pre-contrast image.
    t1w_postcon
        Path to post-contrast T1-weighted image.
    t2w
        Path to T2-weighted image.
    flair
        Path to FLAIR image.

    Outputs
    -------
    t1w_preproc
        Path to preprocessed T1-weighted image (bias-corrected, brain-extracted).
    brain_mask
        Path to brain mask in T1w space.
    wm_probseg
        Path to white matter probability map (None if postcon is reference).
    gm_probseg
        Path to gray matter probability map (None if postcon is reference).
    csf_probseg
        Path to CSF probability map (None if postcon is reference).
    t1w_acpc
        Path to ACPC-aligned T1w (if not skipped).
    acpc_xfm
        Path to ACPC alignment transform.
    t1w_to_mni_xfm
        Path to T1w to MNI transform.
    t1w_postcon_coreg
        T1w post-contrast coregistered to pre-contrast.
    t2w_coreg
        T2w coregistered to T1w.
    flair_coreg
        FLAIR coregistered to T1w.
    t1w_postcon_to_t1w_xfm
        T1w postcon to T1w transform.
    t2w_to_t1w_xfm
        T2w to T1w transform.
    flair_to_t1w_xfm
        FLAIR to T1w transform.
    t1w_subtraction
        T1w subtraction image (post - pre).
    is_postcon_ref
        Flag indicating postcon is the structural reference.
    qc_metrics
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
                "flair_postcon",
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
                "t1w_to_mni_h5",
                # Coregistered modalities
                "t1w_postcon_coreg",
                "t2w_coreg",
                "flair_coreg",
                "flair_postcon_coreg",
                # Coregistration transforms
                "t1w_postcon_to_t1w_xfm",
                "t2w_to_t1w_xfm",
                "flair_to_t1w_xfm",
                "flair_postcon_to_t1w_xfm",
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
                "coreg_filmstrip_flair_postcon",
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
    bias_kwargs: dict[str, Any] = {"denoise": config.denoise}
    if config.sloppy:
        bias_kwargs.update(
            shrink_factor=8, convergence="[20x20,0.001]", denoise=False,
        )

    bias_correct = pe.Node(
        BiasCorrection(**bias_kwargs),
        name="bias_correct",
    )

    workflow.connect([
        (reorient, bias_correct, [("out_file", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # Step 2b: Conform T1w — downsample sub-millimeter voxels to 1mm
    # Prevents excessive memory/compute on high-res acquisitions (e.g. 0.49mm).
    # Passthrough if voxels are already >= 1mm.
    # -------------------------------------------------------------------
    from pmriprep.interfaces.anat import ConformT1w

    conform = pe.Node(ConformT1w(), name="conform")
    # Raw conform for HD-GLIO: same grid, no bias correction
    conform_raw = pe.Node(ConformT1w(), name="conform_raw")
    workflow.connect([
        (bias_correct, conform, [("out_file", "in_file")]),
        (reorient, conform_raw, [("out_file", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # Step 3-4: Secondary modality preprocessing and coregistration
    # Only create nodes for modalities that are available
    # -------------------------------------------------------------------

    secondary_modalities_wf = init_anat_secondary_modalities_wf(
        config=config,
        has_t1w_postcon=has_t1w_postcon,
        has_t2w=has_t2w,
        has_flair=has_flair,
        has_flair_postcon=has_flair_postcon,
    )
    workflow.connect([
        (inputnode, secondary_modalities_wf, [
            ("t1w_postcon", "inputnode.t1w_postcon"),
            ("t2w", "inputnode.t2w"),
            ("flair", "inputnode.flair"),
            ("flair_postcon", "inputnode.flair_postcon"),
        ]),
        (conform, secondary_modalities_wf, [
            ("out_file", "inputnode.reference"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Step 5: Brain Extraction
    # Extract brain using HD-BET (default) or ANTs
    # -------------------------------------------------------------------
    # Resolve device at build time for nipype GPU scheduling
    from pmriprep.utils.device import resolve_device_uses_gpu
    from pmriprep.utils.mem import cap_mem_gb

    brain_extract = pe.Node(
        BrainExtraction(
            tool=config.tool,
            device=config.hdbet_device,
            conda_env=config.hdbet_conda_env or "",
            use_gpu=resolve_device_uses_gpu(config.hdbet_device),
        ),
        name="brain_extract",
        mem_gb=cap_mem_gb(2.0, mem_limit_gb),
    )

    workflow.connect([
        (conform, brain_extract, [("out_file", "in_file")]),
    ])

    # -------------------------------------------------------------------
    # Step 6: Apply brain mask to coregistered images
    # Only create mask nodes for modalities that exist
    # -------------------------------------------------------------------

    workflow.connect([
        (brain_extract, secondary_modalities_wf, [
            ("out_mask", "inputnode.brain_mask"),
        ]),
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
            (conform_raw, tumor_seg_mask_t1w, [("out_file", "in_file")]),
            (brain_extract, tumor_seg_mask_t1w, [("out_mask", "mask_file")]),
            (tumor_seg_mask_t1w, outputnode, [("out_file", "tumor_seg_t1w_pre")]),
        ])

        # T1w postcon: reoriented → apply coreg xfm → mask
        tumor_seg_xfm_postcon = pe.Node(
            ApplyTransform(), name="tumor_seg_xfm_postcon"
        )
        tumor_seg_mask_postcon = pe.Node(ApplyMask(), name="tumor_seg_mask_postcon")
        workflow.connect([
            (secondary_modalities_wf, tumor_seg_xfm_postcon, [
                ("outputnode.t1w_postcon_reoriented", "in_file"),
            ]),
            (conform_raw, tumor_seg_xfm_postcon, [("out_file", "reference")]),
            (secondary_modalities_wf, tumor_seg_xfm_postcon, [
                ("outputnode.t1w_postcon_xfm", "in_matrix"),
            ]),
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
            (secondary_modalities_wf, tumor_seg_xfm_t2w, [
                ("outputnode.t2w_reoriented", "in_file"),
            ]),
            (conform_raw, tumor_seg_xfm_t2w, [("out_file", "reference")]),
            (secondary_modalities_wf, tumor_seg_xfm_t2w, [
                ("outputnode.t2w_xfm", "in_matrix"),
            ]),
            (tumor_seg_xfm_t2w, tumor_seg_mask_t2w, [("out_file", "in_file")]),
            (brain_extract, tumor_seg_mask_t2w, [("out_mask", "mask_file")]),
            (tumor_seg_mask_t2w, outputnode, [("out_file", "tumor_seg_t2w")]),
        ])

        # FLAIR: reoriented → apply coreg xfm → mask
        tumor_seg_xfm_flair = pe.Node(ApplyTransform(), name="tumor_seg_xfm_flair")
        tumor_seg_mask_flair = pe.Node(ApplyMask(), name="tumor_seg_mask_flair")
        workflow.connect([
            (secondary_modalities_wf, tumor_seg_xfm_flair, [
                ("outputnode.flair_reoriented", "in_file"),
            ]),
            (conform_raw, tumor_seg_xfm_flair, [("out_file", "reference")]),
            (secondary_modalities_wf, tumor_seg_xfm_flair, [
                ("outputnode.flair_xfm", "in_matrix"),
            ]),
            (tumor_seg_xfm_flair, tumor_seg_mask_flair, [("out_file", "in_file")]),
            (brain_extract, tumor_seg_mask_flair, [("out_mask", "mask_file")]),
            (tumor_seg_mask_flair, outputnode, [("out_file", "tumor_seg_flair")]),
        ])

    # -------------------------------------------------------------------
    # Step 7: Tissue Segmentation (conditional on precon availability)
    # Segment brain into WM/GM/CSF using FSL FAST
    # SKIPPED in postcon-only mode (gadolinium disrupts tissue boundaries)
    # -------------------------------------------------------------------

    tissue_segmentation_wf = init_anat_tissue_segmentation_wf(
        denoise=config.denoise,
    )
    workflow.connect([
        (brain_extract, tissue_segmentation_wf, [
            ("out_file", "inputnode.in_file"),
            ("out_mask", "inputnode.brain_mask"),
        ]),
        (inputnode, tissue_segmentation_wf, [
            ("t1w_precon", "inputnode.t1w_precon"),
            ("t1w_postcon", "inputnode.t1w_postcon"),
        ]),
        (determine_mode, tissue_segmentation_wf, [
            ("is_postcon_ref", "inputnode.is_postcon_ref"),
        ]),
        (conform, tissue_segmentation_wf, [
            ("out_file", "inputnode.reference_file"),
        ]),
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
            (secondary_modalities_wf, create_subtraction, [
                ("outputnode.t1w_postcon_coreg", "postcon_file"),
            ]),
            (conform, create_subtraction, [("out_file", "precon_file")]),
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
            input_names=["brain_qc", "denoising"],
            output_names=["qc_metrics"],
            function=_merge_qc_metrics,
        ),
        name="merge_qc",
    )
    merge_qc.inputs.denoising = config.denoise

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
            input_names=["t1w_file", "brain_mask_file", "subject_id", "session_id"],
            output_names=["filmstrip_file"],
            function=_create_brain_mask_filmstrip,
        ),
        name="brain_mask_filmstrip",
    )
    brain_mask_filmstrip.inputs.subject_id = subject_id
    brain_mask_filmstrip.inputs.session_id = session_id or ""
    workflow.connect([
        (conform, brain_mask_filmstrip, [("out_file", "t1w_file")]),
        (brain_extract, brain_mask_filmstrip, [("out_mask", "brain_mask_file")]),
    ])

    # Tissue segmentation filmstrip (only when FAST runs, i.e. not postcon-only)
    tissue_seg_filmstrip_node = pe.Node(
        niu.Function(
            input_names=["t1w_file", "wm_probseg_file", "gm_probseg_file",
                         "csf_probseg_file", "subject_id", "session_id"],
            output_names=["filmstrip_file"],
            function=_create_tissue_seg_filmstrip,
        ),
        name="tissue_seg_filmstrip",
    )
    tissue_seg_filmstrip_node.inputs.subject_id = subject_id
    tissue_seg_filmstrip_node.inputs.session_id = session_id or ""
    workflow.connect([
        (brain_extract, tissue_seg_filmstrip_node, [("out_file", "t1w_file")]),
        (tissue_segmentation_wf, tissue_seg_filmstrip_node, [
            ("outputnode.wm_probseg", "wm_probseg_file"),
            ("outputnode.gm_probseg", "gm_probseg_file"),
            ("outputnode.csf_probseg", "csf_probseg_file"),
        ]),
    ])

    # ACPC alignment filmstrip
    acpc_filmstrip_node = pe.Node(
        niu.Function(
            input_names=["t1w_acpc_file", "subject_id", "session_id"],
            output_names=["filmstrip_file"],
            function=_create_acpc_registration_filmstrip,
        ),
        name="acpc_filmstrip",
    )
    acpc_filmstrip_node.inputs.subject_id = subject_id
    acpc_filmstrip_node.inputs.session_id = session_id or ""
    workflow.connect([
        (acpc_align, acpc_filmstrip_node, [("out_file", "t1w_acpc_file")]),
    ])

    # Cross-modal coregistration filmstrips (one per available modality)
    coreg_qc_postcon = None
    coreg_qc_t2w = None
    coreg_qc_flair = None
    coreg_qc_flair_postcon = None
    if has_t1w_postcon:
        coreg_qc_postcon = pe.Node(
            niu.Function(
                input_names=["coregistered_file", "reference_file",
                             "brain_mask_file", "title", "subject_id",
                             "session_id"],
                output_names=["filmstrip_file"],
                function=_create_coreg_qc_filmstrip,
            ),
            name="coreg_qc_postcon",
        )
        coreg_qc_postcon.inputs.title = "T1w PostCon to T1w"
        coreg_qc_postcon.inputs.subject_id = subject_id
        coreg_qc_postcon.inputs.session_id = session_id or ""
        workflow.connect([
            (secondary_modalities_wf, coreg_qc_postcon, [
                ("outputnode.t1w_postcon_masked", "coregistered_file"),
            ]),
            (brain_extract, coreg_qc_postcon, [
                ("out_file", "reference_file"),
                ("out_mask", "brain_mask_file"),
            ]),
        ])

    if has_t2w:
        coreg_qc_t2w = pe.Node(
            niu.Function(
                input_names=["coregistered_file", "reference_file",
                             "brain_mask_file", "title", "subject_id",
                             "session_id"],
                output_names=["filmstrip_file"],
                function=_create_coreg_qc_filmstrip,
            ),
            name="coreg_qc_t2w",
        )
        coreg_qc_t2w.inputs.title = "T2w to T1w"
        coreg_qc_t2w.inputs.subject_id = subject_id
        coreg_qc_t2w.inputs.session_id = session_id or ""
        workflow.connect([
            (secondary_modalities_wf, coreg_qc_t2w, [
                ("outputnode.t2w_masked", "coregistered_file"),
            ]),
            (brain_extract, coreg_qc_t2w, [
                ("out_file", "reference_file"),
                ("out_mask", "brain_mask_file"),
            ]),
        ])

    if has_flair:
        coreg_qc_flair = pe.Node(
            niu.Function(
                input_names=["coregistered_file", "reference_file",
                             "brain_mask_file", "title", "subject_id",
                             "session_id"],
                output_names=["filmstrip_file"],
                function=_create_coreg_qc_filmstrip,
            ),
            name="coreg_qc_flair",
        )
        coreg_qc_flair.inputs.title = "FLAIR to T1w"
        coreg_qc_flair.inputs.subject_id = subject_id
        coreg_qc_flair.inputs.session_id = session_id or ""
        workflow.connect([
            (secondary_modalities_wf, coreg_qc_flair, [
                ("outputnode.flair_masked", "coregistered_file"),
            ]),
            (brain_extract, coreg_qc_flair, [
                ("out_file", "reference_file"),
                ("out_mask", "brain_mask_file"),
            ]),
        ])

    if has_flair_postcon:
        coreg_qc_flair_postcon = pe.Node(
            niu.Function(
                input_names=["coregistered_file", "reference_file",
                             "brain_mask_file", "title", "subject_id",
                             "session_id"],
                output_names=["filmstrip_file"],
                function=_create_coreg_qc_filmstrip,
            ),
            name="coreg_qc_flair_postcon",
        )
        coreg_qc_flair_postcon.inputs.title = "FLAIR PostCon to T1w"
        coreg_qc_flair_postcon.inputs.subject_id = subject_id
        coreg_qc_flair_postcon.inputs.session_id = session_id or ""
        workflow.connect([
            (secondary_modalities_wf, coreg_qc_flair_postcon, [
                ("outputnode.flair_postcon_masked", "coregistered_file"),
            ]),
            (brain_extract, coreg_qc_flair_postcon, [
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
        (tissue_segmentation_wf, outputnode, [
            ("outputnode.wm_probseg", "wm_probseg"),
            ("outputnode.gm_probseg", "gm_probseg"),
            ("outputnode.csf_probseg", "csf_probseg"),
        ]),
        # ACPC alignment outputs
        (acpc_align, outputnode, [
            ("out_file", "t1w_acpc"),
            ("out_matrix", "acpc_xfm"),
            # ACPC transform: .mat for internal consumers, .h5 for derivatives
            ("out_matrix", "t1w_to_mni_xfm"),
            ("out_h5", "t1w_to_mni_h5"),
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
    if coreg_qc_flair_postcon is not None:
        workflow.connect([
            (coreg_qc_flair_postcon, outputnode, [
                ("filmstrip_file", "coreg_filmstrip_flair_postcon"),
            ]),
        ])
    # Coregistered modalities and transforms (only for available modalities)
    if has_t1w_postcon:
        workflow.connect([
            (secondary_modalities_wf, outputnode, [
                ("outputnode.t1w_postcon_masked", "t1w_postcon_coreg"),
                ("outputnode.t1w_postcon_h5", "t1w_postcon_to_t1w_xfm"),
            ]),
        ])
    if has_t2w:
        workflow.connect([
            (secondary_modalities_wf, outputnode, [
                ("outputnode.t2w_masked", "t2w_coreg"),
                ("outputnode.t2w_h5", "t2w_to_t1w_xfm"),
            ]),
        ])
    if has_flair:
        workflow.connect([
            (secondary_modalities_wf, outputnode, [
                ("outputnode.flair_masked", "flair_coreg"),
                ("outputnode.flair_h5", "flair_to_t1w_xfm"),
            ]),
        ])
    if has_flair_postcon:
        workflow.connect([
            (secondary_modalities_wf, outputnode, [
                ("outputnode.flair_postcon_masked", "flair_postcon_coreg"),
                ("outputnode.flair_postcon_h5", "flair_postcon_to_t1w_xfm"),
            ]),
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
