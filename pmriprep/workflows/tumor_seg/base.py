"""Tumor segmentation workflow for PMRIPrep.

This module implements the HD-GLIO deep learning tumor segmentation workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Any, Optional, Union

from nipype.interfaces import utility as niu  # type: ignore[import-untyped]
from nipype.pipeline import engine as pe  # type: ignore[import-untyped]

from pmriprep.config.tumor_seg import TumorSegConfig
from pmriprep.data import get_lobar_atlas_dir
from pmriprep.interfaces.hdglio import HDGLIOPrep, HDGLIOSegment


def _select_enhancing(masks: dict[str, str]) -> str:
    """Select enhancing mask path from masks dict."""
    return masks["enhancing"]


def _select_non_enhancing(masks: dict[str, str]) -> str:
    """Select non-enhancing mask path from masks dict."""
    return masks["non_enhancing"]


def _select_whole_tumor(masks: dict[str, str]) -> str:
    """Select whole tumor mask path from masks dict."""
    return masks["whole_tumor"]


def _extract_nawm_mask_path(nawm_result):  # type: ignore[no-untyped-def]
    """Extract NAWM mask file path from result dict."""
    return str(nawm_result["output_file"])


def _extract_nawm_analysis_dict(nawm_result):  # type: ignore[no-untyped-def]
    """Extract NAWM analysis sub-dict from result dict."""
    return dict(nawm_result["analysis"])


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
    import os

    from pmriprep.utils.tumor_seg import extract_tumor_masks

    # Write to this node's working directory (os.getcwd() in Nipype),
    # not the input file's directory, so outputs survive upstream re-runs.
    return extract_tumor_masks(
        segmentation_file=segmentation_file,
        output_dir=os.getcwd(),
        prefix=prefix,
    )


def _create_nawm_node(tumor_mask_file, wm_probseg_file, nawm_erosion, nawm_min_voxels, t1w_to_mni_xfm='', reference_image='', atlases_dir='', brain_mask_file=''):  # type: ignore[no-untyped-def]
    """Node function to create NAWM ROI.

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
    t1w_to_mni_xfm : str
        Path to T1w-to-MNI transform for atlas-based restriction
    reference_image : str
        Path to T1w reference image for atlas transformation
    atlases_dir : str
        Path to MNI lobar atlas directory
    brain_mask_file : str
        Path to brain mask for base constraint

    Returns
    -------
    dict
        Dictionary with output file path and analysis details
    """
    from pmriprep.utils.tumor_seg import create_nawm_roi

    return create_nawm_roi(
        tumor_mask_file=tumor_mask_file,
        wm_probseg_file=wm_probseg_file,
        output_file=None,  # Auto-generate filename
        nawm_erosion=nawm_erosion,
        nawm_min_voxels=nawm_min_voxels,
        t1w_to_mni_xfm=t1w_to_mni_xfm or None,
        reference_image=reference_image or None,
        atlases_dir=atlases_dir or None,
        brain_mask_file=brain_mask_file or None,
    )


def _create_tumor_on_postcon_filmstrip(
    t1w_postcon_file: str,
    enhancing_mask_file: str,
    non_enhancing_mask_file: str,
    brain_mask_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Create tumor segmentation filmstrip on T1w post-contrast background."""
    import nibabel as nib
    import numpy as np
    import os

    from pmriprep.utils.qc_utils import plot_segmentation_qc_filmstrip, qc_figure_title

    bg_img = nib.load(t1w_postcon_file)
    bg_data = np.asarray(bg_img.dataobj)  # type: ignore[attr-defined]
    brain_mask = np.asarray(nib.load(brain_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    enh = np.asarray(nib.load(enhancing_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    non_enh = np.asarray(nib.load(non_enhancing_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    _zooms = bg_img.header.get_zooms()  # type: ignore[attr-defined]
    voxel_sizes: tuple[float, float, float] = (
        float(_zooms[0]), float(_zooms[1]), float(_zooms[2]),
    )

    output_path = os.path.join(os.getcwd(), "seg_filmstrip_postcon.png")
    plot_segmentation_qc_filmstrip(
        background=bg_data,
        label_masks=[
            ("Non-enhancing", non_enh, "#228833"),
            ("Enhancing", enh, "#EE6677"),
        ],
        output_path=output_path,
        mask=brain_mask,
        title=qc_figure_title(subject_id, session_id, "Tumor Segmentation on T1w Post-Contrast"),
        voxel_sizes=voxel_sizes,
    )
    return output_path


def _create_tumor_on_flair_filmstrip(
    flair_file: str,
    enhancing_mask_file: str,
    non_enhancing_mask_file: str,
    brain_mask_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Create tumor segmentation filmstrip on FLAIR background."""
    import nibabel as nib
    import numpy as np
    import os

    from pmriprep.utils.qc_utils import plot_segmentation_qc_filmstrip, qc_figure_title

    bg_img = nib.load(flair_file)
    bg_data = np.asarray(bg_img.dataobj)  # type: ignore[attr-defined]
    brain_mask = np.asarray(nib.load(brain_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    enh = np.asarray(nib.load(enhancing_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    non_enh = np.asarray(nib.load(non_enhancing_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    _zooms = bg_img.header.get_zooms()  # type: ignore[attr-defined]
    voxel_sizes: tuple[float, float, float] = (
        float(_zooms[0]), float(_zooms[1]), float(_zooms[2]),
    )

    output_path = os.path.join(os.getcwd(), "seg_filmstrip_flair.png")
    plot_segmentation_qc_filmstrip(
        background=bg_data,
        label_masks=[
            ("Non-enhancing", non_enh, "#228833"),
            ("Enhancing", enh, "#EE6677"),
        ],
        output_path=output_path,
        mask=brain_mask,
        title=qc_figure_title(subject_id, session_id, "Tumor Segmentation on FLAIR"),
        voxel_sizes=voxel_sizes,
    )
    return output_path


def _create_nawm_on_flair_filmstrip(
    flair_file: str,
    nawm_mask_file: str,
    whole_tumor_mask_file: str,
    brain_mask_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Create NAWM ROI filmstrip on FLAIR background."""
    import nibabel as nib
    import numpy as np
    import os

    from pmriprep.utils.qc_utils import plot_segmentation_qc_filmstrip, qc_figure_title

    bg_img = nib.load(flair_file)
    bg_data = np.asarray(bg_img.dataobj)  # type: ignore[attr-defined]
    brain_mask = np.asarray(nib.load(brain_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    nawm = np.asarray(nib.load(nawm_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    tumor = np.asarray(nib.load(whole_tumor_mask_file).dataobj) > 0  # type: ignore[attr-defined]
    _zooms = bg_img.header.get_zooms()  # type: ignore[attr-defined]
    voxel_sizes: tuple[float, float, float] = (
        float(_zooms[0]), float(_zooms[1]), float(_zooms[2]),
    )

    output_path = os.path.join(os.getcwd(), "nawm_filmstrip_flair.png")
    plot_segmentation_qc_filmstrip(
        background=bg_data,
        label_masks=[
            ("NAWM", nawm, "#66CCEE"),
            ("Whole Tumor", tumor, "#EE6677"),
        ],
        output_path=output_path,
        mask=brain_mask,
        title=qc_figure_title(subject_id, session_id, "NAWM ROI on FLAIR"),
        voxel_sizes=voxel_sizes,
    )
    return output_path


def _generate_qc_metrics(prep_qc, segment_qc, nawm_analysis):  # type: ignore[no-untyped-def]
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


def init_tumor_seg_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[TumorSegConfig] = None,
    mem_limit_gb: Optional[float] = None,
    name: str = "tumor_seg_wf",
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
    config : TumorSegConfig, optional
        Tumor segmentation configuration. Uses defaults if not provided.
    name : str
        Workflow name (default: "tumor_seg_wf").

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
    inputnode.t1w_to_mni_xfm : str, optional
        Path to T1w-to-MNI transform for atlas-based NAWM restriction.
    inputnode.t1w_preproc : str, optional
        Path to preprocessed T1w image (reference for atlas transformation).

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
    4. Create contralateral NAWM ROI for CBV normalization

    HD-GLIO requires all 4 modalities: T1w-pre, T1w-ce-GAD, T2w, FLAIR.
    """
    # Use default config if not provided
    if config is None:
        config = TumorSegConfig()

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
    # Optional: t1w_to_mni_xfm and t1w_preproc for atlas-based NAWM restriction
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w_pre",
                "t1w_postcon",
                "t2w",
                "flair",
                "brain_mask",
                "wm_probseg",
                "t1w_to_mni_xfm",
                "t1w_preproc",
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
                "nawm_mask",
                "qc_metrics",
                "seg_filmstrip_postcon",
                "seg_filmstrip_flair",
                "nawm_filmstrip_flair",
            ]
        ),
        name="outputnode",
    )

    # Store config in workflow for reference by child nodes
    workflow.config_tumor_seg = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    # -------------------------------------------------------------------
    # Step 1: HD-GLIO Preprocessing (US-078)
    # - Minimal preprocessing: brain extraction only (NO bias correction)
    # - Verify all 4 modalities present and dimension-matched
    # -------------------------------------------------------------------
    tumor_seg_prep = pe.Node(
        HDGLIOPrep(),
        name="tumor_seg_prep",
    )

    # Connect inputs to prep node
    workflow.connect([
        (inputnode, tumor_seg_prep, [
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
    # Resolve device at build time for nipype GPU scheduling
    from pmriprep.utils.device import resolve_device_uses_gpu

    segment_kwargs: dict[str, Any] = {
        "device": config.device,
        "use_gpu": resolve_device_uses_gpu(config.device),
    }
    if config.conda_env:
        segment_kwargs["conda_env"] = config.conda_env
    from pmriprep.utils.mem import cap_mem_gb

    tumor_seg_segment = pe.Node(
        HDGLIOSegment(**segment_kwargs),
        name="tumor_seg_segment",
        mem_gb=cap_mem_gb(4.0, mem_limit_gb),
    )

    # Connect prepped images to segmentation node
    workflow.connect([
        (tumor_seg_prep, tumor_seg_segment, [
            ("t1w_pre_prepped", "t1w_pre"),
            ("t1w_postcon_prepped", "t1w_postcon"),
            ("t2w_prepped", "t2w"),
            ("flair_prepped", "flair"),
        ]),
    ])

    # Connect segmentation output to workflow outputnode
    workflow.connect([
        (tumor_seg_segment, outputnode, [("segmentation", "segmentation")]),
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
        (tumor_seg_segment, extract_masks, [("segmentation", "segmentation_file")]),
    ])

    # Extract individual mask paths from the masks dict using Function nodes
    # NOTE: Named functions are used instead of lambdas because Nipype
    # serializes Function node callables for hashing/caching, and lambdas
    # do not survive this process reliably.
    select_enhancing = pe.Node(
        niu.Function(
            input_names=["masks"],
            output_names=["mask_path"],
            function=_select_enhancing,
        ),
        name="select_enhancing",
    )
    select_non_enhancing = pe.Node(
        niu.Function(
            input_names=["masks"],
            output_names=["mask_path"],
            function=_select_non_enhancing,
        ),
        name="select_non_enhancing",
    )
    select_whole_tumor = pe.Node(
        niu.Function(
            input_names=["masks"],
            output_names=["mask_path"],
            function=_select_whole_tumor,
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
    # Step 4: NAWM ROI Creation (US-081)
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
                "t1w_to_mni_xfm",
                "reference_image",
                "atlases_dir",
                "brain_mask_file",
            ],
            output_names=["nawm_result"],
            function=_create_nawm_node,
        ),
        name="create_nawm",
    )
    create_nawm.inputs.nawm_erosion = config.nawm_erosion
    create_nawm.inputs.nawm_min_voxels = config.nawm_min_voxels
    create_nawm.inputs.atlases_dir = str(get_lobar_atlas_dir())

    # Connect whole tumor mask and WM probseg to NAWM creation
    # Also connect MNI transform and T1w reference for atlas-based restriction
    workflow.connect([
        (select_whole_tumor, create_nawm, [("mask_path", "tumor_mask_file")]),
        (inputnode, create_nawm, [
            ("wm_probseg", "wm_probseg_file"),
            ("t1w_to_mni_xfm", "t1w_to_mni_xfm"),
            ("t1w_preproc", "reference_image"),
            ("brain_mask", "brain_mask_file"),
        ]),
    ])

    # Extract NAWM mask path from result dict
    extract_nawm_path = pe.Node(
        niu.Function(
            input_names=["nawm_result"],
            output_names=["nawm_mask"],
            function=_extract_nawm_mask_path,
        ),
        name="extract_nawm_path",
    )

    workflow.connect([
        (create_nawm, extract_nawm_path, [("nawm_result", "nawm_result")]),
        (extract_nawm_path, outputnode, [("nawm_mask", "nawm_mask")]),
    ])

    # -------------------------------------------------------------------
    # Step 5: Combine QC Metrics
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
            function=_extract_nawm_analysis_dict,
        ),
        name="extract_nawm_analysis",
    )

    workflow.connect([
        (create_nawm, extract_nawm_analysis, [("nawm_result", "nawm_result")]),
        (tumor_seg_prep, combine_qc, [("qc_metrics", "prep_qc")]),
        (tumor_seg_segment, combine_qc, [("qc_metrics", "segment_qc")]),
        (extract_nawm_analysis, combine_qc, [("analysis", "nawm_analysis")]),
        (combine_qc, outputnode, [("qc_metrics", "qc_metrics")]),
    ])

    # -------------------------------------------------------------------
    # Step 6: QC Filmstrips
    # -------------------------------------------------------------------

    # Tumor on T1w post-contrast
    tumor_postcon_filmstrip = pe.Node(
        niu.Function(
            input_names=[
                "t1w_postcon_file",
                "enhancing_mask_file",
                "non_enhancing_mask_file",
                "brain_mask_file",
                "subject_id",
                "session_id",
            ],
            output_names=["filmstrip_path"],
            function=_create_tumor_on_postcon_filmstrip,
        ),
        name="tumor_postcon_filmstrip",
    )
    tumor_postcon_filmstrip.inputs.subject_id = subject_id
    tumor_postcon_filmstrip.inputs.session_id = session_id or ""

    workflow.connect([
        (inputnode, tumor_postcon_filmstrip, [
            ("t1w_postcon", "t1w_postcon_file"),
            ("brain_mask", "brain_mask_file"),
        ]),
        (select_enhancing, tumor_postcon_filmstrip, [
            ("mask_path", "enhancing_mask_file"),
        ]),
        (select_non_enhancing, tumor_postcon_filmstrip, [
            ("mask_path", "non_enhancing_mask_file"),
        ]),
        (tumor_postcon_filmstrip, outputnode, [
            ("filmstrip_path", "seg_filmstrip_postcon"),
        ]),
    ])

    # Tumor on FLAIR
    tumor_flair_filmstrip = pe.Node(
        niu.Function(
            input_names=[
                "flair_file",
                "enhancing_mask_file",
                "non_enhancing_mask_file",
                "brain_mask_file",
                "subject_id",
                "session_id",
            ],
            output_names=["filmstrip_path"],
            function=_create_tumor_on_flair_filmstrip,
        ),
        name="tumor_flair_filmstrip",
    )
    tumor_flair_filmstrip.inputs.subject_id = subject_id
    tumor_flair_filmstrip.inputs.session_id = session_id or ""

    workflow.connect([
        (inputnode, tumor_flair_filmstrip, [
            ("flair", "flair_file"),
            ("brain_mask", "brain_mask_file"),
        ]),
        (select_enhancing, tumor_flair_filmstrip, [
            ("mask_path", "enhancing_mask_file"),
        ]),
        (select_non_enhancing, tumor_flair_filmstrip, [
            ("mask_path", "non_enhancing_mask_file"),
        ]),
        (tumor_flair_filmstrip, outputnode, [
            ("filmstrip_path", "seg_filmstrip_flair"),
        ]),
    ])

    # NAWM on FLAIR
    nawm_flair_filmstrip = pe.Node(
        niu.Function(
            input_names=[
                "flair_file",
                "nawm_mask_file",
                "whole_tumor_mask_file",
                "brain_mask_file",
                "subject_id",
                "session_id",
            ],
            output_names=["filmstrip_path"],
            function=_create_nawm_on_flair_filmstrip,
        ),
        name="nawm_flair_filmstrip",
    )
    nawm_flair_filmstrip.inputs.subject_id = subject_id
    nawm_flair_filmstrip.inputs.session_id = session_id or ""

    workflow.connect([
        (inputnode, nawm_flair_filmstrip, [
            ("flair", "flair_file"),
            ("brain_mask", "brain_mask_file"),
        ]),
        (extract_nawm_path, nawm_flair_filmstrip, [
            ("nawm_mask", "nawm_mask_file"),
        ]),
        (select_whole_tumor, nawm_flair_filmstrip, [
            ("mask_path", "whole_tumor_mask_file"),
        ]),
        (nawm_flair_filmstrip, outputnode, [
            ("filmstrip_path", "nawm_filmstrip_flair"),
        ]),
    ])

    return workflow
