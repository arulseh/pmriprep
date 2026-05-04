"""Anatomical pipeline derivatives factory.

Replaces the monolithic ``AnatDerivativeWriter`` with a small workflow
that wires one ``PMRIPrepDataSink`` per output (NIfTI + sidecar) plus
an :class:`AnatQCWriter` shim for the QC JSON and figure copies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.interfaces.qc_writers import (
    AnatQCWriter,
    PMRIPrepDataSink,
    write_dataset_description_node,
)
from pmriprep.utils.sidecar_builders import (
    build_anat_mask_sidecar,
    build_anat_preproc_sidecar,
    build_anat_probseg_sidecar,
)


def _build_preproc_meta(  # type: ignore[no-untyped-def]
    description,
    source_paths,
    reference_voxel_source,
    bids_dir,
    subject,
    session,
    skull_stripped=True,
    contrast_agent=None,
    is_postcon_ref=False,
):
    """Nipype-Function wrapper for ``build_anat_preproc_sidecar``."""
    from pmriprep.utils.sidecar_builders import build_anat_preproc_sidecar

    return build_anat_preproc_sidecar(
        description=description,
        source_paths=source_paths,
        reference_voxel_source=reference_voxel_source,
        bids_dir=bids_dir,
        subject=subject,
        session=session or None,
        skull_stripped=skull_stripped,
        contrast_agent=contrast_agent,
        is_postcon_ref=is_postcon_ref,
    )


def _build_mask_meta(  # type: ignore[no-untyped-def]
    description,
    source_paths,
    reference_voxel_source,
    bids_dir,
    subject,
    session,
    is_postcon_ref=False,
):
    """Nipype-Function wrapper for ``build_anat_mask_sidecar``."""
    from pmriprep.utils.sidecar_builders import build_anat_mask_sidecar

    return build_anat_mask_sidecar(
        description=description,
        source_paths=source_paths,
        reference_voxel_source=reference_voxel_source,
        bids_dir=bids_dir,
        subject=subject,
        session=session or None,
        is_postcon_ref=is_postcon_ref,
    )


def _build_probseg_meta(  # type: ignore[no-untyped-def]
    tissue_type,
    source_paths,
    reference_voxel_source,
    bids_dir,
    subject,
    session,
    is_postcon_ref=False,
):
    """Nipype-Function wrapper for ``build_anat_probseg_sidecar``."""
    from pmriprep.utils.sidecar_builders import build_anat_probseg_sidecar

    return build_anat_probseg_sidecar(
        tissue_type=tissue_type,
        source_paths=source_paths,
        reference_voxel_source=reference_voxel_source,
        bids_dir=bids_dir,
        subject=subject,
        session=session or None,
        is_postcon_ref=is_postcon_ref,
    )


def _extract_contrast_agent_node(source_path):  # type: ignore[no-untyped-def]
    """Nipype-Function wrapper for ``extract_contrast_agent``."""
    if not source_path:
        return None
    from pmriprep.utils.sidecar_builders import extract_contrast_agent

    return extract_contrast_agent(source_path)


def _add_one_anat_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    *,
    name: str,
    in_attr: str,
    source_attrs: list[str],
    description: str,
    suffix: str,
    desc_value: Optional[str] = None,
    ce_value: Optional[str] = None,
    label_value: Optional[str] = None,
    tissue_type: Optional[str] = None,
    builder: str = "preproc",
    contrast_agent_attr: Optional[str] = None,
    is_postcon_ref: bool = False,
) -> pe.Node:
    """Wire one DataSink node + its sidecar-builder Function node.

    Returns the DataSink node so callers can join on its ``out_file``.
    """
    # Source-list aggregator (from listed source_attrs)
    src_collector = pe.Node(
        niu.Merge(numinputs=len(source_attrs)),
        name=f"{name}_sources",
    )
    for i, attr in enumerate(source_attrs, start=1):
        workflow.connect(inputnode, attr, src_collector, f"in{i}")

    # Per-output meta_dict builder
    builder_fn: Any
    if builder == "preproc":
        builder_fn = _build_preproc_meta
        builder_inputs = [
            "description",
            "source_paths",
            "reference_voxel_source",
            "bids_dir",
            "subject",
            "session",
            "skull_stripped",
            "contrast_agent",
            "is_postcon_ref",
        ]
    elif builder == "mask":
        builder_fn = _build_mask_meta
        builder_inputs = [
            "description",
            "source_paths",
            "reference_voxel_source",
            "bids_dir",
            "subject",
            "session",
            "is_postcon_ref",
        ]
    elif builder == "probseg":
        builder_fn = _build_probseg_meta
        builder_inputs = [
            "tissue_type",
            "source_paths",
            "reference_voxel_source",
            "bids_dir",
            "subject",
            "session",
            "is_postcon_ref",
        ]
    else:
        raise ValueError(f"Unknown builder: {builder}")

    meta_node = pe.Node(
        niu.Function(
            input_names=builder_inputs,
            output_names=["meta_dict"],
            function=builder_fn,
        ),
        name=f"{name}_meta",
    )
    meta_node.inputs.is_postcon_ref = is_postcon_ref
    if builder == "preproc":
        meta_node.inputs.description = description
        meta_node.inputs.skull_stripped = True
        meta_node.inputs.contrast_agent = None
    elif builder == "mask":
        meta_node.inputs.description = description
    elif builder == "probseg":
        meta_node.inputs.tissue_type = tissue_type

    workflow.connect([
        (src_collector, meta_node, [("out", "source_paths")]),
        (inputnode, meta_node, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            (in_attr, "reference_voxel_source"),
        ]),
    ])

    # Optional contrast-agent extraction
    if builder == "preproc" and contrast_agent_attr:
        ca_node = pe.Node(
            niu.Function(
                input_names=["source_path"],
                output_names=["contrast_agent"],
                function=_extract_contrast_agent_node,
            ),
            name=f"{name}_ca",
        )
        workflow.connect([
            (inputnode, ca_node, [(contrast_agent_attr, "source_path")]),
            (ca_node, meta_node, [("contrast_agent", "contrast_agent")]),
        ])

    # The DataSink itself
    sink = pe.Node(
        PMRIPrepDataSink(
            check_hdr=True,
            compress=True,
            allowed_entities=("space", "label", "ceagent"),
            space="T1w",
            suffix=suffix,
        ),
        name=f"{name}_sink",
    )
    if desc_value is not None:
        sink.inputs.desc = desc_value
    if ce_value is not None:
        sink.inputs.ceagent = ce_value
    if label_value is not None:
        sink.inputs.label = label_value

    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "base_directory"),
            (in_attr, "in_file"),
            (source_attrs[0], "source_file"),
        ]),
        (meta_node, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


def _copy_xfm(  # type: ignore[no-untyped-def]
    in_file,
    output_dir,
    subject,
    session,
    from_space,
    to_space,
):
    """Copy a transform .mat into the derivatives anat/ directory."""
    import shutil
    from pathlib import Path

    out_root = Path(output_dir) / "pmriprep" / f"sub-{subject}"
    if session:
        out_root = out_root / f"ses-{session}"
    out_root = out_root / "anat"
    out_root.mkdir(parents=True, exist_ok=True)

    prefix = f"sub-{subject}"
    if session:
        prefix += f"_ses-{session}"
    fname = f"{prefix}_from-{from_space}_to-{to_space}_mode-image_xfm.mat"
    out_path = out_root / fname
    shutil.copy2(in_file, out_path)
    return str(out_path)


def _add_one_xfm_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    *,
    name: str,
    in_attr: str,
    from_space: str,
    to_space: str,
    source_attr: str,
) -> pe.Node:
    """Wire a Function-node copier for a transform .mat (no sidecar)."""
    sink = pe.Node(
        niu.Function(
            input_names=["in_file", "output_dir", "subject", "session",
                         "from_space", "to_space"],
            output_names=["out_file"],
            function=_copy_xfm,
        ),
        name=f"{name}_sink",
    )
    sink.inputs.from_space = from_space
    sink.inputs.to_space = to_space
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            (in_attr, "in_file"),
        ]),
    ])
    return sink


def init_anat_datasinks_wf(
    *,
    output_dir: str,
    bids_dir: str,
    subject_id: str,
    session_id: Optional[str],
    has_t1w_pre: bool,
    has_t1w_postcon: bool,
    has_t2w: bool,
    has_flair: bool,
    has_flair_postcon: bool,
    has_tissue_seg: bool,
    has_subtraction: bool,
    is_postcon_ref: bool = False,
    name: str = "anat_datasinks_wf",
) -> pe.Workflow:
    """Build the anat-pipeline DerivativesDataSink chain.

    Inputs (on ``inputnode``) mirror the legacy ``AnatDerivativeWriter``
    InputSpec one-for-one so ``base.py`` rewiring is mechanical. Outputs
    expose ``qc_json`` (for the report barrier) and
    ``dataset_description_done`` (for the per-session barrier).

    ``is_postcon_ref`` indicates the postcon T1w was used as the structural
    reference (postcon-only mode). Drives sidecar ``Sources`` selection
    and the ``SpatialReference`` URI basename.

    The anat workflow ALWAYS produces ``t1w_preproc``, ``brain_mask``, and
    ``t1w_to_mni_xfm`` — even in postcon-only mode (the postcon is the
    reference). These three outputs are unconditional.
    """
    has_t1w_anat = has_t1w_pre or has_t1w_postcon
    # The "primary" source for sidecars on the unconditional outputs:
    # precon when present, postcon otherwise.
    primary_source_attr = "source_t1w" if has_t1w_pre else "source_t1w_postcon"
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "output_dir",
                "bids_dir",
                "subject_id",
                "session_id",
                # Source paths
                "source_t1w",
                "source_t1w_postcon",
                "source_t2w",
                "source_flair",
                "source_flair_postcon",
                # Derivative inputs
                "t1w_preproc",
                "brain_mask",
                "csf_probseg",
                "gm_probseg",
                "wm_probseg",
                "t1w_postcon_coreg",
                "t2w_coreg",
                "flair_coreg",
                "flair_postcon_coreg",
                "t1w_subtraction",
                "acpc_xfm",
                "t1w_postcon_to_t1w_xfm",
                "t2w_to_t1w_xfm",
                "flair_to_t1w_xfm",
                "flair_postcon_to_t1w_xfm",
                # QC + figures
                "qc_metrics",
                "brain_mask_filmstrip",
                "tissue_seg_filmstrip",
                "acpc_filmstrip",
                "coreg_filmstrip_postcon",
                "coreg_filmstrip_t2w",
                "coreg_filmstrip_flair",
                "coreg_filmstrip_flair_postcon",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.output_dir = output_dir
    inputnode.inputs.bids_dir = bids_dir
    inputnode.inputs.subject_id = subject_id
    inputnode.inputs.session_id = session_id or ""

    sink_outputs: list[tuple[pe.Node, str]] = []

    # --- T1w preproc (always emitted when anat ran; postcon-derived in
    # postcon-only mode) ---
    if has_t1w_anat:
        if is_postcon_ref:
            preproc_desc = (
                "Preprocessed and brain-extracted post-contrast T1w image in "
                "native space (postcon-only mode; postcon is the structural "
                "reference)"
            )
            preproc_ce = "GAD"
        else:
            preproc_desc = (
                "Preprocessed and brain-extracted T1w image in native space "
                "(pre-contrast)"
            )
            preproc_ce = None
        sink = _add_one_anat_sink(
            workflow, inputnode,
            name="t1w_preproc",
            in_attr="t1w_preproc",
            source_attrs=[primary_source_attr],
            description=preproc_desc,
            suffix="T1w",
            desc_value="preproc",
            ce_value=preproc_ce,
            is_postcon_ref=is_postcon_ref,
        )
        sink_outputs.append((sink, "out_file"))

    # --- Brain mask (always emitted; in postcon-only mode it's on postcon) ---
    if has_t1w_anat:
        sink = _add_one_anat_sink(
            workflow, inputnode,
            name="brain_mask",
            in_attr="brain_mask",
            source_attrs=[primary_source_attr],
            description="Brain mask from HD-BET in native anat space",
            suffix="mask",
            desc_value="brain",
            builder="mask",
            is_postcon_ref=is_postcon_ref,
        )
        sink_outputs.append((sink, "out_file"))

    # --- Tissue probsegs ---
    if has_tissue_seg and has_t1w_pre:
        for tissue in ("CSF", "GM", "WM"):
            sink = _add_one_anat_sink(
                workflow, inputnode,
                name=f"{tissue.lower()}_probseg",
                in_attr=f"{tissue.lower()}_probseg",
                source_attrs=["source_t1w"],
                description="",  # ignored by probseg builder
                suffix="probseg",
                label_value=tissue,
                tissue_type=tissue,
                builder="probseg",
            )
            sink_outputs.append((sink, "out_file"))

    # --- Optional coregistered structurals ---
    # Postcon coreg+xfm only exist when both precon and postcon were acquired
    # (in postcon-only mode the postcon IS the reference, no coreg needed).
    if has_t1w_postcon and has_t1w_pre:
        sink = _add_one_anat_sink(
            workflow, inputnode,
            name="t1w_postcon",
            in_attr="t1w_postcon_coreg",
            source_attrs=["source_t1w_postcon", "source_t1w"],
            description=(
                "Preprocessed and brain-extracted post-contrast T1w image "
                "in native space"
            ),
            suffix="T1w",
            desc_value="preproc",
            ce_value="GAD",
            contrast_agent_attr="source_t1w_postcon",
        )
        sink_outputs.append((sink, "out_file"))

    if has_t2w:
        sink = _add_one_anat_sink(
            workflow, inputnode,
            name="t2w",
            in_attr="t2w_coreg",
            source_attrs=["source_t2w"] + (["source_t1w"] if has_t1w_pre else []),
            description=(
                "Preprocessed and brain-extracted T2w image in native anat space"
            ),
            suffix="T2w",
            desc_value="preproc",
        )
        sink_outputs.append((sink, "out_file"))

    if has_flair:
        sink = _add_one_anat_sink(
            workflow, inputnode,
            name="flair",
            in_attr="flair_coreg",
            source_attrs=["source_flair"] + (["source_t1w"] if has_t1w_pre else []),
            description=(
                "Preprocessed and brain-extracted FLAIR image in native anat space"
            ),
            suffix="FLAIR",
            desc_value="preproc",
        )
        sink_outputs.append((sink, "out_file"))

    if has_flair_postcon:
        sink = _add_one_anat_sink(
            workflow, inputnode,
            name="flair_postcon",
            in_attr="flair_postcon_coreg",
            source_attrs=["source_flair_postcon"] + (["source_t1w"] if has_t1w_pre else []),
            description=(
                "Preprocessed and brain-extracted post-contrast FLAIR image "
                "in native anat space"
            ),
            suffix="FLAIR",
            desc_value="preproc",
            ce_value="GAD",
            contrast_agent_attr="source_flair_postcon",
        )
        sink_outputs.append((sink, "out_file"))

    if has_subtraction:
        # source_file=source_t1w (precon) so DataSink auto-derives without
        # ``ce-`` (subtraction is precon-anchored). source_t1w_postcon goes
        # in Sources to record the postcon contribution.
        sink = _add_one_anat_sink(
            workflow, inputnode,
            name="subtraction",
            in_attr="t1w_subtraction",
            source_attrs=["source_t1w", "source_t1w_postcon"],
            description="Post-contrast minus pre-contrast subtraction map",
            suffix="T1w",
            desc_value="subtraction",
            contrast_agent_attr="source_t1w_postcon",
            is_postcon_ref=is_postcon_ref,
        )
        sink_outputs.append((sink, "out_file"))

    # --- Transforms (no sidecars per BIDS convention) ---
    # ACPC xfm always emitted when anat ran (postcon-only mode also produces it).
    if has_t1w_anat:
        sink = _add_one_xfm_sink(
            workflow, inputnode,
            name="acpc_xfm",
            in_attr="acpc_xfm",
            from_space="T1w",
            to_space="MNI152NLin6Asym",
            source_attr=primary_source_attr,
        )
        sink_outputs.append((sink, "out_file"))

    if has_t1w_postcon and has_t1w_pre:
        sink = _add_one_xfm_sink(
            workflow, inputnode,
            name="t1w_postcon_xfm",
            in_attr="t1w_postcon_to_t1w_xfm",
            from_space="T1wCE",
            to_space="T1w",
            source_attr="source_t1w_postcon",
        )
        sink_outputs.append((sink, "out_file"))

    if has_t2w:
        sink = _add_one_xfm_sink(
            workflow, inputnode,
            name="t2w_xfm",
            in_attr="t2w_to_t1w_xfm",
            from_space="T2w",
            to_space="T1w",
            source_attr="source_t2w",
        )
        sink_outputs.append((sink, "out_file"))

    if has_flair:
        sink = _add_one_xfm_sink(
            workflow, inputnode,
            name="flair_xfm",
            in_attr="flair_to_t1w_xfm",
            from_space="FLAIR",
            to_space="T1w",
            source_attr="source_flair",
        )
        sink_outputs.append((sink, "out_file"))

    if has_flair_postcon:
        sink = _add_one_xfm_sink(
            workflow, inputnode,
            name="flair_postcon_xfm",
            in_attr="flair_postcon_to_t1w_xfm",
            from_space="FLAIRCE",
            to_space="T1w",
            source_attr="source_flair_postcon",
        )
        sink_outputs.append((sink, "out_file"))

    # --- AnatQCWriter shim (QC JSON + figures + warnings derivation) ---
    qc_writer = pe.Node(
        AnatQCWriter(
            has_t1w_pre=has_t1w_pre,
            has_t1w_postcon=has_t1w_postcon,
            has_t2w=has_t2w,
            has_flair=has_flair,
            has_tissue_seg=has_tissue_seg,
        ),
        name="qc_writer",
    )
    workflow.connect([
        (inputnode, qc_writer, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject_id"),
            ("session_id", "session_id"),
            ("qc_metrics", "qc_metrics"),
            ("brain_mask_filmstrip", "brain_mask_filmstrip"),
            ("tissue_seg_filmstrip", "tissue_seg_filmstrip"),
            ("acpc_filmstrip", "acpc_filmstrip"),
            ("coreg_filmstrip_postcon", "coreg_filmstrip_postcon"),
            ("coreg_filmstrip_t2w", "coreg_filmstrip_t2w"),
            ("coreg_filmstrip_flair", "coreg_filmstrip_flair"),
            ("coreg_filmstrip_flair_postcon", "coreg_filmstrip_flair_postcon"),
        ]),
    ])

    # --- dataset_description.json barrier ---
    # Joins on every sink's out_file so it runs exactly once after all writes.
    if sink_outputs:
        merge = pe.Node(
            niu.Merge(numinputs=len(sink_outputs) + 1),
            name="dd_merge",
        )
        for i, (sink, attr) in enumerate(sink_outputs, start=1):
            workflow.connect(sink, attr, merge, f"in{i}")
        workflow.connect(qc_writer, "qc_json", merge, f"in{len(sink_outputs) + 1}")
    else:
        merge = pe.Node(
            niu.Merge(numinputs=1),
            name="dd_merge",
        )
        workflow.connect(qc_writer, "qc_json", merge, "in1")

    dd_node = pe.Node(
        niu.Function(
            input_names=["output_dir", "bids_dir", "_trigger"],
            output_names=["dd_path"],
            function=write_dataset_description_node,
        ),
        name="dataset_description",
    )
    workflow.connect([
        (inputnode, dd_node, [
            ("output_dir", "output_dir"),
            ("bids_dir", "bids_dir"),
        ]),
        (merge, dd_node, [("out", "_trigger")]),
    ])

    # --- outputnode ---
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["qc_json", "out_dir", "dataset_description_done"]),
        name="outputnode",
    )
    workflow.connect([
        (qc_writer, outputnode, [("qc_json", "qc_json")]),
        (dd_node, outputnode, [("dd_path", "dataset_description_done")]),
    ])
    # out_dir is a static path
    out_dir_str = str(
        Path(output_dir) / "pmriprep" / f"sub-{subject_id}"
        / (f"ses-{session_id}" if session_id else "")
    )
    outputnode.inputs.out_dir = out_dir_str

    return workflow
