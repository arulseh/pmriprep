"""Tumor-segmentation pipeline derivatives factory.

Wires one ``PMRIPrepDataSink`` per output (NIfTI + sidecar), a
``_dseg.tsv`` companion writer, and a ``TumorSegFiguresWriter`` for
figure filmstrips. Per-region voxel counts and volumes live in the
``desc-tumorseg_dseg.json`` sidecar (``TumorVolumetrics`` block) and
the companion ``desc-tumorseg_dseg.tsv`` lookup table.
"""
from __future__ import annotations

from typing import Optional

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.interfaces.qc_writers import (
    PMRIPrepDataSink,
    TumorSegFiguresWriter,
    write_dataset_description_node,
)


def _build_dseg_meta(
    source_paths: list[str],
    reference_voxel_source: str,
    bids_dir: str,
    subject: str,
    session: str,
    is_postcon_ref: bool = False,
    qc_metrics: dict | None = None,
) -> dict:
    """Nipype-Function wrapper for ``build_tumor_seg_dseg_sidecar``."""
    from pmriprep.utils.sidecar_builders import build_tumor_seg_dseg_sidecar

    return build_tumor_seg_dseg_sidecar(
        source_paths=source_paths,
        reference_voxel_source=reference_voxel_source,
        bids_dir=bids_dir,
        subject=subject,
        session=session or None,
        is_postcon_ref=is_postcon_ref,
        qc_metrics=qc_metrics if isinstance(qc_metrics, dict) else None,
    )


def _build_mask_meta(
    description: str,
    method: str,
    source_paths: list[str],
    reference_voxel_source: str,
    bids_dir: str,
    subject: str,
    session: str,
    is_postcon_ref: bool = False,
) -> dict:
    """Nipype-Function wrapper for ``build_tumor_seg_mask_sidecar``."""
    from pmriprep.utils.sidecar_builders import build_tumor_seg_mask_sidecar

    return build_tumor_seg_mask_sidecar(
        description=description,
        method=method,
        source_paths=source_paths,
        reference_voxel_source=reference_voxel_source,
        bids_dir=bids_dir,
        subject=subject,
        session=session or None,
        is_postcon_ref=is_postcon_ref,
    )


def _build_nawm_meta(
    source_paths: list[str],
    reference_voxel_source: str,
    bids_dir: str,
    subject: str,
    session: str,
    is_postcon_ref: bool = False,
) -> dict:
    """Nipype-Function wrapper for ``build_tumor_seg_nawm_sidecar``."""
    from pmriprep.utils.sidecar_builders import build_tumor_seg_nawm_sidecar

    return build_tumor_seg_nawm_sidecar(
        source_paths=source_paths,
        reference_voxel_source=reference_voxel_source,
        bids_dir=bids_dir,
        subject=subject,
        session=session or None,
        is_postcon_ref=is_postcon_ref,
    )


# HD-GLIO consumes (T1w-pre, T1w-postcon, T2w, FLAIR-precon). source_t1w is
# listed first so DataSink uses it as ``source_file`` for filename auto-
# derivation — that way mask filenames don't sprout a spurious ``ce-GAD``
# entity from the postcon source.
_HD_GLIO_SOURCE_ATTRS = (
    "source_t1w",
    "source_t1w_postcon",
    "source_t2w",
    "source_flair",
)


def _write_tumor_dseg_tsv_node(
    segmentation: str,
    output_dir: str,
    subject: str,
    session: str,
) -> str:
    """Function-node wrapper for ``write_tumor_dseg_tsv``.

    Lands the TSV next to the existing ``desc-tumorseg_dseg.nii.gz`` so
    the companion JSON sidecar (``_dseg.json``) applies to both files
    per BIDS-Common-Derivatives.
    """
    from pathlib import Path

    from pmriprep.utils.lesion_stats import write_tumor_dseg_tsv

    out_root = Path(output_dir) / "pmriprep" / f"sub-{subject}"
    if session:
        out_root = out_root / f"ses-{session}"
    out_root = out_root / "anat"
    out_root.mkdir(parents=True, exist_ok=True)

    prefix = f"sub-{subject}"
    if session:
        prefix += f"_ses-{session}"
    tsv_path = out_root / f"{prefix}_space-T1w_desc-tumorseg_dseg.tsv"
    return str(write_tumor_dseg_tsv(segmentation, tsv_path))


def _add_dseg_tsv_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
) -> pe.Node:
    """Wire the BIDS-Common-Derivatives ``_dseg.tsv`` companion writer."""
    sink = pe.Node(
        niu.Function(
            input_names=["segmentation", "output_dir", "subject", "session"],
            output_names=["out_file"],
            function=_write_tumor_dseg_tsv_node,
        ),
        name="dseg_tsv_sink",
    )
    workflow.connect([
        (inputnode, sink, [
            ("segmentation", "segmentation"),
            ("output_dir", "output_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
        ]),
    ])
    return sink


def _add_dseg_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    is_postcon_ref: bool,
) -> pe.Node:
    src_collector = pe.Node(
        niu.Merge(numinputs=len(_HD_GLIO_SOURCE_ATTRS)),
        name="dseg_sources",
    )
    for i, attr in enumerate(_HD_GLIO_SOURCE_ATTRS, start=1):
        workflow.connect(inputnode, attr, src_collector, f"in{i}")

    meta = pe.Node(
        niu.Function(
            input_names=[
                "source_paths", "reference_voxel_source", "bids_dir",
                "subject", "session", "is_postcon_ref", "qc_metrics",
            ],
            output_names=["meta_dict"],
            function=_build_dseg_meta,
        ),
        name="dseg_meta",
    )
    meta.inputs.is_postcon_ref = is_postcon_ref
    workflow.connect([
        (src_collector, meta, [("out", "source_paths")]),
        (inputnode, meta, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            ("segmentation", "reference_voxel_source"),
            ("qc_metrics", "qc_metrics"),
        ]),
    ])

    sink = pe.Node(
        PMRIPrepDataSink(
            check_hdr=True, compress=True,
            allowed_entities=("space",),
            space="T1w", desc="tumorseg", suffix="dseg",
        ),
        name="dseg_sink",
    )
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "base_directory"),
            ("segmentation", "in_file"),
            ("source_t1w", "source_file"),
        ]),
        (meta, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


def _add_mask_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    *,
    name: str,
    in_attr: str,
    desc_value: str,
    description: str,
    method: str,
    source_attrs: tuple[str, ...],
    is_postcon_ref: bool,
) -> pe.Node:
    src_collector = pe.Node(
        niu.Merge(numinputs=len(source_attrs)),
        name=f"{name}_sources",
    )
    for i, attr in enumerate(source_attrs, start=1):
        workflow.connect(inputnode, attr, src_collector, f"in{i}")

    meta = pe.Node(
        niu.Function(
            input_names=[
                "description", "method", "source_paths",
                "reference_voxel_source", "bids_dir", "subject", "session",
                "is_postcon_ref",
            ],
            output_names=["meta_dict"],
            function=_build_mask_meta,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.method = method
    meta.inputs.is_postcon_ref = is_postcon_ref
    workflow.connect([
        (src_collector, meta, [("out", "source_paths")]),
        (inputnode, meta, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            (in_attr, "reference_voxel_source"),
        ]),
    ])

    sink = pe.Node(
        PMRIPrepDataSink(
            check_hdr=True, compress=True,
            allowed_entities=("space",),
            space="T1w", desc=desc_value, suffix="mask",
        ),
        name=f"{name}_sink",
    )
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "base_directory"),
            (in_attr, "in_file"),
            (source_attrs[0], "source_file"),
        ]),
        (meta, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


def _add_nawm_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    is_postcon_ref: bool,
) -> pe.Node:
    # NAWM is derived from T1w + WM probseg.
    src_collector = pe.Node(niu.Merge(numinputs=1), name="nawm_sources")
    workflow.connect(inputnode, "source_t1w", src_collector, "in1")

    meta = pe.Node(
        niu.Function(
            input_names=[
                "source_paths", "reference_voxel_source", "bids_dir",
                "subject", "session", "is_postcon_ref",
            ],
            output_names=["meta_dict"],
            function=_build_nawm_meta,
        ),
        name="nawm_meta",
    )
    meta.inputs.is_postcon_ref = is_postcon_ref
    workflow.connect([
        (src_collector, meta, [("out", "source_paths")]),
        (inputnode, meta, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            ("nawm_mask", "reference_voxel_source"),
        ]),
    ])

    sink = pe.Node(
        PMRIPrepDataSink(
            check_hdr=True, compress=True,
            allowed_entities=("space",),
            space="T1w", desc="nawm", suffix="mask",
        ),
        name="nawm_sink",
    )
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "base_directory"),
            ("nawm_mask", "in_file"),
            ("source_t1w", "source_file"),
        ]),
        (meta, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


def init_tumor_seg_datasinks_wf(
    *,
    output_dir: str,
    bids_dir: str,
    subject_id: str,
    session_id: Optional[str],
    name: str = "tumor_seg_datasinks_wf",
) -> pe.Workflow:
    """Build the tumor-seg DerivativesDataSink chain.

    HD-GLIO consumes T1w-pre, T1w-postcon, T2w, and FLAIR-precon; ``Sources``
    on every tumor-seg derivative carries that quartet. Tumor-seg only runs
    when all four modalities are present (gated upstream in ``discovery``),
    so the structural reference is always the precon T1w (no ``is_postcon_ref``
    branch needed here, unlike the anat factory).
    """
    is_postcon_ref = False  # invariant: tumor-seg requires t1w_pre
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "output_dir",
                "bids_dir",
                "subject_id",
                "session_id",
                # Source paths for Sources field
                "source_t1w",
                "source_t1w_postcon",
                "source_t2w",
                "source_flair",
                # Derivative inputs
                "segmentation",
                "enhancing_mask",
                "non_enhancing_mask",
                "whole_tumor_mask",
                "nawm_mask",
                # QC + figures
                "qc_metrics",
                "seg_filmstrip_postcon",
                "seg_filmstrip_flair",
                "nawm_filmstrip_flair",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.output_dir = output_dir
    inputnode.inputs.bids_dir = bids_dir
    inputnode.inputs.subject_id = subject_id
    inputnode.inputs.session_id = session_id or ""

    sink_outputs: list[tuple[pe.Node, str]] = []

    sink_outputs.append((_add_dseg_sink(workflow, inputnode, is_postcon_ref), "out_file"))
    sink_outputs.append((_add_dseg_tsv_sink(workflow, inputnode), "out_file"))

    sink_outputs.append((_add_mask_sink(
        workflow, inputnode,
        name="enhancing", in_attr="enhancing_mask", desc_value="enhancing",
        description="Binary mask of enhancing tumor (label 2)",
        method="Derived from tumor segmentation",
        source_attrs=_HD_GLIO_SOURCE_ATTRS, is_postcon_ref=is_postcon_ref,
    ), "out_file"))

    sink_outputs.append((_add_mask_sink(
        workflow, inputnode,
        name="nonenhancing", in_attr="non_enhancing_mask",
        desc_value="nonenhancing",
        description="Binary mask of non-enhancing tumor (label 1)",
        method="Derived from tumor segmentation",
        source_attrs=_HD_GLIO_SOURCE_ATTRS, is_postcon_ref=is_postcon_ref,
    ), "out_file"))

    sink_outputs.append((_add_mask_sink(
        workflow, inputnode,
        name="wholetumor", in_attr="whole_tumor_mask",
        desc_value="wholetumor",
        description="Binary mask of whole tumor (labels 1+2)",
        method="Derived from tumor segmentation",
        source_attrs=_HD_GLIO_SOURCE_ATTRS, is_postcon_ref=is_postcon_ref,
    ), "out_file"))

    # lesion_mask is a duplicate of whole_tumor_mask under a different name.
    sink_outputs.append((_add_mask_sink(
        workflow, inputnode,
        name="lesion", in_attr="whole_tumor_mask",
        desc_value="lesion",
        description="Lesion mask from whole tumor segmentation",
        method="Tumor segmentation (labels 1+2)",
        source_attrs=_HD_GLIO_SOURCE_ATTRS, is_postcon_ref=is_postcon_ref,
    ), "out_file"))

    sink_outputs.append((_add_nawm_sink(workflow, inputnode, is_postcon_ref), "out_file"))

    # Figures writer (figure filmstrips only; clinical context now in dseg.json)
    figures_writer = pe.Node(TumorSegFiguresWriter(), name="figures_writer")
    workflow.connect([
        (inputnode, figures_writer, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject_id"),
            ("session_id", "session_id"),
            ("seg_filmstrip_postcon", "seg_filmstrip_postcon"),
            ("seg_filmstrip_flair", "seg_filmstrip_flair"),
            ("nawm_filmstrip_flair", "nawm_filmstrip_flair"),
        ]),
    ])

    # dataset_description.json barrier
    merge = pe.Node(
        niu.Merge(numinputs=len(sink_outputs) + 1), name="dd_merge",
    )
    for i, (sink, attr) in enumerate(sink_outputs, start=1):
        workflow.connect(sink, attr, merge, f"in{i}")
    workflow.connect(
        figures_writer, "done", merge,
        f"in{len(sink_outputs) + 1}",
    )

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

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["done", "dataset_description_done"],
        ),
        name="outputnode",
    )
    workflow.connect([
        (dd_node, outputnode, [
            ("dd_path", "done"),
            ("dd_path", "dataset_description_done"),
        ]),
    ])

    return workflow
