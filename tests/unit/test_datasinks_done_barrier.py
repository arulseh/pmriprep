"""Regression tests for the per-pipeline ``outputnode.done`` barrier.

``outputnode.done`` must fire only after every per-output sink (including
the resume-sidecar writers) has finished. The four ``init_*_datasinks_wf``
factories source ``done`` from ``dd_node.dd_path`` so that the barrier
inherits the join semantics of ``dd_merge`` (which gathers every
``sink_outputs`` entry plus ``figures_writer``).

If a future change re-routes ``done`` from a single ``figures_writer``
output, ``unified_report`` can race ahead of the resume-sink and read
a partial JSON sidecar. These tests fail fast in that scenario.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def _write_bids(tmp_path: Path) -> Path:
    bids = tmp_path / "bids"
    (bids / "sub-01" / "ses-0001" / "anat").mkdir(parents=True)
    (bids / "sub-01" / "ses-0001" / "perf").mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    return bids


def _ancestors_of(wf, target_name: str) -> set[str]:
    """Return the set of node names upstream of ``target_name`` in ``wf._graph``."""
    target = next(n for n in wf._graph.nodes if n.name == target_name)
    return {n.name for n in nx.ancestors(wf._graph, target)}


def test_anat_done_barrier_routed_through_dd_node(tmp_path: Path) -> None:
    from pmriprep.workflows.anat.datasinks import init_anat_datasinks_wf

    _write_bids(tmp_path)
    wf = init_anat_datasinks_wf(
        output_dir=str(tmp_path / "out"),
        bids_dir=str(tmp_path / "bids"),
        subject_id="01",
        session_id="0001",
        has_t1w_pre=True,
        has_t1w_postcon=False,
        has_t2w=False,
        has_flair=False,
        has_flair_postcon=False,
        has_tissue_seg=False,
        has_subtraction=False,
    )
    ancestors = _ancestors_of(wf, "outputnode")
    assert "dataset_description" in ancestors
    assert "figures_writer" in ancestors


def test_tumor_seg_done_barrier_routed_through_dd_node(tmp_path: Path) -> None:
    from pmriprep.workflows.tumor_seg.datasinks import init_tumor_seg_datasinks_wf

    _write_bids(tmp_path)
    wf = init_tumor_seg_datasinks_wf(
        output_dir=str(tmp_path / "out"),
        bids_dir=str(tmp_path / "bids"),
        subject_id="01",
        session_id="0001",
    )
    ancestors = _ancestors_of(wf, "outputnode")
    assert "dataset_description" in ancestors
    assert "figures_writer" in ancestors


def test_dsc_done_barrier_includes_resume_checkpoint(tmp_path: Path) -> None:
    """DSC: resume_checkpoint must be reachable from outputnode.done."""
    from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

    _write_bids(tmp_path)
    wf = init_dsc_datasinks_wf(
        output_dir=str(tmp_path / "out"),
        bids_dir=str(tmp_path / "bids"),
        subject_id="01",
        session_id="0001",
        inference="vi",
        present_inputs={"dsc_moco"},
    )
    ancestors = _ancestors_of(wf, "outputnode")
    assert "dataset_description" in ancestors
    assert "resume_checkpoint" in ancestors, (
        "DSC resume_checkpoint must join outputnode.done so unified_report "
        "cannot race ahead of the checkpoint writer"
    )
