"""PR1 (BIDS arc finishing): tumor seg datasinks workflow wires dseg.tsv writer.

Asserts the tumor_seg_datasinks_wf builds a ``dseg_tsv_sink`` Function
node that writes ``space-T1w_desc-tumorseg_dseg.tsv`` next to the
existing ``_dseg.nii.gz`` per BIDS-Common-Derivatives.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np

from pmriprep.workflows.tumor_seg.datasinks import (
    _write_tumor_dseg_tsv_node,
    init_tumor_seg_datasinks_wf,
)


def test_dseg_tsv_sink_node_present_in_workflow() -> None:
    wf = init_tumor_seg_datasinks_wf(
        output_dir="/tmp/out", bids_dir="/tmp/bids",
        subject_id="X", session_id="Y",
    )
    nodes = {n.name for n in wf._graph.nodes()}
    assert "dseg_tsv_sink" in nodes
    # The dseg_sink (NIfTI) must also still be present
    assert "dseg_sink" in nodes


def test_write_tumor_dseg_tsv_node_lands_in_anat(tmp_path: Path) -> None:
    # Build a tiny dseg NIfTI with both tumor labels
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[0, 0, 0] = 1
    data[0, 0, 1] = 2
    nii = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(nii))

    out = _write_tumor_dseg_tsv_node(
        segmentation=str(nii),
        output_dir=str(tmp_path),
        subject="X",
        session="Y",
    )
    out_path = Path(out)
    assert out_path.exists()
    # anat/ landing per BIDS-Common-Derivatives (companion to _dseg.nii.gz)
    assert out_path.parent.name == "anat"
    assert out_path.name == "sub-X_ses-Y_space-T1w_desc-tumorseg_dseg.tsv"

    text = out_path.read_text()
    assert text.startswith("index\tname\tabbreviation\tvoxel_count\tvolume_mm3")
    # Both labels appear (background omitted per BIDS dseg convention)
    assert "non_enhancing_tumor\tNET" in text
    assert "contrast_enhancing_tumor\tET" in text


def test_no_session_path(tmp_path: Path) -> None:
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[0, 0, 0] = 1
    nii = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(nii))

    out = _write_tumor_dseg_tsv_node(
        segmentation=str(nii),
        output_dir=str(tmp_path),
        subject="X",
        session="",
    )
    out_path = Path(out)
    assert out_path.name == "sub-X_space-T1w_desc-tumorseg_dseg.tsv"
    assert "ses-" not in str(out_path.parent)
