"""R3 spike: validate niworkflows DerivativesDataSink preserves caller-supplied
``meta_dict`` keys (``Sources``, ``SpatialReference``, ``Resolution``,
``VoxelSize``) without overwriting them.

Architectural gating test for B2/M3 PR-C. If this passes, the per-output
``meta_dict``-based sidecar enrichment pattern is viable for the four
``init_*_datasinks_wf`` factories.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def test_datasink_meta_dict_survives_for_nonstandard_space(tmp_path: Path) -> None:
    """Caller-supplied Sources/SpatialReference/Resolution must round-trip."""
    from niworkflows.interfaces.bids import DerivativesDataSink

    # Build a minimal raw BIDS source so DataSink can derive sub/ses entities.
    bids_root = tmp_path / "bids"
    raw_anat = bids_root / "sub-01" / "ses-0001" / "anat"
    raw_anat.mkdir(parents=True)
    (bids_root / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = raw_anat / "sub-01_ses-0001_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src)
    (raw_anat / "sub-01_ses-0001_T1w.json").write_text(
        json.dumps({"RepetitionTime": 2.0})
    )

    # Build a derivative input image (the file we are sinking).
    deriv_in = tmp_path / "ktrans.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), deriv_in)

    out_root = tmp_path / "out"

    expected_sources = ["bids::sub-01/ses-0001/anat/sub-01_ses-0001_T1w.nii.gz"]
    expected_spatial_ref = "bids::pmriprep/sub-01/ses-0001/anat/sub-01_ses-0001_space-anat_desc-preproc_T1w.nii.gz"
    expected_resolution = "native"
    expected_voxel_size = [1.0, 1.0, 1.0]

    sink = DerivativesDataSink(
        base_directory=str(out_root),
        out_path_base="pmriprep",
        check_hdr=False,  # spike: skip header validation
        compress=True,
        allowed_entities=("space", "model"),
    )
    sink.inputs.in_file = str(deriv_in)
    sink.inputs.source_file = str(src)
    sink.inputs.space = "anat"
    sink.inputs.desc = "preproc"
    sink.inputs.suffix = "T1w"
    sink.inputs.meta_dict = {
        "Sources": expected_sources,
        "SpatialReference": expected_spatial_ref,
        "Resolution": expected_resolution,
        "VoxelSize": expected_voxel_size,
        "Description": "Spike-test sidecar",
    }

    result = sink.run()
    out_files = result.outputs.out_file
    if isinstance(out_files, str):
        out_files = [out_files]
    out_meta = result.outputs.out_meta
    if isinstance(out_meta, list):
        out_meta = out_meta[0]

    assert Path(out_files[0]).exists(), f"Output NIfTI not written: {out_files[0]}"
    assert Path(out_meta).exists(), f"Sidecar JSON not written: {out_meta}"

    sidecar = json.loads(Path(out_meta).read_text())
    assert sidecar["Sources"] == expected_sources
    assert sidecar["SpatialReference"] == expected_spatial_ref
    assert sidecar["Resolution"] == expected_resolution
    assert sidecar["VoxelSize"] == expected_voxel_size
    assert sidecar["Description"] == "Spike-test sidecar"
