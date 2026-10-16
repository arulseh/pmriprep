"""Test entity filtering for cross-modality derivative filenames.

Structural images resampled into perfusion space omit the perfusion
acquisition entity. Masks derived from precontrast anatomy also omit the
contrast-agent entity. Tests exercise PMRIPrepDataSink on synthetic files.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def _make_minimal_nifti(path: Path) -> None:
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))
    nib.save(img, str(path))


class TestDSCAnatSinkStripsAcq:
    """Regression: T1w-in-DSC outputs must not inherit acq-* from DSC source."""

    def test_t1w_in_dsc_strips_acq_ep2d(self, tmp_path) -> None:
        from pmriprep.interfaces.qc_writers import PMRIPrepDataSink

        # Synthesize a BIDS DSC source carrying acq-ep2d (sub-11 case)
        src = tmp_path / "sub-11_ses-01_acq-ep2d_ce-GAD_run-1_dsc.nii.gz"
        _make_minimal_nifti(src)
        in_nii = tmp_path / "in.nii.gz"
        _make_minimal_nifti(in_nii)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        sink = PMRIPrepDataSink(
            base_directory=str(out_dir), check_hdr=False,
            allowed_entities=("space", "datatype", "ceagent"),
            space="dscref", suffix="T1w", datatype="anat",
            dismiss_entities=["acquisition"],
        )
        sink.inputs.source_file = str(src)
        sink.inputs.in_file = str(in_nii)
        res = sink.run()
        out_files = res.outputs.out_file
        out_name = Path(out_files[0] if isinstance(out_files, list) else out_files).name
        assert "acq-" not in out_name, f"acq- should be stripped: {out_name}"
        assert "space-dscref" in out_name, f"space-dscref should be present: {out_name}"
        assert "T1w" in out_name, f"T1w suffix should be present: {out_name}"
        # ce-GAD should still propagate (post-con T1w-in-DSC retains ce)
        assert "ce-GAD" in out_name, f"ce-GAD should propagate: {out_name}"


class TestDSCAnatSinkStripsAcqAndCe:
    """Regression: pre-contrast-derived masks must drop acq-* AND ce-GAD."""

    def test_brain_mask_in_dsc_strips_acq_and_ce(self, tmp_path) -> None:
        from pmriprep.interfaces.qc_writers import PMRIPrepDataSink

        src = tmp_path / "sub-11_ses-01_acq-ep2d_ce-GAD_run-1_dsc.nii.gz"
        _make_minimal_nifti(src)
        in_nii = tmp_path / "in.nii.gz"
        _make_minimal_nifti(in_nii)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        sink = PMRIPrepDataSink(
            base_directory=str(out_dir), check_hdr=False,
            allowed_entities=("space", "datatype", "ceagent"),
            space="dscref", desc="brain", suffix="mask", datatype="anat",
            dismiss_entities=["acquisition", "ceagent"],
        )
        sink.inputs.source_file = str(src)
        sink.inputs.in_file = str(in_nii)
        res = sink.run()
        out_files = res.outputs.out_file
        out_name = Path(out_files[0] if isinstance(out_files, list) else out_files).name
        assert "acq-" not in out_name, f"acq- should be stripped: {out_name}"
        # `ce-` precise check: avoid matching `ce-d` inside `space-dscref`
        assert "_ce-" not in out_name and not out_name.startswith("ce-"), \
            f"ce-* should be stripped (brain mask is pre-con): {out_name}"
        assert "space-dscref" in out_name, f"space-dscref should be present: {out_name}"
        assert "desc-brain" in out_name, f"desc-brain should be present: {out_name}"
        assert "_mask" in out_name, f"mask suffix should be present: {out_name}"


class TestDCEAnatImageSinkStripsAcq:
    """Regression: T1w-in-DCE outputs must not inherit acq-vibe from DCE source."""

    def test_t1w_in_dce_strips_acq_vibe(self, tmp_path) -> None:
        from pmriprep.interfaces.qc_writers import PMRIPrepDataSink

        src = tmp_path / "sub-11_ses-01_acq-vibe_ce-GAD_run-1_dce.nii.gz"
        _make_minimal_nifti(src)
        in_nii = tmp_path / "in.nii.gz"
        _make_minimal_nifti(in_nii)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        sink = PMRIPrepDataSink(
            base_directory=str(out_dir), check_hdr=False,
            allowed_entities=("space", "datatype", "ceagent"),
            space="dceref", suffix="T1w", datatype="anat",
            dismiss_entities=["acquisition"],
        )
        sink.inputs.source_file = str(src)
        sink.inputs.in_file = str(in_nii)
        res = sink.run()
        out_files = res.outputs.out_file
        out_name = Path(out_files[0] if isinstance(out_files, list) else out_files).name
        assert "acq-" not in out_name, f"acq-vibe should be stripped: {out_name}"
        assert "space-dceref" in out_name, f"space-dceref should be present: {out_name}"
        assert "T1w" in out_name, f"T1w suffix should be present: {out_name}"


class TestDSCAnatSinkBaselinePropagation:
    """Sanity: when no acq is present in source, behavior unchanged for sub-01."""

    def test_t1w_in_dsc_no_acq_in_source(self, tmp_path) -> None:
        from pmriprep.interfaces.qc_writers import PMRIPrepDataSink

        # sub-01 case: source has no acq- entity to begin with
        src = tmp_path / "sub-01_ses-01_ce-GAD_run-1_dsc.nii.gz"
        _make_minimal_nifti(src)
        in_nii = tmp_path / "in.nii.gz"
        _make_minimal_nifti(in_nii)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        sink = PMRIPrepDataSink(
            base_directory=str(out_dir), check_hdr=False,
            allowed_entities=("space", "datatype", "ceagent"),
            space="dscref", suffix="T1w", datatype="anat",
            dismiss_entities=["acquisition"],
        )
        sink.inputs.source_file = str(src)
        sink.inputs.in_file = str(in_nii)
        res = sink.run()
        out_files = res.outputs.out_file
        out_name = Path(out_files[0] if isinstance(out_files, list) else out_files).name
        # Should look the same as before — no change for sub-01
        assert "acq-" not in out_name
        assert "space-dscref" in out_name
        assert "ce-GAD" in out_name
        assert "T1w" in out_name
