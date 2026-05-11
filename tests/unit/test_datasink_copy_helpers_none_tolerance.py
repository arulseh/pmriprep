"""Regression tests for v0.50.2 datasink Function-node None-tolerance.

The 6 ``_copy_*`` Function-node helpers across DCE/DSC/anat datasinks
all call ``shutil.copy2(in_file, ...)``. Conditional producers like
``_tortuosity_correction_node`` and ``_normalize_by_nawm`` legitimately
return ``None`` when their preconditions aren't met. Pre-v0.50.2 this
would crash the whole subject inside ``shutil.copy2(None, ...)`` with
``TypeError: stat: path should be string, bytes, os.PathLike or
integer, not NoneType``. Reproduced on sub-0081 (OSIPI test-retest) at
the ``ktrans_tortuosity_corrected_sink`` node.

Each helper now early-returns ``""`` when given Undefined/None/"".
These tests cover the two perfusion-map helpers explicitly (the most
complex signatures); the four xfm/B1+ helpers share the same guard
pattern with simpler signatures and rely on the perf-map tests for
guard validation.
"""
from __future__ import annotations

from pathlib import Path

from traits.trait_base import Undefined


class TestDCECopyPerfMapNoneTolerance:
    """Regression: dce/datasinks._copy_perf_map must no-op on Undefined/None/''."""

    def test_undefined_in_file_returns_empty_no_files_written(self, tmp_path) -> None:
        """The exact sentinel that triggered the sub-0081 crash."""
        from pmriprep.workflows.dce.datasinks import _copy_perf_map
        out = _copy_perf_map(
            in_file=Undefined, output_dir=str(tmp_path), subject="X",
            session="Y", desc="dceTortuosityCorr", suffix="Ktrans",
            meta_dict={},
        )
        assert out == ""
        assert not list(tmp_path.rglob("*.nii.gz"))
        assert not list(tmp_path.rglob("*.json"))

    def test_none_in_file_returns_empty_no_files_written(self, tmp_path) -> None:
        """Nipype converts Undefined → None on some Function-node code paths."""
        from pmriprep.workflows.dce.datasinks import _copy_perf_map
        out = _copy_perf_map(
            in_file=None, output_dir=str(tmp_path), subject="X",
            session="Y", desc="dceTortuosityCorr", suffix="Ktrans",
            meta_dict={},
        )
        assert out == ""
        assert not list(tmp_path.rglob("*.nii.gz"))
        assert not list(tmp_path.rglob("*.json"))

    def test_empty_string_in_file_returns_empty_no_files_written(self, tmp_path) -> None:
        """Function-node-returns-empty-string convention (per _transform_lesion_mask_to_dce)."""
        from pmriprep.workflows.dce.datasinks import _copy_perf_map
        out = _copy_perf_map(
            in_file="", output_dir=str(tmp_path), subject="X",
            session="Y", desc="dceTortuosityCorr", suffix="Ktrans",
            meta_dict={},
        )
        assert out == ""
        assert not list(tmp_path.rglob("*.nii.gz"))

    def test_real_file_still_copies_with_sidecar(self, tmp_path) -> None:
        """Happy path unchanged: real path → NIfTI copied + JSON sidecar written."""
        from pmriprep.workflows.dce.datasinks import _copy_perf_map
        src = tmp_path / "src.nii.gz"
        src.write_bytes(b"fake nifti")
        out_dir = tmp_path / "out"
        out = _copy_perf_map(
            in_file=str(src), output_dir=str(out_dir),
            subject="X", session="Y",
            desc="dceTortuosityCorr", suffix="Ktrans",
            meta_dict={"Description": "test"},
        )
        assert out.endswith("sub-X_ses-Y_desc-dceTortuosityCorrKtrans_stat.nii.gz")
        assert Path(out).is_file()
        assert Path(out.replace(".nii.gz", ".json")).is_file()


class TestDSCCopyPerfMapNoneTolerance:
    """Regression: dsc/datasinks._copy_perf_map must no-op on Undefined/None/''."""

    def test_undefined_in_file_returns_empty_no_files_written(self, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map
        out = _copy_perf_map(
            in_file=Undefined, output_dir=str(tmp_path), subject="X",
            session="Y", desc="dsc", suffix="cbv",
            meta_dict={},
        )
        assert out == ""
        assert not list(tmp_path.rglob("*.nii.gz"))
        assert not list(tmp_path.rglob("*.json"))

    def test_none_in_file_returns_empty_no_files_written(self, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map
        out = _copy_perf_map(
            in_file=None, output_dir=str(tmp_path), subject="X",
            session="Y", desc="dsc", suffix="cbv",
            meta_dict={},
        )
        assert out == ""

    def test_empty_string_in_file_returns_empty_no_files_written(self, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map
        out = _copy_perf_map(
            in_file="", output_dir=str(tmp_path), subject="X",
            session="Y", desc="dsc", suffix="cbv",
            meta_dict={},
        )
        assert out == ""

    def test_real_file_still_copies_with_sidecar(self, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map
        src = tmp_path / "src.nii.gz"
        src.write_bytes(b"fake nifti")
        out_dir = tmp_path / "out"
        out = _copy_perf_map(
            in_file=str(src), output_dir=str(out_dir),
            subject="X", session="Y",
            desc="dsc", suffix="cbv",
            meta_dict={"Description": "test"},
        )
        assert out.endswith("sub-X_ses-Y_desc-dscCbv_stat.nii.gz")
        assert Path(out).is_file()
        assert Path(out.replace(".nii.gz", ".json")).is_file()


