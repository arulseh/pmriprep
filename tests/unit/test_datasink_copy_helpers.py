"""Tests for datasink Function-node path conventions."""

from __future__ import annotations

from pathlib import Path

from traits.trait_base import Undefined


class TestDCECopyPerfMap:
    """DCE map copies accept real paths and the supported missing sentinels."""

    def test_undefined_returns_empty_without_writing(self, tmp_path) -> None:
        from pmriprep.workflows.dce.datasinks import _copy_perf_map

        out = _copy_perf_map(
            in_file=Undefined,
            output_dir=str(tmp_path),
            subject="X",
            session="Y",
            desc="dceKtrans",
            suffix="Ktrans",
            meta_dict={},
        )
        assert out == ""
        assert not list(tmp_path.rglob("*.nii.gz"))
        assert not list(tmp_path.rglob("*.json"))

    def test_empty_string_returns_empty_without_writing(self, tmp_path) -> None:
        from pmriprep.workflows.dce.datasinks import _copy_perf_map

        out = _copy_perf_map(
            in_file="",
            output_dir=str(tmp_path),
            subject="X",
            session="Y",
            desc="dceKtrans",
            suffix="Ktrans",
            meta_dict={},
        )
        assert out == ""
        assert not list(tmp_path.rglob("*.nii.gz"))

    def test_real_file_copies_with_sidecar(self, tmp_path) -> None:
        from pmriprep.workflows.dce.datasinks import _copy_perf_map

        src = tmp_path / "src.nii.gz"
        src.write_bytes(b"fake nifti")
        out = _copy_perf_map(
            in_file=str(src),
            output_dir=str(tmp_path / "out"),
            subject="X",
            session="Y",
            desc="dceKtrans",
            suffix="Ktrans",
            meta_dict={"Description": "test"},
        )
        assert out.endswith("sub-X_ses-Y_desc-dceKtransKtrans_stat.nii.gz")
        assert Path(out).is_file()
        assert Path(out.replace(".nii.gz", ".json")).is_file()


class TestDSCCopyPerfMap:
    """DSC map copies accept real paths and the supported missing sentinels."""

    def test_undefined_returns_empty_without_writing(self, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map

        out = _copy_perf_map(
            in_file=Undefined,
            output_dir=str(tmp_path),
            subject="X",
            session="Y",
            desc="dsc",
            suffix="cbv",
            meta_dict={},
        )
        assert out == ""
        assert not list(tmp_path.rglob("*.nii.gz"))
        assert not list(tmp_path.rglob("*.json"))

    def test_empty_string_returns_empty_without_writing(self, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map

        out = _copy_perf_map(
            in_file="",
            output_dir=str(tmp_path),
            subject="X",
            session="Y",
            desc="dsc",
            suffix="cbv",
            meta_dict={},
        )
        assert out == ""

    def test_real_file_copies_with_sidecar(self, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map

        src = tmp_path / "src.nii.gz"
        src.write_bytes(b"fake nifti")
        out = _copy_perf_map(
            in_file=str(src),
            output_dir=str(tmp_path / "out"),
            subject="X",
            session="Y",
            desc="dsc",
            suffix="cbv",
            meta_dict={"Description": "test"},
        )
        assert out.endswith("sub-X_ses-Y_desc-dscCbv_stat.nii.gz")
        assert Path(out).is_file()
        assert Path(out.replace(".nii.gz", ".json")).is_file()
