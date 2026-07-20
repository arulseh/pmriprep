"""Tests for datasink Function-node path conventions."""

from __future__ import annotations

from pathlib import Path

from traits.trait_base import Undefined


def test_source_entity_prefix_preserves_canonical_distinguishers() -> None:
    from pmriprep.utils.sidecar_builders import source_entity_prefix

    source = (
        "bids:raw:sub-X/ses-Y/perf/"
        "sub-X_ses-Y_task-rest_acq-fast_dir-AP_run-02_echo-1_part-mag_dce.nii.gz"
    )
    assert source_entity_prefix(source, "X", "Y") == (
        "sub-X_ses-Y_task-rest_acq-fast_dir-AP_run-02_echo-1_part-mag"
    )


def test_tb1map_uses_standard_suffix_and_dce_source_entities(tmp_path) -> None:
    from pmriprep.workflows.dce.datasinks import _copy_tb1map

    src = tmp_path / "b1.nii.gz"
    src.write_bytes(b"fake nifti")
    out = _copy_tb1map(
        in_file=str(src), output_dir=str(tmp_path / "out"),
        subject="X", session="Y",
        meta_dict={
            "Sources": [
                "bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_acq-fast_run-2_dce.nii.gz"
            ],
            "Units": "1",
        },
    )
    assert out.endswith(
        "sub-X_ses-Y_acq-fast_run-2_space-dceref_TB1map.nii.gz"
    )


def test_perfusion_spatial_reference_preserves_source_entities() -> None:
    from pmriprep.utils.sidecar_builders import _dce_spatial_reference

    source = "/raw/sub-X/ses-Y/perf/sub-X_ses-Y_acq-fast_run-2_dce.nii.gz"
    assert _dce_spatial_reference("X", "Y", source) == (
        "bids::sub-X/ses-Y/perf/"
        "sub-X_ses-Y_acq-fast_run-2_desc-dceMean_stat.nii.gz"
    )


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

    def test_source_run_entity_prevents_collision(self, tmp_path) -> None:
        from pmriprep.workflows.dce.datasinks import _copy_perf_map

        src = tmp_path / "src.nii.gz"
        src.write_bytes(b"fake nifti")
        outputs = []
        for run in (1, 2):
            outputs.append(_copy_perf_map(
                in_file=str(src), output_dir=str(tmp_path / "out"),
                subject="X", session="Y", desc="dce", suffix="Ktrans",
                meta_dict={
                    "Sources": [
                        f"bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_run-{run}_dce.nii.gz"
                    ],
                },
            ))
        assert outputs[0] != outputs[1]
        assert "_run-1_" in outputs[0]
        assert "_run-2_" in outputs[1]


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

    def test_source_run_entity_prevents_collision(self, tmp_path) -> None:
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map

        src = tmp_path / "src.nii.gz"
        src.write_bytes(b"fake nifti")
        outputs = []
        for run in (1, 2):
            outputs.append(_copy_perf_map(
                in_file=str(src), output_dir=str(tmp_path / "out"),
                subject="X", session="Y", desc="dsc", suffix="cbv",
                meta_dict={
                    "Sources": [
                        f"bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_run-{run}_dsc.nii.gz"
                    ],
                },
            ))
        assert outputs[0] != outputs[1]
        assert "_run-1_" in outputs[0]
        assert "_run-2_" in outputs[1]
