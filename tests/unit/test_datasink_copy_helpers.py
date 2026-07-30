"""Tests for datasink Function-node path conventions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from traits.trait_base import Undefined


def test_tb1map_sink_accepts_undefined_optional_input(tmp_path: Path) -> None:
    """Nipype may omit an unavailable B1 map when T1 mapping is disabled."""
    import nipype.interfaces.utility as niu
    import nipype.pipeline.engine as pe

    from pmriprep.workflows.dce.datasinks import _copy_tb1map

    node = pe.Node(
        niu.Function(
            input_names=[
                "in_file",
                "output_dir",
                "subject",
                "session",
                "meta_dict",
            ],
            output_names=["out_file"],
            function=_copy_tb1map,
        ),
        name="b1_map_sink",
        base_dir=str(tmp_path),
    )
    node.inputs.output_dir = str(tmp_path / "out")
    node.inputs.subject = "X"
    node.inputs.session = "Y"

    result = node.run()

    assert result.outputs.out_file == ""


def test_tb1map_sink_requires_metadata_for_real_map(tmp_path: Path) -> None:
    """A produced B1 map cannot be written without provenance metadata."""
    import pytest

    from pmriprep.workflows.dce.datasinks import _copy_tb1map

    src = tmp_path / "b1.nii.gz"
    src.write_bytes(b"fake nifti")

    with pytest.raises(ValueError, match="metadata is required"):
        _copy_tb1map(
            output_dir=str(tmp_path / "out"),
            subject="X",
            session="Y",
            in_file=str(src),
        )


def test_source_entity_prefix_preserves_canonical_distinguishers() -> None:
    from pmriprep.utils.sidecar_builders import source_entity_prefix

    source = (
        "bids:raw:sub-X/ses-Y/perf/"
        "sub-X_ses-Y_task-rest_acq-fast_dir-AP_run-02_echo-1_part-mag_dce.nii.gz"
    )
    assert source_entity_prefix(source, "X", "Y") == (
        "sub-X_ses-Y_task-rest_acq-fast_dir-AP_run-02_echo-1_part-mag"
    )


def test_transformed_lesion_contract_preserves_labels_and_sources() -> None:
    from pmriprep.config.constants import HDGLIO_BIDS_LABELS
    from pmriprep.workflows.dce.datasinks import _ANAT_MASKS as dce_specs
    from pmriprep.workflows.dce.datasinks import _build_dce_transformed_mask_meta
    from pmriprep.workflows.dsc.datasinks import _ANAT_MASKS as dsc_specs
    from pmriprep.workflows.dsc.datasinks import _build_transformed_mask_meta

    dce = next(spec for spec in dce_specs if spec.in_attr == "lesion_mask_dce")
    dsc = next(spec for spec in dsc_specs if spec.in_attr == "lesion_mask_dsc")
    assert dce.bids_suffix == "dseg"
    assert dce.extra == {"LabelMap": HDGLIO_BIDS_LABELS}
    assert dsc.suffix == "dseg"
    assert dsc.segmentation_labels == HDGLIO_BIDS_LABELS

    base = {"Sources": ["bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_dce.nii.gz"]}
    structural_sources = [
        "/raw/sub-X/ses-Y/anat/sub-X_ses-Y_T1w.nii.gz",
        "/raw/sub-X/ses-Y/anat/sub-X_ses-Y_FLAIR.nii.gz",
    ]
    dce_meta = _build_dce_transformed_mask_meta(
        base, "Tumor segmentation", None,
        {"LabelMap": HDGLIO_BIDS_LABELS},
        structural_sources,
        "/raw/sub-X/ses-Y/perf/sub-X_ses-Y_dce.nii.gz", "/raw",
    )
    dsc_meta = _build_transformed_mask_meta(
        base, "Tumor segmentation", False, None,
        structural_sources,
        "/raw/sub-X/ses-Y/perf/sub-X_ses-Y_dsc.nii.gz", "/raw",
        HDGLIO_BIDS_LABELS,
    )
    for meta in (dce_meta, dsc_meta):
        assert len(meta["Sources"]) == 3
        assert meta["LabelMap"] == HDGLIO_BIDS_LABELS


def test_transformed_mask_metadata_builders_run_in_isolated_functions() -> None:
    """Transformed-mask builders carry no unavailable module globals."""
    import nipype.interfaces.utility as niu

    from pmriprep.config.constants import HDGLIO_BIDS_LABELS
    from pmriprep.workflows.dce.datasinks import _build_dce_transformed_mask_meta
    from pmriprep.workflows.dsc.datasinks import _build_transformed_mask_meta

    common = {
        "base": {"Sources": []},
        "description": "Tumor segmentation",
        "structural_sources": ["/raw/sub-X/anat/sub-X_T1w.nii.gz"],
        "perfusion_source": "/raw/sub-X/perf/sub-X_dsc.nii.gz",
        "bids_dir": "/raw",
    }
    dsc = niu.Function(
        input_names=[
            "base", "description", "skull_stripped", "mask_type",
            "structural_sources", "perfusion_source", "bids_dir",
            "segmentation_labels",
        ],
        output_names=["meta"],
        function=_build_transformed_mask_meta,
    )
    dsc.inputs.trait_set(
        **common,
        skull_stripped=False,
        mask_type=None,
        segmentation_labels=HDGLIO_BIDS_LABELS,
    )
    dce = niu.Function(
        input_names=[
            "base", "description", "units", "extra", "structural_sources",
            "perfusion_source", "bids_dir",
        ],
        output_names=["meta"],
        function=_build_dce_transformed_mask_meta,
    )
    dce.inputs.trait_set(
        **common,
        units=None,
        extra={"LabelMap": HDGLIO_BIDS_LABELS},
    )

    for result in (dsc.run(), dce.run()):
        assert result.outputs.meta["LabelMap"] == HDGLIO_BIDS_LABELS
        assert len(result.outputs.meta["Sources"]) == 2


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


def test_tb1map_provenance_includes_grid_and_selected_b1_sources() -> None:
    from pmriprep.workflows.dce.datasinks import _build_tb1_meta

    base = {
        "Sources": ["bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_run-2_dce.nii.gz"],
        "SpatialReference": "bids::sub-X/ses-Y/perf/ref.nii.gz",
    }
    out = _build_tb1_meta(
        base,
        [
            "/raw/sub-X/ses-Y/fmap/sub-X_ses-Y_run-1_TB1map.nii.gz",
            "/raw/sub-X/ses-Y/fmap/sub-X_ses-Y_run-1_TB1map.nii.gz",
        ],
        "/raw",
        {"b1_mapping": {
            "method": "precomputed", "encoding": "percent",
            "n_valid_voxels": 123, "qc_pass": True,
        }},
    )
    assert out["Sources"] == [
        "bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_run-2_dce.nii.gz",
        "bids:raw:sub-X/ses-Y/fmap/sub-X_ses-Y_run-1_TB1map.nii.gz",
    ]
    assert out["Units"] == "1"
    assert out["MissingValue"] == 0
    assert out["B1Mapping"] == {
        "method": "precomputed", "encoding": "percent",
        "n_valid_voxels": 123, "qc_pass": True,
    }


def test_master_b1_source_selection_matches_configured_method() -> None:
    from pmriprep.config.dce import DCEConfig
    from pmriprep.workflows.base import _selected_dce_b1_sources

    session = SimpleNamespace(modalities=SimpleNamespace(
        b1_precomputed_file=Path("/raw/sub-X_TB1map.nii.gz"),
        afi_mode="4d",
        afi_4d_file=Path("/raw/sub-X_TB1AFI.nii.gz"),
    ))
    config = DCEConfig()
    config.b1_method = "precomputed"
    assert _selected_dce_b1_sources(session, config) == [
        "/raw/sub-X_TB1map.nii.gz"
    ]
    config.b1_method = "afi"
    assert _selected_dce_b1_sources(session, config) == [
        "/raw/sub-X_TB1AFI.nii.gz"
    ]
    config.b1_method = "none"
    assert _selected_dce_b1_sources(session, config) == []


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
