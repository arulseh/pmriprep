"""Tests for datasink Function-node path conventions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from traits.trait_base import Undefined


def test_dce_transform_sink_names_both_directions(tmp_path: Path) -> None:
    """DCE X5 filenames must describe the transform direction they contain."""
    from pmriprep.workflows.dce.datasinks import _copy_dce_xfm

    source = tmp_path / "transform.h5"
    source.write_bytes(b"x5")
    output_dir = tmp_path / "derivatives"

    forward = _copy_dce_xfm(
        str(source), str(output_dir), "01", "02", "T1w", "dceref",
    )
    inverse = _copy_dce_xfm(
        str(source), str(output_dir), "01", "02", "dceref", "T1w",
    )

    assert Path(forward).name == (
        "sub-01_ses-02_from-T1w_to-dceref_mode-image_xfm.h5"
    )
    assert Path(inverse).name == (
        "sub-01_ses-02_from-dceref_to-T1w_mode-image_xfm.h5"
    )
    assert Path(forward).read_bytes() == source.read_bytes()
    assert Path(inverse).read_bytes() == source.read_bytes()


def test_dce_transform_sink_rejects_fsl_matrix(tmp_path: Path) -> None:
    """A FLIRT matrix must not be silently relabelled as an X5 derivative."""
    import pytest

    from pmriprep.workflows.dce.datasinks import _copy_dce_xfm

    source = tmp_path / "transform.mat"
    source.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    with pytest.raises(ValueError, match="requires an X5 .h5 input"):
        _copy_dce_xfm(
            str(source), str(tmp_path / "out"), "01", "", "T1w", "dceref",
        )


def test_anat_transform_sink_requires_x5(tmp_path: Path) -> None:
    """An anatomical FLIRT matrix must not be relabelled as X5."""
    import pytest

    from pmriprep.workflows.anat.datasinks import _copy_xfm

    source = tmp_path / "transform.mat"
    source.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    with pytest.raises(ValueError, match="require X5 .h5 input"):
        _copy_xfm(
            str(source), str(tmp_path / "out"), "01", "", "T1w",
            "MNI152NLin6Asym",
        )


def test_anat_transform_sink_preserves_x5(tmp_path: Path) -> None:
    """An X5 transform is copied under its truthful BIDS direction."""
    from pmriprep.workflows.anat.datasinks import _copy_xfm

    source = tmp_path / "transform.h5"
    source.write_bytes(b"x5")

    result = _copy_xfm(
        str(source), str(tmp_path / "out"), "01", "02", "T1w",
        "MNI152NLin6Asym",
    )

    assert Path(result).name == (
        "sub-01_ses-02_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5"
    )
    assert Path(result).read_bytes() == source.read_bytes()


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
        "/raw/sub-X/ses-Y/perf/sub-X_ses-Y_dce.nii.gz", "/raw", True, None,
    )
    dsc_meta = _build_transformed_mask_meta(
        base, "Tumor segmentation", False, None,
        structural_sources,
        "/raw/sub-X/ses-Y/perf/sub-X_ses-Y_dsc.nii.gz", "/raw",
        HDGLIO_BIDS_LABELS, True, None,
    )
    for meta in (dce_meta, dsc_meta):
        assert len(meta["Sources"]) == 3
        assert meta["LabelMap"] == HDGLIO_BIDS_LABELS
    assert dsc_meta["ResamplingApplied"] is True
    assert dsc_meta["Interpolation"] == "nearest_neighbour"
    assert dsc_meta["TransformDirection"] == "T1w_to_DSC"
    assert dsc_meta["LabelPreservation"] is True
    assert dce_meta["ResamplingApplied"] is True
    assert dce_meta["Interpolation"] == "nearest_neighbour"
    assert dce_meta["TransformDirection"] == "T1w_to_DCE"
    assert dce_meta["LabelPreservation"] is True


def test_dsc_native_nawm_metadata_does_not_claim_resampling() -> None:
    from pmriprep.workflows.dsc.datasinks import _build_transformed_mask_meta

    meta = _build_transformed_mask_meta(
        {},
        "Normal-appearing white matter mask in DSC space",
        False,
        "ROI",
        [],
        "/raw/sub-X/perf/sub-X_dsc.nii.gz",
        "/raw",
        None,
        False,
        "/raw/sub-X/perf/sub-X_desc-userNawm_mask.nii.gz",
    )

    assert meta["ResamplingApplied"] is False
    assert meta["Interpolation"] == "none"
    assert meta["ReorientationApplied"] is True
    assert meta["GeometryValidation"] == "shape_and_affine_against_DSC_reference"
    assert "TransformDirection" not in meta
    assert meta["Sources"][0].endswith("sub-X_desc-userNawm_mask.nii.gz")


def test_dsc_underlay_metadata_records_resampling() -> None:
    from pmriprep.workflows.dsc.datasinks import _build_underlay_meta

    meta = _build_underlay_meta(
        {},
        "FLAIR registered to DSC space",
        False,
        None,
        "/raw/sub-X/anat/sub-X_FLAIR.nii.gz",
        "/raw/sub-X/perf/sub-X_dsc.nii.gz",
        "/raw",
    )

    assert meta["ResamplingApplied"] is True
    assert meta["Interpolation"] == "spline"
    assert meta["TransformDirection"] == "T1w_to_DSC"


def test_dce_native_nawm_metadata_does_not_claim_resampling() -> None:
    from pmriprep.workflows.dce.datasinks import _build_dce_transformed_mask_meta

    meta = _build_dce_transformed_mask_meta(
        {},
        "Normal-appearing white matter mask in DCE space",
        None,
        {"Type": "ROI"},
        [],
        "/raw/sub-X/perf/sub-X_dce.nii.gz",
        "/raw",
        False,
        "/raw/sub-X/perf/sub-X_desc-userNawm_mask.nii.gz",
    )

    assert meta["ResamplingApplied"] is False
    assert meta["Interpolation"] == "none"
    assert meta["ReorientationApplied"] is True
    assert meta["GeometryValidation"] == "shape_and_affine_against_DCE_reference"
    assert "TransformDirection" not in meta
    assert meta["Sources"][0].endswith("sub-X_desc-userNawm_mask.nii.gz")


def test_dce_underlay_metadata_records_resampling() -> None:
    from pmriprep.workflows.dce.datasinks import _build_dce_underlay_meta

    meta = _build_dce_underlay_meta(
        {},
        "FLAIR registered to DCE space",
        None,
        None,
        "/raw/sub-X/anat/sub-X_FLAIR.nii.gz",
        "/raw/sub-X/perf/sub-X_dce.nii.gz",
        "/raw",
        False,
    )

    assert meta["ResamplingApplied"] is True
    assert meta["Interpolation"] == "spline"
    assert meta["TransformDirection"] == "T1w_to_DCE"


def test_native_user_mask_metadata_records_exact_analysis_source() -> None:
    from pmriprep.workflows.dce.datasinks import _ANAT_MASKS as dce_specs
    from pmriprep.workflows.dce.datasinks import _build_dce_native_mask_meta
    from pmriprep.workflows.dsc.datasinks import _ANAT_MASKS as dsc_specs
    from pmriprep.workflows.dsc.datasinks import _build_native_mask_meta

    dce_roi = next(spec for spec in dce_specs if spec.in_attr == "roi_mask_dce")
    dsc_roi = next(spec for spec in dsc_specs if spec.in_attr == "roi_mask_dsc")
    assert dce_roi.description == "User-supplied ROI mask in DCE native space"
    assert dsc_roi.description == "User-supplied ROI mask in DSC native space"

    dsc = _build_native_mask_meta(
        {},
        "User-supplied lesion mask in DSC native space",
        False,
        "Lesion",
        "/raw/sub-X/perf/sub-X_desc-userLesion_mask.nii.gz",
        "/raw/sub-X/perf/sub-X_dsc.nii.gz",
        "/raw",
    )
    dce = _build_dce_native_mask_meta(
        {},
        "ROI mask in DCE native space",
        None,
        {"Type": "ROI"},
        "/raw/sub-X/perf/sub-X_desc-userRoi_mask.nii.gz",
        "/raw/sub-X/perf/sub-X_dce.nii.gz",
        "/raw",
    )

    for meta, source_name, reference in (
        (dsc, "sub-X_desc-userLesion_mask.nii.gz", "DSC"),
        (dce, "sub-X_desc-userRoi_mask.nii.gz", "DCE"),
    ):
        assert meta["ResamplingApplied"] is False
        assert meta["Interpolation"] == "none"
        assert meta["ReorientationApplied"] is True
        assert meta["GeometryValidation"] == (
            f"shape_and_affine_against_{reference}_reference"
        )
        assert meta["Sources"][0].endswith(source_name)


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
            "segmentation_labels", "resampling_applied", "native_source",
        ],
        output_names=["meta"],
        function=_build_transformed_mask_meta,
    )
    dsc.inputs.trait_set(
        **common,
        skull_stripped=False,
        mask_type=None,
        segmentation_labels=HDGLIO_BIDS_LABELS,
        resampling_applied=True,
        native_source=None,
    )
    dce = niu.Function(
        input_names=[
            "base", "description", "units", "extra", "structural_sources",
            "perfusion_source", "bids_dir", "resampling_applied", "native_source",
        ],
        output_names=["meta"],
        function=_build_dce_transformed_mask_meta,
    )
    dce.inputs.trait_set(
        **common,
        units=None,
        extra={"LabelMap": HDGLIO_BIDS_LABELS},
        resampling_applied=True,
        native_source=None,
    )

    for result in (dsc.run(), dce.run()):
        assert result.outputs.meta["LabelMap"] == HDGLIO_BIDS_LABELS
        assert len(result.outputs.meta["Sources"]) == 2


def test_dce_mean_meta_identifies_all_volume_aggregation() -> None:
    from pmriprep.workflows.dce.datasinks import _build_dce_mean_meta

    out = _build_dce_mean_meta(
        {},
        {"preprocessing": {"n_volumes": 75, "baseline_frames_used": 8}},
        motion_correct=False,
    )

    assert out["TemporalAggregation"] == "arithmetic_mean_all_acquired_volumes"
    assert out["IncludesPostContrastVolumes"] is True
    assert out["MotionCorrectionApplied"] is False
    assert out["SkullStripped"] is False
    assert out["NumberOfVolumes"] == 75
    assert out["BaselineFrames"] == 8


def _aif_mask_copy(modality: str) -> Callable[..., str]:
    if modality == "dsc":
        from pmriprep.workflows.dsc.datasinks import _copy_aif_mask
    else:
        from pmriprep.workflows.dce.datasinks import _copy_aif_mask
    return _copy_aif_mask


@pytest.mark.parametrize("modality", ["dsc", "dce"])
def test_aif_mask_sink_skips_nonspatial_curve(tmp_path: Path, modality: str) -> None:
    source = tmp_path / "empty_mask.nii.gz"
    source.write_bytes(b"placeholder")
    output_dir = tmp_path / "out"

    out = _aif_mask_copy(modality)(
        in_file=str(source), output_dir=str(output_dir), subject="X", session="Y",
        desc=f"{modality}Aif", suffix="",
        meta_dict={"SpatialSupportAvailable": False},
    )

    assert out == ""
    assert not output_dir.exists()


@pytest.mark.parametrize("modality", ["dsc", "dce"])
def test_aif_mask_sink_writes_measured_support(tmp_path: Path, modality: str) -> None:
    source = tmp_path / "aif_mask.nii.gz"
    source.write_bytes(b"measured support")
    output_dir = tmp_path / "out"
    metadata = {
        "Sources": [f"bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_{modality}.nii.gz"],
        "SpatialSupportAvailable": True,
    }

    out = _aif_mask_copy(modality)(
        in_file=str(source), output_dir=str(output_dir), subject="X", session="Y",
        desc=f"{modality}Aif", suffix="", meta_dict=metadata,
    )

    assert out.endswith(f"sub-X_ses-Y_desc-{modality}Aif_mask.nii.gz")
    assert Path(out).read_bytes() == b"measured support"
    assert Path(out).with_suffix("").with_suffix(".json").is_file()


def test_perfusion_spatial_reference_preserves_source_entities() -> None:
    from pmriprep.utils.sidecar_builders import _dce_spatial_reference

    source = "/raw/sub-X/ses-Y/perf/sub-X_ses-Y_acq-fast_run-2_dce.nii.gz"
    assert _dce_spatial_reference("X", "Y", source) == (
        "bids::sub-X/ses-Y/perf/"
        "sub-X_ses-Y_acq-fast_run-2_desc-dceMean_stat.nii.gz"
    )


def _perf_map_contract(modality: str) -> tuple[Callable[..., str], str, str, str]:
    if modality == "dsc":
        from pmriprep.workflows.dsc.datasinks import _copy_perf_map
        return _copy_perf_map, "dsc", "cbv", "sub-X_ses-Y_desc-dscCbv_stat.nii.gz"
    from pmriprep.workflows.dce.datasinks import _copy_perf_map
    return _copy_perf_map, "dceKtrans", "Ktrans", "sub-X_ses-Y_desc-dceKtransKtrans_stat.nii.gz"


@pytest.mark.parametrize("modality", ["dsc", "dce"])
@pytest.mark.parametrize("missing", [Undefined, ""], ids=["undefined", "empty"])
def test_perf_map_missing_input_writes_nothing(
    tmp_path: Path,
    modality: str,
    missing: Any,
) -> None:
    copy_map, desc, suffix, _ = _perf_map_contract(modality)
    out = copy_map(
        in_file=missing, output_dir=str(tmp_path), subject="X", session="Y",
        desc=desc, suffix=suffix, meta_dict={},
    )
    assert out == ""
    assert not list(tmp_path.rglob("*.nii.gz"))
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("modality", ["dsc", "dce"])
def test_perf_map_copies_image_and_sidecar(tmp_path: Path, modality: str) -> None:
    copy_map, desc, suffix, expected = _perf_map_contract(modality)
    source = tmp_path / "src.nii.gz"
    source.write_bytes(b"fake nifti")
    out = copy_map(
        in_file=str(source), output_dir=str(tmp_path / "out"), subject="X", session="Y",
        desc=desc, suffix=suffix, meta_dict={"Description": "test"},
    )
    assert out.endswith(expected)
    assert Path(out).read_bytes() == source.read_bytes()
    assert json.loads(Path(out.replace(".nii.gz", ".json")).read_text()) == {
        "Description": "test",
    }


@pytest.mark.parametrize("modality", ["dsc", "dce"])
def test_perf_map_source_run_entity_prevents_collision(tmp_path: Path, modality: str) -> None:
    copy_map, _, suffix, _ = _perf_map_contract(modality)
    source = tmp_path / "src.nii.gz"
    source.write_bytes(b"fake nifti")
    outputs = [
        copy_map(
            in_file=str(source), output_dir=str(tmp_path / "out"), subject="X", session="Y",
            desc=modality, suffix=suffix,
            meta_dict={
                "Sources": [
                    f"bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_run-{run}_{modality}.nii.gz"
                ]
            },
        )
        for run in (1, 2)
    ]
    assert outputs[0] != outputs[1]
    assert "_run-1_" in outputs[0]
    assert "_run-2_" in outputs[1]
