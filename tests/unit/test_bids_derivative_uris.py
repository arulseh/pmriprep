"""Tests for raw-source and within-derivative BIDS URIs."""

from pathlib import Path

from pmriprep.utils.sidecar_builders import (
    build_spatial_reference_uri,
    make_bids_source_uri,
)


def test_raw_source_uses_dataset_link_alias(tmp_path: Path) -> None:
    bids_dir = tmp_path / "raw"
    source = bids_dir / "sub-01" / "perf" / "sub-01_dce.nii.gz"
    source.parent.mkdir(parents=True)
    source.touch()

    assert make_bids_source_uri(source, bids_dir) == (
        "bids:raw:sub-01/perf/sub-01_dce.nii.gz"
    )


def test_external_source_uses_file_uri(tmp_path: Path) -> None:
    bids_dir = tmp_path / "raw"
    bids_dir.mkdir()
    source = tmp_path / "external" / "T1map.nii.gz"
    source.parent.mkdir()
    source.touch()

    assert make_bids_source_uri(source, bids_dir) == source.resolve().as_uri()


def test_spatial_reference_is_relative_to_derivative_root() -> None:
    uri = build_spatial_reference_uri(
        "desc-dceMean_stat.nii.gz",
        subject="01",
        session="02",
        datatype="perf",
    )

    assert uri == (
        "bids::sub-01/ses-02/perf/"
        "sub-01_ses-02_desc-dceMean_stat.nii.gz"
    )
