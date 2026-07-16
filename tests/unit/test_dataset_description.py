"""Tests for ``write_dataset_description`` merge semantics.

Covers the multi-tool ``GeneratedBy`` accumulation case (e.g. when a
NiPreps user runs fmriprep into the same derivatives root before
pmriprep) and the malformed-JSON fallback path.
"""
from __future__ import annotations

import json
from pathlib import Path

from pmriprep.utils.outputs import (
    BIDS_DERIVATIVES_VERSION,
    PMRIPREP_CODE_URL,
    write_bidsignore,
    write_dataset_description,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_first_run_writes_pmriprep_entry(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))

    desc = _read(pipeline_dir / "dataset_description.json")
    assert desc["BIDSVersion"] == BIDS_DERIVATIVES_VERSION
    assert desc["DatasetType"] == "derivative"
    names = [e["Name"] for e in desc["GeneratedBy"]]
    # pmriprep first, then the per-suite third-party tool entries.
    assert names[0] == "pmriprep"
    assert {"FSL", "ANTs", "Nipype", "HD-BET", "HD-GLIO"} <= set(names)
    pmri = desc["GeneratedBy"][0]
    assert pmri["CodeURL"] == PMRIPREP_CODE_URL
    assert "DOI" not in pmri  # not provided
    # SourceDatasets uses canonical bids:: prefix, not the bids:raw: typo
    assert desc["SourceDatasets"][0]["URL"].startswith("bids::")
    # .bidsignore written alongside dataset_description.json
    bidsignore = pipeline_dir / ".bidsignore"
    assert bidsignore.exists()
    content = bidsignore.read_text()
    assert "**/*.html" in content
    assert "**/*_xfm.h5" in content
    assert "**/figures/" in content
    assert "**/log/" in content
    assert "**/logs/" in content


def test_write_bidsignore_idempotent(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    write_bidsignore(pipeline_dir)
    first = (pipeline_dir / ".bidsignore").read_text()
    write_bidsignore(pipeline_dir)
    assert (pipeline_dir / ".bidsignore").read_text() == first


def test_preserves_prior_tool_generated_by(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "dataset_description.json").write_text(json.dumps({
        "Name": "fmriprep",
        "BIDSVersion": "1.7.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "fmriprep", "Version": "23.0.0"}],
    }))

    write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))
    desc = _read(pipeline_dir / "dataset_description.json")

    names = [e["Name"] for e in desc["GeneratedBy"]]
    assert "fmriprep" in names
    assert "pmriprep" in names
    fmri = next(e for e in desc["GeneratedBy"] if e["Name"] == "fmriprep")
    assert fmri["Version"] == "23.0.0"


def test_re_run_replaces_stale_pmriprep_entry(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))
    write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))

    desc = _read(pipeline_dir / "dataset_description.json")
    pmri = [e for e in desc["GeneratedBy"] if e["Name"] == "pmriprep"]
    assert len(pmri) == 1


def test_citation_doi_attached_when_provided(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    write_dataset_description(
        pipeline_dir, str(tmp_path / "raw_bids"),
        citation_doi="10.1234/pmriprep.5678",
    )
    desc = _read(pipeline_dir / "dataset_description.json")
    pmri = next(e for e in desc["GeneratedBy"] if e["Name"] == "pmriprep")
    assert pmri["DOI"] == "10.1234/pmriprep.5678"


def test_handles_malformed_existing_json(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "dataset_description.json").write_text("{not valid json")

    # Must not raise; falls back to fresh GeneratedBy (pmriprep + tool suites).
    write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))
    desc = _read(pipeline_dir / "dataset_description.json")
    names = [e["Name"] for e in desc["GeneratedBy"]]
    assert names[0] == "pmriprep"
    assert {"FSL", "ANTs", "Nipype", "HD-BET", "HD-GLIO"} <= set(names)


