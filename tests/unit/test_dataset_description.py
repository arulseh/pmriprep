"""Tests for ``write_dataset_description`` merge semantics.

Covers the multi-tool ``GeneratedBy`` accumulation case (e.g. when a
NiPreps user runs fmriprep into the same derivatives root before
pmriprep) and the malformed-JSON fallback path.
"""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from pmriprep.utils.outputs import (
    BIDS_DERIVATIVES_VERSION,
    PMRIPREP_CODE_URL,
    write_bidsignore,
    write_dataset_description,
    write_derivative_readme,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_first_run_writes_pmriprep_entry(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))

    desc = _read(pipeline_dir / "dataset_description.json")
    assert desc["Name"] == (
        "PMRIPrep: automated preprocessing and quantification of DSC and DCE MRI"
    )
    assert desc["BIDSVersion"] == BIDS_DERIVATIVES_VERSION
    assert desc["DatasetType"] == "derivative"
    names = [e["Name"] for e in desc["GeneratedBy"]]
    # pmriprep first, then the per-suite third-party tool entries.
    assert names[0] == "pmriprep"
    assert {"FSL", "ANTs", "Nipype", "HD-BET", "HD-GLIO"} <= set(names)
    pmri = desc["GeneratedBy"][0]
    assert pmri["Description"] == (
        "Automated preprocessing and quantification of DSC and DCE MRI"
    )
    assert pmri["CodeURL"] == PMRIPREP_CODE_URL
    assert "DOI" not in pmri  # not provided
    assert desc["DatasetLinks"]["raw"] == "../raw_bids"
    assert desc["SourceDatasets"] == [{"URL": "../raw_bids"}]
    # .bidsignore written alongside dataset_description.json
    bidsignore = pipeline_dir / ".bidsignore"
    assert bidsignore.exists()
    content = bidsignore.read_text()
    assert "**/*.html" in content
    assert "**/*_xfm.h5" in content
    assert "**/figures/" in content
    assert "**/log/" in content
    assert "**/logs/" in content
    readme = pipeline_dir / "README"
    assert readme.exists()
    assert "PMRIPrep derivative dataset" in readme.read_text()


def test_keyword_arguments_are_supported(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    write_dataset_description(
        pipeline_dir=pipeline_dir,
        bids_dir=str(tmp_path / "raw_bids"),
    )
    assert _read(pipeline_dir / "dataset_description.json")["DatasetType"] == "derivative"


def test_write_bidsignore_idempotent(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    write_bidsignore(pipeline_dir)
    first = (pipeline_dir / ".bidsignore").read_text()
    write_bidsignore(pipeline_dir)
    assert (pipeline_dir / ".bidsignore").read_text() == first


def test_write_derivative_readme_idempotent(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    write_derivative_readme(pipeline_dir)
    first = (pipeline_dir / "README").read_text()
    write_derivative_readme(pipeline_dir)
    assert (pipeline_dir / "README").read_text() == first


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
    assert names[0] == "pmriprep"
    assert "fmriprep" in names
    assert "pmriprep" in names
    fmri = next(e for e in desc["GeneratedBy"] if e["Name"] == "fmriprep")
    assert fmri["Version"] == "23.0.0"


def test_preserves_foreign_dataset_links_and_source_datasets(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "dataset_description.json").write_text(json.dumps({
        "Name": "existing",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "other"}],
        "DatasetLinks": {"atlas": "../atlas"},
        "SourceDatasets": [
            {"URL": "../atlas", "DOI": "doi:10.1234/example"},
        ],
    }))

    write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))
    desc = _read(pipeline_dir / "dataset_description.json")

    assert desc["DatasetLinks"] == {
        "atlas": "../atlas",
        "raw": "../raw_bids",
    }
    assert desc["SourceDatasets"] == [
        {"URL": "../atlas", "DOI": "doi:10.1234/example"},
        {"URL": "../raw_bids"},
    ]


def test_rejects_conflicting_raw_dataset_link(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "dataset_description.json").write_text(json.dumps({
        "Name": "existing",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "pmriprep"}],
        "DatasetLinks": {"raw": "../different_raw"},
    }))

    with pytest.raises(ValueError, match="different raw dataset"):
        write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))


def test_accepts_equivalent_local_raw_dataset_link(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "dataset_description.json").write_text(json.dumps({
        "Name": "existing",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "pmriprep"}],
        "DatasetLinks": {"raw": "../raw_bids/"},
        "SourceDatasets": [{"URL": "../raw_bids/"}],
    }))

    write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))
    desc = _read(pipeline_dir / "dataset_description.json")
    assert desc["DatasetLinks"]["raw"] == "../raw_bids"
    assert desc["SourceDatasets"] == [{"URL": "../raw_bids/"}]


def test_rejects_reserved_empty_dataset_link(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "dataset_description.json").write_text(json.dumps({
        "Name": "existing",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "pmriprep"}],
        "DatasetLinks": {"": "../invalid"},
    }))

    with pytest.raises(ValueError, match="reserved empty dataset name"):
        write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))


def test_failed_atomic_replace_preserves_existing_description(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    desc_path = pipeline_dir / "dataset_description.json"
    original = json.dumps({
        "Name": "existing",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "pmriprep"}],
        "DatasetLinks": {"raw": "../raw_bids"},
    })
    desc_path.write_text(original)

    def _fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("pmriprep.utils.outputs.os.replace", _fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))

    assert desc_path.read_text() == original
    assert list(pipeline_dir.glob(".dataset_description.*.tmp")) == []


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


def test_rejects_malformed_existing_json(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "dataset_description.json").write_text("{not valid json")

    with pytest.raises(ValueError, match="not valid JSON"):
        write_dataset_description(pipeline_dir, str(tmp_path / "raw_bids"))


def _write_dataset_description_worker(pipeline_dir: str, bids_dir: str) -> None:
    write_dataset_description(Path(pipeline_dir), bids_dir)


def test_concurrent_writers_leave_valid_complete_metadata(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pmriprep"
    bids_dir = str(tmp_path / "raw_bids")
    processes = [
        multiprocessing.Process(
            target=_write_dataset_description_worker,
            args=(str(pipeline_dir), bids_dir),
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    desc = _read(pipeline_dir / "dataset_description.json")
    assert desc["DatasetType"] == "derivative"
    assert len([e for e in desc["GeneratedBy"] if e["Name"] == "pmriprep"]) == 1
