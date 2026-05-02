"""Tests for the niworkflows-backed enumeration helpers in pmriprep.utils.bids."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from pmriprep.utils.bids import collect_participants_pmriprep, collect_sessions


def _write_dd(root: Path) -> None:
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Test", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )


def _make_subject(root: Path, sub: str, sessions: list[str | None]) -> None:
    sub_dir = root / f"sub-{sub}"
    sub_dir.mkdir()
    for ses in sessions:
        if ses is None:
            anat = sub_dir / "anat"
        else:
            anat = sub_dir / f"ses-{ses}" / "anat"
        anat.mkdir(parents=True)
        prefix = f"sub-{sub}" + (f"_ses-{ses}" if ses else "")
        (anat / f"{prefix}_T1w.nii.gz").touch()
        (anat / f"{prefix}_T1w.json").write_text(
            json.dumps({"RepetitionTime": 2.0, "EchoTime": 0.003})
        )


@pytest.fixture
def two_subject_two_session(tmp_path: Path) -> Path:
    bids = tmp_path / "bids"
    bids.mkdir()
    _write_dd(bids)
    _make_subject(bids, "01", ["0001", "0002"])
    _make_subject(bids, "02", ["0001", "0002"])
    return bids


def test_collect_participants_returns_all(two_subject_two_session: Path) -> None:
    assert collect_participants_pmriprep(two_subject_two_session) == ["01", "02"]


def test_collect_participants_filtered(two_subject_two_session: Path) -> None:
    assert collect_participants_pmriprep(
        two_subject_two_session, participant_label=["02"]
    ) == ["02"]


def test_collect_sessions_returns_none_for_sessionless(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    bids.mkdir()
    _write_dd(bids)
    _make_subject(bids, "01", [None])
    assert collect_sessions(bids, "01") == [None]


def test_collect_sessions_returns_sorted(two_subject_two_session: Path) -> None:
    assert collect_sessions(two_subject_two_session, "01") == ["0001", "0002"]


def test_collect_sessions_filter_warns_on_missing(
    two_subject_two_session: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="pmriprep.utils.bids")
    result = collect_sessions(
        two_subject_two_session, "01", session_label=["0001", "0099"]
    )
    assert result == ["0001"]
    assert any("ses-0099" in rec.message for rec in caplog.records)


def test_legacy_and_layout_paths_agree(two_subject_two_session: Path) -> None:
    """Reusing a layout must produce identical results to building fresh."""
    from bids.layout import BIDSLayout

    layout = BIDSLayout(str(two_subject_two_session), validate=False)
    via_layout = collect_sessions(two_subject_two_session, "01", layout=layout)
    via_path = collect_sessions(two_subject_two_session, "01")
    assert via_layout == via_path
    parts_layout = collect_participants_pmriprep(
        two_subject_two_session, layout=layout
    )
    parts_path = collect_participants_pmriprep(two_subject_two_session)
    assert parts_layout == parts_path
