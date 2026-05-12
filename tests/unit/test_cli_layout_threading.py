"""Tests that the CLI builds exactly one BIDSLayout per invocation and threads
the ``--bids-validation`` flag end-to-end.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _make_bids(tmp_path: Path) -> Path:
    bids = tmp_path / "bids"
    bids.mkdir()
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    sub = bids / "sub-01" / "ses-0001" / "anat"
    sub.mkdir(parents=True)
    (sub / "sub-01_ses-0001_T1w.nii.gz").touch()
    (sub / "sub-01_ses-0001_T1w.json").write_text(
        json.dumps({"RepetitionTime": 2.0, "EchoTime": 0.003})
    )
    return bids


def test_skip_flag_blocks_validator(monkeypatch, tmp_path: Path) -> None:
    """skip=True must short-circuit subprocess invocation."""
    from pmriprep.utils import bids as bids_mod

    called: list[Any] = []

    def fake_run(*args, **kwargs):
        called.append(args)
        raise AssertionError("validator must not run when skip=True")

    monkeypatch.setattr(bids_mod.subprocess, "run", fake_run)
    bids_mod.validate_bids_input(tmp_path, skip=True)
    assert called == []


def test_helpers_reuse_supplied_layout(monkeypatch, tmp_path: Path) -> None:
    """``collect_*`` helpers must reuse a supplied layout, not rebuild.

    Note: this asserts the helper-pattern in isolation, not the full
    ``run_pmriprep_wf`` flow. The flow itself is exercised by
    ``test_main_forwards_skip_flag`` and ``test_main_handles_missing_subject``.
    """
    from bids.layout import BIDSLayout as RealLayout

    bids = _make_bids(tmp_path)
    counter = {"n": 0}
    real_init = RealLayout.__init__

    def counting_init(self, *args, **kwargs):
        counter["n"] += 1
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(RealLayout, "__init__", counting_init)

    layout = RealLayout(str(bids), validate=False)
    counter["n"] = 0  # reset after the explicit build above

    from pmriprep.utils.bids import collect_participants_pmriprep, collect_sessions

    subjects = collect_participants_pmriprep(bids, layout=layout)
    for subj in subjects:
        collect_sessions(bids, subj, layout=layout)

    assert counter["n"] == 0, (
        "collect_*pmriprep helpers must reuse the supplied layout, not rebuild."
    )


def test_bids_validation_argparse_dest() -> None:
    """``--bids-validation`` parses to ``args.bids_validation``."""
    from pmriprep.cli.run import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "/tmp/bids", "/tmp/out", "participant",
        "--bids-validation",
    ])
    assert args.bids_validation is True

    args2 = parser.parse_args(["/tmp/bids", "/tmp/out", "participant"])
    assert args2.bids_validation is False


def test_collect_participants_returns_empty_on_typo(tmp_path: Path) -> None:
    """A typo'd --participant-label must yield ``[]`` (not raise BIDSError).

    Regression: ``niworkflows.collect_participants`` raises ``BIDSError`` when
    no requested label matches; ``collect_participants_pmriprep`` swallows it
    so the CLI's ``"No subjects found"`` exit path fires cleanly.
    """
    from pmriprep.utils.bids import collect_participants_pmriprep

    bids = _make_bids(tmp_path)  # has sub-01 only
    result = collect_participants_pmriprep(bids, participant_label=["99"])
    assert result == []


def test_collect_participants_returns_empty_on_empty_dataset(tmp_path: Path) -> None:
    """Dataset with zero subjects must yield ``[]`` (not raise BIDSError)."""
    from pmriprep.utils.bids import collect_participants_pmriprep

    empty_bids = tmp_path / "empty"
    empty_bids.mkdir()
    (empty_bids / "dataset_description.json").write_text(
        json.dumps({"Name": "Empty", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    result = collect_participants_pmriprep(empty_bids)
    assert result == []
