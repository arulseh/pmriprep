"""Tests for ``pmriprep.utils.bids.validate_bids_input``."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from pmriprep.utils.bids import validate_bids_input


def test_skip_flag_is_noop(monkeypatch, tmp_path: Path) -> None:
    """skip=True must not invoke the validator subprocess."""
    called: list[object] = []

    def fake_run(*args, **kwargs):
        called.append(args)
        raise AssertionError("subprocess.run must not be called when skip=True")

    monkeypatch.setattr(subprocess, "run", fake_run)
    validate_bids_input(tmp_path, skip=True)
    assert called == []


def test_missing_binary_warns(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing bids-validator binary must warn and return without raising."""
    monkeypatch.setattr("pmriprep.utils.bids.shutil.which", lambda _: None)
    caplog.set_level(logging.WARNING, logger="pmriprep.utils.bids")
    validate_bids_input(tmp_path, skip=False)
    assert any(
        "bids-validator not found" in rec.message for rec in caplog.records
    ), caplog.records


def test_validator_failure_exits(monkeypatch, tmp_path: Path) -> None:
    """Non-zero validator exit must SystemExit(1)."""
    monkeypatch.setattr(
        "pmriprep.utils.bids.shutil.which", lambda _: "/fake/bids-validator"
    )

    class FakeResult:
        returncode = 1

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    with pytest.raises(SystemExit) as ei:
        validate_bids_input(tmp_path, skip=False)
    assert ei.value.code == 1


def test_validator_pass_no_exit(monkeypatch, tmp_path: Path) -> None:
    """Zero validator exit must return None without raising."""
    monkeypatch.setattr(
        "pmriprep.utils.bids.shutil.which", lambda _: "/fake/bids-validator"
    )

    class FakeResult:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    assert validate_bids_input(tmp_path, skip=False) is None
