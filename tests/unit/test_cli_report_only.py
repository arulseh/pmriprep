"""Tests for report-only CLI execution."""

from argparse import Namespace
from pathlib import Path
from typing import Any

from pmriprep.cli.run import _run_report_only


def test_report_only_detects_existing_dce_outputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Report-only mode detects a session and calls the unified report builder."""
    bids_dir = tmp_path / "bids"
    output_dir = tmp_path / "derivatives"
    ses_dir = output_dir / "pmriprep" / "sub-01" / "ses-02"
    perf_dir = ses_dir / "perf"
    perf_dir.mkdir(parents=True)
    bids_dir.mkdir()
    (perf_dir / "sub-01_ses-02_desc-dcePreproc_dce.json").touch()

    monkeypatch.setattr(
        "pmriprep.cli.run._find_sessions",
        lambda *_args, **_kwargs: ["02"],
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "pmriprep.utils.report_utils.generate_unified_report",
        lambda **kwargs: calls.append(kwargs),
    )

    args = Namespace(
        bids_dir=bids_dir,
        output_dir=output_dir,
        session_label=None,
    )
    _run_report_only(args, ["01"])

    assert len(calls) == 1
    assert calls[0]["subject_id"] == "01"
    assert calls[0]["session_id"] == "02"
    assert calls[0]["output_path"] == str(ses_dir / "sub-01_ses-02.html")
    assert calls[0]["figures_dir"] == str(ses_dir / "figures")
    assert calls[0]["ran_pipelines"] == {
        "anat": False,
        "tumor_seg": False,
        "dsc": False,
        "dce": True,
    }
    assert calls[0]["command_string"] == "(report regenerated)"
