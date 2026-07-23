"""Work-directory lifecycle tests for the PMRIPrep CLI."""

from pathlib import Path

from pmriprep.cli.run import _clean_work_dir


def test_clean_workdir_removes_work_root(tmp_path: Path) -> None:
    """Cleanup removes the user-selected work directory and its contents."""
    work_root = tmp_path / "work"
    cache = work_root / "pmriprep_sub_01_wf"
    cache.mkdir(parents=True)
    (cache / "result.pklz").write_text("cached")

    assert _clean_work_dir(work_root) == work_root
    assert not work_root.exists()


def test_clean_workdir_missing_root_is_noop(tmp_path: Path) -> None:
    """An absent work directory is a no-op."""
    work_root = tmp_path / "work"
    assert _clean_work_dir(work_root) is None
