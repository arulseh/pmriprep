"""Work-directory lifecycle tests for the PMRIPrep CLI."""

from pathlib import Path

from pmriprep.cli.run import _clean_active_work_dir
from pmriprep.workflows.base import versioned_work_dir


def test_clean_workdir_preserves_other_versions_and_legacy_cache(
    tmp_path: Path,
) -> None:
    """Cleanup removes only the active release namespace."""
    work_root = tmp_path / "work"
    active = versioned_work_dir(work_root, "0.82.249")
    other = versioned_work_dir(work_root, "0.82.248")
    active.mkdir(parents=True)
    other.mkdir(parents=True)
    (active / "result.pklz").write_text("active")
    (other / "result.pklz").write_text("other")
    legacy = work_root / "pmriprep_sub_01_wf"
    legacy.mkdir()

    removed = _clean_active_work_dir(work_root, "0.82.249")

    assert removed == active
    assert not active.exists()
    assert (other / "result.pklz").read_text() == "other"
    assert legacy.exists()
    assert work_root.exists()


def test_clean_workdir_removes_empty_root(tmp_path: Path) -> None:
    """Cleanup removes the user root only when no other cache remains."""
    work_root = tmp_path / "work"
    active = versioned_work_dir(work_root, "0.82.249")
    active.mkdir(parents=True)

    assert _clean_active_work_dir(work_root, "0.82.249") == active
    assert not work_root.exists()


def test_clean_workdir_missing_namespace_is_noop(tmp_path: Path) -> None:
    """An absent active namespace does not disturb the work root."""
    work_root = tmp_path / "work"
    work_root.mkdir()
    legacy = work_root / "legacy.txt"
    legacy.write_text("keep")

    assert _clean_active_work_dir(work_root, "0.82.249") is None
    assert legacy.read_text() == "keep"
