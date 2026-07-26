"""Work-directory lifecycle tests for the PMRIPrep CLI."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pmriprep.cli.run import _clean_work_dir, _configure_nipype, _validate_work_dir


def test_clean_workdir_clears_contents_and_preserves_root(tmp_path: Path) -> None:
    """Cleanup clears prior cache contents but preserves the selected root."""
    work_root = tmp_path / "work"
    cache = work_root / "pmriprep_sub_01_wf"
    cache.mkdir(parents=True)
    (cache / "result.pklz").write_text("cached")

    assert _clean_work_dir(work_root) == work_root
    assert work_root.is_dir()
    assert list(work_root.iterdir()) == []


def test_clean_workdir_missing_root_is_noop(tmp_path: Path) -> None:
    """An absent work directory is a no-op."""
    work_root = tmp_path / "work"
    assert _clean_work_dir(work_root) is None


@pytest.mark.parametrize("candidate", ["bids", "out"])
def test_validate_workdir_rejects_data_and_output_roots(
    tmp_path: Path, candidate: str
) -> None:
    """Cleanup can never target the input dataset or derivative root."""
    bids = tmp_path / "bids"
    output = tmp_path / "out"
    bids.mkdir()
    output.mkdir()
    with pytest.raises(ValueError, match="Unsafe work directory"):
        _validate_work_dir(tmp_path / candidate, bids, output)


def test_validate_workdir_rejects_parent_of_bids(tmp_path: Path) -> None:
    """Cleanup can never target a directory containing protected data."""
    bids = tmp_path / "inputs" / "bids"
    output = tmp_path / "out"
    bids.mkdir(parents=True)
    output.mkdir()
    with pytest.raises(ValueError, match="Unsafe work directory"):
        _validate_work_dir(tmp_path / "inputs", bids, output)

    with pytest.raises(ValueError, match="Unsafe work directory"):
        _validate_work_dir(tmp_path, bids, output)


def test_validate_workdir_accepts_dedicated_path(tmp_path: Path) -> None:
    """A dedicated work root alongside the data and outputs is valid."""
    bids = tmp_path / "bids"
    output = tmp_path / "out"
    bids.mkdir()
    output.mkdir()
    assert _validate_work_dir(tmp_path / "work", bids, output) == tmp_path / "work"


def test_resource_monitor_flag_enables_nipype_monitoring(tmp_path: Path) -> None:
    """The public flag reaches Nipype's process resource profiler."""
    from nipype import config as nipype_config

    nipype_config.disable_resource_monitor()
    try:
        _configure_nipype(
            SimpleNamespace(
                stop_on_first_crash=False,
                resource_monitor=True,
                nprocs=1,
                mem_gb=None,
            ),
            log_dir=tmp_path,
        )
        assert nipype_config.get("monitoring", "enabled") == "true"
    finally:
        nipype_config.disable_resource_monitor()
