"""Work-directory lifecycle tests for the PMRIPrep CLI."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pmriprep.cli.run import (
    _clean_work_dir,
    _configure_nipype,
    _prepare_workflow_cache,
    _validate_work_dir,
)


def _make_bids_dir(root: Path) -> Path:
    """Create the minimum dataset identity used by cache-manifest tests."""
    root.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        '{"Name": "Test dataset", "BIDSVersion": "1.9.0"}\n',
        encoding="utf-8",
    )
    return root


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


def test_validate_workdir_rejects_parent_of_home(tmp_path: Path) -> None:
    """Cleanup can never target a directory containing the user home."""
    bids = tmp_path / "bids"
    output = tmp_path / "out"
    bids.mkdir()
    output.mkdir()
    home = Path.home().resolve()
    candidate = home.parent
    with pytest.raises(ValueError, match="Unsafe work directory"):
        _validate_work_dir(candidate, bids, output)


def test_validate_workdir_accepts_dedicated_path(tmp_path: Path) -> None:
    """A dedicated work root alongside the data and outputs is valid."""
    bids = tmp_path / "bids"
    output = tmp_path / "out"
    bids.mkdir()
    output.mkdir()
    assert _validate_work_dir(tmp_path / "work", bids, output) == tmp_path / "work"


def test_workflow_cache_manifest_is_scoped_to_subject_session(
    tmp_path: Path,
) -> None:
    """Unrelated caches do not block the selected subject/session namespace."""
    work_root = tmp_path / "work"
    unrelated = work_root / "pmriprep_sub_02_ses_01_wf"
    unrelated.mkdir(parents=True)
    (unrelated / "result_old.pklz").write_text("old", encoding="utf-8")
    bids_dir = _make_bids_dir(tmp_path / "bids")

    workflow_dir = _prepare_workflow_cache(
        work_root, bids_dir, "01", "01"
    )

    marker = workflow_dir / ".pmriprep-cache.json"
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    assert manifest["Subject"] == "01"
    assert manifest["Session"] == "01"
    assert manifest["BIDSDirectory"] == str(bids_dir.resolve())
    assert (unrelated / "result_old.pklz").exists()


def test_workflow_cache_manifest_permits_compatible_resume(tmp_path: Path) -> None:
    """A cache created by the current environment remains resumable."""
    work_root = tmp_path / "work"
    bids_dir = _make_bids_dir(tmp_path / "bids")
    workflow_dir = _prepare_workflow_cache(
        work_root, bids_dir, "01", None
    )
    (workflow_dir / "cached_result.pklz").write_text("cached", encoding="utf-8")

    assert (
        _prepare_workflow_cache(work_root, bids_dir, "01", None)
        == workflow_dir
    )


def test_workflow_cache_rejects_legacy_target_namespace(tmp_path: Path) -> None:
    """An occupied target without a manifest fails before Nipype cache loading."""
    workflow_dir = tmp_path / "work" / "pmriprep_sub_01_ses_01_wf"
    workflow_dir.mkdir(parents=True)
    cached = workflow_dir / "result_old.pklz"
    cached.write_text("old", encoding="utf-8")
    bids_dir = _make_bids_dir(tmp_path / "bids")

    with pytest.raises(RuntimeError, match="unknown PMRIPrep/Nipype environment"):
        _prepare_workflow_cache(
            tmp_path / "work", bids_dir, "01", "01"
        )
    assert cached.exists()


def test_workflow_cache_rejects_incompatible_manifest(tmp_path: Path) -> None:
    """Environment or dataset mismatches cannot silently reuse cached nodes."""
    work_root = tmp_path / "work"
    bids_dir = _make_bids_dir(tmp_path / "bids")
    workflow_dir = _prepare_workflow_cache(
        work_root, bids_dir, "01", "01"
    )
    marker = workflow_dir / ".pmriprep-cache.json"
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    manifest["NipypeVersion"] = "0.0"
    marker.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="NipypeVersion='0.0'"):
        _prepare_workflow_cache(
            work_root, bids_dir, "01", "01"
        )


def test_workflow_cache_allows_mount_path_change(
    tmp_path: Path,
) -> None:
    """The dataset-description identity survives remounting."""
    first_bids = _make_bids_dir(tmp_path / "mount_a" / "bids")
    second_bids = _make_bids_dir(tmp_path / "mount_b" / "bids")
    work_root = tmp_path / "work"
    workflow_dir = _prepare_workflow_cache(
        work_root, first_bids, "01", "01"
    )
    assert (
        _prepare_workflow_cache(work_root, second_bids, "01", "01")
        == workflow_dir
    )
    refreshed = json.loads(
        (workflow_dir / ".pmriprep-cache.json").read_text(encoding="utf-8")
    )
    assert refreshed["BIDSDirectory"] == str(second_bids.resolve())


def test_workflow_cache_rejects_pmriprep_version_change(tmp_path: Path) -> None:
    """Class-interface implementation changes cannot reuse prior results."""
    bids_dir = _make_bids_dir(tmp_path / "bids")
    work_root = tmp_path / "work"
    workflow_dir = _prepare_workflow_cache(
        work_root, bids_dir, "01", "01"
    )
    marker = workflow_dir / ".pmriprep-cache.json"
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    manifest["PMRIPrepVersion"] = "0.0"
    marker.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="PMRIPrepVersion='0.0'"):
        _prepare_workflow_cache(work_root, bids_dir, "01", "01")


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
