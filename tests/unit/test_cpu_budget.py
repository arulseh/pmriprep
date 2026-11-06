"""CPU limits must constrain defaults without overriding explicit budgets."""

from pathlib import Path

import pytest

from pmriprep.cli.run import _configure_thread_limits, _effective_nprocs
from pmriprep.utils import cpu


@pytest.mark.parametrize(
    "machine,affinity,slurm,quotas,expected",
    [(64, 64, None, [], 8), (4, 4, None, [], 4),
     (64, 3, "6", [5], 3), (64, 32, "2", [5], 2),
     (64, 32, "6", [1], 1), (None, None, "bad", [], 1)],
)
def test_automatic_budget_uses_smallest_detected_limit(
    monkeypatch: pytest.MonkeyPatch, machine: int | None, affinity: int | None,
    slurm: str | None, quotas: list[int], expected: int,
) -> None:
    """Each external limit can bind; malformed/unavailable hints are harmless."""
    monkeypatch.setattr(cpu.os, "cpu_count", lambda: machine)
    if affinity is None:
        monkeypatch.delattr(cpu.os, "sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda _: set(range(affinity)), raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    if slurm is not None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", slurm)
    monkeypatch.setattr(cpu, "_cgroup_cpu_limits", lambda: quotas)
    count, source = _effective_nprocs(None, None, {})
    assert count == expected
    assert source.startswith("automatic:")
    assert "default cap=8" in source


@pytest.mark.parametrize("version", [1, 2])
def test_nested_cgroup_quota_checks_parent_and_ignores_unlimited(
    tmp_path: Path, version: int,
) -> None:
    """An unlimited child still inherits its parent's finite CPU quota."""
    proc = tmp_path / "proc"
    mount = tmp_path / "cpu"
    leaf = mount / "job"
    proc.mkdir()
    leaf.mkdir(parents=True)
    if version == 2:
        (proc / "cgroup").write_text("0::/allocation/job\n")
        (proc / "mountinfo").write_text(f"1 0 0:1 /allocation {mount} rw - cgroup2 cgroup rw\n")
        (leaf / "cpu.max").write_text("max 100000\n")
        (mount / "cpu.max").write_text("250000 100000\n")
    else:
        (proc / "cgroup").write_text("2:cpu,cpuacct:/allocation/job\n")
        (proc / "mountinfo").write_text(f"1 0 0:1 /allocation {mount} rw - cgroup cgroup rw,cpu,cpuacct\n")
        for directory, quota in [(leaf, "-1"), (mount, "250000")]:
            (directory / "cpu.cfs_quota_us").write_text(quota)
            (directory / "cpu.cfs_period_us").write_text("100000")
    assert cpu._cgroup_cpu_limits(proc) == [2]
    (proc / "cgroup").write_text("0::/\n" if version == 2 else "2:cpu,cpuacct:/\n")
    assert cpu._cgroup_cpu_limits(proc) == [2]
    (proc / "mountinfo").write_text("unparseable mount\n")
    assert cpu._cgroup_cpu_limits(proc) == []


@pytest.mark.parametrize(
    "budget,requested,expected",
    [(1, None, 1), (4, None, 4), (52, None, 8), (52, 16, 16), (4, 8, 4)],
)
def test_thread_defaults_follow_budget_and_allow_override(
    monkeypatch: pytest.MonkeyPatch, budget: int, requested: int | None, expected: int,
) -> None:
    """Export the resolved count consistently to all four numerical libraries."""
    variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
    for variable in variables:
        monkeypatch.setenv(variable, "99")
    assert _configure_thread_limits(budget, requested) == expected
    assert all(cpu.os.environ[variable] == str(expected) for variable in variables)


def test_threaded_preprocessing_reservations_include_nested_n4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reserve actual tool threads while preserving serial nodes and fit pools."""
    import pickle

    import nipype.pipeline.engine as pe
    from nipype.interfaces.utility import IdentityInterface

    from pmriprep.cli.run import _configure_preprocessing_threads
    from pmriprep.interfaces.anat import BiasCorrection, BrainExtraction, Coregistration
    from pmriprep.interfaces.dce import DCEBrainExtract, DCEPreprocess
    from pmriprep.interfaces.dsc import MotionCorrection
    from pmriprep.interfaces.dsc_brain import DSCBrainExtract
    from pmriprep.interfaces.hdglio import HDGLIOSegment
    from pmriprep.workflows.anat.base import init_anat_tissue_segmentation_wf

    workflow = pe.Workflow(name="resources")
    nested = init_anat_tissue_segmentation_wf(denoise=True)
    threaded = [pe.Node(interface(), name=interface.__name__.lower()) for interface in (
        BiasCorrection, BrainExtraction, DCEBrainExtract, DSCBrainExtract, HDGLIOSegment,
        MotionCorrection, DCEPreprocess,
    )]
    for node in threaded:
        if isinstance(node.interface, BrainExtraction):
            node.inputs.device = "cpu"
        elif isinstance(node.interface, (DCEBrainExtract, DSCBrainExtract)):
            node.inputs.hdbet_device = "cpu"
    serial = [pe.Node(Coregistration(), name="flirt"),
              pe.Node(DCEBrainExtract(method="threshold"), name="threshold")]
    fit_pool = pe.Node(IdentityInterface(fields=["value"]), name="fitting", n_procs=4)
    workflow.add_nodes([nested, fit_pool, *threaded, *serial])
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        monkeypatch.setenv(variable, "99")
    for budget, requested, expected in [(4, 8, 4), (8, 2, 2)]:
        threads = _configure_thread_limits(budget, requested)
        _configure_preprocessing_threads(workflow, budget, threads)
        assert all(node.n_procs == expected for node in threaded)
        assert all(node.n_procs == 1 for node in serial)
        assert fit_pool.n_procs == 4
        hybrid = nested.get_node("hybrid_fast_chain")
        assert hybrid.n_procs == expected
        assert pickle.loads(pickle.dumps(hybrid))._pmriprep_threaded is True
        assert nested.get_node("tissue_segment").n_procs == 1
        assert cpu.os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == str(expected)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "auto"])
def test_neural_device_reservations_match_launcher_and_gpu_slots(
    device: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU extraction/segmentation leave CPU capacity free and stay serialized."""
    import nipype.pipeline.engine as pe

    from pmriprep.cli.run import _automatic_gpu_slots, _configure_preprocessing_threads
    from pmriprep.interfaces.anat import BrainExtraction
    from pmriprep.interfaces.dce import DCEBrainExtract
    from pmriprep.interfaces.dsc_brain import DSCBrainExtract
    from pmriprep.interfaces.hdglio import HDGLIOSegment
    from pmriprep.utils.hdbet import build_hdbet_command

    monkeypatch.setattr("pmriprep.utils.device.detect_torch_device", lambda: "cuda")
    monkeypatch.setattr("pmriprep.utils.gpu.detect_gpu_count", lambda: 2)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    workflow = pe.Workflow(name="hdbet_resources")
    nodes = [pe.Node(BrainExtraction(device=device), name="anat"),
             pe.Node(DCEBrainExtract(hdbet_device=device), name="dce"),
             pe.Node(DSCBrainExtract(hdbet_device=device), name="dsc")]
    resolved = "cuda" if device == "auto" else device
    nodes.append(pe.Node(HDGLIOSegment(use_gpu=resolved != "cpu"), name="tumour"))
    workflow.add_nodes(nodes)
    for limit in (8, 1):
        monkeypatch.setenv("OMP_NUM_THREADS", str(limit))
        _configure_preprocessing_threads(workflow, limit, limit)
        command = build_hdbet_command("in.nii.gz", "out.nii.gz", resolved, True)
        expected = min(limit, 2) if resolved != "cpu" else limit
        assert int(command[command.index("--threads") + 1]) == expected
        assert all(node.n_procs == expected for node in nodes)
        assert all(node.inputs.use_gpu == (resolved != "cpu") for node in nodes)
        assert "--disable_tta" not in command
        assert _automatic_gpu_slots(workflow) == (None if resolved == "cpu" else 1)
