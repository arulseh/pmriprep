"""Graph characterization for the cross-modality PMRIPrep workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from pmriprep.config.anat import AnatConfig
from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.config.tumor_seg import TumorSegConfig
from pmriprep.workflows.base import (
    PipelineSkipFlags,
    _prepare_master_workflow_context,
    _select_master_structural_references,
    init_pmriprep_wf,
)
from pmriprep.workflows.discovery import (
    ModalityAvailability,
    PipelineAvailability,
    SessionData,
)

from .test_workflow_graph_regression import extract_public_outputnode_fields


BASELINE_DIR = Path(__file__).parent / "data" / "graph_baselines"
_REGEN = os.environ.get("PMRIPREP_REGEN_MASTER_GRAPH") == "1"


def _write_image(path: Path, *, n_volumes: int | None = None) -> None:
    """Write a tiny valid NIfTI used only during workflow construction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (3, 3, 3) if n_volumes is None else (3, 3, 3, n_volumes)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), path)


def _write_case(root: Path, modalities: set[str]) -> Path:
    """Create a minimal BIDS session containing the selected modalities."""
    bids = root / ("case_" + "_".join(sorted(modalities)))
    raw = bids / "sub-0001" / "ses-0001"
    bids.mkdir(parents=True, exist_ok=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )

    anatomical = {
        "t1w_pre": "sub-0001_ses-0001_T1w.nii.gz",
        "t1w_postcon": "sub-0001_ses-0001_ce-GAD_T1w.nii.gz",
        "t2w": "sub-0001_ses-0001_T2w.nii.gz",
        "flair": "sub-0001_ses-0001_FLAIR.nii.gz",
    }
    for key, filename in anatomical.items():
        if key in modalities:
            _write_image(raw / "anat" / filename)

    perfusion = {
        "dsc": ("sub-0001_ses-0001_dsc.nii.gz", 60),
        "dce": ("sub-0001_ses-0001_dce.nii.gz", 40),
    }
    sidecars = {
        "dsc": {
            "EchoTime": 0.03,
            "RepetitionTime": 1.5,
            "MagneticFieldStrength": 3.0,
        },
        "dce": {
            "EchoTime": 0.0014,
            "RepetitionTime": 4.0,
            "RepetitionTimeExcitation": 0.004,
            "FlipAngle": 12.0,
            "MagneticFieldStrength": 3.0,
            "MRAcquisitionType": "3D",
        },
    }
    for key, (filename, n_volumes) in perfusion.items():
        if key not in modalities:
            continue
        path = raw / "perf" / filename
        _write_image(path, n_volumes=n_volumes)
        path.with_name(path.name.replace(".nii.gz", ".json")).write_text(
            json.dumps(sidecars[key])
        )
    return bids


def _top_level_signature(workflow: Any) -> dict[str, Any]:
    """Capture master orchestration while standalone tests pin sub-workflow internals."""
    graph = workflow._graph
    nodes = sorted(
        (node.name, type(node).__qualname__)
        for node in graph.nodes()
    )
    edges: list[tuple[str, str, str, str]] = []
    for source, target, data in graph.edges(data=True):
        for source_field, target_field in data.get("connect", []) or []:
            edges.append(
                (source.name, str(source_field), target.name, str(target_field))
            )
    edges.sort()
    decision = workflow.pipeline_decision
    return {
        "nodes": nodes,
        "edges": edges,
        "decision": {
            "run_anat": decision.run_anat,
            "run_tumor_seg": decision.run_tumor_seg,
            "run_dsc": decision.run_dsc,
            "run_dce": decision.run_dce,
        },
        "public_outputnode_fields": extract_public_outputnode_fields(graph),
    }


def _build_master(
    root: Path,
    *,
    modalities: set[str],
    skip_anat: bool = False,
    tumor_seg: bool = False,
) -> Any:
    bids = _write_case(root, modalities)
    dce_config = DCEConfig(no_t1=True, b1_method="none")
    return init_pmriprep_wf(
        bids_dir=bids,
        output_dir=root / "out",
        subject_id="0001",
        session_id="0001",
        skip_flags=PipelineSkipFlags(
            skip_anat=skip_anat,
            skip_tumor_seg=not tumor_seg,
            skip_dsc="dsc" not in modalities,
            skip_dce="dce" not in modalities,
        ),
        anat_config=AnatConfig(),
        tumor_seg_config=TumorSegConfig(conda_env="hdglio-test"),
        dsc_config=DSCConfig(),
        dce_config=dce_config,
        work_dir=root / "work",
    )


MASTER_MATRIX = [
    ("master_anat_only", {"t1w_pre"}, False, False),
    ("master_dsc_only", {"t1w_pre", "dsc"}, False, False),
    ("master_dce_only", {"t1w_pre", "dce"}, False, False),
    ("master_combined", {"t1w_pre", "dsc", "dce"}, False, False),
    ("master_postcon_reference", {"t1w_postcon", "dsc", "dce"}, False, False),
    (
        "master_tumor_segmentation",
        {"t1w_pre", "t1w_postcon", "t2w", "flair"},
        False,
        True,
    ),
    ("master_skip_anat_precomputed", {"t1w_pre", "dsc", "dce"}, True, False),
]


@pytest.mark.parametrize(
    "name,modalities,skip_anat,tumor_seg",
    MASTER_MATRIX,
    ids=[row[0] for row in MASTER_MATRIX],
)
def test_master_graph_matches_baseline(
    name: str,
    modalities: set[str],
    skip_anat: bool,
    tumor_seg: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Master topology and public wiring remain stable across session modes."""
    monkeypatch.setattr("pmriprep.utils.device.detect_torch_device", lambda: "cpu")
    signature = _top_level_signature(
        _build_master(
            tmp_path,
            modalities=modalities,
            skip_anat=skip_anat,
            tumor_seg=tumor_seg,
        )
    )
    path = BASELINE_DIR / f"{name}.json"
    if _REGEN:
        path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"regenerated baseline: {name}")

    assert path.exists(), (
        f"missing baseline {path}; regenerate with PMRIPREP_REGEN_MASTER_GRAPH=1"
    )
    expected = json.loads(path.read_text())
    assert [tuple(node) for node in expected["nodes"]] == signature["nodes"]
    assert [tuple(edge) for edge in expected["edges"]] == signature["edges"]
    assert expected["decision"] == signature["decision"]
    assert expected["public_outputnode_fields"] == signature["public_outputnode_fields"]


def test_master_regen_flag_not_set_in_normal_run() -> None:
    """Prevent regeneration mode from silently replacing assertions."""
    assert not _REGEN, (
        "PMRIPREP_REGEN_MASTER_GRAPH=1 is set: baselines were regenerated, not verified"
    )


def test_prepare_master_context_resolves_session_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session preparation preserves defaults, memory buffers, and skip decisions."""
    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(
            t1w_ce_gad=tmp_path / "postcon.nii.gz",
            dsc=tmp_path / "dsc.nii.gz",
            dce=tmp_path / "dce.nii.gz",
        ),
        pipelines=PipelineAvailability(dsc=True, dce=True),
    )
    estimates: list[tuple[Path, int, float | None]] = []

    def _estimate(path: Path, *, n_buffers: int, mem_limit_gb: float | None):
        estimates.append((path, n_buffers, mem_limit_gb))
        return float(n_buffers), float(n_buffers + 1)

    monkeypatch.setattr(
        "pmriprep.workflows.base.discover_session_data", lambda *args, **kwargs: session_data
    )
    monkeypatch.setattr("pmriprep.utils.mem.estimate_4d_mem_gb", _estimate)
    monkeypatch.setattr("pmriprep.utils.mem._system_ram_gb", lambda: 128.0)

    context = _prepare_master_workflow_context(
        bids_dir=tmp_path / "bids" / ".." / "bids",
        output_dir=tmp_path / "out",
        subject_id="0001",
        session_id="0001",
        skip_flags=PipelineSkipFlags(skip_dsc=True),
        anat_config=None,
        tumor_seg_config=None,
        dsc_config=None,
        dce_config=None,
        work_dir=tmp_path / "work",
        mem_limit_gb=12.0,
        layout=None,
    )

    assert context.bids_dir == (tmp_path / "bids").resolve()
    assert context.work_dir == (tmp_path / "work").resolve()
    assert context.is_postcon_ref is True
    assert context.dsc_mem_gb == 4.0
    assert context.dce_mem_gb == 5.0
    assert estimates == [
        (tmp_path / "dsc.nii.gz", 4, 12.0),
        (tmp_path / "dce.nii.gz", 5, 12.0),
    ]
    assert context.decision.run_anat is True
    assert context.decision.run_dsc is False
    assert context.decision.run_dce is True
    assert isinstance(context.anat_config, AnatConfig)
    assert isinstance(context.dsc_config, DSCConfig)


def test_select_master_structural_references_routes_each_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference planning shares candidates and preserves modality-specific intent."""
    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(
            t1w_pre=tmp_path / "t1w.nii.gz",
            dsc=tmp_path / "dsc.nii.gz",
            dce=tmp_path / "dce.nii.gz",
        ),
        pipelines=PipelineAvailability(tumor_seg=True, dsc=True, dce=True),
    )
    candidates = [object()]
    build_calls: list[bool] = []
    select_calls: list[tuple[object, str, str | None]] = []

    monkeypatch.setattr(
        "pmriprep.utils.structural_reference.build_candidates_from_session",
        lambda modalities, *, has_lesion_mask: (
            build_calls.append(has_lesion_mask) or candidates
        ),
    )

    def _select(items: object, *, application: str, perfusion_path: str | None):
        select_calls.append((items, application, perfusion_path))
        return SimpleNamespace(
            candidate=SimpleNamespace(image_type=SimpleNamespace(value=application)),
            confidence=0.9,
            bbr_eligible=True,
        )

    monkeypatch.setattr(
        "pmriprep.utils.structural_reference.select_structural_reference_v2", _select
    )
    plan = _select_master_structural_references(
        session_data=session_data,
        decision=SimpleNamespace(run_dsc=True, run_dce=True),
        dsc_config=DSCConfig(),
        dce_config=DCEConfig(),
    )

    assert build_calls == [True]
    assert select_calls == [
        (candidates, "dsc_registration", str(tmp_path / "dsc.nii.gz")),
        (candidates, "dce_registration", str(tmp_path / "dce.nii.gz")),
    ]
    assert plan.dsc is not None
    assert plan.dce is not None
