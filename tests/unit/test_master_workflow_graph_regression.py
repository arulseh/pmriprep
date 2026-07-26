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
    PipelineDecision,
    PipelineSkipFlags,
    _connect_master_dsc_anatomy,
    _connect_master_dsc_masks,
    _connect_master_dsc_outputs_and_build_provenance,
    _connect_master_dsc_to_dce,
    _connect_master_dce_t1_b1_inputs,
    _init_master_workflow_nodes,
    _prepare_master_workflow_context,
    _prepare_master_preflight_state,
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


def test_connect_master_dsc_anatomy_preserves_live_underlay_routes() -> None:
    """Live anatomy exposes each requested DSC structural input independently."""
    connections: list[Any] = []
    workflow = SimpleNamespace(connect=connections.extend)
    anat_wf = SimpleNamespace()
    dsc_wf = SimpleNamespace()
    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(
            t1w_pre=Path("pre.nii.gz"),
            t1w_ce_gad=Path("post.nii.gz"),
            t2w=Path("t2w.nii.gz"),
            flair=Path("flair.nii.gz"),
            flair_postcon=Path("flair_post.nii.gz"),
        ),
        pipelines=PipelineAvailability(),
    )

    _connect_master_dsc_anatomy(
        workflow=workflow,
        dsc_wf=dsc_wf,
        anat_wf=anat_wf,
        session_data=session_data,
        anat_config=AnatConfig(create_subtraction=True),
        dsc_config=DSCConfig(underlays=[
            "t1w_precon", "t1w_postcon", "t1w_subtraction",
            "flair_precon", "flair_postcon", "t2w",
        ]),
        work_dir=None,
        subject_id="0001",
        session_id="0001",
    )

    fields = {
        pair
        for source, target, pairs in connections
        if source is anat_wf and target is dsc_wf
        for pair in pairs
    }
    assert fields == {
        ("outputnode.t1w_preproc", "inputnode.t1w_reference"),
        ("outputnode.brain_mask", "inputnode.brain_mask"),
        ("outputnode.t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
        ("outputnode.wm_probseg", "inputnode.wm_probseg"),
        ("outputnode.t1w_preproc", "inputnode.t1w_underlay"),
        ("outputnode.t1w_postcon_coreg", "inputnode.t1w_postcon_underlay"),
        ("outputnode.t1w_subtraction", "inputnode.t1w_subtraction_underlay"),
        ("outputnode.flair_coreg", "inputnode.flair_in_t1w"),
        ("outputnode.flair_postcon_coreg", "inputnode.flair_postcon_in_t1w"),
        ("outputnode.t2w_coreg", "inputnode.t2w_in_t1w"),
    }


def test_connect_master_dsc_anatomy_loads_cached_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip-anat mode assigns cached references, masks, transforms, and underlays."""
    previous = {
        "t1w_preproc": tmp_path / "pre.nii.gz",
        "t1w_postcon_coreg": tmp_path / "post.nii.gz",
        "t1w_subtraction": tmp_path / "sub.nii.gz",
        "brain_mask": tmp_path / "brain.nii.gz",
        "wm_probseg": tmp_path / "wm.nii.gz",
        "t1w_to_mni_xfm": tmp_path / "xfm.h5",
        "flair_coreg": tmp_path / "flair.nii.gz",
        "flair_postcon_coreg": tmp_path / "flair_post.nii.gz",
        "t2w_coreg": tmp_path / "t2w.nii.gz",
    }
    monkeypatch.setattr(
        "pmriprep.workflows.base._find_previous_anat_outputs",
        lambda *args: previous,
    )
    inputnode = SimpleNamespace()
    dsc_wf = SimpleNamespace(inputs=SimpleNamespace(inputnode=inputnode))
    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(t1w_pre=tmp_path / "source.nii.gz"),
        pipelines=PipelineAvailability(),
    )

    _connect_master_dsc_anatomy(
        workflow=SimpleNamespace(connect=lambda connections: None),
        dsc_wf=dsc_wf,
        anat_wf=None,
        session_data=session_data,
        anat_config=AnatConfig(),
        dsc_config=DSCConfig(underlays=[
            "t1w_precon", "t1w_postcon", "t1w_subtraction",
            "flair_precon", "flair_postcon", "t2w",
        ]),
        work_dir=tmp_path / "work",
        subject_id="0001",
        session_id="0001",
    )

    assert vars(inputnode) == {
        "t1w_reference": previous["t1w_preproc"],
        "t1w_underlay": previous["t1w_preproc"],
        "t1w_postcon_underlay": previous["t1w_postcon_coreg"],
        "t1w_subtraction_underlay": previous["t1w_subtraction"],
        "brain_mask": previous["brain_mask"],
        "wm_probseg": previous["wm_probseg"],
        "t1w_to_mni_xfm": previous["t1w_to_mni_xfm"],
        "flair_in_t1w": previous["flair_coreg"],
        "flair_postcon_in_t1w": previous["flair_postcon_coreg"],
        "t2w_in_t1w": previous["t2w_coreg"],
    }


def test_connect_master_dsc_masks_loads_cached_tumor_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached tumor masks retain their DSC input roles and availability state."""
    previous = {
        "nawm_mask": tmp_path / "nawm.nii.gz",
        "whole_tumor_mask": tmp_path / "whole.nii.gz",
        "enhancing_mask": tmp_path / "enhancing.nii.gz",
        "segmentation": tmp_path / "seg.nii.gz",
    }
    monkeypatch.setattr(
        "pmriprep.workflows.base._find_previous_tumor_seg_outputs",
        lambda *args: previous,
    )
    inputnode = SimpleNamespace()
    state = _connect_master_dsc_masks(
        workflow=SimpleNamespace(connect=lambda connections: None),
        dsc_wf=SimpleNamespace(inputs=SimpleNamespace(inputnode=inputnode)),
        tumor_seg_wf=None,
        decision=PipelineDecision(run_tumor_seg=False),
        dsc_config=DSCConfig(),
        work_dir=tmp_path / "work",
        subject_id="0001",
        session_id="0001",
    )

    assert vars(inputnode) == {
        "nawm_mask": previous["nawm_mask"],
        "whole_tumor_mask": previous["whole_tumor_mask"],
        "enhancing_mask": previous["enhancing_mask"],
        "tumor_segmentation": previous["segmentation"],
    }
    assert state.has_nawm_source is True
    assert state.has_whole_tumor_source is True


def test_connect_master_dsc_outputs_preserves_provenance_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public DSC fields and user/default provenance sources remain stable."""
    from pmriprep.utils.contrast_agent import ParameterSource

    captured: dict[str, Any] = {}

    def _build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"source": "characterized"}

    monkeypatch.setattr(
        "pmriprep.utils.contrast_agent.build_dsc_provenance", _build
    )
    connections: list[Any] = []
    workflow = SimpleNamespace(connect=connections.extend)
    dsc_wf = SimpleNamespace()
    outputnode = SimpleNamespace()
    config = DSCConfig(dose=0.5, agent_standard_dose=0.1)

    provenance = _connect_master_dsc_outputs_and_build_provenance(
        workflow=workflow,
        dsc_wf=dsc_wf,
        outputnode=outputnode,
        dsc_config=config,
        dsc_sidecar={"contrast_agent": "gadobutrol", "bids_agent_string": "Gadavist"},
        agent_source=ParameterSource.BIDS_SIDECAR,
        agent_dose_source=ParameterSource.DEFAULT_FALLBACK,
        gd_multiplicity=1,
        user_specified={"dsc_dose": True, "dsc_preload": False},
    )

    assert provenance == {"source": "characterized"}
    assert connections == [
        (dsc_wf, outputnode, [
            ("outputnode.rel_cbf_map", "dsc_cbf"),
            ("outputnode.rel_cbv_auc_map", "dsc_cbv"),
            ("outputnode.mtt_map", "dsc_mtt"),
            ("outputnode.tmax_map", "dsc_tmax"),
            ("outputnode.aif_qc", "dsc_aif_qc"),
            ("outputnode.qc_metrics", "dsc_qc"),
        ])
    ]
    assert captured["dose_source"] == ParameterSource.USER_SPECIFIED
    assert captured["preload_source"] == ParameterSource.AUTO_DETECTED
    assert captured["agent_source"] == ParameterSource.BIDS_SIDECAR
    assert captured["agent_standard_dose_source"] == ParameterSource.DEFAULT_FALLBACK
    assert captured["gd_multiplicity"] == 1


def test_connect_master_dsc_to_dce_preserves_live_constraint_routes() -> None:
    """Live DSC supplies CBV, normalized CBV, transforms, and QC to DCE."""
    connections: list[Any] = []
    dsc_wf = SimpleNamespace()
    dce_wf = SimpleNamespace(
        inputs=SimpleNamespace(
            inputnode=SimpleNamespace(),
            dce_dsc_validation_wf=SimpleNamespace(inputnode=SimpleNamespace()),
        )
    )
    _connect_master_dsc_to_dce(
        workflow=SimpleNamespace(connect=connections.extend),
        dsc_wf=dsc_wf,
        dce_wf=dce_wf,
        dsc_config=DSCConfig(dose=0.5, agent_standard_dose=0.1, preload=0.0),
        dce_config=DCEConfig(model="extended_tofts"),
        output_dir=Path("out"),
        work_dir=Path("work"),
        subject_id="0001",
        session_id="0001",
    )

    assert connections == [
        (dsc_wf, dce_wf, [
            ("outputnode.rel_cbv_auc_map", "inputnode.dsc_cbv_file"),
            ("outputnode.ncbv_map", "inputnode.dsc_ncbv_file"),
            ("outputnode.dsc_to_t1w_xfm", "inputnode.dsc_to_t1w_xfm"),
            ("outputnode.aif_qc", "dce_pk_modeling_wf.pk_modeling.dsc_aif_qc"),
            ("outputnode.preload_qc", "dce_pk_modeling_wf.pk_modeling.dsc_preload_qc"),
            (
                "outputnode.preprocessing_stats",
                "dce_pk_modeling_wf.pk_modeling.dsc_preprocessing_stats",
            ),
        ])
    ]
    assert dce_wf.inputs.inputnode.dsc_dose == 0.5
    assert dce_wf.inputs.inputnode.dsc_agent_standard_dose == 0.1
    assert (
        dce_wf.inputs.dce_dsc_validation_wf.inputnode.dsc_preload_configured
        is True
    )


def test_connect_master_dsc_to_dce_ignores_dose_without_dsc_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSC-only dose settings do not alter a DCE-only workflow."""
    dce_input = SimpleNamespace(
        dsc_dose=1.0,
        dsc_agent_standard_dose=0.1,
    )
    dce_wf = SimpleNamespace(
        inputs=SimpleNamespace(
            inputnode=dce_input,
            dce_dsc_validation_wf=SimpleNamespace(inputnode=SimpleNamespace()),
        )
    )
    monkeypatch.setattr(
        "pmriprep.workflows.base._find_previous_dsc_outputs",
        lambda *args, **kwargs: {
            "rel_cbv_map": None,
            "ncbv_map": None,
            "dsc_to_t1w_xfm": None,
            "aif_qc": None,
            "preload_qc": None,
            "preprocessing_stats": None,
        },
    )

    _connect_master_dsc_to_dce(
        workflow=SimpleNamespace(connect=lambda _: None),
        dsc_wf=None,
        dce_wf=dce_wf,
        dsc_config=DSCConfig(dose=0.5, agent_standard_dose=0.05),
        dce_config=DCEConfig(model="extended_tofts"),
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        subject_id="0001",
        session_id="0001",
    )

    assert dce_input.dsc_dose == 1.0
    assert dce_input.dsc_agent_standard_dose == 0.1


def test_connect_master_dsc_to_dce_routes_dose_with_cached_dsc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reusable DSC maps carry their configured dose metadata into DCE."""
    dce_input = SimpleNamespace(
        dsc_dose=1.0,
        dsc_agent_standard_dose=0.1,
    )
    dce_wf = SimpleNamespace(
        inputs=SimpleNamespace(
            inputnode=dce_input,
            dce_dsc_validation_wf=SimpleNamespace(inputnode=SimpleNamespace()),
        )
    )
    monkeypatch.setattr(
        "pmriprep.workflows.base._find_previous_dsc_outputs",
        lambda *args, **kwargs: {
            "rel_cbv_map": "cached_cbv.nii.gz",
            "ncbv_map": None,
            "dsc_to_t1w_xfm": "cached_xfm.mat",
            "aif_qc": None,
            "preload_qc": None,
            "preprocessing_stats": None,
        },
    )

    _connect_master_dsc_to_dce(
        workflow=SimpleNamespace(connect=lambda _: None),
        dsc_wf=None,
        dce_wf=dce_wf,
        dsc_config=DSCConfig(dose=0.5, agent_standard_dose=0.05),
        dce_config=DCEConfig(model="extended_tofts"),
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        subject_id="0001",
        session_id="0001",
    )

    assert dce_input.dsc_dose == 0.5
    assert dce_input.dsc_agent_standard_dose == 0.05
    assert dce_input.dsc_cbv_file == "cached_cbv.nii.gz"


def test_connect_master_dce_t1_b1_inputs_preserves_vfa_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VFA flip angles/TR and precomputed B1 encoding retain their precedence."""
    vfa = [tmp_path / "vfa1.nii.gz", tmp_path / "vfa2.nii.gz"]
    sidecars = {
        vfa[0]: {"flip_angle": 5.0, "pulse_tr": 0.004},
        vfa[1]: {"flip_angle": 15.0, "pulse_tr": 0.004},
    }
    monkeypatch.setattr(
        "pmriprep.workflows.base._read_bids_sidecar", lambda path: sidecars[path]
    )
    connections: list[Any] = []
    inputnode = SimpleNamespace()
    dce_input = SimpleNamespace()
    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(
            vfa=vfa,
            b1_precomputed_file=tmp_path / "b1.nii.gz",
            b1_nominal_prep_fa=60.0,
            b1_map_encoding="percent",
        ),
        pipelines=PipelineAvailability(),
    )
    _connect_master_dce_t1_b1_inputs(
        workflow=SimpleNamespace(connect=connections.extend),
        inputnode=inputnode,
        dce_wf=SimpleNamespace(inputs=SimpleNamespace(inputnode=dce_input)),
        session_data=session_data,
        dce_config=DCEConfig(b1_method="precomputed", b1_map_encoding="auto"),
    )

    assert connections[0][2] == [("vfa_files", "inputnode.vfa_files")]
    assert dce_input.vfa_flip_angles == [5.0, 15.0]
    assert dce_input.vfa_tr == 4.0
    assert dce_input.precomputed_b1_file == str(tmp_path / "b1.nii.gz")
    assert dce_input.b1_nominal_prep_fa == 60.0
    assert dce_input.b1_map_encoding == "percent"


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


def test_prepare_master_context_isolates_session_config_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved T1/B1 state from one session cannot alter the next session."""
    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(),
        pipelines=PipelineAvailability(),
    )
    monkeypatch.setattr(
        "pmriprep.workflows.base.discover_session_data", lambda *args, **kwargs: session_data
    )
    monkeypatch.setattr("pmriprep.utils.mem._system_ram_gb", lambda: 128.0)
    source = DCEConfig(no_t1=False, b1_method="auto")

    first = _prepare_master_workflow_context(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        subject_id="0001",
        session_id="0001",
        skip_flags=None,
        anat_config=None,
        tumor_seg_config=None,
        dsc_config=None,
        dce_config=source,
        work_dir=tmp_path / "work",
        mem_limit_gb=None,
        layout=None,
    )
    first.dce_config.no_t1 = True
    first.dce_config.b1_method = "none"
    second = _prepare_master_workflow_context(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        subject_id="0001",
        session_id="0002",
        skip_flags=None,
        anat_config=None,
        tumor_seg_config=None,
        dsc_config=None,
        dce_config=source,
        work_dir=tmp_path / "work",
        mem_limit_gb=None,
        layout=None,
    )

    assert source.no_t1 is False
    assert source.b1_method == "auto"
    assert second.dce_config.no_t1 is False
    assert second.dce_config.b1_method == "auto"
    assert first.dce_config is not second.dce_config


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


def test_select_master_structural_references_ignores_inactive_dsc_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DSC-only lesion mask does not alter DCE-only reference selection."""
    lesion_mask = tmp_path / "dsc_lesion.nii.gz"
    lesion_mask.touch()
    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(
            t1w_pre=tmp_path / "t1w.nii.gz",
            dce=tmp_path / "dce.nii.gz",
        ),
        pipelines=PipelineAvailability(tumor_seg=False, dsc=False, dce=True),
    )
    build_calls: list[bool] = []
    monkeypatch.setattr(
        "pmriprep.utils.structural_reference.build_candidates_from_session",
        lambda modalities, *, has_lesion_mask: (
            build_calls.append(has_lesion_mask) or []
        ),
    )

    _select_master_structural_references(
        session_data=session_data,
        decision=SimpleNamespace(run_dsc=False, run_dce=True),
        dsc_config=DSCConfig(lesion_mask=lesion_mask),
        dce_config=DCEConfig(),
    )

    assert build_calls == [False]

    build_calls.clear()
    _select_master_structural_references(
        session_data=SessionData(
            subject="0001",
            session="0001",
            modalities=ModalityAvailability(
                t1w_pre=tmp_path / "t1w.nii.gz",
                dsc=tmp_path / "dsc.nii.gz",
            ),
            pipelines=PipelineAvailability(tumor_seg=False, dsc=True, dce=False),
        ),
        decision=SimpleNamespace(run_dsc=True, run_dce=False),
        dsc_config=DSCConfig(),
        dce_config=DCEConfig(lesion_mask=lesion_mask),
    )

    assert build_calls == [False]


def test_prepare_master_preflight_state_preserves_metadata_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight state retains DSC provenance and ordered modality warnings."""
    dsc_path = tmp_path / "dsc.nii.gz"
    dce_path = tmp_path / "dce.nii.gz"
    vfa_paths = [tmp_path / "vfa1.nii.gz", tmp_path / "vfa2.nii.gz"]
    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(
            dsc=dsc_path,
            dce=dce_path,
            vfa=vfa_paths,
        ),
        pipelines=PipelineAvailability(dsc=True, dce=True),
    )
    sidecars = {
        dsc_path: {"contrast_agent": "gadoquatrane", "agent_standard_dose": 0.04},
        dce_path: {"flip_angle": 12.0, "pulse_tr": 0.004},
        vfa_paths[0]: {"flip_angle": 2.0, "pulse_tr": 0.004},
        vfa_paths[1]: {"flip_angle": None, "pulse_tr": 0.005},
    }
    captured: dict[str, Any] = {}

    class _Report:
        def log_all(self) -> None:
            captured["logged"] = True

        def raise_if_errors(self) -> None:
            captured["raised"] = True

        def warning_strings(self, prefix: str = "") -> list[str]:
            return [prefix + "warning"]

    monkeypatch.setattr(
        "pmriprep.workflows.base._read_bids_sidecar", lambda path: sidecars[path]
    )
    monkeypatch.setattr(
        "pmriprep.utils.contrast_agent.lookup_gd_multiplicity", lambda agent: 4
    )

    def _preflight(**kwargs: Any) -> _Report:
        captured.update(kwargs)
        return _Report()

    monkeypatch.setattr("pmriprep.utils.preflight.run_preflight", _preflight)
    dsc_config = DSCConfig()
    state = _prepare_master_preflight_state(
        session_data=session_data,
        decision=PipelineDecision(run_dsc=True, run_dce=True),
        dsc_config=dsc_config,
        dce_config=DCEConfig(),
        user_specified=None,
    )

    assert dsc_config.agent_standard_dose == 0.04
    assert state.gd_multiplicity == 4
    assert state.dsc_agent_dose_source == "bids_sidecar"
    assert state.dsc_agent_source == "bids_sidecar"
    assert captured["vfa_flip_angles"] == [2.0]
    assert captured["vfa_trs"] == [0.004, 0.005]
    assert captured["logged"] is True
    assert captured["raised"] is True
    assert state.dsc_warnings == ["DSC_warning"]
    assert state.dce_warnings == ["DCE_warning", "VFA_warning", "AFI_warning"]


def test_init_master_workflow_nodes_sets_public_contract(tmp_path: Path) -> None:
    """Master node factory initializes discovered inputs and report defaults."""
    import nipype.pipeline.engine as pe

    session_data = SessionData(
        subject="0001",
        session="0001",
        modalities=ModalityAvailability(
            t1w_pre=tmp_path / "t1w.nii.gz",
            dsc=tmp_path / "dsc.nii.gz",
            vfa=[tmp_path / "vfa1.nii.gz", tmp_path / "vfa2.nii.gz"],
        ),
        pipelines=PipelineAvailability(dsc=True),
    )
    workflow = pe.Workflow(name="master")
    nodes = _init_master_workflow_nodes(
        workflow=workflow,
        session_data=session_data,
        decision=PipelineDecision(run_anat=True, run_dsc=True),
        subject_id="0001",
        session_id="0001",
        output_dir=tmp_path / "out",
        command_string="pmriprep ...",
    )

    assert nodes.inputnode.inputs.t1w_pre == str(tmp_path / "t1w.nii.gz")
    assert nodes.inputnode.inputs.vfa_files == [
        str(tmp_path / "vfa1.nii.gz"),
        str(tmp_path / "vfa2.nii.gz"),
    ]
    assert nodes.report_barrier.inputs.anat_done == ""
    assert nodes.report_barrier.inputs.dce_done == ""
    assert nodes.unified_report.inputs.ran_dsc is True
    assert nodes.unified_report.inputs.ran_dce is False
    assert "dce_ktrans" in nodes.outputnode.inputs.copyable_trait_names()
