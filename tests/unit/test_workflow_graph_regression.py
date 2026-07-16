"""Canonical graph-regression harness for the perfusion preproc workflows.

This is the safety net for the ``init_dce_preproc_wf`` / ``init_dsc_preproc_wf``
sub-workflow decomposition. It builds each workflow across a matrix of
construction-gating configurations and compares a *graph signature* against
committed golden baselines.

The signature is computed over the **flattened-collapsed** graph
(``flatten_collapsed_graph``): nipype's ``_create_flat_graph`` expands nested
sub-workflows into leaf nodes, then the INTERNAL sub-workflow ``IdentityInterface``
passthroughs are spliced out with nipype's own ``_remove_identity_node`` while the
top-level inputnode/outputnode (the public contract) are kept. This makes the
signature **invariant under a correct sub-workflow decomposition** (a sub-wf's
inputnode/outputnode collapse away), while still catching a swapped/dropped/
mis-wired field across a sub-wf boundary. Internal-vs-top-level identities are
classified by ``_hierarchy`` (never ``.name`` -- nipype hashes nodes by object
identity, so many share ``.name == 'inputnode'``).

Seven tiers:

1. ``nodes``  -- sorted ``(node_name, interface_qualname)`` (bare name + bare
   ``__qualname__``; a module relocation of an interface class is invisible, a
   class swap differs).
2. ``edges``  -- sorted field-level ``(src_node, src_field, dst_node, dst_field)``
   tuples (every ``connect`` field-mapping flattened).
3. ``function_sha256`` -- ``{node_name: sha256(function_str)}`` per ``niu.Function``
   node (nipype serialises each body via ``getsource`` at construction; unchanged
   by relocation / decomposition).
4. ``function_io`` -- per Function node, declared ``input_names``/``output_names``
   (``output_names`` is not in ``function_str``; catches an unconsumed-output change).
5. ``node_params`` -- per-node statically-set, deterministic, non-path scalar input
   traits (e.g. ``compute_maps.oi_threshold``); path-valued traits dropped.
6. ``node_attrs`` -- per-node ``mem_gb``/``n_procs`` (Node *attributes*, NOT
   interface inputs, so ``node_params`` cannot see them; catches a dropped
   ``mem_gb=_heavy_mem`` when a node is moved into a sub-workflow).
7. ``public_outputnode_fields`` -- the top-level ``outputnode`` field list (pins
   the public contract even if a field becomes unconsumed).

Tiers 3-6 cover leaf nodes, which a decomposition does not change, so they stay
identical (independent proof no node behaviour changed). **Documented blind spot:**
swapping two outputnode fields that carry the *identical* upstream source is
invisible -- a semantic no-op; negative self-tests use *distinct* sources.

Workflow *construction* is data-free. Acquisition sidecar parameters (TR/TE/flip
angle) are *injected* post-construction by ``pmriprep.workflows.base``; only that
runtime injection is out of scope. The standalone build leaves these traits at
their interface defaults, which ARE recorded and pinned by ``node_params`` (so a
change to an interface default is caught), and are deterministic.

Regenerating baselines (the documented path for an *intentional* topology
change, or a deliberate nipype upgrade):

    PMRIPREP_REGEN_GRAPH=1 pytest tests/unit/test_workflow_graph_regression.py -q

then review the JSON diff under ``tests/unit/data/graph_baselines/`` and commit
it as part of the change. A normal run asserts; PR2/PR3 must keep every baseline
unchanged (``git diff --exit-code tests/unit/data/graph_baselines/``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import nipype
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import pytest
from nipype.interfaces.base import isdefined
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine.utils import _remove_identity_node

from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.workflows.dce.base import init_dce_preproc_wf
from pmriprep.workflows.dsc.base import init_dsc_preproc_wf

BASELINE_DIR = Path(__file__).parent / "data" / "graph_baselines"
_REGEN = os.environ.get("PMRIPREP_REGEN_GRAPH") == "1"

EdgeTuple = tuple[str, str, str, str]
NodeTuple = tuple[str, str]

# Sentinel marking a trait value that must be omitted from ``node_params``
# (path-like or non-deterministic / non-serialisable).
_OMIT = object()


# ---------------------------------------------------------------------------
# Graph signature extraction
# ---------------------------------------------------------------------------
def _iface(node: Any) -> Any:
    """Return a node's interface instance (handles either attribute name)."""
    return getattr(node, "interface", None) or getattr(node, "_interface", None)


def _canon_src(src: Any) -> str:
    """Normalise an edge source field to a stable string.

    ``src`` is normally a plain field name. If a transform callable was applied
    in ``.connect()`` nipype stores ``(field, getsource_text, *extra)``; collapse
    that to a deterministic string. (Not exercised in this repo today -- there
    are zero transform-callable edges -- but handled defensively.)
    """
    if isinstance(src, tuple):
        rest = "|".join(repr(x) for x in src[1:])
        return f"{src[0]}\x1e{rest}"
    return str(src)


def extract_edges(graph: Any) -> list[EdgeTuple]:
    """Canonical, sorted field-level edge list from a (flattened) nx graph."""
    edges: list[EdgeTuple] = []
    for u, v, data in graph.edges(data=True):
        for src, dst in data.get("connect", []) or []:
            edges.append((u.name, _canon_src(src), v.name, str(dst)))
    edges.sort()
    return edges


def extract_nodes(graph: Any) -> list[NodeTuple]:
    """Sorted ``(node_name, interface_qualname)`` pairs (bare name + qualname).

    After flatten-collapse, leaf nodes keep their bare ``.name`` (the sub-wf
    hierarchy prefix lives on ``_hierarchy``/``fullname``). All current node names
    are globally unique; the assertion fails loudly if a future sub-wf reuses a
    bare name, which would collapse two distinct leaves into one tuple -- a
    false-pass risk.
    """
    out: list[NodeTuple] = []
    seen: set[str] = set()
    for n in graph.nodes():
        ifc = _iface(n)
        cls = type(ifc) if ifc is not None else type(n)
        assert n.name not in seen, f"duplicate bare node name after flatten: {n.name!r}"
        seen.add(n.name)
        out.append((n.name, cls.__qualname__))
    out.sort()
    return out


def extract_function_sha256(graph: Any) -> dict[str, str]:
    """``{node_name: sha256(function_str)}`` for every Function node."""
    out: dict[str, str] = {}
    for n in graph.nodes():
        ifc = _iface(n)
        fstr = getattr(getattr(ifc, "inputs", None), "function_str", None)
        if isinstance(fstr, str) and fstr.strip():
            out[n.name] = hashlib.sha256(fstr.encode("utf-8")).hexdigest()
    return dict(sorted(out.items()))


def extract_function_io(graph: Any) -> dict[str, dict[str, list[str]]]:
    """Per Function node: declared ``input_names`` and ``output_names``.

    ``output_names`` is a ``niu.Function`` constructor argument that does not
    appear in ``function_str``; without this tier an added/renamed/dropped
    output (especially an unconsumed one, invisible to ``edges``) would pass
    undetected. Names are sorted -- nipype output ordering is semantically inert.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for n in graph.nodes():
        ifc = _iface(n)
        if ifc is None or type(ifc).__qualname__ != "Function":
            continue
        rec: dict[str, list[str]] = {}
        ins = getattr(ifc, "_input_names", None)
        outs = getattr(ifc, "_output_names", None)
        if ins is not None:
            rec["input_names"] = sorted(ins)
        if outs is not None:
            rec["output_names"] = sorted(outs)
        if rec:
            out[n.name] = rec
    return dict(sorted(out.items()))


def _clean(value: Any) -> Any:
    """Keep deterministic, non-path scalars; return ``_OMIT`` otherwise.

    Drops path-like strings (any ``/`` or ``\\``) so temp/install locations and
    File-trait values never enter a baseline, and drops non-JSON scalars
    (e.g. numpy types) to keep baselines portable and deterministic.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        if "/" in value or "\\" in value:
            return _OMIT
        return value
    if isinstance(value, (list, tuple)):
        # If any element is path-like / non-scalar the whole sequence is dropped.
        # No current trait mixes scalars with paths; revisit if one appears, as
        # this would mask a change to a surviving scalar element.
        cleaned = [_clean(v) for v in value]
        if any(c is _OMIT for c in cleaned):
            return _OMIT
        return cleaned
    return _OMIT


def extract_node_params(graph: Any) -> dict[str, dict[str, Any]]:
    """Per-node statically set, deterministic, non-path scalar input traits."""
    out: dict[str, dict[str, Any]] = {}
    for n in graph.nodes():
        ifc = _iface(n)
        inp = getattr(ifc, "inputs", None)
        if inp is None or not hasattr(inp, "copyable_trait_names"):
            continue
        params: dict[str, Any] = {}
        for name in inp.copyable_trait_names():
            try:
                value = getattr(inp, name)
            except Exception:
                continue
            if not isdefined(value):
                continue
            cleaned = _clean(value)
            if cleaned is _OMIT:
                continue
            params[name] = cleaned
        if params:
            out[n.name] = dict(sorted(params.items()))
    return dict(sorted(out.items()))


def extract_node_attrs(graph: Any) -> dict[str, dict[str, Any]]:
    """Per-node nipype Node-level resource attributes (``mem_gb``, ``n_procs``).

    These are Node *attributes*, NOT interface input traits, so ``node_params``
    never sees them. A dropped ``mem_gb=_heavy_mem`` while relocating a node into
    a sub-workflow would otherwise be invisible to every other tier.
    """
    out: dict[str, dict[str, Any]] = {}
    for n in graph.nodes():
        rec: dict[str, Any] = {}
        mem = getattr(n, "mem_gb", None)
        if isinstance(mem, (int, float)) and not isinstance(mem, bool):
            rec["mem_gb"] = mem
        nprocs = getattr(n, "n_procs", None)
        if isinstance(nprocs, int) and not isinstance(nprocs, bool):
            rec["n_procs"] = nprocs
        if rec:
            out[n.name] = dict(sorted(rec.items()))
    return dict(sorted(out.items()))


def extract_public_outputnode_fields(graph: Any) -> list[str]:
    """Sorted field list of the top-level ``outputnode`` IdentityInterface.

    Pins the public output contract: catches a renamed/dropped public field even
    if it becomes unconsumed (which the ``edges`` tier can miss). After collapse
    the only ``outputnode`` is the top-level one (sub-wf outputnodes are spliced).
    Does NOT close the same-source field-swap blind spot (that swaps which output
    gets a value, not the field set).
    """
    for n in graph.nodes():
        if n.name == "outputnode" and isinstance(_iface(n), IdentityInterface):
            inp = getattr(_iface(n), "inputs", None)
            if inp is not None and hasattr(inp, "copyable_trait_names"):
                return sorted(inp.copyable_trait_names())
    return []


def _is_internal_identity(node: Any, top_name: str) -> bool:
    """True for a SUB-workflow inputnode/outputnode (not the top-level pair).

    After ``_create_flat_graph`` every leaf keeps its bare ``.name`` but carries
    the enclosing hierarchy on ``_hierarchy``. The two top-level IdentityInterface
    nodes have ``_hierarchy == top_name``; sub-workflow identities are strictly
    deeper. Key on ``_hierarchy`` (never ``.name``) -- nipype hashes graph nodes
    by object identity, so multiple nodes share ``.name == 'inputnode'``.
    """
    if not isinstance(_iface(node), IdentityInterface):
        return False
    return (getattr(node, "_hierarchy", None) or "") != top_name


def flatten_collapsed_graph(wf: pe.Workflow) -> Any:
    """``wf``'s execution-flattened graph with INTERNAL sub-workflow identity
    passthroughs spliced out and the top-level inputnode/outputnode preserved.

    Computed over a deep copy (``_create_flat_graph`` deepcopies; ``wf`` is never
    mutated). Reuses nipype's own ``_remove_identity_node`` so field-mapping
    propagation across a sub-wf boundary is identical to the execution engine's.
    Splice order follows ``nx.topological_sort`` (chained identities collapse
    correctly). The resulting leaf-to-leaf edge set is invariant under a correct
    decomposition -- the load-bearing premise (verified empirically and pinned by
    ``test_flatten_collapse_matches_legacy_flat_on_monolith``).
    """
    g = wf._create_flat_graph()
    top = wf.name
    for node in list(nx.topological_sort(g)):
        if _is_internal_identity(node, top):
            _remove_identity_node(g, node)
    remaining = [n.name for n in g.nodes() if isinstance(n, pe.Workflow)]
    assert not remaining, f"flatten left nested workflow(s): {remaining}"
    return g


def graph_signature(wf: pe.Workflow) -> dict[str, Any]:
    """Seven-tier signature over the flattened-collapsed graph + a diagnostic
    ``nipype_version`` stamp (NOT compared -- nipype is loosely pinned;
    ``function_sha256`` catches any real serialisation change regardless).

    The graph is flattened (nested sub-workflows expanded to leaf nodes) and the
    internal sub-wf IdentityInterface passthroughs are spliced out, so the
    signature is invariant under a correct sub-workflow decomposition.
    ``function_sha256``/``function_io``/``node_params``/``node_attrs`` cover leaf
    nodes (unchanged by decomposition); ``public_outputnode_fields`` pins the
    public contract. Blind spot: swapping two outputnode fields carrying the
    *identical* source is invisible (a semantic no-op).
    """
    g = flatten_collapsed_graph(wf)
    return {
        "nipype_version": nipype.__version__,
        "nodes": extract_nodes(g),
        "edges": extract_edges(g),
        "function_sha256": extract_function_sha256(g),
        "function_io": extract_function_io(g),
        "node_params": extract_node_params(g),
        "node_attrs": extract_node_attrs(g),
        "public_outputnode_fields": extract_public_outputnode_fields(g),
    }


# ---------------------------------------------------------------------------
# Data-free workflow construction
# ---------------------------------------------------------------------------
def _write_bids(root: Path) -> Path:
    """Minimal BIDS tree sufficient for data-free workflow construction."""
    bids = root / "bids"
    ses = bids / "sub-0001" / "ses-0001"
    (ses / "anat").mkdir(parents=True)
    (ses / "perf").mkdir(parents=True)
    (ses / "anat" / "sub-0001_ses-0001_T1w.nii.gz").touch()
    (ses / "perf" / "sub-0001_ses-0001_dsc.nii.gz").touch()
    (ses / "perf" / "sub-0001_ses-0001_dce.nii.gz").touch()
    (bids / "dataset_description.json").write_text(
        '{"Name": "T", "BIDSVersion": "1.9.0"}'
    )
    return bids


def _ext_t1(bids: Path) -> Path:
    """A touched external-T1-map path to flip ``use_external_t1`` on."""
    p = bids / "external_t1map.nii.gz"
    p.touch()
    return p


def _dce(bids: Path, *, has_tissue_seg: bool = True, **cfg: Any) -> pe.Workflow:
    return init_dce_preproc_wf(
        bids_dir=bids,
        output_dir=bids / "derivatives",
        subject_id="0001",
        session_id="0001",
        config=DCEConfig(**cfg),
        has_tissue_seg=has_tissue_seg,
    )


def _dsc(bids: Path, **cfg: Any) -> pe.Workflow:
    return init_dsc_preproc_wf(
        bids_dir=bids,
        output_dir=bids / "derivatives",
        subject_id="0001",
        session_id="0001",
        config=DSCConfig(**cfg),
    )


_ALL_UNDERLAYS = ["t1w_precon", "t1w_postcon", "flair_precon", "flair_postcon", "t2w"]

# Each row exercises >= 1 verified construction-gating branch. Non-topological
# axes (DCE mask presence; DSC method/registration/leakage/aif/inference) are
# deliberately absent -- their static values are covered by the node_params tier.
# The arterial-atlas-cascade gates (dce/base.py:817, dsc/base.py:2737) are held
# constant by the 3-tier bundled atlas data, so the recorded baseline is the
# populated branch; the <2-tier partial-install branch is out of scope.
Builder = Callable[[Path], pe.Workflow]

DCE_MATRIX: list[tuple[str, Builder]] = [
    ("dce_default", lambda b: _dce(b)),
    # has_tissue_seg=False on the default VFA path -> isolates the 3 probseg
    # transform nodes (dce/base.py:715).
    ("dce_no_tissue_seg", lambda b: _dce(b, has_tissue_seg=False)),
    # has_tissue_seg=True + no_t1=True -> the :715 nodes exist while the
    # probseg->t1_mapping edges (:760-764) drop (use_vfa_t1 False). Together with
    # the row above this isolates the has_tissue_seg x use_vfa_t1 interaction.
    ("dce_no_t1", lambda b: _dce(b, no_t1=True)),
    ("dce_external_t1_b1auto", lambda b: _dce(b, t1_map_file=_ext_t1(b), b1_method="auto")),
    ("dce_external_t1_b1none", lambda b: _dce(b, t1_map_file=_ext_t1(b), b1_method="none")),
    ("dce_b1_none_vfa", lambda b: _dce(b, b1_method="none")),
    ("dce_smooth", lambda b: _dce(b, smooth_fwhm=4.0)),
    # The single-underlay rows are subsumed for detection by ``dce_underlays_all``
    # (underlay handling is independent per-key blocks, no single-underlay branch).
    ("dce_underlays_all", lambda b: _dce(b, underlays=list(_ALL_UNDERLAYS))),
    # model != "auto" drops the one model-driven edge (dce/base.py:1349).
    ("dce_model_fixed", lambda b: _dce(b, model="extended_tofts")),
    ("dce_kitchen_sink", lambda b: _dce(
        b, has_tissue_seg=False, t1_map_file=_ext_t1(b), b1_method="none",
        smooth_fwhm=4.0, underlays=list(_ALL_UNDERLAYS),
    )),
    # motion_correct=False: DCEPreprocess branches internally (no topology
    # change), so this row differs from dce_default only at the node_params
    # tier (the apply_motion_correction=False input on dce_preprocess + the
    # motion_applied inputs on motion_qc_plot / figures_writer).
    ("dce_no_moco", lambda b: _dce(b, motion_correct=False)),
]

DSC_MATRIX: list[tuple[str, Builder]] = [
    ("dsc_default", lambda b: _dsc(b)),
    # smooth_fwhm>0 is the only branch instantiating build_hemo_dict /
    # build_auc_dict -> load-bearing for verifying PR3's embedded-def pull.
    ("dsc_smooth", lambda b: _dsc(b, smooth_fwhm=4.0)),
    # Single-underlay rows and the smooth+underlays combo are subsumed for detection
    # by ``dsc_underlays_all`` plus ``dsc_smooth`` (underlay/smooth axes are orthogonal).
    ("dsc_underlays_all", lambda b: _dsc(b, underlays=list(_ALL_UNDERLAYS))),
    # motion_correct=False: adds the reorient_mean node and reroutes dsc_moco /
    # dsc_mean off dsc_reorient in dsc_motion_wf (a topology change confined to
    # the motion subgraph); the quantification subgraph downstream is unchanged.
    ("dsc_no_moco", lambda b: _dsc(b, motion_correct=False)),
]

MATRIX: list[tuple[str, Builder]] = DCE_MATRIX + DSC_MATRIX


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin device detection so the graph signature is hardware-independent.

    ``use_gpu`` is resolved at construction time (``resolve_device_uses_gpu`` ->
    ``detect_torch_device``), which yields ``"mps"`` on Apple Silicon and ``"cpu"``
    on a Linux CI runner. Forcing ``"cpu"`` keeps the captured signature identical
    across dev and CI so the committed baselines stay portable.
    """
    monkeypatch.setattr("pmriprep.utils.device.detect_torch_device", lambda: "cpu")


@pytest.fixture
def bids(tmp_path: Path) -> Path:
    return _write_bids(tmp_path)


@pytest.mark.parametrize("name,builder", MATRIX, ids=[m[0] for m in MATRIX])
def test_graph_matches_baseline(name: str, builder: Builder, bids: Path) -> None:
    """Each config's built graph must match its committed golden baseline."""
    sig = graph_signature(builder(bids))
    path = BASELINE_DIR / f"{name}.json"

    if _REGEN:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sig, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"regenerated baseline: {name}")

    assert path.exists(), (
        f"missing baseline {path}; regenerate with PMRIPREP_REGEN_GRAPH=1"
    )
    expected = json.loads(path.read_text())
    # nipype_version is diagnostic only (loose pin) -- compare the four tiers.
    assert [tuple(n) for n in expected["nodes"]] == sig["nodes"], f"{name}: node set changed"
    assert [tuple(e) for e in expected["edges"]] == sig["edges"], f"{name}: edge list changed"
    assert expected["function_sha256"] == sig["function_sha256"], (
        f"{name}: a Function-node body (function_str) changed"
    )
    assert expected["function_io"] == sig["function_io"], (
        f"{name}: a Function-node input/output name set changed"
    )
    assert expected["node_params"] == sig["node_params"], f"{name}: a node parameter changed"
    assert expected["node_attrs"] == sig["node_attrs"], (
        f"{name}: a node attribute (mem_gb/n_procs) changed"
    )
    assert expected["public_outputnode_fields"] == sig["public_outputnode_fields"], (
        f"{name}: public outputnode field contract changed"
    )


def test_regen_flag_not_set_in_normal_run() -> None:
    """Guard against a leaked ``PMRIPREP_REGEN_GRAPH=1`` silently disabling the gate.

    Under that env var every ``test_graph_matches_baseline`` case overwrites its
    baseline and skips, so a stray export (inherited shell, copy-paste) would
    rewrite the safety net to match current code and leave the run green. This
    always-on check turns that into a hard failure, so regeneration is only ever
    a deliberate, visibly-red action.
    """
    assert os.environ.get("PMRIPREP_REGEN_GRAPH") != "1", (
        "PMRIPREP_REGEN_GRAPH=1 is set: baselines were (re)generated, not "
        "verified. Unset it and re-run to assert against committed baselines."
    )


def test_dsc_method_is_parameter_not_topology(bids: Path) -> None:
    """Documents that DSC ``method`` is a node parameter, not a topology gate.

    The non-default method (cpi) must leave nodes/edges/function bodies identical
    to the default while differing only in the node_params tier at
    ``compute_maps.method``.
    """
    default = graph_signature(_dsc(bids))
    cpi = graph_signature(_dsc(bids, method="cpi"))
    assert cpi["nodes"] == default["nodes"]
    assert cpi["edges"] == default["edges"]
    assert cpi["function_sha256"] == default["function_sha256"]
    assert cpi["node_params"] != default["node_params"]
    assert cpi["node_params"]["compute_maps"]["method"] == "cpi"
    assert default["node_params"]["compute_maps"]["method"] != "cpi"


# ---------------------------------------------------------------------------
# Sensitivity tests -- prove the harness CATCHES real regressions by building
# REAL workflows (a correct one and a deliberately-broken one) and comparing the
# full graph_signature. (Mutating an extracted list would test Python, not the
# comparator -- so we never do that.) Toy leaves use niu.Merge (non-identity, so
# they survive the collapse); niu.Function is avoided (it rejects
# interactively-defined functions).
# ---------------------------------------------------------------------------
def test_sensitivity_function_str_change(bids: Path) -> None:
    """Perturbing a Function node's serialised source changes its hash."""
    g = flatten_collapsed_graph(_dsc(bids))
    before = extract_function_sha256(g)
    target = next(
        n for n in g.nodes()
        if isinstance(getattr(getattr(_iface(n), "inputs", None), "function_str", None), str)
        and getattr(_iface(n).inputs, "function_str").strip()
    )
    _iface(target).inputs.function_str = (
        _iface(target).inputs.function_str + "\n# perturbation\n"
    )
    after = extract_function_sha256(g)
    assert before != after
    assert before[target.name] != after[target.name]


def test_sensitivity_physics_param_drift(bids: Path) -> None:
    """Changing a physics kwarg (oSVD OI threshold) changes node_params."""
    g = flatten_collapsed_graph(_dsc(bids))
    before = extract_node_params(g)
    cm = next(n for n in g.nodes() if n.name == "compute_maps")
    assert before["compute_maps"]["oi_threshold"] == 0.095
    _iface(cm).inputs.oi_threshold = 0.1
    after = extract_node_params(g)
    assert before != after
    assert after["compute_maps"]["oi_threshold"] == 0.1


# -- toy workflows for the decomposition self-tests -------------------------
def _toy_monolith() -> pe.Workflow:
    """Flat: inputnode.src -> {leaf_a.in1, leaf_b.in1}; leaf_a.out->out.p, leaf_b.out->out.q."""
    wf = pe.Workflow(name="toy_wf")
    inp = pe.Node(niu.IdentityInterface(fields=["src"]), name="inputnode")
    out = pe.Node(niu.IdentityInterface(fields=["p", "q"]), name="outputnode")
    a = pe.Node(niu.Merge(2), name="leaf_a")
    b = pe.Node(niu.Merge(2), name="leaf_b")
    wf.connect([
        (inp, a, [("src", "in1")]), (inp, b, [("src", "in1")]),
        (a, out, [("out", "p")]), (b, out, [("out", "q")]),
    ])
    return wf


def _toy_one_subwf(*, swap: bool = False, drop: bool = False, miswire: bool = False) -> pe.Workflow:
    """Same leaves wrapped in one sub-wf; flags inject a specific regression."""
    sub = pe.Workflow(name="sub_wf")
    si = pe.Node(niu.IdentityInterface(fields=["src"]), name="inputnode")
    fields = ["p"] if drop else ["p", "q"]
    so = pe.Node(niu.IdentityInterface(fields=fields), name="outputnode")
    a = pe.Node(niu.Merge(2), name="leaf_a")
    b = pe.Node(niu.Merge(2), name="leaf_b")
    a_dst = "in2" if miswire else "in1"      # mis-wired inputnode->leaf field
    conn = [(si, a, [("src", a_dst)]), (si, b, [("src", "in1")])]
    if drop:
        conn += [(a, so, [("out", "p")])]
    elif swap:                               # distinct-source crossed outputs
        conn += [(a, so, [("out", "q")]), (b, so, [("out", "p")])]
    else:
        conn += [(a, so, [("out", "p")]), (b, so, [("out", "q")])]
    sub.connect(conn)
    wf = pe.Workflow(name="toy_wf")
    inp = pe.Node(niu.IdentityInterface(fields=["src"]), name="inputnode")
    out = pe.Node(niu.IdentityInterface(fields=fields), name="outputnode")
    wf.connect([(inp, sub, [("src", "inputnode.src")])]
               + [(sub, out, [(f"outputnode.{f}", f)]) for f in fields])
    return wf


def _toy_chain_monolith() -> pe.Workflow:
    """Flat chain: inputnode.src -> leaf_a -> leaf_b -> outputnode.dst."""
    wf = pe.Workflow(name="toy_wf")
    inp = pe.Node(niu.IdentityInterface(fields=["src"]), name="inputnode")
    out = pe.Node(niu.IdentityInterface(fields=["dst"]), name="outputnode")
    a = pe.Node(niu.Merge(2), name="leaf_a")
    b = pe.Node(niu.Merge(2), name="leaf_b")
    wf.connect([(inp, a, [("src", "in1")]), (a, b, [("out", "in1")]),
                (b, out, [("out", "dst")])])
    return wf


def _toy_chain_two_subwf(*, broken: bool = False) -> pe.Workflow:
    """Two sub-wfs chained; ``broken`` omits the sub_a->sub_b handoff edge."""
    sub_a = pe.Workflow(name="sub_a_wf")
    ai = pe.Node(niu.IdentityInterface(fields=["src"]), name="inputnode")
    ao = pe.Node(niu.IdentityInterface(fields=["mid"]), name="outputnode")
    la = pe.Node(niu.Merge(2), name="leaf_a")
    sub_a.connect([(ai, la, [("src", "in1")]), (la, ao, [("out", "mid")])])
    sub_b = pe.Workflow(name="sub_b_wf")
    bi = pe.Node(niu.IdentityInterface(fields=["mid"]), name="inputnode")
    bo = pe.Node(niu.IdentityInterface(fields=["dst"]), name="outputnode")
    lb = pe.Node(niu.Merge(2), name="leaf_b")
    sub_b.connect([(bi, lb, [("mid", "in1")]), (lb, bo, [("out", "dst")])])
    wf = pe.Workflow(name="toy_wf")
    inp = pe.Node(niu.IdentityInterface(fields=["src"]), name="inputnode")
    out = pe.Node(niu.IdentityInterface(fields=["dst"]), name="outputnode")
    conn = [(inp, sub_a, [("src", "inputnode.src")]),
            (sub_b, out, [("outputnode.dst", "dst")])]
    if not broken:
        conn.append((sub_a, sub_b, [("outputnode.mid", "inputnode.mid")]))
    wf.connect(conn)
    return wf


def test_decomposition_invisible_one_subwf() -> None:
    """A correct 1-sub-wf decomposition collapses to the monolith's signature."""
    assert graph_signature(_toy_monolith()) == graph_signature(_toy_one_subwf())


def test_decomposition_invisible_two_subwf() -> None:
    """A correct 2-sub-wf decomposition (mirrors the PR1->PR2 handoff) is invisible."""
    assert graph_signature(_toy_chain_monolith()) == graph_signature(_toy_chain_two_subwf())


def test_wrong_decomposition_dropped_output_caught() -> None:
    assert graph_signature(_toy_monolith()) != graph_signature(_toy_one_subwf(drop=True))


def test_wrong_decomposition_swapped_output_caught() -> None:
    """Crossing two outputnode fields with DISTINCT sources changes the edges.

    (A *same-source* swap is a semantic no-op and intentionally invisible.)
    """
    assert graph_signature(_toy_monolith()) != graph_signature(_toy_one_subwf(swap=True))


def test_wrong_decomposition_miswired_input_caught() -> None:
    assert graph_signature(_toy_monolith()) != graph_signature(_toy_one_subwf(miswire=True))


def test_wrong_decomposition_broken_handoff_caught() -> None:
    """A 2-sub-wf decomposition missing the sub_a->sub_b handoff is caught."""
    assert graph_signature(_toy_chain_monolith()) != graph_signature(_toy_chain_two_subwf(broken=True))


# NOTE: ``test_flatten_collapse_matches_legacy_flat_on_monolith`` was retired once
# DCE began decomposing (no production monolith remains). The flatten-collapse ==
# legacy-flat invariant is now covered by the toy-monolith self-tests above and by
# ``test_graph_matches_baseline`` (collapsed graph == committed baseline).


def test_determinism_path_independent(tmp_path: Path) -> None:
    """Two builds in different absolute locations yield identical signatures.

    Proves the signature is reproducible run-to-run and that no temp/install
    path leaks into any tier (the regression gate would flake otherwise).
    """
    b1 = _write_bids(tmp_path / "a")
    b2 = _write_bids(tmp_path / "b")
    # Same path, built twice -> identical (intra-process determinism; robust to
    # PYTHONHASHSEED since _clean omits all unordered collections).
    assert graph_signature(_dce(b1)) == graph_signature(_dce(b1))
    # Different absolute paths -> identical (no temp/install path leaks in).
    assert graph_signature(_dce(b1)) == graph_signature(_dce(b2))
    assert graph_signature(_dsc(b1)) == graph_signature(_dsc(b2))
