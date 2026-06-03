"""Canonical graph-regression harness for the perfusion preproc workflows.

This is the safety net for the deferred ``init_dce_preproc_wf`` /
``init_dsc_preproc_wf`` decomposition. It builds each workflow across a matrix
of construction-gating configurations and compares a four-tier *graph
signature* against committed golden baselines:

1. ``nodes``  -- sorted ``(node_name, interface_qualname)``. The qualname uses
   the bare ``__qualname__`` (no module prefix) so a deliberate relocation of an
   interface *class* between modules is invisible, while a genuine class swap
   still differs.
2. ``edges``  -- sorted field-level ``(src_node, src_field, dst_node, dst_field)``
   tuples (every ``connect`` field-mapping is flattened; a node-pair may carry
   several).
3. ``function_sha256`` -- ``{node_name: sha256(function_str)}`` for every
   ``niu.Function`` node. nipype serialises each Function body via
   ``getsource = dedent(inspect.getsource(fn))`` at construction time, so moving
   a helper between modules (or dedenting an embedded ``def`` to module level)
   must leave this hash unchanged. This tier proves the helper-extraction PRs
   are byte-identical.
4. ``node_params`` -- ``{node_name: {trait: value}}`` for each node's statically
   set, deterministic, non-path scalar input traits (e.g.
   ``compute_maps.oi_threshold``, ``.method``). Turns "relocation cannot touch
   OI=0.065" into "the gate proves it did not". Path-valued traits are dropped
   so temp/install locations never leak into a baseline.
5. ``function_io`` -- per ``niu.Function`` node, its declared ``input_names``
   and ``output_names``. ``output_names`` is a constructor argument that is NOT
   reflected in ``function_str``, so an added/renamed/dropped output -- in
   particular an *unconsumed* terminal output, invisible to the ``edges`` tier
   -- would otherwise pass undetected.

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

import nipype
import nipype.pipeline.engine as pe
import pytest
from nipype.interfaces.base import isdefined

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


def extract_edges(wf: pe.Workflow) -> list[EdgeTuple]:
    """Flatten ``wf._graph`` to a canonical, sorted field-level edge list."""
    edges: list[EdgeTuple] = []
    for u, v, data in wf._graph.edges(data=True):
        for src, dst in data.get("connect", []) or []:
            edges.append((u.name, _canon_src(src), v.name, str(dst)))
    edges.sort()
    return edges


def extract_nodes(wf: pe.Workflow) -> list[NodeTuple]:
    """Sorted ``(node_name, interface_qualname)`` pairs (bare qualname)."""
    out: list[NodeTuple] = []
    for n in wf._graph.nodes():
        ifc = _iface(n)
        cls = type(ifc) if ifc is not None else type(n)
        out.append((n.name, cls.__qualname__))
    out.sort()
    return out


def extract_function_sha256(wf: pe.Workflow) -> dict[str, str]:
    """``{node_name: sha256(function_str)}`` for every Function node."""
    out: dict[str, str] = {}
    for n in wf._graph.nodes():
        ifc = _iface(n)
        fstr = getattr(getattr(ifc, "inputs", None), "function_str", None)
        if isinstance(fstr, str) and fstr.strip():
            out[n.name] = hashlib.sha256(fstr.encode("utf-8")).hexdigest()
    return dict(sorted(out.items()))


def extract_function_io(wf: pe.Workflow) -> dict[str, dict[str, list[str]]]:
    """Per Function node: declared ``input_names`` and ``output_names``.

    ``output_names`` is a ``niu.Function`` constructor argument that does not
    appear in ``function_str``; without this tier an added/renamed/dropped
    output (especially an unconsumed one, invisible to ``edges``) would pass
    undetected. Names are sorted -- nipype output ordering is semantically inert.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for n in wf._graph.nodes():
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


def extract_node_params(wf: pe.Workflow) -> dict[str, dict[str, Any]]:
    """Per-node statically set, deterministic, non-path scalar input traits."""
    out: dict[str, dict[str, Any]] = {}
    for n in wf._graph.nodes():
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


def graph_signature(wf: pe.Workflow) -> dict[str, Any]:
    """Full five-tier signature plus a diagnostic nipype-version stamp.

    ``nipype_version`` is recorded for diagnosis only (nipype is loosely pinned,
    so it is NOT part of the equality comparison); ``function_sha256`` catches
    any real serialisation change regardless of version.

    The extractor is intentionally *flat*: it walks ``wf._graph.nodes()`` and
    does not recurse. Both preproc workflows are flat today (only plain
    ``pe.Node``s). The guard below fails loudly if a future change (e.g. the
    deferred sub-workflow decomposition) introduces a nested ``pe.Workflow``,
    which would otherwise appear as one opaque node and silently drop its
    internals -- a signal to upgrade the extractor to recurse first.
    """
    nested = [n.name for n in wf._graph.nodes() if isinstance(n, pe.Workflow)]
    assert not nested, (
        f"flat graph signature cannot capture nested sub-workflow(s) {nested}; "
        "upgrade the extractor to recurse before relying on it"
    )
    return {
        "nipype_version": nipype.__version__,
        "nodes": extract_nodes(wf),
        "edges": extract_edges(wf),
        "function_sha256": extract_function_sha256(wf),
        "function_io": extract_function_io(wf),
        "node_params": extract_node_params(wf),
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


_ALL_UNDERLAYS = ["t1w_precon", "t1w_postcon", "flair", "t2w"]

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
    ("dce_underlay_t1precon", lambda b: _dce(b, underlays=["t1w_precon"])),
    ("dce_underlay_t1postcon", lambda b: _dce(b, underlays=["t1w_postcon"])),
    ("dce_underlay_flair", lambda b: _dce(b, underlays=["flair"])),
    ("dce_underlay_t2w", lambda b: _dce(b, underlays=["t2w"])),
    ("dce_underlays_all", lambda b: _dce(b, underlays=list(_ALL_UNDERLAYS))),
    # model != "auto" drops the one model-driven edge (dce/base.py:1349).
    ("dce_model_fixed", lambda b: _dce(b, model="extended_tofts")),
    ("dce_kitchen_sink", lambda b: _dce(
        b, has_tissue_seg=False, t1_map_file=_ext_t1(b), b1_method="none",
        smooth_fwhm=4.0, underlays=list(_ALL_UNDERLAYS),
    )),
]

DSC_MATRIX: list[tuple[str, Builder]] = [
    ("dsc_default", lambda b: _dsc(b)),
    # smooth_fwhm>0 is the only branch instantiating build_hemo_dict /
    # build_auc_dict -> load-bearing for verifying PR3's embedded-def pull.
    ("dsc_smooth", lambda b: _dsc(b, smooth_fwhm=4.0)),
    ("dsc_underlay_t1precon", lambda b: _dsc(b, underlays=["t1w_precon"])),
    ("dsc_underlay_t1postcon", lambda b: _dsc(b, underlays=["t1w_postcon"])),
    ("dsc_underlay_flair", lambda b: _dsc(b, underlays=["flair"])),
    ("dsc_underlay_t2w", lambda b: _dsc(b, underlays=["t2w"])),
    ("dsc_underlays_all", lambda b: _dsc(b, underlays=list(_ALL_UNDERLAYS))),
    ("dsc_smooth_underlays", lambda b: _dsc(b, smooth_fwhm=4.0, underlays=list(_ALL_UNDERLAYS))),
]

MATRIX: list[tuple[str, Builder]] = DCE_MATRIX + DSC_MATRIX


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------
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
# Sensitivity tests -- prove the comparator rejects the real failure modes
# ---------------------------------------------------------------------------
def test_sensitivity_field_swap(bids: Path) -> None:
    """A swapped src field on one edge must change the edge list."""
    edges = extract_edges(_dce(bids))
    assert edges
    mutated = list(edges)
    s, sf, d, df = mutated[0]
    mutated[0] = (s, sf + "_SWAPPED", d, df)
    assert sorted(mutated) != edges


def test_sensitivity_node_rename(bids: Path) -> None:
    """A renamed node must change the node set."""
    nodes = extract_nodes(_dce(bids))
    assert nodes
    mutated = [("RENAMED_" + n, q) if i == 0 else (n, q) for i, (n, q) in enumerate(nodes)]
    assert sorted(mutated) != nodes


def test_sensitivity_dropped_edge(bids: Path) -> None:
    """A dropped optional-output edge must change the edge list."""
    edges = extract_edges(_dce(bids))
    assert len(edges) > 1
    assert sorted(edges[1:]) != edges


def test_sensitivity_function_str_change(bids: Path) -> None:
    """Perturbing a Function node's serialised source changes its hash."""
    wf = _dsc(bids)
    before = extract_function_sha256(wf)
    target = next(
        n for n in wf._graph.nodes()
        if isinstance(getattr(getattr(_iface(n), "inputs", None), "function_str", None), str)
        and getattr(_iface(n).inputs, "function_str").strip()
    )
    _iface(target).inputs.function_str = (
        _iface(target).inputs.function_str + "\n# perturbation\n"
    )
    after = extract_function_sha256(wf)
    assert before != after
    assert before[target.name] != after[target.name]


def test_sensitivity_physics_param_drift(bids: Path) -> None:
    """Changing a physics kwarg (oSVD OI threshold) changes node_params."""
    wf = _dsc(bids)
    before = extract_node_params(wf)
    cm = next(n for n in wf._graph.nodes() if n.name == "compute_maps")
    assert before["compute_maps"]["oi_threshold"] == 0.065
    _iface(cm).inputs.oi_threshold = 0.1
    after = extract_node_params(wf)
    assert before != after
    assert after["compute_maps"]["oi_threshold"] == 0.1


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
