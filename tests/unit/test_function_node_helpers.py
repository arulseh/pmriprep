"""Regression: every helper used as ``niu.Function(function=_X)`` in a workflow
module must be reconstructible by nipype's ``create_function_from_source`` —
which exec()s the function source in an empty namespace.

Three guards:

1. **NameError-on-def** — Type annotations using ``Optional``/``Tuple``/``Any``/
   ``Union`` etc. require ``from typing import …`` to be in scope at
   def-evaluation time. Since nipype exec's the source with ``ns = {}``, any
   such annotation raises ``NameError`` at workflow runtime.

2. **PEP-604 / PEP-585 forward-compat** — ``str | None`` and ``list[...]``
   work on Python 3.10+ but break on Python 3.9 (``requires-python`` floor).
   Function-node helpers should keep signatures plain so we don't accidentally
   regress the supported-Python window.

3. **Closure-leak detector** — A helper that references a module-level
   constant or function inside its body will pass guard 1 (def loads fine)
   but raise ``NameError`` when actually called. Catches the broader bug
   class beyond annotations.
"""
from __future__ import annotations

import ast
import builtins
import re
import textwrap
from pathlib import Path

import pytest

WORKFLOW_FILES = [
    "pmriprep/workflows/anat/base.py",
    "pmriprep/workflows/anat/datasinks.py",
    "pmriprep/workflows/base.py",
    "pmriprep/workflows/dce/base.py",
    "pmriprep/workflows/dce/datasinks.py",
    "pmriprep/workflows/dsc/base.py",
    "pmriprep/workflows/dsc/datasinks.py",
    "pmriprep/workflows/tumor_seg/base.py",
    "pmriprep/workflows/tumor_seg/datasinks.py",
]

PEP585_GENERICS = {"list", "dict", "tuple", "set", "frozenset", "type"}
BUILTIN_NAMES = set(dir(builtins))


def _walk_funcs(node):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child
            yield from _walk_funcs(child)
        else:
            yield from _walk_funcs(child)


def _collect_function_node_helpers():
    cases = []
    repo_root = Path(__file__).resolve().parents[2]
    for relpath in WORKFLOW_FILES:
        f = repo_root / relpath
        if not f.exists():
            continue
        src = f.read_text()
        target_names = set(re.findall(r"function=(_\w+)", src))
        tree = ast.parse(src)
        module_names = _module_level_names(tree)
        for func in _walk_funcs(tree):
            if func.name not in target_names:
                continue
            seg = ast.get_source_segment(src, func)
            if seg is None:
                continue
            cases.append(
                (relpath, func.name, textwrap.dedent(seg), module_names),
            )
    return cases


def _module_level_names(tree: ast.Module) -> set[str]:
    """Names defined at the top level of the module (imports, assigns, defs)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


CASES = _collect_function_node_helpers()


@pytest.mark.parametrize(
    "relpath,name,source,_module_names",
    CASES,
    ids=[f"{rel}::{name}" for rel, name, _, _ in CASES],
)
def test_function_node_helper_exec_in_empty_namespace(
    relpath, name, source, _module_names,
):
    """Helper source must def-load in empty namespace (mimics nipype Function node)."""
    ns: dict = {}
    try:
        exec(source, ns)
    except NameError as e:
        pytest.fail(
            f"{relpath}::{name} fails nipype-style exec: {e}\n"
            f"Function-node helpers cannot use typing-imports annotations "
            f"(Optional/Tuple/Any/Union/etc.) because nipype reconstructs "
            f"them via exec in an empty namespace. Strip the annotations or "
            f"import the typing names inside the function body.",
        )


def _has_pep604_or_pep585(node):
    """Return (kind, snippet) if annotation uses PEP-604 ``X | None`` or PEP-585 ``list[...]``."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
            return ("PEP-604 union", ast.unparse(sub))
        if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
            if sub.value.id in PEP585_GENERICS:
                return ("PEP-585 generic", ast.unparse(sub))
    return None


@pytest.mark.parametrize(
    "relpath,name,source,_module_names",
    CASES,
    ids=[f"{rel}::{name}" for rel, name, _, _ in CASES],
)
def test_function_node_helper_no_pep604_pep585(
    relpath, name, source, _module_names,
):
    """Helper signatures and bodies must avoid ``str | None`` / ``list[...]``.

    These work on Python 3.10+ but break on Python 3.9, and the project
    declares ``requires-python = ">=3.10"``. Keep Function-node helpers
    annotation-free so the supported-Python floor is decoupled from helper
    source. Catches forward-compat regressions before they ship.
    """
    tree = ast.parse(source)
    func = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == name),
        None,
    )
    assert func is not None

    annotations = []
    for arg in (*func.args.args, *func.args.kwonlyargs, *func.args.posonlyargs):
        if arg.annotation is not None:
            annotations.append(arg.annotation)
    if func.returns is not None:
        annotations.append(func.returns)
    for sub in ast.walk(func):
        if isinstance(sub, ast.AnnAssign) and sub.annotation is not None:
            annotations.append(sub.annotation)
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not func:
            for arg in (*sub.args.args, *sub.args.kwonlyargs, *sub.args.posonlyargs):
                if arg.annotation is not None:
                    annotations.append(arg.annotation)
            if sub.returns is not None:
                annotations.append(sub.returns)

    for ann in annotations:
        hit = _has_pep604_or_pep585(ann)
        if hit is not None:
            kind, snippet = hit
            pytest.fail(
                f"{relpath}::{name} uses {kind} `{snippet}` in an annotation. "
                f"Function-node helpers must keep signatures plain to stay "
                f"compatible with the declared Python floor and to avoid "
                f"coupling helper source to interpreter version. Strip the "
                f"annotation or move it to a docstring.",
            )


def _collect_local_names(func):
    """Names made local to the function: args, assignments, imports, nested defs."""
    locals_: set[str] = set()
    for arg in (*func.args.args, *func.args.kwonlyargs, *func.args.posonlyargs):
        locals_.add(arg.arg)
    if func.args.vararg is not None:
        locals_.add(func.args.vararg.arg)
    if func.args.kwarg is not None:
        locals_.add(func.args.kwarg.arg)
    for sub in ast.walk(func):
        if isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                for nm in ast.walk(tgt):
                    if isinstance(nm, ast.Name):
                        locals_.add(nm.id)
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            locals_.add(sub.target.id)
        elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
            locals_.add(sub.target.id)
        elif isinstance(sub, ast.For):
            for nm in ast.walk(sub.target):
                if isinstance(nm, ast.Name):
                    locals_.add(nm.id)
        elif isinstance(sub, ast.With):
            for item in sub.items:
                if item.optional_vars is not None:
                    for nm in ast.walk(item.optional_vars):
                        if isinstance(nm, ast.Name):
                            locals_.add(nm.id)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            locals_.add(sub.name)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                if alias.name == "*":
                    continue
                locals_.add(
                    alias.asname or (
                        alias.name.split(".")[0]
                        if isinstance(sub, ast.Import)
                        else alias.name
                    ),
                )
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and sub is not func:
            locals_.add(sub.name)
        elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in sub.generators:
                for nm in ast.walk(gen.target):
                    if isinstance(nm, ast.Name):
                        locals_.add(nm.id)
        elif isinstance(sub, ast.Lambda):
            for arg in (*sub.args.args, *sub.args.kwonlyargs, *sub.args.posonlyargs):
                locals_.add(arg.arg)
        elif isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
            locals_.add(sub.target.id)
    return locals_


@pytest.mark.parametrize(
    "relpath,name,source,module_names",
    CASES,
    ids=[f"{rel}::{name}" for rel, name, _, _ in CASES],
)
def test_function_node_helper_no_closure_leak(
    relpath, name, source, module_names,
):
    """Helper body must not reference module-level names without inline import.

    nipype reconstructs the function via ``inspect.getsource`` + exec in an
    empty namespace, so module-level constants, imports, and helpers do NOT
    survive. A reference like ``FOO`` to a module-level ``FOO = ...`` will
    pass guard 1 (def loads fine) but raise ``NameError`` when the function
    is actually called by the workflow.
    """
    tree = ast.parse(source)
    func = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == name),
        None,
    )
    assert func is not None

    locals_ = _collect_local_names(func)

    # Collect all Name(Load) references in the body (excluding annotations,
    # which guard 2 handles).
    referenced: set[str] = set()
    for sub in ast.walk(func):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            referenced.add(sub.id)

    leaks = sorted(
        n for n in referenced
        if n in module_names
        and n not in locals_
        and n not in BUILTIN_NAMES
    )

    if leaks:
        pytest.fail(
            f"{relpath}::{name} references module-level names {leaks} "
            f"without importing them inside the function body. nipype "
            f"reconstructs Function helpers in an empty namespace, so these "
            f"references will NameError at runtime. Either inline-import or "
            f"pass the value as a default argument.",
        )
