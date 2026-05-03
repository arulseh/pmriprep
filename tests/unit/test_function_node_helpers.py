"""Regression: every helper used as ``niu.Function(function=_X)`` in a workflow
module must be reconstructible by nipype's ``create_function_from_source`` —
which exec()s the function source in an empty namespace.

Type annotations using ``Optional``/``Tuple``/``Any``/``Union`` etc. require
``from typing import …`` to be in scope at def-evaluation time. Since nipype
exec's the source with ``ns = {}``, any such annotation raises ``NameError``
at workflow runtime (not at import time, not at workflow-build time — only
when the node executes, which is far from the source of the bug).

This test catches the regression before runtime.
"""
from __future__ import annotations

import ast
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
        src = f.read_text()
        target_names = set(re.findall(r"function=(_\w+)", src))
        tree = ast.parse(src)
        for func in _walk_funcs(tree):
            if func.name not in target_names:
                continue
            seg = ast.get_source_segment(src, func)
            if seg is None:
                continue
            cases.append((relpath, func.name, textwrap.dedent(seg)))
    return cases


CASES = _collect_function_node_helpers()


@pytest.mark.parametrize(
    "relpath,name,source",
    CASES,
    ids=[f"{rel}::{name}" for rel, name, _ in CASES],
)
def test_function_node_helper_exec_in_empty_namespace(
    relpath: str, name: str, source: str,
) -> None:
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
