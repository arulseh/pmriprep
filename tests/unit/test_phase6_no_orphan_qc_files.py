"""Sweep regression: no legacy ``desc-*QC_quality.json`` orphans after PR2.

After v0.46.0 the four legacy session-root QC summary files
(``desc-anatQC``, ``desc-tumorsegQC``, ``desc-dscQC``, ``desc-dceQC``)
are gone. This test scans the source tree to confirm:

1. No producer code path writes those filenames.
2. No legacy lesion/ROI ``space-{anat,dsc,dce}_desc-{lesion,roi}Stats.json``
   orphans are written by producer code.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_PMRIPREP = Path(__file__).resolve().parents[2] / "pmriprep"

_FORBIDDEN_FILENAME_PATTERNS = (
    re.compile(r"desc-anatQC_quality\.json"),
    re.compile(r"desc-tumorsegQC_quality\.json"),
    re.compile(r"desc-dscQC_quality\.json"),
    re.compile(r"desc-dceQC_quality\.json"),
    re.compile(r"space-anat_desc-lesionStats\.json"),
    re.compile(r"space-dsc_desc-lesionStats\.json"),
    re.compile(r"space-dce_desc-lesionStats\.json"),
    re.compile(r"space-dsc_desc-roiStats\.json"),
    re.compile(r"space-dce_desc-roiStats\.json"),
    re.compile(r"desc-tumorseg_volumes\.txt"),
)


_LEGACY_MARKERS = (
    "legacy", "removed", "deleted", "no longer", "gone", "must not",
    "should not", "replaces the", "are gone",
)


@pytest.mark.parametrize("pattern", _FORBIDDEN_FILENAME_PATTERNS)
def test_no_legacy_orphan_filenames_in_producers(pattern: re.Pattern) -> None:
    """Producer code (``pmriprep/``) must not WRITE legacy orphan filenames.

    Docstring/comment references that explicitly call out the filename as
    legacy (within ±2 lines of a legacy-marker phrase) are allowed.
    """
    hits: list[str] = []
    for py in _PMRIPREP.rglob("*.py"):
        text = py.read_text(errors="ignore")
        if not pattern.search(text):
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not pattern.search(line):
                continue
            # Look at ±2 lines for a legacy-marker phrase
            window = "\n".join(lines[max(0, i - 2): i + 3]).lower()
            if any(marker in window for marker in _LEGACY_MARKERS):
                continue
            hits.append(
                f"{py.relative_to(_PMRIPREP.parent)}:{i + 1}: {line.strip()}"
            )

    assert not hits, (
        f"Legacy orphan filename `{pattern.pattern}` referenced in producer "
        f"code without a legacy-marker context:\n  " + "\n  ".join(hits)
    )
