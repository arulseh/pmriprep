"""Regression: BIDS space-entity compliance for derivatives.

Two rules enforced here:

1. **Native-acquisition-space outputs in ``perf/`` MUST NOT carry a
   ``space-`` entity.** Per BIDS-Common-Derivatives + ASLPrep precedent,
   the ``perf/`` directory implies native acquisition space; an explicit
   ``space-`` entity is therefore redundant and non-canonical.

2. **The pmriprep-invented ``space-anat`` / ``from-anat`` / ``to-anat``
   values are forbidden in source code.** ``anat`` is not a registered
   BIDS space identifier (the registered individual-reference value is
   ``T1w``). Per fmriprep/aslprep precedent and bids-spec-expert review,
   pmriprep uses ``space-T1w`` (and ``from-T1w`` / ``to-T1w``) uniformly
   for the chosen anat reference space; pre- vs post-contrast
   disambiguation lives in the JSON ``SpatialReference`` field. One
   known exception (the orphan-JSON ``space-anat_desc-lesionStats.json``
   at session root in ``qc_writers.py``) is allowlisted below — it
   shares the deferred BIDS file-organization issue with the analogous
   ``space-dsc``/``space-dce`` lesion stats files.

Cross-modality outputs in ``anat/`` (T1w/FLAIR/T2w resampled to DCE/DSC
space, masks describing anatomy in perf-space) DO carry
``space-dce``/``space-dsc`` — that is the canonical use of the entity
and is intentionally NOT covered by this test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pmriprep.utils.sidecar_builders import (
    anat_reference_basename,
    dce_reference_basename,
    dsc_reference_basename,
)


_SPACE_ENTITY = re.compile(r"_space-")
_PMRIPREP_ROOT = Path(__file__).resolve().parents[2] / "pmriprep"

# Files where the forbidden tokens may legitimately appear (deferred fixes,
# documentation that describes the historical anti-pattern). Each entry is
# (relative_path, allowed_substring, justification).
_ALLOWLIST: list[tuple[str, str, str]] = [
    (
        "interfaces/qc_writers.py",
        "space-anat_desc-lesionStats.json",
        "Orphan-JSON file-org issue at session root; deferred (parallel to"
        " space-dsc/dce orphan stats).",
    ),
]

_FORBIDDEN_TOKENS = ("space-anat", "from-anat", "to-anat")


# -----------------------------------------------------------------------------
# Rule 1: perf/ outputs lack space- entity
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basename",
    [dce_reference_basename(), dsc_reference_basename()],
)
def test_reference_basenames_lack_space_entity(basename: str) -> None:
    """Mean reference image basenames are native perf/ outputs."""
    assert _SPACE_ENTITY.search(basename) is None, (
        f"Reference basename {basename!r} carries space- entity. "
        f"Native perf/ outputs must omit space- per BIDS-fmriprep convention."
    )


def test_dce_copy_perf_map_lacks_space_entity(tmp_path) -> None:
    """End-to-end: _copy_perf_map writes a path without space-."""
    from pmriprep.workflows.dce.datasinks import _copy_perf_map

    src = tmp_path / "in.nii.gz"
    src.write_bytes(b"")  # empty file, just to satisfy shutil.copy2
    out_path = _copy_perf_map(
        in_file=str(src),
        output_dir=str(tmp_path / "out"),
        subject="01",
        session="01",
        desc="dce",
        suffix="Ktrans",
        meta_dict={"Description": "test"},
        datatype="perf",
        model_entity="model-tofts",
    )
    assert _SPACE_ENTITY.search(out_path) is None, (
        f"_copy_perf_map (DCE) wrote path with space-: {out_path}"
    )


def test_dsc_copy_perf_map_lacks_space_entity(tmp_path) -> None:
    """End-to-end: _copy_perf_map writes a path without space-."""
    from pmriprep.workflows.dsc.datasinks import _copy_perf_map

    src = tmp_path / "in.nii.gz"
    src.write_bytes(b"")
    out_path = _copy_perf_map(
        in_file=str(src),
        output_dir=str(tmp_path / "out"),
        subject="01",
        session="01",
        desc="dsc",
        suffix="mtt",
        meta_dict={"Description": "test"},
    )
    assert _SPACE_ENTITY.search(out_path) is None, (
        f"_copy_perf_map (DSC) wrote path with space-: {out_path}"
    )


# -----------------------------------------------------------------------------
# Rule 2: no `space-anat` / `from-anat` / `to-anat` in source (with allowlist)
# -----------------------------------------------------------------------------


def _scan_source_files() -> list[tuple[str, int, str, str]]:
    """Return list of (relpath, line_no, line_content, matched_token) for
    every line in pmriprep/ source that contains a forbidden token but is
    NOT covered by the allowlist."""
    findings = []
    for py_file in _PMRIPREP_ROOT.rglob("*.py"):
        relpath = py_file.relative_to(_PMRIPREP_ROOT).as_posix()
        text = py_file.read_text()
        for line_no, line in enumerate(text.splitlines(), start=1):
            for token in _FORBIDDEN_TOKENS:
                if token not in line:
                    continue
                # Check allowlist
                allowed = any(
                    relpath == allowed_path and allowed_substring in line
                    for allowed_path, allowed_substring, _ in _ALLOWLIST
                )
                if allowed:
                    continue
                findings.append((relpath, line_no, line.rstrip(), token))
    return findings


def test_no_pmriprep_invented_anat_space_tokens() -> None:
    """``space-anat``/``from-anat``/``to-anat`` are pmriprep inventions.

    Use ``space-T1w`` / ``from-T1w`` / ``to-T1w`` for the anat reference
    space (per fmriprep precedent + bids-spec-expert review). One known
    deferred exception is allowlisted (orphan stats JSON at session root).
    """
    findings = _scan_source_files()
    if findings:
        msg = "\nForbidden BIDS-non-canonical tokens found in pmriprep/ source:\n"
        for relpath, line_no, line, token in findings:
            msg += f"  pmriprep/{relpath}:{line_no} [{token}]: {line}\n"
        msg += (
            "\nUse `T1w` instead of `anat` for the BIDS space identifier. "
            "If a new legitimate exception arises (e.g., a deferred BIDS "
            "file-org cleanup), add it to _ALLOWLIST in this test file with "
            "a justification."
        )
        pytest.fail(msg)


def test_anat_reference_basename_uses_T1w_space() -> None:
    """``anat_reference_basename`` returns space-T1w form (not space-anat)."""
    precon = anat_reference_basename(is_postcon_ref=False)
    postcon = anat_reference_basename(is_postcon_ref=True)
    assert "space-T1w" in precon, (
        f"Precon ref basename should use space-T1w; got {precon!r}"
    )
    assert "space-T1w" in postcon, (
        f"Postcon ref basename should use space-T1w; got {postcon!r}"
    )
    assert "space-anat" not in precon
    assert "space-anat" not in postcon
