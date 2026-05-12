"""Workflow directory naming helpers."""
from __future__ import annotations

from typing import Optional

__all__ = ["workflow_dirname"]


def workflow_dirname(subject_id: str, session_id: Optional[str] = None) -> str:
    """Return the Nipype workflow directory name for a subject/session pair."""
    if session_id:
        return f"pmriprep_sub_{subject_id}_ses_{session_id}_wf"
    return f"pmriprep_sub_{subject_id}_wf"
