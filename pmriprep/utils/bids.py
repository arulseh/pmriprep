"""BIDS adoption layer: validator wrapper + layout-backed enumeration.

Wraps :mod:`niworkflows.utils.bids` and the ``bids-validator`` JS CLI so that
``pmriprep`` matches the fmriprep/aslprep input-discovery idiom.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from bids.layout import BIDSLayout

logger = logging.getLogger("pmriprep.utils.bids")


def validate_bids_input(
    bids_dir: Union[str, Path],
    skip: bool = False,
) -> None:
    """Run the ``bids-validator`` JS CLI on ``bids_dir``.

    Parameters
    ----------
    bids_dir : str or Path
        Root of the BIDS dataset.
    skip : bool
        If True, log an info message and return without invoking the validator.

    Behavior
    --------
    - ``skip=True`` → no-op.
    - Validator binary missing on PATH → log warning and return (matches
      fmriprep). Permits running outside the container without Node.
    - Validator returns non-zero → log error and ``SystemExit(1)``.
    """
    if skip:
        logger.info("Skipping BIDS validation (--skip-bids-validation)")
        return

    binary = shutil.which("bids-validator")
    if binary is None:
        logger.warning(
            "bids-validator not found on PATH — skipping validation. "
            "Install via `npm install -g bids-validator` or use the official "
            "container. Pass --skip-bids-validation to silence this warning."
        )
        return

    cmd = [binary, str(bids_dir)]
    logger.info("Running BIDS validator on %s", bids_dir)
    result = subprocess.run(cmd, capture_output=False, check=False)
    if result.returncode != 0:
        logger.error(
            "bids-validator failed (exit %d). Fix the reported issues or "
            "rerun with --skip-bids-validation.",
            result.returncode,
        )
        sys.exit(1)


def collect_participants_pmriprep(
    bids_dir: Union[str, Path, "BIDSLayout"],
    participant_label: Optional[List[str]] = None,
    layout: Optional["BIDSLayout"] = None,
) -> List[str]:
    """Return participant IDs (sub- prefix stripped) discoverable in the layout.

    Wraps :func:`niworkflows.utils.bids.collect_participants` (which raises
    ``BIDSError`` on empty datasets or all-missing labels) and converts the
    error into an empty list, matching the legacy ``Path.glob`` contract.
    The CLI's downstream ``"No subjects found"`` log + ``sys.exit(1)`` then
    fires cleanly instead of an uncaught traceback.
    """
    from niworkflows.utils.bids import BIDSError, collect_participants

    src: Union[str, "BIDSLayout"]
    if layout is not None:
        src = layout
    elif isinstance(bids_dir, (str, Path)):
        src = str(bids_dir)
    else:
        src = bids_dir
    try:
        return list(
            collect_participants(
                src, participant_label=participant_label, bids_validate=False
            )
        )
    except BIDSError as exc:
        logger.warning("collect_participants returned no subjects: %s", exc)
        return []


def collect_sessions(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_label: Optional[List[str]] = None,
    layout: Optional["BIDSLayout"] = None,
) -> List[Optional[str]]:
    """Return session IDs (ses- prefix stripped) for ``subject_id``.

    Returns ``[None]`` for session-less datasets so caller loops still iterate
    once. Filters by ``session_label`` (sub- prefix tolerated) and warns on
    requested-but-missing sessions.
    """
    from bids.layout import BIDSLayout as _BIDSLayout

    if layout is None:
        layout = _BIDSLayout(str(bids_dir), validate=False)

    sessions = layout.get_sessions(subject=subject_id)
    if not sessions:
        return [None]

    if session_label is not None:
        wanted = [s.removeprefix("ses-") for s in session_label]
        present = [s for s in wanted if s in sessions]
        for missing in sorted(set(wanted) - set(present)):
            logger.warning(
                "Requested ses-%s not found for sub-%s", missing, subject_id
            )
        return list(present) if present else [None]

    return sorted(sessions)
