"""Pytest configuration for pmriprep doctests.

Injects a common namespace into ``--doctest-modules`` examples so they can use
``np``/``nib``/``Path`` without an explicit import line, matching the NiPreps
(niworkflows) convention. Only loaded when collecting under ``pmriprep/`` (i.e.
``pytest --doctest-modules pmriprep/``); the default ``pytest tests/`` run does
not load this file and is unaffected.
"""

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _add_namespace(doctest_namespace):
    """Make common symbols available to every doctest in the package."""
    import nibabel as nib

    doctest_namespace["np"] = np
    doctest_namespace["nib"] = nib
    doctest_namespace["os"] = os
    doctest_namespace["Path"] = Path
