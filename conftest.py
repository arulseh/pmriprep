"""Pytest configuration for PMRIPrep doctests."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from shutil import copytree
from typing import Iterator

import pytest


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily change the process working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@pytest.fixture(autouse=True)
def _configure_doctest(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Provide common symbols and existing inputs to package doctests."""
    doctest_plugin = request.config.pluginmanager.getplugin("doctest")
    if doctest_plugin is None or not isinstance(
        request.node, doctest_plugin.DoctestItem
    ):
        yield
        return

    import nibabel as nib
    import numpy as np

    namespace = request.getfixturevalue("doctest_namespace")
    namespace.update(
        {
            "Path": Path,
            "nib": nib,
            "np": np,
            "os": os,
        }
    )

    tmp_path = tmp_path_factory.mktemp("interface_doctest")
    fixture_dir = Path(__file__).parent / "pmriprep" / "interfaces" / "tests" / "data"
    copytree(fixture_dir, tmp_path, dirs_exist_ok=True)
    with _working_directory(tmp_path):
        yield
