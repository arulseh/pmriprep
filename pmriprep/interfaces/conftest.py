"""Doctest harness for pmriprep interface examples.

Nipype interface docstrings demonstrate usage by assigning ``File(exists=True)``
input traits to demo filenames (e.g. ``iface.inputs.in_file = 'T1w_brain.nii.gz'``).
Those assignments execute under ``--doctest-modules`` and would raise ``TraitError``
unless the file exists. Following the niworkflows pattern, this autouse fixture
copies the bundled ``tests/data`` placeholders into a temp dir and ``chdir``s into
it **for doctest items only**, so the existence check passes. The trailing
``result = iface.run()  # doctest: +SKIP`` line is still skipped (no real execution).
"""

import os
from contextlib import contextmanager
from pathlib import Path
from shutil import copytree

import pytest

try:
    from contextlib import chdir as _chdir  # type: ignore[attr-defined]
except ImportError:  # Python < 3.11

    @contextmanager
    def _chdir(path):
        cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(cwd)


@pytest.fixture(autouse=True)
def _docdir(request, tmp_path):
    """Run interface doctests inside a dir populated with placeholder inputs."""
    doctest_plugin = request.config.pluginmanager.getplugin("doctest")
    if isinstance(request.node, doctest_plugin.DoctestItem):
        copytree(Path(__file__).parent / "tests" / "data", tmp_path, dirs_exist_ok=True)
        with _chdir(tmp_path):
            yield
    else:
        yield
