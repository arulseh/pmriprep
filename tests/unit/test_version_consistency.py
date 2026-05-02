"""Verify version strings stay in sync across declaration sites.

CLAUDE.md flags version bumps as the #1-most-violated rule. This test
catches drift between ``pyproject.toml`` and the Dockerfile OCI label
so a missed bump fails CI rather than shipping.

``pmriprep.__version__`` is intentionally NOT checked here because
``importlib.metadata`` reads the install-time version, which lags
behind ``pyproject.toml`` in editable installs (``pip install -e .``).
The CI install step exercises the package-version path implicitly when
it imports the freshly installed wheel.
"""
from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return str(data["project"]["version"])


def _dockerfile_version() -> str:
    text = (REPO_ROOT / "Dockerfile").read_text()
    match = re.search(
        r'LABEL\s+org\.opencontainers\.image\.version\s*=\s*"([^"]+)"',
        text,
    )
    assert match, "Could not find OCI version label in Dockerfile"
    return match.group(1)


def test_pyproject_matches_dockerfile() -> None:
    assert _pyproject_version() == _dockerfile_version(), (
        f"pyproject.toml version={_pyproject_version()!r} but "
        f"Dockerfile OCI label={_dockerfile_version()!r}"
    )
