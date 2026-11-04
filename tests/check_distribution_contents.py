"""Validate the contents of PMRIPrep release archives."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


REQUIRED_PACKAGE_FILES = {
    "pmriprep/models/dce/identification.py",
    "pmriprep/models/dce/cached_input.py",
    "pmriprep/models/dce/native_exchange.py",
    "pmriprep/models/dce/_native_backend.py",
    "pmriprep/models/dce/LICENSE_SCIPY.txt",
    "pmriprep/models/dce/NOTICE_SCIPY.md",
    *{f"pmriprep/models/dce/_trf_{part}.c" for part in
      ("module", "solver", "evaluator", "trust", "recurrence")},
    "pmriprep/models/dsc/signal_cpi/maps.py",
    "pmriprep/models/dsc/signal_cpi/execution.py",
    "pmriprep/models/dsc/signal_cpi/progress.py",
    "pmriprep/models/dsc/signal_cpi/spatial_readout.py",
    "pmriprep/models/dsc/signal_cpi/count.py",
    "pmriprep/models/dsc/signal_cpi/residue.py",
    "pmriprep/models/dsc/signal_cpi/precision.py",
    "pmriprep/models/dsc/signal_cpi/native.py",
    "pmriprep/models/dsc/local_vi.py",
    "pmriprep/models/dsc/vi_adapter.py",
    "pmriprep/models/dsc/batched_fitting.py",
    "pmriprep/models/dsc/spatial_vb.py",
    "pmriprep/utils/bsw_guidance_io.py",
    "pmriprep/templates/report_unified.html",
    "pmriprep/data/atlases/README.md",
    "pmriprep/data/atlases/LICENSE-CC0-1.0.txt",
    "pmriprep/data/atlases/mni_aif/MNI_2mm_MRA_arterial_full_mask.nii.gz",
    "pmriprep/data/atlases/mni_regions/Supra.nii.gz",
}
FORBIDDEN_PARTS = {"bak", "debug", "sourcedata", "test_outputs", "tests", "work"}


def _validate_members(path: Path, members: set[str]) -> None:
    """Validate common package files and excluded local paths."""
    normalized = {member.removeprefix("./") for member in members}
    package_members = {
        member[member.index("pmriprep/"):]
        for member in normalized
        if "pmriprep/" in member
    }
    missing = REQUIRED_PACKAGE_FILES - package_members
    assert not missing, f"{path.name} is missing required files: {sorted(missing)}"
    leaked = sorted(
        member for member in normalized
        if FORBIDDEN_PARTS.intersection(Path(member).parts)
    )
    assert not leaked, f"{path.name} contains local/test paths: {leaked}"


def _validate_license_labels() -> None:
    """Keep the package and both image publication labels consistent."""
    import re
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    for filename in ("Dockerfile", ".github/workflows/docker-publish.yml"):
        match = re.search(r'org\.opencontainers\.image\.licenses="?([^"\n]+)',
                          (root / filename).read_text())
        assert match and match.group(1).strip() == project["license"], filename


def _validate_wheel(path: Path) -> None:
    """Validate a wheel archive."""
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        _validate_members(path, members)
        entry_points = [name for name in members if name.endswith(".dist-info/entry_points.txt")]
        assert len(entry_points) == 1, f"{path.name} has invalid entry-point metadata"
        text = archive.read(entry_points[0]).decode()
        assert "pmriprep = pmriprep.cli.run:main" in text
        native = [name for name in members if
                  name.startswith("pmriprep/models/dsc/signal_cpi/_convolution.")
                  and name.endswith((".so", ".pyd"))]
        assert len(native) == 1, f"{path.name} must contain exactly one compiled CPI kernel"
        native_dce = [name for name in members if
                      name.startswith("pmriprep/models/dce/_native_trf.")
                      and name.endswith((".so", ".pyd"))]
        assert len(native_dce) == 1, f"{path.name} must contain exactly one compiled DCE kernel"
        package_metadata = [name for name in members if name.endswith(".dist-info/METADATA")]
        assert len(package_metadata) == 1
        assert any(line.startswith("License-Expression:") and "BSD-3-Clause" in line
                   for line in archive.read(package_metadata[0]).decode().splitlines())
        metadata = [name for name in members if name.endswith(".dist-info/WHEEL")]
        assert len(metadata) == 1
        assert "Root-Is-Purelib: false" in archive.read(metadata[0]).decode()


def _validate_sdist(path: Path) -> None:
    """Validate a source archive."""
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
    _validate_members(path, members)
    roots = {Path(member).name for member in members}
    assert {
        "CHANGES.rst", "CITATION.cff", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "setup.py",
    } <= roots
    assert any(name.endswith("pmriprep/models/dsc/signal_cpi/_convolution.c") for name in members)
    assert not any(name.endswith((".so", ".pyd", ".dylib")) for name in members)


def main() -> None:
    """Validate one wheel and one source distribution from command-line paths."""
    _validate_license_labels()
    paths = [Path(value) for value in sys.argv[1:]]
    wheels = [path for path in paths if path.suffix == ".whl"]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    assert len(wheels) == 1, f"Expected one wheel, found {wheels}"
    assert len(sdists) == 1, f"Expected one sdist, found {sdists}"
    _validate_wheel(wheels[0])
    _validate_sdist(sdists[0])


if __name__ == "__main__":
    main()
