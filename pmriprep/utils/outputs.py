"""BIDS-Derivatives output writer utilities for PMRIPrep.

This module provides utilities for writing NIfTI outputs with BIDS-compliant
naming and JSON sidecars following the BIDS-Derivatives specification.

It also provides shared helper functions used by all four derivative writer
interfaces (anat, DSC, DCE, HD-GLIO).
"""

import hashlib
import json
import os
import shutil
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Union

import numpy as np

import pmriprep

#: BIDS specification version that the derivative naming and JSON sidecars
#: target. Bump deliberately when migrating to a newer spec; downstream
#: validators key off this string.
BIDS_DERIVATIVES_VERSION: Final[str] = "1.11.1"

#: Public source URL for the pipeline. Surfaced in dataset_description's
#: ``GeneratedBy.CodeURL`` and in ``HowToAcknowledge``.
PMRIPREP_CODE_URL: Final[str] = "https://github.com/arulseh/pmriprep"



# -----------------------------------------------------------------------
# Shared utilities for BIDS-Derivatives output writers
# -----------------------------------------------------------------------


def warn_unproduced_perfusion_outputs(
    output_dir: str,
    subject: str,
    session: str,
    source_file: str,
    map_stems: list[str],
    required_stem: str,
    figure_dependencies: dict[str, str],
    written_files: list,
) -> list[str]:
    """Warn about leftover maps after the current acquisition's sinks complete.

    Parameters
    ----------
    output_dir : str
        Derivatives parent directory containing ``pmriprep``.
    subject, session : str
        Output subject and optional session identifiers.
    source_file : str
        Source image whose entities scope the check to one acquisition.
    map_stems : list of str
        Exact owned map tails, starting with ``desc-`` and excluding extensions.
    required_stem : str
        Map tail that must have been written before checking for leftovers.
        A completed fitting export establishes which fit outputs are current.
    figure_dependencies : dict of str to str
        Owned map tails and their dependent figure tails, including extension.
    written_files : list
        Joined sink results, possibly nested and containing empty outputs.

    Returns
    -------
    list of str
        Flattened current sink results for the downstream completion barrier.

    Notes
    -----
    The warning lists session-relative paths of recognized files absent from
    the current writer results.
    """
    import logging
    from pathlib import Path

    from pmriprep.utils.sidecar_builders import source_entity_prefix

    def flatten(items: list) -> list[str]:
        files: list[str] = []
        for item in items:
            if isinstance(item, (list, tuple)):
                files.extend(flatten(list(item)))
            elif isinstance(item, str) and item:
                files.append(item)
        return files

    files = flatten(written_files)
    current = {Path(file).absolute() for file in files}
    root = Path(output_dir) / "pmriprep" / f"sub-{subject}"
    if session:
        root /= f"ses-{session}"
    prefix = source_entity_prefix(source_file, subject, session)
    required = root / "perf" / f"{prefix}_{required_stem}.nii.gz"
    if required.absolute() not in current:
        return files

    for tail in [*map_stems, required_stem, *figure_dependencies.values()]:
        if not tail.startswith("desc-") or Path(tail).name != tail:
            raise ValueError(f"Invalid owned derivative tail: {tail}")
    if not set(figure_dependencies).issubset(map_stems):
        raise ValueError("Figure dependencies must name owned maps")
    leftovers: list[str] = []
    for stem in map_stems:
        image = root / "perf" / f"{prefix}_{stem}.nii.gz"
        if image.absolute() in current:
            continue
        candidates = [image, image.with_name(f"{prefix}_{stem}.json")]
        if stem in figure_dependencies:
            candidates.append(root / "figures" / f"{prefix}_{figure_dependencies[stem]}")
        for path in candidates:
            if path.is_file() or path.is_symlink():
                leftovers.append(str(path.relative_to(root)))

    warning_prefix = "These files were not produced by the current run and may contain earlier results: "
    warning = warning_prefix + ", ".join(sorted(set(leftovers))) if leftovers else None
    if warning:
        logging.getLogger("nipype.workflow").warning("%s", warning)

    return files


def _dataset_description_locked(func: Any) -> Any:
    """Serialize dataset-description updates across pipeline writer processes."""
    @wraps(func)
    def wrapped(pipeline_dir: Path, *args: Any, **kwargs: Any) -> Any:
        import fcntl

        resolved = str(Path(pipeline_dir).resolve()).encode()
        digest = hashlib.sha256(resolved).hexdigest()[:20]
        lock_path = Path(tempfile.gettempdir()) / f"pmriprep-dd-{digest}.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return func(Path(pipeline_dir), *args, **kwargs)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return wrapped


@_dataset_description_locked
def write_dataset_description(
    pipeline_dir: Path,
    bids_dir: str,
    citation_doi: Optional[str] = None,
) -> None:
    """Write a unified dataset_description.json at the derivative root.

    This is the shared implementation used by all four output writers
    (anat, DSC, DCE, HD-GLIO). Merges this run's pmriprep ``GeneratedBy``
    entry with any pre-existing entries from other NiPreps tools that
    wrote into the same derivative root, preserving cross-tool provenance.

    Parameters
    ----------
    pipeline_dir : Path
        Path to the ``pmriprep/`` derivative root directory.
    bids_dir : str
        Path to the source BIDS dataset root.
    citation_doi : str, optional
        Optional DOI string to attach to the pmriprep
        ``GeneratedBy`` entry (e.g. for cite-by-version manuscripts).

    Raises
    ------
    ValueError
        If existing metadata is not a valid JSON object, links to a different
        raw dataset, or uses the reserved empty dataset-link name.
    """
    desc_path = pipeline_dir / "dataset_description.json"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # pmriprep's own GeneratedBy entry for this run.
    pmriprep_entry: Dict[str, Any] = {
        "Name": "pmriprep",
        "Version": pmriprep.__version__,
        "Description": (
            "Automated preprocessing and quantification of "
            "DSC and DCE MRI"
        ),
        "CodeURL": PMRIPREP_CODE_URL,
    }
    if citation_doi:
        pmriprep_entry["DOI"] = citation_doi

    # This function runs once per pipeline datasink (anat/DSC/DCE/tumor-seg), so
    # it must be idempotent: drop EVERY pmriprep-managed entry (pmriprep + the
    # third-party suites) before re-appending the freshly rebuilt managed block,
    # otherwise the tool entries accumulate across the 4 calls. Genuinely-foreign
    # entries are preserved; a foreign entry literally named e.g. "FSL" would be
    # collapsed, but pmriprep writes only to its own ``pmriprep/`` root so that
    # never occurs in practice.
    from pmriprep.utils.sidecar_builders import (
        _PMRIPREP_MANAGED_TOOL_NAMES,
        build_tool_generatedby_entries,
    )

    existing_generated_by: List[Dict[str, Any]] = []
    existing_dataset_links: Dict[str, str] = {}
    existing_source_datasets: List[Dict[str, str]] = []
    if desc_path.exists():
        try:
            with open(desc_path) as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                raise ValueError(
                    f"Existing derivative dataset description must be a JSON object: {desc_path}"
                )
            existing_generated_by = [
                e for e in existing.get("GeneratedBy", [])
                if isinstance(e, dict)
                and e.get("Name") not in _PMRIPREP_MANAGED_TOOL_NAMES
            ]
            links = existing.get("DatasetLinks")
            if isinstance(links, dict):
                existing_dataset_links = {
                    str(name): str(location)
                    for name, location in links.items()
                }
            sources = existing.get("SourceDatasets")
            if isinstance(sources, list):
                existing_source_datasets = [
                    {str(key): str(value) for key, value in source.items()}
                    for source in sources
                    if isinstance(source, dict)
                ]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Existing derivative dataset description is not valid JSON: {desc_path}"
            ) from exc

    managed_generated_by = [pmriprep_entry] + build_tool_generatedby_entries()
    raw_dataset_link = Path(os.path.relpath(
        Path(bids_dir).resolve(),
        start=pipeline_dir.resolve(),
    )).as_posix()
    prior_raw_link = existing_dataset_links.get("raw")
    prior_raw_path = (
        (pipeline_dir / prior_raw_link).resolve()
        if prior_raw_link is not None and ":" not in prior_raw_link
        else None
    )
    if (
        prior_raw_link is not None
        and prior_raw_link != raw_dataset_link
        and prior_raw_path != Path(bids_dir).resolve()
    ):
        raise ValueError(
            "Existing derivative dataset links to a different raw dataset: "
            f"{prior_raw_link!r} != {raw_dataset_link!r}"
        )
    dataset_links = {**existing_dataset_links, "raw": raw_dataset_link}
    if "" in dataset_links:
        raise ValueError("DatasetLinks must not contain the reserved empty dataset name")

    raw_dataset_path = Path(bids_dir).resolve()
    source_datasets = list(existing_source_datasets)
    if not any(
        source.get("URL") == raw_dataset_link
        or (
            source.get("URL") is not None
            and ":" not in source["URL"]
            and (pipeline_dir / source["URL"]).resolve() == raw_dataset_path
        )
        for source in source_datasets
    ):
        source_datasets.append({"URL": raw_dataset_link})

    description: Dict[str, Any] = {
        "Name": "PMRIPrep: automated preprocessing and quantification of DSC and DCE MRI",
        "BIDSVersion": BIDS_DERIVATIVES_VERSION,
        "DatasetType": "derivative",
        "DatasetLinks": dataset_links,
        "GeneratedBy": managed_generated_by + existing_generated_by,
        "SourceDatasets": source_datasets,
        "HowToAcknowledge": (
            f"Please cite PMRIPrep (version {pmriprep.__version__}), "
            f"{PMRIPREP_CODE_URL}"
        ),
        "License": "Apache-2.0",
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=pipeline_dir,
            prefix=".dataset_description.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(description, temporary, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, desc_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    write_derivative_readme(pipeline_dir)
    write_bidsignore(pipeline_dir)


_BIDSIGNORE_CONTENT = """\
# QC reports
**/*.html

# QC figures
**/figures/

# Pipeline run logs (nipype) and citation boilerplate
**/log/
**/logs/

# ITK/ANTs composite transforms (.h5 not yet in BIDS schema)
**/*_xfm.h5
"""


def write_derivative_readme(pipeline_dir: Path) -> None:
    """Write the required README at the derivative dataset root."""
    readme_path = pipeline_dir / "README"
    readme_path.write_text(
        "PMRIPrep derivative dataset\n"
        "===========================\n\n"
        f"This dataset was generated by PMRIPrep {pmriprep.__version__}.\n"
        "It contains anatomical preprocessing, perfusion MRI derivatives, "
        "and associated quality-control outputs.\n\n"
        "See dataset_description.json and image sidecars for processing "
        "provenance and source-file references.\n"
    )


def write_bidsignore(pipeline_dir: Path) -> None:
    """Write a ``.bidsignore`` file at the derivative root.

    Idempotent — overwrites any existing file with the canonical
    content so it stays in sync with pipeline changes.

    Parameters
    ----------
    pipeline_dir : Path
        Root directory of the BIDS-Derivatives tree.
    """
    ignore_path = pipeline_dir / ".bidsignore"
    ignore_path.write_text(_BIDSIGNORE_CONTENT)


# -----------------------------------------------------------------------
# Shared helpers for derivative writer interfaces
# -----------------------------------------------------------------------


def has_input(inputs: Any, name: str) -> bool:
    """Check if a Nipype interface input is defined and points to an existing file.

    Handles all sentinel patterns used by Nipype traits (``Undefined``,
    empty strings, ``"None"``, and ``"."``).

    Parameters
    ----------
    inputs : Any
        A Nipype ``BaseInterfaceInputSpec`` instance (e.g. ``self.inputs``).
    name : str
        Name of the input attribute to check.

    Returns
    -------
    bool
        ``True`` if the input is set and points to an existing file.
    """
    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    val = getattr(inputs, name, None)
    if val is None or not isdefined(val):
        return False
    if isinstance(val, str):
        if val in ("", "None", "."):
            return False
        return Path(val).exists()
    return False


def copy_derivative(src: str, dst: Path) -> None:
    """Copy a derivative file (NIfTI, transform, etc.) to *dst*.

    Creates parent directories as needed. Preserves filesystem metadata
    via :func:`shutil.copy2`.

    Parameters
    ----------
    src : str
        Source file path.
    dst : Path
        Destination file path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def write_json_sidecar(data: dict, path: Path) -> None:
    """Write a dictionary as a JSON sidecar file.

    Creates parent directories as needed.  Non-serialisable values are
    converted to strings via the ``default=str`` fallback.

    Parameters
    ----------
    data : dict
        Dictionary to serialise.
    path : Path
        Output JSON file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# -----------------------------------------------------------------------
# Motion confounds: .par → BIDS confounds TSV + JSON sidecar
# -----------------------------------------------------------------------

# BIDS confounds column names (BIDS order: translations first, then rotations)
_CONFOUNDS_COLUMNS = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "framewise_displacement",
]


def par_to_confounds_tsv(
    par_file: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Convert a common six-column ``.par`` file to a BIDS confounds TSV.

    FSL column order: rot_x, rot_y, rot_z, trans_x, trans_y, trans_z.
    Output column order: trans_x, trans_y, trans_z, rot_x, rot_y, rot_z,
    framewise_displacement.

    Rotations are kept in **radians** (matching fMRIPrep convention).
    The first row of ``framewise_displacement`` is ``n/a`` because the first
    acquired volume has no preceding volume.

    Parameters
    ----------
    par_file : str or Path
        Path to the ``.par`` motion parameters file.
    output_path : str or Path
        Path for the output ``.tsv`` file.

    Returns
    -------
    Path
        Path to the written TSV file.

    Raises
    ------
    ValueError
        If the loaded motion parameters do not have exactly six columns or
        contain nonfinite values.
    """
    from pmriprep.utils.motion import calculate_framewise_displacement

    par_file = Path(par_file)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load FSL .par: columns are rot_x, rot_y, rot_z, trans_x, trans_y, trans_z
    params = np.loadtxt(str(par_file))
    if params.ndim == 1:
        params = params.reshape(1, -1)
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(
            "Motion parameter file must contain exactly 6 columns; "
            f"got shape {params.shape}"
        )
    if not np.all(np.isfinite(params)):
        raise ValueError("Motion parameter file contains non-finite values")

    # Reorder: FSL [rot(0:3), trans(3:6)] → BIDS [trans, rot]
    trans = params[:, 3:6]
    rot = params[:, :3]

    # Compute derived metrics (functions expect FSL order)
    fd = calculate_framewise_displacement(params)

    # Build output lines
    n_frames = params.shape[0]
    lines: list[str] = ["\t".join(_CONFOUNDS_COLUMNS)]
    for i in range(n_frames):
        row_vals = [
            f"{trans[i, 0]:.6f}",
            f"{trans[i, 1]:.6f}",
            f"{trans[i, 2]:.6f}",
            f"{rot[i, 0]:.6f}",
            f"{rot[i, 1]:.6f}",
            f"{rot[i, 2]:.6f}",
            "n/a" if i == 0 else f"{fd[i]:.6f}",
        ]
        lines.append("\t".join(row_vals))

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def write_confounds_sidecar(
    output_path: Union[str, Path],
    source_path: Optional[str] = None,
    bids_dir: str = "",
    motion_applied: bool = True,
) -> Path:
    """Write the JSON sidecar for a BIDS confounds TSV.

    Describes each column's meaning and units following the BIDS
    common derivatives specification.

    Parameters
    ----------
    output_path : str or Path
        Path for the output ``.json`` sidecar file.
    source_path : str, optional
        Raw BIDS source NIfTI (for Sources URI).
    bids_dir : str, optional
        BIDS rawdata root (for Sources URI construction).
    motion_applied : bool, optional
        Whether the estimated ANTs transforms were applied. The columns
        describe input motion in both cases, not motion remaining after correction.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance
    from pmriprep.utils.sidecar_builders import build_sources_list

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sources = build_sources_list(
        [source_path] if source_path else [], bids_dir or ".",
    )

    motion_note = (
        {
            "Note": (
                "ANTs estimated input motion for QC; correction was not applied "
                "and the original frames were retained without resampling."
            )
        }
        if not motion_applied
        else {}
    )

    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        **({"Sources": sources} if sources else {}),
        "MotionCorrectionApplied": bool(motion_applied),
        "MotionEstimation": "ANTs rigid registration",
        "MotionParameterConvention": "FSL scaled-voxel moving-to-reference; reference intensity centre",
        **motion_note,
        "trans_x": {
            "Description": "Translation along the x-axis",
            "Units": "mm",
        },
        "trans_y": {
            "Description": "Translation along the y-axis",
            "Units": "mm",
        },
        "trans_z": {
            "Description": "Translation along the z-axis",
            "Units": "mm",
        },
        "rot_x": {
            "Description": "Rotation about the x-axis",
            "Units": "rad",
        },
        "rot_y": {
            "Description": "Rotation about the y-axis",
            "Units": "rad",
        },
        "rot_z": {
            "Description": "Rotation about the z-axis",
            "Units": "rad",
        },
        "framewise_displacement": {
            "Description": (
                "Sum of absolute translations plus rotations "
                "converted to mm via 50 mm head radius."
            ),
            "Units": "mm",
            "HeadRadius": 50.0,
        },
    }

    with open(output_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    return output_path
