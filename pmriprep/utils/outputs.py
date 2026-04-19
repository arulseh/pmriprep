"""BIDS-Derivatives output writer utilities for PMRIPrep.

This module provides utilities for writing NIfTI outputs with BIDS-compliant
naming and JSON sidecars following the BIDS-Derivatives specification.

It also provides shared helper functions used by all four derivative writer
interfaces (anat, DSC, DCE, HD-GLIO).
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

import pmriprep
from pmriprep.utils.provenance import get_software_versions


def write_qc_json(
    output_path: Union[str, Path],
    metrics: Dict[str, Any],
    subject: str,
    session: Optional[str] = None,
    pipeline_step: Optional[str] = None,
) -> Path:
    """Write a QC metrics JSON file.

    Args:
        output_path: Path for the output JSON file.
        metrics: Dictionary of QC metrics.
        subject: Subject ID.
        session: Session ID, optional.
        pipeline_step: Name of the pipeline step (e.g., 'aif_detection', 'registration').

    Returns:
        Path to the written JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qc_data: Dict[str, Any] = {
        "Subject": f"sub-{subject}",
        "GeneratedBy": [
            {
                "Name": "pmriprep",
                "Version": pmriprep.__version__,
            }
        ],
    }

    if session:
        qc_data["Session"] = f"ses-{session}"

    if pipeline_step:
        qc_data["PipelineStep"] = pipeline_step

    qc_data["Metrics"] = metrics
    qc_data["SoftwareVersions"] = get_software_versions()

    with open(output_path, "w") as f:
        json.dump(qc_data, f, indent=2, default=str)

    return output_path


# -----------------------------------------------------------------------
# Shared utilities for BIDS-Derivatives output writers
# -----------------------------------------------------------------------

# Model name → BIDS-safe entity value mapping
_MODEL_ENTITY_MAP: Dict[str, str] = {
    "tofts": "tofts",
    "extended_tofts": "extendedTofts",
    "patlak": "patlak",
    "two_compartment_exchange": "2CXM",
    "2cxm": "2CXM",
    "auto": "auto",
}


def sanitize_model_entity(model_name: str) -> str:
    """Convert a pharmacokinetic model name to a BIDS-safe entity value.

    BIDS entity values must be alphanumeric (no underscores or hyphens).
    This function maps known model names to their BIDS-safe equivalents
    and applies camelCase conversion for unknown names.

    Args:
        model_name: Raw model name (e.g., "extended_tofts", "2cxm").

    Returns:
        BIDS-safe entity value (e.g., "extendedTofts", "2CXM").
    """
    lowered = model_name.lower().strip()
    if lowered in _MODEL_ENTITY_MAP:
        return _MODEL_ENTITY_MAP[lowered]

    # Fallback: convert underscores to camelCase
    parts = model_name.split("_")
    if len(parts) > 1:
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    return model_name


def write_dataset_description(pipeline_dir: Path, bids_dir: str) -> None:
    """Write a unified dataset_description.json at the derivative root.

    This is the shared implementation used by all four output writers
    (anat, DSC, DCE, HD-GLIO). Always overwrites to ensure the version
    and metadata stay current across pipeline upgrades.

    Args:
        pipeline_dir: Path to the ``pmriprep/`` derivative root directory.
        bids_dir: Path to the source BIDS dataset root.
    """
    desc_path = pipeline_dir / "dataset_description.json"

    pipeline_dir.mkdir(parents=True, exist_ok=True)
    description: Dict[str, Any] = {
        "Name": "PMRIPrep - Perfusion MRI Preprocessing Pipeline",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "pmriprep",
                "Version": pmriprep.__version__,
                "Description": (
                    "Automated preprocessing and quantification of "
                    "DSC and DCE data"
                ),
            }
        ],
        "SourceDatasets": [
            {"URL": Path(bids_dir).name},
        ],
        "HowToAcknowledge": (
            f"Please cite PMRIPrep (version {pmriprep.__version__}), "
            "https://github.com/arulseh/pmriprep"
        ),
        "License": "BSD-3-Clause",
    }
    with open(desc_path, "w") as f:
        json.dump(description, f, indent=2)


# -----------------------------------------------------------------------
# Shared helpers for derivative writer interfaces
# -----------------------------------------------------------------------


def has_input(inputs: Any, name: str) -> bool:
    """Check if a Nipype interface input is defined and points to an existing file.

    Handles all sentinel patterns used by Nipype traits (``Undefined``,
    empty strings, ``"None"``, and ``"."``).

    Args:
        inputs: A Nipype ``BaseInterfaceInputSpec`` instance (e.g. ``self.inputs``).
        name: Name of the input attribute to check.

    Returns:
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

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def write_json_sidecar(data: dict, path: Path) -> None:
    """Write a dictionary as a JSON sidecar file.

    Creates parent directories as needed.  Non-serialisable values are
    converted to strings via the ``default=str`` fallback.

    Args:
        data: Dictionary to serialise.
        path: Output JSON file path.
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
    "rmsd",
]


def par_to_confounds_tsv(
    par_file: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Convert an FSL mcflirt ``.par`` file to a BIDS confounds TSV.

    FSL column order: rot_x, rot_y, rot_z, trans_x, trans_y, trans_z.
    BIDS column order: trans_x, trans_y, trans_z, rot_x, rot_y, rot_z,
    framewise_displacement, rmsd.

    Rotations are kept in **radians** (matching fMRIPrep convention).
    The first row of ``framewise_displacement`` is ``n/a`` because FD is
    undefined for the reference frame.

    Args:
        par_file: Path to the ``.par`` motion parameters file.
        output_path: Path for the output ``.tsv`` file.

    Returns:
        Path to the written TSV file.
    """
    from pmriprep.utils.motion import (
        calculate_framewise_displacement,
        calculate_rms_displacement,
    )

    par_file = Path(par_file)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load FSL .par: columns are rot_x, rot_y, rot_z, trans_x, trans_y, trans_z
    params = np.loadtxt(str(par_file))
    if params.ndim == 1:
        params = params.reshape(1, -1)

    # Reorder: FSL [rot(0:3), trans(3:6)] → BIDS [trans, rot]
    trans = params[:, 3:6]
    rot = params[:, :3]

    # Compute derived metrics (functions expect FSL order)
    fd = calculate_framewise_displacement(params)
    rmsd = calculate_rms_displacement(params)

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
            f"{rmsd[i]:.6f}",
        ]
        lines.append("\t".join(row_vals))

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def write_confounds_sidecar(output_path: Union[str, Path]) -> Path:
    """Write the JSON sidecar for a BIDS confounds TSV.

    Describes each column's meaning and units following the BIDS
    common derivatives specification.

    Args:
        output_path: Path for the output ``.json`` sidecar file.

    Returns:
        Path to the written JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar: Dict[str, Any] = {
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
                "Framewise displacement (Power et al., 2012). "
                "Sum of absolute translations plus rotations "
                "converted to mm via 50 mm head radius."
            ),
            "Units": "mm",
            "HeadRadius": 50.0,
        },
        "rmsd": {
            "Description": (
                "Root-mean-square displacement relative to reference frame, "
                "combining rotations (converted via 50 mm head radius) "
                "and translations."
            ),
            "Units": "mm",
            "HeadRadius": 50.0,
        },
    }

    with open(output_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    return output_path
