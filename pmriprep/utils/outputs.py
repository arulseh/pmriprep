"""BIDS-Derivatives output writer utilities for PMRIPrep.

This module provides utilities for writing NIfTI outputs with BIDS-compliant
naming and JSON sidecars following the BIDS-Derivatives specification.

It also provides shared helper functions used by all four derivative writer
interfaces (anat, DSC, DCE, HD-GLIO).
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

import pmriprep
from pmriprep.utils.provenance import (
    create_provenance_dict,
    get_bids_derivative_provenance,
    get_software_versions,
)


@dataclass
class DerivativeMetadata:
    """Metadata for a BIDS-Derivatives output file.

    Attributes:
        description: Human-readable description of the derivative.
        sources: List of source file paths used to generate this derivative.
        processing_description: Description of the processing applied.
        units: Measurement units (e.g., "mL/100g/min" for CBF).
        parameter_name: Name of the parameter if this is a map (e.g., "CBF", "Ktrans").
        model: Model used for computation (e.g., "sSVD", "extended_tofts").
        space: Coordinate space (e.g., "native", "anat", "MNI152NLin2009cAsym").
        resolution: Resolution description (e.g., "1mm", "native").
        additional: Additional key-value pairs to include in the sidecar.
    """

    description: Optional[str] = None
    sources: Optional[List[Union[str, Path]]] = None
    processing_description: Optional[str] = None
    units: Optional[str] = None
    parameter_name: Optional[str] = None
    model: Optional[str] = None
    space: Optional[str] = None
    resolution: Optional[str] = None
    additional: Optional[Dict[str, Any]] = None


# Pipeline names for dataset_description.json
PIPELINE_DESCRIPTIONS: Dict[str, str] = {
    "anat": "Anatomical preprocessing including brain extraction and segmentation",
    "dsc": "Dynamic Susceptibility Contrast perfusion MRI processing",
    "dce": "Dynamic Contrast Enhanced perfusion MRI processing",
    "tumor_seg": "Tumor segmentation",
}


def write_derivative(
    data: NDArray[np.floating[Any]],
    output_path: Union[str, Path],
    reference_image: Optional[Union[str, Path]] = None,
    affine: Optional[NDArray[np.floating[Any]]] = None,
    metadata: Optional[DerivativeMetadata] = None,
    config_used: Optional[Dict[str, Any]] = None,
    scanner_metadata: Optional[Dict[str, Any]] = None,
    include_provenance: bool = True,
) -> Path:
    """Write a derivative NIfTI file with BIDS-compliant JSON sidecar.

    This function writes a NIfTI file and creates an accompanying JSON sidecar
    with provenance information following BIDS-Derivatives specification.

    Args:
        data: NumPy array containing the image data.
        output_path: Path for the output NIfTI file (should end in .nii.gz).
        reference_image: Path to a reference image for header information.
            If provided, the output will inherit the affine and header.
        affine: Affine transformation matrix. Required if reference_image not provided.
        metadata: DerivativeMetadata object with description and provenance info.
        config_used: Configuration dictionary used for processing.
        scanner_metadata: Scanner metadata from source BIDS JSON.
        include_provenance: Whether to create JSON sidecar with provenance.

    Returns:
        Path to the written NIfTI file.

    Raises:
        ValueError: If neither reference_image nor affine is provided.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get affine from reference or parameter
    if reference_image is not None:
        ref_img = nib.load(str(reference_image))
        ref_affine = ref_img.affine  # type: ignore[attr-defined]
        ref_header = ref_img.header  # type: ignore[attr-defined]
    elif affine is not None:
        ref_affine = affine
        ref_header = None
    else:
        raise ValueError("Either reference_image or affine must be provided")

    # Create NIfTI image
    if ref_header is not None:
        # Create new header based on reference
        new_header = ref_header.copy()
        new_header.set_data_shape(data.shape)  # type: ignore[attr-defined]
        img = nib.Nifti1Image(data.astype(np.float32), ref_affine, header=new_header)
    else:
        img = nib.Nifti1Image(data.astype(np.float32), ref_affine)

    # Write NIfTI file
    nib.save(img, str(output_path))

    # Create JSON sidecar with provenance
    if include_provenance and metadata is not None:
        json_path = _get_sidecar_path(output_path)
        _write_derivative_sidecar(
            json_path,
            metadata=metadata,
            config_used=config_used,
            scanner_metadata=scanner_metadata,
        )

    return output_path


def _get_sidecar_path(nifti_path: Path) -> Path:
    """Get the JSON sidecar path for a NIfTI file.

    Args:
        nifti_path: Path to the NIfTI file.

    Returns:
        Path for the JSON sidecar (same name, .json extension).
    """
    name = nifti_path.name
    # Handle .nii.gz and .nii extensions
    if name.endswith(".nii.gz"):
        json_name = name[:-7] + ".json"
    elif name.endswith(".nii"):
        json_name = name[:-4] + ".json"
    else:
        json_name = name + ".json"

    return nifti_path.parent / json_name


def _write_derivative_sidecar(
    json_path: Path,
    metadata: DerivativeMetadata,
    config_used: Optional[Dict[str, Any]] = None,
    scanner_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a JSON sidecar for a derivative file.

    Args:
        json_path: Path for the output JSON file.
        metadata: DerivativeMetadata with description and sources.
        config_used: Configuration used for processing.
        scanner_metadata: Scanner metadata from source BIDS JSON.

    Returns:
        Path to the written JSON file.
    """
    # Build provenance dict
    provenance = create_provenance_dict(
        source_files=[str(s) for s in metadata.sources] if metadata.sources else None,
        processing_description=metadata.processing_description,
        config_used=config_used,
        scanner_metadata=scanner_metadata,
        include_versions=True,
        include_timestamp=True,
    )

    # Add derivative-specific fields
    sidecar: Dict[str, Any] = {}

    if metadata.description:
        sidecar["Description"] = metadata.description

    if metadata.units:
        sidecar["Units"] = metadata.units

    if metadata.parameter_name:
        sidecar["ParameterName"] = metadata.parameter_name

    if metadata.model:
        sidecar["Model"] = metadata.model

    if metadata.space:
        sidecar["SkullStripped"] = True  # Common for derivatives
        sidecar["Resolution"] = metadata.resolution or "native"

    # Merge provenance
    sidecar.update(provenance)

    # Add any additional metadata
    if metadata.additional:
        for key, value in metadata.additional.items():
            if key not in sidecar:
                sidecar[key] = value

    # Write JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)

    return json_path


def create_dataset_description(
    output_dir: Union[str, Path],
    pipeline_name: str = "pmriprep",
    pipeline_description: Optional[str] = None,
    bids_version: str = "1.8.0",
    dataset_type: str = "derivative",
    generated_by: Optional[List[Dict[str, Any]]] = None,
    source_datasets: Optional[List[Dict[str, str]]] = None,
) -> Path:
    """Create a dataset_description.json for a BIDS-Derivatives directory.

    This follows the BIDS-Derivatives specification for dataset description,
    including the GeneratedBy field with pipeline information.

    Args:
        output_dir: Path to the derivatives output directory.
        pipeline_name: Name of the pipeline (default: "pmriprep").
        pipeline_description: Description of the pipeline. If None, uses default.
        bids_version: BIDS version string (default: "1.8.0").
        dataset_type: Dataset type, should be "derivative" for derivatives.
        generated_by: List of GeneratedBy entries. If None, creates default.
        source_datasets: List of source dataset references.

    Returns:
        Path to the created dataset_description.json file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build GeneratedBy if not provided
    if generated_by is None:
        generated_by = [
            {
                "Name": pipeline_name,
                "Version": pmriprep.__version__,
            }
        ]

    if pipeline_description:
        generated_by[0]["Description"] = pipeline_description

    # Build dataset description
    description: Dict[str, Any] = {
        "Name": f"{pipeline_name} outputs",
        "BIDSVersion": bids_version,
        "DatasetType": dataset_type,
        "GeneratedBy": generated_by,
    }

    # Add source datasets if provided
    if source_datasets:
        description["SourceDatasets"] = source_datasets

    # Add pipeline-specific metadata
    description["HowToAcknowledge"] = (
        f"Please cite pmriprep (version {pmriprep.__version__}) "
        "when using these outputs in publications."
    )
    description["License"] = "CC-BY-4.0"

    # Write the file
    desc_path = output_dir / "dataset_description.json"
    with open(desc_path, "w") as f:
        json.dump(description, f, indent=2)

    return desc_path


def ensure_dataset_description(
    output_dir: Union[str, Path],
    pipeline: str = "pmriprep",
) -> Path:
    """Ensure dataset_description.json exists in the derivatives directory.

    If the file already exists, returns its path. Otherwise, creates it.

    Args:
        output_dir: Derivatives output directory.
        pipeline: Pipeline name.

    Returns:
        Path to the dataset_description.json file.
    """
    output_dir = Path(output_dir)
    desc_path = output_dir / pipeline / "dataset_description.json"

    if desc_path.exists():
        return desc_path

    # Determine pipeline description
    pipeline_desc = PIPELINE_DESCRIPTIONS.get(
        pipeline.lower(),
        "Perfusion MRI preprocessing and analysis",
    )

    return create_dataset_description(
        output_dir=output_dir / pipeline,
        pipeline_name=pipeline,
        pipeline_description=pipeline_desc,
    )


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
    (anat, DSC, DCE, HD-GLIO).  The first writer to run creates the
    file; subsequent writers skip if it already exists.

    Args:
        pipeline_dir: Path to the ``pmriprep/`` derivative root directory.
        bids_dir: Path to the source BIDS dataset root.
    """
    desc_path = pipeline_dir / "dataset_description.json"
    if desc_path.exists():
        return

    pipeline_dir.mkdir(parents=True, exist_ok=True)
    description: Dict[str, Any] = {
        "Name": "PMRIPrep - Perfusion MRI Preprocessing Pipeline",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "pmriprep",
                "Version": pmriprep.__version__,
                "Description": (
                    "Anatomical preprocessing, DSC/DCE perfusion processing, "
                    "and HD-GLIO tumor segmentation"
                ),
            }
        ],
        "SourceDatasets": [
            {"URL": Path(bids_dir).name},
        ],
        "HowToAcknowledge": (
            f"Please cite pmriprep (version {pmriprep.__version__}) "
            "when using these outputs in publications."
        ),
        "License": "CC-BY-4.0",
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
