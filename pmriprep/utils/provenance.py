"""Provenance tracking utilities for BIDS-Derivatives."""

import datetime
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pmriprep


def get_software_versions() -> Dict[str, str]:
    """Get versions of key software dependencies.

    Returns a dictionary with version information for:
    - Python
    - pmriprep
    - nipype
    - nibabel
    - numpy
    - scipy
    - nilearn

    Returns:
        Dictionary mapping package names to version strings.
    """
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "pmriprep": pmriprep.__version__,
    }

    # Core dependencies - import individually to handle missing packages
    optional_packages = [
        "nipype",
        "nibabel",
        "numpy",
        "scipy",
        "nilearn",
        "niworkflows",
        "templateflow",
    ]

    for pkg_name in optional_packages:
        try:
            pkg = __import__(pkg_name)
            if hasattr(pkg, "__version__"):
                versions[pkg_name] = str(pkg.__version__)
            elif hasattr(pkg, "version"):
                versions[pkg_name] = str(pkg.version)
            else:
                versions[pkg_name] = "unknown"
        except ImportError:
            versions[pkg_name] = "not installed"

    return versions


def get_scanner_metadata(
    json_path: Union[str, Path],
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract scanner metadata from a BIDS JSON sidecar.

    Args:
        json_path: Path to the JSON sidecar file.
        keys: Optional list of specific keys to extract. If None,
              extracts common scanner-related fields.

    Returns:
        Dictionary with extracted scanner metadata.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON sidecar not found: {json_path}")

    with open(json_path) as f:
        sidecar = json.load(f)

    # Default keys for scanner metadata if not specified
    if keys is None:
        keys = [
            # Scanner identification
            "Manufacturer",
            "ManufacturersModelName",
            "MagneticFieldStrength",
            "SoftwareVersions",
            "StationName",
            "InstitutionName",
            "InstitutionalDepartmentName",
            # Sequence parameters
            "PulseSequenceType",
            "SequenceName",
            "ScanningSequence",
            "SequenceVariant",
            "ScanOptions",
            # Acquisition parameters
            "RepetitionTime",
            "EchoTime",
            "InversionTime",
            "FlipAngle",
            "NumberOfAverages",
            "PartialFourier",
            "ParallelReductionFactorInPlane",
            # Contrast agent (for DCE/DSC)
            "ContrastBolusIngredient",
            "ContrastBolusTotalDose",
            "ContrastBolusVolume",
            # Timing for perfusion
            "AcquisitionDuration",
            "NumberOfVolumes",
            "SliceTiming",
        ]

    # Extract requested keys
    metadata: Dict[str, Any] = {}
    for key in keys:
        if key in sidecar:
            metadata[key] = sidecar[key]

    return metadata


def create_provenance_dict(
    source_files: Optional[List[Union[str, Path]]] = None,
    processing_description: Optional[str] = None,
    config_used: Optional[Dict[str, Any]] = None,
    scanner_metadata: Optional[Dict[str, Any]] = None,
    include_versions: bool = True,
    include_timestamp: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a provenance dictionary for BIDS-Derivatives JSON sidecars.

    This follows the BIDS-Derivatives specification for provenance tracking,
    including information about source files, processing parameters, and
    software versions.

    Args:
        source_files: List of paths to input files used to generate this output.
        processing_description: Human-readable description of the processing.
        config_used: Configuration parameters used for processing.
        scanner_metadata: Scanner metadata from BIDS JSON sidecar.
        include_versions: Whether to include software version information.
        include_timestamp: Whether to include processing timestamp.
        additional_metadata: Any additional key-value pairs to include.

    Returns:
        Dictionary suitable for JSON serialization as a sidecar file.
    """
    provenance: Dict[str, Any] = {}

    # Processing timestamp
    if include_timestamp:
        provenance["GeneratedBy"] = [
            {
                "Name": "pmriprep",
                "Version": pmriprep.__version__,
            }
        ]
        provenance["ProcessingDateTime"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

    # Source files
    if source_files:
        provenance["Sources"] = [str(Path(f).name) for f in source_files]
        provenance["RawSources"] = [str(f) for f in source_files]

    # Processing description
    if processing_description:
        provenance["Description"] = processing_description

    # Configuration used
    if config_used:
        provenance["ProcessingConfiguration"] = config_used

    # Software versions
    if include_versions:
        provenance["SoftwareVersions"] = get_software_versions()

    # Scanner metadata
    if scanner_metadata:
        provenance["ScannerMetadata"] = scanner_metadata

    # Additional metadata
    if additional_metadata:
        for key, value in additional_metadata.items():
            if key not in provenance:
                provenance[key] = value

    return provenance


def write_provenance_json(
    output_path: Union[str, Path],
    provenance: Dict[str, Any],
    indent: int = 2,
) -> Path:
    """Write a provenance dictionary to a JSON file.

    Args:
        output_path: Path for the output JSON file.
        provenance: Provenance dictionary to write.
        indent: JSON indentation level (default: 2).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(provenance, f, indent=indent, default=str)

    return output_path


def get_bids_derivative_provenance(
    pipeline_name: str = "pmriprep",
    pipeline_version: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a minimal provenance dict following BIDS-Derivatives GeneratedBy spec.

    This creates the standard "GeneratedBy" field used in dataset_description.json
    and individual derivative sidecars.

    Args:
        pipeline_name: Name of the pipeline (default: "pmriprep").
        pipeline_version: Version string. If None, uses pmriprep.__version__.
        description: Optional description of what the pipeline does.

    Returns:
        Dictionary with GeneratedBy field following BIDS spec.
    """
    version = pipeline_version or pmriprep.__version__

    generated_by: Dict[str, Any] = {
        "Name": pipeline_name,
        "Version": version,
    }

    if description:
        generated_by["Description"] = description

    return {"GeneratedBy": [generated_by]}
