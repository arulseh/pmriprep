"""BIDS-Derivatives output writer utilities for PMRIPrep.

This module provides utilities for writing NIfTI outputs with BIDS-compliant
naming and JSON sidecars following the BIDS-Derivatives specification.
"""

import json
from dataclasses import asdict, dataclass
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
        space: Coordinate space (e.g., "native", "T1w", "MNI152NLin2009cAsym").
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


# Standard suffixes for perfusion outputs following ASLPrep patterns
PERFUSION_SUFFIXES: Dict[str, Dict[str, str]] = {
    # DSC perfusion maps
    "cbf": {"suffix": "cbf", "units": "mL/100g/min", "description": "Cerebral blood flow"},
    "cbv": {"suffix": "cbv", "units": "mL/100g", "description": "Cerebral blood volume"},
    "mtt": {"suffix": "mtt", "units": "s", "description": "Mean transit time"},
    "tmax": {"suffix": "tmax", "units": "s", "description": "Time to maximum"},
    "ttp": {"suffix": "ttp", "units": "s", "description": "Time to peak"},
    "psr": {"suffix": "psr", "units": "percent", "description": "Percent signal recovery"},
    "ph": {"suffix": "ph", "units": "percent", "description": "Peak height"},
    "k2": {"suffix": "k2", "units": "1/s", "description": "Leakage correction factor K2"},
    "ncbv": {"suffix": "ncbv", "units": "ratio", "description": "Normalized CBV (relative to NAWM)"},
    # DCE pharmacokinetic maps
    "ktrans": {"suffix": "Ktrans", "units": "1/min", "description": "Volume transfer constant"},
    "ve": {"suffix": "ve", "units": "fraction", "description": "Extravascular extracellular volume fraction"},
    "vp": {"suffix": "vp", "units": "fraction", "description": "Plasma volume fraction"},
    "kep": {"suffix": "kep", "units": "1/min", "description": "Rate constant (Ktrans/ve)"},
    "fp": {"suffix": "Fp", "units": "mL/min/100mL", "description": "Plasma flow (2CXM)"},
    "ps": {"suffix": "PS", "units": "mL/min/100mL", "description": "Permeability-surface area product"},
    # Mapping outputs
    "t1": {"suffix": "T1map", "units": "ms", "description": "T1 relaxation time map"},
    "b1": {"suffix": "B1plusmap", "units": "ratio", "description": "B1+ transmit field map"},
    # Masks and segmentations
    "brain_mask": {"suffix": "mask", "description": "Brain mask"},
    "lesion_mask": {"suffix": "mask", "description": "Lesion/tumor mask"},
    "aif_mask": {"suffix": "mask", "description": "Arterial input function ROI mask"},
    "nawm_mask": {"suffix": "mask", "description": "Normal-appearing white matter mask"},
    "tumor_seg": {"suffix": "dseg", "description": "Tumor segmentation (HD-GLIO labels)"},
    # Tissue segmentations
    "wm_probseg": {"suffix": "probseg", "description": "White matter probability map"},
    "gm_probseg": {"suffix": "probseg", "description": "Gray matter probability map"},
    "csf_probseg": {"suffix": "probseg", "description": "CSF probability map"},
}

# Pipeline names for dataset_description.json
PIPELINE_DESCRIPTIONS: Dict[str, str] = {
    "anat": "Anatomical preprocessing including brain extraction and segmentation",
    "dsc": "Dynamic Susceptibility Contrast perfusion MRI processing",
    "dce": "Dynamic Contrast Enhanced perfusion MRI processing",
    "hdglio": "HD-GLIO tumor segmentation",
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
        # Update data shape if different
        if data.ndim == 3:
            new_header.set_data_shape(data.shape)  # type: ignore[attr-defined]
        elif data.ndim == 4:
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
                "CodeURL": "https://github.com/your-org/pmriprep",
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


def get_derivative_filename(
    subject: str,
    suffix: str,
    session: Optional[str] = None,
    space: Optional[str] = None,
    desc: Optional[str] = None,
    extension: str = ".nii.gz",
    **entities: str,
) -> str:
    """Generate a BIDS-Derivatives compliant filename.

    Args:
        subject: Subject ID (without 'sub-' prefix).
        suffix: BIDS suffix (e.g., 'cbf', 'T1w', 'mask').
        session: Session ID (without 'ses-' prefix), optional.
        space: Space entity (e.g., 'native', 'T1w', 'MNI152NLin2009cAsym').
        desc: Description entity (e.g., 'preproc', 'brain').
        extension: File extension (default: '.nii.gz').
        **entities: Additional BIDS entities as key-value pairs.

    Returns:
        BIDS-compliant filename string.
    """
    parts: List[str] = [f"sub-{subject}"]

    if session:
        parts.append(f"ses-{session}")

    # Standard entity order for derivatives
    entity_order = ["task", "acq", "ce", "rec", "run", "echo", "part", "space", "res", "label", "desc"]

    all_entities: Dict[str, str] = dict(entities)
    if space:
        all_entities["space"] = space
    if desc:
        all_entities["desc"] = desc

    for key in entity_order:
        if key in all_entities:
            parts.append(f"{key}-{all_entities[key]}")

    # Add any remaining entities not in standard order
    for key, value in all_entities.items():
        if key not in entity_order:
            parts.append(f"{key}-{value}")

    filename = "_".join(parts) + f"_{suffix}{extension}"
    return filename


def get_derivative_path(
    derivatives_dir: Union[str, Path],
    subject: str,
    pipeline: str,
    datatype: str,
    filename: str,
    session: Optional[str] = None,
) -> Path:
    """Build the full path for a derivative file.

    Follows BIDS-Derivatives structure:
    derivatives/<pipeline>/sub-<subject>/[ses-<session>/]<datatype>/<filename>

    Args:
        derivatives_dir: Root derivatives directory.
        subject: Subject ID (without 'sub-' prefix).
        pipeline: Pipeline name (e.g., 'pmriprep').
        datatype: Data type directory (e.g., 'anat', 'perf').
        filename: BIDS-compliant filename.
        session: Session ID (without 'ses-' prefix), optional.

    Returns:
        Complete path to the derivative file.
    """
    path = Path(derivatives_dir) / pipeline / f"sub-{subject}"

    if session:
        path = path / f"ses-{session}"

    path = path / datatype / filename

    return path


def write_perfusion_map(
    data: NDArray[np.floating[Any]],
    output_dir: Union[str, Path],
    subject: str,
    map_type: str,
    reference_image: Union[str, Path],
    session: Optional[str] = None,
    space: str = "native",
    desc: Optional[str] = None,
    model: Optional[str] = None,
    sources: Optional[List[Union[str, Path]]] = None,
    config_used: Optional[Dict[str, Any]] = None,
    pipeline: str = "pmriprep",
    **additional_metadata: Any,
) -> Path:
    """Convenience function to write a perfusion parameter map.

    This handles the common case of writing CBF, CBV, Ktrans, etc. maps
    with appropriate BIDS naming and metadata.

    Args:
        data: NumPy array containing the map data.
        output_dir: Derivatives directory.
        subject: Subject ID (without 'sub-' prefix).
        map_type: Map type key (e.g., 'cbf', 'ktrans'). Must be in PERFUSION_SUFFIXES.
        reference_image: Reference NIfTI for affine and header.
        session: Session ID (without 'ses-' prefix), optional.
        space: Coordinate space (default: 'native').
        desc: Description entity (e.g., 'leakcorr', 'uncorrected').
        model: Model used (e.g., 'sSVD', 'tofts').
        sources: Source files used to generate this output.
        config_used: Configuration dictionary.
        pipeline: Pipeline name (default: 'pmriprep').
        **additional_metadata: Additional metadata for JSON sidecar.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If map_type is not recognized.
    """
    map_type_lower = map_type.lower()
    if map_type_lower not in PERFUSION_SUFFIXES:
        raise ValueError(
            f"Unknown map type: {map_type}. "
            f"Valid types: {list(PERFUSION_SUFFIXES.keys())}"
        )

    map_info = PERFUSION_SUFFIXES[map_type_lower]
    suffix = map_info["suffix"]
    units = map_info.get("units")
    map_description = map_info.get("description", "")

    # Determine datatype based on map
    if map_type_lower in ["t1", "b1"]:
        datatype = "anat"
    elif map_type_lower in ["brain_mask", "lesion_mask", "tumor_seg", "wm_probseg", "gm_probseg", "csf_probseg"]:
        datatype = "anat"
    else:
        datatype = "perf"

    # Generate filename
    filename = get_derivative_filename(
        subject=subject,
        suffix=suffix,
        session=session,
        space=space,
        desc=desc,
    )

    # Build output path
    output_path = get_derivative_path(
        derivatives_dir=output_dir,
        subject=subject,
        pipeline=pipeline,
        datatype=datatype,
        filename=filename,
        session=session,
    )

    # Build metadata
    metadata = DerivativeMetadata(
        description=map_description,
        sources=sources,
        processing_description=f"Generated by pmriprep {model or ''} pipeline",
        units=units,
        parameter_name=map_type_lower.upper(),
        model=model,
        space=space,
        additional=additional_metadata if additional_metadata else None,
    )

    # Write the derivative
    return write_derivative(
        data=data,
        output_path=output_path,
        reference_image=reference_image,
        metadata=metadata,
        config_used=config_used,
        include_provenance=True,
    )


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
        "GeneratedBy": {
            "Name": "pmriprep",
            "Version": pmriprep.__version__,
        },
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
