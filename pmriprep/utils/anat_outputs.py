"""Helper functions for BIDS-compliant anatomical derivative output.

This module provides functions for building JSON sidecars, computing
resolution metadata, and generating BIDS-compliant filenames for
anatomical pipeline outputs.
"""

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def compute_original_resolution(source_nifti: Union[str, Path]) -> Dict[str, Any]:
    """Read NIfTI header and compute resolution metadata.

    Parameters
    ----------
    source_nifti : str or Path
        Path to the source NIfTI file.

    Returns
    -------
    dict
        Dictionary with keys:
        - VoxelDimensions: list of 3 voxel sizes in mm
        - GeometricMean: geometric mean of voxel dimensions
        - IsotropyRatio: max/min voxel dimension ratio
        - ResolutionTier: "excellent" (<= 1.5mm), "good" (<= 3mm), "moderate" (> 3mm)
    """
    import nibabel as nib  # type: ignore[import-untyped]

    img = nib.load(str(source_nifti))
    zooms = list(img.header.get_zooms()[:3])  # type: ignore[attr-defined]

    voxel_dims = [float(z) for z in zooms]
    geo_mean = round(float(math.prod(voxel_dims) ** (1.0 / 3.0)), 3)
    iso_ratio = round(max(voxel_dims) / max(min(voxel_dims), 1e-9), 2)

    if geo_mean <= 1.5:
        tier = "excellent"
    elif geo_mean <= 3.0:
        tier = "good"
    else:
        tier = "moderate"

    return {
        "VoxelDimensions": voxel_dims,
        "GeometricMean": geo_mean,
        "IsotropyRatio": iso_ratio,
        "ResolutionTier": tier,
    }


def make_bids_source_uri(
    source_path: Union[str, Path], bids_dir: Union[str, Path]
) -> str:
    """Convert an absolute path to a BIDS source URI.

    Parameters
    ----------
    source_path : str or Path
        Absolute path to the source file.
    bids_dir : str or Path
        Absolute path to the BIDS dataset root (rawdata directory).

    Returns
    -------
    str
        BIDS URI like ``bids::rawdata/sub-XX/ses-XX/...``
    """
    source_path = Path(source_path).resolve()
    bids_dir = Path(bids_dir).resolve()

    # bids_dir may be the rawdata dir itself, or its parent
    # Try to make relative to bids_dir parent first (to include "rawdata/")
    try:
        rel = source_path.relative_to(bids_dir.parent)
        return f"bids::{rel}"
    except ValueError:
        pass

    # If bids_dir IS the rawdata directory
    try:
        rel = source_path.relative_to(bids_dir)
        return f"bids::rawdata/{rel}"
    except ValueError:
        # Fallback: just use the filename
        return f"bids::{source_path.name}"


def get_fsl_version() -> str:
    """Get FSL version string."""
    try:
        fsldir = os.environ.get("FSLDIR", "")
        # Try reading the version file
        version_file = Path(fsldir) / "etc" / "fslversion"
        if version_file.exists():
            version = version_file.read_text().strip()
            return f"fsl-{version}"
        # Try flirt -version as fallback
        result = subprocess.run(
            ["flirt", "-version"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if line.strip():
                return line.strip()
    except Exception:
        pass
    return "unknown"


def get_ants_version() -> str:
    """Get ANTs version string."""
    try:
        result = subprocess.run(
            ["N4BiasFieldCorrection", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if "ANTs" in line and "Version" in line:
                return line.strip()
            if "ANTs" in line:
                return line.strip()
    except Exception:
        pass
    return "unknown"


def get_imaging_software_versions() -> Dict[str, str]:
    """Return version dict for FSL, ANTs, and HD-BET.

    Returns
    -------
    dict
        Keys: FSL, ANTs, HD-BET with version strings.
    """
    return {
        "FSL": get_fsl_version(),
        "ANTs": get_ants_version(),
        "HD-BET": "unknown",
    }


def build_preproc_sidecar(
    description: str,
    source_path: Union[str, Path],
    bids_dir: Union[str, Path],
    skull_stripped: bool = True,
    contrast_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a JSON sidecar dict for a preprocessed anatomical image.

    Parameters
    ----------
    description : str
        Human-readable description of the image.
    source_path : str or Path
        Path to the original source NIfTI.
    bids_dir : str or Path
        BIDS dataset root directory.
    skull_stripped : bool
        Whether the image is skull-stripped.
    contrast_agent : str, optional
        Generic contrast agent name for post-contrast images.

    Returns
    -------
    dict
        JSON-serializable sidecar dictionary.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    source_uri = make_bids_source_uri(source_path, bids_dir)
    resolution = compute_original_resolution(source_path)

    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": description,
        "SkullStripped": skull_stripped,
        "SpatialReference": "anat",
        "Sources": [source_uri],
        "OriginalResolution": resolution,
        "ProcessingSteps": [
            "Reorientation to RAS",
            "Robust FOV cropping",
            "N4 bias field correction",
            "Coregistration to native anat space",
            "HD-BET brain extraction",
        ],
        "SoftwareVersions": get_imaging_software_versions(),
    }

    if contrast_agent is not None:
        from pmriprep.utils.contrast_agent import get_brand_name
        agent_block: Dict[str, str] = {"GenericName": contrast_agent}
        brand = get_brand_name(contrast_agent)
        if brand:
            agent_block["BrandName"] = brand
        sidecar["PerfusionProcessing"] = {"ContrastAgent": agent_block}

    return sidecar


def build_probseg_sidecar(
    tissue_type: str,
    source_path: Union[str, Path],
    bids_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Build a JSON sidecar dict for a tissue probability map.

    Parameters
    ----------
    tissue_type : str
        Tissue type: "WM", "GM", or "CSF".
    source_path : str or Path
        Path to the original source NIfTI (T1w pre-contrast).
    bids_dir : str or Path
        BIDS dataset root directory.

    Returns
    -------
    dict
        JSON-serializable sidecar dictionary.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    source_uri = make_bids_source_uri(source_path, bids_dir)
    fsl_version = get_fsl_version()

    return {
        **get_bids_derivative_provenance(),
        "Description": f"{tissue_type} probability map from FAST tissue segmentation",
        "Sources": [source_uri],
        "ReferenceContrastType": "precon",
        "SoftwareVersions": {
            "FSL": fsl_version,
            "FAST": "part of FSL",
        },
        "ProcessingSteps": [
            "FAST 3-class tissue segmentation (T1-weighted)",
            "Parameters: -t 1 -n 3 -H 0.1 -I 4 -l 20.0",
        ],
        "TissueType": tissue_type,
        "SegmentationMethod": "FAST",
        "NumberOfClasses": 3,
        "ProbabilityMap": True,
    }


def build_mask_sidecar(
    description: str,
    source_path: Union[str, Path],
    bids_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Build a JSON sidecar dict for a brain mask.

    Parameters
    ----------
    description : str
        Human-readable description of the mask.
    source_path : str or Path
        Path to the original source NIfTI.
    bids_dir : str or Path
        BIDS dataset root directory.

    Returns
    -------
    dict
        JSON-serializable sidecar dictionary.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    source_uri = make_bids_source_uri(source_path, bids_dir)
    resolution = compute_original_resolution(source_path)

    return {
        **get_bids_derivative_provenance(),
        "Description": description,
        "SkullStripped": False,
        "SpatialReference": "anat",
        "Sources": [source_uri],
        "OriginalResolution": resolution,
        "ProcessingSteps": [
            "Reorientation to RAS",
            "Robust FOV cropping",
            "N4 bias field correction",
            "Coregistration to native anat space",
            "HD-BET brain extraction",
        ],
        "SoftwareVersions": get_imaging_software_versions(),
    }


def get_xfm_filename(
    subject: str,
    session: str,
    from_space: str,
    to_space: str,
) -> str:
    """Generate BIDS-compliant transform filename.

    Parameters
    ----------
    subject : str
        Subject ID (without 'sub-' prefix).
    session : str
        Session ID (without 'ses-' prefix).
    from_space : str
        Source space (e.g., "T1wCE", "T2w", "FLAIR").
    to_space : str
        Target space (e.g., "anat", "ACPC").

    Returns
    -------
    str
        Filename like ``sub-XX_ses-XX_from-X_to-Y_mode-image_xfm.mat``.
    """
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append(f"from-{from_space}")
    parts.append(f"to-{to_space}")
    parts.append("mode-image")
    return "_".join(parts) + "_xfm.mat"
