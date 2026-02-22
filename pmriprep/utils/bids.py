"""BIDS path navigation and file discovery utilities for PMRIPrep."""

from pathlib import Path
from typing import Dict, List, Optional, Union


def get_anat_filename(
    subject: str,
    session: Optional[str] = None,
    suffix: str = "T1w",
    desc: Optional[str] = None,
    space: Optional[str] = None,
    extension: str = ".nii.gz",
    **entities: str,
) -> str:
    """Generate a BIDS-compliant anatomical filename.

    Args:
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.
        suffix: BIDS suffix (e.g., 'T1w', 'T2w', 'FLAIR').
        desc: Description entity (e.g., 'preproc', 'brain').
        space: Space entity (e.g., 'MNI152NLin2009cAsym', 'anat').
        extension: File extension (default: '.nii.gz').
        **entities: Additional BIDS entities as key-value pairs.

    Returns:
        BIDS-compliant filename string.
    """
    parts: List[str] = []

    # Required entities
    parts.append(f"sub-{subject}")

    if session:
        parts.append(f"ses-{session}")

    # Optional entities in BIDS order
    entity_order = ["task", "acq", "ce", "rec", "run", "echo", "part", "space", "res", "desc"]

    # Add space and desc if provided
    all_entities: Dict[str, str] = dict(entities)
    if space:
        all_entities["space"] = space
    if desc:
        all_entities["desc"] = desc

    for key in entity_order:
        if key in all_entities:
            parts.append(f"{key}-{all_entities[key]}")

    # Add any remaining entities not in the standard order
    for key, value in all_entities.items():
        if key not in entity_order:
            parts.append(f"{key}-{value}")

    filename = "_".join(parts) + f"_{suffix}{extension}"
    return filename


def get_dsc_filename(
    subject: str,
    session: Optional[str] = None,
    suffix: str = "dseg",
    desc: Optional[str] = None,
    space: Optional[str] = None,
    map_type: Optional[str] = None,
    extension: str = ".nii.gz",
    **entities: str,
) -> str:
    """Generate a BIDS-compliant DSC perfusion output filename.

    Args:
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.
        suffix: BIDS suffix (e.g., 'dseg', 'cbf', 'cbv', 'mtt').
        desc: Description entity (e.g., 'preproc', 'leakcorr').
        space: Space entity (e.g., 'orig', 'anat').
        map_type: Map type for perfusion outputs.
        extension: File extension (default: '.nii.gz').
        **entities: Additional BIDS entities.

    Returns:
        BIDS-compliant filename string.
    """
    parts: List[str] = []

    parts.append(f"sub-{subject}")

    if session:
        parts.append(f"ses-{session}")

    # DSC-specific entity order
    entity_order = ["acq", "rec", "run", "space", "res", "desc"]

    all_entities: Dict[str, str] = dict(entities)
    if space:
        all_entities["space"] = space
    if desc:
        all_entities["desc"] = desc
    if map_type:
        all_entities["map"] = map_type

    for key in entity_order:
        if key in all_entities:
            parts.append(f"{key}-{all_entities[key]}")

    for key, value in all_entities.items():
        if key not in entity_order and key != "map":
            parts.append(f"{key}-{value}")

    # Map type goes before suffix for perfusion maps
    if "map" in all_entities:
        parts.append(f"map-{all_entities['map']}")

    filename = "_".join(parts) + f"_{suffix}{extension}"
    return filename


def get_dce_filename(
    subject: str,
    session: Optional[str] = None,
    suffix: str = "dseg",
    desc: Optional[str] = None,
    space: Optional[str] = None,
    model: Optional[str] = None,
    extension: str = ".nii.gz",
    **entities: str,
) -> str:
    """Generate a BIDS-compliant DCE perfusion output filename.

    Args:
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.
        suffix: BIDS suffix (e.g., 'dseg', 'Ktrans', 've', 'vp').
        desc: Description entity (e.g., 'tofts', 'patlak').
        space: Space entity (e.g., 'orig', 'anat').
        model: PK model used (e.g., 'tofts', 'etofts', 'patlak', '2cxm').
        extension: File extension (default: '.nii.gz').
        **entities: Additional BIDS entities.

    Returns:
        BIDS-compliant filename string.
    """
    parts: List[str] = []

    parts.append(f"sub-{subject}")

    if session:
        parts.append(f"ses-{session}")

    # DCE-specific entity order
    entity_order = ["acq", "rec", "run", "space", "res", "model", "desc"]

    all_entities: Dict[str, str] = dict(entities)
    if space:
        all_entities["space"] = space
    if desc:
        all_entities["desc"] = desc
    if model:
        all_entities["model"] = model

    for key in entity_order:
        if key in all_entities:
            parts.append(f"{key}-{all_entities[key]}")

    for key, value in all_entities.items():
        if key not in entity_order:
            parts.append(f"{key}-{value}")

    filename = "_".join(parts) + f"_{suffix}{extension}"
    return filename


def find_anat_dir(
    bids_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
) -> Optional[Path]:
    """Find the anatomical directory for a subject/session.

    Args:
        bids_dir: Path to the BIDS dataset root.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.

    Returns:
        Path to the anat directory, or None if not found.
    """
    bids_path = Path(bids_dir)

    if session:
        anat_dir = bids_path / f"sub-{subject}" / f"ses-{session}" / "anat"
    else:
        anat_dir = bids_path / f"sub-{subject}" / "anat"

    if anat_dir.is_dir():
        return anat_dir

    return None


def find_dsc_dir(
    bids_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
) -> Optional[Path]:
    """Find the DSC perfusion directory for a subject/session.

    Looks for 'perf' directory which contains DSC data in BIDS.

    Args:
        bids_dir: Path to the BIDS dataset root.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.

    Returns:
        Path to the perf directory, or None if not found.
    """
    bids_path = Path(bids_dir)

    if session:
        perf_dir = bids_path / f"sub-{subject}" / f"ses-{session}" / "perf"
    else:
        perf_dir = bids_path / f"sub-{subject}" / "perf"

    if perf_dir.is_dir():
        return perf_dir

    return None


def find_dce_dir(
    bids_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
) -> Optional[Path]:
    """Find the DCE perfusion directory for a subject/session.

    DCE data may be in 'perf' or sometimes in a separate directory.
    This function checks common locations.

    Args:
        bids_dir: Path to the BIDS dataset root.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.

    Returns:
        Path to the directory containing DCE data, or None if not found.
    """
    bids_path = Path(bids_dir)

    if session:
        base = bids_path / f"sub-{subject}" / f"ses-{session}"
    else:
        base = bids_path / f"sub-{subject}"

    # Check perf directory first (standard BIDS location)
    perf_dir = base / "perf"
    if perf_dir.is_dir():
        # Look for DCE files (typically have 'dce' or 'DCE' in the filename)
        dce_files = list(perf_dir.glob("*dce*")) + list(perf_dir.glob("*DCE*"))
        if dce_files:
            return perf_dir

    # Check for a dedicated dce directory (non-standard but sometimes used)
    dce_dir = base / "dce"
    if dce_dir.is_dir():
        return dce_dir

    return None


def find_t1w_reference(
    bids_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
    prefer_postcon: bool = False,
) -> Optional[Path]:
    """Find the best T1w reference image for a subject/session.

    Smart fallback logic:
    1. If prefer_postcon=True, look for contrast-enhanced T1w (ce-GAD, ce-Gd)
    2. Otherwise prefer native T1w without contrast
    3. Fallback to any available T1w

    Args:
        bids_dir: Path to the BIDS dataset root.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.
        prefer_postcon: Whether to prefer post-contrast T1w (default: False).

    Returns:
        Path to the best T1w reference image, or None if not found.
    """
    anat_dir = find_anat_dir(bids_dir, subject, session)
    if not anat_dir:
        return None

    # Find all T1w files
    t1w_files = list(anat_dir.glob("*_T1w.nii.gz")) + list(anat_dir.glob("*_T1w.nii"))

    if not t1w_files:
        return None

    # Separate post-contrast and native T1w
    postcon_files: List[Path] = []
    native_files: List[Path] = []

    for f in t1w_files:
        fname = f.name.lower()
        if "ce-" in fname or "ce_" in fname or "gad" in fname:
            postcon_files.append(f)
        else:
            native_files.append(f)

    if prefer_postcon:
        # Prefer post-contrast, fallback to native
        candidates = postcon_files if postcon_files else native_files
    else:
        # Prefer native, fallback to post-contrast
        candidates = native_files if native_files else postcon_files

    if not candidates:
        # If neither found, return first available
        return t1w_files[0]

    # If multiple candidates, prefer the one without extra acquisitions
    for f in candidates:
        if "acq-" not in f.name:
            return f

    return candidates[0]


def find_brain_mask(
    derivatives_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
    space: Optional[str] = None,
) -> Optional[Path]:
    """Find a brain mask for a subject/session in derivatives.

    Searches in common locations within derivatives directory.

    Args:
        derivatives_dir: Path to the derivatives directory.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.
        space: Space to look for (e.g., 'anat', 'MNI152NLin2009cAsym').

    Returns:
        Path to the brain mask, or None if not found.
    """
    deriv_path = Path(derivatives_dir)

    # Build search patterns
    patterns: List[str] = []

    base_pattern = f"sub-{subject}"
    if session:
        base_pattern += f"_ses-{session}"

    if space:
        patterns.extend([
            f"{base_pattern}*space-{space}*_mask.nii.gz",
            f"{base_pattern}*space-{space}*_brain_mask.nii.gz",
            f"{base_pattern}*space-{space}*desc-brain*_mask.nii.gz",
        ])

    # Generic patterns without specific space
    patterns.extend([
        f"{base_pattern}*_mask.nii.gz",
        f"{base_pattern}*_brain_mask.nii.gz",
        f"{base_pattern}*desc-brain*_mask.nii.gz",
    ])

    # Search in derivatives
    for pattern in patterns:
        matches = list(deriv_path.rglob(pattern))
        if matches:
            # Prefer files with 'brain' in the name
            for m in matches:
                if "brain" in m.name.lower():
                    return m
            return matches[0]

    return None


def get_bids_entities(filename: Union[str, Path]) -> Dict[str, str]:
    """Extract BIDS entities from a filename.

    Args:
        filename: BIDS-compliant filename or path.

    Returns:
        Dictionary of entity-value pairs.
    """
    name = Path(filename).name

    # Remove extension(s)
    while "." in name:
        name = name.rsplit(".", 1)[0]

    entities: Dict[str, str] = {}

    # Split by underscore and extract entities
    parts = name.split("_")

    for part in parts[:-1]:  # Last part is the suffix
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value

    # Add suffix
    if parts:
        entities["suffix"] = parts[-1]

    return entities


def build_bids_path(
    root: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
    datatype: str = "anat",
    filename: Optional[str] = None,
) -> Path:
    """Build a complete BIDS path from components.

    Args:
        root: BIDS root directory.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.
        datatype: Data type directory (e.g., 'anat', 'perf', 'func').
        filename: Optional filename to append.

    Returns:
        Complete path to the BIDS location.
    """
    path = Path(root) / f"sub-{subject}"

    if session:
        path = path / f"ses-{session}"

    path = path / datatype

    if filename:
        path = path / filename

    return path
