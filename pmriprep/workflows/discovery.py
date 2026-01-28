"""Data discovery logic for automatic detection of available modalities.

This module implements automatic detection of available imaging modalities
in a BIDS dataset and determines which pipeline components can run.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class ModalityAvailability:
    """Tracks availability of different imaging modalities.

    Attributes:
        t1w_pre: Path to pre-contrast T1w image, if available.
        t1w_ce_gad: Path to contrast-enhanced T1w (GAD) image, if available.
        t2w: Path to T2w image, if available.
        flair: Path to FLAIR image, if available.
        dsc: Path to DSC perfusion image, if available.
        dce: Path to DCE perfusion image, if available.
        vfa: List of paths to VFA (Variable Flip Angle) images for T1 mapping.
        afi: List of paths to AFI (Actual Flip Angle Imaging) images for B1+ mapping.
        dwi: Path to DWI image, if available.
    """

    t1w_pre: Optional[Path] = None
    t1w_ce_gad: Optional[Path] = None
    t2w: Optional[Path] = None
    flair: Optional[Path] = None
    pdw: Optional[Path] = None
    dsc: Optional[Path] = None
    dce: Optional[Path] = None
    vfa: List[Path] = field(default_factory=list)
    afi: List[Path] = field(default_factory=list)
    dwi: Optional[Path] = None


@dataclass
class PipelineAvailability:
    """Tracks which pipelines can run based on available data.

    Attributes:
        anat: Whether anatomical pipeline can run (always True).
        hdglio: Whether HD-GLIO tumor segmentation can run.
        dsc: Whether DSC perfusion pipeline can run.
        dce: Whether DCE perfusion pipeline can run.
        hdglio_reason: Reason why HD-GLIO cannot run, if applicable.
        dsc_reason: Reason why DSC cannot run, if applicable.
        dce_reason: Reason why DCE cannot run, if applicable.
    """

    anat: bool = True
    hdglio: bool = False
    dsc: bool = False
    dce: bool = False
    hdglio_reason: Optional[str] = None
    dsc_reason: Optional[str] = None
    dce_reason: Optional[str] = None


@dataclass
class SessionData:
    """Complete data discovery result for a session.

    Attributes:
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), if applicable.
        modalities: Available modalities.
        pipelines: Available pipelines.
    """

    subject: str
    session: Optional[str]
    modalities: ModalityAvailability
    pipelines: PipelineAvailability


def _find_t1w_precontrast(anat_dir: Path) -> Optional[Path]:
    """Find pre-contrast T1w image.

    Pre-contrast T1w is identified by:
    - Having suffix _T1w
    - NOT having 'ce-' entity (contrast enhancement)

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to pre-contrast T1w, or None if not found.
    """
    t1w_files = list(anat_dir.glob("*_T1w.nii.gz")) + list(anat_dir.glob("*_T1w.nii"))

    for f in t1w_files:
        fname = f.name.lower()
        # Exclude contrast-enhanced images
        if "ce-" not in fname and "ce_" not in fname:
            return f

    return None


def _find_t1w_ce_gad(anat_dir: Path) -> Optional[Path]:
    """Find contrast-enhanced T1w (post-gadolinium) image.

    Contrast-enhanced T1w is identified by:
    - Having suffix _T1w
    - Having 'ce-' entity (e.g., ce-GAD, ce-Gd, ce-GADOLINIUM)

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to contrast-enhanced T1w, or None if not found.
    """
    t1w_files = list(anat_dir.glob("*_T1w.nii.gz")) + list(anat_dir.glob("*_T1w.nii"))

    for f in t1w_files:
        fname = f.name.lower()
        # Look for contrast-enhanced images
        if "ce-" in fname or "ce_" in fname:
            # Prefer GAD-specific
            if "gad" in fname:
                return f

    # Second pass: any ce- T1w
    for f in t1w_files:
        fname = f.name.lower()
        if "ce-" in fname or "ce_" in fname:
            return f

    return None


def _find_t2w(anat_dir: Path) -> Optional[Path]:
    """Find T2w image.

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to T2w image, or None if not found.
    """
    t2w_files = list(anat_dir.glob("*_T2w.nii.gz")) + list(anat_dir.glob("*_T2w.nii"))

    if t2w_files:
        # Prefer non-contrast T2w
        for f in t2w_files:
            fname = f.name.lower()
            if "ce-" not in fname:
                return f
        return t2w_files[0]

    return None


def _find_flair(anat_dir: Path) -> Optional[Path]:
    """Find FLAIR image.

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to FLAIR image, or None if not found.
    """
    flair_files = (
        list(anat_dir.glob("*_FLAIR.nii.gz"))
        + list(anat_dir.glob("*_FLAIR.nii"))
        + list(anat_dir.glob("*_flair.nii.gz"))
        + list(anat_dir.glob("*_flair.nii"))
    )

    if flair_files:
        return flair_files[0]

    return None


def _find_pdw(anat_dir: Path) -> Optional[Path]:
    """Find PDw (proton density-weighted) image.

    Prefers non-contrast PDw images.

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to PDw image, or None if not found.
    """
    pdw_files = list(anat_dir.glob("*_PDw.nii.gz")) + list(anat_dir.glob("*_PDw.nii"))

    if pdw_files:
        # Prefer non-contrast PDw
        for f in pdw_files:
            if "ce-" not in f.name.lower():
                return f
        return pdw_files[0]

    return None


def _find_dsc(perf_dir: Path) -> Optional[Path]:
    """Find DSC perfusion image.

    DSC is identified by:
    - Being in perf directory
    - Having 'dsc' or 'asl' in filename, or having _bold suffix with perfusion context

    Args:
        perf_dir: Path to the perfusion directory.

    Returns:
        Path to DSC image, or None if not found.
    """
    # Look for explicit DSC files
    dsc_patterns = ["*dsc*.nii.gz", "*dsc*.nii", "*DSC*.nii.gz", "*DSC*.nii"]
    for pattern in dsc_patterns:
        matches = list(perf_dir.glob(pattern))
        if matches:
            # Prefer 4D files
            for m in matches:
                if "_bold" in m.name.lower() or "perf" in m.name.lower():
                    return m
            return matches[0]

    # Look for perfusion BOLD (common DSC naming)
    bold_patterns = ["*_bold.nii.gz", "*_bold.nii"]
    for pattern in bold_patterns:
        matches = list(perf_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def _find_dce(perf_dir: Path, anat_dir: Optional[Path] = None) -> Optional[Path]:
    """Find DCE perfusion image.

    DCE is identified by:
    - Having 'dce' in filename
    - Being a dynamic contrast-enhanced sequence

    Args:
        perf_dir: Path to the perfusion directory.
        anat_dir: Path to anatomical directory (DCE sometimes stored there).

    Returns:
        Path to DCE image, or None if not found.
    """
    # Search in perf directory
    dce_patterns = ["*dce*.nii.gz", "*dce*.nii", "*DCE*.nii.gz", "*DCE*.nii"]
    for pattern in dce_patterns:
        matches = list(perf_dir.glob(pattern))
        if matches:
            return matches[0]

    # Also check anat_dir if provided (DCE sometimes stored there)
    if anat_dir:
        for pattern in dce_patterns:
            matches = list(anat_dir.glob(pattern))
            if matches:
                return matches[0]

    return None


def _find_vfa(anat_dir: Path, perf_dir: Optional[Path] = None) -> List[Path]:
    """Find VFA (Variable Flip Angle) images for T1 mapping.

    VFA images are identified by:
    - Having 'vfa' or 'VFA' in filename
    - Having multiple flip angles

    Args:
        anat_dir: Path to the anatomical directory.
        perf_dir: Path to the perfusion directory (VFA sometimes stored there).

    Returns:
        List of paths to VFA images.
    """
    vfa_files: List[Path] = []

    # Search patterns for VFA
    vfa_patterns = ["*VFA*.nii.gz", "*VFA*.nii", "*vfa*.nii.gz", "*vfa*.nii"]

    for pattern in vfa_patterns:
        vfa_files.extend(anat_dir.glob(pattern))

    if perf_dir:
        for pattern in vfa_patterns:
            vfa_files.extend(perf_dir.glob(pattern))

    # Remove duplicates while preserving order
    seen = set()
    unique_vfa: List[Path] = []
    for f in vfa_files:
        if f not in seen:
            seen.add(f)
            unique_vfa.append(f)

    return sorted(unique_vfa)


def _find_afi(anat_dir: Path, perf_dir: Optional[Path] = None) -> List[Path]:
    """Find AFI (Actual Flip Angle Imaging) images for B1+ mapping.

    AFI images are identified by:
    - Having 'afi' or 'AFI' or 'b1' or 'B1' in filename
    - Typically pairs of images with different TRs

    Args:
        anat_dir: Path to the anatomical directory.
        perf_dir: Path to the perfusion directory (AFI sometimes stored there).

    Returns:
        List of paths to AFI images.
    """
    afi_files: List[Path] = []

    # Search patterns for AFI and B1 mapping
    afi_patterns = [
        "*AFI*.nii.gz",
        "*AFI*.nii",
        "*afi*.nii.gz",
        "*afi*.nii",
        "*B1*.nii.gz",
        "*B1*.nii",
        "*b1map*.nii.gz",
        "*b1map*.nii",
    ]

    for pattern in afi_patterns:
        afi_files.extend(anat_dir.glob(pattern))

    if perf_dir:
        for pattern in afi_patterns:
            afi_files.extend(perf_dir.glob(pattern))

    # Remove duplicates while preserving order
    seen = set()
    unique_afi: List[Path] = []
    for f in afi_files:
        if f not in seen:
            seen.add(f)
            unique_afi.append(f)

    return sorted(unique_afi)


def _find_dwi(bids_dir: Path, subject: str, session: Optional[str]) -> Optional[Path]:
    """Find DWI image.

    Args:
        bids_dir: BIDS root directory.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.

    Returns:
        Path to DWI image, or None if not found.
    """
    if session:
        dwi_dir = bids_dir / f"sub-{subject}" / f"ses-{session}" / "dwi"
    else:
        dwi_dir = bids_dir / f"sub-{subject}" / "dwi"

    if not dwi_dir.is_dir():
        return None

    dwi_patterns = ["*_dwi.nii.gz", "*_dwi.nii"]
    for pattern in dwi_patterns:
        matches = list(dwi_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def _determine_pipeline_availability(modalities: ModalityAvailability) -> PipelineAvailability:
    """Determine which pipelines can run based on available modalities.

    Rules:
    - anat: Always runs (T1w is minimum requirement but assumed present)
    - hdglio: Requires all 4 modalities (T1w-pre, T1w-ce-GAD, T2w, FLAIR)
    - dsc: Requires DSC data
    - dce: Requires DCE data

    Args:
        modalities: Available modalities.

    Returns:
        Pipeline availability with reasons for unavailable pipelines.
    """
    pipelines = PipelineAvailability()

    # HD-GLIO requires 4 modalities
    hdglio_missing: List[str] = []
    if not modalities.t1w_pre:
        hdglio_missing.append("T1w-pre")
    if not modalities.t1w_ce_gad:
        hdglio_missing.append("T1w-ce-GAD")
    if not modalities.t2w:
        hdglio_missing.append("T2w")
    if not modalities.flair:
        hdglio_missing.append("FLAIR")

    if not hdglio_missing:
        pipelines.hdglio = True
    else:
        pipelines.hdglio = False
        pipelines.hdglio_reason = f"Missing modalities: {', '.join(hdglio_missing)}"

    # DSC requires DSC data
    if modalities.dsc:
        pipelines.dsc = True
    else:
        pipelines.dsc = False
        pipelines.dsc_reason = "No DSC perfusion data found"

    # DCE requires DCE data
    if modalities.dce:
        pipelines.dce = True
    else:
        pipelines.dce = False
        pipelines.dce_reason = "No DCE perfusion data found"

    return pipelines


def discover_session_data(
    bids_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
) -> SessionData:
    """Discover available imaging modalities for a subject/session.

    This function scans the BIDS dataset to find available imaging modalities
    and determines which pipeline components can run.

    Args:
        bids_dir: Path to the BIDS dataset root.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.

    Returns:
        SessionData containing available modalities and runnable pipelines.
    """
    bids_path = Path(bids_dir)
    modalities = ModalityAvailability()

    # Build base path
    if session:
        base_path = bids_path / f"sub-{subject}" / f"ses-{session}"
    else:
        base_path = bids_path / f"sub-{subject}"

    # Find anatomical directory
    anat_dir = base_path / "anat"
    if anat_dir.is_dir():
        modalities.t1w_pre = _find_t1w_precontrast(anat_dir)
        modalities.t1w_ce_gad = _find_t1w_ce_gad(anat_dir)
        modalities.t2w = _find_t2w(anat_dir)
        modalities.flair = _find_flair(anat_dir)
        modalities.pdw = _find_pdw(anat_dir)
        modalities.vfa = _find_vfa(anat_dir)
        modalities.afi = _find_afi(anat_dir)

    # Find perfusion directory
    perf_dir = base_path / "perf"
    if perf_dir.is_dir():
        modalities.dsc = _find_dsc(perf_dir)
        modalities.dce = _find_dce(perf_dir, anat_dir if anat_dir.is_dir() else None)
        # Also check perf for VFA/AFI
        if anat_dir.is_dir():
            modalities.vfa = _find_vfa(anat_dir, perf_dir)
            modalities.afi = _find_afi(anat_dir, perf_dir)

    # Find DWI
    modalities.dwi = _find_dwi(bids_path, subject, session)

    # Determine pipeline availability
    pipelines = _determine_pipeline_availability(modalities)

    return SessionData(
        subject=subject,
        session=session,
        modalities=modalities,
        pipelines=pipelines,
    )


def get_data_summary(session_data: SessionData) -> Dict[str, Union[bool, str, None]]:
    """Get a summary dictionary of available data and pipelines.

    Args:
        session_data: Session data from discover_session_data().

    Returns:
        Dictionary with boolean flags for each modality and pipeline.
    """
    return {
        # Modalities
        "has_t1w_pre": session_data.modalities.t1w_pre is not None,
        "has_t1w_ce_gad": session_data.modalities.t1w_ce_gad is not None,
        "has_t2w": session_data.modalities.t2w is not None,
        "has_flair": session_data.modalities.flair is not None,
        "has_pdw": session_data.modalities.pdw is not None,
        "has_dsc": session_data.modalities.dsc is not None,
        "has_dce": session_data.modalities.dce is not None,
        "has_vfa": len(session_data.modalities.vfa) > 0,
        "has_afi": len(session_data.modalities.afi) > 0,
        "has_dwi": session_data.modalities.dwi is not None,
        # Pipelines
        "can_run_anat": session_data.pipelines.anat,
        "can_run_hdglio": session_data.pipelines.hdglio,
        "can_run_dsc": session_data.pipelines.dsc,
        "can_run_dce": session_data.pipelines.dce,
        # Reasons for unavailable pipelines
        "hdglio_unavailable_reason": session_data.pipelines.hdglio_reason,
        "dsc_unavailable_reason": session_data.pipelines.dsc_reason,
        "dce_unavailable_reason": session_data.pipelines.dce_reason,
    }
