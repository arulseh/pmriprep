"""Data discovery logic for automatic detection of available modalities.

This module implements automatic detection of available imaging modalities
in a BIDS dataset and determines which pipeline components can run.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    from bids.layout import BIDSLayout

logger = logging.getLogger("pmriprep.workflows.discovery")


@dataclass
class ModalityAvailability:
    """Tracks availability of different imaging modalities.

    Attributes:
        t1w_pre: Path to pre-contrast T1w image, if available.
        t1w_ce_gad: Path to contrast-enhanced T1w (GAD) image, if available.
        t2w: Path to T2w image, if available.
        flair: Path to pre-contrast FLAIR image, if available.
        flair_postcon: Path to post-contrast FLAIR image, if available.
        dsc: Path to DSC perfusion image, if available.
        dce: Path to DCE perfusion image, if available.
        vfa: List of paths to VFA (Variable Flip Angle) images for T1 mapping.
        afi: List of paths to AFI (Actual Flip Angle Imaging) images for B1+ mapping.
        dwi: Path to DWI image, if available.
        b1_precomputed_file: Path to pre-computed B1+ flip angle map (e.g., tfl_b1map famp).
        b1_nominal_prep_fa: Nominal preparation flip angle (degrees) from famp JSON sidecar.
        b1_method_detected: Detected B1+ method ("afi" or "tfl_b1map"), if any.
    """

    t1w_pre: Optional[Path] = None
    t1w_ce_gad: Optional[Path] = None
    t2w: Optional[Path] = None
    flair: Optional[Path] = None
    flair_postcon: Optional[Path] = None
    dsc: Optional[Path] = None
    dce: Optional[Path] = None
    vfa: List[Path] = field(default_factory=list)
    afi: List[Path] = field(default_factory=list)
    dwi: Optional[Path] = None
    afi_mode: Optional[str] = None  # "3d" or "4d"
    afi_4d_file: Optional[Path] = None  # Set when afi_mode="4d"
    afi_tr1_file: Optional[Path] = None  # 3D mode: file with shorter TR
    afi_tr2_file: Optional[Path] = None  # 3D mode: file with longer TR
    afi_tr1_ms: Optional[float] = None  # TR1 in ms (from sidecar)
    afi_tr2_ms: Optional[float] = None  # TR2 in ms (from sidecar)
    afi_flip_angle: Optional[float] = None  # Nominal flip angle in degrees (from sidecar)
    b1_precomputed_file: Optional[Path] = None  # Pre-computed B1+ map (e.g., tfl_b1map famp)
    b1_nominal_prep_fa: Optional[float] = None  # FlipAngle from famp JSON sidecar
    b1_method_detected: Optional[str] = None  # "afi" or "tfl_b1map"


@dataclass
class PipelineAvailability:
    """Tracks which pipelines can run based on available data.

    Attributes:
        anat: Whether anatomical pipeline can run (always True).
        tumor_seg: Whether tumor segmentation can run.
        dsc: Whether DSC perfusion pipeline can run.
        dce: Whether DCE perfusion pipeline can run.
        tumor_seg_reason: Reason why tumor segmentation cannot run, if applicable.
        dsc_reason: Reason why DSC cannot run, if applicable.
        dce_reason: Reason why DCE cannot run, if applicable.
    """

    anat: bool = True
    tumor_seg: bool = False
    dsc: bool = False
    dce: bool = False
    tumor_seg_reason: Optional[str] = None
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
    """Find pre-contrast T1w image with acquisition preference.

    Pre-contrast T1w is identified by:
    - Having suffix _T1w
    - NOT having 'ce-' entity (contrast enhancement)

    Preference order for acquisition type:
    1. MPRAGE (3D GRE with inversion recovery) - best for registration
    2. IR/MP2RAGE (other inversion recovery sequences)
    3. SPGR/GRE (3D gradient echo)
    4. Any non-SE acquisition
    5. SE (spin echo) - 2D with thick slices, worst for registration

    This ordering ensures the best-quality precon T1w is returned when
    multiple exist (e.g., both SE T1w and MPRAGE).

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to pre-contrast T1w, or None if not found.
    """
    t1w_files = list(anat_dir.glob("*_T1w.nii.gz")) + list(anat_dir.glob("*_T1w.nii"))

    # Filter to non-contrast images only
    precon_files = [
        f for f in t1w_files
        if "ce-" not in f.name.lower() and "ce_" not in f.name.lower()
    ]

    if not precon_files:
        return None

    if len(precon_files) == 1:
        return precon_files[0]

    # Multiple precon T1w found — apply acquisition preference
    # Priority: MPRAGE > other 3D IR sequences > SPGR/GRE > any non-SE > SE
    preferred_acq = ["mprage", "ir", "mp2rage", "spgr", "gre"]
    for acq_pref in preferred_acq:
        for f in precon_files:
            if acq_pref in f.name.lower():
                logger.info(
                    "Selected precon T1w (preferred acq=%s): %s",
                    acq_pref, f.name
                )
                return f

    # Avoid SE if possible (2D, thick slices, poor for registration)
    non_se = [f for f in precon_files if "acq-se" not in f.name.lower()]
    if non_se:
        logger.info(
            "Selected precon T1w (non-SE fallback): %s", non_se[0].name
        )
        return non_se[0]

    # Last resort: return first match (may be SE)
    logger.warning(
        "Only SE T1w available for precon; registration may be suboptimal: %s",
        precon_files[0].name
    )
    return precon_files[0]


def _find_t1w_ce_gad(anat_dir: Path) -> Optional[Path]:
    """Find contrast-enhanced T1w (post-gadolinium) image.

    Contrast-enhanced T1w is identified by:
    - Having suffix _T1w
    - Having 'ce-' entity (e.g., ce-GAD, ce-Gd, ce-GADOLINIUM)

    Preference order for acquisition type:
    1. MPRAGE (3D GRE with inversion recovery) - best for registration
    2. IR/MP2RAGE (other inversion recovery sequences)
    3. Any non-SE acquisition
    4. SE (spin echo) - 2D with thick slices, worst for registration

    This ordering prevents incorrect T1w selection when multiple ce-GAD
    images exist (e.g., both SE T1w and MPRAGE), which caused registration
    failures in DCE pipeline due to coordinate space mismatches.

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to contrast-enhanced T1w, or None if not found.
    """
    t1w_files = list(anat_dir.glob("*_T1w.nii.gz")) + list(anat_dir.glob("*_T1w.nii"))

    # Filter to ce-GAD files only
    ce_gad_files = [
        f for f in t1w_files
        if ("ce-" in f.name.lower() or "ce_" in f.name.lower())
        and "gad" in f.name.lower()
    ]

    if not ce_gad_files:
        # Second pass: any ce- T1w (without requiring "gad")
        ce_gad_files = [
            f for f in t1w_files
            if "ce-" in f.name.lower() or "ce_" in f.name.lower()
        ]

    if not ce_gad_files:
        return None

    # Priority: MPRAGE > other 3D IR sequences > any non-SE > SE
    # MPRAGE is 3D with isotropic voxels, ideal for registration
    # SE T1w is typically 2D with thick slices (e.g., 5mm), poor for registration
    preferred_acq = ["mprage", "ir", "mp2rage", "spgr", "gre"]
    for acq_pref in preferred_acq:
        for f in ce_gad_files:
            if acq_pref in f.name.lower():
                logger.info(
                    "Selected ce-GAD T1w (preferred acq=%s): %s",
                    acq_pref, f.name
                )
                return f

    # Avoid SE if possible (2D, thick slices, poor for registration)
    non_se = [f for f in ce_gad_files if "acq-se" not in f.name.lower()]
    if non_se:
        logger.info(
            "Selected ce-GAD T1w (non-SE fallback): %s", non_se[0].name
        )
        return non_se[0]

    # Last resort: return first match (may be SE)
    logger.warning(
        "Only SE T1w available for ce-GAD; registration may be suboptimal: %s",
        ce_gad_files[0].name
    )
    return ce_gad_files[0]


def _classify_orientation(iop: List[float]) -> str:
    """Classify image orientation as axial, coronal, or sagittal.

    Args:
        iop: ImageOrientationPatientDICOM (6 direction cosines).

    Returns:
        One of "axial", "coronal", "sagittal".
    """
    # Normal vector = cross product of row and column direction cosines
    normal = [
        iop[1] * iop[5] - iop[2] * iop[4],
        iop[2] * iop[3] - iop[0] * iop[5],
        iop[0] * iop[4] - iop[1] * iop[3],
    ]
    abs_normal = [abs(n) for n in normal]
    max_idx = abs_normal.index(max(abs_normal))
    return ("sagittal", "coronal", "axial")[max_idx]


def _score_t2w_candidate(
    t2w_path: Path,
    flair_resolved: Optional[Path],
) -> int:
    """Score a T2w candidate for selection preference.

    Scoring criteria (additive):
     -100  same resolved path as FLAIR (duplicate data)
      +10  no IR in ScanningSequence (true T2, not FLAIR-like)
       +5  2D acquisition (MRAcquisitionType)
       +3  axial orientation
       +1  coronal orientation
       +1  non-contrast (no ce- entity; soft preference)

    Args:
        t2w_path: Path to T2w NIfTI file.
        flair_resolved: Resolved path of FLAIR file, or None.

    Returns:
        Integer score (higher is better).
    """
    score = 0

    # Duplicate check against FLAIR
    if flair_resolved is not None and t2w_path.resolve() == flair_resolved:
        score -= 100

    # Non-contrast is a soft preference (Gd has minimal T2 effect)
    if "ce-" not in t2w_path.name.lower():
        score += 1

    # Read sidecar JSON
    sidecar_path = t2w_path.with_name(
        t2w_path.name.replace(".nii.gz", ".json").replace(".nii", ".json")
    )
    sidecar: Dict = {}
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read sidecar: %s", sidecar_path)

    # Prefer no inversion recovery (IR = FLAIR-like contrast)
    # Split on separator to avoid false positives (e.g., "SPIR" on Philips)
    scanning_seq = sidecar.get("ScanningSequence", "")
    if isinstance(scanning_seq, list):
        has_ir = "IR" in scanning_seq
    else:
        seq_tokens = [s.strip() for s in scanning_seq.replace("\\", "/").split("/")]
        has_ir = "IR" in seq_tokens
    if not has_ir:
        score += 10

    # Prefer 2D acquisition
    if sidecar.get("MRAcquisitionType") == "2D":
        score += 5

    # Prefer axial orientation
    iop = sidecar.get("ImageOrientationPatientDICOM")
    if iop and len(iop) == 6:
        orientation = _classify_orientation(iop)
        if orientation == "axial":
            score += 3
        elif orientation == "coronal":
            score += 1

    return score


def _find_t2w(
    anat_dir: Path,
    flair_path: Optional[Path] = None,
) -> Optional[Path]:
    """Find the best T2w image using sidecar metadata scoring.

    Selection criteria (by score weight):
    1. Not the same underlying file as FLAIR (-100 penalty)
    2. No inversion recovery in ScanningSequence (+10)
    3. 2D acquisition preferred (+5)
    4. Axial orientation preferred (+3)
    5. Non-contrast preferred (+1, soft — Gd has minimal T2 effect)

    Args:
        anat_dir: Path to the anatomical directory.
        flair_path: Path to the selected FLAIR file (for duplicate check).

    Returns:
        Path to best T2w image, or None if not found.
    """
    t2w_files = (
        list(anat_dir.glob("*_T2w.nii.gz"))
        + list(anat_dir.glob("*_T2w.nii"))
    )
    if not t2w_files:
        return None

    flair_resolved = flair_path.resolve() if flair_path is not None else None

    scored = [(f, _score_t2w_candidate(f, flair_resolved)) for f in t2w_files]
    scored.sort(key=lambda x: (-x[1], x[0].name))  # deterministic tie-breaking

    winner_path, winner_score = scored[0]
    if len(scored) > 1:
        logger.info(
            "T2w selection: %s (score=%d) from %d candidates",
            winner_path.name, winner_score, len(scored),
        )
    if winner_score < 0:
        logger.warning(
            "Best T2w candidate has negative score (%d): %s "
            "— may be FLAIR-like or duplicate of FLAIR",
            winner_score, winner_path.name,
        )

    return winner_path


def _select_best_flair(candidates: list[Path]) -> Path:
    """Select best FLAIR from multiple candidates, preferring 3D.

    Selection criteria (in priority order):
    1. MRAcquisitionType == "3D" in JSON sidecar
    2. "acq-3d" in filename
    3. First candidate (fallback)

    Args:
        candidates: List of FLAIR NIfTI paths (must be non-empty).

    Returns:
        Path to the best FLAIR candidate.
    """
    if len(candidates) == 1:
        return candidates[0]

    # Prefer 3D acquisition (check JSON sidecar MRAcquisitionType)
    for f in candidates:
        json_path = f.with_name(
            f.name.replace(".nii.gz", ".json").replace(".nii", ".json")
        )
        if json_path.exists():
            import json as json_mod
            try:
                meta = json_mod.loads(json_path.read_text())
            except (json_mod.JSONDecodeError, OSError):
                continue
            if meta.get("MRAcquisitionType") == "3D":
                logger.info("Multiple FLAIRs found; selected 3D: %s", f.name)
                return f

    # Fallback: prefer acq-3d in filename
    for f in candidates:
        if "acq-3d" in f.name.lower():
            logger.info("Multiple FLAIRs found; selected acq-3d: %s", f.name)
            return f

    logger.warning(
        "Multiple FLAIRs found, no 3D preference resolved; using: %s",
        candidates[0].name,
    )
    return candidates[0]


def _find_flair_precontrast(anat_dir: Path) -> Optional[Path]:
    """Find pre-contrast FLAIR image.

    Pre-contrast FLAIR is identified by having _FLAIR suffix and NOT having
    a contrast entity (ce-) in the filename. HD-GLIO requires pre-contrast
    FLAIR (trained on BraTS non-enhanced FLAIR).

    When multiple pre-contrast FLAIRs exist, prefers 3D (BraTS/HD-GLIO
    protocol) via ``_select_best_flair``.

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to pre-contrast FLAIR image, or None if not found.
    """
    from pmriprep.utils.structural_reference import is_postcontrast_image

    flair_files = (
        list(anat_dir.glob("*_FLAIR.nii.gz"))
        + list(anat_dir.glob("*_FLAIR.nii"))
        + list(anat_dir.glob("*_flair.nii.gz"))
        + list(anat_dir.glob("*_flair.nii"))
    )

    precon = [f for f in flair_files if not is_postcontrast_image(f)]
    if precon:
        return _select_best_flair(precon)

    return None


def _find_flair_postcontrast(anat_dir: Path) -> Optional[Path]:
    """Find post-contrast FLAIR image.

    Post-contrast FLAIR is identified by having _FLAIR suffix AND having
    a contrast entity (ce-) or post-contrast keyword in the filename.

    When multiple post-contrast FLAIRs exist, prefers 3D via
    ``_select_best_flair``.

    Args:
        anat_dir: Path to the anatomical directory.

    Returns:
        Path to post-contrast FLAIR image, or None if not found.
    """
    from pmriprep.utils.structural_reference import is_postcontrast_image

    flair_files = (
        list(anat_dir.glob("*_FLAIR.nii.gz"))
        + list(anat_dir.glob("*_FLAIR.nii"))
        + list(anat_dir.glob("*_flair.nii.gz"))
        + list(anat_dir.glob("*_flair.nii"))
    )

    postcon = [f for f in flair_files if is_postcontrast_image(f)]
    if postcon:
        return _select_best_flair(postcon)

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


def _find_b1_data(
    anat_dir: Path,
    perf_dir: Optional[Path] = None,
    fmap_dir: Optional[Path] = None,
) -> List[Path]:
    """Find B1+ mapping images (AFI or pre-computed tfl_b1map).

    B1+ images are identified by:
    - Having 'afi' or 'AFI' or 'b1' or 'B1' in filename (AFI)
    - Having 'TB1TFL' in filename (Siemens tfl_b1map)
    - Having 'TB1AFI' in filename (BIDS-standard AFI suffix)
    - Typically pairs of images with different TRs (AFI)
    - Or single pre-computed flip angle map (tfl_b1map)

    Args:
        anat_dir: Path to the anatomical directory.
        perf_dir: Path to the perfusion directory (AFI sometimes stored there).
        fmap_dir: Path to the fieldmap directory (B1 maps stored here per BIDS spec).

    Returns:
        List of paths to B1+ images.
    """
    b1_files: List[Path] = []

    # Search patterns for AFI, tfl_b1map, and general B1 mapping
    b1_patterns = [
        "*TB1TFL*.nii.gz",
        "*TB1TFL*.nii",
        "*TB1AFI*.nii.gz",
        "*TB1AFI*.nii",
        "*AFI*.nii.gz",
        "*AFI*.nii",
        "*afi*.nii.gz",
        "*afi*.nii",
        "*B1*.nii.gz",
        "*B1*.nii",
        "*b1map*.nii.gz",
        "*b1map*.nii",
    ]

    for pattern in b1_patterns:
        b1_files.extend(anat_dir.glob(pattern))

    if perf_dir:
        for pattern in b1_patterns:
            b1_files.extend(perf_dir.glob(pattern))

    if fmap_dir:
        for pattern in b1_patterns:
            b1_files.extend(fmap_dir.glob(pattern))

    # Remove duplicates while preserving order
    seen = set()
    unique_b1: List[Path] = []
    for f in b1_files:
        if f not in seen:
            seen.add(f)
            unique_b1.append(f)

    return sorted(unique_b1)


def _classify_b1_data(
    b1_files: List[Path],
) -> dict[str, Optional[Union[str, Path, float]]]:
    """Classify B1+ files as tfl_b1map (pre-computed), AFI 4D, or AFI 3D.

    First checks for Siemens tfl_b1map pre-computed flip angle maps (TB1TFL),
    then falls through to AFI classification (4D single volume or 3D paired).

    Args:
        b1_files: List of discovered B1+ NIfTI paths.

    Returns:
        Dict with keys: afi_mode ("precomputed", "3d", "4d", or None),
        b1_precomputed_file, b1_nominal_prep_fa, b1_method_detected,
        afi_4d_file, afi_tr1_ms, afi_tr2_ms, afi_flip_angle.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    result: dict[str, Optional[Union[str, Path, float]]] = {
        "afi_mode": None,
        "afi_4d_file": None,
        "afi_tr1_file": None,
        "afi_tr2_file": None,
        "afi_tr1_ms": None,
        "afi_tr2_ms": None,
        "afi_flip_angle": None,
        "b1_precomputed_file": None,
        "b1_nominal_prep_fa": None,
        "b1_method_detected": None,
    }

    if not b1_files:
        return result

    # --- Phase 1: Check for tfl_b1map pre-computed B1+ maps ---
    # TB1TFL files come in pairs: acq-famp (flip angle map) and acq-anat (magnitude).
    # We want only the famp file.
    for b1_path in b1_files:
        fname = b1_path.name.upper()
        if "TB1TFL" not in fname:
            continue

        # Skip the anatomical/magnitude companion
        if "ANAT" in fname:
            logger.debug("Skipping TB1TFL anatomical file: %s", b1_path.name)
            continue

        # Check for famp in filename, or fall back to JSON metadata
        is_famp = "FAMP" in fname

        if not is_famp:
            # Check JSON sidecar for ImageType containing "FLIP ANGLE MAP"
            json_path = Path(
                str(b1_path)
                .replace(".nii.gz", ".json")
                .replace(".nii", ".json")
            )
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
                image_type = meta.get("ImageType", [])
                if isinstance(image_type, list):
                    image_type_str = " ".join(image_type).upper()
                else:
                    image_type_str = str(image_type).upper()
                is_famp = "FLIP ANGLE MAP" in image_type_str

                # Also check PulseSequenceDetails as final fallback
                if not is_famp:
                    psd = str(meta.get("PulseSequenceDetails", ""))
                    is_famp = "tfl_b1map" in psd.lower()

        if not is_famp:
            logger.debug(
                "TB1TFL file %s is not a flip angle map — skipping",
                b1_path.name,
            )
            continue

        # Read FlipAngle from JSON sidecar (required for scaling)
        json_path = Path(
            str(b1_path)
            .replace(".nii.gz", ".json")
            .replace(".nii", ".json")
        )
        nominal_prep_fa: Optional[float] = None
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
            fa = meta.get("FlipAngle")
            if fa is not None:
                nominal_prep_fa = float(fa)

        # Defensive check: famp should have a high FlipAngle (saturation pulse)
        # The anat companion has a low readout FA (e.g., 8°)
        if nominal_prep_fa is not None and nominal_prep_fa < 20:
            raise ValueError(
                f"tfl_b1map file {b1_path.name} has FlipAngle={nominal_prep_fa}° "
                f"(expected ≥20° for saturation pulse). This looks like the "
                f"anatomical companion — use the acq-famp file instead."
            )

        logger.info(
            "Found tfl_b1map pre-computed B1+ map: %s (FlipAngle=%s°)",
            b1_path.name,
            nominal_prep_fa,
        )
        result["afi_mode"] = "precomputed"
        result["b1_precomputed_file"] = b1_path
        result["b1_nominal_prep_fa"] = nominal_prep_fa
        result["b1_method_detected"] = "tfl_b1map"
        return result

    # --- Phase 2: Fall through to AFI classification ---
    # Filter to only AFI files (exclude any remaining TB1TFL files)
    afi_files = [f for f in b1_files if "TB1TFL" not in f.name.upper()]

    if not afi_files:
        return result

    # Check if any single file is 4D with exactly 2 volumes
    for afi_path in afi_files:
        try:
            img = nib.load(str(afi_path))
            shape = img.shape  # type: ignore[attr-defined]
            if len(shape) == 4 and shape[3] == 2:
                logger.info("Found 4D AFI volume (2 timepoints): %s", afi_path)
                result["afi_mode"] = "4d"
                result["afi_4d_file"] = afi_path

                # Try to extract TRs from sidecar
                json_path = Path(
                    str(afi_path)
                    .replace(".nii.gz", ".json")
                    .replace(".nii", ".json")
                )
                if json_path.exists():
                    with open(json_path) as f:
                        metadata = json.load(f)

                    # RepetitionTimeExcitation (BIDS standard)
                    if "RepetitionTimeExcitation" in metadata:
                        tr_exc = metadata["RepetitionTimeExcitation"]
                        if isinstance(tr_exc, list) and len(tr_exc) >= 2:
                            trs = sorted([tr_exc[0] * 1000, tr_exc[1] * 1000])
                            result["afi_tr1_ms"] = trs[0]
                            result["afi_tr2_ms"] = trs[1]

                    # Fallback: RepetitionTime array
                    if result["afi_tr1_ms"] is None and "RepetitionTime" in metadata:
                        rt = metadata["RepetitionTime"]
                        if isinstance(rt, list) and len(rt) >= 2:
                            trs = sorted([rt[0] * 1000, rt[1] * 1000])
                            result["afi_tr1_ms"] = trs[0]
                            result["afi_tr2_ms"] = trs[1]

                    # Extract nominal flip angle from sidecar
                    if "FlipAngle" in metadata:
                        fa = metadata["FlipAngle"]
                        if isinstance(fa, list):
                            fa = fa[0]
                        result["afi_flip_angle"] = float(fa)

                result["b1_method_detected"] = "afi"
                return result
        except Exception:
            continue

    # If we have 2+ 3D files, classify as 3D mode
    if len(afi_files) >= 2:
        result["afi_mode"] = "3d"
        logger.info("Found %d separate AFI volumes (3D mode)", len(afi_files))

        # Read TR from all sidecars to select the correct pair
        file_trs: list[tuple[Path, float]] = []
        for afi_path in afi_files:
            json_path = Path(
                str(afi_path)
                .replace(".nii.gz", ".json")
                .replace(".nii", ".json")
            )
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
                tr_val = meta.get("RepetitionTime")
                if tr_val is not None:
                    file_trs.append((afi_path, float(tr_val) * 1000))

        if len(afi_files) > 2 and len(file_trs) >= 2:
            # >2 files (e.g. duplicate runs from dcm2niix): group by TR,
            # pick one file per unique TR (shortest TR → tr1, longest → tr2)
            tr_groups: dict[float, list[Path]] = {}
            for path, tr_ms in file_trs:
                tr_groups.setdefault(tr_ms, []).append(path)

            unique_trs = sorted(tr_groups.keys())
            if len(unique_trs) >= 2:
                # Select first file from the shortest and longest TR groups
                selected_tr1 = unique_trs[0]
                selected_tr2 = unique_trs[-1]
                selected = [
                    (tr_groups[selected_tr1][0], selected_tr1),
                    (tr_groups[selected_tr2][0], selected_tr2),
                ]
                logger.info(
                    "AFI: %d files found with %d unique TRs. "
                    "Selected pair: TR1=%.1fms (%s), TR2=%.1fms (%s)",
                    len(afi_files), len(unique_trs),
                    selected_tr1, tr_groups[selected_tr1][0].name,
                    selected_tr2, tr_groups[selected_tr2][0].name,
                )
            else:
                # All files have the same TR — degenerate, take first two
                logger.warning(
                    "AFI: %d files but all have TR=%.1fms. "
                    "Cannot form a valid pair; using first two files.",
                    len(afi_files), unique_trs[0],
                )
                selected = [(file_trs[0][0], file_trs[0][1]),
                            (file_trs[1][0], file_trs[1][1])]
        elif len(file_trs) >= 2:
            # Exactly 2 files with TR info — sort by TR
            file_trs.sort(key=lambda x: x[1])
            selected = [(file_trs[0][0], file_trs[0][1]),
                        (file_trs[1][0], file_trs[1][1])]
        else:
            # Fallback: no TR info available, use first two files as-is
            selected = [(afi_files[0], 0.0), (afi_files[1], 0.0)]

        result["afi_tr1_file"] = selected[0][0]
        result["afi_tr2_file"] = selected[1][0]
        if selected[0][1] > 0:
            result["afi_tr1_ms"] = selected[0][1]
        if selected[1][1] > 0:
            result["afi_tr2_ms"] = selected[1][1]

        # Extract nominal flip angle from the TR1 file's sidecar
        tr1_json = Path(
            str(selected[0][0])
            .replace(".nii.gz", ".json")
            .replace(".nii", ".json")
        )
        if tr1_json.exists():
            with open(tr1_json) as f:
                meta = json.load(f)
            if "FlipAngle" in meta:
                fa = meta["FlipAngle"]
                if isinstance(fa, list):
                    fa = fa[0]
                result["afi_flip_angle"] = float(fa)

        result["b1_method_detected"] = "afi"

    return result


def _find_dwi(bids_dir: Path, subject: str, session: Optional[str]) -> Optional[Path]:
    """Find DWI image best suited for ADC calculation.

    When multiple DWI files exist, prefers the one with fewest volumes that
    still has >= 2 unique b-values (sufficient for mono-exponential ADC).

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

    matches: List[Path] = []
    for pattern in ["*_dwi.nii.gz", "*_dwi.nii"]:
        matches.extend(dwi_dir.glob(pattern))

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    # Multiple DWI files: prefer trace over DTI, AP/PA over LR/RL, fewer volumes
    import json as _json

    import numpy as np

    from pmriprep.utils.dce_tortuosity import is_trace_dwi

    scored: List[tuple[tuple[int, int, float], Path]] = []
    for m in matches:
        bval_path = str(m).replace(".nii.gz", ".bval").replace(".nii", ".bval")
        bvec_path = str(m).replace(".nii.gz", ".bvec").replace(".nii", ".bvec")
        json_path = str(m).replace(".nii.gz", ".json").replace(".nii", ".json")

        is_dti = 1  # default: assume DTI (worse)
        is_lr_pe = 0  # default: assume AP/PA (better)
        n_vols: float = float("inf")

        try:
            if Path(bval_path).exists():
                bvals = np.atleast_1d(np.loadtxt(bval_path))
                if len(np.unique(bvals)) >= 2:
                    n_vols = float(len(bvals))
                    # Load bvecs if available for trace classification
                    bvecs = None
                    if Path(bvec_path).exists():
                        bvecs = np.loadtxt(bvec_path)
                    is_dti = 0 if is_trace_dwi(bvals, bvecs) else 1
        except Exception as exc:
            logger.warning("Could not read %s: %s", bval_path, exc)

        # Check phase encoding direction from JSON sidecar
        try:
            if Path(json_path).exists():
                with open(json_path) as f:
                    sidecar = _json.load(f)
                pe_dir = sidecar.get("PhaseEncodingDirection", "")
                if pe_dir.startswith("i"):
                    is_lr_pe = 1  # LR/RL is less preferred
        except Exception:
            pass

        scored.append(((is_dti, is_lr_pe, n_vols), m))

    scored.sort(key=lambda x: x[0])
    winner_key, winner_path = scored[0]
    logger.info(
        "Multiple DWI files found, selected %s "
        "(is_dti=%d, is_lr_pe=%d, n_vols=%s)",
        winner_path.name,
        winner_key[0],
        winner_key[1],
        winner_key[2] if winner_key[2] != float("inf") else "unknown",
    )
    return winner_path


def _determine_pipeline_availability(modalities: ModalityAvailability) -> PipelineAvailability:
    """Determine which pipelines can run based on available modalities.

    Rules:
    - anat: Always runs (T1w is minimum requirement but assumed present)
    - tumor_seg: Requires all 4 modalities (T1w-pre, T1w-ce-GAD, T2w, FLAIR)
    - dsc: Requires DSC data
    - dce: Requires DCE data

    Args:
        modalities: Available modalities.

    Returns:
        Pipeline availability with reasons for unavailable pipelines.
    """
    pipelines = PipelineAvailability()

    # Tumor segmentation requires 4 modalities
    tumor_seg_missing: List[str] = []
    if not modalities.t1w_pre:
        tumor_seg_missing.append("T1w-pre")
    if not modalities.t1w_ce_gad:
        tumor_seg_missing.append("T1w-ce-GAD")
    if not modalities.t2w:
        tumor_seg_missing.append("T2w")
    if not modalities.flair:
        if modalities.flair_postcon:
            tumor_seg_missing.append("FLAIR (pre-contrast)")
        else:
            tumor_seg_missing.append("FLAIR")

    if not tumor_seg_missing:
        pipelines.tumor_seg = True
    else:
        pipelines.tumor_seg = False
        pipelines.tumor_seg_reason = f"Missing modalities: {', '.join(tumor_seg_missing)}"

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
    layout: Optional["BIDSLayout"] = None,
) -> SessionData:
    """Discover available imaging modalities for a subject/session.

    This function scans the BIDS dataset to find available imaging modalities
    and determines which pipeline components can run.

    Args:
        bids_dir: Path to the BIDS dataset root.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.
        layout: Optional pre-built ``BIDSLayout`` to reuse (avoids re-indexing
            on every per-session call). PR1 only threads it; per-modality
            finders still use ``Path.glob`` (migrated in PR2).

    Returns:
        SessionData containing available modalities and runnable pipelines.
    """
    # TODO(B3-PR2): wire layout into per-modality finders below.
    del layout
    bids_path = Path(bids_dir)
    modalities = ModalityAvailability()

    # Build base path
    if session:
        base_path = bids_path / f"sub-{subject}" / f"ses-{session}"
    else:
        base_path = bids_path / f"sub-{subject}"

    # Find anatomical directory
    anat_dir = base_path / "anat"
    # Find fieldmap directory (AFI/B1 maps stored here per BIDS spec)
    fmap_dir = base_path / "fmap"
    fmap_dir_opt = fmap_dir if fmap_dir.is_dir() else None

    if anat_dir.is_dir():
        modalities.t1w_pre = _find_t1w_precontrast(anat_dir)
        modalities.t1w_ce_gad = _find_t1w_ce_gad(anat_dir)
        modalities.flair = _find_flair_precontrast(anat_dir)
        modalities.flair_postcon = _find_flair_postcontrast(anat_dir)
        modalities.t2w = _find_t2w(
            anat_dir,
            flair_path=modalities.flair or modalities.flair_postcon,
        )
        modalities.vfa = _find_vfa(anat_dir)
        modalities.afi = _find_b1_data(anat_dir, fmap_dir=fmap_dir_opt)

    # Find perfusion directory
    perf_dir = base_path / "perf"
    if perf_dir.is_dir():
        modalities.dsc = _find_dsc(perf_dir)
        modalities.dce = _find_dce(perf_dir, anat_dir if anat_dir.is_dir() else None)

        # Validate perfusion files are 4D timeseries
        import nibabel as nib

        if modalities.dsc:
            _shape = nib.load(modalities.dsc).header.get_data_shape()  # type: ignore[attr-defined]
            if len(_shape) < 4 or _shape[3] < 2:
                logger.error(
                    "DSC file %s is %dD with shape %s. "
                    "Expected 4D timeseries (>1 volume). "
                    "Skipping DSC — check DICOM conversion.",
                    modalities.dsc.name, len(_shape), _shape,
                )
                modalities.dsc = None

        if modalities.dce:
            _shape = nib.load(modalities.dce).header.get_data_shape()  # type: ignore[attr-defined]
            if len(_shape) < 4 or _shape[3] < 2:
                logger.error(
                    "DCE file %s is %dD with shape %s. "
                    "Expected 4D timeseries (>1 volume). "
                    "Skipping DCE — check DICOM conversion.",
                    modalities.dce.name, len(_shape), _shape,
                )
                modalities.dce = None

        # Also check perf for VFA/B1
        if anat_dir.is_dir():
            modalities.vfa = _find_vfa(anat_dir, perf_dir)
            modalities.afi = _find_b1_data(anat_dir, perf_dir, fmap_dir=fmap_dir_opt)

    # Classify B1+ data (tfl_b1map, AFI 4D, or AFI 3D)
    if modalities.afi:
        b1_info = _classify_b1_data(modalities.afi)
        if b1_info["afi_mode"] is not None:
            modalities.afi_mode = str(b1_info["afi_mode"])
        if b1_info["afi_4d_file"] is not None:
            modalities.afi_4d_file = Path(str(b1_info["afi_4d_file"]))
        if b1_info["afi_tr1_file"] is not None:
            modalities.afi_tr1_file = Path(str(b1_info["afi_tr1_file"]))
        if b1_info["afi_tr2_file"] is not None:
            modalities.afi_tr2_file = Path(str(b1_info["afi_tr2_file"]))
        if b1_info["afi_tr1_ms"] is not None:
            modalities.afi_tr1_ms = float(b1_info["afi_tr1_ms"])  # type: ignore[arg-type]
        if b1_info["afi_tr2_ms"] is not None:
            modalities.afi_tr2_ms = float(b1_info["afi_tr2_ms"])  # type: ignore[arg-type]
        if b1_info.get("afi_flip_angle") is not None:
            modalities.afi_flip_angle = float(b1_info["afi_flip_angle"])  # type: ignore[arg-type]
        # Wire tfl_b1map-specific fields
        if b1_info.get("b1_precomputed_file") is not None:
            modalities.b1_precomputed_file = Path(str(b1_info["b1_precomputed_file"]))
        if b1_info.get("b1_nominal_prep_fa") is not None:
            modalities.b1_nominal_prep_fa = float(b1_info["b1_nominal_prep_fa"])  # type: ignore[arg-type]
        if b1_info.get("b1_method_detected") is not None:
            modalities.b1_method_detected = str(b1_info["b1_method_detected"])

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


