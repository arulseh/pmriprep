"""Data discovery logic for automatic detection of available modalities.

This module implements automatic detection of available imaging modalities
in a BIDS dataset and determines which pipeline components can run.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from bids.layout import BIDSLayout

logger = logging.getLogger("pmriprep.workflows.discovery")


@dataclass
class ModalityAvailability:
    """Tracks availability of different imaging modalities.

    Attributes
    ----------
    t1w_pre : Path, optional
        Path to pre-contrast T1w image, if available.
    t1w_ce_gad : Path, optional
        Path to contrast-enhanced T1w (GAD) image, if available.
    t1w_subtraction_compatible : bool, optional
        Whether the selected pre- and postcontrast T1w images have no detected
        sequence mismatch and may be used for subtraction. ``None`` when a
        pair was not evaluated.
    t2w : Path, optional
        Path to T2w image, if available.
    flair : Path, optional
        Path to pre-contrast FLAIR image, if available.
    flair_postcon : Path, optional
        Path to post-contrast FLAIR image, if available.
    dsc : Path, optional
        Path to DSC perfusion image, if available.
    dsc_candidates : list of Path
        DSC candidates retained for ambiguity validation when DSC is enabled.
    dce : Path, optional
        Path to DCE perfusion image, if available.
    dce_candidates : list of Path
        DCE candidates retained for ambiguity validation when DCE is enabled.
    vfa : list of Path
        List of paths to VFA (Variable Flip Angle) images for T1 mapping.
    afi : list of Path
        List of paths to AFI (Actual Flip Angle Imaging) images for B1+ mapping.
    b1_precomputed_file : Path, optional
        Path to pre-computed B1+ flip angle map (e.g., tfl_b1map famp).
    b1_precomputed_candidates : list of Path
        Eligible pre-computed B1+ maps retained for ambiguity validation.
    afi_4d_candidates : list of Path
        Eligible 4D AFI acquisitions retained for ambiguity validation.
    b1_nominal_prep_fa : float, optional
        Nominal preparation flip angle (degrees) from famp JSON sidecar.
    b1_method_detected : str, optional
        Detected B1+ method ("afi" or "precomputed"), if any.
    dsc_n_volumes : int, optional
        Number of volumes in the DSC 4D series, if loaded.
    dce_n_volumes : int, optional
        Number of volumes in the DCE 4D series, if loaded.
    """

    t1w_pre: Optional[Path] = None
    t1w_ce_gad: Optional[Path] = None
    t1w_subtraction_compatible: Optional[bool] = None
    t2w: Optional[Path] = None
    flair: Optional[Path] = None
    flair_postcon: Optional[Path] = None
    dsc: Optional[Path] = None
    dsc_candidates: List[Path] = field(default_factory=list)
    dce: Optional[Path] = None
    dce_candidates: List[Path] = field(default_factory=list)
    vfa: List[Path] = field(default_factory=list)
    afi: List[Path] = field(default_factory=list)
    afi_mode: Optional[str] = None  # "4d" or "precomputed"
    afi_4d_file: Optional[Path] = None  # First eligible 4D AFI input
    afi_tr1_ms: Optional[float] = None  # TR1 in ms (from sidecar)
    afi_tr2_ms: Optional[float] = None  # TR2 in ms (from sidecar)
    afi_flip_angle: Optional[float] = None  # Nominal flip angle in degrees (from sidecar)
    dsc_n_volumes: Optional[int] = None
    dce_n_volumes: Optional[int] = None
    b1_precomputed_file: Optional[Path] = None  # Pre-computed B1+ map (TB1TFL famp or TB1map)
    b1_precomputed_candidates: List[Path] = field(default_factory=list)
    afi_4d_candidates: List[Path] = field(default_factory=list)
    b1_nominal_prep_fa: Optional[float] = None  # FlipAngle sidecar (actual-flip encodings)
    b1_method_detected: Optional[str] = None  # "afi" or "precomputed"
    b1_map_encoding: Optional[str] = None  # Pixel encoding hint for a precomputed map


@dataclass
class PipelineAvailability:
    """Tracks which pipelines can run based on available data.

    Attributes
    ----------
    anat : bool
        Whether anatomical pipeline can run (always True).
    tumor_seg : bool
        Whether tumor segmentation can run.
    dsc : bool
        Whether DSC perfusion pipeline can run.
    dce : bool
        Whether DCE perfusion pipeline can run.
    tumor_seg_reason : str, optional
        Reason why tumor segmentation cannot run, if applicable.
    dsc_reason : str, optional
        Reason why DSC cannot run, if applicable.
    dce_reason : str, optional
        Reason why DCE cannot run, if applicable.
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

    Attributes
    ----------
    subject : str
        Subject ID (without 'sub-' prefix).
    session : str, optional
        Session ID (without 'ses-' prefix), if applicable.
    modalities : ModalityAvailability
        Available modalities.
    pipelines : PipelineAvailability
        Available pipelines.
    """

    subject: str
    session: Optional[str]
    modalities: ModalityAvailability
    pipelines: PipelineAvailability


def _select_t1w_by_acquisition(
    candidates: List[Path], label: str
) -> Optional[Path]:
    """Pick best T1w from candidates by acquisition preference.

    Priority: MPRAGE > IR/MP2RAGE > SPGR/GRE > non-SE > SE. SE T1w is
    typically 2D with thick slices, poor for registration; MPRAGE is the
    NiPreps gold standard.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    preferred_acq = ["mprage", "ir", "mp2rage", "spgr", "gre"]
    for acq_pref in preferred_acq:
        for f in candidates:
            if acq_pref in f.name.lower():
                logger.info(
                    "Selected %s T1w (preferred acq=%s): %s",
                    label, acq_pref, f.name,
                )
                return f

    non_se = [f for f in candidates if "acq-se" not in f.name.lower()]
    if non_se:
        logger.info("Selected %s T1w (non-SE fallback): %s", label, non_se[0].name)
        return non_se[0]

    logger.warning(
        "Only SE T1w available for %s; registration may be suboptimal: %s",
        label, candidates[0].name,
    )
    return candidates[0]


def _find_t1w_precontrast(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
) -> Optional[Path]:
    """Find pre-contrast T1w (no ``ce-`` entity) using BIDSLayout.

    Multiple candidates are ranked by acquisition preference (MPRAGE >
    IR/MP2RAGE > SPGR/GRE > non-SE > SE) to avoid picking thick-slice SE
    when a 3D MPRAGE is available.
    """
    from bids.layout import Query

    files = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype="anat",
            suffix="T1w",
            extension=["nii", "nii.gz"],
            ceagent=Query.NONE,
            return_type="filename",
        )
    ]
    return _select_t1w_by_acquisition(files, "precon")


def _find_t1w_ce_gad(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
) -> Optional[Path]:
    """Find contrast-enhanced T1w (post-Gd) using BIDSLayout.

    Prefers ``ce-GAD`` (case-insensitive), falls back to any contrast-
    enhanced T1w. Acquisition preference identical to precontrast.
    """
    from bids.layout import Query

    all_ce = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype="anat",
            suffix="T1w",
            extension=["nii", "nii.gz"],
            ceagent=Query.REQUIRED,
            return_type="filename",
        )
    ]
    if not all_ce:
        return None

    gad = [f for f in all_ce if "gad" in f.name.lower()]
    return _select_t1w_by_acquisition(gad if gad else all_ce, "ce-GAD")


def _normalize_scanning_sequence(val: object) -> Optional[frozenset]:
    """Normalize ScanningSequence — delegates to :func:`pmriprep.utils.bids.normalize_dicom_multi_value`."""
    from pmriprep.utils.bids import normalize_dicom_multi_value
    return normalize_dicom_multi_value(val)


def _scalar(val: object) -> Optional[float]:
    """Extract a scalar float from metadata that may be list-typed (Philips).

    Returns ``None`` on missing, empty, or non-numeric input.
    """
    if val is None:
        return None
    try:
        if isinstance(val, list):
            return float(val[0]) if val else None
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val))
    except (ValueError, TypeError, IndexError):
        return None


def _check_t1w_sequence_compatibility(
    t1w_pre: Path,
    t1w_postcon: Path,
    layout: "BIDSLayout",
) -> bool:
    """Return whether pre- and postcontrast T1w are subtraction-compatible.

    Compares acquisition metadata across three tiers of severity. Logs
    warnings when mismatches are detected; logs info when compatibility
    is confirmed. Never raises — all metadata access is defensive. Missing
    metadata does not by itself disqualify a pair.

    Parameters
    ----------
    t1w_pre : Path
        Path to pre-contrast T1w image.
    t1w_postcon : Path
        Path to post-contrast T1w image.
    layout : BIDSLayout
        BIDS layout for metadata access (sidecar inheritance honored).

    Returns
    -------
    bool
        ``False`` when available metadata demonstrates a sequence mismatch;
        otherwise ``True``.

    Notes
    -----
    Tier 1 detects fundamentally different sequence types (e.g., MPRAGE vs
    SPGR). Tier 2 catches parameter drift within the same sequence type
    that alters tissue contrast. Tier 3 flags structural acquisition
    differences (2D vs 3D).

    Sequence mismatch can produce subtraction-map artifacts that exceed
    gadolinium enhancement signal in low-grade gliomas (5-15% signal
    change vs 10-25% sequence-related difference at 3T).
    """
    try:
        meta_pre = layout.get_metadata(str(t1w_pre)) or {}
    except Exception:
        meta_pre = {}
    try:
        meta_post = layout.get_metadata(str(t1w_postcon)) or {}
    except Exception:
        meta_post = {}

    warned = False

    # --- Tier 1: sequence type mismatch ---

    try:
        entities_pre = layout.parse_file_entities(str(t1w_pre))
    except Exception:
        entities_pre = {}
    try:
        entities_post = layout.parse_file_entities(str(t1w_postcon))
    except Exception:
        entities_post = {}

    acq_pre = (entities_pre.get("acquisition") or "").lower() or None
    acq_post = (entities_post.get("acquisition") or "").lower() or None

    if acq_pre and acq_post and acq_pre != acq_post:
        logger.warning(
            "Pre- and post-contrast T1w use different pulse sequences "
            "(acq-%s vs acq-%s). Subtraction map may show apparent signal "
            "differences unrelated to gadolinium enhancement, particularly "
            "at gray-white matter boundaries.",
            acq_pre, acq_post,
        )
        warned = True

    seq_pre = _normalize_scanning_sequence(meta_pre.get("ScanningSequence"))
    seq_post = _normalize_scanning_sequence(meta_post.get("ScanningSequence"))

    if seq_pre and seq_post and seq_pre != seq_post and not warned:
        logger.warning(
            "Pre- and post-contrast T1w use different pulse sequences "
            "(ScanningSequence %s vs %s). Subtraction map may show apparent "
            "signal differences unrelated to gadolinium enhancement, "
            "particularly at gray-white matter boundaries.",
            sorted(seq_pre), sorted(seq_post),
        )
        warned = True

    ti_pre = _scalar(meta_pre.get("InversionTime"))
    ti_post = _scalar(meta_post.get("InversionTime"))

    if (ti_pre is not None) != (ti_post is not None) and not warned:
        logger.warning(
            "Pre- and post-contrast T1w differ in inversion recovery usage "
            "(InversionTime: precon=%s, postcon=%s). Subtraction map may "
            "show apparent signal differences unrelated to gadolinium "
            "enhancement, particularly at gray-white matter boundaries.",
            ti_pre, ti_post,
        )
        warned = True

    # --- Tier 2: parameter mismatch within same sequence type ---
    # Only fires when Tier 1 did not already identify a fundamental sequence
    # mismatch — parameter differences are expected across different sequences.

    if not warned:
        if ti_pre is not None and ti_post is not None:
            if abs(ti_pre - ti_post) > 0.050:
                logger.warning(
                    "Pre- and post-contrast T1w use the same sequence type "
                    "but differ in InversionTime (%.0f ms vs %.0f ms). This "
                    "may introduce tissue-contrast differences that mimic or "
                    "mask true enhancement.",
                    ti_pre * 1000, ti_post * 1000,
                )
                warned = True

        tr_pre = _scalar(meta_pre.get("RepetitionTime"))
        tr_post = _scalar(meta_post.get("RepetitionTime"))

        if tr_pre and tr_post and tr_pre > 0 and tr_post > 0:
            ratio = max(tr_pre, tr_post) / min(tr_pre, tr_post)
            if ratio > 1.5:
                logger.warning(
                    "Pre- and post-contrast T1w use the same sequence type "
                    "but differ in RepetitionTime (%.3f s vs %.3f s, ratio "
                    "%.1fx). This may introduce tissue-contrast differences "
                    "that mimic or mask true enhancement.",
                    tr_pre, tr_post, ratio,
                )
                warned = True

        fa_pre = _scalar(meta_pre.get("FlipAngle"))
        fa_post = _scalar(meta_post.get("FlipAngle"))

        if fa_pre is not None and fa_post is not None:
            if abs(fa_pre - fa_post) > 3.0:
                logger.warning(
                    "Pre- and post-contrast T1w use the same sequence type "
                    "but differ in FlipAngle (%.1f deg vs %.1f deg). This "
                    "may introduce tissue-contrast differences that mimic or "
                    "mask true enhancement.",
                    fa_pre, fa_post,
                )
                warned = True

    # --- Tier 3: acquisition type mismatch (2D vs 3D) ---

    acq_type_pre = meta_pre.get("MRAcquisitionType")
    acq_type_post = meta_post.get("MRAcquisitionType")

    if (acq_type_pre and acq_type_post
            and acq_type_pre != acq_type_post):
        logger.warning(
            "Pre-contrast T1w is %s and post-contrast is %s. Slice profile "
            "and resolution differences may degrade subtraction map quality "
            "near slice boundaries.",
            acq_type_pre, acq_type_post,
        )
        warned = True

    if not warned:
        logger.info(
            "No T1w sequence mismatch detected for subtraction map.",
        )
    return not warned


def _classify_orientation(iop: List[float]) -> str:
    """Classify image orientation as axial, coronal, or sagittal.

    Parameters
    ----------
    iop : list of float
        ImageOrientationPatientDICOM (6 direction cosines).

    Returns
    -------
    str
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
    sidecar: Dict,
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
    """
    score = 0

    if flair_resolved is not None and t2w_path.resolve() == flair_resolved:
        score -= 100

    if "ce-" not in t2w_path.name.lower():
        score += 1

    scanning_seq = sidecar.get("ScanningSequence", "")
    if isinstance(scanning_seq, list):
        has_ir = "IR" in scanning_seq
    else:
        seq_tokens = [s.strip() for s in scanning_seq.replace("\\", "/").split("/")]
        has_ir = "IR" in seq_tokens
    if not has_ir:
        score += 10

    if sidecar.get("MRAcquisitionType") == "2D":
        score += 5

    iop = sidecar.get("ImageOrientationPatientDICOM")
    if iop and len(iop) == 6:
        orientation = _classify_orientation(iop)
        if orientation == "axial":
            score += 3
        elif orientation == "coronal":
            score += 1

    return score


def _find_t2w(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
    flair_path: Optional[Path] = None,
) -> Optional[Path]:
    """Find the best T2w image using BIDSLayout + sidecar metadata scoring.

    Sidecar inheritance is honored via :meth:`BIDSLayout.get_metadata`.
    """
    t2w_files = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype="anat",
            suffix="T2w",
            extension=["nii", "nii.gz"],
            return_type="filename",
        )
    ]
    if not t2w_files:
        return None

    flair_resolved = flair_path.resolve() if flair_path is not None else None

    scored: List[Tuple[Path, int]] = []
    for f in t2w_files:
        try:
            sidecar = layout.get_metadata(str(f)) or {}
        except Exception:
            sidecar = {}
        scored.append((f, _score_t2w_candidate(f, sidecar, flair_resolved)))
    scored.sort(key=lambda x: (-x[1], x[0].name))

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


def _select_best_flair(
    candidates: list[Path], layout: "BIDSLayout"
) -> Path:
    """Pick best FLAIR from candidates, preferring 3D acquisition.

    3D > acq-3d in filename > first match. Sidecar inheritance honored
    via ``layout.get_metadata``.
    """
    if len(candidates) == 1:
        return candidates[0]

    for f in candidates:
        try:
            meta = layout.get_metadata(str(f)) or {}
        except Exception:
            continue
        if meta.get("MRAcquisitionType") == "3D":
            logger.info("Multiple FLAIRs found; selected 3D: %s", f.name)
            return f

    for f in candidates:
        if "acq-3d" in f.name.lower():
            logger.info("Multiple FLAIRs found; selected acq-3d: %s", f.name)
            return f

    logger.warning(
        "Multiple FLAIRs found, no 3D preference resolved; using: %s",
        candidates[0].name,
    )
    return candidates[0]


def _find_flair_precontrast(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
) -> Optional[Path]:
    """Find pre-contrast FLAIR (no ``ce-`` entity).

    HD-GLIO requires pre-contrast FLAIR (BraTS-trained). Prefers 3D when
    multiple candidates exist.
    """
    from pmriprep.utils.structural_reference import is_postcontrast_image

    flair_files = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype="anat",
            suffix="FLAIR",
            extension=["nii", "nii.gz"],
            return_type="filename",
        )
    ]
    precon = [f for f in flair_files if not is_postcontrast_image(f)]
    if precon:
        return _select_best_flair(precon, layout)
    return None


def _find_flair_postcontrast(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
) -> Optional[Path]:
    """Find post-contrast FLAIR (``ce-`` entity or postcon keyword).

    Prefers 3D when multiple candidates exist.
    """
    from pmriprep.utils.structural_reference import is_postcontrast_image

    flair_files = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype="anat",
            suffix="FLAIR",
            extension=["nii", "nii.gz"],
            return_type="filename",
        )
    ]
    postcon = [f for f in flair_files if is_postcontrast_image(f)]
    if postcon:
        return _select_best_flair(postcon, layout)
    return None


def _find_dsc_candidates(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
) -> List[Path]:
    """Find DSC acquisitions for validation when DSC is enabled.

    Parameters
    ----------
    layout : BIDSLayout
        Indexed input dataset.
    subject, session : str, optional
        Participant and optional session labels without their prefixes.

    Returns
    -------
    list of Path
        DSC candidates, or legacy perfusion BOLD candidates when DSC is absent.
    """
    perf_files = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype="perf",
            extension=["nii", "nii.gz"],
            return_type="filename",
        )
    ]
    dsc = sorted({f for f in perf_files if "dsc" in f.name.lower()})
    candidates = dsc or sorted({f for f in perf_files if "_bold" in f.name.lower()})
    return candidates


def _find_dce_candidates(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
) -> List[Path]:
    """Find DCE acquisitions for validation when DCE is enabled.

    Searches both ``perf`` and ``anat`` datatypes because DCE series may be
    stored under either datatype.

    Parameters
    ----------
    layout : BIDSLayout
        Indexed input dataset.
    subject, session : str, optional
        Participant and optional session labels without their prefixes.

    Returns
    -------
    list of Path
        Sorted, unique DCE candidates in the selected subject/session.
    """
    candidates = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype=["perf", "anat"],
            extension=["nii", "nii.gz"],
            return_type="filename",
        )
    ]
    return sorted({f for f in candidates if "dce" in f.name.lower()})


def _find_vfa(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
) -> List[Path]:
    """Find VFA (Variable Flip Angle) images across anat + perf datatypes."""
    candidates = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype=["anat", "perf"],
            extension=["nii", "nii.gz"],
            return_type="filename",
        )
    ]
    vfa = [f for f in candidates if "vfa" in f.name.lower()]
    return sorted(set(vfa))


def _find_b1_data(
    layout: "BIDSLayout",
    subject: str,
    session: Optional[str],
) -> List[Path]:
    """Find B1+ mapping images: TB1TFL, TB1AFI, or non-canonical AFI/B1 names."""
    candidates = [
        Path(p) for p in layout.get(
            subject=subject,
            session=session,
            datatype=["anat", "perf", "fmap"],
            extension=["nii", "nii.gz"],
            return_type="filename",
        )
    ]
    matches = [
        f for f in candidates
        if any(
            tag in f.name.upper()
            for tag in ("TB1TFL", "TB1AFI", "TB1MAP", "AFI", "B1MAP")
        )
        or "b1map" in f.name.lower()
        or f.name.upper().endswith("_B1.NII.GZ")
        or f.name.upper().endswith("_B1.NII")
    ]
    return sorted(set(matches))


def _classify_b1_data(
    b1_files: List[Path],
    layout: "BIDSLayout",
) -> dict[str, Optional[Union[str, Path, float, List[Path]]]]:
    """Classify B1+ files as Siemens tfl_b1map (pre-computed) or 4D AFI.

    Sidecar reads use ``layout.get_metadata`` so BIDS inheritance is
    honored. NIfTI shape inspection still uses nibabel (pybids exposes
    no shape API).

    Parameters
    ----------
    b1_files : list of Path
        Candidate B1+ mapping files to classify.
    layout : BIDSLayout
        BIDS layout for metadata access (sidecar inheritance honored).

    Returns
    -------
    dict
        Dict with keys: afi_mode ("precomputed", "4d", or None),
        b1_precomputed_file, b1_nominal_prep_fa, b1_method_detected,
        afi_4d_file, afi_tr1_ms, afi_tr2_ms, afi_flip_angle, and the
        eligible precomputed/4D-AFI candidate lists.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    result: dict[str, Optional[Union[str, Path, float, List[Path]]]] = {
        "afi_mode": None,
        "afi_4d_file": None,
        "afi_tr1_ms": None,
        "afi_tr2_ms": None,
        "afi_flip_angle": None,
        "b1_precomputed_file": None,
        "b1_nominal_prep_fa": None,
        "b1_method_detected": None,
        "b1_map_encoding": None,
        "b1_precomputed_candidates": [],
        "afi_4d_candidates": [],
    }

    if not b1_files:
        return result

    def _meta(path: Path) -> Dict:
        try:
            return dict(layout.get_metadata(str(path)) or {})
        except Exception:
            return {}

    # --- Phase 1: tfl_b1map pre-computed B1+ maps (acq-famp) ---
    tfl_candidates: list[tuple[Path, Optional[float]]] = []
    for b1_path in b1_files:
        fname = b1_path.name.upper()
        if "TB1TFL" not in fname:
            continue

        if "ANAT" in fname:
            logger.debug("Skipping TB1TFL anatomical file: %s", b1_path.name)
            continue

        is_famp = "FAMP" in fname
        meta = _meta(b1_path)

        if not is_famp and meta:
            image_type = meta.get("ImageType", [])
            if isinstance(image_type, list):
                image_type_str = " ".join(str(x) for x in image_type).upper()
            else:
                image_type_str = str(image_type).upper()
            is_famp = "FLIP ANGLE MAP" in image_type_str
            if not is_famp:
                psd = str(meta.get("PulseSequenceDetails", ""))
                is_famp = "tfl_b1map" in psd.lower()

        if not is_famp:
            logger.debug(
                "TB1TFL file %s is not a flip angle map — skipping",
                b1_path.name,
            )
            continue

        nominal_prep_fa: Optional[float] = None
        fa = meta.get("FlipAngle")
        if fa is not None:
            if isinstance(fa, list):
                fa = fa[0]
            nominal_prep_fa = float(fa)

        # Defensive: famp expects a high saturation FlipAngle; the anat
        # companion has a low readout FA (e.g., 8°).
        if nominal_prep_fa is not None and nominal_prep_fa < 20:
            raise ValueError(
                f"tfl_b1map file {b1_path.name} has FlipAngle={nominal_prep_fa}° "
                f"(expected ≥20° for saturation pulse). Use the acq-famp file."
            )

        tfl_candidates.append((b1_path, nominal_prep_fa))

    # --- Phase 1b: vendor-neutral pre-computed TB1map (Units-driven) ---
    # BIDS TB1map is a preprocessed parametric map whose encoding is signaled by
    # the ``Units`` sidecar (RECOMMENDED, so may be absent → "auto" median infer).
    _UNITS_TO_ENCODING = {
        "p.u.": "percent", "pu": "percent", "%": "percent", "percent": "percent",
        "1": "ratio", "ratio": "ratio",
        "deg": "actual_flip_deg", "degree": "actual_flip_deg",
        "degrees": "actual_flip_deg",
    }
    tb1map_candidates: list[tuple[Path, Optional[float], str]] = []
    for b1_path in b1_files:
        fname = b1_path.name.upper()
        if "TB1MAP" not in fname or "TB1TFL" in fname:
            continue

        meta = _meta(b1_path)
        units = str(meta.get("Units", "")).strip().lower()
        encoding = _UNITS_TO_ENCODING.get(units, "auto")

        nominal_prep_fa = None
        fa = meta.get("FlipAngle")
        if fa is not None:
            if isinstance(fa, list):
                fa = fa[0]
            nominal_prep_fa = float(fa)

        tb1map_candidates.append((b1_path, nominal_prep_fa, encoding))

    precomputed_paths = [path for path, _ in tfl_candidates]
    precomputed_paths.extend(path for path, _, _ in tb1map_candidates)
    result["b1_precomputed_candidates"] = precomputed_paths

    # --- Phase 2: 4D AFI (TB1AFI or non-canonical *AFI* with 2 volumes) ---
    afi_files = [
        f for f in b1_files
        if "TB1TFL" not in f.name.upper() and "TB1MAP" not in f.name.upper()
    ]
    afi_candidates: list[tuple[Path, Dict]] = []
    for afi_path in afi_files:
        try:
            img = nib.load(str(afi_path))
            shape = img.shape  # type: ignore[attr-defined]
        except Exception:
            continue
        if not (len(shape) == 4 and shape[3] == 2):
            continue

        metadata = _meta(afi_path)
        afi_candidates.append((afi_path, metadata))

    result["afi_4d_candidates"] = [path for path, _ in afi_candidates]

    # Preserve the established automatic priority while retaining every
    # eligible candidate so preflight can reject ambiguity before use.
    if tfl_candidates:
        path, nominal_prep_fa = tfl_candidates[0]
        logger.info(
            "Found tfl_b1map pre-computed B1+ map: %s (FlipAngle=%s°)",
            path.name, nominal_prep_fa,
        )
        result["afi_mode"] = "precomputed"
        result["b1_precomputed_file"] = path
        result["b1_nominal_prep_fa"] = nominal_prep_fa
        result["b1_method_detected"] = "precomputed"
        result["b1_map_encoding"] = "actual_flip_deci_deg"
    elif tb1map_candidates:
        path, nominal_prep_fa, encoding = tb1map_candidates[0]
        logger.info(
            "Found pre-computed TB1map B1+ map: %s (encoding=%s, FlipAngle=%s°)",
            path.name, encoding, nominal_prep_fa,
        )
        result["afi_mode"] = "precomputed"
        result["b1_precomputed_file"] = path
        result["b1_nominal_prep_fa"] = nominal_prep_fa
        result["b1_method_detected"] = "precomputed"
        result["b1_map_encoding"] = encoding
    elif afi_candidates:
        afi_path, metadata = afi_candidates[0]
        logger.info("Found 4D AFI volume (2 timepoints): %s", afi_path)
        result["afi_mode"] = "4d"
        result["b1_method_detected"] = "afi"

    if afi_candidates:
        afi_path, metadata = afi_candidates[0]
        result["afi_4d_file"] = afi_path
        tr_exc = metadata.get("RepetitionTimeExcitation")
        if isinstance(tr_exc, list) and len(tr_exc) >= 2:
            trs = sorted([tr_exc[0] * 1000, tr_exc[1] * 1000])
            result["afi_tr1_ms"] = trs[0]
            result["afi_tr2_ms"] = trs[1]
        elif result["afi_tr1_ms"] is None:
            rt = metadata.get("RepetitionTime")
            if isinstance(rt, list) and len(rt) >= 2:
                trs = sorted([rt[0] * 1000, rt[1] * 1000])
                result["afi_tr1_ms"] = trs[0]
                result["afi_tr2_ms"] = trs[1]

        fa = metadata.get("FlipAngle")
        if fa is not None:
            if isinstance(fa, list):
                fa = fa[0]
            result["afi_flip_angle"] = float(fa)

    elif len(afi_files) >= 2:
        logger.error(
            "Found %d separate AFI files but no 4D TB1AFI volume. "
            "Split-mode AFI is not supported; supply a 4D TB1AFI, a "
            "precomputed TB1TFL map, or set b1_method=none.",
            len(afi_files),
        )
    return result


def _determine_pipeline_availability(modalities: ModalityAvailability) -> PipelineAvailability:
    """Determine which pipelines can run based on available modalities.

    Rules:
    - anat: Always runs (T1w is minimum requirement but assumed present)
    - tumor_seg: Requires all 4 modalities (T1w-pre, T1w-ce-GAD, T2w, FLAIR)
    - dsc: Requires DSC data
    - dce: Requires DCE data

    Parameters
    ----------
    modalities : ModalityAvailability
        Available modalities.

    Returns
    -------
    PipelineAvailability
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
    if modalities.dsc or len(modalities.dsc_candidates) > 1:
        pipelines.dsc = True
    else:
        pipelines.dsc = False
        pipelines.dsc_reason = "No DSC perfusion data found"

    # DCE requires DCE data
    if modalities.dce or len(modalities.dce_candidates) > 1:
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

    Parameters
    ----------
    bids_dir : Path or str
        Path to the BIDS dataset root.
    subject : str
        Subject ID (without 'sub-' prefix).
    session : str, optional
        Session ID (without 'ses-' prefix), optional.
    layout : BIDSLayout, optional
        Pre-built ``BIDSLayout`` for the dataset. If ``None``,
        built lazily; passing one in amortizes indexing cost.

    Returns
    -------
    SessionData
        SessionData containing available modalities and runnable pipelines.
    """
    if layout is None:
        from bids.layout import BIDSLayout
        layout = BIDSLayout(str(bids_dir), validate=False)

    modalities = ModalityAvailability()

    modalities.t1w_pre = _find_t1w_precontrast(layout, subject, session)
    modalities.t1w_ce_gad = _find_t1w_ce_gad(layout, subject, session)

    if modalities.t1w_pre and modalities.t1w_ce_gad:
        modalities.t1w_subtraction_compatible = _check_t1w_sequence_compatibility(
            modalities.t1w_pre, modalities.t1w_ce_gad, layout
        )

    modalities.flair = _find_flair_precontrast(layout, subject, session)
    modalities.flair_postcon = _find_flair_postcontrast(layout, subject, session)
    modalities.t2w = _find_t2w(
        layout, subject, session,
        flair_path=modalities.flair or modalities.flair_postcon,
    )
    modalities.vfa = _find_vfa(layout, subject, session)
    modalities.afi = _find_b1_data(layout, subject, session)
    modalities.dsc_candidates = _find_dsc_candidates(layout, subject, session)
    if len(modalities.dsc_candidates) == 1:
        modalities.dsc = modalities.dsc_candidates[0]
    modalities.dce_candidates = _find_dce_candidates(layout, subject, session)
    if len(modalities.dce_candidates) == 1:
        modalities.dce = modalities.dce_candidates[0]

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
        else:
            modalities.dsc_n_volumes = _shape[3]

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
        else:
            modalities.dce_n_volumes = _shape[3]

    if modalities.afi:
        b1_info = _classify_b1_data(modalities.afi, layout)
        if b1_info["afi_mode"] is not None:
            modalities.afi_mode = str(b1_info["afi_mode"])
        if b1_info["afi_4d_file"] is not None:
            modalities.afi_4d_file = Path(str(b1_info["afi_4d_file"]))
        if b1_info["afi_tr1_ms"] is not None:
            modalities.afi_tr1_ms = float(b1_info["afi_tr1_ms"])  # type: ignore[arg-type]
        if b1_info["afi_tr2_ms"] is not None:
            modalities.afi_tr2_ms = float(b1_info["afi_tr2_ms"])  # type: ignore[arg-type]
        if b1_info.get("afi_flip_angle") is not None:
            modalities.afi_flip_angle = float(b1_info["afi_flip_angle"])  # type: ignore[arg-type]
        # Wire pre-computed-B1-map fields
        if b1_info.get("b1_precomputed_file") is not None:
            modalities.b1_precomputed_file = Path(str(b1_info["b1_precomputed_file"]))
        precomputed_candidates = b1_info.get("b1_precomputed_candidates")
        if isinstance(precomputed_candidates, list):
            modalities.b1_precomputed_candidates = [
                Path(str(path)) for path in precomputed_candidates
            ]
        afi_4d_candidates = b1_info.get("afi_4d_candidates")
        if isinstance(afi_4d_candidates, list):
            modalities.afi_4d_candidates = [
                Path(str(path)) for path in afi_4d_candidates
            ]
        if b1_info.get("b1_nominal_prep_fa") is not None:
            modalities.b1_nominal_prep_fa = float(b1_info["b1_nominal_prep_fa"])  # type: ignore[arg-type]
        if b1_info.get("b1_method_detected") is not None:
            modalities.b1_method_detected = str(b1_info["b1_method_detected"])
        if b1_info.get("b1_map_encoding") is not None:
            modalities.b1_map_encoding = str(b1_info["b1_map_encoding"])

    # Determine pipeline availability
    pipelines = _determine_pipeline_availability(modalities)

    return SessionData(
        subject=subject,
        session=session,
        modalities=modalities,
        pipelines=pipelines,
    )
