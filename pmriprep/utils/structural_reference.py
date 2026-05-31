"""Physics-based structural reference selection for PMRIPrep.

This module provides utilities for selecting the optimal structural reference
image based on resolution quality metrics and application-specific requirements.

Resolution quality is determined by:
1. Geometric mean resolution: (Vx × Vy × Vz)^(1/3) - lower is better
2. Isotropy ratio: max(V)/min(V) - closer to 1.0 is better (isotropic)

Applications have different requirements:
- fast_segmentation: Excludes post-contrast T1w (FAST is trained on non-enhanced)
- dce_registration: Prefers post-contrast T1w (matches DCE contrast)
- brain_extraction: No special preference, just best resolution
- default: No special preference, just best resolution
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger("pmriprep.utils.structural_reference")


class ResolutionTier(str, Enum):
    """Resolution quality tier classification."""

    EXCELLENT = "excellent"  # ≤ 1.0 mm
    GOOD = "good"  # 1.0-1.5 mm
    ACCEPTABLE = "acceptable"  # 1.5-2.0 mm
    POOR = "poor"  # > 2.0 mm


# Resolution tier thresholds (geometric mean in mm)
RESOLUTION_THRESHOLDS = {
    ResolutionTier.EXCELLENT: 1.0,
    ResolutionTier.GOOD: 1.5,
    ResolutionTier.ACCEPTABLE: 2.0,
}

# Isotropy ratio thresholds
ISOTROPY_EXCELLENT = 1.2  # Nearly isotropic
ISOTROPY_ACCEPTABLE = 1.5  # Moderately anisotropic
ISOTROPY_POOR = 2.0  # Highly anisotropic


@dataclass
class ImageResolutionInfo:
    """Resolution information for a structural image.

    Attributes
    ----------
    path : Path
        Path to the NIfTI file.
    voxel_sizes : tuple
        Tuple of (x, y, z) voxel dimensions in mm.
    geometric_mean : float
        Geometric mean resolution (x*y*z)^(1/3).
    isotropy_ratio : float
        Anisotropy measure max(v)/min(v).
    resolution_tier : ResolutionTier
        Quality classification.
    is_postcontrast : bool
        Whether this is a post-contrast image.
    """

    path: Path
    voxel_sizes: Tuple[float, float, float]
    geometric_mean: float
    isotropy_ratio: float
    resolution_tier: ResolutionTier
    is_postcontrast: bool = False


ApplicationType = Literal[
    "fast_segmentation",
    "dce_registration",
    "dsc_registration",
    "brain_extraction",
    "default",
]


def get_geometric_mean_resolution(
    voxel_sizes: Tuple[float, float, float],
) -> float:
    """Calculate geometric mean resolution from voxel sizes.

    The geometric mean provides a single scalar representing the "effective"
    isotropic resolution of an anisotropic image.

    Parameters
    ----------
    voxel_sizes : tuple
        Tuple of (voxel_x, voxel_y, voxel_z) in mm.

    Returns
    -------
    float
        Geometric mean resolution: (Vx × Vy × Vz)^(1/3) in mm.

    Examples
    --------
        >>> get_geometric_mean_resolution((1.0, 1.0, 1.0))
        1.0
        >>> get_geometric_mean_resolution((0.5, 0.5, 2.0))  # 2mm slices
        0.7937...
    """
    vx, vy, vz = voxel_sizes
    return float(np.power(vx * vy * vz, 1.0 / 3.0))


def get_isotropy_ratio(
    voxel_sizes: Tuple[float, float, float],
) -> float:
    """Calculate isotropy ratio from voxel sizes.

    The isotropy ratio measures how anisotropic the voxels are.
    A ratio of 1.0 indicates perfectly isotropic voxels.
    Higher values indicate more anisotropic (non-cubic) voxels.

    Parameters
    ----------
    voxel_sizes : tuple
        Tuple of (voxel_x, voxel_y, voxel_z) in mm.

    Returns
    -------
    float
        Isotropy ratio: max(V)/min(V), always >= 1.0.

    Examples
    --------
        >>> get_isotropy_ratio((1.0, 1.0, 1.0))  # Isotropic
        1.0
        >>> get_isotropy_ratio((0.5, 0.5, 2.0))  # 4x anisotropic
        4.0
    """
    v_min = min(voxel_sizes)
    v_max = max(voxel_sizes)
    # Guard against division by zero (shouldn't happen with valid voxel sizes)
    if v_min <= 0:
        return float("inf")
    return v_max / v_min


def classify_resolution(
    geometric_mean: float,
) -> ResolutionTier:
    """Classify resolution into quality tiers.

    Tiers:
        - excellent: ≤ 1.0 mm (high-resolution structural)
        - good: 1.0-1.5 mm (standard structural)
        - acceptable: 1.5-2.0 mm (low-resolution structural)
        - poor: > 2.0 mm (likely functional or low-quality structural)

    Parameters
    ----------
    geometric_mean : float
        Geometric mean resolution in mm.

    Returns
    -------
    ResolutionTier
        ResolutionTier classification.

    Examples
    --------
        >>> classify_resolution(0.8).name
        'EXCELLENT'
        >>> classify_resolution(1.2).name
        'GOOD'
    """
    if geometric_mean <= RESOLUTION_THRESHOLDS[ResolutionTier.EXCELLENT]:
        return ResolutionTier.EXCELLENT
    elif geometric_mean <= RESOLUTION_THRESHOLDS[ResolutionTier.GOOD]:
        return ResolutionTier.GOOD
    elif geometric_mean <= RESOLUTION_THRESHOLDS[ResolutionTier.ACCEPTABLE]:
        return ResolutionTier.ACCEPTABLE
    else:
        return ResolutionTier.POOR


def is_postcontrast_image(path: Union[str, Path]) -> bool:
    """Determine if an image is post-contrast based on BIDS filename.

    Checks for the 'ce-' (contrast entity) in the BIDS filename,
    which indicates a contrast-enhanced acquisition.

    Parameters
    ----------
    path : Path or str
        Path to the NIfTI file.

    Returns
    -------
    bool
        True if the image appears to be post-contrast.
    """
    path = Path(path)
    filename = path.name.lower()

    # BIDS contrast entity
    if "_ce-" in filename or "_ce_" in filename:
        return True

    # Also check for common naming patterns
    postcon_keywords = ["postcon", "post-con", "gad", "gadolinium", "enhanced"]
    for keyword in postcon_keywords:
        if keyword in filename:
            return True

    return False


def get_image_resolution_info(
    path: Union[str, Path],
) -> ImageResolutionInfo:
    """Get resolution information for a NIfTI image.

    Parameters
    ----------
    path : Path or str
        Path to the NIfTI file.

    Returns
    -------
    ImageResolutionInfo
        ImageResolutionInfo with all resolution metrics.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    # Load header to get voxel sizes
    img = nib.load(path)
    zooms = img.header.get_zooms()  # type: ignore[union-attr,attr-defined]

    # Get first 3 dimensions (spatial)
    voxel_sizes: Tuple[float, float, float] = (
        float(zooms[0]),
        float(zooms[1]),
        float(zooms[2]),
    )

    geometric_mean = get_geometric_mean_resolution(voxel_sizes)
    isotropy_ratio = get_isotropy_ratio(voxel_sizes)
    resolution_tier = classify_resolution(geometric_mean)
    is_postcontrast = is_postcontrast_image(path)

    return ImageResolutionInfo(
        path=path,
        voxel_sizes=voxel_sizes,
        geometric_mean=geometric_mean,
        isotropy_ratio=isotropy_ratio,
        resolution_tier=resolution_tier,
        is_postcontrast=is_postcontrast,
    )


# ---------------------------------------------------------------------------
# Confidence-based structural reference selection (v2)
# ---------------------------------------------------------------------------


class ImageType(str, Enum):
    """Image type classification for confidence scoring."""

    T1W_PRECON = "t1w_precon"
    T1W_POSTCON = "t1w_postcon"
    T2W = "t2w"
    FLAIR_PRECON = "flair_precon"
    FLAIR_POSTCON = "flair_postcon"


# High-resolution threshold: geometric mean <= 1.5 mm
_HIGH_RES_THRESHOLD = 1.5

# Max single-voxel-dimension hard cutoff: any dim > 4 mm → POOR
_MAX_VOXEL_DIM_CUTOFF = 4.0

# Isotropy adjustment thresholds
_ISOTROPY_BONUS_THRESHOLD = 1.2   # Nearly isotropic → +5%
_ISOTROPY_PENALTY_THRESHOLD = 2.5  # Highly anisotropic → -10%

# FAST quality gate thresholds (hybrid FAST path)
_FAST_GATE_RESOLUTION = 2.0   # Max geometric mean (mm) for FAST viability
_FAST_GATE_ISOTROPY = 2.5     # Max isotropy ratio for FAST viability

# Physics-validated base confidence table
# (image_type, is_high_res) -> base confidence
_BASE_CONFIDENCE_TABLE: Dict[Tuple[ImageType, bool], float] = {
    # High-resolution (<=1.5mm geometric mean)
    (ImageType.T1W_PRECON, True): 1.0,
    (ImageType.T1W_POSTCON, True): 0.7,
    (ImageType.T2W, True): 0.7,
    (ImageType.FLAIR_PRECON, True): 0.65,
    (ImageType.FLAIR_POSTCON, True): 0.55,
    # Low-resolution (>1.5mm geometric mean)
    (ImageType.T1W_PRECON, False): 0.4,
    (ImageType.T1W_POSTCON, False): 0.3,
    (ImageType.T2W, False): 0.3,
    (ImageType.FLAIR_PRECON, False): 0.3,
    (ImageType.FLAIR_POSTCON, False): 0.2,
}

# Lesion mask confidence bonuses (replaces base value when mask available)
_LESION_MASK_OVERRIDES: Dict[Tuple[ImageType, bool], float] = {
    (ImageType.T1W_POSTCON, True): 0.9,   # BBR can exclude enhancing regions
    (ImageType.FLAIR_POSTCON, True): 0.70,  # Focal enhancement excluded
    (ImageType.T1W_POSTCON, False): 0.4,   # Low-res doesn't benefit much
    (ImageType.FLAIR_POSTCON, False): 0.25,
}


@dataclass
class StructuralCandidate:
    """A candidate structural image for reference selection.

    Attributes
    ----------
    path : Path
        Path to the NIfTI file.
    image_type : ImageType
        Classification of the image type.
    voxel_sizes : tuple
        Tuple of (x, y, z) voxel dimensions in mm.
    geometric_mean : float
        Geometric mean resolution.
    isotropy_ratio : float
        Anisotropy measure max(v)/min(v).
    is_high_res : bool
        Whether geometric mean <= 1.5mm.
    has_lesion_mask : bool
        Whether a lesion mask will be available.
    """

    path: Path
    image_type: ImageType
    voxel_sizes: Tuple[float, float, float]
    geometric_mean: float
    isotropy_ratio: float
    is_high_res: bool
    has_lesion_mask: bool = False


@dataclass
class StructuralReferenceSelection:
    """Result of confidence-based structural reference selection.

    Attributes
    ----------
    candidate : StructuralCandidate
        The selected candidate.
    confidence : float
        Final confidence score (0.0-1.0).
    application : ApplicationType
        Application context used for selection.
    bbr_eligible : bool
        Whether BBR registration is appropriate.
    all_candidates : list
        All candidates that were considered.
    selection_reason : str
        Human-readable explanation.
    coverage_fraction : float
        FOV overlap fraction (1.0 if not computed).
    """

    candidate: StructuralCandidate
    confidence: float
    application: ApplicationType
    bbr_eligible: bool
    all_candidates: List[StructuralCandidate]
    selection_reason: str
    coverage_fraction: float = 1.0

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Convert to a dict for JSON provenance/QC metadata."""
        return {
            "selected_image": str(self.candidate.path),
            "image_type": self.candidate.image_type.value,
            "confidence": round(self.confidence, 3),
            "application": self.application,
            "bbr_eligible": self.bbr_eligible,
            "geometric_mean_mm": round(self.candidate.geometric_mean, 3),
            "isotropy_ratio": round(self.candidate.isotropy_ratio, 3),
            "is_high_res": self.candidate.is_high_res,
            "has_lesion_mask": self.candidate.has_lesion_mask,
            "coverage_fraction": round(self.coverage_fraction, 3),
            "selection_reason": self.selection_reason,
            "n_candidates": len(self.all_candidates),
        }


def classify_image_type(path: Union[str, Path]) -> ImageType:
    """Classify an image into an ImageType based on BIDS filename.

    Uses the filename to determine modality (T1w/T2w/FLAIR) and contrast
    state (pre/post-contrast).

    Parameters
    ----------
    path : Path or str
        Path to the NIfTI file.

    Returns
    -------
    ImageType
        ImageType classification.
    """
    path = Path(path)
    fname = path.name.lower()

    is_postcon = is_postcontrast_image(path)

    if "_flair" in fname:
        return ImageType.FLAIR_POSTCON if is_postcon else ImageType.FLAIR_PRECON
    elif "_t2w" in fname:
        return ImageType.T2W
    else:
        # Default to T1w
        return ImageType.T1W_POSTCON if is_postcon else ImageType.T1W_PRECON


def _compute_base_confidence(candidate: StructuralCandidate) -> float:
    """Compute base confidence from the physics-validated lookup table.

    Parameters
    ----------
    candidate : StructuralCandidate
        Structural candidate with image type and resolution info.

    Returns
    -------
    float
        Base confidence score (0.0-1.0).
    """
    key = (candidate.image_type, candidate.is_high_res)

    # Apply lesion mask bonus for post-contrast images
    if candidate.has_lesion_mask and key in _LESION_MASK_OVERRIDES:
        return _LESION_MASK_OVERRIDES[key]

    return _BASE_CONFIDENCE_TABLE.get(key, 0.1)


def _apply_isotropy_adjustment(
    confidence: float,
    isotropy_ratio: float,
) -> float:
    """Adjust confidence based on voxel isotropy.

    Nearly isotropic images get a small bonus; highly anisotropic images
    get a penalty.

    Parameters
    ----------
    confidence : float
        Current confidence score.
    isotropy_ratio : float
        max(voxel)/min(voxel), >= 1.0.

    Returns
    -------
    float
        Adjusted confidence score, clamped to [0.0, 1.0].
    """
    if isotropy_ratio < _ISOTROPY_BONUS_THRESHOLD:
        # Nearly isotropic: +5%
        adjusted = confidence * 1.05
    elif isotropy_ratio > _ISOTROPY_PENALTY_THRESHOLD:
        # Highly anisotropic: -10%
        adjusted = confidence * 0.90
    else:
        adjusted = confidence

    return float(np.clip(adjusted, 0.0, 1.0))


def _apply_application_overrides(
    confidence: float,
    candidate: StructuralCandidate,
    application: ApplicationType,
    has_postcon_available: bool = False,
) -> float:
    """Apply application-specific confidence overrides.

    Parameters
    ----------
    confidence : float
        Current confidence score.
    candidate : StructuralCandidate
        The candidate being scored.
    application : ApplicationType
        Target application context.
    has_postcon_available : bool
        Whether any post-contrast image is available.

    Returns
    -------
    float
        Override-adjusted confidence, clamped to [0.0, 1.0].
    """
    img_type = candidate.image_type

    if application == "fast_segmentation":
        # FAST is trained on non-contrast T1w — exclude post-contrast
        if img_type == ImageType.T1W_POSTCON:
            return 0.0
        # Ensure precon T1w has minimum confidence for FAST
        if img_type == ImageType.T1W_PRECON:
            confidence = max(confidence, 0.7)

    elif application == "dce_registration":
        # DCE benefits from post-contrast T1w (matching contrast state)
        if img_type == ImageType.T1W_POSTCON:
            confidence = 0.98
        elif img_type == ImageType.T1W_PRECON and has_postcon_available:
            # Penalize precon when postcon is available
            confidence *= 0.8

    # dsc_registration, brain_extraction, default: no overrides

    return float(np.clip(confidence, 0.0, 1.0))


def _determine_bbr_eligibility(candidate: StructuralCandidate) -> bool:
    """Determine if BBR registration is appropriate for this candidate.

    BBR requires reliable WM/GM tissue boundaries. Post-contrast T1w
    has disrupted boundaries from gadolinium, making BBR unreliable
    unless a lesion mask is available (post-contrast FLAIR case).

    Parameters
    ----------
    candidate : StructuralCandidate
        The candidate to evaluate.

    Returns
    -------
    bool
        True if BBR registration is appropriate.
    """
    img_type = candidate.image_type

    if img_type in (ImageType.T1W_PRECON, ImageType.T2W, ImageType.FLAIR_PRECON):
        return True
    elif img_type == ImageType.T1W_POSTCON:
        # Post-contrast T1w: BBR unreliable due to Gd disrupting boundaries
        return False
    elif img_type == ImageType.FLAIR_POSTCON:
        # Post-contrast FLAIR: BBR OK if lesion mask excludes enhancement
        return candidate.has_lesion_mask

    return False


def compute_fov_coverage(
    structural_path: Union[str, Path],
    perfusion_path: Union[str, Path],
) -> float:
    """Compute FOV coverage: fraction of perfusion FOV covered by structural.

    Uses world-coordinate bounding boxes to compute the overlap ratio.

    Parameters
    ----------
    structural_path : Path or str
        Path to structural NIfTI.
    perfusion_path : Path or str
        Path to perfusion NIfTI.

    Returns
    -------
    float
        Coverage fraction (0.0-1.0). 1.0 means structural fully covers
        the perfusion FOV.
    """
    try:
        struct_img = nib.load(str(structural_path))
        perf_img = nib.load(str(perfusion_path))

        # Get world-coordinate bounding boxes from affine + shape
        def _get_world_bbox(
            img: nib.filebasedimages.FileBasedImage,
        ) -> Tuple[np.ndarray, np.ndarray]:
            shape = np.array(img.shape[:3])  # type: ignore[attr-defined]
            affine = img.affine  # type: ignore[attr-defined]
            # Corner voxel coordinates
            corners = np.array([
                [0, 0, 0],
                [shape[0] - 1, 0, 0],
                [0, shape[1] - 1, 0],
                [0, 0, shape[2] - 1],
                [shape[0] - 1, shape[1] - 1, 0],
                [shape[0] - 1, 0, shape[2] - 1],
                [0, shape[1] - 1, shape[2] - 1],
                [shape[0] - 1, shape[1] - 1, shape[2] - 1],
            ], dtype=float)
            # Transform to world coordinates
            world_corners = nib.affines.apply_affine(affine, corners)
            return world_corners.min(axis=0), world_corners.max(axis=0)

        struct_min, struct_max = _get_world_bbox(struct_img)
        perf_min, perf_max = _get_world_bbox(perf_img)

        # Compute overlap in each dimension
        overlap_min = np.maximum(struct_min, perf_min)
        overlap_max = np.minimum(struct_max, perf_max)
        overlap_extent = np.maximum(overlap_max - overlap_min, 0.0)

        perf_extent = np.maximum(perf_max - perf_min, 1e-9)

        # Coverage = overlap volume / perfusion volume
        coverage_per_dim = overlap_extent / perf_extent
        coverage = float(np.prod(coverage_per_dim))

        return float(np.clip(coverage, 0.0, 1.0))

    except Exception as exc:
        logger.warning("Could not compute FOV coverage: %s", exc)
        return 1.0  # Assume full coverage on error


def _score_candidate(
    candidate: StructuralCandidate,
    application: ApplicationType,
    has_postcon_available: bool = False,
) -> float:
    """Score a single candidate through the full confidence pipeline.

    Chains: base confidence -> isotropy adjustment -> application overrides.

    Parameters
    ----------
    candidate : StructuralCandidate
        Structural candidate to score.
    application : ApplicationType
        Target application context.
    has_postcon_available : bool
        Whether any post-contrast image is available.

    Returns
    -------
    float
        Final confidence score (0.0-1.0).
    """
    confidence = _compute_base_confidence(candidate)
    confidence = _apply_isotropy_adjustment(confidence, candidate.isotropy_ratio)
    confidence = _apply_application_overrides(
        confidence, candidate, application, has_postcon_available
    )
    return confidence


def build_candidates_from_session(
    modalities: Any,
    has_lesion_mask: bool = False,
) -> List[StructuralCandidate]:
    """Build candidate list from a ModalityAvailability object.

    Parameters
    ----------
    modalities : Any
        ModalityAvailability from discovery.
    has_lesion_mask : bool
        Whether a lesion mask will be available at
        perfusion registration time.

    Returns
    -------
    list
        List of StructuralCandidate objects for all available modalities.
    """
    candidates: List[StructuralCandidate] = []

    # Map modality fields to (path_attr, image_type) pairs
    modality_map: List[Tuple[str, ImageType]] = [
        ("t1w_pre", ImageType.T1W_PRECON),
        ("t1w_ce_gad", ImageType.T1W_POSTCON),
        ("t2w", ImageType.T2W),
    ]

    for attr, default_type in modality_map:
        path = getattr(modalities, attr, None)
        if path is None:
            continue

        path = Path(path)
        if not path.exists():
            continue

        try:
            info = get_image_resolution_info(path)
        except Exception:
            continue

        # Classify FLAIR separately (pre/post contrast)
        image_type = default_type

        voxel_sizes = info.voxel_sizes
        geometric_mean = info.geometric_mean
        isotropy_ratio = info.isotropy_ratio

        # Hard downgrade: any voxel dimension > 4mm → force low-res
        is_high_res = geometric_mean <= _HIGH_RES_THRESHOLD
        if max(voxel_sizes) > _MAX_VOXEL_DIM_CUTOFF:
            is_high_res = False

        candidates.append(StructuralCandidate(
            path=path,
            image_type=image_type,
            voxel_sizes=voxel_sizes,
            geometric_mean=geometric_mean,
            isotropy_ratio=isotropy_ratio,
            is_high_res=is_high_res,
            has_lesion_mask=has_lesion_mask,
        ))

    # Handle FLAIR pre-contrast (modalities.flair is strictly precon after discovery)
    flair_path = getattr(modalities, "flair", None)
    if flair_path is not None:
        flair_path = Path(flair_path)
        if flair_path.exists():
            try:
                info = get_image_resolution_info(flair_path)
                is_high_res = info.geometric_mean <= _HIGH_RES_THRESHOLD
                if max(info.voxel_sizes) > _MAX_VOXEL_DIM_CUTOFF:
                    is_high_res = False

                candidates.append(StructuralCandidate(
                    path=flair_path,
                    image_type=ImageType.FLAIR_PRECON,
                    voxel_sizes=info.voxel_sizes,
                    geometric_mean=info.geometric_mean,
                    isotropy_ratio=info.isotropy_ratio,
                    is_high_res=is_high_res,
                    has_lesion_mask=has_lesion_mask,
                ))
            except Exception:
                pass

    # Handle FLAIR post-contrast
    flair_postcon_path = getattr(modalities, "flair_postcon", None)
    if flair_postcon_path is not None:
        flair_postcon_path = Path(flair_postcon_path)
        if flair_postcon_path.exists():
            try:
                info = get_image_resolution_info(flair_postcon_path)
                is_high_res = info.geometric_mean <= _HIGH_RES_THRESHOLD
                if max(info.voxel_sizes) > _MAX_VOXEL_DIM_CUTOFF:
                    is_high_res = False

                candidates.append(StructuralCandidate(
                    path=flair_postcon_path,
                    image_type=ImageType.FLAIR_POSTCON,
                    voxel_sizes=info.voxel_sizes,
                    geometric_mean=info.geometric_mean,
                    isotropy_ratio=info.isotropy_ratio,
                    is_high_res=is_high_res,
                    has_lesion_mask=has_lesion_mask,
                ))
            except Exception:
                pass

    return candidates


def select_structural_reference_v2(
    candidates: List[StructuralCandidate],
    application: ApplicationType = "default",
    perfusion_path: Optional[Union[str, Path]] = None,
) -> Optional[StructuralReferenceSelection]:
    """Select the optimal structural reference using confidence scoring.

    Chains base confidence -> isotropy adjustment -> application overrides
    -> coverage check -> selects highest confidence.

    Parameters
    ----------
    candidates : list
        List of structural candidates.
    application : ApplicationType
        Target application context.
    perfusion_path : Path or str, optional
        Path to perfusion NIfTI for FOV coverage check.

    Returns
    -------
    StructuralReferenceSelection or None
        StructuralReferenceSelection, or None if no candidates.
    """
    if not candidates:
        return None

    # Check if any post-contrast image is available
    has_postcon = any(
        c.image_type in (ImageType.T1W_POSTCON, ImageType.FLAIR_POSTCON)
        for c in candidates
    )

    # Score each candidate
    scored: List[Tuple[float, StructuralCandidate]] = []
    for cand in candidates:
        conf = _score_candidate(cand, application, has_postcon)
        if conf > 0.0:
            scored.append((conf, cand))

    if not scored:
        return None

    # Sort by confidence (highest first)
    scored.sort(key=lambda x: -x[0])
    best_conf, best_cand = scored[0]

    # Compute FOV coverage if perfusion path provided
    coverage = 1.0
    if perfusion_path is not None:
        coverage = compute_fov_coverage(best_cand.path, perfusion_path)

    # Determine BBR eligibility
    bbr_eligible = _determine_bbr_eligibility(best_cand)

    # Build selection reason
    reason_parts = [
        f"{best_cand.image_type.value}",
        f"conf={best_conf:.2f}",
        f"res={best_cand.geometric_mean:.2f}mm",
    ]
    if best_cand.has_lesion_mask:
        reason_parts.append("lesion_mask=yes")
    if coverage < 1.0:
        reason_parts.append(f"coverage={coverage:.0%}")
    reason = ", ".join(reason_parts)

    logger.info(
        "Structural reference selected for %s: %s (%s)",
        application,
        best_cand.path.name,
        reason,
    )

    return StructuralReferenceSelection(
        candidate=best_cand,
        confidence=best_conf,
        application=application,
        bbr_eligible=bbr_eligible,
        all_candidates=candidates,
        selection_reason=reason,
        coverage_fraction=coverage,
    )


def _determine_processing_mode_v2(
    t1w_precon: Optional[Union[str, Path]],
    t1w_postcon: Optional[Union[str, Path]],
) -> Tuple[str, bool, Optional[str]]:
    """Determine reference T1w and processing mode using confidence scoring.

    Returns 3 values enabling the hybrid FAST path: when postcon wins as
    reference, FAST can still run on the precon if it meets a quality gate.

    Parameters
    ----------
    t1w_precon : Path or str, optional
        Path to T1w pre-contrast image, or None.
    t1w_postcon : Path or str, optional
        Path to T1w post-contrast image, or None.

    Returns
    -------
    tuple
        Tuple of:
            reference_t1w: Path to T1w image to use as reference.
            is_postcon_reference: True if postcon is the reference.
            fast_t1w: Path to T1w for FAST (precon if acceptable), or None.

    Raises
    ------
    ValueError
        If neither precon nor postcon T1w is available.
    """
    has_precon = t1w_precon is not None and Path(t1w_precon).exists()
    has_postcon = t1w_postcon is not None and Path(t1w_postcon).exists()

    if not has_precon and not has_postcon:
        raise ValueError("At least one T1w image (precon or postcon) is required")

    if not has_postcon:
        # Only precon: normal mode
        return str(t1w_precon), False, str(t1w_precon)

    if not has_precon:
        # Only postcon: no FAST possible
        return str(t1w_postcon), True, None

    # Both available — score them
    precon_info = get_image_resolution_info(Path(t1w_precon))  # type: ignore[arg-type]
    postcon_info = get_image_resolution_info(Path(t1w_postcon))  # type: ignore[arg-type]

    # Build candidates for brain_extraction application (T1w only)
    precon_is_high_res = precon_info.geometric_mean <= _HIGH_RES_THRESHOLD
    if max(precon_info.voxel_sizes) > _MAX_VOXEL_DIM_CUTOFF:
        precon_is_high_res = False

    postcon_is_high_res = postcon_info.geometric_mean <= _HIGH_RES_THRESHOLD
    if max(postcon_info.voxel_sizes) > _MAX_VOXEL_DIM_CUTOFF:
        postcon_is_high_res = False

    precon_cand = StructuralCandidate(
        path=Path(t1w_precon),  # type: ignore[arg-type]
        image_type=ImageType.T1W_PRECON,
        voxel_sizes=precon_info.voxel_sizes,
        geometric_mean=precon_info.geometric_mean,
        isotropy_ratio=precon_info.isotropy_ratio,
        is_high_res=precon_is_high_res,
    )

    postcon_cand = StructuralCandidate(
        path=Path(t1w_postcon),  # type: ignore[arg-type]
        image_type=ImageType.T1W_POSTCON,
        voxel_sizes=postcon_info.voxel_sizes,
        geometric_mean=postcon_info.geometric_mean,
        isotropy_ratio=postcon_info.isotropy_ratio,
        is_high_res=postcon_is_high_res,
    )

    precon_conf = _score_candidate(precon_cand, "brain_extraction")
    postcon_conf = _score_candidate(postcon_cand, "brain_extraction")

    # Quality-dependent FAST penalty for postcon
    # Better precon → higher penalty (less reason to use postcon)
    precon_tier = classify_resolution(precon_info.geometric_mean)
    precon_acceptable_quality = (
        precon_is_high_res
        and precon_info.isotropy_ratio <= 2.0
    )

    if precon_acceptable_quality and precon_tier in (
        ResolutionTier.EXCELLENT, ResolutionTier.GOOD,
    ):
        fast_penalty = 0.3  # Precon is GOOD/EXCELLENT
    elif precon_acceptable_quality:
        fast_penalty = 0.2  # Precon is ACCEPTABLE
    else:
        fast_penalty = 0.1  # Precon is POOR or highly anisotropic

    # Postcon wins only if its confidence minus penalty exceeds precon
    postcon_wins = (postcon_conf - fast_penalty) > precon_conf

    if postcon_wins:
        # Determine if precon is good enough for hybrid FAST
        precon_meets_fast_gate = (
            precon_info.geometric_mean <= _FAST_GATE_RESOLUTION
            and precon_info.isotropy_ratio <= _FAST_GATE_ISOTROPY
        )
        fast_t1w = str(t1w_precon) if precon_meets_fast_gate else None

        logger.info(
            "Postcon T1w selected as reference (conf=%.2f, penalty=%.2f > "
            "precon conf=%.2f). FAST on %s.",
            postcon_conf, fast_penalty, precon_conf,
            "precon (hybrid)" if fast_t1w else "skipped",
        )

        return str(t1w_postcon), True, fast_t1w
    else:
        # Precon wins: normal mode
        logger.info(
            "Precon T1w selected as reference (conf=%.2f vs postcon "
            "conf=%.2f - penalty=%.2f).",
            precon_conf, postcon_conf, fast_penalty,
        )
        return str(t1w_precon), False, str(t1w_precon)


def _map_selection_to_anat_output(
    image_type: ImageType,
) -> str:
    """Map an ImageType to the corresponding anat workflow output field.

    Parameters
    ----------
    image_type : ImageType
        The selected image type.

    Returns
    -------
    str
        Anat workflow output field name (e.g., "outputnode.t1w_preproc").
    """
    mapping = {
        ImageType.T1W_PRECON: "outputnode.t1w_preproc",
        ImageType.T1W_POSTCON: "outputnode.t1w_postcon_coreg",
        ImageType.T2W: "outputnode.t2w_coreg",
        ImageType.FLAIR_PRECON: "outputnode.flair_coreg",
        ImageType.FLAIR_POSTCON: "outputnode.flair_postcon_coreg",
    }
    return mapping.get(image_type, "outputnode.t1w_preproc")
