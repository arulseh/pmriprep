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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np


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

    Attributes:
        path: Path to the NIfTI file.
        voxel_sizes: Tuple of (x, y, z) voxel dimensions in mm.
        geometric_mean: Geometric mean resolution (x*y*z)^(1/3).
        isotropy_ratio: Anisotropy measure max(v)/min(v).
        resolution_tier: Quality classification.
        is_postcontrast: Whether this is a post-contrast image.
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
    "brain_extraction",
    "default",
]


def get_geometric_mean_resolution(
    voxel_sizes: Tuple[float, float, float],
) -> float:
    """Calculate geometric mean resolution from voxel sizes.

    The geometric mean provides a single scalar representing the "effective"
    isotropic resolution of an anisotropic image.

    Args:
        voxel_sizes: Tuple of (voxel_x, voxel_y, voxel_z) in mm.

    Returns:
        Geometric mean resolution: (Vx × Vy × Vz)^(1/3) in mm.

    Example:
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

    Args:
        voxel_sizes: Tuple of (voxel_x, voxel_y, voxel_z) in mm.

    Returns:
        Isotropy ratio: max(V)/min(V), always >= 1.0.

    Example:
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

    Args:
        geometric_mean: Geometric mean resolution in mm.

    Returns:
        ResolutionTier classification.

    Example:
        >>> classify_resolution(0.8)
        ResolutionTier.EXCELLENT
        >>> classify_resolution(1.2)
        ResolutionTier.GOOD
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

    Args:
        path: Path to the NIfTI file.

    Returns:
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

    Args:
        path: Path to the NIfTI file.

    Returns:
        ImageResolutionInfo with all resolution metrics.

    Raises:
        FileNotFoundError: If the file does not exist.
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


def _score_image(
    info: ImageResolutionInfo,
    application: ApplicationType,
) -> float:
    """Calculate a score for image selection (lower is better).

    The score combines geometric mean resolution with penalties for:
    - High anisotropy (isotropy ratio > thresholds)
    - Application-specific constraints

    Args:
        info: ImageResolutionInfo for the candidate image.
        application: The intended use case.

    Returns:
        Selection score (lower is better). Returns inf for excluded images.
    """
    # Base score is geometric mean resolution
    score = info.geometric_mean

    # Penalize anisotropic images
    if info.isotropy_ratio > ISOTROPY_POOR:
        score += 1.0  # Large penalty
    elif info.isotropy_ratio > ISOTROPY_ACCEPTABLE:
        score += 0.3  # Moderate penalty
    elif info.isotropy_ratio > ISOTROPY_EXCELLENT:
        score += 0.1  # Small penalty

    # Application-specific adjustments
    if application == "fast_segmentation":
        # FSL FAST is trained on non-contrast T1w
        # Exclude post-contrast images
        if info.is_postcontrast:
            return float("inf")

    elif application == "dce_registration":
        # DCE registration benefits from post-contrast T1w
        # Give bonus (reduce score) for post-contrast
        if info.is_postcontrast:
            score -= 0.2  # Bonus for matching contrast

    # brain_extraction and default have no special preferences

    return score


def select_structural_reference(
    candidates: Union[Sequence[Union[str, Path]], Mapping[str, Union[str, Path]]],
    application: ApplicationType = "default",
    require_tier: Optional[ResolutionTier] = None,
) -> Optional[ImageResolutionInfo]:
    """Select the optimal structural reference image from candidates.

    Selection is based on:
    1. Resolution quality (geometric mean - lower is better)
    2. Isotropy (more isotropic preferred)
    3. Application-specific requirements

    Application-specific rules:
    - fast_segmentation: Excludes post-contrast T1w (FAST trained on non-enhanced)
    - dce_registration: Prefers post-contrast T1w (matches DCE contrast)
    - brain_extraction: No special preference, just best resolution
    - default: No special preference, just best resolution

    Args:
        candidates: Either a list of paths, or a dict mapping names to paths.
            Examples:
            - ["/path/to/T1w.nii.gz", "/path/to/T1w_ce-GAD.nii.gz"]
            - {"T1w": "/path/T1w.nii.gz", "T1w_postcon": "/path/T1w_ce-GAD.nii.gz"}
        application: The intended use case for reference selection.
        require_tier: Optional minimum resolution tier requirement.
            If specified, images below this tier are excluded.

    Returns:
        ImageResolutionInfo for the selected image, or None if no suitable
        candidate is found.

    Example:
        >>> candidates = ["/data/sub-01/anat/sub-01_T1w.nii.gz",
        ...               "/data/sub-01/anat/sub-01_ce-GAD_T1w.nii.gz"]
        >>> selected = select_structural_reference(candidates, "fast_segmentation")
        >>> selected.path  # Will be the non-contrast T1w
    """
    # Normalize input to list of paths
    if isinstance(candidates, dict):
        paths = [Path(p) for p in candidates.values()]
    else:
        paths = [Path(p) for p in candidates]

    # Filter to existing files and get resolution info
    candidate_infos: List[ImageResolutionInfo] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            info = get_image_resolution_info(path)
            candidate_infos.append(info)
        except Exception:
            # Skip files that can't be read
            continue

    if not candidate_infos:
        return None

    # Apply tier requirement filter if specified
    if require_tier is not None:
        # Define tier order for comparison
        tier_order = [
            ResolutionTier.EXCELLENT,
            ResolutionTier.GOOD,
            ResolutionTier.ACCEPTABLE,
            ResolutionTier.POOR,
        ]
        required_idx = tier_order.index(require_tier)
        candidate_infos = [
            info
            for info in candidate_infos
            if tier_order.index(info.resolution_tier) <= required_idx
        ]

    if not candidate_infos:
        return None

    # Score each candidate
    scored: List[Tuple[float, ImageResolutionInfo]] = []
    for info in candidate_infos:
        score = _score_image(info, application)
        if score != float("inf"):
            scored.append((score, info))

    if not scored:
        return None

    # Sort by score (lower is better) and return best
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def select_structural_reference_from_session(
    anat_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
    application: ApplicationType = "default",
    require_tier: Optional[ResolutionTier] = None,
) -> Optional[ImageResolutionInfo]:
    """Select the optimal structural reference from a BIDS anat directory.

    Automatically discovers T1w images in the session's anatomical directory
    and selects the best one based on resolution and application requirements.

    Args:
        anat_dir: Path to the BIDS anat directory.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.
        application: The intended use case for reference selection.
        require_tier: Optional minimum resolution tier requirement.

    Returns:
        ImageResolutionInfo for the selected image, or None if no suitable
        T1w is found.
    """
    anat_dir = Path(anat_dir)

    # Build pattern for T1w images
    if session:
        pattern = f"sub-{subject}_ses-{session}*_T1w.nii*"
    else:
        pattern = f"sub-{subject}*_T1w.nii*"

    # Find all T1w images
    candidates = list(anat_dir.glob(pattern))

    if not candidates:
        return None

    return select_structural_reference(
        candidates,
        application=application,
        require_tier=require_tier,
    )


def rank_structural_candidates(
    candidates: Union[Sequence[Union[str, Path]], Mapping[str, Union[str, Path]]],
    application: ApplicationType = "default",
) -> List[ImageResolutionInfo]:
    """Rank all structural candidates by suitability.

    Unlike select_structural_reference which returns only the best,
    this function returns all candidates ranked from best to worst,
    allowing the caller to see alternatives.

    Args:
        candidates: Either a list of paths, or a dict mapping names to paths.
        application: The intended use case for reference selection.

    Returns:
        List of ImageResolutionInfo objects, sorted from best to worst.
        Excluded images (e.g., post-contrast for FAST) are omitted.
    """
    # Normalize input to list of paths
    if isinstance(candidates, dict):
        paths = [Path(p) for p in candidates.values()]
    else:
        paths = [Path(p) for p in candidates]

    # Get resolution info for all existing files
    candidate_infos: List[ImageResolutionInfo] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            info = get_image_resolution_info(path)
            candidate_infos.append(info)
        except Exception:
            continue

    # Score and filter
    scored: List[Tuple[float, ImageResolutionInfo]] = []
    for info in candidate_infos:
        score = _score_image(info, application)
        if score != float("inf"):
            scored.append((score, info))

    # Sort by score (lower is better)
    scored.sort(key=lambda x: x[0])

    return [info for _, info in scored]
