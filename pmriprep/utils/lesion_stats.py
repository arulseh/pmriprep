"""Lesion/tumor ROI statistics extraction utilities.

This module provides functions for extracting perfusion statistics from
lesion/tumor regions of interest (ROIs), supporting both DSC and DCE
perfusion MRI analysis.

Key features:
- Multi-label mask handling (enhancing vs non-enhancing tumor)
- Comprehensive statistics (mean, std, percentiles, CV, IQR)
- Fractional volume calculations (voxels above threshold)
- NAWM reference value extraction
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

import pmriprep
from pmriprep.config.constants import HDGLIO_DSEG_ROWS

logger = logging.getLogger(__name__)


@dataclass
class MapStatistics:
    """Statistics for a single perfusion map within a ROI.

    Attributes
    ----------
        mean: Mean value within ROI.
        std: Standard deviation within ROI.
        median: Median (50th percentile) value.
        min: Minimum value.
        max: Maximum value.
        p10: 10th percentile value.
        p25: 25th percentile (Q1) value.
        p75: 75th percentile (Q3) value.
        p90: 90th percentile value.
        p95: 95th percentile value.
        p98: 98th percentile (robust hotspot).
        p99: 99th percentile (max-robust hotspot).
        cv: Coefficient of variation (std/mean).
        iqr: Interquartile range (p75 - p25).
        skewness: Histogram skewness.
        kurtosis: Histogram kurtosis.
        entropy: Shannon entropy on 64-bin histogram over [p1, p99].
        cohens_d_vs_nawm: Pooled-SD Cohen's d of region vs NAWM reference;
            populated post-hoc by ``LesionStatisticsResult.augment_with_cohens_d``.
        n_valid: Count of finite values used in statistics.
        n_total: Count of all voxels in mask.
    """

    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p10: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p98: Optional[float] = None
    p99: Optional[float] = None
    cv: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    entropy: Optional[float] = None
    cohens_d_vs_nawm: Optional[float] = None
    n_valid: int = 0
    n_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class RegionStatistics:
    """Statistics for all perfusion maps within a single region.

    Attributes
    ----------
        region_name: Name of the region (e.g., "contrast_enhancing_tumor").
        voxel_count: Number of voxels in the region.
        volume_mm3: Volume in cubic millimeters.
        volume_ml: Volume in milliliters.
        perfusion_maps: Dictionary mapping map name to its statistics.
    """

    region_name: str
    voxel_count: int
    volume_mm3: float
    volume_ml: float
    perfusion_maps: Dict[str, MapStatistics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "voxel_count": self.voxel_count,
            "volume_mm3": round(self.volume_mm3, 2),
            "volume_ml": round(self.volume_ml, 3),
            "perfusion_maps": {k: v.to_dict() for k, v in self.perfusion_maps.items()},
        }


@dataclass
class LesionStatisticsResult:
    """Complete lesion statistics result for a subject/session.

    Attributes
    ----------
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session_id: BIDS session ID (without 'ses-' prefix).
        modality: Perfusion modality ("DSC" or "DCE").
        mask_type: Type of lesion mask used.
        mask_source: Path to the lesion mask file.
        total_voxels: Total number of lesion voxels.
        regions: Dictionary mapping region name to its statistics.
        nawm_reference: NAWM reference values for normalization context.
        derived_metrics: Derived metrics (fractional volumes).
    """

    subject_id: str
    session_id: str
    modality: str  # "DSC" or "DCE"
    mask_type: str  # "multi_label_tumor_seg", "separate_binary", "binary"
    mask_source: str
    total_voxels: int
    regions: Dict[str, RegionStatistics] = field(default_factory=dict)
    nawm_reference: Optional[Dict[str, Any]] = None
    derived_metrics: Optional[Dict[str, Any]] = None
    # AIF provenance for DSC stats: a prebuilt ``PerfusionProcessing.AIF``
    # sub-block (Source, DetectionLevel, QualityFlag, GammaFitApplied,
    # CrossSubjectComparable, optionally SyntheticFallbackReason /
    # ExternalAIFVerified). None for DCE stats or when AIF QC is
    # unavailable.
    aif_provenance: Optional[Dict[str, Any]] = None

    def augment_with_cohens_d(self) -> None:
        """Attach pooled-SD Cohen's d vs NAWM to each region's per-map stats.

        For every (region, map) pair where the corresponding NAWM
        reference entry is available, computes Cohen's d from the group means
        and sample-size-weighted within-group variance. Stored map standard
        deviations use the population convention, so their equivalent sums of
        squares are ``n * std**2``::

            pooled_var = (n_region * region.std² + n_nawm * nawm.std²)
                         / (n_region + n_nawm - 2)
            d = (region.mean - nawm.mean) / sqrt(pooled_var)

        and stores it on the region's ``MapStatistics.cohens_d_vs_nawm``.
        Mutates ``self.regions`` in place. No-op when NAWM reference is
        empty or doesn't carry the matching map.

        This is pooled-SD Cohen's d, not Glass's delta. It is a descriptive
        voxel-distribution effect size; spatial voxels are not independent
        samples.
        """
        if not self.nawm_reference:
            return
        for region in self.regions.values():
            for map_name, map_stats in region.perfusion_maps.items():
                nawm_entry = self.nawm_reference.get(map_name)
                if not isinstance(nawm_entry, dict):
                    continue
                nawm_mean = nawm_entry.get("mean")
                nawm_std = nawm_entry.get("std")
                nawm_n = nawm_entry.get("n_valid")
                if (
                    nawm_mean is None
                    or nawm_std is None
                    or nawm_n is None
                    or map_stats.mean is None
                    or map_stats.std is None
                    or map_stats.n_valid < 2
                    or int(nawm_n) < 2
                ):
                    continue
                pooled_var = (
                    map_stats.n_valid * map_stats.std ** 2
                    + int(nawm_n) * float(nawm_std) ** 2
                ) / (map_stats.n_valid + int(nawm_n) - 2)
                if pooled_var <= 0:
                    continue
                d = (map_stats.mean - float(nawm_mean)) / float(np.sqrt(pooled_var))
                map_stats.cohens_d_vs_nawm = round(float(d), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {
            "Description": f"Lesion ROI perfusion statistics extracted from {self.modality}-MRI",
            "GeneratedBy": [{"Name": "pmriprep", "Version": pmriprep.__version__}],
            "GeneratedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Subject": f"sub-{self.subject_id}",
            "Session": f"ses-{self.session_id}" if self.session_id else "",
            "LesionMask": self.mask_source,
            "LesionMaskType": self.mask_type,
            "TotalLesionVoxels": self.total_voxels,
        }
        if self.aif_provenance:
            result["PerfusionProcessing"] = {"AIF": self.aif_provenance}
        if self.nawm_reference:
            result["NAWMReference"] = self.nawm_reference
        result["Regions"] = {k: v.to_dict() for k, v in self.regions.items()}
        if self.derived_metrics:
            result["DerivedMetrics"] = self.derived_metrics
        return result


def _safe_skewness(values: np.ndarray) -> Optional[float]:
    """Return distribution skewness, or None if undefined."""
    if values.size < 3:
        return None
    std = float(np.std(values))
    if std == 0:
        return None
    centered = values - float(np.mean(values))
    return round(float(np.mean(centered ** 3) / (std ** 3)), 6)


def _safe_kurtosis(values: np.ndarray) -> Optional[float]:
    """Return excess kurtosis (Fisher's definition: normal → 0), or None."""
    if values.size < 4:
        return None
    std = float(np.std(values))
    if std == 0:
        return None
    centered = values - float(np.mean(values))
    return round(float(np.mean(centered ** 4) / (std ** 4) - 3.0), 6)


def _safe_shannon_entropy(values: np.ndarray, n_bins: int = 64) -> Optional[float]:
    """Shannon entropy on a 64-bin histogram over [p1, p99].

    Returns None for constant / near-constant data where the histogram
    range collapses.
    """
    if values.size < n_bins:
        return None
    p1 = float(np.percentile(values, 1))
    p99 = float(np.percentile(values, 99))
    if p99 - p1 <= 0:
        return None
    hist, _ = np.histogram(values, bins=n_bins, range=(p1, p99))
    total = hist.sum()
    if total == 0:
        return None
    probs = hist[hist > 0] / total
    return round(float(-np.sum(probs * np.log2(probs))), 6)


def compute_map_statistics(
    map_data: np.ndarray,
    mask: np.ndarray,
    compute_advanced: bool = True,
) -> Optional[MapStatistics]:
    """Compute comprehensive statistics for a perfusion map within a mask.

    Parameters
    ----------
    map_data : np.ndarray
        3D array of perfusion map values.
    mask : np.ndarray
        3D boolean array defining the ROI.
    compute_advanced : bool, optional
        When True, also compute p98, p99, skewness,
        kurtosis, entropy. Disable for NAWM-reference computation
        where these histogram metrics add noise on small ROIs and
        aren't used downstream.

    Returns
    -------
    MapStatistics or None
        MapStatistics object or None if mask is empty.
        When mask is non-empty but all values are invalid, returns a
        ``MapStatistics`` with ``n_valid=0`` so callers can distinguish
        from empty-mask (None).
        Advanced histogram metrics (skewness, kurtosis, entropy) are
        only computed when ``n_valid >= 10`` to avoid unstable values
        on tiny ROIs.
    """
    # Extract values within mask
    values = map_data[mask]
    if len(values) == 0:
        return None

    n_total = len(values)

    # Filter NaN and Inf values
    values = values[np.isfinite(values)]
    n_valid = len(values)

    if n_valid == 0:
        return MapStatistics(n_valid=0, n_total=n_total)

    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))

    # Coefficient of variation (avoid division by zero)
    cv = None
    if mean_val != 0 and not np.isnan(mean_val):
        cv = round(std_val / abs(mean_val), 4)

    stats = MapStatistics(
        mean=round(mean_val, 6),
        std=round(std_val, 6),
        median=round(float(np.median(values)), 6),
        min=round(float(np.min(values)), 6),
        max=round(float(np.max(values)), 6),
        p10=round(float(np.percentile(values, 10)), 6),
        p25=round(p25, 6),
        p75=round(p75, 6),
        p90=round(float(np.percentile(values, 90)), 6),
        p95=round(float(np.percentile(values, 95)), 6),
        cv=cv,
        iqr=round(p75 - p25, 6),
        n_valid=n_valid,
        n_total=n_total,
    )

    if compute_advanced and n_valid >= 10:
        stats.p98 = round(float(np.percentile(values, 98)), 6)
        stats.p99 = round(float(np.percentile(values, 99)), 6)
        stats.skewness = _safe_skewness(values)
        stats.kurtosis = _safe_kurtosis(values)
        stats.entropy = _safe_shannon_entropy(values)

    return stats


def compute_fraction_above_threshold(
    map_data: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> Optional[float]:
    """Compute fraction of voxels above a threshold within mask.

    This is used for descriptive fractional-volume calculations over a
    predefined value grid.

    Parameters
    ----------
    map_data : np.ndarray
        3D array of perfusion map values.
    mask : np.ndarray
        3D boolean array defining the ROI.
    threshold : float
        Value threshold.

    Returns
    -------
    float or None
        Fraction (0-1) of voxels above threshold, or None if mask empty.
    """
    values = map_data[mask]
    if len(values) == 0:
        return None

    # Filter NaN and Inf values
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    fraction = float(np.sum(values > threshold) / len(values))
    return round(fraction, 4)


def compute_hotspot_value(
    map_data: np.ndarray,
    mask: np.ndarray,
    percentile: float = 90.0,
) -> Optional[float]:
    """Compute hotspot value (high percentile) for a map within mask.

    Hotspot values represent the most aggressive tumor regions and are
    clinically relevant for treatment planning.

    Parameters
    ----------
    map_data : np.ndarray
        3D array of perfusion map values.
    mask : np.ndarray
        3D boolean array defining the ROI.
    percentile : float, optional
        Percentile to compute (default 90th).

    Returns
    -------
    float or None
        Hotspot value or None if mask empty.
    """
    values = map_data[mask]
    if len(values) == 0:
        return None

    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    return round(float(np.percentile(values, percentile)), 6)


def detect_mask_type(
    mask_data: np.ndarray,
) -> Tuple[str, int]:
    """Detect the type of lesion mask from its label values.

    Parameters
    ----------
    mask_data : np.ndarray
        3D array of mask values.

    Returns
    -------
    tuple
        Tuple of (mask_type, max_label):
            - mask_type: "multi_label_tumor_seg" if max >= 2, else "binary"
            - max_label: Maximum label value in mask
    """
    max_label = int(np.max(mask_data))
    if max_label >= 2:
        return "multi_label_tumor_seg", max_label
    return "binary", max_label


def extract_label_mask(
    mask_data: np.ndarray,
    label: int,
) -> np.ndarray:
    """Extract a binary mask for a specific label value.

    Parameters
    ----------
    mask_data : np.ndarray
        3D array of mask values (multi-label).
    label : int
        Label value to extract.

    Returns
    -------
    np.ndarray
        Boolean 3D array where mask_data == label.
    """
    return mask_data == label  # type: ignore[no-any-return]


def validate_mask_ndim(
    mask_path: Union[Path, str],
    source_name: str = "mask",
) -> None:
    """Validate that a user-supplied mask file is 3D.

    Parameters
    ----------
    mask_path : Path or str
        Path to the NIfTI mask file.
    source_name : str
        Descriptive name for error messages.

    Raises
    ------
    ValueError
        If mask is not 3D.
    """
    mask_path = Path(mask_path)
    img = nib.load(mask_path)
    shape = img.header.get_data_shape()  # type: ignore[attr-defined]
    ndim = len(shape)
    if ndim != 3:
        raise ValueError(
            f"{source_name} ({mask_path.name}): expected a 3D mask, "
            f"got {ndim}D (shape {shape})"
        )


def is_integer_valued(arr: np.ndarray) -> bool:
    """Check whether a numeric array contains only integer values.

    Parameters
    ----------
    arr : ndarray
        Array to check.

    Returns
    -------
    bool
        True if all finite nonzero values are integers.
    """
    if np.issubdtype(arr.dtype, np.integer):
        return True
    finite = arr[np.isfinite(arr) & (arr != 0)]
    if len(finite) == 0:
        return True
    return bool(np.all(finite == np.round(finite)))


def compute_region_volume(
    mask: np.ndarray,
    voxel_sizes: Tuple[float, float, float],
) -> Tuple[int, float, float]:
    """Compute voxel count and volume for a mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean 3D array.
    voxel_sizes : tuple of float
        Voxel dimensions in mm (x, y, z).

    Returns
    -------
    tuple
        Tuple of (voxel_count, volume_mm3, volume_ml).
    """
    voxel_count = int(np.sum(mask))
    voxel_volume_mm3 = float(np.prod(voxel_sizes))
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0
    return voxel_count, volume_mm3, volume_ml


# Per-map fractional-volume thresholds. Each map key may have multiple
# thresholds; output is keyed by ``above_<threshold>`` (e.g., ``above_2.0``).
DSC_FRACTIONAL_THRESHOLDS: Dict[str, Tuple[float, ...]] = {
    "ncbv": (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0),
    "ncbf": (1.5, 2.0, 3.0),
}
DCE_FRACTIONAL_THRESHOLDS: Dict[str, Tuple[float, ...]] = {
    "ktrans": (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
    "vp": (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10),
}


def _per_map_fractional_volumes(
    maps: Dict[str, Optional[np.ndarray]],
    thresholds_table: Dict[str, Tuple[float, ...]],
    mask: np.ndarray,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute fractional volumes per-map from a thresholds registry.

    Returns a dict ``{map_name: {"above_<t>": fraction, ...}}`` for every
    map present in ``maps`` that has a threshold registry entry and at
    least one finite value above any threshold.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for map_name, thresholds in thresholds_table.items():
        data = maps.get(map_name)
        if data is None:
            continue
        per_map: Dict[str, Optional[float]] = {}
        any_value = False
        for t in thresholds:
            f = compute_fraction_above_threshold(data, mask, t)
            if f is not None:
                any_value = True
            # Use a plain decimal in the key (e.g., "above_2.0"); keys with
            # dots are valid JSON and round-trip through Python/JS without
            # issue.
            per_map[f"above_{t}"] = f
        if any_value:
            out[map_name] = per_map
    return out


def compute_dsc_derived_metrics(
    ncbv_data: Optional[np.ndarray],
    ncbf_data: Optional[np.ndarray],
    cbv_data: Optional[np.ndarray],
    cbf_data: Optional[np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Compute DSC-specific derived metrics.

    Returns a dict with ``fractional_volumes`` keyed per-map (one sub-dict per
    available map). Per-map ``Regions.<r>.stats.p98`` contains each map's
    hotspot statistic.
    """
    derived: Dict[str, Any] = {}
    fractions = _per_map_fractional_volumes(
        {"ncbv": ncbv_data, "ncbf": ncbf_data, "cbv": cbv_data, "cbf": cbf_data},
        DSC_FRACTIONAL_THRESHOLDS,
        mask,
    )
    if fractions:
        derived["fractional_volumes"] = fractions
    return derived


def compute_dce_derived_metrics(
    ktrans_data: Optional[np.ndarray],
    ve_data: Optional[np.ndarray],
    vp_data: Optional[np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Compute DCE-specific derived metrics.

    Returns a dict with ``fractional_volumes`` keyed per-map (one
    sub-dict per available map).
    """
    derived: Dict[str, Any] = {}
    fractions = _per_map_fractional_volumes(
        {"ktrans": ktrans_data, "ve": ve_data, "vp": vp_data},
        DCE_FRACTIONAL_THRESHOLDS,
        mask,
    )
    if fractions:
        derived["fractional_volumes"] = fractions
    return derived


def combine_tumor_masks(
    tumor_segmentation_file: str | None = None,
    whole_tumor_mask_file: str | None = None,
    reference_file: str | None = None,
) -> tuple[str, str]:
    """Select the best tumor mask for lesion statistics.

    Priority:

    1. If tumor segmentation present: use it (already multi-label:
       0=bg, 1=non-enhancing, 2=enhancing)
    2. If only whole_tumor present: use as binary mask
    3. If none present: return empty-string sentinels

    Parameters
    ----------
    tumor_segmentation_file : str, optional
        Path to multi-label tumor segmentation (or None).
    whole_tumor_mask_file : str, optional
        Path to whole tumor binary mask (or None).
    reference_file : str, optional
        Path to reference image for output geometry (unused but
        kept for interface compatibility).

    Returns
    -------
    tuple
        Tuple of (mask_file_path, mask_origin) where mask_origin is
        "tumor_segmentation" or "whole_tumor" or "" if no masks available.
        Returns ("", "") when no masks are available because downstream
        ``traits.Str`` fields require strings.
    """
    from pathlib import Path

    from nipype.interfaces.base import isdefined

    def _is_valid_file(f: str | None) -> bool:
        """Check if file path is valid (not None, not Undefined, exists)."""
        if f is None:
            return False
        if not isdefined(f):
            return False
        return Path(f).exists()

    if _is_valid_file(tumor_segmentation_file):
        return (str(tumor_segmentation_file), "tumor_segmentation")

    if _is_valid_file(whole_tumor_mask_file):
        return (str(whole_tumor_mask_file), "whole_tumor")

    return ("", "")


# Per-map NAWM reference ranges for quantities with interpretable scales.
# PMRIPrep's DSC CBF and CBV maps are relative, so absolute physiological
# ranges in mL/100 g(/min) must not be applied to them. NAWM-normalized maps,
# time parameters, and DCE fractions/rates retain meaningful scales.
NAWM_RANGES: Dict[str, Tuple[float, float]] = {
    "mtt":    (3.0, 8.0),       # s (field-independent)
    "ncbv":   (0.85, 1.15),     # normalization sanity check
    "ncbf":   (0.85, 1.15),
    "ve":     (0.05, 0.30),
    "vp":     (0.005, 0.05),
    "ktrans": (0.0, 0.02),      # NAWM intact BBB
    "Ktrans": (0.0, 0.02),
}


def compute_nawm_reference(
    nawm_mask: np.ndarray,
    maps: Dict[str, np.ndarray],
    voxel_sizes: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Compute NAWM reference values for normalization context.

    Carries two distinct QC fields:
        - ``mask_qc_status`` describes the NAWM mask itself (validity,
          voxel count). Same value on every map's downstream slice.
        - per-map ``value_qc_status`` reports whether THAT map's NAWM
          median falls within its registered reference range. Stored as
          ``<map>_value_qc_status`` keys
          alongside the per-map sub-dict so that
          ``extract_per_map_lesion_stats`` can slice cleanly.

    Parameters
    ----------
    nawm_mask : np.ndarray
        Boolean NAWM mask.
    maps : dict
        Dictionary mapping map name to 3D array.
    voxel_sizes : tuple of float
        Voxel dimensions in mm.

    Returns
    -------
    dict
        Dictionary with NAWM statistics for each available map plus
        the QC fields described above.
    """
    voxel_count, volume_mm3, volume_ml = compute_region_volume(nawm_mask, voxel_sizes)

    if voxel_count == 0:
        return {}

    reference: Dict[str, Any] = {
        "voxel_count": voxel_count,
        "volume_ml": round(volume_ml, 2),
    }

    for map_name, map_data in maps.items():
        if map_data is None:
            continue
        stats = compute_map_statistics(
            map_data, nawm_mask, compute_advanced=False,
        )
        if stats is not None and stats.n_valid > 0:
            reference[map_name] = {
                "mean": stats.mean,
                "median": stats.median,
                "std": stats.std,
                "p25": stats.p25,
                "p75": stats.p75,
                "n_valid": stats.n_valid,
                "n_total": stats.n_total,
            }

    # No mask-size editorializing here. The NAWM mask may come from the
    # upstream auto-NAWM builder (which enforces config.tumor_seg.nawm_min_voxels
    # = 500 by default), the bilateral fallback, or a user-supplied mask in
    # the force_binary ROI path. Each has its own validity contract; consumers
    # can apply their own size threshold using the emitted ``voxel_count`` and
    # ``volume_ml`` fields.

    # Value-level QC: only maps with interpretable scales are registered.
    violations: List[str] = []
    n_maps_checked = 0
    for map_name, sub in list(reference.items()):
        if not isinstance(sub, dict) or "median" not in sub:
            continue
        rng = NAWM_RANGES.get(map_name)
        if rng is None:
            continue
        lo, hi = rng
        median = sub.get("median")
        if median is None:
            continue
        n_maps_checked += 1
        in_range = lo <= median <= hi
        sub["value_qc_status"] = "PASS" if in_range else "WARNING"
        if not in_range:
            violations.append(f"{map_name}={median} outside [{lo}, {hi}]")
    if n_maps_checked == 0:
        reference["value_qc_status"] = "UNKNOWN"
        reference["value_qc_notes"] = (
            "No emitted maps have registered NAWM reference ranges; "
            "range QC not applied."
        )
    elif violations:
        reference["value_qc_status"] = "WARNING"
        reference["value_qc_notes"] = (
            "Out-of-range NAWM medians: " + "; ".join(violations)
        )
    else:
        reference["value_qc_status"] = "PASS"
        reference["value_qc_notes"] = (
            f"All NAWM medians within registered ranges "
            f"({n_maps_checked} map(s) checked)."
        )

    return reference


# ---------------------------------------------------------------------------
# Tumor segmentation dseg.tsv (BIDS-Common-Derivatives ROI summary table)
#
# This is the canonical TSV companion for the existing
# `desc-tumorseg_dseg.nii.gz` label map. Per BIDS, a `_dseg.tsv` is the
def write_tumor_dseg_tsv(
    segmentation_file: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Write a BIDS-Common-Derivatives ``_dseg.tsv`` for HD-GLIO tumor seg.

    Companion lookup table for the existing ``desc-tumorseg_dseg.nii.gz``
    label NIfTI. Columns: ``index, name, abbreviation, voxel_count,
    volume_mm3``. One row per non-background label. The accompanying JSON
    sidecar is built by
    :func:`pmriprep.utils.sidecar_builders.build_tumor_seg_dseg_sidecar`,
    which describes both the NIfTI label map and these TSV columns.

    Parameters
    ----------
    segmentation_file : str or Path
        Path to HD-GLIO dseg NIfTI (int labels 0/1/2).
    output_path : str or Path
        Output ``.tsv`` path. Should match the NIfTI basename
        (e.g. ``sub-X_space-T1w_desc-tumorseg_dseg.tsv``) so the
        companion JSON sidecar applies to both files per BIDS.

    Returns
    -------
    Path
        Resolved ``output_path`` as a ``Path``.

    Raises
    ------
    ValueError
        if the NIfTI contains label values outside the
        registered set (0, 1, 2).
    """
    import csv

    import nibabel as nib  # type: ignore[import-untyped]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(segmentation_file))
    seg_data = np.asarray(img.dataobj, dtype=np.int16)  # type: ignore[attr-defined]
    unique_labels = set(np.unique(seg_data).tolist())
    allowed = {0, 1, 2}
    unexpected = unique_labels - allowed
    if unexpected:
        raise ValueError(
            f"HD-GLIO dseg NIfTI contains unexpected labels {sorted(unexpected)}; "
            f"expected subset of {sorted(allowed)}. Source: {segmentation_file}",
        )
    zooms = img.header.get_zooms()[:3]  # type: ignore[attr-defined]
    voxel_sizes = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    rows: List[Dict[str, Any]] = []
    for index, name, abbreviation, _description in HDGLIO_DSEG_ROWS:
        mask = seg_data == index
        voxel_count, volume_mm3, _volume_ml = compute_region_volume(mask, voxel_sizes)
        rows.append({
            "index": index,
            "name": name,
            "abbreviation": abbreviation,
            "voxel_count": voxel_count,
            "volume_mm3": round(volume_mm3, 2),
        })

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "name", "abbreviation", "voxel_count", "volume_mm3"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return out_path


# ---------------------------------------------------------------------------
# Per-map lesion / ROI stats fragment extraction
# ---------------------------------------------------------------------------


def _slice_nawm_for_map(
    nawm_reference: Optional[Dict[str, Any]],
    map_name: str,
) -> Optional[Dict[str, Any]]:
    """Slice the modality NAWM reference dict for a single map.

    Returns a dict carrying:
        - ``voxel_count``, ``volume_ml`` (global mask context — informational;
          consumers apply their own size policy)
        - ``median``, ``std``, ``p25``, ``p75`` for ``map_name`` (in that
          map's native units; e.g. normalized maps carry ~1.0 ± noise SD)
        - ``value_qc_status`` / ``value_qc_notes`` for that map only
          (cross-map contamination avoided)

    Returns None when no NAWM reference is available for the map.
    """
    if not nawm_reference or not isinstance(nawm_reference, dict):
        return None
    map_stats = nawm_reference.get(map_name)
    if not isinstance(map_stats, dict) or "median" not in map_stats:
        return None

    out: Dict[str, Any] = {}
    for k in ("voxel_count", "volume_ml"):
        if k in nawm_reference:
            out[k] = nawm_reference[k]
    # Per-map stats and per-map value QC verdict.
    for k in ("median", "std", "p25", "p75", "value_qc_status"):
        if k in map_stats:
            out[k] = map_stats[k]

    # Carry value_qc_notes ONLY when relevant to this specific map, to avoid
    # cross-map contamination (a map that PASSed individually shouldn't ship
    # with notes complaining about a sibling map's WARNING). Three cases:
    per_map_status = out.get("value_qc_status")
    if per_map_status == "WARNING":
        # This map itself failed → surface the modality-level range note.
        modality_notes = nawm_reference.get("value_qc_notes")
        if modality_notes:
            out["value_qc_notes"] = modality_notes
    elif per_map_status is None:
        # No per-map verdict (e.g., this map has no entry in the range
        # table). Surface the modality-level status + notes so consumers
        # know QC was not applied to this map.
        modality_status = nawm_reference.get("value_qc_status")
        if modality_status is not None:
            out["value_qc_status"] = modality_status
        modality_notes = nawm_reference.get("value_qc_notes")
        if modality_notes:
            out["value_qc_notes"] = modality_notes
    # When per_map_status == "PASS", deliberately do NOT carry
    # value_qc_notes — the map is fine, no cross-map context needed.

    return out


def load_lesion_stats_fragment(
    stats_json_path: Optional[str],
    map_key: str,
) -> Optional[Dict[str, Any]]:
    """Load an aggregate lesion-stats JSON and return the per-map fragment.

    Parameters
    ----------
    stats_json_path : str or None
        Path to the aggregate lesion-stats JSON file.
    map_key : str
        Perfusion map name (e.g. "ktrans", "cbf").

    Returns
    -------
    dict or None
        Per-map lesion stats fragment, or None.
    """
    from nipype.interfaces.base import isdefined

    if not isdefined(stats_json_path) or not stats_json_path:
        return None
    path = Path(stats_json_path)
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            aggregate = json.load(fh)
        return extract_per_map_lesion_stats(aggregate, map_key)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        logger.debug(
            "Failed to load lesion stats from %s", stats_json_path,
            exc_info=True,
        )
        return None


def extract_per_map_lesion_stats(
    aggregate: Optional[Dict[str, Any]],
    map_name: str,
) -> Optional[Dict[str, Any]]:
    """Extract a per-perfusion-map fragment from an aggregate lesion stats dict.

    Used to inject per-map ``LesionStats`` blocks into perfusion map
    sidecars. The aggregate dict (output of
    :meth:`LesionStatisticsResult.to_dict`) carries per-region per-map
    stats plus modality-level context (NAWM reference, fractional
    volumes, mask type). This function slices out everything relevant
    to ``map_name``.

    Returns a fragment of the form::

        {
          "LesionMaskType": "multi_label_tumor_seg",
          "Regions": {
            "<region_name>": {
              "voxel_count": N, "volume_ml": V,
              "stats": <MapStatistics dict for map_name>,
            }, ...
          },
          "NAWMReference": {... per-map slice in that map's units ...},
          "FractionalVolumes": {"above_2.0": 0.137, "above_2.5": ...},
        }

    Optional fields are omitted when not applicable. Returns ``None``
    when the aggregate is empty or the requested map has no per-region
    stats AND no NAWM/fractional-volume context to forward (so the
    caller can skip injecting the field entirely rather than emit an
    empty block).
    """
    if not aggregate or not isinstance(aggregate, dict):
        return None
    regions = aggregate.get("Regions") or {}
    if not isinstance(regions, dict):
        return None

    out_regions: Dict[str, Dict[str, Any]] = {}
    for region_name, region_block in regions.items():
        if not isinstance(region_block, dict):
            continue
        per_map = region_block.get("perfusion_maps") or {}
        stats = per_map.get(map_name) if isinstance(per_map, dict) else None
        if stats is None:
            continue
        fragment: Dict[str, Any] = {}
        if "voxel_count" in region_block:
            fragment["voxel_count"] = region_block["voxel_count"]
        if "volume_ml" in region_block:
            fragment["volume_ml"] = region_block["volume_ml"]
        fragment["stats"] = stats
        out_regions[region_name] = fragment

    nawm_block = _slice_nawm_for_map(
        aggregate.get("NAWMReference"), map_name,
    )

    fractional_volumes = None
    derived = aggregate.get("DerivedMetrics")
    if isinstance(derived, dict):
        fv_block = derived.get("fractional_volumes")
        if isinstance(fv_block, dict) and map_name in fv_block:
            fractional_volumes = fv_block[map_name]

    mask_type = aggregate.get("LesionMaskType")

    # Require some substantive data — region stats or NAWM reference — to
    # emit a fragment. Standalone context (LesionMaskType, FractionalVolumes)
    # is only useful when paired with actual measurements.
    if not out_regions and not nawm_block:
        return None

    out: Dict[str, Any] = {}
    if mask_type:
        out["LesionMaskType"] = mask_type
    if out_regions:
        out["Regions"] = out_regions
    if nawm_block:
        out["NAWMReference"] = nawm_block
    if fractional_volumes:
        out["FractionalVolumes"] = fractional_volumes

    return out
