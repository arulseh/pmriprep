"""Lesion/tumor ROI statistics extraction utilities.

This module provides functions for extracting perfusion statistics from
lesion/tumor regions of interest (ROIs), supporting both DSC and DCE
perfusion MRI analysis.

Key features:
- Multi-label mask handling (enhancing vs non-enhancing tumor)
- Comprehensive statistics (mean, std, percentiles, CV, IQR)
- Fractional volume calculations (voxels above threshold)
- NAWM reference value extraction
- Clinical context with glioma grading thresholds
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pmriprep


@dataclass
class MapStatistics:
    """Statistics for a single perfusion map within a ROI.

    Attributes:
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
        cv: Coefficient of variation (std/mean).
        iqr: Interquartile range (p75 - p25).
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
    cv: Optional[float] = None
    iqr: Optional[float] = None
    n_valid: int = 0
    n_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class RegionStatistics:
    """Statistics for all perfusion maps within a single region.

    Attributes:
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

    Attributes:
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session_id: BIDS session ID (without 'ses-' prefix).
        modality: Perfusion modality ("DSC" or "DCE").
        mask_type: Type of lesion mask used.
        mask_source: Path to the lesion mask file.
        total_voxels: Total number of lesion voxels.
        regions: Dictionary mapping region name to its statistics.
        nawm_reference: NAWM reference values for normalization context.
        derived_metrics: Derived metrics (fractional volumes, hotspots).
        clinical_context: Clinical interpretation thresholds.
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
    clinical_context: Optional[Dict[str, Any]] = None

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
        if self.clinical_context:
            result["ClinicalContext"] = self.clinical_context
        if self.nawm_reference:
            result["nawm_reference"] = self.nawm_reference
        result["Regions"] = {k: v.to_dict() for k, v in self.regions.items()}
        if self.derived_metrics:
            result["derived_metrics"] = self.derived_metrics
        return result


def compute_map_statistics(
    map_data: np.ndarray,
    mask: np.ndarray,
) -> Optional[MapStatistics]:
    """Compute comprehensive statistics for a perfusion map within a mask.

    Args:
        map_data: 3D array of perfusion map values.
        mask: 3D boolean array defining the ROI.

    Returns:
        MapStatistics object or None if mask is empty or all values invalid.
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
        # Non-empty mask but all values invalid — return stats with
        # n_valid=0 so callers can distinguish from empty-mask (None).
        return MapStatistics(n_valid=0, n_total=n_total)

    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))

    # Coefficient of variation (avoid division by zero)
    cv = None
    if mean_val != 0 and not np.isnan(mean_val):
        cv = round(std_val / abs(mean_val), 4)

    return MapStatistics(
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


def compute_fraction_above_threshold(
    map_data: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> Optional[float]:
    """Compute fraction of voxels above a threshold within mask.

    This is used for fractional volume calculations, e.g., fraction of
    tumor with nCBV > 1.75 (high-grade glioma indicator).

    Args:
        map_data: 3D array of perfusion map values.
        mask: 3D boolean array defining the ROI.
        threshold: Value threshold.

    Returns:
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

    Args:
        map_data: 3D array of perfusion map values.
        mask: 3D boolean array defining the ROI.
        percentile: Percentile to compute (default 90th).

    Returns:
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

    Args:
        mask_data: 3D array of mask values.

    Returns:
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

    Args:
        mask_data: 3D array of mask values (multi-label).
        label: Label value to extract.

    Returns:
        Boolean 3D array where mask_data == label.
    """
    return mask_data == label  # type: ignore[no-any-return]


def compute_region_volume(
    mask: np.ndarray,
    voxel_sizes: Tuple[float, float, float],
) -> Tuple[int, float, float]:
    """Compute voxel count and volume for a mask.

    Args:
        mask: Boolean 3D array.
        voxel_sizes: Voxel dimensions in mm (x, y, z).

    Returns:
        Tuple of (voxel_count, volume_mm3, volume_ml).
    """
    voxel_count = int(np.sum(mask))
    voxel_volume_mm3 = float(np.prod(voxel_sizes))
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0
    return voxel_count, volume_mm3, volume_ml


def get_dsc_clinical_context() -> Dict[str, Any]:
    """Get clinical context for DSC perfusion interpretation.

    Returns:
        Dictionary with clinical thresholds and references.
    """
    return {
        "nCBV_threshold_high_grade": 1.75,
        "nCBV_interpretation": "nCBV > 1.75 suggests high-grade glioma (WHO III-IV)",
        "nCBF_threshold_high_grade": 2.0,
        "nCBF_interpretation": "nCBF > 2.0 suggests high-grade glioma",
        "reference": "Law et al. Radiology 2003; Boxerman et al. Neuro-Oncology 2020",
    }


def get_dce_clinical_context() -> Dict[str, Any]:
    """Get clinical context for DCE perfusion interpretation.

    Returns:
        Dictionary with clinical thresholds and references.
    """
    return {
        "Ktrans_threshold_high_grade": 0.1,
        "Ktrans_interpretation": "Ktrans > 0.1 min^-1 suggests high-grade glioma",
        "nIAUC60_threshold_high_grade": 2.0,
        "nIAUC60_interpretation": "nIAUC60 > 2.0 suggests high-grade glioma",
        "reference": "Zhang et al. Radiology 2014",
    }


def compute_dsc_derived_metrics(
    ncbv_data: Optional[np.ndarray],
    ncbf_data: Optional[np.ndarray],
    cbv_data: Optional[np.ndarray],
    cbf_data: Optional[np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Compute DSC-specific derived metrics.

    Args:
        ncbv_data: Normalized CBV map data (or None).
        ncbf_data: Normalized CBF map data (or None).
        cbv_data: CBV map data (or None).
        cbf_data: CBF map data (or None).
        mask: Boolean mask for the region of interest.

    Returns:
        Dictionary with fractional volumes and hotspot summary.
    """
    derived: Dict[str, Any] = {}

    # Fractional volumes (only if nCBV available)
    if ncbv_data is not None:
        frac_175 = compute_fraction_above_threshold(ncbv_data, mask, 1.75)
        frac_25 = compute_fraction_above_threshold(ncbv_data, mask, 2.5)
        frac_40 = compute_fraction_above_threshold(ncbv_data, mask, 4.0)

        if any(f is not None for f in [frac_175, frac_25, frac_40]):
            derived["fractional_volumes"] = {
                "nCBV_above_1.75": frac_175,
                "nCBV_above_2.5": frac_25,
                "nCBV_above_4.0": frac_40,
                "interpretation": {
                    "nCBV_above_1.75": "Fraction suggesting high-grade features",
                    "nCBV_above_2.5": "Fraction with aggressive tumor characteristics",
                    "nCBV_above_4.0": "Fraction with highly vascular tumor",
                },
            }

    # Hotspot summary (90th percentile values)
    hotspots: Dict[str, Any] = {
        "description": "90th percentile values representing most aggressive tumor regions",
    }
    if ncbv_data is not None:
        val = compute_hotspot_value(ncbv_data, mask, 90.0)
        if val is not None:
            hotspots["ncbv_p90"] = val
    if ncbf_data is not None:
        val = compute_hotspot_value(ncbf_data, mask, 90.0)
        if val is not None:
            hotspots["ncbf_p90"] = val
    if cbv_data is not None:
        val = compute_hotspot_value(cbv_data, mask, 90.0)
        if val is not None:
            hotspots["cbv_p90"] = val
    if cbf_data is not None:
        val = compute_hotspot_value(cbf_data, mask, 90.0)
        if val is not None:
            hotspots["cbf_p90"] = val

    if len(hotspots) > 1:  # More than just description
        derived["hotspot_summary"] = hotspots

    return derived if derived else {}


def compute_dce_derived_metrics(
    ktrans_data: Optional[np.ndarray],
    nktrans_data: Optional[np.ndarray],
    ve_data: Optional[np.ndarray],
    vp_data: Optional[np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Compute DCE-specific derived metrics.

    Args:
        ktrans_data: Ktrans map data (or None).
        nktrans_data: Normalized Ktrans map data (or None).
        ve_data: ve map data (or None).
        vp_data: vp map data (or None).
        mask: Boolean mask for the region of interest.

    Returns:
        Dictionary with fractional volumes and hotspot summary.
    """
    derived: Dict[str, Any] = {}

    # Fractional volumes for Ktrans
    if ktrans_data is not None:
        frac_01 = compute_fraction_above_threshold(ktrans_data, mask, 0.1)
        frac_02 = compute_fraction_above_threshold(ktrans_data, mask, 0.2)

        if any(f is not None for f in [frac_01, frac_02]):
            derived["fractional_volumes"] = {
                "Ktrans_above_0.1": frac_01,
                "Ktrans_above_0.2": frac_02,
                "interpretation": {
                    "Ktrans_above_0.1": "Fraction suggesting high-grade features",
                    "Ktrans_above_0.2": "Fraction with aggressive tumor characteristics",
                },
            }

    # Hotspot summary
    hotspots: Dict[str, Any] = {
        "description": "90th percentile values representing most aggressive tumor regions",
    }
    if ktrans_data is not None:
        val = compute_hotspot_value(ktrans_data, mask, 90.0)
        if val is not None:
            hotspots["ktrans_p90"] = val
    if nktrans_data is not None:
        val = compute_hotspot_value(nktrans_data, mask, 90.0)
        if val is not None:
            hotspots["nktrans_p90"] = val
    if ve_data is not None:
        val = compute_hotspot_value(ve_data, mask, 90.0)
        if val is not None:
            hotspots["ve_p90"] = val
    if vp_data is not None:
        val = compute_hotspot_value(vp_data, mask, 90.0)
        if val is not None:
            hotspots["vp_p90"] = val

    if len(hotspots) > 1:
        derived["hotspot_summary"] = hotspots

    return derived if derived else {}


def combine_tumor_masks(
    tumor_segmentation_file=None,  # Optional[str] - type hint removed for Nipype Function
    whole_tumor_mask_file=None,    # Optional[str] - type hint removed for Nipype Function
    reference_file=None,           # str
):  # -> Tuple[Optional[str], str]
    """Select the best tumor mask for lesion statistics.

    Priority:
    1. If tumor segmentation present: use it (already multi-label:
       0=bg, 1=non-enhancing, 2=enhancing)
    2. If only whole_tumor present: use as binary mask
    3. If none present: return (None, "")

    Args:
        tumor_segmentation_file: Path to multi-label tumor segmentation (or None).
        whole_tumor_mask_file: Path to whole tumor binary mask (or None).
        reference_file: Path to reference image for output geometry (unused but
            kept for interface compatibility).

    Returns:
        Tuple of (mask_file_path, mask_origin) where mask_origin is
        "tumor_segmentation" or "whole_tumor" or "" if no masks available.

    Note:
        This function is used as a Nipype Function node, so type hints are removed
        from the signature (Optional requires import which breaks Function nodes).
    """
    from pathlib import Path

    # Check for Nipype Undefined
    from nipype.interfaces.base import isdefined

    def _is_valid_file(f):
        """Check if file path is valid (not None, not Undefined, exists)."""
        if f is None:
            return False
        if not isdefined(f):
            return False
        return Path(f).exists()

    # Priority 1: Use tumor segmentation directly (already multi-label)
    if _is_valid_file(tumor_segmentation_file):
        return (tumor_segmentation_file, "tumor_segmentation")

    # Priority 2: Use whole_tumor as binary mask
    if _is_valid_file(whole_tumor_mask_file):
        return (whole_tumor_mask_file, "whole_tumor")

    # Priority 3: No masks available
    return (None, "")


def compute_nawm_reference(
    nawm_mask: np.ndarray,
    maps: Dict[str, np.ndarray],
    voxel_sizes: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Compute NAWM reference values for normalization context.

    Args:
        nawm_mask: Boolean NAWM mask.
        maps: Dictionary mapping map name to 3D array.
        voxel_sizes: Voxel dimensions in mm.

    Returns:
        Dictionary with NAWM statistics for each available map.
    """
    voxel_count, volume_mm3, volume_ml = compute_region_volume(nawm_mask, voxel_sizes)

    if voxel_count == 0:
        return {}

    reference: Dict[str, Any] = {
        "voxel_count": voxel_count,
        "volume_ml": round(volume_ml, 2),
    }

    # Compute statistics for each map
    for map_name, map_data in maps.items():
        if map_data is None:
            continue
        stats = compute_map_statistics(map_data, nawm_mask)
        if stats is not None and stats.n_valid > 0:
            reference[map_name] = {
                "median": stats.median,
                "std": stats.std,
                "p25": stats.p25,
                "p75": stats.p75,
            }

    # QC check for physiological plausibility
    qc_status = "PASS"
    qc_notes = ""

    # Check CBV if available (for DSC)
    if "cbv" in reference and isinstance(reference["cbv"], dict):
        cbv_median = reference["cbv"].get("median", 0)
        if cbv_median < 0.5 or cbv_median > 10:
            qc_status = "WARNING"
            qc_notes = "NAWM CBV outside expected range"

    # Check Ktrans if available (for DCE)
    if "ktrans" in reference and isinstance(reference["ktrans"], dict):
        ktrans_median = reference["ktrans"].get("median", 0)
        if ktrans_median < 0 or ktrans_median > 0.5:
            qc_status = "WARNING"
            qc_notes = "NAWM Ktrans outside expected range"

    reference["qc_status"] = qc_status
    if qc_notes:
        reference["qc_notes"] = qc_notes

    return reference
