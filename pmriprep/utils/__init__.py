"""Utility modules for PMRIPrep."""

from pmriprep.utils.qc_standards import (
    COLORBLIND_PALETTE,
    ColorPalette,
    DICE_THRESHOLDS,
    DICE_THRESHOLDS_STRICT,
    DIVERGING_COLORMAP,
    MOTION_FD_THRESHOLDS,
    MOTION_OUTLIER_FRACTION_MAX,
    MOTION_RMS_THRESHOLDS,
    QCAction,
    QCRecommendation,
    QCStatus,
    QCThresholds,
    SEQUENTIAL_COLORMAP,
    aggregate_qc_recommendations,
    build_qc_summary,
    get_qc_recommendation,
    get_qc_status,
    get_status_color,
)
from pmriprep.utils.outputs import (
    copy_derivative,
    has_input,
    write_json_sidecar,
)
from pmriprep.utils.dce_dsc_validation import (
    ValidationMetrics,
    ValidationResult,
    validate_vp_cbv,
    validate_vp_cbv_from_files,
)
from pmriprep.utils.tumor_seg import (
    dilate_mask_from_array,
    erode_mask_from_array,
    extract_tumor_masks,
    get_mask_statistics,
    get_mask_volume_ml,
)
from pmriprep.utils.device import detect_torch_device
from pmriprep.utils.gpu import (
    detect_gpu_count,
    get_device_name,
)
from pmriprep.utils.structural_reference import (
    ApplicationType,
    ImageResolutionInfo,
    ResolutionTier,
    classify_resolution,
    get_geometric_mean_resolution,
    get_image_resolution_info,
    get_isotropy_ratio,
    is_postcontrast_image,
)

__all__ = [
    # qc_standards
    "COLORBLIND_PALETTE",
    "ColorPalette",
    "DICE_THRESHOLDS",
    "DICE_THRESHOLDS_STRICT",
    "DIVERGING_COLORMAP",
    "MOTION_FD_THRESHOLDS",
    "MOTION_OUTLIER_FRACTION_MAX",
    "MOTION_RMS_THRESHOLDS",
    "QCAction",
    "QCRecommendation",
    "QCStatus",
    "QCThresholds",
    "SEQUENTIAL_COLORMAP",
    "aggregate_qc_recommendations",
    "build_qc_summary",
    "get_qc_recommendation",
    "get_qc_status",
    "get_status_color",
    # outputs
    "copy_derivative",
    "has_input",
    "write_json_sidecar",
    # dce_dsc_validation
    "ValidationMetrics",
    "ValidationResult",
    "validate_vp_cbv",
    "validate_vp_cbv_from_files",
    # tumor_seg
    "dilate_mask_from_array",
    "erode_mask_from_array",
    "extract_tumor_masks",
    "get_mask_statistics",
    "get_mask_volume_ml",
    # device
    "detect_torch_device",
    # gpu
    "detect_gpu_count",
    "get_device_name",
    # structural_reference
    "ApplicationType",
    "ImageResolutionInfo",
    "ResolutionTier",
    "classify_resolution",
    "get_geometric_mean_resolution",
    "get_image_resolution_info",
    "get_isotropy_ratio",
    "is_postcontrast_image",
]
