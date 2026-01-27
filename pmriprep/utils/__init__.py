"""Utility modules for PMRIPrep."""

from pmriprep.utils.gpu import (
    array_on_device,
    clear_gpu_memory,
    ensure_contiguous,
    get_array_module,
    get_device_name,
    get_gpu_memory_info,
    is_cupy_available,
    synchronize,
    to_device,
    to_host,
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
    rank_structural_candidates,
    select_structural_reference,
    select_structural_reference_from_session,
)

__all__ = [
    # gpu
    "array_on_device",
    "clear_gpu_memory",
    "ensure_contiguous",
    "get_array_module",
    "get_device_name",
    "get_gpu_memory_info",
    "is_cupy_available",
    "synchronize",
    "to_device",
    "to_host",
    # structural_reference
    "ApplicationType",
    "ImageResolutionInfo",
    "ResolutionTier",
    "classify_resolution",
    "get_geometric_mean_resolution",
    "get_image_resolution_info",
    "get_isotropy_ratio",
    "is_postcontrast_image",
    "rank_structural_candidates",
    "select_structural_reference",
    "select_structural_reference_from_session",
]
