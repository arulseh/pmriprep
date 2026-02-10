"""DCE voxel preparation and validation for PK fitting.

Validates and prepares individual voxel concentration time courses before
pharmacokinetic model fitting, tracking failure reasons for QC.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_constants import MIN_ENHANCEMENT_MM, MIN_SIGNAL_STD


@dataclass
class VoxelPrepResult:
    """Result of voxel preparation for PK fitting.

    Attributes:
        concentration: Prepared concentration time course.
        is_valid: Whether the voxel passed all validation checks.
        failure_reason: Reason for failure or "ok" if valid.
        baseline_mean: Mean baseline concentration (should be ~0 after subtraction).
        enhancement_max: Maximum enhancement in mM.
    """

    concentration: npt.NDArray[np.floating]
    is_valid: bool
    failure_reason: str
    baseline_mean: float
    enhancement_max: float


def prepare_voxel_for_fitting(
    tissue_curve: npt.NDArray[np.floating],
    baseline_frames: int = 5,
) -> VoxelPrepResult:
    """Validate and prepare a voxel's concentration time course for PK fitting.

    Checks performed in order:
    1. Signal variation: std(curve) > MIN_SIGNAL_STD
    2. Enhancement: max(curve) > MIN_ENHANCEMENT_MM

    Args:
        tissue_curve: Concentration time course for one voxel (1D array, mM).
        baseline_frames: Number of baseline frames for quality checks.

    Returns:
        VoxelPrepResult with validation outcome and prepared curve.
    """
    baseline_mean = float(np.mean(tissue_curve[:baseline_frames]))
    enhancement_max = float(np.max(tissue_curve))

    # Check 1: Signal variation
    curve_std = float(np.std(tissue_curve))
    if curve_std < MIN_SIGNAL_STD:
        return VoxelPrepResult(
            concentration=tissue_curve,
            is_valid=False,
            failure_reason="failed_no_variation",
            baseline_mean=baseline_mean,
            enhancement_max=enhancement_max,
        )

    # Check 2: Enhancement above threshold
    if enhancement_max < MIN_ENHANCEMENT_MM:
        return VoxelPrepResult(
            concentration=tissue_curve,
            is_valid=False,
            failure_reason="failed_no_enhancement",
            baseline_mean=baseline_mean,
            enhancement_max=enhancement_max,
        )

    # Clip negative concentrations — physically concentration cannot be negative.
    # Matches perf-plus behavior and prevents NLLS from fitting noise artifacts.
    return VoxelPrepResult(
        concentration=np.maximum(tissue_curve, 0.0),
        is_valid=True,
        failure_reason="ok",
        baseline_mean=baseline_mean,
        enhancement_max=enhancement_max,
    )
