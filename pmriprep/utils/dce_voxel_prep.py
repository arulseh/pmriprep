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

    Attributes
    ----------
    concentration : numpy.ndarray
        Prepared concentration time course.
    is_valid : bool
        Whether the voxel passed all validation checks.
    failure_reason : str
        Reason for failure or "ok" if valid.
    baseline_mean : float
        Mean baseline concentration (should be ~0 after subtraction).
    enhancement_max : float
        Maximum enhancement in mM.
    """

    concentration: npt.NDArray[np.floating]
    is_valid: bool
    failure_reason: str
    baseline_mean: float
    enhancement_max: float


def prepare_voxel_for_fitting(
    tissue_curve: npt.NDArray[np.floating],
    baseline_frames: int = 5,
    skip_variation_check: bool = False,
) -> VoxelPrepResult:
    """Validate and prepare a voxel's concentration time course for PK fitting.

    Checks performed in order:
    1. Signal variation: std(curve) > MIN_SIGNAL_STD (skipped if skip_variation_check)
    2. Enhancement: max(curve) > MIN_ENHANCEMENT_MM

    Parameters
    ----------
    tissue_curve : numpy.ndarray
        Concentration time course for one voxel (1D array, mM).
    baseline_frames : int, optional
        Number of baseline frames for quality checks.
    skip_variation_check : bool, optional
        If True, skip concentration-based variation check.
        Set True when raw signal variation filtering is done upstream
        (raw signal std >= 1.0).

    Returns
    -------
    VoxelPrepResult
        VoxelPrepResult with validation outcome and prepared curve.
    """
    baseline_mean = float(np.mean(tissue_curve[:baseline_frames]))

    # Check 1: Signal variation (on raw concentration)
    # Skipped when raw signal variation is checked upstream in ComputeDCEMaps.
    if not skip_variation_check:
        curve_std = float(np.std(tissue_curve))
        if curve_std < MIN_SIGNAL_STD:
            return VoxelPrepResult(
                concentration=tissue_curve,
                is_valid=False,
                failure_reason="failed_no_variation",
                baseline_mean=baseline_mean,
                enhancement_max=0.0,
            )

    # Concentration conversion owns baseline subtraction. Repeating it here
    # would subtract the positive mean of an already-rectified baseline from
    # Bayesian/MCMC paths while batched NLLS fits the converted curve directly.
    # Retain only idempotent nonnegative sanitation at the fitting boundary.
    prepared = np.maximum(tissue_curve, 0.0)

    enhancement_max = float(np.max(prepared))

    # Check 2: Enhancement above threshold (after baseline subtraction)
    if enhancement_max < MIN_ENHANCEMENT_MM:
        return VoxelPrepResult(
            concentration=prepared,
            is_valid=False,
            failure_reason="failed_no_enhancement",
            baseline_mean=baseline_mean,
            enhancement_max=enhancement_max,
        )

    return VoxelPrepResult(
        concentration=prepared,
        is_valid=True,
        failure_reason="ok",
        baseline_mean=baseline_mean,
        enhancement_max=enhancement_max,
    )
