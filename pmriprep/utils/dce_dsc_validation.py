"""DSC-DCE cross-validation utilities.

Post-hoc comparison of DCE-derived vp (plasma volume fraction) with DSC-derived
CBV (cerebral blood volume) to validate quantification and detect preload effects.

The expected relationship between vp and CBV accounts for:
- Hematocrit (0.45): vp measures plasma, CBV measures whole blood
- Tissue density (1.04 g/mL): converts mL/100mL to mL/100g

Expected ratio at 3T with full dose:
    (vp × 100) / CBV ≈ (1 - Hct) × ρ ≈ 0.572

where:
    vp is in fraction (0-1)
    CBV is in mL/100g (DSC typical units)

Ratio deviations indicate:
- Ratio << expected: Suggests preload effect (reduced DSC signal due to prior contrast)
- Ratio >> expected: Suggests DCE overestimation or DSC underestimation

When constraint_mode is "hard", vp is deterministically derived from CBV,
making correlation = 1.0 by construction (circular). The validation detects
this and reports status = "CIRCULAR" instead of running meaningless metrics.

References:
    - Schmainda et al., Theory and Application of Quantitative DCE-MRI (2021)
    - Comparison of DSC CBV and DCE vp for brain tumor imaging
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_constants import TISSUE_DENSITY
from pmriprep.utils.qc_standards import QCStatus


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

#: Default hematocrit (fraction of red blood cells in whole blood)
#: Note: This differs from dce_constants.HEMATOCRIT (0.42) which is used for
#: plasma concentration conversion. Here we use 0.45 for vp/CBV comparison.
HEMATOCRIT_VP_CBV: float = 0.45

#: Tolerance for ratio validation (±)
#: Acceptable range: expected × (1 ± 0.5)
RATIO_TOLERANCE_FRACTION: float = 0.5

#: Minimum vp value to include in validation (avoid numerical instability)
MIN_VP_FOR_VALIDATION: float = 0.001

#: Minimum CBV value to include in validation (avoid division by zero)
MIN_CBV_FOR_VALIDATION: float = 0.1

#: Maximum vp for validation (above = likely large vessel partial volume effect)
MAX_VP_FOR_VALIDATION: float = 0.15

#: Maximum CBV for validation (above = likely large vessel partial volume effect)
MAX_CBV_FOR_VALIDATION: float = 8.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ValidationMetrics:
    """Detailed metrics from DSC-DCE cross-validation.

    Attributes:
        mean_ratio: Mean (vp * 100) / CBV ratio across valid voxels.
        median_ratio: Median ratio across valid voxels.
        std_ratio: Standard deviation of ratio.
        p10_ratio: 10th percentile ratio.
        p90_ratio: 90th percentile ratio.
        n_valid_voxels: Number of voxels included in validation.
        n_total_voxels: Total number of voxels in mask.
        fraction_valid: Fraction of voxels with valid values.
        correlation: Pearson correlation between vp and CBV.
        mean_vp: Mean vp value in valid voxels (fraction).
        mean_cbv: Mean CBV value in valid voxels (mL/100g).
        expected_ratio: Expected ratio based on hematocrit and tissue density.
    """

    mean_ratio: float = 0.0
    median_ratio: float = 0.0
    std_ratio: float = 0.0
    p10_ratio: float = 0.0
    p90_ratio: float = 0.0
    n_valid_voxels: int = 0
    n_total_voxels: int = 0
    fraction_valid: float = 0.0
    correlation: float = 0.0
    mean_vp: float = 0.0
    mean_cbv: float = 0.0
    expected_ratio: float = 0.0

    def to_dict(self) -> dict[str, Union[float, int]]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "mean_ratio": self.mean_ratio,
            "median_ratio": self.median_ratio,
            "std_ratio": self.std_ratio,
            "p10_ratio": self.p10_ratio,
            "p90_ratio": self.p90_ratio,
            "n_valid_voxels": self.n_valid_voxels,
            "n_total_voxels": self.n_total_voxels,
            "fraction_valid": self.fraction_valid,
            "correlation": self.correlation,
            "mean_vp": self.mean_vp,
            "mean_cbv": self.mean_cbv,
            "expected_ratio": self.expected_ratio,
        }


@dataclass
class ValidationResult:
    """Result of DSC-DCE cross-validation.

    Attributes:
        status: Overall validation status (PASS/WARN/POOR/CIRCULAR).
        preload_effect_detected: Whether preload effect is suspected from
            vp/CBV ratio analysis. Distinct from DSC config ``preload_present``
            which records the user/scanner configuration.
        dsc_preload_configured: Whether DSC preload was configured (from DSC
            config), or None if unknown.
        constraint_mode: CBV constraint mode used (hard/soft/skip/None).
        circular: True when constraint_mode="hard" — vp is deterministically
            derived from CBV, making cross-validation meaningless.
        correction_factor: Suggested correction factor to align vp with CBV.
        metrics: Detailed validation metrics.
        interpretation: Human-readable interpretation of results.
    """

    status: str = QCStatus.UNKNOWN
    preload_effect_detected: bool = False
    dsc_preload_configured: Optional[bool] = None
    constraint_mode: Optional[str] = None
    circular: bool = False
    correction_factor: float = 1.0
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "preload_effect_detected": self.preload_effect_detected,
            "dsc_preload_configured": self.dsc_preload_configured,
            "constraint_mode": self.constraint_mode,
            "circular": self.circular,
            "correction_factor": self.correction_factor,
            "metrics": self.metrics.to_dict(),
            "interpretation": self.interpretation,
        }


# =============================================================================
# Ratio Calculation Functions
# =============================================================================


def calculate_vp_cbv_ratio(
    vp: npt.NDArray[np.floating[npt.NBitBase]],
    cbv: npt.NDArray[np.floating[npt.NBitBase]],
) -> npt.NDArray[np.floating[npt.NBitBase]]:
    """Calculate (vp * 100) / CBV ratio.

    Args:
        vp: Plasma volume fraction array (0-1 scale).
        cbv: Cerebral blood volume array (mL/100g scale).

    Returns:
        Ratio array with same shape as inputs.
    """
    # Avoid division by zero
    cbv_safe = np.maximum(cbv, MIN_CBV_FOR_VALIDATION)
    ratio = (vp * 100.0) / cbv_safe
    return ratio  # type: ignore[no-any-return]


def get_expected_ratio(
    hematocrit: float = HEMATOCRIT_VP_CBV,
    tissue_density: float = TISSUE_DENSITY,
) -> float:
    """Calculate expected vp/CBV ratio from physiological parameters.

    The expected ratio relates plasma volume (vp) to whole blood volume (CBV):
        expected_ratio = (1 - hematocrit) * tissue_density

    Args:
        hematocrit: Hematocrit value (default: 0.45).
        tissue_density: Tissue density in g/mL (default: 1.04).

    Returns:
        Expected (vp * 100) / CBV ratio.
    """
    # vp measures plasma, CBV measures whole blood
    # vp = CBV * (1 - hematocrit) * tissue_density
    # (vp * 100) / CBV = (1 - hematocrit) * tissue_density
    return (1.0 - hematocrit) * tissue_density


def calculate_correction_factor(
    observed_ratio: float,
    expected_ratio: float,
) -> float:
    """Calculate correction factor to align observed ratio with expected.

    The correction factor can be applied to vp or (inversely) to CBV:
        vp_corrected = vp * correction_factor
        cbv_corrected = cbv / correction_factor

    Args:
        observed_ratio: Observed mean or median vp/CBV ratio.
        expected_ratio: Expected ratio from get_expected_ratio().

    Returns:
        Correction factor (>1 if observed ratio is low, <1 if high).
    """
    if observed_ratio <= 0:
        return 1.0

    return expected_ratio / observed_ratio


# =============================================================================
# Validation Functions
# =============================================================================


def validate_vp_cbv(
    vp_data: npt.NDArray[np.floating[npt.NBitBase]],
    cbv_data: npt.NDArray[np.floating[npt.NBitBase]],
    mask: npt.NDArray[np.bool_],
    hematocrit: float = HEMATOCRIT_VP_CBV,
    tissue_density: float = TISSUE_DENSITY,
    constraint_mode: Optional[str] = None,
    dsc_preload_configured: Optional[bool] = None,
) -> ValidationResult:
    """Validate DCE vp against DSC CBV.

    Compares vp (plasma volume fraction from DCE) with CBV (cerebral blood
    volume from DSC) to detect quantification issues or preload effects.

    When constraint_mode is "hard", vp is deterministically derived from CBV
    (circular), so ratio/correlation metrics are skipped and status is set to
    "CIRCULAR". The caller should reference delta-R2 from pk_qc instead.

    Args:
        vp_data: 3D vp map array (fraction, 0-1 scale).
        cbv_data: 3D CBV map array (mL/100g scale).
        mask: 3D boolean mask for analysis region.
        hematocrit: Hematocrit value for expected ratio calculation.
        tissue_density: Tissue density for expected ratio calculation.
        constraint_mode: CBV constraint mode ("hard", "soft", "skip", or None).
        dsc_preload_configured: Whether DSC preload was configured.

    Returns:
        ValidationResult with status, preload detection, and metrics.
    """
    result = ValidationResult(
        constraint_mode=constraint_mode,
        dsc_preload_configured=dsc_preload_configured,
    )
    metrics = ValidationMetrics()

    # Get expected ratio dynamically
    expected_ratio = get_expected_ratio(hematocrit, tissue_density)
    metrics.expected_ratio = expected_ratio

    # Compute thresholds dynamically from expected ratio
    ratio_preload_threshold = expected_ratio * (1.0 - RATIO_TOLERANCE_FRACTION)
    ratio_high_threshold = expected_ratio * (1.0 + RATIO_TOLERANCE_FRACTION)

    # Extract values within mask
    vp_masked = vp_data[mask]
    cbv_masked = cbv_data[mask]
    metrics.n_total_voxels = int(np.sum(mask))

    if metrics.n_total_voxels == 0:
        logger.warning("Empty mask, cannot perform validation")
        result.status = QCStatus.POOR
        result.interpretation = "Empty mask - no voxels for validation"
        result.metrics = metrics
        return result

    # Detect circularity: hard constraint means vp = f(CBV) deterministically
    if constraint_mode == "hard":
        result.circular = True
        result.status = "CIRCULAR"
        result.interpretation = (
            "vp hard-constrained from CBV — cross-validation is circular. "
            "See delta-R² in pk_qc for constraint benefit assessment."
        )
        result.metrics = metrics
        logger.info(
            "DSC-DCE Validation: CIRCULAR (hard constraint), "
            "skipping ratio/correlation metrics"
        )
        return result

    # Filter to valid voxels (positive values above thresholds, below upper bounds)
    valid_mask = (
        (vp_masked > MIN_VP_FOR_VALIDATION)
        & (cbv_masked > MIN_CBV_FOR_VALIDATION)
        & (vp_masked < MAX_VP_FOR_VALIDATION)
        & (cbv_masked < MAX_CBV_FOR_VALIDATION)
        & np.isfinite(vp_masked)
        & np.isfinite(cbv_masked)
    )

    vp_valid = vp_masked[valid_mask]
    cbv_valid = cbv_masked[valid_mask]
    metrics.n_valid_voxels = len(vp_valid)
    metrics.fraction_valid = metrics.n_valid_voxels / metrics.n_total_voxels

    if metrics.n_valid_voxels < 100:
        logger.warning(
            f"Insufficient valid voxels for validation: {metrics.n_valid_voxels}"
        )
        result.status = QCStatus.WARN
        result.interpretation = (
            f"Insufficient valid voxels ({metrics.n_valid_voxels}) for reliable validation"
        )
        result.metrics = metrics
        return result

    # Calculate ratio statistics
    ratios = calculate_vp_cbv_ratio(vp_valid, cbv_valid)

    metrics.mean_ratio = float(np.mean(ratios))
    metrics.median_ratio = float(np.median(ratios))
    metrics.std_ratio = float(np.std(ratios))
    metrics.p10_ratio = float(np.percentile(ratios, 10))
    metrics.p90_ratio = float(np.percentile(ratios, 90))
    metrics.mean_vp = float(np.mean(vp_valid))
    metrics.mean_cbv = float(np.mean(cbv_valid))

    # Calculate correlation with NaN guard
    if len(vp_valid) > 2:
        correlation_matrix = np.corrcoef(vp_valid, cbv_valid)
        corr_val = correlation_matrix[0, 1]
        metrics.correlation = float(corr_val) if np.isfinite(corr_val) else 0.0
    else:
        metrics.correlation = 0.0

    # Calculate correction factor based on median ratio
    result.correction_factor = calculate_correction_factor(
        metrics.median_ratio, expected_ratio
    )

    # Determine status and detect preload
    interpretations: list[str] = []

    if metrics.median_ratio < ratio_preload_threshold:
        # Ratio much lower than expected - suggests preload
        result.status = QCStatus.WARN
        result.preload_effect_detected = True
        interpretations.append(
            f"Low vp/CBV ratio ({metrics.median_ratio:.3f}) suggests preload effect"
        )
        interpretations.append(
            f"Expected ratio: {expected_ratio:.3f}, observed: {metrics.median_ratio:.3f}"
        )
        interpretations.append(
            f"Suggested correction factor: {result.correction_factor:.2f}"
        )

    elif metrics.median_ratio > ratio_high_threshold:
        # Ratio much higher than expected - suggests DCE overestimation
        result.status = QCStatus.WARN
        result.preload_effect_detected = False
        interpretations.append(
            f"High vp/CBV ratio ({metrics.median_ratio:.3f}) suggests DCE vp overestimation"
        )
        interpretations.append(
            f"Expected ratio: {expected_ratio:.3f}, observed: {metrics.median_ratio:.3f}"
        )
        interpretations.append(
            f"Suggested correction factor: {result.correction_factor:.2f}"
        )

    else:
        # Ratio within acceptable range
        result.status = QCStatus.PASS
        result.preload_effect_detected = False
        interpretations.append(
            f"vp/CBV ratio ({metrics.median_ratio:.3f}) within expected range"
        )
        interpretations.append(
            f"Expected: {expected_ratio:.3f}, tolerance: +/-{RATIO_TOLERANCE_FRACTION*100:.0f}%"
        )

    # Soft constraint annotation
    if constraint_mode == "soft":
        interpretations.append(
            "vp soft-constrained — correlation may be inflated"
        )

    # Add correlation assessment
    if metrics.correlation > 0.7:
        interpretations.append(f"Strong vp-CBV correlation (r={metrics.correlation:.3f})")
    elif metrics.correlation > 0.4:
        interpretations.append(f"Moderate vp-CBV correlation (r={metrics.correlation:.3f})")
    else:
        interpretations.append(f"Weak vp-CBV correlation (r={metrics.correlation:.3f})")

    result.interpretation = "; ".join(interpretations)
    result.metrics = metrics

    logger.info(
        f"DSC-DCE Validation: status={result.status}, "
        f"preload_effect_detected={result.preload_effect_detected}, "
        f"median_ratio={metrics.median_ratio:.3f}, "
        f"correlation={metrics.correlation:.3f}"
    )

    return result


def validate_vp_cbv_from_files(
    vp_path: Union[str, Path],
    cbv_path: Union[str, Path],
    mask_path: Union[str, Path],
    hematocrit: float = HEMATOCRIT_VP_CBV,
    tissue_density: float = TISSUE_DENSITY,
    constraint_mode: Optional[str] = None,
    dsc_preload_configured: Optional[bool] = None,
) -> ValidationResult:
    """Validate DCE vp against DSC CBV from NIfTI files.

    Convenience wrapper for validate_vp_cbv() that loads data from files.

    Args:
        vp_path: Path to DCE vp NIfTI file.
        cbv_path: Path to DSC CBV NIfTI file.
        mask_path: Path to mask NIfTI file (brain or tissue mask).
        hematocrit: Hematocrit value for expected ratio calculation.
        tissue_density: Tissue density for expected ratio calculation.
        constraint_mode: CBV constraint mode ("hard", "soft", "skip", or None).
        dsc_preload_configured: Whether DSC preload was configured.

    Returns:
        ValidationResult with status, preload detection, and metrics.

    Raises:
        FileNotFoundError: If any input file does not exist.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    # Verify files exist
    vp_path = Path(vp_path)
    cbv_path = Path(cbv_path)
    mask_path = Path(mask_path)

    for path, name in [(vp_path, "vp"), (cbv_path, "CBV"), (mask_path, "mask")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    # Load data
    vp_img = nib.load(vp_path)
    cbv_img = nib.load(cbv_path)
    mask_img = nib.load(mask_path)

    vp_data = np.asarray(vp_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]
    cbv_data = np.asarray(cbv_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]
    mask = np.asarray(mask_img.get_fdata(), dtype=np.bool_)  # type: ignore[attr-defined]

    return validate_vp_cbv(
        vp_data=vp_data,  # type: ignore[arg-type]
        cbv_data=cbv_data,  # type: ignore[arg-type]
        mask=mask,
        hematocrit=hematocrit,
        tissue_density=tissue_density,
        constraint_mode=constraint_mode,
        dsc_preload_configured=dsc_preload_configured,
    )
