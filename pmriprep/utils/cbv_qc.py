"""DSC CBV QC and handoff to DCE processing.

Quality control assessment of DSC-derived CBV maps to determine whether
CBV can be used as a constraint in DCE pharmacokinetic modeling.

The decision logic follows Prah & Schmainda (2024):
- CNR thresholds determine data quality
- Negative CBV fraction indicates processing issues
- Physiological ranges validate biological plausibility

When QC passes, DSC CBV can constrain DCE vp parameter.
When QC fails, DCE should fit vp freely without constraint.

References:
    - Prah & Schmainda (2024) for CBV QC thresholds
    - DSC CNR threshold of 4.0 from Paulson & Schmainda (2008)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.qc_standards import QCStatus


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# CNR thresholds (fraction of brain voxels with CNR >= 4.0)
CNR_PASS_THRESHOLD: float = 0.84  # >= 84% CNR >= 4.0 -> PASS
CNR_WARN_THRESHOLD: float = 0.65  # >= 65% CNR >= 4.0 -> WARN; < 65% -> FAIL

# Negative CBV threshold
NEGATIVE_CBV_MAX_FRACTION: float = 0.05  # > 5% negative -> FAIL

# Physiological ranges for RELATIVE CBV (when median < 1.0, assumed normalized)
RELATIVE_CBV_MEDIAN_MIN: float = 0.05
RELATIVE_CBV_MEDIAN_MAX: float = 2.0
RELATIVE_CBV_P75_MIN: float = 0.1
RELATIVE_CBV_P75_MAX: float = 5.0

# Physiological ranges for ABSOLUTE CBV (when median >= 1.0, assumed mL/100g)
ABSOLUTE_CBV_MEDIAN_MIN: float = 1.5
ABSOLUTE_CBV_MEDIAN_MAX: float = 6.0
ABSOLUTE_CBV_P75_MIN: float = 2.0
ABSOLUTE_CBV_P75_MAX: float = 10.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CBVQCMetrics:
    """Metrics from CBV quality control assessment.

    Attributes:
        cnr_fraction: Fraction of brain voxels with CNR >= 4.0.
        cnr_status: QC status based on CNR (PASS/WARN/FAIL).
        negative_cbv_fraction: Fraction of brain voxels with negative CBV.
        negative_cbv_status: QC status based on negative CBV (PASS/FAIL).
        cbv_median: Median CBV within brain mask.
        cbv_p75: 75th percentile CBV within brain mask.
        cbv_is_relative: Whether CBV appears to be relative (normalized).
        cbv_physiological_status: Whether CBV is in physiological range.
        cbv_physiological_note: Description of any physiological concerns.
    """

    cnr_fraction: float = 0.0
    cnr_status: str = QCStatus.UNKNOWN
    negative_cbv_fraction: float = 0.0
    negative_cbv_status: str = QCStatus.UNKNOWN
    cbv_median: float = 0.0
    cbv_p75: float = 0.0
    cbv_is_relative: bool = True
    cbv_physiological_status: str = QCStatus.UNKNOWN
    cbv_physiological_note: str = ""

    def to_dict(self) -> dict[str, Union[float, str, bool]]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "cnr_fraction": self.cnr_fraction,
            "cnr_status": self.cnr_status,
            "negative_cbv_fraction": self.negative_cbv_fraction,
            "negative_cbv_status": self.negative_cbv_status,
            "cbv_median": self.cbv_median,
            "cbv_p75": self.cbv_p75,
            "cbv_is_relative": self.cbv_is_relative,
            "cbv_physiological_status": self.cbv_physiological_status,
            "cbv_physiological_note": self.cbv_physiological_note,
        }


@dataclass
class CBVQCResult:
    """Result of CBV quality control for DCE handoff.

    Attributes:
        status: Overall QC status (PASS/WARN/FAIL).
        constraint_enabled: Whether DSC CBV can be used to constrain DCE vp.
        metrics: Detailed QC metrics.
        decision_reason: Human-readable reason for the decision.
    """

    status: str = QCStatus.UNKNOWN
    constraint_enabled: bool = False
    metrics: CBVQCMetrics = field(default_factory=CBVQCMetrics)
    decision_reason: str = ""

    def to_dict(self) -> dict[str, Union[str, bool, dict[str, Union[float, str, bool]]]]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "constraint_enabled": self.constraint_enabled,
            "metrics": self.metrics.to_dict(),
            "decision_reason": self.decision_reason,
        }


# =============================================================================
# CNR Functions
# =============================================================================


def read_cnr_from_json(qc_json_path: Union[str, Path]) -> Optional[float]:
    """Read CNR fraction from DSC AIF QC JSON file.

    Extracts the fraction of brain voxels with CNR >= 4.0 from the
    DSC AIF quality control JSON sidecar.

    Args:
        qc_json_path: Path to the DSC AIF QC JSON file.

    Returns:
        Fraction of voxels with CNR >= 4.0, or None if not found.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    qc_path = Path(qc_json_path)

    if not qc_path.exists():
        raise FileNotFoundError(f"QC JSON file not found: {qc_path}")

    with open(qc_path, "r") as f:
        qc_data = json.load(f)

    # Navigate to cnr_whole_brain.fraction_cnr_gt_4
    cnr_data = qc_data.get("cnr_whole_brain", {})
    cnr_fraction = cnr_data.get("fraction_cnr_gt_4")

    if cnr_fraction is None:
        # Try alternative key paths that might be used
        cnr_fraction = qc_data.get("fraction_cnr_gt_4")

    if cnr_fraction is not None:
        return float(cnr_fraction)

    logger.warning(
        f"CNR fraction not found in QC JSON: {qc_path}. "
        "Expected key: cnr_whole_brain.fraction_cnr_gt_4"
    )
    return None


def check_cnr_status(cnr_fraction: float) -> str:
    """Determine QC status based on CNR fraction.

    Args:
        cnr_fraction: Fraction of brain voxels with CNR >= 4.0.

    Returns:
        QC status string (PASS/WARN/FAIL).
    """
    if cnr_fraction >= CNR_PASS_THRESHOLD:
        return QCStatus.PASS
    elif cnr_fraction >= CNR_WARN_THRESHOLD:
        return QCStatus.WARN
    else:
        return QCStatus.FAIL


# =============================================================================
# Negative CBV Functions
# =============================================================================


def check_negative_cbv_fraction(
    cbv_data: npt.NDArray[np.floating[npt.NBitBase]],
    brain_mask: npt.NDArray[np.bool_],
) -> tuple[float, str]:
    """Calculate fraction of negative CBV voxels within brain mask.

    Args:
        cbv_data: 3D CBV map array.
        brain_mask: 3D boolean brain mask.

    Returns:
        Tuple of (negative_fraction, qc_status).
    """
    # Extract CBV values within brain mask
    brain_cbv = cbv_data[brain_mask]

    # Calculate fraction of negative values
    n_brain_voxels = int(np.sum(brain_mask))
    if n_brain_voxels == 0:
        logger.warning("Empty brain mask, cannot calculate negative CBV fraction")
        return 0.0, QCStatus.FAIL

    n_negative = int(np.sum(brain_cbv < 0))
    negative_fraction = n_negative / n_brain_voxels

    # Determine status
    if negative_fraction <= NEGATIVE_CBV_MAX_FRACTION:
        status = QCStatus.PASS
    else:
        status = QCStatus.FAIL

    return negative_fraction, status


# =============================================================================
# Physiological Range Functions
# =============================================================================


def check_physiological_ranges(
    cbv_data: npt.NDArray[np.floating[npt.NBitBase]],
    brain_mask: npt.NDArray[np.bool_],
) -> tuple[float, float, bool, str, str]:
    """Check if CBV values are within physiologically plausible ranges.

    Determines whether CBV is relative (normalized) or absolute based on
    the median value, then validates against appropriate ranges.

    Args:
        cbv_data: 3D CBV map array.
        brain_mask: 3D boolean brain mask.

    Returns:
        Tuple of (median, p75, is_relative, status, note).
    """
    # Extract CBV values within brain mask (positive values only for statistics)
    brain_cbv = cbv_data[brain_mask]
    positive_cbv = brain_cbv[brain_cbv > 0]

    if len(positive_cbv) == 0:
        logger.warning("No positive CBV values in brain mask")
        return 0.0, 0.0, True, QCStatus.FAIL, "No positive CBV values"

    # Calculate statistics
    median = float(np.median(positive_cbv))
    p75 = float(np.percentile(positive_cbv, 75))

    # Determine if relative or absolute based on median
    # Relative CBV (normalized to NAWM) typically has median < 1.0
    # Absolute CBV is in mL/100g with median typically 2-4
    is_relative = median < 1.0

    # Check against appropriate ranges
    notes: list[str] = []

    if is_relative:
        # Check relative CBV ranges
        if median < RELATIVE_CBV_MEDIAN_MIN:
            notes.append(f"Unusually low median rCBV: {median:.3f}")
        elif median > RELATIVE_CBV_MEDIAN_MAX:
            notes.append(f"Unusually high median rCBV: {median:.3f}")

        if p75 < RELATIVE_CBV_P75_MIN:
            notes.append(f"Unusually low P75 rCBV: {p75:.3f}")
        elif p75 > RELATIVE_CBV_P75_MAX:
            notes.append(f"Unusually high P75 rCBV: {p75:.3f}")
    else:
        # Check absolute CBV ranges
        if median < ABSOLUTE_CBV_MEDIAN_MIN:
            notes.append(f"Unusually low median CBV: {median:.3f} mL/100g")
        elif median > ABSOLUTE_CBV_MEDIAN_MAX:
            notes.append(f"Unusually high median CBV: {median:.3f} mL/100g")

        if p75 < ABSOLUTE_CBV_P75_MIN:
            notes.append(f"Unusually low P75 CBV: {p75:.3f} mL/100g")
        elif p75 > ABSOLUTE_CBV_P75_MAX:
            notes.append(f"Unusually high P75 CBV: {p75:.3f} mL/100g")

    # Determine status
    if notes:
        status = QCStatus.WARN
        note = "; ".join(notes)
    else:
        status = QCStatus.PASS
        note = "CBV in expected physiological range"

    return median, p75, is_relative, status, note


# =============================================================================
# Main Assessment Function
# =============================================================================


def assess_cbv_for_dce(
    cbv_path: Optional[Union[str, Path]] = None,
    brain_mask_path: Optional[Union[str, Path]] = None,
    qc_json_path: Optional[Union[str, Path]] = None,
    cbv_data: Optional[npt.NDArray[np.floating[npt.NBitBase]]] = None,
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    cnr_fraction: Optional[float] = None,
) -> CBVQCResult:
    """Assess DSC CBV quality for DCE handoff.

    Evaluates whether DSC-derived CBV meets quality criteria for use as
    a constraint in DCE pharmacokinetic modeling. The assessment includes:
    1. CNR check (from QC JSON)
    2. Negative CBV fraction check
    3. Physiological range check

    Decision logic:
    - PASS/WARN (CNR >= 65%, negative <= 5%): Enable CBV constraint
    - FAIL (CNR < 65% OR negative > 5%): Disable constraint, fit vp freely

    Args:
        cbv_path: Path to CBV NIfTI file (alternative to cbv_data).
        brain_mask_path: Path to brain mask NIfTI file (alternative to brain_mask).
        qc_json_path: Path to DSC AIF QC JSON file (alternative to cnr_fraction).
        cbv_data: 3D CBV array (alternative to cbv_path).
        brain_mask: 3D brain mask array (alternative to brain_mask_path).
        cnr_fraction: Pre-computed CNR fraction (alternative to qc_json_path).

    Returns:
        CBVQCResult with status, constraint_enabled flag, and detailed metrics.

    Raises:
        ValueError: If neither file paths nor arrays are provided.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    result = CBVQCResult()
    metrics = CBVQCMetrics()

    # Load CBV data if path provided
    if cbv_data is None:
        if cbv_path is None:
            raise ValueError("Either cbv_path or cbv_data must be provided")
        cbv_img = nib.load(cbv_path)
        cbv_data = np.asarray(cbv_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined,arg-type]

    # Load brain mask if path provided
    if brain_mask is None:
        if brain_mask_path is None:
            raise ValueError("Either brain_mask_path or brain_mask must be provided")
        mask_img = nib.load(brain_mask_path)
        brain_mask = np.asarray(mask_img.get_fdata(), dtype=np.bool_)  # type: ignore[attr-defined]

    # Read CNR if path provided
    if cnr_fraction is None:
        if qc_json_path is not None:
            try:
                cnr_fraction = read_cnr_from_json(qc_json_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not read CNR from QC JSON: {e}")
                cnr_fraction = None

    # Assess CNR
    if cnr_fraction is not None:
        metrics.cnr_fraction = cnr_fraction
        metrics.cnr_status = check_cnr_status(cnr_fraction)
        logger.info(
            f"CNR QC: fraction={cnr_fraction:.3f}, status={metrics.cnr_status}"
        )
    else:
        logger.warning("CNR fraction not available, skipping CNR check")
        metrics.cnr_status = QCStatus.UNKNOWN

    # Assess negative CBV fraction
    negative_frac, negative_status = check_negative_cbv_fraction(cbv_data, brain_mask)
    metrics.negative_cbv_fraction = negative_frac
    metrics.negative_cbv_status = negative_status
    logger.info(
        f"Negative CBV QC: fraction={negative_frac:.3f}, status={negative_status}"
    )

    # Assess physiological ranges
    median, p75, is_rel, phys_status, phys_note = check_physiological_ranges(
        cbv_data, brain_mask
    )
    metrics.cbv_median = median
    metrics.cbv_p75 = p75
    metrics.cbv_is_relative = is_rel
    metrics.cbv_physiological_status = phys_status
    metrics.cbv_physiological_note = phys_note
    logger.info(
        f"Physiological QC: median={median:.3f}, P75={p75:.3f}, "
        f"is_relative={is_rel}, status={phys_status}"
    )

    # Make decision
    result.metrics = metrics
    reasons: list[str] = []

    # Primary criteria for enabling constraint
    cnr_ok = metrics.cnr_status in (QCStatus.PASS, QCStatus.WARN, QCStatus.UNKNOWN)
    negative_ok = metrics.negative_cbv_status == QCStatus.PASS

    if cnr_ok and negative_ok:
        result.constraint_enabled = True

        # Determine overall status
        if metrics.cnr_status == QCStatus.PASS:
            result.status = QCStatus.PASS
            reasons.append(f"CNR excellent ({metrics.cnr_fraction:.1%} >= 84%)")
        elif metrics.cnr_status == QCStatus.WARN:
            result.status = QCStatus.WARN
            reasons.append(
                f"CNR acceptable ({metrics.cnr_fraction:.1%} >= 65%, < 84%)"
            )
        else:
            # CNR unknown but negative CBV OK
            result.status = QCStatus.WARN
            reasons.append("CNR not available, negative CBV acceptable")

        reasons.append(
            f"Negative CBV fraction acceptable ({metrics.negative_cbv_fraction:.1%} <= 5%)"
        )
        reasons.append("CBV constraint ENABLED for DCE vp")
    else:
        result.constraint_enabled = False
        result.status = QCStatus.FAIL

        if not cnr_ok:
            reasons.append(
                f"CNR insufficient ({metrics.cnr_fraction:.1%} < 65%)"
            )
        if not negative_ok:
            reasons.append(
                f"Excessive negative CBV ({metrics.negative_cbv_fraction:.1%} > 5%)"
            )
        reasons.append("CBV constraint DISABLED - fit vp freely")

    # Add physiological notes if concerning
    if metrics.cbv_physiological_status == QCStatus.WARN:
        reasons.append(f"Physiological concern: {metrics.cbv_physiological_note}")

    result.decision_reason = "; ".join(reasons)

    logger.info(
        f"CBV QC Decision: status={result.status}, "
        f"constraint_enabled={result.constraint_enabled}"
    )
    logger.info(f"Decision reason: {result.decision_reason}")

    return result
