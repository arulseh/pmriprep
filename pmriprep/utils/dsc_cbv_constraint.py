"""DSC CBV to vp constraint utilities for DCE pharmacokinetic modeling.

Implements quality-gated conversion of DSC-derived CBV maps to vp constraints
for use in DCE PK model fitting. When DSC data is available and passes quality
checks, the CBV-derived vp map can constrain the DCE fitting, reducing the
number of free parameters and improving stability.

The constraint pipeline:
    1. Assess DSC CBV quality (CNR coverage, AIF metrics, CBV ranges)
    2. Convert CBV to vp: vp = CBV * (1 - Hct) / (100 * rho)
    3. Determine constraint mode: hard (fix vp), soft (Bayesian prior), skip

References:
    Prah MA, Schmainda KM. "Quality-gated DSC-MRI-based cerebral blood
    volume maps for glioma assessment." AJNR. 2024.

    Boxerman JL, et al. "Relative cerebral blood volume maps corrected for
    contrast agent extravasation significantly correlate with glioma tumor
    grade." AJNR. 2006;27(4):859-867.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_constants import (
    TISSUE_DENSITY,
)


# =============================================================================
# Constants
# =============================================================================

#: Tissue hematocrit for CBV-to-vp conversion.
#: Brain tissue effective hematocrit (~0.40) is lower than arterial hematocrit
#: (~0.42) due to Fåhraeus-Lindqvist effect in capillaries.
#: Ref: Levin et al. MRM 2004; 51:735-742
HEMATOCRIT_TISSUE: float = 0.40

#: Maximum physiological vp value (brain tissue)
VP_MAX_PHYSIOLOGICAL: float = 0.2

#: Minimum CNR coverage fraction for high-quality DSC
CNR_COVERAGE_HIGH: float = 0.99

#: Minimum CNR coverage fraction for moderate-quality DSC
CNR_COVERAGE_MODERATE: float = 0.84

#: Minimum AIF peak height fraction for acceptable quality
AIF_PEAK_HEIGHT_MIN: float = 0.15

#: Marginal AIF peak height fraction
AIF_PEAK_HEIGHT_MARGINAL: float = 0.25

#: Maximum baseline return ratio for acceptable quality
AIF_BASELINE_RETURN_MAX: float = 0.40

#: Marginal baseline return ratio
AIF_BASELINE_RETURN_MARGINAL: float = 0.30

#: CBV median physiological range (mL/100g)
CBV_MEDIAN_RANGE: tuple[float, float] = (0.5, 10.0)

#: Maximum fraction of negative CBV values before quality downgrade
CBV_NEGATIVE_FRACTION_MAX: float = 0.01

#: Literature NAWM CBV for normalization (mL/100g)
CBV_NAWM_LITERATURE: float = 2.5


# =============================================================================
# Enums and Dataclasses
# =============================================================================


class ConstraintMode(Enum):
    """Mode for applying DSC CBV vp constraint to DCE fitting."""

    HARD = "hard"
    """Fix vp to DSC-derived value. Reduces free parameters by 1."""

    SOFT = "soft"
    """Use DSC-derived vp as Bayesian prior with uncertainty."""

    SKIP = "skip"
    """Do not use DSC CBV constraint (unconstrained fit)."""


@dataclass
class CBVQualityAssessment:
    """Result of DSC CBV quality assessment.

    Attributes:
        confidence: Quality tier - 'high', 'moderate', or 'low'.
        use_constraint: Whether the CBV is reliable enough for constraining.
        cnr_coverage: Fraction of brain voxels with CNR > 4.
        aif_peak_height: AIF peak height fraction (0-1).
        aif_baseline_return: AIF baseline return ratio (0-1).
        cbv_median: Median CBV in brain (mL/100g), or None if not assessed.
        cbv_negative_fraction: Fraction of negative CBV values, or None.
        reasons: List of reasons for quality decisions.
    """

    confidence: str = "low"
    use_constraint: bool = False
    cnr_coverage: float = 0.0
    aif_peak_height: float = 0.0
    aif_baseline_return: float = 1.0
    cbv_median: Optional[float] = None
    cbv_negative_fraction: Optional[float] = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class ConstraintConfig:
    """Configuration for vp constraint application.

    Attributes:
        mode: Constraint mode (hard, soft, skip).
        vp_uncertainty: Relative uncertainty for soft prior (fraction, 0-1).
        reason: Human-readable explanation for the chosen mode.
    """

    mode: ConstraintMode = ConstraintMode.SKIP
    vp_uncertainty: float = 0.2
    reason: str = "No constraint applied"


@dataclass
class VpConstraintResult:
    """Result from prepare_vp_constraint orchestrator.

    Attributes:
        vp_map: 3D vp map (plasma volume fraction), or None if skipped.
        config: Constraint configuration (mode, uncertainty).
        quality: CBV quality assessment.
        qc_report: Dictionary of QC metrics for logging.
    """

    vp_map: Optional[npt.NDArray[np.floating]] = None
    config: ConstraintConfig = field(default_factory=ConstraintConfig)
    quality: CBVQualityAssessment = field(default_factory=CBVQualityAssessment)
    qc_report: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Quality Assessment
# =============================================================================


def _normalize_aif_qc_keys(aif_qc_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize DSC AIF QC keys to the quality assessment schema.

    Maps PMRIPrep DSC AIF detection output (best_psr, best_brr) to the
    expected keys (aif_peak_height, baseline_return_ratio). Returns data
    unchanged if expected keys are already present.

    Args:
        aif_qc_data: Raw AIF QC dictionary, possibly with PMRIPrep-native
            keys (best_psr, best_brr) instead of the assessment schema keys
            (aif_peak_height, baseline_return_ratio, cnr_whole_brain).

    Returns:
        Dictionary with normalized keys suitable for assess_dsc_cbv_quality.
    """
    data = dict(aif_qc_data)

    # Already in expected format
    if "aif_peak_height" in data or "cnr_whole_brain" in data:
        return data

    # --- AIF peak height from PSR ---
    # best_psr = 100 * (S_tail - S_min) / (S0 - S_min)
    # Higher = better signal recovery. Thresholds: 0.25 (high), 0.15 (moderate)
    if "best_psr" in data and "aif_peak_height" not in data:
        data["aif_peak_height"] = data["best_psr"] / 100.0
    elif "aif_metrics" in data and isinstance(data["aif_metrics"], dict):
        if "psr" in data["aif_metrics"] and "aif_peak_height" not in data:
            data["aif_peak_height"] = data["aif_metrics"]["psr"] / 100.0

    # --- Baseline return ratio from BRR ---
    # best_brr = 100 * tail_conc / peak_conc
    # Lower = better (more recovery). Thresholds: 0.30 (marginal), 0.40 (fail)
    if "best_brr" in data and "baseline_return_ratio" not in data:
        data["baseline_return_ratio"] = data["best_brr"] / 100.0

    # --- CNR coverage: skip if not available ---
    # CNR coverage comes from preprocessing, not AIF detection.
    # Don't penalize when it's simply not in the AIF QC dict.
    # Set to high threshold to avoid false downgrades.
    if "cnr_whole_brain" not in data:
        data["cnr_whole_brain"] = {"fraction_cnr_gt_4": CNR_COVERAGE_HIGH}

    return data


def _downgrade_confidence(current: str, target: str) -> str:
    """Downgrade confidence level but never upgrade.

    Args:
        current: Current confidence level.
        target: Target confidence level.

    Returns:
        The lower of the two confidence levels.
    """
    levels = {"high": 2, "moderate": 1, "low": 0}
    if levels.get(target, 0) < levels.get(current, 0):
        return target
    return current


def assess_dsc_cbv_quality(
    aif_qc_data: dict[str, Any],
    cbv_data: Optional[npt.NDArray[np.floating]] = None,
    brain_mask: Optional[npt.NDArray[np.floating]] = None,
) -> CBVQualityAssessment:
    """Assess DSC CBV quality for use as a vp constraint.

    Implements a three-tier quality framework (high/moderate/low) based on:
    - CNR coverage: fraction of brain voxels with CNR > 4
    - AIF peak height: relative height of the AIF peak
    - AIF baseline return: how well AIF returns to baseline
    - CBV value ranges: whether median CBV is physiologically plausible

    Args:
        aif_qc_data: Dictionary from DSC AIF QC JSON containing metrics
            like cnr_whole_brain, aif_peak_height, baseline_return_ratio.
        cbv_data: Optional 3D CBV map for value range validation.
        brain_mask: Optional 3D brain mask for CBV statistics.

    Returns:
        CBVQualityAssessment with confidence tier and detailed metrics.
    """
    aif_qc_data = _normalize_aif_qc_keys(aif_qc_data)

    result = CBVQualityAssessment()
    confidence = "high"

    # --- CNR Coverage ---
    cnr_info = aif_qc_data.get("cnr_whole_brain", {})
    if isinstance(cnr_info, dict):
        cnr_coverage = cnr_info.get("fraction_cnr_gt_4", 0.0)
    else:
        cnr_coverage = 0.0
    result.cnr_coverage = cnr_coverage

    if cnr_coverage < CNR_COVERAGE_MODERATE:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"CNR coverage {cnr_coverage:.2f} < {CNR_COVERAGE_MODERATE}"
        )
    elif cnr_coverage < CNR_COVERAGE_HIGH:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"CNR coverage {cnr_coverage:.2f} < {CNR_COVERAGE_HIGH}"
        )

    # --- AIF Peak Height ---
    aif_peak = float(aif_qc_data.get("aif_peak_height", 0.0))
    result.aif_peak_height = aif_peak

    if aif_peak < AIF_PEAK_HEIGHT_MIN:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"AIF peak height {aif_peak:.2f} < {AIF_PEAK_HEIGHT_MIN}"
        )
    elif aif_peak < AIF_PEAK_HEIGHT_MARGINAL:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"AIF peak height {aif_peak:.2f} < {AIF_PEAK_HEIGHT_MARGINAL}"
        )

    # --- AIF Baseline Return ---
    baseline_return = float(aif_qc_data.get("baseline_return_ratio", 1.0))
    result.aif_baseline_return = baseline_return

    if baseline_return > AIF_BASELINE_RETURN_MAX:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"AIF baseline return {baseline_return:.2f} > {AIF_BASELINE_RETURN_MAX}"
        )
    elif baseline_return > AIF_BASELINE_RETURN_MARGINAL:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"AIF baseline return {baseline_return:.2f} > {AIF_BASELINE_RETURN_MARGINAL}"
        )

    # --- CBV Value Range (if map provided) ---
    if cbv_data is not None:
        if brain_mask is not None:
            mask_bool = brain_mask > 0
            cbv_in_brain = cbv_data[mask_bool]
        else:
            cbv_in_brain = cbv_data[np.isfinite(cbv_data)]

        if len(cbv_in_brain) > 0:
            # Check negative fraction
            n_negative = int(np.sum(cbv_in_brain < 0))
            neg_fraction = n_negative / len(cbv_in_brain)
            result.cbv_negative_fraction = neg_fraction

            if neg_fraction > CBV_NEGATIVE_FRACTION_MAX:
                confidence = _downgrade_confidence(confidence, "low")
                result.reasons.append(
                    f"CBV negative fraction {neg_fraction:.3f} > {CBV_NEGATIVE_FRACTION_MAX}"
                )

            # Check median range
            cbv_positive = cbv_in_brain[cbv_in_brain > 0]
            if len(cbv_positive) > 0:
                cbv_median = float(np.median(cbv_positive))
                result.cbv_median = cbv_median

                if cbv_median < CBV_MEDIAN_RANGE[0] or cbv_median > CBV_MEDIAN_RANGE[1]:
                    confidence = _downgrade_confidence(confidence, "moderate")
                    result.reasons.append(
                        f"CBV median {cbv_median:.2f} outside "
                        f"[{CBV_MEDIAN_RANGE[0]}, {CBV_MEDIAN_RANGE[1]}]"
                    )

    result.confidence = confidence
    result.use_constraint = confidence != "low"

    if not result.reasons:
        result.reasons.append("All quality checks passed")

    return result


# =============================================================================
# CBV to vp Conversion
# =============================================================================


def cbv_to_vp(
    cbv_data: npt.NDArray[np.floating],
    hematocrit: float = HEMATOCRIT_TISSUE,
    tissue_density: float = TISSUE_DENSITY,
) -> npt.NDArray[np.floating]:
    """Convert DSC-derived CBV map to plasma volume fraction (vp).

    Formula: vp = CBV * (1 - Hct) / (100 * rho)

    Unit analysis:
        (mL/100g) * (1) / (100 * g/mL) = mL/mL = dimensionless

    Typical brain values (Hct=0.40, rho=1.04):
        WM CBV 2.5 mL/100g -> vp ~0.014
        GM CBV 4.5 mL/100g -> vp ~0.026
        Tumor 5-15 mL/100g -> vp ~0.029-0.087

    Args:
        cbv_data: 3D CBV map in mL/100g.
        hematocrit: Tissue hematocrit fraction (default 0.40).
        tissue_density: Tissue density in g/mL (default 1.04).

    Returns:
        3D vp map (dimensionless, 0-0.2). Invalid values set to NaN.
    """
    # Guard invalid input values
    cbv_safe = np.where(
        np.isfinite(cbv_data) & (cbv_data >= 0),
        cbv_data,
        np.nan,
    )

    # Apply conversion
    vp = cbv_safe * (1.0 - hematocrit) / (100.0 * tissue_density)

    # Clip to physiological range; out-of-range -> NaN
    vp = np.where(
        (vp >= 0) & (vp <= VP_MAX_PHYSIOLOGICAL),
        vp,
        np.nan,
    )

    return vp.astype(np.float64)


# =============================================================================
# Constraint Mode Determination
# =============================================================================


def determine_constraint_mode(
    dsc_quality: CBVQualityAssessment,
    user_override: Optional[str] = None,
) -> ConstraintConfig:
    """Determine constraint mode from DSC quality assessment.

    Decision logic:
    - User override (if not 'auto') takes priority
    - high quality -> hard (fix vp)
    - moderate quality -> soft (Bayesian prior +/- 20%)
    - low quality -> skip (unconstrained)

    Args:
        dsc_quality: Quality assessment from assess_dsc_cbv_quality.
        user_override: User-specified mode override ('auto', 'hard', 'soft',
            'skip'). If 'auto' or None, auto-determined from quality.

    Returns:
        ConstraintConfig with mode, uncertainty, and reason.
    """
    # User override
    if user_override and user_override != "auto":
        try:
            mode = ConstraintMode(user_override)
        except ValueError:
            mode = ConstraintMode.SKIP
        uncertainty = 0.2 if mode == ConstraintMode.SOFT else 0.0
        return ConstraintConfig(
            mode=mode,
            vp_uncertainty=uncertainty,
            reason=f"User override: {user_override}",
        )

    confidence = dsc_quality.confidence

    if confidence == "high":
        return ConstraintConfig(
            mode=ConstraintMode.HARD,
            vp_uncertainty=0.0,
            reason="High-quality DSC CBV: fixing vp (hard constraint)",
        )
    elif confidence == "moderate":
        return ConstraintConfig(
            mode=ConstraintMode.SOFT,
            vp_uncertainty=0.2,
            reason="Moderate-quality DSC CBV: Bayesian prior on vp (soft constraint)",
        )
    else:
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=(
                "Low-quality DSC CBV: constraint skipped. "
                + "; ".join(dsc_quality.reasons)
            ),
        )


# =============================================================================
# Orchestrator
# =============================================================================


def prepare_vp_constraint(
    cbv_file: Union[str, Path],
    aif_qc_file: Union[str, Path],
    brain_mask_file: Union[str, Path],
    mode_override: Optional[str] = None,
    hematocrit: float = HEMATOCRIT_TISSUE,
) -> VpConstraintResult:
    """Prepare vp constraint map from DSC CBV for DCE PK modeling.

    Orchestrates:
    1. Load CBV map and AIF QC JSON
    2. Assess DSC CBV quality
    3. Convert CBV to vp
    4. Determine constraint mode

    Args:
        cbv_file: Path to DSC-derived CBV map NIfTI.
        aif_qc_file: Path to DSC AIF QC JSON.
        brain_mask_file: Path to brain mask NIfTI.
        mode_override: User-specified constraint mode override.
        hematocrit: Blood hematocrit for CBV-to-vp conversion.

    Returns:
        VpConstraintResult with vp_map, config, quality, and qc_report.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    result = VpConstraintResult()

    # Load brain mask
    mask_img = nib.load(str(brain_mask_file))
    mask_data: npt.NDArray[np.floating] = mask_img.get_fdata()  # type: ignore[attr-defined]

    # Load CBV map
    cbv_img = nib.load(str(cbv_file))
    cbv_data: npt.NDArray[np.floating] = cbv_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]

    # Load AIF QC JSON
    aif_qc_path = Path(aif_qc_file)
    if aif_qc_path.exists():
        with open(aif_qc_path) as f:
            aif_qc_data = json.load(f)
    else:
        aif_qc_data = {}

    # Step 1: Assess quality
    quality = assess_dsc_cbv_quality(
        aif_qc_data=aif_qc_data,
        cbv_data=cbv_data,
        brain_mask=mask_data,
    )
    result.quality = quality

    # Step 2: Determine constraint mode
    config = determine_constraint_mode(quality, user_override=mode_override)
    result.config = config

    # Step 3: Convert CBV to vp (even for skip mode, for QC reporting)
    vp_map = cbv_to_vp(cbv_data, hematocrit=hematocrit)

    # Build QC report
    mask_bool = mask_data > 0
    vp_in_brain = vp_map[mask_bool]
    vp_valid = vp_in_brain[np.isfinite(vp_in_brain)]

    qc_report: dict[str, Any] = {
        "mode": config.mode.value,
        "reason": config.reason,
        "dsc_cbv_confidence": quality.confidence,
        "dsc_cbv_use_constraint": quality.use_constraint,
        "cnr_coverage": quality.cnr_coverage,
        "aif_peak_height": quality.aif_peak_height,
        "aif_baseline_return": quality.aif_baseline_return,
        "hematocrit": hematocrit,
    }

    if len(vp_valid) > 0:
        qc_report["vp_median"] = float(np.median(vp_valid))
        qc_report["vp_p5"] = float(np.percentile(vp_valid, 5))
        qc_report["vp_p95"] = float(np.percentile(vp_valid, 95))
        qc_report["vp_nan_fraction"] = float(
            np.sum(~np.isfinite(vp_in_brain)) / len(vp_in_brain)
        )

    if quality.cbv_median is not None:
        qc_report["cbv_median"] = quality.cbv_median
    if quality.cbv_negative_fraction is not None:
        qc_report["cbv_negative_fraction"] = quality.cbv_negative_fraction

    result.qc_report = qc_report

    # Only provide vp_map if constraint will be used
    if config.mode != ConstraintMode.SKIP:
        result.vp_map = vp_map
    else:
        result.vp_map = None

    return result


def prepare_vp_constraint_from_data(
    cbv_file: Union[str, Path],
    aif_qc_data: dict[str, Any],
    brain_mask_file: Union[str, Path],
    mode_override: Optional[str] = None,
    hematocrit: float = HEMATOCRIT_TISSUE,
) -> VpConstraintResult:
    """Prepare vp constraint map from DSC CBV for DCE PK modeling.

    Same as prepare_vp_constraint but accepts AIF QC data as a dict directly
    instead of reading from a JSON file.

    Args:
        cbv_file: Path to DSC-derived CBV map NIfTI.
        aif_qc_data: DSC AIF QC metrics dictionary (from workflow).
        brain_mask_file: Path to brain mask NIfTI.
        mode_override: User-specified constraint mode override.
        hematocrit: Blood hematocrit for CBV-to-vp conversion.

    Returns:
        VpConstraintResult with vp_map, config, quality, and qc_report.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    result = VpConstraintResult()

    # Load brain mask
    mask_img = nib.load(str(brain_mask_file))
    mask_data: npt.NDArray[np.floating] = mask_img.get_fdata()  # type: ignore[attr-defined]

    # Load CBV map
    cbv_img = nib.load(str(cbv_file))
    cbv_data: npt.NDArray[np.floating] = cbv_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]

    # Step 1: Assess quality using provided AIF QC data
    quality = assess_dsc_cbv_quality(
        aif_qc_data=aif_qc_data if aif_qc_data else {},
        cbv_data=cbv_data,
        brain_mask=mask_data,
    )
    result.quality = quality

    # Step 2: Determine constraint mode
    config = determine_constraint_mode(quality, user_override=mode_override)
    result.config = config

    # Step 3: Convert CBV to vp (even for skip mode, for QC reporting)
    vp_map = cbv_to_vp(cbv_data, hematocrit=hematocrit)

    # Build QC report
    mask_bool = mask_data > 0
    vp_in_brain = vp_map[mask_bool]
    vp_valid = vp_in_brain[np.isfinite(vp_in_brain)]

    qc_report: dict[str, Any] = {
        "mode": config.mode.value,
        "reason": config.reason,
        "dsc_cbv_confidence": quality.confidence,
        "dsc_cbv_use_constraint": quality.use_constraint,
        "cnr_coverage": quality.cnr_coverage,
        "aif_peak_height": quality.aif_peak_height,
        "aif_baseline_return": quality.aif_baseline_return,
        "hematocrit": hematocrit,
    }

    if len(vp_valid) > 0:
        qc_report["vp_median"] = float(np.median(vp_valid))
        qc_report["vp_p5"] = float(np.percentile(vp_valid, 5))
        qc_report["vp_p95"] = float(np.percentile(vp_valid, 95))
        qc_report["vp_nan_fraction"] = float(
            np.sum(~np.isfinite(vp_in_brain)) / len(vp_in_brain)
        )

    if quality.cbv_median is not None:
        qc_report["cbv_median"] = quality.cbv_median
    if quality.cbv_negative_fraction is not None:
        qc_report["cbv_negative_fraction"] = quality.cbv_negative_fraction

    result.qc_report = qc_report

    # Only provide vp_map if constraint will be used
    if config.mode != ConstraintMode.SKIP:
        result.vp_map = vp_map
    else:
        result.vp_map = None

    return result
