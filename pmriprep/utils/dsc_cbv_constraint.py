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
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

from pmriprep.utils.dce_constants import (
    HEMATOCRIT_ARTERIAL,
    TISSUE_DENSITY,
    capillary_hematocrit,
    NAWM_VP_REFERENCE,
    VP_MAX_PHYSIOLOGICAL,
    VP_MEDIAN_MIN,
    VP_MEDIAN_MAX,
    SIGMA_SOFT_DEFAULT,
    SIGMA_SOFT_PRELOAD_SUPPRESSED,
    SIGMA_SOFT_HIGH_K2,
)
from pmriprep.utils.dsc_aif import (
    GAMMA_PEAK_RATIO_MIN,
    GAMMA_PEAK_RATIO_MAX,
)
from pmriprep.utils.dsc_constants import (
    GD_DOTA_STANDARD_DOSE,
    K2_SIGMA_CORRECTION,
    K2_SIGMA_SUPPRESSED,
    LEAKY_SIGMA_HIGH,
    LEAKY_SIGMA_LOW,
    QC_MEAN_CNR_MINIMUM,
)


# =============================================================================
# Constants
# =============================================================================


#: Minimum CNR coverage fraction for high-quality DSC (relaxed from 0.99)
#: Allow 5% low-CNR voxels due to sinuses, edge effects
CNR_COVERAGE_HIGH: float = 0.95

#: Minimum CNR coverage fraction for moderate-quality DSC
CNR_COVERAGE_MODERATE: float = 0.84

#: Minimum CNR coverage fraction before skipping constraint
#: Below this, >30% of brain has unreliable CBV
CNR_COVERAGE_SKIP: float = 0.70

#: Minimum mean CNR for CBV constraint (Rose criterion)
#: Imported from dsc_constants for consistency
CNR_MEAN_MINIMUM: float = QC_MEAN_CNR_MINIMUM

#: Minimum AIF peak height fraction for acceptable quality
AIF_PEAK_HEIGHT_MIN: float = 0.15

#: Marginal AIF peak height fraction
AIF_PEAK_HEIGHT_MARGINAL: float = 0.25

#: Maximum baseline return ratio for acceptable quality
AIF_BASELINE_RETURN_MAX: float = 0.40

#: Marginal baseline return ratio
AIF_BASELINE_RETURN_MARGINAL: float = 0.30

#: nCBV median acceptable range for the vp-constraint feeder (dimensionless,
#: NAWM-normalized). NAWM=1.0 by construction; normal whole-brain median
#: ~0.7–1.5; tumor-heavy subjects up to ~2–3. Outside this range, the
#: normalization reference or the underlying CBV map is compromised.
#: Note: MAX_NCBV_FOR_VALIDATION=5.0 is a PER-VOXEL upper bound that
#: accommodates hypervascular HGG tumor-rim (Law 2008, Hu 2009: rCBV up
#: to 6 in GBM); this median range is for the whole-brain distribution.
CBV_MEDIAN_RANGE: tuple[float, float] = (0.3, 3.0)

#: Maximum fraction of negative CBV values before quality downgrade
CBV_NEGATIVE_FRACTION_MAX: float = 0.01

#: Literature NAWM CBV for normalization (mL/100g)
CBV_NAWM_LITERATURE: float = 2.5

# Composite K2 criteria for preload-aware uncertainty adjustment (sigma tier).
# Thresholds colocated in dsc_constants — see K2_SIGMA_* and LEAKY_SIGMA_*
# there for literature references. This tier is coarser than the primary
# classification tier (K2_PRIMARY_*) used by classify_preload_adequacy_from_k2;
# sigma tuning tolerates looser thresholds than discrete class assignment.

# Gamma peak ratio thresholds for constraint quality gating.
# Moderate band [0.90, 1.10] imported from dsc_aif.py (single source of truth).
# Wider low-confidence band captures severe AIF corruption.
# Ref: peak error >12% exceeds normal CBF CoV (~10%, Mutsaerts JCBFM 2015).
GAMMA_PEAK_RATIO_LOW_MIN: float = 0.80
GAMMA_PEAK_RATIO_LOW_MAX: float = 1.20


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
        gamma_peak_ratio: Gamma-variate peak ratio (fitted/measured), or None.
        reasons: List of reasons for quality decisions.
    """

    confidence: str = "low"
    use_constraint: bool = False
    cnr_coverage: float = 0.0
    aif_peak_height: float = 0.0
    aif_baseline_return: float = 1.0
    cbv_median: Optional[float] = None
    cbv_negative_fraction: Optional[float] = None
    preload_status: Optional[str] = None
    detection_level: Optional[str] = None
    n_aif_voxels: Optional[int] = None
    gamma_peak_ratio: Optional[float] = None
    reasons: list[str] = field(default_factory=list)
    #: Cached (is_preloaded, well_suppressed, high_correction) triple from
    #: :func:`_classify_preload_k2`; populated during quality assessment so
    #: :func:`determine_constraint_mode` can reuse the classification instead
    #: of recomputing it. Avoids a second source of truth for the "no
    #: preload + high K2 correction" decision.
    preload_k2_classification: Optional[tuple[bool, bool, bool]] = None


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

    # --- Gamma peak ratio from auto-fit metrics ---
    # Extract before the early return since gamma_fit_auto_metrics is present
    # in both normalized and raw formats.
    if "gamma_peak_ratio" not in data:
        auto_metrics = data.get("gamma_fit_auto_metrics", {})
        if isinstance(auto_metrics, dict) and "peak_ratio" in auto_metrics:
            data["gamma_peak_ratio"] = auto_metrics["peak_ratio"]

    # Helper to set CNR default with warning (avoids duplication)
    def _set_cnr_default(d: dict[str, Any]) -> None:
        if "cnr_whole_brain" not in d:
            logger.warning(
                "CNR coverage not in AIF QC data; defaulting to %.2f (moderate). "
                "CBV constraint quality assessment may be optimistic.",
                CNR_COVERAGE_MODERATE,
            )
            d["cnr_whole_brain"] = {"fraction_cnr_gt_4": CNR_COVERAGE_MODERATE}

    # Already in expected format (skip PSR/BRR conversion if aif_peak_height present)
    if "aif_peak_height" in data:
        _set_cnr_default(data)
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

    # --- CNR coverage: default to moderate if not available ---
    _set_cnr_default(data)

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


def _classify_preload_k2(
    preload_classification: Optional[str],
    k2_metrics: Optional[dict[str, Any]],
) -> tuple[bool, bool, bool]:
    """Classify preload/K2 status for quality and uncertainty logic.

    Args:
        preload_classification: Preload classification string from DSC QC.
        k2_metrics: K2 leakage correction metrics from dsc_preload.

    Returns:
        Tuple of (is_preloaded, well_suppressed, high_correction).
    """
    if not k2_metrics or not preload_classification:
        return False, False, False

    k2_med = float(k2_metrics.get("k2_median", 0.0))
    pct_leaky = float(k2_metrics.get("percent_leaky_voxels", 0.0))

    is_preloaded = preload_classification in (
        "half_dose", "significant", "severe",
    )
    well_suppressed = k2_med < K2_SIGMA_SUPPRESSED and pct_leaky < LEAKY_SIGMA_LOW
    high_correction = pct_leaky > LEAKY_SIGMA_HIGH or k2_med > K2_SIGMA_CORRECTION

    return is_preloaded, well_suppressed, high_correction


def _bolus_gd_fraction(
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> float:
    """Bolus Gd dose as a fraction of the Gd-DOTA reference (0.1 mmol Gd/kg).

    Peak first-pass ΔR2* and CNR scale linearly with the product
    ``dose × agent_standard_dose`` (Rose criterion; Boxerman AJNR 2006;
    Welker AJNR 2015). CNR coverage thresholds calibrated at full
    standard dose must be scaled by this fraction to remain valid on
    reduced-dose protocols or lower-dose agents.

    Args:
        dose: Bolus dose multiplier (1.0 = full standard, 0.5 = half).
            Must be positive.
        agent_standard_dose: Agent's standard clinical dose in mmol Gd/kg.
            Must be positive.

    Returns:
        Effective Gd fraction relative to 0.1 mmol/kg reference.

    Raises:
        ValueError: If ``dose`` or ``agent_standard_dose`` is non-positive.
            A zero or negative dose would collapse the CNR gate and let
            garbage CBV through unchecked — fail loudly instead.
    """
    dose_f = float(dose)
    agent_f = float(agent_standard_dose)
    if dose_f <= 0:
        raise ValueError(f"dose must be positive, got {dose_f}")
    if agent_f <= 0:
        raise ValueError(
            f"agent_standard_dose must be positive, got {agent_f}"
        )
    return dose_f * agent_f / GD_DOTA_STANDARD_DOSE


def assess_dsc_cbv_quality(
    aif_qc_data: dict[str, Any],
    cbv_data: Optional[npt.NDArray[np.floating]] = None,
    brain_mask: Optional[npt.NDArray[np.floating]] = None,
    preload_classification: Optional[str] = None,
    k2_metrics: Optional[dict[str, Any]] = None,
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> CBVQualityAssessment:
    """Assess DSC CBV quality for use as a vp constraint.

    Implements a three-tier quality framework (high/moderate/low) based on:
    - CNR coverage: fraction of brain voxels with CNR > 4
    - AIF peak height: relative height of the AIF peak
    - AIF baseline return: how well AIF returns to baseline
    - CBV value ranges: whether median CBV is physiologically plausible

    CNR coverage thresholds scale with the administered Gd dose
    (``dose × agent_standard_dose`` relative to 0.1 mmol/kg reference).
    Half-dose protocols and lower-dose agents (gadopiclenol 0.05,
    gadoquatrane 0.04) produce inherently lower CNR; scaling prevents
    spurious SKIP decisions on otherwise-adequate acquisitions.

    Args:
        aif_qc_data: Dictionary from DSC AIF QC JSON containing metrics
            like cnr_whole_brain, aif_peak_height, baseline_return_ratio.
        cbv_data: Optional 3D CBV map for value range validation.
        brain_mask: Optional 3D brain mask for CBV statistics.
        preload_classification: DSC preload classification string.
        k2_metrics: K2 leakage correction metrics from dsc_preload.
        dose: DSC bolus dose multiplier (1.0 = full standard).
        agent_standard_dose: Agent's standard dose in mmol Gd/kg
            (default 0.1 for Gd-DOTA class).

    Returns:
        CBVQualityAssessment with confidence tier and detailed metrics.
    """
    aif_qc_data = _normalize_aif_qc_keys(aif_qc_data)

    result = CBVQualityAssessment()
    confidence = "high"

    # Scale CNR coverage thresholds by the effective bolus Gd fraction.
    # Default (full-dose Gd-DOTA) gives eff=1.0 and preserves v0.25.1
    # thresholds exactly.
    eff_dose = _bolus_gd_fraction(dose, agent_standard_dose)
    cnr_skip = CNR_COVERAGE_SKIP * eff_dose
    cnr_moderate = CNR_COVERAGE_MODERATE * eff_dose
    cnr_high = CNR_COVERAGE_HIGH * eff_dose

    # AIF detection tier is read here (not from result.detection_level,
    # which isn't set yet) to allow the AIF-tier fast-pass below to bypass
    # the CNR-coverage SKIP for high-quality atlas-based AIF detections.
    _detection_level_for_cnr = str(aif_qc_data.get("detection_level", ""))
    aif_high_quality = _detection_level_for_cnr in {
        "atlas_strict",
        "atlas_relaxed",
    }

    # --- CNR Coverage and Mean CNR ---
    cnr_info = aif_qc_data.get("cnr_whole_brain", {})
    if isinstance(cnr_info, dict):
        cnr_coverage = cnr_info.get("fraction_cnr_gt_4", 0.0)
        mean_cnr = cnr_info.get("mean_cnr")
    else:
        cnr_coverage = 0.0
        mean_cnr = None
    result.cnr_coverage = cnr_coverage

    # Mean CNR failsafe: catch total acquisition failures (Rose criterion)
    if mean_cnr is not None and mean_cnr < CNR_MEAN_MINIMUM:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"Mean CNR {mean_cnr:.1f} < {CNR_MEAN_MINIMUM} (inadequate contrast)"
        )
    elif cnr_coverage < cnr_skip and not aif_high_quality:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"CNR coverage {cnr_coverage:.1%} < {cnr_skip:.1%} "
            f"(dose-scaled, eff={eff_dose:.2f}); >30% unreliable and AIF "
            f"tier '{_detection_level_for_cnr}' does not qualify for bypass"
        )
    elif cnr_coverage < cnr_skip and aif_high_quality:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"CNR coverage {cnr_coverage:.1%} < {cnr_skip:.1%} "
            f"(dose-scaled) but AIF tier '{_detection_level_for_cnr}' "
            f"retained at moderate confidence"
        )
    elif cnr_coverage < cnr_moderate and not aif_high_quality:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"CNR coverage {cnr_coverage:.1%} < {cnr_moderate:.1%} "
            f"(dose-scaled, eff={eff_dose:.2f})"
        )
    elif cnr_coverage < cnr_moderate and aif_high_quality:
        # atlas_strict/atlas_relaxed in the [SKIP, MODERATE) band: treat
        # CNR as informational, keep at moderate confidence rather than
        # downgrading to low (which would drive SKIP). The bypass applies
        # across the whole sub-MODERATE CNR range so AIF tier carries the
        # decision when CNR is marginal.
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"CNR coverage {cnr_coverage:.1%} < {cnr_moderate:.1%} "
            f"(dose-scaled) but AIF tier '{_detection_level_for_cnr}' "
            f"retained at moderate confidence"
        )
    elif cnr_coverage < cnr_high:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"CNR coverage {cnr_coverage:.1%} < {cnr_high:.1%} "
            f"(dose-scaled, eff={eff_dose:.2f})"
        )

    # --- AIF Detection Level ---
    # A synthetic AIF produces unreliable CBV regardless of shape metrics.
    # Note: brain_degraded removed in v0.19.16 - cascade now falls through
    # to synthetic when permissive fails.
    _AIF_LEVELS_LOW = {"synthetic"}
    _AIF_LEVELS_MODERATE = {"brain_permissive", "brain_relaxed", "atlas_permissive"}

    detection_level = str(aif_qc_data.get("detection_level", ""))
    result.detection_level = detection_level or None

    if detection_level in _AIF_LEVELS_LOW:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"AIF detection level '{detection_level}' — CBV unreliable"
        )
    elif detection_level in _AIF_LEVELS_MODERATE:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"AIF detection level '{detection_level}' — CBV quality uncertain"
        )

    # --- AIF Candidate Count ---
    # Single-voxel AIF has no noise averaging; AIF SNR < 20 degrades
    # SVD deconvolution (Wu et al. MRM 2003).  Default 3 when key is
    # absent for backward compatibility (no penalty).
    n_candidates = int(aif_qc_data.get(
        "n_aif_voxels_used",
        aif_qc_data.get("n_candidates_effective", 3),
    ))
    result.n_aif_voxels = n_candidates

    if n_candidates < 2:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"Only {n_candidates} AIF voxel(s) — no noise averaging"
        )
    elif n_candidates < 3:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"Only {n_candidates} AIF voxels — limited averaging"
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

    # --- Gamma Peak Ratio ---
    # Peak ratio = gamma_fit_peak / measured_peak. Values far from 1.0
    # indicate AIF shape corruption (motion, partial volume, truncation).
    # A corrupted AIF produces biased CBV for vp constraint.
    gamma_peak_ratio = aif_qc_data.get("gamma_peak_ratio")
    if gamma_peak_ratio is not None:
        result.gamma_peak_ratio = gamma_peak_ratio
        if (gamma_peak_ratio < GAMMA_PEAK_RATIO_LOW_MIN
                or gamma_peak_ratio > GAMMA_PEAK_RATIO_LOW_MAX):
            confidence = _downgrade_confidence(confidence, "low")
            result.reasons.append(
                f"Gamma peak ratio {gamma_peak_ratio:.3f} outside "
                f"[{GAMMA_PEAK_RATIO_LOW_MIN}, {GAMMA_PEAK_RATIO_LOW_MAX}]"
            )
        elif (gamma_peak_ratio < GAMMA_PEAK_RATIO_MIN
                or gamma_peak_ratio > GAMMA_PEAK_RATIO_MAX):
            confidence = _downgrade_confidence(confidence, "moderate")
            result.reasons.append(
                f"Gamma peak ratio {gamma_peak_ratio:.3f} outside "
                f"[{GAMMA_PEAK_RATIO_MIN}, {GAMMA_PEAK_RATIO_MAX}]"
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

    result.preload_status = preload_classification
    classification = _classify_preload_k2(
        preload_classification, k2_metrics,
    )
    result.preload_k2_classification = classification
    is_preloaded, _well_suppressed, high_correction = classification
    if not is_preloaded and high_correction:
        k2_med = float((k2_metrics or {}).get("k2_median", 0.0))
        pct_leaky = float((k2_metrics or {}).get("percent_leaky_voxels", 0.0))
        result.reasons.append(
            f"No preload with substantial leakage correction "
            f"(K2 median={k2_med:.4f}, leaky={pct_leaky:.1f}%): "
            "CBV relies on large K2 correction"
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
    hematocrit: float = HEMATOCRIT_ARTERIAL,
    tissue_density: float = TISSUE_DENSITY,
) -> npt.NDArray[np.floating]:
    """Convert DSC-derived CBV map to plasma volume fraction (vp).

    Formula: vp = CBV * (1 - Hct_capillary) / (100 * rho)

    Capillary hematocrit is derived from arterial via Fåhræus ratio (0.85).

    Unit analysis:
        (mL/100g) * (1) / (100 * g/mL) = mL/mL = dimensionless

    Typical brain values (arterial Hct=0.42, capillary=0.357, rho=1.04):
        WM CBV 2.5 mL/100g -> vp ~0.015
        GM CBV 4.5 mL/100g -> vp ~0.028
        Tumor 5-15 mL/100g -> vp ~0.031-0.093

    Args:
        cbv_data: 3D CBV map in mL/100g.
        hematocrit: Arterial hematocrit (default 0.42). Capillary Hct
            derived internally via Fåhræus ratio.
        tissue_density: Tissue density in g/mL (default 1.04).

    Returns:
        3D vp map (dimensionless, 0-0.2). Invalid values set to NaN.
    """
    hct_cap = capillary_hematocrit(hematocrit)

    # Guard invalid input values
    cbv_safe = np.where(
        np.isfinite(cbv_data) & (cbv_data >= 0),
        cbv_data,
        np.nan,
    )

    # Apply conversion
    vp = cbv_safe * (1.0 - hct_cap) / (100.0 * tissue_density)

    # Clip to physiological range; out-of-range -> NaN
    vp = np.where(
        (vp >= 0) & (vp <= VP_MAX_PHYSIOLOGICAL),
        vp,
        np.nan,
    )

    return vp.astype(np.float64)


def ncbv_to_vp_prior(
    ncbv_data: npt.NDArray[np.floating],
    nawm_vp_reference: float = NAWM_VP_REFERENCE,
) -> npt.NDArray[np.floating]:
    """Convert NAWM-normalized CBV (nCBV) to a vp prior map.

    Assumes NAWM vp ≈ 0.02 (Rostrup 2000, ASFNR Boxerman 2020) and scales
    linearly with the per-voxel nCBV ratio. Because nCBV cancels AIF
    partial-volume, recirculation, and first-pass-truncation biases (all
    affect numerator and denominator equally), this is the scientifically
    defensible conversion from a DSC-derived relative volume measure to a
    DCE vp prior, avoiding the unit ambiguities that plague direct
    absolute-CBV → vp conversion with `cbv_to_vp`.

    Hct convention: NAWM_VP_REFERENCE = 0.02 is the capillary-Hct-consistent
    NAWM plasma volume fraction (Rostrup 2000 reports NAWM CBV ≈ 2.5 mL/100g;
    with capillary Hct from Fåhræus ratio, vp ≈ (1-Hct_cap)·ρ·CBV/100 ≈ 0.016-
    0.02). This is distinct from the arterial Hct used in ``dce_aif.py``'s
    ``/(1-Hct_arterial)`` blood→plasma AIF conversion. The two Hct
    conventions are physically independent and correctly applied to their
    respective scales.

    Args:
        ncbv_data: 3D nCBV map (CBV / median(CBV_NAWM)), dimensionless.
        nawm_vp_reference: Assumed NAWM vp. Default 0.02.

    Returns:
        3D vp prior map (dimensionless), clipped to [0, VP_MAX_PHYSIOLOGICAL].

    References:
        Rostrup E et al., NeuroImage 2000 (NAWM CBV ~2.5 mL/100g).
        Boxerman JL et al., Neuro-Oncology 2020 (ASFNR/NCI consensus —
        contralateral NAWM is the standard normalization reference).
    """
    ncbv_safe = np.where(
        np.isfinite(ncbv_data) & (ncbv_data >= 0),
        ncbv_data,
        np.nan,
    )
    vp = ncbv_safe * nawm_vp_reference
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
    preload_classification: Optional[str] = None,
    k2_metrics: Optional[dict[str, Any]] = None,
) -> ConstraintConfig:
    """Determine constraint mode from DSC quality assessment.

    Decision logic (v0.25.0+ per 2026-04 physics review):

    - User override (if not 'auto') takes priority. Explicit HARD emits a
      warning because DSC CBV and DCE vp measure subtly different vascular
      compartments (all-vessel vs capillary-plasma); HARD equality is an
      unsupported physics claim even when CBV calibration is correct.
    - **auto never emits HARD.** High quality → tight SOFT (σ=0.20);
      moderate → SOFT with sigma that depends on preload/K2 status;
      low → SKIP (unconstrained).

    Soft sigma values (widened from prior 0.2/0.12/0.30 per physics review):
    DSC→DCE scale uncertainty is genuinely 40–50% from compartment differences
    plus preload/K2 residuals. See SIGMA_SOFT_* in dce_constants.

    Args:
        dsc_quality: Quality assessment from assess_dsc_cbv_quality.
        user_override: User-specified mode override ('auto', 'hard', 'soft',
            'skip'). If 'auto' or None, auto-determined from quality.
        preload_classification: DSC preload classification string from QC
            (e.g., 'half_dose', 'significant', 'severe', 'none').
        k2_metrics: K2 leakage correction metrics dict from dsc_preload,
            containing 'k2_median', 'percent_leaky_voxels', etc.

    Returns:
        ConstraintConfig with mode, uncertainty, and reason. HARD override
        additionally annotates the config's ``reason`` with a physics caveat
        so callers can surface the warning in QC / HTML reports.
    """
    # UNKNOWN preload → SKIP (physics consult 2026-04-18: more conservative
    # than defaulting to NONE, which would silently apply a soft prior to
    # data with no basis for any prior). Short-circuits before quality-
    # confidence branches and before user override (user can still request
    # HARD/SOFT/SKIP explicitly; UNKNOWN only bypasses the auto branch).
    from pmriprep.utils.dsc_preload import PreloadClass as _PreloadClass

    if (
        (not user_override or user_override == "auto")
        and preload_classification == _PreloadClass.UNKNOWN.value
    ):
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=(
                "K2-primary preload classification unavailable "
                "(UNKNOWN) — constraint skipped"
            ),
        )

    # User override
    if user_override and user_override != "auto":
        try:
            mode = ConstraintMode(user_override)
        except ValueError:
            mode = ConstraintMode.SKIP
        if mode == ConstraintMode.HARD:
            uncertainty = 0.0
            logger.warning(
                "User explicitly requested HARD vp constraint. DSC CBV and "
                "DCE vp measure different vascular compartments; HARD "
                "equality is an unsupported physics claim and biases "
                "absolute PK estimates. Consider 'soft' or 'skip' unless "
                "fixing vp is required for parameter identifiability."
            )
            reason = (
                f"User override: {user_override} "
                "(WARNING: HARD equality is unsupported physics; "
                "biases absolute PK estimates)"
            )
        elif mode == ConstraintMode.SOFT:
            uncertainty = SIGMA_SOFT_DEFAULT
            reason = f"User override: {user_override}"
        else:
            uncertainty = 0.0
            reason = f"User override: {user_override}"
        return ConstraintConfig(
            mode=mode,
            vp_uncertainty=uncertainty,
            reason=reason,
        )

    confidence = dsc_quality.confidence

    if confidence == "high":
        # High quality no longer emits HARD; use tight SOFT instead.
        return ConstraintConfig(
            mode=ConstraintMode.SOFT,
            vp_uncertainty=0.20,
            reason="high DSC quality (auto → tight SOFT; HARD deprecated)",
        )
    elif confidence == "moderate":
        uncertainty = SIGMA_SOFT_DEFAULT
        reason_suffix = ""
        # Reuse the classification cached during quality assessment when
        # available; fall back to recomputing only if the quality object
        # predates this field (legacy callers building it by hand).
        if dsc_quality.preload_k2_classification is not None:
            is_preloaded, well_suppressed, high_correction = (
                dsc_quality.preload_k2_classification
            )
        else:
            is_preloaded, well_suppressed, high_correction = (
                _classify_preload_k2(preload_classification, k2_metrics)
            )
        if is_preloaded and well_suppressed:
            uncertainty = SIGMA_SOFT_PRELOAD_SUPPRESSED
            reason_suffix = " + preload suppressed"
        elif not is_preloaded and high_correction:
            uncertainty = SIGMA_SOFT_HIGH_K2
            reason_suffix = " + high K2"

        return ConstraintConfig(
            mode=ConstraintMode.SOFT,
            vp_uncertainty=uncertainty,
            reason=f"moderate DSC{reason_suffix}",
        )
    else:
        # Keep detail for skip — it's the diagnostic breadcrumb
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason="; ".join(dsc_quality.reasons) if dsc_quality.reasons else "low DSC quality",
        )


# =============================================================================
# Orchestrator
# =============================================================================


def prepare_vp_constraint(
    ncbv_file: Optional[Union[str, Path]],
    aif_qc_file: Union[str, Path],
    brain_mask_file: Union[str, Path],
    mode_override: Optional[str] = None,
    hematocrit: float = HEMATOCRIT_ARTERIAL,
    preload_classification: Optional[str] = None,
    k2_metrics: Optional[dict[str, Any]] = None,
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> VpConstraintResult:
    """Prepare vp constraint map from DSC NAWM-normalized CBV for DCE PK fitting.

    Orchestrates:
    1. Load nCBV map and AIF QC JSON
    2. Assess DSC CBV quality (on the nCBV distribution)
    3. Convert nCBV to vp prior via ``ncbv_to_vp_prior`` (vp = 0.02 × nCBV)
    4. Determine constraint mode
    5. Magnitude gate: downgrade to SKIP if derived vp is outside physiological range

    Args:
        ncbv_file: Path to DSC-derived NAWM-normalized CBV (nCBV) NIfTI.
            If None or missing (no NAWM reference available), returns a
            SKIP result with reason "nCBV unavailable".
        aif_qc_file: Path to DSC AIF QC JSON.
        brain_mask_file: Path to brain mask NIfTI.
        mode_override: User-specified constraint mode override.
        hematocrit: Blood hematocrit (retained for provenance; not used by
            the nCBV→vp conversion which anchors on NAWM_VP_REFERENCE).
        preload_classification: DSC preload classification string from QC
            (e.g., 'half_dose', 'significant', 'severe', 'none').
        k2_metrics: K2 leakage correction metrics dict from dsc_preload,
            containing 'k2_median', 'percent_leaky_voxels', etc.

    Returns:
        VpConstraintResult with vp_map, config, quality, and qc_report.
    """
    aif_qc_path = Path(aif_qc_file)
    if aif_qc_path.exists():
        with open(aif_qc_path) as f:
            aif_qc_data = json.load(f)
    else:
        aif_qc_data = {}

    return prepare_vp_constraint_from_data(
        ncbv_file=ncbv_file,
        aif_qc_data=aif_qc_data,
        brain_mask_file=brain_mask_file,
        mode_override=mode_override,
        hematocrit=hematocrit,
        preload_classification=preload_classification,
        k2_metrics=k2_metrics,
        dose=dose,
        agent_standard_dose=agent_standard_dose,
    )


def prepare_vp_constraint_from_data(
    ncbv_file: Optional[Union[str, Path]],
    aif_qc_data: dict[str, Any],
    brain_mask_file: Union[str, Path],
    mode_override: Optional[str] = None,
    hematocrit: float = HEMATOCRIT_ARTERIAL,
    preload_classification: Optional[str] = None,
    k2_metrics: Optional[dict[str, Any]] = None,
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> VpConstraintResult:
    """Prepare vp constraint map from DSC nCBV for DCE PK modeling (data form).

    Same as :func:`prepare_vp_constraint` but accepts AIF QC data as a dict
    directly instead of reading from a JSON file.

    If ``ncbv_file`` is None or does not exist, returns a SKIP result with
    ``qc_report["skip_reason"] = "no_nawm_mask"``. This is the expected
    outcome when the upstream DSC workflow could not produce an nCBV map
    (no NAWM mask from tumor segmentation and none user-supplied).

    Args:
        ncbv_file: Path to DSC nCBV NIfTI, or None / missing → clean SKIP.
        aif_qc_data: DSC AIF QC metrics dictionary (from workflow).
        brain_mask_file: Path to brain mask NIfTI.
        mode_override: User-specified constraint mode override.
        hematocrit: Blood hematocrit (retained for provenance).
        preload_classification: DSC preload classification string from QC.
        k2_metrics: K2 leakage correction metrics dict from dsc_preload.

    Returns:
        VpConstraintResult with vp_map (None for SKIP), config, quality,
        and qc_report with unit keys and (on SKIP) a ``skip_reason`` field.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    result = VpConstraintResult()

    # Missing-NAWM handling: if no nCBV file available, cleanly SKIP.
    if ncbv_file is None or not Path(str(ncbv_file)).is_file():
        logger.info(
            "vp constraint skipped: nCBV unavailable (no NAWM reference "
            "mask). DCE vp will be fit unconstrained."
        )
        result.config = ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason="nCBV unavailable (no NAWM reference)",
        )
        result.vp_map = None
        result.qc_report = {
            "mode": ConstraintMode.SKIP.value,
            "reason": result.config.reason,
            "skip_reason": "no_nawm_mask",
            "hematocrit": hematocrit,
        }
        return result

    # Load brain mask
    mask_img = nib.load(str(brain_mask_file))
    mask_data: npt.NDArray[np.floating] = mask_img.get_fdata()  # type: ignore[attr-defined]

    # Load nCBV map (dimensionless, NAWM-normalized)
    ncbv_img = nib.load(str(ncbv_file))
    ncbv_data: npt.NDArray[np.floating] = ncbv_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]

    # Step 1: Assess quality. `assess_dsc_cbv_quality` consumes the CBV
    # distribution; under the nCBV path it sees a NAWM-anchored ratio
    # whose median should sit in CBV_MEDIAN_RANGE=(0.3, 3.0).
    quality = assess_dsc_cbv_quality(
        aif_qc_data=aif_qc_data if aif_qc_data else {},
        cbv_data=ncbv_data,
        brain_mask=mask_data,
        preload_classification=preload_classification,
        k2_metrics=k2_metrics,
        dose=dose,
        agent_standard_dose=agent_standard_dose,
    )
    result.quality = quality

    # Step 2: Determine constraint mode
    config = determine_constraint_mode(
        quality,
        user_override=mode_override,
        preload_classification=preload_classification,
        k2_metrics=k2_metrics,
    )
    result.config = config

    # Step 3: Convert nCBV to vp prior (even for skip mode, for QC reporting).
    # NAWM-anchored: vp = NAWM_VP_REFERENCE × nCBV. Avoids the absolute-CBV
    # calibration ambiguity that `cbv_to_vp` has with relative AUC CBV.
    vp_map = ncbv_to_vp_prior(ncbv_data)

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
        "detection_level": quality.detection_level,
        "n_aif_voxels": quality.n_aif_voxels,
        "gamma_peak_ratio": quality.gamma_peak_ratio,
        "hematocrit": hematocrit,
        # Unit annotations so downstream consumers don't conflate scales.
        "cbv_median_units": "ncbv",
        "vp_median_units": "fraction",
    }

    if len(vp_valid) > 0:
        vp_median = float(np.median(vp_valid))
        qc_report["vp_median"] = vp_median
        qc_report["vp_p5"] = float(np.percentile(vp_valid, 5))
        qc_report["vp_p95"] = float(np.percentile(vp_valid, 95))
        qc_report["vp_nan_fraction"] = float(
            np.sum(~np.isfinite(vp_in_brain)) / len(vp_in_brain)
        )

        # Step 4: Magnitude gate (undershoot + overshoot).
        # Physiological plausibility: WM vp ≈ 0.01–0.02, GM ≈ 0.02–0.04,
        # tumor up to ~0.08. Values outside [VP_MEDIAN_MIN, VP_MEDIAN_MAX]
        # indicate a gross calibration failure — downgrade to SKIP rather
        # than feeding biased priors into DCE fitting. The existing scale-
        # mismatch guard in dce_compute only catches overshoot, so this
        # gate adds the undershoot protection.
        if not (VP_MEDIAN_MIN <= vp_median <= VP_MEDIAN_MAX):
            logger.warning(
                "Derived vp median %.4f outside physiological range "
                "[%.3f, %.2f]; downgrading constraint to SKIP.",
                vp_median, VP_MEDIAN_MIN, VP_MEDIAN_MAX,
            )
            config = ConstraintConfig(
                mode=ConstraintMode.SKIP,
                vp_uncertainty=0.0,
                reason=(
                    f"vp_median {vp_median:.4f} outside "
                    f"[{VP_MEDIAN_MIN}, {VP_MEDIAN_MAX}]"
                ),
            )
            # Critical: reassign into result so caller sees consistent
            # (mode=SKIP, vp_map=None). Without this, the caller sees the
            # pre-gate mode while vp_map is None → dispatch breaks.
            result.config = config
            qc_report["mode"] = config.mode.value
            qc_report["reason"] = config.reason
            qc_report["magnitude_gate_triggered"] = True
            result.vp_map = None
            result.qc_report = qc_report
            return result

    if quality.cbv_median is not None:
        qc_report["cbv_median"] = quality.cbv_median
    if quality.cbv_negative_fraction is not None:
        qc_report["cbv_negative_fraction"] = quality.cbv_negative_fraction

    # Surface HARD-override physics warning in QC (from determine_constraint_mode)
    if (
        config.mode == ConstraintMode.HARD
        and config.reason.startswith("User override:")
    ):
        qc_report["user_override_warning"] = (
            "User explicitly requested HARD constraint. DSC CBV and DCE vp "
            "measure different vascular compartments; HARD equality is an "
            "unsupported physics claim and biases absolute PK estimates."
        )

    result.qc_report = qc_report

    # Only provide vp_map if constraint will be used
    if config.mode != ConstraintMode.SKIP:
        result.vp_map = vp_map
    else:
        result.vp_map = None

    return result
