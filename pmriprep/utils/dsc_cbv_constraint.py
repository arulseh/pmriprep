"""DSC CBV to vp constraint utilities for DCE pharmacokinetic modeling.

Implements quality-gated conversion of DSC-derived CBV maps to vp constraints
for use in DCE PK model fitting. When DSC data is available and passes quality
checks, the CBV-derived vp map can constrain the DCE fitting, reducing the
number of free parameters and improving stability.

The constraint pipeline:
    1. Assess DSC CBV quality (CNR coverage, AIF metrics, CBV ranges)
    2. Scale NAWM-normalized CBV to a vp prior
    3. Determine constraint mode: hard (fix vp), soft (Bayesian prior), skip

"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

from pmriprep.utils.dce_constants import (
    NAWM_VP_REFERENCE,
    VP_MAX_PHYSIOLOGICAL,
    VP_MEDIAN_MIN,
    VP_MEDIAN_MAX,
    SIGMA_SOFT_DEFAULT,
    SIGMA_SOFT_K2_SUPPRESSED,
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
    POSITIVE_TAIL_PRESENT_FRAC,
    POSITIVE_TAIL_SEVERE_FRAC,
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
#: accommodates hypervascular HGG tumor-rim (clinical GBM rCBV up
#: to ~6); this median range is for the whole-brain distribution.
CBV_MEDIAN_RANGE: tuple[float, float] = (0.3, 3.0)

#: Maximum fraction of negative CBV values before quality downgrade
CBV_NEGATIVE_FRACTION_MAX: float = 0.01

# Composite K2 criteria for preload-aware uncertainty adjustment (sigma tier).
# Thresholds colocated in dsc_constants — see K2_SIGMA_* and LEAKY_SIGMA_*
# there for literature references. This tier is coarser than the primary
# classification tier (K2_PRIMARY_*) used by classify_preload_adequacy_from_k2;
# sigma tuning tolerates looser thresholds than discrete class assignment.

# Gamma peak ratio thresholds for constraint quality gating.
# Moderate band [0.90, 1.10] imported from dsc_aif.py (single source of truth).
# Wider low-confidence band captures severe AIF corruption.
# Ref: peak error >12% exceeds the ~10% CBF CoV expected in normal brain.
GAMMA_PEAK_RATIO_LOW_MIN: float = 0.80
GAMMA_PEAK_RATIO_LOW_MAX: float = 1.20


# =============================================================================
# Enums and Dataclasses
# =============================================================================


SKIP_REASON_NO_DSC_RUN = "no_dsc_run"
SKIP_REASON_NO_NAWM_MASK = "no_nawm_mask"
SKIP_REASON_SYNTHETIC_AIF = "synthetic_aif"
NO_DSC_RUN_REASON_STRING = "No DSC data provided"
SYNTHETIC_AIF_REASON_STRING = (
    "AIF is synthetic (cascade exhausted or gamma-R2 fallback); "
    "relative CBV amplitude is not subject-calibrated"
)


class ConstraintMode(Enum):
    """Mode for applying DSC CBV vp constraint to DCE fitting."""

    SOFT = "soft"
    """Use DSC-derived vp as Bayesian prior with uncertainty."""

    SKIP = "skip"
    """Do not use DSC CBV constraint (unconstrained fit)."""


@dataclass
class CBVQualityAssessment:
    """Result of DSC CBV quality assessment.

    Attributes
    ----------
        confidence: Quality tier - 'high', 'moderate', or 'low'.
        use_constraint: Whether the CBV is reliable enough for constraining.
        cnr_coverage: Fraction of brain voxels with CNR > 4.
        aif_peak_height: AIF peak height fraction (0-1).
        aif_baseline_return: AIF baseline return ratio (0-1).
        cbv_median: Median dimensionless nCBV in brain, or None if not assessed.
        cbv_negative_fraction: Fraction of negative nCBV values, or None.
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
    #: Cached (k2_suppressed, high_correction) pair from
    #: :func:`_classify_preload_k2`; populated during quality assessment so
    #: :func:`determine_constraint_mode` can reuse the classification instead
    #: of recomputing it. Avoids a second source of truth for the "high K2
    #: correction" decision.
    preload_k2_classification: Optional[tuple[bool, bool]] = None


@dataclass
class ConstraintConfig:
    """Configuration for vp constraint application.

    Attributes
    ----------
        mode: Constraint mode (hard, soft, skip).
        vp_uncertainty: Relative uncertainty for soft prior (fraction, 0-1).
        reason: Human-readable explanation for the chosen mode.
    """

    mode: ConstraintMode = ConstraintMode.SKIP
    vp_uncertainty: float = 0.2
    reason: str = "No constraint applied"


@dataclass
class VpConstraintResult:
    """Result from prepare_vp_constraint_from_data orchestrator.

    Attributes
    ----------
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

    Maps the producer-shape ``aif_qc.aif_metrics`` block (``psr``,
    ``brr``) to the assessment-schema keys (``aif_peak_height``,
    ``baseline_return_ratio``). Returns data unchanged if expected
    keys are already present.

    Parameters
    ----------
    aif_qc_data : dict[str, Any]
        Raw AIF QC dictionary, possibly carrying the
        producer-shape ``aif_metrics`` nested dict instead of the
        assessment schema keys (``aif_peak_height``,
        ``baseline_return_ratio``, ``cnr_whole_brain``).

    Returns
    -------
    dict[str, Any]
        Dictionary with normalized keys suitable for assess_dsc_cbv_quality.

    Notes
    -----
        This function defines the contract for the producer-shape ``aif_qc``
        dict that the ``--skip-dsc`` resume sidecar must round-trip
        verbatim. If you add a new producer key consumed here, also add it
        to the oracle in ``tests/unit/test_resume_qc_round_trip_consumer_keys.py``
        so the round-trip parity guard catches future schema drift.
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

    aif_metrics = data.get("aif_metrics") or {}

    # --- AIF peak height from PSR ---
    # psr = 100 * (S_tail - S_min) / (S0 - S_min)
    # Higher = better signal recovery. Thresholds: 0.25 (high), 0.15 (moderate)
    if "psr" in aif_metrics and "aif_peak_height" not in data:
        data["aif_peak_height"] = aif_metrics["psr"] / 100.0

    # --- Baseline return ratio from BRR ---
    # brr = 100 * tail_conc / peak_conc
    # Lower = better (more recovery). Thresholds: 0.30 (marginal), 0.40 (fail)
    if "brr" in aif_metrics and "baseline_return_ratio" not in data:
        data["baseline_return_ratio"] = aif_metrics["brr"] / 100.0

    # --- CNR coverage: default to moderate if not available ---
    _set_cnr_default(data)

    return data


def _is_auto(user_override: Optional[str]) -> bool:
    """Return True when the auto-classification branch should be taken.

    Truth table: ``None`` → True; ``""`` → True; ``"auto"`` → True;
    anything else (e.g. ``"soft"``, ``"skip"``) → False.
    """
    return not user_override or user_override == "auto"


def _downgrade_confidence(current: str, target: str) -> str:
    """Downgrade confidence level but never upgrade.

    Parameters
    ----------
    current : str
        Current confidence level.
    target : str
        Target confidence level.

    Returns
    -------
    str
        The lower of the two confidence levels.
    """
    levels = {"high": 2, "moderate": 1, "low": 0}
    if levels.get(target, 0) < levels.get(current, 0):
        return target
    return current


def _classify_preload_k2(
    k2_metrics: Optional[dict[str, Any]],
) -> tuple[bool, bool]:
    """Classify K2-correction status for quality and uncertainty logic.

    Keyed entirely off the K2 residual distribution; the preload label is
    not consulted (low K2 means the correction is reliable regardless of
    whether preload was administered).

    Parameters
    ----------
    k2_metrics : Optional[dict[str, Any]]
        K2 leakage correction metrics from dsc_preload.

    Returns
    -------
    tuple[bool, bool]
        Tuple of (k2_suppressed, high_correction).
    """
    if not k2_metrics:
        return False, False

    k2_med = float(k2_metrics.get("k2_median", 0.0))
    pct_leaky = float(k2_metrics.get("percent_leaky_voxels", 0.0))

    k2_suppressed = k2_med < K2_SIGMA_SUPPRESSED and pct_leaky < LEAKY_SIGMA_LOW
    high_correction = pct_leaky > LEAKY_SIGMA_HIGH or k2_med > K2_SIGMA_CORRECTION

    return k2_suppressed, high_correction


def _bolus_gd_fraction(
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> float:
    """Bolus Gd dose as a fraction of the Gd-DOTA reference (0.1 mmol Gd/kg).

    Peak first-pass ΔR2* and CNR scale linearly with the product
    ``dose × agent_standard_dose`` (Rose criterion). CNR coverage
    thresholds calibrated at full
    standard dose must be scaled by this fraction to remain valid on
    reduced-dose protocols or lower-dose agents.

    Parameters
    ----------
    dose : float, optional
        Bolus dose multiplier (1.0 = full standard, 0.5 = half).
        Must be positive.
    agent_standard_dose : float, optional
        Agent's standard clinical dose in mmol Gd/kg.
        Must be positive.

    Returns
    -------
    float
        Effective Gd fraction relative to 0.1 mmol/kg reference.

    Raises
    ------
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
    - nCBV value ranges: whether whole-brain median nCBV is plausible

    CNR coverage thresholds scale with the administered Gd dose
    (``dose × agent_standard_dose`` relative to 0.1 mmol/kg reference).
    Half-dose protocols and lower-dose agents (gadopiclenol 0.05,
    gadoquatrane 0.04) produce inherently lower CNR; scaling prevents
    spurious SKIP decisions on otherwise-adequate acquisitions.

    Parameters
    ----------
    aif_qc_data : dict[str, Any]
        Dictionary from DSC AIF QC JSON containing metrics
        like cnr_whole_brain, aif_peak_height, baseline_return_ratio.
    cbv_data : npt.NDArray[np.floating], optional
        Optional 3D dimensionless nCBV map for value range validation.
    brain_mask : npt.NDArray[np.floating], optional
        Optional 3D brain mask for nCBV statistics.
    preload_classification : str, optional
        DSC preload classification string.
    k2_metrics : dict[str, Any], optional
        K2 leakage correction metrics from dsc_preload.
    dose : float, optional
        DSC bolus dose multiplier (1.0 = full standard).
    agent_standard_dose : float, optional
        Agent's standard dose in mmol Gd/kg
        (default 0.1 for Gd-DOTA class).

    Returns
    -------
    CBVQualityAssessment
        CBVQualityAssessment with confidence tier and detailed metrics.
    """
    aif_qc_data = _normalize_aif_qc_keys(aif_qc_data)

    result = CBVQualityAssessment()
    confidence = "high"

    # Scale CNR coverage thresholds by the effective bolus Gd fraction.
    # Default (full-dose Gd-DOTA) gives eff=1.0 → unscaled thresholds.
    eff_dose = _bolus_gd_fraction(dose, agent_standard_dose)
    cnr_skip = CNR_COVERAGE_SKIP * eff_dose
    cnr_moderate = CNR_COVERAGE_MODERATE * eff_dose
    cnr_high = CNR_COVERAGE_HIGH * eff_dose

    # AIF detection tier is read here (not from result.detection_level,
    # which isn't set yet) to allow the AIF-tier fast-pass below to bypass
    # the CNR-coverage SKIP for high-quality atlas-based AIF detections.
    # detection_method may be a "+"-joined union (e.g. "atlas_strict+
    # atlas_relaxed") under v3 accumulate-to-quorum semantics; treat the
    # union as high-quality only when EVERY contributing tier is in the
    # bypass set (otherwise a looser tier dilutes the bypass guarantee).
    from pmriprep.utils.dsc_constants import split_detection_level

    _detection_level_for_cnr = str(aif_qc_data.get("detection_method", ""))
    _contributing_tiers = set(split_detection_level(_detection_level_for_cnr))
    _AIF_BYPASS_TIERS = {"atlas_strict", "atlas_relaxed"}
    aif_high_quality = bool(_contributing_tiers) and _contributing_tiers.issubset(
        _AIF_BYPASS_TIERS
    )

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
    # Under v3 accumulate-to-quorum semantics, detection_method may be a
    # "+"-joined union (e.g. "atlas_strict+atlas_relaxed"). A single
    # contributing tier in the LOW or MODERATE set is enough to penalise
    # the union — looser-tier voxels dilute the AIF shape regardless of
    # what stricter tiers contributed.
    _AIF_LEVELS_LOW = {"synthetic"}
    _AIF_LEVELS_MODERATE = {"brain_permissive", "brain_relaxed", "atlas_permissive"}

    detection_level = str(aif_qc_data.get("detection_method", ""))
    result.detection_level = detection_level or None
    _contributing = set(split_detection_level(detection_level))

    if _contributing & _AIF_LEVELS_LOW:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"AIF detection level '{detection_level}' -- CBV unreliable"
        )
    elif _contributing & _AIF_LEVELS_MODERATE:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"AIF detection level '{detection_level}' -- CBV quality uncertain"
        )

    # --- AIF Candidate Count ---
    # Single-voxel AIF has no noise averaging; AIF SNR < 20 degrades
    # SVD deconvolution.  Default 3 when key is
    # absent for backward compatibility (no penalty).
    n_candidates = int(aif_qc_data.get("n_aif_voxels_selected", 3))
    result.n_aif_voxels = n_candidates

    if n_candidates < 2:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"Only {n_candidates} AIF voxel(s) -- no noise averaging"
        )
    elif n_candidates < 3:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"Only {n_candidates} AIF voxels -- limited averaging"
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
                        f"nCBV median {cbv_median:.2f} outside "
                        f"[{CBV_MEDIAN_RANGE[0]}, {CBV_MEDIAN_RANGE[1]}]"
                    )

    result.preload_status = preload_classification
    classification = _classify_preload_k2(k2_metrics)
    result.preload_k2_classification = classification
    _k2_suppressed, high_correction = classification
    if high_correction:
        k2_med = float((k2_metrics or {}).get("k2_median", 0.0))
        pct_leaky = float((k2_metrics or {}).get("percent_leaky_voxels", 0.0))
        result.reasons.append(
            f"Substantial leakage correction "
            f"(K2 median={k2_med:.4f}, leaky={pct_leaky:.1f}%): "
            "CBV relies on large K2 correction"
        )

    result.confidence = confidence
    result.use_constraint = confidence != "low"

    if not result.reasons:
        result.reasons.append("All quality checks passed")

    return result


# =============================================================================
# nCBV to vp Prior
# =============================================================================


def ncbv_to_vp_prior(
    ncbv_data: npt.NDArray[np.floating],
    nawm_vp_reference: float = NAWM_VP_REFERENCE,
) -> npt.NDArray[np.floating]:
    """Convert NAWM-normalized CBV (nCBV) to a vp prior map.

    Anchors NAWM vp at ``NAWM_VP_REFERENCE`` and scales linearly
    with the per-voxel
    nCBV ratio. Because nCBV cancels AIF partial-volume, recirculation,
    and first-pass-truncation biases (all affect numerator and denominator
    equally), this is the scientifically defensible conversion from a
    DSC-derived relative volume measure to a DCE vp prior, avoiding the
    unit ambiguities that preclude direct relative-CBV → vp conversion.

    Hct closure constraint: this anchor uses the same large-vessel Hct =
    0.42 convention as the AIF-path ``/(1-Hct)`` blood→plasma conversion
    in ``dce_aif.py``. Mixing capillary-Hct (for the tissue vp anchor) and
    large-vessel-Hct (for AIF conversion) across a single quantitative
    chain violates plasma-fraction closure.

    Parameters
    ----------
    ncbv_data : npt.NDArray[np.floating]
        3D nCBV map (CBV / median(CBV_NAWM)), dimensionless.
    nawm_vp_reference : float, optional
        Assumed NAWM vp. Default NAWM_VP_REFERENCE (0.010).

    Returns
    -------
    npt.NDArray[np.floating]
        3D vp prior map (dimensionless), clipped to [0, VP_MAX_PHYSIOLOGICAL].

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


def _sigma_from_k2(k2_suppressed: bool, high_correction: bool) -> tuple[float, str]:
    """Select the SOFT vp-prior sigma from K2 leakage-correction adequacy.

    K2 adequacy is a *systematic* (bias) uncertainty on the CBV anchor, distinct
    from DSC confidence (a *random* uncertainty that governs SOFT-vs-SKIP). A
    well-suppressed K2 means the anchor is trustworthy (tight prior); a high
    residual correction means the CBV scale is biased (loose prior). When both
    flags are set (mid-range K2), ``k2_suppressed`` wins — the adequate tier.

    Parameters
    ----------
    k2_suppressed : bool
        K2 residual is low (adequate leakage correction).
    high_correction : bool
        K2 residual is high (inadequate leakage correction).

    Returns
    -------
    tuple[float, str]
        ``(sigma, adequacy_label)`` where ``adequacy_label`` is one of
        ``'adequate'``, ``'partial'``, ``'inadequate'``.
    """
    if k2_suppressed:
        return SIGMA_SOFT_K2_SUPPRESSED, "adequate"
    if high_correction:
        return SIGMA_SOFT_HIGH_K2, "inadequate"
    return SIGMA_SOFT_DEFAULT, "partial"


def determine_constraint_mode(
    dsc_quality: CBVQualityAssessment,
    user_override: Optional[str] = None,
    preload_classification: Optional[str] = None,
    k2_metrics: Optional[dict[str, Any]] = None,
) -> ConstraintConfig:
    """Determine constraint mode from DSC quality assessment.

    Two orthogonal axes decide the outcome:

    - **MODE (SOFT vs SKIP)** is governed by DSC *quality/noise*: the
      ``confidence`` tier from :func:`assess_dsc_cbv_quality` (AIF tier, CNR,
      AIF peak/baseline, gamma-peak ratio) plus the hard SKIP triggers
      (synthetic AIF, ``UNKNOWN`` K2, user ``skip``, and a *severe* BSW-blind
      positive-tail overestimate). low confidence → SKIP; high/moderate → SOFT.
    - **σ (prior width, SOFT only)** is governed by K2 leakage-correction
      *adequacy* (a systematic bias on the CBV anchor): adequate → 0.40,
      partial → 0.50, inadequate → 0.60 (see :func:`_sigma_from_k2`). A
      *present* positive-tail overestimate overrides the K2 label to 0.60,
      because such a voxel has K2 pinned at 0 (looks "suppressed") yet its CBV
      is over-estimated, so the anchor must be loosened, not tightened.

    Decision order: synthetic AIF → SKIP; UNKNOWN K2 → SKIP; user override;
    severe positive-tail → SKIP; then confidence → {SOFT with adequacy-derived
    σ, or SKIP}.

    Parameters
    ----------
    dsc_quality : CBVQualityAssessment
        Quality assessment from assess_dsc_cbv_quality.
    user_override : str, optional
        User-specified mode override ('auto', 'skip'). If 'auto' or None,
        auto-determined from quality.
    preload_classification : str, optional
        DSC preload classification string from QC
        (e.g., 'half_dose', 'significant', 'severe', 'none').
    k2_metrics : dict[str, Any], optional
        K2 leakage correction metrics dict from dsc_preload,
        containing 'k2_median', 'percent_leaky_voxels', etc.

    Returns
    -------
    ConstraintConfig
        ConstraintConfig with mode, uncertainty, and reason.
    """
    # Synthetic-AIF short-circuit: overrides every user selection. If the
    # AIF cascade exhausted and the pipeline fell back to a synthetic
    # gamma variate, the CBV magnitudes are not comparable to calibrated
    # data, so SOFT would lock vp to an arbitrary scale; only SKIP is
    # defensible.
    if dsc_quality.detection_level == "synthetic":
        if (
            user_override is not None
            and user_override.lower() not in ("auto", "skip")
        ):
            logger.warning(
                "User requested constraint mode '%s' but AIF is synthetic; "
                "overriding to SKIP. Absolute CBV magnitudes are not "
                "calibrated against this subject's true arterial input.",
                user_override,
            )
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=SYNTHETIC_AIF_REASON_STRING,
        )

    # UNKNOWN preload → SKIP. More conservative than defaulting to NONE,
    # which would silently apply a soft prior to data with no basis for
    # any prior. Short-circuits before quality-confidence branches and
    # before user override (user can still request SOFT/SKIP explicitly;
    # UNKNOWN only bypasses the auto branch).
    from pmriprep.utils.dsc_preload import K2Adequacy as _K2Adequacy

    if (
        _is_auto(user_override)
        and preload_classification == _K2Adequacy.UNKNOWN.value
    ):
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=(
                "K2-primary preload classification unavailable "
                "(UNKNOWN) -- constraint skipped"
            ),
        )

    # User override
    if not _is_auto(user_override):
        # Only 'skip' (force off) is a valid non-auto override; the auto gate
        # owns when the prior is applied, so there is no force-on. Anything
        # else falls back to SKIP.
        if user_override == "skip":
            return ConstraintConfig(
                mode=ConstraintMode.SKIP,
                vp_uncertainty=0.0,
                reason="User override: skip",
            )
        logger.warning(
            "Unrecognized constraint user_override %r — falling back to "
            "SKIP. Valid values: 'auto', 'skip'.",
            user_override,
        )
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=(
                f"Unrecognized user_override {user_override!r} -> SKIP "
                "(valid: 'auto', 'skip')"
            ),
        )

    # Positive-tail (BSW-blind overestimate) fraction, plumbed from the DSC
    # preload stage via k2_metrics. A T2*-dominant positive ΔR2* tail makes K2
    # pin at 0 (looks "suppressed") while the full-integral CBV over-estimates.
    positive_tail_fraction = float(
        (k2_metrics or {}).get("cbv_positive_tail_fraction", 0.0)
    )

    # Severe positive-tail → SKIP, even at high DSC quality: widening the prior
    # cannot recenter a large systematic overestimate (only removing/skipping
    # the anchor moves vp off the over-estimated value).
    if positive_tail_fraction > POSITIVE_TAIL_SEVERE_FRAC:
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=(
                "BSW-blind positive-tail CBV overestimate "
                f"(fraction={positive_tail_fraction:.2f}) -- constraint skipped"
            ),
        )

    confidence = dsc_quality.confidence

    if confidence in ("high", "moderate"):
        # MODE is set (SOFT); σ is governed by K2 adequacy, not DSC confidence
        # (orthogonal: confidence = random error, K2 adequacy = systematic
        # bias). Reuse the classification cached during quality assessment when
        # available; fall back to recomputing only if the quality object
        # predates this field (legacy callers building it by hand).
        if dsc_quality.preload_k2_classification is not None:
            k2_suppressed, high_correction = (
                dsc_quality.preload_k2_classification
            )
        else:
            k2_suppressed, high_correction = _classify_preload_k2(k2_metrics)

        if positive_tail_fraction > POSITIVE_TAIL_PRESENT_FRAC:
            # Override the K2 label: a present positive tail makes K2 look
            # suppressed (→ tightest σ) on a CBV that is over-estimated, so
            # loosen the anchor instead.
            uncertainty = SIGMA_SOFT_HIGH_K2
            adequacy = "positive-tail overestimate (K2-blind)"
        else:
            uncertainty, adequacy = _sigma_from_k2(k2_suppressed, high_correction)

        return ConstraintConfig(
            mode=ConstraintMode.SOFT,
            vp_uncertainty=uncertainty,
            reason=f"{confidence} DSC quality; K2 {adequacy} (sigma={uncertainty:.2f})",
        )
    else:
        # low → SKIP. Keep detail — it's the diagnostic breadcrumb.
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason="; ".join(dsc_quality.reasons) if dsc_quality.reasons else "low DSC quality",
        )


# =============================================================================
# Orchestrator
# =============================================================================


def prepare_vp_constraint_from_data(
    ncbv_file: Optional[Union[str, Path]],
    aif_qc_data: dict[str, Any],
    brain_mask_file: Union[str, Path],
    mode_override: Optional[str] = None,
    preload_classification: Optional[str] = None,
    k2_metrics: Optional[dict[str, Any]] = None,
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> VpConstraintResult:
    """Prepare vp constraint map from DSC nCBV for DCE PK modeling.

    Orchestrates:

    1. Quality assessment on the AIF QC dict + nCBV distribution
    2. nCBV→vp prior conversion via ``ncbv_to_vp_prior``
       (vp = NAWM_VP_REFERENCE × nCBV)
    3. Constraint mode determination
    4. Magnitude gate: SKIP downgrade when derived vp is non-physiological

    If ``ncbv_file`` is None or does not exist, returns a SKIP result with
    ``qc_report["skip_reason"] = "no_nawm_mask"``. This is the expected
    outcome when the upstream DSC workflow could not produce an nCBV map
    (no NAWM mask from tumor segmentation and none user-supplied).

    Parameters
    ----------
    ncbv_file : Optional[Union[str, Path]]
        Path to DSC nCBV NIfTI, or None / missing → clean SKIP.
    aif_qc_data : dict[str, Any]
        DSC AIF QC metrics dictionary (from workflow).
    brain_mask_file : Union[str, Path]
        Path to brain mask NIfTI.
    mode_override : str, optional
        User-specified constraint mode override.
    preload_classification : str, optional
        DSC preload classification string from QC.
    k2_metrics : dict[str, Any], optional
        K2 leakage correction metrics dict from dsc_preload.

    Returns
    -------
    VpConstraintResult
        VpConstraintResult with vp_map (None for SKIP), config, quality,
        and qc_report with unit keys and (on SKIP) a ``skip_reason`` field.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    result = VpConstraintResult()

    # Synthetic-AIF handling: when the AIF cascade exhausts every tier
    # and the pipeline falls back to a population-average synthetic
    # gamma variate, the ``rel_cbv_auc`` denominator is not calibrated
    # against this subject's true arterial amplitude. Any vp number
    # derived from it is not comparable across subjects, so the DCE vp
    # constraint is unsafe to anchor on synthetic CBV magnitudes. Emit
    # an explicit SKIP with a dedicated reason *before* the existing
    # nCBV-presence check (which happens to fire implicitly today via
    # the confidence-downgrade + magnitude-gate path, but only
    # accidentally — this makes the intent auditable).
    if aif_qc_data and (
        aif_qc_data.get("detection_method") == "synthetic"
        or aif_qc_data.get("cbv_auc_denominator_is_synthetic")
    ):
        logger.info(
            "vp constraint skipped: AIF is synthetic (cascade exhausted "
            "or gamma-R² fallback). Relative CBV amplitude is not "
            "subject-calibrated; DCE vp will be fit unconstrained."
        )
        result.config = ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=SYNTHETIC_AIF_REASON_STRING,
        )
        result.vp_map = None
        result.qc_report = {
            "mode": ConstraintMode.SKIP.value,
            "reason": result.config.reason,
            "vp_uncertainty": result.config.vp_uncertainty,
            "skip_reason": SKIP_REASON_SYNTHETIC_AIF,
        }
        return result

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
            "vp_uncertainty": result.config.vp_uncertainty,
            "skip_reason": SKIP_REASON_NO_NAWM_MASK,
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
    # calibration ambiguity of relative AUC CBV.
    vp_map = ncbv_to_vp_prior(ncbv_data)

    # Build QC report
    mask_bool = mask_data > 0
    vp_in_brain = vp_map[mask_bool]
    vp_valid = vp_in_brain[np.isfinite(vp_in_brain)]

    qc_report: dict[str, Any] = {
        "mode": config.mode.value,
        "reason": config.reason,
        "vp_uncertainty": config.vp_uncertainty,
        "dsc_cbv_confidence": quality.confidence,
        "dsc_cbv_use_constraint": quality.use_constraint,
        "cnr_coverage": quality.cnr_coverage,
        "aif_peak_height": quality.aif_peak_height,
        "aif_baseline_return": quality.aif_baseline_return,
        "detection_method": quality.detection_level,
        "n_aif_voxels_selected": quality.n_aif_voxels,
        "gamma_peak_ratio": quality.gamma_peak_ratio,
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
        # Physiological plausibility: WM vp ≈ 0.005–0.015,
        # GM ≈ 0.015–0.025, tumor up to ~0.05. Values outside
        # [VP_MEDIAN_MIN, VP_MEDIAN_MAX] indicate a gross calibration
        # failure — downgrade to SKIP rather than feeding biased priors
        # into DCE fitting. The existing scale-mismatch guard in
        # dce_compute only catches overshoot, so this gate adds the
        # undershoot protection.
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
            qc_report["vp_uncertainty"] = config.vp_uncertainty
            qc_report["magnitude_gate_triggered"] = True
            result.vp_map = None
            result.qc_report = qc_report
            return result

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
