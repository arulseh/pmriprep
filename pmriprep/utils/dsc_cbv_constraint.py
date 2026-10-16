"""DSC CBV to vp-prior utilities for DCE pharmacokinetic modeling.

Implements quality-gated conversion of DSC-derived CBV maps to Bayesian vp priors
for use in DCE PK model fitting. When DSC data is available and passes quality
checks, the CBV-derived vp map informs Bayesian fitting without bounding NLLS.

The prior pipeline:
    1. Assess DSC CBV quality (CNR coverage, AIF metrics, CBV ranges)
    2. Scale NAWM-normalized CBV to a vp prior
    3. Determine whether to use the soft Bayesian prior or skip it

"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_constants import (
    NAWM_VP_REFERENCE,
    SIGMA_SOFT_DSC_PRIOR,
    VP_MAX_PHYSIOLOGICAL,
    VP_MEDIAN_MIN,
    VP_MEDIAN_MAX,
)
from pmriprep.utils.dsc_aif import (
    GAMMA_PEAK_RATIO_MIN,
    GAMMA_PEAK_RATIO_MAX,
)
from pmriprep.utils.dsc_constants import (
    GD_DOTA_STANDARD_DOSE,
    QC_MEAN_CNR_MINIMUM,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


#: CNR coverage threshold for the high-confidence tier before dose scaling.
CNR_COVERAGE_HIGH: float = 0.95

#: Minimum CNR coverage fraction for moderate-quality DSC
CNR_COVERAGE_MODERATE: float = 0.84

#: CNR coverage skip threshold before dose scaling and AIF-tier overrides.
CNR_COVERAGE_SKIP: float = 0.70

#: Minimum mean CNR for the vp-prior quality gate.
CNR_MEAN_MINIMUM: float = QC_MEAN_CNR_MINIMUM

#: Minimum recovery proxy (producer PSR / 100) for moderate confidence.
AIF_PEAK_HEIGHT_MIN: float = 0.15

#: Minimum recovery proxy (producer PSR / 100) for high confidence.
AIF_PEAK_HEIGHT_MARGINAL: float = 0.25

#: Maximum baseline return ratio for acceptable quality
AIF_BASELINE_RETURN_MAX: float = 0.40

#: Marginal baseline return ratio
AIF_BASELINE_RETURN_MARGINAL: float = 0.30

#: Policy interval for the median of positive brain nCBV values (dimensionless).
CBV_MEDIAN_RANGE: tuple[float, float] = (0.3, 3.0)

#: Maximum fraction of negative CBV values before quality downgrade
CBV_NEGATIVE_FRACTION_MAX: float = 0.01

# Gamma peak ratio thresholds for constraint quality gating.
# Moderate band [0.90, 1.10] imported from dsc_aif.py (single source of truth).
# Values outside the wider interval receive low confidence.
GAMMA_PEAK_RATIO_LOW_MIN: float = 0.80
GAMMA_PEAK_RATIO_LOW_MAX: float = 1.20


# =============================================================================
# Enums and Dataclasses
# =============================================================================


SKIP_REASON_NO_DSC_RUN = "no_dsc_run"
SKIP_REASON_NO_NAWM_MASK = "no_nawm_mask"
SKIP_REASON_SYNTHETIC_AIF = "synthetic_aif"
SKIP_REASON_REGISTRATION_QC = "cross_modal_registration_qc"
SKIP_REASON_NLLS_UNSUPPORTED = "nlls_constraint_unsupported"
NO_DSC_RUN_REASON_STRING = "No DSC data were available for a vp prior."
SYNTHETIC_AIF_REASON_STRING = (
    "The DSC vp prior was skipped by the AIF-source quality gate."
)


class ConstraintMode(Enum):
    """Mode for applying DSC CBV vp constraint to DCE fitting."""

    SOFT = "soft"
    """Use DSC-derived vp as Bayesian prior with uncertainty."""

    SKIP = "skip"
    """Fit with the DCE model's ordinary priors and bounds."""


@dataclass
class CBVQualityAssessment:
    """Result of DSC CBV quality assessment.

    Attributes
    ----------
        confidence: Quality tier - 'high', 'moderate', or 'low'.
        use_constraint: Whether the assessment permits a DSC-derived vp prior.
        cnr_coverage: Fraction of brain voxels with CNR > 4.
        aif_peak_height: Quality proxy populated from producer signal PSR / 100.
        aif_baseline_return: Tail-to-peak concentration ratio.
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
    detection_level: Optional[str] = None
    n_aif_voxels: Optional[int] = None
    gamma_peak_ratio: Optional[float] = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class ConstraintConfig:
    """Configuration for vp constraint application.

    Attributes
    ----------
        mode: Constraint mode (soft Bayesian prior or skip).
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

    # Populate the quality proxy from signal recovery.
    # psr = 100 * (S_tail - S_min) / (S0 - S_min)
    # Thresholds are 0.25 for high and 0.15 for moderate confidence.
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


def _bolus_gd_fraction(
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> float:
    """Bolus Gd dose as a fraction of the Gd-DOTA reference (0.1 mmol Gd/kg).

    Returns ``dose * agent_standard_dose / 0.1``. The vp-prior quality
    policy multiplies its CNR coverage thresholds by this fraction.

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
    ValueError
        If ``dose`` or ``agent_standard_dose`` is non-positive.
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
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> CBVQualityAssessment:
    """Assess DSC CBV quality for use as a vp constraint.

    Implements a three-tier quality framework (high/moderate/low) based on:
    - CNR coverage: fraction of brain voxels with CNR > 4
    - AIF recovery proxy: producer signal PSR / 100
    - AIF baseline return: how well AIF returns to baseline
    - nCBV value ranges: whether whole-brain median nCBV is plausible

    CNR coverage thresholds scale with the administered Gd dose
    (``dose × agent_standard_dose`` relative to 0.1 mmol/kg reference).
    Lower doses reduce the required CNR coverage under this quality policy.

    Parameters
    ----------
    aif_qc_data : dict[str, Any]
        Dictionary from DSC AIF QC JSON containing metrics
        like cnr_whole_brain, aif_peak_height, baseline_return_ratio.
    cbv_data : npt.NDArray[np.floating], optional
        Optional 3D dimensionless nCBV map for value range validation.
    brain_mask : npt.NDArray[np.floating], optional
        Optional 3D brain mask for nCBV statistics.
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

    # AIF-tier overrides apply only when every contributing tier belongs
    # to the strict/relaxed atlas set. Compound levels are joined with "+".
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

    # Low mean CNR takes precedence over the coverage and AIF-tier rules.
    if mean_cnr is not None and mean_cnr < CNR_MEAN_MINIMUM:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"Mean CNR {mean_cnr:.1f} < {CNR_MEAN_MINIMUM} (inadequate contrast)"
        )
    elif cnr_coverage < cnr_skip and not aif_high_quality:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"CNR coverage {cnr_coverage:.1%} < {cnr_skip:.1%} "
            f"(dose-scaled, eff={eff_dose:.2f}); AIF "
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
    # Synthetic AIFs receive low confidence under the prior policy. For
    # compound detection levels, the lowest qualifying tier sets confidence.
    _AIF_LEVELS_LOW = {"synthetic"}
    _AIF_LEVELS_MODERATE = {"brain_permissive", "brain_relaxed", "atlas_permissive"}

    detection_level = str(aif_qc_data.get("detection_method", ""))
    result.detection_level = detection_level or None
    _contributing = set(split_detection_level(detection_level))

    if _contributing & _AIF_LEVELS_LOW:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"AIF detection level '{detection_level}': low prior confidence"
        )
    elif _contributing & _AIF_LEVELS_MODERATE:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"AIF detection level '{detection_level}': moderate prior confidence"
        )

    # --- AIF Candidate Count ---
    # Fewer than three selected candidates lower confidence. An absent
    # count defaults to three.
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

    # --- AIF recovery proxy ---
    aif_peak = float(aif_qc_data.get("aif_peak_height", 0.0))
    result.aif_peak_height = aif_peak

    if aif_peak < AIF_PEAK_HEIGHT_MIN:
        confidence = _downgrade_confidence(confidence, "low")
        result.reasons.append(
            f"AIF signal-recovery fraction {aif_peak:.2f} < {AIF_PEAK_HEIGHT_MIN}"
        )
    elif aif_peak < AIF_PEAK_HEIGHT_MARGINAL:
        confidence = _downgrade_confidence(confidence, "moderate")
        result.reasons.append(
            f"AIF signal-recovery fraction {aif_peak:.2f} < {AIF_PEAK_HEIGHT_MARGINAL}"
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
    # Compare gamma-fit and measured peaks against the quality-policy intervals.
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

    Scales each voxel's nCBV by the assumed NAWM plasma volume fraction:
    ``vp = nawm_vp_reference * ncbv_data``. Normalization cancels common
    multiplicative scaling of voxel and reference CBV.

    Parameters
    ----------
    ncbv_data : npt.NDArray[np.floating]
        3D nCBV map (CBV / median(CBV_NAWM)), dimensionless.
    nawm_vp_reference : float, optional
        Assumed NAWM vp. Default NAWM_VP_REFERENCE (0.010).

    Returns
    -------
    npt.NDArray[np.floating]
        3D vp prior map (dimensionless). Negative or nonfinite nCBV inputs
        and scaled vp values outside [0, VP_MAX_PHYSIOLOGICAL] become NaN.

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
    k2_metrics: Optional[dict[str, Any]] = None,
) -> ConstraintConfig:
    """Determine constraint mode from DSC quality assessment.

    Two independent decisions determine the outcome:

    - **MODE (SOFT vs SKIP)** is governed by DSC *quality/noise*: the
      ``confidence`` tier from :func:`assess_dsc_cbv_quality` (AIF tier, CNR,
      AIF peak/baseline, gamma-peak ratio) plus the hard SKIP triggers
      (synthetic AIF, disabled correction, unavailable leakage reference, and
      user ``skip``). Low
      confidence produces SKIP; high/moderate produces SOFT.
    - **σ (prior width, SOFT only)** uses the fixed 0.60 release-policy width.
      This is a fixed relative standard deviation, not a subject-specific
      uncertainty estimate. Signed leakage metrics remain descriptive and do
      not enable, disable, or tune the Bayesian prior.

    Decision order: synthetic AIF → SKIP; correction disabled → SKIP;
    unavailable leakage reference → SKIP; user override; then confidence →
    {SOFT with the fixed release-policy σ, or SKIP}.

    Parameters
    ----------
    dsc_quality : CBVQualityAssessment
        Quality assessment from assess_dsc_cbv_quality.
    user_override : str, optional
        User-specified mode override ('auto', 'skip'). If 'auto' or None,
        auto-determined from quality.
    k2_metrics : dict[str, Any], optional
        K2 leakage-correction metrics from ``dsc_preload``. Signed metrics
        are reported but are not assigned an adequacy tier.

    Returns
    -------
    ConstraintConfig
        ConstraintConfig with mode, uncertainty, and reason.
    """
    # The vp-prior policy excludes synthetic AIFs before other quality gates.
    if dsc_quality.detection_level == "synthetic":
        if (
            user_override is not None
            and user_override.lower() not in ("auto", "skip")
        ):
            logger.warning(
                "User requested constraint mode '%s' but AIF is synthetic; "
                "the vp-prior policy selects SKIP for this AIF source.",
                user_override,
            )
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=SYNTHETIC_AIF_REASON_STRING,
        )

    # An uncorrected CBV map is not an acceptable Bayesian-prior anchor.
    if (k2_metrics or {}).get("leakage_correction_disabled") is True:
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason="DSC leakage correction was disabled, so its CBV map was not used as a vp prior.",
        )

    # The Bayesian prior requires a successfully estimated leakage reference.
    # Missing and false availability values both disable the prior.
    if _is_auto(user_override) and (k2_metrics or {}).get(
        "k2_reference_available"
    ) is not True:
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason="The DSC leakage-reference curve was unavailable, so its CBV map was not used as a vp prior.",
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
                reason="The DSC-informed vp prior was disabled by configuration.",
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
                f"The prior mode {user_override!r} was invalid, so the DSC-informed "
                "vp prior was not applied."
            ),
        )

    confidence = dsc_quality.confidence

    if confidence in ("high", "moderate"):
        # Signed K2 is descriptive. The prior uses a fixed relative standard
        # deviation.
        return ConstraintConfig(
            mode=ConstraintMode.SOFT,
            vp_uncertainty=SIGMA_SOFT_DSC_PRIOR,
            reason=(
                "DSC signal, AIF, leakage-reference, and cross-modal registration "
                "QC supported use of the NAWM-normalized CBV prior."
            ),
        )
    else:
        # low → SKIP. Keep detail — it's the diagnostic breadcrumb.
        return ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=(
                "DSC QC did not support use of the vp prior: "
                + "; ".join(dsc_quality.reasons)
                if dsc_quality.reasons
                else "DSC QC did not support use of the vp prior."
            ),
        )


# =============================================================================
# Orchestrator
# =============================================================================


def prepare_vp_constraint_from_data(
    ncbv_file: Optional[Union[str, Path]],
    aif_qc_data: dict[str, Any],
    brain_mask_file: Union[str, Path],
    mode_override: Optional[str] = None,
    k2_metrics: Optional[dict[str, Any]] = None,
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
    dsc_registration_qc: Optional[dict[str, Any]] = None,
    dce_registration_qc: Optional[dict[str, Any]] = None,
    nawm_exclusion_reason: Optional[str] = None,
) -> VpConstraintResult:
    """Prepare vp constraint map from DSC nCBV for DCE PK modeling.

    Orchestrates:

    1. Quality assessment on the AIF QC dict + nCBV distribution
    2. nCBV→vp prior conversion via ``ncbv_to_vp_prior``
       (vp = NAWM_VP_REFERENCE × nCBV)
    3. Constraint mode determination
    4. Magnitude gate: SKIP when the derived vp median is outside policy bounds

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
    k2_metrics : dict[str, Any], optional
        K2 leakage correction metrics dict from dsc_preload.
    dose : float, optional
        Administered DSC dose as a multiple of the agent's standard dose.
    agent_standard_dose : float, optional
        Standard agent dose in mmol Gd/kg for dose-adjusted AIF quality checks.
    dsc_registration_qc : dict[str, Any], optional
        QC for the DSC-to-T1w transform used by the spatial prior.
    dce_registration_qc : dict[str, Any], optional
        QC for the T1w-to-DCE transform used by the spatial prior.
    nawm_exclusion_reason : str, optional
        Factual upstream reason an automatically generated NAWM mask was
        withheld from quantitative use.

    Returns
    -------
    VpConstraintResult
        VpConstraintResult with vp_map (None for SKIP), config, quality,
        and qc_report with unit keys and (on SKIP) a ``skip_reason`` field.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    result = VpConstraintResult()

    # Apply the AIF-source gate before checking for an nCBV file. The producer
    # sets the denominator flag for synthetic and external AIFs.
    if aif_qc_data and (
        aif_qc_data.get("detection_method") == "synthetic"
        or aif_qc_data.get("cbv_auc_denominator_is_synthetic")
    ):
        logger.info(
            "DSC vp prior skipped by the AIF-source quality gate. "
            "DCE fitting uses the model's ordinary priors and bounds."
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
            "mask). DCE fitting uses the model's ordinary priors and bounds."
        )
        reason = (
            "Automatically generated NAWM was retained for QC but excluded "
            f"from quantitative use: {nawm_exclusion_reason}."
            if nawm_exclusion_reason
            else "No NAWM-normalized DSC CBV map was available for a vp prior."
        )
        result.config = ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=reason,
        )
        result.vp_map = None
        result.qc_report = {
            "mode": ConstraintMode.SKIP.value,
            "reason": result.config.reason,
            "vp_uncertainty": result.config.vp_uncertainty,
            "skip_reason": SKIP_REASON_NO_NAWM_MASK,
        }
        if nawm_exclusion_reason:
            result.qc_report["nawm_exclusion_reason"] = nawm_exclusion_reason
        return result

    registration_statuses = {
        "dsc_to_t1w": str(
            (dsc_registration_qc or {}).get("qc_status", "UNKNOWN")
        ).upper(),
        "t1w_to_dce": str(
            (dce_registration_qc or {}).get("qc_status", "UNKNOWN")
        ).upper(),
    }
    unacceptable_registration = {
        name: status
        for name, status in registration_statuses.items()
        if status not in {"PASS", "WARN"}
    }
    if unacceptable_registration:
        detail = ", ".join(
            f"{name}={status}" for name, status in unacceptable_registration.items()
        )
        logger.warning(
            "vp constraint skipped: quantitative DSC-to-DCE transfer has "
            "unacceptable registration QC (%s). "
            "DCE fitting uses the model's ordinary priors and bounds.",
            detail,
        )
        reason = (
            "Cross-modal registration QC did not support transfer of the DSC-informed "
            f"vp prior ({detail})."
        )
        result.config = ConstraintConfig(
            mode=ConstraintMode.SKIP,
            vp_uncertainty=0.0,
            reason=reason,
        )
        result.vp_map = None
        result.qc_report = {
            "mode": ConstraintMode.SKIP.value,
            "reason": reason,
            "vp_uncertainty": 0.0,
            "skip_reason": SKIP_REASON_REGISTRATION_QC,
            "registration_statuses": registration_statuses,
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
        dose=dose,
        agent_standard_dose=agent_standard_dose,
    )
    result.quality = quality

    # Step 2: Determine constraint mode
    config = determine_constraint_mode(
        quality,
        user_override=mode_override,
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
        "nawm_vp_reference": NAWM_VP_REFERENCE,
        "registration_statuses": registration_statuses,
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
        # Withhold the prior when its median falls outside the configured
        # interval. This gate tests scale plausibility, not calibration truth.
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
                    f"The derived vp prior median ({vp_median:.4f}) was outside the "
                    f"supported range [{VP_MEDIAN_MIN}, {VP_MEDIAN_MAX}]."
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
