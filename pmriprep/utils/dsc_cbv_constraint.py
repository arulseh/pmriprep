"""DSC CBV to vp-prior utilities for DCE pharmacokinetic modeling.

Implements quality-gated conversion of DSC-derived CBV maps to Bayesian vp priors
for use in DCE PK model fitting. When DSC data is available and passes quality
checks, the CBV-derived vp map informs Bayesian fitting without bounding NLLS.

The prior pipeline:
    1. Assess DSC CBV quality (tissue CNR coverage and nCBV support/ranges)
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

#: Minimum mean CNR for the vp-prior quality gate.
CNR_MEAN_MINIMUM: float = QC_MEAN_CNR_MINIMUM

#: Policy interval for the median of positive brain nCBV values (dimensionless).
CBV_MEDIAN_RANGE: tuple[float, float] = (0.3, 3.0)

#: Maximum fraction of negative CBV values before quality downgrade
CBV_NEGATIVE_FRACTION_MAX: float = 0.01

# =============================================================================
# Enums and Dataclasses
# =============================================================================


SKIP_REASON_NO_DSC_RUN = "no_dsc_run"
SKIP_REASON_NO_NAWM_MASK = "no_nawm_mask"
SKIP_REASON_REGISTRATION_QC = "cross_modal_registration_qc"
SKIP_REASON_NLLS_UNSUPPORTED = "nlls_constraint_unsupported"
NO_DSC_RUN_REASON_STRING = "No DSC data were available for a vp prior."


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
        cbv_median: Median dimensionless nCBV in brain, or None if not assessed.
        cbv_negative_fraction: Fraction of negative nCBV values, or None.
        reasons: List of reasons for quality decisions.
    """

    confidence: str = "low"
    use_constraint: bool = False
    cnr_coverage: Optional[float] = None
    mean_cnr: Optional[float] = None
    cbv_median: Optional[float] = None
    cbv_negative_fraction: Optional[float] = None
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


def _finite_metric(value: Any) -> Optional[float]:
    """Return a finite scalar QC metric, or None when unavailable."""
    try:
        metric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return metric if np.isfinite(metric) else None




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
    dsc_qc_data: dict[str, Any],
    cbv_data: Optional[npt.NDArray[np.floating]] = None,
    brain_mask: Optional[npt.NDArray[np.floating]] = None,
    dose: float = 1.0,
    agent_standard_dose: float = GD_DOTA_STANDARD_DOSE,
) -> CBVQualityAssessment:
    """Assess tissue CNR and nCBV support for a Bayesian vp prior.

    Parameters
    ----------
    dsc_qc_data : dict
        Tissue CNR measurements in ``cnr_whole_brain``.
    cbv_data : ndarray, optional
        Dimensionless NAWM-normalized CBV map.
    brain_mask : ndarray, optional
        Brain support on the same grid as nCBV.
    dose : float
        Administered bolus multiplier relative to the agent's standard dose.
    agent_standard_dose : float
        Standard dose in mmol Gd/kg.

    Returns
    -------
    CBVQualityAssessment
        Finite tissue measurements and the resulting project-policy tier.
    """
    result = CBVQualityAssessment()
    confidence = "high"
    eff_dose = _bolus_gd_fraction(dose, agent_standard_dose)
    moderate = CNR_COVERAGE_MODERATE * eff_dose
    high = CNR_COVERAGE_HIGH * eff_dose
    cnr = dsc_qc_data.get("cnr_whole_brain") or {}
    result.cnr_coverage = _finite_metric(cnr.get("fraction_cnr_gt_4")) if isinstance(cnr, dict) else None
    result.mean_cnr = _finite_metric(cnr.get("mean_cnr")) if isinstance(cnr, dict) else None
    coverage, mean = result.cnr_coverage, result.mean_cnr
    if mean is None or coverage is None or not 0 <= coverage <= 1:
        confidence = "low"
        result.reasons.append("Finite tissue CNR measurements unavailable")
    elif mean < CNR_MEAN_MINIMUM:
        confidence = "low"
        result.reasons.append(f"Mean CNR {mean:.1f} < {CNR_MEAN_MINIMUM}")
    elif coverage < moderate:
        confidence = "low"
        result.reasons.append(f"CNR coverage {coverage:.1%} < {moderate:.1%} (dose-scaled)")
    elif coverage < high:
        confidence = "moderate"
        result.reasons.append(f"CNR coverage {coverage:.1%} < {high:.1%} (dose-scaled)")

    if cbv_data is not None:
        values = cbv_data[brain_mask > 0] if brain_mask is not None else cbv_data.ravel()
        values = values[np.isfinite(values)]
        positive = values[values > 0]
        if not positive.size:
            confidence = "low"
            result.reasons.append("No usable positive nCBV support")
        else:
            result.cbv_negative_fraction = float(np.mean(values < 0))
            result.cbv_median = float(np.median(positive))
            if result.cbv_negative_fraction > CBV_NEGATIVE_FRACTION_MAX:
                confidence = "low"
                result.reasons.append(
                    f"nCBV negative fraction {result.cbv_negative_fraction:.3f} > {CBV_NEGATIVE_FRACTION_MAX}"
                )
            if not CBV_MEDIAN_RANGE[0] <= result.cbv_median <= CBV_MEDIAN_RANGE[1]:
                confidence = _downgrade_confidence(confidence, "moderate")
                result.reasons.append(f"nCBV median {result.cbv_median:.2f} outside {CBV_MEDIAN_RANGE}")
    result.confidence = confidence
    result.use_constraint = confidence != "low"
    if not result.reasons:
        result.reasons.append("Tissue CNR and nCBV checks passed")
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
    """Select a soft prior from tissue QC and leakage-reference availability.

    Low tissue confidence, disabled correction or an unavailable correction
    reference withholds the prior. Eligible maps use a fixed relative SD of
    0.60. K2 and correction residuals remain descriptive.

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
                "DSC tissue signal, leakage-reference, and cross-modal registration "
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
    dsc_qc_data: dict[str, Any],
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

    Requires an available NAWM-normalized map, accepted registration QC,
    finite tissue CNR and valid support after spatial transfer. The derived
    prior median must satisfy the configured magnitude bounds.

    Parameters
    ----------
    ncbv_file : Optional[Union[str, Path]]
        Path to DSC nCBV NIfTI, or None / missing → clean SKIP.
    dsc_qc_data : dict[str, Any]
        Tissue QC metrics dictionary from the DSC workflow.
    brain_mask_file : Union[str, Path]
        Path to brain mask NIfTI.
    mode_override : str, optional
        User-specified constraint mode override.
    k2_metrics : dict[str, Any], optional
        K2 leakage correction metrics dict from dsc_preload.
    dose : float, optional
        Administered DSC dose as a multiple of the agent's standard dose.
    agent_standard_dose : float, optional
        Standard agent dose in mmol Gd/kg for dose-adjusted tissue CNR checks.
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

    # The prior requires an available NAWM-normalized CBV map.
    if ncbv_file is None or not Path(str(ncbv_file)).is_file():
        logger.info(
            "vp constraint skipped: NAWM-normalized CBV map unavailable. "
            "DCE fitting uses the model's ordinary priors and bounds."
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
        dsc_qc_data=dsc_qc_data if dsc_qc_data else {},
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
        "mean_cnr": quality.mean_cnr,
        # Unit annotations so downstream consumers don't conflate scales.
        "cbv_median_units": "ncbv",
        "vp_median_units": "fraction",
        "nawm_vp_reference": NAWM_VP_REFERENCE,
        "registration_statuses": registration_statuses,
    }

    if vp_valid.size == 0:
        result.config = ConstraintConfig(
            mode=ConstraintMode.SKIP, vp_uncertainty=0.0,
            reason="No finite DSC-derived vp support was available in the DCE brain mask.",
        )
        qc_report.update(mode="skip", reason=result.config.reason, vp_uncertainty=0.0,
                         skip_reason="empty_prior_support")
        result.qc_report = qc_report
        return result

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
        result.vp_map = np.where(mask_bool, vp_map, np.nan)
    else:
        result.vp_map = None

    return result
