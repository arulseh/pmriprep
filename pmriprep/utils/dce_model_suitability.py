"""Transparent, non-fatal DCE pharmacokinetic model suitability screening."""

from typing import Any


def _stage_fraction(
    accounting: dict[str, Any],
    stage: str,
    fraction: str,
) -> float | None:
    """Return one named voxel-accounting fraction when available."""
    stage_data = accounting.get(stage)
    if not isinstance(stage_data, dict):
        return None
    value = stage_data.get(fraction)
    return float(value) if value is not None else None


def assess_two_cxm_suitability(
    qc_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Screen 2CXM use for established acquisition and QC concerns.

    This assessment is intentionally non-fatal and does not claim biological
    validity. It warns only from an existing pipeline policy or an unequivocal
    absence of usable output. Other potentially informative metrics are exposed
    without introducing unvalidated subject-level thresholds.

    Parameters
    ----------
    qc_metrics : dict
        Complete producer-shape DCE QC metrics.

    Returns
    -------
    dict
        Reason-coded suitability status, screening scope, and supporting
        acquisition, fit, and input metrics.
    """
    pk = qc_metrics.get("pk_modeling")
    if not isinstance(pk, dict):
        return {
            "model": "2cxm",
            "status": "NOT_EVALUATED",
            "scope": "screening_only_not_biological_validation",
            "reasons": ["pk_modeling_qc_unavailable"],
            "metrics": {},
        }

    model = str(pk.get("model", "unknown"))
    candidates = [str(value) for value in pk.get("candidate_models", [])]
    explicit = model == "2cxm"
    candidate = model == "auto" and "2cxm" in candidates
    if not explicit and not candidate:
        return {
            "model": "2cxm",
            "status": "NOT_APPLICABLE",
            "scope": "screening_only_not_biological_validation",
            "reasons": [],
            "metrics": {},
        }

    accounting = pk.get("voxel_accounting")
    accounting = accounting if isinstance(accounting, dict) else {}
    accepted = accounting.get("quantitative_qc_accepted")
    accepted_count = (
        int(accepted.get("count", 0))
        if isinstance(accepted, dict) and accepted.get("count") is not None
        else None
    )
    aif = qc_metrics.get("aif_detection")
    aif = aif if isinstance(aif, dict) else {}
    t1 = qc_metrics.get("t1_mapping")
    t1 = t1 if isinstance(t1, dict) else {}
    constraint = pk.get("dsc_cbv_constraint")
    constraint = constraint if isinstance(constraint, dict) else {}

    reasons: list[str] = []
    sampling_eligible = pk.get("two_cxm_auto_sampling_eligible")
    if explicit and sampling_eligible is False:
        reasons.append("temporal_sampling_outside_auto_2cxm_eligibility")
    if accepted_count == 0:
        reasons.append("no_quantitatively_accepted_voxels")
    aif_confidence = aif.get("aif_confidence")
    if aif_confidence in {"reduced", "low"}:
        reasons.append("reduced_aif_confidence")

    metrics = {
        "explicit_model": explicit,
        "auto_candidate": candidate,
        "volume_tr_seconds": pk.get("selection_tr_seconds"),
        "acquisition_duration_seconds": pk.get(
            "acquisition_duration_seconds"
        ),
        "auto_sampling_eligible": sampling_eligible,
        "fitting_eligible_fraction_of_brain": _stage_fraction(
            accounting, "fitting_eligible", "fraction_of_brain"
        ),
        "optimizer_converged_fraction_of_attempted": _stage_fraction(
            accounting, "optimizer_converged", "fraction_of_attempted"
        ),
        "retained_fraction_of_attempted": _stage_fraction(
            accounting, "retained_nondegenerate", "fraction_of_attempted"
        ),
        "quantitative_qc_accepted_fraction_of_attempted": _stage_fraction(
            accounting, "quantitative_qc_accepted", "fraction_of_attempted"
        ),
        "quantitative_qc_accepted_fraction_of_retained": _stage_fraction(
            accounting, "quantitative_qc_accepted", "fraction_of_retained"
        ),
        "quantitative_qc_accepted_count": accepted_count,
        "r2_median_retained_converged": pk.get("r2_median"),
        "aif_confidence": aif_confidence,
        "t1_method": t1.get("method", t1.get("t1_method")),
        "b1_correction": t1.get("b1_correction"),
        "vp_prior_mode": constraint.get("mode"),
    }
    status = (
        "WARN"
        if reasons
        else "PASS"
        if accepted_count is not None
        else "NOT_EVALUATED"
    )
    if status == "NOT_EVALUATED":
        reasons.append("voxel_accounting_unavailable")
    return {
        "model": "2cxm",
        "status": status,
        "scope": "screening_only_not_biological_validation",
        "reasons": reasons,
        "metrics": metrics,
    }
