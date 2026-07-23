"""AIF provenance stamping for derivative sidecars.

Every perfusion map sidecar carries a ``PerfusionProcessing.AIF`` block
plus per-map ``AIFDependent`` and ``TemplateAIFUsage`` tags so downstream
consumers can filter correctly when a non-subject-specific AIF was used.

Supports both DSC and DCE modalities via a ``modality`` parameter.
"""
from __future__ import annotations

import logging
from typing import Any, Final, Literal

from pmriprep.utils.dce_constants import DCE_DETECTION_LEVELS
from pmriprep.utils.dsc_constants import DSC_DETECTION_LEVELS, split_detection_level

logger = logging.getLogger(__name__)


Modality = Literal["DSC", "DCE"]


# ── DSC registries ────────────────────────────────────────────────────

#: DSC keys whose computation does not consume the AIF.
AIF_INDEPENDENT_MAPS: Final[frozenset[str]] = frozenset({
    "k2",
    "psr", "ph",
    "cnr", "snr", "tsnr",
    "auc_diff",
    "ttp", "ttpCorrected", "rttp",
    "parametric_r2",
    "fallback",
    "cbv_firstpass",
})


TemplateAIFUsageCategory = Literal[
    "safe", "ratio_defensible", "qualitative_only"
]


#: How to interpret each DSC map under a synthetic AIF.
TEMPLATE_AIF_USAGE: Final[dict[str, TemplateAIFUsageCategory]] = {
    "cbf": "ratio_defensible",
    "cbv": "ratio_defensible",
    "cbv_auc": "ratio_defensible",
    # First-pass CBV is normalized tissue-to-tissue (no AIF): AIF-independent.
    "cbv_firstpass": "safe",
    "auc": "ratio_defensible",
    "auc_corr": "ratio_defensible",
    "mtt": "qualitative_only",
    "tmax": "qualitative_only",
    "psr": "safe",
    "ph": "safe",
    "ttp": "safe",
    "ttpCorrected": "safe",
    "rttp": "safe",
    "cnr": "safe",
    "auc_diff": "safe",
    "snr": "safe",
    "tsnr": "safe",
    "ncbv": "safe",
    "ncbf": "safe",
    "k2": "safe",
    "alpha": "qualitative_only",
    "beta": "qualitative_only",
    "cth": "qualitative_only",
    "cbf_std": "ratio_defensible",
    "cbv_std": "ratio_defensible",
    "cbfCV": "ratio_defensible",
    "cbvCV": "ratio_defensible",
    "cbfCI95lower": "ratio_defensible",
    "cbfCI95upper": "ratio_defensible",
    "cbvCI95lower": "ratio_defensible",
    "cbvCI95upper": "ratio_defensible",
    "rel_cbf": "ratio_defensible",
    "rel_cbv": "ratio_defensible",
    "rel_auc_corr": "ratio_defensible",
    "parametric_r2": "safe",
    "fallback": "safe",
}


# ── DCE registries ────────────────────────────────────────────────────

#: DCE keys whose computation does not consume the AIF.
DCE_AIF_INDEPENDENT_MAPS: Final[frozenset[str]] = frozenset({
    "iauc60", "iauc90", "iauc120",
    "peak_enhancement", "ttp", "ser",
    "wash_in_slope", "wash_out_slope",
    "bat", "curve_type",
})


#: How to interpret each DCE map under a population AIF.
DCE_POPULATION_AIF_USAGE: Final[dict[str, TemplateAIFUsageCategory]] = {
    # PK maps
    "ktrans": "ratio_defensible",
    "ve": "ratio_defensible",
    "kep": "ratio_defensible",
    "vp": "ratio_defensible",
    "fp": "ratio_defensible",
    "ps": "ratio_defensible",
    "e": "ratio_defensible",
    # Uncertainty
    "ktrans_uncertainty": "ratio_defensible",
    "ve_uncertainty": "ratio_defensible",
    "vp_uncertainty": "ratio_defensible",
    "fp_uncertainty": "ratio_defensible",
    "ps_uncertainty": "ratio_defensible",
    # CV (std/mean, AIF cancels)
    "ktrans_cv": "safe",
    "ve_cv": "safe",
    "vp_cv": "safe",
    "fp_cv": "safe",
    "ps_cv": "safe",
    # CI95
    "ktrans_ci95_lower": "ratio_defensible",
    "ktrans_ci95_upper": "ratio_defensible",
    "ve_ci95_lower": "ratio_defensible",
    "ve_ci95_upper": "ratio_defensible",
    "vp_ci95_lower": "ratio_defensible",
    "vp_ci95_upper": "ratio_defensible",
    "fp_ci95_lower": "ratio_defensible",
    "fp_ci95_upper": "ratio_defensible",
    "ps_ci95_lower": "ratio_defensible",
    "ps_ci95_upper": "ratio_defensible",
    # IAUC (tissue concentration integral, AIF-independent)
    "iauc60": "safe",
    "iauc90": "safe",
    "iauc120": "safe",
    # Normalized (ratio cancels)
    "nktrans": "safe",
    "niauc60": "safe",
    # Model-free (AIF-independent)
    "peak_enhancement": "safe",
    "ttp": "safe",
    "ser": "safe",
    "wash_in_slope": "safe",
    "wash_out_slope": "safe",
    "bat": "safe",
    "curve_type": "safe",
    # Smoothed PK maps
    "ktrans_sm": "ratio_defensible",
    "ve_sm": "ratio_defensible",
    "kep_sm": "ratio_defensible",
    "vp_sm": "ratio_defensible",
    "fp_sm": "ratio_defensible",
    "ps_sm": "ratio_defensible",
    # Model selection diagnostics (AIF shapes forward-model residuals)
    "model_selection": "qualitative_only",
    "delta_bic": "qualitative_only",
}


# ── Shared logic ──────────────────────────────────────────────────────

def _is_valid_detection_level(
    detection_level: str,
    modality: Modality = "DSC",
) -> bool:
    """Check whether a detection level is valid for the given modality."""
    if not isinstance(detection_level, str) or not detection_level:
        return False
    if modality == "DCE":
        return detection_level in DCE_DETECTION_LEVELS
    parts = split_detection_level(detection_level)
    return bool(parts) and all(p in DSC_DETECTION_LEVELS for p in parts)


def _derive_aif_source(
    aif_qc: dict[str, Any],
    modality: Modality = "DSC",
) -> str:
    """Derive ``Source`` value from AIF QC metrics."""
    detection_level = aif_qc.get("detection_method", "")
    if detection_level == "synthetic":
        return "synthetic"
    if detection_level == "population":
        return "population"
    if detection_level == "external":
        return "external"
    if modality == "DSC" and aif_qc.get("gamma_fit_applied"):
        return "gamma_fit_measured"
    return "measured"


def _derive_quality_flag(
    aif_qc: dict[str, Any],
    source: str,
    modality: Modality = "DSC",
) -> str:
    """Derive ``QualityFlag`` from source and gate/score quality.

    Non-subject-specific sources (synthetic, population) are forced to
    ``poor``. For DCE the emitted ``quality_score`` is a report-only
    AIF-shape-fidelity metric (systematically compressed, physics-scoped
    as not for accept/reject), so the flag is derived from the gate-based
    ``aif_confidence`` instead — otherwise every normal detected AIF is
    mislabelled ``poor``. DSC's ``quality_score`` is a 0-1 composite whose
    0.7 / 0.5 thresholds (Willats & Calamante / Mouridsen heuristics) hold.
    """
    if source in {"synthetic", "population"}:
        return "poor"
    if modality == "DCE":
        confidence = aif_qc.get("aif_confidence")
        if confidence == "standard":
            return "pass"
        if confidence in {"reduced", "low"}:
            return "warn"
        # confidence absent — fall through to the score heuristic
    score = aif_qc.get("quality_score")
    if score is None:
        return "warn"
    if score >= 0.7:
        return "pass"
    if score >= 0.5:
        return "warn"
    return "poor"


def build_aif_provenance_block(
    qc_metrics: dict[str, Any] | None,
    modality: Modality = "DSC",
) -> dict[str, Any]:
    """Assemble the ``PerfusionProcessing.AIF`` sub-block from AIF QC.

    Accepts either the merged workflow ``qc_metrics``
    (``{"aif_detection": {...}, ...}``) or the flat interface
    ``qc_metrics`` output (``{"detection_method": ..., ...}``).
    Returns an empty dict when AIF QC is absent.
    """
    if not isinstance(qc_metrics, dict):
        return {}
    if "detection_method" in qc_metrics:
        aif_qc = qc_metrics
    else:
        aif_qc = qc_metrics.get("aif_detection") or {}
    if not aif_qc or "detection_method" not in aif_qc:
        return {}
    detection_level = aif_qc.get("detection_method")
    if not _is_valid_detection_level(
        str(detection_level) if detection_level else "",
        modality,
    ):
        logger.warning(
            "Unknown %s AIF detection_level %r. "
            "Skipping AIF provenance block.",
            modality, detection_level,
        )
        return {}

    source = _derive_aif_source(aif_qc, modality)
    block: dict[str, Any] = {
        "Source": source,
        "DetectionLevel": aif_qc.get("detection_method", "unknown"),
        "QualityFlag": _derive_quality_flag(aif_qc, source, modality),
        "CrossSubjectComparable": source in {
            "measured", "gamma_fit_measured", "external",
        },
    }
    qs = aif_qc.get("quality_score")
    if qs is not None:
        block["QualityScore"] = float(qs)
    hct = aif_qc.get("hematocrit") or aif_qc.get("hematocrit_used")
    if hct is not None:
        block["Hematocrit"] = float(hct)

    # DSC-specific fields
    if modality == "DSC":
        block["GammaFitApplied"] = bool(
            aif_qc.get("gamma_fit_applied", False),
        )
        if source == "synthetic":
            reason = (
                qc_metrics.get("synthetic_fallback_reason")
                or aif_qc.get("synthetic_fallback_reason")
                or "cascade_exhausted"
            )
            block["SyntheticFallbackReason"] = str(reason)

    # DCE-specific fields
    if modality == "DCE":
        confidence = aif_qc.get("aif_confidence")
        if confidence is not None:
            block["Confidence"] = str(confidence)
        inflation = aif_qc.get("aif_uncertainty_inflation_factor")
        if inflation is not None:
            block["UncertaintyInflationFactor"] = float(inflation)
        if source == "population":
            block["PopulationModel"] = "parker"
            reason = (
                aif_qc.get("auto_fallback_reason")
                or "cascade_exhausted"
            )
            block["PopulationFallbackReason"] = str(reason)

    if source == "external":
        block["ExternalAIFVerified"] = False

    return block


def apply_aif_sidecar(
    sidecar: dict[str, Any],
    map_key: str,
    qc_metrics: dict[str, Any] | None,
    modality: Modality = "DSC",
) -> None:
    """Stamp AIF dependency tags and merge the provenance block.

    Stamps ``AIFDependent`` + ``TemplateAIFUsage`` on ``sidecar`` and
    merges the ``PerfusionProcessing.AIF`` provenance block.

    Raises ``KeyError`` when ``map_key`` is not registered in the
    appropriate usage registry.
    """
    if modality == "DCE":
        usage_registry = DCE_POPULATION_AIF_USAGE
        independent_maps = DCE_AIF_INDEPENDENT_MAPS
        module_name = "pmriprep.utils.aif_provenance.DCE_POPULATION_AIF_USAGE"
    else:
        usage_registry = TEMPLATE_AIF_USAGE
        independent_maps = AIF_INDEPENDENT_MAPS
        module_name = "pmriprep.utils.aif_provenance.TEMPLATE_AIF_USAGE"

    if map_key not in usage_registry:
        raise KeyError(
            f"Unknown {modality} map key '{map_key}'. Register it in "
            f"{module_name} with the appropriate tag "
            "('safe' / 'ratio_defensible' / 'qualitative_only')."
        )
    sidecar["AIFDependent"] = map_key not in independent_maps
    sidecar["TemplateAIFUsage"] = usage_registry[map_key]
    block = build_aif_provenance_block(qc_metrics, modality)
    if block:
        perf = sidecar.setdefault("PerfusionProcessing", {})
        perf["AIF"] = block


def stamp_aif_dependency(
    sidecar: dict[str, Any],
    map_key: str,
    modality: Modality = "DCE",
) -> None:
    """Stamp ``AIFDependent`` and ``TemplateAIFUsage`` without full provenance.

    For maps routed through ``_build_dce_meta`` that lack ``qc_metrics``
    plumbing but still need the consumer-facing dependency tags.
    """
    if modality == "DCE":
        usage_registry = DCE_POPULATION_AIF_USAGE
        independent_maps = DCE_AIF_INDEPENDENT_MAPS
    else:
        usage_registry = TEMPLATE_AIF_USAGE
        independent_maps = AIF_INDEPENDENT_MAPS

    if map_key not in usage_registry:
        if map_key:
            logger.warning(
                "%s map key %r not registered in AIF usage registry; "
                "no AIFDependent/TemplateAIFUsage tags stamped.",
                modality, map_key,
            )
        return
    sidecar["AIFDependent"] = map_key not in independent_maps
    sidecar["TemplateAIFUsage"] = usage_registry[map_key]
