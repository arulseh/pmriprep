"""DSC AIF provenance stamping for derivative sidecars.

Every DSC map sidecar carries a ``PerfusionProcessing.AIF`` block plus
per-map ``AIFDependent`` and ``TemplateAIFUsage`` tags so downstream
consumers can filter correctly when a synthetic/template AIF was used as
a rescue.

This module is the canonical home for the AIF provenance helpers that
were originally embedded in ``pmriprep/interfaces/dsc_outputs.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Final, Literal

from pmriprep.utils.dsc_constants import DSC_DETECTION_LEVELS, is_valid_detection_level

logger = logging.getLogger(__name__)


#: Keys whose *computation* does not consume the AIF.
#: Under any AIF provenance these maps are unaffected.
AIF_INDEPENDENT_MAPS: Final[frozenset[str]] = frozenset({
    "k2",                                          # BW leakage (reference-tissue based)
    "psr", "ph",                                   # signal features
    "cnr", "snr", "tsnr",                          # quality metrics
    "auc_diff",                                    # tissue AUC difference
    "ttp", "ttpCorrected", "rttp",                 # tissue peak timing
    "parametric_r2",                               # model-fit diagnostic (CPI/Vascular only)
    "fallback",                                    # binary gate decision (R²<0.5)
})


TemplateAIFUsageCategory = Literal[
    "safe", "ratio_defensible", "qualitative_only"
]


#: How to interpret each DSC map under a template (synthetic) AIF:
#:
#: - ``safe``: map is unaffected (or ratio cancels the AIF amplitude).
#: - ``ratio_defensible``: map is AIF-scaled; absolute magnitudes are
#:   not comparable across subjects, but within-subject lesion:NAWM
#:   ratios remain interpretable for grading/trending.
#: - ``qualitative_only``: AIF shape/timing mismatch biases the map
#:   via deconvolution; treat as qualitative.
#:
#: New maps must declare their usage explicitly —
#: :func:`apply_aif_sidecar` raises ``KeyError`` on unknown keys.
TEMPLATE_AIF_USAGE: Final[dict[str, TemplateAIFUsageCategory]] = {
    # rel_maps (keyed by map_suffix)
    "cbf": "ratio_defensible",
    "cbv": "ratio_defensible",
    "cbv_auc": "ratio_defensible",
    "auc": "ratio_defensible",
    "auc_corr": "ratio_defensible",
    # core_maps
    "mtt": "qualitative_only",
    "tmax": "qualitative_only",
    "psr": "safe",
    "ph": "safe",
    # optional_maps
    "ttp": "safe",
    "ttpCorrected": "safe",
    "rttp": "safe",
    "cnr": "safe",
    "auc_diff": "safe",
    "snr": "safe",
    "tsnr": "safe",
    # normalized (NAWM ratio cancels AIF amplitude)
    "ncbv": "safe",
    "ncbf": "safe",
    # leakage
    "k2": "safe",
    # vascular model (deconvolution-based)
    "alpha": "qualitative_only",
    "beta": "qualitative_only",
    "cth": "qualitative_only",
    # Bayesian uncertainty (mirror underlying mean map)
    "cbf_std": "ratio_defensible",
    "cbv_std": "ratio_defensible",
    "cbfCV": "ratio_defensible",
    "cbvCV": "ratio_defensible",
    "cbfCI95lower": "ratio_defensible",
    "cbfCI95upper": "ratio_defensible",
    "cbvCI95lower": "ratio_defensible",
    "cbvCI95upper": "ratio_defensible",
    # smoothed hemodynamic map keys (distinct from raw rel_maps keys)
    "rel_cbf": "ratio_defensible",
    "rel_cbv": "ratio_defensible",
    # smoothed AUC keys
    "rel_auc_corr": "ratio_defensible",
    # parametric-fit diagnostics (R²/fallback gate decision are AIF-amplitude-independent)
    "parametric_r2": "safe",
    "fallback": "safe",
}


def _derive_aif_source(aif_qc: dict[str, Any]) -> str:
    """Derive ``Source`` value from ``qc_metrics['aif_detection']``."""
    detection_level = aif_qc.get("detection_method", "")
    if detection_level == "synthetic":
        return "template"
    if detection_level == "external":
        return "external"
    if aif_qc.get("gamma_fit_applied"):
        return "gamma_fit_measured"
    return "measured"


def _derive_quality_flag(aif_qc: dict[str, Any]) -> str:
    """Derive ``QualityFlag`` (pass/warn/poor) score-primary with
    ``synthetic`` forced to poor. Thresholds 0.7 / 0.5 align with
    Willats & Calamante / Mouridsen composite AIF-QC heuristics.
    """
    detection_level = aif_qc.get("detection_method", "")
    if detection_level == "synthetic":
        return "poor"
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
) -> dict[str, Any]:
    """Assemble the ``PerfusionProcessing.AIF`` sub-block from AIF QC.

    Accepts either the merged workflow ``qc_metrics`` (``{"aif_detection": {...}, ...}``)
    or the flat ``DSCAIFDetection.qc_metrics`` output (``{"detection_method": ..., ...}``).
    Returns an empty dict when AIF QC is absent — callers should then
    skip stamping the block.
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
    if not is_valid_detection_level(str(detection_level) if detection_level else ""):
        logger.warning(
            "Unknown DSC AIF detection_level %r; expected one or more "
            "(joined with '+') of %s. Skipping AIF provenance block.",
            detection_level, DSC_DETECTION_LEVELS,
        )
        return {}
    source = _derive_aif_source(aif_qc)
    block: dict[str, Any] = {
        "Source": source,
        "DetectionLevel": aif_qc.get("detection_method", "unknown"),
        "QualityFlag": _derive_quality_flag(aif_qc),
        "GammaFitApplied": bool(aif_qc.get("gamma_fit_applied", False)),
        "CrossSubjectComparable": source in {
            "measured", "gamma_fit_measured", "external",
        },
    }
    if source == "template":
        reason = (
            qc_metrics.get("synthetic_fallback_reason")
            or aif_qc.get("synthetic_fallback_reason")
            or "cascade_exhausted"
        )
        block["SyntheticFallbackReason"] = str(reason)
    if source == "external":
        block["ExternalAIFVerified"] = False
    return block


def apply_aif_sidecar(
    sidecar: dict[str, Any],
    map_key: str,
    qc_metrics: dict[str, Any] | None,
) -> None:
    """Stamp ``AIFDependent`` + ``TemplateAIFUsage`` on ``sidecar`` and
    merge the ``PerfusionProcessing.AIF`` provenance block.

    Raises ``KeyError`` when ``map_key`` is not registered in
    :data:`TEMPLATE_AIF_USAGE` — new maps must opt in explicitly.
    """
    if map_key not in TEMPLATE_AIF_USAGE:
        raise KeyError(
            f"Unknown DSC map key '{map_key}'. Register it in "
            "pmriprep.utils.dsc_aif_provenance.TEMPLATE_AIF_USAGE with "
            "the appropriate tag ('safe' / 'ratio_defensible' / "
            "'qualitative_only')."
        )
    sidecar["AIFDependent"] = map_key not in AIF_INDEPENDENT_MAPS
    sidecar["TemplateAIFUsage"] = TEMPLATE_AIF_USAGE[map_key]
    block = build_aif_provenance_block(qc_metrics)
    if block:
        perf = sidecar.setdefault("PerfusionProcessing", {})
        perf["AIF"] = block
