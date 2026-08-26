"""AIF provenance stamping for derivative sidecars.

The AIF timeseries, AIF mask, preprocessing, and quantitative-map sidecars
share one ``PerfusionProcessing.AIF`` block. Quantitative maps additionally
carry per-map ``AIFDependent`` and ``TemplateAIFUsage`` tags so downstream
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
    "psr",
    "cnr", "tsnr",
    "k2",
})


TemplateAIFUsageCategory = Literal[
    "safe", "ratio_defensible", "qualitative_only"
]


#: How to interpret each DSC map under a synthetic AIF.
TEMPLATE_AIF_USAGE: Final[dict[str, TemplateAIFUsageCategory]] = {
    "cbf": "ratio_defensible",
    "cbv": "ratio_defensible",
    "cbv_auc": "ratio_defensible",
    "auc": "ratio_defensible",
    "auc_corr": "ratio_defensible",
    "mtt": "qualitative_only",
    "tmax": "qualitative_only",
    "psr": "safe",
    "ttp": "qualitative_only",
    "cnr": "safe",
    "tsnr": "safe",
    "ncbv": "safe",
    "ncbf": "safe",
    "cth": "qualitative_only",
    "cbf_std": "ratio_defensible",
    "rel_cbf": "ratio_defensible",
    "rel_cbv": "ratio_defensible",
    "parametric_r2": "qualitative_only",
    "fallback": "qualitative_only",
    "k2": "safe",
}


# ── DCE registries ────────────────────────────────────────────────────

#: DCE keys whose computation does not consume the AIF.
DCE_AIF_INDEPENDENT_MAPS: Final[frozenset[str]] = frozenset({
    "iauc60", "iauc90", "iauc120",
    "peak_enhancement", "ttp", "ser",
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
    # IAUC (tissue concentration integral, AIF-independent)
    "iauc60": "safe",
    "iauc90": "safe",
    "iauc120": "safe",
    # Model-free (AIF-independent)
    "peak_enhancement": "safe",
    "ttp": "safe",
    "ser": "safe",
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
    ``poor``. DCE uses its gate-based ``aif_confidence``. DSC uses the
    accepted cascade tier, matching the overall QC recommendation and HTML;
    its reported score is a transformed candidate-ranking cost rather than a
    calibrated absolute-quality scale.
    """
    if source in {"synthetic", "population"}:
        return "poor"
    if source == "external":
        return "warn"
    if modality == "DSC":
        detection_level = str(aif_qc.get("detection_method", ""))
        parts = set(split_detection_level(detection_level))
        if not parts:
            return "warn"
        if parts <= {"atlas_strict", "atlas_relaxed"}:
            return "pass"
        if parts & {
            "atlas_permissive",
            "full_atlas_permissive",
            "brain_relaxed",
            "brain_permissive",
            "single_best",
        }:
            return "warn"
        return "warn"
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


def build_peak_scaling_provenance(aif_qc: dict[str, Any]) -> dict[str, Any]:
    """Build the shared DCE AIF ``PeakScaling`` provenance fragment."""
    peak_scaling_keys = {
        "Enabled": "peak_scaling_enabled",
        "Applied": "peak_scaling_applied",
        "RawPeak": "raw_peak_mm",
        "EmittedPeak": "emitted_peak_mm",
        "Factor": "peak_scaling_factor",
        "RequiredFactor": "peak_scaling_required_factor",
        "Policy": "peak_scaling_policy",
        "Trigger": "peak_scaling_trigger",
    }
    peak_scaling: dict[str, Any] = {}
    for output_key, qc_key in peak_scaling_keys.items():
        value = aif_qc.get(qc_key)
        if value is not None:
            peak_scaling[output_key] = value
    if peak_scaling:
        peak_scaling["PeakUnits"] = "mM"
    return peak_scaling


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
    n_voxels = aif_qc.get("n_aif_voxels_selected")
    if n_voxels is not None:
        block["NumberOfVoxelsSelected"] = int(n_voxels)
    raw_metrics = aif_qc.get("aif_metrics")
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    if modality == "DCE":
        peak = aif_qc.get("final_peak_mm")
        if peak is None:
            peak = aif_qc.get("peak_mm_mean")
        if peak is None:
            peak = metrics.get("peak_conc")
    else:
        peak = metrics.get("peak_conc")
    if peak is not None:
        block["PeakAmplitude"] = float(peak)
        block["PeakAmplitudeUnits"] = "1/s" if modality == "DSC" else "mM"
    if modality == "DCE":
        atlas_outcome = aif_qc.get("atlas_search_outcome")
        if atlas_outcome is not None:
            block["AtlasSearchOutcome"] = str(atlas_outcome)
        atlas_support = aif_qc.get("atlas_search_support_voxels")
        if atlas_support is not None:
            block["AtlasSearchSupportVoxels"] = int(atlas_support)
        whole_brain_support = aif_qc.get("whole_brain_search_support_voxels")
        if whole_brain_support is not None:
            block["WholeBrainSearchSupportVoxels"] = int(whole_brain_support)
        symmetry_angle = aif_qc.get("whole_brain_symmetry_angle_degrees")
        if symmetry_angle is not None:
            block["WholeBrainSymmetryAngleDegrees"] = float(symmetry_angle)
        symmetry_overlap = aif_qc.get("whole_brain_symmetry_overlap")
        if symmetry_overlap is not None:
            block["WholeBrainSymmetryOverlap"] = float(symmetry_overlap)
        fwhm = aif_qc.get("final_fwhm_seconds")
        if fwhm is None:
            fwhm = aif_qc.get("fwhm_seconds_mean")
        if fwhm is None:
            fwhm = metrics.get("fwhm")
    else:
        fwhm = metrics.get("fwhm")
    if fwhm is not None:
        block["FWHM"] = float(fwhm)
        block["FWHMUnits"] = (
            "frames"
            if modality == "DSC" and detection_level == "external"
            else "s"
        )
    ttp = metrics.get("ttp")
    if ttp is not None:
        block["TimeToPeak"] = float(ttp)
        block["TimeToPeakUnits"] = "s"
    arrival = aif_qc.get("arrival_time_seconds")
    if arrival is not None:
        block["ArrivalTime"] = float(arrival)
        block["ArrivalTimeUnits"] = "s"
    cascade_tier = aif_qc.get("cascade_final_level")
    if cascade_tier is None:
        cascade_tier = aif_qc.get("arterial_cascade_tier")
    if cascade_tier is not None:
        block["ArterialCascadeTier"] = cascade_tier
    atlas_tier = aif_qc.get("atlas_tier_used")
    if atlas_tier is not None:
        block["AtlasTierUsed"] = str(atlas_tier)
    hemisphere_coverage = aif_qc.get("hemisphere_coverage")
    if hemisphere_coverage is not None:
        block["HemisphereCoverage"] = str(hemisphere_coverage)
    tumor_laterality = aif_qc.get("tumor_laterality")
    if tumor_laterality is not None:
        block["TumorLaterality"] = str(tumor_laterality)
    laterality_mask_source = aif_qc.get("laterality_mask_source")
    if laterality_mask_source is not None:
        block["TumorLateralityMaskSource"] = str(laterality_mask_source)
    n_contralateral = aif_qc.get("n_contralateral")
    if n_contralateral is not None:
        block["NumberOfContralateralCandidates"] = int(n_contralateral)
    n_ipsilateral = aif_qc.get("n_ipsilateral")
    if n_ipsilateral is not None:
        block["NumberOfIpsilateralCandidates"] = int(n_ipsilateral)
    hct = aif_qc.get("hematocrit") or aif_qc.get("hematocrit_used")
    if hct is not None:
        block["Hematocrit"] = float(hct)
    dose_factor = aif_qc.get("dose_factor")
    if dose_factor is not None:
        block["DoseFactor"] = float(dose_factor)

    # DSC-specific fields
    if modality == "DSC":
        brain_edge_exclusion = aif_qc.get("brain_edge_exclusion_target_mm")
        if brain_edge_exclusion is not None:
            block["BrainEdgeExclusionTarget"] = float(brain_edge_exclusion)
            block["BrainEdgeExclusionTargetUnits"] = "mm"
        symmetry_method = aif_qc.get("whole_brain_symmetry_method")
        if symmetry_method is not None:
            block["WholeBrainSymmetryMethod"] = str(symmetry_method)
        symmetry_angle = aif_qc.get("whole_brain_symmetry_angle_degrees")
        if symmetry_angle is not None:
            block["WholeBrainSymmetryAngleDegrees"] = float(symmetry_angle)
        symmetry_overlap = aif_qc.get("whole_brain_symmetry_overlap")
        if symmetry_overlap is not None:
            block["WholeBrainSymmetryOverlap"] = float(symmetry_overlap)
        block["GammaFitApplied"] = bool(
            aif_qc.get("gamma_fit_applied", False),
        )
        gamma_converged = aif_qc.get("gamma_fit_converged")
        if gamma_converged is not None:
            block["GammaFitConverged"] = bool(gamma_converged)
        gamma_failure = aif_qc.get("gamma_fit_failure_reason")
        if gamma_failure is not None:
            block["GammaFitFailureReason"] = str(gamma_failure)
        gamma_r2 = aif_qc.get("gamma_fit_r_squared")
        if gamma_r2 is not None:
            block["GammaFitRSquared"] = float(gamma_r2)
        gamma_auc_ratio = aif_qc.get("gamma_fit_auc_ratio")
        if gamma_auc_ratio is not None:
            block["GammaFitAUCRatio"] = float(gamma_auc_ratio)
        cbv_denominator = aif_qc.get("cbv_auc_denominator_is_synthetic")
        if cbv_denominator is not None:
            block["UsedAsCBVDenominator"] = bool(cbv_denominator)
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
        peak_scaling = build_peak_scaling_provenance(aif_qc)
        if peak_scaling:
            block["PeakScaling"] = peak_scaling

    if source == "external":
        block["ExternalAIFVerified"] = False

    return block


def apply_aif_provenance_block(
    sidecar: dict[str, Any],
    qc_metrics: dict[str, Any] | None,
    modality: Modality,
) -> None:
    """Attach the canonical AIF provenance block without map dependency tags."""
    block = build_aif_provenance_block(qc_metrics, modality)
    if block:
        perf = sidecar.setdefault("PerfusionProcessing", {})
        perf["AIF"] = block


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
    apply_aif_provenance_block(sidecar, qc_metrics, modality)


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
