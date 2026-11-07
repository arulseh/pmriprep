"""AIF provenance stamping for derivative sidecars.

The AIF timeseries and mask carry detailed ``PerfusionProcessing.AIF``
provenance. Other derivatives carry a compact source/scaling summary;
quantitative maps additionally carry the per-map ``AIFDependent`` flag.

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
    "psr", "ncbv", "auc", "auc_corr", "ttp",
    "cnr", "tsnr",
    "k2",
})


#: DSC map keys registered for AIF-dependency provenance.
DSC_AIF_MAP_KEYS: Final[frozenset[str]] = frozenset({
    "cbf", "cbv", "cbv_auc", "auc", "auc_corr", "mtt", "tmax",
    "arrival_delay", "psr", "ttp", "cnr", "tsnr", "ncbv", "ncbf",
    "cbf_std", "rel_cbf", "rel_cbv", "parametric_r2", "fallback", "k2",
})


# ── DCE registries ────────────────────────────────────────────────────

#: DCE keys whose computation does not consume the AIF.
DCE_AIF_INDEPENDENT_MAPS: Final[frozenset[str]] = frozenset({
    "iauc60", "iauc90", "iauc120",
    "peak_enhancement", "ttp", "ser", "tsnr", "mean",
})


#: DCE map keys registered for AIF-dependency provenance.
DCE_AIF_MAP_KEYS: Final[frozenset[str]] = frozenset({
    # PK maps
    "ktrans", "ve", "kep", "vp", "fp", "ps", "e", "delay",
    # Uncertainty
    "ktrans_uncertainty", "ve_uncertainty", "vp_uncertainty",
    "fp_uncertainty", "ps_uncertainty",
    # IAUC (tissue concentration integral, AIF-independent)
    "iauc60", "iauc90", "iauc120",
    # Model-free (AIF-independent)
    "peak_enhancement", "ttp", "ser", "tsnr", "mean",
    # Smoothed PK maps
    "ktrans_sm", "ve_sm", "kep_sm", "vp_sm", "fp_sm", "ps_sm",
    # Fit/model selection diagnostics (AIF shapes forward-model residuals)
    "model_selection", "delta_bic", "r2",
})


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
    if detection_level == "unavailable":
        return True
    parts = split_detection_level(detection_level)
    return bool(parts) and all(p in DSC_DETECTION_LEVELS for p in parts)


def _derive_aif_source(
    aif_qc: dict[str, Any],
    modality: Modality = "DSC",
) -> str:
    """Derive ``Source`` value from AIF QC metrics."""
    detection_level = aif_qc.get("detection_method", "")
    if detection_level == "unavailable":
        return "unavailable"
    if detection_level == "population":
        return "population"
    if detection_level == "external":
        return "external"
    return "measured"


def _derive_quality_flag(
    aif_qc: dict[str, Any],
    source: str,
    modality: Modality = "DSC",
) -> str:
    """Derive ``QualityFlag`` from source and gate/score quality.

    Unavailable and population sources receive ``poor``. DCE uses its reported ``aif_confidence``. DSC uses the
    accepted cascade tier, matching the overall QC recommendation and HTML;
    its reported score is a transformed candidate-ranking cost rather than a
    calibrated absolute-quality scale.
    """
    if source in {"unavailable", "population"}:
        return "poor"
    if source == "external":
        return "warn"
    if modality == "DSC":
        detection_level = str(aif_qc.get("detection_method", ""))
        parts = set(split_detection_level(detection_level))
        if not parts:
            return "warn"
        if parts <= {"atlas_strict", "atlas_relaxed", "user_mask_strict", "user_mask_relaxed"}:
            return "pass"
        if parts & {
            "atlas_permissive",
            "user_mask_permissive",
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
    *,
    compact: bool = False,
) -> dict[str, Any]:
    """Assemble the ``PerfusionProcessing.AIF`` sub-block from AIF QC.

    Accepts either the merged workflow ``qc_metrics``
    (``{"aif_detection": {...}, ...}``) or the flat interface
    ``qc_metrics`` output (``{"detection_method": ..., ...}``).
    Returns an empty dict when AIF QC is absent.
    With ``compact=True``, omit search diagnostics and curve-shape summaries.

    For DSC, ``UsedAsCBVDenominator`` indicates whether the selected curve
    supplies the AUC-CBV denominator.
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
    }
    if aif_qc.get("detection_status"):
        block["DetectionStatus"] = aif_qc["detection_status"]
    for key, field in (
        ("search_region", "SearchRegion"),
        ("failure_reason", "FailureReason"),
        ("failure_detail", "FailureDetail"),
    ):
        if aif_qc.get(key):
            block[field] = aif_qc[key]
    qs = aif_qc.get("quality_score")
    if qs is not None:
        block["QualityScore"] = float(qs)
    n_voxels = aif_qc.get("n_aif_voxels_selected")
    if n_voxels is not None:
        block["NumberOfVoxelsSelected"] = int(n_voxels)
    raw_metrics = aif_qc.get("aif_metrics")
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    if modality == "DCE":
        peak = aif_qc.get("emitted_peak_mm")
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
    if modality == "DCE":
        for source_key, output_key in (
            ("final_peak_time_seconds", "PeakTimeSeconds"),
            ("baseline_reference_time_seconds", "BaselineReferenceTimeSeconds"),
        ):
            if aif_qc.get(source_key) is not None:
                block[output_key] = float(aif_qc[source_key])
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
    laterality_midline = aif_qc.get("laterality_midline_method")
    if laterality_midline is not None:
        block["LateralityMidlineMethod"] = str(laterality_midline)
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
    hct = aif_qc.get("hematocrit")
    if hct is None:
        hct = aif_qc.get("hematocrit_used")
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
        gamma_converged = aif_qc.get("gamma_fit_converged")
        if gamma_converged is not None:
            block["GammaFitConverged"] = bool(gamma_converged)
        gamma_failure = aif_qc.get("gamma_fit_failure_reason")
        if gamma_failure is not None:
            block["GammaFitFailureReason"] = str(gamma_failure)
        gamma_r2 = aif_qc.get("gamma_fit_r_squared")
        if gamma_r2 is not None:
            block["GammaFitRSquared"] = float(gamma_r2)
        if "gamma_fit_evaluation_start_frame" in aif_qc:
            block["GammaFitEvaluationWindow"] = {
                "StartFrame": aif_qc["gamma_fit_evaluation_start_frame"],
                "EndFrameExclusive": aif_qc["gamma_fit_evaluation_end_frame"],
                "RSquaredCurve": "unscaled_gamma_model",
            }
        diagnostics = aif_qc.get("gamma_fit_diagnostics")
        if diagnostics:
            block["GammaFitDiagnostics"] = diagnostics
        block["UsedAsCBVDenominator"] = source in {"measured", "external"}

    # DCE-specific fields
    if modality == "DCE":
        confidence = aif_qc.get("aif_confidence")
        if confidence is not None:
            block["Confidence"] = str(confidence)
        if aif_qc.get("aif_confidence_reason") is not None:
            block["ConfidenceReason"] = str(aif_qc["aif_confidence_reason"])
        if aif_qc.get("aif_support_voxels") is not None:
            block["SupportVoxels"] = int(aif_qc["aif_support_voxels"])
        inflation = aif_qc.get("aif_uncertainty_inflation_factor")
        if inflation is not None:
            block["UncertaintyInflationFactor"] = float(inflation)
        if source == "population":
            block["PopulationModel"] = "parker"
            reason = aif_qc.get("auto_fallback_reason")
            if reason:
                block["PopulationFallbackReason"] = str(reason)
        peak_scaling = build_peak_scaling_provenance(aif_qc)
        if peak_scaling:
            block["PeakScaling"] = peak_scaling

    if source == "external":
        block["ExternalAIFVerified"] = False

    if compact:
        retained = {
            "Source", "DetectionLevel", "QualityFlag", "Confidence",
            "ConfidenceReason", "SupportVoxels", "NumberOfVoxelsSelected",
            "Hematocrit", "PeakScaling", "UncertaintyInflationFactor",
            "PopulationModel", "PopulationFallbackReason", "ExternalAIFVerified",
            "UsedAsCBVDenominator", "DoseFactor", "DetectionStatus",
            "FailureReason", "FailureDetail",
        }
        block = {key: value for key, value in block.items() if key in retained}
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
    """Stamp AIF dependence and merge compact source/scaling provenance.

    Stamps ``AIFDependent`` on ``sidecar`` and
    merges the ``PerfusionProcessing.AIF`` provenance block.

    Raises
    ------
    KeyError
        If the map key is absent from the modality's AIF-dependency registry.
    """
    if modality == "DCE":
        registered_maps = DCE_AIF_MAP_KEYS
        independent_maps = DCE_AIF_INDEPENDENT_MAPS
        module_name = "pmriprep.utils.aif_provenance.DCE_AIF_MAP_KEYS"
    else:
        registered_maps = DSC_AIF_MAP_KEYS
        independent_maps = AIF_INDEPENDENT_MAPS
        module_name = "pmriprep.utils.aif_provenance.DSC_AIF_MAP_KEYS"

    if map_key not in registered_maps:
        raise KeyError(
            f"Unknown {modality} map key '{map_key}'. Register it in "
            f"{module_name}."
        )
    sidecar["AIFDependent"] = map_key not in independent_maps
    block = build_aif_provenance_block(qc_metrics, modality, compact=True)
    if block:
        perf = dict(sidecar.get("PerfusionProcessing", {}))
        perf["AIF"] = block
        sidecar["PerfusionProcessing"] = perf


def stamp_aif_dependency(
    sidecar: dict[str, Any],
    map_key: str,
    modality: Modality = "DCE",
) -> None:
    """Stamp ``AIFDependent`` for a registered map key.

    Used by maps routed through ``_build_dce_meta``. Unknown nonempty keys
    produce a warning; unknown or empty keys leave the sidecar unchanged.
    """
    if modality == "DCE":
        registered_maps = DCE_AIF_MAP_KEYS
        independent_maps = DCE_AIF_INDEPENDENT_MAPS
    else:
        registered_maps = DSC_AIF_MAP_KEYS
        independent_maps = AIF_INDEPENDENT_MAPS

    if map_key not in registered_maps:
        if map_key:
            logger.warning(
                "%s map key %r not registered for AIF provenance; "
                "AIFDependent tag omitted.",
                modality, map_key,
            )
        return
    sidecar["AIFDependent"] = map_key not in independent_maps
