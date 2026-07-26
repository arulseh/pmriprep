"""DCE pipeline derivatives factory.

Wires one ``_copy_perf_map`` Function-based copier per perf-space
output (NIfTI + sidecar) plus a ``DCEFiguresWriter`` for the AIF curve
copy, confounds TSV/JSON, and figure filmstrips. QC summary fields
live on ``desc-dcePreproc_dce.json`` as flat ``QC*`` keys; per-PK-map
lesion stats live on the per-map sidecars as ``LesionStats``
fragments.

Per-output sidecar enrichment uses ``build_dce_base_sidecar`` from
``pmriprep.utils.sidecar_builders``. The PK model name is recorded in
the JSON sidecar (``PKModeling.model`` field) rather than in the
filename — the BIDS schema does not register a ``model-`` entity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.interfaces.qc_writers import (
    DCEFiguresWriter,
    PMRIPrepDataSink,
    write_dataset_description_node,
)


# ---------------------------------------------------------------------------
# Output declarative tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DCEMapSpec:
    """Declarative specification for one regular DCE perfusion-map sink."""

    in_attr: str
    desc: str
    suffix: str
    description: str
    units: str | None
    map_key: str
    extra: dict[str, object] | None = None
    bids_suffix: str = "stat"


@dataclass(frozen=True)
class DCEAnatMaskSpec:
    """Declarative specification for one anatomical DCE-space mask sink."""

    name: str
    in_attr: str
    desc: str
    description: str
    skull_stripped: bool = False
    mask_type: str = "ROI"


@dataclass(frozen=True)
class DCEAnatImageSpec:
    """Declarative specification for one structural DCE-space underlay sink."""

    in_attr: str
    label: str
    description: str
    desc: str = ""
    ceagent: str = ""
    source_attr: str = "source_t1w"


# PK parameter maps. Model name lives in JSON sidecar (PKModeling.model),
# not in the filename (no `model-` entity in BIDS schema).
_PK_MAPS: tuple[DCEMapSpec, ...] = (
    DCEMapSpec("ktrans_map", "dce", "Ktrans", "Volume transfer constant (1/min)", "1/min", "ktrans"),
    DCEMapSpec("ve_map", "dce", "ve", "Extravascular extracellular space volume fraction", "1", "ve"),
    DCEMapSpec("kep_map", "dce", "kep", "Efflux rate constant (1/min)", "1/min", "kep"),
    DCEMapSpec("vp_map", "dce", "vp", "Plasma volume fraction", "1", "vp"),
    DCEMapSpec("fp_map", "dce", "Fp", "Plasma flow (2CXM only)", "mL/min/100mL", "fp"),
    DCEMapSpec("ps_map", "dce", "PS", "Permeability-surface area product (2CXM only)", "mL/min/100mL", "ps"),
    DCEMapSpec("e_map", "dce", "E", "Extraction fraction (2CXM only)", "1", "e"),
)


_CV_MAPS: tuple[DCEMapSpec, ...] = (
    DCEMapSpec("ktrans_cv", "dceCV", "Ktrans", "Ktrans coefficient of variation (std/mean)", "1", "ktrans_cv"),
    DCEMapSpec("ve_cv", "dceCV", "ve", "ve coefficient of variation (std/mean)", "1", "ve_cv"),
    DCEMapSpec("vp_cv", "dceCV", "vp", "vp coefficient of variation (std/mean)", "1", "vp_cv"),
    DCEMapSpec("fp_cv", "dceCV", "Fp", "Fp relative uncertainty (2CXM only)", "1", "fp_cv"),
    DCEMapSpec("ps_cv", "dceCV", "PS", "PS relative uncertainty (2CXM only)", "1", "ps_cv"),
)


# Model-free maps (no model entity). Each entry:
# (input_attr, suffix, description, units, extra_dict_or_None)
_MODEL_FREE_MAPS: tuple[DCEMapSpec, ...] = (
    DCEMapSpec("iauc60_map", "dce", "IAUC60", "Initial Area Under Curve at 60s (mM*min)", "mM*min", "iauc60"),
    DCEMapSpec("iauc90_map", "dce", "IAUC90", "Initial Area Under Curve at 90s (mM*min)", "mM*min", "iauc90"),
    DCEMapSpec("iauc120_map", "dce", "IAUC120", "Initial Area Under Curve at 120s (mM*min)", "mM*min", "iauc120"),
    DCEMapSpec("peak_enhancement_map", "dce", "peakEnhancement", "Peak enhancement percentage", "%", "peak_enhancement"),
    DCEMapSpec("time_to_peak_map", "dce", "ttp", "Time to peak (seconds)", "s", "ttp"),
    DCEMapSpec("ser_map", "dce", "SER", "Signal Enhancement Ratio (washout characterization)", "1", "ser"),
    DCEMapSpec("wash_in_slope_map", "dce", "washInSlope",
     "Maximum upslope of concentration during wash-in (mM/min)", "mM/min",
     "wash_in_slope",
     {"SmoothingMethod": "Savitzky-Golay", "SmoothingWindow": 5,
      "SmoothingPolyorder": 2, "InputCurve": "concentration"}),
    DCEMapSpec("wash_out_slope_map", "dce", "washOutSlope",
     "Linear washout slope (mM/min, negative=washout)", "mM/min",
     "wash_out_slope",
     {"PostPeakStartSeconds": 30.0, "PostPeakEndSeconds": 120.0,
      "MinPoints": 4, "RegressionMethod": "ordinary_least_squares"}),
    DCEMapSpec("bat_map", "dce", "BAT", "Bolus arrival time (seconds)", "s", "bat",
     {"DetectionMethod": "threshold_then_gradient_fallback",
      "ConfidenceThreshold": "high"}),
)


_CURVE_TYPE_MAP = DCEMapSpec(
    "curve_type_map", "dce", "curveType",
    "Curve type classification (1=persistent, 2=plateau, 3=washout)",
    "categorical", "curve_type",
    {"PersistentThreshold": 0.9, "WashoutThreshold": 1.1,
     "ValueMapping": {"0": "unclassified", "1": "persistent",
                      "2": "plateau", "3": "washout"}},
    bids_suffix="dseg",
)


_ANAT_MASKS: tuple[DCEAnatMaskSpec, ...] = (
    DCEAnatMaskSpec(
        "brain_mask_dce", "brain_mask", "brain", "Brain mask in DCE space",
        True, "Brain",
    ),
    DCEAnatMaskSpec(
        "nawm_mask_dce", "nawm_mask", "nawm",
        "Normal-appearing white matter mask in DCE space", mask_type="ROI",
    ),
    DCEAnatMaskSpec(
        "lesion_mask_dce", "lesion_mask_dce", "lesion",
        "Lesion mask registered to DCE space", mask_type="Lesion",
    ),
    DCEAnatMaskSpec(
        "user_lesion_mask_dce", "user_lesion_mask_dce", "userLesion",
        "User-supplied lesion mask in DCE native space", mask_type="Lesion",
    ),
    DCEAnatMaskSpec(
        "roi_mask_dce", "roi_mask_dce", "roi", "ROI mask registered to DCE space",
        mask_type="ROI",
    ),
)


_STRUCTURAL_UNDERLAYS: tuple[DCEAnatImageSpec, ...] = (
    DCEAnatImageSpec("t1w_in_dce", "T1w", "Pre-contrast T1w registered to DCE space"),
    DCEAnatImageSpec(
        "t1w_postcon_in_dce", "T1w", "Post-contrast T1w registered to DCE space",
        ceagent="GAD", source_attr="source_t1w_postcon",
    ),
    DCEAnatImageSpec(
        "t1w_subtraction_in_dce", "T1w",
        "Post-pre T1w subtraction registered to DCE space", desc="subtraction",
    ),
    DCEAnatImageSpec(
        "flair_in_dce", "FLAIR", "FLAIR registered to DCE space",
        source_attr="source_flair",
    ),
    DCEAnatImageSpec(
        "flair_postcon_in_dce", "FLAIR", "Post-contrast FLAIR registered to DCE space",
        ceagent="GAD", source_attr="source_flair_postcon",
    ),
    DCEAnatImageSpec(
        "t2w_in_dce", "T2w", "T2w registered to DCE space",
        source_attr="source_t2w",
    ),
)


_PK_UNITS = {
    "ktrans": "1/min", "ve": "1", "kep": "1/min", "vp": "1",
    "fp": "mL/min/100mL", "ps": "mL/min/100mL", "e": "1",
}
_MF_UNITS = {
    "iauc60": "mM*min", "iauc90": "mM*min", "iauc120": "mM*min",
    "peak_enhancement": "%", "ser": "1",
    "wash_in_slope": "mM/min", "wash_out_slope": "mM/min",
}

# BIDS-correct suffixes for smoothed-map sinks live inline in
# ``_split_dce_smoothed`` (nipype reconstructs Function helpers in an
# empty namespace, so the lookup tables can't be shared via module
# globals). The unsmoothed equivalents are encoded in ``_PK_MAPS`` /
# ``_MODEL_FREE_MAPS`` above; the inline tables mirror those.


# Ktrans is mandatory; every other map is gated by ``present_inputs``.
_ALWAYS_PRESENT_INPUTS = frozenset({"ktrans_map"})


# ---------------------------------------------------------------------------
# Function-node wrappers
# ---------------------------------------------------------------------------


def _build_dce_base(
    source_paths: list[str],
    reference_voxel_source: str,
    bids_dir: str,
    subject: str,
    session: str,
    provenance_dict: dict | None = None,
    qc_metrics: dict | None = None,
    minimal: bool = False,
    pk_dependent: bool = True,
) -> dict:
    from pmriprep.utils.sidecar_builders import build_dce_base_sidecar

    return build_dce_base_sidecar(
        source_paths=source_paths,
        reference_voxel_source=reference_voxel_source,
        bids_dir=bids_dir,
        subject=subject,
        session=session or None,
        provenance_dict=provenance_dict,
        qc_metrics=qc_metrics,
        minimal=minimal,
        pk_dependent=pk_dependent,
    )


def _build_dce_meta(
    base: dict,
    description: str,
    units: str | None = None,
    extra: dict | None = None,
    map_key: str = "",
    lesion_stats_json: str | None = None,
    user_lesion_stats_json: str | None = None,
    roi_stats_json: str | None = None,
) -> dict:
    """Compose a generic DCE sidecar; optionally inject stats fragments.

    When ``map_key`` is non-empty the per-map fragment from each
    aggregate stats JSON is attached as ``LesionStats``,
    ``UserLesionStats``, or ``ROIStats`` respectively.
    Used for model-free maps and NAWM-normalized maps that aren't routed
    through ``_build_dce_pk_map_meta``.
    """
    from pmriprep.utils.aif_provenance import stamp_aif_dependency
    from pmriprep.utils.lesion_stats import load_lesion_stats_fragment

    out = dict(base)
    out["Description"] = description
    if units is not None:
        out["Units"] = units
    if extra:
        out.update(extra)

    if map_key:
        stamp_aif_dependency(out, map_key, modality="DCE")
        fragment = load_lesion_stats_fragment(lesion_stats_json, map_key)
        if fragment:
            out["LesionStats"] = fragment
        user_fragment = load_lesion_stats_fragment(
            user_lesion_stats_json, map_key,
        )
        if user_fragment:
            out["UserLesionStats"] = user_fragment
        roi_fragment = load_lesion_stats_fragment(roi_stats_json, map_key)
        if roi_fragment:
            out["ROIStats"] = roi_fragment
    return out


def _build_aif_mask_meta(
    base: dict,
    description: str,
    qc_metrics: dict | None,
) -> dict:
    """Build an AIF ROI sidecar with the canonical DCE AIF block."""
    from pmriprep.utils.aif_provenance import apply_aif_provenance_block

    out = dict(base)
    out["Description"] = description
    out["Type"] = "ROI"
    apply_aif_provenance_block(out, qc_metrics, modality="DCE")
    return out


def _build_tb1_meta(
    base: dict,
    b1_source_paths: list[str] | None,
    bids_dir: str,
) -> dict:
    """Build TB1map metadata from its DCE grid and selected B1 inputs."""
    from pmriprep.utils.sidecar_builders import build_sources_list

    out = dict(base)
    out["Description"] = (
        "B1+ transmit field map (dimensionless ratio, 1.0 = nominal flip angle)"
    )
    out["Units"] = "1"
    sources = list(out.get("Sources", []))
    sources.extend(build_sources_list(b1_source_paths or [], bids_dir))
    out["Sources"] = list(dict.fromkeys(sources))
    return out


def _build_uncertainty_extra(
    config_dict: dict | None = None,
    qc_metrics: dict | None = None,
    umap_key: str = "ktrans_uncertainty",
) -> dict:
    """Build inference-specific provenance for parameter-uncertainty maps."""
    from pmriprep.utils.uncertainty_provenance import (
        build_uncertainty_provenance,
    )

    estimation = "unknown"
    if isinstance(config_dict, dict):
        estimation = config_dict.get("inference") or "unknown"
    extra = build_uncertainty_provenance(
        inference=str(estimation), map_key=umap_key, modality="DCE"
    )
    if isinstance(qc_metrics, dict):
        uncertainty_qc = qc_metrics.get("parameter_uncertainty")
        if isinstance(uncertainty_qc, dict):
            factors = uncertainty_qc.get("aif_parameter_scale_factors")
            if isinstance(factors, dict):
                extra["AIFUncertaintyInflation"] = {
                    "Applied": any(float(value) > 1.0 for value in factors.values()),
                    "InputFactor": uncertainty_qc.get("aif_inflation_input_factor"),
                    "ParameterScaleFactors": dict(factors),
                    "Method": "heuristic_parameter_specific_scaling",
                    "AppliedTo": "spread_and_interval_half_width",
                }
    if str(estimation).lower() == "map" and isinstance(qc_metrics, dict):
        accounting = qc_metrics.get("map_uncertainty")
        if isinstance(accounting, dict):
            extra["MAPUncertaintyAccounting"] = dict(accounting)
    return extra


def _build_dce_preproc_meta(
    base: dict,
    description: str,
    qc_metrics: dict | None = None,
    provenance_dict: dict | None = None,
    preflight_warnings: list | None = None,
    presence_warnings: list | None = None,
    config_dict: dict | None = None,
) -> dict:
    """Compose the ``desc-dcePreproc_dce.json`` sidecar with flat-key QC.

    Mirrors the DSC ``_build_dsc_preproc_meta`` pattern: starts from the
    perfusion base sidecar, layers on flat ``T1*``/``B1*``/motion/signal
    cosmetic keys via ``enrich_dce_preproc_sidecar``, and finally
    attaches QC recommendation/warnings flat keys via
    ``enrich_recommendation_keys`` so the HTML aggregator can read the
    overall recommendation banner directly from this anchor sidecar.
    ``PipelineQC`` and ``PipelineConfiguration`` nest the full merged
    QC envelope and processing config for the HTML aggregator.
    """
    from pmriprep.utils.sidecar_builders import (
        build_provenance_block,
        enrich_dce_preproc_sidecar,
        enrich_recommendation_keys,
        filter_pipeline_qc,
        sanitize_for_json,
    )

    out = dict(base)
    out["Description"] = description
    out["SkullStripped"] = True
    if isinstance(config_dict, dict):
        out["MotionCorrectionApplied"] = bool(
            config_dict.get("motion_correct", True)
        )
    enrich_dce_preproc_sidecar(
        out,
        qc_metrics=qc_metrics if isinstance(qc_metrics, dict) else None,
    )
    enrich_recommendation_keys(
        out,
        modality="DCE",
        qc_metrics=qc_metrics if isinstance(qc_metrics, dict) else None,
        provenance_dict=(
            provenance_dict if isinstance(provenance_dict, dict) else None
        ),
        preflight_warnings=(
            list(preflight_warnings) if preflight_warnings else None
        ),
        presence_warnings=(
            list(presence_warnings) if presence_warnings else None
        ),
    )
    if isinstance(qc_metrics, dict):
        out["PipelineQC"] = filter_pipeline_qc(qc_metrics)
    if isinstance(config_dict, dict):
        out["PipelineConfiguration"] = build_provenance_block(config_dict)
    return dict(sanitize_for_json(out))



def _build_dce_pk_map_meta(
    base: dict,
    description: str,
    units: str | None = None,
    qc_metrics: dict | None = None,
    map_key: str = "",
    lesion_stats_json: str | None = None,
    user_lesion_stats_json: str | None = None,
    roi_stats_json: str | None = None,
) -> dict:
    """Compose a DCE per-PK-map sidecar with cosmetic enrichment + stats fragments.

    Extracts the per-map fragment for ``map_key`` (e.g. ``"ktrans"``)
    from each aggregate stats JSON and attaches as ``LesionStats``,
    ``UserLesionStats``, or ``ROIStats``. Cross-modality fields
    (``DSCDCECrossValCorrelation``, ``CBVNegativeFraction``) land only on
    the maps they apply to.
    """
    from pmriprep.utils.lesion_stats import load_lesion_stats_fragment
    from pmriprep.utils.sidecar_builders import enrich_dce_pk_map_sidecar

    out = dict(base)
    out["Description"] = description
    if units is not None:
        out["Units"] = units

    fragment = load_lesion_stats_fragment(lesion_stats_json, map_key)

    enrich_dce_pk_map_sidecar(
        out,
        qc_metrics=qc_metrics if isinstance(qc_metrics, dict) else None,
        map_key=map_key,
        lesion_fragment=fragment,
    )

    user_fragment = load_lesion_stats_fragment(
        user_lesion_stats_json, map_key,
    )
    if user_fragment:
        out["UserLesionStats"] = user_fragment
    roi_fragment = load_lesion_stats_fragment(roi_stats_json, map_key)
    if roi_fragment:
        out["ROIStats"] = roi_fragment
    return out


def _build_dce_smoothed_meta(
    base: dict,
    description: str,
    fwhm_mm: str | float,
    units: str | None = None,
    map_key: str = "",
    inference: str = "vi",
) -> dict:
    from pmriprep.utils.aif_provenance import stamp_aif_dependency

    out = dict(base)
    out["Description"] = description
    out["SmoothingFWHM"] = (
        float(fwhm_mm) if str(fwhm_mm).replace(".", "").isdigit() else fwhm_mm
    )
    out["SmoothingType"] = "Gaussian"
    out["OutputMaskApplied"] = True
    out["IntendedFor"] = "visualization"
    out["Note"] = "Use non-smoothed maps for quantitative ROI analysis"
    if units is not None:
        out["Units"] = units
    if map_key not in {
        "ktrans",
        "ve",
        "kep",
        "vp",
        "fp",
        "ps",
        "e",
        "ktrans_uncertainty",
        "ve_uncertainty",
        "vp_uncertainty",
        "fp_uncertainty",
        "ps_uncertainty",
    }:
        out.pop("InputFunctionCompartment", None)
    if map_key:
        stamp_aif_dependency(out, map_key, modality="DCE")
    if map_key.endswith("_uncertainty"):
        from pmriprep.utils.uncertainty_provenance import build_uncertainty_provenance

        out.update(build_uncertainty_provenance(inference, map_key, "DCE"))
    return out


def _build_dce_underlay_meta(
    base: dict,
    description: str,
    units: str | None,
    extra: dict | None,
    structural_source: str,
    perfusion_source: str,
    bids_dir: str,
) -> dict:
    """Build metadata for a structural image resampled to DCE space."""
    from pmriprep.utils.sidecar_builders import build_sources_list

    out = dict(base)
    out["Description"] = description
    if units is not None:
        out["Units"] = units
    if extra:
        out.update(extra)
    out["Sources"] = build_sources_list(
        [structural_source, perfusion_source], bids_dir,
    )
    return out


def _build_model_selection_meta(
    base: dict,
    qc_metrics: dict | None = None,
) -> dict:
    from pmriprep.utils.aif_provenance import stamp_aif_dependency

    out = dict(base)
    out["Description"] = (
        "Per-voxel BIC model selection: complexity-ordered encoding"
    )
    out["LabelEncoding"] = {
        "0": "unfitted",
        "1": "patlak",
        "2": "tofts",
        "3": "extended_tofts",
        "4": "2cxm",
    }
    stamp_aif_dependency(out, "model_selection", modality="DCE")
    if isinstance(qc_metrics, dict):
        pk_modeling = qc_metrics.get("pk_modeling")
        if isinstance(pk_modeling, dict):
            regions = pk_modeling.get("model_selection_regions")
            if regions:
                out["ModelDistribution"] = regions
    return out


def _build_delta_bic_meta(
    base: dict,
    qc_metrics: dict | None = None,
) -> dict:
    from pmriprep.utils.aif_provenance import stamp_aif_dependency

    out = dict(base)
    out["Description"] = (
        "BIC confidence: runner-up minus winner. "
        ">2 = positive, >6 = strong, >10 = very strong "
        "(Kass & Raftery 1995)"
    )
    stamp_aif_dependency(out, "delta_bic", modality="DCE")
    if isinstance(qc_metrics, dict):
        pk = qc_metrics.get("pk_modeling")
        if isinstance(pk, dict):
            if pk.get("r2_median") is not None:
                out["R2Median"] = pk["r2_median"]
            accounting = pk.get("voxel_accounting")
            if isinstance(accounting, dict):
                converged = accounting.get("optimizer_converged")
                if isinstance(converged, dict):
                    fraction = converged.get("fraction_of_attempted")
                    if fraction is not None:
                        out["OptimizerConvergenceFraction"] = fraction
    return out


def _copy_perf_map(
    in_file: str,
    output_dir: str,
    subject: str,
    session: str,
    desc: str,
    suffix: str,
    meta_dict: dict,
    datatype: str = "perf",
    bids_suffix: str = "stat",
    space: str = "",
) -> str:
    """Copy a DCE NIfTI + sidecar with BIDS-compliant naming.

    ``suffix`` is the parameter name (e.g. ``"Ktrans"``, ``"mtt"``)
    which gets folded into the ``desc-`` entity.  ``bids_suffix`` is
    the BIDS-canonical suffix (``stat``, ``mask``, ``dseg``, …).  Pass
    ``suffix=""`` when ``desc`` already encodes everything and only the
    BIDS suffix is needed.
    """
    import logging
    import shutil
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    # Conditional producers may represent a missing output as Nipype
    # Undefined, None, or an empty string.
    if not isdefined(in_file) or in_file == "":
        logging.getLogger("nipype.workflow").warning(
            "Skipping _copy_perf_map sink (desc=%s, suffix=%s): upstream "
            "produced no output (in_file=%r)",
            desc, suffix, in_file,
        )
        return ""

    out_root = Path(output_dir) / "pmriprep" / f"sub-{subject}"
    if session:
        out_root = out_root / f"ses-{session}"
    out_root = out_root / datatype
    out_root.mkdir(parents=True, exist_ok=True)

    from pmriprep.utils.sidecar_builders import source_entity_prefix

    sources = meta_dict.get("Sources", []) if isinstance(meta_dict, dict) else []
    prefix = source_entity_prefix(sources[0] if sources else None, subject, session)
    parts = [prefix]
    if space:
        parts.append(f"space-{space}")
    if desc:
        folded = suffix[:1].upper() + suffix[1:] if suffix else ""
        parts.append(f"desc-{desc}{folded}")
    parts.append(bids_suffix)
    stem = "_".join(parts)
    nii_out = out_root / f"{stem}.nii.gz"
    json_out = out_root / f"{stem}.json"

    shutil.copy2(in_file, nii_out)
    from pmriprep.utils.sidecar_builders import dump_sidecar_json

    json_out.write_text(dump_sidecar_json(meta_dict))
    return str(nii_out)


def _copy_tb1map(
    output_dir: str,
    subject: str,
    session: str,
    meta_dict: dict | None = None,
    in_file: str | None = None,
) -> str:
    """Copy a TB1map NIfTI + sidecar to anat/.

    Bypasses ``PMRIPrepDataSink`` because ``TB1map`` is not in the
    niworkflows default anat-suffix allow-list (`T1w|T2w|T1map|FLAIR|...`).
    All other DCE-space anat outputs (T1w/FLAIR/T2w/T1map) go through
    ``_add_anat_image_sink`` → ``PMRIPrepDataSink``.
    """
    import logging
    import shutil
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    # Defensive no-op when upstream produced no B1+ map (e.g., subject
    # configured b1_method != "none" but the actual mapping failed).
    # See _copy_perf_map for the sentinel-handling rationale.
    if in_file is None or not isdefined(in_file) or in_file == "":
        logging.getLogger("nipype.workflow").warning(
            "Skipping _copy_tb1map sink: upstream produced no output "
            "(in_file=%r)", in_file,
        )
        return ""

    if meta_dict is None:
        raise ValueError("TB1map metadata is required when a B1 map is present")

    out_root = Path(output_dir) / "pmriprep" / f"sub-{subject}"
    if session:
        out_root = out_root / f"ses-{session}"
    out_root = out_root / "anat"
    out_root.mkdir(parents=True, exist_ok=True)

    from pmriprep.utils.sidecar_builders import source_entity_prefix

    sources = meta_dict.get("Sources", []) if isinstance(meta_dict, dict) else []
    prefix = source_entity_prefix(sources[0] if sources else None, subject, session)
    stem = f"{prefix}_space-dceref_TB1map"
    nii_out = out_root / f"{stem}.nii.gz"
    json_out = out_root / f"{stem}.json"

    shutil.copy2(in_file, nii_out)
    from pmriprep.utils.sidecar_builders import dump_sidecar_json

    json_out.write_text(dump_sidecar_json(meta_dict))
    return str(nii_out)


def _copy_dce_xfm(
    in_file: str,
    output_dir: str,
    subject: str,
    session: str,
    source_file: str | None = None,
) -> str:
    """Copy DCE-to-T1w transform .h5 into the derivatives anat/."""
    import logging
    import shutil
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    # Defensive no-op when upstream registration produced no transform
    # (rare, but possible when a subject has DCE imaging but registration
    # to T1w failed). See _copy_perf_map for sentinel-handling rationale.
    if not isdefined(in_file) or in_file == "":
        logging.getLogger("nipype.workflow").warning(
            "Skipping _copy_dce_xfm sink: upstream produced no output "
            "(in_file=%r)", in_file,
        )
        return ""

    out_root = Path(output_dir) / "pmriprep" / f"sub-{subject}"
    if session:
        out_root = out_root / f"ses-{session}"
    out_root = out_root / "anat"
    out_root.mkdir(parents=True, exist_ok=True)

    from pmriprep.utils.sidecar_builders import source_entity_prefix

    prefix = source_entity_prefix(source_file, subject, session)
    fname = f"{prefix}_from-dceref_to-T1w_mode-image_xfm.h5"
    out_path = out_root / fname
    shutil.copy2(in_file, out_path)
    return str(out_path)


def _split_dce_smoothed(
    smooth_pk: dict[str, str] | None = None,
    smooth_mf: dict[str, str] | None = None,
    pk_units: dict[str, str] | None = None,
    mf_units: dict[str, str] | None = None,
) -> tuple[
    list[str], list[str], list[str], list[str],
    list[str], list[str], list[str],
]:
    """Flatten the PK and model-free smoothed-map dicts into output tuples."""
    import re as _re
    from pathlib import Path

    if pk_units is None:
        pk_units = {}
    if mf_units is None:
        mf_units = {}

    # BIDS-correct suffixes for smoothed maps. Inline because this
    # function is reconstructed by nipype in an empty namespace and
    # cannot reference module-level constants.
    pk_suffix_bids = {
        "ktrans": "Ktrans", "ve": "ve", "kep": "kep", "vp": "vp",
        "fp": "Fp", "ps": "PS",
    }
    mf_suffix_bids = {
        "iauc60": "IAUC60", "iauc90": "IAUC90", "iauc120": "IAUC120",
        "peak_enhancement": "peakEnhancement", "ser": "SER",
        "wash_in_slope": "washInSlope", "wash_out_slope": "washOutSlope",
    }

    def _extract_fwhm(src_path: str) -> tuple[str, str]:
        m = _re.search(r"_fwhm(\d+(?:\.\d+)?)", str(src_path))
        if m:
            return f"Sm{m.group(1)}", m.group(1)
        return "SmN", "N"

    in_files: list[str] = []
    descs: list[str] = []
    fwhms: list[str] = []
    units_list: list[str] = []
    desc_values: list[str] = []
    suffixes: list[str] = []
    map_keys: list[str] = []

    if isinstance(smooth_pk, dict):
        for name, src in smooth_pk.items():
            if not (src and Path(src).exists()):
                continue
            fwhm_str, fwhm_mm = _extract_fwhm(src)
            in_files.append(str(src))
            descs.append(f"Smoothed {name} map")
            fwhms.append(fwhm_mm)
            units_list.append(pk_units.get(name, ""))
            desc_values.append(f"dce{fwhm_str}")
            suffixes.append(pk_suffix_bids.get(name, name))
            map_keys.append(f"{name}_sm")

    if isinstance(smooth_mf, dict):
        for name, src in smooth_mf.items():
            if not (src and Path(src).exists()):
                continue
            fwhm_str, fwhm_mm = _extract_fwhm(src)
            in_files.append(str(src))
            descs.append(f"Smoothed model-free {name} map")
            fwhms.append(fwhm_mm)
            units_list.append(mf_units.get(name, ""))
            desc_values.append(f"dce{fwhm_str}")
            suffixes.append(mf_suffix_bids.get(name, name))
            map_keys.append(name)

    return in_files, descs, fwhms, units_list, desc_values, suffixes, map_keys


# ---------------------------------------------------------------------------
# Sink helpers
# ---------------------------------------------------------------------------


def _add_perf_map_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    base_node: pe.Node,
    *,
    name: str,
    in_attr: str,
    desc_value: str,
    suffix: str,
    description: str,
    units: Optional[str],
    extra: Optional[dict],
    extra_node: pe.Node | None = None,
    minimal_base: bool = False,
    map_key: str = "",
    bids_suffix: str = "stat",
) -> pe.Node:
    """Wire one DCE perf-space map sink (Function builder + Function copy).

    When ``map_key`` is non-empty, the meta builder also slices per-map
    fragments from the aggregate stats JSONs (``LesionStats``,
    ``UserLesionStats``, ``ROIStats``) sourced from inputnode.
    """
    meta = pe.Node(
        niu.Function(
            input_names=[
                "base", "description", "units", "extra",
                "map_key", "lesion_stats_json", "user_lesion_stats_json",
                "roi_stats_json",
            ],
            output_names=["meta_dict"],
            function=_build_dce_meta,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.units = units
    if extra_node is None:
        meta.inputs.extra = extra
    else:
        workflow.connect(extra_node, "extra", meta, "extra")
    meta.inputs.map_key = map_key
    workflow.connect(base_node, "meta_dict", meta, "base")
    if map_key:
        workflow.connect(inputnode, "lesion_stats_json", meta, "lesion_stats_json")
        workflow.connect(
            inputnode, "user_lesion_stats_json",
            meta, "user_lesion_stats_json",
        )
        workflow.connect(inputnode, "roi_stats_json", meta, "roi_stats_json")

    sink = pe.Node(
        niu.Function(
            input_names=[
                "in_file", "output_dir", "subject", "session",
                "desc", "suffix", "meta_dict", "datatype", "bids_suffix",
            ],
            output_names=["out_file"],
            function=_copy_perf_map,
        ),
        name=f"{name}_sink",
    )
    sink.inputs.desc = desc_value
    sink.inputs.suffix = suffix
    sink.inputs.datatype = "perf"
    sink.inputs.bids_suffix = bids_suffix
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            (in_attr, "in_file"),
        ]),
        (meta, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


# Model-free maps that should carry per-region LesionStats. Excluded
# (intentional): bat_map (timing offset, not
# magnitude biomarker), curve_type_map (categorical, not amenable to
# ROI summary stats).
_MF_LESIONSTATS_MAP_KEY: dict[str, str] = {
    "iauc60_map": "iauc60",
    "iauc90_map": "iauc90",
    "iauc120_map": "iauc120",
    "peak_enhancement_map": "peak_enhancement",
    "time_to_peak_map": "ttp",
    "ser_map": "ser",
    "wash_in_slope_map": "wash_in_slope",
    "wash_out_slope_map": "wash_out_slope",
}

# NAWM-normalized DCE maps that carry LesionStats.
_NAWMNORM_LESIONSTATS_MAP_KEY: dict[str, str] = {
    "niauc60_map": "niauc60",
}


def _add_dce_pk_map_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    base_node: pe.Node,
    *,
    name: str,
    in_attr: str,
    desc_value: str,
    suffix: str,
    description: str,
    units: Optional[str],
    map_key: str,
    bids_suffix: str = "stat",
) -> pe.Node:
    """Wire a DCE PK-map sink with flat-key cosmetic enrichment.

    Adds ``PK*`` keys + per-map ``LesionStats`` fragment via
    ``_build_dce_pk_map_meta``. Cross-modality fields land only where
    they apply (Ktrans → ``DSCDCECrossValCorrelation``; vp →
    ``CBVNegativeFraction``).
    """
    meta = pe.Node(
        niu.Function(
            input_names=[
                "base", "description", "units", "qc_metrics", "map_key",
                "lesion_stats_json", "user_lesion_stats_json",
                "roi_stats_json",
            ],
            output_names=["meta_dict"],
            function=_build_dce_pk_map_meta,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.units = units
    meta.inputs.map_key = map_key
    workflow.connect([
        (base_node, meta, [("meta_dict", "base")]),
        (inputnode, meta, [
            ("qc_metrics", "qc_metrics"),
            ("lesion_stats_json", "lesion_stats_json"),
            ("user_lesion_stats_json", "user_lesion_stats_json"),
            ("roi_stats_json", "roi_stats_json"),
        ]),
    ])

    sink = pe.Node(
        niu.Function(
            input_names=[
                "in_file", "output_dir", "subject", "session",
                "desc", "suffix", "meta_dict", "datatype", "bids_suffix",
            ],
            output_names=["out_file"],
            function=_copy_perf_map,
        ),
        name=f"{name}_sink",
    )
    sink.inputs.desc = desc_value
    sink.inputs.suffix = suffix
    sink.inputs.datatype = "perf"
    sink.inputs.bids_suffix = bids_suffix
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            (in_attr, "in_file"),
        ]),
        (meta, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


def _add_anat_image_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    base_minimal_node: pe.Node,
    *,
    name: str,
    in_attr: str,
    label: str,
    description: str,
    ce_label: str = "",
    desc: str = "",
    skull_stripped: bool = False,
    units: Optional[str] = None,
    extra: Optional[dict] = None,
    source_attr: str = "source_dce",
    dismiss_entities: tuple = ("acquisition",),
) -> pe.Node:
    """Wire an anat-side DCE sink (T1w/FLAIR/T2w/T1map in DCE space).

    Mirrors DSC's ``_add_anat_sink`` — uses ``PMRIPrepDataSink`` (niworkflows
    ``DerivativesDataSink`` subclass) for BIDS-canonical anat suffixes. The
    space-dceref entity correctly indicates cross-modality registration to DCE
    acquisition geometry.

    Filename entities are inherited from the structural source, not from the
    DCE series that supplies only the target grid.
    """
    is_underlay = source_attr != "source_dce"
    meta = pe.Node(
        niu.Function(
            input_names=(
                ["base", "description", "units", "extra",
                 "structural_source", "perfusion_source", "bids_dir"]
                if is_underlay
                else ["base", "description", "units", "extra"]
            ),
            output_names=["meta_dict"],
            function=_build_dce_underlay_meta if is_underlay else _build_dce_meta,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.units = units
    metadata_extra = dict(extra or {})
    if skull_stripped:
        metadata_extra["SkullStripped"] = True
    meta.inputs.extra = metadata_extra or None
    workflow.connect(base_minimal_node, "meta_dict", meta, "base")
    if is_underlay:
        workflow.connect([
            (inputnode, meta, [
                (source_attr, "structural_source"),
                ("source_dce", "perfusion_source"),
                ("bids_dir", "bids_dir"),
            ]),
        ])

    sink = pe.Node(
        PMRIPrepDataSink(
            check_hdr=True, compress=True,
            allowed_entities=("space", "datatype", "ceagent"),
            space="dceref", suffix=label, datatype="anat",
            dismiss_entities=list(dismiss_entities),
        ),
        name=f"{name}_sink",
    )
    if ce_label:
        sink.inputs.ceagent = ce_label
    if desc:
        sink.inputs.desc = desc
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "base_directory"),
            (in_attr, "in_file"),
            (source_attr, "source_file"),
        ]),
        (meta, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


def _add_anat_mask_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    base_minimal_node: pe.Node,
    *,
    name: str,
    in_attr: str,
    desc_value: str,
    description: str,
    skull_stripped: bool = False,
    mask_type: str = "ROI",
    space: str = "dceref",
) -> pe.Node:
    """Wire a DCE-space mask (brain/nawm/lesion/roi) into anat/."""
    meta = pe.Node(
        niu.Function(
            input_names=["base", "description", "units", "extra"],
            output_names=["meta_dict"],
            function=_build_dce_meta,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.units = None
    meta.inputs.extra = {
        "Type": mask_type,
        **({"SkullStripped": True} if skull_stripped else {}),
    }
    workflow.connect(base_minimal_node, "meta_dict", meta, "base")

    sink = pe.Node(
        niu.Function(
            input_names=[
                "in_file", "output_dir", "subject", "session",
                "desc", "suffix", "meta_dict", "datatype", "bids_suffix",
                "space",
            ],
            output_names=["out_file"],
            function=_copy_perf_map,
        ),
        name=f"{name}_sink",
    )
    sink.inputs.desc = desc_value
    sink.inputs.suffix = ""
    sink.inputs.datatype = "anat"
    sink.inputs.bids_suffix = "mask"
    sink.inputs.space = space
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            (in_attr, "in_file"),
        ]),
        (meta, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def init_dce_datasinks_wf(
    *,
    output_dir: str,
    bids_dir: str,
    subject_id: str,
    session_id: Optional[str],
    pk_model: str,
    inference: str,
    present_inputs: Optional[set[str]] = None,
    has_smoothing: bool = False,
    motion_correct: bool = True,
    b1_source_files: Optional[list[str]] = None,
    t1_map_expected: bool = True,
    b1_map_expected: bool = True,
    name: str = "dce_datasinks_wf",
) -> pe.Workflow:
    """Build the DCE-pipeline DerivativesDataSink chain.

    ``pk_model`` is the resolved model name recorded in PK-map sidecars.
    Model names are not encoded as filename entities because ``model-`` is
    not registered by BIDS.
    """
    if present_inputs is None:
        present_inputs = set()
    present = set(_ALWAYS_PRESENT_INPUTS) | set(present_inputs)
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "output_dir", "bids_dir", "subject_id", "session_id",
                "source_dce",
                "source_t1w", "source_t1w_postcon",
                "source_flair", "source_flair_postcon", "source_t2w",
                # PK
                "ktrans_map", "ve_map", "kep_map", "vp_map", "fp_map", "ps_map",
                "e_map",
                # Inference-specific uncertainty
                "ktrans_uncertainty", "ve_uncertainty", "vp_uncertainty",
                "fp_uncertainty", "ps_uncertainty",
                # CV
                "ktrans_cv", "ve_cv", "vp_cv", "fp_cv", "ps_cv",
                # CI95
                "ktrans_ci95_lower", "ktrans_ci95_upper",
                "ve_ci95_lower", "ve_ci95_upper",
                "vp_ci95_lower", "vp_ci95_upper",
                "fp_ci95_lower", "fp_ci95_upper",
                "ps_ci95_lower", "ps_ci95_upper",
                # Model-free
                "iauc60_map", "iauc90_map", "iauc120_map",
                "peak_enhancement_map", "time_to_peak_map", "ser_map",
                "wash_in_slope_map", "wash_out_slope_map",
                "bat_map", "curve_type_map",
                # QC + model-selection
                "tsnr_map", "qc_mask", "r2_map", "quality_flag_map",
                "model_selection_map", "delta_bic_map",
                # NAWM-norm
                "niauc60_map",
                # Smoothed dicts (variable cardinality)
                "smooth_pk_maps", "smooth_model_free_maps",
                # T1/B1 + masks (anat/)
                "t1_map", "t1_support_map", "b1_map", "brain_mask", "nawm_mask",
                # Motion + AIF
                "dce_moco", "dce_mean", "motion_params",
                "aif_curve", "aif_mask",
                # Structural underlays + xfm
                "t1w_in_dce", "t1w_postcon_in_dce", "t1w_subtraction_in_dce",
                "flair_in_dce", "flair_postcon_in_dce", "t2w_in_dce", "dce_to_t1w_xfm",
                # Lesion/ROI
                "lesion_mask_dce", "lesion_stats_json",
                "user_lesion_mask_dce", "user_lesion_stats_json",
                "roi_mask_dce", "roi_stats_json",
                # Sidecar drivers
                "qc_metrics", "provenance_dict", "preflight_warnings",
                "config_dict",
                # Figures
                "brain_mask_filmstrip", "motion_qc_plot",
                "aif_curve_plot", "aif_location_plot", "aif_candidates_plot",
                "registration_filmstrip", "b1_qc_plot", "t1_qc_plot",
                "preprocessing_qc_plot", "pk_modeling_qc_plot",
                "pk_param_maps_qc_plot", "model_selection_qc_plot",
                "dsc_dce_xval_plot",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.output_dir = output_dir
    inputnode.inputs.bids_dir = bids_dir
    inputnode.inputs.subject_id = subject_id
    inputnode.inputs.session_id = session_id or ""

    # --- Source-paths Merge for DCE ---
    perf_src_merge = pe.Node(niu.Merge(numinputs=1), name="perf_sources")
    workflow.connect(inputnode, "source_dce", perf_src_merge, "in1")

    _base_input_names = [
        "source_paths", "reference_voxel_source", "bids_dir",
        "subject", "session",
        "provenance_dict", "qc_metrics", "minimal",
        "pk_dependent",
    ]

    base_perf = pe.Node(
        niu.Function(
            input_names=_base_input_names,
            output_names=["meta_dict"],
            function=_build_dce_base,
        ),
        name="base_perf_meta",
    )
    base_perf.inputs.minimal = False
    base_perf.inputs.pk_dependent = True
    workflow.connect([
        (perf_src_merge, base_perf, [("out", "source_paths")]),
        (inputnode, base_perf, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            ("ktrans_map", "reference_voxel_source"),
            ("provenance_dict", "provenance_dict"),
            ("qc_metrics", "qc_metrics"),
        ]),
    ])

    base_perf_no_ktrans = pe.Node(
        niu.Function(
            input_names=_base_input_names,
            output_names=["meta_dict"],
            function=_build_dce_base,
        ),
        name="base_perf_no_ktrans_meta",
    )
    base_perf_no_ktrans.inputs.minimal = False
    base_perf_no_ktrans.inputs.pk_dependent = False
    workflow.connect([
        (perf_src_merge, base_perf_no_ktrans, [("out", "source_paths")]),
        (inputnode, base_perf_no_ktrans, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            ("ktrans_map", "reference_voxel_source"),
            ("provenance_dict", "provenance_dict"),
            ("qc_metrics", "qc_metrics"),
        ]),
    ])

    base_minimal = pe.Node(
        niu.Function(
            input_names=_base_input_names,
            output_names=["meta_dict"],
            function=_build_dce_base,
        ),
        name="base_minimal_meta",
    )
    base_minimal.inputs.minimal = True
    base_minimal.inputs.pk_dependent = False
    base_minimal.inputs.qc_metrics = None
    workflow.connect([
        (perf_src_merge, base_minimal, [("out", "source_paths")]),
        (inputnode, base_minimal, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            ("ktrans_map", "reference_voxel_source"),
            ("provenance_dict", "provenance_dict"),
        ]),
    ])

    sink_outputs: list[tuple[pe.Node, str]] = []

    # --- PK maps (flat-key cosmetic enrichment) ---
    for spec in _PK_MAPS:
        if spec.in_attr not in present:
            continue
        sink = _add_dce_pk_map_sink(
            workflow, inputnode, base_perf,
            name=spec.in_attr, in_attr=spec.in_attr,
            desc_value=spec.desc, suffix=spec.suffix,
            description=spec.description, units=spec.units,
            map_key=spec.map_key,
        )
        sink_outputs.append((sink, "out_file"))

    # --- Inference-specific relative uncertainty ---
    for spec in _CV_MAPS:
        if spec.in_attr not in present:
            continue
        if pk_model.lower() == "2cxm" and spec.in_attr == "ktrans_cv":
            continue
        parameter = spec.map_key.split("_", 1)[0]
        uncertainty_extra = pe.Node(
            niu.Function(
                input_names=["config_dict", "qc_metrics", "umap_key"],
                output_names=["extra"],
                function=_build_uncertainty_extra,
            ),
            name=f"{spec.in_attr}_extra",
        )
        uncertainty_extra.inputs.umap_key = spec.map_key
        workflow.connect(inputnode, "config_dict", uncertainty_extra, "config_dict")
        workflow.connect(inputnode, "qc_metrics", uncertainty_extra, "qc_metrics")
        is_nlls = inference.lower() == "nlls"
        desc_value = "dceRelativeStandardError" if is_nlls else "dcePosteriorCV"
        description = (
            f"{parameter} standard error divided by absolute estimate"
            if is_nlls
            else f"{parameter} posterior standard deviation divided by absolute estimate"
        )
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf,
            name=spec.in_attr, in_attr=spec.in_attr,
            desc_value=desc_value, suffix=spec.suffix,
            description=description, units=spec.units,
            extra=spec.extra, extra_node=uncertainty_extra, map_key=spec.map_key,
            bids_suffix=spec.bids_suffix,
        )
        sink_outputs.append((sink, "out_file"))

    # --- Model-free maps (no PK-input compartment metadata) ---
    for spec in _MODEL_FREE_MAPS:
        if spec.in_attr not in present:
            continue
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name=spec.in_attr, in_attr=spec.in_attr,
            desc_value=spec.desc, suffix=spec.suffix,
            description=spec.description, units=spec.units, extra=spec.extra,
            map_key=spec.map_key, bids_suffix=spec.bids_suffix,
        )
        sink_outputs.append((sink, "out_file"))

    # Categorical segmentation: explicit because its BIDS suffix is dseg.
    spec = _CURVE_TYPE_MAP
    if spec.in_attr in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name=spec.in_attr, in_attr=spec.in_attr,
            desc_value=spec.desc, suffix=spec.suffix,
            description=spec.description, units=spec.units, extra=spec.extra,
            map_key=spec.map_key, bids_suffix=spec.bids_suffix,
        )
        sink_outputs.append((sink, "out_file"))

    # --- QC maps (no PK-input compartment metadata) ---
    if "tsnr_map" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="tsnr_map", in_attr="tsnr_map",
            desc_value="dce", suffix="tsnr",
            description="Temporal signal-to-noise ratio", units="arbitrary",
            extra=None,
        )
        sink_outputs.append((sink, "out_file"))
    if "qc_mask" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="qc_mask", in_attr="qc_mask",
            desc_value="dceQc", suffix="",
            description="QC mask (1=good fit, 0=unfitted/poor)",
            units=None, extra={"Type": "ROI"},
            bids_suffix="mask",
        )
        sink_outputs.append((sink, "out_file"))
    if "r2_map" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf,
            name="r2_map", in_attr="r2_map",
            desc_value="dce", suffix="Rsquared",
            description=(
                "Coefficient of determination for the per-voxel PK fit"
            ),
            units="1", extra=None,
        )
        sink_outputs.append((sink, "out_file"))
    if "quality_flag_map" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="quality_flag_map", in_attr="quality_flag_map",
            desc_value="dceQualityFlag", suffix="",
            description=(
                "Quality flag map (0=unfitted, 1=good, 2=poor_convergence, "
                "3=high_uncertainty, 4=unreliable_high_kep)"
            ),
            units=None,
            extra={
                "LabelEncoding": {
                    "0": "unfitted", "1": "good", "2": "poor_convergence",
                    "3": "high_uncertainty", "4": "unreliable_high_kep",
                },
            },
            bids_suffix="dseg",
        )
        sink_outputs.append((sink, "out_file"))

    # --- Model selection + delta-BIC (custom builders use minimal base + qc) ---
    if "model_selection_map" in present:
        ms_meta = pe.Node(
            niu.Function(
                input_names=["base", "qc_metrics"],
                output_names=["meta_dict"],
                function=_build_model_selection_meta,
            ),
            name="model_selection_meta",
        )
        workflow.connect([
            (base_minimal, ms_meta, [("meta_dict", "base")]),
            (inputnode, ms_meta, [("qc_metrics", "qc_metrics")]),
        ])
        ms_sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype", "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="model_selection_sink",
        )
        ms_sink.inputs.desc = "dceModelSelection"
        ms_sink.inputs.suffix = ""
        ms_sink.inputs.datatype = "perf"
        ms_sink.inputs.bids_suffix = "dseg"
        workflow.connect([
            (inputnode, ms_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("model_selection_map", "in_file"),
            ]),
            (ms_meta, ms_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((ms_sink, "out_file"))

    if "delta_bic_map" in present:
        dbic_meta = pe.Node(
            niu.Function(
                input_names=["base", "qc_metrics"],
                output_names=["meta_dict"],
                function=_build_delta_bic_meta,
            ),
            name="delta_bic_meta",
        )
        workflow.connect([
            (base_minimal, dbic_meta, [("meta_dict", "base")]),
            (inputnode, dbic_meta, [("qc_metrics", "qc_metrics")]),
        ])
        dbic_sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype", "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="delta_bic_sink",
        )
        dbic_sink.inputs.desc = "dceDeltaBIC"
        dbic_sink.inputs.suffix = ""
        dbic_sink.inputs.datatype = "perf"
        dbic_sink.inputs.bids_suffix = "stat"
        workflow.connect([
            (inputnode, dbic_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("delta_bic_map", "in_file"),
            ]),
            (dbic_meta, dbic_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((dbic_sink, "out_file"))

    # --- NAWM-normalized maps ---
    if "niauc60_map" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="niauc60_map", in_attr="niauc60_map",
            desc_value="dceNawmNorm", suffix="IAUC60",
            description="IAUC at 60s normalized by the NAWM median",
            units="1", extra=None,
            map_key="niauc60",
        )
        sink_outputs.append((sink, "out_file"))

    # --- Motion-corrected 4D + mean ref + AIF mask (perf/) ---
    if "dce_moco" in present:
        # Preproc-4D sidecar is the canonical anchor for DCE: it carries
        # flat-prefixed cosmetic T1*/B1*/motion/signal QC keys AND the
        # QC* recommendation/warnings keys the HTML aggregator reads.
        # The verbatim resume sidecar (below) carries the producer-shape
        # qc_metrics envelope for downstream consumers that need the
        # full nested-dict shape.
        presence_warnings_dce = [
            warn for cond, warn in (
                (b1_map_expected and "b1_map" not in present,
                 "B1+ transmit field map not available"),
                (t1_map_expected and "t1_map" not in present,
                 "T1 relaxation time map not available"),
                ("brain_mask" not in present,
                 "Brain mask in DCE space not available"),
                ("nawm_mask" not in present,
                 "NAWM mask in DCE space not available"),
                ("lesion_mask_dce" not in present,
                 "Lesion mask in DCE space not available"),
                ("dce_to_t1w_xfm" not in present,
                 "DCE to T1w registration transform not available"),
            ) if cond
        ]
        moco_meta = pe.Node(
            niu.Function(
                input_names=[
                    "base", "description", "qc_metrics",
                    "provenance_dict", "preflight_warnings",
                    "presence_warnings", "config_dict",
                ],
                output_names=["meta_dict"],
                function=_build_dce_preproc_meta,
            ),
            name="dce_moco_meta",
        )
        moco_meta.inputs.description = "Motion-corrected 4D DCE time series"
        moco_meta.inputs.presence_warnings = presence_warnings_dce
        workflow.connect([
            (base_perf_no_ktrans, moco_meta, [("meta_dict", "base")]),
            (inputnode, moco_meta, [
                ("qc_metrics", "qc_metrics"),
                ("provenance_dict", "provenance_dict"),
                ("preflight_warnings", "preflight_warnings"),
                ("config_dict", "config_dict"),
            ]),
        ])
        moco_sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype", "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="dce_moco_sink",
        )
        moco_sink.inputs.desc = "dcePreproc"
        moco_sink.inputs.suffix = ""
        moco_sink.inputs.datatype = "perf"
        moco_sink.inputs.bids_suffix = "dce"
        workflow.connect([
            (inputnode, moco_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("dce_moco", "in_file"),
            ]),
            (moco_meta, moco_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((moco_sink, "out_file"))

    if "dce_mean" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="dce_mean", in_attr="dce_mean",
            desc_value="dceMean", suffix="",
            description="Mean DCE reference image",
            units=None, extra=None,
            bids_suffix="stat",
        )
        sink_outputs.append((sink, "out_file"))
    if "aif_mask" in present:
        meta = pe.Node(
            niu.Function(
                input_names=["base", "description", "qc_metrics"],
                output_names=["meta_dict"],
                function=_build_aif_mask_meta,
            ),
            name="aif_mask_meta",
        )
        meta.inputs.description = "AIF detection mask in DCE space"
        workflow.connect([
            (base_minimal, meta, [("meta_dict", "base")]),
            (inputnode, meta, [("qc_metrics", "qc_metrics")]),
        ])
        sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype", "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="aif_mask_sink",
        )
        sink.inputs.desc = "dceAif"
        sink.inputs.suffix = ""
        sink.inputs.datatype = "perf"
        sink.inputs.bids_suffix = "mask"
        workflow.connect([
            (inputnode, sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("aif_mask", "in_file"),
            ]),
            (meta, sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((sink, "out_file"))

    # --- T1map + B1+map (anat/) ---
    if "t1_map" in present:
        sink = _add_anat_image_sink(
            workflow, inputnode, base_minimal,
            name="t1_map", in_attr="t1_map", label="T1map",
            description="T1 relaxation time map (ms)",
            units="ms",
        )
        sink_outputs.append((sink, "out_file"))
    if "t1_support_map" in present:
        sink = _add_anat_image_sink(
            workflow, inputnode, base_minimal,
            name="t1_support_map", in_attr="t1_support_map", label="dseg",
            desc="t1Support",
            description="T1 support provenance labels: 0 outside DCE analysis "
            "domain, 1 directly measured, 2 locally interpolated, 3 unavailable, "
            "4 CSF excluded, 5 uniform assumption",
            extra={
                "LabelMap": {
                    "0": "Outside DCE analysis domain",
                    "1": "Directly measured T1",
                    "2": "Locally interpolated T1",
                    "3": "T1 unavailable in DCE analysis domain",
                    "4": "CSF excluded",
                    "5": "Uniform T1 assumption",
                },
            },
        )
        sink_outputs.append((sink, "out_file"))
    if "b1_map" in present:
        meta = pe.Node(
            niu.Function(
                input_names=["base", "b1_source_paths", "bids_dir"],
                output_names=["meta_dict"],
                function=_build_tb1_meta,
            ),
            name="b1_map_meta",
        )
        meta.inputs.b1_source_paths = list(b1_source_files or [])
        meta.inputs.bids_dir = bids_dir
        workflow.connect(base_minimal, "meta_dict", meta, "base")
        sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session", "meta_dict",
                ],
                output_names=["out_file"],
                function=_copy_tb1map,
            ),
            name="b1_map_sink",
        )
        workflow.connect([
            (inputnode, sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("b1_map", "in_file"),
            ]),
            (meta, sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((sink, "out_file"))

    # --- Anat-side masks ---
    for mask_spec in _ANAT_MASKS:
        if mask_spec.in_attr not in present:
            continue
        sink = _add_anat_mask_sink(
            workflow, inputnode, base_minimal,
            name=mask_spec.name, in_attr=mask_spec.in_attr,
            desc_value=mask_spec.desc, description=mask_spec.description,
            skull_stripped=mask_spec.skull_stripped,
            mask_type=mask_spec.mask_type,
        )
        sink_outputs.append((sink, "out_file"))

    # --- Structural underlays (T1w/FLAIR/T2w in DCE) ---
    for anat_spec in _STRUCTURAL_UNDERLAYS:
        if anat_spec.in_attr not in present:
            continue
        sink = _add_anat_image_sink(
            workflow, inputnode, base_minimal,
            name=anat_spec.in_attr, in_attr=anat_spec.in_attr,
            label=anat_spec.label, description=anat_spec.description,
            desc=anat_spec.desc, ce_label=anat_spec.ceagent,
            source_attr=anat_spec.source_attr,
            dismiss_entities=(),
        )
        sink_outputs.append((sink, "out_file"))

    if "dce_to_t1w_xfm" in present:
        xfm_sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session", "source_file",
                ],
                output_names=["out_file"],
                function=_copy_dce_xfm,
            ),
            name="dce_to_t1w_xfm_sink",
        )
        workflow.connect([
            (inputnode, xfm_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("dce_to_t1w_xfm", "in_file"),
                ("source_dce", "source_file"),
            ]),
        ])
        sink_outputs.append((xfm_sink, "out_file"))

    # --- Smoothed maps (variable cardinality → MapNode) ---
    if has_smoothing:
        splitter = pe.Node(
            niu.Function(
                input_names=[
                    "smooth_pk", "smooth_mf", "pk_units", "mf_units",
                ],
                output_names=[
                    "in_files", "descs", "fwhms", "units_list",
                    "desc_values", "suffixes", "map_keys",
                ],
                function=_split_dce_smoothed,
            ),
            name="smoothed_split",
        )
        splitter.inputs.pk_units = dict(_PK_UNITS)
        splitter.inputs.mf_units = dict(_MF_UNITS)
        workflow.connect([
            (inputnode, splitter, [
                ("smooth_pk_maps", "smooth_pk"),
                ("smooth_model_free_maps", "smooth_mf"),
            ]),
        ])

        sm_meta = pe.MapNode(
            niu.Function(
                input_names=["base", "description", "fwhm_mm", "units",
                             "map_key", "inference"],
                output_names=["meta_dict"],
                function=_build_dce_smoothed_meta,
            ),
            iterfield=["description", "fwhm_mm", "units", "map_key"],
            name="smoothed_meta",
        )
        sm_meta.inputs.inference = inference
        workflow.connect([
            (base_perf, sm_meta, [("meta_dict", "base")]),
            (splitter, sm_meta, [
                ("descs", "description"),
                ("fwhms", "fwhm_mm"),
                ("units_list", "units"),
                ("map_keys", "map_key"),
            ]),
        ])

        sm_sink = pe.MapNode(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype", "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            iterfield=["in_file", "desc", "suffix", "meta_dict"],
            name="smoothed_sink",
        )
        sm_sink.inputs.datatype = "perf"
        sm_sink.inputs.bids_suffix = "stat"
        workflow.connect([
            (inputnode, sm_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
            ]),
            (splitter, sm_sink, [
                ("in_files", "in_file"),
                ("desc_values", "desc"),
                ("suffixes", "suffix"),
            ]),
            (sm_meta, sm_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((sm_sink, "out_file"))

    # --- DCE figures writer (figures, AIF curve, confounds TSV/JSON) ---
    figures_writer = pe.Node(DCEFiguresWriter(), name="figures_writer")
    figures_writer.inputs.motion_applied = motion_correct
    workflow.connect([
        (inputnode, figures_writer, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject_id"),
            ("session_id", "session_id"),
            ("qc_metrics", "qc_metrics"),
            ("aif_curve", "aif_curve"),
            ("motion_params", "motion_params"),
            ("bids_dir", "bids_dir"),
            ("source_dce", "source_dce"),
            ("brain_mask_filmstrip", "brain_mask_filmstrip"),
            ("motion_qc_plot", "motion_qc_plot"),
            ("aif_curve_plot", "aif_curve_plot"),
            ("aif_location_plot", "aif_location_plot"),
            ("aif_candidates_plot", "aif_candidates_plot"),
            ("registration_filmstrip", "registration_filmstrip"),
            ("b1_qc_plot", "b1_qc_plot"),
            ("t1_qc_plot", "t1_qc_plot"),
            ("preprocessing_qc_plot", "preprocessing_qc_plot"),
            ("pk_modeling_qc_plot", "pk_modeling_qc_plot"),
            ("pk_param_maps_qc_plot", "pk_param_maps_qc_plot"),
            ("model_selection_qc_plot", "model_selection_qc_plot"),
            ("dsc_dce_xval_plot", "dsc_dce_xval_plot"),
        ]),
    ])

    # --- dataset_description.json barrier ---
    merge = pe.Node(
        niu.Merge(numinputs=len(sink_outputs) + 1), name="dd_merge",
    )
    for i, (sink, attr) in enumerate(sink_outputs, start=1):
        workflow.connect(sink, attr, merge, f"in{i}")
    workflow.connect(
        figures_writer, "aif_curve_txt", merge, f"in{len(sink_outputs) + 1}",
    )

    dd_node = pe.Node(
        niu.Function(
            input_names=["output_dir", "bids_dir", "_trigger"],
            output_names=["dd_path"],
            function=write_dataset_description_node,
        ),
        name="dataset_description",
    )

    # These nodes write outside their Nipype working directories. Function
    # outputs are untyped strings, so Nipype cannot detect that a derivative
    # was deleted while the node cache remained intact. Always execute only
    # the external writer boundary; all scientific producers stay cacheable.
    for sink, _ in sink_outputs:
        sink._interface._always_run = True
    figures_writer._interface._always_run = True
    dd_node._interface._always_run = True
    workflow.connect([
        (inputnode, dd_node, [
            ("output_dir", "output_dir"),
            ("bids_dir", "bids_dir"),
        ]),
        (merge, dd_node, [("out", "_trigger")]),
    ])

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["done", "dataset_description_done"],
        ),
        name="outputnode",
    )
    workflow.connect([
        (dd_node, outputnode, [
            ("dd_path", "done"),
            ("dd_path", "dataset_description_done"),
        ]),
    ])

    return workflow
