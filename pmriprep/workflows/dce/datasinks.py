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


# PK parameter maps. Model name lives in JSON sidecar (PKModeling.model),
# not in the filename (no `model-` entity in BIDS schema).
# (input_attr, suffix, description, units)
_PK_MAPS: tuple[tuple[str, str, str, str], ...] = (
    ("ktrans_map", "Ktrans", "Volume transfer constant (1/min)", "1/min"),
    ("ve_map", "ve", "Extravascular extracellular space volume fraction", "1"),
    ("kep_map", "kep", "Efflux rate constant (1/min)", "1/min"),
    ("vp_map", "vp", "Plasma volume fraction", "1"),
)


# Bayesian uncertainty (posterior std) — per-PK-map
# (input_attr, suffix-name, description)
_UNCERTAINTY_MAPS: tuple[tuple[str, str, str], ...] = (
    ("ktrans_uncertainty", "Ktrans", "Ktrans uncertainty (posterior std)"),
    ("ve_uncertainty", "ve", "ve uncertainty (posterior std)"),
    ("vp_uncertainty", "vp", "vp uncertainty (posterior std)"),
    ("fp_uncertainty", "Fp", "Fp uncertainty (posterior std, 2CXM only)"),
    ("ps_uncertainty", "PS", "PS uncertainty (posterior std, 2CXM only)"),
)


_CV_MAPS: tuple[tuple[str, str, str], ...] = (
    ("ktrans_cv", "Ktrans", "Ktrans coefficient of variation (std/mean)"),
    ("ve_cv", "ve", "ve coefficient of variation (std/mean)"),
    ("vp_cv", "vp", "vp coefficient of variation (std/mean)"),
)


# 95% CI maps — pairs of (lower, upper)
_CI95_MAPS: tuple[tuple[str, str, str, str], ...] = (
    ("ktrans_ci95_lower", "Ktrans", "lower",
     "Ktrans 95% credible interval lower bound"),
    ("ktrans_ci95_upper", "Ktrans", "upper",
     "Ktrans 95% credible interval upper bound"),
    ("ve_ci95_lower", "ve", "lower", "ve 95% credible interval lower bound"),
    ("ve_ci95_upper", "ve", "upper", "ve 95% credible interval upper bound"),
    ("vp_ci95_lower", "vp", "lower", "vp 95% credible interval lower bound"),
    ("vp_ci95_upper", "vp", "upper", "vp 95% credible interval upper bound"),
)


# Model-free maps (no model entity). Each entry:
# (input_attr, suffix, description, units, extra_dict_or_None)
_MODEL_FREE_MAPS: tuple[tuple[str, str, str, str, dict | None], ...] = (
    ("iauc60_map", "IAUC60", "Initial Area Under Curve at 60s (mM*min)", "mM*min", None),
    ("iauc90_map", "IAUC90", "Initial Area Under Curve at 90s (mM*min)", "mM*min", None),
    ("iauc120_map", "IAUC120", "Initial Area Under Curve at 120s (mM*min)", "mM*min", None),
    ("peak_enhancement_map", "peakEnhancement", "Peak enhancement percentage", "%", None),
    ("time_to_peak_map", "ttp", "Time to peak (seconds)", "s", None),
    ("ser_map", "SER", "Signal Enhancement Ratio (washout characterization)", "1", None),
    ("wash_in_slope_map", "washInSlope",
     "Maximum upslope of concentration during wash-in (mM/min)", "mM/min",
     {"SmoothingMethod": "Savitzky-Golay", "SmoothingWindow": 5,
      "SmoothingPolyorder": 2, "InputCurve": "concentration"}),
    ("wash_out_slope_map", "washOutSlope",
     "Linear washout slope (mM/min, negative=washout)", "mM/min",
     {"PostPeakStartSeconds": 30.0, "PostPeakEndSeconds": 120.0,
      "MinPoints": 4, "RegressionMethod": "ordinary_least_squares"}),
    ("bat_map", "BAT", "Bolus arrival time (seconds)", "s",
     {"DetectionMethod": "threshold_then_gradient_fallback",
      "ConfidenceThreshold": "high"}),
    ("curve_type_map", "curveType",
     "Curve type classification (1=persistent, 2=plateau, 3=washout)",
     "categorical",
     {"PersistentThreshold": 0.9, "WashoutThreshold": 1.1,
      "ValueMapping": {"0": "unclassified", "1": "persistent",
                       "2": "plateau", "3": "washout"}}),
)


_PK_UNITS = {
    "ktrans": "1/min", "ve": "1", "kep": "1/min", "vp": "1",
    "fp": "mL/min/100mL", "ps": "mL/min/100mL",
}
_MF_UNITS = {
    "iauc60": "mM*min", "iauc90": "mM*min", "iauc120": "mM*min",
    "peak_enhancement": "%", "ser": "1",
    "wash_in_slope": "mM/min", "wash_out_slope": "mM/min",
}

# curveType is a discrete segmentation (dseg), not a continuous stat.
_BIDS_SUFFIX_OVERRIDE: dict[str, str] = {"curveType": "dseg"}

# BIDS-correct suffixes for smoothed-map sinks live inline in
# ``_split_dce_smoothed`` (nipype reconstructs Function helpers in an
# empty namespace, so the lookup tables can't be shared via module
# globals). The unsmoothed equivalents are encoded in ``_PK_MAPS`` /
# ``_MODEL_FREE_MAPS`` above; the inline tables mirror those.


# Inputs always treated as present (the four PK maps have ktrans_map mandatory
# in the legacy writer; everything else is gated by present_inputs).
_ALWAYS_PRESENT_INPUTS = frozenset({"ktrans_map"})


# ---------------------------------------------------------------------------
# Function-node wrappers
# ---------------------------------------------------------------------------


def _build_dce_base(  # type: ignore[no-untyped-def]
    source_paths,
    reference_voxel_source,
    bids_dir,
    subject,
    session,
    provenance_dict=None,
    qc_metrics=None,
    minimal=False,
    ktrans_convention=True,
):
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
        ktrans_convention=ktrans_convention,
    )


def _build_dce_meta(  # type: ignore[no-untyped-def]
    base, description, units=None, extra=None,
    map_key="", lesion_stats_json=None, user_lesion_stats_json=None,
    roi_stats_json=None,
):
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


def _build_dce_preproc_meta(  # type: ignore[no-untyped-def]
    base, description, qc_metrics=None,
    provenance_dict=None,
    preflight_warnings=None,
    presence_warnings=None,
    config_dict=None,
):
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
    return sanitize_for_json(out)



def _build_dce_pk_map_meta(  # type: ignore[no-untyped-def]
    base, description, units=None, qc_metrics=None, map_key="",
    lesion_stats_json=None, user_lesion_stats_json=None,
    roi_stats_json=None,
):
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


def _build_dce_smoothed_meta(  # type: ignore[no-untyped-def]
    base, description, fwhm_mm, units=None, map_key="",
):
    from pmriprep.utils.aif_provenance import stamp_aif_dependency

    out = dict(base)
    out["Description"] = description
    out["SmoothingFWHM"] = (
        float(fwhm_mm) if str(fwhm_mm).replace(".", "").isdigit() else fwhm_mm
    )
    out["SmoothingType"] = "Gaussian"
    out["SmoothingMasked"] = True
    out["IntendedFor"] = "visualization"
    out["Note"] = "Use non-smoothed maps for quantitative ROI analysis"
    if units is not None:
        out["Units"] = units
    if map_key:
        stamp_aif_dependency(out, map_key, modality="DCE")
    return out


def _build_model_selection_meta(  # type: ignore[no-untyped-def]
    base, qc_metrics=None,
):
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
        regions = (qc_metrics.get("pk_modeling") or {}).get(
            "model_selection_regions",
        )
        if regions:
            out["ModelDistribution"] = regions
    return out


def _build_delta_bic_meta(  # type: ignore[no-untyped-def]
    base, qc_metrics=None,
):
    from pmriprep.utils.aif_provenance import stamp_aif_dependency

    out = dict(base)
    out["Description"] = (
        "BIC confidence: runner-up minus winner. "
        ">2 = positive, >6 = strong, >10 = very strong "
        "(Kass & Raftery 1995)"
    )
    stamp_aif_dependency(out, "delta_bic", modality="DCE")
    if isinstance(qc_metrics, dict):
        pk = qc_metrics.get("pk_modeling") or {}
        if pk.get("r2_median") is not None:
            out["R2Median"] = pk["r2_median"]
        if pk.get("convergence_rate") is not None:
            out["ConvergenceRate"] = pk["convergence_rate"]
    return out


def _copy_perf_map(  # type: ignore[no-untyped-def]
    in_file, output_dir, subject, session, desc, suffix, meta_dict,
    datatype="perf", model_entity="", bids_suffix="stat", space="",
):
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

    # Defensive no-op when upstream conditional producer returned no output
    # (e.g., NAWM normalization had < 10 NAWM voxels, optional transform
    # absent). The producer's None-return is the canonical "I had nothing
    # to write" signal — see _normalize_by_nawm.
    # Sentinels: isdefined(Undefined) == False, BUT isdefined(None) == True
    # (Nipype gotcha) and "" is the documented Function-node-returns-empty
    # convention — all three need explicit handling.
    if not isdefined(in_file) or in_file is None or in_file == "":
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

    prefix = f"sub-{subject}"
    if session:
        prefix += f"_ses-{session}"
    parts = [prefix]
    if space:
        parts.append(f"space-{space}")
    if model_entity:
        parts.append(model_entity)
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


def _copy_b1plusmap(  # type: ignore[no-untyped-def]
    in_file, output_dir, subject, session, meta_dict,
):
    """Copy a B1plusmap NIfTI + sidecar to anat/.

    Bypasses ``PMRIPrepDataSink`` because ``B1plusmap`` is not in the
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
    if not isdefined(in_file) or in_file is None or in_file == "":
        logging.getLogger("nipype.workflow").warning(
            "Skipping _copy_b1plusmap sink: upstream produced no output "
            "(in_file=%r)", in_file,
        )
        return ""

    out_root = Path(output_dir) / "pmriprep" / f"sub-{subject}"
    if session:
        out_root = out_root / f"ses-{session}"
    out_root = out_root / "anat"
    out_root.mkdir(parents=True, exist_ok=True)

    prefix = f"sub-{subject}"
    if session:
        prefix += f"_ses-{session}"
    stem = f"{prefix}_space-dceref_B1plusmap"
    nii_out = out_root / f"{stem}.nii.gz"
    json_out = out_root / f"{stem}.json"

    shutil.copy2(in_file, nii_out)
    from pmriprep.utils.sidecar_builders import dump_sidecar_json

    json_out.write_text(dump_sidecar_json(meta_dict))
    return str(nii_out)


def _copy_dce_xfm(  # type: ignore[no-untyped-def]
    in_file, output_dir, subject, session,
):
    """Copy DCE-to-T1w transform .h5 into the derivatives anat/."""
    import logging
    import shutil
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    # Defensive no-op when upstream registration produced no transform
    # (rare, but possible when a subject has DCE imaging but registration
    # to T1w failed). See _copy_perf_map for sentinel-handling rationale.
    if not isdefined(in_file) or in_file is None or in_file == "":
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

    prefix = f"sub-{subject}"
    if session:
        prefix += f"_ses-{session}"
    fname = f"{prefix}_from-dceref_to-T1w_mode-image_xfm.h5"
    out_path = out_root / fname
    shutil.copy2(in_file, out_path)
    return str(out_path)


def _split_dce_smoothed(  # type: ignore[no-untyped-def]
    smooth_pk=None, smooth_mf=None, smooth_uncertainty=None,
    pk_units=None, mf_units=None,
):
    """Flatten the three smoothed-map dicts into per-output tuples."""
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

    def _extract_fwhm(src_path):  # type: ignore[no-untyped-def]
        m = _re.search(r"_fwhm(\d+(?:\.\d+)?)", str(src_path))
        if m:
            return f"Sm{m.group(1)}", m.group(1)
        return "SmN", "N"

    in_files = []
    descs = []
    fwhms = []
    units_list = []
    desc_values = []
    suffixes = []
    use_model = []
    map_keys = []

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
            use_model.append(True)
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
            use_model.append(False)
            map_keys.append(name)

    uncertainty_map_key = {
        "Ktrans_std": "ktrans_uncertainty",
        "ve_std": "ve_uncertainty",
        "vp_std": "vp_uncertainty",
        "Fp_std": "fp_uncertainty",
        "PS_std": "ps_uncertainty",
    }
    if isinstance(smooth_uncertainty, dict):
        for name, src in smooth_uncertainty.items():
            if not (src and Path(src).exists()):
                continue
            fwhm_str, fwhm_mm = _extract_fwhm(src)
            param = name.replace("_std", "")
            in_files.append(str(src))
            descs.append(f"Smoothed {name} uncertainty map")
            fwhms.append(fwhm_mm)
            units_list.append(pk_units.get(param, ""))
            desc_values.append(f"dceUncertainty{fwhm_str}")
            suffixes.append(param)
            use_model.append(True)
            map_keys.append(uncertainty_map_key.get(name, ""))

    return in_files, descs, fwhms, units_list, desc_values, suffixes, use_model, map_keys


def _model_entity_per_item(use_model_flags, model_entity):  # type: ignore[no-untyped-def]
    return [model_entity if flag else "" for flag in use_model_flags]


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
    model_entity: str = "",
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
    meta.inputs.extra = extra
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
                "desc", "suffix", "meta_dict", "datatype", "model_entity",
                "bids_suffix",
            ],
            output_names=["out_file"],
            function=_copy_perf_map,
        ),
        name=f"{name}_sink",
    )
    sink.inputs.desc = desc_value
    sink.inputs.suffix = suffix
    sink.inputs.datatype = "perf"
    sink.inputs.model_entity = model_entity
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


_PK_MAP_KEY: dict[str, str] = {
    "ktrans_map": "ktrans",
    "ve_map": "ve",
    "kep_map": "kep",
    "vp_map": "vp",
}


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

_MF_AIF_MAP_KEY: dict[str, str] = {
    "iauc60_map": "iauc60",
    "iauc90_map": "iauc90",
    "iauc120_map": "iauc120",
    "peak_enhancement_map": "peak_enhancement",
    "time_to_peak_map": "ttp",
    "ser_map": "ser",
    "wash_in_slope_map": "wash_in_slope",
    "wash_out_slope_map": "wash_out_slope",
    "bat_map": "bat",
    "curve_type_map": "curve_type",
}


# NAWM-normalized DCE maps that carry LesionStats.
_NAWMNORM_LESIONSTATS_MAP_KEY: dict[str, str] = {
    "nktrans_map": "nktrans",
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
    model_entity: str = "",
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
                "desc", "suffix", "meta_dict", "datatype", "model_entity",
                "bids_suffix",
            ],
            output_names=["out_file"],
            function=_copy_perf_map,
        ),
        name=f"{name}_sink",
    )
    sink.inputs.desc = desc_value
    sink.inputs.suffix = suffix
    sink.inputs.datatype = "perf"
    sink.inputs.model_entity = model_entity
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
    skull_stripped: bool = False,
    units: Optional[str] = None,
    source_attr: str = "source_dce",
    dismiss_entities: tuple = ("acquisition",),
) -> pe.Node:
    """Wire an anat-side DCE sink (T1w/FLAIR/T2w/T1map in DCE space).

    Mirrors DSC's ``_add_anat_sink`` — uses ``PMRIPrepDataSink`` (niworkflows
    ``DerivativesDataSink`` subclass) for BIDS-canonical anat suffixes. The
    space-dceref entity correctly indicates cross-modality registration to DCE
    acquisition geometry.

    ``dismiss_entities`` strips entities the niworkflows ``DerivativesDataSink``
    would otherwise inherit from the source DCE file. Default drops
    ``acquisition`` because the DCE source's acquisition (e.g. VIBE) describes
    the spatial-reference grid, not the T1w/FLAIR/T2w content being resampled.

    No current caller overrides this default — the kwarg is plumbed defensively
    for symmetry with DSC's ``_add_anat_sink`` (which DOES have call sites
    needing ``("acquisition", "ceagent")`` for pre-con-derived masks). DCE
    masks go through ``_add_anat_mask_sink`` → ``_copy_perf_map`` (manual
    filename construction; no entity inheritance), so the DCE side has no
    pre-con-mask sinks routed through this helper today.
    """
    meta = pe.Node(
        niu.Function(
            input_names=["base", "description", "units", "extra"],
            output_names=["meta_dict"],
            function=_build_dce_meta,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.units = units
    meta.inputs.extra = {"SkullStripped": True} if skull_stripped else None
    workflow.connect(base_minimal_node, "meta_dict", meta, "base")

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
    meta.inputs.extra = {"SkullStripped": True} if skull_stripped else None
    workflow.connect(base_minimal_node, "meta_dict", meta, "base")

    sink = pe.Node(
        niu.Function(
            input_names=[
                "in_file", "output_dir", "subject", "session",
                "desc", "suffix", "meta_dict", "datatype", "model_entity",
                "bids_suffix", "space",
            ],
            output_names=["out_file"],
            function=_copy_perf_map,
        ),
        name=f"{name}_sink",
    )
    sink.inputs.desc = desc_value
    sink.inputs.suffix = ""
    sink.inputs.datatype = "anat"
    sink.inputs.model_entity = ""
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
    present_inputs: Optional[set[str]] = None,
    has_smoothing: bool = False,
    motion_correct: bool = True,
    name: str = "dce_datasinks_wf",
) -> pe.Workflow:
    """Build the DCE-pipeline DerivativesDataSink chain.

    ``pk_model`` is the resolved (non-"auto") model name; the
    ``model-{pk_model}`` entity is added to PK / uncertainty / CV / CI95 /
    R²/qualityFlag/modelSelection/deltaBIC/normalized
    sinks. Model-free maps (IAUC, peakEnhancement, ttp, SER, washIn/Out,
    BAT, curveType, tsnr, qcMask, nIAUC60) drop the model entity.
    """
    if present_inputs is None:
        present_inputs = set()
    present = set(_ALWAYS_PRESENT_INPUTS) | set(present_inputs)
    # `model-` is not a BIDS entity; model name lives in the JSON sidecar
    # (PKModeling.model). Empty string here propagates through all callers.
    model_entity = ""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "output_dir", "bids_dir", "subject_id", "session_id",
                "source_dce",
                # PK
                "ktrans_map", "ve_map", "kep_map", "vp_map",
                # Bayesian uncertainty
                "ktrans_uncertainty", "ve_uncertainty", "vp_uncertainty",
                "fp_uncertainty", "ps_uncertainty",
                # CV
                "ktrans_cv", "ve_cv", "vp_cv",
                # CI95
                "ktrans_ci95_lower", "ktrans_ci95_upper",
                "ve_ci95_lower", "ve_ci95_upper",
                "vp_ci95_lower", "vp_ci95_upper",
                # Model-free
                "iauc60_map", "iauc90_map", "iauc120_map",
                "peak_enhancement_map", "time_to_peak_map", "ser_map",
                "wash_in_slope_map", "wash_out_slope_map",
                "bat_map", "curve_type_map",
                # QC + model-selection
                "tsnr_map", "qc_mask", "r2_map", "quality_flag_map",
                "model_selection_map", "delta_bic_map",
                # NAWM-norm
                "nktrans_map", "niauc60_map",
                # Smoothed dicts (variable cardinality)
                "smooth_pk_maps", "smooth_model_free_maps",
                "smooth_uncertainty_maps",
                # T1/B1 + masks (anat/)
                "t1_map", "b1_map", "brain_mask", "nawm_mask",
                # Motion + AIF
                "dce_moco", "dce_mean", "motion_params",
                "aif_curve", "aif_mask",
                # Structural underlays + xfm
                "t1w_in_dce", "t1w_postcon_in_dce",
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
        "ktrans_convention",
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
    base_perf.inputs.ktrans_convention = True
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
    base_perf_no_ktrans.inputs.ktrans_convention = False
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
    base_minimal.inputs.ktrans_convention = False
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

    # --- PK maps (carry model entity + flat-key cosmetic enrichment) ---
    for in_attr, suffix, description, units in _PK_MAPS:
        if in_attr not in present:
            continue
        sink = _add_dce_pk_map_sink(
            workflow, inputnode, base_perf,
            name=in_attr, in_attr=in_attr,
            desc_value="dce", suffix=suffix,
            description=description, units=units,
            map_key=_PK_MAP_KEY.get(in_attr, ""),
            model_entity=model_entity,
        )
        sink_outputs.append((sink, "out_file"))

    # --- Bayesian uncertainty / CV / CI95 (model entity) ---
    for in_attr, suffix, description in _UNCERTAINTY_MAPS:
        if in_attr not in present:
            continue
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf,
            name=in_attr, in_attr=in_attr,
            desc_value="dceUncertainty", suffix=suffix,
            description=description, units=None, extra=None,
            model_entity=model_entity,
            map_key=in_attr,
        )
        sink_outputs.append((sink, "out_file"))

    for in_attr, suffix, description in _CV_MAPS:
        if in_attr not in present:
            continue
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf,
            name=in_attr, in_attr=in_attr,
            desc_value="dceCV", suffix=suffix,
            description=description, units=None, extra=None,
            model_entity=model_entity,
            map_key=in_attr,
        )
        sink_outputs.append((sink, "out_file"))

    for in_attr, suffix, bound, description in _CI95_MAPS:
        if in_attr not in present:
            continue
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf,
            name=in_attr, in_attr=in_attr,
            desc_value=f"dceCI95{bound}", suffix=suffix,
            description=description, units=None, extra=None,
            model_entity=model_entity,
            map_key=in_attr,
        )
        sink_outputs.append((sink, "out_file"))

    # --- Model-free maps (no model entity, no KtransConvention) ---
    for in_attr, suffix, description, units, extra in _MODEL_FREE_MAPS:
        if in_attr not in present:
            continue
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name=in_attr, in_attr=in_attr,
            desc_value="dce", suffix=suffix,
            description=description, units=units, extra=extra,
            model_entity="",
            map_key=_MF_AIF_MAP_KEY.get(in_attr, ""),
            bids_suffix=_BIDS_SUFFIX_OVERRIDE.get(suffix, "stat"),
        )
        sink_outputs.append((sink, "out_file"))

    # --- QC maps (no KtransConvention) ---
    if "tsnr_map" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="tsnr_map", in_attr="tsnr_map",
            desc_value="dce", suffix="tsnr",
            description="Temporal signal-to-noise ratio", units="arbitrary",
            extra=None, model_entity="",
        )
        sink_outputs.append((sink, "out_file"))
    if "qc_mask" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="qc_mask", in_attr="qc_mask",
            desc_value="dceQc", suffix="",
            description="QC mask (1=good fit, 0=unfitted/poor)",
            units=None, extra=None, model_entity="",
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
            units=None, extra=None,
            model_entity=model_entity,
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
            units=None, extra=None, model_entity=model_entity,
            bids_suffix="mask",
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
                    "desc", "suffix", "meta_dict", "datatype", "model_entity",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="model_selection_sink",
        )
        ms_sink.inputs.desc = "dceModelSelection"
        ms_sink.inputs.suffix = ""
        ms_sink.inputs.datatype = "perf"
        ms_sink.inputs.model_entity = model_entity
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
                    "desc", "suffix", "meta_dict", "datatype", "model_entity",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="delta_bic_sink",
        )
        dbic_sink.inputs.desc = "dceDeltaBIC"
        dbic_sink.inputs.suffix = ""
        dbic_sink.inputs.datatype = "perf"
        dbic_sink.inputs.model_entity = model_entity
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
    if "nktrans_map" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf,
            name="nktrans_map", in_attr="nktrans_map",
            desc_value="dceNawmNorm", suffix="Ktrans",
            description="Ktrans normalized by NAWM (normal-appearing white matter) mean",
            units="1", extra=None, model_entity=model_entity,
            map_key="nktrans",
        )
        sink_outputs.append((sink, "out_file"))
    if "niauc60_map" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="niauc60_map", in_attr="niauc60_map",
            desc_value="dceNawmNorm", suffix="IAUC60",
            description="IAUC at 60s normalized by NAWM (normal-appearing white matter) mean",
            units="1", extra=None, model_entity="",
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
                ("b1_map" not in present,
                 "B1+ transmit field map not available"),
                ("t1_map" not in present,
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
            (base_perf, moco_meta, [("meta_dict", "base")]),
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
                    "desc", "suffix", "meta_dict", "datatype", "model_entity",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="dce_moco_sink",
        )
        moco_sink.inputs.desc = "dcePreproc"
        moco_sink.inputs.suffix = ""
        moco_sink.inputs.datatype = "perf"
        moco_sink.inputs.model_entity = ""
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
            units=None, extra=None, model_entity="",
            bids_suffix="stat",
        )
        sink_outputs.append((sink, "out_file"))
    if "aif_mask" in present:
        sink = _add_perf_map_sink(
            workflow, inputnode, base_perf_no_ktrans,
            name="aif_mask", in_attr="aif_mask",
            desc_value="dceAif", suffix="",
            description="AIF detection mask in DCE space",
            units=None, extra=None, model_entity="",
            bids_suffix="mask",
        )
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
    if "b1_map" in present:
        meta = pe.Node(
            niu.Function(
                input_names=["base", "description", "units", "extra"],
                output_names=["meta_dict"],
                function=_build_dce_meta,
            ),
            name="b1_map_meta",
        )
        meta.inputs.description = (
            "B1+ transmit field map (dimensionless ratio, 1.0 = nominal flip angle)"
        )
        meta.inputs.units = "1"
        meta.inputs.extra = None
        workflow.connect(base_minimal, "meta_dict", meta, "base")
        sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session", "meta_dict",
                ],
                output_names=["out_file"],
                function=_copy_b1plusmap,
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
    if "brain_mask" in present:
        sink = _add_anat_mask_sink(
            workflow, inputnode, base_minimal,
            name="brain_mask_dce", in_attr="brain_mask",
            desc_value="brain", description="Brain mask in DCE space",
            skull_stripped=True,
        )
        sink_outputs.append((sink, "out_file"))
    if "nawm_mask" in present:
        sink = _add_anat_mask_sink(
            workflow, inputnode, base_minimal,
            name="nawm_mask_dce", in_attr="nawm_mask",
            desc_value="nawm",
            description="Normal-appearing white matter mask in DCE space",
        )
        sink_outputs.append((sink, "out_file"))
    if "lesion_mask_dce" in present:
        sink = _add_anat_mask_sink(
            workflow, inputnode, base_minimal,
            name="lesion_mask_dce", in_attr="lesion_mask_dce",
            desc_value="lesion",
            description="Lesion mask registered to DCE space",
        )
        sink_outputs.append((sink, "out_file"))
    if "user_lesion_mask_dce" in present:
        sink = _add_anat_mask_sink(
            workflow, inputnode, base_minimal,
            name="user_lesion_mask_dce", in_attr="user_lesion_mask_dce",
            desc_value="userLesion",
            description="User-supplied lesion mask in DCE native space",
        )
        sink_outputs.append((sink, "out_file"))
    if "roi_mask_dce" in present:
        sink = _add_anat_mask_sink(
            workflow, inputnode, base_minimal,
            name="roi_mask_dce", in_attr="roi_mask_dce",
            desc_value="roi",
            description="ROI mask registered to DCE space",
        )
        sink_outputs.append((sink, "out_file"))

    # --- Structural underlays (T1w/FLAIR/T2w in DCE) ---
    if "t1w_in_dce" in present:
        sink = _add_anat_image_sink(
            workflow, inputnode, base_minimal,
            name="t1w_in_dce", in_attr="t1w_in_dce", label="T1w",
            description="Pre-contrast T1w registered to DCE space",
            ce_label="",
        )
        sink_outputs.append((sink, "out_file"))
    if "t1w_postcon_in_dce" in present:
        sink = _add_anat_image_sink(
            workflow, inputnode, base_minimal,
            name="t1w_postcon_in_dce", in_attr="t1w_postcon_in_dce",
            label="T1w",
            description="Post-contrast T1w registered to DCE space",
            ce_label="GAD",
        )
        sink_outputs.append((sink, "out_file"))
    if "flair_in_dce" in present:
        sink = _add_anat_image_sink(
            workflow, inputnode, base_minimal,
            name="flair_in_dce", in_attr="flair_in_dce", label="FLAIR",
            description="FLAIR registered to DCE space",
        )
        sink_outputs.append((sink, "out_file"))
    if "flair_postcon_in_dce" in present:
        sink = _add_anat_image_sink(
            workflow, inputnode, base_minimal,
            name="flair_postcon_in_dce", in_attr="flair_postcon_in_dce", label="FLAIR",
            description="Post-contrast FLAIR registered to DCE space",
            ce_label="GAD",
        )
        sink_outputs.append((sink, "out_file"))
    if "t2w_in_dce" in present:
        sink = _add_anat_image_sink(
            workflow, inputnode, base_minimal,
            name="t2w_in_dce", in_attr="t2w_in_dce", label="T2w",
            description="T2w registered to DCE space",
        )
        sink_outputs.append((sink, "out_file"))

    if "dce_to_t1w_xfm" in present:
        xfm_sink = pe.Node(
            niu.Function(
                input_names=["in_file", "output_dir", "subject", "session"],
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
            ]),
        ])
        sink_outputs.append((xfm_sink, "out_file"))

    # --- Smoothed maps (variable cardinality → MapNode) ---
    if has_smoothing:
        splitter = pe.Node(
            niu.Function(
                input_names=[
                    "smooth_pk", "smooth_mf",
                    "smooth_uncertainty",
                    "pk_units", "mf_units",
                ],
                output_names=[
                    "in_files", "descs", "fwhms", "units_list",
                    "desc_values", "suffixes", "use_model", "map_keys",
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
                ("smooth_uncertainty_maps", "smooth_uncertainty"),
            ]),
        ])

        sm_meta = pe.MapNode(
            niu.Function(
                input_names=["base", "description", "fwhm_mm", "units",
                             "map_key"],
                output_names=["meta_dict"],
                function=_build_dce_smoothed_meta,
            ),
            iterfield=["description", "fwhm_mm", "units", "map_key"],
            name="smoothed_meta",
        )
        workflow.connect([
            (base_perf, sm_meta, [("meta_dict", "base")]),
            (splitter, sm_meta, [
                ("descs", "description"),
                ("fwhms", "fwhm_mm"),
                ("units_list", "units"),
                ("map_keys", "map_key"),
            ]),
        ])

        # Per-item model_entity (PK + uncertainty use it; MF doesn't)
        model_picker = pe.Node(
            niu.Function(
                input_names=["use_model_flags", "model_entity"],
                output_names=["model_entities"],
                function=_model_entity_per_item,
            ),
            name="smoothed_model_picker",
        )
        model_picker.inputs.model_entity = model_entity
        workflow.connect(splitter, "use_model", model_picker, "use_model_flags")

        sm_sink = pe.MapNode(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype", "model_entity",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            iterfield=["in_file", "desc", "suffix", "meta_dict", "model_entity"],
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
            (model_picker, sm_sink, [("model_entities", "model_entity")]),
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
