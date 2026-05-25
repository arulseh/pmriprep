"""DSC pipeline derivatives factory.

Wires one ``PMRIPrepDataSink`` per output (NIfTI + sidecar) plus a
``DSCFiguresWriter`` for the AIF curve copy, confounds TSV/JSON, and
figure filmstrips. QC summary fields live on
``desc-dscPreproc_dsc.json`` as flat ``QC*`` keys; per-map lesion stats
live on the per-PK-map sidecars as ``LesionStats`` fragments.

Per-output sidecar enrichment uses ``apply_aif_sidecar`` from
``pmriprep.utils.aif_provenance`` to stamp every map with the
detection-method block + ``AIFDependent`` + ``TemplateAIFUsage`` tags.
"""
from __future__ import annotations

from typing import Any, Optional

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.interfaces.qc_writers import (
    DSCFiguresWriter,
    PMRIPrepDataSink,
    write_dataset_description_node,
)


# ---------------------------------------------------------------------------
# Output declarative tables
# ---------------------------------------------------------------------------


# Always-present DSC perfusion maps. Each entry:
# (input_attr, desc_value, suffix, description, units, map_key, extra_fields)
_CORE_MAPS: tuple[tuple[str, str, str, str, str, str, dict | None], ...] = (
    ("mtt_map", "dsc", "mtt", "Mean transit time", "s", "mtt", None),
    ("tmax_map", "dsc", "tmax", "Time to maximum of residue function", "s", "tmax", None),
    ("psr_map", "dsc", "psr", "Percent signal recovery", "%", "psr", None),
    ("ph_map", "dsc", "ph", "Peak height", "arbitrary", "ph", None),
)


# Relative maps (always produced, but inputs may be undefined). Note the
# bids_suffix/desc remap retained from the legacy writer for back-compat.
_REL_MAPS: tuple[tuple[str, str, str, str, str, str, dict | None], ...] = (
    ("rel_cbf_map", "dscRel", "cbf",
     "Cerebral blood flow (relative)", "1/min", "cbf", None),
    ("rel_cbv_map", "dscRel", "cbv",
     "Cerebral blood volume (relative, deconvolution)", "arbitrary", "cbv",
     {"CBVMethod": "deconvolution_residue_integral"}),
    ("rel_cbv_auc_map", "dscRelAUC", "cbv",
     "Cerebral blood volume (K2-corrected full-bolus AUC, relative)",
     "arbitrary", "cbv_auc",
     {"CBVMethod": "k2_corrected_full_bolus_auc"}),
    ("rel_auc_raw_map", "dscRel", "auc",
     "AUC raw concentration (relative)", "arbitrary", "auc", None),
    ("rel_auc_corr_map", "dscRelCorrected", "auc",
     "AUC corrected concentration (relative)", "arbitrary", "auc_corr", None),
)


# Optional maps with per-map desc + bids_suffix
_OPTIONAL_MAPS: tuple[tuple[str, str, str, str, str | None, str, dict | None], ...] = (
    ("ttp_map", "dsc", "ttp", "Time to peak concentration", "seconds", "ttp",
     {"ReferencePoint": "start of acquisition (volume 0)"}),
    ("ttp_corrected_map", "dscAifCorrected", "ttp",
     "Time to peak, corrected for AIF bolus arrival delay. "
     "Comparable across patients and centers.",
     "seconds", "ttpCorrected",
     {"ReferencePoint": "AIF bolus arrival time"}),
    ("rttp_map", "dscNawmRel", "ttp",
     "Time to peak relative to contralateral NAWM median. "
     "Positive values indicate delayed perfusion.",
     "seconds", "rttp",
     {"ReferencePoint": "contralateral NAWM median TTP"}),
    ("cnr_map", "dsc", "cnr", "Contrast-to-noise ratio", "arbitrary", "cnr", None),
    ("auc_diff_map", "dscLeakage", "auc",
     "Leakage contribution (AUC_raw - AUC_corr)", None, "auc_diff", None),
    ("snr_map", "dsc", "snr", "Signal-to-noise ratio (baseline)", "arbitrary", "snr", None),
    ("tsnr_map", "dsc", "tsnr", "Temporal signal-to-noise ratio", "arbitrary", "tsnr", None),
)


# Normalized maps
_NORMALIZED_MAPS: tuple[tuple[str, str, str, str, str, str, dict | None], ...] = (
    ("ncbv_map", "dscNawmNorm", "cbv",
     "Normalized CBV (NAWM-normalized)", "ratio", "ncbv", None),
    ("ncbf_map", "dscNawmNorm", "cbf",
     "Normalized CBF (NAWM-normalized)", "ratio", "ncbf", None),
)


# Vascular model parameters
_VASCULAR_MAPS: tuple[tuple[str, str, str, str, str | None, str, dict | None], ...] = (
    ("alpha_map", "dsc", "alpha", "Vascular model alpha parameter", None, "alpha", None),
    ("beta_map", "dsc", "beta", "Vascular model beta parameter", None, "beta", None),
    ("cth_map", "dsc", "cth", "Capillary transit time heterogeneity", None, "cth", None),
)


# Bayesian uncertainty maps. Same shape as above.
_BAYESIAN_MAPS: tuple[tuple[str, str, str, str, None, str, None], ...] = (
    ("cbf_std_map", "dscUncertainty", "cbf", "CBF uncertainty (posterior std)", None, "cbf_std", None),
    ("cbv_std_map", "dscUncertainty", "cbv", "CBV uncertainty (posterior std)", None, "cbv_std", None),
    ("cbf_cv_map", "dscCV", "cbf", "CBF coefficient of variation", None, "cbfCV", None),
    ("cbv_cv_map", "dscCV", "cbv", "CBV coefficient of variation", None, "cbvCV", None),
    ("cbf_ci95_lower_map", "dscCI95lower", "cbf",
     "CBF 95% credible interval lower bound", None, "cbfCI95lower", None),
    ("cbf_ci95_upper_map", "dscCI95upper", "cbf",
     "CBF 95% credible interval upper bound", None, "cbfCI95upper", None),
    ("cbv_ci95_lower_map", "dscCI95lower", "cbv",
     "CBV 95% credible interval lower bound", None, "cbvCI95lower", None),
    ("cbv_ci95_upper_map", "dscCI95upper", "cbv",
     "CBV 95% credible interval upper bound", None, "cbvCI95upper", None),
)


#: Canonical list of map_keys passed to ``apply_aif_sidecar`` from the factory.
#: Used by the drift-guard test in ``test_dsc_outputs_sidecar.py`` to verify
#: every emitted key is registered in :data:`TEMPLATE_AIF_USAGE`.
_DSC_OUTPUT_KEYS: frozenset[str] = frozenset(
    [m[5] for m in _CORE_MAPS + _REL_MAPS + _OPTIONAL_MAPS
     + _NORMALIZED_MAPS + _VASCULAR_MAPS + _BAYESIAN_MAPS]
    + ["k2", "parametric_r2", "fallback",
       "rel_cbf", "rel_cbv", "rel_auc_corr", "ncbf", "ncbv", "mtt"]  # smoothed names
)


# ---------------------------------------------------------------------------
# Function-node wrappers
# ---------------------------------------------------------------------------


def _build_base(  # type: ignore[no-untyped-def]
    source_paths,
    reference_voxel_source,
    bids_dir,
    subject,
    session,
    provenance_dict=None,
    qc_metrics=None,
    minimal=False,
):
    from pmriprep.utils.sidecar_builders import build_dsc_base_sidecar

    return build_dsc_base_sidecar(
        source_paths=source_paths,
        reference_voxel_source=reference_voxel_source,
        bids_dir=bids_dir,
        subject=subject,
        session=session or None,
        provenance_dict=provenance_dict,
        qc_metrics=qc_metrics,
        minimal=minimal,
    )


def _build_map_meta(  # type: ignore[no-untyped-def]
    base,
    description,
    map_key,
    qc_metrics=None,
    units=None,
    extra=None,
    lesion_stats_json=None,
):
    """Compose a DSC per-map sidecar; injects per-map LesionStats fragment.

    The aggregate lesion-stats JSON (one file per pipeline) is loaded
    when present and split into per-map fragments via
    :func:`extract_per_map_lesion_stats`. The fragment is merged into
    the ``extra`` dict alongside any caller-provided extras (e.g.
    K2/Bayesian/Vascular extras).
    """
    import json as _json
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats
    from pmriprep.utils.sidecar_builders import build_dsc_map_sidecar

    fragment = None
    if isdefined(lesion_stats_json) and lesion_stats_json:
        path = Path(lesion_stats_json)
        if path.exists():
            try:
                with open(path) as fh:
                    aggregate = _json.load(fh)
                fragment = extract_per_map_lesion_stats(aggregate, map_key)
            except Exception:
                fragment = None

    if fragment:
        merged_extra = dict(extra) if extra else {}
        merged_extra["LesionStats"] = fragment
    else:
        merged_extra = extra

    return build_dsc_map_sidecar(
        base=base,
        description=description,
        map_key=map_key,
        qc_metrics=qc_metrics,
        units=units,
        extra=merged_extra,
    )


def _build_minimal_meta(  # type: ignore[no-untyped-def]
    base, description, skull_stripped=False,
):
    out = dict(base)
    out["Description"] = description
    if skull_stripped:
        out["SkullStripped"] = True
    return out


def _build_dsc_preproc_meta(  # type: ignore[no-untyped-def]
    base,
    description,
    preload_qc=None,
    preprocessing_stats=None,
    qc_metrics=None,
    provenance_dict=None,
    preflight_warnings=None,
    presence_warnings=None,
    config_dict=None,
):
    """Compose the ``desc-dscPreproc_dsc.json`` sidecar.

    Starts from the minimal-base meta, layers on flat-prefixed cosmetic
    QC keys via ``enrich_dsc_preproc_sidecar``, and finally attaches
    QC recommendation/warnings flat keys via
    ``enrich_recommendation_keys`` so the HTML aggregator can read the
    overall recommendation banner directly from this anchor sidecar.
    ``PipelineQC`` and ``PipelineConfiguration`` nest the full merged
    QC envelope and processing config for the HTML aggregator.
    """
    from pmriprep.utils.sidecar_builders import (
        build_provenance_block,
        enrich_dsc_preproc_sidecar,
        enrich_recommendation_keys,
        filter_pipeline_qc,
    )

    out = dict(base)
    out["Description"] = description
    enrich_dsc_preproc_sidecar(
        out,
        preload_qc=preload_qc if isinstance(preload_qc, dict) else None,
        preprocessing_stats=(
            preprocessing_stats if isinstance(preprocessing_stats, dict) else None
        ),
    )
    enrich_recommendation_keys(
        out,
        modality="DSC",
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
    return out


def _write_dsc_resume_checkpoint(  # type: ignore[no-untyped-def]
    work_dir, subject_id, session_id,
    aif_qc=None, preload_qc=None, preprocessing_stats=None,
    motion_qc=None, aif_metrics=None,
):
    """Write the DSC→DCE handoff checkpoint to the work directory."""
    import json as _json
    from pathlib import Path

    if not isinstance(work_dir, str):
        return None

    pp = dict(preprocessing_stats) if isinstance(preprocessing_stats, dict) else {}
    if isinstance(motion_qc, dict):
        for k in (
            "fd_mean", "fd_max", "motion_status",
            "n_outliers", "outlier_frames", "motion_thresh",
            "fd_list", "n_volumes", "percent_outliers",
        ):
            if k in motion_qc:
                pp[k] = motion_qc[k]

    aq = dict(aif_qc) if isinstance(aif_qc, dict) else {}
    upstream_metrics = aq.get("aif_metrics")
    merged_metrics = (
        dict(upstream_metrics) if isinstance(upstream_metrics, dict) else {}
    )
    if isinstance(aif_metrics, dict) and aif_metrics:
        merged_metrics.update(aif_metrics)
    if merged_metrics:
        aq["aif_metrics"] = merged_metrics

    checkpoint: dict = {}
    if aq:
        checkpoint["aif_qc"] = aq
    if isinstance(preload_qc, dict) and preload_qc:
        checkpoint["preload_qc"] = dict(preload_qc)
    if pp:
        checkpoint["preprocessing_stats"] = pp

    from pmriprep.utils.naming import workflow_dirname

    wf_name = workflow_dirname(subject_id, session_id)
    out_dir = Path(work_dir) / wf_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dsc_resume.json"
    out_path.write_text(_json.dumps(checkpoint, indent=2, default=str))
    return str(out_path)


def _copy_xfm(  # type: ignore[no-untyped-def]
    in_file, output_dir, subject, session, from_space, to_space, datatype="anat",
):
    """Copy a transform .h5 into the derivatives ``anat/`` directory."""
    import logging
    import shutil
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    # Defensive no-op when upstream registration produced no transform.
    # See pmriprep.workflows.dce.datasinks._copy_perf_map for the
    # sentinel-handling rationale.
    if not isdefined(in_file) or in_file is None or in_file == "":
        logging.getLogger("nipype.workflow").warning(
            "Skipping DSC _copy_xfm sink (from=%s, to=%s): upstream "
            "produced no output (in_file=%r)",
            from_space, to_space, in_file,
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
    fname = f"{prefix}_from-{from_space}_to-{to_space}_mode-image_xfm.h5"
    out_path = out_root / fname
    shutil.copy2(in_file, out_path)
    return str(out_path)


def _copy_perf_map(  # type: ignore[no-untyped-def]
    in_file, output_dir, subject, session, desc, suffix, meta_dict,
    datatype="perf", bids_suffix="stat",
):
    """Copy a DSC perfusion map + write its sidecar with BIDS-compliant naming.

    ``suffix`` is the parameter name (e.g. ``"mtt"``, ``"cbf"``) which
    gets folded into the ``desc-`` entity.  ``bids_suffix`` is the
    BIDS-canonical suffix (``stat``, ``mask``, ``dseg``, …).  Pass
    ``suffix=""`` when ``desc`` already encodes everything and only the
    BIDS suffix is needed.
    """
    import json as _json
    import logging
    import shutil
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    # Defensive no-op when upstream conditional producer returned no
    # output. See pmriprep.workflows.dce.datasinks._copy_perf_map for
    # the sentinel-handling rationale (Undefined/None/"" all need
    # explicit handling because isdefined(None) == True is a Nipype
    # gotcha).
    if not isdefined(in_file) or in_file is None or in_file == "":
        logging.getLogger("nipype.workflow").warning(
            "Skipping DSC _copy_perf_map sink (desc=%s, suffix=%s): upstream "
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
    folded = suffix[:1].upper() + suffix[1:] if suffix else ""
    stem = f"{prefix}_desc-{desc}{folded}_{bids_suffix}"
    nii_out = out_root / f"{stem}.nii.gz"
    json_out = out_root / f"{stem}.json"

    shutil.copy2(in_file, nii_out)
    json_out.write_text(_json.dumps(meta_dict, indent=2, default=str))
    return str(nii_out)


def _split_smooth_dict(  # type: ignore[no-untyped-def]
    smooth_hemo=None, smooth_auc=None, fwhm=0.0,
):
    """Flatten the two smoothed-map dicts into per-output ``(name, path, descs)`` tuples.

    Returns three lists: ``in_files``, ``descs``, ``map_keys``,
    ``map_suffixes``, ``desc_prefixes``, ``units_list`` matching the legacy
    writer's templated naming (``dsc{Sm{N}}``, ``dscRel{Sm{N}}``, etc.).
    """
    from pathlib import Path
    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    out_files = []
    out_descs = []
    out_keys = []
    out_map_suffixes = []
    out_desc_prefixes = []
    out_units = []

    if not (isdefined(fwhm) and fwhm and float(fwhm) > 0):
        return out_files, out_descs, out_keys, out_map_suffixes, out_desc_prefixes, out_units
    sm_label = f"Sm{int(float(fwhm))}"

    hemo_table = {
        "mtt": ("Mean transit time (smoothed)", f"dsc{sm_label}", "mtt", "s"),
        "rel_cbf": ("Cerebral blood flow, relative (smoothed)",
                    f"dscRel{sm_label}", "cbf", "1/min"),
        "rel_cbv": ("Cerebral blood volume, AUC-based (smoothed)",
                    f"dscRelAUC{sm_label}", "cbv", "arbitrary"),
        "ncbf": ("Normalized CBF (smoothed)",
                 f"dscNawmNorm{sm_label}", "cbf", "ratio"),
        "ncbv": ("Normalized CBV (smoothed)",
                 f"dscNawmNorm{sm_label}", "cbv", "ratio"),
    }
    if isdefined(smooth_hemo) and smooth_hemo:
        for name, fp in smooth_hemo.items():
            if not (fp and Path(fp).exists()):
                continue
            if name in hemo_table:
                desc_text, desc_prefix, map_suffix, units = hemo_table[name]
            else:
                desc_text = f"{name} (smoothed)"
                desc_prefix = f"dsc{sm_label}"
                map_suffix = name.replace("rel_", "").replace("abs_", "")
                units = ""
            out_files.append(str(fp))
            out_descs.append(desc_text)
            out_keys.append(name)
            out_map_suffixes.append(map_suffix)
            out_desc_prefixes.append(desc_prefix)
            out_units.append(units)

    if isdefined(smooth_auc) and smooth_auc:
        for name, fp in smooth_auc.items():
            if not (fp and Path(fp).exists()):
                continue
            desc_text = (
                "Relative AUC corrected (smoothed)"
                if name == "rel_auc_corr"
                else f"{name} (smoothed)"
            )
            desc_prefix = (
                f"dscRelCorrected{sm_label}"
                if name == "rel_auc_corr"
                else f"dsc{sm_label}"
            )
            out_files.append(str(fp))
            out_descs.append(desc_text)
            out_keys.append(name)
            out_map_suffixes.append("auc")
            out_desc_prefixes.append(desc_prefix)
            out_units.append("arbitrary")

    return out_files, out_descs, out_keys, out_map_suffixes, out_desc_prefixes, out_units


def _build_k2_extra(  # type: ignore[no-untyped-def]
    preload_qc=None,
):
    """Compose the K2 sidecar's ``extra`` dict from ``preload_qc``."""
    extra = {"LeakageCorrectionMethod": "bsw"}
    if isinstance(preload_qc, dict):
        method = preload_qc.get("k2_method")
        if method:
            extra["LeakageCorrectionMethod"] = str(method)
        kep = preload_qc.get("kep_used")
        if kep is not None:
            extra["KepUsed"] = kep
            extra["KepUnits"] = "1/s"
    return extra


def _build_parametric_extra(  # type: ignore[no-untyped-def]
    qc_metrics=None,
):
    """FitMethod + InferenceEngine + FallbackThreshold from qc_metrics.deconvolution."""
    extra = {"FallbackThreshold": 0.5}
    if isinstance(qc_metrics, dict):
        deconv = qc_metrics.get("deconvolution") or {}
        if isinstance(deconv, dict):
            if deconv.get("method"):
                extra["FitMethod"] = str(deconv["method"])
            if deconv.get("inference"):
                extra["InferenceEngine"] = str(deconv["inference"])
    return extra


def _build_bayes_extra(  # type: ignore[no-untyped-def]
    config_dict=None, umap_key="cbf_std",
):
    """EstimationMethod + UncertaintyMethod for Bayesian uncertainty maps."""
    estimation = "unknown"
    if isinstance(config_dict, dict):
        # Newer field is "inference"; legacy key was "bayesian".
        estimation = (
            config_dict.get("inference")
            or config_dict.get("bayesian")
            or "unknown"
        )
    if "CV" in umap_key:
        kind = "coefficient_of_variation"
    elif "CI95" in umap_key:
        kind = "95pct_credible_interval"
    else:
        kind = "std"
    return {
        "EstimationMethod": str(estimation),
        "UncertaintyMethod": kind,
    }


def _build_smoothed_meta(  # type: ignore[no-untyped-def]
    base, description, map_key, qc_metrics, units, fwhm,
):
    from pmriprep.utils.sidecar_builders import build_dsc_map_sidecar

    extra = {
        "SmoothingFWHM": float(fwhm),
        "SmoothingType": "Gaussian",
        "SmoothingMasked": True,
        "IntendedFor": "visualization",
        "Note": "Use non-smoothed maps for quantitative ROI analysis",
    }
    return build_dsc_map_sidecar(
        base=base,
        description=description,
        map_key=map_key,
        qc_metrics=qc_metrics,
        units=units or None,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Sink helpers
# ---------------------------------------------------------------------------


_PERF_SOURCES = ("source_dsc",)


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
    map_key: str,
    units: Optional[str],
    extra: Optional[dict],
    bids_suffix: str = "stat",
) -> pe.Node:
    """Wire one perf-space DSC map copier (Function builder + Function copy).

    ``suffix`` is the parameter name folded into ``desc-``.
    ``bids_suffix`` is the BIDS-canonical suffix (``stat``, ``mask``, …).
    """
    meta = pe.Node(
        niu.Function(
            input_names=[
                "base", "description", "map_key", "qc_metrics", "units",
                "extra", "lesion_stats_json",
            ],
            output_names=["meta_dict"],
            function=_build_map_meta,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.map_key = map_key
    meta.inputs.units = units
    meta.inputs.extra = extra
    workflow.connect([
        (base_node, meta, [("meta_dict", "base")]),
        (inputnode, meta, [
            ("qc_metrics", "qc_metrics"),
            ("lesion_stats_json", "lesion_stats_json"),
        ]),
    ])

    sink = pe.Node(
        niu.Function(
            input_names=[
                "in_file", "output_dir", "subject", "session",
                "desc", "suffix", "meta_dict", "datatype",
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


def _add_anat_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    base_minimal_node: pe.Node,
    *,
    name: str,
    in_attr: str,
    desc_value: Optional[str],
    suffix: str,
    description: str,
    skull_stripped: bool = False,
    ce_value: Optional[str] = None,
    source_attr: str = "source_dsc",
    dismiss_entities: tuple = ("acquisition",),
) -> pe.Node:
    """Wire one anat/-side DSC sink (mask, structural underlay).

    Forces ``datatype="anat"`` because the source_file is a perf-datatype
    NIfTI; structural suffixes (T1w/FLAIR/T2w) in perf/ violate BIDS, and
    even masks belong under anat/ when they describe anatomy in DSC space.

    ``dismiss_entities`` strips entities the niworkflows ``DerivativesDataSink``
    would otherwise inherit from the source DSC file. Default drops
    ``acquisition`` because the DSC source's acquisition (e.g. EP2D) doesn't
    describe T1w/mask data — it describes the spatial-reference grid.
    Pre-contrast-derived masks (brain, NAWM) should additionally pass
    ``("acquisition", "ceagent")`` to drop ``ce-GAD`` since they were computed
    from pre-con T1w.
    """
    meta = pe.Node(
        niu.Function(
            input_names=["base", "description", "skull_stripped"],
            output_names=["meta_dict"],
            function=_build_minimal_meta,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.skull_stripped = skull_stripped
    workflow.connect([(base_minimal_node, meta, [("meta_dict", "base")])])

    sink = pe.Node(
        PMRIPrepDataSink(
            check_hdr=True, compress=True,
            allowed_entities=("space", "datatype", "ceagent"),
            space="dscref", suffix=suffix, datatype="anat",
            dismiss_entities=list(dismiss_entities),
        ),
        name=f"{name}_sink",
    )
    if desc_value is not None:
        sink.inputs.desc = desc_value
    if ce_value is not None:
        sink.inputs.ceagent = ce_value
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "base_directory"),
            (in_attr, "in_file"),
            (source_attr, "source_file"),
        ]),
        (meta, sink, [("meta_dict", "meta_dict")]),
    ])
    return sink


def _add_xfm_sink(
    workflow: pe.Workflow,
    inputnode: pe.Node,
    *,
    name: str,
    in_attr: str,
    from_space: str,
    to_space: str,
    source_attr: str = "source_dsc",
) -> pe.Node:
    sink = pe.Node(
        niu.Function(
            input_names=["in_file", "output_dir", "subject", "session",
                         "from_space", "to_space", "datatype"],
            output_names=["out_file"],
            function=_copy_xfm,
        ),
        name=f"{name}_sink",
    )
    sink.inputs.from_space = from_space
    sink.inputs.to_space = to_space
    sink.inputs.datatype = "anat"
    workflow.connect([
        (inputnode, sink, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            (in_attr, "in_file"),
        ]),
    ])
    return sink


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_ALWAYS_PRESENT_INPUTS = frozenset({
    # The four DSC core perfusion maps are mandatory inputs to the factory.
    # All other outputs (moco, mean ref, AIF mask, optional/normalized/
    # vascular/Bayesian/k2/parametric, structural underlays, masks, xfms)
    # are gated by ``present_inputs``.
    "mtt_map", "tmax_map", "psr_map", "ph_map",
})


def init_dsc_datasinks_wf(
    *,
    output_dir: str,
    bids_dir: str,
    subject_id: str,
    session_id: Optional[str],
    present_inputs: Optional[set[str]] = None,
    has_smoothing: bool = False,
    name: str = "dsc_datasinks_wf",
) -> pe.Workflow:
    """Build the DSC-pipeline DerivativesDataSink chain.

    ``present_inputs`` is a set of inputnode field names that will be
    wired by the caller. Sinks for absent inputs are NOT created (avoids
    nipype's mandatory-input check failing). The mandatory-perfusion
    inputs (``mtt_map``, ``tmax_map``, ``psr_map``, ``ph_map``,
    ``dsc_moco``, ``dsc_mean``, ``aif_mask``) are always treated as
    present.
    """
    if present_inputs is None:
        present_inputs = set()
    present = set(_ALWAYS_PRESENT_INPUTS) | set(present_inputs)

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "output_dir", "bids_dir", "subject_id", "session_id",
                "work_dir",
                # Sources
                "source_dsc",
                # Always-present perfusion maps
                "mtt_map", "tmax_map", "psr_map", "ph_map",
                # Relative
                "rel_cbf_map", "rel_cbv_map", "rel_cbv_auc_map",
                "rel_auc_raw_map", "rel_auc_corr_map",
                # Optional
                "ttp_map", "ttp_corrected_map", "rttp_map",
                "cnr_map", "auc_diff_map", "snr_map", "tsnr_map",
                # Normalized
                "ncbv_map", "ncbf_map",
                # Vascular
                "alpha_map", "beta_map", "cth_map",
                # Bayesian uncertainty
                "cbf_std_map", "cbv_std_map", "cbf_cv_map", "cbv_cv_map",
                "cbf_ci95_lower_map", "cbf_ci95_upper_map",
                "cbv_ci95_lower_map", "cbv_ci95_upper_map",
                # Parametric diagnostics + leakage
                "k2_map", "parametric_r2_map", "fallback_mask",
                "preload_qc", "preprocessing_stats", "aif_qc",
                "motion_qc", "aif_metrics",
                # Smoothing
                "smooth_hemodynamic", "smooth_auc", "smooth_fwhm",
                # Motion + AIF
                "dsc_moco", "dsc_mean", "motion_params",
                "aif_curve", "aif_mask",
                # Anat-side: masks + structural underlays + xfms + lesion/roi
                "brain_mask_dsc", "nawm_mask_dsc",
                "lesion_mask_dsc", "lesion_stats_json",
                "roi_mask_dsc", "roi_stats_json",
                "t1w_in_dsc", "t1w_postcon_in_dsc",
                "flair_in_dsc", "t2w_in_dsc",
                "dsc_to_t1w_xfm", "t1w_to_dsc_xfm",
                # Sidecar drivers
                "qc_metrics", "provenance_dict", "preflight_warnings",
                "config_dict",
                # Figures (passed through to DSCFiguresWriter)
                "brain_mask_filmstrip", "motion_qc_plot",
                "aif_curve_plot", "aif_location_plot", "aif_candidates_plot",
                "registration_filmstrip",
                "dsc_preprocessing_qc_plot", "dsc_param_maps_qc_plot",
                "dsc_fit_qc_plot", "dsc_leakage_qc_plot",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.output_dir = output_dir
    inputnode.inputs.bids_dir = bids_dir
    inputnode.inputs.subject_id = subject_id
    inputnode.inputs.session_id = session_id or ""

    # Source-paths Merge for the perfusion-side base sidecar
    perf_src_merge = pe.Node(niu.Merge(numinputs=1), name="perf_sources")
    workflow.connect(inputnode, "source_dsc", perf_src_merge, "in1")

    # Base sidecar (perfusion: enriched with provenance + AIF QC)
    base_perf = pe.Node(
        niu.Function(
            input_names=[
                "source_paths", "reference_voxel_source", "bids_dir",
                "subject", "session",
                "provenance_dict", "qc_metrics", "minimal",
            ],
            output_names=["meta_dict"],
            function=_build_base,
        ),
        name="base_perf_meta",
    )
    base_perf.inputs.minimal = False
    workflow.connect([
        (perf_src_merge, base_perf, [("out", "source_paths")]),
        (inputnode, base_perf, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            ("mtt_map", "reference_voxel_source"),
            ("provenance_dict", "provenance_dict"),
            ("qc_metrics", "qc_metrics"),
        ]),
    ])

    # Base sidecar (anat-side: minimal — no perfusion enrichment)
    base_minimal = pe.Node(
        niu.Function(
            input_names=[
                "source_paths", "reference_voxel_source", "bids_dir",
                "subject", "session",
                "provenance_dict", "qc_metrics", "minimal",
            ],
            output_names=["meta_dict"],
            function=_build_base,
        ),
        name="base_minimal_meta",
    )
    base_minimal.inputs.minimal = True
    base_minimal.inputs.qc_metrics = None
    workflow.connect([
        (perf_src_merge, base_minimal, [("out", "source_paths")]),
        (inputnode, base_minimal, [
            ("bids_dir", "bids_dir"),
            ("subject_id", "subject"),
            ("session_id", "session"),
            ("mtt_map", "reference_voxel_source"),
            ("provenance_dict", "provenance_dict"),
        ]),
    ])

    sink_outputs: list[tuple[pe.Node, str]] = []

    # --- Core + Rel + Optional + Normalized + Vascular maps ---
    for table in (
        _CORE_MAPS, _REL_MAPS, _OPTIONAL_MAPS,
        _NORMALIZED_MAPS, _VASCULAR_MAPS,
    ):
        for entry in table:
            in_attr, desc_value, suffix, description, units, map_key, extra = entry
            if in_attr not in present:
                continue
            sink = _add_perf_map_sink(
                workflow, inputnode, base_perf,
                name=f"{in_attr}",
                in_attr=in_attr,
                desc_value=desc_value, suffix=suffix,
                description=description, map_key=map_key,
                units=units, extra=extra,
            )
            sink_outputs.append((sink, "out_file"))

    # --- Bayesian uncertainty maps (extras built per-map from config_dict) ---
    for entry in _BAYESIAN_MAPS:
        in_attr, desc_value, suffix, description, units, map_key, _extra = entry
        if in_attr not in present:
            continue
        bayes_extra = pe.Node(
            niu.Function(
                input_names=["config_dict", "umap_key"],
                output_names=["extra"],
                function=_build_bayes_extra,
            ),
            name=f"{in_attr}_extra",
        )
        bayes_extra.inputs.umap_key = map_key
        workflow.connect(inputnode, "config_dict", bayes_extra, "config_dict")
        bayes_meta = pe.Node(
            niu.Function(
                input_names=[
                    "base", "description", "map_key", "qc_metrics", "units", "extra",
                ],
                output_names=["meta_dict"],
                function=_build_map_meta,
            ),
            name=f"{in_attr}_meta",
        )
        bayes_meta.inputs.description = description
        bayes_meta.inputs.map_key = map_key
        bayes_meta.inputs.units = units
        workflow.connect([
            (base_perf, bayes_meta, [("meta_dict", "base")]),
            (inputnode, bayes_meta, [("qc_metrics", "qc_metrics")]),
            (bayes_extra, bayes_meta, [("extra", "extra")]),
        ])
        b_sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name=f"{in_attr}_sink",
        )
        b_sink.inputs.desc = desc_value
        b_sink.inputs.suffix = suffix
        b_sink.inputs.datatype = "perf"
        b_sink.inputs.bids_suffix = "stat"
        workflow.connect([
            (inputnode, b_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                (in_attr, "in_file"),
            ]),
            (bayes_meta, b_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((b_sink, "out_file"))

    # --- K2 leakage map ---
    if "k2_map" in present:
        # K2 sidecar needs the preload-derived method + kep details from
        # ``preload_qc``; build extras via a Function node so they reflect
        # the actual run rather than a hardcoded "bsw" default.
        k2_extra_node = pe.Node(
            niu.Function(
                input_names=["preload_qc"],
                output_names=["extra"],
                function=_build_k2_extra,
            ),
            name="k2_extra",
        )
        workflow.connect(inputnode, "preload_qc", k2_extra_node, "preload_qc")
        k2_meta = pe.Node(
            niu.Function(
                input_names=[
                    "base", "description", "map_key", "qc_metrics", "units",
                    "extra", "lesion_stats_json",
                ],
                output_names=["meta_dict"],
                function=_build_map_meta,
            ),
            name="k2_meta",
        )
        k2_meta.inputs.description = "K2 leakage correction map"
        k2_meta.inputs.map_key = "k2"
        k2_meta.inputs.units = "1/s"
        workflow.connect([
            (base_perf, k2_meta, [("meta_dict", "base")]),
            (inputnode, k2_meta, [
                ("qc_metrics", "qc_metrics"),
                ("lesion_stats_json", "lesion_stats_json"),
            ]),
            (k2_extra_node, k2_meta, [("extra", "extra")]),
        ])
        k2_sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="k2_sink",
        )
        k2_sink.inputs.desc = "dsc"
        k2_sink.inputs.suffix = "k2"
        k2_sink.inputs.datatype = "perf"
        k2_sink.inputs.bids_suffix = "stat"
        workflow.connect([
            (inputnode, k2_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("k2_map", "in_file"),
            ]),
            (k2_meta, k2_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((k2_sink, "out_file"))

    # --- Parametric R² + fallback mask ---
    # Verbatim long descriptions from the legacy writer — they explain that
    # R² is preserved on fallback voxels, that those voxels' CBF/CBV/MTT/Tmax
    # come from oSVD (NOT from the parametric fit with this R²), and that
    # uncertainty maps are zero-filled in fallback voxels.
    _PARAMETRIC_R2_DESC = (
        "Per-voxel R² (coefficient of determination) of the parametric DSC "
        "fit (CPI / Vascular) against the K2-corrected tissue concentration. "
        "Raw parametric R² is preserved on voxels that fell back to oSVD — "
        "see dscFallback_mask to identify those. The CBF/CBV/MTT/Tmax saved "
        "in fallback voxels comes from oSVD, NOT from a fit with this R²."
    )
    _FALLBACK_DESC = (
        "Binary oSVD-fallback mask. 1 = the parametric R²<0.5 gate rejected "
        "this voxel and the saved CBF/CBV/MTT/Tmax come from "
        "svd_deconvolve_oscillation; the parametric uncertainty maps "
        "(cbf_std, cbv_std, CV, CI95) are zero-filled in these voxels. "
        "0 = the parametric fit was retained."
    )
    for in_attr, desc_value, suffix, description, key, bids_sfx in (
        ("parametric_r2_map", "dscParametric", "Rsquared",
         _PARAMETRIC_R2_DESC, "parametric_r2", "stat"),
        ("fallback_mask", "dscFallback", "",
         _FALLBACK_DESC, "fallback", "mask"),
    ):
        if in_attr not in present:
            continue
        # Build extras from qc_metrics["deconvolution"] (FitMethod +
        # InferenceEngine + FallbackThreshold).
        param_extra = pe.Node(
            niu.Function(
                input_names=["qc_metrics"],
                output_names=["extra"],
                function=_build_parametric_extra,
            ),
            name=f"{in_attr}_extra",
        )
        workflow.connect(inputnode, "qc_metrics", param_extra, "qc_metrics")
        param_meta = pe.Node(
            niu.Function(
                input_names=[
                    "base", "description", "map_key", "qc_metrics", "units", "extra",
                ],
                output_names=["meta_dict"],
                function=_build_map_meta,
            ),
            name=f"{in_attr}_meta",
        )
        param_meta.inputs.description = description
        param_meta.inputs.map_key = key
        param_meta.inputs.units = None
        workflow.connect([
            (base_perf, param_meta, [("meta_dict", "base")]),
            (inputnode, param_meta, [("qc_metrics", "qc_metrics")]),
            (param_extra, param_meta, [("extra", "extra")]),
        ])
        sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name=f"{in_attr}_sink",
        )
        sink.inputs.desc = desc_value
        sink.inputs.suffix = suffix
        sink.inputs.datatype = "perf"
        sink.inputs.bids_suffix = bids_sfx
        workflow.connect([
            (inputnode, sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                (in_attr, "in_file"),
            ]),
            (param_meta, sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((sink, "out_file"))

    # --- Motion-corrected 4D + mean ref ---
    # Motion-corrected 4D + mean ref + AIF mask (non-canonical suffixes →
    # Function-based copy bypassing pybids path validation).
    def _add_minimal_perf_sink(
        nm: str, in_attr: str, desc_value: str, suffix: str, description: str,
        bids_sfx: str = "stat",
    ) -> pe.Node:
        meta = pe.Node(
            niu.Function(
                input_names=["base", "description", "skull_stripped"],
                output_names=["meta_dict"],
                function=_build_minimal_meta,
            ),
            name=f"{nm}_meta",
        )
        meta.inputs.description = description
        meta.inputs.skull_stripped = False
        workflow.connect([(base_minimal, meta, [("meta_dict", "base")])])
        sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name=f"{nm}_sink",
        )
        sink.inputs.desc = desc_value
        sink.inputs.suffix = suffix
        sink.inputs.datatype = "perf"
        sink.inputs.bids_suffix = bids_sfx
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

    if "dsc_moco" in present:
        # The preproc-4D sidecar is the canonical anchor for DSC: it
        # carries flat-prefixed cosmetic Preload* / motion / signal QC
        # keys AND the QC* recommendation/warnings keys the HTML
        # aggregator reads. The verbatim resume sidecar (below) carries
        # the producer-shape dicts for --skip-dsc round-trip.
        presence_warnings_dsc = [
            warn for cond, warn in (
                ("brain_mask_dsc" not in present,
                 "Brain mask in DSC space not available"),
                ("nawm_mask_dsc" not in present,
                 "NAWM mask in DSC space not available"),
                ("lesion_mask_dsc" not in present,
                 "Lesion mask in DSC space not available"),
                ("dsc_to_t1w_xfm" not in present,
                 "DSC to T1w registration transform not available"),
            ) if cond
        ]
        moco_meta = pe.Node(
            niu.Function(
                input_names=[
                    "base", "description", "preload_qc", "preprocessing_stats",
                    "qc_metrics", "provenance_dict", "preflight_warnings",
                    "presence_warnings", "config_dict",
                ],
                output_names=["meta_dict"],
                function=_build_dsc_preproc_meta,
            ),
            name="moco_meta",
        )
        moco_meta.inputs.description = "Motion-corrected 4D DSC time series"
        moco_meta.inputs.presence_warnings = presence_warnings_dsc
        workflow.connect([
            (base_minimal, moco_meta, [("meta_dict", "base")]),
            (inputnode, moco_meta, [
                ("preload_qc", "preload_qc"),
                ("preprocessing_stats", "preprocessing_stats"),
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
                    "desc", "suffix", "meta_dict", "datatype",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_perf_map,
            ),
            name="moco_sink",
        )
        moco_sink.inputs.desc = "dscPreproc"
        moco_sink.inputs.suffix = ""
        moco_sink.inputs.datatype = "perf"
        moco_sink.inputs.bids_suffix = "dsc"
        workflow.connect([
            (inputnode, moco_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("dsc_moco", "in_file"),
            ]),
            (moco_meta, moco_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((moco_sink, "out_file"))

        resume_checkpoint = pe.Node(
            niu.Function(
                input_names=[
                    "work_dir", "subject_id", "session_id",
                    "aif_qc", "preload_qc", "preprocessing_stats",
                    "motion_qc", "aif_metrics",
                ],
                output_names=["out_file"],
                function=_write_dsc_resume_checkpoint,
            ),
            name="resume_checkpoint",
        )
        workflow.connect([
            (inputnode, resume_checkpoint, [
                ("work_dir", "work_dir"),
                ("subject_id", "subject_id"),
                ("session_id", "session_id"),
                ("aif_qc", "aif_qc"),
                ("preload_qc", "preload_qc"),
                ("preprocessing_stats", "preprocessing_stats"),
                ("motion_qc", "motion_qc"),
                ("aif_metrics", "aif_metrics"),
            ]),
        ])
        sink_outputs.append((resume_checkpoint, "out_file"))
    if "dsc_mean" in present:
        sink_outputs.append((_add_minimal_perf_sink(
            "mean", "dsc_mean", "dscMean", "",
            "Mean DSC reference image", "stat",
        ), "out_file"))
    if "aif_mask" in present:
        sink_outputs.append((_add_minimal_perf_sink(
            "aif_mask", "aif_mask", "dscAif", "",
            "AIF detection mask in DSC space", "mask",
        ), "out_file"))

    # --- Anat-side: brain mask, NAWM mask in DSC space ---
    if "brain_mask_dsc" in present:
        # Brain mask was computed from pre-contrast T1w (HD-BET on MPRAGE pre);
        # drop ce-GAD which would falsely imply post-contrast derivation.
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name="brain_mask_dsc",
            in_attr="brain_mask_dsc",
            desc_value="brain", suffix="mask",
            description="Brain mask in DSC space",
            skull_stripped=True,
            dismiss_entities=("acquisition", "ceagent"),
        )
        sink_outputs.append((sink, "out_file"))
    if "nawm_mask_dsc" in present:
        # NAWM mask is FAST-derived from pre-contrast tissue probabilities;
        # drop ce-GAD for the same reason as brain_mask.
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name="nawm_mask_dsc",
            in_attr="nawm_mask_dsc",
            desc_value="nawm", suffix="mask",
            description="Normal-appearing white matter mask in DSC space",
            dismiss_entities=("acquisition", "ceagent"),
        )
        sink_outputs.append((sink, "out_file"))
    if "lesion_mask_dsc" in present:
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name="lesion_mask_dsc",
            in_attr="lesion_mask_dsc",
            desc_value="lesion", suffix="mask",
            description="Lesion mask registered to DSC space",
        )
        sink_outputs.append((sink, "out_file"))
    if "roi_mask_dsc" in present:
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name="roi_mask_dsc",
            in_attr="roi_mask_dsc",
            desc_value="roi", suffix="mask",
            description="ROI mask registered to DSC space",
        )
        sink_outputs.append((sink, "out_file"))

    # --- Structural underlays in DSC space ---
    if "t1w_in_dsc" in present:
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name="t1w_in_dsc",
            in_attr="t1w_in_dsc",
            desc_value=None, suffix="T1w",
            description="Pre-contrast T1w registered to DSC space",
            ce_value=None,
        )
        sink_outputs.append((sink, "out_file"))
    if "t1w_postcon_in_dsc" in present:
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name="t1w_postcon_in_dsc",
            in_attr="t1w_postcon_in_dsc",
            desc_value=None, suffix="T1w",
            description="Post-contrast T1w registered to DSC space",
            ce_value="GAD",
        )
        sink_outputs.append((sink, "out_file"))
    if "flair_in_dsc" in present:
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name="flair_in_dsc",
            in_attr="flair_in_dsc",
            desc_value=None, suffix="FLAIR",
            description="FLAIR registered to DSC space",
        )
        sink_outputs.append((sink, "out_file"))
    if "t2w_in_dsc" in present:
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name="t2w_in_dsc",
            in_attr="t2w_in_dsc",
            desc_value=None, suffix="T2w",
            description="T2w registered to DSC space",
        )
        sink_outputs.append((sink, "out_file"))

    # --- Transform .mat files ---
    if "dsc_to_t1w_xfm" in present:
        sink = _add_xfm_sink(
            workflow, inputnode,
            name="dsc_to_t1w_xfm",
            in_attr="dsc_to_t1w_xfm",
            from_space="dscref", to_space="T1w",
        )
        sink_outputs.append((sink, "out_file"))
    if "t1w_to_dsc_xfm" in present:
        sink = _add_xfm_sink(
            workflow, inputnode,
            name="t1w_to_dsc_xfm",
            in_attr="t1w_to_dsc_xfm",
            from_space="T1w", to_space="dscref",
        )
        sink_outputs.append((sink, "out_file"))

    # --- Smoothed maps (variable cardinality → MapNode) ---
    if has_smoothing:
        splitter = pe.Node(
            niu.Function(
                input_names=["smooth_hemo", "smooth_auc", "fwhm"],
                output_names=[
                    "in_files", "descs", "map_keys",
                    "map_suffixes", "desc_prefixes", "units_list",
                ],
                function=_split_smooth_dict,
            ),
            name="smoothed_split",
        )
        workflow.connect([
            (inputnode, splitter, [
                ("smooth_hemodynamic", "smooth_hemo"),
                ("smooth_auc", "smooth_auc"),
                ("smooth_fwhm", "fwhm"),
            ]),
        ])

        # MapNode meta-builder over each smoothed file
        sm_meta = pe.MapNode(
            niu.Function(
                input_names=["base", "description", "map_key", "qc_metrics",
                             "units", "fwhm"],
                output_names=["meta_dict"],
                function=_build_smoothed_meta,
            ),
            iterfield=["description", "map_key", "units"],
            name="smoothed_meta",
        )
        workflow.connect([
            (base_perf, sm_meta, [("meta_dict", "base")]),
            (inputnode, sm_meta, [
                ("qc_metrics", "qc_metrics"),
                ("smooth_fwhm", "fwhm"),
            ]),
            (splitter, sm_meta, [
                ("descs", "description"),
                ("map_keys", "map_key"),
                ("units_list", "units"),
            ]),
        ])

        sm_sink = pe.MapNode(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype",
                    "bids_suffix",
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
                ("desc_prefixes", "desc"),
                ("map_suffixes", "suffix"),
            ]),
            (sm_meta, sm_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((sm_sink, "out_file"))

    # --- DSC figures writer (figures, AIF curve, confounds TSV/JSON) ---
    figures_writer = pe.Node(DSCFiguresWriter(), name="figures_writer")
    workflow.connect([
        (inputnode, figures_writer, [
            ("output_dir", "output_dir"),
            ("subject_id", "subject_id"),
            ("session_id", "session_id"),
            ("qc_metrics", "qc_metrics"),
            ("aif_curve", "aif_curve"),
            ("motion_params", "motion_params"),
            ("bids_dir", "bids_dir"),
            ("source_dsc", "source_dsc"),
            ("brain_mask_filmstrip", "brain_mask_filmstrip"),
            ("motion_qc_plot", "motion_qc_plot"),
            ("aif_curve_plot", "aif_curve_plot"),
            ("aif_location_plot", "aif_location_plot"),
            ("aif_candidates_plot", "aif_candidates_plot"),
            ("registration_filmstrip", "registration_filmstrip"),
            ("dsc_preprocessing_qc_plot", "dsc_preprocessing_qc_plot"),
            ("dsc_param_maps_qc_plot", "dsc_param_maps_qc_plot"),
            ("dsc_fit_qc_plot", "dsc_fit_qc_plot"),
            ("dsc_leakage_qc_plot", "dsc_leakage_qc_plot"),
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
