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

from dataclasses import dataclass
from typing import Callable, Optional

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.config.constants import HDGLIO_BIDS_LABELS
from pmriprep.interfaces.qc_writers import (
    DSCFiguresWriter,
    PMRIPrepDataSink,
    write_dataset_description_node,
)


# ---------------------------------------------------------------------------
# Output declarative tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSCMapSpec:
    """Declarative specification for one regular DSC perfusion-map sink."""

    in_attr: str
    desc: str
    suffix: str
    description: str
    units: str | None
    map_key: str
    extra: dict[str, object] | None = None


@dataclass(frozen=True)
class DSCAnatSpec:
    """Declarative specification for one anatomical DSC-space sink."""

    in_attr: str
    desc: str | None
    suffix: str
    description: str
    skull_stripped: bool = False
    mask_type: str | None = None
    ceagent: str | None = None
    dismiss_entities: tuple[str, ...] = ("acquisition",)
    source_attr: str = "source_dsc"
    segmentation_labels: dict[str, str] | None = None
    is_subtraction: bool = False


# Always-present DSC perfusion maps.
_CORE_MAPS: tuple[DSCMapSpec, ...] = (
    DSCMapSpec(
        "mtt_map", "dsc", "mtt", "Mean transit time", "s", "mtt",
        {
            "PostFitClipping": "none",
            "UndefinedValue": (
                "NaN when MTT is negative/non-finite or, for SVD-family methods, "
                "when AUC-CBV/CBF is undefined"
            ),
            "LowFlowCaution": (
                "AUC-CBV/CBF becomes unstable as positive CBF approaches zero; "
                "no unvalidated amplitude threshold is applied"
            ),
        },
    ),
    DSCMapSpec(
        "tmax_map", "dsc", "tmax", "Time to maximum of residue function",
        "s", "tmax",
        {
            "Clipping": "none",
            "NegativeValues": (
                "Valid signed delay estimates for oSVD and parametric models; "
                "sSVD and Tikhonov Tmax are non-negative by construction"
            ),
            "TimingConvention": (
                "Residue/model delay; unlike tissue concentration TTP, a negative "
                "value is retained as an estimate rather than marked undefined"
            ),
        },
    ),
    DSCMapSpec(
        "psr_map",
        "dsc",
        "psr",
        "Percent signal recovery from the raw, non-leakage-corrected signal",
        "%",
        "psr",
        {
            "Estimator": "100 * (S1 - Smin) / (S0 - Smin)",
            "BaselineSignal": "mean over detected pre-bolus volumes",
            "RecoverySignal": (
                "mean over final acquired volumes; resolved count recorded in PipelineQC"
            ),
            "MinimumSignal": "minimum over post-baseline dynamic series",
            "MinimumSignalDrop": (
                "greater than 5 times the larger of voxelwise baseline noise and the "
                "median positive baseline-noise estimate within the brain mask"
            ),
            "UndefinedValue": "NaN when raw signal is unavailable or signal drop is indeterminate",
            "ProtocolDependence": (
                "PSR depends on acquisition duration, TR, TE, flip angle, dose, and preload; "
                "thresholds require a matching acquisition and processing convention"
            ),
        },
    ),
)


# Relative maps (always produced, but inputs may be undefined). The
# bids_suffix/desc mapping defines the public derivative naming contract.
_REL_MAPS: tuple[DSCMapSpec, ...] = (
    DSCMapSpec("rel_cbf_map", "dscRel", "cbf",
     "Cerebral blood flow (relative)", "1/min", "cbf",
     {"PostFitClipping": "none", "UndefinedValue": "NaN for negative/non-finite estimates"}),
    DSCMapSpec("rel_cbv_map", "dscRel", "cbv",
     "Cerebral blood volume (relative, deconvolution)", "arbitrary", "cbv",
     {"CBVMethod": "deconvolution_residue_integral", "PostFitClipping": "none",
      "UndefinedValue": "NaN for negative/non-finite estimates"}),
    DSCMapSpec("rel_cbv_auc_map", "dscRelAUC", "cbv",
     "Cerebral blood volume (K2-corrected full-bolus AUC, relative)",
     "arbitrary", "cbv_auc",
     {"CBVMethod": "k2_corrected_full_bolus_auc", "PostFitClipping": "none",
      "UndefinedValue": "NaN for negative/non-finite estimates"}),
)


# Optional maps with per-map desc + bids_suffix
_OPTIONAL_MAPS: tuple[DSCMapSpec, ...] = (
    DSCMapSpec("ttp_map", "dsc", "ttp",
     "Tissue concentration time to peak relative to AIF bolus arrival.",
     "s", "ttp",
     {"ReferencePoint": "AIF bolus arrival time",
      "UndefinedValue": "NaN when tissue peak precedes AIF bolus arrival"}),
    DSCMapSpec(
        "cnr_map", "dsc", "cnr", "DSC signal-drop contrast-to-noise ratio", "1", "cnr",
        {
            "Estimator": "(S0 - Smin) / sigma0",
            "BaselineSignal": "mean over detected pre-bolus volumes",
            "MinimumSignal": "minimum over post-baseline dynamic series",
            "NoiseEstimate": (
                "larger of voxelwise baseline standard deviation and the median positive "
                "baseline standard deviation within the brain mask"
            ),
            "UndefinedValue": "NaN when raw signal or a positive signal drop is unavailable",
        },
    ),
    DSCMapSpec(
        "tsnr_map", "dsc", "tsnr", "Baseline temporal signal-to-noise ratio", "1", "tsnr",
        {
            "Estimator": "mean(S0) / sample_standard_deviation(S0)",
            "TemporalWindow": "detected pre-bolus volumes only",
            "UndefinedValue": "NaN when raw signal or baseline variance is unavailable",
            "PrecisionNote": (
                "Voxelwise precision depends on the number of detected baseline volumes"
            ),
        },
    ),
)


# Normalized maps
_NORMALIZED_MAPS: tuple[DSCMapSpec, ...] = (
    DSCMapSpec("ncbv_map", "dscNawmNorm", "cbv",
     "Normalized CBV (NAWM-normalized)", "1", "ncbv"),
    DSCMapSpec("ncbf_map", "dscNawmNorm", "cbf",
     "Normalized CBF (NAWM-normalized)", "1", "ncbf"),
)


# Vascular-model physiological derivative
_VASCULAR_MAPS: tuple[DSCMapSpec, ...] = (
    DSCMapSpec(
        "cth_map", "dsc", "cth", "Capillary transit time heterogeneity", "s", "cth",
        {
            "Estimator": "standard_deviation_of_gamma_transit_time_distribution",
            "Formula": "beta * sqrt(alpha) = MTT / sqrt(alpha)",
            "ModelDependency": "vascular_gamma_transit_time_model",
            "UndefinedValue": (
                "NaN when the vascular fit is unavailable, replaced by oSVD, "
                "or alpha is at either fitted bound"
            ),
        },
    ),
)


# CBF uncertainty maps. CBV is derived and lacks retained joint covariance.
_UNCERTAINTY_MAPS: tuple[DSCMapSpec, ...] = (
    DSCMapSpec("cbf_std_map", "dscUncertainty", "cbf", "CBF absolute parameter uncertainty", "1/min", "cbf_std"),
)


_ANAT_MASKS: tuple[DSCAnatSpec, ...] = (
    DSCAnatSpec(
        "brain_mask_dsc", "brain", "mask", "Brain mask in DSC space",
        skull_stripped=True, mask_type="Brain",
        dismiss_entities=("acquisition", "ceagent"),
    ),
    DSCAnatSpec(
        "nawm_mask_dsc", "nawm", "mask",
        "Normal-appearing white matter mask in DSC space",
        mask_type="ROI", dismiss_entities=(), source_attr="source_t1w",
    ),
    DSCAnatSpec(
        "lesion_mask_dsc", "lesion", "dseg",
        "Tumor segmentation registered to DSC space",
        dismiss_entities=(), source_attr="source_t1w",
        segmentation_labels=dict(HDGLIO_BIDS_LABELS),
    ),
    DSCAnatSpec(
        "user_lesion_mask_dsc", "userLesion", "mask",
        "User-supplied lesion mask in DSC native space", mask_type="Lesion",
    ),
    DSCAnatSpec(
        "roi_mask_dsc", "roi", "mask", "User-supplied ROI mask in DSC native space",
        mask_type="ROI",
    ),
)


_STRUCTURAL_UNDERLAYS: tuple[DSCAnatSpec, ...] = (
    DSCAnatSpec(
        "t1w_in_dsc", None, "T1w", "Pre-contrast T1w registered to DSC space",
        dismiss_entities=(), source_attr="source_t1w",
    ),
    DSCAnatSpec(
        "t1w_postcon_in_dsc", None, "T1w",
        "Post-contrast T1w registered to DSC space", ceagent="GAD",
        dismiss_entities=(), source_attr="source_t1w_postcon",
    ),
    DSCAnatSpec(
        "t1w_subtraction_in_dsc", "subtraction", "T1w",
        "T1w subtraction (post-pre raw-signal difference) registered to DSC space",
        dismiss_entities=(), source_attr="source_t1w", is_subtraction=True,
    ),
    DSCAnatSpec(
        "flair_in_dsc", None, "FLAIR", "FLAIR registered to DSC space",
        dismiss_entities=(), source_attr="source_flair",
    ),
    DSCAnatSpec(
        "flair_postcon_in_dsc", None, "FLAIR",
        "Post-contrast FLAIR registered to DSC space", ceagent="GAD",
        dismiss_entities=(), source_attr="source_flair_postcon",
    ),
    DSCAnatSpec(
        "t2w_in_dsc", None, "T2w", "T2w registered to DSC space",
        dismiss_entities=(), source_attr="source_t2w",
    ),
)


#: Canonical list of map_keys passed to ``apply_aif_sidecar`` from the factory.
#: Used by the drift-guard test in ``test_dsc_outputs_sidecar.py`` to verify
#: every emitted key is registered in :data:`TEMPLATE_AIF_USAGE`.
_DSC_OUTPUT_KEYS: frozenset[str] = frozenset(
    [m.map_key for m in _CORE_MAPS + _REL_MAPS + _OPTIONAL_MAPS
     + _NORMALIZED_MAPS + _VASCULAR_MAPS + _UNCERTAINTY_MAPS]
    + ["parametric_r2", "fallback",
       "rel_cbf", "rel_cbv", "ncbf", "ncbv", "mtt"]  # smoothed names
)


# ---------------------------------------------------------------------------
# Function-node wrappers
# ---------------------------------------------------------------------------


def _build_base(
    source_paths: list[str],
    reference_voxel_source: str,
    bids_dir: str,
    subject: str,
    session: str,
    provenance_dict: dict | None = None,
    qc_metrics: dict | None = None,
    minimal: bool = False,
) -> dict:
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


def _build_map_meta(
    base: dict,
    description: str,
    map_key: str,
    qc_metrics: dict | None = None,
    units: str | None = None,
    extra: dict | None = None,
    lesion_stats_json: str | None = None,
    user_lesion_stats_json: str | None = None,
    roi_stats_json: str | None = None,
) -> dict:
    """Compose a DSC per-map sidecar; injects per-map stats fragments.

    Each aggregate stats JSON (lesion, user-lesion, ROI) is loaded when
    present and split into per-map fragments via
    :func:`extract_per_map_lesion_stats`. Fragments are merged into
    the ``extra`` dict as ``LesionStats``, ``UserLesionStats``, and
    ``ROIStats`` alongside any caller-provided extras.
    """
    from pmriprep.utils.lesion_stats import load_lesion_stats_fragment
    from pmriprep.utils.sidecar_builders import build_dsc_map_sidecar

    fragment = load_lesion_stats_fragment(lesion_stats_json, map_key)
    user_fragment = load_lesion_stats_fragment(user_lesion_stats_json, map_key)

    merged_extra = dict(extra) if extra else {}
    if map_key in {"cbv_auc", "ncbv"}:
        from pmriprep.workflows.dsc.datasinks import (
            _build_signed_auc_cbv_provenance,
        )

        merged_extra["SignedEstimator"] = _build_signed_auc_cbv_provenance(
            qc_metrics,
            normalized=map_key == "ncbv",
        )
    if map_key == "fallback":
        merged_extra["Type"] = "ROI"
    if map_key == "cbf_std" and isinstance(qc_metrics, dict):
        deconvolution = qc_metrics.get("deconvolution")
        if isinstance(deconvolution, dict):
            accounting = deconvolution.get("map_uncertainty")
            if isinstance(accounting, dict):
                merged_extra["MAPUncertaintyAccounting"] = dict(accounting)
    if fragment:
        merged_extra["LesionStats"] = fragment
    if user_fragment:
        merged_extra["UserLesionStats"] = user_fragment
    roi_fragment = load_lesion_stats_fragment(roi_stats_json, map_key)
    if roi_fragment:
        merged_extra["ROIStats"] = roi_fragment
    return build_dsc_map_sidecar(
        base=base,
        description=description,
        map_key=map_key,
        qc_metrics=qc_metrics,
        units=units,
        extra=merged_extra or extra,
    )


def _build_minimal_meta(
    base: dict,
    description: str,
    skull_stripped: bool = False,
    mask_type: str | None = None,
    segmentation_labels: dict[str, str] | None = None,
) -> dict:
    out = dict(base)
    out["Description"] = description
    if skull_stripped:
        out["SkullStripped"] = True
    if mask_type is not None:
        out["Type"] = mask_type
    if segmentation_labels:
        out["LabelMap"] = dict(segmentation_labels)
    return out


def _build_native_mask_meta(
    base: dict,
    description: str,
    skull_stripped: bool,
    mask_type: str | None,
    native_source: str,
    perfusion_source: str,
    bids_dir: str,
    segmentation_labels: dict[str, str] | None = None,
) -> dict:
    """Build metadata for a validated user mask in native DSC space."""
    from pmriprep.utils.sidecar_builders import build_sources_list

    out = dict(base)
    out["Description"] = description
    if skull_stripped:
        out["SkullStripped"] = True
    if mask_type is not None:
        out["Type"] = mask_type
    if segmentation_labels:
        out["LabelMap"] = dict(segmentation_labels)
    out.update({
        "ResamplingApplied": False,
        "Interpolation": "none",
        "ReorientationApplied": True,
        "GeometryValidation": "shape_and_affine_against_DSC_reference",
    })
    out["Sources"] = build_sources_list(
        [native_source, perfusion_source], bids_dir,
    )
    return out


def _build_aif_mask_meta(
    base: dict,
    description: str,
    qc_metrics: dict | None,
) -> dict:
    """Build an AIF ROI sidecar with the canonical DSC AIF block."""
    from pmriprep.utils.aif_provenance import build_aif_provenance_block

    out = dict(base)
    out["Description"] = description
    out["Type"] = "ROI"
    aif_block = build_aif_provenance_block(qc_metrics, modality="DSC")
    if aif_block:
        out.setdefault("PerfusionProcessing", {})["AIF"] = aif_block
    source = aif_block.get("Source")
    if source:
        spatial_support_available = source in {
            "measured", "gamma_fit_measured",
        }
        out["SpatialSupportAvailable"] = spatial_support_available
        if not spatial_support_available:
            out["EmptyMaskMeaning"] = (
                "No spatial AIF voxels were selected for this non-measured curve"
            )
    return out


def _build_underlay_meta(
    base: dict,
    description: str,
    skull_stripped: bool,
    mask_type: str | None,
    structural_source: str,
    perfusion_source: str,
    bids_dir: str,
    segmentation_labels: dict[str, str] | None = None,
    is_subtraction: bool = False,
    postcontrast_source: str = "",
) -> dict:
    """Build metadata for a structural image resampled to DSC space."""
    from pmriprep.utils.sidecar_builders import (
        build_sources_list,
        subtraction_signal_metadata,
    )

    out = dict(base)
    out["Description"] = description
    if skull_stripped:
        out["SkullStripped"] = True
    if mask_type is not None:
        out["Type"] = mask_type
    if segmentation_labels:
        out["LabelMap"] = dict(segmentation_labels)
    out.update({
        "ResamplingApplied": True,
        "Interpolation": "spline",
        "TransformDirection": "T1w_to_DSC",
    })
    sources = [structural_source]
    if is_subtraction:
        sources.append(postcontrast_source)
    sources.append(perfusion_source)
    out["Sources"] = build_sources_list(sources, bids_dir)
    if is_subtraction:
        out.update(subtraction_signal_metadata())
    return out


def _build_transformed_mask_meta(
    base: dict,
    description: str,
    skull_stripped: bool,
    mask_type: str | None,
    structural_sources: list[str],
    perfusion_source: str,
    bids_dir: str,
    segmentation_labels: dict[str, str] | None = None,
    resampling_applied: bool = True,
    native_source: str | None = None,
) -> dict:
    """Build metadata for an anatomical mask transformed to DSC space."""
    from pmriprep.utils.sidecar_builders import build_sources_list

    out = dict(base)
    out["Description"] = description
    if skull_stripped:
        out["SkullStripped"] = True
    if mask_type is not None:
        out["Type"] = mask_type
    if segmentation_labels:
        out["LabelMap"] = dict(segmentation_labels)
    if resampling_applied:
        out.update({
            "ResamplingApplied": True,
            "Interpolation": "nearest_neighbour",
            "TransformDirection": "T1w_to_DSC",
            "LabelPreservation": True,
        })
    else:
        out.update({
            "ResamplingApplied": False,
            "Interpolation": "none",
            "ReorientationApplied": True,
            "GeometryValidation": "shape_and_affine_against_DSC_reference",
        })
    source_paths = (
        [native_source, perfusion_source]
        if not resampling_applied and native_source
        else [*(structural_sources or []), perfusion_source]
    )
    out["Sources"] = build_sources_list(
        source_paths, bids_dir,
    )
    return out


def _build_dsc_preproc_meta(
    base: dict,
    description: str,
    preload_qc: dict | None = None,
    preprocessing_stats: dict | None = None,
    qc_metrics: dict | None = None,
    provenance_dict: dict | None = None,
    preflight_warnings: list | None = None,
    presence_warnings: list | None = None,
    config_dict: dict | None = None,
) -> dict:
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
        sanitize_for_json,
    )

    out = dict(base)
    out["Description"] = description
    out["SkullStripped"] = False
    if isinstance(config_dict, dict):
        out["MotionCorrectionApplied"] = bool(
            config_dict.get("motion_correct", True)
        )
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
    return dict(sanitize_for_json(out))


def _build_dsc_mean_meta(
    base: dict,
    preprocessing_stats: dict | None = None,
    motion_correct: bool = True,
) -> dict:
    """Describe the all-volume DSC temporal mean without implying baseline contrast."""
    out = dict(base)
    out.update({
        "Description": "Temporal mean across all acquired DSC volumes",
        "TemporalAggregation": "arithmetic_mean_all_acquired_volumes",
        "IncludesPostContrastVolumes": True,
        "MotionCorrectionApplied": bool(motion_correct),
        "SkullStripped": False,
    })
    if isinstance(preprocessing_stats, dict):
        n_volumes = preprocessing_stats.get("n_volumes")
        if n_volumes is not None:
            out["NumberOfVolumes"] = int(n_volumes)
        baseline_frames = preprocessing_stats.get(
            "baseline_frames_used",
            preprocessing_stats.get("baseline_frames_detected"),
        )
        if baseline_frames is not None:
            out["BaselineFrames"] = int(baseline_frames)
    return out


def _write_dsc_resume_checkpoint(
    work_dir: str,
    subject_id: str,
    session_id: str,
    aif_qc: dict | None = None,
    preload_qc: dict | None = None,
    preprocessing_stats: dict | None = None,
    motion_qc: dict | None = None,
    aif_metrics: dict | None = None,
) -> str:
    """Write the DSC→DCE handoff checkpoint to the work directory."""
    import json as _json
    from pathlib import Path

    if not isinstance(work_dir, str):
        return ""

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


def _copy_xfm(
    in_file: str,
    output_dir: str,
    subject: str,
    session: str,
    from_space: str,
    to_space: str,
    source_file: str | None = None,
    datatype: str = "anat",
) -> str:
    """Copy a transform .h5 into the derivatives ``anat/`` directory."""
    import logging
    import shutil
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    # Defensive no-op when upstream registration produced no transform.
    # See pmriprep.workflows.dce.datasinks._copy_perf_map for the
    # sentinel-handling rationale.
    if not isdefined(in_file) or in_file == "":
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

    from pmriprep.utils.sidecar_builders import source_entity_prefix

    prefix = source_entity_prefix(source_file, subject, session)
    fname = f"{prefix}_from-{from_space}_to-{to_space}_mode-image_xfm.h5"
    out_path = out_root / fname
    shutil.copy2(in_file, out_path)
    return str(out_path)


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
) -> str:
    """Copy a DSC perfusion map + write its sidecar with BIDS-compliant naming.

    ``suffix`` is the parameter name (e.g. ``"mtt"``, ``"cbf"``) which
    gets folded into the ``desc-`` entity.  ``bids_suffix`` is the
    BIDS-canonical suffix (``stat``, ``mask``, ``dseg``, …).  Pass
    ``suffix=""`` when ``desc`` already encodes everything and only the
    BIDS suffix is needed.
    """
    import logging
    import shutil
    from pathlib import Path

    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    # Defensive no-op when upstream conditional producer returned no
    # output. See pmriprep.workflows.dce.datasinks._copy_perf_map for
    # the sentinel-handling rationale (Undefined/None/"" all need
    # explicit handling because isdefined(None) == True is a Nipype
    # gotcha).
    if not isdefined(in_file) or in_file == "":
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

    from pmriprep.utils.sidecar_builders import source_entity_prefix

    sources = meta_dict.get("Sources", []) if isinstance(meta_dict, dict) else []
    prefix = source_entity_prefix(sources[0] if sources else None, subject, session)
    folded = suffix[:1].upper() + suffix[1:] if suffix else ""
    stem = f"{prefix}_desc-{desc}{folded}_{bids_suffix}"
    nii_out = out_root / f"{stem}.nii.gz"
    json_out = out_root / f"{stem}.json"

    shutil.copy2(in_file, nii_out)
    from pmriprep.utils.sidecar_builders import dump_sidecar_json

    json_out.write_text(dump_sidecar_json(meta_dict))
    return str(nii_out)


def _copy_aif_mask(
    in_file: str,
    output_dir: str,
    subject: str,
    session: str,
    desc: str,
    suffix: str,
    meta_dict: dict,
    datatype: str = "perf",
    bids_suffix: str = "mask",
) -> str:
    """Write an AIF support mask only when the AIF has spatial support."""
    import logging

    if isinstance(meta_dict, dict) and meta_dict.get("SpatialSupportAvailable") is False:
        logging.getLogger("nipype.workflow").info(
            "Skipping DSC AIF mask derivative: the selected AIF has no spatial support"
        )
        return ""

    from pmriprep.workflows.dsc.datasinks import _copy_perf_map

    return _copy_perf_map(
        in_file=in_file,
        output_dir=output_dir,
        subject=subject,
        session=session,
        desc=desc,
        suffix=suffix,
        meta_dict=meta_dict,
        datatype=datatype,
        bids_suffix=bids_suffix,
    )


def _split_smooth_dict(
    smooth_hemo: dict[str, str] | None = None,
    fwhm: float = 0.0,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Flatten the smoothed-map dict into per-output ``(name, path, descs)`` tuples.

    Returns six lists: ``in_files``, ``descs``, ``map_keys``,
    ``map_suffixes``, ``desc_prefixes``, and ``units_list`` matching the
    derivative naming templates (``dsc{Sm{N}}``, ``dscRel{Sm{N}}``, etc.).
    """
    from pathlib import Path
    from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

    out_files: list[str] = []
    out_descs: list[str] = []
    out_keys: list[str] = []
    out_map_suffixes: list[str] = []
    out_desc_prefixes: list[str] = []
    out_units: list[str] = []

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
                 f"dscNawmNorm{sm_label}", "cbf", "1"),
        "ncbv": ("Normalized CBV (smoothed)",
                 f"dscNawmNorm{sm_label}", "cbv", "1"),
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

    return out_files, out_descs, out_keys, out_map_suffixes, out_desc_prefixes, out_units


def _build_parametric_extra(qc_metrics: dict | None = None) -> dict:
    """Build parametric-fit method, fallback, and denominator provenance."""
    extra: dict[str, object] = {
        "FallbackMinimumRSquared": 0.0,
        "FallbackMinimumRSquaredInclusive": True,
    }
    if isinstance(qc_metrics, dict):
        deconv = qc_metrics.get("deconvolution") or {}
        if isinstance(deconv, dict):
            if deconv.get("method"):
                extra["FitMethod"] = str(deconv["method"])
            if deconv.get("inference"):
                extra["InferenceEngine"] = str(deconv["inference"])
            for source, target in (
                ("n_mask_voxels", "MaskVoxelCount"),
                ("n_fit_admissible", "FitAdmissibleVoxelCount"),
                ("n_fit_inadmissible", "FitInadmissibleVoxelCount"),
                ("n_fallback_requested", "FallbackRequestedVoxelCount"),
                ("n_fallback_attempted", "FallbackAttemptedVoxelCount"),
                ("n_fallback_osvd", "FallbackSucceededVoxelCount"),
                ("n_fallback_failed", "FallbackFailedVoxelCount"),
                ("fallback_requested_rate", "FallbackRequestedFraction"),
                ("fallback_rate", "FallbackSucceededFraction"),
            ):
                if source in deconv:
                    extra[target] = deconv[source]
    return extra


def _build_uncertainty_extra(
    config_dict: dict | None = None,
    umap_key: str = "cbf_std",
) -> dict:
    """Build inference-specific provenance for CBF uncertainty maps."""
    from pmriprep.utils.uncertainty_provenance import (
        build_uncertainty_provenance,
    )

    estimation = "unknown"
    if isinstance(config_dict, dict):
        estimation = config_dict.get("inference") or "unknown"
    return build_uncertainty_provenance(
        inference=str(estimation), map_key=umap_key, modality="DSC"
    )


def _build_smoothed_meta(
    base: dict,
    description: str,
    map_key: str,
    qc_metrics: dict | None,
    units: str | None,
    fwhm: float,
) -> dict:
    from pmriprep.utils.sidecar_builders import build_dsc_map_sidecar
    from pmriprep.workflows.dsc.datasinks import (
        _build_signed_auc_cbv_provenance,
    )

    extra = {
        "SmoothingFWHM": float(fwhm),
        "SmoothingType": "Gaussian",
        "SmoothingSupport": "finite source voxels within the brain mask",
        "InvalidSupportPreserved": True,
        "OutputMaskApplied": True,
        "IntendedFor": "visualization",
        "Note": "Use non-smoothed maps for quantitative ROI analysis",
    }
    if map_key in {"rel_cbv", "ncbv"}:
        extra["SignedEstimator"] = _build_signed_auc_cbv_provenance(
            qc_metrics,
            normalized=map_key == "ncbv",
            include_summary=False,
        )
    return build_dsc_map_sidecar(
        base=base,
        description=description,
        map_key=map_key,
        qc_metrics=qc_metrics,
        units=units or None,
        extra=extra,
    )


def _build_signed_auc_cbv_provenance(
    qc_metrics: dict | None,
    normalized: bool = False,
    include_summary: bool = True,
) -> dict[str, object]:
    """Describe AUC-CBV validity handling and raw negative-estimate frequency."""
    provenance: dict[str, object] = {
        "ValueDomain": "nonnegative_or_NaN",
        "NegativeValuesRetained": False,
        "NegativeValueHandling": "encoded_as_NaN",
        "NegativeValueInterpretation": (
            "Non-physiological estimator residuals from noise, baseline error, "
            "motion, or leakage correction"
        ),
    }
    if normalized:
        provenance["NormalizationReferenceDomain"] = (
            "finite_nonnegative_NAWM_with_positive_median_required"
        )
    if not include_summary or not isinstance(qc_metrics, dict):
        return provenance
    deconvolution = qc_metrics.get("deconvolution")
    if not isinstance(deconvolution, dict):
        return provenance
    keys = {
        "cbv_auc_finite_voxels": "FiniteInBrainVoxelCount",
        "cbv_auc_negative_voxels": "NegativeInBrainVoxelCount",
        "cbv_auc_negative_fraction": "NegativeInBrainVoxelFraction",
    }
    for source, target in keys.items():
        value = deconvolution.get(source)
        if value is not None:
            provenance[target] = value
    if "NegativeInBrainVoxelFraction" in provenance:
        provenance["NegativeFractionSource"] = "pre_sanitization_AUC_CBV_estimator"
        provenance["NegativeFractionDenominator"] = (
            "finite_pre_sanitization_in_brain_voxels"
        )
    return provenance


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
                "extra", "lesion_stats_json", "user_lesion_stats_json",
                "roi_stats_json",
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
            ("user_lesion_stats_json", "user_lesion_stats_json"),
            ("roi_stats_json", "roi_stats_json"),
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
    mask_type: str | None = None,
    ce_value: Optional[str] = None,
    source_attr: str = "source_dsc",
    dismiss_entities: tuple = ("acquisition",),
    segmentation_labels: dict[str, str] | None = None,
    structural_sources_node: pe.Node | None = None,
    resampling_applied: bool = True,
    native_source: str | None = None,
    is_subtraction: bool = False,
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
    is_underlay = source_attr != "source_dsc"
    is_transformed_mask = is_underlay and suffix in {"mask", "dseg"}
    is_native_user_mask = (
        not is_underlay
        and in_attr in {"user_lesion_mask_dsc", "roi_mask_dsc"}
    )
    meta_function: Callable[..., dict]
    if is_transformed_mask:
        meta_input_names = [
            "base", "description", "skull_stripped", "mask_type",
            "structural_sources", "perfusion_source", "bids_dir",
            "segmentation_labels", "resampling_applied", "native_source",
        ]
        meta_function = _build_transformed_mask_meta
    elif is_underlay:
        meta_input_names = [
            "base", "description", "skull_stripped", "mask_type",
            "structural_source", "postcontrast_source", "perfusion_source", "bids_dir",
            "segmentation_labels", "is_subtraction",
        ]
        meta_function = _build_underlay_meta
    elif is_native_user_mask:
        meta_input_names = [
            "base", "description", "skull_stripped", "mask_type",
            "native_source", "perfusion_source", "bids_dir",
            "segmentation_labels",
        ]
        meta_function = _build_native_mask_meta
    else:
        meta_input_names = [
            "base", "description", "skull_stripped", "mask_type",
            "segmentation_labels",
        ]
        meta_function = _build_minimal_meta
    meta = pe.Node(
        niu.Function(
            input_names=meta_input_names,
            output_names=["meta_dict"],
            function=meta_function,
        ),
        name=f"{name}_meta",
    )
    meta.inputs.description = description
    meta.inputs.skull_stripped = skull_stripped
    meta.inputs.mask_type = mask_type
    meta.inputs.segmentation_labels = segmentation_labels
    if is_underlay and not is_transformed_mask:
        meta.inputs.is_subtraction = is_subtraction
    if is_transformed_mask:
        meta.inputs.resampling_applied = resampling_applied
        meta.inputs.native_source = native_source
    elif is_native_user_mask:
        meta.inputs.native_source = native_source
        workflow.connect(inputnode, "source_dsc", meta, "perfusion_source")
        workflow.connect(inputnode, "bids_dir", meta, "bids_dir")
    workflow.connect([(base_minimal_node, meta, [("meta_dict", "base")])])
    if is_underlay:
        workflow.connect(inputnode, "source_dsc", meta, "perfusion_source")
        workflow.connect(inputnode, "bids_dir", meta, "bids_dir")
        if is_transformed_mask:
            if structural_sources_node is None:
                raise ValueError(
                    "structural_sources_node is required for anatomical masks"
                )
            workflow.connect(structural_sources_node, "out", meta, "structural_sources")
        else:
            workflow.connect(inputnode, source_attr, meta, "structural_source")
            workflow.connect(
                inputnode, "source_t1w_postcon", meta, "postcontrast_source",
            )

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
                         "from_space", "to_space", "source_file", "datatype"],
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
            (source_attr, "source_file"),
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
    "mtt_map", "tmax_map", "psr_map",
})


def init_dsc_datasinks_wf(
    *,
    output_dir: str,
    bids_dir: str,
    subject_id: str,
    session_id: Optional[str],
    inference: str,
    present_inputs: Optional[set[str]] = None,
    has_smoothing: bool = False,
    motion_correct: bool = True,
    nawm_needs_transform: bool = True,
    nawm_source_file: str | None = None,
    user_lesion_source_file: str | None = None,
    roi_source_file: str | None = None,
    name: str = "dsc_datasinks_wf",
) -> pe.Workflow:
    """Build the DSC-pipeline DerivativesDataSink chain.

    ``present_inputs`` is a set of inputnode field names that will be
    wired by the caller. Sinks for absent inputs are NOT created (avoids
    nipype's mandatory-input check failing). The mandatory-perfusion
    inputs (``mtt_map``, ``tmax_map``, ``psr_map``,
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
                "mtt_map", "tmax_map", "psr_map",
                # Relative
                "rel_cbf_map", "rel_cbv_map", "rel_cbv_auc_map",
                # Optional
                "ttp_map",
                "cnr_map", "tsnr_map",
                # Normalized
                "ncbv_map", "ncbf_map",
                # Vascular
                "cth_map",
                # Inference-specific CBF uncertainty
                "cbf_std_map",
                # Parametric diagnostics
                "parametric_r2_map", "fallback_mask",
                "preload_qc", "preprocessing_stats", "aif_qc",
                "motion_qc", "aif_metrics",
                # Smoothing
                "smooth_hemodynamic", "smooth_fwhm",
                # Motion + AIF
                "dsc_moco", "dsc_mean", "motion_params",
                "aif_curve", "aif_mask",
                # Anat-side: masks + structural underlays + xfms + lesion/roi
                "brain_mask_dsc", "nawm_mask_dsc",
                "lesion_mask_dsc", "lesion_stats_json",
                "user_lesion_mask_dsc", "user_lesion_stats_json",
                "roi_mask_dsc", "roi_stats_json",
                "t1w_in_dsc", "t1w_postcon_in_dsc", "t1w_subtraction_in_dsc",
                "flair_in_dsc", "flair_postcon_in_dsc", "t2w_in_dsc",
                "source_t1w", "source_t1w_postcon",
                "source_flair", "source_flair_postcon", "source_t2w",
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
        for spec in table:
            if spec.in_attr not in present:
                continue
            sink = _add_perf_map_sink(
                workflow, inputnode, base_perf,
                name=spec.in_attr,
                in_attr=spec.in_attr,
                desc_value=spec.desc, suffix=spec.suffix,
                description=spec.description, map_key=spec.map_key,
                units=spec.units, extra=spec.extra,
            )
            sink_outputs.append((sink, "out_file"))

    # --- Inference-specific CBF uncertainty maps ---
    for spec in _UNCERTAINTY_MAPS:
        if spec.in_attr not in present:
            continue
        bayes_extra = pe.Node(
            niu.Function(
                input_names=["config_dict", "umap_key"],
                output_names=["extra"],
                function=_build_uncertainty_extra,
            ),
            name=f"{spec.in_attr}_extra",
        )
        bayes_extra.inputs.umap_key = spec.map_key
        workflow.connect(inputnode, "config_dict", bayes_extra, "config_dict")
        bayes_meta = pe.Node(
            niu.Function(
                input_names=[
                    "base", "description", "map_key", "qc_metrics", "units", "extra",
                ],
                output_names=["meta_dict"],
                function=_build_map_meta,
            ),
            name=f"{spec.in_attr}_meta",
        )
        is_nlls = inference.lower() == "nlls"
        desc_value = "dscStandardError" if is_nlls else "dscPosteriorSD"
        description = (
            "CBF parameter standard error"
            if is_nlls
            else "CBF posterior standard deviation"
        )
        bayes_meta.inputs.description = description
        bayes_meta.inputs.map_key = spec.map_key
        bayes_meta.inputs.units = spec.units
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
            name=f"{spec.in_attr}_sink",
        )
        b_sink.inputs.desc = desc_value
        b_sink.inputs.suffix = spec.suffix
        b_sink.inputs.datatype = "perf"
        b_sink.inputs.bids_suffix = "stat"
        workflow.connect([
            (inputnode, b_sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                (spec.in_attr, "in_file"),
            ]),
            (bayes_meta, b_sink, [("meta_dict", "meta_dict")]),
        ])
        sink_outputs.append((b_sink, "out_file"))
    # --- Parametric R2 + fallback mask ---
    # R2 is preserved on fallback voxels, while their CBF/CBV/MTT/Tmax values
    # come from oSVD and their parametric uncertainty maps are undefined.
    _PARAMETRIC_R2_DESC = (
        "Per-voxel R2 (coefficient of determination) of the parametric DSC "
        "fit (CPI / Vascular) against the K2-corrected tissue concentration. "
        "Raw parametric R2 is preserved on voxels that fell back to oSVD -- "
        "see dscFallback_mask to identify those. The CBF/CBV/MTT/Tmax saved "
        "in fallback voxels comes from oSVD, NOT from a fit with this R2."
    )
    _FALLBACK_DESC = (
        "Binary oSVD-fallback mask. 1 = the parametric fit was rejected "
        "(R2<0.5, solver nonconvergence, or fitting failure), oSVD fallback "
        "succeeded, and the saved CBF/CBV/MTT/Tmax come from "
        "svd_deconvolve_oscillation; parametric uncertainty is NaN in these "
        "voxels. 0 = no successful oSVD replacement; this includes retained "
        "parametric fits, inadmissible curves, and failed fallback attempts. "
        "Use the sidecar counts to distinguish these cases."
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
        # InferenceEngine + fallback adequacy criterion).
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
                    "base", "description", "map_key", "qc_metrics", "units",
                    "extra", "lesion_stats_json", "user_lesion_stats_json",
                    "roi_stats_json",
                ],
                output_names=["meta_dict"],
                function=_build_map_meta,
            ),
            name=f"{in_attr}_meta",
        )
        param_meta.inputs.description = description
        param_meta.inputs.map_key = key
        param_meta.inputs.units = "1" if in_attr == "parametric_r2_map" else None
        workflow.connect([
            (base_perf, param_meta, [("meta_dict", "base")]),
            (inputnode, param_meta, [
                ("qc_metrics", "qc_metrics"),
                ("lesion_stats_json", "lesion_stats_json"),
                ("user_lesion_stats_json", "user_lesion_stats_json"),
                ("roi_stats_json", "roi_stats_json"),
            ]),
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
        bids_sfx: str = "stat", mask_type: str | None = None,
    ) -> pe.Node:
        meta = pe.Node(
            niu.Function(
                input_names=["base", "description", "skull_stripped", "mask_type"],
                output_names=["meta_dict"],
                function=_build_minimal_meta,
            ),
            name=f"{nm}_meta",
        )
        meta.inputs.description = description
        meta.inputs.skull_stripped = False
        meta.inputs.mask_type = mask_type
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
        meta = pe.Node(
            niu.Function(
                input_names=["base", "preprocessing_stats", "motion_correct"],
                output_names=["meta_dict"],
                function=_build_dsc_mean_meta,
            ),
            name="mean_meta",
        )
        meta.inputs.motion_correct = motion_correct
        workflow.connect([
            (base_minimal, meta, [("meta_dict", "base")]),
            (inputnode, meta, [("preprocessing_stats", "preprocessing_stats")]),
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
            name="mean_sink",
        )
        sink.inputs.desc = "dscMean"
        sink.inputs.suffix = ""
        sink.inputs.datatype = "perf"
        sink.inputs.bids_suffix = "stat"
        workflow.connect([
            (inputnode, sink, [
                ("output_dir", "output_dir"),
                ("subject_id", "subject"),
                ("session_id", "session"),
                ("dsc_mean", "in_file"),
            ]),
            (meta, sink, [("meta_dict", "meta_dict")]),
        ])
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
        meta.inputs.description = "AIF spatial-support mask in DSC space"
        workflow.connect([
            (base_minimal, meta, [("meta_dict", "base")]),
            (inputnode, meta, [("qc_metrics", "qc_metrics")]),
        ])
        sink = pe.Node(
            niu.Function(
                input_names=[
                    "in_file", "output_dir", "subject", "session",
                    "desc", "suffix", "meta_dict", "datatype",
                    "bids_suffix",
                ],
                output_names=["out_file"],
                function=_copy_aif_mask,
            ),
            name="aif_mask_sink",
        )
        sink.inputs.desc = "dscAif"
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

    # Precontrast-derived masks explicitly dismiss ceagent to avoid false
    # postcontrast provenance; the distinction is encoded per specification.
    anat_mask_sources = pe.Node(niu.Merge(5), name="anat_mask_sources")
    workflow.connect([
        (inputnode, anat_mask_sources, [
            ("source_t1w", "in1"),
            ("source_t1w_postcon", "in2"),
            ("source_flair", "in3"),
            ("source_flair_postcon", "in4"),
            ("source_t2w", "in5"),
        ]),
    ])
    for anat_spec in _ANAT_MASKS + _STRUCTURAL_UNDERLAYS:
        if anat_spec.in_attr not in present:
            continue
        sink = _add_anat_sink(
            workflow, inputnode, base_minimal,
            name=anat_spec.in_attr, in_attr=anat_spec.in_attr,
            desc_value=anat_spec.desc, suffix=anat_spec.suffix,
            description=anat_spec.description,
            skull_stripped=anat_spec.skull_stripped,
            mask_type=anat_spec.mask_type,
            ce_value=anat_spec.ceagent,
            source_attr=anat_spec.source_attr,
            dismiss_entities=anat_spec.dismiss_entities,
            segmentation_labels=anat_spec.segmentation_labels,
            is_subtraction=anat_spec.is_subtraction,
            structural_sources_node=anat_mask_sources,
            resampling_applied=(
                nawm_needs_transform
                if anat_spec.in_attr == "nawm_mask_dsc"
                else True
            ),
            native_source=(
                nawm_source_file
                if anat_spec.in_attr == "nawm_mask_dsc"
                else user_lesion_source_file
                if anat_spec.in_attr == "user_lesion_mask_dsc"
                else roi_source_file
                if anat_spec.in_attr == "roi_mask_dsc"
                else None
            ),
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
                input_names=["smooth_hemo", "fwhm"],
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
