"""HTML report generation utilities for NiPreps-style QC reports.

This module provides utilities for generating a unified HTML QC report
with external figure files, following NiPreps conventions for visual
quality control.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, PackageLoader


logger = logging.getLogger(__name__)

# Mapping: internal image key -> (desc_label, default_suffix)
# The suffix is the BIDS modality suffix (dsc, dce, T1w, FLAIR).
# For keys shared between DSC and DCE (motion, aif_curve, aif_location,
# registration), the pipeline argument overrides the suffix.
FIGURE_NAME_MAP: dict[str, tuple[str, str]] = {
    # DSC
    "motion": ("motion", "dsc"),
    "aif_curve": ("aifcurve", "dsc"),
    "aif_location": ("aiflocation", "dsc"),
    "registration": ("registration", "dsc"),
    "dsc_preprocessing_qc": ("preprocessing", "dsc"),
    "dsc_param_maps_qc": ("parammaps", "dsc"),
    "dsc_fit_qc": ("deconvfit", "dsc"),
    "dsc_leakage_qc": ("leakage", "dsc"),
    # DSC brain mask
    "dsc_brain_mask": ("brainmask", "dsc"),
    # DCE brain mask
    "dce_brain_mask": ("brainmask", "dce"),
    # DCE
    "b1": ("b1map", "dce"),
    "t1": ("t1map", "dce"),
    "preprocessing_qc": ("preprocessing", "dce"),
    "pk_modeling_qc": ("pkfit", "dce"),
    "pk_param_maps_qc": ("pkparammaps", "dce"),
    # Anatomical
    "brain_mask": ("brainmask", "anat"),
    "tissue_seg": ("tissueseg", "anat"),
    "acpc": ("acpc", "anat"),
    "coreg_postcon": ("coregpostcon", "anat"),
    "coreg_t2w": ("coregt2w", "anat"),
    "coreg_flair": ("coregflair", "anat"),
    "coreg_flair_postcon": ("coregflairpostcon", "anat"),
    # HD-GLIO
    "seg_postcon": ("segpostcon", "anat"),
    "seg_flair": ("segflair", "FLAIR"),
    "nawm_flair": ("nawm", "FLAIR"),
}

# Keys that are shared between DSC and DCE pipelines.
# For these, the suffix is determined by the pipeline argument.
_SHARED_KEYS = {"motion", "aif_curve", "aif_location", "registration"}

# Which figure keys belong to each pipeline
_PIPELINE_FIGURE_KEYS: dict[str, list[str]] = {
    "anat": [
        "brain_mask", "tissue_seg", "acpc",
        "coreg_postcon", "coreg_t2w", "coreg_flair",
        "coreg_flair_postcon",
    ],
    "tumor_seg": ["seg_postcon", "seg_flair", "nawm_flair"],
    "dsc": [
        "dsc_brain_mask", "motion", "dsc_preprocessing_qc", "aif_curve",
        "aif_location", "dsc_leakage_qc", "dsc_param_maps_qc", "dsc_fit_qc",
        "registration",
    ],
    "dce": [
        "dce_brain_mask", "motion", "b1", "t1", "preprocessing_qc",
        "aif_curve", "aif_location", "pk_modeling_qc", "pk_param_maps_qc",
        "registration",
    ],
}

# QC JSON file desc suffixes per pipeline
_PIPELINE_QC_SUFFIXES: dict[str, str] = {
    "anat": "anatQC",
    "tumor_seg": "tumorsegQC",
    "dsc": "dscQC",
    "dce": "dceQC",
}


def figure_filename(
    subject_id: str,
    session_id: str,
    key: str,
    pipeline: str = "",
) -> str:
    """Build a BIDS-compliant figure filename for the given image key.

    Args:
        subject_id: BIDS subject ID (without ``sub-`` prefix).
        session_id: BIDS session ID (without ``ses-`` prefix).
        key: Internal image key (must be in ``FIGURE_NAME_MAP``).
        pipeline: Pipeline name (``'dsc'`` or ``'dce'``).  Used to
            disambiguate shared keys.

    Returns:
        Filename string, e.g.
        ``sub-0011_ses-0001_desc-motion_dsc.png``.

    Raises:
        KeyError: If *key* is not in ``FIGURE_NAME_MAP``.
    """
    desc_label, suffix = FIGURE_NAME_MAP[key]

    # For shared keys, override suffix with pipeline
    if key in _SHARED_KEYS and pipeline:
        suffix = pipeline

    parts = [f"sub-{subject_id}"]
    if session_id:
        parts.append(f"ses-{session_id}")
    parts.append(f"desc-{desc_label}_{suffix}.png")

    return "_".join(parts)


def _collect_pipeline_images(
    subject_id: str,
    session_id: str,
    pipeline: str,
    figures_dir: Path,
) -> dict[str, str]:
    """Collect figure paths for a single pipeline.

    Scans figures_dir for matching files and returns relative paths
    (``figures/<filename>``).

    Args:
        subject_id: BIDS subject ID.
        session_id: BIDS session ID.
        pipeline: Pipeline name.
        figures_dir: Absolute path to figures directory.

    Returns:
        Dict mapping image key to relative path string.
    """
    images: dict[str, str] = {}
    for key in _PIPELINE_FIGURE_KEYS.get(pipeline, []):
        fname = figure_filename(subject_id, session_id, key, pipeline)
        if (figures_dir / fname).exists():
            images[key] = f"figures/{fname}"
    return images


def _load_qc_json(ses_dir: Path, prefix: str, pipeline: str) -> dict[str, Any]:
    """Load QC metrics JSON for a pipeline.

    Args:
        ses_dir: Session directory path.
        prefix: BIDS prefix (e.g. ``sub-0001_ses-0002``).
        pipeline: Pipeline name.

    Returns:
        QC metrics dict, or empty dict if file not found.
    """
    suffix = _PIPELINE_QC_SUFFIXES.get(pipeline)
    if not suffix:
        return {}
    qc_path = ses_dir / f"{prefix}_desc-{suffix}_quality.json"
    if qc_path.exists():
        try:
            with open(qc_path) as f:
                result: dict[str, Any] = json.load(f)
                return result
        except json.JSONDecodeError:
            logger.warning("Malformed QC JSON, skipping: %s", qc_path)
    return {}


def _compute_overall_recommendation(
    pipeline_data: dict[str, Optional[dict[str, Any]]],
) -> Optional[dict[str, str]]:
    """Compute the worst-case recommendation across all pipelines.

    Args:
        pipeline_data: Dict of pipeline name to data dict (or None).

    Returns:
        Overall recommendation dict, or None if no recommendations exist.
    """
    severity = {"include": 0, "review": 1, "exclude": 2}
    worst_action = "include"
    reasons: list[str] = []

    for name, data in pipeline_data.items():
        if data is None:
            continue
        rec = data.get("qc_metrics", {}).get("recommendation")
        if rec:
            action = rec.get("action", "review")
            if severity.get(action, 1) > severity.get(worst_action, 0):
                worst_action = action
            reasons.append(f"{name.upper()}: {rec.get('reason', '')}")

    if not reasons:
        return None
    return {"action": worst_action, "reason": "; ".join(reasons)}


def _collect_all_warnings(
    pipeline_data: dict[str, Optional[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Collect warnings from all pipelines, grouped by pipeline name.

    Args:
        pipeline_data: Dict of pipeline name to data dict (or None).

    Returns:
        Dict mapping pipeline display name to list of warning strings.
    """
    all_warnings: dict[str, list[str]] = {}
    display_names = {
        "anat": "Anatomical",
        "tumor_seg": "Tumor Segmentation",
        "dsc": "DSC Perfusion",
        "dce": "DCE Perfusion",
    }
    for name, data in pipeline_data.items():
        if data is None:
            continue
        warnings = data.get("qc_metrics", {}).get("warnings", [])
        if warnings:
            all_warnings[display_names.get(name, name)] = warnings
    return all_warnings


def generate_unified_report(
    subject_id: str,
    session_id: str,
    output_path: str,
    ses_dir: str,
    figures_dir: str,
    ran_pipelines: dict[str, bool],
) -> str:
    """Generate a unified HTML QC report for all pipelines in a session.

    Creates a single HTML report with navigation and conditional sections
    for each pipeline that ran. Figures are referenced via relative paths
    (``figures/<filename>``).

    Args:
        subject_id: BIDS subject ID (without sub- prefix).
        session_id: BIDS session ID (without ses- prefix).
        output_path: Output HTML file path.
        ses_dir: Session directory path (contains QC JSONs and figures/).
        figures_dir: Directory containing figure PNG files.
        ran_pipelines: Dict mapping pipeline name to whether it ran.

    Returns:
        Path to generated HTML report.
    """
    import pmriprep

    ses_path = Path(ses_dir)
    fig_path = Path(figures_dir)

    # Build BIDS prefix
    prefix = f"sub-{subject_id}"
    if session_id:
        prefix += f"_ses-{session_id}"

    # Collect data for each pipeline that ran
    pipeline_data: dict[str, Optional[dict[str, Any]]] = {}
    for pipeline_name, did_run in ran_pipelines.items():
        if not did_run:
            pipeline_data[pipeline_name] = None
            continue

        qc_metrics = _load_qc_json(ses_path, prefix, pipeline_name)
        images = _collect_pipeline_images(
            subject_id, session_id, pipeline_name, fig_path,
        )
        pipeline_data[pipeline_name] = {
            "qc_metrics": qc_metrics,
            "images": images,
        }

    # Build template context
    overall_rec = _compute_overall_recommendation(pipeline_data)
    all_warnings = _collect_all_warnings(pipeline_data)

    context: dict[str, Any] = {
        "subject_id": subject_id,
        "session_id": session_id,
        "version": pmriprep.__version__,
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # Pipeline data (None if not run)
        "anat": pipeline_data.get("anat"),
        "tumor_seg": pipeline_data.get("tumor_seg"),
        "dsc": pipeline_data.get("dsc"),
        "dce": pipeline_data.get("dce"),
        # Summary context
        "ran_pipelines": ran_pipelines,
        "overall_recommendation": overall_rec,
        "all_warnings": all_warnings,
    }

    # Render template
    env = Environment(loader=PackageLoader("pmriprep", "templates"))
    template = env.get_template("report_unified.html")
    html = template.render(**context)

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    # Clean up legacy per-pipeline HTML reports at the derivatives root
    if session_id:
        legacy_dir = ses_path.parent.parent  # sub-XX -> pmriprep/
    else:
        legacy_dir = ses_path.parent  # pmriprep/
    for pipeline_name in ["anat", "dsc", "dce", "tumor_seg"]:
        if session_id:
            legacy_name = f"sub-{subject_id}_ses-{session_id}_{pipeline_name}.html"
        else:
            legacy_name = f"sub-{subject_id}_{pipeline_name}.html"
        legacy_path = legacy_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
            logger.info("Removed legacy report: %s", legacy_path)

    return output_path
