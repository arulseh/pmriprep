"""HTML report generation utilities for NiPreps-style QC reports.

This module provides utilities for generating HTML QC reports with
external figure files, following NiPreps conventions for visual
quality control.
"""

import json
import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, TemplateNotFound


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
    "coreg_pdw": ("coregpdw", "anat"),
    # HD-GLIO
    "seg_postcon": ("segpostcon", "anat"),
    "seg_flair": ("segflair", "FLAIR"),
    "nawm_flair": ("nawm", "FLAIR"),
}

# Keys that are shared between DSC and DCE pipelines.
# For these, the suffix is determined by the pipeline argument.
_SHARED_KEYS = {"motion", "aif_curve", "aif_location", "registration"}


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


def generate_qc_report(
    subject_id: str,
    session_id: str,
    pipeline: str,
    qc_json_path: str,
    output_path: str,
    figures_dir: str,
) -> str:
    """Generate HTML QC report referencing external figure files.

    Creates a NiPreps-style HTML report that references figures via
    relative paths (``figures/<filename>``).  Figures are expected to
    already exist in *figures_dir* with BIDS-compliant names built by
    :func:`figure_filename`.

    Args:
        subject_id: BIDS subject ID (without sub- prefix).
        session_id: BIDS session ID (without ses- prefix).
        pipeline: Pipeline name ('dsc', 'dce', 'anat', or 'hdglio').
        qc_json_path: Path to QC metrics JSON file.
        output_path: Output HTML file path.
        figures_dir: Directory containing figure PNG files.

    Returns:
        Path to generated HTML report.

    Raises:
        FileNotFoundError: If qc_json_path does not exist.
        TemplateNotFound: If the report template is not found.
    """
    # Load QC metrics
    with open(qc_json_path) as f:
        qc_metrics = json.load(f)

    # Scan figures_dir for matching files and build relative paths
    images: dict[str, str] = {}
    fig_path = Path(figures_dir)
    rel_figures = os.path.relpath(figures_dir, Path(output_path).parent)
    for key in FIGURE_NAME_MAP:
        fname = figure_filename(subject_id, session_id, key, pipeline)
        if (fig_path / fname).exists():
            images[key] = f"{rel_figures}/{fname}"

    # Load and render template
    try:
        env = Environment(loader=PackageLoader("pmriprep", "templates"))
        template = env.get_template(f"report_{pipeline}.html")
    except TemplateNotFound:
        # Fall back to generic template if pipeline-specific one doesn't exist
        try:
            template = env.get_template("report_generic.html")
        except TemplateNotFound:
            # If no templates exist, create a basic report
            html = _generate_fallback_report(
                subject_id=subject_id,
                session_id=session_id,
                pipeline=pipeline,
                qc_metrics=qc_metrics,
                images=images,
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            return output_path

    html = template.render(
        subject_id=subject_id,
        session_id=session_id,
        pipeline=pipeline.upper(),
        qc_metrics=qc_metrics,
        images=images,
    )

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def _generate_fallback_report(
    subject_id: str,
    session_id: str,
    pipeline: str,
    qc_metrics: dict[str, Any],
    images: dict[str, str],
) -> str:
    """Generate a fallback HTML report when templates are not available.

    This function creates a basic but functional HTML report without
    requiring Jinja2 templates.  Image values are relative paths
    (e.g. ``figures/sub-XX_ses-YY_desc-motion_dsc.png``).

    Args:
        subject_id: BIDS subject ID.
        session_id: BIDS session ID.
        pipeline: Pipeline name.
        qc_metrics: QC metrics dictionary.
        images: Dictionary of relative image paths.

    Returns:
        HTML string for the report.
    """
    # Build image sections
    image_sections = []

    if "motion" in images:
        motion_metrics_html = ""
        motion_data = qc_metrics.get("motion", {})
        if isinstance(motion_data, dict):
            motion_summary = motion_data.get("motion_summary", {})
            if isinstance(motion_summary, dict) and motion_summary:
                boxes = []
                for label, key, fmt in [
                    ("Mean FD", "fd_mean", "{:.3f} mm"),
                    ("Max FD", "fd_max", "{:.3f} mm"),
                    ("Outlier %", "pct_outliers", "{:.1f}%"),
                ]:
                    val = motion_summary.get(key)
                    if val is not None:
                        boxes.append(
                            f'<div class="metric"><div class="metric-label">{label}</div>'
                            f'<div class="metric-value">{fmt.format(val)}</div></div>'
                        )
                if boxes:
                    motion_metrics_html = f'<div class="metrics">{"".join(boxes)}</div>'
        image_sections.append(f'''
        <div class="section">
            <h2>Motion Quality Control</h2>
            {motion_metrics_html}
            <img src="{images["motion"]}" alt="Motion QC">
        </div>
        ''')

    # DSC Preprocessing (before AIF for DSC)
    if "dsc_preprocessing_qc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>DSC Preprocessing QC</h2>
            <img src="{images["dsc_preprocessing_qc"]}" alt="DSC Preprocessing QC">
        </div>
        ''')

    if "aif_curve" in images or "aif_location" in images:
        # AIF metrics boxes
        aif_metrics_html = ""
        aif_data = qc_metrics.get("aif", {})
        if isinstance(aif_data, dict):
            aif_metrics = aif_data.get("aif_metrics", {})
            if isinstance(aif_metrics, dict) and aif_metrics:
                boxes = []
                detection = aif_metrics.get("detection_level", "")
                if detection:
                    # Color-coded detection badge
                    det_colors = {"atlas": "#228833", "relaxed": "#CCBB44", "synthetic": "#EE6677"}
                    det_color = det_colors.get(detection, "#999999")
                    boxes.append(
                        f'<div class="metric" style="border-left: 4px solid {det_color};">'
                        f'<div class="metric-label">Detection</div>'
                        f'<div class="metric-value">{detection}</div></div>'
                    )
                for label, key, fmt in [
                    ("Peak Conc", "peak_conc", "{:.2f}"),
                    ("TTP", "ttp", "{:.1f} s"),
                    ("FWHM", "fwhm", "{:.1f} s"),
                    ("BRR", "brr", "{:.1f}%"),
                    ("PSR", "psr", "{:.1f}%"),
                ]:
                    val = aif_metrics.get(key)
                    if val is not None:
                        boxes.append(
                            f'<div class="metric"><div class="metric-label">{label}</div>'
                            f'<div class="metric-value">{fmt.format(val)}</div></div>'
                        )
                if boxes:
                    aif_metrics_html = f'<div class="metrics">{"".join(boxes)}</div>'

        aif_html = f'<div class="section"><h2>Arterial Input Function (AIF) Detection</h2>{aif_metrics_html}<div class="plot-grid">'
        if "aif_curve" in images:
            aif_html += f'<div><img src="{images["aif_curve"]}" alt="AIF Curve"></div>'
        if "aif_location" in images:
            aif_html += f'<div><img src="{images["aif_location"]}" alt="AIF Location"></div>'
        aif_html += '</div></div>'
        image_sections.append(aif_html)

    # DSC Leakage (after AIF)
    if "dsc_leakage_qc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Leakage Correction QC</h2>
            <img src="{images["dsc_leakage_qc"]}" alt="DSC Leakage QC">
        </div>
        ''')

    # DSC Perfusion Maps
    if "dsc_param_maps_qc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>DSC Perfusion Maps</h2>
            <img src="{images["dsc_param_maps_qc"]}" alt="DSC Parameter Maps QC">
        </div>
        ''')

    if "dsc_fit_qc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Deconvolution Fit QC</h2>
            <img src="{images["dsc_fit_qc"]}" alt="DSC Fit QC">
        </div>
        ''')

    # DCE sections: B1 before T1
    if "b1" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>B1+ Mapping QC</h2>
            <img src="{images["b1"]}" alt="B1+ Mapping QC">
        </div>
        ''')

    if "t1" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>T1 Mapping QC</h2>
            <img src="{images["t1"]}" alt="T1 Mapping QC">
        </div>
        ''')

    if "preprocessing_qc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>DCE Preprocessing QC</h2>
            <img src="{images["preprocessing_qc"]}" alt="DCE Preprocessing QC">
        </div>
        ''')

    if "pk_modeling_qc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>PK Modeling QC</h2>
            <img src="{images["pk_modeling_qc"]}" alt="PK Modeling QC">
        </div>
        ''')

    if "pk_param_maps_qc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>PK Parameter Maps</h2>
            <img src="{images["pk_param_maps_qc"]}" alt="PK Parameter Maps QC">
        </div>
        ''')

    # Anatomical sections
    if "brain_mask" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Brain Extraction QC</h2>
            <img src="{images["brain_mask"]}" alt="Brain Mask QC">
        </div>
        ''')

    if "tissue_seg" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Tissue Segmentation</h2>
            <p>WM (blue), GM (red), and CSF (cyan) boundary contours on brain-extracted T1w.</p>
            <img src="{images["tissue_seg"]}" alt="Tissue Segmentation QC">
        </div>
        ''')

    if "acpc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>ACPC-MNI Alignment QC</h2>
            <img src="{images["acpc"]}" alt="ACPC Alignment QC">
        </div>
        ''')

    for coreg_key, coreg_label in [
        ("coreg_postcon", "T1w Post-Contrast to T1w"),
        ("coreg_t2w", "T2w to T1w"),
        ("coreg_flair", "FLAIR to T1w"),
        ("coreg_pdw", "PDw to T1w"),
    ]:
        if coreg_key in images:
            image_sections.append(f'''
        <div class="section">
            <h2>Coregistration: {coreg_label}</h2>
            <img src="{images[coreg_key]}" alt="{coreg_label} QC">
        </div>
            ''')

    if "seg_postcon" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Tumor Segmentation on T1w Post-Contrast</h2>
            <img src="{images["seg_postcon"]}" alt="Tumor on T1w PostCon">
        </div>
        ''')

    if "seg_flair" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Tumor Segmentation on FLAIR</h2>
            <img src="{images["seg_flair"]}" alt="Tumor on FLAIR">
        </div>
        ''')

    if "nawm_flair" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>NAWM ROI on FLAIR</h2>
            <img src="{images["nawm_flair"]}" alt="NAWM on FLAIR">
        </div>
        ''')

    # Registration always last
    if "registration" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Registration Quality</h2>
            <img src="{images["registration"]}" alt="Registration QC">
        </div>
        ''')

    # Build recommendation section
    recommendation_html = ""
    if "recommendation" in qc_metrics:
        rec = qc_metrics["recommendation"]
        action = rec.get("action", "review")
        reason = rec.get("reason", "No reason provided")
        action_class = action.lower() if action.lower() in ["include", "review", "exclude"] else "review"
        recommendation_html = f'''
        <div class="section {action_class}">
            <h2>QC Recommendation: {action.upper()}</h2>
            <p>{reason}</p>
        </div>
        '''

    # Build warnings section
    warnings_html = ""
    warnings = qc_metrics.get("warnings", [])
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warnings_html = f'''
        <div class="section review">
            <h2>Processing Notes</h2>
            <ul>{items}</ul>
        </div>
        '''

    # Build metrics section
    metrics_html = ""
    if qc_metrics:
        metrics_items = []
        for key, value in qc_metrics.items():
            if key != "recommendation" and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str)):
                        if isinstance(sub_value, float):
                            metrics_items.append(f'<div class="metric"><div class="metric-label">{sub_key}</div><div class="metric-value">{sub_value:.3f}</div></div>')
                        else:
                            metrics_items.append(f'<div class="metric"><div class="metric-label">{sub_key}</div><div class="metric-value">{sub_value}</div></div>')
        if metrics_items:
            metrics_html = f'''
            <div class="section">
                <h2>QC Metrics</h2>
                <div class="metrics">
                    {"".join(metrics_items[:12])}
                </div>
            </div>
            '''

    images_html = "\n".join(image_sections)

    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PMRIPrep {pipeline.upper()} Report - sub-{subject_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #4477AA; border-bottom: 2px solid #4477AA; }}
        .section {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .section h2 {{ color: #333; margin-top: 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ background: white; padding: 15px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric-label {{ font-size: 0.85em; color: #666; }}
        .metric-value {{ font-size: 1.4em; font-weight: bold; }}
        .include {{ background: #e8f5e9; border-left: 4px solid #228833; }}
        .review {{ background: #fff8e1; border-left: 4px solid #CCBB44; }}
        .exclude {{ background: #ffebee; border-left: 4px solid #EE6677; }}
        img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .plot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 800px) {{ .plot-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <h1>PMRIPrep {pipeline.upper()} Processing Report</h1>
    <p><strong>Subject:</strong> sub-{subject_id} | <strong>Session:</strong> ses-{session_id}</p>

    {recommendation_html}
    {warnings_html}
    {metrics_html}
    {images_html}

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.85em;">
        Generated by PMRIPrep | <a href="https://github.com/arulseh/pmriprep">GitHub</a>
    </footer>
</body>
</html>
'''
