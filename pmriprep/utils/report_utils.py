"""HTML report generation utilities for NiPreps-style QC reports.

This module provides utilities for generating self-contained HTML QC reports
with embedded images, following NiPreps conventions for visual quality control.
"""

import base64
import json
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, PackageLoader, TemplateNotFound


def image_to_base64(image_path: str) -> str:
    """Convert PNG image to base64 string for HTML embedding.

    Args:
        image_path: Path to the PNG image file.

    Returns:
        Base64-encoded string of the image data.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_qc_report(
    subject_id: str,
    session_id: str,
    pipeline: str,
    qc_json_path: str,
    output_path: str,
    motion_qc_plot: Optional[str] = None,
    aif_curve_plot: Optional[str] = None,
    aif_location_plot: Optional[str] = None,
    registration_filmstrip: Optional[str] = None,
    b1_qc_plot: Optional[str] = None,
    t1_qc_plot: Optional[str] = None,
    brain_mask_filmstrip: Optional[str] = None,
    acpc_filmstrip: Optional[str] = None,
    coreg_filmstrip_postcon: Optional[str] = None,
    coreg_filmstrip_t2w: Optional[str] = None,
    coreg_filmstrip_flair: Optional[str] = None,
    coreg_filmstrip_pdw: Optional[str] = None,
    seg_filmstrip_postcon: Optional[str] = None,
    seg_filmstrip_flair: Optional[str] = None,
    nawm_filmstrip_flair: Optional[str] = None,
) -> str:
    """Generate self-contained HTML QC report with embedded images.

    Creates a NiPreps-style HTML report that includes:
    - QC metrics and recommendations
    - Motion quality control visualization
    - AIF detection quality plots
    - Registration quality filmstrip

    All images are embedded as base64 to create a self-contained HTML file
    that can be viewed without external dependencies.

    Args:
        subject_id: BIDS subject ID (without sub- prefix).
        session_id: BIDS session ID (without ses- prefix).
        pipeline: Pipeline name ('dsc' or 'dce').
        qc_json_path: Path to QC metrics JSON file.
        output_path: Output HTML file path.
        motion_qc_plot: Path to motion QC plot PNG (optional).
        aif_curve_plot: Path to AIF curve plot PNG (optional).
        aif_location_plot: Path to AIF location plot PNG (optional).
        registration_filmstrip: Path to registration filmstrip PNG (optional).

    Returns:
        Path to generated HTML report.

    Raises:
        FileNotFoundError: If qc_json_path does not exist.
        TemplateNotFound: If the report template is not found.
    """
    # Load QC metrics
    with open(qc_json_path) as f:
        qc_metrics = json.load(f)

    # Convert images to base64
    images: dict[str, str] = {}
    for name, path in [
        ("motion", motion_qc_plot),
        ("aif_curve", aif_curve_plot),
        ("aif_location", aif_location_plot),
        ("registration", registration_filmstrip),
        ("b1", b1_qc_plot),
        ("t1", t1_qc_plot),
        ("brain_mask", brain_mask_filmstrip),
        ("acpc", acpc_filmstrip),
        ("coreg_postcon", coreg_filmstrip_postcon),
        ("coreg_t2w", coreg_filmstrip_t2w),
        ("coreg_flair", coreg_filmstrip_flair),
        ("coreg_pdw", coreg_filmstrip_pdw),
        ("seg_postcon", seg_filmstrip_postcon),
        ("seg_flair", seg_filmstrip_flair),
        ("nawm_flair", nawm_filmstrip_flair),
    ]:
        if path and Path(path).exists():
            images[name] = image_to_base64(path)

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
    requiring Jinja2 templates.

    Args:
        subject_id: BIDS subject ID.
        session_id: BIDS session ID.
        pipeline: Pipeline name.
        qc_metrics: QC metrics dictionary.
        images: Dictionary of base64-encoded images.

    Returns:
        HTML string for the report.
    """
    # Build image sections
    image_sections = []

    if "motion" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Motion Quality Control</h2>
            <img src="data:image/png;base64,{images["motion"]}" alt="Motion QC">
        </div>
        ''')

    if "aif_curve" in images or "aif_location" in images:
        aif_html = '<div class="section"><h2>Arterial Input Function (AIF) Detection</h2><div class="plot-grid">'
        if "aif_curve" in images:
            aif_html += f'<div><img src="data:image/png;base64,{images["aif_curve"]}" alt="AIF Curve"></div>'
        if "aif_location" in images:
            aif_html += f'<div><img src="data:image/png;base64,{images["aif_location"]}" alt="AIF Location"></div>'
        aif_html += '</div></div>'
        image_sections.append(aif_html)

    if "b1" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>B1+ Mapping QC</h2>
            <img src="data:image/png;base64,{images["b1"]}" alt="B1+ Mapping QC">
        </div>
        ''')

    if "t1" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>T1 Mapping QC</h2>
            <img src="data:image/png;base64,{images["t1"]}" alt="T1 Mapping QC">
        </div>
        ''')

    if "registration" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Registration Quality</h2>
            <img src="data:image/png;base64,{images["registration"]}" alt="Registration QC">
        </div>
        ''')

    if "brain_mask" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Brain Extraction QC</h2>
            <img src="data:image/png;base64,{images["brain_mask"]}" alt="Brain Mask QC">
        </div>
        ''')

    if "acpc" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>ACPC-MNI Alignment QC</h2>
            <img src="data:image/png;base64,{images["acpc"]}" alt="ACPC Alignment QC">
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
            <img src="data:image/png;base64,{images[coreg_key]}" alt="{coreg_label} QC">
        </div>
            ''')

    if "seg_postcon" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Tumor Segmentation on T1w Post-Contrast</h2>
            <img src="data:image/png;base64,{images["seg_postcon"]}" alt="Tumor on T1w PostCon">
        </div>
        ''')

    if "seg_flair" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>Tumor Segmentation on FLAIR</h2>
            <img src="data:image/png;base64,{images["seg_flair"]}" alt="Tumor on FLAIR">
        </div>
        ''')

    if "nawm_flair" in images:
        image_sections.append(f'''
        <div class="section">
            <h2>NAWM ROI on FLAIR</h2>
            <img src="data:image/png;base64,{images["nawm_flair"]}" alt="NAWM on FLAIR">
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
    {metrics_html}
    {images_html}

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.85em;">
        Generated by PMRIPrep | <a href="https://github.com/arulseh/pmriprep">GitHub</a>
    </footer>
</body>
</html>
'''
