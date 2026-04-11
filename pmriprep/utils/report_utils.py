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

# Boilerplate text mappings for DSC deconvolution methods
_DSC_DECONV_TEXT: dict[str, str] = {
    "sSVD": "standard singular value decomposition (sSVD; Ostergaard et al., 1996)",
    "cSVD": "block-circulant singular value decomposition (cSVD; Wu et al., 2003)",
    "oSVD": "oscillation-index regularized SVD (oSVD; Wu et al., 2003)",
    "tikhonov": (
        "Tikhonov-regularized deconvolution with generalized cross-validation "
        "(Hansen, 1992; Calamante et al., 1999)"
    ),
    "vascular": (
        "the vascular model with gamma-variate transit time distribution "
        "(Mouridsen et al., 2006)"
    ),
    "cpi": "control point interpolation (CPI; Mehndiratta et al., 2013)",
}

# Boilerplate text mappings for DCE pharmacokinetic models
_DCE_PK_MODEL_TEXT: dict[str, str] = {
    "tofts": "the standard Tofts model (Tofts & Kermode, 1991)",
    "extended_tofts": "the extended Tofts model (Tofts et al., 1999)",
    "patlak": "the Patlak model (Patlak et al., 1983)",
    "2cxm": "the two-compartment exchange model (Brix et al., 2004)",
}

# Boilerplate text for T1 mapping methods
_T1_METHOD_TEXT: dict[str, str] = {
    "hybrid": (
        "Baseline T1 was estimated from variable flip angle (VFA) images using "
        "hybrid fitting (linearized DESPOT1 with nonlinear refinement for "
        "ill-conditioned voxels; Deoni et al., 2003)"
    ),
    "wlls": (
        "Baseline T1 was estimated from variable flip angle (VFA) images using "
        "linearized DESPOT1 fitting (Deoni et al., 2003)"
    ),
    "nlls": (
        "Baseline T1 was estimated from variable flip angle (VFA) images using "
        "nonlinear least squares fitting"
    ),
    "external": "using a user-supplied T1 map",
    "assumed": "using literature-based T1 (no T1 mapping performed)",
}

# Boilerplate text for B1+ correction
_B1_CORRECTION_TEXT: dict[str, str] = {
    "afi": (
        " with B1+ transmit field correction using the actual flip angle imaging "
        "(AFI) method (Yarnykh, 2007)"
    ),
    "tfl_b1map": " with B1+ transmit field correction",
    "none": "",
}

# Boilerplate text for Bayesian inference methods
_BAYESIAN_TEXT: dict[str, str] = {
    "vi": (
        "Parameter estimation used variational Bayesian inference "
        "(Chappell et al., 2009)."
    ),
    "map": (
        "Parameter estimation used maximum a posteriori (MAP) inference "
        "with Laplace approximation for uncertainty quantification "
        "(Woolrich et al., 2009)."
    ),
    "none": "Parameters were estimated using nonlinear least squares.",
}

# Boilerplate text for brain extraction tools
_BRAIN_EXTRACTION_TEXT: dict[str, str] = {
    "hdbet": "HD-BET (Isensee et al., 2019)",
    "ants": "ANTs brain extraction (Avants et al., 2011)",
}

def _generate_boilerplate_text(
    version: str,
    ran_pipelines: dict[str, bool],
    pipeline_data: dict[str, Optional[dict[str, Any]]],
) -> str:
    """Generate copy-paste methods text from processing results.

    Creates CC0-licensed methods text suitable for inclusion in manuscripts,
    reflecting the actual processing performed based on QC metrics.

    Args:
        version: PMRIPrep version string.
        ran_pipelines: Dict indicating which pipelines ran.
        pipeline_data: QC metrics dicts for each pipeline (None if not run).

    Returns:
        Formatted methods paragraph suitable for manuscripts.
    """
    paragraphs: list[str] = []

    # Opening sentence
    paragraphs.append(
        f"Data were preprocessed using PMRIPrep version {version} "
        "(CITATION PLACEHOLDER), a Nipype-based (Gorgolewski et al., 2011) "
        "tool for perfusion MRI analysis."
    )

    # Anatomical preprocessing
    anat_data = pipeline_data.get("anat")
    if ran_pipelines.get("anat") and anat_data is not None:
        anat_qc = anat_data.get("qc_metrics", {})
        anat_text = _build_anat_boilerplate(anat_qc)
        if anat_text:
            paragraphs.append(anat_text)

    # Tumor segmentation
    if ran_pipelines.get("tumor_seg"):
        paragraphs.append(
            "Automated glioma segmentation was performed using HD-GLIO "
            "(Kickingereder et al., 2019), producing enhancing tumor, "
            "non-enhancing tumor, and whole tumor probability maps."
        )

    # DSC perfusion
    dsc_data = pipeline_data.get("dsc")
    if ran_pipelines.get("dsc") and dsc_data is not None:
        dsc_qc = dsc_data.get("qc_metrics", {})
        dsc_text = _build_dsc_boilerplate(dsc_qc)
        if dsc_text:
            paragraphs.append(dsc_text)

    # DCE perfusion
    dce_data = pipeline_data.get("dce")
    if ran_pipelines.get("dce") and dce_data is not None:
        dce_qc = dce_data.get("qc_metrics", {})
        dce_text = _build_dce_boilerplate(dce_qc)
        if dce_text:
            paragraphs.append(dce_text)

    return "\n\n".join(paragraphs)


def _build_anat_boilerplate(anat_qc: dict[str, Any]) -> str:
    """Build anatomical preprocessing boilerplate text.

    Args:
        anat_qc: Anatomical QC metrics dict.

    Returns:
        Anatomical methods paragraph.
    """
    sentences: list[str] = []

    # Bias field correction (always performed)
    bias_text = (
        "Anatomical images were corrected for intensity non-uniformity using "
        "ANTs N4BiasFieldCorrection (Tustison et al., 2010)"
    )
    # Check for denoising
    preprocessing = anat_qc.get("preprocessing", {})
    if preprocessing.get("denoising"):
        bias_text += " with prior denoising using ANTs DenoiseImage"
    bias_text += "."
    sentences.append(bias_text)

    # Brain extraction
    brain_ex = anat_qc.get("brain_extraction", {})
    tool = brain_ex.get("tool", "hdbet")
    tool_text = _BRAIN_EXTRACTION_TEXT.get(tool, _BRAIN_EXTRACTION_TEXT["hdbet"])
    sentences.append(f"Brain extraction was performed using {tool_text}.")

    # Tissue segmentation (check for tissue_segmentation key)
    if anat_qc.get("tissue_segmentation"):
        sentences.append(
            "Tissue segmentation (GM, WM, CSF) was performed using "
            "FSL FAST (Zhang et al., 2001)."
        )

    return " ".join(sentences)


def _build_dsc_boilerplate(dsc_qc: dict[str, Any]) -> str:
    """Build DSC perfusion boilerplate text.

    Args:
        dsc_qc: DSC QC metrics dict.

    Returns:
        DSC methods paragraph.
    """
    sentences: list[str] = []

    # Opening: motion correction and concentration conversion
    sentences.append(
        "DSC perfusion images were motion-corrected using FSL MCFLIRT "
        "(Jenkinson et al., 2002) and converted to concentration-time "
        "curves via ΔR2* estimation."
    )

    # AIF detection
    aif_info = dsc_qc.get("aif", {})
    config_info = dsc_qc.get("config", {})
    aif_source = config_info.get("aif_source") or aif_info.get("aif_method", "auto")

    if aif_source == "synthetic":
        sentences.append(
            "A synthetic population-derived AIF was used, based on "
            "gamma-variate parameters from the literature "
            "(Parker et al., 2006; Mouridsen et al., 2006)."
        )
    elif aif_source == "external":
        sentences.append("A user-supplied arterial input function was used.")
    else:
        sentences.append(
            "The arterial input function (AIF) was automatically detected "
            "using atlas-based voxel selection (PMRIPrep; CITATION PLACEHOLDER)."
        )

    # Deconvolution method
    deconv = dsc_qc.get("deconvolution", {})
    method = deconv.get("method", "oSVD")
    method_text = _DSC_DECONV_TEXT.get(method, _DSC_DECONV_TEXT.get("oSVD", ""))
    sentences.append(f"Perfusion parameter maps were quantified using {method_text}.")

    # Bayesian inference (only for vascular/cpi methods)
    if method.lower() in ("vascular", "cpi"):
        bayesian = deconv.get("bayesian", "vi")
        if bayesian in ("vi", "map"):
            sentences.append(_BAYESIAN_TEXT.get(bayesian, ""))

    # Leakage correction
    preload = dsc_qc.get("preload", {})
    if preload.get("preload_correction_applied"):
        sentences.append(
            "K2 leakage correction was applied to account for contrast "
            "extravasation (Boxerman et al., 2006)."
        )

    # Registration
    reg_bbr = config_info.get("registration_bbr", False)
    if reg_bbr:
        sentences.append(
            "DSC images were registered to anatomical space using "
            "FSL FLIRT (Jenkinson et al., 2002) with boundary-based "
            "registration (BBR) refinement (Greve & Fischl, 2009)."
        )
    else:
        sentences.append(
            "DSC images were registered to anatomical space using "
            "FSL FLIRT (Jenkinson et al., 2002)."
        )

    return " ".join(sentences)


def _build_dce_boilerplate(dce_qc: dict[str, Any]) -> str:
    """Build DCE perfusion boilerplate text.

    Args:
        dce_qc: DCE QC metrics dict.

    Returns:
        DCE methods paragraph.
    """
    sentences: list[str] = []

    # Opening: motion correction and signal conversion
    opening = (
        "DCE permeability images were motion-corrected and converted to "
        "gadolinium concentration using the spoiled gradient echo signal model"
    )

    # T1 mapping
    t1_info = dce_qc.get("t1_mapping", {})
    t1_method = t1_info.get("method", "hybrid")
    t1_text = _T1_METHOD_TEXT.get(t1_method, _T1_METHOD_TEXT["hybrid"])

    # B1+ correction
    b1_method = t1_info.get("b1_correction", "none")
    b1_text = _B1_CORRECTION_TEXT.get(b1_method, "")

    # Combine T1 and B1+ into opening
    if t1_method in ("external", "assumed"):
        # Short form for external/assumed
        sentences.append(opening + f" {t1_text}.")
    else:
        # Full T1 mapping text
        sentences.append(opening + ".")
        t1_sentence = t1_text
        if b1_text:
            t1_sentence += b1_text
        t1_sentence += "."
        sentences.append(t1_sentence)

    # AIF detection
    aif_info = dce_qc.get("aif_detection", {})
    aif_source = aif_info.get("aif_source") or aif_info.get("detection_method", "auto")

    if aif_source == "population":
        sentences.append(
            "The Parker population AIF was used (Parker et al., 2006)."
        )
    elif aif_source == "external":
        sentences.append("A user-supplied arterial input function was used.")
    else:
        sentences.append(
            "The arterial input function was automatically detected using "
            "atlas-based candidate selection (PMRIPrep; CITATION PLACEHOLDER)."
        )

    # Pharmacokinetic modeling
    pk_info = dce_qc.get("pk_modeling", {})
    model = pk_info.get("model", "auto")

    if model == "auto":
        candidates = pk_info.get("candidate_models", [])
        if candidates:
            # Format candidate list nicely
            model_names = [_DCE_PK_MODEL_TEXT.get(m, m) for m in candidates]
            if len(model_names) > 1:
                model_list = ", ".join(model_names[:-1]) + f", and {model_names[-1]}"
            else:
                model_list = model_names[0] if model_names else "available models"
            sentences.append(
                f"Pharmacokinetic analysis was performed using per-voxel "
                f"model selection (BIC) among {model_list}."
            )
        else:
            sentences.append(
                "Pharmacokinetic analysis was performed using per-voxel "
                "model selection (BIC)."
            )
    else:
        model_text = _DCE_PK_MODEL_TEXT.get(model, model)
        sentences.append(
            f"Pharmacokinetic analysis was performed using {model_text}."
        )

    # Bayesian inference
    bayesian = pk_info.get("bayesian", "none")
    bayesian_text = _BAYESIAN_TEXT.get(bayesian, "")
    if bayesian_text:
        sentences.append(bayesian_text)

    # Registration (DCE uses FLIRT only, no BBR)
    sentences.append(
        "DCE images were registered to anatomical space using "
        "FSL FLIRT (Jenkinson et al., 2002)."
    )

    return " ".join(sentences)


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
    "model_selection_qc": ("modelselection", "dce"),
    "dsc_dce_xval": ("dscxval", "dce"),
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
        "model_selection_qc", "dsc_dce_xval", "registration",
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
    command_string: str = "",
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
        command_string: CLI command used to run PMRIPrep (for reproducibility).

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
    boilerplate_text = _generate_boilerplate_text(
        pmriprep.__version__, ran_pipelines, pipeline_data
    )

    context: dict[str, Any] = {
        "subject_id": subject_id,
        "session_id": session_id,
        "version": pmriprep.__version__,
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "command_string": command_string,
        # Pipeline data (None if not run)
        "anat": pipeline_data.get("anat"),
        "tumor_seg": pipeline_data.get("tumor_seg"),
        "dsc": pipeline_data.get("dsc"),
        "dce": pipeline_data.get("dce"),
        # Summary context
        "ran_pipelines": ran_pipelines,
        "overall_recommendation": overall_rec,
        "all_warnings": all_warnings,
        # Boilerplate methods text
        "boilerplate_text": boilerplate_text,
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
