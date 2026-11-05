"""HTML report generation utilities for NiPreps-style QC reports.

This module provides utilities for generating a unified HTML QC report
with external figure files, following NiPreps conventions for visual
quality control.
"""


import json
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jinja2 import ChainableUndefined, Environment, PackageLoader

from pmriprep.utils.dce_constants import (
    QC_B1_MEAN_HIGH,
    QC_B1_MEAN_LOW,
    QC_DCE_BASELINE_TSNR_MIN,
    QC_DCE_BASELINE_TSNR_WARN,
    QC_PEAK_ENHANCEMENT_MIN,
    QC_PEAK_ENHANCEMENT_WARN,
)


logger = logging.getLogger(__name__)

# Boilerplate text mappings for DSC deconvolution methods
_DSC_DECONV_TEXT: dict[str, str] = {
    "sSVD": "standard singular value decomposition [@ostergaard_1996]",
    "oSVD": "oscillation-index regularized singular value decomposition [@wu_2003]",
    "tikhonov": (
        "Tikhonov regularization with a modified first-difference penalty "
        "and two leading magnitude-penalized samples [@calamante_2003]. "
        "The L-curve rule selected the strict interior curvature maximum "
        "nearest the upper bound, allowing either curvature sign, from 100 "
        "log-spaced regularization parameters spanning 0.001 to 10 times "
        "the largest singular value of the forward matrix. If no interior "
        "maximum existed, the upper endpoint was selected "
        "[@sourbron_2007_optimization]"
    ),
    "cpi": "control point interpolation [@mehndiratta_2013]",
}

# Boilerplate text mappings for DCE pharmacokinetic models
_DCE_PK_MODEL_TEXT: dict[str, str] = {
    "tofts": "the standard Tofts model [@tofts_1991]",
    "extended_tofts": "the extended Tofts model [@tofts_1999]",
    "patlak": "the Patlak model [@patlak_1983]",
    "2cxm": "the two-compartment exchange model [@sourbron_2013; @brix_2004]",
}

# Boilerplate text for T1 mapping methods
_T1_METHOD_TEXT: dict[str, str] = {
    "wlls": (
        "DCE baseline T1 was estimated from variable flip angle images using "
        "the linearized DESPOT1 signal model [@deoni_2003]. "
        "Fitting used weighted linear least squares"
    ),
    "nlls": (
        "DCE baseline T1 was estimated from variable flip angle images using "
        "the spoiled gradient-echo signal model underlying DESPOT1 "
        "[@deoni_2003]. Fitting used weighted linear least squares "
        "initialization followed by nonlinear least squares in signal space"
    ),
    "external": "a user-supplied T1 map",
    "assumed": "a uniform field-strength-dependent T1",
}

# Boilerplate text for B1+ correction
_B1_CORRECTION_TEXT: dict[str, str] = {
    "afi": (
        " with B1+ transmit field correction using the actual flip angle imaging "
        "method [@yarnykh_2007]"
    ),
    "precomputed": " with B1+ transmit field correction",
    "none": "",
}

# Boilerplate text for parametric-deconvolution inference methods
_INFERENCE_TEXT: dict[str, str] = {
    "vi": (
        "nonlinear variational Bayesian inference "
        "[@chappell_2009]."
    ),
    "svb": (
        "nonlinear variational Bayesian inference "
        "[@chappell_2009] with a CBF-only spatial M-prior "
        "[@penny_2005; @groves_2009]."
    ),
    "map": (
        "maximum a posteriori inference "
        "with Laplace approximation for uncertainty quantification "
        "[@woolrich_2009]."
    ),
    "nlls": "nonlinear least squares.",
}

# Boilerplate text for brain extraction tools
_BRAIN_EXTRACTION_TEXT: dict[str, str] = {
    "hdbet": "HD-BET [@isensee_2019]",
    "ants": "ANTs brain extraction [@tustison_2014; @tustison_2021]",
}


def _inject_tool_versions(
    md: str, versions: Optional[dict[str, str]],
) -> str:
    """Insert third-party software versions before each tool's citation marker.

    Post-assembly substitution keyed on the tool-name-then-citation pattern so
    every tool site (including HD-BET, whose name and cite are fused) is handled
    uniformly. Tokens use
    a bare "(x.y.z)" when the suite name already precedes the tool (ANTs, HD-BET,
    HD-GLIO) and "(FSL x.y.z)" for FSL binaries (FSL is not otherwise named).
    Unknown/missing versions inject nothing, so the wording is byte-identical to
    the version-less boilerplate.
    """
    if not versions:
        return md

    def bare(name: str) -> str:
        ver = versions.get(name)
        return f" (v{ver})" if ver else ""

    def suite(name: str) -> str:
        ver = versions.get(name)
        return f" ({name} v{ver})" if ver else ""

    rules = [
        ("N4BiasFieldCorrection", bare("ANTs")),
        ("DenoiseImage", bare("ANTs")),
        ("ANTs brain extraction", bare("ANTs")),
        ("FAST", suite("FSL")),
        ("ANTs", bare("ANTs")),
        ("FLIRT", suite("FSL")),
        ("HD-BET", bare("HD-BET")),
        ("HD-GLIO", bare("HD-GLIO")),
    ]
    # Insert before the citation's opening bracket (each tool is followed by a
    # Pandoc citation in the rendered text). Anchoring on the bracket rather than
    # the citation marker keeps this source free of stray citation-key patterns.
    for anchor, token in rules:
        if token:
            md = md.replace(f"{anchor} [", f"{anchor}{token} [")

    nipype_ver = versions.get("Nipype")
    if nipype_ver:
        md = md.replace("a Nipype-based", f"a Nipype v{nipype_ver}-based")
    return md


def _load_tool_versions(ses_path: Path, session_id: str) -> dict[str, str]:
    """Build the ``{tool: version}`` map for boilerplate version rendering.

    Prefers versions recorded at run time in the pipeline-root
    ``dataset_description.json`` ``GeneratedBy`` (keeping ``--report-only``
    faithful to the run); falls back to a live query only when the file or its
    entries are missing. Only tools with a resolvable version are included.
    """
    from pmriprep.utils.sidecar_builders import (
        _normalize_tool_version,
        get_imaging_software_versions,
    )

    pipeline_dir = ses_path.parent.parent if session_id else ses_path.parent
    versions: dict[str, str] = {}
    dd_path = pipeline_dir / "dataset_description.json"
    if dd_path.exists():
        try:
            generated_by = json.loads(dd_path.read_text()).get("GeneratedBy", [])
            for entry in generated_by:
                name, ver = entry.get("Name"), entry.get("Version")
                if name and ver and name != "pmriprep":
                    versions[str(name)] = str(ver)
        except (OSError, ValueError):
            versions = {}

    if not versions:
        for name, raw in get_imaging_software_versions().items():
            norm = _normalize_tool_version(raw)
            if norm:
                versions[name] = norm
    return versions


def _generate_boilerplate_text(
    version: str,
    ran_pipelines: dict[str, bool],
    pipeline_data: dict[str, Optional[dict[str, Any]]],
    versions: Optional[dict[str, str]] = None,
    sloppy: bool = False,
) -> str:
    """Generate copy-paste methods text from processing results.

    Creates CC0-licensed methods text suitable for inclusion in manuscripts,
    reflecting the actual processing performed based on QC metrics.

    Parameters
    ----------
    version : str
        PMRIPrep version string.
    ran_pipelines : dict
        Dict indicating which pipelines ran.
    pipeline_data : dict
        QC metrics dicts for each pipeline (None if not run).
    versions : dict of str, optional
        Third-party software versions to insert beside the named tools.
    sloppy : bool
        Whether the run used the reduced-quality testing preset.

    Returns
    -------
    str
        Methods text in Markdown with Pandoc ``@key`` citation markers. The
        reference list is produced from the bundled ``boilerplate.bib`` by
        :func:`_render_citations`; this function emits no reference section.
    """
    sentences: list[str] = []

    # Opening sentence
    sentences.append(
        f"Data were preprocessed using PMRIPrep version {version} "
        "[@rulseh_inprep], a Nipype-based [@gorgolewski_2011] tool."
    )
    if sloppy:
        sentences.append(
            "The run used reduced-quality testing settings (--sloppy); "
            "results and QC are approximate and for testing only."
        )

    # Anatomical preprocessing
    anat_data = pipeline_data.get("anat")
    if ran_pipelines.get("anat") and anat_data is not None:
        anat_text = _build_anat_boilerplate(anat_data.get("qc_metrics", {}))
        if anat_text:
            sentences.append(anat_text)

    # Tumor segmentation
    if ran_pipelines.get("tumor_seg"):
        sentences.append(
            "Automated glioma segmentation was performed using HD-GLIO "
            "[@kickingereder_2019]."
        )

    # DSC perfusion
    dsc_data = pipeline_data.get("dsc")
    if ran_pipelines.get("dsc") and dsc_data is not None:
        dsc_text = _build_dsc_boilerplate(dsc_data.get("qc_metrics", {}))
        if dsc_text:
            sentences.append(dsc_text)

    # DCE perfusion
    dce_data = pipeline_data.get("dce")
    if ran_pipelines.get("dce") and dce_data is not None:
        dce_text = _build_dce_boilerplate(dce_data.get("qc_metrics", {}))
        if dce_text:
            sentences.append(dce_text)

    return _inject_tool_versions(" ".join(sentences), versions)


def _build_anat_boilerplate(anat_qc: dict[str, Any]) -> str:
    """Build anatomical preprocessing boilerplate markdown.

    Parameters
    ----------
    anat_qc : dict
        Anatomical QC metrics dict.

    Returns
    -------
    str
        Methods text with Pandoc ``@key`` citation markers.
    """
    sentences: list[str] = []

    # Bias field correction (always performed)
    bias_text = (
        "Anatomical images were corrected for intensity non-uniformity using "
        "ANTs N4BiasFieldCorrection [@tustison_2010]"
    )
    preprocessing = anat_qc.get("preprocessing", {})
    if preprocessing.get("denoising"):
        bias_text += " with prior denoising using ANTs DenoiseImage [@manjon_2010]"
    bias_text += "."
    sentences.append(bias_text)

    # Brain extraction. Only emit a tool-specific citation when the
    # producer recorded a tool name we have boilerplate for; an unknown
    # or missing tool would otherwise be silently miscited as HD-BET.
    brain_ex = anat_qc.get("brain_extraction", {})
    tool = brain_ex.get("tool")
    if tool in _BRAIN_EXTRACTION_TEXT:
        sentences.append(
            f"Brain extraction was performed using {_BRAIN_EXTRACTION_TEXT[tool]}."
        )
    else:
        sentences.append("Brain extraction was performed.")

    # Tissue segmentation (check for tissue_segmentation key)
    if anat_qc.get("tissue_segmentation"):
        sentences.append(
            "Tissue segmentation of the pre-contrast T1-weighted image into "
            "gray matter, white matter, and cerebrospinal fluid was performed "
            "using FAST [@zhang_2001]."
        )

    # Registration. The secondary-modality coregistration clause is emitted
    # only when secondary structurals were actually coregistered; the T1w->MNI152
    # transform always runs. When both apply they share a single FLIRT citation.
    if anat_qc.get("coregistration", {}).get("secondary_modalities"):
        sentences.append(
            "Secondary structural modalities were coregistered to the "
            "T1-weighted reference, and a rigid-body transform between "
            "T1-weighted and MNI152 space was computed, both using FLIRT "
            "[@jenkinson_2002]."
        )
    else:
        sentences.append(
            "A rigid-body transform between T1-weighted and MNI152 space "
            "was computed using FLIRT [@jenkinson_2002]."
        )

    return " ".join(sentences)


def _motion_was_applied(pipeline_qc: dict[str, Any]) -> bool:
    """Read the ``motion_applied`` provenance flag from a pipeline QC dict.

    Both DSC and DCE surface a flat ``preprocessing.motion_applied``; a nested
    ``motion_summary.motion_applied`` is accepted as a fallback for live (unfiltered)
    QC dicts. Defaults to True (motion correction applied) when absent.
    """
    preproc = pipeline_qc.get("preprocessing", {})
    return bool(
        preproc.get(
            "motion_applied",
            preproc.get("motion_summary", {}).get("motion_applied", True),
        )
    )


def _build_motion_boilerplate(qc: dict[str, Any], modality: str) -> str:
    """Describe recorded motion estimation and whether transforms were applied.

    Parameters
    ----------
    qc : dict
        Pipeline QC containing preprocessing provenance.
    modality : str
        Modality name used in the generated text.

    Returns
    -------
    str
        Motion methods sentence, with tool details only when recorded.
    """
    preprocessing = qc.get("preprocessing", {})
    tool = ""
    if preprocessing.get("motion_tool") == "ANTs":
        from pmriprep.utils.sidecar_builders import _normalize_tool_version

        version = _normalize_tool_version(preprocessing.get("motion_version") or "")
        version_text = f" (v{version})" if version else ""
        tool = (
            f" using ANTs{version_text} [@tustison_2021; @avants_2014] "
            "with rigid registration and mutual information"
        )
    reference = {
        "baseline_mean": "the mean baseline image",
        "mean": "the temporal mean image",
        "first": "the first volume",
    }.get(preprocessing.get("reference_type"))
    reference_text = f" relative to {reference}" if reference else ""
    if _motion_was_applied(qc):
        return f"{modality} time-series data were motion-corrected{tool}{reference_text}."
    return (
        f"{modality} motion correction was omitted; motion was estimated for "
        f"quality control without resampling{tool}{reference_text}."
    )


def _build_dsc_boilerplate(dsc_qc: dict[str, Any]) -> str:
    """Build DSC perfusion boilerplate markdown.

    Parameters
    ----------
    dsc_qc : dict
        DSC QC metrics dict.

    Returns
    -------
    str
        Methods text with Pandoc ``@key`` citation markers.
    """
    sentences: list[str] = []

    deconv = dsc_qc.get("deconvolution", {})
    method = deconv.get("method", "oSVD")
    is_cpi = method.lower() == "cpi"
    fallback_used = (
        is_cpi
        and deconv.get("cpi_fallback") == "osvd"
        and (deconv.get("n_fallback_osvd") or 0) > 0
    )

    # Motion correction (conditional). Kept a separate sentence so concentration
    # conversion can follow the baseline (S0) detection it depends on.
    sentences.append(_build_motion_boilerplate(dsc_qc, "DSC"))

    # Registration
    sentences.append(
        "A rigid-body transform between DSC and T1-weighted space "
        "was computed using FLIRT [@jenkinson_2002]."
    )

    # Baseline detection
    sentences.append(
        "DSC baseline frames were identified by automated bolus-arrival "
        "detection [@rulseh_inprep]."
    )

    # Concentration conversion (uses the S0 baseline)
    sentences.append(
        "DSC signal was converted to concentration-time curves via ΔR2* "
        "estimation [@rosen_1990]."
    )

    # AIF detection
    aif_info = dsc_qc.get("aif_detection") or dsc_qc.get("aif", {})
    config_info = dsc_qc.get("config", {})
    aif_metrics = aif_info.get("aif_metrics", {})
    if not isinstance(aif_metrics, dict):
        aif_metrics = {}
    configured_aif_source = config_info.get("aif_source")
    detected_aif_source = (
        aif_info.get("detection_method")
        or aif_info.get("detection_level")
        or aif_info.get("aif_method")
        or aif_metrics.get("detection_method")
    )
    if configured_aif_source == "external":
        aif_source = "external"
    else:
        aif_source = detected_aif_source or configured_aif_source or "auto"

    if aif_source == "unavailable":
        sentences.append("AIF-dependent perfusion parameters were not estimated.")
    elif aif_source == "external":
        sentences.append("A user-supplied DSC arterial input function was used.")
    elif aif_info.get("search_region") == "user_mask":
        sentences.append(
            "The DSC arterial input function was detected within a user-supplied "
            "search mask using a multi-level criteria cascade [@rulseh_inprep]."
        )
    else:
        sentences.append(
            "The DSC arterial input function was automatically "
            "detected using MRA-atlas-guided voxel selection with a "
            "multi-level criteria cascade [@mouches_2019; @rulseh_inprep]."
        )

    preload = dsc_qc.get("preload", {})
    if preload.get("preload_correction_applied"):
        if is_cpi:
            fallback_clause = (
                " and to the concentration curves used for oSVD fallback voxels"
                if fallback_used else ""
            )
            sentences.append(
                "Per-voxel BSW leakage correction was applied for AUC-CBV "
                f"quantification{fallback_clause} [@boxerman_2006]."
            )
        else:
            sentences.append(
                "Per-voxel BSW leakage correction was applied to account for "
                "contrast extravasation [@boxerman_2006]."
            )
    elif preload.get("leakage_correction_disabled"):
        sentences.append(
            "BSW leakage correction was disabled; tissue concentration curves "
            "were baseline-centered."
        )
    elif preload.get("k2_method") == "bsw" or (
        preload.get("k2_reference_available") is False
    ):
        sentences.append(
            "BSW leakage correction was not applied as an eligible reference "
            "curve could not be constructed; tissue concentration curves remained "
            "baseline-centered."
        )

    auc_cbv_available = deconv.get("cbv_auc_available", aif_source != "unavailable")
    if auc_cbv_available:
        tissue_curve = (
            "BSW-corrected" if preload.get("preload_correction_applied")
            else "baseline-centered"
        )
        if aif_source == "external":
            auc_aif = "user-supplied AIF"
        else:
            auc_aif = "measured AIF"
        sentences.append(
            "DSC AUC-CBV was computed as the "
            f"ratio of the post-baseline integrals of the {tissue_curve} tissue "
            f"concentration curve and the {auc_aif}."
        )

    if deconv.get("deconvolution_performed", aif_source != "unavailable"):
        if is_cpi:
            if deconv.get("cpi_leakage_method") == "bsw":
                sentences.append(
                    "Other model-derived DSC parameter maps were estimated using "
                    "control point interpolation fitted to the original DSC signal "
                    "[@mehndiratta_2013], jointly estimating signed "
                    "leakage with voxel-specific Gaussian priors on K2 "
                    "centred on the BSW estimate "
                    "[@boxerman_2006; @rulseh_inprep]."
                )
            elif deconv.get("cpi_leakage_method") == "none":
                sentences.append(
                    "Other model-derived DSC parameter maps were estimated using "
                    "control point interpolation fitted to the original DSC signal "
                    "with a perfusion-only signal model "
                    "[@mehndiratta_2013]."
                )

            if fallback_used:
                sentences.append(
                    "Voxels with a rejected CPI fit were replaced by oSVD "
                    "estimates [@wu_2003]. The fallback retained signed residues "
                    "without post-fit clipping and used the full 2N-sample "
                    "periodic residue area, including circular closure."
                )
        else:
            method_text = _DSC_DECONV_TEXT.get(
                method, _DSC_DECONV_TEXT.get("oSVD", ""),
            )
            sentences.append(
                f"Other model-derived DSC parameter maps were produced using {method_text}."
            )
            if method in ("sSVD", "tikhonov"):
                sentences.append(
                    "The full N-sample causal system used Volterra product "
                    "integration [@sourbron_2007]."
                )
            sentences.append(
                "Signed unconstrained residues were retained without post-fit clipping. "
                "Reported MTT was computed as AUC-CBV divided by CBF."
            )
            sentences.append(
                "Secondary residue-integral CBV used the full 2N-sample periodic "
                "area, including circular closure."
                if method == "oSVD" else
                "Secondary residue-integral CBV used finite trapezoidal integration "
                "over all N residue samples."
            )

        if is_cpi and (deconv.get("cpi_gamma_fraction") or 0) > 0:
            sentences.append(
                "DSC arterial bolus dispersion was modelled with a gamma vascular "
                "transport function in the retained dispersion-aware CPI fits "
                "[@mehndiratta_2014]."
            )

        # Inference engine (only for cpi parametric methods).
        if is_cpi:
            inference = deconv.get("inference", "vi")
            if inference in _INFERENCE_TEXT:
                sentences.append(f"CPI parameter estimation used {_INFERENCE_TEXT[inference]}")

    # NAWM normalization
    for output_name, source_name, factor_key in (
        ("nCBF", "CBF", "cbf_normalization_factor"),
        ("nCBV", "the post-baseline tissue concentration integral", "normalization_factor"),
    ):
        norm_factor = dsc_qc.get(factor_key)
        if norm_factor is not None and math.isfinite(norm_factor) and norm_factor > 0:
            sentences.append(
                f"DSC {output_name} was obtained by dividing {source_name} by "
                "its median value within the normal-appearing white matter reference."
            )

    return " ".join(sentences)


def _build_dce_boilerplate(dce_qc: dict[str, Any]) -> str:
    """Build DCE perfusion boilerplate markdown.

    Parameters
    ----------
    dce_qc : dict
        DCE QC metrics dict.

    Returns
    -------
    str
        Methods text with Pandoc ``@key`` citation markers.
    """
    sentences: list[str] = []

    # Motion correction (conditional). Kept a separate sentence so concentration
    # conversion can follow the T1 mapping it depends on (strict pipeline order).
    sentences.append(_build_motion_boilerplate(dce_qc, "DCE"))

    # Registration to the anatomical reference is independent of the separate
    # VFA-to-DCE and B1+-map registrations used to prepare quantitative inputs.
    sentences.append(
        "DCE and T1-weighted images were aligned by comparing header alignment "
        "with FLIRT rigid-body registration candidates using foreground overlap "
        "[@jenkinson_2002]."
    )

    # T1 mapping
    t1_info = dce_qc.get("t1_mapping", {})
    signal_info = dce_qc.get("signal_conversion", {})
    mapping_method = t1_info.get("method", t1_info.get("t1_method"))
    t1_method = mapping_method or "nlls"
    # Conversion provenance determines the input actually used, even when a
    # mapping step ran but its result could not be used.
    t1_source = signal_info.get("t1_source")
    if t1_source in ("assumed", "external"):
        t1_method = t1_source
    t1_text = _T1_METHOD_TEXT.get(t1_method, _T1_METHOD_TEXT["nlls"])

    # B1+ correction
    b1_info = dce_qc.get("b1_mapping", {})
    b1_method = t1_info.get("b1_correction", b1_info.get("method", "none"))
    b1_applied = signal_info.get("b1_correction_applied")
    if not isinstance(b1_applied, bool):
        b1_applied = b1_method != "none"
    b1_text = _B1_CORRECTION_TEXT.get(b1_method, "") if b1_applied else ""

    conc = (
        "DCE signal was converted to gadolinium concentration by inversion of "
        "the spoiled gradient-echo signal equation [@schabel_2008]"
    )

    if t1_method in ("external", "assumed"):
        # T1 source stated inline in the concentration sentence (no separate
        # mapping step to order before it).
        if t1_method == "assumed":
            if mapping_method in ("wlls", "nlls") and not t1_info.get("skipped", False):
                sentences.append(
                    "Variable flip angle T1 mapping did not provide a usable map, "
                    "so a uniform T1 was used."
                )
            assumed_t1 = signal_info.get(
                "assumed_t1_ms", t1_info.get("assumed_t1_ms")
            )
            field_strength = signal_info.get(
                "assumed_t1_field_strength_t",
                t1_info.get("assumed_t1_field_strength_t"),
            )
            policy = signal_info.get("assumed_t1_policy", t1_info.get("assumed_t1_policy")) or ""
            if "configured" in policy:
                t1_text = "a user-configured uniform T1"
            else:
                t1_text = "a uniform T1"
            if isinstance(assumed_t1, (int, float)):
                t1_text += f" of {assumed_t1:g} ms"
            if "field_strength" in policy:
                field_text = f" at {field_strength:g} T" if isinstance(field_strength, (int, float)) else ""
                t1_text += f" (field-strength reference{field_text})"
        sentences.append(conc + f", using {t1_text}{b1_text}.")
    else:
        # Full VFA T1 mapping: describe the mapping first, then the
        # concentration conversion that uses it.
        t1_sentence = t1_text
        if b1_text:
            t1_sentence += b1_text
        t1_sentence += "."
        sentences.append(t1_sentence)

        sentences.append(conc + ".")
        if b1_text:
            sentences.append(
                "The same B1+ correction was applied during DCE signal conversion."
            )

    # AIF detection
    aif_info = dce_qc.get("aif_detection", {})
    aif_source = aif_info.get("aif_source") or aif_info.get("detection_method", "auto")
    singleton = aif_info.get("aif_support_voxels") == 1
    pooling_text = (
        "selection of a single voxel, reported with reduced confidence"
        if singleton else "quality-weighted averaging"
    )

    if aif_source == "population":
        sentences.append(
            "The Parker population AIF was used for DCE analysis [@parker_2006]."
        )
    elif aif_source == "external":
        sentences.append("A user-supplied DCE arterial input function was used.")
    elif aif_info.get("search_region") == "user_mask":
        sentences.append(
            "The DCE arterial input function was detected within a user-supplied "
            f"search mask using dose-aware candidate assessment and {pooling_text} "
            "[@rulseh_inprep]."
        )
    elif aif_info.get("detection_method") == "whole_brain":
        sentences.append(
            "The DCE arterial input function was detected by whole-brain search "
            f"with dose-aware candidate assessment and {pooling_text} [@rulseh_inprep]."
        )
    else:
        sentences.append(
            "The DCE arterial input function was automatically "
            "detected using MRA-atlas-guided candidate selection with "
            f"dose-aware candidate assessment and {pooling_text} "
            "[@mouches_2019; @rulseh_inprep]."
        )

    # Detected and population AIFs are blood curves converted to plasma;
    # user-supplied DCE AIFs are required to be plasma concentration already.
    if aif_source != "external":
        hct = aif_info.get("hematocrit", 0.42)
        sentences.append(
            f"The DCE arterial input function was corrected from whole-blood to "
            f"plasma concentration using a hematocrit of {hct} "
            f"[@tofts_1999]."
        )

    # Pharmacokinetic modeling
    pk_info = dce_qc.get("pk_modeling", {})
    model = pk_info.get("model", "auto")

    if model == "auto":
        candidates = pk_info.get("candidate_models", [])
        if candidates:
            # Format candidate list nicely (citations live in the @key markers)
            model_names = [_DCE_PK_MODEL_TEXT.get(m, m) for m in candidates]
            if len(model_names) > 2:
                model_list = ", ".join(model_names[:-1]) + f", and {model_names[-1]}"
            elif len(model_names) == 2:
                model_list = f"{model_names[0]} and {model_names[1]}"
            else:
                model_list = model_names[0] if model_names else "available models"
            sentences.append(
                f"DCE pharmacokinetic analysis used per-voxel "
                f"model selection by the Bayesian information criterion "
                f"[@schwarz_1978] "
                f"among {model_list}."
            )
        else:
            sentences.append(
                "DCE pharmacokinetic analysis used per-voxel "
                "model selection by the Bayesian information criterion "
                "[@schwarz_1978]."
            )
        if "2cxm" in candidates:
            sentences.append(
                "The exchange model was selected only where an extended Tofts fit "
                "at the same delay detected both Ktrans and vp, and the exchange "
                "model had the lower BIC."
            )
        if "patlak" in candidates and len(candidates) > 1:
            sentences.append(
                "When a valid Patlak fit was available, a more complex model "
                "required a BIC at least 6 lower."
            )
    else:
        model_text = _DCE_PK_MODEL_TEXT.get(model, model)
        sentences.append(
            f"DCE pharmacokinetic analysis used {model_text}."
        )

    sentences.append(
        "Model tissue curves were computed by analytic convolution of each model's "
        "impulse response with a shape-preserving piecewise cubic Hermite (PCHIP) "
        "interpolant of the sampled plasma AIF [@fritsch_1984], with the first and "
        "last AIF values held outside the acquired interval."
    )
    timing = pk_info.get("delay_correction", {})
    if timing.get("enabled") is True:
        limit = timing.get("maximum_seconds")
        step = timing.get("grid_step_seconds")
        grid_text = (
            f" over 0–{limit:g} s in {step:g}-s steps"
            if isinstance(limit, (int, float)) and isinstance(step, (int, float)) else ""
        )
        sentences.append(
            f"An effective AIF delay was estimated for each model family{grid_text}. "
            "Each voxel used a reference delay estimated from its in-plane "
            "neighbourhood, with zero used when pooled evidence was insufficient "
            "or flat. A voxel-specific delay was retained if it lowered BIC by "
            "at least 6 after accounting for the additional delay parameter. "
            "The fitted delay is a model offset, not a vascular transit time"
            + (f"; {limit:g} s is the search limit." if isinstance(limit, (int, float)) else ".")
        )
    elif timing.get("enabled") is False:
        sentences.append("The AIF delay was fixed at zero.")

    # Inference engine
    inference = pk_info.get("inference", "nlls")
    inference_text = _INFERENCE_TEXT["vi"] if inference == "vi" else ""
    if inference == "nlls":
        fitted_models = pk_info.get("candidate_models", []) if model == "auto" else [model]
        if not fitted_models:
            inference_text = "bounded least squares."
        elif "patlak" not in fitted_models:
            inference_text = "bounded nonlinear least squares."
        elif len(fitted_models) == 1:
            inference_text = "bounded linear least squares."
        else:
            inference_text = (
                "bounded linear least squares for Patlak and "
                "bounded nonlinear least squares for the other candidate models."
            )
    if inference_text:
        sentences.append(f"DCE parameter estimation used {inference_text}")
    if inference == "vi":
        sentences.append(
            "VI refined least-squares estimates with the model and delay held fixed."
        )
        if pk_info.get("inference_fallback", {}).get("n_nlls_fallback", 0) > 0:
            sentences.append(
                "Unsuccessful VI fits retained the least-squares estimate "
                "for the same model and delay."
            )

    if pk_info.get("qc_acceptance_policy", {}).get("identification", {}).get("basis") == "selected_nlls":
        sentences.append(
            "The binary DCE QC mask combined fit-quality checks on the reported "
            "estimate with primary-parameter identification checks at the corresponding "
            "least-squares fit, including uncertainty across acceptable delays. "
            "Quantitative maps remained unfiltered."
        )
        if inference == "vi":
            sentences.append(
                "VI posterior uncertainty was reported separately and was not used "
                "for identification screening."
            )

    constraint = pk_info.get("dsc_cbv_constraint", {})
    n_constrained = pk_info.get("n_constrained", 0)
    if (
        isinstance(constraint, dict)
        and constraint.get("mode") == "soft"
        and isinstance(n_constrained, (int, float))
        and n_constrained > 0
    ):
        relative_sd = constraint.get("vp_uncertainty")
        width_text = (
            f" with a relative standard deviation of {float(relative_sd):.0%}"
            if isinstance(relative_sd, (int, float))
            else ""
        )
        sentences.append(
            "A spatially varying DSC-informed vp prior"
            f"{width_text} was used by retained Bayesian fits."
        )

    return " ".join(sentences)


# Mapping: internal image key -> (desc_label, default_suffix)
# The suffix is the BIDS modality suffix (dsc, dce, T1w, FLAIR).
# For keys shared between DSC and DCE (motion, aif_curve, aif_location,
# aif_candidates, registration), the pipeline argument overrides the suffix.
FIGURE_NAME_MAP: dict[str, tuple[str, str]] = {
    # DSC
    "motion": ("motion", "dsc"),
    "aif_curve": ("aifcurve", "dsc"),
    "aif_location": ("aiflocation", "dsc"),
    "aif_candidates": ("aifcandidates", "dsc"),
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
    "dsc_dce_region_xval": ("dscxvalregion", "dce"),
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
_SHARED_KEYS = {"motion", "aif_curve", "aif_location", "aif_candidates", "registration"}

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
        "aif_location", "aif_candidates", "dsc_leakage_qc", "dsc_param_maps_qc",
        "dsc_fit_qc", "registration",
    ],
    "dce": [
        "dce_brain_mask", "motion", "b1", "t1", "preprocessing_qc",
        "aif_curve", "aif_location", "aif_candidates", "pk_modeling_qc", "pk_param_maps_qc",
        "model_selection_qc", "dsc_dce_xval", "dsc_dce_region_xval",
        "registration",
    ],
}

# Per-image canonical sidecar paths whose existence proves the pipeline
# ran. These are the report-discovery anchors
# at session root. The aggregator uses them as
# presence probes AND as the source of QC* recommendation/warnings flat
# keys. Anat / tumor_seg patterns use ``{prefix}*`` between the bare
# ``sub-XX_ses-YY`` prefix and the trailing ``_space-...`` so optional
# BIDS run/acq entities propagated from raw inputs are absorbed.
_PIPELINE_PRESENCE_PROBES: dict[str, str] = {
    "anat": "anat/{prefix}*_space-T1w_desc-preproc_T1w.json",
    "tumor_seg": "anat/{prefix}*_space-T1w_desc-tumorseg_dseg.json",
    # DSC/DCE retain recognized source entities (for example ``run`` and
    # ``acq``), so absorb them between the session prefix and descriptor.
    "dsc": "perf/{prefix}*_desc-dscPreproc_dsc.json",
    "dce": "perf/{prefix}*_desc-dcePreproc_dce.json",
}


def _resolve_bids_anchor(
    ses_dir: Path,
    glob_pattern: str,
    *,
    exclude_entities: tuple[str, ...] = ("_ce-",),
) -> Path:
    """Resolve a BIDS file pattern allowing optional run/acq entities.

    Glob patterns use ``{prefix}*`` between the bare ``sub-XX_ses-YY``
    prefix and the trailing entities/suffix to absorb arbitrary BIDS
    run/acq entities written by upstream nodes.

    Filenames containing any token in ``exclude_entities`` are
    de-prioritized: a match without an excluded token is preferred, but
    when every match carries one (e.g. acquisitions where the only
    anatomical is contrast-enhanced) the excluded set is used as a
    fallback rather than returning nothing. Default excludes ``"_ce-"``
    so an anatomy probe like ``{prefix}*_space-T1w_desc-preproc_T1w.json``
    prefers the native T1w when it coexists with a contrast-enhanced
    postcon T1w. Sorted-ascending returns ``_run-1`` before ``_run-2`` etc.

    Returns the first match after filtering, or a non-existent path
    when nothing matches (so callers' ``Path.exists()`` semantics still
    hold for resume / fallback logic).
    """
    all_matches = sorted(ses_dir.glob(glob_pattern))
    preferred = [
        m for m in all_matches
        if not any(token in m.name for token in exclude_entities)
    ]
    # Prefer matches without an excluded entity (e.g. native over postcon
    # T1w when both exist), but fall back to the excluded set when that is
    # all the dataset provides (e.g. acquisitions where every anatomical
    # image is contrast-enhanced).
    chosen = preferred or all_matches
    return chosen[0] if chosen else (ses_dir / glob_pattern)


def figure_filename(
    subject_id: str,
    session_id: str,
    key: str,
    pipeline: str = "",
) -> str:
    """Build a BIDS-compliant figure filename for the given image key.

    Parameters
    ----------
    subject_id : str
        BIDS subject ID (without ``sub-`` prefix).
    session_id : str
        BIDS session ID (without ``ses-`` prefix).
    key : str
        Internal image key (must be in ``FIGURE_NAME_MAP``).
    pipeline : str, optional
        Pipeline name (``'dsc'`` or ``'dce'``).  Used to
        disambiguate shared keys.

    Returns
    -------
    str
        Filename string, e.g.
        ``sub-0011_ses-0001_desc-motion_dsc.png``.

    Raises
    ------
    KeyError
        If *key* is not in ``FIGURE_NAME_MAP``.
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
    """Collect figure relative paths for a single pipeline.

    Scans *figures_dir* for matching files and returns relative paths
    suitable for ``<img src="...">`` in the HTML report (NiPreps pattern).

    Parameters
    ----------
    subject_id : str
        BIDS subject ID.
    session_id : str
        BIDS session ID.
    pipeline : str
        Pipeline name.
    figures_dir : Path
        Absolute path to figures directory.

    Returns
    -------
    dict
        Dict mapping image key to relative path string
        (e.g. ``figures/sub-01_ses-01_desc-brainmask_dsc.png``).
    """
    prefix_parts = [f"sub-{subject_id}"]
    if session_id:
        prefix_parts.append(f"ses-{session_id}")
    prefix = "_".join(prefix_parts)

    images: dict[str, str] = {}
    for key in _PIPELINE_FIGURE_KEYS.get(pipeline, []):
        fname = figure_filename(subject_id, session_id, key, pipeline)
        fpath = figures_dir / fname
        if fpath.exists():
            images[key] = f"figures/{fname}"
            continue

        # Perfusion figures retain source entities (for example acq, ce,
        # rec, and run) between the subject/session prefix and desc. Keep
        # accepting the exact entity-free name above, then discover
        # the entity-rich form by its exact desc/suffix tail.
        tail = fname[len(prefix):]
        matches = sorted(figures_dir.glob(f"{prefix}*{tail}"))
        if matches:
            if len(matches) > 1:
                logger.warning(
                    "Multiple %s QC figures match %s; using %s",
                    pipeline,
                    key,
                    matches[0].name,
                )
            images[key] = f"figures/{matches[0].name}"
    return images


def _classify_metric(
    value: float | None,
    warn_threshold: float,
    min_threshold: float,
) -> str:
    """Classify a QC metric as pass/warn/poor for template display.

    Parameters
    ----------
    value : float or None
        Metric value (or None/NaN if unavailable).
    warn_threshold : float
        Values >= this are "pass".
    min_threshold : float
        Values >= this (but < warn) are "warn".

    Returns
    -------
    str
        Status string: "pass", "warn", "poor", or "na".
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "na"
    if value >= warn_threshold:
        return "pass"
    if value >= min_threshold:
        return "warn"
    return "poor"


def _classify_b1(value: float | None, low: float, high: float) -> str:
    """Classify B1+ bias as pass/warn for template display.

    Parameters
    ----------
    value : float or None
        Mean B1+ value (or None/NaN if unavailable).
    low : float
        Lower acceptable bound (inclusive).
    high : float
        Upper acceptable bound (inclusive).

    Returns
    -------
    str
        Status string: "pass", "warn", or "na".
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "na"
    if low <= value <= high:
        return "pass"
    return "warn"


def _classify_metric_inverse(
    value: float | None,
    warn_threshold: float,
    poor_threshold: float,
) -> str:
    """Classify a QC metric where lower is better for template display.

    Parameters
    ----------
    value : float or None
        Metric value (or None/NaN if unavailable).
    warn_threshold : float
        Values < this are "pass".
    poor_threshold : float
        Values < this (but >= warn) are "warn".

    Returns
    -------
    str
        Status string: "pass", "warn", "poor", or "na".
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "na"
    if value < warn_threshold:
        return "pass"
    if value < poor_threshold:
        return "warn"
    return "poor"


def _read_json_safe(path: Path) -> dict[str, Any]:
    """Load a JSON file; return empty dict on error / missing."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Malformed sidecar JSON, skipping: %s", path)
        return {}


def _drop_report_nulls(value: Any) -> Any:
    """Treat null mapping values as unavailable report fields.

    Report templates use ``is defined`` to decide whether optional metrics
    should be formatted. JSON ``null`` represents the same unavailable state
    but remains defined to Jinja, causing numeric filters and arithmetic to
    fail. Remove null mapping entries recursively while preserving false,
    zero, empty containers, and list positions.
    """
    if isinstance(value, dict):
        return {
            key: _drop_report_nulls(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [
            None if item is None else _drop_report_nulls(item)
            for item in value
        ]
    return value


def _qc_keys_from_anchor(anchor: dict[str, Any]) -> dict[str, Any]:
    """Translate flat ``QC*`` recommendation/warning keys to report structure.

    The canonical anchor sidecars carry ``recommendation``, ``warnings``,
    ``provenance_warnings``, ``acquisition_warnings``, and
    ``processing_warnings`` as flat ``QC*``-prefixed keys. This translates
    them back to the nested shape on ``qc_metrics`` so
    ``_compute_overall_recommendation`` / ``_collect_*_warnings`` can be
    consumed unchanged.
    """
    out: dict[str, Any] = {}
    action = anchor.get("QCRecommendationAction")
    reason = anchor.get("QCRecommendationReason")
    if action or reason:
        out["recommendation"] = {
            "action": str(action or "review"),
            "reason": str(reason or ""),
        }
    warnings = anchor.get("QCWarnings")
    if warnings:
        out["warnings"] = list(warnings)
    prov = anchor.get("QCProvenanceWarnings")
    if prov:
        out["provenance_warnings"] = list(prov)
    acq = anchor.get("QCAcquisitionWarnings")
    if acq:
        out["acquisition_warnings"] = list(acq)
    proc = anchor.get("QCProcessingWarnings")
    if proc:
        out["processing_warnings"] = list(proc)
    return out


def _load_pipeline_sidecars(
    ses_dir: Path, prefix: str, pipeline: str,
) -> dict[str, Any]:
    """Build the report ``qc_metrics`` structure from per-image sidecars.

    Composes the QC summary from the canonical per-image sidecars. The
    nested dict is assembled from:

    - The anchor sidecar (``QC*`` recommendation/warnings flat keys).
    - The anchor's ``PipelineQC`` block (filtered producer-shape QC).

    For pipelines without that block (anat / tumor_seg) the
    anchor sidecar's QC* keys are surfaced; for anat, the loader also
    reads sibling brain-mask and probseg sidecars to recover the
    ``preprocessing`` / ``brain_extraction`` / ``tissue_segmentation``
    nested fields that ``_build_anat_boilerplate`` consumes.
    """
    anchor_pattern = _PIPELINE_PRESENCE_PROBES.get(pipeline)
    if not anchor_pattern:
        return {}
    anchor_path = _resolve_bids_anchor(
        ses_dir, anchor_pattern.format(prefix=prefix),
    )
    anchor = _read_json_safe(anchor_path)

    qc_metrics: dict[str, Any] = {}

    pqc = anchor.get("PipelineQC")
    if isinstance(pqc, dict):
        qc_metrics.update(pqc)
    elif pipeline == "tumor_seg":
        tsqc = anchor.get("TumorSegQC")
        if isinstance(tsqc, dict):
            qc_metrics.update(tsqc)

    if pipeline == "anat":
        qc_metrics.update(_anat_nested_from_sidecars(ses_dir, prefix, anchor))

    # The two updaters write disjoint key sets:
    # _anat_nested_from_sidecars / PipelineQC → preprocessing /
    #     brain_extraction / tissue_segmentation / aif_detection /
    #     preload / pk_modeling / etc.
    # _qc_keys_from_anchor → recommendation / warnings /
    #     provenance_warnings / acquisition_warnings / processing_warnings.
    # Ordering does not matter; if the key sets ever overlap, audit
    # both helpers — the silent overwrite would be a regression.
    qc_metrics.update(_qc_keys_from_anchor(anchor))
    return qc_metrics


def _anat_nested_from_sidecars(
    ses_dir: Path, prefix: str, anchor: dict[str, Any],
) -> dict[str, Any]:
    """Compose anat ``qc_metrics`` nested fields from per-image sidecars.

    The anat boilerplate at :func:`_build_anat_boilerplate` reads three
    nested groups: ``preprocessing.denoising`` (anchor flat key
    ``PreprocessingDenoising``), ``brain_extraction.tool`` (brain-mask
    sidecar flat key ``BrainExtractionTool``), and ``tissue_segmentation``
    (presence of any probseg sidecar). This helper translates the flat
    keys back to the nested shape.
    """
    out: dict[str, Any] = {}

    # preprocessing.denoising — comes from the anchor's flat keys
    preproc: dict[str, Any] = {}
    if "PreprocessingDenoising" in anchor:
        preproc["denoising"] = bool(anchor["PreprocessingDenoising"])
    if preproc:
        out["preprocessing"] = preproc

    # brain_extraction.tool — read from the brain-mask sidecar.
    # Glob accommodates optional ``_run-N`` entity propagated from raw.
    mask_path = _resolve_bids_anchor(
        ses_dir, f"anat/{prefix}*_space-T1w_desc-brain_mask.json",
    )
    mask_sidecar = _read_json_safe(mask_path)
    brain_ex: dict[str, Any] = {}
    if "BrainExtractionTool" in mask_sidecar:
        # Normalize tool name to the lowercase token the boilerplate
        # expects (e.g. "HD-BET" → "hdbet").
        raw_tool = str(mask_sidecar["BrainExtractionTool"])
        brain_ex["tool"] = raw_tool.lower().replace("-", "")
    if "BrainExtractionVolume" in mask_sidecar:
        brain_ex["brain_volume_ml"] = float(mask_sidecar["BrainExtractionVolume"])
    if "BrainExtractionVolumeMm3" in mask_sidecar:
        brain_ex["brain_volume_mm3"] = float(mask_sidecar["BrainExtractionVolumeMm3"])
    elif "BrainExtractionVolume" in mask_sidecar:
        # Fallback when only the mL value was written.
        brain_ex["brain_volume_mm3"] = (
            float(mask_sidecar["BrainExtractionVolume"]) * 1000.0
        )
    if brain_ex:
        out["brain_extraction"] = brain_ex

    # tissue_segmentation — presence of any probseg sidecar is sufficient.
    # Glob accommodates optional ``_run-N`` entity propagated from raw.
    probseg_dir = ses_dir / "anat"
    if probseg_dir.is_dir():
        probsegs: dict[str, Any] = {}
        for tissue in ("WM", "GM", "CSF"):
            ps = _resolve_bids_anchor(
                ses_dir,
                f"anat/{prefix}*_space-T1w_label-{tissue}_probseg.json",
            )
            if ps.exists():
                probsegs[tissue] = {}
        if probsegs:
            out["tissue_segmentation"] = probsegs

    # CoregistrationApplied is derived from the structured processing kind,
    # rather than inferred from human-readable ProcessingSteps.
    coreg_candidates = (
        f"anat/{prefix}*_ce-GAD_desc-preproc_T1w.json",
        f"anat/{prefix}*_desc-preproc_T2w.json",
        f"anat/{prefix}*_desc-preproc_FLAIR.json",
    )
    for pat in coreg_candidates:
        cand = _resolve_bids_anchor(ses_dir, pat)
        if not cand.exists():
            continue
        if _read_json_safe(cand).get("CoregistrationApplied") is True:
            out["coregistration"] = {"secondary_modalities": True}
            break

    return out


def _compute_overall_recommendation(
    pipeline_data: dict[str, Optional[dict[str, Any]]],
) -> Optional[dict[str, str]]:
    """Compute the worst-case recommendation across all pipelines.

    Parameters
    ----------
    pipeline_data : dict
        Dict of pipeline name to data dict (or None).

    Returns
    -------
    dict or None
        Overall recommendation dict, or None if no recommendations exist.
    """
    severity = {"include": 0, "review": 1, "exclude": 2}
    recommendations: list[tuple[str, str, str]] = []

    for name, data in pipeline_data.items():
        if data is None:
            continue
        rec = data.get("qc_metrics", {}).get("recommendation")
        if rec:
            action = rec.get("action", "review")
            recommendations.append((name, action, str(rec.get("reason", ""))))

    if not recommendations:
        return None
    worst_action = max(
        (action for _, action, _ in recommendations),
        key=lambda action: severity.get(action, 1),
    )
    if worst_action == "include":
        reason = "No evaluated pipeline QC criterion requires review"
    else:
        reason = "; ".join(
            f"{name.upper()}: {text}"
            for name, action, text in recommendations
            if severity.get(action, 1) > 0
        )
    return {"action": worst_action, "reason": reason}


def _collect_all_warnings(
    pipeline_data: dict[str, Optional[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Collect warnings from all pipelines, grouped by pipeline name.

    Parameters
    ----------
    pipeline_data : dict
        Dict of pipeline name to data dict (or None).

    Returns
    -------
    dict
        Dict mapping pipeline display name to list of warning strings.
    """
    all_warnings: dict[str, list[str]] = {}
    display_names = {
        "anat": "Anatomical Processing",
        "tumor_seg": "Tumor Segmentation",
        "dsc": "DSC Processing",
        "dce": "DCE Processing",
    }
    for name, data in pipeline_data.items():
        if data is None:
            continue
        warnings = data.get("qc_metrics", {}).get("warnings", [])
        if warnings:
            all_warnings[display_names.get(name, name)] = warnings
    return all_warnings


def _collect_provenance_warnings(
    pipeline_data: dict[str, Optional[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Collect provenance warnings for dedicated HTML banner.

    Reads the ``provenance_warnings`` key from QC metrics (written by
    :func:`~pmriprep.utils.contrast_agent.collect_default_warnings`),
    which is structurally separate from data-quality warnings.

    Parameters
    ----------
    pipeline_data : dict
        Dict of pipeline name to data dict (or None).

    Returns
    -------
    dict
        Dict mapping pipeline display name to list of provenance warning
        strings. Empty dict if no provenance warnings exist.
    """
    result: dict[str, list[str]] = {}
    display_names = {
        "dsc": "DSC Processing",
        "dce": "DCE Processing",
    }
    for name in ("dsc", "dce"):
        data = pipeline_data.get(name)
        if data is None:
            continue
        prov_w = data.get("qc_metrics", {}).get("provenance_warnings", [])
        if prov_w:
            result[display_names[name]] = prov_w
    return result


def _collect_pipeline_warnings(
    pipeline_data: dict[str, Optional[dict[str, Any]]],
    key: str,
) -> dict[str, list[str]]:
    """Collect per-pipeline warnings of a given type for HTML banners.

    Parameters
    ----------
    pipeline_data : dict
        Dict of pipeline name to data dict (or None).
    key : str
        QC metrics key to read (e.g. ``"acquisition_warnings"``).

    Returns
    -------
    dict
        Dict mapping pipeline display name to list of warning strings.
    """
    result: dict[str, list[str]] = {}
    display_names = {
        "dsc": "DSC Processing",
        "dce": "DCE Processing",
    }
    for name in ("dsc", "dce"):
        data = pipeline_data.get(name)
        if data is None:
            continue
        warnings = data.get("qc_metrics", {}).get(key, [])
        if warnings:
            result[display_names[name]] = warnings
    return result


def _render_citations(methods_md: str, ses_path: Path) -> tuple[str, str]:
    """Render the methods markdown into citation artifacts and an HTML fragment.

    Writes ``logs/CITATION.md`` and ``logs/CITATION.bib`` under *ses_path* and,
    when pandoc is available (bundled via ``pypandoc-binary``), renders
    ``logs/CITATION.html`` (an embeddable fragment, not a standalone page) with
    numbered, cross-referenced citations resolved against the boilerplate
    bibliography. The HTML fragment is returned for
    embedding in the report; if pandoc is unavailable or fails, the raw
    ``@key`` markdown is returned instead so the report never loses the Methods.

    Parameters
    ----------
    methods_md : str
        Methods text in Markdown with Pandoc ``@key`` citation markers.
    ses_path : Path
        Session derivatives directory; artifacts are written to ``ses_path/logs``.

    Returns
    -------
    tuple of (str, str)
        ``(embed_html, raw_markdown)`` — the HTML fragment to embed and the
        raw markdown (for the report's copy-for-manuscript button).
    """
    from pmriprep.data import get_boilerplate_bib, get_citation_csl

    logs_dir = ses_path / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        bib = get_boilerplate_bib()
        (logs_dir / "CITATION.md").write_text(methods_md, encoding="utf-8")
        shutil.copyfile(bib, logs_dir / "CITATION.bib")
    except OSError as exc:
        logger.warning("Could not write citation artifacts: %s", exc)

    try:
        import pypandoc

        html = pypandoc.convert_text(
            methods_md,
            to="html",
            format="markdown",
            extra_args=[
                "--citeproc",
                f"--bibliography={get_boilerplate_bib()}",
                f"--csl={get_citation_csl()}",
            ],
        )
        try:
            (logs_dir / "CITATION.html").write_text(html, encoding="utf-8")
        except OSError:
            pass
        return html, methods_md
    except Exception as exc:  # pandoc missing / render failure — degrade gracefully
        logger.warning(
            "pandoc citation rendering unavailable (%s); embedding raw markdown.",
            exc,
        )
        return methods_md, methods_md


def generate_unified_report(
    subject_id: str,
    session_id: str,
    output_path: str,
    ses_dir: str,
    figures_dir: str,
    ran_pipelines: dict[str, bool],
    command_string: str = "",
    sloppy: bool | None = None,
) -> str:
    """Generate a unified HTML QC report for all pipelines in a session.

    Creates a single HTML report with navigation and conditional sections
    for each pipeline that ran. Figures are referenced via relative paths
    (``figures/<filename>``).

    Parameters
    ----------
    subject_id : str
        BIDS subject ID (without sub- prefix).
    session_id : str
        BIDS session ID (without ses- prefix).
    output_path : str
        Output HTML file path.
    ses_dir : str
        Session directory path (contains QC JSONs and figures/).
    figures_dir : str
        Directory containing figure PNG files.
    ran_pipelines : dict
        Dict mapping pipeline name to whether it ran.
    command_string : str, optional
        CLI command used to run PMRIPrep (for reproducibility).
    sloppy : bool, optional
        Run-wide testing preset. Report-only regeneration reads the most recent
        saved session run configuration when this is not supplied.

    Returns
    -------
    str
        Path to generated HTML report.
    """
    import pmriprep

    ses_path = Path(ses_dir)
    fig_path = Path(figures_dir)
    if sloppy is None:
        import toml  # type: ignore[import-untyped]

        subject_path = ses_path.parent if session_id else ses_path
        pattern = f"*/ses-{session_id}/pmriprep.toml" if session_id else "*/pmriprep.toml"
        snapshots = sorted((subject_path / "log").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        sloppy = False
        if snapshots:
            try:
                sloppy = toml.load(snapshots[0]).get("execution", {}).get("sloppy") is True
            except (OSError, ValueError):
                logger.warning("Could not read testing-mode status from %s", snapshots[0])

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

        qc_metrics = _drop_report_nulls(
            _load_pipeline_sidecars(ses_path, prefix, pipeline_name),
        )
        images = _collect_pipeline_images(
            subject_id, session_id, pipeline_name, fig_path,
        )

        # Compute display status for metrics (NiPreps: logic in Python, not templates)
        display_status = None
        if pipeline_name == "dce" and qc_metrics:
            preproc = qc_metrics.get("preprocessing", {})
            b1 = qc_metrics.get("b1_mapping", {})
            pk = qc_metrics.get("pk_modeling", {})
            cbv_c = qc_metrics.get("cbv_constraint", {})
            display_status = {
                # Signal quality
                "peak_enh": _classify_metric(
                    preproc.get("peak_enh_pct"),
                    QC_PEAK_ENHANCEMENT_WARN,
                    QC_PEAK_ENHANCEMENT_MIN,
                ),
                "baseline_tsnr": _classify_metric(
                    preproc.get("baseline_tsnr"),
                    QC_DCE_BASELINE_TSNR_WARN,
                    QC_DCE_BASELINE_TSNR_MIN,
                ),
                "b1_bias": _classify_b1(
                    b1.get("mean_b1"),
                    QC_B1_MEAN_LOW,
                    QC_B1_MEAN_HIGH,
                ),
                # Motion outliers (lower is better)
                "outliers": _classify_metric_inverse(
                    preproc.get("percent_outliers"), 5, 10
                ),
                # B1 correction extent (lower is better)
                "b1_correction": _classify_metric_inverse(
                    b1.get("pct_correction_gt_20"), 10, 30
                ),
                # PK fit quality
                "rsquared": _classify_metric(
                    pk.get("qc_voxel_rsquared"), 0.9, 0.7,
                ),
                # CBV constraint negative fraction (lower is better)
                "cbv_negative": _classify_metric_inverse(
                    cbv_c.get("cbv_negative_fraction"), 0.01, 0.05
                ),
            }
        elif pipeline_name == "dsc" and qc_metrics:
            preproc = qc_metrics.get("preprocessing", {})
            deconv = qc_metrics.get("deconvolution", {})
            display_status = {
                # Motion outliers (lower is better)
                "outliers": _classify_metric_inverse(
                    preproc.get("percent_outliers"), 5, 10
                ),
                # Deconvolution fit quality
                "rsquared": _classify_metric(
                    deconv.get("qc_voxel_rsquared"), 0.9, 0.7,
                ),
            }

        pipeline_data[pipeline_name] = {
            "qc_metrics": qc_metrics,
            "images": images,
            "display_status": display_status,
        }

    # Build template context
    overall_rec = _compute_overall_recommendation(pipeline_data)
    all_warnings = _collect_all_warnings(pipeline_data)
    provenance_warnings = _collect_provenance_warnings(pipeline_data)
    acquisition_warnings = _collect_pipeline_warnings(
        pipeline_data, "acquisition_warnings",
    )
    processing_warnings = _collect_pipeline_warnings(
        pipeline_data, "processing_warnings",
    )
    boilerplate_md = _generate_boilerplate_text(
        pmriprep.__version__, ran_pipelines, pipeline_data,
        versions=_load_tool_versions(ses_path, session_id),
        sloppy=sloppy,
    )
    boilerplate_html, boilerplate_md = _render_citations(boilerplate_md, ses_path)

    context: dict[str, Any] = {
        "subject_id": subject_id,
        "session_id": session_id,
        "version": pmriprep.__version__,
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "command_string": command_string,
        "sloppy": sloppy,
        # Pipeline data (None if not run)
        "anat": pipeline_data.get("anat"),
        "tumor_seg": pipeline_data.get("tumor_seg"),
        "dsc": pipeline_data.get("dsc"),
        "dce": pipeline_data.get("dce"),
        # Summary context
        "ran_pipelines": ran_pipelines,
        "overall_recommendation": overall_rec,
        "all_warnings": all_warnings,
        "provenance_warnings": provenance_warnings,
        "acquisition_warnings": acquisition_warnings,
        "processing_warnings": processing_warnings,
        # Boilerplate methods: pandoc-rendered HTML + raw markdown (copy button)
        "boilerplate_html": boilerplate_html,
        "boilerplate_md": boilerplate_md,
    }

    # Render template
    env = Environment(
        loader=PackageLoader("pmriprep", "templates"),
        undefined=ChainableUndefined,
    )
    template = env.get_template("report_unified.html")
    html = template.render(**context)

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    # Remove superseded per-pipeline HTML reports at the derivatives root.
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
            logger.info("Removed superseded report: %s", legacy_path)

    return output_path
