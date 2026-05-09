"""HTML report generation utilities for NiPreps-style QC reports.

This module provides utilities for generating a unified HTML QC report
with external figure files, following NiPreps conventions for visual
quality control.
"""


import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, PackageLoader

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
    "sSVD": "standard singular value decomposition (sSVD; Ostergaard et al., 1996)",
    "cSVD": "block-circulant singular value decomposition (cSVD; Wu et al., 2003)",
    "oSVD": "oscillation-index regularized SVD (oSVD; Wu et al., 2003)",
    "tikhonov": (
        "Tikhonov-regularized deconvolution with generalized cross-validation "
        "(Hansen, 1992; Calamante et al., 2000)"
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
    "2cxm": "the two-compartment exchange model (Sourbron & Buckley, 2011)",
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

# Boilerplate text for parametric-deconvolution inference methods
_INFERENCE_TEXT: dict[str, str] = {
    "vi": (
        "Parameter estimation used variational Bayesian inference "
        "(Chappell et al., 2009)."
    ),
    "svb": (
        "Parameter estimation used spatial variational Bayes with the "
        "M-prior (Penny et al., 2005; Groves et al., 2009)."
    ),
    "map": (
        "Parameter estimation used maximum a posteriori (MAP) inference "
        "with Laplace approximation for uncertainty quantification "
        "(Woolrich et al., 2009)."
    ),
    "nlls": "Parameter estimation used nonlinear least squares.",
}

# Boilerplate text for brain extraction tools
_BRAIN_EXTRACTION_TEXT: dict[str, str] = {
    "hdbet": "HD-BET (Isensee et al., 2019)",
    "ants": "ANTs brain extraction (Avants et al., 2011)",
}

# Full bibliographic references for all cited works
_REFERENCES: dict[str, str] = {
    "avants_2011": (
        "Avants BB, Tustison NJ, Song G, Cook PA, Klein A, Gee JC. "
        "A reproducible evaluation of ANTs similarity metric performance in brain image registration. "
        "NeuroImage. 2011;54(3):2033-2044. doi:10.1016/j.neuroimage.2010.09.025"
    ),
    "boxerman_2006": (
        "Boxerman JL, Schmainda KM, Weisskoff RM. "
        "Relative cerebral blood volume maps corrected for contrast agent extravasation "
        "significantly correlate with glioma tumor grade, whereas uncorrected maps do not. "
        "AJNR Am J Neuroradiol. 2006;27(4):859-867."
    ),
    "calamante_2000": (
        "Calamante F, Gadian DG, Connelly A. "
        "Delay and dispersion effects in dynamic susceptibility contrast MRI: "
        "simulations using singular value decomposition. "
        "Magn Reson Med. 2000;44(3):466-473. doi:10.1002/1522-2594(200009)44:3<466::AID-MRM18>3.0.CO;2-M"
    ),
    "chappell_2009": (
        "Chappell MA, Groves AR, Whitcher B, Woolrich MW. "
        "Variational Bayesian inference for a nonlinear forward model. "
        "IEEE Trans Signal Process. 2009;57(1):223-236. doi:10.1109/TSP.2008.2005752"
    ),
    "deoni_2003": (
        "Deoni SCL, Rutt BK, Peters TM. "
        "Rapid combined T1 and T2 mapping using gradient recalled acquisition in the steady state. "
        "Magn Reson Med. 2003;49(3):515-526. doi:10.1002/mrm.10407"
    ),
    "gorgolewski_2011": (
        "Gorgolewski K, Burns CD, Madison C, et al. "
        "Nipype: a flexible, lightweight and extensible neuroimaging data processing framework in Python. "
        "Front Neuroinform. 2011;5:13. doi:10.3389/fninf.2011.00013"
    ),
    "hansen_1992": (
        "Hansen PC. "
        "Analysis of discrete ill-posed problems by means of the L-curve. "
        "SIAM Rev. 1992;34(4):561-580. doi:10.1137/1034115"
    ),
    "isensee_2019": (
        "Isensee F, Schell M, Pflueger I, et al. "
        "Automated brain extraction of multisequence MRI using artificial neural networks. "
        "Hum Brain Mapp. 2019;40(17):4952-4964. doi:10.1002/hbm.24750"
    ),
    "jenkinson_2002": (
        "Jenkinson M, Bannister P, Brady M, Smith S. "
        "Improved optimization for the robust and accurate linear registration and motion correction of brain images. "
        "NeuroImage. 2002;17(2):825-841. doi:10.1006/nimg.2002.1132"
    ),
    "kickingereder_2019": (
        "Kickingereder P, Isensee F, Tursunova I, et al. "
        "Automated quantitative tumour response assessment of MRI in neuro-oncology with artificial neural networks: "
        "a multicentre, retrospective study. "
        "Lancet Oncol. 2019;20(5):728-740. doi:10.1016/S1470-2045(19)30098-1"
    ),
    "mehndiratta_2013": (
        "Mehndiratta A, MacIntosh BJ, Crane DE, Payne SJ, Chappell MA. "
        "A control point interpolation method for the non-parametric quantification of cerebral haemodynamics "
        "from dynamic susceptibility contrast MRI. "
        "NeuroImage. 2013;64:560-570. doi:10.1016/j.neuroimage.2012.08.083"
    ),
    "nicholson_1998": (
        "Nicholson C, Syková E. "
        "Extracellular space structure revealed by diffusion analysis. "
        "Trends Neurosci. 1998;21(5):207-215. doi:10.1016/S0166-2236(98)01261-2"
    ),
    "mouridsen_2006": (
        "Mouridsen K, Friston K, Hjort N, Gyldensted L, Ostergaard L, Kiebel S. "
        "Bayesian estimation of cerebral perfusion using a physiological model of microvasculature. "
        "NeuroImage. 2006;33(2):570-579. doi:10.1016/j.neuroimage.2006.06.015"
    ),
    "ostergaard_1996": (
        "Ostergaard L, Weisskoff RM, Chesler DA, Gyldensted C, Rosen BR. "
        "High resolution measurement of cerebral blood flow using intravascular tracer bolus passages. "
        "Part I: Mathematical approach and statistical analysis. "
        "Magn Reson Med. 1996;36(5):715-725. doi:10.1002/mrm.1910360510"
    ),
    "parker_2006": (
        "Parker GJM, Roberts C, Macdonald A, et al. "
        "Experimentally-derived functional form for a population-averaged high-temporal-resolution "
        "arterial input function for dynamic contrast-enhanced MRI. "
        "Magn Reson Med. 2006;56(5):993-1000. doi:10.1002/mrm.21066"
    ),
    "patlak_1983": (
        "Patlak CS, Blasberg RG, Fenstermacher JD. "
        "Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data. "
        "J Cereb Blood Flow Metab. 1983;3(1):1-7. doi:10.1038/jcbfm.1983.1"
    ),
    "rulseh_inprep": (
        "Rulseh AM, et al. "
        "PMRIPrep: A standardized preprocessing pipeline for perfusion and permeability MRI. "
        "In preparation."
    ),
    "sourbron_2011": (
        "Sourbron SP, Buckley DL. "
        "On the scope and interpretation of the Tofts models for DCE-MRI. "
        "Magn Reson Med. 2011;66(3):735-745. doi:10.1002/mrm.22861"
    ),
    "tofts_1997": (
        "Tofts PS. "
        "Modeling tracer kinetics in dynamic Gd-DTPA MR imaging. "
        "J Magn Reson Imaging. 1997;7(1):91-101. doi:10.1002/jmri.1880070113"
    ),
    "tofts_1991": (
        "Tofts PS, Kermode AG. "
        "Measurement of the blood-brain barrier permeability and leakage space using dynamic MR imaging. "
        "1. Fundamental concepts. "
        "Magn Reson Med. 1991;17(2):357-367. doi:10.1002/mrm.1910170208"
    ),
    "tofts_1999": (
        "Tofts PS, Brix G, Buckley DL, et al. "
        "Estimating kinetic parameters from dynamic contrast-enhanced T1-weighted MRI of a diffusable tracer: "
        "standardized quantities and symbols. "
        "J Magn Reson Imaging. 1999;10(3):223-232. doi:10.1002/(SICI)1522-2586(199909)10:3<223::AID-JMRI2>3.0.CO;2-S"
    ),
    "tustison_2010": (
        "Tustison NJ, Avants BB, Cook PA, et al. "
        "N4ITK: improved N3 bias correction. "
        "IEEE Trans Med Imaging. 2010;29(6):1310-1320. doi:10.1109/TMI.2010.2046908"
    ),
    "leu_2016": (
        "Leu K, Boxerman JL, Cloughesy TF, et al. "
        "Improved leakage correction for single-echo dynamic susceptibility contrast "
        "perfusion MRI estimates of relative cerebral blood volume in high-grade gliomas "
        "by accounting for bidirectional contrast agent exchange. "
        "AJNR Am J Neuroradiol. 2016;37(8):1440-1446. doi:10.3174/ajnr.A4759"
    ),
    "weisskoff_1994": (
        "Weisskoff RM, Boxerman JL, Sorensen AG, Kulke SM, Campbell TA, Rosen BR. "
        "Simultaneous blood volume and permeability mapping using a single Gd-based contrast injection. "
        "Proc SMR 2nd Annual Meeting. 1994:279."
    ),
    "woolrich_2009": (
        "Woolrich MW, Jbabdi S, Patenaude B, et al. "
        "Bayesian analysis of neuroimaging data in FSL. "
        "NeuroImage. 2009;45(1 Suppl):S173-S186. doi:10.1016/j.neuroimage.2008.10.055"
    ),
    "wu_2003": (
        "Wu O, Ostergaard L, Weisskoff RM, Benner T, Rosen BR, Sorensen AG. "
        "Tracer arrival timing-insensitive technique for estimating flow in MR perfusion-weighted imaging "
        "using singular value decomposition with a block-circulant deconvolution matrix. "
        "Magn Reson Med. 2003;50(1):164-174. doi:10.1002/mrm.10522"
    ),
    "yarnykh_2007": (
        "Yarnykh VL. "
        "Actual flip-angle imaging in the pulsed steady state: a method for rapid three-dimensional mapping "
        "of the transmitted radiofrequency field. "
        "Magn Reson Med. 2007;57(1):192-200. doi:10.1002/mrm.21120"
    ),
    "zhang_2001": (
        "Zhang Y, Brady M, Smith S. "
        "Segmentation of brain MR images through a hidden Markov random field model "
        "and the expectation-maximization algorithm. "
        "IEEE Trans Med Imaging. 2001;20(1):45-57. doi:10.1109/42.906424"
    ),
}


def _generate_references_text(used_refs: set[str]) -> str:
    """Generate formatted references section for used citations.

    Args:
        used_refs: Set of reference keys that were used.

    Returns:
        Formatted references text with numbered entries.
    """
    if not used_refs:
        return ""
    # Sort by first author surname
    sorted_keys = sorted(used_refs, key=lambda k: _REFERENCES[k].split()[0].lower())
    lines = [f"[{i + 1}] {_REFERENCES[k]}" for i, k in enumerate(sorted_keys)]
    return "References:\n" + "\n".join(lines)


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
        Formatted methods text with references section.
    """
    sentences: list[str] = []
    used_refs: set[str] = {"gorgolewski_2011", "rulseh_inprep"}  # Always include Nipype and PMRIPrep

    # Opening sentence
    sentences.append(
        f"Data were preprocessed using PMRIPrep version {version} "
        "(Rulseh et al., in preparation), a Nipype-based (Gorgolewski et al., 2011) "
        "tool for MRI perfusion and permeability analysis."
    )

    # Anatomical preprocessing
    anat_data = pipeline_data.get("anat")
    if ran_pipelines.get("anat") and anat_data is not None:
        anat_qc = anat_data.get("qc_metrics", {})
        anat_text, anat_refs = _build_anat_boilerplate(anat_qc)
        if anat_text:
            sentences.append(anat_text)
            used_refs.update(anat_refs)

    # Tumor segmentation
    if ran_pipelines.get("tumor_seg"):
        sentences.append(
            "Automated lesion segmentation was performed using HD-GLIO "
            "(Kickingereder et al., 2019), providing lesion masks for "
            "AIF candidate exclusion and contralateral NAWM ROI "
            "lateralization."
        )
        used_refs.add("kickingereder_2019")

    # DSC perfusion
    dsc_data = pipeline_data.get("dsc")
    if ran_pipelines.get("dsc") and dsc_data is not None:
        dsc_qc = dsc_data.get("qc_metrics", {})
        dsc_text, dsc_refs = _build_dsc_boilerplate(dsc_qc)
        if dsc_text:
            sentences.append(dsc_text)
            used_refs.update(dsc_refs)

    # DCE perfusion
    dce_data = pipeline_data.get("dce")
    if ran_pipelines.get("dce") and dce_data is not None:
        dce_qc = dce_data.get("qc_metrics", {})
        dce_text, dce_refs = _build_dce_boilerplate(dce_qc)
        if dce_text:
            sentences.append(dce_text)
            used_refs.update(dce_refs)

    # Build final text: methods as single paragraph, then references
    methods_text = " ".join(sentences)
    if used_refs:
        refs_text = _generate_references_text(used_refs)
        return f"{methods_text}\n\n{refs_text}"
    return methods_text


def _build_anat_boilerplate(anat_qc: dict[str, Any]) -> tuple[str, set[str]]:
    """Build anatomical preprocessing boilerplate text.

    Args:
        anat_qc: Anatomical QC metrics dict.

    Returns:
        Tuple of (methods text, set of reference keys used).
    """
    sentences: list[str] = []
    refs: set[str] = set()

    # Bias field correction (always performed)
    bias_text = (
        "Anatomical images were corrected for intensity non-uniformity using "
        "ANTs N4BiasFieldCorrection (Tustison et al., 2010)"
    )
    refs.add("tustison_2010")
    # Check for denoising
    preprocessing = anat_qc.get("preprocessing", {})
    if preprocessing.get("denoising"):
        bias_text += " with prior denoising using ANTs DenoiseImage"
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
        if tool == "hdbet":
            refs.add("isensee_2019")
        else:
            refs.add("avants_2011")
    else:
        sentences.append("Brain extraction was performed.")

    # Tissue segmentation (check for tissue_segmentation key)
    if anat_qc.get("tissue_segmentation"):
        sentences.append(
            "Tissue segmentation (GM, WM, CSF) was performed using "
            "FSL FAST (Zhang et al., 2001)."
        )
        refs.add("zhang_2001")

    sentences.append(
        "Secondary structural modalities were coregistered to the "
        "T1-weighted reference using FLIRT (Jenkinson et al., 2002). "
        "A rigid-body transform between T1-weighted and MNI152 space "
        "was computed using FLIRT."
    )
    refs.add("jenkinson_2002")

    return " ".join(sentences), refs


def _build_dsc_boilerplate(dsc_qc: dict[str, Any]) -> tuple[str, set[str]]:
    """Build DSC perfusion boilerplate text.

    Args:
        dsc_qc: DSC QC metrics dict.

    Returns:
        Tuple of (methods text, set of reference keys used).
    """
    sentences: list[str] = []
    refs: set[str] = set()

    # Opening: motion correction and concentration conversion
    sentences.append(
        "DSC perfusion images were motion-corrected using FSL MCFLIRT "
        "(Jenkinson et al., 2002) and converted to concentration-time "
        "curves via ΔR2* estimation."
    )
    refs.add("jenkinson_2002")

    # Baseline detection
    sentences.append(
        "Baseline frames were automatically identified using a "
        "multi-phase detection algorithm."
    )

    # Registration
    sentences.append(
        "A rigid-body transform between DSC and T1-weighted space "
        "was computed using FLIRT (Jenkinson et al., 2002)."
    )

    # AIF detection
    aif_info = dsc_qc.get("aif_detection") or dsc_qc.get("aif", {})
    config_info = dsc_qc.get("config", {})
    aif_source = config_info.get("aif_source") or aif_info.get("aif_method", "auto")

    if aif_source == "synthetic":
        sentences.append(
            "A synthetic population-derived AIF was used, based on "
            "gamma-variate parameters from the literature "
            "(Parker et al., 2006; Mouridsen et al., 2006)."
        )
        refs.update(["parker_2006", "mouridsen_2006"])
    elif aif_source == "external":
        sentences.append("A user-supplied arterial input function was used.")
    else:
        sentences.append(
            "The DSC arterial input function (AIF) was automatically "
            "detected using MRA-atlas-guided voxel selection with a "
            "multi-level criteria cascade and bilateral hemisphere "
            "preference (Rulseh et al., in preparation)."
        )

    # Deconvolution method
    deconv = dsc_qc.get("deconvolution", {})
    method = deconv.get("method", "oSVD")
    method_text = _DSC_DECONV_TEXT.get(method, _DSC_DECONV_TEXT.get("oSVD", ""))
    sentences.append(f"Perfusion parameter maps were quantified using {method_text}.")

    # Add deconvolution method references
    method_lower = method.lower()
    if method_lower == "ssvd":
        refs.add("ostergaard_1996")
    elif method_lower in ("csvd", "osvd"):
        refs.add("wu_2003")
    elif method_lower == "tikhonov":
        refs.update(["hansen_1992", "calamante_2000"])
    elif method_lower == "vascular":
        refs.add("mouridsen_2006")
    elif method_lower == "cpi":
        refs.add("mehndiratta_2013")

    # Inference engine (only for vascular/cpi parametric methods).
    # Backward-compat: older sidecars may carry "bayesian"/"bayesian_inference"
    # keys; their values are already in canonical "vi"/"svb"/"map" naming.
    if method_lower in ("vascular", "cpi"):
        inference = deconv.get(
            "inference",
            deconv.get("bayesian_inference", deconv.get("bayesian", "vi")),
        )
        if inference in _INFERENCE_TEXT:
            sentences.append(_INFERENCE_TEXT[inference])
            if inference == "vi":
                refs.add("chappell_2009")
            elif inference == "svb":
                refs.add("groves_2009")
                refs.add("penny_2005")
            elif inference == "map":
                refs.add("woolrich_2009")

    # Leakage correction
    preload = dsc_qc.get("preload", {})
    if preload.get("preload_correction_applied"):
        k2_method = preload.get("k2_method", "bsw")
        if k2_method == "bidirectional":
            sentences.append(
                "Bidirectional K2 leakage correction was applied to account for "
                "contrast extravasation and washout (Leu et al., 2016)."
            )
            refs.add("leu_2016")
        else:
            sentences.append(
                "K2 leakage correction was applied to account for contrast "
                "extravasation (Weisskoff et al., 1994; Boxerman et al., 2006)."
            )
            refs.update(["weisskoff_1994", "boxerman_2006"])

    # NAWM normalization
    norm_factor = dsc_qc.get("normalization_factor")
    if norm_factor is not None and norm_factor > 0:
        sentences.append(
            "Hemodynamic maps (CBF, CBV) were normalized to "
            "automatically segmented contralateral normal-appearing "
            "white matter (NAWM)."
        )

    return " ".join(sentences), refs


def _build_dce_boilerplate(dce_qc: dict[str, Any]) -> tuple[str, set[str]]:
    """Build DCE perfusion boilerplate text.

    Args:
        dce_qc: DCE QC metrics dict.

    Returns:
        Tuple of (methods text, set of reference keys used).
    """
    sentences: list[str] = []
    refs: set[str] = set()

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
        t1_correction = t1_info.get("t1_correction", {})
        tissue_used = (
            t1_correction.get("tissue_t1_used", False)
            or t1_info.get("tissue_t1_used", False)
        )
        if t1_method == "assumed" and tissue_used:
            sentences.append(
                opening + " using a tissue-segmented T1 map constructed "
                "from WM/GM probability maps with literature T1 values "
                "(no VFA T1 mapping performed)."
            )
        else:
            sentences.append(opening + f" {t1_text}.")
    else:
        # Full T1 mapping text
        sentences.append(opening + ".")
        t1_sentence = t1_text
        if b1_text:
            t1_sentence += b1_text
        t1_sentence += "."
        sentences.append(t1_sentence)
        if t1_method in ("hybrid", "wlls", "nlls"):
            refs.add("deoni_2003")
        if b1_method == "afi":
            refs.add("yarnykh_2007")

        # T1 correction status
        t1_correction = t1_info.get("t1_correction", {})
        corr_applied = t1_correction.get("correction_applied", "none")
        if corr_applied == "offset":
            sentences.append(
                "An offset correction was applied to the T1 map to "
                "adjust for systematic overestimation."
            )
        elif t1_correction.get("tissue_t1_used", False):
            sentences.append(
                "Due to poor VFA conditioning, a tissue-segmented T1 "
                "map constructed from WM/GM probability maps using "
                "literature T1 values was used for concentration "
                "conversion."
            )

    # Registration
    sentences.append(
        "A rigid-body transform between DCE and T1-weighted space "
        "was computed using FLIRT (Jenkinson et al., 2002)."
    )
    refs.add("jenkinson_2002")

    # AIF detection
    aif_info = dce_qc.get("aif_detection", {})
    aif_source = aif_info.get("aif_source") or aif_info.get("detection_method", "auto")

    if aif_source == "population":
        sentences.append(
            "The Parker population AIF was used (Parker et al., 2006)."
        )
        refs.add("parker_2006")
    elif aif_source == "external":
        sentences.append("A user-supplied arterial input function was used.")
    else:
        sentences.append(
            "The DCE arterial input function (AIF) was automatically "
            "detected using MRA-atlas-guided candidate selection with "
            "dose-aware scaling and quality-weighted averaging "
            "(Rulseh et al., in preparation)."
        )

    # Hematocrit correction (only for auto-detected AIF; population AIF
    # already includes hematocrit correction)
    if aif_source not in ("population", "external"):
        hct = aif_info.get("hematocrit", 0.42)
        sentences.append(
            f"The AIF was corrected from whole-blood to plasma "
            f"concentration using an assumed hematocrit of {hct}."
        )

    # Pharmacokinetic modeling
    pk_info = dce_qc.get("pk_modeling", {})
    model = pk_info.get("model", "auto")

    # Track models used for references
    models_used: list[str] = []
    if model == "auto":
        candidates = pk_info.get("candidate_models", [])
        if candidates:
            models_used = candidates
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
        models_used = [model]
        model_text = _DCE_PK_MODEL_TEXT.get(model, model)
        sentences.append(
            f"Pharmacokinetic analysis was performed using {model_text}."
        )

    # Add PK model references
    for m in models_used:
        m_lower = m.lower()
        if m_lower == "tofts":
            refs.add("tofts_1991")
        elif m_lower == "extended_tofts":
            refs.add("tofts_1999")
        elif m_lower == "patlak":
            refs.add("patlak_1983")
        elif m_lower == "2cxm":
            refs.add("sourbron_2011")

    # Inference engine — DCE NLLS path produces no boilerplate sentence
    # (the `nlls` entry is generic; emit only when a Bayesian engine
    # was actually used).
    inference = pk_info.get("inference", "nlls")
    inference_text = _INFERENCE_TEXT.get(inference, "")
    if inference_text:
        sentences.append(inference_text)
        if inference == "vi":
            refs.add("chappell_2009")
        elif inference == "map":
            refs.add("woolrich_2009")

    # Tortuosity correction
    if dce_qc.get("tortuosity_applied", False):
        sentences.append(
            "A tortuosity-corrected Ktrans map was additionally "
            "computed using DWI-derived ADC "
            "(Nicholson & Syková, 1998; Tofts, 1997)."
        )
        refs.update(["nicholson_1998", "tofts_1997"])

    # NAWM normalization
    if dce_qc.get("nawm_normalization", False):
        sentences.append(
            "Permeability maps (Ktrans, IAUC60) were normalized to "
            "automatically segmented contralateral normal-appearing "
            "white matter (NAWM)."
        )

    return " ".join(sentences), refs


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
        "model_selection_qc", "dsc_dce_xval", "registration",
    ],
}

# Per-image canonical sidecar paths whose existence proves the pipeline
# ran. These replace the legacy ``desc-*QC_quality.json`` summary files
# at session root (deleted in v0.46.0). The aggregator uses them as
# presence probes AND as the source of QC* recommendation/warnings flat
# keys.
_PIPELINE_PRESENCE_PROBES: dict[str, str] = {
    "anat": "anat/{prefix}_space-T1w_desc-preproc_T1w.json",
    "tumor_seg": "anat/{prefix}_space-T1w_desc-tumorseg_dseg.json",
    "dsc": "perf/{prefix}_desc-dscPreproc_dsc.json",
    "dce": "perf/{prefix}_desc-dcePreproc_dce.json",
}

# Pipeline-internal resume sidecars carrying the verbatim producer-shape
# qc_metrics envelope. Used by the HTML aggregator to recover the
# nested-dict shape boilerplate generators expect, mirroring the
# DSC --skip-dsc round-trip pattern.
_PIPELINE_RESUME_SIDECARS: dict[str, str] = {
    "dsc": "perf/{prefix}_desc-dscResume_pmriprep.json",
    "dce": "perf/{prefix}_desc-dceResume_pmriprep.json",
    "tumor_seg": "anat/{prefix}_desc-tumorResume_pmriprep.json",
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
    """Collect figure relative paths for a single pipeline.

    Scans *figures_dir* for matching files and returns relative paths
    suitable for ``<img src="...">`` in the HTML report (NiPreps pattern).

    Args:
        subject_id: BIDS subject ID.
        session_id: BIDS session ID.
        pipeline: Pipeline name.
        figures_dir: Absolute path to figures directory.

    Returns:
        Dict mapping image key to relative path string
        (e.g. ``figures/sub-01_ses-01_desc-brainmask_dsc.png``).
    """
    images: dict[str, str] = {}
    for key in _PIPELINE_FIGURE_KEYS.get(pipeline, []):
        fname = figure_filename(subject_id, session_id, key, pipeline)
        fpath = figures_dir / fname
        if fpath.exists():
            images[key] = f"figures/{fname}"
    return images


def _classify_metric(
    value: float | None,
    warn_threshold: float,
    min_threshold: float,
) -> str:
    """Classify a QC metric as pass/warn/poor for template display.

    Args:
        value: Metric value (or None/NaN if unavailable).
        warn_threshold: Values >= this are "pass".
        min_threshold: Values >= this (but < warn) are "warn".

    Returns:
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

    Args:
        value: Mean B1+ value (or None/NaN if unavailable).
        low: Lower acceptable bound (inclusive).
        high: Upper acceptable bound (inclusive).

    Returns:
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

    Args:
        value: Metric value (or None/NaN if unavailable).
        warn_threshold: Values < this are "pass".
        poor_threshold: Values < this (but >= warn) are "warn".

    Returns:
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


def _qc_keys_from_anchor(anchor: dict[str, Any]) -> dict[str, Any]:
    """Translate flat ``QC*`` recommendation/warnings keys to legacy nested.

    The pre-v0.46.0 aggregator read ``recommendation``, ``warnings``,
    ``provenance_warnings``, and ``acquisition_warnings`` as top-level
    keys on ``qc_metrics``. PR2 moves them onto the canonical anchor
    sidecars as flat ``QC*``-prefixed keys (per bids-spec-expert
    guidance). This translates them back to the nested shape so
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
    return out


def _load_pipeline_sidecars(
    ses_dir: Path, prefix: str, pipeline: str,
) -> dict[str, Any]:
    """Build the legacy ``qc_metrics`` nested shape from per-image sidecars.

    Replaces the pre-v0.46.0 ``_load_qc_json`` helper, which read a
    single ``desc-*QC_quality.json`` summary file at session root. The
    summary files are gone after v0.46.0; this loader composes the same
    nested dict from:

    - The anchor sidecar (``QC*`` recommendation/warnings flat keys).
    - The pipeline-internal resume sidecar (full producer-shape
      ``qc_metrics`` envelope; DSC/DCE only).

    For pipelines without a resume sidecar (anat / tumor_seg) the
    anchor sidecar's QC* keys are surfaced; for anat, the loader also
    reads sibling brain-mask and probseg sidecars to recover the
    ``preprocessing`` / ``brain_extraction`` / ``tissue_segmentation``
    nested fields that ``_build_anat_boilerplate`` consumes.
    """
    anchor_pattern = _PIPELINE_PRESENCE_PROBES.get(pipeline)
    if not anchor_pattern:
        return {}
    anchor_path = ses_dir / anchor_pattern.format(prefix=prefix)
    anchor = _read_json_safe(anchor_path)

    qc_metrics: dict[str, Any] = {}

    resume_pattern = _PIPELINE_RESUME_SIDECARS.get(pipeline)
    if resume_pattern:
        resume_path = ses_dir / resume_pattern.format(prefix=prefix)
        resume = _read_json_safe(resume_path)
        if pipeline == "dsc":
            # DSC resume carries verbatim aif_qc / preload_qc /
            # preprocessing_stats; legacy qc_metrics nests them under
            # aif_detection / preload / preprocessing.
            aif_qc = resume.get("aif_qc")
            if isinstance(aif_qc, dict):
                qc_metrics["aif_detection"] = dict(aif_qc)
            preload = resume.get("preload_qc")
            if isinstance(preload, dict):
                qc_metrics["preload"] = dict(preload)
            preproc = resume.get("preprocessing_stats")
            if isinstance(preproc, dict):
                qc_metrics["preprocessing"] = dict(preproc)
        elif pipeline == "dce":
            # DCE resume stores the full producer-shape envelope verbatim.
            envelope = resume.get("qc_metrics")
            if isinstance(envelope, dict):
                qc_metrics.update(dict(envelope))
        elif pipeline == "tumor_seg":
            # Tumor-seg resume carries the same envelope as DCE
            # (preprocessing / segmentation / nawm blocks).
            envelope = resume.get("qc_metrics")
            if isinstance(envelope, dict):
                qc_metrics.update(dict(envelope))

    if pipeline == "anat":
        qc_metrics.update(_anat_nested_from_sidecars(ses_dir, prefix, anchor))

    # The two updaters write disjoint key sets:
    # _anat_nested_from_sidecars / resume → preprocessing /
    #     brain_extraction / tissue_segmentation / aif_detection /
    #     preload / pk_modeling / etc.
    # _qc_keys_from_anchor → recommendation / warnings /
    #     provenance_warnings / acquisition_warnings.
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
    if "PreprocessingBiasCorrection" in anchor:
        preproc["bias_correction"] = str(anchor["PreprocessingBiasCorrection"])
    if "PreprocessingACPCAligned" in anchor:
        preproc["acpc_aligned"] = bool(anchor["PreprocessingACPCAligned"])
    if preproc:
        out["preprocessing"] = preproc

    # brain_extraction.tool — read from the brain-mask sidecar
    mask_path = (
        ses_dir / "anat" / f"{prefix}_space-T1w_desc-brain_mask.json"
    )
    mask_sidecar = _read_json_safe(mask_path)
    brain_ex: dict[str, Any] = {}
    if "BrainExtractionTool" in mask_sidecar:
        # Normalize tool name to the lowercase token the boilerplate
        # expects (e.g. "HD-BET" → "hdbet").
        raw_tool = str(mask_sidecar["BrainExtractionTool"])
        brain_ex["tool"] = raw_tool.lower().replace("-", "")
    if "BrainExtractionVersion" in mask_sidecar:
        brain_ex["version"] = str(mask_sidecar["BrainExtractionVersion"])
    if "BrainExtractionVolume" in mask_sidecar:
        brain_ex["brain_volume_ml"] = float(mask_sidecar["BrainExtractionVolume"])
    if "BrainExtractionVolumeMm3" in mask_sidecar:
        brain_ex["brain_volume_mm3"] = float(mask_sidecar["BrainExtractionVolumeMm3"])
    elif "BrainExtractionVolume" in mask_sidecar:
        # Fallback when only the mL value was written.
        brain_ex["brain_volume_mm3"] = (
            float(mask_sidecar["BrainExtractionVolume"]) * 1000.0
        )
    if "BrainExtractionVoxelCount" in mask_sidecar:
        brain_ex["voxel_count"] = int(mask_sidecar["BrainExtractionVoxelCount"])
    if "BrainExtractionQCStatus" in mask_sidecar:
        brain_ex["qc_status"] = str(mask_sidecar["BrainExtractionQCStatus"])
    if "BrainExtractionWarnings" in mask_sidecar:
        brain_ex["warnings"] = list(mask_sidecar["BrainExtractionWarnings"])
    if brain_ex:
        out["brain_extraction"] = brain_ex

    # tissue_segmentation — presence of any probseg sidecar is sufficient
    probseg_dir = ses_dir / "anat"
    if probseg_dir.is_dir():
        probsegs: dict[str, Any] = {}
        for tissue in ("WM", "GM", "CSF"):
            ps = probseg_dir / f"{prefix}_space-T1w_label-{tissue}_probseg.json"
            if ps.exists():
                ps_meta = _read_json_safe(ps)
                tissue_block: dict[str, Any] = {}
                if "TissueSegmentationMeanProbability" in ps_meta:
                    tissue_block["mean_probability"] = float(
                        ps_meta["TissueSegmentationMeanProbability"],
                    )
                if "TissueSegmentationVoxelCount" in ps_meta:
                    tissue_block["voxel_count"] = int(
                        ps_meta["TissueSegmentationVoxelCount"],
                    )
                if "TissueSegmentationVolume" in ps_meta:
                    tissue_block["volume_ml"] = float(
                        ps_meta["TissueSegmentationVolume"],
                    )
                probsegs[tissue] = tissue_block
        if probsegs:
            out["tissue_segmentation"] = probsegs

    return out


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
        "dce": "DCE Permeability",
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

    Args:
        pipeline_data: Dict of pipeline name to data dict (or None).

    Returns:
        Dict mapping pipeline display name to list of provenance warning
        strings. Empty dict if no provenance warnings exist.
    """
    result: dict[str, list[str]] = {}
    display_names = {
        "dsc": "DSC Perfusion",
        "dce": "DCE Permeability",
    }
    for name in ("dsc", "dce"):
        data = pipeline_data.get(name)
        if data is None:
            continue
        prov_w = data.get("qc_metrics", {}).get("provenance_warnings", [])
        if prov_w:
            result[display_names[name]] = prov_w
    return result


def _collect_acquisition_warnings(
    pipeline_data: dict[str, Optional[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Collect acquisition/data-quality warnings for dedicated HTML banner.

    Reads the ``acquisition_warnings`` key from QC metrics, which holds
    preflight warnings and AIF fallback messages.

    Args:
        pipeline_data: Dict of pipeline name to data dict (or None).

    Returns:
        Dict mapping pipeline display name to list of acquisition warning
        strings. Empty dict if no acquisition warnings exist.
    """
    result: dict[str, list[str]] = {}
    display_names = {
        "dsc": "DSC Perfusion",
        "dce": "DCE Permeability",
    }
    for name in ("dsc", "dce"):
        data = pipeline_data.get(name)
        if data is None:
            continue
        acq_w = data.get("qc_metrics", {}).get("acquisition_warnings", [])
        if acq_w:
            result[display_names[name]] = acq_w
    return result


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

        qc_metrics = _load_pipeline_sidecars(ses_path, prefix, pipeline_name)
        images = _collect_pipeline_images(
            subject_id, session_id, pipeline_name, fig_path,
        )

        # Compute display status for metrics (NiPreps: logic in Python, not templates)
        display_status = None
        if pipeline_name == "dce" and qc_metrics:
            preproc = qc_metrics.get("preprocessing", {})
            b1 = qc_metrics.get("b1_mapping", {})
            pk = qc_metrics.get("pk_modeling", {})
            xval = qc_metrics.get("dsc_dce_validation", {})
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
                # Cross-validation correlation
                "xval_correlation": _classify_metric(
                    xval.get("metrics", {}).get("correlation"), 0.7, 0.4
                ),
                # CBV constraint negative fraction (lower is better)
                "cbv_negative": _classify_metric_inverse(
                    cbv_c.get("cbv_negative_fraction"), 0.01, 0.05
                ),
            }
        elif pipeline_name == "dsc" and qc_metrics:
            preproc = qc_metrics.get("preprocessing", {})
            preload = qc_metrics.get("preload", {})
            deconv = qc_metrics.get("deconvolution", {})
            display_status = {
                # Motion outliers (lower is better)
                "outliers": _classify_metric_inverse(
                    preproc.get("percent_outliers"), 5, 10
                ),
                # Leaky voxels (lower is better)
                "leaky_voxels": _classify_metric_inverse(
                    preload.get("percent_leaky_voxels"), 5, 20
                ),
                # Deconvolution fit quality
                "rsquared": _classify_metric(
                    deconv.get("qc_voxel_rsquared"), 0.9, 0.7,
                ),
                # oSVD fallback rate for parametric methods (lower is better:
                # high fallback fraction means few voxels passed the R² gate).
                "fallback": _classify_metric_inverse(
                    deconv.get("fallback_rate"), 0.10, 0.25
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
    acquisition_warnings = _collect_acquisition_warnings(pipeline_data)
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
        "provenance_warnings": provenance_warnings,
        "acquisition_warnings": acquisition_warnings,
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
