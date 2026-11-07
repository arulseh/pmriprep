"""Provenance tracking utilities for BIDS-Derivatives."""

import datetime
import json
import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pmriprep

logger = logging.getLogger(__name__)

def get_software_versions() -> Dict[str, str]:
    """Get versions of key software dependencies.

    Returns a dictionary with version information for:
    - Python
    - pmriprep
    - nipype
    - nibabel
    - numpy
    - scipy
    - nilearn

    Returns
    -------
    dict
        Dictionary mapping package names to version strings.
    """
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "pmriprep": pmriprep.__version__,
    }

    # Core dependencies - import individually to handle missing packages
    optional_packages = [
        "nipype",
        "nibabel",
        "numpy",
        "scipy",
        "nilearn",
        "niworkflows",
        "templateflow",
    ]

    for pkg_name in optional_packages:
        try:
            pkg = __import__(pkg_name)
            if hasattr(pkg, "__version__"):
                versions[pkg_name] = str(pkg.__version__)
            elif hasattr(pkg, "version"):
                versions[pkg_name] = str(pkg.version)
            else:
                versions[pkg_name] = "unknown"
        except ImportError:
            versions[pkg_name] = "not installed"

    return versions



def create_provenance_dict(
    source_files: Optional[List[Union[str, Path]]] = None,
    processing_description: Optional[str] = None,
    config_used: Optional[Dict[str, Any]] = None,
    scanner_metadata: Optional[Dict[str, Any]] = None,
    include_timestamp: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a provenance dictionary for BIDS-Derivatives JSON sidecars.

    This follows the BIDS-Derivatives specification for provenance tracking,
    including information about source files, processing parameters, and
    software versions.

    Parameters
    ----------
    source_files : list of str or Path, optional
        List of paths to input files used to generate this output.
    processing_description : str, optional
        Human-readable description of the processing.
    config_used : dict, optional
        Configuration parameters used for processing.
    scanner_metadata : dict, optional
        Scanner metadata from BIDS JSON sidecar.
    include_timestamp : bool, optional
        Whether to include processing timestamp.
    additional_metadata : dict, optional
        Any additional key-value pairs to include.

    Returns
    -------
    dict
        Dictionary suitable for JSON serialization as a sidecar file.
    """
    provenance: Dict[str, Any] = {}

    # Processing timestamp
    if include_timestamp:
        provenance["GeneratedBy"] = [
            {
                "Name": "pmriprep",
                "Version": pmriprep.__version__,
            }
        ]
        provenance["ProcessingDateTime"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

    # Source files
    if source_files:
        provenance["Sources"] = [str(Path(f).name) for f in source_files]

    # Processing description
    if processing_description:
        provenance["Description"] = processing_description

    # Configuration used (strip internal/runtime fields and None values)
    if config_used:
        _PROVENANCE_EXCLUDE = {
            "hdbet_device", "hdbet_conda_env",
            "num_processes",
        }
        provenance["ProcessingConfiguration"] = {
            k: v for k, v in config_used.items()
            if k not in _PROVENANCE_EXCLUDE and v is not None
        }

    # Scanner metadata
    if scanner_metadata:
        provenance["ScannerMetadata"] = scanner_metadata

    # Additional metadata
    if additional_metadata:
        for key, value in additional_metadata.items():
            if key not in provenance:
                provenance[key] = value

    return provenance


def write_provenance_json(
    output_path: Union[str, Path],
    provenance: Dict[str, Any],
    indent: int = 2,
) -> Path:
    """Write a provenance dictionary to a JSON file.

    Parameters
    ----------
    output_path : str or Path
        Path for the output JSON file.
    provenance : dict
        Provenance dictionary to write.
    indent : int, optional
        JSON indentation level (default: 2).

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(provenance, f, indent=indent, default=str)

    return output_path



def get_bids_derivative_provenance(
    pipeline_name: str = "pmriprep",
    pipeline_version: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a minimal provenance dict following BIDS-Derivatives GeneratedBy spec.

    This creates the standard "GeneratedBy" field used in dataset_description.json
    and individual derivative sidecars.

    Parameters
    ----------
    pipeline_name : str, optional
        Name of the pipeline (default: "pmriprep").
    pipeline_version : str, optional
        Version string. If None, uses pmriprep.__version__.
    description : str, optional
        Optional description of what the pipeline does.

    Returns
    -------
    dict
        Dictionary with GeneratedBy field following BIDS spec.
    """
    version = pipeline_version or pmriprep.__version__

    generated_by: Dict[str, Any] = {
        "Name": pipeline_name,
        "Version": version,
    }

    if description:
        generated_by["Description"] = description

    return {"GeneratedBy": [generated_by]}


def _safe_float(value: Any) -> Optional[float]:
    """Convert to Python float or return None."""
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def enrich_provenance_from_qc(
    sidecar: Dict[str, Any],
    qc_metrics: Dict[str, Any],
    modality: Literal["DCE", "DSC"],
) -> Dict[str, Any]:
    """Enrich sidecar PerfusionProcessing with dynamic QC metrics.

    Merges processing parameters from QC metrics into the existing
    PerfusionProcessing namespace in the sidecar. Missing QC keys
    are silently skipped.

    Parameters
    ----------
    sidecar : dict
        Existing sidecar dict containing the PerfusionProcessing namespace.
    qc_metrics : dict
        QC metrics dict from the writer's qc_metrics input.
    modality : {"DCE", "DSC"}
        "DCE" or "DSC".

    Returns
    -------
    dict
        Dict with updated PerfusionProcessing to merge into sidecar.
    """
    perf = sidecar.get("PerfusionProcessing", {})
    if not perf:
        return {}

    result: Dict[str, Any] = {"PerfusionProcessing": dict(perf)}
    out = result["PerfusionProcessing"]

    if modality == "DCE":
        _enrich_dce(out, qc_metrics)
    elif modality == "DSC":
        _enrich_dsc(out, qc_metrics)

    return result


def build_model_selection_provenance(pk: Dict[str, Any]) -> Dict[str, Any]:
    """Describe automatic candidates and fractions among selected model labels."""
    selection: Dict[str, Any] = {"Method": "per-voxel BIC comparison"}
    candidates = pk.get("candidate_models", [])
    if candidates:
        selection["CandidateModels"] = list(candidates)
        distribution = {
            name: float(pk[f"{name}_fraction"])
            for name in candidates if pk.get(f"{name}_fraction") is not None
        }
        if distribution:
            selection["ModelDistribution"] = distribution
            selection["ModelDistributionUnits"] = "fraction"
    return selection


def _enrich_dce(
    perf: Dict[str, Any],
    qc: Dict[str, Any],
) -> None:
    """Enrich DCE PerfusionProcessing in-place from QC metrics."""
    # T1Mapping
    t1 = qc.get("t1_mapping")
    if t1 and isinstance(t1, dict):
        t1_block: Dict[str, Any] = {}
        method = t1.get("method", t1.get("t1_method"))
        if method is not None:
            t1_block["method"] = str(method)
        if "b1_correction" in t1:
            t1_block["B1_correction"] = str(t1["b1_correction"])
        elif "b1_corrected" in t1:
            t1_block["B1_correction"] = bool(t1["b1_corrected"])
        if "b1_valid_fraction" in t1:
            t1_block["B1_valid_fraction"] = _safe_float(
                t1["b1_valid_fraction"],
            )
        if "tissue_statistics_source" in t1:
            t1_block["tissue_statistics_source"] = str(
                t1["tissue_statistics_source"],
            )
        correction = t1.get("t1_correction")
        if isinstance(correction, dict):
            t1_block["correction"] = dict(correction)
        if method == "assumed":
            t1_block["tissue_segmented"] = bool(t1.get("tissue_t1_used", False))
            if "assumed_t1_ms" in t1 and t1["assumed_t1_ms"] is not None:
                t1_block["AssumedT1_ms"] = _safe_float(t1["assumed_t1_ms"])
            if (
                "assumed_t1_field_strength_t" in t1
                and t1["assumed_t1_field_strength_t"] is not None
            ):
                t1_block["FieldStrength_T"] = _safe_float(
                    t1["assumed_t1_field_strength_t"]
                )
            if "assumed_t1_policy" in t1 and t1["assumed_t1_policy"] is not None:
                t1_block["assumption_policy"] = str(t1["assumed_t1_policy"])
            if "mean_t1_wm_ms" in t1:
                t1_block["T1_WM_ms"] = _safe_float(t1["mean_t1_wm_ms"])
            if "mean_t1_gm_ms" in t1:
                t1_block["T1_GM_ms"] = _safe_float(t1["mean_t1_gm_ms"])
        if t1_block:
            perf["T1Mapping"] = t1_block

    conversion = qc.get("signal_conversion")
    if isinstance(conversion, dict):
        settings = {
            key: conversion[key] for key in (
                "t1_source", "assumed_t1_ms", "assumed_t1_field_strength_t",
                "assumed_t1_policy", "b1_correction_applied",
            ) if key in conversion
        }
        if settings:
            perf["SignalConversion"] = settings

    # AIF detection
    aif = qc.get("aif_detection")
    if aif and isinstance(aif, dict):
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        canonical_aif = dict(aif)
        if "detection_method" not in canonical_aif:
            source = str(canonical_aif.get("aif_source", ""))
            if canonical_aif.get("population_aif") or source == "population":
                canonical_aif["detection_method"] = "population"
            elif source == "external":
                canonical_aif["detection_method"] = "external"
            elif source:
                canonical_aif["detection_method"] = "atlas"
        aif_block = build_aif_provenance_block(
            canonical_aif,
            modality="DCE",
            compact=True,
        )
        if aif_block:
            perf["AIF"] = aif_block

        # External AIFs are supplied in plasma units; internally generated
        # curves undergo blood-to-plasma conversion. Emit unit traceability
        # only when a peak or hematocrit was recorded.
        final_peak_plasma = aif.get("emitted_peak_mm")
        hct_used = aif_block.get("Hematocrit")
        if hct_used is not None:
            perf["Hematocrit"] = {
                "value": hct_used,
                "Source": "configuration",
                "Units": "1",
            }
        if final_peak_plasma is not None or hct_used is not None:
            units_block: Dict[str, Any] = {
                "description": (
                    "External AIF supplied in plasma [Gd] (mM); "
                    "no hematocrit conversion applied."
                    if aif_block.get("Source") == "external"
                    else "AIF is in plasma [Gd] (mM) after /(1-Hct) conversion."
                ),
            }
            if final_peak_plasma is not None:
                units_block["aif_peak_plasma_mM"] = _safe_float(
                    final_peak_plasma
                )
            if hct_used is not None:
                units_block["hematocrit_used"] = _safe_float(hct_used)
            perf["AIFUnits"] = units_block

        # Literature blood T1 is currently an AIF amplitude/QC reference; the
        # detected magnitude signal is converted upstream with the tissue T1
        # policy. Name and role must not imply arterial signal conversion.
        if "blood_t1_seconds" in aif and aif["blood_t1_seconds"]:
            fs = perf.get("Relaxivity", {}).get("field_strength_T")
            if fs is None:
                fs = _safe_float(aif.get("field_strength_t"))
            perf["BloodT1Reference"] = {
                "value_s": _safe_float(aif["blood_t1_seconds"]),
                "Source": "literature_lookup",
                "Role": "aif_expected_peak_and_qc_reference",
                "AppliedToArterialSignalConversion": False,
            }
            if fs is not None:
                perf["BloodT1Reference"]["field_strength_T"] = fs

    # PK Modeling (consolidates former top-level PharmacokineticModel,
    # BayesianInference, ModelSelection)
    pk = qc.get("pk_modeling")
    if pk and isinstance(pk, dict):
        pk_block: Dict[str, Any] = {}
        if "model" in pk:
            pk_block["model"] = str(pk["model"])
        if "inference" in pk:
            pk_block["inference"] = str(pk["inference"])
        if pk.get("model") == "auto":
            pk_block["model_selection"] = build_model_selection_provenance(pk)
        if "r2_median" in pk:
            pk_block["r2_median"] = _safe_float(pk["r2_median"])
        if pk_block:
            perf["PKModeling"] = pk_block


def _enrich_dsc(
    perf: Dict[str, Any],
    qc: Dict[str, Any],
) -> None:
    """Enrich DSC PerfusionProcessing in-place from QC metrics."""
    # Deconvolution details
    deconv = qc.get("deconvolution")
    if deconv and isinstance(deconv, dict):
        deconv_block = dict(perf.get("Deconvolution", {}))
        if deconv.get("deconvolution_performed") is False:
            deconv_block = {
                "configured_method": deconv.get("configured_method"),
                "performed": False,
                "unavailable_reason": deconv.get("deconvolution_unavailable_reason"),
            }
        elif deconv.get("deconvolution_performed") is True:
            deconv_block["performed"] = True
        for key in (
            "method",
            "inference",
            "fit_domain",
            "processing_status",
            "unavailable_maps",
            "ncbv_integration_rule",
            "cbv_auc_bat_frame",
            "cbv_auc_end_frame",
            "cpi_dispersion",
            "cpi_infer_delay",
            "cpi_control_points",
            "cpi_leakage_method",
            "cpi_fallback",
            "cpi_selection_stage",
            "cpi_gamma_fraction",
            "spatial_sweeps",
            "regularization_selection",
            "regularization_penalty",
            "penalty_leading_samples",
            "lambda_grid_size",
            "lambda_search_bounds_relative",
            "lambda_continuous_refinement",
            "lcurve_selection_counts",
            "residue_cbv_integration",
            "residue_post_fit_clipping",
            "residue_diagnostics",
            "lcurve_corner_fraction",
            "lcurve_upper_endpoint_fraction",
            "oscillation_target",
            "fallback_oscillation_target",
        ):
            if key in deconv:
                deconv_block[key] = deconv[key]
        if "threshold_mean" in deconv:
            deconv_block["threshold_mean"] = _safe_float(deconv["threshold_mean"])
        if "oi_mean" in deconv:
            deconv_block["oscillation_index_mean"] = _safe_float(deconv["oi_mean"])
        if "lambda_mean" in deconv:
            deconv_block["lambda_mean"] = _safe_float(deconv["lambda_mean"])
        if "solver_convergence_rate" in deconv:
            deconv_block["solver_convergence_rate"] = _safe_float(
                deconv["solver_convergence_rate"]
            )
        if "fallback_rate" in deconv:
            deconv_block["fallback_rate"] = _safe_float(deconv["fallback_rate"])
        perf["Deconvolution"] = deconv_block

    # Leakage correction details
    preload = qc.get("preload")
    if preload and isinstance(preload, dict):
        leak_block = perf.get("LeakageCorrection", {})
        if "preload_correction_applied" in preload:
            leak_block["applied"] = bool(preload["preload_correction_applied"])
        if "leakage_correction_disabled" in preload:
            leak_block["disabled_by_configuration"] = bool(
                preload["leakage_correction_disabled"]
            )
        if "k2_median" in preload:
            leak_block["K2_median"] = float(preload["k2_median"])
        for key in ("nawm_auc_raw_median", "nawm_auc_corrected_median"):
            if key not in preload:
                continue
            leak_block[key] = _safe_float(preload[key])
        perf["LeakageCorrection"] = leak_block

        # Administered-preload provenance and continuous advisory measurement.
        preload_det = perf.get("PreloadDetection", {})
        if "preload_source" in preload:
            preload_det["source"] = str(preload["preload_source"])
        if preload.get("preload_administered_dose_fraction") is not None:
            preload_det["administered_dose_fraction"] = _safe_float(
                preload["preload_administered_dose_fraction"]
            )
        # preload_baseline_delta_r2_advisory: tissue-heterogeneity-biased
        # advisory metric, not used for classification.
        if "preload_baseline_delta_r2_advisory" in preload:
            preload_det["baseline_delta_r2_advisory"] = _safe_float(
                preload["preload_baseline_delta_r2_advisory"]
            )
        perf["PreloadDetection"] = preload_det

    # AIF and baseline frames (both from qc["aif_detection"])
    aif = qc.get("aif_detection")
    if aif and isinstance(aif, dict):
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        aif_block = build_aif_provenance_block(aif, modality="DSC", compact=True)
        if aif_block:
            perf["AIF"] = aif_block

        # Baseline frames (also from aif sub-dict)
        baseline_block: Dict[str, Any] = {}
        if "n_baseline_frames" in aif:
            baseline_block["n_frames"] = int(aif["n_baseline_frames"])
        if "baseline_detection_method" in aif:
            baseline_block["detection_method"] = str(aif["baseline_detection_method"])
        if baseline_block:
            perf["BaselineFrames"] = baseline_block

    # Quantification mode (always relative for DSC)
    perf["QuantificationMode"] = "relative"
