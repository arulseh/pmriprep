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

    Returns:
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
    include_versions: bool = True,
    include_timestamp: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a provenance dictionary for BIDS-Derivatives JSON sidecars.

    This follows the BIDS-Derivatives specification for provenance tracking,
    including information about source files, processing parameters, and
    software versions.

    Args:
        source_files: List of paths to input files used to generate this output.
        processing_description: Human-readable description of the processing.
        config_used: Configuration parameters used for processing.
        scanner_metadata: Scanner metadata from BIDS JSON sidecar.
        include_versions: Whether to include software version information.
        include_timestamp: Whether to include processing timestamp.
        additional_metadata: Any additional key-value pairs to include.

    Returns:
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
            "mcmc_samples", "cpi_control_points",
            "hdbet_device", "hdbet_conda_env",
            "num_processes",
        }
        provenance["ProcessingConfiguration"] = {
            k: v for k, v in config_used.items()
            if k not in _PROVENANCE_EXCLUDE and v is not None
        }

    # Software versions
    if include_versions:
        provenance["SoftwareVersions"] = get_software_versions()

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

    Args:
        output_path: Path for the output JSON file.
        provenance: Provenance dictionary to write.
        indent: JSON indentation level (default: 2).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(provenance, f, indent=indent, default=str)

    return output_path


def write_modality_provenance(
    modality: str,
    qc_metrics: Optional[Dict[str, Any]],
    derivatives_dir: str,
    subject_id: str,
    session_id: Optional[str],
    config_dict: Optional[Dict[str, Any]],
    source_file: Optional[str],
) -> tuple[str, str]:
    """Write QC JSON and provenance JSON for a perfusion modality.

    Shared implementation for DSC and DCE pipelines. Writes two files
    into the perf/ subdirectory of the BIDS-Derivatives output:
      - ``desc-{modality}_qc.json`` — comprehensive QC metrics
      - ``desc-{modality}_provenance.json`` — processing config and versions

    Args:
        modality: Pipeline modality identifier (``"dsc"`` or ``"dce"``).
        qc_metrics: Merged QC metrics dictionary from all pipeline steps.
        derivatives_dir: Path to BIDS derivatives directory.
        subject_id: Subject ID (without ``sub-`` prefix).
        session_id: Session ID (without ``ses-`` prefix), or ``None``.
        config_dict: Processing config as a dictionary.
        source_file: Path to the source perfusion file.

    Returns:
        Tuple of ``(qc_json_path, provenance_json_path)``.
    """
    from pmriprep.utils.outputs import write_qc_json

    sub_dir = f"sub-{subject_id}"
    ses_part = f"ses-{session_id}" if session_id else ""
    if ses_part:
        out_dir = Path(derivatives_dir) / "pmriprep" / sub_dir / ses_part / "perf"
    else:
        out_dir = Path(derivatives_dir) / "pmriprep" / sub_dir / "perf"
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_path = out_dir / f"{sub_dir}{'_' + ses_part if ses_part else ''}_desc-{modality}_qc.json"
    qc_json_path = str(write_qc_json(
        output_path=qc_path,
        metrics=qc_metrics if qc_metrics else {},
        subject=subject_id,
        session=session_id,
        pipeline_step=modality,
    ))

    descriptions = {
        "dsc": "DSC perfusion pipeline (PMRIPrep)",
        "dce": "DCE perfusion pipeline (PMRIPrep)",
    }
    provenance = create_provenance_dict(
        source_files=[source_file] if source_file else None,
        processing_description=descriptions.get(modality, f"{modality} pipeline (PMRIPrep)"),
        config_used=config_dict if config_dict else None,
    )
    prov_path = out_dir / f"{sub_dir}{'_' + ses_part if ses_part else ''}_desc-{modality}_provenance.json"
    prov_json_path = str(write_provenance_json(
        output_path=prov_path,
        provenance=provenance,
    ))

    return qc_json_path, prov_json_path


def get_bids_derivative_provenance(
    pipeline_name: str = "pmriprep",
    pipeline_version: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a minimal provenance dict following BIDS-Derivatives GeneratedBy spec.

    This creates the standard "GeneratedBy" field used in dataset_description.json
    and individual derivative sidecars.

    Args:
        pipeline_name: Name of the pipeline (default: "pmriprep").
        pipeline_version: Version string. If None, uses pmriprep.__version__.
        description: Optional description of what the pipeline does.

    Returns:
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

    Args:
        sidecar: Existing sidecar dict (must contain PerfusionProcessing
            from Phase 1 static provenance).
        qc_metrics: QC metrics dict from the writer's qc_metrics input.
        modality: "DCE" or "DSC".

    Returns:
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


def _enrich_dce(
    perf: Dict[str, Any],
    qc: Dict[str, Any],
) -> None:
    """Enrich DCE PerfusionProcessing in-place from QC metrics."""
    # T1Mapping
    t1 = qc.get("t1_mapping")
    if t1 and isinstance(t1, dict):
        t1_block: Dict[str, Any] = {}
        if "method" in t1:
            t1_block["method"] = str(t1["method"])
        if "b1_correction" in t1:
            t1_block["B1_correction"] = str(t1["b1_correction"])
        if t1.get("method") == "assumed":
            t1_block["tissue_segmented"] = bool(t1.get("tissue_t1_used", False))
            if "mean_t1_wm_ms" in t1:
                t1_block["T1_WM_ms"] = _safe_float(t1["mean_t1_wm_ms"])
            if "mean_t1_gm_ms" in t1:
                t1_block["T1_GM_ms"] = _safe_float(t1["mean_t1_gm_ms"])
        if t1_block:
            perf["T1Mapping"] = t1_block

    # AIF detection
    aif = qc.get("aif_detection")
    if aif and isinstance(aif, dict):
        aif_block: Dict[str, Any] = {}
        if "aif_source" in aif:
            aif_block["method"] = str(aif["aif_source"])
        if "n_aif_voxels" in aif:
            aif_block["n_voxels"] = int(aif["n_aif_voxels"])
        peak = aif.get("peak_mm_mean")
        if peak is None:
            peak = aif.get("final_peak_mm")
        if peak is not None:
            aif_block["peak_mM"] = _safe_float(peak)
        fwhm = aif.get("fwhm_seconds_mean")
        if fwhm is None:
            fwhm = aif.get("final_fwhm_seconds")
        if fwhm is not None:
            aif_block["fwhm_s"] = _safe_float(fwhm)
        if "quality_score" in aif:
            aif_block["quality_score"] = _safe_float(aif["quality_score"])
        if "arterial_cascade_tier" in aif:
            aif_block["arterial_cascade_tier"] = int(aif["arterial_cascade_tier"])
        if "hemisphere_coverage" in aif:
            aif_block["hemisphere_coverage"] = str(aif["hemisphere_coverage"])
        if aif.get("population_aif"):
            aif_block["population_model"] = "parker"
            if "dose_factor" in aif:
                aif_block["dose_factor"] = _safe_float(aif["dose_factor"])
        if aif_block:
            perf["AIF"] = aif_block

        # v0.25.3+: explicit unit traceability. The AIF curve consumed by
        # PK modelling is in true plasma [Gd] (mM) after /(1-Hct); Parker
        # reference peaks are blood-voxel units pre-Hct. Emit the block
        # only when at least one concrete measurement (peak or Hct) is
        # available, to avoid description-only blocks.
        final_peak_plasma = aif.get("final_peak_mm")
        hct_used = aif.get("hematocrit")
        if final_peak_plasma is not None or hct_used is not None:
            units_block: Dict[str, Any] = {
                "description": (
                    "AIF is in true plasma [Gd] (mM) after /(1-Hct) "
                    "conversion; Parker reference peaks are blood-voxel "
                    "(pre-Hct)."
                ),
            }
            if final_peak_plasma is not None:
                units_block["aif_peak_plasma_mM"] = _safe_float(
                    final_peak_plasma
                )
            if hct_used is not None:
                units_block["hematocrit_used"] = _safe_float(hct_used)
            perf["AIFUnits"] = units_block

        # BloodT1 from AIF detection
        if "blood_t1_seconds" in aif and aif["blood_t1_seconds"]:
            fs = perf.get("Relaxivity", {}).get("field_strength_T")
            if fs is None:
                fs = _safe_float(aif.get("field_strength_t"))
            perf["BloodT1"] = {
                "value_s": _safe_float(aif["blood_t1_seconds"]),
                "Source": "literature_lookup",
            }
            if fs is not None:
                perf["BloodT1"]["field_strength_T"] = fs

    # Hematocrit
    from pmriprep.utils.dce_constants import HEMATOCRIT_ARTERIAL
    perf["Hematocrit"] = {
        "value": float(HEMATOCRIT_ARTERIAL),
        "Source": "default_population_average",
    }

    # PK Modeling (consolidates former top-level PharmacokineticModel,
    # BayesianInference, ModelSelection)
    pk = qc.get("pk_modeling")
    if pk and isinstance(pk, dict):
        pk_block: Dict[str, Any] = {}
        if "model" in pk:
            pk_block["model"] = str(pk["model"])
        if "bayesian" in pk:
            pk_block["bayesian_inference"] = str(pk["bayesian"])
        if pk.get("model") == "auto":
            selection: Dict[str, Any] = {"Method": "per-voxel BIC comparison"}
            candidates = pk.get("candidate_models", [])
            if candidates:
                selection["CandidateModels"] = candidates
                dist = {}
                for mn in candidates:
                    frac = pk.get(f"{mn}_fraction")
                    if frac is not None:
                        dist[mn] = round(float(frac), 4)
                if dist:
                    selection["ModelDistribution"] = dist
            pk_block["model_selection"] = selection
        if "convergence_rate" in pk:
            pk_block["convergence_rate"] = _safe_float(pk["convergence_rate"])
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
        deconv_block = perf.get("Deconvolution", {})
        if "threshold_mean" in deconv:
            deconv_block["threshold_mean"] = _safe_float(deconv["threshold_mean"])
        if "oi_mean" in deconv:
            deconv_block["oscillation_index_mean"] = _safe_float(deconv["oi_mean"])
        if "lambda_mean" in deconv:
            deconv_block["lambda_mean"] = _safe_float(deconv["lambda_mean"])
        if "convergence_rate" in deconv:
            deconv_block["convergence_rate"] = _safe_float(deconv["convergence_rate"])
        perf["Deconvolution"] = deconv_block

    # Leakage correction details
    preload = qc.get("preload")
    if preload and isinstance(preload, dict):
        leak_block = perf.get("LeakageCorrection", {})
        if "preload_correction_applied" in preload:
            leak_block["applied"] = bool(preload["preload_correction_applied"])
        if "k2_median" in preload:
            leak_block["K2_median"] = _safe_float(preload["k2_median"])
        perf["LeakageCorrection"] = leak_block

        # PreloadDetection enrichment
        preload_det = perf.get("PreloadDetection", {})
        if "preload_classification" in preload:
            preload_det["classification"] = str(preload["preload_classification"])
        if "preload_classification_source" in preload:
            preload_det["source"] = str(preload["preload_classification_source"])
        if "preload_k2_classification" in preload:
            preload_det["k2_classification"] = str(
                preload["preload_k2_classification"]
            )
        # v0.25.2: key renamed from baseline_delta_r2 → preload_baseline_delta_r2_advisory
        # (tissue-heterogeneity-biased advisory metric, not used for classification).
        if "preload_baseline_delta_r2_advisory" in preload:
            preload_det["baseline_delta_r2_advisory"] = _safe_float(
                preload["preload_baseline_delta_r2_advisory"]
            )
        perf["PreloadDetection"] = preload_det

    # AIF and baseline frames (both from qc["aif"])
    aif = qc.get("aif")
    if aif and isinstance(aif, dict):
        aif_block: Dict[str, Any] = {}
        if "detection_level" in aif:
            aif_block["method"] = str(aif["detection_level"])
            aif_block["synthetic"] = aif["detection_level"] == "synthetic"
        if "n_aif_voxels_used" in aif:
            aif_block["n_voxels"] = int(aif["n_aif_voxels_used"])
        if "quality_score" in aif:
            aif_block["quality_score"] = _safe_float(aif["quality_score"])
        if "best_fwhm" in aif:
            aif_block["fwhm_frames"] = _safe_float(aif["best_fwhm"])
        if "arrival_time_seconds" in aif:
            aif_block["arrival_time_s"] = _safe_float(aif["arrival_time_seconds"])
        if aif_block:
            perf["AIF"] = aif_block

        # Baseline frames (also from aif sub-dict)
        baseline_block: Dict[str, Any] = {}
        if "baseline_frames_used" in aif:
            baseline_block["n_frames"] = int(aif["baseline_frames_used"])
        if "baseline_detection_method" in aif:
            baseline_block["detection_method"] = str(aif["baseline_detection_method"])
        if baseline_block:
            perf["BaselineFrames"] = baseline_block

    # Quantification mode (always relative for DSC)
    perf["QuantificationMode"] = "relative"

    # Normalization reference (conditional on actual normalization)
    norm_factor = qc.get("normalization_factor")
    if norm_factor is not None:
        perf["NormalizationReference"] = {
            "tissue": "NAWM",
            "cbv_factor": _safe_float(norm_factor),
        }
