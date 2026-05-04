"""Pure-function sidecar builders for BIDS-Derivatives outputs.

These functions build the per-output ``meta_dict`` that is passed to
:class:`niworkflows.interfaces.bids.DerivativesDataSink`. They are
stateless — no ``self.inputs`` reads — so the per-pipeline factory
workflows can call them from inside ``pe.Node(Function(...))`` nodes.

Sidecar field strategy (M3):

- **Sources**: per-output list of BIDS URIs (``bids::sub-X/ses-Y/...``) to
  the immediate raw inputs that produced the derivative.
- **SpatialReference**: BIDS URI to the on-disk reference image whose
  grid the derivative occupies (e.g., the preproc T1w for ``space-T1w``,
  the mean DCE/DSC ref for native acquisition space). For native
  acquisition space in perf/, per BIDS-Common-Derivatives convention,
  outputs omit the ``space-`` entity. For anat/ outputs, ``space-T1w``
  is used uniformly per fmriprep precedent; pre- vs post-contrast
  disambiguation lives in this URI.
- **Resolution**: literal string ``"native"`` (BIDS-Derivatives 1.9
  keyword for unresampled images). The numeric voxel sizes go in
  ``VoxelSize: [vx, vy, vz]`` (custom pmriprep field).
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Paths and URIs
# ---------------------------------------------------------------------------


def make_bids_source_uri(source_path: PathLike, bids_dir: PathLike) -> str:
    """Return a ``bids::`` URI for ``source_path`` relative to ``bids_dir``."""
    source_path = Path(source_path).resolve()
    bids_dir = Path(bids_dir).resolve()
    try:
        rel = source_path.relative_to(bids_dir)
        return f"bids::{rel}"
    except ValueError:
        return f"bids::{source_path.name}"


def build_sources_list(
    source_paths: Sequence[PathLike],
    bids_dir: PathLike,
) -> List[str]:
    """Return ``[bids::...]`` URIs for each input path (skip empty/None)."""
    out: List[str] = []
    for p in source_paths:
        if p is None or p == "":
            continue
        out.append(make_bids_source_uri(p, bids_dir))
    return out


def build_spatial_reference_uri(
    reference_filename: str,
    subject: str,
    session: Optional[str],
    datatype: str = "anat",
    pipeline: str = "pmriprep",
) -> str:
    """Return a derivative-relative ``bids::`` URI to a reference image.

    ``reference_filename`` is the basename (without sub-/ses- prefix) of the
    reference image inside the derivatives tree, e.g.
    ``"space-T1w_desc-preproc_T1w.nii.gz"``.
    """
    prefix = f"sub-{subject}"
    if session:
        prefix += f"_ses-{session}"
    full_name = f"{prefix}_{reference_filename}"
    parts = [pipeline, f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.extend([datatype, full_name])
    return "bids::" + "/".join(parts)


# ---------------------------------------------------------------------------
# Resolution / voxel-size
# ---------------------------------------------------------------------------


def voxel_sizes_from_nifti(source_nifti: PathLike) -> List[float]:
    """Return the first three voxel sizes (mm) from a NIfTI header."""
    import nibabel as nib  # type: ignore[import-untyped]

    img = nib.load(str(source_nifti))
    zooms = img.header.get_zooms()[:3]  # type: ignore[attr-defined]
    return [float(z) for z in zooms]


def compute_original_resolution(source_nifti: PathLike) -> Dict[str, Any]:
    """Return the legacy pmriprep ``OriginalResolution`` block."""
    voxel_dims = voxel_sizes_from_nifti(source_nifti)
    geo_mean = round(float(math.prod(voxel_dims) ** (1.0 / 3.0)), 3)
    iso_ratio = round(max(voxel_dims) / max(min(voxel_dims), 1e-9), 2)
    if geo_mean <= 1.5:
        tier = "excellent"
    elif geo_mean <= 3.0:
        tier = "good"
    else:
        tier = "moderate"
    return {
        "VoxelDimensions": voxel_dims,
        "GeometricMean": geo_mean,
        "IsotropyRatio": iso_ratio,
        "ResolutionTier": tier,
    }


def resolution_native_block(source_nifti: PathLike) -> Dict[str, Any]:
    """BIDS-Derivatives 1.9 ``Resolution: native`` + custom ``VoxelSize``."""
    return {
        "Resolution": "native",
        "VoxelSize": voxel_sizes_from_nifti(source_nifti),
    }


# ---------------------------------------------------------------------------
# Software versions
# ---------------------------------------------------------------------------


def get_fsl_version() -> str:
    """Return FSL version string, or ``"unknown"`` if not detected."""
    try:
        fsldir = os.environ.get("FSLDIR", "")
        version_file = Path(fsldir) / "etc" / "fslversion"
        if version_file.exists():
            return f"fsl-{version_file.read_text().strip()}"
        result = subprocess.run(
            ["flirt", "-version"], capture_output=True, text=True, timeout=5
        )
        for line in (result.stdout + result.stderr).splitlines():
            if line.strip():
                return line.strip()
    except Exception:
        pass
    return "unknown"


def get_ants_version() -> str:
    """Return ANTs version string, or ``"unknown"`` if not detected."""
    try:
        result = subprocess.run(
            ["N4BiasFieldCorrection", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if "ANTs" in line:
                return line.strip()
    except Exception:
        pass
    return "unknown"


def get_imaging_software_versions() -> Dict[str, str]:
    """Return version dict for FSL, ANTs, and HD-BET."""
    return {
        "FSL": get_fsl_version(),
        "ANTs": get_ants_version(),
        "HD-BET": "unknown",
    }


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def get_xfm_filename(
    subject: str,
    session: Optional[str],
    from_space: str,
    to_space: str,
) -> str:
    """BIDS xfm filename ``sub-X[_ses-Y]_from-A_to-B_mode-image_xfm.mat``."""
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.extend([f"from-{from_space}", f"to-{to_space}", "mode-image"])
    return "_".join(parts) + "_xfm.mat"


def extract_contrast_agent(source_nifti_path: str) -> Optional[str]:
    """Read ``ContrastBolusAgent`` from a BIDS sidecar; return generic name."""
    import json

    src = Path(source_nifti_path)
    sidecar_path = src.with_name(
        src.name.replace(".nii.gz", ".json").replace(".nii", ".json")
    )
    if not sidecar_path.exists():
        return None
    try:
        with open(sidecar_path) as f:
            meta = json.load(f)
        raw_agent = meta.get("ContrastBolusAgent", "")
        if not raw_agent or not str(raw_agent).strip():
            return None
        from pmriprep.utils.contrast_agent import parse_contrast_agent_string
        parsed = parse_contrast_agent_string(raw_agent)
        return parsed.get("agent_name") or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Anat sidecar builders
# ---------------------------------------------------------------------------


_ANAT_PREPROC_STEPS = [
    "Reorientation to RAS",
    "Robust FOV cropping",
    "N4 bias field correction",
    "Coregistration to native anat space",
    "HD-BET brain extraction",
]


def anat_reference_basename(is_postcon_ref: bool = False) -> str:
    """Basename of the anat reference image inside the derivatives tree.

    In postcon-only mode the postcon T1w IS the reference, so the file is
    ``ce-GAD_space-T1w_desc-preproc_T1w.nii.gz``. Otherwise the precon
    T1w is the reference. Note: the BIDS ``T1w`` space label is canonical
    per fmriprep/aslprep precedent; pre- vs post-contrast disambiguation
    lives in the JSON ``SpatialReference`` field, which points to the
    actual reference image (with or without the ``ce-GAD`` entity).
    """
    if is_postcon_ref:
        return "ce-GAD_space-T1w_desc-preproc_T1w.nii.gz"
    return "space-T1w_desc-preproc_T1w.nii.gz"


def _anat_spatial_reference(
    subject: str, session: Optional[str], is_postcon_ref: bool = False
) -> str:
    return build_spatial_reference_uri(
        anat_reference_basename(is_postcon_ref),
        subject, session, datatype="anat",
    )


def build_anat_preproc_sidecar(
    *,
    description: str,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    skull_stripped: bool = True,
    contrast_agent: Optional[str] = None,
    is_postcon_ref: bool = False,
) -> Dict[str, Any]:
    """Build sidecar for a preprocessed/coregistered anat NIfTI."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sources = build_sources_list(source_paths, bids_dir)
    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": description,
        "SkullStripped": skull_stripped,
        "SpatialReference": _anat_spatial_reference(subject, session, is_postcon_ref),
        "Sources": sources,
        **resolution_native_block(reference_voxel_source),
        "OriginalResolution": compute_original_resolution(reference_voxel_source),
        "ProcessingSteps": list(_ANAT_PREPROC_STEPS),
        "SoftwareVersions": get_imaging_software_versions(),
    }
    if contrast_agent is not None:
        from pmriprep.utils.contrast_agent import get_brand_name
        agent_block: Dict[str, str] = {"GenericName": contrast_agent}
        brand = get_brand_name(contrast_agent)
        if brand:
            agent_block["BrandName"] = brand
        sidecar["PerfusionProcessing"] = {"ContrastAgent": agent_block}
    return sidecar


def build_anat_mask_sidecar(
    *,
    description: str,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    is_postcon_ref: bool = False,
) -> Dict[str, Any]:
    """Build sidecar for an anat-space brain mask."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    return {
        **get_bids_derivative_provenance(),
        "Description": description,
        "SkullStripped": True,
        "SpatialReference": _anat_spatial_reference(subject, session, is_postcon_ref),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
        "OriginalResolution": compute_original_resolution(reference_voxel_source),
        "ProcessingSteps": list(_ANAT_PREPROC_STEPS),
        "SoftwareVersions": get_imaging_software_versions(),
    }


_TUMOR_SEG_LABELS = {
    "0": "Background",
    "1": "Non-enhancing tumor",
    "2": "Enhancing tumor",
}


def build_tumor_seg_dseg_sidecar(
    *,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    is_postcon_ref: bool = False,
) -> Dict[str, Any]:
    """Sidecar for the multi-class tumor ``dseg`` label map.

    Per BIDS-Common-Derivatives, a single ``_dseg.json`` describes both
    the ``_dseg.nii.gz`` label NIfTI and the ``_dseg.tsv`` lookup table
    (when the latter is present) since they share a basename. The
    ``Columns`` block here documents the TSV columns
    (``index, name, voxel_count, volume_mm3, volume_ml``) for any
    consumer reading the companion ``_dseg.tsv``.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    return {
        **get_bids_derivative_provenance(),
        "Description": "Deep learning tumor segmentation",
        "Manual": False,
        "SegmentationLabels": dict(_TUMOR_SEG_LABELS),
        "Method": "HD-GLIO (Isensee et al.)",
        "SpatialReference": _anat_spatial_reference(subject, session, is_postcon_ref),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
        "Columns": {
            "index": {
                "LongName": "Region label index",
                "Description": (
                    "Integer label matching the corresponding ``_dseg.nii.gz`` "
                    "voxel value. Background (0) is omitted from the TSV per "
                    "BIDS dseg convention."
                ),
            },
            "name": {
                "LongName": "Region name",
                "Description": "Tumor region identifier (snake_case).",
            },
            "voxel_count": {
                "LongName": "Voxel count",
                "Description": "Number of voxels with this label.",
                "Units": "voxels",
            },
            "volume_mm3": {
                "LongName": "Region volume",
                "Description": "Region volume in cubic millimetres.",
                "Units": "mm^3",
            },
            "volume_ml": {
                "LongName": "Region volume",
                "Description": "Region volume in millilitres.",
                "Units": "mL",
            },
        },
    }


def build_tumor_seg_mask_sidecar(
    *,
    description: str,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    method: str = "Derived from tumor segmentation",
    is_postcon_ref: bool = False,
) -> Dict[str, Any]:
    """Sidecar for a binary tumor-derived mask (enhancing/nonenhancing/whole/lesion)."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    return {
        **get_bids_derivative_provenance(),
        "Description": description,
        "Type": "Brain",
        "Method": method,
        "SpatialReference": _anat_spatial_reference(subject, session, is_postcon_ref),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
    }


def build_tumor_seg_nawm_sidecar(
    *,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    is_postcon_ref: bool = False,
) -> Dict[str, Any]:
    """Sidecar for the contralateral NAWM mask."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    return {
        **get_bids_derivative_provenance(),
        "Description": (
            "Contralateral normal-appearing white matter mask for CBV "
            "normalization"
        ),
        "Type": "Brain",
        "Method": (
            "Contralateral WM thresholded at p>=0.9, excluding tumor margin"
        ),
        "SpatialReference": _anat_spatial_reference(subject, session, is_postcon_ref),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
    }


def dce_reference_basename() -> str:
    """Basename of the DCE reference image inside the derivatives tree."""
    return "desc-dceMean_ref.nii.gz"


def _dce_spatial_reference(subject: str, session: Optional[str]) -> str:
    return build_spatial_reference_uri(
        dce_reference_basename(), subject, session, datatype="perf",
    )


def build_dce_base_sidecar(
    *,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    provenance_dict: Optional[Dict[str, Any]] = None,
    qc_metrics: Optional[Dict[str, Any]] = None,
    minimal: bool = False,
) -> Dict[str, Any]:
    """Common base sidecar for DCE derivatives.

    ``minimal=True`` suppresses perfusion-specific provenance enrichment
    and the ``KtransConvention`` block (used for masks, structural
    underlays, motion-corrected 4D, T1map, B1+map, etc.).
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "SpatialReference": _dce_spatial_reference(subject, session),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
    }
    if provenance_dict:
        sidecar.update(provenance_dict)
    if not minimal:
        if provenance_dict and qc_metrics:
            from pmriprep.utils.provenance import enrich_provenance_from_qc

            enriched = enrich_provenance_from_qc(sidecar, qc_metrics, "DCE")
            sidecar.update(enriched)
        sidecar["KtransConvention"] = "plasma_Tofts1999"
        if isinstance(qc_metrics, dict):
            hct = qc_metrics.get("aif_detection", {}).get("hematocrit")
            if hct is not None:
                sidecar["KtransConventionHct"] = float(hct)
    return sidecar


def dsc_reference_basename() -> str:
    """Basename of the DSC reference image inside the derivatives tree."""
    return "desc-dscMean_ref.nii.gz"


def _dsc_spatial_reference(subject: str, session: Optional[str]) -> str:
    return build_spatial_reference_uri(
        dsc_reference_basename(), subject, session, datatype="perf",
    )


def build_dsc_base_sidecar(
    *,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    provenance_dict: Optional[Dict[str, Any]] = None,
    qc_metrics: Optional[Dict[str, Any]] = None,
    minimal: bool = False,
) -> Dict[str, Any]:
    """Common base sidecar for DSC derivatives.

    ``minimal=True`` suppresses perfusion-specific provenance enrichment
    (used for masks, structural underlays, motion-corrected 4D, etc.).
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "SpatialReference": _dsc_spatial_reference(subject, session),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
    }
    if provenance_dict:
        sidecar.update(provenance_dict)
    if not minimal and provenance_dict and qc_metrics:
        from pmriprep.utils.provenance import enrich_provenance_from_qc

        enriched = enrich_provenance_from_qc(sidecar, qc_metrics, "DSC")
        sidecar.update(enriched)
    return sidecar


def build_dsc_map_sidecar(
    *,
    base: Dict[str, Any],
    description: str,
    map_key: str,
    qc_metrics: Optional[Dict[str, Any]] = None,
    units: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-output DSC sidecar with description + units + AIF stamp + extras."""
    from pmriprep.utils.dsc_aif_provenance import apply_aif_sidecar

    sidecar: Dict[str, Any] = {**base, "Description": description}
    if units is not None:
        sidecar["Units"] = units
    if extra:
        sidecar.update(extra)
    apply_aif_sidecar(sidecar, map_key, qc_metrics)
    return sidecar


def build_anat_probseg_sidecar(
    *,
    tissue_type: str,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    is_postcon_ref: bool = False,
) -> Dict[str, Any]:
    """Build sidecar for a tissue probability map (``label-{WM,GM,CSF}``)."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    fsl_version = get_fsl_version()
    return {
        **get_bids_derivative_provenance(),
        "Description": f"{tissue_type} probability map from FAST tissue segmentation",
        "SpatialReference": _anat_spatial_reference(subject, session, is_postcon_ref),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
        "ReferenceContrastType": "precon",
        "SoftwareVersions": {"FSL": fsl_version, "FAST": "part of FSL"},
        "ProcessingSteps": [
            "FAST 3-class tissue segmentation (T1-weighted)",
            "Parameters: -t 1 -n 3 -H 0.1 -I 4 -l 20.0",
        ],
        "TissueType": tissue_type,
        "SegmentationMethod": "FAST",
        "NumberOfClasses": 3,
        "ProbabilityMap": True,
    }


# ---------------------------------------------------------------------------
# Per-image anat sidecar enrichment (Option A QC distribution)
#
# These helpers take a base sidecar dict + the upstream `qc_metrics` dict
# and inject the relevant subset of pipeline state matching what was
# previously aggregated into the standalone anat QC summary JSON.
#
# Pattern: each helper mutates the sidecar in-place AND returns it for
# chaining. Missing qc fields are silently skipped — sidecars that lack
# upstream qc data still emit a valid (smaller) sidecar.
# ---------------------------------------------------------------------------


def enrich_brain_mask_sidecar(
    sidecar: Dict[str, Any],
    qc_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Add a ``BrainExtraction`` block to a brain-mask sidecar.

    Reads ``qc_metrics["brain_extraction"]`` (tool, version, parameters).
    Used during 0.45.0 distribution of anat QC content.
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    be = qc_metrics.get("brain_extraction")
    if not be or not isinstance(be, dict):
        return sidecar
    block: Dict[str, Any] = {}
    if "tool" in be:
        block["Tool"] = str(be["tool"])
    if "version" in be:
        block["Version"] = str(be["version"])
    if "device" in be:
        block["Device"] = str(be["device"])
    if "voxel_count" in be:
        block["BrainVoxelCount"] = int(be["voxel_count"])
    if "volume_ml" in be:
        block["BrainVolume_mL"] = float(be["volume_ml"])
    if block:
        sidecar["BrainExtraction"] = block
    return sidecar


def enrich_preproc_anat_sidecar(
    sidecar: Dict[str, Any],
    qc_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Add a ``Preprocessing`` block to an anat preproc sidecar.

    Reads ``qc_metrics["preprocessing"]`` (denoising, bias correction,
    ACPC alignment) and ``qc_metrics["coregistration"]`` (per-modality
    coreg metrics if present in the calling sidecar's modality scope).
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    prep = qc_metrics.get("preprocessing")
    if prep and isinstance(prep, dict):
        block: Dict[str, Any] = {}
        if "denoising" in prep:
            block["Denoising"] = bool(prep["denoising"])
        if "bias_correction" in prep:
            block["BiasCorrection"] = str(prep["bias_correction"])
        if "acpc_aligned" in prep:
            block["ACPCAligned"] = bool(prep["acpc_aligned"])
        if block:
            sidecar["Preprocessing"] = block
    coreg = qc_metrics.get("coregistration")
    if coreg and isinstance(coreg, dict):
        # The caller knows which modality this sidecar describes; coreg
        # metrics are passed pre-filtered. Accept whatever subset is given.
        sidecar["Coregistration"] = dict(coreg)
    return sidecar


def enrich_probseg_sidecar(
    sidecar: Dict[str, Any],
    qc_metrics: Optional[Dict[str, Any]],
    tissue_type: str,
) -> Dict[str, Any]:
    """Add a ``TissueSegmentationStats`` block to a probseg sidecar.

    Reads ``qc_metrics["tissue_segmentation"][tissue_type]`` (mean
    probability, voxel count) for the target tissue (WM/GM/CSF).
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    seg = qc_metrics.get("tissue_segmentation")
    if not seg or not isinstance(seg, dict):
        return sidecar
    block_for_tissue = seg.get(tissue_type) if isinstance(seg.get(tissue_type), dict) else None
    if not block_for_tissue:
        return sidecar
    sidecar["TissueSegmentationStats"] = dict(block_for_tissue)
    return sidecar


# ---------------------------------------------------------------------------
# AIF timeseries sidecars
# ---------------------------------------------------------------------------


def _aif_cosmetic_keys(aif_qc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flat AIF*-prefixed top-level keys for AIF curve sidecar enrichment.

    Reads from the producer-shape ``aif_qc`` dict (the same dict that
    flows through the DSC outputnode). Conditional fields are omitted
    when not applicable (per fmriprep convention for optional sidecar
    fields).
    """
    if not aif_qc or not isinstance(aif_qc, dict):
        return {}

    out: Dict[str, Any] = {}
    raw_metrics = aif_qc.get("aif_metrics")
    metrics: Dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}

    if "detection_method" in aif_qc:
        out["AIFMethod"] = str(aif_qc["detection_method"])
    if "n_aif_voxels_selected" in aif_qc:
        out["AIFNumberOfVoxelsSelected"] = int(aif_qc["n_aif_voxels_selected"])
    if "peak_conc" in metrics:
        out["AIFPeakAmplitude"] = float(metrics["peak_conc"])
    if "fwhm" in metrics:
        out["AIFFWHM"] = float(metrics["fwhm"])
    if "ttp" in metrics:
        out["AIFTimeToPeak"] = float(metrics["ttp"])
    if "arrival" in metrics:
        out["AIFArrival"] = float(metrics["arrival"])
    if "best_brr" in aif_qc:
        out["AIFBaselineReturnRatio"] = float(aif_qc["best_brr"]) / 100.0
    if "best_psr" in aif_qc:
        out["AIFPeakSignalRatio"] = float(aif_qc["best_psr"]) / 100.0
    if "quality_score" in aif_qc:
        out["AIFQualityScore"] = float(aif_qc["quality_score"])
    if "cascade_final_level" in aif_qc:
        out["AIFArterialCascadeTier"] = str(aif_qc["cascade_final_level"])
    if "hemisphere_coverage" in aif_qc:
        out["AIFHemisphereCoverage"] = str(aif_qc["hemisphere_coverage"])
    if "gamma_fit_applied" in aif_qc:
        out["AIFGammaFitApplied"] = bool(aif_qc["gamma_fit_applied"])
    if "gamma_fit_r_squared" in aif_qc:
        out["AIFGammaFitRSquared"] = float(aif_qc["gamma_fit_r_squared"])
    if "gamma_fit_auc_ratio" in aif_qc:
        out["AIFGammaFitAUCRatio"] = float(aif_qc["gamma_fit_auc_ratio"])
    if "synthetic" in aif_qc:
        out["AIFSynthetic"] = bool(aif_qc["synthetic"])
    if "cbv_auc_denominator_is_synthetic" in aif_qc:
        out["AIFUsedAsCBVDenominator"] = bool(aif_qc["cbv_auc_denominator_is_synthetic"])
    if "hematocrit" in aif_qc:
        out["AIFHematocrit"] = float(aif_qc["hematocrit"])
    return out


def build_aif_timeseries_sidecar(
    *,
    modality: str,
    source_paths: Sequence[PathLike],
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    aif_qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON sidecar for the AIF curve text file.

    Per BIDS-Common-Derivatives, every data file should carry its own
    sidecar describing how it was derived. This builder emits AIF
    detection metadata as flat ``AIF*``-prefixed top-level keys, matching
    fmriprep/aslprep convention for pipeline-derived sidecar fields.

    Note: the ``.txt`` extension is non-canonical (BIDS uses
    ``_timeseries.tsv``); this is a separate, pre-existing issue. The
    sidecar attaches metadata regardless and the validator treats it as
    a free sidecar.

    Args:
        modality: ``"DCE"`` or ``"DSC"``.
        source_paths: Source 4D NIfTI paths the AIF was extracted from.
        bids_dir: BIDS rawdata root for ``Sources`` BIDS-URI construction.
        subject: Subject ID without ``sub-`` prefix.
        session: Session ID without ``ses-`` prefix (or None).
        aif_qc: Producer-shape AIF QC dict (the ``aif_qc`` dict from the
            DSC outputnode for DSC, or the equivalent for DCE).

    Returns:
        Dict suitable for ``json.dumps``.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    if modality not in {"DCE", "DSC"}:
        raise ValueError(f"modality must be 'DCE' or 'DSC'; got {modality!r}")

    # AIF curve content differs by modality:
    #   DSC: ΔR2* (1/s) — susceptibility-induced relaxation-rate change
    #   DCE: plasma [Gd] in mM (after /(1-Hct) conversion)
    if modality == "DSC":
        column_name = "delta_r2_star"
        units = "1/s"
        column_description = (
            "Susceptibility-induced change in transverse relaxation rate "
            "(ΔR2*) at the AIF voxel(s)."
        )
    else:  # DCE
        column_name = "plasma_concentration_mM"
        units = "mM"
        column_description = (
            "Plasma gadolinium concentration after blood-to-plasma "
            "conversion (/(1-Hct))."
        )

    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": f"Arterial input function (AIF) curve for {modality}-MRI.",
        "Sources": build_sources_list(source_paths, bids_dir),
        "Columns": [column_name],
        "Units": units,
        "ColumnDescriptions": {column_name: column_description},
    }
    sidecar["Subject"] = f"sub-{subject}"
    if session:
        sidecar["Session"] = f"ses-{session}"

    sidecar.update(_aif_cosmetic_keys(aif_qc))
    return sidecar


# ---------------------------------------------------------------------------
# DSC preprocessed-4D sidecar enrichment (Phase 2a)
# ---------------------------------------------------------------------------


def _preload_cosmetic_keys(preload_qc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flat ``Preload*``-prefixed keys for the DSC preproc-4D sidecar.

    Reads from the producer-shape ``preload_qc`` dict. Optional fields
    (e.g., ``PreloadKepUsed`` only meaningful for the bidirectional K2
    method) are omitted when not present in the source dict.
    """
    if not preload_qc or not isinstance(preload_qc, dict):
        return {}

    out: Dict[str, Any] = {}
    if "k2_method" in preload_qc:
        out["PreloadK2Method"] = str(preload_qc["k2_method"])
    if preload_qc.get("kep_used") is not None:
        out["PreloadKepUsed"] = float(preload_qc["kep_used"])
    if "preload_classification" in preload_qc:
        out["PreloadClassification"] = str(preload_qc["preload_classification"])
    if "preload_classification_source" in preload_qc:
        out["PreloadClassificationSource"] = str(preload_qc["preload_classification_source"])
    if "preload_k2_classification" in preload_qc:
        out["PreloadK2Classification"] = str(preload_qc["preload_k2_classification"])
    if "preload_k2_confidence" in preload_qc:
        out["PreloadK2Confidence"] = str(preload_qc["preload_k2_confidence"])
    if "k2_median" in preload_qc:
        out["PreloadK2Median"] = float(preload_qc["k2_median"])
    if "percent_leaky_voxels" in preload_qc:
        out["PreloadPercentLeakyVoxels"] = float(preload_qc["percent_leaky_voxels"])
    if "percent_severe_leakage" in preload_qc:
        out["PreloadPercentSevereLeakage"] = float(preload_qc["percent_severe_leakage"])
    if "preload_user_specified" in preload_qc:
        out["PreloadUserSpecified"] = bool(preload_qc["preload_user_specified"])
    return out


def _preprocessing_cosmetic_keys(
    preprocessing_stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Flat motion/signal QC keys for the DSC preproc-4D sidecar.

    No top-level prefix — the sidecar's ``desc-`` entity already conveys
    "preprocessing." Future ``desc-dcePreproc_dce.json`` will reuse the
    same key names. ``RepetitionTime`` is the canonical BIDS key (in
    seconds, converted from the producer's ``tr_ms``).
    """
    if not preprocessing_stats or not isinstance(preprocessing_stats, dict):
        return {}

    p = preprocessing_stats
    out: Dict[str, Any] = {}
    if "fd_mean" in p:
        out["MotionFDMean"] = float(p["fd_mean"])
    if "fd_max" in p:
        out["MotionFDMax"] = float(p["fd_max"])
    if "percent_outliers" in p:
        out["MotionPercentOutliers"] = float(p["percent_outliers"])
    if "motion_status" in p:
        out["MotionStatus"] = str(p["motion_status"])
    if "mean_cnr" in p:
        out["CNRWholeBrain"] = float(p["mean_cnr"])
    if "fraction_cnr_gt_4" in p:
        out["FractionCNRGreaterThan4"] = float(p["fraction_cnr_gt_4"])
    if "mean_tsnr" in p:
        out["MeanTSNR"] = float(p["mean_tsnr"])
    if "peak_drop_pct" in p:
        out["PeakSignalDropPercent"] = float(p["peak_drop_pct"])
    if "n_volumes" in p:
        out["NumberOfVolumes"] = int(p["n_volumes"])
    if "tr_ms" in p:
        out["RepetitionTime"] = float(p["tr_ms"]) / 1000.0
    return out


def enrich_dsc_preproc_sidecar(
    sidecar: Dict[str, Any],
    *,
    preload_qc: Optional[Dict[str, Any]] = None,
    preprocessing_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add flat-prefixed cosmetic QC keys to a DSC preproc-4D sidecar.

    Mutates and returns ``sidecar``. Adds ``Preload*`` and unprefixed
    motion/signal keys from the producer-shape input dicts. Existing
    sidecar fields (``SkullStripped``, ``SpatialReference``, ``Sources``,
    ``Resolution``) are preserved.
    """
    sidecar.update(_preload_cosmetic_keys(preload_qc))
    sidecar.update(_preprocessing_cosmetic_keys(preprocessing_stats))
    return sidecar


# ---------------------------------------------------------------------------
# DSC resume sidecar (Phase 2b)
#
# Pipeline-internal sidecar that round-trips the verbatim producer-shape
# QC dicts so the ``--skip-dsc`` resume helper can recover them without
# JSON-loading the legacy ``desc-dscQC_quality.json``.
# ---------------------------------------------------------------------------


def build_dsc_resume_sidecar(
    *,
    aif_qc: Optional[Dict[str, Any]] = None,
    preload_qc: Optional[Dict[str, Any]] = None,
    preprocessing_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the verbatim resume sidecar for ``--skip-dsc`` round-trip.

    Stores the producer-shape ``aif_qc``, ``preload_qc``, and
    ``preprocessing_stats`` dicts verbatim under top-level snake_case
    keys. The resume helper at ``workflows/base.py:_find_previous_dsc_outputs``
    reads these and feeds them to the same downstream consumers
    (``dce_compute.py`` PK setup, ``dsc_cbv_constraint.py``) that the
    in-process flow uses, so no key translation happens at the resume
    boundary.

    The ``_pmriprep`` filename suffix and ``PMRIPrepInternal: true`` flag
    mark this as pipeline-internal scope (per NiPreps precedent for
    non-canonical-tree files).
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": "Pipeline-internal resume state for --skip-dsc.",
        "PMRIPrepInternal": True,
    }
    if aif_qc:
        sidecar["aif_qc"] = dict(aif_qc)
    if preload_qc:
        sidecar["preload_qc"] = dict(preload_qc)
    if preprocessing_stats:
        sidecar["preprocessing_stats"] = dict(preprocessing_stats)
    return sidecar
