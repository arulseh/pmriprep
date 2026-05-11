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

import copy
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pmriprep.config.constants import HDGLIO_BIDS_LABELS

PathLike = Union[str, Path]

_PIPELINE_QC_STRIP_KEYS = frozenset({
    "fd_list", "rms_list", "outlier_frames",
    "candidate_coords", "candidate_scores",
    "cascade_anchor_outcomes", "motion_summary",
    "ktrans_histogram", "ktrans_histogram_edges",
    "ve_histogram", "ve_histogram_edges",
    "gamma_fit_auto_metrics",
    "aif_centroid_mm", "aif_voxel_coord",
})

_CONFIG_STRIP_KEYS = frozenset({
    "n_procs", "nprocs", "hdbet_device", "tumor_seg_device",
    "disable_aif_fallback", "n_aif_candidates", "cpi_infer_cpt",
    "underlays", "no_t1", "no_tortuosity", "no_dsc_validation",
    "scale_aif",
})


def filter_pipeline_qc(qc: Dict[str, Any]) -> Dict[str, Any]:
    """Strip per-frame arrays and debugging internals from the QC envelope."""
    filtered = copy.deepcopy(qc)

    def _strip(d: dict) -> None:
        for key in list(d):
            if key in _PIPELINE_QC_STRIP_KEYS:
                del d[key]
            elif isinstance(d[key], dict):
                _strip(d[key])

    _strip(filtered)
    return filtered


def build_provenance_block(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``PipelineConfiguration`` block for a paired sidecar."""
    from pmriprep.utils.provenance import get_software_versions

    return {
        "ProcessingConfiguration": {
            k: v for k, v in config_dict.items()
            if k not in _CONFIG_STRIP_KEYS
        },
        "SoftwareVersions": get_software_versions(),
    }


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
    """BIDS xfm filename ``sub-X[_ses-Y]_from-A_to-B_mode-image_xfm.h5``."""
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.extend([f"from-{from_space}", f"to-{to_space}", "mode-image"])
    return "_".join(parts) + "_xfm.h5"


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


def _tumor_volumetrics(seg_path: PathLike) -> Dict[str, Any]:
    """Compute per-region tumor volumes from the dseg NIfTI."""
    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    from pmriprep.config.constants import HDGLIO_LABEL_TO_NAME
    from pmriprep.utils.lesion_stats import compute_region_volume

    img = nib.load(str(seg_path))
    seg_data = np.asarray(img.dataobj, dtype=np.int16)  # type: ignore[attr-defined]
    zooms = img.header.get_zooms()[:3]  # type: ignore[attr-defined]
    voxel_sizes = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    enh_count, enh_mm3, enh_ml = compute_region_volume(seg_data == 2, voxel_sizes)
    nenh_count, nenh_mm3, nenh_ml = compute_region_volume(seg_data == 1, voxel_sizes)
    wt_count, wt_mm3, wt_ml = compute_region_volume(seg_data > 0, voxel_sizes)

    enhancing_fraction: Optional[float] = None
    if wt_ml > 0:
        enhancing_fraction = round(enh_ml / wt_ml, 4)

    return {
        "TotalLesionVoxels": wt_count,
        "EnhancingFraction": enhancing_fraction,
        "Regions": {
            HDGLIO_LABEL_TO_NAME[2]: {
                "voxel_count": enh_count,
                "volume_mm3": round(enh_mm3, 2),
                "volume_ml": round(enh_ml, 3),
            },
            HDGLIO_LABEL_TO_NAME[1]: {
                "voxel_count": nenh_count,
                "volume_mm3": round(nenh_mm3, 2),
                "volume_ml": round(nenh_ml, 3),
            },
            "whole_tumor": {
                "voxel_count": wt_count,
                "volume_mm3": round(wt_mm3, 2),
                "volume_ml": round(wt_ml, 3),
            },
        },
    }


def build_tumor_seg_dseg_sidecar(
    *,
    source_paths: Sequence[PathLike],
    reference_voxel_source: PathLike,
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    is_postcon_ref: bool = False,
    qc_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sidecar for the multi-class tumor ``dseg`` label map.

    Per BIDS-Common-Derivatives, a single ``_dseg.json`` describes both
    the ``_dseg.nii.gz`` label NIfTI and the ``_dseg.tsv`` lookup table
    (when the latter is present) since they share a basename. The
    ``Columns`` block documents the TSV columns; ``TumorVolumetrics``
    carries per-region volumes and the enhancing fraction (replaces the
    removed standalone ``desc-tumorClinicalContext.json``).
    ``TumorSegQC`` carries preprocessing/segmentation/nawm QC blocks.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sc: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": "Deep learning tumor segmentation",
        "Manual": False,
        "SegmentationLabels": dict(HDGLIO_BIDS_LABELS),
        "Method": "HD-GLIO (Isensee et al.)",
        "SpatialReference": _anat_spatial_reference(subject, session, is_postcon_ref),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
        "TumorVolumetrics": _tumor_volumetrics(reference_voxel_source),
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
            "abbreviation": {
                "LongName": "Region abbreviation",
                "Description": (
                    "Short identifier for the region (e.g., ``ET`` for "
                    "contrast-enhancing tumor, ``NET`` for non-enhancing tumor)."
                ),
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
        },
    }
    if isinstance(qc_metrics, dict) and qc_metrics:
        sc["TumorSegQC"] = qc_metrics
    return sc


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
    return "desc-dceMean_stat.nii.gz"


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
    return "desc-dscMean_stat.nii.gz"


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
    """Add flat ``BrainExtraction*`` keys to a brain-mask sidecar.

    Reads ``qc_metrics["brain_extraction"]`` (tool, version, parameters).
    Volume in ``BrainExtractionVolume`` is in mL (documented in sidecar
    Description; no unit suffix in key per BIDS convention).
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    be = qc_metrics.get("brain_extraction")
    if not be or not isinstance(be, dict):
        return sidecar
    if "tool" in be:
        sidecar["BrainExtractionTool"] = str(be["tool"])
    if "version" in be:
        sidecar["BrainExtractionVersion"] = str(be["version"])
    if "device" in be:
        sidecar["BrainExtractionDevice"] = str(be["device"])
    if "voxel_count" in be:
        sidecar["BrainExtractionVoxelCount"] = int(be["voxel_count"])
    # Producer emits brain_volume_ml / brain_volume_mm3; legacy
    # volume_ml is also accepted for callers that pre-stripped the
    # prefix.
    vol_ml = be.get("brain_volume_ml", be.get("volume_ml"))
    if vol_ml is not None:
        sidecar["BrainExtractionVolume"] = float(vol_ml)
    vol_mm3 = be.get("brain_volume_mm3", be.get("volume_mm3"))
    if vol_mm3 is not None:
        sidecar["BrainExtractionVolumeMm3"] = float(vol_mm3)
    if "qc_status" in be:
        sidecar["BrainExtractionQCStatus"] = str(be["qc_status"])
    if "warnings" in be and be["warnings"]:
        sidecar["BrainExtractionWarnings"] = list(be["warnings"])
    return sidecar


def enrich_preproc_anat_sidecar(
    sidecar: Dict[str, Any],
    qc_metrics: Optional[Dict[str, Any]],
    coreg_modality: Optional[str] = None,
) -> Dict[str, Any]:
    """Add flat ``Preprocessing*`` and ``Coregistration<Modality>*`` keys.

    Reads ``qc_metrics["preprocessing"]`` (denoising, bias correction,
    ACPC alignment) and ``qc_metrics["coregistration"]`` (per-modality
    coreg metrics if present in the calling sidecar's modality scope).

    ``coreg_modality`` is the modality this sidecar describes (e.g.,
    ``"T2w"``, ``"FLAIR"``); used to namespace coregistration keys
    (``CoregistrationT2wMethod`` etc.). Pass ``None`` for the T1w
    reference sidecar (no coregistration applies — T1w is the target).
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    prep = qc_metrics.get("preprocessing")
    if prep and isinstance(prep, dict):
        if "denoising" in prep:
            sidecar["PreprocessingDenoising"] = bool(prep["denoising"])
        if "bias_correction" in prep:
            sidecar["PreprocessingBiasCorrection"] = str(prep["bias_correction"])
        if "acpc_aligned" in prep:
            sidecar["PreprocessingACPCAligned"] = bool(prep["acpc_aligned"])
    if coreg_modality:
        coreg = qc_metrics.get("coregistration")
        if coreg and isinstance(coreg, dict):
            prefix = f"Coregistration{coreg_modality}"
            if "method" in coreg:
                sidecar[f"{prefix}Method"] = str(coreg["method"])
            if "cost_function" in coreg:
                sidecar[f"{prefix}CostFunction"] = str(coreg["cost_function"])
            if "correlation_ratio" in coreg:
                sidecar[f"{prefix}CorrelationRatio"] = float(
                    coreg["correlation_ratio"],
                )
    return sidecar


def enrich_probseg_sidecar(
    sidecar: Dict[str, Any],
    qc_metrics: Optional[Dict[str, Any]],
    tissue_type: str,
) -> Dict[str, Any]:
    """Add flat ``TissueSegmentation*`` keys to a probseg sidecar.

    Reads ``qc_metrics["tissue_segmentation"][tissue_type]``
    (``mean_probability``, ``voxel_count``, ``volume_ml``) for the target
    tissue (WM/GM/CSF). Per-tissue scope is encoded by the sidecar's
    ``label-`` entity; no need to repeat tissue type in key names.
    Volume in ``TissueSegmentationVolume`` is in mL (documented in
    sidecar Description).
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    seg = qc_metrics.get("tissue_segmentation")
    if not seg or not isinstance(seg, dict):
        return sidecar
    raw_block = seg.get(tissue_type)
    block: Dict[str, Any] = raw_block if isinstance(raw_block, dict) else {}
    if "mean_probability" in block:
        sidecar["TissueSegmentationMeanProbability"] = float(block["mean_probability"])
    if "voxel_count" in block:
        sidecar["TissueSegmentationVoxelCount"] = int(block["voxel_count"])
    if "volume_ml" in block:
        sidecar["TissueSegmentationVolume"] = float(block["volume_ml"])
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
    if "brr" in metrics:
        out["AIFBaselineReturnRatio"] = float(metrics["brr"]) / 100.0
    if "psr" in metrics:
        out["AIFPeakSignalRatio"] = float(metrics["psr"]) / 100.0
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
# QC recommendation / warnings enrichment (PR2 of arc-finishing work)
#
# Adds flat top-level ``QC*``-prefixed keys carrying the recommendation
# block and three categories of warnings to the canonical anchor
# sidecars (``desc-preproc_T1w.json``, ``desc-dscPreproc_dsc.json``,
# ``desc-dcePreproc_dce.json``). Eliminates the need for separate
# ``desc-*QC_quality.json`` orphan summary files at session root.
# ---------------------------------------------------------------------------


def enrich_recommendation_keys(
    sidecar: Dict[str, Any],
    *,
    modality: str,
    qc_metrics: Optional[Dict[str, Any]] = None,
    provenance_dict: Optional[Dict[str, Any]] = None,
    preflight_warnings: Optional[Sequence[str]] = None,
    presence_warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Add flat ``QC*``-prefixed recommendation/warnings keys to *sidecar*.

    The aggregator reads these keys from the canonical anchor sidecars
    (the same files used as pipeline presence probes) to build the
    overall recommendation banner and the three warnings banners in the
    HTML report. Empty fields are omitted (per fmriprep convention for
    optional sidecar fields).

    Args:
        sidecar: Sidecar dict to mutate.
        modality: ``"DSC"``, ``"DCE"``, or ``"ANAT"`` (case-insensitive).
            ANAT does not have a recommendation pathway today; only
            ``QCWarnings`` is populated for ANAT.
        qc_metrics: Producer-shape QC metrics envelope. Required for
            recommendation + acquisition-warnings derivation.
        provenance_dict: Provenance dict (DSC/DCE only) — drives
            ``QCProvenanceWarnings``.
        preflight_warnings: Pre-pipeline preflight warnings. Folded into
            ``QCAcquisitionWarnings`` for DSC/DCE; ignored for ANAT.
        presence_warnings: Modality-presence warnings (e.g.,
            "Brain mask in DSC space not available"). Written to
            ``QCWarnings``.

    Returns:
        The mutated *sidecar* dict (for chaining).
    """
    modality = modality.upper()
    if modality not in {"DSC", "DCE", "ANAT"}:
        raise ValueError(
            f"modality must be 'DSC', 'DCE', or 'ANAT'; got {modality!r}"
        )

    if modality in {"DSC", "DCE"} and qc_metrics:
        from pmriprep.utils.qc_standards import recommend_dce_qc, recommend_dsc_qc

        recommender = recommend_dsc_qc if modality == "DSC" else recommend_dce_qc
        enhanced = recommender(qc_metrics)
        rec = enhanced.get("recommendation") or {}
        if rec:
            action = str(rec.get("action", "review"))
            reason = str(rec.get("reason", ""))
            if action:
                sidecar["QCRecommendationAction"] = action
            if reason:
                sidecar["QCRecommendationReason"] = reason

    presence_list: List[str] = []
    if presence_warnings:
        presence_list.extend(str(w) for w in presence_warnings if w)
    if presence_list:
        sidecar["QCWarnings"] = presence_list

    if modality in {"DSC", "DCE"} and provenance_dict and qc_metrics:
        from pmriprep.utils.contrast_agent import collect_default_warnings

        if modality == "DSC":
            prov = collect_default_warnings(
                provenance_dict, "DSC", qc_metrics=qc_metrics,
            )
        else:
            prov = collect_default_warnings(provenance_dict, "DCE")
        if prov:
            sidecar["QCProvenanceWarnings"] = list(prov)

    if modality in {"DSC", "DCE"}:
        from pmriprep.utils.contrast_agent import collect_aif_fallback_warning

        acq: List[str] = []
        if qc_metrics:
            aif_warn = collect_aif_fallback_warning(qc_metrics, modality)
            if aif_warn:
                acq.append(aif_warn)
        if preflight_warnings:
            acq.extend(str(w) for w in preflight_warnings if w)
        if acq:
            sidecar["QCAcquisitionWarnings"] = acq

    return sidecar


# ---------------------------------------------------------------------------
# DCE cosmetic enrichment (PR1 of arc-finishing work)
#
# Mirror of the DSC enrichment pattern shipped in v0.45.0. Reads the
# producer-shape ``qc_metrics`` sub-blocks (``t1_mapping``, ``b1_mapping``,
# ``preprocessing``, ``pk_modeling``, ``dsc_dce_validation``,
# ``cbv_constraint``) and writes flat top-level PascalCase keys per
# bids-spec-expert (round 1, PR1).
# ---------------------------------------------------------------------------


def _dce_t1_cosmetic_keys(t1_qc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flat ``T1*`` keys for DCE preproc-4D sidecar."""
    if not t1_qc or not isinstance(t1_qc, dict):
        return {}
    out: Dict[str, Any] = {}
    if "method" in t1_qc:
        out["T1MappingMethod"] = str(t1_qc["method"])
    if "b1_correction" in t1_qc:
        out["T1B1Correction"] = str(t1_qc["b1_correction"])
    raw_corr = t1_qc.get("t1_correction")
    corr: Dict[str, Any] = raw_corr if isinstance(raw_corr, dict) else {}
    if "correction_applied" in corr:
        out["T1CorrectionApplied"] = str(corr["correction_applied"])
    tissue_used = corr.get("tissue_t1_used", t1_qc.get("tissue_t1_used"))
    if tissue_used is not None:
        out["T1UseTissueLiterature"] = bool(tissue_used)
    return out


def _dce_b1_cosmetic_keys(b1_qc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flat ``B1*`` keys for DCE preproc-4D sidecar."""
    if not b1_qc or not isinstance(b1_qc, dict):
        return {}
    out: Dict[str, Any] = {}
    if "mean_b1" in b1_qc:
        out["B1MeanWholeBrain"] = float(b1_qc["mean_b1"])
    if "pct_correction_gt_20" in b1_qc:
        out["B1PercentVoxelsCorrectionAbove20"] = float(b1_qc["pct_correction_gt_20"])
    return out


def _dce_preprocessing_cosmetic_keys(
    preproc: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Flat motion/signal QC keys for DCE preproc-4D sidecar.

    Mirrors the DSC unprefixed convention (``MotionFDMean``, ...). Adds
    DCE-specific ``BaselineTSNR`` and ``PeakEnhancementPercent``.
    """
    if not preproc or not isinstance(preproc, dict):
        return {}
    out: Dict[str, Any] = {}
    if "fd_mean" in preproc:
        out["MotionFDMean"] = float(preproc["fd_mean"])
    if "fd_max" in preproc:
        out["MotionFDMax"] = float(preproc["fd_max"])
    if "percent_outliers" in preproc:
        out["MotionPercentOutliers"] = float(preproc["percent_outliers"])
    if "motion_status" in preproc:
        out["MotionStatus"] = str(preproc["motion_status"])
    if "baseline_tsnr" in preproc:
        out["BaselineTSNR"] = float(preproc["baseline_tsnr"])
    if "peak_enh_pct" in preproc:
        out["PeakEnhancementPercent"] = float(preproc["peak_enh_pct"])
    if "n_volumes" in preproc:
        out["NumberOfVolumes"] = int(preproc["n_volumes"])
    if "tr_ms" in preproc:
        out["RepetitionTime"] = float(preproc["tr_ms"]) / 1000.0
    return out


def enrich_dce_preproc_sidecar(
    sidecar: Dict[str, Any],
    *,
    qc_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add flat-prefixed cosmetic QC keys to a DCE preproc-4D sidecar.

    Mutates and returns ``sidecar``. Adds ``T1*``, ``B1*``, motion/signal,
    and top-level ``TortuosityCorrectionApplied`` / ``NAWMNormalizationApplied``
    flags from the producer-shape ``qc_metrics`` envelope. Existing
    sidecar fields (``SkullStripped``, ``SpatialReference``, ``Sources``,
    ``Resolution``, ``KtransConvention``) are preserved.
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    sidecar.update(_dce_t1_cosmetic_keys(qc_metrics.get("t1_mapping")))
    sidecar.update(_dce_b1_cosmetic_keys(qc_metrics.get("b1_mapping")))
    sidecar.update(_dce_preprocessing_cosmetic_keys(qc_metrics.get("preprocessing")))
    if "tortuosity_applied" in qc_metrics:
        sidecar["TortuosityCorrectionApplied"] = bool(qc_metrics["tortuosity_applied"])
    if "nawm_normalization" in qc_metrics:
        sidecar["NAWMNormalizationApplied"] = bool(qc_metrics["nawm_normalization"])
    return sidecar


def _dce_pk_cosmetic_keys(pk_qc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flat ``PK*`` keys for DCE per-PK-map sidecars."""
    if not pk_qc or not isinstance(pk_qc, dict):
        return {}
    out: Dict[str, Any] = {}
    if "model" in pk_qc:
        out["PKModel"] = str(pk_qc["model"])
    if "candidate_models" in pk_qc:
        candidates = pk_qc["candidate_models"]
        if isinstance(candidates, (list, tuple)):
            out["PKCandidateModels"] = [str(m) for m in candidates]
    if "qc_voxel_rsquared" in pk_qc:
        out["PKVoxelRsquared"] = float(pk_qc["qc_voxel_rsquared"])
    if "inference" in pk_qc:
        out["PKInferenceEngine"] = str(pk_qc["inference"])
    return out


def enrich_dce_pk_map_sidecar(
    sidecar: Dict[str, Any],
    *,
    qc_metrics: Optional[Dict[str, Any]] = None,
    map_key: str = "",
    lesion_fragment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add flat-prefixed cosmetic keys to a DCE per-PK-map sidecar.

    Mutates and returns ``sidecar``. Adds ``PK*`` keys from
    ``qc_metrics["pk_modeling"]``; cross-modality fields land only on
    the maps they apply to (``DSCDCECrossValCorrelation`` on ``ktrans``,
    ``CBVNegativeFraction`` on ``vp``). AIF provenance
    (``AIFConfidence``, ``AIFUncertaintyInflationFactor``,
    ``AIFFwhmConfidence``, ``AIFGateDecisions``) attached on every PK
    map sidecar so downstream tools can detect reduced-confidence runs
    without cross-referencing the resume sidecar. Per-map
    ``LesionStats`` fragment attached when present.
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        if lesion_fragment:
            sidecar["LesionStats"] = dict(lesion_fragment)
        return sidecar
    sidecar.update(_dce_pk_cosmetic_keys(qc_metrics.get("pk_modeling")))
    aif_block = qc_metrics.get("aif_detection")
    if isinstance(aif_block, dict):
        if "aif_confidence" in aif_block:
            sidecar["AIFConfidence"] = str(aif_block["aif_confidence"])
        if "aif_uncertainty_inflation_factor" in aif_block:
            sidecar["AIFUncertaintyInflationFactor"] = float(
                aif_block["aif_uncertainty_inflation_factor"]
            )
        if "aif_fwhm_confidence" in aif_block:
            sidecar["AIFFwhmConfidence"] = str(aif_block["aif_fwhm_confidence"])
        if "aif_gate_decisions" in aif_block:
            sidecar["AIFGateDecisions"] = list(aif_block["aif_gate_decisions"])
    if map_key == "ktrans":
        xval = qc_metrics.get("dsc_dce_validation")
        if isinstance(xval, dict):
            metrics = xval.get("metrics")
            if isinstance(metrics, dict) and "correlation" in metrics:
                sidecar["DSCDCECrossValCorrelation"] = float(metrics["correlation"])
    if map_key == "vp":
        cbv = qc_metrics.get("cbv_constraint")
        if isinstance(cbv, dict) and "cbv_negative_fraction" in cbv:
            sidecar["CBVNegativeFraction"] = float(cbv["cbv_negative_fraction"])
    if lesion_fragment:
        sidecar["LesionStats"] = dict(lesion_fragment)
    return sidecar
