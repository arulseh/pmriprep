"""Pure-function sidecar builders for BIDS-Derivatives outputs.

These functions build the per-output ``meta_dict`` that is passed to
:class:`niworkflows.interfaces.bids.DerivativesDataSink`. They are
stateless — no ``self.inputs`` reads — so the per-pipeline factory
workflows can call them from inside ``pe.Node(Function(...))`` nodes.

Sidecar field strategy (M3):

- **Sources**: per-output list of BIDS URIs (``bids:raw:sub-X/ses-Y/...``) to
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
import json
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

import numpy as np

from pmriprep.config.constants import HDGLIO_BIDS_LABELS

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

_PIPELINE_QC_STRIP_KEYS = frozenset({
    "fd_list", "rms_list", "outlier_frames",
    "candidate_coords", "candidate_scores",
    "cascade_anchor_outcomes", "cascade_gate_decisions",
    "aif_gate_decisions", "motion_summary",
    "ktrans_histogram", "ktrans_histogram_edges",
    "ve_histogram", "ve_histogram_edges",
    "gamma_fit_auto_metrics",
    "aif_centroid_mm", "aif_voxel_coord",
})

_CONFIG_STRIP_KEYS = frozenset({
    "n_procs", "nprocs", "hdbet_device", "tumor_seg_device",
    "disable_aif_fallback", "n_aif_candidates", "cpi_infer_cpt", "cpi_infer_delay",
    "underlays", "no_t1", "no_dsc_validation",
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


def _is_nonfinite_number(value: Any) -> bool:
    """Return True only for a NaN/Inf real number (python OR numpy scalar).

    Uses ``math.isfinite``, which accepts anything with ``__float__`` (so it
    catches numpy float32/float16 that ``isinstance(v, float)`` misses), and
    returns False for containers, strings, None, and bool (bool is finite).
    """
    if isinstance(value, bool):
        return False
    try:
        return not math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def sanitize_for_json(obj: Any) -> Any:
    """Recursively make a structure safe for strict JSON serialization.

    Two problems this solves for sidecars serialized with ``json.dumps``:

    1. Non-finite numbers (NaN/Inf) have no JSON representation and are emitted
       as bare ``NaN``/``Infinity`` tokens that strict parsers (e.g. the BIDS
       validator) reject. A non-finite numeric value means "not available", so
       dict entries are dropped (readers see the key as absent) and non-finite
       sequence elements become ``null``.
    2. Numpy scalars that are not python-float subclasses (float32/float16/
       int32/...) are not natively serializable, so under a ``default=str``
       encoder they become quoted strings instead of numbers. Finite numpy
       scalars are coerced to their python-native equivalents.

    Tuples are treated like lists.

    Parameters
    ----------
    obj : Any
        A JSON-serializable structure (dict/list/tuple/scalar), possibly nested.

    Returns
    -------
    Any
        The sanitized structure.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            cv = v.item() if isinstance(v, np.generic) else v
            if _is_nonfinite_number(cv):
                continue
            out[k] = sanitize_for_json(cv)
        return out
    if isinstance(obj, (list, tuple)):
        cleaned: List[Any] = []
        for x in obj:
            cx = x.item() if isinstance(x, np.generic) else x
            cleaned.append(None if _is_nonfinite_number(cx) else sanitize_for_json(cx))
        return cleaned
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.generic):
        native = obj.item()
        return None if _is_nonfinite_number(native) else native
    return obj


def dump_sidecar_json(obj: Any) -> str:
    """Serialize a sidecar/QC dict to a JSON string, sanitized for strict validity.

    Runs :func:`sanitize_for_json` first (drops NaN/Inf, coerces numpy scalars to
    native types) so the output never carries a bare ``NaN``/``Infinity`` token or
    a numpy-scalar-turned-string, then dumps with the pipeline's standard
    ``indent=2`` and a ``default=str`` fallback for any remaining non-serializable.
    """
    return json.dumps(sanitize_for_json(obj), indent=2, default=str)


def build_provenance_block(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``PipelineConfiguration`` block for a paired sidecar."""
    return {
        "ProcessingConfiguration": {
            k: v for k, v in config_dict.items()
            if k not in _CONFIG_STRIP_KEYS and v is not None
        },
    }


# ---------------------------------------------------------------------------
# Paths and URIs
# ---------------------------------------------------------------------------


def make_bids_source_uri(source_path: PathLike, bids_dir: PathLike) -> str:
    """Return a BIDS URI for a raw-dataset or external source path."""
    source_path = Path(source_path).resolve()
    bids_dir = Path(bids_dir).resolve()
    try:
        rel = source_path.relative_to(bids_dir)
        return f"bids:raw:{rel}"
    except ValueError:
        return source_path.as_uri()


def build_sources_list(
    source_paths: Sequence[PathLike],
    bids_dir: PathLike,
) -> List[str]:
    """Return source URIs for each input path, skipping absent values."""
    out: List[str] = []
    for p in source_paths:
        if p is None or p == "":
            continue
        out.append(make_bids_source_uri(p, bids_dir))
    return out


def source_entity_prefix(
    source: PathLike | None,
    subject: str,
    session: Optional[str],
) -> str:
    """Build a derivative prefix while preserving source filename entities.

    Parameters
    ----------
    source : path-like or None
        Raw source path or BIDS URI. Only recognized filename entities are
        propagated; the source suffix and extension are discarded.
    subject, session : str
        Output hierarchy identifiers. These remain authoritative so a malformed
        source URI cannot redirect an output into another subject/session.

    Returns
    -------
    str
        Canonically ordered ``sub-``/``ses-`` plus relevant source entities.
    """
    prefix = [f"sub-{subject}"]
    if session:
        prefix.append(f"ses-{session}")
    if not source:
        return "_".join(prefix)

    # Values use BIDS labels/indices and therefore cannot contain underscores.
    # Parsing the basename also works for ``bids:raw:sub-...`` URIs.
    name = Path(str(source)).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = Path(name).stem
    found: dict[str, str] = {}
    aliases = {
        "task": "task", "acq": "acq", "ce": "ce", "rec": "rec",
        "dir": "dir", "run": "run", "echo": "echo", "part": "part",
        "chunk": "chunk",
    }
    for token in name.split("_"):
        key, separator, value = token.partition("-")
        if separator and key in aliases and value:
            found[aliases[key]] = value
    for key in ("task", "acq", "ce", "rec", "dir", "run", "echo", "part", "chunk"):
        if key in found:
            prefix.append(f"{key}-{found[key]}")
    return "_".join(prefix)


def build_spatial_reference_uri(
    reference_filename: str,
    subject: str,
    session: Optional[str],
    datatype: str = "anat",
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
    parts = [f"sub-{subject}"]
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
    """Return the PMRIPrep ``OriginalResolution`` block."""
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


# Dockerfile-pinned fallback versions for tools that expose no ``__version__``.
_HDBET_PINNED_VERSION = "2.0.1"
_HDGLIO_PINNED_VERSION = "2.0"

# Third-party software suites credited in the dataset-level ``GeneratedBy`` and
# rendered (with version) in the methods boilerplate. One entry per *suite*
# (a suite has a single install version) — binaries are named in Description.
# The idempotency filter (``_PMRIPREP_MANAGED_TOOL_NAMES``) is derived from the
# same list so it can never drift from ``build_tool_generatedby_entries``.
_THIRD_PARTY_TOOL_NAMES = ("FSL", "ANTs", "Nipype", "HD-BET", "HD-GLIO")
_PMRIPREP_MANAGED_TOOL_NAMES = frozenset({"pmriprep", *_THIRD_PARTY_TOOL_NAMES})


def _normalize_tool_version(raw: str) -> str:
    """Reduce a version probe to a bare version string, or ``""`` if unresolved.

    ``"fsl-6.0.7.1"`` -> ``"6.0.7.1"``, ``"ANTs Version: 2.6.5"`` -> ``"2.6.5"``.
    Not forced to 3-component semver (FSL ``6.0.7.1`` is kept as-is). Any probe
    without a dotted numeric version (``"unknown"``, ``"not installed"``, empty)
    returns ``""`` so callers omit the key rather than emit a garbage version.
    """
    import re

    if not raw:
        return ""
    match = re.search(r"\d+(?:\.\d+)+[A-Za-z0-9.\-]*", raw.strip())
    return match.group(0) if match else ""


def _dist_version(dist_name: str, fallback: str) -> str:
    """Read an installed distribution's version from package metadata.

    Uses ``importlib.metadata`` (reads the dist-info, does NOT import the
    package) so probing e.g. HD-BET does not execute its ``__init__`` — which
    prints a citation banner to stdout. Falls back to the Dockerfile-pinned
    value when the distribution is not installed.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(dist_name)
        except PackageNotFoundError:
            return fallback
    except Exception:
        return fallback


def get_hdbet_version() -> str:
    """Return the HD-BET version, falling back to the Dockerfile-pinned value."""
    return _dist_version("hd-bet", _HDBET_PINNED_VERSION)


def get_hdglio_version() -> str:
    """Return the HD-GLIO version, falling back to the Dockerfile-pinned value."""
    return _dist_version("hd-glio", _HDGLIO_PINNED_VERSION)


def get_imaging_software_versions() -> Dict[str, str]:
    """Return the version map for the third-party imaging tools pmriprep uses.

    Keys match the ``GeneratedBy`` ``Name`` values so the boilerplate's
    live-query fallback aligns with the recorded provenance.
    """
    from pmriprep.utils.provenance import get_software_versions

    return {
        "FSL": get_fsl_version(),
        "ANTs": get_ants_version(),
        "Nipype": get_software_versions().get("nipype", "unknown"),
        "HD-BET": get_hdbet_version(),
        "HD-GLIO": get_hdglio_version(),
    }


def build_tool_generatedby_entries() -> List[Dict[str, Any]]:
    """Build per-suite ``GeneratedBy`` objects for pmriprep's third-party tools.

    One object per software suite (FSL, ANTs, Nipype, HD-BET, HD-GLIO), each with
    ``Name`` (+ ``Version`` when resolvable — never the literal ``"unknown"``),
    ``Description`` naming the constituent binaries, and ``CodeURL``. Version
    probes run once here (they carry subprocess timeouts).
    """
    from pmriprep.utils.provenance import get_software_versions

    # Keyed by the shared _THIRD_PARTY_TOOL_NAMES so the entry set and the
    # idempotency filter stay in lockstep. Version probes run once here.
    raw_versions = {
        "FSL": get_fsl_version(),
        "ANTs": get_ants_version(),
        "Nipype": get_software_versions().get("nipype", ""),
        "HD-BET": get_hdbet_version(),
        "HD-GLIO": get_hdglio_version(),
    }
    metadata = {
        "FSL": ("Tissue segmentation (FAST), linear registration and motion "
                "correction (FLIRT, MCFLIRT)", "https://fsl.fmrib.ox.ac.uk"),
        "ANTs": ("Intensity non-uniformity correction (N4BiasFieldCorrection) "
                 "and denoising (DenoiseImage)", "https://github.com/ANTsX/ANTs"),
        "Nipype": ("Workflow engine", "https://github.com/nipy/nipype"),
        "HD-BET": ("Brain extraction", "https://github.com/MIC-DKFZ/HD-BET"),
        "HD-GLIO": ("Automated glioma segmentation",
                    "https://github.com/NeuroAI-HD/HD-GLIO"),
    }
    entries: List[Dict[str, Any]] = []
    for name in _THIRD_PARTY_TOOL_NAMES:
        description, code_url = metadata[name]
        entry: Dict[str, Any] = {"Name": name}
        version = _normalize_tool_version(raw_versions[name])
        if version:
            entry["Version"] = version
        entry["Description"] = description
        entry["CodeURL"] = code_url
        entries.append(entry)
    return entries


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
    subject: str,
    session: Optional[str],
    is_postcon_ref: bool = False,
    source: PathLike | None = None,
) -> str:
    if source is None:
        return build_spatial_reference_uri(
            anat_reference_basename(is_postcon_ref),
            subject, session, datatype="anat",
        )
    prefix = source_entity_prefix(source, subject, session)
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.extend([
        "anat",
        f"{prefix}_space-T1w_desc-preproc_T1w.nii.gz",
    ])
    return "bids::" + "/".join(parts)


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
    reference_source: PathLike | None = None,
) -> Dict[str, Any]:
    """Build sidecar for a preprocessed/coregistered anat NIfTI."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sources = build_sources_list(source_paths, bids_dir)
    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": description,
        "SkullStripped": skull_stripped,
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or (source_paths[0] if source_paths else None),
        ),
        "Sources": sources,
        **resolution_native_block(reference_voxel_source),
        "OriginalResolution": compute_original_resolution(reference_voxel_source),
        "ProcessingSteps": list(_ANAT_PREPROC_STEPS),
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
    reference_source: PathLike | None = None,
) -> Dict[str, Any]:
    """Build sidecar for an anat-space brain mask."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    return {
        **get_bids_derivative_provenance(),
        "Description": description,
        "SkullStripped": True,
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or (source_paths[0] if source_paths else None),
        ),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
        "OriginalResolution": compute_original_resolution(reference_voxel_source),
        "ProcessingSteps": list(_ANAT_PREPROC_STEPS),
    }


def _tumor_volumetrics(seg_path: PathLike) -> Dict[str, Any]:
    """Compute per-region tumor volumes from the dseg NIfTI."""
    import nibabel as nib  # type: ignore[import-untyped]

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
    reference_source: PathLike | None = None,
) -> Dict[str, Any]:
    """Sidecar for the multi-class tumor ``dseg`` label map.

    Per BIDS-Common-Derivatives, a single ``_dseg.json`` describes both
    the ``_dseg.nii.gz`` label NIfTI and the ``_dseg.tsv`` lookup table
    (when the latter is present) since they share a basename. The
    ``Columns`` block documents the TSV columns; ``TumorVolumetrics``
    carries per-region volumes and the enhancing fraction.
    ``TumorSegQC`` carries preprocessing/segmentation/nawm QC blocks.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sc: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": "Deep learning tumor segmentation",
        "Manual": False,
        "SegmentationLabels": dict(HDGLIO_BIDS_LABELS),
        "Method": "HD-GLIO",
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or (source_paths[0] if source_paths else None),
        ),
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
                "Units": "1",
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
    reference_source: PathLike | None = None,
) -> Dict[str, Any]:
    """Sidecar for a binary tumor-derived mask (enhancing/nonenhancing/whole/lesion)."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    return {
        **get_bids_derivative_provenance(),
        "Description": description,
        "Type": "Lesion",
        "Method": method,
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or (source_paths[0] if source_paths else None),
        ),
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
    reference_source: PathLike | None = None,
) -> Dict[str, Any]:
    """Sidecar for the contralateral NAWM mask."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    return {
        **get_bids_derivative_provenance(),
        "Description": (
            "Contralateral normal-appearing white matter mask for CBV "
            "normalization"
        ),
        "Type": "ROI",
        "Method": (
            "Contralateral WM thresholded at p>=0.9, excluding tumor margin"
        ),
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or (source_paths[0] if source_paths else None),
        ),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
    }


def dce_reference_basename() -> str:
    """Basename of the DCE reference image inside the derivatives tree."""
    return "desc-dceMean_stat.nii.gz"


def _perfusion_spatial_reference(
    reference_basename: str,
    subject: str,
    session: Optional[str],
    source: PathLike | None,
) -> str:
    """Return a derivative-local perfusion reference URI with source entities."""
    prefix = source_entity_prefix(source, subject, session)
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.extend(["perf", f"{prefix}_{reference_basename}"])
    return "bids::" + "/".join(parts)


def _dce_spatial_reference(
    subject: str,
    session: Optional[str],
    source: PathLike | None = None,
) -> str:
    return _perfusion_spatial_reference(
        dce_reference_basename(), subject, session, source,
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
    pk_dependent: bool = True,
) -> Dict[str, Any]:
    """Build common base sidecar for DCE derivatives.

    Parameters
    ----------
    minimal
        Suppress perfusion-specific provenance enrichment and
        PK-input metadata (used for masks, structural underlays,
        motion-corrected 4D, T1map, B1+map, etc.).
    pk_dependent
        Record that the derivative uses a plasma input function. Only
        relevant when ``minimal=False``; set to ``False`` for model-free
        and QC maps.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "SpatialReference": _dce_spatial_reference(
            subject, session, source_paths[0] if source_paths else None,
        ),
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
        if pk_dependent:
            sidecar["InputFunctionCompartment"] = "plasma"
    return sidecar


def dsc_reference_basename() -> str:
    """Basename of the DSC reference image inside the derivatives tree."""
    return "desc-dscMean_stat.nii.gz"


def _dsc_spatial_reference(
    subject: str,
    session: Optional[str],
    source: PathLike | None = None,
) -> str:
    return _perfusion_spatial_reference(
        dsc_reference_basename(), subject, session, source,
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
    """Build common base sidecar for DSC derivatives.

    ``minimal=True`` suppresses perfusion-specific provenance enrichment
    (used for masks, structural underlays, motion-corrected 4D, etc.).
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "SpatialReference": _dsc_spatial_reference(
            subject, session, source_paths[0] if source_paths else None,
        ),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
    }
    if provenance_dict:
        sidecar.update(provenance_dict)
    if minimal:
        pp = sidecar.get("PerfusionProcessing")
        if isinstance(pp, dict):
            for key in ("Deconvolution", "LeakageCorrection",
                        "PreloadDetection"):
                pp.pop(key, None)
    elif provenance_dict and qc_metrics:
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
    from pmriprep.utils.aif_provenance import apply_aif_sidecar

    sidecar: Dict[str, Any] = {**base, "Description": description}
    if units is not None:
        sidecar["Units"] = units
    if extra:
        sidecar.update(extra)
    # Surface the per-run first-pass-CBV reference so the named reference and
    # its voxel count land on the map's own sidecar.
    if map_key == "cbv_firstpass" and isinstance(qc_metrics, dict):
        ref = qc_metrics.get("firstpass_reference")
        if ref is not None:
            sidecar["FirstPassReference"] = str(ref)
        ref_n = qc_metrics.get("firstpass_reference_n")
        if ref_n is not None:
            sidecar["FirstPassReferenceVoxelCount"] = int(ref_n)
        unfit = qc_metrics.get("firstpass_unfittable_fraction")
        if unfit is not None:
            sidecar["FirstPassUnfittableFraction"] = float(unfit)
        fwhm = qc_metrics.get("firstpass_prefit_fwhm_mm")
        if fwhm is not None:
            sidecar["FirstPassPreSmoothingFWHMmm"] = float(fwhm)
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
    reference_source: PathLike | None = None,
) -> Dict[str, Any]:
    """Build sidecar for a tissue probability map (``label-{WM,GM,CSF}``)."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    return {
        **get_bids_derivative_provenance(),
        "Description": f"{tissue_type} probability map from FAST tissue segmentation",
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or (source_paths[0] if source_paths else None),
        ),
        "Sources": build_sources_list(source_paths, bids_dir),
        **resolution_native_block(reference_voxel_source),
        "ReferenceContrastType": "precon",
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
# and inject the relevant subset of pipeline QC state into the sidecar.
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
    if "device" in be:
        sidecar["BrainExtractionDevice"] = str(be["device"])
    if "voxel_count" in be:
        sidecar["BrainExtractionVoxelCount"] = int(be["voxel_count"])
    # Accept both fully qualified and short volume keys.
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


def aif_timeseries_columns(modality: str) -> tuple[str, str, str]:
    """Column name, units, and description for an AIF timeseries TSV.

    Single source of truth for both the TSV column header (written by the
    ``qc_writers`` AIF emitters) and the JSON sidecar
    ``Columns``/``Units``/``ColumnDescriptions`` built by
    :func:`build_aif_timeseries_sidecar`, so the two cannot drift.

    Parameters
    ----------
    modality : str
        ``"DCE"`` or ``"DSC"``.

    Returns
    -------
    tuple of (str, str, str)
        ``(column_name, units, column_description)``.
    """
    if modality not in {"DCE", "DSC"}:
        raise ValueError(f"modality must be 'DCE' or 'DSC'; got {modality!r}")

    # AIF curve content differs by modality:
    #   DSC: ΔR2* (1/s) — susceptibility-induced relaxation-rate change
    #   DCE: plasma [Gd] in mM (after /(1-Hct) conversion)
    if modality == "DSC":
        return (
            "delta_r2_star",
            "1/s",
            "Susceptibility-induced change in transverse relaxation rate "
            "(delta-R2*) at the AIF voxel(s).",
        )
    return (  # DCE
        "plasma_concentration_mM",
        "mM",
        "Plasma gadolinium concentration after blood-to-plasma "
        "conversion (/(1-Hct)).",
    )


def build_aif_timeseries_sidecar(
    *,
    modality: str,
    source_paths: Sequence[PathLike],
    bids_dir: PathLike,
    subject: str,
    session: Optional[str],
    aif_qc: Optional[Dict[str, Any]] = None,
    volume_tr_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the JSON sidecar for the AIF curve text file.

    Per BIDS-Common-Derivatives, every data file should carry its own
    sidecar describing how it was derived. This builder emits AIF
    detection metadata in the canonical ``PerfusionProcessing.AIF`` block
    shared by AIF masks and quantitative-map sidecars.

    Parameters
    ----------
    modality : str
        ``"DCE"`` or ``"DSC"``.
    source_paths : Sequence of PathLike
        Source 4D NIfTI paths the AIF was extracted from.
    bids_dir : PathLike
        BIDS rawdata root for ``Sources`` BIDS-URI construction.
    subject : str
        Subject ID without ``sub-`` prefix.
    session : str or None
        Session ID without ``ses-`` prefix (or None).
    aif_qc : dict, optional
        Producer-shape AIF QC dict (the ``aif_qc`` dict from the
        DSC outputnode for DSC, or the equivalent for DCE).
    volume_tr_s : float, optional
        Time between successive AIF samples in seconds. When positive, the
        sidecar records ``SamplingFrequency`` and ``StartTime``.

    Returns
    -------
    dict
        Dict suitable for ``json.dumps``.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    column_name, units, column_description = aif_timeseries_columns(modality)

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
    if volume_tr_s is not None and volume_tr_s > 0:
        sidecar["SamplingFrequency"] = 1.0 / float(volume_tr_s)
        sidecar["StartTime"] = 0.0

    from pmriprep.utils.aif_provenance import apply_aif_provenance_block

    apply_aif_provenance_block(
        sidecar,
        aif_qc,
        modality=cast(Literal["DSC", "DCE"], modality),
    )
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
        out["PreloadK2Adequacy"] = str(preload_qc["preload_k2_classification"])
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
    overall recommendation banner and the four warnings banners in the
    HTML report. Empty fields are omitted (per fmriprep convention for
    optional sidecar fields).

    Parameters
    ----------
    sidecar : dict
        Sidecar dict to mutate.
    modality : str
        ``"DSC"``, ``"DCE"``, or ``"ANAT"`` (case-insensitive).
        ANAT does not have a recommendation pathway today; only
        ``QCWarnings`` is populated for ANAT.
    qc_metrics : dict, optional
        Producer-shape QC metrics envelope. Required for
        recommendation + acquisition-warnings derivation.
    provenance_dict : dict, optional
        Provenance dict (DSC/DCE only) — drives
        ``QCProvenanceWarnings``.
    preflight_warnings : Sequence of str, optional
        Pre-pipeline preflight warnings. Folded into
        ``QCAcquisitionWarnings`` for DSC/DCE; ignored for ANAT.
    presence_warnings : Sequence of str, optional
        Modality-presence warnings (e.g.,
        "Brain mask in DSC space not available"). Written to
        ``QCWarnings``.

    Returns
    -------
    dict
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
            prov = collect_default_warnings(provenance_dict, "DSC")
        else:
            prov = collect_default_warnings(provenance_dict, "DCE")
        if prov:
            sidecar["QCProvenanceWarnings"] = list(prov)

    if modality in {"DSC", "DCE"}:
        from pmriprep.utils.contrast_agent import collect_aif_fallback_warning

        acq: List[str] = []
        if preflight_warnings:
            acq.extend(str(w) for w in preflight_warnings if w)
        if acq:
            sidecar["QCAcquisitionWarnings"] = acq

        if qc_metrics:
            processing_warnings: List[str] = []
            aif_warn = collect_aif_fallback_warning(qc_metrics, modality)
            if aif_warn:
                processing_warnings.append(aif_warn)
            if modality == "DCE":
                preprocessing = qc_metrics.get("preprocessing") or {}
                for warning_key in ("baseline_warning", "motion_qc_warning"):
                    warning = preprocessing.get(warning_key)
                    if warning:
                        processing_warnings.append(str(warning))
                t1_mapping = qc_metrics.get("t1_mapping") or {}
                vfa_preprocessing = t1_mapping.get("vfa_preprocessing") or {}
                motion_assessment = vfa_preprocessing.get("motion_assessment") or {}
                vfa_warning = motion_assessment.get("warning")
                if vfa_warning:
                    processing_warnings.append(str(vfa_warning))
            if processing_warnings:
                sidecar["QCProcessingWarnings"] = processing_warnings

    return sidecar


# ---------------------------------------------------------------------------
# DCE cosmetic enrichment (PR1 of arc-finishing work)
#
# Enriches the sidecar with QC metrics read from the
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
    method = t1_qc.get("method", t1_qc.get("t1_method"))
    if method is not None:
        out["T1MappingMethod"] = str(method)
    if "b1_correction" in t1_qc:
        out["T1B1Correction"] = str(t1_qc["b1_correction"])
    if "assumed_t1_ms" in t1_qc and t1_qc["assumed_t1_ms"] is not None:
        out["T1AssumedValue"] = float(t1_qc["assumed_t1_ms"])
        out["T1AssumedValueUnits"] = "ms"
    if (
        "assumed_t1_field_strength_t" in t1_qc
        and t1_qc["assumed_t1_field_strength_t"] is not None
    ):
        out["T1AssumptionFieldStrength"] = float(
            t1_qc["assumed_t1_field_strength_t"]
        )
    if "assumed_t1_policy" in t1_qc and t1_qc["assumed_t1_policy"] is not None:
        out["T1AssumptionPolicy"] = str(t1_qc["assumed_t1_policy"])
    raw_corr = t1_qc.get("t1_correction")
    corr: Dict[str, Any] = raw_corr if isinstance(raw_corr, dict) else {}
    if "correction_applied" in corr:
        out["T1CorrectionApplied"] = str(corr["correction_applied"])
    if method == "assumed":
        out["T1Source"] = "uniform_field_strength_assumption"
        out["T1MappingAttempted"] = False
    elif corr.get("correction_applied") == "fallback":
        out["T1Source"] = "uniform_field_strength_fallback"
        out["T1MappingAttempted"] = True
        out["T1MappingAccepted"] = False
        if corr.get("fallback_value_ms") is not None:
            out["T1AssumedValue"] = float(corr["fallback_value_ms"])
            out["T1AssumedValueUnits"] = "ms"
        if corr.get("reason"):
            out["T1FallbackReason"] = str(corr["reason"])
    else:
        out["T1Source"] = "external_map" if method == "external" else "measured_vfa"
        out["T1MappingAttempted"] = method != "external"
        if method != "external":
            out["T1MappingAccepted"] = True
    support = t1_qc.get("t1_support")
    if isinstance(support, dict):
        out["T1Support"] = dict(support)
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
    if "motion_qc_available" in preproc:
        out["MotionQCAvailable"] = bool(preproc["motion_qc_available"])
    if "motion_qc_error" in preproc:
        out["MotionQCError"] = str(preproc["motion_qc_error"])
    tsnr = preproc.get("baseline_tsnr")
    if tsnr is not None and tsnr == tsnr:  # present and not NaN
        out["BaselineTSNR"] = float(tsnr)
    snr_low = preproc.get("baseline_snr_low_fraction")
    if snr_low is not None and snr_low == snr_low:  # present and not NaN
        out["BaselineSNRLowFraction"] = float(snr_low)
    if "peak_enh_pct" in preproc:
        out["PeakEnhancementPercent"] = float(preproc["peak_enh_pct"])
    if "n_volumes" in preproc:
        out["NumberOfVolumes"] = int(preproc["n_volumes"])
    if "baseline_frames_used" in preproc:
        out["BaselineFramesUsed"] = int(preproc["baseline_frames_used"])
    if "baseline_minimum_frames" in preproc:
        out["BaselineMinimumFrames"] = int(preproc["baseline_minimum_frames"])
    if "detected_bolus_frame" in preproc:
        out["BolusArrivalFrameDetected"] = int(preproc["detected_bolus_frame"])
    if "baseline_fallback_applied" in preproc:
        out["BaselineFallbackApplied"] = bool(preproc["baseline_fallback_applied"])
    if "baseline_fallback_reason" in preproc:
        out["BaselineFallbackReason"] = str(preproc["baseline_fallback_reason"])
    if "baseline_status" in preproc:
        out["BaselineQCStatus"] = str(preproc["baseline_status"])
    if "volume_tr_s" in preproc:
        out["RepetitionTime"] = float(preproc["volume_tr_s"])
        if "tr_ms" in preproc:
            out["RepetitionTimeExcitation"] = float(preproc["tr_ms"]) / 1000.0
    return out


def enrich_dce_preproc_sidecar(
    sidecar: Dict[str, Any],
    *,
    qc_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add flat-prefixed cosmetic QC keys to a DCE preproc-4D sidecar.

    Mutates and returns ``sidecar``. Adds ``T1*``, ``B1*``, motion/signal,
    and top-level ``NAWMNormalizationApplied``
    flags from the producer-shape ``qc_metrics`` envelope. Existing
    sidecar fields (``SkullStripped``, ``SpatialReference``, ``Sources``,
    ``Resolution``) are preserved.
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    sidecar.update(_dce_t1_cosmetic_keys(qc_metrics.get("t1_mapping")))
    sidecar.update(_dce_b1_cosmetic_keys(qc_metrics.get("b1_mapping")))
    sidecar.update(_dce_preprocessing_cosmetic_keys(qc_metrics.get("preprocessing")))
    if "nawm_normalization" in qc_metrics:
        sidecar["NAWMNormalizationApplied"] = bool(qc_metrics["nawm_normalization"])
    return sidecar


def enrich_dce_pk_map_sidecar(
    sidecar: Dict[str, Any],
    *,
    qc_metrics: Optional[Dict[str, Any]] = None,
    map_key: str = "",
    lesion_fragment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Enrich a DCE per-PK-map sidecar with AIF provenance and cross-modality fields.

    Stamps ``PerfusionProcessing.AIF`` block plus flat ``AIFDependent``
    and ``TemplateAIFUsage`` tags via the shared AIF provenance module.
    Cross-modality fields land only on the maps they apply to.
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        if lesion_fragment:
            sidecar["LesionStats"] = dict(lesion_fragment)
        return sidecar
    if map_key:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        apply_aif_sidecar(sidecar, map_key, qc_metrics, modality="DCE")
    # AIF->tissue BAT pre-alignment provenance (records the applied shift, or
    # the reason none was applied) on every AIF-dependent PK map. Read from both
    # nesting levels (top-level for a direct node qc dict; under "pk_modeling"
    # for the merged pipeline qc) — same both-levels handling as the AIF block.
    bat = qc_metrics.get("bat_correction") or qc_metrics.get("pk_modeling", {}).get("bat_correction")
    if isinstance(bat, dict):
        sidecar.setdefault("PerfusionProcessing", {})["BATCorrection"] = dict(bat)
    pk_modeling = qc_metrics.get("pk_modeling")
    fallback = qc_metrics.get("inference_fallback")
    if fallback is None and isinstance(pk_modeling, dict):
        fallback = pk_modeling.get("inference_fallback")
    if isinstance(fallback, dict):
        sidecar.setdefault("PerfusionProcessing", {})["InferenceFallback"] = dict(
            fallback
        )
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
