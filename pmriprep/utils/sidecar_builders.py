"""Pure-function sidecar builders for BIDS-Derivatives outputs.

These functions build the per-output ``meta_dict`` that is passed to
:class:`niworkflows.interfaces.bids.DerivativesDataSink`. They are
stateless — no ``self.inputs`` reads — so the per-pipeline factory
workflows can call them from inside ``pe.Node(Function(...))`` nodes.

Sidecar field strategy:

- **Sources**: per-output list of BIDS URIs (``bids:raw:sub-X/ses-Y/...``) to
  the immediate raw inputs that produced the derivative.
- **SpatialReference**: BIDS URI to the on-disk reference image whose
  grid the derivative occupies (e.g., the preproc T1w for ``space-T1w``,
  the mean DCE/DSC ref for native acquisition space). For native
  acquisition space in perf/, per BIDS-Common-Derivatives convention,
  outputs omit the ``space-`` entity. For anat/ outputs, ``space-T1w``
  is used uniformly per fmriprep precedent; pre- vs post-contrast
  disambiguation lives in this URI.
- **VoxelSize**: numeric voxel sizes in millimetres for the emitted image.
  The BIDS ``Resolution`` metadata field is reserved for describing ``res-``
  entity values and is not used as a resampling-status flag.
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
    "fd_list", "outlier_frames",
    "observation_indices", "observation_times_seconds",
    "original_observation_times_seconds",
    "tissue_baseline_diagnostics", "gamma_fit_diagnostics",
    "cascade_postfilter_diags", "cascade_spatial_filter_diags",
    "motion_command", "motion_executable", "motion_reference",
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
    "n_aif_candidates",
    "underlays", "no_t1", "no_dsc_validation",
})


def filter_pipeline_qc(qc: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal QC fields and metadata inapplicable to the processing path."""
    filtered = copy.deepcopy(qc)

    def _strip(d: dict) -> None:
        for key in list(d):
            if key in _PIPELINE_QC_STRIP_KEYS:
                del d[key]
            elif isinstance(d[key], dict):
                _strip(d[key])

    _strip(filtered)
    registration = filtered.get("registration", {})
    if isinstance(registration, dict) and registration.get("registration_method") == "header":
        selection = registration.get("path_selection", {})
        if isinstance(selection, dict):
            for key in list(selection):
                if key.startswith("selected_vs_header_"):
                    del selection[key]
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
       values become ``null`` while their keys or sequence positions are retained.
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
    return None if _is_nonfinite_number(obj) else obj


def dump_sidecar_json(obj: Any) -> str:
    """Serialize a sidecar/QC dict to a JSON string, sanitized for strict validity.

    Runs :func:`sanitize_for_json` first (maps NaN/Inf to null, coerces numpy scalars to
    native types) so the output never carries a bare ``NaN``/``Infinity`` token or
    a numpy-scalar-turned-string, then dumps with the pipeline's standard
    ``indent=2`` and a ``default=str`` fallback for any remaining non-serializable.
    """
    return json.dumps(sanitize_for_json(obj), indent=2, default=str)


def build_provenance_block(
    config_dict: Dict[str, Any], qc_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``PipelineConfiguration`` block for a paired sidecar."""
    config = {
        k: v for k, v in config_dict.items()
        if k not in _CONFIG_STRIP_KEYS and v is not None
    }
    if config.get("method") in {"sSVD", "oSVD", "tikhonov"}:
        config = {k: v for k, v in config.items()
                  if not k.startswith("cpi_") and k != "inference"}
    t1 = (qc_metrics or {}).get("t1_mapping") or {}
    if t1.get("method") == "assumed" and t1.get("skipped"):
        config.pop("t1_method", None)
    return {"ProcessingConfiguration": config}


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
    return {
        "VoxelDimensions": voxel_dims,
        "GeometricMean": geo_mean,
        "IsotropyRatio": iso_ratio,
    }


def voxel_size_block(source_nifti: PathLike) -> Dict[str, Any]:
    """Return the emitted image voxel sizes in millimetres."""
    return {"VoxelSize": voxel_sizes_from_nifti(source_nifti)}


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
                return line.strip().removeprefix("ANTs Version:").strip()
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
        "FSL": ("Tissue segmentation (FAST) and linear registration (FLIRT)", "https://fsl.fmrib.ox.ac.uk"),
        "ANTs": ("Intensity non-uniformity correction (N4BiasFieldCorrection) "
                 "denoising (DenoiseImage) and rigid motion correction (antsMotionCorr)", "https://github.com/ANTsX/ANTs"),
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


_ANAT_REFERENCE_STEPS = [
    "Reorientation to RAS and robust FOV cropping",
    "N4 bias field correction",
    "Conditional conformation of submillimeter input voxels to 1 mm",
    "Brain extraction",
]

_ANAT_COREGISTERED_STEPS = [
    "Reorientation to RAS",
    "N4 bias field correction",
    "Rigid registration to the anatomical reference",
    "Application of the anatomical brain mask",
]

_ANAT_CROPPED_COREGISTERED_STEPS = [
    "Reorientation to RAS and robust FOV cropping",
    "N4 bias field correction",
    "Rigid registration to the anatomical reference",
    "Application of the anatomical brain mask",
]

_ANAT_SUBTRACTION_STEPS = [
    "Rigid registration of the post-contrast T1w image to the pre-contrast reference",
    "Post-contrast minus pre-contrast signal subtraction within the brain mask",
]


def subtraction_signal_metadata() -> Dict[str, Any]:
    """Describe the non-quantitative signal scale of a T1w subtraction image."""
    return {
        "Units": "arbitrary",
        "Estimator": "registered_postcontrast_signal_minus_precontrast_signal",
        "IntensityNormalizationApplied": False,
        "Quantitative": False,
        "Interpretation": (
            "Within-subject visualization of signal change; values retain "
            "scanner-dependent intensity scaling and are not suitable for "
            "cross-subject or fixed-threshold quantification."
        ),
    }


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
    if is_postcon_ref and "_ce-" not in f"_{prefix}":
        entities = prefix.split("_")
        insert_at = next(
            (
                index for index, entity in enumerate(entities)
                if entity.startswith(("rec-", "dir-", "run-", "echo-", "part-", "chunk-"))
            ),
            len(entities),
        )
        entities.insert(insert_at, "ce-GAD")
        prefix = "_".join(entities)
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.extend([
        "anat",
        f"{prefix}_space-T1w_desc-preproc_T1w.nii.gz",
    ])
    return "bids::" + "/".join(parts)


def _anat_derivative_uri(
    subject: str,
    session: Optional[str],
    source: PathLike | None,
    filename_tail: str,
) -> str:
    """Build a derivative-local URI using the anatomical source entities."""
    prefix = source_entity_prefix(source, subject, session)
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.extend(["anat", f"{prefix}_{filename_tail}"])
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
    processing_kind: str = "reference",
) -> Dict[str, Any]:
    """Build sidecar for a preprocessed/coregistered anat NIfTI.

    Raises
    ------
    ValueError
        If ``processing_kind`` is unrecognized.
    """
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    sources = build_sources_list(source_paths, bids_dir)
    processing_steps = {
        "reference": _ANAT_REFERENCE_STEPS,
        "coregistered": _ANAT_COREGISTERED_STEPS,
        "cropped_coregistered": _ANAT_CROPPED_COREGISTERED_STEPS,
        "subtraction": _ANAT_SUBTRACTION_STEPS,
    }
    if processing_kind not in processing_steps:
        raise ValueError(f"Unknown anatomical processing kind: {processing_kind!r}")
    original_source = source_paths[0] if source_paths else reference_voxel_source
    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": description,
        "SkullStripped": skull_stripped,
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or (source_paths[0] if source_paths else None),
        ),
        "Sources": sources,
        **voxel_size_block(reference_voxel_source),
        "OriginalResolution": compute_original_resolution(original_source),
        "ProcessingSteps": list(processing_steps[processing_kind]),
        "CoregistrationApplied": processing_kind in {
            "coregistered", "cropped_coregistered", "subtraction",
        },
    }
    if processing_kind == "subtraction":
        sidecar.update(subtraction_signal_metadata())
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
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or (source_paths[0] if source_paths else None),
        ),
        "Sources": build_sources_list(source_paths, bids_dir),
        **voxel_size_block(reference_voxel_source),
        "OriginalResolution": compute_original_resolution(
            source_paths[0] if source_paths else reference_voxel_source
        ),
        "ProcessingSteps": [
            "Reorientation to RAS and robust FOV cropping",
            "N4 bias field correction",
            "Conditional conformation of submillimeter input voxels to 1 mm",
            "Brain-mask estimation",
        ],
    }


def _tumor_volumetrics(seg_path: PathLike) -> Dict[str, Any]:
    """Compute per-region tumor volumes from the dseg NIfTI."""
    from pmriprep.config.constants import HDGLIO_LABEL_TO_NAME
    from pmriprep.utils.lesion_stats import load_validated_tumor_dseg

    seg_data, voxel_volume_mm3 = load_validated_tumor_dseg(seg_path)
    enh_count = int(np.sum(seg_data == 2))
    nenh_count = int(np.sum(seg_data == 1))
    wt_count = int(np.sum(seg_data > 0))
    enh_mm3 = enh_count * voxel_volume_mm3
    nenh_mm3 = nenh_count * voxel_volume_mm3
    wt_mm3 = wt_count * voxel_volume_mm3
    enh_ml = enh_mm3 / 1000.0
    nenh_ml = nenh_mm3 / 1000.0
    wt_ml = wt_mm3 / 1000.0

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
        **voxel_size_block(reference_voxel_source),
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
        sc["TumorSegQC"] = copy.deepcopy(qc_metrics)
        nawm = sc["TumorSegQC"].get("nawm")
        if isinstance(nawm, dict) and isinstance(nawm.get("n_voxels"), (int, float)):
            nawm["n_voxels"] = int(nawm["n_voxels"])
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

    source = source_paths[0] if source_paths else None
    dseg_source = _anat_derivative_uri(
        subject, session, source,
        "space-T1w_desc-tumorseg_dseg.nii.gz",
    )

    return {
        **get_bids_derivative_provenance(),
        "Description": description,
        "Type": "Lesion",
        "Method": method,
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or source,
        ),
        "Sources": [dseg_source],
        **voxel_size_block(reference_voxel_source),
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
    qc_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sidecar for the supratentorial NAWM normalization mask."""
    from pmriprep.utils.provenance import get_bids_derivative_provenance

    source = source_paths[0] if source_paths else None
    sources = [
        _anat_derivative_uri(
            subject, session, source,
            "space-T1w_desc-wholetumor_mask.nii.gz",
        ),
        _anat_derivative_uri(
            subject, session, source,
            "space-T1w_label-WM_probseg.nii.gz",
        ),
        _anat_derivative_uri(
            subject, session, source,
            "space-T1w_desc-brain_mask.nii.gz",
        ),
    ]
    nawm_qc = qc_metrics.get("nawm") if isinstance(qc_metrics, dict) else None
    atlas_used = nawm_qc.get("atlas_used") if isinstance(nawm_qc, dict) else None
    if atlas_used is True:
        sources.append(
            _anat_derivative_uri(
                subject, session, source,
                "space-T1w_desc-preproc_T1w.nii.gz",
            )
        )

    restriction = (
        "atlas-defined supratentorial restriction for every tumor location; "
        "tumor compartment from registered atlas overlap"
        if atlas_used is True
        else "hemisphere restriction because the supratentorial atlas was unavailable"
    )
    description = (
        "Supratentorial normal-appearing white matter mask for perfusion "
        "reference normalization"
        if atlas_used is True
        else "Hemisphere-restricted normal-appearing white matter candidate "
        "retained for QC because supratentorial atlas restriction was unavailable"
    )
    sidecar: Dict[str, Any] = {
        **get_bids_derivative_provenance(),
        "Description": description,
        "Type": "ROI",
        "Method": (
            "Brain-masked WM probability >= 0.9 with "
            f"{restriction}; hemisphere selection from the subject-space brain "
            "symmetry midline using a 60% tumor-laterality rule; 20 mm "
            "whole-tumor exclusion; and adaptive erosion"
        ),
        "SpatialReference": _anat_spatial_reference(
            subject, session, is_postcon_ref,
            reference_source or source,
        ),
        "Sources": sources,
        **voxel_size_block(reference_voxel_source),
    }
    if isinstance(nawm_qc, dict):
        sidecar["WMProbabilityThreshold"] = 0.9
        sidecar["TumorExclusionMarginMillimeters"] = 20.0
        if isinstance(atlas_used, bool):
            sidecar["AtlasRestrictionApplied"] = atlas_used
        erosion_voxels = nawm_qc.get("erosion_voxels_final")
        if isinstance(erosion_voxels, (int, float)):
            sidecar["ErosionVoxels"] = int(erosion_voxels)
        erosion_mm = nawm_qc.get("erosion_mm_final")
        if isinstance(erosion_mm, (int, float)):
            sidecar["ErosionMillimeters"] = float(erosion_mm)
        valid = nawm_qc.get("valid")
        if isinstance(valid, bool):
            sidecar["ValidForQuantification"] = valid
        n_voxels = nawm_qc.get("n_voxels")
        if isinstance(n_voxels, (int, float)):
            sidecar["VoxelCount"] = int(n_voxels)
        minimum_voxels = nawm_qc.get("minimum_required_voxels")
        if isinstance(minimum_voxels, (int, float)):
            sidecar["MinimumRequiredVoxels"] = int(minimum_voxels)
        exclusion_reason = nawm_qc.get("quantitative_use_exclusion_reason")
        if isinstance(exclusion_reason, str) and exclusion_reason:
            sidecar["QuantitativeUseExclusionReason"] = exclusion_reason
        string_fields = {
            "compartment_label": "TumorCompartment",
            "tumor_laterality": "TumorLaterality",
            "selected_hemisphere": "SelectedHemisphere",
            "hemisphere_selection_rule": "HemisphereSelectionRule",
            "laterality_method": "LateralityMethod",
            "nawm_compartment": "NAWMCompartment",
        }
        for source_key, output_key in string_fields.items():
            value = nawm_qc.get(source_key)
            if isinstance(value, str):
                sidecar[output_key] = value
        numeric_fields = {
            "f_supra": "SupratentorialOverlapFraction",
            "f_cereb": "CerebellarOverlapFraction",
            "f_other": "OtherOverlapFraction",
            "atlas_coverage": "AtlasCoverageFraction",
            "left_fraction": "TumorLeftFraction",
            "right_fraction": "TumorRightFraction",
            "midline_angle_degrees": "MidlineAngleDegrees",
            "symmetry_dice": "SymmetryDice",
        }
        for source_key, output_key in numeric_fields.items():
            value = nawm_qc.get(source_key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                sidecar[output_key] = float(value)
        warnings = nawm_qc.get("warnings")
        if isinstance(warnings, list) and all(isinstance(item, str) for item in warnings):
            sidecar["Warnings"] = warnings
    return sidecar


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


def build_dce_delay_metadata(
    qc_metrics: Optional[Dict[str, Any]], *, include_grid: bool = False,
) -> Dict[str, Any]:
    """Extract compact delay metadata from recorded DCE QC.

    Parameters
    ----------
    qc_metrics : dict or None
        Node QC or merged pipeline QC containing the executed delay settings.
    include_grid : bool, optional
        Include search limits and spacing for the selected-delay sidecar.

    Returns
    -------
    dict
        Recorded settings only; run-level diagnostics remain in PipelineQC.
    """
    qc = qc_metrics or {}
    delay = qc.get("delay_correction") or qc.get("pk_modeling", {}).get("delay_correction")
    if not isinstance(delay, dict):
        return {}
    metadata: Dict[str, Any] = {}
    if isinstance(delay.get("enabled"), bool):
        metadata["DelayCorrectionApplied"] = delay["enabled"]
    if include_grid:
        for source, target in (
            ("maximum_seconds", "MaximumDelaySeconds"),
            ("grid_step_seconds", "DelayStepSeconds"),
        ):
            if delay.get(source) is not None:
                metadata[target] = delay[source]
        metadata["MinimumDelaySeconds"] = 0.0
    return metadata


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
    source_paths : sequence of str or Path
        Source files recorded relative to ``bids_dir``.
    reference_voxel_source : str or Path
        NIfTI source of voxel-size metadata.
    bids_dir : str or Path
        Root of the source dataset.
    subject : str
        Participant identifier without the BIDS prefix.
    session : str or None
        Session identifier without the BIDS prefix, when available.
    provenance_dict : dict, optional
        Additional provenance fields merged into the sidecar.
    qc_metrics : dict, optional
        QC used to enrich perfusion provenance when ``minimal=False``.
    minimal : bool, optional
        Suppress perfusion-specific provenance enrichment and
        PK-input metadata (used for masks, structural underlays,
        motion-corrected 4D, T1map, B1+map, etc.).
    pk_dependent : bool, optional
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
        **voxel_size_block(reference_voxel_source),
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
        **voxel_size_block(reference_voxel_source),
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


def _scope_dsc_processing(sidecar: Dict[str, Any], map_key: str) -> None:
    """Keep only processing used by this DSC map, without changing its base.

    Parameters
    ----------
    sidecar : dict
        Map metadata to update in place.
    map_key : str
        Registered DSC map key; ``rel_cbv`` denotes smoothed AUC-CBV.
    """
    perfusion = sidecar.get("PerfusionProcessing")
    if not isinstance(perfusion, dict):
        return
    perfusion = dict(perfusion)
    deconvolution = perfusion.get("Deconvolution")
    if isinstance(deconvolution, dict):
        deconvolution = dict(deconvolution)
        integration = {
            key: deconvolution.pop(key)
            for key in (
                "ncbv_integration_rule", "cbv_auc_bat_frame", "cbv_auc_end_frame",
            )
            if key in deconvolution
        }
        uses_auc_cbv = map_key in {"cbv_auc", "ncbv", "rel_cbv"} or (
            map_key == "mtt"
            and sidecar.get("MTTMethod") == "full_bolus_auc_cbv_over_deconvolution_cbf"
        )
        if uses_auc_cbv and integration:
            perfusion["CBVIntegration"] = integration
        perfusion["Deconvolution"] = deconvolution
    if map_key in {
        "psr", "cnr", "tsnr", "ttp", "k2", "auc", "auc_corr",
        "cbv_auc", "ncbv", "rel_cbv",
    }:
        perfusion.pop("Deconvolution", None)
    if sidecar.get("AIFDependent") is False:
        perfusion.pop("AIF", None)
    if map_key in {"psr", "cnr", "tsnr", "ttp", "auc"}:
        perfusion.pop("LeakageCorrection", None)
        perfusion.pop("PreloadDetection", None)
    sidecar["PerfusionProcessing"] = perfusion


def build_dsc_map_sidecar(
    *,
    base: Dict[str, Any],
    description: str,
    map_key: str,
    qc_metrics: Optional[Dict[str, Any]] = None,
    units: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build DSC map metadata with provenance limited to its calculation."""
    from pmriprep.utils.aif_provenance import apply_aif_sidecar

    sidecar: Dict[str, Any] = {**base, "Description": description}
    if units is not None:
        sidecar["Units"] = units
    if extra:
        sidecar.update(extra)
    if map_key in {"ncbv", "ncbf"} and isinstance(qc_metrics, dict):
        factor_key = (
            "normalization_factor"
            if map_key == "ncbv"
            else "cbf_normalization_factor"
        )
        factor = qc_metrics.get(factor_key)
        if factor is not None:
            perfusion = sidecar.get("PerfusionProcessing")
            sidecar["PerfusionProcessing"] = {
                **(perfusion if isinstance(perfusion, dict) else {}),
                "NormalizationReference": {
                    "tissue": "NAWM",
                    "method": "median_finite_values_positive_median_required",
                    "factor": float(factor),
                    "parameter": "TissueAUC" if map_key == "ncbv" else "CBF",
                },
            }
    if map_key == "mtt" and isinstance(qc_metrics, dict):
        deconvolution = qc_metrics.get("deconvolution")
        method_metrics = (
            deconvolution if isinstance(deconvolution, dict) else qc_metrics
        )
        method = method_metrics.get("method", qc_metrics.get("method"))
        if method in {"sSVD", "oSVD", "tikhonov"}:
            sidecar["MTTMethod"] = (
                "full_bolus_auc_cbv_over_deconvolution_cbf"
            )
        elif method == "cpi":
            sidecar["MTTMethod"] = "fitted_residue_integral"
            if (method_metrics.get("n_fallback_osvd") or 0) > 0:
                sidecar["FallbackMTTMethod"] = (
                    "osvd_residue_integral_over_osvd_cbf"
                )
                sidecar["FallbackMask"] = "desc-dscFallback_mask (value 1)"
    if map_key in {"tmax", "arrival_delay"} and isinstance(qc_metrics, dict):
        deconvolution = qc_metrics.get("deconvolution")
        method_metrics = (
            deconvolution if isinstance(deconvolution, dict) else qc_metrics
        )
        if map_key == "arrival_delay":
            timing_method = method_metrics.get(
                "arrival_delay_method",
                qc_metrics.get("arrival_delay_method"),
            )
            if timing_method is not None:
                sidecar["ArrivalDelayMethod"] = str(timing_method)
            finite_voxels = method_metrics.get(
                "arrival_delay_finite_voxels",
                qc_metrics.get("arrival_delay_finite_voxels"),
            )
            if finite_voxels is not None:
                sidecar["ArrivalDelayFiniteVoxelCount"] = int(finite_voxels)
            finite_fraction = method_metrics.get(
                "arrival_delay_finite_fraction",
                qc_metrics.get("arrival_delay_finite_fraction"),
            )
            if finite_fraction is not None:
                sidecar["ArrivalDelayFiniteVoxelFraction"] = float(finite_fraction)
                sidecar["ArrivalDelayCoverageDenominator"] = "DSC analysis mask"
        else:
            tmax_method = method_metrics.get(
                "tmax_method",
                qc_metrics.get("tmax_method"),
            )
            if tmax_method is not None:
                sidecar["TmaxMethod"] = str(tmax_method)
            fallback_tmax_method = method_metrics.get(
                "fallback_tmax_method",
                qc_metrics.get("fallback_tmax_method"),
            )
            if fallback_tmax_method is not None:
                sidecar["FallbackTmaxMethod"] = str(fallback_tmax_method)
        at_bound_fraction = method_metrics.get(
            "delay_at_bound_fraction",
            qc_metrics.get("delay_at_bound_fraction"),
        )
        if at_bound_fraction is not None:
            sidecar["FittedDelayAtBoundFraction"] = float(at_bound_fraction)
            sidecar["FittedDelayAtBoundFractionDenominator"] = (
                "accepted fitted-delay parametric fits"
            )
    if map_key in {"cbv", "mtt"} and isinstance(
        qc_metrics, dict
    ):
        deconvolution = qc_metrics.get("deconvolution")
        method_metrics = (
            deconvolution if isinstance(deconvolution, dict) else qc_metrics
        )
        method = method_metrics.get("method", qc_metrics.get("method"))
        at_bound_fraction = method_metrics.get(
            "cpi_spacing_at_bound_fraction",
            qc_metrics.get("cpi_spacing_at_bound_fraction"),
        )
        if method == "cpi" and at_bound_fraction is not None:
            sidecar["CPIControlPointSpacingAtBoundFraction"] = float(
                at_bound_fraction
            )
            sidecar["CPIControlPointSpacingAtBoundFractionDenominator"] = (
                "accepted CPI fits with inferred control-point positions"
            )
    apply_aif_sidecar(sidecar, map_key, qc_metrics)
    _scope_dsc_processing(sidecar, map_key)
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
        **voxel_size_block(reference_voxel_source),
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
# Per-image anatomical sidecar enrichment
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

    Reads measured fields from ``qc_metrics["brain_extraction"]``.
    Volume keys explicitly identify mL or mm³.
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    be = qc_metrics.get("brain_extraction")
    if not be or not isinstance(be, dict):
        return sidecar
    if "tool" in be:
        sidecar["BrainExtractionTool"] = str(be["tool"])
    # Accept both fully qualified and short volume keys.
    vol_ml = be.get("brain_volume_ml", be.get("volume_ml"))
    if vol_ml is not None:
        sidecar["BrainExtractionVolumeMl"] = float(vol_ml)
    vol_mm3 = be.get("brain_volume_mm3", be.get("volume_mm3"))
    if vol_mm3 is not None:
        sidecar["BrainExtractionVolumeMm3"] = float(vol_mm3)
    return sidecar


def enrich_preproc_anat_sidecar(
    sidecar: Dict[str, Any],
    qc_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Add measured preprocessing settings to an anatomical sidecar.

    Reads the settings that production records in
    ``qc_metrics["preprocessing"]``. Registration and other anatomical
    operations are described by the derivative's ``ProcessingSteps``.
    """
    if not qc_metrics or not isinstance(qc_metrics, dict):
        return sidecar
    prep = qc_metrics.get("preprocessing")
    if prep and isinstance(prep, dict):
        if "denoising" in prep:
            sidecar["PreprocessingDenoising"] = bool(prep["denoising"])
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

    Raises
    ------
    ValueError
        If modality is neither ``DCE`` nor ``DSC``.
    """
    if modality not in {"DCE", "DSC"}:
        raise ValueError(f"modality must be 'DCE' or 'DSC'; got {modality!r}")

    # AIF curve content differs by modality:
    #   DSC: ΔR2* (1/s) — susceptibility-induced relaxation-rate change
    #   DCE: plasma [Gd] in mM
    if modality == "DSC":
        return (
            "delta_r2_star",
            "1/s",
            "Susceptibility-induced change in transverse relaxation rate "
            "(delta-R2*) used as the arterial input function.",
        )
    return (  # DCE
        "plasma_concentration_mM",
        "mM",
        "Plasma gadolinium concentration used as the arterial input function.",
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
    detection metadata in the full ``PerfusionProcessing.AIF`` block,
    also used by AIF masks. Quantitative maps carry a compact summary.

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
# DSC preprocessed-4D sidecar enrichment
# ---------------------------------------------------------------------------


def enrich_dsc_preproc_sidecar(
    sidecar: Dict[str, Any],
    *,
    preprocessing_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add series acquisition metadata; signal and motion QC live in PipelineQC."""
    p = preprocessing_stats or {}
    if "n_volumes" in p:
        sidecar["NumberOfVolumes"] = int(p["n_volumes"])
    if "tr_ms" in p:
        sidecar["RepetitionTime"] = float(p["tr_ms"]) / 1000.0
    return sidecar


def scope_preproc_processing(sidecar: Dict[str, Any]) -> None:
    """Keep acquisition context without duplicating the run's PipelineQC record."""
    perf = sidecar.get("PerfusionProcessing")
    if isinstance(perf, dict):
        duplicated = {
            "AIF", "Hematocrit", "AIFUnits", "BloodT1Reference", "T1Mapping",
            "SignalConversion", "PKModeling", "InferenceFallback", "Deconvolution",
            "LeakageCorrection", "PreloadDetection", "BaselineFrames", "QuantificationMode",
        }
        sidecar["PerfusionProcessing"] = {
            key: value for key, value in perf.items() if key not in duplicated
        }


# ---------------------------------------------------------------------------
# QC recommendation and warning enrichment
#
# Adds flat top-level ``QC*``-prefixed keys carrying the recommendation
# block and three categories of warnings to the canonical anchor
# sidecars (``desc-preproc_T1w.json``, ``desc-dscPreproc_dsc.json``,
# ``desc-dcePreproc_dce.json``).
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

    Raises
    ------
    ValueError
        If modality is not ``DCE``, ``DSC`` or ``ANAT``.
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
# DCE sidecar enrichment
#
# T1-map source/acceptance metadata and preprocessed-series acquisition fields.
# Run-level diagnostics stay in the anchor PipelineQC record.
# ---------------------------------------------------------------------------


def _dce_t1_cosmetic_keys(
    t1_qc: Optional[Dict[str, Any]],
    signal_qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe the retained T1 source and mapping outcome on the T1 sidecar."""
    if not isinstance(t1_qc, dict):
        t1_qc = {}
    out: Dict[str, Any] = {}
    method = t1_qc.get("method", t1_qc.get("t1_method"))
    if method is not None:
        out["T1MappingMethod"] = str(method)
    if "b1_correction" in t1_qc:
        out["T1B1Correction"] = str(t1_qc["b1_correction"])
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
    mapping_fallback_reason = (
        str(corr["reason"])
        if corr.get("correction_applied") == "fallback" and corr.get("reason")
        else None
    )
    support = t1_qc.get("t1_support")
    if isinstance(support, dict):
        out["T1Support"] = dict(support)

    signal = signal_qc if isinstance(signal_qc, dict) else {}
    signal_source = signal.get("t1_source")
    if signal_source == "assumed":
        fallback_used = method not in {None, "assumed"}
        mapping_attempted = method not in {None, "assumed", "external"}
        configured = signal.get("assumed_t1_policy") == "uniform_configured"
        source = "uniform_configured" if configured else "uniform_field_strength"
        out["T1Source"] = source + ("_fallback" if fallback_used else "_assumption")
        if signal.get("assumed_t1_policy") is not None:
            out["T1AssumptionPolicy"] = str(signal["assumed_t1_policy"])
        if signal.get("assumed_t1_field_strength_t") is not None:
            out["T1AssumptionFieldStrength"] = float(signal["assumed_t1_field_strength_t"])
        out["T1MappingAttempted"] = mapping_attempted
        if mapping_attempted:
            out["T1MappingAccepted"] = False
        if signal.get("assumed_t1_ms") is not None:
            out["T1AssumedValue"] = float(signal["assumed_t1_ms"])
            out["T1AssumedValueUnits"] = "ms"
        registration_qc = signal.get("vfa_to_dce_registration")
        if mapping_fallback_reason is not None:
            out["T1FallbackReason"] = mapping_fallback_reason
        if mapping_attempted and isinstance(registration_qc, dict):
            status = str(registration_qc.get("qc_status", "UNKNOWN")).upper()
            if status not in {"PASS", "WARN"} and "T1FallbackReason" not in out:
                out["T1FallbackReason"] = (
                    f"VFA-to-DCE registration QC was {status}"
                )
        if fallback_used and "T1FallbackReason" not in out:
            out["T1FallbackReason"] = (
                "Configured T1 source was unavailable during signal conversion"
            )
    elif signal_source == "map":
        out["T1Source"] = "measured_vfa"
        out["T1MappingAttempted"] = True
        out["T1MappingAccepted"] = True
    elif signal_source == "external":
        out["T1Source"] = "external_map"
        out["T1MappingAttempted"] = False
    else:
        out["T1Source"] = "not_established"
        out["T1MappingAttempted"] = method not in {None, "assumed", "external"}
        out["T1MappingAccepted"] = False
        out["T1FallbackReason"] = (
            "Signal-conversion T1 source was not recorded"
        )
    return out


def enrich_dce_preproc_sidecar(
    sidecar: Dict[str, Any],
    *,
    qc_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add acquisition metadata; applied conversion and QC remain in PipelineQC."""
    p = (qc_metrics or {}).get("preprocessing") or {}
    if "n_volumes" in p:
        sidecar["NumberOfVolumes"] = int(p["n_volumes"])
    if "volume_tr_s" in p:
        sidecar["RepetitionTime"] = float(p["volume_tr_s"])
        if "excitation_tr_ms" in p:
            sidecar["RepetitionTimeExcitation"] = float(p["excitation_tr_ms"]) / 1000.0
    return sidecar


def scope_dce_map_processing(sidecar: Dict[str, Any], map_key: str) -> None:
    """Restrict a DCE map's processing metadata to its inputs.

    Parameters
    ----------
    sidecar : dict
        Map sidecar to update without mutating its shared processing base.
    map_key : str
        Registered DCE map key, optionally carrying the smoothing suffix.
    """
    from pmriprep.utils.aif_provenance import stamp_aif_dependency

    key = map_key.removesuffix("_sm")
    stamp_aif_dependency(sidecar, key, modality="DCE")
    if sidecar.get("AIFDependent") is not False:
        return
    sidecar.pop("InputFunctionCompartment", None)
    perfusion = sidecar.get("PerfusionProcessing")
    if not isinstance(perfusion, dict):
        return
    perfusion = dict(perfusion)
    for field in (
        "AIF", "Hematocrit", "AIFUnits", "BloodT1Reference", "PKModeling",
        "InferenceFallback",
    ):
        perfusion.pop(field, None)
    if key in {"peak_enhancement", "ser", "tsnr", "mean"}:
        for field in ("T1Mapping", "SignalConversion", "Relaxivity"):
            perfusion.pop(field, None)
    sidecar["PerfusionProcessing"] = perfusion


def enrich_dce_pk_map_sidecar(
    sidecar: Dict[str, Any],
    *,
    qc_metrics: Optional[Dict[str, Any]] = None,
    map_key: str = "",
    lesion_fragment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Enrich a DCE per-PK-map sidecar with AIF provenance and cross-modality fields.

    Stamps the ``PerfusionProcessing.AIF`` block and ``AIFDependent``
    field via the shared AIF provenance module.
    Cross-modality fields land only on the maps they apply to.
    """
    if map_key == "ktrans":
        sidecar.update({
            "ValueDomain": "nonnegative volume transfer constant, or NaN",
            "BoundaryHandling": (
                "Zero Ktrans may be retained for Patlak fits; the QC mask "
                "excludes these lower-bound estimates. Other model families "
                "require positive Ktrans for retention"
            ),
        })
    if map_key == "kep":
        sidecar.update({
            "Estimator": "Ktrans / ve",
            "ValueDomain": "0 to 10 1/min, or NaN",
            "UndefinedValue": (
                "NaN when ve is not positive, the EES term is non-identifiable, "
                "or the derived rate exceeds the identifiable reporting range"
            ),
        })
    if map_key == "ve":
        sidecar.update({
            "ValueDomain": "positive volume fraction up to 1, or NaN",
            "BoundaryHandling": (
                "Zero ve is rejected. The upper endpoint may be retained; "
                "the QC mask separately screens endpoints, joint volume "
                "constraints and identification"
            ),
        })
    if map_key == "vp":
        sidecar.update({
            "ValueDomain": "plasma volume fraction from 0 to 1 inclusive, or NaN",
            "BoundaryHandling": (
                "Zero and upper-endpoint vp values may be retained; the QC "
                "mask separately screens endpoints, joint volume constraints "
                "and identification"
            ),
        })
    if map_key in {"fp", "ps"}:
        sidecar.update({
            "ValueDomain": "positive interior optimizer range, or NaN",
            "BoundaryHandling": (
                "Fits at either optimizer boundary are rejected rather than "
                f"published as ordinary {map_key} estimates"
            ),
        })
    if map_key == "e":
        sidecar.update({
            "Estimator": "PS / (Fp + PS)",
            "ValueDomain": "greater than 0 and less than 1, or NaN",
            "UndefinedValue": (
                "NaN when the fitted 2CXM parameter set is rejected, including "
                "zero-exchange and optimizer-boundary solutions"
            ),
        })
    if not qc_metrics or not isinstance(qc_metrics, dict):
        if lesion_fragment:
            sidecar["LesionStats"] = dict(lesion_fragment)
        return sidecar
    if map_key:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        apply_aif_sidecar(sidecar, map_key, qc_metrics, modality="DCE")
    sidecar.update(build_dce_delay_metadata(qc_metrics))
    pk_modeling = qc_metrics.get("pk_modeling")
    fallback = qc_metrics.get("inference_fallback")
    if fallback is None and isinstance(pk_modeling, dict):
        fallback = pk_modeling.get("inference_fallback")
    if isinstance(fallback, dict):
        sidecar.setdefault("PerfusionProcessing", {})["InferenceFallback"] = dict(
            fallback
        )
    if map_key == "vp":
        cbv = qc_metrics.get("cbv_constraint")
        if isinstance(cbv, dict) and "cbv_negative_fraction" in cbv:
            sidecar["CBVNegativeFraction"] = float(cbv["cbv_negative_fraction"])
    if lesion_fragment:
        sidecar["LesionStats"] = dict(lesion_fragment)
    return sidecar
