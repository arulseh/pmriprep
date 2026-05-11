"""Shared fixtures for OSIPI validation tests.

Provides path constants, NIfTI loaders, inter-session registration,
perf-plus reference value loading, and DRO ground truth loading.
"""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "debug" / "osipi"
ANAT_OUTPUT_ROOT = PROJECT_ROOT / "debug" / "osipi"

PERFPLUS_REFERENCE_JSON = (
    Path.home()
    / "Developer"
    / "perf-plus"
    / "osi_stuff"
    / "clinical_test_retest_analysis.json"
)

# DRO ground truth paths
OSIPI_GT_ROOT = PROJECT_ROOT / "sourcedata" / "osipi"

PERFPLUS_DRO_JSON = (
    Path.home()
    / "Developer"
    / "perf-plus"
    / "osi_stuff"
    / "osipi_validation.json"
)

PERFPLUS_DRO_BAYESIAN_JSON = (
    Path.home()
    / "Developer"
    / "perf-plus"
    / "osi_stuff"
    / "osipi_validation_bayesian.json"
)

# GT Ktrans stored in mM/min × 1000; multiply by this to get min⁻¹
GT_SCALE = 0.001

# Mapping from (subject, session) to OSIPI synthetic DRO file names
DRO_SESSION_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("sub-0091", "ses-0001"): ("Synthetic_P1_Visit1", "sub-0091_ses-0001"),
    ("sub-0091", "ses-0002"): ("Synthetic_P1_Visit2", "sub-0091_ses-0002"),
    ("sub-0092", "ses-0001"): ("Synthetic_P2_Visit1", "sub-0092_ses-0001"),
    ("sub-0092", "ses-0002"): ("Synthetic_P2_Visit2", "sub-0092_ses-0002"),
}

DRO_SESSIONS = list(DRO_SESSION_MAP.keys())

# File name patterns (BIDS-Common-Derivatives compliant).
# Native-space perf/ outputs omit the space- entity (perf/ directory
# implies native acquisition space). Cross-modality anat/ outputs
# (e.g., a brain mask describing anat in DCE space) keep space-dceref.
PERF_SPACE_PREFIX = ""
ANAT_SPACE_PREFIX = "space-dceref_"

# Map suffix patterns
MAP_SUFFIXES = {
    "Ktrans": "desc-dceKtrans_stat",
    "ve": "desc-dceVe_stat",
    "vp": "desc-dceVp_stat",
    "kep": "desc-dceKep_stat",
    "Rsquared": "desc-dceRsquared_stat",
    "IAUC60": "desc-dceIAUC60_stat",
    "IAUC90": "desc-dceIAUC90_stat",
    "IAUC120": "desc-dceIAUC120_stat",
    "SER": "desc-dceSER_stat",
    "TTP": "desc-dceTtp_stat",
    "tSNR": "desc-dceTsnr_stat",
}


# ---------------------------------------------------------------------------
# NIfTI loading
# ---------------------------------------------------------------------------


def load_map(
    subject: str,
    session: str,
    suffix: str,
) -> Optional[NDArray[np.floating]]:
    """Load a parameter map NIfTI file.

    Args:
        subject: Subject ID (e.g., "sub-0081").
        session: Session ID (e.g., "ses-0001").
        suffix: Map suffix key from MAP_SUFFIXES.

    Returns:
        3D numpy array of map data, or None if file not found.
    """
    import nibabel as nib

    map_suffix = MAP_SUFFIXES.get(suffix, suffix)
    fname = f"{subject}_{session}_{PERF_SPACE_PREFIX}{map_suffix}.nii.gz"
    fpath = OUTPUT_ROOT / subject / session / "perf" / fname

    if not fpath.exists():
        return None

    img = nib.load(fpath)
    return np.asarray(img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]


def load_mask(
    subject: str,
    session: str,
    mask_type: str = "brain",
) -> Optional[NDArray[np.floating]]:
    """Load a mask NIfTI file.

    Args:
        subject: Subject ID (e.g., "sub-0081").
        session: Session ID (e.g., "ses-0001").
        mask_type: One of "brain", "qc", "aif".

    Returns:
        3D numpy array of mask data, or None if file not found.
    """
    import nibabel as nib

    if mask_type == "brain":
        fname = f"{subject}_{session}_{ANAT_SPACE_PREFIX}desc-brain_mask.nii.gz"
        fpath = OUTPUT_ROOT / subject / session / "anat" / fname
    elif mask_type == "qc":
        fname = f"{subject}_{session}_{PERF_SPACE_PREFIX}desc-dceQc_mask.nii.gz"
        fpath = OUTPUT_ROOT / subject / session / "perf" / fname
    elif mask_type == "aif":
        fname = f"{subject}_{session}_{PERF_SPACE_PREFIX}desc-dceAif_mask.nii.gz"
        fpath = OUTPUT_ROOT / subject / session / "perf" / fname
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    if not fpath.exists():
        return None

    img = nib.load(fpath)
    return np.asarray(img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]


def load_ref_dce(
    subject: str,
    session: str,
) -> Optional[NDArray[np.floating]]:
    """Load the DCE reference image for registration.

    Args:
        subject: Subject ID.
        session: Session ID.

    Returns:
        3D numpy array, or None if not found.
    """
    import nibabel as nib

    fname = f"{subject}_{session}_{PERF_SPACE_PREFIX}desc-dceMean_stat.nii.gz"
    fpath = OUTPUT_ROOT / subject / session / "perf" / fname

    if not fpath.exists():
        return None

    img = nib.load(fpath)
    return np.asarray(img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]


def get_ref_dce_path(subject: str, session: str) -> Path:
    """Get path to DCE reference image."""
    fname = f"{subject}_{session}_{PERF_SPACE_PREFIX}desc-dceMean_stat.nii.gz"
    return OUTPUT_ROOT / subject / session / "perf" / fname


def get_map_path(subject: str, session: str, suffix: str) -> Path:
    """Get path to a parameter map file."""
    map_suffix = MAP_SUFFIXES.get(suffix, suffix)
    fname = f"{subject}_{session}_{PERF_SPACE_PREFIX}{map_suffix}.nii.gz"
    return OUTPUT_ROOT / subject / session / "perf" / fname


def get_mask_path(subject: str, session: str, mask_type: str) -> Path:
    """Get path to a mask file."""
    if mask_type == "brain":
        fname = f"{subject}_{session}_{ANAT_SPACE_PREFIX}desc-brain_mask.nii.gz"
        return OUTPUT_ROOT / subject / session / "anat" / fname
    elif mask_type == "qc":
        fname = f"{subject}_{session}_{PERF_SPACE_PREFIX}desc-dceQc_mask.nii.gz"
        return OUTPUT_ROOT / subject / session / "perf" / fname
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


# ---------------------------------------------------------------------------
# Perf-plus reference loading
# ---------------------------------------------------------------------------


def load_perfplus_reference() -> Optional[Dict]:
    """Load perf-plus clinical test-retest analysis JSON.

    Returns:
        Parsed JSON dict, or None if file not found.
    """
    if not PERFPLUS_REFERENCE_JSON.exists():
        return None

    with open(PERFPLUS_REFERENCE_JSON) as f:
        return json.load(f)


def get_perfplus_test_retest(subject: str) -> Optional[Dict]:
    """Get perf-plus test-retest metrics for a subject.

    Args:
        subject: Subject ID (e.g., "sub-0081").

    Returns:
        Dict with ses1_median, ses2_median, percent_difference,
        coefficient_of_variation. None if unavailable.
    """
    ref = load_perfplus_reference()
    if ref is None:
        return None

    return ref.get("test_retest", {}).get(subject)


def get_perfplus_session_stats(
    subject: str,
    session: str,
    parameter: str,
) -> Optional[Dict]:
    """Get perf-plus summary statistics for a subject/session/parameter.

    Args:
        subject: Subject ID (e.g., "sub-0081").
        session: Session ID (e.g., "ses-0001").
        parameter: Parameter name (e.g., "Ktrans").

    Returns:
        Dict with p5/p25/p50/p75/p95/mean/std/n_voxels. None if unavailable.
    """
    ref = load_perfplus_reference()
    if ref is None:
        return None

    return ref.get("subjects", {}).get(subject, {}).get(session, {}).get(parameter)


# ---------------------------------------------------------------------------
# Inter-session registration (Tier 2)
# ---------------------------------------------------------------------------

# Cache for registration results
_registration_cache: Dict[str, Tuple[str, str]] = {}


def has_fsl() -> bool:
    """Check if FSL FLIRT is available on the system."""
    return shutil.which("flirt") is not None


def _get_anat_path(subject: str, session: str) -> Path:
    """Get path to preprocessed T1w in anat reference space."""
    fname = f"{subject}_{session}_space-T1w_desc-preproc_T1w.nii.gz"
    return OUTPUT_ROOT / subject / session / "anat" / fname


def _get_dce_to_anat_xfm(subject: str, session: str) -> Path:
    """Get path to the pipeline's DCE-to-T1w FLIRT transform."""
    fname = f"{subject}_{session}_from-dceref_to-T1w_mode-image_xfm.mat"
    return OUTPUT_ROOT / subject / session / "anat" / fname


def _invert_xfm(mat_in: Path, mat_out: Path) -> bool:
    """Invert a FLIRT transform matrix."""
    cmd = ["convert_xfm", "-omat", str(mat_out), "-inverse", str(mat_in)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return mat_out.exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return False


def _concat_xfm(mat_a: Path, mat_b: Path, mat_out: Path) -> bool:
    """Concatenate two FLIRT transforms: mat_out = mat_b * mat_a.

    The resulting transform applies mat_a first, then mat_b.
    """
    cmd = [
        "convert_xfm", "-omat", str(mat_out),
        "-concat", str(mat_b), str(mat_a),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return mat_out.exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return False


def _brain_mask_dice(
    subject: str,
    ses_ref: str,
    ses_mov: str,
    mat_file: Path,
    out_dir: Path,
    tag: str,
) -> float:
    """Compute brain mask Dice for a given transform to evaluate quality."""
    import nibabel as nib

    brain1 = load_mask(subject, ses_ref, "brain")
    if brain1 is None:
        return 0.0

    brain2_in = get_mask_path(subject, ses_mov, "brain")
    brain2_out = out_dir / f"{subject}_brain_mask_{tag}_reg.nii.gz"
    ref_path = get_ref_dce_path(subject, ses_ref)

    result = apply_transform(
        brain2_in, ref_path, str(mat_file), brain2_out, "nearestneighbour"
    )
    if result is None:
        return 0.0

    brain2_img = nib.load(brain2_out)
    brain2 = np.asarray(brain2_img.get_fdata(), dtype=np.float64)

    bin1 = brain1 > 0.5
    bin2 = brain2 > 0.5
    intersection = int(np.sum(bin1 & bin2))
    total = int(np.sum(bin1)) + int(np.sum(bin2))
    return 2.0 * intersection / total if total > 0 else 0.0


def register_sessions(
    subject: str,
    ses_ref: str = "ses-0001",
    ses_mov: str = "ses-0002",
) -> Optional[Tuple[str, str]]:
    """Register ses_mov DCE space to ses_ref DCE space.

    Tries two approaches and picks the one with better brain mask overlap:

    1. **Direct**: FLIRT DCE-ref → DCE-ref (works well when slabs overlap
       sufficiently).
    2. **Anat-mediated**: Chain DCE→T1w→T1w→DCE using the pipeline's
       DCE-to-T1w transforms plus an inter-session T1w registration.
       More robust for large positioning differences, but the 3-transform
       chain can accumulate interpolation error.

    Args:
        subject: Subject ID (e.g., "sub-0081").
        ses_ref: Reference session ID.
        ses_mov: Moving session ID.

    Returns:
        Tuple of (transform_mat_path, output_dir) or None if registration
        fails or FSL is unavailable.
    """
    cache_key = f"{subject}_{ses_ref}_{ses_mov}"
    if cache_key in _registration_cache:
        return _registration_cache[cache_key]

    if not has_fsl():
        return None

    # Create output directory
    out_dir = (
        PROJECT_ROOT / "test_outputs" / "pmriprep" / subject
        / "test_retest_qc"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list = []  # list of (dice, mat_path_str)

    # --- Approach 1: Direct DCE-to-DCE registration ---
    ref_path = get_ref_dce_path(subject, ses_ref)
    mov_path = get_ref_dce_path(subject, ses_mov)
    direct_mat = out_dir / f"{ses_mov}_to_{ses_ref}_direct.mat"

    if ref_path.exists() and mov_path.exists():
        cmd = [
            "flirt",
            "-in", str(mov_path),
            "-ref", str(ref_path),
            "-omat", str(direct_mat),
            "-dof", "6",
        ]
        try:
            subprocess.run(
                cmd, check=True, capture_output=True, timeout=120
            )
            if direct_mat.exists():
                dice = _brain_mask_dice(
                    subject, ses_ref, ses_mov, direct_mat, out_dir, "direct"
                )
                candidates.append((dice, direct_mat))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            pass

    # --- Approach 2: Anat-mediated registration ---
    xfm_dce2anat_mov = _get_dce_to_anat_xfm(subject, ses_mov)
    xfm_dce2anat_ref = _get_dce_to_anat_xfm(subject, ses_ref)
    t1_ref = _get_anat_path(subject, ses_ref)
    t1_mov = _get_anat_path(subject, ses_mov)
    anat_mat_path = out_dir / f"{ses_mov}_to_{ses_ref}_anat.mat"

    if (xfm_dce2anat_mov.exists() and xfm_dce2anat_ref.exists()
            and t1_ref.exists() and t1_mov.exists()):
        # Step 1: T1(ses_mov) → T1(ses_ref)
        t1_mat = out_dir / f"{ses_mov}_anat_to_{ses_ref}_anat.mat"
        cmd = [
            "flirt",
            "-in", str(t1_mov),
            "-ref", str(t1_ref),
            "-omat", str(t1_mat),
            "-dof", "6",
        ]
        try:
            subprocess.run(
                cmd, check=True, capture_output=True, timeout=180
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            t1_mat = None

        if t1_mat is not None and t1_mat.exists():
            # Step 2: Invert DCE→anat(ses_ref) to get anat→DCE(ses_ref)
            xfm_anat2dce_ref = out_dir / f"{ses_ref}_anat_to_DCE.mat"
            ok = _invert_xfm(xfm_dce2anat_ref, xfm_anat2dce_ref)

            if ok:
                # Step 3: Chain DCE(mov) → anat(mov) → anat(ref) → DCE(ref)
                intermediate = (
                    out_dir / f"{ses_mov}_DCE_to_{ses_ref}_anat.mat"
                )
                ok = _concat_xfm(
                    xfm_dce2anat_mov, t1_mat, intermediate
                )
                if ok:
                    ok = _concat_xfm(
                        intermediate, xfm_anat2dce_ref, anat_mat_path
                    )
                    if ok:
                        dice = _brain_mask_dice(
                            subject, ses_ref, ses_mov,
                            anat_mat_path, out_dir, "anat",
                        )
                        candidates.append((dice, anat_mat_path))

    if not candidates:
        return None

    # Pick the transform with the best brain mask overlap
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_dice, best_mat = candidates[0]

    # Copy winner to canonical name
    mat_path = out_dir / f"{ses_mov}_to_{ses_ref}.mat"
    if best_mat != mat_path:
        import shutil as _shutil
        _shutil.copy2(str(best_mat), str(mat_path))

    result = (str(mat_path), str(out_dir))
    _registration_cache[cache_key] = result
    return result


def apply_transform(
    in_file: Path,
    ref_file: Path,
    mat_file: str,
    out_file: Path,
    interp: str = "trilinear",
) -> Optional[Path]:
    """Apply a FLIRT transform to a NIfTI file.

    Args:
        in_file: Path to input image.
        ref_file: Path to reference image (defines output grid).
        mat_file: Path to FLIRT .mat transform.
        out_file: Path for output image.
        interp: Interpolation method ("trilinear" or "nearestneighbour").

    Returns:
        Path to output file, or None if transform fails.
    """
    if not in_file.exists():
        return None

    cmd = [
        "flirt",
        "-in", str(in_file),
        "-ref", str(ref_file),
        "-applyxfm",
        "-init", str(mat_file),
        "-out", str(out_file),
        "-interp", interp,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None

    return out_file if out_file.exists() else None


def compute_joint_mask(
    subject: str,
    mat_file: str,
    ses_ref: str = "ses-0001",
    ses_mov: str = "ses-0002",
    min_voxels: int = 500,
    require_qc: bool = False,
) -> Optional[NDArray[np.bool_]]:
    """Compute joint mask across registered sessions.

    By default, uses the intersection of brain masks only. With
    ``require_qc=True``, also requires QC mask overlap (much more
    restrictive, especially with model-auto per-voxel model selection).

    Args:
        subject: Subject ID.
        mat_file: Path to FLIRT transform (ses_mov → ses_ref).
        ses_ref: Reference session.
        ses_mov: Moving session.
        min_voxels: Minimum required voxels in joint mask.
        require_qc: If True, also intersect with QC masks from both
            sessions. Default False.

    Returns:
        Boolean 3D array, or None if insufficient voxels or files missing.
    """
    import nibabel as nib

    # Load reference session masks
    brain1 = load_mask(subject, ses_ref, "brain")
    if brain1 is None:
        return None

    # Register moving session brain mask
    ref_path = get_ref_dce_path(subject, ses_ref)
    out_dir = Path(mat_file).parent

    brain2_in = get_mask_path(subject, ses_mov, "brain")
    brain2_out = out_dir / f"{subject}_{ses_mov}_brain_mask_reg.nii.gz"
    result = apply_transform(brain2_in, ref_path, mat_file, brain2_out, "nearestneighbour")
    if result is None:
        return None
    brain2_img = nib.load(brain2_out)
    brain2 = np.asarray(brain2_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

    joint = (brain1 > 0.5) & (brain2 > 0.5)

    if require_qc:
        qc1 = load_mask(subject, ses_ref, "qc")
        if qc1 is None:
            return None

        qc2_in = get_mask_path(subject, ses_mov, "qc")
        qc2_out = out_dir / f"{subject}_{ses_mov}_qc_mask_reg.nii.gz"
        result = apply_transform(qc2_in, ref_path, mat_file, qc2_out, "nearestneighbour")
        if result is None:
            return None
        qc2_img = nib.load(qc2_out)
        qc2 = np.asarray(qc2_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

        joint = joint & (qc1 > 0.5) & (qc2 > 0.5)

    if int(np.sum(joint)) < min_voxels:
        return None

    return joint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SES1 = "ses-0001"
SES2 = "ses-0002"

# OSIPI clinical test-retest cohort: sub-0081 through sub-0088
TEST_RETEST_SUBJECTS = [f"sub-00{i}" for i in range(81, 89)]


def _outputs_exist(subject: str, session: str) -> bool:
    """Check if pmriprep outputs exist for a subject/session."""
    ktrans_path = get_map_path(subject, session, "Ktrans")
    return ktrans_path.exists()


def has_perfplus_reference(subject: str) -> bool:
    """Check if perf-plus test-retest reference exists for a subject."""
    ref = load_perfplus_reference()
    if ref is None:
        return False
    return subject in ref.get("test_retest", {})


skip_no_fsl = pytest.mark.skipif(
    not has_fsl(),
    reason="FSL FLIRT not available",
)


# ---------------------------------------------------------------------------
# DRO ground truth loading
# ---------------------------------------------------------------------------


def load_ground_truth_ktrans(
    subject: str,
    session: str,
) -> Optional[NDArray[np.floating]]:
    """Load DRO ground truth Ktrans, resampled to pmriprep output space.

    The GT mask may have a different affine (e.g., LAS vs RAS). This
    function resamples the GT Ktrans to match the pmriprep output grid
    and applies GT_SCALE to convert to min⁻¹.

    Args:
        subject: Subject ID (e.g., "sub-0091").
        session: Session ID (e.g., "ses-0001").

    Returns:
        3D numpy array of Ktrans in min⁻¹, or None if files not found.
    """
    import nibabel as nib

    key = (subject, session)
    if key not in DRO_SESSION_MAP:
        return None

    dro_prefix, gt_dir_name = DRO_SESSION_MAP[key]
    gt_dir = OSIPI_GT_ROOT / gt_dir_name
    gt_path = gt_dir / f"{dro_prefix}_Ktrans_aligned_from_dicom.nii"

    if not gt_path.exists():
        return None

    gt_img = nib.load(gt_path)
    gt_data = np.asarray(gt_img.get_fdata(), dtype=np.float64) * GT_SCALE  # type: ignore[attr-defined]

    return gt_data


def load_tumor_mask(
    subject: str,
    session: str,
) -> Optional[NDArray[np.bool_]]:
    """Load DRO tumor mask using direct voxel indexing.

    The GT tumor mask NIfTI has a different affine (LAS) than the GT
    Ktrans and pmriprep outputs (RAS), but shares the same voxel grid.
    Since both originate from the same synthetic DRO, direct voxel
    indexing is correct — the affine difference is an artifact of the
    DICOM conversion, not a real spatial offset.

    Args:
        subject: Subject ID (e.g., "sub-0091").
        session: Session ID (e.g., "ses-0001").

    Returns:
        Boolean 3D array, or None if files not found.
    """
    import nibabel as nib

    key = (subject, session)
    if key not in DRO_SESSION_MAP:
        return None

    dro_prefix, gt_dir_name = DRO_SESSION_MAP[key]
    gt_dir = OSIPI_GT_ROOT / gt_dir_name
    mask_path = gt_dir / f"{dro_prefix}_TumorMask.nii"

    if not mask_path.exists():
        return None

    mask_img = nib.load(mask_path)
    mask_data = np.asarray(mask_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

    return mask_data > 0.5


def load_perfplus_dro_reference() -> Optional[Dict]:
    """Load perf-plus DRO validation JSON (non-Bayesian).

    Returns:
        Parsed JSON dict, or None if file not found.
    """
    if not PERFPLUS_DRO_JSON.exists():
        return None

    with open(PERFPLUS_DRO_JSON) as f:
        return json.load(f)


def load_perfplus_dro_bayesian() -> Optional[Dict]:
    """Load perf-plus DRO Bayesian validation JSON.

    Returns:
        Parsed JSON dict, or None if file not found.
    """
    if not PERFPLUS_DRO_BAYESIAN_JSON.exists():
        return None

    with open(PERFPLUS_DRO_BAYESIAN_JSON) as f:
        return json.load(f)


def get_perfplus_dro_session(
    subject: str,
    session: str,
    bayesian: bool = False,
) -> Optional[Dict]:
    """Get perf-plus DRO stats for a specific session.

    Args:
        subject: Subject ID (e.g., "sub-0091").
        session: Session ID (e.g., "ses-0001").
        bayesian: If True, load from Bayesian JSON.

    Returns:
        Dict with ground_truth, calculated, comparison dicts.
        None if unavailable.
    """
    ref = load_perfplus_dro_bayesian() if bayesian else load_perfplus_dro_reference()
    if ref is None:
        return None

    session_key = f"{subject}_{session}"
    return ref.get("subjects", {}).get(session_key)


def load_ci_map(
    subject: str,
    session: str,
    bound: str,
    parameter: str = "Ktrans",
) -> Optional[NDArray[np.floating]]:
    """Load a Bayesian confidence interval bound map.

    Args:
        subject: Subject ID.
        session: Session ID.
        bound: "lower" or "upper".
        parameter: Parameter name (e.g., "Ktrans").

    Returns:
        3D numpy array, or None if file not found.
    """
    import nibabel as nib

    suffix = f"desc-dceCI95{bound}{parameter[:1].upper()}{parameter[1:]}_stat"
    fname = f"{subject}_{session}_{PERF_SPACE_PREFIX}{suffix}.nii.gz"
    fpath = OUTPUT_ROOT / subject / session / "perf" / fname

    if not fpath.exists():
        return None

    img = nib.load(fpath)
    return np.asarray(img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]


# DRO skip markers

def _dro_outputs_exist() -> bool:
    """Check if at least one DRO session has pmriprep outputs."""
    return _outputs_exist("sub-0091", "ses-0001")


def _dro_gt_exists() -> bool:
    """Check if DRO ground truth directory exists."""
    return OSIPI_GT_ROOT.exists() and (OSIPI_GT_ROOT / "sub-0091_ses-0001").exists()


skip_no_dro_outputs = pytest.mark.skipif(
    not _dro_outputs_exist(),
    reason="PMRIPrep DRO outputs not found for sub-0091 ses-0001",
)

skip_no_dro_gt = pytest.mark.skipif(
    not _dro_gt_exists(),
    reason="OSIPI DRO ground truth directory not found",
)

skip_no_perfplus_dro = pytest.mark.skipif(
    not PERFPLUS_DRO_JSON.exists(),
    reason="Perf-plus DRO validation JSON not found",
)

# OSIPI DSC DRO data path
OSIPI_DSC_DRO_CSV = OSIPI_GT_ROOT / "dsc_data.csv"

# OSIPI ground truth AIF path
OSIPI_AIF_PATH = OSIPI_GT_ROOT / "AIF.npz"

skip_no_osipi_aif = pytest.mark.skipif(
    not OSIPI_AIF_PATH.exists(),
    reason="OSIPI GT AIF file not found (sourcedata/osipi/AIF.npz)",
)


def load_osipi_aif() -> Optional[NDArray[np.floating]]:
    """Load OSIPI ground truth AIF (plasma concentration in mM).

    Returns the Cp array from ``sourcedata/osipi/AIF.npz``. Values may
    include small negative baseline values — these are NOT clipped, as
    they represent noise in the pre-contrast baseline.

    Returns:
        1D array of plasma concentration (mM), or None if file not found.
    """
    if not OSIPI_AIF_PATH.exists():
        return None

    npz = np.load(str(OSIPI_AIF_PATH))
    return np.asarray(npz["Cp"], dtype=np.float64)


# ---------------------------------------------------------------------------
# OSIPI DSC DRO loading
# ---------------------------------------------------------------------------


@dataclass
class OSIPIDSCCase:
    """A single OSIPI DSC DRO test case.

    Attributes:
        label: Test case label from CSV.
        c_tis: Tissue concentration time course (1D).
        c_aif: AIF concentration time course (1D).
        cbv: Ground truth CBV in ml/100ml.
        cbf: Ground truth CBF in ml/100ml/min.
        tr: Repetition time in seconds.
    """

    label: str
    c_tis: NDArray[np.floating]
    c_aif: NDArray[np.floating]
    cbv: float
    cbf: float
    tr: float


def load_osipi_dsc_dro() -> Optional[list]:
    """Load OSIPI DSC DRO test cases from CSV.

    Parses ``sourcedata/osipi/dsc_data.csv`` (from OSIPI Code Collection
    commit a55b83d8, 2022-04-21). Contains 14 concentration-domain test
    cases with known CBV/CBF ground truth and gamma-variate residue (λ=3).

    Returns:
        List of OSIPIDSCCase, or None if CSV not found.
    """
    import csv

    if not OSIPI_DSC_DRO_CSV.exists():
        return None

    cases = []
    with open(OSIPI_DSC_DRO_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            c_tis = np.array(
                [float(x) for x in row["C_tis"].strip().split()],
                dtype=np.float64,
            )
            c_aif = np.array(
                [float(x) for x in row["C_aif"].strip().split()],
                dtype=np.float64,
            )
            cases.append(
                OSIPIDSCCase(
                    label=row["label"].strip(),
                    c_tis=c_tis,
                    c_aif=c_aif,
                    cbv=float(row["cbv"]),
                    cbf=float(row["cbf"]),
                    tr=float(row["tr"]),
                )
            )
    return cases


skip_no_osipi_dsc_dro = pytest.mark.skipif(
    not OSIPI_DSC_DRO_CSV.exists(),
    reason="OSIPI DSC DRO CSV not found (sourcedata/osipi/dsc_data.csv)",
)
