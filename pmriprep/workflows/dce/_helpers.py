"""Serializable function-node helpers for the DCE preprocessing workflow.

Each helper imports its runtime dependencies locally so Nipype can execute the
source captured by ``niu.Function`` in an isolated worker.
"""

from typing import Any

__all__ = [
    "_build_model_free_map_dict",
    "_build_pk_map_dict",
    "_collect_qc_metrics",
    "_compute_early_mean",
    "_create_aif_search_mask",
    "_create_dce_registration_filmstrip",
    "_dsc_dce_validation_node",
    "_dsc_dce_xval_plot_node",
    "_generate_dce_motion_qc_plot",
    "_generate_dce_pk_fit_qc",
    "_generate_dce_pk_maps_qc",
    "_generate_preprocessing_qc",
    "_normalize_by_nawm",
    "_resample_dsc_cbv_to_dce",
    "_resample_dsc_ncbv_to_dce",
    "_signal_to_concentration_node",
    "_derive_smoothed_kep",
    "_smooth_dce_maps",
    "_transform_atlas_cascade_to_dce",
    "_transform_atlas_to_dce",
    "_transform_b1_to_dce",
    "_transform_b1_to_vfa",
    "_transform_external_t1_to_dce",
    "_transform_lesion_mask_to_dce",
    "_transform_mask_to_dce",
    "_transform_mask_to_dce_conditional",
    "_transform_probseg_to_dce",
    "_transform_structural_to_dce",
    "_transform_t1_to_dce",
]
def _transform_mask_to_dce(
    mask_file: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
) -> str:
    """Transform brain mask from T1w space to DCE space.

    Thin Nipype Function-node wrapper around
    ``pmriprep.utils.registration.transform_mask_to_grid``. The import is
    intentionally inline because ``niu.Function`` serialises this body
    via ``inspect.getsource`` and runs it in a worker that cannot see
    module-level imports.

    Parameters
    ----------
    mask_file : str or None
        Path to brain mask in T1w space (or None).
    reference_file : str or None
        Path to DCE reference image for output geometry.
    transform_matrix : str or None
        Path to T1w→DCE FLIRT matrix.

    Returns
    -------
    str
        Path to transformed brain mask in DCE space, or "" if no mask.
    """
    from pmriprep.utils.registration import transform_mask_to_grid

    return transform_mask_to_grid(mask_file, reference_file, transform_matrix)


def _transform_mask_to_dce_conditional(
    mask_file: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
    needs_transform: bool = True,
) -> str:
    """Transform mask to DCE space, or pass through if already native.

    When ``needs_transform`` is True (tumor-seg mask in T1w space), delegates
    to ``transform_mask_to_grid``.  When False (user-supplied mask already in
    DCE native space), validates geometry and returns the input path.

    Parameters
    ----------
    mask_file : str or None
        Path to mask (or None).
    reference_file : str or None
        Path to DCE reference image for output geometry.
    transform_matrix : str or None
        Path to T1w→DCE FLIRT matrix (only used when needs_transform=True).
    needs_transform : bool
        If False, skip FLIRT and return input path as-is.

    Returns
    -------
    str
        Path to mask in DCE space, or "" if no mask.
    """
    from nipype.interfaces.base import isdefined

    if not isdefined(mask_file) or mask_file is None:
        return ""

    from pathlib import Path

    if not Path(mask_file).is_file():
        return ""

    if not needs_transform:
        import nibabel as nib
        import numpy as np

        mask_img = nib.load(str(mask_file))
        ref_img = nib.load(str(reference_file))
        if mask_img.shape[:3] != ref_img.shape[:3]:  # type: ignore[attr-defined]
            raise ValueError(
                f"User-supplied mask shape {mask_img.shape[:3]} does not "  # type: ignore[attr-defined]
                f"match DCE volume shape {ref_img.shape[:3]}. "  # type: ignore[attr-defined]
                f"Masks under [dce] config must be in DCE native space."
            )
        affine_diff = float(
            np.abs(mask_img.affine - ref_img.affine).max()  # type: ignore[attr-defined]
        )
        if affine_diff > 1.0:
            raise ValueError(
                f"User-supplied mask affine differs from DCE volume by "
                f"{affine_diff:.2f} (max element). "
                f"Masks under [dce] config must be in DCE native space."
            )
        return str(mask_file)

    from pmriprep.utils.registration import transform_mask_to_grid

    return transform_mask_to_grid(mask_file, reference_file, transform_matrix)


def _transform_probseg_to_dce(
    probseg_file: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
) -> str:
    """Transform a probability segmentation map from T1w space to DCE space.

    Uses FSL FLIRT with trilinear interpolation (appropriate for continuous
    probability maps, unlike nearest-neighbour for binary masks). Clips output
    to [0, 1] to correct slight over/undershoot from interpolation.

    Parameters
    ----------
    probseg_file : str or None
        Path to probability map in T1w space (or None).
    reference_file : str or None
        Path to DCE reference image for output geometry.
    transform_matrix : str or None
        Path to T1w→DCE FLIRT matrix.

    Returns
    -------
    str
        Path to transformed probability map in DCE space, or "" if no input
        (the empty-string convention required by the downstream probseg traits).
    """
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(probseg_file) or probseg_file is None:
        return ""
    if not isdefined(reference_file) or reference_file is None:
        return ""
    if not isdefined(transform_matrix) or transform_matrix is None:
        return ""

    from pathlib import Path
    import subprocess

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np


    probseg_path = Path(probseg_file)
    if not probseg_path.is_file():
        return ""

    out_dir = Path.cwd()
    # Derive output name from input (e.g., *_pve_0 → csf, *_pve_1 → gm, *_pve_2 → wm)
    stem = probseg_path.name.replace(".nii.gz", "").replace(".nii", "")
    out_file = str(out_dir / f"{stem}_dce.nii.gz")

    # Same-space shortcut: skip transform if already aligned
    probseg_img = nib.load(probseg_file)
    ref_img = nib.load(reference_file)
    if probseg_img.shape[:3] == ref_img.shape[:3]:  # type: ignore[attr-defined]
        affine_diff = float(np.abs(probseg_img.affine - ref_img.affine).max())  # type: ignore[attr-defined]
        if affine_diff < 1e-4:
            # Already in same space, just copy (clip to [0, 1])
            data = probseg_img.get_fdata()  # type: ignore[attr-defined]
            data = np.clip(data, 0.0, 1.0).astype(np.float32)
            out_img = nib.Nifti1Image(data, ref_img.affine)  # type: ignore[attr-defined]
            nib.save(out_img, out_file)
            return out_file

    # Transform with FLIRT using trilinear interpolation
    tmp_file = str(out_dir / f"_tmp_{stem}_dce.nii.gz")
    cmd = [
        "flirt",
        "-in", str(probseg_file),
        "-ref", str(reference_file),
        "-out", tmp_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "trilinear",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Clip to [0, 1] — FLIRT trilinear can produce slight over/undershoot
    tmp_img = nib.load(tmp_file)
    data = tmp_img.get_fdata()  # type: ignore[attr-defined]
    data = np.clip(data, 0.0, 1.0).astype(np.float32)
    out_img = nib.Nifti1Image(data, tmp_img.affine)  # type: ignore[attr-defined]
    nib.save(out_img, out_file)

    # Clean up temp file
    try:
        Path(tmp_file).unlink(missing_ok=True)
    except Exception:
        pass

    return out_file


def _transform_lesion_mask_to_dce(
    lesion_mask_file: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
) -> str:
    """Transform lesion mask from T1w space to DCE space.

    Uses FSL FLIRT with nearest-neighbour interpolation to resample
    the lesion mask into DCE space, preserving integer label values.

    Parameters
    ----------
    lesion_mask_file : str or None
        Path to lesion mask in T1w space (or None).
    reference_file : str or None
        Path to DCE reference image for output geometry.
    transform_matrix : str or None
        Path to T1w→DCE FLIRT matrix.

    Returns
    -------
    str
        Path to transformed lesion mask in DCE space, or "" if no mask.
        Returns "" (not None) because downstream traits.Str rejects None.
    """
    from nipype.interfaces.base import isdefined

    if not isdefined(lesion_mask_file) or lesion_mask_file is None:
        return ""
    if not isdefined(reference_file) or reference_file is None:
        return ""
    if not isdefined(transform_matrix) or transform_matrix is None:
        return ""

    from pathlib import Path
    import subprocess

    lesion_path = Path(lesion_mask_file)
    if not lesion_path.is_file():
        return ""

    out_dir = Path.cwd()
    out_file = str(out_dir / "lesion_mask_dce.nii.gz")

    cmd = [
        "flirt",
        "-in", str(lesion_mask_file),
        "-ref", str(reference_file),
        "-out", out_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "nearestneighbour",  # Preserve integer labels
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    return out_file


def _create_aif_search_mask(
    t1w_mask_in_dce: str | None = None,
    dce_native_mask: str | None = None,
) -> str | None:
    """Create a conservative AIF search mask.

    Creates an intersection of T1w-transformed mask with the native DCE
    brain mask (from HD-BET or BET) to exclude non-brain regions that may
    be incorrectly included in one mask but not the other.

    The intersection approach ensures:
    1. Regions outside T1w brain coverage are excluded
    2. Regions outside DCE brain extraction are excluded
    3. Only voxels in both masks are used for AIF search

    Parameters
    ----------
    t1w_mask_in_dce : str or None
        Path to T1w brain mask transformed to DCE space.
    dce_native_mask : str or None
        Path to native DCE brain mask from DCEBrainExtract.

    Returns
    -------
    str or None
        Path to conservative AIF search mask, or None if inputs missing.
    """
    if not t1w_mask_in_dce or t1w_mask_in_dce is None:
        # If no T1w mask, fall back to native DCE mask
        return dce_native_mask

    if not dce_native_mask or dce_native_mask is None:
        # If no native DCE mask, fall back to T1w mask
        return t1w_mask_in_dce

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    from scipy.ndimage import binary_fill_holes  # type: ignore[import-untyped]

    t1w_mask_path = Path(t1w_mask_in_dce)
    dce_mask_path = Path(dce_native_mask)

    if not t1w_mask_path.is_file():
        return str(dce_native_mask)
    if not dce_mask_path.is_file():
        return str(t1w_mask_in_dce)

    # Load masks
    t1w_mask_img = nib.load(t1w_mask_in_dce)
    t1w_mask_data = t1w_mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    dce_mask_img = nib.load(dce_native_mask)
    dce_mask_data = dce_mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    # Intersect masks - conservative approach
    aif_mask = np.logical_and(t1w_mask_data, dce_mask_data).astype(np.int16)

    # Fill holes in the intersection
    aif_mask = binary_fill_holes(aif_mask).astype(np.int16)

    # Save
    out_file = str(Path.cwd() / "aif_search_mask.nii.gz")
    out_img = nib.Nifti1Image(aif_mask, t1w_mask_img.affine)  # type: ignore[attr-defined]
    nib.save(out_img, out_file)

    return out_file


def _transform_atlas_to_dce(
    atlas_file: str | None = None,
    t1w_to_mni_xfm: str | None = None,
    t1w_to_dce_xfm: str | None = None,
    reference_file: str | None = None,
) -> str | None:
    """Transform an MNI atlas mask to DCE space via T1w.

    Creates a concatenated transform MNI→T1w→DCE and applies it to the
    atlas mask. Includes morphological refinement (dilate-dilate-erode)
    to clean up edges after interpolation.

    Parameters
    ----------
    atlas_file : str, optional
        Path to MNI-space atlas mask (e.g., MCA/ACA territories).
    t1w_to_mni_xfm : str, optional
        Path to T1w→MNI FLIRT matrix (from anat pipeline).
    t1w_to_dce_xfm : str, optional
        Path to T1w→DCE FLIRT matrix (from DCE registration).
    reference_file : str, optional
        Path to DCE reference image for output geometry.

    Returns
    -------
    str or None
        Path to atlas mask in DCE space, or None if inputs missing.
    """
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(atlas_file) or atlas_file is None:
        return None
    if not isdefined(t1w_to_mni_xfm) or t1w_to_mni_xfm is None:
        return None
    if not isdefined(t1w_to_dce_xfm) or t1w_to_dce_xfm is None:
        return None
    if not isdefined(reference_file) or reference_file is None:
        return None

    from pathlib import Path
    import subprocess

    atlas_path = Path(atlas_file)
    if not atlas_path.is_file():
        return None

    out_dir = Path.cwd()
    # Use atlas-specific temp filenames to avoid collisions in cascade mode
    atlas_stem = atlas_path.name.replace(".nii.gz", "").replace(".nii", "")

    # Step 1: Invert T1w→MNI to get MNI→T1w
    mni_to_t1w_xfm = str(out_dir / f"_tmp_{atlas_stem}_mni_to_t1w.mat")
    invert_cmd = [
        "convert_xfm", "-omat", mni_to_t1w_xfm,
        "-inverse", str(t1w_to_mni_xfm),
    ]
    result = subprocess.run(invert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    # Step 2: Concatenate MNI→T1w + T1w→DCE = MNI→DCE
    mni_to_dce_xfm = str(out_dir / f"_tmp_{atlas_stem}_mni_to_dce.mat")
    concat_cmd = [
        "convert_xfm", "-omat", mni_to_dce_xfm,
        "-concat", str(t1w_to_dce_xfm), mni_to_t1w_xfm,
    ]
    result = subprocess.run(concat_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up
        Path(mni_to_t1w_xfm).unlink(missing_ok=True)
        return None

    # Step 3: Apply transform with nearest-neighbour interpolation
    tmp_atlas = str(out_dir / f"_tmp_{atlas_stem}_dce_raw.nii.gz")
    flirt_cmd = [
        "flirt",
        "-in", str(atlas_file),
        "-ref", str(reference_file),
        "-out", tmp_atlas,
        "-init", mni_to_dce_xfm,
        "-applyxfm",
        "-interp", "nearestneighbour",
    ]
    result = subprocess.run(flirt_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up
        Path(mni_to_t1w_xfm).unlink(missing_ok=True)
        Path(mni_to_dce_xfm).unlink(missing_ok=True)
        return None

    # Step 4: Morphological refinement - dilate-dilate-erode (1mm sphere)
    # This cleans up edges after interpolation artifacts
    # Derive unique output name from atlas filename to support cascade transforms
    atlas_stem = Path(atlas_file).name.replace(".nii.gz", "").replace(".nii", "")
    out_file = str(out_dir / f"{atlas_stem}_dce.nii.gz")
    morph_cmd = [
        "fslmaths", tmp_atlas,
        "-kernel", "sphere", "1",
        "-dilM", "-dilM", "-ero",
        "-bin", out_file,
    ]
    result = subprocess.run(morph_cmd, capture_output=True, text=True)

    # Clean up temp files
    Path(mni_to_t1w_xfm).unlink(missing_ok=True)
    Path(mni_to_dce_xfm).unlink(missing_ok=True)
    Path(tmp_atlas).unlink(missing_ok=True)

    if result.returncode != 0:
        return None

    return out_file


def _transform_atlas_cascade_to_dce(
    atlas_files: list[str] | None = None,
    t1w_to_mni_xfm: str | None = None,
    t1w_to_dce_xfm: str | None = None,
    reference_file: str | None = None,
) -> list[str]:
    """Transform a list of MNI atlas masks to DCE space via T1w.

    Computes the MNI→DCE composite transform once, then applies it to each
    atlas in the cascade, preserving ordering (tightest → broadest).

    Note: This function is self-contained because Nipype Function nodes
    serialize and execute functions in isolation — calls to sibling
    functions in the same module are not in scope at runtime.

    Parameters
    ----------
    atlas_files : list of str or None
        List of MNI-space atlas mask paths (ordered tightest first).
    t1w_to_mni_xfm : str or None
        Path to T1w→MNI FLIRT matrix.
    t1w_to_dce_xfm : str or None
        Path to T1w→DCE FLIRT matrix.
    reference_file : str or None
        Path to DCE reference image for output geometry.

    Returns
    -------
    list of str
        List of atlas mask paths in DCE space. Empty if inputs are missing
        or all transforms fail.
    """
    from pathlib import Path
    import subprocess

    from nipype.interfaces.base import isdefined

    if not atlas_files:
        return []

    # Validate shared inputs
    for inp in (t1w_to_mni_xfm, t1w_to_dce_xfm, reference_file):
        if not isdefined(inp) or inp is None:
            return []

    out_dir = Path.cwd()

    # Step 1: Compute composite MNI→DCE transform once (shared across atlases)
    mni_to_t1w_xfm = str(out_dir / "_tmp_cascade_mni_to_t1w.mat")
    invert_cmd = [
        "convert_xfm", "-omat", mni_to_t1w_xfm,
        "-inverse", str(t1w_to_mni_xfm),
    ]
    result = subprocess.run(invert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    mni_to_dce_xfm = str(out_dir / "_tmp_cascade_mni_to_dce.mat")
    concat_cmd = [
        "convert_xfm", "-omat", mni_to_dce_xfm,
        "-concat", str(t1w_to_dce_xfm), mni_to_t1w_xfm,
    ]
    result = subprocess.run(concat_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Path(mni_to_t1w_xfm).unlink(missing_ok=True)
        return []

    # Step 2: Apply the composite transform to each atlas
    out_files: list[str] = []
    for atlas_file in atlas_files:
        atlas_path = Path(atlas_file)
        if not atlas_path.is_file():
            continue

        atlas_stem = atlas_path.name.replace(".nii.gz", "").replace(".nii", "")
        tmp_atlas = str(out_dir / f"_tmp_{atlas_stem}_dce_raw.nii.gz")

        flirt_cmd = [
            "flirt",
            "-in", str(atlas_file),
            "-ref", str(reference_file),
            "-out", tmp_atlas,
            "-init", mni_to_dce_xfm,
            "-applyxfm",
            "-interp", "nearestneighbour",
        ]
        result = subprocess.run(flirt_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            Path(tmp_atlas).unlink(missing_ok=True)
            continue

        # Morphological refinement: dilate-dilate-erode (1mm sphere)
        out_file = str(out_dir / f"{atlas_stem}_dce.nii.gz")
        morph_cmd = [
            "fslmaths", tmp_atlas,
            "-kernel", "sphere", "1",
            "-dilM", "-dilM", "-ero",
            "-bin", out_file,
        ]
        result = subprocess.run(morph_cmd, capture_output=True, text=True)
        Path(tmp_atlas).unlink(missing_ok=True)

        if result.returncode == 0:
            out_files.append(out_file)

    # Clean up shared temp files
    Path(mni_to_t1w_xfm).unlink(missing_ok=True)
    Path(mni_to_dce_xfm).unlink(missing_ok=True)

    return out_files


def _signal_to_concentration_node(
    dce_file: str,
    brain_mask_file: str,
    t1_map_file: str | None,
    b1_map_file: str | None,
    flip_angle: float,
    tr: float,
    field_strength: float,
    r1_relaxivity: float | None,
    baseline_frames: int,
    no_t1: bool,
    external_t1: bool = False,
) -> tuple[str, dict[str, object]]:
    """Convert DCE signal to contrast-agent concentration.

    This function wraps the signal_to_concentration utility for use in
    a Nipype workflow.

    Parameters
    ----------
    dce_file : str
        Path to 4D DCE NIfTI file.
    brain_mask_file : str
        Path to the DCE-space brain mask defining the QC denominator.
    t1_map_file : str or None
        Path to T1 map in DCE space (None when no_t1 is True).
    b1_map_file : str or None
        Path to B1+ map in DCE space (None when no B1 correction).
    flip_angle : float
        DCE nominal flip angle in degrees.
    tr : float
        Pulse repetition time for the SPGR signal equation.
    field_strength : float
        Magnetic field strength in Tesla.
    r1_relaxivity : float or None
        r1 relaxivity of the contrast agent (L/mmol/s); None to use lookup.
    baseline_frames : int
        Number of pre-contrast baseline frames.
    no_t1 : bool
        If True, skip T1 mapping and use a field-strength default T1.
    external_t1 : bool, optional
        Whether the T1 map is a user-supplied map requiring alignment QC.

    Returns
    -------
    tuple
        Path to the 4D concentration NIfTI file and conversion QC metrics.
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    from pmriprep.utils.dce_constants import EPSILON_SIGNAL
    from pmriprep.utils.dce_signal import signal_to_concentration
    from pmriprep.utils.dce_t1_validation import (
        T1_TISSUE_REFERENCE,
        get_field_strength_key,
        valid_t1_support,
    )

    # Load DCE data
    dce_img = nib.load(dce_file)
    dce_data = dce_img.get_fdata()  # type: ignore[attr-defined]
    affine = dce_img.affine  # type: ignore[attr-defined]
    header = dce_img.header  # type: ignore[attr-defined]
    brain_mask = np.asarray(
        nib.load(brain_mask_file).dataobj  # type: ignore[attr-defined]
    ) > 0
    if brain_mask.shape != dce_data.shape[:3]:
        raise ValueError(
            f"DCE brain mask shape {brain_mask.shape} does not match "
            f"signal spatial shape {dce_data.shape[:3]}"
        )

    # Load T1 map if not skipping T1
    t1_map = None
    if not no_t1 and t1_map_file:
        try:
            t1_img = nib.load(t1_map_file)
            t1_map = t1_img.get_fdata()  # type: ignore[attr-defined]
        except Exception as exc:
            import logging as _logging

            _logging.getLogger("pmriprep.workflows.dce").warning(
                "Failed to load T1 map (%s) — using default T1", exc,
            )

    # Preserve the nonpositive B1 sentinel consumed by signal_to_concentration.
    b1_map = None
    if b1_map_file:
        try:
            b1_img = nib.load(b1_map_file)
            b1_map = b1_img.get_fdata()  # type: ignore[attr-defined]
        except Exception as exc:
            import logging as _logging

            _logging.getLogger("pmriprep.workflows.dce").warning(
                "Failed to load B1 map (%s) — using no B1 correction", exc,
            )

    # Convert signal to concentration
    result = signal_to_concentration(
        signal=dce_data,
        flip_angle_deg=flip_angle,
        tr=tr,
        t1_map=t1_map,
        b1_map=b1_map,
        r1_relaxivity=r1_relaxivity,
        field_strength=field_strength,
        baseline_frames=baseline_frames if baseline_frames > 0 else 5,
        convert_to_plasma=False,
        qc_mask=brain_mask,
    )

    # Zero out concentration for voxels with invalid T1 (NaN, sub-physiological).
    # After resampling, out-of-FOV voxels may have T1=NaN which produces
    # extreme concentration values that corrupt AIF detection thresholds.
    # invalid_t1 is 3D (spatial); fancy-indexing into the 4D concentration
    # array broadcasts the assignment across the trailing time axis.
    conversion_qc = dict(result.conversion_qc)
    conversion_qc["qc_domain"] = "dce_brain_mask"
    conversion_qc["t1_source"] = (
        "assumed" if t1_map is None else "external" if external_t1 else "map"
    )
    if t1_map is None:
        field_key = get_field_strength_key(field_strength)
        conversion_qc["assumed_t1_ms"] = float(
            T1_TISSUE_REFERENCE[field_key]["default"]
        )
        conversion_qc["assumed_t1_field_strength_t"] = float(field_strength)
        conversion_qc["assumed_t1_policy"] = "uniform_field_strength_reference"
    conversion_qc["b1_correction_applied"] = bool(
        b1_map is not None
        and np.any(
            np.isfinite(b1_map)
            & (b1_map > 0)
            & ~np.isclose(b1_map, 1.0, rtol=0.0, atol=1e-6)
        )
    )
    baseline_signal = np.asarray(result.baseline_signal)
    signal_support = (
        np.isfinite(baseline_signal)
        & (np.abs(baseline_signal) > EPSILON_SIGNAL)
        & brain_mask
    )
    if t1_map is not None:
        invalid_t1 = ~valid_t1_support(t1_map, field_strength)
        result.concentration[invalid_t1] = 0.0
        conversion_qc["t1_invalid_voxels"] = int(
            np.count_nonzero(invalid_t1 & signal_support)
        )
        conversion_qc["t1_invalid_fraction"] = (
            conversion_qc["t1_invalid_voxels"]
            / max(1, int(np.count_nonzero(signal_support)))
        )
        if external_t1:
            from scipy.ndimage import gaussian_gradient_magnitude  # type: ignore[import-untyped]

            t1_support = np.isfinite(t1_map) & (t1_map > 0)
            brain_voxels = max(1, int(np.count_nonzero(brain_mask)))
            support_overlap = int(np.count_nonzero(t1_support & brain_mask))
            gradient_domain = t1_support & brain_mask & signal_support
            gradient_ncc = float("nan")
            if np.count_nonzero(gradient_domain) >= 100:
                t1_values = np.where(t1_support, t1_map, np.nan)
                dce_values = np.where(signal_support, baseline_signal, np.nan)

                def _scaled_gradient(values: np.ndarray) -> np.ndarray:
                    finite = values[np.isfinite(values)]
                    low, high = (float(value) for value in np.percentile(finite, [1, 99]))
                    scaled = np.nan_to_num(
                        np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
                    )
                    return np.asarray(
                        gaussian_gradient_magnitude(scaled, sigma=1.0),
                        dtype=np.float64,
                    )

                t1_gradient = _scaled_gradient(t1_values)
                dce_gradient = _scaled_gradient(dce_values)
                if (
                    np.std(t1_gradient[gradient_domain]) > 0
                    and np.std(dce_gradient[gradient_domain]) > 0
                ):
                    gradient_ncc = float(
                        np.corrcoef(
                            t1_gradient[gradient_domain],
                            dce_gradient[gradient_domain],
                        )[0, 1]
                    )
            conversion_qc["external_t1_alignment"] = {
                "assessment": "informational_not_gated",
                "brain_support_fraction": support_overlap / brain_voxels,
                "gradient_magnitude_ncc": (
                    gradient_ncc if np.isfinite(gradient_ncc) else None
                ),
                "resampling": "nifti_world_affine",
            }
    else:
        conversion_qc["t1_invalid_voxels"] = 0
        conversion_qc["t1_invalid_fraction"] = 0.0
    conversion_qc["nonfinite_output_samples"] = int(
        np.count_nonzero(
            (~np.isfinite(result.concentration))
            & signal_support[..., np.newaxis]
        )
    )
    conversion_qc["total_grid_nonfinite_output_samples"] = int(
        np.count_nonzero(~np.isfinite(result.concentration))
    )

    # Save concentration file
    out_file = str(Path.cwd() / "dce_concentration.nii.gz")
    conc_img = nib.Nifti1Image(result.concentration.astype(np.float32), affine, header)
    nib.save(conc_img, out_file)

    return out_file, conversion_qc


def _dsc_dce_validation_node(
    vp_file: str | None = None,
    dsc_cbv_file: str | None = None,
    dsc_ncbv_file: str | None = None,
    brain_mask: str | None = None,
    no_dsc_validation: bool = False,
    pk_qc: dict[str, object] | str | None = None,
    dsc_preload_configured: bool | None = None,
) -> dict[str, object]:
    """Run DSC-DCE cross-validation.

    Parameters
    ----------
    vp_file : str or None
        Path to the DCE vp (plasma volume) map.
    dsc_cbv_file : str or None
        Path to resampled DSC CBV map in DCE space.
    dsc_ncbv_file : str or None
        Optional NAWM-normalized CBV. When provided, validation
        uses the nCBV-anchored expected ratio (``NAWM_VP_REFERENCE``).
    brain_mask : str or None
        Path to brain mask in DCE space.
    no_dsc_validation : bool
        If True, skip cross-validation.
    pk_qc : dict or str or None
        Full ``pk_modeling.qc_metrics`` dict or a constraint-mode string. The
        function extracts the constraint mode and relative prior width from
        ``dsc_cbv_constraint`` when present.
    dsc_preload_configured : bool or None
        Whether DSC preload was configured.

    Returns
    -------
    dict
        Dictionary with validation results.
    """
    import os

    from pmriprep.utils.dce_dsc_validation import validate_vp_cbv_from_files
    from pmriprep.utils.dsc_cbv_constraint import (
        NO_DSC_RUN_REASON_STRING,
        SKIP_REASON_NO_DSC_RUN,
        SKIP_REASON_SYNTHETIC_AIF,
        SYNTHETIC_AIF_REASON_STRING,
    )

    resolved_mode: str | None = None
    constraint_relative_width: float | None = None
    constraint_skip_reason: str | None = None
    if isinstance(pk_qc, dict):
        constraint_value = pk_qc.get("dsc_cbv_constraint")
        if isinstance(constraint_value, dict):
            mode_value = constraint_value.get("mode")
            if isinstance(mode_value, str):
                resolved_mode = mode_value
            width_value = constraint_value.get("vp_uncertainty")
            if isinstance(width_value, (int, float)):
                constraint_relative_width = float(width_value)
            skip_value = constraint_value.get("skip_reason")
            if isinstance(skip_value, str):
                constraint_skip_reason = skip_value
    elif isinstance(pk_qc, str):
        resolved_mode = pk_qc

    have_cbv = bool(dsc_cbv_file) and os.path.isfile(str(dsc_cbv_file))
    have_ncbv = bool(dsc_ncbv_file) and os.path.isfile(str(dsc_ncbv_file))
    dsc_inputs_present = (
        have_cbv or have_ncbv or (dsc_preload_configured is not None)
    )

    if no_dsc_validation or not vp_file:
        return {
            "skipped": True,
            "status": "skipped",
            "reason": "Validation skipped or missing data",
        }

    # Synthetic-AIF CBV does not have the subject-specific scale required by
    # the NAWM-anchored cross-validation ratio bands.
    if constraint_skip_reason == SKIP_REASON_SYNTHETIC_AIF:
        return {
            "skipped": True,
            "status": "SKIPPED",
            "reason": SYNTHETIC_AIF_REASON_STRING,
            "skip_reason": SKIP_REASON_SYNTHETIC_AIF,
        }

    if not dsc_inputs_present:
        return {
            "skipped": True,
            "status": "SKIPPED",
            "reason": NO_DSC_RUN_REASON_STRING,
            "skip_reason": SKIP_REASON_NO_DSC_RUN,
        }

    # Cross-validation is defined for NAWM-normalized CBV; the expected
    # vp-to-CBV ratio is undefined for unanchored relative CBV.
    if dsc_ncbv_file is None or not have_ncbv:
        return {
            "skipped": True,
            "status": "SKIPPED",
            "reason": (
                "DSC-DCE xval skipped: nCBV unavailable (no NAWM mask). "
                "Pipeline does not produce absolute mL/100g CBV, so the "
                "literature-derived expected ratio against rel-AUC-CBV "
                "is undefined. Provide a NAWM mask to enable xval."
            ),
        }

    # The file-based validator requires a DCE-space brain mask.
    if not brain_mask:
        return {
            "skipped": True,
            "status": "SKIPPED",
            "reason": "DSC-DCE xval skipped: brain_mask unavailable.",
        }

    try:
        result = validate_vp_cbv_from_files(
            vp_path=vp_file,
            ncbv_path=dsc_ncbv_file,
            mask_path=brain_mask,
            constraint_mode=resolved_mode,
            dsc_preload_configured=dsc_preload_configured,
            constraint_relative_width=constraint_relative_width,
        )
        return result.to_dict()
    except Exception as e:
        import logging

        logging.getLogger("pmriprep.workflows.dce").warning(
            "DSC-DCE validation raised: %s", e, exc_info=True
        )
        return {
            "skipped": True,
            "status": "error",
            "reason": str(e),
        }


def _compute_early_mean(dce_file: str, n_frames: int = 3) -> str:
    """Compute mean of first N DCE volumes for brain extraction.

    Provides a pre-contrast structural image for HD-BET before motion
    correction. The first few frames are guaranteed pre-contrast,
    producing clean gray/white contrast without gadolinium effects.

    Parameters
    ----------
    dce_file : str
        Path to reoriented 4D DCE NIfTI.
    n_frames : int, optional
        Number of early frames to average (default 3).

    Returns
    -------
    str
        Path to the 3D mean NIfTI file.
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    img = nib.load(dce_file)
    data = img.get_fdata()  # type: ignore[attr-defined]

    if data.ndim != 4:
        raise ValueError(f"Expected 4D DCE data, got {data.ndim}D")

    n = min(n_frames, data.shape[3])
    mean_vol = np.mean(data[..., :n], axis=3).astype(np.float32)

    out_file = str(Path(dce_file).parent / "early_mean.nii.gz")
    nib.save(
        nib.Nifti1Image(mean_vol, img.affine, img.header),  # type: ignore[attr-defined]
        out_file,
    )
    return out_file


def _dsc_dce_xval_plot_node(
    vp_file: str | None = None,
    dsc_ncbv_file: str | None = None,
    brain_mask: str | None = None,
    validation_result: dict[str, object] | None = None,
    subject_id: str = "",
    session_id: str = "",
    model_selection_file: str | None = None,
) -> str | None:
    """Generate the DSC-DCE cross-validation plot.

    Parameters
    ----------
    vp_file : str or None
        Path to the DCE vp (plasma volume) map.
    dsc_ncbv_file : str or None
        NAWM-normalized CBV. The plot is drawn on the dimensionless
        nCBV scale with expected ratio =
        ``NAWM_VP_REFERENCE``, matching the validation metrics.
    brain_mask : str or None
        Path to brain mask in DCE space.
    validation_result : dict or None
        Result dict from the DSC-DCE validation node.
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.
    model_selection_file : str or None
        Path to the model selection map NIfTI (auto mode), for coloring.

    Returns
    -------
    str or None
        Path to plot PNG, or None if skipped.
    """
    import os

    if not vp_file or not brain_mask:
        return None
    if validation_result and validation_result.get("skipped"):
        return None
    have_ncbv = bool(dsc_ncbv_file) and os.path.isfile(str(dsc_ncbv_file))
    # Cross-validation requires NAWM-normalized CBV.
    if not have_ncbv or dsc_ncbv_file is None:
        return None

    try:
        import nibabel as nib  # type: ignore[import-untyped]
        import numpy as np

        from pmriprep.utils.dsc_dce_xval_plots import generate_xval_plot

        cbv_source = dsc_ncbv_file

        vp_data = np.asarray(
            nib.load(vp_file).get_fdata(), dtype=np.float64  # type: ignore[attr-defined]
        )
        cbv_data = np.asarray(
            nib.load(cbv_source).get_fdata(), dtype=np.float64  # type: ignore[attr-defined]
        )
        mask = np.asarray(
            nib.load(brain_mask).get_fdata(), dtype=np.bool_  # type: ignore[attr-defined]
        )

        model_sel = None
        if model_selection_file:
            model_sel = np.asarray(
                nib.load(model_selection_file).get_fdata(),  # type: ignore[attr-defined]
                dtype=np.int8,
            )

        return generate_xval_plot(
            vp_data=vp_data,
            cbv_data=cbv_data,
            mask=mask,
            validation_result=validation_result or {},
            subject_id=subject_id,
            session_id=session_id,
            model_selection=model_sel,
        )
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "DSC-DCE xval plot failed", exc_info=True
        )
        return None


def _collect_qc_metrics(
    b1_qc: dict[str, object] | None,
    t1_qc: dict[str, object] | None,
    signal_conversion_qc: dict[str, object] | None,
    preprocess_qc: dict[str, object] | None,
    registration_qc: dict[str, object] | None,
    aif_qc: dict[str, object] | None,
    pk_qc: dict[str, object] | None,
    validation_result: dict[str, object] | None,
    vfa_qc: dict[str, object] | None = None,
    preproc_display: dict[str, object] | None = None,
    pk_fit_qc_metrics: dict[str, object] | None = None,
    pk_maps_qc_metrics: dict[str, object] | None = None,
    vp_at_bound_fractions: dict[str, object] | None = None,
    use_external_t1: bool = False,
    no_t1: bool = False,
    b1_method: str = "auto",
    nawm_available: bool = False,
) -> dict[str, object]:
    """Collect all QC metrics into a single dictionary.

    Parameters
    ----------
    b1_qc : dict or None
        B1+ mapping QC metrics (None if not performed).
    t1_qc : dict or None
        T1 mapping QC metrics (None if external or no_t1).
    preprocess_qc : dict or None
        Preprocessing QC metrics.
    registration_qc : dict or None
        Registration QC metrics.
    aif_qc : dict or None
        AIF detection QC metrics.
    pk_qc : dict or None
        Pharmacokinetic modeling QC metrics.
    validation_result : dict or None
        DSC-DCE validation result.
    vfa_qc : dict or None
        VFA inter-volume preprocessing and motion-assessment metrics.
    preproc_display : dict, optional
        Preprocessing display stats.
    pk_fit_qc_metrics : dict, optional
        PK fit QC metrics from plot node.
    pk_maps_qc_metrics : dict, optional
        Region→model-distribution mapping from plot node.
    vp_at_bound_fractions : dict, optional
        Scalar whole-brain/ROI vp-at-bound fractions.
    use_external_t1 : bool, optional
        Whether external T1 map was used.
    no_t1 : bool, optional
        Whether T1 mapping was skipped (assumed T1).
    b1_method : str, optional
        B1 correction method used.
    nawm_available : bool, optional
        Whether NAWM mask was available for normalization.

    Returns
    -------
    dict
        Combined QC metrics dictionary.
    """
    qc_metrics: dict[str, object] = {}

    if b1_qc is not None:
        qc_metrics["b1_mapping"] = b1_qc
    if signal_conversion_qc is not None:
        qc_metrics["signal_conversion"] = signal_conversion_qc

    # T1 mapping section - handle VFA, external, and assumed cases
    if t1_qc is not None:
        # VFA T1 fitting was performed - t1_method already in qc
        t1_mapping = dict(t1_qc)
        # Add B1 correction method for boilerplate
        if b1_qc is not None:
            b1_correction = b1_qc.get("method", b1_method)
        else:
            b1_correction = "none"
        t1_mapping["b1_correction"] = b1_correction
        if vfa_qc is not None:
            t1_mapping["vfa_preprocessing"] = dict(vfa_qc)
        qc_metrics["t1_mapping"] = t1_mapping
    elif use_external_t1:
        # External T1 map was provided
        alignment = (
            signal_conversion_qc.get("external_t1_alignment")
            if isinstance(signal_conversion_qc, dict)
            else None
        )
        qc_metrics["t1_mapping"] = {
            "method": "external",
            "b1_correction": b1_method if b1_method != "none" else "none",
            "external_alignment": alignment,
        }
    elif no_t1:
        # No T1 mapping - using assumed literature values
        conversion = (
            signal_conversion_qc
            if isinstance(signal_conversion_qc, dict)
            else {}
        )
        if b1_qc is not None:
            b1_correction = str(b1_qc.get("method", b1_method))
        else:
            b1_correction = "none"
        qc_metrics["t1_mapping"] = {
            "method": "assumed",
            "t1_status": "Skipped by configuration",
            "skipped": True,
            "assumed_t1_ms": conversion.get("assumed_t1_ms"),
            "assumed_t1_field_strength_t": conversion.get(
                "assumed_t1_field_strength_t"
            ),
            "assumed_t1_policy": conversion.get("assumed_t1_policy"),
            "b1_correction": b1_correction,
        }

    if preprocess_qc is not None:
        qc_metrics["preprocessing"] = preprocess_qc
    if registration_qc is not None:
        qc_metrics["registration"] = registration_qc
    if aif_qc is not None:
        qc_metrics["aif_detection"] = aif_qc
    pk_modeling = dict(pk_qc) if pk_qc is not None else {}
    have_pk_modeling = pk_qc is not None
    if pk_fit_qc_metrics is not None:
        fit_metrics = dict(pk_fit_qc_metrics)
        regional_accounting = fit_metrics.pop(
            "voxel_accounting_regions", None
        )
        pk_modeling.update(fit_metrics)
        if regional_accounting:
            accounting = pk_modeling.get("voxel_accounting")
            if isinstance(accounting, dict):
                accounting["regions"] = regional_accounting
        have_pk_modeling = True
    if pk_maps_qc_metrics is not None:
        pk_modeling["model_selection_regions"] = pk_maps_qc_metrics
        have_pk_modeling = True
    if vp_at_bound_fractions:
        pk_modeling["vp_at_bound_fractions"] = vp_at_bound_fractions
        have_pk_modeling = True
    if have_pk_modeling:
        qc_metrics["pk_modeling"] = pk_modeling
    if validation_result is not None:
        qc_metrics["dsc_dce_validation"] = validation_result

    # Merge display stats (TR, tSNR, peak enhancement) into preprocessing.
    if preproc_display is not None:
        preprocessing = dict(preprocess_qc) if preprocess_qc is not None else {}
        preprocessing.update(preproc_display)
        qc_metrics["preprocessing"] = preprocessing

    qc_metrics["nawm_normalization"] = bool(nawm_available)

    from pmriprep.utils.dce_model_suitability import (
        assess_two_cxm_suitability,
    )

    qc_metrics["model_suitability"] = {
        "2cxm": assess_two_cxm_suitability(qc_metrics)
    }

    return qc_metrics


def _resample_b1_ratio_to_img(b1_img: Any, ref_img: Any) -> Any:
    """Resample a B1+ ratio without diluting its boundary with missing zeros."""
    import warnings

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np
    from nilearn.image import resample_to_img  # type: ignore[import-untyped]

    data = np.asarray(b1_img.dataobj, dtype=np.float32)  # type: ignore[attr-defined]
    support = np.isfinite(data) & (data > 0)
    values = np.where(support, data, 0.0)
    value_img = nib.Nifti1Image(values, b1_img.affine)  # type: ignore[attr-defined]
    support_img = nib.Nifti1Image(support.astype(np.float32), b1_img.affine)  # type: ignore[attr-defined]
    numerator = resample_to_img(
        value_img, ref_img, interpolation="continuous", force_resample=True,
        copy_header=True,
    )
    with warnings.catch_warnings():
        # Continuous support weights are intentional: they normalize the
        # interpolated numerator. The nearest-neighbour map below defines the
        # retained binary support.
        warnings.filterwarnings(
            "ignore", message="Resampling binary images with continuous"
        )
        weights = resample_to_img(
            support_img, ref_img, interpolation="continuous", force_resample=True,
            copy_header=True,
        )
    target_support = resample_to_img(
        support_img, ref_img, interpolation="nearest", force_resample=True,
        copy_header=True,
    )
    num = np.asarray(numerator.dataobj, dtype=np.float32)  # type: ignore[attr-defined]
    den = np.asarray(weights.dataobj, dtype=np.float32)  # type: ignore[attr-defined]
    keep = (np.asarray(target_support.dataobj) > 0.5) & (den > 1e-6)  # type: ignore[attr-defined]
    out = np.zeros(ref_img.shape[:3], dtype=np.float32)  # type: ignore[attr-defined]
    np.divide(num, den, out=out, where=keep)
    return nib.Nifti1Image(out, ref_img.affine, ref_img.header)  # type: ignore[attr-defined]


def _transform_b1_to_vfa(
    b1_map_file: str | None = None,
    vfa_ref_file: str | None = None,
) -> str | None:
    """Resample B1+ map from AFI space to VFA space.

    Uses nilearn continuous interpolation via NIfTI affine headers.
    Passes through None unchanged if B1 mapping was skipped.

    Parameters
    ----------
    b1_map_file : str or None
        Path to B1+ map in AFI space (or None).
    vfa_ref_file : str or None
        Path to VFA reference image for output geometry.

    Returns
    -------
    str or None
        Path to B1+ map resampled to VFA space, or original if same space.
    """
    if b1_map_file is None or vfa_ref_file is None:
        return b1_map_file

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    from pmriprep.workflows.dce._helpers import _resample_b1_ratio_to_img

    b1_img = nib.load(b1_map_file)
    ref_img = nib.load(vfa_ref_file)

    # Check if already in the same space (same shape + affine)
    if b1_img.shape[:3] == ref_img.shape[:3]:  # type: ignore[attr-defined]
        affine_diff = float(np.abs(b1_img.affine - ref_img.affine).max())  # type: ignore[attr-defined]
        if affine_diff < 1e-4:
            return b1_map_file

    resampled = _resample_b1_ratio_to_img(b1_img, ref_img)
    out_file = str(Path.cwd() / "b1_map_vfa_space.nii.gz")
    nib.save(resampled, out_file)
    return out_file


def _transform_b1_to_dce(
    b1_map_file: str | None = None,
    reference_file: str | None = None,
) -> str | None:
    """Resample B1+ map to DCE space for signal-to-concentration.

    Uses nilearn continuous interpolation via NIfTI affine headers.

    Parameters
    ----------
    b1_map_file : str or None
        Path to B1+ map (or None).
    reference_file : str or None
        Path to DCE mean image for output geometry.

    Returns
    -------
    str or None
        Path to B1+ map resampled to DCE space, or original if same space.
    """
    if b1_map_file is None or reference_file is None:
        return b1_map_file

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    from pmriprep.workflows.dce._helpers import _resample_b1_ratio_to_img

    b1_img = nib.load(b1_map_file)
    ref_img = nib.load(reference_file)

    # Check if already in the same space
    if b1_img.shape[:3] == ref_img.shape[:3]:  # type: ignore[attr-defined]
        affine_diff = float(np.abs(b1_img.affine - ref_img.affine).max())  # type: ignore[attr-defined]
        if affine_diff < 1e-4:
            return b1_map_file

    resampled = _resample_b1_ratio_to_img(b1_img, ref_img)
    out_file = str(Path.cwd() / "b1_map_dce_space.nii.gz")
    nib.save(resampled, out_file)
    return out_file


def _resample_dsc_cbv_to_dce(
    dsc_cbv_file: str | None = None,
    dsc_to_t1w_xfm: str | None = None,
    t1w_to_dce_xfm: str | None = None,
    dce_reference: str | None = None,
) -> str:
    """Resample DSC CBV map to DCE native space.

    Chains two existing transforms: DSC→T1w + T1w→DCE = DSC→DCE,
    then applies the combined transform with trilinear interpolation
    (not spline, to avoid negative CBV values).

    Parameters
    ----------
    dsc_cbv_file : str or None
        Path to DSC-derived relative CBV map (DSC native space).
    dsc_to_t1w_xfm : str or None
        Path to DSC→T1w FLIRT transform matrix.
    t1w_to_dce_xfm : str or None
        Path to T1w→DCE FLIRT transform matrix (from DCE registration step).
    dce_reference : str or None
        Path to DCE reference image for output geometry.

    Returns
    -------
    str
        Path to CBV map resampled to DCE space, or empty string if inputs
        are missing (empty string avoids nipype File trait validation errors
        when None flows through workflow connections).
    """
    import logging
    import os

    _log = logging.getLogger("pmriprep.workflows.dce")

    if not dsc_cbv_file or not os.path.isfile(str(dsc_cbv_file)):
        return ""
    if not dsc_to_t1w_xfm or not os.path.isfile(str(dsc_to_t1w_xfm)):
        return ""
    if not t1w_to_dce_xfm or not os.path.isfile(str(t1w_to_dce_xfm)):
        return ""
    if not dce_reference or not os.path.isfile(str(dce_reference)):
        return ""

    import subprocess

    out_dir = os.getcwd()
    dsc_to_dce_xfm = os.path.join(out_dir, "dsc_to_dce.mat")
    cbv_in_dce = os.path.join(out_dir, "dsc_cbv_in_dce.nii.gz")

    # Chain: DSC→T1w + T1w→DCE = DSC→DCE
    result = subprocess.run(
        [
            "convert_xfm",
            "-omat", dsc_to_dce_xfm,
            "-concat", str(t1w_to_dce_xfm), str(dsc_to_t1w_xfm),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _log.warning(
            "convert_xfm failed (rc=%d): %s", result.returncode, result.stderr
        )
        return ""

    # Apply combined transform with trilinear interpolation
    result = subprocess.run(
        [
            "flirt",
            "-in", str(dsc_cbv_file),
            "-ref", str(dce_reference),
            "-out", cbv_in_dce,
            "-init", dsc_to_dce_xfm,
            "-applyxfm",
            "-interp", "trilinear",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _log.warning(
            "FLIRT resampling of DSC CBV failed (rc=%d): %s",
            result.returncode, result.stderr,
        )
        return ""

    return cbv_in_dce


def _resample_dsc_ncbv_to_dce(
    dsc_cbv_file: str | None = None,
    dsc_to_t1w_xfm: str | None = None,
    t1w_to_dce_xfm: str | None = None,
    dce_reference: str | None = None,
) -> str:
    """Resample DSC nCBV to DCE space and clip trilinear-ringing negatives.

    Nipype executes this helper in an isolated scope, so it contains the full
    resampling operation before clipping negative interpolation artifacts.

    Trilinear interpolation introduces small negative values near
    high-contrast boundaries (tumor edges, vessels). nCBV=0 is physically
    meaningful, but negatives would silently become NaN downstream in
    ``ncbv_to_vp_prior`` (it NaN-masks negatives); that masks useful
    voxels and inflates the nan-fraction QC metric. We clip to [0, +inf)
    and log the affected fraction.
    """
    import logging
    import os
    import subprocess

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    _log = logging.getLogger("pmriprep.workflows.dce")

    if not dsc_cbv_file or not os.path.isfile(str(dsc_cbv_file)):
        return ""
    if not dsc_to_t1w_xfm or not os.path.isfile(str(dsc_to_t1w_xfm)):
        return ""
    if not t1w_to_dce_xfm or not os.path.isfile(str(t1w_to_dce_xfm)):
        return ""
    if not dce_reference or not os.path.isfile(str(dce_reference)):
        return ""

    out_dir = os.getcwd()
    dsc_to_dce_xfm = os.path.join(out_dir, "dsc_to_dce_ncbv.mat")
    raw_out = os.path.join(out_dir, "dsc_ncbv_in_dce_raw.nii.gz")

    # Chain: DSC→T1w + T1w→DCE = DSC→DCE
    result = subprocess.run(
        [
            "convert_xfm",
            "-omat", dsc_to_dce_xfm,
            "-concat", str(t1w_to_dce_xfm), str(dsc_to_t1w_xfm),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _log.warning(
            "convert_xfm (nCBV) failed (rc=%d): %s",
            result.returncode, result.stderr,
        )
        return ""

    result = subprocess.run(
        [
            "flirt",
            "-in", str(dsc_cbv_file),
            "-ref", str(dce_reference),
            "-out", raw_out,
            "-init", dsc_to_dce_xfm,
            "-applyxfm",
            "-interp", "trilinear",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _log.warning(
            "FLIRT resampling of DSC nCBV failed (rc=%d): %s",
            result.returncode, result.stderr,
        )
        return ""

    img = nib.load(raw_out)
    data = np.asarray(img.get_fdata(), dtype=np.float32)  # type: ignore[attr-defined]
    finite = np.isfinite(data)
    negative = finite & (data < 0)
    n_neg = int(np.sum(negative))
    if n_neg:
        frac = n_neg / max(1, int(np.sum(finite)))
        _log.info(
            "Resampled nCBV had %d voxels (%.2f%%) with negative values "
            "from trilinear ringing; clipped to 0.",
            n_neg, 100.0 * frac,
        )
    clipped = np.where(finite, np.maximum(data, 0.0), data)

    clipped_out = os.path.join(out_dir, "dsc_ncbv_in_dce.nii.gz")
    nib.save(
        nib.Nifti1Image(clipped.astype(np.float32), img.affine, img.header),  # type: ignore[attr-defined]
        clipped_out,
    )
    return clipped_out


def _transform_t1_to_dce(
    t1_map_file: str | None = None,
    reference_file: str | None = None,
) -> str | None:
    """Resample T1 map from VFA space to DCE space.

    Uses nilearn continuous interpolation via NIfTI affine headers.
    This is critical to avoid shape mismatches in signal_to_concentration
    and pk_modeling.

    Parameters
    ----------
    t1_map_file : str or None
        Path to T1 map in VFA space (or None).
    reference_file : str or None
        Path to DCE mean image for output geometry.

    Returns
    -------
    str or None
        Path to T1 map resampled to DCE space, or original if same space.
    """
    if t1_map_file is None or reference_file is None:
        return t1_map_file

    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    t1_img = nib.load(t1_map_file)
    ref_img = nib.load(reference_file)

    # Check if already in the same space
    if t1_img.shape[:3] == ref_img.shape[:3]:  # type: ignore[attr-defined]
        affine_diff = float(np.abs(t1_img.affine - ref_img.affine).max())  # type: ignore[attr-defined]
        if affine_diff < 1e-4:
            return t1_map_file

    from nilearn.image import resample_to_img  # type: ignore[import-untyped]

    resampled = resample_to_img(t1_img, ref_img, interpolation="continuous")

    # Restore NaN for voxels that were NaN before resampling.
    # nilearn's continuous interpolation replaces NaN/out-of-FOV with 0,
    # which downstream causes T1=0 → R1=inf → concentration artifacts.
    data = resampled.get_fdata()
    n_invalid = np.sum(data <= 0)
    if n_invalid > 0:
        data[data <= 0] = np.nan
        resampled = nib.Nifti1Image(
            data.astype(np.float32), resampled.affine, resampled.header,
        )

    out_file = str(Path.cwd() / "t1_map_dce_space.nii.gz")
    nib.save(resampled, out_file)
    return out_file


def _transform_external_t1_to_dce(
    t1_map_file: str | None = None,
    reference_file: str | None = None,
) -> str | None:
    """Resample external T1 map to DCE space with unit validation.

    Validates that T1 values are in milliseconds (expected units for the
    pipeline). If median T1 < 10, assumes values are in seconds and
    auto-converts, then resamples to DCE space.

    Parameters
    ----------
    t1_map_file : str or None
        Path to external T1 map (NIfTI).
    reference_file : str or None
        Path to DCE mean image for output geometry.

    Returns
    -------
    str or None
        Path to T1 map resampled to DCE space in milliseconds.
    """
    if t1_map_file is None or reference_file is None:
        return t1_map_file

    import logging
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np

    logger = logging.getLogger("pmriprep.workflows.dce")

    def _resample(
        src_t1_file: str | None,
        ref_file: str | None,
    ) -> str | None:
        # Nipype executes Function helpers without module-level helpers.
        if src_t1_file is None or ref_file is None:
            return src_t1_file
        from nilearn.image import resample_to_img  # type: ignore[import-untyped]
        t1_img_local = nib.load(src_t1_file)
        ref_img_local = nib.load(ref_file)
        if t1_img_local.shape[:3] == ref_img_local.shape[:3]:  # type: ignore[attr-defined]
            affine_diff = float(np.abs(t1_img_local.affine - ref_img_local.affine).max())  # type: ignore[attr-defined]
            if affine_diff < 1e-4:
                return src_t1_file
        resampled = resample_to_img(t1_img_local, ref_img_local, interpolation="continuous")
        data_local = resampled.get_fdata()
        if np.sum(data_local <= 0) > 0:
            data_local[data_local <= 0] = np.nan
            resampled = nib.Nifti1Image(
                data_local.astype(np.float32), resampled.affine, resampled.header,
            )
        out_local = str(Path.cwd() / "t1_map_dce_space.nii.gz")
        nib.save(resampled, out_local)
        return out_local

    t1_img = nib.load(t1_map_file)
    t1_data = t1_img.get_fdata()  # type: ignore[attr-defined]
    valid = t1_data[t1_data > 0]
    if len(valid) == 0:
        logger.warning("External T1 map has no positive values.")
        return _resample(t1_map_file, reference_file)

    median_t1 = float(np.nanmedian(valid))

    if median_t1 < 10:  # Likely seconds, not ms
        logger.warning(
            "External T1 map median=%.3f — appears to be in seconds. "
            "Converting to milliseconds (expected units).",
            median_t1,
        )
        t1_data = t1_data * 1000.0
        t1_img = nib.Nifti1Image(
            t1_data.astype(np.float32), t1_img.affine, t1_img.header,  # type: ignore[attr-defined]
        )
        corrected = str(Path.cwd() / "external_t1_ms.nii.gz")
        nib.save(t1_img, corrected)
        t1_map_file = corrected
    elif median_t1 > 10000:
        logger.warning(
            "External T1 map median=%.0f ms — unusually large. "
            "Verify units are in milliseconds.",
            median_t1,
        )

    return _resample(t1_map_file, reference_file)


def _external_t1_alignment_plot_node(
    t1_map_file: str,
    dce_reference_file: str,
    brain_mask_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Plot the DCE-space external T1 map and its edges on the DCE reference."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np
    from scipy.ndimage import gaussian_gradient_magnitude  # type: ignore[import-untyped]

    t1 = np.asarray(nib.load(t1_map_file).get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]
    dce = np.asarray(nib.load(dce_reference_file).get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]
    mask = np.asarray(nib.load(brain_mask_file).get_fdata(), dtype=np.float64) > 0  # type: ignore[attr-defined]
    support = mask & np.isfinite(t1) & (t1 > 0)

    def _scaled(data: np.ndarray, domain: np.ndarray) -> np.ndarray:
        values = data[domain & np.isfinite(data)]
        if values.size == 0:
            return np.zeros(data.shape, dtype=float)
        low, high = (float(value) for value in np.percentile(values, [2, 98]))
        return np.clip((data - low) / max(high - low, 1e-8), 0.0, 1.0)

    dce_scaled = _scaled(dce, mask)
    t1_scaled = _scaled(t1, support)
    edges = gaussian_gradient_magnitude(np.nan_to_num(t1_scaled), sigma=1.0)
    edge_values = edges[support]
    edge_cut = float(np.percentile(edge_values, 82)) if edge_values.size else 1.0
    occupied = np.flatnonzero(mask.sum(axis=(0, 1)) > 0)
    if occupied.size == 0:
        slices = [dce.shape[2] // 2] * 3
    else:
        slices = [
            int(occupied[int(fraction * (occupied.size - 1))])
            for fraction in (0.28, 0.50, 0.72)
        ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), facecolor="white")
    for column, index in enumerate(slices):
        base = np.rot90(dce_scaled[:, :, index])
        t1_slice = np.rot90(np.where(support[:, :, index], t1[:, :, index], np.nan))
        edge_slice = np.rot90(edges[:, :, index] > edge_cut)
        axes[0, column].imshow(base, cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(t1_slice, cmap="magma", vmin=600, vmax=2000)
        axes[2, column].imshow(base, cmap="gray", vmin=0, vmax=1)
        if np.any(edge_slice):
            axes[2, column].contour(
                edge_slice, levels=[0.5], colors="#25d9ff", linewidths=0.8
            )
        axes[0, column].set_title(f"Axial slice {index}", fontsize=11)
        for row in range(3):
            axes[row, column].axis("off")
    for row, label in enumerate(
        ("Precontrast DCE reference", "External T1 used (ms)", "T1 edges on DCE")
    ):
        axes[row, 0].text(
            -0.08, 0.5, label, rotation=90, va="center", ha="right",
            transform=axes[row, 0].transAxes, fontsize=11, fontweight="bold",
        )
    identity = f"sub-{subject_id}" if subject_id else "External T1"
    if session_id:
        identity += f" | ses-{session_id}"
    fig.suptitle(f"{identity} | External T1–DCE Alignment", fontsize=16, y=0.98)
    support_fraction = (
        100.0 * int(np.count_nonzero(support))
        / max(1, int(np.count_nonzero(mask)))
    )
    fig.text(0.5, 0.025, f"Brain support: {support_fraction:.2f}%", ha="center")
    fig.tight_layout(rect=(0.04, 0.05, 1, 0.95), h_pad=0.8, w_pad=0.25)
    out_file = str(Path.cwd() / "external_t1_alignment.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _transform_structural_to_dce(
    structural_file: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
    output_name: str = "structural_in_dce.nii.gz",
) -> str:
    """Transform structural image from T1w space to DCE space via FLIRT.

    Parameters
    ----------
    structural_file : str or None
        Path to structural image in T1w space (or None).
    reference_file : str or None
        Path to DCE reference image for output geometry.
    transform_matrix : str or None
        Path to T1w→DCE FLIRT matrix.
    output_name : str, optional
        Output filename.

    Returns
    -------
    str
        Path to transformed structural image in DCE space, or '' if inputs
        missing.
    """
    from nipype.interfaces.base import isdefined

    # Returns "" (not None) because downstream traits.Str / niworkflows
    # File rejects None.
    if not isdefined(structural_file) or structural_file is None:
        return ""
    if not isdefined(reference_file) or reference_file is None:
        return ""
    if not isdefined(transform_matrix) or transform_matrix is None:
        return ""

    from pathlib import Path
    import subprocess

    if not Path(structural_file).is_file():
        return ""

    out_file = str(Path.cwd() / output_name)
    subprocess.run(
        [
            "flirt",
            "-in", str(structural_file),
            "-ref", str(reference_file),
            "-out", out_file,
            "-init", str(transform_matrix),
            "-applyxfm",
            "-interp", "spline",
        ],
        check=True,
        capture_output=True,
    )
    return out_file


def _generate_dce_motion_qc_plot(
    motion_params_file: str,
    tr: float,
    subject_id: str = "",
    session_id: str = "",
    motion_applied: bool = True,
) -> str:
    """Generate motion QC plot from DCE motion parameters.

    Creates a two-panel figure showing raw motion parameters and
    framewise displacement over time.

    Parameters
    ----------
    motion_params_file : str
        Path to motion parameters text file.
    tr : float
        Repetition time in seconds.
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.
    motion_applied : bool, optional
        Whether motion correction was applied. When False, the plotted values
        are residual motion (mcflirt was run for measurement only) and the
        figure is annotated accordingly.

    Returns
    -------
    str
        Path to the generated motion QC plot PNG.
    """
    import numpy as np
    from pathlib import Path
    from pmriprep.utils.qc_utils import plot_motion_qc, qc_figure_title

    note = None
    if not motion_applied:
        note = (
            "Motion correction NOT applied (input assumed pre-corrected); "
            "plotted values are residual motion, near the registration noise floor."
        )

    motion_params = np.loadtxt(motion_params_file)
    out_path = Path.cwd() / "dce_motion_qc.png"
    plot_motion_qc(
        motion_params, out_path, tr=tr,
        title=qc_figure_title(subject_id, session_id, "DCE Motion QC"),
        note=note,
    )
    return str(out_path)


def _generate_preprocessing_qc(
    dce_file: str,
    brain_mask_file: str,
    baseline_frames: int,
    bolus_arrival_frame: int,
    tr_ms: float | None,
    volume_tr_s: float | None,
    te: object = None,
    subject_id: str = "",
    session_id: str = "",
) -> tuple[str, dict[str, object]]:
    """Generate DCE preprocessing QC plot.

    Thin wrapper that loads NIfTI data and delegates to
    generate_preprocessing_qc_plot().

    Parameters
    ----------
    dce_file : str
        Path to 4D motion-corrected DCE NIfTI.
    brain_mask_file : str
        Path to brain mask NIfTI.
    baseline_frames : int
        Number of baseline frames.
    bolus_arrival_frame : int
        Bolus arrival frame index.
    tr_ms : float
        Pulse TR in seconds (from inputnode ``tr``).
    volume_tr_s : float
        Volume TR in seconds.
    te : float, optional
        Echo time in seconds (optional; surfaced as ms in QC row).
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.

    Returns
    -------
    tuple
        Tuple of (plot_file path, preproc_display_stats dict).
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]

    from pmriprep.utils.dce_pk_qc_plots import generate_preprocessing_qc_plot

    dce_img = nib.load(dce_file)
    dce_data = dce_img.get_fdata()  # type: ignore[attr-defined]
    brain_mask = nib.load(brain_mask_file).get_fdata() > 0.5  # type: ignore[attr-defined]

    _zooms = dce_img.header.get_zooms()  # type: ignore[attr-defined]
    voxel_size_mm = (
        float(_zooms[0]), float(_zooms[1]), float(_zooms[2]),
    )

    # Convert TR from seconds (inputnode) to milliseconds
    tr_ms_val = tr_ms * 1000.0 if tr_ms is not None else 0.0
    volume_tr_val = volume_tr_s if volume_tr_s is not None else 1.0
    # isinstance guard defends against the Nipype ``Undefined`` sentinel,
    # which is not None but is also not a number — matches DSC's pattern.
    te_ms_val = te * 1000.0 if isinstance(te, (int, float)) else None

    out_path = str(Path(dce_file).parent / "dce_preprocessing_qc.png")

    plot_file, stats = generate_preprocessing_qc_plot(
        dce_signal_4d=dce_data,
        brain_mask=brain_mask,
        baseline_frames=baseline_frames,
        bolus_arrival_frame=bolus_arrival_frame,
        volume_tr_s=volume_tr_val,
        subject_id=subject_id,
        session_id=session_id,
        output_path=out_path,
        voxel_size_mm=voxel_size_mm,
    )

    preproc_display_stats: dict[str, object] = {
        "tr_ms": tr_ms_val,
        "volume_tr_s": volume_tr_val,
        "n_volumes": dce_data.shape[3],
        **stats,
    }
    if te_ms_val is not None:
        preproc_display_stats["te_ms"] = te_ms_val
    return plot_file, preproc_display_stats


def _create_dce_registration_filmstrip(
    dce_mean_file: str | None = None,
    registered_t1w_file: str | None = None,
    brain_mask_file: str | None = None,
    registered_t1w_mask_file: str | None = None,
    subject_id: str = "",
    session_id: str = "",
) -> str | None:
    """Create registration QC filmstrip for DCE-T1w registration.

    Generates a multi-slice filmstrip showing the DCE mean image with
    contours of the registered T1w image overlaid. This allows visual
    assessment of registration quality.

    Parameters
    ----------
    dce_mean_file : str or None
        Path to DCE mean image.
    registered_t1w_file : str or None
        Path to T1w image registered to DCE space.
    brain_mask_file : str or None
        Path to the native DCE brain mask (optional).
    registered_t1w_mask_file : str or None
        Path to the anatomical brain mask transformed into DCE space.
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.

    Returns
    -------
    str or None
        Path to registration QC filmstrip PNG, or None if inputs missing.
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]

    from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip, qc_figure_title

    if dce_mean_file is None or registered_t1w_file is None:
        return None

    if not Path(dce_mean_file).is_file():
        return None
    if not Path(registered_t1w_file).is_file():
        return None

    # Load data
    dce_img = nib.load(dce_mean_file)
    dce_data = dce_img.get_fdata()  # type: ignore[attr-defined]
    voxel_sizes = tuple(dce_img.header.get_zooms()[:3])  # type: ignore[attr-defined]

    t1w_img = nib.load(registered_t1w_file)
    t1w_data = t1w_img.get_fdata()  # type: ignore[attr-defined]

    # Load mask if provided
    mask = None
    if brain_mask_file is not None and Path(brain_mask_file).is_file():
        mask_img = nib.load(brain_mask_file)
        mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    moving_mask = None
    if (
        registered_t1w_mask_file is not None
        and Path(registered_t1w_mask_file).is_file()
    ):
        moving_mask_img = nib.load(registered_t1w_mask_file)
        moving_mask = moving_mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    # Generate filmstrip — DCE as background, T1w edges overlaid
    out_path = Path.cwd() / "registration_qc_filmstrip.png"
    plot_registration_qc_filmstrip(
        fixed=dce_data,
        moving=t1w_data,
        output_path=out_path,
        mask=mask,
        moving_mask=moving_mask,
        title=qc_figure_title(subject_id, session_id, "DCE-T1w Registration"),
        voxel_sizes=voxel_sizes,
    )

    return str(out_path)


def _smooth_dce_maps(
    map_files: dict[str, str],
    brain_mask_file: str | None,
    fwhm: float,
) -> dict[str, str]:
    """Apply Gaussian smoothing to a dict of DCE maps.

    Uses normalized convolution for maps containing NaN (PK maps in auto
    mode, wash-in/out slopes). This prevents edge dilution where unfitted
    voxels would otherwise contribute 0 to the Gaussian kernel.

    Parameters
    ----------
    map_files : dict
        Dict mapping map name to file path.
    brain_mask_file : str
        Path to brain mask for re-masking after smoothing.
    fwhm : float
        Smoothing FWHM in mm.

    Returns
    -------
    dict
        Dict mapping map name to smoothed file path.
    """
    import logging
    import subprocess
    from pathlib import Path

    import nibabel as nib
    import numpy as np
    from nipype.interfaces.base import isdefined

    logger = logging.getLogger("nipype.workflow")

    # Maps that are non-negative by definition
    _NON_NEGATIVE_MAPS = {
        "ktrans", "ve", "vp", "kep",
        "iauc60", "iauc90", "iauc120",
        "peak_enhancement", "ser",
    }

    results: dict[str, str] = {}
    if not map_files:
        return results
    if not brain_mask_file or not isdefined(brain_mask_file):
        return results
    if not Path(brain_mask_file).is_file():
        return results

    sigma = fwhm / 2.355
    for name, filepath in map_files.items():
        if not filepath or not isdefined(filepath):
            continue
        if isinstance(filepath, str) and filepath in ("", "None"):
            continue
        if not Path(filepath).is_file():
            continue

        out_path = str(Path.cwd() / f"{Path(filepath).stem}_fwhm{int(fwhm)}.nii.gz")

        # Check for NaN values
        img = nib.load(filepath)
        data = img.get_fdata()  # type: ignore[attr-defined]
        has_nan = np.any(np.isnan(data))

        if has_nan:
            logger.info(
                "Normalized convolution for '%s' (NaN voxels present)", name,
            )
            # Normalized convolution: smooth(data) / smooth(valid_mask)
            valid_mask = np.isfinite(data).astype(np.float32)
            vmask_path = str(Path.cwd() / f"{name}_vmask.nii.gz")
            nib.save(nib.Nifti1Image(valid_mask, img.affine, img.header), vmask_path)  # type: ignore[attr-defined]

            num_path = str(Path.cwd() / f"{name}_num.nii.gz")
            den_path = str(Path.cwd() / f"{name}_den.nii.gz")

            thr_args = ["-thr", "0"] if name in _NON_NEGATIVE_MAPS else []

            try:
                # Numerator: NaN→0, smooth
                subprocess.run(
                    ["fslmaths", filepath, "-nan", "-s", str(sigma), num_path],
                    check=True, capture_output=True,
                )
                # Denominator: smooth valid mask
                subprocess.run(
                    ["fslmaths", vmask_path, "-s", str(sigma), den_path],
                    check=True, capture_output=True,
                )
                # Normalize; -nan cleans 0/0→NaN; conditionally threshold
                preserve_support = name in {"ser", "wash_in_slope", "wash_out_slope"}
                support_args = ["-mas", vmask_path] if preserve_support else []
                subprocess.run(
                    ["fslmaths", num_path, "-div", den_path,
                     "-nan", *thr_args, "-mas", brain_mask_file,
                     *support_args, out_path],
                    check=True, capture_output=True,
                )
            finally:
                for tmp in [vmask_path, num_path, den_path]:
                    Path(tmp).unlink(missing_ok=True)
        else:
            # No NaN: simple smoothing + brain mask
            subprocess.run(
                ["fslmaths", filepath, "-s", str(sigma),
                 "-mas", brain_mask_file, out_path],
                check=True, capture_output=True,
            )

        results[name] = out_path
    return results


def _build_pk_map_dict(
    ktrans: str | None = None,
    ve: str | None = None,
    vp: str | None = None,
) -> dict[str, str]:
    """Build dictionary of independently estimated PK maps for smoothing."""
    from nipype.interfaces.base import isdefined

    d: dict[str, str] = {}
    if ktrans and isdefined(ktrans):
        d["ktrans"] = ktrans
    if ve and isdefined(ve):
        d["ve"] = ve
    if vp and isdefined(vp):
        d["vp"] = vp
    return d


def _derive_smoothed_kep(
    smoothed_pk: dict[str, str] | None,
    native_kep_file: str | None,
    brain_mask_file: str | None,
    epsilon_ve: float,
    max_kep: float,
) -> dict[str, str]:
    """Add Kep derived from the spatially smoothed Ktrans and ve maps.

    Parameters
    ----------
    smoothed_pk : dict or None
        Smoothed PK-map paths containing ``ktrans`` and ``ve``.
    native_kep_file : str or None
        Native Kep map used only to preserve the voxelwise domain where Kep
        is defined, notably excluding Patlak voxels in automatic model mode.
    brain_mask_file : str or None
        DCE-space brain mask used to restrict the derived map.
    epsilon_ve : float
        Minimum valid ve denominator.
    max_kep : float
        Maximum reported Kep in inverse minutes.

    Returns
    -------
    dict
        Copy of ``smoothed_pk`` with a derived ``kep`` path when its inputs
        are available and geometrically compatible.
    """
    from pathlib import Path
    from typing import cast

    import nibabel as nib
    import numpy as np
    from nipype.interfaces.base import isdefined

    result = dict(smoothed_pk or {})
    ktrans_file = result.get("ktrans")
    ve_file = result.get("ve")
    if not ktrans_file or not ve_file:
        return result
    if not Path(ktrans_file).is_file() or not Path(ve_file).is_file():
        return result

    ktrans_img = cast("nib.Nifti1Image", nib.load(ktrans_file))
    ve_img = cast("nib.Nifti1Image", nib.load(ve_file))
    ktrans = ktrans_img.get_fdata()  # type: ignore[attr-defined]
    ve = ve_img.get_fdata()  # type: ignore[attr-defined]
    if ktrans.shape != ve.shape or not np.allclose(ktrans_img.affine, ve_img.affine):
        raise ValueError("Smoothed Ktrans and ve maps must have matching geometry")

    valid = np.isfinite(ktrans) & np.isfinite(ve) & (ve >= epsilon_ve)
    if native_kep_file and isdefined(native_kep_file) and Path(native_kep_file).is_file():
        native_kep_img = cast("nib.Nifti1Image", nib.load(native_kep_file))
        native_kep = native_kep_img.get_fdata()  # type: ignore[attr-defined]
        if native_kep.shape != ktrans.shape or not np.allclose(
            native_kep_img.affine, ktrans_img.affine
        ):
            raise ValueError("Native Kep and smoothed PK maps must have matching geometry")
        valid &= np.isfinite(native_kep)
    brain_mask = np.ones(ktrans.shape, dtype=bool)
    if brain_mask_file and isdefined(brain_mask_file) and Path(brain_mask_file).is_file():
        mask_img = cast("nib.Nifti1Image", nib.load(brain_mask_file))
        brain_mask = mask_img.get_fdata() > 0  # type: ignore[attr-defined]
        if brain_mask.shape != ktrans.shape or not np.allclose(
            mask_img.affine, ktrans_img.affine
        ):
            raise ValueError("Brain mask and smoothed PK maps must have matching geometry")
        valid &= brain_mask

    kep = np.full(ktrans.shape, np.nan, dtype=np.float32)
    np.divide(ktrans, ve, out=kep, where=valid)
    np.clip(kep, 0.0, max_kep, out=kep)
    kep[~brain_mask] = 0.0
    out_path = Path.cwd() / f"{Path(ktrans_file).stem}_derived_kep.nii.gz"
    header = ktrans_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(kep, ktrans_img.affine, header), out_path)  # type: ignore[attr-defined]
    result["kep"] = str(out_path)
    return result


def _build_model_free_map_dict(
    iauc60: str | None = None,
    iauc90: str | None = None,
    iauc120: str | None = None,
    peak: str | None = None,
    ser: str | None = None,
    wash_in_slope: str | None = None,
    wash_out_slope: str | None = None,
) -> dict[str, str]:
    """Build dictionary of model-free maps for smoothing.

    TTP and BAT are discrete timing maps, and curve type is categorical; none
    are included because smoothing would create non-physical intermediate
    values.
    """
    from nipype.interfaces.base import isdefined

    d: dict[str, str] = {}
    if iauc60 and isdefined(iauc60):
        d["iauc60"] = iauc60
    if iauc90 and isdefined(iauc90):
        d["iauc90"] = iauc90
    if iauc120 and isdefined(iauc120):
        d["iauc120"] = iauc120
    if peak and isdefined(peak):
        d["peak_enhancement"] = peak
    if ser and isdefined(ser):
        d["ser"] = ser
    if wash_in_slope and isdefined(wash_in_slope):
        d["wash_in_slope"] = wash_in_slope
    if wash_out_slope and isdefined(wash_out_slope):
        d["wash_out_slope"] = wash_out_slope
    return d


def _normalize_by_nawm(
    param_map: str | None = None,
    nawm_mask: str | None = None,
) -> str:
    """Normalize parameter map by NAWM median.

    Median is used (matching DSC nCBV in ``utils/dsc_signal.py``) so the
    normalization is robust to a small fraction of non-WM voxels that FSL
    FAST classifies as WM (notably globus pallidus, parts of thalamus and
    hippocampus, where iron- and myelin-rich T1 intensity matches WM).

    Parameters
    ----------
    param_map : str or None
        Path to parameter map NIfTI file.
    nawm_mask : str or None
        Path to NAWM mask NIfTI file in the same space.

    Returns
    -------
    str
        Path to normalized parameter map, or ``""`` if normalization fails.
        The empty string is compatible with downstream ``traits.Str`` inputs.
    """
    from pathlib import Path

    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np
    from nipype.interfaces.base import isdefined

    if not isdefined(nawm_mask) or nawm_mask is None:
        return ""
    if not isdefined(param_map) or param_map is None:
        return ""
    if not Path(nawm_mask).is_file():
        return ""
    if not Path(param_map).is_file():
        return ""

    param_img = nib.load(param_map)
    nawm_img = nib.load(nawm_mask)

    param_data = param_img.get_fdata()  # type: ignore[attr-defined]
    nawm_data = nawm_img.get_fdata() > 0  # type: ignore[attr-defined]

    # Check shapes match
    if param_data.shape[:3] != nawm_data.shape[:3]:
        return ""

    nawm_values = param_data[nawm_data]
    nawm_values = nawm_values[np.isfinite(nawm_values) & (nawm_values > 0)]
    if len(nawm_values) < 10:
        # Not enough valid NAWM voxels
        return ""

    nawm_median = float(np.median(nawm_values))
    if nawm_median <= 0:
        return ""

    # Normalize
    normalized = param_data / nawm_median

    # Save
    param_stem = Path(param_map).name.replace(".nii.gz", "").replace(".nii", "")
    out_file = str(Path.cwd() / f"n{param_stem}.nii.gz")
    nib.save(
        nib.Nifti1Image(normalized.astype(np.float32), param_img.affine),  # type: ignore[attr-defined]
        out_file,
    )
    return out_file


def _generate_dce_pk_maps_qc(
    ktrans_file: str,
    ve_file: str | None = None,
    vp_file: str | None = None,
    fp_file: str | None = None,
    ps_file: str | None = None,
    brain_mask_file: str | None = None,
    qc_mask_file: str | None = None,
    model_selection_file: str | None = None,
    delta_bic_file: str | None = None,
    enhancing_mask_file: str | None = None,
    non_enhancing_mask_file: str | None = None,
    nawm_mask_file: str | None = None,
    user_lesion_mask_file: str | None = None,
    roi_mask_file: str | None = None,
    vp_at_bound_map_file: str | None = None,
    model_name: str = "tofts",
    subject_id: str = "",
    session_id: str = "",
) -> tuple[
    str | None,
    str | None,
    dict[str, dict[str, float]] | None,
    dict[str, float] | None,
]:
    """Generate DCE PK parameter maps QC and model selection QC plots.

    Lightweight wrapper that loads saved NIfTIs and calls plot functions.
    Follows the DSC QC wrapper pattern (dsc/base.py).

    Parameters
    ----------
    ktrans_file : str
        Path to Ktrans map NIfTI.
    ve_file : str, optional
        Path to ve map NIfTI (may be undefined for patlak).
    vp_file : str, optional
        Path to vp map NIfTI (may be undefined for tofts).
    fp_file : str, optional
        Path to Fp map NIfTI (may be undefined for non-2cxm).
    ps_file : str, optional
        Path to PS map NIfTI (may be undefined for non-2cxm).
    brain_mask_file : str, optional
        Path to brain mask NIfTI.
    qc_mask_file : str, optional
        Path to quantitative QC mask NIfTI. Its acceptance fraction is reported
        separately from the finite parameter support shown in the montage.
    model_selection_file : str, optional
        Path to model selection map NIfTI (auto mode).
    delta_bic_file : str, optional
        Path to delta-BIC map NIfTI (auto mode).
    enhancing_mask_file : str, optional
        Path to enhancing tumor mask NIfTI (optional).
    non_enhancing_mask_file : str, optional
        Path to non-enhancing tumor mask NIfTI (optional).
    nawm_mask_file : str, optional
        Path to NAWM mask NIfTI (optional).
    user_lesion_mask_file : str, optional
        Path to user-supplied lesion mask NIfTI (optional).
    roi_mask_file : str, optional
        Path to user-supplied ROI mask NIfTI (optional).
    vp_at_bound_map_file : str, optional
        Path to the vp-at-bound map NIfTI (optional).
    model_name : str, optional
        PK model name.
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.

    Returns
    -------
    tuple
        Tuple of (pk_param_maps_qc_plot, model_selection_qc_plot,
        model_selection_regions, vp_at_bound_fractions).
        ``model_selection_regions`` is a pure ``{region: {model: pct}}``
        mapping; ``vp_at_bound_fractions`` carries scalar whole-brain and
        per-ROI vp-at-bound fractions separately so downstream consumers
        (report template, sidecar writer) can rely on a homogeneous shape.
    """
    import logging

    from pathlib import Path as _P

    import nibabel as nib
    import numpy as np

    logger = logging.getLogger("pmriprep.workflows.dce")

    def _safe_load(filepath: str | None) -> "np.ndarray | None":
        if not filepath or not _P(str(filepath)).is_file():
            return None
        return nib.load(filepath).get_fdata()  # type: ignore[attr-defined,no-any-return]

    pk_param_maps_qc_plot: str | None = None
    model_selection_qc_plot: str | None = None
    model_selection_regions: dict[str, dict[str, float]] | None = None
    vp_at_bound_fractions: dict[str, float] = {}

    try:
        from pmriprep.utils.dce_pk_qc_plots import (
            generate_model_selection_qc_plot,
            generate_pk_parameter_maps_qc_plot,
        )

        if not brain_mask_file:
            raise ValueError("brain_mask_file is required for DCE PK maps QC")
        mask_data = nib.load(brain_mask_file).get_fdata()  # type: ignore[attr-defined]
        qc_mask_data = _safe_load(qc_mask_file)
        if qc_mask_data is not None and qc_mask_data.shape != mask_data.shape:
            raise ValueError(
                "DCE QC mask shape does not match the brain mask: "
                f"{qc_mask_data.shape} != {mask_data.shape}"
            )

        # Build param_maps dict from available files
        param_maps = {}
        ktrans_data = _safe_load(ktrans_file)
        if ktrans_data is not None:
            param_maps["Ktrans"] = ktrans_data

        ve_data = _safe_load(ve_file)
        if ve_data is not None:
            param_maps["ve"] = ve_data

        vp_data = _safe_load(vp_file)
        if vp_data is not None:
            param_maps["vp"] = vp_data

        fp_data = _safe_load(fp_file)
        if fp_data is not None:
            param_maps["Fp"] = fp_data

        ps_data = _safe_load(ps_file)
        if ps_data is not None:
            param_maps["PS"] = ps_data

        # Generate parameter maps QC plot
        pk_param_maps_qc_plot = generate_pk_parameter_maps_qc_plot(
            param_maps=param_maps,
            brain_mask=mask_data,
            qc_mask=qc_mask_data,
            subject_id=subject_id,
            session_id=session_id,
        )

        # Generate model selection QC plot (auto mode only)
        model_sel_data = _safe_load(model_selection_file)
        delta_bic_data = _safe_load(delta_bic_file)

        if model_sel_data is not None and delta_bic_data is not None:
            enh_data = _safe_load(enhancing_mask_file)
            if enh_data is not None:
                enh_data = (enh_data > 0.5).astype(np.float64)

            ne_data = _safe_load(non_enhancing_mask_file)
            if ne_data is not None:
                ne_data = (ne_data > 0.5).astype(np.float64)

            nawm_data = _safe_load(nawm_mask_file)
            if nawm_data is not None:
                nawm_data = (nawm_data > 0.5).astype(np.float64)

            ul_data = _safe_load(user_lesion_mask_file)
            if ul_data is not None:
                ul_data = (ul_data > 0.5).astype(np.float64)

            roi_data = _safe_load(roi_mask_file)
            if roi_data is not None:
                roi_data = (roi_data > 0.5).astype(np.float64)

            regional_stats: dict[str, dict[str, float]] = {}
            model_selection_qc_plot = generate_model_selection_qc_plot(
                model_selection_map=model_sel_data.astype(np.int8),
                delta_bic_map=delta_bic_data.astype(np.float32),
                brain_mask=mask_data,
                enhancing_mask=enh_data,
                non_enhancing_mask=ne_data,
                nawm_mask=nawm_data,
                user_lesion_mask=ul_data,
                roi_mask=roi_data,
                subject_id=subject_id,
                session_id=session_id,
                regional_stats_out=regional_stats,
            )
            if regional_stats:
                model_selection_regions = regional_stats

        # Stratify vp_at_bound map by ROI masks already loaded above
        # (enh_data / ne_data / nawm_data). Only emitted when the
        # vp_at_bound NIfTI exists (ComputeDCEMaps always writes it;
        # non-zero only for SOFT NLLS runs). Kept in a dedicated
        # ``vp_at_bound_fractions`` output so ``model_selection_regions``
        # stays a homogeneous region→model-dist mapping.
        vp_at_bound_data = _safe_load(vp_at_bound_map_file)
        if vp_at_bound_data is not None:
            brain_bool = mask_data > 0.5
            vpab_bool = vp_at_bound_data > 0.5
            finite = np.isfinite(vp_at_bound_data)
            brain_finite = brain_bool & finite
            if brain_finite.any():
                vp_at_bound_fractions["whole_brain"] = float(
                    np.mean(vpab_bool[brain_finite])
                )
            else:
                vp_at_bound_fractions["whole_brain"] = 0.0

            def _frac_in_roi(roi_bool: "np.ndarray") -> float | None:
                sel = roi_bool & finite
                return float(np.mean(vpab_bool[sel])) if sel.any() else None

            # Load ROI masks if model-selection branch didn't already
            if "enh_data" not in locals():
                enh_data = _safe_load(enhancing_mask_file)
                if enh_data is not None:
                    enh_data = (enh_data > 0.5).astype(np.float64)
                ne_data = _safe_load(non_enhancing_mask_file)
                if ne_data is not None:
                    ne_data = (ne_data > 0.5).astype(np.float64)
                nawm_data = _safe_load(nawm_mask_file)
                if nawm_data is not None:
                    nawm_data = (nawm_data > 0.5).astype(np.float64)

            if enh_data is not None:
                val = _frac_in_roi(enh_data > 0.5)
                if val is not None:
                    vp_at_bound_fractions["enhancing"] = val
            if ne_data is not None:
                val = _frac_in_roi(ne_data > 0.5)
                if val is not None:
                    vp_at_bound_fractions["tumor_core"] = val
            if nawm_data is not None:
                val = _frac_in_roi(nawm_data > 0.5)
                if val is not None:
                    vp_at_bound_fractions["nawm"] = val

    except Exception as e:
        logger.warning("Failed to generate PK maps QC plots: %s", e, exc_info=True)

    return (
        pk_param_maps_qc_plot,
        model_selection_qc_plot,
        model_selection_regions,
        vp_at_bound_fractions or None,
    )


def _generate_dce_pk_fit_qc(
    concentration_file: str,
    dce_signal_file: str | None = None,
    aif_file: str | None = None,
    brain_mask_file: str | None = None,
    ktrans_file: str | None = None,
    ve_file: str | None = None,
    vp_file: str | None = None,
    fp_file: str | None = None,
    ps_file: str | None = None,
    r2_file: str | None = None,
    fitting_mask_file: str | None = None,
    optimizer_converged_mask_file: str | None = None,
    quality_flag_file: str | None = None,
    qc_mask_file: str | None = None,
    fitted_curves_file: str | None = None,
    model_selection_file: str | None = None,
    enhancing_mask_file: str | None = None,
    non_enhancing_mask_file: str | None = None,
    nawm_mask_file: str | None = None,
    volume_tr: float = 1.0,
    baseline_frames: int = 10,
    model_name: str = "tofts",
    subject_id: str = "",
    session_id: str = "",
) -> tuple[str | None, dict[str, object]]:
    """Generate DCE PK modeling fit QC plot.

    Lightweight wrapper that loads saved NIfTIs, recomputes the single-voxel
    fitted curve, and calls the QC plot function. Follows the DSC QC wrapper
    pattern (dsc/base.py).

    Parameters
    ----------
    concentration_file : str
        Path to 4D concentration NIfTI.
    dce_signal_file : str, optional
        Path to 4D raw DCE signal NIfTI.
    aif_file : str, optional
        Path to AIF curve (.npy or NIfTI).
    brain_mask_file : str, optional
        Path to brain mask NIfTI.
    ktrans_file : str, optional
        Path to Ktrans map NIfTI.
    ve_file : str, optional
        Path to ve map NIfTI (may be undefined).
    vp_file : str, optional
        Path to vp map NIfTI (may be undefined).
    fp_file : str, optional
        Path to Fp map NIfTI (may be undefined).
    ps_file : str, optional
        Path to PS map NIfTI (may be undefined).
    r2_file : str, optional
        Path to R² map NIfTI.
    fitting_mask_file : str, optional
        Path to the PK fitting-eligibility mask.
    optimizer_converged_mask_file : str, optional
        Path to the optimizer-convergence mask.
    quality_flag_file : str, optional
        Path to the retained-fit categorical quality map.
    qc_mask_file : str, optional
        Path to QC mask NIfTI.
    fitted_curves_file : str, optional
        Path to 4D fitted curves NIfTI (optional).
    model_selection_file : str, optional
        Path to model selection map NIfTI (auto mode).
    enhancing_mask_file : str, optional
        Path to enhancing tumor mask NIfTI (optional).
    non_enhancing_mask_file : str, optional
        Path to non-enhancing tumor mask NIfTI (optional).
    nawm_mask_file : str, optional
        Path to the NAWM mask NIfTI (optional).
    volume_tr : float, optional
        Volume TR in seconds.
    baseline_frames : int, optional
        Number of baseline frames.
    model_name : str, optional
        PK model name.
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.

    Returns
    -------
    tuple
        Tuple of (pk_modeling_qc_plot path, pk_voxel_metrics dict).
    """
    import logging

    from pathlib import Path as _P

    import nibabel as nib
    import numpy as np

    logger = logging.getLogger("pmriprep.workflows.dce")

    def _safe_load(filepath: str | None) -> "np.ndarray | None":
        if not filepath or not _P(str(filepath)).is_file():
            return None
        return nib.load(filepath).get_fdata()  # type: ignore[attr-defined,no-any-return]

    if not brain_mask_file or not aif_file:
        return None, {}

    try:
        from pmriprep.utils.dce_pk_qc_plots import (
            _select_qc_voxel,
            generate_pk_modeling_qc_plot,
        )

        # Load required data
        conc_data = nib.load(concentration_file).get_fdata()  # type: ignore[attr-defined]
        mask_data = nib.load(brain_mask_file).get_fdata()  # type: ignore[attr-defined]
        mask_bool = mask_data > 0.5

        n_timepoints = conc_data.shape[3]
        time_points = np.arange(n_timepoints) * (volume_tr / 60.0)  # seconds -> minutes

        # Load AIF (always a 1-D text file from AIF detection)
        aif_path = _P(str(aif_file))
        if aif_path.suffix == ".npy":
            aif = np.load(str(aif_path))
        elif aif_path.suffix == ".npz":
            npz = np.load(str(aif_path))
            aif = npz.get("Cp", npz.get("aif", next(iter(npz.values()))))
        elif aif_path.suffix in (".txt", ".csv", ".tsv"):
            aif = np.loadtxt(str(aif_path))
        else:
            # Last resort: try as text, then as NIfTI
            try:
                aif = np.loadtxt(str(aif_path))
            except ValueError:
                aif = nib.load(str(aif_path)).get_fdata().flatten()  # type: ignore[attr-defined]

        # Load parameter maps
        ktrans = _safe_load(ktrans_file)
        if ktrans is None:
            return None, {}

        is_auto = model_name == "auto"
        ve = _safe_load(ve_file)
        vp = _safe_load(vp_file)
        fp = _safe_load(fp_file)
        ps = _safe_load(ps_file)
        r2_map = _safe_load(r2_file)
        if r2_map is None:
            r2_map = np.zeros(ktrans.shape, dtype=np.float32)
        qc_mask = _safe_load(qc_mask_file)
        if qc_mask is not None:
            qc_mask = qc_mask.astype(np.uint8)
        fitting_mask_data = _safe_load(fitting_mask_file)
        optimizer_converged_data = _safe_load(
            optimizer_converged_mask_file
        )
        quality_flag_data = _safe_load(quality_flag_file)

        signal_data = _safe_load(dce_signal_file)
        fitted_curves = _safe_load(fitted_curves_file)

        # Load enhancing mask
        enhancing_mask_data = None
        enh_raw = _safe_load(enhancing_mask_file)
        if enh_raw is not None:
            enhancing_mask_data = (enh_raw > 0.5).astype(np.float64)
        non_enhancing_mask_data = None
        ne_raw = _safe_load(non_enhancing_mask_file)
        if ne_raw is not None:
            non_enhancing_mask_data = (ne_raw > 0.5).astype(np.float64)
        nawm_mask_data = None
        nawm_raw = _safe_load(nawm_mask_file)
        if nawm_raw is not None:
            nawm_mask_data = (nawm_raw > 0.5).astype(np.float64)

        # Recompute single-voxel fitted curve
        qc_fitted_curve_1d = None
        # Preload model-selection map so downstream stratified-Patlak code
        # can reference it even if the inner fit-replay block raises.
        model_sel_data = _safe_load(model_selection_file)
        try:
            from scipy.ndimage import binary_erosion

            from pmriprep.models.dce import (
                ExtendedToftsModel,
                PatlakModel,
                ToftsModel,
                TwoCompartmentExchangeModel,
            )

            qc_vx, qc_vy, qc_vz = _select_qc_voxel(
                ktrans, mask_bool, qc_mask, binary_erosion,
                enhancing_mask=enhancing_mask_data,
            )

            model_cls_map = {
                "tofts": ToftsModel,
                "extended_tofts": ExtendedToftsModel,
                "patlak": PatlakModel,
                "2cxm": TwoCompartmentExchangeModel,
            }

            # Determine model for the QC voxel
            _CODE_TO_MODEL = {1: "patlak", 2: "tofts", 3: "extended_tofts", 4: "2cxm"}

            if is_auto and model_sel_data is not None:
                qc_code = int(model_sel_data[qc_vx, qc_vy, qc_vz])
                qc_model_name = _CODE_TO_MODEL.get(qc_code, model_name)
            else:
                qc_model_name = model_name

            if qc_model_name in model_cls_map:
                qc_model = model_cls_map[qc_model_name]()  # type: ignore[abstract]
                param_map = {"Ktrans": ktrans, "ve": ve, "vp": vp, "Fp": fp, "PS": ps}
                qc_params = []
                for n in qc_model.get_parameter_names():
                    arr = param_map.get(n)
                    if arr is not None:
                        qc_params.append(float(arr[qc_vx, qc_vy, qc_vz]))
                    else:
                        qc_params.append(0.0)
                qc_fitted_curve_1d = qc_model.evaluate(
                    qc_params, aif, time_points,
                ).astype(np.float32)
        except Exception:
            pass  # Fall back to fitted_curves array in the plot function

        # NaN-safe maps for display
        ktrans_display = np.nan_to_num(ktrans, nan=0.0) if is_auto else ktrans
        ve_display = np.nan_to_num(ve, nan=0.0) if is_auto and ve is not None else ve
        vp_display = np.nan_to_num(vp, nan=0.0) if is_auto and vp is not None else vp

        pk_modeling_qc_path, pk_voxel_metrics = generate_pk_modeling_qc_plot(
            conc_data=conc_data,
            signal_data=signal_data,
            aif=aif,
            time_points_s=time_points * 60.0,  # min -> sec
            ktrans=ktrans_display,
            ve=ve_display,
            vp=vp_display,
            r2_map=r2_map,
            brain_mask=mask_bool.astype(np.float64),
            fitted_curves=fitted_curves,
            model_name=model_name,
            baseline_frames=baseline_frames,
            qc_mask=qc_mask,
            qc_fitted_curve=qc_fitted_curve_1d,
            fp=fp,
            ps=ps,
            enhancing_mask=enhancing_mask_data,
            subject_id=subject_id,
            session_id=session_id,
        )

        if (
            fitting_mask_data is not None
            and optimizer_converged_data is not None
            and quality_flag_data is not None
            and qc_mask is not None
        ):
            from pmriprep.utils.dce_voxel_prep import (
                summarize_regional_voxel_accounting,
            )

            stage_masks = {
                "fitting_eligible": fitting_mask_data > 0.5,
                "optimizer_converged": optimizer_converged_data > 0.5,
                "retained_nondegenerate": quality_flag_data > 0,
                "quantitative_qc_accepted": qc_mask > 0,
            }
            regions = {
                "enhancing_tumor": (
                    enhancing_mask_data > 0.5
                    if enhancing_mask_data is not None
                    else None
                ),
                "non_enhancing_tumor": (
                    non_enhancing_mask_data > 0.5
                    if non_enhancing_mask_data is not None
                    else None
                ),
                "nawm": (
                    nawm_mask_data > 0.5
                    if nawm_mask_data is not None
                    else None
                ),
            }
            regional_accounting = summarize_regional_voxel_accounting(
                mask_bool,
                stage_masks,
                regions,
            )
            if regional_accounting:
                pk_voxel_metrics["voxel_accounting_regions"] = (
                    regional_accounting
                )

        # Calculate the enhancing-tumor peak-concentration risk metrics.
        peak_ct = np.nanmax(conc_data, axis=-1)

        if enhancing_mask_data is not None:
            enh_bool = enhancing_mask_data > 0.5
            if np.any(enh_bool):
                tumor_peak = peak_ct[enh_bool]
                tumor_peak = tumor_peak[np.isfinite(tumor_peak)]
                if tumor_peak.size > 0:
                    frac_fxl_tumor = float(np.mean(tumor_peak > 0.8))
                    pk_voxel_metrics["frac_fxl_risk_tumor"] = frac_fxl_tumor
                    pk_voxel_metrics["p95_tumor_peak_Ct"] = float(
                        np.percentile(tumor_peak, 95)
                    )
                    if frac_fxl_tumor > 0.15:
                        pk_voxel_metrics["fxl_status"] = "FAIL"
                    elif frac_fxl_tumor > 0.02:
                        pk_voxel_metrics["fxl_status"] = "WARN"
                    else:
                        pk_voxel_metrics["fxl_status"] = "PASS"
                else:
                    pk_voxel_metrics["frac_fxl_risk_tumor"] = None
                    pk_voxel_metrics["p95_tumor_peak_Ct"] = None
                    pk_voxel_metrics["fxl_status"] = "PASS"
            else:
                pk_voxel_metrics["frac_fxl_risk_tumor"] = None
                pk_voxel_metrics["p95_tumor_peak_Ct"] = None
                pk_voxel_metrics["fxl_status"] = "PASS"
        else:
            pk_voxel_metrics["frac_fxl_risk_tumor"] = None
            pk_voxel_metrics["p95_tumor_peak_Ct"] = None
            pk_voxel_metrics["fxl_status"] = "PASS"

        # Calculate Patlak fractions separately within each tissue class.
        _PATLAK_CODE = 1  # from _MODEL_SELECTION_CODES in dce_compute.py
        if is_auto and model_sel_data is not None:
            ne_bool = (
                non_enhancing_mask_data > 0.5
                if non_enhancing_mask_data is not None
                else None
            )

            def _patlak_frac(region_mask: "np.ndarray") -> float | None:
                region_fitted = region_mask & (model_sel_data > 0)
                n_region = int(region_fitted.sum())
                if n_region == 0:
                    return None
                n_patlak = int(np.sum((model_sel_data == _PATLAK_CODE) & region_fitted))
                return n_patlak / n_region

            if enhancing_mask_data is not None:
                pk_voxel_metrics["patlak_fraction_enhancing_tumor"] = _patlak_frac(
                    enhancing_mask_data > 0.5
                )
            else:
                pk_voxel_metrics["patlak_fraction_enhancing_tumor"] = None

            if ne_bool is not None:
                pk_voxel_metrics["patlak_fraction_non_enh_tumor"] = _patlak_frac(ne_bool)
            else:
                pk_voxel_metrics["patlak_fraction_non_enh_tumor"] = None

            # NAWM/GM proxy = brain minus all tumor
            non_tumor = mask_bool.copy()
            if enhancing_mask_data is not None:
                non_tumor &= ~(enhancing_mask_data > 0.5)
            if ne_bool is not None:
                non_tumor &= ~ne_bool
            pk_voxel_metrics["patlak_fraction_nawm_gm"] = _patlak_frac(non_tumor)

        return pk_modeling_qc_path, pk_voxel_metrics

    except Exception as e:
        logger.warning("Failed to generate PK fit QC plot: %s", e, exc_info=True)
        return None, {}
