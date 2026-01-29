"""DSC (Dynamic Susceptibility Contrast) preprocessing workflow for PMRIPrep.

This module implements the main DSC perfusion processing workflow,
following the NiPreps/fMRIPrep workflow design patterns.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]
from nipype.interfaces.base import (  # type: ignore[import-untyped]
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
    isdefined,
)


# Custom FSL wrappers to avoid nipype FSL module import bug (LooseVersion issue)
class _ConvertXFMInputSpec(CommandLineInputSpec):
    """Input spec for convert_xfm."""
    # Explicit positions for correct FSL convert_xfm syntax:
    # -omat out.mat -inverse in.mat  OR  -omat out.mat -concat second.mat first.mat
    out_file = File(argstr="-omat %s", position=0, genfile=True, desc="Output matrix file")
    invert_xfm = traits.Bool(argstr="-inverse", position=1, xor=["concat_xfm"],
                             desc="Invert transform")
    concat_xfm = traits.Bool(argstr="-concat", position=1, xor=["invert_xfm"],
                             desc="Concatenate transforms")
    in_file2 = File(exists=True, argstr="%s", position=2,
                    desc="Second input for concatenation (second transform in chain)")
    in_file = File(exists=True, mandatory=True, argstr="%s", position=3,
                   desc="Primary input transform matrix (first transform in chain)")


class _ConvertXFMOutputSpec(TraitedSpec):
    """Output spec for convert_xfm."""
    out_file = File(exists=True, desc="Output transform matrix")


class ConvertXFM(CommandLine):
    """FSL convert_xfm - transform matrix manipulation."""
    _cmd = "convert_xfm"
    input_spec = _ConvertXFMInputSpec
    output_spec = _ConvertXFMOutputSpec

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        return None

    def _gen_outfilename(self):
        from pathlib import Path
        in_file = Path(self.inputs.in_file)
        if self.inputs.invert_xfm:
            suffix = "_inv"
        elif self.inputs.concat_xfm:
            suffix = "_concat"
        else:
            suffix = "_out"
        return str(Path.cwd() / f"{in_file.stem}{suffix}.mat")

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs["out_file"] = self.inputs.out_file
        else:
            outputs["out_file"] = self._gen_outfilename()
        return outputs


class _FLIRTApplyXFMInputSpec(CommandLineInputSpec):
    """Input spec for FLIRT applyxfm."""
    in_file = File(exists=True, mandatory=True, argstr="-in %s",
                   desc="Input file to transform")
    reference = File(exists=True, mandatory=True, argstr="-ref %s",
                     desc="Reference image for output geometry")
    in_matrix_file = File(exists=True, mandatory=True, argstr="-init %s",
                          desc="Transform matrix to apply")
    out_file = File(argstr="-out %s", genfile=True, desc="Output file")
    interp = traits.Enum("trilinear", "nearestneighbour", "sinc", "spline",
                         argstr="-interp %s", desc="Interpolation method")
    apply_xfm = traits.Bool(True, argstr="-applyxfm", usedefault=True,
                            desc="Apply transform (required)")


class _FLIRTApplyXFMOutputSpec(TraitedSpec):
    """Output spec for FLIRT applyxfm."""
    out_file = File(exists=True, desc="Transformed output file")


class FLIRTApplyXFM(CommandLine):
    """FSL FLIRT - apply transform to image."""
    _cmd = "flirt"
    input_spec = _FLIRTApplyXFMInputSpec
    output_spec = _FLIRTApplyXFMOutputSpec

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        return None

    def _gen_outfilename(self):
        from pathlib import Path
        in_file = Path(self.inputs.in_file)
        stem = in_file.name.replace(".nii.gz", "").replace(".nii", "")
        return str(Path.cwd() / f"{stem}_transformed.nii.gz")

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs["out_file"] = self.inputs.out_file
        else:
            outputs["out_file"] = self._gen_outfilename()
        return outputs

from pmriprep.config.dsc import DSCConfig
from pmriprep.data import get_arterial_atlas
from pmriprep.interfaces.dsc import MotionCorrection, DSCRegistration
from pmriprep.interfaces.dsc_aif import AIFDetection
from pmriprep.interfaces.dsc_compute import ComputeDSCMaps


def _extract_dsc_brain(dsc_mean_file, brain_mask_file, t1w_reference_file):
    """Create native DSC-space brain mask by transforming T1w mask to DSC space.

    Strategy: Quick FLIRT registration of T1w to DSC mean, then apply
    the resulting transform to the T1w brain mask using nearest-neighbour
    interpolation.

    Fallback: intensity-based thresholding if FLIRT registration fails.

    Args:
        dsc_mean_file: Path to mean DSC image (reference geometry).
        brain_mask_file: Path to brain mask in T1w space.
        t1w_reference_file: Path to T1w structural image.

    Returns:
        Path to DSC-space brain mask NIfTI file.
    """
    import os
    import subprocess
    import numpy as np
    import nibabel as nib
    from pathlib import Path

    out_dir = Path.cwd()
    dsc_brain_mask_file = str(out_dir / "dsc_brain_mask.nii.gz")

    try:
        # Quick FLIRT: register T1w to DSC mean (6DOF, normcorr for speed)
        quick_mat = str(out_dir / "_quick_t1w2dsc.mat")
        flirt_cmd = [
            "flirt",
            "-in", t1w_reference_file,
            "-ref", dsc_mean_file,
            "-omat", quick_mat,
            "-dof", "6", "-cost", "normcorr",
            "-searchrx", "-30", "30",
            "-searchry", "-30", "30",
            "-searchrz", "-30", "30",
            "-coarsesearch", "15", "-finesearch", "5",
        ]
        subprocess.run(flirt_cmd, check=True, capture_output=True)

        # Apply transform to brain mask with nearest-neighbour interpolation
        apply_cmd = [
            "flirt",
            "-in", brain_mask_file,
            "-ref", dsc_mean_file,
            "-out", dsc_brain_mask_file,
            "-init", quick_mat,
            "-applyxfm",
            "-interp", "nearestneighbour",
        ]
        subprocess.run(apply_cmd, check=True, capture_output=True)

        # Threshold at 0.5 and binarize
        thr_cmd = [
            "fslmaths", dsc_brain_mask_file,
            "-thr", "0.5", "-bin", dsc_brain_mask_file,
        ]
        subprocess.run(thr_cmd, check=True, capture_output=True)

        # Clean up temporary matrix
        if os.path.exists(quick_mat):
            os.remove(quick_mat)

        # Verify the mask has reasonable coverage
        mask_img = nib.load(dsc_brain_mask_file)
        mask_data = mask_img.get_fdata()
        mask_fraction = np.sum(mask_data > 0) / mask_data.size
        if mask_fraction < 0.01:
            raise RuntimeError(
                f"Transformed mask covers only {mask_fraction:.1%} of volume"
            )

    except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError, OSError):
        # Fallback: intensity-based thresholding on DSC mean
        # Note: RuntimeError from mask coverage check (<1%) intentionally
        # triggers this fallback — a near-empty mask means FLIRT failed.
        import warnings
        warnings.warn(
            "FLIRT-based brain mask transform failed. "
            "Falling back to intensity-based DSC brain mask."
        )
        dsc_img = nib.load(dsc_mean_file)
        dsc_data = dsc_img.get_fdata()
        # Use Otsu-like threshold: percentile-based
        nonzero = dsc_data[dsc_data > 0]
        if len(nonzero) > 0:
            threshold = np.percentile(nonzero, 10)
        else:
            threshold = 0.0
        mask_data = (dsc_data > threshold).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, dsc_img.affine, dsc_img.header)
        nib.save(mask_img, dsc_brain_mask_file)

    return dsc_brain_mask_file


def _signal_to_concentration(
    dsc_file,
    te,
    baseline_frames,
    r2_star,
    field_strength,
):
    """Convert DSC signal to concentration.

    This is a Function node wrapper for the signal_to_concentration utility.

    Args:
        dsc_file: Path to motion-corrected 4D DSC NIfTI.
        te: Echo time in seconds.
        baseline_frames: Number of pre-bolus baseline frames.
        r2_star: Optional r2* relaxivity for absolute quantification.
        field_strength: Optional field strength for r2* lookup.

    Returns:
        Tuple of (concentration_file, signal_file) paths.
    """
    import numpy as np
    import nibabel as nib  # type: ignore[import-untyped]
    from pathlib import Path

    from pmriprep.utils.dsc_signal import signal_to_concentration

    # Load DSC data
    dsc_img = nib.load(dsc_file)
    signal = dsc_img.get_fdata()  # type: ignore[attr-defined]
    affine = dsc_img.affine  # type: ignore[attr-defined]
    header = dsc_img.header  # type: ignore[attr-defined]

    # Convert to concentration
    concentration = signal_to_concentration(
        signal=signal,
        te=te,
        baseline_frames=baseline_frames,
        r2_star=r2_star,
        field_strength=field_strength,
    )

    # Save concentration file
    dsc_path = Path(dsc_file)
    if dsc_path.name.endswith(".nii.gz"):
        base = dsc_path.name[:-7]
    else:
        base = dsc_path.stem

    out_dir = Path.cwd()
    conc_file = str(out_dir / f"{base}_concentration.nii.gz")
    conc_img = nib.Nifti1Image(concentration.astype(np.float32), affine, header)  # type: ignore[attr-defined]
    nib.save(conc_img, conc_file)  # type: ignore[attr-defined]

    # Also save signal file path for PSR/PH calculation
    signal_file = str(dsc_file)

    return conc_file, signal_file


def _detect_and_correct_preload(
    concentration_file,
    aif_file,
    brain_mask_file,
    te,
    field_strength,
    tr,
    preload_detected,
    save_k2_map,
    signal_file=None,
    baseline_frames=5,
):
    """Detect preload and apply K2 correction if needed.

    This is a Function node wrapper for preload detection and K2 correction.

    Args:
        concentration_file: Path to 4D concentration NIfTI.
        aif_file: Path to AIF curve text file.
        brain_mask_file: Path to brain mask NIfTI.
        te: Echo time in seconds.
        field_strength: Magnetic field strength in Tesla.
        tr: Repetition time in seconds.
        preload_detected: User-specified preload flag (True/False/None for auto).
        save_k2_map: Whether to save the K2 leakage map.

    Returns:
        Tuple of (corrected_concentration_file, k2_map_file, qc_dict).
    """
    import numpy as np
    import nibabel as nib  # type: ignore[import-untyped]
    from pathlib import Path
    from typing import Optional

    from pmriprep.utils.dsc_preload import (
        apply_k2_correction,
    )

    # Load concentration data
    conc_img = nib.load(concentration_file)
    conc_data = conc_img.get_fdata()  # type: ignore[attr-defined]
    affine = conc_img.affine  # type: ignore[attr-defined]
    header = conc_img.header  # type: ignore[attr-defined]

    mask_img = nib.load(brain_mask_file)
    brain_mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    # Verify brain mask is in DSC space (guaranteed by extract_dsc_brain node)
    if brain_mask.shape[:3] != conc_data.shape[:3]:
        raise RuntimeError(
            f"Brain mask shape {brain_mask.shape[:3]} does not match "
            f"concentration data shape {conc_data.shape[:3]}. "
            "The brain mask must be in native DSC space. "
            "Ensure extract_dsc_brain node is connected upstream."
        )

    # Load AIF
    aif = np.loadtxt(aif_file)

    qc_metrics: dict = {}
    k2_map_file: str = ""

    # Detect preload if not explicitly specified
    needs_correction = False
    if preload_detected is True:
        needs_correction = True
        qc_metrics["preload_user_specified"] = True
    elif preload_detected is False:
        needs_correction = False
        qc_metrics["preload_user_specified"] = True
    else:
        # Auto-detect using full preload detection from utils
        from pmriprep.utils.dsc_preload import detect_preload

        qc_metrics["preload_user_specified"] = False

        signal_data = None
        if signal_file:
            signal_data = nib.load(signal_file).get_fdata()

        preload_result = detect_preload(
            signal=signal_data,
            te=te,
            baseline_frames=baseline_frames,
            brain_mask=brain_mask,
            field_strength=field_strength,
        )
        needs_correction = preload_result.classification.value not in (
            "none",
            "minimal",
        )
        qc_metrics["preload_auto_detected"] = needs_correction
        qc_metrics["preload_classification"] = preload_result.classification.value
        qc_metrics["preload_confidence"] = preload_result.confidence
        qc_metrics["preload_detection_method"] = preload_result.detection_method
        qc_metrics["baseline_delta_r2"] = preload_result.baseline_delta_r2

    qc_metrics["preload_correction_applied"] = needs_correction

    # Apply K2 correction if needed
    if needs_correction:
        # Apply K2 leakage correction
        correction_result = apply_k2_correction(
            concentration=conc_data,
            aif=aif,
            brain_mask=brain_mask,
            tr=tr,
        )

        corrected_conc = correction_result.concentration_corrected
        k2_map = correction_result.k2_map

        # Update QC metrics
        if correction_result.qc_metrics:
            qc_metrics.update(correction_result.qc_metrics)

        # Save corrected concentration
        conc_path = Path(concentration_file)
        if conc_path.name.endswith(".nii.gz"):
            base = conc_path.name[:-7]
        else:
            base = conc_path.stem

        out_dir = Path.cwd()
        corrected_file = str(out_dir / f"{base}_k2corrected.nii.gz")
        corrected_img = nib.Nifti1Image(corrected_conc.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(corrected_img, corrected_file)  # type: ignore[attr-defined]

        # Save K2 map if requested
        if save_k2_map and k2_map is not None:
            k2_map_file = str(out_dir / f"{base}_k2map.nii.gz")
            k2_img = nib.Nifti1Image(k2_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(k2_img, k2_map_file)  # type: ignore[attr-defined]

        return corrected_file, k2_map_file, qc_metrics
    else:
        # No correction needed, pass through
        return concentration_file, None, qc_metrics


def _compute_prebolus_mean(moco_file, baseline_frames):
    """Compute mean of pre-bolus frames as a clean reference volume.

    The standard mean image includes bolus passage frames, which are
    contaminated by contrast-induced signal drop. Using only pre-bolus
    frames provides a cleaner reference for brain extraction and
    registration.

    Args:
        moco_file: Path to motion-corrected 4D DSC NIfTI.
        baseline_frames: Number of pre-bolus frames to average.

    Returns:
        Path to pre-bolus mean NIfTI file.
    """
    import numpy as np
    import nibabel as nib
    from pathlib import Path

    img = nib.load(moco_file)
    data = img.get_fdata()

    # Average only pre-bolus frames
    n_frames = min(baseline_frames, data.shape[-1])
    prebolus_mean = np.mean(data[..., :n_frames], axis=-1)

    out_dir = Path.cwd()
    out_file = str(out_dir / "dsc_prebolus_mean.nii.gz")
    mean_img = nib.Nifti1Image(
        prebolus_mean.astype(np.float32), img.affine, img.header
    )
    nib.save(mean_img, out_file)

    return out_file


def _normalize_cbv(
    cbv_file,
    brain_mask_file,
    derivatives_dir,
    subject_id,
    session_id,
    nawm_mask_file=None,
):
    """Normalize CBV to NAWM median.

    This is a Function node wrapper for CBV normalization.

    Args:
        cbv_file: Path to CBV map NIfTI.
        nawm_mask_file: Path to NAWM mask NIfTI (optional).
        brain_mask_file: Path to brain mask for fallback.
        derivatives_dir: Path to derivatives directory for auto-detection.
        subject_id: Subject ID for auto-detection.
        session_id: Session ID for auto-detection.

    Returns:
        Tuple of (ncbv_file, normalization_factor).
    """
    from pathlib import Path
    from pmriprep.utils.dsc_signal import normalize_cbv_from_files

    cbv_path = Path(cbv_file)
    if cbv_path.name.endswith(".nii.gz"):
        base = cbv_path.name[:-7]
    else:
        base = cbv_path.stem

    out_dir = Path.cwd()
    ncbv_file = str(out_dir / f"{base.replace('_cbv', '')}_ncbv.nii.gz")

    output_path, norm_factor = normalize_cbv_from_files(
        cbv_file=cbv_file,
        output_file=ncbv_file,
        nawm_mask_file=nawm_mask_file,
        derivatives_dir=derivatives_dir,
        subject=subject_id,
        session=session_id,
        brain_mask_file=brain_mask_file,
    )

    return str(output_path), norm_factor


def _normalize_cbf(
    cbf_file,
    brain_mask_file,
    derivatives_dir,
    subject_id,
    session_id,
    nawm_mask_file=None,
):
    """Normalize CBF to NAWM median (same method as CBV normalization).

    Args:
        cbf_file: Path to CBF map NIfTI.
        brain_mask_file: Path to brain mask for fallback.
        derivatives_dir: Path to derivatives directory for auto-detection.
        subject_id: Subject ID for auto-detection.
        session_id: Session ID for auto-detection.
        nawm_mask_file: Path to NAWM mask NIfTI (optional).

    Returns:
        Tuple of (ncbf_file, normalization_factor).
    """
    from pathlib import Path
    from pmriprep.utils.dsc_signal import normalize_cbv_from_files

    cbf_path = Path(cbf_file)
    if cbf_path.name.endswith(".nii.gz"):
        base = cbf_path.name[:-7]
    else:
        base = cbf_path.stem

    out_dir = Path.cwd()
    ncbf_file = str(out_dir / f"{base.replace('_cbf', '')}_ncbf.nii.gz")

    output_path, norm_factor = normalize_cbv_from_files(
        cbv_file=cbf_file,
        output_file=ncbf_file,
        nawm_mask_file=nawm_mask_file,
        derivatives_dir=derivatives_dir,
        subject=subject_id,
        session=session_id,
        brain_mask_file=brain_mask_file,
    )

    return str(output_path), norm_factor


def _create_registration_filmstrip(
    dsc_mean_file,
    registered_t1w_file,
    brain_mask_file,
):
    """Create a registration QC filmstrip PNG.

    Generates a multi-slice overlay showing the registered T1w on the DSC mean
    with brain mask contours for visual QC of registration quality.

    Args:
        dsc_mean_file: Path to mean DSC image (reference).
        registered_t1w_file: Path to registered T1w in DSC space.
        brain_mask_file: Path to DSC brain mask.

    Returns:
        Path to the registration filmstrip PNG file.
    """
    import nibabel as nib
    from pathlib import Path
    from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip

    out_dir = Path.cwd()
    output_path = str(out_dir / "registration_qc_filmstrip.png")

    dsc_data = nib.load(dsc_mean_file).get_fdata()
    t1w_data = nib.load(registered_t1w_file).get_fdata()
    mask_data = nib.load(brain_mask_file).get_fdata()

    plot_registration_qc_filmstrip(
        fixed=dsc_data,
        moving=t1w_data,
        output_path=output_path,
        mask=mask_data,
        title="DSC-T1w Registration QC",
    )

    return output_path


def _extract_preload_class(qc_dict):
    """Extract preload classification from preload_correct QC dict.

    Args:
        qc_dict: QC metrics dictionary from preload correction node.

    Returns:
        Preload classification string (e.g. 'none', 'minimal', 'half_dose').
    """
    if not qc_dict:
        return "none"
    return qc_dict.get("preload_classification", "none")


def _smooth_maps(map_files, brain_mask_file, fwhm):
    """Apply Gaussian smoothing to a dict of maps.

    Uses fslmaths -s for smoothing and -mas for re-masking.

    Args:
        map_files: Dict mapping map name to file path.
        brain_mask_file: Path to brain mask for re-masking after smoothing.
        fwhm: Smoothing FWHM in mm.

    Returns:
        Dict mapping map name to smoothed file path.
    """
    import subprocess
    from pathlib import Path

    sigma = fwhm / 2.355
    results = {}
    for name, filepath in map_files.items():
        if not filepath or filepath in ("", "None"):
            continue
        if not Path(filepath).exists():
            continue
        out_path = str(Path.cwd() / f"{Path(filepath).stem}_fwhm{int(fwhm)}.nii.gz")
        subprocess.run(
            ["fslmaths", filepath, "-s", str(sigma), "-mas", brain_mask_file, out_path],
            check=True,
            capture_output=True,
        )
        results[name] = out_path
    return results


def _merge_qc_metrics(
    motion_qc,
    preload_qc,
    aif_qc,
    deconv_qc,
    registration_qc,
    normalization_factor,
):
    """Merge all QC metrics into a single dictionary.

    Args:
        motion_qc: Motion correction QC metrics.
        preload_qc: Preload detection and correction QC metrics.
        aif_qc: AIF detection QC metrics.
        deconv_qc: Deconvolution QC metrics.
        registration_qc: Registration QC metrics.
        normalization_factor: CBV normalization factor.

    Returns:
        Combined QC metrics dictionary.
    """
    merged = {
        "motion": motion_qc if motion_qc else {},
        "preload": preload_qc if preload_qc else {},
        "aif": aif_qc if aif_qc else {},
        "deconvolution": deconv_qc if deconv_qc else {},
        "registration": registration_qc if registration_qc else {},
        "normalization_factor": normalization_factor,
    }
    return merged


def _save_transforms(
    dsc_to_t1w_matrix,
    t1w_to_dsc_matrix,
    derivatives_dir,
    subject_id,
    session_id,
):
    """Copy transform matrices to BIDS xfm/ directory with proper naming.

    Args:
        dsc_to_t1w_matrix: Path to DSC-to-T1w transform matrix.
        t1w_to_dsc_matrix: Path to T1w-to-DSC transform matrix (inverted).
        derivatives_dir: Path to BIDS derivatives directory.
        subject_id: Subject ID (without 'sub-' prefix).
        session_id: Session ID (without 'ses-' prefix), or None.

    Returns:
        Tuple of (dsc_to_t1w_path, t1w_to_dsc_path) in BIDS layout.
    """
    import shutil
    from pathlib import Path

    sub_dir = f"sub-{subject_id}"
    ses_part = f"ses-{session_id}" if session_id else ""
    if ses_part:
        out_dir = Path(derivatives_dir) / "pmriprep" / sub_dir / ses_part / "xfm"
    else:
        out_dir = Path(derivatives_dir) / "pmriprep" / sub_dir / "xfm"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{sub_dir}{'_' + ses_part if ses_part else ''}"

    dsc_to_t1w_path = str(
        out_dir / f"{prefix}_from-DSC_to-T1w_mode-image_xfm.mat"
    )
    t1w_to_dsc_path = str(
        out_dir / f"{prefix}_from-T1w_to-DSC_mode-image_xfm.mat"
    )

    shutil.copy2(dsc_to_t1w_matrix, dsc_to_t1w_path)
    shutil.copy2(t1w_to_dsc_matrix, t1w_to_dsc_path)

    return dsc_to_t1w_path, t1w_to_dsc_path


def _write_pipeline_provenance(
    merged_qc,
    derivatives_dir,
    subject_id,
    session_id,
    config_dict,
    dsc_file,
):
    """Write QC JSON and provenance JSON to derivatives directory.

    Args:
        merged_qc: Merged QC metrics dictionary from all pipeline steps.
        derivatives_dir: Path to BIDS derivatives directory.
        subject_id: Subject ID (without 'sub-' prefix).
        session_id: Session ID (without 'ses-' prefix), or None.
        config_dict: DSC config as a dictionary.
        dsc_file: Path to the source DSC file.

    Returns:
        Tuple of (qc_json_path, provenance_json_path).
    """
    from pathlib import Path
    from pmriprep.utils.outputs import write_qc_json
    from pmriprep.utils.provenance import create_provenance_dict, write_provenance_json

    # Build output directory
    sub_dir = f"sub-{subject_id}"
    ses_part = f"ses-{session_id}" if session_id else ""
    if ses_part:
        out_dir = Path(derivatives_dir) / "pmriprep" / sub_dir / ses_part / "perf"
    else:
        out_dir = Path(derivatives_dir) / "pmriprep" / sub_dir / "perf"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write QC JSON
    qc_path = out_dir / f"{sub_dir}{'_' + ses_part if ses_part else ''}_desc-dsc_qc.json"
    qc_json_path = str(write_qc_json(
        output_path=qc_path,
        metrics=merged_qc if merged_qc else {},
        subject=subject_id,
        session=session_id,
        pipeline_step="dsc",
    ))

    # Write provenance JSON
    provenance = create_provenance_dict(
        source_files=[dsc_file] if dsc_file else None,
        processing_description="DSC perfusion pipeline (PMRIPrep)",
        config_used=config_dict if config_dict else None,
    )
    prov_path = out_dir / f"{sub_dir}{'_' + ses_part if ses_part else ''}_desc-dsc_provenance.json"
    prov_json_path = str(write_provenance_json(
        output_path=prov_path,
        provenance=provenance,
    ))

    return qc_json_path, prov_json_path


def init_dsc_preproc_wf(
    bids_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    config: Optional[DSCConfig] = None,
    name: str = "dsc_preproc_wf",
) -> pe.Workflow:
    """Initialize DSC perfusion preprocessing workflow.

    This workflow performs DSC perfusion processing including motion correction,
    signal-to-concentration conversion, AIF detection, deconvolution, and
    perfusion parameter extraction.

    Parameters
    ----------
    bids_dir : Path or str
        Path to BIDS dataset root directory.
    subject_id : str
        BIDS subject identifier (without 'sub-' prefix).
    session_id : str, optional
        BIDS session identifier (without 'ses-' prefix).
    config : DSCConfig, optional
        DSC processing configuration. Uses defaults if not provided.
    name : str
        Workflow name (default: "dsc_preproc_wf").

    Returns
    -------
    pe.Workflow
        Nipype workflow for DSC perfusion processing.

    Workflow Inputs
    ---------------
    inputnode.dsc_file : str
        Path to raw 4D DSC perfusion image.
    inputnode.t1w_reference : str
        Path to T1-weighted reference image for registration.
    inputnode.brain_mask : str
        Path to brain mask in T1w space.
    inputnode.wm_probseg : str, optional
        Path to white matter probability map for NAWM mask.
    inputnode.lesion_mask : str, optional
        Path to lesion mask for registration cost weighting and AIF exclusion.
    inputnode.nawm_mask : str, optional
        Path to pre-computed NAWM mask for CBV normalization.
    inputnode.te : float, optional
        Echo time in seconds (default from config or 0.03).
    inputnode.field_strength : float, optional
        Magnetic field strength in Tesla (default 3.0).
    inputnode.tr : float, optional
        Repetition time in seconds (default 1.5).

    Workflow Outputs
    ----------------
    outputnode.dsc_moco : str
        Path to motion-corrected DSC image.
    outputnode.dsc_mean : str
        Path to mean DSC image across time.
    outputnode.motion_params : str
        Path to motion parameters file.
    outputnode.rel_cbf_map, outputnode.rel_cbv_map : str
        Relative cerebral blood flow/volume maps (always produced).
    outputnode.rel_auc_raw_map, outputnode.rel_auc_corr_map : str
        Relative AUC maps (always produced).
    outputnode.abs_cbf_map, outputnode.abs_cbv_map : str, optional
        Absolute CBF/CBV maps (only when gating passes).
    outputnode.abs_auc_raw_map, outputnode.abs_auc_corr_map : str, optional
        Absolute AUC maps (only when gating passes).
    outputnode.mtt_map, outputnode.tmax_map : str
        Mean transit time and time-to-maximum maps.
    outputnode.psr_map, outputnode.ph_map : str
        Percent signal recovery and peak height maps.
    outputnode.ttp_map, outputnode.cnr_map : str
        Time-to-peak and contrast-to-noise ratio maps.
    outputnode.auc_diff_map : str
        AUC leakage contribution (raw - corrected).
    outputnode.snr_map, outputnode.tsnr_map : str
        SNR (baseline) and tSNR (temporal) quality metric maps.
    outputnode.aif_curve : str
        Path to detected/used AIF curve.
    outputnode.aif_mask : str
        Path to AIF detection mask.
    outputnode.k2_map : str, optional
        Path to K2 leakage correction map (if save_k2_map=True).
    outputnode.ncbv_map, outputnode.ncbf_map : str, optional
        Normalized CBV/CBF maps (if NAWM mask available).
    outputnode.smooth_quant_indep, outputnode.smooth_rel, outputnode.smooth_abs : dict, optional
        Smoothed map dictionaries (if smooth_fwhm > 0).
    outputnode.dsc_to_t1w_xfm : str
        Path to DSC to T1w transform.
    outputnode.qc_metrics : dict
        Dictionary of QC metrics from processing steps.

    Notes
    -----
    The workflow follows this processing flow:
    1. Motion correction (mcflirt)
    2. Signal-to-concentration conversion (ΔR2*)
    3. AIF detection (atlas-based → automatic → synthetic fallback)
    4. Preload detection and K2 leakage correction (if needed)
    5. Deconvolution (method from config: sSVD, cSVD, oSVD, tikhonov, vascular, cpi)
    6. Perfusion parameter extraction (CBF, CBV, MTT, Tmax, PSR, PH)
    7. Registration to T1w space
    8. CBV normalization (if NAWM mask available)

    See Also
    --------
    pmriprep.config.dsc.DSCConfig : Configuration options for DSC processing.
    """
    # Use default config if not provided
    if config is None:
        config = DSCConfig()

    # Store parameters for use in workflow nodes
    bids_dir = Path(bids_dir)

    # Create workflow
    workflow = pe.Workflow(name=name)

    # -------------------------------------------------------------------
    # Input Node
    # Define all possible inputs for DSC processing
    # -------------------------------------------------------------------
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dsc_file",  # Raw 4D DSC perfusion image
                "t1w_reference",  # T1w reference for registration
                "brain_mask",  # Brain mask in T1w space
                "wm_probseg",  # White matter probability map (optional)
                "lesion_mask",  # Lesion mask (optional)
                "nawm_mask",  # NAWM mask for normalization (optional)
                "t1w_to_mni_xfm",  # T1w to MNI transform (from ACPC alignment)
                "te",  # Echo time in seconds
                "field_strength",  # Magnetic field strength
                "tr",  # Repetition time in seconds
            ]
        ),
        name="inputnode",
    )

    # Set defaults for optional inputs
    inputnode.inputs.te = 0.03  # Default 30ms echo time
    inputnode.inputs.field_strength = 3.0  # Default 3T
    inputnode.inputs.tr = 1.5  # Default 1.5s TR

    # -------------------------------------------------------------------
    # Output Node
    # Define all outputs from DSC processing
    # -------------------------------------------------------------------
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Motion-corrected data
                "dsc_moco",
                "dsc_mean",
                "motion_params",
                # Quantification-independent parameter maps
                "mtt_map",
                "tmax_map",
                "psr_map",
                "ph_map",
                "ttp_map",
                "cnr_map",
                # AUC decomposition
                "auc_diff_map",
                # Relative maps (always produced)
                "rel_cbf_map",
                "rel_cbv_map",
                "rel_auc_raw_map",
                "rel_auc_corr_map",
                # Absolute maps (conditional on gating)
                "abs_cbf_map",
                "abs_cbv_map",
                "abs_auc_raw_map",
                "abs_auc_corr_map",
                # Quality metrics
                "snr_map",
                "tsnr_map",
                # AIF outputs
                "aif_curve",
                "aif_mask",
                "aif_qc",
                # Leakage correction
                "k2_map",
                # Normalized maps
                "ncbv_map",
                "ncbf_map",
                # Registration
                "dsc_to_t1w_xfm",
                "registration_filmstrip",
                # Smoothed maps (populated only if smooth_fwhm > 0)
                "smooth_quant_indep",
                "smooth_rel",
                "smooth_abs",
                # QC metrics
                "qc_metrics",
            ]
        ),
        name="outputnode",
    )

    # -------------------------------------------------------------------
    # Node 1: Motion Correction
    # -------------------------------------------------------------------
    motion_correct = pe.Node(
        MotionCorrection(
            ref_vol="mean",
            motion_thresh=config.motion_thresh,
        ),
        name="motion_correct",
    )

    # -------------------------------------------------------------------
    # Node 1a: Pre-Bolus Mean Reference
    # Compute mean of pre-bolus frames only, avoiding bolus passage
    # contamination in the reference volume used for brain extraction
    # and registration.
    # -------------------------------------------------------------------
    prebolus_mean = pe.Node(
        niu.Function(
            input_names=["moco_file", "baseline_frames"],
            output_names=["prebolus_mean_file"],
            function=_compute_prebolus_mean,
        ),
        name="prebolus_mean",
    )
    prebolus_mean.inputs.baseline_frames = config.pre_bolus

    # -------------------------------------------------------------------
    # Node 1b: DSC Brain Extraction
    # Transform T1w brain mask to native DSC space via quick FLIRT.
    # This provides a properly-dimensioned brain mask for all DSC-space
    # computations (AIF detection, deconvolution, preload correction).
    # -------------------------------------------------------------------
    extract_dsc_brain = pe.Node(
        niu.Function(
            input_names=["dsc_mean_file", "brain_mask_file", "t1w_reference_file"],
            output_names=["dsc_brain_mask_file"],
            function=_extract_dsc_brain,
        ),
        name="extract_dsc_brain",
    )

    # -------------------------------------------------------------------
    # Node 2: Signal to Concentration Conversion
    # -------------------------------------------------------------------
    # Determine r2_star based on config
    r2_star_value: Optional[float] = None
    if config.absolute_quant and config.r2_star is not None:
        r2_star_value = config.r2_star

    signal_to_conc = pe.Node(
        niu.Function(
            input_names=["dsc_file", "te", "baseline_frames", "r2_star", "field_strength"],
            output_names=["concentration_file", "signal_file"],
            function=_signal_to_concentration,
        ),
        name="signal_to_conc",
    )
    signal_to_conc.inputs.baseline_frames = config.pre_bolus
    signal_to_conc.inputs.r2_star = r2_star_value

    # -------------------------------------------------------------------
    # Node 3: AIF Detection
    # AIF detection uses the uncorrected concentration - leakage doesn't
    # affect arterial voxels (which should have intact BBB)
    # -------------------------------------------------------------------
    aif_detect = pe.Node(
        AIFDetection(
            aif_method=config.aif_method,
            criteria=config.aif_criteria,
            n_candidates=config.n_aif_candidates,
            lesion_distance=config.lesion_distance,
            prefer_contralateral=config.prefer_contralateral,
            gamma_fit=config.gamma_fit,
            disable_fallback=config.disable_aif_fallback,
            baseline_frames=config.pre_bolus,
        ),
        name="aif_detect",
    )

    # -------------------------------------------------------------------
    # Node 4: Preload Detection and K2 Correction
    # K2 correction needs the AIF, so runs after AIF detection
    # -------------------------------------------------------------------
    preload_correct = pe.Node(
        niu.Function(
            input_names=[
                "concentration_file",
                "aif_file",
                "brain_mask_file",
                "te",
                "field_strength",
                "tr",
                "preload_detected",
                "save_k2_map",
                "signal_file",
                "baseline_frames",
            ],
            output_names=["corrected_concentration_file", "k2_map_file", "qc_dict"],
            function=_detect_and_correct_preload,
        ),
        name="preload_correct",
    )
    preload_correct.inputs.preload_detected = config.preload_detected if config.preload_detected else None
    preload_correct.inputs.save_k2_map = config.save_k2_map
    preload_correct.inputs.baseline_frames = config.pre_bolus

    # -------------------------------------------------------------------
    # Node 4b: Extract preload classification from QC dict
    # -------------------------------------------------------------------
    extract_preload = pe.Node(
        niu.Function(
            input_names=["qc_dict"],
            output_names=["preload_classification"],
            function=_extract_preload_class,
        ),
        name="extract_preload",
    )

    # -------------------------------------------------------------------
    # Node 5: DSC Map Computation (Deconvolution)
    # -------------------------------------------------------------------
    compute_maps = pe.Node(
        ComputeDSCMaps(
            method=config.method,
            svd_threshold=0.2,  # Default SVD threshold
            oi_threshold=0.065,  # Default OI threshold for oSVD
            cpi_control_points=config.cpi_control_points,
            bayesian=config.bayesian,
            mcmc_samples=config.mcmc_samples,
            use_gpu=config.use_gpu,
            baseline_frames=config.pre_bolus,
            tail_frames=10,  # Default tail frames for PSR
        ),
        name="compute_maps",
    )

    # -------------------------------------------------------------------
    # Node 6: DSC to T1w Registration
    # -------------------------------------------------------------------
    register = pe.Node(
        DSCRegistration(
            use_flirt=config.use_flirt,
            dof=6,  # Rigid body
            cost="normmi",
        ),
        name="register",
    )

    # -------------------------------------------------------------------
    # Node 6b: Transform Chain for MNI Atlas to DSC Space
    # Creates MNI→DSC transform by concatenating:
    #   MNI→T1w (inverted T1w→MNI) + T1w→DSC (inverted DSC→T1w)
    # -------------------------------------------------------------------

    # Invert T1w→MNI to get MNI→T1w
    invert_mni_xfm = pe.Node(
        ConvertXFM(invert_xfm=True),
        name="invert_mni_xfm",
    )

    # Invert DSC→T1w to get T1w→DSC
    invert_dsc_xfm = pe.Node(
        ConvertXFM(invert_xfm=True),
        name="invert_dsc_xfm",
    )

    # Concatenate: MNI→T1w + T1w→DSC = MNI→DSC
    concat_mni_to_dsc = pe.Node(
        ConvertXFM(concat_xfm=True),
        name="concat_mni_to_dsc",
    )

    # Transform arterial atlas from MNI to DSC space
    # Use nearest neighbor interpolation for binary mask
    transform_atlas = pe.Node(
        FLIRTApplyXFM(
            apply_xfm=True,
            interp="nearestneighbour",
        ),
        name="transform_atlas",
    )
    # Set the arterial atlas as input (bundled with pmriprep)
    try:
        transform_atlas.inputs.in_file = str(get_arterial_atlas("2mm"))
    except FileNotFoundError:
        import warnings
        warnings.warn(
            "Arterial atlas not found. AIF detection will use whole-brain search."
        )

    # -------------------------------------------------------------------
    # Node 7: CBV Normalization
    # -------------------------------------------------------------------
    normalize = pe.Node(
        niu.Function(
            input_names=[
                "cbv_file",
                "brain_mask_file",
                "derivatives_dir",
                "subject_id",
                "session_id",
                "nawm_mask_file",
            ],
            output_names=["ncbv_file", "normalization_factor"],
            function=_normalize_cbv,
        ),
        name="normalize",
    )
    normalize.inputs.derivatives_dir = str(bids_dir / "derivatives")
    normalize.inputs.subject_id = subject_id
    normalize.inputs.session_id = session_id
    normalize.inputs.nawm_mask_file = None  # Optional, will use fallback if not provided

    # -------------------------------------------------------------------
    # Node 7b: CBF Normalization (same method as CBV normalization)
    # -------------------------------------------------------------------
    normalize_cbf = pe.Node(
        niu.Function(
            input_names=[
                "cbf_file",
                "brain_mask_file",
                "derivatives_dir",
                "subject_id",
                "session_id",
                "nawm_mask_file",
            ],
            output_names=["ncbf_file", "normalization_factor"],
            function=_normalize_cbf,
        ),
        name="normalize_cbf",
    )
    normalize_cbf.inputs.derivatives_dir = str(bids_dir / "derivatives")
    normalize_cbf.inputs.subject_id = subject_id
    normalize_cbf.inputs.session_id = session_id
    normalize_cbf.inputs.nawm_mask_file = None

    # -------------------------------------------------------------------
    # Node 7c: Smoothing (conditional on config.smooth_fwhm > 0)
    # -------------------------------------------------------------------
    smooth_quant_indep = None
    smooth_rel = None
    smooth_abs = None

    if config.smooth_fwhm > 0:
        # Smoothing for quantification-independent maps
        smooth_quant_indep = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_maps,
            ),
            name="smooth_quant_indep",
        )
        smooth_quant_indep.inputs.fwhm = config.smooth_fwhm

        # Smoothing for relative AUC maps
        smooth_rel = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_maps,
            ),
            name="smooth_rel",
        )
        smooth_rel.inputs.fwhm = config.smooth_fwhm

        # Smoothing for absolute AUC maps (only produces output if abs maps exist)
        smooth_abs = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_maps,
            ),
            name="smooth_abs",
        )
        smooth_abs.inputs.fwhm = config.smooth_fwhm

    # -------------------------------------------------------------------
    # Node 8: Merge QC Metrics
    # -------------------------------------------------------------------
    merge_qc = pe.Node(
        niu.Function(
            input_names=[
                "motion_qc",
                "preload_qc",
                "aif_qc",
                "deconv_qc",
                "registration_qc",
                "normalization_factor",
            ],
            output_names=["merged_qc"],
            function=_merge_qc_metrics,
        ),
        name="merge_qc",
    )

    # -------------------------------------------------------------------
    # Node 9: Registration QC Filmstrip
    # -------------------------------------------------------------------
    reg_filmstrip = pe.Node(
        niu.Function(
            input_names=["dsc_mean_file", "registered_t1w_file", "brain_mask_file"],
            output_names=["filmstrip_file"],
            function=_create_registration_filmstrip,
        ),
        name="reg_filmstrip",
    )

    # -------------------------------------------------------------------
    # Node 10: Provenance Logging
    # -------------------------------------------------------------------
    import dataclasses
    config_dict = dataclasses.asdict(config)
    # Convert Path objects to strings for JSON serialization
    for k, v in config_dict.items():
        if isinstance(v, Path):
            config_dict[k] = str(v)

    write_provenance = pe.Node(
        niu.Function(
            input_names=[
                "merged_qc",
                "derivatives_dir",
                "subject_id",
                "session_id",
                "config_dict",
                "dsc_file",
            ],
            output_names=["qc_json_path", "provenance_json_path"],
            function=_write_pipeline_provenance,
        ),
        name="write_provenance",
    )
    write_provenance.inputs.derivatives_dir = str(bids_dir / "derivatives")
    write_provenance.inputs.subject_id = subject_id
    write_provenance.inputs.session_id = session_id
    write_provenance.inputs.config_dict = config_dict

    # -------------------------------------------------------------------
    # Node 11: Save Registration Transforms
    # -------------------------------------------------------------------
    save_transforms = pe.Node(
        niu.Function(
            input_names=[
                "dsc_to_t1w_matrix",
                "t1w_to_dsc_matrix",
                "derivatives_dir",
                "subject_id",
                "session_id",
            ],
            output_names=["dsc_to_t1w_path", "t1w_to_dsc_path"],
            function=_save_transforms,
        ),
        name="save_transforms",
    )
    save_transforms.inputs.derivatives_dir = str(bids_dir / "derivatives")
    save_transforms.inputs.subject_id = subject_id
    save_transforms.inputs.session_id = session_id

    # -------------------------------------------------------------------
    # Workflow Connections
    # -------------------------------------------------------------------

    # Input to Motion Correction
    workflow.connect([
        (inputnode, motion_correct, [("dsc_file", "in_file")]),
    ])

    # Motion Correction to Pre-Bolus Mean
    workflow.connect([
        (motion_correct, prebolus_mean, [("out_file", "moco_file")]),
    ])

    # Pre-Bolus Mean to DSC Brain Extraction
    # Use pre-bolus mean (not full mean) as reference to avoid bolus contamination
    workflow.connect([
        (prebolus_mean, extract_dsc_brain, [("prebolus_mean_file", "dsc_mean_file")]),
        (inputnode, extract_dsc_brain, [
            ("brain_mask", "brain_mask_file"),
            ("t1w_reference", "t1w_reference_file"),
        ]),
    ])

    # Motion Correction to Signal Conversion
    workflow.connect([
        (motion_correct, signal_to_conc, [("out_file", "dsc_file")]),
        (inputnode, signal_to_conc, [
            ("te", "te"),
            ("field_strength", "field_strength"),
        ]),
    ])

    # Signal Conversion to AIF Detection
    # Use DSC-space brain mask (not T1w-space) for proper geometry match
    # Pass raw signal for S0-based partial volume rejection
    workflow.connect([
        (signal_to_conc, aif_detect, [
            ("concentration_file", "concentration_file"),
            ("signal_file", "signal_file"),
        ]),
        (extract_dsc_brain, aif_detect, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, aif_detect, [
            ("lesion_mask", "lesion_mask_file"),
        ]),
    ])

    # AIF Detection and Signal Conversion to Preload Correction
    # Use DSC-space brain mask for proper geometry match
    workflow.connect([
        (signal_to_conc, preload_correct, [
            ("concentration_file", "concentration_file"),
            ("signal_file", "signal_file"),
        ]),
        (aif_detect, preload_correct, [("out_aif_curve", "aif_file")]),
        (extract_dsc_brain, preload_correct, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, preload_correct, [
            ("te", "te"),
            ("field_strength", "field_strength"),
            ("tr", "tr"),
        ]),
    ])

    # Preload QC dict → extract_preload classification
    workflow.connect([
        (preload_correct, extract_preload, [("qc_dict", "qc_dict")]),
    ])

    # Preload Correction to DSC Map Computation
    # Use DSC-space brain mask for proper geometry match
    workflow.connect([
        (preload_correct, compute_maps, [
            ("corrected_concentration_file", "concentration_file"),
        ]),
        (aif_detect, compute_maps, [
            ("out_aif_curve", "aif_file"),
            ("detection_level", "aif_detection_level"),
        ]),
        (extract_dsc_brain, compute_maps, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, compute_maps, [
            ("tr", "tr"),
            ("field_strength", "field_strength"),
        ]),
        (signal_to_conc, compute_maps, [
            ("signal_file", "signal_file"),
            ("concentration_file", "raw_concentration_file"),
        ]),
        (extract_preload, compute_maps, [
            ("preload_classification", "preload_classification"),
        ]),
    ])

    # DSC Map Computation to Registration
    # Use pre-bolus mean for registration reference (cleaner than full mean)
    workflow.connect([
        (inputnode, register, [
            ("t1w_reference", "t1w_file"),
            ("lesion_mask", "lesion_mask"),
            ("brain_mask", "brain_mask"),
            ("wm_probseg", "wm_seg"),
        ]),
        (prebolus_mean, register, [("prebolus_mean_file", "dsc_mean")]),
    ])

    # Transform Chain: Create MNI→DSC transform for atlas
    # Step 1: Invert T1w→MNI to get MNI→T1w
    workflow.connect([
        (inputnode, invert_mni_xfm, [("t1w_to_mni_xfm", "in_file")]),
    ])

    # Step 2: Invert DSC→T1w to get T1w→DSC
    workflow.connect([
        (register, invert_dsc_xfm, [("out_matrix", "in_file")]),
    ])

    # Step 3: Concatenate MNI→T1w + T1w→DSC = MNI→DSC
    workflow.connect([
        (invert_mni_xfm, concat_mni_to_dsc, [("out_file", "in_file")]),
        (invert_dsc_xfm, concat_mni_to_dsc, [("out_file", "in_file2")]),
    ])

    # Step 4: Transform arterial atlas from MNI to DSC space
    # Use pre-bolus mean as reference for output geometry
    workflow.connect([
        (concat_mni_to_dsc, transform_atlas, [("out_file", "in_matrix_file")]),
        (prebolus_mean, transform_atlas, [("prebolus_mean_file", "reference")]),
    ])

    # Connect transformed atlas to AIF detection
    workflow.connect([
        (transform_atlas, aif_detect, [("out_file", "atlas_mask_file")]),
    ])

    # DSC Map Computation to CBV Normalization (uses relative CBV)
    # Use DSC-space brain mask for fallback normalization
    workflow.connect([
        (compute_maps, normalize, [("out_rel_cbv", "cbv_file")]),
        (inputnode, normalize, [
            ("nawm_mask", "nawm_mask_file"),
        ]),
        (extract_dsc_brain, normalize, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # DSC Map Computation to CBF Normalization (uses relative CBF)
    workflow.connect([
        (compute_maps, normalize_cbf, [("out_rel_cbf", "cbf_file")]),
        (inputnode, normalize_cbf, [
            ("nawm_mask", "nawm_mask_file"),
        ]),
        (extract_dsc_brain, normalize_cbf, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # Registration QC filmstrip
    workflow.connect([
        (prebolus_mean, reg_filmstrip, [("prebolus_mean_file", "dsc_mean_file")]),
        (register, reg_filmstrip, [("out_file", "registered_t1w_file")]),
        (extract_dsc_brain, reg_filmstrip, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Smoothing connections (only if smooth_fwhm > 0)
    # -------------------------------------------------------------------
    if smooth_quant_indep is not None:
        # Build dict of quant-independent maps for smoothing: MTT, TTP, Tmax, PSR, PH, K2, auc_diff
        def _build_quant_indep_dict(mtt, ttp, tmax, psr, ph, k2, auc_diff):
            d = {
                "mtt": mtt, "ttp": ttp, "tmax": tmax,
                "psr": psr, "ph": ph, "auc_diff": auc_diff,
            }
            if k2:
                d["k2"] = k2
            return d

        build_qi_dict = pe.Node(
            niu.Function(
                input_names=["mtt", "ttp", "tmax", "psr", "ph", "k2", "auc_diff"],
                output_names=["map_dict"],
                function=_build_quant_indep_dict,
            ),
            name="build_qi_dict",
        )
        workflow.connect([
            (compute_maps, build_qi_dict, [
                ("out_mtt", "mtt"),
                ("out_ttp", "ttp"),
                ("out_tmax", "tmax"),
                ("out_psr", "psr"),
                ("out_ph", "ph"),
                ("out_auc_diff", "auc_diff"),
            ]),
            (preload_correct, build_qi_dict, [("k2_map_file", "k2")]),
            (build_qi_dict, smooth_quant_indep, [("map_dict", "map_files")]),
            (extract_dsc_brain, smooth_quant_indep, [
                ("dsc_brain_mask_file", "brain_mask_file"),
            ]),
        ])

        # Build dict of relative AUC maps for smoothing
        def _build_rel_dict(auc_raw, auc_corr):
            return {"auc_raw": auc_raw, "auc_corr": auc_corr}

        build_rel_dict = pe.Node(
            niu.Function(
                input_names=["auc_raw", "auc_corr"],
                output_names=["map_dict"],
                function=_build_rel_dict,
            ),
            name="build_rel_dict",
        )
        workflow.connect([
            (compute_maps, build_rel_dict, [
                ("out_rel_auc_raw", "auc_raw"),
                ("out_rel_auc_corr", "auc_corr"),
            ]),
            (build_rel_dict, smooth_rel, [("map_dict", "map_files")]),
            (extract_dsc_brain, smooth_rel, [
                ("dsc_brain_mask_file", "brain_mask_file"),
            ]),
        ])

        # Build dict of absolute AUC maps for smoothing (may be empty)
        # NOTE: When absolute gating fails, these inputs may be Nipype
        # Undefined objects. We use isinstance(str) + path existence check
        # to safely handle any non-string or missing-file values.
        def _build_abs_dict(auc_raw, auc_corr):
            from pathlib import Path as _P
            d = {}
            if isinstance(auc_raw, str) and auc_raw not in ("", "None") and _P(auc_raw).exists():
                d["auc_raw"] = auc_raw
            if isinstance(auc_corr, str) and auc_corr not in ("", "None") and _P(auc_corr).exists():
                d["auc_corr"] = auc_corr
            return d

        build_abs_dict = pe.Node(
            niu.Function(
                input_names=["auc_raw", "auc_corr"],
                output_names=["map_dict"],
                function=_build_abs_dict,
            ),
            name="build_abs_dict",
        )
        workflow.connect([
            (compute_maps, build_abs_dict, [
                ("out_abs_auc_raw", "auc_raw"),
                ("out_abs_auc_corr", "auc_corr"),
            ]),
            (build_abs_dict, smooth_abs, [("map_dict", "map_files")]),
            (extract_dsc_brain, smooth_abs, [
                ("dsc_brain_mask_file", "brain_mask_file"),
            ]),
        ])

    # Merge QC metrics
    workflow.connect([
        (motion_correct, merge_qc, [("qc_metrics", "motion_qc")]),
        (preload_correct, merge_qc, [("qc_dict", "preload_qc")]),
        (aif_detect, merge_qc, [("qc_metrics", "aif_qc")]),
        (compute_maps, merge_qc, [("qc_metrics", "deconv_qc")]),
        (register, merge_qc, [("qc_metrics", "registration_qc")]),
        (normalize, merge_qc, [("normalization_factor", "normalization_factor")]),
    ])

    # Provenance logging
    workflow.connect([
        (merge_qc, write_provenance, [("merged_qc", "merged_qc")]),
        (inputnode, write_provenance, [("dsc_file", "dsc_file")]),
    ])

    # Save registration transforms to BIDS xfm/ directory
    # register.out_matrix is T1w→DSC (FLIRT: -in T1w -ref DSC)
    # invert_dsc_xfm.out_file is DSC→T1w (inverted)
    workflow.connect([
        (register, save_transforms, [("out_matrix", "t1w_to_dsc_matrix")]),
        (invert_dsc_xfm, save_transforms, [("out_file", "dsc_to_t1w_matrix")]),
    ])

    # Connect to Output Node
    workflow.connect([
        # Motion correction outputs
        (motion_correct, outputnode, [
            ("out_file", "dsc_moco"),
            ("mean_img", "dsc_mean"),
            ("out_params", "motion_params"),
        ]),
        # Quantification-independent maps
        (compute_maps, outputnode, [
            ("out_mtt", "mtt_map"),
            ("out_tmax", "tmax_map"),
            ("out_psr", "psr_map"),
            ("out_ph", "ph_map"),
            ("out_ttp", "ttp_map"),
            ("out_cnr", "cnr_map"),
            ("out_auc_diff", "auc_diff_map"),
        ]),
        # Relative maps
        (compute_maps, outputnode, [
            ("out_rel_cbf", "rel_cbf_map"),
            ("out_rel_cbv", "rel_cbv_map"),
            ("out_rel_auc_raw", "rel_auc_raw_map"),
            ("out_rel_auc_corr", "rel_auc_corr_map"),
        ]),
        # Absolute maps (may be undefined if gating failed)
        (compute_maps, outputnode, [
            ("out_abs_cbf", "abs_cbf_map"),
            ("out_abs_cbv", "abs_cbv_map"),
            ("out_abs_auc_raw", "abs_auc_raw_map"),
            ("out_abs_auc_corr", "abs_auc_corr_map"),
        ]),
        # Quality metrics
        (compute_maps, outputnode, [
            ("out_snr", "snr_map"),
            ("out_tsnr", "tsnr_map"),
        ]),
        # AIF outputs
        (aif_detect, outputnode, [
            ("out_aif_curve", "aif_curve"),
            ("out_aif_mask", "aif_mask"),
            ("qc_metrics", "aif_qc"),
        ]),
        # Leakage correction
        (preload_correct, outputnode, [("k2_map_file", "k2_map")]),
        # Normalized CBV and CBF
        (normalize, outputnode, [("ncbv_file", "ncbv_map")]),
        (normalize_cbf, outputnode, [("ncbf_file", "ncbf_map")]),
        # Registration
        (register, outputnode, [("out_matrix", "dsc_to_t1w_xfm")]),
        (reg_filmstrip, outputnode, [("filmstrip_file", "registration_filmstrip")]),
        # QC metrics
        (merge_qc, outputnode, [("merged_qc", "qc_metrics")]),
    ])

    # Smoothed outputs (only if smooth_fwhm > 0)
    if smooth_quant_indep is not None:
        workflow.connect([
            (smooth_quant_indep, outputnode, [("smoothed_files", "smooth_quant_indep")]),
            (smooth_rel, outputnode, [("smoothed_files", "smooth_rel")]),
            (smooth_abs, outputnode, [("smoothed_files", "smooth_abs")]),
        ])

    # Store config in workflow for reference by child nodes
    workflow.config_dsc = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
