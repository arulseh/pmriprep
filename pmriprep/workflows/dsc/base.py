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
from pmriprep.interfaces.common import Reorient
from pmriprep.interfaces.dsc import MotionCorrection, DSCRegistration
from pmriprep.interfaces.dsc_aif import AIFDetection
from pmriprep.interfaces.dsc_aif_qc import AIFQCPlot
from pmriprep.interfaces.dsc_compute import ComputeDSCMaps


def _histogram_match_t1w_to_dsc(
    t1w_file: str, dsc_brain_file: str, brain_mask_file: str,
) -> str:
    """Match T1w histogram to DSC for improved FLIRT convergence.

    T1w and DSC have inverted contrasts (bright WM vs bright GM).
    Without histogram matching, FLIRT's normcorr coarse-search stage
    converges to a different local optimum. This matches perf-plus
    registration_utils.py:161-208.

    The T1w is brain-extracted via *brain_mask_file* before matching so
    that skull/scalp/background voxels do not distort the histogram.
    Perf-plus uses ``$T1W_BRAIN`` (brain-extracted) for this step.

    Args:
        t1w_file: Path to whole-head bias-corrected T1w image.
        dsc_brain_file: Path to DSC brain-extracted image.
        brain_mask_file: Path to T1w brain mask (binary).

    Returns:
        Path to histogram-matched T1w image.
    """
    import numpy as np
    import nibabel as nib
    from pathlib import Path
    from skimage.exposure import match_histograms

    src_nii = nib.load(t1w_file)
    tpl_nii = nib.load(dsc_brain_file)
    mask_nii = nib.load(brain_mask_file)
    src_data = src_nii.get_fdata()  # type: ignore[attr-defined]
    tpl_data = tpl_nii.get_fdata()  # type: ignore[attr-defined]
    mask_data = mask_nii.get_fdata() > 0  # type: ignore[attr-defined]
    src_brain = src_data * mask_data
    src_mask = src_brain > 0
    tpl_mask = tpl_data > 0

    # Erode template mask by 1 voxel to exclude HD-BET boundary voxels,
    # which are non-deterministic between GPU runs and have partial-volume
    # contamination. Interior voxels are stable across runs.
    from scipy.ndimage import binary_erosion
    tpl_mask_eroded = binary_erosion(tpl_mask, iterations=1)
    # Guard for thin-slab acquisitions: only erode if ≥80% of voxels remain
    if tpl_mask_eroded.sum() >= 0.8 * tpl_mask.sum():
        tpl_mask = tpl_mask_eroded

    matched = np.zeros_like(src_data)
    if src_mask.sum() > 0 and tpl_mask.sum() > 0:
        matched[src_mask] = match_histograms(
            src_brain[src_mask].reshape(-1, 1),
            tpl_data[tpl_mask].reshape(-1, 1),
            channel_axis=1,
        ).flatten()

    out_file = str(Path.cwd() / "t1w_brain_matched.nii.gz")
    nib.save(nib.Nifti1Image(matched.astype(np.float32),
                              src_nii.affine, src_nii.header), out_file)  # type: ignore[attr-defined]
    return out_file


def _create_perfusion_brain_mask_filmstrip(mean_file, brain_mask_file):
    """Create brain extraction QC filmstrip for DSC or DCE perfusion data.

    Overlays the brain mask contour (solid red line) on the mean perfusion
    image to verify brain extraction quality.

    Parameters
    ----------
    mean_file : str
        Path to mean perfusion image (pre-bolus mean for DSC, baseline
        mean for DCE).
    brain_mask_file : str
        Path to brain mask NIfTI file.

    Returns
    -------
    str
        Path to output filmstrip PNG.
    """
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from pmriprep.utils.qc_utils import plot_segmentation_qc_filmstrip

    out_dir = Path.cwd()
    output_path = str(out_dir / "brain_mask_qc_filmstrip.png")

    mean_img = nib.load(mean_file)
    mean_data = mean_img.get_fdata()
    mask_data = nib.load(brain_mask_file).get_fdata().astype(bool)
    voxel_sizes = tuple(mean_img.header.get_zooms()[:3])

    # Pass full binary mask — contour at level=0.5 gives solid line
    label_masks = [("Brain mask", mask_data.astype(np.float64), "#EE6677")]

    plot_segmentation_qc_filmstrip(
        background=mean_data,
        label_masks=label_masks,
        output_path=output_path,
        mask=mask_data,
        title="Brain Extraction QC",
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _hdbet_dsc_brain(dsc_mean_file, hdbet_device="auto", hdbet_conda_env=""):
    """Extract DSC brain using HD-BET for improved registration quality.

    HD-BET produces a brain-extracted DSC image that yields better
    T1w-to-DSC registration. The final processing mask comes from
    transforming the T1w brain mask through that registration (see
    _transform_brain_mask_to_dsc).

    Fallback: intensity-based thresholding if HD-BET is unavailable.

    Args:
        dsc_mean_file: Path to pre-bolus mean DSC image.
        hdbet_device: Device for HD-BET inference.
        hdbet_conda_env: Conda environment name for HD-BET. When non-empty,
            HD-BET is invoked via ``conda run -n <env>``. When empty (the
            default), HD-BET is called directly first, with a ``conda run``
            fallback.

    Returns:
        Tuple of (dsc_brain_file, dsc_brain_mask_file).
    """
    import subprocess
    import numpy as np
    import nibabel as nib
    from pathlib import Path

    out_dir = Path.cwd()
    dsc_brain_file = str(out_dir / "dsc_brain.nii.gz")
    dsc_brain_mask_file = str(out_dir / "dsc_brain_mask.nii.gz")

    hdbet_succeeded = False

    # Resolve HD-BET device
    if hdbet_device == "auto":
        from pmriprep.utils.device import detect_torch_device
        hdbet_device = detect_torch_device()

    hdbet_args = [
        "hd-bet",
        "-i", dsc_mean_file,
        "-o", dsc_brain_file,
        "-device", hdbet_device,
        "--disable_tta",
    ]

    if hdbet_conda_env:
        # Explicit conda env requested — try it, then fall back to direct
        cmd_prefixes = [
            ["conda", "run", "-n", hdbet_conda_env, "--no-capture-output"],
            [],
        ]
    else:
        # No conda env — try direct first, then fall back (no env name)
        cmd_prefixes = [[], ]

    for cmd_prefix in cmd_prefixes:
        if hdbet_succeeded:
            break
        try:
            hdbet_cmd = cmd_prefix + hdbet_args
            subprocess.run(hdbet_cmd, check=True, capture_output=True)
            hdbet_succeeded = True
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            continue

    if hdbet_succeeded:
        # HD-BET writes brain at -o path and mask with _mask suffix.
        # With -o dsc_brain.nii.gz, mask lands at dsc_brain_mask.nii.gz
        # which is exactly our target path — no move needed.
        hdbet_mask_path = Path(dsc_brain_file.replace(".nii.gz", "_mask.nii.gz"))
        if not hdbet_mask_path.exists():
            # Derive mask from brain-extracted image
            brain_img = nib.load(dsc_brain_file)
            brain_data = brain_img.get_fdata()
            mask_data = (brain_data > 0).astype(np.uint8)
            nib.save(
                nib.Nifti1Image(mask_data, brain_img.affine, brain_img.header),
                dsc_brain_mask_file,
            )
        elif str(hdbet_mask_path) != dsc_brain_mask_file:
            import shutil
            shutil.move(str(hdbet_mask_path), dsc_brain_mask_file)
    else:
        # Fallback: intensity-based brain extraction
        import warnings
        warnings.warn(
            "HD-BET unavailable. Falling back to intensity-based "
            "DSC brain extraction."
        )
        dsc_img = nib.load(dsc_mean_file)
        dsc_data = dsc_img.get_fdata()

        nonzero = dsc_data[dsc_data > 0]
        if len(nonzero) > 0:
            threshold = np.percentile(nonzero, 10)
        else:
            threshold = 0.0

        mask_data = (dsc_data > threshold).astype(np.uint8)
        brain_data = dsc_data * mask_data

        nib.save(
            nib.Nifti1Image(brain_data.astype(np.float32), dsc_img.affine, dsc_img.header),
            dsc_brain_file,
        )
        nib.save(
            nib.Nifti1Image(mask_data, dsc_img.affine, dsc_img.header),
            dsc_brain_mask_file,
        )

    return dsc_brain_file, dsc_brain_mask_file


def _transform_brain_mask_to_dsc(
    brain_mask_file,
    reference_file,
    transform_matrix,
    hdbet_mask_file=None,
):
    """Transform T1w brain mask to DSC space using registration transform.

    This is the final brain mask used for all DSC-space processing.
    The registration transform comes from a higher-quality alignment
    (T1w brain → HD-BET DSC brain), producing a more accurate mask
    than a quick independent FLIRT.

    Optionally computes DICE overlap with the HD-BET mask for QC.

    Args:
        brain_mask_file: Path to brain mask in T1w space.
        reference_file: Path to DSC reference image for output geometry.
        transform_matrix: Path to T1w→DSC FLIRT matrix.
        hdbet_mask_file: Path to HD-BET DSC mask for DICE QC (optional).

    Returns:
        Tuple of (dsc_brain_mask_file, dice_score).
    """
    import subprocess
    import numpy as np
    import nibabel as nib
    from pathlib import Path

    out_dir = Path.cwd()
    dsc_brain_mask_file = str(out_dir / "dsc_brain_mask.nii.gz")

    # Apply transform to brain mask with nearest-neighbour interpolation
    apply_cmd = [
        "flirt",
        "-in", str(brain_mask_file),
        "-ref", str(reference_file),
        "-out", dsc_brain_mask_file,
        "-init", str(transform_matrix),
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

    # Compute DICE with HD-BET mask for QC
    dice_score = -1.0
    if hdbet_mask_file is not None:
        try:
            mask_a = nib.load(dsc_brain_mask_file).get_fdata() > 0.5
            mask_b = nib.load(hdbet_mask_file).get_fdata() > 0.5
            intersection = np.sum(mask_a & mask_b)
            total = np.sum(mask_a) + np.sum(mask_b)
            if total > 0:
                dice_score = 2.0 * intersection / total
            else:
                dice_score = 0.0
        except Exception:
            dice_score = -1.0

    return dsc_brain_mask_file, dice_score


def _signal_to_concentration(
    dsc_file,
    te,
    baseline_frames,
):
    """Convert DSC signal to ΔR2* concentration.

    This is a Function node wrapper for the signal_to_concentration utility.
    Always returns ΔR2* in s⁻¹. All downstream quantification is relative.

    Args:
        dsc_file: Path to motion-corrected 4D DSC NIfTI.
        te: Echo time in seconds.
        baseline_frames: Number of pre-bolus baseline frames.

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

    # Convert to ΔR2* concentration (always s⁻¹)
    concentration = signal_to_concentration(
        signal=signal,
        te=te,
        baseline_frames=baseline_frames,
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
    preload_present,
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
        preload_present: User-specified preload flag (True/False/None for auto).

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

    # Verify brain mask is in DSC space (guaranteed by transform_brain_mask node)
    if brain_mask.shape[:3] != conc_data.shape[:3]:
        raise RuntimeError(
            f"Brain mask shape {brain_mask.shape[:3]} does not match "
            f"concentration data shape {conc_data.shape[:3]}. "
            "The brain mask must be in native DSC space. "
            "Ensure transform_brain_mask node is connected upstream."
        )

    # Load AIF
    aif = np.loadtxt(aif_file)

    qc_metrics: dict = {}
    k2_map_file: str = ""

    # Detect preload for QC classification (does not gate K2 correction)
    if preload_present is True:
        qc_metrics["preload_user_specified"] = True
        qc_metrics["preload_classification"] = "user_specified_preload"
    elif preload_present is False:
        qc_metrics["preload_user_specified"] = True
        qc_metrics["preload_classification"] = "user_specified_no_preload"
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
        qc_metrics["preload_auto_detected"] = preload_result.classification.value not in (
            "none",
            "minimal",
        )
        qc_metrics["preload_classification"] = preload_result.classification.value
        qc_metrics["preload_confidence"] = preload_result.confidence
        qc_metrics["preload_detection_method"] = preload_result.detection_method
        qc_metrics["baseline_delta_r2"] = preload_result.baseline_delta_r2

    # Always apply per-voxel K2 leakage correction (Boxerman 2006).
    # K2 correction addresses BBB disruption regardless of preload status.
    # Voxels with K2 < 1e-4 are effectively unchanged.
    qc_metrics["preload_correction_applied"] = True

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

    # Always save K2 map (needed for leakage QC and downstream nodes)
    if k2_map is not None:
        k2_map_file = str(out_dir / f"{base}_k2map.nii.gz")
        k2_img = nib.Nifti1Image(k2_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(k2_img, k2_map_file)  # type: ignore[attr-defined]

    return corrected_file, k2_map_file, qc_metrics


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


def _transform_nawm_to_dsc(
    nawm_mask_file=None,
    reference_file=None,
    transform_matrix=None,
):
    """Transform NAWM mask from T1w space to DSC space.

    Uses FSL FLIRT with nearest-neighbour interpolation to resample
    the NAWM mask into DSC space.

    Args:
        nawm_mask_file: Path to NAWM mask in T1w space (or None).
        reference_file: Path to DSC reference image for output geometry.
        transform_matrix: Path to T1w→DSC FLIRT matrix.

    Returns:
        Path to transformed NAWM mask in DSC space, or None if no mask.
    """
    if nawm_mask_file is None:
        return None

    from pathlib import Path
    import subprocess

    nawm_path = Path(nawm_mask_file)
    if not nawm_path.exists():
        return None

    out_dir = Path.cwd()
    out_file = str(out_dir / "nawm_mask_dsc.nii.gz")

    cmd = [
        "flirt",
        "-in", str(nawm_mask_file),
        "-ref", str(reference_file),
        "-out", out_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "nearestneighbour",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    return out_file


def _transform_lesion_mask_to_dsc(
    lesion_mask_file=None,
    reference_file=None,
    transform_matrix=None,
):
    """Transform lesion mask from T1w space to DSC space.

    Uses FSL FLIRT with nearest-neighbour interpolation to resample
    the lesion mask into DSC space, preserving integer labels.

    Args:
        lesion_mask_file: Path to lesion mask in T1w space (or None).
        reference_file: Path to DSC reference image for output geometry.
        transform_matrix: Path to T1w→DSC FLIRT matrix.

    Returns:
        Path to transformed lesion mask in DSC space, or None if no mask.
    """
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(lesion_mask_file) or lesion_mask_file is None:
        return None
    if not isdefined(reference_file) or reference_file is None:
        return None
    if not isdefined(transform_matrix) or transform_matrix is None:
        return None

    from pathlib import Path
    import subprocess

    lesion_path = Path(lesion_mask_file)
    if not lesion_path.exists():
        return None

    out_dir = Path.cwd()
    out_file = str(out_dir / "lesion_mask_dsc.nii.gz")

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


def _transform_structural_to_dsc(
    structural_file=None,
    reference_file=None,
    transform_matrix=None,
    output_name="structural_in_dsc.nii.gz",
):
    """Transform structural image from T1w space to DSC space via FLIRT.

    Handles None/undefined inputs gracefully by returning None.

    Args:
        structural_file: Path to structural image in T1w space (or None).
        reference_file: Path to DSC reference image for output geometry.
        transform_matrix: Path to T1w→DSC FLIRT matrix.
        output_name: Output filename.

    Returns:
        Path to transformed structural image in DSC space, or None if no input.
    """
    from nipype.interfaces.base import isdefined

    # Handle both None and Nipype Undefined
    if not isdefined(structural_file) or structural_file is None:
        return None
    if not isdefined(reference_file) or reference_file is None:
        return None
    if not isdefined(transform_matrix) or transform_matrix is None:
        return None

    from pathlib import Path
    import subprocess

    structural_path = Path(structural_file)
    if not structural_path.exists():
        return None

    out_dir = Path.cwd()
    out_file = str(out_dir / output_name)

    cmd = [
        "flirt",
        "-in", str(structural_file),
        "-ref", str(reference_file),
        "-out", out_file,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "spline",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

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
    Returns (None, None) when NAWM mask is not available - normalization
    requires NAWM reference for clinically valid results.

    Args:
        cbv_file: Path to CBV map NIfTI.
        nawm_mask_file: Path to NAWM mask NIfTI (required for normalization).
        brain_mask_file: Path to brain mask (unused, kept for interface compat).
        derivatives_dir: Path to derivatives directory for auto-detection.
        subject_id: Subject ID for auto-detection.
        session_id: Session ID for auto-detection.

    Returns:
        Tuple of (ncbv_file, normalization_factor), or (None, None) if no NAWM mask.
    """
    from pathlib import Path

    # Skip normalization if no NAWM mask available
    if nawm_mask_file is None:
        return None, None

    nawm_path = Path(nawm_mask_file)
    if not nawm_path.exists():
        return None, None

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
        brain_mask_file=None,  # Don't use fallback
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

    Returns (None, None) when NAWM mask is not available - normalization
    requires NAWM reference for clinically valid results.

    Args:
        cbf_file: Path to CBF map NIfTI.
        brain_mask_file: Path to brain mask (unused, kept for interface compat).
        derivatives_dir: Path to derivatives directory for auto-detection.
        subject_id: Subject ID for auto-detection.
        session_id: Session ID for auto-detection.
        nawm_mask_file: Path to NAWM mask NIfTI (required for normalization).

    Returns:
        Tuple of (ncbf_file, normalization_factor), or (None, None) if no NAWM mask.
    """
    from pathlib import Path

    # Skip normalization if no NAWM mask available
    if nawm_mask_file is None:
        return None, None

    nawm_path = Path(nawm_mask_file)
    if not nawm_path.exists():
        return None, None

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
        brain_mask_file=None,  # Don't use fallback
    )

    return str(output_path), norm_factor


def _create_registration_filmstrip(
    dsc_mean_file,
    registered_t1w_file,
    brain_mask_file,
    subject_id="",
    session_id="",
):
    """Create a registration QC filmstrip PNG.

    Generates a multi-slice overlay showing the registered T1w on the DSC mean
    with brain mask contours for visual QC of registration quality.

    Args:
        dsc_mean_file: Path to mean DSC image (reference).
        registered_t1w_file: Path to registered T1w in DSC space.
        brain_mask_file: Path to DSC brain mask.
        subject_id: Subject ID for QC figure title.
        session_id: Session ID for QC figure title.

    Returns:
        Path to the registration filmstrip PNG file.
    """
    import nibabel as nib
    from pathlib import Path
    from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip, qc_figure_title

    out_dir = Path.cwd()
    output_path = str(out_dir / "registration_qc_filmstrip.png")

    dsc_img = nib.load(dsc_mean_file)
    dsc_data = dsc_img.get_fdata()
    t1w_data = nib.load(registered_t1w_file).get_fdata()
    mask_data = nib.load(brain_mask_file).get_fdata()
    voxel_sizes = tuple(dsc_img.header.get_zooms()[:3])

    plot_registration_qc_filmstrip(
        fixed=dsc_data,
        moving=t1w_data,
        output_path=output_path,
        mask=mask_data,
        title=qc_figure_title(subject_id, session_id, "DSC-T1w Registration"),
        voxel_sizes=voxel_sizes,
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


def _generate_motion_qc_plot(motion_params_file, out_dir, tr, subject_id="", session_id=""):
    """Generate motion QC plot from motion parameters.

    Creates a three-panel figure showing raw motion parameters and
    framewise displacement over time.

    Args:
        motion_params_file: Path to motion parameters text file.
        out_dir: Output directory for the plot.
        tr: Repetition time in seconds.
        subject_id: Subject ID for QC figure title.
        session_id: Session ID for QC figure title.

    Returns:
        Tuple of (plot_file path, motion_summary dict).
    """
    import numpy as np
    from pathlib import Path
    from pmriprep.utils.qc_utils import plot_motion_qc

    motion_params = np.loadtxt(motion_params_file)
    out_path = Path(out_dir) / "dsc_motion_qc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, motion_summary = plot_motion_qc(
        motion_params, out_path, tr=tr,
        subject_id=subject_id, session_id=session_id,
    )
    return str(out_path), motion_summary


def _generate_dsc_preprocessing_qc(
    dsc_signal_file,
    brain_mask_file,
    baseline_frames,
    bolus_arrival_frame,
    tr,
    te=None,
    subject_id="",
    session_id="",
):
    """Generate DSC preprocessing QC plot.

    Args:
        dsc_signal_file: Path to motion-corrected 4D DSC signal NIfTI.
        brain_mask_file: Path to brain mask NIfTI.
        baseline_frames: Number of pre-bolus baseline frames.
        bolus_arrival_frame: Frame index of bolus arrival.
        tr: Repetition time in seconds.
        subject_id: Subject identifier.
        session_id: Session identifier.

    Returns:
        Tuple of (plot_file path, stats dict).
    """
    import nibabel as nib
    from pmriprep.utils.dsc_qc_plots import generate_dsc_preprocessing_qc_plot

    signal_img = nib.load(dsc_signal_file)
    signal_4d = signal_img.get_fdata()
    mask_data = nib.load(brain_mask_file).get_fdata()

    plot_path, stats = generate_dsc_preprocessing_qc_plot(
        dsc_signal_4d=signal_4d,
        brain_mask=mask_data,
        baseline_frames=baseline_frames,
        bolus_arrival_frame=bolus_arrival_frame,
        volume_tr_s=tr,
        te_s=te,
        subject_id=subject_id,
        session_id=session_id,
    )
    return plot_path, stats


def _generate_dsc_param_maps_qc(
    rel_cbf_file,
    rel_cbv_file,
    mtt_file,
    tmax_file,
    psr_file,
    ttp_file,
    brain_mask_file,
    subject_id="",
    session_id="",
):
    """Generate DSC parameter maps QC plot.

    Args:
        rel_cbf_file: Path to relative CBF map NIfTI.
        rel_cbv_file: Path to relative CBV map NIfTI.
        mtt_file: Path to MTT map NIfTI.
        tmax_file: Path to Tmax map NIfTI.
        psr_file: Path to PSR map NIfTI.
        ttp_file: Path to TTP map NIfTI.
        brain_mask_file: Path to brain mask NIfTI.
        subject_id: Subject identifier.
        session_id: Session identifier.

    Returns:
        Path to the generated QC plot PNG.
    """
    import nibabel as nib
    from pmriprep.utils.dsc_qc_plots import generate_dsc_parameter_maps_qc_plot

    mask_data = nib.load(brain_mask_file).get_fdata()

    param_maps = {}
    for name, filepath in [
        ("rCBF", rel_cbf_file),
        ("rCBV", rel_cbv_file),
        ("MTT", mtt_file),
        ("Tmax", tmax_file),
        ("PSR", psr_file),
        ("TTP\u1d9c", ttp_file),
    ]:
        if filepath:
            from pathlib import Path as _P
            if _P(filepath).exists():
                param_maps[name] = nib.load(filepath).get_fdata()

    plot_path = generate_dsc_parameter_maps_qc_plot(
        param_maps=param_maps,
        brain_mask=mask_data,
        subject_id=subject_id,
        session_id=session_id,
    )
    return plot_path


def _generate_dsc_leakage_qc(
    k2_map_file,
    auc_raw_file,
    auc_corr_file,
    auc_diff_file,
    brain_mask_file,
    concentration_file=None,
    corrected_concentration_file=None,
    psr_file=None,
    tr=1.5,
    subject_id="",
    session_id="",
):
    """Generate DSC leakage correction QC plot.

    Args:
        k2_map_file: Path to K2 map NIfTI.
        auc_raw_file: Path to AUC raw map NIfTI.
        auc_corr_file: Path to AUC corrected map NIfTI.
        auc_diff_file: Path to AUC diff map NIfTI.
        brain_mask_file: Path to brain mask NIfTI.
        concentration_file: Path to pre-K2 4D concentration NIfTI (optional).
        corrected_concentration_file: Path to post-K2 4D concentration NIfTI (optional).
        psr_file: Path to PSR map NIfTI (optional).
        tr: Repetition time in seconds.
        subject_id: Subject identifier.
        session_id: Session identifier.

    Returns:
        Path to the generated QC plot PNG, or empty string if K2 unavailable.
    """
    from pathlib import Path as _P

    # Skip if no K2 map (guard against "." or directory paths from unset outputs)
    if not k2_map_file or not _P(k2_map_file).is_file():
        return ""

    import nibabel as nib
    from pmriprep.utils.dsc_qc_plots import generate_dsc_leakage_qc_plot

    k2_data = nib.load(k2_map_file).get_fdata()
    auc_raw_data = nib.load(auc_raw_file).get_fdata()
    auc_corr_data = nib.load(auc_corr_file).get_fdata()
    auc_diff_data = nib.load(auc_diff_file).get_fdata()
    mask_data = nib.load(brain_mask_file).get_fdata()

    conc_raw_4d = None
    if concentration_file and _P(concentration_file).exists():
        conc_raw_4d = nib.load(concentration_file).get_fdata()

    conc_corr_4d = None
    if corrected_concentration_file and _P(corrected_concentration_file).exists():
        conc_corr_4d = nib.load(corrected_concentration_file).get_fdata()

    psr_data = None
    if psr_file and _P(psr_file).exists():
        psr_data = nib.load(psr_file).get_fdata()

    plot_path = generate_dsc_leakage_qc_plot(
        k2_map=k2_data,
        auc_raw=auc_raw_data,
        auc_corr=auc_corr_data,
        auc_diff=auc_diff_data,
        brain_mask=mask_data,
        conc_raw_4d=conc_raw_4d,
        conc_corr_4d=conc_corr_4d,
        psr=psr_data,
        tr=tr,
        subject_id=subject_id,
        session_id=session_id,
    )
    return plot_path


def _generate_dsc_fit_qc(
    qc_fit_data_file: str,
    brain_mask_file: str,
    cbf_file: str,
    subject_id: str,
    session_id: str,
) -> str:
    """Generate DSC fit QC plot from saved QC fit data JSON.

    Args:
        qc_fit_data_file: Path to QC fit data JSON from ComputeDSCMaps.
        brain_mask_file: Path to brain mask NIfTI.
        cbf_file: Path to relative CBF map NIfTI.
        subject_id: Subject identifier.
        session_id: Session identifier.

    Returns:
        Path to the generated QC plot PNG, or "" if unavailable.
    """
    import json
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    if not qc_fit_data_file or not Path(qc_fit_data_file).exists():
        return ""

    with open(qc_fit_data_file) as f:
        fit_data = json.load(f)

    tissue_conc = np.array(fit_data["tissue_concentration"])
    model_pred = np.array(fit_data["model_prediction"])
    aif_curve = np.array(fit_data["aif"])
    time_points = np.array(fit_data["time_points"])
    voxel = tuple(fit_data["voxel"])
    voxel_cbf = fit_data["voxel_cbf"]
    r_squared = fit_data.get("r_squared", 0.0)
    method = fit_data.get("method", "")
    raw_signal_data = fit_data.get("raw_signal")
    raw_signal = np.array(raw_signal_data) if raw_signal_data is not None else None

    # Load CBF map for spatial context panel
    cbf_img = nib.load(cbf_file)
    cbf_data = cbf_img.get_fdata()  # type: ignore[attr-defined]
    mask_img = nib.load(brain_mask_file)
    mask_data = mask_img.get_fdata() > 0  # type: ignore[attr-defined]

    out_dir = Path.cwd() / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(
        out_dir / f"sub-{subject_id}_ses-{session_id}_dsc_fit_qc.png"
    )

    from pmriprep.utils.dsc_qc_plots import generate_dsc_fit_qc_plot

    plot_path, _ = generate_dsc_fit_qc_plot(
        tissue_concentration=tissue_conc,
        model_prediction=model_pred,
        aif=aif_curve,
        time_points=time_points,
        cbf_map=cbf_data,
        brain_mask=mask_data,
        qc_voxel=voxel,
        qc_voxel_cbf=voxel_cbf,
        r_squared=r_squared,
        method=method,
        output_path=out_path,
        subject_id=subject_id,
        session_id=session_id,
        raw_signal=raw_signal,
    )
    return plot_path


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
    motion_summary=None,
    aif_metrics=None,
    preprocessing_stats=None,
):
    """Merge all QC metrics into a single dictionary.

    Args:
        motion_qc: Motion correction QC metrics.
        preload_qc: Preload detection and correction QC metrics.
        aif_qc: AIF detection QC metrics.
        deconv_qc: Deconvolution QC metrics.
        registration_qc: Registration QC metrics.
        normalization_factor: CBV normalization factor.
        motion_summary: Motion summary dict from plot_motion_qc (optional).
        aif_metrics: AIF metrics dict from AIFQCPlot (optional).
        preprocessing_stats: Preprocessing stats dict (optional).

    Returns:
        Combined QC metrics dictionary.
    """
    motion_data = motion_qc if motion_qc else {}
    if motion_summary:
        motion_data = dict(motion_data) if motion_data else {}
        motion_data["motion_summary"] = motion_summary

    aif_data = aif_qc if aif_qc else {}
    if aif_metrics:
        aif_data = dict(aif_data) if aif_data else {}
        aif_data["aif_metrics"] = aif_metrics

    merged = {
        "motion": motion_data,
        "preload": preload_qc if preload_qc else {},
        "aif": aif_data,
        "deconvolution": deconv_qc if deconv_qc else {},
        "registration": registration_qc if registration_qc else {},
        "normalization_factor": normalization_factor,
    }
    if preprocessing_stats:
        merged["preprocessing"] = preprocessing_stats
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
        out_dir / f"{prefix}_from-DSC_to-anat_mode-image_xfm.mat"
    )
    t1w_to_dsc_path = str(
        out_dir / f"{prefix}_from-anat_to-DSC_mode-image_xfm.mat"
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
    random_seed: Optional[int] = None,
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
    inputnode.t1w_underlay : str
        T1w image to transform to DSC space as anatomical underlay.
        CE-T1w if post-contrast available, otherwise native T1w.
    inputnode.brain_mask : str
        Path to brain mask in T1w space.
    inputnode.wm_probseg : str, optional
        Path to white matter probability map for NAWM mask.
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
    outputnode.k2_map : str
        Path to K2 leakage correction map.
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
                "t1w_underlay",  # T1w underlay for DSC-space visualization (CE-T1w if available, else native)
                "brain_mask",  # Brain mask in T1w space
                "wm_probseg",  # White matter probability map (optional)
                "whole_tumor_mask",  # Whole-tumor mask for AIF exclusion + lesion stats (optional)
                "enhancing_mask",  # Enhancing tumor mask for laterality (optional)
                "tumor_segmentation",  # Multi-label tumor segmentation for lesion stats (optional)
                "nawm_mask",  # NAWM mask for normalization (optional)
                "roi_mask",  # Generic ROI mask for stats extraction only (optional)
                "t1w_to_mni_xfm",  # T1w to MNI transform (from ACPC alignment)
                "te",  # Echo time in seconds
                "field_strength",  # Magnetic field strength
                "tr",  # Repetition time in seconds
                "flair_in_t1w",  # FLAIR coregistered to T1w space (from anat)
                "t2w_in_t1w",  # T2w coregistered to T1w space (from anat)
            ]
        ),
        name="inputnode",
    )

    # Fallback defaults — overridden by BIDS sidecar values in base.py
    inputnode.inputs.te = 0.035  # 35ms echo time (typical DSC-MRI at 3T)
    inputnode.inputs.field_strength = 3.0  # 3T
    inputnode.inputs.tr = 1.5  # 1.5s TR

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
                "ttp_corrected_map",
                "rttp_map",
                "cnr_map",
                # AUC decomposition
                "auc_diff_map",
                # Vascular model parameters (optional)
                "alpha_map",
                "beta_map",
                "cth_map",
                # Bayesian uncertainty maps (optional)
                "cbf_std_map",
                "cbv_std_map",
                "cbf_cv_map",
                "cbv_cv_map",
                "cbf_ci95_lower_map",
                "cbf_ci95_upper_map",
                "cbv_ci95_lower_map",
                "cbv_ci95_upper_map",
                # Relative maps (always produced)
                "rel_cbf_map",
                "rel_cbv_map",
                "rel_auc_raw_map",
                "rel_auc_corr_map",
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
                "t1w_to_dsc_xfm",
                "registration_filmstrip",
                # Structural underlays
                "t1w_in_dsc",
                "flair_in_dsc",
                "t2w_in_dsc",
                # AIF QC plots
                "aif_curve_plot",
                "aif_location_plot",
                # Smoothed maps (populated only if smooth_fwhm > 0)
                "smooth_hemodynamic",
                "smooth_auc",
                # QC metrics
                "qc_metrics",
                # Masks in DSC space
                "brain_mask_dsc",
                "nawm_mask_dsc",
                # Lesion statistics (conditional on lesion mask)
                "lesion_stats_json",
                "lesion_stats_txt",
                "lesion_mask_dsc",
                # ROI statistics (conditional on roi_mask)
                "roi_stats_json",
                "roi_stats_txt",
                "roi_mask_dsc",
                # Motion QC
                "motion_qc_plot",
                # DSC QC plots
                "dsc_preprocessing_qc_plot",
                "dsc_param_maps_qc_plot",
                "dsc_fit_qc_plot",
                "dsc_leakage_qc_plot",
                # Brain mask filmstrip
                "brain_mask_filmstrip",
            ]
        ),
        name="outputnode",
    )

    # -------------------------------------------------------------------
    # Node 0: Reorient DSC to RAS orientation
    # -------------------------------------------------------------------
    dsc_reorient = pe.Node(
        Reorient(),
        name="dsc_reorient",
    )

    workflow.connect([
        (inputnode, dsc_reorient, [("dsc_file", "in_file")]),
    ])

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
    # Node 1a: Motion QC Plot
    # Generate motion QC visualization from motion parameters.
    # -------------------------------------------------------------------
    motion_qc_plot = pe.Node(
        niu.Function(
            input_names=["motion_params_file", "out_dir", "tr", "subject_id", "session_id"],
            output_names=["plot_file", "motion_summary"],
            function=_generate_motion_qc_plot,
        ),
        name="motion_qc_plot",
    )
    motion_qc_plot.inputs.out_dir = str(bids_dir / "derivatives" / "pmriprep")
    motion_qc_plot.inputs.subject_id = subject_id
    motion_qc_plot.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 1b: Pre-Bolus Mean Reference
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
    # Node 1b: HD-BET Brain Extraction (for registration quality)
    # HD-BET extracts DSC brain so registration uses brain-extracted
    # images on both sides. The final processing mask comes from
    # transforming the T1w brain mask through the registration.
    # -------------------------------------------------------------------
    hdbet_dsc = pe.Node(
        niu.Function(
            input_names=["dsc_mean_file", "hdbet_device", "hdbet_conda_env"],
            output_names=["dsc_brain_file", "dsc_brain_mask_file"],
            function=_hdbet_dsc_brain,
        ),
        name="hdbet_dsc",
    )
    hdbet_dsc.inputs.hdbet_device = config.hdbet_device
    hdbet_dsc.inputs.hdbet_conda_env = config.hdbet_conda_env or ""

    # -------------------------------------------------------------------
    # Node 1b2: Histogram Match T1w to DSC (for FLIRT convergence)
    # T1w and DSC have inverted contrasts. Without histogram matching,
    # FLIRT's normcorr coarse-search converges to a wrong local optimum.
    # Matches perf-plus registration_utils.py:161-208.
    # -------------------------------------------------------------------
    hist_match_t1w = pe.Node(
        niu.Function(
            input_names=["t1w_file", "dsc_brain_file", "brain_mask_file"],
            output_names=["matched_t1w_file"],
            function=_histogram_match_t1w_to_dsc,
        ),
        name="hist_match_t1w",
    )

    # -------------------------------------------------------------------
    # Node 1c: Transform T1w Brain Mask to DSC Space
    # NOTE: T1w brain mask transform kept for future use. To reinstate,
    # uncomment and reconnect downstream consumers.
    # -------------------------------------------------------------------
    # transform_brain_mask = pe.Node(
    #     niu.Function(
    #         input_names=["brain_mask_file", "reference_file", "transform_matrix",
    #                      "hdbet_mask_file"],
    #         output_names=["dsc_brain_mask_file", "dice_score"],
    #         function=_transform_brain_mask_to_dsc,
    #     ),
    #     name="transform_brain_mask",
    # )

    # -------------------------------------------------------------------
    # Node 2: Signal to Concentration Conversion
    # Always produces ΔR2* (s⁻¹). All downstream quantification is relative.
    # -------------------------------------------------------------------
    signal_to_conc = pe.Node(
        niu.Function(
            input_names=["dsc_file", "te", "baseline_frames"],
            output_names=["concentration_file", "signal_file"],
            function=_signal_to_concentration,
        ),
        name="signal_to_conc",
    )
    signal_to_conc.inputs.baseline_frames = config.pre_bolus

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
            gamma_fit=config.gamma_fit,
            disable_fallback=config.disable_aif_fallback,
            baseline_frames=config.pre_bolus,
        ),
        name="aif_detect",
    )

    # Wire user-supplied AIF mask if provided (overrides atlas mask)
    if config.aif_mask_file is not None:
        aif_detect.inputs.aif_mask_file = str(config.aif_mask_file)

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
                "preload_present",
                "signal_file",
                "baseline_frames",
            ],
            output_names=["corrected_concentration_file", "k2_map_file", "qc_dict"],
            function=_detect_and_correct_preload,
        ),
        name="preload_correct",
    )
    preload_correct.inputs.preload_present = config.preload_present if config.preload_present is not None else None
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
    dsc_map_kwargs: dict[str, Any] = dict(
        method=config.method,
        svd_threshold=0.2,  # Default SVD threshold
        oi_threshold=0.065,  # Default OI threshold for oSVD
        cpi_control_points=config.cpi_control_points,
        bayesian=config.bayesian,
        mcmc_samples=config.mcmc_samples,
        use_gpu=config.use_gpu,
        n_procs=config.num_processes if config.num_processes > 0 else 1,
        baseline_frames=config.pre_bolus,
        tail_frames=5,  # Post-bolus frames for PSR (Cha et al. AJNR 2002)
        sloppy=config.sloppy,
    )
    if random_seed is not None:
        dsc_map_kwargs["random_seed"] = random_seed
    compute_maps = pe.Node(
        ComputeDSCMaps(**dsc_map_kwargs),
        name="compute_maps",
    )

    # -------------------------------------------------------------------
    # Node 6: DSC to T1w Registration
    # -------------------------------------------------------------------
    register = pe.Node(
        DSCRegistration(
            registration=config.registration,
            dof=6,  # Rigid body
            cost="normmi",
        ),
        name="register",
    )

    # -------------------------------------------------------------------
    # Node 6b: Transform Chain for MNI Atlas to DSC Space
    # Creates MNI→DSC transform by concatenating:
    #   MNI→T1w (inverted T1w→MNI) + T1w→DSC (register output)
    # -------------------------------------------------------------------

    # Invert T1w→MNI to get MNI→T1w
    invert_mni_xfm = pe.Node(
        ConvertXFM(invert_xfm=True),
        name="invert_mni_xfm",
    )

    # Invert T1w→DSC to get DSC→T1w (for output labeling)
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
    # Set the arterial atlas as input (bundled with pmriprep, FSL MNI152 space)
    try:
        transform_atlas.inputs.in_file = str(get_arterial_atlas("2mm", "dil2"))
    except FileNotFoundError:
        import warnings
        warnings.warn(
            "Arterial atlas not found. AIF detection will use whole-brain search."
        )

    # -------------------------------------------------------------------
    # Node 6c: Transform NAWM mask from T1w to DSC space
    # -------------------------------------------------------------------
    transform_nawm = pe.Node(
        niu.Function(
            input_names=["nawm_mask_file", "reference_file", "transform_matrix"],
            output_names=["nawm_mask_dsc_file"],
            function=_transform_nawm_to_dsc,
        ),
        name="transform_nawm",
    )

    # -------------------------------------------------------------------
    # Node 6d: Select tumor mask for lesion statistics
    # Priority: tumor_segmentation (multi-label) > whole_tumor > None
    # -------------------------------------------------------------------
    from pmriprep.utils.lesion_stats import combine_tumor_masks

    combine_masks = pe.Node(
        niu.Function(
            input_names=[
                "tumor_segmentation_file",
                "whole_tumor_mask_file",
                "reference_file",
            ],
            output_names=["combined_mask_file"],
            function=combine_tumor_masks,
        ),
        name="combine_masks",
    )

    # -------------------------------------------------------------------
    # Node 6e: Transform combined lesion mask from T1w to DSC space
    # -------------------------------------------------------------------
    transform_lesion = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dsc_file"],
            function=_transform_lesion_mask_to_dsc,
        ),
        name="transform_lesion",
    )

    # -------------------------------------------------------------------
    # Node 6f: Lesion Statistics Extraction
    # Conditional - only run if lesion mask is available
    # -------------------------------------------------------------------
    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

    lesion_stats = pe.Node(
        DSCLesionStats(),
        name="lesion_stats",
    )
    lesion_stats.inputs.subject_id = subject_id
    lesion_stats.inputs.session_id = session_id
    lesion_stats.inputs.output_dir = str(bids_dir / "derivatives" / "pmriprep")

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

    # -------------------------------------------------------------------
    # Node 7c: Smoothing (conditional on config.smooth_fwhm > 0)
    # -------------------------------------------------------------------
    smooth_hemodynamic = None
    smooth_auc = None

    if config.smooth_fwhm > 0:
        # Smoothing for hemodynamic maps: CBF, CBV, MTT, nCBF, nCBV
        # Physics-approved: intensive quantities that vary smoothly
        smooth_hemodynamic = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_maps,
            ),
            name="smooth_hemodynamic",
        )
        smooth_hemodynamic.inputs.fwhm = config.smooth_fwhm

        # Smoothing for leakage-corrected AUC (CBV surrogate)
        # Only auc_corr; auc_raw contains leakage artifact
        smooth_auc = pe.Node(
            niu.Function(
                input_names=["map_files", "brain_mask_file", "fwhm"],
                output_names=["smoothed_files"],
                function=_smooth_maps,
            ),
            name="smooth_auc",
        )
        smooth_auc.inputs.fwhm = config.smooth_fwhm

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
                "motion_summary",
                "aif_metrics",
                "preprocessing_stats",
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
            input_names=["dsc_mean_file", "registered_t1w_file", "brain_mask_file",
                         "subject_id", "session_id"],
            output_names=["filmstrip_file"],
            function=_create_registration_filmstrip,
        ),
        name="reg_filmstrip",
    )
    reg_filmstrip.inputs.subject_id = subject_id
    reg_filmstrip.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9b: AIF QC Plots
    # -------------------------------------------------------------------
    aif_qc_plot = pe.Node(
        AIFQCPlot(
            subject_id=subject_id,
            session_id=session_id or "",
        ),
        name="aif_qc_plot",
    )

    # -------------------------------------------------------------------
    # Node 9c: DSC Preprocessing QC Plot
    # -------------------------------------------------------------------
    dsc_preproc_qc = pe.Node(
        niu.Function(
            input_names=[
                "dsc_signal_file", "brain_mask_file",
                "baseline_frames", "bolus_arrival_frame",
                "tr", "te", "subject_id", "session_id",
            ],
            output_names=["plot_file", "stats"],
            function=_generate_dsc_preprocessing_qc,
        ),
        name="dsc_preproc_qc",
    )
    dsc_preproc_qc.inputs.baseline_frames = config.pre_bolus
    dsc_preproc_qc.inputs.bolus_arrival_frame = config.pre_bolus  # default, may be overridden
    dsc_preproc_qc.inputs.subject_id = subject_id
    dsc_preproc_qc.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9d: DSC Parameter Maps QC Plot
    # -------------------------------------------------------------------
    dsc_param_maps_qc = pe.Node(
        niu.Function(
            input_names=[
                "rel_cbf_file", "rel_cbv_file", "mtt_file",
                "tmax_file", "psr_file", "ttp_file",
                "brain_mask_file", "subject_id", "session_id",
            ],
            output_names=["plot_file"],
            function=_generate_dsc_param_maps_qc,
        ),
        name="dsc_param_maps_qc",
    )
    dsc_param_maps_qc.inputs.subject_id = subject_id
    dsc_param_maps_qc.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9e: DSC Leakage Correction QC Plot (conditional on K2)
    # -------------------------------------------------------------------
    dsc_leakage_qc = pe.Node(
        niu.Function(
            input_names=[
                "k2_map_file", "auc_raw_file", "auc_corr_file",
                "auc_diff_file", "brain_mask_file",
                "concentration_file", "corrected_concentration_file",
                "psr_file", "tr", "subject_id", "session_id",
            ],
            output_names=["plot_file"],
            function=_generate_dsc_leakage_qc,
        ),
        name="dsc_leakage_qc",
    )
    dsc_leakage_qc.inputs.subject_id = subject_id
    dsc_leakage_qc.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9e2: DSC Fit QC Plot (all deconvolution methods)
    # -------------------------------------------------------------------
    dsc_fit_qc = pe.Node(
        niu.Function(
            input_names=[
                "qc_fit_data_file", "brain_mask_file", "cbf_file",
                "subject_id", "session_id",
            ],
            output_names=["plot_file"],
            function=_generate_dsc_fit_qc,
        ),
        name="dsc_fit_qc",
    )
    dsc_fit_qc.inputs.subject_id = subject_id
    dsc_fit_qc.inputs.session_id = session_id or ""

    # -------------------------------------------------------------------
    # Node 9f: Transform structural underlays to DSC Space
    # register.out_matrix is T1w→DSC, so we use it directly.
    # Underlays are conditional based on config.underlays list.
    # Uses Function interface to handle undefined inputs gracefully.
    # -------------------------------------------------------------------
    transform_t1_to_dsc = None
    if "t1w_precon" in config.underlays or "t1w_postcon" in config.underlays:
        transform_t1_to_dsc = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dsc,
            ),
            name="transform_t1_to_dsc",
        )
        transform_t1_to_dsc.inputs.output_name = "t1w_in_dsc.nii.gz"

    transform_flair_to_dsc = None
    if "flair" in config.underlays:
        transform_flair_to_dsc = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dsc,
            ),
            name="transform_flair_to_dsc",
        )
        transform_flair_to_dsc.inputs.output_name = "flair_in_dsc.nii.gz"

    transform_t2w_to_dsc = None
    if "t2w" in config.underlays:
        transform_t2w_to_dsc = pe.Node(
            niu.Function(
                input_names=["structural_file", "reference_file", "transform_matrix", "output_name"],
                output_names=["out_file"],
                function=_transform_structural_to_dsc,
            ),
            name="transform_t2w_to_dsc",
        )
        transform_t2w_to_dsc.inputs.output_name = "t2w_in_dsc.nii.gz"

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

    # Input to Motion Correction (using reoriented DSC file)
    workflow.connect([
        (dsc_reorient, motion_correct, [("out_file", "in_file")]),
    ])

    # Motion Correction to Pre-Bolus Mean
    workflow.connect([
        (motion_correct, prebolus_mean, [("out_file", "moco_file")]),
    ])

    # Motion Correction to Motion QC Plot
    workflow.connect([
        (motion_correct, motion_qc_plot, [("out_params", "motion_params_file")]),
        (inputnode, motion_qc_plot, [("tr", "tr")]),
    ])

    # Pre-Bolus Mean to HD-BET Brain Extraction
    # HD-BET on pre-bolus mean for brain-extracted DSC (improves registration)
    workflow.connect([
        (prebolus_mean, hdbet_dsc, [("prebolus_mean_file", "dsc_mean_file")]),
    ])

    # Brain mask filmstrip QC (after HD-BET)
    dsc_brain_mask_qc = pe.Node(
        niu.Function(
            input_names=["mean_file", "brain_mask_file"],
            output_names=["out_file"],
            function=_create_perfusion_brain_mask_filmstrip,
        ),
        name="dsc_brain_mask_qc",
    )
    workflow.connect([
        (prebolus_mean, dsc_brain_mask_qc, [
            ("prebolus_mean_file", "mean_file"),
        ]),
        (hdbet_dsc, dsc_brain_mask_qc, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # NOTE: transform_brain_mask disconnected — using HD-BET mask directly.
    # To reinstate T1w-transformed mask, uncomment the node definition
    # above and restore these connections:
    # workflow.connect([
    #     (inputnode, transform_brain_mask, [("brain_mask", "brain_mask_file")]),
    #     (prebolus_mean, transform_brain_mask, [("prebolus_mean_file", "reference_file")]),
    #     (register, transform_brain_mask, [("out_matrix", "transform_matrix")]),
    #     (hdbet_dsc, transform_brain_mask, [("dsc_brain_mask_file", "hdbet_mask_file")]),
    # ])

    # Motion Correction to Signal Conversion
    workflow.connect([
        (motion_correct, signal_to_conc, [("out_file", "dsc_file")]),
        (inputnode, signal_to_conc, [
            ("te", "te"),
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
        (hdbet_dsc, aif_detect, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, aif_detect, [
            ("whole_tumor_mask", "lesion_mask_file"),  # Undilated mask for AIF distance exclusion
            ("enhancing_mask", "enhancing_mask_file"),
            ("tr", "tr"),
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
        (hdbet_dsc, preload_correct, [
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
            ("k2_map_file", "k2_map_input_file"),
        ]),
        (aif_detect, compute_maps, [
            ("out_aif_curve", "aif_file"),
            ("out_aif_mask", "aif_mask_file"),
            ("out_aif_coord", "aif_best_coord"),
            ("detection_level", "aif_detection_level"),
        ]),
        (hdbet_dsc, compute_maps, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, compute_maps, [
            ("tr", "tr"),
            ("te", "te"),
            ("field_strength", "field_strength"),
            ("enhancing_mask", "enhancing_mask_file"),
        ]),
        (motion_correct, compute_maps, [
            ("fd", "fd_values"),
        ]),
        (signal_to_conc, compute_maps, [
            ("signal_file", "signal_file"),
            ("concentration_file", "raw_concentration_file"),
        ]),
        (extract_preload, compute_maps, [
            ("preload_classification", "preload_classification"),
        ]),
    ])

    # Histogram match T1w to DSC for registration convergence
    # Brain mask applied to T1w before matching (perf-plus uses $T1W_BRAIN)
    workflow.connect([
        (inputnode, hist_match_t1w, [
            ("t1w_reference", "t1w_file"),
            ("brain_mask", "brain_mask_file"),
        ]),
        (hdbet_dsc, hist_match_t1w, [("dsc_brain_file", "dsc_brain_file")]),
    ])

    # DSC Map Computation to Registration
    # Use brain-extracted DSC (from HD-BET) and histogram-matched T1w
    workflow.connect([
        (hist_match_t1w, register, [("matched_t1w_file", "t1w_file")]),
        (inputnode, register, [
            ("brain_mask", "brain_mask"),
            ("wm_probseg", "wm_seg"),
        ]),
        (hdbet_dsc, register, [("dsc_brain_file", "dsc_mean")]),
    ])

    # Transform Chain: Create MNI→DSC transform for atlas
    # Step 1: Invert T1w→MNI to get MNI→T1w
    workflow.connect([
        (inputnode, invert_mni_xfm, [("t1w_to_mni_xfm", "in_file")]),
    ])

    # Step 2: Invert T1w→DSC to get DSC→T1w (for output labeling)
    workflow.connect([
        (register, invert_dsc_xfm, [("out_matrix", "in_file")]),
    ])

    # Step 3: Concatenate MNI→T1w + T1w→DSC = MNI→DSC
    workflow.connect([
        (invert_mni_xfm, concat_mni_to_dsc, [("out_file", "in_file")]),
        (register, concat_mni_to_dsc, [("out_matrix", "in_file2")]),
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

    # Transform NAWM mask from T1w to DSC space
    workflow.connect([
        (inputnode, transform_nawm, [("nawm_mask", "nawm_mask_file")]),
        (prebolus_mean, transform_nawm, [("prebolus_mean_file", "reference_file")]),
        (register, transform_nawm, [("out_matrix", "transform_matrix")]),
    ])

    # Wire NAWM mask to compute_maps for rTTP normalization
    workflow.connect([
        (transform_nawm, compute_maps, [
            ("nawm_mask_dsc_file", "nawm_mask_file"),
        ]),
    ])

    # DSC Map Computation to CBV Normalization (uses relative CBV)
    # Use DSC-space brain mask for fallback normalization
    workflow.connect([
        (compute_maps, normalize, [("out_rel_cbv", "cbv_file")]),
        (transform_nawm, normalize, [
            ("nawm_mask_dsc_file", "nawm_mask_file"),
        ]),
        (hdbet_dsc, normalize, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # DSC Map Computation to CBF Normalization (uses relative CBF)
    workflow.connect([
        (compute_maps, normalize_cbf, [("out_rel_cbf", "cbf_file")]),
        (transform_nawm, normalize_cbf, [
            ("nawm_mask_dsc_file", "nawm_mask_file"),
        ]),
        (hdbet_dsc, normalize_cbf, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # Registration QC filmstrip
    workflow.connect([
        (prebolus_mean, reg_filmstrip, [("prebolus_mean_file", "dsc_mean_file")]),
        (register, reg_filmstrip, [("out_file", "registered_t1w_file")]),
        (hdbet_dsc, reg_filmstrip, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
    ])

    # AIF QC plots
    workflow.connect([
        (aif_detect, aif_qc_plot, [
            ("out_aif_curve", "aif_curve_file"),
            ("out_aif_mask", "aif_mask_file"),
            ("qc_metrics", "qc_metrics"),
            ("out_raw_aif_curve", "raw_aif_curve_file"),
            ("gamma_fit_applied", "gamma_fit_applied"),
        ]),
        (motion_correct, aif_qc_plot, [("out_file", "dsc_file")]),
        (hdbet_dsc, aif_qc_plot, [
            ("dsc_brain_mask_file", "brain_mask_file"),
        ]),
        (inputnode, aif_qc_plot, [("tr", "tr")]),
    ])

    # DSC Preprocessing QC
    workflow.connect([
        (motion_correct, dsc_preproc_qc, [("out_file", "dsc_signal_file")]),
        (hdbet_dsc, dsc_preproc_qc, [("dsc_brain_mask_file", "brain_mask_file")]),
        (inputnode, dsc_preproc_qc, [("tr", "tr"), ("te", "te")]),
    ])

    # DSC Parameter Maps QC
    workflow.connect([
        (compute_maps, dsc_param_maps_qc, [
            ("out_rel_cbf", "rel_cbf_file"),
            ("out_rel_cbv", "rel_cbv_file"),
            ("out_mtt", "mtt_file"),
            ("out_tmax", "tmax_file"),
            ("out_psr", "psr_file"),
            ("out_ttp_corrected", "ttp_file"),
        ]),
        (hdbet_dsc, dsc_param_maps_qc, [("dsc_brain_mask_file", "brain_mask_file")]),
    ])

    # DSC Leakage Correction QC
    workflow.connect([
        (compute_maps, dsc_leakage_qc, [
            ("out_k2", "k2_map_file"),
            ("out_auc_raw", "auc_raw_file"),
            ("out_auc_corr", "auc_corr_file"),
            ("out_auc_diff", "auc_diff_file"),
            ("out_psr", "psr_file"),
        ]),
        (hdbet_dsc, dsc_leakage_qc, [("dsc_brain_mask_file", "brain_mask_file")]),
        (signal_to_conc, dsc_leakage_qc, [("concentration_file", "concentration_file")]),
        (preload_correct, dsc_leakage_qc, [("corrected_concentration_file", "corrected_concentration_file")]),
        (inputnode, dsc_leakage_qc, [("tr", "tr")]),
    ])

    # DSC Fit QC (all deconvolution methods)
    workflow.connect([
        (compute_maps, dsc_fit_qc, [
            ("out_qc_fit_data", "qc_fit_data_file"),
            ("out_rel_cbf", "cbf_file"),
        ]),
        (hdbet_dsc, dsc_fit_qc, [("dsc_brain_mask_file", "brain_mask_file")]),
    ])

    # -------------------------------------------------------------------
    # Lesion mask selection and transform (conditional)
    # -------------------------------------------------------------------
    # Select best available tumor mask for lesion stats
    # Priority: tumor_segmentation (multi-label) > whole_tumor
    workflow.connect([
        (inputnode, combine_masks, [
            ("tumor_segmentation", "tumor_segmentation_file"),
            ("whole_tumor_mask", "whole_tumor_mask_file"),
        ]),
        (prebolus_mean, combine_masks, [("prebolus_mean_file", "reference_file")]),
    ])

    # Transform combined lesion mask from T1w to DSC space
    workflow.connect([
        (combine_masks, transform_lesion, [("combined_mask_file", "lesion_mask_file")]),
        (prebolus_mean, transform_lesion, [("prebolus_mean_file", "reference_file")]),
        (register, transform_lesion, [("out_matrix", "transform_matrix")]),
    ])

    # Lesion statistics extraction
    # Connect lesion mask in DSC space
    workflow.connect([
        (transform_lesion, lesion_stats, [("lesion_mask_dsc_file", "lesion_mask")]),
        (transform_nawm, lesion_stats, [("nawm_mask_dsc_file", "nawm_mask")]),
    ])
    # Connect perfusion maps
    workflow.connect([
        (compute_maps, lesion_stats, [
            ("out_rel_cbf", "cbf_map"),
            ("out_rel_cbv", "cbv_map"),
            ("out_mtt", "mtt_map"),
            ("out_tmax", "tmax_map"),
            ("out_psr", "psr_map"),
            ("out_ph", "ph_map"),
            ("out_ttp_corrected", "ttp_corrected_map"),
            ("out_rttp", "rttp_map"),
        ]),
        (normalize, lesion_stats, [("ncbv_file", "ncbv_map")]),
        (normalize_cbf, lesion_stats, [("ncbf_file", "ncbf_map")]),
        (compute_maps, lesion_stats, [("out_k2", "k2_map")]),
    ])
    # Connect lesion stats outputs to outputnode
    workflow.connect([
        (transform_lesion, outputnode, [("lesion_mask_dsc_file", "lesion_mask_dsc")]),
        (lesion_stats, outputnode, [
            ("stats_json", "lesion_stats_json"),
            ("stats_txt", "lesion_stats_txt"),
        ]),
    ])

    # -------------------------------------------------------------------
    # ROI statistics (conditional on roi_mask input)
    # -------------------------------------------------------------------
    transform_roi = pe.Node(
        niu.Function(
            input_names=["lesion_mask_file", "reference_file", "transform_matrix"],
            output_names=["lesion_mask_dsc_file"],
            function=_transform_lesion_mask_to_dsc,
        ),
        name="transform_roi",
    )

    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats as _DSCLesionStatsROI

    roi_stats = pe.Node(
        _DSCLesionStatsROI(force_binary=True),
        name="roi_stats",
    )
    roi_stats.inputs.subject_id = subject_id
    roi_stats.inputs.session_id = session_id
    roi_stats.inputs.output_dir = str(bids_dir / "derivatives" / "pmriprep")

    # Transform ROI mask from T1w to DSC space
    workflow.connect([
        (inputnode, transform_roi, [("roi_mask", "lesion_mask_file")]),
        (prebolus_mean, transform_roi, [("prebolus_mean_file", "reference_file")]),
        (register, transform_roi, [("out_matrix", "transform_matrix")]),
    ])
    # Connect perfusion maps to ROI stats
    workflow.connect([
        (transform_roi, roi_stats, [("lesion_mask_dsc_file", "lesion_mask")]),
        (compute_maps, roi_stats, [
            ("out_rel_cbf", "cbf_map"),
            ("out_rel_cbv", "cbv_map"),
            ("out_mtt", "mtt_map"),
            ("out_tmax", "tmax_map"),
            ("out_psr", "psr_map"),
            ("out_ph", "ph_map"),
            ("out_ttp_corrected", "ttp_corrected_map"),
            ("out_rttp", "rttp_map"),
        ]),
        (normalize, roi_stats, [("ncbv_file", "ncbv_map")]),
        (normalize_cbf, roi_stats, [("ncbf_file", "ncbf_map")]),
        (compute_maps, roi_stats, [("out_k2", "k2_map")]),
    ])
    # Connect ROI stats outputs to outputnode
    workflow.connect([
        (transform_roi, outputnode, [("lesion_mask_dsc_file", "roi_mask_dsc")]),
        (roi_stats, outputnode, [
            ("stats_json", "roi_stats_json"),
            ("stats_txt", "roi_stats_txt"),
        ]),
    ])

    # Expose DSC-space masks on outputnode
    workflow.connect([
        (hdbet_dsc, outputnode, [("dsc_brain_mask_file", "brain_mask_dsc")]),
        (transform_nawm, outputnode, [("nawm_mask_dsc_file", "nawm_mask_dsc")]),
    ])

    # Transform structural underlays to DSC space (conditional)
    # register.out_matrix is T1w→DSC
    # Use prebolus_mean (full FOV) as reference for complete DSC geometry
    if transform_t1_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_t1_to_dsc, [("t1w_underlay", "structural_file")]),
            (prebolus_mean, transform_t1_to_dsc, [("prebolus_mean_file", "reference_file")]),
            (register, transform_t1_to_dsc, [("out_matrix", "transform_matrix")]),
        ])

    if transform_flair_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_flair_to_dsc, [("flair_in_t1w", "structural_file")]),
            (prebolus_mean, transform_flair_to_dsc, [("prebolus_mean_file", "reference_file")]),
            (register, transform_flair_to_dsc, [("out_matrix", "transform_matrix")]),
        ])

    if transform_t2w_to_dsc is not None:
        workflow.connect([
            (inputnode, transform_t2w_to_dsc, [("t2w_in_t1w", "structural_file")]),
            (prebolus_mean, transform_t2w_to_dsc, [("prebolus_mean_file", "reference_file")]),
            (register, transform_t2w_to_dsc, [("out_matrix", "transform_matrix")]),
        ])

    # -------------------------------------------------------------------
    # Smoothing connections (only if smooth_fwhm > 0)
    # -------------------------------------------------------------------
    if smooth_hemodynamic is not None:
        # Build dict of hemodynamic maps for smoothing:
        # MTT, rel_CBF, rel_CBV, plus optional abs_CBF, abs_CBV, nCBF, nCBV
        # Physics-approved: intensive quantities that vary smoothly across tissue.
        # NOT smoothed: Tmax, TTP, delay (timing), PSR, PH, K2, auc_diff (BBB/leakage)
        def _build_hemodynamic_dict(
            mtt, rel_cbf, rel_cbv, ncbf, ncbv,
        ):
            from pathlib import Path as _P
            d = {"mtt": mtt, "rel_cbf": rel_cbf, "rel_cbv": rel_cbv}
            for name, val in [
                ("ncbf", ncbf), ("ncbv", ncbv),
            ]:
                if isinstance(val, str) and val not in ("", "None") and _P(val).exists():
                    d[name] = val
            return d

        build_hemo_dict = pe.Node(
            niu.Function(
                input_names=[
                    "mtt", "rel_cbf", "rel_cbv",
                    "ncbf", "ncbv",
                ],
                output_names=["map_dict"],
                function=_build_hemodynamic_dict,
            ),
            name="build_hemo_dict",
        )
        workflow.connect([
            (compute_maps, build_hemo_dict, [
                ("out_mtt", "mtt"),
                ("out_rel_cbf", "rel_cbf"),
                ("out_rel_cbv", "rel_cbv"),
            ]),
            (normalize_cbf, build_hemo_dict, [("ncbf_file", "ncbf")]),
            (normalize, build_hemo_dict, [("ncbv_file", "ncbv")]),
            (build_hemo_dict, smooth_hemodynamic, [("map_dict", "map_files")]),
            (hdbet_dsc, smooth_hemodynamic, [
                ("dsc_brain_mask_file", "brain_mask_file"),
            ]),
        ])

        # Build dict of leakage-corrected AUC maps for smoothing (CBV surrogate).
        # Only auc_corr is smoothed; auc_raw contains leakage artifact.
        def _build_auc_corr_dict(rel_auc_corr):
            return {"rel_auc_corr": rel_auc_corr}

        build_auc_dict = pe.Node(
            niu.Function(
                input_names=["rel_auc_corr"],
                output_names=["map_dict"],
                function=_build_auc_corr_dict,
            ),
            name="build_auc_dict",
        )
        workflow.connect([
            (compute_maps, build_auc_dict, [
                ("out_rel_auc_corr", "rel_auc_corr"),
            ]),
            (build_auc_dict, smooth_auc, [("map_dict", "map_files")]),
            (hdbet_dsc, smooth_auc, [
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
        (motion_qc_plot, merge_qc, [("motion_summary", "motion_summary")]),
        (aif_qc_plot, merge_qc, [("out_aif_metrics", "aif_metrics")]),
        (dsc_preproc_qc, merge_qc, [("stats", "preprocessing_stats")]),
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
            ("out_ttp_corrected", "ttp_corrected_map"),
            ("out_rttp", "rttp_map"),
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
        # Quality metrics
        (compute_maps, outputnode, [
            ("out_snr", "snr_map"),
            ("out_tsnr", "tsnr_map"),
        ]),
        # Vascular model parameters (optional — undefined for SVD methods)
        (compute_maps, outputnode, [
            ("out_alpha", "alpha_map"),
            ("out_beta", "beta_map"),
            ("out_cth", "cth_map"),
        ]),
        # Bayesian uncertainty maps (optional — undefined when bayesian=none)
        (compute_maps, outputnode, [
            ("out_cbf_std", "cbf_std_map"),
            ("out_cbv_std", "cbv_std_map"),
            ("out_cbf_cv", "cbf_cv_map"),
            ("out_cbv_cv", "cbv_cv_map"),
            ("out_cbf_ci95_lower", "cbf_ci95_lower_map"),
            ("out_cbf_ci95_upper", "cbf_ci95_upper_map"),
            ("out_cbv_ci95_lower", "cbv_ci95_lower_map"),
            ("out_cbv_ci95_upper", "cbv_ci95_upper_map"),
        ]),
        # AIF outputs
        (aif_detect, outputnode, [
            ("out_aif_curve", "aif_curve"),
            ("out_aif_mask", "aif_mask"),
            ("qc_metrics", "aif_qc"),
        ]),
        # Leakage correction (use recomputed K2 from compute_maps when baseline differs)
        (compute_maps, outputnode, [("out_k2", "k2_map")]),
        # Normalized CBV and CBF
        (normalize, outputnode, [("ncbv_file", "ncbv_map")]),
        (normalize_cbf, outputnode, [("ncbf_file", "ncbf_map")]),
        # Registration
        (invert_dsc_xfm, outputnode, [("out_file", "dsc_to_t1w_xfm")]),
        (register, outputnode, [("out_matrix", "t1w_to_dsc_xfm")]),
        (reg_filmstrip, outputnode, [("filmstrip_file", "registration_filmstrip")]),
    ])

    # Structural underlay connections (conditional)
    if transform_t1_to_dsc is not None:
        workflow.connect([
            (transform_t1_to_dsc, outputnode, [("out_file", "t1w_in_dsc")]),
        ])

    if transform_flair_to_dsc is not None:
        workflow.connect([
            (transform_flair_to_dsc, outputnode, [("out_file", "flair_in_dsc")]),
        ])

    if transform_t2w_to_dsc is not None:
        workflow.connect([
            (transform_t2w_to_dsc, outputnode, [("out_file", "t2w_in_dsc")]),
        ])

    workflow.connect([
        # AIF QC plots
        (aif_qc_plot, outputnode, [
            ("out_aif_plot", "aif_curve_plot"),
            ("out_aif_location", "aif_location_plot"),
        ]),
        # Motion QC plot
        (motion_qc_plot, outputnode, [("plot_file", "motion_qc_plot")]),
        # DSC QC plots
        (dsc_preproc_qc, outputnode, [("plot_file", "dsc_preprocessing_qc_plot")]),
        (dsc_param_maps_qc, outputnode, [("plot_file", "dsc_param_maps_qc_plot")]),
        (dsc_leakage_qc, outputnode, [("plot_file", "dsc_leakage_qc_plot")]),
        # Brain mask filmstrip
        (dsc_brain_mask_qc, outputnode, [("out_file", "brain_mask_filmstrip")]),
    ])

    workflow.connect([
        (dsc_fit_qc, outputnode, [("plot_file", "dsc_fit_qc_plot")]),
    ])

    workflow.connect([
        # QC metrics
        (merge_qc, outputnode, [("merged_qc", "qc_metrics")]),
    ])

    # Smoothed outputs (only if smooth_fwhm > 0)
    if smooth_hemodynamic is not None:
        workflow.connect([
            (smooth_hemodynamic, outputnode, [("smoothed_files", "smooth_hemodynamic")]),
            (smooth_auc, outputnode, [("smoothed_files", "smooth_auc")]),
        ])

    # Store config in workflow for reference by child nodes
    workflow.config_dsc = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
