"""FSL wrappers and serializable helpers for the DSC preprocessing workflow."""
from typing import Any, Optional

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
    isdefined,
)
__all__ = [
    "ConvertXFM",
    "FLIRTApplyXFM",
    "_ConvertXFMInputSpec",
    "_ConvertXFMOutputSpec",
    "_FLIRTApplyXFMInputSpec",
    "_FLIRTApplyXFMOutputSpec",
    "_build_hemodynamic_dict",
    "_compute_mean_4d",
    "_compute_prebolus_mean",
    "_create_perfusion_brain_mask_filmstrip",
    "_create_registration_filmstrip",
    "_detect_and_correct_preload",
    "_extract_preload_class",
    "_fsl_to_x5_node",
    "_generate_dsc_fit_qc",
    "_generate_dsc_leakage_qc",
    "_generate_dsc_param_maps_qc",
    "_generate_dsc_preprocessing_qc",
    "_generate_motion_qc_plot",
    "_histogram_match_t1w_to_dsc",
    "_merge_qc_metrics",
    "_normalize_cbf",
    "_normalize_cbv",
    "_signal_to_concentration",
    "_smooth_maps",
    "_transform_brain_mask_to_dsc",
    "_transform_lesion_mask_to_dsc",
    "_transform_nawm_to_dsc",
    "_transform_structural_to_dsc",
]


# Lightweight FSL command wrappers with explicit argument ordering.
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

    def _gen_filename(self, name: str) -> Optional[str]:
        if name == "out_file":
            return self._gen_outfilename()
        return None

    def _gen_outfilename(self) -> str:
        from pathlib import Path
        in_file = Path(self.inputs.in_file)
        if self.inputs.invert_xfm:
            suffix = "_inv"
        elif self.inputs.concat_xfm:
            suffix = "_concat"
        else:
            suffix = "_out"
        return str(Path.cwd() / f"{in_file.stem}{suffix}.mat")

    def _list_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = self.output_spec().get()
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

    def _gen_filename(self, name: str) -> Optional[str]:
        if name == "out_file":
            return self._gen_outfilename()
        return None

    def _gen_outfilename(self) -> str:
        from pathlib import Path
        in_file = Path(self.inputs.in_file)
        stem = in_file.name.replace(".nii.gz", "").replace(".nii", "")
        return str(Path.cwd() / f"{stem}_transformed.nii.gz")

    def _list_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs["out_file"] = self.inputs.out_file
        else:
            outputs["out_file"] = self._gen_outfilename()
        return outputs


def _fsl_to_x5_node(mat_path: str, reference: str, moving: str) -> str:
    """Nipype Function-node wrapper for FSL .mat to ITK X5 conversion."""
    from pmriprep.utils.transforms import fsl_to_x5

    return fsl_to_x5(mat_path, reference, moving)


def _histogram_match_t1w_to_dsc(
    t1w_file: str, dsc_brain_file: str, brain_mask_file: str,
) -> str:
    """Match T1w histogram to DSC for improved FLIRT convergence.

    T1w and DSC have inverted contrasts (bright WM vs bright GM).
    Without histogram matching, FLIRT's normcorr coarse-search stage
    converges to a different local optimum.

    The T1w is brain-extracted via *brain_mask_file* before matching so
    that skull/scalp/background voxels do not distort the histogram.

    Parameters
    ----------
    t1w_file : str
        Path to whole-head bias-corrected T1w image.
    dsc_brain_file : str
        Path to DSC brain-extracted image.
    brain_mask_file : str
        Path to T1w brain mask (binary).

    Returns
    -------
    str
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


def _create_perfusion_brain_mask_filmstrip(
    mean_file: str,
    brain_mask_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
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

    from pmriprep.utils.qc_utils import plot_segmentation_qc_filmstrip, qc_figure_title

    out_dir = Path.cwd()
    output_path = str(out_dir / "brain_mask_qc_filmstrip.png")

    mean_img = nib.load(mean_file)
    mean_data = mean_img.get_fdata()  # type: ignore[attr-defined]
    mask_data = nib.load(brain_mask_file).get_fdata().astype(bool)  # type: ignore[attr-defined]
    voxel_sizes = tuple(mean_img.header.get_zooms()[:3])  # type: ignore[attr-defined]

    # Pass full binary mask — contour at level=0.5 gives solid line
    label_masks = [("Brain mask", mask_data.astype(np.float64), "#EE6677")]

    plot_segmentation_qc_filmstrip(
        background=mean_data,
        label_masks=label_masks,
        output_path=output_path,
        mask=mask_data,
        title=qc_figure_title(subject_id, session_id, "Brain Extraction QC"),
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _transform_brain_mask_to_dsc(
    brain_mask_file: str,
    reference_file: str,
    transform_matrix: str,
    hdbet_mask_file: str | None = None,
) -> tuple[str, float]:
    """Transform T1w brain mask to DSC space using registration transform.

    This mask is retained for registration QC only. Quantification continues
    to use the native DSC brain mask. The registration transform maps the
    anatomical mask onto the DSC reference grid without estimating a second
    alignment.

    Optionally computes DICE overlap with the HD-BET mask for QC.

    Parameters
    ----------
    brain_mask_file : str
        Path to brain mask in T1w space.
    reference_file : str
        Path to DSC reference image for output geometry.
    transform_matrix : str
        Path to T1w→DSC FLIRT matrix.
    hdbet_mask_file : str, optional
        Path to HD-BET DSC mask for DICE QC.

    Returns
    -------
    tuple of (str, float)
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
            mask_a = nib.load(dsc_brain_mask_file).get_fdata() > 0.5  # type: ignore[attr-defined]
            mask_b = nib.load(hdbet_mask_file).get_fdata() > 0.5  # type: ignore[attr-defined]
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
    dsc_file: str,
    te: float,
    baseline_frames: int,
) -> tuple[str, str]:
    """Convert DSC signal to ΔR2* concentration.

    This is a Function node wrapper for the signal_to_concentration utility.
    Always returns ΔR2* in s⁻¹. All downstream quantification is relative.

    Parameters
    ----------
    dsc_file : str
        Path to motion-corrected 4D DSC NIfTI.
    te : float
        Echo time in seconds.
    baseline_frames : int
        Number of pre-bolus baseline frames.

    Returns
    -------
    tuple of (str, str)
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
    concentration_file: str,
    brain_mask_file: str,
    te: float,
    field_strength: float,
    tr: float,
    preload: str | float,
    signal_file: str | None = None,
    baseline_frames: int = 5,
    leakage_method: str = "bsw",
    kep: float | None = None,
    agent_standard_dose: float = 0.1,
) -> tuple[str, str, dict[str, object]]:
    """Detect preload and apply K2 correction if needed.

    This is a Function node wrapper for preload detection and K2 correction.

    Parameters
    ----------
    concentration_file : str
        Path to 4D concentration NIfTI.
    brain_mask_file : str
        Path to brain mask NIfTI.
    te : float
        Echo time in seconds.
    field_strength : float
        Magnetic field strength in Tesla.
    tr : float
        Repetition time in seconds.
    preload : str or float
        User-specified preload — "auto" for signal-based detection,
        or a float administered dose fraction (0.0 = none, 0.5 = half).
        Auto-detection always runs; user override (if not "auto") takes
        precedence for the effective class emitted to downstream. User
        administered fraction is binned separately from auto-detection
        residual ΔR2* thresholds (different physical quantities, same
        class names). Disagreement is logged to QC when both exist.
    signal_file : str, optional
        Optional path to raw signal NIfTI for preload detection.
    baseline_frames : int, optional
        Number of pre-bolus baseline frames.
    leakage_method : str, optional
        K2 correction method ("bsw" or "bidirectional").
    kep : float, optional
        EES efflux rate in s⁻¹ (bidirectional only).
    agent_standard_dose : float, optional
        Agent's standard dose in mmol Gd/kg. Scales auto-detection
        ΔR2* thresholds linearly.

    Returns
    -------
    tuple of (str, str, dict)
        Tuple of (corrected_concentration_file, k2_map_file, qc_dict).
    """
    import logging
    import numpy as np
    import nibabel as nib  # type: ignore[import-untyped]
    from pathlib import Path

    from pmriprep.utils.dsc_preload import (
        apply_k2_correction,
        apply_k2_correction_bidirectional,
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

    qc_metrics: dict[str, object] = {}
    qc_metrics["baseline_frames_used"] = int(baseline_frames)
    k2_map_file: str = ""

    # K2-primary is the sole classifier. Run detect_preload() only for the
    # ΔR2* advisory metric (logged as tissue-heterogeneity-biased motion/
    # S0-baseline diagnostic; never drives effective_class). K2 correction
    # runs unconditionally (preload-class-agnostic), and the K2 residual
    # distribution is then used to classify
    # preload adequacy (ADEQUATE / PARTIAL / INADEQUATE / UNKNOWN).
    from pmriprep.utils.dsc_preload import (
        K2Adequacy,
        classify_preload_adequacy_from_k2,
        detect_preload,
        preload_class_from_dose,
    )

    signal_data = None
    if signal_file:
        signal_data = nib.load(signal_file).get_fdata()  # type: ignore[attr-defined]

    # --- Advisory ΔR2* metric (NOT used for classification) -------------
    preload_result = detect_preload(
        signal=signal_data,
        te=te,
        baseline_frames=baseline_frames,
        brain_mask=brain_mask,
        field_strength=field_strength,
        agent_standard_dose=agent_standard_dose,
    )
    advisory_class = preload_result.classification.value
    # Label keys explicitly "advisory_*" so consumers don't confuse this
    # with the decision-making classification.
    qc_metrics["preload_baseline_delta_r2_advisory"] = (
        preload_result.baseline_delta_r2
    )
    qc_metrics["preload_advisory_classification"] = advisory_class
    qc_metrics["preload_advisory_method"] = preload_result.detection_method
    qc_metrics["preload_advisory_note"] = (
        "delta-R2* advisory: tissue-heterogeneity-biased; used for acquisition "
        "QC (motion, S0 baseline) only. Not used to classify preload."
    )

    auto_details = preload_result.details or {}
    effective_thresholds = auto_details.get("effective_thresholds")
    if effective_thresholds is not None:
        qc_metrics["preload_thresholds_effective"] = {
            **effective_thresholds,
            "agent_standard_dose": agent_standard_dose,
        }

    residual_equivalent = auto_details.get("residual_equivalent_fraction")
    if residual_equivalent is not None:
        qc_metrics["preload_residual_equivalent_fraction"] = residual_equivalent

    # --- K2 correction (preload-agnostic, always runs) ------------------
    qc_metrics["preload_correction_applied"] = True
    if leakage_method == "bidirectional":
        correction_result = apply_k2_correction_bidirectional(
            concentration=conc_data,
            brain_mask=brain_mask,
            tr=tr,
            kep=kep,
            baseline_frames=baseline_frames,
        )
    else:
        correction_result = apply_k2_correction(
            concentration=conc_data,
            brain_mask=brain_mask,
            tr=tr,
            baseline_frames=baseline_frames,
        )

    corrected_conc = correction_result.concentration_corrected
    k2_map = correction_result.k2_map

    # Update QC metrics with K2 correction stats (k2_median,
    # percent_leaky_voxels, etc.). Required for the K2-primary classifier.
    if correction_result.qc_metrics:
        qc_metrics.update(correction_result.qc_metrics)

    # --- Preload classification: user override > K2-primary -------------
    if isinstance(preload, str) and preload == "auto":
        k2_class, k2_confidence, k2_reason = (
            classify_preload_adequacy_from_k2(qc_metrics)
        )
        effective_class = k2_class.value
        qc_metrics["preload_user_specified"] = False
        qc_metrics["preload_k2_classification"] = effective_class
        qc_metrics["preload_k2_confidence"] = k2_confidence
        qc_metrics["preload_k2_reason"] = k2_reason
        if k2_class == K2Adequacy.UNKNOWN:
            qc_metrics["preload_classification_source"] = "k2_unavailable"
            logging.getLogger("pmriprep.workflows").warning(
                "K2-primary preload classification unavailable (%s); "
                "downstream DCE constraint will SKIP.",
                k2_reason,
            )
        else:
            qc_metrics["preload_classification_source"] = "k2_primary"
    else:
        user_dose = float(preload)
        user_class = preload_class_from_dose(user_dose).value
        effective_class = user_class
        qc_metrics["preload_user_specified"] = True
        qc_metrics["preload_user_specified_dose"] = user_dose
        qc_metrics["preload_user_specified_classification"] = user_class
        qc_metrics["preload_classification_source"] = "user_specified"
        # Log K2-primary classification for reference (even with user override).
        k2_class, k2_confidence, _ = classify_preload_adequacy_from_k2(
            qc_metrics
        )
        qc_metrics["preload_k2_classification"] = k2_class.value
        qc_metrics["preload_k2_confidence"] = k2_confidence

    qc_metrics["preload_classification"] = effective_class

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


def _augment_qc_positive_tail(
    qc_dict: dict[str, object] | None,
    concentration_file: str,
    k2_map_file: str,
    lesion_mask_dsc: str | None = None,
) -> dict[str, object]:
    """Add the positive-tail (BSW-blind overestimate) fraction to the preload QC.

    A T2*-dominant persistent positive ΔR2* tail makes K2 pin at 0 (the
    non-negative BSW gate) while the full-integral CBV over-estimates. This
    flags the tumor-voxel fraction where the late-window tail persists AND K2
    is zero. Runs as a separate node (after preload correction AND the tumor-
    mask transform) so the expensive deconvolution — fed directly by the
    correction node — is not serialized behind tumor segmentation.

    Parameters
    ----------
    qc_dict : dict
        Preload/K2 QC dict from ``_detect_and_correct_preload``.
    concentration_file : str
        Path to the (uncorrected) 4D concentration NIfTI — the tail is measured
        on the observed curve, before add-back.
    k2_map_file : str
        Path to the K2 map NIfTI (gated to 0.0 where BSW applied no correction).
    lesion_mask_dsc : str, optional
        Whole-tumor mask in DSC space (empty/None → fraction 0.0).

    Returns
    -------
    dict
        ``qc_dict`` with ``cbv_positive_tail_fraction`` added.
    """
    import logging
    import numpy as np
    import nibabel as nib  # type: ignore[import-untyped]

    from pmriprep.utils.dsc_constants import TAIL_LATE_FRAC, TAIL_RATIO_TAU

    qc_dict = dict(qc_dict) if qc_dict else {}
    positive_tail_fraction = 0.0
    if lesion_mask_dsc and k2_map_file:
        try:
            conc = nib.load(concentration_file).get_fdata()  # type: ignore[attr-defined]
            k2_map = nib.load(k2_map_file).get_fdata()  # type: ignore[attr-defined]
            tumor = nib.load(lesion_mask_dsc).get_fdata() > 0.5  # type: ignore[attr-defined]
            if (
                tumor.shape[:3] == conc.shape[:3]
                and tumor.shape[:3] == k2_map.shape[:3]
                and int(tumor.sum()) >= 10
            ):
                t_curves = conc[tumor]  # (nvox, T)
                k2_vox = k2_map[tumor]
                n_late = max(1, int(round(TAIL_LATE_FRAC * t_curves.shape[-1])))
                peak = np.nanmax(t_curves, axis=-1)
                late = np.nanmean(t_curves[:, -n_late:], axis=-1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(peak > 1e-6, late / peak, 0.0)
                # k2_map is gated to exactly 0.0 by apply_k2_correction; a zero
                # here is precisely a BSW-uncorrected (blind) voxel.
                positive_tail_fraction = float(
                    np.mean((ratio > TAIL_RATIO_TAU) & (k2_vox == 0.0))
                )
        except Exception as exc:  # noqa: BLE001 — QC metric must never break the node
            logging.getLogger("pmriprep.workflows").warning(
                "Positive-tail fraction computation failed (%s); defaulting to 0.0.",
                exc,
            )
            positive_tail_fraction = 0.0

    qc_dict["cbv_positive_tail_fraction"] = positive_tail_fraction
    return qc_dict


def _compute_prebolus_mean(moco_file: str, baseline_frames: int) -> str:
    """Compute mean of pre-bolus frames as a clean reference volume.

    The standard mean image includes bolus passage frames, which are
    contaminated by contrast-induced signal drop. Using only pre-bolus
    frames provides a cleaner reference for brain extraction and
    registration.

    Parameters
    ----------
    moco_file : str
        Path to motion-corrected 4D DSC NIfTI.
    baseline_frames : int
        Number of pre-bolus frames to average.

    Returns
    -------
    str
        Path to pre-bolus mean NIfTI file.
    """
    import numpy as np
    import nibabel as nib
    from pathlib import Path

    img = nib.load(moco_file)
    data = img.get_fdata()  # type: ignore[attr-defined]

    # Average only pre-bolus frames
    n_frames = min(baseline_frames, data.shape[-1])
    prebolus_mean = np.mean(data[..., :n_frames], axis=-1)

    out_dir = Path.cwd()
    out_file = str(out_dir / "dsc_prebolus_mean.nii.gz")
    mean_img = nib.Nifti1Image(
        prebolus_mean.astype(np.float32), img.affine, img.header  # type: ignore[attr-defined]
    )
    nib.save(mean_img, out_file)

    return out_file


def _compute_mean_4d(in_file: str) -> str:
    """Compute the temporal mean of a 4D series as a reference volume.

    Full-volume equivalent of the mean that ``MotionCorrection`` emits, for the
    ``motion_correct=False`` path where the reoriented (uncorrected) series is
    processed and no realigned mean is available.

    Parameters
    ----------
    in_file : str
        Path to a 4D (or 3D) DSC NIfTI.

    Returns
    -------
    str
        Path to the mean NIfTI file.
    """
    import numpy as np
    import nibabel as nib
    from pathlib import Path

    img = nib.load(in_file)
    data = img.get_fdata()  # type: ignore[attr-defined]
    mean_data = np.mean(data, axis=-1) if data.ndim == 4 else data

    out_file = str(Path.cwd() / "dsc_mean.nii.gz")
    mean_img = nib.Nifti1Image(
        mean_data.astype(np.float32), img.affine, img.header  # type: ignore[attr-defined]
    )
    nib.save(mean_img, out_file)

    return out_file


def _transform_nawm_to_dsc(
    nawm_mask_file: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
    needs_transform: bool = True,
) -> str:
    """Transform NAWM mask to DSC space, or pass through if already native.

    When ``needs_transform`` is True (tumor-seg NAWM in T1w space), applies
    FSL FLIRT with nearest-neighbour interpolation.  When False (user-supplied
    mask already in DSC native space), validates geometry and returns the
    input path unchanged.

    Parameters
    ----------
    nawm_mask_file : str or None
        Path to NAWM mask (or None).
    reference_file : str or None
        Path to DSC reference image for output geometry.
    transform_matrix : str or None
        Path to T1w→DSC FLIRT matrix (only used when needs_transform=True).
    needs_transform : bool
        If False, skip FLIRT and return input path as-is.

    Returns
    -------
    str
        Path to NAWM mask in DSC space, or "" if no mask.
    """
    from nipype.interfaces.base import isdefined

    if not isdefined(nawm_mask_file) or nawm_mask_file is None:
        return ""

    from pathlib import Path

    nawm_path = Path(nawm_mask_file)
    if not nawm_path.is_file():
        return ""

    if not needs_transform:
        import nibabel as nib
        import numpy as np

        mask_img = nib.load(str(nawm_mask_file))
        ref_img = nib.load(str(reference_file))
        if mask_img.shape[:3] != ref_img.shape[:3]:  # type: ignore[attr-defined]
            raise ValueError(
                f"User-supplied NAWM mask shape {mask_img.shape[:3]} does not "  # type: ignore[attr-defined]
                f"match DSC volume shape {ref_img.shape[:3]}. "  # type: ignore[attr-defined]
                f"Masks under [dsc] config must be in DSC native space."
            )
        affine_diff = float(
            np.abs(mask_img.affine - ref_img.affine).max()  # type: ignore[attr-defined]
        )
        if affine_diff > 1.0:
            raise ValueError(
                f"User-supplied NAWM mask affine differs from DSC volume by "
                f"{affine_diff:.2f} (max element). "
                f"Masks under [dsc] config must be in DSC native space."
            )
        return str(nawm_mask_file)

    import subprocess

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


def _transform_wm_probseg_to_dsc(
    wm_probseg: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
) -> str:
    """Threshold + erode the FAST WM probability map into DSC space.

    Resamples the T1w-space white-matter probability map into DSC space with
    FLIRT (trilinear, since it is a probability map, not a binary mask),
    thresholds at 0.5, and erodes one voxel in DSC space to drop GM/CSF
    partial-volume edges. The eroded binary mask is the first-pass-CBV
    normalization reference. Returns ``""`` when no probability map is
    supplied, in which case the first-pass map is not produced.

    Parameters
    ----------
    wm_probseg : str or None
        Path to the FAST white-matter probability map in T1w space (or None).
    reference_file : str or None
        Path to the DSC reference image for output geometry.
    transform_matrix : str or None
        Path to the T1w→DSC FLIRT matrix.

    Returns
    -------
    str
        Path to the eroded binary WM mask in DSC space, or ``""`` if no map.
    """
    from nipype.interfaces.base import isdefined

    if (
        not isdefined(wm_probseg)
        or wm_probseg is None
        or not isdefined(reference_file)
        or reference_file is None
        or not isdefined(transform_matrix)
        or transform_matrix is None
    ):
        return ""

    from pathlib import Path

    if not Path(str(wm_probseg)).is_file():
        return ""

    import logging
    import subprocess

    import nibabel as nib
    import numpy as np
    from scipy import ndimage

    out_dir = Path.cwd()
    resampled = str(out_dir / "wm_probseg_dsc.nii.gz")
    cmd = [
        "flirt",
        "-in", str(wm_probseg),
        "-ref", str(reference_file),
        "-out", resampled,
        "-init", str(transform_matrix),
        "-applyxfm",
        "-interp", "trilinear",
    ]
    # The first-pass map is an opt-in additive output, so a resample failure
    # must not abort the DSC run: fail soft to "" (map simply not produced).
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        img = nib.load(resampled)
        prob = img.get_fdata()  # type: ignore[attr-defined]
        mask = ndimage.binary_erosion(prob > 0.5, iterations=1)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("nipype.interface").warning(
            "First-pass CBV WM reference resample failed (%s); "
            "map not produced.", exc,
        )
        return ""

    out_file = str(out_dir / "wm_mask_dsc.nii.gz")
    out_img = nib.Nifti1Image(
        mask.astype(np.uint8), img.affine, img.header,  # type: ignore[attr-defined]
    )
    out_img.header.set_data_dtype(np.uint8)
    nib.save(out_img, out_file)
    return out_file


def _transform_lesion_mask_to_dsc(
    lesion_mask_file: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
) -> str:
    """Transform lesion mask from T1w space to DSC space.

    Uses FSL FLIRT with nearest-neighbour interpolation to resample
    the lesion mask into DSC space, preserving integer labels.

    Parameters
    ----------
    lesion_mask_file : str or None
        Path to lesion mask in T1w space (or None).
    reference_file : str or None
        Path to DSC reference image for output geometry.
    transform_matrix : str or None
        Path to T1w→DSC FLIRT matrix.

    Returns
    -------
    str
        Path to transformed lesion mask in DSC space, or "" if no mask.
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
    structural_file: str | None = None,
    reference_file: str | None = None,
    transform_matrix: str | None = None,
    output_name: str = "structural_in_dsc.nii.gz",
) -> str:
    """Transform structural image from T1w space to DSC space via FLIRT.

    Handles None/undefined inputs gracefully by returning None.

    Parameters
    ----------
    structural_file : str or None
        Path to structural image in T1w space (or None).
    reference_file : str or None
        Path to DSC reference image for output geometry.
    transform_matrix : str or None
        Path to T1w→DSC FLIRT matrix.
    output_name : str
        Output filename.

    Returns
    -------
    str
        Path to transformed structural image in DSC space, or '' if no input.
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

    structural_path = Path(structural_file)
    if not structural_path.is_file():
        return ""

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
    cbv_file: str,
    derivatives_dir: str,
    subject_id: str,
    session_id: str,
    nawm_mask_file: str | None = None,
) -> tuple[str, float | None]:
    """Normalize CBV to NAWM median.

    This is a Function node wrapper for CBV normalization.
    Returns ("", None) when NAWM mask is not available - normalization
    requires NAWM reference for clinically valid results.

    Parameters
    ----------
    cbv_file : str
        Path to CBV map NIfTI.
    derivatives_dir : str
        Path to derivatives directory for auto-detection.
    subject_id : str
        Subject ID for auto-detection.
    session_id : str
        Session ID for auto-detection.
    nawm_mask_file : str, optional
        Path to NAWM mask NIfTI (required for normalization).

    Returns
    -------
    tuple of (str, float or None)
        Tuple of (ncbv_file, normalization_factor). The first element is the
        empty-string sentinel ``""`` (not ``None``) when no NAWM mask is
        available, so the downstream ``traits.Str`` consumers
        (``lesion_stats.ncbv_map`` / ``roi_stats.ncbv_map``) accept it
        without TraitError. The second element is ``None`` because it feeds
        a Function-node consumer (``_merge_qc_metrics``) that embeds the
        value into a dict.
    """
    from pathlib import Path

    # Skip normalization if no NAWM mask available
    if nawm_mask_file is None:
        return "", None

    nawm_path = Path(nawm_mask_file)
    if not nawm_path.is_file():
        return "", None

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
        brain_mask_file=None,
    )

    return str(output_path), norm_factor


def _normalize_cbf(
    cbf_file: str,
    derivatives_dir: str,
    subject_id: str,
    session_id: str,
    nawm_mask_file: str | None = None,
) -> tuple[str, float | None]:
    """Normalize CBF to NAWM median (same method as CBV normalization).

    Returns ("", None) when NAWM mask is not available - normalization
    requires NAWM reference for clinically valid results.

    Parameters
    ----------
    cbf_file : str
        Path to CBF map NIfTI.
    derivatives_dir : str
        Path to derivatives directory for auto-detection.
    subject_id : str
        Subject ID for auto-detection.
    session_id : str
        Session ID for auto-detection.
    nawm_mask_file : str, optional
        Path to NAWM mask NIfTI (required for normalization).

    Returns
    -------
    tuple of (str, float or None)
        Tuple of (ncbf_file, normalization_factor). First element is the
        ``""`` empty-string sentinel (not ``None``) on the no-NAWM path —
        same rationale as :func:`_normalize_cbv`.
    """
    from pathlib import Path

    # Skip normalization if no NAWM mask available
    if nawm_mask_file is None:
        return "", None

    nawm_path = Path(nawm_mask_file)
    if not nawm_path.is_file():
        return "", None

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
        brain_mask_file=None,
    )

    return str(output_path), norm_factor


def _create_registration_filmstrip(
    dsc_mean_file: str,
    registered_t1w_file: str,
    brain_mask_file: str,
    registered_t1w_mask_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Create a registration QC filmstrip PNG.

    Generates a multi-slice overlay showing the registered T1w on the DSC mean
    with brain mask contours for visual QC of registration quality.

    Parameters
    ----------
    dsc_mean_file : str
        Path to mean DSC image (reference).
    registered_t1w_file : str
        Path to registered T1w in DSC space.
    brain_mask_file : str
        Path to DSC brain mask.
    registered_t1w_mask_file : str
        Path to the anatomical brain mask transformed into DSC space.
    subject_id : str, optional
        Subject ID for QC figure title.
    session_id : str, optional
        Session ID for QC figure title.

    Returns
    -------
    str
        Path to the registration filmstrip PNG file.
    """
    import nibabel as nib
    from pathlib import Path
    from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip, qc_figure_title

    out_dir = Path.cwd()
    output_path = str(out_dir / "registration_qc_filmstrip.png")

    dsc_img = nib.load(dsc_mean_file)
    dsc_data = dsc_img.get_fdata()  # type: ignore[attr-defined]
    t1w_data = nib.load(registered_t1w_file).get_fdata()  # type: ignore[attr-defined]
    mask_data = nib.load(brain_mask_file).get_fdata()  # type: ignore[attr-defined]
    moving_mask_data = nib.load(registered_t1w_mask_file).get_fdata()  # type: ignore[attr-defined]
    voxel_sizes = tuple(dsc_img.header.get_zooms()[:3])  # type: ignore[attr-defined]

    plot_registration_qc_filmstrip(
        fixed=dsc_data,
        moving=t1w_data,
        output_path=output_path,
        mask=mask_data,
        moving_mask=moving_mask_data,
        title=qc_figure_title(subject_id, session_id, "DSC-T1w Registration"),
        voxel_sizes=voxel_sizes,
    )

    return output_path


def _extract_preload_class(qc_dict: dict[str, object] | None) -> str:
    """Extract preload classification from preload_correct QC dict.

    Parameters
    ----------
    qc_dict : dict
        QC metrics dictionary from preload correction node.

    Returns
    -------
    str
        Effective classification string: a K2-adequacy value on the auto
        path (e.g. 'adequate', 'partial', 'inadequate', 'unknown') or a
        preload-dose class on the user path (e.g. 'none', 'half_dose').
    """
    if not qc_dict:
        return "none"
    return str(qc_dict.get("preload_classification", "none"))


def _generate_motion_qc_plot(
    motion_params_file: str,
    tr: float,
    subject_id: str = "",
    session_id: str = "",
    motion_applied: bool = True,
) -> tuple[str, dict[str, object]]:
    """Generate motion QC plot from motion parameters.

    Creates a three-panel figure showing raw motion parameters and
    framewise displacement over time.

    Parameters
    ----------
    motion_params_file : str
        Path to motion parameters text file.
    tr : float
        Repetition time in seconds.
    subject_id : str, optional
        Subject ID for QC figure title.
    session_id : str, optional
        Session ID for QC figure title.
    motion_applied : bool, optional
        Whether motion correction was applied. When False, the plotted values
        are residual motion (mcflirt was run for measurement only), and the
        figure/summary are annotated accordingly.

    Returns
    -------
    tuple of (str, dict)
        Tuple of (plot_file path, motion_summary dict). ``motion_summary``
        carries ``motion_applied`` and, when False, a ``motion_note``.
    """
    import numpy as np
    from pathlib import Path
    from pmriprep.utils.qc_utils import plot_motion_qc

    note = None
    if not motion_applied:
        note = (
            "Motion correction NOT applied (input assumed pre-corrected); "
            "plotted values are residual motion, near the registration noise floor."
        )

    motion_params = np.loadtxt(motion_params_file)
    out_path = Path.cwd() / "dsc_motion_qc.png"
    _, motion_summary = plot_motion_qc(
        motion_params, out_path, tr=tr,
        subject_id=subject_id, session_id=session_id, note=note,
    )
    motion_summary["motion_applied"] = bool(motion_applied)
    if note:
        motion_summary["motion_note"] = note
    return str(out_path), motion_summary


def _generate_dsc_preprocessing_qc(
    dsc_signal_file: str,
    brain_mask_file: str,
    baseline_frames: int,
    tr: float,
    te: float | None = None,
    subject_id: str = "",
    session_id: str = "",
) -> tuple[str, dict[str, object]]:
    """Generate DSC preprocessing QC plot.

    Parameters
    ----------
    dsc_signal_file : str
        Path to motion-corrected 4D DSC signal NIfTI.
    brain_mask_file : str
        Path to brain mask NIfTI.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    tr : float
        Repetition time in seconds.
    te : float, optional
        Echo time in seconds.
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.

    Returns
    -------
    tuple of (str, dict)
        Tuple of (plot_file path, stats dict).
    """
    import nibabel as nib
    from pmriprep.utils.dsc_qc_plots import generate_dsc_preprocessing_qc_plot

    signal_img = nib.load(dsc_signal_file)
    signal_4d = signal_img.get_fdata()  # type: ignore[attr-defined]
    mask_data = nib.load(brain_mask_file).get_fdata()  # type: ignore[attr-defined]

    _zooms = signal_img.header.get_zooms()  # type: ignore[attr-defined]
    voxel_size_mm = (
        float(_zooms[0]), float(_zooms[1]), float(_zooms[2]),
    )

    plot_path, stats = generate_dsc_preprocessing_qc_plot(
        dsc_signal_4d=signal_4d,
        brain_mask=mask_data,
        baseline_frames=baseline_frames,
        volume_tr_s=tr,
        te_s=te,
        subject_id=subject_id,
        session_id=session_id,
        voxel_size_mm=voxel_size_mm,
    )
    return plot_path, stats


def _generate_dsc_param_maps_qc(
    rel_cbf_file: str,
    rel_cbv_file: str,
    mtt_file: str,
    tmax_file: str,
    psr_file: str,
    ttp_file: str,
    brain_mask_file: str,
    subject_id: str = "",
    session_id: str = "",
) -> str:
    """Generate DSC parameter maps QC plot.

    Parameters
    ----------
    rel_cbf_file : str
        Path to relative CBF map NIfTI.
    rel_cbv_file : str
        Path to relative CBV map NIfTI.
    mtt_file : str
        Path to MTT map NIfTI.
    tmax_file : str
        Path to Tmax map NIfTI.
    psr_file : str
        Path to PSR map NIfTI.
    ttp_file : str
        Path to TTP map NIfTI.
    brain_mask_file : str
        Path to brain mask NIfTI.
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.

    Returns
    -------
    str
        Path to the generated QC plot PNG.
    """
    import nibabel as nib
    from pmriprep.utils.dsc_qc_plots import generate_dsc_parameter_maps_qc_plot

    mask_data = nib.load(brain_mask_file).get_fdata()  # type: ignore[attr-defined]

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
            if _P(filepath).is_file():
                param_maps[name] = nib.load(filepath).get_fdata()  # type: ignore[attr-defined]

    plot_path = generate_dsc_parameter_maps_qc_plot(
        param_maps=param_maps,
        brain_mask=mask_data,
        subject_id=subject_id,
        session_id=session_id,
    )
    return plot_path


def _generate_dsc_leakage_qc(
    k2_map_file: str,
    auc_raw_file: str,
    auc_corr_file: str,
    auc_diff_file: str,
    brain_mask_file: str,
    concentration_file: str | None = None,
    corrected_concentration_file: str | None = None,
    psr_file: str | None = None,
    tr: float = 1.5,
    subject_id: str = "",
    session_id: str = "",
    preload_qc: dict[str, object] | None = None,
) -> str | None:
    """Generate DSC leakage correction QC plot.

    Parameters
    ----------
    k2_map_file : str
        Path to K2 map NIfTI.
    auc_raw_file : str
        Path to AUC raw map NIfTI.
    auc_corr_file : str
        Path to AUC corrected map NIfTI.
    auc_diff_file : str
        Path to AUC diff map NIfTI.
    brain_mask_file : str
        Path to brain mask NIfTI.
    concentration_file : str, optional
        Path to pre-K2 4D concentration NIfTI.
    corrected_concentration_file : str, optional
        Path to post-K2 4D concentration NIfTI.
    psr_file : str, optional
        Path to PSR map NIfTI.
    tr : float, optional
        Repetition time in seconds.
    subject_id : str, optional
        Subject identifier.
    session_id : str, optional
        Session identifier.
    preload_qc : dict, optional
        Preload/K2 QC metrics dict.

    Returns
    -------
    str
        Path to the generated QC plot PNG, or ``None`` if K2 is unavailable.
    """
    from pathlib import Path as _P

    # Skip if no K2 map (guard against "." or directory paths from unset outputs)
    if not k2_map_file or not _P(k2_map_file).is_file():
        return None

    import nibabel as nib
    from pmriprep.utils.dsc_qc_plots import generate_dsc_leakage_qc_plot

    if not auc_raw_file or not auc_corr_file or not auc_diff_file:
        return None
    k2_data = nib.load(k2_map_file).get_fdata()  # type: ignore[attr-defined]
    auc_raw_data = nib.load(auc_raw_file).get_fdata()  # type: ignore[attr-defined]
    auc_corr_data = nib.load(auc_corr_file).get_fdata()  # type: ignore[attr-defined]
    auc_diff_data = nib.load(auc_diff_file).get_fdata()  # type: ignore[attr-defined]
    mask_data = nib.load(brain_mask_file).get_fdata()  # type: ignore[attr-defined]

    conc_raw_4d = None
    if concentration_file and _P(concentration_file).is_file():
        conc_raw_4d = nib.load(concentration_file).get_fdata()  # type: ignore[attr-defined]

    conc_corr_4d = None
    if corrected_concentration_file and _P(corrected_concentration_file).is_file():
        conc_corr_4d = nib.load(corrected_concentration_file).get_fdata()  # type: ignore[attr-defined]

    psr_data = None
    if psr_file and _P(psr_file).is_file():
        psr_data = nib.load(psr_file).get_fdata()  # type: ignore[attr-defined]

    # Extract method and kep_used from preload_qc dict
    method = "bsw"
    kep_used: float | None = None
    if preload_qc:
        method_value = preload_qc.get("k2_method", "bsw")
        if isinstance(method_value, str):
            method = method_value
        kep_value = preload_qc.get("kep_used")
        if isinstance(kep_value, (int, float)):
            kep_used = float(kep_value)

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
        method=method,
        kep_used=kep_used,
    )
    return plot_path


def _generate_dsc_fit_qc(
    qc_fit_data_file: str | None = None,
    brain_mask_file: str | None = None,
    cbf_file: str | None = None,
    subject_id: str = "",
    session_id: str = "",
) -> str | None:
    """Generate DSC fit QC plot from saved QC fit data JSON.

    Parameters
    ----------
    qc_fit_data_file : str or None
        Path to QC fit data JSON from ComputeDSCMaps.
    brain_mask_file : str or None
        Path to brain mask NIfTI.
    cbf_file : str or None
        Path to relative CBF map NIfTI.
    subject_id : str
        Subject identifier.
    session_id : str
        Session identifier.

    Returns
    -------
    str
        Path to the generated QC plot PNG, or ``None`` if unavailable.
    """
    import json
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    required_files = (qc_fit_data_file, brain_mask_file, cbf_file)
    if any(not path or not Path(path).is_file() for path in required_files):
        return None

    assert qc_fit_data_file is not None
    assert brain_mask_file is not None
    assert cbf_file is not None
    with open(qc_fit_data_file) as f:
        fit_data = json.load(f)

    # dtype=float coerces JSON null (a sanitized non-finite element) back to nan
    # so the curves stay float arrays and plot as gaps rather than object arrays.
    tissue_conc = np.array(fit_data["tissue_concentration"], dtype=float)
    model_pred = np.array(fit_data["model_prediction"], dtype=float)
    aif_curve = np.array(fit_data["aif"], dtype=float)
    time_points = np.array(fit_data["time_points"], dtype=float)
    voxel = tuple(fit_data["voxel"])
    voxel_cbf = fit_data.get("voxel_cbf", float("nan"))
    r_squared = fit_data.get("r_squared", 0.0)
    method = fit_data.get("method", "")
    raw_signal_data = fit_data.get("raw_signal")
    raw_signal = (
        np.array(raw_signal_data, dtype=float) if raw_signal_data is not None else None
    )

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


def _smooth_maps(
    map_files: dict[str, str],
    brain_mask_file: str,
    fwhm: float,
) -> dict[str, str]:
    """Apply Gaussian smoothing to a dict of maps.

    Uses normalized convolution for maps containing NaN to prevent edge
    dilution where unfitted voxels would otherwise contribute 0 to the
    Gaussian kernel.

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

    logger = logging.getLogger("nipype.workflow")

    sigma = fwhm / 2.355
    results: dict[str, str] = {}
    for name, filepath in map_files.items():
        if not filepath or filepath in ("", "None"):
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
                # Normalize; -nan cleans 0/0→NaN; threshold (DSC maps are
                # non-negative: CBF, CBV, MTT, nCBF, nCBV, AUC)
                subprocess.run(
                    ["fslmaths", num_path, "-div", den_path,
                     "-nan", "-thr", "0", "-mas", brain_mask_file, out_path],
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


def _merge_qc_metrics(
    motion_qc: dict[str, object] | None,
    preload_qc: dict[str, object] | None,
    aif_qc: dict[str, object] | None,
    deconv_qc: dict[str, object] | None,
    registration_qc: dict[str, object] | None,
    normalization_factor: float | None,
    motion_summary: dict[str, object] | None = None,
    aif_metrics: dict[str, object] | None = None,
    preprocessing_stats: dict[str, object] | None = None,
    aif_source: str = "auto",
) -> dict[str, object]:
    """Merge all QC metrics into a single dictionary.

    Parameters
    ----------
    motion_qc : dict
        Motion correction QC metrics.
    preload_qc : dict
        Preload detection and correction QC metrics.
    aif_qc : dict
        AIF detection QC metrics.
    deconv_qc : dict
        Deconvolution QC metrics.
    registration_qc : dict
        Registration QC metrics.
    normalization_factor : float
        CBV normalization factor.
    motion_summary : dict, optional
        Motion summary dict from plot_motion_qc.
    aif_metrics : dict, optional
        AIF metrics dict from AIFQCPlot.
    preprocessing_stats : dict, optional
        Preprocessing stats dict.
    aif_source : str, optional
        AIF source type ("auto", "synthetic", or "external").

    Returns
    -------
    dict
        Combined QC metrics dictionary.
    """
    motion_data = motion_qc if motion_qc else {}
    if motion_summary:
        motion_data = dict(motion_data) if motion_data else {}
        motion_data["motion_summary"] = motion_summary
        # Surface a flat motion_applied alongside the (strippable) motion_summary
        # so it survives filter_pipeline_qc into the sidecar PipelineQC — the
        # report boilerplate reads it from there. Mirrors DCE's flat placement.
        motion_data["motion_applied"] = bool(
            motion_summary.get("motion_applied", True)
        )

    preprocessing = dict(preprocessing_stats) if preprocessing_stats else {}
    if motion_data:
        preprocessing.update(motion_data)

    aif_data = aif_qc if aif_qc else {}
    if aif_metrics:
        aif_data = dict(aif_data) if aif_data else {}
        aif_data["aif_metrics"] = aif_metrics

    merged: dict[str, object] = {
        "preprocessing": preprocessing,
        "preload": preload_qc if preload_qc else {},
        "aif_detection": aif_data,
        "deconvolution": deconv_qc if deconv_qc else {},
        "registration": registration_qc if registration_qc else {},
        "normalization_factor": normalization_factor,
        "config": {
            "aif_source": aif_source,
        },
    }
    return merged


def _build_hemodynamic_dict(
    mtt: str,
    rel_cbf: str,
    rel_cbv: str,
    ncbf: str,
    ncbv: str,
) -> dict[str, str]:
    from pathlib import Path as _P
    d = {"mtt": mtt, "rel_cbf": rel_cbf, "rel_cbv": rel_cbv}
    for name, val in [
        ("ncbf", ncbf), ("ncbv", ncbv),
    ]:
        if isinstance(val, str) and val not in ("", "None") and _P(val).is_file():
            d[name] = val
    return d
