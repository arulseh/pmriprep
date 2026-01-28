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

    # Check if brain mask dimensions match concentration data
    # If not (e.g., mask is in T1w space, data is in DSC space), create mask from data
    if brain_mask.shape[:3] != conc_data.shape[:3]:
        import warnings
        warnings.warn(
            f"Brain mask shape {brain_mask.shape[:3]} does not match "
            f"concentration data shape {conc_data.shape[:3]}. "
            "Creating mask from DSC data intensity."
        )
        # Create simple mask from mean DSC intensity
        mean_intensity = np.mean(np.abs(conc_data), axis=3)
        threshold = np.percentile(mean_intensity[mean_intensity > 0], 10)
        brain_mask = mean_intensity > threshold

    # Load AIF
    aif = np.loadtxt(aif_file)

    qc_metrics: dict = {}
    k2_map_file: Optional[str] = None

    # Detect preload if not explicitly specified
    needs_correction = False
    if preload_detected is True:
        needs_correction = True
        qc_metrics["preload_user_specified"] = True
    elif preload_detected is False:
        needs_correction = False
        qc_metrics["preload_user_specified"] = True
    else:
        # Auto-detect based on baseline concentration characteristics
        qc_metrics["preload_user_specified"] = False
        # Simplified auto-detection based on baseline concentration
        baseline_mean = float(np.mean(conc_data[brain_mask][..., :5]))
        if baseline_mean > 0.5:  # Elevated baseline suggests preload
            needs_correction = True
        qc_metrics["preload_auto_detected"] = needs_correction
        qc_metrics["baseline_concentration_mean"] = baseline_mean

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
    outputnode.cbf_map : str
        Path to cerebral blood flow map.
    outputnode.cbv_map : str
        Path to cerebral blood volume map.
    outputnode.mtt_map : str
        Path to mean transit time map.
    outputnode.tmax_map : str
        Path to time-to-maximum map.
    outputnode.psr_map : str
        Path to percent signal recovery map.
    outputnode.ph_map : str
        Path to peak height map.
    outputnode.aif_curve : str
        Path to detected/used AIF curve.
    outputnode.aif_mask : str
        Path to AIF detection mask.
    outputnode.k2_map : str, optional
        Path to K2 leakage correction map (if save_k2_map=True).
    outputnode.ncbv_map : str, optional
        Path to normalized CBV map (if NAWM mask available).
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
                # Perfusion parameter maps
                "cbf_map",
                "cbv_map",
                "mtt_map",
                "tmax_map",
                "psr_map",
                "ph_map",
                # AIF outputs
                "aif_curve",
                "aif_mask",
                "aif_qc",
                # Leakage correction
                "k2_map",
                # Normalized maps
                "ncbv_map",
                # Registration
                "dsc_to_t1w_xfm",
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
            ],
            output_names=["corrected_concentration_file", "k2_map_file", "qc_dict"],
            function=_detect_and_correct_preload,
        ),
        name="preload_correct",
    )
    preload_correct.inputs.preload_detected = config.preload_detected if config.preload_detected else None
    preload_correct.inputs.save_k2_map = config.save_k2_map

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
    # Workflow Connections
    # -------------------------------------------------------------------

    # Input to Motion Correction
    workflow.connect([
        (inputnode, motion_correct, [("dsc_file", "in_file")]),
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
    workflow.connect([
        (signal_to_conc, aif_detect, [("concentration_file", "concentration_file")]),
        (inputnode, aif_detect, [
            ("brain_mask", "brain_mask_file"),
            ("lesion_mask", "lesion_mask_file"),
        ]),
    ])

    # AIF Detection and Signal Conversion to Preload Correction
    workflow.connect([
        (signal_to_conc, preload_correct, [("concentration_file", "concentration_file")]),
        (aif_detect, preload_correct, [("out_aif_curve", "aif_file")]),
        (inputnode, preload_correct, [
            ("brain_mask", "brain_mask_file"),
            ("te", "te"),
            ("field_strength", "field_strength"),
            ("tr", "tr"),
        ]),
    ])

    # Preload Correction to DSC Map Computation
    workflow.connect([
        (preload_correct, compute_maps, [
            ("corrected_concentration_file", "concentration_file"),
        ]),
        (aif_detect, compute_maps, [("out_aif_curve", "aif_file")]),
        (inputnode, compute_maps, [
            ("brain_mask", "brain_mask_file"),
            ("tr", "tr"),
        ]),
        (signal_to_conc, compute_maps, [("signal_file", "signal_file")]),
    ])

    # DSC Map Computation to Registration
    workflow.connect([
        (inputnode, register, [
            ("t1w_reference", "t1w_file"),
            ("lesion_mask", "lesion_mask"),
            ("brain_mask", "brain_mask"),
            ("wm_probseg", "wm_seg"),
        ]),
        (motion_correct, register, [("mean_img", "dsc_mean")]),
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
    # Use motion-corrected mean DSC as reference for output geometry
    workflow.connect([
        (concat_mni_to_dsc, transform_atlas, [("out_file", "in_matrix_file")]),
        (motion_correct, transform_atlas, [("mean_img", "reference")]),
    ])

    # Connect transformed atlas to AIF detection
    workflow.connect([
        (transform_atlas, aif_detect, [("out_file", "atlas_mask_file")]),
    ])

    # DSC Map Computation to CBV Normalization
    workflow.connect([
        (compute_maps, normalize, [("out_cbv", "cbv_file")]),
        (inputnode, normalize, [
            ("nawm_mask", "nawm_mask_file"),
            ("brain_mask", "brain_mask_file"),
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

    # Connect to Output Node
    workflow.connect([
        # Motion correction outputs
        (motion_correct, outputnode, [
            ("out_file", "dsc_moco"),
            ("mean_img", "dsc_mean"),
            ("out_params", "motion_params"),
        ]),
        # Perfusion maps
        (compute_maps, outputnode, [
            ("out_cbf", "cbf_map"),
            ("out_cbv", "cbv_map"),
            ("out_mtt", "mtt_map"),
            ("out_tmax", "tmax_map"),
            ("out_psr", "psr_map"),
            ("out_ph", "ph_map"),
        ]),
        # AIF outputs
        (aif_detect, outputnode, [
            ("out_aif_curve", "aif_curve"),
            ("out_aif_mask", "aif_mask"),
            ("qc_metrics", "aif_qc"),
        ]),
        # Leakage correction
        (preload_correct, outputnode, [("k2_map_file", "k2_map")]),
        # Normalized CBV
        (normalize, outputnode, [("ncbv_file", "ncbv_map")]),
        # Registration
        (register, outputnode, [("out_matrix", "dsc_to_t1w_xfm")]),
        # QC metrics
        (merge_qc, outputnode, [("merged_qc", "qc_metrics")]),
    ])

    # Store config in workflow for reference by child nodes
    workflow.config_dsc = config  # type: ignore[attr-defined]
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]

    return workflow
