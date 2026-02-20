"""DSC (Dynamic Susceptibility Contrast) preprocessing interfaces for PMRIPrep.

This module implements Nipype interfaces for DSC perfusion preprocessing steps
including motion correction with QC metrics.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


def calculate_framewise_displacement(
    motion_params: npt.NDArray[Any],
    head_radius: float = 50.0,
) -> npt.NDArray[Any]:
    """Calculate framewise displacement from motion parameters.

    Framewise displacement (FD) is calculated as the sum of absolute
    displacements, with rotations converted to mm using head radius.

    FD formula: |Δtrans| + r × |Δrot|

    Parameters
    ----------
    motion_params : np.ndarray
        Motion parameters array with shape (N, 6).
        Columns: rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm).
        This is FSL mcflirt output format.
    head_radius : float
        Head radius in mm for converting rotations to displacements (default: 50mm).

    Returns
    -------
    np.ndarray
        Framewise displacement for each frame (length N-1, first frame is reference).

    Notes
    -----
    FD is undefined for the first frame (reference), so output has length N-1.
    For convenience, we prepend 0 for the first frame to maintain array length N.

    References
    ----------
    Power et al. (2012). "Spurious but systematic correlations in functional
    connectivity MRI networks arise from subject motion."
    """
    # Ensure 2D array with correct shape
    if motion_params.ndim == 1:
        motion_params = motion_params.reshape(-1, 6)

    n_frames = motion_params.shape[0]
    if n_frames < 2:
        return np.zeros(n_frames, dtype=np.float64)

    # Calculate frame-to-frame differences
    diff = np.diff(motion_params, axis=0)

    # Rotations (columns 0-2) in radians, translations (columns 3-5) in mm
    rot_diff = diff[:, :3]  # radians
    trans_diff = diff[:, 3:]  # mm

    # Convert rotations to mm displacement using head radius
    # Arc length = radius × angle (radians)
    rot_disp = np.abs(rot_diff) * head_radius  # mm

    # Sum absolute displacements: |Δtrans| + r×|Δrot|
    fd = np.sum(np.abs(trans_diff), axis=1) + np.sum(rot_disp, axis=1)

    # Prepend 0 for first frame (reference)
    fd_full = np.zeros(n_frames, dtype=np.float64)
    fd_full[1:] = fd

    return fd_full


def calculate_rms_displacement(
    motion_params: npt.NDArray[Any],
    head_radius: float = 50.0,
) -> npt.NDArray[Any]:
    """Calculate RMS displacement from motion parameters.

    RMS displacement combines rotations and translations into a single
    measure of total displacement from the reference position.

    Parameters
    ----------
    motion_params : np.ndarray
        Motion parameters array with shape (N, 6).
        Columns: rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm).
    head_radius : float
        Head radius in mm for converting rotations to displacements (default: 50mm).

    Returns
    -------
    np.ndarray
        RMS displacement for each frame relative to reference (length N).

    Notes
    -----
    First frame RMS is always 0 (it's the reference).
    """
    # Ensure 2D array with correct shape
    if motion_params.ndim == 1:
        motion_params = motion_params.reshape(-1, 6)

    # Rotations (columns 0-2) in radians, translations (columns 3-5) in mm
    rot = motion_params[:, :3]  # radians
    trans = motion_params[:, 3:]  # mm

    # Convert rotations to mm displacement using head radius
    rot_disp = rot * head_radius  # mm

    # RMS displacement: sqrt(sum of squared displacements / 6)
    all_disp = np.hstack([rot_disp, trans])  # (N, 6)
    rms = np.sqrt(np.mean(all_disp ** 2, axis=1))

    return rms  # type: ignore[no-any-return]


class MotionCorrectionInputSpec(CommandLineInputSpec):
    """Input specification for MotionCorrection interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input 4D DSC time series to motion correct",
    )
    out_file = File(
        genfile=True,
        desc="Output motion-corrected 4D image",
    )
    out_params = File(
        genfile=True,
        desc="Output motion parameters file (.par)",
    )
    ref_vol = traits.Either(
        traits.Int,
        traits.Enum("mean", "median", "first"),
        default="mean",
        usedefault=True,
        desc="Reference volume: int for volume index, 'mean', 'median', or 'first'",
    )
    motion_thresh = traits.Float(
        0.5,
        usedefault=True,
        desc="Framewise displacement threshold (mm) for outlier detection",
    )
    head_radius = traits.Float(
        50.0,
        usedefault=True,
        desc="Head radius (mm) for FD calculation (default: 50mm)",
    )
    cost = traits.Enum(
        "normcorr",
        "corratio",
        "normmi",
        "mutualinfo",
        "leastsq",
        "woods",
        usedefault=True,
        desc="Cost function for motion correction (default: normcorr)",
    )
    interpolation = traits.Enum(
        "trilinear",
        "sinc",
        "spline",
        usedefault=True,
        desc="Interpolation method (default: trilinear)",
    )
    sinc_width = traits.Int(
        7,
        usedefault=True,
        desc="Sinc interpolation window width (default: 7)",
    )
    stages = traits.Int(
        3,
        usedefault=True,
        desc="Number of search stages (default: 3, matching MCFLIRT default)",
    )
    pre_bolus_frames = traits.Int(
        5,
        usedefault=True,
        desc="Number of pre-bolus frames for reference volume computation (default: 5)",
    )


class MotionCorrectionOutputSpec(TraitedSpec):
    """Output specification for MotionCorrection interface."""

    out_file = File(exists=True, desc="Motion-corrected 4D image")
    out_params = File(exists=True, desc="Motion parameters file (.par)")
    mean_img = File(exists=True, desc="Mean image of motion-corrected data")
    fd = traits.List(traits.Float, desc="Framewise displacement per volume (mm)")
    rms = traits.List(traits.Float, desc="RMS displacement per volume (mm)")
    fd_mean = traits.Float(desc="Mean framewise displacement (mm)")
    fd_max = traits.Float(desc="Maximum framewise displacement (mm)")
    rms_mean = traits.Float(desc="Mean RMS displacement (mm)")
    n_outliers = traits.Int(desc="Number of outlier volumes (FD > threshold)")
    outlier_frames = traits.List(traits.Int, desc="Indices of outlier volumes")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class MotionCorrection(CommandLine):
    """DSC motion correction with QC metrics using FSL mcflirt.

    This interface wraps FSL mcflirt to perform motion correction on 4D DSC
    perfusion time series, with comprehensive motion QC metrics including
    framewise displacement (FD) and RMS displacement.

    Parameters
    ----------
    in_file : str
        Input 4D DSC time series
    ref_vol : int or str
        Reference volume: integer index, 'mean', 'median', or 'first' (default: 'mean')
    motion_thresh : float
        FD threshold (mm) for flagging outlier volumes (default: 0.5mm)
    head_radius : float
        Head radius (mm) for FD calculation (default: 50mm)
    cost : str
        Cost function: normcorr, corratio, normmi, etc. (default: normcorr)
    interpolation : str
        Interpolation: trilinear, sinc, or spline (default: trilinear)

    Returns
    -------
    out_file : str
        Motion-corrected 4D image
    out_params : str
        Motion parameters file (FSL format)
    mean_img : str
        Mean image of motion-corrected data
    fd : list
        Framewise displacement per volume (mm)
    rms : list
        RMS displacement per volume (mm)
    fd_mean : float
        Mean FD across all volumes
    fd_max : float
        Maximum FD
    n_outliers : int
        Number of volumes with FD > motion_thresh
    outlier_frames : list
        Indices of outlier volumes
    qc_metrics : dict
        Complete QC metrics dictionary

    Examples
    --------
    >>> moco = MotionCorrection()
    >>> moco.inputs.in_file = 'dsc_raw.nii.gz'
    >>> moco.inputs.motion_thresh = 0.5
    >>> result = moco.run()  # doctest: +SKIP

    Notes
    -----
    Framewise displacement formula: FD = |Δtrans| + r × |Δrot|
    where r is head radius (default 50mm), rotations in radians.

    FSL mcflirt outputs motion parameters in columns:
    rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm)

    Volumes are flagged as outliers when FD > motion_thresh.
    For DSC, motion correction is critical as small movements can cause
    large signal changes due to susceptibility gradients.

    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL mcflirt: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MCFLIRT

    References
    ----------
    Power et al. (2012). "Spurious but systematic correlations in functional
    connectivity MRI networks arise from subject motion."
    """

    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    _cmd = "mcflirt"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        in_file = Path(self.inputs.in_file)
        # Handle .nii.gz extension
        if in_file.name.endswith(".nii.gz"):
            base = in_file.name[:-7]
            ext = ".nii.gz"
        elif in_file.name.endswith(".nii"):
            base = in_file.stem
            ext = ".nii"
        else:
            base = in_file.stem
            ext = in_file.suffix

        return str(Path.cwd() / f"{base}_moco{ext}")

    def _gen_params_filename(self) -> str:
        """Generate motion parameters filename."""
        if self.inputs.out_params:
            return str(self.inputs.out_params)

        # mcflirt outputs {basename}.par
        out_file = Path(self._gen_outfilename())
        if out_file.name.endswith(".nii.gz"):
            base = out_file.name[:-7]
        elif out_file.name.endswith(".nii"):
            base = out_file.stem
        else:
            base = out_file.stem

        return str(out_file.parent / f"{base}.par")

    def _gen_mean_filename(self) -> str:
        """Generate mean image filename."""
        out_file = Path(self._gen_outfilename())
        if out_file.name.endswith(".nii.gz"):
            base = out_file.name[:-7]
            ext = ".nii.gz"
        elif out_file.name.endswith(".nii"):
            base = out_file.stem
            ext = ".nii"
        else:
            base = out_file.stem
            ext = out_file.suffix

        return str(out_file.parent / f"{base}_mean_reg{ext}")

    def _compute_pre_bolus_mean(self) -> str:
        """Compute pre-bolus mean reference image and return its path."""
        import nibabel as nib
        import tempfile

        in_file = str(self.inputs.in_file)
        pre_bolus_frames = int(self.inputs.pre_bolus_frames)

        img = nib.load(in_file)
        data = img.get_fdata()  # type: ignore[attr-defined]
        n_timepoints = data.shape[-1]
        # Match perf-plus: use first 5 volumes for reference (fixed),
        # regardless of configured pre_bolus_frames.
        n_frames = min(5, n_timepoints)
        n_frames = max(1, n_frames)

        pre_bolus_mean = np.mean(data[..., :n_frames], axis=-1)
        fd, ref_path = tempfile.mkstemp(
            suffix="_prebolus_mean.nii.gz", dir=str(Path.cwd())
        )
        import os
        os.close(fd)
        ref_img = nib.Nifti1Image(pre_bolus_mean.astype(np.float32), img.affine, img.header)  # type: ignore[attr-defined]
        nib.save(ref_img, ref_path)
        return ref_path

    @property
    def cmdline(self) -> str:
        """Build the mcflirt command line."""
        in_file = str(self.inputs.in_file)
        out_file = self._gen_outfilename()

        # Remove extension for mcflirt output basename
        out_path = Path(out_file)
        if out_path.name.endswith(".nii.gz"):
            out_base = str(out_path)[:-7]
        elif out_path.name.endswith(".nii"):
            out_base = str(out_path)[:-4]
        else:
            out_base = str(out_path)

        # Compute pre-bolus mean reference image
        ref_file = self._compute_pre_bolus_mean()

        # Build command
        cmd_parts = [
            "mcflirt",
            f"-in {in_file}",
            f"-out {out_base}",
            "-plots",  # Output motion parameters
            "-stats",  # Output mean volume
            "-mats",  # Output transformation matrices
        ]

        # Use pre-bolus mean as reference via -reffile
        cmd_parts.append(f"-reffile {ref_file}")

        # Cost function
        cost = self.inputs.cost
        cmd_parts.append(f"-cost {cost}")

        # Interpolation
        interp = self.inputs.interpolation
        if interp == "sinc":
            cmd_parts.append("-sinc_final")
            sinc_width = self.inputs.sinc_width
            cmd_parts.append(f"-sinc_width {sinc_width}")
        elif interp == "spline":
            cmd_parts.append("-spline_final")
        # trilinear is default, no flag needed

        # Number of stages
        stages = self.inputs.stages
        cmd_parts.append(f"-stages {stages}")

        return " ".join(cmd_parts)

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs and compute QC metrics."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        out_file = self._gen_outfilename()
        outputs["out_file"] = out_file

        # mcflirt outputs
        out_path = Path(out_file)
        if out_path.name.endswith(".nii.gz"):
            base = str(out_path)[:-7]
        elif out_path.name.endswith(".nii"):
            base = str(out_path)[:-4]
        else:
            base = str(out_path)

        # Motion parameters file
        params_file = f"{base}.par"
        outputs["out_params"] = params_file

        # Mean image -- with -reffile mcflirt does not produce _mean_reg.nii.gz,
        # so we compute the mean of the motion-corrected output ourselves.
        mean_file = f"{base}_mean_reg.nii.gz"
        if not Path(mean_file).exists() and Path(out_file).exists():
            import nibabel as nib
            moco_img = nib.load(out_file)
            moco_data = moco_img.get_fdata()  # type: ignore[attr-defined]
            mean_data = np.mean(moco_data, axis=-1).astype(np.float32)
            mean_img = nib.Nifti1Image(mean_data, moco_img.affine, moco_img.header)  # type: ignore[attr-defined]
            nib.save(mean_img, mean_file)
        outputs["mean_img"] = mean_file

        # Compute motion metrics if params file exists
        qc_metrics: dict[str, Any] = {}

        if Path(params_file).exists():
            try:
                # Load motion parameters
                motion_params = np.loadtxt(params_file)

                # Calculate FD and RMS
                head_radius = float(self.inputs.head_radius)
                fd = calculate_framewise_displacement(motion_params, head_radius)
                rms = calculate_rms_displacement(motion_params, head_radius)

                # Convert to lists for Nipype traits
                fd_list = [float(x) for x in fd]
                rms_list = [float(x) for x in rms]

                outputs["fd"] = fd_list
                outputs["rms"] = rms_list

                # Summary metrics
                fd_mean = float(np.mean(fd[1:]))  # Exclude first frame (always 0)
                fd_max = float(np.max(fd))
                rms_mean = float(np.mean(rms))

                outputs["fd_mean"] = fd_mean
                outputs["fd_max"] = fd_max
                outputs["rms_mean"] = rms_mean

                # Outlier detection
                motion_thresh = float(self.inputs.motion_thresh)
                outlier_mask = fd > motion_thresh
                n_outliers = int(np.sum(outlier_mask))
                outlier_frames = [int(i) for i in np.where(outlier_mask)[0]]

                outputs["n_outliers"] = n_outliers
                outputs["outlier_frames"] = outlier_frames

                from pmriprep.utils.qc_standards import (
                    get_composite_motion_status,
                )
                pct_outliers = 100.0 * n_outliers / max(1, len(fd_list))
                motion_status = get_composite_motion_status(
                    fd_mean, fd_max, pct_outliers,
                )

                # QC metrics dict
                qc_metrics = {
                    "fd_mean": fd_mean,
                    "fd_max": fd_max,
                    "motion_status": motion_status,
                    "fd_list": fd_list,
                    "rms_mean": rms_mean,
                    "rms_list": rms_list,
                    "n_outliers": n_outliers,
                    "outlier_frames": outlier_frames,
                    "motion_thresh": motion_thresh,
                    "n_volumes": len(fd_list),
                    "percent_outliers": 100.0 * n_outliers / max(1, len(fd_list)),
                }

            except Exception:
                # If metrics computation fails, provide empty defaults
                pass

        outputs["qc_metrics"] = qc_metrics
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        if name == "out_params":
            return self._gen_params_filename()
        return None


class DSCRegistrationInputSpec(CommandLineInputSpec):
    """Input specification for DSCRegistration interface."""

    t1w_file = File(
        exists=True,
        mandatory=True,
        desc="T1-weighted structural image to register to DSC space",
    )
    dsc_mean = File(
        exists=True,
        mandatory=True,
        desc="Mean DSC image (reference/target space)",
    )
    out_file = File(
        genfile=True,
        desc="Output registered T1w in DSC space",
    )
    out_matrix = File(
        genfile=True,
        desc="Output transformation matrix (FLIRT .mat format)",
    )
    lesion_mask = File(
        exists=True,
        desc="Optional lesion mask for cost function weighting (excludes lesion region)",
    )
    brain_mask = File(
        exists=True,
        desc="Brain mask for QC metric calculation (optional)",
    )
    use_flirt = traits.Bool(
        True,
        usedefault=True,
        desc="Use two-stage FLIRT registration (default: True). Set False for BBR with FLIRT init",
    )
    dof = traits.Int(
        6,
        usedefault=True,
        desc="Degrees of freedom for registration (default: 6 for rigid)",
    )
    cost = traits.Enum(
        "normmi",
        "normcorr",
        "corratio",
        "mutualinfo",
        "leastsq",
        usedefault=True,
        desc="Cost function for FLIRT registration (default: normmi)",
    )
    wm_seg = File(
        exists=True,
        desc="White matter segmentation (required for BBR mode)",
    )
    reference_type = traits.Enum(
        "precon",
        "postcon",
        "any",
        usedefault=True,
        desc="Expected T1w type for provenance/QC metadata: precon (default), postcon, or any",
    )


class DSCRegistrationOutputSpec(TraitedSpec):
    """Output specification for DSCRegistration interface."""

    out_file = File(exists=True, desc="Registered T1w in DSC space")
    out_matrix = File(exists=True, desc="Affine transformation matrix (FLIRT format)")
    dice = traits.Float(desc="DICE coefficient (if brain mask provided)")
    correlation = traits.Float(desc="Correlation coefficient (if brain mask provided)")
    nmi = traits.Float(desc="Normalized mutual information")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class DSCRegistration(CommandLine):
    """DSC to T1w structural registration for perfusion map overlays.

    This interface wraps FSL FLIRT (and optionally BBR) to register a T1-weighted
    structural image to DSC space. The registration enables overlay of perfusion
    maps on anatomical images for clinical interpretation.

    Two-stage registration is used in both modes:
      Stage 1: Coarse global search (normcorr, -180 to 180 deg, coarsesearch 30)
      Stage 2: Refined (normmi, -30 to 30 deg, init from stage 1)

    Two modes are available:

    1. **FLIRT mode** (default, use_flirt=True): Uses two-stage FSL FLIRT rigid
       registration. Recommended for most cases.

    2. **BBR mode** (use_flirt=False): Uses two-stage FLIRT initialization followed
       by boundary-based registration with white matter segmentation. Requires wm_seg.
       May not be appropriate when T1w and DSC have different contrast states.

    Both modes support lesion mask cost function weighting, where the lesion
    region is excluded from the cost function to prevent tumors from biasing
    the registration.

    Parameters
    ----------
    t1w_file : str
        T1-weighted image to register to DSC space
    dsc_mean : str
        Mean DSC image (target space)
    lesion_mask : str, optional
        Binary lesion mask - lesion voxels will have reduced weight
    brain_mask : str, optional
        Brain mask for computing QC metrics
    use_flirt : bool
        If True, use FLIRT only; if False (default), use BBR with FLIRT init
    dof : int
        Degrees of freedom (default: 6 for rigid body)
    cost : str
        Cost function for FLIRT: normmi, normcorr, corratio (default: normmi)
    wm_seg : str
        White matter segmentation (required for BBR mode)

    Returns
    -------
    out_file : str
        Registered T1w in DSC space
    out_matrix : str
        FSL FLIRT transformation matrix
    dice : float
        DICE coefficient of brain masks (if brain_mask provided)
    correlation : float
        Correlation coefficient within brain mask
    nmi : float
        Normalized mutual information
    qc_metrics : dict
        Complete QC metrics dictionary

    Examples
    --------
    Using BBR mode (recommended):

    >>> reg = DSCRegistration()
    >>> reg.inputs.t1w_file = 'T1w_brain.nii.gz'
    >>> reg.inputs.dsc_mean = 'dsc_mean.nii.gz'
    >>> reg.inputs.wm_seg = 'T1w_wm_probseg.nii.gz'
    >>> result = reg.run()  # doctest: +SKIP

    Using FLIRT mode with lesion mask:

    >>> reg = DSCRegistration()
    >>> reg.inputs.t1w_file = 'T1w_brain.nii.gz'
    >>> reg.inputs.dsc_mean = 'dsc_mean.nii.gz'
    >>> reg.inputs.use_flirt = True
    >>> reg.inputs.lesion_mask = 'lesion.nii.gz'
    >>> result = reg.run()  # doctest: +SKIP

    Notes
    -----
    BBR (Boundary-Based Registration) uses the white matter segmentation to
    drive the registration based on the boundary between white matter and
    gray matter, which is visible in both T1w and EPI-based images.

    Lesion mask weighting creates a cost function weight image where:
    weight = 1.0 - lesion_mask (i.e., lesion voxels have weight 0)

    QC metrics computed:
    - DICE: Overlap of brain masks (requires brain_mask input)
    - Correlation: Pearson correlation within brain mask
    - NMI: Normalized mutual information (intensity correspondence)

    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL FLIRT: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT
    FSL BBR: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT_BBR
    """

    input_spec = DSCRegistrationInputSpec
    output_spec = DSCRegistrationOutputSpec

    _cmd = "flirt"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        t1w_file = Path(self.inputs.t1w_file)
        if t1w_file.name.endswith(".nii.gz"):
            base = t1w_file.name[:-7]
            ext = ".nii.gz"
        elif t1w_file.name.endswith(".nii"):
            base = t1w_file.stem
            ext = ".nii"
        else:
            base = t1w_file.stem
            ext = t1w_file.suffix

        return str(Path.cwd() / f"{base}_dsc{ext}")

    def _gen_matfilename(self) -> str:
        """Generate output matrix filename if not specified."""
        if self.inputs.out_matrix:
            return str(self.inputs.out_matrix)

        out_file = Path(self._gen_outfilename())
        if out_file.name.endswith(".nii.gz"):
            base = out_file.name[:-7]
        elif out_file.name.endswith(".nii"):
            base = out_file.stem
        else:
            base = out_file.stem

        return str(out_file.parent / f"{base}.mat")

    @property
    def cmdline(self) -> str:
        """Build the two-stage FLIRT registration command.

        Implements two-stage FLIRT registration:
          Stage 1: Coarse global search (normcorr, wide search range)
          Stage 2: Refined (normmi, narrow search, init from stage 1)

        For BBR mode (use_flirt=False), the two-stage FLIRT serves as
        initialization before the BBR refinement step.
        """
        t1w_file = str(self.inputs.t1w_file)
        dsc_mean = str(self.inputs.dsc_mean)
        out_file = self._gen_outfilename()
        out_matrix = self._gen_matfilename()
        use_flirt = self.inputs.use_flirt
        dof = self.inputs.dof
        cost = self.inputs.cost

        cmd_parts: list[str] = []

        # Determine file extension for temp files
        out_path = Path(out_file)
        ext = ".nii.gz" if out_path.name.endswith(".nii.gz") else ".nii"

        # Check if we have a lesion mask for cost function weighting
        lesion_mask = getattr(self.inputs, "lesion_mask", None)
        weight_file: Optional[str] = None
        if lesion_mask and Path(lesion_mask).exists():
            weight_file = str(out_path.parent / f"_tmp_weight{ext}")
            invert_cmd = f"fslmaths {lesion_mask} -mul -1 -add 1 {weight_file}"
            cmd_parts.append(invert_cmd)

        # Temporary matrix for stage 1 coarse registration
        stage1_matrix = str(out_path.parent / "_tmp_stage1.mat")

        # Weight flags string (reused across stages)
        # Lesion weight is in T1w space (same as -in), so only use -inweight.
        weight_flags = ""
        if weight_file:
            weight_flags = f" -inweight {weight_file}"

        # --- Stage 1: Coarse global search ---
        stage1_cmd = (
            f"flirt -in {t1w_file} -ref {dsc_mean} "
            f"-omat {stage1_matrix} "
            f"-dof {dof} -cost normcorr "
            f"-searchrx -180 180 -searchry -180 180 -searchrz -180 180 "
            f"-coarsesearch 30 -finesearch 10"
        )
        stage1_cmd += weight_flags
        cmd_parts.append(stage1_cmd)

        if use_flirt:
            # --- Stage 2: Refined registration ---
            stage2_cmd = (
                f"flirt -in {t1w_file} -ref {dsc_mean} "
                f"-out {out_file} -omat {out_matrix} "
                f"-dof {dof} -cost {cost} "
                f"-init {stage1_matrix} "
                f"-searchrx -30 30 -searchry -30 30 -searchrz -30 30"
            )
            stage2_cmd += weight_flags
            cmd_parts.append(stage2_cmd)

            cmd_parts.append(f"rm -f {stage1_matrix}")
        else:
            # BBR mode: two-stage FLIRT as initialization for BBR
            import warnings
            warnings.warn(
                "BBR registration requested. BBR assumes similar contrast "
                "states between T1w and DSC. If DSC was acquired after "
                "contrast injection, BBR may be suboptimal. Consider FLIRT "
                "mode (use_flirt=True)."
            )
            wm_seg = getattr(self.inputs, "wm_seg", None)

            stage2_matrix = str(out_path.parent / "_tmp_stage2.mat")
            stage2_cmd = (
                f"flirt -in {t1w_file} -ref {dsc_mean} "
                f"-omat {stage2_matrix} "
                f"-dof {dof} -cost {cost} "
                f"-init {stage1_matrix} "
                f"-searchrx -30 30 -searchry -30 30 -searchrz -30 30"
            )
            stage2_cmd += weight_flags
            cmd_parts.append(stage2_cmd)

            cmd_parts.append(f"rm -f {stage1_matrix}")

            if wm_seg and Path(wm_seg).exists():
                wm_thresh = str(out_path.parent / f"_tmp_wm_thresh{ext}")
                cmd_parts.append(
                    f"fslmaths {wm_seg} -thr 0.5 -bin {wm_thresh}"
                )

                bbr_cmd = (
                    f"flirt -in {t1w_file} -ref {dsc_mean} "
                    f"-out {out_file} -omat {out_matrix} "
                    f"-dof {dof} -cost bbr "
                    f"-init {stage2_matrix} "
                    f"-wmseg {wm_thresh}"
                )
                if weight_file:
                    bbr_cmd += f" -inweight {weight_file}"

                cmd_parts.append(bbr_cmd)
                cmd_parts.append(f"rm -f {stage2_matrix} {wm_thresh}")
            else:
                apply_cmd = (
                    f"flirt -in {t1w_file} -ref {dsc_mean} "
                    f"-out {out_file} -init {stage2_matrix} -applyxfm"
                )
                cmd_parts.append(apply_cmd)
                cmd_parts.append(f"cp {stage2_matrix} {out_matrix}")
                cmd_parts.append(f"rm -f {stage2_matrix}")

        # Clean up weight file if created
        if weight_file:
            cmd_parts.append(f"rm -f {weight_file}")

        return "bash -c '" + " && ".join(cmd_parts) + "'"

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs and compute QC metrics."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        outputs["out_matrix"] = self._gen_matfilename()

        # Compute QC metrics after registration
        qc_metrics: dict[str, Any] = {}

        out_file = self._gen_outfilename()
        dsc_mean = str(self.inputs.dsc_mean)

        # Only compute metrics if output file exists
        if Path(out_file).exists():
            try:
                # Import here to avoid circular imports
                from pmriprep.utils.registration import (
                    calculate_correlation_from_files,
                    calculate_nmi_from_files,
                    calculate_dice_from_files,
                    calculate_dice,
                )

                # Get optional brain mask
                brain_mask = getattr(self.inputs, "brain_mask", None)
                mask_path: Optional[str] = None
                if brain_mask and Path(brain_mask).exists():
                    mask_path = str(brain_mask)

                # Calculate NMI (always possible)
                try:
                    nmi = calculate_nmi_from_files(
                        out_file, dsc_mean, mask_path=mask_path
                    )
                    outputs["nmi"] = nmi
                    qc_metrics["nmi"] = nmi
                except Exception:
                    pass

                # Calculate correlation within brain mask
                try:
                    corr = calculate_correlation_from_files(
                        out_file, dsc_mean, mask_path=mask_path
                    )
                    outputs["correlation"] = corr
                    qc_metrics["correlation"] = corr
                except Exception:
                    pass

                # Calculate DICE if brain mask provided
                # Threshold registered T1w to create pseudo brain mask, compare
                # with the DSC brain mask
                if mask_path:
                    import nibabel as nib

                    try:
                        # Load registered T1w and threshold to create pseudo mask
                        reg_img = nib.load(out_file)
                        reg_data = reg_img.get_fdata()  # type: ignore[attr-defined]
                        nonzero = reg_data[reg_data > 0]
                        if len(nonzero) > 100:
                            thresh = np.percentile(nonzero, 15)
                            reg_mask = (reg_data > thresh).astype(np.float64)
                        else:
                            reg_mask = (reg_data > 0).astype(np.float64)

                        # Load DSC brain mask
                        dsc_mask_data = nib.load(mask_path).get_fdata()  # type: ignore[attr-defined]

                        # Compute DICE
                        dice = calculate_dice(reg_mask, dsc_mask_data, threshold=0.5)
                        outputs["dice"] = dice
                        qc_metrics["dice"] = dice
                        qc_metrics["brain_mask_used"] = True

                        if dice < 0.8:
                            import warnings
                            warnings.warn(
                                f"Registration DICE coefficient is low ({dice:.3f} < 0.8). "
                                "Check registration quality."
                            )
                    except Exception:
                        qc_metrics["brain_mask_used"] = True
                        qc_metrics["dice_computation_failed"] = True

            except Exception:
                # If QC computation fails, continue without metrics
                pass

        # Add registration metadata for provenance
        qc_metrics["reference_type"] = self.inputs.reference_type
        qc_metrics["use_flirt"] = self.inputs.use_flirt
        qc_metrics["dof"] = self.inputs.dof
        qc_metrics["cost_function"] = self.inputs.cost

        outputs["qc_metrics"] = qc_metrics
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        if name == "out_matrix":
            return self._gen_matfilename()
        return None
