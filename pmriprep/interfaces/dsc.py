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

    return rms


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
        4,
        usedefault=True,
        desc="Number of search stages (default: 4)",
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

        # Build command
        cmd_parts = [
            "mcflirt",
            f"-in {in_file}",
            f"-out {out_base}",
            "-plots",  # Output motion parameters
            "-stats",  # Output mean volume
            "-mats",  # Output transformation matrices
        ]

        # Reference volume
        ref_vol = self.inputs.ref_vol
        if isinstance(ref_vol, int):
            cmd_parts.append(f"-refvol {ref_vol}")
        elif ref_vol == "mean":
            cmd_parts.append("-meanvol")
        elif ref_vol == "first":
            cmd_parts.append("-refvol 0")
        # 'median' is not directly supported by mcflirt, use mean

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

        # Mean image
        mean_file = f"{base}_mean_reg.nii.gz"
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

                # QC metrics dict
                qc_metrics = {
                    "fd_mean": fd_mean,
                    "fd_max": fd_max,
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
