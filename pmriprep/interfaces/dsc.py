"""DSC (Dynamic Susceptibility Contrast) preprocessing interfaces for PMRIPrep.

This module implements Nipype interfaces for DSC perfusion preprocessing steps
including motion correction with QC metrics.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


from pmriprep.utils.motion import (  # noqa: E402
    calculate_framewise_displacement,
    calculate_rms_displacement,
)


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
    baseline_frames = traits.Range(
        low=1,
        value=5,
        usedefault=True,
        desc="Number of pre-bolus frames for the motion reference mean",
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
    baseline_frames : int
        Number of pre-bolus frames for the motion reference mean.
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
    Framewise displacement formula: ``FD = |Δtrans| + r × |Δrot|``
    where r is head radius (default 50mm), rotations in radians.

    FSL mcflirt outputs motion parameters in columns:
    rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm)

    Volumes are flagged as outliers when FD > motion_thresh.
    For DSC, motion correction is critical as small movements can cause
    large signal changes due to susceptibility gradients.

    Requires FSL to be installed and available in PATH.

    """

    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    _cmd = "mcflirt"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        from pmriprep.utils.nifti import nifti_gz_name

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_moco"))

    def _gen_params_filename(self) -> str:
        """Generate motion parameters filename."""
        if self.inputs.out_params:
            return str(self.inputs.out_params)

        from pmriprep.utils.nifti import nifti_gz_name

        # mcflirt outputs {basename}.par
        base = nifti_gz_name(self.inputs.in_file, "_moco")[:-7]  # strip .nii.gz
        return str(Path.cwd() / f"{base}.par")

    def _compute_pre_bolus_mean(self) -> str:
        """Compute pre-bolus mean reference image and return its path."""
        import nibabel as nib
        import tempfile

        in_file = str(self.inputs.in_file)
        baseline_frames = int(self.inputs.baseline_frames)

        img = nib.load(in_file)
        data = img.get_fdata()  # type: ignore[attr-defined]
        n_timepoints = data.shape[-1]
        n_frames = min(max(1, baseline_frames), n_timepoints)

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


# =============================================================================
# DSC Registration Constants
# =============================================================================

# FLIRT search step sizes (degrees) — 15° coarse gives ~7.5° worst-case
# distance to true optimum, much better than 30° (15° worst-case)
DSC_COARSE_SEARCH: int = 15
DSC_FINE_SEARCH: int = 6


class DSCRegistrationInputSpec(BaseInterfaceInputSpec):
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
    brain_mask = File(
        exists=True,
        desc="Brain mask for QC metric calculation (optional)",
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
    out_h5 = File(exists=True, desc="Affine transform in ITK composite (.h5) format")
    dice = traits.Float(desc="DICE coefficient (if brain mask provided)")
    correlation = traits.Float(desc="Correlation coefficient (if brain mask provided)")
    nmi = traits.Float(desc="Normalized mutual information")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class DSCRegistration(SimpleInterface):
    """DSC to T1w structural registration with dual-path selection.

    Registers a T1-weighted structural image to DSC EPI space using FSL FLIRT.
    Uses a dual-path Stage 1 strategy to guard
    against FLIRT coarse-search local minima caused by HD-BET non-determinism.

    Dual-path Stage 1 (coarse global search with normcorr):
      Path A (header-init): Header-aligned T1w, [-30,30] deg search
      Path B (full-search): Original histogram-matched T1w, [-180,180] deg search
      Best path selected by DICE overlap (B must beat A by margin to win).

    Stage 2 (refinement): normmi, [-30,30] deg, initialised from best Stage 1.

    The dual-path design stabilises registration against small input
    variations (e.g. HD-BET GPU non-determinism changing a few boundary
    voxels) that can tip a single-path coarse search into a different
    basin of attraction.

    Parameters
    ----------
    t1w_file : str
        T1-weighted image to register to DSC space.
    dsc_mean : str
        Mean DSC image (target space).
    brain_mask : str, optional
        Brain mask for computing QC metrics.
    dof : int, optional
        Degrees of freedom (default: 6 for rigid body).
    cost : {'normmi', 'normcorr', 'corratio', 'mutualinfo', 'leastsq'}, optional
        Cost function for FLIRT (default: normmi).

    Returns
    -------
    out_file : str
        Registered T1w in DSC space.
    out_matrix : str
        FSL FLIRT transformation matrix.
    dice : float
        DICE coefficient of brain masks (if brain_mask provided).
    correlation : float
        Correlation coefficient within brain mask.
    nmi : float
        Normalized mutual information.
    qc_metrics : dict
        Complete QC metrics dictionary including selection metadata.

    Examples
    --------
    >>> reg = DSCRegistration()
    >>> reg.inputs.t1w_file = 'T1w_brain.nii.gz'
    >>> reg.inputs.dsc_mean = 'dsc_mean.nii.gz'
    >>> result = reg.run()  # doctest: +SKIP

    Notes
    -----
    The rigid transform does not correct DSC EPI susceptibility distortion.
    """

    input_spec = DSCRegistrationInputSpec
    output_spec = DSCRegistrationOutputSpec

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if isdefined(self.inputs.out_file) and self.inputs.out_file:
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
        if isdefined(self.inputs.out_matrix) and self.inputs.out_matrix:
            return str(self.inputs.out_matrix)

        out_file = Path(self._gen_outfilename())
        if out_file.name.endswith(".nii.gz"):
            base = out_file.name[:-7]
        elif out_file.name.endswith(".nii"):
            base = out_file.stem
        else:
            base = out_file.stem

        return str(out_file.parent / f"{base}.mat")

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DSC to T1w registration with dual-path selection.

        Parameters
        ----------
        runtime : Any
            Nipype runtime object.

        Returns
        -------
        Any
            Updated runtime object.
        """
        import logging
        import shutil
        import subprocess
        import tempfile

        from pmriprep.utils.registration import (
            DICE_SELECTION_MARGIN,
            compute_dice_for_selection,
            compute_header_init_matrix,
        )

        logger = logging.getLogger("pmriprep.interfaces.dsc")

        out_file = self._gen_outfilename()
        out_matrix = self._gen_matfilename()
        t1w_file = str(self.inputs.t1w_file)
        dsc_mean = str(self.inputs.dsc_mean)
        dof = self.inputs.dof
        cost = self.inputs.cost

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ext = ".nii.gz"

            # =============================================================
            # Stage 1: Dual-path coarse global search (normcorr)
            # =============================================================
            # Path A: header-aligned T1w with narrow search
            # Path B: original T1w with full-sphere search
            # Best path selected by DICE overlap.

            # --- Path A: header-initialised, narrow search ---
            # Compute a FLIRT-format matrix from NIfTI headers, then
            # run FLIRT with t1w_file as input and this matrix as -init.
            # Both paths use t1w_file as -in, keeping matrix spaces consistent.
            stage1a_matrix = str(temp_path / "stage1a.mat")
            stage1a_out = str(temp_path / f"stage1a_out{ext}")
            error_a: Optional[str] = None

            try:
                header_init_mat = compute_header_init_matrix(
                    t1w_file, dsc_mean, temp_path
                )
                stage1a_cmd = [
                    "flirt", "-in", t1w_file, "-ref", dsc_mean,
                    "-out", stage1a_out, "-omat", stage1a_matrix,
                    "-init", header_init_mat,
                    "-dof", str(dof), "-cost", "normcorr",
                    "-searchrx", "-30", "30",
                    "-searchry", "-30", "30",
                    "-searchrz", "-30", "30",
                    "-coarsesearch", str(DSC_COARSE_SEARCH),
                    "-finesearch", str(DSC_FINE_SEARCH),
                ]
                subprocess.run(stage1a_cmd, check=True)
            except Exception as exc:
                error_a = str(exc)
                logger.warning("Stage 1 path A (header-init) failed: %s", exc)

            # --- Path B: full-sphere search ---
            stage1b_matrix = str(temp_path / "stage1b.mat")
            stage1b_out = str(temp_path / f"stage1b_out{ext}")
            error_b: Optional[str] = None

            try:
                stage1b_cmd = [
                    "flirt", "-in", t1w_file, "-ref", dsc_mean,
                    "-out", stage1b_out, "-omat", stage1b_matrix,
                    "-dof", str(dof), "-cost", "normcorr",
                    "-searchrx", "-180", "180",
                    "-searchry", "-180", "180",
                    "-searchrz", "-180", "180",
                    "-coarsesearch", str(DSC_COARSE_SEARCH),
                    "-finesearch", str(DSC_FINE_SEARCH),
                ]
                subprocess.run(stage1b_cmd, check=True)
            except Exception as exc:
                error_b = str(exc)
                logger.warning("Stage 1 path B (full-search) failed: %s", exc)

            if error_a and error_b:
                raise RuntimeError(
                    f"Both Stage 1 paths failed:\n"
                    f"Path A (header-aligned): {error_a}\n"
                    f"Path B (full-search): {error_b}"
                )

            # --- Select best Stage 1 via DICE ---
            file_a_ok = error_a is None and Path(stage1a_out).exists()
            file_b_ok = error_b is None and Path(stage1b_out).exists()

            dice_a = (
                compute_dice_for_selection(stage1a_out, dsc_mean)
                if file_a_ok else 0.0
            )
            dice_b = (
                compute_dice_for_selection(stage1b_out, dsc_mean)
                if file_b_ok else 0.0
            )

            # B must beat A by margin to be selected (preference for header-aligned)
            if file_b_ok and (
                not file_a_ok or dice_b > dice_a + DICE_SELECTION_MARGIN
            ):
                stage1_matrix = stage1b_matrix
                stage1_path = "B"
            else:
                stage1_matrix = stage1a_matrix
                stage1_path = "A"

            logger.info(
                "Stage 1 path selection: %s (DICE A=%.4f, B=%.4f)",
                stage1_path, dice_a, dice_b,
            )

            # =============================================================
            # Stage 2: Refined registration (normmi, narrow search)
            # =============================================================
            flirt_matrix = str(temp_path / "flirt.mat")
            flirt_out = str(temp_path / f"flirt_out{ext}")

            stage2_cmd = [
                "flirt", "-in", t1w_file, "-ref", dsc_mean,
                "-out", flirt_out, "-omat", flirt_matrix,
                "-dof", str(dof), "-cost", cost,
                "-init", stage1_matrix,
                "-searchrx", "-30", "30",
                "-searchry", "-30", "30",
                "-searchrz", "-30", "30",
            ]
            subprocess.run(stage2_cmd, check=True)

            # Selection metadata for QC
            selection_meta: dict[str, Any] = {
                "registration_method": "flirt",
                "selection_reason": "dual_path_flirt",
                "stage1_path_selected": stage1_path,
                "stage1_dice_a": round(dice_a, 4),
                "stage1_dice_b": round(dice_b, 4),
            }
            shutil.copy(flirt_out, out_file)
            shutil.copy(flirt_matrix, out_matrix)

        # Compute QC metrics on the final selected output
        qc_metrics = self._compute_qc_metrics(out_file, dsc_mean)
        qc_metrics.update(selection_meta)
        qc_metrics["reference_type"] = self.inputs.reference_type
        qc_metrics["dof"] = self.inputs.dof
        qc_metrics["cost_function"] = self.inputs.cost

        # Set outputs
        from pmriprep.utils.transforms import fsl_to_x5

        self._results["out_file"] = out_file
        self._results["out_matrix"] = out_matrix
        self._results["out_h5"] = fsl_to_x5(
            out_matrix,
            reference=str(self.inputs.dsc_mean),
            moving=str(self.inputs.t1w_file),
        )
        self._results["qc_metrics"] = qc_metrics

        if "dice" in qc_metrics:
            self._results["dice"] = qc_metrics["dice"]
        if "correlation" in qc_metrics:
            self._results["correlation"] = qc_metrics["correlation"]
        if "nmi" in qc_metrics:
            self._results["nmi"] = qc_metrics["nmi"]

        return runtime

    def _compute_qc_metrics(
        self, out_file: str, dsc_mean: str
    ) -> dict[str, Any]:
        """Compute registration QC metrics.

        Parameters
        ----------
        out_file : str
            Registered T1w image path.
        dsc_mean : str
            DSC reference image path.

        Returns
        -------
        dict
            Dictionary of QC metrics.
        """
        import nibabel as nib

        qc_metrics: dict[str, Any] = {}

        if not Path(out_file).exists():
            return qc_metrics

        try:
            from pmriprep.utils.registration import (
                calculate_correlation_from_files,
                calculate_nmi_from_files,
                calculate_dice,
            )

            brain_mask = None
            if (
                isdefined(self.inputs.brain_mask)
                and Path(self.inputs.brain_mask).exists()
            ):
                brain_mask = str(self.inputs.brain_mask)

            # NMI
            try:
                nmi = calculate_nmi_from_files(
                    out_file, dsc_mean, mask_path=brain_mask
                )
                qc_metrics["nmi"] = nmi
            except Exception:
                pass

            # Correlation
            try:
                corr = calculate_correlation_from_files(
                    out_file, dsc_mean, mask_path=brain_mask
                )
                qc_metrics["correlation"] = corr
            except Exception:
                pass

            # DICE
            if brain_mask:
                try:
                    reg_img = nib.load(out_file)
                    reg_data = reg_img.get_fdata()  # type: ignore[attr-defined]
                    nonzero = reg_data[reg_data > 0]
                    if len(nonzero) > 100:
                        thresh = np.percentile(nonzero, 15)
                        reg_mask = (reg_data > thresh).astype(np.float64)
                    else:
                        reg_mask = (reg_data > 0).astype(np.float64)

                    dsc_mask_data = nib.load(brain_mask).get_fdata()  # type: ignore[attr-defined]
                    dice = calculate_dice(reg_mask, dsc_mask_data, threshold=0.5)
                    qc_metrics["dice"] = dice
                    qc_metrics["brain_mask_used"] = True

                    if dice < 0.8:
                        logging.getLogger(__name__).warning(
                            "Registration DICE coefficient is low "
                            "(%.3f < 0.8). Check registration quality.",
                            dice,
                        )
                except Exception:
                    qc_metrics["brain_mask_used"] = True
                    qc_metrics["dice_computation_failed"] = True

        except Exception:
            pass

        return qc_metrics
