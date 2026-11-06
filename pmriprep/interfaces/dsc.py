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
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


from pmriprep.utils.motion import (  # noqa: E402
    summarize_motion,
)


class MotionCorrectionInputSpec(BaseInterfaceInputSpec):
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
    apply_motion_correction = traits.Bool(
        True, usedefault=True,
        desc="Apply estimated ANTs transforms; otherwise retain the original series",
    )
    num_threads = traits.Range(
        low=1, value=1, usedefault=True,
        desc="ANTs ITK threads reserved for this node",
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
    fd_mean = traits.Float(desc="Mean framewise displacement (mm)")
    fd_max = traits.Float(desc="Maximum framewise displacement (mm)")
    n_outliers = traits.Int(desc="Number of outlier volumes (FD > threshold)")
    outlier_frames = traits.List(traits.Int, desc="Indices of outlier volumes")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class MotionCorrection(SimpleInterface):
    """Estimate rigid ANTs motion and report common motion QC for DSC.

    Parameters
    ----------
    in_file : str
        Input 4D DSC time series.
    baseline_frames : int
        Number of initial frames defining the fixed reference mean.
    apply_motion_correction : bool
        Apply transforms or retain the input unchanged while estimating motion.
    motion_thresh : float
        FD threshold in millimetres for QC flags.
    head_radius : float
        Radius in millimetres for angular displacement conversion.
    num_threads : int
        ANTs thread count reserved for this node by the scheduler.

    Notes
    -----
    Motion parameters use FSL-convention rotations in radians followed by
    translations in millimetres, about the reference intensity centre.
    """

    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    def _gen_outfilename(self) -> str:
        """Return the requested or generated series filename."""
        from pmriprep.utils.nifti import nifti_gz_name

        if isdefined(self.inputs.out_file):
            return str(Path(self.inputs.out_file).absolute())
        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_moco"))

    def _compute_pre_bolus_mean(self) -> str:
        """Write the fixed baseline mean used for every volume."""
        import nibabel as nib

        image: Any = nib.load(self.inputs.in_file)
        if len(image.shape) != 4:
            raise ValueError("DSC motion input must be four-dimensional")
        count = min(int(self.inputs.baseline_frames), image.shape[3])
        mean = image.get_fdata()[..., :count].mean(axis=-1)
        path = str(Path.cwd() / "motion_reference.nii.gz")
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(mean.astype(np.float32), image.affine, header), path)
        return path

    def _run_interface(self, runtime: Any) -> Any:
        """Run motion estimation and summarize the selected series."""
        import nibabel as nib
        import shutil
        from pmriprep.utils.ants_motion import run_motion_correction

        result = run_motion_correction(
            str(self.inputs.in_file), self._compute_pre_bolus_mean(),
            self._gen_outfilename(), apply=bool(self.inputs.apply_motion_correction),
            num_threads=int(self.inputs.num_threads),
        )
        params_file = result.params_file
        if isdefined(self.inputs.out_params):
            requested = str(Path(self.inputs.out_params).absolute())
            if requested != params_file:
                shutil.copyfile(params_file, requested)
            params_file = requested
        metrics = summarize_motion(np.loadtxt(params_file, ndmin=2),
                                   float(self.inputs.motion_thresh), float(self.inputs.head_radius))
        metrics.update(result.provenance)
        metrics["reference_type"] = "baseline_mean"
        metrics["motion_reference_frames"] = min(int(self.inputs.baseline_frames), metrics["n_volumes"])
        image: Any = nib.load(result.out_file)
        mean_file = str(Path.cwd() / "motion_mean.nii.gz")
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        mean = image.get_fdata().mean(axis=-1).astype(np.float32)
        nib.save(nib.Nifti1Image(mean, image.affine, header), mean_file)
        self._results.update(out_file=result.out_file, out_params=params_file,
                             mean_img=mean_file, fd=metrics["fd_list"],
                             qc_metrics=metrics)
        for key in ("fd_mean", "fd_max", "n_outliers", "outlier_frames"):
            self._results[key] = metrics[key]
        return runtime


# =============================================================================
# DSC Registration Constants
# =============================================================================

# FLIRT search step sizes (degrees) — 15° coarse gives ~7.5° worst-case
# distance to true optimum, much better than 30° (15° worst-case)
DSC_COARSE_SEARCH: int = 15
DSC_FINE_SEARCH: int = 6
# Final DSC-space foreground/mask Dice required before a registration may
# carry quantitative information into another modality.
DSC_QUANTITATIVE_TRANSFER_DICE_MIN: float = 0.90


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
    native_brain_mask = File(
        exists=True,
        desc=(
            "Independently extracted native DSC brain mask for descriptive "
            "registration QC (optional)"
        ),
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
    dice = traits.Float(desc="DICE coefficient in DSC space (if brain mask provided)")
    correlation = traits.Float(desc="Correlation coefficient in DSC space")
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

        Raises
        ------
        RuntimeError
            If both registration paths fail.
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
        logger = logging.getLogger(__name__)

        if not Path(out_file).exists():
            qc_metrics["qc_status"] = "UNKNOWN"
            return qc_metrics

        try:
            from pmriprep.utils.registration import (
                calculate_correlation_from_files,
                calculate_nmi_from_files,
                calculate_dice,
                transform_mask_to_grid,
            )

            brain_mask_dsc = ""
            if (
                isdefined(self.inputs.brain_mask)
                and Path(self.inputs.brain_mask).exists()
            ):
                try:
                    brain_mask_dsc = transform_mask_to_grid(
                        str(self.inputs.brain_mask),
                        dsc_mean,
                        self._gen_matfilename(),
                        out_dir=Path(out_file).parent,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not transform the anatomical brain mask for "
                        "DSC registration QC: %s",
                        exc,
                    )
                    brain_mask_dsc = ""

            # NMI
            try:
                nmi = calculate_nmi_from_files(
                    out_file, dsc_mean, mask_path=brain_mask_dsc or None
                )
                qc_metrics["nmi"] = nmi
            except Exception:
                pass

            # Correlation
            try:
                corr = calculate_correlation_from_files(
                    out_file, dsc_mean, mask_path=brain_mask_dsc or None
                )
                qc_metrics["correlation"] = corr
            except Exception:
                pass

            # DICE
            if brain_mask_dsc:
                try:
                    reg_img = nib.load(out_file)
                    reg_data = reg_img.get_fdata()  # type: ignore[attr-defined]
                    nonzero = reg_data[reg_data > 0]
                    if len(nonzero) > 100:
                        thresh = np.percentile(nonzero, 15)
                        reg_mask = (reg_data > thresh).astype(np.float64)
                    else:
                        reg_mask = (reg_data > 0).astype(np.float64)

                    dsc_mask_data = nib.load(brain_mask_dsc).get_fdata()  # type: ignore[attr-defined]
                    dice = calculate_dice(reg_mask, dsc_mask_data, threshold=0.5)
                    qc_metrics["dice"] = dice
                    qc_metrics["brain_mask_used"] = True

                    if dice < DSC_QUANTITATIVE_TRANSFER_DICE_MIN:
                        logger.warning(
                            "Registration DICE coefficient is low "
                            "(%.3f < %.2f). Quantitative cross-modal "
                            "transfer will be disabled.",
                            dice,
                            DSC_QUANTITATIVE_TRANSFER_DICE_MIN,
                        )
                except Exception as exc:
                    logger.warning(
                        "Could not compute final DSC registration Dice: %s",
                        exc,
                    )
                    qc_metrics["brain_mask_used"] = True
                    qc_metrics["dice_computation_failed"] = True
            else:
                qc_metrics["brain_mask_used"] = False

            native_mask = (
                str(self.inputs.native_brain_mask)
                if isdefined(self.inputs.native_brain_mask)
                and Path(self.inputs.native_brain_mask).exists()
                else ""
            )
            qc_metrics["native_perfusion_mask_metrics_role"] = (
                "descriptive_not_gated"
            )
            qc_metrics["native_perfusion_mask_used"] = bool(native_mask)
            if native_mask:
                try:
                    qc_metrics["native_perfusion_mask_correlation"] = (
                        calculate_correlation_from_files(
                            out_file, dsc_mean, mask_path=native_mask
                        )
                    )
                    qc_metrics["native_perfusion_mask_nmi"] = (
                        calculate_nmi_from_files(
                            out_file, dsc_mean, mask_path=native_mask
                        )
                    )
                    if brain_mask_dsc:
                        transformed_mask = nib.load(brain_mask_dsc).get_fdata()  # type: ignore[attr-defined]
                        native_mask_data = nib.load(native_mask).get_fdata()  # type: ignore[attr-defined]
                        qc_metrics["native_perfusion_mask_dice"] = calculate_dice(
                            transformed_mask, native_mask_data, threshold=0.5
                        )
                except Exception as exc:
                    logger.warning(
                        "Could not compute native-mask DSC registration metrics: %s",
                        exc,
                    )
                    qc_metrics["native_perfusion_mask_metrics_failed"] = True

        except Exception as exc:
            logger.warning("DSC registration QC failed: %s", exc)

        status_dice = qc_metrics.get("dice")
        qc_metrics["qc_status_basis"] = "final_dsc_space_foreground_mask_dice"
        qc_metrics["qc_status_threshold"] = DSC_QUANTITATIVE_TRANSFER_DICE_MIN
        if isinstance(status_dice, (int, float)) and np.isfinite(status_dice):
            qc_metrics["qc_status"] = (
                "PASS"
                if status_dice >= DSC_QUANTITATIVE_TRANSFER_DICE_MIN
                else "POOR"
            )
        else:
            qc_metrics["qc_status"] = "UNKNOWN"

        return qc_metrics
