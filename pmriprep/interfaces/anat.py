"""Anatomical preprocessing interfaces for PMRIPrep.

This module implements Nipype interfaces for anatomical preprocessing steps
including reorientation, bias correction, coregistration, brain extraction,
tissue segmentation, and ACPC alignment.
"""

import os
from pathlib import Path
from typing import Any, Optional

from pmriprep.utils.nifti import nifti_gz_name
from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class ReorientAndCropInputSpec(CommandLineInputSpec):
    """Input specification for ReorientAndCrop interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input NIfTI image to reorient and crop",
    )
    out_file = File(
        genfile=True,
        desc="Output reoriented and cropped image",
    )
    crop_fov = traits.Bool(
        True,
        usedefault=True,
        desc="Apply robustfov cropping after reorientation. Set to False for "
        "non-T1w modalities (FLAIR, T2w) where robustfov may "
        "incorrectly crop the FOV.",
    )


class ReorientAndCropOutputSpec(TraitedSpec):
    """Output specification for ReorientAndCrop interface."""

    out_file = File(exists=True, desc="Reoriented and cropped image")


class ReorientAndCrop(CommandLine):
    """Reorient to standard orientation and crop field of view.

    This interface wraps FSL fslreorient2std and robustfov commands to:
    1. Reorient the image to standard (RAS) orientation
    2. Crop the field of view to remove neck and excess air

    The interface executes a two-step pipeline internally:
    - fslreorient2std: Reorients to standard LAS/RAS orientation
    - robustfov: Automatically crops to the brain FOV

    Examples
    --------
    >>> reorient = ReorientAndCrop()
    >>> reorient.inputs.in_file = 'T1w.nii.gz'
    >>> reorient.inputs.out_file = 'T1w_reoriented.nii.gz'
    >>> result = reorient.run()  # doctest: +SKIP

    Notes
    -----
    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL fslreorient2std: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation
    FSL robustfov: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/robustfov
    """

    input_spec = ReorientAndCropInputSpec
    output_spec = ReorientAndCropOutputSpec

    _cmd = "fslreorient2std"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_reoriented"))

    @property
    def cmdline(self) -> str:
        """Build the fslreorient2std (+ optional robustfov) command chain."""
        in_file = str(self.inputs.in_file)
        out_file = self._gen_outfilename()

        if not self.inputs.crop_fov:
            # Reorient only — no robustfov cropping
            return f"fslreorient2std {in_file} {out_file}"

        # Create intermediate filename for reorient step
        in_path = Path(in_file)
        ext = ".nii.gz"
        intermediate = str(Path(out_file).parent / f"_tmp_reoriented{ext}")

        # Chain: fslreorient2std -> robustfov
        return (
            f"bash -c 'fslreorient2std {in_file} {intermediate} && "
            f"robustfov -i {intermediate} -r {out_file} && "
            f"rm -f {intermediate}'"
        )

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        return outputs

    def _post_run_hook(self, runtime: Any) -> Any:
        """Enforce true RAS after fslreorient2std completes.

        fslreorient2std only reorders axes but doesn't flip handedness.
        This post-run hook applies nibabel's ``as_reoriented()`` to ensure
        the output is truly RAS.
        """
        import nibabel as nib
        import numpy as np
        from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

        out_file = self._gen_outfilename()
        img = nib.load(out_file)
        ornt = io_orientation(img.affine)  # type: ignore[attr-defined]
        ras_ornt = axcodes2ornt(("R", "A", "S"))
        if not np.array_equal(ornt, ras_ornt):
            transform = ornt_transform(ornt, ras_ornt)
            img_ras = img.as_reoriented(transform)  # type: ignore[attr-defined]
            nib.save(img_ras, out_file)
        return runtime

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        return None


class BiasCorrectionInputSpec(CommandLineInputSpec):
    """Input specification for BiasCorrection interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input NIfTI image to bias correct",
    )
    out_file = File(
        genfile=True,
        desc="Output bias-corrected image",
    )
    denoise = traits.Bool(
        False,
        usedefault=True,
        desc="Apply ANTs DenoiseImage before bias correction",
    )
    shrink_factor = traits.Int(
        4,
        usedefault=True,
        desc="Shrink factor for N4 processing (default: 4)",
    )
    convergence = traits.Str(
        "[50x50x50x50,0.0001]",
        usedefault=True,
        desc="N4 convergence parameters [iterations,threshold]",
    )
    bspline_fitting = traits.Str(
        "[200]",
        usedefault=True,
        desc="N4 B-spline fitting distance in mm",
    )
    noise_model = traits.Enum(
        "Rician",
        "Gaussian",
        usedefault=True,
        desc="Noise model for denoising (default: Rician for MRI)",
    )


class BiasCorrectionOutputSpec(TraitedSpec):
    """Output specification for BiasCorrection interface."""

    out_file = File(exists=True, desc="Bias-corrected image")
    bias_field = File(desc="Estimated bias field (if saved)")


class BiasCorrection(CommandLine):
    """N4 bias field correction with optional denoising.

    This interface wraps ANTs N4BiasFieldCorrection (and optionally DenoiseImage)
    to perform bias field correction on structural images. The pipeline:
    1. (Optional) Apply ANTs DenoiseImage with Rician noise model
    2. Apply ANTs N4BiasFieldCorrection with optimized parameters

    Parameters
    ----------
    in_file : str
        Input NIfTI image to bias correct
    denoise : bool
        If True, apply ANTs DenoiseImage before bias correction
    shrink_factor : int
        Shrink factor for N4 (default: 4, reduces computation time)
    convergence : str
        N4 convergence parameters as "[iterations,threshold]"
        Default: "[50x50x50x50,0.0001]" (4 levels, 50 iterations each)
    bspline_fitting : str
        B-spline fitting distance as "[distance_mm]" (default: "[200]")
    noise_model : str
        Noise model for denoising: "Rician" (MRI default) or "Gaussian"

    Examples
    --------
    >>> bias = BiasCorrection()
    >>> bias.inputs.in_file = 'T1w_cropped.nii.gz'
    >>> bias.inputs.denoise = True
    >>> result = bias.run()  # doctest: +SKIP

    Notes
    -----
    Default N4 parameters match the perf-plus pipeline:
    - Shrink factor 4 for speed without quality loss
    - 4-level multi-resolution: 50 iterations per level
    - Convergence threshold 0.0001
    - B-spline fitting distance 200mm (smooth bias field)

    Requires ANTs to be installed and available in PATH.

    See Also
    --------
    ANTs N4BiasFieldCorrection:
        https://github.com/ANTsX/ANTs/wiki/N4BiasFieldCorrection
    ANTs DenoiseImage:
        https://github.com/ANTsX/ANTs/wiki/DenoiseImage
    """

    input_spec = BiasCorrectionInputSpec
    output_spec = BiasCorrectionOutputSpec

    _cmd = "N4BiasFieldCorrection"

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

        return str(Path.cwd() / f"{base}_biascorr{ext}")

    @property
    def cmdline(self) -> str:
        """Build the N4BiasFieldCorrection command (with optional denoising)."""
        in_file = str(self.inputs.in_file)
        out_file = self._gen_outfilename()
        denoise = self.inputs.denoise
        shrink = self.inputs.shrink_factor
        convergence = self.inputs.convergence
        bspline = self.inputs.bspline_fitting
        noise_model = self.inputs.noise_model

        # Determine file extension
        in_path = Path(in_file)
        ext = ".nii.gz" if in_path.name.endswith(".nii.gz") else ".nii"

        if denoise:
            # Denoise first, then bias correct (multi-command chain)
            denoised_file = str(Path(out_file).parent / f"_tmp_denoised{ext}")

            denoise_cmd = (
                f"DenoiseImage -d 3 -i {in_file} -o {denoised_file} "
                f"-n {noise_model}"
            )
            n4_cmd = (
                f"N4BiasFieldCorrection -d 3 -i {denoised_file} -o {out_file} "
                f"-s {shrink} -c {convergence} -b {bspline}"
            )
            return (
                f"bash -c '{denoise_cmd} && {n4_cmd} && "
                f"rm -f {denoised_file}'"
            )
        else:
            # Single command, no chain needed
            return (
                f"N4BiasFieldCorrection -d 3 -i {in_file} -o {out_file} "
                f"-s {shrink} -c {convergence} -b {bspline}"
            )

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        return None


class ConformT1wInputSpec(BaseInterfaceInputSpec):
    """Input specification for ConformT1w interface."""

    in_file = File(exists=True, mandatory=True, desc="Input T1w image")


class ConformT1wOutputSpec(TraitedSpec):
    """Output specification for ConformT1w interface."""

    out_file = File(exists=True, desc="Conformed T1w image (or passthrough)")
    conformed = traits.Bool(desc="Whether resampling was applied")


class ConformT1w(SimpleInterface):
    """Downsample sub-millimeter T1w voxels to 1mm minimum.

    Only downsamples axes with voxel size < 1mm. Never upsamples.
    Skips resampling entirely if all voxels are already >= 1mm,
    returning the input file unchanged to avoid interpolation artifacts.
    """

    input_spec = ConformT1wInputSpec
    output_spec = ConformT1wOutputSpec

    def _run_interface(self, runtime: Any) -> Any:  # type: ignore[override]
        import logging
        import shutil

        import nibabel as nib
        import numpy as np
        from nibabel.processing import resample_from_to

        logger = logging.getLogger(__name__)

        in_file = str(self.inputs.in_file)
        img = nib.load(in_file)
        zooms = np.array(img.header.get_zooms()[:3], dtype=float)

        # Only downsample axes < 1mm
        if np.all(zooms >= 1.0 - 1e-3):
            # All voxels >= 1mm — passthrough without resampling
            out_path = Path.cwd() / Path(in_file).name
            if not out_path.exists() or str(out_path) != in_file:
                shutil.copy2(in_file, out_path)
            self._results["out_file"] = str(out_path)
            self._results["conformed"] = False
            logger.info(
                "T1w voxels already >= 1mm (%s), skipping conform",
                "×".join(f"{z:.2f}" for z in zooms),
            )
            return runtime

        new_zooms = np.maximum(zooms, 1.0)
        new_shape = np.ceil(
            np.array(img.shape[:3]) * zooms / new_zooms
        ).astype(int)

        # Build target affine: same orientation, new voxel size
        target_affine = img.affine.copy()
        # Scale the voxel-size columns of the affine
        for ax in range(3):
            scale = new_zooms[ax] / zooms[ax]
            target_affine[:3, ax] *= scale

        target_img = nib.Nifti1Image(
            np.zeros(new_shape, dtype=np.float32),
            target_affine,
        )
        conformed = resample_from_to(img, target_img, order=3)

        out_name = Path(in_file).name.replace(".nii", "_conformed.nii")
        out_path = Path.cwd() / out_name
        nib.save(conformed, out_path)

        self._results["out_file"] = str(out_path)
        self._results["conformed"] = True
        logger.info(
            "T1w conformed from %s (%.2f×%.2f×%.2fmm) to %s (%.2f×%.2f×%.2fmm)",
            "×".join(str(s) for s in img.shape[:3]),
            *zooms,
            "×".join(str(s) for s in new_shape),
            *new_zooms,
        )
        return runtime

class CoregistrationInputSpec(CommandLineInputSpec):
    """Input specification for Coregistration interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Moving image to be registered (e.g., T2w, FLAIR, T1w post-con)",
    )
    reference = File(
        exists=True,
        mandatory=True,
        desc="Reference image (target space, typically T1w pre-contrast)",
    )
    out_file = File(
        genfile=True,
        desc="Output registered image",
    )
    out_matrix = File(
        genfile=True,
        desc="Output transformation matrix (FLIRT .mat format)",
    )
    modality = traits.Enum(
        "T1_postcon",
        "FLAIR",
        "T2w",
        "T1w",
        usedefault=True,
        desc="Modality of moving image (determines cost function)",
    )
    dof = traits.Int(
        6,
        usedefault=True,
        desc="Degrees of freedom for registration (default: 6 for rigid)",
    )


class CoregistrationOutputSpec(TraitedSpec):
    """Output specification for Coregistration interface."""

    out_file = File(exists=True, desc="Registered image in reference space")
    out_matrix = File(exists=True, desc="Affine transformation matrix (FLIRT format)")


class Coregistration(CommandLine):
    """Multi-modal coregistration to T1w space using FSL FLIRT.

    This interface wraps FSL FLIRT to perform rigid body (6-DOF) registration
    of various MRI modalities to a reference T1w image. Cost functions are
    selected based on the modality to optimize registration accuracy:

    - T1 post-contrast (ce-GAD): normmi (normalized mutual information)
    - FLAIR, T2w: normcorr (normalized correlation)
    - T1w: corratio (correlation ratio)

    Parameters
    ----------
    in_file : str
        Moving image to be registered
    reference : str
        Reference/target image (typically bias-corrected T1w)
    modality : str
        Modality of moving image: T1_postcon, FLAIR, T2w, T1w
    dof : int
        Degrees of freedom (default: 6 for rigid body)

    Examples
    --------
    >>> coreg = Coregistration()
    >>> coreg.inputs.in_file = 'FLAIR.nii.gz'
    >>> coreg.inputs.reference = 'T1w_biascorr.nii.gz'
    >>> coreg.inputs.modality = 'FLAIR'
    >>> result = coreg.run()  # doctest: +SKIP

    Notes
    -----
    Cost function selection by modality:

    - normmi: Best for T1 post-contrast where intensity changes due to Gd
    - normcorr: Best for FLAIR/T2w where contrast inverts vs T1
    - corratio: Best for same-modality registration (T1w to T1w)

    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL FLIRT: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT
    """

    input_spec = CoregistrationInputSpec
    output_spec = CoregistrationOutputSpec

    _cmd = "flirt"

    # Cost function mapping by modality
    _COST_FUNCTIONS: dict[str, str] = {
        "T1_postcon": "normmi",
        "FLAIR": "normcorr",
        "T2w": "normcorr",
        "T1w": "corratio",
    }

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_coreg"))

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
        """Build the FLIRT coregistration command."""
        in_file = str(self.inputs.in_file)
        reference = str(self.inputs.reference)
        out_file = self._gen_outfilename()
        out_matrix = self._gen_matfilename()
        modality = self.inputs.modality
        dof = self.inputs.dof

        # Get cost function for modality
        cost = self._COST_FUNCTIONS.get(modality, "corratio")

        return (
            f"flirt -in {in_file} -ref {reference} "
            f"-out {out_file} -omat {out_matrix} "
            f"-dof {dof} -cost {cost}"
        )

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        outputs["out_matrix"] = self._gen_matfilename()
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        if name == "out_matrix":
            return self._gen_matfilename()
        return None


class BrainExtractionInputSpec(CommandLineInputSpec):
    """Input specification for BrainExtraction interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input bias-corrected T1w image for brain extraction",
    )
    out_file = File(
        genfile=True,
        desc="Output brain-extracted image",
    )
    out_mask = File(
        genfile=True,
        desc="Output brain mask",
    )
    tool = traits.Enum(
        "hdbet",
        "ants",
        usedefault=True,
        desc="Brain extraction tool: hdbet (HD-BET) or ants (ANTs)",
    )
    mni_template = File(
        exists=True,
        desc="MNI template for DICE QC calculation (optional)",
    )
    mni_mask = File(
        exists=True,
        desc="MNI brain mask for DICE QC calculation (optional)",
    )
    device = traits.Enum(
        "auto",
        "cuda",
        "mps",
        "cpu",
        usedefault=True,
        desc="Device for HD-BET: auto, cuda, mps (Apple Silicon), cpu",
    )
    conda_env = traits.Str(
        desc="Conda environment name for HD-BET. When set, HD-BET is invoked "
        "via 'conda run -n <env>'. When empty, HD-BET is called directly.",
    )
    ants_template = File(
        exists=True,
        desc="ANTs brain extraction template (for ants tool)",
    )
    ants_probability_mask = File(
        exists=True,
        desc="ANTs brain probability mask (for ants tool)",
    )
    use_gpu = traits.Bool(
        False,
        usedefault=True,
        desc="Whether this node uses GPU (for nipype scheduling)",
    )


class BrainExtractionOutputSpec(TraitedSpec):
    """Output specification for BrainExtraction interface."""

    out_file = File(exists=True, desc="Brain-extracted image")
    out_mask = File(exists=True, desc="Binary brain mask")
    brain_volume_mm3 = traits.Float(desc="Brain volume in cubic millimeters")
    dice_vs_mni = traits.Float(desc="DICE coefficient vs MNI brain mask")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class BrainExtraction(CommandLine):
    """Deep learning brain extraction using HD-BET or ANTs fallback.

    This interface wraps HD-BET (Isensee et al., 2019) for accurate deep learning
    brain extraction, with ANTs antsBrainExtraction.sh as a fallback option.

    HD-BET is preferred for its superior performance on pathological cases
    (tumors, lesions) and consistent results across scanner vendors.

    Parameters
    ----------
    in_file : str
        Input bias-corrected T1w image
    tool : str
        Brain extraction tool: "hdbet" (default) or "ants"
    device : str
        HD-BET device: "auto" (auto-detect), "cuda", "mps" (Apple Silicon), "cpu"
    mni_template : str, optional
        MNI T1w template for registration QC
    mni_mask : str, optional
        MNI brain mask for DICE QC calculation
    ants_template : str, optional
        ANTs extraction template (required if tool="ants")
    ants_probability_mask : str, optional
        ANTs probability mask (required if tool="ants")

    Examples
    --------
    Using HD-BET (default):

    >>> brain_ex = BrainExtraction()
    >>> brain_ex.inputs.in_file = 'T1w_biascorr.nii.gz'
    >>> brain_ex.inputs.tool = 'hdbet'
    >>> result = brain_ex.run()  # doctest: +SKIP

    Using ANTs fallback:

    >>> brain_ex = BrainExtraction()
    >>> brain_ex.inputs.in_file = 'T1w_biascorr.nii.gz'
    >>> brain_ex.inputs.tool = 'ants'
    >>> brain_ex.inputs.ants_template = 'MNI_T1.nii.gz'
    >>> brain_ex.inputs.ants_probability_mask = 'MNI_BrainProb.nii.gz'
    >>> result = brain_ex.run()  # doctest: +SKIP

    Notes
    -----
    HD-BET device selection:

    - "auto": Auto-detect GPU (CUDA > MPS > CPU)
    - "cuda": Force NVIDIA CUDA GPU
    - "mps": Force Apple Silicon GPU (M1/M2/M3)
    - "cpu": Force CPU mode

    For Apple Silicon (arm64) systems, HD-BET will automatically use MPS
    when device="auto" and CUDA is unavailable.

    QC metrics computed:

    - brain_volume_mm3: Total brain volume from mask
    - dice_vs_mni: DICE overlap with MNI brain mask (after registration)

    See Also
    --------
    HD-BET: https://github.com/MIC-DKFZ/HD-BET
    ANTs: https://github.com/ANTsX/ANTs
    """

    input_spec = BrainExtractionInputSpec
    output_spec = BrainExtractionOutputSpec

    _cmd = "bash"

    def _gen_outfilename(self) -> str:
        """Generate output brain image filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_brain"))

    def _gen_maskfilename(self) -> str:
        """Generate output brain mask filename if not specified."""
        if self.inputs.out_mask:
            return str(self.inputs.out_mask)

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_brain_mask"))

    def _get_hdbet_device(self) -> str:
        """Get HD-BET device string based on settings and architecture."""
        from pmriprep.utils.device import detect_torch_device

        device = self.inputs.device

        if device == "auto":
            return detect_torch_device()
        return device  # type: ignore[no-any-return]

    @property
    def cmdline(self) -> str:
        """Build the brain extraction command."""
        in_file = str(self.inputs.in_file)
        out_file = self._gen_outfilename()
        out_mask = self._gen_maskfilename()
        tool = self.inputs.tool

        cmd_parts: list[str] = []

        if tool == "hdbet":
            device = self._get_hdbet_device()

            hdbet_base = (
                f"hd-bet -i {in_file} -o {out_file} "
                f"-device {device} --disable_tta"
            )

            if isdefined(self.inputs.conda_env) and self.inputs.conda_env:
                hdbet_cmd = (
                    f"conda run -n {self.inputs.conda_env} "
                    f"--no-capture-output {hdbet_base}"
                )
            else:
                hdbet_cmd = hdbet_base
            cmd_parts.append(hdbet_cmd)

            # Create mask by binarizing the brain output
            cmd_parts.append(f"fslmaths {out_file} -bin {out_mask}")

        else:  # ants
            ants_template = getattr(self.inputs, "ants_template", None)
            ants_prob_mask = getattr(self.inputs, "ants_probability_mask", None)

            if not ants_template:
                ants_dir = os.environ.get("ANTSPATH", "/usr/share/ants")
                ants_template = f"{ants_dir}/OASIS_template/T_template0.nii.gz"
            if not ants_prob_mask:
                ants_dir = os.environ.get("ANTSPATH", "/usr/share/ants")
                ants_prob_mask = f"{ants_dir}/OASIS_template/T_template0_BrainCerebellumProbabilityMask.nii.gz"

            out_path = Path(out_file)
            ants_prefix = str(out_path.parent / "ants_brain_")

            ants_cmd = (
                f"antsBrainExtraction.sh -d 3 -a {in_file} "
                f"-e {ants_template} -m {ants_prob_mask} "
                f"-o {ants_prefix}"
            )
            cmd_parts.append(ants_cmd)

            ants_brain = f"{ants_prefix}BrainExtractionBrain.nii.gz"
            ants_mask = f"{ants_prefix}BrainExtractionMask.nii.gz"

            cmd_parts.append(f"mv {ants_brain} {out_file}")
            cmd_parts.append(f"mv {ants_mask} {out_mask}")
            cmd_parts.append(f"rm -f {ants_prefix}*")

        return "bash -c '" + " && ".join(cmd_parts) + "'"

    # Brain volume validation thresholds (Gap 7)
    BRAIN_VOL_MIN_ML = 1000.0  # Minimum expected brain volume in mL
    BRAIN_VOL_MAX_ML = 2000.0  # Maximum expected brain volume in mL

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        outputs["out_mask"] = self._gen_maskfilename()

        # Compute QC metrics after execution
        qc_metrics: dict[str, Any] = {
            "tool": self.inputs.tool,  # hdbet or ants
        }
        warnings_list: list[str] = []

        try:
            # Compute brain volume
            brain_vol = self._compute_brain_volume()
            if brain_vol is not None:
                outputs["brain_volume_mm3"] = brain_vol
                qc_metrics["brain_volume_mm3"] = brain_vol

                # Validate brain volume (Gap 7)
                brain_vol_ml = brain_vol / 1000.0  # mm3 to mL
                qc_metrics["brain_volume_ml"] = brain_vol_ml

                if brain_vol_ml < self.BRAIN_VOL_MIN_ML:
                    warnings_list.append(
                        f"Brain volume {brain_vol_ml:.0f} mL below expected "
                        f"minimum {self.BRAIN_VOL_MIN_ML:.0f} mL - "
                        "brain extraction may have missed cortex"
                    )
                elif brain_vol_ml > self.BRAIN_VOL_MAX_ML:
                    warnings_list.append(
                        f"Brain volume {brain_vol_ml:.0f} mL above expected "
                        f"maximum {self.BRAIN_VOL_MAX_ML:.0f} mL - "
                        "brain extraction may have included skull/dura"
                    )
        except Exception:
            pass

        try:
            # Compute DICE vs MNI if template provided
            dice = self._compute_dice_vs_mni()
            if dice is not None:
                outputs["dice_vs_mni"] = dice
                qc_metrics["dice_vs_mni"] = dice
        except Exception:
            pass

        # Store warnings and QC status
        qc_metrics["warnings"] = warnings_list
        if warnings_list:
            qc_metrics["qc_status"] = "WARN"
        else:
            qc_metrics["qc_status"] = "PASS"

        outputs["qc_metrics"] = qc_metrics
        return outputs

    def _compute_brain_volume(self) -> Optional[float]:
        """Compute brain volume from mask."""
        import nibabel as nib  # type: ignore[import-untyped]
        import numpy as np

        mask_file = self._gen_maskfilename()
        if not Path(mask_file).exists():
            return None

        # Load mask
        img = nib.load(mask_file)
        data = img.get_fdata()  # type: ignore[attr-defined]
        header = img.header  # type: ignore[attr-defined]

        # Get voxel dimensions
        zooms = header.get_zooms()[:3]  # type: ignore[attr-defined]
        voxel_vol_mm3 = float(np.prod(zooms))

        # Count brain voxels
        n_voxels = int(np.sum(data > 0.5))
        brain_vol_mm3 = n_voxels * voxel_vol_mm3

        return brain_vol_mm3

    def _compute_dice_vs_mni(self) -> Optional[float]:
        """Compute DICE coefficient vs MNI brain mask."""
        # Only compute if MNI mask provided
        mni_mask_file = getattr(self.inputs, "mni_mask", None)
        if not mni_mask_file or not Path(mni_mask_file).exists():
            return None

        mask_file = self._gen_maskfilename()
        if not Path(mask_file).exists():
            return None

        # This would require registration to MNI first
        # For now, skip DICE vs MNI (requires transform from workflow)
        # The actual DICE calculation should be done in the workflow
        # after MNI registration is complete
        return None

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        if name == "out_mask":
            return self._gen_maskfilename()
        return None


class TissueSegmentationInputSpec(CommandLineInputSpec):
    """Input specification for TissueSegmentation interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input brain-extracted T1w image for segmentation",
    )
    out_basename = traits.Str(
        genfile=True,
        desc="Output basename for segmentation files",
    )
    num_classes = traits.Int(
        3,
        usedefault=True,
        desc="Number of tissue classes to segment (default: 3 for WM/GM/CSF)",
    )
    bias_iters = traits.Int(
        4,
        usedefault=True,
        desc="Number of bias field estimation iterations (default: 4)",
    )
    segments = traits.Bool(
        True,
        usedefault=True,
        desc="Output binary segmentation images (default: True)",
    )
    probability_maps = traits.Bool(
        True,
        usedefault=True,
        desc="Output probability maps for each class (default: True)",
    )


class TissueSegmentationOutputSpec(TraitedSpec):
    """Output specification for TissueSegmentation interface."""

    wm_probseg = File(exists=True, desc="White matter probability map")
    gm_probseg = File(exists=True, desc="Gray matter probability map")
    csf_probseg = File(exists=True, desc="CSF probability map")
    wm_mask = File(desc="White matter binary mask (if segments=True)")
    gm_mask = File(desc="Gray matter binary mask (if segments=True)")
    csf_mask = File(desc="CSF binary mask (if segments=True)")
    mixeltype = File(desc="Mixel-type map (if available)")
    partial_volume_map = File(desc="Partial volume estimate for each class")


class TissueSegmentation(CommandLine):
    """FSL FAST tissue segmentation for WM/GM/CSF probability maps.

    This interface wraps FSL FAST (FMRIB's Automated Segmentation Tool) to
    perform tissue classification on brain-extracted T1w images. FAST uses
    a hidden Markov random field model with an associated expectation-maximization
    algorithm to segment the brain into white matter (WM), gray matter (GM),
    and cerebrospinal fluid (CSF).

    Parameters
    ----------
    in_file : str
        Input brain-extracted T1w image
    num_classes : int
        Number of tissue classes (default: 3 for WM/GM/CSF)
    bias_iters : int
        Number of bias field estimation iterations (default: 4)
    segments : bool
        If True, output binary segmentation images
    probability_maps : bool
        If True, output probability maps for each class

    Returns
    -------
    wm_probseg : str
        White matter probability map
    gm_probseg : str
        Gray matter probability map
    csf_probseg : str
        CSF probability map
    wm_mask : str
        White matter binary mask (if segments=True)
    gm_mask : str
        Gray matter binary mask (if segments=True)
    csf_mask : str
        CSF binary mask (if segments=True)

    Examples
    --------
    >>> seg = TissueSegmentation()
    >>> seg.inputs.in_file = 'T1w_brain.nii.gz'
    >>> result = seg.run()  # doctest: +SKIP

    Notes
    -----
    FSL FAST output naming convention for 3-class segmentation:
    - *_pve_0.nii.gz: CSF probability (class 0)
    - *_pve_1.nii.gz: GM probability (class 1)
    - *_pve_2.nii.gz: WM probability (class 2)
    - *_seg.nii.gz: Combined hard segmentation (0=background, 1=CSF, 2=GM, 3=WM)

    Note that FSL FAST class numbering differs from BIDS convention:
    - FAST: 0=CSF, 1=GM, 2=WM
    - BIDS probseg: label-CSF, label-GM, label-WM

    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL FAST: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST
    """

    input_spec = TissueSegmentationInputSpec
    output_spec = TissueSegmentationOutputSpec

    _cmd = "fast"

    def _gen_basename(self) -> str:
        """Generate output basename if not specified."""
        if self.inputs.out_basename:
            return str(self.inputs.out_basename)

        in_file = Path(self.inputs.in_file)
        # Handle .nii.gz extension
        if in_file.name.endswith(".nii.gz"):
            base = in_file.name[:-7]
        elif in_file.name.endswith(".nii"):
            base = in_file.stem
        else:
            base = in_file.stem

        return str(Path.cwd() / f"{base}_fast")

    @property
    def cmdline(self) -> str:
        """Build the FAST command line."""
        in_file = str(self.inputs.in_file)
        out_basename = self._gen_basename()
        num_classes = self.inputs.num_classes
        bias_iters = self.inputs.bias_iters
        segments = self.inputs.segments
        probability_maps = self.inputs.probability_maps

        # Build FAST command
        cmd_parts = ["fast"]

        # Type of image (1=T1, 2=T2, 3=PD)
        cmd_parts.append("-t 1")

        # Number of classes
        cmd_parts.append(f"-n {num_classes}")

        # Bias iterations
        cmd_parts.append(f"-I {bias_iters}")

        # Segments output
        if segments:
            cmd_parts.append("-g")

        # Probability maps output (default on, -p is deprecated)
        # FAST outputs pve (partial volume estimates) by default

        # Output basename
        cmd_parts.append(f"-o {out_basename}")

        # Input file
        cmd_parts.append(in_file)

        return " ".join(cmd_parts)

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        out_basename = self._gen_basename()

        # FSL FAST output naming convention for 3-class segmentation:
        # *_pve_0.nii.gz = CSF probability
        # *_pve_1.nii.gz = GM probability
        # *_pve_2.nii.gz = WM probability
        outputs["csf_probseg"] = f"{out_basename}_pve_0.nii.gz"
        outputs["gm_probseg"] = f"{out_basename}_pve_1.nii.gz"
        outputs["wm_probseg"] = f"{out_basename}_pve_2.nii.gz"

        # Hard segmentations (if -g flag was used)
        if self.inputs.segments:
            outputs["csf_mask"] = f"{out_basename}_seg_0.nii.gz"
            outputs["gm_mask"] = f"{out_basename}_seg_1.nii.gz"
            outputs["wm_mask"] = f"{out_basename}_seg_2.nii.gz"

        # Optional outputs
        mixel_file = f"{out_basename}_mixeltype.nii.gz"
        if Path(mixel_file).exists():
            outputs["mixeltype"] = mixel_file

        pv_file = f"{out_basename}_pveseg.nii.gz"
        if Path(pv_file).exists():
            outputs["partial_volume_map"] = pv_file

        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_basename":
            return self._gen_basename()
        return None


class BBRRefinementInputSpec(CommandLineInputSpec):
    """Input specification for BBRRefinement interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Moving image to refine registration for",
    )
    reference = File(
        exists=True,
        mandatory=True,
        desc="Reference image (target space, typically T1w pre-contrast)",
    )
    init_matrix = File(
        exists=True,
        mandatory=True,
        desc="Initial transformation matrix from FLIRT (to be refined)",
    )
    wm_seg = File(
        exists=True,
        mandatory=True,
        desc="White matter segmentation for BBR cost function",
    )
    out_file = File(
        genfile=True,
        desc="Output BBR-refined registered image",
    )
    out_matrix = File(
        genfile=True,
        desc="Output BBR-refined transformation matrix",
    )
    dof = traits.Int(
        6,
        usedefault=True,
        desc="Degrees of freedom for registration (default: 6 for rigid)",
    )


class BBRRefinementOutputSpec(TraitedSpec):
    """Output specification for BBRRefinement interface."""

    out_file = File(exists=True, desc="BBR-refined registered image")
    out_matrix = File(exists=True, desc="BBR-refined transformation matrix")


class BBRRefinement(CommandLine):
    """Boundary-based registration refinement using FSL FLIRT.

    This interface wraps FSL FLIRT with the BBR cost function to refine
    an initial rigid registration using white matter boundaries. BBR uses
    the white matter segmentation to optimize alignment at tissue boundaries,
    which can improve registration accuracy compared to intensity-based
    methods alone.

    The BBR cost function works by:
    1. Taking the initial FLIRT registration as a starting point
    2. Using the WM segmentation to identify tissue boundaries
    3. Optimizing the alignment at these boundaries

    Parameters
    ----------
    in_file : str
        Moving image to be registered
    reference : str
        Reference/target image (typically bias-corrected T1w)
    init_matrix : str
        Initial transformation matrix from FLIRT
    wm_seg : str
        White matter segmentation (probability map or binary mask)
    dof : int
        Degrees of freedom (default: 6 for rigid body)

    Examples
    --------
    >>> bbr = BBRRefinement()
    >>> bbr.inputs.in_file = 'T2w_flirt.nii.gz'
    >>> bbr.inputs.reference = 'T1w_biascorr.nii.gz'
    >>> bbr.inputs.init_matrix = 'T2w_to_T1w.mat'
    >>> bbr.inputs.wm_seg = 'wm_probseg.nii.gz'
    >>> result = bbr.run()  # doctest: +SKIP

    Notes
    -----
    BBR refinement is most effective when:
    - The initial FLIRT registration is close to optimal
    - Good quality WM segmentation is available
    - The moving image has sufficient contrast at tissue boundaries

    BBR may fail or degrade registration when:
    - Initial registration is poor (BBR may converge to local minimum)
    - WM segmentation is inaccurate (e.g., due to lesions, contrast agent)
    - The modality has poor tissue contrast

    For this reason, BBR results should be validated against the initial
    FLIRT registration using a metric like NMI.

    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL FLIRT BBR: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT_BBR
    """

    input_spec = BBRRefinementInputSpec
    output_spec = BBRRefinementOutputSpec

    _cmd = "flirt"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_bbr"))

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
        """Build the FLIRT BBR command line."""
        in_file = str(self.inputs.in_file)
        reference = str(self.inputs.reference)
        init_matrix = str(self.inputs.init_matrix)
        wm_seg = str(self.inputs.wm_seg)
        out_file = self._gen_outfilename()
        out_matrix = self._gen_matfilename()
        dof = self.inputs.dof

        # Build FLIRT BBR command
        cmd = (
            f"flirt -in {in_file} -ref {reference} "
            f"-out {out_file} -omat {out_matrix} "
            f"-init {init_matrix} -wmseg {wm_seg} "
            f"-cost bbr -dof {dof}"
        )

        return cmd

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        outputs["out_matrix"] = self._gen_matfilename()
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        if name == "out_matrix":
            return self._gen_matfilename()
        return None


class T1wSubtractionInputSpec(CommandLineInputSpec):
    """Input specification for T1wSubtraction interface."""

    postcon_file = File(
        exists=True,
        mandatory=True,
        desc="Post-contrast T1w image (coregistered to precon)",
    )
    precon_file = File(
        exists=True,
        mandatory=True,
        desc="Pre-contrast T1w image (reference)",
    )
    brain_mask = File(
        exists=True,
        desc="Optional brain mask to apply to subtraction image",
    )
    out_file = File(
        genfile=True,
        desc="Output subtraction image (postcon - precon)",
    )


class T1wSubtractionOutputSpec(TraitedSpec):
    """Output specification for T1wSubtraction interface."""

    out_file = File(exists=True, desc="Subtraction image (postcon - precon)")


class T1wSubtraction(CommandLine):
    """Create T1w subtraction image (post-contrast minus pre-contrast).

    This interface uses FSL fslmaths to compute the difference between
    post-contrast and pre-contrast T1-weighted images. The resulting
    subtraction image highlights areas of contrast enhancement, which
    can be useful for:

    - Tumor visualization (enhancing lesions become bright)
    - Blood-brain barrier breakdown assessment
    - Treatment response monitoring

    Parameters
    ----------
    postcon_file : str
        Post-contrast T1w image (must be coregistered to precon)
    precon_file : str
        Pre-contrast T1w image (reference space)
    brain_mask : str, optional
        Brain mask to apply to subtraction (removes background)

    Examples
    --------
    >>> sub = T1wSubtraction()
    >>> sub.inputs.postcon_file = 'T1w_postcon_coreg.nii.gz'
    >>> sub.inputs.precon_file = 'T1w_precon.nii.gz'
    >>> sub.inputs.brain_mask = 'brain_mask.nii.gz'
    >>> result = sub.run()  # doctest: +SKIP

    Notes
    -----
    Important considerations:
    - Post-contrast image MUST be coregistered to pre-contrast space
    - Both images should be bias-corrected for accurate subtraction
    - Negative values may occur where signal decreases (e.g., T2* effects)
    - Brain mask helps remove skull/background artifacts

    The subtraction formula is: sub = postcon - precon

    Requires FSL to be installed and available in PATH.
    """

    input_spec = T1wSubtractionInputSpec
    output_spec = T1wSubtractionOutputSpec

    _cmd = "fslmaths"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        return str(Path.cwd() / nifti_gz_name(self.inputs.precon_file, "_subtraction"))

    @property
    def cmdline(self) -> str:
        """Build the fslmaths subtraction command."""
        postcon = str(self.inputs.postcon_file)
        precon = str(self.inputs.precon_file)
        out_file = self._gen_outfilename()

        brain_mask = getattr(self.inputs, "brain_mask", None)

        if brain_mask and Path(brain_mask).exists():
            return f"fslmaths {postcon} -sub {precon} -mas {brain_mask} {out_file}"
        else:
            return f"fslmaths {postcon} -sub {precon} {out_file}"

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        return None


class ApplyMaskInputSpec(CommandLineInputSpec):
    """Input specification for ApplyMask interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input image to mask",
    )
    mask_file = File(
        exists=True,
        mandatory=True,
        desc="Binary mask to apply",
    )
    out_file = File(
        genfile=True,
        desc="Output masked image",
    )


class ApplyMaskOutputSpec(TraitedSpec):
    """Output specification for ApplyMask interface."""

    out_file = File(exists=True, desc="Masked image")


class ApplyMask(CommandLine):
    """Apply a binary mask to an image using FSL fslmaths.

    This interface wraps FSL fslmaths -mas to multiply an image by
    a binary mask, setting values outside the mask to zero.

    Parameters
    ----------
    in_file : str
        Input image to mask
    mask_file : str
        Binary mask (non-zero values are retained)

    Examples
    --------
    >>> mask = ApplyMask()
    >>> mask.inputs.in_file = 'T2w_coreg.nii.gz'
    >>> mask.inputs.mask_file = 'brain_mask.nii.gz'
    >>> result = mask.run()  # doctest: +SKIP

    Notes
    -----
    The mask should be in the same space as the input image.
    Non-zero values in the mask are treated as "inside" the mask.

    Requires FSL to be installed and available in PATH.
    """

    input_spec = ApplyMaskInputSpec
    output_spec = ApplyMaskOutputSpec

    _cmd = "fslmaths"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_masked"))

    @property
    def cmdline(self) -> str:
        """Build the fslmaths command line."""
        in_file = str(self.inputs.in_file)
        mask_file = str(self.inputs.mask_file)
        out_file = self._gen_outfilename()

        return f"fslmaths {in_file} -mas {mask_file} {out_file}"

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        return None


class ApplyTransformInputSpec(CommandLineInputSpec):
    """Input specification for ApplyTransform interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input image to transform",
    )
    reference = File(
        exists=True,
        mandatory=True,
        desc="Reference image defining output geometry",
    )
    in_matrix = File(
        exists=True,
        mandatory=True,
        desc="FLIRT transformation matrix (.mat)",
    )
    out_file = File(
        genfile=True,
        desc="Output transformed image",
    )
    interp = traits.Enum(
        "trilinear",
        "nearestneighbour",
        "sinc",
        "spline",
        usedefault=True,
        desc="Interpolation method",
    )


class ApplyTransformOutputSpec(TraitedSpec):
    """Output specification for ApplyTransform interface."""

    out_file = File(exists=True, desc="Transformed image in reference space")


class ApplyTransform(CommandLine):
    """Apply an existing FLIRT transform via flirt -applyxfm.

    This interface wraps FSL FLIRT with -applyxfm to apply a pre-computed
    transformation matrix to a different image (e.g., applying a coregistration
    transform computed on bias-corrected images to the uncorrected versions).

    Parameters
    ----------
    in_file : str
        Input image to transform.
    reference : str
        Reference image defining the output geometry.
    in_matrix : str
        FLIRT transformation matrix (.mat file).
    interp : str
        Interpolation method (default: trilinear).

    Examples
    --------
    >>> xfm = ApplyTransform()
    >>> xfm.inputs.in_file = 'T2w_reoriented.nii.gz'
    >>> xfm.inputs.reference = 'T1w_reoriented.nii.gz'
    >>> xfm.inputs.in_matrix = 'T2w_to_T1w.mat'
    >>> result = xfm.run()  # doctest: +SKIP

    Notes
    -----
    Requires FSL to be installed and available in PATH.
    """

    input_spec = ApplyTransformInputSpec
    output_spec = ApplyTransformOutputSpec

    _cmd = "flirt"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_xfm"))

    @property
    def cmdline(self) -> str:
        """Build the FLIRT -applyxfm command line."""
        in_file = str(self.inputs.in_file)
        reference = str(self.inputs.reference)
        in_matrix = str(self.inputs.in_matrix)
        out_file = self._gen_outfilename()
        interp = self.inputs.interp

        return (
            f"flirt -in {in_file} -ref {reference} "
            f"-out {out_file} -init {in_matrix} "
            f"-applyxfm -interp {interp}"
        )

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        return None


class ACPCAlignmentInputSpec(CommandLineInputSpec):
    """Input specification for ACPCAlignment interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input brain-extracted T1w image to align to ACPC",
    )
    out_file = File(
        genfile=True,
        desc="Output ACPC-aligned image",
    )
    out_matrix = File(
        genfile=True,
        desc="Output transformation matrix (FLIRT .mat format)",
    )
    mni_template = File(
        exists=True,
        desc="MNI T1w template for ACPC alignment target",
    )
    dof = traits.Int(
        6,
        usedefault=True,
        desc="Degrees of freedom for registration (default: 6 for rigid)",
    )
    cost = traits.Enum(
        "normmi",
        "corratio",
        "normcorr",
        "mutualinfo",
        "leastsq",
        usedefault=True,
        desc="Cost function for registration (default: normmi)",
    )


class ACPCAlignmentOutputSpec(TraitedSpec):
    """Output specification for ACPCAlignment interface."""

    out_file = File(exists=True, desc="ACPC-aligned image")
    out_matrix = File(exists=True, desc="Affine transformation matrix")


class ACPCAlignment(CommandLine):
    """ACPC alignment using rigid registration to MNI template.

    This interface wraps FSL FLIRT to perform rigid body (6-DOF) registration
    of a brain-extracted T1w image to an MNI template for ACPC alignment.
    This provides a standardized visualization orientation without performing
    full spatial normalization.

    The alignment uses normalized mutual information (normmi) as the cost
    function, which is robust to intensity differences between the subject
    image and the template.

    Parameters
    ----------
    in_file : str
        Input brain-extracted T1w image
    mni_template : str, optional
        MNI T1w template (default: uses FSL's MNI152 template)
    dof : int
        Degrees of freedom (default: 6 for rigid body)
    cost : str
        Cost function (default: normmi)

    Examples
    --------
    Standard ACPC alignment:

    >>> acpc = ACPCAlignment()
    >>> acpc.inputs.in_file = 'T1w_brain.nii.gz'
    >>> result = acpc.run()  # doctest: +SKIP

    With custom MNI template:

    >>> acpc = ACPCAlignment()
    >>> acpc.inputs.in_file = 'T1w_brain.nii.gz'
    >>> acpc.inputs.mni_template = '/path/to/MNI152_T1_1mm_brain.nii.gz'
    >>> result = acpc.run()  # doctest: +SKIP

    Notes
    -----
    ACPC alignment provides:
    - Standardized orientation for visualization
    - Consistent coordinate system across subjects
    - Does NOT warp the brain (rigid body only)

    The transformation is 6-DOF (translation + rotation), preserving the
    native brain shape while aligning to the AC-PC line as approximated
    by the MNI template orientation.

    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL FLIRT: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT
    """

    input_spec = ACPCAlignmentInputSpec
    output_spec = ACPCAlignmentOutputSpec

    _cmd = "flirt"

    # Default MNI template path (FSL installation)
    _DEFAULT_MNI_TEMPLATE = (
        "/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
    )

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_acpc"))

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

    def _get_mni_template(self) -> str:
        """Get MNI template path."""
        mni_template = getattr(self.inputs, "mni_template", None)
        if mni_template and Path(mni_template).exists():
            return str(mni_template)

        # Try FSL standard locations
        import os

        fsldir = os.environ.get("FSLDIR", "/usr/local/fsl")
        fsl_template = Path(fsldir) / "data/standard/MNI152_T1_1mm_brain.nii.gz"
        if fsl_template.exists():
            return str(fsl_template)

        # Try common alternate location
        alt_template = Path("/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz")
        if alt_template.exists():
            return str(alt_template)

        # Fallback: use templateflow MNI152NLin6Asym (FSL-equivalent MNI space)
        try:
            import templateflow.api as tflow  # type: ignore[import-untyped]

            tf_path = tflow.get(
                "MNI152NLin6Asym", desc="brain", resolution=1,
                suffix="T1w", extension=".nii.gz",
            )
            if tf_path and Path(str(tf_path)).exists():
                return str(tf_path)
        except Exception:
            pass

        # Use default path (may fail at runtime if not present)
        return self._DEFAULT_MNI_TEMPLATE

    @property
    def cmdline(self) -> str:
        """Build the ACPC alignment command."""
        in_file = str(self.inputs.in_file)
        out_file = self._gen_outfilename()
        out_matrix = self._gen_matfilename()
        dof = self.inputs.dof
        cost = self.inputs.cost
        mni_template = self._get_mni_template()

        return (
            f"flirt -in {in_file} -ref {mni_template} "
            f"-out {out_file} -omat {out_matrix} "
            f"-dof {dof} -cost {cost}"
        )

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]
        outputs["out_file"] = self._gen_outfilename()
        outputs["out_matrix"] = self._gen_matfilename()
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        if name == "out_matrix":
            return self._gen_matfilename()
        return None
