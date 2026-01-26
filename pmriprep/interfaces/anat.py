"""Anatomical preprocessing interfaces for PMRIPrep.

This module implements Nipype interfaces for anatomical preprocessing steps
including reorientation, bias correction, coregistration, brain extraction,
tissue segmentation, and ACPC alignment.
"""

from pathlib import Path
from typing import Any, Optional

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


class ReorientAndCropInputSpec(CommandLineInputSpec):
    """Input specification for ReorientAndCrop interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=-2,
        desc="Input NIfTI image to reorient and crop",
    )
    out_file = File(
        argstr="%s",
        position=-1,
        genfile=True,
        desc="Output reoriented and cropped image",
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

    # We'll use bash to chain the commands
    _cmd = "bash -c"

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

        return str(Path.cwd() / f"{base}_reoriented{ext}")

    def _format_arg(
        self, name: str, trait_spec: Any, value: Any
    ) -> str:
        """Format command line arguments."""
        if name == "in_file":
            # Build the full bash command string
            in_file = str(value)
            out_file = self._gen_outfilename()

            # Create intermediate filename for reorient step
            in_path = Path(in_file)
            if in_path.name.endswith(".nii.gz"):
                ext = ".nii.gz"
            else:
                ext = ".nii"
            intermediate = str(Path(out_file).parent / f"_tmp_reoriented{ext}")

            # Chain: fslreorient2std -> robustfov
            cmd = (
                f"'fslreorient2std {in_file} {intermediate} && "
                f"robustfov -i {intermediate} -r {out_file} && "
                f"rm -f {intermediate}'"
            )
            return cmd
        result: str = super()._format_arg(name, trait_spec, value)
        return result

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


class BiasCorrectionInputSpec(CommandLineInputSpec):
    """Input specification for BiasCorrection interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=-2,
        desc="Input NIfTI image to bias correct",
    )
    out_file = File(
        argstr="%s",
        position=-1,
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

    # Use bash to chain commands
    _cmd = "bash -c"

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

    def _format_arg(
        self, name: str, trait_spec: Any, value: Any
    ) -> str:
        """Format command line arguments."""
        if name == "in_file":
            # Build the full bash command string
            in_file = str(value)
            out_file = self._gen_outfilename()
            denoise = self.inputs.denoise
            shrink = self.inputs.shrink_factor
            convergence = self.inputs.convergence
            bspline = self.inputs.bspline_fitting
            noise_model = self.inputs.noise_model

            # Determine file extension
            in_path = Path(in_file)
            if in_path.name.endswith(".nii.gz"):
                ext = ".nii.gz"
            else:
                ext = ".nii"

            cmd_parts = []

            if denoise:
                # Denoise first, then bias correct
                denoised_file = str(Path(out_file).parent / f"_tmp_denoised{ext}")

                # DenoiseImage command
                denoise_cmd = (
                    f"DenoiseImage -d 3 -i {in_file} -o {denoised_file} "
                    f"-n {noise_model}"
                )
                cmd_parts.append(denoise_cmd)

                # N4 on denoised image
                n4_input = denoised_file
            else:
                n4_input = in_file
                denoised_file = None

            # N4BiasFieldCorrection command
            n4_cmd = (
                f"N4BiasFieldCorrection -d 3 -i {n4_input} -o {out_file} "
                f"-s {shrink} -c {convergence} -b {bspline}"
            )
            cmd_parts.append(n4_cmd)

            # Cleanup intermediate file if denoising was applied
            if denoised_file:
                cmd_parts.append(f"rm -f {denoised_file}")

            # Join with && for sequential execution
            cmd = "'" + " && ".join(cmd_parts) + "'"
            return cmd

        result: str = super()._format_arg(name, trait_spec, value)
        return result

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
    lesion_mask = File(
        exists=True,
        desc="Optional lesion mask for cost function weighting (excludes lesion region)",
    )
    modality = traits.Enum(
        "T1_postcon",
        "FLAIR",
        "T2w",
        "PDw",
        "T1w",
        usedefault=True,
        desc="Modality of moving image (determines cost function)",
    )
    dof = traits.Int(
        6,
        usedefault=True,
        desc="Degrees of freedom for registration (default: 6 for rigid)",
    )
    use_template_t2 = traits.Bool(
        False,
        usedefault=True,
        desc="Use MNI T2 template for T2w registration when PDw unavailable",
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
    - FLAIR, PDw, T2w: normcorr (normalized correlation)
    - T1w: corratio (correlation ratio)

    Optionally supports lesion mask cost function weighting, where the lesion
    region is excluded from the cost function calculation to prevent tumor
    distorting the registration.

    Parameters
    ----------
    in_file : str
        Moving image to be registered
    reference : str
        Reference/target image (typically bias-corrected T1w)
    lesion_mask : str, optional
        Binary lesion mask - lesion voxels will have reduced weight
    modality : str
        Modality of moving image: T1_postcon, FLAIR, T2w, PDw, T1w
    dof : int
        Degrees of freedom (default: 6 for rigid body)
    use_template_t2 : bool
        If True and modality is T2w, use MNI T2 template as intermediate

    Examples
    --------
    >>> coreg = Coregistration()
    >>> coreg.inputs.in_file = 'FLAIR.nii.gz'
    >>> coreg.inputs.reference = 'T1w_biascorr.nii.gz'
    >>> coreg.inputs.modality = 'FLAIR'
    >>> result = coreg.run()  # doctest: +SKIP

    With lesion mask:
    >>> coreg = Coregistration()
    >>> coreg.inputs.in_file = 'T1w_ce.nii.gz'
    >>> coreg.inputs.reference = 'T1w_biascorr.nii.gz'
    >>> coreg.inputs.modality = 'T1_postcon'
    >>> coreg.inputs.lesion_mask = 'lesion.nii.gz'
    >>> result = coreg.run()  # doctest: +SKIP

    Notes
    -----
    Cost function selection by modality:

    - normmi: Best for T1 post-contrast where intensity changes due to Gd
    - normcorr: Best for FLAIR/T2w/PDw where contrast inverts vs T1
    - corratio: Best for same-modality registration (T1w to T1w)

    Lesion mask weighting creates a cost function weight image where:
    weight = 1.0 - lesion_mask (i.e., lesion voxels have weight 0)

    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL FLIRT: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT
    """

    input_spec = CoregistrationInputSpec
    output_spec = CoregistrationOutputSpec

    _cmd = "bash -c"

    # Cost function mapping by modality
    _COST_FUNCTIONS: dict[str, str] = {
        "T1_postcon": "normmi",
        "FLAIR": "normcorr",
        "T2w": "normcorr",
        "PDw": "normcorr",
        "T1w": "corratio",
    }

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        in_file = Path(self.inputs.in_file)
        if in_file.name.endswith(".nii.gz"):
            base = in_file.name[:-7]
            ext = ".nii.gz"
        elif in_file.name.endswith(".nii"):
            base = in_file.stem
            ext = ".nii"
        else:
            base = in_file.stem
            ext = in_file.suffix

        return str(Path.cwd() / f"{base}_coreg{ext}")

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

    def _format_arg(
        self, name: str, trait_spec: Any, value: Any
    ) -> str:
        """Format command line arguments."""
        if name == "in_file":
            in_file = str(value)
            reference = str(self.inputs.reference)
            out_file = self._gen_outfilename()
            out_matrix = self._gen_matfilename()
            modality = self.inputs.modality
            dof = self.inputs.dof

            # Get cost function for modality
            cost = self._COST_FUNCTIONS.get(modality, "corratio")

            cmd_parts = []

            # Check if we have a lesion mask for cost function weighting
            lesion_mask = getattr(self.inputs, "lesion_mask", None)
            if lesion_mask and Path(lesion_mask).exists():
                # Create weight image: weight = 1 - lesion_mask
                out_path = Path(out_file)
                ext = ".nii.gz" if out_path.name.endswith(".nii.gz") else ".nii"
                weight_file = str(out_path.parent / f"_tmp_weight{ext}")

                # fslmaths to create weight: 1 - lesion_mask
                invert_cmd = f"fslmaths {lesion_mask} -mul -1 -add 1 {weight_file}"
                cmd_parts.append(invert_cmd)

                # FLIRT with cost function weighting
                flirt_cmd = (
                    f"flirt -in {in_file} -ref {reference} "
                    f"-out {out_file} -omat {out_matrix} "
                    f"-dof {dof} -cost {cost} "
                    f"-inweight {weight_file} -refweight {weight_file}"
                )
                cmd_parts.append(flirt_cmd)

                # Clean up weight file
                cmd_parts.append(f"rm -f {weight_file}")
            else:
                # Standard FLIRT without weighting
                flirt_cmd = (
                    f"flirt -in {in_file} -ref {reference} "
                    f"-out {out_file} -omat {out_matrix} "
                    f"-dof {dof} -cost {cost}"
                )
                cmd_parts.append(flirt_cmd)

            cmd = "'" + " && ".join(cmd_parts) + "'"
            return cmd

        result: str = super()._format_arg(name, trait_spec, value)
        return result

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
