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
