"""Common interfaces shared across workflows.

This module contains interfaces that are used by multiple workflows,
such as reorientation to standard RAS orientation.
"""

from pathlib import Path
from typing import Any, Optional

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    isdefined,
)


def reorient_to_ras(in_file=None):
    """Reorient NIfTI to RAS orientation using fslreorient2std.

    This function handles None/undefined inputs gracefully, making it
    suitable for use with optional inputs in nipype Function nodes.

    IMPORTANT: This function must be self-contained for nipype Function nodes.
    All imports must be inside the function.

    Args:
        in_file: Path to input NIfTI file, or None if not provided.

    Returns:
        Path to reoriented NIfTI file, or None if input was None.
    """
    if in_file is None:
        return None

    import subprocess
    from pathlib import Path as _Path

    in_path = _Path(in_file)
    if not in_path.exists():
        return None

    # Generate output filename
    if in_path.name.endswith(".nii.gz"):
        base = in_path.name[:-7]
        ext = ".nii.gz"
    elif in_path.name.endswith(".nii"):
        base = in_path.stem
        ext = ".nii"
    else:
        base = in_path.stem
        ext = in_path.suffix

    out_file = str(_Path.cwd() / f"{base}_reoriented{ext}")

    # Run fslreorient2std
    cmd = ["fslreorient2std", str(in_file), out_file]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        import warnings
        warnings.warn(f"fslreorient2std failed for {in_file}: {e.stderr.decode()}")
        return None

    return out_file


def reorient_list_to_ras(in_files=None):
    """Reorient a list of NIfTI files to RAS orientation.

    IMPORTANT: This function must be self-contained for nipype Function nodes.
    All imports must be inside the function.

    Args:
        in_files: List of paths to input NIfTI files, or None.

    Returns:
        List of paths to reoriented NIfTI files, or None if input was None.
    """
    if in_files is None:
        return None

    if not in_files:
        return []

    import subprocess
    from pathlib import Path as _Path

    def _reorient_single(in_file):
        """Reorient a single file (inlined for nipype compatibility)."""
        if in_file is None:
            return None

        in_path = _Path(in_file)
        if not in_path.exists():
            return None

        # Generate output filename
        if in_path.name.endswith(".nii.gz"):
            base = in_path.name[:-7]
            ext = ".nii.gz"
        elif in_path.name.endswith(".nii"):
            base = in_path.stem
            ext = ".nii"
        else:
            base = in_path.stem
            ext = in_path.suffix

        out_file = str(_Path.cwd() / f"{base}_reoriented{ext}")

        # Run fslreorient2std
        cmd = ["fslreorient2std", str(in_file), out_file]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            import warnings
            warnings.warn(f"fslreorient2std failed for {in_file}: {e.stderr.decode()}")
            return None

        return out_file

    return [_reorient_single(f) for f in in_files]


class ReorientInputSpec(CommandLineInputSpec):
    """Input specification for Reorient interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input NIfTI image (3D or 4D) to reorient",
    )
    out_file = File(
        genfile=True,
        desc="Output reoriented NIfTI image",
    )


class ReorientOutputSpec(TraitedSpec):
    """Output specification for Reorient interface."""

    out_file = File(exists=True, desc="Reoriented NIfTI in RAS orientation")


class Reorient(CommandLine):
    """Reorient NIfTI to standard RAS orientation using FSL fslreorient2std.

    Handles both 3D structural and 4D timeseries data. Unlike ReorientAndCrop,
    this does NOT apply robustfov (which would incorrectly crop 4D data or
    non-structural images like perfusion data).

    Use this for: DCE, DSC, VFA, AFI, DWI
    Use ReorientAndCrop for: T1w, T2w, FLAIR (structurals)

    Examples
    --------
    >>> reorient = Reorient()
    >>> reorient.inputs.in_file = 'dce.nii.gz'
    >>> result = reorient.run()  # doctest: +SKIP

    Notes
    -----
    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL fslreorient2std: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation
    """

    input_spec = ReorientInputSpec
    output_spec = ReorientOutputSpec
    _cmd = "fslreorient2std"

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if isdefined(self.inputs.out_file) and self.inputs.out_file:
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

    @property
    def cmdline(self) -> str:
        """Build the fslreorient2std command."""
        return f"fslreorient2std {self.inputs.in_file} {self._gen_outfilename()}"

    def _list_outputs(self) -> dict[str, Any]:
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()
        outputs["out_file"] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        return None
