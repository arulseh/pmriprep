"""Common interfaces shared across workflows.

This module contains interfaces that are used by multiple workflows,
such as reorientation to standard RAS orientation.
"""

from pathlib import Path
from typing import Any, Optional

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
)


def reorient_list_to_ras(in_files=None):
    """Reorient a list of NIfTI files to RAS orientation with nibabel.

    IMPORTANT: This function must be self-contained for nipype Function nodes.
    All imports must be inside the function.

    Parameters
    ----------
    in_files : list or None
        List of paths to input NIfTI files, or None.

    Returns
    -------
    list or None
        List of paths to reoriented NIfTI files, or None if input was None.
    """
    if in_files is None:
        return None

    if not in_files:
        return []

    from pathlib import Path as _Path

    def _reorient_single(in_file):
        """Reorient a single file (inlined for nipype compatibility)."""
        if in_file is None:
            return None

        in_path = _Path(in_file)
        if not in_path.exists():
            return None

        import nibabel as _nib

        from pmriprep.utils.nifti import nifti_gz_name, reorient_img_to_ras

        out_file = str(_Path.cwd() / nifti_gz_name(in_file, "_reoriented"))
        _nib.save(reorient_img_to_ras(_nib.load(str(in_file))), out_file)
        return out_file

    return [_reorient_single(f) for f in in_files]


class ReorientInputSpec(BaseInterfaceInputSpec):
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


class Reorient(SimpleInterface):
    """Reorient a NIfTI to canonical RAS orientation with nibabel.

    Reorders the data array and affine together so the voxel axes run R, A, S,
    preserving physical positions. Handles both 3D structural and 4D timeseries
    data (only the first three axes are permuted/flipped). Unlike
    ``ReorientAndCrop``, this does NOT apply robustfov (which would incorrectly
    crop 4D data or non-structural images like perfusion data).

    Use this for: DCE, DSC, VFA, AFI
    Use ReorientAndCrop for: T1w, T2w, FLAIR (structurals)

    Examples
    --------
    >>> reorient = Reorient()
    >>> reorient.inputs.in_file = 'dce.nii.gz'
    >>> result = reorient.run()  # doctest: +SKIP

    Notes
    -----
    Pure nibabel; no external tools are required.
    """

    input_spec = ReorientInputSpec
    output_spec = ReorientOutputSpec

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if isdefined(self.inputs.out_file) and self.inputs.out_file:
            return str(self.inputs.out_file)

        from pmriprep.utils.nifti import nifti_gz_name

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_reoriented"))

    def _run_interface(self, runtime: Any) -> Any:
        """Reorient the input to RAS and write the output (always writes)."""
        import nibabel as nib

        from pmriprep.utils.nifti import reorient_img_to_ras

        out_file = self._gen_outfilename()
        img = nib.load(str(self.inputs.in_file))
        nib.save(reorient_img_to_ras(img), out_file)
        self._results["out_file"] = out_file
        return runtime

    def _gen_filename(self, name: str) -> Optional[str]:
        """Generate filename for outputs."""
        if name == "out_file":
            return self._gen_outfilename()
        return None
