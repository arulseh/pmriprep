"""Common interfaces shared across workflows.

This module contains interfaces that are used by multiple workflows,
such as reorientation to standard RAS orientation.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

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

    Parameters
    ----------
    in_file : str or None
        Path to input NIfTI file, or None if not provided.

    Returns
    -------
    str or None
        Path to reoriented NIfTI file, or None if input was None.
    """
    if in_file is None:
        return None

    import subprocess
    from pathlib import Path as _Path

    in_path = _Path(in_file)
    if not in_path.exists():
        return None

    # Generate output filename (always .nii.gz — FSLOUTPUTTYPE=NIFTI_GZ)
    from pmriprep.utils.nifti import nifti_gz_name

    out_file = str(_Path.cwd() / nifti_gz_name(in_file, "_reoriented"))

    # Run fslreorient2std
    cmd = ["fslreorient2std", str(in_file), out_file]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.warning("fslreorient2std failed for %s: %s", in_file, e.stderr.decode())
        return None

    # Enforce true RAS orientation (fslreorient2std only reorders axes,
    # doesn't flip handedness — e.g. LAS stays LAS)
    import nibabel as _nib
    from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
    import numpy as _np

    _img = _nib.load(str(out_file))
    _ornt = io_orientation(_img.affine)
    _ras_ornt = axcodes2ornt(("R", "A", "S"))
    if not _np.array_equal(_ornt, _ras_ornt):
        _transform = ornt_transform(_ornt, _ras_ornt)
        _img_ras = _img.as_reoriented(_transform)
        _nib.save(_img_ras, str(out_file))

    return out_file


def reorient_list_to_ras(in_files=None):
    """Reorient a list of NIfTI files to RAS orientation.

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
        from pmriprep.utils.nifti import nifti_gz_name

        out_file = str(_Path.cwd() / nifti_gz_name(in_file, "_reoriented"))

        # Run fslreorient2std
        cmd = ["fslreorient2std", str(in_file), out_file]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            import warnings
            warnings.warn(f"fslreorient2std failed for {in_file}: {e.stderr.decode()}")
            return None

        # Enforce true RAS orientation (fslreorient2std only reorders axes,
        # doesn't flip handedness — e.g. LAS stays LAS)
        import nibabel as _nib
        from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
        import numpy as _np

        _img = _nib.load(str(out_file))
        _ornt = io_orientation(_img.affine)
        _ras_ornt = axcodes2ornt(("R", "A", "S"))
        if not _np.array_equal(_ornt, _ras_ornt):
            _transform = ornt_transform(_ornt, _ras_ornt)
            _img_ras = _img.as_reoriented(_transform)
            _nib.save(_img_ras, str(out_file))

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

    Use this for: DCE, DSC, VFA, AFI
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

        from pmriprep.utils.nifti import nifti_gz_name

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_reoriented"))

    @property
    def cmdline(self) -> str:
        """Build the fslreorient2std command."""
        return f"fslreorient2std {self.inputs.in_file} {self._gen_outfilename()}"

    def _list_outputs(self) -> dict[str, Any]:
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()
        outputs["out_file"] = self._gen_outfilename()
        return outputs

    def _post_run_hook(self, runtime: Any) -> Any:
        """Enforce true RAS after fslreorient2std completes.

        fslreorient2std only reorders axes but doesn't flip handedness.
        For example, LAS (negative x determinant) stays LAS after
        fslreorient2std. This post-run hook applies nibabel's
        ``as_reoriented()`` to ensure the output is truly RAS.
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
