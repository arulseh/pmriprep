"""HD-GLIO tumor segmentation interfaces for PMRIPrep.

This module implements Nipype interfaces for HD-GLIO deep learning tumor
segmentation preprocessing. HD-GLIO requires minimal preprocessing:
brain extraction only, NO bias correction, NO intensity normalization.
"""

from pathlib import Path
from typing import Any, Optional

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)


class HDGLIOPrepInputSpec(BaseInterfaceInputSpec):
    """Input specification for HDGLIOPrep interface."""

    t1w_pre = File(
        exists=True,
        mandatory=True,
        desc="T1-weighted pre-contrast image",
    )
    t1w_postcon = File(
        exists=True,
        mandatory=True,
        desc="T1-weighted post-contrast (ce-GAD) image",
    )
    t2w = File(
        exists=True,
        mandatory=True,
        desc="T2-weighted image",
    )
    flair = File(
        exists=True,
        mandatory=True,
        desc="FLAIR image",
    )
    brain_mask = File(
        exists=True,
        mandatory=True,
        desc="Brain mask from anatomical preprocessing (for brain extraction)",
    )
    output_dir = traits.Directory(
        desc="Output directory for prepped images (default: current directory)",
    )
    check_dimensions = traits.Bool(
        True,
        usedefault=True,
        desc="Verify all images have matching dimensions (default: True)",
    )


class HDGLIOPrepOutputSpec(TraitedSpec):
    """Output specification for HDGLIOPrep interface."""

    t1w_pre_prepped = File(exists=True, desc="Brain-extracted T1w pre-contrast image")
    t1w_postcon_prepped = File(exists=True, desc="Brain-extracted T1w post-contrast image")
    t2w_prepped = File(exists=True, desc="Brain-extracted T2w image")
    flair_prepped = File(exists=True, desc="Brain-extracted FLAIR image")
    dimensions_matched = traits.Bool(desc="Whether all image dimensions matched")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class HDGLIOPrep(BaseInterface):
    """Minimal preprocessing for HD-GLIO tumor segmentation.

    This interface prepares input images for HD-GLIO deep learning segmentation.
    HD-GLIO requires minimal preprocessing to preserve image characteristics
    that the neural network was trained on:

    - Brain extraction: Applied using provided brain mask
    - NO bias correction: HD-GLIO is robust to B1 inhomogeneity
    - NO intensity normalization: Neural network handles normalization internally

    All 4 modalities (T1w-pre, T1w-ce-GAD, T2w, FLAIR) must be present and
    have matching dimensions for HD-GLIO inference.

    Parameters
    ----------
    t1w_pre : str
        Path to T1-weighted pre-contrast image
    t1w_postcon : str
        Path to T1-weighted post-contrast (ce-GAD) image
    t2w : str
        Path to T2-weighted image
    flair : str
        Path to FLAIR image
    brain_mask : str
        Path to brain mask from anatomical preprocessing
    check_dimensions : bool
        If True (default), verify all images have matching dimensions

    Returns
    -------
    t1w_pre_prepped : str
        Path to brain-extracted T1w pre-contrast image
    t1w_postcon_prepped : str
        Path to brain-extracted T1w post-contrast image
    t2w_prepped : str
        Path to brain-extracted T2w image
    flair_prepped : str
        Path to brain-extracted FLAIR image
    dimensions_matched : bool
        Whether all image dimensions matched
    qc_metrics : dict
        QC metrics including dimension check results

    Examples
    --------
    >>> prep = HDGLIOPrep()
    >>> prep.inputs.t1w_pre = 'T1w_pre.nii.gz'
    >>> prep.inputs.t1w_postcon = 'T1w_ce-GAD.nii.gz'
    >>> prep.inputs.t2w = 'T2w.nii.gz'
    >>> prep.inputs.flair = 'FLAIR.nii.gz'
    >>> prep.inputs.brain_mask = 'brain_mask.nii.gz'
    >>> result = prep.run()  # doctest: +SKIP

    Notes
    -----
    HD-GLIO (High-Definition GlioMa) requirements:

    1. **All 4 modalities required**: T1w-pre, T1w-ce-GAD, T2w, FLAIR
    2. **Brain extraction**: Images should be skull-stripped
    3. **NO bias correction**: The network is trained on uncorrected images
    4. **NO intensity normalization**: Internal normalization handles this
    5. **Matching dimensions**: All images must have same shape

    The brain mask is applied via multiplication to preserve original
    voxel intensities within the brain region. Voxels outside the brain
    mask are set to zero.

    This interface only prepares images; actual segmentation is performed
    by the HDGLIOSegment interface.

    See Also
    --------
    HDGLIOSegment : HD-GLIO deep learning segmentation interface
    HD-GLIO: https://github.com/MIC-DKFZ/HD-GLIO
    """

    input_spec = HDGLIOPrepInputSpec
    output_spec = HDGLIOPrepOutputSpec

    def _get_output_dir(self) -> Path:
        """Get output directory for prepped images."""
        if self.inputs.output_dir:
            return Path(self.inputs.output_dir)
        return Path.cwd()

    def _gen_output_filename(self, input_file: str, suffix: str = "_hdglio") -> str:
        """Generate output filename for prepped image."""
        in_path = Path(input_file)
        out_dir = self._get_output_dir()

        # Handle .nii.gz extension
        if in_path.name.endswith(".nii.gz"):
            base = in_path.name[:-7]
            ext = ".nii.gz"
        elif in_path.name.endswith(".nii"):
            base = in_path.stem
            ext = ".nii"
        else:
            base = in_path.stem
            ext = in_path.suffix

        return str(out_dir / f"{base}{suffix}{ext}")

    def _check_dimensions(
        self,
        images: dict[str, str],
    ) -> tuple[bool, dict[str, Any]]:
        """Check if all images have matching dimensions.

        Parameters
        ----------
        images : dict
            Dictionary mapping modality name to file path

        Returns
        -------
        matched : bool
            True if all dimensions match
        details : dict
            Dictionary with dimension details for each image
        """
        import nibabel as nib  # type: ignore[import-untyped]
        import numpy as np

        details: dict[str, Any] = {}
        shapes: list[tuple[int, ...]] = []
        affines: list[Any] = []

        for name, filepath in images.items():
            img = nib.load(filepath)
            shape = img.shape[:3]  # type: ignore[attr-defined]
            affine = img.affine  # type: ignore[attr-defined]
            zooms = img.header.get_zooms()[:3]  # type: ignore[attr-defined]

            details[name] = {
                "shape": shape,
                "zooms": tuple(float(z) for z in zooms),
                "file": filepath,
            }
            shapes.append(shape)
            affines.append(affine)

        # Check if all shapes match
        reference_shape = shapes[0]
        shapes_match = all(s == reference_shape for s in shapes)

        # Check if affines are similar (within tolerance for floating point)
        reference_affine = affines[0]
        affines_match = all(
            np.allclose(a, reference_affine, atol=1e-4) for a in affines
        )

        matched = shapes_match and affines_match
        details["shapes_match"] = shapes_match
        details["affines_match"] = affines_match
        details["all_matched"] = matched

        return matched, details

    def _apply_brain_mask(
        self,
        input_file: str,
        mask_file: str,
        output_file: str,
    ) -> None:
        """Apply brain mask to image (multiply image by mask).

        Parameters
        ----------
        input_file : str
            Path to input image
        mask_file : str
            Path to brain mask
        output_file : str
            Path to output brain-extracted image
        """
        import nibabel as nib  # type: ignore[import-untyped]
        import numpy as np

        # Load image and mask
        img = nib.load(input_file)
        mask_img = nib.load(mask_file)

        data = img.get_fdata()  # type: ignore[attr-defined]
        mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]

        # Binarize mask (in case it's a probability map)
        mask_binary = mask_data > 0.5

        # Apply mask
        masked_data = data * mask_binary

        # Save output
        out_img = nib.Nifti1Image(  # type: ignore[attr-defined]
            masked_data.astype(np.float32),
            img.affine,  # type: ignore[attr-defined]
            img.header,  # type: ignore[attr-defined]
        )
        nib.save(out_img, output_file)

    def _run_interface(self, runtime: Any) -> Any:
        """Execute the interface."""
        import nibabel as nib  # type: ignore[import-untyped]

        # Collect input images
        images = {
            "t1w_pre": str(self.inputs.t1w_pre),
            "t1w_postcon": str(self.inputs.t1w_postcon),
            "t2w": str(self.inputs.t2w),
            "flair": str(self.inputs.flair),
        }
        mask_file = str(self.inputs.brain_mask)

        # Check dimensions if requested
        qc_metrics: dict[str, Any] = {}
        dimensions_matched = True

        if self.inputs.check_dimensions:
            # Add mask to dimension check
            images_with_mask = {**images, "brain_mask": mask_file}
            dimensions_matched, dim_details = self._check_dimensions(images_with_mask)
            qc_metrics["dimensions"] = dim_details

            if not dimensions_matched:
                raise ValueError(
                    f"Image dimensions do not match. Check QC metrics for details: {dim_details}"
                )

        # Create output directory if needed
        out_dir = self._get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Apply brain mask to each image (NO bias correction, NO normalization)
        output_files: dict[str, str] = {}

        for modality, input_file in images.items():
            output_file = self._gen_output_filename(input_file, f"_{modality}_hdglio")
            self._apply_brain_mask(input_file, mask_file, output_file)
            output_files[modality] = output_file

        # Store results for _list_outputs
        self._results = {
            "t1w_pre_prepped": output_files["t1w_pre"],
            "t1w_postcon_prepped": output_files["t1w_postcon"],
            "t2w_prepped": output_files["t2w"],
            "flair_prepped": output_files["flair"],
            "dimensions_matched": dimensions_matched,
            "qc_metrics": qc_metrics,
        }

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)
        else:
            # Generate expected output filenames before run
            outputs["t1w_pre_prepped"] = self._gen_output_filename(
                str(self.inputs.t1w_pre), "_t1w_pre_hdglio"
            )
            outputs["t1w_postcon_prepped"] = self._gen_output_filename(
                str(self.inputs.t1w_postcon), "_t1w_postcon_hdglio"
            )
            outputs["t2w_prepped"] = self._gen_output_filename(
                str(self.inputs.t2w), "_t2w_hdglio"
            )
            outputs["flair_prepped"] = self._gen_output_filename(
                str(self.inputs.flair), "_flair_hdglio"
            )
            outputs["dimensions_matched"] = True
            outputs["qc_metrics"] = {}

        return outputs
