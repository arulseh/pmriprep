"""Mask manipulation utilities for PMRIPrep.

Provides Nipype SimpleInterface subclasses for mask merging,
used to combine user-supplied masks with tumor segmentation masks.
"""

from pathlib import Path
from typing import Any

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _MergeMasksInputSpec(BaseInterfaceInputSpec):
    """Input specification for MergeMasks."""

    mask_a = File(exists=True, desc="First mask file (optional)")
    mask_b = File(exists=True, desc="Second mask file (optional)")


class _MergeMasksOutputSpec(TraitedSpec):
    """Output specification for MergeMasks."""

    merged_mask = traits.Either(
        File(exists=True), None, desc="Merged mask (voxel-wise OR)"
    )


class MergeMasks(SimpleInterface):
    """Merge two binary masks via voxel-wise OR.

    If geometries differ, ``mask_b`` is resampled to ``mask_a`` space
    using nearest-neighbour interpolation. If only one mask is provided,
    it is passed through. If neither is provided, output is None.
    """

    input_spec = _MergeMasksInputSpec
    output_spec = _MergeMasksOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        from nipype.interfaces.base import isdefined

        has_a = isdefined(self.inputs.mask_a) and self.inputs.mask_a is not None
        has_b = isdefined(self.inputs.mask_b) and self.inputs.mask_b is not None

        if not has_a and not has_b:
            self._results["merged_mask"] = None
            return runtime

        if has_a and not has_b:
            self._results["merged_mask"] = self.inputs.mask_a
            return runtime

        if has_b and not has_a:
            self._results["merged_mask"] = self.inputs.mask_b
            return runtime

        # Both masks provided — merge with voxel-wise OR
        img_a = nib.load(self.inputs.mask_a)
        img_b = nib.load(self.inputs.mask_b)
        data_a = np.asarray(img_a.dataobj) > 0  # type: ignore[attr-defined]

        # Check if geometries match
        shape_a = img_a.header.get_data_shape()[:3]  # type: ignore[attr-defined]
        shape_b = img_b.header.get_data_shape()[:3]  # type: ignore[attr-defined]
        geom_match = (
            shape_a == shape_b
            and float(np.abs(img_a.affine - img_b.affine).max()) < 1e-4  # type: ignore[attr-defined]
        )

        if geom_match:
            data_b = np.asarray(img_b.dataobj) > 0  # type: ignore[attr-defined]
        else:
            from nilearn.image import resample_to_img  # type: ignore[import-untyped]

            resampled = resample_to_img(img_b, img_a, interpolation="nearest")
            data_b = np.asarray(resampled.dataobj) > 0  # type: ignore[attr-defined]

        merged = (data_a | data_b).astype(np.uint8)
        out_path = str(Path.cwd() / "merged_mask.nii.gz")
        nib.save(
            nib.Nifti1Image(merged, img_a.affine, img_a.header),  # type: ignore[attr-defined]
            out_path,
        )
        self._results["merged_mask"] = out_path
        return runtime


