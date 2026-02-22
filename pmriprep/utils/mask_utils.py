"""Mask manipulation utilities for PMRIPrep.

Provides Nipype SimpleInterface subclasses for mask merging and dilation,
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
from scipy import ndimage  # type: ignore[import-untyped]


class _MergeMasksInputSpec(BaseInterfaceInputSpec):
    """Input specification for MergeMasks."""

    mask_a = traits.Either(
        File(exists=True), None, desc="First mask file (optional)"
    )
    mask_b = traits.Either(
        File(exists=True), None, desc="Second mask file (optional)"
    )


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


class _DilateMaskInputSpec(BaseInterfaceInputSpec):
    """Input specification for DilateMask."""

    mask_file = File(exists=True, mandatory=True, desc="Binary mask to dilate")
    radius_mm = traits.Float(
        15.0, usedefault=True, desc="Dilation radius in millimeters"
    )


class _DilateMaskOutputSpec(TraitedSpec):
    """Output specification for DilateMask."""

    dilated_mask = File(exists=True, desc="Dilated binary mask")


class DilateMask(SimpleInterface):
    """Dilate a binary mask using a spherical structuring element.

    Applies morphological dilation accounting for anisotropic voxel
    dimensions. The output filename always contains 'dilated'.
    """

    input_spec = _DilateMaskInputSpec
    output_spec = _DilateMaskOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        mask_path = Path(self.inputs.mask_file)
        mask_img = nib.load(mask_path)
        mask_data = np.asarray(mask_img.dataobj) > 0  # type: ignore[attr-defined]
        header = mask_img.header  # type: ignore[attr-defined]
        zooms = header.get_zooms()[:3]  # type: ignore[attr-defined]
        voxel_sizes: tuple[float, float, float] = (
            float(zooms[0]), float(zooms[1]), float(zooms[2]),
        )

        radius_mm = float(self.inputs.radius_mm)

        # Create spherical kernel (same algorithm as hdglio.dilate_mask)
        kernel = _create_spherical_kernel(radius_mm, voxel_sizes)
        dilated = ndimage.binary_dilation(mask_data, structure=kernel)

        # Build output filename with "dilated" in name
        if mask_path.name.endswith(".nii.gz"):
            base = mask_path.name[:-7]
            ext = ".nii.gz"
        else:
            base = mask_path.stem
            ext = mask_path.suffix

        out_path = str(Path.cwd() / f"{base}_dilated{ext}")
        nib.save(
            nib.Nifti1Image(  # type: ignore[attr-defined]
                dilated.astype(np.uint8),
                mask_img.affine,  # type: ignore[attr-defined]
                mask_img.header,  # type: ignore[attr-defined]
            ),
            out_path,
        )
        self._results["dilated_mask"] = out_path
        return runtime


def _create_spherical_kernel(
    radius_mm: float,
    voxel_sizes: tuple[float, float, float],
) -> np.ndarray:
    """Create a spherical structuring element respecting voxel anisotropy.

    Parameters
    ----------
    radius_mm : float
        Radius of the sphere in millimeters.
    voxel_sizes : tuple of float
        Voxel dimensions (x, y, z) in millimeters.

    Returns
    -------
    np.ndarray
        Boolean array representing the spherical kernel.
    """
    MAX_KERNEL_RADIUS_VOXELS = 50

    vx, vy, vz = voxel_sizes
    nx = int(np.ceil(radius_mm / vx))
    ny = int(np.ceil(radius_mm / vy))
    nz = int(np.ceil(radius_mm / vz))

    if nx > MAX_KERNEL_RADIUS_VOXELS or ny > MAX_KERNEL_RADIUS_VOXELS or nz > MAX_KERNEL_RADIUS_VOXELS:
        raise ValueError(
            f"Kernel radius exceeds safety cap ({MAX_KERNEL_RADIUS_VOXELS} voxels): "
            f"nx={nx}, ny={ny}, nz={nz} for radius_mm={radius_mm}, "
            f"voxel_sizes=({vx}, {vy}, {vz})"
        )

    size_x = 2 * nx + 1
    size_y = 2 * ny + 1
    size_z = 2 * nz + 1

    x = np.arange(size_x, dtype=np.float64) - nx
    y = np.arange(size_y, dtype=np.float64) - ny
    z = np.arange(size_z, dtype=np.float64) - nz

    xx, yy, zz = np.meshgrid(x * vx, y * vy, z * vz, indexing="ij")
    distance = np.sqrt(xx**2 + yy**2 + zz**2)

    return np.asarray(distance <= radius_mm)
