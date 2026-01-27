"""DCE (Dynamic Contrast Enhanced) preprocessing interfaces for PMRIPrep.

This module implements Nipype interfaces for DCE-MRI preprocessing steps
including B1+ mapping, T1 mapping, motion correction, registration, and
signal processing.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class B1MappingInputSpec(BaseInterfaceInputSpec):
    """Input specification for B1Mapping interface."""

    tr1_file = File(
        exists=True,
        mandatory=True,
        desc="AFI image acquired with short TR (TR1, typically ~20ms)",
    )
    tr2_file = File(
        exists=True,
        mandatory=True,
        desc="AFI image acquired with long TR (TR2, typically ~100ms)",
    )
    tr1 = traits.Float(
        mandatory=True,
        desc="Short TR value in milliseconds (TR1, typically 20ms)",
    )
    tr2 = traits.Float(
        mandatory=True,
        desc="Long TR value in milliseconds (TR2, typically 100ms)",
    )
    nominal_flip_angle = traits.Float(
        60.0,
        usedefault=True,
        desc="Nominal flip angle in degrees (default: 60)",
    )
    brain_mask = File(
        exists=True,
        desc="Brain mask for constraining B1+ calculation",
    )
    out_file = File(
        genfile=True,
        desc="Output B1+ correction factor map",
    )
    smoothing_fwhm = traits.Float(
        8.0,
        usedefault=True,
        desc="Smoothing kernel FWHM in mm for B1+ map (default: 8.0mm)",
    )
    skip_b1 = traits.Bool(
        False,
        usedefault=True,
        desc="Skip B1+ mapping (return unity map)",
    )


class B1MappingOutputSpec(TraitedSpec):
    """Output specification for B1Mapping interface."""

    out_file = File(exists=True, desc="B1+ correction factor map")
    mean_b1 = traits.Float(desc="Mean B1+ correction factor within brain mask")
    std_b1 = traits.Float(desc="Standard deviation of B1+ within brain mask")
    qc_pass = traits.Bool(desc="Whether B1+ map passed QC criteria")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class B1Mapping(SimpleInterface):
    """B1+ mapping from Actual Flip angle Imaging (AFI) acquisition.

    This interface implements the AFI method (Yarnykh, 2007) for mapping
    the B1+ transmit field inhomogeneity. B1+ correction is critical for
    accurate T1 mapping and DCE quantification, particularly at higher
    field strengths (3T, 7T).

    The AFI method acquires two spoiled gradient echo images with
    identical flip angles but different repetition times (TR1 << TR2).
    The ratio of these images depends only on the actual flip angle,
    allowing B1+ to be calculated without knowing T1.

    Parameters
    ----------
    tr1_file : str
        AFI image with short TR (typically 20ms)
    tr2_file : str
        AFI image with long TR (typically 100ms)
    tr1 : float
        Short TR value in milliseconds
    tr2 : float
        Long TR value in milliseconds
    nominal_flip_angle : float
        Nominal flip angle in degrees (default: 60)
    brain_mask : str, optional
        Brain mask for QC calculations
    smoothing_fwhm : float
        Smoothing kernel FWHM in mm (default: 8.0)
    skip_b1 : bool
        If True, skip B1+ mapping and return unity map

    Returns
    -------
    out_file : str
        B1+ correction factor map (actual_flip_angle / nominal_flip_angle)
    mean_b1 : float
        Mean B1+ value within brain mask (ideally 1.0)
    std_b1 : float
        Standard deviation of B1+ within brain mask
    qc_pass : bool
        Whether B1+ passed QC (mean: 0.85-1.15, std: <0.3)
    qc_metrics : dict
        Dictionary of QC metrics

    Notes
    -----
    The AFI method calculates B1+ using:

        r = S1/S2 = (1 + n*cos(alpha)*E2) / (n + cos(alpha)*E2)

    Where:
        - S1, S2 are the signal intensities from TR1 and TR2
        - n = TR2/TR1 (TR ratio)
        - alpha is the actual flip angle
        - E2 = exp(-TR2/T1)

    For TR2 >> T1, E2 ≈ 0, and the equation simplifies to:

        cos(alpha) = (r*n - 1) / (n - r)

    The B1+ correction factor is then:

        B1+ = actual_flip_angle / nominal_flip_angle

    QC criteria:
        - Mean B1+: 0.85-1.15 (expected ~1.0 for well-calibrated system)
        - Spatial std: <0.3 (B1+ should be spatially smooth)

    References
    ----------
    Yarnykh VL. "Actual flip-angle imaging in the pulsed steady state:
    A method for rapid three-dimensional mapping of the transmitted
    radiofrequency field." Magn Reson Med. 2007;57(1):192-200.

    See Also
    --------
    T1Mapping : VFA T1 mapping with B1+ correction
    """

    input_spec = B1MappingInputSpec
    output_spec = B1MappingOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute B1+ mapping computation."""
        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

        from pmriprep.utils.dce_constants import (
            B1_EXPECTED_MEAN,
            B1_MAX_STD,
            B1_MEAN_RANGE,
            EPSILON_B1,
        )

        # Generate output filename
        out_file = self._gen_outfilename()

        # Check if B1 mapping should be skipped
        if self.inputs.skip_b1:
            # Load one image to get the shape and affine
            tr1_img = nib.load(self.inputs.tr1_file)
            shape = tr1_img.shape[:3]  # type: ignore[attr-defined]  # 3D shape
            affine = tr1_img.affine  # type: ignore[attr-defined]

            # Create unity B1+ map
            b1_map = np.ones(shape, dtype=np.float32)

            # Save output
            out_img = nib.Nifti1Image(b1_map, affine)
            nib.save(out_img, out_file)

            # Set outputs for skipped case
            self._results["out_file"] = out_file
            self._results["mean_b1"] = 1.0
            self._results["std_b1"] = 0.0
            self._results["qc_pass"] = True
            self._results["qc_metrics"] = {
                "mean_b1": 1.0,
                "std_b1": 0.0,
                "qc_pass": True,
                "skipped": True,
            }
            return runtime

        # Load AFI images
        tr1_img = nib.load(self.inputs.tr1_file)
        tr2_img = nib.load(self.inputs.tr2_file)

        s1 = tr1_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
        s2 = tr2_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
        affine = tr1_img.affine  # type: ignore[attr-defined]
        header = tr1_img.header  # type: ignore[attr-defined]

        # Get TR ratio
        tr1_ms = self.inputs.tr1
        tr2_ms = self.inputs.tr2
        n = tr2_ms / tr1_ms

        # Compute signal ratio r = S1/S2
        # Add small epsilon to prevent division by zero
        r = s1 / np.maximum(s2, EPSILON_B1)

        # AFI equation (assuming TR2 >> T1, i.e., E2 ≈ 0):
        # cos(alpha) = (r*n - 1) / (n - r)
        # We need to handle edge cases where denominator approaches zero
        numerator = r * n - 1.0
        denominator = n - r

        # Clip denominator to prevent division by zero
        # When r approaches n, the flip angle is very large (>90 deg)
        denominator_safe = np.where(
            np.abs(denominator) < EPSILON_B1,
            np.sign(denominator) * EPSILON_B1,
            denominator,
        )

        cos_alpha = numerator / denominator_safe

        # Clip cos(alpha) to valid range [-1, 1]
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

        # Calculate actual flip angle in degrees
        alpha_actual = np.degrees(np.arccos(cos_alpha))

        # Calculate B1+ correction factor
        # B1+ = actual_flip_angle / nominal_flip_angle
        nominal_fa = self.inputs.nominal_flip_angle
        b1_map = alpha_actual / nominal_fa

        # Apply smoothing to reduce noise
        smoothing_fwhm = self.inputs.smoothing_fwhm
        if smoothing_fwhm > 0:
            # Get voxel sizes for proper FWHM calculation
            voxel_sizes = header.get_zooms()[:3]  # type: ignore[attr-defined]
            # Convert FWHM to sigma in voxels
            sigma_voxels = [
                smoothing_fwhm / (2.355 * vs) for vs in voxel_sizes
            ]
            b1_map = gaussian_filter(b1_map, sigma=sigma_voxels)

        # Convert to float32 for storage
        b1_map = b1_map.astype(np.float32)

        # Load brain mask for QC if provided
        if self.inputs.brain_mask:
            mask_img = nib.load(self.inputs.brain_mask)
            mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]
        else:
            # Use simple thresholding on S2 image as rough brain mask
            threshold = np.percentile(s2[s2 > 0], 10)
            mask = s2 > threshold

        # Calculate QC metrics within brain mask
        b1_in_mask = b1_map[mask]

        # Remove outliers for robust statistics
        b1_clean = b1_in_mask[
            (b1_in_mask > 0.5) & (b1_in_mask < 1.5)
        ]

        if len(b1_clean) > 0:
            mean_b1 = float(np.mean(b1_clean))
            std_b1 = float(np.std(b1_clean))
        else:
            mean_b1 = B1_EXPECTED_MEAN
            std_b1 = 0.0

        # QC pass/fail criteria
        qc_pass = (
            B1_MEAN_RANGE[0] <= mean_b1 <= B1_MEAN_RANGE[1]
            and std_b1 < B1_MAX_STD
        )

        # Build QC metrics dictionary
        qc_metrics = {
            "mean_b1": mean_b1,
            "std_b1": std_b1,
            "qc_pass": qc_pass,
            "skipped": False,
            "smoothing_fwhm_mm": smoothing_fwhm,
            "nominal_flip_angle_deg": nominal_fa,
            "tr1_ms": tr1_ms,
            "tr2_ms": tr2_ms,
            "tr_ratio": n,
            "b1_range_expected": B1_MEAN_RANGE,
            "b1_max_std_expected": B1_MAX_STD,
        }

        # Save output B1+ map
        out_img = nib.Nifti1Image(b1_map, affine, header)
        nib.save(out_img, out_file)

        # Set outputs
        self._results["out_file"] = out_file
        self._results["mean_b1"] = mean_b1
        self._results["std_b1"] = std_b1
        self._results["qc_pass"] = qc_pass
        self._results["qc_metrics"] = qc_metrics

        return runtime

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        in_file = Path(self.inputs.tr1_file)
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

        # Remove any AFI-specific suffixes and add B1plus suffix
        base = base.replace("_AFI", "").replace("_afi", "")
        base = base.replace("_TR1", "").replace("_tr1", "")

        return str(Path.cwd() / f"{base}_B1plus{ext}")
