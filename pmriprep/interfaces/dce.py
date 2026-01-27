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


class T1MappingInputSpec(BaseInterfaceInputSpec):
    """Input specification for T1Mapping interface."""

    vfa_files = traits.List(
        File(exists=True),
        mandatory=True,
        minlen=2,
        maxlen=5,
        desc="List of VFA images (2-5 images) at different flip angles",
    )
    flip_angles = traits.List(
        traits.Float,
        mandatory=True,
        minlen=2,
        maxlen=5,
        desc="List of flip angles in degrees corresponding to vfa_files",
    )
    tr = traits.Float(
        mandatory=True,
        desc="Repetition time in milliseconds (same for all VFA images)",
    )
    b1_map = File(
        exists=True,
        desc="B1+ correction factor map (optional, from B1Mapping)",
    )
    brain_mask = File(
        exists=True,
        desc="Brain mask for constraining T1 calculation",
    )
    out_file = File(
        genfile=True,
        desc="Output T1 map in milliseconds",
    )
    skip_t1 = traits.Bool(
        False,
        usedefault=True,
        desc="Skip T1 mapping (return uniform T1=1200ms map)",
    )
    lenient_qc = traits.Bool(
        False,
        usedefault=True,
        desc="Use lenient T1 QC ranges (500-2500ms) for clinical variability",
    )
    field_strength = traits.Float(
        3.0,
        usedefault=True,
        desc="Magnetic field strength in Tesla for QC range selection",
    )


class T1MappingOutputSpec(TraitedSpec):
    """Output specification for T1Mapping interface."""

    out_file = File(exists=True, desc="T1 map in milliseconds")
    mean_t1_wm = traits.Float(desc="Mean T1 in white matter region (ms)")
    mean_t1_gm = traits.Float(desc="Mean T1 in gray matter region (ms)")
    qc_pass = traits.Bool(desc="Whether T1 map passed QC criteria")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class T1Mapping(SimpleInterface):
    """Variable Flip Angle (VFA) T1 mapping with B1+ correction.

    This interface implements VFA T1 mapping using the Ernst equation.
    B1+ correction is applied when a B1+ map is provided to compensate
    for transmit field inhomogeneity.

    The VFA method acquires multiple spoiled gradient echo (SPGR) images
    at different flip angles with the same TR. The signal follows the
    Ernst equation, allowing T1 to be estimated by fitting.

    Parameters
    ----------
    vfa_files : list of str
        List of VFA image paths (2-5 images at different flip angles)
    flip_angles : list of float
        List of nominal flip angles in degrees (must match vfa_files)
    tr : float
        Repetition time in milliseconds
    b1_map : str, optional
        B1+ correction factor map from B1Mapping
    brain_mask : str, optional
        Brain mask for QC calculations
    skip_t1 : bool
        If True, skip T1 mapping and return uniform T1=1200ms map
    lenient_qc : bool
        If True, use expanded QC range (500-2500ms) for clinical cases
    field_strength : float
        Magnetic field strength in Tesla for QC range lookup

    Returns
    -------
    out_file : str
        T1 map in milliseconds
    mean_t1_wm : float
        Mean T1 in white matter region
    mean_t1_gm : float
        Mean T1 in gray matter region
    qc_pass : bool
        Whether T1 passed QC (tissue values within expected ranges)
    qc_metrics : dict
        Dictionary of QC metrics

    Notes
    -----
    The Ernst equation for SPGR signal is:

        S(α,TR,T1) = S0 × sin(α) × (1 - E1) / (1 - cos(α) × E1)

    Where:
        - S is the measured signal
        - S0 is the equilibrium signal
        - α is the flip angle
        - E1 = exp(-TR/T1)
        - TR is the repetition time

    With B1+ correction, the actual flip angle is:

        α_actual = α_nominal × B1+

    For linearized fitting with two flip angles, we use the DESPOT1
    approach (Deoni et al., 2003):

        S(α)/sin(α) = E1 × S(α)/tan(α) + S0 × (1 - E1)

    Which is linear: Y = m × X + b, where:
        - Y = S/sin(α)
        - X = S/tan(α)
        - m = E1
        - b = S0 × (1 - E1)

    T1 is then calculated from E1:
        T1 = -TR / ln(E1)

    QC ranges by field strength (3T):
        - WM T1: 800-1000ms (lenient: 500-2500ms)
        - GM T1: 1200-1600ms (lenient: 500-2500ms)

    References
    ----------
    Deoni SCL, Rutt BK, Peters TM. "Rapid combined T1 and T2 mapping
    using gradient recalled acquisition in the steady state."
    Magn Reson Med. 2003;49(3):515-526.

    See Also
    --------
    B1Mapping : B1+ mapping for flip angle correction
    """

    input_spec = T1MappingInputSpec
    output_spec = T1MappingOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute VFA T1 mapping computation."""
        import nibabel as nib  # type: ignore[import-untyped]

        from pmriprep.utils.dce_constants import (
            DEFAULT_T1_MS,
            EPSILON_T1,
            LENIENT_T1_RANGE,
            get_tissue_t1_ranges,
        )

        # Generate output filename
        out_file = self._gen_outfilename()

        # Load first VFA image to get shape and affine
        first_img = nib.load(self.inputs.vfa_files[0])
        shape = first_img.shape[:3]  # type: ignore[attr-defined]
        affine = first_img.affine  # type: ignore[attr-defined]
        header = first_img.header  # type: ignore[attr-defined]

        # Check if T1 mapping should be skipped
        if self.inputs.skip_t1:
            # Create uniform T1 map with default value
            t1_map = np.full(shape, DEFAULT_T1_MS, dtype=np.float32)

            # Save output
            out_img = nib.Nifti1Image(t1_map, affine)
            nib.save(out_img, out_file)

            # Set outputs for skipped case
            self._results["out_file"] = out_file
            self._results["mean_t1_wm"] = DEFAULT_T1_MS
            self._results["mean_t1_gm"] = DEFAULT_T1_MS
            self._results["qc_pass"] = True
            self._results["qc_metrics"] = {
                "mean_t1_wm_ms": DEFAULT_T1_MS,
                "mean_t1_gm_ms": DEFAULT_T1_MS,
                "qc_pass": True,
                "skipped": True,
            }
            return runtime

        # Get parameters
        tr_ms = self.inputs.tr
        tr_s = tr_ms / 1000.0  # Convert to seconds for calculation
        flip_angles_deg = np.array(self.inputs.flip_angles)
        flip_angles_rad = np.radians(flip_angles_deg)

        # Load all VFA images
        n_fa = len(self.inputs.vfa_files)
        signal_data = np.zeros((n_fa,) + shape, dtype=np.float64)

        for i, vfa_file in enumerate(self.inputs.vfa_files):
            img = nib.load(vfa_file)
            signal_data[i] = img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]

        # Load B1+ map if provided, otherwise use unity
        if self.inputs.b1_map:
            b1_img = nib.load(self.inputs.b1_map)
            b1_map = b1_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
        else:
            b1_map = np.ones(shape, dtype=np.float64)

        # Apply B1+ correction to flip angles
        # α_actual = α_nominal × B1+
        # Shape: (n_fa, x, y, z) after broadcasting
        corrected_fa_rad = flip_angles_rad[:, np.newaxis, np.newaxis, np.newaxis] * b1_map

        # Calculate sin and tan of corrected flip angles
        sin_fa = np.sin(corrected_fa_rad)
        tan_fa = np.tan(corrected_fa_rad)

        # Prevent division by zero
        sin_fa = np.where(np.abs(sin_fa) < EPSILON_T1, EPSILON_T1, sin_fa)
        tan_fa = np.where(np.abs(tan_fa) < EPSILON_T1, EPSILON_T1, tan_fa)

        # Calculate Y = S/sin(α) and X = S/tan(α) for DESPOT1 linearization
        y_data = signal_data / sin_fa  # Shape: (n_fa, x, y, z)
        x_data = signal_data / tan_fa  # Shape: (n_fa, x, y, z)

        # Perform linear regression: Y = m × X + b
        # Using least squares for each voxel
        # m = E1, b = S0 × (1 - E1)
        t1_map = self._fit_t1_linear(x_data, y_data, tr_s, shape)

        # Convert T1 from seconds to milliseconds
        t1_map = t1_map * 1000.0

        # Clamp T1 to reasonable range (0 to 10000 ms)
        t1_map = np.clip(t1_map, 0.0, 10000.0)

        # Convert to float32 for storage
        t1_map = t1_map.astype(np.float32)

        # Load brain mask for QC if provided
        if self.inputs.brain_mask:
            mask_img = nib.load(self.inputs.brain_mask)
            brain_mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]
        else:
            # Use simple thresholding on first VFA image as rough brain mask
            threshold = np.percentile(signal_data[0][signal_data[0] > 0], 10)
            brain_mask = signal_data[0] > threshold

        # Calculate QC metrics
        field_strength = self.inputs.field_strength
        qc_ranges = get_tissue_t1_ranges(field_strength)

        # Use lenient range if specified
        if self.inputs.lenient_qc:
            wm_range = LENIENT_T1_RANGE
            gm_range = LENIENT_T1_RANGE
        else:
            wm_range = (qc_ranges.white_matter.min, qc_ranges.white_matter.max)
            gm_range = (qc_ranges.gray_matter.min, qc_ranges.gray_matter.max)

        # Estimate WM and GM mean T1 from T1 distribution in brain
        t1_in_brain = t1_map[brain_mask]

        # WM: lower T1 values (shorter relaxation time)
        # GM: higher T1 values
        # Simple approach: use histogram percentiles
        wm_t1_estimate = float(np.percentile(t1_in_brain[t1_in_brain > 0], 25))
        gm_t1_estimate = float(np.percentile(t1_in_brain[t1_in_brain > 0], 75))

        # QC pass/fail criteria
        wm_pass = wm_range[0] <= wm_t1_estimate <= wm_range[1]
        gm_pass = gm_range[0] <= gm_t1_estimate <= gm_range[1]
        qc_pass = wm_pass and gm_pass

        # Build QC metrics dictionary
        qc_metrics = {
            "mean_t1_wm_ms": wm_t1_estimate,
            "mean_t1_gm_ms": gm_t1_estimate,
            "qc_pass": qc_pass,
            "wm_pass": wm_pass,
            "gm_pass": gm_pass,
            "skipped": False,
            "b1_corrected": self.inputs.b1_map is not None,
            "n_flip_angles": n_fa,
            "flip_angles_deg": flip_angles_deg.tolist(),
            "tr_ms": tr_ms,
            "field_strength_t": field_strength,
            "wm_range_ms": wm_range,
            "gm_range_ms": gm_range,
            "lenient_qc": self.inputs.lenient_qc,
        }

        # Save output T1 map
        out_img = nib.Nifti1Image(t1_map, affine, header)
        nib.save(out_img, out_file)

        # Set outputs
        self._results["out_file"] = out_file
        self._results["mean_t1_wm"] = wm_t1_estimate
        self._results["mean_t1_gm"] = gm_t1_estimate
        self._results["qc_pass"] = qc_pass
        self._results["qc_metrics"] = qc_metrics

        return runtime

    def _fit_t1_linear(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        tr_s: float,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Fit T1 using linearized DESPOT1 approach.

        Y = E1 × X + S0 × (1 - E1)

        Where:
            - Y = S/sin(α)
            - X = S/tan(α)
            - m = E1 = exp(-TR/T1)
            - b = S0 × (1 - E1)

        Args:
            x_data: S/tan(α) data, shape (n_fa, x, y, z)
            y_data: S/sin(α) data, shape (n_fa, x, y, z)
            tr_s: TR in seconds
            shape: Output shape (x, y, z)

        Returns:
            T1 map in seconds
        """
        from pmriprep.utils.dce_constants import EPSILON_T1

        n_fa = x_data.shape[0]
        n_voxels = np.prod(shape)

        # Reshape to (n_fa, n_voxels) for vectorized regression
        x_flat = x_data.reshape(n_fa, n_voxels)
        y_flat = y_data.reshape(n_fa, n_voxels)

        # Linear regression: Y = m × X + b
        # Using the formula: m = (n*sum(XY) - sum(X)*sum(Y)) / (n*sum(X²) - sum(X)²)

        # Calculate sums
        sum_x = np.sum(x_flat, axis=0)
        sum_y = np.sum(y_flat, axis=0)
        sum_xy = np.sum(x_flat * y_flat, axis=0)
        sum_x2 = np.sum(x_flat * x_flat, axis=0)

        # Calculate slope (E1)
        denom = n_fa * sum_x2 - sum_x * sum_x
        denom = np.where(np.abs(denom) < EPSILON_T1, EPSILON_T1, denom)

        e1 = (n_fa * sum_xy - sum_x * sum_y) / denom

        # Clamp E1 to valid range (0 < E1 < 1 for positive T1)
        # E1 values outside this range indicate invalid fits
        e1 = np.clip(e1, EPSILON_T1, 1.0 - EPSILON_T1)

        # Calculate T1 from E1
        # E1 = exp(-TR/T1)
        # -TR/T1 = ln(E1)
        # T1 = -TR / ln(E1)
        t1_flat = -tr_s / np.log(e1)

        # Reshape back to 3D
        t1_map: np.ndarray = t1_flat.reshape(shape)  # type: ignore[assignment]

        return t1_map

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        in_file = Path(self.inputs.vfa_files[0])
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

        # Remove any VFA-specific suffixes and add T1map suffix
        base = base.replace("_VFA", "").replace("_vfa", "")
        base = base.replace("_fa-", "_").split("_fa-")[0]

        return str(Path.cwd() / f"{base}_T1map{ext}")
