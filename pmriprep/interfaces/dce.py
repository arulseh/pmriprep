"""DCE (Dynamic Contrast Enhanced) preprocessing interfaces for PMRIPrep.

This module implements Nipype interfaces for DCE-MRI preprocessing steps
including B1+ mapping, T1 mapping, motion correction, registration, and
signal processing.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
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


class DCEPreprocessInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCEPreprocess interface."""

    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input 4D DCE time series to preprocess",
    )
    out_file = File(
        genfile=True,
        desc="Output motion-corrected 4D DCE image",
    )
    out_params = File(
        genfile=True,
        desc="Output motion parameters file (.par)",
    )
    baseline_frames = traits.Int(
        0,
        usedefault=True,
        desc="Number of baseline frames (0 = auto-detect from global signal)",
    )
    ref_vol = traits.Either(
        traits.Int,
        traits.Enum("mean", "first", "baseline_mean"),
        default="baseline_mean",
        usedefault=True,
        desc="Reference volume: int for index, 'mean', 'first', or 'baseline_mean' (default)",
    )
    motion_thresh = traits.Float(
        0.5,
        usedefault=True,
        desc="Framewise displacement threshold (mm) for outlier detection",
    )
    head_radius = traits.Float(
        50.0,
        usedefault=True,
        desc="Head radius (mm) for FD calculation (default: 50mm)",
    )
    cost = traits.Enum(
        "normcorr",
        "corratio",
        "normmi",
        "mutualinfo",
        "leastsq",
        usedefault=True,
        desc="Cost function for motion correction (default: normcorr)",
    )
    interpolation = traits.Enum(
        "trilinear",
        "sinc",
        "spline",
        usedefault=True,
        desc="Interpolation method (default: trilinear)",
    )
    brain_mask = File(
        exists=True,
        desc="Brain mask for baseline detection and QC (optional)",
    )
    contrast_threshold = traits.Float(
        1.5,
        usedefault=True,
        desc="Signal increase threshold (%) for bolus arrival detection (default: 1.5%)",
    )


class DCEPreprocessOutputSpec(TraitedSpec):
    """Output specification for DCEPreprocess interface."""

    out_file = File(exists=True, desc="Motion-corrected 4D DCE image")
    out_params = File(exists=True, desc="Motion parameters file (.par)")
    mean_img = File(exists=True, desc="Mean image of motion-corrected data")
    baseline_frames = traits.Int(desc="Number of detected/specified baseline frames")
    baseline_indices = traits.List(traits.Int, desc="List of baseline frame indices")
    bolus_arrival_frame = traits.Int(desc="Detected bolus arrival frame index")
    fd = traits.List(traits.Float, desc="Framewise displacement per volume (mm)")
    fd_mean = traits.Float(desc="Mean framewise displacement (mm)")
    fd_max = traits.Float(desc="Maximum framewise displacement (mm)")
    n_outliers = traits.Int(desc="Number of outlier volumes (FD > threshold)")
    outlier_frames = traits.List(traits.Int, desc="Indices of outlier volumes")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class DCEPreprocess(SimpleInterface):
    """DCE preprocessing with motion correction and baseline detection.

    This interface performs motion correction on 4D DCE perfusion time series
    using FSL mcflirt, with automatic detection of baseline frames from the
    global signal time course. The baseline period is identified as the
    pre-contrast phase before bolus arrival.

    Motion correction aligns all volumes to a reference (by default, the mean
    of baseline frames), which minimizes misregistration due to motion during
    the DCE acquisition.

    Parameters
    ----------
    in_file : str
        Input 4D DCE time series
    baseline_frames : int
        Number of baseline frames (0 = auto-detect from global signal)
    ref_vol : int or str
        Reference volume: integer index, 'mean', 'first', or 'baseline_mean' (default)
    motion_thresh : float
        FD threshold (mm) for flagging outlier volumes (default: 0.5mm)
    head_radius : float
        Head radius (mm) for FD calculation (default: 50mm)
    cost : str
        Cost function: normcorr, corratio, normmi, etc. (default: normcorr)
    interpolation : str
        Interpolation: trilinear, sinc, or spline (default: trilinear)
    brain_mask : str, optional
        Brain mask for baseline detection (improves accuracy)
    contrast_threshold : float
        Signal increase threshold (%) for bolus arrival detection (default: 1.5%)

    Returns
    -------
    out_file : str
        Motion-corrected 4D DCE image
    out_params : str
        Motion parameters file (FSL format)
    mean_img : str
        Mean image of motion-corrected data
    baseline_frames : int
        Number of detected/specified baseline frames
    baseline_indices : list
        List of baseline frame indices (0 to baseline_frames-1)
    bolus_arrival_frame : int
        Detected bolus arrival frame index
    fd : list
        Framewise displacement per volume (mm)
    fd_mean : float
        Mean FD across all volumes
    fd_max : float
        Maximum FD
    n_outliers : int
        Number of volumes with FD > motion_thresh
    outlier_frames : list
        Indices of outlier volumes
    qc_metrics : dict
        Complete QC metrics dictionary

    Notes
    -----
    **Baseline Detection Algorithm:**

    When baseline_frames=0 (auto-detect), the algorithm:

    1. Computes the global mean signal for each time point (within brain mask if provided)
    2. Computes the mean and std of signal in a sliding window
    3. Detects bolus arrival as the first frame where signal increases by more than
       contrast_threshold × std above the baseline mean
    4. Sets baseline_frames to the frame before bolus arrival

    For DCE with gadolinium contrast, the signal typically INCREASES (T1-shortening)
    at bolus arrival, unlike DSC where signal DECREASES (T2* effect).

    **Motion Correction:**

    Uses FSL mcflirt with the baseline mean as reference by default. This ensures
    the motion correction reference has optimal contrast for registration (pre-contrast).

    Motion parameters are output in FSL format:
    rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm)

    Framewise displacement formula: FD = |Δtrans| + r × |Δrot|

    Requires FSL to be installed and available in PATH.

    Examples
    --------
    >>> preproc = DCEPreprocess()
    >>> preproc.inputs.in_file = 'dce_raw.nii.gz'
    >>> preproc.inputs.baseline_frames = 5  # Known baseline
    >>> result = preproc.run()  # doctest: +SKIP

    Auto-detect baseline:

    >>> preproc = DCEPreprocess()
    >>> preproc.inputs.in_file = 'dce_raw.nii.gz'
    >>> preproc.inputs.baseline_frames = 0  # Auto-detect
    >>> preproc.inputs.brain_mask = 'brain_mask.nii.gz'  # Improves accuracy
    >>> result = preproc.run()  # doctest: +SKIP

    See Also
    --------
    FSL mcflirt: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MCFLIRT
    MotionCorrection : DSC motion correction interface
    """

    input_spec = DCEPreprocessInputSpec
    output_spec = DCEPreprocessOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE preprocessing."""
        import subprocess
        import nibabel as nib  # type: ignore[import-untyped]

        # Generate output filenames
        out_file = self._gen_outfilename()
        out_params = self._gen_params_filename()
        mean_file = self._gen_mean_filename()

        # Load input DCE data
        dce_img = nib.load(self.inputs.in_file)
        dce_data = dce_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
        affine = dce_img.affine  # type: ignore[attr-defined]
        header = dce_img.header  # type: ignore[attr-defined]
        n_volumes = dce_data.shape[3]

        # Load brain mask if provided
        brain_mask: Optional[np.ndarray] = None
        if self.inputs.brain_mask:
            mask_img = nib.load(self.inputs.brain_mask)
            brain_mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

        # Detect or use specified baseline frames
        if self.inputs.baseline_frames > 0:
            # Use specified baseline
            baseline_frames = self.inputs.baseline_frames
            # Detect bolus arrival for QC
            bolus_frame = self._detect_bolus_arrival(dce_data, brain_mask)
        else:
            # Auto-detect baseline from global signal
            baseline_frames, bolus_frame = self._auto_detect_baseline(
                dce_data, brain_mask
            )

        # Ensure baseline is at least 3 frames and doesn't exceed available volumes
        baseline_frames = max(3, min(baseline_frames, n_volumes - 1))
        baseline_indices = list(range(baseline_frames))

        # Determine reference volume for motion correction
        ref_vol = self.inputs.ref_vol
        ref_vol_index: Optional[int] = None

        if ref_vol == "baseline_mean":
            # Create mean of baseline frames as reference
            baseline_data = np.mean(dce_data[..., :baseline_frames], axis=3)
            ref_file = str(Path(out_file).parent / "_tmp_baseline_mean.nii.gz")
            ref_img = nib.Nifti1Image(baseline_data.astype(np.float32), affine, header)
            nib.save(ref_img, ref_file)
        elif ref_vol == "mean":
            ref_file = None  # mcflirt will compute mean
        elif ref_vol == "first":
            ref_vol_index = 0
            ref_file = None
        elif isinstance(ref_vol, int):
            ref_vol_index = ref_vol
            ref_file = None
        else:
            ref_file = None

        # Build mcflirt command
        in_file = str(self.inputs.in_file)

        # Remove extension for mcflirt output basename
        out_path = Path(out_file)
        if out_path.name.endswith(".nii.gz"):
            out_base = str(out_path)[:-7]
        elif out_path.name.endswith(".nii"):
            out_base = str(out_path)[:-4]
        else:
            out_base = str(out_path)

        cmd_parts = [
            "mcflirt",
            f"-in {in_file}",
            f"-out {out_base}",
            "-plots",  # Output motion parameters
            "-stats",  # Output mean volume
            "-mats",  # Output transformation matrices
        ]

        # Reference volume
        if ref_vol == "baseline_mean" and ref_file:
            cmd_parts.append(f"-reffile {ref_file}")
        elif ref_vol == "mean":
            cmd_parts.append("-meanvol")
        elif ref_vol_index is not None:
            cmd_parts.append(f"-refvol {ref_vol_index}")

        # Cost function
        cmd_parts.append(f"-cost {self.inputs.cost}")

        # Interpolation
        interp = self.inputs.interpolation
        if interp == "sinc":
            cmd_parts.append("-sinc_final")
        elif interp == "spline":
            cmd_parts.append("-spline_final")

        # Stages (default 4)
        cmd_parts.append("-stages 4")

        # Run mcflirt
        cmd = " ".join(cmd_parts)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"mcflirt failed: {result.stderr}")

        # Clean up temp reference file
        if ref_vol == "baseline_mean" and ref_file:
            try:
                Path(ref_file).unlink()
            except Exception:
                pass

        # Load motion parameters and compute QC metrics
        params_file = f"{out_base}.par"
        qc_metrics: dict[str, Any] = {}

        try:
            motion_params = np.loadtxt(params_file)

            # Calculate FD
            head_radius = float(self.inputs.head_radius)
            fd = self._calculate_fd(motion_params, head_radius)

            fd_list = [float(x) for x in fd]
            fd_mean = float(np.mean(fd[1:]))  # Exclude first frame
            fd_max = float(np.max(fd))

            # Outlier detection
            motion_thresh = float(self.inputs.motion_thresh)
            outlier_mask = fd > motion_thresh
            n_outliers = int(np.sum(outlier_mask))
            outlier_frames = [int(i) for i in np.where(outlier_mask)[0]]

            qc_metrics = {
                "fd_mean": fd_mean,
                "fd_max": fd_max,
                "fd_list": fd_list,
                "n_outliers": n_outliers,
                "outlier_frames": outlier_frames,
                "motion_thresh": motion_thresh,
                "n_volumes": n_volumes,
                "percent_outliers": 100.0 * n_outliers / max(1, n_volumes),
                "baseline_frames": baseline_frames,
                "bolus_arrival_frame": bolus_frame,
                "reference_type": str(ref_vol),
            }

            self._results["fd"] = fd_list
            self._results["fd_mean"] = fd_mean
            self._results["fd_max"] = fd_max
            self._results["n_outliers"] = n_outliers
            self._results["outlier_frames"] = outlier_frames

        except Exception:
            # If metrics computation fails, provide empty defaults
            self._results["fd"] = []
            self._results["fd_mean"] = 0.0
            self._results["fd_max"] = 0.0
            self._results["n_outliers"] = 0
            self._results["outlier_frames"] = []

        # Set outputs
        self._results["out_file"] = out_file
        self._results["out_params"] = params_file
        self._results["mean_img"] = f"{out_base}_mean_reg.nii.gz"
        self._results["baseline_frames"] = baseline_frames
        self._results["baseline_indices"] = baseline_indices
        self._results["bolus_arrival_frame"] = bolus_frame
        self._results["qc_metrics"] = qc_metrics

        return runtime

    def _auto_detect_baseline(
        self,
        dce_data: np.ndarray,
        brain_mask: Optional[np.ndarray],
    ) -> tuple[int, int]:
        """Auto-detect baseline frames from global signal time course.

        For DCE with gadolinium, signal INCREASES at bolus arrival due to
        T1-shortening effect. We detect the first significant increase.

        Args:
            dce_data: 4D DCE data array (x, y, z, t)
            brain_mask: Optional brain mask

        Returns:
            Tuple of (baseline_frames, bolus_arrival_frame)
        """
        from pmriprep.utils.dce_constants import DEFAULT_BASELINE_FRAMES

        n_volumes = dce_data.shape[3]

        # Compute global mean signal time course
        if brain_mask is not None:
            # Use mean within brain mask
            signal = np.array([
                np.mean(dce_data[..., t][brain_mask])
                for t in range(n_volumes)
            ])
        else:
            # Use global mean (all non-zero voxels)
            signal = np.array([
                np.mean(dce_data[..., t][dce_data[..., t] > 0])
                for t in range(n_volumes)
            ])

        # Detect bolus arrival
        bolus_frame = self._detect_bolus_arrival_from_signal(signal)

        # Baseline is everything before bolus arrival
        # Use at least 3 frames, default if detection fails
        if bolus_frame <= 3:
            baseline_frames = DEFAULT_BASELINE_FRAMES
        else:
            baseline_frames = bolus_frame

        return baseline_frames, bolus_frame

    def _detect_bolus_arrival(
        self,
        dce_data: np.ndarray,
        brain_mask: Optional[np.ndarray],
    ) -> int:
        """Detect bolus arrival frame from data.

        Args:
            dce_data: 4D DCE data array
            brain_mask: Optional brain mask

        Returns:
            Bolus arrival frame index
        """
        n_volumes = dce_data.shape[3]

        # Compute global mean signal
        if brain_mask is not None:
            signal = np.array([
                np.mean(dce_data[..., t][brain_mask])
                for t in range(n_volumes)
            ])
        else:
            signal = np.array([
                np.mean(dce_data[..., t][dce_data[..., t] > 0])
                for t in range(n_volumes)
            ])

        return self._detect_bolus_arrival_from_signal(signal)

    def _detect_bolus_arrival_from_signal(self, signal: np.ndarray) -> int:
        """Detect bolus arrival from global signal time course.

        For DCE, signal INCREASES at bolus arrival (T1-shortening).
        We look for the first significant increase above baseline.

        Args:
            signal: 1D global signal time course

        Returns:
            Bolus arrival frame index
        """
        from pmriprep.utils.dce_constants import DEFAULT_BASELINE_FRAMES

        n_volumes = len(signal)

        # Use first few frames to estimate baseline statistics
        # Start with a small initial window (3 frames minimum)
        init_window = min(5, n_volumes // 4)
        baseline_mean = np.mean(signal[:init_window])
        baseline_std = np.std(signal[:init_window])

        # Threshold for detecting contrast arrival
        # Signal must increase by contrast_threshold × baseline_std
        # OR by contrast_threshold % of baseline mean (whichever is larger)
        threshold_pct = self.inputs.contrast_threshold / 100.0
        threshold = max(
            baseline_mean * threshold_pct,
            baseline_std * 2.0  # At least 2 std above baseline
        )

        # Find first frame where signal exceeds threshold
        for t in range(init_window, n_volumes):
            if signal[t] > baseline_mean + threshold:
                return t

        # If no bolus detected, return default
        return DEFAULT_BASELINE_FRAMES

    def _calculate_fd(
        self,
        motion_params: npt.NDArray[Any],
        head_radius: float = 50.0,
    ) -> npt.NDArray[Any]:
        """Calculate framewise displacement from motion parameters.

        Args:
            motion_params: Motion parameters (N, 6)
            head_radius: Head radius in mm

        Returns:
            Framewise displacement array
        """
        if motion_params.ndim == 1:
            motion_params = motion_params.reshape(-1, 6)

        n_frames = motion_params.shape[0]
        if n_frames < 2:
            return np.zeros(n_frames, dtype=np.float64)

        # Calculate frame-to-frame differences
        diff = np.diff(motion_params, axis=0)

        # Rotations (columns 0-2) in radians, translations (columns 3-5) in mm
        rot_diff = diff[:, :3]
        trans_diff = diff[:, 3:]

        # Convert rotations to mm displacement
        rot_disp = np.abs(rot_diff) * head_radius

        # Sum absolute displacements
        fd = np.sum(np.abs(trans_diff), axis=1) + np.sum(rot_disp, axis=1)

        # Prepend 0 for first frame
        fd_full = np.zeros(n_frames, dtype=np.float64)
        fd_full[1:] = fd

        return fd_full

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

        return str(Path.cwd() / f"{base}_moco{ext}")

    def _gen_params_filename(self) -> str:
        """Generate motion parameters filename."""
        if self.inputs.out_params:
            return str(self.inputs.out_params)

        out_file = Path(self._gen_outfilename())
        if out_file.name.endswith(".nii.gz"):
            base = out_file.name[:-7]
        elif out_file.name.endswith(".nii"):
            base = out_file.stem
        else:
            base = out_file.stem

        return str(out_file.parent / f"{base}.par")

    def _gen_mean_filename(self) -> str:
        """Generate mean image filename."""
        out_file = Path(self._gen_outfilename())
        if out_file.name.endswith(".nii.gz"):
            base = out_file.name[:-7]
            ext = ".nii.gz"
        elif out_file.name.endswith(".nii"):
            base = out_file.stem
            ext = ".nii"
        else:
            base = out_file.stem
            ext = out_file.suffix

        return str(out_file.parent / f"{base}_mean_reg{ext}")


class DCERegistrationInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCERegistration interface."""

    t1w_file = File(
        exists=True,
        mandatory=True,
        desc="T1-weighted structural image to register to DCE space",
    )
    dce_mean = File(
        exists=True,
        mandatory=True,
        desc="Mean DCE image (reference/target space)",
    )
    out_file = File(
        genfile=True,
        desc="Output registered T1w in DCE space",
    )
    out_matrix = File(
        genfile=True,
        desc="Output transformation matrix (FLIRT .mat format)",
    )
    lesion_mask = File(
        exists=True,
        desc="Optional lesion mask for cost function weighting (excludes lesion region)",
    )
    brain_mask = File(
        exists=True,
        desc="Brain mask for QC metric calculation (optional)",
    )
    skip_bbr = traits.Bool(
        False,
        usedefault=True,
        desc="Skip BBR and use FLIRT only (default: use BBR with FLIRT init)",
    )
    dof = traits.Int(
        6,
        usedefault=True,
        desc="Degrees of freedom for registration (default: 6 for rigid)",
    )
    cost = traits.Enum(
        "normmi",
        "normcorr",
        "corratio",
        "mutualinfo",
        "leastsq",
        usedefault=True,
        desc="Cost function for FLIRT registration (default: normmi)",
    )
    wm_seg = File(
        exists=True,
        desc="White matter segmentation (required for BBR mode)",
    )
    reference_type = traits.Enum(
        "postcon",
        "precon",
        "any",
        default="postcon",
        usedefault=True,
        desc="Expected T1w reference type: postcon (post-contrast), precon, or any",
    )
    lesion_dilation = traits.Float(
        15.0,
        usedefault=True,
        desc="Dilation radius (mm) for lesion mask in cost function weighting",
    )


class DCERegistrationOutputSpec(TraitedSpec):
    """Output specification for DCERegistration interface."""

    out_file = File(exists=True, desc="Registered T1w in DCE space")
    out_matrix = File(exists=True, desc="Affine transformation matrix (FLIRT format)")
    dice = traits.Float(desc="DICE coefficient (if brain mask provided)")
    correlation = traits.Float(desc="Correlation coefficient (if brain mask provided)")
    nmi = traits.Float(desc="Normalized mutual information")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class DCERegistration(SimpleInterface):
    """DCE to T1w structural registration for perfusion map overlays.

    This interface wraps FSL FLIRT (and optionally BBR) to register a T1-weighted
    structural image to DCE space. The registration enables overlay of perfusion
    maps on anatomical images for clinical interpretation.

    Two modes are available:

    1. **BBR mode** (default, skip_bbr=False): Uses FSL's boundary-based registration
       with white matter segmentation for accurate coregistration. Requires wm_seg.

    2. **FLIRT mode** (skip_bbr=True): Uses standard FSL FLIRT rigid registration
       with normalized mutual information cost function.

    Both modes support lesion mask cost function weighting, where the lesion
    region is excluded from the cost function to prevent tumors from biasing
    the registration.

    Parameters
    ----------
    t1w_file : str
        T1-weighted image to register to DCE space
    dce_mean : str
        Mean DCE image (target space)
    lesion_mask : str, optional
        Binary lesion mask - lesion voxels will have reduced weight
    brain_mask : str, optional
        Brain mask for computing QC metrics
    skip_bbr : bool
        If True, use FLIRT only; if False (default), use BBR with FLIRT init
    dof : int
        Degrees of freedom (default: 6 for rigid body)
    cost : str
        Cost function for FLIRT: normmi, normcorr, corratio (default: normmi)
    wm_seg : str
        White matter segmentation (required for BBR mode)
    reference_type : str
        Expected T1w type: postcon (default for DCE), precon, or any
    lesion_dilation : float
        Dilation radius (mm) for lesion mask in cost weighting (default: 15.0)

    Returns
    -------
    out_file : str
        Registered T1w in DCE space
    out_matrix : str
        FSL FLIRT transformation matrix
    dice : float
        DICE coefficient of brain masks (if brain_mask provided)
    correlation : float
        Correlation coefficient within brain mask
    nmi : float
        Normalized mutual information
    qc_metrics : dict
        Complete QC metrics dictionary

    Examples
    --------
    Using BBR mode (recommended):

    >>> reg = DCERegistration()
    >>> reg.inputs.t1w_file = 'T1w_brain.nii.gz'
    >>> reg.inputs.dce_mean = 'dce_mean.nii.gz'
    >>> reg.inputs.wm_seg = 'T1w_wm_probseg.nii.gz'
    >>> result = reg.run()  # doctest: +SKIP

    Using FLIRT mode (skip BBR) with lesion mask:

    >>> reg = DCERegistration()
    >>> reg.inputs.t1w_file = 'T1w_brain.nii.gz'
    >>> reg.inputs.dce_mean = 'dce_mean.nii.gz'
    >>> reg.inputs.skip_bbr = True
    >>> reg.inputs.lesion_mask = 'lesion.nii.gz'
    >>> result = reg.run()  # doctest: +SKIP

    Notes
    -----
    BBR (Boundary-Based Registration) uses the white matter segmentation to
    drive the registration based on the boundary between white matter and
    gray matter, which is visible in both T1w and EPI-based images.

    Lesion mask weighting creates a cost function weight image where:
    weight = 1.0 - lesion_mask (i.e., lesion voxels have weight 0)

    For DCE, the default reference is post-contrast T1w since DCE has contrast
    enhancement similar to post-contrast T1w, providing better intensity
    correspondence for registration.

    QC metrics computed:
    - DICE: Overlap of brain masks (requires brain_mask input)
    - Correlation: Pearson correlation within brain mask
    - NMI: Normalized mutual information (intensity correspondence)

    Requires FSL to be installed and available in PATH.

    See Also
    --------
    FSL FLIRT: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT
    FSL BBR: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT_BBR
    """

    input_spec = DCERegistrationInputSpec
    output_spec = DCERegistrationOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE to T1w registration."""
        import subprocess

        import nibabel as nib  # type: ignore[import-untyped]

        # Generate output filenames
        out_file = self._gen_outfilename()
        out_matrix = self._gen_matfilename()

        t1w_file = str(self.inputs.t1w_file)
        dce_mean = str(self.inputs.dce_mean)
        skip_bbr = self.inputs.skip_bbr
        dof = self.inputs.dof
        cost = self.inputs.cost

        # Determine file extension for temp files
        out_path = Path(out_file)
        ext = ".nii.gz" if out_path.name.endswith(".nii.gz") else ".nii"
        temp_dir = out_path.parent

        cmd_parts: list[str] = []
        temp_files: list[str] = []

        # Handle lesion mask for cost function weighting
        weight_file: Optional[str] = None
        if self.inputs.lesion_mask and Path(self.inputs.lesion_mask).exists():
            # Create weight image: weight = 1 - lesion_mask
            weight_file = str(temp_dir / f"_tmp_dce_weight{ext}")
            temp_files.append(weight_file)

            # Optionally dilate the lesion mask first
            lesion_dilation = self.inputs.lesion_dilation
            if lesion_dilation > 0:
                # Load lesion mask to get voxel sizes for kernel
                lesion_img = nib.load(self.inputs.lesion_mask)
                header = lesion_img.header  # type: ignore[attr-defined]
                voxel_sizes = header.get_zooms()[:3]  # type: ignore[attr-defined]
                # Calculate kernel radius in voxels (use mean voxel size)
                mean_vox_size = float(np.mean(voxel_sizes))
                kernel_radius = max(1.0, lesion_dilation / mean_vox_size)

                # Dilate then invert: -dilM -mul -1 -add 1
                invert_cmd = (
                    f"fslmaths {self.inputs.lesion_mask} "
                    f"-kernel sphere {kernel_radius:.1f} -dilM "
                    f"-mul -1 -add 1 {weight_file}"
                )
            else:
                # Just invert: 1 - lesion_mask
                invert_cmd = (
                    f"fslmaths {self.inputs.lesion_mask} "
                    f"-mul -1 -add 1 {weight_file}"
                )
            cmd_parts.append(invert_cmd)

        # Check if we have WM segmentation for BBR
        wm_seg = getattr(self.inputs, "wm_seg", None)
        has_wm_seg = wm_seg is not None and Path(wm_seg).exists()

        if skip_bbr or not has_wm_seg:
            # Standard FLIRT registration
            flirt_cmd = (
                f"flirt -in {t1w_file} -ref {dce_mean} "
                f"-out {out_file} -omat {out_matrix} "
                f"-dof {dof} -cost {cost}"
            )

            # Add weight flags if available
            if weight_file:
                flirt_cmd += f" -inweight {weight_file} -refweight {weight_file}"

            cmd_parts.append(flirt_cmd)
        else:
            # BBR mode with FLIRT initialization
            # Initial FLIRT registration for starting estimate
            init_matrix = str(temp_dir / "_tmp_dce_init.mat")
            temp_files.append(init_matrix)

            flirt_init_cmd = (
                f"flirt -in {t1w_file} -ref {dce_mean} "
                f"-omat {init_matrix} "
                f"-dof {dof} -cost {cost}"
            )

            if weight_file:
                flirt_init_cmd += f" -inweight {weight_file} -refweight {weight_file}"

            cmd_parts.append(flirt_init_cmd)

            # Threshold WM segmentation for BBR
            wm_thresh = str(temp_dir / f"_tmp_dce_wm_thresh{ext}")
            temp_files.append(wm_thresh)
            cmd_parts.append(f"fslmaths {wm_seg} -thr 0.5 -bin {wm_thresh}")

            # FLIRT with BBR cost function
            bbr_cmd = (
                f"flirt -in {t1w_file} -ref {dce_mean} "
                f"-out {out_file} -omat {out_matrix} "
                f"-dof {dof} -cost bbr "
                f"-init {init_matrix} "
                f"-wmseg {wm_thresh} "
                f"-schedule ${{FSLDIR}}/etc/flirtsch/bbr.sch"
            )

            if weight_file:
                bbr_cmd += f" -refweight {weight_file}"

            cmd_parts.append(bbr_cmd)

        # Clean up temp files at the end
        if temp_files:
            cleanup_cmd = "rm -f " + " ".join(temp_files)
            cmd_parts.append(cleanup_cmd)

        # Execute the commands
        full_cmd = " && ".join(cmd_parts)
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"DCE registration failed: {result.stderr}")

        # Compute QC metrics
        qc_metrics = self._compute_qc_metrics(out_file, dce_mean)

        # Set outputs
        self._results["out_file"] = out_file
        self._results["out_matrix"] = out_matrix
        self._results["qc_metrics"] = qc_metrics

        if "dice" in qc_metrics:
            self._results["dice"] = qc_metrics["dice"]
        if "correlation" in qc_metrics:
            self._results["correlation"] = qc_metrics["correlation"]
        if "nmi" in qc_metrics:
            self._results["nmi"] = qc_metrics["nmi"]

        return runtime

    def _compute_qc_metrics(
        self, registered_t1w: str, dce_mean: str
    ) -> dict[str, Any]:
        """Compute registration QC metrics.

        Args:
            registered_t1w: Path to registered T1w image
            dce_mean: Path to reference DCE mean image

        Returns:
            Dictionary of QC metrics
        """
        qc_metrics: dict[str, Any] = {}

        if not Path(registered_t1w).exists():
            return qc_metrics

        try:
            from pmriprep.utils.registration import (
                calculate_correlation_from_files,
                calculate_nmi_from_files,
                calculate_dice_from_files,
            )

            # Get optional brain mask
            brain_mask = getattr(self.inputs, "brain_mask", None)
            mask_path: Optional[str] = None
            if brain_mask and Path(brain_mask).exists():
                mask_path = str(brain_mask)

            # Calculate NMI (always possible)
            try:
                nmi = calculate_nmi_from_files(
                    registered_t1w, dce_mean, mask_path=mask_path
                )
                qc_metrics["nmi"] = nmi
            except Exception:
                pass

            # Calculate correlation within brain mask
            try:
                corr = calculate_correlation_from_files(
                    registered_t1w, dce_mean, mask_path=mask_path
                )
                qc_metrics["correlation"] = corr
            except Exception:
                pass

            # Calculate DICE if brain mask provided
            # Note: DICE is computed between the thresholded registered image
            # and the thresholded reference, or between two brain masks
            if mask_path:
                try:
                    import nibabel as nib  # type: ignore[import-untyped]

                    # Create a rough brain mask from registered T1w by thresholding
                    reg_img = nib.load(registered_t1w)
                    reg_data = reg_img.get_fdata()  # type: ignore[attr-defined]

                    # Threshold at mean intensity
                    threshold = np.percentile(reg_data[reg_data > 0], 10)
                    reg_mask = reg_data > threshold

                    # Load the provided brain mask
                    mask_img = nib.load(mask_path)
                    ref_mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

                    # Compute DICE between the two masks
                    from pmriprep.utils.registration import calculate_dice

                    dice = calculate_dice(
                        reg_mask.astype(np.float64),
                        ref_mask.astype(np.float64)
                    )
                    qc_metrics["dice"] = dice
                except Exception:
                    pass

            # Add registration metadata
            qc_metrics["skip_bbr"] = self.inputs.skip_bbr
            qc_metrics["dof"] = self.inputs.dof
            qc_metrics["cost_function"] = self.inputs.cost
            qc_metrics["reference_type"] = self.inputs.reference_type
            qc_metrics["brain_mask_used"] = mask_path is not None

            # Determine lesion mask usage
            lesion_used = (
                self.inputs.lesion_mask is not None
                and Path(self.inputs.lesion_mask).exists()
            )
            qc_metrics["lesion_mask_used"] = lesion_used

        except ImportError:
            # If registration utilities not available, skip QC
            pass
        except Exception:
            # If QC computation fails, continue without metrics
            pass

        return qc_metrics

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        if self.inputs.out_file:
            return str(self.inputs.out_file)

        t1w_file = Path(self.inputs.t1w_file)
        if t1w_file.name.endswith(".nii.gz"):
            base = t1w_file.name[:-7]
            ext = ".nii.gz"
        elif t1w_file.name.endswith(".nii"):
            base = t1w_file.stem
            ext = ".nii"
        else:
            base = t1w_file.stem
            ext = t1w_file.suffix

        return str(Path.cwd() / f"{base}_dce{ext}")

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
