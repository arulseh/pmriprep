"""DCE (Dynamic Contrast Enhanced) preprocessing interfaces for PMRIPrep.

This module implements Nipype interfaces for DCE-MRI preprocessing steps
including B1+ mapping, T1 mapping, motion correction, registration, and
signal processing.
"""

import logging
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.fsl import FLIRT  # type: ignore[import-untyped]

logger = logging.getLogger("pmriprep.interfaces.dce")


def _reorient_to_ras(in_file: Optional[str]) -> Optional[str]:
    """Reorient NIfTI to RAS orientation using fslreorient2std.

    Args:
        in_file: Path to input NIfTI file, or None.

    Returns:
        Path to reoriented NIfTI file, or None if input was None.
    """
    if in_file is None:
        return None

    import subprocess

    in_path = Path(in_file)
    if not in_path.exists():
        return None

    # Generate output filename in current working directory
    if in_path.name.endswith(".nii.gz"):
        base = in_path.name[:-7]
        ext = ".nii.gz"
    elif in_path.name.endswith(".nii"):
        base = in_path.stem
        ext = ".nii"
    else:
        base = in_path.stem
        ext = in_path.suffix

    out_file = str(Path.cwd() / f"{base}_reoriented{ext}")

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
    _ornt = io_orientation(_img.affine)  # type: ignore[attr-defined]
    _ras_ornt = axcodes2ornt(("R", "A", "S"))
    if not _np.array_equal(_ornt, _ras_ornt):
        _transform = ornt_transform(_ornt, _ras_ornt)
        _img_ras = _img.as_reoriented(_transform)  # type: ignore[attr-defined]
        _nib.save(_img_ras, str(out_file))

    return out_file


def _b1_interior_mask(valid: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Compute interior mask for B1+ normalized convolution.

    Fills internal holes (CSF, noisy B1+) so erosion targets only the
    outer brain boundary, then intersects with the original valid mask
    to exclude filled-but-invalid voxels.

    Args:
        valid: Boolean mask of valid B1+ voxels.
        iterations: Number of erosion iterations (default 2).

    Returns:
        Boolean interior mask.
    """
    from scipy.ndimage import binary_erosion, binary_fill_holes  # type: ignore[import-untyped]

    return binary_erosion(binary_fill_holes(valid), iterations=iterations) & valid  # type: ignore[no-any-return]


class B1MappingInputSpec(BaseInterfaceInputSpec):
    """Input specification for B1Mapping interface."""

    tr1_file = File(
        exists=True,
        desc="AFI image acquired with short TR (TR1, typically ~20ms)",
    )
    tr2_file = File(
        exists=True,
        desc="AFI image acquired with long TR (TR2, typically ~100ms)",
    )
    tr1 = traits.Float(
        desc="Short TR value in milliseconds (TR1, typically 20ms)",
    )
    tr2 = traits.Float(
        desc="Long TR value in milliseconds (TR2, typically 100ms)",
    )
    afi_4d_file = File(
        exists=True,
        desc="Single 4D AFI volume with 2 timepoints (alternative to separate tr1/tr2 files)",
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
    reference_file = File(
        exists=True,
        desc="Reference NIfTI for output shape/affine when skip_b1=True and no AFI or brain_mask",
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
    field_strength = traits.Float(
        3.0,
        usedefault=True,
        desc="Magnetic field strength in Tesla",
    )
    subject_id = traits.Str(
        "",
        usedefault=True,
        desc="Subject ID for QC figure title",
    )
    session_id = traits.Str(
        "",
        usedefault=True,
        desc="Session ID for QC figure title",
    )
    precomputed_b1_file = File(
        exists=True,
        desc="Pre-computed B1+ flip angle map (e.g., tfl_b1map). "
             "Values in 0.1 degree units (Siemens convention).",
    )
    b1_nominal_prep_fa = traits.Float(
        desc="Nominal preparation/saturation flip angle (degrees) for "
             "pre-computed B1+ map scaling. Read from FlipAngle in the "
             "acq-famp JSON sidecar.",
    )


class B1MappingOutputSpec(TraitedSpec):
    """Output specification for B1Mapping interface."""

    out_file = File(exists=True, desc="B1+ correction factor map")
    mean_b1 = traits.Float(desc="Mean B1+ correction factor within brain mask")
    std_b1 = traits.Float(desc="Standard deviation of B1+ within brain mask")
    qc_pass = traits.Bool(desc="Whether B1+ map passed QC criteria")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")
    qc_plot = File(desc="B1+ mapping QC plot PNG")
    registration_qc_plot = File(
        desc="AFI TR1-TR2 registration QC filmstrip (only if coregistration performed)",
    )


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

    def _load_afi_data(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Any, Any, float, float, dict[str, Any]]:
        """Load AFI data from either a single 4D volume or separate 3D volumes.

        For 4D input, auto-detects TR1/TR2 ordering from JSON sidecar using:
        1. RepetitionTimeExcitation array (BIDS standard for per-volume TRs)
        2. RepetitionTime array (if present)
        3. PerFrameFunctionalGroupsSequence (enhanced DICOM)
        4. Fallback to default TR values (20/100ms) if no JSON metadata

        Returns
        -------
        tuple
            (s1, s2, affine, header, tr1_ms, tr2_ms, coreg_info) where s1 is the short-TR
            signal, s2 is the long-TR signal, TRs are in milliseconds, and coreg_info
            contains coregistration metadata including s1_original, s2_original if
            coregistration was performed.
        """
        import json as _json

        import nibabel as nib  # type: ignore[import-untyped]
        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        afi_4d = self.inputs.afi_4d_file if isdefined(self.inputs.afi_4d_file) else None
        tr1_file = self.inputs.tr1_file if isdefined(self.inputs.tr1_file) else None
        tr2_file = self.inputs.tr2_file if isdefined(self.inputs.tr2_file) else None
        tr1_override = self.inputs.tr1 if isdefined(self.inputs.tr1) else None
        tr2_override = self.inputs.tr2 if isdefined(self.inputs.tr2) else None

        # Reorient AFI files to RAS before loading
        if afi_4d is not None:
            afi_4d_reoriented = _reorient_to_ras(afi_4d)
            if afi_4d_reoriented:
                afi_4d = afi_4d_reoriented
                logger.info("Reoriented 4D AFI to RAS")
        if tr1_file is not None:
            tr1_reoriented = _reorient_to_ras(tr1_file)
            if tr1_reoriented:
                tr1_file = tr1_reoriented
                logger.info("Reoriented AFI TR1 to RAS")
        if tr2_file is not None:
            tr2_reoriented = _reorient_to_ras(tr2_file)
            if tr2_reoriented:
                tr2_file = tr2_reoriented
                logger.info("Reoriented AFI TR2 to RAS")

        if afi_4d is not None:
            # --- 4D mode: single file with 2 volumes ---
            logger.info("Loading 4D AFI: %s", afi_4d)

            afi_img = nib.load(afi_4d)
            afi_data = afi_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            affine = afi_img.affine  # type: ignore[attr-defined]
            header = afi_img.header  # type: ignore[attr-defined]

            if afi_data.ndim != 4:
                raise ValueError(f"Expected 4D AFI volume, got {afi_data.ndim}D")
            if afi_data.shape[3] != 2:
                raise ValueError(
                    f"Expected 2 volumes in 4D AFI, got {afi_data.shape[3]}"
                )

            # Try to read TR values from JSON sidecar
            afi_path = Path(str(afi_4d))
            json_path = Path(
                str(afi_4d).replace(".nii.gz", ".json").replace(".nii", ".json")
            )

            vol0_tr: Optional[float] = None
            vol1_tr: Optional[float] = None

            if json_path.exists():
                logger.info("Reading JSON sidecar: %s", json_path.name)
                with open(json_path) as f:
                    metadata = _json.load(f)

                # Priority 1: RepetitionTimeExcitation (BIDS standard)
                if "RepetitionTimeExcitation" in metadata:
                    tr_exc = metadata["RepetitionTimeExcitation"]
                    if isinstance(tr_exc, list) and len(tr_exc) >= 2:
                        vol0_tr = tr_exc[0] * 1000  # s -> ms
                        vol1_tr = tr_exc[1] * 1000
                        logger.info(
                            "Found RepetitionTimeExcitation: [%.1f, %.1f] ms",
                            vol0_tr, vol1_tr,
                        )

                # Priority 2: RepetitionTime array
                if vol0_tr is None and "RepetitionTime" in metadata:
                    rt = metadata["RepetitionTime"]
                    if isinstance(rt, list) and len(rt) >= 2:
                        vol0_tr = rt[0] * 1000
                        vol1_tr = rt[1] * 1000
                        logger.info(
                            "Found RepetitionTime array: [%.1f, %.1f] ms",
                            vol0_tr, vol1_tr,
                        )

                # Priority 3: PerFrameFunctionalGroupsSequence (enhanced DICOM)
                # NOTE: DICOM RepetitionTime (0018,0080) is natively in ms.
                # dcm2niix does NOT normalize this field when embedded in
                # PerFrameFunctionalGroupsSequence, so values are already in ms.
                if vol0_tr is None and "PerFrameFunctionalGroupsSequence" in metadata:
                    pfgs = metadata["PerFrameFunctionalGroupsSequence"]
                    if isinstance(pfgs, list) and len(pfgs) >= 2:
                        try:
                            vol0_tr = pfgs[0].get(
                                "MRTimingAndRelatedParametersSequence", [{}]
                            )[0].get("RepetitionTime", None)
                            vol1_tr = pfgs[1].get(
                                "MRTimingAndRelatedParametersSequence", [{}]
                            )[0].get("RepetitionTime", None)
                            if vol0_tr is not None and vol1_tr is not None:
                                logger.info(
                                    "Found per-frame TRs: [%.1f, %.1f] ms",
                                    vol0_tr, vol1_tr,
                                )
                        except (KeyError, IndexError, TypeError):
                            pass

            # Apply override values if provided
            if tr1_override is not None and tr2_override is not None:
                logger.info(
                    "Using override TR values: TR1=%.1fms, TR2=%.1fms",
                    tr1_override, tr2_override,
                )
                tr1_ms = float(tr1_override)
                tr2_ms = float(tr2_override)
                s1 = afi_data[:, :, :, 0]
                s2 = afi_data[:, :, :, 1]
            elif vol0_tr is not None and vol1_tr is not None:
                # Auto-detect volume ordering: shorter TR = TR1
                if vol0_tr < vol1_tr:
                    tr1_ms = vol0_tr
                    tr2_ms = vol1_tr
                    s1 = afi_data[:, :, :, 0]
                    s2 = afi_data[:, :, :, 1]
                    logger.info(
                        "Volume ordering: vol0=TR1 (%.1fms), vol1=TR2 (%.1fms)",
                        tr1_ms, tr2_ms,
                    )
                else:
                    tr1_ms = vol1_tr
                    tr2_ms = vol0_tr
                    s1 = afi_data[:, :, :, 1]
                    s2 = afi_data[:, :, :, 0]
                    logger.info(
                        "Volume ordering: vol0=TR2 (%.1fms), vol1=TR1 (%.1fms)",
                        tr2_ms, tr1_ms,
                    )
            else:
                # No TR info found - use defaults, assume standard ordering
                logger.warning(
                    "No TR values in JSON sidecar - using defaults (20/100 ms)"
                )
                tr1_ms = tr1_override if tr1_override is not None else 20.0
                tr2_ms = tr2_override if tr2_override is not None else 100.0
                s1 = afi_data[:, :, :, 0]
                s2 = afi_data[:, :, :, 1]

            # 4D mode: no coregistration needed (same acquisition)
            coreg_info: dict[str, Any] = {"coregistered": False}
            return s1, s2, affine, header, tr1_ms, tr2_ms, coreg_info

        elif tr1_file is not None and tr2_file is not None:
            # --- 3D mode: separate files ---
            logger.info("Loading separate 3D AFI volumes")

            tr1_img = nib.load(tr1_file)
            tr2_img = nib.load(tr2_file)

            s1 = tr1_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            s2 = tr2_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            affine = tr1_img.affine  # type: ignore[attr-defined]
            header = tr1_img.header  # type: ignore[attr-defined]

            # Get TR values: override > JSON sidecar > defaults
            if tr1_override is not None:
                tr1_ms = float(tr1_override)
            else:
                tr1_json = Path(
                    str(tr1_file).replace(".nii.gz", ".json").replace(".nii", ".json")
                )
                if tr1_json.exists():
                    with open(tr1_json) as f:
                        tr1_ms = _json.load(f).get("RepetitionTime", 0.020) * 1000
                else:
                    tr1_ms = 20.0

            if tr2_override is not None:
                tr2_ms = float(tr2_override)
            else:
                tr2_json = Path(
                    str(tr2_file).replace(".nii.gz", ".json").replace(".nii", ".json")
                )
                if tr2_json.exists():
                    with open(tr2_json) as f:
                        tr2_ms = _json.load(f).get("RepetitionTime", 0.100) * 1000
                else:
                    tr2_ms = 100.0

            # Auto-correct TR ordering: TR1 must be the short TR, TR2 the long TR.
            # Files may arrive in arbitrary order (e.g., sorted alphabetically).
            if tr1_ms > tr2_ms:
                logger.info(
                    "Swapping TR ordering: tr1_file had TR=%.1fms > tr2_file TR=%.1fms",
                    tr1_ms, tr2_ms,
                )
                s1, s2 = s2, s1
                tr1_ms, tr2_ms = tr2_ms, tr1_ms
                # Also swap the image references so coregistration uses correct files
                tr1_img, tr2_img = tr2_img, tr1_img
                tr1_file, tr2_file = tr2_file, tr1_file

            # Coregister TR2 → TR1 if affines differ (Gap 5, matches perf-plus).
            # Separate 3D AFI acquisitions may have slight inter-scan motion.
            coreg_info: dict[str, Any] = {"coregistered": False}  # type: ignore[no-redef]
            s1_original = s1.copy()
            s2_original = s2.copy()

            affine_diff = float(np.abs(tr1_img.affine - tr2_img.affine).max())  # type: ignore[attr-defined]
            if affine_diff > 1e-4:
                import subprocess as _sp

                coreg_out = str(Path.cwd() / "afi_tr2_coreg.nii.gz")
                coreg_mat = str(Path.cwd() / "afi_tr2_to_tr1.mat")
                cmd = [
                    "flirt",
                    "-in", str(tr2_file),
                    "-ref", str(tr1_file),
                    "-out", coreg_out,
                    "-omat", coreg_mat,
                    "-dof", "6",
                    "-cost", "normcorr",
                    "-searchrx", "-5", "5",
                    "-searchry", "-5", "5",
                    "-searchrz", "-5", "5",
                ]
                try:
                    _sp.run(cmd, check=True, capture_output=True)
                    s2 = nib.load(coreg_out).get_fdata().astype(np.float64)  # type: ignore[attr-defined]
                    logger.info(
                        "Coregistered AFI TR2 → TR1 (affine diff=%.4f)", affine_diff,
                    )
                    # Track coregistration for QC plot generation
                    coreg_info = {
                        "coregistered": True,
                        "s1_original": s1_original,
                        "s2_original": s2_original,
                        "s2_coregistered": s2.copy(),
                        "affine_diff": affine_diff,
                        "tr1_ms": tr1_ms,
                        "tr2_ms": tr2_ms,
                    }
                except (FileNotFoundError, _sp.CalledProcessError) as exc:
                    logger.warning(
                        "AFI coregistration failed (%s) — using unregistered volumes",
                        exc,
                    )

            return s1, s2, affine, header, tr1_ms, tr2_ms, coreg_info

        else:
            raise ValueError(
                "Must provide either afi_4d_file or both tr1_file and tr2_file"
            )

    def _run_interface(self, runtime: Any) -> Any:
        """Execute B1+ mapping computation."""
        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

        from pmriprep.utils.dce_constants import (
            B1_DENOMINATOR_THRESHOLD,
            B1_EXPECTED_MEAN,
            B1_MIN_COVERAGE,
            B1_MIN_SIGMA_VOXELS,
            B1_RATIO_VALID_RANGE,
            EPSILON_RATIO,
            get_b1_thresholds,
        )

        # Check for pre-computed B1+ map (e.g., tfl_b1map)
        if isdefined(self.inputs.precomputed_b1_file) and self.inputs.precomputed_b1_file:
            return self._run_precomputed_b1(runtime)

        # Generate output filename
        out_file = self._gen_outfilename()

        # Check if B1 mapping should be skipped
        if self.inputs.skip_b1:
            # Try to get shape/affine from AFI data; fall back to brain_mask
            try:
                s1, s2, affine, header, tr1_ms, tr2_ms, _coreg_info = self._load_afi_data()
                shape = s1.shape[:3]
            except ValueError:
                # No AFI data available — use brain mask for shape/affine
                if isdefined(self.inputs.brain_mask) and self.inputs.brain_mask:
                    mask_img = nib.load(self.inputs.brain_mask)
                    shape = mask_img.shape[:3]  # type: ignore[attr-defined]
                    affine = mask_img.affine  # type: ignore[attr-defined]
                elif isdefined(self.inputs.reference_file) and self.inputs.reference_file:
                    ref_img = nib.load(self.inputs.reference_file)
                    shape = ref_img.shape[:3]  # type: ignore[attr-defined]
                    affine = ref_img.affine  # type: ignore[attr-defined]
                else:
                    raise ValueError(
                        "skip_b1=True but no AFI data, brain_mask, or reference_file "
                        "provided to determine output shape"
                    )

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
            # Note: registration_qc_plot left undefined (File traits cannot be None)
            return runtime

        # Load AFI data (supports both 4D and 3D modes)
        s1, s2, affine, header, tr1_ms, tr2_ms, coreg_info = self._load_afi_data()
        shape = s1.shape[:3]

        # Get TR ratio
        n = tr2_ms / tr1_ms
        # Guard against Undefined flowing through workflow connections
        # when afi_flip_angle is not set on the inputnode.
        if isdefined(self.inputs.nominal_flip_angle):
            nominal_fa = float(self.inputs.nominal_flip_angle)
        else:
            nominal_fa = 60.0
        if nominal_fa == 60.0:
            logger.info(
                "Using default AFI nominal flip angle = 60 deg. "
                "Set FlipAngle in BIDS sidecar or afi_flip_angle in config file if different.",
            )

        # -----------------------------------------------------------------
        # Signal masking: only compute B1+ where both signals are above
        # noise floor. Uses 5th-percentile threshold to preserve
        # low-signal dropout regions (Gap 4, matches perf-plus).
        # -----------------------------------------------------------------
        positive_signals = np.concatenate([
            s1[s1 > 0].ravel(), s2[s2 > 0].ravel(),
        ])
        if len(positive_signals) > 0:
            signal_threshold = max(
                float(np.percentile(positive_signals, 5)), 1e-6,
            )
        else:
            signal_threshold = 1e-6
        signal_mask = (s1 > signal_threshold) & (s2 > signal_threshold)

        # -----------------------------------------------------------------
        # Compute signal ratio r = S1/S2
        # -----------------------------------------------------------------
        r = np.ones(shape, dtype=np.float64)
        r[signal_mask] = s1[signal_mask] / np.maximum(s2[signal_mask], EPSILON_RATIO)

        # -----------------------------------------------------------------
        # Ratio validation + denominator stability (Gaps 1, 9).
        # perf-plus: ratio in (0.05, 1.5) and |n - r| > 0.1
        # -----------------------------------------------------------------
        ratio_valid = (
            (r > B1_RATIO_VALID_RANGE[0]) & (r < B1_RATIO_VALID_RANGE[1])
        )
        denominator_stable = np.abs(n - r) > B1_DENOMINATOR_THRESHOLD
        valid = signal_mask & ratio_valid & denominator_stable

        # -----------------------------------------------------------------
        # Simplified Yarnykh (MRM 2007) equation, accurate to first order
        # in TR1/T1:  cos(alpha) = (r*n - 1) / (n - r)
        # Compute only for valid voxels; set the rest to B1+=1.0.
        # -----------------------------------------------------------------
        b1_map = np.ones(shape, dtype=np.float64)

        numerator = r[valid] * n - 1.0
        denominator = n - r[valid]
        cos_alpha = numerator / (denominator + EPSILON_RATIO)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha_actual = np.degrees(np.arccos(cos_alpha))
        b1_map[valid] = alpha_actual / nominal_fa

        # -----------------------------------------------------------------
        # B1+ physiological clipping (Gap 2, field-strength-dependent)
        # -----------------------------------------------------------------
        field_strength = self.inputs.field_strength
        b1_thresh = get_b1_thresholds(field_strength)
        b1_map = np.clip(b1_map, b1_thresh.clip_low, b1_thresh.clip_high)
        b1_map[~signal_mask] = 1.0  # Background = nominal

        # Flag extreme values for QC
        n_extreme_low = int(np.sum((b1_map < b1_thresh.extreme_low) & signal_mask))
        n_extreme_high = int(np.sum((b1_map > b1_thresh.extreme_high) & signal_mask))
        n_signal = max(int(np.sum(signal_mask)), 1)
        if n_extreme_low > 0:
            logger.info(
                "%.1f%% of brain has B1+ < %.2f (expected in temporal lobes at %.1fT)",
                100.0 * n_extreme_low / n_signal,
                b1_thresh.extreme_low,
                field_strength,
            )
        if n_extreme_high > 0:
            logger.warning(
                "%.1f%% of brain has B1+ > %.2f (check for dielectric effects at %.1fT)",
                100.0 * n_extreme_high / n_signal,
                b1_thresh.extreme_high,
                field_strength,
            )

        # -----------------------------------------------------------------
        # Normalized convolution smoothing (Gap 3, matches perf-plus).
        # smooth(B1*mask) / smooth(mask) averages only within brain
        # voxels and avoids bias at brain edges.
        # -----------------------------------------------------------------
        smoothing_fwhm = self.inputs.smoothing_fwhm
        n_b1_outliers = 0
        if smoothing_fwhm > 0:
            voxel_sizes = header.get_zooms()[:3]  # type: ignore[attr-defined]
            sigma_voxels = [
                max(smoothing_fwhm / (2.355 * vs), B1_MIN_SIGMA_VOXELS)
                for vs in voxel_sizes
            ]
            # Pass 1: robust local B1+ from interior voxels.
            # Eroding by 2 voxels excludes partial-volume edges from the
            # reference, so outlier detection at edges uses tissue-only neighbors.
            # Fill internal holes (CSF, noisy B1+) so erosion targets only the
            # outer boundary.  Intersect with valid so only real B1+
            # measurements contribute to the normalized convolution reference.
            # (Robustness improvement for outlier detection; Pass 2 smoothing
            # already handles these voxels.)
            sigma_wide = [s * 1.5 for s in sigma_voxels]
            interior = _b1_interior_mask(valid)
            interior_float = interior.astype(np.float64)
            s_num = gaussian_filter(b1_map * interior_float, sigma=sigma_wide)
            s_den = gaussian_filter(interior_float, sigma=sigma_wide)

            # Only test where interior coverage is reliable.
            # Unreliable locations (thin cortex, brain tips) are left untested.
            has_coverage = s_den > B1_MIN_COVERAGE
            b1_local = np.where(has_coverage, s_num / s_den, 0.0)

            # Flag voxels deviating beyond field-strength-dependent threshold
            # from interior reference (catches CSF/edge artifacts).
            outlier = valid & has_coverage & (
                np.abs(b1_map - b1_local) > b1_thresh.outlier_deviation
            )
            n_b1_outliers = int(np.sum(outlier))
            if n_b1_outliers > 0:
                logger.info(
                    "B1+ outlier rejection: excluded %d voxels (%.1f%% of valid)",
                    n_b1_outliers, 100.0 * n_b1_outliers / max(int(np.sum(valid)), 1),
                )
            valid = valid & ~outlier
            b1_map[outlier] = 1.0

            # Pass 2: re-smooth with cleaned mask (standard sigma)
            mask_float = valid.astype(np.float64)
            smoothed_num = gaussian_filter(b1_map * mask_float, sigma=sigma_voxels)
            smoothed_den = gaussian_filter(mask_float, sigma=sigma_voxels)
            smoothed_den = np.maximum(smoothed_den, 1e-6)
            b1_map = np.where(mask_float > 0, smoothed_num / smoothed_den, 1.0)
            b1_map = np.clip(b1_map, b1_thresh.clip_low, b1_thresh.clip_high)

        # Convert to float32 for storage
        b1_map = b1_map.astype(np.float32)

        # -----------------------------------------------------------------
        # Detect uniform (pathological) B1+ maps (Gap 7).
        # If all brain voxels are zero or all ones, the map is unusable.
        # -----------------------------------------------------------------
        uniform_detected = False
        b1_unique = np.unique(b1_map[signal_mask])
        if len(b1_unique) <= 1:
            if len(b1_unique) == 0 or np.isclose(b1_unique[0], 0.0, atol=1e-6):
                logger.warning("B1+ map is all zeros — returning unity map")
                b1_map = np.ones(shape, dtype=np.float32)
                uniform_detected = True
            elif np.isclose(b1_unique[0], 1.0, atol=1e-3):
                logger.info("B1+ map is all ones (no correction needed)")
                uniform_detected = True

        # Load brain mask for QC if provided.
        # Brain mask may be in T1w space (different resolution from AFI).
        # Following perf-plus: require exact shape+affine match, ignore mask
        # entirely if mismatched (no resampling for QC).
        mask: Optional[np.ndarray] = None
        if self.inputs.brain_mask:
            mask_img = nib.load(self.inputs.brain_mask)
            mask_data = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]
            mask_affine = mask_img.affine  # type: ignore[attr-defined]

            shape_match = mask_data.shape[:3] == b1_map.shape[:3]
            affine_diff = float(np.abs(affine - mask_affine).max())

            if not shape_match:
                logger.warning(
                    "Brain mask shape %s != B1+ map shape %s — ignoring mask",
                    mask_data.shape[:3], b1_map.shape[:3],
                )
            elif affine_diff > 1e-4:
                logger.warning(
                    "Brain mask affine mismatch (max diff=%.6f) — ignoring mask",
                    affine_diff,
                )
            else:
                mask = mask_data

        # Calculate QC metrics within brain mask (or whole volume if no mask)
        if mask is not None:
            b1_in_roi = b1_map[mask]
        else:
            # No usable mask — use all non-zero voxels
            b1_in_roi = b1_map[b1_map > 0]

        # Remove outliers for robust statistics
        b1_clean = b1_in_roi[
            (b1_in_roi > 0.5) & (b1_in_roi < 2.0)
        ]

        if len(b1_clean) > 0:
            mean_b1 = float(np.mean(b1_clean))
            std_b1 = float(np.std(b1_clean))
        else:
            mean_b1 = B1_EXPECTED_MEAN
            std_b1 = 0.0

        # QC pass/fail criteria (field-strength-dependent)
        qc_pass = (
            b1_thresh.mean_range[0] <= mean_b1 <= b1_thresh.mean_range[1]
            and std_b1 < b1_thresh.max_std
        )

        # Compute additional B1 metrics
        median_b1 = float(np.median(b1_clean)) if len(b1_clean) > 0 else B1_EXPECTED_MEAN

        # B1 QC status: PASS or WARN only (no FAIL for B1 quality)
        from pmriprep.utils.qc_standards import QCStatus
        b1_status = QCStatus.PASS if qc_pass else QCStatus.WARN

        # Build QC metrics dictionary
        qc_metrics = {
            "method": "afi",
            "mean_b1": mean_b1,
            "std_b1": std_b1,
            "median_b1": median_b1,
            "b1_status": b1_status,
            "qc_pass": qc_pass,
            "skipped": False,
            "smoothing_fwhm_mm": smoothing_fwhm,
            "nominal_flip_angle_deg": nominal_fa,
            "tr1_ms": tr1_ms,
            "tr2_ms": tr2_ms,
            "tr_ratio": n,
            "b1_range_expected": b1_thresh.mean_range,
            "b1_max_std_expected": b1_thresh.max_std,
            "n_signal_voxels": n_signal,
            "n_valid_voxels": int(np.sum(valid)),
            "n_ratio_invalid": int(np.sum(signal_mask & ~ratio_valid)),
            "n_denominator_unstable": int(np.sum(signal_mask & ~denominator_stable)),
            "n_extreme_low_b1": n_extreme_low,
            "n_extreme_high_b1": n_extreme_high,
            "n_b1_outliers_rejected": n_b1_outliers,
            "uniform_detected": uniform_detected,
            "pct_correction_gt_10": (
                float(100.0 * np.sum(np.abs(b1_clean - 1.0) > 0.1) / len(b1_clean))
                if len(b1_clean) > 0 else 0.0
            ),
            "pct_correction_gt_20": (
                float(100.0 * np.sum(np.abs(b1_clean - 1.0) > 0.2) / len(b1_clean))
                if len(b1_clean) > 0 else 0.0
            ),
        }

        # Save output B1+ map
        out_img = nib.Nifti1Image(b1_map, affine, header)
        nib.save(out_img, out_file)

        # Generate QC plot
        qc_plot_path: Optional[str] = None
        try:
            from pmriprep.utils.dce_qc_plots import generate_b1_qc_plot

            qc_plot_path = generate_b1_qc_plot(
                s1=s1,
                s2=s2,
                signal_ratio=r,
                b1_map=b1_map,
                nominal_flip_angle=nominal_fa,
                signal_mask=signal_mask,
                valid_mask=valid,
                mean_b1=mean_b1,
                std_b1=std_b1,
                subject_id=self.inputs.subject_id,
                session_id=self.inputs.session_id,
            )
        except Exception as e:
            logger.warning("Failed to generate B1+ QC plot: %s", e, exc_info=True)

        # Generate AFI registration QC plot (only if coregistration was performed)
        reg_qc_plot_path: Optional[str] = None
        if coreg_info.get("coregistered", False):
            try:
                from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip

                # Use s1 (reference) and s2_original (before coregistration)
                s1_ref = coreg_info.get("s1_original", s1)
                s2_orig = coreg_info.get("s2_original", s2)
                s2_coreg = coreg_info.get("s2_coregistered", s2)
                tr1_val = coreg_info.get("tr1_ms", tr1_ms)
                tr2_val = coreg_info.get("tr2_ms", tr2_ms)

                reg_qc_path = Path.cwd() / "afi_registration_qc.png"
                plot_registration_qc_filmstrip(
                    fixed=s1_ref,
                    moving=s2_coreg,
                    output_path=reg_qc_path,
                    title=f"AFI Registration QC (ref=TR1={tr1_val:.0f}ms, moving=TR2={tr2_val:.0f}ms)",
                )
                reg_qc_plot_path = str(reg_qc_path)
                logger.info("Generated AFI registration QC plot: %s", reg_qc_plot_path)
            except Exception as exc:
                logger.warning("Failed to generate AFI registration QC plot: %s", exc)

        # Set outputs
        self._results["out_file"] = out_file
        self._results["mean_b1"] = mean_b1
        self._results["std_b1"] = std_b1
        self._results["qc_pass"] = qc_pass
        self._results["qc_metrics"] = qc_metrics
        if qc_plot_path is not None:
            self._results["qc_plot"] = qc_plot_path
        if reg_qc_plot_path is not None:
            self._results["registration_qc_plot"] = reg_qc_plot_path

        return runtime

    def _run_precomputed_b1(self, runtime: Any) -> Any:
        """Process pre-computed B1+ flip angle map (e.g., Siemens tfl_b1map).

        Scaling: b1_ratio = pixel_value / (nominal_prep_fa * 10)
        where pixel values are in 0.1 degree units (Siemens convention).
        """
        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

        from pmriprep.utils.dce_constants import (
            B1_EXPECTED_MEAN,
            B1_MIN_COVERAGE,
            B1_MIN_SIGMA_VOXELS,
            get_b1_thresholds,
        )

        out_file = self._gen_outfilename()

        # Load pre-computed B1+ map
        b1_img = nib.load(self.inputs.precomputed_b1_file)

        # Reorient to RAS (no-op if already RAS).
        # Uses as_reoriented() which correctly preserves the world-space
        # affine — the previous inv_ornt_aff() approach dropped it.
        from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

        orig_ornt = io_orientation(b1_img.affine)  # type: ignore[attr-defined]
        ras_ornt = axcodes2ornt(("R", "A", "S"))
        if not np.array_equal(orig_ornt, ras_ornt):
            transform = ornt_transform(orig_ornt, ras_ornt)
            b1_img = b1_img.as_reoriented(transform)  # type: ignore[attr-defined]
            logger.info("Reoriented pre-computed B1+ map to RAS")

        b1_raw = np.asarray(b1_img.dataobj, dtype=np.float64)  # type: ignore[attr-defined]
        affine = b1_img.affine  # type: ignore[attr-defined]
        header = b1_img.header  # type: ignore[attr-defined]
        shape = b1_raw.shape[:3]

        # Read nominal preparation flip angle
        if not isdefined(self.inputs.b1_nominal_prep_fa) or not self.inputs.b1_nominal_prep_fa:
            raise ValueError(
                "b1_nominal_prep_fa is required for pre-computed B1+ maps. "
                "This should be set from the FlipAngle field in the acq-famp "
                "JSON sidecar."
            )
        nominal_prep_fa = float(self.inputs.b1_nominal_prep_fa)
        scale_factor = nominal_prep_fa * 10.0
        logger.info(
            "Pre-computed B1+ scaling: pixel / %.1f "
            "(FlipAngle=%.0f°, scale=FA×10)",
            scale_factor, nominal_prep_fa,
        )

        # Validate pixel values — catch grossly wrong files (e.g., anatomical
        # companion with signal-intensity values), but allow high B1+ at FOV
        # edges. Downstream physiological clipping handles the actual range.
        p99 = float(np.percentile(b1_raw[b1_raw > 0], 99)) if np.any(b1_raw > 0) else 0.0
        if p99 > scale_factor * 5.0:
            raise ValueError(
                f"Pre-computed B1+ map pixel p99={p99:.0f} exceeds "
                f"5× expected maximum ({scale_factor * 5.0:.0f}). "
                f"Check that this is the correct file (acq-famp, not acq-anat)."
            )
        if p99 > scale_factor * 2.0:
            logger.warning(
                "Pre-computed B1+ map pixel p99=%.0f exceeds 2× nominal "
                "(%.0f). Some voxels have B1+ ratio > 2.0 — likely edge/artifact "
                "voxels that will be clipped.",
                p99, scale_factor * 2.0,
            )

        # Create valid mask (non-zero pixels)
        valid_mask = b1_raw > 0

        # Scale to B1+ ratio
        b1_map = np.ones(shape, dtype=np.float64)
        b1_map[valid_mask] = b1_raw[valid_mask] / scale_factor

        # -----------------------------------------------------------------
        # Physiological clipping (field-strength-dependent)
        # -----------------------------------------------------------------
        field_strength = self.inputs.field_strength
        b1_thresh = get_b1_thresholds(field_strength)
        b1_map = np.clip(b1_map, b1_thresh.clip_low, b1_thresh.clip_high)
        b1_map[~valid_mask] = 1.0  # Background = nominal

        # Flag extreme values for QC
        n_valid = max(int(np.sum(valid_mask)), 1)
        n_extreme_low = int(np.sum((b1_map < b1_thresh.extreme_low) & valid_mask))
        n_extreme_high = int(np.sum((b1_map > b1_thresh.extreme_high) & valid_mask))
        if n_extreme_low > 0:
            logger.info(
                "%.1f%% of brain has B1+ < %.2f (expected in temporal lobes at %.1fT)",
                100.0 * n_extreme_low / n_valid,
                b1_thresh.extreme_low,
                field_strength,
            )
        if n_extreme_high > 0:
            logger.warning(
                "%.1f%% of brain has B1+ > %.2f (check for dielectric effects at %.1fT)",
                100.0 * n_extreme_high / n_valid,
                b1_thresh.extreme_high,
                field_strength,
            )

        # -----------------------------------------------------------------
        # Normalized convolution smoothing (same as AFI path)
        # -----------------------------------------------------------------
        smoothing_fwhm = self.inputs.smoothing_fwhm
        n_b1_outliers = 0
        if smoothing_fwhm > 0:
            voxel_sizes = header.get_zooms()[:3]  # type: ignore[attr-defined]
            sigma_voxels = [
                max(smoothing_fwhm / (2.355 * vs), B1_MIN_SIGMA_VOXELS)
                for vs in voxel_sizes
            ]
            # Pass 1: robust local B1+ from interior voxels.
            # Fill internal holes so erosion targets only the outer boundary.
            # Intersect with valid_mask so only real B1+ measurements contribute.
            sigma_wide = [s * 1.5 for s in sigma_voxels]
            interior = _b1_interior_mask(valid_mask)
            interior_float = interior.astype(np.float64)
            s_num = gaussian_filter(b1_map * interior_float, sigma=sigma_wide)
            s_den = gaussian_filter(interior_float, sigma=sigma_wide)
            has_coverage = s_den > B1_MIN_COVERAGE
            b1_local = np.where(has_coverage, s_num / s_den, 0.0)

            outlier = valid_mask & has_coverage & (
                np.abs(b1_map - b1_local) > b1_thresh.outlier_deviation
            )
            n_b1_outliers = int(np.sum(outlier))
            if n_b1_outliers > 0:
                logger.info(
                    "B1+ outlier rejection: excluded %d voxels (%.1f%% of valid)",
                    n_b1_outliers, 100.0 * n_b1_outliers / n_valid,
                )
            valid_mask = valid_mask & ~outlier
            b1_map[outlier] = 1.0

            # Pass 2: re-smooth with cleaned mask
            mask_float = valid_mask.astype(np.float64)
            smoothed_num = gaussian_filter(b1_map * mask_float, sigma=sigma_voxels)
            smoothed_den = gaussian_filter(mask_float, sigma=sigma_voxels)
            smoothed_den = np.maximum(smoothed_den, 1e-6)
            b1_map = np.where(mask_float > 0, smoothed_num / smoothed_den, 1.0)
            b1_map = np.clip(b1_map, b1_thresh.clip_low, b1_thresh.clip_high)

        b1_map = b1_map.astype(np.float32)

        # -----------------------------------------------------------------
        # QC metrics
        # -----------------------------------------------------------------
        b1_in_roi = b1_map[valid_mask]
        b1_clean = b1_in_roi[(b1_in_roi > 0.5) & (b1_in_roi < 2.0)]

        if len(b1_clean) > 0:
            mean_b1 = float(np.mean(b1_clean))
            std_b1 = float(np.std(b1_clean))
            median_b1 = float(np.median(b1_clean))
        else:
            mean_b1 = B1_EXPECTED_MEAN
            std_b1 = 0.0
            median_b1 = B1_EXPECTED_MEAN

        qc_pass = (
            b1_thresh.mean_range[0] <= mean_b1 <= b1_thresh.mean_range[1]
            and std_b1 < b1_thresh.max_std
        )

        from pmriprep.utils.qc_standards import QCStatus
        b1_status = QCStatus.PASS if qc_pass else QCStatus.WARN

        qc_metrics = {
            "method": "tfl_b1map",
            "mean_b1": mean_b1,
            "std_b1": std_b1,
            "median_b1": median_b1,
            "b1_status": b1_status,
            "qc_pass": qc_pass,
            "skipped": False,
            "smoothing_fwhm_mm": smoothing_fwhm,
            "nominal_prep_fa_deg": nominal_prep_fa,
            "scale_factor": scale_factor,
            "pixel_p99": p99,
            "b1_range_expected": b1_thresh.mean_range,
            "b1_max_std_expected": b1_thresh.max_std,
            "n_valid_voxels": int(np.sum(valid_mask)),
            "n_extreme_low_b1": n_extreme_low,
            "n_extreme_high_b1": n_extreme_high,
            "n_b1_outliers_rejected": n_b1_outliers,
            "pct_correction_gt_10": (
                float(100.0 * np.sum(np.abs(b1_clean - 1.0) > 0.1) / len(b1_clean))
                if len(b1_clean) > 0 else 0.0
            ),
            "pct_correction_gt_20": (
                float(100.0 * np.sum(np.abs(b1_clean - 1.0) > 0.2) / len(b1_clean))
                if len(b1_clean) > 0 else 0.0
            ),
        }

        # Save output
        out_img = nib.Nifti1Image(b1_map, affine, header)
        nib.save(out_img, out_file)

        # Generate QC plot
        qc_plot_path: Optional[str] = None
        try:
            from pmriprep.utils.dce_qc_plots import generate_b1_precomputed_qc_plot

            qc_plot_path = generate_b1_precomputed_qc_plot(
                b1_raw=b1_raw,
                b1_map=b1_map,
                valid_mask=valid_mask,
                nominal_prep_fa=nominal_prep_fa,
                mean_b1=mean_b1,
                std_b1=std_b1,
                subject_id=self.inputs.subject_id,
                session_id=self.inputs.session_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to generate pre-computed B1+ QC plot: %s", e, exc_info=True,
            )

        # Set outputs
        self._results["out_file"] = out_file
        self._results["mean_b1"] = mean_b1
        self._results["std_b1"] = std_b1
        self._results["qc_pass"] = qc_pass
        self._results["qc_metrics"] = qc_metrics
        if qc_plot_path is not None:
            self._results["qc_plot"] = qc_plot_path

        return runtime

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified."""
        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        if isdefined(self.inputs.out_file) and self.inputs.out_file:
            return str(self.inputs.out_file)

        # Determine source file for naming
        if isdefined(self.inputs.precomputed_b1_file) and self.inputs.precomputed_b1_file:
            in_file = Path(str(self.inputs.precomputed_b1_file))
        elif isdefined(self.inputs.afi_4d_file) and self.inputs.afi_4d_file:
            in_file = Path(str(self.inputs.afi_4d_file))
        elif isdefined(self.inputs.tr1_file) and self.inputs.tr1_file:
            in_file = Path(self.inputs.tr1_file)
        else:
            # No input file available (e.g., skip_b1 with no AFI data)
            return str(Path.cwd() / "B1plus.nii.gz")

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

        # Remove any B1-specific suffixes and add B1plus suffix
        base = base.replace("_AFI", "").replace("_afi", "")
        base = base.replace("_TR1", "").replace("_tr1", "")
        base = base.replace("_TB1TFL", "").replace("_tb1tfl", "")
        base = base.replace("_TB1AFI", "").replace("_tb1afi", "")

        return str(Path.cwd() / f"{base}_B1plus{ext}")


class T1MappingInputSpec(BaseInterfaceInputSpec):
    """Input specification for T1Mapping interface."""

    vfa_files = traits.List(
        File(exists=True),
        mandatory=True,
        minlen=2,
        maxlen=20,
        desc="List of VFA images at different flip angles",
    )
    flip_angles = traits.List(
        traits.Float,
        mandatory=True,
        minlen=2,
        maxlen=20,
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
    csf_probseg = File(
        exists=True,
        desc="CSF probability map for excluding CSF voxels from T1 fitting",
    )
    wm_probseg = File(
        exists=True,
        desc="WM probability map for tissue-segmented T1 fallback (optional)",
    )
    gm_probseg = File(
        exists=True,
        desc="GM probability map for tissue-segmented T1 fallback (optional)",
    )
    csf_threshold = traits.Float(
        0.3,
        usedefault=True,
        desc="CSF probability threshold for exclusion (default: 0.3 = 30%)",
    )
    t1_correction_mode = traits.Enum(
        "auto",
        "none",
        "offset",
        "fallback",
        usedefault=True,
        desc="T1 correction mode: auto, none, offset, or fallback",
    )
    b1_qc_pass = traits.Bool(
        True,
        usedefault=True,
        desc="B1 QC pass status (from B1Mapping). When False, lowers "
             "T1 correction threshold for auto mode.",
    )
    subject_id = traits.Str(
        "",
        usedefault=True,
        desc="Subject ID for QC figure title",
    )
    session_id = traits.Str(
        "",
        usedefault=True,
        desc="Session ID for QC figure title",
    )


class T1MappingOutputSpec(TraitedSpec):
    """Output specification for T1Mapping interface."""

    out_file = File(exists=True, desc="T1 map in milliseconds")
    mean_t1_wm = traits.Float(desc="Mean T1 in white matter region (ms)")
    mean_t1_gm = traits.Float(desc="Mean T1 in gray matter region (ms)")
    qc_pass = traits.Bool(desc="Whether T1 map passed QC criteria")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")
    qc_plot = File(desc="VFA T1 mapping QC plot PNG")


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
        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        from pmriprep.utils.dce_constants import (
            DEFAULT_T1_MS,
            EPSILON_TRIG,
            LENIENT_T1_RANGE,
            SLAB_EROSION_MAX_SLICES,
            get_b1_thresholds,
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
            from pmriprep.utils.dce_t1_validation import (
                T1_TISSUE_REFERENCE,
                build_tissue_t1_map,
                get_field_strength_key,
            )

            field_key = get_field_strength_key(self.inputs.field_strength)
            default_t1 = T1_TISSUE_REFERENCE[field_key]["default"]
            tissue_t1_used = False

            # Use tissue-segmented T1 if probsegs are available
            if (
                isdefined(self.inputs.wm_probseg) and self.inputs.wm_probseg
                and isdefined(self.inputs.gm_probseg) and self.inputs.gm_probseg
            ):
                wm_img = nib.load(self.inputs.wm_probseg)
                gm_img = nib.load(self.inputs.gm_probseg)
                # Resample to VFA space if needed
                if wm_img.shape[:3] != shape:  # type: ignore[attr-defined]
                    from nilearn.image import resample_to_img  # type: ignore[import-untyped]
                    wm_img = resample_to_img(wm_img, first_img, interpolation="continuous")
                    gm_img = resample_to_img(gm_img, first_img, interpolation="continuous")
                wm_arr = wm_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
                gm_arr = gm_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
                t1_map = build_tissue_t1_map(
                    wm_arr, gm_arr, self.inputs.field_strength,
                )
                tissue_t1_used = True
                logger.info(
                    "skip_t1: using tissue-segmented T1 map (WM=%.0fms, GM=%.0fms)",
                    float((T1_TISSUE_REFERENCE[field_key]["wm"][0] + T1_TISSUE_REFERENCE[field_key]["wm"][1]) / 2),
                    float((T1_TISSUE_REFERENCE[field_key]["gm"][0] + T1_TISSUE_REFERENCE[field_key]["gm"][1]) / 2),
                )
            else:
                # Uniform fallback
                t1_map = np.full(shape, default_t1, dtype=np.float32)

            # Save output
            out_img = nib.Nifti1Image(t1_map, affine)
            nib.save(out_img, out_file)

            # Set outputs for skipped case
            self._results["out_file"] = out_file
            if tissue_t1_used:
                wm_mid = float((T1_TISSUE_REFERENCE[field_key]["wm"][0] + T1_TISSUE_REFERENCE[field_key]["wm"][1]) / 2)
                gm_mid = float((T1_TISSUE_REFERENCE[field_key]["gm"][0] + T1_TISSUE_REFERENCE[field_key]["gm"][1]) / 2)
            else:
                wm_mid = default_t1
                gm_mid = default_t1
            self._results["mean_t1_wm"] = wm_mid
            self._results["mean_t1_gm"] = gm_mid
            self._results["qc_pass"] = True
            self._results["qc_metrics"] = {
                "mean_t1_wm_ms": wm_mid,
                "mean_t1_gm_ms": gm_mid,
                "qc_pass": True,
                "skipped": True,
                "tissue_t1_used": tissue_t1_used,
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
        # Clip to VFA-appropriate range to allow dielectric resonance
        # effects while preventing extreme corrections (Gap 8).
        b1_thresh = get_b1_thresholds(self.inputs.field_strength)
        if isdefined(self.inputs.b1_map) and self.inputs.b1_map:
            b1_img = nib.load(self.inputs.b1_map)
            b1_map = b1_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            # Mark voxels with no B1 data BEFORE clipping so they aren't
            # silently mapped to the clip minimum (which halves flip angles).
            b1_missing = b1_map <= 0
            b1_map = np.clip(b1_map, b1_thresh.vfa_clip_low, b1_thresh.vfa_clip_high)
            b1_map[b1_missing] = 1.0  # No B1 data → nominal flip angles
        else:
            b1_map = np.ones(shape, dtype=np.float64)

        # Apply B1+ correction to flip angles
        # α_actual = α_nominal × B1+
        # Shape: (n_fa, x, y, z) after broadcasting
        corrected_fa_rad = flip_angles_rad[:, np.newaxis, np.newaxis, np.newaxis] * b1_map

        # Calculate sin and tan of corrected flip angles
        sin_fa = np.sin(corrected_fa_rad)
        tan_fa = np.tan(corrected_fa_rad)

        # Prevent division by zero (use tight epsilon; B1+ floor of 0.5
        # ensures corrected angles >= ~1° so sin/tan are well above EPSILON_TRIG)
        sin_fa = np.where(np.abs(sin_fa) < EPSILON_TRIG, EPSILON_TRIG, sin_fa)
        tan_fa = np.where(np.abs(tan_fa) < EPSILON_TRIG, EPSILON_TRIG, tan_fa)

        # Calculate Y = S/sin(α) and X = S/tan(α) for DESPOT1 linearization
        y_data = signal_data / sin_fa  # Shape: (n_fa, x, y, z)
        x_data = signal_data / tan_fa  # Shape: (n_fa, x, y, z)

        # Perform linear regression: Y = m × X + b
        # Using least squares for each voxel
        # m = E1, b = S0 × (1 - E1)
        cos_fa = np.cos(corrected_fa_rad)
        t1_map = self._fit_t1_wlls(
            x_data, y_data, sin_fa, cos_fa, tr_s, shape,
        )

        # Convert T1 from seconds to milliseconds
        t1_map = t1_map * 1000.0

        # Load brain mask for QC if provided
        if isdefined(self.inputs.brain_mask) and self.inputs.brain_mask:
            mask_img = nib.load(self.inputs.brain_mask)
            # Resample mask to VFA space if dimensions differ (e.g. T1w MPRAGE vs VFA VIBE)
            if mask_img.shape[:3] != shape:  # type: ignore[attr-defined]
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]

                mask_img = resample_to_img(
                    mask_img, first_img, interpolation="nearest",
                )
            brain_mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]
        else:
            # Use simple thresholding on first VFA image as rough brain mask
            threshold = np.percentile(signal_data[0][signal_data[0] > 0], 10)
            brain_mask = signal_data[0] > threshold

        # Set background voxels to NaN BEFORE median filter to prevent
        # finite T1 values from E1-clipped background from contaminating
        # brain-boundary voxels through the 3x3x3 kernel.
        t1_map[~brain_mask] = np.nan

        # Apply 3x3x3 median filter to reduce noise spikes.
        # Background NaN voxels are replaced with the brain-median sentinel
        # so they cast "neutral" votes in the kernel — boundary voxels with
        # a majority of real brain neighbors are preserved (unlike a -1.0
        # sentinel which poisons ~15-20K boundary voxels via median pull-down).
        # After filtering, original NaN positions are restored.
        # Applied BEFORE CSF exclusion to avoid introducing artificial zeros
        # that would contaminate periventricular tissue during filtering.
        # Matches perf-plus order: median filter → CSF exclusion → range clipping.
        from scipy.ndimage import median_filter  # type: ignore[import-untyped]

        nan_mask = np.isnan(t1_map)
        valid_vals = t1_map[~nan_mask]
        sentinel = float(np.median(valid_vals)) if len(valid_vals) > 0 else 1000.0
        t1_for_filter = np.where(nan_mask, sentinel, t1_map)
        t1_map = median_filter(t1_for_filter, size=3)
        t1_map[nan_mask] = np.nan

        # CSF exclusion: mask out voxels with high CSF probability
        csf_excluded = False
        if isdefined(self.inputs.csf_probseg) and self.inputs.csf_probseg:
            csf_img = nib.load(self.inputs.csf_probseg)
            # Resample CSF probseg to VFA space if needed
            if csf_img.shape[:3] != shape:  # type: ignore[attr-defined]
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]

                csf_img = resample_to_img(
                    csf_img, first_img, interpolation="continuous",
                )
            csf_prob = csf_img.get_fdata()  # type: ignore[attr-defined]
            csf_mask = csf_prob >= self.inputs.csf_threshold

            n_brain_before = int(np.sum(brain_mask))
            n_csf = int(np.sum(csf_mask & brain_mask))
            # Set CSF voxels to NaN (excluded from all downstream statistics)
            t1_map[csf_mask & brain_mask] = np.nan
            brain_mask = brain_mask & (~csf_mask)
            n_brain_after = int(np.sum(brain_mask))

            logger.info(
                "CSF exclusion: removed %d voxels (%.1f%% of brain), "
                "%d tissue voxels remaining",
                n_csf,
                100.0 * n_csf / max(n_brain_before, 1),
                n_brain_after,
            )
            csf_excluded = True

        # Set out-of-range T1 to NaN using field-strength-dependent bounds.
        # Lower bound (reject_min) depends on field strength: 450/600/800ms at 1.5/3/7T.
        # Upper bound depends on field strength and whether CSF was excluded:
        #   - With CSF exclusion: use full t1_upper_bound (allows some CSF partial volume)
        #   - Without CSF exclusion: use tighter t1_parenchyma_upper to reject CSF voxels
        field_strength = self.inputs.field_strength
        correction_mode = self.inputs.t1_correction_mode

        from pmriprep.utils.dce_t1_validation import (
            T1_TISSUE_REFERENCE,
            apply_t1_correction,
            get_field_strength_key,
        )

        field_key = get_field_strength_key(field_strength)
        if csf_excluded:
            t1_upper = T1_TISSUE_REFERENCE[field_key]["t1_upper_bound"]
        else:
            t1_upper = T1_TISSUE_REFERENCE[field_key]["t1_parenchyma_upper"]
            logger.info(
                "No CSF probseg: using tighter T1 upper bound (%.0fms)", t1_upper,
            )
        t1_lower = T1_TISSUE_REFERENCE[field_key]["reject_min"]
        t1_map = np.where(
            (t1_map >= t1_lower) & (t1_map <= t1_upper), t1_map, np.nan,
        )

        # Erode valid-T1 region BEFORE T1 correction so the correction median
        # reflects interior brain tissue, not partial-volume edge voxels.
        # 6-connectivity (face-adjacent only): MRI PSF is separable along
        # encoding axes, so contamination is strongest face-adjacent.
        # Low T1 (~500ms) causes ~2.5x Ktrans overestimation.
        from scipy.ndimage import binary_erosion, binary_fill_holes, generate_binary_structure  # type: ignore[import-untyped]

        valid_t1_mask = np.isfinite(t1_map) & (t1_map > 0)
        # Fill internal holes (from CSF exclusion, range clipping, noisy VFA)
        # so erosion only targets the outer brain-CSF/skull boundary where
        # partial-volume contamination actually occurs.
        valid_filled = binary_fill_holes(valid_t1_mask)

        # Defensive check: slab-aware erosion assumes axis 2 = slices (RAS).
        # The workflow guarantees RAS via reorientation nodes, but warn if
        # the interface is used directly with non-RAS data.
        from nibabel.orientations import axcodes2ornt, io_orientation

        ornt = io_orientation(first_img.affine)  # type: ignore[attr-defined]
        ras_ornt = axcodes2ornt(("R", "A", "S"))
        if not np.array_equal(ornt, ras_ornt):
            logger.warning(
                "T1Mapping input is not RAS-oriented; slab-aware erosion "
                "assumes axis 2 is the slice axis. Results may be incorrect."
            )

        # Slab-aware erosion: for thin 3D slabs (≤SLAB_EROSION_MAX_SLICES),
        # through-plane erosion is not physically motivated — 3D VIBE has
        # well-defined slice profile within the slab, so partial volume is
        # an in-plane phenomenon. Through-plane erosion of a 24-slice slab
        # loses 2 slices (8% of z-coverage) for no benefit.
        n_slices = shape[2]  # RAS orientation: axis 2 = slices
        if n_slices <= SLAB_EROSION_MAX_SLICES:
            struct_2d = generate_binary_structure(2, 1)  # 4-connectivity in-plane
            eroded = np.zeros_like(valid_filled)
            for z in range(n_slices):
                eroded[:, :, z] = binary_erosion(
                    valid_filled[:, :, z], structure=struct_2d, iterations=1,
                )
        else:
            struct = generate_binary_structure(3, 1)  # 6-connectivity
            eroded = binary_erosion(valid_filled, structure=struct, iterations=1)
        # Erode the filled volume, then remove original valid voxels that fall
        # within the erosion shell.  Holes (filled but not originally valid) are
        # already NaN, so they need no further masking.
        edge_mask = valid_t1_mask & ~eroded
        n_edge = int(np.sum(edge_mask))
        t1_map[edge_mask] = np.nan
        # Update brain_mask to match (mirrors CSF exclusion pattern at line 1514)
        brain_mask = brain_mask & ~edge_mask
        logger.info(
            "T1 edge erosion: removed %d partial-volume voxels "
            "(%.1f%% of brain mask)",
            n_edge, 100.0 * n_edge / max(int(np.sum(brain_mask) + n_edge), 1),
        )

        # Convert to float32 for storage
        t1_map = t1_map.astype(np.float32)

        # Compute NaN fraction in brain before correction
        nan_in_brain = np.isnan(t1_map) & brain_mask
        nan_fraction = float(np.sum(nan_in_brain)) / max(1, int(np.sum(brain_mask)))

        # VFA conditioning QC metric
        # Compute median T1 from valid voxels for E1 calculation
        valid_for_e1 = t1_map[brain_mask & np.isfinite(t1_map) & (t1_map > 0)]
        if len(valid_for_e1) > 0:
            median_t1_original = float(np.median(valid_for_e1))
            e1_median = float(np.exp(-tr_s / (median_t1_original / 1000.0)))
        else:
            median_t1_original = 0.0
            e1_median = 0.99  # default to moderately ill-conditioned

        if e1_median > 0.995:
            vfa_conditioning = "severe"
        elif e1_median > 0.99:
            vfa_conditioning = "poor"
        else:
            vfa_conditioning = "good"

        logger.info(
            "VFA conditioning: E1=%.4f (%s), NaN fraction=%.1f%%",
            e1_median, vfa_conditioning, nan_fraction * 100,
        )

        # Load WM/GM probsegs for tissue-segmented fallback
        wm_probseg_arr: Optional[np.ndarray] = None
        gm_probseg_arr: Optional[np.ndarray] = None
        if (
            isdefined(self.inputs.wm_probseg) and self.inputs.wm_probseg
            and isdefined(self.inputs.gm_probseg) and self.inputs.gm_probseg
        ):
            wm_ps_img = nib.load(self.inputs.wm_probseg)
            gm_ps_img = nib.load(self.inputs.gm_probseg)
            if wm_ps_img.shape[:3] != shape:  # type: ignore[attr-defined]
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]
                wm_ps_img = resample_to_img(wm_ps_img, first_img, interpolation="continuous")
                gm_ps_img = resample_to_img(gm_ps_img, first_img, interpolation="continuous")
            wm_probseg_arr = wm_ps_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            gm_probseg_arr = gm_ps_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]

        t1_map_arr, correction_result = apply_t1_correction(
            t1_map,
            field_strength=field_strength,
            correction_mode=correction_mode,
            brain_mask=brain_mask,
            b1_qc_pass=self.inputs.b1_qc_pass,
            nan_fraction=nan_fraction,
            wm_probseg=wm_probseg_arr,
            gm_probseg=gm_probseg_arr,
            e1_median=e1_median,
        )
        t1_map = t1_map_arr.astype(np.float32)
        correction_info = correction_result.to_dict()
        logger.info(
            "T1 correction: %s (median %.0fms -> %.0fms)",
            correction_result.correction_applied,
            correction_result.median_t1_original,
            correction_result.median_t1_corrected,
        )

        # Calculate QC metrics
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
        # Use nanpercentile since T1 map may contain NaN (CSF, out-of-range)
        valid_t1 = t1_in_brain[np.isfinite(t1_in_brain) & (t1_in_brain > 0)]
        wm_t1_estimate = float(np.percentile(valid_t1, 25))
        gm_t1_estimate = float(np.percentile(valid_t1, 75))

        # QC pass/fail criteria
        wm_pass = wm_range[0] <= wm_t1_estimate <= wm_range[1]
        gm_pass = gm_range[0] <= gm_t1_estimate <= gm_range[1]
        qc_pass = wm_pass and gm_pass

        # Compute additional metrics for QC report
        median_t1 = float(np.median(valid_t1))
        fraction_valid = float(len(valid_t1)) / max(1, int(np.sum(brain_mask)))

        # Build QC metrics dictionary
        has_b1 = isdefined(self.inputs.b1_map) and bool(self.inputs.b1_map)
        qc_metrics: dict[str, Any] = {
            "mean_t1_wm_ms": wm_t1_estimate,
            "mean_t1_gm_ms": gm_t1_estimate,
            "median_t1_ms": median_t1,
            "fraction_valid": fraction_valid,
            "qc_pass": qc_pass,
            "wm_pass": wm_pass,
            "gm_pass": gm_pass,
            "skipped": False,
            "b1_corrected": has_b1,
            "n_flip_angles": n_fa,
            "flip_angles_deg": flip_angles_deg.tolist(),
            "tr_ms": tr_ms,
            "field_strength_t": field_strength,
            "wm_range_ms": wm_range,
            "gm_range_ms": gm_range,
            "lenient_qc": self.inputs.lenient_qc,
            "csf_excluded": csf_excluded,
            "median_filtered": True,
        }
        qc_metrics["edge_voxels_eroded"] = n_edge
        qc_metrics["edge_erosion_mode"] = (
            "2d_per_slice" if n_slices <= SLAB_EROSION_MAX_SLICES else "3d"
        )
        qc_metrics["t1_correction"] = correction_info
        qc_metrics["vfa_conditioning"] = {
            "e1_median": round(e1_median, 5),
            "conditioning": vfa_conditioning,
            "tr_ms": tr_ms,
            "nan_fraction": round(nan_fraction, 4),
        }

        # Hemispheric asymmetry: detect systematic left/right coverage loss
        midplane = shape[0] // 2
        right_valid = brain_mask[:midplane] & np.isfinite(t1_map_arr[:midplane])
        left_valid = brain_mask[midplane:] & np.isfinite(t1_map_arr[midplane:])
        right_t1 = t1_map_arr[:midplane][right_valid]
        left_t1 = t1_map_arr[midplane:][left_valid]
        if len(right_t1) > 100 and len(left_t1) > 100:
            med_r = float(np.median(right_t1))
            med_l = float(np.median(left_t1))
            denom = (med_r + med_l) / 2.0
            asym = abs(med_r - med_l) / denom if denom > 0 else 0.0
            qc_metrics["hemispheric_asymmetry"] = {
                "asymmetry_index": round(float(asym), 4),
                "median_right_ms": round(med_r, 1),
                "median_left_ms": round(med_l, 1),
                "warn": bool(asym > 0.15),
                "fail": bool(asym > 0.30),
            }

        # Derive hierarchical T1 QC status
        from pmriprep.utils.qc_standards import QCStatus
        corr_type = correction_info.get("correction_applied", "none")
        if corr_type in ("fallback", "tissue_segmented"):
            t1_status = QCStatus.POOR
        elif corr_type == "offset":
            t1_status = QCStatus.POOR if (not wm_pass and not gm_pass) else QCStatus.WARN
        else:  # "none"
            if wm_pass and gm_pass:
                t1_status = QCStatus.PASS
            elif wm_pass or gm_pass:
                t1_status = QCStatus.WARN
            else:
                t1_status = QCStatus.POOR
        qc_metrics["t1_status"] = t1_status
        qc_metrics["t1_usable"] = (t1_status != QCStatus.POOR) or (corr_type != "none")

        # Flip angle validation for 2-angle VFA
        # Compare observed vs expected signal ratio using field-strength T1
        if n_fa >= 2:
            from pmriprep.utils.dce_t1_validation import (
                T1_TISSUE_REFERENCE,
                get_field_strength_key,
            )

            field_key = get_field_strength_key(field_strength)
            assumed_t1_s = T1_TISSUE_REFERENCE[field_key]["expected_median"] / 1000.0
            e1_expected = np.exp(-tr_s / assumed_t1_s)
            # Ernst equation: S ∝ sin(α)(1-E1) / (1-E1·cos(α))
            fa1_rad = flip_angles_rad[0]
            fa2_rad = flip_angles_rad[1]
            s1_expected = (
                np.sin(fa1_rad) * (1 - e1_expected)
                / (1 - e1_expected * np.cos(fa1_rad))
            )
            s2_expected = (
                np.sin(fa2_rad) * (1 - e1_expected)
                / (1 - e1_expected * np.cos(fa2_rad))
            )
            ratio_expected = float(s2_expected / s1_expected) if s1_expected > 0 else 1.0

            # Observed median S2/S1 ratio within brain mask
            s1_brain = signal_data[0][brain_mask]
            s2_brain = signal_data[1][brain_mask]
            valid = s1_brain > 0
            if np.any(valid):
                ratio_observed = float(np.median(s2_brain[valid] / s1_brain[valid]))
            else:
                ratio_observed = 1.0

            ratio_error_pct = abs(ratio_observed - ratio_expected) / ratio_expected * 100.0
            validation_pass = ratio_error_pct < 20.0

            if not validation_pass:
                logger.warning(
                    "Flip angle validation: observed S2/S1 ratio %.3f differs from "
                    "expected %.3f by %.1f%% (threshold 20%%). Check flip angles.",
                    ratio_observed, ratio_expected, ratio_error_pct,
                )
            else:
                logger.info(
                    "Flip angle validation: observed S2/S1 ratio %.3f vs expected "
                    "%.3f (error %.1f%%)",
                    ratio_observed, ratio_expected, ratio_error_pct,
                )

            qc_metrics["flip_angle_validation"] = {
                "ratio_observed": round(ratio_observed, 4),
                "ratio_expected": round(ratio_expected, 4),
                "ratio_error_pct": round(ratio_error_pct, 1),
                "validation_pass": validation_pass,
            }

        # Save output T1 map
        out_img = nib.Nifti1Image(t1_map, affine, header)
        nib.save(out_img, out_file)

        # Generate QC plot
        qc_plot_path: Optional[str] = None
        try:
            from pmriprep.utils.dce_qc_plots import generate_vfa_t1_qc_plot

            vfa_data_list = [signal_data[i] for i in range(n_fa)]
            qc_plot_path = generate_vfa_t1_qc_plot(
                vfa_data=vfa_data_list,
                flip_angles=flip_angles_deg.tolist(),
                t1_map=t1_map,
                brain_mask=brain_mask,
                field_strength=field_strength,
                correction_info=correction_info,
                subject_id=self.inputs.subject_id,
                session_id=self.inputs.session_id,
            )
        except Exception as e:
            logger.warning("Failed to generate VFA T1 QC plot: %s", e, exc_info=True)

        # Set outputs
        self._results["out_file"] = out_file
        self._results["mean_t1_wm"] = wm_t1_estimate
        self._results["mean_t1_gm"] = gm_t1_estimate
        self._results["qc_pass"] = qc_pass
        self._results["qc_metrics"] = qc_metrics
        if qc_plot_path is not None:
            self._results["qc_plot"] = qc_plot_path

        return runtime

    def _fit_t1_wlls(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        sin_fa: np.ndarray,
        cos_fa: np.ndarray,
        tr_s: float,
        shape: tuple[int, ...],
        n_iterations: int = 3,
    ) -> np.ndarray:
        """Fit T1 using weighted linear least squares (WLLS) DESPOT1.

        Iteratively reweighted least squares that eliminates 10-20% T1
        overestimation bias inherent in standard GLLS at clinical SNR.

        The weights account for heteroscedastic noise in the linearized
        variables Y=S/sin(a) and X=S/tan(a), which share the same noisy
        signal S. Without weighting, the correlated errors systematically
        bias E1 (and thus T1) upward.

        Args:
            x_data: S/tan(a) data, shape (n_fa, x, y, z)
            y_data: S/sin(a) data, shape (n_fa, x, y, z)
            sin_fa: sin(corrected_fa), shape (n_fa, x, y, z)
            cos_fa: cos(corrected_fa), shape (n_fa, x, y, z)
            tr_s: TR in seconds
            shape: Output shape (x, y, z)
            n_iterations: Number of IRLS iterations (default: 3)

        Returns:
            T1 map in seconds

        References:
            Chang et al. (2008) PMC4196213, Eq. 5
        """
        from pmriprep.utils.dce_constants import EPSILON_E1_UPPER, EPSILON_T1

        n_fa = x_data.shape[0]
        n_voxels = int(np.prod(shape))

        # Reshape to (n_fa, n_voxels) for vectorized regression
        x_flat = x_data.reshape(n_fa, n_voxels)
        y_flat = y_data.reshape(n_fa, n_voxels)
        sin_flat = sin_fa.reshape(n_fa, n_voxels)
        cos_flat = cos_fa.reshape(n_fa, n_voxels)

        # Iteration 0: unweighted GLLS to initialize E1
        sum_x = np.sum(x_flat, axis=0)
        sum_y = np.sum(y_flat, axis=0)
        sum_xy = np.sum(x_flat * y_flat, axis=0)
        sum_x2 = np.sum(x_flat * x_flat, axis=0)

        denom = n_fa * sum_x2 - sum_x * sum_x
        degenerate = np.abs(denom) < EPSILON_T1
        denom = np.where(
            degenerate,
            np.where(denom >= 0, EPSILON_T1, -EPSILON_T1),
            denom,
        )
        e1 = (n_fa * sum_xy - sum_x * sum_y) / denom
        e1 = np.clip(e1, EPSILON_T1, 1.0 - EPSILON_E1_UPPER)

        # Iterative reweighting (Chang et al. 2008, Eq. 5)
        for _ in range(n_iterations):
            # w_i = (sin(a_i) / (1 - E1 * cos(a_i)))^2
            denom_w = 1.0 - e1[np.newaxis, :] * cos_flat  # (n_fa, n_voxels)
            denom_w = np.where(
                np.abs(denom_w) < EPSILON_T1,
                np.where(denom_w >= 0, EPSILON_T1, -EPSILON_T1),
                denom_w,
            )
            w = (sin_flat / denom_w) ** 2  # (n_fa, n_voxels)

            # Weighted linear regression: Y = E1*X + b
            sum_w = np.sum(w, axis=0)
            sum_wx = np.sum(w * x_flat, axis=0)
            sum_wy = np.sum(w * y_flat, axis=0)
            sum_wxy = np.sum(w * x_flat * y_flat, axis=0)
            sum_wx2 = np.sum(w * x_flat * x_flat, axis=0)

            denom_reg = sum_w * sum_wx2 - sum_wx * sum_wx
            degenerate_w = np.abs(denom_reg) < EPSILON_T1
            denom_reg = np.where(
                degenerate_w,
                np.where(denom_reg >= 0, EPSILON_T1, -EPSILON_T1),
                denom_reg,
            )
            e1 = (sum_w * sum_wxy - sum_wx * sum_wy) / denom_reg
            e1 = np.clip(e1, EPSILON_T1, 1.0 - EPSILON_E1_UPPER)

        # T1 = -TR / ln(E1)
        t1_flat = -tr_s / np.log(e1)
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


class VFAPreprocessInputSpec(BaseInterfaceInputSpec):
    """Input specification for VFAPreprocess interface."""

    vfa_files = traits.List(
        File(exists=True),
        mandatory=True,
        minlen=2,
        maxlen=20,
        desc="List of VFA images at different flip angles",
    )
    flip_angles = traits.List(
        traits.Float,
        mandatory=True,
        minlen=2,
        maxlen=20,
        desc="Flip angles in degrees corresponding to vfa_files",
    )
    skip_coregistration = traits.Bool(
        False,
        usedefault=True,
        desc="Skip inter-volume coregistration (if affines are identical)",
    )
    field_strength = traits.Float(
        3.0,
        usedefault=True,
        desc="Magnetic field strength in Tesla for flip angle validation",
    )
    skip_flip_validation = traits.Bool(
        False,
        usedefault=True,
        desc="Skip flip angle validation",
    )
    tr = traits.Float(
        desc="Repetition time in milliseconds (for flip angle validation)",
    )


class VFAPreprocessOutputSpec(TraitedSpec):
    """Output specification for VFAPreprocess interface."""

    out_files = traits.List(
        File(exists=True),
        desc="Coregistered VFA images",
    )
    reference_file = File(
        exists=True,
        desc="VFA image used as registration reference",
    )
    reference_index = traits.Int(desc="Index of reference volume in vfa_files")
    qc_metrics = traits.Dict(desc="VFA preprocessing QC metrics")
    registration_qc_plot = File(
        desc="VFA inter-volume registration QC filmstrip (only if coregistration performed)",
    )


class VFAPreprocess(SimpleInterface):
    """VFA inter-volume coregistration and quality assessment.

    This interface performs:
    1. Quality assessment (SNR/contrast scoring) per VFA volume
    2. Selection of best-quality VFA as registration reference
    3. Affine check: skip FLIRT if affines are identical
    4. FLIRT 6DOF rigid coregistration for non-reference volumes
    5. Optional flip angle validation from signal ratio

    Parameters
    ----------
    vfa_files : list of str
        VFA image paths at different flip angles.
    flip_angles : list of float
        Nominal flip angles in degrees.
    skip_coregistration : bool
        If True, skip FLIRT coregistration.
    field_strength : float
        Field strength in Tesla for flip angle validation.
    skip_flip_validation : bool
        If True, skip flip angle validation.
    tr : float, optional
        Repetition time in ms for flip angle validation.

    Returns
    -------
    out_files : list of str
        Coregistered VFA images.
    reference_file : str
        Path to the reference VFA image.
    reference_index : int
        Index of the reference volume.
    qc_metrics : dict
        QC metrics dictionary.
    """

    input_spec = VFAPreprocessInputSpec
    output_spec = VFAPreprocessOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute VFA preprocessing."""
        import shutil
        import subprocess

        import nibabel as nib  # type: ignore[import-untyped]
        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        vfa_files = list(self.inputs.vfa_files)
        flip_angles = list(self.inputs.flip_angles)
        n_fa = len(vfa_files)

        qc_metrics: dict[str, Any] = {
            "n_flip_angles": n_fa,
            "flip_angles_deg": flip_angles,
            "coregistered": False,
            "affines_identical": False,
        }

        # Load all VFA images
        vfa_imgs = [nib.load(f) for f in vfa_files]
        vfa_data = [
            img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            for img in vfa_imgs
        ]

        # Quality assessment: pick reference by SNR
        snr_scores = []
        for i, data in enumerate(vfa_data):
            nonzero = data[data > 0]
            if len(nonzero) > 100:
                snr = float(np.mean(nonzero) / max(np.std(nonzero), 1e-10))
            else:
                snr = 0.0
            snr_scores.append(snr)

        ref_idx = int(np.argmax(snr_scores))
        qc_metrics["snr_scores"] = snr_scores
        qc_metrics["reference_index"] = ref_idx

        # Check if affines are identical (skip coregistration if so)
        ref_affine = vfa_imgs[ref_idx].affine  # type: ignore[attr-defined]
        affines_identical = True
        for i, img in enumerate(vfa_imgs):
            if i == ref_idx:
                continue
            diff = float(np.abs(img.affine - ref_affine).max())  # type: ignore[attr-defined]
            if diff > 1e-4:
                affines_identical = False
                break

        qc_metrics["affines_identical"] = affines_identical

        # Determine whether to coregister
        do_coreg = (
            not self.inputs.skip_coregistration
            and not affines_identical
            and n_fa >= 2
        )

        out_dir = Path.cwd()
        out_files: list[str] = []

        if do_coreg:
            logger.info(
                "VFA affines differ - coregistering %d volumes to reference (index %d)",
                n_fa, ref_idx,
            )
            # Save reference as-is
            ref_path = vfa_files[ref_idx]
            correlations = []

            for i in range(n_fa):
                if i == ref_idx:
                    # Copy reference unchanged
                    out_name = out_dir / f"vfa_coreg_{i:02d}.nii.gz"
                    shutil.copy(vfa_files[i], str(out_name))
                    out_files.append(str(out_name))
                    correlations.append(1.0)
                    continue

                # FLIRT 6DOF rigid coregistration
                out_name = out_dir / f"vfa_coreg_{i:02d}.nii.gz"
                mat_name = out_dir / f"vfa_coreg_{i:02d}.mat"

                cmd = [
                    "flirt",
                    "-in", str(vfa_files[i]),
                    "-ref", str(ref_path),
                    "-out", str(out_name),
                    "-omat", str(mat_name),
                    "-dof", "6",
                    "-cost", "normmi",
                    "-searchrx", "-5", "5",
                    "-searchry", "-5", "5",
                    "-searchrz", "-5", "5",
                    "-interp", "spline",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.warning(
                        "FLIRT coregistration failed for VFA %d: %s",
                        i, result.stderr,
                    )
                    # Fall back to original
                    shutil.copy(vfa_files[i], str(out_name))

                out_files.append(str(out_name))

                # Compute correlation for QC
                try:
                    coreg_img = nib.load(str(out_name))
                    coreg_data = coreg_img.get_fdata()  # type: ignore[attr-defined]
                    ref_data = vfa_data[ref_idx]
                    mask = (ref_data > 0) & (coreg_data > 0)
                    if np.sum(mask) > 100:
                        corr = float(np.corrcoef(
                            ref_data[mask].ravel(),
                            coreg_data[mask].ravel(),
                        )[0, 1])
                    else:
                        corr = 0.0
                    correlations.append(corr)
                except Exception:
                    correlations.append(0.0)

            qc_metrics["coregistered"] = True
            qc_metrics["correlations"] = correlations

            # Generate registration QC filmstrip
            try:
                from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip

                # Load reference and first coregistered non-reference volume
                ref_img = nib.load(out_files[ref_idx])
                ref_data = ref_img.get_fdata()  # type: ignore[attr-defined]

                # Find first non-reference volume
                first_coreg_idx = 0 if ref_idx != 0 else 1
                coreg_img = nib.load(out_files[first_coreg_idx])
                coreg_data = coreg_img.get_fdata()  # type: ignore[attr-defined]

                qc_plot_path = out_dir / "vfa_registration_qc.png"
                plot_registration_qc_filmstrip(
                    fixed=ref_data,
                    moving=coreg_data,
                    output_path=qc_plot_path,
                    title=f"VFA Registration QC (ref=FA{flip_angles[ref_idx]:.0f}, moving=FA{flip_angles[first_coreg_idx]:.0f})",
                )
                self._results["registration_qc_plot"] = str(qc_plot_path)
            except Exception as exc:
                logger.warning("Failed to generate VFA registration QC plot: %s", exc)
                # Note: registration_qc_plot left undefined (File traits cannot be None)
        else:
            # No coregistration needed — copy files as-is
            for i, vfa_file in enumerate(vfa_files):
                out_name = out_dir / f"vfa_coreg_{i:02d}.nii.gz"
                shutil.copy(vfa_file, str(out_name))
                out_files.append(str(out_name))

            if affines_identical:
                logger.info("VFA affines identical - skipping coregistration")
            elif self.inputs.skip_coregistration:
                logger.info("VFA coregistration skipped by user")

            # Note: registration_qc_plot left undefined when no coregistration (File traits cannot be None)

        # Flip angle validation
        if (
            not self.inputs.skip_flip_validation
            and n_fa >= 2
            and isdefined(self.inputs.tr)
        ):
            fa_validation = self._validate_flip_angles(
                vfa_data, flip_angles, self.inputs.tr, self.inputs.field_strength,
            )
            qc_metrics["flip_angle_validation"] = fa_validation

        # Set outputs
        self._results["out_files"] = out_files
        self._results["reference_file"] = out_files[ref_idx]
        self._results["reference_index"] = ref_idx
        self._results["qc_metrics"] = qc_metrics

        return runtime

    def _validate_flip_angles(
        self,
        vfa_data: list[np.ndarray],
        flip_angles: list[float],
        tr_ms: float,
        field_strength: float,
    ) -> dict[str, Any]:
        """Validate flip angles by comparing observed vs theoretical signal ratio.

        Parameters
        ----------
        vfa_data : list of ndarray
            VFA signal arrays.
        flip_angles : list of float
            Nominal flip angles in degrees.
        tr_ms : float
            Repetition time in milliseconds.
        field_strength : float
            Field strength in Tesla.

        Returns
        -------
        dict
            Validation results with confidence level.
        """
        from pmriprep.utils.dce_t1_validation import (
            T1_TISSUE_REFERENCE,
            get_field_strength_key,
        )

        field_key = get_field_strength_key(field_strength)
        t1_assumed = T1_TISSUE_REFERENCE[field_key]["default"]

        s1 = vfa_data[0]
        s2 = vfa_data[1]
        a1 = flip_angles[0]
        a2 = flip_angles[1]

        # Use central brain voxels
        nz, ny, nx = s1.shape
        z_sl = slice(int(nz * 0.3), int(nz * 0.7))
        y_sl = slice(int(ny * 0.3), int(ny * 0.7))
        x_sl = slice(int(nx * 0.3), int(nx * 0.7))
        center_mask = np.zeros_like(s1, dtype=bool)
        center_mask[z_sl, y_sl, x_sl] = (
            (s1[z_sl, y_sl, x_sl] > 0) & (s2[z_sl, y_sl, x_sl] > 0)
        )

        if np.sum(center_mask) < 100:
            return {"confidence": "low", "reason": "Insufficient voxels"}

        ratio_observed = float(np.median(s2[center_mask]) / np.median(s1[center_mask]))

        E1 = np.exp(-tr_ms / t1_assumed)
        a1_rad = np.radians(a1)
        a2_rad = np.radians(a2)

        ratio_expected = (
            (np.sin(a2_rad) / np.sin(a1_rad))
            * ((1 - E1 * np.cos(a1_rad)) / (1 - E1 * np.cos(a2_rad)))
        )

        ratio_error = abs(ratio_observed - ratio_expected) / max(ratio_expected, 1e-6)

        result: dict[str, Any] = {
            "ratio_observed": ratio_observed,
            "ratio_expected": ratio_expected,
            "ratio_error_pct": ratio_error * 100,
            "flip_angles_deg": [a1, a2],
            "t1_assumed_ms": t1_assumed,
        }

        # Short TR tolerance
        tolerance = 0.50 if tr_ms < 10 else 0.35

        if ratio_error < 0.20:
            result["confidence"] = "high"
            result["status"] = "nominal_angles_correct"
        elif ratio_error < tolerance:
            result["confidence"] = "high"
            result["status"] = "nominal_angles_accepted"
        else:
            result["confidence"] = "low"
            result["status"] = "large_ratio_error"
            # Check for swapped images
            if ratio_observed < 0.8 and ratio_expected > 1.5:
                ratio_swapped = 1.0 / ratio_observed
                error_swapped = abs(ratio_swapped - ratio_expected) / ratio_expected
                if error_swapped < 0.30:
                    result["status"] = "images_may_be_swapped"
                    result["confidence"] = "medium"

            logger.warning(
                "Flip angle validation: large ratio error (%.1f%%) "
                "- T1 values may be unreliable",
                ratio_error * 100,
            )

        return result


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
    volume_tr = traits.Float(
        0.0,
        usedefault=True,
        desc="Volume TR in seconds for TR-adaptive baseline smoothing (0 = use default sigma)",
    )


class DCEPreprocessOutputSpec(TraitedSpec):
    """Output specification for DCEPreprocess interface."""

    out_file = File(exists=True, desc="Motion-corrected 4D DCE image")
    out_params = File(exists=True, desc="Motion parameters file (.par)")
    mean_img = File(exists=True, desc="Mean image of motion-corrected data")
    baseline_mean_img = File(exists=True, desc="Mean of baseline (pre-contrast) frames")
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
            mask_data = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]
            if mask_data.shape == dce_data.shape[:3]:
                brain_mask = mask_data
            else:
                logger.debug(
                    "Brain mask shape %s != DCE data shape %s, "
                    "ignoring mask (will use signal-based threshold)",
                    mask_data.shape, dce_data.shape[:3],
                )

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
        # Clamp baseline to bolus arrival so baseline never extends into bolus.
        if bolus_frame > 0 and baseline_frames > bolus_frame:
            baseline_frames = bolus_frame
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

            from pmriprep.utils.qc_standards import (
                get_composite_motion_status,
            )
            pct_outliers = 100.0 * n_outliers / max(1, n_volumes)
            motion_status = get_composite_motion_status(
                fd_mean, fd_max, pct_outliers,
            )

            qc_metrics = {
                "fd_mean": fd_mean,
                "fd_max": fd_max,
                "motion_status": motion_status,
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

        # Compute mean of motion-corrected data for downstream registration.
        # Always compute from the 4D output rather than relying on mcflirt's
        # _meanvol.nii.gz, which is an undocumented side effect of -stats and
        # may not be produced in all configurations.
        mean_out = self._gen_mean_filename()
        moco_path = f"{out_base}.nii.gz"

        if not Path(moco_path).exists():
            raise FileNotFoundError(
                f"Motion-corrected output not found: {moco_path}"
            )

        moco_img = nib.load(moco_path)
        moco_data = moco_img.get_fdata()  # type: ignore[attr-defined]
        mean_data = np.mean(moco_data, axis=3).astype(np.float32)
        mean_nii = nib.Nifti1Image(
            mean_data,
            moco_img.affine,  # type: ignore[attr-defined]
            moco_img.header,  # type: ignore[attr-defined]
        )
        nib.save(mean_nii, mean_out)

        if not Path(mean_out).exists():
            raise FileNotFoundError(
                f"Failed to create mean image at: {mean_out}"
            )

        # Save baseline mean for downstream use (HD-BET input)
        # Using pre-contrast baseline mean avoids contrast-enhanced vessels
        # inflating the brain mask
        baseline_mean_out = self._gen_baseline_mean_filename()
        baseline_data = np.mean(moco_data[..., :baseline_frames], axis=3)
        baseline_nii = nib.Nifti1Image(
            baseline_data.astype(np.float32),
            moco_img.affine,  # type: ignore[attr-defined]
            moco_img.header,  # type: ignore[attr-defined]
        )
        nib.save(baseline_nii, baseline_mean_out)

        # Set outputs
        self._results["out_file"] = out_file
        self._results["out_params"] = params_file
        self._results["mean_img"] = mean_out
        self._results["baseline_mean_img"] = baseline_mean_out
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

        # Baseline is everything before bolus arrival.
        # _detect_bolus_arrival_from_signal already returns DEFAULT_BASELINE_FRAMES
        # on detection failure and enforces min_baseline=3, so trust its result.
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

        Uses derivative-based onset detection: computes smoothed first
        derivative and finds where rate-of-change first exceeds threshold.
        Detects bolus ONSET (when contrast first arrives), not accumulated
        effect (matching perf-plus dce_preprocess.py:216-236).
        """
        from scipy.ndimage import gaussian_filter1d
        from pmriprep.utils.dce_constants import DEFAULT_BASELINE_FRAMES

        n_volumes = len(signal)
        if n_volumes < 6:
            return DEFAULT_BASELINE_FRAMES

        min_baseline = DEFAULT_BASELINE_FRAMES
        n_baseline_noise = 5  # Frames for noise estimation (matches perf-plus)

        # TR-adaptive sigma: at volume_tr=7.4s, sigma=2.0 smoothes over
        # ~14.8s — wider than the bolus FWHM — causing Gaussian bleed-back
        # that triggers false early detection. Target ~4s smoothing window
        # in temporal units, converted to sample space.
        volume_tr = self.inputs.volume_tr if self.inputs.volume_tr > 0 else 0.0
        if volume_tr > 0:
            sigma = max(0.5, 4.0 / volume_tr)
        else:
            sigma = 2.0  # legacy default

        # Compute first derivative and smooth
        signal_deriv = np.diff(signal)
        signal_deriv_smooth = gaussian_filter1d(signal_deriv, sigma=sigma)

        # Threshold: 2x std of baseline derivative, with absolute floor
        # to prevent false triggers when baseline is very quiet
        baseline_end = min(n_baseline_noise, len(signal_deriv_smooth))
        baseline_deriv = signal_deriv_smooth[:baseline_end]
        noise_threshold = 2.0 * np.std(baseline_deriv)
        abs_floor = 0.01 * np.mean(signal[:baseline_end + 1])
        deriv_threshold = max(noise_threshold, abs_floor)

        # Find first pair of consecutive above-threshold frames
        # (single spikes are likely motion, not bolus onset)
        above = signal_deriv_smooth > deriv_threshold
        for i in range(len(above) - 1):
            if above[i] and above[i + 1]:
                # deriv[i] = signal[i+1] - signal[i], so bolus onset is frame i+1
                return max(i + 1, min_baseline)

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

    def _gen_baseline_mean_filename(self) -> str:
        """Generate baseline mean image filename."""
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

        return str(out_file.parent / f"{base}_baseline_mean{ext}")


# =============================================================================
# DCE Brain Extraction Interface
# =============================================================================


class DCEBrainExtractInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCEBrainExtract interface."""

    dce_mean = File(
        exists=True,
        mandatory=True,
        desc="Mean DCE image (typically from DCEPreprocess)",
    )
    t1w_brain_mask = File(
        exists=True,
        desc="T1w-derived brain mask (fallback if native extraction fails)",
    )
    out_brain = File(
        genfile=True,
        desc="Output brain-extracted DCE image",
    )
    out_mask = File(
        genfile=True,
        desc="Output brain mask in DCE space",
    )
    method = traits.Enum(
        "hdbet",
        "bet",
        "threshold",
        usedefault=True,
        desc="Brain extraction method: hdbet (default), bet (FSL), or threshold",
    )
    bet_frac = traits.Float(
        0.3,
        usedefault=True,
        desc="BET fractional intensity threshold (default: 0.3 for DCE)",
    )
    fill_holes = traits.Bool(
        True,
        usedefault=True,
        desc="Fill holes in the brain mask (default: True)",
    )
    hdbet_device = traits.Enum(
        "auto",
        "cuda",
        "mps",
        "cpu",
        usedefault=True,
        desc="Device for HD-BET: auto, cuda, mps, or cpu (default: auto)",
    )
    hdbet_conda_env = traits.Str(
        desc="Conda environment name for HD-BET. When set, HD-BET is invoked "
        "via 'conda run -n <env>'. When empty, HD-BET is called directly.",
    )


class DCEBrainExtractOutputSpec(TraitedSpec):
    """Output specification for DCEBrainExtract interface."""

    out_brain = File(exists=True, desc="Brain-extracted DCE image")
    out_mask = File(exists=True, desc="Brain mask in DCE native space")
    method_used = traits.Str(desc="Actual method used (may differ if fallback)")
    qc_metrics = traits.Dict(desc="QC metrics for brain extraction")


class DCEBrainExtract(SimpleInterface):
    """Extract brain from DCE mean image using HD-BET with BET fallback.

    This interface performs brain extraction on the DCE mean image to generate
    a native-space brain mask. This is critical for accurate pharmacokinetic
    modeling because:

    1. The T1w brain mask may not align perfectly with DCE space after
       registration, especially near edges
    2. DCE acquisitions often have different coverage than T1w (e.g., thinner
       slab covering only the tumor region)
    3. Native extraction ensures the mask accurately reflects what's actually
       in the DCE field of view

    The method priority is:
    1. HD-BET (deep learning, most accurate) - via conda environment
    2. FSL BET (traditional, robust) - if HD-BET unavailable
    3. Threshold-based (simple, fast) - if both fail

    This matches the perf-plus approach which runs HD-BET on the DCE reference
    image (see perf-plus lines 360-382).

    Parameters
    ----------
    dce_mean : str
        Mean DCE image (from DCEPreprocess.mean_img)
    t1w_brain_mask : str, optional
        T1w brain mask as fallback reference
    method : str
        Preferred method: 'hdbet' (default), 'bet', or 'threshold'
    bet_frac : float
        BET fractional intensity threshold (default: 0.3)
    fill_holes : bool
        Fill holes in the mask (default: True)

    Returns
    -------
    out_brain : str
        Brain-extracted DCE mean image
    out_mask : str
        Binary brain mask in DCE native space
    method_used : str
        Actual method used (may differ if preferred method failed)
    qc_metrics : dict
        QC metrics including volume, coverage, etc.

    Examples
    --------
    >>> brain_extract = DCEBrainExtract()
    >>> brain_extract.inputs.dce_mean = 'dce_mean.nii.gz'
    >>> result = brain_extract.run()  # doctest: +SKIP

    Notes
    -----
    When ``hdbet_conda_env`` is set, HD-BET is invoked via::

        conda run -n <env> --no-capture-output hd-bet -i <input> -o <output>

    When unset (default, e.g. inside Docker), HD-BET is called directly.

    If HD-BET is not available, falls back to FSL BET with parameters
    optimized for DCE contrast (lower fractional intensity threshold).

    References
    ----------
    Isensee F et al. "HD-BET: Robust skull stripping using deep learning."
    Human Brain Mapping. 2019.
    """

    input_spec = DCEBrainExtractInputSpec
    output_spec = DCEBrainExtractOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE brain extraction."""
        import subprocess

        dce_mean = Path(self.inputs.dce_mean)

        # Generate output filenames
        out_brain = self._gen_brain_filename()
        out_mask = self._gen_mask_filename()

        # Track which method was used
        method_used = self.inputs.method
        qc_metrics: dict[str, Any] = {}

        # Try preferred method with fallback
        if self.inputs.method == "hdbet":
            success = self._run_hdbet(dce_mean, out_brain, out_mask)
            if not success:
                logger.warning("HD-BET failed, falling back to FSL BET")
                method_used = "bet"
                success = self._run_bet(dce_mean, out_brain, out_mask)
            if not success:
                logger.warning("FSL BET failed, falling back to threshold")
                method_used = "threshold"
                self._run_threshold(dce_mean, out_brain, out_mask)
        elif self.inputs.method == "bet":
            success = self._run_bet(dce_mean, out_brain, out_mask)
            if not success:
                logger.warning("FSL BET failed, falling back to threshold")
                method_used = "threshold"
                self._run_threshold(dce_mean, out_brain, out_mask)
        else:
            self._run_threshold(dce_mean, out_brain, out_mask)

        # Fill holes if requested
        if self.inputs.fill_holes and Path(out_mask).exists():
            self._fill_holes(out_mask)

        # Compute QC metrics
        qc_metrics = self._compute_qc_metrics(out_mask, dce_mean)
        qc_metrics["method_used"] = method_used

        # Set outputs
        self._results["out_brain"] = out_brain
        self._results["out_mask"] = out_mask
        self._results["method_used"] = method_used
        self._results["qc_metrics"] = qc_metrics

        return runtime

    def _run_hdbet(self, dce_mean: Path, out_brain: str, out_mask: str) -> bool:
        """Run HD-BET brain extraction, optionally via conda environment."""
        import subprocess

        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        # HD-BET requires output to end with .nii.gz
        # It creates <output>.nii.gz (brain) and <output>_mask.nii.gz (mask)
        hdbet_out = out_brain  # Already ends with .nii.gz

        # Get device setting
        device = self.inputs.hdbet_device
        if device == "auto":
            # Auto-detect: try cuda first, then mps, then cpu
            device = self._detect_hdbet_device()

        hdbet_args = [
            "hd-bet",
            "-i", str(dce_mean),
            "-o", hdbet_out,
            "-device", device,
            "--disable_tta",
        ]
        conda_env = (
            self.inputs.hdbet_conda_env
            if isdefined(self.inputs.hdbet_conda_env)
            else ""
        )
        if conda_env:
            cmd = [
                "conda", "run", "-n", conda_env, "--no-capture-output",
            ] + hdbet_args
        else:
            cmd = hdbet_args

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            logger.info("HD-BET completed successfully")

            # HD-BET writes brain-extracted image directly to out_brain
            if not Path(out_brain).exists():
                logger.warning("HD-BET brain not found at expected location")
                return False

            # Derive mask by binarizing the brain-extracted output
            import nibabel as nib
            brain_img = nib.load(out_brain)
            brain_data = brain_img.get_fdata()  # type: ignore[attr-defined]
            mask_data = (brain_data > 0).astype(np.uint8)
            mask_img = nib.Nifti1Image(mask_data, brain_img.affine, brain_img.header)  # type: ignore[attr-defined]
            nib.save(mask_img, out_mask)

            return True

        except subprocess.CalledProcessError as e:
            logger.warning("HD-BET failed: %s", e.stderr if e.stderr else str(e))
            return False
        except subprocess.TimeoutExpired:
            logger.warning("HD-BET timed out after 5 minutes")
            return False
        except FileNotFoundError:
            logger.warning("HD-BET executable not found")
            return False

    def _detect_hdbet_device(self) -> str:
        """Auto-detect best device for HD-BET.

        Returns
        -------
        str
            Device name: 'cuda', 'mps', or 'cpu'.
        """
        from pmriprep.utils.device import detect_torch_device

        return detect_torch_device()

    def _run_bet(self, dce_mean: Path, out_brain: str, out_mask: str) -> bool:
        """Run FSL BET brain extraction."""
        import subprocess

        cmd = [
            "bet",
            str(dce_mean),
            out_brain,
            "-f", str(self.inputs.bet_frac),
            "-m",  # Generate mask
            "-R",  # Robust (iterative) mode
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("FSL BET completed successfully")

            # BET generates <output>_mask.nii.gz
            bet_mask = Path(out_brain).with_name(
                Path(out_brain).name.replace(".nii.gz", "_mask.nii.gz")
                .replace(".nii", "_mask.nii")
            )
            if bet_mask.exists():
                shutil.copy(str(bet_mask), out_mask)
            return True

        except subprocess.CalledProcessError as e:
            logger.warning("FSL BET failed: %s", e.stderr if e.stderr else str(e))
            return False
        except FileNotFoundError:
            logger.warning("FSL bet command not found")
            return False

    def _run_threshold(self, dce_mean: Path, out_brain: str, out_mask: str) -> None:
        """Create brain mask using intensity thresholding."""
        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation

        logger.info("Using threshold-based brain extraction (fallback)")

        img = nib.load(str(dce_mean))
        data = img.get_fdata()  # type: ignore[attr-defined]

        # Use Otsu's method for thresholding
        from skimage.filters import threshold_otsu  # type: ignore[import-untyped]

        thresh = threshold_otsu(data[data > 0])
        mask = data > thresh * 0.5  # Use half of Otsu threshold

        # Clean up mask: fill holes and smooth edges
        mask = binary_fill_holes(mask)
        mask = binary_erosion(mask, iterations=1)
        mask = binary_dilation(mask, iterations=1)

        # Save mask
        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_mask)

        # Apply mask to create brain image
        brain_data = data * mask
        brain_img = nib.Nifti1Image(brain_data.astype(np.float32), img.affine)  # type: ignore[attr-defined]
        nib.save(brain_img, out_brain)

    def _fill_holes(self, mask_path: str) -> None:
        """Fill holes in brain mask using morphological operations."""
        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import binary_fill_holes

        img = nib.load(mask_path)
        mask = img.get_fdata() > 0  # type: ignore[attr-defined]

        # Fill holes slice by slice (2D) and in 3D
        filled = binary_fill_holes(mask)

        # Save filled mask
        filled_img = nib.Nifti1Image(filled.astype(np.float32), img.affine)  # type: ignore[attr-defined]
        nib.save(filled_img, mask_path)

    def _apply_mask(self, in_file: str, mask_file: str, out_file: str) -> None:
        """Apply brain mask to image."""
        import nibabel as nib  # type: ignore[import-untyped]

        img = nib.load(in_file)
        mask_img = nib.load(mask_file)

        data = img.get_fdata()  # type: ignore[attr-defined]
        mask = mask_img.get_fdata() > 0  # type: ignore[attr-defined]

        brain_data = data * mask
        brain_img = nib.Nifti1Image(brain_data.astype(np.float32), img.affine)  # type: ignore[attr-defined]
        nib.save(brain_img, out_file)

    def _compute_qc_metrics(self, mask_path: str, dce_mean: Path) -> dict[str, Any]:
        """Compute QC metrics for brain extraction."""
        import nibabel as nib  # type: ignore[import-untyped]

        metrics: dict[str, Any] = {}

        if not Path(mask_path).exists():
            return metrics

        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata() > 0  # type: ignore[attr-defined]
        voxel_vol = float(np.prod(mask_img.header.get_zooms()[:3]))  # type: ignore[attr-defined]

        # Volume statistics
        n_voxels = int(np.sum(mask_data))
        volume_ml = n_voxels * voxel_vol / 1000.0  # Convert mm³ to ml

        metrics["brain_voxels"] = n_voxels
        metrics["brain_volume_ml"] = volume_ml

        # Coverage: what fraction of the FOV is brain
        total_voxels = int(np.prod(mask_data.shape))
        metrics["brain_coverage_pct"] = 100.0 * n_voxels / total_voxels

        # Check if volume is physiologically reasonable (300-1800 ml for adult brain)
        # Note: DCE may only cover part of the brain
        metrics["volume_reasonable"] = 50.0 < volume_ml < 2000.0

        return metrics

    def _gen_brain_filename(self) -> str:
        """Generate output brain filename."""
        if isdefined(self.inputs.out_brain) and self.inputs.out_brain:
            return str(self.inputs.out_brain)

        dce_mean = Path(self.inputs.dce_mean)
        if dce_mean.name.endswith(".nii.gz"):
            base = dce_mean.name[:-7]
            ext = ".nii.gz"
        else:
            base = dce_mean.stem
            ext = ".nii.gz"

        return str(Path.cwd() / f"{base}_brain{ext}")

    def _gen_mask_filename(self) -> str:
        """Generate output mask filename."""
        if isdefined(self.inputs.out_mask) and self.inputs.out_mask:
            return str(self.inputs.out_mask)

        dce_mean = Path(self.inputs.dce_mean)
        if dce_mean.name.endswith(".nii.gz"):
            base = dce_mean.name[:-7]
            ext = ".nii.gz"
        else:
            base = dce_mean.stem
            ext = ".nii.gz"

        return str(Path.cwd() / f"{base}_brain_mask{ext}")


# =============================================================================
# DCE Registration Constants
# =============================================================================

# Percentile threshold for brain mask creation in QC
QC_MASK_THRESHOLD_PERCENTILE: int = 10

# FLIRT search step sizes (degrees)
FLIRT_COARSE_SEARCH: int = 15
FLIRT_FINE_SEARCH: int = 6

# QC thresholds for registration validation
REG_QC_THRESHOLDS: dict[str, float] = {
    "dice_min": 0.80,
    "dice_good": 0.85,
    "correlation_min": 0.65,
    "correlation_good": 0.70,
    "nmi_min": 1.40,
    "determinant_min": 0.85,
    "determinant_max": 1.15,
}


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
    brain_mask = File(
        exists=True,
        desc="Brain mask for QC metric calculation (optional)",
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
    reference_type = traits.Enum(
        "postcon",
        "precon",
        "any",
        default="postcon",
        usedefault=True,
        desc="Expected T1w reference type: postcon (post-contrast), precon, or any",
    )
    search_range = traits.Int(
        30,
        usedefault=True,
        desc="FLIRT search range in degrees for all axes (default: 30)",
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

    Uses FSL FLIRT to register a T1-weighted structural image to DCE space.
    DCE uses SPGR/FLASH (gradient echo), which has good geometric fidelity
    and similar contrast to structural T1w, so standard FLIRT with normmi
    cost function is appropriate. BBR is not used for DCE.

    Dual-path registration is used:
      Path A: Header pre-aligned T1w -> FLIRT
      Path B: Original T1w -> FLIRT
      Best path selected by DICE overlap.

    Args:
        t1w_file: T1-weighted image to register to DCE space.
        dce_mean: Mean DCE image (target space).
        brain_mask: Brain mask for computing QC metrics.
        dof: Degrees of freedom (default: 6 for rigid body).
        cost: Cost function for FLIRT (default: normmi).
        reference_type: Expected T1w type: postcon (default for DCE), precon,
            or any.
    Returns:
        out_file: Registered T1w in DCE space.
        out_matrix: FSL FLIRT transformation matrix.
        dice: DICE coefficient of brain masks (if brain_mask provided).
        correlation: Correlation coefficient within brain mask.
        nmi: Normalized mutual information.
        qc_metrics: Complete QC metrics dictionary.

    Example:
        Standard usage::

            >>> reg = DCERegistration()
            >>> reg.inputs.t1w_file = 'T1w_brain.nii.gz'
            >>> reg.inputs.dce_mean = 'dce_mean.nii.gz'
            >>> result = reg.run()  # doctest: +SKIP

    Note:
        For DCE, the default reference is post-contrast T1w since DCE has contrast
        enhancement similar to post-contrast T1w, providing better intensity
        correspondence for registration.

        QC metrics computed:

        - DICE: Overlap of brain masks (requires brain_mask input)
        - Correlation: Pearson correlation within brain mask
        - NMI: Normalized mutual information (intensity correspondence)

        Requires FSL to be installed and available in PATH.

    See Also:
        - FSL FLIRT: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT
    """

    input_spec = DCERegistrationInputSpec
    output_spec = DCERegistrationOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE to T1w registration.

        Implements dual-path registration with DICE selection:

        - Path A: Header pre-aligned T1w → FLIRT
        - Path B: Original T1w → FLIRT
        - Select path with best DICE (B must beat A by margin)

        Args:
            runtime: Nipype runtime object.

        Returns:
            Updated runtime object.

        Raises:
            RuntimeError: If both registration paths fail or produce no output.
        """
        # Generate output filenames
        out_file = self._gen_outfilename()
        out_matrix = self._gen_matfilename()

        t1w_file = str(self.inputs.t1w_file)
        dce_mean = str(self.inputs.dce_mean)
        dof = self.inputs.dof
        cost = self.inputs.cost
        search_range = self.inputs.search_range

        # Use tempfile for temp directory management
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ext = ".nii.gz"

            # Header-based pre-alignment: resample T1w into DCE space
            from pmriprep.utils.registration import create_header_aligned

            header_aligned_file = create_header_aligned(
                t1w_file, dce_mean, temp_path, ext
            )

            # Setup output paths for dual-path registration
            out_file_a = str(temp_path / f"t1w_reg_pathA{ext}")
            out_matrix_a = str(temp_path / "t1w_reg_pathA.mat")
            out_file_b = str(temp_path / f"t1w_reg_pathB{ext}")
            out_matrix_b = str(temp_path / "t1w_reg_pathB.mat")

            # Run dual-path FLIRT registration
            self._run_flirt_dual_path(
                header_aligned_file=header_aligned_file,
                t1w_file=t1w_file,
                dce_mean=dce_mean,
                out_file_a=out_file_a,
                out_matrix_a=out_matrix_a,
                out_file_b=out_file_b,
                out_matrix_b=out_matrix_b,
                dof=dof,
                cost=cost,
                search_range=search_range,
            )

            # Select best path based on DICE
            selected_out_file, selected_matrix, dice_a, dice_b, selected_path = (
                self._select_best_path(
                    out_file_a=out_file_a,
                    out_matrix_a=out_matrix_a,
                    out_file_b=out_file_b,
                    out_matrix_b=out_matrix_b,
                    dce_mean=dce_mean,
                )
            )

            logger.info(
                "Registration path selection: A=%.3f, B=%.3f → selected %s",
                dice_a,
                dice_b,
                selected_path,
            )

            # Copy selected outputs to final locations
            shutil.copy(selected_out_file, out_file)
            shutil.copy(selected_matrix, out_matrix)

        # Compute QC metrics (outside temp context so outputs exist)
        qc_metrics = self._compute_qc_metrics(out_file, dce_mean)
        qc_metrics["path_selection"] = {
            "dice_a": dice_a,
            "dice_b": dice_b,
            "selected": selected_path,
        }

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

    def _run_flirt_dual_path(
        self,
        header_aligned_file: str,
        t1w_file: str,
        dce_mean: str,
        out_file_a: str,
        out_matrix_a: str,
        out_file_b: str,
        out_matrix_b: str,
        dof: int,
        cost: str,
        search_range: int,
    ) -> None:
        """Run dual-path FLIRT registration without BBR.

        Path A uses header-aligned T1w, Path B uses original T1w.
        At least one path must succeed.

        Args:
            header_aligned_file: Header-aligned T1w image.
            t1w_file: Original T1w image.
            dce_mean: DCE mean reference image.
            out_file_a: Output file for path A.
            out_matrix_a: Output matrix for path A.
            out_file_b: Output file for path B.
            out_matrix_b: Output matrix for path B.
            dof: Degrees of freedom.
            cost: Cost function name.
            search_range: Search range in degrees.

        Raises:
            RuntimeError: If both registration paths fail.
        """
        # Path A: header-aligned T1w
        flirt_a = FLIRT()
        flirt_a.inputs.output_type = 'NIFTI_GZ'
        flirt_a.inputs.in_file = header_aligned_file
        flirt_a.inputs.reference = dce_mean
        flirt_a.inputs.out_file = out_file_a
        flirt_a.inputs.out_matrix_file = out_matrix_a
        flirt_a.inputs.dof = dof
        flirt_a.inputs.cost = cost
        flirt_a.inputs.searchr_x = [-search_range, search_range]
        flirt_a.inputs.searchr_y = [-search_range, search_range]
        flirt_a.inputs.searchr_z = [-search_range, search_range]
        flirt_a.inputs.coarse_search = FLIRT_COARSE_SEARCH
        flirt_a.inputs.fine_search = FLIRT_FINE_SEARCH

        # Path B: original T1w
        flirt_b = FLIRT()
        flirt_b.inputs.output_type = 'NIFTI_GZ'
        flirt_b.inputs.in_file = t1w_file
        flirt_b.inputs.reference = dce_mean
        flirt_b.inputs.out_file = out_file_b
        flirt_b.inputs.out_matrix_file = out_matrix_b
        flirt_b.inputs.dof = dof
        flirt_b.inputs.cost = cost
        flirt_b.inputs.searchr_x = [-search_range, search_range]
        flirt_b.inputs.searchr_y = [-search_range, search_range]
        flirt_b.inputs.searchr_z = [-search_range, search_range]
        flirt_b.inputs.coarse_search = FLIRT_COARSE_SEARCH
        flirt_b.inputs.fine_search = FLIRT_FINE_SEARCH

        # Run both paths, capture failures
        error_a: Optional[str] = None
        error_b: Optional[str] = None

        try:
            flirt_a.run()
        except Exception as exc:
            error_a = str(exc)
            logger.warning("FLIRT path A failed: %s", exc)

        try:
            flirt_b.run()
        except Exception as exc:
            error_b = str(exc)
            logger.warning("FLIRT path B failed: %s", exc)

        # Check if both failed
        if error_a and error_b:
            raise RuntimeError(
                f"Both FLIRT paths failed:\n"
                f"Path A (header-aligned): {error_a}\n"
                f"Path B (original): {error_b}"
            )

    def _select_best_path(
        self,
        out_file_a: str,
        out_matrix_a: str,
        out_file_b: str,
        out_matrix_b: str,
        dce_mean: str,
    ) -> tuple[str, str, float, float, str]:
        """Select best registration path based on DICE overlap.

        Path B must exceed Path A by DICE_SELECTION_MARGIN to be selected,
        providing a preference for the header-aligned path when close.

        Args:
            out_file_a: Output file from path A.
            out_matrix_a: Output matrix from path A.
            out_file_b: Output file from path B.
            out_matrix_b: Output matrix from path B.
            dce_mean: DCE mean reference for DICE computation.

        Returns:
            Tuple of (selected_file, selected_matrix, dice_a, dice_b, path_name).

        Raises:
            RuntimeError: If neither path produced valid output.
        """
        file_a_exists = Path(out_file_a).exists()
        file_b_exists = Path(out_file_b).exists()

        if not file_a_exists and not file_b_exists:
            raise RuntimeError(
                f"FLIRT registration produced no valid output files.\n"
                f"Expected file A: {out_file_a}\n"
                f"Expected file B: {out_file_b}\n"
                f"Check FSL installation and input images."
            )

        from pmriprep.utils.registration import (
            DICE_SELECTION_MARGIN,
            compute_dice_for_selection,
        )

        dice_a = (
            compute_dice_for_selection(out_file_a, dce_mean)
            if file_a_exists
            else 0.0
        )
        dice_b = (
            compute_dice_for_selection(out_file_b, dce_mean)
            if file_b_exists
            else 0.0
        )

        # Select best path (B must beat A by margin)
        if file_b_exists and (
            not file_a_exists or dice_b > dice_a + DICE_SELECTION_MARGIN
        ):
            return out_file_b, out_matrix_b, dice_a, dice_b, "B"
        elif file_a_exists:
            return out_file_a, out_matrix_a, dice_a, dice_b, "A"
        else:
            raise RuntimeError(
                "FLIRT registration produced no valid output files. "
                "Check FSL installation and input images."
            )

    def _compute_qc_metrics(
        self, registered_t1w: str, dce_mean: str
    ) -> dict[str, Any]:
        """Compute registration QC metrics with validation thresholds.

        Args:
            registered_t1w: Path to registered T1w image.
            dce_mean: Path to reference DCE mean image.

        Returns:
            Dictionary of QC metrics including qc_status (PASS/WARN/FAIL).
        """
        qc_metrics: dict[str, Any] = {}
        warnings_list: list[str] = []

        if not Path(registered_t1w).exists():
            qc_metrics["qc_status"] = "POOR"
            qc_metrics["warnings"] = ["Registered file does not exist"]
            return qc_metrics

        try:
            from pmriprep.utils.registration import (
                calculate_correlation_from_files,
                calculate_dice,
                calculate_nmi_from_files,
            )

            # Get optional brain mask using isdefined
            mask_path: Optional[str] = None
            if isdefined(self.inputs.brain_mask) and Path(
                self.inputs.brain_mask
            ).exists():
                mask_path = str(self.inputs.brain_mask)

            # Calculate NMI (always possible)
            try:
                nmi = calculate_nmi_from_files(
                    registered_t1w, dce_mean, mask_path=mask_path
                )
                qc_metrics["nmi"] = nmi
                if nmi < REG_QC_THRESHOLDS["nmi_min"]:
                    warnings_list.append(
                        f"NMI {nmi:.2f} below threshold "
                        f"{REG_QC_THRESHOLDS['nmi_min']}"
                    )
            except Exception as exc:
                logger.debug("NMI calculation failed: %s", exc)

            # Calculate correlation within brain mask
            try:
                corr = calculate_correlation_from_files(
                    registered_t1w, dce_mean, mask_path=mask_path
                )
                qc_metrics["correlation"] = corr
                if corr < REG_QC_THRESHOLDS["correlation_min"]:
                    warnings_list.append(
                        f"Correlation {corr:.2f} below threshold "
                        f"{REG_QC_THRESHOLDS['correlation_min']}"
                    )
            except Exception as exc:
                logger.debug("Correlation calculation failed: %s", exc)

            # Calculate DICE if brain mask provided
            if mask_path:
                try:
                    # Create a rough brain mask from registered T1w by thresholding
                    reg_img = nib.load(registered_t1w)
                    reg_data = reg_img.get_fdata()  # type: ignore[attr-defined]

                    # Threshold at percentile of positive values
                    reg_positive = reg_data[reg_data > 0]
                    if len(reg_positive) > 0:
                        threshold = np.percentile(
                            reg_positive, QC_MASK_THRESHOLD_PERCENTILE
                        )
                        reg_mask = reg_data > threshold

                        # Load the provided brain mask
                        mask_img = nib.load(mask_path)
                        ref_mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]

                        # Compute DICE between the two masks
                        dice = calculate_dice(
                            reg_mask.astype(np.float64),
                            ref_mask.astype(np.float64),
                        )
                        qc_metrics["dice"] = dice
                        if dice < REG_QC_THRESHOLDS["dice_min"]:
                            warnings_list.append(
                                f"DICE {dice:.2f} below threshold "
                                f"{REG_QC_THRESHOLDS['dice_min']}"
                            )
                except Exception as exc:
                    logger.debug("DICE calculation failed: %s", exc)

            # Check transform determinant for volume preservation
            out_matrix = self._gen_matfilename()
            if Path(out_matrix).exists():
                try:
                    det = self._check_transform_determinant(out_matrix)
                    qc_metrics["transform_determinant"] = det
                    if det < REG_QC_THRESHOLDS["determinant_min"]:
                        warnings_list.append(
                            f"Transform determinant {det:.3f} < "
                            f"{REG_QC_THRESHOLDS['determinant_min']} "
                            "(volume shrinkage)"
                        )
                    elif det > REG_QC_THRESHOLDS["determinant_max"]:
                        warnings_list.append(
                            f"Transform determinant {det:.3f} > "
                            f"{REG_QC_THRESHOLDS['determinant_max']} "
                            "(volume expansion)"
                        )
                except Exception as exc:
                    logger.debug("Determinant check failed: %s", exc)

            # Add registration metadata
            qc_metrics["dof"] = self.inputs.dof
            qc_metrics["cost_function"] = self.inputs.cost
            qc_metrics["reference_type"] = self.inputs.reference_type
            qc_metrics["brain_mask_used"] = mask_path is not None

            # Determine overall QC status
            qc_metrics["warnings"] = warnings_list
            if warnings_list:
                # Check if any are failures (below min thresholds)
                dice = qc_metrics.get("dice", 1.0)
                corr = qc_metrics.get("correlation", 1.0)
                if (
                    dice < REG_QC_THRESHOLDS["dice_min"]
                    or corr < REG_QC_THRESHOLDS["correlation_min"]
                ):
                    qc_metrics["qc_status"] = "POOR"
                else:
                    qc_metrics["qc_status"] = "WARN"
            else:
                qc_metrics["qc_status"] = "PASS"

        except ImportError as exc:
            logger.debug("Registration utilities not available: %s", exc)
            qc_metrics["qc_status"] = "UNKNOWN"
        except Exception as exc:
            logger.warning("QC metric computation failed: %s", exc)
            qc_metrics["qc_status"] = "UNKNOWN"

        return qc_metrics

    def _check_transform_determinant(self, mat_file: str) -> float:
        """Check affine determinant for volume preservation.

        The determinant of the rotation/scale part of the affine should
        be near 1.0 for a rigid body transform. Values significantly
        different from 1.0 indicate scaling/shearing.

        Args:
            mat_file: Path to FLIRT .mat file.

        Returns:
            Determinant of the 3x3 rotation/scale submatrix.
        """
        mat = np.loadtxt(mat_file)
        det = float(np.linalg.det(mat[:3, :3]))
        return det

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified.

        Returns:
            Path to output registered T1w file.
        """
        if isdefined(self.inputs.out_file):
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
        """Generate output matrix filename if not specified.

        Returns:
            Path to output transformation matrix file.
        """
        if isdefined(self.inputs.out_matrix):
            return str(self.inputs.out_matrix)

        out_file = Path(self._gen_outfilename())
        if out_file.name.endswith(".nii.gz"):
            base = out_file.name[:-7]
        elif out_file.name.endswith(".nii"):
            base = out_file.stem
        else:
            base = out_file.stem

        return str(out_file.parent / f"{base}.mat")
