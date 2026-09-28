"""DCE (Dynamic Contrast Enhanced) preprocessing interfaces for PMRIPrep.

This module implements Nipype interfaces for DCE-MRI preprocessing steps
including B1+ mapping, T1 mapping, motion correction, registration, and
signal processing.
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.fsl import FLIRT  # type: ignore[import-untyped]

from pmriprep.utils.b1_encoding import (
    B1Encoding,
    expected_pixel_max,
    resolve_b1_divisor,
)

logger = logging.getLogger("pmriprep.interfaces.dce")


def _reorient_to_ras(in_file: Optional[str]) -> Optional[str]:
    """Reorient a NIfTI to canonical RAS orientation.

    Reorders the data array and affine together so the voxel axes run R, A, S;
    physical (world) positions are preserved. The reoriented volume is read back
    into numpy for the B1 fit, so no external tools are required.

    Parameters
    ----------
    in_file : str or None
        Path to input NIfTI file, or None.

    Returns
    -------
    str or None
        Path to the reoriented NIfTI (or the input path if already RAS); None if
        input was None or missing.
    """
    if in_file is None:
        return None

    in_path = Path(in_file)
    if not in_path.exists():
        return None

    from pmriprep.utils.nifti import nifti_gz_name, reorient_img_to_ras

    img = nib.load(str(in_path))
    reoriented = reorient_img_to_ras(img)
    if reoriented is img:
        return in_file  # already RAS; no rewrite needed

    out_file = str(Path.cwd() / nifti_gz_name(in_file, "_reoriented"))
    nib.save(reoriented, out_file)
    return out_file


def _b1_interior_mask(valid: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Compute interior mask for B1+ normalized convolution.

    Fills internal holes (CSF, noisy B1+) so erosion targets only the
    outer brain boundary, then intersects with the original valid mask
    to exclude filled-but-invalid voxels.

    Parameters
    ----------
    valid : numpy.ndarray
        Boolean mask of valid B1+ voxels.
    iterations : int
        Number of erosion iterations (default 2).

    Returns
    -------
    numpy.ndarray
        Boolean interior mask.
    """
    from scipy.ndimage import binary_erosion, binary_fill_holes  # type: ignore[import-untyped]

    return binary_erosion(binary_fill_holes(valid), iterations=iterations) & valid  # type: ignore[no-any-return]


def _brain_mask_in_target_space(
    brain_mask_file: str | None,
    target_img: Any,
) -> tuple[np.ndarray | None, str]:
    """Load a brain mask on a B1 acquisition grid using header geometry.

    This is affine-header resampling, not intensity registration. A mask with
    no target-grid overlap is ignored so an unrelated/mislabelled anatomical
    mask cannot erase all B1 support.
    """
    if not brain_mask_file:
        return None, "not_provided"

    import nibabel as nib  # type: ignore[import-untyped]
    from nilearn.image import resample_to_img  # type: ignore[import-untyped]

    mask_img = nib.load(brain_mask_file)
    same_shape = mask_img.shape[:3] == target_img.shape[:3]  # type: ignore[attr-defined]
    same_affine = bool(np.allclose(mask_img.affine, target_img.affine, atol=1e-4))  # type: ignore[attr-defined]
    if same_shape and same_affine:
        mask = np.asarray(mask_img.dataobj) > 0.5  # type: ignore[attr-defined]
        status = "native_grid"
    else:
        mask_resampled = resample_to_img(
            mask_img, target_img, interpolation="nearest", force_resample=True,
            copy_header=True,
        )
        mask = np.asarray(mask_resampled.dataobj) > 0.5  # type: ignore[attr-defined]
        status = "header_resampled"

    if not np.any(mask):
        logger.warning(
            "Brain mask has no overlap with the B1+ grid after header-based "
            "resampling — ignoring it"
        )
        return None, "no_overlap_ignored"
    return mask, status


class B1MappingInputSpec(BaseInterfaceInputSpec):
    """Input specification for B1Mapping interface."""

    tr1 = traits.Float(
        desc="Short TR value in milliseconds (TR1, typically 20ms)",
    )
    tr2 = traits.Float(
        desc="Long TR value in milliseconds (TR2, typically 100ms)",
    )
    afi_4d_file = File(
        exists=True,
        desc="4D AFI volume with 2 interleaved-TR timepoints (TB1AFI).",
    )
    nominal_flip_angle = traits.Float(
        desc="Nominal AFI excitation flip angle in degrees (required)",
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
        desc="Pre-computed B1+ map (vendor-neutral). Pixel encoding is set by "
             "b1_map_encoding (auto-detected from BIDS suffix/Units when 'auto').",
    )
    b1_map_encoding = traits.Enum(
        *[e.value for e in B1Encoding],
        usedefault=True,
        desc="Pixel-value encoding of the pre-computed B1+ map: 'auto' "
             "(infer from suffix/Units, else pixel median), 'actual_flip_deci_deg' "
             "(Siemens tfl, pixel = flip×10), 'actual_flip_deg', 'percent' "
             "(BIDS p.u., 100=nominal), or 'ratio' (1.0=nominal).",
    )
    b1_nominal_prep_fa = traits.Float(
        desc="Nominal preparation/saturation flip angle (degrees), required only "
             "for actual-flip encodings. Read from FlipAngle in the sidecar.",
    )


class B1MappingOutputSpec(TraitedSpec):
    """Output specification for B1Mapping interface."""

    out_file = File(exists=True, desc="B1+ correction factor map")
    reference_file = File(
        exists=True,
        desc="Long-TR AFI magnitude image used for cross-acquisition registration",
    )
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

    This interface implements the AFI method for mapping
    the B1+ transmit field inhomogeneity. B1+ correction is critical for
    accurate T1 mapping and DCE quantification, particularly at higher
    field strengths (3T, 7T).

    The AFI method acquires two spoiled gradient echo images with
    identical flip angles but different repetition times (TR1 << TR2).
    The ratio of these images depends only on the actual flip angle,
    allowing B1+ to be calculated without knowing T1.

    Parameters
    ----------
    afi_4d_file : str
        4D NIfTI with two interleaved-TR volumes (TB1AFI BIDS suffix).
    tr1 : float
        Short TR value in milliseconds
    tr2 : float
        Long TR value in milliseconds
    nominal_flip_angle : float
        Nominal AFI excitation flip angle in degrees (required).
    brain_mask : str, optional
        Brain mask constraining estimation and QC, header-resampled when needed.
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
    The AFI method calculates B1+ from the ratio of the long-TR to the
    short-TR signal:

        r = S2/S1 = (1 + n*cos(alpha)) / (n + cos(alpha))

    Where:
        - S1, S2 are the signal intensities from TR1 (short) and TR2 (long)
        - n = TR2/TR1 (TR ratio)
        - alpha is the actual flip angle

    In the regime TR1, TR2 << T1 the relaxation terms cancel and the flip
    angle follows from:

        cos(alpha) = (r*n - 1) / (n - r)

    The B1+ correction factor is then:

        B1+ = actual_flip_angle / nominal_flip_angle

    QC criteria:
        - Mean B1+: 0.85-1.15 (expected ~1.0 for well-calibrated system)
        - Spatial std: <0.3 (B1+ should be spatially smooth)

    See Also
    --------
    T1Mapping : VFA T1 mapping with B1+ correction
    """

    input_spec = B1MappingInputSpec
    output_spec = B1MappingOutputSpec

    def _load_afi_data(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Any, Any, float, float, dict[str, Any]]:
        """Load AFI data from a 4D TB1AFI volume.

        Auto-detects TR1/TR2 ordering from JSON sidecar using:
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
        tr1_override = self.inputs.tr1 if isdefined(self.inputs.tr1) else None
        tr2_override = self.inputs.tr2 if isdefined(self.inputs.tr2) else None

        # Reorient AFI 4D file to RAS before loading (returns the input path
        # unchanged when it is already RAS)
        if afi_4d is not None:
            afi_4d_reoriented = _reorient_to_ras(afi_4d)
            if afi_4d_reoriented and afi_4d_reoriented != afi_4d:
                afi_4d = afi_4d_reoriented
                logger.info("Reoriented 4D AFI to RAS")

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
                # Discovery supplies the sorted short/long TR values. Preserve
                # the volume association from per-volume metadata when it is
                # available instead of assuming volume 0 is always short-TR.
                if vol0_tr is not None and vol1_tr is not None and vol0_tr > vol1_tr:
                    s1 = afi_data[:, :, :, 1]
                    s2 = afi_data[:, :, :, 0]
                else:
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
                # No TR info found. Vendor-specific defaults (Siemens 20/100,
                # GE 25/125, Philips 30/150) cannot be silently assumed —
                # wrong ratio yields wrong B1+ → wrong T1 → wrong Ktrans.
                if tr1_override is not None and tr2_override is not None:
                    tr1_ms = float(tr1_override)
                    tr2_ms = float(tr2_override)
                    s1 = afi_data[:, :, :, 0]
                    s2 = afi_data[:, :, :, 1]
                else:
                    raise ValueError(
                        "AFI 4D acquisition has no RepetitionTime metadata in "
                        "JSON sidecar and no tr1/tr2 override provided. "
                        "AFI requires explicit TR1 and TR2 (no vendor defaults)."
                    )

            # 4D mode: no coregistration needed (same acquisition)
            coreg_info: dict[str, Any] = {"coregistered": False}
            return s1, s2, affine, header, tr1_ms, tr2_ms, coreg_info

        else:
            raise ValueError("afi_4d_file is required (4D TB1AFI).")

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

        # Load 4D AFI data
        s1, s2, affine, header, tr1_ms, tr2_ms, coreg_info = self._load_afi_data()
        shape = s1.shape[:3]

        # Get TR ratio
        if not np.isfinite(tr1_ms) or not np.isfinite(tr2_ms):
            raise ValueError(f"AFI TR values must be finite, got {tr1_ms} and {tr2_ms} ms")
        if tr1_ms <= 0 or tr2_ms <= tr1_ms:
            raise ValueError(
                f"AFI requires 0 < TR1 < TR2, got TR1={tr1_ms:g} ms and "
                f"TR2={tr2_ms:g} ms"
            )
        n = tr2_ms / tr1_ms
        # AFI nominal flip angle must be explicit — vendor defaults (60° Siemens,
        # ~55° GE) silently bias B1+ → T1 → concentration → Ktrans.
        if not isdefined(self.inputs.nominal_flip_angle):
            raise ValueError(
                "AFI nominal_flip_angle not set. Required from BIDS sidecar "
                "FlipAngle or afi_flip_angle config; no vendor default applied."
            )
        nominal_fa = float(self.inputs.nominal_flip_angle)
        if not np.isfinite(nominal_fa) or nominal_fa <= 0 or nominal_fa >= 180:
            raise ValueError(
                f"AFI nominal flip angle must be finite and between 0 and 180 degrees, "
                f"got {nominal_fa!r}"
            )

        brain_mask_file = (
            str(self.inputs.brain_mask)
            if isdefined(self.inputs.brain_mask) and self.inputs.brain_mask
            else None
        )
        mask, mask_status = _brain_mask_in_target_space(
            brain_mask_file, nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine, header)
        )

        # -----------------------------------------------------------------
        # Signal masking: only compute B1+ where both signals are above
        # noise floor. Uses 5th-percentile threshold to preserve
        # low-signal dropout regions (Gap 4).
        # -----------------------------------------------------------------
        threshold_domain = mask if mask is not None else np.ones(shape, dtype=bool)
        positive_signals = np.concatenate([
            s1[threshold_domain & (s1 > 0)].ravel(),
            s2[threshold_domain & (s2 > 0)].ravel(),
        ])
        if len(positive_signals) > 0:
            signal_threshold = max(
                float(np.percentile(positive_signals, 5)), 1e-6,
            )
        else:
            signal_threshold = 1e-6
        signal_mask = (s1 >= signal_threshold) & (s2 >= signal_threshold)
        if mask is not None:
            signal_mask &= mask

        # -----------------------------------------------------------------
        # Compute signal ratio r = S2/S1 (long-TR over short-TR); this is
        # the orientation required by cos(alpha) = (r*n - 1) / (n - r).
        # -----------------------------------------------------------------
        r = np.ones(shape, dtype=np.float64)
        r[signal_mask] = s2[signal_mask] / np.maximum(s1[signal_mask], EPSILON_RATIO)

        # -----------------------------------------------------------------
        # Ratio validation + denominator stability (Gaps 1, 9).
        # Ratio validity bounds: (0.05, 1.5) and |n - r| > 0.1
        # -----------------------------------------------------------------
        ratio_valid = (
            (r > B1_RATIO_VALID_RANGE[0]) & (r < B1_RATIO_VALID_RANGE[1])
        )
        denominator_stable = np.abs(n - r) > B1_DENOMINATOR_THRESHOLD
        candidate_valid = signal_mask & ratio_valid & denominator_stable

        # -----------------------------------------------------------------
        # Simplified AFI equation, accurate to first order
        # in TR1/T1:  cos(alpha) = (r*n - 1) / (n - r)
        # Compute only for valid voxels; set the rest to B1+=1.0.
        # -----------------------------------------------------------------
        b1_map = np.zeros(shape, dtype=np.float64)

        cos_alpha_map = np.full(shape, np.nan, dtype=np.float64)
        numerator = r[candidate_valid] * n - 1.0
        denominator = n - r[candidate_valid]
        cos_alpha_map[candidate_valid] = numerator / (denominator + EPSILON_RATIO)
        cosine_valid = np.isfinite(cos_alpha_map) & (
            (cos_alpha_map >= -1.0) & (cos_alpha_map <= 1.0)
        )
        valid = candidate_valid & cosine_valid
        alpha_actual = np.degrees(np.arccos(cos_alpha_map[valid]))
        b1_map[valid] = alpha_actual / nominal_fa

        # -----------------------------------------------------------------
        # B1+ physiological clipping (Gap 2, field-strength-dependent)
        # -----------------------------------------------------------------
        field_strength = self.inputs.field_strength
        b1_thresh = get_b1_thresholds(field_strength)
        b1_map = np.clip(b1_map, b1_thresh.clip_low, b1_thresh.clip_high)
        b1_map[~valid] = 0.0  # Preserve missing B1+ support in the derivative

        # Flag extreme values for QC
        n_extreme_low = int(np.sum((b1_map < b1_thresh.extreme_low) & valid))
        n_extreme_high = int(np.sum((b1_map > b1_thresh.extreme_high) & valid))
        n_signal_actual = int(np.sum(signal_mask))
        n_signal = max(n_signal_actual, 1)
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
        # Normalized convolution smoothing (Gap 3).
        # smooth(B1*mask) / smooth(mask) averages only within brain
        # voxels and avoids bias at brain edges.
        # -----------------------------------------------------------------
        smoothing_fwhm = self.inputs.smoothing_fwhm
        n_b1_outliers = 0
        n_b1_interpolated = 0
        output_support = valid.copy()
        if smoothing_fwhm > 0:
            from scipy.ndimage import binary_fill_holes  # type: ignore[import-untyped]

            support_domain = mask if mask is not None else binary_fill_holes(valid)
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
            b1_local = np.zeros_like(s_num)
            np.divide(s_num, s_den, out=b1_local, where=has_coverage)

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
            b1_map[outlier] = 0.0

            # Pass 2: re-smooth with cleaned mask (standard sigma)
            mask_float = valid.astype(np.float64)
            smoothed_num = gaussian_filter(b1_map * mask_float, sigma=sigma_voxels)
            smoothed_den = gaussian_filter(mask_float, sigma=sigma_voxels)
            supported_gap = (
                support_domain
                & ~valid
                & ~outlier
                & (smoothed_den > B1_MIN_COVERAGE)
            )
            output_support = valid | supported_gap
            n_b1_interpolated = int(np.sum(supported_gap))
            smoothed_den = np.maximum(smoothed_den, 1e-6)
            b1_map = np.where(output_support, smoothed_num / smoothed_den, 0.0)
            b1_map = np.clip(b1_map, b1_thresh.clip_low, b1_thresh.clip_high)
            b1_map[~output_support] = 0.0

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
                logger.warning("B1+ mapping produced no valid physical support")
                b1_map = np.zeros(shape, dtype=np.float32)
                uniform_detected = True
            elif np.isclose(b1_unique[0], 1.0, atol=1e-3):
                logger.info("B1+ map is all ones (no correction needed)")
                uniform_detected = True

        # Calculate QC metrics within measured brain support.
        qc_support = output_support & mask if mask is not None else output_support
        b1_in_roi = b1_map[qc_support]

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
        qc_pass = bool(len(b1_clean) > 0) and (
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
            "clip_low": b1_thresh.clip_low,
            "clip_high": b1_thresh.clip_high,
            "nominal_flip_angle_deg": nominal_fa,
            "tr1_ms": tr1_ms,
            "tr2_ms": tr2_ms,
            "tr_ratio": n,
            "b1_range_expected": b1_thresh.mean_range,
            "b1_max_std_expected": b1_thresh.max_std,
            "n_signal_voxels": n_signal_actual,
            "n_valid_voxels": int(np.sum(valid)),
            "n_ratio_invalid": int(np.sum(signal_mask & ~ratio_valid)),
            "n_denominator_unstable": int(np.sum(signal_mask & ~denominator_stable)),
            "n_cosine_invalid": int(np.sum(candidate_valid & ~cosine_valid)),
            "n_extreme_low_b1": n_extreme_low,
            "n_extreme_high_b1": n_extreme_high,
            "n_b1_outliers_rejected": n_b1_outliers,
            "n_b1_interpolated_voxels": n_b1_interpolated,
            "uniform_detected": uniform_detected,
            "valid_fraction_of_signal": float(np.sum(valid)) / max(1, n_signal),
            "qc_support_voxels": int(np.sum(qc_support)),
            "brain_mask_application": mask_status,
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

        # Cross-acquisition registration uses AFI magnitude data rather than
        # the smooth B1+ parameter map.
        afi_reference_file = str(Path.cwd() / "afi_long_tr_reference.nii.gz")
        nib.save(
            nib.Nifti1Image(s2.astype(np.float32), affine, header),
            afi_reference_file,
        )

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
        self._results["reference_file"] = afi_reference_file
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
        """Process a pre-computed B1+ map into a B1+ correction ratio map.

        The pixel encoding is resolved from ``b1_map_encoding`` (one of
        ``actual_flip_deci_deg`` = Siemens tfl 0.1° units, ``actual_flip_deg``,
        ``percent`` = BIDS p.u., ``ratio``), or inferred when ``auto``. The B1+
        ratio is ``pixel / divisor`` where ``divisor`` is the pixel value at the
        nominal flip angle (see :mod:`pmriprep.utils.b1_encoding`).
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

        # Reorient to RAS while preserving the world-space affine.
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

        brain_mask_file = (
            str(self.inputs.brain_mask)
            if isdefined(self.inputs.brain_mask) and self.inputs.brain_mask
            else None
        )
        brain_mask, mask_status = _brain_mask_in_target_space(brain_mask_file, b1_img)
        raw_support = np.isfinite(b1_raw) & (b1_raw > 0)
        if brain_mask is not None:
            raw_support &= brain_mask

        # Resolve the pixel→ratio scaling from the encoding (auto → inferred
        # from suffix/Units upstream, or from the pixel median as a last resort).
        nominal_prep_fa = (
            float(self.inputs.b1_nominal_prep_fa)
            if isdefined(self.inputs.b1_nominal_prep_fa)
            and self.inputs.b1_nominal_prep_fa
            else None
        )
        b1_res = resolve_b1_divisor(
            encoding=B1Encoding(self.inputs.b1_map_encoding),
            nominal_prep_fa=nominal_prep_fa,
            b1_raw=b1_raw,
            brain_mask=brain_mask,
        )
        scale_factor = b1_res.divisor
        logger.info(
            "Pre-computed B1+ scaling: pixel / %.1f (encoding=%s, source=%s)",
            scale_factor, b1_res.resolved_encoding.value, b1_res.source,
        )

        # Validate pixel values — catch grossly wrong files (e.g., anatomical
        # companion with signal-intensity values), but allow high B1+ at FOV
        # edges. Downstream physiological clipping handles the actual range.
        full_scale = expected_pixel_max(scale_factor)
        p99 = float(np.percentile(b1_raw[raw_support], 99)) if np.any(raw_support) else 0.0
        if p99 > full_scale * 5.0:
            raise ValueError(
                f"Pre-computed B1+ map pixel p99={p99:.0f} exceeds "
                f"5× expected maximum ({full_scale * 5.0:.0f}) for encoding "
                f"{b1_res.resolved_encoding.value}. Check that this is the correct "
                "B1+ map file and that its encoding is set correctly."
            )
        if p99 > full_scale * 2.0:
            logger.warning(
                "Pre-computed B1+ map pixel p99=%.0f exceeds 2× nominal "
                "(%.0f). Some voxels have B1+ ratio > 2.0 — likely edge/artifact "
                "voxels that will be clipped.",
                p99, full_scale * 2.0,
            )

        # Create valid mask (non-zero pixels)
        valid_mask = raw_support

        # Scale to B1+ ratio
        b1_map = np.zeros(shape, dtype=np.float64)
        b1_map[valid_mask] = b1_raw[valid_mask] / scale_factor

        # -----------------------------------------------------------------
        # Physiological clipping (field-strength-dependent)
        # -----------------------------------------------------------------
        field_strength = self.inputs.field_strength
        b1_thresh = get_b1_thresholds(field_strength)
        b1_map = np.clip(b1_map, b1_thresh.clip_low, b1_thresh.clip_high)
        b1_map[~valid_mask] = 0.0  # Preserve missing B1+ support in the derivative

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
        n_b1_interpolated = 0
        output_support = valid_mask.copy()
        if smoothing_fwhm > 0:
            from scipy.ndimage import binary_fill_holes  # type: ignore[import-untyped]

            support_domain = (
                brain_mask if brain_mask is not None else binary_fill_holes(valid_mask)
            )
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
            b1_local = np.zeros_like(s_num)
            np.divide(s_num, s_den, out=b1_local, where=has_coverage)

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
            b1_map[outlier] = 0.0

            # Pass 2: re-smooth with cleaned mask
            mask_float = valid_mask.astype(np.float64)
            smoothed_num = gaussian_filter(b1_map * mask_float, sigma=sigma_voxels)
            smoothed_den = gaussian_filter(mask_float, sigma=sigma_voxels)
            supported_gap = (
                support_domain
                & ~valid_mask
                & ~outlier
                & (smoothed_den > B1_MIN_COVERAGE)
            )
            output_support = valid_mask | supported_gap
            n_b1_interpolated = int(np.sum(supported_gap))
            smoothed_den = np.maximum(smoothed_den, 1e-6)
            b1_map = np.where(output_support, smoothed_num / smoothed_den, 0.0)
            b1_map = np.clip(b1_map, b1_thresh.clip_low, b1_thresh.clip_high)
            b1_map[~output_support] = 0.0

        b1_map = b1_map.astype(np.float32)

        # -----------------------------------------------------------------
        # QC metrics
        # -----------------------------------------------------------------
        b1_in_roi = b1_map[output_support]
        b1_clean = b1_in_roi[(b1_in_roi > 0.5) & (b1_in_roi < 2.0)]

        if len(b1_clean) > 0:
            mean_b1 = float(np.mean(b1_clean))
            std_b1 = float(np.std(b1_clean))
            median_b1 = float(np.median(b1_clean))
        else:
            mean_b1 = B1_EXPECTED_MEAN
            std_b1 = 0.0
            median_b1 = B1_EXPECTED_MEAN

        qc_pass = bool(len(b1_clean) > 0) and (
            b1_thresh.mean_range[0] <= mean_b1 <= b1_thresh.mean_range[1]
            and std_b1 < b1_thresh.max_std
        )

        from pmriprep.utils.qc_standards import QCStatus
        b1_status = QCStatus.PASS if qc_pass else QCStatus.WARN

        qc_metrics = {
            "method": "precomputed",
            "encoding": b1_res.resolved_encoding.value,
            "encoding_source": b1_res.source,
            "mean_b1": mean_b1,
            "std_b1": std_b1,
            "median_b1": median_b1,
            "b1_status": b1_status,
            "qc_pass": qc_pass,
            "skipped": False,
            "smoothing_fwhm_mm": smoothing_fwhm,
            "clip_low": b1_thresh.clip_low,
            "clip_high": b1_thresh.clip_high,
            "nominal_prep_fa_deg": nominal_prep_fa,
            "divisor": scale_factor,
            "pixel_p99": p99,
            "b1_range_expected": b1_thresh.mean_range,
            "b1_max_std_expected": b1_thresh.max_std,
            "n_valid_voxels": int(np.sum(valid_mask)),
            "n_extreme_low_b1": n_extreme_low,
            "n_extreme_high_b1": n_extreme_high,
            "n_b1_outliers_rejected": n_b1_outliers,
            "n_b1_interpolated_voxels": n_b1_interpolated,
            "brain_mask_application": mask_status,
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
                divisor=scale_factor,
                nominal_prep_fa=nominal_prep_fa,
                encoding=b1_res.resolved_encoding.value,
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
    field_strength = traits.Float(
        3.0,
        usedefault=True,
        desc="Magnetic field strength in Tesla for QC range selection",
    )
    csf_probseg = traits.Either(
        File(exists=True),
        traits.Enum(""),
        mandatory=False,
        desc="CSF probability map for excluding CSF voxels from T1 fitting. "
        "Empty string when unavailable (Function-node empty-string convention).",
    )
    wm_probseg = traits.Either(
        File(exists=True),
        traits.Enum(""),
        mandatory=False,
        desc="WM probability map for measured-T1 tissue QC (optional)",
    )
    gm_probseg = traits.Either(
        File(exists=True),
        traits.Enum(""),
        mandatory=False,
        desc="GM probability map for measured-T1 tissue QC (optional)",
    )
    csf_threshold = traits.Float(
        0.5,
        usedefault=True,
        desc="CSF probability threshold for exclusion (default: 0.5 = 50%, "
             "matches ASLPrep/FreeSurfer standard)",
    )
    t1_method = traits.Enum(
        "nlls", "wlls",
        usedefault=True,
        desc="VFA T1 fitting method: nlls (WLLS-initialized nonlinear least "
             "squares) or wlls (linearized DESPOT1).",
    )
    t1_correction_mode = traits.Enum(
        "auto",
        "none",
        "fallback",
        usedefault=True,
        desc="T1 correction mode: auto, none, or uniform fallback",
    )
    b1_qc_pass = traits.Bool(
        True,
        usedefault=True,
        desc="B1 QC pass status (from B1Mapping), retained as a report-only "
             "T1 acquisition diagnostic.",
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
    support_file = File(exists=True, desc="Categorical T1 support provenance map")
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
    approach:

        S(α)/sin(α) = E1 × S(α)/tan(α) + S0 × (1 - E1)

    Which is linear: Y = m × X + b, where:
        - Y = S/sin(α)
        - X = S/tan(α)
        - m = E1
        - b = S0 × (1 - E1)

    T1 is then calculated from E1:
        T1 = -TR / ln(E1)

    QC ranges by field strength (3T):
        - WM T1: 800-1000ms
        - GM T1: 1200-1600ms

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
            EPSILON_TRIG,
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
                get_field_strength_key,
            )

            field_key = get_field_strength_key(self.inputs.field_strength)
            default_t1 = T1_TISSUE_REFERENCE[field_key]["default"]
            t1_map = np.full(shape, default_t1, dtype=np.float32)
            logger.info(
                "skip_t1: using uniform field-strength T1 (%.0fms)",
                default_t1,
            )

            # Save output
            out_img = nib.Nifti1Image(t1_map, affine)
            nib.save(out_img, out_file)

            # Set outputs for skipped case
            self._results["out_file"] = out_file
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
                "tissue_t1_used": False,
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
            b1_missing = ~np.isfinite(b1_map) | (b1_map <= 0)
            b1_map = np.clip(b1_map, b1_thresh.vfa_clip_low, b1_thresh.vfa_clip_high)
            b1_map[b1_missing] = 1.0  # No B1 data → nominal flip angles
        else:
            b1_map = np.ones(shape, dtype=np.float64)
            b1_missing = np.ones(shape, dtype=bool)

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

        # Perform T1 fitting based on selected method
        cos_fa = np.cos(corrected_fa_rad)
        t1_method = self.inputs.t1_method

        if t1_method == "wlls":
            t1_map = self._fit_t1_wlls(
                x_data, y_data, sin_fa, cos_fa, tr_s, shape,
            )
        elif t1_method == "nlls":
            t1_map = self._fit_t1_nlls(
                signal_data, sin_fa, cos_fa, tr_s, shape,
            )
        else:
            raise ValueError(f"Unknown t1_method: {t1_method!r}")

        # Convert T1 from seconds to milliseconds
        t1_map = t1_map * 1000.0

        # Load brain mask for QC if provided
        if isdefined(self.inputs.brain_mask) and self.inputs.brain_mask:
            mask_img = nib.load(self.inputs.brain_mask)
            # Resample if either dimensions or world-space geometry differ.
            if (
                mask_img.shape[:3] != shape  # type: ignore[attr-defined]
                or not np.allclose(
                    mask_img.affine, affine, rtol=0.0, atol=1e-4,  # type: ignore[attr-defined]
                )
            ):
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]

                mask_img = resample_to_img(
                    mask_img, first_img, interpolation="nearest",
                )
            brain_mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]
        else:
            # Use simple thresholding on first VFA image as rough brain mask
            threshold = np.percentile(signal_data[0][signal_data[0] > 0], 10)
            brain_mask = signal_data[0] > threshold
        analysis_domain_mask = brain_mask.copy()
        fallback_brain_mask = brain_mask.copy()
        tissue_qc_mask = brain_mask.copy()

        # --- VFA Fit Residual (NRMSE) for fit-quality QC ---
        # For n_fa >= 3, compute per-voxel NRMSE to assess VFA fit quality.
        # Low NRMSE means the SPGR model with nominal flip angles fits the
        # multi-FA data well, implying B1 is close to nominal.
        # For n_fa = 2, the system is exactly determined (NRMSE ≡ 0).
        # Smooth B1 error without a B1 map is invisible to this metric —
        # fundamental VFA limitation.
        vfa_fit_nrmse: Optional[float] = None
        if n_fa >= 3:
            t1_s = t1_map / 1000.0  # back to seconds for E1
            valid_nrmse = (t1_s > 0) & np.isfinite(t1_s) & brain_mask
            e1_fit = np.exp(-tr_s / np.where(valid_nrmse, t1_s, 1.0))

            # S0 from closed-form LS: S_i = S0·f_i → S0 = Σ(S·f)/Σ(f²).
            # Use the per-voxel B1-corrected flip angles the T1 was fit from, not
            # the nominal ones, so the goodness-of-fit is evaluated at the same
            # angles (otherwise NRMSE is inflated wherever B1 != 1).
            f_all = (
                np.sin(corrected_fa_rad)
                * (1 - e1_fit[np.newaxis])
                / (1 - e1_fit[np.newaxis] * np.cos(corrected_fa_rad))
            )
            f2_sum = np.sum(f_all ** 2, axis=0)
            f2_sum = np.where(f2_sum > 1e-20, f2_sum, 1e-20)  # guard
            s0_fit = np.sum(signal_data * f_all, axis=0) / f2_sum

            s_pred = s0_fit[np.newaxis] * f_all
            s_mean = np.mean(signal_data, axis=0)
            s_mean = np.where(s_mean > 1e-10, s_mean, 1e-10)  # guard
            nrmse_map = (
                np.sqrt(np.mean((signal_data - s_pred) ** 2, axis=0)) / s_mean
            )
            nrmse_valid = nrmse_map[valid_nrmse]
            if len(nrmse_valid) > 0:
                vfa_fit_nrmse = float(np.nanmedian(nrmse_valid))
                p25 = float(np.nanpercentile(nrmse_valid, 25))
                p75 = float(np.nanpercentile(nrmse_valid, 75))
                logger.info(
                    "VFA fit NRMSE: median=%.2f%%, IQR=[%.2f%%, %.2f%%] "
                    "(n_fa=%d, %d voxels)",
                    vfa_fit_nrmse * 100, p25 * 100, p75 * 100,
                    n_fa, len(nrmse_valid),
                )

        # Exclude background before quantitative validity checks.
        t1_map[~brain_mask] = np.nan

        # Restrict parenchymal summary/QC statistics with the anatomical CSF
        # probability map. Do not turn a tissue-classification result into a
        # quantitative T1 support mask: FAST may classify tumor, edema, stroke,
        # or other pathology as CSF even when VFA produced a valid estimate.
        csf_qc_restricted = False
        n_csf_qc = 0
        if isdefined(self.inputs.csf_probseg) and self.inputs.csf_probseg:
            csf_img = nib.load(self.inputs.csf_probseg)
            # Resample CSF probseg to VFA space if needed
            if (
                csf_img.shape[:3] != shape  # type: ignore[attr-defined]
                or not np.allclose(
                    csf_img.affine, affine, rtol=0.0, atol=1e-4,  # type: ignore[attr-defined]
                )
            ):
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]

                csf_img = resample_to_img(
                    csf_img, first_img, interpolation="continuous",
                )
            csf_prob = csf_img.get_fdata()  # type: ignore[attr-defined]
            csf_mask = csf_prob >= self.inputs.csf_threshold
            n_brain = int(np.sum(brain_mask))
            n_csf_qc = int(np.sum(csf_mask & brain_mask))
            tissue_qc_mask = brain_mask & ~csf_mask

            logger.info(
                "CSF QC restriction: excluded %d voxels (%.1f%% of brain) "
                "from parenchymal summary statistics; T1 support retained",
                n_csf_qc,
                100.0 * n_csf_qc / max(n_brain, 1),
            )
            csf_qc_restricted = True

        # Reject only values outside the broad numerical safety envelope.
        # Normal-tissue reference intervals remain QC diagnostics: using them
        # as voxel filters would censor prolonged T1 in tumor, edema, or stroke.
        field_strength = self.inputs.field_strength
        correction_mode = self.inputs.t1_correction_mode

        from pmriprep.utils.dce_t1_validation import (
            T1_TISSUE_REFERENCE,
            MIN_VALID_T1_MS,
            apply_t1_correction,
            get_field_strength_key,
        )

        field_key = get_field_strength_key(field_strength)
        t1_upper = T1_TISSUE_REFERENCE[field_key]["t1_upper_bound"]
        t1_lower = MIN_VALID_T1_MS
        measured_t1_map = t1_map.copy()
        numeric_low_mask = np.isfinite(t1_map) & (t1_map < t1_lower) & brain_mask
        numeric_high_mask = np.isfinite(t1_map) & (t1_map > t1_upper) & brain_mask
        t1_map = np.where(
            (t1_map >= t1_lower) & (t1_map <= t1_upper), t1_map, np.nan,
        )
        measured_support = np.isfinite(t1_map) & brain_mask
        population_lower = T1_TISSUE_REFERENCE[field_key]["wm"][0]
        population_upper = T1_TISSUE_REFERENCE[field_key]["gm"][1]
        outside_reference = measured_support & (
            (t1_map < population_lower) | (t1_map > population_upper)
        )

        # Edge erosion: only needed when CSF probseg is NOT available.
        # When CSF exclusion at threshold 0.5 is applied, the problematic
        # partial-volume voxels at the brain-CSF boundary are already removed.
        # The numerical safety filter catches rare impossible outliers, and
        # downstream fit QC and any explicitly configured tSNR gate catch bad
        # PK fits. Blind erosion after CSF exclusion needlessly removes
        # ~12-17k gray matter voxels at the cortical surface.
        n_edge = 0
        edge_erosion_skipped = False
        edge_mask = np.zeros_like(brain_mask, dtype=bool)
        n_slices = shape[2]  # RAS orientation: axis 2 = slices

        if csf_qc_restricted:
            # CSF probseg supports parenchymal QC — skip blind erosion. Valid
            # quantitative T1 estimates remain available throughout the brain.
            edge_erosion_skipped = True
            logger.info(
                "T1 edge erosion: SKIPPED (CSF probability at threshold %.2f "
                "restricts parenchymal QC statistics)",
                self.inputs.csf_threshold,
            )
        else:
            # Fallback: blind morphological erosion when no CSF probseg
            from scipy.ndimage import binary_erosion, binary_fill_holes, generate_binary_structure  # type: ignore[import-untyped]

            valid_t1_mask = np.isfinite(t1_map) & (t1_map > 0)
            valid_filled = binary_fill_holes(valid_t1_mask)

            # Defensive check: slab-aware erosion assumes axis 2 = slices (RAS).
            from nibabel.orientations import axcodes2ornt, io_orientation

            ornt = io_orientation(first_img.affine)  # type: ignore[attr-defined]
            ras_ornt = axcodes2ornt(("R", "A", "S"))
            if not np.array_equal(ornt, ras_ornt):
                logger.warning(
                    "T1Mapping input is not RAS-oriented; slab-aware erosion "
                    "assumes axis 2 is the slice axis. Results may be incorrect."
                )

            # Slab-aware erosion for thin 3D slabs
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

            edge_mask = valid_t1_mask & ~eroded
            n_edge = int(np.sum(edge_mask))
            t1_map[edge_mask] = np.nan
            brain_mask = brain_mask & ~edge_mask
            logger.info(
                "T1 edge erosion (fallback, no CSF probseg): removed %d "
                "partial-volume voxels (%.1f%% of brain mask)",
                n_edge, 100.0 * n_edge / max(int(np.sum(brain_mask) + n_edge), 1),
            )

        # Convert to float32 for storage
        t1_map = t1_map.astype(np.float32)

        # Compute NaN fraction in brain before correction.
        # Use pre-erosion count as denominator. For the numerator, count all
        # NaN voxels within the original brain extent — eroded voxels are NaN
        # in t1_map but excluded from the narrowed brain_mask, so we also
        # count NaN voxels that were eroded away (n_edge).
        nan_in_brain = np.isnan(t1_map) & analysis_domain_mask
        nan_fraction = float(np.sum(nan_in_brain)) / max(1, int(np.sum(analysis_domain_mask)))

        # VFA conditioning QC metric
        # Compute median T1 from valid voxels for E1 calculation
        valid_for_e1 = t1_map[tissue_qc_mask & np.isfinite(t1_map) & (t1_map > 0)]
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

        # Load WM/GM probsegs for measured-map tissue QC summaries.
        wm_probseg_arr: Optional[np.ndarray] = None
        gm_probseg_arr: Optional[np.ndarray] = None
        if (
            isdefined(self.inputs.wm_probseg) and self.inputs.wm_probseg
            and isdefined(self.inputs.gm_probseg) and self.inputs.gm_probseg
        ):
            wm_ps_img = nib.load(self.inputs.wm_probseg)
            gm_ps_img = nib.load(self.inputs.gm_probseg)
            if (
                wm_ps_img.shape[:3] != shape  # type: ignore[attr-defined]
                or not np.allclose(
                    wm_ps_img.affine, affine, rtol=0.0, atol=1e-4,  # type: ignore[attr-defined]
                )
                or not np.allclose(
                    gm_ps_img.affine, affine, rtol=0.0, atol=1e-4,  # type: ignore[attr-defined]
                )
            ):
                from nilearn.image import resample_to_img  # type: ignore[import-untyped]
                wm_ps_img = resample_to_img(wm_ps_img, first_img, interpolation="continuous")
                gm_ps_img = resample_to_img(gm_ps_img, first_img, interpolation="continuous")
            wm_probseg_arr = wm_ps_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            gm_probseg_arr = gm_ps_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]

        # A B1 map file may exist but be a synthetic uniform map (skip_b1=True,
        # no AFI data). Check for actual spatial variation — std=0 means no
        # real B1 correction was applied.
        if isdefined(self.inputs.b1_map) and bool(self.inputs.b1_map):
            b1_std = float(np.std(b1_map[tissue_qc_mask]))
            b1_corrected = b1_std > 1e-6
            if not b1_corrected:
                logger.info(
                    "B1 map provided but uniform (std=%.2e) — treating as uncorrected",
                    b1_std,
                )
        else:
            b1_corrected = False

        t1_map_arr, correction_result = apply_t1_correction(
            t1_map,
            field_strength=field_strength,
            correction_mode=correction_mode,
            brain_mask=tissue_qc_mask,
            b1_qc_pass=self.inputs.b1_qc_pass,
            nan_fraction=nan_fraction,
            wm_probseg=wm_probseg_arr,
            gm_probseg=gm_probseg_arr,
            e1_median=e1_median,
            b1_corrected=b1_corrected,
            population_reference_applicable=csf_qc_restricted,
        )
        t1_map = t1_map_arr.astype(np.float32)
        if correction_result.correction_applied == "fallback":
            default_t1 = T1_TISSUE_REFERENCE[field_key]["default"]
            t1_map = np.where(
                fallback_brain_mask,
                default_t1,
                np.nan,
            ).astype(np.float32)
        correction_info = correction_result.to_dict()
        logger.info(
            "T1 correction: %s (median %.0fms -> %.0fms)",
            correction_result.correction_applied,
            correction_result.median_t1_original,
            correction_result.median_t1_corrected,
        )

        # Fill interior NaN gaps (B1+ failures, range-filter, VFA noise)
        from pmriprep.utils.dce_t1_validation import fill_t1_gaps

        voxel_sizes = np.array(first_img.header.get_zooms()[:3], dtype=np.float64)  # type: ignore[attr-defined]
        t1_map, gap_fill_result = fill_t1_gaps(
            t1_map,
            brain_mask=brain_mask,
            field_strength=field_strength,
            voxel_sizes_mm=voxel_sizes,
        )
        correction_info["gap_fill"] = gap_fill_result.to_dict()

        # Preserve why each voxel is or is not supported. Values are:
        # 0 outside the original DCE analysis domain, 1 directly measured,
        # 2 locally interpolated, 3 unavailable, and 5 a user-requested or
        # no-usable-data uniform assumption. Anatomical tissue classes do not
        # alter quantitative support.
        final_support = np.isfinite(t1_map) & brain_mask
        interpolated_support = final_support & ~measured_support
        support_map = np.zeros(shape, dtype=np.uint8)
        support_map[analysis_domain_mask] = 3
        support_map[measured_support & brain_mask] = 1
        support_map[interpolated_support] = 2
        if correction_result.correction_applied == "fallback":
            support_map[analysis_domain_mask] = 5

        domain_count = int(np.sum(analysis_domain_mask))
        support_counts = {
            "analysis_domain_voxels": domain_count,
            "directly_measured_voxels": int(np.sum(support_map == 1)),
            "locally_interpolated_voxels": int(np.sum(support_map == 2)),
            "unavailable_voxels": int(np.sum(support_map == 3)),
            "csf_qc_restricted_voxels": n_csf_qc,
            "uniform_assumption_voxels": int(np.sum(support_map == 5)),
            "rejected_below_numeric_min_voxels": int(np.sum(numeric_low_mask)),
            "rejected_above_numeric_max_voxels": int(np.sum(numeric_high_mask)),
            "retained_outside_population_reference_voxels": int(np.sum(outside_reference)),
            "numeric_min_ms": float(t1_lower),
            "numeric_max_ms": float(t1_upper),
            "population_reference_min_ms": float(population_lower),
            "population_reference_max_ms": float(population_upper),
        }
        for key in (
            "directly_measured_voxels", "locally_interpolated_voxels",
            "unavailable_voxels",
            "uniform_assumption_voxels",
            "retained_outside_population_reference_voxels",
        ):
            support_counts[key.replace("_voxels", "_fraction")] = (
                float(support_counts[key]) / max(1, domain_count)
            )

        # Calculate QC metrics
        qc_ranges = get_tissue_t1_ranges(field_strength)

        wm_range = (qc_ranges.white_matter.min, qc_ranges.white_matter.max)
        gm_range = (qc_ranges.gray_matter.min, qc_ranges.gray_matter.max)

        # Estimate WM and GM T1 from actual high-confidence tissue masks when
        # available. Without segmentations, retain distribution summaries but
        # label them honestly rather than presenting quartiles as tissues.
        t1_in_brain = t1_map[tissue_qc_mask]

        # WM: lower T1 values (shorter relaxation time)
        # GM: higher T1 values
        # Simple approach: use histogram percentiles
        # Use nanpercentile since T1 map may contain NaN (CSF, out-of-range)
        valid_t1 = t1_in_brain[np.isfinite(t1_in_brain) & (t1_in_brain > 0)]
        tissue_stats_source = "brain_distribution_quartiles"
        wm_t1_estimate = float(np.percentile(valid_t1, 25))
        gm_t1_estimate = float(np.percentile(valid_t1, 75))
        if wm_probseg_arr is not None and gm_probseg_arr is not None:
            wm_support = (
                brain_mask
                & np.isfinite(t1_map)
                & (wm_probseg_arr >= 0.8)
                & (gm_probseg_arr < 0.2)
            )
            gm_support = (
                brain_mask
                & np.isfinite(t1_map)
                & (gm_probseg_arr >= 0.8)
                & (wm_probseg_arr < 0.2)
            )
            if int(np.sum(wm_support)) >= 100 and int(np.sum(gm_support)) >= 100:
                wm_t1_estimate = float(np.mean(t1_map[wm_support]))
                gm_t1_estimate = float(np.mean(t1_map[gm_support]))
                tissue_stats_source = "high_confidence_probseg"

        # QC pass/fail criteria
        wm_pass = wm_range[0] <= wm_t1_estimate <= wm_range[1]
        gm_pass = gm_range[0] <= gm_t1_estimate <= gm_range[1]
        qc_pass = wm_pass and gm_pass

        # Compute additional metrics for QC report
        median_t1 = float(np.median(valid_t1))
        fraction_valid = float(
            np.sum(np.isfinite(t1_map) & (t1_map > 0) & brain_mask)
        ) / max(1, int(np.sum(brain_mask)))

        # Build QC metrics dictionary
        has_b1 = b1_corrected
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
            "b1_valid_fraction": round(
                float(np.sum(brain_mask & ~b1_missing))
                / max(1, int(np.sum(brain_mask))),
                4,
            ),
            "n_flip_angles": n_fa,
            "flip_angles_deg": flip_angles_deg.tolist(),
            "tr_ms": tr_ms,
            "field_strength_t": field_strength,
            "wm_range_ms": wm_range,
            "gm_range_ms": gm_range,
            "csf_qc_restricted": csf_qc_restricted,
            "median_filtered": False,
            "t1_method": t1_method,
            "tissue_statistics_source": tissue_stats_source,
            "t1_support": support_counts,
        }
        qc_metrics["edge_voxels_eroded"] = n_edge
        qc_metrics["edge_erosion_skipped"] = edge_erosion_skipped
        qc_metrics["edge_erosion_mode"] = (
            "skipped" if edge_erosion_skipped
            else ("2d_per_slice" if n_slices <= SLAB_EROSION_MAX_SLICES else "3d")
        )
        qc_metrics["t1_correction"] = correction_info
        qc_metrics["vfa_conditioning"] = {
            "e1_median": round(e1_median, 5),
            "conditioning": vfa_conditioning,
            "tr_ms": tr_ms,
            "nan_fraction": round(nan_fraction, 4),
            "vfa_fit_nrmse": round(vfa_fit_nrmse, 5) if vfa_fit_nrmse is not None else None,
            "n_flip_angles": n_fa,
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
        if corr_type == "fallback":
            t1_status = QCStatus.POOR
        else:  # "none": reference deviations warn but do not reject support
            support_warn = (
                support_counts["unavailable_voxels"] > 0
                or support_counts["rejected_below_numeric_min_voxels"] > 0
                or support_counts["rejected_above_numeric_max_voxels"] > 0
                or not self.inputs.b1_qc_pass
            )
            if wm_pass and gm_pass and not support_warn:
                t1_status = QCStatus.PASS
            else:
                t1_status = QCStatus.WARN
        qc_metrics["t1_status"] = t1_status
        qc_metrics["t1_usable"] = bool(np.any(np.isfinite(t1_map) & brain_mask))

        # Flip angle validation for 2-angle VFA
        # Compare observed vs expected signal ratio using field-strength T1
        if n_fa >= 2:
            from pmriprep.utils.dce_t1_validation import (
                T1_TISSUE_REFERENCE,
                get_field_strength_key,
            )

            field_key = get_field_strength_key(field_strength)
            assumed_t1_ms = T1_TISSUE_REFERENCE[field_key]["expected_median"]
            assumed_t1_s = assumed_t1_ms / 1000.0
            e1_expected = np.exp(-tr_s / assumed_t1_s)

            # Select optimal FA pair: widest angular separation
            flip_angles_deg = np.degrees(flip_angles_rad)
            best_i, best_j = 0, 1
            max_sep = 0.0
            for ii in range(n_fa):
                for jj in range(ii + 1, n_fa):
                    sep = abs(flip_angles_deg[jj] - flip_angles_deg[ii])
                    if sep > max_sep:
                        max_sep = sep
                        best_i, best_j = ii, jj

            # Ernst equation: S ∝ sin(α)(1-E1) / (1-E1·cos(α))
            fa1_rad = flip_angles_rad[best_i]
            fa2_rad = flip_angles_rad[best_j]
            s1_expected = (
                np.sin(fa1_rad) * (1 - e1_expected)
                / (1 - e1_expected * np.cos(fa1_rad))
            )
            s2_expected = (
                np.sin(fa2_rad) * (1 - e1_expected)
                / (1 - e1_expected * np.cos(fa2_rad))
            )
            ratio_expected = float(s2_expected / s1_expected) if s1_expected > 0 else 1.0

            # Observed per-voxel median(S2/S1) within brain mask.
            # Per-voxel ratios are more robust to spatially varying B1+
            # than ratio-of-medians (median(S2)/median(S1)).
            s1_brain = signal_data[best_i][brain_mask]
            s2_brain = signal_data[best_j][brain_mask]
            valid = s1_brain > 0
            if np.any(valid):
                ratio_observed = float(np.median(s2_brain[valid] / s1_brain[valid]))
            else:
                ratio_observed = 1.0

            ratio_error_pct = abs(ratio_observed - ratio_expected) / ratio_expected * 100.0

            # Tiered tolerance: short-TR VFA is inherently noisier because
            # E1→1 amplifies the ratio's sensitivity to T1 variability
            tr_ms = float(self.inputs.tr)
            tolerance_pct = 50.0 if tr_ms < 10 else 35.0

            if ratio_error_pct < 20.0:
                validation_pass = True
                fa_confidence = "high"
                fa_status = "nominal_angles_correct"
            elif ratio_error_pct < tolerance_pct:
                validation_pass = True
                fa_confidence = "medium"
                fa_status = "nominal_angles_accepted"
            else:
                validation_pass = False
                fa_confidence = "low"
                fa_status = "large_ratio_error"
                # Symmetric swap detection: check if 1/ratio is closer
                if ratio_observed > 0:
                    error_swapped_pct = (
                        abs(1.0 / ratio_observed - ratio_expected)
                        / ratio_expected * 100
                    )
                    if error_swapped_pct < 30.0:
                        fa_status = "images_may_be_swapped"
                        fa_confidence = "medium"

            if not validation_pass:
                logger.warning(
                    "Flip angle validation: observed S[%d°]/S[%d°] ratio %.3f differs from "
                    "expected %.3f by %.1f%% (tolerance %.0f%%). Check flip angles.",
                    flip_angles_deg[best_i], flip_angles_deg[best_j],
                    ratio_observed, ratio_expected, ratio_error_pct, tolerance_pct,
                )
            else:
                logger.info(
                    "Flip angle validation: observed S[%d°]/S[%d°] ratio %.3f vs expected "
                    "%.3f (error %.1f%%, %s)",
                    flip_angles_deg[best_i], flip_angles_deg[best_j],
                    ratio_observed, ratio_expected, ratio_error_pct, fa_status,
                )

            qc_metrics["flip_angle_validation"] = {
                "ratio_observed": round(ratio_observed, 4),
                "ratio_expected": round(ratio_expected, 4),
                "ratio_error_pct": round(ratio_error_pct, 1),
                "validation_pass": validation_pass,
                "confidence": fa_confidence,
                "status": fa_status,
                "t1_assumed_ms": round(assumed_t1_ms, 1),
                "fa_pair_deg": [round(flip_angles_deg[best_i], 1), round(flip_angles_deg[best_j], 1)],
            }

        # Save output T1 map
        out_img = nib.Nifti1Image(t1_map, affine, header)
        nib.save(out_img, out_file)
        support_file = str(Path(runtime.cwd) / "t1_support_dseg.nii.gz")
        support_img = nib.Nifti1Image(support_map, affine, header)
        support_img.set_data_dtype(np.uint8)
        nib.save(support_img, support_file)

        # Generate QC for every VFA attempt. When the measured map was rejected,
        # the plot labels it explicitly and shows the final fallback map beside
        # it so the diagnostic evidence is retained without implying that the
        # rejected values were used for concentration conversion.
        qc_plot_path: Optional[str] = None
        try:
            from pmriprep.utils.dce_qc_plots import generate_vfa_t1_qc_plot

            vfa_data_list = [signal_data[i] for i in range(n_fa)]
            qc_plot_path = generate_vfa_t1_qc_plot(
                vfa_data=vfa_data_list,
                flip_angles=flip_angles_deg.tolist(),
                t1_map=t1_map,
                measured_t1_map=measured_t1_map,
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
        self._results["support_file"] = support_file
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

        Parameters
        ----------
        x_data : numpy.ndarray
            S/tan(a) data, shape (n_fa, x, y, z)
        y_data : numpy.ndarray
            S/sin(a) data, shape (n_fa, x, y, z)
        sin_fa : numpy.ndarray
            sin(corrected_fa), shape (n_fa, x, y, z)
        cos_fa : numpy.ndarray
            cos(corrected_fa), shape (n_fa, x, y, z)
        tr_s : float
            TR in seconds
        shape : tuple
            Output shape (x, y, z)
        n_iterations : int
            Number of IRLS iterations (default: 3)

        Returns
        -------
        numpy.ndarray
            T1 map in seconds

        """
        from pmriprep.utils.dce_constants import (
            EPSILON_DENOM_ABS,
            EPSILON_DENOM_RTOL,
            EPSILON_E1_LOWER,
            EPSILON_E1_UPPER,
            EPSILON_WEIGHT_DENOM,
        )

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
        # Scale-relative degeneracy guard (amplitude-invariant collinearity test),
        # kept strictly positive so an all-zero-signal voxel (sum_x2=0) yields a
        # finite E1 (0, then clipped) rather than 0/0 = NaN.
        thr = np.maximum(EPSILON_DENOM_RTOL * (n_fa * sum_x2), EPSILON_DENOM_ABS)
        denom = np.where(
            np.abs(denom) < thr,
            np.where(denom >= 0, thr, -thr),
            denom,
        )
        e1 = (n_fa * sum_xy - sum_x * sum_y) / denom
        e1 = np.clip(e1, EPSILON_E1_LOWER, 1.0 - EPSILON_E1_UPPER)

        # Iterative reweighting
        for _ in range(n_iterations):
            # w_i = (sin(a_i) / (1 - E1 * cos(a_i)))^2. The denominator is meant to
            # become small (large weight) for the informative long-T1/small-flip
            # points, so it is floored only against a true divide-by-zero.
            denom_w = 1.0 - e1[np.newaxis, :] * cos_flat  # (n_fa, n_voxels)
            denom_w = np.where(
                np.abs(denom_w) < EPSILON_WEIGHT_DENOM,
                np.where(denom_w >= 0, EPSILON_WEIGHT_DENOM, -EPSILON_WEIGHT_DENOM),
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
            thr_reg = np.maximum(
                EPSILON_DENOM_RTOL * (sum_w * sum_wx2), EPSILON_DENOM_ABS
            )
            denom_reg = np.where(
                np.abs(denom_reg) < thr_reg,
                np.where(denom_reg >= 0, thr_reg, -thr_reg),
                denom_reg,
            )
            e1 = (sum_w * sum_wxy - sum_wx * sum_wy) / denom_reg
            e1 = np.clip(e1, EPSILON_E1_LOWER, 1.0 - EPSILON_E1_UPPER)

        # T1 = -TR / ln(E1)
        t1_flat = -tr_s / np.log(e1)
        t1_map: np.ndarray = t1_flat.reshape(shape)  # type: ignore[assignment]

        return t1_map

    def _fit_t1_nlls(
        self,
        signal_data: np.ndarray,
        sin_fa: np.ndarray,
        cos_fa: np.ndarray,
        tr_s: float,
        shape: tuple[int, ...],
        e1_init: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit T1 using nonlinear least squares in signal space.

        Uses a vectorized Levenberg-Marquardt (damped Gauss-Newton) solver
        operating in R1 = 1/T1 space, with analytical Jacobian. This avoids
        the correlated-noise bias of DESPOT1 linearization that causes
        systematic T1 underestimation when E1 > 0.999 (T1 >> TR).

        Reparameterization rationale: at E1 ~ 0.9999, dS/dT1 -> 0 for all
        flip angles, making the T1-space Jacobian degenerate. But dS/dR1
        remains finite as R1 -> 0 because E1 = exp(-TR*R1) and
        dE1/dR1 = -TR*exp(-TR*R1).

        With n_fa=2 flip angles and 2 parameters (R1, S0), the system is
        exactly determined — residual-based QC (R², chi²) is degenerate
        (always perfect). This is a known limitation documented here.

        Parameters
        ----------
        signal_data : numpy.ndarray
            Raw VFA signal, shape (n_fa, x, y, z).
        sin_fa : numpy.ndarray
            sin(corrected_fa), shape (n_fa, x, y, z).
        cos_fa : numpy.ndarray
            cos(corrected_fa), shape (n_fa, x, y, z).
        tr_s : float
            TR in seconds.
        shape : tuple
            Output volume shape (x, y, z).
        e1_init : numpy.ndarray, optional
            Optional initial E1 from WLLS, shape (n_voxels,).
            If None, computes a WLLS estimate as the starting point.
        mask : numpy.ndarray, optional
            Boolean mask of voxels to fit. If None, fits all voxels.

        Returns
        -------
        numpy.ndarray
            T1 map in seconds, shape (x, y, z). Unfitted voxels are 0.
        """
        from pmriprep.utils.dce_constants import EPSILON_E1_LOWER, EPSILON_E1_UPPER

        n_fa = signal_data.shape[0]
        n_voxels = int(np.prod(shape))

        # R1 bounds: [0.05, 100] s⁻¹ → T1 in [10ms, 20s]
        R1_MIN = 0.05
        R1_MAX = 100.0

        # Reshape to (n_fa, n_voxels)
        sig_flat = signal_data.reshape(n_fa, n_voxels)
        sin_flat = sin_fa.reshape(n_fa, n_voxels)
        cos_flat = cos_fa.reshape(n_fa, n_voxels)

        # Determine which voxels to fit
        if mask is not None:
            fit_idx = mask.ravel().nonzero()[0]
        else:
            fit_idx = np.arange(n_voxels)

        if len(fit_idx) == 0:
            return np.zeros(shape, dtype=np.float64)

        # Extract fitting subset: (n_fa, N_fit)
        S = sig_flat[:, fit_idx]
        sin_a = sin_flat[:, fit_idx]
        cos_a = cos_flat[:, fit_idx]

        # --- Initial guess for R1 ---
        if e1_init is not None:
            # Use WLLS E1 estimate → R1 = -ln(E1) / TR
            e1_sub = e1_init[fit_idx]
            # Upper bound uses the small EPSILON_E1_UPPER (1e-4) offset, not the
            # E1 lower clip value: at short pulse TR (e.g., 4 ms) an E1=0.99 cap
            # would limit T1 to ~397 ms, miscapping long-T1 tissue at the seed.
            e1_sub = np.clip(e1_sub, EPSILON_E1_LOWER, 1.0 - EPSILON_E1_UPPER)
            r1 = -np.log(e1_sub) / tr_s
        else:
            tan_fa = sin_fa / np.where(np.abs(cos_fa) < 1e-10, 1e-10, cos_fa)
            x_data = signal_data / np.where(
                np.abs(tan_fa) < 1e-10, 1e-10, tan_fa,
            )
            y_data = signal_data / np.where(
                np.abs(sin_fa) < 1e-10, 1e-10, sin_fa,
            )
            wlls = self._fit_t1_wlls(
                x_data, y_data, sin_fa, cos_fa, tr_s, shape,
            )
            e1_wlls = np.exp(
                -tr_s / np.where(wlls > 0, wlls, 1.0),
            ).ravel()
            e1_sub = np.clip(
                e1_wlls[fit_idx], EPSILON_E1_LOWER, 1.0 - EPSILON_E1_UPPER,
            )
            r1 = -np.log(e1_sub) / tr_s

        r1 = np.clip(r1, R1_MIN, R1_MAX)

        # Initial S0 estimate from first flip angle:
        # S = S0 * sin(a) * (1 - E1) / (1 - cos(a) * E1)
        E1 = np.exp(-tr_s * r1)  # (N_fit,)
        numer_s0 = 1.0 - cos_a[0] * E1
        denom_s0 = sin_a[0] * (1.0 - E1)
        denom_s0 = np.where(np.abs(denom_s0) < 1e-12, 1e-12, denom_s0)
        s0 = S[0] * numer_s0 / denom_s0
        s0 = np.maximum(s0, 1e-6)

        # --- Vectorized Levenberg-Marquardt ---
        lam = 1e-4  # Fixed damping parameter
        max_iter = 20
        tol = 1e-6

        for _iteration in range(max_iter):
            E1 = np.exp(-tr_s * r1)  # (N_fit,)
            one_minus_E1 = 1.0 - E1
            # Predicted signal: S_pred(a) = S0 * sin(a) * (1-E1) / (1 - cos(a)*E1)
            denom_model = 1.0 - cos_a * E1[np.newaxis, :]  # (n_fa, N_fit)
            denom_model = np.where(
                np.abs(denom_model) < 1e-12, 1e-12, denom_model,
            )
            S_pred = s0[np.newaxis, :] * sin_a * one_minus_E1[np.newaxis, :] / denom_model

            # Residuals: (n_fa, N_fit)
            r_vec = S - S_pred

            # Analytical Jacobian: dS/dS0 and dS/dR1
            # dS/dS0 = sin(a) * (1-E1) / (1 - cos(a)*E1) = S_pred / S0
            J_s0 = S_pred / np.where(np.abs(s0) < 1e-12, 1e-12, s0)[np.newaxis, :]

            # dS/dR1 = S0 * sin(a) * dE1/dR1 * d/dE1[(1-E1)/(1-cos(a)*E1)]
            # dE1/dR1 = -TR * E1
            # d/dE1[(1-E1)/(1-cos(a)*E1)] = [-1*(1-cos(a)*E1) - (1-E1)*(-cos(a))] / (1-cos(a)*E1)^2
            #   = [-(1-cos(a)*E1) + cos(a)*(1-E1)] / (1-cos(a)*E1)^2
            #   = [-1 + cos(a)*E1 + cos(a) - cos(a)*E1] / (1-cos(a)*E1)^2
            #   = (cos(a) - 1) / (1-cos(a)*E1)^2
            dE1_dR1 = -tr_s * E1  # (N_fit,)
            d_ratio_dE1 = (cos_a - 1.0) / (denom_model ** 2)  # (n_fa, N_fit)
            J_r1 = (
                s0[np.newaxis, :] * sin_a * dE1_dR1[np.newaxis, :] * d_ratio_dE1
            )  # (n_fa, N_fit)

            # Gauss-Newton: solve (J^T J + lam*I) dp = J^T r
            # For 2 params, this is a 2x2 system per voxel — fully vectorizable.
            # J^T J entries (all shape N_fit):
            JtJ_00 = np.sum(J_r1 * J_r1, axis=0)      # d²/dR1²
            JtJ_01 = np.sum(J_r1 * J_s0, axis=0)      # cross
            JtJ_11 = np.sum(J_s0 * J_s0, axis=0)      # d²/dS0²

            # J^T r entries:
            Jtr_0 = np.sum(J_r1 * r_vec, axis=0)       # dR1
            Jtr_1 = np.sum(J_s0 * r_vec, axis=0)       # dS0

            # Add damping
            JtJ_00_d = JtJ_00 + lam
            JtJ_11_d = JtJ_11 + lam

            # 2x2 solve: Cramer's rule
            det = JtJ_00_d * JtJ_11_d - JtJ_01 * JtJ_01
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)

            d_r1 = (JtJ_11_d * Jtr_0 - JtJ_01 * Jtr_1) / det
            d_s0 = (JtJ_00_d * Jtr_1 - JtJ_01 * Jtr_0) / det

            # Update and enforce bounds
            r1_new = np.clip(r1 + d_r1, R1_MIN, R1_MAX)
            s0_new = np.maximum(s0 + d_s0, 1e-6)

            # Convergence check: relative parameter change
            rel_change_r1 = np.abs(r1_new - r1) / np.maximum(np.abs(r1), 1e-12)
            rel_change_s0 = np.abs(s0_new - s0) / np.maximum(np.abs(s0), 1e-12)
            max_change = np.maximum(rel_change_r1, rel_change_s0)

            r1 = r1_new
            s0 = s0_new

            if np.max(max_change) < tol:
                break

        # Convert R1 → T1
        t1_flat = np.zeros(n_voxels, dtype=np.float64)
        t1_flat[fit_idx] = 1.0 / np.clip(r1, R1_MIN, R1_MAX)

        return t1_flat.reshape(shape)

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
    target_flip_angle = traits.Float(
        mandatory=True,
        desc="DCE flip angle used to select the closest-angle VFA reference",
    )
    skip_coregistration = traits.Bool(
        False,
        usedefault=True,
        desc="Skip inter-volume coregistration (if affines are identical)",
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

    This interface selects the VFA angle closest to the DCE flip angle as a
    nominal contrast proxy for the shared DCE and AFI registration reference.
    It skips FLIRT when the affines are identical and otherwise performs 6DOF
    rigid coregistration of non-reference volumes.

    Flip angle alone does not determine SPGR contrast; differing TR, TE,
    spoiling, and tissue T1 can make the selected images dissimilar.

    Parameters
    ----------
    vfa_files : list of str
        VFA image paths at different flip angles.
    flip_angles : list of float
        Nominal flip angles in degrees.
    target_flip_angle : float
        DCE flip angle in degrees used for nominal-angle reference selection.
    skip_coregistration : bool
        If True, skip FLIRT coregistration.

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

    @staticmethod
    def _select_reference_index(
        flip_angles: list[float],
        target_flip_angle: float,
    ) -> int:
        """Select the VFA angle closest to the DCE acquisition angle."""
        angles = np.asarray(flip_angles, dtype=np.float64)
        if angles.size < 2:
            raise ValueError("VFA T1 mapping requires at least two flip angles")
        if (
            not np.all(np.isfinite(angles))
            or np.any(angles <= 0.0)
            or not np.isfinite(target_flip_angle)
            or target_flip_angle <= 0.0
        ):
            raise ValueError(
                "VFA and target DCE flip angles must be finite and positive"
            )
        if np.unique(angles).size < 2:
            raise ValueError(
                "VFA T1 mapping requires at least two distinct flip angles"
            )
        differences = np.abs(angles - float(target_flip_angle))
        candidates = np.flatnonzero(
            np.isclose(differences, np.min(differences), rtol=0.0, atol=1e-8)
        )
        return int(candidates[np.argmax(angles[candidates])])

    @staticmethod
    def _motion_warning_state(
        *,
        translation_mm: float,
        rotation_deg: float,
        min_voxel_size_mm: float,
        gradient_ncc_delta: float,
        foreground_dice_delta: float,
        spgr_nrmse_fractional_change: float,
    ) -> str:
        """Classify concordant evidence of inter-VFA misalignment.

        Parameters
        ----------
        translation_mm : float
            Norm of the candidate rigid translation in millimetres.
        rotation_deg : float
            Norm of the candidate rigid rotation in degrees.
        min_voxel_size_mm : float
            Smallest spatial voxel dimension in millimetres.
        gradient_ncc_delta : float
            Candidate-minus-identity gradient correlation.
        foreground_dice_delta : float
            Candidate-minus-identity foreground overlap.
        spgr_nrmse_fractional_change : float
            Fractional reduction in median multi-flip SPGR residual.

        Returns
        -------
        str
            ``"WARN"`` only when transform, anatomy, and SPGR evidence agree;
            otherwise ``"PASS"``.
        """
        nontrivial_transform = (
            translation_mm >= 0.25 * min_voxel_size_mm
            or rotation_deg >= 0.3
        )
        anatomy_improved = (
            gradient_ncc_delta >= 0.01
            and foreground_dice_delta >= -0.01
        )
        signal_model_improved = spgr_nrmse_fractional_change >= 0.05
        return (
            "WARN"
            if nontrivial_transform and anatomy_improved and signal_model_improved
            else "PASS"
        )

    def _assess_same_grid_motion(
        self,
        *,
        vfa_files: list[str],
        vfa_imgs: list[Any],
        vfa_data: list[npt.NDArray[np.float64]],
        flip_angles: list[float],
        ref_idx: int,
    ) -> dict[str, Any]:
        """Estimate, but do not apply, rigid inter-VFA motion.

        The candidate registration uses robust gradient-magnitude images so
        flip-angle-dependent tissue contrast does not directly drive FLIRT.
        A warning requires agreement between anatomical edge alignment and the
        multi-flip SPGR signal model.
        """
        import subprocess

        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import (  # type: ignore[import-untyped]
            binary_closing,
            binary_fill_holes,
            binary_opening,
            gaussian_filter,
            label,
        )
        from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

        def _foreground(data: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
            positive = data[np.isfinite(data) & (data > 0)]
            if positive.size < 100:
                return np.zeros(data.shape, dtype=bool)
            low, high = np.percentile(positive, [2, 98])
            mask = np.isfinite(data) & (data > low + 0.08 * (high - low))
            mask = binary_opening(mask, iterations=1)
            mask = binary_closing(mask, iterations=2)
            mask = binary_fill_holes(mask)
            components, count = label(mask)
            if count:
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                mask = components == int(np.argmax(sizes))
            return np.asarray(mask, dtype=bool)

        def _gradient(
            data: npt.NDArray[np.float64],
            zooms: tuple[float, ...],
        ) -> npt.NDArray[np.float32]:
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return np.zeros(data.shape, dtype=np.float32)
            low, high = np.percentile(finite, [1, 99])
            scaled = np.clip((data - low) / max(high - low, 1e-8), 0.0, 1.0)
            sigma = tuple(max(0.25, 1.0 / float(zoom)) for zoom in zooms[:3])
            smooth = gaussian_filter(scaled, sigma=sigma)
            derivatives = np.gradient(smooth, *zooms[:3])
            magnitude = np.sqrt(sum(item * item for item in derivatives))
            scale = max(float(np.percentile(magnitude, 99)), 1e-8)
            return np.asarray(np.clip(magnitude / scale, 0.0, 1.0), dtype=np.float32)

        def _ncc(
            fixed: npt.NDArray[Any],
            moving: npt.NDArray[Any],
            mask: npt.NDArray[np.bool_],
        ) -> float:
            valid = mask & np.isfinite(fixed) & np.isfinite(moving)
            if np.count_nonzero(valid) < 100:
                return float("nan")
            fixed_values = fixed[valid]
            moving_values = moving[valid]
            if np.std(fixed_values) == 0 or np.std(moving_values) == 0:
                return float("nan")
            return float(np.corrcoef(fixed_values, moving_values)[0, 1])

        def _dice(
            fixed: npt.NDArray[np.bool_],
            moving: npt.NDArray[np.bool_],
        ) -> float:
            denominator = int(np.sum(fixed) + np.sum(moving))
            if denominator == 0:
                return float("nan")
            return 2.0 * float(np.count_nonzero(fixed & moving)) / denominator

        def _median_spgr_nrmse(
            signals: npt.NDArray[np.float64],
            mask: npt.NDArray[np.bool_],
        ) -> float:
            angles = np.radians(np.asarray(flip_angles, dtype=np.float64))
            angle_shape = (len(angles),) + (1,) * (signals.ndim - 1)
            angles = angles.reshape(angle_shape)
            x_values = signals / np.tan(angles)
            y_values = signals / np.sin(angles)
            x_mean = np.mean(x_values, axis=0)
            y_mean = np.mean(y_values, axis=0)
            denominator = np.sum((x_values - x_mean) ** 2, axis=0)
            e1 = np.sum(
                (x_values - x_mean) * (y_values - y_mean),
                axis=0,
            ) / np.maximum(denominator, 1e-12)
            valid = mask & np.isfinite(e1) & (e1 > 0) & (e1 < 0.999999)
            basis = np.sin(angles) * (1.0 - e1) / np.maximum(
                1.0 - np.cos(angles) * e1,
                1e-12,
            )
            amplitude = np.sum(signals * basis, axis=0) / np.maximum(
                np.sum(basis * basis, axis=0),
                1e-12,
            )
            predicted = amplitude * basis
            nrmse = np.sqrt(np.mean((signals - predicted) ** 2, axis=0))
            nrmse /= np.maximum(np.mean(signals, axis=0), 1e-8)
            values = nrmse[valid & np.isfinite(nrmse)]
            return float(np.median(values)) if values.size >= 100 else float("nan")

        ref_img = vfa_imgs[ref_idx]
        ref_data = vfa_data[ref_idx]
        ref_mask = _foreground(ref_data)
        ref_gradient = _gradient(
            ref_data,
            tuple(float(value) for value in ref_img.header.get_zooms()[:3]),
        )
        out_dir = Path.cwd()
        ref_gradient_file = out_dir / "vfa_motion_reference_gradient.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.where(ref_mask, ref_gradient, 0.0).astype(np.float32),
                ref_img.affine,
            ),
            ref_gradient_file,
        )
        identity_signals = np.stack(vfa_data)
        identity_nrmse = _median_spgr_nrmse(identity_signals, ref_mask)
        min_voxel_size = min(float(value) for value in ref_img.header.get_zooms()[:3])
        comparisons: list[dict[str, Any]] = []

        for index, (path, img, data) in enumerate(
            zip(vfa_files, vfa_imgs, vfa_data, strict=True)
        ):
            if index == ref_idx:
                continue
            moving_mask = _foreground(data)
            moving_gradient = _gradient(
                data,
                tuple(float(value) for value in img.header.get_zooms()[:3]),
            )
            moving_gradient_file = out_dir / f"vfa_motion_{index:02d}_gradient.nii.gz"
            matrix_file = out_dir / f"vfa_motion_{index:02d}.mat"
            candidate_file = out_dir / f"vfa_motion_{index:02d}_candidate.nii.gz"
            candidate_mask_file = out_dir / f"vfa_motion_{index:02d}_mask.nii.gz"
            moving_mask_file = out_dir / f"vfa_motion_{index:02d}_native_mask.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.where(moving_mask, moving_gradient, 0.0).astype(np.float32),
                    img.affine,
                ),
                moving_gradient_file,
            )
            nib.save(
                nib.Nifti1Image(moving_mask.astype(np.uint8), img.affine),
                moving_mask_file,
            )
            estimate = subprocess.run(
                [
                    "flirt", "-in", str(moving_gradient_file),
                    "-ref", str(ref_gradient_file),
                    "-omat", str(matrix_file), "-dof", "6",
                    "-cost", "normcorr", "-searchrx", "-5", "5",
                    "-searchry", "-5", "5", "-searchrz", "-5", "5",
                ],
                capture_output=True,
                text=True,
            )
            if estimate.returncode != 0 or not matrix_file.exists():
                raise RuntimeError(
                    f"FLIRT candidate estimation failed for VFA index {index}: "
                    f"{estimate.stderr.strip()}"
                )
            for source, destination, interpolation in (
                (path, candidate_file, "spline"),
                (moving_mask_file, candidate_mask_file, "nearestneighbour"),
            ):
                apply = subprocess.run(
                    [
                        "flirt", "-in", str(source), "-ref", str(vfa_files[ref_idx]),
                        "-out", str(destination), "-init", str(matrix_file),
                        "-applyxfm", "-interp", interpolation,
                    ],
                    capture_output=True,
                    text=True,
                )
                if apply.returncode != 0:
                    raise RuntimeError(
                        f"FLIRT candidate application failed for VFA index {index}: "
                        f"{apply.stderr.strip()}"
                    )

            candidate = np.asarray(
                nib.load(candidate_file).get_fdata(),  # type: ignore[attr-defined]
                dtype=np.float64,
            )
            candidate_mask = (
                nib.load(candidate_mask_file).get_fdata() > 0.5  # type: ignore[attr-defined]
            )
            common_identity = ref_mask & moving_mask
            common_candidate = ref_mask & candidate_mask
            identity_edge_ncc = _ncc(ref_gradient, moving_gradient, common_identity)
            candidate_gradient = _gradient(
                candidate,
                tuple(float(value) for value in ref_img.header.get_zooms()[:3]),
            )
            candidate_edge_ncc = _ncc(
                ref_gradient,
                candidate_gradient,
                common_candidate,
            )
            identity_dice = _dice(ref_mask, moving_mask)
            candidate_dice = _dice(ref_mask, candidate_mask)
            candidate_signals = identity_signals.copy()
            candidate_signals[index] = candidate
            candidate_nrmse = _median_spgr_nrmse(candidate_signals, ref_mask)
            nrmse_change = (
                (identity_nrmse - candidate_nrmse) / identity_nrmse
                if np.isfinite(identity_nrmse)
                and identity_nrmse > 0
                and np.isfinite(candidate_nrmse)
                else float("nan")
            )
            matrix = np.loadtxt(matrix_file)
            linear = np.asarray(matrix[:3, :3], dtype=np.float64)
            left, _, right = np.linalg.svd(linear)
            rotation = left @ right
            rotation_deg = float(
                np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())
                * 180.0 / np.pi
            )
            translation_mm = float(np.linalg.norm(matrix[:3, 3]))
            state = self._motion_warning_state(
                translation_mm=translation_mm,
                rotation_deg=rotation_deg,
                min_voxel_size_mm=min_voxel_size,
                gradient_ncc_delta=candidate_edge_ncc - identity_edge_ncc,
                foreground_dice_delta=candidate_dice - identity_dice,
                spgr_nrmse_fractional_change=nrmse_change,
            )
            comparisons.append(
                {
                    "moving_index": index,
                    "flip_angle_deg": float(flip_angles[index]),
                    "candidate_translation_mm": translation_mm,
                    "candidate_rotation_deg": rotation_deg,
                    "candidate_rigid_matrix": [
                        [float(value) for value in row]
                        for row in np.asarray(matrix, dtype=np.float64)
                    ],
                    "identity_gradient_ncc": identity_edge_ncc,
                    "candidate_gradient_ncc": candidate_edge_ncc,
                    "gradient_ncc_delta": candidate_edge_ncc - identity_edge_ncc,
                    "identity_foreground_dice": identity_dice,
                    "candidate_foreground_dice": candidate_dice,
                    "foreground_dice_delta": candidate_dice - identity_dice,
                    "identity_spgr_median_nrmse": identity_nrmse,
                    "candidate_spgr_median_nrmse": candidate_nrmse,
                    "spgr_nrmse_fractional_reduction": nrmse_change,
                    "qc_status": state,
                }
            )

        overall = (
            "WARN"
            if any(item["qc_status"] == "WARN" for item in comparisons)
            else "PASS"
        )
        result: dict[str, Any] = {
            "qc_status": overall,
            "reference_index": ref_idx,
            "reference_flip_angle_deg": float(flip_angles[ref_idx]),
            "correction_applied": False,
            "estimator": "gradient_magnitude_rigid_candidate",
            "comparisons": comparisons,
        }
        if overall == "WARN":
            result["warning"] = (
                "Potential inter-VFA misalignment detected; VFA images were not "
                "resampled. Review source images and T1-map QC."
            )
        return result

    def _run_interface(self, runtime: Any) -> Any:
        """Execute VFA preprocessing."""
        import shutil
        import subprocess

        import nibabel as nib  # type: ignore[import-untyped]

        vfa_files = list(self.inputs.vfa_files)
        flip_angles = list(self.inputs.flip_angles)
        n_fa = len(vfa_files)
        if len(flip_angles) != n_fa:
            raise ValueError(
                "VFA file and FlipAngle counts differ: "
                f"{n_fa} files, {len(flip_angles)} angles"
            )

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

        target_flip_angle = float(self.inputs.target_flip_angle)
        ref_idx = self._select_reference_index(flip_angles, target_flip_angle)
        qc_metrics["reference_selection"] = {
            "method": "closest_nominal_flip_angle_to_DCE",
            "target_dce_flip_angle_deg": target_flip_angle,
            "absolute_angle_differences_deg": [
                abs(float(angle) - target_flip_angle) for angle in flip_angles
            ],
            "tie_break": "higher_flip_angle",
        }
        qc_metrics["reference_index"] = ref_idx
        qc_metrics["reference_flip_angle_deg"] = float(flip_angles[ref_idx])

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
        if affines_identical:
            try:
                qc_metrics["motion_assessment"] = self._assess_same_grid_motion(
                    vfa_files=vfa_files,
                    vfa_imgs=vfa_imgs,
                    vfa_data=vfa_data,
                    flip_angles=flip_angles,
                    ref_idx=ref_idx,
                )
            except Exception as exc:
                logger.warning("VFA motion diagnostic was not evaluated: %s", exc)
                qc_metrics["motion_assessment"] = {
                    "qc_status": "NOT_EVALUATED",
                    "reference_index": ref_idx,
                    "reference_flip_angle_deg": float(flip_angles[ref_idx]),
                    "correction_applied": False,
                    "reason": str(exc),
                }
        else:
            qc_metrics["motion_assessment"] = {
                "qc_status": "NOT_EVALUATED",
                "reference_index": ref_idx,
                "reference_flip_angle_deg": float(flip_angles[ref_idx]),
                    "correction_applied": False,
                    "reason": "VFA grids differ; required grid harmonization used",
                }

        # Determine whether to coregister
        do_coreg = (
            not self.inputs.skip_coregistration
            and not affines_identical
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

        # Set outputs
        self._results["out_files"] = out_files
        self._results["reference_file"] = out_files[ref_idx]
        self._results["reference_index"] = ref_idx
        self._results["qc_metrics"] = qc_metrics

        return runtime

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
    volume_tr = traits.Float(
        0.0,
        usedefault=True,
        desc="Volume TR in seconds for TR-adaptive baseline smoothing (0 = use default sigma)",
    )
    apply_motion_correction = traits.Bool(
        True,
        usedefault=True,
        desc="Apply motion correction. When False, the input is assumed already "
        "motion-corrected: mcflirt still runs to report residual motion for QC, "
        "but the uncorrected series is passed through to avoid a second "
        "interpolation.",
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


@dataclass(frozen=True)
class _DCEPreprocessPreparation:
    """Loaded DCE series and resolved baseline state for preprocessing."""

    in_file: str
    out_file: str
    data: npt.NDArray[Any]
    affine: npt.NDArray[Any]
    header: Any
    n_volumes: int
    brain_mask: Optional[npt.NDArray[np.bool_]]
    baseline_frames: int
    baseline_indices: tuple[int, ...]
    bolus_frame: int
    baseline_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _DCEMCFLIRTPlan:
    """Resolved MCFLIRT reference, output base, and command."""

    out_base: str
    reference_file: Optional[str]
    command: tuple[str, ...]


@dataclass(frozen=True)
class _DCEMotionQC:
    """Reduced MCFLIRT motion parameters and QC state."""

    fd: tuple[float, ...]
    fd_mean: float
    fd_max: float
    n_outliers: int
    outlier_frames: tuple[int, ...]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _DCEPreprocessOutputs:
    """Serialized DCE preprocessing output paths."""

    out_file: str
    params_file: str
    mean_file: str
    baseline_mean_file: str


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

    When ``baseline_frames=0``, bolus arrival is estimated from an ensemble of
    change-point and derivative detectors applied to the brain signal time
    course. The detected arrival frame defines the pre-contrast baseline.

    For DCE with gadolinium contrast, the signal typically INCREASES (T1-shortening)
    at bolus arrival, unlike DSC where signal DECREASES (T2* effect).

    **Motion Correction:**

    Uses FSL mcflirt with the baseline mean as reference by default. This ensures
    the motion correction reference has optimal contrast for registration (pre-contrast).

    Motion parameters are output in FSL format:
    rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm)

    Framewise displacement formula: ``FD = |Δtrans| + r × |Δrot|``

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
    MotionCorrection : DSC motion correction interface
    """

    input_spec = DCEPreprocessInputSpec
    output_spec = DCEPreprocessOutputSpec

    def _prepare_preprocess_inputs(self) -> _DCEPreprocessPreparation:
        """Load DCE inputs and resolve the baseline window.

        Returns
        -------
        _DCEPreprocessPreparation
            Native image state, optional same-grid mask, and the existing
            detected or configured baseline state.
        """
        out_file = self._gen_outfilename()
        in_file = str(self.inputs.in_file)
        dce_img = nib.load(in_file)
        dce_data = dce_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
        affine = dce_img.affine  # type: ignore[attr-defined]
        header = dce_img.header  # type: ignore[attr-defined]
        n_volumes = dce_data.shape[3]

        brain_mask: Optional[npt.NDArray[np.bool_]] = None
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

        from pmriprep.utils.dce_constants import DEFAULT_BASELINE_FRAMES

        configured_baseline = self.inputs.baseline_frames > 0
        if configured_baseline:
            baseline_frames = self.inputs.baseline_frames
            bolus_frame, baseline_diag = self._detect_bolus_arrival(
                dce_data, brain_mask
            )
        else:
            baseline_frames, bolus_frame, baseline_diag = self._auto_detect_baseline(
                dce_data, brain_mask
            )

        detected_bolus_frame = int(bolus_frame)
        usable_fallback = max(1, min(DEFAULT_BASELINE_FRAMES, n_volumes - 1))
        if n_volumes - 1 < DEFAULT_BASELINE_FRAMES:
            fallback_reason = "insufficient_volumes_for_minimum_baseline"
        elif configured_baseline and baseline_frames < DEFAULT_BASELINE_FRAMES:
            fallback_reason = "configured_baseline_below_minimum"
        elif (
            not configured_baseline
            and (
                baseline_frames < DEFAULT_BASELINE_FRAMES
                or bolus_frame < DEFAULT_BASELINE_FRAMES
            )
        ):
            fallback_reason = "detected_bolus_before_minimum_baseline"
        else:
            fallback_reason = None

        if fallback_reason is not None:
            baseline_frames = usable_fallback
            if fallback_reason != "configured_baseline_below_minimum":
                bolus_frame = usable_fallback
            baseline_diag = dict(baseline_diag)
            baseline_diag.update({
                "baseline_status": "POOR",
                "baseline_minimum_frames": DEFAULT_BASELINE_FRAMES,
                "baseline_frames_used": baseline_frames,
                "detected_bolus_frame": detected_bolus_frame,
                "baseline_fallback_applied": True,
                "baseline_fallback_reason": fallback_reason,
                "baseline_warning": (
                    "DCE acquisition or baseline configuration does not provide "
                    "the required "
                    f"{DEFAULT_BASELINE_FRAMES}-frame pre-contrast baseline; "
                    f"{baseline_frames} frames were used as a computational "
                    "fallback. The S0 estimate and quantitative PK maps are "
                    "unreliable."
                ),
            })

        baseline_frames = max(1, min(baseline_frames, n_volumes - 1))
        if bolus_frame > 0 and baseline_frames > bolus_frame:
            baseline_frames = bolus_frame

        return _DCEPreprocessPreparation(
            in_file=in_file,
            out_file=out_file,
            data=dce_data,
            affine=affine,
            header=header,
            n_volumes=n_volumes,
            brain_mask=brain_mask,
            baseline_frames=baseline_frames,
            baseline_indices=tuple(range(baseline_frames)),
            bolus_frame=bolus_frame,
            baseline_diagnostics=baseline_diag,
        )

    def _prepare_mcflirt_plan(
        self, preparation: _DCEPreprocessPreparation
    ) -> _DCEMCFLIRTPlan:
        """Create the existing MCFLIRT reference and command plan.

        Parameters
        ----------
        preparation : _DCEPreprocessPreparation
            Loaded DCE image and resolved baseline state.

        Returns
        -------
        _DCEMCFLIRTPlan
            Output base, optional temporary reference, and command arguments.
        """
        ref_vol = self.inputs.ref_vol
        ref_vol_index: Optional[int] = None
        if ref_vol == "baseline_mean":
            baseline_data = np.mean(
                preparation.data[..., :preparation.baseline_frames], axis=3
            )
            ref_file = str(
                Path(preparation.out_file).parent / "_tmp_baseline_mean.nii.gz"
            )
            ref_img = nib.Nifti1Image(
                baseline_data.astype(np.float32),
                preparation.affine,
                preparation.header,
            )
            nib.save(ref_img, ref_file)
        elif ref_vol == "mean":
            ref_file = None
        elif ref_vol == "first":
            ref_vol_index = 0
            ref_file = None
        elif isinstance(ref_vol, int):
            ref_vol_index = ref_vol
            ref_file = None
        else:
            ref_file = None

        out_path = Path(preparation.out_file)
        if out_path.name.endswith(".nii.gz"):
            out_base = str(out_path)[:-7]
        elif out_path.name.endswith(".nii"):
            out_base = str(out_path)[:-4]
        else:
            out_base = str(out_path)

        command = [
            "mcflirt",
            "-in", preparation.in_file,
            "-out", out_base,
            "-plots",
            "-stats",
            "-mats",
        ]
        if ref_vol == "baseline_mean" and ref_file:
            command.extend(["-reffile", ref_file])
        elif ref_vol == "mean":
            command.append("-meanvol")
        elif ref_vol_index is not None:
            command.extend(["-refvol", str(ref_vol_index)])

        command.extend(["-cost", self.inputs.cost])
        if self.inputs.interpolation == "sinc":
            command.append("-sinc_final")
        elif self.inputs.interpolation == "spline":
            command.append("-spline_final")
        command.extend(["-stages", "4"])
        return _DCEMCFLIRTPlan(out_base, ref_file, tuple(command))

    def _summarize_motion_qc(
        self,
        params_file: str,
        preparation: _DCEPreprocessPreparation,
    ) -> _DCEMotionQC:
        """Reduce MCFLIRT parameters and return zeroed QC on read failure.

        Parameters
        ----------
        params_file : str
            MCFLIRT six-column motion-parameter file.
        preparation : _DCEPreprocessPreparation
            Baseline and volume state included in the QC record.

        Returns
        -------
        _DCEMotionQC
            Framewise displacement, outliers, and composite QC metrics.
        """
        try:
            motion_params = np.loadtxt(params_file)
            fd = self._calculate_fd(motion_params, float(self.inputs.head_radius))
            fd_list = tuple(float(value) for value in fd)
            fd_mean = float(np.mean(fd[1:]))
            fd_max = float(np.max(fd))
            motion_thresh = float(self.inputs.motion_thresh)
            outlier_mask = fd > motion_thresh
            n_outliers = int(np.sum(outlier_mask))
            outlier_frames = tuple(int(i) for i in np.where(outlier_mask)[0])

            from pmriprep.utils.qc_standards import get_composite_motion_status

            pct_outliers = 100.0 * n_outliers / max(1, preparation.n_volumes)
            motion_status = get_composite_motion_status(
                fd_mean, fd_max, pct_outliers,
            )
            metrics = self._preprocess_qc_context(preparation)
            metrics.update({
                "fd_mean": fd_mean,
                "fd_max": fd_max,
                "motion_status": motion_status,
                "motion_qc_available": True,
                "fd_list": list(fd_list),
                "n_outliers": n_outliers,
                "outlier_frames": list(outlier_frames),
                "motion_thresh": motion_thresh,
                "percent_outliers": pct_outliers,
            })
            if preparation.baseline_diagnostics.get("confidence", "high") != "high":
                logger.warning(
                    "DCE baseline detection low confidence: method=%s, "
                    "confidence=%s, bolus_frame=%d (S0 window may include "
                    "enhanced frames; concentration may be biased).",
                    preparation.baseline_diagnostics.get("method"),
                    preparation.baseline_diagnostics.get("confidence"),
                    preparation.bolus_frame,
                )
            return _DCEMotionQC(
                fd_list,
                fd_mean,
                fd_max,
                n_outliers,
                outlier_frames,
                metrics,
            )
        except Exception as exc:
            logger.warning("DCE motion QC calculation failed: %s", exc)
            metrics = self._preprocess_qc_context(preparation)
            metrics.update({
                "motion_status": "UNKNOWN",
                "motion_qc_available": False,
                "motion_qc_error": type(exc).__name__,
                "motion_qc_warning": (
                    "DCE motion parameters could not be summarized. "
                    "Motion quality is unknown; review the motion plot and "
                    "quantitative maps before use."
                ),
            })
            return _DCEMotionQC(
                (),
                0.0,
                0.0,
                0,
                (),
                metrics,
            )

    def _preprocess_qc_context(
        self, preparation: _DCEPreprocessPreparation,
    ) -> dict[str, Any]:
        """Build baseline and execution provenance shared by both motion paths.

        Parameters
        ----------
        preparation : _DCEPreprocessPreparation
            Resolved acquisition and baseline state.

        Returns
        -------
        dict
            Common preprocessing QC metrics, including any baseline fallback.
        """
        metrics = {
            "n_volumes": preparation.n_volumes,
            "baseline_frames": preparation.baseline_frames,
            "bolus_arrival_frame": preparation.bolus_frame,
            "baseline_detection_method": preparation.baseline_diagnostics.get(
                "method", "unknown"
            ),
            "baseline_detection_confidence": preparation.baseline_diagnostics.get(
                "confidence", "unknown"
            ),
            "reference_type": str(self.inputs.ref_vol),
            "motion_applied": bool(self.inputs.apply_motion_correction),
        }
        for key in (
            "baseline_status",
            "baseline_minimum_frames",
            "baseline_frames_used",
            "detected_bolus_frame",
            "baseline_fallback_applied",
            "baseline_fallback_reason",
            "baseline_warning",
        ):
            if key in preparation.baseline_diagnostics:
                metrics[key] = preparation.baseline_diagnostics[key]
        return metrics

    def _run_mcflirt(self, plan: _DCEMCFLIRTPlan) -> str:
        """Execute a prepared MCFLIRT command and return its parameter file.

        Parameters
        ----------
        plan : _DCEMCFLIRTPlan
            Command, output base, and optional temporary reference.

        Returns
        -------
        str
            Expected MCFLIRT motion-parameter path.
        """
        import subprocess

        try:
            result = subprocess.run(plan.command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"mcflirt failed: {result.stderr}")
        finally:
            if self.inputs.ref_vol == "baseline_mean" and plan.reference_file:
                try:
                    Path(plan.reference_file).unlink()
                except Exception:
                    pass
        return f"{plan.out_base}.par"

    def _write_preprocess_outputs(
        self,
        preparation: _DCEPreprocessPreparation,
        mcflirt_plan: _DCEMCFLIRTPlan,
        params_file: str,
    ) -> _DCEPreprocessOutputs:
        """Select the processed series and write whole/baseline means.

        Parameters
        ----------
        preparation : _DCEPreprocessPreparation
            Input path and resolved baseline state.
        mcflirt_plan : _DCEMCFLIRTPlan
            Output base locating the realigned MCFLIRT series.
        params_file : str
            Motion-parameter output returned with the serialized products.

        Returns
        -------
        _DCEPreprocessOutputs
            Paths for the selected series, motion parameters, and mean images.
        """
        moco_path = f"{mcflirt_plan.out_base}.nii.gz"
        if not Path(moco_path).exists():
            raise FileNotFoundError(
                f"Motion-corrected output not found: {moco_path}"
            )

        if bool(self.inputs.apply_motion_correction):
            series_path = moco_path
        else:
            shutil.copyfile(preparation.in_file, preparation.out_file)
            series_path = preparation.in_file

        series_img = nib.load(series_path)
        series_data = series_img.get_fdata()  # type: ignore[attr-defined]
        mean_out = self._gen_mean_filename()
        mean_data = np.mean(series_data, axis=3).astype(np.float32)
        mean_nii = nib.Nifti1Image(
            mean_data,
            series_img.affine,  # type: ignore[attr-defined]
            series_img.header,  # type: ignore[attr-defined]
        )
        nib.save(mean_nii, mean_out)
        if not Path(mean_out).exists():
            raise FileNotFoundError(f"Failed to create mean image at: {mean_out}")

        baseline_mean_out = self._gen_baseline_mean_filename()
        baseline_data = np.mean(
            series_data[..., :preparation.baseline_frames], axis=3
        )
        baseline_nii = nib.Nifti1Image(
            baseline_data.astype(np.float32),
            series_img.affine,  # type: ignore[attr-defined]
            series_img.header,  # type: ignore[attr-defined]
        )
        nib.save(baseline_nii, baseline_mean_out)
        return _DCEPreprocessOutputs(
            preparation.out_file,
            params_file,
            mean_out,
            baseline_mean_out,
        )

    def _assemble_preprocess_results(
        self,
        preparation: _DCEPreprocessPreparation,
        motion_qc: _DCEMotionQC,
        outputs: _DCEPreprocessOutputs,
    ) -> None:
        """Populate the Nipype result contract from typed preprocessing state."""
        self._results["fd"] = list(motion_qc.fd)
        self._results["fd_mean"] = motion_qc.fd_mean
        self._results["fd_max"] = motion_qc.fd_max
        self._results["n_outliers"] = motion_qc.n_outliers
        self._results["outlier_frames"] = list(motion_qc.outlier_frames)
        self._results["out_file"] = outputs.out_file
        self._results["out_params"] = outputs.params_file
        self._results["mean_img"] = outputs.mean_file
        self._results["baseline_mean_img"] = outputs.baseline_mean_file
        self._results["baseline_frames"] = preparation.baseline_frames
        self._results["baseline_indices"] = list(preparation.baseline_indices)
        self._results["bolus_arrival_frame"] = preparation.bolus_frame
        self._results["qc_metrics"] = motion_qc.metrics

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE preprocessing."""
        preparation = self._prepare_preprocess_inputs()
        mcflirt_plan = self._prepare_mcflirt_plan(preparation)
        params_file = self._run_mcflirt(mcflirt_plan)
        motion_qc = self._summarize_motion_qc(params_file, preparation)
        outputs = self._write_preprocess_outputs(
            preparation, mcflirt_plan, params_file
        )
        self._assemble_preprocess_results(preparation, motion_qc, outputs)

        return runtime

    def _brain_signal_timecourse(
        self,
        dce_data: np.ndarray,
        brain_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute brain signal time course for bolus detection.

        Uses the 95th percentile across brain voxels to capture
        arterial/early-enhancing timing without the ~1-frame transit
        delay that affects the mean brain signal.  Falls back to mean
        when the mask has too few voxels for a stable percentile.

        Parameters
        ----------
        dce_data : numpy.ndarray
            4D DCE data array (x, y, z, t).
        brain_mask : numpy.ndarray or None
            Optional 3D boolean brain mask.

        Returns
        -------
        numpy.ndarray
            1D signal time course of length n_volumes.
        """
        n_volumes = dce_data.shape[3]

        if brain_mask is not None:
            brain_voxels = dce_data[brain_mask]  # (n_voxels, n_volumes)
            if brain_voxels.shape[0] > 100:
                return np.asarray(np.percentile(brain_voxels, 95, axis=0))
            # Small mask: mean is more stable
            return np.array([
                np.mean(dce_data[..., t][brain_mask])
                for t in range(n_volumes)
            ])

        # No mask: mean of non-zero voxels
        return np.array([
            np.mean(dce_data[..., t][dce_data[..., t] > 0])
            for t in range(n_volumes)
        ])

    def _auto_detect_baseline(
        self,
        dce_data: np.ndarray,
        brain_mask: Optional[np.ndarray],
    ) -> tuple[int, int, dict[str, Any]]:
        """Auto-detect baseline frames from brain signal time course.

        For DCE with gadolinium, signal INCREASES at bolus arrival due to
        T1-shortening effect. We detect the first significant increase
        using the 95th percentile signal, which captures arterial timing
        without the transit delay that makes the mean lag by 0-1 frames.

        Parameters
        ----------
        dce_data : numpy.ndarray
            4D DCE data array (x, y, z, t)
        brain_mask : numpy.ndarray or None
            Optional brain mask

        Returns
        -------
        tuple
            Tuple of (baseline_frames, bolus_arrival_frame, diagnostics).
            ``diagnostics`` is the per-call diag dict from
            ``detect_baseline_ensemble`` (method, confidence, ...) so
            silent low-confidence fallbacks can be surfaced in QC.
        """
        from pmriprep.utils.baseline_detection import detect_baseline_ensemble
        from pmriprep.utils.dce_constants import DEFAULT_BASELINE_FRAMES

        signal = self._brain_signal_timecourse(dce_data, brain_mask)
        volume_tr = self.inputs.volume_tr if self.inputs.volume_tr > 0 else 7.0
        bolus_frame, diag = detect_baseline_ensemble(
            signal, tr=volume_tr, min_baseline=DEFAULT_BASELINE_FRAMES,
        )
        baseline_frames = bolus_frame

        return baseline_frames, bolus_frame, diag

    def _detect_bolus_arrival(
        self,
        dce_data: np.ndarray,
        brain_mask: Optional[np.ndarray],
    ) -> tuple[int, dict[str, Any]]:
        """Detect bolus arrival frame from data.

        Parameters
        ----------
        dce_data : numpy.ndarray
            4D DCE data array
        brain_mask : numpy.ndarray or None
            Optional brain mask

        Returns
        -------
        tuple
            Tuple of (bolus_arrival_frame, diagnostics) — see
            ``_auto_detect_baseline`` for the diag contract.
        """
        from pmriprep.utils.baseline_detection import detect_baseline_ensemble
        from pmriprep.utils.dce_constants import DEFAULT_BASELINE_FRAMES

        signal = self._brain_signal_timecourse(dce_data, brain_mask)
        volume_tr = self.inputs.volume_tr if self.inputs.volume_tr > 0 else 7.0
        bolus_frame, diag = detect_baseline_ensemble(
            signal, tr=volume_tr, min_baseline=DEFAULT_BASELINE_FRAMES,
        )
        return bolus_frame, diag

    def _calculate_fd(
        self,
        motion_params: npt.NDArray[Any],
        head_radius: float = 50.0,
    ) -> npt.NDArray[Any]:
        """Calculate framewise displacement from motion parameters.

        Parameters
        ----------
        motion_params : numpy.ndarray
            Motion parameters (N, 6)
        head_radius : float
            Head radius in mm

        Returns
        -------
        numpy.ndarray
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

        from pmriprep.utils.nifti import nifti_gz_name

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_moco"))

    def _gen_params_filename(self) -> str:
        """Generate motion parameters filename."""
        if self.inputs.out_params:
            return str(self.inputs.out_params)

        from pmriprep.utils.nifti import nifti_gz_name

        base = nifti_gz_name(self.inputs.in_file, "_moco")[:-7]
        return str(Path.cwd() / f"{base}.par")

    def _gen_mean_filename(self) -> str:
        """Generate mean image filename."""
        from pmriprep.utils.nifti import nifti_gz_name

        return str(Path.cwd() / nifti_gz_name(self.inputs.in_file, "_moco_mean_reg"))

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
    use_gpu = traits.Bool(
        False,
        usedefault=True,
        desc="Whether this node uses GPU (for nipype scheduling)",
    )
    sloppy = traits.Bool(
        False,
        usedefault=True,
        desc="Disable HD-BET test-time augmentation for fast testing",
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

    Runs HD-BET on the DCE reference image.

    Parameters
    ----------
    dce_mean : str
        Mean DCE image (from DCEPreprocess.mean_img)
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

    """

    input_spec = DCEBrainExtractInputSpec
    output_spec = DCEBrainExtractOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE brain extraction."""
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
        if method_used == "hdbet":
            from pmriprep.utils.hdbet import resolve_hdbet_execution

            device, use_tta = resolve_hdbet_execution(
                self.inputs.hdbet_device,
                self.inputs.sloppy,
            )
            qc_metrics.update({"hdbet_device": device, "hdbet_tta_used": use_tta})

        # Set outputs
        self._results["out_brain"] = out_brain
        self._results["out_mask"] = out_mask
        self._results["method_used"] = method_used
        self._results["qc_metrics"] = qc_metrics

        return runtime

    def _run_hdbet(self, dce_mean: Path, out_brain: str, out_mask: str) -> bool:
        """Run HD-BET brain extraction via shared utility."""
        import shutil

        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        from pmriprep.utils.hdbet import run_hdbet

        conda_env = (
            self.inputs.hdbet_conda_env
            if isdefined(self.inputs.hdbet_conda_env)
            else ""
        )

        succeeded, mask_path = run_hdbet(
            in_file=str(dce_mean),
            out_brain=out_brain,
            device=self.inputs.hdbet_device,
            conda_env=conda_env,
            sloppy=self.inputs.sloppy,
        )
        if succeeded and mask_path:
            if str(mask_path) != out_mask:
                shutil.copy2(mask_path, out_mask)
        return succeeded

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

# Cross-modal T1w↔DCE registration QC thresholds. Dice (brain-mask
# overlap) and Pearson correlation drive the qc_status gate.
# Cross-modal Pearson is intrinsically low (different intensity
# distributions in T1w vs post-contrast DCE), so correlation_min
# is set near the noise floor to discriminate failed alignment
# from normal cross-modal contrast.
# NMI is computed for the metric record but not gated — the
# Strehl-Ghosh formula 2*MI/(H1+H2) is bounded above by 1 and
# its absolute scale is sensitive to histogram binning, so an
# absolute threshold across sites is unreliable.
REG_QC_THRESHOLDS: dict[str, float] = {
    "dice_min": 0.80,
    "dice_good": 0.85,
    "correlation_min": 0.03,
    "correlation_good": 0.10,
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
    native_brain_mask = File(
        exists=True,
        desc=(
            "Independently extracted native DCE brain mask for descriptive "
            "registration QC (optional)"
        ),
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
    out_h5 = File(exists=True, desc="Affine transform in ITK composite (.h5) format")
    dice = traits.Float(desc="DICE coefficient (if brain mask provided)")
    correlation = traits.Float(desc="Correlation coefficient (if brain mask provided)")
    nmi = traits.Float(desc="Normalized mutual information")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class DCERegistration(SimpleInterface):
    """DCE to T1w structural registration for perfusion map overlays.

    Uses FSL FLIRT to register a T1-weighted structural image to DCE space.
    DCE uses SPGR/FLASH (gradient echo), which has good geometric fidelity
    and similar contrast to structural T1w, so standard FLIRT with normmi
    cost function is appropriate.

    Three registration candidates are evaluated:
      Header: Header/world alignment without optimization
      Path A: Header pre-aligned T1w -> FLIRT
      Path B: Original T1w -> FLIRT
      Candidate selected by foreground-overlap DICE.

    Parameters
    ----------
    t1w_file : str
        T1-weighted image to register to DCE space.
    dce_mean : str
        Mean DCE image (target space).
    brain_mask : str, optional
        Brain mask for computing QC metrics.
    dof : int, optional
        Degrees of freedom (default: 6 for rigid body).
    cost : {'normmi', 'normcorr', 'corratio', 'mutualinfo', 'leastsq'}, optional
        Cost function for FLIRT (default: normmi).
    reference_type : {'postcon', 'precon', 'any'}, optional
        Expected T1w type: postcon (default for DCE), precon, or any.

    Returns
    -------
    out_file : str
        Registered T1w in DCE space.
    out_matrix : str
        FSL FLIRT transformation matrix.
    dice : float
        DICE coefficient of brain masks (if brain_mask provided).
    correlation : float
        Correlation coefficient within brain mask.
    nmi : float
        Normalized mutual information.
    qc_metrics : dict
        Complete QC metrics dictionary.

    Examples
    --------
    Standard usage:

    >>> reg = DCERegistration()
    >>> reg.inputs.t1w_file = 'T1w_brain.nii.gz'
    >>> reg.inputs.dce_mean = 'dce_mean.nii.gz'
    >>> result = reg.run()  # doctest: +SKIP

    Notes
    -----
    For DCE, the default reference is post-contrast T1w since DCE has contrast
    enhancement similar to post-contrast T1w, providing better intensity
    correspondence for registration.

    QC metrics computed:

    - DICE: Overlap of brain masks (requires brain_mask input)
    - Correlation: Pearson correlation within brain mask
    - NMI: Normalized mutual information (intensity correspondence)

    Requires FSL to be installed and available in PATH.

    """

    input_spec = DCERegistrationInputSpec
    output_spec = DCERegistrationOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute T1-weighted magnitude registration to DCE space.

        Implements three-candidate registration with DICE selection:

        - Header: Header/world alignment without optimization
        - Path A: Header pre-aligned T1w → FLIRT
        - Path B: Original T1w → FLIRT
        - Retain header alignment unless a fitted candidate improves relative
          foreground overlap by the configured DICE margin

        Parameters
        ----------
        runtime : Any
            Nipype runtime object.

        Returns
        -------
        Any
            Updated runtime object.

        Raises
        ------
        RuntimeError
            If both registration paths fail or produce no output.
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

            from pmriprep.utils.registration import reframe_fsl_transform

            # Setup output paths for dual-path registration
            header_aligned_file = str(temp_path / f"t1w_header_aligned{ext}")
            header_matrix = str(temp_path / "t1w_header.mat")
            out_file_a = str(temp_path / f"t1w_reg_pathA{ext}")
            out_matrix_a_header = str(temp_path / "t1w_reg_pathA_header.mat")
            out_matrix_a = str(temp_path / "t1w_reg_pathA.mat")
            out_file_b = str(temp_path / f"t1w_reg_pathB{ext}")
            out_matrix_b = str(temp_path / "t1w_reg_pathB.mat")
            try:
                self._run_header_alignment(
                    t1w_file,
                    dce_mean,
                    header_aligned_file,
                    header_matrix,
                )
            except Exception as exc:
                logger.warning(
                    "Header-based registration candidate failed; "
                    "continuing with direct FLIRT path: %s",
                    exc,
                )

            # Run dual-path FLIRT registration
            try:
                self._run_flirt_dual_path(
                    header_aligned_file=header_aligned_file,
                    t1w_file=t1w_file,
                    dce_mean=dce_mean,
                    out_file_a=out_file_a,
                    out_matrix_a=out_matrix_a_header,
                    out_file_b=out_file_b,
                    out_matrix_b=out_matrix_b,
                    dof=dof,
                    cost=cost,
                    search_range=search_range,
                )
            except RuntimeError:
                if not (
                    Path(header_aligned_file).is_file()
                    and Path(header_matrix).is_file()
                ):
                    raise
                logger.warning(
                    "Both optimized registration paths failed; evaluating "
                    "the header-based candidate"
                )
            if (
                Path(out_file_a).is_file()
                and Path(out_matrix_a_header).is_file()
            ):
                reframe_fsl_transform(
                    out_matrix_a_header,
                    header_aligned_file,
                    t1w_file,
                    dce_mean,
                    out_matrix_a,
                )

            # Select best path based on DICE
            (
                selected_out_file,
                selected_matrix,
                dice_header,
                dice_a,
                dice_b,
                selected_path,
                selected_dice,
            ) = self._select_best_path(
                header_aligned_file=header_aligned_file,
                header_matrix=header_matrix,
                out_file_a=out_file_a,
                out_matrix_a=out_matrix_a,
                out_file_b=out_file_b,
                out_matrix_b=out_matrix_b,
                dce_mean=dce_mean,
            )
            transform_offset = self._transform_offset_from_header(
                selected_matrix=selected_matrix,
                header_matrix=header_matrix,
                moving_file=t1w_file,
                reference_file=dce_mean,
            )

            logger.info(
                "Registration path selection: header=%.3f, A=%.3f, B=%.3f "
                "→ selected %s",
                dice_header,
                dice_a,
                dice_b,
                selected_path,
            )

            # Copy selected outputs to final locations
            shutil.copy(selected_out_file, out_file)
            shutil.copy(selected_matrix, out_matrix)

        # Compute QC metrics (outside temp context so outputs exist)
        qc_metrics = self._compute_qc_metrics(
            out_file,
            dce_mean,
            foreground_dice=selected_dice,
        )
        qc_metrics["path_selection"] = {
            "dice_header": dice_header,
            "dice_a": dice_a,
            "dice_b": dice_b,
            "selected": selected_path,
            "selected_dice": selected_dice,
            "selection_metric": "relative_foreground_overlap",
            **transform_offset,
        }

        # Set outputs
        from pmriprep.utils.transforms import fsl_to_x5

        self._results["out_file"] = out_file
        self._results["out_matrix"] = out_matrix
        self._results["out_h5"] = fsl_to_x5(
            out_matrix,
            reference=str(self.inputs.dce_mean),
            moving=str(self.inputs.t1w_file),
        )
        self._results["qc_metrics"] = qc_metrics

        if "dice" in qc_metrics:
            self._results["dice"] = qc_metrics["dice"]
        if "correlation" in qc_metrics:
            self._results["correlation"] = qc_metrics["correlation"]
        if "nmi" in qc_metrics:
            self._results["nmi"] = qc_metrics["nmi"]

        return runtime

    @staticmethod
    def _run_header_alignment(
        moving_file: str,
        reference_file: str,
        out_file: str,
        out_matrix: str,
    ) -> None:
        """Apply NIfTI header geometry and emit a matching FSL matrix."""
        header = FLIRT()
        header.inputs.output_type = "NIFTI_GZ"
        header.inputs.in_file = moving_file
        header.inputs.reference = reference_file
        header.inputs.out_file = out_file
        header.inputs.out_matrix_file = out_matrix
        header.inputs.apply_xfm = True
        header.inputs.uses_qform = True
        header.inputs.interp = "trilinear"
        header.run()

    @staticmethod
    def _transform_offset_from_header(
        *,
        selected_matrix: str,
        header_matrix: str,
        moving_file: str,
        reference_file: str,
    ) -> dict[str, float]:
        """Measure the selected registration relative to header alignment.

        Parameters
        ----------
        selected_matrix : str
            Selected FSL transform from the original moving grid to the reference.
        header_matrix : str
            FSL transform representing NIfTI header/world alignment.
        moving_file : str
            Original moving magnitude image.
        reference_file : str
            Fixed reference image.

        Returns
        -------
        dict[str, float]
            Translation of the relative transform at the world origin, rigid
            rotation angle, and RMS corner displacement across the moving FOV.

        Notes
        -----
        These values describe the magnitude of the correction selected over
        header alignment. They are reported for QC and are not thresholded.
        """
        from nitransforms.linear import Affine

        selected = np.asarray(
            Affine.from_filename(
                selected_matrix,
                fmt="fsl",
                reference=reference_file,
                moving=moving_file,
            ).matrix,
            dtype=np.float64,
        )
        header = np.asarray(
            Affine.from_filename(
                header_matrix,
                fmt="fsl",
                reference=reference_file,
                moving=moving_file,
            ).matrix,
            dtype=np.float64,
        )
        relative = selected @ np.linalg.inv(header)
        translation_mm = float(np.linalg.norm(relative[:3, 3]))

        u, _, vh = np.linalg.svd(relative[:3, :3])
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        rotation_argument = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_degrees = float(np.degrees(np.arccos(rotation_argument)))

        moving_img: Any = nib.load(moving_file)
        corners_ijk = np.array(
            [
                [i, j, k, 1.0]
                for i in (0.0, float(moving_img.shape[0] - 1))
                for j in (0.0, float(moving_img.shape[1] - 1))
                for k in (0.0, float(moving_img.shape[2] - 1))
            ],
            dtype=np.float64,
        )
        corners_ras = (moving_img.affine @ corners_ijk.T).T
        selected_ras = (selected @ corners_ras.T).T[:, :3]
        header_ras = (header @ corners_ras.T).T[:, :3]
        rms_displacement_mm = float(
            np.sqrt(np.mean(np.sum((selected_ras - header_ras) ** 2, axis=1)))
        )
        return {
            "selected_vs_header_origin_translation_mm": translation_mm,
            "selected_vs_header_rotation_degrees": rotation_degrees,
            "selected_vs_header_rms_displacement_mm": rms_displacement_mm,
        }

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
        """Run dual-path FLIRT registration.

        Path A uses header-aligned T1w, Path B uses original T1w.
        At least one path must succeed.

        Parameters
        ----------
        header_aligned_file : str
            Header-aligned T1w image.
        t1w_file : str
            Original T1w image.
        dce_mean : str
            DCE mean reference image.
        out_file_a : str
            Output file for path A.
        out_matrix_a : str
            Output matrix for path A.
        out_file_b : str
            Output file for path B.
        out_matrix_b : str
            Output matrix for path B.
        dof : int
            Degrees of freedom.
        cost : str
            Cost function name.
        search_range : int
            Search range in degrees.

        Raises
        ------
        RuntimeError
            If both registration paths fail.
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
        header_aligned_file: str,
        header_matrix: str,
        out_file_a: str,
        out_matrix_a: str,
        out_file_b: str,
        out_matrix_b: str,
        dce_mean: str,
    ) -> tuple[str, str, float, float, float, str, float]:
        """Select best registration path based on DICE overlap.

        Path B must exceed Path A by ``DICE_SELECTION_MARGIN`` to be the
        preferred fitted candidate. When header/world alignment is available,
        a fitted candidate replaces it only when overlap improves by the same
        margin.

        Parameters
        ----------
        header_aligned_file : str
            T1w image resampled to the DCE grid using image headers.
        header_matrix : str
            Sqform-derived FSL matrix representing world/header alignment in
            the original moving-image frame.
        out_file_a : str
            Output file from path A.
        out_matrix_a : str
            Output matrix from path A.
        out_file_b : str
            Output file from path B.
        out_matrix_b : str
            Output matrix from path B.
        dce_mean : str
            DCE mean reference for DICE computation.

        Returns
        -------
        tuple
            Selected file and matrix, header/A/B overlap Dice, path name, and
            the overlap Dice belonging to the selected candidate.

        Raises
        ------
        RuntimeError
            If neither path produced valid output.
        """
        header_exists = Path(header_aligned_file).is_file() and Path(
            header_matrix
        ).is_file()
        file_a_exists = Path(out_file_a).is_file() and Path(
            out_matrix_a
        ).is_file()
        file_b_exists = Path(out_file_b).is_file() and Path(
            out_matrix_b
        ).is_file()

        if not header_exists and not file_a_exists and not file_b_exists:
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
        dice_header = (
            compute_dice_for_selection(header_aligned_file, dce_mean)
            if header_exists
            else 0.0
        )

        # Select the fitted path first, retaining the header-initialized path
        # unless the original-grid fit improves overlap by a clear margin.
        if file_b_exists and (
            not file_a_exists or dice_b > dice_a + DICE_SELECTION_MARGIN
        ):
            selected_file = out_file_b
            selected_matrix = out_matrix_b
            selected_dice = dice_b
            selected_path = "B"
        elif file_a_exists:
            selected_file = out_file_a
            selected_matrix = out_matrix_a
            selected_dice = dice_a
            selected_path = "A"
        elif not header_exists:
            raise RuntimeError(
                "FLIRT registration produced no valid output files. "
                "Check FSL installation and input images."
            )
        else:
            selected_file = header_aligned_file
            selected_matrix = header_matrix
            selected_dice = dice_header
            selected_path = "header"

        # Header/world alignment is the non-optimized baseline. Avoid applying
        # a fitted transform unless it materially improves relative overlap.
        if header_exists and not (
            selected_dice > dice_header + DICE_SELECTION_MARGIN
        ):
            selected_file = header_aligned_file
            selected_matrix = header_matrix
            selected_dice = dice_header
            selected_path = "header"

        return (
            selected_file,
            selected_matrix,
            dice_header,
            dice_a,
            dice_b,
            selected_path,
            selected_dice,
        )

    def _compute_qc_metrics(
        self,
        registered_t1w: str,
        dce_mean: str,
        *,
        foreground_dice: float | None = None,
    ) -> dict[str, Any]:
        """Compute registration QC metrics with validation thresholds.

        Parameters
        ----------
        registered_t1w : str
            Path to registered T1w image.
        dce_mean : str
            Path to reference DCE mean image.
        foreground_dice : float or None, optional
            Foreground-overlap DICE for the selected registration candidate.
            Used only when no explicit source-space brain mask is available.

        Returns
        -------
        dict
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
                transform_mask_to_grid,
            )

            # Get optional brain mask using isdefined
            mask_path: Optional[str] = None
            if isdefined(self.inputs.brain_mask) and Path(
                self.inputs.brain_mask
            ).exists():
                mask_path = str(self.inputs.brain_mask)
            dice_source = "brain_mask" if mask_path is not None else "foreground_overlap"
            qc_metrics["dice_source"] = dice_source

            # Resample the input mask onto the DCE grid using the just-
            # computed FLIRT matrix. Without this step the mask remains
            # in T1w/anat space while registered_t1w and dce_mean live on
            # the DCE grid, causing every QC helper to raise on shape
            # mismatch and the resulting metrics to be silently dropped.
            mask_path_dce: Optional[str] = None
            if mask_path is not None:
                resampled = transform_mask_to_grid(
                    mask_path, dce_mean, self._gen_matfilename()
                )
                mask_path_dce = resampled or None

            # NMI is recorded for reference. Not gated: cross-modal
            # Strehl-Ghosh NMI is bounded above by 1 and its absolute
            # scale is histogram-bin-sensitive, so an absolute threshold
            # across sites would be unreliable.
            try:
                nmi = calculate_nmi_from_files(
                    registered_t1w, dce_mean, mask_path=mask_path_dce
                )
                qc_metrics["nmi"] = nmi
            except Exception as exc:
                logger.warning("NMI calculation failed: %s", exc)

            # Calculate correlation within brain mask
            try:
                corr = calculate_correlation_from_files(
                    registered_t1w, dce_mean, mask_path=mask_path_dce
                )
                qc_metrics["correlation"] = corr
                if corr < REG_QC_THRESHOLDS["correlation_min"]:
                    warnings_list.append(
                        f"Correlation {corr:.2f} below threshold "
                        f"{REG_QC_THRESHOLDS['correlation_min']}"
                    )
            except Exception as exc:
                logger.warning("Correlation calculation failed: %s", exc)

            # Calculate DICE if brain mask provided
            if mask_path_dce:
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

                        # Load the DCE-grid brain mask
                        mask_img = nib.load(mask_path_dce)
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
                    logger.warning("DICE calculation failed: %s", exc)

            if (
                mask_path is None
                and foreground_dice is not None
                and np.isfinite(foreground_dice)
            ):
                qc_metrics["dice"] = float(foreground_dice)
                if foreground_dice < REG_QC_THRESHOLDS["dice_min"]:
                    warnings_list.append(
                        f"Foreground overlap {foreground_dice:.2f} below threshold "
                        f"{REG_QC_THRESHOLDS['dice_min']}"
                    )

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
                    logger.warning("Determinant check failed: %s", exc)

            # Add registration metadata
            qc_metrics["dof"] = self.inputs.dof
            qc_metrics["cost_function"] = self.inputs.cost
            qc_metrics["reference_type"] = self.inputs.reference_type
            qc_metrics["brain_mask_used"] = mask_path_dce is not None

            native_mask_path = (
                str(self.inputs.native_brain_mask)
                if isdefined(self.inputs.native_brain_mask)
                and Path(self.inputs.native_brain_mask).exists()
                else ""
            )
            qc_metrics["native_perfusion_mask_metrics_role"] = (
                "descriptive_not_gated"
            )
            qc_metrics["native_perfusion_mask_used"] = bool(native_mask_path)
            if native_mask_path:
                try:
                    qc_metrics["native_perfusion_mask_correlation"] = (
                        calculate_correlation_from_files(
                            registered_t1w,
                            dce_mean,
                            mask_path=native_mask_path,
                        )
                    )
                    qc_metrics["native_perfusion_mask_nmi"] = (
                        calculate_nmi_from_files(
                            registered_t1w,
                            dce_mean,
                            mask_path=native_mask_path,
                        )
                    )
                    if mask_path_dce:
                        transformed_mask = nib.load(mask_path_dce).get_fdata()  # type: ignore[attr-defined]
                        native_mask = nib.load(native_mask_path).get_fdata()  # type: ignore[attr-defined]
                        qc_metrics["native_perfusion_mask_dice"] = calculate_dice(
                            transformed_mask, native_mask, threshold=0.5
                        )
                except Exception as exc:
                    logger.warning(
                        "Could not compute native-mask DCE registration metrics: %s",
                        exc,
                    )
                    qc_metrics["native_perfusion_mask_metrics_failed"] = True

            # Determine overall QC status
            qc_metrics["warnings"] = warnings_list
            qc_metrics["qc_status_basis"] = (
                "final_dce_space_brain_mask_dice_and_correlation"
                if dice_source == "brain_mask"
                else "final_dce_space_foreground_overlap_and_correlation"
            )
            qc_metrics["qc_status_thresholds"] = {
                "dice_min": REG_QC_THRESHOLDS["dice_min"],
                "correlation_min": REG_QC_THRESHOLDS["correlation_min"],
            }
            missing_required = [
                metric for metric in ("dice", "correlation")
                if not isinstance(qc_metrics.get(metric), (int, float))
                or not np.isfinite(qc_metrics[metric])
            ]
            if missing_required:
                qc_metrics["missing_required_metrics"] = missing_required
                qc_metrics["qc_status"] = "UNKNOWN"
            elif (
                qc_metrics["dice"] < REG_QC_THRESHOLDS["dice_min"]
                or qc_metrics["correlation"]
                < REG_QC_THRESHOLDS["correlation_min"]
            ):
                qc_metrics["qc_status"] = "POOR"
            elif warnings_list:
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

        Parameters
        ----------
        mat_file : str
            Path to FLIRT .mat file.

        Returns
        -------
        float
            Determinant of the 3x3 rotation/scale submatrix.
        """
        mat = np.loadtxt(mat_file)
        det = float(np.linalg.det(mat[:3, :3]))
        return det

    def _gen_outfilename(self) -> str:
        """Generate output filename if not specified.

        Returns
        -------
        str
            Path to output registered T1w file.
        """
        if isdefined(self.inputs.out_file):
            return str(self.inputs.out_file)

        from pmriprep.utils.nifti import nifti_gz_name

        return str(Path.cwd() / nifti_gz_name(self.inputs.t1w_file, "_dce"))

    def _gen_matfilename(self) -> str:
        """Generate output matrix filename if not specified.

        Returns
        -------
        str
            Path to output transformation matrix file.
        """
        if isdefined(self.inputs.out_matrix):
            return str(self.inputs.out_matrix)

        from pmriprep.utils.nifti import nifti_gz_name

        base = nifti_gz_name(self.inputs.t1w_file, "_dce")[:-7]
        return str(Path.cwd() / f"{base}.mat")
