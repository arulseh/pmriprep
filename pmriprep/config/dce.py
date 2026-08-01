"""DCE (Dynamic Contrast Enhanced) configuration dataclass."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from .constants import DOSE_PRESETS

logger = logging.getLogger(__name__)

@dataclass
class DCEConfig:
    """Configuration for DCE perfusion processing.

    This dataclass contains all parameters for DCE perfusion processing,
    including registration, AIF detection, structural imaging, T1 mapping,
    processing options, pharmacokinetic modeling, and resource settings.

    Attributes
    ----------
    lesion_mask : Optional[Path]
        Path to lesion mask in DCE native space for AIF exclusion and
        lesion statistics.
    nawm_mask : Optional[Path]
        Path to NAWM mask in DCE native space for Ktrans normalization.
    roi_mask : Optional[Path]
        Path to ROI mask in DCE native space for stats extraction only.
    aif_method : Literal["auto", "population"]
        AIF detection method: auto or population.
    aif_peak_floor : bool
        Raise a low detected AIF to the dose-adjusted expected lower peak.
        When false, preserve measured amplitude while retaining amplitude QC.
    target_peak : Optional[float]
        Target AIF peak concentration (mM) for scaling, in blood-voxel units
        (pre-Hct-correction). The pipeline divides
        by (1-Hct) internally before comparing against the plasma-scale
        aif_curve. When set, this explicit calibration overrides
        ``aif_peak_floor``. None uses the optional dose-adjusted lower floor.
    aif_mask_file : Optional[Path]
        Path to custom AIF mask file.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
    n_aif_candidates : int
        Maximum number of AIF candidate voxels to retain and average; fewer
        may survive selection.
    underlays : List[str]
        List of underlay images for QC visualizations.
    b1_method : Literal["auto", "afi", "precomputed", "none"]
        B1+ correction method: auto (detect from data), afi,
        precomputed (pre-computed B1+ map), none (skip B1+ correction).
    b1_map_encoding : Literal["auto", "actual_flip_deci_deg", "actual_flip_deg", "percent", "ratio"]
        Pixel encoding of a pre-computed B1+ map: auto (infer from BIDS
        suffix/Units, else pixel median), actual_flip_deci_deg (Siemens tfl,
        pixel = flip×10), actual_flip_deg, percent (BIDS p.u., 100=nominal),
        or ratio (1.0=nominal).
    no_t1 : bool
        Skip T1 mapping (use field-strength-appropriate default T1).
    no_dsc_validation : bool
        Skip DSC-DCE cross-validation.
    motion_correct : bool
        Apply motion correction. True (default) runs mcflirt and processes the
        realigned series. False assumes the input was already motion-corrected
        (e.g. on-scanner); mcflirt still runs to report residual motion for QC,
        but the uncorrected series is processed to avoid a second interpolation.
    motion_thresh : float
        Framewise-displacement threshold (mm) for flagging motion outliers in QC.
    smooth_fwhm : float
        Smoothing kernel FWHM (mm) for output maps.
    baseline_frames : int
        Number of baseline frames (0 = auto-detect).
    dose : float
        Contrast agent dose multiplier (0.5 = half, 1.0 = full, 2.0 = double).
    tsnr_threshold : float
        tSNR threshold for voxel inclusion (0 = no threshold).
    ser_early_time_s : float
        Early SER sample time after the selected global reference, in seconds.
    ser_late_time_s : float
        Late SER sample time after the selected global reference, in seconds.
    model : str
        Pharmacokinetic model: auto (per-voxel BIC selection),
        tofts, extended_tofts, patlak, or 2cxm.
    inference : str
        Inference engine: nlls (default, scipy least_squares),
        vi (variational Bayes), map (Laplace MAP),
        or mcmc (YAML-only — not exposed via CLI).
    mcmc_samples : int
        Number of MCMC samples for Bayesian inference.
    num_processes : int
        Number of parallel processes (0 = use system default).
    hdbet_device : str
        Device for HD-BET brain extraction: auto, cuda, mps, or cpu.
    """

    # User-supplied masks (DCE native space)
    lesion_mask: Optional[Path] = None
    nawm_mask: Optional[Path] = None
    roi_mask: Optional[Path] = None

    # AIF detection settings
    aif_method: Literal["auto", "population"] = "auto"
    aif_file: Optional[Path] = None  # External AIF curve file (bypasses detection)
    aif_peak_floor: bool = True
    target_peak: Optional[float] = None  # None = follow aif_peak_floor
    aif_mask_file: Optional[Path] = None
    lesion_distance: float = 15.0
    n_aif_candidates: int = 10
    # AIF->tissue bolus-arrival pre-alignment: "auto" shifts a measured AIF to
    # correct the arterial-to-tissue delay (guarded; benefits fine temporal
    # resolution), "off" applies no shift.
    aif_bat_alignment: Literal["auto", "off"] = "auto"

    # Structural settings
    underlays: List[str] = field(default_factory=list)

    # T1 mapping settings
    b1_method: Literal["auto", "afi", "precomputed", "none"] = "auto"
    b1_map_encoding: Literal[
        "auto", "actual_flip_deci_deg", "actual_flip_deg", "percent", "ratio"
    ] = "auto"
    no_t1: bool = False
    t1_method: Literal["nlls", "wlls"] = "nlls"
    t1_map_file: Optional[Path] = None  # External T1 map (NIfTI, milliseconds)

    # Processing settings
    no_dsc_validation: bool = False
    motion_correct: bool = True  # False = assume input already motion-corrected
    motion_thresh: float = 0.5  # Framewise-displacement outlier threshold (mm)
    smooth_fwhm: float = 0.0  # No smoothing by default
    baseline_frames: int = 0  # 0 = auto-detect
    dose: float = 1.0
    tsnr_threshold: float = 0.0  # 0 = no threshold
    ser_early_time_s: float = 120.0
    ser_late_time_s: float = 320.0

    # Model settings
    model: str = "auto"
    inference: str = "nlls"
    mcmc_samples: int = 500

    # Relaxivity override
    r1_relaxivity: Optional[float] = None  # User-supplied r1 (s^-1/mM)

    # DSC CBV constraint settings
    cbv_constraint_mode: str = "auto"  # "auto", "skip"
    hematocrit: float = 0.42  # Arterial Hct for blood-to-plasma conversion.

    # Resource settings
    num_processes: int = 0  # 0 = use system default
    hdbet_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    hdbet_conda_env: Optional[str] = None

    # Testing / debug
    sloppy: bool = False  # Use fast, low-quality settings for testing

    @property
    def uses_vfa_t1(self) -> bool:
        """Whether this configuration requires VFA-derived T1 mapping."""
        return not self.no_t1 and self.t1_map_file is None

    def __post_init__(self) -> None:
        """Ensure numeric fields have correct types after initialization.

        This handles cases where values may be passed as strings from
        YAML/JSON config files or CLI arguments.
        """
        categorical_options = {
            "model": ({"auto", "tofts", "extended_tofts", "patlak", "2cxm"}, self.model),
            "inference": ({"nlls", "map", "vi", "mcmc"}, self.inference),
            "aif_method": ({"auto", "population"}, self.aif_method),
            "aif_bat_alignment": ({"auto", "off"}, self.aif_bat_alignment),
            "t1_method": ({"nlls", "wlls"}, self.t1_method),
            "hdbet_device": ({"auto", "cuda", "mps", "cpu"}, self.hdbet_device),
        }
        for option, (valid, value) in categorical_options.items():
            if value not in valid:
                raise ValueError(
                    f"Invalid {option}={value!r}. Valid: {sorted(valid)}"
                )

        for option in (
            "aif_peak_floor",
            "no_t1",
            "no_dsc_validation",
            "motion_correct",
            "sloppy",
        ):
            if not isinstance(getattr(self, option), bool):
                raise TypeError(f"{option} must be a boolean")

        # Validate underlay values
        _VALID_UNDERLAYS = {
            "t1w_precon", "t1w_postcon", "t1w_subtraction",
            "flair_precon", "flair_postcon", "t2w",
        }
        invalid = set(self.underlays) - _VALID_UNDERLAYS
        if invalid:
            raise ValueError(
                f"Invalid underlay(s): {sorted(invalid)}. "
                f"Valid: {sorted(_VALID_UNDERLAYS)}. "
                f"Note: 't1w' and 'flair' are no longer valid — use their "
                f"'_precon'/'_postcon' variants."
            )

        # Validate b1_method
        _VALID_B1_METHODS = {"auto", "afi", "precomputed", "none"}
        if self.b1_method not in _VALID_B1_METHODS:
            raise ValueError(
                f"Invalid b1_method: {self.b1_method!r}. "
                f"Valid: {sorted(_VALID_B1_METHODS)}"
            )

        # Validate b1_map_encoding
        _VALID_B1_ENCODINGS = {
            "auto", "actual_flip_deci_deg", "actual_flip_deg", "percent", "ratio",
        }
        if self.b1_map_encoding not in _VALID_B1_ENCODINGS:
            raise ValueError(
                f"Invalid b1_map_encoding: {self.b1_map_encoding!r}. "
                f"Valid: {sorted(_VALID_B1_ENCODINGS)}"
            )

        # Validate cbv_constraint_mode
        _VALID_CBV_CONSTRAINT_MODES = {"auto", "skip"}
        if self.cbv_constraint_mode not in _VALID_CBV_CONSTRAINT_MODES:
            raise ValueError(
                f"Invalid cbv_constraint_mode: {self.cbv_constraint_mode!r}. "
                f"Valid: {sorted(_VALID_CBV_CONSTRAINT_MODES)}"
            )

        # Coerce integer fields (these have non-None defaults)
        try:
            self.baseline_frames = int(self.baseline_frames)
            self.mcmc_samples = int(self.mcmc_samples)
            self.n_aif_candidates = int(self.n_aif_candidates)
            self.num_processes = int(self.num_processes)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DCEConfig integer field has invalid type: {e}") from e

        # Coerce dose: string preset → float, numeric → float
        if isinstance(self.dose, str):
            if self.dose in DOSE_PRESETS:
                self.dose = DOSE_PRESETS[self.dose]
            else:
                try:
                    self.dose = float(self.dose)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid dose={self.dose!r}. Use a named preset "
                        f"({', '.join(sorted(DOSE_PRESETS))}) or a positive number."
                    ) from e
        else:
            self.dose = float(self.dose)
        import math
        if not math.isfinite(self.dose) or self.dose <= 0:
            raise ValueError(f"dose must be a finite positive number, got {self.dose}")
        if self.dose < 0.2 or self.dose > 3.0:
            logger.warning(
                "DCE dose=%.2f is outside typical range [0.2, 3.0]. "
                "Values <0.2 may have insufficient SNR; >3.0 risk signal saturation.",
                self.dose,
            )

        # Coerce float fields (these have non-None defaults)
        try:
            self.smooth_fwhm = float(self.smooth_fwhm)
            self.lesion_distance = float(self.lesion_distance)
            self.tsnr_threshold = float(self.tsnr_threshold)
            self.hematocrit = float(self.hematocrit)
            self.motion_thresh = float(self.motion_thresh)
            self.ser_early_time_s = float(self.ser_early_time_s)
            self.ser_late_time_s = float(self.ser_late_time_s)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DCEConfig float field has invalid type: {e}") from e

        # Handle Optional[float] fields
        if self.target_peak is not None:
            try:
                self.target_peak = float(self.target_peak)
            except (ValueError, TypeError) as e:
                raise TypeError(f"DCEConfig.target_peak has invalid type: {e}") from e

        if self.r1_relaxivity is not None:
            try:
                self.r1_relaxivity = float(self.r1_relaxivity)
            except (ValueError, TypeError) as e:
                raise TypeError(f"DCEConfig.r1_relaxivity has invalid type: {e}") from e
            if not math.isfinite(self.r1_relaxivity) or self.r1_relaxivity <= 0:
                raise ValueError(
                    "r1_relaxivity must be a finite positive value in s^-1/mM, "
                    f"got {self.r1_relaxivity}"
                )

        if not math.isfinite(self.hematocrit) or not 0 <= self.hematocrit < 1:
            raise ValueError(
                "hematocrit must be finite and in [0, 1), "
                f"got {self.hematocrit}"
            )

        if self.baseline_frames != 0 and self.baseline_frames < 5:
            raise ValueError("baseline_frames must be 0 (auto) or at least 5")
        if self.mcmc_samples <= 0:
            raise ValueError("mcmc_samples must be greater than zero")
        if self.n_aif_candidates < 3:
            raise ValueError("n_aif_candidates must be at least 3")
        if self.num_processes < 0:
            raise ValueError("num_processes must be non-negative (0 = automatic)")
        for option in (
            "smooth_fwhm",
            "lesion_distance",
            "tsnr_threshold",
            "motion_thresh",
            "ser_early_time_s",
            "ser_late_time_s",
        ):
            value = getattr(self, option)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{option} must be finite and non-negative")
        if self.ser_late_time_s <= self.ser_early_time_s:
            raise ValueError("ser_late_time_s must be greater than ser_early_time_s")
        if self.ser_early_time_s <= 0:
            raise ValueError("ser_early_time_s must be greater than zero")
        if self.target_peak is not None and (
            not math.isfinite(self.target_peak) or self.target_peak <= 0
        ):
            raise ValueError("target_peak must be finite and greater than zero")

        # Validate t1_map_file / no_t1 mutual exclusivity
        if self.t1_map_file and self.no_t1:
            raise ValueError(
                "Cannot set both t1_map_file and no_t1=True. "
                "Use t1_map_file to supply an external T1 map, or no_t1 to skip T1 mapping."
            )
        if (self.no_t1 or self.t1_map_file) and self.t1_method != "nlls":
            logger.warning(
                "t1_method=%r is ignored when %s is set.",
                self.t1_method,
                "no_t1=True" if self.no_t1 else "t1_map_file",
            )
