"""DCE (Dynamic Contrast Enhanced) configuration dataclass."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from .constants import DOSE_PRESETS

logger = logging.getLogger(__name__)

# Alias for backwards compatibility within this module
_DOSE_PRESETS = DOSE_PRESETS


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
    scale_aif : bool
        Scale AIF amplitude to population reference.
    target_peak : Optional[float]
        Target AIF peak concentration (mM) for scaling, in blood-voxel units
        (pre-Hct-correction). The pipeline divides
        by (1-Hct) internally before comparing against the plasma-scale
        aif_curve. None = use field-strength default.
    aif_mask_file : Optional[Path]
        Path to custom AIF mask file.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
    n_aif_candidates : int
        Number of AIF candidate voxels to average.
    underlays : List[str]
        List of underlay images for QC visualizations.
    b1_method : Literal["auto", "afi", "tfl_b1map", "none"]
        B1+ correction method: auto (detect from data), afi (Yarnykh AFI),
        tfl_b1map (pre-computed flip angle map), none (skip B1+ correction).
    no_t1 : bool
        Skip T1 mapping (use field-strength-appropriate default T1).
    no_dsc_validation : bool
        Skip DSC-DCE cross-validation.
    smooth_fwhm : float
        Smoothing kernel FWHM (mm) for output maps.
    baseline_frames : int
        Number of baseline frames (0 = auto-detect).
    dose : float
        Contrast agent dose multiplier (0.5 = half, 1.0 = full, 2.0 = double).
    tsnr_threshold : float
        tSNR threshold for voxel inclusion (0 = no threshold).
    model : str
        Pharmacokinetic model: auto (per-voxel BIC selection),
        tofts, extended_tofts, patlak, or 2cxm.
    inference : str
        Inference engine: nlls (default, scipy least_squares),
        vi (recommended; variational Bayes), map (Laplace MAP),
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
    scale_aif: bool = True
    target_peak: Optional[float] = None  # None = use field-strength default
    aif_mask_file: Optional[Path] = None
    lesion_distance: float = 15.0
    n_aif_candidates: int = 10

    # Structural settings
    underlays: List[str] = field(default_factory=list)

    # T1 mapping settings
    b1_method: Literal["auto", "afi", "tfl_b1map", "none"] = "auto"
    no_t1: bool = False
    t1_correction_mode: Literal["auto", "none", "offset", "fallback"] = "auto"
    t1_method: Literal["hybrid", "wlls", "nlls"] = "hybrid"
    t1_map_file: Optional[Path] = None  # External T1 map (NIfTI, milliseconds)
    # Processing settings
    no_dsc_validation: bool = False
    smooth_fwhm: float = 0.0  # No smoothing by default
    baseline_frames: int = 0  # 0 = auto-detect
    dose: float = 1.0
    tsnr_threshold: float = 0.0  # 0 = no threshold

    # Model settings
    model: str = "auto"
    inference: str = "nlls"
    mcmc_samples: int = 500

    # Relaxivity override
    r1_relaxivity: Optional[float] = None  # User-supplied r1 (s^-1/mM)

    # DSC CBV constraint settings
    cbv_constraint_mode: str = "auto"  # "auto", "soft", "skip"
    hematocrit: float = 0.42  # Arterial Hct; Parker 2006 MRM used 0.42 in the original population-AIF derivation, so using Parker's fallback with 0.42 is internally self-consistent. Closer to GBM-cohort mean (~0.38) than QIBA Profile v2.0's adult-healthy-male 0.45.

    # Resource settings
    num_processes: int = 0  # 0 = use system default
    hdbet_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    hdbet_conda_env: Optional[str] = None

    # Testing / debug
    sloppy: bool = False  # Use fast, low-quality settings for testing

    def __post_init__(self):
        """Ensure numeric fields have correct types after initialization.

        This handles cases where values may be passed as strings from
        YAML/JSON config files or CLI arguments.
        """
        # Validate underlay values
        _VALID_UNDERLAYS = {"t1w_precon", "t1w_postcon", "flair", "t2w"}
        invalid = set(self.underlays) - _VALID_UNDERLAYS
        if invalid:
            raise ValueError(
                f"Invalid underlay(s): {sorted(invalid)}. "
                f"Valid: {sorted(_VALID_UNDERLAYS)}. "
                f"Note: 't1w' is no longer valid — use 't1w_precon' or 't1w_postcon'."
            )

        # Validate b1_method
        _VALID_B1_METHODS = {"auto", "afi", "tfl_b1map", "none"}
        if self.b1_method not in _VALID_B1_METHODS:
            raise ValueError(
                f"Invalid b1_method: {self.b1_method!r}. "
                f"Valid: {sorted(_VALID_B1_METHODS)}"
            )

        # Validate cbv_constraint_mode
        _VALID_CBV_CONSTRAINT_MODES = {"auto", "soft", "skip"}
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
            if self.dose in _DOSE_PRESETS:
                self.dose = _DOSE_PRESETS[self.dose]
            else:
                try:
                    self.dose = float(self.dose)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid dose={self.dose!r}. Use a named preset "
                        f"({', '.join(sorted(_DOSE_PRESETS))}) or a positive number."
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

        # Validate t1_map_file / no_t1 mutual exclusivity
        if self.t1_map_file and self.no_t1:
            raise ValueError(
                "Cannot set both t1_map_file and no_t1=True. "
                "Use t1_map_file to supply an external T1 map, or no_t1 to skip T1 mapping."
            )
        if (self.no_t1 or self.t1_map_file) and self.t1_method != "hybrid":
            logger.warning(
                "t1_method=%r is ignored when %s is set.",
                self.t1_method,
                "no_t1=True" if self.no_t1 else "t1_map_file",
            )
