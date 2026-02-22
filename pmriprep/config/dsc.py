"""DSC (Dynamic Susceptibility Contrast) configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class DSCConfig:
    """Configuration for DSC perfusion processing.

    This dataclass contains all parameters for DSC perfusion processing,
    including preload detection, deconvolution, AIF detection, registration,
    quantification, and processing options.

    Attributes
    ----------
    preload_present : bool
        Whether contrast preload is present (e.g., prior DCE injection).
    dose : Literal["half", "full", "double"]
        Contrast agent dose level.
    method : str
        Deconvolution method: sSVD, cSVD, oSVD, tikhonov, vascular, or cpi.
    bayesian : str
        Bayesian inference method: none, map, vi, or mcmc.
    mcmc_samples : int
        Number of MCMC samples for Bayesian inference.
    cpi_control_points : int
        Number of control points for CPI model (0 = auto-select).
    aif_mask_file : Optional[Path]
        Path to custom AIF mask file.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
    aif_criteria : Literal["strict", "relaxed"]
        AIF quality criteria threshold.
    n_aif_candidates : int
        Number of AIF candidate voxels to average.
    gamma_fit : bool
        Apply gamma-variate fitting to AIF.
    disable_aif_fallback : bool
        Disable synthetic AIF fallback when detection fails.
    lesion_mask : Optional[Path]
        Path to lesion mask for registration weighting, AIF exclusion,
        and lesion statistics.
    nawm_mask : Optional[Path]
        Path to NAWM mask for CBV normalization.
    roi_mask : Optional[Path]
        Path to ROI mask for stats extraction only (no processing impact).
    registration : Literal["auto", "flirt", "bbr"]
        Registration method: auto (try both, pick best), flirt, or bbr.
    smooth_fwhm : float
        Smoothing kernel FWHM (mm) for output maps.
    underlays : List[str]
        List of underlay images for QC visualizations.
    motion_thresh : float
        Framewise displacement threshold (mm) for motion outliers.
    pre_bolus : int
        Number of pre-bolus baseline frames.
    num_processes : int
        Number of parallel processes (0 = use system default).
    use_gpu : bool
        Enable GPU acceleration.
    hdbet_device : Literal["auto", "cuda", "mps", "cpu"]
        Device for HD-BET brain extraction: auto, cuda, mps, or cpu.
    aif_method : Literal["auto", "population"]
        AIF detection method: auto or population.
    """

    # Preload settings
    preload_present: bool = False
    dose: Literal["half", "full", "double"] = "full"

    # Deconvolution settings
    method: str = "oSVD"
    bayesian: str = "none"
    mcmc_samples: int = 500
    cpi_control_points: int = 0  # 0 = auto-select

    # AIF detection settings
    aif_mask_file: Optional[Path] = None
    lesion_distance: float = 15.0
    aif_criteria: Literal["strict", "relaxed"] = "strict"
    n_aif_candidates: int = 10
    gamma_fit: bool = False
    disable_aif_fallback: bool = False
    aif_method: Literal["auto", "population"] = "auto"

    # Registration settings
    lesion_mask: Optional[Path] = None
    nawm_mask: Optional[Path] = None
    roi_mask: Optional[Path] = None
    registration: Literal["auto", "flirt", "bbr"] = "auto"

    # Processing settings
    smooth_fwhm: float = 0.0  # No smoothing by default
    underlays: List[str] = field(default_factory=list)
    motion_thresh: float = 0.5
    pre_bolus: int = 12

    # Resource settings
    num_processes: int = 0  # 0 = use system default
    use_gpu: bool = False
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

        # Coerce integer fields (these have non-None defaults)
        try:
            self.pre_bolus = int(self.pre_bolus)
            self.mcmc_samples = int(self.mcmc_samples)
            self.cpi_control_points = int(self.cpi_control_points)
            self.n_aif_candidates = int(self.n_aif_candidates)
            self.num_processes = int(self.num_processes)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DSCConfig integer field has invalid type: {e}") from e

        # Coerce float fields (these have non-None defaults)
        try:
            self.smooth_fwhm = float(self.smooth_fwhm)
            self.lesion_distance = float(self.lesion_distance)
            self.motion_thresh = float(self.motion_thresh)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DSCConfig float field has invalid type: {e}") from e

