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
    preload_detected : bool
        Whether contrast preload is detected/specified.
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
    prefer_contralateral : bool
        Prefer contralateral AIF candidates when lesion present.
    aif_criteria : Literal["strict", "relaxed"]
        AIF quality criteria threshold.
    n_aif_candidates : int
        Number of AIF candidate voxels to average.
    gamma_fit : bool
        Apply gamma-variate fitting to AIF.
    disable_aif_fallback : bool
        Disable synthetic AIF fallback when detection fails.
    lesion_mask : Optional[Path]
        Path to lesion mask for registration weighting.
    nawm_mask : Optional[Path]
        Path to NAWM mask for CBV normalization.
    use_flirt : bool
        Use FLIRT instead of BBR for registration.
    smooth_fwhm : float
        Smoothing kernel FWHM (mm) for output maps.
    underlays : List[str]
        List of underlay images for QC visualizations.
    motion_thresh : float
        Framewise displacement threshold (mm) for motion outliers.
    pre_bolus : int
        Number of pre-bolus baseline frames.
    save_k2_map : bool
        Deprecated, ignored. K2 map is always saved.
    num_processes : int
        Number of parallel processes (0 = use system default).
    use_gpu : bool
        Enable GPU acceleration.
    aif_method : Literal["auto", "population"]
        AIF detection method: auto or population.
    """

    # Preload settings
    preload_detected: bool = False
    dose: Literal["half", "full", "double"] = "full"

    # Deconvolution settings
    method: str = "sSVD"
    bayesian: str = "none"
    mcmc_samples: int = 500
    cpi_control_points: int = 0  # 0 = auto-select

    # AIF detection settings
    aif_mask_file: Optional[Path] = None
    lesion_distance: float = 15.0
    prefer_contralateral: bool = True
    aif_criteria: Literal["strict", "relaxed"] = "strict"
    n_aif_candidates: int = 10
    gamma_fit: bool = False
    disable_aif_fallback: bool = False
    aif_method: Literal["auto", "population"] = "auto"

    # Registration settings
    lesion_mask: Optional[Path] = None
    nawm_mask: Optional[Path] = None
    use_flirt: bool = True

    # Processing settings
    smooth_fwhm: float = 0.0  # No smoothing by default
    underlays: List[str] = field(default_factory=lambda: ["t1w", "flair"])
    motion_thresh: float = 0.5
    pre_bolus: int = 12
    save_k2_map: bool = True  # Deprecated: K2 map is always saved

    # Resource settings
    num_processes: int = 0  # 0 = use system default
    use_gpu: bool = False

    def __post_init__(self):
        """Ensure numeric fields have correct types after initialization.

        This handles cases where values may be passed as strings from
        YAML/JSON config files or CLI arguments.
        """
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

    def apply_gbm_preset(self) -> "DSCConfig":
        """Apply GBM clinical preset settings.

        Returns a new DSCConfig with GBM-optimized settings:
        - oSVD deconvolution method
        - relaxed AIF criteria
        - n_aif_candidates=5
        - gamma_fit=True (gamma-variate AIF fitting for cleaner deconvolution)
        - K2 map always saved

        Returns
        -------
        DSCConfig
            New config with GBM preset applied.
        """
        return DSCConfig(
            preload_detected=self.preload_detected,
            dose=self.dose,
            method="oSVD",
            bayesian=self.bayesian,
            mcmc_samples=self.mcmc_samples,
            cpi_control_points=self.cpi_control_points,
            aif_mask_file=self.aif_mask_file,
            lesion_distance=self.lesion_distance,
            prefer_contralateral=self.prefer_contralateral,
            aif_criteria="relaxed",
            n_aif_candidates=5,
            gamma_fit=True,
            disable_aif_fallback=self.disable_aif_fallback,
            aif_method=self.aif_method,
            lesion_mask=self.lesion_mask,
            nawm_mask=self.nawm_mask,
            use_flirt=self.use_flirt,
            smooth_fwhm=self.smooth_fwhm,
            underlays=self.underlays,
            motion_thresh=self.motion_thresh,
            pre_bolus=self.pre_bolus,
            save_k2_map=True,  # Always saved, kept for compat
            num_processes=self.num_processes,
            use_gpu=self.use_gpu,
        )
