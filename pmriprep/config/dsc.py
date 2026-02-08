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
    mni_atlas : str
        MNI atlas name for AIF detection.
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
    lesion_dilation : float
        Dilation radius (mm) for lesion mask in registration.
    nawm_mask : Optional[Path]
        Path to NAWM mask for CBV normalization.
    use_flirt : bool
        Use FLIRT instead of BBR for registration.
    allow_scaling : bool
        Allow intensity scaling in registration.
    r2_star : Optional[float]
        Relaxivity value for absolute quantification (None = relative).
    absolute_quant : bool
        Enable absolute quantification.
    smooth_fwhm : float
        Smoothing kernel FWHM (mm) for output maps.
    underlays : List[str]
        List of underlay images for QC visualizations.
    motion_thresh : float
        Framewise displacement threshold (mm) for motion outliers.
    pre_bolus : int
        Number of pre-bolus baseline frames.
    save_k2_map : bool
        Save K2 leakage correction map.
    save_uncertainty : bool
        Save uncertainty maps from Bayesian inference.
    k2_method : str
        K2 leakage correction method: "integral" or "exponential".
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
    mni_atlas: str = "MNI152NLin2009cAsym"
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
    lesion_dilation: float = 15.0
    nawm_mask: Optional[Path] = None
    use_flirt: bool = True
    allow_scaling: bool = True

    # Quantification settings
    r2_star: Optional[float] = None
    absolute_quant: bool = False

    # Processing settings
    smooth_fwhm: float = 0.0  # No smoothing by default
    underlays: List[str] = field(default_factory=lambda: ["t1w", "flair"])
    motion_thresh: float = 0.5
    pre_bolus: int = 12
    save_k2_map: bool = False
    save_uncertainty: bool = False

    # K2 leakage correction
    k2_method: str = "exponential"  # "integral" (physics-correct) or "exponential" (legacy perf-plus)

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
            self.lesion_dilation = float(self.lesion_dilation)
            self.motion_thresh = float(self.motion_thresh)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DSCConfig float field has invalid type: {e}") from e

        # Handle Optional[float] field
        if self.r2_star is not None:
            try:
                self.r2_star = float(self.r2_star)
            except (ValueError, TypeError) as e:
                raise TypeError(f"DSCConfig.r2_star has invalid type: {e}") from e

    def apply_gbm_preset(self) -> "DSCConfig":
        """Apply GBM clinical preset settings.

        Returns a new DSCConfig with GBM-optimized settings:
        - oSVD deconvolution method
        - pre_bolus=10 frames
        - relaxed AIF criteria
        - n_aif_candidates=5
        - gamma_fit=False (perf-plus GBM preset does not enable gamma-fit)
        - save_k2_map=True

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
            mni_atlas=self.mni_atlas,
            aif_mask_file=self.aif_mask_file,
            lesion_distance=self.lesion_distance,
            prefer_contralateral=self.prefer_contralateral,
            aif_criteria="relaxed",
            n_aif_candidates=5,
            gamma_fit=False,
            disable_aif_fallback=self.disable_aif_fallback,
            aif_method=self.aif_method,
            lesion_mask=self.lesion_mask,
            lesion_dilation=self.lesion_dilation,
            nawm_mask=self.nawm_mask,
            use_flirt=self.use_flirt,
            allow_scaling=self.allow_scaling,
            r2_star=self.r2_star,
            absolute_quant=self.absolute_quant,
            smooth_fwhm=self.smooth_fwhm,
            underlays=self.underlays,
            motion_thresh=self.motion_thresh,
            pre_bolus=10,
            save_k2_map=True,
            save_uncertainty=self.save_uncertainty,
            k2_method=self.k2_method,
            num_processes=self.num_processes,
            use_gpu=self.use_gpu,
        )
