"""DSC (Dynamic Susceptibility Contrast) configuration dataclass."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from .constants import DOSE_PRESETS

logger = logging.getLogger(__name__)

# Alias for backwards compatibility within this module
_DOSE_PRESETS = DOSE_PRESETS


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
    dose : float
        Contrast agent dose multiplier (0.5 = half, 1.0 = full, 2.0 = double).
    agent_standard_dose : float
        Agent's recommended dose in mmol Gd/kg. Default 0.1 (standard).
        Use 0.05 for gadopiclenol, 0.04 for gadoquatrane.
    method : str
        Deconvolution method: sSVD, cSVD, oSVD, tikhonov, vascular, or cpi.
    bayesian : str
        Bayesian inference method: vi (default) or map.
    mcmc_samples : int
        Number of MCMC samples for Bayesian inference.
    cpi_control_points : int
        Number of control points for CPI model (0 = auto-select).
    aif_file : Optional[Path]
        Path to external AIF curve file (.txt, .npz, or .npy).
        Values must be DeltaR2* concentration in s^-1.
        Bypasses AIF detection when provided.
    aif_mask_file : Optional[Path]
        Path to custom AIF mask file.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
    n_aif_candidates : int
        Number of AIF candidate voxels to average.
    gamma_fit : str
        Gamma-variate fitting mode: "auto", "on", or "off".
    disable_aif_fallback : bool
        Disable synthetic AIF fallback when detection fails.
    lesion_mask : Optional[Path]
        Path to lesion mask for AIF exclusion and lesion statistics.
    nawm_mask : Optional[Path]
        Path to NAWM mask for CBV normalization.
    roi_mask : Optional[Path]
        Path to ROI mask for stats extraction only (no processing impact).
    registration : Literal["flirt", "flirt_bbr"]
        Registration method: 'flirt' (FLIRT only) or 'flirt_bbr' (FLIRT + BBR
        refinement, keeping whichever is best).
    smooth_fwhm : float
        Smoothing kernel FWHM (mm) for output maps.
    underlays : List[str]
        List of underlay images for QC visualizations.
    motion_thresh : float
        Framewise displacement threshold (mm) for motion outliers.
    pre_bolus : int
        Number of pre-bolus baseline frames for S0 estimation.
        AIF detection auto-detects actual bolus arrival independently.
    num_processes : int
        Number of parallel processes (0 = use system default).
    hdbet_device : Literal["auto", "cuda", "mps", "cpu"]
        Device for HD-BET brain extraction: auto, cuda, mps, or cpu.
    aif_method : Literal["auto", "synthetic"]
        AIF detection method: auto or synthetic.
    leakage_method : Literal["bsw", "bidirectional"]
        K2 leakage correction method. "bsw" = standard Boxerman-Schmainda-
        Weisskoff (cumulative integral, no washout). "bidirectional" = Leu 2016
        (accounts for EES washout, ~16% lower rCBV in HGG).
    kep : float, optional
        EES efflux rate constant in s⁻¹ for bidirectional correction.
        Only used when leakage_method="bidirectional". Default: 0.005 s⁻¹.
        Note: DSC uses s⁻¹ (like K2), not min⁻¹ (like DCE kep).
    """

    # Preload settings
    preload_present: bool = False
    dose: float = 1.0
    agent_standard_dose: float = 0.1  # mmol Gd/kg (0.1=standard, 0.05=gadopiclenol, 0.04=gadoquatrane)

    # Deconvolution settings
    method: str = "oSVD"
    bayesian: str = "vi"
    mcmc_samples: int = 500
    cpi_control_points: int = 0  # 0 = auto-select

    # AIF detection settings
    aif_file: Optional[Path] = None
    aif_mask_file: Optional[Path] = None
    lesion_distance: float = 15.0
    n_aif_candidates: int = 10
    gamma_fit: str = "auto"
    disable_aif_fallback: bool = False
    aif_method: Literal["auto", "synthetic"] = "auto"

    # Registration settings
    lesion_mask: Optional[Path] = None
    nawm_mask: Optional[Path] = None
    roi_mask: Optional[Path] = None
    registration: Literal["flirt", "flirt_bbr"] = "flirt"

    # Processing settings
    smooth_fwhm: float = 0.0  # No smoothing by default
    underlays: List[str] = field(default_factory=list)
    motion_thresh: float = 0.5
    pre_bolus: int = 5

    # Resource settings
    num_processes: int = 0  # 0 = use system default
    hdbet_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    hdbet_conda_env: Optional[str] = None

    # Leakage correction settings
    leakage_method: Literal["bsw", "bidirectional"] = "bsw"
    kep: Optional[float] = None  # s⁻¹, only used when leakage_method="bidirectional"

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
                "DSC dose=%.2f is outside typical range [0.2, 3.0]. "
                "Values <0.2 may have insufficient SNR; >3.0 risk signal saturation.",
                self.dose,
            )

        # Coerce agent_standard_dose: validate range
        try:
            self.agent_standard_dose = float(self.agent_standard_dose)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"agent_standard_dose must be a number, got {self.agent_standard_dose!r}"
            ) from e
        if not math.isfinite(self.agent_standard_dose) or self.agent_standard_dose <= 0:
            raise ValueError(
                f"agent_standard_dose must be a finite positive number, "
                f"got {self.agent_standard_dose}"
            )
        if self.agent_standard_dose < 0.01 or self.agent_standard_dose > 0.2:
            logger.warning(
                "agent_standard_dose=%.3f outside typical range [0.01, 0.2]. "
                "Use 0.1 for standard GBCA, 0.05 for Vueway, 0.04 for gadoquatrane.",
                self.agent_standard_dose,
            )

        # Coerce float fields (these have non-None defaults)
        try:
            self.smooth_fwhm = float(self.smooth_fwhm)
            self.lesion_distance = float(self.lesion_distance)
            self.motion_thresh = float(self.motion_thresh)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DSCConfig float field has invalid type: {e}") from e

        # Coerce gamma_fit: bool→str for backward compatibility
        if isinstance(self.gamma_fit, bool):
            self.gamma_fit = "on" if self.gamma_fit else "off"
        _valid_registration = {"flirt", "flirt_bbr"}
        if self.registration not in _valid_registration:
            raise ValueError(
                f"Invalid registration={self.registration!r}. "
                f"Valid: {sorted(_valid_registration)}"
            )

        _valid_gamma = {"auto", "on", "off"}
        if self.gamma_fit not in _valid_gamma:
            raise ValueError(
                f"Invalid gamma_fit={self.gamma_fit!r}. "
                f"Valid: {sorted(_valid_gamma)}"
            )

        # Validate leakage correction settings
        _valid_leakage = {"bsw", "bidirectional"}
        if self.leakage_method not in _valid_leakage:
            raise ValueError(
                f"Invalid leakage_method={self.leakage_method!r}. "
                f"Valid: {sorted(_valid_leakage)}"
            )

        # Validate kep if provided (only used for bidirectional)
        if self.kep is not None:
            try:
                self.kep = float(self.kep)
            except (ValueError, TypeError) as e:
                raise TypeError(f"kep must be a number, got {self.kep!r}") from e

            # Import bounds from constants
            from pmriprep.utils.dsc_constants import KEP_DSC_LOWER, KEP_DSC_UPPER
            if not (KEP_DSC_LOWER <= self.kep <= KEP_DSC_UPPER):
                logger.warning(
                    "kep=%.4f s^-1 outside typical range [%.3f, %.3f] s^-1 (Leu 2016)",
                    self.kep, KEP_DSC_LOWER, KEP_DSC_UPPER,
                )

