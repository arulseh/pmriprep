"""DSC (Dynamic Susceptibility Contrast) configuration dataclass."""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

#: Administered preload as a multiplier of the agent's standard dose, or
#: ``"unknown"`` when the administered preload is not known.
PreloadInput = Union[Literal["unknown"], float]

logger = logging.getLogger(__name__)

@dataclass
class DSCConfig:
    """Configuration for DSC perfusion processing.

    This dataclass contains all parameters for DSC perfusion processing,
    including preload provenance, deconvolution, AIF detection, registration,
    quantification, and processing options.

    Attributes
    ----------
    preload : "unknown" or float
        Contrast preload from a prior injection. `"unknown"` (default) records
        the administered preload as unknown; the baseline ΔR2* measurement is
        retained as an advisory acquisition-QC diagnostic. A float (0.0–2.0)
        specifies the administered
        preload as a fraction of the agent's standard dose (0.0 = no preload,
        0.25 = quarter, 0.5 = half, 1.0 = full). It is provenance only and does
        not gate or tune leakage correction or the DCE vp prior.
    dose : float
        Contrast agent dose multiplier (0.5 = half, 1.0 = full, 2.0 = double).
    agent_standard_dose : float
        Agent's recommended dose in mmol Gd/kg. Default 0.1 (standard).
        Use 0.05 for gadopiclenol, 0.04 for gadoquatrane.
    method : str
        Deconvolution method: sSVD, oSVD, tikhonov, vascular, or cpi.
    inference : str
        Inference method for parametric deconvolution:
        ``vi`` (default — linearized variational inference), ``svb`` (spatial
        variational Bayes). Additional inference engines are retained for
        internal validation but are not part of the documented user surface.
    mcmc_samples : int
        Number of MCMC samples for Bayesian inference.
    cpi_control_points : int
        Number of control points for CPI (0 = auto-select; manual 5–12).
    cpi_infer_delay : bool
        Infer arrival delay. False conditions it at zero and reports it missing.
    cpi_dispersion : {"auto", "none", "gamma"}
        Automatic plain/gamma selection or a specified arterial transport model.
    cpi_fallback : {"osvd", "none"}
        Substitute rejected CPI measurements with oSVD, or leave them missing.
    aif_file : Optional[Path]
        Path to external AIF curve file (.txt, .npz, or .npy).
        Values must be DeltaR2* concentration in s^-1.
        Bypasses AIF detection when provided.
    aif_mask_file : Optional[Path]
        Path to custom AIF mask file.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
    n_aif_candidates : int
        Maximum number of AIF candidate voxels to retain and average; fewer
        may survive selection.
    disable_aif_fallback : bool
        Disable the synthetic AIF fallback. Default False (on a detection
        failure the pipeline substitutes a synthetic gamma-variate AIF and flags
        it). When True, the pipeline instead **errors** if no AIF candidate is
        found at any cascade tier, and **retains the detected AIF (flagged in
        QC)** rather than substituting synthetic when its gamma fit is below the
        quality floor.
    lesion_mask : Optional[Path]
        Path to lesion mask in DSC native space for AIF exclusion and
        lesion statistics.
    nawm_mask : Optional[Path]
        Path to NAWM mask in DSC native space for CBV normalization.
    roi_mask : Optional[Path]
        Path to ROI mask in DSC native space for stats extraction only.
    smooth_fwhm : float
        Smoothing kernel FWHM (mm) for output maps.
    underlays : List[str]
        List of underlay images for QC visualizations.
    motion_thresh : float
        Framewise displacement threshold (mm) for motion outliers.
    motion_correct : bool
        Apply motion correction. True (default) runs mcflirt and processes the
        realigned series. False assumes the input was already motion-corrected
        (e.g. on-scanner); mcflirt still runs to report residual motion for QC,
        but the uncorrected series is processed to avoid a second interpolation.
    pre_bolus : int
        Number of pre-bolus baseline frames for S0 estimation.
        AIF detection auto-detects actual bolus arrival independently.
    num_processes : int
        Number of parallel processes (0 = use system default).
    hdbet_device : Literal["auto", "cuda", "mps", "cpu"]
        Device for HD-BET brain extraction: auto, cuda, mps, or cpu.
    aif_method : Literal["auto", "synthetic"]
        AIF detection method: auto or synthetic.
    leakage_method : Literal["bsw", "none"]
        Leakage-correction policy. "bsw" applies signed BSW correction;
        "none" retains the uncorrected concentration series.
    """

    # Preload settings
    preload: PreloadInput = "unknown"
    dose: float = 1.0
    agent_standard_dose: float = 0.1  # mmol Gd/kg (0.1=standard, 0.05=gadopiclenol, 0.04=gadoquatrane)

    # Deconvolution settings
    method: str = "oSVD"
    inference: Literal["nlls", "map", "vi", "svb"] = "vi"
    mcmc_samples: int = 500
    cpi_control_points: int = 0  # 0 = auto-select
    cpi_infer_delay: bool = True
    cpi_dispersion: Literal["auto", "none", "gamma"] = "none"
    cpi_fallback: Literal["osvd", "none"] = "osvd"

    # AIF detection settings
    aif_file: Optional[Path] = None
    aif_mask_file: Optional[Path] = None
    lesion_distance: float = 15.0
    n_aif_candidates: int = 10
    disable_aif_fallback: bool = False
    aif_method: Literal["auto", "synthetic"] = "auto"

    # User-supplied masks (DSC native space)
    lesion_mask: Optional[Path] = None
    nawm_mask: Optional[Path] = None
    roi_mask: Optional[Path] = None

    # Processing settings
    smooth_fwhm: float = 0.0  # No smoothing by default
    underlays: List[str] = field(default_factory=list)
    motion_thresh: float = 0.5
    motion_correct: bool = True  # False = assume input already motion-corrected
    pre_bolus: int = 5

    # Resource settings
    num_processes: int = 0  # 0 = use system default
    hdbet_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    hdbet_conda_env: Optional[str] = None

    # Leakage correction settings
    leakage_method: Literal["bsw", "none"] = "bsw"

    # Testing / debug
    sloppy: bool = False  # Use fast, low-quality settings for testing

    def __post_init__(self) -> None:
        """Ensure numeric fields have correct types after initialization.

        This handles cases where values may be passed as strings from
        YAML/JSON config files or CLI arguments.
        """
        categorical_options = {
            "method": ({"sSVD", "oSVD", "tikhonov", "vascular", "cpi"}, self.method),
            "inference": ({"nlls", "map", "vi", "svb"}, self.inference),
            "aif_method": ({"auto", "synthetic"}, self.aif_method),
            "hdbet_device": ({"auto", "cuda", "mps", "cpu"}, self.hdbet_device),
        }
        for option, (valid, value) in categorical_options.items():
            if value not in valid:
                raise ValueError(
                    f"Invalid {option}={value!r}. Valid: {sorted(valid)}"
                )

        for option in (
            "cpi_infer_delay",
            "disable_aif_fallback",
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

        # Coerce integer fields (these have non-None defaults)
        try:
            self.pre_bolus = int(self.pre_bolus)
            self.mcmc_samples = int(self.mcmc_samples)
            self.n_aif_candidates = int(self.n_aif_candidates)
            self.num_processes = int(self.num_processes)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DSCConfig integer field has invalid type: {e}") from e

        if isinstance(self.dose, (str, bool)):
            raise TypeError(
                f"DSC dose must be an unquoted numeric multiplier, got {self.dose!r}"
            )
        try:
            self.dose = float(self.dose)
        except (ValueError, TypeError) as e:
            raise TypeError("DSC dose must be a numeric multiplier") from e
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

        # preload: "unknown" or float dose multiplier. Reject enum strings.
        if isinstance(self.preload, str):
            if self.preload != "unknown":
                raise ValueError(
                    f"Invalid preload={self.preload!r}. Use 'unknown' when the "
                    "administered preload is unknown, or a float dose multiplier (e.g., 0.5 for "
                    "half-dose preload). String enums "
                    "('auto'/'none'/'minimal'/'half_dose'/...) are not accepted; "
                    "use a numeric fraction."
                )
        else:
            try:
                self.preload = float(self.preload)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"preload must be 'unknown' or a number, got {self.preload!r}"
                ) from e
            if not math.isfinite(self.preload):
                raise ValueError(
                    f"preload must be finite, got {self.preload}"
                )
            if not (0.0 <= self.preload <= 2.0):
                raise ValueError(
                    f"preload={self.preload} outside valid range [0.0, 2.0]"
                )
            if self.preload > 1.5:
                logger.warning(
                    "preload=%.2f is unusually high (>1.5x standard); "
                    "verify this matches the administered preload dose.",
                    self.preload,
                )

        # Validate leakage correction settings
        _valid_leakage = {"bsw", "none"}
        if self.leakage_method not in _valid_leakage:
            raise ValueError(
                f"Invalid leakage_method={self.leakage_method!r}. "
                f"Valid: {sorted(_valid_leakage)}"
            )

        if self.pre_bolus < 3:
            raise ValueError("pre_bolus must be at least 3")
        if self.mcmc_samples <= 0:
            raise ValueError("mcmc_samples must be greater than zero")
        from pmriprep.models.dsc.signal_cpi.settings import CPIOptions
        CPIOptions(self.cpi_control_points, self.cpi_infer_delay,
                   self.cpi_dispersion, self.cpi_fallback)
        if self.method == "cpi" and self.inference not in ("vi", "svb"):
            raise ValueError("CPI supports vi or svb inference")
        if self.n_aif_candidates < 3:
            raise ValueError("n_aif_candidates must be at least 3")
        if self.num_processes < 0:
            raise ValueError("num_processes must be non-negative (0 = automatic)")
        for option in ("smooth_fwhm", "lesion_distance", "motion_thresh"):
            value = getattr(self, option)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{option} must be finite and non-negative")
