"""DSC (Dynamic Susceptibility Contrast) configuration dataclass."""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

from .constants import DOSE_PRESETS

#: Preload input: "auto" classifies preload adequacy from the K2 leakage
#: residuals (K2-primary; the ΔR2* signal detection still runs but is an
#: advisory QC diagnostic only, not used for classification). A float is the
#: administered preload as a multiplier of the agent's standard dose (0.0 = no
#: preload, 0.5 = half-dose, 1.0 = full standard dose). Both map to a
#: PreloadClass (utils/dsc_preload.py) for K2 uncertainty tuning, via separate
#: bin-edge tables (administered dose vs K2-residual adequacy are distinct).
PreloadInput = Union[Literal["auto"], float]

logger = logging.getLogger(__name__)

@dataclass
class DSCConfig:
    """Configuration for DSC perfusion processing.

    This dataclass contains all parameters for DSC perfusion processing,
    including preload detection, deconvolution, AIF detection, registration,
    quantification, and processing options.

    Attributes
    ----------
    preload : "auto" or float
        Contrast preload from a prior injection. `"auto"` (default) classifies
        preload adequacy from the K2 leakage-correction residuals (K2-primary
        classifier); the baseline ΔR2* signal detection still runs but is an
        advisory acquisition-QC diagnostic only and does not drive the
        classification. A float (0.0–2.0) instead specifies the administered
        preload as a fraction of the agent's standard dose (0.0 = no preload,
        0.25 = quarter, 0.5 = half, 1.0 = full). User float and K2 adequacy are
        distinct quantities (administered dose vs residual leakage) and use
        separate bin-edge tables in `dsc_preload.py`, but both map to the same
        class system for K2 uncertainty tuning.
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
        variational Bayes), ``map``, or ``nlls`` (nonlinear least squares).
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
    preload: PreloadInput = "auto"
    dose: float = 1.0
    agent_standard_dose: float = 0.1  # mmol Gd/kg (0.1=standard, 0.05=gadopiclenol, 0.04=gadoquatrane)

    # Deconvolution settings
    method: str = "oSVD"
    inference: Literal["nlls", "map", "vi", "svb"] = "vi"
    mcmc_samples: int = 500
    cpi_control_points: int = 0  # 0 = auto-select
    # Default True: lets CPs migrate to optimal locations via inferred
    # log-spacings; cuts CBF bias on Normal WM and Long MTT tissues by ~50%.
    # Costs ~3x per-voxel compute. Deliberate deviation from FABBER, which ships
    # infer-cpt off ("not recommended"); the log-spacing parameterization
    # (positive, ordered by construction) avoids the instability FABBER warns of.
    # Set False to match FABBER --infer-cpt false.
    cpi_infer_cpt: bool = True
    # FABBER dsc_cpi `inferdelay`, OFF by default (matches FABBER default).
    # When True, fits a per-voxel bolus arrival delay applied as a post-hoc
    # shift of the convolved curve; the residue (hence CBV and MTT) is unaffected.
    cpi_infer_delay: bool = False

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
    leakage_method: Literal["bsw", "bidirectional"] = "bsw"
    kep: Optional[float] = None  # s⁻¹, only used when leakage_method="bidirectional"

    # Additional outputs
    # First-pass relative CBV map for stroke / non-tumor users. Off by default:
    # adds a per-voxel gamma-variate first-pass fit over the brain mask (extra
    # compute, incl. a FAST-WM resample) and is inert (nothing internal uses it).
    first_pass_cbv: bool = False

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
            "cpi_infer_cpt",
            "cpi_infer_delay",
            "disable_aif_fallback",
            "motion_correct",
            "first_pass_cbv",
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
            self.cpi_control_points = int(self.cpi_control_points)
            self.n_aif_candidates = int(self.n_aif_candidates)
            self.num_processes = int(self.num_processes)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DSCConfig integer field has invalid type: {e}") from e

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

        # preload: "auto" or float dose multiplier. Reject enum strings.
        if isinstance(self.preload, str):
            if self.preload != "auto":
                raise ValueError(
                    f"Invalid preload={self.preload!r}. Use 'auto' to "
                    "auto-detect, or a float dose multiplier (e.g., 0.5 for "
                    "half-dose preload). String enums "
                    "('none'/'minimal'/'half_dose'/...) are not accepted; "
                    "use a numeric fraction."
                )
        else:
            try:
                self.preload = float(self.preload)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"preload must be 'auto' or a number, got {self.preload!r}"
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

        if self.pre_bolus < 3:
            raise ValueError("pre_bolus must be at least 3")
        if self.mcmc_samples <= 0:
            raise ValueError("mcmc_samples must be greater than zero")
        if self.cpi_control_points < 0:
            raise ValueError("cpi_control_points must be non-negative (0 = automatic)")
        if self.n_aif_candidates < 3:
            raise ValueError("n_aif_candidates must be at least 3")
        if self.num_processes < 0:
            raise ValueError("num_processes must be non-negative (0 = automatic)")
        for option in ("smooth_fwhm", "lesion_distance", "motion_thresh"):
            value = getattr(self, option)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{option} must be finite and non-negative")
