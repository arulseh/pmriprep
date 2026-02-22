"""DCE (Dynamic Contrast Enhanced) configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class DCEConfig:
    """Configuration for DCE perfusion processing.

    This dataclass contains all parameters for DCE perfusion processing,
    including registration, AIF detection, structural imaging, T1 mapping,
    processing options, model selection, and resource settings.

    Attributes
    ----------
    reference : str
        Reference image for registration (default: postcon T1w).
    lesion_mask : Optional[Path]
        Path to lesion mask for registration weighting, AIF exclusion,
        and lesion statistics.
    nawm_mask : Optional[Path]
        Path to NAWM mask in T1w space (NAWM normalization).
    roi_mask : Optional[Path]
        Path to ROI mask for stats extraction only (no processing impact).
    aif_method : Literal["auto", "population"]
        AIF detection method: auto or population.
    scale_aif : bool
        Scale AIF amplitude to population reference.
    target_peak : Optional[float]
        Target AIF peak concentration (mM) for scaling.
    aif_mask_file : Optional[Path]
        Path to custom AIF mask file.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
    n_aif_candidates : int
        Number of AIF candidate voxels to average.
    underlays : List[str]
        List of underlay images for QC visualizations.
    no_b1 : bool
        Skip B1+ mapping (use nominal flip angles).
    no_t1 : bool
        Skip T1 mapping (use field-strength-appropriate default T1).
    skip_flip_validation : bool
        Skip flip angle validation in VFA T1 mapping.
    lenient_t1_qc : bool
        Use lenient T1 QC ranges (500-2500ms vs strict).
    no_tortuosity : bool
        Skip DWI-based tortuosity correction.
    no_dsc_validation : bool
        Skip DSC-DCE cross-validation.
    smooth_fwhm : float
        Smoothing kernel FWHM (mm) for output maps.
    baseline_frames : int
        Number of baseline frames (0 = auto-detect).
    dose : Literal["half", "full", "double"]
        Contrast agent dose level (default: full).
    tsnr_threshold : float
        tSNR threshold for voxel inclusion (0 = no threshold).
    model : str
        Pharmacokinetic model: auto, tofts, extended_tofts, patlak, 2cxm.
    bayesian : str
        Bayesian inference method: none, map, or mcmc.
    mcmc_samples : int
        Number of MCMC samples for Bayesian inference.
    num_processes : int
        Number of parallel processes (0 = use system default).
    use_gpu : bool
        Enable GPU acceleration.
    hdbet_device : str
        Device for HD-BET brain extraction: auto, cuda, mps, or cpu.
    """

    # Registration settings
    reference: str = "postcon"
    lesion_mask: Optional[Path] = None
    nawm_mask: Optional[Path] = None
    roi_mask: Optional[Path] = None

    # AIF detection settings
    aif_method: Literal["auto", "population"] = "auto"
    scale_aif: bool = True
    target_peak: Optional[float] = None  # None = use field-strength default
    aif_mask_file: Optional[Path] = None
    lesion_distance: float = 15.0
    n_aif_candidates: int = 10

    # Structural settings
    underlays: List[str] = field(default_factory=lambda: ["t1w", "flair"])

    # T1 mapping settings
    no_b1: bool = False
    no_t1: bool = False
    skip_flip_validation: bool = False
    lenient_t1_qc: bool = False
    t1_correction: Literal["auto", "none", "offset", "fallback"] = "auto"
    # Processing settings
    no_tortuosity: bool = False
    no_dsc_validation: bool = False
    smooth_fwhm: float = 0.0  # No smoothing by default
    baseline_frames: int = 0  # 0 = auto-detect
    dose: Literal["half", "full", "double"] = "full"
    tsnr_threshold: float = 0.0  # 0 = no threshold

    # Model settings
    model: str = "auto"
    bayesian: str = "none"
    mcmc_samples: int = 500

    # DSC CBV constraint settings
    cbv_constraint_mode: str = "auto"  # "auto", "hard", "soft", "skip"
    hematocrit: float = 0.40  # Blood hematocrit for CBV-to-vp conversion

    # Resource settings
    num_processes: int = 0  # 0 = use system default
    use_gpu: bool = False
    hdbet_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

    def __post_init__(self):
        """Ensure numeric fields have correct types after initialization.

        This handles cases where values may be passed as strings from
        YAML/JSON config files or CLI arguments.
        """
        # Coerce integer fields (these have non-None defaults)
        try:
            self.baseline_frames = int(self.baseline_frames)
            self.mcmc_samples = int(self.mcmc_samples)
            self.n_aif_candidates = int(self.n_aif_candidates)
            self.num_processes = int(self.num_processes)
        except (ValueError, TypeError) as e:
            raise TypeError(f"DCEConfig integer field has invalid type: {e}") from e

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
