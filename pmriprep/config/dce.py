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
        Path to lesion mask for registration weighting.
    lesion_dilation : float
        Dilation radius (mm) for lesion mask in registration.
    skip_bbr : bool
        Skip boundary-based registration (use FLIRT only).
    aif_method : Literal["auto", "population"]
        AIF detection method: auto or population.
    scale_aif : bool
        Scale AIF amplitude to population reference.
    target_peak : Optional[float]
        Target AIF peak concentration (mM) for scaling.
    mni_atlas : str
        MNI atlas name for AIF detection.
    aif_mask_file : Optional[Path]
        Path to custom AIF mask file.
    lesion_distance : float
        Minimum distance (mm) from lesion for AIF candidates.
    prefer_contralateral : bool
        Prefer contralateral AIF candidates when lesion present.
    underlays : List[str]
        List of underlay images for QC visualizations.
    reference_roi : Optional[Path]
        Path to ROI for reference-based analysis.
    no_b1 : bool
        Skip B1+ mapping (use nominal flip angles).
    no_t1 : bool
        Skip T1 mapping (use default T1=1200ms).
    skip_flip_validation : bool
        Skip flip angle validation in VFA T1 mapping.
    lenient_t1_qc : bool
        Use lenient T1 QC ranges (500-2500ms vs strict).
    no_tortuosity : bool
        Skip DWI-based tortuosity correction.
    no_dsc_validation : bool
        Skip DSC-DCE cross-validation.
    smooth : float
        Smoothing kernel FWHM (mm) for output maps.
    save_uncertainty : bool
        Save uncertainty maps from Bayesian inference.
    leakage : str
        Leakage correction method (none, k2, patlak).
    cbv_source : str
        CBV source for DSC constraint (dsc, free).
    baseline_frames : int
        Number of baseline frames (0 = auto-detect).
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
    """

    # Registration settings
    reference: str = "postcon"
    lesion_mask: Optional[Path] = None
    lesion_dilation: float = 15.0
    skip_bbr: bool = False

    # AIF detection settings
    aif_method: Literal["auto", "population"] = "auto"
    scale_aif: bool = False
    target_peak: Optional[float] = None  # None = use field-strength default
    mni_atlas: str = "MNI152NLin2009cAsym"
    aif_mask_file: Optional[Path] = None
    lesion_distance: float = 20.0
    prefer_contralateral: bool = True

    # Structural settings
    underlays: List[str] = field(default_factory=list)
    reference_roi: Optional[Path] = None

    # T1 mapping settings
    no_b1: bool = False
    no_t1: bool = False
    skip_flip_validation: bool = False
    lenient_t1_qc: bool = False

    # Processing settings
    no_tortuosity: bool = False
    no_dsc_validation: bool = False
    smooth: float = 0.0  # No smoothing by default
    save_uncertainty: bool = False
    leakage: str = "none"
    cbv_source: str = "dsc"  # Use DSC CBV when available
    baseline_frames: int = 0  # 0 = auto-detect
    tsnr_threshold: float = 0.0  # 0 = no threshold

    # Model settings
    model: str = "auto"
    bayesian: str = "none"
    mcmc_samples: int = 500

    # Resource settings
    num_processes: int = 0  # 0 = use system default
    use_gpu: bool = False
