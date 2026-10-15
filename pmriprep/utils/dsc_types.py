"""DSC perfusion type definitions.

Type-safe data structures for DSC perfusion processing results,
including voxel-level perfusion parameters and uncertainty estimates.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DSCVoxelResult:
    """Complete result for a single voxel from DSC perfusion analysis.

    Contains all perfusion parameters derived from DSC-MRI processing,
    including basic hemodynamic parameters, model-derived metrics,
    and Bayesian uncertainty estimates.

    Attributes
    ----------
    cbf : float
        Relative cerebral blood flow supplied by the estimator.
    cbv : float
        Relative cerebral blood volume supplied by the estimator.
    mtt : float
        Mean transit time in seconds.
    ttp : float
        Tissue-concentration peak time in seconds.
    psr : float
        Percent signal recovery, without diagnostic classification.
    tmax : float
        Residue-peak time in seconds, without diagnostic classification.
    auc_raw, auc_corr : float
        Areas under the uncorrected and leakage-corrected delta-R2-star curves.
    k2 : float
        Fitted leakage correction coefficient in s^-1.
    correction_applied : bool
        Whether leakage correction was applied.
    cbf_std, mtt_std : float, optional
        Posterior standard deviations when supplied by the estimator.
    cbf_credible_lower, cbf_credible_upper : float, optional
        Lower and upper CBF 95% credible bounds.
    mtt_credible_lower, mtt_credible_upper : float, optional
        Lower and upper MTT 95% credible bounds.
    cbv_std, cbv_credible_lower, cbv_credible_upper : float, optional
        Reserved for covariance-aware CBV posterior uncertainty.
    """

    # Core perfusion parameters
    cbf: float = 0.0
    cbv: float = 0.0
    mtt: float = 0.0
    ttp: float = 0.0

    # Derived metrics
    psr: float = 0.0
    tmax: float = 0.0

    # AUC metrics
    auc_raw: float = 0.0
    auc_corr: float = 0.0

    # Leakage correction
    k2: float = 0.0
    correction_applied: bool = False

    # Bayesian uncertainty estimates (optional, populated for Bayesian methods)
    cbf_std: Optional[float] = None
    cbv_std: Optional[float] = None
    cbf_credible_lower: Optional[float] = None
    cbf_credible_upper: Optional[float] = None
    cbv_credible_lower: Optional[float] = None
    cbv_credible_upper: Optional[float] = None
    mtt_std: Optional[float] = None
    mtt_credible_lower: Optional[float] = None
    mtt_credible_upper: Optional[float] = None

    def has_uncertainty(self) -> bool:
        """Check if Bayesian uncertainty estimates are available.

        Returns
        -------
        bool
            True if any uncertainty estimates are populated.
        """
        return any([
            self.cbf_std is not None,
            self.cbv_std is not None,
            self.cbf_credible_lower is not None,
        ])


@dataclass
class DSCQCMetrics:
    """Quality control metrics for DSC perfusion analysis.

    Attributes
    ----------
    cnr : float
        Contrast-to-noise ratio of the bolus passage.
    motion_max_fd, motion_mean_fd : float
        Maximum and mean framewise displacement in millimeters.
    outlier_frames : int
        Number of frames flagged as motion outliers.
    aif_quality : float
        AIF detection quality score (0–1).
    aif_detection_level : str
        AIF detection provenance supplied by the caller.
    rhat_max : float, optional
        Maximum Gelman-Rubin R-hat when supplied by a sampling estimator.
    ess_min : int, optional
        Minimum effective sample size when supplied by a sampling estimator.
    """

    cnr: float = 0.0
    motion_max_fd: float = 0.0
    motion_mean_fd: float = 0.0
    outlier_frames: int = 0
    aif_quality: float = 0.0
    aif_detection_level: str = ""
    rhat_max: Optional[float] = None
    ess_min: Optional[int] = None

    def passes_motion_qc(self, threshold: float = 0.5) -> bool:
        """Check if motion QC passes based on framewise displacement.

        Parameters
        ----------
        threshold : float, optional
            Maximum FD threshold in mm (default 0.5).

        Returns
        -------
        bool
            True if motion QC passes (max FD < threshold).
        """
        return self.motion_max_fd < threshold

    def passes_bayesian_qc(self) -> bool:
        """Check if Bayesian convergence diagnostics pass.

        Returns
        -------
        bool
            True if R-hat < 1.1 and ESS > 100, or if not applicable.
        """
        if self.rhat_max is None or self.ess_min is None:
            return True  # Not applicable for non-Bayesian methods
        return self.rhat_max < 1.1 and self.ess_min > 100


@dataclass
class DSCAIFResult:
    """Result from AIF detection.

    Attributes
    ----------
        aif_curve: The arterial input function concentration time course.
        aif_mask: Binary mask indicating AIF voxel locations.
        detection_level: Method used ('atlas', 'relaxed', 'synthetic').
        quality_score: Overall AIF quality score (0-1).
        gamma_fit_applied: Whether gamma-variate fitting was applied.
        candidate_count: Number of candidate voxels evaluated.
        metrics: Detailed AIF quality metrics.
    """

    aif_curve: list[float] = field(default_factory=list)
    aif_mask: Optional[list[list[list[int]]]] = None  # 3D mask
    detection_level: str = ""
    quality_score: float = 0.0
    gamma_fit_applied: bool = False
    candidate_count: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if AIF detection was successful.

        Returns
        -------
        bool
            True if AIF curve has non-zero values and quality is acceptable.
        """
        if not self.aif_curve:
            return False
        return len(self.aif_curve) > 0 and self.quality_score > 0.0

    def is_synthetic(self) -> bool:
        """Check if synthetic (population) AIF was used.

        Returns
        -------
        bool
            True if synthetic AIF was used as fallback.
        """
        return self.detection_level == "synthetic"
