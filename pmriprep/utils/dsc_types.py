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
    vascular model parameters, and Bayesian uncertainty estimates.

    Attributes
    ----------
        # Core perfusion parameters
        cbf: Cerebral blood flow (mL/100g/min). Relative if no r2* provided,
            absolute if r2* is used. Extracted from max(R(t)).
        cbv: Cerebral blood volume (mL/100g). Computed as integral of
            residue function.
        mtt: Mean transit time (seconds). Central volume theorem: CBV/CBF.
        ttp: Time to peak of tissue concentration curve (seconds).

        # Derived metrics
        psr: Percent signal recovery (%). Values <70% indicate BBB disruption,
            >90% indicate intact BBB.
        ph: Peak height as percentage of baseline signal (%).
        tmax: Time to maximum of residue function (seconds). Ischemic tissue
            typically shows Tmax > 6s.

        # AUC metrics
        auc_raw: Area under uncorrected ΔR2* curve (a.u.).
        auc_corr: Area under leakage-corrected ΔR2* curve (a.u.).

        # Leakage correction
        k2: K2 leakage correction factor (s^-1). Values ~0 indicate intact BBB,
            >0.5 indicate significant BBB disruption.
        correction_applied: Whether leakage correction was applied.

        # Vascular (gamma-variate) model parameters
        alpha: Shape parameter of gamma-variate vascular transit time
            distribution. Typical range: 0.5-10.
        beta: Scale parameter (seconds) of gamma-variate distribution.
            Typical range: 0.1-10.
        cth: Capillary transit time heterogeneity (seconds).
            CTH = beta * sqrt(alpha).

        # Bayesian uncertainty estimates (only populated for Bayesian methods)
        cbf_std: Standard deviation of CBF posterior distribution.
        cbv_std: Standard deviation of CBV posterior distribution.
        cbf_credible_lower: Lower bound of CBF 95% credible interval.
        cbf_credible_upper: Upper bound of CBF 95% credible interval.
        cbv_credible_lower: Lower bound of CBV 95% credible interval.
        cbv_credible_upper: Upper bound of CBV 95% credible interval.
        mtt_std: Standard deviation of MTT posterior distribution.
        mtt_credible_lower: Lower bound of MTT 95% credible interval.
        mtt_credible_upper: Upper bound of MTT 95% credible interval.
    """

    # Core perfusion parameters
    cbf: float = 0.0
    cbv: float = 0.0
    mtt: float = 0.0
    ttp: float = 0.0

    # Derived metrics
    psr: float = 0.0
    ph: float = 0.0
    tmax: float = 0.0

    # AUC metrics
    auc_raw: float = 0.0
    auc_corr: float = 0.0

    # Leakage correction
    k2: float = 0.0
    correction_applied: bool = False

    # Vascular model parameters (optional, populated for vascular method)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    cth: Optional[float] = None

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

    def compute_derived_metrics(self) -> None:
        """Compute derived metrics from core parameters.

        Updates MTT from CBV/CBF (central volume theorem) and CTH
        from vascular model parameters if available.
        """
        # MTT from central volume theorem
        if self.cbf > 0:
            # CBV is in mL/100g, CBF is in mL/100g/min
            # MTT = CBV / CBF * 60 to convert from min to seconds
            self.mtt = (self.cbv / self.cbf) * 60.0

        # CTH from vascular model
        if self.alpha is not None and self.beta is not None:
            import math
            self.cth = self.beta * math.sqrt(self.alpha)

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

    def has_vascular_params(self) -> bool:
        """Check if vascular model parameters are available.

        Returns
        -------
        bool
            True if vascular model parameters (alpha, beta) are populated.
        """
        return self.alpha is not None and self.beta is not None


@dataclass
class DSCQCMetrics:
    """Quality control metrics for DSC perfusion analysis.

    Attributes
    ----------
        cnr: Contrast-to-noise ratio of the bolus passage.
        motion_max_fd: Maximum framewise displacement during acquisition.
        motion_mean_fd: Mean framewise displacement during acquisition.
        outlier_frames: Number of frames flagged as motion outliers.
        aif_quality: AIF detection quality score (0-1).
        aif_detection_level: AIF detection method used ('atlas', 'relaxed', 'synthetic').
        preload_classification: Detected preload state.
        rhat_max: Maximum Gelman-Rubin R-hat for Bayesian fits.
        ess_min: Minimum effective sample size for Bayesian fits.
    """

    cnr: float = 0.0
    motion_max_fd: float = 0.0
    motion_mean_fd: float = 0.0
    outlier_frames: int = 0
    aif_quality: float = 0.0
    aif_detection_level: str = ""
    preload_classification: str = ""
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
