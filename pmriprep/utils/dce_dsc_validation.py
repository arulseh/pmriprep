"""DSC-DCE cross-validation utilities.

Post-hoc comparison of DCE-derived vp (plasma volume fraction) with a DSC-derived
volume measure to validate quantification and detect preload effects.

Two calibration paths are supported:

- **nCBV path (default when the workflow supplies a NAWM-normalized CBV):**
  expected ratio = ``NAWM_VP_REFERENCE`` (dimensionless); the comparison is
  ``vp / nCBV`` voxelwise. NAWM normalization cancels AIF partial-volume,
  recirculation, and first-pass truncation biases, which is why this is
  the scientifically defensible path for clinical DSC.

- **Legacy absolute-CBV path:** expected ratio = ``(1 - Hct_cap) × ρ ≈ 0.669``
  assuming CBV in mL/100g and vp × 100 in mL/100mL. Retained for backward
  compatibility and for any caller that explicitly passes raw CBV.

Ratio deviations indicate:
- Ratio << expected: preload effect (reduced DSC signal from prior contrast).
- Ratio >> expected: DCE overestimation or DSC underestimation.

When constraint_mode is "hard", vp is deterministically derived from CBV and
correlation is 1.0 by construction; validation reports status = "CIRCULAR"
instead of running meaningless metrics. The circular-std threshold scales
with the ratio magnitude (0.002 on the nCBV path, 0.01 on the absolute path).

References:
    - Boxerman et al., Neuro-Oncology 2020 (ASFNR/NCI consensus — NAWM
      normalization is the reference scale for clinical DSC).
    - Larsson et al., MRM 2009;62:1270 (direct DCE-measured NAWM vp ≈ 0.0089
      at matched Hct=0.42; anchor for ``NAWM_VP_REFERENCE``).
    - Schmainda et al., Theory and Application of Quantitative DCE-MRI (2021).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_constants import (
    HEMATOCRIT_ARTERIAL,
    TISSUE_DENSITY,
    capillary_hematocrit,
)
from pmriprep.utils.qc_standards import QCStatus


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


#: Two-tier ratio tolerance for validation.
#: PASS band: expected × (1 ± 0.30). WARN band: expected × (1 ± 0.50).
#: ±50% (3:1 range) is too generous for PASS; ±30% captures clinically
#: meaningful deviation while allowing for systematic Hct/density uncertainty.
RATIO_TOLERANCE_PASS: float = 0.30
RATIO_TOLERANCE_WARN: float = 0.50

#: Minimum vp value to include in validation (avoid numerical instability)
MIN_VP_FOR_VALIDATION: float = 0.001

#: Minimum CBV value to include in validation (avoid division by zero)
MIN_CBV_FOR_VALIDATION: float = 0.1

#: Maximum vp for validation (above = likely large vessel partial volume effect)
MAX_VP_FOR_VALIDATION: float = 0.15

#: Maximum CBV for validation (above = likely large vessel partial volume effect)
MAX_CBV_FOR_VALIDATION: float = 8.0

#: Spearman rank correlation thresholds for unconstrained vp-CBV validation.
#: ρ=0.40 → ~16% shared variance, "moderate" in literature (Quarles et al.,
#: NMR Biomed 2009; Shin et al., JMRI 2013). ρ<0.20 is effectively noise.
CORRELATION_PASS_UNCONSTRAINED: float = 0.40
CORRELATION_WARN_UNCONSTRAINED: float = 0.20

#: Correlation thresholds for soft-constrained validation.
#: Higher bar because Bayesian prior from CBV should inflate correlation.
CORRELATION_PASS_SOFT: float = 0.50
CORRELATION_WARN_SOFT: float = 0.30

#: Minimum fraction of valid voxels for reliable validation.
#: Thresholds depend on DCE constraint mode because soft-constrained Bayesian
#: vp fitting concentrates WM voxels at the lower bound (prior mean
#: ~NAWM_VP_REFERENCE, σ = 0.40 × prior_mean), so most fall below
#: MIN_VP_FOR_VALIDATION = 0.001.
#: The 5-9% that pass represent well-perfused GM — the most informative
#: subset for cross-validation.
#: Unconstrained fitting has wider scatter → 35-60% expected coverage.

# Unconstrained or skip: 30%+ valid ensures representative brain sample.
# 70% filtered means validation operates on a non-representative subset.
FRACTION_VALID_PASS_UNCONSTRAINED: float = 0.30
FRACTION_VALID_WARN_UNCONSTRAINED: float = 0.15

# Soft constraint: Bayesian prior concentrates vp, reducing valid fraction
FRACTION_VALID_PASS_SOFT: float = 0.05
FRACTION_VALID_WARN_SOFT: float = 0.02

#: Absolute minimum valid voxels for statistical reliability.
MIN_VALID_VOXELS: int = 500

#: Standard deviation threshold for ratio-based circularity detection.
#: When vp is soft-constrained to DSC CBV with tight bounds, the vp/CBV
#: ratio becomes near-constant (std < 0.01). Empirically, circular soft
#: constraints produce std_ratio < 0.001 while genuine fits produce > 0.1.
CIRCULAR_STD_RATIO_THRESHOLD: float = 0.01


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ValidationMetrics:
    """Detailed metrics from DSC-DCE cross-validation.

    Attributes:
        mean_ratio: Mean (vp * 100) / CBV ratio across valid voxels.
        median_ratio: Median ratio across valid voxels.
        std_ratio: Standard deviation of ratio.
        p10_ratio: 10th percentile ratio.
        p90_ratio: 90th percentile ratio.
        n_valid_voxels: Number of voxels included in validation.
        n_total_voxels: Total number of voxels in mask.
        fraction_valid: Fraction of voxels with valid values.
        correlation: Spearman rank correlation between vp and CBV.
            Robust to outliers and non-linear monotonic relationships,
            which is the expected vp-CBV relationship (different contrast
            mechanisms: T1 vs T2*).
        pearson_correlation: Supplementary Pearson correlation.
            Discrepancy with Spearman indicates outlier leverage inflation.
        mean_vp: Mean vp value in valid voxels (fraction).
        mean_cbv: Mean CBV value in valid voxels (mL/100g).
        expected_ratio: Expected ratio based on hematocrit and tissue density.
    """

    mean_ratio: float = 0.0
    median_ratio: float = 0.0
    std_ratio: float = 0.0
    p10_ratio: float = 0.0
    p90_ratio: float = 0.0
    n_valid_voxels: int = 0
    n_total_voxels: int = 0
    fraction_valid: float = 0.0
    correlation: float = 0.0
    pearson_correlation: float = 0.0
    mean_vp: float = 0.0
    mean_cbv: float = 0.0
    expected_ratio: float = 0.0

    def to_dict(self) -> dict[str, Union[float, int]]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "mean_ratio": self.mean_ratio,
            "median_ratio": self.median_ratio,
            "std_ratio": self.std_ratio,
            "p10_ratio": self.p10_ratio,
            "p90_ratio": self.p90_ratio,
            "n_valid_voxels": self.n_valid_voxels,
            "n_total_voxels": self.n_total_voxels,
            "fraction_valid": self.fraction_valid,
            "correlation": self.correlation,
            "pearson_correlation": self.pearson_correlation,
            "mean_vp": self.mean_vp,
            "mean_cbv": self.mean_cbv,
            "expected_ratio": self.expected_ratio,
        }


@dataclass
class ValidationResult:
    """Result of DSC-DCE cross-validation.

    Attributes:
        status: Overall validation status (PASS/WARN/POOR/CIRCULAR).
        preload_effect_detected: Whether preload effect is suspected from
            vp/CBV ratio analysis. Distinct from DSC config ``preload`` which
            records the user/scanner configuration.
        dsc_preload_configured: True when user explicitly overrode
            auto-detection with a preload class (i.e., ``preload != "auto"``).
            None if unknown.
        constraint_mode: CBV constraint mode used (hard/soft/skip/None).
        circular: True when constraint_mode="hard" — vp is deterministically
            derived from CBV, making cross-validation meaningless.
        correction_factor: Suggested correction factor to align vp with CBV.
        metrics: Detailed validation metrics.
        interpretation: Human-readable interpretation of results.
        acquisition_limited: True when the AIF-tissue overshoot fraction
            exceeds the prior-dominance threshold and the nCBV-anchored
            ratio tolerance was widened (Buckley MRM 2002). Informational
            flag; does not imply data-quality failure.
    """

    status: str = QCStatus.UNKNOWN
    preload_effect_detected: bool = False
    dsc_preload_configured: Optional[bool] = None
    constraint_mode: Optional[str] = None
    circular: bool = False
    correction_factor: float = 1.0
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    interpretation: str = ""
    acquisition_limited: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "preload_effect_detected": self.preload_effect_detected,
            "dsc_preload_configured": self.dsc_preload_configured,
            "constraint_mode": self.constraint_mode,
            "circular": self.circular,
            "correction_factor": self.correction_factor,
            "metrics": self.metrics.to_dict(),
            "interpretation": self.interpretation,
            "acquisition_limited": self.acquisition_limited,
        }


# =============================================================================
# Ratio Calculation Functions
# =============================================================================


def calculate_vp_cbv_ratio(
    vp: npt.NDArray[np.floating[npt.NBitBase]],
    cbv: npt.NDArray[np.floating[npt.NBitBase]],
) -> npt.NDArray[np.floating[npt.NBitBase]]:
    """Calculate (vp * 100) / CBV ratio.

    Args:
        vp: Plasma volume fraction array (0-1 scale).
        cbv: Cerebral blood volume array (mL/100g scale).

    Returns:
        Ratio array with same shape as inputs.
    """
    # Avoid division by zero
    cbv_safe = np.maximum(cbv, MIN_CBV_FOR_VALIDATION)
    ratio = (vp * 100.0) / cbv_safe
    return np.asarray(ratio)


def get_expected_ratio(
    hematocrit: float = HEMATOCRIT_ARTERIAL,
    tissue_density: float = TISSUE_DENSITY,
) -> float:
    """Calculate expected vp/CBV ratio from physiological parameters.

    The expected ratio relates plasma volume (vp) to whole blood volume (CBV):
        expected_ratio = (1 - Hct_capillary) * tissue_density

    Capillary hematocrit is derived from arterial via Fåhræus ratio (0.85).

    Args:
        hematocrit: Arterial hematocrit (default: 0.42).
        tissue_density: Tissue density in g/mL (default: 1.04).

    Returns:
        Expected (vp * 100) / CBV ratio.
    """
    hct_cap = capillary_hematocrit(hematocrit)
    return (1.0 - hct_cap) * tissue_density


def calculate_correction_factor(
    observed_ratio: float,
    expected_ratio: float,
) -> float:
    """Calculate correction factor to align observed ratio with expected.

    The correction factor can be applied to vp or (inversely) to CBV:
        vp_corrected = vp * correction_factor
        cbv_corrected = cbv / correction_factor

    Args:
        observed_ratio: Observed mean or median vp/CBV ratio.
        expected_ratio: Expected ratio from get_expected_ratio().

    Returns:
        Correction factor (>1 if observed ratio is low, <1 if high).
    """
    if observed_ratio <= 0:
        return 1.0

    return expected_ratio / observed_ratio


# =============================================================================
# Validation Functions
# =============================================================================


def validate_vp_cbv(
    vp_data: npt.NDArray[np.floating[npt.NBitBase]],
    cbv_data: npt.NDArray[np.floating[npt.NBitBase]],
    mask: npt.NDArray[np.bool_],
    hematocrit: float = HEMATOCRIT_ARTERIAL,
    tissue_density: float = TISSUE_DENSITY,
    constraint_mode: Optional[str] = None,
    dsc_preload_configured: Optional[bool] = None,
    use_ncbv_scale: bool = False,
    *,
    aif_tissue_overshoot_fraction: Optional[float] = None,
) -> ValidationResult:
    """Validate DCE vp against DSC CBV.

    Compares vp (plasma volume fraction from DCE) with CBV (cerebral blood
    volume from DSC) to detect quantification issues or preload effects.

    When constraint_mode is "hard", vp is deterministically derived from CBV
    (circular), so ratio/correlation metrics are skipped and status is set to
    "CIRCULAR". The caller should reference delta-R2 from pk_qc instead.

    Two calibration modes:

    - **Absolute CBV (``use_ncbv_scale=False``):** CBV in mL/100g.
      Expected ratio = ``(1-Hct_cap)·ρ ≈ 0.669`` (Tofts 1999 conversion
      of blood-voxel CBV to plasma volume fraction). Circular threshold
      applies to ``std(vp/CBV) < 0.01``.

      .. warning::
         pmriprep itself does NOT produce absolute mL/100g CBV — the
         only DSC CBV outputs are dimensionless ``rel_cbv_auc_map``
         (AIF-normalized) and ``ncbv_map`` (NAWM-normalized). Against
         dimensionless rel-AUC-CBV the literature expected ratio 0.669
         is miscalibrated and this path will emit false POOR labels.
         The pmriprep workflow skips xval (returns ``status="SKIPPED"``)
         on subjects without NAWM rather than routing through this
         path. This code path is retained for external callers that
         legitimately have absolute CBV (e.g., pipelines with a κ
         calibration or steady-state reference); no universal
         ``vp / rel_AUC_CBV`` expected ratio exists without NAWM
         anchoring (Rempp 1994; Knutsson 2010).

    - **nCBV (``use_ncbv_scale=True``):** CBV is NAWM-normalized
      (dimensionless, NAWM=1). Expected ratio = ``NAWM_VP_REFERENCE`` by
      construction — vp prior is ``NAWM_VP_REFERENCE × nCBV``. Circular
      threshold scales to ``std(vp/nCBV) < 0.002`` (~10–20% of the ratio
      center); the non-nCBV path's 0.01 would misflag genuine 50%
      posterior spread as circular at this scale.

    Args:
        vp_data: 3D vp map array (fraction, 0-1 scale).
        cbv_data: 3D CBV map array — absolute/relative in mL/100g if
            ``use_ncbv_scale=False``, dimensionless nCBV if True.
        mask: 3D boolean mask for analysis region.
        hematocrit: Hematocrit value (only used when ``use_ncbv_scale=False``).
        tissue_density: Tissue density (only used when ``use_ncbv_scale=False``).
        constraint_mode: CBV constraint mode ("hard", "soft", "skip", or None).
        dsc_preload_configured: Whether DSC preload was configured.
        use_ncbv_scale: When True, interpret ``cbv_data`` as NAWM-normalized
            nCBV and use the NAWM-anchored expected ratio + scaled circular
            threshold. Default False (legacy behavior).
        aif_tissue_overshoot_fraction: Upstream constraint-block diagnostic
            (fraction of voxels where ``vp_prior × Cp(peak) > observed
            C_tissue(peak)``). When greater than
            ``ACQUISITION_LIMITED_OVERSHOOT_THRESHOLD`` on the nCBV path,
            vp is prior-dominated by first-pass temporal undersampling
            (Buckley MRM 2002); the ratio tolerance is widened to
            [``VP_NCBV_RATIO_MIN_LIMITED``, ``VP_NCBV_RATIO_MAX_LIMITED``]
            and ``result.acquisition_limited`` is set. Pass ``None`` to
            skip (legacy tight-band behavior).

    Returns:
        ValidationResult with status, preload detection, and metrics.
    """
    from pmriprep.utils.dce_constants import (
        ACQUISITION_LIMITED_OVERSHOOT_THRESHOLD,
        NAWM_VP_REFERENCE,
        CIRCULAR_STD_THRESHOLD_NCBV,
        CIRCULAR_STD_THRESHOLD_ABS_CBV,
        MIN_NCBV_FOR_VALIDATION,
        MAX_NCBV_FOR_VALIDATION,
        VP_NCBV_RATIO_MIN_LIMITED,
        VP_NCBV_RATIO_MAX_LIMITED,
    )

    result = ValidationResult(
        constraint_mode=constraint_mode,
        dsc_preload_configured=dsc_preload_configured,
    )
    metrics = ValidationMetrics()

    # Expected ratio depends on calibration mode
    if use_ncbv_scale:
        expected_ratio = NAWM_VP_REFERENCE
        circular_std_threshold = CIRCULAR_STD_THRESHOLD_NCBV
        cbv_min = MIN_NCBV_FOR_VALIDATION
        cbv_max = MAX_NCBV_FOR_VALIDATION
    else:
        expected_ratio = get_expected_ratio(hematocrit, tissue_density)
        circular_std_threshold = CIRCULAR_STD_THRESHOLD_ABS_CBV
        cbv_min = MIN_CBV_FOR_VALIDATION
        cbv_max = MAX_CBV_FOR_VALIDATION
    metrics.expected_ratio = expected_ratio

    # Compute two-tier ratio thresholds from expected ratio
    ratio_pass_low = expected_ratio * (1.0 - RATIO_TOLERANCE_PASS)
    ratio_pass_high = expected_ratio * (1.0 + RATIO_TOLERANCE_PASS)
    ratio_warn_low = expected_ratio * (1.0 - RATIO_TOLERANCE_WARN)
    ratio_warn_high = expected_ratio * (1.0 + RATIO_TOLERANCE_WARN)

    # Extract values within mask
    vp_masked = vp_data[mask]
    cbv_masked = cbv_data[mask]
    metrics.n_total_voxels = int(np.sum(mask))

    if metrics.n_total_voxels == 0:
        logger.warning("Empty mask, cannot perform validation")
        result.status = QCStatus.POOR
        result.interpretation = "Empty mask - no voxels for validation"
        result.metrics = metrics
        return result

    # Detect circularity: hard constraint means vp = f(CBV) deterministically
    if constraint_mode == "hard":
        result.circular = True
        result.status = QCStatus.CIRCULAR
        result.interpretation = (
            "vp hard-constrained from CBV -- cross-validation is circular. "
            "See delta-R2 in pk_qc for constraint benefit assessment."
        )
        result.metrics = metrics
        logger.info(
            "DSC-DCE Validation: CIRCULAR (hard constraint), "
            "skipping ratio/correlation metrics"
        )
        return result

    # Filter to valid voxels (positive values above thresholds, below upper bounds).
    # Range for CBV/nCBV depends on calibration mode (see cbv_min/cbv_max above).
    valid_mask = (
        (vp_masked > MIN_VP_FOR_VALIDATION)
        & (cbv_masked > cbv_min)
        & (vp_masked < MAX_VP_FOR_VALIDATION)
        & (cbv_masked < cbv_max)
        & np.isfinite(vp_masked)
        & np.isfinite(cbv_masked)
    )

    vp_valid = vp_masked[valid_mask]
    cbv_valid = cbv_masked[valid_mask]
    metrics.n_valid_voxels = len(vp_valid)
    metrics.fraction_valid = metrics.n_valid_voxels / metrics.n_total_voxels

    if metrics.n_valid_voxels < MIN_VALID_VOXELS:
        logger.warning(
            f"Insufficient valid voxels for validation: {metrics.n_valid_voxels} "
            f"(minimum {MIN_VALID_VOXELS})"
        )
        result.status = QCStatus.WARN
        result.interpretation = (
            f"Insufficient valid voxels ({metrics.n_valid_voxels}, "
            f"minimum {MIN_VALID_VOXELS}) for reliable validation"
        )
        result.metrics = metrics
        return result

    # Calculate ratio statistics. On the nCBV path both vp and nCBV are
    # dimensionless, so the ratio is just vp/nCBV (no ×100 unit conversion).
    if use_ncbv_scale:
        cbv_safe = np.maximum(cbv_valid, cbv_min)
        ratios = vp_valid / cbv_safe
    else:
        ratios = calculate_vp_cbv_ratio(vp_valid, cbv_valid)

    metrics.mean_ratio = float(np.mean(ratios))
    metrics.median_ratio = float(np.median(ratios))
    metrics.std_ratio = float(np.std(ratios))
    metrics.p10_ratio = float(np.percentile(ratios, 10))
    metrics.p90_ratio = float(np.percentile(ratios, 90))
    metrics.mean_vp = float(np.mean(vp_valid))
    metrics.mean_cbv = float(np.mean(cbv_valid))

    # Calculate correlations with NaN guard.
    # Spearman (rank) is the primary metric — robust to outliers and
    # non-linear monotonic relationships (vp-CBV uses different contrast
    # mechanisms: T1 vs T2*).  Pearson is supplementary; discrepancy
    # with Spearman indicates outlier leverage inflation.
    if len(vp_valid) > 2:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(vp_valid, cbv_valid)
        metrics.correlation = float(rho) if np.isfinite(rho) else 0.0

        pearson_matrix = np.corrcoef(vp_valid, cbv_valid)
        pearson_val = pearson_matrix[0, 1]
        metrics.pearson_correlation = (
            float(pearson_val) if np.isfinite(pearson_val) else 0.0
        )
    else:
        metrics.correlation = 0.0
        metrics.pearson_correlation = 0.0

    # Calculate correction factor based on median ratio
    result.correction_factor = calculate_correction_factor(
        metrics.median_ratio, expected_ratio
    )

    # Detect near-circularity: soft constraint effectively locking vp to CBV.
    # Threshold scales with the ratio magnitude: for nCBV-anchored vp (ratio
    # centered at NAWM_VP_REFERENCE), std ≈ 0.002 is ~10–20% of the ratio
    # center; for absolute CBV (ratio centered at ~0.006 mL/100g), the
    # legacy 0.01 still flags tight-soft and HARD overrides.
    if metrics.std_ratio < circular_std_threshold:
        result.circular = True
        result.status = QCStatus.CIRCULAR
        result.metrics = metrics
        result.interpretation = (
            f"vp/CBV ratio variance too low (std={metrics.std_ratio:.4f}); "
            "soft constraint is functionally hard -- correlation is circular"
        )
        logger.info(
            "Cross-validation CIRCULAR: std_ratio=%.4f < %.4f "
            "(soft constraint effectively hard)",
            metrics.std_ratio, circular_std_threshold,
        )
        return result

    # Determine status from ratio, correlation, and coverage
    interpretations: list[str] = []

    # Acquisition-limited flag: when the upstream overshoot detector fires
    # on the nCBV-anchored path, vp is prior-dominated by first-pass
    # temporal undersampling (Buckley MRM 2002). Widen the ratio band to
    # the physiological envelope instead of flagging WARN on what is a
    # physics ceiling, not a data-quality failure.
    acquisition_limited = (
        use_ncbv_scale
        and aif_tissue_overshoot_fraction is not None
        and aif_tissue_overshoot_fraction > ACQUISITION_LIMITED_OVERSHOOT_THRESHOLD
    )
    result.acquisition_limited = acquisition_limited

    if acquisition_limited:
        # --- Widened ratio band (Pardridge 2011 physiological floor → NAWM ceiling) ---
        if (VP_NCBV_RATIO_MIN_LIMITED
                <= metrics.median_ratio
                <= VP_NCBV_RATIO_MAX_LIMITED):
            ratio_status = QCStatus.PASS
            result.preload_effect_detected = False
            interpretations.append(
                f"vp/nCBV ratio ({metrics.median_ratio:.3f}) within widened "
                f"acquisition-limited band "
                f"[{VP_NCBV_RATIO_MIN_LIMITED:.3f}, {VP_NCBV_RATIO_MAX_LIMITED:.3f}]"
            )
        else:
            ratio_status = QCStatus.WARN
            result.preload_effect_detected = (
                metrics.median_ratio < VP_NCBV_RATIO_MIN_LIMITED
            )
            interpretations.append(
                f"vp/nCBV ratio ({metrics.median_ratio:.3f}) outside widened "
                f"acquisition-limited band "
                f"[{VP_NCBV_RATIO_MIN_LIMITED:.3f}, {VP_NCBV_RATIO_MAX_LIMITED:.3f}]"
            )
        interpretations.append(
            f"Acquisition-limited regime "
            f"(aif_tissue_overshoot_fraction={aif_tissue_overshoot_fraction:.2f}"
            f" > {ACQUISITION_LIMITED_OVERSHOOT_THRESHOLD:.2f}): vp is "
            "prior-dominated by first-pass temporal undersampling of the AIF "
            "(Buckley MRM 2002). Widened ratio band applied; magnitude "
            "validation is informational only."
        )
    # --- Ratio status (two-tier: PASS within ±30%, WARN within ±50%) ---
    elif metrics.median_ratio < ratio_warn_low:
        ratio_status = QCStatus.WARN
        result.preload_effect_detected = True
        interpretations.append(
            f"Low vp/CBV ratio ({metrics.median_ratio:.3f}) suggests preload effect"
        )
        interpretations.append(
            f"Expected ratio: {expected_ratio:.3f}, observed: {metrics.median_ratio:.3f}"
        )
        interpretations.append(
            f"Suggested correction factor: {result.correction_factor:.2f}"
        )
    elif metrics.median_ratio > ratio_warn_high:
        ratio_status = QCStatus.WARN
        result.preload_effect_detected = False
        interpretations.append(
            f"High vp/CBV ratio ({metrics.median_ratio:.3f}) suggests DCE vp overestimation"
        )
        interpretations.append(
            f"Expected ratio: {expected_ratio:.3f}, observed: {metrics.median_ratio:.3f}"
        )
        interpretations.append(
            f"Suggested correction factor: {result.correction_factor:.2f}"
        )
    elif (metrics.median_ratio < ratio_pass_low
            or metrics.median_ratio > ratio_pass_high):
        ratio_status = QCStatus.WARN
        result.preload_effect_detected = False
        interpretations.append(
            f"vp/CBV ratio ({metrics.median_ratio:.3f}) marginally outside "
            f"expected range [{ratio_pass_low:.3f}, {ratio_pass_high:.3f}]"
        )
    else:
        ratio_status = QCStatus.PASS
        result.preload_effect_detected = False
        interpretations.append(
            f"vp/CBV ratio ({metrics.median_ratio:.3f}) within expected range"
        )
        interpretations.append(
            f"Expected: {expected_ratio:.3f}, tolerance: +/-{RATIO_TOLERANCE_PASS*100:.0f}%"
        )

    # --- Correlation status ---
    if constraint_mode == "soft":
        corr_pass = CORRELATION_PASS_SOFT
        corr_warn = CORRELATION_WARN_SOFT
        interpretations.append(
            "vp soft-constrained -- correlation may be inflated"
        )
    else:
        corr_pass = CORRELATION_PASS_UNCONSTRAINED
        corr_warn = CORRELATION_WARN_UNCONSTRAINED

    if metrics.correlation < 0:
        corr_status = QCStatus.POOR
        interpretations.append(
            f"Negative vp-CBV correlation (rho={metrics.correlation:.3f}): "
            "physically implausible"
        )
    elif metrics.correlation < corr_warn:
        corr_status = QCStatus.POOR
        interpretations.append(
            f"No meaningful vp-CBV correlation (rho={metrics.correlation:.3f})"
        )
    elif metrics.correlation < corr_pass:
        corr_status = QCStatus.WARN
        interpretations.append(
            f"Weak vp-CBV correlation (rho={metrics.correlation:.3f})"
        )
    elif metrics.correlation > 0.7:
        corr_status = QCStatus.PASS
        interpretations.append(
            f"Strong vp-CBV correlation (rho={metrics.correlation:.3f})"
        )
    else:
        corr_status = QCStatus.PASS
        interpretations.append(
            f"Moderate vp-CBV correlation (rho={metrics.correlation:.3f})"
        )

    # --- Coverage status (constraint-mode-dependent thresholds) ---
    if constraint_mode == "soft":
        frac_pass = FRACTION_VALID_PASS_SOFT
        frac_warn = FRACTION_VALID_WARN_SOFT
    else:
        frac_pass = FRACTION_VALID_PASS_UNCONSTRAINED
        frac_warn = FRACTION_VALID_WARN_UNCONSTRAINED

    if metrics.fraction_valid < frac_warn:
        coverage_status = QCStatus.POOR
        interpretations.append(
            f"Insufficient overlap: only {metrics.fraction_valid:.1%} of voxels "
            f"passed filtering (threshold {frac_warn:.0%}); validation unreliable"
        )
    elif metrics.fraction_valid < frac_pass:
        coverage_status = QCStatus.WARN
        interpretations.append(
            f"Limited overlap: only {metrics.fraction_valid:.1%} of voxels "
            f"passed filtering (threshold {frac_pass:.0%})"
        )
    else:
        coverage_status = QCStatus.PASS

    # --- Final status = worst of all three ---
    result.status = QCStatus.worst([ratio_status, corr_status, coverage_status])

    result.interpretation = "; ".join(interpretations)
    result.metrics = metrics

    logger.info(
        f"DSC-DCE Validation: status={result.status}, "
        f"preload_effect_detected={result.preload_effect_detected}, "
        f"median_ratio={metrics.median_ratio:.3f}, "
        f"spearman={metrics.correlation:.3f}, "
        f"pearson={metrics.pearson_correlation:.3f}"
    )

    return result


def validate_vp_cbv_from_files(
    vp_path: Union[str, Path],
    cbv_path: Optional[Union[str, Path]],
    mask_path: Union[str, Path],
    hematocrit: float = HEMATOCRIT_ARTERIAL,
    tissue_density: float = TISSUE_DENSITY,
    constraint_mode: Optional[str] = None,
    dsc_preload_configured: Optional[bool] = None,
    ncbv_path: Optional[Union[str, Path]] = None,
    *,
    aif_tissue_overshoot_fraction: Optional[float] = None,
) -> ValidationResult:
    """Validate DCE vp against DSC CBV from NIfTI files.

    Convenience wrapper for validate_vp_cbv() that loads data from files.

    Args:
        vp_path: Path to DCE vp NIfTI file.
        cbv_path: Path to DSC CBV NIfTI file (absolute/relative mL/100g).
            Only used when ``ncbv_path`` is None; may be None if ``ncbv_path``
            is provided.
        mask_path: Path to mask NIfTI file (brain or tissue mask).
        hematocrit: Hematocrit value for expected ratio calculation.
        tissue_density: Tissue density for expected ratio calculation.
        constraint_mode: CBV constraint mode ("hard", "soft", "skip", or None).
        dsc_preload_configured: Whether DSC preload was configured.
        ncbv_path: Optional path to NAWM-normalized CBV (dimensionless nCBV).
            When provided, validation uses the nCBV path: expected ratio is
            NAWM_VP_REFERENCE and the circular-std threshold is scaled to
            match (CIRCULAR_STD_THRESHOLD_NCBV ≈ 0.002).

    Returns:
        ValidationResult with status, preload detection, and metrics.

    Raises:
        FileNotFoundError: If any required input file does not exist.
        ValueError: If both ``cbv_path`` and ``ncbv_path`` are None.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    vp_path = Path(vp_path)
    mask_path = Path(mask_path)
    use_ncbv_scale = ncbv_path is not None
    if use_ncbv_scale:
        data_path = Path(ncbv_path)  # type: ignore[arg-type]
    elif cbv_path is not None:
        data_path = Path(cbv_path)
    else:
        raise ValueError(
            "validate_vp_cbv_from_files requires either cbv_path or ncbv_path"
        )

    for path, name in [(vp_path, "vp"), (data_path, "CBV/nCBV"), (mask_path, "mask")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    vp_img = nib.load(vp_path)
    cbv_img = nib.load(data_path)
    mask_img = nib.load(mask_path)

    vp_data = np.asarray(vp_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]
    cbv_data = np.asarray(cbv_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]
    mask = np.asarray(mask_img.get_fdata(), dtype=np.bool_)  # type: ignore[attr-defined]

    return validate_vp_cbv(
        vp_data=vp_data,  # type: ignore[arg-type]
        cbv_data=cbv_data,  # type: ignore[arg-type]
        mask=mask,
        hematocrit=hematocrit,
        tissue_density=tissue_density,
        constraint_mode=constraint_mode,
        dsc_preload_configured=dsc_preload_configured,
        use_ncbv_scale=use_ncbv_scale,
        aif_tissue_overshoot_fraction=aif_tissue_overshoot_fraction,
    )
