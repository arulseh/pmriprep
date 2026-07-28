"""DSC-DCE cross-validation utilities.

Post-hoc comparison of DCE-derived vp (plasma volume fraction) with DSC-derived
NAWM-normalized relative CBV (nCBV). The expected ``vp / nCBV`` ratio is
``NAWM_VP_REFERENCE`` by construction. PMRIPrep does not calculate absolute
CBV, so unnormalized relative-CBV maps are not valid inputs here.

For unconstrained fits, the comparison is a coarse cross-modality
plausibility check. When DSC nCBV supplies the vp prior, the same metrics are
descriptive constraint diagnostics and are not independent validation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.qc_standards import QCStatus


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


#: Minimum vp value to include in validation (avoid numerical instability)
MIN_VP_FOR_VALIDATION: float = 0.001

#: Maximum vp for validation (above = likely large vessel partial volume effect)
MAX_VP_FOR_VALIDATION: float = 0.15

#: Spearman rank correlation thresholds for unconstrained vp-nCBV validation.
#: ρ=0.40 → ~16% shared variance, a conventional "moderate" cutoff. DSC and
#: DCE measure related but distinct vascular features. ρ<0.20 is
#: effectively noise.
CORRELATION_PASS_UNCONSTRAINED: float = 0.40
CORRELATION_WARN_UNCONSTRAINED: float = 0.20

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

#: Absolute minimum valid voxels for statistical reliability.
MIN_VALID_VOXELS: int = 500


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ValidationMetrics:
    """Detailed metrics from DSC-DCE cross-validation.

    Attributes
    ----------
    mean_ratio : float
        Mean (vp * 100) / CBV ratio across valid voxels.
    median_ratio : float
        Median ratio across valid voxels.
    std_ratio : float
        Standard deviation of ratio.
    p10_ratio : float
        10th percentile ratio.
    p90_ratio : float
        90th percentile ratio.
    n_valid_voxels : int
        Number of voxels included in validation.
    n_total_voxels : int
        Total number of voxels in mask.
    fraction_valid : float
        Fraction of voxels with valid values.
    correlation : float
        Spearman rank correlation between vp and CBV.
        Robust to outliers and non-linear monotonic relationships,
        which is the expected vp-CBV relationship (different contrast
        mechanisms: T1 vs T2*).
    pearson_correlation : float
        Supplementary Pearson correlation.
        Discrepancy with Spearman indicates outlier leverage inflation.
    mean_vp : float
        Mean vp value in valid voxels (fraction).
    mean_cbv : float
        Mean dimensionless nCBV value in valid voxels.
    expected_ratio : float
        Expected NAWM-anchored vp/nCBV ratio.
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

    Attributes
    ----------
    status : str
        Overall validation status (PASS/WARN/POOR/CONSTRAINT-COUPLED).
    dsc_preload_configured : bool, optional
        True when user explicitly overrode
        auto-detection with a preload class (i.e., ``preload != "auto"``).
        None if unknown.
    constraint_mode : str, optional
        CBV constraint mode used (soft/skip/None).
    constraint_coupled : bool
        True when DSC nCBV supplied the vp prior, so the comparison is
        constraint-coupled rather than independent validation.
    constraint_relative_width : float, optional
        Relative standard deviation of the DSC-informed vp prior.
    metrics : ValidationMetrics
        Detailed validation metrics.
    interpretation : str
        Human-readable interpretation of results.
    """

    status: str = QCStatus.UNKNOWN
    dsc_preload_configured: Optional[bool] = None
    constraint_mode: Optional[str] = None
    constraint_coupled: bool = False
    constraint_relative_width: Optional[float] = None
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "dsc_preload_configured": self.dsc_preload_configured,
            "constraint_mode": self.constraint_mode,
            "constraint_coupled": self.constraint_coupled,
            "constraint_relative_width": self.constraint_relative_width,
            "metrics": self.metrics.to_dict(),
            "interpretation": self.interpretation,
        }


# =============================================================================
# Validation Functions
# =============================================================================


def build_vp_cbv_validation_mask(
    vp_data: npt.NDArray[np.floating[npt.NBitBase]],
    cbv_data: npt.NDArray[np.floating[npt.NBitBase]],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Return the voxel domain used for DSC-DCE cross-validation.

    Parameters
    ----------
    vp_data : numpy.ndarray
        DCE plasma-volume fraction map.
    cbv_data : numpy.ndarray
        DSC NAWM-normalized CBV map.
    mask : numpy.ndarray
        Boolean analysis mask.

    Returns
    -------
    numpy.ndarray
        Boolean mask restricted to finite values within the physiological
        validation bounds.
    """
    from pmriprep.utils.dce_constants import (
        MAX_NCBV_FOR_VALIDATION,
        MIN_NCBV_FOR_VALIDATION,
    )

    if vp_data.shape != cbv_data.shape or vp_data.shape != mask.shape:
        raise ValueError(
            "vp, nCBV, and validation mask shapes must match: "
            f"{vp_data.shape}, {cbv_data.shape}, {mask.shape}"
        )

    return (
        mask.astype(bool)
        & np.isfinite(vp_data)
        & np.isfinite(cbv_data)
        & (vp_data > MIN_VP_FOR_VALIDATION)
        & (vp_data < MAX_VP_FOR_VALIDATION)
        & (cbv_data > MIN_NCBV_FOR_VALIDATION)
        & (cbv_data < MAX_NCBV_FOR_VALIDATION)
    )


def validate_vp_cbv(
    vp_data: npt.NDArray[np.floating[npt.NBitBase]],
    cbv_data: npt.NDArray[np.floating[npt.NBitBase]],
    mask: npt.NDArray[np.bool_],
    constraint_mode: Optional[str] = None,
    dsc_preload_configured: Optional[bool] = None,
    constraint_relative_width: Optional[float] = None,
) -> ValidationResult:
    """Validate DCE vp against DSC NAWM-normalized relative CBV.

    Compares vp (plasma volume fraction from DCE) with CBV (cerebral blood
    volume from DSC) to detect quantification issues or preload effects.

    CBV must be dimensionless nCBV (NAWM=1). Expected ratio is
    ``NAWM_VP_REFERENCE`` by construction.

    Parameters
    ----------
    vp_data : numpy.ndarray
        3D vp map array (fraction, 0-1 scale).
    cbv_data : numpy.ndarray
        3D dimensionless nCBV map array.
    mask : numpy.ndarray
        3D boolean mask for analysis region.
    constraint_mode : str, optional
        CBV constraint mode ("soft", "skip", or None).
    dsc_preload_configured : bool, optional
        Whether DSC preload was configured.
    constraint_relative_width : float, optional
        Relative standard deviation of the DSC-informed vp prior.

    Returns
    -------
    ValidationResult
        ValidationResult with comparison status and metrics.
    """
    from pmriprep.utils.dce_constants import (
        NAWM_VP_REFERENCE,
        MIN_NCBV_FOR_VALIDATION,
        VP_NCBV_RATIO_MAX,
        VP_NCBV_RATIO_MIN,
    )

    result = ValidationResult(
        constraint_mode=constraint_mode,
        dsc_preload_configured=dsc_preload_configured,
        constraint_coupled=constraint_mode == "soft",
        constraint_relative_width=constraint_relative_width,
    )
    metrics = ValidationMetrics()

    expected_ratio = NAWM_VP_REFERENCE
    cbv_min = MIN_NCBV_FOR_VALIDATION
    metrics.expected_ratio = expected_ratio

    # Extract values within mask
    mask_bool = mask.astype(bool)
    vp_masked = vp_data[mask_bool]
    cbv_masked = cbv_data[mask_bool]
    metrics.n_total_voxels = int(np.sum(mask))

    if metrics.n_total_voxels == 0:
        logger.warning("Empty mask, cannot perform validation")
        result.status = QCStatus.POOR
        result.interpretation = "Empty mask - no voxels for validation"
        result.metrics = metrics
        return result

    # Use the shared bounded domain so numerical metrics and QC plots agree.
    valid_mask = build_vp_cbv_validation_mask(
        vp_data, cbv_data, mask_bool
    )[mask_bool]

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

    # Both vp and nCBV are dimensionless.
    cbv_safe = np.maximum(cbv_valid, cbv_min)
    ratios = vp_valid / cbv_safe

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

    # A DSC-informed vp fit is coupled to nCBV regardless of how much the
    # posterior moves away from its prior. Correlation and agreement are
    # therefore descriptive constraint diagnostics, not validation.
    if result.constraint_coupled:
        result.status = QCStatus.CONSTRAINT_COUPLED
        result.metrics = metrics
        result.interpretation = (
            "DSC nCBV supplied the soft spatial prior for vp; DSC-DCE "
            "agreement is constraint-coupled and is not an independent "
            "validation result"
        )
        logger.info(
            "DSC-DCE constraint diagnostic: width=%s, median_ratio=%.4f",
            constraint_relative_width,
            metrics.median_ratio,
        )
        return result

    # Determine status from ratio, correlation, and coverage
    interpretations: list[str] = []

    if not (VP_NCBV_RATIO_MIN <= metrics.median_ratio <= VP_NCBV_RATIO_MAX):
        ratio_status = QCStatus.WARN
        interpretations.append(
            f"Median vp/nCBV ratio ({metrics.median_ratio:.3f}) is outside the "
            f"plausibility interval [{VP_NCBV_RATIO_MIN:.3f}, "
            f"{VP_NCBV_RATIO_MAX:.3f}]"
        )
    else:
        ratio_status = QCStatus.PASS
        interpretations.append(
            f"Median vp/nCBV ratio ({metrics.median_ratio:.3f}) is within the "
            f"plausibility interval [{VP_NCBV_RATIO_MIN:.3f}, "
            f"{VP_NCBV_RATIO_MAX:.3f}]"
        )

    # --- Correlation status ---
    corr_pass = CORRELATION_PASS_UNCONSTRAINED
    corr_warn = CORRELATION_WARN_UNCONSTRAINED

    if metrics.correlation < 0:
        corr_status = QCStatus.POOR
        interpretations.append(
            f"Negative vp-nCBV correlation (rho={metrics.correlation:.3f})"
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
        f"median_ratio={metrics.median_ratio:.3f}, "
        f"spearman={metrics.correlation:.3f}, "
        f"pearson={metrics.pearson_correlation:.3f}"
    )

    return result


def validate_vp_cbv_from_files(
    vp_path: Union[str, Path],
    ncbv_path: Union[str, Path],
    mask_path: Union[str, Path],
    constraint_mode: Optional[str] = None,
    dsc_preload_configured: Optional[bool] = None,
    constraint_relative_width: Optional[float] = None,
) -> ValidationResult:
    """Validate DCE vp against DSC CBV from NIfTI files.

    Convenience wrapper for validate_vp_cbv() that loads data from files.

    Parameters
    ----------
    vp_path : str or pathlib.Path
        Path to DCE vp NIfTI file.
    ncbv_path : str or pathlib.Path
        Path to dimensionless NAWM-normalized DSC CBV.
    mask_path : str or pathlib.Path
        Path to mask NIfTI file (brain or tissue mask).
    constraint_mode : str, optional
        CBV constraint mode ("soft", "skip", or None).
    dsc_preload_configured : bool, optional
        Whether DSC preload was configured.

    Returns
    -------
    ValidationResult
        ValidationResult with comparison status and metrics.

    Raises
    ------
    FileNotFoundError
        If any required input file does not exist.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    vp_path = Path(vp_path)
    data_path = Path(ncbv_path)
    mask_path = Path(mask_path)

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
        constraint_mode=constraint_mode,
        dsc_preload_configured=dsc_preload_configured,
        constraint_relative_width=constraint_relative_width,
    )
