"""Descriptive DSC-DCE vascular concordance utilities.

DCE ``vp`` and DSC ``nCBV`` are each normalized to their median in the same
DCE-space NAWM mask, smoothed with matched physical kernels, and compared over
their jointly finite positive support.  Whole-brain concordance is always
reported; lesion-region concordance is added when a lesion mask supplies enough
jointly valid voxels.

When a retained DCE fit used the DSC-informed ``vp`` prior, the measurements
are prior diagnostics rather than independent cross-modal concordance.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

MIN_VALID_VOXELS: int = 500
CONCORDANCE_SMOOTHING_SIGMA_MM: float = 1.0


@dataclass
class ValidationMetrics:
    """Metrics for one spatial comparison domain."""

    n_valid_voxels: int = 0
    n_total_voxels: int = 0
    fraction_valid: float = 0.0
    correlation: float = 0.0
    pearson_correlation: float = 0.0
    mean_vp: float = 0.0
    mean_cbv: float = 0.0

    def to_dict(self) -> dict[str, Union[float, int]]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "n_valid_voxels": self.n_valid_voxels,
            "n_total_voxels": self.n_total_voxels,
            "fraction_valid": self.fraction_valid,
            "correlation": self.correlation,
            "pearson_correlation": self.pearson_correlation,
            "mean_vp": self.mean_vp,
            "mean_cbv": self.mean_cbv,
        }


@dataclass
class ValidationResult:
    """Result of DSC-DCE vascular concordance measurement."""

    status: str = "UNAVAILABLE"
    dsc_preload_configured: Optional[bool] = None
    prior_informed: bool = False
    prior_relative_sd: Optional[float] = None
    prior_informed_fraction: Optional[float] = None
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    region_metrics: Optional[ValidationMetrics] = None
    region_label: Optional[str] = None
    normalization: dict[str, Any] = field(default_factory=dict)
    smoothing_sigma_mm: float = CONCORDANCE_SMOOTHING_SIGMA_MM
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        out: dict[str, Any] = {
            "status": self.status,
            "dsc_preload_configured": self.dsc_preload_configured,
            "prior_informed": self.prior_informed,
            "prior_relative_sd": self.prior_relative_sd,
            "prior_informed_fraction": self.prior_informed_fraction,
            "metrics": self.metrics.to_dict(),
            "normalization": self.normalization,
            "smoothing_sigma_mm": self.smoothing_sigma_mm,
            "interpretation": self.interpretation,
        }
        if self.region_metrics is not None:
            out["region_metrics"] = self.region_metrics.to_dict()
            out["region_label"] = self.region_label or "Lesion region"
        return out


def build_vp_cbv_validation_mask(
    vp_data: npt.NDArray[Any],
    cbv_data: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Return jointly finite positive support within ``mask``."""
    if vp_data.shape != cbv_data.shape or vp_data.shape != mask.shape:
        raise ValueError(
            "vp, nCBV, and validation mask shapes must match: "
            f"{vp_data.shape}, {cbv_data.shape}, {mask.shape}"
        )
    return np.asarray(
        mask.astype(bool)
        & np.isfinite(vp_data)
        & np.isfinite(cbv_data)
        & (vp_data > 0)
        & (cbv_data > 0),
        dtype=np.bool_,
    )


def support_aware_smooth(
    data: npt.NDArray[np.float64],
    sigma_vox: tuple[float, float, float],
) -> npt.NDArray[np.float64]:
    """Gaussian-smooth finite support without bleeding missing values."""
    from scipy.ndimage import gaussian_filter

    support = np.isfinite(data)
    values = np.where(support, data, 0.0)
    numerator = gaussian_filter(values, sigma=sigma_vox, mode="constant")
    denominator = gaussian_filter(
        support.astype(np.float64), sigma=sigma_vox, mode="constant"
    )
    out = np.full(data.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 1e-6)
    return out


def _domain_metrics(
    vp_data: npt.NDArray[np.float64],
    cbv_data: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> ValidationMetrics:
    """Calculate descriptive concordance metrics in one domain."""
    from scipy.stats import spearmanr

    metrics = ValidationMetrics(n_total_voxels=int(np.count_nonzero(mask)))
    valid = build_vp_cbv_validation_mask(vp_data, cbv_data, mask)
    metrics.n_valid_voxels = int(np.count_nonzero(valid))
    if metrics.n_total_voxels:
        metrics.fraction_valid = metrics.n_valid_voxels / metrics.n_total_voxels
    if metrics.n_valid_voxels < MIN_VALID_VOXELS:
        return metrics

    vp = vp_data[valid]
    cbv = cbv_data[valid]
    metrics.mean_vp = float(np.mean(vp))
    metrics.mean_cbv = float(np.mean(cbv))
    rho = spearmanr(vp, cbv).statistic
    pearson = np.corrcoef(vp, cbv)[0, 1]
    metrics.correlation = float(rho) if np.isfinite(rho) else 0.0
    metrics.pearson_correlation = float(pearson) if np.isfinite(pearson) else 0.0
    return metrics


def validate_vp_cbv(
    vp_data: npt.NDArray[Any],
    cbv_data: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    nawm_mask: npt.NDArray[np.bool_] | None = None,
    region_mask: npt.NDArray[np.bool_] | None = None,
    voxel_sizes_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    prior_informed: bool = False,
    dsc_preload_configured: Optional[bool] = None,
    prior_relative_sd: Optional[float] = None,
    prior_informed_fraction: Optional[float] = None,
) -> ValidationResult:
    """Measure NAWM-normalized DSC-DCE vascular concordance.

    NAWM is used only to establish a subject-specific scale for each map. The
    universal comparison domain is whole brain; a lesion-region comparison is
    emitted only when at least ``MIN_VALID_VOXELS`` jointly valid voxels exist.
    """
    vp = np.asarray(vp_data, dtype=np.float64)
    cbv = np.asarray(cbv_data, dtype=np.float64)
    brain = np.asarray(mask, dtype=np.bool_)
    if nawm_mask is None:
        raise ValueError("A DCE-space NAWM mask is required for DSC-DCE concordance")
    nawm = np.asarray(nawm_mask, dtype=np.bool_)
    if vp.shape != cbv.shape or vp.shape != brain.shape or vp.shape != nawm.shape:
        raise ValueError("vp, nCBV, brain, and NAWM mask shapes must match")

    nawm_valid = nawm & np.isfinite(vp) & np.isfinite(cbv) & (vp > 0) & (cbv > 0)
    n_nawm_valid = int(np.count_nonzero(nawm_valid))
    if n_nawm_valid < MIN_VALID_VOXELS:
        return ValidationResult(
            status="UNAVAILABLE",
            dsc_preload_configured=dsc_preload_configured,
            prior_informed=prior_informed,
            prior_relative_sd=prior_relative_sd,
            prior_informed_fraction=prior_informed_fraction,
            interpretation=(
                "NAWM normalization requires at least "
                f"{MIN_VALID_VOXELS} jointly valid voxels; found {n_nawm_valid}"
            ),
        )
    vp_nawm = float(np.median(vp[nawm_valid]))
    cbv_nawm = float(np.median(cbv[nawm_valid]))
    if not np.isfinite(vp_nawm) or not np.isfinite(cbv_nawm) or min(vp_nawm, cbv_nawm) <= 0:
        return ValidationResult(
            status="UNAVAILABLE",
            dsc_preload_configured=dsc_preload_configured,
            prior_informed=prior_informed,
            prior_relative_sd=prior_relative_sd,
            prior_informed_fraction=prior_informed_fraction,
            interpretation="NAWM normalization medians are not finite and positive",
        )

    vp_relative = vp / vp_nawm
    cbv_relative = cbv / cbv_nawm
    sigma_vox = (
        CONCORDANCE_SMOOTHING_SIGMA_MM / max(float(voxel_sizes_mm[0]), 1e-6),
        CONCORDANCE_SMOOTHING_SIGMA_MM / max(float(voxel_sizes_mm[1]), 1e-6),
        CONCORDANCE_SMOOTHING_SIGMA_MM / max(float(voxel_sizes_mm[2]), 1e-6),
    )
    vp_smooth = support_aware_smooth(vp_relative, sigma_vox)
    cbv_smooth = support_aware_smooth(cbv_relative, sigma_vox)

    result = ValidationResult(
        status="PRIOR_INFORMED" if prior_informed else "INDEPENDENT",
        dsc_preload_configured=dsc_preload_configured,
        prior_informed=prior_informed,
        prior_relative_sd=prior_relative_sd,
        prior_informed_fraction=prior_informed_fraction,
        metrics=_domain_metrics(vp_smooth, cbv_smooth, brain),
        normalization={
            "method": "within_subject_nawm_median",
            "nawm_valid_voxels": int(np.count_nonzero(nawm_valid)),
            "dce_vp_nawm_median": vp_nawm,
            "dsc_ncbv_nawm_median": cbv_nawm,
        },
    )
    if result.metrics.n_valid_voxels < MIN_VALID_VOXELS:
        result.status = "UNAVAILABLE"
        result.interpretation = "Insufficient jointly valid whole-brain voxels"
        return result

    if region_mask is not None:
        region = np.asarray(region_mask, dtype=np.bool_)
        if region.shape != vp.shape:
            raise ValueError("Lesion-region mask geometry must match the DCE vp map")
        region_metrics = _domain_metrics(vp_smooth, cbv_smooth, region)
        if region_metrics.n_valid_voxels >= MIN_VALID_VOXELS:
            result.region_metrics = region_metrics
            result.region_label = "Lesion region"

    result.interpretation = (
        "DSC nCBV supplied the vp prior used by retained Bayesian fits; the "
        "comparison is not independent"
        if prior_informed
        else "Independent descriptive DSC-DCE concordance after within-subject "
        "NAWM normalization and matched 1 mm Gaussian smoothing"
    )
    return result


def validate_vp_cbv_from_files(
    vp_path: Union[str, Path],
    ncbv_path: Union[str, Path],
    mask_path: Union[str, Path],
    *,
    nawm_mask_path: Union[str, Path],
    region_mask_path: Union[str, Path, None] = None,
    prior_informed: bool = False,
    dsc_preload_configured: Optional[bool] = None,
    prior_relative_sd: Optional[float] = None,
    prior_informed_fraction: Optional[float] = None,
) -> ValidationResult:
    """Load NIfTI inputs and measure DSC-DCE vascular concordance."""
    import nibabel as nib  # type: ignore[import-untyped]

    paths = [Path(vp_path), Path(ncbv_path), Path(mask_path), Path(nawm_mask_path)]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"DSC-DCE concordance input not found: {path}")
    vp_img = nib.load(paths[0])
    vp = np.asarray(vp_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]
    cbv = np.asarray(nib.load(paths[1]).get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]
    brain = np.asarray(nib.load(paths[2]).get_fdata(), dtype=np.bool_)  # type: ignore[attr-defined]
    nawm = np.asarray(nib.load(paths[3]).get_fdata(), dtype=np.bool_)  # type: ignore[attr-defined]
    region = None
    if region_mask_path and Path(region_mask_path).is_file():
        region = np.asarray(
            nib.load(region_mask_path).get_fdata(), dtype=np.bool_  # type: ignore[attr-defined]
        )
    zooms = vp_img.header.get_zooms()  # type: ignore[attr-defined]
    return validate_vp_cbv(
        vp, cbv, brain,
        nawm_mask=nawm,
        region_mask=region,
        voxel_sizes_mm=tuple(float(x) for x in zooms[:3]),  # type: ignore[arg-type]
        prior_informed=prior_informed,
        dsc_preload_configured=dsc_preload_configured,
        prior_relative_sd=prior_relative_sd,
        prior_informed_fraction=prior_informed_fraction,
    )
