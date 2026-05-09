"""QC standards and colorblind-safe visualization palette.

Standardized quality control thresholds and colorblind-safe color palette
for consistent QC assessment and visualization across the pipeline.

References:
    - Paul Tol's color schemes: https://personal.sron.nl/~pault/
    - Motion QC thresholds based on Power et al. (2012)
    - Physiological T1 values from Stanisz et al. (2005)
"""

from typing import Any, Final, NamedTuple


# =============================================================================
# Paul Tol Colorblind-Safe Palette
# =============================================================================
# Based on Paul Tol's accessible color schemes for scientific visualization
# These colors are distinguishable by most people with color vision deficiencies


class ColorPalette(NamedTuple):
    """Colorblind-safe color palette for visualizations.

    Colors are hex values suitable for matplotlib and other plotting libraries.
    """
    primary: str        # Main/default color
    secondary: str      # Secondary/alternate color
    success: str        # Pass/good status
    warning: str        # Warning/marginal status
    error: str          # Fail/bad status


#: Paul Tol's bright qualitative scheme with semantic mapping
#: - Primary: blue (accessible, professional)
#: - Secondary: cyan (contrast with primary)
#: - Success: green (universal "good" association)
#: - Warning: yellow/orange (caution)
#: - Error: red (universal "problem" association)
COLORBLIND_PALETTE: Final[ColorPalette] = ColorPalette(
    primary="#4477AA",      # Tol bright blue
    secondary="#66CCEE",    # Tol bright cyan
    success="#228833",      # Tol bright green
    warning="#CCBB44",      # Tol bright yellow
    error="#EE6677",        # Tol bright red (muted for accessibility)
)


#: Extended palette for multi-class visualizations (up to 7 colors)
#: All colors from Paul Tol's bright qualitative scheme
EXTENDED_PALETTE: Final[tuple[str, ...]] = (
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Purple
    "#BBBBBB",  # Grey
)


#: Sequential colormap endpoints for heatmaps (blue to red via white)
#: Safe for most color vision deficiencies
SEQUENTIAL_COLORMAP: Final[tuple[str, str, str]] = (
    "#364B9A",  # Dark blue (low values)
    "#FFFFFF",  # White (middle)
    "#A50026",  # Dark red (high values)
)


#: Diverging colormap for centered data (e.g., correlation, difference maps)
DIVERGING_COLORMAP: Final[tuple[str, str, str]] = (
    "#4575B4",  # Blue (negative)
    "#FFFFBF",  # Light yellow (zero)
    "#D73027",  # Red (positive)
)


# =============================================================================
# Motion QC Thresholds
# =============================================================================
# Based on Power et al. (2012) framewise displacement thresholds
# FD = sum(|d_i - d_{i-1}|) for 6 motion parameters, with rotation scaled by 50mm


class QCThresholds(NamedTuple):
    """QC thresholds for pass/warn/poor classification."""
    pass_threshold: float   # Values below this: PASS
    warn_threshold: float   # Values between pass and warn: WARN; above warn: POOR


#: Framewise displacement (FD) thresholds in mm
#: Pass: FD < 0.3mm (minimal motion)
#: Warn: 0.3mm <= FD < 0.5mm (moderate motion)
#: Poor: FD >= 0.5mm (excessive motion)
MOTION_FD_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.3,
    warn_threshold=0.5,
)

#: RMS displacement thresholds in mm
#: Slightly more lenient than FD as RMS is less sensitive to transients
MOTION_RMS_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.5,
    warn_threshold=1.0,
)

#: Maximum fraction of outlier frames (FD > 0.5mm) allowed
MOTION_OUTLIER_FRACTION_MAX: Final[float] = 0.20

#: Percent outlier frames thresholds
#: Pass: <5% outlier frames (minimal motion artifacts)
#: Warn: 5-20% (some motion-corrupted volumes)
#: Poor: >=20% (excessive motion corruption)
#: References: Power et al. (2012), Satterthwaite et al. (2013)
MOTION_OUTLIER_PCT_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=5.0,
    warn_threshold=20.0,
)

#: Maximum framewise displacement thresholds in mm
#: Pass: max FD <1mm (no large motion spikes)
#: Warn: 1-2mm (transient spike, may affect bolus tracking)
#: Poor: >=2mm (large spike, likely corrupts perfusion estimates)
#: References: Power et al. (2012), Boxerman et al. (2020)
MOTION_FD_MAX_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=1.0,
    warn_threshold=2.0,
)


# =============================================================================
# DICE Coefficient Thresholds
# =============================================================================
# For mask overlap QC (registration, segmentation)


#: DICE coefficient thresholds for mask comparison
#: Pass: DICE > 0.85 (excellent overlap)
#: Warn: 0.75 < DICE <= 0.85 (acceptable overlap)
#: Poor: DICE <= 0.75 (poor overlap)
DICE_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.85,
    warn_threshold=0.75,
)

#: Stricter DICE thresholds for critical masks (e.g., brain mask)
DICE_THRESHOLDS_STRICT: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.90,
    warn_threshold=0.85,
)


# =============================================================================
# Correlation Thresholds
# =============================================================================
# For image similarity assessment (registration QC, reproducibility)


#: Correlation coefficient thresholds for image comparison
#: Pass: r > 0.7 (strong correlation)
#: Warn: 0.5 < r <= 0.7 (moderate correlation)
#: Poor: r <= 0.5 (weak correlation)
CORRELATION_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.7,
    warn_threshold=0.5,
)

#: Normalized mutual information (NMI) thresholds for multi-modal registration
NMI_THRESHOLDS: Final[QCThresholds] = QCThresholds(
    pass_threshold=0.3,     # NMI typically lower than correlation
    warn_threshold=0.2,
)


# =============================================================================
# Physiological Ranges at 3T
# =============================================================================
# Expected T1 values for QC validation at 3.0 Tesla
# Based on Stanisz et al. (2005), Wright et al. (2008)


# =============================================================================
# QC Status Enumeration Helper
# =============================================================================


class QCStatus:
    """QC status strings for consistent reporting."""
    PASS: Final[str] = "PASS"
    WARN: Final[str] = "WARN"
    POOR: Final[str] = "POOR"
    UNKNOWN: Final[str] = "UNKNOWN"
    CIRCULAR: Final[str] = "CIRCULAR"

    #: Severity ordering for worst-of comparisons.
    _SEVERITY: Final[dict[str, int]] = {
        "PASS": 0, "WARN": 1, "POOR": 2,
    }

    @classmethod
    def worst(cls, statuses: list[str]) -> str:
        """Return the most severe status from a list.

        Args:
            statuses: List of QC status strings (PASS, WARN, POOR).

        Returns:
            The worst (highest severity) status.

        Raises:
            KeyError: If any status is not in PASS/WARN/POOR.
        """
        return max(statuses, key=lambda s: cls._SEVERITY[s])


def get_qc_status(value: float, thresholds: QCThresholds, higher_is_better: bool = True) -> str:
    """Determine QC status based on value and thresholds.

    Args:
        value: The metric value to evaluate
        thresholds: QCThresholds with pass/warn thresholds
        higher_is_better: If True, values above pass_threshold are PASS.
                         If False, values below pass_threshold are PASS.

    Returns:
        QC status string: "PASS", "WARN", or "POOR"
    """
    if higher_is_better:
        # For metrics like DICE, correlation where higher is better
        if value > thresholds.pass_threshold:
            return QCStatus.PASS
        elif value > thresholds.warn_threshold:
            return QCStatus.WARN
        else:
            return QCStatus.POOR
    else:
        # For metrics like motion FD where lower is better
        if value < thresholds.pass_threshold:
            return QCStatus.PASS
        elif value < thresholds.warn_threshold:
            return QCStatus.WARN
        else:
            return QCStatus.POOR


def get_composite_motion_status(
    fd_mean: float,
    fd_max: float,
    pct_outliers: float,
) -> str:
    """Determine composite motion QC status from multiple criteria.

    Uses the worst (most severe) status across mean FD, max FD, and
    percent outlier frames. Mean FD alone is insufficient for perfusion
    MRI where even single large spikes can corrupt bolus-tracking.

    Args:
        fd_mean: Mean framewise displacement in mm (excluding first frame).
        fd_max: Maximum framewise displacement in mm.
        pct_outliers: Percentage of frames exceeding the motion threshold.

    Returns:
        QC status string: "PASS", "WARN", or "POOR".

    References:
        Power et al. (2012). doi:10.1016/j.neuroimage.2011.10.018
        Satterthwaite et al. (2013). doi:10.1016/j.neuroimage.2012.08.052
        Boxerman et al. (2020). doi:10.3174/ajnr.A6365
    """
    status_mean = get_qc_status(
        fd_mean, MOTION_FD_THRESHOLDS, higher_is_better=False,
    )
    status_max = get_qc_status(
        fd_max, MOTION_FD_MAX_THRESHOLDS, higher_is_better=False,
    )
    status_pct = get_qc_status(
        pct_outliers, MOTION_OUTLIER_PCT_THRESHOLDS, higher_is_better=False,
    )

    return QCStatus.worst([status_mean, status_max, status_pct])


def get_status_color(status: str) -> str:
    """Get the appropriate color for a QC status.

    Args:
        status: QC status string ("PASS", "WARN", "POOR", or "UNKNOWN")

    Returns:
        Hex color code from the colorblind-safe palette
    """
    status_colors = {
        QCStatus.PASS: COLORBLIND_PALETTE.success,
        QCStatus.WARN: COLORBLIND_PALETTE.warning,
        QCStatus.POOR: COLORBLIND_PALETTE.error,
        QCStatus.UNKNOWN: COLORBLIND_PALETTE.secondary,
        QCStatus.CIRCULAR: COLORBLIND_PALETTE.secondary,
    }
    return status_colors.get(status, COLORBLIND_PALETTE.secondary)


# =============================================================================
# QC Exclusion Recommendations (NiPreps-aligned)
# =============================================================================


class QCRecommendation(NamedTuple):
    """QC recommendation for data exclusion or review.

    Following NiPreps conventions for providing actionable QC feedback.

    Attributes:
        action: Recommended action ("include", "review", "exclude").
        reason: Brief explanation of the recommendation.
        severity: Severity level (0=none, 1=minor, 2=moderate, 3=severe).
    """
    action: str
    reason: str
    severity: int


#: Action constants for QC recommendations
class QCAction:
    """QC action strings for consistent recommendations."""
    INCLUDE: Final[str] = "include"
    REVIEW: Final[str] = "review"
    EXCLUDE: Final[str] = "exclude"


def get_qc_recommendation(
    status: str,
    metric_name: str,
    value: float,
    thresholds: QCThresholds,
    higher_is_better: bool = True,
) -> QCRecommendation:
    """Generate a QC recommendation based on metric value and status.

    Follows NiPreps convention of providing actionable exclusion recommendations
    rather than just pass/warn/fail status.

    Args:
        status: QC status string from get_qc_status().
        metric_name: Human-readable name of the metric (e.g., "FD mean").
        value: The metric value.
        thresholds: QCThresholds used for evaluation.
        higher_is_better: Whether higher values indicate better quality.

    Returns:
        QCRecommendation with action, reason, and severity.
    """
    if status == QCStatus.PASS:
        return QCRecommendation(
            action=QCAction.INCLUDE,
            reason=f"{metric_name} within acceptable range ({value:.3f})",
            severity=0,
        )
    elif status == QCStatus.WARN:
        return QCRecommendation(
            action=QCAction.REVIEW,
            reason=f"{metric_name} marginal ({value:.3f}); visual inspection recommended",
            severity=1,
        )
    else:  # POOR
        if higher_is_better:
            threshold_desc = f"below {thresholds.warn_threshold}"
        else:
            threshold_desc = f"above {thresholds.warn_threshold}"
        return QCRecommendation(
            action=QCAction.EXCLUDE,
            reason=f"{metric_name} {threshold_desc} ({value:.3f}); consider exclusion",
            severity=3,
        )


def aggregate_qc_recommendations(
    recommendations: list[QCRecommendation],
) -> QCRecommendation:
    """Aggregate multiple QC recommendations into a single overall recommendation.

    Uses the most severe recommendation as the overall action.

    Args:
        recommendations: List of QCRecommendation objects.

    Returns:
        Aggregated QCRecommendation with combined reasons.
    """
    if not recommendations:
        return QCRecommendation(
            action=QCAction.INCLUDE,
            reason="No QC metrics evaluated",
            severity=0,
        )

    # Find the most severe recommendation
    max_severity = max(r.severity for r in recommendations)

    # Determine overall action based on max severity
    if max_severity >= 3:
        action = QCAction.EXCLUDE
    elif max_severity >= 1:
        action = QCAction.REVIEW
    else:
        action = QCAction.INCLUDE

    # Collect reasons from all non-passing recommendations
    reasons = [r.reason for r in recommendations if r.severity > 0]
    if not reasons:
        combined_reason = "All QC metrics passed"
    else:
        combined_reason = "; ".join(reasons)

    return QCRecommendation(
        action=action,
        reason=combined_reason,
        severity=max_severity,
    )


def build_qc_summary(
    metrics: dict[str, tuple[float, str]],
    overall_status: str,
    overall_recommendation: QCRecommendation,
) -> dict[str, Any]:
    """Build a NiPreps-style QC summary dictionary.

    Creates a structured QC summary that includes both detailed metrics
    and actionable recommendations, following MRIQC/fMRIPrep conventions.

    Args:
        metrics: Dictionary mapping metric names to (value, status) tuples.
        overall_status: Overall QC status (PASS/WARN/POOR).
        overall_recommendation: Aggregated QC recommendation.

    Returns:
        Dictionary suitable for JSON serialization with:
        - overall_status: The aggregate status
        - recommendation: Action recommendation with reason
        - metrics: Per-metric details with values and status
    """
    return {
        "overall_status": overall_status,
        "recommendation": {
            "action": overall_recommendation.action,
            "reason": overall_recommendation.reason,
            "severity": overall_recommendation.severity,
        },
        "metrics": {
            name: {"value": value, "status": status}
            for name, (value, status) in metrics.items()
        },
    }


# ---------------------------------------------------------------------------
# Per-pipeline recommendation enrichers (lifted from bespoke writers)
# ---------------------------------------------------------------------------


def recommend_dsc_qc(qc_metrics: dict[str, Any]) -> dict[str, Any]:
    """Add a NiPreps-style overall ``recommendation`` block to DSC QC metrics.

    Mirrors the legacy ``DSCDerivativeWriter._add_qc_recommendation`` —
    evaluates AIF detection, motion (composite ``motion_status``), and CNR
    (with a Rose-criterion floor on mean CNR), aggregates per-axis severity,
    and returns ``qc_metrics`` augmented with ``recommendation``.
    """
    from pmriprep.utils.dsc_constants import QC_MEAN_CNR_MINIMUM

    recommendations: list[QCRecommendation] = []
    enhanced = dict(qc_metrics)

    # AIF detection quality.  ``detection_method`` may be a "+"-joined
    # union (e.g. "atlas_relaxed+atlas_permissive") when the union-pool
    # fallback accepts contributions from multiple tiers.  Apply
    # worst-tier-wins semantics: any contributing tier downgrades the
    # recommendation.
    aif_qc = qc_metrics.get("aif_detection", {})
    detection_level = aif_qc.get("detection_method", "unknown")
    parts = (
        str(detection_level).split("+")
        if isinstance(detection_level, str) and detection_level
        else ["unknown"]
    )
    parts_set = set(parts)
    _KNOWN_TIERS = {
        "atlas_strict", "atlas_relaxed", "atlas_permissive",
        "full_atlas_permissive", "brain_relaxed", "brain_permissive",
        "external", "synthetic", "none", "unknown",
    }
    _unknown_parts = parts_set - _KNOWN_TIERS
    if parts_set & {"synthetic", "none", "unknown"} or _unknown_parts:
        # Unknown parts are treated as ``unknown`` semantically — no way
        # to vouch for an AIF tier we don't recognise.
        aif_rec = QCRecommendation(
            QCAction.EXCLUDE,
            f"AIF detection failed or synthetic ({detection_level})", 3,
        )
    elif "external" in parts_set:
        aif_rec = QCRecommendation(
            QCAction.REVIEW,
            "External AIF provided; verify curve quality", 1,
        )
    elif parts_set & {
        "atlas_permissive", "full_atlas_permissive",
        "brain_relaxed", "brain_permissive",
    }:
        aif_rec = QCRecommendation(
            QCAction.REVIEW,
            f"AIF detection marginal ({detection_level})", 1,
        )
    elif parts_set <= {"atlas_strict", "atlas_relaxed"}:
        aif_rec = QCRecommendation(
            QCAction.INCLUDE,
            f"AIF detected ({detection_level})", 0,
        )
    else:
        aif_rec = QCRecommendation(
            QCAction.REVIEW,
            f"AIF detection: {detection_level}", 1,
        )
    recommendations.append(aif_rec)

    # Motion via composite status
    preprocessing = qc_metrics.get("preprocessing", {})
    motion_status = preprocessing.get("motion_status")
    if motion_status is not None:
        fd_mean = preprocessing.get("fd_mean")
        fd_max = preprocessing.get("fd_max")
        pct_outliers = preprocessing.get("percent_outliers")
        parts = []
        if fd_mean is not None:
            parts.append(f"mean FD={fd_mean:.2f}mm")
        if fd_max is not None:
            parts.append(f"max FD={fd_max:.2f}mm")
        if pct_outliers is not None:
            parts.append(f"outliers={pct_outliers:.1f}%")
        detail = f"({', '.join(parts)})" if parts else ""

        if motion_status == QCStatus.PASS:
            motion_rec = QCRecommendation(QCAction.INCLUDE, f"Motion acceptable {detail}", 0)
        elif motion_status == QCStatus.WARN:
            motion_rec = QCRecommendation(QCAction.REVIEW, f"Motion marginal {detail}", 1)
        elif motion_status == QCStatus.POOR:
            motion_rec = QCRecommendation(QCAction.REVIEW, f"Excessive motion {detail}", 2)
        else:
            motion_rec = None
        if motion_rec is not None:
            recommendations.append(motion_rec)

    # CNR (Rose-criterion floor + fraction-based grading)
    cnr_fraction = preprocessing.get("fraction_cnr_gt_4")
    mean_cnr = preprocessing.get("mean_cnr")
    if mean_cnr is not None and mean_cnr < QC_MEAN_CNR_MINIMUM:
        cnr_rec = QCRecommendation(
            QCAction.EXCLUDE,
            f"Mean CNR critically low ({mean_cnr:.1f} < {QC_MEAN_CNR_MINIMUM}, possible failed injection)",
            3,
        )
        recommendations.append(cnr_rec)
    elif cnr_fraction is not None:
        if cnr_fraction >= 0.84:
            cnr_rec = QCRecommendation(QCAction.INCLUDE, f"CNR excellent ({cnr_fraction:.0%} >4)", 0)
        elif cnr_fraction >= 0.65:
            cnr_rec = QCRecommendation(QCAction.REVIEW, f"CNR marginal ({cnr_fraction:.0%} >4)", 1)
        else:
            cnr_rec = QCRecommendation(QCAction.EXCLUDE, f"CNR poor ({cnr_fraction:.0%} >4)", 3)
        recommendations.append(cnr_rec)

    overall = aggregate_qc_recommendations(recommendations)
    enhanced["recommendation"] = {
        "action": overall.action,
        "reason": overall.reason,
        "severity": overall.severity,
    }
    return enhanced


def recommend_dce_qc(qc_metrics: dict[str, Any]) -> dict[str, Any]:
    """Add a NiPreps-style overall ``recommendation`` block to DCE QC metrics.

    Mirrors the legacy ``DCEDerivativeWriter._add_qc_recommendation`` —
    evaluates AIF quality_score, PK fit residual, Ktrans plausibility,
    motion (composite ``motion_status``), peak enhancement + baseline
    tSNR (with ``QC_*`` constants from ``dce_constants``), B1+ bias,
    and DSC-DCE cross-validation. Aggregates per-axis severity and
    returns ``qc_metrics`` augmented with ``recommendation``.
    """
    from pmriprep.utils.dce_constants import (
        QC_B1_MEAN_HIGH,
        QC_B1_MEAN_LOW,
        QC_DCE_BASELINE_TSNR_MIN,
        QC_DCE_BASELINE_TSNR_WARN,
        QC_PEAK_ENHANCEMENT_MIN,
        QC_PEAK_ENHANCEMENT_WARN,
    )

    recommendations: list[QCRecommendation] = []
    enhanced = dict(qc_metrics)

    # AIF quality
    aif_qc = qc_metrics.get("aif_detection", {})
    quality_score = aif_qc.get("quality_score")
    if quality_score is not None:
        if quality_score >= 0.7:
            aif_rec = QCRecommendation(QCAction.INCLUDE, f"AIF quality good ({quality_score:.2f})", 0)
        elif quality_score >= 0.4:
            aif_rec = QCRecommendation(QCAction.REVIEW, f"AIF quality marginal ({quality_score:.2f})", 1)
        else:
            aif_rec = QCRecommendation(QCAction.EXCLUDE, f"AIF quality poor ({quality_score:.2f})", 3)
        recommendations.append(aif_rec)

    # PK model fit
    pk_qc = qc_metrics.get("pk_modeling", {})
    fit_residual = pk_qc.get("median_residual")
    if fit_residual is not None:
        if fit_residual < 0.1:
            fit_rec = QCRecommendation(QCAction.INCLUDE, f"PK fit residual acceptable ({fit_residual:.3f})", 0)
        elif fit_residual < 0.2:
            fit_rec = QCRecommendation(QCAction.REVIEW, f"PK fit residual marginal ({fit_residual:.3f})", 1)
        else:
            fit_rec = QCRecommendation(QCAction.EXCLUDE, f"PK fit residual high ({fit_residual:.3f})", 2)
        recommendations.append(fit_rec)

    # Ktrans plausibility
    ktrans_mean = pk_qc.get("ktrans_mean")
    if ktrans_mean is not None:
        if 0.0001 < ktrans_mean < 1.0:
            kt_rec = QCRecommendation(QCAction.INCLUDE, f"Ktrans physiologically plausible (mean={ktrans_mean:.4f})", 0)
        else:
            kt_rec = QCRecommendation(QCAction.REVIEW, f"Ktrans outside typical range (mean={ktrans_mean:.4f})", 1)
        recommendations.append(kt_rec)

    # Motion via composite status
    preprocessing_qc = qc_metrics.get("preprocessing", {})
    motion_status = preprocessing_qc.get("motion_status")
    if motion_status is not None:
        fd_mean = preprocessing_qc.get("fd_mean")
        fd_max = preprocessing_qc.get("fd_max")
        pct_outliers = preprocessing_qc.get("percent_outliers")
        parts = []
        if fd_mean is not None:
            parts.append(f"mean FD={fd_mean:.2f}mm")
        if fd_max is not None:
            parts.append(f"max FD={fd_max:.2f}mm")
        if pct_outliers is not None:
            parts.append(f"outliers={pct_outliers:.1f}%")
        detail = f"({', '.join(parts)})" if parts else ""
        if motion_status == QCStatus.PASS:
            motion_rec = QCRecommendation(QCAction.INCLUDE, f"Motion acceptable {detail}", 0)
        elif motion_status == QCStatus.WARN:
            motion_rec = QCRecommendation(QCAction.REVIEW, f"Motion marginal {detail}", 1)
        elif motion_status == QCStatus.POOR:
            motion_rec = QCRecommendation(QCAction.REVIEW, f"Excessive motion {detail}", 2)
        else:
            motion_rec = None
        if motion_rec is not None:
            recommendations.append(motion_rec)

    # Peak enhancement + baseline tSNR
    peak_enh = preprocessing_qc.get("peak_enh_pct")
    baseline_tsnr = preprocessing_qc.get("baseline_tsnr")
    aif_source = aif_qc.get("aif_source", "")
    if peak_enh is not None:
        if peak_enh < QC_PEAK_ENHANCEMENT_MIN:
            recommendations.append(QCRecommendation(
                QCAction.EXCLUDE,
                f"Peak enhancement critically low ({peak_enh:.1f}% < {QC_PEAK_ENHANCEMENT_MIN}%, possible failed injection)",
                3,
            ))
        elif peak_enh < QC_PEAK_ENHANCEMENT_WARN:
            recommendations.append(QCRecommendation(
                QCAction.REVIEW,
                f"Peak enhancement marginal ({peak_enh:.1f}%)",
                1,
            ))
            if aif_source == "population":
                recommendations.append(QCRecommendation(
                    QCAction.REVIEW,
                    "Injection quality cannot be verified with population AIF",
                    1,
                ))
    if baseline_tsnr is not None:
        if baseline_tsnr < QC_DCE_BASELINE_TSNR_MIN:
            recommendations.append(QCRecommendation(
                QCAction.EXCLUDE,
                f"Baseline tSNR critically low ({baseline_tsnr:.1f} < {QC_DCE_BASELINE_TSNR_MIN})",
                3,
            ))
        elif baseline_tsnr < QC_DCE_BASELINE_TSNR_WARN:
            recommendations.append(QCRecommendation(
                QCAction.REVIEW,
                f"Baseline tSNR marginal ({baseline_tsnr:.1f})",
                1,
            ))

    # B1+ bias
    b1_qc = qc_metrics.get("b1_mapping", {})
    mean_b1 = b1_qc.get("mean_b1")
    if mean_b1 is not None:
        if mean_b1 < QC_B1_MEAN_LOW or mean_b1 > QC_B1_MEAN_HIGH:
            recommendations.append(QCRecommendation(
                QCAction.REVIEW,
                f"B1+ bias outside expected range (mean={mean_b1:.2f}, expected {QC_B1_MEAN_LOW}-{QC_B1_MEAN_HIGH})",
                1,
            ))

    # DSC-DCE cross-validation
    xval = qc_metrics.get("dsc_dce_validation", {})
    if not xval.get("skipped", False) and not xval.get("circular", False):
        xval_status = xval.get("status")
        xval_metrics = xval.get("metrics", {})
        corr = xval_metrics.get("correlation", 0.0)
        if xval_status == QCStatus.WARN:
            preload = xval.get("preload_effect_detected", False)
            detail = "low vp/CBV ratio" if preload else "ratio outside range"
            recommendations.append(QCRecommendation(
                QCAction.REVIEW, f"DSC-DCE: {detail} (r={corr:.2f})", 1,
            ))
        elif xval_status == QCStatus.PASS:
            recommendations.append(QCRecommendation(
                QCAction.INCLUDE, f"DSC-DCE validation passed (r={corr:.2f})", 0,
            ))

    overall = aggregate_qc_recommendations(recommendations)
    enhanced["recommendation"] = {
        "action": overall.action,
        "reason": overall.reason,
        "severity": overall.severity,
    }
    return enhanced
