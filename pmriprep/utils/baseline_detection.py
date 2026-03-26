"""Ensemble baseline detection for DSC and DCE perfusion MRI.

Combines five orthogonal bolus-arrival detectors into a robust consensus
estimate of the pre-bolus baseline length.  Each detector targets a
different failure mode:

    A. Amplitude z-score — clean non-preloaded data
    B. Gradient z-score — preloaded/elevated baselines
    C. CUSUM — gradual onset (cumulative evidence)
    D. Variance ratio — detects variance explosion at bolus
    E. Max-slope backtrack — anchored to peak, immune to early noise

All detectors are scale-invariant, so the same function works on DSC
ΔR2* concentration (0–50 s⁻¹) and DCE raw signal (100–2000 AU).
"""

import logging
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual detectors — each returns Optional[int] (arrival frame or None)
# ---------------------------------------------------------------------------


def _detect_amplitude_zscore(
    signal: npt.NDArray[np.floating],
    baseline_window: int,
    n_consecutive: int,
) -> Optional[int]:
    """Detector A: amplitude z-score (existing method)."""
    from pmriprep.utils.dsc_signal import detect_bolus_arrival

    try:
        arrival_idx, _ = detect_bolus_arrival(
            signal,
            baseline_frames=baseline_window,
            n_consecutive=n_consecutive,
        )
        return int(arrival_idx)
    except Exception:
        return None


def _detect_gradient_zscore(
    signal: npt.NDArray[np.floating],
    baseline_window: int,
    n_consecutive: int,
) -> Optional[int]:
    """Detector B: gradient z-score (existing method)."""
    from pmriprep.utils.dsc_signal import detect_bolus_arrival_gradient

    try:
        arrival_idx, _ = detect_bolus_arrival_gradient(
            signal,
            baseline_window=baseline_window,
            n_consecutive=n_consecutive,
        )
        return int(arrival_idx)
    except Exception:
        return None


def _detect_cusum(
    signal: npt.NDArray[np.floating],
    baseline_window: int,
    threshold: float = 5.0,
) -> Optional[int]:
    """Detector C: cumulative sum (CUSUM) change-point detection.

    Computes the cumulative sum of z-scored deviations from the baseline
    mean.  Triggers when the cumsum exceeds ``threshold``.  Scale-invariant
    because deviations are normalized by baseline std.
    """
    try:
        n = len(signal)
        bl = min(baseline_window, n // 4)
        if bl < 3:
            return None

        bl_mean = float(np.mean(signal[:bl]))
        bl_std = float(np.std(signal[:bl]))
        bl_std = max(bl_std, EPSILON_CONCENTRATION)

        z = (signal - bl_mean) / bl_std
        cumsum = np.cumsum(z[bl:])

        above = np.where(cumsum > threshold)[0]
        if len(above) > 0:
            return int(above[0]) + bl
        return None
    except Exception:
        return None


def _detect_variance_ratio(
    signal: npt.NDArray[np.floating],
    baseline_window: int,
    window_size: int = 5,
    threshold: float = 3.0,
) -> Optional[int]:
    """Detector D: sliding-window variance ratio.

    Compares variance in a sliding window to baseline variance.  Triggers
    when the ratio exceeds ``threshold``.  Scale-invariant because the
    ratio cancels absolute scale.
    """
    try:
        n = len(signal)
        bl = min(baseline_window, n // 4)
        if bl < 3:
            return None

        bl_var = float(np.var(signal[:bl]))
        bl_var = max(bl_var, EPSILON_CONCENTRATION ** 2)

        ws = min(window_size, max(2, bl // 2))

        for i in range(bl, n - ws + 1):
            win_var = float(np.var(signal[i : i + ws]))
            if win_var / bl_var > threshold:
                return i
        return None
    except Exception:
        return None


def _detect_max_slope_backtrack(
    signal: npt.NDArray[np.floating],
    min_baseline: int = 3,
    backtrack_fraction: float = 0.1,
) -> Optional[int]:
    """Detector E: max-slope backtrack.

    Finds the frame with the largest derivative, then walks backward to
    where the derivative first drops below ``backtrack_fraction`` of the
    maximum.  Anchored to the bolus peak, immune to early noise.
    Scale-invariant because the fraction is relative.
    """
    try:
        n = len(signal)
        if n < 4:
            return None

        deriv = np.diff(signal)
        if np.max(deriv) <= 0:
            return None

        peak_idx = int(np.argmax(deriv))
        max_deriv = float(deriv[peak_idx])
        backtrack_thresh = backtrack_fraction * max_deriv

        # Walk backward from peak
        arrival = peak_idx
        for i in range(peak_idx - 1, min_baseline - 1, -1):
            if deriv[i] < backtrack_thresh:
                arrival = i + 1
                break
        else:
            arrival = min_baseline

        return max(arrival, min_baseline)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Weighted median helper
# ---------------------------------------------------------------------------


def _weighted_median(values: list[int], weights: list[float]) -> int:
    """Compute weighted median of integer values."""
    if not values:
        return 0
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    cumulative = 0.0
    for val, w in pairs:
        cumulative += w
        if cumulative >= total / 2.0:
            return val
    return pairs[-1][0]


# ---------------------------------------------------------------------------
# Ensemble function
# ---------------------------------------------------------------------------


def detect_baseline_ensemble(
    signal: npt.NDArray[np.floating],
    tr: float = 1.5,
    min_baseline: int = 3,
    cusum_threshold: float = 5.0,
    variance_ratio_threshold: float = 3.0,
) -> tuple[int, dict[str, Any]]:
    """Detect bolus arrival using an ensemble of 5 orthogonal methods.

    Runs amplitude z-score, gradient z-score, CUSUM, sliding-window
    variance ratio, and max-slope backtrack on the input signal.
    Combines results via outlier rejection and weighted median, with
    a conservative bias toward longer baselines when confidence is low.

    Works on both DSC ΔR2* concentration and DCE raw signal — all
    detectors are scale-invariant.

    Args:
        signal: 1D signal time course (positive bolus polarity).
        tr: Repetition time in seconds (for TR-adaptive parameters).
        min_baseline: Minimum allowed baseline frames.
        cusum_threshold: CUSUM trigger threshold in baseline-std units.
        variance_ratio_threshold: Variance ratio trigger threshold.

    Returns:
        Tuple of ``(arrival_frame, diagnostics)``:
            - *arrival_frame*: Consensus bolus arrival frame (>= min_baseline).
            - *diagnostics*: Dict with per-method estimates, confidence,
              outlier flags, and consensus method.
    """
    n = len(signal)
    diag: dict[str, Any] = {"n_timepoints": n, "tr": tr}

    # Edge case guards
    if n < max(6, min_baseline + 3):
        diag["method"] = "fallback_short"
        return min_baseline, diag

    if np.all(~np.isfinite(signal)):
        diag["method"] = "fallback_nan"
        return min_baseline, diag

    # Replace NaN with 0 for robustness
    signal = np.where(np.isfinite(signal), signal, 0.0)

    if float(np.std(signal)) < EPSILON_CONCENTRATION:
        diag["method"] = "fallback_constant"
        return min_baseline, diag

    # TR-adaptive parameters
    safe_tr = max(tr, 0.1)
    n_consec = max(1, min(3, int(4.0 / safe_tr)))

    # Two baseline windows:
    # - Small (5 frames) for amplitude-based methods to avoid bolus
    #   contamination when the baseline is short.
    # - Large (20 frames) for gradient/CUSUM/variance methods that
    #   need more frames to characterize preload washout dynamics.
    #   These methods are immune to bolus contamination because they
    #   operate on derivatives or z-scores of the gradient.
    bl_small = min(5, n // 4)
    bl_large = min(20, n // 4)

    # Run all 5 detectors
    estimates: dict[str, Optional[int]] = {
        "amplitude_zscore": _detect_amplitude_zscore(
            signal, bl_small, n_consec,
        ),
        "gradient_zscore": _detect_gradient_zscore(
            signal, bl_large, n_consec,
        ),
        "cusum": _detect_cusum(signal, bl_small, cusum_threshold),
        "variance_ratio": _detect_variance_ratio(
            signal, bl_small, threshold=variance_ratio_threshold,
        ),
        "max_slope_backtrack": _detect_max_slope_backtrack(
            signal, min_baseline=min_baseline,
        ),
    }
    diag["per_method"] = {k: v for k, v in estimates.items()}

    # Collect non-None results
    valid = {k: v for k, v in estimates.items() if v is not None}
    diag["n_valid_methods"] = len(valid)

    if len(valid) < 2:
        diag["method"] = "fallback_insufficient_methods"
        # Use the single valid estimate if available, else min_baseline
        if len(valid) == 1:
            result = max(list(valid.values())[0], min_baseline)
            diag["method"] = f"single_method_{list(valid.keys())[0]}"
            return result, diag
        return min_baseline, diag

    values = list(valid.values())
    method_names = list(valid.keys())

    # IQR outlier rejection
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    median_val = float(np.median(values))

    if iqr > 0:
        inlier_mask = [
            abs(v - median_val) <= 2.0 * iqr for v in values
        ]
    else:
        # All values are close — no outliers
        inlier_mask = [True] * len(values)

    inlier_values = [v for v, m in zip(values, inlier_mask) if m]
    inlier_names = [n for n, m in zip(method_names, inlier_mask) if m]
    outlier_names = [n for n, m in zip(method_names, inlier_mask) if not m]
    diag["outlier_methods"] = outlier_names

    if len(inlier_values) < 2:
        # Outlier rejection removed too many — use all valid
        inlier_values = values
        inlier_names = method_names

    # Weighted median (B and C get 1.5x weight)
    weight_map = {
        "gradient_zscore": 1.5,
        "cusum": 1.5,
        "amplitude_zscore": 1.0,
        "variance_ratio": 1.0,
        "max_slope_backtrack": 1.0,
    }
    weights = [weight_map.get(n, 1.0) for n in inlier_names]
    consensus = _weighted_median(inlier_values, weights)

    # Conservative bias: if low agreement, take latest (longest baseline)
    n_agreeing = sum(
        1 for v in inlier_values if abs(v - consensus) <= 3
    )
    diag["n_agreeing"] = n_agreeing
    diag["consensus_raw"] = consensus

    if n_agreeing < 3 and len(inlier_values) >= 3:
        # Low confidence — take the latest (most conservative) estimate
        consensus = max(inlier_values)
        diag["method"] = "conservative_latest"
    else:
        diag["method"] = "weighted_median"

    result = max(consensus, min_baseline)
    diag["arrival_frame"] = result
    return result, diag
