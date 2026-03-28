"""Derivative-anchored baseline detection for DSC and DCE perfusion MRI.

Two-pass approach that solves the circular dependency problem: z-score
methods need a baseline_window, but choosing the window requires knowing
where the baseline ends.

**Phase 1 — Derivative Anchor:** Find bolus arrival from the largest
derivative event (max-slope backtrack + 2nd derivative peak).  These
methods need NO baseline_window and are immune to preload washout.

**Phase 2 — Z-Score Refinement:** Use the Phase 1 anchor to safely
set ``baseline_window = anchor - 2``, then run z-score methods to
refine ±1-2 frames.  Asymmetric: earlier detection is useful (catches
gradual onset); much later detection is suspicious (preload noise).

**Phase 3 — Multi-Signal Consensus:** When multiple signal
representations are available (p99, p90, mean), run Phase 1+2 on each
and take the median.

**Phase 4 — Confidence:** Report HIGH/MEDIUM/LOW for QC.
"""

import logging
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION

logger = logging.getLogger(__name__)

#: Minimum fractional signal change to accept Phase 2 earlier refinement.
#: At clinical Gd doses (0.1 mmol/kg), first-frame tissue enhancement
#: is 5-15%. 1.5% is conservatively below this for low-dose/saturated cases.
#: Ref: physics expert review (2026-03-28).
_MIN_FRACTIONAL_CHANGE: float = 0.015

#: Minimum baseline frames for reliable z-score estimation.
#: With N<8, std has >28% coefficient of variation (chi-distribution),
#: making 3-sigma thresholds unreliable for detecting bolus onset.
_MIN_BASELINE_ZSCORE: int = 8


# ---------------------------------------------------------------------------
# Phase 1 detectors — no baseline_window needed
# ---------------------------------------------------------------------------


def _detect_max_slope_backtrack(
    signal: npt.NDArray[np.floating],
    min_baseline: int = 3,
    backtrack_fraction: float = 0.1,
) -> Optional[int]:
    """Find bolus arrival by backtracking from the largest derivative.

    Smooths the derivative with a 3-frame boxcar to reject motion spikes,
    skips the first 2 frames (T1 recovery artifacts), then walks backward
    from the peak slope to where the derivative drops below
    ``backtrack_fraction`` of the maximum.
    """
    try:
        n = len(signal)
        if n < 5:
            return None

        raw_deriv = np.diff(signal.astype(np.float64))

        # 3-frame boxcar smoothing to reject single-frame motion spikes
        deriv_smooth = raw_deriv.copy()
        if len(deriv_smooth) >= 3:
            kernel = np.ones(3) / 3.0
            deriv_smooth = np.convolve(deriv_smooth, kernel, mode="same")

        # Skip first 2 frames (T1 recovery artifacts)
        search_start = min(2, n - 4)
        deriv_search = deriv_smooth[search_start:]

        if len(deriv_search) == 0 or np.max(deriv_search) <= 0:
            return None

        # Find peak on smoothed (motion-spike resistant)
        peak_idx = int(np.argmax(deriv_search)) + search_start
        backtrack_thresh = backtrack_fraction * float(deriv_smooth[peak_idx])

        # Walk backward from peak on smoothed derivative
        arrival = peak_idx
        for i in range(peak_idx - 1, min_baseline - 1, -1):
            if deriv_smooth[i] < backtrack_thresh:
                arrival = i + 1
                break
        else:
            arrival = min_baseline

        # Forward-only raw verification: the boxcar can bleed the bolus
        # derivative 1-2 frames backward.  Advance while the raw
        # derivative is below threshold.  Can only move FORWARD (toward
        # the bolus), never backward past motion spikes.
        raw_thresh = backtrack_fraction * float(raw_deriv[peak_idx])
        while arrival < peak_idx and raw_deriv[arrival] < raw_thresh:
            arrival += 1

        return max(arrival, min_baseline)
    except Exception:
        return None


def _detect_second_derivative_peak(
    signal: npt.NDArray[np.floating],
    min_baseline: int = 3,
) -> Optional[int]:
    """Find bolus onset from the peak of the 2nd derivative (acceleration).

    The 2nd derivative peaks where the bolus curve accelerates most —
    typically 1-2 frames before the maximum slope.  Only reliable when
    the bolus rising edge spans >= 3 frames (FWHM/TR >= 2).
    """
    try:
        n = len(signal)
        if n < 6:
            return None

        deriv = np.diff(signal.astype(np.float64))

        # 3-frame boxcar smoothing (same as max-slope detector)
        if len(deriv) >= 3:
            kernel = np.ones(3) / 3.0
            deriv = np.convolve(deriv, kernel, mode="same")

        deriv2 = np.diff(deriv)
        search_start = min(2, len(deriv2) - 1)
        d2_search = deriv2[search_start:]

        if len(d2_search) == 0 or np.max(d2_search) <= 0:
            return None

        peak = int(np.argmax(d2_search)) + search_start
        return max(peak, min_baseline)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 2 detectors — require baseline_window (derived from Phase 1)
# ---------------------------------------------------------------------------


def _detect_amplitude_zscore(
    signal: npt.NDArray[np.floating],
    baseline_window: int,
    n_consecutive: int,
) -> Optional[int]:
    """Amplitude z-score detection (3-sigma above baseline mean)."""
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
    """Gradient z-score detection (z-score of first derivative)."""
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
    """CUSUM change-point detection (cumulative sum of z-scored deviations)."""
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


# ---------------------------------------------------------------------------
# Core: single-signal Phase 1 + Phase 2
# ---------------------------------------------------------------------------

_TR_THRESHOLD_2ND_DERIV = 4.0  # seconds — below this, 2nd derivative is useful


def _detect_single_signal(
    signal: npt.NDArray[np.floating],
    tr: float,
    min_baseline: int,
) -> tuple[int, str, dict[str, Any]]:
    """Run Phase 1 + Phase 2 on a single signal.

    Returns (arrival_frame, confidence, phase_diagnostics).
    """
    n = len(signal)
    diag: dict[str, Any] = {}

    # --- Phase 1: Derivative Anchor ---
    anchor_slope = _detect_max_slope_backtrack(signal, min_baseline=min_baseline)

    if tr < _TR_THRESHOLD_2ND_DERIV:
        anchor_d2 = _detect_second_derivative_peak(signal, min_baseline=min_baseline)
        if anchor_slope is not None and anchor_d2 is not None:
            # Only accept 2nd derivative if within 10 frames of max-slope.
            # Larger gap suggests the d2 peak is from a motion artifact,
            # not the same bolus event.
            if abs(anchor_slope - anchor_d2) <= 10:
                anchor = min(anchor_slope, anchor_d2)
            else:
                anchor = anchor_slope
        else:
            anchor = anchor_slope or anchor_d2
    else:
        # Long TR: 2nd derivative unreliable (FWHM/TR < 2)
        anchor = anchor_slope
        anchor_d2 = None

    diag["phase1_max_slope"] = anchor_slope
    diag["phase1_d2_peak"] = anchor_d2
    diag["phase1_tr_rule"] = "min" if tr < _TR_THRESHOLD_2ND_DERIV else "slope_only"

    if anchor is None:
        diag["phase1_anchor"] = None
        return min_baseline, "low", diag

    anchor = max(anchor, min_baseline)
    diag["phase1_anchor"] = anchor

    # --- Phase 2: Z-Score Refinement ---
    # baseline_window derived from anchor — guaranteed no bolus contamination
    bl_window = max(3, anchor - 2)
    n_consec = max(1, min(3, int(4.0 / max(tr, 0.1))))

    phase2: dict[str, Optional[int]] = {
        "amplitude_zscore": _detect_amplitude_zscore(signal, bl_window, n_consec),
        "gradient_zscore": _detect_gradient_zscore(signal, bl_window, n_consec),
        "cusum": _detect_cusum(signal, bl_window),
    }
    diag["phase2"] = {k: v for k, v in phase2.items()}

    valid_p2 = {k: v for k, v in phase2.items() if v is not None}

    # Asymmetric refinement:
    # - 2+ methods detect EARLIER by 1-3 frames → refine (gradient onset)
    # - 2+ methods detect LATER by >3 frames → keep anchor (preload noise)
    #
    # Two guards prevent false early detection:
    # 1. Short-baseline skip: z-score std has >35% CV with <8 samples
    # 2. Magnitude guard: signal change must exceed 1.5% (physiological minimum)
    skip_earlier = bl_window < _MIN_BASELINE_ZSCORE
    earlier = [v for v in valid_p2.values() if 0 < anchor - v <= 3]
    if len(earlier) >= 2 and not skip_earlier:
        refined = max(earlier)  # latest of the earlier estimates

        # Magnitude guard: verify signal change is physiologically meaningful
        bl_mean = float(np.mean(signal[:bl_window]))
        if abs(bl_mean) > 1e-6:
            frac_change = abs(float(signal[refined]) - bl_mean) / abs(bl_mean)
        else:
            # Near-zero baseline (e.g., DSC concentration domain):
            # any detectable change above noise is meaningful
            frac_change = (
                float("inf") if abs(float(signal[refined])) > 1e-6 else 0.0
            )

        if frac_change >= _MIN_FRACTIONAL_CHANGE:
            diag["phase2_action"] = f"refined_earlier_to_{refined}"
            # Phase 1 and Phase 2 agree within a few frames → high confidence
            confidence = "high"
            return max(refined, min_baseline), confidence, diag
        else:
            diag["phase2_action"] = (
                f"earlier_rejected_low_magnitude_{frac_change:.4f}"
            )
            # Fall through to confirmation/anchor logic below
    elif len(earlier) >= 2 and skip_earlier:
        diag["phase2_action"] = "earlier_skipped_short_baseline"

    # Z-score methods have inherent 1-2 frame detection lag.  If 2+
    # methods detect 1-3 frames LATER, this confirms the anchor with
    # expected latency — upgrade confidence but don't move the anchor.
    later_confirming = [v for v in valid_p2.values() if 0 < v - anchor <= 3]
    if len(later_confirming) >= 2:
        diag["phase2_action"] = "confirmed_by_lagged_zscore"
        return max(anchor, min_baseline), "high", diag

    # Check agreement: how many Phase 2 methods are within ±2 of anchor?
    near_anchor = [v for v in valid_p2.values() if abs(v - anchor) <= 2]

    if len(near_anchor) >= 2:
        confidence = "high"
        diag["phase2_action"] = "confirmed_anchor"
    elif len(valid_p2) == 0:
        confidence = "medium"
        diag["phase2_action"] = "no_phase2_data"
    else:
        # Phase 2 disagrees — check if max derivative is strong enough
        deriv = np.diff(signal.astype(np.float64))
        max_d = float(np.max(deriv)) if len(deriv) > 0 else 0
        med_d = float(np.median(np.abs(deriv))) if len(deriv) > 0 else 0
        if med_d > 0 and max_d / med_d >= 3.0:
            confidence = "medium"
            diag["phase2_action"] = "anchor_strong_derivative"
        else:
            confidence = "low"
            diag["phase2_action"] = "anchor_weak_derivative"

    return max(anchor, min_baseline), confidence, diag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_baseline_ensemble(
    signal: npt.NDArray[np.floating],
    tr: float = 1.5,
    min_baseline: int = 3,
    additional_signals: Optional[list[npt.NDArray[np.floating]]] = None,
) -> tuple[int, dict[str, Any]]:
    """Detect bolus arrival via derivative-anchored two-pass ensemble.

    **Phase 1:** Derivative-based anchor (max-slope backtrack + 2nd
    derivative peak).  Needs no baseline_window.  Immune to preload.

    **Phase 2:** Z-score refinement using Phase 1 baseline window.
    Asymmetric: earlier detection refines; much later is suspicious.

    **Phase 3:** Multi-signal consensus (when ``additional_signals``
    provided).  Median of per-signal Phase 1+2 results.

    **Phase 4:** Confidence scoring (HIGH/MEDIUM/LOW).

    Args:
        signal: 1D signal time course (positive bolus polarity).
        tr: Repetition time in seconds.
        min_baseline: Minimum allowed baseline frames.
        additional_signals: Optional list of additional signal curves
            (e.g., p90, mean) for multi-signal consensus.

    Returns:
        Tuple of ``(arrival_frame, diagnostics)`` where diagnostics
        includes ``"confidence"`` (high/medium/low), per-phase details,
        and per-signal results when multi-signal is used.
    """
    n = len(signal)
    diag: dict[str, Any] = {"n_timepoints": n, "tr": tr}

    # Edge case guards
    if n < max(6, min_baseline + 3):
        diag.update(method="fallback_short", confidence="low")
        return min_baseline, diag

    if np.all(~np.isfinite(signal)):
        diag.update(method="fallback_nan", confidence="low")
        return min_baseline, diag

    # Replace NaN with 0 for robustness
    signal = np.where(np.isfinite(signal), signal, 0.0)

    if float(np.std(signal)) < EPSILON_CONCENTRATION:
        diag.update(method="fallback_constant", confidence="low")
        return min_baseline, diag

    # --- Phase 1 + 2 on primary signal ---
    primary_arrival, primary_conf, primary_diag = _detect_single_signal(
        signal, tr, min_baseline,
    )
    diag["primary"] = primary_diag
    diag["primary_arrival"] = primary_arrival
    diag["primary_confidence"] = primary_conf

    # --- Phase 3: Multi-signal consensus ---
    if additional_signals:
        all_arrivals = [(primary_arrival, primary_conf)]

        for i, extra_signal in enumerate(additional_signals):
            extra = np.where(np.isfinite(extra_signal), extra_signal, 0.0)
            if float(np.std(extra)) < EPSILON_CONCENTRATION:
                continue
            arr, conf, sig_diag = _detect_single_signal(extra, tr, min_baseline)
            all_arrivals.append((arr, conf))
            diag[f"signal_{i}"] = sig_diag
            diag[f"signal_{i}_arrival"] = arr
            diag[f"signal_{i}_confidence"] = conf

        # Weighted median: halve weight of low-confidence signals
        values = [a for a, _ in all_arrivals]
        weights = [0.5 if c == "low" else 1.0 for _, c in all_arrivals]

        if len(values) > 1:
            # Sort and compute weighted median
            pairs = sorted(zip(values, weights))
            total = sum(weights)
            cumulative = 0.0
            consensus = pairs[-1][0]
            for val, w in pairs:
                cumulative += w
                if cumulative >= total / 2.0:
                    consensus = val
                    break

            primary_arrival = max(consensus, min_baseline)
            diag["phase3_consensus"] = primary_arrival
            diag["phase3_all_arrivals"] = [
                {"arrival": a, "confidence": c} for a, c in all_arrivals
            ]

            # Upgrade confidence if all signals agree
            arrivals_only = [a for a, _ in all_arrivals]
            spread = max(arrivals_only) - min(arrivals_only)
            if spread <= 3 and primary_conf != "low":
                primary_conf = "high"
            elif spread > 5:
                primary_conf = "medium" if primary_conf == "high" else primary_conf

    # --- Phase 4: Confidence ---
    diag["confidence"] = primary_conf
    diag["method"] = "derivative_anchored"
    diag["arrival_frame"] = primary_arrival

    if primary_conf == "low":
        logger.warning(
            "Low-confidence baseline detection: arrival=%d, tr=%.1f",
            primary_arrival, tr,
        )

    return primary_arrival, diag
