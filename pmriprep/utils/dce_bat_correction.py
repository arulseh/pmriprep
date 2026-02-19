"""Bolus Arrival Time (BAT) correction for DCE population AIFs.

Ported from perf-plus dce_pk_modeling.py:898-1083. When using a population
AIF (Parker model), the AIF timing is generic and may not match the actual
tissue enhancement onset. This module detects the mismatch and applies a
time shift to align the AIF with tissue data before PK fitting.
"""

import logging
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger("nipype.interface")

# Minimum signal value to avoid division by zero (matches perf-plus)
_EPSILON_SIGNAL = 1e-9


def detect_tissue_bat(
    dce_signal: np.ndarray,
    brain_mask: np.ndarray,
    time_vector_s: np.ndarray,
    baseline_frames: int = 5,
) -> dict[str, Any]:
    """Detect bolus arrival time (BAT) from tissue enhancement curves.

    Uses the median brain signal to find when contrast agent arrives in
    tissue. Critical for aligning population AIF with patient data.

    Args:
        dce_signal: 4D DCE raw signal (x, y, z, t).
        brain_mask: 3D brain mask (boolean-like).
        time_vector_s: Time points in seconds, shape (nt,).
        baseline_frames: Number of pre-contrast baseline frames.

    Returns:
        Dict with tissue_bat_s, tissue_bat_frame, tissue_peak_s,
        tissue_peak_frame, max_enhancement_pct, confidence, method.
    """
    # Get median signal curve across all brain voxels
    brain_voxels = dce_signal[brain_mask > 0]  # Shape: (n_voxels, nt)
    median_curve = np.median(brain_voxels, axis=0)

    # Calculate baseline and enhancement
    baseline = np.mean(median_curve[:baseline_frames])
    if baseline < _EPSILON_SIGNAL:
        baseline = _EPSILON_SIGNAL

    # Normalize to percent enhancement
    enhancement = (median_curve - baseline) / baseline * 100

    # Method 1: Threshold-based detection (10% of max enhancement)
    max_enhancement = float(np.max(enhancement[baseline_frames:]))
    threshold = max_enhancement * 0.10

    # Find first frame exceeding threshold after baseline
    arrival_candidates = np.where(enhancement[baseline_frames:] > threshold)[0]

    if len(arrival_candidates) > 0:
        tissue_bat_frame = int(arrival_candidates[0] + baseline_frames)
        tissue_bat_s = float(time_vector_s[tissue_bat_frame])
        method = "threshold_10pct"
        confidence = "high" if max_enhancement > 30 else "medium"
    else:
        # Fallback: use first frame after baseline with positive slope
        gradient = np.gradient(enhancement)
        positive_slope = np.where(gradient[baseline_frames:] > 1.0)[0]
        if len(positive_slope) > 0:
            tissue_bat_frame = int(positive_slope[0] + baseline_frames)
            tissue_bat_s = float(time_vector_s[tissue_bat_frame])
            method = "gradient"
            confidence = "low"
        else:
            # Last resort: assume BAT at baseline_frames
            tissue_bat_frame = baseline_frames
            tissue_bat_s = float(time_vector_s[tissue_bat_frame])
            method = "default"
            confidence = "low"

    # Find peak enhancement time
    tissue_peak_frame = int(
        np.argmax(enhancement[baseline_frames:]) + baseline_frames
    )
    tissue_peak_s = float(time_vector_s[tissue_peak_frame])

    return {
        "tissue_bat_s": tissue_bat_s,
        "tissue_bat_frame": tissue_bat_frame,
        "tissue_peak_s": tissue_peak_s,
        "tissue_peak_frame": tissue_peak_frame,
        "max_enhancement_pct": max_enhancement,
        "confidence": confidence,
        "method": method,
    }


def detect_aif_bat(
    aif_concentration: np.ndarray,
    time_vector_s: np.ndarray,
) -> dict[str, Any]:
    """Detect bolus arrival time from AIF concentration curve.

    Args:
        aif_concentration: AIF concentration in mM, shape (nt,).
        time_vector_s: Time points in seconds, shape (nt,).

    Returns:
        Dict with aif_bat_s, aif_bat_frame, aif_peak_s, aif_peak_frame.
    """
    # Find AIF peak
    aif_peak_frame = int(np.argmax(aif_concentration))
    aif_peak_s = float(time_vector_s[aif_peak_frame])

    # Find BAT: first frame with concentration > 10% of peak
    peak_conc = aif_concentration[aif_peak_frame]
    threshold = peak_conc * 0.10

    arrival_candidates = np.where(aif_concentration > threshold)[0]
    if len(arrival_candidates) > 0:
        aif_bat_frame = int(arrival_candidates[0])
        aif_bat_s = float(time_vector_s[aif_bat_frame])
    else:
        # Fallback to first non-zero frame
        nonzero = np.where(aif_concentration > _EPSILON_SIGNAL)[0]
        if len(nonzero) > 0:
            aif_bat_frame = int(nonzero[0])
            aif_bat_s = float(time_vector_s[aif_bat_frame])
        else:
            aif_bat_frame = 0
            aif_bat_s = 0.0

    return {
        "aif_bat_s": aif_bat_s,
        "aif_bat_frame": aif_bat_frame,
        "aif_peak_s": aif_peak_s,
        "aif_peak_frame": aif_peak_frame,
    }


def apply_aif_time_shift(
    aif_concentration: np.ndarray,
    time_vector_s: np.ndarray,
    shift_s: float,
) -> np.ndarray:
    """Apply time shift to AIF concentration curve.

    Positive shift delays the AIF (shifts right in time).

    Args:
        aif_concentration: Original AIF concentration, shape (nt,).
        time_vector_s: Time points in seconds, shape (nt,).
        shift_s: Time shift in seconds (positive = delay AIF).

    Returns:
        Time-shifted AIF concentration, shape (nt,).
    """
    if abs(shift_s) < 0.1:
        return aif_concentration.copy()

    # Create interpolation function for original AIF
    aif_interp = interp1d(
        time_vector_s,
        aif_concentration,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Sample at shifted time points
    # If shift_s > 0, AIF appears later: sample at (t - shift)
    shifted_times = time_vector_s - shift_s
    aif_shifted = aif_interp(shifted_times)

    # Ensure non-negative
    aif_shifted = np.maximum(aif_shifted, 0.0)

    return aif_shifted  # type: ignore[no-any-return]
