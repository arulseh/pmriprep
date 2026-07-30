"""DSC signal processing utilities.

Functions for converting DSC-MRI signal to concentration (ΔR2*)
and computing derived metrics like PSR, peak height, and normalized CBV.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
    EPSILON_SIGNAL,
)


def signal_to_concentration(
    signal: npt.NDArray[np.floating],
    te: float,
    baseline_frames: int = 5,
) -> npt.NDArray[np.floating]:
    """Convert DSC signal to ΔR2* concentration.

    Computes concentration from the signal decay using the relationship:
        ΔR2* = -ln(S(t)/S0) / TE

    Always returns ΔR2* in s⁻¹. All downstream quantification is relative.

    Parameters
    ----------
    signal : np.ndarray
        4D array of DSC signal (x, y, z, time) or 1D time series.
        Signal values should be positive.
    te : float
        Echo time in seconds. Typical values: 0.025-0.040s.
    baseline_frames : int, optional
        Number of pre-bolus frames for S0 estimation.
        Default is 5 frames.

    Returns
    -------
    np.ndarray
        ΔR2* array with same shape as input signal, units s⁻¹.

    Raises
    ------
        ValueError: If signal array is empty or has invalid dimensions.
        ValueError: If te is not positive.
        ValueError: If baseline_frames exceeds signal length.

    Notes
    -----
        - Signal ratio is lower-bounded at EPSILON_SIGNAL to prevent
          log(0). No upper bound: S(t)/S0 > 1 produces negative ΔR2*,
          which preserves symmetric noise in post-bolus frames.
        - Baseline S0 is computed as mean of first baseline_frames.
        - For 4D input, baseline is computed per-voxel.
    """
    # Validate inputs
    if signal.size == 0:
        raise ValueError("Signal array is empty")

    if te <= 0:
        raise ValueError(f"TE must be positive, got {te}")

    # Handle 1D input
    if signal.ndim == 1:
        n_timepoints = signal.shape[0]
    elif signal.ndim == 4:
        n_timepoints = signal.shape[3]
    else:
        raise ValueError(
            f"Signal must be 1D (time) or 4D (x,y,z,time), got {signal.ndim}D"
        )

    if baseline_frames >= n_timepoints:
        raise ValueError(
            f"baseline_frames ({baseline_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    if baseline_frames < 1:
        raise ValueError(f"baseline_frames must be at least 1, got {baseline_frames}")

    # Compute baseline S0
    if signal.ndim == 1:
        s0 = np.mean(signal[:baseline_frames])
    else:
        # 4D: mean over first baseline_frames along time axis
        s0 = np.mean(signal[..., :baseline_frames], axis=-1, keepdims=True)

    # Compute signal ratio S(t)/S0, clip to valid range
    # Clip s0 to prevent division by zero
    s0_safe = np.maximum(s0, EPSILON_SIGNAL)
    signal_ratio = signal / s0_safe

    # Lower-bound signal ratio to prevent log(0). No upper bound:
    # S(t)/S0 > 1.0 produces negative ΔR2*, preserving symmetric noise
    # in post-bolus frames. Clipping to 1.0 creates half-rectified noise
    # with positive bias.
    signal_ratio_clipped = np.maximum(signal_ratio, EPSILON_SIGNAL)

    # Compute ΔR2* = -ln(S/S0) / TE
    # Note: negative of log because signal decreases with contrast
    delta_r2_star = -np.log(signal_ratio_clipped) / te

    return delta_r2_star  # type: ignore[no-any-return]


def signal_to_delta_r2_star(
    signal: npt.NDArray[np.floating],
    te: float,
    baseline_frames: int = 5,
) -> npt.NDArray[np.floating]:
    """Convert DSC signal to ΔR2* (relative concentration).

    Convenience alias for signal_to_concentration, which now always
    returns ΔR2* in s⁻¹.

    Parameters
    ----------
    signal : np.ndarray
        4D array of DSC signal (x, y, z, time) or 1D time series.
    te : float
        Echo time in seconds.
    baseline_frames : int, optional
        Number of pre-bolus frames for S0 estimation.

    Returns
    -------
    np.ndarray
        ΔR2* array with same shape as input signal, units s⁻¹.
    """
    return signal_to_concentration(
        signal=signal,
        te=te,
        baseline_frames=baseline_frames,
    )


def get_baseline_signal(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 5,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute baseline signal S0 from pre-bolus frames.

    Parameters
    ----------
    signal : np.ndarray
        DSC signal array (1D, 3D, or 4D).
    baseline_frames : int, optional
        Number of pre-bolus frames to average.

    Returns
    -------
    float or np.ndarray
        Baseline signal value (scalar for 1D) or array (for 3D/4D).
    """
    if signal.ndim == 1:
        return float(np.mean(signal[:baseline_frames]))
    elif signal.ndim == 4:
        return np.mean(signal[..., :baseline_frames], axis=-1)  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Signal must be 1D or 4D, got {signal.ndim}D")


def find_bolus_arrival(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 5,
    threshold_factor: float = 3.0,
) -> int:
    """Find the bolus arrival time point.

    Detects when signal drops significantly below baseline,
    indicating contrast agent arrival.

    Parameters
    ----------
    signal : np.ndarray
        1D DSC signal time course.
    baseline_frames : int, optional
        Number of pre-bolus frames.
    threshold_factor : float, optional
        Number of baseline standard deviations
        for threshold (default 3.0).

    Returns
    -------
    int
        Index of bolus arrival frame.

    Raises
    ------
        ValueError: If signal is not 1D.
        ValueError: If no bolus arrival is detected.
    """
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got {signal.ndim}D")

    baseline = signal[:baseline_frames]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)

    # Threshold: signal drops below baseline - threshold_factor * std
    threshold = baseline_mean - threshold_factor * baseline_std

    # Find first point below threshold after baseline
    for i in range(baseline_frames, len(signal)):
        if signal[i] < threshold:
            return i

    raise ValueError("No bolus arrival detected in signal")


def detect_bolus_arrival(
    concentration_curve: npt.NDArray[np.floating],
    baseline_frames: int = 3,
    n_consecutive: int = 1,
) -> Tuple[int, int]:
    """Detect bolus arrival in a concentration curve.

    Uses z-score normalization (3-sigma) above the initial baseline to
    find the first frame with significant contrast arrival, then returns
    both the arrival index and the number of usable baseline frames.

    The z-score approach is scale-invariant: z = (x - mean) / std cancels
    any constant scaling factor, so detection returns the same frame
    whether input is in ΔR2* (s⁻¹) or concentration (mM).

    Parameters
    ----------
    concentration_curve : np.ndarray
        1D concentration time course (positive values
        indicate contrast arrival in concentration space).
    baseline_frames : int, optional
        Number of initial frames for baseline estimation.
        Default is 3 to avoid early bolus contamination.
    n_consecutive : int, optional
        Number of consecutive frames that must all exceed
        the 3-sigma threshold to count as bolus arrival. Using
        n_consecutive >= 2 rejects isolated noise spikes common in
        preloaded acquisitions with noisy elevated baselines.

    Returns
    -------
    tuple
        Tuple of (arrival_idx, n_baseline_frames):
            - arrival_idx: Frame index where bolus arrives
            - n_baseline_frames: Number of usable pre-bolus frames
              (always >= 2)
    """
    n_baseline = min(baseline_frames, len(concentration_curve) // 4)
    n_baseline = max(2, n_baseline)
    baseline_mean = np.mean(concentration_curve[:n_baseline])
    baseline_std = np.std(concentration_curve[:n_baseline])

    # Use z-score normalization for scale-invariant detection.
    # z = (x - mean) / std cancels any constant scaling factor (e.g. r2*),
    # so detection returns the same frame whether input is in s⁻¹ or mM.
    z_scores = (concentration_curve - baseline_mean) / max(baseline_std, EPSILON_CONCENTRATION)

    n_consecutive = max(1, n_consecutive)
    above_threshold = z_scores > 3.0

    arrival_idx = n_baseline  # fallback
    if n_consecutive == 1:
        # Fast path: first single frame above 3-sigma
        above = np.where(above_threshold)[0]
        if len(above) > 0:
            arrival_idx = int(above[0])
    else:
        # Require n_consecutive frames all above threshold to reject
        # isolated noise spikes in preloaded acquisitions
        for i in range(len(above_threshold) - n_consecutive + 1):
            if all(above_threshold[i : i + n_consecutive]):
                arrival_idx = i
                break

    return arrival_idx, max(2, arrival_idx - 1)


def detect_bolus_arrival_gradient(
    concentration_curve: npt.NDArray[np.floating],
    baseline_window: int = 20,
    z_threshold: float = 3.0,
    n_consecutive: int = 3,
) -> Tuple[int, int]:
    """Detect bolus arrival using gradient z-score thresholding.

    Operates on the first derivative (gradient) of the concentration curve
    rather than raw values.  This makes detection robust to preloaded
    acquisitions where the baseline is elevated or slowly decaying —
    situations where the amplitude-based ``detect_bolus_arrival`` triggers
    on washout noise.

    The gradient of a slowly-decaying preload baseline is a small negative
    constant, while the gradient at true bolus arrival is a large positive
    spike.  Z-scoring the gradient cleanly separates these two regimes at
    any baseline elevation.

    Parameters
    ----------
    concentration_curve : np.ndarray
        1D concentration time course.
    baseline_window : int, optional
        Maximum number of initial frames used to estimate
        gradient baseline statistics.  Clipped to ``len // 4`` so it
        never exceeds one quarter of the curve.  Default 20 provides
        stable mean/std for most acquisition lengths.
    z_threshold : float, optional
        Number of standard deviations above baseline gradient
        mean to count as significant.  Default 3.0.
    n_consecutive : int, optional
        Number of consecutive above-threshold gradient
        frames required.  Should be TR-adaptive at the call site
        (e.g. ``max(1, min(3, int(4.0 / tr)))``).

    Returns
    -------
    tuple
        Tuple of (arrival_idx, n_baseline_frames):
            - arrival_idx: Frame index where bolus arrives.
            - n_baseline_frames: Usable pre-bolus frames (always >= 2).
    """
    n = len(concentration_curve)

    # Compute first derivative via central differences
    grad = np.gradient(concentration_curve)

    # Baseline window: at most 1/4 of the curve
    bl = min(baseline_window, n // 4)

    # Minimum sample guard — need at least 4 gradient values for
    # a reliable mean/std estimate; otherwise return fallback.
    if bl < 4:
        fallback = min(baseline_window, max(0, n - 1))
        return fallback, max(2, fallback - 1)

    # Baseline gradient statistics with EPSILON floor on sigma to
    # handle flat/zero curves where np.gradient returns exact zeros.
    mu = float(np.mean(grad[:bl]))
    sigma = float(np.std(grad[:bl]))
    sigma = max(sigma, EPSILON_CONCENTRATION)

    z_scores = (grad - mu) / sigma

    n_consecutive = max(1, n_consecutive)
    # Require BOTH z-score above threshold AND positive gradient.
    # Bolus arrival always produces positive dC/dt (concentration
    # increasing).  Preload washout has negative dC/dt throughout,
    # but its z-score drifts upward as the gradient decays toward
    # zero.  The positivity check eliminates these false triggers.
    above = (z_scores > z_threshold) & (grad > 0)

    # Find first run of n_consecutive frames above threshold
    arrival_idx: Optional[int] = None
    for i in range(len(above) - n_consecutive + 1):
        if all(above[i : i + n_consecutive]):
            arrival_idx = i
            break

    if arrival_idx is None:
        # No sustained gradient spike found — return fallback
        fallback = min(baseline_window, max(0, n - 1))
        return fallback, max(2, fallback - 1)

    # Post-detection amplitude validation: verify concentration actually
    # rose in a small window after the gradient spike.  np.gradient uses
    # central differences so the gradient fires 1-2 frames before the
    # amplitude crosses threshold.
    lookahead = max(n_consecutive, 2)
    window_end = min(arrival_idx + lookahead, n)
    amp_baseline_mean = float(np.mean(concentration_curve[:bl]))
    amp_baseline_std = float(np.std(concentration_curve[:bl]))
    amp_threshold = amp_baseline_mean + 2.0 * max(amp_baseline_std, EPSILON_CONCENTRATION)

    if float(np.max(concentration_curve[arrival_idx:window_end])) < amp_threshold:
        # Gradient spike without sustained concentration rise — spurious
        fallback = min(baseline_window, max(0, n - 1))
        return fallback, max(2, fallback - 1)

    return arrival_idx, max(2, arrival_idx - 1)


def compute_psr(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 10,
    tail_frames: int = 10,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute percent signal recovery (PSR).

    PSR quantifies how much signal recovers after the bolus passage,
    indicating blood-brain barrier integrity.

    Formula:
        PSR = 100 × (S_tail - S_min) / (S_baseline - S_min)

    Clinical interpretation:
        - PSR < 70%: Blood-brain barrier disruption (leakage)
        - PSR 70-90%: Partial BBB disruption
        - PSR > 90%: Intact blood-brain barrier

    Parameters
    ----------
    signal : np.ndarray
        DSC signal array. Can be:
            - 1D (time,): Single voxel time course
            - 4D (x, y, z, time): Full 4D volume
    baseline_frames : int, optional
        Number of pre-bolus frames for S_baseline.
        Default is 10.
    tail_frames : int, optional
        Number of post-bolus frames for S_tail.
        Default is 10 (last 10 frames).

    Returns
    -------
    float or np.ndarray
        PSR value(s) as percentage (0-100+ range).
        - Scalar float for 1D input
        - 3D array (x, y, z) for 4D input

    Raises
    ------
        ValueError: If signal has invalid dimensions.
        ValueError: If baseline_frames or tail_frames exceeds signal length.

    Notes
    -----
        - ``compute_psr_legacy`` provides the alternate formula used by
          validation comparisons.
        - s_min uses post-baseline frames only (signal[..., baseline_frames:])
          to avoid baseline contamination.
        - PSR can exceed 100% in cases where the signal overshoots baseline
          after bolus passage (rare, usually indicates motion artifact).
        - PSR can be negative if signal never recovers (very high leakage
          or signal artifact).
        - Epsilon is added to denominator to prevent division by zero.
    """
    # Validate input dimensions
    if signal.ndim not in (1, 4):
        raise ValueError(f"Signal must be 1D or 4D, got {signal.ndim}D")

    # Get number of timepoints
    n_timepoints = signal.shape[-1] if signal.ndim == 4 else signal.shape[0]

    if baseline_frames >= n_timepoints:
        raise ValueError(
            f"baseline_frames ({baseline_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    if tail_frames >= n_timepoints:
        raise ValueError(
            f"tail_frames ({tail_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    if baseline_frames < 1 or tail_frames < 1:
        raise ValueError("baseline_frames and tail_frames must be at least 1")

    # Compute signal components
    # S_min uses post-baseline frames only to find minimum during/after bolus,
    # avoiding spurious baseline noise minima
    if signal.ndim == 1:
        # 1D time series
        s_baseline = float(np.mean(signal[:baseline_frames]))
        s_min = float(np.min(signal[baseline_frames:]))
        s_tail = float(np.mean(signal[-tail_frames:]))
    else:
        # 4D volume (x, y, z, time)
        s_baseline = np.mean(signal[..., :baseline_frames], axis=-1)
        s_min = np.min(signal[..., baseline_frames:], axis=-1)
        s_tail = np.mean(signal[..., -tail_frames:], axis=-1)

    # Compute PSR with epsilon guard in denominator
    denominator = s_baseline - s_min
    # Add small epsilon to prevent division by zero
    denominator_safe = np.maximum(np.abs(denominator), EPSILON_SIGNAL)
    # Preserve sign of denominator
    denominator_safe = np.where(
        denominator >= 0, denominator_safe, -denominator_safe
    )

    psr = 100.0 * (s_tail - s_min) / denominator_safe

    # Return appropriate type
    if signal.ndim == 1:
        return float(psr)
    else:
        return psr  # type: ignore[no-any-return]


def compute_psr_legacy(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 5,
    tail_frames: int = 10,
    brain_mask: npt.NDArray[np.bool_] | None = None,
    min_drop_noise_sd: float = 0.0,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute percent signal recovery from the DSC signal curve.

    Formula::

        PSR = 100 × (S1 - Smin) / (S0 - Smin)

    where S0 = mean baseline signal, S1 = mean tail signal,
    Smin = minimum signal in post-baseline frames.

    Parameters
    ----------
    signal : np.ndarray
        DSC signal array (1D or 4D).
    baseline_frames : int, optional
        Number of pre-bolus frames for S0.
    tail_frames : int, optional
        Number of post-bolus tail frames for S1 (recovery).
    brain_mask : np.ndarray, optional
        Spatial mask used to pool baseline-noise estimates for 4D input.
    min_drop_noise_sd : float, optional
        Minimum baseline-to-minimum signal drop in effective noise standard
        deviations. The effective noise is the larger of the voxel estimate
        and the median positive estimate within ``brain_mask``.

    Returns
    -------
    float or np.ndarray
        PSR as an unclipped percentage. Undefined voxels are NaN.
    """
    n_timepoints = signal.shape[-1]
    if baseline_frames < 2 or tail_frames < 1:
        raise ValueError("baseline_frames must be at least 2 and tail_frames at least 1")
    if baseline_frames + tail_frames > n_timepoints:
        raise ValueError("baseline and recovery windows must not overlap")

    if signal.ndim == 1:
        s0 = float(np.mean(signal[:baseline_frames]))
        s_min = float(np.min(signal[baseline_frames:]))
        s1 = float(np.mean(signal[-tail_frames:]))
        denom = s0 - s_min
        noise = float(np.std(signal[:baseline_frames], ddof=1))
        if (
            not np.isfinite([s0, s_min, s1, noise]).all()
            or s0 <= EPSILON_SIGNAL
            or denom <= min_drop_noise_sd * noise
        ):
            return float("nan")
        return 100.0 * (s1 - s_min) / denom
    elif signal.ndim == 4:
        s0 = np.mean(signal[..., :baseline_frames], axis=-1)
        s_min = np.min(signal[..., baseline_frames:], axis=-1)
        s1 = np.mean(signal[..., -tail_frames:], axis=-1)
        denom = s0 - s_min
        noise = np.std(signal[..., :baseline_frames], axis=-1, ddof=1)
        pool = np.ones(s0.shape, dtype=bool) if brain_mask is None else brain_mask
        pool_values = noise[
            pool & np.isfinite(noise) & (noise > EPSILON_SIGNAL)
        ]
        pooled_noise = float(np.median(pool_values)) if pool_values.size else 0.0
        effective_noise = np.maximum(noise, pooled_noise)
        valid = (
            np.isfinite(s0)
            & np.isfinite(s_min)
            & np.isfinite(s1)
            & np.isfinite(effective_noise)
            & (s0 > EPSILON_SIGNAL)
            & (denom > min_drop_noise_sd * effective_noise)
        )
        psr = np.full(s0.shape, np.nan, dtype=np.float64)
        np.divide(100.0 * (s1 - s_min), denom, out=psr, where=valid)
        return psr
    else:
        raise ValueError(f"Signal must be 1D or 4D, got {signal.ndim}D")


def compute_ttp_legacy(
    concentration: npt.NDArray[np.floating],
    pre_bolus_frames: int,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Compute TTP as the index of maximum concentration.

    Finds time to peak concentration using unsmoothed post-bolus argmax.
    Returns unclipped acquisition-referenced peak time so callers can derive
    AIF-arrival-referenced TTP before applying output clipping.

    Parameters
    ----------
    concentration : np.ndarray
        Concentration array (4D: x,y,z,time or 1D: time).
    pre_bolus_frames : int
        Number of pre-bolus baseline frames to skip.
    dt : float
        Time step in seconds (TR).

    Returns
    -------
    np.ndarray
        TTP map in seconds (unclipped).

    Raises
    ------
        ValueError: If pre_bolus_frames >= number of timepoints.
    """
    n_timepoints = concentration.shape[-1]
    if pre_bolus_frames >= n_timepoints:
        raise ValueError(
            f"pre_bolus_frames ({pre_bolus_frames}) >= n_timepoints "
            f"({n_timepoints}): no post-bolus window to search"
        )
    if concentration.ndim == 4:
        conc_post = concentration[..., pre_bolus_frames:]
        peak_idx = np.argmax(conc_post, axis=-1)
        ttp = (pre_bolus_frames + peak_idx).astype(np.float64) * dt
    elif concentration.ndim == 1:
        conc_post = concentration[pre_bolus_frames:]
        peak_idx = int(np.argmax(conc_post))
        ttp = np.float64((pre_bolus_frames + peak_idx) * dt)
    else:
        raise ValueError(f"Concentration must be 1D or 4D, got {concentration.ndim}D")
    return ttp  # type: ignore[no-any-return]


def find_nawm_mask(
    derivatives_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
) -> Optional[Path]:
    """Find a NAWM (normal appearing white matter) mask in derivatives.

    Searches for HD-GLIO or other NAWM ROI outputs.

    Parameters
    ----------
    derivatives_dir : str or Path
        Path to the derivatives directory.
    subject : str
        Subject ID (without 'sub-' prefix).
    session : str, optional
        Session ID (without 'ses-' prefix), optional.

    Returns
    -------
    Path or None
        Path to the NAWM mask, or None if not found.
    """
    deriv_path = Path(derivatives_dir)

    # Build search patterns
    patterns: List[str] = []

    base_pattern = f"sub-{subject}"
    if session:
        base_pattern += f"_ses-{session}"

    # Common NAWM mask naming patterns
    nawm_keywords = ["nawm", "NAWM", "normalwm", "normwm", "contrawm"]

    for keyword in nawm_keywords:
        patterns.extend([
            f"{base_pattern}*{keyword}*_mask.nii.gz",
            f"{base_pattern}*desc-{keyword}*_mask.nii.gz",
            f"{base_pattern}*label-{keyword}*_mask.nii.gz",
        ])

    # Also look for white matter probability maps as fallback
    patterns.extend([
        f"{base_pattern}*label-WM*_probseg.nii.gz",
        f"{base_pattern}*desc-wm*_probseg.nii.gz",
    ])

    # Search in derivatives
    for pattern in patterns:
        matches = list(deriv_path.rglob(pattern))
        if matches:
            # Prefer 'nawm' or 'contra' (contralateral) masks
            for m in matches:
                name_lower = m.name.lower()
                if "nawm" in name_lower or "contra" in name_lower:
                    return m
            return matches[0]

    return None


def normalize_cbv(
    cbv_map: npt.NDArray[np.floating],
    nawm_mask: Optional[npt.NDArray[np.bool_]] = None,
    nawm_mask_file: Optional[Union[str, Path]] = None,
    derivatives_dir: Optional[Union[str, Path]] = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    brain_mask: Optional[npt.NDArray[np.bool_]] = None,
    return_normalization_factor: bool = False,
) -> Union[
    npt.NDArray[np.floating],
    Tuple[npt.NDArray[np.floating], float],
]:
    """Normalize CBV map to NAWM median (nCBV).

    Computes normalized CBV (nCBV) by dividing the CBV map by the median
    CBV value within the normal appearing white matter (NAWM) region.

    nCBV = CBV / median(CBV_NAWM)

    The NAWM mask can be provided directly, loaded from a file path, or
    auto-detected from the derivatives directory (HD-GLIO outputs).

    Parameters
    ----------
    cbv_map : np.ndarray
        3D array of CBV values to normalize.
    nawm_mask : np.ndarray, optional
        Binary 3D mask of NAWM region. If None, will attempt
        to load from nawm_mask_file or auto-detect.
    nawm_mask_file : str or Path, optional
        Path to a NAWM mask NIfTI file. Used if nawm_mask
        is not provided.
    derivatives_dir : str or Path, optional
        Path to derivatives directory for auto-detection.
        Used if neither nawm_mask nor nawm_mask_file is provided.
    subject : str, optional
        Subject ID (without 'sub-' prefix) for auto-detection.
    session : str, optional
        Session ID (without 'ses-' prefix) for auto-detection.
    brain_mask : np.ndarray, optional
        Optional brain mask for fallback normalization if NAWM
        mask is unavailable. Uses whole-brain median.
    return_normalization_factor : bool, optional
        If True, returns tuple of (nCBV, factor).

    Returns
    -------
    np.ndarray or tuple
        If return_normalization_factor is False:
            Normalized CBV map (nCBV) as a 3D array.
        If return_normalization_factor is True:
            Tuple of (nCBV map, normalization factor).

    Raises
    ------
        ValueError: If CBV map is not 3D.
        ValueError: If mask dimensions don't match CBV map.

    Notes
    -----
        - If no NAWM mask is available and no brain mask is provided,
          the function returns the original CBV map unchanged with
          a normalization factor of 1.0.
        - nCBV is unitless and is approximately 1.0 in the normalization
          reference by construction.
    """
    # Validate CBV map
    if cbv_map.ndim != 3:
        raise ValueError(f"CBV map must be 3D, got {cbv_map.ndim}D")

    # Track if we have a valid NAWM mask
    effective_mask: Optional[npt.NDArray[np.bool_]] = None

    # Priority 1: Use provided NAWM mask directly
    if nawm_mask is not None:
        if nawm_mask.shape != cbv_map.shape:
            raise ValueError(
                f"NAWM mask shape {nawm_mask.shape} doesn't match "
                f"CBV map shape {cbv_map.shape}"
            )
        effective_mask = nawm_mask

    # Priority 2: Load from provided file path
    if effective_mask is None and nawm_mask_file is not None:
        nawm_path = Path(nawm_mask_file)
        if nawm_path.exists():
            import nibabel as nib  # type: ignore[import-untyped]

            nawm_img = nib.load(nawm_path)
            nawm_data = nawm_img.get_fdata()  # type: ignore[attr-defined]
            # Convert to boolean mask (threshold at 0.5 for probability maps)
            effective_mask = nawm_data > 0.5
            if effective_mask.shape != cbv_map.shape:
                raise ValueError(
                    f"NAWM mask shape {effective_mask.shape} doesn't match "
                    f"perfusion map shape {cbv_map.shape}"
                )

    # Priority 3: Auto-detect from derivatives
    if (
        effective_mask is None
        and derivatives_dir is not None
        and subject is not None
    ):
        found_mask_path = find_nawm_mask(derivatives_dir, subject, session)
        if found_mask_path is not None:
            import nibabel as nib  # type: ignore[import-untyped]

            nawm_img = nib.load(found_mask_path)
            nawm_data = nawm_img.get_fdata()  # type: ignore[attr-defined]
            effective_mask = nawm_data > 0.5
            if effective_mask.shape != cbv_map.shape:
                raise ValueError(
                    f"Auto-detected NAWM mask shape {effective_mask.shape} "
                    f"doesn't match perfusion map shape {cbv_map.shape}"
                )

    # Compute normalization factor
    # An explicitly requested tissue reference must yield a valid factor.
    # Keep the historical identity behavior only when no reference was supplied.
    normalization_factor = (
        float("nan")
        if effective_mask is not None or brain_mask is not None
        else 1.0
    )

    if effective_mask is not None:
        # Ensure mask is boolean
        mask_bool = effective_mask.astype(bool)

        # Count voxels in NAWM region
        n_voxels = int(np.sum(mask_bool))

        if n_voxels > 0:
            # Get CBV values within NAWM
            cbv_nawm = cbv_map[mask_bool]

            # Production hemodynamic maps encode non-physical negative
            # estimates as NaN, so only their finite values define the
            # normalization reference.
            cbv_nawm_valid = cbv_nawm[np.isfinite(cbv_nawm)]

            if len(cbv_nawm_valid) > 0:
                # Compute median CBV in NAWM
                median_cbv_nawm = float(np.median(cbv_nawm_valid))

                # Ensure non-zero normalization factor
                if median_cbv_nawm > EPSILON_CONCENTRATION:
                    normalization_factor = median_cbv_nawm

    # Fallback: Use brain mask for whole-brain normalization
    if not np.isfinite(normalization_factor) and brain_mask is not None:
        if brain_mask.shape == cbv_map.shape:
            brain_bool = brain_mask.astype(bool)
            n_brain_voxels = int(np.sum(brain_bool))

            if n_brain_voxels > 0:
                cbv_brain = cbv_map[brain_bool]
                cbv_brain_valid = cbv_brain[np.isfinite(cbv_brain)]

                if len(cbv_brain_valid) > 0:
                    # Use median of whole brain as fallback
                    median_cbv_brain = float(np.median(cbv_brain_valid))

                    if median_cbv_brain > EPSILON_CONCENTRATION:
                        normalization_factor = median_cbv_brain

    # An invalid explicit reference is undefined, never an identity transform
    # mislabeled as tissue-normalized.
    if np.isfinite(normalization_factor) and normalization_factor > 0.0:
        ncbv_map = cbv_map / normalization_factor
    else:
        ncbv_map = np.full(cbv_map.shape, np.nan, dtype=np.float64)

    if return_normalization_factor:
        return ncbv_map, normalization_factor
    else:
        return ncbv_map


def normalize_cbv_from_files(
    cbv_file: Union[str, Path],
    output_file: Union[str, Path],
    nawm_mask_file: Optional[Union[str, Path]] = None,
    derivatives_dir: Optional[Union[str, Path]] = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    brain_mask_file: Optional[Union[str, Path]] = None,
) -> Tuple[Path, float]:
    """Normalize a CBV NIfTI file and save the result.

    Convenience wrapper that loads the CBV map from a file, performs
    NAWM normalization, and saves the result.

    Parameters
    ----------
    cbv_file : str or Path
        Path to input CBV NIfTI file.
    output_file : str or Path
        Path to save normalized CBV NIfTI.
    nawm_mask_file : str or Path, optional
        Path to NAWM mask NIfTI file.
    derivatives_dir : str or Path, optional
        Path to derivatives for auto-detection.
    subject : str, optional
        Subject ID for auto-detection.
    session : str, optional
        Session ID for auto-detection.
    brain_mask_file : str or Path, optional
        Path to brain mask for fallback normalization.

    Returns
    -------
    tuple
        Tuple of (output_file path, normalization_factor).

    Raises
    ------
        FileNotFoundError: If CBV file does not exist.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    cbv_path = Path(cbv_file)
    if not cbv_path.exists():
        raise FileNotFoundError(f"CBV file not found: {cbv_path}")

    # Load CBV map
    cbv_img = nib.load(cbv_path)
    cbv_data = cbv_img.get_fdata()  # type: ignore[attr-defined]

    if nawm_mask_file is not None:
        nawm_path = Path(nawm_mask_file)
        if nawm_path.exists():
            nawm_img = nib.load(nawm_path)
            if not np.allclose(nawm_img.affine, cbv_img.affine):  # type: ignore[attr-defined]
                raise ValueError("NAWM mask affine does not match perfusion map affine")

    # Load brain mask if provided
    brain_mask: Optional[npt.NDArray[np.bool_]] = None
    if brain_mask_file is not None:
        brain_path = Path(brain_mask_file)
        if brain_path.exists():
            brain_img = nib.load(brain_path)
            brain_mask = brain_img.get_fdata() > 0.5  # type: ignore[attr-defined]

    # Normalize
    ncbv_data, norm_factor = normalize_cbv(
        cbv_map=cbv_data,
        nawm_mask=None,
        nawm_mask_file=nawm_mask_file,
        derivatives_dir=derivatives_dir,
        subject=subject,
        session=session,
        brain_mask=brain_mask,
        return_normalization_factor=True,
    )

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncbv_img = nib.Nifti1Image(ncbv_data, cbv_img.affine, cbv_img.header)  # type: ignore[attr-defined]
    nib.save(ncbv_img, output_path)

    return output_path, norm_factor
