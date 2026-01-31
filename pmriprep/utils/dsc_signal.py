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
    R2_STAR_BY_FIELD,
    get_r2_star,
)


def signal_to_concentration(
    signal: npt.NDArray[np.floating],
    te: float,
    baseline_frames: int = 5,
    r2_star: Optional[float] = None,
    field_strength: Optional[float] = None,
) -> npt.NDArray[np.floating]:
    """Convert DSC signal to concentration via ΔR2*.

    Computes concentration from the signal decay using the relationship:
        ΔR2* = -ln(S(t)/S0) / TE
        C(t) = ΔR2* / r2*  (if absolute quantification)

    In relative mode (no r2* provided), returns ΔR2* directly.
    In absolute mode (r2* provided), returns concentration in mM.

    Args:
        signal: 4D array of DSC signal (x, y, z, time) or 1D time series.
            Signal values should be positive.
        te: Echo time in seconds. Typical values: 0.025-0.040s.
        baseline_frames: Number of pre-bolus frames for S0 estimation.
            Default is 5 frames.
        r2_star: Transverse relaxivity in s^-1 per mM for absolute
            quantification. If None, returns relative ΔR2*.
        field_strength: Magnetic field strength in Tesla. Used to look up
            r2* if r2_star is not provided but absolute quantification
            is desired. Common values: 1.5, 3.0, 7.0.

    Returns:
        Concentration array with same shape as input signal.
        Units are s^-1 (ΔR2*) for relative mode or mM for absolute mode.

    Raises:
        ValueError: If signal array is empty or has invalid dimensions.
        ValueError: If te is not positive.
        ValueError: If baseline_frames exceeds signal length.

    Notes:
        - Signal ratio is clipped to (EPSILON_SIGNAL, 1.0) before log
          to prevent numerical issues with zero or negative values.
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

    # Clip signal ratio to (EPSILON_SIGNAL, 1.0) before log
    # Upper bound of 1.0 prevents log of values > 1 (which would give negative ΔR2*)
    # Lower bound prevents log(0) or log(negative)
    signal_ratio_clipped = np.clip(signal_ratio, EPSILON_SIGNAL, 1.0)

    # Compute ΔR2* = -ln(S/S0) / TE
    # Note: negative of log because signal decreases with contrast
    delta_r2_star = -np.log(signal_ratio_clipped) / te

    # Determine r2* for conversion
    effective_r2_star: Optional[float] = None

    if r2_star is not None:
        effective_r2_star = r2_star
    elif field_strength is not None:
        effective_r2_star = get_r2_star(field_strength)

    # Return relative ΔR2* or absolute concentration
    if effective_r2_star is not None:
        # Absolute quantification: C = ΔR2* / r2*
        concentration = delta_r2_star / effective_r2_star
    else:
        # Relative mode: return ΔR2* directly (units: s^-1)
        concentration = delta_r2_star

    return concentration


def signal_to_delta_r2_star(
    signal: npt.NDArray[np.floating],
    te: float,
    baseline_frames: int = 5,
) -> npt.NDArray[np.floating]:
    """Convert DSC signal to ΔR2* (relative concentration).

    Convenience wrapper for signal_to_concentration with relative mode.

    Args:
        signal: 4D array of DSC signal (x, y, z, time) or 1D time series.
        te: Echo time in seconds.
        baseline_frames: Number of pre-bolus frames for S0 estimation.

    Returns:
        ΔR2* array with same shape as input signal, units s^-1.
    """
    return signal_to_concentration(
        signal=signal,
        te=te,
        baseline_frames=baseline_frames,
        r2_star=None,
        field_strength=None,
    )


def get_baseline_signal(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 5,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute baseline signal S0 from pre-bolus frames.

    Args:
        signal: DSC signal array (1D, 3D, or 4D).
        baseline_frames: Number of pre-bolus frames to average.

    Returns:
        Baseline signal value (scalar for 1D) or array (for 3D/4D).
    """
    if signal.ndim == 1:
        return float(np.mean(signal[:baseline_frames]))
    elif signal.ndim == 4:
        return np.mean(signal[..., :baseline_frames], axis=-1)
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

    Args:
        signal: 1D DSC signal time course.
        baseline_frames: Number of pre-bolus frames.
        threshold_factor: Number of baseline standard deviations
            for threshold (default 3.0).

    Returns:
        Index of bolus arrival frame.

    Raises:
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
) -> Tuple[int, int]:
    """Detect bolus arrival in a concentration curve.

    Uses a 3-sigma threshold above the initial baseline to find the first
    frame with significant contrast arrival, then returns both the arrival
    index and the number of usable baseline frames.

    Ported from perf-plus perfusion_utils.detect_bolus_arrival().

    Args:
        concentration_curve: 1D concentration time course (positive values
            indicate contrast arrival in concentration space).
        baseline_frames: Number of initial frames for baseline estimation.
            Default is 3 to avoid early bolus contamination.

    Returns:
        Tuple of (arrival_idx, n_baseline_frames):
            - arrival_idx: Frame index where bolus arrives
            - n_baseline_frames: Number of usable pre-bolus frames
              (always >= 2)
    """
    n_baseline = min(baseline_frames, len(concentration_curve) // 4)
    n_baseline = max(2, n_baseline)
    baseline = np.mean(concentration_curve[:n_baseline])
    baseline_std = np.std(concentration_curve[:n_baseline])
    threshold = baseline + 3 * baseline_std
    arrival_indices = np.where(concentration_curve > threshold)[0]
    if len(arrival_indices) > 0:
        arrival_idx = int(arrival_indices[0])
    else:
        arrival_idx = 5  # fallback: 5 frames if no clear arrival
    return arrival_idx, max(2, arrival_idx)


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

    Args:
        signal: DSC signal array. Can be:
            - 1D (time,): Single voxel time course
            - 4D (x, y, z, time): Full 4D volume
        baseline_frames: Number of pre-bolus frames for S_baseline.
            Default is 10.
        tail_frames: Number of post-bolus frames for S_tail.
            Default is 10 (last 10 frames).

    Returns:
        PSR value(s) as percentage (0-100+ range).
        - Scalar float for 1D input
        - 3D array (x, y, z) for 4D input

    Raises:
        ValueError: If signal has invalid dimensions.
        ValueError: If baseline_frames or tail_frames exceeds signal length.

    Notes:
        - This is the physics-correct version for post-integration expert
          review. The legacy version (compute_psr_legacy) matches perf-plus
          ground truth and uses the Cha et al. (AJNR 2002) formula.
        - TODO(physics-review): s_min currently uses the entire timeseries.
          The physics expert recommends using post-baseline frames only
          (signal[..., baseline_frames:]) to avoid baseline contamination,
          matching the perf-plus convention.
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
    if signal.ndim == 1:
        # 1D time series
        s_baseline = float(np.mean(signal[:baseline_frames]))
        s_min = float(np.min(signal))
        s_tail = float(np.mean(signal[-tail_frames:]))
    else:
        # 4D volume (x, y, z, time)
        s_baseline = np.mean(signal[..., :baseline_frames], axis=-1)
        s_min = np.min(signal, axis=-1)
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
        return psr


def compute_psr_legacy(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 5,
    tail_frames: int = 5,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute PSR using the perf-plus legacy formula.

    Legacy formula (Cha et al., AJNR 2002):
        PSR = 100 × (S1 - Smin) / (S0 - Smin)
    where S0 = mean baseline signal, S1 = mean tail signal,
    Smin = minimum signal in post-baseline frames.

    Returns percentage (~0-100+ range). Kept for integration test
    compatibility with perf-plus ground truth.

    Args:
        signal: DSC signal array (1D or 4D).
        baseline_frames: Number of pre-bolus frames for S0.
        tail_frames: Number of post-bolus tail frames for S1 (recovery).

    Returns:
        PSR as percentage (0-100+ range).
    """
    if signal.ndim == 1:
        s0 = float(np.mean(signal[:baseline_frames]))
        s_min = float(np.min(signal[baseline_frames:]))
        s1 = float(np.mean(signal[-tail_frames:]))
        denom = s0 - s_min
        denom_safe = max(abs(denom), EPSILON_SIGNAL)
        if denom < 0:
            denom_safe = -denom_safe
        return 100.0 * (s1 - s_min) / denom_safe
    elif signal.ndim == 4:
        s0 = np.mean(signal[..., :baseline_frames], axis=-1)
        s_min = np.min(signal[..., baseline_frames:], axis=-1)
        s1 = np.mean(signal[..., -tail_frames:], axis=-1)
        denom = s0 - s_min
        denom_safe = np.maximum(np.abs(denom), EPSILON_SIGNAL)
        denom_safe = np.where(denom >= 0, denom_safe, -denom_safe)
        return 100.0 * (s1 - s_min) / denom_safe
    else:
        raise ValueError(f"Signal must be 1D or 4D, got {signal.ndim}D")


def compute_peak_height(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 10,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute peak height (PH) of DSC signal.

    PH quantifies the maximum signal drop during bolus passage,
    reflecting contrast agent concentration and vascular density.

    Formula:
        PH = 100 × (S_baseline - S_min) / S_baseline

    Clinical interpretation:
        - Higher PH indicates greater signal drop (more contrast effect)
        - PH correlates with cerebral blood volume (CBV)
        - Typical values: 20-60% in normal tissue, may be higher in tumors

    Args:
        signal: DSC signal array. Can be:
            - 1D (time,): Single voxel time course
            - 4D (x, y, z, time): Full 4D volume
        baseline_frames: Number of pre-bolus frames for S_baseline.
            Default is 10.

    Returns:
        Peak height value(s) as percentage (0-100 range typically).
        - Scalar float for 1D input
        - 3D array (x, y, z) for 4D input

    Raises:
        ValueError: If signal has invalid dimensions.
        ValueError: If baseline_frames exceeds signal length.

    Notes:
        - This is the physics-correct version for post-integration expert
          review. The legacy version (compute_peak_height_legacy) matches
          perf-plus ground truth.
        - TODO(physics-review): s_min currently uses the entire timeseries
          (lines using np.min(signal) / np.min(signal, axis=-1)). The
          physics expert recommends using post-baseline frames only
          (signal[..., baseline_frames:]) to avoid baseline contamination,
          matching the perf-plus convention.
        - PH is always positive when computed from valid DSC signal
          (signal drops during bolus passage).
        - PH can exceed 100% if signal drops below zero (artifact).
        - Epsilon is added to baseline to prevent division by zero.
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

    if baseline_frames < 1:
        raise ValueError("baseline_frames must be at least 1")

    # Compute signal components
    if signal.ndim == 1:
        # 1D time series
        s_baseline = float(np.mean(signal[:baseline_frames]))
        s_min = float(np.min(signal))
    else:
        # 4D volume (x, y, z, time)
        s_baseline = np.mean(signal[..., :baseline_frames], axis=-1)
        s_min = np.min(signal, axis=-1)

    # Compute PH with epsilon guard in denominator
    s_baseline_safe = np.maximum(np.abs(s_baseline), EPSILON_SIGNAL)
    # Preserve sign of baseline
    s_baseline_safe = np.where(
        s_baseline >= 0, s_baseline_safe, -s_baseline_safe
    )

    ph = 100.0 * (s_baseline - s_min) / s_baseline_safe

    # Return appropriate type
    if signal.ndim == 1:
        return float(ph)
    else:
        return ph


def compute_peak_height_legacy(
    signal: npt.NDArray[np.floating],
    baseline_frames: int = 5,
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute peak height using the perf-plus legacy formula.

    Legacy formula (standard % signal drop):
        PH = 100 × (S0 - Smin) / S0
    where S0 = mean baseline signal and Smin = minimum signal in
    post-baseline frames only.

    Returns percentage (0-100 range). Kept for integration test
    compatibility with perf-plus ground truth.

    Args:
        signal: DSC signal array (1D or 4D).
        baseline_frames: Number of pre-bolus frames for S0.

    Returns:
        Peak height as percentage (0-100 range).
    """
    if signal.ndim == 1:
        s0 = float(np.mean(signal[:baseline_frames]))
        s_min = float(np.min(signal[baseline_frames:]))
        s0_safe = max(abs(s0), EPSILON_SIGNAL)
        if s0 >= 0:
            return 100.0 * (s0 - s_min) / s0_safe
        else:
            return 100.0 * (s0 - s_min) / (-s0_safe)
    elif signal.ndim == 4:
        s0 = np.mean(signal[..., :baseline_frames], axis=-1)
        s_min = np.min(signal[..., baseline_frames:], axis=-1)
        s0_safe = np.maximum(np.abs(s0), EPSILON_SIGNAL)
        s0_safe = np.where(s0 >= 0, s0_safe, -s0_safe)
        return 100.0 * (s0 - s_min) / s0_safe
    else:
        raise ValueError(f"Signal must be 1D or 4D, got {signal.ndim}D")


def compute_peak_height_conc(
    signal: npt.NDArray[np.floating],
    te_seconds: float,
    baseline_end: int,
) -> npt.NDArray[np.floating]:
    """Compute peak height in concentration domain (ideal method).

    PH = max(delta_R2*) in s^-1 units. Linear with contrast agent
    concentration, TE-independent, compatible with K2 correction.

    Args:
        signal: DSC signal array (4D: x,y,z,time).
        te_seconds: Echo time in seconds.
        baseline_end: Index of end of baseline period.

    Returns:
        Peak height in s^-1 units.
    """
    s0 = np.mean(signal[..., :baseline_end], axis=-1, keepdims=True)
    s0_safe = np.maximum(s0, EPSILON_SIGNAL)
    s_ratio = np.clip(signal / s0_safe, EPSILON_SIGNAL, None)
    delta_r2star = -np.log(s_ratio) / te_seconds
    ph = np.max(delta_r2star, axis=-1)
    return np.clip(ph, 0.0, None)


def compute_ttp_legacy(
    concentration: npt.NDArray[np.floating],
    pre_bolus_frames: int,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Compute TTP using the perf-plus legacy method.

    Finds time to peak concentration using unsmoothed post-bolus argmax,
    clipping to [0, 60] seconds. Matches perf-plus ground truth.

    Args:
        concentration: Concentration array (4D: x,y,z,time or 1D: time).
        pre_bolus_frames: Number of pre-bolus baseline frames to skip.
        dt: Time step in seconds (TR).

    Returns:
        TTP map in seconds, clipped to [0, 60].
    """
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
    return np.clip(ttp, 0.0, 60.0)


def compute_ttp(
    concentration: npt.NDArray[np.floating],
    pre_bolus_frames: int,
    dt: float,
    sigma: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Compute TTP using Gaussian-smoothed post-bolus argmax (ideal method).

    Applies Gaussian smoothing (sigma=1 sample, FWHM ~3.5s) to suppress
    single-frame noise spikes before finding peak. NaN for out-of-range values.

    Args:
        concentration: Concentration array (4D: x,y,z,time or 1D: time).
        pre_bolus_frames: Number of pre-bolus baseline frames to skip.
        dt: Time step in seconds (TR).
        sigma: Gaussian smoothing sigma in samples (default 1.0).

    Returns:
        TTP map in seconds. Out-of-range values set to NaN.
    """
    from scipy.ndimage import gaussian_filter1d

    conc_smooth = gaussian_filter1d(concentration, sigma=sigma, axis=-1)
    if conc_smooth.ndim == 4:
        conc_post = conc_smooth[..., pre_bolus_frames:]
        peak_idx = np.argmax(conc_post, axis=-1)
        ttp = (pre_bolus_frames + peak_idx).astype(np.float64) * dt
    elif conc_smooth.ndim == 1:
        conc_post = conc_smooth[pre_bolus_frames:]
        peak_idx = int(np.argmax(conc_post))
        ttp = np.float64((pre_bolus_frames + peak_idx) * dt)
    else:
        raise ValueError(f"Concentration must be 1D or 4D, got {concentration.ndim}D")
    # NaN for out-of-range (preserves data quality info)
    ttp_arr = np.asarray(ttp, dtype=np.float64)
    ttp_arr[(ttp_arr < pre_bolus_frames * dt) | (ttp_arr > 60.0)] = np.nan
    return ttp_arr


def find_nawm_mask(
    derivatives_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
) -> Optional[Path]:
    """Find a NAWM (normal appearing white matter) mask in derivatives.

    Searches for HD-GLIO or other NAWM ROI outputs.

    Args:
        derivatives_dir: Path to the derivatives directory.
        subject: Subject ID (without 'sub-' prefix).
        session: Session ID (without 'ses-' prefix), optional.

    Returns:
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

    Args:
        cbv_map: 3D array of CBV values to normalize.
        nawm_mask: Binary 3D mask of NAWM region. If None, will attempt
            to load from nawm_mask_file or auto-detect.
        nawm_mask_file: Path to a NAWM mask NIfTI file. Used if nawm_mask
            is not provided.
        derivatives_dir: Path to derivatives directory for auto-detection.
            Used if neither nawm_mask nor nawm_mask_file is provided.
        subject: Subject ID (without 'sub-' prefix) for auto-detection.
        session: Session ID (without 'ses-' prefix) for auto-detection.
        brain_mask: Optional brain mask for fallback normalization if NAWM
            mask is unavailable. Uses whole-brain median.
        return_normalization_factor: If True, returns tuple of (nCBV, factor).

    Returns:
        If return_normalization_factor is False:
            Normalized CBV map (nCBV) as a 3D array.
        If return_normalization_factor is True:
            Tuple of (nCBV map, normalization factor).

    Raises:
        ValueError: If CBV map is not 3D.
        ValueError: If mask dimensions don't match CBV map.

    Notes:
        - If no NAWM mask is available and no brain mask is provided,
          the function returns the original CBV map unchanged with
          a normalization factor of 1.0.
        - nCBV is unitless (ratio), typically ~1.0 in NAWM and elevated
          (>1.7-2.0) in tumor regions.
        - In glioblastoma, nCBV > 1.75 often indicates tumor vs. edema.
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
                # Silently fail if dimensions don't match
                effective_mask = None

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
                effective_mask = None

    # Compute normalization factor
    normalization_factor: float = 1.0

    if effective_mask is not None:
        # Ensure mask is boolean
        mask_bool = effective_mask.astype(bool)

        # Count voxels in NAWM region
        n_voxels = int(np.sum(mask_bool))

        if n_voxels > 0:
            # Get CBV values within NAWM
            cbv_nawm = cbv_map[mask_bool]

            # Filter out non-positive values for median calculation
            cbv_nawm_valid = cbv_nawm[cbv_nawm > EPSILON_CONCENTRATION]

            if len(cbv_nawm_valid) > 0:
                # Compute median CBV in NAWM
                median_cbv_nawm = float(np.median(cbv_nawm_valid))

                # Ensure non-zero normalization factor
                if median_cbv_nawm > EPSILON_CONCENTRATION:
                    normalization_factor = median_cbv_nawm

    # Fallback: Use brain mask for whole-brain normalization
    if normalization_factor == 1.0 and brain_mask is not None:
        if brain_mask.shape == cbv_map.shape:
            brain_bool = brain_mask.astype(bool)
            n_brain_voxels = int(np.sum(brain_bool))

            if n_brain_voxels > 0:
                cbv_brain = cbv_map[brain_bool]
                cbv_brain_valid = cbv_brain[cbv_brain > EPSILON_CONCENTRATION]

                if len(cbv_brain_valid) > 0:
                    # Use median of whole brain as fallback
                    median_cbv_brain = float(np.median(cbv_brain_valid))

                    if median_cbv_brain > EPSILON_CONCENTRATION:
                        normalization_factor = median_cbv_brain

    # Compute normalized CBV
    ncbv_map = cbv_map / normalization_factor

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

    Args:
        cbv_file: Path to input CBV NIfTI file.
        output_file: Path to save normalized CBV NIfTI.
        nawm_mask_file: Path to NAWM mask NIfTI file.
        derivatives_dir: Path to derivatives for auto-detection.
        subject: Subject ID for auto-detection.
        session: Session ID for auto-detection.
        brain_mask_file: Path to brain mask for fallback normalization.

    Returns:
        Tuple of (output_file path, normalization_factor).

    Raises:
        FileNotFoundError: If CBV file does not exist.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    cbv_path = Path(cbv_file)
    if not cbv_path.exists():
        raise FileNotFoundError(f"CBV file not found: {cbv_path}")

    # Load CBV map
    cbv_img = nib.load(cbv_path)
    cbv_data = cbv_img.get_fdata()  # type: ignore[attr-defined]

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
