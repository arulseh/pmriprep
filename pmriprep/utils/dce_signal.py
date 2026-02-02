"""DCE signal processing utilities.

Functions for converting DCE-MRI signal to concentration using the Ernst
equation and related T1-based calculations for pharmacokinetic analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from pmriprep.utils.dce_constants import (
    DEFAULT_BASELINE_FRAMES,
    DEFAULT_T1_MS,
    EPSILON_B1,
    EPSILON_E1_UPPER,
    EPSILON_SIGNAL,
    EPSILON_T1,
    HEMATOCRIT,
    get_r1_relaxivity,
)


@dataclass
class DCEConversionResult:
    """Result of DCE signal to concentration conversion.

    Attributes:
        concentration: Concentration array in mM.
        r1_dynamic: Dynamic R1 values (1/T1) in s^-1.
        r1_baseline: Baseline R1 value (1/T1_pre) in s^-1.
        is_plasma: Whether concentration is plasma (vs whole blood).
        effective_r1: The r1 relaxivity used for conversion (s^-1/mM).
        baseline_signal: Mean baseline signal used for normalization.
    """

    concentration: npt.NDArray[np.floating]
    r1_dynamic: npt.NDArray[np.floating]
    r1_baseline: Union[float, npt.NDArray[np.floating]]
    is_plasma: bool
    effective_r1: float
    baseline_signal: Union[float, npt.NDArray[np.floating]]


def ernst_signal_model(
    s0: Union[float, npt.NDArray[np.floating]],
    flip_angle_rad: float,
    tr: float,
    t1: Union[float, npt.NDArray[np.floating]],
) -> Union[float, npt.NDArray[np.floating]]:
    """Compute MRI signal using the Ernst equation.

    The Ernst equation describes the steady-state signal for a spoiled
    gradient echo sequence:

        S = S0 × sin(α) × (1 - E1) / (1 - cos(α) × E1)

    where E1 = exp(-TR/T1)

    Args:
        s0: Proton density / fully relaxed signal.
        flip_angle_rad: Flip angle in radians.
        tr: Repetition time in seconds.
        t1: T1 relaxation time in seconds.

    Returns:
        Computed signal intensity.

    Notes:
        - Input T1 should be positive. Values <= 0 will produce NaN.
        - Typical flip angles are 5-30 degrees (0.087-0.524 radians).
        - The Ernst angle (optimal flip angle) = arccos(exp(-TR/T1)).
    """
    # Compute E1 = exp(-TR/T1) with T1 safety guard
    t1_safe = np.maximum(t1, EPSILON_T1)
    e1 = np.exp(-tr / t1_safe)

    # Ernst equation
    sin_alpha = np.sin(flip_angle_rad)
    cos_alpha = np.cos(flip_angle_rad)

    signal = s0 * sin_alpha * (1.0 - e1) / (1.0 - cos_alpha * e1)

    return signal


def solve_e1_from_signal_ratio(
    signal_ratio: Union[float, npt.NDArray[np.floating]],
    flip_angle_rad: Union[float, npt.NDArray[np.floating]],
    e1_baseline: Union[float, npt.NDArray[np.floating]],
) -> Union[float, npt.NDArray[np.floating]]:
    """Solve for E1(t) from the signal ratio S(t)/S(baseline).

    Given the Ernst equation ratio:
        S(t)/S(0) = [(1 - E1(t)) / (1 - cos(α) × E1(t))] /
                   [(1 - E1(0)) / (1 - cos(α) × E1(0))]

    This function solves for E1(t) given the signal ratio and baseline E1.

    Args:
        signal_ratio: S(t) / S(baseline).
        flip_angle_rad: Flip angle in radians. Can be scalar or per-voxel array.
        e1_baseline: Baseline E1 value = exp(-TR/T1_baseline).

    Returns:
        E1(t) values corresponding to the signal ratio.

    Notes:
        - This is an analytical solution derived from the Ernst equation.
        - E1 values are clipped to (0, 1) to ensure physical validity.
    """
    cos_alpha = np.cos(flip_angle_rad)

    # Pre-compute baseline terms
    # S(baseline) ∝ (1 - E1_baseline) / (1 - cos_alpha * E1_baseline)
    baseline_num = 1.0 - e1_baseline
    baseline_den = 1.0 - cos_alpha * e1_baseline

    # Signal ratio = [(1 - E1) / (1 - cos_alpha * E1)] / [baseline_num / baseline_den]
    # Let R = signal_ratio * baseline_num / baseline_den
    # Then: R = (1 - E1) / (1 - cos_alpha * E1)
    # R * (1 - cos_alpha * E1) = 1 - E1
    # R - R * cos_alpha * E1 = 1 - E1
    # E1 - R * cos_alpha * E1 = 1 - R
    # E1 * (1 - R * cos_alpha) = 1 - R
    # E1 = (1 - R) / (1 - R * cos_alpha)

    # Prevent division by zero in baseline_den
    baseline_den_safe = np.where(
        np.abs(baseline_den) < EPSILON_T1,
        np.sign(baseline_den) * EPSILON_T1 + (baseline_den == 0) * EPSILON_T1,
        baseline_den,
    )

    r = signal_ratio * baseline_num / baseline_den_safe

    # Compute E1
    denominator = 1.0 - r * cos_alpha

    # Prevent division by zero
    denominator_safe = np.where(
        np.abs(denominator) < EPSILON_T1,
        np.sign(denominator) * EPSILON_T1 + (denominator == 0) * EPSILON_T1,
        denominator,
    )

    e1 = (1.0 - r) / denominator_safe

    # Clip E1 to valid range (0, 1) since E1 = exp(-TR/T1)
    # E1 must be positive (T1 > 0) and less than 1 (TR > 0)
    # Use EPSILON_E1_UPPER (1e-4) for upper bound to preserve long-T1 voxels
    e1_clipped = np.clip(e1, EPSILON_T1, 1.0 - EPSILON_E1_UPPER)

    return e1_clipped


def e1_to_r1(
    e1: Union[float, npt.NDArray[np.floating]],
    tr: float,
) -> Union[float, npt.NDArray[np.floating]]:
    """Convert E1 to R1 (1/T1).

    E1 = exp(-TR/T1) = exp(-TR × R1)
    => R1 = -ln(E1) / TR

    Args:
        e1: E1 values, where E1 = exp(-TR/T1).
        tr: Repetition time in seconds.

    Returns:
        R1 values in s^-1 (reciprocal of T1).

    Notes:
        - E1 must be in range (0, 1) for valid R1.
        - R1 = 1/T1, so typical values are 0.5-2.0 s^-1 for tissue.
    """
    # Ensure E1 is in valid range for log
    e1_safe = np.clip(e1, EPSILON_T1, 1.0 - EPSILON_E1_UPPER)

    r1 = -np.log(e1_safe) / tr

    return r1


def signal_to_concentration(
    signal: npt.NDArray[np.floating],
    flip_angle_deg: float,
    tr: float,
    t1_map: Optional[npt.NDArray[np.floating]] = None,
    t1_default_ms: float = DEFAULT_T1_MS,
    b1_map: Optional[npt.NDArray[np.floating]] = None,
    r1_relaxivity: Optional[float] = None,
    field_strength: float = 3.0,
    baseline_frames: int = DEFAULT_BASELINE_FRAMES,
    convert_to_plasma: bool = True,
    hematocrit: float = HEMATOCRIT,
) -> DCEConversionResult:
    """Convert DCE signal to concentration using the Ernst equation.

    This function implements the standard DCE-MRI signal to concentration
    conversion using the SPGR/Ernst equation approach:

    1. Compute baseline signal S0 and E1_baseline from T1 map
    2. For each timepoint, solve for E1(t) from signal ratio
    3. Convert E1(t) to R1(t) = -ln(E1)/TR
    4. Compute concentration: C(t) = (R1(t) - R1(0)) / r1
    5. Optionally convert to plasma concentration

    Args:
        signal: 4D DCE signal array (x, y, z, time) or 1D time series.
        flip_angle_deg: Flip angle in degrees. Typical values: 10-30°.
        tr: Repetition time in seconds. Typical values: 0.003-0.010s.
        t1_map: 3D pre-contrast T1 map in milliseconds. If None, uses
            t1_default_ms for all voxels.
        t1_default_ms: Default T1 value in milliseconds when t1_map is None.
            Default is 1200ms (typical gray matter at 3T).
        b1_map: Optional 3D B1+ correction map (relative flip angle).
            Values typically 0.8-1.2. If provided, actual flip angle =
            nominal flip angle × B1+.
        r1_relaxivity: Longitudinal relaxivity in s^-1/mM. If None, will
            be looked up from field_strength.
        field_strength: Magnetic field strength in Tesla. Used for r1
            lookup if r1_relaxivity is not provided. Default is 3.0T.
        baseline_frames: Number of pre-contrast frames for baseline.
            Default is 5 frames.
        convert_to_plasma: If True, converts to plasma concentration using
            hematocrit correction: C_plasma = C_blood / (1 - Hct).
            Default is True.
        hematocrit: Hematocrit value for plasma conversion. Default is 0.42.

    Returns:
        DCEConversionResult containing concentration and intermediate values.

    Raises:
        ValueError: If signal is empty or has invalid dimensions.
        ValueError: If flip_angle_deg is not in valid range (0, 90).
        ValueError: If tr is not positive.
        ValueError: If baseline_frames exceeds signal timepoints.
        ValueError: If T1 map dimensions don't match signal spatial dims.
        ValueError: If B1 map dimensions don't match signal spatial dims.

    Notes:
        - Concentration is in mM (millimolar).
        - T1 map should be in milliseconds (will be converted to seconds).
        - For AIF voxels, concentration represents whole blood or plasma
          depending on convert_to_plasma setting.
        - Tissue concentration is typically lower than plasma due to
          extravascular extracellular space distribution.

    Example:
        >>> # Convert DCE signal with T1 map at 3T
        >>> result = signal_to_concentration(
        ...     signal=dce_data,       # (x, y, z, time)
        ...     flip_angle_deg=15.0,
        ...     tr=0.0050,             # 5ms
        ...     t1_map=t1_data,        # (x, y, z) in ms
        ...     field_strength=3.0,
        ... )
        >>> concentration = result.concentration  # (x, y, z, time) in mM
    """
    # Validate inputs
    if signal.size == 0:
        raise ValueError("Signal array is empty")

    if signal.ndim == 1:
        n_timepoints = signal.shape[0]
        spatial_shape: Tuple[int, ...] = ()
    elif signal.ndim == 4:
        n_timepoints = signal.shape[3]
        spatial_shape = signal.shape[:3]
    else:
        raise ValueError(
            f"Signal must be 1D (time) or 4D (x,y,z,time), got {signal.ndim}D"
        )

    if not (0 < flip_angle_deg < 90):
        raise ValueError(
            f"flip_angle_deg must be in (0, 90), got {flip_angle_deg}"
        )

    if tr <= 0:
        raise ValueError(f"TR must be positive, got {tr}")

    if baseline_frames >= n_timepoints:
        raise ValueError(
            f"baseline_frames ({baseline_frames}) must be less than "
            f"number of timepoints ({n_timepoints})"
        )

    if baseline_frames < 1:
        raise ValueError(f"baseline_frames must be at least 1, got {baseline_frames}")

    # Validate T1 map dimensions if provided
    if t1_map is not None and signal.ndim == 4:
        if t1_map.shape != spatial_shape:
            raise ValueError(
                f"T1 map shape {t1_map.shape} doesn't match signal "
                f"spatial shape {spatial_shape}"
            )

    # Validate B1 map dimensions if provided
    if b1_map is not None and signal.ndim == 4:
        if b1_map.shape != spatial_shape:
            raise ValueError(
                f"B1 map shape {b1_map.shape} doesn't match signal "
                f"spatial shape {spatial_shape}"
            )

    # Get r1 relaxivity
    if r1_relaxivity is not None:
        effective_r1 = r1_relaxivity
    else:
        effective_r1 = get_r1_relaxivity(field_strength)

    # Convert flip angle to radians
    flip_angle_rad = np.deg2rad(flip_angle_deg)

    # Apply B1+ correction if provided
    if b1_map is not None:
        # B1+ map is relative: actual_flip = nominal_flip × B1+
        b1_safe = np.maximum(b1_map, EPSILON_B1)
        if signal.ndim == 4:
            # Expand B1 map for broadcasting with flip angle
            effective_flip_rad = flip_angle_rad * b1_safe[..., np.newaxis]
        else:
            # 1D signal: assume scalar B1 (should be provided as scalar or single value)
            effective_flip_rad = flip_angle_rad * float(np.mean(b1_safe))
    else:
        effective_flip_rad = flip_angle_rad

    # Get T1 values (convert from ms to seconds)
    if t1_map is not None:
        # T1 map provided (in ms)
        t1_seconds = t1_map / 1000.0
    else:
        # Use default T1 value
        t1_seconds = t1_default_ms / 1000.0

    # Ensure T1 is positive
    if isinstance(t1_seconds, np.ndarray):
        t1_safe = np.maximum(t1_seconds, EPSILON_T1)
    else:
        t1_safe = max(t1_seconds, EPSILON_T1)

    # Compute baseline E1 = exp(-TR/T1)
    e1_baseline = np.exp(-tr / t1_safe)

    # Compute baseline R1 = 1/T1
    if isinstance(t1_safe, np.ndarray):
        r1_baseline: Union[float, npt.NDArray[np.floating]] = 1.0 / t1_safe
    else:
        r1_baseline = 1.0 / t1_safe

    # Compute baseline signal (mean of first baseline_frames)
    if signal.ndim == 1:
        baseline_signal: Union[float, npt.NDArray[np.floating]] = float(
            np.mean(signal[:baseline_frames])
        )
    else:
        baseline_signal = np.mean(signal[..., :baseline_frames], axis=-1)

    # Prevent division by zero in baseline signal
    if isinstance(baseline_signal, np.ndarray):
        baseline_signal_safe = np.maximum(np.abs(baseline_signal), EPSILON_SIGNAL)
        baseline_signal_safe = np.where(
            baseline_signal >= 0, baseline_signal_safe, -baseline_signal_safe
        )
    else:
        baseline_signal_safe = max(abs(baseline_signal), EPSILON_SIGNAL)
        if baseline_signal < 0:
            baseline_signal_safe = -baseline_signal_safe

    # Compute signal ratio S(t)/S(baseline) for each timepoint
    if signal.ndim == 1:
        signal_ratio = signal / baseline_signal_safe
    else:
        # For 4D, broadcast baseline_signal across time
        signal_ratio = signal / baseline_signal_safe[..., np.newaxis]

    # Allocate arrays for results
    if signal.ndim == 1:
        r1_dynamic = np.zeros_like(signal)
        concentration = np.zeros_like(signal)
    else:
        r1_dynamic = np.zeros_like(signal)
        concentration = np.zeros_like(signal)

    # Process each timepoint
    if signal.ndim == 1:
        # 1D: process as vector
        for t in range(n_timepoints):
            # Get effective flip angle (scalar for 1D)
            if isinstance(effective_flip_rad, np.ndarray):
                flip_t = float(effective_flip_rad[t]) if effective_flip_rad.ndim == 1 else float(np.mean(effective_flip_rad))
            else:
                flip_t = float(effective_flip_rad)

            # Solve for E1(t)
            e1_t = solve_e1_from_signal_ratio(
                signal_ratio[t],
                flip_t,
                e1_baseline,
            )

            # Convert to R1
            r1_dynamic[t] = e1_to_r1(e1_t, tr)

            # Compute concentration: C = (R1(t) - R1(0)) / r1
            concentration[t] = (r1_dynamic[t] - r1_baseline) / effective_r1
    else:
        # 4D: vectorized processing for efficiency
        for t in range(n_timepoints):
            # Get effective flip angle for this timepoint
            flip_t_4d: Union[float, npt.NDArray[np.floating]]
            if isinstance(effective_flip_rad, np.ndarray) and effective_flip_rad.ndim == 4:
                flip_t_4d = effective_flip_rad[..., t]
            elif isinstance(effective_flip_rad, np.ndarray):
                flip_t_4d = effective_flip_rad
            else:
                flip_t_4d = effective_flip_rad

            # Solve for E1(t) for all voxels
            e1_t = solve_e1_from_signal_ratio(
                signal_ratio[..., t],
                flip_t_4d,
                e1_baseline,
            )

            # Convert to R1
            r1_dynamic[..., t] = e1_to_r1(e1_t, tr)

        # Compute concentration for all timepoints
        # C = (R1(t) - R1(0)) / r1
        if isinstance(r1_baseline, np.ndarray):
            concentration = (r1_dynamic - r1_baseline[..., np.newaxis]) / effective_r1
        else:
            concentration = (r1_dynamic - r1_baseline) / effective_r1

    # Ensure non-negative concentration (negative values are non-physical)
    # Small negative values can occur due to noise before contrast arrival
    # We clip to zero to maintain physical meaning
    concentration = np.maximum(concentration, 0.0)

    # Convert to plasma concentration if requested
    # C_plasma = C_blood / (1 - Hct)
    if convert_to_plasma:
        concentration = concentration / (1.0 - hematocrit)

    return DCEConversionResult(
        concentration=concentration,
        r1_dynamic=r1_dynamic,
        r1_baseline=r1_baseline,
        is_plasma=convert_to_plasma,
        effective_r1=effective_r1,
        baseline_signal=baseline_signal,
    )


def signal_to_concentration_simple(
    signal: npt.NDArray[np.floating],
    flip_angle_deg: float,
    tr: float,
    t1_pre_ms: float = DEFAULT_T1_MS,
    r1_relaxivity: Optional[float] = None,
    field_strength: float = 3.0,
    baseline_frames: int = DEFAULT_BASELINE_FRAMES,
    convert_to_plasma: bool = True,
) -> npt.NDArray[np.floating]:
    """Simplified signal to concentration conversion.

    Convenience function that returns only the concentration array,
    using a single T1 value for all voxels.

    Args:
        signal: DCE signal array (1D or 4D).
        flip_angle_deg: Flip angle in degrees.
        tr: Repetition time in seconds.
        t1_pre_ms: Pre-contrast T1 value in milliseconds.
        r1_relaxivity: Longitudinal relaxivity in s^-1/mM.
        field_strength: Field strength in Tesla for r1 lookup.
        baseline_frames: Number of baseline frames.
        convert_to_plasma: If True, convert to plasma concentration.

    Returns:
        Concentration array in mM.
    """
    result = signal_to_concentration(
        signal=signal,
        flip_angle_deg=flip_angle_deg,
        tr=tr,
        t1_map=None,
        t1_default_ms=t1_pre_ms,
        b1_map=None,
        r1_relaxivity=r1_relaxivity,
        field_strength=field_strength,
        baseline_frames=baseline_frames,
        convert_to_plasma=convert_to_plasma,
    )
    return result.concentration


def concentration_to_plasma(
    concentration_blood: npt.NDArray[np.floating],
    hematocrit: float = HEMATOCRIT,
) -> npt.NDArray[np.floating]:
    """Convert whole blood concentration to plasma concentration.

    Gadolinium contrast agents distribute only in the plasma compartment
    of blood, not in red blood cells. The plasma concentration is
    therefore higher than the whole blood concentration:

        C_plasma = C_blood / (1 - Hct)

    Args:
        concentration_blood: Whole blood concentration in mM.
        hematocrit: Hematocrit value (volume fraction of red blood cells).
            Default is 0.42 (42%).

    Returns:
        Plasma concentration in mM.

    Notes:
        - Typical hematocrit values: 0.38-0.46 (38-46%)
        - For typical Hct=0.42: C_plasma ≈ 1.72 × C_blood
        - This conversion is important for AIF measurements, as
          pharmacokinetic models assume plasma concentration.
    """
    return concentration_blood / (1.0 - hematocrit)


def plasma_to_blood_concentration(
    concentration_plasma: npt.NDArray[np.floating],
    hematocrit: float = HEMATOCRIT,
) -> npt.NDArray[np.floating]:
    """Convert plasma concentration to whole blood concentration.

    Inverse of concentration_to_plasma:

        C_blood = C_plasma × (1 - Hct)

    Args:
        concentration_plasma: Plasma concentration in mM.
        hematocrit: Hematocrit value. Default is 0.42.

    Returns:
        Whole blood concentration in mM.
    """
    return concentration_plasma * (1.0 - hematocrit)


def signal_to_concentration_from_files(
    signal_file: Union[str, Path],
    output_file: Union[str, Path],
    flip_angle_deg: float,
    tr: float,
    t1_map_file: Optional[Union[str, Path]] = None,
    t1_default_ms: float = DEFAULT_T1_MS,
    b1_map_file: Optional[Union[str, Path]] = None,
    r1_relaxivity: Optional[float] = None,
    field_strength: float = 3.0,
    baseline_frames: int = DEFAULT_BASELINE_FRAMES,
    convert_to_plasma: bool = True,
) -> Tuple[Path, DCEConversionResult]:
    """Convert DCE signal from NIfTI file to concentration.

    File-based convenience wrapper that loads the DCE signal, performs
    conversion, and saves the result.

    Args:
        signal_file: Path to input DCE NIfTI file.
        output_file: Path to save concentration NIfTI.
        flip_angle_deg: Flip angle in degrees.
        tr: Repetition time in seconds.
        t1_map_file: Optional path to T1 map NIfTI file (in ms).
        t1_default_ms: Default T1 value in ms if no T1 map.
        b1_map_file: Optional path to B1+ map NIfTI file.
        r1_relaxivity: Longitudinal relaxivity in s^-1/mM.
        field_strength: Field strength in Tesla.
        baseline_frames: Number of baseline frames.
        convert_to_plasma: If True, convert to plasma concentration.

    Returns:
        Tuple of (output_path, DCEConversionResult).

    Raises:
        FileNotFoundError: If input signal file does not exist.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    signal_path = Path(signal_file)
    if not signal_path.exists():
        raise FileNotFoundError(f"Signal file not found: {signal_path}")

    # Load signal
    signal_img = nib.load(signal_path)
    signal_data = signal_img.get_fdata()  # type: ignore[attr-defined]

    # Load T1 map if provided
    t1_map: Optional[npt.NDArray[np.floating]] = None
    if t1_map_file is not None:
        t1_path = Path(t1_map_file)
        if t1_path.exists():
            t1_img = nib.load(t1_path)
            t1_map = t1_img.get_fdata()  # type: ignore[attr-defined]

    # Load B1 map if provided
    b1_map: Optional[npt.NDArray[np.floating]] = None
    if b1_map_file is not None:
        b1_path = Path(b1_map_file)
        if b1_path.exists():
            b1_img = nib.load(b1_path)
            b1_map = b1_img.get_fdata()  # type: ignore[attr-defined]

    # Convert
    result = signal_to_concentration(
        signal=signal_data,
        flip_angle_deg=flip_angle_deg,
        tr=tr,
        t1_map=t1_map,
        t1_default_ms=t1_default_ms,
        b1_map=b1_map,
        r1_relaxivity=r1_relaxivity,
        field_strength=field_strength,
        baseline_frames=baseline_frames,
        convert_to_plasma=convert_to_plasma,
    )

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concentration_img = nib.Nifti1Image(
        result.concentration,
        signal_img.affine,  # type: ignore[attr-defined]
        signal_img.header,  # type: ignore[attr-defined]
    )
    nib.save(concentration_img, output_path)

    return output_path, result
