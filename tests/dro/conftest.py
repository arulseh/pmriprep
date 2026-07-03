"""Shared DRO (Digital Reference Object) utilities and fixtures.

Provides synthetic AIF generators, noise functions, DSC forward models,
signal-domain helpers, and pytest fixtures for parametric recovery tests.
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import gammainc

from tests.conftest import generate_parker_aif


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass(frozen=True)
class DSCSignalConfig:
    """Frozen configuration for DSC signal generation.

    Follows the pattern from qiba/conftest.py (QIBATissueParams).

    Attributes
    ----------
    tr_sec : float
        Repetition time in seconds.
    te_sec : float
        Echo time in seconds.
    s0 : float
        Baseline signal intensity (arbitrary units).
    field_strength : float
        Magnetic field strength in Tesla.
    label : str
        Human-readable label for test IDs.
    """

    tr_sec: float
    te_sec: float
    s0: float
    field_strength: float
    label: str


# Pre-defined signal configurations for parametrized tests
DSC_SIGNAL_CONFIGS: Dict[str, DSCSignalConfig] = {
    "typical_3T": DSCSignalConfig(1.5, 0.030, 1000.0, 3.0, "typical_3T"),
    "short_te": DSCSignalConfig(1.5, 0.025, 1000.0, 3.0, "short_te"),
    "long_te": DSCSignalConfig(1.5, 0.040, 1000.0, 3.0, "long_te"),
    "low_s0": DSCSignalConfig(1.5, 0.030, 500.0, 3.0, "low_s0"),
    "high_s0": DSCSignalConfig(1.5, 0.030, 2000.0, 3.0, "high_s0"),
    "1.5T": DSCSignalConfig(1.5, 0.030, 1000.0, 1.5, "1.5T"),
}


@dataclass(frozen=True)
class DSCTissueParams:
    """Ground-truth tissue parameters for DSC DRO tests.

    Attributes
    ----------
    cbf : float
        Cerebral blood flow (relative units, same as deconvolution output).
    mtt : float
        Mean transit time in seconds.
    name : str
        Tissue type name.
    """

    cbf: float
    mtt: float
    name: str

    @property
    def cbv(self) -> float:
        """CBV = CBF * MTT / 60 (if CBF in mL/100g/min and MTT in seconds)."""
        return self.cbf * self.mtt / 60.0


DSC_TISSUE_PARAMS: Dict[str, DSCTissueParams] = {
    "gm": DSCTissueParams(60.0, 4.0, "gray_matter"),
    "wm": DSCTissueParams(20.0, 4.8, "white_matter"),
    "tumor": DSCTissueParams(100.0, 6.0, "tumor"),
}


# =============================================================================
# AIF generators
# =============================================================================


def generate_parker_aif_dro(
    n_timepoints: int = 80,
    dt_min: float = 0.0667,
    peak_concentration: float = 5.0,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Generate a Parker-style AIF for DCE DRO tests.

    Wraps the existing Parker AIF generator from tests/conftest.py.

    Parameters
    ----------
    n_timepoints : int
        Number of time points.
    dt_min : float
        Time step in minutes.
    peak_concentration : float
        Peak AIF concentration in mM.

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        Tuple of (aif_mM, time_min).
    """
    time_min = np.arange(n_timepoints, dtype=np.float64) * dt_min
    # Arrival at ~10% of the time series
    arrival_time_min = time_min[int(n_timepoints * 0.1)]
    aif = generate_parker_aif(
        t_min=time_min,
        arrival_time_min=arrival_time_min,
        peak_concentration=peak_concentration,
    )
    return np.asarray(aif, dtype=np.float64), time_min


def generate_dsc_aif_dro(
    n_timepoints: int = 60,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    peak_delta_r2: float = 40.0,
    alpha: float = 2.5,
    beta: float = 3.0,
    oversample: int = 1,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Generate a gamma-variate AIF in concentration domain for DSC DRO tests.

    The AIF is a(t) = A*(t-t0)^alpha * exp(-(t-t0)/beta) for t > t0 (t0 the
    bolus arrival), with A set so the analytic peak height equals
    peak_delta_r2. The default alpha=2.5/beta=3.0 is a broad (venous-tier,
    FWHM ~10 s) bolus used by the method-recovery regression guards; the
    arterial first-pass shape is alpha=3.0/beta=1.5 (FWHM ~5 s). For a narrow
    AIF on a coarse axis, pass oversample>1 to build the curve on a fine grid
    (dt_sec/oversample) and block-average back to dt_sec, which models the TR
    integration and avoids a peak-sampling bias in the ground-truth convolution.

    Parameters
    ----------
    n_timepoints : int
        Number of time points.
    dt_sec : float
        Time step in seconds.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    peak_delta_r2 : float
        Analytic peak delta-R2* value (s^-1).
    alpha : float
        Gamma-variate shape (power) parameter.
    beta : float
        Gamma-variate scale (decay) parameter in seconds.
    oversample : int
        Fine-grid oversampling factor; 1 evaluates on the coarse axis directly.

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        Tuple of (aif_concentration, time_sec).
    """
    os_factor = max(int(oversample), 1)
    fine_dt = dt_sec / os_factor
    n_fine = n_timepoints * os_factor
    time_fine = np.arange(n_fine, dtype=np.float64) * fine_dt
    arrival_time = baseline_frames * dt_sec

    aif_fine = np.zeros(n_fine, dtype=np.float64)
    peak_time = alpha * beta
    peak_factor = (peak_time**alpha) * np.exp(-alpha)
    amplitude = peak_delta_r2 / peak_factor if peak_factor > 1e-10 else peak_delta_r2

    mask = time_fine > arrival_time
    if np.any(mask):
        dt = time_fine[mask] - arrival_time
        aif_fine[mask] = amplitude * (dt**alpha) * np.exp(-dt / beta)

    if os_factor == 1:
        aif = aif_fine
    else:
        aif = aif_fine.reshape(n_timepoints, os_factor).mean(axis=1)

    time_sec = np.arange(n_timepoints, dtype=np.float64) * dt_sec
    return np.asarray(aif, dtype=np.float64), time_sec


# =============================================================================
# AIF effect helpers (recirculation, dispersion)
# =============================================================================


def apply_recirculation(
    aif: npt.NDArray[np.floating],
    dt: float,
    delay_sec: float = 8.0,
    tau_sec: float = 30.0,
) -> npt.NDArray[np.floating]:
    """Add a recirculation second pass to a first-pass AIF.

    The recirculation component is the first-pass AIF convolved with a delayed
    exponential kernel h(t) = exp(-(t - delay)/tau) for t >= delay (zero
    before), normalized to unit area so the second pass conserves the bolus
    mass. The tau time constant disperses and the delay shifts the returning
    bolus, producing a low, broad second hump on the AIF tail.

    Parameters
    ----------
    aif : npt.NDArray[np.floating]
        First-pass AIF concentration array (1D).
    dt : float
        Time step in seconds.
    delay_sec : float
        Recirculation delay in seconds.
    tau_sec : float
        Exponential dispersion time constant in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        AIF with recirculation added (same length as input).
    """
    n = len(aif)
    t = np.arange(n, dtype=np.float64) * dt
    kernel = np.where(t >= delay_sec, np.exp(-(t - delay_sec) / tau_sec), 0.0)
    kernel_area = float(kernel.sum()) * dt
    if kernel_area > 1e-12:
        kernel = kernel / kernel_area
    recirc = np.convolve(aif, kernel)[:n] * dt
    return np.asarray(aif + recirc, dtype=np.float64)


#: Pre-committed moderate dispersion severity (log-normal VTF sigma, mu) for the
#: dispersed DSC DRO: mean transit E = exp(mu + sigma**2/2) = 1.14 s, a physiological
#: arterial-to-tissue bolus dispersion. The middle of the mild/moderate/severe
#: presets, fixed independent of any deconvolution result (not tuned to favour an
#: operator).
DSC_DRO_VTF_MODERATE = (0.75, -0.15)


def apply_vtf_dispersion(
    aif: npt.NDArray[np.floating],
    dt: float,
    sigma: float,
    mu: float,
) -> npt.NDArray[np.floating]:
    """Disperse an AIF by convolving with a log-normal vascular transport function.

    The tissue-input AIF is the measured AIF convolved with a vascular
    transport function (VTF) describing the distribution of transit times
    between the AIF measurement site and the tissue input. The VTF is a
    unit-area log-normal density
    VTF(t) = 1/(t*sigma*sqrt(2*pi)) * exp(-(ln t - mu)^2 / (2*sigma^2)), t > 0,
    with mean transit time E = exp(mu + sigma^2/2). Representative dispersion
    severities (sigma, mu): mild (1.0, -1.0), moderate (0.75, -0.15), severe
    (0.78, 0.59), giving E = 0.61, 1.14, 2.45 s respectively.

    Parameters
    ----------
    aif : npt.NDArray[np.floating]
        Measured AIF concentration array (1D).
    dt : float
        Time step in seconds.
    sigma : float
        Log-normal shape parameter (standard deviation of ln t).
    mu : float
        Log-normal scale parameter (mean of ln t).

    Returns
    -------
    npt.NDArray[np.floating]
        Dispersed AIF (same length as input).
    """
    n = len(aif)
    t = np.arange(n, dtype=np.float64) * dt
    vtf = np.zeros(n, dtype=np.float64)
    pos = t > 0
    vtf[pos] = (
        1.0
        / (t[pos] * sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-((np.log(t[pos]) - mu) ** 2) / (2.0 * sigma**2))
    )
    vtf_area = float(vtf.sum()) * dt
    if vtf_area > 1e-12:
        vtf = vtf / vtf_area
    dispersed = np.convolve(aif, vtf)[:n] * dt
    return np.asarray(dispersed, dtype=np.float64)


def delay_aif(
    aif: npt.NDArray[np.floating],
    time_sec: npt.NDArray[np.floating],
    delay_sec: float,
) -> npt.NDArray[np.floating]:
    """Delay a DSC AIF by ``delay_sec`` seconds (positive = later arrival).

    Thin seconds-native wrapper over ``pmriprep.utils.dce_bat.shift_aif``,
    which resamples the AIF at ``t - delay`` on a minutes axis. The seconds
    time axis is converted to minutes so the shared, validated interpolation
    and baseline fill apply. A positive delay shifts the AIF later, modelling
    tissue that enhances after the AIF sampling site — the arrival offset sSVD
    is sensitive to and the block-circulant methods are not.

    Parameters
    ----------
    aif : npt.NDArray[np.floating]
        AIF concentration array (1D).
    time_sec : npt.NDArray[np.floating]
        Time axis in seconds (may be non-uniform).
    delay_sec : float
        Delay in seconds. Positive shifts later; zero returns the AIF unchanged.

    Returns
    -------
    npt.NDArray[np.floating]
        The time-shifted AIF, same shape as the input.
    """
    from pmriprep.utils.dce_bat import shift_aif

    time_min = np.asarray(time_sec, dtype=np.float64) / 60.0
    return np.asarray(shift_aif(aif, time_min, delay_sec), dtype=np.float64)


# =============================================================================
# Signal-domain helpers
# =============================================================================


def concentration_to_signal(
    concentration: npt.NDArray[np.floating],
    te: float,
    s0: float,
    baseline_frames: int = 10,
) -> npt.NDArray[np.floating]:
    """Convert ΔR2* concentration to DSC signal S(t) = S0 * exp(-TE * ΔR2*(t)).

    Prepends a stable baseline of length baseline_frames where concentration
    is zero (pure S0). Named to avoid confusion with existing dsc_forward_model
    (AIF * residue convolution).

    Parameters
    ----------
    concentration : npt.NDArray[np.floating]
        ΔR2* array (1D time or 4D x,y,z,time). Must not
        include baseline frames — they are prepended here.
    te : float
        Echo time in seconds.
    s0 : float
        Baseline signal intensity.
    baseline_frames : int
        Number of baseline frames to prepend.

    Returns
    -------
    npt.NDArray[np.floating]
        Signal array with baseline prepended, same dtype as float64.
    """
    if concentration.ndim == 1:
        n_post = len(concentration)
        baseline = np.full(baseline_frames, s0, dtype=np.float64)
        post_signal = s0 * np.exp(-te * concentration)
        return np.concatenate([baseline, post_signal])
    elif concentration.ndim == 4:
        spatial = concentration.shape[:3]
        n_post = concentration.shape[3]
        baseline = np.full((*spatial, baseline_frames), s0, dtype=np.float64)
        post_signal = s0 * np.exp(-te * concentration)
        return np.concatenate([baseline, post_signal], axis=3)
    else:
        raise ValueError(
            f"concentration must be 1D or 4D, got {concentration.ndim}D"
        )


def generate_leaky_dsc_tissue(
    c_tissue_noleak: npt.NDArray[np.floating],
    reference_curve: npt.NDArray[np.floating],
    k2: float,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Apply BSW leakage model to a pre-computed clean tissue curve.

    T1 leakage reduces apparent ΔR2*, so the leaky curve is LOWER than
    the non-leaky curve:
        C_leaky(t) = C_noleak(t) - K2 * cumsum(reference) * dt

    The reference_curve should be C̄(t) (mean brain reference) for K2 DRO
    phantoms, matching the estimator. For single-voxel tests, AIF
    may be used as reference.

    Parameters
    ----------
    c_tissue_noleak : npt.NDArray[np.floating]
        Clean tissue concentration (1D, no leakage).
    reference_curve : npt.NDArray[np.floating]
        Reference curve for leakage (1D, same length).
        Typically mean brain C̄(t) or AIF.
    k2 : float
        K2 leakage rate parameter.
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Leaky tissue concentration (1D). May contain negative values
        at high K2 (physically correct: T1 leakage overshoot).
    """
    if len(c_tissue_noleak) != len(reference_curve):
        raise ValueError(
            f"Tissue ({len(c_tissue_noleak)}) and reference ({len(reference_curve)}) "
            f"must have same length"
        )
    cumsum_ref = np.cumsum(reference_curve) * dt
    c_leaky = c_tissue_noleak - k2 * cumsum_ref
    return np.asarray(c_leaky, dtype=np.float64)


def generate_leaky_dsc_tissue_bidirectional(
    c_tissue_noleak: npt.NDArray[np.floating],
    reference_curve: npt.NDArray[np.floating],
    k2: float,
    kep: float,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Apply bidirectional leakage model to a pre-computed clean tissue curve.

    Uses the Leu 2016 model which accounts for EES washout:
        C_leaky(t) = C_noleak(t) - K2 * C_e(t)

    where C_e(t) = reference_curve ⊗ exp(-kep*t) (instantaneous with washout).

    Parameters
    ----------
    c_tissue_noleak : npt.NDArray[np.floating]
        Clean tissue concentration (1D, no leakage).
    reference_curve : npt.NDArray[np.floating]
        Reference curve for leakage (1D, same length).
    k2 : float
        K2 leakage rate parameter.
    kep : float
        EES efflux rate constant in s⁻¹.
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Leaky tissue concentration (1D).

    References
    ----------
        Leu K et al. AJNR 2016;37(8):1440-1446.
    """
    if len(c_tissue_noleak) != len(reference_curve):
        raise ValueError(
            f"Tissue ({len(c_tissue_noleak)}) and reference ({len(reference_curve)}) "
            f"must have same length"
        )
    n_tp = len(reference_curve)
    time = np.arange(n_tp) * dt
    exp_kernel = np.exp(-kep * time)

    # Convolve and scale by dt (rectangular integration)
    c_e = np.convolve(reference_curve, exp_kernel, mode="full")[:n_tp] * dt

    c_leaky = c_tissue_noleak - k2 * c_e
    return np.asarray(c_leaky, dtype=np.float64)


class K2DROPhantom(NamedTuple):
    """Result of generate_k2_dro_phantom()."""

    concentration: npt.NDArray[np.floating]
    """4D leaky concentration (x, y, z, time) with baseline."""
    clean: npt.NDArray[np.floating]
    """4D clean concentration (x, y, z, time) with baseline."""
    aif: npt.NDArray[np.floating]
    """1D AIF (n_timepoints total including baseline)."""
    brain_mask: npt.NDArray[np.bool_]
    """3D brain mask (all tissue + AIF voxels)."""
    enhancing_mask: npt.NDArray[np.bool_]
    """3D mask of enhancing (leaky) voxels."""
    dt: float
    """Time step in seconds."""


def generate_k2_dro_phantom(
    k2_enhancing: float = 0.1,
    n_brain_voxels: int = 100,
    n_enhancing_voxels: int = 10,
    dt_sec: float = 1.5,
    n_timepoints: int = 60,
    baseline_frames: int = 10,
    aif_peak: float = 40.0,
    noise_frac: float = 0.0,
    seed: int = 42,
) -> K2DROPhantom:
    """Generate a multi-voxel 4D phantom for K2 validation.

    Creates a single-slice phantom with non-enhancing brain voxels
    (varied CBF/MTT, K2=0) and enhancing voxels (tumor-like, K2>0).
    The forward leakage model uses the mean brain reference C̄(t),
    matching what the estimator recovers.

    Parameters
    ----------
    k2_enhancing : float
        K2 value for enhancing voxels.
    n_brain_voxels : int
        Number of non-enhancing brain voxels.
    n_enhancing_voxels : int
        Number of enhancing (leaky) voxels.
    dt_sec : float
        Time step in seconds.
    n_timepoints : int
        Number of post-baseline timepoints.
    baseline_frames : int
        Number of baseline frames.
    aif_peak : float
        Peak AIF delta-R2* in s⁻¹.
    noise_frac : float
        Fraction of peak for Gaussian noise (0=noiseless).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    K2DROPhantom
        K2DROPhantom with all fields populated.
    """
    rng = np.random.default_rng(seed)

    # Generate AIF (post-baseline only)
    aif_post, _ = generate_dsc_aif_dro(
        n_timepoints=n_timepoints,
        dt_sec=dt_sec,
        baseline_frames=0,
        peak_delta_r2=aif_peak,
    )

    # Determine phantom spatial dimensions
    # Need enough voxels: n_brain + n_enhancing + margin
    total_voxels = n_brain_voxels + n_enhancing_voxels
    nx = 12
    ny = max(12, (total_voxels + nx - 1) // nx + 2)
    nz = 1
    shape = (nx, ny, nz)

    n_total = baseline_frames + n_timepoints

    # Generate varied non-enhancing brain tissue curves
    brain_cbfs = rng.uniform(20.0, 60.0, n_brain_voxels)
    brain_mtts = rng.uniform(3.0, 6.0, n_brain_voxels)

    brain_curves_post = np.zeros((n_brain_voxels, n_timepoints), dtype=np.float64)
    for i in range(n_brain_voxels):
        residue = make_exponential_residue(
            brain_cbfs[i], brain_mtts[i], n_timepoints, dt_sec,
        )
        brain_curves_post[i] = dsc_forward_model(aif_post, residue, dt_sec)

    # Compute mean brain reference C̄(t) from non-enhancing voxels
    c_bar_post = np.mean(brain_curves_post, axis=0)

    # Generate enhancing (tumor) tissue curves
    tumor_cbf = 100.0
    tumor_mtt = 6.0
    tumor_residue = make_exponential_residue(
        tumor_cbf, tumor_mtt, n_timepoints, dt_sec,
    )
    tumor_clean_post = dsc_forward_model(aif_post, tumor_residue, dt_sec)

    # Apply leakage using mean brain reference (NOT AIF)
    tumor_leaky_post = generate_leaky_dsc_tissue(
        tumor_clean_post, c_bar_post, k2_enhancing, dt_sec,
    )

    # Build 4D volumes
    concentration = np.zeros((*shape, n_total), dtype=np.float64)
    clean = np.zeros((*shape, n_total), dtype=np.float64)
    brain_mask = np.zeros(shape, dtype=bool)
    enhancing_mask = np.zeros(shape, dtype=bool)

    # Place brain voxels
    idx = 0
    for i in range(n_brain_voxels):
        x = idx % nx
        y = idx // nx
        if y >= ny:
            break
        concentration[x, y, 0, baseline_frames:] = brain_curves_post[i]
        clean[x, y, 0, baseline_frames:] = brain_curves_post[i]
        brain_mask[x, y, 0] = True
        idx += 1

    # Place enhancing voxels
    for i in range(n_enhancing_voxels):
        x = idx % nx
        y = idx // nx
        if y >= ny:
            break
        concentration[x, y, 0, baseline_frames:] = tumor_leaky_post
        clean[x, y, 0, baseline_frames:] = tumor_clean_post
        brain_mask[x, y, 0] = True
        enhancing_mask[x, y, 0] = True
        idx += 1

    # Add noise if requested
    if noise_frac > 0:
        peak = float(np.max(np.abs(clean)))
        if peak > 0:
            noise = rng.normal(0.0, noise_frac * peak, concentration.shape)
            concentration = concentration + noise

    # Full AIF with baseline
    aif_full = np.zeros(n_total, dtype=np.float64)
    aif_full[baseline_frames:] = aif_post

    return K2DROPhantom(
        concentration=concentration,
        clean=clean,
        aif=aif_full,
        brain_mask=brain_mask,
        enhancing_mask=enhancing_mask,
        dt=dt_sec,
    )


def generate_dsc_3d_phantom(
    shape: Tuple[int, int, int],
    aif_coords: Set[Tuple[int, int, int]],
    aif: npt.NDArray[np.floating],
    tissue_curves: Dict[str, npt.NDArray[np.floating]],
    tissue_map: npt.NDArray[np.integer],
    tissue_labels: Dict[str, int],
    baseline_frames: int = 10,
) -> Tuple[npt.NDArray[np.floating], Dict[str, npt.NDArray[np.bool_]]]:
    """Build a multi-tissue 3D+t concentration volume.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Spatial dimensions (x, y, z).
    aif_coords : Set[Tuple[int, int, int]]
        Set of (x, y, z) tuples for AIF voxels.
    aif : npt.NDArray[np.floating]
        1D AIF concentration (post-baseline only, n_post frames).
    tissue_curves : Dict[str, npt.NDArray[np.floating]]
        Dict mapping tissue name -> 1D concentration curve
        (post-baseline only).
    tissue_map : npt.NDArray[np.integer]
        3D integer array mapping each voxel to a tissue label.
    tissue_labels : Dict[str, int]
        Dict mapping tissue name -> integer label in tissue_map.
    baseline_frames : int
        Number of zero-concentration baseline frames to prepend.

    Returns
    -------
    Tuple[npt.NDArray[np.floating], Dict[str, npt.NDArray[np.bool_]]]
        Tuple of:
            - 4D concentration array (x, y, z, baseline+post_frames)
            - Dict of tissue name -> 3D boolean mask
    """
    n_post = len(aif)
    n_total = baseline_frames + n_post
    volume = np.zeros((*shape, n_total), dtype=np.float64)

    # Fill AIF voxels
    for x, y, z in aif_coords:
        volume[x, y, z, baseline_frames:] = aif

    # Fill tissue voxels
    for tissue_name, label in tissue_labels.items():
        if tissue_name not in tissue_curves:
            continue
        mask = tissue_map == label
        curve = tissue_curves[tissue_name]
        # Broadcast: mask is (x,y,z), curve is (n_post,)
        volume[mask, baseline_frames:] = curve[np.newaxis, :]

    # Override AIF voxels (in case they overlap with tissue_map)
    for x, y, z in aif_coords:
        volume[x, y, z, baseline_frames:] = aif

    # Build masks dict
    masks: Dict[str, npt.NDArray[np.bool_]] = {}
    for tissue_name, label in tissue_labels.items():
        masks[tissue_name] = tissue_map == label

    # AIF mask
    aif_mask = np.zeros(shape, dtype=bool)
    for x, y, z in aif_coords:
        aif_mask[x, y, z] = True
    masks["aif"] = aif_mask

    # Brain mask (everything non-zero label or AIF)
    masks["brain"] = (tissue_map > 0) | aif_mask

    return volume, masks


# =============================================================================
# Noise functions
# =============================================================================


def add_gaussian_noise(
    signal: npt.NDArray[np.floating],
    noise_std: float,
    seed: int = 42,
) -> npt.NDArray[np.floating]:
    """Add Gaussian noise to a signal.

    Parameters
    ----------
    signal : npt.NDArray[np.floating]
        Clean signal array.
    noise_std : float
        Standard deviation of noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    npt.NDArray[np.floating]
        Noisy signal array.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, signal.shape)
    return np.asarray(signal + noise, dtype=np.float64)


def add_rician_noise(
    magnitude_signal: npt.NDArray[np.floating],
    noise_std: float,
    seed: int = 42,
) -> npt.NDArray[np.floating]:
    """Add Rician noise to a magnitude signal.

    Rician noise is the appropriate noise model for magnitude MRI data.
    It is computed as |S + n_real + j*n_imag| where n_real, n_imag are
    independent Gaussian noise.

    Parameters
    ----------
    magnitude_signal : npt.NDArray[np.floating]
        Clean magnitude signal array.
    noise_std : float
        Standard deviation of noise in each channel.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    npt.NDArray[np.floating]
        Noisy magnitude signal array (always positive).
    """
    rng = np.random.default_rng(seed)
    n_real = rng.normal(0.0, noise_std, magnitude_signal.shape)
    n_imag = rng.normal(0.0, noise_std, magnitude_signal.shape)
    noisy = np.sqrt((magnitude_signal + n_real) ** 2 + n_imag**2)
    return np.asarray(noisy, dtype=np.float64)


def snr_to_noise_std(s0: float, snr: float) -> float:
    """Convert SNR to noise standard deviation.

    SNR = S0 / sigma, so sigma = S0 / SNR.

    Parameters
    ----------
    s0 : float
        Baseline signal intensity.
    snr : float
        Signal-to-noise ratio.

    Returns
    -------
    float
        Noise standard deviation.
    """
    if snr <= 0 or np.isinf(snr):
        return 0.0
    return s0 / snr


# =============================================================================
# DSC forward model utilities
# =============================================================================


def dsc_forward_model(
    aif: npt.NDArray[np.floating],
    residue: npt.NDArray[np.floating],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Compute tissue concentration via Toeplitz convolution.

    C(t) = A @ R where A is the (1, 4, 1)/6 Toeplitz matrix of the AIF.
    Matched forward model for sSVD and Tikhonov deconvolution.

    Parameters
    ----------
    aif : npt.NDArray[np.floating]
        Arterial input function concentration array.
    residue : npt.NDArray[np.floating]
        Residue function array (CBF * R(t)).
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Tissue concentration array.
    """
    from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix

    A = build_toeplitz_matrix(aif, dt)
    return np.asarray(A @ residue, dtype=np.float64)


def dsc_forward_model_volterra(
    aif: npt.NDArray[np.floating],
    residue: npt.NDArray[np.floating],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Compute tissue concentration via Volterra convolution.

    C(t) = A @ R where A is the (1, 4, 1)/6 Volterra matrix of the AIF.
    Matched forward model for the CPI parametric deconvolution.

    Parameters
    ----------
    aif : npt.NDArray[np.floating]
        Arterial input function concentration array.
    residue : npt.NDArray[np.floating]
        Residue function array (CBF * R(t)).
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Tissue concentration array.
    """
    from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

    A = build_volterra_matrix(aif, dt)
    return np.asarray(A @ residue, dtype=np.float64)


def dsc_forward_model_circulant(
    aif: npt.NDArray[np.floating],
    residue: npt.NDArray[np.floating],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Compute tissue concentration via block-circulant convolution.

    Matched forward model for oSVD and cSVD deconvolution (Wu 2003).
    Uses the (1, 4, 1)/6 stencil, consistent with the circulant inverse.

    Parameters
    ----------
    aif : npt.NDArray[np.floating]
        Arterial input function concentration array.
    residue : npt.NDArray[np.floating]
        Residue function array (CBF * R(t)).
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Tissue concentration array (length N, truncated from 2N).
    """
    from pmriprep.utils.dsc_deconvolution import build_block_circulant_matrix

    n = len(aif)
    A = build_block_circulant_matrix(aif, dt)
    residue_padded = np.zeros(2 * n, dtype=np.float64)
    residue_padded[:n] = residue
    return np.asarray((A @ residue_padded)[:n], dtype=np.float64)


def make_exponential_residue(
    cbf: float,
    mtt: float,
    n_points: int,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Create an exponential residue function R(t) = CBF * exp(-t/MTT).

    Parameters
    ----------
    cbf : float
        Cerebral blood flow (relative units).
    mtt : float
        Mean transit time in seconds.
    n_points : int
        Number of time points.
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Residue array R(t) = CBF * exp(-t/MTT).
    """
    t = np.arange(n_points, dtype=np.float64) * dt
    mtt_safe = max(mtt, 1e-6)
    return np.asarray(cbf * np.exp(-t / mtt_safe), dtype=np.float64)


def make_boxcar_residue(
    cbf: float,
    mtt: float,
    n_points: int,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Create a box-car (plug-flow) residue R(t) = CBF for t <= MTT, else 0.

    Models a vascular bed with plug flow (capillaries in parallel with equal
    transit times): the residue is constant until MTT then drops to zero. Its
    integral is CBF*MTT (central volume theorem), matching the other families.

    Parameters
    ----------
    cbf : float
        Cerebral blood flow (relative units).
    mtt : float
        Mean transit time in seconds.
    n_points : int
        Number of time points.
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Residue array: CBF for t <= MTT, 0 otherwise.
    """
    t = np.arange(n_points, dtype=np.float64) * dt
    return np.asarray(np.where(t <= mtt, cbf, 0.0), dtype=np.float64)


def make_triangle_residue(
    cbf: float,
    mtt: float,
    n_points: int,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Create a triangular residue R(t) = CBF*(1 - t/(2*MTT)) for t <= 2*MTT, else 0.

    A linearly decreasing residue reaching zero at 2*MTT. Its integral is
    CBF*MTT and its mean transit time (integral / R(0)) is MTT, consistent with
    the exponential and box-car families.

    Parameters
    ----------
    cbf : float
        Cerebral blood flow (relative units).
    mtt : float
        Mean transit time in seconds.
    n_points : int
        Number of time points.
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Residue array: CBF*(1 - t/(2*MTT)) for t <= 2*MTT, 0 otherwise.
    """
    t = np.arange(n_points, dtype=np.float64) * dt
    mtt_safe = max(mtt, 1e-6)
    residue = cbf * (1.0 - t / (2.0 * mtt_safe))
    return np.asarray(np.where(t <= 2.0 * mtt_safe, residue, 0.0), dtype=np.float64)


def make_gamma_residue(
    cbf: float,
    mtt: float,
    alpha: float,
    n_points: int,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Create a gamma (CTH) residue: the survival function of gamma transit times.

    Transit times follow a gamma distribution with shape alpha and scale
    beta = MTT/alpha (so the mean alpha*beta == MTT). The residue is the
    fraction of tracer remaining, R(t) = CBF*(1 - P(alpha, t/beta)), where P is
    the regularized lower incomplete gamma function. R(0) = CBF, R is
    monotonically decreasing, and its integral is CBF*MTT (central volume
    theorem). alpha = 1 recovers the exponential residue CBF*exp(-t/MTT); large
    alpha approaches a box-car, spanning the capillary-transit-heterogeneity
    range with a single shape parameter.

    Parameters
    ----------
    cbf : float
        Cerebral blood flow (relative units).
    mtt : float
        Mean transit time in seconds.
    alpha : float
        Gamma shape parameter (transit-time heterogeneity); alpha=1 -> exponential.
    n_points : int
        Number of time points.
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Gamma-survival residue array R(t).
    """
    t = np.arange(n_points, dtype=np.float64) * dt
    mtt_safe = max(mtt, 1e-6)
    alpha_safe = max(alpha, 1e-6)
    beta = mtt_safe / alpha_safe
    survival = 1.0 - gammainc(alpha_safe, t / beta)
    return np.asarray(cbf * survival, dtype=np.float64)


def make_cbf_mtt_grid(
    cbf_values: Sequence[float],
    mtt_values: Sequence[float],
    cbv_min_pct: float = 1.0,
    cbv_max_pct: float = 14.0,
) -> List[Tuple[float, float, float]]:
    """Build a CBF x MTT factorial grid of physiological tissue conditions.

    Returns every (CBF, MTT, CBV%) triple whose CBV = CBF*MTT/60 falls within
    [cbv_min_pct, cbv_max_pct], the physiological cerebral blood volume range.
    Combinations outside that range (e.g. high CBF with long MTT giving
    supra-physiological CBV) are dropped, so the grid holds only realisable
    tissues. Units: CBF in mL/100g/min, MTT in s, CBV in mL/100g (== % by mass).

    Parameters
    ----------
    cbf_values : Sequence[float]
        CBF grid values in mL/100g/min.
    mtt_values : Sequence[float]
        MTT grid values in seconds.
    cbv_min_pct : float
        Lower CBV bound in percent (inclusive).
    cbv_max_pct : float
        Upper CBV bound in percent (inclusive).

    Returns
    -------
    List[Tuple[float, float, float]]
        List of (CBF, MTT, CBV%) triples within the CBV bounds.
    """
    grid: List[Tuple[float, float, float]] = []
    for cbf in cbf_values:
        for mtt in mtt_values:
            cbv_pct = cbf * mtt / 60.0
            if cbv_min_pct <= cbv_pct <= cbv_max_pct:
                grid.append((float(cbf), float(mtt), float(cbv_pct)))
    return grid


def score_percent_repeatability(
    measurement_a: npt.NDArray[np.floating],
    measurement_b: npt.NDArray[np.floating],
) -> float:
    """Percent repeatability coefficient %RC = 2.77 * within-subject CV.

    Wraps ``pmriprep.utils.reproducibility.compute_within_subject_cv`` (the k=2
    test-retest within-subject CV, in percent) and scales by 2.77 (1.96*sqrt(2))
    to give the repeatability coefficient as a percentage, the standard
    quantitative-imaging metrology summary of reproducibility across two
    realisations.

    Parameters
    ----------
    measurement_a : npt.NDArray[np.floating]
        First realisation (scalar-like or 1D array of per-condition estimates).
    measurement_b : npt.NDArray[np.floating]
        Second realisation, same shape as ``measurement_a``.

    Returns
    -------
    float
        Percent repeatability coefficient.
    """
    from pmriprep.utils.reproducibility import compute_within_subject_cv

    return 2.77 * compute_within_subject_cv(measurement_a, measurement_b)


def make_dsc_tissue_concentration(
    cbf: float,
    mtt: float,
    aif: npt.NDArray[np.floating],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Generate tissue concentration curve from parameters and AIF.

    Convenience wrapper combining make_exponential_residue + dsc_forward_model.

    Parameters
    ----------
    cbf : float
        Cerebral blood flow (relative units).
    mtt : float
        Mean transit time in seconds.
    aif : npt.NDArray[np.floating]
        AIF concentration array (1D).
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Tissue concentration array (1D, same length as AIF).
    """
    residue = make_exponential_residue(cbf, mtt, len(aif), dt)
    return dsc_forward_model(aif, residue, dt)


# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture
def dce_aif_and_time() -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Parker AIF for DCE DRO tests.

    80 timepoints, dt=4s (0.0667 min), peak 5 mM.

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        Tuple of (aif_mM, time_min).
    """
    return generate_parker_aif_dro(
        n_timepoints=80,
        dt_min=4.0 / 60.0,
        peak_concentration=5.0,
    )


@pytest.fixture
def dsc_aif_and_time() -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Gamma-variate AIF for DSC DRO tests.

    60 timepoints, dt=1.5s, peak delta-R2*=40 s^-1.

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        Tuple of (aif_concentration, time_sec).
    """
    return generate_dsc_aif_dro(
        n_timepoints=60,
        dt_sec=1.5,
        baseline_frames=10,
        peak_delta_r2=40.0,
    )
