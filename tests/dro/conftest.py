"""Shared DRO (Digital Reference Object) utilities and fixtures.

Provides synthetic AIF generators, noise functions, DSC forward models,
signal-domain helpers, and pytest fixtures for parametric recovery tests.
"""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from tests.conftest import generate_parker_aif


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass(frozen=True)
class DSCSignalConfig:
    """Frozen configuration for DSC signal generation.

    Follows the pattern from qiba/conftest.py (QIBATissueParams).

    Attributes:
        tr_sec: Repetition time in seconds.
        te_sec: Echo time in seconds.
        s0: Baseline signal intensity (arbitrary units).
        field_strength: Magnetic field strength in Tesla.
        label: Human-readable label for test IDs.
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

    Attributes:
        cbf: Cerebral blood flow (relative units, same as deconvolution output).
        mtt: Mean transit time in seconds.
        name: Tissue type name.
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

    Args:
        n_timepoints: Number of time points.
        dt_min: Time step in minutes.
        peak_concentration: Peak AIF concentration in mM.

    Returns:
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
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Generate a gamma-variate AIF in concentration domain for DSC DRO tests.

    Args:
        n_timepoints: Number of time points.
        dt_sec: Time step in seconds.
        baseline_frames: Number of pre-bolus baseline frames.
        peak_delta_r2: Peak delta-R2* value (s^-1).

    Returns:
        Tuple of (aif_concentration, time_sec).
    """
    time_sec = np.arange(n_timepoints, dtype=np.float64) * dt_sec
    arrival_time = baseline_frames * dt_sec

    aif = np.zeros(n_timepoints, dtype=np.float64)

    alpha = 2.5
    beta = 3.0

    mask = time_sec > arrival_time
    if np.any(mask):
        dt = time_sec[mask] - arrival_time
        peak_time = alpha * beta
        peak_factor = (peak_time**alpha) * np.exp(-alpha)
        if peak_factor > 1e-10:
            amplitude = peak_delta_r2 / peak_factor
        else:
            amplitude = peak_delta_r2
        aif[mask] = amplitude * (dt**alpha) * np.exp(-dt / beta)

    return aif, time_sec


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

    Args:
        concentration: ΔR2* array (1D time or 4D x,y,z,time). Must not
            include baseline frames — they are prepended here.
        te: Echo time in seconds.
        s0: Baseline signal intensity.
        baseline_frames: Number of baseline frames to prepend.

    Returns:
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
    phantoms, matching the estimator. For legacy single-voxel tests, AIF
    may be used as reference.

    Args:
        c_tissue_noleak: Clean tissue concentration (1D, no leakage).
        reference_curve: Reference curve for leakage (1D, same length).
            Typically mean brain C̄(t) or AIF.
        k2: K2 leakage rate parameter.
        dt: Time step in seconds.

    Returns:
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

    Args:
        c_tissue_noleak: Clean tissue concentration (1D, no leakage).
        reference_curve: Reference curve for leakage (1D, same length).
        k2: K2 leakage rate parameter.
        kep: EES efflux rate constant in s⁻¹.
        dt: Time step in seconds.

    Returns:
        Leaky tissue concentration (1D).

    References:
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

    Args:
        k2_enhancing: K2 value for enhancing voxels.
        n_brain_voxels: Number of non-enhancing brain voxels.
        n_enhancing_voxels: Number of enhancing (leaky) voxels.
        dt_sec: Time step in seconds.
        n_timepoints: Number of post-baseline timepoints.
        baseline_frames: Number of baseline frames.
        aif_peak: Peak AIF delta-R2* in s⁻¹.
        noise_frac: Fraction of peak for Gaussian noise (0=noiseless).
        seed: Random seed for reproducibility.

    Returns:
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

    Args:
        shape: Spatial dimensions (x, y, z).
        aif_coords: Set of (x, y, z) tuples for AIF voxels.
        aif: 1D AIF concentration (post-baseline only, n_post frames).
        tissue_curves: Dict mapping tissue name -> 1D concentration curve
            (post-baseline only).
        tissue_map: 3D integer array mapping each voxel to a tissue label.
        tissue_labels: Dict mapping tissue name -> integer label in tissue_map.
        baseline_frames: Number of zero-concentration baseline frames to prepend.

    Returns:
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

    Args:
        signal: Clean signal array.
        noise_std: Standard deviation of noise.
        seed: Random seed for reproducibility.

    Returns:
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

    Args:
        magnitude_signal: Clean magnitude signal array.
        noise_std: Standard deviation of noise in each channel.
        seed: Random seed for reproducibility.

    Returns:
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

    Args:
        s0: Baseline signal intensity.
        snr: Signal-to-noise ratio.

    Returns:
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

    C(t) = A @ R where A is the trapezoidal Toeplitz matrix of the AIF.
    Matched forward model for sSVD, Tikhonov, and CPI deconvolution.

    Args:
        aif: Arterial input function concentration array.
        residue: Residue function array (CBF * R(t)).
        dt: Time step in seconds.

    Returns:
        Tissue concentration array.
    """
    from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix

    A = build_toeplitz_matrix(aif, dt)
    return np.asarray(A @ residue, dtype=np.float64)


def dsc_forward_model_circulant(
    aif: npt.NDArray[np.floating],
    residue: npt.NDArray[np.floating],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Compute tissue concentration via block-circulant convolution.

    Matched forward model for oSVD and cSVD deconvolution (Wu 2003).
    Uses rectangular weighting, consistent with the circulant inverse.

    Args:
        aif: Arterial input function concentration array.
        residue: Residue function array (CBF * R(t)).
        dt: Time step in seconds.

    Returns:
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

    Args:
        cbf: Cerebral blood flow (relative units).
        mtt: Mean transit time in seconds.
        n_points: Number of time points.
        dt: Time step in seconds.

    Returns:
        Residue array R(t) = CBF * exp(-t/MTT).
    """
    t = np.arange(n_points, dtype=np.float64) * dt
    mtt_safe = max(mtt, 1e-6)
    return np.asarray(cbf * np.exp(-t / mtt_safe), dtype=np.float64)


def make_dsc_tissue_concentration(
    cbf: float,
    mtt: float,
    aif: npt.NDArray[np.floating],
    dt: float,
) -> npt.NDArray[np.floating]:
    """Generate tissue concentration curve from parameters and AIF.

    Convenience wrapper combining make_exponential_residue + dsc_forward_model.

    Args:
        cbf: Cerebral blood flow (relative units).
        mtt: Mean transit time in seconds.
        aif: AIF concentration array (1D).
        dt: Time step in seconds.

    Returns:
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

    Returns:
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

    Returns:
        Tuple of (aif_concentration, time_sec).
    """
    return generate_dsc_aif_dro(
        n_timepoints=60,
        dt_sec=1.5,
        baseline_frames=10,
        peak_delta_r2=40.0,
    )
