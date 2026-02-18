"""Shared DRO (Digital Reference Object) utilities and fixtures.

Provides synthetic AIF generators, noise functions, DSC forward models,
and pytest fixtures for parametric recovery tests.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest

from tests.conftest import generate_parker_aif


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
