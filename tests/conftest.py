"""Pytest configuration and fixtures for PMRIPrep tests.

This module provides pytest fixtures for:
- Synthetic NIfTI image generation
- Synthetic time series generation (DSC, DCE curves)
- Synthetic AIF generation
- Test configuration file paths
- Temporary directories and file paths
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pytest


# =============================================================================
# Directory Fixtures
# =============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests (multi-seed noise sweeps with 50 seeds)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip slow and publication tests unless opted in."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Publication tests only run when explicitly selected via -m publication
    # (strict markers + addopts handles this via pytest's default behavior)


# =============================================================================
# Directory Fixtures
# =============================================================================


@pytest.fixture
def test_fixtures_dir() -> Path:
    """Return the path to the tests/fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_work_dir(tmp_path: Path) -> Path:
    """Return a temporary working directory."""
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


# =============================================================================
# Synthetic NIfTI Generation
# =============================================================================


def create_synthetic_3d_volume(
    shape: Tuple[int, int, int] = (32, 32, 16),
    voxel_sizes: Tuple[float, float, float] = (2.0, 2.0, 4.0),
    dtype: type = np.float32,
    fill_value: float = 0.0,
    add_noise: bool = False,
    noise_std: float = 10.0,
    random_seed: Optional[int] = 42,
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """Create a synthetic 3D volume with a specified shape and voxel sizes.

    Args:
        shape: Volume dimensions (x, y, z).
        voxel_sizes: Voxel dimensions in mm.
        dtype: Data type for the array.
        fill_value: Value to fill the volume with.
        add_noise: Whether to add Gaussian noise.
        noise_std: Standard deviation of the noise.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (data array, affine matrix).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create data array
    data = np.full(shape, fill_value, dtype=dtype)

    if add_noise:
        data = data + np.random.normal(0, noise_std, shape).astype(dtype)

    # Create affine matrix with voxel sizes
    affine = np.eye(4)
    affine[0, 0] = voxel_sizes[0]
    affine[1, 1] = voxel_sizes[1]
    affine[2, 2] = voxel_sizes[2]

    return data, affine


def create_synthetic_brain_volume(
    shape: Tuple[int, int, int] = (32, 32, 16),
    voxel_sizes: Tuple[float, float, float] = (2.0, 2.0, 4.0),
    brain_intensity: float = 1000.0,
    gm_wm_contrast: float = 0.3,
    add_noise: bool = True,
    noise_std: float = 20.0,
    random_seed: Optional[int] = 42,
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.float64], npt.NDArray[Any]]:
    """Create a synthetic brain-like volume with tissue contrast.

    Creates an ellipsoidal brain region with white matter core and
    gray matter shell, plus background.

    Args:
        shape: Volume dimensions.
        voxel_sizes: Voxel sizes in mm.
        brain_intensity: Base intensity for brain tissue.
        gm_wm_contrast: Gray/white matter contrast ratio.
        add_noise: Whether to add Gaussian noise.
        noise_std: Standard deviation of noise.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (intensity volume, affine matrix, brain mask).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    x, y, z = shape
    data = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.int_)

    # Create coordinate grids centered on volume
    cx, cy, cz = x // 2, y // 2, z // 2
    X, Y, Z = np.ogrid[:x, :y, :z]

    # Ellipsoid radii (brain shape)
    rx, ry, rz = x * 0.4, y * 0.4, z * 0.4

    # Brain ellipsoid equation
    brain_dist = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2

    # White matter core (inner ellipsoid)
    wm_dist = (
        ((X - cx) / (rx * 0.6)) ** 2
        + ((Y - cy) / (ry * 0.6)) ** 2
        + ((Z - cz) / (rz * 0.6)) ** 2
    )

    # Assign intensities
    # Background (outside brain)
    background_mask = brain_dist >= 1.0
    data[background_mask] = 0.0

    # Brain tissue
    brain_mask_bool = brain_dist < 1.0
    mask[brain_mask_bool] = 1

    # White matter (inner)
    wm_mask = (brain_dist < 1.0) & (wm_dist < 1.0)
    data[wm_mask] = brain_intensity * (1.0 + gm_wm_contrast)

    # Gray matter (outer shell)
    gm_mask = (brain_dist < 1.0) & (wm_dist >= 1.0)
    data[gm_mask] = brain_intensity * (1.0 - gm_wm_contrast)

    # Add noise
    if add_noise:
        noise = np.random.normal(0, noise_std, shape).astype(np.float32)
        data = data + noise * mask  # Only add noise within brain

    # Create affine
    affine = np.eye(4)
    affine[0, 0] = voxel_sizes[0]
    affine[1, 1] = voxel_sizes[1]
    affine[2, 2] = voxel_sizes[2]

    return data, affine, mask


def create_synthetic_brain_mask(
    shape: Tuple[int, int, int] = (32, 32, 16),
    voxel_sizes: Tuple[float, float, float] = (2.0, 2.0, 4.0),
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """Create a synthetic brain mask (ellipsoid).

    Args:
        shape: Volume dimensions.
        voxel_sizes: Voxel sizes in mm.

    Returns:
        Tuple of (mask array, affine matrix).
    """
    x, y, z = shape
    mask = np.zeros(shape, dtype=np.int_)

    cx, cy, cz = x // 2, y // 2, z // 2
    X, Y, Z = np.ogrid[:x, :y, :z]

    rx, ry, rz = x * 0.4, y * 0.4, z * 0.4
    dist = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2

    mask[dist < 1.0] = 1

    affine = np.eye(4)
    affine[0, 0] = voxel_sizes[0]
    affine[1, 1] = voxel_sizes[1]
    affine[2, 2] = voxel_sizes[2]

    return mask, affine


def create_synthetic_lesion_mask(
    brain_mask: npt.NDArray[Any],
    lesion_center: Optional[Tuple[int, int, int]] = None,
    lesion_radius_voxels: int = 3,
    random_seed: Optional[int] = 42,
) -> npt.NDArray[Any]:
    """Create a synthetic lesion mask within a brain mask.

    Args:
        brain_mask: 3D brain mask array.
        lesion_center: (x, y, z) center of lesion. If None, randomly placed.
        lesion_radius_voxels: Radius of spherical lesion in voxels.
        random_seed: Random seed for reproducibility.

    Returns:
        Binary lesion mask array.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    shape = brain_mask.shape
    lesion_mask = np.zeros(shape, dtype=np.int_)

    # Find brain voxels
    brain_coords = np.where(brain_mask > 0)
    if len(brain_coords[0]) == 0:
        return lesion_mask

    # Determine lesion center
    if lesion_center is None:
        # Random position within brain
        idx = np.random.randint(0, len(brain_coords[0]))
        cx = int(brain_coords[0][idx])
        cy = int(brain_coords[1][idx])
        cz = int(brain_coords[2][idx])
    else:
        cx, cy, cz = lesion_center

    # Create spherical lesion
    X, Y, Z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)

    lesion_mask[(dist <= lesion_radius_voxels) & (brain_mask > 0)] = 1

    return lesion_mask


def create_synthetic_4d_volume(
    shape: Tuple[int, int, int] = (32, 32, 16),
    n_timepoints: int = 60,
    voxel_sizes: Tuple[float, float, float] = (2.0, 2.0, 4.0),
    tr: float = 1.5,
    baseline_value: float = 1000.0,
    add_noise: bool = True,
    noise_std: float = 10.0,
    random_seed: Optional[int] = 42,
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """Create a synthetic 4D volume with constant baseline.

    Args:
        shape: Spatial dimensions (x, y, z).
        n_timepoints: Number of time points.
        voxel_sizes: Voxel sizes in mm.
        tr: Repetition time in seconds.
        baseline_value: Baseline signal value.
        add_noise: Whether to add Gaussian noise.
        noise_std: Standard deviation of noise.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (4D data array, affine matrix).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create 4D array
    full_shape = (*shape, n_timepoints)
    data = np.full(full_shape, baseline_value, dtype=np.float32)

    if add_noise:
        data = data + np.random.normal(0, noise_std, full_shape).astype(np.float32)

    # Create affine with TR in 4th dimension
    affine = np.eye(4)
    affine[0, 0] = voxel_sizes[0]
    affine[1, 1] = voxel_sizes[1]
    affine[2, 2] = voxel_sizes[2]

    return data, affine


# =============================================================================
# Synthetic DSC Time Series Generation
# =============================================================================


def generate_dsc_gamma_variate(
    n_timepoints: int = 60,
    tr: float = 1.5,
    baseline_frames: int = 10,
    peak_amplitude: float = 0.3,
    alpha: float = 2.5,
    beta: float = 3.0,
    arrival_time: Optional[float] = None,
    noise_std: float = 0.01,
    random_seed: Optional[int] = 42,
) -> npt.NDArray[Any]:
    """Generate a synthetic DSC signal using gamma-variate bolus model.

    The DSC signal follows: S(t) = S0 * exp(-TE * ΔR2*(t))
    where ΔR2* follows a gamma-variate curve representing the bolus.

    Args:
        n_timepoints: Number of time points.
        tr: Repetition time in seconds.
        baseline_frames: Number of pre-bolus baseline frames.
        peak_amplitude: Peak signal drop fraction (0-1).
        alpha: Gamma-variate shape parameter.
        beta: Gamma-variate scale parameter.
        arrival_time: Bolus arrival time in seconds. If None, uses baseline_frames * tr.
        noise_std: Standard deviation of Gaussian noise (fraction of baseline).
        random_seed: Random seed for reproducibility.

    Returns:
        1D array of DSC signal values.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Time array in seconds
    t = np.arange(n_timepoints) * tr

    # Default arrival time
    if arrival_time is None:
        arrival_time = baseline_frames * tr

    # Initialize signal at baseline (normalized to 1.0)
    signal = np.ones(n_timepoints, dtype=np.float64)

    # Gamma-variate for concentration (ΔR2*)
    # C(t) = A * (t - t0)^alpha * exp(-(t - t0) / beta) for t > t0
    mask = t > arrival_time
    if np.any(mask):
        dt = t[mask] - arrival_time
        # Normalize so peak is at specified amplitude
        peak_time = alpha * beta
        peak_factor = (peak_time**alpha) * np.exp(-alpha)
        if peak_factor > 1e-10:
            normalized_amplitude = peak_amplitude / peak_factor
        else:
            normalized_amplitude = peak_amplitude

        concentration = normalized_amplitude * (dt**alpha) * np.exp(-dt / beta)

        # Signal drop: S = S0 * exp(-concentration) simplified
        # For DSC, this represents the T2* signal decay
        signal[mask] = 1.0 - concentration

    # Ensure signal doesn't go negative
    signal = np.maximum(signal, 0.1)

    # Add noise
    if noise_std > 0:
        signal = signal + np.random.normal(0, noise_std, n_timepoints)

    # Scale to typical DSC signal range
    signal = signal * 1000.0

    return signal.astype(np.float32)


def generate_dsc_tissue_curve(
    n_timepoints: int = 60,
    tr: float = 1.5,
    baseline_frames: int = 10,
    cbf: float = 60.0,
    cbv: float = 4.0,
    mtt: float = 4.0,
    noise_std: float = 0.01,
    random_seed: Optional[int] = 42,
) -> npt.NDArray[Any]:
    """Generate a synthetic DSC tissue curve based on perfusion parameters.

    Creates a tissue curve using convolution of AIF with residue function,
    with parameters approximating real tissue perfusion.

    Args:
        n_timepoints: Number of time points.
        tr: Repetition time in seconds.
        baseline_frames: Number of baseline frames.
        cbf: Cerebral blood flow (mL/100g/min).
        cbv: Cerebral blood volume (mL/100g).
        mtt: Mean transit time in seconds.
        noise_std: Noise standard deviation.
        random_seed: Random seed.

    Returns:
        1D array of DSC tissue signal.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate AIF-like bolus
    aif = generate_dsc_gamma_variate(
        n_timepoints=n_timepoints,
        tr=tr,
        baseline_frames=baseline_frames,
        peak_amplitude=0.4,  # Stronger bolus for AIF
        alpha=2.0,
        beta=2.5,
        noise_std=0.0,  # Clean AIF for convolution
        random_seed=random_seed,
    )

    # Convert to concentration-like
    aif_conc = 1000.0 - aif  # Invert signal drop
    aif_conc = np.maximum(aif_conc, 0)

    # Create exponential residue function: R(t) = CBF * exp(-t/MTT)
    t = np.arange(n_timepoints) * tr
    residue = cbf * np.exp(-t / max(mtt, 0.1))

    # Convolve AIF with residue to get tissue curve
    tissue_conc = np.convolve(aif_conc, residue, mode="full")[:n_timepoints] * tr

    # Scale by CBV relationship
    tissue_conc = tissue_conc * (cbv / 100.0)

    # Convert back to signal
    signal = 1000.0 - tissue_conc

    # Add noise
    if noise_std > 0:
        signal = signal + np.random.normal(0, noise_std * 1000, n_timepoints)

    return signal.astype(np.float32)


def generate_dsc_4d_volume(
    shape: Tuple[int, int, int] = (32, 32, 16),
    n_timepoints: int = 60,
    tr: float = 1.5,
    baseline_frames: int = 10,
    brain_mask: Optional[npt.NDArray[Any]] = None,
    add_perfusion_variation: bool = True,
    noise_std: float = 0.01,
    random_seed: Optional[int] = 42,
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.float64], npt.NDArray[Any]]:
    """Generate a synthetic 4D DSC volume with brain-like perfusion.

    Args:
        shape: Spatial dimensions.
        n_timepoints: Number of time points.
        tr: Repetition time in seconds.
        baseline_frames: Number of baseline frames.
        brain_mask: Optional brain mask. If None, one is generated.
        add_perfusion_variation: Add spatial variation in perfusion.
        noise_std: Noise standard deviation.
        random_seed: Random seed.

    Returns:
        Tuple of (4D data, affine, brain_mask).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate brain mask if not provided
    if brain_mask is None:
        brain_mask_arr, affine = create_synthetic_brain_mask(shape)
    else:
        brain_mask_arr = brain_mask
        affine = np.eye(4)
        affine[0, 0] = 2.0
        affine[1, 1] = 2.0
        affine[2, 2] = 4.0

    # Initialize 4D volume with baseline signal
    data = np.full((*shape, n_timepoints), 1000.0, dtype=np.float32)

    # Generate DSC curves for brain voxels
    brain_coords = np.where(brain_mask_arr > 0)
    n_brain_voxels = len(brain_coords[0])

    for i in range(n_brain_voxels):
        x, y, z = brain_coords[0][i], brain_coords[1][i], brain_coords[2][i]

        # Vary perfusion parameters spatially
        if add_perfusion_variation:
            cbf_var = 60.0 + np.random.normal(0, 10)
            cbv_var = 4.0 + np.random.normal(0, 0.5)
            mtt_var = 4.0 + np.random.normal(0, 0.5)
        else:
            cbf_var, cbv_var, mtt_var = 60.0, 4.0, 4.0

        curve = generate_dsc_tissue_curve(
            n_timepoints=n_timepoints,
            tr=tr,
            baseline_frames=baseline_frames,
            cbf=max(cbf_var, 10),
            cbv=max(cbv_var, 0.5),
            mtt=max(mtt_var, 0.5),
            noise_std=noise_std,
            random_seed=None,  # Use current random state
        )
        data[x, y, z, :] = curve

    return data, affine, brain_mask_arr


# =============================================================================
# Synthetic DCE Time Series Generation
# =============================================================================


def generate_dce_tofts_curve(
    n_timepoints: int = 80,
    tr: float = 4.0,
    baseline_frames: int = 10,
    ktrans: float = 0.1,
    ve: float = 0.2,
    vp: float = 0.02,
    extended_model: bool = True,
    aif_peak: float = 5.0,
    noise_std: float = 0.05,
    random_seed: Optional[int] = 42,
) -> npt.NDArray[Any]:
    """Generate a synthetic DCE tissue curve using Tofts model.

    The Extended Tofts model:
        C_t(t) = vp * Cp(t) + Ktrans * conv(Cp, exp(-kep*t))

    where kep = Ktrans / ve

    Args:
        n_timepoints: Number of time points.
        tr: Repetition time in seconds.
        baseline_frames: Number of baseline frames.
        ktrans: Transfer constant (min^-1).
        ve: Extravascular extracellular volume fraction.
        vp: Plasma volume fraction.
        extended_model: Use Extended Tofts (with vp) vs Standard Tofts.
        aif_peak: Peak AIF concentration (mM).
        noise_std: Noise standard deviation (mM).
        random_seed: Random seed.

    Returns:
        1D array of tissue concentration (mM).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Time array in minutes (pharmacokinetic convention)
    t_sec = np.arange(n_timepoints) * tr
    t_min = t_sec / 60.0

    # Generate AIF using Parker population model (simplified)
    arrival_time_min = baseline_frames * tr / 60.0
    aif = generate_parker_aif(
        t_min=t_min,
        arrival_time_min=arrival_time_min,
        peak_concentration=aif_peak,
    )

    # Compute kep
    ve_safe = max(ve, 0.001)
    kep = ktrans / ve_safe

    # Create impulse response: exp(-kep * t) for t >= 0
    impulse = np.exp(-kep * t_min)

    # Convolve AIF with impulse response
    convolution = np.convolve(aif, impulse, mode="full")[:n_timepoints]
    convolution = convolution * (t_min[1] - t_min[0])  # dt for integration

    # Tissue concentration
    if extended_model:
        # Extended Tofts: C_t = vp * Cp + Ktrans * conv(Cp, exp(-kep*t))
        tissue_conc = vp * aif + ktrans * convolution
    else:
        # Standard Tofts: C_t = Ktrans * conv(Cp, exp(-kep*t))
        tissue_conc = ktrans * convolution

    # Ensure baseline is zero
    tissue_conc[:baseline_frames] = 0.0

    # Add noise
    if noise_std > 0:
        tissue_conc = tissue_conc + np.random.normal(0, noise_std, n_timepoints)
        tissue_conc = np.maximum(tissue_conc, 0.0)

    return tissue_conc.astype(np.float32)


def generate_parker_aif(
    t_min: npt.NDArray[Any],
    arrival_time_min: float = 0.0,
    peak_concentration: float = 5.0,
) -> npt.NDArray[Any]:
    """Generate a synthetic AIF using Parker population model (simplified).

    Based on Parker et al. (2006) population-averaged AIF.

    Args:
        t_min: Time array in minutes.
        arrival_time_min: Bolus arrival time in minutes.
        peak_concentration: Peak AIF concentration (mM).

    Returns:
        AIF concentration array (mM).
    """
    aif = np.zeros_like(t_min)

    mask = t_min > arrival_time_min
    if np.any(mask):
        dt = t_min[mask] - arrival_time_min

        # Simplified Parker model: sum of two Gaussians
        # First pass (early peak)
        sigma1 = 0.1  # minutes
        A1 = peak_concentration

        # Recirculation (broader, delayed)
        sigma2 = 0.3
        A2 = peak_concentration * 0.3
        t_recirc = 0.3  # delay for recirculation

        aif[mask] = A1 * np.exp(-dt**2 / (2 * sigma1**2)) + A2 * np.exp(
            -(dt - t_recirc) ** 2 / (2 * sigma2**2)
        )

    return aif


def generate_dce_4d_volume(
    shape: Tuple[int, int, int] = (32, 32, 16),
    n_timepoints: int = 80,
    tr: float = 4.0,
    baseline_frames: int = 10,
    brain_mask: Optional[npt.NDArray[Any]] = None,
    add_parameter_variation: bool = True,
    noise_std: float = 0.05,
    random_seed: Optional[int] = 42,
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.float64], npt.NDArray[Any]]:
    """Generate a synthetic 4D DCE volume with Tofts-model perfusion.

    Args:
        shape: Spatial dimensions.
        n_timepoints: Number of time points.
        tr: Repetition time in seconds.
        baseline_frames: Number of baseline frames.
        brain_mask: Optional brain mask.
        add_parameter_variation: Add spatial variation in PK parameters.
        noise_std: Noise standard deviation.
        random_seed: Random seed.

    Returns:
        Tuple of (4D concentration data, affine, brain_mask).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate brain mask if not provided
    if brain_mask is None:
        brain_mask_arr, affine = create_synthetic_brain_mask(shape)
    else:
        brain_mask_arr = brain_mask
        affine = np.eye(4)
        affine[0, 0] = 2.0
        affine[1, 1] = 2.0
        affine[2, 2] = 4.0

    # Initialize 4D volume
    data = np.zeros((*shape, n_timepoints), dtype=np.float32)

    # Generate DCE curves for brain voxels
    brain_coords = np.where(brain_mask_arr > 0)
    n_brain_voxels = len(brain_coords[0])

    for i in range(n_brain_voxels):
        x, y, z = brain_coords[0][i], brain_coords[1][i], brain_coords[2][i]

        # Vary PK parameters spatially
        if add_parameter_variation:
            ktrans_var = 0.1 + np.random.normal(0, 0.02)
            ve_var = 0.2 + np.random.normal(0, 0.03)
            vp_var = 0.02 + np.random.normal(0, 0.005)
        else:
            ktrans_var, ve_var, vp_var = 0.1, 0.2, 0.02

        curve = generate_dce_tofts_curve(
            n_timepoints=n_timepoints,
            tr=tr,
            baseline_frames=baseline_frames,
            ktrans=max(ktrans_var, 0.001),
            ve=max(ve_var, 0.01),
            vp=max(vp_var, 0.001),
            noise_std=noise_std,
            random_seed=None,
        )
        data[x, y, z, :] = curve

    return data, affine, brain_mask_arr


# =============================================================================
# Synthetic AIF Generation
# =============================================================================


def generate_synthetic_dsc_aif(
    n_timepoints: int = 60,
    tr: float = 1.5,
    baseline_frames: int = 10,
    peak_delta_r2: float = 40.0,
    alpha: float = 2.5,
    beta: float = 3.0,
    add_recirculation: bool = True,
    recirculation_fraction: float = 0.3,
    noise_std: float = 0.5,
    random_seed: Optional[int] = 42,
) -> npt.NDArray[Any]:
    """Generate a synthetic DSC AIF (ΔR2* concentration curve).

    Args:
        n_timepoints: Number of time points.
        tr: Repetition time in seconds.
        baseline_frames: Number of baseline frames.
        peak_delta_r2: Peak ΔR2* value (s^-1).
        alpha: Gamma-variate shape parameter.
        beta: Gamma-variate scale parameter.
        add_recirculation: Add recirculation component.
        recirculation_fraction: Amplitude of recirculation relative to first pass.
        noise_std: Noise standard deviation.
        random_seed: Random seed.

    Returns:
        1D array of AIF concentration (ΔR2* in s^-1).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Time array
    t = np.arange(n_timepoints) * tr
    arrival_time = baseline_frames * tr

    # Initialize concentration
    aif = np.zeros(n_timepoints, dtype=np.float64)

    # First pass (gamma-variate)
    mask = t > arrival_time
    if np.any(mask):
        dt = t[mask] - arrival_time

        # Normalize amplitude
        peak_time = alpha * beta
        peak_factor = (peak_time**alpha) * np.exp(-alpha)
        if peak_factor > 1e-10:
            amplitude = peak_delta_r2 / peak_factor
        else:
            amplitude = peak_delta_r2

        first_pass = amplitude * (dt**alpha) * np.exp(-dt / beta)
        aif[mask] = first_pass

        # Recirculation (delayed, broader peak)
        if add_recirculation:
            delay = 10.0 * tr  # Recirculation delay
            mask_recirc = t > (arrival_time + delay)
            if np.any(mask_recirc):
                dt_recirc = t[mask_recirc] - arrival_time - delay
                alpha_recirc = alpha * 0.8
                beta_recirc = beta * 1.5
                peak_factor_recirc = (
                    (alpha_recirc * beta_recirc) ** alpha_recirc
                ) * np.exp(-alpha_recirc)
                if peak_factor_recirc > 1e-10:
                    amplitude_recirc = (
                        peak_delta_r2 * recirculation_fraction / peak_factor_recirc
                    )
                else:
                    amplitude_recirc = peak_delta_r2 * recirculation_fraction
                recirculation = amplitude_recirc * (dt_recirc**alpha_recirc) * np.exp(
                    -dt_recirc / beta_recirc
                )
                aif[mask_recirc] = aif[mask_recirc] + recirculation

    # Add noise
    if noise_std > 0:
        aif = aif + np.random.normal(0, noise_std, n_timepoints)
        aif = np.maximum(aif, 0.0)

    return aif.astype(np.float32)


def generate_synthetic_dce_aif(
    n_timepoints: int = 80,
    tr: float = 4.0,
    baseline_frames: int = 10,
    peak_concentration: float = 5.0,
    dose_level: float = 1.0,
    random_seed: Optional[int] = 42,
) -> npt.NDArray[Any]:
    """Generate a synthetic DCE AIF (plasma concentration).

    Args:
        n_timepoints: Number of time points.
        tr: Repetition time in seconds.
        baseline_frames: Number of baseline frames.
        peak_concentration: Peak AIF concentration (mM).
        dose_level: Dose multiplier (0.5=half, 1.0=full, 2.0=double).
        random_seed: Random seed.

    Returns:
        1D array of AIF plasma concentration (mM).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Adjust peak based on dose level (dose IS the multiplier)
    adjusted_peak = peak_concentration * float(dose_level)

    # Time in minutes
    t_sec = np.arange(n_timepoints) * tr
    t_min = t_sec / 60.0
    arrival_time_min = baseline_frames * tr / 60.0

    # Generate Parker-style AIF
    aif = generate_parker_aif(
        t_min=t_min,
        arrival_time_min=arrival_time_min,
        peak_concentration=adjusted_peak,
    )

    return aif.astype(np.float32)


# =============================================================================
# NIfTI File I/O Helpers
# =============================================================================


def save_synthetic_nifti(
    data: npt.NDArray[Any],
    affine: npt.NDArray[np.float64],
    filepath: Union[str, Path],
    dtype: Optional[type] = None,
) -> Path:
    """Save synthetic data as a NIfTI file.

    Args:
        data: Array data to save.
        affine: Affine transformation matrix.
        filepath: Output file path.
        dtype: Optional data type for output.

    Returns:
        Path to saved file.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if dtype is not None:
        data = data.astype(dtype)

    img = nib.Nifti1Image(data, affine)
    nib.save(img, filepath)

    return filepath


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def synthetic_brain_volume() -> (
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64], npt.NDArray[Any]]
):
    """Fixture providing a synthetic brain volume."""
    return create_synthetic_brain_volume()


@pytest.fixture
def synthetic_brain_mask() -> Tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """Fixture providing a synthetic brain mask."""
    return create_synthetic_brain_mask()


@pytest.fixture
def synthetic_dsc_signal() -> npt.NDArray[Any]:
    """Fixture providing a synthetic DSC signal curve."""
    return generate_dsc_gamma_variate()


@pytest.fixture
def synthetic_dsc_4d() -> (
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64], npt.NDArray[Any]]
):
    """Fixture providing a synthetic 4D DSC volume."""
    return generate_dsc_4d_volume()


@pytest.fixture
def synthetic_dce_signal() -> npt.NDArray[Any]:
    """Fixture providing a synthetic DCE concentration curve."""
    return generate_dce_tofts_curve()


@pytest.fixture
def synthetic_dce_4d() -> (
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64], npt.NDArray[Any]]
):
    """Fixture providing a synthetic 4D DCE volume."""
    return generate_dce_4d_volume()


@pytest.fixture
def synthetic_dsc_aif() -> npt.NDArray[Any]:
    """Fixture providing a synthetic DSC AIF."""
    return generate_synthetic_dsc_aif()


@pytest.fixture
def synthetic_dce_aif() -> npt.NDArray[Any]:
    """Fixture providing a synthetic DCE AIF."""
    return generate_synthetic_dce_aif()


@pytest.fixture
def sample_dsc_config() -> Dict[str, Any]:
    """Fixture providing a sample DSC configuration dict."""
    return {
        "preload_present": False,
        "dose": 1.0,
        "method": "sSVD",
        "bayesian": "map",
        "aif_criteria": "strict",
        "n_aif_candidates": 10,
        "pre_bolus": 5,
        "use_gpu": False,
    }


@pytest.fixture
def sample_dce_config() -> Dict[str, Any]:
    """Fixture providing a sample DCE configuration dict."""
    return {
        "model": "extended_tofts",
        "bayesian": "none",
        "aif_method": "auto",
        "scale_aif": False,
        "b1_method": "auto",
        "no_t1": False,
        "tsnr_threshold": 0.0,
        "use_gpu": False,
    }
