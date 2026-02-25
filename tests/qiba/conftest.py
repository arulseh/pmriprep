"""QIBA DSC-MRI DRO utilities, parameter tables, and fixtures.

Provides QIBA-representative AIF with recirculation (Patil & Johnson 2011),
CTH (capillary transit time heterogeneity) residue functions (Mouridsen 2006),
and published tissue parameter tables (Semmineh 2017, Boxerman 2020).

Phase 1: Code-generated synthetic data for CI.
Phase 2: External GBM-DSC-MRI-DRO (Phantomas/BNI) from qibadscdro.rsna.org.

References:
    Patil V, Johnson G. An improved model for describing the contrast
    bolus in perfusion MRI. Med Phys 2011;38:6380-6383.

    Mouridsen K, Friston K, Hjort N, et al. Bayesian estimation of
    cerebral perfusion using a physiological model of microvasculature.
    NeuroImage 2006;33:570-579.

    Semmineh NB, Stokes AM, Bell LC, Boxerman JL, Quarles CC.
    A population-based digital reference object (DRO) for optimizing
    dynamic susceptibility contrast (DSC)-MRI methods for clinical trials.
    Tomography 2017;3:41-49.

    Boxerman JL, Quarles CC, Hu LS, et al. Consensus recommendations
    for a dynamic susceptibility contrast MRI protocol for use in
    high-grade gliomas. Neuro-Oncology 2020;22:1262-1275.

    Ostergaard L, Weisskoff RM, Chesler DA, Rosen BR. High resolution
    measurement of cerebral blood flow using intravascular tracer bolus
    passages. Part I: Mathematical approach and statistical analysis.
    MRM 1996;36:715-725.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import gammainc

from tests.dro.conftest import (
    add_gaussian_noise,
    dsc_forward_model,
    make_exponential_residue,
)

# =============================================================================
# Path constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QIBA_DRO_ROOT = PROJECT_ROOT / "sourcedata" / "qiba" / "gbm_dsc_dro"

# Re-export for use by test_dsc_dro.py
__all__ = [
    "QIBATissueParams",
    "CTHConfig",
    "QIBANoiseLevel",
    "QIBA_TISSUE_PARAMS",
    "CTH_CONFIGS",
    "QIBA_NOISE_LEVELS",
    "generate_qiba_aif",
    "generate_cth_residue",
    "add_gaussian_noise",
    "dsc_forward_model",
    "make_exponential_residue",
    "QIBA_DRO_ROOT",
    "skip_no_qiba_dro",
    "QIBADROData",
    "load_qiba_dro",
    "deconvolve_dro_batch",
]


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class QIBATissueParams:
    """QIBA-representative tissue perfusion parameters.

    CBV is derived via central volume theorem: CBV = CBF * MTT / 60.
    Never independently specified.

    Attributes:
        name: Human-readable tissue label.
        cbf: Cerebral blood flow (mL/100g/min).
        mtt: Mean transit time (seconds).
        description: Source reference and notes.
    """

    name: str
    cbf: float
    mtt: float
    description: str

    @property
    def cbv(self) -> float:
        """CBV in mL/100g, derived from central volume theorem."""
        return self.cbf * self.mtt / 60.0

    @property
    def cbv_raw(self) -> float:
        """CBV in raw deconvolution units (CBF * MTT), for assertion."""
        return self.cbf * self.mtt


@dataclass(frozen=True)
class CTHConfig:
    """Capillary transit time heterogeneity configuration.

    Controls the shape of the gamma-variate residue function.
    alpha → ∞ gives exponential (no heterogeneity).
    alpha → 1 gives maximum heterogeneity.

    Attributes:
        alpha: Gamma shape parameter.
        rcth: Relative CTH = 1/sqrt(alpha) (dimensionless).
        description: Human-readable description.
    """

    alpha: float
    rcth: float
    description: str


@dataclass(frozen=True)
class QIBANoiseLevel:
    """Noise level for QIBA DRO tests.

    Attributes:
        fraction: Noise std as fraction of peak tissue concentration.
        label: Short label for test IDs.
        description: Human-readable description.
    """

    fraction: float
    label: str
    description: str


# =============================================================================
# Parameter tables (published values)
# =============================================================================

# CBV derived via central volume theorem: CBV = CBF * MTT / 60
QIBA_TISSUE_PARAMS: Dict[str, QIBATissueParams] = {
    "normal_gm": QIBATissueParams(
        name="Normal GM",
        cbf=60.0,
        mtt=4.6,
        description="Ostergaard 1996 (CBF=60, CBV=4.6 → MTT=4.6s)",
    ),
    "normal_wm": QIBATissueParams(
        name="Normal WM",
        cbf=20.0,
        mtt=4.3,
        description="Ostergaard 1996",
    ),
    "tumor": QIBATissueParams(
        name="Tumor",
        cbf=100.0,
        mtt=6.0,
        description="Semmineh 2017 / Lacerda & Law 2009",
    ),
    "tumor_high": QIBATissueParams(
        name="Tumor (hot-spot)",
        cbf=150.0,
        mtt=8.0,
        description="Semmineh 2017 hot-spot voxel",
    ),
    "tumor_low": QIBATissueParams(
        name="Tumor (low-end)",
        cbf=50.0,
        mtt=4.0,
        description="Semmineh 2017 low-end tumor",
    ),
    "peritumoral": QIBATissueParams(
        name="Peritumoral",
        cbf=35.0,
        mtt=5.0,
        description="Boxerman 2020",
    ),
    "contralateral_gm": QIBATissueParams(
        name="Contralateral GM",
        cbf=55.0,
        mtt=3.5,
        description="Boxerman 2020",
    ),
    "low_perfusion": QIBATissueParams(
        name="Low Perfusion",
        cbf=12.0,
        mtt=8.0,
        description="Ischemic penumbra",
    ),
}

# Gamma-variate residue shape parameters (Mouridsen 2006)
CTH_CONFIGS: Dict[str, CTHConfig] = {
    "low_cth": CTHConfig(
        alpha=9.0,
        rcth=0.33,
        description="Near-exponential (alpha=9, RCTH=0.33)",
    ),
    "moderate_cth": CTHConfig(
        alpha=4.0,
        rcth=0.5,
        description="Typical GM (alpha=4, RCTH=0.5)",
    ),
    "high_cth": CTHConfig(
        alpha=2.0,
        rcth=0.71,
        description="Pathological (alpha=2, RCTH=0.71)",
    ),
}

QIBA_NOISE_LEVELS: Dict[str, QIBANoiseLevel] = {
    "zero": QIBANoiseLevel(0.0, "zero", "No noise"),
    "low": QIBANoiseLevel(0.02, "low", "2% of peak"),
    "medium": QIBANoiseLevel(0.05, "medium", "5% of peak"),
    "high": QIBANoiseLevel(0.10, "high", "10% of peak"),
}


# =============================================================================
# AIF generator with recirculation
# =============================================================================


def generate_qiba_aif(
    n_timepoints: int = 80,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    peak_delta_r2: float = 40.0,
    kappa: float = 0.35,
    recirc_delay_sec: float = 15.0,
    recirc_sigma_sec: float = 5.0,
    alpha_fp: float = 3.0,
    beta_fp: float = 2.5,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Generate a QIBA-representative AIF with SCR recirculation.

    First-pass is a gamma-variate; recirculation is modeled as a
    delayed, broadened copy of the first pass (Patil & Johnson 2011).

    Recirculation: c_recirc = kappa * convolve(c_fp, G(t - delay, sigma))

    Args:
        n_timepoints: Number of time points (80 for QIBA).
        dt_sec: Temporal resolution in seconds.
        baseline_frames: Number of pre-bolus baseline frames.
        peak_delta_r2: Peak first-pass delta-R2* (s^-1).
        kappa: Recirculation fraction (Patil 2011: 0.34 +/- 0.12).
        recirc_delay_sec: Recirculation delay in seconds.
        recirc_sigma_sec: Gaussian dispersion sigma for recirculation (s).
        alpha_fp: First-pass gamma shape parameter.
        beta_fp: First-pass gamma scale parameter (seconds).

    Returns:
        Tuple of (aif_concentration, time_sec).
    """
    time_sec = np.arange(n_timepoints, dtype=np.float64) * dt_sec
    arrival_time = baseline_frames * dt_sec

    # First pass: gamma-variate (same math as generate_dsc_aif_dro)
    c_fp = np.zeros(n_timepoints, dtype=np.float64)
    mask = time_sec > arrival_time
    if np.any(mask):
        dt_arr = time_sec[mask] - arrival_time
        peak_time = alpha_fp * beta_fp
        peak_factor = (peak_time**alpha_fp) * np.exp(-alpha_fp)
        if peak_factor > 1e-10:
            amplitude = peak_delta_r2 / peak_factor
        else:
            amplitude = peak_delta_r2
        c_fp[mask] = amplitude * (dt_arr**alpha_fp) * np.exp(-dt_arr / beta_fp)

    # Recirculation: Gaussian kernel for dispersion
    if kappa > 0:
        # Build Gaussian kernel centered at recirc_delay
        kernel_length = n_timepoints
        kernel_time = np.arange(kernel_length, dtype=np.float64) * dt_sec
        gaussian_kernel = np.exp(
            -0.5 * ((kernel_time - recirc_delay_sec) / recirc_sigma_sec) ** 2
        )
        # Normalize kernel so convolution preserves area
        kernel_sum = np.sum(gaussian_kernel) * dt_sec
        if kernel_sum > 1e-10:
            gaussian_kernel /= kernel_sum

        # Convolve first pass with gaussian kernel, scale by kappa
        c_recirc_full = np.convolve(c_fp, gaussian_kernel * dt_sec, mode="full")
        c_recirc = kappa * c_recirc_full[:n_timepoints]

        aif = c_fp + c_recirc
    else:
        aif = c_fp.copy()

    return np.asarray(aif, dtype=np.float64), time_sec


# =============================================================================
# CTH (gamma-variate) residue function
# =============================================================================


def generate_cth_residue(
    cbf: float,
    alpha: float,
    beta: float,
    n_points: int,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Generate a gamma-variate residue function R(t) = CBF * [1 - gammainc(alpha, t/beta)].

    This represents the capillary transit time heterogeneity (CTH) model
    from Mouridsen et al. 2006. The residue function is the survival
    function of the gamma distribution of transit times.

    Args:
        cbf: Cerebral blood flow (mL/100g/min).
        alpha: Gamma shape parameter (controls heterogeneity).
        beta: Gamma scale parameter (seconds). Related to MTT: beta = MTT / alpha.
        n_points: Number of time points.
        dt: Time step in seconds.

    Returns:
        Residue array R(t) = CBF * [1 - gammainc(alpha, t / beta)].
    """
    t = np.arange(n_points, dtype=np.float64) * dt
    # scipy.special.gammainc is the regularized lower incomplete gamma function
    # R(t) = CBF * [1 - P(alpha, t/beta)] where P is regularized lower gamma
    beta_safe = max(beta, 1e-10)
    residue = cbf * (1.0 - gammainc(alpha, t / beta_safe))
    return np.asarray(residue, dtype=np.float64)


# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture
def qiba_aif_and_time() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
):
    """QIBA AIF with recirculation for DSC DRO tests.

    80 timepoints, dt=1.5s, kappa=0.35.

    Returns:
        Tuple of (aif_concentration, time_sec).
    """
    return generate_qiba_aif(
        n_timepoints=80,
        dt_sec=1.5,
        baseline_frames=10,
        peak_delta_r2=40.0,
        kappa=0.35,
    )


@pytest.fixture
def qiba_aif_no_recirc() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
):
    """QIBA AIF without recirculation (for comparison).

    Same first-pass parameters as qiba_aif_and_time but kappa=0.

    Returns:
        Tuple of (aif_concentration, time_sec).
    """
    return generate_qiba_aif(
        n_timepoints=80,
        dt_sec=1.5,
        baseline_frames=10,
        peak_delta_r2=40.0,
        kappa=0.0,
    )


# =============================================================================
# Phase 2: External GBM-DSC-MRI-DRO (Phantomas/BNI)
# =============================================================================

skip_no_qiba_dro = pytest.mark.skipif(
    not (QIBA_DRO_ROOT / "concentration.npz").exists(),
    reason="QIBA GBM-DSC-MRI-DRO data not found "
    f"(expected at {QIBA_DRO_ROOT / 'concentration.npz'})",
)


@dataclass
class QIBADROData:
    """Loaded QIBA GBM-DSC-MRI-DRO data.

    Attributes:
        delta_r2: DeltaR2* concentration, shape (X, Y, Z, T).
        baseline: Baseline signal S0, shape (X, Y, Z).
        aif: AIF deltaR2* curve, shape (T,).
        time_sec: Time array in seconds, shape (T,).
        dt: Temporal resolution in seconds.
        roi_masks: Dict of boolean ROI masks (tissue_high, tissue_low, background).
        metadata: Acquisition metadata dict.
    """

    delta_r2: npt.NDArray[np.floating]
    baseline: npt.NDArray[np.floating]
    aif: npt.NDArray[np.floating]
    time_sec: npt.NDArray[np.floating]
    dt: float
    roi_masks: Dict[str, npt.NDArray[np.bool_]]
    metadata: Dict[str, Any]


_qiba_dro_cache: Optional[QIBADROData] = None


def load_qiba_dro() -> QIBADROData:
    """Load converted QIBA DRO data with session-scoped caching.

    Returns:
        QIBADROData with all arrays and metadata.

    Raises:
        FileNotFoundError: If converted NPZ files don't exist.
    """
    global _qiba_dro_cache
    if _qiba_dro_cache is not None:
        return _qiba_dro_cache

    conc = np.load(QIBA_DRO_ROOT / "concentration.npz")
    aif_data = np.load(QIBA_DRO_ROOT / "aif.npz")
    mask_data = np.load(QIBA_DRO_ROOT / "roi_masks.npz")

    with open(QIBA_DRO_ROOT / "metadata.json") as f:
        metadata = json.load(f)

    _qiba_dro_cache = QIBADROData(
        delta_r2=conc["delta_r2"].astype(np.float64),
        baseline=conc["baseline"].astype(np.float64),
        aif=aif_data["aif_delta_r2"].astype(np.float64),
        time_sec=aif_data["time_sec"].astype(np.float64),
        dt=float(metadata["dt_sec"]),
        roi_masks={
            "tissue_high": mask_data["tissue_high"].astype(bool),
            "tissue_low": mask_data["tissue_low"].astype(bool),
            "background": mask_data["background"].astype(bool),
        },
        metadata=metadata,
    )
    return _qiba_dro_cache


def deconvolve_dro_batch(
    concentration_2d: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    dt: float,
    method: str,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Deconvolve DRO tissue curves using specified method.

    Args:
        concentration_2d: Tissue curves, shape (n_voxels, n_timepoints).
        aif: AIF curve, shape (n_timepoints,).
        dt: Time step in seconds.
        method: One of "ssvd", "osvd", "tikhonov".

    Returns:
        Tuple of (residues, cbf, cbv, mtt) arrays.
    """
    from pmriprep.utils.dsc_deconvolution import (
        svd_deconvolve_oscillation_batch,
        svd_deconvolve_standard_batch,
        tikhonov_deconvolve_batch,
    )

    if method == "ssvd":
        residues, cbf, cbv, mtt = svd_deconvolve_standard_batch(
            concentration_2d, aif, dt=dt
        )
    elif method == "osvd":
        residues, cbf, cbv, mtt, _, _ = svd_deconvolve_oscillation_batch(
            concentration_2d, aif, dt=dt
        )
    elif method == "tikhonov":
        residues, cbf, cbv, mtt, _ = tikhonov_deconvolve_batch(
            concentration_2d, aif, dt=dt
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return residues, cbf, cbv, mtt
